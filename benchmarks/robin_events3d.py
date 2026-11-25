"""Short stationary MC event profile for #32, independent of budget selection."""

import argparse
import hashlib
import inspect
import platform
import subprocess
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import numba
import numpy as np
from numba import njit

from pdelab.boundary3d import FACES, BoundaryOperator3D, make_boundary_problem3d
from pdelab.boundary_solvers3d import _reflected_step, _walks

from .grouped_mc3d import write_json

N = 12
WALKS = 512
BATCH_SIZE = 32
ROOT_SEED = 321125
CASES = (("x0",), ("x0", "y0", "z0"), FACES)


def inputs(faces):
    kinds = {face: "robin" if face in faces else "dirichlet" for face in FACES}
    p = make_boundary_problem3d("polynomial", N, kinds)
    points = np.array([[N // 2] * 3], dtype=np.int64)
    return replace(p, exact=None), points, float(p.exact[tuple(points[0])])


def seed_for(*indices):
    return int(np.random.SeedSequence([ROOT_SEED, *indices]).generate_state(1)[0])


@njit
def _record(fixed, killing, start, count, seed, positions):
    """Replay the baseline RNG and record the origin of each reward stage."""
    np.random.seed(seed)
    n = fixed.shape[0] - 1
    events = 0
    reflected = 0
    killed = 0
    for _ in range(count):
        z, y, x = start
        while not fixed[z, y, x]:
            if events == len(positions):
                raise RuntimeError(
                    "Diagnostic replay exceeded measured baseline stages"
                )
            positions[events, 0] = z
            positions[events, 1] = y
            positions[events, 2] = x
            events += 1
            if killing[z, y, x] > 0 and np.random.random() < killing[z, y, x]:
                killed += 1
                break
            direction = np.random.randint(6)
            coordinate = (z, y, x)[direction // 2]
            if coordinate == (n if direction % 2 == 0 else 0):
                reflected += 1
            z, y, x = _reflected_step(z, y, x, n, direction)
    return events, reflected, killed


@njit
def _replay(positions, fixed, boundary, reward, killing, loops, seed):
    """Measure per-origin work, consuming RNG and checking the proposed endpoint."""
    np.random.seed(seed)
    n = fixed.shape[0] - 1
    checksum = 0.0
    for _ in range(loops):
        for i in range(len(positions)):
            z, y, x = positions[i]
            checksum += reward[z, y, x]
            if killing[z, y, x] > 0 and np.random.random() < killing[z, y, x]:
                continue
            z, y, x = _reflected_step(z, y, x, n, np.random.randint(6))
            if fixed[z, y, x]:
                checksum += boundary[z, y, x]
    return checksum


def replay_cost(positions, arrays, seed):
    loops = max(1, (250_000 + len(positions) - 1) // len(positions))
    _replay(positions[:1], *arrays, 1, seed)
    start = perf_counter()
    _replay(positions, *arrays, loops, seed)
    return (perf_counter() - start) / (loops * len(positions))


def profile(case_id, faces, repeat):
    p, points, _ = inputs(faces)
    seed = seed_for(0, case_id, repeat)
    start = perf_counter()
    op = BoundaryOperator3D(p)
    op.check_compatibility()
    op_seconds = perf_counter() - start
    start = perf_counter()
    probabilities, reward, killing = op.mc_coefficients()
    coefficients_seconds = perf_counter() - start
    arrays = (op.fixed, op.boundary, reward, killing)
    # Warm this signature before timing. The next call resets the RNG.
    _walks(*arrays, points, 2, seed, 1000 * N**2)
    start = perf_counter()
    _, _, steps, _ = _walks(*arrays, points, WALKS, seed, 1000 * N**2)
    kernel_seconds = perf_counter() - start
    positions = np.empty((steps, 3), dtype=np.int64)
    count, reflected, killed = _record(
        op.fixed, killing, points[0], WALKS, seed, positions
    )
    if count != steps:
        raise RuntimeError("Diagnostic replay differs from published path length")
    codimension = np.sum((positions == 0) | (positions == N), axis=1)
    inner, border = positions[codimension == 0], positions[codimension > 0]
    order = (("inner", inner), ("border", border))
    costs = {
        key: replay_cost(nodes, arrays, seed)
        for key, nodes in (order if repeat % 2 == 0 else reversed(order))
    }
    border_cost = len(border) * costs["border"]
    return dict(
        robin_faces=faces,
        repeat=repeat + 1,
        seed=seed,
        stages=steps,
        face_stages=int(np.sum(codimension == 1)),
        edge_stages=int(np.sum(codimension == 2)),
        corner_stages=int(np.sum(codimension == 3)),
        reflected_outward_proposals=reflected,
        robin_kills=killed,
        operator_seconds=op_seconds,
        coefficients_seconds=coefficients_seconds,
        unused_probabilities_bytes=probabilities.nbytes,
        kernel_seconds=kernel_seconds,
        replay_inner_ns=costs["inner"] * 1e9,
        replay_border_ns=costs["border"] * 1e9,
        boundary_stage_fraction=len(border) / steps,
        replay_boundary_time_fraction=border_cost
        / (border_cost + len(inner) * costs["inner"]),
    )


def metadata():
    return dict(
        git_head=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        file_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        baseline_function_sha256={
            f.__name__: hashlib.sha256(
                inspect.getsource(getattr(f, "py_func", f)).encode()
            ).hexdigest()
            for f in (
                _walks,
                _reflected_step,
                BoundaryOperator3D.mc_coefficients,
                BoundaryOperator3D.__init__,
            )
        },
        python=platform.python_version(),
        platform=platform.platform(),
        numpy=np.__version__,
        numba=numba.__version__,
        n=N,
        walks_per_point=WALKS,
        point_zyx=[N // 2] * 3,
        repeats=3,
        master_seed=ROOT_SEED,
        beta=1.0,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("Choose a new output file")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    data = metadata()
    # Exclude first operator/coefficient allocations from the short repeated profile.
    op = BoundaryOperator3D(inputs(CASES[0])[0])
    op.mc_coefficients()
    data["trials"] = [
        profile(case_id, faces, repeat)
        for case_id, faces in enumerate(CASES)
        for repeat in range(3)
    ]
    write_json(args.out, data)


if __name__ == "__main__":
    main()
