"""Fixed-budget profile of the published transient MC, not time-to-accuracy.

Each case/kernel gets a fresh process. Numerical reference and expected visit
counts are evaluated after all timing and RSS samples, without instrumenting
or reimplementing the production path kernels.
"""

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import numba
import numpy as np
import scipy
from scipy.sparse.linalg import spsolve

from pdelab.boundary3d import FACES
from pdelab.heat3d_mc import solve_frozen_mc3d, warmup_mc3d
from pdelab.heat3d_operator import freeze_heat3d
from pdelab.heat3d_problem import make_heat_problem3d

from .grouped_mc3d import peak_rss_bytes, source_state, write_csv, write_json

CASES = ((), ("x0",), ("x0", "y0", "z0"), FACES)
PLAN = dict(
    n=8,
    alpha=0.5,
    beta=1.0,
    case="polynomial",
    steps=[0.002, 0.02],
    walks=512,
    batch_size=128,
    repeats=5,
    root_seed=320026,
    max_steps=1_000_000,
    max_walks=10_000_000,
    max_transitions=1_000_000_000,
)


def snapshot(faces, dt, n=PLAN["n"]):
    kinds = {face: 3 if face in faces else 1 for face in FACES}
    problem = replace(
        make_heat_problem3d(
            PLAN["case"], n, PLAN["alpha"], kinds=kinds, beta=PLAN["beta"]
        ),
        exact=None,
    )
    previous = problem.initial_values()
    x, y, z = problem.coordinates()
    iterate = previous + 0.1 * (x + 2 * y + 3 * z)
    operator = freeze_heat3d(problem, previous, iterate, dt, dt)
    points = np.indices(problem.shape, dtype=np.int64).reshape(3, -1).T.copy()
    return operator, points


def expected_visits(operator, points, walks):
    """Expected visit totals, including the final reward/absorption stage.

    With initial path counts s, v = s + P.T v. Since I-P = D^-1 A,
    v = D A^-T s. Known query nodes start no path. This is an analysis-only
    sparse solve, never an input to MC and never inside its timer or RSS sample.
    """
    starts = np.zeros(operator.shape)
    free_points = points[~operator.fixed[tuple(points.T)]]
    np.add.at(starts, tuple(free_points.T), walks)
    matrix, _ = operator.assemble()
    visits = operator.diagonal.ravel() * spsolve(matrix.T.tocsc(), starts.ravel())
    if not np.isfinite(visits).all() or np.any(visits < -1e-10):
        raise RuntimeError("Invalid expected visit counts")
    coordinates = np.indices(operator.shape)
    codimension = np.sum((coordinates == 0) | (coordinates == operator.n), axis=0)
    totals = [float(visits[codimension.ravel() == k].sum()) for k in range(4)]
    paths = int(starts.sum())
    total = sum(totals)
    return dict(
        expected_interior_visits=totals[0],
        expected_face_visits=totals[1],
        expected_edge_visits=totals[2],
        expected_corner_visits=totals[3],
        expected_boundary_visit_fraction=sum(totals[1:]) / total if total else 0.0,
        expected_transitions=total - paths,
        expected_moves_per_path=(total - paths) / paths if paths else 0.0,
        paths=paths,
    )


def measure(case_index, step_index, grouped):
    dt = PLAN["steps"][step_index]
    faces = CASES[case_index]
    batch_size = PLAN["batch_size"] if grouped else None
    warmup_mc3d(batch_size=batch_size)
    # Warm allocation/data/validation paths too, without using timed sample seeds.
    operator, points = snapshot(faces, dt)
    solve_frozen_mc3d(operator, points, walks=2, seed=0, batch_size=batch_size)
    rows, values = [], []
    for repeat in range(PLAN["repeats"]):
        # Paired root seeds across kernels; grouping changes the draw order.
        seed = int(
            np.random.SeedSequence(
                [PLAN["root_seed"], case_index, step_index, repeat]
            ).generate_state(1)[0]
        )
        started = perf_counter()
        operator, points = snapshot(faces, dt)
        setup_seconds = perf_counter() - started
        result = solve_frozen_mc3d(
            operator,
            points,
            walks=PLAN["walks"],
            seed=seed,
            batch_size=batch_size,
            max_steps=PLAN["max_steps"],
            max_walks=PLAN["max_walks"],
            max_transitions=PLAN["max_transitions"],
        )
        elapsed = perf_counter() - started
        # RSS is the process high-water mark, including imports/JIT/warmup and
        # earlier repeats, but excluding the subsequent sparse diagnostics.
        rss = peak_rss_bytes()
        values.append(result.values)
        rows.append(
            dict(
                case_index=case_index,
                kind3_faces=list(faces),
                dt=dt,
                method=result.method,
                batch_size=batch_size,
                repeat=repeat + 1,
                seed=seed,
                status=result.status,
                reason=result.reason,
                setup_seconds=setup_seconds,
                coefficients_seconds=result.preparation_seconds,
                sampling_and_result_seconds=result.seconds - result.preparation_seconds,
                full_seconds=elapsed,
                walks=result.walks,
                completed_walks=(
                    int(result.completed_paths.sum())
                    if result.completed_paths is not None
                    else 0
                ),
                transitions=result.transitions,
                workspace_bytes=result.workspace_bytes,
                peak_rss_bytes=rss,
            )
        )
    # One diagnostic reference to this frozen system, not the continuous u(t).
    matrix, rhs = operator.assemble()
    target = spsolve(matrix, rhs)
    free = ~operator.fixed.ravel()
    target_norm = np.linalg.norm(target[free])
    for row, estimate in zip(rows, values):
        row["relative_l2_error_pct"] = (
            float(100 * np.linalg.norm(estimate[free] - target[free]) / target_norm)
            if estimate is not None and target_norm > 0
            else None
        )
    return dict(
        trials=rows,
        expected=expected_visits(operator, points, PLAN["walks"]),
    )


def summarize(packet):
    rows = packet["trials"]
    first = rows[0]
    complete = all(row["status"] == "completed" for row in rows)
    result = {
        key: first[key]
        for key in ("case_index", "kind3_faces", "dt", "method", "batch_size")
    }
    result.update(
        completed=sum(row["status"] == "completed" for row in rows),
        repeats=len(rows),
        **packet["expected"],
    )
    for key in (
        "full_seconds",
        "setup_seconds",
        "coefficients_seconds",
        "sampling_and_result_seconds",
        "transitions",
    ):
        result[key + "_mean"] = float(np.mean([row[key] for row in rows]))
        result[key + "_std"] = float(np.std([row[key] for row in rows], ddof=1))
    result["relative_l2_error_pct_mean"] = (
        float(np.mean([row["relative_l2_error_pct"] for row in rows]))
        if complete and all(row["relative_l2_error_pct"] is not None for row in rows)
        else None
    )
    result["workspace_bytes"] = max(row["workspace_bytes"] for row in rows)
    result["peak_rss_bytes"] = max(row["peak_rss_bytes"] for row in rows)
    return result


def provenance():
    return dict(
        **source_state(),
        runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--worker", nargs=3, type=int, metavar=("CASE", "STEP", "GROUPED")
    )
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("Choose a new output path")
    if args.worker is not None:
        write_json(args.out, measure(*args.worker))
        return
    args.out.mkdir(parents=True)
    before = provenance()
    environment = dict(os.environ)
    for key in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMBA_NUM_THREADS",
    ):
        environment[key] = "1"
    manifest = dict(
        schema="heat3d_boundary_profile_v1",
        plan=PLAN,
        cases=CASES,
        source_before=before,
        python=platform.python_version(),
        platform=platform.platform(),
        numpy=np.__version__,
        scipy=scipy.__version__,
        numba=numba.__version__,
        worker_threads=1,
        timer="input fields + operator + published frozen MC; warmed, no reference",
        error="mean of run relative L2 errors on free nodes against A^-1 b",
        visits="expected, not observed or timed: v = D A^-T s",
        rss="fresh worker high-water, imports/JIT included, sparse diagnostics excluded",
    )
    write_json(args.out / "manifest.json", manifest)
    summary = []
    for case_index in range(len(CASES)):
        for step_index in range(len(PLAN["steps"])):
            # Alternate method order. No concurrent workers or reference solving.
            order = (0, 1) if (case_index + step_index) % 2 == 0 else (1, 0)
            for grouped in order:
                output = (
                    args.out / f"case{case_index}-step{step_index}-group{grouped}.json"
                )
                subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "benchmarks.heat3d_boundary_profile",
                        "--worker",
                        str(case_index),
                        str(step_index),
                        str(grouped),
                        "--out",
                        str(output),
                    ],
                    env=environment,
                    check=True,
                )
                packet = json.loads(output.read_text())
                summary.append(summarize(packet))
                print(output.name, flush=True)
    manifest["source_after"] = provenance()
    manifest["sources_unchanged"] = before == manifest["source_after"]
    write_json(args.out / "manifest.json", manifest)
    if not manifest["sources_unchanged"]:
        raise RuntimeError("Sources changed during measurements")
    write_csv(args.out / "summary.csv", summary)


if __name__ == "__main__":
    main()
