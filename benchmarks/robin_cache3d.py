"""Small fixed-N stationary comparison. Each method/case gets a fresh process."""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from time import perf_counter

from pdelab.boundary_solvers3d import solve_boundary3d
from pdelab.grouped_solver3d import GroupedWalkFailure
from pdelab.measurement_protocol import query_error

from .grouped_mc3d import peak_rss_bytes, source_state, write_json
from .robin_events3d import BATCH_SIZE, CASES, WALKS, N, inputs, metadata, seed_for

METHODS = ("mc", "mc_cached", "mc_grouped", "mc_grouped_cached")


def worker(case_id, method):
    faces = CASES[case_id]
    p, points, reference = inputs(faces)
    prewarm_rss = peak_rss_bytes()
    start = perf_counter()
    solve_boundary3d(p, points, method, walks=2, batch_size=BATCH_SIZE)
    warmup_seconds = perf_counter() - start
    warm_rss = peak_rss_bytes()
    trials = []
    for repeat in range(3):
        seed = seed_for(1, case_id, METHODS.index(method), repeat)
        row = dict(repeat=repeat + 1, seed=seed, value=None, standard_error=None)
        start = perf_counter()
        try:
            result = solve_boundary3d(
                p, points, method, walks=WALKS, seed=seed, batch_size=BATCH_SIZE
            )
            wall_seconds = perf_counter() - start
            work = result.extras
            row.update(
                status="completed",
                reason="",
                seconds=wall_seconds,
                solver_seconds=result.elapsed,
                walks=result.walks,
                started_walks=work.get("started_walks", result.walks),
                transitions=result.steps,
                value=float(result.values[tuple(points[0])]),
                standard_error=float(result.stderr[tuple(points[0])]),
            )
            del result
        except GroupedWalkFailure as exc:
            wall_seconds = perf_counter() - start
            work = exc.diagnostics
            row.update(
                status=exc.status,
                reason=str(exc),
                seconds=wall_seconds,
                solver_seconds=exc.seconds,
                walks=work["walks"],
                started_walks=work["started_walks"],
                transitions=work["transitions"],
            )
        except (RuntimeError, ValueError, MemoryError) as exc:
            work = {}
            row.update(
                status="solver_failed",
                reason=str(exc),
                seconds=perf_counter() - start,
                solver_seconds=None,
                walks=None,
                started_walks=None,
                transitions=None,
            )
        for key in (
            "preparation_seconds",
            "path_seconds",
            "workspace_bytes",
            "reflection_table_bytes",
        ):
            row[key] = work.get(key)
        trials.append(row)
        del work
    complete = all(row["status"] == "completed" for row in trials)
    values = [[row["value"]] for row in trials]
    absolute, relative = query_error(values, [reference]) if complete else (None, None)
    return dict(
        method=method,
        robin_faces=faces,
        reference=reference,
        trials=trials,
        requested_repeats=len(trials),
        completed_repeats=sum(r["status"] == "completed" for r in trials),
        mean_absolute_error=absolute,
        mean_relative_error_pct=relative,
        peak_rss_bytes=peak_rss_bytes(),
        prewarm_rss_bytes=prewarm_rss,
        warmed_rss_bytes=warm_rss,
        warmup_seconds=warmup_seconds,
    )


def benchmark_hashes():
    return {
        name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
        for name in ("robin_cache3d.py", "robin_events3d.py")
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--out", type=Path)
    mode.add_argument(
        "--worker", nargs=2, metavar=("CASE", "METHOD"), help=argparse.SUPPRESS
    )
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(worker(int(args.worker[0]), args.worker[1]), allow_nan=False))
        return
    if args.out.exists():
        raise ValueError("Choose a new output file")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    before = source_state()
    scripts = benchmark_hashes()
    results = []
    for case_id in range(len(CASES)):
        for method in METHODS:
            command = [
                sys.executable,
                "-m",
                "benchmarks.robin_cache3d",
                "--worker",
                str(case_id),
                method,
            ]
            results.append(json.loads(subprocess.check_output(command, text=True)))
    after = source_state()
    if (
        before["combined_sha256"] != after["combined_sha256"]
        or before["commit"] != after["commit"]
        or scripts != benchmark_hashes()
    ):
        raise RuntimeError("Measured source changed during the comparison")
    write_json(
        args.out,
        dict(
            baseline_metadata=metadata(),
            source=before,
            benchmark_sha256=scripts,
            batch_size=BATCH_SIZE,
            max_steps=1000 * N**2,
            results=results,
        ),
    )


if __name__ == "__main__":
    main()
