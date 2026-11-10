"""Run with python -m benchmarks.profile_kernels --out runs/profile.json."""

import argparse
import json
import platform
import subprocess
import tracemalloc
from pathlib import Path

import numba
import numpy as np

from pdelab.problem import make_problem, select_points
from pdelab.solvers import METHODS, solve, warmup


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    warmup()
    rows = []
    for n in (16, 32, 64):
        for kind in ("hot_top", "polynomial", "sine", "affine"):
            problem = make_problem(kind, n)
            for mode in ("point", "offcenter", "patch", "field"):
                points = select_points(
                    n,
                    "point" if mode == "offcenter" else mode,
                    (n // 4, n // 3) if mode == "offcenter" else None,
                )
                for method in METHODS:
                    results = [
                        solve(
                            problem,
                            points,
                            method,
                            walks=32,
                            batches=4,
                            seed=4200 + repeat,
                            record=False,
                        )
                        for repeat in range(args.repeats)
                    ]
                    tracemalloc.start()
                    measured = solve(
                        problem,
                        points,
                        method,
                        walks=32,
                        batches=4,
                        seed=4200,
                        record=False,
                    )
                    _, peak = tracemalloc.get_traced_memory()
                    tracemalloc.stop()
                    rows.append(
                        dict(
                            n=n,
                            problem=kind,
                            mode=mode,
                            method=method,
                            seconds=[r.elapsed for r in results],
                            python_peak_bytes=peak,
                            steps=measured.steps,
                            walks=measured.walks,
                            iterations=measured.iterations,
                            values=measured.values[tuple(points.T)].tolist(),
                        )
                    )
        print(f"n={n} complete", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            dict(
                commit=subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], text=True
                ).strip(),
                python=platform.python_version(),
                numpy=np.__version__,
                numba=numba.__version__,
                repeats=args.repeats,
                walks_per_start=32,
                seeds=[4200 + i for i in range(args.repeats)],
                memory_scope="tracemalloc peak in separate call; native Numba allocations may be invisible",
                rows=rows,
            )
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
