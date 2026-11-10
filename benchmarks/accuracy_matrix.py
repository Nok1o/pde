"""Reproduce bounded accuracy trials and repeated-control query costs."""

import argparse
import json
from pathlib import Path

import numpy as np

from pdelab.control import prepare_control
from pdelab.experiment import Config
from pdelab.problem import make_problem, select_points
from pdelab.research import StudySettings, accuracy_study, export_study
from pdelab.solvers import solve, warmup


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-walks", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seconds", type=float, default=3)
    args = parser.parse_args()
    warmup()
    reuse_rows = []
    for n in (16, 32, 64):
        for kind in ("hot_top", "polynomial", "sine", "affine"):
            for mode in ("point", "offcenter", "patch", "field"):
                config = Config(
                    problem=kind,
                    n=n,
                    mode="point" if mode == "offcenter" else mode,
                    point=(n // 4, n // 3) if mode == "offcenter" else None,
                )
                study = accuracy_study(
                    StudySettings(config, args.repeats, args.max_walks, args.seconds)
                )
                export_study(study, args.out / f"{kind}-{n}-{mode}")
                problem = make_problem(kind, n)
                points = select_points(n, config.mode, config.point)
                control = prepare_control(problem)
                results = [
                    solve(
                        problem,
                        points,
                        "mc_control",
                        control=control,
                        walks=64,
                        seed=9000 + i,
                    )
                    for i in range(args.repeats)
                ]
                query_seconds = float(np.mean([r.elapsed for r in results]))
                reuse_rows.append(
                    dict(
                        n=n,
                        problem=kind,
                        mode=mode,
                        queries=args.repeats,
                        preparation_seconds=control.preparation_seconds,
                        query_seconds=query_seconds,
                        first_seconds=control.preparation_seconds + query_seconds,
                        amortized_seconds=control.preparation_seconds / args.repeats
                        + query_seconds,
                    )
                )
                print(f"{kind} n={n} {mode}", flush=True)
    (args.out / "repeated_queries.json").write_text(
        json.dumps(reuse_rows), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
