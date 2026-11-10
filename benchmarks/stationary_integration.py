"""Run with python -m benchmarks.stationary_integration; no plotting in timings."""

import csv
import hashlib
import json
import platform
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import numpy as np

from pdelab.boundary_experiment import BoundaryConfig, run_boundary_experiment
from pdelab.boundary_experiment3d import BoundaryConfig3D, run_boundary_experiment3d
from pdelab.experiment import Config, run_experiment
from pdelab.experiment3d import Config3D, run_experiment3d


def main():
    rows = []
    for n in (8, 16):
        cases = [
            ("dirichlet2d", Config(n=n, problem="sine"), run_experiment),
            ("dirichlet3d", Config3D(n=n, problem="sine"), run_experiment3d),
            ("mixed2d", BoundaryConfig(n=n, problem="smooth"), run_boundary_experiment),
            (
                "mixed3d",
                BoundaryConfig3D(n=n, problem="smooth"),
                run_boundary_experiment3d,
            ),
        ]
        for case_index, (case, config, run) in enumerate(cases):
            for budget in (256, 1024):
                for repeat in range(4):
                    seed = int(
                        np.random.SeedSequence(
                            [20261110, n, case_index, repeat]
                        ).generate_state(1)[0]
                    )
                    experiment = run(replace(config, walks=budget, seed=seed))
                    for metric in experiment.metrics:
                        rows.append(
                            dict(
                                case=case,
                                n=n,
                                budget=budget,
                                repeat=repeat,
                                seed=seed,
                                **metric,
                            )
                        )
    out = Path("benchmarks/results/stationary-integration")
    out.mkdir(parents=True, exist_ok=True)
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with (out / "trials.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    groups = defaultdict(list)
    for row in rows:
        groups[(row["case"], row["n"], row["budget"], row["method"])].append(row)
    summary = [
        dict(
            case=case,
            n=n,
            budget=budget,
            method=method,
            seconds=float(np.mean([r["seconds"] for r in group])),
            relative_rms_pct=float(
                np.sqrt(np.mean([r["relative_error_pct"] ** 2 for r in group]))
            ),
            runs=len(group),
        )
        for (case, n, budget, method), group in sorted(groups.items())
    ]
    with (out / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, list(summary[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(summary)
    (out / "settings.json").write_text(
        json.dumps(
            dict(
                code_sha256={
                    str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in sorted(Path("pdelab").glob("*.py"))
                },
                seed=20261110,
                repeats=4,
                grids=[8, 16],
                budgets=[256, 1024],
                python=platform.python_version(),
                platform=platform.platform(),
                point="centre",
                timing="per solver including preparation; excludes JIT/reference/plots",
                purpose="fixed-budget comparison, not independent confirmation of thresholds",
            ),
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    print(f"Saved {len(rows)} independent-run rows to {out}")


if __name__ == "__main__":
    main()
