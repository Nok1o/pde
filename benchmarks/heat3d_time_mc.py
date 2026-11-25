"""Small fixed-budget check of finite-sample error against a full Newton history."""

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np

from pdelab.heat3d_experiment import HeatConfig3D, compare_heat3d, csv_bytes, json_bytes
from pdelab.heat3d_solver import HeatSettings3D
from pdelab.study_protocol import error_metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    config = HeatConfig3D(
        case="polynomial",
        n=6,
        alpha=0.5,
        dt=0.004,
        T=0.007,
        kinds=(3, 1, 3, 1, 3, 1),
        point=(0, 0, 0),
        newton=True,
    )
    reference = compare_heat3d(
        config, HeatSettings3D(tolerance=1e-11, change_tolerance=1e-11)
    )
    if reference["grid_diagnostic"]["status"] != "completed":
        raise RuntimeError("Newton reference failed")
    target = float(reference["arrays"]["newton"][0])
    analytic = float(reference["arrays"]["analytic"][0])
    repeats = 6
    trials, summary = [], []
    for method in ("mc", "mc_control"):
        for budget, tol, change in (
            (256, 0.2, 0.1),
            (1024, 0.2, 0.1),
            (1024, 0.01, 0.001),
        ):
            sample = []
            for repeat in range(repeats):
                seed = int(np.random.SeedSequence([2803, repeat]).generate_state(1)[0])
                settings = HeatSettings3D(
                    walks=budget,
                    seed=seed,
                    tolerance=tol,
                    change_tolerance=change,
                    max_outer=8,
                )
                result = compare_heat3d(
                    replace(config, methods=(method,), newton=False), settings
                )
                row = result["rows"][0]
                value = (
                    float(result["arrays"][method][0])
                    if row["status"] == "completed"
                    else None
                )
                sample.append(dict(**row, repeat=repeat + 1, value=value))
            trials.extend(sample)
            values = [r["value"] for r in sample if r["status"] == "completed"]
            complete = len(values) == repeats
            summary.append(
                dict(
                    method=method,
                    walks_per_node=budget,
                    tolerance=tol,
                    change_tolerance=change,
                    requested=repeats,
                    completed=len(values),
                    signed_difference=(
                        float(np.mean(values) - target) if complete else None
                    ),
                    standard_error=(
                        float(np.std(values, ddof=1) / np.sqrt(repeats))
                        if complete
                        else None
                    ),
                    analytic_mean_relative_pct=(
                        error_metrics(values, [analytic])["relative_error_pct"]
                        if complete
                        else None
                    ),
                    seconds_all_attempts=sum(r["seconds"] for r in sample),
                    walks_all_attempts=sum(r["walks"] for r in sample),
                    reasons=sorted(
                        {r["reason"] for r in sample if r["status"] != "completed"}
                    ),
                )
            )
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "trials.csv").write_bytes(csv_bytes(trials))
    (args.out / "summary.csv").write_bytes(csv_bytes(summary))
    (args.out / "settings.json").write_bytes(
        json_bytes(
            dict(
                reference=reference["metadata"],
                newton_value=target,
                analytic_value=analytic,
                comparison="signed mean at T minus independent grid Newton, complete series only",
                matched_repeat_seeds=True,
                max_outer=8,
                repeats=repeats,
            )
        )
    )


if __name__ == "__main__":
    main()
