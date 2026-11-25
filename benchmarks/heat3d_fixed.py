"""Fixed-N whole histories for group-size comparisons using the public API."""

from dataclasses import replace
from statistics import mean, stdev

import numpy as np

from pdelab.heat3d_experiment import compare_heat3d, csv_bytes
from pdelab.study_protocol import error_metrics


def fixed_runs(plan, config, settings):
    trials = []
    metadata = None
    for repeat in range(plan["repeats"]):
        seed = int(np.random.SeedSequence([plan["seed"], repeat]).generate_state(1)[0])
        run = compare_heat3d(config, replace(settings, seed=seed))
        row = run["rows"][0]
        method = row["method"]
        trials.append(
            {
                **row,
                "repeat": repeat + 1,
                "values": run["arrays"].get(method),
            }
        )
        metadata = run["metadata"]
    complete = all(row["status"] == "completed" for row in trials)
    summary = {
        "requested_repeats": plan["repeats"],
        "completed_repeats": sum(row["status"] == "completed" for row in trials),
        "complete": complete,
        "walks_per_free_node": settings.walks,
        "batch_size": trials[0]["batch_size"],
        "method": config.methods[0],
        "relative_error_pct": None,
        "absolute_error": None,
        "completed_mean_seconds": (
            mean(row["seconds"] for row in trials) if complete else None
        ),
        "attempted_mean_seconds": mean(row["seconds"] for row in trials),
        "attempted_seconds_std": stdev(row["seconds"] for row in trials),
        "seconds_all_attempts": sum(row["seconds"] for row in trials),
        "preparation_seconds": sum(row["preparation_seconds"] for row in trials),
        "coarse_seconds": sum(row["coarse_seconds"] for row in trials),
        "walks": sum(row["walks"] for row in trials),
        "completed_walks": sum(row["completed_walks"] for row in trials),
        "transitions": sum(row["transitions"] for row in trials),
        "workspace_bytes": max(row["workspace_bytes"] for row in trials),
        "reason": (
            "Все повторы завершены"
            if complete
            else " | ".join(
                sorted(
                    {row["reason"] for row in trials if row["status"] != "completed"}
                )
            )
        ),
    }
    if complete:
        summary.update(
            error_metrics([row["values"] for row in trials], run["arrays"]["analytic"])
        )
    return {
        "metadata": metadata,
        "seed_rule": "SeedSequence([plan.seed, zero_based_repeat]), paired across variants",
        "summary": summary,
        "trials": trials,
    }


def summarize_fixed(results, out):
    summaries, trials = [], []
    for name, payload in results:
        result = payload["fixed_budget"]
        summaries.append(
            {"scenario_variant": name, **result["summary"], **payload["process"]}
        )
        trials.extend({"scenario_variant": name, **row} for row in result["trials"])
    (out / "summary.csv").write_bytes(csv_bytes(summaries))
    (out / "trials.csv").write_bytes(csv_bytes(trials))
