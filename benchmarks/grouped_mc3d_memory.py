"""Replay frozen final budgets in fresh processes to measure their peak RSS.

This adds memory observations, never changes a selected budget or accuracy
status, and never replaces original time-to-accuracy measurements. One worker
replays the first recorded repeat of each distinct final series. Allocation
sizes depend on its grid, path budget and batch size, not the number of repeats.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .grouped_mc3d import (
    CASES,
    VARIANTS,
    rss_diagnostics,
    source_state,
    write_csv,
    write_json,
)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    raw = args.results.read_bytes()
    study = json.loads(raw)
    cases = {case["id"]: case for case in CASES}
    variants = {variant["id"]: variant for variant in VARIANTS}
    plan = [
        trial
        for trial in study["trials"]
        if trial["phase"] == "final" and trial["repeat"] == 1
    ]
    source = source_state()
    manifest = dict(
        source=source,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        input_results_sha256=hashlib.sha256(raw).hexdigest(),
        sampling="first repeat of every distinct frozen final series; no reselection",
        scope="whole fresh worker RSS including imports and warmup; no subtraction",
        plan=[
            {
                key: trial[key]
                for key in ("case_id", "variant", "candidate", "budget", "seed")
            }
            for trial in plan
        ],
        completed=False,
    )
    args.out.mkdir(parents=True, exist_ok=True)
    write_json(args.out / "memory-at-target-manifest.json", manifest)
    observations = []
    for trial in plan:
        case_id, variant_id = trial["case_id"], trial["variant"]
        print(f"{case_id} / {variant_id} / {trial['budget']:g}", flush=True)
        memory = rss_diagnostics(
            cases[case_id],
            variants[variant_id],
            budget=trial["budget"],
            seed=trial["seed"],
        )
        values_match = memory.get("values") is not None and bool(
            np.allclose(memory["values"], trial["values"], rtol=1e-13, atol=0)
        )
        memory.update(
            candidate=trial["candidate"],
            budget=trial["budget"],
            original_repeat=trial["repeat"],
            original_values=trial["values"],
            replay_values_match=values_match,
            target_results=[
                {
                    key: row[key]
                    for key in (
                        "target_pct",
                        "status_code",
                        "relative_error_pct",
                        "seconds",
                    )
                }
                for row in study["rows"]
                if row["case_id"] == case_id
                and row["variant"] == variant_id
                and row["candidate"] == trial["candidate"]
                and row["final_series"]
            ],
        )
        observations.append(memory)
        write_json(args.out / "memory-at-target.json", observations)
        write_csv(args.out / "memory-at-target.csv", observations)
    after = source_state()
    manifest.update(
        source_after=after,
        source_unchanged=(
            source["commit"] == after["commit"]
            and source["combined_sha256"] == after["combined_sha256"]
        ),
        all_replays_match=all(row["replay_values_match"] for row in observations),
        completed=True,
    )
    write_json(args.out / "memory-at-target-manifest.json", manifest)
    if not manifest["source_unchanged"] or not manifest["all_replays_match"]:
        raise RuntimeError(
            "Memory replay changed source or failed to reproduce frozen final observations"
        )


if __name__ == "__main__":
    main()
