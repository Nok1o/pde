"""Independent robustness check: denser baseline, frozen MC candidates, fresh seeds."""

import json
from time import perf_counter

import numpy as np

from .project_audit import (
    OUT,
    TARGETS,
    bootstrap_rms,
    cases,
    prepare,
    run,
    seed_for,
    write_csv,
)

DENSE_TOLS = sorted(
    set(
        np.geomspace(0.1, 1e-8, 57).tolist()
        + [0.05, 0.01, 0.005, 0.001, 0.0001, 0.000001, 0.00000001]
    ),
    reverse=True,
)


def main():
    (OUT / "refinement-protocol.json").write_text(
        json.dumps(
            dict(
                purpose="robustness to finer baseline tuning; not a rescue of failed MC thresholds",
                pilot=3,
                validation=20,
                baseline_tolerances=DENSE_TOLS,
                mc_candidates="fixed from initial pilot; only initially confirmed rows eligible; no retuning",
                seed_namespace=6,
                strong_win="both stages passed, both MC bootstrap upper errors below target, final speed CI lower > 1",
            ),
            indent=2,
        )
        + "\n"
    )
    final, all_trials = [], []
    for case_index, case in enumerate(cases()):
        path = OUT / (case["id"] + "-refined.json")
        if path.exists():
            data = json.loads(path.read_text())
            final.extend(data["comparisons"])
            all_trials.extend(data["trials"])
            continue
        original = json.loads((OUT / (case["id"] + ".json")).read_text())
        prepared = prepare(case)
        old = {(r["method"], r["target_pct"]): r for r in original["comparisons"]}
        trials = []

        def measure(method, params, phase, repeat, candidate):
            mi = case["methods"].index(method)
            seed = seed_for(case_index, mi, 6, phase, repeat, candidate)
            row = dict(
                case=case["id"],
                method=method,
                parameters=params,
                phase=phase,
                repeat=repeat,
                candidate=candidate,
                seed=seed,
            )
            started = perf_counter()
            try:
                row.update(run(case, prepared, method, params, seed), status="ok")
            except (ValueError, RuntimeError) as exc:
                row.update(seconds=perf_counter() - started, status=str(exc))
            trials.append(row)

        # Compile signatures outside measured series.
        for method in case["methods"]:
            row = next(
                (
                    r
                    for r in original["comparisons"]
                    if r["method"] == method and r["status"] == "confirmed"
                ),
                None,
            )
            if row:
                run(
                    case,
                    prepared,
                    method,
                    row["parameters"],
                    seed_for(case_index, 6, 99),
                )
        for repeat in range(3):
            for index in np.random.default_rng(
                seed_for(case_index, 6, 0, repeat)
            ).permutation(len(DENSE_TOLS)):
                measure("lod", dict(tol=DENSE_TOLS[index]), 0, repeat, int(index))
        selected = {}
        for target in TARGETS:
            eligible = []
            for index, tol in enumerate(DENSE_TOLS):
                group = [r for r in trials if r["candidate"] == index]
                if (
                    all(r["status"] == "ok" for r in group)
                    and np.sqrt(np.mean([r["squared_error"] for r in group])) <= target
                ):
                    eligible.append(
                        (np.mean([r["seconds"] for r in group]), index, dict(tol=tol))
                    )
            selected["lod", target] = min(eligible)[2] if eligible else None
        for method in case["methods"][1:]:
            for target in TARGETS:
                row = old[method, target]
                selected[method, target] = (
                    row["parameters"] if row["status"] == "confirmed" else None
                )
        jobs = []
        for (method, target), params in selected.items():
            if params is not None and (method, params) not in jobs:
                jobs.append((method, params))
        for repeat in range(20):
            for index in np.random.default_rng(
                seed_for(case_index, 6, 1, repeat)
            ).permutation(len(jobs)):
                method, params = jobs[index]
                measure(method, params, 1, repeat, int(index))
        rows = []
        for (method, target), params in selected.items():
            row = dict(
                case=case["id"],
                method=method,
                target_pct=target,
                parameters=params,
                initial_status=old[method, target]["status"],
                initial_error=old[method, target]["error_pct"],
                status=(
                    "not_selected" if method == "lod" else old[method, target]["status"]
                ),
                seconds=None,
                error_pct=None,
                error_ci_low=None,
                error_ci_high=None,
                speedup=None,
                speed_ci_low=None,
                speed_ci_high=None,
                robust_win=False,
            )
            group = [
                r
                for r in trials
                if r["phase"] == 1
                and r["method"] == method
                and r["parameters"] == params
            ]
            if group:
                if all(r["status"] == "ok" for r in group):
                    error = float(np.sqrt(np.mean([r["squared_error"] for r in group])))
                    low, high = bootstrap_rms(
                        [r["squared_error"] for r in group],
                        seed_for(case_index, case["methods"].index(method), 6, 2),
                    )
                    row.update(
                        status="confirmed" if error <= target else "replication_failed",
                        seconds=(
                            float(np.mean([r["seconds"] for r in group]))
                            if error <= target
                            else None
                        ),
                        error_pct=error,
                        error_ci_low=low,
                        error_ci_high=high,
                    )
                else:
                    row["status"] = "solver_failed"
            rows.append(row)
        for row in rows:
            method, target = row["method"], row["target_pct"]
            baseline = next(
                r for r in rows if r["method"] == "lod" and r["target_pct"] == target
            )
            if method != "lod" and row["status"] == baseline["status"] == "confirmed":

                def times(r):
                    return [
                        t["seconds"]
                        for t in trials
                        if t["phase"] == 1
                        and t["method"] == r["method"]
                        and t["parameters"] == r["parameters"]
                    ]

                a, b = times(baseline), times(row)
                rng = np.random.default_rng(
                    seed_for(case_index, case["methods"].index(method), 6, 3)
                )
                ratios = np.mean(rng.choice(a, (5000, len(a))), axis=1) / np.mean(
                    rng.choice(b, (5000, len(b))), axis=1
                )
                low, high = np.quantile(ratios, [0.025, 0.975])
                row.update(
                    speedup=baseline["seconds"] / row["seconds"],
                    speed_ci_low=float(low),
                    speed_ci_high=float(high),
                    robust_win=bool(
                        low > 1
                        and row["error_ci_high"] <= target
                        and baseline["error_ci_high"] <= target
                        and old[method, target]["error_ci_high"] <= target
                    ),
                )
        path.write_text(
            json.dumps(
                dict(case=case, comparisons=rows, trials=trials),
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
            + "\n"
        )
        final.extend(rows)
        all_trials.extend(trials)
        write_csv(OUT / "refined-comparisons.csv", final)
        write_csv(OUT / "refined-trials.csv", all_trials)
        print(case["id"], "refined", len(trials), "runs", flush=True)
    print("DONE", len(all_trials), "runs", flush=True)


if __name__ == "__main__":
    main()
