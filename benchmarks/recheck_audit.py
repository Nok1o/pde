"""Recheck every archived row, replay validation, then run a fresh MAE study.

Run with python -m benchmarks.recheck_audit. Never overwrites historical evidence.
All algorithm parameters are selected from pilot data before fresh validation.
"""

import argparse
import json
import platform
import subprocess
from pathlib import Path
from time import perf_counter

import numpy as np

from benchmarks import project_audit as audit
from benchmarks.project_audit_refine import DENSE_TOLS
from pdelab.study_policy import mean_relative_error

OUT = Path("benchmarks/results/audit-recheck-20261111")
SEED = 202611113
REPEATS = 50
PILOT = 5
TOLS = sorted(set(DENSE_TOLS + [0.9, 0.75, 0.5, 0.3, 0.2, 0.15]), reverse=True)


def save(name, data):
    (OUT / name).write_text(
        json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )


def measure(case, prepared, method, params, seed):
    problem, points, target, grid = prepared
    family = case["family"]
    tol = params.get("tol", 1e-8)
    walks = params.get("walks", 64)
    started = perf_counter()
    if family in ("2d", "3d", "mixed2d", "mixed3d"):
        solver = {
            "2d": audit.solve,
            "3d": audit.solve3d,
            "mixed2d": audit.solve_boundary,
            "mixed3d": audit.solve_boundary3d,
        }[family]
        extra = dict(record=False) if family in ("2d", "mixed2d") else {}
        result = solver(
            problem,
            points,
            "lod_serial" if method == "lod" else method,
            walks=walks,
            seed=seed,
            tol=tol,
            max_iter=20000,
            max_steps=1000 * case["n"] ** 2,
            **extra,
        )
        if not result.converged:
            raise RuntimeError("not_converged")
        values = result.values[tuple(points.T)]
        elapsed = result.elapsed
        stats = dict(
            walks=result.walks,
            steps=result.steps,
            iterations=result.iterations,
            preparation_seconds=result.extras.get("preparation_seconds"),
            path_seconds=result.extras.get("path_seconds"),
            coarse_n=result.extras.get("coarse_n"),
        )
    elif family == "heat":
        if method == "lod":
            result = audit.solve_heat(problem, case["dt"], case["T"], "lod", tol)
            values = result.fields[-1][tuple(points.T)]
            stats = dict(iterations=int(result.iterations.sum()))
        else:
            result = audit.solve_heat_mc(
                problem, case["dt"], case["T"], points, walks, seed
            )
            values = result.values
            stats = dict(
                walks=result.walks,
                preparation_seconds=result.preparation_seconds,
                path_seconds=result.path_seconds,
            )
        elapsed = result.seconds
    else:
        settings = audit.NonlinearSettings(
            tolerance=tol,
            change_tolerance=tol,
            linear_tolerance=1e-10,
            relaxation=0.7,
            max_outer=50,
            walks=walks,
            max_walks=4 * walks,
            schedule="geometric",
            seed=seed,
        )
        result = audit.solve_nonlinear(
            problem,
            case["dt"],
            case["T"],
            "picard" if method == "lod" else "mc",
            settings,
        )
        if not result.completed:
            raise RuntimeError(result.reason)
        values = result.fields[-1][tuple(points.T)]
        elapsed = result.seconds
        stats = dict(
            walks=result.walks,
            steps=result.steps,
            iterations=sum(len(layer.history) for layer in result.layers),
        )
    wall = perf_counter() - started
    values = np.asarray(values)
    return dict(
        seconds=elapsed,
        external_seconds=wall,
        value=values.tolist(),
        relative_error_pct=mean_relative_error(values, target),
        squared_error=float(
            10000 * np.mean((values - target) ** 2) / np.mean(target**2)
        ),
        value_mean=float(values.mean()),
        **stats,
    )


def archive_check():
    checked = 0
    issues = []
    for case in audit.cases():
        for suffix in ("", "-refined"):
            data = json.loads((audit.OUT / (case["id"] + suffix + ".json")).read_text())
            for row in data["comparisons"]:
                checked += 1
                if row["seconds"] is None:
                    continue
                group = [
                    t
                    for t in data["trials"]
                    if t["phase"] in (1, "validation")
                    and t["method"] == row["method"]
                    and t["parameters"] == row["parameters"]
                ]
                if not group or any(t["status"] != "ok" for t in group):
                    issues.append(
                        [case["id"], suffix, row["method"], "missing successful trials"]
                    )
                    continue
                error = float(np.sqrt(np.mean([t["squared_error"] for t in group])))
                seconds = float(np.mean([t["seconds"] for t in group]))
                if not np.isclose(error, row["error_pct"], rtol=1e-12, atol=1e-12):
                    issues.append(
                        [case["id"], suffix, row["method"], "error aggregation"]
                    )
                if not np.isclose(seconds, row["seconds"], rtol=1e-12):
                    issues.append(
                        [case["id"], suffix, row["method"], "time aggregation"]
                    )
                if row.get("speedup") is not None:
                    baseline = next(
                        b
                        for b in data["comparisons"]
                        if b["method"] == "lod" and b["target_pct"] == row["target_pct"]
                    )
                    if not np.isclose(
                        baseline["seconds"] / seconds, row["speedup"], rtol=1e-12
                    ):
                        issues.append(
                            [case["id"], suffix, row["method"], "speed ratio"]
                        )
    return dict(checked_rows=checked, issues=issues)


def quantile_mean(values, rng):
    values = np.asarray(values)
    draws = rng.integers(0, len(values), (5000, len(values)))
    return np.quantile(values[draws].mean(axis=1), [0.025, 0.975]).tolist()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cases = audit.cases() + [
        dict(
            id=f"C{n}",
            family="2d",
            problem="sine",
            n=n,
            mode="point",
            methods=["lod", "mc_control"],
        )
        for n in (24, 48)
    ]
    save(
        "protocol.json",
        dict(
            revision=subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            seed=SEED,
            pilot=PILOT,
            validation=REPEATS,
            tolerances=TOLS,
            budgets=audit.BUDGETS,
            cases=cases,
            metric="mean_run_relative_l1_v1",
            margin=0.5,
            selection="fastest pilot mean among eligible candidates; no retuning on final data",
            timing="full solver incl preparation; JIT, common inputs, reference and plots excluded; randomized serial order",
            uncertainty="paired bootstrap means; 95% intervals per comparison, no multiple-testing correction",
            platform=platform.platform(),
            python=platform.python_version(),
        ),
    )
    save("archive-check.json", archive_check())
    all_rows = []
    for case_index, case in enumerate(cases):
        name = case["id"] + ".json"
        if (OUT / name).exists():
            all_rows.extend(json.loads((OUT / name).read_text())["comparisons"])
            continue
        print("START", case["id"], flush=True)
        prepared = audit.prepare(case)
        # Warm each method using real shapes, never include compilation in timings.
        for method in case["methods"]:
            params = dict(tol=0.01) if method == "lod" else dict(walks=64, tol=0.01)
            try:
                measure(case, prepared, method, params, SEED)
            except RuntimeError:
                pass
        replay = []
        archived_path = audit.OUT / (case["id"] + "-refined.json")
        archived = (
            json.loads(archived_path.read_text()) if archived_path.exists() else None
        )
        if archived:
            for old in archived["trials"]:
                if old["phase"] != 1:
                    continue
                item = dict(
                    method=old["method"],
                    parameters=old["parameters"],
                    seed=old["seed"],
                    old_seconds=old["seconds"],
                    old_squared_error=old.get("squared_error"),
                )
                try:
                    item.update(
                        measure(
                            case,
                            prepared,
                            old["method"],
                            old["parameters"],
                            old["seed"],
                        ),
                        status="ok",
                    )
                except RuntimeError as exc:
                    item["status"] = str(exc)
                replay.append(item)
        jobs = []
        for method in case["methods"]:
            parameters = (
                [dict(tol=t) for t in TOLS]
                if method == "lod"
                else audit.candidates(case, method)
            )
            jobs.extend((method, params) for params in parameters)
        trials = []

        def run_job(index, phase, repeat):
            method, params = jobs[index]
            seed = int(
                np.random.SeedSequence(
                    [SEED, case_index, index, phase, repeat]
                ).generate_state(1)[0]
            )
            item = dict(
                method=method,
                parameters=params,
                candidate=int(index),
                phase=phase,
                repeat=repeat,
                seed=seed,
            )
            started = perf_counter()
            try:
                item.update(measure(case, prepared, method, params, seed), status="ok")
            except (ValueError, RuntimeError) as exc:
                item.update(status=str(exc), seconds=perf_counter() - started)
            trials.append(item)

        for repeat in range(PILOT):
            rng = np.random.default_rng(
                np.random.SeedSequence([SEED, case_index, 0, repeat])
            )
            for index in rng.permutation(len(jobs)):
                run_job(int(index), 0, repeat)
        selected = {}
        for method in case["methods"]:
            for target in audit.TARGETS:
                threshold = target if method == "lod" else target * 0.5
                eligible = []
                for index, (m, params) in enumerate(jobs):
                    if m != method:
                        continue
                    group = [
                        t for t in trials if t["candidate"] == index and t["phase"] == 0
                    ]
                    if len(group) == PILOT and all(t["status"] == "ok" for t in group):
                        error = np.mean([t["relative_error_pct"] for t in group])
                        if error <= threshold:
                            eligible.append(
                                (np.mean([t["seconds"] for t in group]), index)
                            )
                selected[method, target] = min(eligible)[1] if eligible else None
        chosen = sorted(set(i for i in selected.values() if i is not None))
        for repeat in range(REPEATS):
            rng = np.random.default_rng(
                np.random.SeedSequence([SEED, case_index, 1, repeat])
            )
            for index in rng.permutation(chosen):
                run_job(int(index), 1, repeat)
        comparisons = []
        for (method, target), index in selected.items():
            row = dict(
                case=case["id"],
                method=method,
                target_pct=target,
                parameters=None if index is None else jobs[index][1],
                status="no_pilot_margin",
                seconds=None,
                error_pct=None,
                speedup=None,
                repeat_win=False,
            )
            if index is not None:
                group = [
                    t for t in trials if t["phase"] == 1 and t["candidate"] == index
                ]
                if all(t["status"] == "ok" for t in group) and len(group) == REPEATS:
                    errors = [t["relative_error_pct"] for t in group]
                    seconds = [t["seconds"] for t in group]
                    error = float(np.mean(errors))
                    row.update(
                        status="confirmed" if error <= target else "validation_failed",
                        seconds=float(np.mean(seconds)),
                        median_seconds=float(np.median(seconds)),
                        error_pct=error,
                        error_ci=quantile_mean(
                            errors, np.random.default_rng([SEED, case_index, index, 2])
                        ),
                        rms_pct=float(
                            np.sqrt(np.mean([t["squared_error"] for t in group]))
                        ),
                        external_seconds=float(
                            np.mean([t["external_seconds"] for t in group])
                        ),
                    )
                else:
                    row["status"] = "solver_failed"
            comparisons.append(row)
        for row in comparisons:
            if row["method"] == "lod":
                continue
            baseline = next(
                r
                for r in comparisons
                if r["method"] == "lod" and r["target_pct"] == row["target_pct"]
            )
            if row["status"] != "confirmed" or baseline["status"] != "confirmed":
                continue

            def times(r):
                idx = selected[r["method"], r["target_pct"]]
                return np.array(
                    [
                        t["seconds"]
                        for t in trials
                        if t["phase"] == 1 and t["candidate"] == idx
                    ]
                )

            a, b = times(baseline), times(row)
            rng = np.random.default_rng(
                [SEED, case_index, selected[row["method"], row["target_pct"]], 3]
            )
            draws = rng.integers(0, REPEATS, (5000, REPEATS))
            ci = np.quantile(
                a[draws].mean(axis=1) / b[draws].mean(axis=1), [0.025, 0.975]
            ).tolist()
            row.update(
                speedup=float(a.mean() / b.mean()),
                speed_ci=ci,
                repeat_win=bool(ci[0] > 1 and row["error_ci"][1] <= row["target_pct"]),
            )
        same = [r for r in replay if r["status"] == "ok"]
        maxdiff = max(
            (abs(r["squared_error"] - r["old_squared_error"]) for r in same), default=0
        )
        save(
            name,
            dict(
                case=case,
                target=prepared[2].tolist(),
                points=prepared[1].tolist(),
                replay=replay,
                replay_max_squared_error_difference=maxdiff,
                trials=trials,
                comparisons=comparisons,
            ),
        )
        all_rows.extend(comparisons)
        audit.write_csv(OUT / "comparisons.csv", all_rows)
        print(
            "DONE",
            case["id"],
            "replay",
            len(replay),
            "fresh",
            len(trials),
            "wins",
            sum(r["repeat_win"] for r in comparisons),
            "max replay error delta",
            maxdiff,
            flush=True,
        )
    print("COMPLETE", len(all_rows), "rows", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    OUT = args.output
    main()
