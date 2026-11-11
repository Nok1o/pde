"""Confirm selected UI budgets on fresh data, without retuning any parameter."""

import hashlib
import json
from pathlib import Path

import numpy as np

from benchmarks.project_audit import prepare
from pdelab.boundary_solvers3d import solve_boundary3d
from pdelab.heat import solve_heat
from pdelab.heat_mc import solve_heat_mc
from pdelab.solvers import solve
from pdelab.solvers3d import solve3d

OUT = Path("benchmarks/results/demo-20261111")
REPEATS = 50
SEED = 202611115


def run(data, prepared, method, budget, seed):
    problem, points, target, _ = prepared
    case = data["case"]
    family = case["family"]
    deterministic = method in ("lod", "lod_serial")
    if family == "heat":
        if deterministic:
            result = solve_heat(problem, case["dt"], case["T"], tol=budget)
            assert result.times[-1] == case["T"]
            value = result.fields[-1][tuple(points[0])]
        else:
            result = solve_heat_mc(
                problem, case["dt"], case["T"], points, walks=int(budget), seed=seed
            )
            value = result.values[0]
        seconds = result.seconds
    else:
        config = data["config"] if family == "mixed3d" else data["settings"]["config"]
        kwargs = dict(
            walks=64 if deterministic else int(budget),
            seed=seed,
            tol=float(budget) if deterministic else config["tol"],
            max_iter=config["max_iter"],
        )
        if family == "2d":
            kwargs.update(
                record=False, batches=config["batches"], coarse_n=config["coarse_n"]
            )
        if family == "mixed3d":
            kwargs["max_steps"] = config["max_steps"]
        solver = {"2d": solve, "3d": solve3d, "mixed3d": solve_boundary3d}[family]
        result = solver(problem, points, method, **kwargs)
        assert result.converged
        value = result.values[tuple(points[0])]
        seconds = result.elapsed
    return dict(
        value=float(value),
        seconds=float(seconds),
        error_pct=float(100 * abs(value - target[0]) / abs(target[0])),
    )


def main():
    protocol = json.loads((OUT / "protocol.json").read_text())
    loaded = [
        json.loads((OUT / (c["id"] + ".json")).read_text()) for c in protocol["cases"]
    ]
    old_hashes = protocol["source_sha256"]
    changes = [
        name
        for name, digest in old_hashes.items()
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != digest
    ]
    # Another chat changed only the unrelated nonlinear UI during the studies.
    assert set(changes) <= {"pdelab/nonlinear_ui.py"}, changes
    audit = dict(
        changed_files=changes,
        computational_sources_unchanged=True,
        note="nonlinear_ui.py is not imported or exercised by these scenarios",
    )
    (OUT / "source-audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2)
    )
    results = dict(
        repeats=REPEATS,
        seed=SEED,
        comparisons=[],
        trials=[],
        selection="UI candidate budgets fixed before this independent series; no retuning",
        interval="Paired percentile bootstrap, 5000 resamples, 95%; no multiplicity adjustment",
    )
    saved = OUT / "confirmation.json"
    if saved.exists():
        results = json.loads(saved.read_text())
    completed = {r["case"] for r in results["comparisons"]}
    for case_index, data in enumerate(loaded):
        if data["case"]["id"] in completed:
            continue
        baseline = {
            r["target_pct"]: r
            for r in data["rows"]
            if r["method"] in ("lod", "lod_serial")
        }
        selected = []
        jobs = set()
        for row in data["rows"]:
            b = baseline[row["target_pct"]]
            if (
                row["method"] in ("lod", "lod_serial")
                or not row["seconds"]
                or not b["seconds"]
                or row["seconds"] >= b["seconds"]
            ):
                continue
            selected.append((row, b))
            jobs.add((row["method"], row["budget"]))
            jobs.add((b["method"], b["budget"]))
        if not selected:
            continue
        jobs = sorted(jobs)
        case = dict(data["case"], mode="point")
        prepared = prepare(case)
        # Compile signatures / perform one unrecorded call outside timing series.
        for method, budget in jobs:
            run(data, prepared, method, budget, SEED)
        samples = {job: [] for job in jobs}
        for repeat in range(REPEATS):
            rng = np.random.default_rng([SEED, case_index, repeat])
            for job_index in rng.permutation(len(jobs)):
                method, budget = jobs[job_index]
                seed = int(
                    np.random.SeedSequence(
                        [SEED, case_index, repeat, int(job_index)]
                    ).generate_state(1)[0]
                )
                measurement = run(data, prepared, method, budget, seed)
                samples[(method, budget)].append(measurement)
                results["trials"].append(
                    dict(
                        case=case["id"],
                        method=method,
                        budget=budget,
                        repeat=repeat,
                        seed=seed,
                        **measurement,
                    )
                )
        bootstrap = np.random.default_rng([SEED, case_index, 999]).integers(
            0, REPEATS, size=(5000, REPEATS)
        )
        for row, base in selected:
            x = samples[(row["method"], row["budget"])]
            b = samples[(base["method"], base["budget"])]
            errors = np.array([v["error_pct"] for v in x])
            baseline_error = float(np.mean([v["error_pct"] for v in b]))
            times = np.array([v["seconds"] for v in x])
            baseline_times = np.array([v["seconds"] for v in b])
            speedup = float(np.mean(baseline_times) / np.mean(times))
            error_ci = np.quantile(
                np.mean(errors[bootstrap], axis=1), [0.025, 0.975]
            ).tolist()
            speed_ci = np.quantile(
                np.mean(baseline_times[bootstrap], axis=1)
                / np.mean(times[bootstrap], axis=1),
                [0.025, 0.975],
            ).tolist()
            target = row["target_pct"]
            reasons = []
            if baseline_error > target:
                reasons.append("ЛОМ не прошёл порог")
            if float(np.mean(errors)) > target:
                reasons.append("MC не прошёл порог")
            elif error_ci[1] > target:
                reasons.append("недостаточный запас точности в новой серии")
            if speedup <= 1:
                reasons.append("MC медленнее ЛОМ")
            elif speed_ci[0] <= 1:
                reasons.append("малый выигрыш неустойчив по времени")
            results["comparisons"].append(
                dict(
                    case=case["id"],
                    method=row["method"],
                    target_pct=target,
                    budget=row["budget"],
                    baseline_budget=base["budget"],
                    error_pct=float(np.mean(errors)),
                    baseline_error_pct=baseline_error,
                    seconds=float(np.mean(times)),
                    baseline_seconds=float(np.mean(baseline_times)),
                    speedup=speedup,
                    error_ci=error_ci,
                    speed_ci=speed_ci,
                    confirmed=not reasons,
                    reason="; ".join(reasons) or "повторено",
                )
            )
            print(
                case["id"],
                row["method"],
                target,
                "ratio",
                round(speedup, 2),
                "error",
                round(float(np.mean(errors)), 3),
                "OK" if not reasons else reasons,
                flush=True,
            )
        (OUT / "confirmation.json").write_text(
            json.dumps(results, ensure_ascii=False, indent=2, allow_nan=False)
        )
    print("Confirmation complete", flush=True)


if __name__ == "__main__":
    main()
