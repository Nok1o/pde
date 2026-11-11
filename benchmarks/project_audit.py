"""Predeclared equal-accuracy audit. Run: python -m benchmarks.project_audit.

No solver changes, no selection on final data. Full repetitions are independent.
The PDF builder consumes the saved protocol, raw runs and final comparisons.
"""

import csv
import hashlib
import json
import platform
import subprocess
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import numpy as np

from pdelab.boundary import BoundaryOperator
from pdelab.boundary3d import BoundaryOperator3D, select_boundary_points3d
from pdelab.boundary_experiment import BoundaryConfig, problem_from_config
from pdelab.boundary_experiment3d import BoundaryConfig3D, problem_from_config3d
from pdelab.boundary_solvers import (
    select_boundary_points,
    solve_boundary,
    warmup_boundary,
)
from pdelab.boundary_solvers3d import solve_boundary3d, warmup_boundary3d
from pdelab.heat import make_heat_problem, solve_heat
from pdelab.heat_mc import solve_heat_mc
from pdelab.nonlinear import NonlinearSettings, make_nonlinear_problem, solve_nonlinear
from pdelab.problem import make_problem, reference_solution, select_points
from pdelab.problem3d import make_problem3d, select_points3d
from pdelab.reference import choose_reference
from pdelab.solvers import solve, warmup
from pdelab.solvers3d import solve3d, warmup3d

OUT = Path("benchmarks/results/project-audit")
SEED = 202611101
PILOT = 5
VALIDATION = 20
TARGETS = (1.0, 5.0, 10.0)
TOLS = (0.1, 0.05, 0.01, 0.005, 0.001, 0.0001, 0.000001, 0.00000001)
BUDGETS = (64, 256, 1024, 4096, 16384)


def cases():
    return [
        dict(
            id="S01",
            family="2d",
            problem="sine",
            n=16,
            mode="point",
            methods=["lod", "mc", "mc_control", "reuse", "layers", "wos", "graded"],
        ),
        dict(
            id="S02",
            family="2d",
            problem="sine",
            n=64,
            mode="point",
            methods=["lod", "mc", "mc_control", "wos", "graded"],
        ),
        dict(
            id="S03",
            family="2d",
            problem="polynomial",
            n=64,
            mode="point",
            point=[16, 21],
            methods=["lod", "mc", "mc_control", "wos", "graded"],
        ),
        dict(
            id="S04",
            family="2d",
            problem="hot_top",
            n=64,
            mode="point",
            methods=["lod", "mc", "mc_control", "wos", "graded"],
        ),
        dict(
            id="S05",
            family="2d",
            problem="sine",
            n=16,
            mode="patch",
            max_walks=1024,
            methods=["lod", "mc", "mc_control", "reuse", "layers", "wos", "graded"],
        ),
        dict(
            id="S06",
            family="2d",
            problem="sine",
            n=16,
            mode="field",
            max_walks=1024,
            methods=["lod", "mc", "mc_control", "reuse", "layers", "wos", "graded"],
        ),
        dict(
            id="S07",
            family="3d",
            problem="sine",
            n=8,
            mode="point",
            methods=["lod", "mc", "wos", "graded"],
        ),
        dict(
            id="S08",
            family="3d",
            problem="sine",
            n=16,
            mode="point",
            methods=["lod", "mc", "wos", "graded"],
        ),
        dict(
            id="S09",
            family="mixed2d",
            problem="smooth",
            n=16,
            methods=["lod", "mc", "graded"],
        ),
        dict(
            id="S10",
            family="mixed3d",
            problem="smooth",
            n=16,
            methods=["lod", "mc", "graded"],
        ),
        dict(
            id="S11",
            family="mixed3d",
            problem="polynomial",
            n=16,
            point=[4, 6, 8],
            kinds=[
                "neumann",
                "dirichlet",
                "dirichlet",
                "dirichlet",
                "dirichlet",
                "robin",
            ],
            methods=["lod", "mc", "graded"],
        ),
        dict(
            id="S12",
            family="2d",
            problem="sine",
            n=64,
            mode="point",
            point=[1, 24],
            methods=["lod", "mc", "mc_control", "wos", "graded"],
        ),
        dict(
            id="S13",
            family="mixed2d",
            problem="polynomial",
            n=16,
            point=[4, 6],
            kinds=["neumann"] * 4,
            methods=["lod"],
        ),
        dict(
            id="H01",
            family="heat",
            problem="decay",
            n=16,
            dt=0.001,
            T=0.01,
            methods=["lod", "mc"],
        ),
        dict(
            id="H02",
            family="heat",
            problem="manufactured",
            n=12,
            dt=0.013,
            T=0.04,
            point=[4, 7],
            methods=["lod", "mc"],
        ),
        dict(
            id="Q01",
            family="nonlinear",
            alpha=0.5,
            n=8,
            dt=0.01,
            T=0.1,
            methods=["lod", "mc"],
        ),
        dict(
            id="Q02",
            family="nonlinear",
            alpha=2.0,
            n=8,
            dt=0.01,
            T=0.1,
            methods=["lod", "mc"],
        ),
    ]


def seed_for(*parts):
    return int(np.random.SeedSequence([SEED, *parts]).generate_state(1)[0])


def prepare(case):
    family, n = case["family"], case["n"]
    point = case.get("point", [n // 2] * (3 if family in ("3d", "mixed3d") else 2))
    if family == "2d":
        problem = make_problem(case["problem"], n)
        points = select_points(n, case["mode"], point)
        grid = reference_solution(problem)
        reference = choose_reference(problem, points, grid)
        target = reference.values[tuple(points.T)]
        warmup()
    elif family == "3d":
        from pdelab.problem3d import reference_solution3d

        problem = make_problem3d(case["problem"], n)
        points = select_points3d(n, point=point)
        grid = reference_solution3d(problem)
        target = problem.exact[tuple(points.T)]
        warmup3d()
    elif family in ("mixed2d", "mixed3d"):
        is3d = family == "mixed3d"
        cls = BoundaryConfig3D if is3d else BoundaryConfig
        config = cls(n=n, problem=case["problem"], point=tuple(point))
        if "kinds" in case:
            config.kinds = tuple(case["kinds"])
        problem = (problem_from_config3d if is3d else problem_from_config)(config)
        operator = (BoundaryOperator3D if is3d else BoundaryOperator)(problem)
        operator.check_compatibility()
        points = (select_boundary_points3d if is3d else select_boundary_points)(
            n, point=point
        )
        grid = operator.solve()
        target = operator.normalize(problem.exact)[tuple(points.T)]
        (warmup_boundary3d if is3d else warmup_boundary)()
    elif family == "heat":
        problem = make_heat_problem(case["problem"], n)
        points = np.array([point[::-1]])
        grid = solve_heat(problem, case["dt"], case["T"], "direct").fields[-1]
        target = problem.evaluate(problem.exact, case["T"])[tuple(points.T)]
    else:
        problem = make_nonlinear_problem(n, case["alpha"])
        points = np.array([point[::-1]])
        reference = solve_nonlinear(
            problem, case["dt"], case["T"], "newton", NonlinearSettings(tolerance=1e-11)
        )
        if not reference.completed:
            raise RuntimeError("Newton reference failed")
        grid = reference.fields[-1]
        target = problem.evaluate(problem.exact, case["T"])[tuple(points.T)]
    return problem, points, np.asarray(target), grid[tuple(points.T)]


def candidates(case, method):
    if method == "lod":
        return [dict(tol=t) for t in TOLS]
    if case["family"] == "nonlinear":
        return [
            dict(tol=t, walks=n)
            for t in (0.01, 0.001, 0.0001)
            for n in (32, 128, 512, 2048)
        ]
    return [dict(walks=n) for n in BUDGETS if n <= case.get("max_walks", 16384)]


def run(case, prepared, method, params, seed):
    problem, points, target, grid = prepared
    family = case["family"]
    tol, walks = params.get("tol", 1e-8), params.get("walks", 64)
    if family in ("2d", "3d", "mixed2d", "mixed3d"):
        solver = {
            "2d": solve,
            "3d": solve3d,
            "mixed2d": solve_boundary,
            "mixed3d": solve_boundary3d,
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
        values, elapsed = result.values[tuple(points.T)], result.elapsed
        stats = dict(
            steps=result.steps, walks=result.walks, iterations=result.iterations
        )
    elif family == "heat":
        if method == "lod":
            result = solve_heat(problem, case["dt"], case["T"], "lod", tol)
            values = result.fields[-1][tuple(points.T)]
            stats = dict(iterations=int(result.iterations.sum()))
        else:
            result = solve_heat_mc(problem, case["dt"], case["T"], points, walks, seed)
            values = result.values
            stats = dict(
                steps=result.spatial_transitions + result.temporal_transitions,
                walks=walks,
            )
        elapsed = result.seconds
    else:
        settings = NonlinearSettings(
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
        result = solve_nonlinear(
            problem,
            case["dt"],
            case["T"],
            "picard" if method == "lod" else "mc",
            settings,
        )
        if not result.completed:
            raise RuntimeError(result.reason)
        values, elapsed = result.fields[-1][tuple(points.T)], result.seconds
        stats = dict(
            walks=result.walks,
            steps=result.steps,
            iterations=sum(len(layer.history) for layer in result.layers),
        )
    scale = float(np.mean(target**2))
    if not scale:
        raise ValueError("zero_reference")
    return dict(
        seconds=elapsed,
        squared_error=float(np.mean((values - target) ** 2) / scale * 10000),
        value_mean=float(np.mean(values)),
        grid_rmse=float(np.sqrt(np.mean((values - grid) ** 2))),
        **stats,
    )


def bootstrap_rms(values, seed):
    rng = np.random.default_rng(seed)
    estimates = np.sqrt(
        np.mean(rng.choice(values, (5000, len(values)), replace=True), axis=1)
    )
    return np.quantile(estimates, [0.025, 0.975]).tolist()


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, list(dict.fromkeys(k for r in rows for k in r)), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = dict(
        revision=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        seed=SEED,
        pilot=PILOT,
        validation=VALIDATION,
        targets=TARGETS,
        tolerances=TOLS,
        budgets=BUDGETS,
        cases=cases(),
        selection="lowest pilot mean time among candidates passing pilot RMS; exactly one independent final series per selected candidate",
        timing="full solver wall time after JIT; common input sampling and independent reference excluded; all method-specific preparation included",
        python=platform.python_version(),
        platform=platform.platform(),
        versions={k: version(k) for k in ["numpy", "scipy", "numba"]},
        code_sha256={
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(Path("pdelab").glob("*.py"))
        },
    )
    (OUT / "protocol.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2) + "\n"
    )
    all_rows, comparisons = [], []
    for case_index, case in enumerate(cases()):
        checkpoint = OUT / (case["id"] + ".json")
        if checkpoint.exists():
            saved = json.loads(checkpoint.read_text())
            if saved["case"] != case:
                raise RuntimeError("Existing checkpoint has different protocol")
            all_rows.extend(saved["trials"])
            comparisons.extend(saved["comparisons"])
            print(case["id"], "loaded", flush=True)
            continue
        print(case["id"], case, flush=True)
        prepared = prepare(case)
        trials, summaries, selected = [], {}, {}
        # Untimed compilation of every method signature before pilot measurements.
        for mi, method in enumerate(case["methods"]):
            run(
                case,
                prepared,
                method,
                candidates(case, method)[0],
                seed_for(case_index, mi, 99),
            )
        jobs = [
            (mi, method, ci, params)
            for mi, method in enumerate(case["methods"])
            for ci, params in enumerate(candidates(case, method))
        ]
        for repeat in range(PILOT):
            order = np.random.default_rng(
                seed_for(case_index, 0, repeat, 9)
            ).permutation(len(jobs))
            for j in order:
                mi, method, ci, params = jobs[j]
                row = dict(
                    case=case["id"],
                    method=method,
                    candidate=ci,
                    phase="pilot",
                    repeat=repeat,
                    seed=seed_for(case_index, mi, ci, 0, repeat),
                    parameters=params,
                )
                started = perf_counter()
                try:
                    row.update(
                        run(case, prepared, method, params, row["seed"]), status="ok"
                    )
                except (ValueError, RuntimeError) as exc:
                    row.update(status=str(exc), seconds=perf_counter() - started)
                trials.append(row)
        for mi, method in enumerate(case["methods"]):
            for target in TARGETS:
                eligible = []
                for ci, params in enumerate(candidates(case, method)):
                    group = [
                        r
                        for r in trials
                        if r["method"] == method and r["candidate"] == ci
                    ]
                    if all(r["status"] == "ok" for r in group):
                        error = float(
                            np.sqrt(np.mean([r["squared_error"] for r in group]))
                        )
                        elapsed = float(np.mean([r["seconds"] for r in group]))
                        if error <= target:
                            eligible.append((elapsed, ci, params, error))
                selected[method, target] = min(eligible) if eligible else None
        validation_jobs = list(
            dict.fromkeys(
                (method, entry[1])
                for (method, target), entry in selected.items()
                if entry
            )
        )
        for repeat in range(VALIDATION):
            order = np.random.default_rng(
                seed_for(case_index, 1, repeat, 9)
            ).permutation(len(validation_jobs))
            for j in order:
                method, ci = validation_jobs[j]
                mi = case["methods"].index(method)
                params = candidates(case, method)[ci]
                row = dict(
                    case=case["id"],
                    method=method,
                    candidate=ci,
                    phase="validation",
                    repeat=repeat,
                    seed=seed_for(case_index, mi, ci, 1, repeat),
                    parameters=params,
                )
                started = perf_counter()
                try:
                    row.update(
                        run(case, prepared, method, params, row["seed"]), status="ok"
                    )
                except (ValueError, RuntimeError) as exc:
                    row.update(status=str(exc), seconds=perf_counter() - started)
                trials.append(row)
        for (method, target), entry in selected.items():
            row = dict(
                case=case["id"],
                method=method,
                target_pct=target,
                status="not_selected",
                seconds=None,
                error_pct=None,
                error_ci_low=None,
                error_ci_high=None,
                parameters=entry[2] if entry else None,
                pilot_error=entry[3] if entry else None,
            )
            if entry:
                group = [
                    r
                    for r in trials
                    if r["method"] == method
                    and r["candidate"] == entry[1]
                    and r["phase"] == "validation"
                ]
                if all(r["status"] == "ok" for r in group):
                    errors = np.array([r["squared_error"] for r in group])
                    error = float(np.sqrt(errors.mean()))
                    ci = bootstrap_rms(
                        errors, seed_for(case_index, case["methods"].index(method), 2)
                    )
                    row.update(
                        error_pct=error,
                        error_ci_low=ci[0],
                        error_ci_high=ci[1],
                        status="confirmed" if error <= target else "validation_failed",
                        seconds=(
                            float(np.mean([r["seconds"] for r in group]))
                            if error <= target
                            else None
                        ),
                    )
                else:
                    row["status"] = "solver_failed"
            summaries[method, target] = row
        for (method, target), row in summaries.items():
            baseline = summaries["lod", target]
            row.update(
                speedup=None, speed_ci_low=None, speed_ci_high=None, robust_win=False
            )
            if method != "lod" and row["status"] == baseline["status"] == "confirmed":
                a = [
                    r["seconds"]
                    for r in trials
                    if r["method"] == "lod"
                    and r["candidate"] == selected["lod", target][1]
                    and r["phase"] == "validation"
                ]
                b = [
                    r["seconds"]
                    for r in trials
                    if r["method"] == method
                    and r["candidate"] == selected[method, target][1]
                    and r["phase"] == "validation"
                ]
                rng = np.random.default_rng(
                    seed_for(case_index, case["methods"].index(method), 3)
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
                    ),
                )
        saved = dict(
            case=case,
            reference_values=prepared[2].tolist(),
            grid_values=prepared[3].tolist(),
            trials=trials,
            comparisons=list(summaries.values()),
        )
        checkpoint.write_text(
            json.dumps(saved, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        )
        all_rows.extend(trials)
        comparisons.extend(summaries.values())
        write_csv(OUT / "comparisons.csv", comparisons)
        write_csv(OUT / "trials.csv", all_rows)
        print(case["id"], "complete", len(trials), "runs", flush=True)
    print("DONE", len(all_rows), "runs", len(comparisons), "threshold rows", flush=True)


if __name__ == "__main__":
    main()
