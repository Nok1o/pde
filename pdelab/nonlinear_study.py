"""Independent full-chain ensembles, held-out targets and separated errors."""

from dataclasses import asdict, dataclass, replace

import numpy as np
from scipy.integrate import solve_ivp
from scipy.stats import t as student_t

from .nonlinear import (
    NonlinearSettings,
    make_nonlinear_problem,
    newton_system,
    solve_nonlinear,
    solve_stationary_nonlinear,
)


def child_seed(seed, *indices):
    return int(np.random.SeedSequence([int(seed), *indices]).generate_state(1)[0])


def validate_query(n, point):
    point = (n // 2, n // 2) if point is None else tuple(point)
    if len(point) != 2 or any(
        isinstance(i, (bool, np.bool_))
        or not np.isfinite(i)
        or int(i) != i
        or not 0 <= i <= n
        for i in point
    ):
        raise ValueError("Нужны индексы (x,y) от 0 до n")
    return tuple(int(i) for i in point)


def ensemble_summary(results, grid, exact, point, alpha):
    x, y = point
    complete = all(result.completed for result in results)
    row = dict(
        method="mc",
        completed=complete,
        completed_runs=sum(r.completed for r in results),
        repeats=len(results),
        seconds=float(np.mean([r.seconds for r in results])),
        total_seconds=float(sum(r.seconds for r in results)),
        walks=sum(r.walks for r in results),
        steps=sum(r.steps for r in results),
        workspace_bytes=max(r.workspace_bytes for r in results),
        value=None,
        rmse=None,
        relative_error_pct=None,
        rmse_grid=None,
        grid_bias=None,
        bias_ci_low=None,
        bias_ci_high=None,
        chain_std=None,
        conductivity_plugin_bias=None,
    )
    if not complete:
        return row
    samples = np.array([r.fields[-1, y, x] for r in results])
    errors = samples - grid[y, x]
    bias = float(np.mean(errors))
    deviation = float(np.std(samples, ddof=1))
    half_width = float(
        student_t.ppf(0.975, len(samples) - 1) * deviation / np.sqrt(len(samples))
    )
    target = grid[y, x] if exact is None else exact[y, x]
    rmse = float(np.sqrt(np.mean((samples - target) ** 2)))
    row.update(
        value=float(np.mean(samples)),
        rmse=rmse,
        relative_error_pct=100 * rmse / abs(target) if target else None,
        rmse_grid=float(np.sqrt(np.mean(errors**2))),
        grid_bias=bias,
        bias_ci_low=bias - half_width,
        bias_ci_high=bias + half_width,
        chain_std=deviation,
        conductivity_plugin_bias=alpha * deviation**2,
    )
    return row


@dataclass
class NonlinearComparison:
    point: tuple
    reference: object
    deterministic: object
    stochastic: list
    metrics: list
    matched_picard: object


def compare_nonlinear(
    problem, dt=0.01, T=0.1, point=None, repeats=4, deterministic=None, stochastic=None
):
    if isinstance(repeats, bool) or int(repeats) != repeats or not 2 <= repeats <= 100:
        raise ValueError("Нужно от 2 до 100 независимых полных повторов")
    point = validate_query(problem.n, point)
    deterministic = NonlinearSettings() if deterministic is None else deterministic
    stochastic = (
        NonlinearSettings(
            tolerance=0.001, change_tolerance=0.001, relaxation=0.7, max_outer=12
        )
        if stochastic is None
        else stochastic
    )
    from .heat import time_grid

    times, _ = time_grid(dt, T)
    if len(times) * (problem.n + 1) ** 2 * (repeats + 3) > 8_000_000:
        raise ValueError(
            "Уменьшите число узлов, слоёв или повторов: ансамбль ограничен 8 млн значений"
        )
    reference = solve_nonlinear(
        problem, dt, T, "newton", replace(deterministic, tolerance=1e-11, max_outer=50)
    )
    if not reference.completed:
        raise RuntimeError(f"Независимый эталон не сошёлся: {reference.reason}")
    lod = solve_nonlinear(problem, dt, T, "picard", deterministic)
    matched = solve_nonlinear(
        problem,
        dt,
        T,
        "picard",
        replace(stochastic, linear_tolerance=deterministic.linear_tolerance),
    )
    runs = [
        solve_nonlinear(
            problem,
            dt,
            T,
            "mc",
            replace(stochastic, seed=child_seed(stochastic.seed, 100, repeat)),
        )
        for repeat in range(int(repeats))
    ]
    exact = problem.evaluate(problem.exact, T) if problem.exact is not None else None
    x, y = point
    target = reference.fields[-1, y, x] if exact is None else exact[y, x]
    error = abs(lod.fields[-1, y, x] - target) if lod.completed else None
    deterministic_row = dict(
        method="picard",
        completed=lod.completed,
        completed_runs=int(lod.completed),
        repeats=1,
        seconds=lod.seconds,
        total_seconds=lod.seconds,
        walks=0,
        steps=0,
        workspace_bytes=lod.workspace_bytes,
        value=float(lod.fields[-1, y, x]) if lod.completed else None,
        rmse=error,
        relative_error_pct=(
            100 * error / abs(target) if lod.completed and target else None
        ),
        rmse_grid=(
            abs(float(lod.fields[-1, y, x] - reference.fields[-1, y, x]))
            if lod.completed
            else None
        ),
    )
    stochastic_row = ensemble_summary(
        runs, reference.fields[-1], exact, point, problem.alpha
    )
    stochastic_row["matched_picard_value"] = (
        float(matched.fields[-1, y, x]) if matched.completed else None
    )
    stochastic_row["matched_picard_seconds"] = matched.seconds
    stochastic_row["matched_picard_relative_error_pct"] = (
        100 * abs(matched.fields[-1, y, x] - target) / abs(target)
        if matched.completed and target
        else None
    )
    stochastic_row["bias_vs_matched_picard"] = (
        stochastic_row["value"] - stochastic_row["matched_picard_value"]
        if stochastic_row["completed"] and matched.completed
        else None
    )
    return NonlinearComparison(
        point,
        reference,
        lod,
        runs,
        [
            deterministic_row,
            stochastic_row,
        ],
        matched,
    )


def nonlinear_accuracy_study():
    """Spatial control, temporal error vs method-of-lines, outer iteration error."""
    rows = []
    for alpha in (0.0, 0.5, 2.0):
        previous = None
        for n in (8, 16, 32):
            problem = make_nonlinear_problem(n, alpha, stationary=True)
            result = solve_stationary_nonlinear(
                n, alpha, "newton", NonlinearSettings(tolerance=1e-11)
            )
            if not result.converged:
                raise RuntimeError(result.reason)
            error = float(np.max(np.abs(result.values - problem.initial)))
            rows.append(
                dict(
                    study="space_stationary",
                    alpha=alpha,
                    n=n,
                    dt=None,
                    T=None,
                    tolerance=1e-11,
                    error=error,
                    order=(
                        None if previous is None else float(np.log2(previous / error))
                    ),
                    reference="analytic_stationary",
                )
            )
            previous = error
        problem = make_nonlinear_problem(8, alpha)
        n, T = problem.n, 0.08

        def system(t, vector):
            values = problem.evaluate(problem.boundary, t)
            values[1:-1, 1:-1] = vector.reshape(n - 1, n - 1)
            return newton_system(values, alpha, problem.evaluate(problem.source, t), 0)

        continuous_time = solve_ivp(
            lambda t, u: -system(t, u)[0].ravel(),
            (0, T),
            problem.initial[1:-1, 1:-1].ravel(),
            method="Radau",
            jac=lambda t, u: -system(t, u)[1],
            rtol=1e-10,
            atol=1e-12,
        )
        if not continuous_time.success:
            raise RuntimeError(continuous_time.message)
        previous = None
        for dt in (0.005, 0.0025, 0.00125):
            result = solve_nonlinear(
                problem, dt, T, "newton", NonlinearSettings(tolerance=1e-11)
            )
            if not result.completed:
                raise RuntimeError(result.reason)
            error = float(
                np.max(
                    np.abs(
                        result.fields[-1, 1:-1, 1:-1].ravel() - continuous_time.y[:, -1]
                    )
                )
            )
            rows.append(
                dict(
                    study="time",
                    alpha=alpha,
                    n=n,
                    dt=dt,
                    T=T,
                    tolerance=1e-11,
                    error=error,
                    order=(
                        None if previous is None else float(np.log2(previous / error))
                    ),
                    reference="semidiscrete_Radau",
                )
            )
            previous = error
        direct = solve_nonlinear(
            problem, 0.02, T, "newton", NonlinearSettings(tolerance=1e-11)
        )
        for tolerance in (1e-3, 1e-6, 1e-9):
            result = solve_nonlinear(
                problem,
                0.02,
                T,
                settings=NonlinearSettings(
                    tolerance=tolerance,
                    change_tolerance=tolerance,
                    linear_tolerance=1e-12,
                    relaxation=0.7,
                ),
            )
            rows.append(
                dict(
                    study="nonlinear_iteration",
                    alpha=alpha,
                    n=n,
                    dt=0.02,
                    T=T,
                    tolerance=tolerance,
                    error=(
                        float(np.max(np.abs(result.fields - direct.fields)))
                        if result.completed
                        else None
                    ),
                    order=None,
                    reference="Newton_same_grid_steps",
                )
            )
    return rows


def nonlinear_mc_study(repeats=4, seed=42):
    """A fixed small factorial set plus budget-policy/relaxation controls."""
    rows, runs = [], []
    cases = [
        (alpha, n, dt, budget, "geometric", 0.7, 0.001, 12)
        for alpha in (0.0, 0.5, 2.0)
        for n in (4, 8)
        for dt in (0.02, 0.01)
        for budget in (32, 128)
    ]
    cases += [
        (0.5, 8, 0.02, 128, schedule, relaxation, 0.001, 12)
        for schedule in ("constant", "linear", "geometric")
        for relaxation in (0.5, 1.0)
    ]
    cases.append((2.0, 8, 0.02, 8, "constant", 1.0, 1e-10, 1))
    for index, (
        alpha,
        n,
        dt,
        budget,
        schedule,
        relaxation,
        tolerance,
        max_outer,
    ) in enumerate(cases):
        settings = NonlinearSettings(
            tolerance=tolerance,
            change_tolerance=tolerance,
            walks=budget,
            max_walks=budget * 4,
            schedule=schedule,
            relaxation=relaxation,
            max_outer=max_outer,
            seed=child_seed(seed, 200, index),
        )
        comparison = compare_nonlinear(
            make_nonlinear_problem(n, alpha),
            dt,
            0.04,
            repeats=repeats,
            stochastic=settings,
        )
        parameters = dict(
            case=index,
            alpha=alpha,
            n=n,
            dt=dt,
            T=0.04,
            initial_walks=budget,
            max_walks=budget * 4,
            schedule=schedule,
            relaxation=relaxation,
            tolerance=tolerance,
            max_outer=max_outer,
        )
        for metric in comparison.metrics:
            actual = (
                comparison.deterministic.settings
                if metric["method"] == "picard"
                else settings
            )
            rows.append(
                {
                    **parameters,
                    **asdict(actual),
                    "initial_walks": actual.walks if metric["method"] == "mc" else None,
                    "max_walks": actual.max_walks if metric["method"] == "mc" else None,
                    "schedule": actual.schedule if metric["method"] == "mc" else None,
                    **metric,
                }
            )
        target = comparison.reference.fields[-1, n // 2, n // 2]
        for repeat, result in enumerate(comparison.stochastic):
            runs.append(
                {
                    **parameters,
                    "repeat": repeat,
                    "settings": asdict(result.settings),
                    "seed": result.settings.seed,
                    "completed": result.completed,
                    "reason": result.reason,
                    "seconds": result.seconds,
                    "walks": result.walks,
                    "steps": result.steps,
                    "grid_error": (
                        float(result.fields[-1, n // 2, n // 2] - target)
                        if result.completed
                        else None
                    ),
                    "layers": [
                        dict(reason=layer.reason, history=layer.history)
                        for layer in result.layers
                    ],
                }
            )
    return rows, runs


def nonlinear_targets(
    problem,
    dt=0.01,
    T=0.1,
    point=None,
    repeats=4,
    budgets=(32, 128, 512),
    seed=42,
    *,
    settings=None,
    deterministic=None,
):
    """Select on pilot chains, confirm each selected budget on fresh full chains."""
    if not budgets or any(isinstance(n, bool) or int(n) != n or n < 2 for n in budgets):
        raise ValueError("Нужен хотя бы один целый бюджет MC ≥2")
    pilots = []
    base = (
        NonlinearSettings(
            tolerance=0.001, change_tolerance=0.001, relaxation=0.7, max_outer=12
        )
        if settings is None
        else settings
    )
    if any(budget > base.max_walks for budget in budgets):
        raise ValueError("Начальный бюджет исследования не должен превышать max_walks")
    for index, budget in enumerate(budgets):
        candidate = replace(
            base,
            walks=int(budget),
            max_walks=base.max_walks,
            seed=child_seed(seed, 300, index),
        )
        comparison = compare_nonlinear(
            problem, dt, T, point, repeats, deterministic, candidate
        )
        pilots.append((candidate, comparison))
    rows = []
    validations = {}
    deterministic_runs = [pilots[0][1].deterministic]
    deterministic_runs.extend(
        solve_nonlinear(problem, dt, T, "picard", deterministic)
        for _ in range(int(repeats) - 1)
    )
    deterministic_seconds = float(np.mean([run.seconds for run in deterministic_runs]))
    for target in (1.0, 5.0, 10.0):
        lod = pilots[0][1].metrics[0]
        lod_error = lod["relative_error_pct"]
        lod_pass = (
            all(run.completed for run in deterministic_runs)
            and lod_error is not None
            and lod_error <= target
        )
        rows.append(
            dict(
                target_pct=target,
                method="picard",
                status="confirmed" if lod_pass else "not_reached",
                seconds=(deterministic_seconds if lod_pass else None),
                relative_error_pct=lod_error,
                walks=None,
            )
        )
        eligible = [
            (i, c.metrics[1])
            for i, (_, c) in enumerate(pilots)
            if c.metrics[1]["completed"]
            and c.metrics[1]["relative_error_pct"] is not None
            and c.metrics[1]["relative_error_pct"] <= target
        ]
        if not eligible:
            rows.append(
                dict(
                    target_pct=target,
                    method="mc",
                    status="not_reached",
                    seconds=None,
                    relative_error_pct=None,
                    walks=None,
                )
            )
            continue
        chosen = min(eligible, key=lambda item: item[1]["seconds"])[0]
        if chosen not in validations:
            settings = replace(pilots[chosen][0], seed=child_seed(seed, 301, chosen))
            validations[chosen] = compare_nonlinear(
                problem, dt, T, point, repeats, deterministic, settings
            )
        metric = validations[chosen].metrics[1]
        passed = (
            metric["completed"]
            and metric["relative_error_pct"] is not None
            and metric["relative_error_pct"] <= target
        )
        rows.append(
            dict(
                target_pct=target,
                method="mc",
                status="confirmed" if passed else "validation_failed",
                seconds=metric["seconds"] if passed else None,
                relative_error_pct=metric["relative_error_pct"],
                walks=pilots[chosen][0].walks,
            )
        )
    evidence = dict(
        deterministic_runs=[
            dict(seconds=run.seconds, completed=run.completed, reason=run.reason)
            for run in deterministic_runs
        ],
        pilots=[
            dict(
                settings=asdict(s),
                metrics=c.metrics,
                seeds=[r.settings.seed for r in c.stochastic],
            )
            for s, c in pilots
        ],
        validations=[
            dict(
                candidate=i,
                metrics=c.metrics,
                seeds=[r.settings.seed for r in c.stochastic],
            )
            for i, c in validations.items()
        ],
    )
    return rows, evidence
