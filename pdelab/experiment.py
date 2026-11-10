import os
import platform
from dataclasses import asdict, dataclass
from time import perf_counter

import numpy as np

from .problem import make_problem, reference_solution, select_points
from .reference import REFERENCE_POLICY, choose_reference
from .solvers import LOD_STOPPING_RULE, METHODS, solve, trace_walk, warmup

METRICS_POLICY = "control-mc-relative-v3"
LOD_TOLERANCES = (1e-2, 1e-4, 1e-6, 1e-8)


@dataclass
class Config:
    problem: str = "hot_top"
    n: int = 24
    mode: str = "point"
    point: tuple[int, int] | None = None  # public coordinate order: (x,y)
    walks: int = 2048  # per fresh start point, not guaranteed equal precision
    batches: int = 8
    seed: int = 42
    tol: float = 1e-7  # Relative infinity-norm residual, including Dirichlet RHS.
    max_iter: int = 20000
    top: float = 100.0
    bottom: float = 0.0
    left: float = 0.0
    right: float = 0.0
    source: float = 0.0
    coarse_n: int | None = None


@dataclass
class Experiment:
    config: Config
    problem: object
    points: np.ndarray
    reference: (
        np.ndarray
    )  # Direct grid diagnostic; primary target is comparison.values.
    results: dict
    metrics: list
    trace: dict
    metadata: dict

    @property
    def comparison(self):
        return choose_reference(self.problem, self.points, self.reference)


def metrics(result, problem, points, ref):
    comparison = choose_reference(problem, points, ref)
    y, x = points.T
    values = result.values[y, x]
    grid_error = values - ref[y, x]
    reference_values = comparison.values[y, x]
    primary_error = values - reference_values
    rmse = float(np.sqrt(np.mean(primary_error**2)))
    reference_rms = float(np.sqrt(np.mean(reference_values**2)))
    relative_error = 100 * rmse / reference_rms if reference_rms > 0 else None
    row = {
        "method": result.method,
        "name": METHODS[result.method],
        "seconds": result.elapsed,
        "preparation_seconds": result.extras.get("preparation_seconds", 0.0),
        "path_seconds": result.extras.get("path_seconds"),
        "coarse_n": result.extras.get("coarse_n"),
        "mean_path_variance": (
            float(np.nanmean(result.stderr[y, x] ** 2 * result.sample_count[y, x]))
            if result.method != "lod_serial"
            else None
        ),
        "query_mean": float(values.mean()),
        "reference_mean": float(reference_values.mean()),
        "reference_kind": comparison.kind,
        "reference_label": comparison.label,
        "rmse": rmse,
        "relative_error_pct": relative_error,
        "rmse_grid": float(np.sqrt(np.mean(grid_error**2))),
        "rmse_exact": (
            None
            if comparison.kind == "grid"
            else float(np.sqrt(np.mean(primary_error**2)))
        ),
        "walks": int(result.walks),
        "steps": int(result.steps),
        "mean_path_steps": result.steps / result.walks if result.walks else None,
        "iterations": int(result.iterations),
        "residual": result.residual,
        "converged": result.converged,
        "relative_residual": getattr(result, "relative_residual", None),
        "residual_scale": result.extras.get("residual_scale"),
        "stopping_rule": (
            result.extras.get("stopping_rule")
            if result.method == "lod_serial"
            else "fixed_walk_budget"
        ),
        "min_observations": int(result.sample_count[y, x].min()),
        "max_observations": int(result.sample_count[y, x].max()),
    }
    return row


def refresh_comparison(exp):
    """Upgrade already computed UI results without rerunning any solver."""
    results = {
        method: result for method, result in exp.results.items() if method in METHODS
    }
    refreshed = Experiment(
        Config(
            **{
                key: value
                for key, value in asdict(exp.config).items()
                if key in Config.__dataclass_fields__
            }
        ),
        exp.problem,
        exp.points,
        exp.reference,
        results,
        [
            metrics(result, exp.problem, exp.points, exp.reference)
            for result in results.values()
        ],
        exp.trace,
        dict(exp.metadata),
    )
    comparison = refreshed.comparison
    refreshed.metadata.update(
        reference_policy=REFERENCE_POLICY,
        reference_kind=comparison.kind,
        reference_label=comparison.label,
        reference_description=comparison.description,
        field_reference_label=comparison.field_label,
        metrics_policy=METRICS_POLICY,
    )
    refreshed.metadata = {
        key: value for key, value in refreshed.metadata.items() if key != "workers"
    }
    return refreshed


def run_experiment(cfg=None, methods=None, progress=None, record=True):
    cfg = cfg or Config()
    methods = list(METHODS) if methods is None else list(methods)
    if (
        not methods
        or len(set(methods)) != len(methods)
        or any(method not in METHODS for method in methods)
    ):
        raise ValueError("Нужен непустой список различных известных методов")
    problem = make_problem(
        cfg.problem, cfg.n, cfg.top, cfg.bottom, cfg.left, cfg.right, cfg.source
    )
    points = select_points(cfg.n, cfg.mode, cfg.point)
    start = perf_counter()
    warmup()
    warm_seconds = perf_counter() - start
    start = perf_counter()
    grid_reference = reference_solution(problem)
    reference_seconds = perf_counter() - start
    results = {}
    rows = []
    for method_index, method in enumerate(methods):
        if progress:
            progress(method_index / len(methods), METHODS[method])
        result = solve(
            problem,
            points,
            method,
            walks=cfg.walks,
            seed=cfg.seed,
            tol=cfg.tol,
            max_iter=cfg.max_iter,
            batches=cfg.batches,
            coarse_n=cfg.coarse_n,
            record=record,
        )
        if not result.converged:
            raise RuntimeError(
                f"{METHODS[method]} не сошёлся: относительная невязка {result.relative_residual:g} > {cfg.tol:g}. Увеличьте max_iter или ослабьте tol."
            )
        results[method] = result
        rows.append(metrics(result, problem, points, grid_reference))
    if progress:
        progress(1.0, "Готово")
    return refresh_comparison(
        Experiment(
            cfg,
            problem,
            points,
            grid_reference,
            results,
            rows,
            trace_walk(problem, tuple(points[0]), cfg.seed),
            {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "cpu_count": os.cpu_count(),
                "jit_warmup_seconds": warm_seconds,
                "reference_seconds": reference_seconds,
                "lod_stopping_rule": LOD_STOPPING_RULE,
                "timing": "Solver wall time after JIT warm-up; reference and plots excluded. Layers include all preparation and independent batches.",
                "units": "Условная температура; x,y — единичный квадрат",
                "default_query": "one center point",
            },
        )
    )


def benchmark(
    cfg, budgets=(256, 1024, 4096), repeats=3, tolerances=LOD_TOLERANCES, progress=None
):
    if repeats < 2:
        raise ValueError("Для сравнения нужны минимум два повторения")
    if not budgets or any(n < 2 * cfg.batches for n in budgets):
        raise ValueError("Бюджеты должны быть >= 2*batches")
    if not tolerances or any(not np.isfinite(t) or not 0 < t < 1 for t in tolerances):
        raise ValueError("Относительные допуски ЛОМ должны быть между 0 и 1")
    warmup()
    problem = make_problem(
        cfg.problem, cfg.n, cfg.top, cfg.bottom, cfg.left, cfg.right, cfg.source
    )
    points = select_points(cfg.n, cfg.mode, cfg.point)
    grid_reference = reference_solution(problem)
    rows = []
    schedules = [
        (method, v)
        for method in METHODS
        for v in (tolerances if method.startswith("lod_") else budgets)
    ]
    for repeat_index in range(repeats):
        # Alternate order, so always being first is not tied to one method.
        schedule = schedules if repeat_index % 2 == 0 else schedules[::-1]
        for method, budget in schedule:
            if progress:
                progress(
                    len(rows) / (repeats * len(schedules)),
                    f"{METHODS[method]} · повтор {repeat_index+1}",
                )
            is_lod = method.startswith("lod_")
            result = solve(
                problem,
                points,
                method,
                walks=cfg.walks if is_lod else int(budget),
                seed=(cfg.seed + 1009 * repeat_index) % 2**32,
                tol=float(budget) if is_lod else cfg.tol,
                max_iter=cfg.max_iter,
                batches=cfg.batches,
                coarse_n=cfg.coarse_n,
                record=False,
            )
            if not result.converged:
                raise RuntimeError(f"Несходимость {method}")
            row = metrics(result, problem, points, grid_reference)
            row.update(
                repeat=repeat_index,
                budget=float(budget),
                budget_kind=(
                    "relative_residual_tolerance" if is_lod else "walks_per_start"
                ),
            )
            rows.append(row)
    return rows
