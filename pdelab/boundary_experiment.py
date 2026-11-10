"""Experiments and portable exports for mixed boundary conditions."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .boundary import (
    SIDES,
    BoundaryCondition,
    BoundaryOperator,
    BoundaryProblem,
    make_boundary_problem,
)
from .boundary_solvers import (
    METHODS_BOUNDARY,
    select_boundary_points,
    solve_boundary,
    warmup_boundary,
)
from .experiment3d import rows_csv


@dataclass
class BoundaryConfig:
    problem: str = "polynomial"
    n: int = 24
    kinds: tuple = ("robin", "dirichlet", "dirichlet", "dirichlet")
    kappas: tuple = (1.0, 1.0, 1.0, 1.0)
    data: tuple = (0.0, 0.0, 0.0, 0.0)
    source: float = 0.0
    mode: str = "point"
    point: tuple | None = None
    walks: int = 2048
    seed: int = 42
    tol: float = 1e-8
    max_iter: int = 20000
    max_steps: int | None = None


def problem_from_config(config):
    if any(len(a) != 4 for a in (config.kinds, config.kappas, config.data)):
        raise ValueError("Нужно по четыре параметра сторон: left, right, bottom, top")
    if config.problem == "custom":
        sides = {
            side: BoundaryCondition(kind, value, kappa if kind == "robin" else 0)
            for side, kind, value, kappa in zip(
                SIDES, config.kinds, config.data, config.kappas
            )
        }
        return BoundaryProblem(config.n, config.source, sides, name="Свои данные")
    return make_boundary_problem(
        config.problem,
        config.n,
        dict(zip(SIDES, config.kinds)),
        dict(zip(SIDES, config.kappas)),
    )


@dataclass
class BoundaryExperiment:
    config: BoundaryConfig
    problem: BoundaryProblem
    points: np.ndarray
    reference: np.ndarray
    reference_kind: str
    grid_reference: np.ndarray
    results: dict
    metrics: list


def boundary_metrics(result, points, reference, grid_reference):
    indices = tuple(points.T)
    target = reference[indices]
    value = result.values[indices]
    rmse = float(np.sqrt(np.mean((value - target) ** 2)))
    scale = float(np.sqrt(np.mean(target**2)))
    return {
        "method": result.method,
        "name": METHODS_BOUNDARY[result.method],
        "value": float(np.mean(value)),
        "reference_value": float(np.mean(target)),
        "seconds": result.elapsed,
        "rmse": rmse,
        "relative_error_pct": 100 * rmse / scale if scale > 0 else None,
        "rmse_grid": float(np.sqrt(np.mean((value - grid_reference[indices]) ** 2))),
        "iterations": result.iterations,
        "walks": result.walks,
        "steps": result.steps,
        "relative_residual": result.relative_residual,
        "converged": result.converged,
    }


def run_boundary_experiment(config=None, methods=None):
    config = BoundaryConfig() if config is None else config
    problem = problem_from_config(config)
    operator = BoundaryOperator(problem)
    operator.check_compatibility()
    if methods is None:
        methods = ["lod_serial"] if operator.pure_neumann else list(METHODS_BOUNDARY)
    if (
        not methods
        or len(set(methods)) != len(methods)
        or any(m not in METHODS_BOUNDARY for m in methods)
    ):
        raise ValueError("Выберите ЛОМ и/или обычный MC")
    if operator.pure_neumann and "mc" in methods:
        raise ValueError("MC для чистого Неймана пока не поддерживается")
    points = select_boundary_points(problem.n, config.mode, config.point)
    grid = operator.solve()
    reference = grid if problem.exact is None else operator.normalize(problem.exact)
    warmup_boundary()
    results = {}
    for method in methods:
        result = solve_boundary(
            problem,
            points,
            method,
            walks=config.walks,
            seed=config.seed,
            tol=config.tol,
            max_iter=config.max_iter,
            max_steps=config.max_steps,
        )
        if not result.converged:
            raise RuntimeError(
                f"ЛОМ не сошёлся за {result.iterations} итераций; относительная невязка {result.relative_residual:.3g}"
            )
        results[method] = result
    return BoundaryExperiment(
        config,
        problem,
        points,
        reference,
        "grid" if problem.exact is None else "analytic",
        grid,
        results,
        [boundary_metrics(r, points, reference, grid) for r in results.values()],
    )


def boundary_payload(experiment):
    operator = BoundaryOperator(experiment.problem)
    return {
        "config": asdict(experiment.config),
        "reference_kind": experiment.reference_kind,
        "metrics": experiment.metrics,
        "boundary": {
            side: {"kind": c.kind, "value": c.value.tolist(), "kappa": c.kappa.tolist()}
            for side, c in experiment.problem.sides.items()
        },
        "normal_convention": "outward derivative du/dn; physical flux q=-du/dn for k=1",
        "corner_rule": "Dirichlet takes precedence; two Dirichlet values must agree; otherwise both side fluxes contribute",
        "normalization": "volume_weighted_zero_mean" if operator.pure_neumann else None,
        "mc_pure_neumann_supported": False,
        "scheme": "nodal_half_cell_finite_volume_k1_v1",
        "timing": "Includes operator preparation and solver; excludes reference, JIT and plots",
    }


def export_boundary(experiment, folder):
    from .boundary_views import boundary_field_figure

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "experiment.json").write_text(
        json.dumps(boundary_payload(experiment), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (folder / "metrics.csv").write_text(
        rows_csv(experiment.metrics), encoding="utf-8-sig"
    )
    np.savez_compressed(
        folder / "fields.npz",
        reference=experiment.reference,
        grid_reference=experiment.grid_reference,
        f=experiment.problem.f,
        points_yx=experiment.points,
        axis=experiment.problem.axis,
        **{m: r.values for m, r in experiment.results.items()},
    )
    boundary_field_figure(experiment).write_html(
        folder / "field.html", include_plotlyjs=True
    )


def benchmark_boundary(
    config, budgets=(128, 512, 2048), repeats=3, tolerances=(1e-3, 1e-6, 1e-9)
):
    """Independent runs: compare error/cost, without selecting on validation data."""
    if isinstance(repeats, bool) or int(repeats) != repeats or repeats < 2:
        raise ValueError("Нужно минимум два независимых повтора")
    problem = problem_from_config(config)
    operator = BoundaryOperator(problem)
    operator.check_compatibility()
    points = select_boundary_points(problem.n, config.mode, config.point)
    grid = operator.solve()
    reference = grid if problem.exact is None else operator.normalize(problem.exact)
    warmup_boundary()
    jobs = [("lod_serial", tol) for tol in tolerances]
    if not operator.pure_neumann:
        jobs += [("mc", walks) for walks in budgets]
    rows = []
    for method, budget in jobs:
        for repeat in range(int(repeats)):
            seed = int(
                np.random.SeedSequence([config.seed, repeat]).generate_state(1)[0]
            )
            result = solve_boundary(
                problem,
                points,
                method,
                tol=budget if method == "lod_serial" else config.tol,
                walks=config.walks if method == "lod_serial" else budget,
                max_iter=config.max_iter,
                max_steps=config.max_steps,
                seed=seed,
                record=False,
            )
            row = boundary_metrics(result, points, reference, grid)
            row.update(
                budget=budget,
                repeat=repeat,
                seed=seed,
                budget_kind=(
                    "relative_residual_tolerance"
                    if method == "lod_serial"
                    else "walks_per_start"
                ),
            )
            rows.append(row)
    return rows
