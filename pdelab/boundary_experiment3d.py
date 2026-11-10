"""Comparison, independent budget series and exports for six mixed cube faces."""

import io
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .boundary import BoundaryCondition
from .boundary3d import (
    FACES,
    BoundaryOperator3D,
    BoundaryProblem3D,
    make_boundary_problem3d,
    select_boundary_points3d,
)
from .boundary_experiment import (
    BoundaryConfig,
    boundary_metrics,
    boundary_payload,
    selected_boundary_methods,
    validate_boundary_budgets,
)
from .boundary_solvers3d import solve_boundary3d, warmup_boundary3d
from .experiment3d import rows_csv
from .problem3d import DIRECT_MAX_N
from .views3d import PLANES, slice_figure3d


@dataclass
class BoundaryConfig3D(BoundaryConfig):
    n: int = 16
    kinds: tuple = (
        "robin",
        "dirichlet",
        "dirichlet",
        "dirichlet",
        "dirichlet",
        "dirichlet",
    )
    kappas: tuple = (1.0,) * 6
    data: tuple = (0.0,) * 6
    plane: str = "xy"
    index: int | None = None


def problem_from_config3d(config):
    if any(len(a) != 6 for a in (config.kinds, config.kappas, config.data)):
        raise ValueError("Нужно по шесть параметров граней: x0, x1, y0, y1, z0, z1")
    if config.problem == "custom":
        faces = {
            face: BoundaryCondition(kind, value, kappa if kind == "robin" else 0)
            for face, kind, value, kappa in zip(
                FACES, config.kinds, config.data, config.kappas
            )
        }
        return BoundaryProblem3D(
            config.n, config.source, faces, name="Свои данные в кубе"
        )
    return make_boundary_problem3d(
        config.problem,
        config.n,
        dict(zip(FACES, config.kinds)),
        dict(zip(FACES, config.kappas)),
    )


@dataclass
class BoundaryExperiment3D:
    config: BoundaryConfig3D
    problem: BoundaryProblem3D
    points: np.ndarray
    reference: np.ndarray
    reference_kind: str
    grid_reference: np.ndarray | None
    results: dict
    metrics: list


def _prepare(config, methods, budgets=None, tolerances=None):
    problem = problem_from_config3d(config)
    if problem.exact is None and problem.n > DIRECT_MAX_N:
        raise ValueError(
            f"Для своих данных нужен прямой эталон: выберите n ≤ {DIRECT_MAX_N}"
        )
    operator = BoundaryOperator3D(problem)
    operator.check_compatibility()
    methods = selected_boundary_methods(operator, methods)
    if budgets is not None and tolerances is not None:
        validate_boundary_budgets(methods, budgets, tolerances)
    points = select_boundary_points3d(
        problem.n, config.mode, config.point, config.plane, config.index
    )
    grid = operator.solve() if problem.n <= DIRECT_MAX_N else None
    reference = grid if problem.exact is None else operator.normalize(problem.exact)
    warmup_boundary3d()
    return problem, points, grid, reference, methods


def run_boundary_experiment3d(config=None, methods=None):
    config = BoundaryConfig3D() if config is None else config
    problem, points, grid, reference, methods = _prepare(config, methods)
    results = {}
    for method in methods:
        result = solve_boundary3d(
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
    return BoundaryExperiment3D(
        config,
        problem,
        points,
        reference,
        "grid" if problem.exact is None else "analytic",
        grid,
        results,
        [boundary_metrics(r, points, reference, grid) for r in results.values()],
    )


def boundary_files3d(experiment, plane=None, index=None):
    """One export path shared by CLI and the browser download."""
    config = experiment.config
    plane = config.plane if plane is None else plane
    if plane not in PLANES:
        raise ValueError("Плоскость: xy, xz или yz")
    if index is None:
        if config.mode == "slice":
            index = config.n // 2 if config.index is None else config.index
        else:
            point = (config.n // 2,) * 3 if config.point is None else config.point
            index = point[2 - PLANES[plane][0]]
    payload = boundary_payload(experiment)
    payload.update(
        face_order=list(FACES),
        face_array_order={"x": "zy", "y": "zx", "z": "yx"},
        view={"plane": plane, "index": int(index)},
    )
    arrays = {
        "reference": experiment.reference,
        "f": experiment.problem.f,
        "points_zyx": experiment.points,
        "axis": experiment.problem.axis,
        **{m: r.values for m, r in experiment.results.items()},
    }
    if experiment.grid_reference is not None:
        arrays["grid_reference"] = experiment.grid_reference
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    return {
        "experiment.json": json.dumps(
            payload, ensure_ascii=False, indent=2, allow_nan=False
        ).encode("utf-8"),
        "metrics.csv": rows_csv(experiment.metrics).encode("utf-8-sig"),
        "fields.npz": buffer.getvalue(),
        "solution.html": slice_figure3d(experiment, plane, index)
        .to_html(include_plotlyjs=True)
        .encode("utf-8"),
        "error.html": slice_figure3d(experiment, plane, index, error=True)
        .to_html(include_plotlyjs=True)
        .encode("utf-8"),
    }


def boundary_archive3d(experiment, plane=None, index=None):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, data in boundary_files3d(experiment, plane, index).items():
            archive.writestr(name, data)
    return buffer.getvalue()


def export_boundary3d(experiment, folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for name, data in boundary_files3d(experiment).items():
        (folder / name).write_bytes(data)


def benchmark_boundary3d(
    config,
    budgets=(128, 512, 2048),
    repeats=3,
    tolerances=(1e-3, 1e-6, 1e-9),
    *,
    methods=None,
):
    if isinstance(repeats, (bool, np.bool_)) or int(repeats) != repeats or repeats < 2:
        raise ValueError("Нужно минимум два независимых повтора")
    problem, points, grid, reference, methods = _prepare(
        config, methods, budgets, tolerances
    )
    rows = []
    for method in methods:
        for budget in (tolerances if method == "lod_serial" else budgets):
            for repeat in range(int(repeats)):
                seed = int(
                    np.random.SeedSequence([config.seed, repeat]).generate_state(1)[0]
                )
                result = solve_boundary3d(
                    problem,
                    points,
                    method,
                    seed=seed,
                    walks=budget if method != "lod_serial" else config.walks,
                    tol=budget if method == "lod_serial" else config.tol,
                    max_iter=config.max_iter,
                    max_steps=config.max_steps,
                )
                row = boundary_metrics(result, points, reference, grid)
                row.update(
                    budget=budget,
                    repeat=repeat,
                    seed=seed,
                    budget_kind=(
                        "walks_per_start"
                        if method != "lod_serial"
                        else "relative_residual_tolerance"
                    ),
                )
                rows.append(row)
    return rows
