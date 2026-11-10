"""Запуск и переносимый экспорт эксперимента в кубе."""

import csv
import json
import platform
from dataclasses import asdict, dataclass
from io import StringIO
from pathlib import Path

import numpy as np

from .problem3d import (
    DIRECT_MAX_N,
    Problem3D,
    make_problem3d,
    reference_solution3d,
    select_points3d,
    workspace_bytes,
)
from .solvers import Result
from .solvers3d import METHODS_3D, solve3d, warmup3d


@dataclass
class Config3D:
    problem: str = "sine"
    n: int = 16
    mode: str = "point"
    point: tuple[int, int, int] | None = None
    plane: str = "xy"
    index: int | None = None
    walks: int = 2048
    seed: int = 42
    tol: float = 1e-7
    max_iter: int = 20000


@dataclass
class Experiment3D:
    config: Config3D
    problem: Problem3D
    points: np.ndarray
    reference: np.ndarray
    reference_kind: str
    grid_reference: np.ndarray | None
    results: dict[str, Result]
    metrics: list[dict]
    metadata: dict


def comparison3d(problem):
    grid = reference_solution3d(problem) if problem.n <= DIRECT_MAX_N else None
    if problem.exact is not None:
        return problem.exact, "analytic", grid
    if grid is None:
        raise ValueError("Для большой 3D-сетки требуется аналитический эталон")
    return grid, "grid", grid


def metrics3d(result, points, reference, grid_reference=None):
    indices = tuple(points.T)
    values = result.values[indices]
    target = reference[indices]
    rmse = float(np.sqrt(np.mean((values - target) ** 2)))
    target_rms = float(np.sqrt(np.mean(target**2)))
    return {
        "method": result.method,
        "scheme": result.extras.get("scheme", "uniform_finite_difference"),
        "data_model": result.extras.get("data_model", "nodal_input_data"),
        "wos_epsilon": result.extras.get("epsilon"),
        "name": METHODS_3D[result.method],
        "value": float(np.mean(values)),
        "reference_value": float(np.mean(target)),
        "seconds": result.elapsed,
        "rmse": rmse,
        "relative_error_pct": 100 * rmse / target_rms if target_rms > 0 else None,
        "rmse_grid": (
            None
            if grid_reference is None
            else float(np.sqrt(np.mean((values - grid_reference[indices]) ** 2)))
        ),
        "walks": result.walks,
        "steps": result.steps,
        "iterations": result.iterations,
        "relative_residual": result.relative_residual,
        "tau": result.extras.get("tau"),
        "converged": result.converged,
    }


def run_experiment3d(config=None, methods=None):
    config = Config3D() if config is None else config
    methods = list(METHODS_3D) if methods is None else list(methods)
    if (
        not methods
        or len(set(methods)) != len(methods)
        or any(method not in METHODS_3D for method in methods)
    ):
        raise ValueError("Выберите ЛОМ и/или обычный MC для 3D")
    memory = workspace_bytes(config.n)
    problem = make_problem3d(config.problem, config.n)
    points = select_points3d(
        config.n, config.mode, config.point, config.plane, config.index
    )
    reference, reference_kind, grid_reference = comparison3d(problem)
    warmup3d()
    results = {}
    for method in methods:
        result = solve3d(
            problem,
            points,
            method,
            walks=config.walks,
            seed=config.seed,
            tol=config.tol,
            max_iter=config.max_iter,
        )
        if not result.converged:
            raise RuntimeError(
                f"ЛОМ не сошёлся за {result.iterations} итераций: относительная невязка {result.relative_residual:g}"
            )
        results[method] = result
    metadata = {
        "dimension": 3,
        "array_order": ["z", "y", "x"],
        "coordinate_order": ["x", "y", "z"],
        "point_indices_order": ["z", "y", "x"],
        "reference_kind": reference_kind,
        "grid_reference_available": grid_reference is not None,
        "workspace_reserve_bytes": memory,
        "memory_scope": "24 dense volumes; excludes JIT, plots and sparse LU",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "timing": "Solver wall time after JIT warmup; reference and plots excluded",
    }
    return Experiment3D(
        config,
        problem,
        points,
        reference,
        reference_kind,
        grid_reference,
        results,
        [
            metrics3d(result, points, reference, grid_reference)
            for result in results.values()
        ],
        metadata,
    )


def rows_csv(rows):
    stream = StringIO(newline="")
    if rows:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return stream.getvalue()


def experiment_payload3d(experiment):
    return {
        "config": asdict(experiment.config),
        "metadata": experiment.metadata,
        "metrics": experiment.metrics,
    }


def export3d(experiment, folder):
    from .views3d import slice_figure3d

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "experiment.json").write_text(
        json.dumps(
            experiment_payload3d(experiment),
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    (folder / "metrics.csv").write_text(
        rows_csv(experiment.metrics), encoding="utf-8-sig", newline=""
    )
    arrays = {
        "reference": experiment.reference,
        "points_zyx": experiment.points,
        "axis": experiment.problem.axis,
    }
    arrays.update(
        {method: result.values for method, result in experiment.results.items()}
    )
    if experiment.grid_reference is not None:
        arrays["grid_reference"] = experiment.grid_reference
    np.savez_compressed(folder / "fields.npz", **arrays)
    plane = experiment.config.plane
    index = (
        experiment.config.index
        if experiment.config.index is not None
        else experiment.config.n // 2
    )
    if experiment.config.mode == "point":
        index = int(experiment.points[0, {"xy": 0, "xz": 1, "yz": 2}[plane]])
    value = slice_figure3d(experiment, plane, index)
    error = slice_figure3d(experiment, plane, index, error=True)
    html = "<!doctype html><html lang='ru'><meta charset='utf-8'><title>PDE Lab — 3D</title><body>"
    html += value.to_html(full_html=False, include_plotlyjs=True)
    html += error.to_html(full_html=False, include_plotlyjs=False)
    html += "</body></html>"
    (folder / "slices.html").write_text(html, encoding="utf-8")
