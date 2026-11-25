"""One service for the supported 3D nonlinear UI, CLI and portable evidence."""

import csv
import io
import json
import platform
import shlex
import subprocess
import zipfile
from dataclasses import asdict, dataclass, fields, replace
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from time import perf_counter

import numpy as np

from .heat3d_problem import make_heat_problem3d
from .problem3d import select_points3d
from .study_protocol import (
    METRIC_DESCRIPTION,
    METRIC_ID,
    error_metrics,
    run_accuracy_study,
)

METHOD_NAME = "Пикар + ЛОМ (направленные прогонки, 3D)"
STATUS_LABELS = {
    "not_requested": "Проверка не выбрана",
    "completed": "Завершён",
    "confirmed": "Достигнута",
    "not_reached": "Нет заданного запаса точности",
    "pilot_incomplete": "Предварительная серия не завершена",
    "validation_failed": "Финальная проверка не подтвердила цель",
    "validation_incomplete": "Финальная серия не завершена",
    "relative_target_not_applicable": "Относительная цель неприменима",
    "linear_limit": "Лимит линейных итераций",
    "outer_limit": "Лимит итераций Пикара",
    "walk_budget": "Лимит числа путей",
    "path_limit": "Лимит длины пути",
    "time_limit": "Лимит времени полного запуска",
    "memory_limit": "Лимит рабочей памяти",
    "nonfinite": "Неконечные значения",
    "invalid_result": "Нет корректного решения в T",
    "unsupported": "Не поддерживается",
}
APPLICABILITY = (
    "Нелинейная нестационарная теплопроводность в 3D, все условия 1 рода. "
    "MC для этой модели и условия 2/3 рода здесь не реализованы; "
    "они не заменяются стационарными или двумерными методами."
)


@dataclass(frozen=True)
class HeatConfig3D:
    case: str = "sine"
    n: int = 8
    alpha: float = 0.5
    dt: float = 0.01
    T: float = 0.02
    mode: str = "point"
    point: tuple[int, int, int] | None = None  # public x,y,z
    plane: str = "xy"
    index: int | None = None
    newton: bool = False


def _inputs(config, settings):
    from .heat3d_solver import HeatSettings3D

    if config.n > 64:
        raise ValueError("Для этого режима требуется n ≤ 64")
    if not np.isfinite(config.dt) or config.dt <= 0:
        raise ValueError("Шаг времени должен быть положительным и конечным")
    if not np.isfinite(config.T) or config.T < 0:
        raise ValueError("Конечное время должно быть неотрицательным и конечным")
    problem = make_heat_problem3d(config.case, config.n, config.alpha)
    points = select_points3d(
        config.n, config.mode, config.point, config.plane, config.index
    )
    settings = HeatSettings3D() if settings is None else settings
    settings.validate()
    return problem, points, settings


def _versions():
    result = {}
    for name in ("numpy", "scipy", "numba", "streamlit"):
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = None
    return result


def _metadata(config, settings, points):
    root = Path(__file__).resolve().parents[1]
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, stderr=subprocess.DEVNULL, text=True
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"],
                cwd=root,
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
        )
    except (OSError, subprocess.SubprocessError):
        commit, dirty = None, None
    return dict(
        schema_version="heat3d_v1",
        configuration=asdict(config),
        solver_settings=asdict(settings),
        method="lod",
        method_name=METHOD_NAME,
        boundary_kinds=dict.fromkeys(("x0", "x1", "y0", "y1", "z0", "z1"), 1),
        applicability=APPLICABILITY,
        points_zyx=points.tolist(),
        array_order="zyx",
        coordinate_order="xyz",
        query_size=len(points),
        target_time=config.T,
        reference_kind="analytic_manufactured",
        accuracy_metric=METRIC_ID,
        accuracy=METRIC_DESCRIPTION,
        nonlinear_residual="||A(u)u-rhs(u)||inf / fixed layer scale S; separate from analytic error",
        commit=commit,
        working_tree_dirty=dirty,
        python=platform.python_version(),
        platform=platform.platform(),
        versions=_versions(),
        timing="Each solve includes preparation, all physical layers, Picard and linear iterations; analytic/Newton references and UI/export are separate. preparation_seconds is a subset of seconds.",
        memory="workspace_bytes is an estimated numerical working set; peak_rss_bytes is null unless measured, never inferred from workspace_bytes.",
    )


def _record(result, exact):
    record = {
        field.name: getattr(result, field.name)
        for field in fields(result)
        if field.name not in ("values", "points_zyx", "history", "diagnostic_field")
    }
    record.update(
        method_name=(
            METHOD_NAME
            if result.method == "lod"
            else "Ньютон · независимый сеточный эталон"
        ),
        relative_error_pct=None,
        absolute_error=None,
        accuracy_metric=METRIC_ID,
        walks_applicable=False,
        error_reason="" if result.status == "completed" else "Расчёт не достиг T",
        diagnostic_field_available=result.diagnostic_field is not None,
    )
    if result.status == "completed":
        record.update(error_metrics(result.values, exact))
        if not record["relative_applicable"]:
            record["error_reason"] = "Нулевой эталон: относительная цель неприменима"
    return record


def _reproduction(
    config, settings, command, *, repeats=5, seed=42, budgets=(), margin=0.8
):
    parts = [
        "python",
        "-m",
        "pdelab.heat3d_cli",
        command,
        "--case",
        config.case,
        "--n",
        str(config.n),
        "--alpha",
        str(config.alpha),
        "--dt",
        str(config.dt),
        "--T",
        str(config.T),
        "--query",
        config.mode,
        "--tol",
        str(settings.tolerance),
        "--change-tol",
        str(settings.change_tolerance),
        "--linear-tol",
        str(settings.linear_tolerance),
        "--relaxation",
        str(settings.relaxation),
        "--max-outer",
        str(settings.max_outer),
        "--max-inner",
        str(settings.max_inner),
        "--max-seconds",
        str(settings.max_seconds),
        "--max-workspace-bytes",
        str(settings.max_workspace_bytes),
    ]
    if config.mode == "point" and config.point is not None:
        parts += ["--point", *(str(v) for v in config.point)]
    if config.mode == "slice":
        parts += [
            "--plane",
            config.plane,
            "--index",
            str(config.n // 2 if config.index is None else config.index),
        ]
    if command == "run" and config.newton:
        parts.append("--newton")
    if command == "study":
        parts += [
            "--repeats",
            str(repeats),
            "--seed",
            str(seed),
            "--margin",
            str(margin),
            "--budgets",
            *(str(v) for v in budgets),
        ]
    parts += ["--out", "runs/heat3d-" + command]
    return shlex.join(parts)


def compare_heat3d(config=HeatConfig3D(), settings=None, progress=None):
    from .heat3d_solver import solve_heat3d

    problem, points, settings = _inputs(config, settings)
    started = perf_counter()
    exact = problem.evaluate(problem.exact, config.T)[tuple(points.T)]
    analytic_seconds = perf_counter() - started
    result = solve_heat3d(
        problem,
        points,
        dt=config.dt,
        T=config.T,
        settings=settings,
        method="lod",
        progress=progress,
    )
    record = _record(result, exact)
    grid_diagnostic = dict(
        status="not_requested", reason="Независимый сеточный эталон не запрошен"
    )
    arrays = {"points_zyx": points, "analytic": exact}
    if result.values is not None:
        arrays["lod"] = result.values
    if result.diagnostic_field is not None:
        arrays["failed_candidate"] = result.diagnostic_field
    histories = [{"method": "lod", **row} for row in result.history]
    reference_seconds = 0.0
    if config.newton:
        if config.n > 16:
            grid_diagnostic = dict(
                status="unsupported",
                reason="Независимый Ньютон ограничен n ≤ 16; аналитический эталон остаётся доступен",
            )
        else:
            if progress:
                progress(0.0, "Независимая сеточная проверка Ньютоном…")
            grid = solve_heat3d(
                problem,
                points,
                dt=config.dt,
                T=config.T,
                settings=replace(settings, tolerance=1e-11, change_tolerance=1e-11),
                method="newton",
                progress=progress,
            )
            reference_seconds = grid.seconds
            grid_diagnostic = _record(grid, exact)
            if grid.diagnostic_field is not None:
                arrays["newton_failed_candidate"] = grid.diagnostic_field
            histories += [{"method": "newton", **row} for row in grid.history]
            if grid.values is not None:
                arrays["newton"] = grid.values
                if result.values is not None:
                    grid_diagnostic["lod_vs_grid"] = error_metrics(
                        result.values, grid.values
                    )
    metadata = _metadata(config, settings, points)
    metadata.update(
        command="run",
        reproduction=_reproduction(config, settings, "run"),
        analytic_reference_seconds=analytic_seconds,
        grid_reference_seconds=reference_seconds,
    )
    return dict(
        metadata=metadata,
        rows=[record],
        grid_diagnostic=grid_diagnostic,
        history=histories,
        arrays=arrays,
    )


def study_heat3d(
    config=HeatConfig3D(),
    settings=None,
    *,
    repeats=5,
    seed=42,
    budgets=(1e-3, 1e-5, 1e-7, 1e-9),
    calibration_fraction=0.8,
    progress=None,
):
    from .heat3d_solver import solve_heat3d

    if config.newton:
        raise ValueError(
            "Исследование использует аналитический эталон; Ньютон доступен в сравнении"
        )
    budgets = tuple(budgets)
    if not budgets or any(
        not np.isfinite(budget) or not 0 < budget < 1 for budget in budgets
    ):
        raise ValueError("Допуски исследования должны быть строго между 0 и 1")
    problem, points, settings = _inputs(config, settings)
    started = perf_counter()
    exact = problem.evaluate(problem.exact, config.T)[tuple(points.T)]
    analytic_seconds = perf_counter() - started

    def runner(budget, trial_seed):
        # Deterministic runner ignores the independent stream; retain its ID in evidence.
        return solve_heat3d(
            problem,
            points,
            dt=config.dt,
            T=config.T,
            settings=replace(
                settings, tolerance=float(budget), change_tolerance=float(budget)
            ),
            method="lod",
        )

    study = run_accuracy_study(
        runner,
        budgets,
        exact,
        repeats=repeats,
        seed=seed,
        calibration_fraction=calibration_fraction,
        method="lod",
        progress=progress,
    )
    for row in study["rows"]:
        row["method_name"] = METHOD_NAME
    metadata = _metadata(config, settings, points)
    metadata.update(
        command="study",
        protocol=study["protocol"],
        repeats=repeats,
        seed=seed,
        budgets=list(budgets),
        calibration_fraction=calibration_fraction,
        budget_parameters=["tolerance", "change_tolerance"],
        reproduction=_reproduction(
            config,
            settings,
            "study",
            repeats=repeats,
            seed=seed,
            budgets=budgets,
            margin=calibration_fraction,
        ),
        analytic_reference_seconds=analytic_seconds,
        seed_usage="Independent phase/candidate/repeat stream identifiers; deterministic LOM does not consume random numbers.",
    )
    return dict(metadata=metadata, **study)


def _json_data(value):
    if isinstance(value, np.ndarray):
        return _json_data(value.tolist())
    if isinstance(value, np.generic):
        return _json_data(value.item())
    if isinstance(value, dict):
        return {str(k): _json_data(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_data(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def json_bytes(value):
    return json.dumps(
        _json_data(value), ensure_ascii=False, indent=2, allow_nan=False
    ).encode("utf-8")


def csv_bytes(rows):
    buffer = io.StringIO(newline="")
    if rows:
        columns = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(buffer, fieldnames=columns)
        writer.writeheader()
        for row in _json_data(rows):
            writer.writerow(
                {
                    key: (
                        json.dumps(value, ensure_ascii=False)
                        if isinstance(value, (dict, list))
                        else value
                    )
                    for key, value in row.items()
                }
            )
    return buffer.getvalue().encode("utf-8-sig")


def result_files(result):
    payload = {key: value for key, value in result.items() if key != "arrays"}
    files = {
        "result.json": json_bytes(payload),
        "settings.json": json_bytes(result["metadata"]),
        "table.csv": csv_bytes(result["rows"]),
        "reproduce.txt": (result["metadata"]["reproduction"] + "\n").encode(),
    }
    for key in ("history", "trials", "series"):
        if key in result:
            files[key + ".csv"] = csv_bytes(result[key])
    if "arrays" in result:
        buffer = io.BytesIO()
        np.savez_compressed(buffer, **result["arrays"])
        files["query-values.npz"] = buffer.getvalue()
    return files


def result_archive(result):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in result_files(result).items():
            archive.writestr(name, content)
    return buffer.getvalue()


def export_result(result, folder):
    files = result_files(result)
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    # Only remove this exporter's obsolete optional artifacts, not user files.
    for name in ("history.csv", "trials.csv", "series.csv", "query-values.npz"):
        if name not in files:
            (folder / name).unlink(missing_ok=True)
    for name, content in files.items():
        (folder / name).write_bytes(content)
    (folder / "heat3d.zip").write_bytes(result_archive(result))
