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

from .boundary3d import FACES
from .heat3d_picard_mc import CONTROL_MC_METHODS, GROUPED_MC_METHODS, MC_METHODS
from .heat3d_problem import make_heat_problem3d
from .problem3d import select_points3d
from .study_protocol import (
    METRIC_DESCRIPTION,
    METRIC_ID,
    error_metrics,
    run_accuracy_study,
)

METHOD_NAME = "Пикар + ЛОМ (направленные прогонки, 3D)"
METHOD_NAMES = {
    "lod": METHOD_NAME,
    "block": "Пикар + матричная прогонка (3D)",
    "mc": "Пикар + обычный MC (3D)",
    "mc_grouped": "Пикар + групповой MC (3D)",
    "mc_control": "Пикар + грубая сетка и MC-поправка (3D)",
    "mc_control_grouped": "Пикар + грубая сетка и групповая MC-поправка (3D)",
}
BLOCK_DIAGNOSTICS = (
    "assembly_seconds",
    "factorization_seconds",
    "solve_seconds",
    "factorizations",
    "factor_reuses",
)
MC_DIAGNOSTICS = (
    "warmup_seconds",
    "completed_walks",
    "validation_walks",
    "coarse_iterations",
    "coarse_seconds",
    "interpolation_seconds",
    "defect_seconds",
)
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
    "walk_budget": "Лимит числа путей или переходов",
    "path_limit": "Лимит длины пути",
    "time_limit": "Лимит времени полного запуска",
    "memory_limit": "Лимит рабочей памяти",
    "nonfinite": "Неконечные значения",
    "invalid_result": "Нет корректного решения в T",
    "unsupported": "Не поддерживается",
}
APPLICABILITY = (
    "Нелинейная нестационарная теплопроводность в 3D, условия 1 или 3 рода на каждой грани. "
    "Все варианты MC вычисляют полное поле на каждой итерации Пикара."
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
    methods: tuple[str, ...] = ("lod",)
    kinds: tuple[int, ...] = (1, 1, 1, 1, 1, 1)  # x0,x1,y0,y1,z0,z1
    betas: tuple[float, ...] = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0)


def _inputs(config, settings):
    from .heat3d_solver import HeatSettings3D

    if (
        not isinstance(config.methods, (tuple, list))
        or not config.methods
        or any(method not in METHOD_NAMES for method in config.methods)
        or len(set(config.methods)) != len(config.methods)
    ):
        raise ValueError("Выберите различные доступные методы 3D-теплопроводности")
    if config.n > 64:
        raise ValueError("Для этого режима требуется n ≤ 64")
    if not np.isfinite(config.dt) or config.dt <= 0:
        raise ValueError("Шаг времени должен быть положительным и конечным")
    if not np.isfinite(config.T) or config.T < 0:
        raise ValueError("Конечное время должно быть неотрицательным и конечным")
    if (
        not isinstance(config.kinds, (tuple, list))
        or len(config.kinds) != 6
        or any(
            isinstance(kind, (bool, np.bool_)) or kind not in (1, 3)
            for kind in config.kinds
        )
    ):
        raise ValueError("Задайте условия 1 или 3 рода для всех шести граней")
    if (
        not isinstance(config.betas, (tuple, list))
        or len(config.betas) != 6
        or any(
            not np.isscalar(beta)
            or np.iscomplexobj(beta)
            or not np.isfinite(beta)
            or beta <= 0
            for beta in config.betas
        )
    ):
        raise ValueError("Для каждой грани нужен конечный beta > 0")
    problem = make_heat_problem3d(
        config.case,
        config.n,
        config.alpha,
        kinds=dict(zip(FACES, config.kinds, strict=True)),
        beta=dict(zip(FACES, config.betas, strict=True)),
    )
    points = select_points3d(
        config.n, config.mode, config.point, config.plane, config.index
    )
    settings = (
        HeatSettings3D.for_methods(config.methods) if settings is None else settings
    )
    settings.validate()
    for method in config.methods:
        if method in MC_METHODS:
            settings.for_method(method).validate()
            settings.validate_mc(method)
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
        configuration=_json_data(asdict(config)),
        solver_settings=asdict(settings),
        methods=list(config.methods),
        method_names=[METHOD_NAMES[method] for method in config.methods],
        method=config.methods[0] if len(config.methods) == 1 else "multiple",
        method_name="; ".join(METHOD_NAMES[method] for method in config.methods),
        boundary_kinds=dict(zip(FACES, config.kinds, strict=True)),
        boundary_betas={
            face: beta if kind == 3 else None
            for face, kind, beta in zip(FACES, config.kinds, config.betas, strict=True)
        },
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
        timing="Solver seconds include preparation, all physical layers, Picard iterations, and independent MC checks. JIT warmup_seconds, external references and UI/export are separate. Preparation, block assembly/factorization/solve and coarse/interpolation/defect times are subsets of solver seconds.",
        memory="workspace_bytes is an estimated numerical working set; peak_rss_bytes is null unless measured, never inferred from workspace_bytes.",
    )


def _record(result, exact):
    record = {
        field.name: getattr(result, field.name)
        for field in fields(result)
        if field.name not in ("values", "points_zyx", "history", "diagnostic_field")
    }
    record.update(
        method_name=METHOD_NAMES.get(
            result.method, "Ньютон · независимый сеточный эталон"
        ),
        relative_error_pct=None,
        absolute_error=None,
        accuracy_metric=METRIC_ID,
        walks_applicable=result.method in MC_METHODS,
        error_reason="" if result.status == "completed" else "Расчёт не достиг T",
        diagnostic_field_available=result.diagnostic_field is not None,
    )
    if result.status == "completed":
        record.update(error_metrics(result.values, exact))
        if not record["relative_applicable"]:
            record["error_reason"] = "Нулевой эталон: относительная цель неприменима"
    return record


def _reproduction(
    config,
    settings,
    command,
    *,
    repeats=5,
    seed=42,
    budgets=(),
    walk_budgets=(),
    margin=0.8,
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
        "--methods",
        *config.methods,
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
    for face, kind, beta in zip(FACES, config.kinds, config.betas, strict=True):
        parts += ["--" + face, str(kind), "--beta-" + face, str(beta)]
    if config.mode == "slice":
        parts += [
            "--plane",
            config.plane,
            "--index",
            str(config.n // 2 if config.index is None else config.index),
        ]
    if command == "run" and config.newton:
        parts.append("--newton")
    if any(m in MC_METHODS for m in config.methods):
        if settings.mc_tolerance is not None:
            parts += ["--mc-tol", str(settings.mc_tolerance)]
        if settings.mc_change_tolerance is not None:
            parts += ["--mc-change-tol", str(settings.mc_change_tolerance)]
        parts += [
            "--walks",
            str(settings.walks),
            "--coarse-n",
            str(settings.coarse_n),
            "--max-walks",
            str(settings.max_walks),
            "--max-transitions",
            str(settings.max_transitions),
            "--max-path-steps",
            str(settings.max_path_steps),
        ]
        if command == "run":
            parts += ["--seed", str(settings.seed)]
        elif walk_budgets:
            parts += ["--walk-budgets", *(str(v) for v in walk_budgets)]
    if any(m in GROUPED_MC_METHODS for m in config.methods):
        parts += ["--batch-size", str(settings.batch_size)]
    if command == "study":
        parts += [
            "--repeats",
            str(repeats),
            "--seed",
            str(seed),
            "--margin",
            str(margin),
        ]
        if budgets:
            parts += ["--budgets", *(str(v) for v in budgets)]
    parts += ["--out", "runs/heat3d-" + command]
    return shlex.join(parts)


def compare_heat3d(config=HeatConfig3D(), settings=None, progress=None):
    from .heat3d_solver import solve_heat3d

    problem, points, settings = _inputs(config, settings)
    started = perf_counter()
    exact = problem.evaluate(problem.exact, config.T)[tuple(points.T)]
    analytic_seconds = perf_counter() - started
    records, histories, runs = [], [], {}
    arrays = {"points_zyx": points, "analytic": exact}
    progress_count = len(config.methods) + int(config.newton and config.n <= 16)
    for index, method in enumerate(config.methods):
        result = solve_heat3d(
            problem,
            points,
            dt=config.dt,
            T=config.T,
            settings=settings,
            method=method,
            progress=_method_progress(progress, method, index, progress_count),
        )
        runs[method] = result
        record = _record(result, exact)
        effective = settings.for_method(method)
        record.update(
            tolerance=effective.tolerance, change_tolerance=effective.change_tolerance
        )
        if method in MC_METHODS:
            record.update(
                walks_per_free_node=settings.walks,
                seed=settings.seed,
                coarse_n=settings.coarse_n if method in CONTROL_MC_METHODS else None,
            )
        records.append(record)
        if result.values is not None:
            arrays[method] = result.values
        if result.diagnostic_field is not None:
            # Existing consumers of the default LOM export retain their key.
            key = (
                "failed_candidate" if method == "lod" else method + "_failed_candidate"
            )
            arrays[key] = result.diagnostic_field
        histories += [{"method": method, **row} for row in result.history]
    grid_diagnostic = dict(
        status="not_requested", reason="Независимый сеточный эталон не запрошен"
    )
    reference_seconds = 0.0
    if config.newton:
        if config.n > 16:
            grid_diagnostic = dict(
                status="unsupported",
                reason="Независимый Ньютон ограничен n ≤ 16; аналитический эталон остаётся доступен",
            )
        else:
            if progress:
                _method_progress(
                    progress, "newton", len(config.methods), progress_count
                )(0.0, "Независимая сеточная проверка…")
            grid = solve_heat3d(
                problem,
                points,
                dt=config.dt,
                T=config.T,
                settings=replace(settings, tolerance=1e-11, change_tolerance=1e-11),
                method="newton",
                progress=_method_progress(
                    progress, "newton", len(config.methods), progress_count
                ),
            )
            reference_seconds = grid.seconds
            grid_diagnostic = _record(grid, exact)
            if grid.diagnostic_field is not None:
                arrays["newton_failed_candidate"] = grid.diagnostic_field
            histories += [{"method": "newton", **row} for row in grid.history]
            if grid.values is not None:
                arrays["newton"] = grid.values
                for method, result in runs.items():
                    if result.values is not None:
                        grid_diagnostic[method + "_vs_grid"] = error_metrics(
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
        rows=records,
        grid_diagnostic=grid_diagnostic,
        history=histories,
        arrays=arrays,
    )


def _method_progress(progress, method, index, count):
    if progress is None:
        return None

    def report(fraction, message):
        progress(
            (index + fraction) / count,
            METHOD_NAMES.get(method, "Ньютон") + ": " + message,
        )

    return report


def _study_diagnostics(study, method, diagnostics, settings):
    """Attach backend costs without changing candidate selection or its metric."""
    for trial, diagnostic in zip(study["trials"], diagnostics, strict=True):
        trial.update(diagnostic)
    for collection in ("rows", "trials", "series"):
        for row in study[collection]:
            row.update(
                method=method,
                method_name=METHOD_NAMES[method],
                batch_size=(
                    settings.batch_size if method in GROUPED_MC_METHODS else None
                ),
            )
            if method in MC_METHODS:
                row.update(
                    budget_kind="walks_per_free_node_per_solve",
                    coarse_n=(
                        settings.coarse_n if method in CONTROL_MC_METHODS else None
                    ),
                    tolerance=settings.tolerance,
                    change_tolerance=settings.change_tolerance,
                )
            if collection != "trials":
                trials = [
                    trial
                    for trial in study["trials"]
                    if (trial["phase"], trial["candidate"])
                    == (row["phase"], row["candidate"])
                ]
                for key in (*BLOCK_DIAGNOSTICS, *MC_DIAGNOSTICS):
                    values = [trial.get(key) for trial in trials]
                    if any(value is not None for value in values):
                        row[key] = (
                            sum(values)
                            if all(value is not None for value in values)
                            else None
                        )


def study_heat3d(
    config=HeatConfig3D(),
    settings=None,
    *,
    repeats=5,
    seed=42,
    budgets=(1e-3, 1e-5, 1e-7, 1e-9),
    walk_budgets=(256, 1024, 4096),
    calibration_fraction=0.8,
    progress=None,
):
    from .heat3d_solver import solve_heat3d

    if config.newton:
        raise ValueError(
            "Исследование использует аналитический эталон; Ньютон доступен в сравнении"
        )
    budgets = tuple(budgets)
    if any(m not in MC_METHODS for m in config.methods) and (
        not budgets
        or any(not np.isfinite(budget) or not 0 < budget < 1 for budget in budgets)
    ):
        raise ValueError("Допуски исследования должны быть строго между 0 и 1")
    walk_budgets = tuple(walk_budgets)
    if any(m in MC_METHODS for m in config.methods) and (
        not walk_budgets
        or any(
            isinstance(b, (bool, np.bool_))
            or not isinstance(b, (int, np.integer))
            or not 2 <= b < 2**31
            for b in walk_budgets
        )
    ):
        raise ValueError("Бюджеты MC должны быть целыми от 2 до 2³¹−1")
    problem, points, settings = _inputs(config, settings)
    if any(method in MC_METHODS for method in config.methods):
        settings = replace(settings, seed=seed)
    started = perf_counter()
    exact = problem.evaluate(problem.exact, config.T)[tuple(points.T)]
    analytic_seconds = perf_counter() - started

    studies = []
    for index, method in enumerate(config.methods):
        diagnostics = []
        stochastic = method in MC_METHODS

        def runner(budget, trial_seed):
            trial_settings = (
                replace(settings, walks=int(budget), seed=trial_seed)
                if stochastic
                else replace(
                    settings, tolerance=float(budget), change_tolerance=float(budget)
                )
            )
            result = solve_heat3d(
                problem,
                points,
                dt=config.dt,
                T=config.T,
                settings=trial_settings,
                method=method,
            )
            diagnostics.append(
                {
                    key: getattr(result, key)
                    for key in (MC_DIAGNOSTICS if stochastic else BLOCK_DIAGNOSTICS)
                    if getattr(result, key, None) is not None
                }
            )
            return result

        study = run_accuracy_study(
            runner,
            walk_budgets if stochastic else budgets,
            exact,
            repeats=repeats,
            seed=seed,
            calibration_fraction=calibration_fraction,
            method=method,
            budget_kind=(
                "walks_per_free_node_per_solve"
                if stochastic
                else "relative_residual_tolerance"
            ),
            walks_applicable=stochastic,
            retain_values=stochastic and len(points) == 1,
            progress=_method_progress(progress, method, index, len(config.methods)),
        )
        _study_diagnostics(study, method, diagnostics, settings.for_method(method))
        studies.append(study)
    study = dict(studies[0])
    study["budgets_by_method"] = {
        method: list(walk_budgets if method in MC_METHODS else budgets)
        for method in config.methods
    }
    if any(m in MC_METHODS for m in config.methods) and any(
        m not in MC_METHODS for m in config.methods
    ):
        study["budgets"] = (
            None  # candidates have different units; see budgets_by_method
        )
    for key in ("rows", "trials", "series"):
        study[key] = [row for item in studies for row in item[key]]
    for key in ("pilot_seconds", "final_seconds", "elapsed_seconds"):
        study[key] = sum(item[key] for item in studies)
    metadata = _metadata(config, settings, points)
    metadata.update(
        command="study",
        protocol=study["protocol"],
        repeats=repeats,
        seed=seed,
        budgets=list(budgets),
        walk_budgets=list(walk_budgets),
        calibration_fraction=calibration_fraction,
        budget_parameters=(
            {
                method: (
                    ["walks"]
                    if method in MC_METHODS
                    else ["tolerance", "change_tolerance"]
                )
                for method in config.methods
            }
            if any(m in MC_METHODS for m in config.methods)
            else ["tolerance", "change_tolerance"]
        ),
        reproduction=_reproduction(
            config,
            settings,
            "study",
            repeats=repeats,
            seed=seed,
            budgets=budgets,
            walk_budgets=walk_budgets,
            margin=calibration_fraction,
        ),
        analytic_reference_seconds=analytic_seconds,
        seed_usage="Each MC repeat starts the whole history from the prescribed initial data. Trial seeds derive from phase/candidate/repeat, and layer/outer/update-or-check select separate streams. Methods share trial seeds for paired comparisons.",
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
