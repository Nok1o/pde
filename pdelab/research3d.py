"""Подбор бюджета и независимая итоговая проверка точности в одной 3D-точке."""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from time import perf_counter

import numpy as np

from .experiment3d import Config3D, comparison3d, rows_csv
from .problem3d import make_problem3d, select_points3d
from .solvers3d import METHODS_3D, solve3d, warmup3d

TARGETS_3D = (1.0, 5.0, 10.0)


@dataclass
class StudySettings3D:
    config: Config3D
    repeats: int = 5
    max_walks: int = 32768
    seconds_per_method: float = 60.0
    methods: tuple[str, ...] = tuple(METHODS_3D)


@dataclass
class Study3D:
    settings: StudySettings3D
    reference_kind: str
    reference_value: float
    rows: list[dict] = field(default_factory=list)
    trials: list[dict] = field(default_factory=list)


def _budgets3d(method, maximum):
    if method == "lod_serial":
        return [0.1, 0.03, 0.01, 0.003, 0.001, 1e-4, 1e-6, 1e-8, 1e-10]
    result = [min(64, maximum)]
    while result[-1] < maximum:
        result.append(min(2 * result[-1], maximum))
    return result


def _series3d(study, problem, points, method, budget, budget_index, phase, deadline):
    config = study.settings.config
    repeats = (
        1
        if method == "lod_serial" and phase == "calibration"
        else int(study.settings.repeats)
    )
    values, times = [], []
    for repeat in range(repeats):
        if perf_counter() >= deadline:
            raise TimeoutError("Лимит времени поиска")
        seed = int(
            np.random.SeedSequence(
                [
                    int(config.seed),
                    int(method != "lod_serial"),
                    budget_index,
                    int(phase == "validation"),
                    repeat,
                ]
            ).generate_state(1)[0]
        )
        result = solve3d(
            problem,
            points,
            method,
            walks=int(budget) if method != "lod_serial" else 2,
            tol=float(budget) if method == "lod_serial" else config.tol,
            seed=seed,
            max_iter=config.max_iter,
        )
        if not result.converged:
            raise RuntimeError("ЛОМ не достиг допуска невязки")
        value = float(result.values[tuple(points[0])])
        if not np.isfinite(value):
            raise RuntimeError("Неконечная оценка решения")
        values.append(value)
        times.append(result.elapsed)
        study.trials.append(
            {
                "method": method,
                "phase": phase,
                "repeat": repeat + 1,
                "seed": seed,
                "budget": budget,
                "value": value,
                "reference_value": study.reference_value,
                "seconds": result.elapsed,
                "steps": result.steps,
                "walks": result.walks,
                "iterations": result.iterations,
            }
        )
    error = (
        100
        * float(np.sqrt(np.mean((np.array(values) - study.reference_value) ** 2)))
        / abs(study.reference_value)
    )
    return error, float(np.mean(times))


def accuracy_study3d(settings, progress=None):
    config = settings.config
    methods = tuple(settings.methods)
    if (
        not methods
        or len(set(methods)) != len(methods)
        or any(method not in METHODS_3D for method in methods)
    ):
        raise ValueError(
            "Для исследования выберите различные методы lod_serial и/или mc"
        )
    if config.mode != "point":
        raise ValueError("Исследование проводится в одной точке")
    for name, value, lower in (
        ("repeats", settings.repeats, 2),
        ("max_walks", settings.max_walks, 2),
    ):
        if not np.isfinite(value) or int(value) != value or value < lower:
            raise ValueError(f"{name} должно быть целым >= {lower}")
    if not np.isfinite(settings.seconds_per_method) or settings.seconds_per_method <= 0:
        raise ValueError("Лимит времени должен быть положительным")
    if (
        not np.isfinite(config.seed)
        or int(config.seed) != config.seed
        or not 0 <= config.seed < 2**32
    ):
        raise ValueError("Seed должен быть целым в [0, 2³²)")
    problem = make_problem3d(config.problem, config.n)
    points = select_points3d(config.n, point=config.point)
    reference, kind, _ = comparison3d(problem)
    target_value = float(reference[tuple(points[0])])
    if target_value == 0:
        raise ValueError("Эталон равен нулю: относительная точность не определена")
    study = Study3D(settings, kind, target_value)
    warmup3d()
    for method_index, method in enumerate(methods):
        rows = [
            {
                "target_pct": target,
                "method": method,
                "name": METHODS_3D[method],
                "seconds": None,
                "relative_error_pct": None,
                "calibration_error_pct": None,
                "budget": None,
                "budget_kind": (
                    "relative_residual_tolerance"
                    if method == "lod_serial"
                    else "walks_per_start"
                ),
                "status": "Не достигнута в пределах поиска",
            }
            for target in TARGETS_3D
        ]
        study.rows.extend(rows)
        deadline = perf_counter() + settings.seconds_per_method
        selected = {}
        try:
            # Сначала полностью фиксируем бюджеты. Проверочная серия не влияет на выбор.
            for budget_index, budget in enumerate(
                _budgets3d(method, settings.max_walks)
            ):
                if progress:
                    progress(
                        method_index / len(methods),
                        f"{METHODS_3D[method]} · подбор · {budget:g}",
                    )
                error, _ = _series3d(
                    study,
                    problem,
                    points,
                    method,
                    budget,
                    budget_index,
                    "calibration",
                    deadline,
                )
                for row in rows:
                    if row["target_pct"] not in selected:
                        row["calibration_error_pct"] = error
                        if error <= row["target_pct"]:
                            selected[row["target_pct"]] = (budget, budget_index)
                            row["budget"] = budget
                if len(selected) == len(rows):
                    break
        except (TimeoutError, RuntimeError) as exc:
            for row in rows:
                if row["target_pct"] not in selected:
                    row["status"] = str(exc)

        # Даже после неудачи более строгого бюджета проверяем уже найденные.
        # Общий deadline не продлевается, подбор после проверки не возобновляется.
        confirmations = {}
        for row in rows:
            chosen = selected.get(row["target_pct"])
            if chosen is None:
                continue
            if chosen not in confirmations:
                if progress:
                    progress(
                        method_index / len(methods),
                        f"{METHODS_3D[method]} · контроль · {chosen[0]:g}",
                    )
                try:
                    confirmations[chosen] = _series3d(
                        study, problem, points, method, *chosen, "validation", deadline
                    )
                except (TimeoutError, RuntimeError) as exc:
                    confirmations[chosen] = exc
            confirmation = confirmations[chosen]
            if isinstance(confirmation, Exception):
                row["status"] = f"Бюджет найден, проверка не завершена: {confirmation}"
                continue
            error, seconds = confirmation
            row["relative_error_pct"] = error
            row["status"] = (
                "Достигнута"
                if error <= row["target_pct"]
                else "Не достигнута на контрольной серии"
            )
            if error <= row["target_pct"]:
                row["seconds"] = seconds
        if progress:
            progress((method_index + 1) / len(methods), f"{METHODS_3D[method]}: готово")
    return study


def timing_rows3d(study):
    return [
        {
            "Относительная ошибка, %": target,
            **{
                f"{row['name']}, с": row["seconds"]
                for row in study.rows
                if row["target_pct"] == target
            },
        }
        for target in TARGETS_3D
    ]


def study_payload3d(study):
    return {
        "settings": asdict(study.settings),
        "dimension": 3,
        "coordinate_order": ["x", "y", "z"],
        "reference_kind": study.reference_kind,
        "reference_value": study.reference_value,
        "selection": "First calibration budget, fixed before final independent validation; no retry after validation failure",
        "accuracy": "Relative RMSE of individual runs, not error of their mean; empirical estimate",
        "timing": "Mean wall time of validation runs after warmup; search and reference excluded",
    }


def export_study3d(study, folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for name, rows in (
        ("timings", timing_rows3d(study)),
        ("thresholds", study.rows),
        ("trials", study.trials),
    ):
        (folder / f"{name}.csv").write_text(
            rows_csv(rows), encoding="utf-8-sig", newline=""
        )
    (folder / "settings.json").write_text(
        json.dumps(
            study_payload3d(study), ensure_ascii=False, indent=2, allow_nan=False
        ),
        encoding="utf-8",
    )
