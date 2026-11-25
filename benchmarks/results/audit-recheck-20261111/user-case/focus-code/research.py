"""Время достижения заданной относительной точности на выбранных узлах."""

import csv
import json
import platform
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from time import perf_counter

import numba
import numpy as np

from .experiment import Config
from .problem import make_problem, reference_solution, select_points
from .reference import choose_reference
from .solvers import METHODS, solve, warmup
from .study_policy import (
    accuracy_metadata,
    mean_relative_error,
    phase_seed_index,
    search_budgets,
)

TARGETS_PCT = (1.0, 5.0, 10.0)
LOD_SEARCH_TOLERANCES = (
    1e-1,
    5e-2,
    1e-2,
    5e-3,
    1e-3,
    5e-4,
    1e-4,
    1e-5,
    1e-6,
    1e-7,
    1e-8,
    1e-10,
)


@dataclass
class StudySettings:
    config: Config
    repeats: int = 5
    max_walks: int = 131072
    seconds_per_method: float = 60.0


@dataclass
class StudyResult:
    settings: StudySettings
    reference_kind: str
    reference_label: str
    reference_value: float | list[float]
    rows: list[dict] = field(default_factory=list)
    trials: list[dict] = field(default_factory=list)
    search_seconds: float = 0.0


def _walk_budgets(max_walks, batches):
    """Удваиваем бюджет, обязательно проверяя заданный верхний предел."""
    budget = min(max_walks, max(64, 2 * batches))
    budgets = [budget]
    while budget < max_walks:
        budget = min(2 * budget, max_walks)
        budgets.append(budget)
    return budgets


def _trial_seed(seed, budget_index, phase, repeat_index):
    """Подбор и проверка используют разные воспроизводимые серии путей."""
    phase_index = phase_seed_index(phase)
    sequence = np.random.SeedSequence([seed, budget_index, phase_index, repeat_index])
    return int(sequence.generate_state(1)[0])


def _measure_budget(
    study,
    problem,
    points,
    method,
    budget,
    budget_index,
    phase,
    deadline,
    progress,
    grid_values,
):
    config = study.settings.config
    is_lod = method == "lod_serial"
    # Детерминированному ЛОМ достаточно одного предварительного расчёта.
    repeats = 1 if is_lod and phase == "calibration" else study.settings.repeats
    values = []
    times = []
    phase_label = "проверка" if phase == "validation" else "подбор"
    budget_label = f"tol={budget:g}" if is_lod else f"N={budget}"

    for repeat_index in range(repeats):
        if perf_counter() >= deadline:
            raise TimeoutError("Лимит времени поиска")
        if progress:
            progress(
                f"{METHODS[method]} · {budget_label} · {phase_label} "
                f"{repeat_index + 1}/{repeats}"
            )
        seed = _trial_seed(config.seed, budget_index, phase, repeat_index)
        result = solve(
            problem,
            points,
            method,
            walks=max(64, 2 * config.batches) if is_lod else int(budget),
            seed=seed,
            tol=float(budget) if is_lod else config.tol,
            max_iter=config.max_iter,
            batches=config.batches,
            record=False,
            coarse_n=config.coarse_n,
        )
        if not result.converged:
            raise RuntimeError("ЛОМ не достиг допуска невязки")
        value = result.values[tuple(points.T)]
        if len(points) == 1:
            value = float(value[0])
        if not np.isfinite(value).all():
            raise RuntimeError("Решатель вернул неконечное значение")
        values.append(value)
        times.append(result.elapsed)
        study.trials.append(
            {
                "method": method,
                "name": METHODS[method],
                "phase": phase,
                "repeat": repeat_index + 1,
                "seed": seed,
                "budget": budget,
                "budget_kind": (
                    "relative_residual_tolerance" if is_lod else "walks_per_start"
                ),
                "value": np.asarray(value).tolist(),
                "reference_value": study.reference_value,
                "relative_error_pct": mean_relative_error(
                    [value], study.reference_value
                ),
                "preparation_seconds": result.extras.get("preparation_seconds", 0.0),
                "path_seconds": result.extras.get("path_seconds", result.elapsed),
                "seconds": result.elapsed,
                "walks": result.walks,
                "steps": result.steps,
                "iterations": result.iterations,
                "residual": result.residual,
                "coarse_n": result.extras.get("coarse_n"),
                "grid_rmse": float(
                    np.sqrt(np.mean((np.asarray(value) - grid_values) ** 2))
                ),
                "mean_path_variance": (
                    float(
                        np.nanmean(
                            result.stderr[tuple(points.T)] ** 2
                            * result.sample_count[tuple(points.T)]
                        )
                    )
                    if not is_lod
                    else None
                ),
            }
        )

    return {
        "relative_error_pct": mean_relative_error(values, study.reference_value),
        "seconds": float(np.mean(times)),
    }


def accuracy_study(settings: StudySettings, progress=None):
    """Подбираем бюджет и подтверждаем точность новой независимой серией.

    В таблицу попадает среднее время одного подтверждающего запуска. Подготовка
    слоёв включена в solve.elapsed. Поиск бюджета, эталон и прогрев не включены.
    Порог проверяется по среднему модулю относительной ошибки отдельных запусков.
    Это эмпирическая точность на сериях, не вероятностная гарантия.
    """
    config = settings.config
    if int(settings.repeats) != settings.repeats or settings.repeats < 2:
        raise ValueError("Нужно минимум два независимых запуска")
    if int(config.batches) != config.batches or config.batches < 2:
        raise ValueError("Для метода слоёв нужно минимум два полных расчёта")
    if (
        int(settings.max_walks) != settings.max_walks
        or settings.max_walks < 2 * config.batches
    ):
        raise ValueError(
            "Лимит путей должен быть целым и не меньше 2 × числа расчётов слоёв"
        )
    if not np.isfinite(settings.seconds_per_method) or settings.seconds_per_method <= 0:
        raise ValueError("Лимит времени должен быть положительным")
    if int(config.seed) != config.seed or not 0 <= config.seed < 2**32:
        raise ValueError("Seed должен быть целым в диапазоне от 0 до 2³² − 1")

    problem = make_problem(
        config.problem,
        config.n,
        config.top,
        config.bottom,
        config.left,
        config.right,
        config.source,
    )
    points = select_points(config.n, config.mode, config.point)
    grid_reference = reference_solution(problem)
    reference = choose_reference(problem, points, grid_reference)
    reference_values = reference.values[tuple(points.T)]
    reference_value = (
        float(reference_values[0]) if len(points) == 1 else reference_values.tolist()
    )
    if not np.any(reference_values):
        raise ValueError(
            "Эталон в этой точке равен нулю: относительная точность не определена. Выберите другую точку или задачу."
        )
    study = StudyResult(settings, reference.kind, reference.label, reference_value)
    if progress:
        progress(0.0, "Подготовка решателей…")
    warmup()
    started_at = perf_counter()
    walk_budgets = _walk_budgets(settings.max_walks, config.batches)
    for method_index, method in enumerate(METHODS):
        budgets = LOD_SEARCH_TOLERANCES if method == "lod_serial" else walk_budgets

        def measure(budget, budget_index, phase, deadline):
            def report(label):
                if progress:
                    progress(method_index / len(METHODS), label)

            return _measure_budget(
                study,
                problem,
                points,
                method,
                budget,
                budget_index,
                phase,
                deadline,
                report,
                grid_reference[tuple(points.T)],
            )

        rows = search_budgets(
            method, budgets, measure, settings.seconds_per_method, clock=perf_counter
        )
        for row in rows:
            row["name"] = METHODS[method]
        study.rows.extend(rows)
        if progress:
            progress((method_index + 1) / len(METHODS), f"{METHODS[method]}: готово")
    study.rows.sort(key=lambda row: row["target_pct"])
    study.search_seconds = perf_counter() - started_at
    return study


def timing_rows(study):
    """Три строки точности и по одному столбцу времени на метод."""
    return [
        {
            "Относительная ошибка, %": target,
            **{
                f"{row['name']}, с": row["seconds"]
                for row in study.rows
                if row["target_pct"] == target
            },
        }
        for target in TARGETS_PCT
    ]


def export_study(study, folder):
    """Сохраняем таблицу, выбранные бюджеты, все замеры и настройки."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for name, rows in (
        ("timings.csv", timing_rows(study)),
        ("thresholds.csv", study.rows),
        ("trials.csv", study.trials),
    ):
        if rows:
            with (folder / name).open("w", encoding="utf-8-sig", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    metadata = {
        "commit": commit,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "numba": numba.__version__,
        "settings": asdict(study.settings),
        "targets_pct": TARGETS_PCT,
        "reference_kind": study.reference_kind,
        "reference_label": study.reference_label,
        "reference_value": study.reference_value,
        "search_seconds": study.search_seconds,
        **accuracy_metadata(study.rows),
        "timing": "Mean solver time of validation runs; search, reference and JIT excluded; all layer preparation included",
        "seconds_per_phase": study.settings.seconds_per_method,
    }
    (folder / "settings.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
