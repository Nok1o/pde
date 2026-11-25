"""Подбор бюджета и независимая проверка точности на выбранной области 3D."""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from time import perf_counter

import numpy as np

from .experiment3d import Config3D, comparison3d, rows_csv
from .problem3d import make_problem3d, select_points3d
from .solvers3d import METHODS_3D, solve3d, warmup3d
from .stationary_study3d import study_stationary3d
from .study_protocol import (
    ABSOLUTE_METRIC_ID,
    METRIC_DESCRIPTION,
    METRIC_ID,
    PROTOCOL_ID,
    error_metrics,
)

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
    reference_value: float | list[float]
    rows: list[dict] = field(default_factory=list)
    trials: list[dict] = field(default_factory=list)
    series: list[dict] = field(default_factory=list)


def _budgets3d(method, maximum):
    if method == "lod_serial":
        return [0.1, 0.03, 0.01, 0.003, 0.001, 1e-4, 1e-6, 1e-8, 1e-10]
    result = [min(64, maximum)]
    while result[-1] < maximum:
        result.append(min(2 * result[-1], maximum))
    return result


def accuracy_study3d(settings, progress=None):
    config = settings.config
    methods = tuple(settings.methods)
    if (
        not methods
        or len(set(methods)) != len(methods)
        or any(method not in METHODS_3D for method in methods)
    ):
        raise ValueError("Для исследования выберите различные доступные методы 3D")
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
    points = select_points3d(
        config.n, config.mode, config.point, config.plane, config.index
    )
    reference, kind, _ = comparison3d(problem)
    reference_values = reference[tuple(points.T)]
    target_value = (
        float(reference_values[0]) if len(points) == 1 else reference_values.tolist()
    )
    study = Study3D(settings, kind, target_value)
    warmup3d()
    result = study_stationary3d(
        problem,
        points,
        reference_values,
        methods,
        walk_budgets=_budgets3d("mc", int(settings.max_walks)),
        tolerances=_budgets3d("lod_serial", settings.max_walks),
        repeats=int(settings.repeats),
        seed=int(config.seed),
        batch_size=config.batch_size,
        coarse_n=config.coarse_n,
        max_iter=config.max_iter,
        seconds_per_phase=settings.seconds_per_method,
        solver=solve3d,
        clock=perf_counter,
        progress=progress,
    )
    study.rows = result["rows"]
    study.trials = result["trials"]
    study.series = result["series"]
    for trial in study.trials:
        trial["reference_value"] = target_value
        # Keep the old scalar CSV field alongside the protocol's query vector.
        if len(points) == 1 and trial["values"] is not None:
            trial["value"] = trial["values"][0]
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
    reference_info = error_metrics(study.reference_value, study.reference_value)
    row = study.rows[0] if study.rows else {}
    metric = row.get("accuracy_metric", "relative_rmse_v1")
    protocol = row.get(
        "protocol",
        row.get("selection_policy", PROTOCOL_ID if metric == METRIC_ID else "legacy"),
    )
    descriptions = {
        METRIC_ID: METRIC_DESCRIPTION,
        "mean_run_relative_l1_v1": (
            "100 * mean_runs(sum_points(abs(u_run-reference)) / "
            "sum_points(abs(reference))); error of individual runs in the validation series"
        ),
        "relative_rmse_v1": (
            "100 * RMS(u_run-reference) / RMS(reference); individual runs"
        ),
    }
    return {
        "settings": asdict(study.settings),
        "dimension": 3,
        "coordinate_order": ["x", "y", "z"],
        "points_zyx": select_points3d(
            study.settings.config.n,
            study.settings.config.mode,
            study.settings.config.point,
            study.settings.config.plane,
            study.settings.config.index,
        ).tolist(),
        "reference_kind": study.reference_kind,
        "reference_value": study.reference_value,
        "protocol": protocol,
        "selection": protocol,
        "selection_policy": protocol,
        "accuracy_metric": metric,
        "accuracy": descriptions.get(metric, "Metric recorded in threshold rows"),
        "absolute_metric": row.get(
            "absolute_metric", ABSOLUTE_METRIC_ID if metric == METRIC_ID else None
        ),
        "calibration_fraction": row.get("calibration_fraction"),
        "relative_applicable": reference_info["relative_applicable"],
        "reference_norm": reference_info["reference_norm"],
        "seconds_per_phase": study.settings.seconds_per_method,
        "timing": "Mean wall time of complete final runs after warmup; search and reference excluded",
    }


def export_study3d(study, folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for name, rows in (
        ("timings", timing_rows3d(study)),
        ("thresholds", study.rows),
        ("trials", study.trials),
        ("series", study.series),
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
