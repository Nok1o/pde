from dataclasses import replace

import numpy as np
import pytest

from pdelab import research3d
from pdelab.experiment3d import Config3D
from pdelab.problem3d import blank_field3d
from pdelab.research3d import StudySettings3D, accuracy_study3d, export_study3d
from pdelab.solvers import Result


def test_validation_failure_never_restarts_selection(monkeypatch):
    calls = {"mc": 0, "lod_serial": 0}

    def fake_solve(problem, points, method, **kwargs):
        calls[method] += 1
        values = blank_field3d(problem)
        values[tuple(points.T)] = (
            1.0 if calls[method] <= (2 if method == "mc" else 1) else 1.2
        )
        return Result(
            method,
            values,
            np.zeros_like(values),
            np.zeros_like(values, dtype=int),
            elapsed=0.001,
        )

    monkeypatch.setattr(research3d, "solve3d", fake_solve)
    monkeypatch.setattr(research3d, "warmup3d", lambda: None)
    study = accuracy_study3d(StudySettings3D(Config3D(n=8), repeats=2, max_walks=128))
    assert all(row["seconds"] is None for row in study.rows)
    assert all(
        row["status"] == "Не достигнута на контрольной серии" for row in study.rows
    )
    assert calls == {"mc": 4, "lod_serial": 3}
    for method in calls:
        calibration = {
            trial["seed"]
            for trial in study.trials
            if trial["method"] == method and trial["phase"] == "calibration"
        }
        validation = {
            trial["seed"]
            for trial in study.trials
            if trial["method"] == method and trial["phase"] == "validation"
        }
        assert calibration.isdisjoint(validation)


def test_real_study_and_export(tmp_path):
    study = accuracy_study3d(
        StudySettings3D(Config3D(problem="constant", n=8), repeats=2, max_walks=64)
    )
    assert len(study.rows) == 6
    assert all(row["status"] == "Достигнута" for row in study.rows)
    for row in study.rows:
        matching = [
            trial["seconds"]
            for trial in study.trials
            if trial["method"] == row["method"]
            and trial["budget"] == row["budget"]
            and trial["phase"] == "validation"
        ]
        assert row["seconds"] == pytest.approx(np.mean(matching))
    export_study3d(study, tmp_path)
    assert (tmp_path / "thresholds.csv").exists() and (
        tmp_path / "settings.json"
    ).exists()


def test_zero_exact_value_and_time_limit(monkeypatch):
    with pytest.raises(ValueError, match="равен нулю"):
        accuracy_study3d(StudySettings3D(Config3D(n=8, point=(8, 4, 4))))
    clock = iter(range(1000))
    monkeypatch.setattr(research3d, "perf_counter", lambda: next(clock))
    study = accuracy_study3d(StudySettings3D(Config3D(n=8), seconds_per_method=0.1))
    assert all(
        row["seconds"] is None and row["status"] == "Лимит времени поиска"
        for row in study.rows
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"repeats": 1},
        {"max_walks": 1.5},
        {"seconds_per_method": 0},
        {"methods": ()},
        {"methods": ("mc", "mc")},
        {"methods": ("reuse",)},
    ],
)
def test_bad_settings(changes):
    with pytest.raises(ValueError):
        accuracy_study3d(replace(StudySettings3D(Config3D(n=8)), **changes))
