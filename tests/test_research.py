import csv
import itertools
import json
from collections import defaultdict
from dataclasses import replace

import numpy as np
import pytest

from pdelab import research
from pdelab.experiment import Config
from pdelab.problem import blank_field
from pdelab.research import StudySettings, accuracy_study, export_study, timing_rows
from pdelab.solvers import METHODS, Result


def test_accuracy_uses_individual_errors_not_error_of_the_average():
    assert research.mean_relative_error([0.0, 50.0], 25.0) == pytest.approx(100.0)
    assert research.mean_relative_error([-24.0, -26.0], -25.0) == pytest.approx(4.0)


def _mock_solver(monkeypatch):
    calls = defaultdict(int)

    def solve(problem, points, method, *, walks, **kwargs):
        calls[(method, walks)] += 1
        value = 25.125
        if method == "lod_serial":
            value = 25.0
        elif walks == 64:
            # Удачный подбор ещё не означает успех: новая серия не проходит.
            value = 25.0 if calls[(method, walks)] <= 2 else 30.0
        values = blank_field(problem)
        values[tuple(points[0])] = value
        return Result(
            method=method,
            values=values,
            stderr=np.zeros_like(values),
            sample_count=np.full_like(values, walks, dtype=int),
            elapsed=0.002 if method == "lod_serial" else walks / 10000,
            walks=walks,
        )

    monkeypatch.setattr(research, "solve", solve)
    monkeypatch.setattr(research, "warmup", lambda: None)


def test_independent_confirmation_is_required_and_mean_runtime_is_reported(monkeypatch):
    _mock_solver(monkeypatch)
    study = accuracy_study(StudySettings(Config(n=8), repeats=2, max_walks=128))
    assert study.reference_kind == "analytic_point"
    assert study.reference_value == 25
    assert len(study.rows) == 21
    assert [row["Относительная ошибка, %"] for row in timing_rows(study)] == [1, 5, 10]

    for row in study.rows:
        if row["method"] == "lod_serial":
            assert row["status"] == "Достигнута"
            assert row["seconds"] == pytest.approx(0.002)
        else:
            assert row["budget"] == 64
            assert row["seconds"] is None
            assert row["relative_error_pct"] == pytest.approx(20)
    assert not any(t["budget"] == 128 for t in study.trials)
    calibration = {t["seed"] for t in study.trials if t["phase"] == "calibration"}
    validation = {t["seed"] for t in study.trials if t["phase"] == "validation"}
    assert calibration.isdisjoint(validation)


def test_failed_confirmation_does_not_produce_a_time(monkeypatch):
    _mock_solver(monkeypatch)
    study = accuracy_study(StudySettings(Config(n=8), repeats=2, max_walks=64))
    for row in study.rows:
        if row["method"] != "lod_serial":
            assert row["seconds"] is None
            assert row["relative_error_pct"] == pytest.approx(20)
            assert row["budget"] == 64
            assert row["status"] == "Не достигнута на контрольной серии"


def test_time_limit_preserves_unreached_cells(monkeypatch):
    _mock_solver(monkeypatch)
    clock = itertools.count()
    monkeypatch.setattr(research, "perf_counter", lambda: next(clock))
    study = accuracy_study(StudySettings(Config(n=8), seconds_per_method=1))
    assert not study.trials
    assert all(row["seconds"] is None for row in study.rows)
    assert all(row["status"] == "Лимит времени поиска" for row in study.rows)


@pytest.mark.parametrize("preset", ["hot_top", "sine"])
def test_real_study_checks_one_point_and_exports_metadata(tmp_path, preset):
    settings = StudySettings(Config(problem=preset, n=8), repeats=2, max_walks=64)
    study = accuracy_study(settings)
    assert len(study.rows) == 3 * len(METHODS)
    assert {trial["method"] for trial in study.trials} == set(METHODS)
    for row in study.rows:
        if row["seconds"] is not None:
            assert row["relative_error_pct"] <= row["target_pct"]
            assert row["calibration_error_pct"] <= row["target_pct"]
    export_study(study, tmp_path)
    metadata = json.loads((tmp_path / "settings.json").read_text())
    assert metadata["settings"]["config"]["n"] == 8
    assert metadata["settings"]["repeats"] == 2
    assert metadata["reference_kind"] == (
        "analytic" if preset == "sine" else "analytic_point"
    )
    with (tmp_path / "timings.csv").open(encoding="utf-8-sig") as stream:
        table = list(csv.DictReader(stream))
    assert len(table) == 3 and len(table[0]) == 8


def test_reference_falls_back_to_grid_for_noncentral_point(monkeypatch):
    _mock_solver(monkeypatch)
    study = accuracy_study(
        StudySettings(Config(n=8, point=(2, 3)), repeats=2, max_walks=64)
    )
    assert study.reference_kind == "grid"


def test_zero_reference_is_not_silently_replaced_with_an_absolute_target():
    with pytest.raises(ValueError, match="равен нулю"):
        accuracy_study(StudySettings(Config(n=8, top=0.0)))


@pytest.mark.parametrize(
    "overrides",
    [
        {"repeats": 1},
        {"repeats": 2.5},
        {"max_walks": 1},
        {"seconds_per_method": 0},
        {"seconds_per_method": float("nan")},
    ],
)
def test_invalid_study_settings_are_rejected(overrides):
    with pytest.raises(ValueError):
        accuracy_study(replace(StudySettings(Config(n=8)), **overrides))


def test_real_center_48_reaches_one_percent_with_margin(monkeypatch):
    monkeypatch.setattr(research, "METHODS", {"mc": "Монте-Карло", "reuse": "Хвосты"})
    study = accuracy_study(
        StudySettings(
            Config(problem="sine", n=48, point=(24, 24)),
            repeats=5,
            max_walks=32768,
            seconds_per_method=60,
        )
    )
    for row in study.rows:
        assert row["seconds"] is not None
        assert row["calibration_error_pct"] <= row["target_pct"] / 2
        assert row["relative_error_pct"] <= row["target_pct"]
        if row["target_pct"] == 1:
            assert row["budget"] > 2048
