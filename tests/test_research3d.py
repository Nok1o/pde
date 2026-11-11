import json
from dataclasses import replace

import numpy as np
import pytest

from pdelab import research3d
from pdelab.experiment3d import Config3D
from pdelab.problem3d import blank_field3d, select_points3d
from pdelab.research3d import StudySettings3D, accuracy_study3d, export_study3d
from pdelab.solvers import Result


def test_failed_validation_retries_with_larger_budget_and_fresh_streams(monkeypatch):
    from collections import defaultdict

    calls = defaultdict(int)

    def fake_solve(problem, points, method, *, walks, **kwargs):
        calls[(method, walks)] += 1
        values = blank_field3d(problem)
        fails = method != "lod_serial" and walks == 64 and calls[(method, walks)] > 6
        values[tuple(points.T)] = 1.2 if fails else 1.0
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
    assert all(row["seconds"] is not None for row in study.rows)
    for row in study.rows:
        if row["method"] != "lod_serial":
            assert row["budget"] == 128
            assert row["validation_attempts"] == 2
    for method in ("mc", "graded", "wos"):
        assert calls[(method, 64)] == calls[(method, 128)] == 8
        trials = [trial for trial in study.trials if trial["method"] == method]
        assert len({trial["seed"] for trial in trials}) == len(trials)


def test_real_study_and_export(tmp_path):
    study = accuracy_study3d(
        StudySettings3D(Config3D(problem="constant", n=8), repeats=2, max_walks=64)
    )
    assert len(study.rows) == 12
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


def test_stricter_lod_failure_preserves_reachable_target():
    study = accuracy_study3d(
        StudySettings3D(Config3D(n=8, max_iter=11), repeats=2, methods=("lod_serial",))
    )
    loose = next(row for row in study.rows if row["target_pct"] == 10)
    assert loose["budget"] == 0.1
    assert loose["seconds"] is not None
    assert loose["relative_error_pct"] < 10
    assert len([t for t in study.trials if t["phase"] == "validation"]) == 2
    assert all(row["seconds"] is None for row in study.rows if row["target_pct"] < 10)


def test_calibration_timeout_leaves_a_separate_confirmation_allowance(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(research3d, "perf_counter", lambda: clock[0])
    monkeypatch.setattr(research3d, "warmup3d", lambda: None)

    def series(study, problem, points, method, budget, budget_index, phase, deadline):
        if phase == "calibration" and budget_index == 0:
            clock[0] = 2.0
            return 7.0, 0.01
        if phase == "validation":
            assert clock[0] < deadline
            return 7.0, 0.02
        assert clock[0] >= deadline
        raise TimeoutError("Лимит времени поиска")

    monkeypatch.setattr(research3d, "_series3d", series)
    study = accuracy_study3d(
        StudySettings3D(
            Config3D(n=8), repeats=2, seconds_per_method=1, methods=("lod_serial",)
        )
    )
    loose = next(row for row in study.rows if row["target_pct"] == 10)
    assert loose["budget"] == 0.1 and loose["seconds"] == 0.02
    assert loose["relative_error_pct"] == 7.0
    assert loose["status"] == "Достигнута"


@pytest.mark.parametrize(
    "mode,plane,index",
    [("patch", "xy", None), ("field", "xy", None)]
    + [("slice", plane, 2) for plane in ("xy", "xz", "yz")],
)
def test_region_study_uses_every_point_and_exports_coordinates(
    monkeypatch, tmp_path, mode, plane, index
):
    config = Config3D(problem="affine", n=4, mode=mode, plane=plane, index=index)
    expected_points = select_points3d(4, mode, plane=plane, index=index)
    offsets = np.linspace(-0.04, 0.04, len(expected_points))
    offsets[0] = 0  # An exact first point must not hide errors in the region.

    def fake_solve(problem, points, method, **kwargs):
        np.testing.assert_array_equal(points, expected_points)
        values = blank_field3d(problem)
        values[tuple(points.T)] = problem.exact[tuple(points.T)] + offsets
        return Result(
            method,
            values,
            np.zeros_like(values),
            np.zeros_like(values, dtype=int),
            elapsed=0.125,
        )

    monkeypatch.setattr(research3d, "solve3d", fake_solve)
    monkeypatch.setattr(research3d, "warmup3d", lambda: None)
    study = accuracy_study3d(
        StudySettings3D(config, repeats=2, max_walks=64, methods=("mc",))
    )
    expected_error = 100 * np.sum(np.abs(offsets)) / np.sum(study.reference_value)
    assert expected_error > 0
    assert [row["target_pct"] for row in study.rows] == [1, 5, 10]
    for row in study.rows:
        assert row["relative_error_pct"] == pytest.approx(expected_error)
        assert row["seconds"] == pytest.approx(0.125)
    assert all(len(trial["value"]) == len(expected_points) for trial in study.trials)
    export_study3d(study, tmp_path)
    payload = json.loads((tmp_path / "settings.json").read_text())
    assert payload["points_zyx"] == expected_points.tolist()
    assert payload["reference_value"] == study.reference_value


@pytest.mark.parametrize("mode", ["patch", "slice", "field"])
def test_real_region_study_all_methods(mode):
    study = accuracy_study3d(
        StudySettings3D(
            Config3D(problem="constant", n=4, mode=mode),
            repeats=2,
            max_walks=64,
        )
    )
    assert len(study.rows) == 3 * len(research3d.METHODS_3D)
    assert all(row["status"] == "Достигнута" for row in study.rows)
    assert all(row["seconds"] is not None for row in study.rows)


def test_zero_reference_region_is_rejected():
    with pytest.raises(ValueError, match="равен нулю"):
        accuracy_study3d(
            StudySettings3D(Config3D(n=8, mode="slice", plane="yz", index=0))
        )
