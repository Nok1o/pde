import json
from collections import defaultdict
from dataclasses import replace

import numpy as np
import pytest

from pdelab import research3d
from pdelab.experiment3d import Config3D
from pdelab.problem3d import blank_field3d, select_points3d
from pdelab.research3d import (
    Study3D,
    StudySettings3D,
    accuracy_study3d,
    export_study3d,
    study_payload3d,
)
from pdelab.solvers import Result
from pdelab.study_protocol import METRIC_DESCRIPTION, METRIC_ID, PROTOCOL_ID


def _result(problem, points, method, value, *, elapsed=0.001):
    values = blank_field3d(problem)
    values[tuple(points.T)] = value
    return Result(
        method,
        values,
        np.zeros_like(values),
        np.zeros_like(values, dtype=int),
        elapsed=elapsed,
    )


def test_failed_final_never_retunes_budget_and_keeps_fresh_streams(monkeypatch):
    calls = defaultdict(list)

    def fake_solve(problem, points, method, *, walks, tol, seed, **kwargs):
        budget = tol if method == "lod_serial" else walks
        calls[method].append((budget, seed))
        value = 1.0 if len(calls[method]) <= 2 else 1.2
        return _result(problem, points, method, value)

    monkeypatch.setattr(research3d, "solve3d", fake_solve)
    monkeypatch.setattr(research3d, "warmup3d", lambda: None)
    study = accuracy_study3d(StudySettings3D(Config3D(n=8), repeats=2, max_walks=128))
    assert len(study.rows) == 18
    assert all(row["seconds"] is None for row in study.rows)
    assert all(row["status_code"] == "validation_failed" for row in study.rows)
    for method in research3d.METHODS_3D:
        expected_budget = 0.1 if method == "lod_serial" else 64
        assert [budget for budget, _ in calls[method]] == [expected_budget] * 4
        assert len({seed for _, seed in calls[method]}) == 4
        trials = [trial for trial in study.trials if trial["method"] == method]
        assert [trial["phase"] for trial in trials] == ["pilot"] * 2 + ["final"] * 2
        assert [trial["value"] for trial in trials] == [1.0, 1.0, 1.2, 1.2]
        series = [series for series in study.series if series["method"] == method]
        assert [series["phase"] for series in series] == ["pilot", "final"]


def test_real_study_and_export(tmp_path):
    study = accuracy_study3d(
        StudySettings3D(Config3D(problem="constant", n=8), repeats=2, max_walks=64)
    )
    assert len(study.rows) == 18
    assert all(row["status"] == "Достигнута" for row in study.rows)
    for row in study.rows:
        matching = [
            trial["seconds"]
            for trial in study.trials
            if trial["method"] == row["method"]
            and trial["candidate"] == row["candidate"]
            and trial["phase"] == "final"
        ]
        assert row["seconds"] == pytest.approx(np.mean(matching))
        assert row["protocol"] == PROTOCOL_ID
        assert row["accuracy_metric"] == METRIC_ID
    export_study3d(study, tmp_path)
    for name in (
        "timings.csv",
        "thresholds.csv",
        "trials.csv",
        "series.csv",
        "settings.json",
    ):
        assert (tmp_path / name).exists()
    payload = json.loads((tmp_path / "settings.json").read_text())
    assert payload["protocol"] == payload["selection"] == PROTOCOL_ID
    assert payload["selection_policy"] == PROTOCOL_ID
    assert payload["accuracy_metric"] == METRIC_ID
    assert payload["accuracy"] == METRIC_DESCRIPTION
    assert payload["calibration_fraction"] == 0.8
    assert payload["settings"]["config"]["batch_size"] == 128


def test_zero_exact_value_reports_absolute_diagnostics():
    study = accuracy_study3d(
        StudySettings3D(Config3D(n=4, point=(4, 2, 2)), repeats=2, max_walks=64)
    )
    assert all(
        row["status_code"] == "relative_target_not_applicable" for row in study.rows
    )
    assert all(
        row["seconds"] is None and row["relative_error_pct"] is None
        for row in study.rows
    )
    assert all(row["absolute_error"] == 0.0 for row in study.rows)
    assert all(trial["phase"] == "absolute_diagnostic" for trial in study.trials)
    assert len(study.trials) == 2 * len(research3d.METHODS_3D)
    assert not study_payload3d(study)["relative_applicable"]


def test_time_limit_before_run_records_no_fictitious_trials(monkeypatch):
    clock = iter(range(1000))
    monkeypatch.setattr(research3d, "perf_counter", lambda: next(clock))
    monkeypatch.setattr(research3d, "warmup3d", lambda: None)

    def unexpected_solve(*args, **kwargs):
        pytest.fail("The exhausted pilot allowance must prevent a new solve")

    monkeypatch.setattr(research3d, "solve3d", unexpected_solve)
    study = accuracy_study3d(StudySettings3D(Config3D(n=8), seconds_per_method=0.1))
    assert all(
        row["seconds"] is None
        and row["status_code"] == "pilot_incomplete"
        and row["solver_status"] == "time_limit"
        for row in study.rows
    )
    assert study.trials == []
    assert all(series["recorded_repeats"] == 0 for series in study.series)


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


def test_stricter_lod_failure_preserves_reachable_target_and_failed_trials():
    study = accuracy_study3d(
        StudySettings3D(Config3D(n=8, max_iter=11), repeats=2, methods=("lod_serial",))
    )
    loose = next(row for row in study.rows if row["target_pct"] == 10)
    assert loose["budget"] == 0.1
    assert loose["seconds"] is not None
    assert loose["relative_error_pct"] < 10
    assert len([trial for trial in study.trials if trial["phase"] == "final"]) == 2
    assert all(row["seconds"] is None for row in study.rows if row["target_pct"] < 10)
    failed = [trial for trial in study.trials if trial["status"] == "linear_limit"]
    assert failed and all(trial["values"] is None for trial in failed)
    assert sum(series["recorded_repeats"] for series in study.series) == len(
        study.trials
    )


def test_pilot_timeout_leaves_a_fresh_final_allowance(monkeypatch):
    clock, calls = [0.0], []
    monkeypatch.setattr(research3d, "perf_counter", lambda: clock[0])
    monkeypatch.setattr(research3d, "warmup3d", lambda: None)

    def fake_solve(problem, points, method, *, tol, **kwargs):
        calls.append((tol, clock[0]))
        # The second pilot starts before the deadline and finishes after it.
        # Its completed series can select 10%; tighter pilots cannot start.
        elapsed = 0.6 if len(calls) <= 2 else 0.1
        clock[0] += elapsed
        return _result(problem, points, method, 1.07, elapsed=elapsed)

    monkeypatch.setattr(research3d, "solve3d", fake_solve)
    study = accuracy_study3d(
        StudySettings3D(
            Config3D(n=8), repeats=2, seconds_per_method=1, methods=("lod_serial",)
        )
    )
    loose = next(row for row in study.rows if row["target_pct"] == 10)
    assert loose["budget"] == 0.1 and loose["seconds"] == 0.1
    assert loose["relative_error_pct"] == pytest.approx(7.0)
    assert loose["status_code"] == "confirmed"
    assert len(calls) == 4 and calls[2][1] == pytest.approx(1.2)
    assert [series["phase"] for series in study.series] == ["pilot", "pilot", "final"]
    assert study.series[1]["solver_status"] == "time_limit"
    assert study.series[1]["recorded_repeats"] == 0


def test_failed_final_attempt_is_retained_with_its_time_and_no_retry(monkeypatch):
    clock, calls = [0.0], []
    monkeypatch.setattr(research3d, "perf_counter", lambda: clock[0])
    monkeypatch.setattr(research3d, "warmup3d", lambda: None)

    def fake_solve(problem, points, method, *, walks, **kwargs):
        calls.append(walks)
        if len(calls) == 3:
            clock[0] += 0.25
            raise RuntimeError("deliberate incomplete path")
        clock[0] += 0.1
        return _result(problem, points, method, 1.0, elapsed=0.1)

    monkeypatch.setattr(research3d, "solve3d", fake_solve)
    study = accuracy_study3d(
        StudySettings3D(Config3D(n=4), repeats=2, max_walks=128, methods=("mc",))
    )
    assert calls == [64] * 4
    assert len(study.trials) == 4
    failed = study.trials[2]
    assert failed["status"] == "solver_failed" and failed["values"] is None
    assert failed["steps"] is None and failed["walks"] is None
    assert failed["seconds"] == pytest.approx(0.25)
    assert all(row["status_code"] == "validation_incomplete" for row in study.rows)
    assert all(
        row["seconds"] is None and row["relative_error_pct"] is None
        for row in study.rows
    )
    final = study.series[-1]
    assert final["recorded_repeats"] == 2 and final["completed_repeats"] == 1
    assert final["elapsed_seconds"] == pytest.approx(0.35)
    assert sum(series["elapsed_seconds"] for series in study.series) == pytest.approx(
        0.55
    )


@pytest.mark.parametrize(
    "mode,plane,index",
    [("patch", "xy", None), ("field", "xy", None)]
    + [("slice", plane, 2) for plane in ("xy", "xz", "yz")],
)
def test_region_study_uses_l2_every_point_and_exports_coordinates(
    monkeypatch, tmp_path, mode, plane, index
):
    config = Config3D(problem="affine", n=4, mode=mode, plane=plane, index=index)
    expected_points = select_points3d(4, mode, plane=plane, index=index)
    offsets = np.linspace(-0.04, 0.04, len(expected_points))
    offsets[0] = 0  # An exact first point must not hide errors in the region.

    def fake_solve(problem, points, method, **kwargs):
        np.testing.assert_array_equal(points, expected_points)
        value = problem.exact[tuple(points.T)] + offsets
        return _result(problem, points, method, value, elapsed=0.125)

    monkeypatch.setattr(research3d, "solve3d", fake_solve)
    monkeypatch.setattr(research3d, "warmup3d", lambda: None)
    study = accuracy_study3d(
        StudySettings3D(config, repeats=2, max_walks=64, methods=("mc",))
    )
    expected_error = (
        100 * np.linalg.norm(offsets) / np.linalg.norm(study.reference_value)
    )
    old_l1_error = 100 * np.sum(np.abs(offsets)) / np.sum(study.reference_value)
    assert expected_error > 0 and not np.isclose(expected_error, old_l1_error)
    assert [row["target_pct"] for row in study.rows] == [1, 5, 10]
    for row in study.rows:
        assert row["relative_error_pct"] == pytest.approx(expected_error)
        assert row["seconds"] == pytest.approx(0.125)
    assert all(len(trial["values"]) == len(expected_points) for trial in study.trials)
    assert all(trial["value"] == trial["values"] for trial in study.trials)
    export_study3d(study, tmp_path)
    payload = json.loads((tmp_path / "settings.json").read_text())
    assert payload["points_zyx"] == expected_points.tolist()
    assert payload["reference_value"] == study.reference_value
    assert payload["accuracy_metric"] == METRIC_ID


def test_error_is_mean_of_run_errors_not_error_of_the_mean(monkeypatch):
    calls = []

    def fake_solve(problem, points, method, **kwargs):
        calls.append(kwargs)
        value = 7 * (0.95 if len(calls) % 2 else 1.05)
        return _result(problem, points, method, value)

    monkeypatch.setattr(research3d, "solve3d", fake_solve)
    monkeypatch.setattr(research3d, "warmup3d", lambda: None)
    study = accuracy_study3d(
        StudySettings3D(
            Config3D(problem="constant", n=4), repeats=2, max_walks=64, methods=("mc",)
        )
    )
    assert all(row["relative_error_pct"] == pytest.approx(5) for row in study.rows)
    assert [row["status_code"] for row in study.rows] == [
        "not_reached",
        "not_reached",
        "confirmed",
    ]


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


def test_zero_reference_region_keeps_absolute_l2_diagnostics():
    study = accuracy_study3d(
        StudySettings3D(
            Config3D(n=4, mode="slice", plane="yz", index=0),
            repeats=2,
            max_walks=64,
            methods=("mc", "mc_grouped"),
        )
    )
    assert all(
        row["status_code"] == "relative_target_not_applicable" for row in study.rows
    )
    assert all(
        row["absolute_error"] == 0 and row["relative_error_pct"] is None
        for row in study.rows
    )
    assert all(len(trial["values"]) == 9 for trial in study.trials)


def test_config_and_progress_are_forwarded_to_shared_service(monkeypatch):
    seen = []
    progress = []

    def fake_solve(problem, points, method, **kwargs):
        seen.append((method, kwargs))
        assert kwargs["batch_size"] == 7 and kwargs["max_iter"] == 81
        assert kwargs.get("coarse_n") == (4 if method == "mc_control" else None)
        return _result(problem, points, method, 7)

    monkeypatch.setattr(research3d, "solve3d", fake_solve)
    monkeypatch.setattr(research3d, "warmup3d", lambda: None)
    study = accuracy_study3d(
        StudySettings3D(
            Config3D(problem="constant", n=8, batch_size=7, coarse_n=4, max_iter=81),
            repeats=2,
            max_walks=37,
            methods=("mc_grouped", "mc_control"),
        ),
        progress=lambda fraction, message: progress.append((fraction, message)),
    )
    assert seen and all(kwargs["walks"] == 37 for _, kwargs in seen)
    grouped = [trial for trial in study.trials if trial["method"] == "mc_grouped"]
    assert all(trial["batch_size"] == 7 for trial in grouped)
    assert progress[-1][0] == 1 and all(0 <= fraction <= 1 for fraction, _ in progress)


@pytest.mark.parametrize(
    "row,metric,protocol",
    [
        ({}, "relative_rmse_v1", "legacy"),
        (
            {
                "accuracy_metric": "mean_run_relative_l1_v1",
                "selection_policy": "stable_pilot_adaptive_retry_v6",
            },
            "mean_run_relative_l1_v1",
            "stable_pilot_adaptive_retry_v6",
        ),
    ],
)
def test_old_rows_are_not_relabelled_with_new_protocol(row, metric, protocol):
    study = Study3D(StudySettings3D(Config3D(n=4)), "analytic", 1.0, rows=[row])
    payload = study_payload3d(study)
    assert payload["accuracy_metric"] == metric and payload["protocol"] == protocol
    assert payload["accuracy"] != METRIC_DESCRIPTION
    assert study.series == []
