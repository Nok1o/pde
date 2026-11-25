"""Independent final series, complete-run accounting and the agreed Q metric."""

from types import SimpleNamespace

import numpy as np
import pytest

from pdelab.study_protocol import (
    METRIC_ID,
    PROTOCOL_ID,
    error_metrics,
    run_accuracy_study,
)


def outcome(values, status="completed", **kwargs):
    return SimpleNamespace(
        values=np.atleast_1d(values) if status == "completed" else None,
        status=status,
        reason="ok" if status == "completed" else "deliberate " + status,
        target_time=0.02,
        completed_until=0.02 if status == "completed" else 0.01,
        seconds=kwargs.get("seconds", 1.25),
        preparation_seconds=0.25,
        workspace_bytes=1234,
        peak_rss_bytes=None,
        physical_steps=2 if status == "completed" else 1,
        outer_iterations=5,
        inner_iterations=20,
        walks=kwargs.get("walks", 0),
        transitions=kwargs.get("walks", 0) * 3,
        history=[dict(nonlinear_relative_residual=1e-8, residual_scale=7)],
    )


def test_point_metric_is_mean_absolute_error_of_individual_runs():
    assert error_metrics([9, 9, 13], 10)["relative_error_pct"] == pytest.approx(100 / 6)
    assert error_metrics([8, 12], 10)["relative_error_pct"] == pytest.approx(20)
    assert error_metrics([-9, -9, -13], -10)["relative_error_pct"] == pytest.approx(
        100 / 6
    )


def test_area_metric_is_l2_not_l1_rmse_or_pointwise_percentages():
    metric = error_metrics([[2, 9], [-2, 11]], [0, 10])
    assert metric["relative_error_pct"] == pytest.approx(10 * np.sqrt(5))
    assert metric["accuracy_metric"] == METRIC_ID
    assert metric["absolute_error"] == pytest.approx(np.sqrt(5))


def test_zero_reference_preserves_absolute_l2_and_no_percent_claim():
    metric = error_metrics([[3, 4], [0, 0]], [0, 0])
    assert metric["absolute_error"] == 2.5
    assert metric["relative_error_pct"] is None
    assert not metric["relative_applicable"]
    values = iter(([3, 4], [0, 0]))
    study = run_accuracy_study(
        lambda budget, seed: outcome(next(values)), [0.1, 0.01], [0, 0], repeats=2
    )
    assert len(study["trials"]) == 2
    assert all(
        row["status"] == "relative_target_not_applicable" for row in study["rows"]
    )
    assert all(
        row["absolute_error"] == 2.5 and row["seconds"] is None for row in study["rows"]
    )
    assert study["elapsed_seconds"] == 2.5
    assert study["final_seconds"] == 0


def test_failed_final_does_not_trigger_larger_budget_or_new_selection():
    calls = []

    def runner(budget, seed):
        calls.append((budget, seed))
        return outcome(10.04 if len(calls) <= 2 else 10.2)

    study = run_accuracy_study(runner, [0.1, 0.01], [10], targets=(1,), repeats=2)
    assert [budget for budget, _ in calls] == [0.1] * 4
    assert len({seed for _, seed in calls}) == 4
    row = study["rows"][0]
    assert row["status"] == "validation_failed" and row["seconds"] is None
    assert row["relative_error_pct"] == pytest.approx(2)
    assert row["elapsed_seconds"] == 2.5
    assert study["pilot_seconds"] == 2.5 and study["final_seconds"] == 2.5
    assert study["protocol"] == PROTOCOL_ID


def test_all_candidate_choices_are_frozen_before_any_final_series():
    study = run_accuracy_study(
        lambda budget, seed: outcome(10.3 if budget == 0.1 else 10.04),
        [0.1, 0.01],
        [10],
        repeats=2,
    )
    assert [series["phase"] for series in study["series"]] == [
        "pilot",
        "pilot",
        "final",
        "final",
    ]
    assert [row["budget"] for row in study["rows"]] == [0.01, 0.1, 0.1]
    assert all(row["status"] == "confirmed" for row in study["rows"])
    pilots = {r["seed"] for r in study["trials"] if r["phase"] == "pilot"}
    final = {r["seed"] for r in study["trials"] if r["phase"] == "final"}
    assert pilots.isdisjoint(final)
    assert len(study["trials"]) == 8  # final at 0.1 serves both 5% and 10%
    assert all("values" not in row and "history" not in row for row in study["trials"])
    assert all(row["total_walks"] is None for row in study["rows"])


def test_last_budget_never_silently_drops_the_fixed_margin():
    study = run_accuracy_study(
        lambda budget, seed: outcome(10.09),
        [0.1, 0.01],
        [10],
        targets=(1,),
        repeats=2,
        calibration_fraction=0.8,
    )
    assert study["rows"][0]["status"] == "not_reached"
    assert study["rows"][0]["calibration_target_pct"] == 0.8
    assert not any(series["phase"] == "final" for series in study["series"])
    assert study["rows"][0]["relative_error_pct"] == pytest.approx(0.9)


def test_partial_final_keeps_failures_costs_and_requested_repeats():
    calls = []

    def runner(budget, seed):
        calls.append(seed)
        return outcome(
            10.04, status="outer_limit" if len(calls) == 4 else "completed", walks=11
        )

    study = run_accuracy_study(
        runner, [10, 20], [10], targets=(1,), repeats=2, walks_applicable=True
    )
    row = study["rows"][0]
    assert row["status"] == "validation_incomplete"
    assert row["solver_status"] == "outer_limit"
    assert row["requested_repeats"] == 2 and row["completed_repeats"] == 1
    assert row["relative_error_pct"] is None and row["absolute_error"] is None
    assert row["seconds"] is None and row["elapsed_seconds"] == 2.5
    assert row["total_walks"] == 22 and row["physical_steps"] == 3
    assert row["outer_iterations"] == 10 and row["inner_iterations"] == 40
    assert "outer_limit" in row["reason"]
    assert study["elapsed_seconds"] == 5


@pytest.mark.parametrize(
    "status",
    [
        "linear_limit",
        "outer_limit",
        "walk_budget",
        "path_limit",
        "time_limit",
        "memory_limit",
        "nonfinite",
        "unsupported",
    ],
)
def test_solver_status_is_not_collapsed_into_a_generic_failure(status):
    study = run_accuracy_study(
        lambda budget, seed: outcome(10, status), [0.1], [10], repeats=2
    )
    assert all(
        row["solver_status"] == status and row["completed_repeats"] == 0
        for row in study["rows"]
    )
    assert all(
        row["status"] == "pilot_incomplete" and row["seconds"] is None
        for row in study["rows"]
    )
    assert len(study["trials"]) == 2


@pytest.mark.parametrize(
    "values,reference", [([], [1]), ([[1, 2]], [1]), ([np.nan], [1]), ([1], [np.inf])]
)
def test_metric_rejects_missing_misaligned_or_nonfinite_values(values, reference):
    with pytest.raises(ValueError):
        error_metrics(values, reference)
