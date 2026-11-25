"""Synthetic contract checks, not benchmark results."""

import csv
import io
import json
from dataclasses import replace

import numpy as np
import pytest

from pdelab.measurement_protocol import (
    Trial,
    export_series,
    query_error,
    run_seed,
    summarize,
)


def trial(repeat=0, values=None):
    return Trial(
        repeat=repeat,
        seed=100 + repeat,
        status="completed",
        reason="",
        values=[10.0] if values is None else values,
        seconds=2.0,
        preparation_seconds=0.5,
        workspace_bytes=1024,
        peak_rss_bytes=None,
        paths_per_estimate=64,
        walks=64,
        transitions=200,
        physical_steps=0,
        outer_iterations=0,
        inner_iterations=0,
        completed_until=None,
    )


def summary(trials, reference=(10,), repeats=2, phase="control", target=5):
    return summarize(
        trials,
        reference,
        requested_repeats=repeats,
        target_pct=target,
        phase=phase,
        stochastic=True,
    )


def test_mean_error_is_neither_error_of_mean_nor_rmse():
    absolute, relative = query_error([[8], [12], [13]], [10])
    assert absolute == pytest.approx(7 / 3)
    assert relative == pytest.approx(70 / 3)
    assert query_error([[-8], [-12], [-13]], [-10]) == (absolute, relative)


def test_field_error_is_l2_of_each_run_not_l1_or_pointwise_percentage():
    # Each difference has norm 5; reference norm 10, while L1 would give 70%.
    absolute, relative = query_error([[3, 14], [-3, 6]], [0, 10])
    assert absolute == 5 and relative == 50


@pytest.mark.parametrize(
    "values,reference",
    [
        ([1, 2], [1]),
        ([[1, 2]], [1]),
        ([[1]], [[1]]),
        ([], [1]),
        ([[np.nan]], [1]),
        ([[1]], [np.inf]),
    ],
)
def test_invalid_shapes_and_nonfinite_values_are_rejected(values, reference):
    with pytest.raises(ValueError):
        query_error(values, reference)


def test_zero_reference_reports_absolute_error_without_confirming():
    result = summary([trial(0, [1]), trial(1, [-1])], reference=[0])
    assert result["status"] == "relative_not_applicable"
    assert result["absolute_l2_error"] == 1
    assert result["relative_error_pct"] is None
    assert result["seconds_to_target"] is None


def test_failed_runs_are_not_filtered_out_and_their_cost_is_retained():
    failed = replace(trial(1), values=None, status="time_limit", reason="deadline")
    result = summary([trial(), failed])
    assert result["status"] == "incomplete"
    assert result["spent_seconds"] == 4
    assert result["walks"] == 128
    assert result["completed_repeats"] == 1
    assert result["relative_error_pct"] is None
    assert result["seconds_to_target"] is None
    assert summary([])["status"] == "incomplete"


def test_failed_holdout_keeps_actual_error_and_cost_without_success_time():
    result = summary([trial(0, [11]), trial(1, [11])])
    assert result["status"] == "not_reached"
    assert result["relative_error_pct"] == pytest.approx(10)
    assert result["spent_seconds"] == 4
    assert result["seconds_to_target"] is None


def test_pilot_cannot_confirm_and_full_control_can():
    trials = [trial(), trial(1)]
    assert summary(trials, phase="pilot")["status"] == "pilot_only"
    result = summary(trials)
    assert result["status"] == "confirmed"
    assert result["seconds_to_target"] == 2
    assert result["seconds_std"] == 0
    assert result["peak_rss_bytes"] is None
    assert result["workspace_bytes"] == 1024


def test_streams_are_stable_and_separate_across_phase_repeat_method_and_attempt():
    seeds = {
        run_seed(20261125, 0, method, attempt, phase, repeat)
        for method in range(2)
        for attempt in range(2)
        for phase in ("pilot", "control")
        for repeat in range(5)
    }
    assert len(seeds) == 40
    assert run_seed(42, 0, 0, 0, "control", 0) == run_seed(42, 0, 0, 0, "control", 0)


def test_duplicate_repeats_seeds_and_mixed_budgets_are_rejected():
    for trials in (
        [trial(), trial()],
        [trial(), replace(trial(1), seed=100)],
        [trial(), replace(trial(1), paths_per_estimate=128)],
    ):
        with pytest.raises(ValueError):
            summary(trials)


def test_trial_rejects_missing_failure_reason_and_invalid_cost():
    for kwargs in (
        dict(values=None, status="time_limit"),
        dict(seconds=-1),
        dict(preparation_seconds=3),
        dict(walks=None),
        dict(walks=-1),
        dict(seconds=np.nan),
    ):
        with pytest.raises(ValueError):
            replace(trial(), **kwargs)


def test_export_preserves_failure_details_large_seed_and_null_memory():
    failed = replace(trial(1), status="outer_limit", reason="Picard limit", values=None)
    good = replace(trial(), seed=2**63 + 123)
    metadata = dict(
        series_id="synthetic-only",
        commit="synthetic",
        dirty=True,
        environment={},
        configuration={},
        reference={},
        query={},
        budget={},
        timing={},
        memory={"reason": "not measured"},
        diagnostics={},
    )
    files = export_series(
        metadata,
        [good, failed],
        [10],
        requested_repeats=2,
        target_pct=5,
        phase="control",
        stochastic=True,
    )
    payload = json.loads(files["series.json"])
    assert payload["trials"][0]["seed"] == good.seed
    assert payload["summary"]["status"] == "incomplete"
    assert payload["summary"]["peak_rss_bytes"] is None
    rows = list(csv.DictReader(io.StringIO(files["trials.csv"].decode("utf-8-sig"))))
    assert len(rows) == 2 and rows[1]["reason"] == "Picard limit"
    assert rows[1]["values_json"] == "null"
    with pytest.raises(ValueError, match="Missing metadata"):
        export_series({}, [], [10], requested_repeats=2, target_pct=5, phase="control")


def test_adaptive_mc_needs_distinct_seeds_even_without_fixed_path_budget():
    trials = [replace(trial(r), seed=42, paths_per_estimate=None) for r in range(2)]
    with pytest.raises(ValueError, match="distinct seeds"):
        summary(trials)
    result = summarize(
        trials,
        [10],
        requested_repeats=2,
        target_pct=1,
        phase="control",
        stochastic=False,
    )
    assert result["status"] == "confirmed"
    assert result["stochastic"] is False
    trials[1] = replace(trials[1], seed=43)
    assert summary(trials)["status"] == "confirmed"


@pytest.mark.parametrize("scale", [1e200, 1e-200, 1e-310])
def test_metric_preserves_large_and_tiny_nonzero_references(scale):
    absolute, relative = query_error([[2 * scale]], [scale])
    assert absolute / scale == pytest.approx(1)
    assert relative == pytest.approx(100)
    absolute, relative = query_error([[2 * scale, 0], [0, 2 * scale]], [scale, 0])
    assert absolute / scale == pytest.approx((1 + np.sqrt(5)) / 2)
    assert relative == pytest.approx(50 * (1 + np.sqrt(5)))


def test_unrepresentable_metric_is_rejected_before_json_export():
    with pytest.raises(ValueError, match="finite float range"):
        query_error([[-1e308]], [1e308])
