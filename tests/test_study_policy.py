"""Accuracy selection must not learn from the final held-out series."""

import pytest

from pdelab.study_policy import (
    SELECTION_POLICY,
    calibration_target,
    display_timing_rows,
    search_budgets,
)


def test_margin_selects_a_larger_budget_before_any_final_checks():
    calls = []

    def measure(budget, index, phase, deadline):
        calls.append((budget, phase))
        error = {64: 0.8, 128: 0.4}[budget] if phase == "calibration" else 0.9
        return dict(relative_error_pct=error, seconds=budget / 1000)

    rows = search_budgets("mc", [64, 128, 256], measure, 60, targets=(1.0,))
    assert calls == [(64, "calibration"), (128, "calibration"), (128, "validation")]
    assert rows[0]["budget"] == 128 and rows[0]["seconds"] == 0.128
    assert rows[0]["calibration_target_pct"] == 0.5
    assert rows[0]["selection_policy"] == SELECTION_POLICY


def test_failed_holdout_is_never_retried_and_reports_actual_error():
    calls = []

    def measure(budget, index, phase, deadline):
        calls.append((budget, phase))
        return dict(
            relative_error_pct=0.4 if phase == "calibration" else 1.4, seconds=0.1
        )

    rows = search_budgets("mc", [64, 128, 524288], measure, 60, targets=(1.0,))
    assert calls == [(64, "calibration"), (64, "validation")]
    assert rows[0]["seconds"] is None and rows[0]["validation_seconds"] == 0.1
    cell = display_timing_rows(rows)[0]["mc"]
    assert "Проверка не прошла" in cell and "N=64" in cell and "1.4%" in cell
    assert "None" not in cell


def test_exhausted_budget_keeps_last_completed_pilot_instead_of_nulls():
    def measure(budget, index, phase, deadline):
        assert phase == "calibration"
        return dict(relative_error_pct=0.7, seconds=0.01)

    rows = search_budgets("mc", [64, 128], measure, 60, targets=(1.0,))
    row = rows[0]
    assert row["budget"] == 128 and row["calibration_error_pct"] == 0.7
    assert row["relative_error_pct"] is None and row["seconds"] is None
    cell = display_timing_rows(rows)[0]["mc"]
    assert "Нет запаса" in cell and "N=128" in cell and "подбор 0.7%" in cell


def test_partial_confirmation_is_not_published_as_success():
    def measure(budget, index, phase, deadline):
        if phase == "validation":
            raise TimeoutError()
        return dict(relative_error_pct=0.1, seconds=0.01)

    row = search_budgets("mc", [64], measure, 1, targets=(1.0,))[0]
    assert row["seconds"] is None and row["relative_error_pct"] is None
    assert row["budget"] == 64 and row["calibration_error_pct"] == 0.1
    assert "проверка не завершена" in row["status"]


@pytest.mark.parametrize("method", ["lod", "lod_serial", "picard"])
def test_deterministic_baseline_is_not_forced_to_over_solve(method):
    assert calibration_target(1.0, method) == 1.0
    row = search_budgets(
        method,
        [0.1],
        lambda *args: dict(relative_error_pct=0.8, seconds=0.01),
        60,
        targets=(1.0,),
    )[0]
    assert row["seconds"] == 0.01


def test_zero_reference_message_is_human_readable():
    rows = [
        dict(
            method="mc",
            target_pct=1.0,
            seconds=None,
            status="zero_reference",
            budget=None,
            relative_error_pct=None,
        )
    ]
    assert "Нулевой эталон" in display_timing_rows(rows)[0]["mc"]


def test_interrupted_next_budget_does_not_relabel_previous_error():
    def measure(budget, index, phase, deadline):
        if budget == 128:
            raise TimeoutError()
        return dict(relative_error_pct=0.8, seconds=0.01)

    row = search_budgets("mc", [64, 128], measure, 60, targets=(1.0,))[0]
    assert row["budget"] == 64
    assert row["calibration_error_pct"] == 0.8
    assert row["interrupted_budget"] == 128
    assert row["relative_error_pct"] is None


def test_nonfinite_validation_does_not_leak_nan_to_exports():
    def measure(budget, index, phase, deadline):
        return dict(
            relative_error_pct=0.4 if phase == "calibration" else float("nan"),
            seconds=0.01,
        )

    row = search_budgets("mc", [64], measure, 60, targets=(1.0,))[0]
    assert row["seconds"] is None and row["relative_error_pct"] is None
    assert row["status_code"] == "validation_incomplete"
