"""Zero-reference diagnostics must retain work without confirming percentages."""

import csv
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from pdelab.boundary_experiment import BoundaryConfig
from pdelab.boundary_research import boundary_accuracy_study
from pdelab.heat import HeatProblem
from pdelab.heat_mc_study import export_heat_mc_study, heat_mc_study
from pdelab.nonlinear import NonlinearSettings, make_nonlinear_problem
from pdelab.nonlinear_study import nonlinear_targets
from pdelab.study_policy import (
    SolverSeriesError,
    display_timing_rows,
    search_budgets,
    study_error_metrics,
)


def zero_heat():
    return HeatProblem.from_functions(
        8,
        lambda x, y: 0,
        lambda x, y, t: 0,
        lambda x, y, t: 0,
        exact=lambda x, y, t: 0,
    )


def test_zero_query_reports_mean_l2_without_relative_relabelling():
    metrics = study_error_metrics([[3, 4], [-3, -4]], [0, 0])
    assert metrics["absolute_error"] == 5
    assert metrics["absolute_metric"] == "mean_run_absolute_l2_v1"
    assert metrics["relative_error_pct"] is None
    assert not metrics["relative_applicable"]
    # Nonzero legacy studies still use their original relative L1 metric.
    metrics = study_error_metrics([[3, 5]], [0, 1])
    assert metrics["relative_error_pct"] == 700
    assert metrics["absolute_error"] == 5


def test_zero_diagnostic_never_searches_or_validates_a_percent_target():
    calls = []

    def measure(budget, index, phase, deadline):
        calls.append((budget, phase))
        return dict(
            **study_error_metrics([3, -5], 0),
            seconds=0.2,
            requested_repeats=2,
            completed_repeats=2,
        )

    rows = search_budgets("mc", [64, 256], measure, 60, relative_applicable=False)
    assert calls == [(64, "absolute_diagnostic")]
    for row in rows:
        assert row["status_code"] == "relative_target_not_applicable"
        assert row["absolute_error"] == 4
        assert row["seconds"] is None and row["validation_attempts"] == 0
        assert row["budget"] == 64 and row["diagnostic_seconds"] == 0.2
    cell = display_timing_rows(rows)[0]["mc"]
    assert "абсолютная L2-ошибка 4" in cell and "None" not in cell


def test_failed_numerical_zero_reference_stops_without_a_retry():
    calls = []

    def measure(budget, *args):
        calls.append(budget)
        raise SolverSeriesError("walk_budget", relative_applicable=False)

    rows = search_budgets("mc", [64, 256], measure, 60)
    assert calls == [64]
    for row in rows:
        assert row["status_code"] == "relative_target_not_applicable"
        assert row["solver_status"] == "solver_failed"
        assert row["interrupted_budget"] == 64
        assert row["absolute_error"] is None
        assert row["attempts"][0]["status"] == "solver_failed"
        assert "walk_budget" in row["reason"]


def test_heat_partial_zero_series_retains_failure_costs_and_unknown_counters(
    monkeypatch, tmp_path
):
    import pdelab.heat_mc_study as module

    calls = []

    def estimate(*args, walks, **kwargs):
        calls.append(walks)
        if len(calls) == 2:
            raise RuntimeError("transition limit")
        return SimpleNamespace(
            values=np.array([2.0]),
            seconds=0.01,
            preparation_seconds=0.002,
            spatial_transitions=10,
            temporal_transitions=4,
            memory_bytes=100,
        )

    monkeypatch.setattr(module, "solve_heat_mc", estimate)
    study = heat_mc_study(zero_heat(), repeats=3, max_walks=256, methods=("mc", "lod"))
    assert calls == [64, 64]
    failed = [t for t in study["trials"] if t["status"] != "completed"]
    assert len(failed) == 1 and failed[0]["seconds"] >= 0
    assert failed[0]["transitions"] is None and failed[0]["walks"] is None
    assert failed[0]["value"] is None
    for row in study["rows"]:
        assert row["status"] == "relative_target_not_applicable"
        assert row["seconds"] is None
        if row["method"] == "mc":
            assert row["recorded_repeats"] == 2 and row["completed_repeats"] == 1
            assert row["requested_repeats"] == 3 and row["absolute_error"] is None
            assert row["spent_seconds"] >= 0.01
            assert row["total_walks"] is None
    export_heat_mc_study(study, tmp_path)
    with (tmp_path / "trials.csv").open(encoding="utf-8-sig") as stream:
        exported = list(csv.DictReader(stream))
    assert len(exported) == 5 and exported[1]["transitions"] == ""
    assert exported[1]["status"] == "solver_failed"


def test_mixed_boundary_zero_case_measures_absolute_error_and_exports_cost():
    study = boundary_accuracy_study(
        BoundaryConfig(problem="custom", n=4),
        ["lod_serial", "mc"],
        repeats=2,
        budgets=[64, 256],
        tolerances=[1e-6],
    )
    assert len(study["trials"]) == 4
    for row in study["rows"]:
        assert row["status"] == "relative_target_not_applicable"
        assert row["absolute_error"] == 0 and row["seconds"] is None
        assert row["completed_repeats"] == row["requested_repeats"] == 2
        assert row["spent_seconds"] >= 0


@pytest.mark.parametrize("analytic", [True, False])
@pytest.mark.parametrize("fail", [True, False])
def test_nonlinear_zero_reference_with_complete_and_failed_histories(analytic, fail):
    problem = replace(
        make_nonlinear_problem(4),
        initial=np.zeros((5, 5)),
        source=lambda x, y, t: 0,
        boundary=lambda x, y, t: 0,
        exact=(lambda x, y, t: 0) if analytic else None,
    )
    if fail:
        # A zero boundary query still requires a nontrivial full time solve.
        def exact(x, y, t):
            return t * x * (1 - x) * y * (1 - y)

        def source(x, y, t):
            return x * (1 - x) * y * (1 - y) + 2 * t * (x * (1 - x) + y * (1 - y))

        problem = replace(
            problem, alpha=0, source=source, exact=exact if analytic else None
        )
    settings = NonlinearSettings(max_total_walks=1) if fail else None
    rows, evidence = nonlinear_targets(
        problem,
        T=0.01,
        repeats=2,
        budgets=[8, 32],
        point=(0, 0),
        settings=settings,
    )
    assert len(evidence["pilots"]) == 1 and not evidence["validations"]
    for row in rows:
        assert row["status"] == "relative_target_not_applicable"
        assert row["relative_error_pct"] is None and row["seconds"] is None
        if row["method"] == "mc" and fail:
            assert (
                row["absolute_error"] is None
                and row["solver_status"] == "solver_failed"
            )
            assert evidence["pilots"][0]["failures"]
        else:
            assert row["absolute_error"] == 0


def test_absolute_diagnostic_is_visible_in_legacy_ui():
    app = AppTest.from_string(
        "from pdelab.study_policy import render_accuracy_results\n"
        "from pdelab.heat_mc_study import heat_mc_study\n"
        "from pdelab.heat import make_heat_problem\n"
        "study = heat_mc_study(make_heat_problem('decay', 8), T=.02, "
        "point=(0, 4), repeats=2, max_walks=64, methods=('mc',))\n"
        "render_accuracy_results(study['rows'], key='zero')"
    ).run()
    assert not app.exception
    assert any("цели 1/5/10% неприменимы" in c.value for c in app.caption)
    assert app.dataframe[1].value["Абсолютная L2-ошибка"].iloc[0] == 0
