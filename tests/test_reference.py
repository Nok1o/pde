import json
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pytest

from pdelab.experiment import Config, benchmark, refresh_comparison, run_experiment
from pdelab.reference import REFERENCE_POLICY
from pdelab.views import (
    benchmark_figure,
    convergence_plot,
    export,
    interactive_field,
    metrics_plot,
    surface_plot,
)


def test_sine_error_includes_spatial_bias_in_metrics_and_plots(tmp_path):
    exp = run_experiment(Config(problem="sine", n=8, walks=32, batches=4, tol=1e-11))
    row = exp.metrics[0]
    assert row["reference_kind"] == "analytic"
    assert row["reference_mean"] == pytest.approx(100.0)
    assert row["rmse"] > 1.0  # n=8 has visible spatial bias, despite tiny residual.
    assert row["rmse_grid"] < 1e-7
    assert row["rmse"] == row["rmse_exact"]
    np.testing.assert_array_equal(interactive_field(exp).data[0].z, exp.problem.exact)
    assert interactive_field(exp, "error").data[0].z[4, 4] == pytest.approx(row["rmse"])
    assert convergence_plot(exp, True).layout.shapes[0].y0 == pytest.approx(100.0)
    np.testing.assert_array_equal(surface_plot(exp).data[0].z, exp.problem.exact)
    fig = metrics_plot(exp)
    assert fig.axes[1].patches[0].get_height() == pytest.approx(row["rmse"])
    plt.close(fig)
    export(exp, tmp_path, with_gifs=False)
    stored = np.load(tmp_path / "fields.npz")
    np.testing.assert_array_equal(stored["reference"], exp.problem.exact)
    np.testing.assert_array_equal(stored["grid_reference"], exp.reference)
    data = json.loads((tmp_path / "experiment.json").read_text())
    assert data["metadata"]["reference_kind"] == "analytic"
    assert "Точное аналитическое решение" in (tmp_path / "interactive.html").read_text()


def test_exact_centre_is_not_presented_as_an_exact_full_field():
    exp = run_experiment(
        Config(n=8, top=80.0, bottom=20.0, left=10.0, right=30.0),
        methods=["lod_serial"],
    )
    assert exp.comparison.kind == "analytic_point"
    assert exp.comparison.values[4, 4] == 35.0
    assert np.isfinite(exp.comparison.values).sum() == 1
    np.testing.assert_array_equal(exp.comparison.field_values, exp.reference)
    assert exp.metrics[0]["reference_mean"] == 35.0


@pytest.mark.parametrize(
    "kwargs", [{"point": (1, 1)}, {"source": -1.0}, {"mode": "patch"}]
)
def test_grid_fallback_is_explicit_when_exact_query_values_are_unavailable(kwargs):
    exp = run_experiment(Config(n=8, **kwargs), methods=["lod_serial"])
    assert exp.comparison.kind == "grid"
    assert exp.metrics[0]["rmse"] == exp.metrics[0]["rmse_grid"]
    assert exp.metrics[0]["rmse_exact"] is None


def test_benchmark_uses_analytic_reference():
    rows = benchmark(
        Config(problem="sine", n=8, walks=16, batches=4),
        budgets=[16],
        repeats=2,
        tolerances=[1e-5],
    )
    assert all(row["reference_kind"] == "analytic" for row in rows)
    lod = [row for row in rows if row["method"].startswith("lod_")]
    assert all(row["rmse"] > 1 and row["relative_residual"] <= 1e-5 for row in lod)
    fig = benchmark_figure(rows)
    assert fig.data[0].y[0] > 1
    assert "Точное аналитическое решение" in fig.layout.title.text


def test_existing_results_upgrade_without_rerunning_solvers():
    exp = run_experiment(Config(problem="sine", n=8), methods=["lod_serial"])
    legacy = SimpleNamespace(**vars(exp))
    legacy.metadata = {}
    legacy.metrics = [{"rmse_grid": exp.metrics[0]["rmse_grid"]}]
    refreshed = refresh_comparison(legacy)
    assert refreshed.metadata["reference_policy"] == REFERENCE_POLICY
    assert refreshed.results["lod_serial"] is exp.results["lod_serial"]
    assert refreshed.metrics[0]["rmse"] > 1
