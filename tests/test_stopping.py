import numpy as np
import pytest

from pdelab.experiment import Config, benchmark, run_experiment
from pdelab.problem import Problem, laplacian, make_problem, select_points
from pdelab.solvers import LOD_STOPPING_RULE, solve
from pdelab.views import residual_plot


@pytest.mark.parametrize("preset", ["hot_top", "sine"])
def test_relative_stopping_is_invariant_to_data_scale(preset):
    p = make_problem(preset, 12)
    q = select_points(12)
    base = solve(p, q, "lod_serial", tol=1e-7)
    assert base.iterations > 0
    for scale in (1e-9, 1e9, -3.0):
        scaled = Problem(p.name, p.n, p.f * scale, p.boundary * scale)
        result = solve(scaled, q, "lod_serial", tol=1e-7)
        assert result.converged and result.iterations == base.iterations
        np.testing.assert_allclose(
            result.values / scale, base.values, atol=1e-10, rtol=1e-12
        )
        np.testing.assert_allclose(
            result.extras["relative_residuals"],
            base.extras["relative_residuals"],
            atol=1e-13,
            rtol=1e-8,
        )


def test_zero_source_uses_boundary_rhs_and_stops_at_first_relative_threshold():
    p = make_problem(n=12)
    tol = 1e-5
    result = solve(p, select_points(12), "lod_serial", tol=tol)
    assert np.all(p.f == 0)
    assert result.extras["residual_scale"] == pytest.approx(100 / p.h**2)
    absolute = np.max(np.abs(laplacian(result.values, p.h) - p.f[1:-1, 1:-1]))
    assert result.relative_residual == pytest.approx(absolute / (100 / p.h**2))
    assert result.residual == pytest.approx(absolute)
    assert result.residual > tol  # This is no longer an absolute stopping rule.
    curve = result.extras["relative_residuals"]
    assert curve[0] == 1 and curve[-1] <= tol and curve[-2] > tol
    assert result.extras["stopping_rule"] == LOD_STOPPING_RULE


def test_zero_rhs_stops_immediately_without_artificial_denominator():
    p = make_problem(n=8, top=0.0)
    result = solve(p, select_points(8), "lod_serial")
    assert result.converged and result.iterations == 0
    assert (
        result.residual
        == result.relative_residual
        == result.extras["residual_scale"]
        == 0
    )
    np.testing.assert_array_equal(result.values, 0)
    np.testing.assert_array_equal(result.extras["relative_residuals"], [0.0])


@pytest.mark.parametrize("tol", [0.0, 1.0, 2.0, -1.0, float("nan"), float("inf")])
def test_invalid_relative_tolerance_is_rejected(tol):
    p = make_problem(n=8)
    with pytest.raises(ValueError, match="Относительный допуск"):
        solve(p, select_points(8), "lod_serial", tol=tol)


def test_exhausted_iterations_are_not_reported_as_convergence():
    p = make_problem(n=8)
    result = solve(p, select_points(8), "lod_serial", max_iter=1, tol=1e-8)
    assert not result.converged and result.relative_residual > 1e-8
    with pytest.raises(RuntimeError, match="относительная невязка"):
        run_experiment(Config(n=8, max_iter=1), methods=["lod_serial"])


def test_relative_history_and_threshold_are_used_in_both_plot_scales():
    exp = run_experiment(Config(n=8), methods=["lod_serial"])
    for scale in ("linear", "log"):
        fig = residual_plot(exp, scale)
        np.testing.assert_array_equal(
            fig.data[0].y, exp.results["lod_serial"].extras["relative_residuals"]
        )
        assert fig.layout.yaxis.type == scale
        assert fig.layout.yaxis.title.text == "Относительная невязка"
        assert fig.layout.shapes[0].y0 == exp.config.tol
    rows = benchmark(
        Config(n=8, batches=4), budgets=[8], repeats=2, tolerances=[1e-2, 1e-5]
    )
    lod = [row for row in rows if row["method"] == "lod_serial"]
    assert all(row["relative_residual"] <= row["budget"] for row in lod)
    assert all(row["budget_kind"] == "relative_residual_tolerance" for row in lod)
