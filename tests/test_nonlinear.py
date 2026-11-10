from dataclasses import replace

import numpy as np
import pytest

from pdelab.frozen_solvers import (
    FrozenResult,
    WalkLimitError,
    solve_frozen_lod,
    solve_frozen_mc,
    with_boundary,
)
from pdelab.heat import HeatProblem, solve_heat
from pdelab.nonlinear import (
    NonlinearSettings,
    freeze,
    make_nonlinear_problem,
    newton_system,
    nonlinear_residual,
    picard_layer,
    solve_nonlinear,
    solve_stationary_nonlinear,
)
from pdelab.nonlinear_study import (
    compare_nonlinear,
    nonlinear_accuracy_study,
    nonlinear_targets,
)


def layer_problem(alpha=0.5, n=6):
    problem = make_nonlinear_problem(n, alpha)
    dt = 0.02
    boundary = problem.evaluate(problem.boundary, dt)
    b = problem.initial / dt + problem.evaluate(problem.source, dt)
    return problem, freeze(problem, problem.initial, b, boundary, 1 / dt)


def test_manufactured_source_contains_conductivity_gradient():
    problem = make_nonlinear_problem(6, 2.0)
    x, y, t, h = 0.27, 0.38, 0.1, 1e-4
    u = problem.exact

    def flux(x, y, axis):
        derivative = (
            (u(x + h, y, t) - u(x - h, y, t))
            if axis == 0
            else (u(x, y + h, t) - u(x, y - h, t))
        ) / (2 * h)
        return (1 + problem.alpha * u(x, y, t) ** 2) * derivative

    divergence = (
        flux(x + h, y, 0) - flux(x - h, y, 0) + flux(x, y + h, 1) - flux(x, y - h, 1)
    ) / (2 * h)
    ut = (u(x, y, t + h) - u(x, y, t - h)) / (2 * h)
    assert problem.source(x, y, t) == pytest.approx(ut - divergence, rel=2e-6)


@pytest.mark.parametrize("alpha", [0.0, 0.5, 2.0])
def test_directional_linear_sweeps_match_direct_frozen_system(alpha):
    problem, operator = layer_problem(alpha)
    original = operator.wx.copy()
    result = solve_frozen_lod(operator, problem.initial, tol=1e-12)
    assert result.converged
    np.testing.assert_allclose(result.values, operator.solve(), atol=1e-10)
    np.testing.assert_array_equal(operator.wx, original)
    limited = solve_frozen_lod(operator, max_iter=1, tol=1e-14)
    assert not limited.converged


@pytest.mark.parametrize("control", [False, True])
def test_one_frozen_mc_layer_matches_direct_mean(control):
    problem, operator = layer_problem()
    points = (np.array([1, 3, 5]), np.array([2, 3, 1]))
    target = operator.solve()[points]
    samples = np.array(
        [
            solve_frozen_mc(
                operator,
                walks=2048,
                seed=seed,
                control=problem.initial if control else None,
            ).values[points]
            for seed in range(32)
        ]
    )
    uncertainty = samples.std(axis=0, ddof=1) / np.sqrt(len(samples))
    assert np.all(np.abs(samples.mean(axis=0) - target) < 5 * uncertainty)
    probabilities, reward, killing = operator.mc_coefficients()
    np.testing.assert_allclose(probabilities.sum(axis=-1) + killing, 1)
    assert np.ptp(probabilities) > 0.01
    np.testing.assert_allclose(reward, operator.b[1:-1, 1:-1] / operator.diagonal)


def test_newton_jacobian_is_independent_and_matches_finite_differences():
    problem, operator = layer_problem(n=4)
    u = problem.initial.copy()
    u[1:-1, 1:-1] += np.random.default_rng(2).normal(0, 0.1, (3, 3))
    residual, jacobian = newton_system(u, problem.alpha, operator.b, 50.0)
    np.testing.assert_allclose(
        residual,
        nonlinear_residual(problem, u, operator.b, operator.boundary, 50.0),
        atol=1e-12,
    )
    direction = np.random.default_rng(3).normal(size=(3, 3))
    high, low = u.copy(), u.copy()
    high[1:-1, 1:-1] += 1e-6 * direction
    low[1:-1, 1:-1] -= 1e-6 * direction
    difference = (
        newton_system(high, problem.alpha, operator.b, 50.0)[0]
        - newton_system(low, problem.alpha, operator.b, 50.0)[0]
    ) / (2e-6)
    np.testing.assert_allclose(
        jacobian @ direction.ravel(), difference.ravel(), rtol=1e-8, atol=1e-7
    )


@pytest.mark.parametrize("alpha", [0.0, 0.5, 2.0])
def test_stationary_then_full_history_picard_match_newton(alpha):
    settings = NonlinearSettings(tolerance=1e-9, change_tolerance=1e-9)
    stationary = solve_stationary_nonlinear(8, alpha, settings=settings)
    direct = solve_stationary_nonlinear(8, alpha, "newton", settings)
    assert stationary.converged and direct.converged
    np.testing.assert_allclose(stationary.values, direct.values, atol=2e-8)
    problem = make_nonlinear_problem(8, alpha)
    result = solve_nonlinear(problem, dt=0.013, T=0.04, settings=settings)
    reference = solve_nonlinear(
        problem, dt=0.013, T=0.04, method="newton", settings=settings
    )
    assert result.completed and reference.completed
    np.testing.assert_allclose(result.fields, reference.fields, atol=2e-8)
    assert result.times[-1] == 0.04 and result.times[-1] - result.times[
        -2
    ] == pytest.approx(0.001)
    assert all(
        layer.history[-1]["nonlinear_residual"] <= settings.tolerance
        for layer in result.layers
    )
    np.testing.assert_allclose(result.fields[:, 0, :], 1)


def test_alpha_zero_reproduces_issue10_heat_solver():
    problem = make_nonlinear_problem(8, 0.0)
    linear = HeatProblem(
        problem.name,
        problem.n,
        1.0,
        problem.initial,
        problem.source,
        problem.boundary,
        problem.exact,
    )
    expected = solve_heat(linear, dt=0.013, T=0.04, method="direct")
    result = solve_nonlinear(
        problem,
        dt=0.013,
        T=0.04,
        settings=NonlinearSettings(
            tolerance=1e-10, change_tolerance=1e-10, linear_tolerance=1e-12
        ),
    )
    np.testing.assert_allclose(result.fields, expected.fields, atol=1e-10)


def test_zero_data_and_initial_only_have_defined_relative_norms():
    problem = make_nonlinear_problem(4)
    zero = replace(
        problem,
        initial=np.zeros_like(problem.initial),
        boundary=lambda x, y, t: 0.0,
        source=lambda x, y, t: 0.0,
        exact=None,
    )
    for method in ("picard", "mc", "newton"):
        result = solve_nonlinear(zero, T=0.02, method=method)
        assert result.completed
        np.testing.assert_array_equal(result.fields, 0)
        assert result.walks == 0
        initial = solve_nonlinear(problem, T=0, method=method)
        assert initial.completed and len(initial.times) == 1


def test_small_update_alone_cannot_stop_picard(monkeypatch):
    problem, operator = layer_problem(n=4)
    monkeypatch.setattr(
        "pdelab.nonlinear.solve_frozen_lod",
        lambda operator, initial, *args: FrozenResult(initial.copy()),
    )
    result = picard_layer(
        problem,
        problem.initial,
        operator.b,
        operator.boundary,
        50.0,
        NonlinearSettings(max_outer=1),
    )
    assert not result.converged and result.reason == "max_outer"
    assert result.history[0]["relative_change"] == 0
    assert result.history[0]["nonlinear_residual"] > 1e-7


def test_mc_independent_validation_can_reject_candidate(monkeypatch):
    problem, operator = layer_problem(n=4)
    direct = solve_nonlinear(problem, T=0.02, dt=0.02, method="newton").fields[-1]
    calls = []

    def fake(operator, walks, seed, max_steps, control):
        calls.append(seed)
        values = (
            direct
            if len(calls) == 1
            else with_boundary(np.zeros_like(direct), operator.boundary)
        )
        return FrozenResult(values, walks=9 * walks)

    monkeypatch.setattr("pdelab.nonlinear.solve_frozen_mc", fake)
    result = picard_layer(
        problem,
        problem.initial,
        operator.b,
        operator.boundary,
        50.0,
        NonlinearSettings(tolerance=0.1, change_tolerance=0.1, max_outer=1),
        "mc",
    )
    assert len(calls) == 2 and calls[0] != calls[1]
    assert not result.converged
    assert result.history[0]["validation_residual"] > 0.1


def test_mc_full_chains_reproducible_without_analytic_or_lod_oracle(monkeypatch):
    problem = make_nonlinear_problem(6, 0.5)
    settings = NonlinearSettings(
        tolerance=0.001,
        change_tolerance=0.001,
        walks=64,
        max_walks=512,
        max_outer=12,
        relaxation=0.7,
    )
    first = solve_nonlinear(problem, T=0.04, method="mc", settings=settings)

    def forbidden(*args, **kwargs):
        raise AssertionError("MC cannot use a deterministic solve")

    monkeypatch.setattr("pdelab.nonlinear.solve_frozen_lod", forbidden)
    monkeypatch.setattr("pdelab.nonlinear.newton_layer", forbidden)
    second = solve_nonlinear(
        replace(problem, exact=None), T=0.04, method="mc", settings=settings
    )
    assert first.completed and second.completed
    np.testing.assert_array_equal(first.fields, second.fields)
    assert first.walks == second.walks > 0
    assert all(layer.reason == "independent_validation" for layer in first.layers)


def test_limits_are_failed_results_and_walk_truncation_has_no_payoff():
    problem = make_nonlinear_problem(8, 2.0)
    limited = solve_nonlinear(
        problem,
        T=0.1,
        settings=NonlinearSettings(
            max_outer=1, tolerance=1e-12, change_tolerance=1e-12
        ),
    )
    assert not limited.completed and len(limited.times) == 2
    assert "max_outer" in limited.reason
    no_budget = solve_nonlinear(
        problem, T=0.1, method="mc", settings=NonlinearSettings(max_total_walks=1)
    )
    assert not no_budget.completed and "walk_budget" in no_budget.reason
    _, operator = layer_problem(n=8)
    with pytest.raises(WalkLimitError) as caught:
        solve_frozen_mc(operator, walks=128, max_steps=1)
    assert caught.value.steps > 0 and caught.value.walks > 0
    interrupted = solve_nonlinear(
        problem, T=0.1, method="mc", settings=NonlinearSettings(max_steps=1)
    )
    assert not interrupted.completed and interrupted.walks > 0 and interrupted.steps > 0


def test_budget_rules_and_invalid_settings():
    for schedule, expected in [
        ("constant", [8, 8, 8, 8]),
        ("linear", [8, 16, 24, 32]),
        ("geometric", [8, 16, 32, 32]),
    ]:
        settings = NonlinearSettings(walks=8, max_walks=32, schedule=schedule)
        assert [settings.budget(i) for i in range(1, 5)] == expected
    for changes in (
        dict(relaxation=0),
        dict(walks=1),
        dict(max_walks=2),
        dict(seed=-1),
        dict(schedule="unknown"),
        dict(max_outer=0),
    ):
        with pytest.raises(ValueError):
            replace(NonlinearSettings(), **changes).validate()


def test_separated_error_orders_and_nonlinear_accuracy():
    rows = nonlinear_accuracy_study()
    for alpha in (0.0, 0.5, 2.0):
        space = [
            r for r in rows if r["alpha"] == alpha and r["study"] == "space_stationary"
        ]
        time = [r for r in rows if r["alpha"] == alpha and r["study"] == "time"]
        nonlinear = [
            r
            for r in rows
            if r["alpha"] == alpha and r["study"] == "nonlinear_iteration"
        ]
        assert all(1.9 < r["order"] < 2.1 for r in space[1:])
        assert all(0.8 < r["order"] < 1.3 for r in time[1:])
        assert nonlinear[-1]["error"] < nonlinear[0]["error"] * 1e-4


def test_full_chain_statistics_targets_and_disjoint_validation_seeds():
    problem = make_nonlinear_problem(4, 0.5)
    comparison = compare_nonlinear(problem, T=0.02, repeats=4)
    row = comparison.metrics[1]
    samples = np.array([r.fields[-1, 2, 2] for r in comparison.stochastic])
    assert row["chain_std"] == pytest.approx(samples.std(ddof=1))
    assert row["conductivity_plugin_bias"] == pytest.approx(0.5 * samples.var(ddof=1))
    rows, evidence = nonlinear_targets(problem, T=0.02, repeats=2, budgets=(8, 32))
    assert len(rows) == 6
    pilot_seeds = {s for item in evidence["pilots"] for s in item["seeds"]}
    control_seeds = {s for item in evidence["validations"] for s in item["seeds"]}
    assert control_seeds and pilot_seeds.isdisjoint(control_seeds)
    failed = compare_nonlinear(
        problem, T=0.02, repeats=2, stochastic=NonlinearSettings(max_total_walks=1)
    )
    assert failed.metrics[1]["rmse"] is None and not failed.metrics[1]["completed"]
