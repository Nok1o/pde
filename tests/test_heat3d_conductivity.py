"""Delayed k is a trial; every accepted layer still satisfies the true equation."""

from dataclasses import replace

import numpy as np
import pytest

from pdelab.boundary3d import FACES
from pdelab.heat3d_operator import freeze_heat3d
from pdelab.heat3d_problem import make_heat_problem3d
from pdelab.heat3d_solver import HeatSettings3D, solve_heat3d


@pytest.mark.parametrize(
    "method", ["lod", "block", "mc", "mc_grouped", "mc_control", "mc_control_grouped"]
)
def test_constant_k_reuse_preserves_whole_history_including_mc(method):
    problem = make_heat_problem3d(
        "polynomial", 6, 0, capacity_gamma=0.2, reaction_sigma=0.1
    )
    problem = replace(problem, exact=None)
    settings = HeatSettings3D(
        tolerance=1e-5,
        change_tolerance=1e-5,
        mc_tolerance=0.4,
        mc_change_tolerance=0.25,
        walks=512,
        max_outer=12,
        max_walks=20_000_000,
        batch_size=7,
    )
    baseline = solve_heat3d(
        problem, dt=0.002, T=0.005, method=method, settings=settings
    )
    delayed = solve_heat3d(
        problem,
        dt=0.002,
        T=0.005,
        method=method,
        settings=replace(settings, conductivity_refresh_layers=4),
    )
    assert baseline.status == delayed.status == "completed"
    np.testing.assert_array_equal(delayed.values, baseline.values)
    assert delayed.walks == baseline.walks
    assert delayed.transitions == baseline.transitions
    assert delayed.outer_iterations == baseline.outer_iterations
    assert delayed.validation_walks == baseline.validation_walks
    assert delayed.conductivity_reuses == 2
    for original, current in zip(baseline.history, delayed.history, strict=True):
        for key in (
            "nonlinear_relative_residual",
            "relative_change",
            "accepted",
            "update_seed",
            "validation_seed",
            "walks",
            "transitions",
        ):
            assert original.get(key) == current.get(key)


@pytest.mark.parametrize("faces", [(), FACES])
@pytest.mark.parametrize("method", ["lod", "block"])
def test_rejected_delayed_update_refreshes_and_matches_independent_newton(
    faces, method
):
    problem = make_heat_problem3d(
        "sine",
        4,
        2,
        kinds={face: 3 if face in faces else 1 for face in FACES},
        capacity_gamma=0.5,
        reaction_sigma=1,
    )
    settings = HeatSettings3D(
        conductivity_refresh_layers=4, tolerance=1e-8, change_tolerance=1e-8
    )
    result = solve_heat3d(problem, dt=0.01, T=0.025, settings=settings, method=method)
    reference = solve_heat3d(problem, dt=0.01, T=0.025, method="newton")
    assert result.status == reference.status == "completed"
    assert result.conductivity_reuses == 2
    assert result.conductivity_fallbacks > 0
    assert reference.conductivity_refresh_layers == 1
    np.testing.assert_allclose(result.values, reference.values, rtol=2e-7)
    for row in result.history:
        if row["accepted"]:
            assert row["nonlinear_relative_residual"] <= settings.tolerance
            assert row["relative_change"] <= settings.change_tolerance
    # An accurately solved stale linear system is not accepted on its own.
    rejected = [
        row
        for row in result.history
        if row["lagged_conductivity"] and not row["accepted"]
    ]
    assert rejected
    assert any(
        row["nonlinear_relative_residual"] > settings.tolerance for row in rejected
    )


def test_reuse_bound_and_identical_block_factors_on_linear_layers():
    problem = make_heat_problem3d("polynomial", 4, 0)
    # Accept an exactly solved linear layer in one update, so this isolates
    # periodic renewal rather than the conservative change-test fallback.
    settings = HeatSettings3D(conductivity_refresh_layers=2, change_tolerance=0.01)
    result = solve_heat3d(problem, dt=0.001, T=0.005, method="block", settings=settings)
    assert result.status == "completed"
    assert result.conductivity_reuses == 2  # layers 2 and 4; layers 3 and 5 renew k
    assert result.factorizations == 1 and result.factor_reuses > 0
    second = solve_heat3d(problem, dt=0.001, T=0.005, method="block", settings=settings)
    np.testing.assert_array_equal(result.values, second.values)
    assert second.factorizations == 1  # no matrix or coefficient cache across runs


@pytest.mark.parametrize("interval", [0, -1, 1.5, True, 10001])
def test_invalid_refresh_interval(interval):
    with pytest.raises(ValueError, match="conductivity_refresh_layers"):
        HeatSettings3D(conductivity_refresh_layers=interval).validate()


def test_supplied_trial_k_does_not_delay_capacity_or_reaction():
    problem = make_heat_problem3d(
        "polynomial", 4, 0.5, capacity_gamma=0.4, reaction_sigma=0.3
    )
    previous = problem.initial_values()
    current = previous + 0.2
    known_k = np.ones(problem.shape)
    trial = freeze_heat3d(
        problem, previous, current, 0.003, 0.001, conductivity=known_k
    )
    equivalent = freeze_heat3d(
        replace(problem, alpha=0), previous, current, 0.003, 0.001
    )
    # alpha also enters manufactured F/r, so replace keeps the same callables.
    np.testing.assert_array_equal(trial.diagonal, equivalent.diagonal)
    np.testing.assert_array_equal(trial.weights, equivalent.weights)
    np.testing.assert_array_equal(trial.rhs, equivalent.rhs)
    np.testing.assert_array_equal(known_k, 1)
    with pytest.raises(ValueError, match="проводимость"):
        freeze_heat3d(
            problem,
            previous,
            current,
            0.003,
            0.001,
            conductivity=np.zeros(problem.shape),
        )


def test_custom_operator_factory_does_not_silently_bypass_the_delay_contract():
    settings = HeatSettings3D(conductivity_refresh_layers=2)
    result = solve_heat3d(
        make_heat_problem3d(),
        settings=settings,
        operator_factory=lambda *args: freeze_heat3d(*args),
    )
    assert result.status == "unsupported" and result.values is None
    assert result.walks == result.physical_steps == 0
