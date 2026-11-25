"""Material families preserve positive frozen rows and the true nonlinear problem."""

from dataclasses import replace

import numpy as np
import pytest

from pdelab.boundary3d import FACES
from pdelab.heat3d_mc import frozen_mc_coefficients
from pdelab.heat3d_operator import freeze_heat3d
from pdelab.heat3d_problem import HeatFace3D, HeatProblem3D, make_heat_problem3d
from pdelab.heat3d_solver import solve_heat3d


@pytest.mark.parametrize("name", ["capacity_gamma", "reaction_sigma"])
@pytest.mark.parametrize("invalid", [-0.1, np.inf, np.nan, True, [0.5], 0.5 + 1j])
def test_material_parameters_reject_unsupported_values(name, invalid):
    with pytest.raises(ValueError, match=name):
        make_heat_problem3d(**{name: invalid})
    with pytest.raises(ValueError, match=name):
        replace(make_heat_problem3d(), **{name: invalid})


@pytest.mark.parametrize("kind3", [(), ("x0", "y0", "z0"), FACES])
def test_material_absorption_and_markov_identity(kind3):
    p = make_heat_problem3d(
        "polynomial",
        n=4,
        alpha=0.7,
        capacity_gamma=0.4,
        reaction_sigma=0.3,
        kinds={face: 3 if face in kind3 else 1 for face in FACES},
    )
    previous = p.initial_values()
    iterate = previous + np.indices(p.shape)[0] / 10
    op = freeze_heat3d(p, previous, iterate, 0.013, 0.013)
    base = freeze_heat3d(
        replace(p, capacity_gamma=0, reaction_sigma=0), previous, iterate, 0.013, 0.013
    )
    iterate[op.fixed] = op.fixed_values[op.fixed]
    free = ~op.fixed
    expected_extra = 0.4 * iterate**2 / 0.013 + 0.3 * iterate**2
    np.testing.assert_array_equal(op.weights, base.weights)
    np.testing.assert_allclose(
        (op.diagonal - base.diagonal)[free], expected_extra[free]
    )
    probabilities, reward, stopping = frozen_mc_coefficients(op)
    np.testing.assert_allclose(probabilities.sum(axis=-1) + stopping, 1, atol=1e-15)
    assert np.all(stopping[free] > 0) and np.all(probabilities >= 0)
    expected_absorption = (
        1 / 0.013
        + expected_extra
        + base.diagonal
        - 1 / 0.013
        - base.weights.sum(axis=-1)
    ) / op.diagonal
    np.testing.assert_allclose(stopping[free], expected_absorption[free], atol=1e-15)
    assert np.isfinite(reward).all()


def test_material_zero_limit_and_safe_extreme_values():
    p = make_heat_problem3d(alpha=0, capacity_gamma=0, reaction_sigma=0)
    np.testing.assert_array_equal(p.capacity([1e200, -1e200]), [1, 1])
    np.testing.assert_array_equal(p.reaction_coefficient([1e200, -1e200]), [0, 0])
    p = replace(p, capacity_gamma=1e-300, reaction_sigma=1e-300)
    np.testing.assert_allclose(p.capacity([1e200]), [1e100])
    np.testing.assert_allclose(p.reaction_coefficient([1e200]), [1e100])
    for function in (p.capacity, p.reaction_coefficient):
        with pytest.raises(ValueError):
            function([np.inf])
    p = replace(p, capacity_gamma=1, reaction_sigma=1)
    for function in (p.capacity, p.reaction_coefficient):
        with pytest.raises(ValueError):
            function([1e200])


def test_large_finite_capacity_parameter_keeps_zero_solution_finite():
    def zero(x, y, z, t):
        return 0.0

    problem = HeatProblem3D(
        name="zero",
        n=4,
        alpha=0,
        initial=zero,
        source=zero,
        faces={face: HeatFace3D(1, zero) for face in FACES},
        capacity_gamma=1e308,
    )
    for method in ("lod", "newton"):
        result = solve_heat3d(problem, method=method, dt=0.01, T=0.02)
        assert result.status == "completed"
        np.testing.assert_array_equal(result.values, 0.0)
