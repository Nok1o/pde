import numpy as np
import pytest

from pdelab.graded import estimate_graded, graded_axis, graded_operator


def test_graded_axis_has_symmetric_odd_widths():
    axis = graded_axis(8)
    np.testing.assert_array_equal(axis[[0, 4, 8]], [0.0, 0.5, 1.0])
    np.testing.assert_array_equal(np.diff(axis) * 32, [1, 3, 5, 7, 7, 5, 3, 1])
    with pytest.raises(ValueError, match="чётное"):
        graded_axis(7)


def test_uniform_axis_recovers_uniform_walk():
    axis = np.linspace(0, 1, 9)
    source = np.full((9, 9), 4.0)
    operator = graded_operator(axis, source, np.zeros_like(source))
    probabilities, reward, killing = operator.mc_coefficients()
    np.testing.assert_array_equal(probabilities, 0.25)
    np.testing.assert_array_equal(reward, -4.0 / (4 * 8**2))
    np.testing.assert_array_equal(killing, 0.0)


def test_quadratic_solution_and_weighted_walk():
    axis = graded_axis(8)
    x, y = np.meshgrid(axis, axis)
    exact = x**2 + 2 * y**2 + x - y
    operator = graded_operator(axis, np.full_like(exact, 6.0), exact)
    np.testing.assert_allclose(operator.solve(), exact, atol=1e-12)

    point = (3, 2)
    estimate = estimate_graded(operator, point, walks=8192, seed=2718)
    assert estimate == estimate_graded(operator, point, walks=8192, seed=2718)
    value, stderr, steps = estimate
    assert stderr > 0 and steps >= 8192
    assert value == pytest.approx(exact[point], abs=5 * stderr)
    assert estimate_graded(operator, (0, 2)) == (exact[0, 2], 0.0, 0)
    with pytest.raises(RuntimeError, match="max_steps"):
        estimate_graded(operator, (4, 4), walks=2, max_steps=1)
