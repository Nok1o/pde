"""Compare full XYZ line cycles against independent block and sparse solves."""

import numpy as np
import pytest
from scipy.sparse.linalg import spsolve

from pdelab.heat3d_linear import solve_lines3d, warmup_lines3d
from pdelab.heat3d_operator import freeze_heat3d
from pdelab.heat3d_problem import make_heat_problem3d


def variable_operator(n=4):
    problem = make_heat_problem3d("polynomial", n=n, alpha=0.5)
    previous = problem.initial_values()
    iterate = previous + np.random.default_rng(7).uniform(-0.1, 0.2, problem.shape)
    return freeze_heat3d(problem, previous, iterate, 0.04, 0.04)


def independent_xyz(operator, initial):
    """Dense block solves with current off-block values, no Thomas recurrence."""
    matrix, rhs = operator.assemble()
    matrix = matrix.toarray()
    indices = np.arange(initial.size).reshape(initial.shape)
    values = initial.copy()
    values[operator.fixed] = operator.fixed_values[operator.fixed]
    flat = values.ravel()
    size = initial.shape[0]
    for axis in (2, 1, 0):
        for first in range(size):
            for second in range(size):
                index = (
                    (first, second, slice(None))
                    if axis == 2
                    else (
                        (first, slice(None), second)
                        if axis == 1
                        else (slice(None), first, second)
                    )
                )
                nodes = indices[index]
                block = matrix[np.ix_(nodes, nodes)]
                load = rhs[nodes] - matrix[nodes] @ flat + block @ flat[nodes]
                flat[nodes] = np.linalg.solve(block, load)
    return values


def test_one_xyz_cycle_matches_independent_line_blocks_and_order():
    operator = variable_operator()
    initial = np.random.default_rng(2).normal(size=operator.rhs.shape)
    original = initial.copy()
    result = solve_lines3d(operator, initial, tolerance=1e-14, max_iterations=1)
    assert result.status == "linear_limit"
    assert result.iterations == 1
    np.testing.assert_allclose(
        result.values, independent_xyz(operator, initial), atol=3e-14
    )
    np.testing.assert_array_equal(initial, original)
    np.testing.assert_array_equal(
        result.values[operator.fixed], operator.fixed_values[operator.fixed]
    )


@pytest.mark.parametrize("n", [4, 8])
def test_variable_conductivity_lines_match_direct_solution(n):
    operator = variable_operator(n)
    matrix, rhs = operator.assemble()
    result = solve_lines3d(
        operator, np.zeros(operator.rhs.shape), tolerance=1e-12, max_iterations=2000
    )
    assert result.status == "completed"
    assert result.iterations > 0
    assert result.relative_residual <= 1e-12
    np.testing.assert_allclose(
        result.values.ravel(), spsolve(matrix, rhs), rtol=2e-11, atol=2e-11
    )
    free = ~operator.fixed
    expected = np.max(
        np.abs((matrix @ result.values.ravel() - rhs).reshape(operator.rhs.shape)[free])
    ) / np.max(np.abs(operator.rhs[free]))
    assert result.relative_residual == pytest.approx(expected, abs=2e-15)


def test_exact_initial_state_uses_zero_cycles_and_restores_fixed_values():
    operator = variable_operator()
    matrix, rhs = operator.assemble()
    initial = spsolve(matrix, rhs).reshape(operator.rhs.shape)
    initial[operator.fixed] = 999
    result = solve_lines3d(operator, initial, tolerance=1e-12, max_iterations=1)
    assert result.status == "completed"
    assert result.iterations == 0
    np.testing.assert_array_equal(
        result.values[operator.fixed], operator.fixed_values[operator.fixed]
    )


def test_zero_rhs_scale_is_not_replaced_by_one():
    operator = variable_operator().with_rhs(np.zeros((5, 5, 5)))
    exact = solve_lines3d(
        operator, np.zeros(operator.rhs.shape), tolerance=1e-8, max_iterations=1
    )
    assert exact.status == "completed"
    assert exact.relative_residual == 0
    assert exact.iterations == 0
    limited = solve_lines3d(
        operator, np.ones(operator.rhs.shape), tolerance=1e-8, max_iterations=1
    )
    assert limited.status == "linear_limit"
    assert limited.relative_residual == float("inf")
    assert limited.iterations == 1


def test_nonfinite_initial_and_overflow_are_explicit_failures():
    operator = variable_operator()
    initial = np.ones(operator.rhs.shape)
    initial[2, 2, 2] = np.nan
    result = solve_lines3d(operator, initial, tolerance=1e-8, max_iterations=2)
    assert result.status == "nonfinite"
    assert result.iterations == 0
    np.testing.assert_array_equal(
        result.values[operator.fixed], operator.fixed_values[operator.fixed]
    )
    initial.fill(1)
    initial[2, 2, 2] = np.finfo(float).max
    result = solve_lines3d(operator, initial, tolerance=1e-8, max_iterations=2)
    assert result.status == "nonfinite"
    assert result.reason == "nonfinite_residual"
    assert result.iterations == 0


@pytest.mark.parametrize(
    "tolerance,iterations",
    [(0, 1), (1, 1), (np.nan, 1), (1e-6, 0), (1e-6, True), (1e-6, 1.2)],
)
def test_invalid_linear_settings(tolerance, iterations):
    operator = variable_operator()
    with pytest.raises(ValueError):
        solve_lines3d(
            operator,
            np.zeros(operator.rhs.shape),
            tolerance=tolerance,
            max_iterations=iterations,
        )


def test_warmup_is_repeatable():
    warmup_lines3d()
    warmup_lines3d()
