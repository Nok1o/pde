"""Independent matrix checks for full-plane block Thomas elimination."""

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

import pdelab.heat3d_block as block
from pdelab.heat3d_block import (
    BLOCK_MAX_N,
    BlockLinearResult3D,
    BlockThomasSolver3D,
    block_workspace_bytes,
)
from pdelab.heat3d_operator import HeatOperator3D, freeze_heat3d
from pdelab.heat3d_problem import make_heat_problem3d


def independent_csr(operator):
    """Scalar row assembly, without production assemble or neighbor helpers."""
    size = operator.rhs.shape[0]
    rows, columns, data = [], [], []
    for z, y, x in np.ndindex(operator.rhs.shape):
        row = (z * size + y) * size + x
        rows.append(row)
        columns.append(row)
        data.append(operator.diagonal[z, y, x])
        for direction, (zz, yy, xx) in enumerate(
            (
                (z + 1, y, x),
                (z - 1, y, x),
                (z, y + 1, x),
                (z, y - 1, x),
                (z, y, x + 1),
                (z, y, x - 1),
            )
        ):
            if min(zz, yy, xx) < 0 or max(zz, yy, xx) >= size:
                continue
            weight = operator.weights[z, y, x, direction]
            if weight:
                rows.append(row)
                columns.append((zz * size + yy) * size + xx)
                data.append(-weight)
    return coo_matrix((data, (rows, columns)), shape=(size**3, size**3)).tocsr()


def variable_operator(n=4, alpha=0.5, dt=0.04):
    problem = make_heat_problem3d("polynomial", n=n, alpha=alpha)
    previous = problem.initial_values()
    iterate = previous + np.random.default_rng(7).uniform(-0.1, 0.2, problem.shape)
    return freeze_heat3d(problem, previous, iterate, dt, dt)


def nonsymmetric_operator(n=4):
    """Directed, nonuniform weights with known nodes on and inside the cube."""
    shape = (n + 1,) * 3
    rng = np.random.default_rng(123)
    fixed = np.zeros(shape, dtype=bool)
    fixed[0, :, :] = True
    fixed[-1, 1, 2] = True
    fixed[2, 2, 2] = True
    boundary = np.zeros(shape)
    boundary[fixed] = rng.uniform(1, 5, fixed.sum())
    weights = np.zeros((*shape, 6))
    diagonal = rng.uniform(0.2, 2, shape)
    rhs = rng.uniform(-2, 3, shape)
    for z, y, x in np.ndindex(shape):
        node = (z, y, x)
        if fixed[node]:
            diagonal[node] = 1
            rhs[node] = boundary[node]
            continue
        for direction, neighbor in enumerate(
            (
                (z + 1, y, x),
                (z - 1, y, x),
                (z, y + 1, x),
                (z, y - 1, x),
                (z, y, x + 1),
                (z, y, x - 1),
            )
        ):
            if min(neighbor) < 0 or max(neighbor) > n:
                continue
            weight = rng.uniform(0.1, 7)
            diagonal[node] += weight
            if fixed[neighbor]:
                rhs[node] += weight * boundary[neighbor]
            else:
                weights[node + (direction,)] = weight
    return HeatOperator3D(
        diagonal, weights, rhs, fixed, boundary, rng.uniform(0.1, 1, shape)
    )


def run(solver, operator, *, tolerance=1e-12, initial=None):
    if initial is None:
        initial = np.zeros(operator.rhs.shape)
    return solver(operator, initial, tolerance=tolerance, max_iterations=1)


@pytest.mark.parametrize("n", [4, 8, BLOCK_MAX_N])
@pytest.mark.parametrize("alpha", [0.0, 0.5])
def test_frozen_heat_matches_independent_csr(n, alpha):
    operator = variable_operator(n, alpha)
    matrix = independent_csr(operator)
    initial = np.random.default_rng(4).normal(size=operator.rhs.shape)
    original = initial.copy()
    result = run(BlockThomasSolver3D(), operator, initial=initial)
    assert result.status == "completed"
    assert result.iterations == 1
    assert result.factorizations == 1
    assert result.factor_reuses == 0
    assert result.workspace_bytes == block_workspace_bytes(n)
    np.testing.assert_allclose(
        result.values.ravel(), spsolve(matrix, operator.rhs.ravel()), atol=2e-13
    )
    residual = (matrix @ result.values.ravel()).reshape(operator.rhs.shape)
    residual -= operator.rhs
    expected = np.max(np.abs(residual[~operator.fixed])) / np.max(
        np.abs(operator.rhs[~operator.fixed])
    )
    assert result.relative_residual == pytest.approx(expected, abs=2e-15)
    np.testing.assert_array_equal(initial, original)
    np.testing.assert_array_equal(
        result.values[operator.fixed], operator.fixed_values[operator.fixed]
    )


def test_nonuniform_nonsymmetric_matrix_and_eliminated_fixed_rows():
    operator = nonsymmetric_operator()
    matrix = independent_csr(operator)
    assert np.max(np.abs((matrix - matrix.T).data)) > 1
    fixed = np.flatnonzero(operator.fixed.ravel())
    for node in fixed:
        assert matrix[node].nnz == 1
        assert matrix[:, node].nnz == 1
        assert matrix[node, node] == 1
    result = run(BlockThomasSolver3D(), operator)
    assert result.status == "completed"
    np.testing.assert_allclose(
        result.values.ravel(), spsolve(matrix, operator.rhs.ravel()), atol=4e-14
    )
    np.testing.assert_array_equal(
        result.values[operator.fixed], operator.fixed_values[operator.fixed]
    )


def test_actual_plane_lu_and_multiple_rhs_solves_without_inverse_or_sparse_lu(
    monkeypatch,
):
    operator = nonsymmetric_operator()
    factor_shapes, solve_shapes = [], []
    original_factor, original_solve = block.lu_factor, block.lu_solve

    def factor(matrix, **kwargs):
        factor_shapes.append(matrix.shape)
        return original_factor(matrix, **kwargs)

    def solve(factors, rhs, **kwargs):
        solve_shapes.append(rhs.shape)
        return original_solve(factors, rhs, **kwargs)

    def forbidden(*args, **kwargs):
        raise AssertionError("Block Thomas must not assemble/invert the global matrix")

    monkeypatch.setattr(block, "lu_factor", factor)
    monkeypatch.setattr(block, "lu_solve", solve)
    monkeypatch.setattr(HeatOperator3D, "assemble", forbidden)
    monkeypatch.setattr(np.linalg, "inv", forbidden)
    monkeypatch.setattr("scipy.linalg.inv", forbidden)
    monkeypatch.setattr("scipy.sparse.linalg.spsolve", forbidden)
    monkeypatch.setattr("scipy.sparse.linalg.splu", forbidden)
    assert run(BlockThomasSolver3D(), operator).status == "completed"
    size = operator.rhs.shape[0]
    assert factor_shapes == [(size**2, size**2)] * size
    assert solve_shapes == [(size**2, size**2)] * (size - 1) + [(size**2,)] * size


def test_with_rhs_reuses_factors_and_does_not_repeat_boundary_loads():
    operator = nonsymmetric_operator()
    solver = BlockThomasSolver3D()
    first = run(solver, operator)
    control = np.random.default_rng(5).normal(size=operator.rhs.shape)
    corrected = operator.with_rhs(operator.rhs - operator.apply(control))
    assert corrected.diagonal is operator.diagonal
    assert corrected.weights is operator.weights
    second = run(solver, corrected)
    assert second.status == "completed"
    np.testing.assert_allclose(control + second.values, first.values, atol=2e-14)
    assert (first.factorizations, first.factor_reuses) == (1, 0)
    assert (second.factorizations, second.factor_reuses) == (0, 1)
    assert second.factorization_seconds == 0
    assert first.factorization_seconds > 0
    assert first.assembly_seconds > 0
    assert first.solve_seconds > 0
    assert second.solve_seconds > 0


def test_equal_distinct_frozen_matrix_reuses_but_changed_dt_replaces_cache():
    first_operator = variable_operator(alpha=0)
    same_operator = variable_operator(alpha=0)
    assert same_operator.diagonal is not first_operator.diagonal
    assert same_operator.weights is not first_operator.weights
    solver = BlockThomasSolver3D()
    first = run(solver, first_operator)
    second = run(solver, same_operator)
    assert first.factorizations == 1
    assert (second.factorizations, second.factor_reuses) == (0, 1)
    assert solver._factors.diagonal is same_operator.diagonal
    assert solver._factors.weights is same_operator.weights
    third = run(solver, variable_operator(alpha=0, dt=0.03))
    assert (third.factorizations, third.factor_reuses) == (1, 0)
    fourth = run(solver, first_operator)
    assert (fourth.factorizations, fourth.factor_reuses) == (1, 0)


def test_changed_weights_invalidate_cache_even_with_same_diagonal():
    operator = nonsymmetric_operator()
    changed_weights = np.array(operator.weights)
    changed_weights[1, 1, 1, 0] *= 0.5
    changed = HeatOperator3D(
        operator.diagonal,
        changed_weights,
        operator.rhs,
        operator.fixed,
        operator.fixed_values,
        operator.volumes,
    )
    solver = BlockThomasSolver3D()
    run(solver, operator)
    result = run(solver, changed)
    assert (result.factorizations, result.factor_reuses) == (1, 0)
    np.testing.assert_allclose(
        result.values.ravel(),
        spsolve(independent_csr(changed), changed.rhs.ravel()),
        atol=3e-14,
    )


def test_larger_cache_is_released_before_smaller_initial_field_allocation():
    solver = BlockThomasSolver3D()
    assert run(solver, variable_operator(n=8)).status == "completed"
    operator = variable_operator(n=4)

    class Initial:
        shape = operator.rhs.shape

        def __array__(self, dtype=None, copy=None):
            assert solver._factors is None
            return np.zeros(self.shape, dtype=dtype)

    result = run(solver, operator, initial=Initial())
    assert result.status == "completed"
    assert (result.factorizations, result.factor_reuses) == (1, 0)


class UncopyableInitial:
    def __init__(self, shape):
        self.shape = shape

    def __array__(self, *args, **kwargs):
        raise AssertionError("Preflight must run before converting initial")


@pytest.mark.parametrize("failure", ["size", "budget"])
def test_preflight_before_initial_copy_or_dense_allocation(monkeypatch, failure):
    n = BLOCK_MAX_N + 1 if failure == "size" else 4
    shape = (n + 1,) * 3
    operator = HeatOperator3D(
        np.ones(shape),
        np.zeros((*shape, 6)),
        np.ones(shape),
        np.zeros(shape, dtype=bool),
        np.zeros(shape),
        np.ones(shape),
    )
    initial = UncopyableInitial(shape)

    def forbidden(*args, **kwargs):
        raise AssertionError("Preflight must run before dense block allocation")

    monkeypatch.setattr(block, "_plane_block", forbidden)
    solver = BlockThomasSolver3D(max_workspace_bytes=block_workspace_bytes(n) - 1)
    result = run(solver, operator, initial=initial)
    assert result.status == ("unsupported" if failure == "size" else "memory_limit")
    assert result.values is initial
    assert result.iterations == result.factorizations == result.factor_reuses == 0
    assert result.assembly_seconds == result.factorization_seconds == 0


def test_exact_budget_is_allowed_and_estimate_covers_retained_factors():
    operator = nonsymmetric_operator()
    estimate = block_workspace_bytes(operator.n)
    solver = BlockThomasSolver3D(max_workspace_bytes=estimate)
    assert run(solver, operator).status == "completed"
    retained = sum(lu.nbytes + pivots.nbytes for lu, pivots in solver._factors.lu)
    retained += sum(upper.nbytes for upper in solver._factors.upper)
    assert estimate > retained + 3 * (operator.n + 1) ** 4 * 8
    assert block_workspace_bytes(BLOCK_MAX_N) < 32 * 1024**2


def test_memory_failure_discards_old_and_partial_cache(monkeypatch):
    solver = BlockThomasSolver3D()
    assert run(solver, variable_operator()).status == "completed"
    original = block._plane_block

    def fail(operator, z, previous):
        assert solver._factors is None
        if z == 1:
            raise MemoryError
        return original(operator, z, previous)

    monkeypatch.setattr(block, "_plane_block", fail)
    result = run(solver, variable_operator(dt=0.03))
    assert result.status == "memory_limit"
    assert result.reason == "block_allocation_failed"
    assert result.iterations == result.factorizations == 0
    assert result.factorization_seconds > 0
    assert solver._factors is None


def test_initial_solution_and_zero_rhs_need_no_direct_pass():
    operator = variable_operator().with_rhs(np.zeros((5,) * 3))
    result = run(BlockThomasSolver3D(), operator)
    assert result.status == "completed"
    assert result.iterations == result.factorizations == 0
    assert result.relative_residual == 0
    # Zero RHS scaling remains exact rather than being replaced by one.
    assert block._relative_residual(operator, np.ones(operator.rhs.shape)) == np.inf
    solved = run(BlockThomasSolver3D(), operator, initial=np.ones(operator.rhs.shape))
    assert solved.status == "completed"
    assert solved.iterations == 1
    np.testing.assert_array_equal(solved.values, 0)


def test_too_strict_tolerance_reports_actual_residual_and_complete_pass():
    operator = nonsymmetric_operator()
    result = run(BlockThomasSolver3D(), operator, tolerance=1e-30)
    assert result.status == "linear_limit"
    assert result.iterations == 1
    assert result.relative_residual > 1e-30
    assert result.relative_residual == block._relative_residual(operator, result.values)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_nonfinite_initial_is_explicit_and_input_is_unchanged(bad):
    operator = variable_operator()
    initial = np.zeros(operator.rhs.shape)
    initial[2, 2, 2] = bad
    result = run(BlockThomasSolver3D(), operator, initial=initial)
    assert result.status == "nonfinite"
    assert result.reason == "nonfinite_initial"
    assert result.iterations == result.factorizations == 0
    np.testing.assert_equal(initial[2, 2, 2], bad)


def test_nonfinite_direct_substitution_does_not_count_a_completed_pass():
    shape = (3,) * 3
    operator = HeatOperator3D(
        np.full(shape, 1e-308),
        np.zeros((*shape, 6)),
        np.full(shape, 1e308),
        np.zeros(shape, dtype=bool),
        np.zeros(shape),
        np.ones(shape),
    )
    solver = BlockThomasSolver3D()
    result = run(solver, operator)
    assert result.status == "nonfinite"
    assert result.iterations == 0
    assert result.factorizations == 1
    assert solver._factors is None


def test_singular_block_refuses_without_partial_cache_or_completed_pass():
    shape = (3,) * 3
    weights = np.zeros((*shape, 6))
    weights[1, 1, 0, 4] = weights[1, 1, 1, 5] = 1
    operator = HeatOperator3D(
        np.ones(shape),
        weights,
        np.ones(shape),
        np.zeros(shape, dtype=bool),
        np.zeros(shape),
        np.ones(shape),
    )
    solver = BlockThomasSolver3D()
    result = run(solver, operator)
    assert result.status == "linear_limit"
    assert result.reason == "singular_block"
    assert result.iterations == result.factorizations == 0
    assert solver._factors is None


def test_mutable_operator_protocol_is_explicitly_unsupported():
    operator = SimpleNamespace(rhs=np.zeros((5,) * 3))
    result = run(BlockThomasSolver3D(), operator)
    assert result.status == "unsupported"
    assert result.reason == "immutable_heat_operator_required"


@pytest.mark.parametrize(
    "tolerance,iterations",
    [(0, 1), (1, 1), (np.nan, 1), (True, 1), (1j, 1), (1e-6, 0), (1e-6, True)],
)
def test_invalid_solve_settings(tolerance, iterations):
    operator = variable_operator()
    with pytest.raises(ValueError):
        BlockThomasSolver3D()(
            operator,
            np.zeros(operator.rhs.shape),
            tolerance=tolerance,
            max_iterations=iterations,
        )


@pytest.mark.parametrize("budget", [-1, True, np.nan, np.inf, 1.5, "1"])
def test_invalid_budget(budget):
    with pytest.raises(ValueError):
        BlockThomasSolver3D(max_workspace_bytes=budget)


@pytest.mark.parametrize("n", [0, 1, True, np.nan, np.inf, 4.5, "4"])
def test_invalid_workspace_grid(n):
    with pytest.raises(ValueError):
        block_workspace_bytes(n)


def test_block_result_metrics_have_zero_defaults():
    result = BlockLinearResult3D(np.zeros((3,) * 3), "completed", "test", 0)
    assert result.assembly_seconds == result.factorization_seconds == 0
    assert result.solve_seconds == result.factorizations == result.factor_reuses == 0
    assert result.workspace_bytes == 0
