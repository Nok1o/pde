"""Small-grid block Thomas elimination for the immutable 3D heat system.

Each full z-plane is one dense block, with x varying fastest. The two
off-plane blocks are diagonal, but neither they nor the in-plane blocks
are assumed symmetric. Boundary contributions are already in operator.rhs.
"""

import warnings
from dataclasses import dataclass
from time import perf_counter

import numpy as np
from scipy.linalg import LinAlgWarning, lu_factor, lu_solve

from .heat3d_contract import LinearResult3D
from .heat3d_operator import HeatOperator3D

BLOCK_MAX_N = 16
DEFAULT_BLOCK_WORKSPACE_BYTES = 256 * 1024**2


def _integer(value, name, minimum):
    if (
        isinstance(value, (bool, np.bool_))
        or not np.isscalar(value)
        or np.iscomplexobj(value)
    ):
        raise ValueError(f"{name}: требуется целое число >= {minimum}")
    try:
        number = int(value)
        valid = np.isfinite(value) and number == value and number >= minimum
    except (TypeError, ValueError, OverflowError):
        valid = False
    if not valid:
        raise ValueError(f"{name}: требуется целое число >= {minimum}")
    return number


def block_workspace_bytes(n):
    """Conservative numerical-storage estimate, not a measured peak RSS.

    For s=n+1 and m=s**2, retain s LU blocks and s-1 upper transfer
    blocks. Reserve three further dense blocks for assembly/LAPACK copies,
    int64-sized pivots, and twenty-four full fields for RHS, candidates,
    residuals, temporary arrays and the previous matrix's seven fields
    retained briefly during an equality check. The current immutable operator is
    excluded and must be included in the caller's total workspace estimate.
    This is a pure integer calculation, including for n > BLOCK_MAX_N.
    """
    n = _integer(n, "n", 2)
    size = n + 1
    plane = size**2
    return 8 * ((2 * size + 2) * plane**2 + 24 * size**3 + size * plane)


@dataclass
class BlockLinearResult3D(LinearResult3D):
    """Per-call counters; times are subsets of the enclosing solve timer.

    A factorization counts one completely prepared matrix, not individual
    planes. RHS substitutions and residual checks contribute to solve_seconds.
    workspace_bytes estimates this backend's storage, excluding the operator.
    """

    assembly_seconds: float = 0.0
    factorization_seconds: float = 0.0
    solve_seconds: float = 0.0
    factorizations: int = 0
    factor_reuses: int = 0
    workspace_bytes: int = 0


@dataclass
class _BlockFactors:
    diagonal: np.ndarray
    weights: np.ndarray
    lu: list
    upper: list


class _NonfiniteBlockError(ArithmeticError):
    pass


def _plane_block(operator, z, previous_upper):
    """B_z - L_z G_(z-1); L_z is minus the stored lower z weights."""
    size = operator.rhs.shape[0]
    plane = size**2
    block = np.empty((plane, plane), dtype=float, order="F")
    if previous_upper is None:
        block.fill(0)
    else:
        lower = operator.weights[z, :, :, 1].reshape(plane)
        np.multiply(lower[:, None], previous_upper, out=block)
    # Add the original in-plane stencil to the Schur correction. Full rows
    # preserve identity constraints without eliminating boundary loads twice.
    for y in range(size):
        for x in range(size):
            row = y * size + x
            block[row, row] += operator.diagonal[z, y, x]
            if y + 1 < size:
                block[row, row + size] -= operator.weights[z, y, x, 2]
            if y:
                block[row, row - size] -= operator.weights[z, y, x, 3]
            if x + 1 < size:
                block[row, row + 1] -= operator.weights[z, y, x, 4]
            if x:
                block[row, row - 1] -= operator.weights[z, y, x, 5]
    return block


def _factorize(operator, result):
    """Build Schur LU factors and G_z = solve(D_z, U_z), never an inverse."""
    size = operator.rhs.shape[0]
    plane = size**2
    factors, upper = [], []
    with warnings.catch_warnings(), np.errstate(over="ignore", invalid="ignore"):
        warnings.simplefilter("error", LinAlgWarning)
        for z in range(size):
            started = perf_counter()
            try:
                block = _plane_block(operator, z, upper[-1] if upper else None)
            finally:
                result.assembly_seconds += perf_counter() - started
            started = perf_counter()
            try:
                if not np.isfinite(block).all():
                    raise _NonfiniteBlockError
                factor = lu_factor(block, overwrite_a=True, check_finite=False)
                if not np.isfinite(factor[0]).all():
                    raise _NonfiniteBlockError
                factors.append(factor)
            finally:
                result.factorization_seconds += perf_counter() - started
            if z == size - 1:
                continue
            started = perf_counter()
            try:
                coupling = np.zeros((plane, plane), dtype=float, order="F")
                for row in range(plane):
                    y, x = divmod(row, size)
                    coupling[row, row] = -operator.weights[z, y, x, 0]
            finally:
                result.assembly_seconds += perf_counter() - started
            started = perf_counter()
            try:
                transfer = lu_solve(
                    factor, coupling, overwrite_b=True, check_finite=False
                )
                if not np.isfinite(transfer).all():
                    raise _NonfiniteBlockError
                upper.append(transfer)
            finally:
                result.factorization_seconds += perf_counter() - started
    return _BlockFactors(operator.diagonal, operator.weights, factors, upper)


def _substitute(operator, factors):
    """Forward RHS elimination and backward substitution using cached factors."""
    size = operator.rhs.shape[0]
    plane = size**2
    values = np.array(operator.rhs, dtype=float, order="C", copy=True)
    rows = values.reshape(size, plane)
    with np.errstate(over="ignore", invalid="ignore"):
        for z in range(size):
            if z:
                lower = operator.weights[z, :, :, 1].reshape(plane)
                rows[z] += lower * rows[z - 1]
            rows[z] = lu_solve(
                factors.lu[z], rows[z], overwrite_b=True, check_finite=False
            )
        for z in range(size - 2, -1, -1):
            rows[z] -= factors.upper[z] @ rows[z + 1]
    if not np.isfinite(values).all():
        raise _NonfiniteBlockError
    values[operator.fixed] = operator.fixed_values[operator.fixed]
    return values


def _relative_residual(operator, values):
    with np.errstate(over="ignore", invalid="ignore"):
        residual = operator.apply(values) - operator.rhs
    if not np.isfinite(residual).all():
        raise _NonfiniteBlockError
    free = ~operator.fixed
    if not np.any(free):
        return 0.0
    norm = float(np.max(np.abs(residual[free])))
    scale = float(np.max(np.abs(operator.rhs[free])))
    if scale == 0:
        return 0.0 if norm == 0 else float("inf")
    return norm / scale


class BlockThomasSolver3D:
    """A bounded, one-matrix cache for a true block tridiagonal direct solve.

    Only immutable HeatOperator3D snapshots are supported. Reuse requires
    identity of both matrix arrays, as provided by with_rhs, or exact array
    equality of both immutable arrays. The old cache is released
    before a different matrix is prepared. An instance is not thread-safe.

    Size and budget refusals happen before copying initial or building any
    block. Their diagnostic result refers to initial without mutating it.
    """

    def __init__(self, max_workspace_bytes=DEFAULT_BLOCK_WORKSPACE_BYTES):
        self.max_workspace_bytes = _integer(
            max_workspace_bytes, "max_workspace_bytes", 0
        )
        self._factors = None

    def __call__(self, operator, initial, *, tolerance, max_iterations):
        if (
            isinstance(tolerance, (bool, np.bool_))
            or not np.isscalar(tolerance)
            or np.iscomplexobj(tolerance)
        ):
            raise ValueError("Нужен относительный допуск между 0 и 1")
        try:
            valid_tolerance = np.isfinite(tolerance) and 0 < tolerance < 1
        except TypeError:
            valid_tolerance = False
        if not valid_tolerance:
            raise ValueError("Нужен относительный допуск между 0 и 1")
        _integer(max_iterations, "max_iterations", 1)
        result = BlockLinearResult3D(
            initial, "unsupported", "immutable_heat_operator_required", float("inf")
        )
        if not isinstance(operator, HeatOperator3D):
            self._factors = None
            return result
        n = operator.rhs.shape[0] - 1
        if (
            self._factors is not None
            and self._factors.diagonal.shape != operator.diagonal.shape
        ):
            # A previous larger grid must not survive into the smaller grid's
            # initial-field allocations or its smaller workspace estimate.
            self._factors = None
        result.workspace_bytes = block_workspace_bytes(n)
        if n > BLOCK_MAX_N:
            self._factors = None
            result.reason = "block_grid_limit"
            return result
        if result.workspace_bytes > self.max_workspace_bytes:
            self._factors = None
            result.status, result.reason = "memory_limit", "block_workspace_limit"
            return result
        if np.shape(initial) != operator.rhs.shape or np.iscomplexobj(initial):
            raise ValueError("Начальное приближение должно иметь форму оператора")
        try:
            values = np.array(initial, dtype=float, order="C", copy=True)
            finite_initial = np.isfinite(values).all()
            values[operator.fixed] = operator.fixed_values[operator.fixed]
            result.values = values
            if not finite_initial:
                self._factors = None
                result.status, result.reason = "nonfinite", "nonfinite_initial"
                return result
            started = perf_counter()
            try:
                result.relative_residual = _relative_residual(operator, values)
            finally:
                result.solve_seconds += perf_counter() - started
            if result.relative_residual <= tolerance:
                result.status, result.reason = "completed", "linear_residual"
                return result
            started = perf_counter()
            try:
                same_matrix = self._factors is not None and (
                    (
                        self._factors.diagonal is operator.diagonal
                        and self._factors.weights is operator.weights
                    )
                    or (
                        np.array_equal(self._factors.diagonal, operator.diagonal)
                        and np.array_equal(self._factors.weights, operator.weights)
                    )
                )
            finally:
                result.assembly_seconds += perf_counter() - started
            if same_matrix:
                # Keep references to the current immutable arrays, releasing
                # the old snapshot after an exact equality-based reuse.
                self._factors.diagonal = operator.diagonal
                self._factors.weights = operator.weights
                result.factor_reuses = 1
            else:
                # Drop the old matrix before any new dense allocation; never
                # publish a partial factorization into the persistent cache.
                self._factors = None
                self._factors = _factorize(operator, result)
                result.factorizations = 1
            started = perf_counter()
            try:
                result.values = _substitute(operator, self._factors)
                result.iterations = 1
                result.relative_residual = _relative_residual(operator, result.values)
            finally:
                result.solve_seconds += perf_counter() - started
        except MemoryError:
            self._factors = None
            result.status, result.reason = "memory_limit", "block_allocation_failed"
            return result
        except (LinAlgWarning, np.linalg.LinAlgError):
            self._factors = None
            result.status, result.reason = "linear_limit", "singular_block"
            return result
        except _NonfiniteBlockError:
            self._factors = None
            result.status, result.reason = "nonfinite", "nonfinite_block_solve"
            result.relative_residual = float("inf")
            return result
        result.status = (
            "completed" if result.relative_residual <= tolerance else "linear_limit"
        )
        result.reason = "linear_residual"
        return result
