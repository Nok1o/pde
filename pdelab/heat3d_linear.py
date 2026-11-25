"""XYZ line Gauss--Seidel for one immutable 3D heat system."""

import numpy as np
from numba import njit

from .heat3d_contract import LinearResult3D
from .heat3d_operator import HeatOperator3D
from .solvers import NUMBA_LOCK


@njit(cache=True, nogil=True)
def _xyz_cycle(values, diagonal, weights, rhs):
    """Solve complete lines in x/y/z order, with the full system diagonal."""
    size = values.shape[0]
    alpha = np.empty(size)
    beta = np.empty(size)
    for axis in (2, 1, 0):
        for first in range(size):
            for second in range(size):
                for i in range(size):
                    if axis == 2:
                        z, y, x = first, second, i
                    elif axis == 1:
                        z, y, x = first, i, second
                    else:
                        z, y, x = i, first, second
                    load = rhs[z, y, x]
                    if axis != 0:
                        if z + 1 < size:
                            load += weights[z, y, x, 0] * values[z + 1, y, x]
                        if z > 0:
                            load += weights[z, y, x, 1] * values[z - 1, y, x]
                    if axis != 1:
                        if y + 1 < size:
                            load += weights[z, y, x, 2] * values[z, y + 1, x]
                        if y > 0:
                            load += weights[z, y, x, 3] * values[z, y - 1, x]
                    if axis != 2:
                        if x + 1 < size:
                            load += weights[z, y, x, 4] * values[z, y, x + 1]
                        if x > 0:
                            load += weights[z, y, x, 5] * values[z, y, x - 1]
                    lower = weights[z, y, x, 2 * axis + 1]
                    upper = weights[z, y, x, 2 * axis]
                    pivot = diagonal[z, y, x]
                    if i:
                        pivot -= lower * alpha[i - 1]
                        load += lower * beta[i - 1]
                    if not np.isfinite(pivot) or pivot <= 0 or not np.isfinite(load):
                        return False
                    alpha[i] = upper / pivot
                    beta[i] = load / pivot
                    if not np.isfinite(alpha[i]) or not np.isfinite(beta[i]):
                        return False
                following = 0.0
                for i in range(size - 1, -1, -1):
                    value = beta[i] + alpha[i] * following
                    if not np.isfinite(value):
                        return False
                    if axis == 2:
                        values[first, second, i] = value
                    elif axis == 1:
                        values[first, i, second] = value
                    else:
                        values[i, first, second] = value
                    following = value
    return True


def _residual(operator, values, free, scale):
    with np.errstate(over="ignore", invalid="ignore"):
        residual = operator.apply(values) - operator.rhs
    if not np.isfinite(residual).all():
        return float("inf"), False
    norm = float(np.max(np.abs(residual[free]))) if np.any(free) else 0.0
    if scale == 0:
        return (0.0 if norm == 0 else float("inf")), True
    return norm / scale, True


def solve_lines3d(operator, initial, *, tolerance, max_iterations):
    """Solve a frozen system; count only completed XYZ cycles as iterations."""
    if not np.isfinite(tolerance) or not 0 < tolerance < 1:
        raise ValueError("Нужен относительный допуск между 0 и 1")
    if (
        isinstance(max_iterations, (bool, np.bool_))
        or not np.isscalar(max_iterations)
        or not np.isfinite(max_iterations)
        or int(max_iterations) != max_iterations
        or max_iterations < 1
    ):
        raise ValueError("max_iterations должно быть положительным целым")
    values = np.array(initial, dtype=float, order="C", copy=True)
    if values.shape != operator.rhs.shape:
        raise ValueError("Начальное приближение должно иметь форму оператора")
    finite_initial = np.isfinite(values).all()
    values[operator.fixed] = operator.fixed_values[operator.fixed]
    if not finite_initial:
        return LinearResult3D(values, "nonfinite", "nonfinite_initial", float("inf"))
    free = ~operator.fixed
    scale = float(np.max(np.abs(operator.rhs[free]))) if np.any(free) else 0.0
    with NUMBA_LOCK:
        for iteration in range(int(max_iterations) + 1):
            relative, finite = _residual(operator, values, free, scale)
            if not finite:
                return LinearResult3D(
                    values, "nonfinite", "nonfinite_residual", relative, iteration
                )
            if relative <= tolerance:
                return LinearResult3D(
                    values, "completed", "linear_residual", relative, iteration
                )
            if iteration == max_iterations:
                return LinearResult3D(
                    values, "linear_limit", "max_iterations", relative, iteration
                )
            completed = _xyz_cycle(
                values, operator.diagonal, operator.weights, operator.rhs
            )
            values[operator.fixed] = operator.fixed_values[operator.fixed]
            if not completed:
                return LinearResult3D(
                    values, "nonfinite", "nonfinite_line_solve", float("inf"), iteration
                )
    raise RuntimeError("Не завершён линейный 3D-расчёт")


_warmed = False


def warmup_lines3d():
    """Compile the production array signatures outside the caller's solve timer."""
    global _warmed
    with NUMBA_LOCK:
        if _warmed:
            return
        shape = (3, 3, 3)
        operator = HeatOperator3D(
            np.ones(shape),
            np.zeros((*shape, 6)),
            np.zeros(shape),
            np.ones(shape, dtype=bool),
            np.zeros(shape),
            np.ones(shape),
        )
        _xyz_cycle(np.zeros(shape), operator.diagonal, operator.weights, operator.rhs)
        _warmed = True
