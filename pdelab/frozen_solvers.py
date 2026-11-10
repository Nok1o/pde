"""Directional line relaxation and conditional MC for a FrozenOperator.

Both methods use the same immutable face weights, reaction and source.
MC may use a supplied field as a control approximation, never as an oracle.
"""

from dataclasses import dataclass

import numpy as np
from numba import njit

from .operator import FrozenOperator
from .solvers import NUMBA_LOCK


def relative_norm(values, scale):
    norm = float(np.max(np.abs(values)))
    return norm / scale if scale else (0.0 if norm == 0 else float("inf"))


def with_boundary(values, boundary):
    result = np.array(values, dtype=float, copy=True)
    if result.shape != boundary.shape or not np.isfinite(result).all():
        raise ValueError("Ожидается конечное поле той же формы, что граница")
    result[[0, -1], :] = boundary[[0, -1], :]
    result[:, [0, -1]] = boundary[:, [0, -1]]
    return result


def adjusted_rhs(operator):
    rhs = operator.b[1:-1, 1:-1].copy()
    g = operator.boundary
    rhs[0] += operator.wy[0, 1:-1] * g[0, 1:-1]
    rhs[-1] += operator.wy[-1, 1:-1] * g[-1, 1:-1]
    rhs[:, 0] += operator.wx[1:-1, 0] * g[1:-1, 0]
    rhs[:, -1] += operator.wx[1:-1, -1] * g[1:-1, -1]
    return rhs


@dataclass
class FrozenResult:
    values: np.ndarray
    iterations: int = 0
    relative_residual: float = 0.0
    converged: bool = True
    walks: int = 0
    steps: int = 0
    stderr: np.ndarray | None = None


class WalkLimitError(RuntimeError):
    def __init__(self, walks, steps):
        super().__init__("Лимит шагов MC: усечённый ответ не возвращён")
        self.walks = walks
        self.steps = steps


@njit(cache=True, nogil=True)
def _directional_lines(u, wx, wy, diagonal, b):
    """Row block Gauss–Seidel, then column block Gauss–Seidel.

    The coefficient field stays fixed throughout both passes. Only the
    unknown solution changes. Each block is solved by Thomas elimination.
    """
    n = u.shape[0] - 1
    alpha = np.empty(n + 1)
    beta = np.empty(n + 1)
    for axis in range(2):
        for line in range(1, n):
            alpha[0] = 0.0
            beta[0] = u[line, 0] if axis == 0 else u[0, line]
            for i in range(1, n):
                y, x = (line, i) if axis == 0 else (i, line)
                if axis == 0:
                    lower, upper = wx[y, x - 1], wx[y, x]
                    rhs = b[y, x] + wy[y - 1, x] * u[y - 1, x] + wy[y, x] * u[y + 1, x]
                else:
                    lower, upper = wy[y - 1, x], wy[y, x]
                    rhs = b[y, x] + wx[y, x - 1] * u[y, x - 1] + wx[y, x] * u[y, x + 1]
                denominator = diagonal[y - 1, x - 1] - lower * alpha[i - 1]
                alpha[i] = upper / denominator
                beta[i] = (rhs + lower * beta[i - 1]) / denominator
            following = u[line, n] if axis == 0 else u[n, line]
            for i in range(n - 1, 0, -1):
                value = alpha[i] * following + beta[i]
                if axis == 0:
                    u[line, i] = value
                else:
                    u[i, line] = value
                following = value


def solve_frozen_lod(operator, initial=None, tol=1e-10, max_iter=20000):
    if not 0 < tol < 1 or not np.isfinite(tol):
        raise ValueError("Нужен относительный допуск между 0 и 1")
    if isinstance(max_iter, bool) or int(max_iter) != max_iter or max_iter < 1:
        raise ValueError("max_iter должно быть положительным целым")
    u = with_boundary(
        np.zeros_like(operator.b) if initial is None else initial, operator.boundary
    )
    scale = float(np.max(np.abs(adjusted_rhs(operator))))
    if scale == 0:
        u[1:-1, 1:-1] = 0
        return FrozenResult(u)
    diagonal = operator.diagonal
    with NUMBA_LOCK:
        for iteration in range(int(max_iter) + 1):
            residual = relative_norm(operator.apply(u) - operator.b[1:-1, 1:-1], scale)
            if residual <= tol or iteration == max_iter:
                return FrozenResult(u, iteration, residual, residual <= tol)
            _directional_lines(u, operator.wx, operator.wy, diagonal, operator.b)
    raise RuntimeError("Не завершён внутренний линейный расчёт")


@njit(cache=True, nogil=True)
def _walks(probabilities, reward, boundary, walks, seed, max_steps):
    np.random.seed(seed)
    n = boundary.shape[0] - 1
    values = boundary.copy()
    errors = np.zeros_like(boundary)
    steps = 0
    paths = 0
    for start_y in range(1, n):
        for start_x in range(1, n):
            mean, variance_sum = 0.0, 0.0
            for path in range(walks):
                y, x = start_y, start_x
                payoff = 0.0
                length = 0
                while 0 < x < n and 0 < y < n:
                    if length == max_steps:
                        return values, errors, steps + length, paths + 1, False
                    payoff += reward[y - 1, x - 1]
                    length += 1
                    draw = np.random.random()
                    cumulative = 0.0
                    direction = 4
                    for candidate in range(4):
                        cumulative += probabilities[y - 1, x - 1, candidate]
                        if draw < cumulative:
                            direction = candidate
                            break
                    if direction == 4:
                        break
                    if direction == 0:
                        y += 1
                    elif direction == 1:
                        y -= 1
                    elif direction == 2:
                        x += 1
                    else:
                        x -= 1
                else:
                    payoff += boundary[y, x]
                steps += length
                paths += 1
                difference = payoff - mean
                mean += difference / (path + 1)
                variance_sum += difference * (payoff - mean)
            values[start_y, start_x] = mean
            errors[start_y, start_x] = np.sqrt(
                max(0.0, variance_sum) / (walks * (walks - 1))
            )
    return values, errors, steps, paths, True


def solve_frozen_mc(operator, walks=1024, seed=42, max_steps=100000, control=None):
    for name, value, minimum, maximum in (
        ("walks", walks, 2, 2**31 - 1),
        ("seed", seed, 0, 2**32 - 1),
        ("max_steps", max_steps, 1, 2**31 - 1),
    ):
        if (
            isinstance(value, (bool, np.bool_))
            or not np.isfinite(value)
            or int(value) != value
            or not minimum <= value <= maximum
        ):
            raise ValueError(f"{name}: требуется целое в [{minimum}, {maximum}]")
    probabilities, reward, _ = operator.mc_coefficients()
    boundary = operator.boundary.copy()
    base = None
    if control is not None:
        base = with_boundary(control, boundary)
        reward = (operator.b[1:-1, 1:-1] - operator.apply(base)) / operator.diagonal
        boundary = np.zeros_like(boundary)
    with NUMBA_LOCK:
        values, errors, steps, paths, completed = _walks(
            probabilities, reward, boundary, int(walks), int(seed), int(max_steps)
        )
    if not completed:
        raise WalkLimitError(paths, steps)
    if base is not None:
        values += base
    scale = float(np.max(np.abs(adjusted_rhs(operator))))
    residual = relative_norm(operator.apply(values) - operator.b[1:-1, 1:-1], scale)
    return FrozenResult(
        values, relative_residual=residual, walks=paths, steps=steps, stderr=errors
    )


_warmed = False


def warmup_frozen():
    global _warmed
    with NUMBA_LOCK:
        if _warmed:
            return
        operator = FrozenOperator.from_nodes(np.ones((5, 5)), 1, 0, sigma=2)
        solve_frozen_lod(operator, max_iter=1)
        solve_frozen_mc(operator, walks=2)
        _warmed = True
