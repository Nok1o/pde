"""Poisson random walks on a square grid refined toward its boundary."""

from operator import index

import numpy as np
from numba import njit

from .operator import FrozenOperator


def graded_axis(n):
    """Return n+1 nodes with widths proportional to 1, 3, 5, …, 5, 3, 1."""
    n = index(n)
    if n < 4 or n % 2:
        raise ValueError("Для симметричной сетки требуется чётное n >= 4")
    left = 0.5 * np.linspace(0.0, 1.0, n // 2 + 1) ** 2
    return np.concatenate((left, 1.0 - left[-2::-1]))


def graded_operator(axis, source, boundary):
    """Assemble the volume-scaled -Δu = -f system from nodal arrays [y, x]."""
    axis = np.asarray(axis, dtype=float)
    if axis.ndim != 1 or len(axis) < 3 or axis[0] != 0 or axis[-1] != 1:
        raise ValueError("Ось должна идти от 0 до 1 и содержать внутренние узлы")
    widths = np.diff(axis)
    if not np.isfinite(widths).all() or np.any(widths <= 0):
        raise ValueError("Узлы оси должны строго возрастать")
    shape = (len(axis), len(axis))
    source = np.asarray(source, dtype=float)
    if source.shape != shape:
        raise ValueError("Источник должен быть задан в каждом узле сетки")

    # A shared face conductance is its length divided by the normal spacing.
    # Scaling the RHS by the dual cell area recovers the unequal-step stencil.
    dual = np.concatenate((widths[:1], widths[:-1] + widths[1:], widths[-1:])) / 2
    return FrozenOperator(
        wx=dual[:, None] / widths[None, :],
        wy=dual[None, :] / widths[:, None],
        sigma=np.zeros(shape),
        b=-source * dual[:, None] * dual[None, :],
        boundary=boundary,
    )


@njit(cache=True)
def _walk(probabilities, reward, boundary, y0, x0, walks, seed, max_steps):
    np.random.seed(seed)
    n = len(boundary) - 1
    mean = 0.0
    squared_deviations = 0.0
    steps = 0
    for sample in range(1, walks + 1):
        y, x = y0, x0
        value = 0.0
        path_steps = 0
        while 0 < y < n and 0 < x < n:
            if path_steps >= max_steps:
                raise RuntimeError(
                    "Траектория на неравномерной сетке исчерпала max_steps"
                )
            p = probabilities[y - 1, x - 1]
            value += reward[y - 1, x - 1]
            draw = np.random.random()
            if draw < p[0]:
                y += 1
            elif draw < p[0] + p[1]:
                y -= 1
            elif draw < p[0] + p[1] + p[2]:
                x += 1
            else:
                x -= 1
            path_steps += 1
        value += boundary[y, x]
        steps += path_steps
        delta = value - mean
        mean += delta / sample
        squared_deviations += delta * (value - mean)
    return mean, np.sqrt(squared_deviations / (walks * (walks - 1.0))), steps


def estimate_graded(operator, point, *, walks=2048, seed=42, max_steps=None):
    """Return (value, standard error, total transitions) at nodal indices (y, x)."""
    y, x = map(index, point)
    n = len(operator.boundary) - 1
    walks, seed = index(walks), index(seed)
    limit = 100 * n * n if max_steps is None else index(max_steps)
    if not (0 <= y <= n and 0 <= x <= n):
        raise ValueError("Индексы точки должны принадлежать сетке")
    if walks < 2 or not 0 <= seed < 2**32 or limit < 1:
        raise ValueError("Требуются walks >= 2, max_steps >= 1 и 0 <= seed < 2**32")
    if x in (0, n) or y in (0, n):
        return float(operator.boundary[y, x]), 0.0, 0
    probabilities, reward, killing = operator.mc_coefficients()
    if np.any(killing):
        raise ValueError("Прототип поддерживает оператор Пуассона с sigma=0")
    return _walk(probabilities, reward, operator.boundary, y, x, walks, seed, limit)
