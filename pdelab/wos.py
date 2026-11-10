"""Pointwise walk-on-spheres for Δu = f on the unit square."""

from dataclasses import dataclass
from math import cos, isfinite, log, sin, sqrt, tau
from operator import index

import numpy as np
from numba import njit
from numba.core.registry import CPUDispatcher


@dataclass(frozen=True)
class WosEstimate:
    value: float
    stderr: float
    walks: int
    steps: int


class WosWalkLimitError(RuntimeError):
    """A trajectory did not reach the boundary tolerance within its step limit."""


@njit
def _estimate(x0, y0, source, boundary, walks, seed, epsilon, max_steps):
    np.random.seed(seed)
    mean = 0.0
    squared_deviations = 0.0
    steps = 0
    for sample in range(1, walks + 1):
        x, y = x0, y0
        value = 0.0
        for step in range(max_steps + 1):
            radius = min(x, 1.0 - x, y, 1.0 - y)
            if radius <= epsilon:
                if radius == x:
                    x = 0.0
                elif radius == 1.0 - x:
                    x = 1.0
                elif radius == y:
                    y = 0.0
                else:
                    y = 1.0
                value += boundary(x, y)
                break
            if step == max_steps:
                raise WosWalkLimitError("Траектория WoS исчерпала max_steps")

            # U is in (0, 1]. A uniform disk sample has radius R*sqrt(U).
            # The disk area times its positive Green function is -R² log(U)/4.
            u = 1.0 - np.random.random()
            r = radius * sqrt(u)
            angle = tau * np.random.random()
            value += (
                0.25
                * radius**2
                * log(u)
                * source(x + r * cos(angle), y + r * sin(angle))
            )

            angle = tau * np.random.random()
            x += radius * cos(angle)
            y += radius * sin(angle)
            steps += 1

        delta = value - mean
        mean += delta / sample
        squared_deviations += delta * (value - mean)

    stderr = sqrt(squared_deviations / (walks * (walks - 1.0)))
    return mean, stderr, steps


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name}: ожидается целое число")
    try:
        return index(value)
    except TypeError as exc:
        raise ValueError(f"{name}: ожидается целое число") from exc


def estimate_wos(
    point, source, boundary, *, walks=2048, seed=42, epsilon=1e-4, max_steps=1024
):
    """Estimate u at (x, y) with Numba-compatible scalar functions f(x, y), g(x, y).

    Pass pre-jitted functions to reuse compilation across queries. The standard
    error describes Monte Carlo sampling and excludes the boundary stopping bias.
    A point already on the boundary needs zero trajectories.
    """
    x, y = map(float, point)
    if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
        raise ValueError("Точка WoS должна принадлежать единичному квадрату")
    walks = _integer(walks, "walks")
    seed = _integer(seed, "seed")
    max_steps = _integer(max_steps, "max_steps")
    epsilon = float(epsilon)
    if walks < 2 or max_steps < 1 or not 0 <= seed < 2**32:
        raise ValueError("Требуются walks >= 2, max_steps >= 1 и 0 <= seed < 2**32")
    if not 0.0 < epsilon < 0.5:
        raise ValueError("epsilon должен лежать между 0 и 0.5")

    if min(x, 1.0 - x, y, 1.0 - y) == 0.0:
        value, stderr, steps, walks = float(boundary(x, y)), 0.0, 0, 0
    else:
        source = source if isinstance(source, CPUDispatcher) else njit(source)
        boundary = boundary if isinstance(boundary, CPUDispatcher) else njit(boundary)
        value, stderr, steps = _estimate(
            x, y, source, boundary, walks, seed, epsilon, max_steps
        )
    if not (isfinite(value) and isfinite(stderr)):
        raise ValueError("WoS получил неконечную оценку")
    return WosEstimate(value, stderr, walks, steps)
