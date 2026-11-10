"""Independent backward space-time paths for the backward Euler heat system.

No estimated or deterministic previous solution fields are used by solve_heat_mc.
The source reward is paid on each visit; a temporal jump is paid only at t=0.
"""

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from numba import njit

from .heat import _field, time_grid
from .operator import FrozenOperator
from .problem import validate_points
from .solvers import NUMBA_LOCK


@njit(cache=True, nogil=True)
def _paths(
    initial,
    rewards,
    boundary,
    probabilities,
    temporal,
    kinds,
    points,
    walks,
    seed,
    max_transitions,
    start_index,
):
    np.random.seed(seed)
    n = initial.shape[0] - 1
    means = np.zeros(len(points))
    m2 = np.zeros(len(points))
    spatial_steps = 0
    temporal_steps = 0
    for point_index in range(len(points)):
        for walk in range(walks):
            y, x = points[point_index]
            m = start_index
            reward = 0.0
            transitions = 0
            while True:
                if m == 0:
                    reward += initial[y, x]
                    break
                if y == 0 or y == n or x == 0 or x == n:
                    reward += boundary[m, y, x]
                    break
                if transitions >= max_transitions:
                    raise RuntimeError(
                        "Предел пространственно-временных переходов; результат не возвращён"
                    )
                reward += rewards[m, y, x]
                kind = kinds[m]
                u = np.random.random()
                threshold = temporal[kind, y - 1, x - 1]
                if u < threshold:
                    m -= 1
                    temporal_steps += 1
                else:
                    weights = probabilities[kind, y - 1, x - 1]
                    threshold += weights[0]
                    if u < threshold:
                        y += 1
                    elif u < threshold + weights[1]:
                        y -= 1
                    elif u < threshold + weights[1] + weights[2]:
                        x += 1
                    else:
                        x -= 1
                    spatial_steps += 1
                transitions += 1
            delta = reward - means[point_index]
            means[point_index] += delta / (walk + 1)
            m2[point_index] += delta * (reward - means[point_index])
    stderr = np.sqrt(np.maximum(m2, 0) / (walks * (walks - 1)))
    return means, stderr, spatial_steps, temporal_steps


@dataclass
class HeatMCResult:
    values: np.ndarray
    stderr: np.ndarray
    points: np.ndarray
    times: np.ndarray
    walks_per_point: int
    seed: int
    max_transitions: int
    seconds: float
    preparation_seconds: float
    path_seconds: float
    spatial_transitions: int
    temporal_transitions: int
    memory_bytes: int
    diagnostic_step: bool

    @property
    def walks(self):
        return len(self.points) * self.walks_per_point

    def rows(self):
        return [
            dict(
                x_index=int(x),
                y_index=int(y),
                t=float(self.times[-1]),
                value=float(value),
                stderr=float(se),
                walks=self.walks_per_point,
            )
            for (y, x), value, se in zip(self.points, self.values, self.stderr)
        ]


def _settings(n, points, walks, seed, max_transitions):
    points = validate_points(n, [[n // 2, n // 2]] if points is None else points)
    for value, low, high, name in (
        (walks, 2, 2**31 - 1, "walks"),
        (seed, 0, 2**32 - 1, "seed"),
        (max_transitions, 1, 2**63 - 1, "max_transitions"),
    ):
        if (
            isinstance(value, bool)
            or not np.isfinite(value)
            or int(value) != value
            or not low <= value <= high
        ):
            raise ValueError(f"{name}: требуется целое от {low} до {high}")
    return points, int(walks), int(seed), int(max_transitions)


def _warmup():
    with NUMBA_LOCK:
        _paths(
            np.zeros((3, 3)),
            np.zeros((1, 3, 3)),
            np.zeros((1, 3, 3)),
            np.zeros((1, 1, 1, 4)),
            np.ones((1, 1, 1)),
            np.zeros(1, dtype=np.int64),
            np.array([[1, 1]], dtype=np.int64),
            2,
            0,
            1,
            0,
        )


def _estimate(
    problem, initial, times, steps, points, walks, seed, max_transitions, diagnostic
):
    points, walks, seed, max_transitions = _settings(
        problem.n, points, walks, seed, max_transitions
    )
    if len(times) * (problem.n + 1) ** 2 > 16_000_000:
        raise ValueError(
            "История коэффициентов ограничена 16 млн узлов; уменьшите сетку или число шагов"
        )
    _warmup()
    started = perf_counter()
    initial = _field(initial, problem.initial.shape)
    rewards = np.zeros((len(times), *initial.shape))
    boundary = np.zeros_like(rewards)
    boundary[0] = initial
    kinds = np.zeros(len(times), dtype=np.int64)
    probabilities, temporal = [], []
    cache = {}
    for m, step in enumerate(steps, 1):
        source = problem.evaluate(problem.source, times[m])
        boundary[m] = problem.evaluate(problem.boundary, times[m])
        if step not in cache:
            # sigma=1 is a jump to the previous time layer, not zero-payoff killing.
            op = FrozenOperator.from_nodes(
                np.full_like(initial, problem.a2 * step), 0, 0, sigma=1
            )
            p, _, q = op.mc_coefficients()
            if not np.all(q > 0):
                raise ValueError(
                    "Вероятность перехода к предыдущему слою должна быть положительной"
                )
            cache[step] = len(probabilities)
            probabilities.append(p)
            temporal.append(q)
        kind = cache[step]
        kinds[m] = kind
        rewards[m, 1:-1, 1:-1] = step * source[1:-1, 1:-1] * temporal[kind]
        if not np.isfinite(rewards[m]).all():
            raise ValueError("Вклад источника должен быть конечным")
    # The t=0 case still needs correctly shaped arrays for the compiled kernel.
    shape = (problem.n - 1, problem.n - 1)
    probabilities = (
        np.array(probabilities) if probabilities else np.zeros((1, *shape, 4))
    )
    temporal = np.array(temporal) if temporal else np.ones((1, *shape))
    prepared = perf_counter()
    values, stderr, spatial, time_steps = _paths(
        initial,
        rewards,
        boundary,
        probabilities,
        temporal,
        kinds,
        points,
        walks,
        seed,
        max_transitions,
        len(times) - 1,
    )
    finished = perf_counter()
    # Explicit resident numerical buffers, not process RSS or JIT/compiler memory.
    memory = sum(
        a.nbytes
        for a in (
            initial,
            rewards,
            boundary,
            probabilities,
            temporal,
            kinds,
            points,
            times,
            values,
            stderr,
        )
    )
    return HeatMCResult(
        values,
        stderr,
        points,
        times,
        walks,
        seed,
        max_transitions,
        finished - started,
        prepared - started,
        finished - prepared,
        spatial,
        time_steps,
        memory,
        diagnostic,
    )


def solve_heat_mc(
    problem, dt=0.01, T=0.2, points=None, walks=4096, seed=42, max_transitions=1_000_000
):
    """Standalone estimate at T, including preparation of all known F/g data.

    Paths terminate exactly on the initial field or Dirichlet data. Hitting
    max_transitions raises instead of returning a truncated, biased estimate.
    Each complete path is an independent observation including all its time jumps.
    """
    times, steps = time_grid(dt, T)
    return _estimate(
        problem,
        problem.initial,
        times,
        steps,
        points,
        walks,
        seed,
        max_transitions,
        False,
    )


def heat_mc_step(
    problem,
    previous,
    dt,
    t,
    points=None,
    walks=4096,
    seed=42,
    max_transitions=1_000_000,
):
    """Diagnostic only: previous is a supplied deterministic field at t-dt.

    Preparing previous is explicitly outside this diagnostic's measured time.
    This is not a standalone multi-layer MC calculation.
    """
    if not np.isfinite(t) or not np.isfinite(dt) or dt <= 0 or t < dt:
        raise ValueError("Для диагностики требуются конечные t≥dt>0")
    return _estimate(
        problem,
        previous,
        np.array([t - dt, t]),
        np.array([dt]),
        points,
        walks,
        seed,
        max_transitions,
        True,
    )
