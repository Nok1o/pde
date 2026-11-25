"""Independent backward space-time paths for the backward Euler heat system.

Ordinary MC uses only the input data. Controlled MC adds a deterministic coarse
history and estimates its fine-grid residual with independent complete paths.
The source reward is paid on each visit; a temporal jump is paid only at t=0.
"""

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from numba import njit

from .control import _interpolate
from .heat import HeatProblem, _field, solve_heat, time_grid
from .operator import FrozenOperator
from .problem import laplacian, validate_points
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
    method: str = "mc"
    coarse_n: int | None = None
    control_reused: bool = False

    @property
    def walks(self):
        return len(self.points) * self.walks_per_point

    def rows(self):
        return [
            dict(
                method=self.method,
                coarse_n=self.coarse_n,
                control_reused=self.control_reused,
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
    problem,
    initial,
    times,
    steps,
    points,
    walks,
    seed,
    max_transitions,
    diagnostic,
    coarse_n=None,
    control=None,
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
    base = None
    control_reused = control is not None
    if coarse_n is not None:
        control = prepare_heat_control(problem, times, steps, coarse_n)
    if control is not None:
        base = np.asarray(control, dtype=float)
        if base.shape != rewards.shape or not np.isfinite(base).all():
            raise ValueError("Контрольное поле должно иметь форму (время, z, x)")
        # Exact residual of backward Euler, including the previous control layer.
        initial = initial - base[0]
        boundary = boundary - base
        for m, step in enumerate(steps, 1):
            residual = base[m - 1, 1:-1, 1:-1] - base[m, 1:-1, 1:-1]
            residual += step * problem.a2 * laplacian(base[m], 1 / problem.n)
            rewards[m, 1:-1, 1:-1] += temporal[kinds[m]] * residual
        if not all(np.isfinite(a).all() for a in (initial, boundary, rewards)):
            raise ValueError("Контрольная поправка должна быть конечной")
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
    if base is not None:
        values += base[-1][tuple(points.T)]
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
        memory + (base.nbytes if base is not None else 0),
        diagnostic,
        "mc_control" if base is not None else "mc",
        coarse_n,
        control_reused,
    )


def solve_heat_mc(
    problem,
    dt=0.01,
    T=0.2,
    points=None,
    walks=4096,
    seed=42,
    max_transitions=1_000_000,
    *,
    method="mc",
    coarse_n=None,
    control=None,
):
    """Standalone estimate at T, including preparation of all known F/g data.

    Paths terminate exactly on the initial field or Dirichlet data. Hitting
    max_transitions raises instead of returning a truncated, biased estimate.
    Each complete path is an independent observation including all its time jumps.
    Supplied control history preparation is outside the timer and marked as reused.
    """
    if method not in HEAT_MC_METHODS:
        raise ValueError("Неизвестный метод MC для теплопроводности")
    if method == "mc" and (coarse_n is not None or control is not None):
        raise ValueError("Контрольное поле требует method=mc_control")
    if method == "mc_control" and control is None:
        coarse_n = max(4, problem.n // 4) if coarse_n is None else coarse_n
        validate_coarse(problem.n, coarse_n)
    elif coarse_n is not None:
        raise ValueError("Выберите готовое поле или контрольную сетку")
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
        coarse_n=coarse_n,
        control=control,
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


HEAT_MC_METHODS = {"mc": "Обычный MC", "mc_control": "MC с контрольным приближением"}


def validate_coarse(n, coarse_n):
    if (
        isinstance(coarse_n, (bool, np.bool_))
        or not np.isfinite(coarse_n)
        or int(coarse_n) != coarse_n
        or not 4 <= coarse_n <= n
    ):
        raise ValueError("Контрольная сетка: целое от 4 до основной сетки")


def prepare_heat_control(problem, times, steps, coarse_n):
    """Coarse backward Euler from input data, never from problem.exact.

    Time layers are unchanged, including a short final step. Spatial transfer
    is bilinear. The correction uses the fine operator, so coarse error is not
    silently accepted as an answer. Called inside the measured preparation.
    """
    validate_coarse(problem.n, coarse_n)
    axis = np.linspace(0, 1, int(coarse_n) + 1)
    coarse = HeatProblem(
        "Контрольное поле",
        int(coarse_n),
        problem.a2,
        _interpolate(problem.initial, problem.axis, axis),
        lambda x, z, t: _interpolate(
            problem.evaluate(problem.source, t), problem.axis, axis
        ),
        lambda x, z, t: _interpolate(
            problem.evaluate(problem.boundary, t), problem.axis, axis
        ),
    )
    result = solve_heat(
        coarse, steps[0] if len(steps) else 1.0, times[-1], method="direct"
    )
    if not np.allclose(result.times, times, rtol=0, atol=1e-14):
        raise ValueError("Контрольное поле требует одинаковых временных слоёв")
    return np.array([_interpolate(u, axis, problem.axis) for u in result.fields])
