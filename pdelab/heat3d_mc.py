"""Ordinary MC for one immutable A*w=b from the 3D heat contract (PDE-28).

This is a frozen-system estimator, not a nonlinear solve or a space-time walk.
Known-neighbor contributions already belong to rhs; do not pay g a second time.
"""

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from numba import njit

from .heat3d_operator import HeatOperator3D
from .problem3d import validate_points3d
from .solvers import NUMBA_LOCK


@dataclass
class FrozenMCResult3D:
    status: str
    reason: str
    points_zyx: np.ndarray
    values: np.ndarray | None
    stderr: np.ndarray | None
    diagnostic_values: np.ndarray | None
    completed_paths: np.ndarray | None
    walks: int
    transitions: int
    walks_per_estimate: int
    seed: int
    seconds: float
    preparation_seconds: float
    workspace_bytes: int
    peak_rss_bytes: int | None = None


def frozen_mc_coefficients(operator):
    """P=a/d, reward=b/d, killing=(d-sum(a))/d, including identity rows.

    The initial increment requires strictly positive killing on every free row.
    Reject an operator outside this sufficient termination condition explicitly.
    """
    if not isinstance(operator, HeatOperator3D):
        raise ValueError("Ожидается проверенный HeatOperator3D")
    probabilities = operator.weights / operator.diagonal[..., None]
    reward = operator.rhs / operator.diagonal
    killing = (operator.diagonal - operator.weights.sum(axis=-1)) / operator.diagonal
    if not np.isfinite(reward).all():
        raise ValueError("Неконечная выплата MC")
    if np.any(killing[~operator.fixed] <= 0):
        raise ValueError(
            "MC этого этапа требует положительное поглощение в каждой свободной строке"
        )
    return probabilities, reward, killing


@njit(cache=True, nogil=True)
def _frozen_paths(
    probabilities,
    reward,
    killing,
    fixed,
    points,
    walks,
    seed,
    max_steps,
    max_walks,
    max_transitions,
):
    np.random.seed(seed)
    means = np.zeros(len(points))
    m2 = np.zeros(len(points))
    counts = np.zeros(len(points), dtype=np.int64)
    started = 0
    transitions = 0
    for index in range(len(points)):
        sz, sy, sx = points[index]
        if fixed[sz, sy, sx]:
            means[index] = reward[sz, sy, sx]
            continue
        for _ in range(walks):
            if started >= max_walks:
                return means, m2, counts, started, transitions, 2
            started += 1
            z, y, x = sz, sy, sx
            payoff = 0.0
            steps = 0
            while True:
                payoff += reward[z, y, x]
                if not np.isfinite(payoff):
                    return means, m2, counts, started, transitions, 4
                choice = np.random.random()
                if choice < killing[z, y, x]:
                    break
                if steps >= max_steps:
                    return means, m2, counts, started, transitions, 1
                if transitions >= max_transitions:
                    return means, m2, counts, started, transitions, 3
                threshold = killing[z, y, x]
                direction = -1
                for axis_direction in range(6):
                    threshold += probabilities[z, y, x, axis_direction]
                    if choice < threshold:
                        direction = axis_direction
                        break
                # Only roundoff can leave a draw beyond the cumulative row sum.
                if direction == -1:
                    break
                if direction == 0:
                    z += 1
                elif direction == 1:
                    z -= 1
                elif direction == 2:
                    y += 1
                elif direction == 3:
                    y -= 1
                elif direction == 4:
                    x += 1
                else:
                    x -= 1
                steps += 1
                transitions += 1
            counts[index] += 1
            delta = payoff - means[index]
            means[index] += delta / counts[index]
            m2[index] += delta * (payoff - means[index])
            if not np.isfinite(means[index]) or not np.isfinite(m2[index]):
                return means, m2, counts, started, transitions, 4
    return means, m2, counts, started, transitions, 0


_warmed = False


def warmup_mc3d():
    """Compile production signatures outside the solve timer."""
    global _warmed
    with NUMBA_LOCK:
        if _warmed:
            return
        shape = (5, 5, 5)
        _frozen_paths(
            np.zeros((*shape, 6)),
            np.zeros(shape),
            np.ones(shape),
            np.frombuffer(
                np.zeros(shape, dtype=np.bool_).tobytes(), dtype=np.bool_
            ).reshape(shape),
            np.array([[2, 2, 2]], dtype=np.int64),
            2,
            0,
            1,
            2,
            1,
        )
        _warmed = True


def _integer(value, name, minimum, maximum):
    if (
        isinstance(value, (bool, np.bool_))
        or not np.isscalar(value)
        or not np.isfinite(value)
        or int(value) != value
        or not minimum <= value <= maximum
    ):
        raise ValueError(f"{name}: нужно целое в [{minimum}, {maximum}]")
    return int(value)


def solve_frozen_mc3d(
    operator,
    points=None,
    *,
    walks=2048,
    seed=42,
    max_steps=1000000,
    max_walks=1000000,
    max_transitions=10000000,
    max_workspace_bytes=256 * 1024**2,
):
    """Estimate requested nodes of a fixed system with independent complete paths.

    completed means a fixed sample budget finished; it is NOT residual convergence.
    On failure values/stderr are None; partial estimates are diagnostic only.
    All paths started (including an unfinished one) and moves executed are counted.
    Operator construction is caller-owned and excluded from seconds. Known queries
    cost zero paths. A full-field query must explicitly list all desired free nodes.
    """
    if not isinstance(operator, HeatOperator3D):
        raise ValueError("Ожидается HeatOperator3D")
    walks = _integer(walks, "walks", 2, 2**31 - 1)
    seed = _integer(seed, "seed", 0, 2**32 - 1)
    max_steps = _integer(max_steps, "max_steps", 1, 2**63 - 1)
    max_walks = _integer(max_walks, "max_walks", 1, 2**63 - 1)
    max_transitions = _integer(max_transitions, "max_transitions", 1, 2**63 - 1)
    max_workspace_bytes = _integer(
        max_workspace_bytes, "max_workspace_bytes", 1, 2**63 - 1
    )
    points = validate_points3d(
        operator.n, [[operator.n // 2] * 3] if points is None else points
    )
    workspace = (
        sum(
            getattr(operator, name).nbytes
            for name in (
                "weights",
                "rhs",
                "diagonal",
                "fixed",
                "fixed_values",
                "volumes",
            )
        )
        + 96 * operator.rhs.size
        + 128 * len(points)
    )
    started = perf_counter()

    def result(
        status, reason, means=None, counts=None, total_walks=0, steps=0, preparation=0.0
    ):
        completed = status == "completed"
        stderr = None
        if completed:
            stderr = np.zeros(len(points))
            free = ~operator.fixed[tuple(points.T)]
            stderr[free] = np.sqrt(
                np.maximum(m2[free], 0) / (counts[free] * (counts[free] - 1))
            )
        return FrozenMCResult3D(
            status,
            reason,
            points.copy(),
            means.copy() if completed else None,
            stderr,
            means,
            counts,
            total_walks,
            steps,
            walks,
            seed,
            perf_counter() - started,
            preparation,
            workspace,
        )

    if workspace > max_workspace_bytes:
        return result("memory_limit", "estimated_workspace_exceeds_limit")
    warmup_mc3d()
    started = perf_counter()
    probabilities, reward, killing = frozen_mc_coefficients(operator)
    prepared = perf_counter() - started
    with NUMBA_LOCK:
        means, m2, counts, total_walks, steps, code = _frozen_paths(
            probabilities,
            reward,
            killing,
            operator.fixed,
            points,
            walks,
            seed,
            max_steps,
            max_walks,
            max_transitions,
        )
    known = operator.fixed[tuple(points.T)]
    means[known] = reward[tuple(points.T)][known]
    # Rows not visited before a failure must not look like zero-valued estimates.
    means[(counts == 0) & ~known] = np.nan
    if code:
        status, reason = {
            1: ("path_limit", "max_steps"),
            2: ("walk_budget", "max_walks"),
            3: ("walk_budget", "max_transitions"),
            4: ("nonfinite", "payoff_or_statistics"),
        }[code]
        return result(status, reason, means, counts, total_walks, steps, prepared)
    return result(
        "completed",
        "fixed_sample_budget_completed",
        means,
        counts,
        total_walks,
        steps,
        prepared,
    )
