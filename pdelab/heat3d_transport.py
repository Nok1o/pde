"""Transport integer particle counts or real mass through a frozen heat operator.

Counts preserve the law of independent walks without retaining particle labels.
Mass computes their expectation, hence is a deterministic method, not Monte Carlo.
Known-neighbor payments have already been eliminated into the operator's rhs.
"""

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from numba import njit

from .heat3d_contract import LinearResult3D
from .heat3d_mc import _integer, frozen_mc_coefficients
from .heat3d_operator import HeatOperator3D
from .problem3d import validate_points3d
from .solvers import NUMBA_LOCK


@dataclass
class FrozenTransportResult3D:
    status: str
    reason: str
    points_zyx: np.ndarray
    values: np.ndarray | None
    diagnostic_values: np.ndarray | None
    completed_paths: np.ndarray | None
    walks: int
    transitions: int
    walks_per_estimate: int | None
    seed: int | None
    seconds: float
    preparation_seconds: float
    workspace_bytes: int
    method: str
    node_updates: int = 0
    transport_sweeps: int = 0
    edge_transfers: int = 0
    tail_error_bound: np.ndarray | None = None
    stderr: None = None
    peak_rss_bytes: None = None
    batch_size: None = None


def transport_workspace_bytes(operator, point_count):
    """Conservative numerical-buffer allowance, including coefficient temporaries."""
    inputs = sum(
        getattr(operator, name).nbytes
        for name in ("diagonal", "weights", "rhs", "fixed", "fixed_values", "volumes")
    )
    return inputs + 160 * operator.rhs.size + 160 * point_count


def _points_and_budget(operator, points, max_workspace_bytes):
    if not isinstance(operator, HeatOperator3D):
        raise ValueError("Ожидается HeatOperator3D")
    points = validate_points3d(
        operator.n, [[operator.n // 2] * 3] if points is None else points
    )
    limit = _integer(max_workspace_bytes, "max_workspace_bytes", 1, 2**63 - 1)
    return points, transport_workspace_bytes(operator, len(points)), limit


def _relative_tolerance(tolerance):
    if (
        isinstance(tolerance, (bool, np.bool_))
        or not np.isscalar(tolerance)
        or not np.isfinite(tolerance)
        or not 0 < tolerance < 1
    ):
        raise ValueError("tolerance: нужен относительный допуск между 0 и 1")
    return float(tolerance)


@njit(cache=True, nogil=True)
def _counts_kernel(
    probabilities,
    reward,
    killing,
    fixed,
    points,
    side,
    walks,
    seed,
    max_steps,
    max_walks,
    max_transitions,
):
    np.random.seed(seed)
    size = len(reward)
    current = np.zeros(size, dtype=np.int64)
    following = np.zeros(size, dtype=np.int64)
    active = np.empty(size, dtype=np.int64)
    next_active = np.empty(size, dtype=np.int64)
    offsets = np.array([side * side, -side * side, side, -side, 1, -1])
    means = np.full(len(points), np.nan)
    completed = np.zeros(len(points), dtype=np.int64)
    started = 0
    transitions = 0
    updates = 0
    sweeps = 0
    edge_transfers = 0
    outgoing = np.empty(6, dtype=np.int64)

    for point_index in range(len(points)):
        z, y, x = points[point_index]
        source = (z * side + y) * side + x
        if fixed[source]:
            means[point_index] = reward[source]
            continue
        launched = min(walks, max_walks - started)
        if launched == 0:
            return (
                means,
                completed,
                started,
                transitions,
                updates,
                sweeps,
                edge_transfers,
                2,
            )
        started += launched
        current[source] = launched
        active[0] = source
        active_size = 1
        total = 0.0
        depth = 0

        while active_size:
            sweeps += 1
            next_size = 0
            for index in range(active_size):
                node = active[index]
                count = current[node]
                current[node] = 0
                # Normalize before multiplying: N*r may overflow even when
                # the requested mean and every individual reward are finite.
                total += (count / launched) * reward[node]
                updates += 1
                means[point_index] = total
                if not np.isfinite(total):
                    return (
                        means,
                        completed,
                        started,
                        transitions,
                        updates,
                        sweeps,
                        edge_transfers,
                        4,
                    )

                # Conditional binomials give one multinomial allocation, not
                # independent binomials that could create or lose particles.
                killed = np.random.binomial(count, killing[node])
                remaining = count - killed
                probability_left = 1.0 - killing[node]
                for direction in range(6):
                    probability = probabilities[node, direction]
                    sent = 0
                    if remaining and probability > 0 and probability_left > 0:
                        conditional = min(1.0, probability / probability_left)
                        sent = np.random.binomial(remaining, conditional)
                    outgoing[direction] = sent
                    remaining -= sent
                    probability_left = max(0.0, probability_left - probability)
                # The scalar kernel also absorbs a draw beyond the cumulative
                # floating-point row sum. No residual count is moved outside.
                killed += remaining
                completed[point_index] += killed
                moves = count - killed
                if moves and depth >= max_steps:
                    return (
                        means,
                        completed,
                        started,
                        transitions,
                        updates,
                        sweeps,
                        edge_transfers,
                        1,
                    )
                if moves > max_transitions - transitions:
                    return (
                        means,
                        completed,
                        started,
                        transitions,
                        updates,
                        sweeps,
                        edge_transfers,
                        3,
                    )
                for direction in range(6):
                    sent = outgoing[direction]
                    if sent:
                        neighbor = node + offsets[direction]
                        if following[neighbor] == 0:
                            next_active[next_size] = neighbor
                            next_size += 1
                        following[neighbor] += sent
                        edge_transfers += 1
                transitions += moves
            current, following = following, current
            active, next_active = next_active, active
            active_size = next_size
            depth += 1
        if launched != walks:
            return (
                means,
                completed,
                started,
                transitions,
                updates,
                sweeps,
                edge_transfers,
                2,
            )
    return means, completed, started, transitions, updates, sweeps, edge_transfers, 0


@njit(cache=True, nogil=True)
def _mass_kernel(
    probabilities, reward, fixed, points, side, bound, tolerance, max_iterations
):
    smallest_normal = np.finfo(np.float64).tiny
    size = len(reward)
    current = np.zeros(size)
    following = np.zeros(size)
    active = np.empty(size, dtype=np.int64)
    next_active = np.empty(size, dtype=np.int64)
    offsets = np.array([side * side, -side * side, side, -side, 1, -1])
    values = np.full(len(points), np.nan)
    tails = np.full(len(points), np.inf)
    updates = 0
    sweeps = 0
    edge_transfers = 0
    for point_index in range(len(points)):
        z, y, x = points[point_index]
        source = (z * side + y) * side + x
        if fixed[source]:
            values[point_index] = reward[source]
            tails[point_index] = 0.0
            continue
        if bound == 0:
            values[point_index] = 0.0
            tails[point_index] = 0.0
            continue
        current[source] = 1.0
        active[0] = source
        active_size = 1
        total = 0.0
        completed = False
        for _ in range(max_iterations):
            sweeps += 1
            next_size = 0
            for index in range(active_size):
                node = active[index]
                mass = current[node]
                current[node] = 0.0
                contribution = mass * reward[node]
                if reward[node] != 0 and abs(contribution) < smallest_normal:
                    values[point_index] = total
                    return values, tails, updates, sweeps, edge_transfers, 6
                total += contribution
                updates += 1
                for direction in range(6):
                    probability = probabilities[node, direction]
                    sent = mass * probability
                    # A tiny mass can still carry a large future reward. Dropping
                    # underflowed/subnormal flow would invalidate the tail bound.
                    if probability > 0 and sent < smallest_normal:
                        values[point_index] = total
                        return values, tails, updates, sweeps, edge_transfers, 6
                    if sent > 0:
                        neighbor = node + offsets[direction]
                        if following[neighbor] == 0:
                            next_active[next_size] = neighbor
                            next_size += 1
                        following[neighbor] += sent
                        edge_transfers += 1
            remaining = 0.0
            for index in range(next_size):
                remaining += following[next_active[index]]
            tail = remaining * bound
            values[point_index], tails[point_index] = total, tail
            if remaining > 0 and tail < smallest_normal:
                tails[point_index] = np.inf
                return values, tails, updates, sweeps, edge_transfers, 6
            if not np.isfinite(total) or not np.isfinite(tail):
                return values, tails, updates, sweeps, edge_transfers, 4
            current, following = following, current
            active, next_active = next_active, active
            active_size = next_size
            # |u-s| <= b and b <= eps*|s|/(1+eps) imply
            # |u-s| <= eps*|u|, also for signed rewards. No absolute floor.
            if tail == 0 or tail <= tolerance / (1.0 + tolerance) * abs(total):
                completed = True
                break
        # Clear any surviving frontier before another independent source.
        for index in range(active_size):
            current[active[index]] = 0.0
        if not completed:
            return values, tails, updates, sweeps, edge_transfers, 5
    return values, tails, updates, sweeps, edge_transfers, 0


def _tail_parameters(probabilities, reward, fixed):
    free = ~fixed
    if not np.any(free):
        return 0.0, 0.0
    rho = float(probabilities.sum(axis=-1)[free].max())
    if not 0 <= rho < 1:
        raise ValueError("Перенос массы требует строго субстохастические строки")
    bound = float(np.max(np.abs(reward[free]))) / (1.0 - rho)
    if not np.isfinite(bound):
        raise ValueError("Неконечная оценка хвоста переноса массы")
    return rho, bound


def solve_frozen_counts3d(
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
    """Estimate with merged integer occupancies and original transition laws.

    Counters count every initialized particle and actually executed move. A node's
    outgoing group is atomic for the global transition budget: it is not partly
    sent to fit a limit. Partial rewards are diagnostics, never a successful mean.
    Individual payoffs are not retained, so per-path standard errors are absent.
    """
    points, workspace, limit = _points_and_budget(operator, points, max_workspace_bytes)
    walks = _integer(walks, "walks", 2, 2**31 - 1)
    seed = _integer(seed, "seed", 0, 2**32 - 1)
    max_steps = _integer(max_steps, "max_steps", 1, 2**63 - 1)
    max_walks = _integer(max_walks, "max_walks", 1, 2**63 - 1)
    max_transitions = _integer(max_transitions, "max_transitions", 0, 2**63 - 1)
    started = perf_counter()
    result = FrozenTransportResult3D(
        "memory_limit",
        "estimated_workspace_exceeds_limit",
        points.copy(),
        None,
        None,
        None,
        0,
        0,
        walks,
        seed,
        0.0,
        0.0,
        workspace,
        "mc_counts",
    )
    if workspace > limit:
        result.seconds = perf_counter() - started
        return result
    warmup_transport3d()
    started = perf_counter()
    probabilities, reward, killing = frozen_mc_coefficients(operator)
    result.preparation_seconds = perf_counter() - started
    with NUMBA_LOCK:
        values, counts, launched, moves, updates, sweeps, transfers, code = (
            _counts_kernel(
                probabilities.reshape(-1, 6),
                reward.ravel(),
                killing.ravel(),
                operator.fixed.ravel(),
                points,
                operator.n + 1,
                walks,
                seed,
                max_steps,
                max_walks,
                max_transitions,
            )
        )
    result.status, result.reason = {
        0: ("completed", "all_particles_absorbed"),
        1: ("path_limit", "max_steps"),
        2: ("walk_budget", "max_walks"),
        3: ("walk_budget", "max_transitions"),
        4: ("nonfinite", "transport_reward"),
    }[code]
    result.values = values.copy() if code == 0 else None
    result.diagnostic_values = values
    result.completed_paths = counts
    result.walks, result.transitions = launched, moves
    result.node_updates, result.transport_sweeps, result.edge_transfers = (
        updates,
        sweeps,
        transfers,
    )
    result.seconds = perf_counter() - started
    return result


def solve_frozen_mass3d(
    operator,
    points=None,
    *,
    tolerance=1e-7,
    max_iterations=20000,
    max_workspace_bytes=256 * 1024**2,
):
    """Forward real mass from each queried source, bounded uncomputed tail.

    The initial mass is 1, with no particle count parameter or integer rounding.
    Success requires a relative bound to the unknown solution, not a small mass
    alone. Zero reward is exact; cancellation at zero may require the full tail.
    """
    points, workspace, limit = _points_and_budget(operator, points, max_workspace_bytes)
    tolerance = _relative_tolerance(tolerance)
    max_iterations = _integer(max_iterations, "max_iterations", 1, 1000000)
    started = perf_counter()
    result = FrozenTransportResult3D(
        "memory_limit",
        "estimated_workspace_exceeds_limit",
        points.copy(),
        None,
        None,
        None,
        0,
        0,
        None,
        None,
        0.0,
        0.0,
        workspace,
        "mass_transport",
    )
    if workspace > limit:
        result.seconds = perf_counter() - started
        return result
    warmup_transport3d()
    started = perf_counter()
    probabilities, reward, _ = frozen_mc_coefficients(operator)
    _, bound = _tail_parameters(probabilities, reward, operator.fixed)
    result.preparation_seconds = perf_counter() - started
    with NUMBA_LOCK:
        values, tails, updates, sweeps, transfers, code = _mass_kernel(
            probabilities.reshape(-1, 6),
            reward.ravel(),
            operator.fixed.ravel(),
            points,
            operator.n + 1,
            bound,
            tolerance,
            max_iterations,
        )
    result.status, result.reason = {
        0: ("completed", "relative_tail_bound"),
        4: ("nonfinite", "mass_reward_or_tail"),
        5: ("linear_limit", "max_iterations_before_relative_tail_bound"),
        6: ("nonfinite", "mass_arithmetic_underflow"),
    }[code]
    result.values = values.copy() if code == 0 else None
    result.diagnostic_values = values
    result.tail_error_bound = tails
    result.node_updates, result.transport_sweeps, result.edge_transfers = (
        updates,
        sweeps,
        transfers,
    )
    result.seconds = perf_counter() - started
    return result


@njit(cache=True, nogil=True)
def _mass_sweep(probabilities, reward, fixed, values, following, side):
    offsets = np.array([side * side, -side * side, side, -side, 1, -1])
    updates = 0
    transfers = 0
    for node in range(len(values)):
        value = reward[node]
        if not fixed[node]:
            updates += 1
            for direction in range(6):
                probability = probabilities[node, direction]
                if probability > 0:
                    value += probability * values[node + offsets[direction]]
                    transfers += 1
        following[node] = value
        if not np.isfinite(value):
            return updates, transfers, False
    return updates, transfers, True


@njit(cache=True, nogil=True)
def _relative_residual(diagonal, weights, rhs, fixed, values, side, scale):
    offsets = np.array([side * side, -side * side, side, -side, 1, -1])
    residual = 0.0
    for node in range(len(values)):
        if fixed[node]:
            continue
        value = diagonal[node] * values[node] - rhs[node]
        for direction in range(6):
            weight = weights[node, direction]
            if weight > 0:
                value -= weight * values[node + offsets[direction]]
        if not np.isfinite(value):
            return np.inf
        residual = max(residual, abs(value))
    if scale == 0:
        return 0.0 if residual == 0 else np.inf
    return residual / scale


def solve_mass_lines3d(operator, initial, *, tolerance, max_iterations, deadline=None):
    """Evaluate the same expected-reward series for every start simultaneously.

    u_{k+1}=r+P*u_k, starting from the supplied Picard iterate as the line solver
    does. This is the dual evaluation of forward mass with the initial field as
    the terminal payment, also called Jacobi iteration. With initial=0 it is
    precisely the forward expected-reward series. Actual relative frozen-system
    residual is the stopping criterion; no random sampling is involved.
    """
    started = perf_counter()
    if not isinstance(operator, HeatOperator3D):
        raise ValueError("Ожидается HeatOperator3D")
    tolerance = _relative_tolerance(tolerance)
    max_iterations = _integer(max_iterations, "max_iterations", 1, 1000000)
    if deadline is not None and not np.isfinite(deadline):
        raise ValueError("deadline должен быть конечным")
    initial = np.asarray(initial)
    if initial.shape != operator.shape:
        raise ValueError("Начальное приближение должно иметь форму оператора")
    if not np.isfinite(initial).all():
        return LinearResult3D(
            initial.copy(),
            "nonfinite",
            "nonfinite_initial",
            np.inf,
            preparation_seconds=perf_counter() - started,
        )
    probabilities, reward, _ = frozen_mc_coefficients(operator)
    rho, _ = _tail_parameters(probabilities, reward, operator.fixed)
    free = ~operator.fixed
    scale = float(np.max(np.abs(operator.rhs[free]))) if np.any(free) else 0.0
    values = np.array(initial, dtype=float, order="C", copy=True).ravel()
    values[operator.fixed.ravel()] = reward[operator.fixed]
    if scale == 0:
        # Strict absorption makes the homogeneous free system nonsingular.
        # Its exact zero solution needs no relative scale or absolute fallback.
        values[free.ravel()] = 0.0
        return LinearResult3D(
            values.reshape(operator.shape),
            "completed",
            "zero_rhs",
            0.0,
            tail_error_bound=0.0,
            preparation_seconds=perf_counter() - started,
        )
    initial_defect = (
        operator.rhs - operator.apply(values.reshape(operator.shape))
    ) / operator.diagonal
    bound = (
        float(np.max(np.abs(initial_defect[free]))) / (1.0 - rho)
        if np.any(free)
        else 0.0
    )
    if not np.isfinite(bound):
        return LinearResult3D(
            values.reshape(operator.shape),
            "nonfinite",
            "nonfinite_initial_defect",
            np.inf,
            preparation_seconds=perf_counter() - started,
        )
    following = np.empty_like(values)
    probabilities = probabilities.reshape(-1, 6)
    weights = operator.weights.reshape(-1, 6)
    diagonal, rhs, fixed = (
        operator.diagonal.ravel(),
        operator.rhs.ravel(),
        operator.fixed.ravel(),
    )
    updates = 0
    transfers = 0
    tail = bound
    prepared = perf_counter() - started
    with NUMBA_LOCK:
        for iteration in range(max_iterations + 1):
            residual = _relative_residual(
                diagonal, weights, rhs, fixed, values, operator.n + 1, scale
            )
            status, reason = "linear_limit", "max_iterations"
            if not np.isfinite(residual):
                status, reason = "nonfinite", "nonfinite_mass_residual"
            elif residual <= tolerance:
                status, reason = "completed", "linear_residual"
            elif deadline is not None and perf_counter() >= deadline:
                status, reason = "time_limit", "mass_transport_deadline"
            elif iteration < max_iterations:
                added, edges, finite = _mass_sweep(
                    probabilities,
                    reward.ravel(),
                    fixed,
                    values,
                    following,
                    operator.n + 1,
                )
                updates += added
                transfers += edges
                values, following = following, values
                if tail is not None:
                    next_tail = tail * rho
                    tail = (
                        None
                        if tail > 0 and rho > 0 and next_tail < np.finfo(float).tiny
                        else next_tail
                    )
                if finite:
                    continue
                status, reason = "nonfinite", "nonfinite_mass_update"
                iteration += 1
            return LinearResult3D(
                values.reshape(operator.shape),
                status,
                reason,
                residual,
                iteration,
                node_updates=updates,
                transport_sweeps=iteration,
                edge_transfers=transfers,
                tail_error_bound=tail,
                preparation_seconds=prepared,
            )
    raise RuntimeError("Не завершён перенос массы")


_warmed = False


def warmup_transport3d():
    """Compile mutable frontier kernels and immutable operator signatures once."""
    global _warmed
    with NUMBA_LOCK:
        if _warmed:
            return
        shape = (3, 3, 3)
        operator = HeatOperator3D(
            np.ones(shape),
            np.zeros((*shape, 6)),
            np.zeros(shape),
            np.zeros(shape, dtype=bool),
            np.zeros(shape),
            np.ones(shape),
        )
        probabilities, reward, killing = frozen_mc_coefficients(operator)
        fixed = operator.fixed.ravel()
        probabilities = probabilities.reshape(-1, 6)
        points = np.array([[1, 1, 1]], dtype=np.int64)
        _counts_kernel(
            probabilities,
            reward.ravel(),
            killing.ravel(),
            fixed,
            points,
            3,
            2,
            0,
            1,
            2,
            1,
        )
        _mass_kernel(probabilities, reward.ravel(), fixed, points, 3, 0.0, 1e-7, 1)
        values = np.zeros(operator.rhs.size)
        following = np.empty_like(values)
        _mass_sweep(probabilities, reward.ravel(), fixed, values, following, 3)
        _relative_residual(
            operator.diagonal.ravel(),
            operator.weights.reshape(-1, 6),
            operator.rhs.ravel(),
            fixed,
            values,
            3,
            1.0,
        )
        _warmed = True
