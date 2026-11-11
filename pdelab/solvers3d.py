"""Последовательный ЛОМ и шестисоседнее MC для семиточечной системы."""

from time import perf_counter

import numpy as np
from numba import njit

from .problem3d import blank_field3d, laplacian3d, make_problem3d, validate_points3d
from .solvers import NUMBA_LOCK, Result

METHODS_3D = {
    "lod_serial": "ЛОМ",
    "mc": "Монте-Карло",
    "graded": "MC · сгущённая сетка",
    "wos": "WoS · 1-й род",
}
STOPPING_RULE_3D = "relative_rhs_inf_3d_v1"


def splitting_parameter3d(n):
    """Численный minimax по модам куба для одного постоянного τ.

    При фиксированных двух λ отношение τΣλ/Π(1+τλ) монотонно по третьему.
    Поэтому достаточно четырёх сочетаний λ_min/λ_max с учётом симметрии осей.
    Это выбор на логарифмической сетке τ, не утверждение точного оптимума.
    """
    low = 4 * n**2 * np.sin(np.pi / (2 * n)) ** 2
    high = 4 * n**2 * np.cos(np.pi / (2 * n)) ** 2
    candidates = np.geomspace(0.1 / high, 10 / low, 512)
    reductions = []
    for high_count in range(4):
        eigenvalues = np.array([high] * high_count + [low] * (3 - high_count))
        reductions.append(
            candidates
            * eigenvalues.sum()
            / np.prod(1 + candidates[:, None] * eigenvalues, axis=1)
        )
    worst_reduction = np.min(reductions, axis=0)
    best = int(np.argmax(worst_reduction))
    return float(candidates[best]), float(1 - worst_reduction[best])


@njit(cache=True, nogil=True)
def sweep3d(rhs, ratio, axis):
    """Прогонка (I - τ L_axis) v = rhs; axis=2: x, 1: y, 0: z."""
    size = rhs.shape[0]
    solution = np.empty_like(rhs)
    alpha = np.zeros(size + 1)
    denominator = np.empty(size)
    beta = np.empty(size + 1)
    a, b, c = -ratio, 1 + 2 * ratio, -ratio
    for i in range(size):
        denominator[i] = a * alpha[i] + b
        alpha[i + 1] = -c / denominator[i]
    for first in range(size):
        for second in range(size):
            beta[0] = 0.0
            for i in range(size):
                if axis == 2:
                    value = rhs[first, second, i]
                elif axis == 1:
                    value = rhs[first, i, second]
                else:
                    value = rhs[i, first, second]
                beta[i + 1] = (value - a * beta[i]) / denominator[i]
            following = 0.0
            for i in range(size - 1, -1, -1):
                value = alpha[i + 1] * following + beta[i + 1]
                if axis == 2:
                    solution[first, second, i] = value
                elif axis == 1:
                    solution[first, i, second] = value
                else:
                    solution[i, first, second] = value
                following = value
    return solution


def _lod3d(problem, tol, max_iter, tau):
    values = blank_field3d(problem)
    values[1:-1, 1:-1, 1:-1] = 0.0
    default_tau, predicted_factor = splitting_parameter3d(problem.n)
    tau = default_tau if tau is None else tau
    ratio = tau / problem.h**2
    residual = laplacian3d(values, problem.h) - problem.f[1:-1, 1:-1, 1:-1]
    scale = float(np.max(np.abs(residual)))
    history = []
    for iteration in range(max_iter + 1):
        absolute = float(np.max(np.abs(residual)))
        if not np.isfinite(absolute):
            raise RuntimeError("Неконечная невязка 3D-решателя")
        relative = (
            absolute / scale if scale else (0.0 if absolute == 0 else float("inf"))
        )
        history.append(relative)
        if relative <= tol or iteration == max_iter:
            break
        correction = tau * residual
        for axis in (2, 1, 0):
            correction = sweep3d(correction, ratio, axis)
        values[1:-1, 1:-1, 1:-1] += correction
        residual = laplacian3d(values, problem.h) - problem.f[1:-1, 1:-1, 1:-1]
    return Result(
        method="lod_serial",
        values=values,
        stderr=np.full_like(values, np.nan),
        sample_count=np.zeros_like(values, dtype=np.int64),
        iterations=iteration,
        residual=absolute,
        relative_residual=relative,
        converged=relative <= tol,
        extras={
            "relative_residuals": np.array(history),
            "residual_scale": scale,
            "stopping_rule": STOPPING_RULE_3D,
            "tolerance": tol,
            "tau": tau,
            "predicted_factor": predicted_factor if tau == default_tau else None,
        },
    )


@njit(cache=True, nogil=True)
def _mc3d(source, boundary, points, walks, seed, max_steps):
    np.random.seed(seed)
    n = source.shape[0] - 1
    means = np.zeros(len(points))
    squared_deviations = np.zeros(len(points))
    counts = np.zeros(len(points), dtype=np.int64)
    samples = np.empty(walks)
    total_steps = 0
    for point_index in range(len(points)):
        start_z, start_y, start_x = points[point_index]
        if (
            start_z == 0
            or start_z == n
            or start_y == 0
            or start_y == n
            or start_x == 0
            or start_x == n
        ):
            means[point_index] = boundary[start_z, start_y, start_x]
            if point_index == 0:
                samples[:] = means[point_index]
            continue
        for walk_index in range(walks):
            z, y, x = start_z, start_y, start_x
            reward = 0.0
            steps = 0
            while 0 < z < n and 0 < y < n and 0 < x < n:
                if steps >= max_steps:
                    raise RuntimeError(
                        "Превышен предел шагов 3D-пути; ответ не возвращён"
                    )
                reward -= source[z, y, x] / (6 * n * n)
                direction = np.random.randint(6)
                if direction == 0:
                    x += 1
                elif direction == 1:
                    x -= 1
                elif direction == 2:
                    y += 1
                elif direction == 3:
                    y -= 1
                elif direction == 4:
                    z += 1
                else:
                    z -= 1
                steps += 1
            reward += boundary[z, y, x]
            counts[point_index] += 1
            difference = reward - means[point_index]
            means[point_index] += difference / counts[point_index]
            squared_deviations[point_index] += difference * (
                reward - means[point_index]
            )
            total_steps += steps
            if point_index == 0:
                samples[walk_index] = reward
    return means, squared_deviations, counts, total_steps, samples


def solve3d(
    problem,
    points,
    method="mc",
    walks=2048,
    seed=42,
    tol=1e-7,
    max_iter=20000,
    max_steps=None,
    tau=None,
):
    points = validate_points3d(problem.n, points)
    if method in ("wos", "graded"):
        from .stationary_extensions import solve_extended

        return solve_extended(problem, points, method, walks, seed, max_steps)
    if method not in METHODS_3D:
        raise ValueError(
            "В 3D доступны только lod_serial и mc; хвосты и слои ещё не перенесены"
        )
    for name, value, lower, upper in (
        ("walks", walks, 2, 2**31 - 1),
        ("seed", seed, 0, 2**32 - 1),
        ("max_iter", max_iter, 1, 2**31 - 1),
    ):
        if (
            isinstance(value, (bool, np.bool_))
            or not np.isfinite(value)
            or int(value) != value
            or not lower <= value <= upper
        ):
            raise ValueError(f"{name}: нужно целое в [{lower}, {upper}]")
    if not np.isfinite(tol) or not 0 < tol < 1:
        raise ValueError("Относительный допуск должен лежать между 0 и 1")
    if tau is not None and (not np.isfinite(tau) or tau <= 0):
        raise ValueError("τ должен быть положительным конечным числом")
    limit = 100 * problem.n**2 if max_steps is None else max_steps
    if not np.isfinite(limit) or int(limit) != limit or limit < 1:
        raise ValueError("max_steps должно быть положительным целым")
    with NUMBA_LOCK:
        started = perf_counter()
        if method == "lod_serial":
            result = _lod3d(problem, tol, int(max_iter), tau)
        else:
            means, squared, counts, steps, samples = _mc3d(
                problem.f, problem.boundary, points, int(walks), int(seed), int(limit)
            )
            values = blank_field3d(problem)
            errors = np.full_like(values, np.nan)
            sample_count = np.zeros_like(values, dtype=np.int64)
            indices = tuple(points.T)
            values[indices] = means
            errors[indices] = np.sqrt(
                np.maximum(squared, 0)
                / (np.maximum(counts - 1, 1) * np.maximum(counts, 1))
            )
            sample_count[indices] = counts
            result = Result(
                method="mc",
                values=values,
                stderr=errors,
                sample_count=sample_count,
                steps=steps,
                walks=int(counts.sum()),
                extras={"samples": samples},
            )
        result.elapsed = perf_counter() - started
    return result


_warmed = False


def warmup3d():
    global _warmed
    with NUMBA_LOCK:
        if _warmed:
            return
        problem = make_problem3d(n=4)
        for method in METHODS_3D:
            if method in ("wos", "graded"):
                continue
            solve3d(problem, [(2, 2, 2)], method, walks=4, max_iter=1)
        _warmed = True
