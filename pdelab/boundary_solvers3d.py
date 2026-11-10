"""Three directional sweeps and six-neighbor reflected/absorbed grid walks."""

from time import perf_counter

import numpy as np
from numba import njit

from .boundary3d import (
    FACE_AXES,
    BoundaryOperator3D,
    BoundaryProblem3D,
    make_boundary_problem3d,
    select_boundary_points3d,
)
from .boundary_solvers import _factor_lines
from .problem3d import Problem3D, validate_points3d
from .solvers import NUMBA_LOCK, Result
from .solvers3d import METHODS_3D, solve3d, splitting_parameter3d, warmup3d


def _factors(operator, tau, axis):
    minus, plus = operator.weights[..., 2 * axis + 1], operator.weights[..., 2 * axis]
    lower, upper = -tau * minus, -tau * plus
    diagonal = 1 + tau * (minus + plus + operator.reaction_axes[axis])
    lower[operator.fixed], upper[operator.fixed], diagonal[operator.fixed] = 0, 0, 1
    size = operator.problem.n + 1
    lower, diagonal, upper = [
        np.ascontiguousarray(np.moveaxis(a, axis, -1)).reshape(-1, size)
        for a in (lower, diagonal, upper)
    ]
    alpha, denominators = _factor_lines(lower, diagonal, upper)
    return lower, alpha, denominators


@njit(cache=True, nogil=True)
def _sweep(rhs, out, lower, alpha, denominators, beta, axis):
    size = rhs.shape[0]
    for first in range(size):
        for second in range(size):
            line = first * size + second
            previous = 0.0
            for i in range(size):
                if axis == 2:
                    value = rhs[first, second, i]
                elif axis == 1:
                    value = rhs[first, i, second]
                else:
                    value = rhs[i, first, second]
                beta[i] = (value - lower[line, i] * previous) / denominators[line, i]
                previous = beta[i]
            following = 0.0
            for i in range(size - 1, -1, -1):
                value = alpha[line, i] * following + beta[i]
                if axis == 2:
                    out[first, second, i] = value
                elif axis == 1:
                    out[first, i, second] = value
                else:
                    out[i, first, second] = value
                following = value


def _lod(operator, tol, max_iter):
    u = operator.boundary.copy()
    rhs = operator.rhs()
    residual = rhs - operator.apply(u)
    scale = float(np.max(np.abs(residual)))
    # Reuse the cube's constant-coefficient shift as an iteration parameter.
    # Its Dirichlet modal convergence prediction does NOT apply to mixed faces.
    tau, _ = splitting_parameter3d(operator.problem.n)
    factors = {axis: _factors(operator, tau, axis) for axis in (2, 1, 0)}
    first, second = np.empty_like(u), np.empty_like(u)
    beta = np.empty(operator.problem.n + 1)
    history = []
    for iteration in range(max_iter + 1):
        absolute = float(np.max(np.abs(residual)))
        if not np.isfinite(absolute):
            raise RuntimeError("Неконечная невязка граничного 3D-решателя")
        relative = (
            absolute / scale if scale else (0.0 if absolute == 0 else float("inf"))
        )
        history.append(relative)
        if relative <= tol or iteration == max_iter:
            break
        _sweep(tau * residual, first, *factors[2], beta, 2)
        _sweep(first, second, *factors[1], beta, 1)
        _sweep(second, first, *factors[0], beta, 0)
        u += first
        if operator.pure_neumann:
            u = operator.normalize(u)
        residual = rhs - operator.apply(u)
    return Result(
        "lod_serial",
        u,
        np.full_like(u, np.nan),
        np.zeros_like(u, dtype=np.int64),
        iterations=iteration,
        residual=absolute,
        relative_residual=relative,
        converged=relative <= tol,
        extras={
            "relative_residuals": np.asarray(history),
            "residual_scale": scale,
            "stopping_rule": "relative_boundary_system_inf_3d_v1",
            "tolerance": tol,
            "tau": tau,
        },
    )


@njit(inline="always")
def _reflected_step(z, y, x, n, direction):
    # Each outward proposal is reflected into the cube, giving inward weight 2.
    if direction == 0:
        z += 1 if z < n else -1
    elif direction == 1:
        z -= 1 if z > 0 else -1
    elif direction == 2:
        y += 1 if y < n else -1
    elif direction == 3:
        y -= 1 if y > 0 else -1
    elif direction == 4:
        x += 1 if x < n else -1
    else:
        x -= 1 if x > 0 else -1
    return z, y, x


@njit(cache=True, nogil=True)
def _walks(fixed, boundary, reward, killing, points, walks, seed, max_steps):
    np.random.seed(seed)
    n = fixed.shape[0] - 1
    means, deviations = np.zeros(len(points)), np.zeros(len(points))
    samples = np.empty(walks)
    total_steps = 0
    for point_index in range(len(points)):
        start_z, start_y, start_x = points[point_index]
        if fixed[start_z, start_y, start_x]:
            means[point_index] = boundary[start_z, start_y, start_x]
            if point_index == 0:
                samples[:] = means[point_index]
            continue
        for walk_index in range(walks):
            z, y, x = start_z, start_y, start_x
            payoff, steps = 0.0, 0
            while not fixed[z, y, x]:
                if steps >= max_steps:
                    raise RuntimeError(
                        "Превышен предел шагов 3D-пути; усечённый ответ не возвращён"
                    )
                payoff += reward[z, y, x]
                steps += 1
                if killing[z, y, x] > 0 and np.random.random() < killing[z, y, x]:
                    break
                z, y, x = _reflected_step(z, y, x, n, np.random.randint(6))
            else:
                payoff += boundary[z, y, x]
            total_steps += steps
            difference = payoff - means[point_index]
            means[point_index] += difference / (walk_index + 1)
            deviations[point_index] += difference * (payoff - means[point_index])
            if point_index == 0:
                samples[walk_index] = payoff
    return (
        means,
        np.sqrt(np.maximum(deviations, 0) / (walks * (walks - 1))),
        total_steps,
        samples,
    )


def _mc(operator, points, walks, seed, max_steps):
    _, reward, killing = operator.mc_coefficients()
    means, errors, steps, samples = _walks(
        operator.fixed,
        operator.boundary,
        reward,
        killing,
        points,
        walks,
        seed,
        max_steps,
    )
    values = np.where(operator.fixed, operator.boundary, np.nan)
    stderr = np.where(operator.fixed, 0.0, np.nan)
    counts = np.zeros_like(values, dtype=np.int64)
    indices = tuple(points.T)
    values[indices], stderr[indices] = means, errors
    counts[indices] = np.where(operator.fixed[indices], 0, walks)
    return Result(
        "mc",
        values,
        stderr,
        counts,
        steps=steps,
        walks=int(counts.sum()),
        extras={"samples": samples},
    )


def solve_boundary3d(
    problem,
    points,
    method="lod_serial",
    *,
    walks=2048,
    seed=42,
    tol=1e-8,
    max_iter=20000,
    max_steps=None,
):
    if not isinstance(problem, BoundaryProblem3D):
        raise ValueError("Ожидается BoundaryProblem3D")
    if method not in METHODS_3D:
        raise ValueError("В 3D доступны только ЛОМ и обычный MC")
    points = validate_points3d(problem.n, points)
    for name, value, minimum, maximum in (
        ("walks", walks, 2, 2**31 - 1),
        ("seed", seed, 0, 2**32 - 1),
        ("max_iter", max_iter, 1, 2**31 - 1),
    ):
        if (
            isinstance(value, (bool, np.bool_))
            or not np.isfinite(value)
            or int(value) != value
            or not minimum <= value <= maximum
        ):
            raise ValueError(f"{name}: нужно целое в [{minimum}, {maximum}]")
    if not np.isfinite(tol) or not 0 < tol < 1:
        raise ValueError("Относительный допуск должен лежать между 0 и 1")
    limit = 1000 * problem.n**2 if max_steps is None else max_steps
    if (
        isinstance(limit, (bool, np.bool_))
        or not np.isfinite(limit)
        or int(limit) != limit
        or limit < 1
    ):
        raise ValueError("max_steps должно быть положительным целым")
    with NUMBA_LOCK:
        started = perf_counter()
        operator = BoundaryOperator3D(problem)
        operator.check_compatibility()
        if operator.pure_neumann and method == "mc":
            raise ValueError(
                "MC для чистого Неймана пока не поддерживается: нет поглощения"
            )
        if all(c.kind == "dirichlet" for c in problem.faces.values()):
            legacy = Problem3D(
                problem.name, problem.n, problem.f, operator.boundary, problem.exact
            )
            result = solve3d(
                legacy,
                points,
                method,
                int(walks),
                int(seed),
                tol,
                int(max_iter),
                int(limit),
            )
        else:
            result = (
                _lod(operator, tol, int(max_iter))
                if method == "lod_serial"
                else _mc(operator, points, int(walks), int(seed), int(limit))
            )
        result.elapsed = perf_counter() - started
        result.extras["normalization"] = (
            "volume_weighted_zero_mean" if operator.pure_neumann else None
        )
        return result


_warmed = False


def warmup_boundary3d():
    global _warmed
    with NUMBA_LOCK:
        if _warmed:
            return
        p = make_boundary_problem3d(n=4)
        for method in METHODS_3D:
            solve_boundary3d(p, [[2, 2, 2]], method, walks=4, max_iter=1)
        warmup3d()
        _warmed = True


def trace_boundary_walk3d(problem, point=None, seed=2026, max_steps=20000):
    operator = BoundaryOperator3D(problem)
    operator.check_compatibility()
    if operator.pure_neumann:
        raise ValueError("Для чистого Неймана поглощающей траектории нет")
    if (
        isinstance(max_steps, (bool, np.bool_))
        or not np.isfinite(max_steps)
        or int(max_steps) != max_steps
        or max_steps < 1
    ):
        raise ValueError("max_steps должно быть положительным целым")
    position = tuple(select_boundary_points3d(problem.n, point=point)[0])
    rng = np.random.default_rng(seed)
    payoff, history = 0.0, []
    _, reward, killing = operator.mc_coefficients()
    for step in range(int(max_steps) + 1):
        faces = [
            name
            for name, (axis, end) in FACE_AXES.items()
            if position[axis] == (0 if end == 0 else problem.n)
        ]
        z, y, x = position
        event = {"step": step, "x": int(x), "y": int(y), "z": int(z), "faces": faces}
        if operator.fixed[position]:
            value = float(operator.boundary[position])
            history.append(
                {
                    **event,
                    "event": "dirichlet",
                    "source": 0.0,
                    "flux": 0.0,
                    "boundary": value,
                    "payoff": payoff + value,
                }
            )
            return history
        if step == max_steps:
            raise RuntimeError(
                "Траектория не завершилась; усечённый ответ не возвращён"
            )
        source = -problem.f[position] / operator.diagonal[position]
        flux = operator.boundary_load[position] / operator.diagonal[position]
        payoff += reward[position]
        absorbed = killing[position] > 0 and rng.random() < killing[position]
        event_name = (
            "robin_absorption"
            if absorbed
            else (
                "robin"
                if operator.reaction[position] > 0
                else "neumann" if faces else "interior"
            )
        )
        history.append(
            {
                **event,
                "event": event_name,
                "source": float(source),
                "flux": float(flux),
                "boundary": 0.0,
                "payoff": float(payoff),
            }
        )
        if absorbed:
            return history
        position = _reflected_step(z, y, x, problem.n, rng.integers(6))
    raise RuntimeError("Траектория не завершилась")
