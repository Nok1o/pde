"""Directional residual correction and absorbing walks for the SAME stencil."""

from time import perf_counter

import numpy as np
from numba import njit

from .boundary import BoundaryOperator, BoundaryProblem, make_boundary_problem
from .problem import Problem
from .solvers import NUMBA_LOCK, Result, solve

METHODS_BOUNDARY = {"lod_serial": "ЛОМ", "mc": "Монте-Карло"}


def select_boundary_points(n, mode="point", point=None):
    if mode in ("point", "patch"):
        x, y = (n // 2, n // 2) if point is None else point
        y, x = validate_boundary_points(n, [(y, x)])[0]
    if mode == "point":
        points = [(y, x)]
    elif mode == "patch":
        points = [
            (j, i)
            for j in range(max(0, y - 1), min(n, y + 1) + 1)
            for i in range(max(0, x - 1), min(n, x + 1) + 1)
        ]
    elif mode == "field":
        points = [(y, x) for y in range(n + 1) for x in range(n + 1)]
    else:
        raise ValueError("Запрос: point, patch или field")
    return validate_boundary_points(n, points)


def validate_boundary_points(n, points):
    points = np.asarray(points)
    if (
        points.ndim != 2
        or points.shape[1] != 2
        or len(points) == 0
        or not np.isfinite(points).all()
        or np.any(points != np.floor(points))
        or np.any(points < 0)
        or np.any(points > n)
    ):
        raise ValueError("Нужны целые индексы (y,x) от 0 до n")
    points = np.ascontiguousarray(points, dtype=np.int64)
    if len(np.unique(points, axis=0)) != len(points):
        raise ValueError("Запрошенные точки не должны повторяться")
    return points


@njit(cache=True)
def _factor_lines(lower, diagonal, upper):
    alpha = np.zeros_like(diagonal)
    denominators = np.empty_like(diagonal)
    for line in range(diagonal.shape[0]):
        previous = 0.0
        for i in range(diagonal.shape[1]):
            denominator = diagonal[line, i] + lower[line, i] * previous
            denominators[line, i] = denominator
            alpha[line, i] = -upper[line, i] / denominator
            previous = alpha[line, i]
    return alpha, denominators


def _directional_factors(operator, tau, axis):
    minus, plus = (3, 2) if axis == 0 else (1, 0)
    reaction = operator.reaction_x if axis == 0 else operator.reaction_y
    lower = -tau * operator.weights[..., minus]
    upper = -tau * operator.weights[..., plus]
    diagonal = 1 + tau * (
        operator.weights[..., minus] + operator.weights[..., plus] + reaction
    )
    lower[operator.fixed] = 0
    upper[operator.fixed] = 0
    diagonal[operator.fixed] = 1
    if axis == 1:
        lower, diagonal, upper = [
            np.ascontiguousarray(a.T) for a in (lower, diagonal, upper)
        ]
    alpha, denominators = _factor_lines(lower, diagonal, upper)
    return lower, alpha, denominators


@njit(cache=True, nogil=True)
def _sweep(rhs, out, lower, alpha, denominators, beta, axis):
    size = rhs.shape[0]
    for line in range(size):
        previous = 0.0
        for i in range(size):
            value = rhs[line, i] if axis == 0 else rhs[i, line]
            beta[i] = (value - lower[line, i] * previous) / denominators[line, i]
            previous = beta[i]
        following = 0.0
        for i in range(size - 1, -1, -1):
            value = alpha[line, i] * following + beta[i]
            if axis == 0:
                out[line, i] = value
            else:
                out[i, line] = value
            following = value


def _lod(operator, tol, max_iter, record):
    n = operator.problem.n
    u = operator.boundary.copy()
    rhs = operator.rhs()
    initial = rhs - operator.apply(u)
    scale = float(np.max(np.abs(initial)))
    u = operator.correct_constant(u)
    tau = 1 / (2 * np.pi * n)
    x_factors = _directional_factors(operator, tau, 0)
    y_factors = _directional_factors(operator, tau, 1)
    first = np.empty_like(u)
    correction = np.empty_like(u)
    beta = np.empty(n + 1)
    history = []
    for iteration in range(max_iter + 1):
        residual = rhs - operator.apply(u)
        absolute = float(np.max(np.abs(residual)))
        relative = absolute / scale if scale else 0.0
        if record and (
            iteration % max(1, n) == 0 or relative <= tol or iteration == max_iter
        ):
            history.append((iteration, relative))
        if relative <= tol or iteration == max_iter:
            break
        _sweep(tau * residual, first, *x_factors, beta, 0)
        _sweep(first, correction, *y_factors, beta, 1)
        u += correction
        u = operator.correct_constant(u)
        if not np.isfinite(u).all():
            raise RuntimeError("ЛОМ вернул неконечное поле")
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
            "history": history,
            "residual_scale": scale,
            "stopping_rule": "relative_boundary_system_inf_v1",
            "tau": tau,
        },
    )


@njit(cache=True, nogil=True)
def _walks(
    fixed, boundary, reward, killing, conditional_cdf, points, walks, seed, max_steps
):
    np.random.seed(seed)
    means = np.zeros(len(points))
    deviations = np.zeros(len(points))
    samples = np.empty(walks)
    total_steps = 0
    for point_index in range(len(points)):
        start_y, start_x = points[point_index]
        if fixed[start_y, start_x]:
            means[point_index] = boundary[start_y, start_x]
            if point_index == 0:
                samples[:] = boundary[start_y, start_x]
            continue
        for walk_index in range(walks):
            y, x = points[point_index]
            payoff = 0.0
            steps = 0
            while not fixed[y, x]:
                if steps >= max_steps:
                    raise RuntimeError(
                        "Превышен предел шагов пути; усечённый ответ не возвращён"
                    )
                payoff += reward[y, x]
                draw = np.random.random()
                steps += 1
                if draw < killing[y, x]:
                    break
                # Conditional direction after surviving Robin absorption.
                draw = (draw - killing[y, x]) / (1 - killing[y, x])
                if draw < conditional_cdf[y, x, 0]:
                    y += 1
                elif draw < conditional_cdf[y, x, 1]:
                    y -= 1
                elif draw < conditional_cdf[y, x, 2]:
                    x += 1
                else:
                    x -= 1
            else:
                payoff += boundary[y, x]
            total_steps += steps
            # Welford avoids subtracting two large second moments.
            difference = payoff - means[point_index]
            means[point_index] += difference / (walk_index + 1)
            deviations[point_index] += difference * (payoff - means[point_index])
            if point_index == 0:
                samples[walk_index] = payoff
    errors = np.sqrt(np.maximum(deviations, 0.0) / (walks * (walks - 1)))
    return means, errors, total_steps, samples


def _mc(operator, points, walks, seed, max_steps):
    if all(c.kind == "dirichlet" for c in operator.problem.sides.values()):
        # The old estimator and its RNG stream remain exactly reproducible.
        legacy = Problem(
            operator.problem.name,
            operator.problem.n,
            operator.problem.f,
            operator.boundary,
        )
        interior = ~operator.fixed[tuple(points.T)]
        if interior.any():
            result = solve(
                legacy,
                points[interior],
                "mc",
                walks=walks,
                seed=seed,
                max_steps=max_steps,
            )
        else:
            values = np.where(operator.fixed, operator.boundary, np.nan)
            result = Result(
                "mc",
                values,
                np.where(operator.fixed, 0.0, np.nan),
                np.zeros_like(values, dtype=np.int64),
            )
        if not interior[0]:
            result.extras["samples"] = np.full(
                walks, operator.boundary[tuple(points[0])]
            )
        result.walks = int(np.count_nonzero(interior)) * walks
        return result
    _, reward, killing = operator.mc_coefficients()
    directions = operator.weights / operator.weights.sum(axis=-1)[..., None]
    cdf = np.cumsum(directions, axis=-1)
    means, errors, steps, samples = _walks(
        operator.fixed,
        operator.boundary,
        reward,
        killing,
        cdf,
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
        walks=walks * int(np.count_nonzero(~operator.fixed[indices])),
        extras={"samples": samples},
    )


def _integer(value, name, minimum):
    if (
        isinstance(value, bool)
        or not np.isfinite(value)
        or int(value) != value
        or value < minimum
    ):
        raise ValueError(f"{name} должно быть целым ≥ {minimum}")
    return int(value)


def solve_boundary(
    problem,
    points,
    method="lod_serial",
    *,
    walks=2048,
    seed=42,
    tol=1e-7,
    max_iter=20000,
    max_steps=None,
    record=True,
):
    if not isinstance(problem, BoundaryProblem):
        raise ValueError("Ожидается BoundaryProblem")
    if method not in METHODS_BOUNDARY:
        raise ValueError("Для смешанной границы доступны только ЛОМ и обычный MC")
    points = validate_boundary_points(problem.n, points)
    walks = _integer(walks, "walks", 2)
    max_iter = _integer(max_iter, "max_iter", 1)
    seed = _integer(seed, "seed", 0)
    if seed >= 2**32:
        raise ValueError("seed должен быть меньше 2³²")
    if not np.isfinite(tol) or not 0 < tol < 1:
        raise ValueError("Относительный допуск должен быть между 0 и 1")
    max_steps = (
        1000 * problem.n**2
        if max_steps is None
        else _integer(max_steps, "max_steps", 1)
    )
    with NUMBA_LOCK:
        started = perf_counter()
        operator = BoundaryOperator(problem)
        operator.check_compatibility()
        if method == "mc" and operator.pure_neumann:
            raise ValueError(
                "MC для чистого Неймана пока не поддерживается: нет поглощения. Используйте ЛОМ или прямой решатель с нулевым средним."
            )
        result = (
            _lod(operator, tol, max_iter, record)
            if method == "lod_serial"
            else _mc(operator, points, walks, seed, max_steps)
        )
        result.elapsed = perf_counter() - started
        result.extras["normalization"] = (
            "volume_weighted_zero_mean" if operator.pure_neumann else None
        )
    return result


_warmed = False


def warmup_boundary():
    global _warmed
    with NUMBA_LOCK:
        if _warmed:
            return
        problem = make_boundary_problem(n=4)
        points = select_boundary_points(4)
        for method in METHODS_BOUNDARY:
            solve_boundary(problem, points, method, walks=4, max_iter=2, record=False)
        from .solvers import warmup

        warmup()
        _warmed = True


def trace_boundary_walk(problem, point=None, seed=2026, max_steps=None):
    """Independent explanatory path; every flux and absorption is recorded."""
    operator = BoundaryOperator(problem)
    operator.check_compatibility()
    if operator.pure_neumann:
        raise ValueError("Для чистого Неймана поглощающей траектории нет")
    y, x = select_boundary_points(problem.n, point=point)[0]
    seed = _integer(seed, "seed", 0)
    if seed >= 2**32:
        raise ValueError("seed должен быть меньше 2³²")
    max_steps = (
        1000 * problem.n**2
        if max_steps is None
        else _integer(max_steps, "max_steps", 1)
    )
    rng = np.random.default_rng(seed)
    _, reward, killing = operator.mc_coefficients()
    history = []
    payoff = 0.0
    for step in range(max_steps + 1):
        sides = [
            s
            for s, yes in zip(
                ("left", "right", "bottom", "top"),
                (x == 0, x == problem.n, y == 0, y == problem.n),
            )
            if yes
        ]
        event = {"step": step, "x": int(x), "y": int(y), "sides": sides}
        if operator.fixed[y, x]:
            payoff += operator.boundary[y, x]
            history.append(
                {
                    **event,
                    "event": "dirichlet",
                    "source": 0.0,
                    "flux": 0.0,
                    "boundary": float(operator.boundary[y, x]),
                    "payoff": float(payoff),
                }
            )
            return history
        if step == max_steps:
            raise RuntimeError(
                "Траектория не завершилась; усечённый ответ не возвращён"
            )
        source = -problem.f[y, x] / operator.diagonal[y, x]
        flux = operator.boundary_load[y, x] / operator.diagonal[y, x]
        payoff += reward[y, x]
        draw = rng.random()
        absorbed = draw < killing[y, x]
        event_name = (
            "robin_absorption"
            if absorbed
            else (
                "robin"
                if operator.reaction[y, x] > 0
                else "neumann" if sides else "interior"
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
        draw = (draw - killing[y, x]) / (1 - killing[y, x])
        cdf = np.cumsum(operator.weights[y, x] / operator.weights[y, x].sum())
        direction = min(int(np.searchsorted(cdf, draw, side="right")), 3)
        dy, dx = ((1, 0), (-1, 0), (0, 1), (0, -1))[direction]
        y, x = y + dy, x + dx
    raise RuntimeError("Траектория не завершилась")
