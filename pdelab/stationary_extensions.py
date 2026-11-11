"""Point estimators on reconstructed input data; never use the exact solution.

WoS solves the continuous Dirichlet problem with multilinear f/g data.
Graded MC solves nodal finite volumes, including reflecting/Robin faces.
Both return estimates at the ORIGINAL physical query points.
"""

from time import perf_counter

import numpy as np
from numba import njit
from scipy.interpolate import RegularGridInterpolator

from .graded import graded_axis

_warmed_dimensions = set()

EXTENDED_METHODS = {"wos": "WoS · Дирихле", "graded": "MC · сгущённая сетка"}


@njit(cache=True)
def interpolate_flat(values, position, size, dimension):
    lower = np.empty(dimension, dtype=np.int64)
    fractions = np.empty(dimension)
    for axis in range(dimension):
        scaled = min(size - 1.0, max(0.0, position[axis] * (size - 1)))
        lower[axis] = min(size - 2, int(scaled))
        fractions[axis] = scaled - lower[axis]
    value = 0.0
    for corner in range(1 << dimension):
        weight = 1.0
        index = 0
        for axis in range(dimension):
            bit = (corner >> axis) & 1
            weight *= fractions[axis] if bit else 1 - fractions[axis]
            index = index * size + lower[axis] + bit
        value += weight * values[index]
    return value


@njit(cache=True)
def _direction(dimension):
    v = np.random.normal(0.0, 1.0, dimension)
    norm = np.sqrt(np.sum(v * v))
    while norm == 0:
        v = np.random.normal(0.0, 1.0, dimension)
        norm = np.sqrt(np.sum(v * v))
    return v / norm


@njit(cache=True)
def _wos(source, boundary, points, size, walks, seed, epsilon, max_steps):
    np.random.seed(seed)
    dimension = points.shape[1]
    means = np.zeros(len(points))
    errors = np.zeros(len(points))
    steps = 0
    for p in range(len(points)):
        m2 = 0.0
        for sample in range(1, walks + 1):
            position = points[p].copy() / (size - 1)
            value = 0.0
            for step in range(max_steps + 1):
                radius = 1.0
                nearest, end = 0, 0.0
                for axis in range(dimension):
                    if position[axis] < radius:
                        radius, nearest, end = position[axis], axis, 0.0
                    if 1 - position[axis] < radius:
                        radius, nearest, end = 1 - position[axis], axis, 1.0
                if radius <= epsilon:
                    position[nearest] = end
                    value += interpolate_flat(boundary, position, size, dimension)
                    break
                if step == max_steps:
                    raise RuntimeError(
                        "WoS: исчерпан лимит пути; усечённая оценка не возвращается"
                    )
                uniform = 1 - np.random.random()
                radial = uniform ** (1.0 / dimension)
                source_point = position + radius * radial * _direction(dimension)
                # Volume times Green function of -Delta at the ball centre.
                green = (
                    -(radius**2) * np.log(uniform) / 4
                    if dimension == 2
                    else radius**2 * (1 / radial - 1) / 3
                )
                value -= green * interpolate_flat(source, source_point, size, dimension)
                position += radius * _direction(dimension)
                steps += 1
            delta = value - means[p]
            means[p] += delta / sample
            m2 += delta * (value - means[p])
        errors[p] = np.sqrt(max(0.0, m2) / (walks * (walks - 1)))
    return means, errors, steps


@njit(cache=True)
def _graded_paths(
    weights, diagonal, rhs, fixed, boundary, points, axis, walks, seed, max_steps
):
    np.random.seed(seed)
    size = len(axis)
    dimension = points.shape[1]
    strides = np.array([size ** (dimension - k - 1) for k in range(dimension)])
    means = np.zeros(len(points))
    errors = np.zeros(len(points))
    steps = 0
    for p in range(len(points)):
        m2 = 0.0
        for sample in range(1, walks + 1):
            index = 0
            # Random interpolation is unbiased for the multilinear nodal value.
            for k in range(dimension):
                coordinate = points[p, k] / (size - 1)
                lower = min(size - 2, max(0, np.searchsorted(axis, coordinate) - 1))
                fraction = (coordinate - axis[lower]) / (axis[lower + 1] - axis[lower])
                index += (lower + int(np.random.random() < fraction)) * strides[k]
            value = 0.0
            for step in range(max_steps + 1):
                if fixed[index]:
                    value += boundary[index]
                    break
                if step == max_steps:
                    raise RuntimeError("Сгущённая сетка: исчерпан лимит пути")
                value += rhs[index] / diagonal[index]
                draw = np.random.random() * diagonal[index]
                total = 0.0
                direction = -1
                for k in range(2 * dimension):
                    total += weights[index, k]
                    if draw < total:
                        direction = k
                        break
                steps += 1
                if direction == -1:
                    break  # Robin killing: boundary load already included in rhs.
                index += strides[direction // 2] * (1 if direction % 2 == 0 else -1)
            delta = value - means[p]
            means[p] += delta / sample
            m2 += delta * (value - means[p])
        errors[p] = np.sqrt(max(0.0, m2) / (walks * (walks - 1)))
    return means, errors, steps


def graded_boundary_operator(problem):
    from .boundary import BoundaryCondition, BoundaryOperator, BoundaryProblem
    from .boundary3d import BoundaryOperator3D, BoundaryProblem3D

    dimension = problem.f.ndim
    cls = BoundaryProblem if dimension == 2 else BoundaryProblem3D
    op_type = BoundaryOperator if dimension == 2 else BoundaryOperator3D
    if not hasattr(problem, "sides"):
        problem = cls.from_problem(problem)
    if problem.n % 2 == 0:
        axis = graded_axis(problem.n)
    else:
        uniform = np.linspace(0, 1, problem.n + 1)
        axis = np.where(uniform <= 0.5, 2 * uniform**2, 1 - 2 * (1 - uniform) ** 2)

    def resample(values, dim):
        coordinates = np.stack(np.meshgrid(*([axis] * dim), indexing="ij"), axis=-1)
        return RegularGridInterpolator((problem.axis,) * dim, values)(coordinates)

    conditions = {
        name: BoundaryCondition(
            condition.kind,
            resample(condition.value, dimension - 1),
            resample(condition.kappa, dimension - 1),
        )
        for name, condition in problem.sides.items()
    }
    sampled = cls(
        problem.n, resample(problem.f, dimension), conditions, name=problem.name
    )
    operator = op_type(sampled, axis=axis)
    operator.check_compatibility()
    if operator.pure_neumann:
        raise ValueError("MC без поглощения для чистого Неймана не реализован")
    return operator, axis


def solve_extended(
    problem, points, method, walks=2048, seed=42, max_steps=None, epsilon=1e-4
):
    from .solvers import NUMBA_LOCK, Result

    points = np.asarray(points)
    dimension = problem.f.ndim
    if (
        points.ndim != 2
        or points.shape[1] != dimension
        or len(points) == 0
        or not np.isfinite(points).all()
        or np.any(points != points.astype(int))
        or np.any(points < 0)
        or np.any(points > problem.n)
    ):
        raise ValueError("Нужны индексы узлов исходной сетки")
    points = np.ascontiguousarray(points, dtype=np.int64)
    limit = 1000 * problem.n**2 if max_steps is None else max_steps
    for value, minimum, maximum in (
        (walks, 2, 2**31 - 1),
        (seed, 0, 2**32 - 1),
        (limit, 1, 2**63 - 1),
    ):
        if (
            isinstance(value, (bool, np.bool_))
            or not np.isfinite(value)
            or int(value) != value
            or not minimum <= value <= maximum
        ):
            raise ValueError("Некорректный бюджет, seed или лимит")
    if method not in EXTENDED_METHODS:
        raise ValueError("Метод: wos или graded")
    if not np.isfinite(epsilon) or not 0 < epsilon < 0.5:
        raise ValueError("0 < epsilon < .5")
    if (
        method == "wos"
        and hasattr(problem, "sides")
        and any(c.kind != "dirichlet" for c in problem.sides.values())
    ):
        raise ValueError("Непрерывный WoS требует Дирихле на всех сторонах/гранях")
    with NUMBA_LOCK:
        if dimension not in _warmed_dimensions:
            # Compile small signatures before timing. No solution/reference preparation here.
            dummy = np.zeros((1, dimension), dtype=np.int64)
            _wos(
                np.zeros(3**dimension),
                np.zeros(3**dimension),
                dummy,
                3,
                2,
                0,
                epsilon,
                1,
            )
            _graded_paths(
                np.zeros((3**dimension, 2 * dimension)),
                np.ones(3**dimension),
                np.zeros(3**dimension),
                np.ones(3**dimension, dtype=np.bool_),
                np.zeros(3**dimension),
                dummy,
                np.linspace(0, 1, 3),
                2,
                0,
                1,
            )
            _warmed_dimensions.add(dimension)
        started = perf_counter()
        values = np.full_like(problem.f, np.nan)
        fixed = np.zeros_like(problem.f, dtype=bool)
        if hasattr(problem, "sides"):
            from .boundary import BoundaryOperator
            from .boundary3d import BoundaryOperator3D

            original = (
                BoundaryOperator(problem)
                if dimension == 2
                else BoundaryOperator3D(problem)
            )
            original.check_compatibility()
            fixed = original.fixed
            values[fixed] = original.boundary[fixed]
        else:
            for coordinate in range(dimension):
                for end in (0, -1):
                    index = [slice(None)] * dimension
                    index[coordinate] = end
                    values[tuple(index)] = problem.boundary[tuple(index)]
                    fixed[tuple(index)] = True
        queries = points[~fixed[tuple(points.T)]]
        steps = 0
        errors = means = np.empty(0)
        if len(queries) and method == "wos":
            means, errors, steps = _wos(
                problem.f.ravel(),
                np.nan_to_num(values).ravel(),
                queries,
                problem.n + 1,
                int(walks),
                int(seed),
                epsilon,
                int(limit),
            )
        elif len(queries):
            operator, axis = graded_boundary_operator(problem)
            means, errors, steps = _graded_paths(
                operator.weights.reshape(-1, 2 * dimension),
                operator.diagonal.ravel(),
                operator.b.ravel(),
                operator.fixed.ravel(),
                operator.boundary.ravel(),
                queries,
                axis,
                int(walks),
                int(seed),
                int(limit),
            )
        stderr = np.full_like(values, np.nan)
        stderr[fixed] = 0
        counts = np.zeros_like(values, dtype=np.int64)
        indices = tuple(queries.T)
        values[indices], stderr[indices], counts[indices] = means, errors, walks
        return Result(
            method,
            values,
            stderr,
            counts,
            elapsed=perf_counter() - started,
            steps=steps,
            walks=int(walks) * len(queries),
            extras=dict(
                data_model="multilinear_input_data",
                epsilon=epsilon if method == "wos" else None,
                query_grid="original_uniform",
                scheme="continuous_wos" if method == "wos" else "graded_finite_volume",
            ),
        )


def render_method_scope(dimension, mixed=False):
    import streamlit as st

    with st.expander(
        "Методы: схемы и область применимости", key="stationary_extensions_expander_326"
    ):
        st.write(
            "ЛОМ и обычный MC решают одну равномерную разностную систему. Сгущённая сетка и WoS имеют собственные погрешности аппроксимации; сравниваются значения в одной физической точке."
        )
        st.latex(r"x_i=\tfrac12(2i/n)^2\ (i\le n/2),\quad x_{n-i}=1-x_i")
        st.write(
            "Сгущённая сетка: консервативные контрольные объёмы; отражение на Неймане и частичное поглощение на Робине. Между узлами используется случайная интерполяция начальной точки."
        )
        if mixed:
            st.write(
                "Непрерывный WoS для Неймана/Робина не реализован. Чистый Нейман: только ЛОМ с проверкой баланса и фиксацией среднего; поглощающий MC неприменим."
            )
        else:
            st.latex(
                r"u(x)=\mathbb E\left[g(X_\tau)-\int_0^\tau f(X_s)\,ds\right]\quad(\text{генератор }\Delta)"
            )
            st.write(
                f"WoS: выходы из вписанных {'дисков' if dimension == 2 else 'шаров'}, ε=10⁻⁴. Источник и граничные значения реконструируются из входных массивов линейно по каждой координате; аналитический ответ не используется. Допуск ε и реконструкция данных дают систематическую ошибку, которую увеличение числа путей не убирает."
            )
        if dimension == 3 or mixed:
            st.write(
                "Контрольное приближение, хвосты и слои из исходного 2D-режима ещё не перенесены на эту задачу; под их названиями другие методы не запускаются."
            )
