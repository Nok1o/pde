"""Physical time integration of u_t = a² Δu + F with Dirichlet data.

Backward Euler is solved to a relative RHS residual at each time layer.
Directional sweeps are an inner iteration, never a physical time step.
"""

from dataclasses import dataclass
from time import perf_counter
from typing import Callable

import numpy as np
from scipy.sparse import diags, eye, kronsum
from scipy.sparse.linalg import factorized

from .problem import laplacian
from .solvers import NUMBA_LOCK, _line_coefficients, _sweep_into

HEAT_PRESETS = {
    "decay": "Затухание синусоидального поля",
    "manufactured": "Временные источник и граница",
}


def _field(value, shape):
    result = np.broadcast_to(np.asarray(value, dtype=float), shape).copy()
    if not np.isfinite(result).all():
        raise ValueError("Поля должны содержать конечные значения")
    return result


@dataclass
class HeatProblem:
    name: str
    n: int
    a2: float
    initial: np.ndarray
    source: Callable
    boundary: Callable
    exact: Callable | None = None

    def __post_init__(self):
        if (
            isinstance(self.n, bool)
            or not isinstance(self.n, (int, np.integer))
            or not 4 <= self.n <= 512
        ):
            raise ValueError("n должно быть целым от 4 до 512")
        self.n = int(self.n)
        if not np.isfinite(self.a2) or self.a2 <= 0:
            raise ValueError("a² должно быть конечным и положительным")
        self.initial = _field(self.initial, (self.n + 1, self.n + 1))
        g = self.evaluate(self.boundary, 0.0)
        if not np.allclose(
            self.initial[[0, -1], :], g[[0, -1], :], atol=1e-12, rtol=1e-12
        ) or not np.allclose(
            self.initial[:, [0, -1]], g[:, [0, -1]], atol=1e-12, rtol=1e-12
        ):
            raise ValueError("Начальное поле должно совпадать с границей при t=0")

    @property
    def axis(self):
        return np.linspace(0, 1, self.n + 1)

    def evaluate(self, fn, t):
        x, y = np.meshgrid(self.axis, self.axis)
        return _field(fn(x, y, t), x.shape)

    @classmethod
    def from_functions(
        cls,
        n,
        initial,
        source,
        boundary,
        exact=None,
        a2=1.0,
        name="Своя задача теплопроводности",
    ):
        if (
            isinstance(n, bool)
            or not isinstance(n, (int, np.integer))
            or not 4 <= n <= 512
        ):
            raise ValueError("n должно быть целым от 4 до 512")
        x, y = np.meshgrid(np.linspace(0, 1, n + 1), np.linspace(0, 1, n + 1))
        return cls(name, n, a2, _field(initial(x, y), x.shape), source, boundary, exact)


def make_heat_problem(kind="decay", n=24, a2=1.0):
    if kind == "decay":

        def exact(x, y, t):
            return (
                np.exp(-2 * np.pi**2 * a2 * t) * np.sin(np.pi * x) * np.sin(np.pi * y)
            )

        def source(x, y, t):
            return 0.0

        boundary = source
    elif kind == "manufactured":

        def exact(x, y, t):
            return 1 + x + y + t * (1 + x) + np.exp(t) * x * (1 - x) * y * (1 - y)

        def source(x, y, t):
            return (
                1
                + x
                + np.exp(t)
                * (x * (1 - x) * y * (1 - y) + 2 * a2 * (x * (1 - x) + y * (1 - y)))
            )

        boundary = exact
    else:
        raise ValueError(f"Неизвестная задача теплопроводности: {kind}")
    return HeatProblem.from_functions(
        n,
        lambda x, y: exact(x, y, 0.0),
        source,
        boundary,
        exact,
        a2,
        HEAT_PRESETS[kind],
    )


def time_grid(dt, T):
    if not np.isfinite(dt) or dt <= 0 or not np.isfinite(T) or T < 0:
        raise ValueError("Требуются конечные dt>0 и T≥0")
    if T == 0:
        return np.array([0.0]), np.empty(0)
    ratio = T / dt
    if not np.isfinite(ratio) or ratio > 10000:
        raise ValueError("Не более 10000 временных шагов за один расчёт")
    nearest = round(ratio)
    count = (
        nearest
        if nearest >= 1 and abs(ratio - nearest) <= 8 * np.finfo(float).eps * ratio
        else int(np.floor(ratio))
    )
    remainder = T - count * dt
    steps = np.full(count, dt)
    times = np.arange(count + 1, dtype=float) * dt
    if remainder > 8 * np.finfo(float).eps * T:
        steps = np.append(steps, remainder)
        times = np.append(times, T)
    times[-1] = T
    return times, steps


@dataclass
class HeatResult:
    problem: HeatProblem
    dt: float
    T: float
    method: str
    tolerance: float
    max_iter: int
    times: np.ndarray
    fields: np.ndarray
    iterations: np.ndarray
    relative_residuals: np.ndarray
    seconds: float
    factorizations: int = 0

    def rows(self):
        rows = []
        for index, (t, u) in enumerate(zip(self.times, self.fields)):
            exact = (
                self.problem.evaluate(self.problem.exact, t)
                if self.problem.exact
                else None
            )
            rows.append(
                dict(
                    time_index=index,
                    t=float(t),
                    dt=0.0 if index == 0 else float(t - self.times[index - 1]),
                    iterations=int(self.iterations[index]),
                    relative_residual=float(self.relative_residuals[index]),
                    center=float(u[self.problem.n // 2, self.problem.n // 2]),
                    rmse_exact=(
                        None
                        if exact is None
                        else float(np.sqrt(np.mean((u - exact) ** 2)))
                    ),
                )
            )
        return rows


def _dirichlet_rhs(rhs, g, ratio):
    rhs = rhs.copy()
    rhs[0, :] += ratio * g[0, 1:-1]
    rhs[-1, :] += ratio * g[-1, 1:-1]
    rhs[:, 0] += ratio * g[1:-1, 0]
    rhs[:, -1] += ratio * g[1:-1, -1]
    return rhs


def _relative(residual, scale):
    norm = float(np.max(np.abs(residual)))
    return norm / scale if scale else (0.0 if norm == 0 else float("inf"))


def _factors(n, c):
    """Optimal fixed ADI relaxation for the spectrum of .5 I - c L_x.

    This relaxation is dimensionless and independent of physical time dt.
    P=(I+q(.5I-cLx))(I+q(.5I-cLy)); correction=2q P^-1 r.
    """
    low = 0.5 + 4 * c * n * n * np.sin(np.pi / (2 * n)) ** 2
    high = 0.5 + 4 * c * n * n * np.cos(np.pi / (2 * n)) ** 2
    q = 1 / np.sqrt(low) / np.sqrt(high)
    s = 1 + q / 2
    ratio = q * c * n * n / s
    alpha, denominators = _line_coefficients(n - 1, ratio)
    return ratio, alpha, denominators, 2 * q / (s * s)


def solve_heat(problem, dt=0.01, T=0.2, method="lod", tol=1e-10, max_iter=20000):
    """Return every physical time layer; fail explicitly if an inner solve fails.

    Direct factorization and sweep coefficients are reused only for identical
    step sizes. A shortened final step gets its own matrix and coefficients.
    """
    times, steps = time_grid(dt, T)
    if method not in ("lod", "direct"):
        raise ValueError("Метод должен быть lod или direct")
    if not np.isfinite(tol) or not 0 < tol < 1:
        raise ValueError("Требуется 0<tol<1")
    if (
        isinstance(max_iter, bool)
        or not np.isfinite(max_iter)
        or int(max_iter) != max_iter
        or max_iter < 1
    ):
        raise ValueError("max_iter должно быть положительным целым")
    max_iter = int(max_iter)
    n = problem.n
    if len(times) * (n + 1) ** 2 > 16_000_000:
        raise ValueError(
            "История ограничена 16 млн значений; уменьшите n или число шагов"
        )
    if len(steps) and not np.isfinite(4 * max(steps) * problem.a2 * n * n):
        raise ValueError("Коэффициенты неявного слоя должны быть конечными")
    # Compile sweeps before timing, under the same lock as stationary kernels.
    if method == "lod" and len(steps):
        with NUMBA_LOCK:
            ratio, alpha, denominators, _ = _factors(n, steps[0] * problem.a2)
            work = np.zeros((n - 1, n - 1))
            _sweep_into(work, work.copy(), ratio, 0, alpha, denominators, np.empty(n))
    started = perf_counter()
    fields = np.empty((len(times), n + 1, n + 1))
    fields[0] = problem.initial
    iterations = np.zeros(len(times), dtype=np.int64)
    residuals = np.zeros(len(times))
    cache = {}
    xwork = np.empty((n - 1, n - 1))
    correction = np.empty_like(xwork)
    beta = np.empty(n)
    for index, step in enumerate(steps, 1):
        c = step * problem.a2
        ratio = c * n * n
        g = problem.evaluate(problem.boundary, times[index])
        rhs = (
            fields[index - 1, 1:-1, 1:-1]
            + step * problem.evaluate(problem.source, times[index])[1:-1, 1:-1]
        )
        adjusted = _dirichlet_rhs(rhs, g, ratio)
        if not np.isfinite(adjusted).all():
            raise ValueError("Правая часть слоя должна быть конечной")
        scale = float(np.max(np.abs(adjusted)))
        u = g.copy()
        if scale == 0:
            # The reduced system is nonsingular: A*u=0 has the exact solution 0.
            u[1:-1, 1:-1] = 0.0
            fields[index] = u
            continue
        if method == "direct":
            if step not in cache:
                m = n - 1
                line = diags(
                    [-np.ones(m - 1), 2 * np.ones(m), -np.ones(m - 1)],
                    [-1, 0, 1],
                    format="csc",
                )
                matrix = eye(m * m, format="csc") + ratio * kronsum(
                    line, line, format="csc"
                )
                cache[step] = factorized(matrix)
            u[1:-1, 1:-1] = cache[step](adjusted.ravel()).reshape(n - 1, n - 1)
            residual = rhs - u[1:-1, 1:-1] + c * laplacian(u, 1 / n)
            residuals[index] = _relative(residual, scale)
        else:
            if step not in cache:
                cache[step] = _factors(n, c)
            sweep_ratio, alpha, denominators, multiplier = cache[step]
            u[1:-1, 1:-1] = fields[index - 1, 1:-1, 1:-1]
            for iteration in range(max_iter + 1):
                residual = rhs - u[1:-1, 1:-1] + c * laplacian(u, 1 / n)
                relative = _relative(residual, scale)
                if relative <= tol:
                    break
                if iteration == max_iter:
                    raise RuntimeError(
                        f"Слой {index}, t={times[index]:g}: не достигнут допуск {tol:g}"
                    )
                _sweep_into(
                    multiplier * residual,
                    xwork,
                    sweep_ratio,
                    0,
                    alpha,
                    denominators,
                    beta,
                )
                _sweep_into(
                    xwork, correction, sweep_ratio, 1, alpha, denominators, beta
                )
                u[1:-1, 1:-1] += correction
            iterations[index] = iteration
            residuals[index] = relative
        fields[index] = u
    return HeatResult(
        problem,
        dt,
        T,
        method,
        tol,
        max_iter,
        times,
        fields,
        iterations,
        residuals,
        perf_counter() - started,
        len(cache) if method == "direct" else 0,
    )
