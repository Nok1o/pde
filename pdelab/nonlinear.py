"""Lecture notation: alpha is gamma in k(u)=1+gamma*u**2;
relaxation is xi, dt is tau, source is f, and capacity c=1.
These API names are retained for saved configurations.

C=1, k(u)=1+alpha*u²: physical time, Picard, inner solves and Newton.

All fields use [y,x]. Backward Euler freezes conductivity on a whole field.
An incomplete nonlinear layer terminates the history with an explicit status.
"""

from dataclasses import dataclass, field
from sys import getsizeof
from time import perf_counter

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

from .frozen_solvers import (
    relative_norm,
    solve_frozen_lod,
    solve_frozen_mc,
    warmup_frozen,
    with_boundary,
)
from .heat import HeatProblem, time_grid
from .operator import FrozenOperator


@dataclass
class NonlinearProblem(HeatProblem):
    alpha: float = 1.0

    def __post_init__(self):
        super().__post_init__()
        if not np.isfinite(self.alpha) or self.alpha < 0:
            raise ValueError("α должно быть конечным и неотрицательным")
        if self.n > 64 or self.a2 != 1:
            raise ValueError("Нелинейный MVP: n≤64, C=1, k(u)=1+αu²")

    def conductivity(self, values):
        with np.errstate(over="ignore", invalid="ignore"):
            k = 1 + self.alpha * np.asarray(values) ** 2
        if not np.isfinite(k).all():
            raise ValueError("Нелинейная проводимость вышла за диапазон конечных чисел")
        return k


def make_nonlinear_problem(n=8, alpha=0.5, stationary=False):
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or not 4 <= n <= 64:
        raise ValueError("n должно быть целым от 4 до 64")
    if not np.isfinite(alpha) or alpha < 0:
        raise ValueError("α должно быть конечным и неотрицательным")

    def exact(x, y, t):
        amplitude = 1.0 if stationary else np.exp(-t)
        return 1 + amplitude * np.sin(np.pi * x) * np.sin(np.pi * y)

    def source(x, y, t):
        amplitude = 1.0 if stationary else np.exp(-t)
        sine = amplitude * np.sin(np.pi * x) * np.sin(np.pi * y)
        u = 1 + sine
        gradient_squared = (amplitude * np.pi) ** 2 * (
            (np.cos(np.pi * x) * np.sin(np.pi * y)) ** 2
            + (np.sin(np.pi * x) * np.cos(np.pi * y)) ** 2
        )
        divergence = (
            -2 * np.pi**2 * (1 + alpha * u * u) * sine
            + 2 * alpha * u * gradient_squared
        )
        return -divergence if stationary else -sine - divergence

    axis = np.linspace(0, 1, n + 1)
    x, y = np.meshgrid(axis, axis)
    return NonlinearProblem(
        "Нелинейная теплопроводность",
        n,
        1.0,
        exact(x, y, 0),
        source,
        lambda x, y, t: 1.0,
        exact,
        alpha,
    )


@dataclass
class NonlinearSettings:
    tolerance: float = 1e-7
    change_tolerance: float = 1e-7
    linear_tolerance: float = 1e-10
    relaxation: float = 1.0
    max_outer: int = 100
    max_inner: int = 20000
    walks: int = 128
    max_walks: int = 2048
    schedule: str = "geometric"
    max_steps: int = 100000
    seed: int = 42
    max_total_walks: int = 20_000_000

    def validate(self):
        for name in ("tolerance", "change_tolerance", "linear_tolerance"):
            value = getattr(self, name)
            if not np.isfinite(value) or not 0 < value < 1:
                raise ValueError(f"{name}: требуется допуск между 0 и 1")
        if not np.isfinite(self.relaxation) or not 0 < self.relaxation <= 1:
            raise ValueError("Релаксация должна лежать в (0,1]")
        for name in (
            "max_outer",
            "max_inner",
            "walks",
            "max_walks",
            "max_steps",
            "max_total_walks",
            "seed",
        ):
            value = getattr(self, name)
            minimum = (
                0 if name == "seed" else 2 if name in ("walks", "max_walks") else 1
            )
            if (
                isinstance(value, (bool, np.bool_))
                or not np.isfinite(value)
                or int(value) != value
                or not minimum <= value <= 2**32 - 1
            ):
                raise ValueError(f"{name}: некорректное целое значение")
        if self.max_walks < self.walks or self.schedule not in (
            "constant",
            "linear",
            "geometric",
        ):
            raise ValueError("Нужны max_walks≥walks и схема constant/linear/geometric")

    def budget(self, outer):
        if self.schedule == "constant":
            return int(self.walks)
        if self.schedule == "linear":
            return int(min(self.max_walks, self.walks * outer))
        return int(min(self.max_walks, self.walks * 2 ** min(outer - 1, 32)))


@dataclass
class LayerResult:
    values: np.ndarray
    converged: bool
    reason: str
    history: list = field(default_factory=list)
    inner_iterations: int = 0
    walks: int = 0
    steps: int = 0


def freeze(problem, values, b, boundary, sigma):
    return FrozenOperator.from_nodes(problem.conductivity(values), b, boundary, sigma)


def nonlinear_residual(problem, values, b, boundary, sigma):
    return freeze(problem, values, b, boundary, sigma).apply(values) - b[1:-1, 1:-1]


def _scale(problem, initial, b, boundary, sigma):
    applied = freeze(problem, initial, b, boundary, sigma).apply(initial)
    return float(max(np.max(np.abs(b[1:-1, 1:-1])), np.max(np.abs(applied))))


def _change(current, previous):
    scale = float(max(np.max(np.abs(current)), np.max(np.abs(previous))))
    return relative_norm(current - previous, scale)


def _seed(seed, layer, outer, validation=0):
    return int(
        np.random.SeedSequence([int(seed), layer, outer, validation]).generate_state(1)[
            0
        ]
    )


def picard_layer(
    problem,
    initial,
    b,
    boundary,
    sigma,
    settings,
    method="picard",
    layer=0,
    walk_allowance=None,
):
    settings.validate()
    if method not in ("picard", "mc"):
        raise ValueError("Внешняя итерация: picard или mc")
    u = with_boundary(initial, boundary)
    scale = _scale(problem, u, b, boundary, sigma)
    result = LayerResult(u, False, "max_outer")
    allowance = settings.max_total_walks if walk_allowance is None else walk_allowance
    initial_residual = relative_norm(
        nonlinear_residual(problem, u, b, boundary, sigma), scale
    )
    if initial_residual == 0:
        result.converged, result.reason = True, "exact_initial"
        return result
    for outer in range(1, int(settings.max_outer) + 1):
        operator = freeze(problem, u, b, boundary, sigma)
        budget = settings.budget(outer) if method == "mc" else 0
        cost = (problem.n - 1) ** 2 * budget
        if result.walks + cost > allowance:
            result.reason = "walk_budget"
            return result
        try:
            if method == "picard":
                linear = solve_frozen_lod(
                    operator, u, settings.linear_tolerance, settings.max_inner
                )
            else:
                linear = solve_frozen_mc(
                    operator,
                    budget,
                    _seed(settings.seed, layer, outer),
                    settings.max_steps,
                    control=u,
                )
        except (ValueError, RuntimeError) as exc:
            result.walks += getattr(exc, "walks", 0)
            result.steps += getattr(exc, "steps", 0)
            result.reason = f"inner_failure: {exc}"
            return result
        result.inner_iterations += linear.iterations
        result.walks += linear.walks
        result.steps += linear.steps
        candidate = with_boundary(
            (1 - settings.relaxation) * u + settings.relaxation * linear.values,
            boundary,
        )
        change = _change(candidate, u)
        residual = relative_norm(
            nonlinear_residual(problem, candidate, b, boundary, sigma), scale
        )
        row = dict(
            outer=outer,
            budget=budget,
            inner_iterations=linear.iterations,
            linear_residual=linear.relative_residual,
            relative_change=change,
            nonlinear_residual=residual,
            validation_residual=None,
            validation_change=None,
            accepted=False,
        )
        result.history.append(row)
        result.values = candidate
        if not linear.converged:
            result.reason = "linear_limit"
            return result
        passes = residual <= settings.tolerance and change <= settings.change_tolerance
        if passes and method == "mc":
            # An independent series checks the candidate with freshly frozen k.
            # Full PDE residuals are deterministic evaluations, not noisy SE tests.
            if result.walks + cost > allowance:
                result.reason = "validation_budget"
                return result
            try:
                control_operator = freeze(problem, candidate, b, boundary, sigma)
                check = solve_frozen_mc(
                    control_operator,
                    budget,
                    _seed(settings.seed, layer, outer, 1),
                    settings.max_steps,
                    control=candidate,
                )
            except (ValueError, RuntimeError) as exc:
                result.walks += getattr(exc, "walks", 0)
                result.steps += getattr(exc, "steps", 0)
                result.reason = f"validation_failure: {exc}"
                return result
            result.walks += check.walks
            result.steps += check.steps
            row["validation_change"] = _change(check.values, candidate)
            row["validation_residual"] = relative_norm(
                nonlinear_residual(problem, check.values, b, boundary, sigma), scale
            )
            passes = (
                row["validation_change"] <= settings.change_tolerance
                and row["validation_residual"] <= settings.tolerance
            )
        if passes:
            row["accepted"] = True
            result.converged, result.reason = (
                True,
                (
                    "residual_and_change"
                    if method == "picard"
                    else "independent_validation"
                ),
            )
            return result
        u = candidate
    return result


def newton_system(values, alpha, b, sigma):
    """Independent edge assembly of residual AND analytic nonlinear Jacobian.

    Does not call FrozenOperator or Picard. Dirichlet derivatives are removed.
    """
    n = len(values) - 1
    m = n - 1
    center = values[1:-1, 1:-1]
    residual = sigma * center - b[1:-1, 1:-1]
    diagonal = np.full_like(center, sigma)
    indices = np.arange(m * m).reshape(m, m)
    rows, columns, coefficients = [], [], []
    neighbors = (
        values[2:, 1:-1],
        values[:-2, 1:-1],
        values[1:-1, 2:],
        values[1:-1, :-2],
    )
    for direction, neighbor in enumerate(neighbors):
        face_k = 1 + alpha * (center**2 + neighbor**2) / 2
        difference = center - neighbor
        residual = residual + n * n * face_k * difference
        diagonal += n * n * (face_k + alpha * center * difference)
        derivative = n * n * (-face_k + alpha * neighbor * difference)
        if direction == 0:
            origin, target = (
                (slice(None, -1), slice(None)),
                (
                    slice(1, None),
                    slice(None),
                ),
            )
        elif direction == 1:
            origin, target = (
                (slice(1, None), slice(None)),
                (
                    slice(None, -1),
                    slice(None),
                ),
            )
        elif direction == 2:
            origin, target = (
                (slice(None), slice(None, -1)),
                (
                    slice(None),
                    slice(1, None),
                ),
            )
        else:
            origin, target = (
                (slice(None), slice(1, None)),
                (
                    slice(None),
                    slice(None, -1),
                ),
            )
        rows.append(indices[origin].ravel())
        columns.append(indices[target].ravel())
        coefficients.append(derivative[origin].ravel())
    rows.append(indices.ravel())
    columns.append(indices.ravel())
    coefficients.append(diagonal.ravel())
    jacobian = coo_matrix(
        (np.concatenate(coefficients), (np.concatenate(rows), np.concatenate(columns))),
        shape=(m * m, m * m),
    ).tocsr()
    return residual, jacobian


def newton_layer(problem, initial, b, boundary, sigma, tolerance=1e-11, max_iter=50):
    if problem.n > 32:
        raise ValueError("Независимый Newton-эталон ограничен n≤32")
    u = with_boundary(initial, boundary)
    residual, jacobian = newton_system(u, problem.alpha, b, sigma)
    scale = float(
        max(np.max(np.abs(b[1:-1, 1:-1])), np.max(np.abs(residual + b[1:-1, 1:-1])))
    )
    history = []
    for iteration in range(int(max_iter) + 1):
        relative = relative_norm(residual, scale)
        if relative <= tolerance:
            return LayerResult(u, True, "newton_residual", history, iteration)
        if iteration == max_iter:
            break
        correction = spsolve(jacobian, -residual.ravel()).reshape(
            problem.n - 1, problem.n - 1
        )
        factor = 1.0
        accepted = False
        for _ in range(20):
            candidate = u.copy()
            candidate[1:-1, 1:-1] += factor * correction
            next_residual, next_jacobian = newton_system(
                candidate, problem.alpha, b, sigma
            )
            if np.isfinite(next_residual).all() and np.max(
                np.abs(next_residual)
            ) < np.max(np.abs(residual)):
                accepted = True
                break
            factor *= 0.5
        if not accepted:
            return LayerResult(u, False, "newton_line_search", history, iteration + 1)
        history.append(
            dict(
                outer=iteration + 1,
                nonlinear_residual=relative_norm(next_residual, scale),
                relative_change=_change(candidate, u),
                relaxation=factor,
            )
        )
        u, residual, jacobian = candidate, next_residual, next_jacobian
    return LayerResult(u, False, "newton_limit", history, max_iter)


@dataclass
class NonlinearResult:
    problem: NonlinearProblem
    method: str
    settings: NonlinearSettings
    dt: float
    T: float
    times: np.ndarray
    fields: np.ndarray
    layers: list
    seconds: float
    completed: bool
    reason: str
    workspace_bytes: int

    @property
    def walks(self):
        return sum(layer.walks for layer in self.layers)

    @property
    def steps(self):
        return sum(layer.steps for layer in self.layers)


def solve_nonlinear(problem, dt=0.01, T=0.02, method="picard", settings=None):
    settings = NonlinearSettings() if settings is None else settings
    settings.validate()
    if method not in ("picard", "mc", "newton"):
        raise ValueError("Метод: picard, mc или newton")
    times, steps = time_grid(dt, T)
    if len(times) * (problem.n + 1) ** 2 > 16_000_000:
        raise ValueError("История ограничена 16 млн значений")
    if method == "newton" and problem.n > 32:
        raise ValueError("Независимый Newton-эталон ограничен n≤32")
    if method != "newton" and len(steps):
        warmup_frozen()
    started = perf_counter()
    fields = [problem.initial.copy()]
    layers = []
    total_walks = 0
    completed, reason = True, "completed"
    for index, step in enumerate(steps, 1):
        boundary = problem.evaluate(problem.boundary, times[index])
        b = fields[-1] / step + problem.evaluate(problem.source, times[index])
        initial = with_boundary(fields[-1], boundary)
        if method == "newton":
            layer = newton_layer(
                problem,
                initial,
                b,
                boundary,
                1 / step,
                tolerance=settings.tolerance,
                max_iter=settings.max_outer,
            )
        else:
            layer = picard_layer(
                problem,
                initial,
                b,
                boundary,
                1 / step,
                settings,
                method,
                layer=index,
                walk_allowance=settings.max_total_walks - total_walks,
            )
        layers.append(layer)
        fields.append(layer.values)
        total_walks += layer.walks
        if not layer.converged:
            completed, reason = False, f"layer {index}: {layer.reason}"
            break
    fields = np.array(fields)
    for index, layer in enumerate(layers, 1):
        layer.values = fields[index]
    log_bytes = sum(
        getsizeof(layer.history)
        + sum(
            getsizeof(row) + sum(getsizeof(value) for value in row.values())
            for row in layer.history
        )
        for layer in layers
    )
    # Includes the temporary history copy and Python iteration records.
    # This is a reserve estimate, not measured RSS; JIT/plots/sparse LU excluded.
    workspace = 2 * fields.nbytes + 40 * problem.initial.nbytes + log_bytes
    return NonlinearResult(
        problem,
        method,
        settings,
        dt,
        T,
        times[: len(fields)],
        fields,
        layers,
        perf_counter() - started,
        completed,
        reason,
        workspace,
    )


def solve_stationary_nonlinear(n=8, alpha=0.5, method="picard", settings=None):
    problem = make_nonlinear_problem(n, alpha, stationary=True)
    settings = NonlinearSettings() if settings is None else settings
    settings.validate()
    boundary = problem.evaluate(problem.boundary, 0)
    initial = with_boundary(np.zeros_like(boundary), boundary)
    b = problem.evaluate(problem.source, 0)
    if method == "newton":
        return newton_layer(
            problem, initial, b, boundary, 0, settings.tolerance, settings.max_outer
        )
    warmup_frozen()
    return picard_layer(problem, initial, b, boundary, 0, settings, method)
