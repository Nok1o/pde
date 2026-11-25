"""Backward Euler in physical time, with Picard/XYZ or a small Newton reference.

Only the previous and current full fields are kept. Requested values are returned
only after the whole time interval has completed; failed layers remain diagnostic.
"""

import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field
from sys import getsizeof
from time import perf_counter

import numpy as np
from scipy.sparse.linalg import MatrixRankWarning, spsolve

from .heat import time_grid
from .heat3d_contract import HeatRunResult3D
from .heat3d_linear import solve_lines3d, warmup_lines3d
from .heat3d_operator import (
    IncompatibleHeatBoundaryError,
    UnsupportedHeatBoundaryError,
    dirichlet_data3d,
    freeze_heat3d,
)
from .heat3d_reference import (
    newton_residual3d,
    newton_scale3d,
    newton_system3d,
)
from .problem3d import DIRECT_MAX_N, validate_points3d


@dataclass(frozen=True)
class HeatSettings3D:
    tolerance: float = 1e-7
    change_tolerance: float = 1e-7
    linear_tolerance: float = 1e-10
    relaxation: float = 1.0
    max_outer: int = 100
    max_inner: int = 20000
    max_seconds: float = 60.0
    max_workspace_bytes: int = 256 * 1024**2

    def validate(self):
        for name in ("tolerance", "change_tolerance", "linear_tolerance"):
            value = getattr(self, name)
            if not np.isfinite(value) or not 0 < value < 1:
                raise ValueError(f"{name}: требуется относительный допуск от 0 до 1")
        if not np.isfinite(self.relaxation) or not 0 < self.relaxation <= 1:
            raise ValueError("Релаксация должна лежать в (0, 1]")
        if not np.isfinite(self.max_seconds) or self.max_seconds <= 0:
            raise ValueError("Лимит времени должен быть конечным и положительным")
        for name, maximum in (
            ("max_outer", 10000),
            ("max_inner", 1_000_000),
            ("max_workspace_bytes", 2**63 - 1),
        ):
            value = getattr(self, name)
            if (
                isinstance(value, (bool, np.bool_))
                or not np.isscalar(value)
                or not np.isfinite(value)
                or int(value) != value
                or not 1 <= value <= maximum
            ):
                raise ValueError(f"{name}: требуется целое от 1 до {maximum}")


def _norm(values):
    values = np.asarray(values)
    if not np.isfinite(values).all():
        raise FloatingPointError("Неконечное поле или невязка")
    return float(np.max(np.abs(values)))


def _relative(values, scale):
    norm = _norm(values)
    return norm / scale if scale else (0.0 if norm == 0 else float("inf"))


def _change(current, previous):
    return _relative(current - previous, max(_norm(current), _norm(previous)))


def _history_bytes(history):
    return getsizeof(history) + sum(
        getsizeof(row) + sum(getsizeof(value) for value in row.values())
        for row in history
    )


def _workspace_estimate(n, method, points, initial_only=False):
    # Explicit numerical-buffer estimate; this is not a measured process RSS.
    buffers = (1 if initial_only else 40) * (n + 1) ** 3 * 8
    if method == "newton" and not initial_only:
        # Conservative dense-fill storage allowance for the small sparse LU.
        buffers += 16 * (n - 1) ** 6
    return buffers + points.nbytes + len(points) * 8


@dataclass
class _Layer:
    values: np.ndarray
    status: str = "outer_limit"
    reason: str = "Лимит внешних итераций"
    history: list = field(default_factory=list)
    outer_iterations: int = 0
    inner_iterations: int = 0
    preparation_seconds: float = 0.0


@contextmanager
def _preparing(layer):
    started = perf_counter()
    try:
        yield
    finally:
        layer.preparation_seconds += perf_counter() - started


def _record(layer, *, outer, residual, scale, change=None, linear=None, accepted=False):
    row = dict(
        outer=outer,
        linear_relative_residual=linear,
        nonlinear_relative_residual=residual,
        residual_scale=scale,
        relative_change=change,
        accepted=accepted,
        walks=0,
        transitions=0,
        reason="completed" if accepted else layer.reason,
    )
    layer.history.append(row)
    return row


def _initial(problem, previous, t):
    fixed, boundary = dirichlet_data3d(problem, t)
    values = previous.copy()
    values[fixed] = boundary[fixed]
    return values, fixed, boundary


def _picard_layer(
    problem, previous, t, dt, settings, deadline, factory, backend, layer
):
    with _preparing(layer):
        values, fixed, boundary = _initial(problem, previous, t)
    layer.values = values

    def freeze(candidate):
        with _preparing(layer):
            return factory(problem, previous, candidate, t, dt)

    operator = freeze(values)
    free = ~fixed
    applied = operator.apply(values)
    scale = max(_norm(operator.rhs[free]), _norm(applied[free]))
    residual = _relative((applied - operator.rhs)[free], scale)
    if residual == 0:
        layer.status, layer.reason = "completed", "Нулевая начальная невязка"
        _record(layer, outer=0, residual=0.0, scale=scale, change=0.0, accepted=True)
        return layer
    for outer in range(1, int(settings.max_outer) + 1):
        if perf_counter() >= deadline:
            layer.status, layer.reason = "time_limit", "Лимит времени между итерациями"
            _record(layer, outer=outer - 1, residual=residual, scale=scale)
            return layer
        layer.outer_iterations += 1
        linear = backend(
            operator,
            values,
            tolerance=settings.linear_tolerance,
            max_iterations=int(settings.max_inner),
        )
        layer.inner_iterations += linear.iterations
        candidate = (
            1 - settings.relaxation
        ) * values + settings.relaxation * linear.values
        candidate[fixed] = boundary[fixed]
        change = _change(candidate, values)
        candidate_operator = freeze(candidate)
        residual = _relative(
            (candidate_operator.apply(candidate) - candidate_operator.rhs)[free], scale
        )
        layer.values = candidate
        passed = (
            linear.status == "completed"
            and residual <= settings.tolerance
            and change <= settings.change_tolerance
        )
        row = _record(
            layer,
            outer=outer,
            residual=residual,
            scale=scale,
            change=change,
            linear=linear.relative_residual,
            accepted=passed,
        )
        if linear.status != "completed":
            layer.status, layer.reason = linear.status, linear.reason
            row["reason"] = linear.reason
            return layer
        if passed:
            layer.status, layer.reason = "completed", "Невязка и изменение поля"
            return layer
        values, operator = candidate, candidate_operator
    return layer


def _newton_layer(problem, previous, t, dt, settings, deadline, layer):
    with _preparing(layer):
        values, _, _ = _initial(problem, previous, t)
        source = problem.evaluate(problem.source, t)
        args = (previous, problem.alpha, source, dt)
        scale = newton_scale3d(values, *args)
        residual, jacobian = newton_system3d(values, *args)
    layer.values = values
    relative = _relative(residual, scale)
    if relative == 0:
        layer.status, layer.reason = "completed", "Нулевая начальная невязка"
        _record(layer, outer=0, residual=0.0, scale=scale, change=0.0, accepted=True)
        return layer
    for outer in range(1, int(settings.max_outer) + 1):
        if perf_counter() >= deadline:
            layer.status, layer.reason = "time_limit", "Лимит времени между итерациями"
            _record(layer, outer=outer - 1, residual=relative, scale=scale)
            return layer
        layer.outer_iterations += 1
        with warnings.catch_warnings():
            warnings.simplefilter("error", MatrixRankWarning)
            correction = spsolve(jacobian, -residual.ravel())
        layer.inner_iterations += 1
        linear_residual = _relative(
            jacobian @ correction + residual.ravel(), _norm(residual)
        )
        correction = correction.reshape((problem.n - 1,) * 3)
        _norm(correction)
        factor = 1.0
        for _ in range(24):
            candidate = values.copy()
            candidate[1:-1, 1:-1, 1:-1] += factor * correction
            try:
                next_residual = newton_residual3d(candidate, *args)
                next_relative = _relative(next_residual, scale)
            except (ValueError, FloatingPointError):
                next_relative = float("inf")
            if (
                next_relative <= (1 - 1e-4 * factor) * relative
                or next_relative <= settings.tolerance
            ):
                break
            factor *= 0.5
        else:
            layer.reason = "Не удалось уменьшить невязку Ньютона"
            _record(
                layer,
                outer=outer,
                residual=relative,
                scale=scale,
                linear=linear_residual,
            )
            return layer
        change = _change(candidate, values)
        # Newton is an independent residual-based diagnostic, not the Picard backend.
        passed = next_relative <= settings.tolerance
        row = _record(
            layer,
            outer=outer,
            residual=next_relative,
            scale=scale,
            change=change,
            linear=linear_residual,
            accepted=passed,
        )
        row["relaxation"] = factor
        layer.values = candidate
        if passed:
            layer.status, layer.reason = "completed", "Невязка независимого Ньютона"
            return layer
        values, relative = candidate, next_relative
        with _preparing(layer):
            residual, jacobian = newton_system3d(values, *args)
    return layer


def solve_heat3d(
    problem,
    points_zyx=None,
    *,
    dt=0.01,
    T=0.02,
    settings=None,
    method="lod",
    progress=None,
    operator_factory=freeze_heat3d,
    backend=solve_lines3d,
):
    """Solve the complete history, returning queries only at the requested T.

    The time limit is soft: checked between complete outer/physical iterations.
    `seconds` includes all data/operator preparation; JIT and any external exact
    solution evaluation are excluded. Newton is limited to n <= 16.
    """
    settings = HeatSettings3D() if settings is None else settings
    settings.validate()
    if method not in ("lod", "newton"):
        raise ValueError("Методы 3D-теплопроводности: lod или newton")
    points = validate_points3d(
        problem.n, [(problem.n // 2,) * 3] if points_zyx is None else points_zyx
    )
    times, steps = time_grid(dt, T)
    unsupported = any(face.kind != 1 for face in problem.faces.values()) or (
        method == "newton" and problem.n > DIRECT_MAX_N
    )
    storage = _workspace_estimate(
        problem.n, method, points, initial_only=not len(steps)
    )
    if (
        method == "lod"
        and backend is solve_lines3d
        and len(steps)
        and not unsupported
        and storage <= settings.max_workspace_bytes
    ):
        warmup_lines3d()
    started = perf_counter()
    deadline = started + settings.max_seconds
    result = HeatRunResult3D(
        method=method,
        status="completed",
        reason="Расчёт завершён",
        target_time=float(T),
        completed_until=0.0,
        points_zyx=points.copy(),
        values=None,
        physical_steps=0,
        outer_iterations=0,
        inner_iterations=0,
        walks=0,
        transitions=0,
        seconds=0.0,
        preparation_seconds=0.0,
        workspace_bytes=storage,
        peak_rss_bytes=None,
        history=[],
    )

    def finish(status=None, reason=None):
        if status is not None:
            result.status = status
            result.reason = reason
        result.seconds = perf_counter() - started
        return result

    if unsupported:
        return finish("unsupported", "Поддержаны условия 1 рода; Newton требует n≤16")
    if storage > settings.max_workspace_bytes:
        return finish("memory_limit", "Оценка рабочих буферов превышает лимит памяти")
    try:
        previous = problem.initial_values()
        result.preparation_seconds = perf_counter() - started
        history_storage = 0
        for index, step in enumerate(steps, 1):
            if perf_counter() >= deadline:
                return finish("time_limit", "Лимит времени между физическими слоями")
            if progress:
                progress(
                    (index - 1) / len(steps), f"Временной слой {index}/{len(steps)}"
                )
            args = (
                problem,
                previous,
                float(times[index]),
                float(step),
                settings,
                deadline,
            )
            layer = _Layer(previous)
            try:
                if method == "lod":
                    _picard_layer(*args, operator_factory, backend, layer)
                else:
                    _newton_layer(*args, layer)
            except (UnsupportedHeatBoundaryError, IncompatibleHeatBoundaryError) as exc:
                layer.status, layer.reason = "unsupported", str(exc)
            except MemoryError:
                layer.status, layer.reason = (
                    "memory_limit",
                    "Не удалось выделить память",
                )
            except MatrixRankWarning as exc:
                layer.status, layer.reason = (
                    "linear_limit",
                    f"Вырожденный якобиан: {exc}",
                )
            except (ValueError, FloatingPointError) as exc:
                layer.status, layer.reason = "nonfinite", str(exc)
            if layer.status != "completed" and (
                not layer.history or layer.history[-1]["reason"] != layer.reason
            ):
                _record(layer, outer=layer.outer_iterations, residual=None, scale=None)
            result.outer_iterations += layer.outer_iterations
            result.inner_iterations += layer.inner_iterations
            result.preparation_seconds += layer.preparation_seconds
            records = [
                dict(layer=index, t=float(times[index]), dt=float(step), **row)
                for row in layer.history
            ]
            history_storage += _history_bytes(records) - getsizeof(records)
            result.history.extend(records)
            result.workspace_bytes = (
                storage + history_storage + getsizeof(result.history)
            )
            if layer.status != "completed":
                return finish(layer.status, f"Слой {index}: {layer.reason}")
            if result.workspace_bytes > settings.max_workspace_bytes:
                result.history[-1]["accepted"] = False
                result.history[-1]["reason"] = "Лимит памяти диагностической истории"
                return finish(
                    "memory_limit", "Диагностическая история превысила лимит памяти"
                )
            previous = layer.values
            result.physical_steps += 1
            result.completed_until = float(times[index])
            if progress:
                progress(index / len(steps), f"Завершён слой {index}/{len(steps)}")
        result.values = previous[tuple(points.T)].copy()
    except (UnsupportedHeatBoundaryError, IncompatibleHeatBoundaryError) as exc:
        return finish("unsupported", str(exc))
    except MemoryError:
        return finish("memory_limit", "Не удалось выделить память")
    except MatrixRankWarning as exc:
        return finish("linear_limit", f"Вырожденный якобиан: {exc}")
    except (ValueError, FloatingPointError) as exc:
        return finish("nonfinite", str(exc))
    return finish()
