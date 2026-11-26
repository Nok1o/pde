"""Backward Euler with deterministic or Monte Carlo Picard and a Newton reference.

Only the previous and current full fields are kept. Requested values are returned
only after the whole time interval has completed; failed layers remain diagnostic.
"""

import warnings
from contextlib import contextmanager
from dataclasses import dataclass, replace
from functools import partial
from sys import getsizeof
from time import perf_counter

import numpy as np
from scipy.sparse.linalg import MatrixRankWarning, spsolve

from .heat import time_grid
from .heat3d_block import BLOCK_MAX_N, BlockThomasSolver3D, block_workspace_bytes
from .heat3d_conductivity import ConductivityCache3D
from .heat3d_contract import HeatRunResult3D
from .heat3d_history import HISTORY_MODES, store_record
from .heat3d_linear import solve_lines3d, warmup_lines3d
from .heat3d_mc import _integer, warmup_mc3d
from .heat3d_operator import (
    IncompatibleHeatBoundaryError,
    UnsupportedHeatBoundaryError,
    dirichlet_data3d,
    freeze_heat3d,
)
from .heat3d_picard_mc import (
    CONTROL_MC_METHODS,
    GROUPED_MC_METHODS,
    MC_METHODS,
    PicardSampler3D,
)
from .heat3d_reference import (
    newton_residual3d,
    newton_scale3d,
    newton_system3d,
)
from .heat3d_transport import solve_mass_lines3d, warmup_transport3d
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
    walks: int = 4096  # per free node per update or independent check
    seed: int = 42
    coarse_n: int = 4
    max_walks: int = 8_000_000  # for the entire time history
    max_transitions: int = 80_000_000
    max_path_steps: int = 1_000_000
    mc_tolerance: float | None = None
    mc_change_tolerance: float | None = None
    batch_size: int = 128
    conductivity_refresh_layers: int = 1
    history_mode: str = "full"

    @classmethod
    def for_methods(cls, methods):
        if any(method in MC_METHODS for method in methods):
            return cls(mc_tolerance=0.2, mc_change_tolerance=0.1)
        return cls()

    def for_method(self, method):
        if method not in MC_METHODS:
            return self
        return replace(
            self,
            tolerance=(
                self.tolerance if self.mc_tolerance is None else self.mc_tolerance
            ),
            change_tolerance=(
                self.change_tolerance
                if self.mc_change_tolerance is None
                else self.mc_change_tolerance
            ),
        )

    def validate(self):
        _integer(
            self.conductivity_refresh_layers, "conductivity_refresh_layers", 1, 10000
        )
        if self.history_mode not in HISTORY_MODES:
            raise ValueError("history_mode: требуется full или layers")
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

    def validate_mc(self, method):
        _integer(self.walks, "walks", 2, 2**31 - 1)
        _integer(self.seed, "seed", 0, 2**32 - 1)
        for name in ("max_walks", "max_transitions", "max_path_steps"):
            _integer(getattr(self, name), name, 1, 2**63 - 1)
        if method in CONTROL_MC_METHODS:
            _integer(self.coarse_n, "coarse_n", 4, 64)
        if method in GROUPED_MC_METHODS:
            _integer(self.batch_size, "batch_size", 1, 65536)


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


def _workspace_estimate(n, method, points, initial_only=False, free_nodes=None):
    # Explicit numerical-buffer estimate; this is not a measured process RSS.
    # Initial data evaluation also creates several complete temporary fields.
    buffers = 40 * (n + 1) ** 3 * 8
    if method == "newton" and not initial_only:
        # Conservative dense-fill storage allowance for the small sparse LU.
        unknowns = (n - 1) ** 3 if free_nodes is None else free_nodes
        buffers += 16 * unknowns**2
    if method == "block" and not initial_only:
        buffers += block_workspace_bytes(n)
    return buffers + 2 * points.nbytes + len(points) * 8


def _time_grid_bytes(dt, T):
    """Bound the shared time_grid arrays before allocating them (same limits)."""
    if not np.isfinite(dt) or dt <= 0 or not np.isfinite(T) or T < 0:
        raise ValueError("Требуются конечные dt>0 и T≥0")
    ratio = T / dt
    if not np.isfinite(ratio) or ratio > 10000:
        raise ValueError("Не более 10000 временных шагов за один расчёт")
    count = int(np.ceil(ratio))
    # Include arange/multiply/append temporaries, not just the two returned arrays.
    return (3 * count + 4 if count else 1) * np.dtype(float).itemsize


@dataclass
class _Layer:
    values: np.ndarray | None
    history: list  # the complete run's list; each record is stored once
    number: int
    t: float
    dt: float
    history_budget: int
    history_storage: int
    history_mode: str = "full"
    history_records_discarded: int = 0
    status: str = "outer_limit"
    reason: str = "Лимит внешних итераций"
    outer_iterations: int = 0
    inner_iterations: int = 0
    preparation_seconds: float = 0.0
    assembly_seconds: float = 0.0
    factorization_seconds: float = 0.0
    solve_seconds: float = 0.0
    factorizations: int = 0
    factor_reuses: int = 0
    mc_attempt: dict | None = None
    conductivity_reuses: int = 0
    conductivity_fallbacks: int = 0
    lagged_conductivity: bool = False
    node_updates: int = 0
    transport_sweeps: int = 0
    edge_transfers: int = 0
    transport_attempt: dict | None = None


@contextmanager
def _preparing(layer):
    started = perf_counter()
    try:
        yield
    finally:
        layer.preparation_seconds += perf_counter() - started


def _record(
    layer,
    *,
    outer,
    residual,
    scale,
    change=None,
    linear=None,
    accepted=False,
    relaxation=None,
):
    row = dict(
        layer=layer.number,
        t=layer.t,
        dt=layer.dt,
        outer=outer,
        linear_relative_residual=linear,
        nonlinear_relative_residual=residual,
        residual_scale=scale,
        relative_change=change,
        accepted=accepted,
        walks=0,
        transitions=0,
        node_updates=0,
        transport_sweeps=0,
        edge_transfers=0,
        reason="completed" if accepted else layer.reason,
        lagged_conductivity=layer.lagged_conductivity,
        conductivity_reuses=layer.conductivity_reuses,
        conductivity_fallbacks=layer.conductivity_fallbacks,
    )
    if relaxation is not None:
        row["relaxation"] = relaxation
    if layer.mc_attempt is not None:
        row.update(layer.mc_attempt)
        layer.mc_attempt = None
    if layer.transport_attempt is not None:
        row.update(layer.transport_attempt)
        layer.transport_attempt = None
    stored_bytes, discarded = store_record(layer.history, row, layer.history_mode)
    layer.history_storage += stored_bytes
    layer.history_records_discarded += discarded
    if layer.history_storage + getsizeof(layer.history) > layer.history_budget:
        layer.status = "memory_limit"
        layer.reason = "Лимит памяти диагностической истории"
        layer.history_storage += getsizeof(layer.reason) - getsizeof(row["reason"])
        row.update(accepted=False, reason=layer.reason)
    return row


def _initial(problem, previous, t):
    fixed, boundary = dirichlet_data3d(problem, t)
    values = previous.copy()
    values[fixed] = boundary[fixed]
    return values, fixed, boundary


def _deterministic_update(layer, backend, operator, values, settings):
    linear = backend(
        operator,
        values,
        tolerance=settings.linear_tolerance,
        max_iterations=int(settings.max_inner),
    )
    layer.inner_iterations += linear.iterations
    if linear.node_updates or linear.transport_sweeps:
        layer.node_updates += linear.node_updates
        layer.transport_sweeps += linear.transport_sweeps
        layer.edge_transfers += linear.edge_transfers
        layer.transport_attempt = dict(
            node_updates=linear.node_updates,
            transport_sweeps=linear.transport_sweeps,
            edge_transfers=linear.edge_transfers,
            linear_tail_error_bound=linear.tail_error_bound,
        )
    for name in (
        "assembly_seconds",
        "factorization_seconds",
        "solve_seconds",
        "factorizations",
        "factor_reuses",
    ):
        setattr(layer, name, getattr(layer, name) + getattr(linear, name, 0))
    layer.preparation_seconds += getattr(linear, "assembly_seconds", 0)
    layer.preparation_seconds += linear.preparation_seconds
    if linear.status != "completed":
        layer.status, layer.reason = linear.status, linear.reason
    return linear


def _picard_layer(
    problem,
    previous,
    t,
    dt,
    settings,
    deadline,
    factory,
    backend,
    layer,
    sampler=None,
    conductivity_cache=None,
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
    if conductivity_cache is not None:
        with _preparing(layer):
            if conductivity_cache.can_reuse(layer.number):
                operator = freeze_heat3d(
                    problem,
                    previous,
                    values,
                    t,
                    dt,
                    conductivity=conductivity_cache.values,
                )
                layer.lagged_conductivity = True
            else:
                conductivity_cache.refresh(problem, values, layer.number)
    for outer in range(1, int(settings.max_outer) + 1):
        if perf_counter() >= deadline:
            layer.status, layer.reason = "time_limit", "Лимит времени между итерациями"
            _record(layer, outer=outer - 1, residual=residual, scale=scale)
            return layer
        layer.outer_iterations += 1
        if layer.lagged_conductivity:
            layer.conductivity_reuses += 1
        if sampler is not None:
            # Keep counters available to the outer failure handler if evaluating
            # the nonlinear residual fails after a successful batch.
            layer.mc_attempt = {}
            estimate = sampler.draw(layer, operator, previous, values, outer)
            if estimate is None:
                _record(
                    layer,
                    outer=outer,
                    residual=residual,
                    scale=scale,
                )
                return layer
            linear_residual = _relative(
                (operator.apply(estimate) - operator.rhs)[free],
                _norm(operator.rhs[free]),
            )
            linear_ok = True
        else:
            linear = _deterministic_update(layer, backend, operator, values, settings)
            estimate, linear_residual = linear.values, linear.relative_residual
            linear_ok = linear.status == "completed"
        candidate = (1 - settings.relaxation) * values + settings.relaxation * estimate
        candidate[fixed] = boundary[fixed]
        layer.values = candidate
        change = _change(candidate, values)
        candidate_operator = freeze(candidate)
        residual = _relative(
            (candidate_operator.apply(candidate) - candidate_operator.rhs)[free], scale
        )
        passed = (
            linear_ok
            and residual <= settings.tolerance
            and change <= settings.change_tolerance
        )
        if passed and sampler is not None:
            passed = False
            if perf_counter() < deadline:
                check = sampler.draw(
                    layer,
                    candidate_operator,
                    previous,
                    candidate,
                    outer,
                    check=True,
                )
                if check is not None:
                    check_operator = freeze(check)
                    check_residual = _relative(
                        (check_operator.apply(check) - check_operator.rhs)[free], scale
                    )
                    check_change = _change(check, candidate)
                    layer.mc_attempt.update(
                        validation_residual=check_residual,
                        validation_change=check_change,
                    )
                    passed = (
                        check_residual <= settings.tolerance
                        and check_change <= settings.change_tolerance
                    )
        if (
            sampler is not None
            and perf_counter() >= deadline
            and layer.status == "outer_limit"
        ):
            layer.status, layer.reason = (
                "time_limit",
                "Истёк лимит времени полного запуска MC",
            )
            passed = False
        _record(
            layer,
            outer=outer,
            residual=residual,
            scale=scale,
            change=change,
            linear=linear_residual,
            accepted=passed,
        )
        if layer.status != "outer_limit":
            return layer
        if passed:
            layer.status = "completed"
            layer.reason = (
                "Независимая проверка MC"
                if sampler is not None
                else "Невязка и изменение поля"
            )
            return layer
        values, operator = candidate, candidate_operator
        if conductivity_cache is not None and outer < settings.max_outer:
            with _preparing(layer):
                conductivity_cache.refresh(problem, values, layer.number)
            if layer.lagged_conductivity:
                layer.conductivity_fallbacks += 1
            layer.lagged_conductivity = False
    return layer


def _newton_layer(problem, previous, t, dt, settings, deadline, layer):
    with _preparing(layer):
        values, fixed, _ = _initial(problem, previous, t)
        free = ~fixed
        source = problem.evaluate(problem.source, t)
        args = (previous, problem.alpha, source, dt)
        boundary_args = dict(problem=problem, t=t)
        scale = newton_scale3d(values, *args, **boundary_args)
        residual, jacobian = newton_system3d(values, *args, **boundary_args)
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
        _norm(correction)
        for backtrack in range(24):
            factor = 2.0 ** (-backtrack)
            candidate = values.copy()
            candidate[free] += factor * correction
            layer.values = candidate
            try:
                next_residual = newton_residual3d(candidate, *args, **boundary_args)
                next_relative = _relative(next_residual, scale)
            except (ValueError, FloatingPointError):
                next_relative = float("inf")
            if (
                next_relative <= (1 - 1e-4 * factor) * relative
                or next_relative <= settings.tolerance
            ):
                break
        else:
            layer.reason = "Не удалось уменьшить невязку Ньютона"
            _record(
                layer,
                outer=outer,
                residual=next_relative,
                scale=scale,
                linear=linear_residual,
                change=(
                    _change(candidate, values) if np.isfinite(candidate).all() else None
                ),
                relaxation=factor,
            )
            return layer
        change = _change(candidate, values)
        # Newton is an independent residual-based diagnostic, not the Picard backend.
        passed = next_relative <= settings.tolerance
        _record(
            layer,
            outer=outer,
            residual=next_relative,
            scale=scale,
            change=change,
            linear=linear_residual,
            accepted=passed,
            relaxation=factor,
        )
        if layer.status == "memory_limit":
            return layer
        layer.values = candidate
        if passed:
            layer.status, layer.reason = "completed", "Невязка независимого Ньютона"
            return layer
        values, relative = candidate, next_relative
        with _preparing(layer):
            residual, jacobian = newton_system3d(values, *args, **boundary_args)
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
    started = perf_counter()
    is_mc = method in MC_METHODS
    if settings is None:
        settings = HeatSettings3D.for_methods((method,))
    settings = settings.for_method(method)
    settings.validate()
    if is_mc:
        settings.validate_mc(method)
    if method not in ("lod", "block", "mass_transport", "newton", *MC_METHODS):
        raise ValueError("Неизвестный метод 3D-теплопроводности")
    batch_size = int(settings.batch_size) if method in GROUPED_MC_METHODS else None
    points = validate_points3d(
        problem.n, [(problem.n // 2,) * 3] if points_zyx is None else points_zyx
    )
    time_storage = _time_grid_bytes(dt, T)
    unsupported = (
        any(face.kind not in (1, 3) for face in problem.faces.values())
        or (method == "newton" and problem.n > DIRECT_MAX_N)
        or (method == "block" and problem.n > BLOCK_MAX_N)
        or (method in CONTROL_MC_METHODS and T > 0 and settings.coarse_n >= problem.n)
        or (
            settings.conductivity_refresh_layers > 1
            and method != "newton"
            and operator_factory is not freeze_heat3d
        )
    )
    # Count the Cartesian free-node set without allocating a grid or sampling data.
    free_nodes = 1
    for axis in "xyz":
        free_nodes *= (
            problem.n
            + 1
            - sum(problem.faces[f"{axis}{side}"].kind == 1 for side in (0, 1))
        )
    storage = (
        _workspace_estimate(
            problem.n, method, points, initial_only=T == 0, free_nodes=free_nodes
        )
        + time_storage
    )
    if is_mc and T > 0:
        # Full-field query indices, the frozen estimator's buffers and, for
        # control, coarse solve/interpolation. Outer snapshots are in base storage.
        size = (problem.n + 1) ** 3
        storage += (
            768 * size + 512 * (settings.coarse_n + 1) ** 3
            if method in CONTROL_MC_METHODS
            else (448 if method == "mc_counts" else 384) * size
        )
        storage += 24 * size + getsizeof(set())
        if batch_size is not None:
            storage += 40 * min(batch_size, int(settings.walks))
    if T > 0 and method != "newton" and settings.conductivity_refresh_layers > 1:
        # Retained k and the replacement allocation during refresh.
        storage += 16 * (problem.n + 1) ** 3
    warmup_seconds = 0.0
    if T > 0 and not unsupported and storage <= settings.max_workspace_bytes:
        warm_start = perf_counter()
        if method in ("mc_counts", "mass_transport"):
            warmup_transport3d()
        elif is_mc:
            warmup_mc3d(batch_size=batch_size)
        if method in CONTROL_MC_METHODS or (
            method == "lod" and backend is solve_lines3d
        ):
            warmup_lines3d()
        warmup_seconds = perf_counter() - warm_start
        started += warmup_seconds
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
        warmup_seconds=warmup_seconds,
        batch_size=batch_size,
        conductivity_refresh_layers=(
            1 if method == "newton" else int(settings.conductivity_refresh_layers)
        ),
        history_mode=settings.history_mode,
    )

    def finish(status=None, reason=None):
        if status is not None:
            result.status = status
            result.reason = reason
        result.seconds = perf_counter() - started
        return result

    if unsupported:
        if any(face.kind not in (1, 3) for face in problem.faces.values()):
            reason = "Поддержаны граничные условия 1 и 3 рода"
        elif method == "block" and problem.n > BLOCK_MAX_N:
            reason = (
                f"Матричная прогонка ограничена n≤{BLOCK_MAX_N} из-за плотных блоков"
            )
        elif method in CONTROL_MC_METHODS:
            reason = (
                "Для MC с поправкой требуется 4 ≤ coarse_n < n"
                if settings.coarse_n >= problem.n
                else "Задержка проводимости требует стандартный оператор"
            )
        elif method != "newton":
            reason = "Задержка проводимости требует стандартный оператор"
        else:
            reason = f"Независимый Ньютон ограничен n≤{DIRECT_MAX_N}"
        return finish("unsupported", reason)
    if storage > settings.max_workspace_bytes:
        return finish("memory_limit", "Оценка рабочих буферов превышает лимит памяти")
    if method == "block" and backend is solve_lines3d:
        backend = BlockThomasSolver3D(
            max_workspace_bytes=int(settings.max_workspace_bytes)
        )
    if method == "mass_transport" and backend is solve_lines3d:
        backend = partial(solve_mass_lines3d, deadline=deadline)
    try:
        with _preparing(result):
            times, steps = time_grid(dt, T)
            previous = problem.initial_values()
            sampler = (
                PicardSampler3D(problem, settings, result)
                if is_mc and len(steps)
                else None
            )
            conductivity_cache = (
                ConductivityCache3D(int(settings.conductivity_refresh_layers))
                if method != "newton" and settings.conductivity_refresh_layers > 1
                else None
            )
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
            records_before = len(result.history)
            layer = _Layer(
                values=None,
                history=result.history,
                number=index,
                t=float(times[index]),
                dt=float(step),
                history_budget=settings.max_workspace_bytes - storage,
                history_storage=history_storage,
                history_mode=settings.history_mode,
            )
            try:
                if method == "newton":
                    _newton_layer(*args, layer)
                else:
                    _picard_layer(
                        *args,
                        operator_factory,
                        backend,
                        layer,
                        sampler,
                        conductivity_cache,
                    )
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
                len(layer.history) == records_before
                or layer.history[-1]["reason"] != layer.reason
            ):
                _record(layer, outer=layer.outer_iterations, residual=None, scale=None)
            result.outer_iterations += layer.outer_iterations
            result.history_records_discarded += layer.history_records_discarded
            result.inner_iterations += layer.inner_iterations
            result.preparation_seconds += layer.preparation_seconds
            result.conductivity_reuses += layer.conductivity_reuses
            result.conductivity_fallbacks += layer.conductivity_fallbacks
            result.node_updates += layer.node_updates
            result.transport_sweeps += layer.transport_sweeps
            result.edge_transfers += layer.edge_transfers
            for name in (
                "assembly_seconds",
                "factorization_seconds",
                "solve_seconds",
                "factorizations",
                "factor_reuses",
            ):
                setattr(result, name, getattr(result, name) + getattr(layer, name))
            history_storage = layer.history_storage
            result.workspace_bytes = (
                storage + history_storage + getsizeof(result.history)
            )
            if layer.status != "completed":
                result.diagnostic_field = layer.values
                if layer.values is not None:
                    result.diagnostic_time = float(times[index])
                return finish(layer.status, f"Слой {index}: {layer.reason}")
            if result.workspace_bytes > settings.max_workspace_bytes:
                result.history[-1]["accepted"] = False
                result.history[-1]["reason"] = "Лимит памяти диагностической истории"
                result.diagnostic_field = layer.values
                result.diagnostic_time = float(times[index])
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
