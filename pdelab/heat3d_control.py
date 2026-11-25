"""PDE-28 stage 2: a coarse frozen solve plus a fine-operator MC correction.

No Picard/time advancement here. All fresh coarse preparation is charged.
The supplied-control entry point is diagnostic and reports excluded construction.
"""

from dataclasses import dataclass, replace
from time import perf_counter

import numpy as np

from .control3d import _interpolate3d
from .heat3d_linear import solve_lines3d, warmup_lines3d
from .heat3d_mc import FrozenMCResult3D, _integer, solve_frozen_mc3d, warmup_mc3d
from .heat3d_operator import HeatOperator3D, _field, freeze_heat3d
from .problem3d import validate_points3d


@dataclass
class ControlledFrozenMCResult3D(FrozenMCResult3D):
    method: str = "mc_control"
    coarse_n: int | None = None
    coarse_iterations: int = 0
    coarse_relative_residual: float | None = None
    coarse_seconds: float = 0.0
    interpolation_seconds: float = 0.0
    operator_seconds: float = 0.0
    defect_seconds: float = 0.0
    control_origin: str = "supplied_construction_excluded"


def _options(settings):
    defaults = dict(
        walks=2048,
        seed=42,
        max_steps=1000000,
        max_walks=1000000,
        max_transitions=10000000,
        max_workspace_bytes=256 * 1024**2,
        batch_size=None,
    )
    if settings.keys() - defaults.keys():
        raise ValueError(
            f"Неизвестные настройки MC: {sorted(settings.keys()-defaults.keys())}"
        )
    defaults.update(settings)
    batch_size = defaults.pop("batch_size")
    for name, value in defaults.items():
        minimum = 0 if name == "seed" else 2 if name == "walks" else 1
        maximum = (
            2**32 - 1 if name == "seed" else 2**31 - 1 if name == "walks" else 2**63 - 1
        )
        defaults[name] = _integer(value, name, minimum, maximum)
    defaults["batch_size"] = (
        None if batch_size is None else _integer(batch_size, "batch_size", 1, 65536)
    )
    return defaults


def _failure(points, options, workspace, status, reason, seconds):
    return ControlledFrozenMCResult3D(
        status,
        reason,
        points.copy(),
        None,
        None,
        None,
        None,
        0,
        0,
        options["walks"],
        options["seed"],
        seconds,
        seconds,
        workspace,
        method="mc_control" if options["batch_size"] is None else "mc_control_grouped",
        batch_size=options["batch_size"],
    )


def solve_frozen_correction_mc3d(operator, control, points=None, **settings):
    """Diagnostic supplied v: estimate v+e with A e=b-A v on the SAME matrix.

    Fixed rows contain g-v (not zero unless v satisfies the fine boundary).
    Operator and supplied v construction are excluded, explicitly in origin.
    batch_size=None uses ordinary paths; a positive size groups the same paths.
    """
    validation_start = perf_counter()
    if not isinstance(operator, HeatOperator3D):
        raise ValueError("Ожидается HeatOperator3D")
    options = _options(settings)
    points = validate_points3d(
        operator.n, [[operator.n // 2] * 3] if points is None else points
    )
    _field(control, operator.shape, "control")
    workspace = (
        sum(
            getattr(operator, k).nbytes
            for k in ("diagonal", "weights", "rhs", "fixed", "fixed_values", "volumes")
        )
        + 160 * operator.rhs.size
        + 256 * len(points)
    )
    if options["batch_size"] is not None:
        workspace += 40 * min(options["batch_size"], options["walks"])
    if workspace > options["max_workspace_bytes"]:
        return _failure(
            points,
            options,
            workspace,
            "memory_limit",
            "control_workspace_exceeds_limit",
            perf_counter() - validation_start,
        )
    warmup_mc3d(batch_size=options["batch_size"])
    start = perf_counter()
    control = np.array(control, dtype=float, copy=True)
    with np.errstate(over="ignore", invalid="ignore"):
        defect = operator.rhs - operator.apply(control)
    if not np.isfinite(defect).all():
        return _failure(
            points,
            options,
            workspace,
            "nonfinite",
            "nonfinite_control_defect",
            perf_counter() - start,
        )
    correction = operator.with_rhs(defect)
    defect_seconds = perf_counter() - start
    estimate = solve_frozen_mc3d(correction, points, **options)
    shift = control[tuple(points.T)]
    with np.errstate(over="ignore", invalid="ignore"):
        values = None if estimate.values is None else shift + estimate.values
        diagnostics = (
            None
            if estimate.diagnostic_values is None
            else shift + estimate.diagnostic_values
        )
    known = operator.fixed[tuple(points.T)]
    if values is not None:
        values[known] = operator.fixed_values[tuple(points.T)][known]
    if diagnostics is not None:
        diagnostics[known] = operator.fixed_values[tuple(points.T)][known]
    # A completed correction can overflow when added to an extreme supplied v.
    if values is not None and not np.isfinite(values).all():
        estimate.status, estimate.reason = (
            "nonfinite",
            "nonfinite_control_plus_correction",
        )
        values = None
        estimate.stderr = None
    return ControlledFrozenMCResult3D(
        **{
            **vars(estimate),
            "values": values,
            "diagnostic_values": diagnostics,
            "seconds": perf_counter() - start,
            "preparation_seconds": defect_seconds + estimate.preparation_seconds,
            "workspace_bytes": workspace,
            "method": (
                "mc_control" if options["batch_size"] is None else "mc_control_grouped"
            ),
        },
        defect_seconds=defect_seconds,
    )


def solve_controlled_frozen_mc3d(
    problem,
    previous,
    iterate,
    t,
    dt,
    points=None,
    *,
    coarse_n=4,
    coarse_tolerance=1e-8,
    coarse_max_iterations=20000,
    **settings,
):
    """Build v from a strictly smaller grid, then correct the fine frozen layer.

    Restrict caller-computed previous/iterate fields, sample known F/g/r on
    coarse coordinates with the same face kinds and beta, freeze once, solve
    coarse lines, prolong v without overwriting its boundary, and pay the full
    fine residual. No exact or fine solve access.
    batch_size selects only the correction kernel; coarse preparation is unchanged.
    """
    validation_start = perf_counter()
    options = _options(settings)
    coarse_n = _integer(coarse_n, "coarse_n", 4, problem.n - 1)
    coarse_max_iterations = _integer(
        coarse_max_iterations, "coarse_max_iterations", 1, 2**31 - 1
    )
    if not np.isfinite(coarse_tolerance) or not 0 < coarse_tolerance < 1:
        raise ValueError("coarse_tolerance должен лежать между 0 и 1")
    previous = _field(previous, problem.shape, "previous")
    iterate = _field(iterate, problem.shape, "iterate")
    points = validate_points3d(
        problem.n, [[problem.n // 2] * 3] if points is None else points
    )
    # Includes fine/coarse fields, immutable copies, interpolation coordinates,
    # residual/snapshot, MC arrays and temporary line-solver buffers; not RSS.
    workspace = 512 * (np.prod(problem.shape).item() + (coarse_n + 1) ** 3) + 256 * len(
        points
    )
    if options["batch_size"] is not None:
        workspace += 40 * min(options["batch_size"], options["walks"])
    if workspace > options["max_workspace_bytes"]:
        result = _failure(
            points,
            options,
            workspace,
            "memory_limit",
            "coarse_control_workspace_exceeds_limit",
            perf_counter() - validation_start,
        )
        return replace(result, coarse_n=coarse_n, control_origin="fresh_coarse_grid")
    warmup_lines3d()
    warmup_mc3d(batch_size=options["batch_size"])
    start = perf_counter()
    fine = freeze_heat3d(problem, previous, iterate, t, dt)
    operator_seconds = perf_counter() - start
    fine_axis = np.linspace(0, 1, problem.n + 1)
    coarse_axis = np.linspace(0, 1, coarse_n + 1)
    interpolation_start = perf_counter()
    coarse_previous = _interpolate3d(previous, fine_axis, coarse_axis)
    coarse_iterate = _interpolate3d(iterate, fine_axis, coarse_axis)
    interpolation_seconds = perf_counter() - interpolation_start
    coarse_start = perf_counter()
    coarse_problem = replace(problem, n=coarse_n, exact=None)
    coarse = freeze_heat3d(coarse_problem, coarse_previous, coarse_iterate, t, dt)
    solved = solve_lines3d(
        coarse,
        coarse_iterate,
        tolerance=coarse_tolerance,
        max_iterations=coarse_max_iterations,
    )
    coarse_seconds = perf_counter() - coarse_start
    if solved.status != "completed":
        result = _failure(
            points,
            options,
            workspace,
            solved.status,
            f"coarse_{solved.reason}",
            perf_counter() - start,
        )
        return replace(
            result,
            coarse_n=coarse_n,
            coarse_iterations=solved.iterations,
            coarse_relative_residual=solved.relative_residual,
            coarse_seconds=coarse_seconds,
            interpolation_seconds=interpolation_seconds,
            operator_seconds=operator_seconds,
            control_origin="fresh_coarse_grid",
        )
    interpolation_start = perf_counter()
    control = _interpolate3d(solved.values, coarse_axis, fine_axis)
    interpolation_seconds += perf_counter() - interpolation_start
    correction_start = perf_counter()
    result = solve_frozen_correction_mc3d(fine, control, points, **options)
    return replace(
        result,
        seconds=perf_counter() - start,
        preparation_seconds=correction_start - start + result.preparation_seconds,
        workspace_bytes=workspace,
        coarse_n=coarse_n,
        coarse_iterations=solved.iterations,
        coarse_relative_residual=solved.relative_residual,
        coarse_seconds=coarse_seconds,
        interpolation_seconds=interpolation_seconds,
        operator_seconds=operator_seconds,
        control_origin="fresh_coarse_grid",
    )
