"""Grouped walks for the existing stationary, uniform, k=1 cube only."""

from time import perf_counter

import numpy as np

from .boundary3d import (
    BoundaryOperator3D,
    BoundaryProblem3D,
    boundary_workspace_bytes3d,
)
from .grouped_mc3d import grouped_walks3d, grouped_workspace_bytes3d
from .problem3d import Problem3D, validate_points3d
from .solvers import NUMBA_LOCK, Result

GROUPED_NAME = "MC · группы частиц"
MAX_BATCH_SIZE = 65536


class GroupedWalkFailure(RuntimeError):
    """No query estimate; actual work before the failed path is retained."""

    def __init__(self, reason, *, seconds, diagnostics, status="path_limit"):
        super().__init__(reason)
        self.seconds = seconds
        self.diagnostics = diagnostics
        self.status = status


def _integer(value, name, minimum, maximum):
    if (
        isinstance(value, (bool, np.bool_))
        or not np.isscalar(value)
        or np.iscomplexobj(value)
    ):
        raise ValueError(f"{name}: нужно целое в [{minimum}, {maximum}]")
    try:
        valid = (
            np.isfinite(value) and int(value) == value and minimum <= value <= maximum
        )
    except (TypeError, ValueError, OverflowError):
        valid = False
    if not valid:
        raise ValueError(f"{name}: нужно целое в [{minimum}, {maximum}]")
    return int(value)


def grouped_solver_workspace3d(problem, walks, batch_size, point_count):
    """Conservative numerical storage, including prepared fields; not peak RSS."""
    fields = (
        boundary_workspace_bytes3d(problem.n)
        if isinstance(problem, BoundaryProblem3D)
        else 12 * problem.f.size * 8
    )
    return (
        fields
        + grouped_workspace_bytes3d(walks, batch_size, point_count)
        + 24 * point_count
    )


def solve_grouped3d(
    problem,
    points,
    *,
    walks=2048,
    seed=42,
    max_steps=None,
    batch_size=128,
    max_workspace_bytes=256 * 1024**2,
):
    if not isinstance(problem, (Problem3D, BoundaryProblem3D)):
        raise ValueError(
            "Групповой MC поддерживает только стационарный 3D-оператор k=1"
        )
    points = validate_points3d(problem.n, points)
    walks = _integer(walks, "walks", 2, 2**31 - 1)
    seed = _integer(seed, "seed", 0, 2**32 - 1)
    batch_size = _integer(batch_size, "batch_size", 1, MAX_BATCH_SIZE)
    limit = (1000 if isinstance(problem, BoundaryProblem3D) else 100) * problem.n**2
    max_steps = _integer(
        limit if max_steps is None else max_steps, "max_steps", 1, 2**63 - 1
    )
    budget = _integer(max_workspace_bytes, "max_workspace_bytes", 0, 2**63 - 1)
    workspace = grouped_solver_workspace3d(problem, walks, batch_size, len(points))
    if workspace > budget:
        raise ValueError("Оценка рабочих массивов группового MC превышает лимит памяти")
    with NUMBA_LOCK:
        started = perf_counter()
        if isinstance(problem, BoundaryProblem3D):
            operator = BoundaryOperator3D(problem)
            operator.check_compatibility()
            if operator.pure_neumann:
                raise ValueError(
                    "MC для условий только 2-го рода не поддерживается: нет поглощения"
                )
            _, reward, killing = operator.mc_coefficients()
            fixed, boundary = operator.fixed, operator.boundary
        else:
            fixed = np.ones(problem.f.shape, dtype=bool)
            fixed[1:-1, 1:-1, 1:-1] = False
            boundary = problem.boundary
            reward = -problem.f / (6 * problem.n**2)
            killing = np.zeros_like(problem.f)
        preparation_seconds = perf_counter() - started
        path_started = perf_counter()
        means, errors, counts, steps, samples, failed, initialized = grouped_walks3d(
            fixed, boundary, reward, killing, points, walks, seed, max_steps, batch_size
        )
        diagnostics = dict(
            batch_size=batch_size,
            effective_batch_size=min(batch_size, walks),
            preparation_seconds=preparation_seconds,
            path_seconds=perf_counter() - path_started,
            workspace_bytes=workspace,
            started_walks=int(initialized),
            walks=int(counts.sum()),
            transitions=int(steps),
            rng="numba_mt19937_interleaved_groups_v1",
            normalization=None,
        )
        if failed >= 0:
            raise GroupedWalkFailure(
                f"Предел {max_steps} этапов пути в точке {points[failed].tolist()}; усечённая оценка не возвращена",
                seconds=perf_counter() - started,
                diagnostics=diagnostics,
            )
        if not np.isfinite(means).all() or not np.isfinite(errors).all():
            raise GroupedWalkFailure(
                "Неконечная оценка группового MC",
                seconds=perf_counter() - started,
                diagnostics=diagnostics,
                status="nonfinite",
            )
        values = np.where(fixed, boundary, np.nan)
        stderr = np.where(fixed, 0.0, np.nan)
        sample_count = np.zeros_like(values, dtype=np.int64)
        indices = tuple(points.T)
        values[indices], stderr[indices], sample_count[indices] = means, errors, counts
        diagnostics["samples"] = samples
        return Result(
            "mc_grouped",
            values,
            stderr,
            sample_count,
            elapsed=perf_counter() - started,
            steps=int(steps),
            walks=int(counts.sum()),
            extras=diagnostics,
        )
