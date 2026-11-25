"""Coarse control and exact residual for the seven-point Poisson system."""

from time import perf_counter

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from .control import ControlApproximation, _signature
from .problem3d import DIRECT_MAX_N, Problem3D, laplacian3d, reference_solution3d


def validate_coarse3d(n, coarse_n):
    maximum = min(n, DIRECT_MAX_N)
    if (
        isinstance(coarse_n, (bool, np.bool_))
        or not np.isscalar(coarse_n)
        or not np.isfinite(coarse_n)
        or int(coarse_n) != coarse_n
        or not 4 <= coarse_n <= maximum
    ):
        raise ValueError(f"Контрольная сетка 3D: целое от 4 до {maximum}")
    return int(coarse_n)


def _interpolate3d(values, axis, target):
    coordinates = np.stack(np.meshgrid(target, target, target, indexing="ij"), axis=-1)
    return RegularGridInterpolator((axis, axis, axis), values)(coordinates)


def prepare_control3d(problem, coarse_n=None, approximation=None):
    """Build from f/g only; never use the analytic or fine-grid reference.

    Supplied approximations are for residual diagnostics. Production controls
    solve a coarse system, then interpolate in all three coordinates [z,y,x].
    """
    started = perf_counter()
    if approximation is None:
        coarse_n = max(4, problem.n // 4) if coarse_n is None else coarse_n
        coarse_n = validate_coarse3d(problem.n, coarse_n)
        axis = np.linspace(0, 1, coarse_n + 1)
        coarse = Problem3D(
            "Контрольное поле 3D",
            coarse_n,
            _interpolate3d(problem.f, problem.axis, axis),
            _interpolate3d(problem.boundary, problem.axis, axis),
        )
        values = _interpolate3d(reference_solution3d(coarse), axis, problem.axis)
    else:
        if coarse_n is not None:
            raise ValueError("Задайте контрольную сетку или готовое приближение")
        values = np.asarray(approximation, dtype=float).copy()
        if values.shape != problem.f.shape or not np.isfinite(values).all():
            raise ValueError("Приближение должно быть конечным массивом формы 3D-сетки")
    source = np.zeros_like(problem.f)
    source[1:-1, 1:-1, 1:-1] = problem.f[1:-1, 1:-1, 1:-1] - laplacian3d(
        values, problem.h
    )
    boundary = problem.boundary - values
    if not all(np.isfinite(a).all() for a in (values, source, boundary)):
        raise ValueError("Контрольная поправка должна быть конечной")
    values, source, boundary = [
        np.frombuffer(a.tobytes(), dtype=a.dtype).reshape(a.shape)
        for a in (values, source, boundary)
    ]
    return ControlApproximation(
        values,
        source,
        boundary,
        _signature(problem),
        perf_counter() - started,
        coarse_n,
    )
