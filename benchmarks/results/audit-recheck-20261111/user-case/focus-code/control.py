"""Fixed control approximation for the discrete equation Delta_h u = f."""

import hashlib
from dataclasses import dataclass
from time import perf_counter

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from .problem import Problem, laplacian, reference_solution


def _signature(problem):
    digest = hashlib.sha256()
    digest.update(problem.f.tobytes())
    digest.update(problem.boundary.tobytes())
    return digest.digest()


def _interpolate(values, axis, target):
    y, x = np.meshgrid(target, target, indexing="ij")
    return RegularGridInterpolator((axis, axis), values)(np.stack((y, x), axis=-1))


@dataclass(frozen=True)
class ControlApproximation:
    values: np.ndarray
    source: np.ndarray
    boundary: np.ndarray
    signature: bytes
    preparation_seconds: float
    coarse_n: int | None

    def validate(self, problem):
        if self.values.shape != problem.f.shape or self.signature != _signature(
            problem
        ):
            raise ValueError("Контрольное приближение построено для другой задачи")


def prepare_control(problem, coarse_n=None, approximation=None):
    """Own immutable snapshots; caller can reuse them across independent queries.

    Arbitrary supplied v is for diagnostics. Production v uses a coarse direct
    solve of sampled f and g, never the analytic or fine-grid reference.
    """
    started = perf_counter()
    if approximation is None:
        coarse_n = max(4, problem.n // 4) if coarse_n is None else coarse_n
        if (
            isinstance(coarse_n, bool)
            or not np.isfinite(coarse_n)
            or int(coarse_n) != coarse_n
            or not 4 <= coarse_n <= problem.n
        ):
            raise ValueError("coarse_n должно быть целым от 4 до n")
        coarse_n = int(coarse_n)
        axis = np.linspace(0, 1, coarse_n + 1)
        coarse = Problem(
            "Control",
            coarse_n,
            _interpolate(problem.f, problem.axis, axis),
            _interpolate(problem.boundary, problem.axis, axis),
        )
        values = _interpolate(reference_solution(coarse), axis, problem.axis)
    else:
        values = np.asarray(approximation, dtype=float).copy()
        if values.shape != problem.f.shape or not np.isfinite(values).all():
            raise ValueError("Приближение должно быть конечным массивом формы сетки")
        coarse_n = None
    source = np.zeros_like(problem.f)
    source[1:-1, 1:-1] = problem.f[1:-1, 1:-1] - laplacian(values, problem.h)
    boundary = problem.boundary - values
    # Immutable byte-backed arrays also prevent setflags(write=True).
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
