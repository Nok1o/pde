"""Пуассон в кубе: координаты (x, y, z), массивы [z, y, x]."""

from dataclasses import dataclass

import numpy as np
from scipy.sparse import diags, kronsum
from scipy.sparse.linalg import spsolve

MAX_N = 64
DIRECT_MAX_N = 16
PRESETS_3D = {
    "sine": "Пуассон: синус в кубе",
    "affine": "Лаплас: линейное поле",
    "constant": "Лаплас: постоянное поле",
}


def validate_n(n):
    if isinstance(n, (bool, np.bool_)) or not np.isscalar(n):
        raise ValueError(f"n должно быть целым от 4 до {MAX_N}")
    if not np.isfinite(n) or int(n) != n or not 4 <= n <= MAX_N:
        raise ValueError(f"n должно быть целым от 4 до {MAX_N}")
    return int(n)


def workspace_bytes(n):
    """Резерв для 24 объёмных массивов; не включает JIT, графики и sparse LU."""
    n = validate_n(n)
    return 24 * (n + 1) ** 3 * np.dtype(float).itemsize


@dataclass
class Problem3D:
    name: str
    n: int
    f: np.ndarray
    boundary: np.ndarray
    exact: np.ndarray | None = None

    def __post_init__(self):
        self.n = validate_n(self.n)
        for name in ("f", "boundary", "exact"):
            value = getattr(self, name)
            if value is None and name == "exact":
                continue
            array = np.asarray(value, dtype=float)
            if array.shape != (self.n + 1,) * 3 or not np.isfinite(array).all():
                raise ValueError(
                    f"{name}: нужен конечный массив [z, y, x] формы {(self.n + 1,) * 3}"
                )
            setattr(self, name, array.copy())

    @property
    def h(self):
        return 1.0 / self.n

    @property
    def axis(self):
        return np.linspace(0, 1, self.n + 1)

    @classmethod
    def from_functions(cls, n, source, boundary, exact=None, name="Своя 3D-задача"):
        n = validate_n(n)  # Ограничиваем размер до выделения массивов.
        axis = np.linspace(0, 1, n + 1)
        x, y, z = axis[None, None, :], axis[None, :, None], axis[:, None, None]

        def field(function):
            return np.broadcast_to(function(x, y, z), (n + 1,) * 3)

        return cls(
            name,
            n,
            field(source),
            field(boundary),
            None if exact is None else field(exact),
        )


def make_problem3d(kind="sine", n=16):
    if kind not in PRESETS_3D:
        raise ValueError(f"Неизвестная 3D-задача: {kind}")
    if kind == "sine":

        def exact(x, y, z):
            def factor(coordinate):
                return np.where(
                    (coordinate == 0) | (coordinate == 1),
                    0.0,
                    np.sin(np.pi * coordinate),
                )

            return factor(x) * factor(y) * factor(z)

        def source(x, y, z):
            return -3 * np.pi**2 * exact(x, y, z)

    elif kind == "affine":

        def exact(x, y, z):
            return 1 + 2 * x + 3 * y + 5 * z

        def source(x, y, z):
            return 0.0

    else:

        def exact(x, y, z):
            return 7.0

        def source(x, y, z):
            return 0.0

    return Problem3D.from_functions(n, source, exact, exact, PRESETS_3D[kind])


def validate_points3d(n, points):
    """Внутренние индексы (z, y, x), включая известные граничные узлы."""
    n = validate_n(n)
    indices = np.asarray(points)
    if indices.ndim != 2 or indices.shape[1] != 3 or len(indices) == 0:
        raise ValueError("Нужен непустой список индексов (z, y, x)")
    if not np.isfinite(indices).all() or np.any(indices != np.floor(indices)):
        raise ValueError("Индексы узлов должны быть целыми")
    if np.any(indices < 0) or np.any(indices > n):
        raise ValueError("Индексы узлов должны лежать в [0, n]")
    indices = np.ascontiguousarray(indices, dtype=np.int64)
    if len(np.unique(indices, axis=0)) != len(indices):
        raise ValueError("Точки не должны повторяться")
    return indices


def select_points3d(n, mode="point", point=None, plane="xy", index=None, points=None):
    """Публичные координаты point/points — (x, y, z); результат — (z, y, x)."""
    n = validate_n(n)
    if mode == "point":
        xyz = (n // 2,) * 3 if point is None else point
        coordinates = np.asarray([xyz])
        if coordinates.shape != (1, 3):
            raise ValueError("point: нужны три координаты (x, y, z)")
        result = coordinates[:, ::-1]
    elif mode == "points":
        xyz = np.asarray(points)
        if xyz.ndim != 2 or xyz.shape[1] != 3:
            raise ValueError("points: нужен список координат (x, y, z)")
        result = xyz[:, ::-1]
    elif mode in ("field", "patch"):
        axis = (
            np.arange(1, n)
            if mode == "field"
            else np.arange(max(1, n // 2 - 1), min(n, n // 2 + 2))
        )
        result = np.stack(
            np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1
        ).reshape(-1, 3)
    elif mode == "slice":
        if plane not in ("xy", "xz", "yz"):
            raise ValueError("Плоскость: xy, xz или yz")
        index = n // 2 if index is None else index
        first, second = np.meshgrid(np.arange(1, n), np.arange(1, n), indexing="ij")
        fixed = np.full(first.shape, index)
        coordinates = {
            "xy": (fixed, first, second),
            "xz": (first, fixed, second),
            "yz": (first, second, fixed),
        }[plane]
        result = np.stack(coordinates, axis=-1).reshape(-1, 3)
    else:
        raise ValueError(f"Неизвестный режим: {mode}")
    return validate_points3d(n, result)


def blank_field3d(problem):
    values = np.full_like(problem.f, np.nan)
    for axis in range(3):
        for end in (0, -1):
            face = [slice(None)] * 3
            face[axis] = end
            values[tuple(face)] = problem.boundary[tuple(face)]
    return values


def laplacian3d(values, h):
    return (
        values[2:, 1:-1, 1:-1]
        + values[:-2, 1:-1, 1:-1]
        + values[1:-1, 2:, 1:-1]
        + values[1:-1, :-2, 1:-1]
        + values[1:-1, 1:-1, 2:]
        + values[1:-1, 1:-1, :-2]
        - 6 * values[1:-1, 1:-1, 1:-1]
    ) / h**2


def discrete_system3d(problem):
    """Независимая система A u = b для A = -h² Lh, с шестью гранями в b."""
    if problem.n > DIRECT_MAX_N:
        raise ValueError(
            f"Прямой 3D-эталон ограничен n <= {DIRECT_MAX_N}; используйте точное решение"
        )
    size = problem.n - 1
    line = diags(
        [-np.ones(size - 1), 2 * np.ones(size), -np.ones(size - 1)],
        [-1, 0, 1],
        format="csr",
    )
    matrix = kronsum(line, kronsum(line, line, format="csr"), format="csr")
    rhs = -problem.h**2 * problem.f[1:-1, 1:-1, 1:-1].copy()
    for axis in range(3):
        for end in (0, -1):
            interior_face = [slice(None)] * 3
            boundary_face = [slice(1, -1)] * 3
            interior_face[axis] = end
            boundary_face[axis] = end
            rhs[tuple(interior_face)] += problem.boundary[tuple(boundary_face)]
    return matrix, rhs.ravel()


def reference_solution3d(problem):
    matrix, rhs = discrete_system3d(problem)
    values = blank_field3d(problem)
    values[1:-1, 1:-1, 1:-1] = spsolve(matrix, rhs).reshape((problem.n - 1,) * 3)
    return values
