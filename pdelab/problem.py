from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.sparse import diags, kronsum
from scipy.sparse.linalg import spsolve

PRESETS = {
    "hot_top": "Нагретый верхний край",
    "polynomial": "Пуассон: полином",
    "sine": "Пуассон: синус",
    "affine": "Лаплас: линейное поле",
}


@dataclass
class Problem:
    name: str
    n: int
    f: np.ndarray
    boundary: np.ndarray
    exact: np.ndarray | None = None

    def __post_init__(self):
        if isinstance(self.n, bool) or int(self.n) != self.n or not 4 <= self.n <= 512:
            raise ValueError("Число интервалов n должно быть целым от 4 до 512")
        self.n = int(self.n)
        for key in ("f", "boundary", "exact"):
            a = getattr(self, key)
            if a is not None:
                a = np.asarray(a, dtype=float).copy()
                if a.shape != (self.n + 1, self.n + 1) or not np.isfinite(a).all():
                    raise ValueError(
                        f"{key}: ожидается конечный массив формы {(self.n+1, self.n+1)}"
                    )
                setattr(self, key, a)

    @property
    def h(self):
        return 1.0 / self.n

    @property
    def axis(self):
        return np.linspace(0, 1, self.n + 1)

    @classmethod
    def from_functions(
        cls,
        n: int,
        source: Callable,
        boundary: Callable,
        exact: Callable | None = None,
        name="Своя задача",
    ):
        X, Y = np.meshgrid(np.linspace(0, 1, n + 1), np.linspace(0, 1, n + 1))

        def field(fn):
            return np.broadcast_to(np.asarray(fn(X, Y), dtype=float), X.shape).copy()

        return cls(
            name,
            n,
            field(source),
            field(boundary),
            None if exact is None else field(exact),
        )


def make_problem(
    kind="hot_top", n=24, top=100.0, bottom=0.0, left=0.0, right=0.0, source=0.0
):
    if kind not in PRESETS:
        raise ValueError(f"Неизвестная задача: {kind}")
    if kind == "hot_top":
        problem = Problem.from_functions(
            n, lambda x, y: source, lambda x, y: 0.0, name=PRESETS[kind]
        )
        problem.boundary[0, :] = bottom
        problem.boundary[-1, :] = top
        problem.boundary[:, 0] = left
        problem.boundary[:, -1] = right
        problem.boundary[0, 0] = (bottom + left) / 2
        problem.boundary[0, -1] = (bottom + right) / 2
        problem.boundary[-1, 0] = (top + left) / 2
        problem.boundary[-1, -1] = (top + right) / 2
        if not np.isfinite(problem.boundary).all():
            raise ValueError("Граничные значения должны быть конечными")
        return problem
    if kind == "polynomial":

        def exact_solution(x, y):
            return 1000 * x * (1 - x) * y * (1 - y)

        return Problem.from_functions(
            n,
            source=lambda x, y: -2000 * (x * (1 - x) + y * (1 - y)),
            boundary=exact_solution,
            exact=exact_solution,
            name=PRESETS[kind],
        )
    if kind == "sine":

        def exact_solution(x, y):
            return 100 * np.sin(np.pi * x) * np.sin(np.pi * y)

        return Problem.from_functions(
            n,
            source=lambda x, y: -2 * np.pi**2 * exact_solution(x, y),
            boundary=exact_solution,
            exact=exact_solution,
            name=PRESETS[kind],
        )

    def exact_solution(x, y):
        return 10 + 20 * x + 30 * y

    return Problem.from_functions(
        n,
        source=lambda x, y: 0.0,
        boundary=exact_solution,
        exact=exact_solution,
        name=PRESETS[kind],
    )


def select_points(n, mode="point", point=None):
    if mode == "point":
        x, y = point or (n // 2, n // 2)
        result = np.array([[y, x]], dtype=np.int64)
    elif mode == "patch":
        center = n // 2
        result = np.array(
            [
                (y, x)
                for y in range(max(1, center - 1), min(n, center + 2))
                for x in range(max(1, center - 1), min(n, center + 2))
            ],
            dtype=np.int64,
        )
    elif mode == "field":
        result = np.array(
            [(y, x) for y in range(1, n) for x in range(1, n)], dtype=np.int64
        )
    else:
        raise ValueError(f"Неизвестный режим: {mode}")
    return validate_points(n, result)


def validate_points(n, points):
    indices = np.asarray(points)
    if (
        indices.ndim != 2
        or indices.shape[1] != 2
        or len(indices) == 0
        or not np.isfinite(indices).all()
        or np.any(indices != indices.astype(np.int64))
    ):
        raise ValueError("Точки задаются непустым списком целых индексов (y,x)")
    indices = np.ascontiguousarray(indices, dtype=np.int64)
    if (
        np.any(indices < 1)
        or np.any(indices >= n)
        or len(np.unique(indices, axis=0)) != len(indices)
    ):
        raise ValueError("Нужны уникальные внутренние узлы сетки")
    return indices


def blank_field(p):
    u = np.full_like(p.f, np.nan)
    u[0, :] = p.boundary[0, :]
    u[-1, :] = p.boundary[-1, :]
    u[:, 0] = p.boundary[:, 0]
    u[:, -1] = p.boundary[:, -1]
    return u


def laplacian(u, h):
    return (
        u[2:, 1:-1] + u[:-2, 1:-1] + u[1:-1, 2:] + u[1:-1, :-2] - 4 * u[1:-1, 1:-1]
    ) / h**2


def reference_solution(p):
    """Independent sparse direct solution of exactly the same 5-point system."""
    interior_size = p.n - 1
    line_matrix = diags(
        [
            -np.ones(interior_size - 1),
            2 * np.ones(interior_size),
            -np.ones(interior_size - 1),
        ],
        [-1, 0, 1],
        format="csr",
    )
    system_matrix = kronsum(line_matrix, line_matrix, format="csr")
    rhs = -p.h**2 * p.f[1:-1, 1:-1].copy()
    rhs[0, :] += p.boundary[0, 1:-1]
    rhs[-1, :] += p.boundary[-1, 1:-1]
    rhs[:, 0] += p.boundary[1:-1, 0]
    rhs[:, -1] += p.boundary[1:-1, -1]
    u = blank_field(p)
    u[1:-1, 1:-1] = spsolve(system_matrix, rhs.ravel()).reshape(
        interior_size, interior_size
    )
    return u
