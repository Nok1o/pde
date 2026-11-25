"""Problem data for issue #24; no time or nonlinear solver is implemented here.

Coordinates are (x, y, z), arrays [z, y, x]. Boundary kind 3 means
k(u) * outward_normal_derivative(u) + beta*u = r, not a bare derivative.
"""

from dataclasses import dataclass
from types import MappingProxyType
from typing import Callable, Literal, Mapping

import numpy as np

from .boundary3d import FACES, face_indices
from .problem3d import validate_n

Field3D = Callable[[np.ndarray, np.ndarray, np.ndarray, float], object]


def _finite_field(value, shape):
    values = np.broadcast_to(np.asarray(value, dtype=float), shape).copy()
    if not np.isfinite(values).all():
        raise ValueError("Нужны конечные значения поля")
    return values


def _time(t):
    if not np.isfinite(t) or t < 0:
        raise ValueError("Время должно быть конечным и неотрицательным")


@dataclass(frozen=True)
class HeatFace3D:
    kind: Literal[1, 3]
    value: Field3D
    beta: float = 0.0

    def __post_init__(self):
        if isinstance(self.kind, (bool, np.bool_)) or self.kind not in (1, 3):
            raise ValueError("Начальный контракт поддерживает условия 1 и 3 рода")
        if not callable(self.value):
            raise ValueError("Граничное значение должно быть функцией (x,y,z,t)")
        if not np.isfinite(self.beta) or self.beta < 0:
            raise ValueError("beta должна быть конечной и неотрицательной")
        if (self.kind == 1 and self.beta != 0) or (self.kind == 3 and self.beta == 0):
            raise ValueError("Для 1 рода beta=0, для 3 рода beta>0")


@dataclass(frozen=True)
class HeatProblem3D:
    name: str
    n: int
    alpha: float
    initial: Field3D
    source: Field3D
    faces: Mapping[str, HeatFace3D]
    exact: Field3D | None = None

    def __post_init__(self):
        object.__setattr__(self, "n", validate_n(self.n))
        if not np.isfinite(self.alpha) or self.alpha < 0:
            raise ValueError("alpha должна быть конечной и неотрицательной")
        if not callable(self.initial) or not callable(self.source):
            raise ValueError("Начальные данные и источник должны быть функциями")
        if self.exact is not None and not callable(self.exact):
            raise ValueError("Эталон должен быть функцией или None")
        if set(self.faces) != set(FACES) or not all(
            isinstance(face, HeatFace3D) for face in self.faces.values()
        ):
            raise ValueError("Нужны условия для x0,x1,y0,y1,z0,z1")
        object.__setattr__(self, "faces", MappingProxyType(dict(self.faces)))
        # Reject incompatible initial/essential data before a solver allocates history.
        initial = self.initial_values()
        for face, index in face_indices().items():
            if self.faces[face].kind == 1 and not np.allclose(
                initial[index], self.face_values(face, 0.0), rtol=1e-12, atol=1e-12
            ):
                raise ValueError("Начальное поле несовместимо с условием 1 рода")

    @property
    def shape(self):
        return (self.n + 1,) * 3

    @property
    def h(self):
        return 1.0 / self.n

    def coordinates(self):
        axis = np.linspace(0.0, 1.0, self.n + 1)
        return axis[None, None, :], axis[None, :, None], axis[:, None, None]

    def evaluate(self, function: Field3D, t):
        _time(t)
        return _finite_field(function(*self.coordinates(), t), self.shape)

    def initial_values(self):
        return self.evaluate(self.initial, 0.0)

    def face_values(self, face, t):
        _time(t)
        index = face_indices()[face]
        coordinates = tuple(
            np.broadcast_to(coordinate, self.shape)[index]
            for coordinate in self.coordinates()
        )
        return _finite_field(self.faces[face].value(*coordinates, t), (self.n + 1,) * 2)

    def conductivity(self, values):
        with np.errstate(over="ignore", invalid="ignore"):
            result = 1 + self.alpha * np.asarray(values, dtype=float) ** 2
        if not np.isfinite(result).all() or np.any(result <= 0):
            raise ValueError("Проводимость должна быть конечной и положительной")
        return result


def _manufactured_fields(kind, x, y, z, t):
    """Return u, u_t, gradient in xyz order, and Laplacian for test data."""
    if kind == "sine":
        sx, sy, sz = np.sin(np.pi * x), np.sin(np.pi * y), np.sin(np.pi * z)
        amplitude = np.exp(-t)
        s = amplitude * sx * sy * sz
        gradient = (
            amplitude * np.pi * np.cos(np.pi * x) * sy * sz,
            amplitude * np.pi * sx * np.cos(np.pi * y) * sz,
            amplitude * np.pi * sx * sy * np.cos(np.pi * z),
        )
        return 1 + s, -s, gradient, -3 * np.pi**2 * s
    if kind == "polynomial":
        polynomial = x + 2 * y + 3 * z + x * y + x * z + y * z + x * x + y * y + z * z
        gradient = (
            (1 + 2 * x + y + z) / 4,
            (2 + x + 2 * y + z) / 4,
            (3 + x + y + 2 * z) / 4,
        )
        return 1 + t + polynomial / 4, 1.0, gradient, 1.5
    raise ValueError("Проверочные случаи: sine, polynomial")


def make_heat_problem3d(kind="sine", n=8, alpha=0.5, kinds=None, beta=1.0):
    """Manufacture fixed F and face data from u*, never from a solver iterate."""
    if kind not in ("sine", "polynomial"):
        raise ValueError("Проверочные случаи: sine, polynomial")
    kinds = dict.fromkeys(FACES, 1) if kinds is None else dict(kinds)
    if set(kinds) != set(FACES):
        raise ValueError("Укажите тип всех шести граней")

    def exact(x, y, z, t):
        return _manufactured_fields(kind, x, y, z, t)[0]

    def source(x, y, z, t):
        u, ut, gradient, laplacian = _manufactured_fields(kind, x, y, z, t)
        divergence = (1 + alpha * u * u) * laplacian + 2 * alpha * u * sum(
            g * g for g in gradient
        )
        return ut - divergence

    def boundary_data(face):
        component = "xyz".index(face[0])
        sign = -1 if face[1] == "0" else 1

        def value(x, y, z, t):
            u, _, gradient, _ = _manufactured_fields(kind, x, y, z, t)
            return (1 + alpha * u * u) * sign * gradient[component] + beta * u

        return value

    faces = {
        face: (
            HeatFace3D(kinds[face], exact)
            if kinds[face] == 1
            else HeatFace3D(kinds[face], boundary_data(face), beta)
        )
        for face in FACES
    }
    return HeatProblem3D(kind, n, alpha, exact, source, faces, exact)
