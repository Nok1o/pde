"""Mixed boundary data on six faces; coordinates (x,y,z), arrays [z,y,x]."""

from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from .boundary import BoundaryCondition, BoundaryOperator, _field
from .problem3d import DIRECT_MAX_N, validate_n, validate_points3d

FACE_AXES = {
    "x0": (2, 0),
    "x1": (2, -1),
    "y0": (1, 0),
    "y1": (1, -1),
    "z0": (0, 0),
    "z1": (0, -1),
}
FACES = tuple(FACE_AXES)
PRESETS_BOUNDARY3D = {
    "polynomial": "Полином в кубе",
    "smooth": "Косинусы и линейное поле",
}


def face_indices():
    result = {}
    for name, (axis, end) in FACE_AXES.items():
        index = [slice(None)] * 3
        index[axis] = end
        result[name] = tuple(index)
    return result


@dataclass(frozen=True)
class BoundaryProblem3D:
    n: int
    f: object
    faces: dict
    exact: object = None
    name: str = "Смешанные границы куба"

    def __post_init__(self):
        n = validate_n(self.n)
        object.__setattr__(self, "n", n)
        if set(self.faces) != set(FACES):
            raise ValueError("Нужны шесть граней: x0, x1, y0, y1, z0, z1")
        faces = {name: self.faces[name].sampled((n + 1, n + 1)) for name in FACES}
        object.__setattr__(self, "faces", MappingProxyType(faces))
        object.__setattr__(self, "f", _field(self.f, (n + 1,) * 3, "f"))
        if self.exact is not None:
            object.__setattr__(self, "exact", _field(self.exact, (n + 1,) * 3, "u*"))

    @property
    def axis(self):
        return np.linspace(0, 1, self.n + 1)

    @property
    def sides(self):
        """Common boundary-operator interface, also used by the 2D model."""
        return self.faces

    @property
    def pure_neumann(self):
        return all(
            c.kind != "dirichlet" and not np.any(c.kappa) for c in self.faces.values()
        )

    @classmethod
    def from_problem(cls, problem):
        return cls(
            problem.n,
            problem.f,
            {
                name: BoundaryCondition("dirichlet", problem.boundary[index])
                for name, index in face_indices().items()
            },
            problem.exact,
            problem.name,
        )


class BoundaryOperator3D(BoundaryOperator):
    """Same finite-volume rules, with half/quarter/eighth boundary cells."""

    face_axes = FACE_AXES

    def assemble(self):
        if self.problem.n > DIRECT_MAX_N:
            raise ValueError(
                f"Прямой 3D-эталон ограничен n ≤ {DIRECT_MAX_N}; используйте аналитический пример"
            )
        return super().assemble()


def boundary_workspace_bytes3d(n):
    """Reserve 64 dense volumes for operator, factors and work arrays; not sparse LU."""
    return 64 * (validate_n(n) + 1) ** 3 * np.dtype(float).itemsize


def make_boundary_problem3d(kind="polynomial", n=16, kinds=None, kappas=None):
    n = validate_n(n)
    if kind not in PRESETS_BOUNDARY3D:
        raise ValueError(f"Неизвестный 3D-пример: {kind}")
    kinds = (
        {name: "robin" if name == "x0" else "dirichlet" for name in FACES}
        if kinds is None
        else kinds
    )
    if set(kinds) != set(FACES):
        raise ValueError("Нужно задать тип каждой из шести граней")
    kappas = {} if kappas is None else kappas
    axis = np.linspace(0, 1, n + 1)
    x, y, z = axis[None, None, :], axis[None, :, None], axis[:, None, None]
    if kind == "polynomial":
        exact = 1 + x + 2 * y + 3 * z + x * x + y * y + z * z
        f = 6.0
        gradients = (3 + 2 * z, 2 + 2 * y, 1 + 2 * x)
    else:
        cosine = (
            np.cos(2 * np.pi * x)
            + 0.5 * np.cos(2 * np.pi * y)
            + 0.25 * np.cos(2 * np.pi * z)
        )
        exact = 3 + x + 2 * y + 3 * z + cosine
        f = -4 * np.pi**2 * cosine
        gradients = (
            3 - 0.5 * np.pi * np.sin(2 * np.pi * z),
            2 - np.pi * np.sin(2 * np.pi * y),
            1 - 2 * np.pi * np.sin(2 * np.pi * x),
        )
    gradients = [np.broadcast_to(g, exact.shape) for g in gradients]
    faces = {}
    for name, index in face_indices().items():
        axis_index, end = FACE_AXES[name]
        kind_on_face = kinds[name]
        kappa = kappas.get(name, 1.0) if kind_on_face == "robin" else 0.0
        normal = gradients[axis_index][index] * (-1 if end == 0 else 1)
        value = (
            exact[index]
            if kind_on_face == "dirichlet"
            else normal + kappa * exact[index]
        )
        faces[name] = BoundaryCondition(kind_on_face, value, kappa)
    return BoundaryProblem3D(n, f, faces, exact, PRESETS_BOUNDARY3D[kind])


def select_boundary_points3d(n, mode="point", point=None, plane="xy", index=None):
    n = validate_n(n)
    if mode in ("point", "patch"):
        xyz = (n // 2,) * 3 if point is None else point
        coordinates = np.asarray([xyz])
        if coordinates.shape != (1, 3):
            raise ValueError("point: нужны координаты (x,y,z)")
        center = validate_points3d(n, coordinates[:, ::-1])[0]
        if mode == "point":
            return center[None, :].copy()
        axes = [np.arange(max(0, c - 1), min(n, c + 1) + 1) for c in center]
        points = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    elif mode == "field":
        points = np.indices((n + 1,) * 3).reshape(3, -1).T
    elif mode == "slice":
        if plane not in ("xy", "xz", "yz"):
            raise ValueError("Плоскость: xy, xz или yz")
        index = n // 2 if index is None else index
        first, second = np.meshgrid(np.arange(n + 1), np.arange(n + 1), indexing="ij")
        fixed = np.full(first.shape, index)
        coordinates = {
            "xy": (fixed, first, second),
            "xz": (first, fixed, second),
            "yz": (first, second, fixed),
        }[plane]
        points = np.stack(coordinates, axis=-1).reshape(-1, 3)
    else:
        raise ValueError("Запрос: point, patch, slice или field")
    return validate_points3d(n, points)
