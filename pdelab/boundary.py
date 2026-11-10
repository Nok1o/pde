"""Nodal finite volumes for Δu=f with Dirichlet, Neumann and Robin sides.

Neumann data is the OUTWARD derivative, not the physical flux -du/dn.
Half cells on sides and quarter cells at corners keep all neighbor weights
positive. See docs/boundaries.md for the one-dimensional derivation.
"""

from dataclasses import dataclass
from types import MappingProxyType

import numpy as np
from scipy.sparse import bmat, coo_matrix, csr_matrix, diags
from scipy.sparse.linalg import spsolve

from .operator import FrozenOperator

SIDES = ("left", "right", "bottom", "top")
KINDS = ("dirichlet", "neumann", "robin")
PRESETS_BOUNDARY = {"polynomial": "Полином", "smooth": "Косинусы и линейное поле"}


def _frozen(values):
    array = np.ascontiguousarray(values)
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


def _field(values, shape, name):
    try:
        array = np.broadcast_to(np.asarray(values, dtype=float), shape)
    except ValueError as exc:
        raise ValueError(f"{name}: ожидается скаляр или массив формы {shape}") from exc
    if not np.isfinite(array).all():
        raise ValueError(f"{name}: нужны конечные значения")
    return _frozen(array)


@dataclass(frozen=True)
class BoundaryCondition:
    kind: str
    value: object
    kappa: object = 0.0

    def sampled(self, size):
        if self.kind not in KINDS:
            raise ValueError(f"Неизвестное граничное условие: {self.kind}")
        value = _field(self.value, (size,), "Граничные данные")
        kappa = _field(self.kappa, (size,), "κ")
        if np.any(kappa < 0) or (self.kind != "robin" and np.any(kappa != 0)):
            raise ValueError("κ ≥ 0 задаётся только для условия Робина")
        return BoundaryCondition(self.kind, value, kappa)

    @classmethod
    def from_flux(cls, outward_flux, conductivity=1.0):
        """Physical outward flux q=-k*du/dn -> Neumann derivative -q/k."""
        k = np.asarray(conductivity, dtype=float)
        if not np.isfinite(k).all() or np.any(k <= 0):
            raise ValueError(
                "Проводимость для перевода потока должна быть положительной"
            )
        return cls("neumann", -np.asarray(outward_flux, dtype=float) / k)


def side_indices(n):
    return {
        "left": (slice(None), 0),
        "right": (slice(None), n),
        "bottom": (0, slice(None)),
        "top": (n, slice(None)),
    }


@dataclass(frozen=True)
class BoundaryProblem:
    n: int
    f: object
    sides: dict
    exact: object = None
    name: str = "Смешанная граница"

    def __post_init__(self):
        if isinstance(self.n, bool) or int(self.n) != self.n or not 4 <= self.n <= 256:
            raise ValueError("n должно быть целым от 4 до 256")
        object.__setattr__(self, "n", int(self.n))
        if set(self.sides) != set(SIDES):
            raise ValueError("Нужно задать left, right, bottom и top")
        sides = {name: self.sides[name].sampled(self.n + 1) for name in SIDES}
        object.__setattr__(self, "sides", MappingProxyType(sides))
        shape = (self.n + 1, self.n + 1)
        object.__setattr__(self, "f", _field(self.f, shape, "f"))
        if self.exact is not None:
            object.__setattr__(self, "exact", _field(self.exact, shape, "u*"))

    @property
    def axis(self):
        return np.linspace(0, 1, self.n + 1)

    @classmethod
    def from_problem(cls, problem):
        """Preserve legacy Dirichlet values, including its corner convention."""
        return cls(
            problem.n,
            problem.f,
            {
                s: BoundaryCondition("dirichlet", problem.boundary[i])
                for s, i in side_indices(problem.n).items()
            },
            problem.exact,
            problem.name,
        )


class BoundaryOperator:
    """Shared positive stencil for direct solution, directional sweeps and MC.

    Neighbor order agrees with FrozenOperator in issue #8: y+, y-, x+, x-.
    This MVP has k=1. Neumann/Robin nodes remain unknowns.
    """

    def __init__(self, problem):
        self.problem = problem
        n = problem.n
        size = n + 1
        shape = (size, size)
        volume_axis = np.ones(size)
        volume_axis[[0, -1]] = 0.5
        self.volumes = _frozen(np.outer(volume_axis, volume_axis) / n**2)
        # Reuse issue #8's face coefficients. Half boundary cells change only
        # the volume-to-face ratio; the interior stencil remains identical.
        core = FrozenOperator.from_nodes(np.ones(shape), -problem.f, 0.0)
        weights = np.zeros((*shape, 4))
        weights[:-1, :, 0] = core.wy
        weights[1:, :, 1] = core.wy
        weights[:, :-1, 2] = core.wx
        weights[:, 1:, 3] = core.wx
        weights[0, :, 0] *= 2
        weights[-1, :, 1] *= 2
        weights[:, 0, 2] *= 2
        weights[:, -1, 3] *= 2
        fixed = np.zeros(shape, dtype=bool)
        values = np.zeros(shape)
        reaction_x = np.zeros(shape)
        reaction_y = np.zeros(shape)
        boundary_load = np.zeros(shape)
        for side, indices in side_indices(n).items():
            condition = problem.sides[side]
            if condition.kind == "dirichlet":
                overlap = fixed[indices]
                if not np.allclose(
                    values[indices][overlap],
                    condition.value[overlap],
                    rtol=1e-12,
                    atol=1e-12,
                ):
                    raise ValueError("В углу заданы несовместимые значения Дирихле")
                values[indices] = condition.value
                fixed[indices] = True
            else:
                boundary_load[indices] += 2 * n * condition.value
                reaction = reaction_x if side in ("left", "right") else reaction_y
                reaction[indices] += 2 * n * condition.kappa
        # Dirichlet takes precedence in a corner adjoining a derivative side.
        boundary_load[fixed] = 0
        reaction_x[fixed] = 0
        reaction_y[fixed] = 0
        self.weights = _frozen(weights)
        self.fixed = _frozen(fixed)
        self.boundary = _frozen(values)
        self.reaction_x = _frozen(reaction_x)
        self.reaction_y = _frozen(reaction_y)
        self.reaction = _frozen(reaction_x + reaction_y)
        self.boundary_load = _frozen(boundary_load)
        self.diagonal = _frozen(weights.sum(axis=-1) + self.reaction)
        self.b = _frozen(-problem.f + boundary_load)
        if not np.isfinite(self.diagonal).all() or not np.isfinite(self.b).all():
            raise ValueError("Слишком большие коэффициенты граничной задачи")
        self.pure_neumann = not fixed.any() and not self.reaction.any()

    def normalize(self, values):
        values = np.array(values, dtype=float, copy=True)
        if self.pure_neumann:
            values -= np.sum(self.volumes * values) / self.volumes.sum()
        return values

    def check_compatibility(self):
        if not self.pure_neumann:
            return
        terms = self.volumes * self.b
        imbalance = float(np.sum(terms))
        scale = float(np.sum(np.abs(terms)))
        if abs(imbalance) > 1e-12 * scale:
            raise ValueError(
                "Несовместимая задача Неймана: дискретный баланс источника и "
                f"внешних производных нарушен (разность {imbalance:.6g})."
            )

    def apply(self, values):
        u = np.asarray(values, dtype=float)
        if u.shape != self.b.shape or not np.isfinite(u).all():
            raise ValueError("Ожидается конечное поле формы (n+1, n+1)")
        out = self.diagonal * u
        out[:-1, :] -= self.weights[:-1, :, 0] * u[1:, :]
        out[1:, :] -= self.weights[1:, :, 1] * u[:-1, :]
        out[:, :-1] -= self.weights[:, :-1, 2] * u[:, 1:]
        out[:, 1:] -= self.weights[:, 1:, 3] * u[:, :-1]
        out[self.fixed] = u[self.fixed]
        return out

    def rhs(self):
        return np.where(self.fixed, self.boundary, self.b)

    def assemble(self):
        """Independent sparse assembly on unknown nodes; fixed nodes eliminated."""
        unknown = ~self.fixed
        size = self.b.shape[0]
        indices = np.arange(size * size).reshape(size, size)
        rows = [indices.ravel()]
        columns = [indices.ravel()]
        data = [self.diagonal.ravel()]
        for origin, target, direction in (
            ((slice(None, -1), slice(None)), (slice(1, None), slice(None)), 0),
            ((slice(1, None), slice(None)), (slice(None, -1), slice(None)), 1),
            ((slice(None), slice(None, -1)), (slice(None), slice(1, None)), 2),
            ((slice(None), slice(1, None)), (slice(None), slice(None, -1)), 3),
        ):
            rows.append(indices[origin].ravel())
            columns.append(indices[target].ravel())
            data.append(-self.weights[origin][..., direction].ravel())
        matrix = coo_matrix(
            (np.concatenate(data), (np.concatenate(rows), np.concatenate(columns))),
            shape=(size * size, size * size),
        ).tocsr()
        mask = unknown.ravel()
        rhs = (
            self.b.ravel()[mask] - matrix[mask][:, ~mask] @ self.boundary.ravel()[~mask]
        )
        return matrix[mask][:, mask], rhs

    def solve(self):
        self.check_compatibility()
        matrix, rhs = self.assemble()
        unknown = ~self.fixed
        if self.pure_neumann:
            mass = self.volumes.ravel()
            constraint = csr_matrix(mass[:, None])
            augmented = bmat(
                [[diags(mass) @ matrix, constraint], [constraint.T, None]], format="csr"
            )
            values = spsolve(augmented, np.append(mass * rhs, 0.0))[:-1]
        else:
            values = spsolve(matrix, rhs)
        result = self.boundary.copy()
        result[unknown] = values
        return self.normalize(result)

    def mc_coefficients(self):
        """u=P*u+reward. Fixed nodes pay g, then terminate with probability 1."""
        probabilities = self.weights / self.diagonal[..., None]
        reward = self.b / self.diagonal
        killing = self.reaction / self.diagonal
        probabilities[self.fixed] = 0
        reward[self.fixed] = self.boundary[self.fixed]
        killing[self.fixed] = 1
        return probabilities, reward, killing


def make_boundary_problem(kind="polynomial", n=24, kinds=None, kappas=None):
    if isinstance(n, bool) or not np.isfinite(n) or int(n) != n or not 4 <= n <= 256:
        raise ValueError("n должно быть целым от 4 до 256")
    n = int(n)
    kinds = (
        dict(zip(SIDES, ("robin", "dirichlet", "dirichlet", "dirichlet")))
        if kinds is None
        else kinds
    )
    if set(kinds) != set(SIDES):
        raise ValueError("Нужно задать тип каждой из четырёх сторон")
    kappas = {} if kappas is None else kappas
    x, y = np.meshgrid(np.linspace(0, 1, n + 1), np.linspace(0, 1, n + 1))
    if kind == "polynomial":
        exact = 1 + x + 2 * y + x**2 + y**2
        f = np.full_like(x, 4.0)
        ux, uy = 1 + 2 * x, 2 + 2 * y
    elif kind == "smooth":
        exact = 2 + x + 2 * y + np.cos(2 * np.pi * x) + 0.5 * np.cos(2 * np.pi * y)
        f = -4 * np.pi**2 * np.cos(2 * np.pi * x) - 2 * np.pi**2 * np.cos(2 * np.pi * y)
        ux, uy = 1 - 2 * np.pi * np.sin(2 * np.pi * x), 2 - np.pi * np.sin(
            2 * np.pi * y
        )
    else:
        raise ValueError(f"Неизвестный пример: {kind}")
    normals = {
        "left": -ux[:, 0],
        "right": ux[:, -1],
        "bottom": -uy[0, :],
        "top": uy[-1, :],
    }
    sides = {}
    for side, index in side_indices(n).items():
        boundary_kind = kinds[side]
        kappa = kappas.get(side, 1.0) if boundary_kind == "robin" else 0.0
        value = (
            exact[index]
            if boundary_kind == "dirichlet"
            else normals[side] + kappa * exact[index]
        )
        sides[side] = BoundaryCondition(boundary_kind, value, kappa)
    return BoundaryProblem(n, f, sides, exact, PRESETS_BOUNDARY[kind])
