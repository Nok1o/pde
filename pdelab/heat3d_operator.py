"""Conservative frozen heat layers on a cube with essential and Robin faces.

The public arrays follow ``heat3d_contract``: all rows are present, and known
neighbors have already been eliminated from the right-hand side.
"""

from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix

from .boundary3d import FACE_AXES, face_indices
from .heat3d_contract import NEIGHBOR_OFFSETS_ZYX
from .problem3d import DIRECT_MAX_N


class UnsupportedHeatBoundaryError(ValueError):
    """Compatibility exception for callers of the original heat implementation."""


class IncompatibleHeatBoundaryError(ValueError):
    """Finite essential values disagree where two or more faces intersect."""


def _readonly(values, dtype=float):
    """Own immutable storage; even setflags(write=True) cannot expose a writer."""
    array = np.ascontiguousarray(values, dtype=dtype)
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


def _field(values, shape, name):
    array = np.asarray(values, dtype=float)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{name}: требуется конечное поле формы {shape}")
    return array


def _neighbor_slices(direction):
    offset = NEIGHBOR_OFFSETS_ZYX[direction]
    axis = next(axis for axis, step in enumerate(offset) if step)
    origin = [slice(None)] * 3
    neighbor = [slice(None)] * 3
    if offset[axis] == 1:
        origin[axis], neighbor[axis] = slice(None, -1), slice(1, None)
    else:
        origin[axis], neighbor[axis] = slice(1, None), slice(None, -1)
    return tuple(origin), tuple(neighbor)


def dirichlet_data3d(problem, t):
    """Sample kind 1 data; essential faces take priority at mixed intersections."""
    if not np.isfinite(t) or t < 0:
        raise ValueError("Время должно быть конечным и неотрицательным")
    fixed = np.zeros(problem.shape, dtype=bool)
    values = np.zeros(problem.shape)
    for face, index in face_indices().items():
        if problem.faces[face].kind != 1:
            continue
        sampled = problem.face_values(face, t)
        overlap = fixed[index]
        if not np.allclose(
            values[index][overlap], sampled[overlap], rtol=1e-12, atol=1e-12
        ):
            raise IncompatibleHeatBoundaryError(
                "На ребре или в углу несовместимы условия 1 рода"
            )
        values[index] = sampled
        fixed[index] = True
    return fixed, values


@dataclass(frozen=True)
class HeatOperator3D:
    """Immutable full-grid snapshot, with no remaining links to known nodes."""

    diagonal: np.ndarray
    weights: np.ndarray
    rhs: np.ndarray
    fixed: np.ndarray
    fixed_values: np.ndarray
    volumes: np.ndarray

    def __post_init__(self):
        shape = np.shape(self.rhs)
        if len(shape) != 3 or len(set(shape)) != 1 or shape[0] < 3:
            raise ValueError("Оператор требует кубический массив с внутренними узлами")
        for name in ("diagonal", "weights", "rhs", "fixed_values", "volumes"):
            expected = (*shape, 6) if name == "weights" else shape
            array = _field(getattr(self, name), expected, name)
            object.__setattr__(self, name, _readonly(array))
        fixed = np.asarray(self.fixed)
        if fixed.shape != shape or fixed.dtype != np.dtype(bool):
            raise ValueError("fixed: требуется булева маска формы поля")
        object.__setattr__(self, "fixed", _readonly(fixed, bool))
        if np.any(self.diagonal <= 0) or np.any(self.volumes <= 0):
            raise ValueError(
                "Диагональ и контрольные объёмы должны быть положительными"
            )
        if np.any(self.weights < 0):
            raise ValueError("Веса соседей должны быть неотрицательными")
        with np.errstate(over="ignore", invalid="ignore"):
            weight_sum = self.weights.sum(axis=-1)
        if not np.isfinite(weight_sum).all() or np.any(self.diagonal < weight_sum):
            raise ValueError("Оператор должен иметь диагональное преобладание")
        if (
            np.any(self.diagonal[self.fixed] != 1)
            or np.any(self.weights[self.fixed] != 0)
            or not np.array_equal(self.rhs[self.fixed], self.fixed_values[self.fixed])
            or np.any(self.fixed_values[~self.fixed] != 0)
        ):
            raise ValueError("Фиксированные строки должны быть единичными с rhs=g")
        for direction, offset in enumerate(NEIGHBOR_OFFSETS_ZYX):
            axis = next(axis for axis, step in enumerate(offset) if step)
            outside = [slice(None)] * 3
            outside[axis] = -1 if offset[axis] > 0 else 0
            if np.any(self.weights[tuple(outside) + (direction,)] != 0):
                raise ValueError("Связи за пределы куба должны быть нулевыми")
            origin, neighbor = _neighbor_slices(direction)
            if np.any(self.weights[origin + (direction,)][self.fixed[neighbor]] != 0):
                raise ValueError("Связи к известным узлам должны быть исключены")

    @property
    def shape(self):
        return self.rhs.shape

    @property
    def n(self):
        return self.shape[0] - 1

    def apply(self, values):
        values = _field(values, self.shape, "values")
        with np.errstate(over="ignore", invalid="ignore"):
            result = self.diagonal * values
            for direction in range(6):
                origin, neighbor = _neighbor_slices(direction)
                result[origin] -= self.weights[origin + (direction,)] * values[neighbor]
        return result

    def assemble(self):
        """Return the same full-grid system; refuse large grids before allocation."""
        if self.n > DIRECT_MAX_N:
            raise ValueError(f"Прямая 3D-сборка ограничена n ≤ {DIRECT_MAX_N}")
        indices = np.arange(self.rhs.size).reshape(self.shape)
        rows, columns, data = (
            [indices.ravel()],
            [indices.ravel()],
            [self.diagonal.ravel()],
        )
        for direction in range(6):
            origin, neighbor = _neighbor_slices(direction)
            weights = self.weights[origin + (direction,)]
            present = weights != 0
            rows.append(indices[origin][present])
            columns.append(indices[neighbor][present])
            data.append(-weights[present])
        matrix = coo_matrix(
            (np.concatenate(data), (np.concatenate(rows), np.concatenate(columns))),
            shape=(self.rhs.size, self.rhs.size),
        ).tocsr()
        return matrix, self.rhs.ravel().copy()

    def with_rhs(self, rhs):
        """Replace an already adjusted load, including correction boundary rows."""
        rhs = _field(rhs, self.shape, "rhs")
        values = np.zeros(self.shape)
        values[self.fixed] = rhs[self.fixed]
        result = object.__new__(type(self))
        for name in ("diagonal", "weights", "fixed", "volumes"):
            object.__setattr__(result, name, getattr(self, name))
        object.__setattr__(result, "rhs", _readonly(rhs))
        object.__setattr__(result, "fixed_values", _readonly(values))
        return result


def freeze_heat3d(problem, previous, iterate, t, dt, *, conductivity=None):
    """Freeze k, c and cubic loss for one implicit physical-time layer."""
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("Длина слоя dt должна быть конечной и положительной")
    previous = _field(previous, problem.shape, "previous")
    iterate = _field(iterate, problem.shape, "iterate").copy()
    fixed, boundary = dirichlet_data3d(problem, t)
    iterate[fixed] = boundary[fixed]
    if conductivity is None:
        conductivity = problem.conductivity(iterate)
    else:
        conductivity = _field(conductivity, problem.shape, "conductivity")
        if np.any(conductivity <= 0):
            raise ValueError("Пробная проводимость должна быть положительной")
    capacity = problem.capacity(iterate)
    reaction = problem.reaction_coefficient(iterate)
    weights = np.zeros((*problem.shape, 6))
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        for direction in range(0, 6, 2):
            origin, neighbor = _neighbor_slices(direction)
            # Halve before adding, as in the 2D operator, to avoid needless overflow.
            edge = (
                conductivity[origin] / 2 + conductivity[neighbor] / 2
            ) * problem.n**2
            weights[origin + (direction,)] = edge
            weights[neighbor + (direction + 1,)] = edge
        # A boundary node owns a half-width dual cell in the face-normal
        # direction. Only its inward row coefficient doubles; tangential
        # weights keep their usual values, also at edges and corners.
        for face, index in face_indices().items():
            axis, end = FACE_AXES[face]
            inward = 2 * axis + (end != 0)
            weights[index + (inward,)] *= 2
        diagonal = capacity / dt + weights.sum(axis=-1) + reaction
        rhs = capacity * (previous / dt) + problem.evaluate(problem.source, t)
        for face, index in face_indices().items():
            condition = problem.faces[face]
            if condition.kind == 3:
                # k * du/dn = r - beta*u is already a physical flux.
                # Face area / dual volume is 2/h for every incident face.
                diagonal[index] += 2 * problem.n * condition.beta
                rhs[index] += 2 * problem.n * problem.face_values(face, t)
        for direction in range(6):
            origin, neighbor = _neighbor_slices(direction)
            edge = weights[origin + (direction,)]
            rhs[origin] += edge * boundary[neighbor]
            edge[fixed[neighbor]] = 0
        weights[fixed] = 0
        diagonal[fixed] = 1
        rhs[fixed] = boundary[fixed]
    volumes = np.full(problem.shape, problem.h**3)
    for index in face_indices().values():
        volumes[index] *= 0.5
    return HeatOperator3D(diagonal, weights, rhs, fixed, boundary, volumes)
