"""Independent small-grid Newton assembly for physical heat layers in 3D.

All input fields use [z, y, x]; unknowns use C order, so x varies fastest.
The legacy call has interior unknowns; supplying a problem also admits free
Robin boundary nodes. Neither assembly uses the frozen production operator
or substitutes manufactured solutions.
"""

import numpy as np
from scipy.sparse import csr_matrix

from .problem3d import DIRECT_MAX_N, validate_n

_INTERIOR = (slice(1, -1),) * 3


def _scalar(value, name, *, positive=False):
    if (
        isinstance(value, (bool, np.bool_))
        or not np.isscalar(value)
        or np.iscomplexobj(value)
    ):
        raise ValueError(f"{name}: нужен конечный вещественный параметр")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name}: нужен конечный вещественный параметр") from error
    if not np.isfinite(result) or (result <= 0 if positive else result < 0):
        condition = "положительным" if positive else "неотрицательным"
        raise ValueError(f"{name} должен быть конечным и {condition}")
    return result


def _validated_inputs(values, previous, alpha, source, dt):
    # Inspect the dimensions before dtype conversion or stencil allocation.
    shape = np.shape(values)
    if len(shape) != 3 or len(set(shape)) != 1:
        raise ValueError("values: нужен кубический массив [z, y, x]")
    n = validate_n(shape[0] - 1)
    if n > DIRECT_MAX_N:
        raise ValueError(f"Прямой 3D-эталон ограничен n <= {DIRECT_MAX_N}")
    alpha = _scalar(alpha, "alpha")
    dt = _scalar(dt, "dt", positive=True)
    with np.errstate(over="ignore", divide="ignore"):
        inverse_dt = np.float64(1.0) / dt
    if not np.isfinite(inverse_dt):
        raise ValueError("1/dt должен быть конечным")
    fields = []
    for name, field in (("values", values), ("previous", previous), ("source", source)):
        if np.shape(field) != shape or np.iscomplexobj(field):
            raise ValueError(f"{name}: нужен вещественный массив формы {shape}")
        field = np.asarray(field, dtype=float)
        if not np.isfinite(field).all():
            raise ValueError(f"{name}: все значения поля должны быть конечными")
        fields.append(field)
    return n, *fields, alpha, inverse_dt


def _evaluate(values, previous, alpha, source, dt, *, jacobian=False, scale=False):
    n, values, previous, source, alpha, inverse_dt = _validated_inputs(
        values, previous, alpha, source, dt
    )
    center = values[_INTERIOR]
    m = n - 1
    rows, columns, coefficients = [], [], []
    indices = np.arange(m**3).reshape((m,) * 3) if jacobian else None
    diagonal = np.full(center.shape, inverse_dt) if jacobian else None
    with np.errstate(over="ignore", invalid="ignore"):
        # Multiplying by sqrt(alpha) first also handles alpha=0 and large u.
        scaled = np.sqrt(alpha) * values
        conductivity = 1.0 + scaled * scaled
        if not np.isfinite(conductivity).all():
            raise ValueError("Проводимость должна быть конечной")
        center_k = conductivity[_INTERIOR]
        residual = (center - previous[_INTERIOR]) * inverse_dt - source[_INTERIOR]
        rhs = previous[_INTERIOR] * inverse_dt + source[_INTERIOR] if scale else None
        for axis in range(3):
            for sign in (1, -1):
                neighbor_index = list(_INTERIOR)
                neighbor_index[axis] = slice(2, None) if sign == 1 else slice(None, -2)
                neighbor_index = tuple(neighbor_index)
                neighbor = values[neighbor_index]
                # The two halves avoid overflow in k_i + k_j before division.
                weight = (0.5 * center_k + 0.5 * conductivity[neighbor_index]) * n**2
                difference = center - neighbor
                residual += weight * difference
                if jacobian:
                    diagonal += weight + (alpha * center * difference) * n**2
                    derivative = -weight + (alpha * neighbor * difference) * n**2
                    origin = [slice(None)] * 3
                    target = [slice(None)] * 3
                    origin[axis] = slice(None, -1) if sign == 1 else slice(1, None)
                    target[axis] = slice(1, None) if sign == 1 else slice(None, -1)
                    origin, target = tuple(origin), tuple(target)
                    rows.append(indices[origin].ravel())
                    columns.append(indices[target].ravel())
                    coefficients.append(derivative[origin].ravel())
                if scale:
                    boundary = [slice(None)] * 3
                    boundary[axis] = -1 if sign == 1 else 0
                    boundary = tuple(boundary)
                    rhs[boundary] += weight[boundary] * neighbor[boundary]
        if not np.isfinite(residual).all():
            raise ValueError("Нелинейная невязка должна быть конечной")
        matrix = None
        if jacobian:
            rows.append(indices.ravel())
            columns.append(indices.ravel())
            coefficients.append(diagonal.ravel())
            data = np.concatenate(coefficients)
            if not np.isfinite(data).all():
                raise ValueError("Якобиан должен быть конечным")
            matrix = csr_matrix(
                (data, (np.concatenate(rows), np.concatenate(columns))),
                shape=(m**3, m**3),
            )
            if not np.isfinite(matrix.data).all():
                raise ValueError("Якобиан должен быть конечным")
        residual_scale = None
        if scale:
            applied = residual + rhs
            if not np.isfinite(rhs).all() or not np.isfinite(applied).all():
                raise ValueError("Масштаб невязки должен быть конечным")
            residual_scale = float(max(np.max(np.abs(rhs)), np.max(np.abs(applied))))
    return residual, matrix, residual_scale


def _evaluate_boundaries(
    values, previous, alpha, source, dt, *, problem, t, jacobian=False, scale=False
):
    """Assemble half-control-volume rows independently of the production stencil."""
    n, values, previous, source, alpha, inverse_dt = _validated_inputs(
        values, previous, alpha, source, dt
    )
    t = _scalar(t, "t")
    if problem.n != n:
        raise ValueError("problem: размер сетки должен совпадать с полями")
    fixed = np.zeros(values.shape, dtype=bool)
    boundary = np.zeros(values.shape)
    robin = []
    # Deliberately derive the geometry here, without production masks or weights.
    for axis, letter in enumerate("zyx"):
        for side in (0, 1):
            face = problem.faces[f"{letter}{side}"]
            index = [slice(None)] * 3
            index[axis] = -1 if side else 0
            index = tuple(index)
            sampled = problem.face_values(f"{letter}{side}", t)
            if face.kind == 1:
                overlap = fixed[index]
                if not np.allclose(
                    boundary[index][overlap], sampled[overlap], rtol=1e-12, atol=1e-12
                ):
                    raise ValueError("На ребре или в углу несовместимы условия 1 рода")
                boundary[index] = sampled
                fixed[index] = True
            elif face.kind == 3:
                robin.append((index, face.beta, sampled))
            else:
                raise ValueError("Поддержаны граничные условия 1 и 3 рода")
    values = values.copy()
    values[fixed] = boundary[fixed]
    free = ~fixed
    size = int(free.sum())
    rows, columns, coefficients = [], [], []
    indices = np.full(values.shape, -1, dtype=int) if jacobian else None
    if jacobian:
        indices[free] = np.arange(size)
    diagonal = np.full(values.shape, inverse_dt) if jacobian else None
    with np.errstate(over="ignore", invalid="ignore"):
        scaled = np.sqrt(alpha) * values
        conductivity = 1 + scaled * scaled
        if not np.isfinite(conductivity).all():
            raise ValueError("Проводимость должна быть конечной")
        residual = (values - previous) * inverse_dt - source
        rhs = previous * inverse_dt + source if scale else None
        for index, beta, load in robin:
            # k * du/dn is already present in r; no additional k belongs here.
            residual[index] += 2 * n * (beta * values[index] - load)
            if jacobian:
                diagonal[index] += 2 * n * beta
            if scale:
                rhs[index] += 2 * n * load
        for axis in range(3):
            for sign in (-1, 1):
                origin = [slice(None)] * 3
                neighbor = [slice(None)] * 3
                origin[axis] = slice(1, None) if sign < 0 else slice(None, -1)
                neighbor[axis] = slice(None, -1) if sign < 0 else slice(1, None)
                origin, neighbor = tuple(origin), tuple(neighbor)
                center, adjacent = values[origin], values[neighbor]
                geometry = np.full(center.shape, float(n**2))
                boundary_origin = [slice(None)] * 3
                boundary_origin[axis] = -1 if sign < 0 else 0
                # The inward normal edge spans a half width at the boundary.
                geometry[tuple(boundary_origin)] *= 2
                weight = (
                    0.5 * conductivity[origin] + 0.5 * conductivity[neighbor]
                ) * geometry
                difference = center - adjacent
                residual[origin] += weight * difference
                if jacobian:
                    diagonal[origin] += weight + alpha * center * difference * geometry
                    derivative = -weight + alpha * adjacent * difference * geometry
                    connected = free[origin] & free[neighbor]
                    rows.append(indices[origin][connected])
                    columns.append(indices[neighbor][connected])
                    coefficients.append(derivative[connected])
                if scale:
                    known = fixed[neighbor]
                    rhs[origin] += weight * np.where(known, adjacent, 0)
        residual = residual[free]
        if not np.isfinite(residual).all():
            raise ValueError("Нелинейная невязка должна быть конечной")
        matrix = None
        if jacobian:
            rows.append(indices[free])
            columns.append(indices[free])
            coefficients.append(diagonal[free])
            data = np.concatenate(coefficients)
            if not np.isfinite(data).all():
                raise ValueError("Якобиан должен быть конечным")
            matrix = csr_matrix(
                (data, (np.concatenate(rows), np.concatenate(columns))),
                shape=(size, size),
            )
            if not np.isfinite(matrix.data).all():
                raise ValueError("Якобиан должен быть конечным")
        residual_scale = None
        if scale:
            rhs = rhs[free]
            applied = residual + rhs
            if not np.isfinite(rhs).all() or not np.isfinite(applied).all():
                raise ValueError("Масштаб невязки должен быть конечным")
            residual_scale = float(max(np.max(np.abs(rhs)), np.max(np.abs(applied))))
    return residual, matrix, residual_scale


def _dispatch(values, previous, alpha, source, dt, *, problem, t, **options):
    if problem is None:
        return _evaluate(values, previous, alpha, source, dt, **options)
    return _evaluate_boundaries(
        values, previous, alpha, source, dt, problem=problem, t=t, **options
    )


def newton_system3d(values, previous, alpha, source, dt, *, problem=None, t=0.0):
    """Return the nonlinear residual and its independent CSR Jacobian.

    ``values`` already contains the current Dirichlet boundary data;
    ``previous`` is the actual previous time layer. ``source`` is the fixed
    full-grid F at the current time, and dt is the actual layer duration.
    With ``problem``, physical face data are sampled at ``t`` and the residual
    is a flat C-order vector of all nodes free of kind-1 constraints, including
    kind-3 faces, edges and corners. Without it, the original interior residual
    shape (n-1, n-1, n-1) and supplied boundary values are preserved.
    No derivative or matrix column is introduced for a fixed boundary node.
    All fields must have shape (n+1, n+1, n+1), with 4 <= n <= DIRECT_MAX_N.
    """
    residual, matrix, _ = _dispatch(
        values, previous, alpha, source, dt, problem=problem, t=t, jacobian=True
    )
    return residual, matrix


def newton_residual3d(values, previous, alpha, source, dt, *, problem=None, t=0.0):
    """Evaluate the same residual without allocating a sparse Jacobian."""
    return _dispatch(values, previous, alpha, source, dt, problem=problem, t=t)[0]


def newton_scale3d(values, previous, alpha, source, dt, *, problem=None, t=0.0):
    """Return max(||rhs||inf, ||A(values) values||inf) on free nodes.

    The fixed boundary contributions are included in rhs and removed from A.
    Call once at the start of a layer to obtain its fixed residual scale.
    """
    return _dispatch(
        values, previous, alpha, source, dt, problem=problem, t=t, scale=True
    )[2]
