"""Independent small-grid Newton assembly for Dirichlet heat layers in 3D.

Only interior temperatures are unknown. All input fields use [z, y, x];
the sparse matrix uses C order, so x varies fastest. This reference does
not use the frozen production operator or substitute manufactured solutions.
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


def newton_system3d(values, previous, alpha, source, dt):
    """Return the interior nonlinear residual and its independent CSR Jacobian.

    ``values`` already contains the current Dirichlet boundary data;
    ``previous`` is the actual previous time layer. ``source`` is the fixed
    full-grid F at the current time, and dt is the actual layer duration.
    No derivative or matrix column is introduced for a fixed boundary node.
    All fields must have shape (n+1, n+1, n+1), with 4 <= n <= DIRECT_MAX_N.
    """
    residual, matrix, _ = _evaluate(values, previous, alpha, source, dt, jacobian=True)
    return residual, matrix


def newton_residual3d(values, previous, alpha, source, dt):
    """Evaluate the same residual without allocating a sparse Jacobian."""
    return _evaluate(values, previous, alpha, source, dt)[0]


def newton_scale3d(values, previous, alpha, source, dt):
    """Return max(||rhs||inf, ||A(values) values||inf) on interior nodes.

    The fixed boundary contributions are included in rhs and removed from A.
    Call once at the start of a layer to obtain its fixed residual scale.
    """
    return _evaluate(values, previous, alpha, source, dt, scale=True)[2]
