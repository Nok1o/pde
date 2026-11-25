"""Independent checks of the small-grid nonlinear heat reference assembly."""

import numpy as np
import pytest
from scipy.sparse import diags, eye, isspmatrix_csr, kronsum
from scipy.sparse.linalg import spsolve

import pdelab.heat3d_reference as reference
from pdelab.heat3d_reference import (
    newton_residual3d,
    newton_scale3d,
    newton_system3d,
)
from pdelab.problem3d import DIRECT_MAX_N


def fields(n=4):
    z, y, x = np.indices((n + 1,) * 3, dtype=float) / n
    values = 0.3 + x + 2 * y - 0.7 * z + x * y + 0.2 * z**2
    previous = -0.2 + 0.8 * x - 0.4 * y + z**2
    source = 0.5 + x**2 - 3 * y + 4 * z
    return values, previous, source


def manual_residual_and_scale(values, previous, alpha, source, dt):
    """Scalar row construction with explicit coordinates and fixed-node tests."""
    n = values.shape[0] - 1
    residual = np.empty((n - 1,) * 3)
    rhs = np.empty_like(residual)
    applied = np.empty_like(residual)
    for interior_index in np.ndindex(residual.shape):
        node = tuple(component + 1 for component in interior_index)
        u = values[node]
        rhs_value = previous[node] / dt + source[node]
        applied_value = u / dt
        residual_value = (u - previous[node]) / dt - source[node]
        for axis in range(3):
            for shift in (-1, 1):
                neighbor = list(node)
                neighbor[axis] += shift
                neighbor = tuple(neighbor)
                v = values[neighbor]
                edge = (2 + alpha * u**2 + alpha * v**2) * n**2 / 2
                residual_value += edge * (u - v)
                applied_value += edge * u
                if 0 in neighbor or n in neighbor:
                    rhs_value += edge * v
                else:
                    applied_value -= edge * v
        residual[interior_index] = residual_value
        rhs[interior_index] = rhs_value
        applied[interior_index] = applied_value
    scale = max(np.max(np.abs(rhs)), np.max(np.abs(applied)))
    return residual, scale


@pytest.mark.parametrize("n", [4, 5])
@pytest.mark.parametrize("alpha", [0.0, 0.5, 2.0])
def test_residual_and_scale_match_independent_rows(n, alpha):
    values, previous, source = fields(n)
    originals = [array.copy() for array in (values, previous, source)]
    residual, jacobian = newton_system3d(values, previous, alpha, source, 0.07)
    expected, expected_scale = manual_residual_and_scale(
        values, previous, alpha, source, 0.07
    )
    assert residual.shape == (n - 1,) * 3
    assert isspmatrix_csr(jacobian)
    assert jacobian.shape == ((n - 1) ** 3,) * 2
    assert jacobian.nnz == (n - 1) ** 3 + 6 * (n - 2) * (n - 1) ** 2
    np.testing.assert_allclose(residual, expected, rtol=1e-13, atol=1e-12)
    np.testing.assert_array_equal(
        newton_residual3d(values, previous, alpha, source, 0.07), residual
    )
    np.testing.assert_allclose(
        newton_scale3d(values, previous, alpha, source, 0.07), expected_scale
    )
    for actual, original in zip((values, previous, source), originals):
        np.testing.assert_array_equal(actual, original)


@pytest.mark.parametrize("alpha", [0.0, 0.7, 2.0])
def test_every_jacobian_column_against_numerical_derivative(alpha):
    values, previous, source = fields()
    values[1:-1, 1:-1, 1:-1] += np.random.default_rng(21).normal(size=(3,) * 3)
    _, jacobian = newton_system3d(values, previous, alpha, source, 0.13)
    step = 2e-6
    # All columns cover each axis, plus nodes next to faces, edges and corners.
    for column, index in enumerate(np.ndindex((3,) * 3)):
        node = tuple(component + 1 for component in index)
        high, low = values.copy(), values.copy()
        high[node] += step
        low[node] -= step
        expected = (
            manual_residual_and_scale(high, previous, alpha, source, 0.13)[0]
            - manual_residual_and_scale(low, previous, alpha, source, 0.13)[0]
        ) / (2 * step)
        np.testing.assert_allclose(
            jacobian[:, column].toarray().ravel(),
            expected.ravel(order="C"),
            atol=2e-6,
            rtol=2e-8,
        )


def test_linear_limit_is_kronecker_laplacian():
    n, dt = 5, 0.03
    values, previous, source = fields(n)
    _, actual = newton_system3d(values, previous, 0.0, source, dt)
    one_axis = diags((-np.ones(n - 2), 2 * np.ones(n - 1), -np.ones(n - 2)), (-1, 0, 1))
    expected = n**2 * kronsum(kronsum(one_axis, one_axis), one_axis)
    expected += eye((n - 1) ** 3) / dt
    difference = actual - expected
    assert np.max(np.abs(difference.data), initial=0) < 1e-12


def test_direct_linear_correction_recovers_every_node():
    n, dt = 5, 0.04
    target, previous, _ = fields(n)
    source = np.zeros_like(target)
    # Manufacture a discrete load using scalar rows, without an exact callback.
    source[1:-1, 1:-1, 1:-1] = manual_residual_and_scale(
        target, previous, 0.0, source, dt
    )[0]
    candidate = target.copy()
    candidate[1:-1, 1:-1, 1:-1] = np.random.default_rng(22).normal(size=(n - 1,) * 3)
    residual, jacobian = newton_system3d(candidate, previous, 0.0, source, dt)
    candidate[1:-1, 1:-1, 1:-1] -= spsolve(jacobian, residual.ravel()).reshape(
        (n - 1,) * 3
    )
    np.testing.assert_allclose(candidate, target, atol=5e-15, rtol=5e-15)
    np.testing.assert_allclose(
        newton_residual3d(candidate, previous, 0.0, source, dt), 0, atol=3e-13
    )


def test_source_and_actual_previous_layer_enter_residual():
    values, previous, source = fields()
    residual = newton_residual3d(values, previous, 0.5, source, 0.2)
    changed_previous = previous.copy()
    changed_previous[2, 1, 3] += 0.7
    changed_source = source.copy()
    changed_source[1, 2, 3] -= 2.0
    actual = newton_residual3d(values, changed_previous, 0.5, changed_source, 0.2)
    expected = residual.copy()
    expected[1, 0, 2] -= 0.7 / 0.2
    expected[0, 1, 2] += 2.0
    np.testing.assert_allclose(actual, expected, atol=1e-12)


def test_residual_and_scale_do_not_allocate_jacobian(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("A residual-only evaluation must not build a matrix")

    monkeypatch.setattr(reference, "csr_matrix", forbidden)
    values, previous, source = fields()
    assert np.isfinite(newton_residual3d(values, previous, 0.5, source, 0.1)).all()
    assert np.isfinite(newton_scale3d(values, previous, 0.5, source, 0.1))


@pytest.mark.parametrize(
    "function", [newton_system3d, newton_residual3d, newton_scale3d]
)
def test_direct_size_limit_is_checked_before_conversion(function):
    class TooLarge:
        shape = (DIRECT_MAX_N + 2,) * 3

        def __array__(self, *args, **kwargs):
            raise AssertionError("Oversized input must not be converted or copied")

    with pytest.raises(ValueError, match="n <= 16"):
        function(TooLarge(), None, 0.5, None, 0.1)


@pytest.mark.parametrize("shape", [(5, 5), (5, 5, 6), (4, 4, 4), (0, 0, 0)])
def test_reject_invalid_grid_shape(shape):
    values = np.zeros(shape)
    with pytest.raises(ValueError):
        newton_system3d(values, values, 0.5, values, 0.1)


@pytest.mark.parametrize("field", ["previous", "source"])
def test_reject_mismatched_fields(field):
    values, previous, source = fields()
    inputs = dict(values=values, previous=previous, alpha=0.5, source=source, dt=0.1)
    inputs[field] = np.zeros((4,) * 3)
    with pytest.raises(ValueError, match=field):
        newton_system3d(**inputs)


@pytest.mark.parametrize("field", ["values", "previous", "source"])
@pytest.mark.parametrize("bad", [np.inf, -np.inf, np.nan])
@pytest.mark.parametrize("node", [(0, 0, 0), (2, 2, 2)])
def test_reject_nonfinite_values_on_entire_grid(field, bad, node):
    values, previous, source = fields()
    inputs = dict(values=values, previous=previous, alpha=0.5, source=source, dt=0.1)
    inputs[field][node] = bad
    with pytest.raises(ValueError, match=field):
        newton_system3d(**inputs)


@pytest.mark.parametrize(
    "field,bad",
    [
        ("alpha", -1.0),
        ("alpha", np.nan),
        ("alpha", np.inf),
        ("alpha", True),
        ("alpha", [0.5]),
        ("alpha", np.complex128(0.5 + 0.1j)),
        ("dt", 0.0),
        ("dt", -0.1),
        ("dt", np.inf),
        ("dt", np.nan),
        ("dt", False),
        ("dt", [0.1]),
        ("dt", np.complex128(0.1 + 0.2j)),
        ("dt", np.nextafter(0.0, 1.0)),
    ],
)
def test_reject_invalid_parameters(field, bad):
    values, previous, source = fields()
    inputs = dict(values=values, previous=previous, alpha=0.5, source=source, dt=0.1)
    inputs[field] = bad
    with pytest.raises(ValueError, match=field):
        newton_system3d(**inputs)


@pytest.mark.parametrize(
    "function", [newton_system3d, newton_residual3d, newton_scale3d]
)
def test_reject_conductivity_and_residual_overflow(function):
    values, previous, source = fields()
    values[2, 2, 2] = 1e200
    with pytest.raises(ValueError, match="Проводимость"):
        function(values, previous, 0.5, source, 0.1)
    values[2, 2, 2] = 1e307
    with pytest.raises(ValueError, match="невязка"):
        function(values, previous, 0.0, source, 0.1)


def test_zero_scale_and_large_constant_linear_field():
    values = np.zeros((5,) * 3)
    assert newton_scale3d(values, values, 0.5, values, 0.1) == 0
    values.fill(1e200)
    source = np.zeros_like(values)
    residual, jacobian = newton_system3d(values, values, 0.0, source, 0.1)
    np.testing.assert_array_equal(residual, 0)
    assert np.isfinite(jacobian.data).all()


def test_reject_jacobian_and_scale_overflow_with_finite_residual():
    values = np.full((5,) * 3, 1.5e153)
    source = np.zeros_like(values)
    np.testing.assert_array_equal(
        newton_residual3d(values, values, 1.0, source, 0.1), 0
    )
    with pytest.raises(ValueError, match="Якобиан"):
        newton_system3d(values, values, 1.0, source, 0.1)
    with pytest.raises(ValueError, match="Масштаб"):
        newton_scale3d(values, values, 1.0, source, 0.1)


@pytest.mark.parametrize("field", ["values", "previous", "source"])
def test_reject_complex_fields(field):
    values, previous, source = fields()
    inputs = dict(values=values, previous=previous, alpha=0.5, source=source, dt=0.1)
    inputs[field] = inputs[field].astype(complex) + 1j
    with pytest.raises(ValueError, match=field):
        newton_system3d(**inputs)


def test_maximum_reference_grid_uses_sparse_storage():
    shape = (DIRECT_MAX_N + 1,) * 3
    values = np.zeros(shape)
    residual, jacobian = newton_system3d(values, values, 0.5, values, 0.1)
    unknowns = (DIRECT_MAX_N - 1) ** 3
    assert residual.size == unknowns
    assert jacobian.shape == (unknowns, unknowns)
    assert jacobian.nnz < 7 * unknowns
    assert (
        jacobian.data.nbytes + jacobian.indices.nbytes + jacobian.indptr.nbytes
        < 400_000
    )
