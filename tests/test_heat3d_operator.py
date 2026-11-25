"""Independent small-matrix checks for the common 3D heat operator."""

from dataclasses import replace

import numpy as np
import pytest
from scipy.sparse.linalg import spsolve

from pdelab.boundary3d import FACES
from pdelab.heat3d_operator import (
    HeatOperator3D,
    IncompatibleHeatBoundaryError,
    UnsupportedHeatBoundaryError,
    dirichlet_data3d,
    freeze_heat3d,
)
from pdelab.heat3d_problem import HeatFace3D, make_heat_problem3d


def independent_matrix(problem, previous, iterate, t, dt):
    """Coordinate loops and raw edge fluxes, independent of production slices."""
    n, shape = problem.n, problem.shape
    values = iterate.copy()
    boundary = problem.evaluate(problem.exact, t)
    fixed = np.zeros(shape, dtype=bool)
    fixed[[0, n], :, :] = True
    fixed[:, [0, n], :] = True
    fixed[:, :, [0, n]] = True
    values[fixed] = boundary[fixed]
    conductivity = 1 + problem.alpha * values**2
    source = problem.evaluate(problem.source, t)
    matrix = np.zeros((values.size, values.size))
    rhs = np.empty(values.size)
    for node in np.ndindex(shape):
        row = np.ravel_multi_index(node, shape)
        if fixed[node]:
            matrix[row, row], rhs[row] = 1, boundary[node]
            continue
        matrix[row, row] = 1 / dt
        rhs[row] = previous[node] / dt + source[node]
        for axis in range(3):
            for sign in (-1, 1):
                neighbor = list(node)
                neighbor[axis] += sign
                neighbor = tuple(neighbor)
                coefficient = (conductivity[node] + conductivity[neighbor]) / (
                    2 * problem.h**2
                )
                matrix[row, row] += coefficient
                if fixed[neighbor]:
                    rhs[row] += coefficient * boundary[neighbor]
                else:
                    matrix[row, np.ravel_multi_index(neighbor, shape)] = -coefficient
    return matrix, rhs


@pytest.mark.parametrize("alpha", [0.0, 0.5])
def test_full_operator_matches_independent_edge_assembly(alpha):
    problem = make_heat_problem3d("polynomial", n=4, alpha=alpha)
    previous = problem.initial_values()
    iterate = previous + np.random.default_rng(9).normal(0, 0.2, problem.shape)
    original = iterate.copy()
    operator = freeze_heat3d(problem, previous, iterate, 0.03, 0.03)
    expected, rhs = independent_matrix(problem, previous, iterate, 0.03, 0.03)
    matrix, actual_rhs = operator.assemble()
    np.testing.assert_allclose(matrix.toarray(), expected, rtol=2e-15, atol=2e-13)
    np.testing.assert_allclose(actual_rhs, rhs, rtol=2e-15, atol=2e-13)
    test_values = np.random.default_rng(14).normal(size=problem.shape)
    np.testing.assert_allclose(
        operator.apply(test_values).ravel(), expected @ test_values.ravel(), atol=5e-13
    )
    np.testing.assert_array_equal(iterate, original)
    assert np.all(operator.diagonal - operator.weights.sum(axis=-1) > 0)
    assert operator.volumes[2, 2, 2] == problem.h**3
    assert operator.volumes[0, 2, 2] == problem.h**3 / 2
    assert operator.volumes[0, 0, 2] == problem.h**3 / 4
    assert operator.volumes[0, 0, 0] == problem.h**3 / 8
    np.testing.assert_allclose(operator.volumes.sum(), 1)


def test_correction_rhs_is_not_adjusted_twice_and_does_not_mutate_snapshot():
    problem = make_heat_problem3d("polynomial", n=4)
    values = problem.initial_values()
    operator = freeze_heat3d(problem, values, values, 0.01, 0.01)
    original_rhs = operator.rhs.copy()
    control = np.random.default_rng(4).normal(size=problem.shape)
    rhs = operator.rhs - operator.apply(control)
    correction = operator.with_rhs(rhs)
    np.testing.assert_array_equal(correction.rhs, rhs)
    np.testing.assert_array_equal(
        correction.fixed_values[operator.fixed], rhs[operator.fixed]
    )
    np.testing.assert_array_equal(correction.fixed_values[~operator.fixed], 0)
    assert correction.diagonal is operator.diagonal
    assert correction.weights is operator.weights
    matrix, expected_rhs = operator.assemble()
    corrected_matrix, corrected_rhs = correction.assemble()
    np.testing.assert_array_equal(corrected_matrix.toarray(), matrix.toarray())
    restored = spsolve(corrected_matrix, corrected_rhs) + control.ravel()
    np.testing.assert_allclose(restored, spsolve(matrix, expected_rhs), atol=2e-14)
    rhs[:] = 123
    np.testing.assert_array_equal(operator.rhs, original_rhs)
    assert not np.all(correction.rhs == 123)


def test_snapshot_arrays_are_owned_and_immutable():
    shape = (3, 3, 3)
    diagonal = np.ones(shape)
    operator = HeatOperator3D(
        diagonal,
        np.zeros((*shape, 6)),
        np.zeros(shape),
        np.ones(shape, dtype=bool),
        np.zeros(shape),
        np.ones(shape),
    )
    diagonal[:] = 10
    np.testing.assert_array_equal(operator.diagonal, 1)
    for name in ("diagonal", "weights", "rhs", "fixed", "fixed_values", "volumes"):
        array = getattr(operator, name)
        with pytest.raises(ValueError):
            array.flat[0] = 0
        with pytest.raises(ValueError):
            array.setflags(write=True)


def test_boundary_intersections_are_checked_again_at_later_times():
    problem = make_heat_problem3d("polynomial", n=4)
    faces = dict(problem.faces)
    faces["x0"] = HeatFace3D(1, lambda x, y, z, t: problem.exact(x, y, z, t) + t)
    conflicting = replace(problem, faces=faces)
    dirichlet_data3d(conflicting, 0)
    with pytest.raises(IncompatibleHeatBoundaryError, match="несовместимы"):
        dirichlet_data3d(conflicting, 0.1)
    with pytest.raises(IncompatibleHeatBoundaryError, match="несовместимы"):
        freeze_heat3d(
            conflicting,
            conflicting.initial_values(),
            conflicting.initial_values(),
            0.1,
            0.1,
        )


def test_reject_robin_and_invalid_layer_data():
    kinds = dict.fromkeys(FACES, 1)
    kinds["y0"] = 3
    mixed = make_heat_problem3d(n=4, kinds=kinds)
    with pytest.raises(UnsupportedHeatBoundaryError):
        freeze_heat3d(mixed, mixed.initial_values(), mixed.initial_values(), 0.1, 0.1)
    problem = make_heat_problem3d(n=4)
    initial = problem.initial_values()
    for dt in (0, -1, np.nan, np.inf):
        with pytest.raises(ValueError):
            freeze_heat3d(problem, initial, initial, 0.1, dt)
    for invalid in (initial[:-1], np.full(problem.shape, np.nan)):
        with pytest.raises(ValueError):
            freeze_heat3d(problem, invalid, initial, 0.1, 0.1)
        with pytest.raises(ValueError):
            freeze_heat3d(problem, initial, invalid, 0.1, 0.1)
    with pytest.raises(ValueError):
        freeze_heat3d(problem, initial, initial, -0.1, 0.1)


def test_factory_never_reads_exact_solution():
    problem = make_heat_problem3d("polynomial", n=4)

    def forbidden(*args):
        pytest.fail("the operator accessed the exact solution")

    problem = replace(problem, exact=forbidden)
    initial = problem.initial_values()
    operator = freeze_heat3d(problem, initial, initial, 0.01, 0.01)
    assert np.isfinite(operator.rhs).all()


def test_assembly_limit_precedes_index_allocation(monkeypatch):
    problem = make_heat_problem3d(n=17)
    initial = problem.initial_values()
    operator = freeze_heat3d(problem, initial, initial, 0.01, 0.01)

    def forbidden(*args, **kwargs):
        pytest.fail("large sparse assembly allocated indices before checking the limit")

    monkeypatch.setattr("pdelab.heat3d_operator.np.arange", forbidden)
    with pytest.raises(ValueError, match="n ≤ 16"):
        operator.assemble()


@pytest.mark.parametrize(
    "attribute", ["diagonal", "weights", "rhs", "fixed_values", "volumes"]
)
def test_nonfinite_snapshot_is_rejected(attribute):
    problem = make_heat_problem3d(n=4)
    initial = problem.initial_values()
    operator = freeze_heat3d(problem, initial, initial, 0.01, 0.01)
    invalid = getattr(operator, attribute).copy()
    invalid.flat[0] = np.inf
    with pytest.raises(ValueError):
        replace(operator, **{attribute: invalid})
