import numpy as np
import pytest

from pdelab.problem3d import (
    Problem3D,
    blank_field3d,
    discrete_system3d,
    laplacian3d,
    make_problem3d,
    reference_solution3d,
    select_points3d,
    validate_points3d,
    workspace_bytes,
)


def test_matrix_matches_stencil_with_six_nonzero_faces():
    rng = np.random.default_rng(12)
    values = rng.normal(size=(7, 7, 7))
    problem = Problem3D("random", 6, rng.normal(size=values.shape), values)
    matrix, rhs = discrete_system3d(problem)
    matrix_residual = matrix @ values[1:-1, 1:-1, 1:-1].ravel() - rhs
    stencil_residual = -problem.h**2 * (
        laplacian3d(values, problem.h) - problem.f[1:-1, 1:-1, 1:-1]
    )
    np.testing.assert_allclose(matrix_residual, stencil_residual.ravel(), atol=1e-14)


@pytest.mark.parametrize("preset", ["constant", "affine"])
def test_direct_reference_reproduces_harmonic_fields(preset):
    problem = make_problem3d(preset, 8)
    np.testing.assert_allclose(reference_solution3d(problem), problem.exact, atol=1e-13)


def test_sine_has_second_order_spatial_error():
    errors = []
    for n in (8, 16):
        problem = make_problem3d("sine", n)
        errors.append(np.max(np.abs(reference_solution3d(problem) - problem.exact)))
    assert 3.9 < errors[0] / errors[1] < 4.1


def test_coordinate_contract_and_all_faces():
    problem = make_problem3d("affine", 8)
    points = select_points3d(8, "points", points=[(1, 2, 3), (0, 0, 0), (8, 8, 8)])
    np.testing.assert_array_equal(points, [[3, 2, 1], [0, 0, 0], [8, 8, 8]])
    assert problem.exact[3, 2, 1] == pytest.approx(1 + 2 / 8 + 3 * 2 / 8 + 5 * 3 / 8)
    values = blank_field3d(problem)
    assert np.isnan(values[1:-1, 1:-1, 1:-1]).all()
    for axis in range(3):
        for end in (0, -1):
            np.testing.assert_equal(
                np.take(values, end, axis=axis), np.take(problem.exact, end, axis=axis)
            )


@pytest.mark.parametrize("plane,axis", [("xy", 0), ("xz", 1), ("yz", 2)])
def test_slice_selects_correct_axis(plane, axis):
    points = select_points3d(8, "slice", plane=plane, index=2)
    assert len(points) == 7**2
    assert np.all(points[:, axis] == 2)
    assert len(select_points3d(8, "patch")) == 27
    assert len(select_points3d(8, "field")) == 7**3


@pytest.mark.parametrize(
    "points",
    [
        [],
        [[1, 2]],
        [[1.5, 2, 3]],
        [[-1, 2, 3]],
        [[9, 2, 3]],
        [[1, 2, 3], [1, 2, 3]],
        [[np.nan, 2, 3]],
    ],
)
def test_invalid_queries_are_rejected(points):
    with pytest.raises(ValueError):
        validate_points3d(8, points)


def test_size_limits_are_checked_before_calling_functions():
    def should_not_run(*coordinates):
        raise AssertionError("Размер должен проверяться до построения поля")

    with pytest.raises(ValueError):
        Problem3D.from_functions(1000, should_not_run, should_not_run)
    with pytest.raises(ValueError, match="Прямой 3D-эталон"):
        reference_solution3d(make_problem3d(n=24))
    assert workspace_bytes(16) > workspace_bytes(8)


def test_custom_problem_validates_shape_and_finiteness():
    with pytest.raises(ValueError, match="конечный массив"):
        Problem3D("invalid", 4, np.zeros((5, 5)), np.zeros((5, 5, 5)))
    with pytest.raises(ValueError, match="конечный массив"):
        Problem3D.from_functions(4, lambda x, y, z: np.nan, lambda x, y, z: 0)
