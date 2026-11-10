import numpy as np
import pytest

from pdelab.problem3d import (
    Problem3D,
    laplacian3d,
    make_problem3d,
    reference_solution3d,
    select_points3d,
)
from pdelab.solvers3d import solve3d, splitting_parameter3d, sweep3d, warmup3d


@pytest.fixture(scope="module", autouse=True)
def compiled():
    warmup3d()


@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize("size", [1, 2, 7])
def test_each_sweep_matches_independent_line_system(axis, size):
    rhs = np.random.default_rng(8).normal(size=(size,) * 3)
    ratio = 2.5
    matrix = (
        np.diag(np.full(size, 1 + 2 * ratio))
        + np.diag(np.full(size - 1, -ratio), 1)
        + np.diag(np.full(size - 1, -ratio), -1)
    )
    moved = np.moveaxis(rhs, axis, -1)
    expected = np.linalg.solve(matrix, moved.reshape(-1, size).T).T.reshape(moved.shape)
    np.testing.assert_allclose(
        sweep3d(rhs, ratio, axis), np.moveaxis(expected, -1, axis), atol=1e-13
    )


def test_three_direction_amplification_matches_derived_formula():
    n = 8
    tau, bound = splitting_parameter3d(n)
    axis = np.arange(1, n) / n
    for frequencies in ((1, 1, 1), (7, 7, 7), (1, 3, 7)):
        z, y, x = (np.sin(np.pi * k * axis) for k in frequencies)
        error = z[:, None, None] * y[None, :, None] * x[None, None, :]
        eigenvalues = 4 * n**2 * np.sin(np.pi * np.array(frequencies) / (2 * n)) ** 2
        correction = -tau * eigenvalues.sum() * error
        for direction in (2, 1, 0):
            correction = sweep3d(correction, tau * n**2, direction)
        factor = 1 - tau * eigenvalues.sum() / np.prod(1 + tau * eigenvalues)
        np.testing.assert_allclose(error + correction, factor * error, atol=1e-14)
        assert 0 < factor <= bound + 1e-14 < 1


@pytest.mark.parametrize("preset", ["sine", "affine", "constant"])
def test_lod_matches_independent_3d_system(preset):
    problem = make_problem3d(preset, 8)
    result = solve3d(problem, select_points3d(8), "lod_serial", tol=1e-11)
    assert result.converged
    np.testing.assert_allclose(
        result.values, reference_solution3d(problem), atol=1e-8, rtol=0
    )
    measured = np.max(
        np.abs(laplacian3d(result.values, problem.h) - problem.f[1:-1, 1:-1, 1:-1])
    )
    assert result.relative_residual == pytest.approx(
        measured / result.extras["residual_scale"]
    )
    assert result.extras["relative_residuals"][-2] > 1e-11


def test_relative_stopping_and_zero_rhs():
    problem = make_problem3d("affine", 8)
    points = select_points3d(8)
    base = solve3d(problem, points, "lod_serial")
    for scale in (1e-6, -1e6):
        scaled = Problem3D(problem.name, 8, problem.f * scale, problem.boundary * scale)
        result = solve3d(scaled, points, "lod_serial")
        assert result.iterations == base.iterations
        np.testing.assert_allclose(result.values / scale, base.values, atol=1e-12)
    zero = Problem3D.from_functions(8, lambda x, y, z: 0, lambda x, y, z: 0)
    result = solve3d(zero, points, "lod_serial")
    assert result.converged and result.iterations == 0 and result.relative_residual == 0
    assert not solve3d(problem, points, "lod_serial", max_iter=1).converged


def test_mc_constant_is_exact_and_boundary_requires_no_walks():
    problem = make_problem3d("constant", 8)
    points = select_points3d(8, "points", points=[(2, 3, 4), (0, 0, 0), (8, 4, 5)])
    result = solve3d(problem, points, walks=64)
    np.testing.assert_array_equal(result.values[tuple(points.T)], 7)
    np.testing.assert_array_equal(result.stderr[tuple(points.T)], 0)
    assert result.walks == 64
    np.testing.assert_array_equal(result.sample_count[tuple(points.T)], [64, 0, 0])
    assert np.isnan(result.values[1, 1, 1])


@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize("end", [0, -1])
def test_mc_can_exit_through_every_face(axis, end):
    problem = make_problem3d("constant", 8)
    problem.boundary[:] = 0
    face = [slice(None)] * 3
    face[axis] = end
    problem.boundary[tuple(face)] = 1
    result = solve3d(problem, [[4, 4, 4]], walks=16000, seed=930 + 2 * axis + end)
    assert abs(result.values[4, 4, 4] - 1 / 6) < 5 * result.stderr[4, 4, 4]


def test_mc_noncentral_source_and_reproducibility():
    problem = Problem3D.from_functions(
        8,
        lambda x, y, z: -2
        * (
            y * (1 - y) * z * (1 - z)
            + x * (1 - x) * z * (1 - z)
            + x * (1 - x) * y * (1 - y)
        ),
        lambda x, y, z: 0,
        lambda x, y, z: x * (1 - x) * y * (1 - y) * z * (1 - z),
    )
    points = select_points3d(8, "points", points=[(2, 3, 5), (4, 4, 4)])
    result = solve3d(problem, points, walks=20000, seed=31)
    indices = tuple(points.T)
    assert np.all(
        np.abs(result.values[indices] - problem.exact[indices])
        < 5 * result.stderr[indices]
    )
    repeated = solve3d(problem, points, walks=20000, seed=31)
    np.testing.assert_equal(result.values, repeated.values)
    assert result.steps == repeated.steps


def test_asymmetric_solution_with_nonzero_source_and_all_boundary_values():
    def exact(x, y, z):
        return 1 + x**2 + 2 * y**2 + 3 * z**2 + x * y + 2 * y * z + 3 * x * z

    problem = Problem3D.from_functions(8, lambda x, y, z: 12.0, exact, exact)
    points = select_points3d(8, "points", points=[(1, 3, 6), (6, 2, 5)])
    lod = solve3d(problem, points, "lod_serial", tol=1e-11)
    np.testing.assert_allclose(lod.values, problem.exact, atol=1e-8, rtol=0)
    mc = solve3d(problem, points, "mc", walks=30000, seed=182)
    indices = tuple(points.T)
    assert np.all(
        np.abs(mc.values[indices] - problem.exact[indices]) < 5 * mc.stderr[indices]
    )


@pytest.mark.parametrize("n", [24, 64])
def test_large_grid_lod_matches_exact_discrete_sine_mode(n):
    problem = make_problem3d("sine", n)
    result = solve3d(problem, select_points3d(n), "lod_serial", tol=1e-9)
    # Точный ответ семиточечной системы для синуса известен из её собственных значений.
    discrete = problem.exact * (np.pi / (2 * n * np.sin(np.pi / (2 * n)))) ** 2
    assert result.converged
    np.testing.assert_allclose(result.values, discrete, atol=2e-9, rtol=0)


def test_walk_limit_and_unsupported_methods_are_explicit():
    problem = make_problem3d(n=8)
    with pytest.raises(RuntimeError, match="предел шагов"):
        solve3d(problem, [[4, 4, 4]], max_steps=1)
    for method in ("reuse", "layers", "lod_parallel"):
        with pytest.raises(ValueError, match="только"):
            solve3d(problem, [[4, 4, 4]], method)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"walks": 1},
        {"tol": 0},
        {"tol": np.nan},
        {"max_iter": 0},
        {"seed": -1},
        {"tau": -1},
        {"max_steps": 1.5},
    ],
)
def test_invalid_solver_settings(kwargs):
    with pytest.raises(ValueError):
        solve3d(make_problem3d(n=4), [[2, 2, 2]], **kwargs)
