import numpy as np
import pytest
from scipy.sparse import diags, kronsum
from scipy.sparse.linalg import spsolve

from pdelab.operator import FrozenOperator
from pdelab.problem import laplacian, make_problem
from pdelab.solvers import solve_tridiagonal


def manufactured(n):
    x, y = np.meshgrid(np.linspace(0, 1, n + 1), np.linspace(0, 1, n + 1))
    u = 2 + x + np.sin(np.pi * x) * np.sin(np.pi * y)
    ux = 1 + np.pi * np.cos(np.pi * x) * np.sin(np.pi * y)
    uy = np.pi * np.sin(np.pi * x) * np.cos(np.pi * y)
    k = 1 + x + y
    sigma = 1 + x * y
    b = 2 * np.pi**2 * k * (u - 2 - x) - ux - uy + sigma * u
    return FrozenOperator.from_nodes(k, b, u, sigma), u


def test_operator_matrix_fluxes_and_probabilities():
    op, exact = manufactured(8)
    rng = np.random.default_rng(11)
    u = exact.copy()
    u[1:-1, 1:-1] = rng.normal(size=(7, 7))
    matrix, rhs = op.assemble()
    np.testing.assert_allclose(
        matrix @ u[1:-1, 1:-1].ravel() - rhs,
        (op.apply(u) - op.b[1:-1, 1:-1]).ravel(),
        atol=1e-12,
    )
    np.testing.assert_allclose(matrix.toarray(), matrix.toarray().T)
    assert np.linalg.eigvalsh(matrix.toarray()).min() > 0
    qx, qy = op.fluxes(u)
    # All interior edge contributions telescope; only boundary flux remains.
    diffusion = op.apply(u) - op.sigma[1:-1, 1:-1] * u[1:-1, 1:-1]
    boundary_flux = (qx[1:-1, -1] - qx[1:-1, 0]).sum()
    boundary_flux += (qy[-1, 1:-1] - qy[0, 1:-1]).sum()
    assert diffusion.sum() == pytest.approx(8 * boundary_flux)
    p, reward, killing = op.mc_coefficients()
    assert np.all(p > 0) and np.all(killing >= 0)
    np.testing.assert_allclose(p.sum(axis=-1) + killing, 1)
    np.testing.assert_allclose(reward, op.b[1:-1, 1:-1] / op.diagonal)
    solved = op.solve()
    np.testing.assert_allclose(op.apply(solved), op.b[1:-1, 1:-1], atol=1e-11)
    np.testing.assert_array_equal(solved[0], exact[0])
    np.testing.assert_array_equal(solved[-1], exact[-1])
    np.testing.assert_array_equal(solved[:, 0], exact[:, 0])
    np.testing.assert_array_equal(solved[:, -1], exact[:, -1])


@pytest.mark.parametrize("preset", ["sine", "polynomial", "affine", "hot_top"])
def test_legacy_adapter(preset):
    problem = make_problem(preset, 8)
    op = FrozenOperator.from_problem(problem)
    m = problem.n - 1
    line = diags([-np.ones(m - 1), 2 * np.ones(m), -np.ones(m - 1)], [-1, 0, 1])
    rhs = -problem.h**2 * problem.f[1:-1, 1:-1].copy()
    rhs[0] += problem.boundary[0, 1:-1]
    rhs[-1] += problem.boundary[-1, 1:-1]
    rhs[:, 0] += problem.boundary[1:-1, 0]
    rhs[:, -1] += problem.boundary[1:-1, -1]
    u = problem.boundary.copy()
    u[1:-1, 1:-1] = spsolve(kronsum(line, line).tocsr(), rhs.ravel()).reshape(m, m)
    np.testing.assert_allclose(op.apply(u), -laplacian(u, problem.h), atol=1e-10)
    np.testing.assert_allclose(op.solve(), u, atol=1e-10)
    p, reward, killing = op.mc_coefficients()
    np.testing.assert_array_equal(p, 0.25)
    np.testing.assert_array_equal(killing, 0)
    np.testing.assert_allclose(reward, -problem.h**2 * problem.f[1:-1, 1:-1] / 4)


def test_second_order_manufactured_variable_conductivity():
    errors = []
    defects = []
    for n in (12, 24, 48):
        op, exact = manufactured(n)
        errors.append(np.max(np.abs(op.solve() - exact)))
        defects.append(np.max(np.abs(op.apply(exact) - op.b[1:-1, 1:-1])))
    for sequence in (errors, defects):
        assert np.all((np.array(sequence[:-1]) / sequence[1:] > 3.8))
        assert np.all((np.array(sequence[:-1]) / sequence[1:] < 4.2))


@pytest.mark.parametrize("size", [1, 2, 11])
def test_variable_tridiagonal(size):
    rng = np.random.default_rng(7)
    a = -rng.uniform(0.1, 2, size)
    c = -rng.uniform(0.1, 2, size)
    b = 1 + abs(a) + abs(c)
    rhs = rng.normal(size=size)
    matrix = np.diag(b) + np.diag(a[1:], -1) + np.diag(c[:-1], 1)
    np.testing.assert_allclose(
        solve_tridiagonal(a, b, c, rhs), np.linalg.solve(matrix, rhs), atol=1e-14
    )


@pytest.mark.parametrize(
    "k,sigma", [(0, 0), (-1, 0), (np.nan, 0), (1, -1), (1, np.inf)]
)
def test_invalid_physical_coefficients(k, sigma):
    with pytest.raises(ValueError):
        FrozenOperator.from_nodes(np.full((5, 5), k), 0, 0, sigma)


def test_frozen_coefficients_copy_inputs_and_validate_shapes():
    k = np.ones((5, 5))
    op = FrozenOperator.from_nodes(k, 1, 0)
    k[:] = 20
    np.testing.assert_array_equal(op.wx, 16)
    with pytest.raises(ValueError):
        op.wx[0, 0] = 0
    with pytest.raises(ValueError):
        FrozenOperator(np.ones((5, 5)), op.wy, op.sigma, op.b, op.boundary)
