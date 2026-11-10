import numpy as np
import pytest

from pdelab.solvers import _sweep_serial


@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("size", [1, 2, 9])
@pytest.mark.parametrize("diffusion_ratio", [0.0, 0.5, 20.0])
def test_sweep_matches_direct_linear_solve(axis, size, diffusion_ratio):
    """Проверяем обе оси по независимому решению полной матрицы."""
    rhs = np.random.default_rng(42).normal(size=(size, size))
    diagonal = np.full(size, 1 + 2 * diffusion_ratio)
    off_diagonal = np.full(size - 1, -diffusion_ratio)
    matrix = (
        np.diag(diagonal) + np.diag(off_diagonal, k=-1) + np.diag(off_diagonal, k=1)
    )

    if axis == 0:
        expected = np.linalg.solve(matrix, rhs.T).T
    else:
        expected = np.linalg.solve(matrix, rhs)

    actual = _sweep_serial(rhs, diffusion_ratio, axis)

    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
