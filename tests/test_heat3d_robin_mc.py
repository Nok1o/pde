"""PDE-27: existing frozen MC kernels use the physical mixed heat operator."""

import numpy as np
import pytest
from scipy.sparse.linalg import spsolve

from pdelab.boundary3d import FACES
from pdelab.heat3d_contract import NEIGHBOR_OFFSETS_ZYX
from pdelab.heat3d_mc import frozen_mc_coefficients, solve_frozen_mc3d
from pdelab.heat3d_operator import HeatOperator3D, freeze_heat3d
from pdelab.heat3d_problem import make_heat_problem3d

_ROBIN_FACES = {
    "all1": (),
    "one3": ("y0",),
    "several3": ("x0", "y0", "z0"),
    "all3": FACES,
}
_DT = 0.013


def _operator(configuration, alpha):
    kinds = {face: 3 if face in _ROBIN_FACES[configuration] else 1 for face in FACES}
    betas = {face: (index + 1) / 3 for index, face in enumerate(FACES)}
    problem = make_heat_problem3d("polynomial", 4, alpha, kinds=kinds, beta=betas)
    previous = problem.initial_values()
    # A non-exact current iterate makes the edge conductivities unequal.
    iterate = previous + 0.1 * np.indices(problem.shape)[0]
    return problem, freeze_heat3d(problem, previous, iterate, _DT, _DT), iterate


@pytest.mark.parametrize("configuration", _ROBIN_FACES)
@pytest.mark.parametrize("alpha", [0, 0.5, 2])
def test_robin_markov_identity_and_physical_absorption(configuration, alpha):
    problem, op, iterate = _operator(configuration, alpha)
    probabilities, reward, killing = frozen_mc_coefficients(op)
    np.testing.assert_allclose(probabilities.sum(axis=-1) + killing, 1, atol=2e-15)
    assert np.all(probabilities >= 0) and np.all(killing > 0)

    # Recover the absorption independently from the physical terms, including
    # the pre-elimination couplings to known nodes. No k multiplies beta or r.
    iterate[op.fixed] = op.fixed_values[op.fixed]
    conductivity = 1 + alpha * iterate**2
    expected = np.ones(op.shape)
    for point in np.ndindex(op.shape):
        if op.fixed[point]:
            continue
        reaction = sum(
            2 * problem.n * problem.faces[f"{axis}{int(coordinate == problem.n)}"].beta
            for axis, coordinate in zip("zyx", point)
            if coordinate in (0, problem.n)
        )
        eliminated = 0.0
        for offset in NEIGHBOR_OFFSETS_ZYX:
            neighbor = tuple(i + step for i, step in zip(point, offset))
            if min(neighbor) < 0 or max(neighbor) > problem.n or not op.fixed[neighbor]:
                continue
            axis = next(axis for axis, step in enumerate(offset) if step)
            geometry = 2 if point[axis] in (0, problem.n) else 1
            eliminated += (
                geometry
                * problem.n**2
                * (conductivity[point] + conductivity[neighbor])
                / 2
            )
        expected[point] = (1 / _DT + reaction + eliminated) / op.diagonal[point]
    np.testing.assert_allclose(killing, expected, rtol=2e-14, atol=2e-15)

    field = np.random.default_rng(27).normal(size=op.shape)
    markov = reward.copy()
    for direction, offset in enumerate(NEIGHBOR_OFFSETS_ZYX):
        markov += probabilities[..., direction] * np.roll(
            field, tuple(-step for step in offset), axis=(0, 1, 2)
        )
    matrix, rhs = op.assemble()
    applied = (matrix @ field.ravel()).reshape(op.shape)
    np.testing.assert_allclose(applied, op.apply(field), rtol=2e-14, atol=2e-12)
    np.testing.assert_allclose(
        markov, field - (applied - rhs.reshape(op.shape)) / op.diagonal, atol=3e-15
    )

    if configuration == "all3":
        assert not op.fixed.any()
        # The mass remains at every node, so the all-Robin time layer has no
        # constant null mode even away from the boundary reaction.
        assert np.all(op.apply(np.ones(op.shape)) > 0)
        assert np.all(killing >= (1 / _DT) / op.diagonal - 2e-15)


@pytest.mark.parametrize("configuration", ["several3", "all3"])
@pytest.mark.parametrize("alpha", [0, 0.5, 2])
@pytest.mark.parametrize("batch_size", [None, 7])
def test_independent_robin_mc_runs_match_the_weighted_csr(
    configuration, alpha, batch_size
):
    _, op, _ = _operator(configuration, alpha)
    # Corner, edge, face, interior, and a fixed point in the mixed case.
    points = np.array([[0, 0, 0], [0, 0, 2], [2, 0, 2], [2, 2, 2], [4, 4, 4]])
    matrix, rhs = op.assemble()
    target = spsolve(matrix, rhs).reshape(op.shape)[tuple(points.T)]
    results = [
        solve_frozen_mc3d(op, points, walks=2048, seed=seed, batch_size=batch_size)
        for seed in range(16)
    ]
    assert all(result.status == "completed" for result in results)
    values = np.array([result.values for result in results])
    uncertainty = values.std(axis=0, ddof=1) / np.sqrt(len(results))
    assert np.all(np.abs(values.mean(axis=0) - target) < 6 * uncertainty + 1e-12)
    observed = values[:, 2].std(ddof=1)
    predicted = np.sqrt(np.mean([result.stderr[2] ** 2 for result in results]))
    assert 0.3 < observed / predicted < 2.1
    known = op.fixed[tuple(points.T)]
    for result in results:
        assert result.walks == 2048 * np.count_nonzero(~known)
        np.testing.assert_array_equal(result.completed_paths[known], 0)
        np.testing.assert_array_equal(result.stderr[known], 0)


@pytest.mark.parametrize("configuration", ["several3", "all3"])
@pytest.mark.parametrize("batch_size", [None, 7])
def test_robin_correction_has_no_repeated_boundary_payment(
    configuration, batch_size, monkeypatch
):
    _, physical, _ = _operator(configuration, 0.5)
    z, y, x = np.indices(physical.shape) / physical.n
    field = 2 + x + y * z
    control = 4 - z + x * y
    op = physical.with_rhs(physical.apply(field))
    saved_rhs = op.rhs.copy()
    defect = op.with_rhs(op.rhs - op.apply(control))
    expected = field - control
    np.testing.assert_allclose(defect.fixed_values[op.fixed], expected[op.fixed])
    assert defect.diagonal is op.diagonal and defect.weights is op.weights

    def forbidden(*args, **kwargs):
        pytest.fail("Frozen MC must not solve a deterministic reference internally")

    monkeypatch.setattr(HeatOperator3D, "assemble", forbidden)
    points = np.array([[0, 0, 0], [2, 0, 2], [2, 2, 2], [4, 4, 4]])
    result = solve_frozen_mc3d(
        defect, points, walks=20000, seed=27, batch_size=batch_size
    )
    assert result.status == "completed"
    assert np.all(
        np.abs(result.values - expected[tuple(points.T)]) < 6 * result.stderr + 1e-12
    )
    # In particular, physical g and r are not paid again when the adjusted
    # right-hand side has been replaced by zero.
    zero = solve_frozen_mc3d(
        op.with_rhs(np.zeros(op.shape)),
        points,
        walks=31,
        seed=27,
        batch_size=batch_size,
    )
    assert zero.status == "completed"
    np.testing.assert_array_equal(zero.values, 0)
    np.testing.assert_array_equal(zero.stderr, 0)
    np.testing.assert_array_equal(op.rhs, saved_rhs)
    assert not op.rhs.flags.writeable and not defect.rhs.flags.writeable
