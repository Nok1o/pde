"""Physical Robin fluxes, conservative dual cells and shared frozen backends."""

from dataclasses import replace

import numpy as np
import pytest
from scipy.sparse.linalg import spsolve

from pdelab.boundary3d import FACES
from pdelab.heat3d_block import BlockThomasSolver3D
from pdelab.heat3d_linear import solve_lines3d
from pdelab.heat3d_mc import frozen_mc_coefficients
from pdelab.heat3d_operator import (
    IncompatibleHeatBoundaryError,
    dirichlet_data3d,
    freeze_heat3d,
)
from pdelab.heat3d_problem import HeatFace3D, HeatProblem3D, make_heat_problem3d


def _problem(essential=(), alpha=0.6, gamma=0.0, sigma=0.0):
    def initial(x, y, z, t):
        return 1 + x + 2 * y + 3 * z + t

    return HeatProblem3D(
        "independent Robin data",
        4,
        alpha,
        initial,
        lambda x, y, z, t: 3 + t,
        {
            face: (
                HeatFace3D(1, initial)
                if face in essential
                else HeatFace3D(3, lambda x, y, z, t, i=i: 7 + i + t, 0.25 + i / 2)
            )
            for i, face in enumerate(FACES)
        },
        capacity_gamma=gamma,
        reaction_sigma=sigma,
    )


def _incident_faces(node, n):
    for axis, coordinate in enumerate(node):
        if coordinate in (0, n):
            yield "zyx"[axis] + str(int(coordinate == n))


def _independent_system(problem, previous, iterate, t, dt):
    """Scalar control-volume rows, independent of production slicing helpers."""
    shape, n = problem.shape, problem.n
    fixed = np.zeros(shape, dtype=bool)
    boundary = np.zeros(shape)
    reaction = np.zeros(shape)
    load = problem.evaluate(problem.source, t).copy()
    field = iterate.copy()
    for node in np.ndindex(shape):
        z, y, x = np.array(node) / n
        for face in _incident_faces(node, n):
            condition = problem.faces[face]
            datum = condition.value(x, y, z, t)
            if condition.kind == 1:
                fixed[node], boundary[node] = True, datum
            else:
                reaction[node] += 2 * n * condition.beta
                load[node] += 2 * n * datum
    field[fixed] = boundary[fixed]
    k = 1 + problem.alpha * field**2
    mass = (1 + problem.capacity_gamma * field**2) / dt
    loss = problem.reaction_sigma * field**2
    load += mass * previous
    matrix = np.zeros((field.size, field.size))
    rhs = load.ravel().copy()
    for node in np.ndindex(shape):
        row = np.ravel_multi_index(node, shape)
        if fixed[node]:
            matrix[row, row], rhs[row] = 1, boundary[node]
            continue
        matrix[row, row] = mass[node] + loss[node] + reaction[node]
        for axis in range(3):
            for step in (-1, 1):
                neighbor = list(node)
                neighbor[axis] += step
                if neighbor[axis] not in range(n + 1):
                    continue
                neighbor = tuple(neighbor)
                width = 0.5 if node[axis] in (0, n) else 1
                weight = (k[node] + k[neighbor]) * n**2 / (2 * width)
                matrix[row, row] += weight
                if fixed[neighbor]:
                    rhs[row] += weight * boundary[neighbor]
                else:
                    matrix[row, np.ravel_multi_index(neighbor, shape)] = -weight
    return matrix, rhs


@pytest.mark.parametrize("essential", [(), ("x0",), ("x0", "y1", "z0"), FACES])
@pytest.mark.parametrize("gamma,sigma", [(0.0, 0.0), (0.7, 0.3)])
def test_scalar_dual_volume_rows_include_mixed_elimination_and_all_kind1(
    essential, gamma, sigma
):
    problem = _problem(essential, gamma=gamma, sigma=sigma)
    rng = np.random.default_rng(26)
    previous = rng.uniform(0.5, 1.5, problem.shape)
    iterate = rng.uniform(-1, 2, problem.shape)
    original = iterate.copy()
    operator = freeze_heat3d(problem, previous, iterate, 0.03, 0.02)
    expected_matrix, expected_rhs = _independent_system(
        problem, previous, iterate, 0.03, 0.02
    )
    matrix, rhs = operator.assemble()
    np.testing.assert_allclose(
        matrix.toarray(), expected_matrix, rtol=3e-15, atol=1e-12
    )
    np.testing.assert_allclose(rhs, expected_rhs, rtol=3e-15, atol=1e-12)
    np.testing.assert_allclose(
        operator.apply(iterate).ravel(), expected_matrix @ iterate.ravel(), atol=2e-12
    )
    np.testing.assert_array_equal(iterate, original)
    for row in np.flatnonzero(operator.fixed.ravel()):
        assert matrix[row].nnz == matrix[:, row].nnz == 1


@pytest.mark.parametrize(
    "node,directions,faces",
    [
        ((2, 2, 0), (4,), ("x0",)),
        ((2, 2, 4), (5,), ("x1",)),
        ((2, 0, 2), (2,), ("y0",)),
        ((2, 4, 2), (3,), ("y1",)),
        ((0, 2, 2), (0,), ("z0",)),
        ((4, 2, 2), (1,), ("z1",)),
        ((2, 0, 4), (2, 5), ("x1", "y0")),
        ((4, 0, 4), (1, 2, 5), ("x1", "y0", "z1")),
    ],
)
def test_hand_rows_double_only_normal_weights_and_sum_incident_physical_fluxes(
    node, directions, faces
):
    problem = _problem()
    # Nonunit k=1+0.6*2**2 detects accidentally multiplying beta/r by k.
    iterate = np.full(problem.shape, 2.0)
    previous = np.ones(problem.shape)
    t, dt = 0.02, 0.01
    operator = freeze_heat3d(problem, previous, iterate, t, dt)
    base_weight = (1 + problem.alpha * 4) * problem.n**2
    weights = operator.weights[node]
    np.testing.assert_allclose(weights[list(directions)], 2 * base_weight)
    for direction in range(6):
        if direction not in directions:
            expected = 0 if (direction ^ 1) in directions else base_weight
            assert weights[direction] == pytest.approx(expected)
    beta = sum(problem.faces[face].beta for face in faces)
    r = sum(7 + FACES.index(face) + t for face in faces)
    assert operator.diagonal[node] == pytest.approx(1 / dt + 6 * base_weight + 8 * beta)
    assert operator.rhs[node] == pytest.approx(1 / dt + 3 + t + 8 * r)


def test_all_robin_conserves_internal_flux_and_full_volume_weighted_balance():
    problem = _problem()
    rng = np.random.default_rng(7)
    previous = rng.uniform(0.1, 2, problem.shape)
    iterate = rng.uniform(-2, 3, problem.shape)
    values = rng.uniform(-1, 1, problem.shape)
    t, dt = 0.03, 0.02
    operator = freeze_heat3d(problem, previous, iterate, t, dt)
    assert not operator.fixed.any()
    reaction = np.zeros(problem.shape)
    boundary_balance = 0.0
    for node in np.ndindex(problem.shape):
        for face in _incident_faces(node, problem.n):
            condition = problem.faces[face]
            area = operator.volumes[node] * 2 * problem.n
            r = 7 + FACES.index(face) + t
            reaction[node] += 2 * problem.n * condition.beta
            boundary_balance += area * (condition.beta * values[node] - r)
    diffusion = operator.apply(values) - (1 / dt + reaction) * values
    assert np.sum(operator.volumes * diffusion) == pytest.approx(0, abs=2e-13)
    actual = np.sum(operator.volumes * (operator.apply(values) - operator.rhs))
    expected = (
        np.sum(
            operator.volumes
            * ((values - previous) / dt - problem.evaluate(problem.source, t))
        )
        + boundary_balance
    )
    assert actual == pytest.approx(expected, abs=2e-13)
    np.testing.assert_allclose(operator.volumes.sum(), 1)
    matrix, _ = operator.assemble()
    weighted = operator.volumes.ravel()[:, None] * matrix.toarray()
    np.testing.assert_allclose(weighted, weighted.T, rtol=2e-15, atol=2e-15)
    probabilities, _, absorption = frozen_mc_coefficients(operator)
    assert np.all(probabilities >= 0) and np.all(absorption > 0)
    np.testing.assert_allclose(
        absorption, (1 / dt + reaction) / operator.diagonal, atol=3e-16
    )


def test_kind1_priority_and_only_essential_intersections_are_checked():
    problem = _problem(("x0", "y0"))
    fixed, values = dirichlet_data3d(problem, 0.1)
    assert fixed[0, 0, 0] and fixed[2, 0, 4] and fixed[4, 2, 0]
    assert not fixed[0, 2, 2]
    np.testing.assert_array_equal(values[~fixed], 0)
    faces = dict(problem.faces)
    faces["y0"] = HeatFace3D(1, lambda x, y, z, t: 1 + x + 3 * z + 2 * t)
    conflicting = replace(problem, faces=faces)
    dirichlet_data3d(conflicting, 0)
    with pytest.raises(IncompatibleHeatBoundaryError):
        dirichlet_data3d(conflicting, 0.1)


def test_conductivity_uses_iterate_with_fixed_values_overwritten_and_never_exact():
    problem = _problem(("x0",))

    def forbidden(*args):
        raise AssertionError("Exact solution accessed by frozen operator")

    problem = replace(problem, exact=forbidden)
    previous = problem.initial_values()
    iterate = previous + 0.2
    changed = iterate.copy()
    changed[:, :, 0] = -100
    operator = freeze_heat3d(problem, previous, iterate, 0.02, 0.02)
    same = freeze_heat3d(problem, previous, changed, 0.02, 0.02)
    np.testing.assert_array_equal(operator.diagonal, same.diagonal)
    np.testing.assert_array_equal(operator.weights, same.weights)
    np.testing.assert_array_equal(operator.rhs, same.rhs)
    changed[2, 2, 2] += 1
    different = freeze_heat3d(problem, previous, changed, 0.02, 0.02)
    assert different.diagonal[2, 2, 2] != operator.diagonal[2, 2, 2]


@pytest.mark.parametrize("alpha", [0.0, 0.7])
def test_manufactured_robin_data_use_each_beta_and_opposite_outward_normals(alpha):
    betas = {face: (i + 1) / 4 for i, face in enumerate(FACES)}
    problem = make_heat_problem3d(
        "polynomial", 4, alpha, kinds=dict.fromkeys(FACES, 3), beta=betas
    )
    t = 0.06
    for face in FACES:
        x, y, z = 0.5, 0.5, 0.5
        if face[0] == "x":
            x = float(face[1])
        elif face[0] == "y":
            y = float(face[1])
        else:
            z = float(face[1])
        u = (
            1
            + t
            + (x + 2 * y + 3 * z + x * y + x * z + y * z + x * x + y * y + z * z) / 4
        )
        gradient = ((1 + 2 * x + y + z), (2 + x + 2 * y + z), (3 + x + y + 2 * z))
        du = gradient["xyz".index(face[0])] / 4
        normal = -du if face[1] == "0" else du
        expected = (1 + alpha * u * u) * normal + betas[face] * u
        assert problem.face_values(face, t)[2, 2] == pytest.approx(expected)
        assert problem.faces[face].beta == betas[face]
    betas["x0"] = 999
    assert problem.faces["x0"].beta == 0.25
    # The factory copies mapping input, including the closure used for r.
    assert problem.face_values("x0", t)[2, 2] < 100


def test_beta_mapping_validation_and_scalar_compatibility():
    kinds = dict.fromkeys(FACES, 3)
    with pytest.raises(ValueError, match="beta всех шести"):
        make_heat_problem3d(n=4, kinds=kinds, beta={"x0": 1})
    for invalid in (0, -1, np.inf, np.nan):
        betas = dict.fromkeys(FACES, 1)
        betas["z1"] = invalid
        with pytest.raises(ValueError, match="beta"):
            make_heat_problem3d(n=4, kinds=kinds, beta=betas)
    scalar = make_heat_problem3d(n=4, kinds=kinds, beta=1.5)
    mapping = make_heat_problem3d(n=4, kinds=kinds, beta=dict.fromkeys(FACES, 1.5))
    for face in FACES:
        np.testing.assert_array_equal(
            scalar.face_values(face, 0.1), mapping.face_values(face, 0.1)
        )


@pytest.mark.parametrize("essential", [(), ("x0", "z1")])
def test_robin_frozen_lines_and_block_solve_same_csr(essential):
    problem = _problem(essential)
    previous = problem.initial_values()
    operator = freeze_heat3d(problem, previous, previous + 0.2, 0.01, 0.01)
    matrix, rhs = operator.assemble()
    expected = spsolve(matrix, rhs).reshape(problem.shape)
    line = solve_lines3d(operator, previous, tolerance=1e-12, max_iterations=500)
    block = BlockThomasSolver3D()(operator, previous, tolerance=1e-12, max_iterations=1)
    for result in (line, block):
        assert result.status == "completed"
        np.testing.assert_allclose(result.values, expected, rtol=2e-11, atol=2e-11)
