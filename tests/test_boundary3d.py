import numpy as np
import pytest
from scipy.sparse import diags

from pdelab.boundary import BoundaryCondition
from pdelab.boundary3d import (
    FACES,
    BoundaryOperator3D,
    BoundaryProblem3D,
    boundary_workspace_bytes3d,
    face_indices,
    make_boundary_problem3d,
    select_boundary_points3d,
)


@pytest.mark.parametrize("face", FACES)
@pytest.mark.parametrize("kind", ["neumann", "robin"])
def test_each_outward_normal_reproduces_polynomial(face, kind):
    kinds = dict.fromkeys(FACES, "dirichlet")
    kinds[face] = kind
    problem = make_boundary_problem3d(n=4, kinds=kinds)
    operator = BoundaryOperator3D(problem)
    np.testing.assert_allclose(
        operator.apply(problem.exact), operator.rhs(), atol=1e-12
    )
    np.testing.assert_allclose(operator.solve(), problem.exact, atol=1e-12)


def test_edge_corner_fluxes_volumes_and_markov_balance():
    problem = make_boundary_problem3d(n=4, kinds=dict.fromkeys(FACES, "robin"))
    operator = BoundaryOperator3D(problem)
    assert operator.volumes[2, 2, 2] == pytest.approx(1 / 4**3)
    assert operator.volumes[2, 2, 0] == pytest.approx(1 / (2 * 4**3))
    assert operator.volumes[2, 0, 0] == pytest.approx(1 / (4 * 4**3))
    assert operator.volumes[0, 0, 0] == pytest.approx(1 / (8 * 4**3))
    assert operator.volumes.sum() == pytest.approx(1)
    assert operator.reaction[0, 0, 0] == 3 * 2 * 4
    assert operator.boundary_load[0, 0, 0] == pytest.approx(2 * 4 * (0 - 1 - 2))
    np.testing.assert_array_equal(operator.weights[0, 0, 0], [32, 0, 32, 0, 32, 0])
    probability, reward, killing = operator.mc_coefficients()
    assert np.all(probability >= 0) and np.all(killing >= 0)
    np.testing.assert_allclose(probability.sum(axis=-1) + killing, 1)
    np.testing.assert_allclose(
        operator.apply(problem.exact), operator.rhs(), atol=1e-12
    )
    assert reward[0, 0, 0] == pytest.approx((-6 - 24) / (96 + 24))


def test_assembly_matches_stencil_and_mass_symmetry_with_variable_robin():
    n = 4
    face_array = np.arange((n + 1) ** 2).reshape(n + 1, n + 1) / 30 + 0.2
    problem = make_boundary_problem3d(
        "smooth", n, dict.fromkeys(FACES, "robin"), dict.fromkeys(FACES, face_array)
    )
    operator = BoundaryOperator3D(problem)
    matrix, rhs = operator.assemble()
    values = np.random.default_rng(123).normal(size=problem.f.shape)
    np.testing.assert_allclose(
        matrix @ values.ravel() - rhs,
        (operator.apply(values) - operator.rhs()).ravel(),
        atol=1e-12,
    )
    weighted = diags(operator.volumes.ravel()) @ matrix
    np.testing.assert_allclose((weighted - weighted.T).toarray(), 0, atol=1e-15)


def test_dirichlet_priority_and_conflicting_edges():
    problem = make_boundary_problem3d(n=4)
    operator = BoundaryOperator3D(problem)
    assert operator.fixed[0, 0, 0] and not operator.fixed[2, 2, 0]
    assert operator.reaction[0, 0, 0] == operator.boundary_load[0, 0, 0] == 0
    faces = dict(problem.faces)
    faces["z0"] = BoundaryCondition("dirichlet", 100.0)
    with pytest.raises(ValueError, match="несовместимые"):
        BoundaryOperator3D(BoundaryProblem3D(4, 0, faces))


def test_neumann_compatibility_gauge_and_immutable_face_arrays():
    problem = make_boundary_problem3d(n=4, kinds=dict.fromkeys(FACES, "neumann"))
    operator = BoundaryOperator3D(problem)
    operator.check_compatibility()
    values = operator.solve()
    assert abs(np.sum(values * operator.volumes)) < 1e-13
    np.testing.assert_allclose(values, operator.normalize(problem.exact), atol=1e-12)
    with pytest.raises(ValueError, match="баланс"):
        BoundaryOperator3D(BoundaryProblem3D(4, 7.0, dict(problem.faces))).solve()
    with pytest.raises(ValueError):
        problem.faces["x0"].value[0, 0] = 0


@pytest.mark.parametrize(
    "kinds",
    [
        dict(
            zip(
                FACES,
                ["neumann", "dirichlet", "robin", "dirichlet", "neumann", "robin"],
            )
        ),
        dict.fromkeys(FACES, "neumann"),
        dict.fromkeys(FACES, "robin"),
    ],
)
def test_smooth_reference_converges_with_second_order(kinds):
    errors = []
    for n in (8, 16):
        problem = make_boundary_problem3d("smooth", n, kinds)
        operator = BoundaryOperator3D(problem)
        errors.append(
            np.max(np.abs(operator.solve() - operator.normalize(problem.exact)))
        )
    assert 3.8 < errors[0] / errors[1] < 4.5


@pytest.mark.parametrize("plane,axis", [("xy", 0), ("xz", 1), ("yz", 2)])
def test_queries_include_boundary_faces_edges_and_corners(plane, axis):
    points = select_boundary_points3d(8, "slice", plane=plane, index=0)
    assert len(points) == 9**2 and np.all(points[:, axis] == 0)
    assert len(select_boundary_points3d(8, "field")) == 9**3
    assert len(select_boundary_points3d(8, "patch", point=(0, 0, 0))) == 8
    np.testing.assert_array_equal(
        select_boundary_points3d(8, point=(1, 2, 3)), [[3, 2, 1]]
    )


def test_face_array_coordinate_order_and_direct_size_guard():
    problem = make_boundary_problem3d(n=4, kinds=dict.fromkeys(FACES, "dirichlet"))
    for face, index in face_indices().items():
        np.testing.assert_array_equal(problem.faces[face].value, problem.exact[index])
    with pytest.raises(ValueError, match="Прямой 3D"):
        BoundaryOperator3D(make_boundary_problem3d(n=24)).assemble()
    with pytest.raises(ValueError):
        make_boundary_problem3d(n=10000)
    assert boundary_workspace_bytes3d(64) < 150 * 1024**2
