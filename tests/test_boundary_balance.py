from math import fsum

import numpy as np
import pytest

from pdelab.boundary import (
    SIDES,
    BoundaryCondition,
    BoundaryOperator,
    BoundaryProblem,
    make_boundary_problem,
)
from pdelab.boundary3d import (
    FACES,
    BoundaryOperator3D,
    BoundaryProblem3D,
    make_boundary_problem3d,
)
from pdelab.boundary_solvers import solve_boundary
from pdelab.boundary_solvers3d import solve_boundary3d


@pytest.mark.parametrize(
    "dimension,faces,make,solve",
    [
        (2, SIDES, make_boundary_problem, solve_boundary),
        (3, FACES, make_boundary_problem3d, solve_boundary3d),
    ],
)
def test_weak_robin_convergence_preserves_constant_component(
    dimension, faces, make, solve
):
    problem = make(
        n=8, kinds=dict.fromkeys(faces, "robin"), kappas=dict.fromkeys(faces, 1e-8)
    )
    result = solve(problem, [[4] * dimension], tol=1e-8)
    assert result.converged and result.relative_residual <= 1e-8
    np.testing.assert_allclose(result.values, problem.exact, atol=2e-7, rtol=0)


@pytest.mark.parametrize(
    "dimension,faces,make,operator_type,solve",
    [
        (2, SIDES, make_boundary_problem, BoundaryOperator, solve_boundary),
        (3, FACES, make_boundary_problem3d, BoundaryOperator3D, solve_boundary3d),
    ],
)
def test_weak_variable_robin_with_neumann_faces(
    dimension, faces, make, operator_type, solve
):
    kinds = dict.fromkeys(faces, "neumann")
    kinds[faces[0]] = "robin"
    kappa = 1e-8 * (
        1
        + np.arange(9 ** (dimension - 1)).reshape((9,) * (dimension - 1))
        / 9 ** (dimension - 1)
    )
    problem = make(n=8, kinds=kinds, kappas={faces[0]: kappa})
    operator = operator_type(problem)
    result = solve(problem, [[4] * dimension], tol=1e-8)
    assert result.converged
    np.testing.assert_allclose(result.values, problem.exact, atol=3e-7, rtol=0)
    np.testing.assert_allclose(operator.solve(), problem.exact, atol=1e-8, rtol=0)
    # This checks the physical balance independently of the LOM stopping test.
    balance = fsum((operator.volumes * operator.b).ravel())
    uptake = np.sum(operator.volumes * operator.reaction * result.values)
    assert uptake == pytest.approx(balance, rel=1e-14)


@pytest.mark.parametrize(
    "dimension,faces,problem_type,operator_type,solve",
    [
        (2, SIDES, BoundaryProblem, BoundaryOperator, solve_boundary),
        (3, FACES, BoundaryProblem3D, BoundaryOperator3D, solve_boundary3d),
    ],
)
@pytest.mark.parametrize("kappa", [1e-8, 1e-12])
def test_constant_solution_has_stable_residual_and_direct_reference(
    dimension, faces, problem_type, operator_type, solve, kappa
):
    # No analytic solution is supplied to either numerical solver.
    problem = problem_type(
        8, 0, {face: BoundaryCondition("robin", 7 * kappa, kappa) for face in faces}
    )
    result = solve(problem, [[4] * dimension], tol=1e-8, max_iter=100)
    assert result.converged
    np.testing.assert_allclose(result.values, 7, atol=1e-13, rtol=0)
    np.testing.assert_allclose(operator_type(problem).solve(), 7, atol=1e-12, rtol=0)
