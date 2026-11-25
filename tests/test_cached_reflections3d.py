"""Cached stationary walks must reproduce the six-neighbor finite-volume law."""

from dataclasses import replace

import numpy as np
import pytest
from scipy.sparse.linalg import spsolve

from pdelab.boundary import BoundaryCondition
from pdelab.boundary3d import (
    FACES,
    BoundaryOperator3D,
    BoundaryProblem3D,
    make_boundary_problem3d,
)
from pdelab.boundary_solvers3d import solve_boundary3d
from pdelab.grouped_solver3d import GroupedWalkFailure
from pdelab.reflections3d import reflected_step, reflection_table

CASES = [("x0",), ("x0", "z0", "x1"), FACES]
METHODS = [("mc", "mc_cached"), ("mc_grouped", "mc_grouped_cached")]
# Array coordinates are (z, y, x). Start on a face, then visit edge, corner,
# interior and the opposite corner; the latter can be fixed or reflecting.
POINTS = np.array([[2, 2, 0], [2, 0, 0], [0, 0, 0], [2, 2, 2], [4, 4, 4]])


def _problem(robin_faces, n=4):
    a, b = np.indices((n + 1, n + 1))
    kinds = dict.fromkeys(FACES, "dirichlet")
    kinds["y0"] = "neumann"
    kinds.update(dict.fromkeys(robin_faces, "robin"))
    kappas = {face: (1 + i + a + 2 * b) / 7 for i, face in enumerate(robin_faces)}
    problem = make_boundary_problem3d("smooth", n, kinds, kappas)
    return replace(problem, exact=None)


@pytest.mark.parametrize("robin_faces", CASES)
def test_cached_row_matches_physical_stencil_at_every_node(robin_faces):
    problem = _problem(robin_faces)
    operator = BoundaryOperator3D(problem)
    table = reflection_table(problem.n)
    reward, killing = operator.mc_payoffs()
    assert np.all((0 <= killing) & (killing <= 1))
    np.testing.assert_array_equal(
        reward[operator.fixed], operator.boundary[operator.fixed]
    )
    np.testing.assert_array_equal(killing[operator.fixed], 1.0)
    for pos in zip(*np.nonzero(~operator.fixed), strict=True):
        diagonal = operator.diagonal[pos]
        assert reward[pos] == pytest.approx(operator.b[pos] / diagonal)
        assert killing[pos] == pytest.approx(operator.reaction[pos] / diagonal)
        expected = {}
        for direction, weight in enumerate(operator.weights[pos]):
            if weight > 0:
                neighbor = list(pos)
                neighbor[direction // 2] += 1 if direction % 2 == 0 else -1
                expected[tuple(neighbor)] = weight / diagonal
        observed = {}
        for direction in range(6):
            neighbor = reflected_step(*pos, problem.n, direction, table)
            observed[neighbor] = observed.get(neighbor, 0.0) + (1 - killing[pos]) / 6
        assert observed == pytest.approx(expected)
        assert sum(observed.values()) + killing[pos] == pytest.approx(1.0)


@pytest.mark.parametrize("ordinary,cached", METHODS)
@pytest.mark.parametrize("robin_faces", CASES)
def test_same_seed_preserves_samples_errors_and_path_events(
    ordinary, cached, robin_faces
):
    problem = _problem(robin_faces)
    settings = dict(walks=73, seed=10321, batch_size=7)
    base = solve_boundary3d(problem, POINTS, ordinary, **settings)
    result = solve_boundary3d(problem, POINTS, cached, **settings)
    for key in ("values", "stderr", "sample_count"):
        np.testing.assert_array_equal(getattr(result, key), getattr(base, key))
    np.testing.assert_array_equal(result.extras["samples"], base.extras["samples"])
    fixed = BoundaryOperator3D(problem).fixed[tuple(POINTS.T)]
    assert (
        result.extras["started_walks"] == result.walks == 73 * np.count_nonzero(~fixed)
    )
    assert result.steps == base.steps
    assert result.extras["transitions"] == result.steps


@pytest.mark.parametrize("method", [cached for _, cached in METHODS])
def test_independent_ensembles_agree_with_small_sparse_solution(method):
    problem = _problem(FACES)
    matrix, rhs = BoundaryOperator3D(problem).assemble()
    target = spsolve(matrix, rhs).reshape(problem.f.shape)[tuple(POINTS.T)]
    means, variances = [], []
    for seed in (541, 717, 1823, 9199):
        result = solve_boundary3d(
            problem, POINTS, method, walks=768, seed=seed, batch_size=11
        )
        means.append(result.values[tuple(POINTS.T)])
        variances.append(result.stderr[tuple(POINTS.T)] ** 2)
    # Independent seeds give Var(mean over runs) = sum(Var(run)) / R².
    uncertainty = np.sqrt(np.sum(variances, axis=0)) / len(means)
    assert np.all(np.abs(np.mean(means, axis=0) - target) < 6 * uncertainty)


@pytest.mark.parametrize("ordinary,cached", METHODS)
def test_changed_boundary_data_coefficients_and_source_use_current_operator(
    ordinary, cached
):
    problem = _problem(("x0",))
    face = problem.faces["x0"]
    changed_face = BoundaryCondition("robin", 3 + face.value, 2 * face.kappa)
    changed = replace(problem, faces={**problem.faces, "x0": changed_face})
    samples = []
    for current in (problem, changed, replace(changed, f=changed.f + 12), problem):
        settings = dict(walks=67, seed=122, batch_size=7)
        result = solve_boundary3d(current, POINTS[:1], cached, **settings)
        base = solve_boundary3d(current, POINTS[:1], ordinary, **settings)
        samples.append(result.extras["samples"])
        np.testing.assert_array_equal(samples[-1], base.extras["samples"])
    assert not np.array_equal(samples[0], samples[1])
    assert not np.array_equal(samples[1], samples[2])
    np.testing.assert_array_equal(samples[-1], samples[0])


@pytest.mark.parametrize("ordinary,cached", METHODS)
def test_reflections_follow_new_grid_and_face_types(ordinary, cached):
    for n, robin_faces in [(4, ("x1",)), (7, FACES), (4, ("x1",))]:
        problem = _problem(robin_faces, n)
        starts = [[n // 2, n // 2, n], [n, n, n]]
        settings = dict(walks=41, seed=101, batch_size=7)
        result = solve_boundary3d(problem, starts, cached, **settings)
        base = solve_boundary3d(problem, starts, ordinary, **settings)
        np.testing.assert_array_equal(result.extras["samples"], base.extras["samples"])
        np.testing.assert_array_equal(result.sample_count, base.sample_count)
        assert result.steps == base.steps


@pytest.mark.parametrize("ordinary,cached", METHODS)
def test_zero_beta_matches_reflecting_flux_and_requires_absorption(ordinary, cached):
    problem = _problem(("x0",))
    value = problem.faces["x0"].value
    reflecting = replace(
        problem, faces={**problem.faces, "x0": BoundaryCondition("neumann", value)}
    )
    zero_beta = replace(
        problem, faces={**problem.faces, "x0": BoundaryCondition("robin", value, 0)}
    )
    settings = dict(walks=91, seed=1357, batch_size=11)
    base = solve_boundary3d(reflecting, POINTS, ordinary, **settings)
    result = solve_boundary3d(zero_beta, POINTS, cached, **settings)
    np.testing.assert_array_equal(result.extras["samples"], base.extras["samples"])
    assert result.steps == base.steps
    nonabsorbing = BoundaryProblem3D(
        4, 0, dict.fromkeys(FACES, BoundaryCondition("robin", 0, 0))
    )
    with pytest.raises(ValueError):
        solve_boundary3d(nonabsorbing, POINTS, cached, **settings)


@pytest.mark.parametrize(
    "method,unfinished", [("mc_cached", 1), ("mc_grouped_cached", 7)]
)
def test_limit_keeps_completed_paths_and_counts_absorption_stages(method, unfinished):
    # The large finite beta rounds the first face's kill probability to 1.
    # Every path from the face completes after one paid stage and zero moves.
    faces = dict.fromkeys(FACES, BoundaryCondition("dirichlet", 2))
    faces["x0"] = BoundaryCondition("robin", 2**60, 2**60)
    problem = BoundaryProblem3D(4, 0, faces)
    points = [[2, 2, 0], [2, 2, 2]]
    with pytest.raises(GroupedWalkFailure) as caught:
        solve_boundary3d(problem, points, method, walks=37, batch_size=7, max_steps=1)
    assert caught.value.status == "path_limit"
    assert not hasattr(caught.value, "values")
    work = caught.value.diagnostics
    assert work["walks"] == 37
    assert work["started_walks"] == work["transitions"] == 37 + unfinished
