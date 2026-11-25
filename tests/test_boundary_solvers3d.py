import numpy as np
import pytest

from pdelab.boundary import BoundaryCondition
from pdelab.boundary3d import (
    FACES,
    BoundaryOperator3D,
    BoundaryProblem3D,
    make_boundary_problem3d,
    select_boundary_points3d,
)
from pdelab.boundary_solvers3d import (
    _reflected_step,
    solve_boundary3d,
    trace_boundary_walk3d,
    warmup_boundary3d,
)
from pdelab.problem3d import make_problem3d
from pdelab.solvers3d import solve3d


@pytest.mark.parametrize("face", FACES)
@pytest.mark.parametrize("kind", ["neumann", "robin"])
def test_lod_three_sweeps_match_direct(face, kind):
    kinds = dict.fromkeys(FACES, "dirichlet")
    kinds[face] = kind
    problem = make_boundary_problem3d("smooth", 6, kinds)
    result = solve_boundary3d(problem, [[3, 3, 3]], tol=1e-10)
    assert result.converged and result.relative_residual <= 1e-10
    np.testing.assert_allclose(
        result.values, BoundaryOperator3D(problem).solve(), atol=5e-8
    )


@pytest.mark.parametrize("kind", ["neumann", "robin"])
def test_lod_all_derivative_faces_and_variable_robin(kind):
    kappa = 0.1 + np.arange(49).reshape(7, 7) / 10
    problem = make_boundary_problem3d(
        "smooth", 6, dict.fromkeys(FACES, kind), dict.fromkeys(FACES, kappa)
    )
    operator = BoundaryOperator3D(problem)
    result = solve_boundary3d(problem, [[0, 0, 0]], tol=1e-10)
    assert result.converged
    np.testing.assert_allclose(result.values, operator.solve(), atol=2e-8)
    if kind == "neumann":
        assert abs(np.sum(result.values * operator.volumes)) < 1e-13


def test_relative_stopping_scaling_and_iteration_limit():
    problem = make_boundary_problem3d(n=6)
    scaled = BoundaryProblem3D(
        6,
        problem.f * 1e6,
        {
            s: BoundaryCondition(c.kind, c.value * 1e6, c.kappa)
            for s, c in problem.faces.items()
        },
    )
    first = solve_boundary3d(problem, [[3, 3, 3]], tol=1e-7)
    second = solve_boundary3d(scaled, [[3, 3, 3]], tol=1e-7)
    assert first.iterations == second.iterations
    np.testing.assert_allclose(second.values / 1e6, first.values, atol=1e-12)
    limited = solve_boundary3d(problem, [[3, 3, 3]], max_iter=1)
    assert not limited.converged and limited.iterations == 1


@pytest.mark.parametrize(
    "kinds",
    [
        dict(
            zip(
                FACES,
                [
                    "neumann",
                    "dirichlet",
                    "neumann",
                    "dirichlet",
                    "neumann",
                    "dirichlet",
                ],
            )
        ),
        dict.fromkeys(FACES, "robin"),
    ],
)
def test_mc_independent_ensembles_at_interior_face_edge_and_corner(kinds):
    problem = make_boundary_problem3d("smooth", 6, kinds)
    points = np.array([[3, 3, 3], [3, 3, 0], [3, 0, 0], [0, 0, 0]])
    target = BoundaryOperator3D(problem).solve()[tuple(points.T)]
    means = np.array(
        [
            solve_boundary3d(problem, points, "mc", walks=4096, seed=seed).values[
                tuple(points.T)
            ]
            for seed in range(64)
        ]
    )
    uncertainty = means.std(axis=0, ddof=1) / np.sqrt(len(means))
    assert np.all(np.abs(means.mean(axis=0) - target) < 5 * uncertainty + 1e-10)


@pytest.mark.parametrize("method", ["lod_serial", "mc"])
def test_all_dirichlet_delegates_without_changing_rng_or_values(method):
    problem = make_problem3d("sine", 6)
    points = select_boundary_points3d(6, "field")
    old = solve3d(problem, points, method, walks=64, seed=913, tol=1e-8)
    new = solve_boundary3d(
        BoundaryProblem3D.from_problem(problem),
        points,
        method,
        walks=64,
        seed=913,
        tol=1e-8,
    )
    np.testing.assert_array_equal(new.values, old.values)
    np.testing.assert_array_equal(new.stderr, old.stderr)


def test_reflected_transition_probabilities_match_every_stencil_row():
    problem = make_boundary_problem3d(n=4, kinds=dict.fromkeys(FACES, "robin"))
    operator = BoundaryOperator3D(problem)
    probabilities, _, killing = operator.mc_coefficients()
    offsets = np.array(
        [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]
    )
    for position in np.ndindex(problem.f.shape):
        observed = {}
        expected = {}
        for direction, offset in enumerate(offsets):
            target = tuple(_reflected_step(*position, problem.n, direction))
            observed[target] = observed.get(target, 0) + (1 - killing[position]) / 6
            probability = probabilities[position + (direction,)]
            if probability:
                expected[tuple(np.array(position) + offset)] = probability
        assert observed.keys() == expected.keys()
        for target in expected:
            assert observed[target] == pytest.approx(expected[target])


def test_known_dirichlet_point_has_zero_work_unknown_nodes_stay_nan():
    result = solve_boundary3d(make_boundary_problem3d(n=6), [[0, 0, 0]], "mc", walks=32)
    assert result.values[0, 0, 0] == 1.0 and result.stderr[0, 0, 0] == 0
    assert result.walks == result.steps == result.sample_count[0, 0, 0] == 0
    assert np.isnan(result.values[3, 3, 0]) and np.isnan(result.values[3, 3, 3])


def test_pure_neumann_and_unfinished_walks_are_explicit_errors():
    problem = make_boundary_problem3d(n=6, kinds=dict.fromkeys(FACES, "neumann"))
    with pytest.raises(ValueError, match="условий только 2-го рода"):
        solve_boundary3d(problem, [[3, 3, 3]], "mc")
    with pytest.raises(ValueError, match="поглощающей"):
        trace_boundary_walk3d(problem)
    problem = make_boundary_problem3d(n=6)
    with pytest.raises(RuntimeError, match="усечённый"):
        solve_boundary3d(problem, [[3, 3, 3]], "mc", max_steps=1)
    with pytest.raises(RuntimeError, match="усечённый"):
        trace_boundary_walk3d(problem, max_steps=1)


def test_trace_accounts_for_nonzero_flux_and_absorption():
    problem = make_boundary_problem3d(n=6, kinds=dict.fromkeys(FACES, "robin"))
    events = trace_boundary_walk3d(problem, (0, 0, 0), seed=3)
    assert len(events[0]["faces"]) == 3
    assert events[-1]["event"] == "robin_absorption"
    assert any(e["flux"] for e in events)
    assert events[-1]["payoff"] == pytest.approx(
        sum(e["source"] + e["flux"] + e["boundary"] for e in events)
    )


def test_zero_robin_matches_neumann_and_warmup_is_repeatable():
    kinds = dict.fromkeys(FACES, "dirichlet")
    kinds["x0"] = "neumann"
    neumann = make_boundary_problem3d(n=6, kinds=kinds)
    kinds["x0"] = "robin"
    robin = make_boundary_problem3d(n=6, kinds=kinds, kappas={"x0": 0})
    first = solve_boundary3d(neumann, [[3, 3, 0]], "mc", walks=128, seed=4)
    second = solve_boundary3d(robin, [[3, 3, 0]], "mc", walks=128, seed=4)
    np.testing.assert_array_equal(first.extras["samples"], second.extras["samples"])
    warmup_boundary3d()
    warmup_boundary3d()
