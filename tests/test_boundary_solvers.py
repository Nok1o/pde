import numpy as np
import pytest

from pdelab.boundary import (
    SIDES,
    BoundaryCondition,
    BoundaryOperator,
    BoundaryProblem,
    make_boundary_problem,
)
from pdelab.boundary_solvers import (
    select_boundary_points,
    solve_boundary,
    trace_boundary_walk,
    warmup_boundary,
)
from pdelab.problem import make_problem, select_points
from pdelab.solvers import solve


@pytest.mark.parametrize("kind", ["neumann", "robin"])
@pytest.mark.parametrize("side", SIDES)
def test_lod_changes_boundary_rows_and_matches_direct(side, kind):
    kinds = {s: kind if s == side else "dirichlet" for s in SIDES}
    p = make_boundary_problem("smooth", 12, kinds)
    op = BoundaryOperator(p)
    result = solve_boundary(p, select_boundary_points(12), tol=1e-10)
    assert result.converged and result.relative_residual <= 1e-10
    np.testing.assert_allclose(result.values, op.solve(), atol=1e-7)


def test_lod_pure_neumann_is_normalized_and_stops_relatively():
    p = make_boundary_problem(n=8, kinds=dict.fromkeys(SIDES, "neumann"))
    op = BoundaryOperator(p)
    result = solve_boundary(p, select_boundary_points(8), tol=1e-10)
    assert result.converged and abs(np.sum(op.volumes * result.values)) < 1e-13
    np.testing.assert_allclose(result.values, op.solve(), atol=1e-8)
    limited = solve_boundary(p, select_boundary_points(8), max_iter=1)
    assert not limited.converged and limited.iterations == 1


@pytest.mark.parametrize(
    "kinds",
    [
        dict(zip(SIDES, ("neumann", "dirichlet", "neumann", "dirichlet"))),
        dict.fromkeys(SIDES, "robin"),
    ],
)
def test_mc_flux_and_killing_agree_with_independent_grid_solution(kinds):
    p = make_boundary_problem("smooth", 8, kinds)
    points = np.array([[4, 4], [4, 0], [0, 0]])
    target = BoundaryOperator(p).solve()[tuple(points.T)]
    values = np.array(
        [
            solve_boundary(p, points, "mc", walks=4096, seed=s).values[tuple(points.T)]
            for s in range(8)
        ]
    )
    uncertainty = values.std(axis=0, ddof=1) / np.sqrt(len(values))
    assert np.all(np.abs(values.mean(axis=0) - target) < 5 * uncertainty + 1e-10)


def test_dirichlet_rng_and_values_are_unchanged_including_full_field():
    p = make_problem("sine", 8)
    boundary = BoundaryProblem.from_problem(p)
    old = solve(p, select_points(8, "field"), "mc", walks=64, seed=913)
    new = solve_boundary(
        boundary, select_boundary_points(8, "field"), "mc", walks=64, seed=913
    )
    np.testing.assert_array_equal(old.values, new.values)
    np.testing.assert_array_equal(old.stderr, new.stderr)


def test_known_boundary_point_and_unknown_nodes():
    p = make_boundary_problem(n=8)
    r = solve_boundary(p, select_boundary_points(8, point=(0, 0)), "mc", walks=32)
    assert r.values[0, 0] == p.exact[0, 0]
    assert r.stderr[0, 0] == 0 and r.steps == 0
    assert r.walks == 0 and r.sample_count[0, 0] == 0
    assert np.isnan(r.values[4, 4]) and np.isnan(r.values[4, 0])


def test_neumann_mc_and_truncation_are_explicit_errors():
    p = make_boundary_problem(n=8, kinds=dict.fromkeys(SIDES, "neumann"))
    with pytest.raises(ValueError, match="условий только 2-го рода"):
        solve_boundary(p, select_boundary_points(8), "mc")
    with pytest.raises(ValueError, match="поглощающей"):
        trace_boundary_walk(p)
    p = make_boundary_problem(n=8)
    with pytest.raises(RuntimeError, match="усечённый"):
        solve_boundary(p, select_boundary_points(8), "mc", max_steps=1)
    with pytest.raises(RuntimeError, match="усечённый"):
        trace_boundary_walk(p, max_steps=1)


def test_trace_contains_flux_and_robin_absorption():
    p = make_boundary_problem(n=8, kinds=dict.fromkeys(SIDES, "robin"))
    history = trace_boundary_walk(p, point=(0, 4), seed=3)
    assert history[-1]["event"] == "robin_absorption"
    assert any(e["flux"] != 0 for e in history)
    assert history[-1]["payoff"] == pytest.approx(
        sum(e["source"] + e["flux"] + e["boundary"] for e in history)
    )


def test_warmup_and_unsupported_methods():
    warmup_boundary()
    warmup_boundary()
    with pytest.raises(ValueError, match="только ЛОМ"):
        solve_boundary(make_boundary_problem(n=8), [[4, 4]], "reuse")


def test_relative_stopping_does_not_depend_on_units():
    p = make_boundary_problem(n=8)
    scaled = BoundaryProblem(
        p.n,
        p.f * 1e6,
        {
            s: BoundaryCondition(c.kind, c.value * 1e6, c.kappa)
            for s, c in p.sides.items()
        },
    )
    original = solve_boundary(p, [[4, 4]], tol=1e-7)
    large = solve_boundary(scaled, [[4, 4]], tol=1e-7)
    assert large.iterations == original.iterations
    np.testing.assert_allclose(large.values / 1e6, original.values, atol=1e-12)


def test_robin_zero_is_neumann_for_mc_and_small_kappa_is_continuous():
    kinds = dict(zip(SIDES, ("neumann", "dirichlet", "dirichlet", "dirichlet")))
    neumann = make_boundary_problem(n=8, kinds=kinds)
    kinds["left"] = "robin"
    zero = make_boundary_problem(n=8, kinds=kinds, kappas={"left": 0})
    small = make_boundary_problem(n=8, kinds=kinds, kappas={"left": 1e-8})
    a = solve_boundary(neumann, [[4, 0]], "mc", walks=128, seed=4)
    b = solve_boundary(zero, [[4, 0]], "mc", walks=128, seed=4)
    np.testing.assert_array_equal(a.values, b.values)
    np.testing.assert_array_equal(a.extras["samples"], b.extras["samples"])
    np.testing.assert_allclose(
        BoundaryOperator(small).solve(), BoundaryOperator(neumann).solve(), atol=1e-8
    )
