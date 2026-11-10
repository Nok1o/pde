import numpy as np
import pytest
from scipy.sparse import diags

from pdelab.boundary import (
    SIDES,
    BoundaryCondition,
    BoundaryOperator,
    BoundaryProblem,
    make_boundary_problem,
)
from pdelab.operator import FrozenOperator
from pdelab.problem import make_problem, reference_solution


def mixed_kinds(side, kind):
    return {s: kind if s == side else "dirichlet" for s in SIDES}


@pytest.mark.parametrize("side", SIDES)
@pytest.mark.parametrize("kind", ["neumann", "robin"])
def test_polynomial_signs_and_nonzero_boundary_values(side, kind):
    p = make_boundary_problem(n=12, kinds=mixed_kinds(side, kind))
    op = BoundaryOperator(p)
    np.testing.assert_allclose(op.apply(p.exact), op.rhs(), atol=3e-12)
    np.testing.assert_allclose(op.solve(), p.exact, atol=1e-11)


@pytest.mark.parametrize(
    "kinds",
    [
        mixed_kinds("left", "neumann"),
        mixed_kinds("top", "robin"),
        dict.fromkeys(SIDES, "robin"),
        dict.fromkeys(SIDES, "neumann"),
    ],
)
def test_second_order_on_smooth_solution(kinds):
    errors = []
    for n in (12, 24, 48):
        p = make_boundary_problem("smooth", n, kinds)
        op = BoundaryOperator(p)
        errors.append(np.max(np.abs(op.solve() - op.normalize(p.exact))))
    ratios = np.array(errors[:-1]) / errors[1:]
    assert np.all((ratios > 3.8) & (ratios < 4.3))


def test_matrix_matches_stencil_and_has_positive_mc_weights():
    kinds = dict(zip(SIDES, ("neumann", "robin", "robin", "dirichlet")))
    p = make_boundary_problem(n=8, kinds=kinds, kappas={"right": np.linspace(0, 2, 9)})
    op = BoundaryOperator(p)
    matrix, rhs = op.assemble()
    u = np.random.default_rng(12).normal(size=p.f.shape)
    u[op.fixed] = op.boundary[op.fixed]
    np.testing.assert_allclose(
        matrix @ u[~op.fixed] - rhs, (op.apply(u) - op.rhs())[~op.fixed], atol=1e-12
    )
    symmetric = (diags(op.volumes[~op.fixed]) @ matrix).toarray()
    np.testing.assert_allclose(symmetric, symmetric.T, atol=1e-13)
    assert np.linalg.eigvalsh(symmetric).min() > 0
    probabilities, reward, killing = op.mc_coefficients()
    assert np.all(probabilities >= 0) and np.all(killing >= 0)
    np.testing.assert_allclose(probabilities.sum(axis=-1) + killing, 1)
    solved = op.solve()
    neighbours = np.zeros_like(op.weights)
    neighbours[:-1, :, 0] = solved[1:, :]
    neighbours[1:, :, 1] = solved[:-1, :]
    neighbours[:, :-1, 2] = solved[:, 1:]
    neighbours[:, 1:, 3] = solved[:, :-1]
    np.testing.assert_allclose(
        (np.sum(probabilities * neighbours, axis=-1) + reward)[~op.fixed],
        solved[~op.fixed],
        atol=1e-12,
    )


def test_extruded_one_dimensional_boundary_equations():
    n = 8
    x, _ = np.meshgrid(np.linspace(0, 1, n + 1), np.linspace(0, 1, n + 1))
    exact = 1 + x + x * x
    p = BoundaryProblem(
        n,
        2,
        {
            "left": BoundaryCondition("neumann", -1),
            "right": BoundaryCondition("robin", 9, 2),
            "bottom": BoundaryCondition("neumann", 0),
            "top": BoundaryCondition("neumann", 0),
        },
        exact,
    )
    np.testing.assert_allclose(BoundaryOperator(p).solve(), exact, atol=1e-12)


@pytest.mark.parametrize("preset", ["sine", "affine", "hot_top"])
def test_all_dirichlet_reproduces_existing_shared_operator(preset):
    legacy = make_problem(preset, 13)
    op = BoundaryOperator(BoundaryProblem.from_problem(legacy))
    frozen = FrozenOperator.from_problem(legacy)
    np.testing.assert_array_equal(op.weights[1:-1, 1:-1], frozen.weights)
    np.testing.assert_allclose(op.solve(), reference_solution(legacy), atol=1e-11)
    matrix, rhs = op.assemble()
    old_matrix, old_rhs = frozen.assemble()
    np.testing.assert_array_equal(matrix.toarray(), old_matrix.toarray())
    np.testing.assert_array_equal(rhs, old_rhs)


def test_neumann_balance_and_volume_weighted_gauge():
    p = make_boundary_problem(n=12, kinds=dict.fromkeys(SIDES, "neumann"))
    op = BoundaryOperator(p)
    matrix, _ = op.assemble()
    np.testing.assert_allclose(matrix @ np.ones(169), 0, atol=1e-12)
    u = op.solve()
    assert abs(np.sum(u * op.volumes)) < 1e-13
    np.testing.assert_allclose(u, op.normalize(p.exact), atol=1e-12)
    bad = BoundaryProblem(p.n, p.f + 0.1, dict(p.sides))
    with pytest.raises(ValueError, match="баланс"):
        BoundaryOperator(bad).solve()


def test_robin_zero_limit_and_corner_rules():
    neumann = make_boundary_problem(n=8, kinds=dict.fromkeys(SIDES, "neumann"))
    robin = make_boundary_problem(
        n=8, kinds=dict.fromkeys(SIDES, "robin"), kappas=dict.fromkeys(SIDES, 0)
    )
    np.testing.assert_array_equal(
        BoundaryOperator(neumann).solve(), BoundaryOperator(robin).solve()
    )
    p = make_boundary_problem(n=8)
    op = BoundaryOperator(p)
    assert op.fixed[0, 0] and not op.fixed[4, 0]
    sides = dict(p.sides)
    sides["top"] = BoundaryCondition("dirichlet", 100)
    with pytest.raises(ValueError, match="углу"):
        BoundaryOperator(BoundaryProblem(8, p.f, sides))


def test_flux_conversion_and_immutable_inputs():
    np.testing.assert_array_equal(
        BoundaryCondition.from_flux([2, -4], 2).value, [-1, 2]
    )
    p = make_boundary_problem(n=8)
    with pytest.raises(ValueError):
        p.f.setflags(write=True)
    with pytest.raises(ValueError):
        BoundaryOperator(p).weights.setflags(write=True)


@pytest.mark.parametrize("n", [True, 2, 4.5, np.nan])
def test_invalid_grid(n):
    with pytest.raises(ValueError):
        make_boundary_problem(n=n)


@pytest.mark.parametrize(
    "condition",
    [
        BoundaryCondition("robin", 1, -1),
        BoundaryCondition("neumann", 0, 1),
        BoundaryCondition("unknown", 0),
        BoundaryCondition("dirichlet", np.nan),
    ],
)
def test_invalid_conditions(condition):
    with pytest.raises(ValueError):
        condition.sampled(9)
