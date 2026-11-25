"""Independent Robin Jacobians and complete physical-time solver comparisons."""

from dataclasses import replace

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from pdelab.heat3d_operator import freeze_heat3d
from pdelab.heat3d_problem import make_heat_problem3d
from pdelab.heat3d_reference import (
    newton_residual3d,
    newton_scale3d,
    newton_system3d,
)
from pdelab.heat3d_solver import HeatSettings3D, solve_heat3d

STRICT = HeatSettings3D(tolerance=1e-10, change_tolerance=1e-10, linear_tolerance=1e-12)


def face_kinds(configuration):
    kinds = dict.fromkeys(("x0", "x1", "y0", "y1", "z0", "z1"), 3)
    if configuration == "mixed":
        kinds.update(x0=1, z1=1)
    return kinds


def free_nodes(problem):
    free = np.ones(problem.shape, dtype=bool)
    for axis, letter in enumerate("zyx"):
        for side in (0, 1):
            if problem.faces[f"{letter}{side}"].kind == 1:
                index = [slice(None)] * 3
                index[axis] = 0 if side == 0 else -1
                free[tuple(index)] = False
    return free


@pytest.mark.parametrize("configuration", ["mixed", "all3"])
@pytest.mark.parametrize("alpha", [0.0, 0.5])
def test_every_robin_jacobian_column_matches_finite_differences(configuration, alpha):
    problem = make_heat_problem3d(
        "polynomial",
        n=4,
        alpha=alpha,
        kinds=face_kinds(configuration),
        beta={face: 0.8 + 0.3 * i for i, face in enumerate(face_kinds(configuration))},
    )
    rng = np.random.default_rng(427)
    values = problem.initial_values() + rng.normal(scale=0.2, size=problem.shape)
    previous = problem.initial_values() + rng.normal(scale=0.1, size=problem.shape)
    source = problem.evaluate(problem.source, 0.03)
    originals = [field.copy() for field in (values, previous, source)]
    args = (previous, alpha, source, 0.013)
    kwargs = dict(problem=problem, t=0.03)
    residual, jacobian = newton_system3d(values, *args, **kwargs)
    free = free_nodes(problem)
    assert residual.shape == (int(free.sum()),)
    assert jacobian.shape == (int(free.sum()),) * 2
    step = 2e-6
    # Every column includes free face, edge and corner temperatures in C order.
    for column, node in enumerate(zip(*np.where(free))):
        high, low = values.copy(), values.copy()
        high[node] += step
        low[node] -= step
        numerical = (
            newton_residual3d(high, *args, **kwargs)
            - newton_residual3d(low, *args, **kwargs)
        ) / (2 * step)
        np.testing.assert_allclose(
            jacobian[:, column].toarray().ravel(), numerical, rtol=2e-8, atol=2e-6
        )
    operator = freeze_heat3d(problem, previous, values, 0.03, 0.013)
    expected = (operator.apply(values) - operator.rhs)[free]
    np.testing.assert_allclose(residual, expected, rtol=1e-12, atol=1e-11)
    expected_scale = max(
        np.max(abs(operator.rhs[free])), np.max(abs(operator.apply(values)[free]))
    )
    assert newton_scale3d(values, *args, **kwargs) == pytest.approx(expected_scale)
    for actual, original in zip((values, previous, source), originals):
        np.testing.assert_array_equal(actual, original)


def test_reference_does_not_call_frozen_operator_or_analytic_solution(monkeypatch):
    import pdelab.heat3d_operator as production

    problem = make_heat_problem3d(n=4, kinds=face_kinds("all3"))

    def forbidden(*args, **kwargs):
        pytest.fail("Independent assembly must use its own rows and physical data")

    problem = replace(problem, exact=forbidden)
    monkeypatch.setattr(production, "freeze_heat3d", forbidden)
    monkeypatch.setattr(production, "dirichlet_data3d", forbidden)
    values = problem.initial_values()
    residual, matrix = newton_system3d(
        values,
        values,
        problem.alpha,
        problem.evaluate(problem.source, 0.01),
        0.01,
        problem=problem,
        t=0.01,
    )
    assert np.isfinite(residual).all() and np.isfinite(matrix.data).all()
    assert matrix.shape == (125, 125)


def test_explicit_dirichlet_problem_preserves_legacy_rows_and_scale():
    problem = make_heat_problem3d("polynomial", n=4)
    values = problem.evaluate(problem.exact, 0.03)
    values[1:-1, 1:-1, 1:-1] += 0.2
    args = (
        problem.initial_values(),
        problem.alpha,
        problem.evaluate(problem.source, 0.03),
        0.03,
    )
    old_residual, old_matrix = newton_system3d(values, *args)
    residual, matrix = newton_system3d(values, *args, problem=problem, t=0.03)
    assert old_residual.shape == (3, 3, 3)
    np.testing.assert_allclose(residual, old_residual.ravel(), atol=1e-12)
    np.testing.assert_allclose(matrix.toarray(), old_matrix.toarray(), atol=1e-12)
    assert newton_scale3d(values, *args, problem=problem, t=0.03) == pytest.approx(
        newton_scale3d(values, *args)
    )


@pytest.mark.parametrize("configuration", ["mixed", "all3"])
@pytest.mark.parametrize("case", ["sine", "polynomial"])
@pytest.mark.parametrize("alpha", [0.0, 0.5])
def test_complete_robin_chain_matches_three_solvers(configuration, case, alpha):
    problem = make_heat_problem3d(
        case, n=4, alpha=alpha, kinds=face_kinds(configuration)
    )
    points = np.array(list(np.ndindex(problem.shape)))
    exact = problem.evaluate(problem.exact, 0.03)

    def forbidden(*args):
        pytest.fail("The physical solver must not read the analytic reference")

    problem = replace(problem, exact=forbidden)
    runs = {
        method: solve_heat3d(
            problem, points, method=method, dt=0.013, T=0.03, settings=STRICT
        )
        for method in ("lod", "block", "newton")
    }
    for result in runs.values():
        assert result.status == "completed", result.reason
        assert result.physical_steps == 3 and result.completed_until == 0.03
        assert result.history[-1]["dt"] == pytest.approx(0.004)
        assert result.walks == result.transitions == 0
        assert all(
            row["nonlinear_relative_residual"] <= STRICT.tolerance
            for row in result.history
            if row["accepted"]
        )
        for layer in (1, 2, 3):
            assert (
                len(
                    {
                        row["residual_scale"]
                        for row in result.history
                        if row["layer"] == layer
                    }
                )
                == 1
            )
        # Includes every interior, face, edge and corner query at the target time.
        np.testing.assert_allclose(
            result.values, runs["newton"].values, atol=3e-9, rtol=0
        )
        fixed = ~free_nodes(problem)
        np.testing.assert_array_equal(
            result.values.reshape(problem.shape)[fixed], exact[fixed]
        )
    for result in (runs["lod"], runs["block"]):
        assert all(
            row["relative_change"] <= STRICT.change_tolerance
            for row in result.history
            if row["accepted"]
        )


@pytest.mark.parametrize("configuration", ["mixed", "all3"])
def test_robin_previous_layer_is_the_computed_field(configuration):
    problem = make_heat_problem3d(n=4, kinds=face_kinds(configuration))
    previous_fields, iterates = {}, {}

    def factory(problem, previous, iterate, t, dt):
        previous_fields.setdefault(t, previous.copy())
        iterates[t] = iterate.copy()
        return freeze_heat3d(problem, previous, iterate, t, dt)

    result = solve_heat3d(problem, operator_factory=factory, settings=STRICT)
    assert result.status == "completed"
    np.testing.assert_array_equal(previous_fields[0.02], iterates[0.01])
    assert (
        np.max(abs(previous_fields[0.02] - problem.evaluate(problem.exact, 0.01)))
        > 1e-4
    )


def test_newton_budget_includes_free_boundary_unknowns_before_work(monkeypatch):
    import pdelab.heat3d_solver as solver

    settings = replace(STRICT, max_workspace_bytes=100000)
    dirichlet = solve_heat3d(
        make_heat_problem3d(n=4), method="newton", T=0.01, settings=settings
    )
    assert dirichlet.status == "completed"

    def forbidden(*args, **kwargs):
        pytest.fail("Memory refusal must precede reference or time-grid allocation")

    monkeypatch.setattr(solver, "newton_system3d", forbidden)
    monkeypatch.setattr(solver, "time_grid", forbidden)
    robin = solve_heat3d(
        make_heat_problem3d(n=4, kinds=face_kinds("all3")),
        method="newton",
        T=0.01,
        settings=settings,
    )
    assert robin.status == "memory_limit" and robin.physical_steps == 0
    assert robin.workspace_bytes >= 16 * 125**2
    assert robin.values is None and robin.diagnostic_field is None


@pytest.mark.parametrize("configuration", ["mixed", "all3"])
def test_robin_spatial_error_decreases_at_fixed_small_time_step(configuration):
    errors = []
    for n in (4, 6, 8):
        problem = make_heat_problem3d(n=n, alpha=0.5, kinds=face_kinds(configuration))
        points = np.array(list(np.ndindex(problem.shape)))
        result = solve_heat3d(problem, points, dt=0.0005, T=0.005, settings=STRICT)
        assert result.status == "completed"
        errors.append(
            np.max(abs(result.values - problem.evaluate(problem.exact, 0.005).ravel()))
        )
    assert 2.1 < errors[0] / errors[1] < 2.5
    assert 1.6 < errors[1] / errors[2] < 2.0


@pytest.mark.parametrize("configuration", ["mixed", "all3"])
def test_robin_temporal_error_decreases_against_independent_ode(configuration):
    problem = make_heat_problem3d(n=4, alpha=0.5, kinds=face_kinds(configuration))
    free = free_nodes(problem)
    initial = problem.initial_values()

    def rhs(t, unknowns):
        field = initial.copy()
        field[free] = unknowns
        # Equal current/previous free values remove the time term, leaving the
        # independent spatial residual. Boundary data are sampled at ODE time t.
        return -newton_residual3d(
            field,
            field,
            problem.alpha,
            problem.evaluate(problem.source, t),
            1.0,
            problem=problem,
            t=t,
        )

    reference = solve_ivp(
        rhs, (0, 0.04), initial[free], method="DOP853", rtol=2e-12, atol=1e-13
    )
    assert reference.success
    errors = []
    for dt in (0.01, 0.005, 0.0025):
        result = solve_heat3d(
            problem, np.argwhere(free), method="newton", dt=dt, T=0.04, settings=STRICT
        )
        assert result.status == "completed"
        errors.append(np.max(abs(result.values - reference.y[:, -1])))
    assert 1.7 < errors[0] / errors[1] < 2.3
    assert 1.7 < errors[1] / errors[2] < 2.2
