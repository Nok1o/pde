"""Independent full-history checks, separate h/dt errors and explicit failures."""

from dataclasses import replace

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from pdelab.heat3d_contract import LinearResult3D
from pdelab.heat3d_operator import freeze_heat3d
from pdelab.heat3d_problem import HeatFace3D, HeatProblem3D, make_heat_problem3d
from pdelab.heat3d_reference import newton_residual3d
from pdelab.heat3d_solver import HeatSettings3D, solve_heat3d

STRICT = HeatSettings3D(tolerance=1e-10, change_tolerance=1e-10, linear_tolerance=1e-12)


@pytest.mark.parametrize("case", ["sine", "polynomial"])
@pytest.mark.parametrize("alpha", [0.0, 0.5])
def test_picard_matches_independent_newton_through_short_last_step(case, alpha):
    problem = make_heat_problem3d(case, n=6, alpha=alpha)
    points = np.array([[3, 3, 3], [2, 3, 4], [0, 2, 4], [6, 6, 6]])
    initial = problem.initial_values()
    result = solve_heat3d(problem, points, dt=0.013, T=0.04, settings=STRICT)
    reference = solve_heat3d(
        problem, points, dt=0.013, T=0.04, method="newton", settings=STRICT
    )
    assert result.status == reference.status == "completed"
    assert result.physical_steps == reference.physical_steps == 4
    assert result.completed_until == result.target_time == 0.04
    assert result.history[-1]["dt"] == pytest.approx(0.001)
    np.testing.assert_allclose(result.values, reference.values, atol=2e-9, rtol=0)
    np.testing.assert_array_equal(problem.initial_values(), initial)
    exact = problem.evaluate(problem.exact, 0.04)
    np.testing.assert_array_equal(result.values[2:], exact[tuple(points[2:].T)])
    assert result.walks == result.transitions == 0
    assert result.seconds >= result.preparation_seconds > 0
    assert result.peak_rss_bytes is None
    for row in result.history:
        if row["accepted"]:
            assert row["nonlinear_relative_residual"] <= STRICT.tolerance
            assert row["relative_change"] <= STRICT.change_tolerance
    for layer in range(1, 5):
        scales = [
            row["residual_scale"] for row in result.history if row["layer"] == layer
        ]
        assert len(set(scales)) == 1


def test_linear_limit_matches_backward_euler_amplitude_without_grid_reference():
    problem = make_heat_problem3d(n=8, alpha=0.0)
    points = np.array([[4, 4, 4], [2, 3, 5]])
    amplitude = 1.0
    eigenvalue = 12 * 8**2 * np.sin(np.pi / 16) ** 2
    for time, step in [(0.013, 0.013), (0.026, 0.013), (0.03, 0.004)]:
        amplitude = (amplitude + step * (3 * np.pi**2 - 1) * np.exp(-time)) / (
            1 + step * eigenvalue
        )
    result = solve_heat3d(problem, points, dt=0.013, T=0.03, settings=STRICT)
    shape = np.prod(np.sin(np.pi * points / 8), axis=1)
    np.testing.assert_allclose(result.values, 1 + amplitude * shape, atol=1e-10)


def test_temporal_order_against_exact_semidiscrete_evolution():
    problem = make_heat_problem3d(n=8, alpha=0.0)
    final = 0.04
    eigenvalue = 12 * 8**2 * np.sin(np.pi / 16) ** 2
    amplitude = np.exp(-eigenvalue * final) + (3 * np.pi**2 - 1) * (
        np.exp(-final) - np.exp(-eigenvalue * final)
    ) / (eigenvalue - 1)
    errors = []
    for step in (0.02, 0.01, 0.005):
        result = solve_heat3d(problem, dt=step, T=final, settings=STRICT)
        assert result.status == "completed"
        errors.append(abs(result.values[0] - 1 - amplitude))
    assert 1.7 < errors[0] / errors[1] < 2.1
    assert 1.8 < errors[1] / errors[2] < 2.1


@pytest.mark.parametrize("gamma,sigma", [(0.0, 0.0), (0.4, 0.3)])
def test_spatial_order_with_small_fixed_time_step(gamma, sigma):
    errors = []
    for n in (4, 8, 16):
        problem = make_heat_problem3d(
            n=n, alpha=0.5, capacity_gamma=gamma, reaction_sigma=sigma
        )
        result = solve_heat3d(problem, dt=0.001, T=0.01, settings=STRICT)
        assert result.status == "completed"
        errors.append(
            abs(result.values[0] - problem.evaluate(problem.exact, 0.01)[(n // 2,) * 3])
        )
    assert 3.7 < errors[0] / errors[1] < 4.5
    assert 3.7 < errors[1] / errors[2] < 4.5


@pytest.mark.parametrize("gamma,sigma", [(0.0, 0.0), (0.4, 0.3)])
def test_nonlinear_temporal_order_against_independent_semidiscrete_ode(gamma, sigma):
    n = 6
    problem = make_heat_problem3d(
        n=n, alpha=0.5, capacity_gamma=gamma, reaction_sigma=sigma
    )
    interior = np.s_[1:-1, 1:-1, 1:-1]

    def rhs(t, unknowns):
        # Exact data are used only on the prescribed boundary; overwrite the interior.
        values = problem.evaluate(problem.exact, t)
        values[interior] = unknowns.reshape((n - 1,) * 3)
        conductivity = 1 + 0.5 * values**2
        derivative = problem.evaluate(problem.source, t)[interior].copy()
        for axis in range(3):
            for sign in (-1, 1):
                neighbor = [slice(1, -1)] * 3
                neighbor[axis] = slice(None, -2) if sign < 0 else slice(2, None)
                neighbor = tuple(neighbor)
                derivative += (
                    0.5
                    * n**2
                    * (conductivity[interior] + conductivity[neighbor])
                    * (values[neighbor] - values[interior])
                )
        derivative -= sigma * values[interior] ** 3
        derivative /= 1 + gamma * values[interior] ** 2
        return derivative.ravel()

    reference = solve_ivp(
        rhs,
        (0, 0.04),
        problem.initial_values()[interior].ravel(),
        method="DOP853",
        rtol=2e-12,
        atol=1e-13,
    )
    assert reference.success
    points = np.array(list(np.ndindex((n - 1,) * 3))) + 1
    errors = []
    for dt in (0.01, 0.005, 0.0025):
        result = solve_heat3d(problem, points, dt=dt, T=0.04, settings=STRICT)
        assert result.status == "completed"
        errors.append(np.max(abs(result.values - reference.y[:, -1])))
    assert 1.9 < errors[0] / errors[1] < 2.2
    assert 1.9 < errors[1] / errors[2] < 2.2


def test_previous_layer_is_computed_and_exact_is_not_read():
    problem = make_heat_problem3d(n=4)
    previous_fields, iterates = {}, {}

    def forbidden(*args):
        pytest.fail("Solver must not read the analytic reference")

    def factory(problem, previous, iterate, t, dt):
        previous_fields.setdefault(t, previous.copy())
        iterates[t] = iterate.copy()
        return freeze_heat3d(problem, previous, iterate, t, dt)

    result = solve_heat3d(replace(problem, exact=forbidden), operator_factory=factory)
    assert result.status == "completed"
    np.testing.assert_array_equal(previous_fields[0.02], iterates[0.01])
    assert (
        np.max(abs(previous_fields[0.02] - problem.evaluate(problem.exact, 0.01)))
        > 1e-4
    )


def test_zero_time_and_constant_solution_need_no_linear_iterations():
    problem = make_heat_problem3d(n=4)

    def forbidden(*args, **kwargs):
        pytest.fail("No linear solve is needed")

    result = solve_heat3d(problem, T=0, backend=forbidden)
    assert result.status == "completed"
    assert result.values[0] == problem.initial_values()[2, 2, 2]
    assert (
        result.outer_iterations == result.inner_iterations == result.physical_steps == 0
    )

    def constant(x, y, z, t):
        return 2.0

    problem = HeatProblem3D(
        "constant",
        4,
        0.0,
        constant,
        lambda *args: 0,
        {face: HeatFace3D(1, constant) for face in problem.faces},
    )
    result = solve_heat3d(problem, dt=0.02, T=0.04, backend=forbidden)
    assert result.status == "completed" and result.values[0] == 2
    assert result.physical_steps == 2
    assert result.outer_iterations == result.inner_iterations == 0


def test_dt_greater_than_final_time_and_progress():
    events = []
    result = solve_heat3d(
        make_heat_problem3d(n=4),
        dt=1,
        T=0.01,
        progress=lambda fraction, text: events.append((fraction, text)),
    )
    assert result.status == "completed" and result.physical_steps == 1
    assert result.history[-1]["dt"] == 0.01
    assert events[0][0] == 0 and events[-1][0] == 1


@pytest.mark.parametrize(
    "settings,status",
    [
        (replace(STRICT, max_outer=1), "outer_limit"),
        (replace(STRICT, max_inner=1), "linear_limit"),
    ],
)
def test_iteration_limit_does_not_publish_candidate_at_target_time(settings, status):
    result = solve_heat3d(make_heat_problem3d(n=8), settings=settings)
    assert result.status == status
    assert result.values is None and result.completed_until == 0
    assert result.physical_steps == 0 and result.outer_iterations == 1
    assert not result.history[-1]["accepted"]
    assert result.diagnostic_time == 0.01
    assert result.diagnostic_field.shape == (9, 9, 9)
    assert np.isfinite(result.diagnostic_field).all()


def test_failure_after_accepted_layer_preserves_its_time_and_work():
    def factory(problem, previous, iterate, t, dt):
        if t > 0.01:
            raise FloatingPointError("Источник второго слоя не конечен")
        return freeze_heat3d(problem, previous, iterate, t, dt)

    result = solve_heat3d(make_heat_problem3d(n=4), operator_factory=factory)
    assert result.status == "nonfinite" and result.values is None
    assert result.completed_until == 0.01 and result.physical_steps == 1
    assert result.outer_iterations > 0 and result.inner_iterations > 0
    assert result.history[-1]["layer"] == 2 and not result.history[-1]["accepted"]
    assert result.preparation_seconds > 0


def test_nonfinite_candidate_keeps_started_and_completed_work_counters():
    def broken(operator, initial, **kwargs):
        return LinearResult3D(
            np.full_like(initial, np.nan), "nonfinite", "failed", np.inf, iterations=3
        )

    result = solve_heat3d(make_heat_problem3d(n=4), backend=broken)
    assert result.status == "nonfinite" and result.values is None
    assert result.outer_iterations == 1 and result.inner_iterations == 3
    assert result.history[-1]["outer"] == 1


def test_later_boundary_conflict_is_reported():
    problem = make_heat_problem3d(n=4)
    faces = dict(problem.faces)
    faces["x0"] = HeatFace3D(1, lambda x, y, z, t: 1 + t)
    result = solve_heat3d(replace(problem, faces=faces))
    assert result.status == "unsupported" and result.values is None
    assert result.completed_until == 0 and result.history[-1]["layer"] == 1


def test_memory_and_reference_limits_are_checked_before_work(monkeypatch):
    import pdelab.heat3d_solver as module

    def forbidden(*args, **kwargs):
        pytest.fail("Work must not begin after a resource refusal")

    monkeypatch.setattr(module, "warmup_lines3d", forbidden)
    result = solve_heat3d(
        make_heat_problem3d(n=4), settings=replace(STRICT, max_workspace_bytes=1)
    )
    assert result.status == "memory_limit" and result.values is None
    monkeypatch.setattr(module, "newton_system3d", forbidden)
    result = solve_heat3d(make_heat_problem3d(n=20), method="newton")
    assert result.status == "unsupported" and result.outer_iterations == 0


def test_soft_time_limit_between_layers(monkeypatch):
    import pdelab.heat3d_solver as module

    ticks = iter(range(20))
    monkeypatch.setattr(module, "perf_counter", lambda: float(next(ticks)))
    result = solve_heat3d(
        make_heat_problem3d(n=4), settings=replace(STRICT, max_seconds=0.5)
    )
    assert result.status == "time_limit" and result.values is None
    assert result.physical_steps == 0 and result.completed_until == 0
    assert result.seconds >= result.preparation_seconds > 0


@pytest.mark.parametrize(
    "n,dt,T,budget",
    [(64, 0.01, 0.0, 3 * 1024**2), (4, 1 / 8192, 1.0, 45000)],
)
def test_initial_and_time_buffers_are_budgeted_before_allocation(
    monkeypatch, n, dt, T, budget
):
    import pdelab.heat3d_solver as module

    problem = make_heat_problem3d(n=n)

    def forbidden(*args, **kwargs):
        pytest.fail("Memory refusal must precede warmup and time-grid allocation")

    monkeypatch.setattr(module, "warmup_lines3d", forbidden)
    monkeypatch.setattr(module, "time_grid", forbidden)
    result = solve_heat3d(
        problem, dt=dt, T=T, settings=replace(STRICT, max_workspace_bytes=budget)
    )
    assert result.status == "memory_limit"
    assert result.workspace_bytes > budget and result.physical_steps == 0
    assert result.diagnostic_field is None


def test_history_memory_limit_is_checked_inside_a_long_layer():
    def unconverged(operator, initial, **kwargs):
        return LinearResult3D(initial + 1, "completed", "test", 0, iterations=1)

    result = solve_heat3d(
        make_heat_problem3d(n=4, alpha=0),
        backend=unconverged,
        settings=replace(STRICT, max_outer=10000, max_workspace_bytes=50000),
    )
    assert result.status == "memory_limit"
    assert result.values is None and result.diagnostic_field is not None
    assert 1 < result.outer_iterations < 50
    assert not result.history[-1]["accepted"]


def test_history_budget_accounts_for_final_records_without_layer_copies():
    budget = 500000
    result = solve_heat3d(
        make_heat_problem3d(n=4, alpha=0),
        T=0.01,
        settings=replace(
            STRICT, max_outer=10000, max_workspace_bytes=budget, relaxation=1e-5
        ),
    )
    assert result.status == "memory_limit"
    assert len(result.history) == result.outer_iterations
    # At most the final diagnostic row/list growth may cross the estimated budget.
    assert budget < result.workspace_bytes < budget + 5000
    assert all(row["layer"] == 1 and row["dt"] == 0.01 for row in result.history)


def test_failed_newton_backtracking_records_the_saved_candidate_residual():
    def zero(*args):
        return 0.0

    faces = {face: HeatFace3D(1, zero) for face in ("x0", "x1", "y0", "y1", "z0", "z1")}
    problem = HeatProblem3D("strong source", 4, 1.0, zero, lambda *args: 1e14, faces)
    result = solve_heat3d(problem, T=0.01, method="newton")
    assert result.status == "outer_limit" and result.values is None
    assert result.outer_iterations == result.inner_iterations == 1
    row = result.history[-1]
    residual = newton_residual3d(
        result.diagnostic_field,
        problem.initial_values(),
        problem.alpha,
        problem.evaluate(problem.source, 0.01),
        0.01,
    )
    assert row["nonlinear_relative_residual"] == pytest.approx(
        np.max(abs(residual)) / row["residual_scale"]
    )
    assert row["nonlinear_relative_residual"] > 100
    assert row["relaxation"] == 2.0**-23
    assert row["relative_change"] == 1.0 and not row["accepted"]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_outer": True},
        {"max_inner": 0},
        {"max_seconds": np.inf},
        {"tolerance": 0},
        {"relaxation": 2},
    ],
)
def test_invalid_settings_are_rejected(kwargs):
    with pytest.raises(ValueError):
        solve_heat3d(make_heat_problem3d(n=4), settings=HeatSettings3D(**kwargs))
