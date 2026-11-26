"""Merged occupancies, expected mass, independent moments and honest limits."""

from time import perf_counter

import numpy as np
import pytest
from scipy.sparse.linalg import spsolve

from pdelab.boundary3d import FACES
from pdelab.heat3d_operator import HeatOperator3D, freeze_heat3d
from pdelab.heat3d_problem import make_heat_problem3d
from pdelab.heat3d_transport import (
    solve_frozen_counts3d,
    solve_frozen_mass3d,
    solve_mass_lines3d,
    transport_workspace_bytes,
    warmup_transport3d,
)


def heat_operator(kind3=(), signed=False):
    problem = make_heat_problem3d(
        "polynomial",
        4,
        0.5,
        kinds={face: 3 if face in kind3 else 1 for face in FACES},
        capacity_gamma=0.5,
        reaction_sigma=1.0,
    )
    previous = problem.initial_values()
    iterate = previous + np.indices(problem.shape)[0] * 0.03
    operator = freeze_heat3d(problem, previous, iterate, 0.01, 0.01)
    if signed:
        target = np.random.default_rng(12).normal(size=problem.shape)
        operator = operator.with_rhs(operator.apply(target))
    return operator


def diamond():
    """Unequal probabilities, signed rewards and repeated merging in a cycle."""
    points = np.array([[2, 2, 2], [2, 2, 1], [2, 1, 2], [2, 1, 1]])
    transition = np.array(
        [[0, 0.3, 0.4, 0], [0.1, 0, 0, 0.6], [0.2, 0, 0, 0.5], [0, 0.25, 0.15, 0]]
    )
    rewards = np.array([0.3, -2.0, 1.7, 0.8])
    shape = (5, 5, 5)
    diagonal = np.ones(shape)
    rhs = np.zeros(shape)
    weights = np.zeros((*shape, 6))
    offsets = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    for i, point in enumerate(points):
        index = tuple(point)
        diagonal[index] = i + 1
        rhs[index] = rewards[i] * diagonal[index]
        for j, neighbor in enumerate(points):
            if transition[i, j]:
                direction = offsets.index(tuple(neighbor - point))
                weights[index + (direction,)] = transition[i, j] * diagonal[index]
    return (
        HeatOperator3D(
            diagonal,
            weights,
            rhs,
            np.zeros(shape, dtype=bool),
            np.zeros(shape),
            np.ones(shape),
        ),
        points,
        transition,
        rewards,
    )


@pytest.fixture(scope="module", autouse=True)
def compiled():
    warmup_transport3d()


def test_counts_match_mean_and_variance_with_merging_and_signed_rewards():
    operator, points, transition, rewards = diamond()
    target = np.linalg.solve(np.eye(4) - transition, rewards)
    second_moment = np.linalg.solve(
        np.eye(4) - transition, rewards**2 + 2 * rewards * (transition @ target)
    )
    walks = 256
    expected_variance = (second_moment - target**2) / walks
    results = [
        solve_frozen_counts3d(operator, points, walks=walks, seed=seed)
        for seed in range(256)
    ]
    assert all(result.status == "completed" for result in results)
    values = np.array([result.values for result in results])
    np.testing.assert_array_less(
        np.abs(values.mean(axis=0) - target),
        5 * np.sqrt(expected_variance / len(results)),
    )
    variance_ratio = values.var(axis=0, ddof=1) / expected_variance
    assert np.all((variance_ratio > 0.65) & (variance_ratio < 1.4))
    assert np.max(np.abs(np.corrcoef(values.T) - np.eye(4))) < 0.2
    for result in results:
        assert result.walks == result.completed_paths.sum() == walks * len(points)
        assert result.stderr is None
        assert result.node_updates < result.transitions
        assert result.edge_transfers <= 6 * result.node_updates


@pytest.mark.parametrize("kind3", [(), ("x0",), ("x0", "y0", "z0"), FACES])
@pytest.mark.parametrize("signed", [False, True])
def test_mass_and_counts_match_independent_small_heat_system(kind3, signed):
    operator = heat_operator(kind3, signed)
    points = np.array([[0, 0, 0], [0, 0, 2], [0, 2, 2], [2, 2, 2], [4, 4, 4]])
    matrix, rhs = operator.assemble()
    target = spsolve(matrix, rhs).reshape(operator.shape)[tuple(points.T)]
    mass = solve_frozen_mass3d(operator, points, tolerance=1e-8)
    assert mass.status == "completed"
    error = np.abs(mass.values - target)
    assert np.all(error <= mass.tail_error_bound + 1e-12)
    assert np.all(error <= 1.01e-8 * np.abs(target) + 1e-12)
    assert mass.seed is mass.walks_per_estimate is mass.stderr is None
    assert mass.walks == mass.transitions == 0
    results = [
        solve_frozen_counts3d(operator, points, walks=2048, seed=seed)
        for seed in range(16)
    ]
    assert all(result.status == "completed" for result in results)
    values = np.array([result.values for result in results])
    uncertainty = values.std(axis=0, ddof=1) / np.sqrt(len(results))
    assert np.all(np.abs(values.mean(axis=0) - target) < 6 * uncertainty + 1e-12)


def test_finite_mass_series_equals_all_start_jacobi_series():
    operator, points, transition, rewards = diamond()
    expected = np.zeros(4)
    for terms in (1, 2, 3, 7):
        expected = np.zeros(4)
        for _ in range(terms):
            expected = rewards + transition @ expected
        for index, point in enumerate(points):
            result = solve_frozen_mass3d(
                operator, [point], tolerance=1e-15, max_iterations=terms
            )
            assert result.status == "linear_limit"
            assert result.values is None
            np.testing.assert_allclose(
                result.diagnostic_values, [expected[index]], atol=1e-14
            )
        all_starts = solve_mass_lines3d(
            operator, np.zeros(operator.shape), tolerance=1e-15, max_iterations=terms
        )
        assert all_starts.status == "linear_limit"
        np.testing.assert_allclose(
            all_starts.values[tuple(points.T)], expected, atol=1e-14
        )
        assert all_starts.transport_sweeps == all_starts.iterations == terms


def test_mass_backend_true_residual_and_initial_field_tail():
    operator = heat_operator(FACES, signed=True)
    matrix, rhs = operator.assemble()
    target = spsolve(matrix, rhs).reshape(operator.shape)
    settings = dict(tolerance=1e-10, max_iterations=10000)
    result = solve_mass_lines3d(operator, np.ones(operator.shape), **settings)
    assert result.status == "completed"
    relative = np.max(np.abs(operator.apply(result.values) - operator.rhs)) / np.max(
        np.abs(operator.rhs)
    )
    assert relative <= settings["tolerance"]
    assert result.relative_residual == pytest.approx(relative, abs=2e-15)
    assert np.max(np.abs(result.values - target)) <= result.tail_error_bound + 1e-12
    exact_initial = solve_mass_lines3d(operator, target, **settings)
    assert exact_initial.status == "completed" and exact_initial.iterations == 0
    assert exact_initial.node_updates == exact_initial.edge_transfers == 0


def test_zero_rhs_and_cancellation_have_no_absolute_tolerance_floor():
    operator, points, _, _ = diamond()
    zero = operator.with_rhs(np.zeros(operator.shape))
    result = solve_frozen_mass3d(zero, points, tolerance=1e-12)
    assert result.status == "completed"
    np.testing.assert_array_equal(result.values, 0)
    np.testing.assert_array_equal(result.tail_error_bound, 0)
    linear = solve_mass_lines3d(
        zero, np.zeros(operator.shape), tolerance=1e-12, max_iterations=5
    )
    assert linear.status == "completed" and linear.iterations == 0
    nonzero_initial = solve_mass_lines3d(
        zero, np.ones(operator.shape), tolerance=1e-12, max_iterations=5
    )
    assert nonzero_initial.status == "completed"
    assert nonzero_initial.relative_residual == 0
    np.testing.assert_array_equal(nonzero_initial.values, 0)

    # Give the first queried node an exact zero solution by manufacturing rhs;
    # a short nonzero tail must not be accepted merely because the answer is tiny.
    field = np.zeros(operator.shape)
    field[tuple(points[1])] = 1.0
    field[tuple(points[2])] = -1.0
    cancelled = operator.with_rhs(operator.apply(field))
    result = solve_frozen_mass3d(
        cancelled, [points[0]], tolerance=1e-7, max_iterations=2
    )
    assert result.status == "linear_limit" and result.values is None
    assert result.tail_error_bound[0] > 0


def test_reproducible_counts_known_nodes_and_no_reference_calls(monkeypatch):
    operator = heat_operator()

    def forbidden(*args, **kwargs):
        raise AssertionError("no reference solve in transport")

    monkeypatch.setattr(HeatOperator3D, "assemble", forbidden)
    points = [[0, 0, 0], [2, 2, 2]]
    a = solve_frozen_counts3d(operator, points, walks=1024, seed=71)
    b = solve_frozen_counts3d(operator, points, walks=1024, seed=71)
    np.testing.assert_array_equal(a.values, b.values)
    assert (a.transitions, a.node_updates, a.edge_transfers) == (
        b.transitions,
        b.node_updates,
        b.edge_transfers,
    )
    assert a.completed_paths.tolist() == [0, 1024]
    assert a.values[0] == operator.fixed_values[0, 0, 0]
    mass = solve_frozen_mass3d(operator, [[0, 0, 0]])
    assert mass.values[0] == a.values[0] and mass.node_updates == 0
    solve_mass_lines3d(
        operator, np.zeros(operator.shape), tolerance=1e-7, max_iterations=1000
    )


def test_limits_preserve_work_and_reject_unfinished_flows():
    operator, points, _, _ = diamond()
    result = solve_frozen_counts3d(operator, [points[0]], walks=1024, max_steps=1)
    assert result.status == "path_limit" and result.values is None
    assert result.walks == 1024 and result.completed_paths[0] < 1024
    result = solve_frozen_counts3d(operator, points, walks=1024, max_walks=3)
    assert result.status == "walk_budget" and result.reason == "max_walks"
    assert result.walks == result.completed_paths.sum() == 3
    assert result.values is None
    result = solve_frozen_counts3d(operator, [points[0]], walks=1024, max_transitions=1)
    assert result.status == "walk_budget" and result.reason == "max_transitions"
    assert result.transitions == 0 and result.node_updates == 1
    assert result.values is None
    result = solve_mass_lines3d(
        operator,
        np.zeros(operator.shape),
        tolerance=1e-7,
        max_iterations=100,
        deadline=perf_counter() - 1,
    )
    assert result.status == "time_limit" and result.iterations == 0
    assert result.node_updates == result.edge_transfers == 0


@pytest.mark.parametrize("solver", [solve_frozen_counts3d, solve_frozen_mass3d])
def test_memory_limit_checked_before_coefficient_buffers(solver, monkeypatch):
    operator = heat_operator()

    def forbidden(*args, **kwargs):
        raise AssertionError("coefficient buffers allocated above memory limit")

    monkeypatch.setattr("pdelab.heat3d_transport.frozen_mc_coefficients", forbidden)
    result = solver(operator, max_workspace_bytes=1)
    assert result.status == "memory_limit" and result.values is None
    assert result.node_updates == result.transport_sweeps == result.edge_transfers == 0
    assert result.workspace_bytes == transport_workspace_bytes(operator, 1)


def test_large_counts_do_not_overflow_a_finite_normalized_reward():
    shape = (5, 5, 5)
    operator = HeatOperator3D(
        np.ones(shape),
        np.zeros((*shape, 6)),
        np.full(shape, 1e308),
        np.zeros(shape, dtype=bool),
        np.zeros(shape),
        np.ones(shape),
    )
    result = solve_frozen_counts3d(operator, walks=2**31 - 1, max_walks=2**31 - 1)
    assert result.status == "completed" and result.values[0] == 1e308
    assert result.walks == result.completed_paths[0] == 2**31 - 1
    assert result.node_updates == result.transport_sweeps == 1
    assert result.transitions == result.edge_transfers == 0
    mass = solve_frozen_mass3d(operator)
    assert mass.status == "completed" and mass.values[0] == 1e308


@pytest.mark.parametrize(
    "probabilities", [(1e-200, 1e-200), (1e-155, 1e-155), (1e-200, 1e-123)]
)
def test_mass_does_not_silently_drop_underflowed_or_subnormal_flows(probabilities):
    shape = (5, 5, 5)
    fixed = np.ones(shape, dtype=bool)
    for x in (1, 2, 3):
        fixed[2, 2, x] = False
    weights = np.zeros((*shape, 6))
    first, second = probabilities
    weights[2, 2, 1, 4] = first
    weights[2, 2, 2, 4] = second
    rhs = np.zeros(shape)
    rhs[2, 2, 3] = 1e308
    operator = HeatOperator3D(
        np.ones(shape), weights, rhs, fixed, np.zeros(shape), np.ones(shape)
    )
    result = solve_frozen_mass3d(operator, [[2, 2, 1]], tolerance=1e-7)
    assert result.status == "nonfinite"
    assert result.reason == "mass_arithmetic_underflow"
    assert result.values is None
    assert result.tail_error_bound[0] > 0
    # A vanishing forward flow does not imply a vanishing requested solution.
    expected = first * (second * 1e308)
    assert expected > np.finfo(float).tiny
    matrix, load = operator.assemble()
    target = spsolve(matrix, load).reshape(shape)[2, 2, 1]
    assert target == pytest.approx(expected, rel=1e-14, abs=0)


@pytest.mark.parametrize(
    "settings",
    [
        dict(walks=1),
        dict(walks=2**31),
        dict(walks=True),
        dict(seed=-1),
        dict(max_transitions=-1),
        dict(max_steps=0),
    ],
)
def test_count_invalid_parameters(settings):
    with pytest.raises(ValueError):
        solve_frozen_counts3d(heat_operator(), **settings)


@pytest.mark.parametrize(
    "settings",
    [
        dict(tolerance=0),
        dict(tolerance=True),
        dict(tolerance=1),
        dict(max_iterations=0),
    ],
)
def test_mass_invalid_parameters(settings):
    with pytest.raises(ValueError):
        solve_frozen_mass3d(heat_operator(), **settings)
