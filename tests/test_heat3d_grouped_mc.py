"""Weighted frozen grouped paths: baseline parity, CSR and honest budgets."""

from dataclasses import replace

import numpy as np
import pytest
from scipy.sparse.linalg import spsolve

from pdelab import heat3d_mc
from pdelab.heat3d_grouped_mc import _grouped_frozen_paths
from pdelab.heat3d_mc import (
    _frozen_paths,
    frozen_mc_coefficients,
    solve_frozen_mc3d,
    warmup_mc3d,
)
from pdelab.heat3d_operator import HeatOperator3D, freeze_heat3d
from pdelab.heat3d_problem import make_heat_problem3d


def _operator(alpha=0.5, dt=0.013):
    problem = make_heat_problem3d("polynomial", 4, alpha)
    previous = problem.initial_values()
    return freeze_heat3d(problem, previous, previous, dt, dt)


def _two_node_operator(*, weight=0.5, first=2.0, second=3.0, reverse=False):
    shape = (5, 5, 5)
    weights = np.zeros((*shape, 6))
    fixed = np.ones(shape, dtype=bool)
    fixed[2, 2, 2:4] = False
    weights[2, 2, 2, 4] = weight
    if reverse:
        weights[2, 2, 3, 5] = weight
    rhs = np.zeros(shape)
    rhs[2, 2, 2], rhs[2, 2, 3] = first, second
    return HeatOperator3D(
        np.ones(shape), weights, rhs, fixed, np.zeros(shape), np.ones(shape)
    )


@pytest.mark.parametrize(
    "max_steps,max_walks,max_transitions,expected_code",
    [
        (10000, 10000, 100000, 0),
        (1, 10000, 100000, 1),
        (10000, 3, 100000, 2),
        (10000, 10000, 1, 3),
    ],
)
def test_batch_one_is_bitwise_baseline_including_failure_counters(
    max_steps, max_walks, max_transitions, expected_code
):
    op = _operator(dt=10)
    args = (
        *frozen_mc_coefficients(op),
        op.fixed,
        np.array([[0, 0, 0], [2, 2, 2], [1, 2, 3]]),
        37,
        3,
        max_steps,
        max_walks,
        max_transitions,
    )
    baseline = _frozen_paths(*args)
    grouped = _grouped_frozen_paths(*args, 1)
    assert baseline[-1] == grouped[-1] == expected_code
    for actual, expected in zip(grouped, baseline):
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("alpha", [0, 0.5, 2])
def test_weighted_groups_match_independent_frozen_csr(alpha):
    op = _operator(alpha=alpha)
    points = np.array([[1, 2, 3], [2, 2, 2], [0, 0, 0]])
    matrix, rhs = op.assemble()
    target = spsolve(matrix, rhs).reshape(op.shape)[tuple(points.T)]
    results = [
        solve_frozen_mc3d(op, points, walks=2048, seed=seed, batch_size=17)
        for seed in range(16)
    ]
    assert all(result.status == "completed" for result in results)
    values = np.array([result.values for result in results])
    uncertainty = values.std(axis=0, ddof=1) / np.sqrt(len(results))
    assert np.all(np.abs(values.mean(axis=0) - target) < 6 * uncertainty + 1e-12)
    observed = values[:, 1].std(ddof=1)
    predicted = np.sqrt(np.mean([result.stderr[1] ** 2 for result in results]))
    assert 0.3 < observed / predicted < 2.1


@pytest.mark.parametrize("batch_size", [1, 7, 37, 128, 65536])
def test_remainder_groups_fixed_queries_and_reproducible_seed(batch_size):
    op = _operator()
    points = [[0, 0, 0], [2, 2, 2], [1, 2, 3], [4, 4, 4]]
    first = solve_frozen_mc3d(op, points, walks=37, seed=51, batch_size=batch_size)
    second = solve_frozen_mc3d(op, points, walks=37, seed=51, batch_size=batch_size)
    assert first.status == second.status == "completed"
    assert first.method == "mc_grouped" and first.batch_size == batch_size
    np.testing.assert_array_equal(first.completed_paths, [0, 37, 37, 0])
    np.testing.assert_array_equal(first.values, second.values)
    np.testing.assert_array_equal(first.stderr, second.stderr)
    assert first.walks == second.walks == 74
    assert first.transitions == second.transitions
    known = solve_frozen_mc3d(
        op,
        [[0, 0, 0], [4, 4, 4]],
        walks=37,
        batch_size=batch_size,
        max_walks=1,
        max_transitions=1,
    )
    assert known.status == "completed" and known.walks == known.transitions == 0
    np.testing.assert_array_equal(known.values, op.rhs[[0, 4], [0, 4], [0, 4]])
    np.testing.assert_array_equal(known.stderr, 0)


def test_fixed_query_does_not_consume_random_draws_and_default_stays_baseline():
    op = _operator()
    alone = solve_frozen_mc3d(op, [[2, 2, 2]], walks=37, seed=19, batch_size=7)
    prefixed = solve_frozen_mc3d(
        op, [[0, 0, 0], [2, 2, 2]], walks=37, seed=19, batch_size=7
    )
    np.testing.assert_array_equal(prefixed.values[1:], alone.values)
    np.testing.assert_array_equal(prefixed.stderr[1:], alone.stderr)
    assert prefixed.transitions == alone.transitions
    old = solve_frozen_mc3d(op, walks=37, seed=19)
    batch_one = solve_frozen_mc3d(op, walks=37, seed=19, batch_size=1)
    assert old.method == "mc" and old.batch_size is None
    np.testing.assert_array_equal(old.values, batch_one.values)
    np.testing.assert_array_equal(old.stderr, batch_one.stderr)
    assert old.transitions == batch_one.transitions


@pytest.mark.parametrize("batch_size", [3, 7, 128])
def test_terminal_reward_and_killing_precede_exhausted_move_limits(batch_size):
    # All draws for this seed survive the tiny initial killing and then enter
    # a zero-coupling free row. Its payoff is required even at both move limits.
    op = _two_node_operator(weight=1 - 1e-12)
    result = solve_frozen_mc3d(
        op,
        walks=7,
        seed=19,
        batch_size=batch_size,
        max_steps=1,
        max_transitions=7,
        max_walks=7,
    )
    assert result.status == "completed"
    assert result.walks == result.transitions == result.completed_paths[0] == 7
    np.testing.assert_array_equal(result.values, [5])
    np.testing.assert_array_equal(result.stderr, [0])


def test_global_path_budget_initializes_only_remaining_partial_group():
    op = _operator()
    points = [[0, 0, 0], [2, 2, 2], [1, 2, 3], [1, 1, 1], [4, 4, 4]]
    result = solve_frozen_mc3d(op, points, walks=7, max_walks=9, batch_size=4)
    assert result.status == "walk_budget" and result.reason == "max_walks"
    assert result.values is None and result.stderr is None
    assert result.walks == 9
    np.testing.assert_array_equal(result.completed_paths, [0, 7, 2, 0, 0])
    assert np.isnan(result.diagnostic_values[3])
    assert result.diagnostic_values[4] == op.rhs[4, 4, 4]


@pytest.mark.parametrize(
    "limit,reason",
    [(dict(max_steps=1), "max_steps"), (dict(max_transitions=1), "max_transitions")],
)
def test_failed_group_counts_initialized_and_completed_paths_honestly(limit, reason):
    result = solve_frozen_mc3d(
        _operator(dt=10), walks=29, seed=3, batch_size=13, **limit
    )
    assert result.reason == reason and result.values is None and result.stderr is None
    assert result.walks == 13
    assert result.completed_paths.sum() < result.walks
    if reason == "max_transitions":
        assert result.transitions == 1
    else:
        assert 1 <= result.transitions <= result.walks


def test_group_workspace_is_exact_extra_and_guard_precedes_preparation(monkeypatch):
    op = _operator()
    baseline = solve_frozen_mc3d(op, walks=37)
    grouped = solve_frozen_mc3d(op, walks=37, batch_size=65536)
    assert grouped.workspace_bytes == baseline.workspace_bytes + 40 * 37
    limit = baseline.workspace_bytes + 40 * 7 - 1

    def forbidden(*args, **kwargs):
        pytest.fail("Memory rejection must precede coefficient arrays and JIT warmup")

    monkeypatch.setattr(heat3d_mc, "frozen_mc_coefficients", forbidden)
    monkeypatch.setattr(heat3d_mc, "warmup_mc3d", forbidden)
    result = solve_frozen_mc3d(op, walks=37, batch_size=7, max_workspace_bytes=limit)
    assert result.status == "memory_limit" and result.workspace_bytes == limit + 1
    assert result.values is None and result.diagnostic_values is None
    assert result.completed_paths is None and result.walks == result.transitions == 0
    assert result.method == "mc_grouped" and result.batch_size == 7


@pytest.mark.parametrize("failure", ["payoff", "statistics"])
@pytest.mark.parametrize("batch_size", [1, 7])
def test_nonfinite_payoff_and_statistics_never_become_solution(failure, batch_size):
    op = (
        _two_node_operator(weight=0.9, first=1e308, second=1e308, reverse=True)
        if failure == "payoff"
        else _two_node_operator(weight=0.5, first=0, second=1e200)
    )
    result = solve_frozen_mc3d(op, walks=37, seed=0, batch_size=batch_size)
    assert result.status == "nonfinite" and result.reason == "payoff_or_statistics"
    assert result.values is None and result.stderr is None
    assert 0 <= result.completed_paths.sum() <= result.walks
    if batch_size == 1:
        baseline = solve_frozen_mc3d(op, walks=37, seed=0)
        assert baseline.status == result.status
        assert (
            baseline.walks == result.walks
            and baseline.transitions == result.transitions
        )
        np.testing.assert_array_equal(baseline.completed_paths, result.completed_paths)
        np.testing.assert_array_equal(
            baseline.diagnostic_values, result.diagnostic_values
        )


def test_with_rhs_has_no_extra_boundary_payment_and_inputs_stay_immutable(monkeypatch):
    op = _operator()
    field = np.random.default_rng(9).normal(size=op.shape)
    op = op.with_rhs(op.apply(field))
    saved = {
        name: getattr(op, name).copy()
        for name in ("diagonal", "weights", "rhs", "fixed", "fixed_values", "volumes")
    }

    def forbidden(*args, **kwargs):
        pytest.fail("The grouped backend must not use a direct/reference solve")

    monkeypatch.setattr(HeatOperator3D, "assemble", forbidden)
    result = solve_frozen_mc3d(
        op, [[0, 0, 0], [2, 2, 2]], walks=30000, seed=15, batch_size=127
    )
    assert result.status == "completed"
    assert result.values[0] == field[0, 0, 0] and result.completed_paths[0] == 0
    assert abs(result.values[1] - field[2, 2, 2]) < 6 * result.stderr[1]
    for name, expected in saved.items():
        np.testing.assert_array_equal(getattr(op, name), expected)
        assert not getattr(op, name).flags.writeable


def test_full_field_uses_given_previous_field_without_exact_access():
    def forbidden(*args):
        pytest.fail("Exact solution must not be accessed by the backend")

    problem = replace(make_heat_problem3d("sine", 4, 0), exact=forbidden)
    previous = np.full(problem.shape, 2.0)
    op = freeze_heat3d(problem, previous, previous, 0.01, 0.01)
    points = np.indices(op.shape).reshape(3, -1).T
    matrix, rhs = op.assemble()
    target = spsolve(matrix, rhs)
    result = solve_frozen_mc3d(op, points, walks=2048, seed=12, batch_size=31)
    assert result.status == "completed" and result.walks == 27 * 2048
    assert np.all(np.abs(result.values - target) < 6 * result.stderr + 1e-12)


@pytest.mark.parametrize("batch_size", [0, -1, 1.5, True, np.nan, 65537])
def test_invalid_batch_size_is_explicit_before_work(batch_size):
    with pytest.raises(ValueError, match="batch_size"):
        solve_frozen_mc3d(_operator(), batch_size=batch_size)
    with pytest.raises(ValueError, match="batch_size"):
        warmup_mc3d(batch_size=batch_size)


def test_both_warmups_are_repeatable():
    warmup_mc3d()
    warmup_mc3d(batch_size=7)
    warmup_mc3d(batch_size=65536)
    warmup_mc3d()
