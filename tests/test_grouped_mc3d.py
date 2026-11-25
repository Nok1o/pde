"""Transition, independent-reference and bookkeeping tests for grouped walks."""

import numpy as np
import pytest
from scipy.sparse.linalg import spsolve

from pdelab.boundary3d import (
    FACES,
    BoundaryOperator3D,
    BoundaryProblem3D,
    make_boundary_problem3d,
)
from pdelab.boundary_solvers3d import _walks
from pdelab.grouped_mc3d import grouped_walks3d, grouped_workspace_bytes3d
from pdelab.problem3d import make_problem3d


def _coefficients(operator):
    _, reward, killing = operator.mc_coefficients()
    return operator.fixed, operator.boundary, reward, killing


def _mixed_operator():
    kinds = dict(
        zip(FACES, ["neumann", "dirichlet", "robin", "dirichlet", "robin", "dirichlet"])
    )
    kappa = 0.2 + np.arange(25).reshape(5, 5) / 10
    return BoundaryOperator3D(
        make_boundary_problem3d("smooth", 4, kinds, {"y0": kappa, "z0": 0.8})
    )


@pytest.mark.parametrize("batch_size", [1, 7, 37, 128, 65536])
def test_constant_solution_partial_groups_and_fixed_points(batch_size):
    operator = BoundaryOperator3D(
        BoundaryProblem3D.from_problem(make_problem3d("constant", 4))
    )
    points = np.array([[2, 2, 2], [0, 0, 0], [4, 2, 1], [1, 2, 3]])
    means, errors, counts, steps, samples, failed, started = grouped_walks3d(
        *_coefficients(operator), points, 37, 31, 10000, batch_size
    )
    np.testing.assert_array_equal(means, 7.0)
    np.testing.assert_array_equal(errors, 0.0)
    np.testing.assert_array_equal(counts, [37, 0, 0, 37])
    np.testing.assert_array_equal(samples, 7.0)
    assert samples.shape == (37,)
    assert failed == -1 and started == 74 and steps >= 74


def test_batch_one_reproduces_reflected_baseline_draw_order():
    operator = _mixed_operator()
    points = np.array([[2, 2, 2], [2, 2, 0], [2, 0, 0], [0, 0, 0]])
    coefficients = _coefficients(operator)
    old = _walks(*coefficients, points, 257, 310, 10000)
    new = grouped_walks3d(*coefficients, points, 257, 310, 10000, 1)
    for actual, expected in ((new[0], old[0]), (new[1], old[1]), (new[4], old[3])):
        np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(new[2], 257)
    assert new[3] == old[2] and new[5] == -1 and new[6] == 4 * 257


@pytest.mark.parametrize("batch_size", [1, 11, 4096])
def test_mixed_faces_match_independent_csr_across_seeds(batch_size):
    operator = _mixed_operator()
    points = np.array([[2, 2, 2], [2, 2, 0], [2, 0, 0], [0, 0, 0]])
    # The reference assembles physical stencil weights, independently of the
    # kernel's reflection, killing, packing, and path-estimator implementation.
    matrix, rhs = operator.assemble()
    reference = operator.boundary.copy()
    reference[~operator.fixed] = spsolve(matrix, rhs)
    target = reference[tuple(points.T)]
    ensembles = []
    variances = []
    for seed in range(16):
        means, errors, counts, _, samples, failed, started = grouped_walks3d(
            *_coefficients(operator), points, 2048, 1000 + seed, 10000, batch_size
        )
        assert failed == -1 and started == 4 * 2048
        np.testing.assert_array_equal(counts, 2048)
        assert means[0] == pytest.approx(samples.mean(), abs=2e-13)
        assert errors[0] == pytest.approx(samples.std(ddof=1) / np.sqrt(2048))
        ensembles.append(means)
        variances.append(errors**2)
    ensembles = np.asarray(ensembles)
    uncertainty = ensembles.std(axis=0, ddof=1) / np.sqrt(len(ensembles))
    assert np.all(np.abs(ensembles.mean(axis=0) - target) < 6 * uncertainty)
    # Independently seeded estimates should fluctuate on the scale reported
    # by their within-run errors, rather than replaying a stream per batch.
    ratio = ensembles.var(axis=0, ddof=1) / np.mean(variances, axis=0)
    assert np.all((ratio > 0.15) & (ratio < 3.5))


def test_affine_solution_and_fixed_values():
    problem = make_problem3d("affine", 4)
    operator = BoundaryOperator3D(BoundaryProblem3D.from_problem(problem))
    points = np.array([[2, 1, 3], [1, 3, 1], [0, 2, 3]])
    estimates = []
    for seed in range(12):
        result = grouped_walks3d(
            *_coefficients(operator), points, 4096, 800 + seed, 10000, 23
        )
        assert result[5] == -1
        assert result[0][-1] == problem.exact[tuple(points[-1])]
        assert result[1][-1] == result[2][-1] == 0
        estimates.append(result[0][:2])
    estimates = np.asarray(estimates)
    target = problem.exact[tuple(points[:2].T)]
    uncertainty = estimates.std(axis=0, ddof=1) / np.sqrt(len(estimates))
    assert np.all(np.abs(estimates.mean(axis=0) - target) < 6 * uncertainty)


@pytest.mark.parametrize("batch_size", [1, 7, 100])
def test_same_seed_and_batch_reproduce_all_outputs(batch_size):
    operator = _mixed_operator()
    args = (*_coefficients(operator), np.array([[0, 0, 0], [2, 2, 2]]), 67, 28)
    first = grouped_walks3d(*args, 10000, batch_size)
    second = grouped_walks3d(*args, 10000, batch_size)
    for actual, expected in zip(first, second):
        np.testing.assert_array_equal(actual, expected)


def test_fixed_start_consumes_no_draws_and_has_zero_work():
    operator = _mixed_operator()
    coefficients = _coefficients(operator)
    fixed_point = [4, 2, 2]
    active_point = [1, 2, 0]
    only_fixed = grouped_walks3d(*coefficients, np.array([fixed_point]), 79, 51, 1, 17)
    np.testing.assert_array_equal(only_fixed[4], operator.boundary[tuple(fixed_point)])
    assert only_fixed[2][0] == only_fixed[3] == only_fixed[6] == 0
    assert only_fixed[5] == -1
    with_fixed = grouped_walks3d(
        *coefficients, np.array([fixed_point, active_point]), 79, 51, 10000, 17
    )
    without_fixed = grouped_walks3d(
        *coefficients, np.array([active_point]), 79, 51, 10000, 17
    )
    np.testing.assert_array_equal(with_fixed[0][1:], without_fixed[0])
    np.testing.assert_array_equal(with_fixed[1][1:], without_fixed[1])
    assert with_fixed[3] == without_fixed[3]
    assert with_fixed[6] == without_fixed[6] == 79


@pytest.mark.parametrize("batch_size", [1, 6, 100])
def test_killing_one_pays_reward_before_absorption_without_boundary(batch_size):
    shape = (5, 5, 5)
    fixed = np.zeros(shape, dtype=bool)
    boundary = np.full(shape, 100.0)
    reward = np.full(shape, 3.25)
    killing = np.ones(shape)
    result = grouped_walks3d(
        fixed, boundary, reward, killing, np.array([[0, 0, 0]]), 29, 5, 1, batch_size
    )
    np.testing.assert_array_equal(result[0], 3.25)
    np.testing.assert_array_equal(result[1], 0.0)
    np.testing.assert_array_equal(result[4], 3.25)
    assert result[2][0] == result[3] == result[6] == 29
    assert result[5] == -1


@pytest.mark.parametrize("point", [(0, 0, 0), (0, 2, 0), (2, 2, 0), (2, 2, 2)])
def test_six_uniform_proposals_reflect_at_faces_edges_and_corners(point):
    n = 4
    shape = (n + 1,) * 3
    fixed = np.ones(shape, dtype=bool)
    fixed[point] = False
    boundary = np.arange(np.prod(shape), dtype=float).reshape(shape)
    reward = np.zeros(shape)
    reward[point] = 2.5
    killing = np.zeros(shape)
    expected_payoffs = []
    for axis in range(3):
        for delta in (-1, 1):
            neighbor = list(point)
            candidate = point[axis] + delta
            neighbor[axis] = candidate if 0 <= candidate <= n else point[axis] - delta
            expected_payoffs.append(2.5 + boundary[tuple(neighbor)])
    result = grouped_walks3d(
        fixed, boundary, reward, killing, np.array([point]), 24000, 183, 1, 127
    )
    assert result[5] == -1
    assert result[3] == result[6] == 24000
    values, expected_counts = np.unique(expected_payoffs, return_counts=True)
    for value, count in zip(values, expected_counts):
        frequency = np.count_nonzero(result[4] == value) / 24000
        probability = count / 6
        uncertainty = np.sqrt(probability * (1 - probability) / 24000)
        assert abs(frequency - probability) < 6 * uncertainty
    assert np.isin(result[4], expected_payoffs).all()


@pytest.mark.parametrize("batch_size", [1, 7, 100])
def test_max_steps_failure_counts_completed_and_unfinished_stages(batch_size):
    shape = (5, 5, 5)
    fixed = np.zeros(shape, dtype=bool)
    boundary = np.zeros(shape)
    reward = np.ones(shape)
    killing = np.full(shape, 0.25)
    result = grouped_walks3d(
        fixed, boundary, reward, killing, np.array([[2, 2, 2]]), 37, 51, 3, batch_size
    )
    means, errors, counts, steps, samples, failed, started = result
    assert failed == 0
    assert np.isnan(means).all() and np.isnan(errors).all()
    assert counts[0] == np.isfinite(samples).sum()
    assert counts[0] < started <= 37
    assert steps == np.nansum(samples) + (started - counts[0]) * 3
    assert np.all(
        (samples[np.isfinite(samples)] >= 1) & (samples[np.isfinite(samples)] <= 3)
    )


def test_failure_preserves_prior_counts_and_does_not_start_later_points():
    shape = (11, 11, 11)
    fixed = np.zeros(shape, dtype=bool)
    fixed[0, 0, 0] = True
    boundary = np.full(shape, 9.0)
    reward = np.ones(shape)
    killing = np.zeros(shape)
    killing[2, 2, 2] = 1
    points = np.array([[0, 0, 0], [2, 2, 2], [7, 7, 7], [5, 5, 5]])
    means, errors, counts, steps, samples, failed, started = grouped_walks3d(
        fixed, boundary, reward, killing, points, 29, 91, 1, 7
    )
    assert failed == 2 and steps == started == 36
    np.testing.assert_array_equal(counts, [0, 29, 0, 0])
    np.testing.assert_array_equal(means[:2], [9.0, 1.0])
    np.testing.assert_array_equal(errors[:2], 0.0)
    assert np.isnan(means[2:]).all() and np.isnan(errors[2:]).all()
    np.testing.assert_array_equal(samples, 9.0)


def test_workspace_counts_arrays_and_caps_particle_buffers_at_walks():
    assert grouped_workspace_bytes3d(37, 7, 4) == 48 * 7 + 24 * 4 + 8 * 37
    assert grouped_workspace_bytes3d(37, 65536, 4) == 48 * 37 + 24 * 4 + 8 * 37
    assert grouped_workspace_bytes3d(37, 7) == 48 * 7 + 24 + 8 * 37


def test_kernel_does_not_modify_operator_inputs():
    operator = _mixed_operator()
    coefficients = _coefficients(operator)
    copies = [array.copy() for array in coefficients]
    points = np.array([[0, 0, 0], [2, 2, 2]])
    original_points = points.copy()
    grouped_walks3d(*coefficients, points, 37, 28, 10000, 7)
    for actual, expected in zip(coefficients, copies):
        np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(points, original_points)
