import numpy as np
import pytest

from pdelab import (
    Problem,
    make_problem,
    reference_solution,
    select_points,
    solve,
    warmup,
)
from pdelab.problem import laplacian
from pdelab.solvers import trace_walk


@pytest.fixture(scope="session", autouse=True)
def compiled():
    warmup()


@pytest.mark.parametrize("preset", ["polynomial", "affine", "hot_top"])
def test_lod_solves_discrete_system(preset):
    p = make_problem(preset, 12)
    points = select_points(12)
    ref = reference_solution(p)
    a = solve(p, points, "lod_serial", tol=1e-12)
    assert a.converged
    np.testing.assert_allclose(a.values, ref, rtol=0, atol=1e-8)
    assert a.relative_residual <= 1e-12


def test_polynomial_exact_discretization_and_sine_grid_error():
    p = make_problem("polynomial", 10)
    np.testing.assert_allclose(laplacian(p.exact, p.h), p.f[1:-1, 1:-1], atol=1e-10)
    errors = []
    for n in [12, 24]:
        p = make_problem("sine", n)
        errors.append(np.max(abs(reference_solution(p) - p.exact)))
    assert 3.8 < errors[0] / errors[1] < 4.2


@pytest.mark.parametrize("method", ["mc", "reuse", "layers"])
def test_constant_boundary_is_exact_without_noise(method):
    p = Problem.from_functions(8, lambda x, y: 0.0, lambda x, y: 7.0)
    q = select_points(8, "patch")
    r = solve(p, q, method, walks=64, batches=4)
    np.testing.assert_array_equal(r.values[q[:, 0], q[:, 1]], 7.0)
    np.testing.assert_array_equal(r.stderr[q[:, 0], q[:, 1]], 0.0)


def test_nonzero_source_sign_and_normalization():
    p = make_problem("polynomial", 12)
    q = select_points(12)
    r = solve(p, q, "mc", walks=20000, seed=134)
    assert abs(r.values[6, 6] - 62.5) < 5 * r.stderr[6, 6]
    assert r.values[6, 6] > 0


def test_reuse_single_point_degenerates_to_baseline():
    p = make_problem("polynomial", 12)
    q = select_points(12)
    a = solve(p, q, "mc", walks=1024, seed=9)
    b = solve(p, q, "reuse", walks=1024, seed=9)
    np.testing.assert_equal(a.values, b.values)
    np.testing.assert_equal(a.stderr, b.stderr)
    assert a.steps == b.steps and a.walks == b.walks


def test_reuse_counts_first_visits_and_handles_source_tails():
    p = make_problem("polynomial", 8)
    q = select_points(8, "patch")
    a = solve(p, q, "mc", walks=3000, seed=56)
    b = solve(p, q, "reuse", walks=3000, seed=56)
    y, x = q.T
    assert a.steps == b.steps
    assert np.all(b.sample_count[y, x] >= 3000)
    assert np.all(b.sample_count[y, x] <= b.walks)
    assert np.any(b.sample_count[y, x] > 3000)
    assert np.max(abs(b.values[y, x] - p.exact[y, x]) / (b.stderr[y, x] + 1e-12)) < 5


def test_layers_uncertainty_includes_full_independent_preparations():
    p = make_problem("polynomial", 8)
    q = select_points(8, "patch")
    r = solve(p, q, "layers", walks=2048, batches=16, seed=88)
    y, x = q.T
    reps = r.extras["replicates"]
    np.testing.assert_allclose(r.values[y, x], reps.mean(axis=0))
    np.testing.assert_allclose(r.stderr[y, x], reps.std(axis=0, ddof=1) / 4)
    assert r.walks > len(q) * 2048  # all preparation counted for a patch
    assert np.max(abs(r.values[y, x] - p.exact[y, x]) / (r.stderr[y, x] + 1e-12)) < 5


@pytest.mark.parametrize("method", ["mc", "reuse", "layers"])
def test_walk_limit_does_not_silently_bias_answer(method):
    p = make_problem(n=12)
    q = select_points(12)
    with pytest.raises(RuntimeError):
        solve(p, q, method, walks=64, max_steps=1)


def test_trace_source_reward_and_boundary_stop():
    p = make_problem("polynomial", 8)
    t = trace_walk(p, seed=8)
    path = t["points"]
    assert np.all(abs(np.diff(path, axis=0)).sum(axis=1) == 1)
    y, x = path[:-1].T
    expected = -p.h**2 / 4 * p.f[y, x].sum() + p.boundary[tuple(path[-1])]
    assert t["reward"] == pytest.approx(expected)
    assert np.all(path[1:-1] > 0) and np.all(path[1:-1] < p.n)


def test_parameter_validation_and_failed_convergence():
    p = make_problem(n=8)
    q = select_points(8)
    with pytest.raises(ValueError):
        solve(p, [(0, 0)])
    with pytest.raises(ValueError):
        solve(p, q, "layers", walks=4, batches=8)
    with pytest.raises(ValueError):
        solve(p, q, walks=1)
    with pytest.raises(ValueError):
        solve(p, q, max_steps=0)
    r = solve(p, q, "lod_serial", max_iter=1)
    assert not r.converged
