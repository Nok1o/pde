import numpy as np
import pytest

from pdelab.control import prepare_control
from pdelab.problem import (
    Problem,
    laplacian,
    make_problem,
    reference_solution,
    select_points,
)
from pdelab.solvers import solve


def test_zero_control_reproduces_mc_and_preserves_unknown_nodes():
    p = make_problem("sine", 12)
    points = select_points(p.n, "patch")
    control = prepare_control(p, approximation=np.zeros_like(p.f))
    ordinary = solve(p, points, "mc", walks=128)
    corrected = solve(p, points, "mc_control", walks=128, control=control)
    np.testing.assert_array_equal(ordinary.values, corrected.values)
    np.testing.assert_array_equal(ordinary.stderr, corrected.stderr)
    np.testing.assert_array_equal(
        ordinary.extras["samples"], corrected.extras["samples"]
    )
    assert ordinary.steps == corrected.steps
    assert corrected.extras["preparation_seconds"] == 0


def test_exact_discrete_control_has_zero_variance_and_correct_sign():
    p = make_problem("polynomial", 16)
    v = p.exact.copy()
    f = np.zeros_like(v)
    f[1:-1, 1:-1] = laplacian(v, p.h)
    p = Problem("manufactured discrete", p.n, f, v)
    control = prepare_control(p, approximation=v)
    result = solve(p, select_points(p.n), "mc_control", control=control, walks=64)
    assert np.count_nonzero(control.source) == 0
    assert result.values[8, 8] == v[8, 8]
    assert result.stderr[8, 8] == 0


@pytest.mark.parametrize("n", [8, 13, 16])
def test_control_is_fixed_and_residual_solution_reconstructs_original(n):
    p = make_problem("sine", n)
    control = prepare_control(p)
    residual = Problem("residual", n, control.source, control.boundary)
    np.testing.assert_allclose(
        control.values + reference_solution(residual), reference_solution(p), atol=1e-10
    )
    with pytest.raises(ValueError):
        control.values.setflags(write=True)
    p.f[1, 1] += 1
    with pytest.raises(ValueError, match="другой задачи"):
        solve(p, select_points(n), "mc_control", control=control)


def test_control_distribution_agrees_with_independent_grid_reference():
    p = make_problem("sine", 16)
    points = np.array([[5, 7]])
    control = prepare_control(p)
    estimates = [
        solve(p, points, "mc_control", control=control, walks=1024, seed=s)
        for s in range(20)
    ]
    values = np.array([r.values[5, 7] for r in estimates])
    assert abs(values.mean() - reference_solution(p)[5, 7]) < 5 * values.std(
        ddof=1
    ) / np.sqrt(len(values))


@pytest.mark.parametrize("mode", ["patch", "field"])
def test_accuracy_study_supports_multiple_targets(mode):
    from pdelab.experiment import Config
    from pdelab.research import StudySettings, accuracy_study

    study = accuracy_study(
        StudySettings(Config(n=8, mode=mode), repeats=2, max_walks=64)
    )
    assert isinstance(study.reference_value, list)
    assert all(np.isfinite(t["relative_error_pct"]) for t in study.trials)
