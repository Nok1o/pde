"""Fine-grid residual identity, unbiased paths and complete preparation costs."""

import json

import numpy as np
import pytest

from pdelab.cli3d import main
from pdelab.control3d import prepare_control3d
from pdelab.experiment3d import Config3D, run_experiment3d
from pdelab.problem3d import Problem3D, make_problem3d, reference_solution3d
from pdelab.research3d import StudySettings3D, accuracy_study3d
from pdelab.solvers3d import solve3d, warmup3d


def test_zero_control_preserves_ordinary_paths_and_samples():
    problem = make_problem3d(n=8)
    points = [[2, 3, 5], [4, 4, 4]]
    control = prepare_control3d(problem, approximation=np.zeros_like(problem.f))
    ordinary = solve3d(problem, points, walks=256, seed=19)
    corrected = solve3d(
        problem, points, "mc_control", walks=256, seed=19, control=control
    )
    np.testing.assert_equal(corrected.values, ordinary.values)
    np.testing.assert_equal(corrected.stderr, ordinary.stderr)
    np.testing.assert_equal(corrected.extras["samples"], ordinary.extras["samples"])
    assert corrected.steps == ordinary.steps and corrected.walks == ordinary.walks


def test_exact_discrete_control_removes_noise_and_preserves_all_faces():
    problem = Problem3D.from_functions(
        8,
        lambda x, y, z: 12,
        lambda x, y, z: 1 + x**2 + 2 * y**2 + 3 * z**2,
    )
    exact = reference_solution3d(problem)
    control = prepare_control3d(problem, approximation=exact)
    points = np.array(
        [
            [2, 3, 5],
            [0, 3, 4],
            [8, 3, 4],
            [3, 0, 4],
            [3, 8, 4],
            [3, 4, 0],
            [3, 4, 8],
            [0, 0, 0],
        ]
    )
    result = solve3d(problem, points, "mc_control", walks=64, control=control)
    indices = tuple(points.T)
    np.testing.assert_allclose(result.values[indices], exact[indices], atol=1e-12)
    assert np.max(result.stderr[indices]) < 1e-12
    assert result.walks == 64
    np.testing.assert_array_equal(
        result.sample_count[indices], [64, 0, 0, 0, 0, 0, 0, 0]
    )
    assert np.isnan(result.values[1, 1, 1])


def test_nonnested_coarse_control_matches_fine_system_without_oracle(monkeypatch):
    import pdelab.control3d as module

    problem = Problem3D.from_functions(
        12,
        lambda x, y, z: 12,
        lambda x, y, z: 1 + x**2 + 2 * y**2 + 3 * z**2,
    )
    target = reference_solution3d(problem)[3, 5, 8]
    original = module.reference_solution3d

    def coarse_only(p):
        assert p.n == 5 and p.exact is None
        return original(p)

    monkeypatch.setattr(module, "reference_solution3d", coarse_only)
    problem.exact = np.full_like(problem.f, np.nan)
    control = prepare_control3d(problem, coarse_n=5)
    values = [
        solve3d(
            problem, [[3, 5, 8]], "mc_control", walks=1024, seed=seed, control=control
        ).values[3, 5, 8]
        for seed in range(24)
    ]
    assert abs(np.mean(values) - target) < 5 * np.std(values, ddof=1) / np.sqrt(
        len(values)
    )


def test_control_reduces_variance_for_sine():
    problem = make_problem3d(n=16)
    control = prepare_control3d(problem, coarse_n=4)
    samples = {}
    for method in ("mc", "mc_control"):
        samples[method] = [
            solve3d(
                problem,
                [[8, 8, 8]],
                method,
                walks=512,
                seed=seed,
                control=control if method == "mc_control" else None,
            ).values[8, 8, 8]
            for seed in range(24)
        ]
    assert np.var(samples["mc_control"]) < np.var(samples["mc"])


def test_preparation_is_timed_on_each_run(monkeypatch):
    import pdelab.control3d as controls
    import pdelab.solvers3d as solvers

    warmup3d()
    clock = [0.0]
    direct = controls.reference_solution3d
    paths = solvers._mc3d

    def build(problem):
        clock[0] += 2
        return direct(problem)

    def walk(*args):
        clock[0] += 3
        return paths(*args)

    monkeypatch.setattr(controls, "perf_counter", lambda: clock[0])
    monkeypatch.setattr(solvers, "perf_counter", lambda: clock[0])
    monkeypatch.setattr(controls, "reference_solution3d", build)
    monkeypatch.setattr(solvers, "_mc3d", walk)
    for _ in range(2):
        result = solve3d(make_problem3d(n=8), [[4, 4, 4]], "mc_control", walks=64)
        assert result.elapsed == 5
        assert result.extras["preparation_seconds"] == 2
        assert result.extras["path_seconds"] == 3
        assert not result.extras["control_reused"]


def test_invalid_control_and_truncation_are_rejected():
    problem = make_problem3d(n=8)
    for n in (True, 3, 9, 4.5, np.nan):
        with pytest.raises(ValueError):
            prepare_control3d(problem, coarse_n=n)
    with pytest.raises(ValueError):
        prepare_control3d(make_problem3d(n=32), coarse_n=17)
    with pytest.raises(ValueError):
        prepare_control3d(problem, approximation=np.zeros((9, 9)))
    control = prepare_control3d(problem)
    with pytest.raises(ValueError):
        control.values.setflags(write=True)
    with pytest.raises(ValueError):
        solve3d(make_problem3d("affine", 8), [[4, 4, 4]], "mc_control", control=control)
    with pytest.raises(RuntimeError, match="предел шагов"):
        solve3d(problem, [[4, 4, 4]], "mc_control", walks=2, max_steps=1)


def test_large_fine_grid_coarse_limit_and_independent_study(tmp_path):
    experiment = run_experiment3d(Config3D(n=24, walks=64, coarse_n=5), ["mc_control"])
    assert experiment.grid_reference is None
    assert experiment.metrics[0]["coarse_n"] == 5
    assert experiment.metrics[0]["preparation_seconds"] > 0
    study = accuracy_study3d(
        StudySettings3D(
            Config3D(problem="affine", n=8, coarse_n=5),
            repeats=2,
            max_walks=64,
            methods=("mc_control",),
        )
    )
    assert all(row["seconds"] is not None for row in study.rows)
    pilot_seeds = {t["seed"] for t in study.trials if t["phase"] == "pilot"}
    check_seeds = {t["seed"] for t in study.trials if t["phase"] == "final"}
    assert check_seeds and pilot_seeds.isdisjoint(check_seeds)
    assert all(
        t["coarse_n"] == 5 and t["preparation_seconds"] > 0 for t in study.trials
    )
    main(
        [
            "run",
            "--n",
            "8",
            "--coarse-n",
            "5",
            "--walks",
            "64",
            "--methods",
            "mc_control",
            "--out",
            str(tmp_path),
        ]
    )
    saved = json.loads((tmp_path / "experiment.json").read_text())
    assert saved["config"]["coarse_n"] == saved["metrics"][0]["coarse_n"] == 5
