"""Residual identities, frozen-operator reuse and complete-chain integration."""

import io
import json
import zipfile
from dataclasses import replace

import numpy as np
import pytest

from pdelab.frozen_solvers import solve_frozen_mc
from pdelab.heat import make_heat_problem, solve_heat
from pdelab.heat_mc import solve_heat_mc
from pdelab.heat_mc_study import heat_mc_study
from pdelab.heat_mc_views import heat_mc_archive
from pdelab.nonlinear import NonlinearSettings, make_nonlinear_problem, solve_nonlinear
from pdelab.nonlinear_study import compare_nonlinear_methods, nonlinear_targets
from pdelab.operator import FrozenOperator


@pytest.mark.parametrize("preset", ["decay", "manufactured"])
def test_heat_exact_discrete_control_eliminates_noise_with_short_final_step(preset):
    problem = make_heat_problem(preset, 8, a2=0.3)
    direct = solve_heat(problem, 0.013, 0.04, method="direct")
    points = [[3, 5], [0, 4], [8, 8]]
    result = solve_heat_mc(
        problem,
        0.013,
        0.04,
        points,
        walks=64,
        method="mc_control",
        control=direct.fields,
    )
    np.testing.assert_allclose(
        result.values, direct.fields[-1][tuple(np.array(points).T)], atol=2e-13
    )
    assert max(result.stderr) < 2e-14
    assert result.control_reused
    with zipfile.ZipFile(
        io.BytesIO(heat_mc_archive(result, problem, 0.013))
    ) as archive:
        metadata = json.loads(archive.read("experiment.json"))
    assert metadata["control_reused"]
    assert "build not included" in metadata["previous_solution_preparation"]
    assert all(row["control_reused"] for row in result.rows())


def test_heat_zero_control_reproduces_paths_and_values():
    problem = make_heat_problem("manufactured", 8)
    direct = solve_heat(problem, 0.013, 0.04, method="direct")
    ordinary = solve_heat_mc(problem, 0.013, 0.04, walks=512, seed=19)
    control = solve_heat_mc(
        problem,
        0.013,
        0.04,
        walks=512,
        seed=19,
        method="mc_control",
        control=np.zeros_like(direct.fields),
    )
    np.testing.assert_array_equal(control.values, ordinary.values)
    np.testing.assert_array_equal(control.stderr, ordinary.stderr)
    assert control.spatial_transitions == ordinary.spatial_transitions
    assert control.temporal_transitions == ordinary.temporal_transitions


@pytest.mark.parametrize("preset", ["decay", "manufactured"])
def test_heat_coarse_control_is_unbiased_for_fine_system_without_oracle(preset):
    problem = make_heat_problem(preset, 12)
    target = solve_heat(problem, 0.013, 0.04, method="direct").fields[-1, 6, 6]
    problem.exact = lambda *args: pytest.fail("Exact field must not build control")
    results = [
        solve_heat_mc(
            problem, 0.013, 0.04, walks=512, seed=s, method="mc_control", coarse_n=5
        )
        for s in range(30)
    ]
    values = np.array([r.values[0] for r in results])
    assert abs(values.mean() - target) < 5 * values.std(ddof=1) / np.sqrt(len(values))
    assert all(
        r.seconds == pytest.approx(r.preparation_seconds + r.path_seconds)
        for r in results
    )
    assert all(r.coarse_n == 5 for r in results)
    assert all(not r.control_reused for r in results)


def test_heat_control_initial_time_and_invalid_inputs():
    problem = make_heat_problem("decay", 8)
    result = solve_heat_mc(problem, T=0, method="mc_control", coarse_n=5, walks=2)
    assert result.values[0] == pytest.approx(problem.initial[4, 4])
    assert result.spatial_transitions + result.temporal_transitions == 0
    for n in [True, 3, 9, 4.5, np.nan]:
        with pytest.raises(ValueError):
            solve_heat_mc(problem, method="mc_control", coarse_n=n)
    with pytest.raises(ValueError):
        solve_heat_mc(problem, method="mc_control", control=np.zeros((2, 4, 4)))
    with pytest.raises(RuntimeError):
        solve_heat_mc(problem, T=1, walks=2, method="mc_control", max_transitions=1)


def variable_operator():
    z, x = np.meshgrid(np.linspace(0, 1, 7), np.linspace(0, 1, 7), indexing="ij")
    return FrozenOperator.from_nodes(1 + x + z, 2 + x, x + 2 * z, sigma=5)


def test_reuse_preserves_fresh_paths_and_handles_reaction_killing():
    operator = variable_operator()
    ordinary = solve_frozen_mc(operator, walks=64, seed=12)
    reused = solve_frozen_mc(operator, walks=64, seed=12, reuse=True)
    assert ordinary.walks == reused.walks == 25 * 64
    assert ordinary.steps == reused.steps
    target = operator.solve()
    fields = np.array(
        [
            solve_frozen_mc(operator, walks=128, seed=s, reuse=True).values
            for s in range(32)
        ]
    )
    error = abs(fields.mean(axis=0) - target)
    stderr = fields.std(axis=0, ddof=1) / np.sqrt(len(fields))
    assert np.all(error[1:-1, 1:-1] <= 5 * stderr[1:-1, 1:-1])


def test_exact_frozen_control_with_reuse_and_max_steps():
    operator = variable_operator()
    exact = operator.solve()
    result = solve_frozen_mc(operator, walks=16, control=exact, reuse=True)
    np.testing.assert_allclose(result.values, exact, atol=1e-12)
    assert np.max(result.stderr) < 1e-12
    with pytest.raises(RuntimeError):
        solve_frozen_mc(operator, walks=16, reuse=True, max_steps=1)


def test_heat_study_includes_control_and_independent_phases():
    study = heat_mc_study(
        make_heat_problem("decay", 8),
        dt=0.01,
        T=0.02,
        repeats=3,
        max_walks=256,
        seconds_per_method=30,
        methods=("lod", "mc", "mc_control"),
        coarse_n=4,
    )
    assert len(study["rows"]) == 9
    assert {r["method"] for r in study["rows"]} == {"lod", "mc", "mc_control"}
    trials = [r for r in study["trials"] if r["method"] == "mc_control"]
    pilots = {r["seed"] for r in trials if r["phase"] != "validation"}
    checks = {r["seed"] for r in trials if r["phase"] == "validation"}
    assert checks and not checks.intersection(pilots)


def test_nonlinear_variants_complete_and_are_exported_in_study():
    problem = make_nonlinear_problem(4, 0.5)
    settings = NonlinearSettings(
        tolerance=0.03,
        change_tolerance=0.03,
        walks=64,
        max_walks=256,
        max_outer=30,
        relaxation=0.7,
    )
    for variant in ("mc_plain", "mc", "mc_reuse"):
        result = solve_nonlinear(
            problem, 0.01, 0.02, "mc", replace(settings, mc_variant=variant)
        )
        assert result.walks > 0
        assert result.settings.mc_variant == variant
        assert result.completed or result.reason != "completed"
    comparisons = compare_nonlinear_methods(
        problem,
        0.01,
        0.02,
        (2, 2),
        2,
        NonlinearSettings(),
        settings,
        ["mc", "mc_reuse"],
    )
    assert set(comparisons) == {"mc", "mc_reuse"}
    assert all(c.stochastic[0].completed for c in comparisons.values())
    rows, evidence = nonlinear_targets(
        problem,
        0.01,
        0.02,
        repeats=2,
        budgets=[64, 256],
        settings=settings,
        variants=["mc", "mc_reuse"],
    )
    assert len(rows) == 9
    assert {r["method"] for r in rows} == {"picard", "mc", "mc_reuse"}
    assert {p["variant"] for p in evidence["pilots"]} == {"mc", "mc_reuse"}


def test_cli_exports_selected_heat_control_and_nonlinear_variants(tmp_path):
    from pdelab.cli import main as heat_cli
    from pdelab.nonlinear_cli import main as nonlinear_cli

    heat_folder = tmp_path / "heat"
    heat_cli(
        [
            "heat-mc",
            "--n",
            "8",
            "--T",
            ".02",
            "--walks",
            "64",
            "--method",
            "mc_control",
            "--coarse-n",
            "4",
            "--out",
            str(heat_folder),
        ]
    )
    metadata = json.loads((heat_folder / "experiment.json").read_text())
    assert metadata["method"] == "mc_control" and metadata["coarse_n"] == 4
    assert not metadata["control_reused"]
    assert "included" in metadata["previous_solution_preparation"]
    nonlinear_folder = tmp_path / "nonlinear"
    nonlinear_cli(
        [
            "targets",
            "--n",
            "4",
            "--T",
            ".02",
            "--repeats",
            "2",
            "--variants",
            "mc",
            "mc_reuse",
            "--budgets",
            "32",
            "--out",
            str(nonlinear_folder),
        ]
    )
    evidence = json.loads((nonlinear_folder / "target-evidence.json").read_text())
    assert {p["variant"] for p in evidence["pilots"]} == {"mc", "mc_reuse"}
