"""PDE-28 stage 2: residual identity, coarse construction and complete costs."""

import json
from dataclasses import replace

import numpy as np
import pytest
from scipy.sparse.linalg import spsolve

from pdelab import heat3d_control as module
from pdelab.control3d import _interpolate3d
from pdelab.heat3d_control import (
    solve_controlled_frozen_mc3d,
    solve_frozen_correction_mc3d,
)
from pdelab.heat3d_mc import solve_frozen_mc3d
from pdelab.heat3d_mc_cli import main
from pdelab.heat3d_operator import HeatOperator3D, freeze_heat3d
from pdelab.heat3d_problem import make_heat_problem3d


def context(n=8, alpha=0.5):
    p = make_heat_problem3d("polynomial", n, alpha)
    previous = p.initial_values()
    return p, previous, freeze_heat3d(p, previous, previous, 0.013, 0.013)


def test_zero_control_reproduces_ordinary_mc_exactly():
    _, _, op = context()
    points = [[0, 1, 1], [4, 4, 4]]
    plain = solve_frozen_mc3d(op, points, walks=256, seed=51)
    corrected = solve_frozen_correction_mc3d(
        op, np.zeros(op.shape), points, walks=256, seed=51
    )
    np.testing.assert_array_equal(plain.values, corrected.values)
    np.testing.assert_array_equal(plain.stderr, corrected.stderr)
    assert plain.walks == corrected.walks and plain.transitions == corrected.transitions
    assert corrected.control_origin == "supplied_construction_excluded"


def test_exact_discrete_control_gives_zero_variance_without_reference_access():
    _, _, op = context(8)
    field = np.random.default_rng(12).normal(size=op.shape)
    manufactured = op.with_rhs(op.apply(field))
    result = solve_frozen_correction_mc3d(
        manufactured, field, [[0, 1, 1], [4, 4, 4]], walks=128
    )
    np.testing.assert_array_equal(
        result.values, field[tuple(np.array([[0, 1, 1], [4, 4, 4]]).T)]
    )
    np.testing.assert_array_equal(result.stderr, 0)


def test_rhs_contains_full_defect_including_nonzero_boundary(monkeypatch):
    _, _, op = context()
    control = np.full(op.shape, 0.25)
    original_rhs = op.rhs.copy()
    original_weights = op.weights.copy()
    matrix, b = op.assemble()
    exact = spsolve(matrix, b).reshape(op.shape)
    expected = op.rhs - op.apply(control)
    np.testing.assert_allclose(op.apply(exact - control), expected, atol=2e-11)
    backend = module.solve_frozen_mc3d

    def checked(correction, *args, **kwargs):
        assert np.shares_memory(correction.weights, op.weights)
        np.testing.assert_array_equal(correction.rhs, expected)
        np.testing.assert_array_equal(
            correction.fixed_values[op.fixed],
            op.fixed_values[op.fixed] - control[op.fixed],
        )
        return backend(correction, *args, **kwargs)

    monkeypatch.setattr(module, "solve_frozen_mc3d", checked)
    result = solve_frozen_correction_mc3d(
        op, control, [[0, 1, 1], [4, 4, 4]], walks=20000, seed=1
    )
    assert result.values[0] == op.fixed_values[0, 1, 1]
    assert abs(result.values[1] - exact[4, 4, 4]) < 5 * result.stderr[1]
    np.testing.assert_array_equal(op.rhs, original_rhs)
    np.testing.assert_array_equal(op.weights, original_weights)


@pytest.mark.parametrize("n,coarse_n", [(8, 4), (10, 4), (8, 5)])
@pytest.mark.parametrize("alpha", [0, 0.5])
def test_independent_full_coarse_preparations_match_fine_frozen_direct(
    n, coarse_n, alpha
):
    p, previous, op = context(n, alpha)
    points = np.array([[n // 2] * 3, [0, 1, 1]])
    matrix, b = op.assemble()
    target = spsolve(matrix, b).reshape(op.shape)[tuple(points.T)]
    runs = [
        solve_controlled_frozen_mc3d(
            p,
            previous,
            previous,
            0.013,
            0.013,
            points,
            coarse_n=coarse_n,
            walks=2048,
            seed=r,
        )
        for r in range(12)
    ]
    assert all(r.status == "completed" for r in runs)
    values = np.array([r.values for r in runs])
    se = values.std(axis=0, ddof=1) / np.sqrt(len(runs))
    assert np.all(np.abs(values.mean(axis=0) - target) < 5 * se + 1e-11)
    assert all(
        r.coarse_n == coarse_n and r.control_origin == "fresh_coarse_grid" for r in runs
    )
    for r in runs:
        assert r.preparation_seconds <= r.seconds
        assert (
            r.coarse_seconds
            + r.operator_seconds
            + r.interpolation_seconds
            + r.defect_seconds
            <= r.preparation_seconds + 1e-12
        )
        assert r.coarse_iterations > 0 and r.walks == 2048
        assert r.coarse_relative_residual <= 1e-8
        assert r.peak_rss_bytes is None


def test_good_coarse_field_reduces_variance_without_claiming_universal_speed():
    p, previous, op = context(8, 0)
    plain = [solve_frozen_mc3d(op, walks=512, seed=r).values[0] for r in range(16)]
    controlled = [
        solve_controlled_frozen_mc3d(
            p, previous, previous, 0.013, 0.013, coarse_n=4, walks=512, seed=r
        ).values[0]
        for r in range(16)
    ]
    assert np.std(controlled, ddof=1) < 0.2 * np.std(plain, ddof=1)


def test_coarse_solve_reads_only_known_data_and_restricted_caller_fields(monkeypatch):
    p, previous, _ = context(10)

    def forbidden(*args, **kwargs):
        raise AssertionError("No exact or fine direct solution")

    p = replace(p, exact=forbidden)
    previous = previous + 0.37
    iterate = previous + 0.21
    freeze = module.freeze_heat3d
    solves = module.solve_lines3d
    observed = []

    def check_freeze(problem, past, current, *args):
        if problem.n == 4:
            axis = np.linspace(0, 1, 11)
            coarse = np.linspace(0, 1, 5)
            np.testing.assert_allclose(past, _interpolate3d(previous, axis, coarse))
            np.testing.assert_allclose(current, _interpolate3d(iterate, axis, coarse))
            assert problem.exact is None
        return freeze(problem, past, current, *args)

    def check_solve(op, *args, **kwargs):
        observed.append(op.n)
        assert op.n == 4
        return solves(op, *args, **kwargs)

    monkeypatch.setattr(HeatOperator3D, "assemble", forbidden)
    monkeypatch.setattr(module, "freeze_heat3d", check_freeze)
    monkeypatch.setattr(module, "solve_lines3d", check_solve)
    result = solve_controlled_frozen_mc3d(
        p, previous, iterate, 0.013, 0.013, walks=64, coarse_n=4
    )
    assert result.status == "completed" and observed == [4]


def test_prolongated_boundary_defect_is_not_overwritten(monkeypatch):
    p, previous, _ = context()
    backend = module.solve_frozen_correction_mc3d
    seen = []

    def check(op, control, *args, **kwargs):
        seen.append(np.max(np.abs((op.rhs - control)[op.fixed])))
        return backend(op, control, *args, **kwargs)

    monkeypatch.setattr(module, "solve_frozen_correction_mc3d", check)
    result = solve_controlled_frozen_mc3d(
        p, previous, previous, 0.013, 0.013, [[0, 1, 1]], walks=64
    )
    assert seen[0] > 0.001
    assert result.values[0] == p.face_values("z0", 0.013)[1, 1]
    assert result.walks == result.transitions == 0


def test_full_field_controlled_query_matches_fine_system():
    p, previous, op = context(5)
    points = np.indices(op.shape).reshape(3, -1).T
    result = solve_controlled_frozen_mc3d(
        p, previous, previous, 0.013, 0.013, points, coarse_n=4, walks=512, seed=15
    )
    matrix, rhs = op.assemble()
    target = spsolve(matrix, rhs)
    assert result.status == "completed" and result.walks == 64 * 512
    assert np.all(np.abs(result.values - target) < 6 * result.stderr + 1e-11)


def test_known_value_survives_extreme_supplied_control_cancellation():
    _, _, op = context()
    control = np.zeros(op.shape)
    control[op.fixed] = 1e300
    result = solve_frozen_correction_mc3d(op, control, [[0, 1, 1]], walks=2)
    assert result.status == "completed" and result.values[0] == op.fixed_values[0, 1, 1]
    assert result.stderr[0] == 0 and result.walks == 0


def test_coarse_and_mc_failures_keep_preparation_and_no_solution():
    p, previous, _ = context()
    limited = solve_controlled_frozen_mc3d(
        p,
        previous,
        previous,
        0.013,
        0.013,
        coarse_max_iterations=1,
        coarse_tolerance=1e-14,
        walks=64,
    )
    assert limited.status == "linear_limit" and limited.reason.startswith("coarse_")
    assert limited.coarse_iterations == 1 and limited.values is None
    assert limited.walks == limited.transitions == 0
    assert limited.seconds >= limited.coarse_seconds > 0
    incomplete = solve_controlled_frozen_mc3d(
        p, previous, previous, 0.013, 0.013, walks=64, max_walks=1
    )
    assert incomplete.status == "walk_budget" and incomplete.values is None
    assert incomplete.walks == 1 and incomplete.coarse_iterations > 0
    assert (
        incomplete.seconds
        >= incomplete.preparation_seconds
        >= incomplete.coarse_seconds
    )


def test_workspace_guard_runs_before_fine_or_coarse_construction(monkeypatch):
    p, previous, op = context()

    def forbidden(*args, **kwargs):
        raise AssertionError("Memory guard must run before construction")

    monkeypatch.setattr(module, "freeze_heat3d", forbidden)
    result = solve_controlled_frozen_mc3d(
        p, previous, previous, 0.013, 0.013, max_workspace_bytes=1
    )
    assert result.status == "memory_limit" and result.coarse_n == 4
    assert result.values is None and result.walks == 0
    supplied = solve_frozen_correction_mc3d(
        op, np.zeros(op.shape), max_workspace_bytes=1
    )
    assert supplied.status == "memory_limit" and supplied.walks == 0


@pytest.mark.parametrize("coarse_n", [3, 8, 9, 4.5, True])
def test_coarse_grid_must_be_strictly_smaller(coarse_n):
    p, previous, _ = context()
    with pytest.raises(ValueError):
        solve_controlled_frozen_mc3d(
            p, previous, previous, 0.013, 0.013, coarse_n=coarse_n
        )


def test_cli_exports_coarse_costs_and_failed_coarse_status(tmp_path):
    out = tmp_path / "controlled"
    assert (
        main(
            [
                "--n",
                "8",
                "--method",
                "mc_control",
                "--coarse-n",
                "4",
                "--walks",
                "128",
                "--direct",
                "--out",
                str(out),
            ]
        )
        == 0
    )
    data = json.loads((out / "result.json").read_text(encoding="utf-8"))
    assert data["result"]["control_origin"] == "fresh_coarse_grid"
    assert data["result"]["coarse_n"] == 4 and data["result"]["coarse_iterations"] > 0
    assert data["result"]["coarse_seconds"] > 0
    assert data["metadata"]["configuration"]["method"] == "mc_control"
    assert data["metadata"]["physical_layers_accepted"] == 0
    assert data["rows"][0]["direct_reference"] is not None
    out = tmp_path / "failed"
    with pytest.raises(SystemExit) as error:
        main(
            [
                "--method",
                "mc_control",
                "--coarse-max-iterations",
                "1",
                "--coarse-tol",
                "1e-14",
                "--out",
                str(out),
            ]
        )
    assert error.value.code == 2
    data = json.loads((out / "result.json").read_text(encoding="utf-8"))
    assert data["result"]["status"] == "linear_limit"
    assert data["result"]["values"] is None and data["result"]["walks"] == 0
