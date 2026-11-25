"""Group selection throughout the full nonlinear time history, PDE-29."""

from dataclasses import replace

import numpy as np
import pytest

from pdelab import heat3d_picard_mc as sampling
from pdelab import heat3d_solver as solver
from pdelab.boundary3d import FACES
from pdelab.heat3d_picard_mc import GROUPED_MC_METHODS
from pdelab.heat3d_problem import make_heat_problem3d
from pdelab.heat3d_solver import HeatSettings3D, solve_heat3d

SETTINGS = HeatSettings3D(
    tolerance=0.2, change_tolerance=0.12, walks=1024, max_outer=12, batch_size=7
)
PAIRS = (("mc", "mc_grouped"), ("mc_control", "mc_control_grouped"))


def problem():
    return make_heat_problem3d("polynomial", 5, 0.5, kinds={face: 3 for face in FACES})


@pytest.mark.parametrize("plain_method,grouped_method", PAIRS)
@pytest.mark.parametrize(
    "limit,status",
    [
        ({}, "completed"),
        ({"max_walks": 3}, "walk_budget"),
        ({"max_transitions": 1}, "walk_budget"),
        ({"max_path_steps": 1}, "path_limit"),
        (
            {"tolerance": 1e-12, "change_tolerance": 1e-12, "max_outer": 1},
            "outer_limit",
        ),
    ],
)
def test_group_one_reproduces_complete_history_and_rejections(
    plain_method, grouped_method, limit, status
):
    p = problem()
    settings = replace(SETTINGS, batch_size=1, **limit)
    args = dict(dt=0.002, T=0.003, settings=settings)
    plain = solve_heat3d(p, method=plain_method, **args)
    grouped = solve_heat3d(p, method=grouped_method, **args)
    assert plain.status == grouped.status == status
    assert plain.batch_size is None and grouped.batch_size == 1
    assert grouped.method == grouped_method
    for name in (
        "reason",
        "completed_until",
        "physical_steps",
        "outer_iterations",
        "inner_iterations",
        "walks",
        "completed_walks",
        "transitions",
        "validation_walks",
        "coarse_iterations",
        "diagnostic_time",
        "history",
    ):
        assert getattr(plain, name) == getattr(grouped, name)
    np.testing.assert_array_equal(plain.values, grouped.values)
    np.testing.assert_array_equal(plain.diagnostic_field, grouped.diagnostic_field)
    assert grouped.workspace_bytes == plain.workspace_bytes + 40


@pytest.mark.parametrize("plain_method,grouped_method", PAIRS)
def test_group_memory_guard_precedes_preparation_and_selected_warmup(
    monkeypatch, plain_method, grouped_method
):
    p = problem()
    settings = replace(SETTINGS, walks=37)
    baseline = solve_heat3d(
        p, method=plain_method, settings=replace(settings, max_workspace_bytes=1)
    )
    assert baseline.status == "memory_limit"
    for batch_size in (7, 65536):
        extra = 40 * min(batch_size, settings.walks)

        def forbidden(*args, **kwargs):
            pytest.fail("Insufficient memory must fail before warmup or preparation")

        with monkeypatch.context() as patch:
            patch.setattr(solver, "warmup_mc3d", forbidden)
            patch.setattr(solver, "warmup_lines3d", forbidden)
            patch.setattr(solver, "PicardSampler3D", forbidden)
            failed = solve_heat3d(
                p,
                method=grouped_method,
                settings=replace(
                    settings,
                    batch_size=batch_size,
                    max_workspace_bytes=baseline.workspace_bytes + extra - 1,
                ),
                operator_factory=forbidden,
            )
        assert failed.status == "memory_limit" and failed.values is None
        assert failed.workspace_bytes == baseline.workspace_bytes + extra
        assert failed.method == grouped_method and failed.batch_size == batch_size
        assert failed.walks == failed.transitions == failed.warmup_seconds == 0


@pytest.mark.parametrize("method", GROUPED_MC_METHODS)
def test_both_updates_and_independent_checks_use_grouped_kernel(monkeypatch, method):
    p = problem()
    draws = []
    name = (
        "_controlled_estimate"
        if method == "mc_control_grouped"
        else "solve_frozen_mc3d"
    )
    backend = getattr(sampling, name)
    warmed = []
    warmup = solver.warmup_mc3d

    def record(*args, **kwargs):
        options = kwargs["settings"] if method == "mc_control_grouped" else kwargs
        draws.append((options["batch_size"], options["seed"]))
        return backend(*args, **kwargs)

    def record_warmup(batch_size=None):
        warmed.append(batch_size)
        return warmup(batch_size=batch_size)

    monkeypatch.setattr(sampling, name, record)
    monkeypatch.setattr(solver, "warmup_mc3d", record_warmup)
    run = solve_heat3d(p, method=method, dt=0.002, T=0.003, settings=SETTINGS)
    assert run.status == "completed" and run.physical_steps == 2
    expected_seeds = [
        row[key]
        for row in run.history
        for key in ("update_seed", "validation_seed")
        if key in row
    ]
    assert len(expected_seeds) == len(set(expected_seeds))
    assert draws == [(SETTINGS.batch_size, seed) for seed in expected_seeds]
    assert warmed == [SETTINGS.batch_size]
    assert run.validation_walks > 0


@pytest.mark.parametrize("method", GROUPED_MC_METHODS)
def test_known_output_still_advances_full_field_and_t_zero_uses_no_groups(method):
    p = make_heat_problem3d("polynomial", 5, 0.5)
    points = np.indices(p.shape).reshape(3, -1).T
    args = dict(method=method, dt=0.002, T=0.003, settings=SETTINGS)
    known = solve_heat3d(p, [[0, 0, 0]], **args)
    full = solve_heat3d(p, points, **args)
    assert known.status == full.status == "completed"
    assert known.walks == full.walks > 0
    assert known.transitions == full.transitions
    assert known.validation_walks == full.validation_walks
    assert known.history == full.history
    assert known.values[0] == full.values[0] == p.face_values("z0", 0.003)[0, 0]
    initial = solve_heat3d(p, points, method=method, T=0, settings=SETTINGS)
    np.testing.assert_array_equal(initial.values, p.initial_values().ravel())
    assert initial.batch_size == SETTINGS.batch_size
    assert initial.walks == initial.transitions == initial.warmup_seconds == 0


@pytest.mark.parametrize("method", GROUPED_MC_METHODS)
@pytest.mark.parametrize("batch_size", [0, -1, 1.5, True, np.nan, 65537])
def test_invalid_batch_rejected_before_work(monkeypatch, method, batch_size):
    def forbidden(*args, **kwargs):
        pytest.fail("Invalid batch must be rejected before numerical work")

    monkeypatch.setattr(solver, "warmup_mc3d", forbidden)
    with pytest.raises(ValueError, match="batch_size"):
        solve_heat3d(
            problem(), method=method, settings=replace(SETTINGS, batch_size=batch_size)
        )
