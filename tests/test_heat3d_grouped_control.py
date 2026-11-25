"""Composition of grouped paths and a fresh coarse-grid control, PDE-29."""

import numpy as np
import pytest

from pdelab import heat3d_control as module
from pdelab.heat3d_operator import freeze_heat3d
from pdelab.heat3d_problem import make_heat_problem3d


@pytest.fixture(params=["fresh", "supplied"])
def controlled_run(request):
    problem = make_heat_problem3d("polynomial", 8, 0.5)
    previous = problem.initial_values()
    operator = freeze_heat3d(problem, previous, previous, 0.013, 0.013)
    control = previous.copy()
    points = [[0, 0, 0], [4, 4, 4], [1, 2, 3], [8, 8, 8]]

    def run(**settings):
        if request.param == "fresh":
            return module.solve_controlled_frozen_mc3d(
                problem, previous, previous, 0.013, 0.013, points, **settings
            )
        return module.solve_frozen_correction_mc3d(
            operator, control, points, **settings
        )

    return run


@pytest.mark.parametrize(
    "limits,status",
    [
        ({}, "completed"),
        ({"max_walks": 3}, "walk_budget"),
        ({"max_steps": 1}, "path_limit"),
        ({"max_transitions": 1}, "walk_budget"),
    ],
)
def test_group_of_one_matches_baseline_and_failure_counters(
    controlled_run, limits, status
):
    plain = controlled_run(walks=37, seed=3, **limits)
    grouped = controlled_run(walks=37, seed=3, batch_size=1, **limits)
    assert grouped.status == plain.status == status
    assert grouped.reason == plain.reason
    assert grouped.method == "mc_control_grouped" and grouped.batch_size == 1
    for name in ("values", "stderr", "completed_paths", "diagnostic_values"):
        np.testing.assert_array_equal(getattr(grouped, name), getattr(plain, name))
    assert grouped.walks == plain.walks
    assert grouped.transitions == plain.transitions
    assert grouped.coarse_iterations == plain.coarse_iterations
    assert grouped.coarse_relative_residual == plain.coarse_relative_residual


@pytest.mark.parametrize("batch_size", [7, 128, 65536])
def test_remainder_and_large_groups_preserve_reproducibility_and_costs(
    controlled_run, batch_size
):
    first = controlled_run(walks=37, seed=19, batch_size=batch_size)
    second = controlled_run(walks=37, seed=19, batch_size=batch_size)
    assert first.status == second.status == "completed"
    assert first.method == "mc_control_grouped" and first.batch_size == batch_size
    for name in ("values", "stderr", "completed_paths", "diagnostic_values"):
        np.testing.assert_array_equal(getattr(first, name), getattr(second, name))
    np.testing.assert_array_equal(first.completed_paths, [0, 37, 37, 0])
    assert first.walks == second.walks == 74
    assert first.transitions == second.transitions
    assert first.seconds >= first.preparation_seconds >= first.coarse_seconds
    assert first.peak_rss_bytes is None


def test_group_buffers_are_budgeted_before_preparation(controlled_run, monkeypatch):
    plain = controlled_run(walks=37)
    grouped = controlled_run(walks=37, batch_size=65536)
    assert grouped.workspace_bytes == plain.workspace_bytes + 40 * 37
    limit = plain.workspace_bytes + 40 * 7 - 1

    def forbidden(*args, **kwargs):
        pytest.fail("Reject insufficient memory before construction or JIT warmup")

    monkeypatch.setattr(module, "freeze_heat3d", forbidden)
    monkeypatch.setattr(module, "warmup_mc3d", forbidden)
    monkeypatch.setattr(module, "warmup_lines3d", forbidden)
    result = controlled_run(walks=37, batch_size=7, max_workspace_bytes=limit)
    assert result.status == "memory_limit" and result.workspace_bytes == limit + 1
    assert result.method == "mc_control_grouped" and result.batch_size == 7
    assert result.values is None and result.diagnostic_values is None
    assert result.walks == result.transitions == 0


def test_warmup_selects_grouped_kernel_before_coarse_timer(controlled_run, monkeypatch):
    observed = []
    warmup = module.warmup_mc3d

    def record(batch_size=None):
        observed.append(batch_size)
        return warmup(batch_size=batch_size)

    monkeypatch.setattr(module, "warmup_mc3d", record)
    result = controlled_run(walks=37, batch_size=7)
    assert result.status == "completed"
    assert observed and all(size == 7 for size in observed)


def test_incomplete_group_never_returns_a_solution(controlled_run):
    result = controlled_run(walks=37, batch_size=7, max_walks=40)
    assert result.status == "walk_budget" and result.walks == 40
    np.testing.assert_array_equal(result.completed_paths, [0, 37, 3, 0])
    assert result.values is None and result.stderr is None
    assert result.method == "mc_control_grouped" and result.batch_size == 7


def test_failed_coarse_solve_preserves_group_identity_and_does_not_start_mc(
    monkeypatch,
):
    problem = make_heat_problem3d("polynomial", 8, 0.5)
    previous = problem.initial_values()

    def forbidden(*args, **kwargs):
        pytest.fail("A failed coarse solve must not start correction paths")

    monkeypatch.setattr(module, "solve_frozen_mc3d", forbidden)
    result = module.solve_controlled_frozen_mc3d(
        problem,
        previous,
        previous,
        0.013,
        0.013,
        walks=37,
        batch_size=7,
        coarse_max_iterations=1,
        coarse_tolerance=1e-14,
    )
    assert result.status == "linear_limit" and result.coarse_iterations == 1
    assert result.method == "mc_control_grouped" and result.batch_size == 7
    assert result.control_origin == "fresh_coarse_grid"
    assert result.seconds >= result.preparation_seconds >= result.coarse_seconds > 0
    assert result.values is None and result.walks == result.transitions == 0


@pytest.mark.parametrize("batch_size", [0, -1, 1.5, True, np.nan, 65537])
def test_invalid_group_size_rejected_before_work(
    controlled_run, monkeypatch, batch_size
):
    def forbidden(*args, **kwargs):
        pytest.fail("Invalid group size must be rejected before work")

    monkeypatch.setattr(module, "freeze_heat3d", forbidden)
    monkeypatch.setattr(module, "warmup_mc3d", forbidden)
    with pytest.raises(ValueError, match="batch_size"):
        controlled_run(batch_size=batch_size)
