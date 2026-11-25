from dataclasses import replace

import numpy as np
import pytest

from pdelab.boundary3d import FACES, make_boundary_problem3d
from pdelab.boundary_solvers3d import solve_boundary3d
from pdelab.grouped_solver3d import GroupedWalkFailure, solve_grouped3d
from pdelab.problem3d import make_problem3d
from pdelab.solvers3d import solve3d
from pdelab.stationary_study3d import study_stationary3d


@pytest.mark.parametrize("mixed", [False, True])
def test_grouped_public_failure_retains_costs_and_no_estimate(mixed):
    problem = make_boundary_problem3d(n=8) if mixed else make_problem3d(n=8)
    solver = solve_boundary3d if mixed else solve3d
    with pytest.raises(GroupedWalkFailure) as caught:
        solver(problem, [[4, 4, 4]], "mc_grouped", walks=19, max_steps=1, batch_size=7)
    failure = caught.value
    assert failure.seconds >= failure.diagnostics["preparation_seconds"] >= 0
    assert failure.diagnostics["walks"] == 0
    assert failure.diagnostics["started_walks"] == 7
    assert failure.diagnostics["transitions"] == 7
    assert not hasattr(failure, "values")


@pytest.mark.parametrize("batch", [0, -1, 65537, 1.5, True, np.nan, "128"])
def test_invalid_batch_refused(batch):
    with pytest.raises(ValueError, match="batch_size"):
        solve_grouped3d(make_problem3d(n=4), [[2, 2, 2]], batch_size=batch)


def test_group_memory_refusal_precedes_preparation_and_kernel(monkeypatch):
    import pdelab.grouped_solver3d as module

    def forbidden(*args, **kwargs):
        pytest.fail("A refused solve must not allocate an operator or start paths")

    problem = make_boundary_problem3d(n=4)
    monkeypatch.setattr(module, "BoundaryOperator3D", forbidden)
    monkeypatch.setattr(module, "grouped_walks3d", forbidden)
    with pytest.raises(ValueError, match="лимит памяти"):
        solve_grouped3d(problem, [[2, 2, 2]], max_workspace_bytes=1)


def test_nonabsorbing_all_second_kind_refused():
    problem = make_boundary_problem3d(n=4, kinds=dict.fromkeys(FACES, "neumann"))
    with pytest.raises(ValueError, match="нет поглощения"):
        solve_grouped3d(problem, [[2, 2, 2]])


def test_stationary_failure_series_keeps_actual_cost_and_no_percent_error():
    problem = make_problem3d(n=8)
    study = study_stationary3d(
        problem,
        [[4, 4, 4]],
        [1.0],
        ["mc_grouped"],
        walk_budgets=(19,),
        repeats=2,
        max_steps=1,
        batch_size=7,
    )
    assert len(study["trials"]) == 2
    assert all(
        t["status"] == "path_limit" and t["values"] is None for t in study["trials"]
    )
    assert all(
        t["started_walks"] == 7 and t["transitions"] == 7 for t in study["trials"]
    )
    assert all(
        r["seconds"] is None and r["relative_error_pct"] is None for r in study["rows"]
    )
    assert all(
        r["transitions"] == 14 and r["completed_repeats"] == 0 for r in study["rows"]
    )


def test_mixed_no_retuning_after_final_failure_and_fresh_seeds():
    problem = make_boundary_problem3d(n=4)
    points = np.array([[2, 2, 2]])
    target = problem.exact[tuple(points.T)]
    calls, seeds = [], []

    def solver(problem, points, method, **kwargs):
        result = solve_boundary3d(problem, points, "mc_grouped", walks=2)
        calls.append(kwargs["walks"])
        seeds.append(kwargs["seed"])
        values = result.values.copy()
        values[tuple(points.T)] = target * (1 if len(calls) <= 2 else 1.2)
        return replace(result, values=values)

    study = study_stationary3d(
        problem,
        points,
        target,
        ["mc_grouped"],
        walk_budgets=(64, 256),
        repeats=2,
        solver=solver,
    )
    assert calls == [64] * 4
    assert len(set(seeds)) == 4
    assert all(
        r["status_code"] == "validation_failed" and r["seconds"] is None
        for r in study["rows"]
    )
    assert [t["phase"] for t in study["trials"]] == ["pilot"] * 2 + ["final"] * 2
