from types import SimpleNamespace

import numpy as np
import pytest

from pdelab.study_protocol import run_accuracy_study


def observation(value=1.0):
    return SimpleNamespace(
        values=np.array([value]),
        status="completed",
        reason="",
        seconds=0.01,
        preparation_seconds=0.0,
        workspace_bytes=20,
        peak_rss_bytes=None,
        walks=64,
        transitions=100,
    )


@pytest.mark.parametrize("point", [[[1.9, 2, 3]], [[5, 2, 3]], [[1, 2, 3], [1, 2, 3]]])
def test_stationary_adapter_rejects_invalid_points_before_any_solve(point):
    from pdelab.problem3d import make_problem3d
    from pdelab.stationary_study3d import study_stationary3d

    def forbidden(*args, **kwargs):
        pytest.fail("Invalid query must fail before running or confirming a solve")

    with pytest.raises(ValueError):
        study_stationary3d(
            make_problem3d("affine", 4),
            point,
            [1.0],
            ["lod_serial"],
            tolerances=[1e-8],
            repeats=2,
            solver=forbidden,
        )


def test_stationary_observations_have_no_fictitious_time_and_retain_values():
    study = run_accuracy_study(
        lambda *args: observation(), [64], [1.0], repeats=2, retain_values=True
    )
    assert all(
        t["target_time"] is None and t["completed_until"] is None
        for t in study["trials"]
    )
    assert all(t["values"] == [1.0] for t in study["trials"])
    assert all(r["status"] == "confirmed" for r in study["rows"])


def test_phase_budget_retains_partial_series_and_gives_final_its_own_deadline():
    time = [0.0]
    calls = []

    def runner(budget, seed):
        calls.append(budget)
        time[0] += 0.4
        return observation(1.06)

    study = run_accuracy_study(
        runner,
        [64, 256, 1024],
        [1.0],
        repeats=2,
        seconds_per_phase=1.0,
        clock=lambda: time[0],
        retain_values=True,
    )
    # First pilot completes; second runs once then hits the shared pilot deadline.
    # The already chosen 10% target is still checked in a fresh final allowance.
    assert calls == [64, 64, 256, 64, 64]
    partial = next(s for s in study["series"] if s["candidate"] == 1)
    assert partial["recorded_repeats"] == partial["completed_repeats"] == 1
    assert not partial["complete"] and partial["relative_error_pct"] is None
    assert (
        next(r for r in study["rows"] if r["target_pct"] == 10)["status"] == "confirmed"
    )


def test_deadline_before_first_run_is_an_explicit_empty_incomplete_series():
    clock = iter(range(10))

    def forbidden(*args):
        pytest.fail("Deadline precedes the first solve")

    study = run_accuracy_study(
        forbidden,
        [64],
        [1.0],
        repeats=2,
        seconds_per_phase=0.1,
        clock=lambda: next(clock),
    )
    assert not study["trials"]
    assert all(
        r["status"] == "pilot_incomplete" and r["recorded_repeats"] == 0
        for r in study["rows"]
    )
