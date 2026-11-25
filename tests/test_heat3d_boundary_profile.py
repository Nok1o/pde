"""Check the analysis-only visit formula and that failed trials remain visible."""

import numpy as np
import pytest

from benchmarks import heat3d_boundary_profile as profile
from pdelab.heat3d_operator import HeatOperator3D


def test_expected_visits_of_two_state_chain_with_duplicate_and_known_queries():
    shape = (3, 3, 3)
    face, center = (0, 1, 1), (1, 1, 1)
    fixed = np.ones(shape, dtype=bool)
    fixed[face] = fixed[center] = False
    weights = np.zeros((*shape, 6))
    weights[face + (0,)] = 0.25
    weights[center + (1,)] = 0.5
    zero = np.zeros(shape)
    operator = HeatOperator3D(
        np.ones(shape), weights, zero, fixed, zero, np.ones(shape)
    )
    points = np.array([face, center, center, (0, 0, 0)])
    result = profile.expected_visits(operator, points, 8)
    # v_face = 8 + v_center/2; v_center = 16 + v_face/4.
    assert result["paths"] == 24
    assert result["expected_face_visits"] == pytest.approx(128 / 7)
    assert result["expected_interior_visits"] == pytest.approx(144 / 7)
    assert result["expected_transitions"] == pytest.approx(104 / 7)
    assert result["expected_edge_visits"] == result["expected_corner_visits"] == 0
    known_only = profile.expected_visits(operator, points[-1:], 8)
    assert known_only["paths"] == known_only["expected_transitions"] == 0
    assert known_only["expected_boundary_visit_fraction"] == 0


@pytest.mark.parametrize("grouped", [0, 1])
def test_profile_records_real_runs_and_keeps_failed_series(monkeypatch, grouped):
    monkeypatch.setitem(profile.PLAN, "repeats", 2)
    monkeypatch.setitem(profile.PLAN, "walks", 8)
    snapshot = profile.snapshot
    monkeypatch.setattr(profile, "snapshot", lambda faces, dt: snapshot(faces, dt, n=4))
    packet = profile.measure(2, 0, grouped)
    summary = profile.summarize(packet)
    assert summary["completed"] == 2
    assert summary["relative_l2_error_pct_mean"] >= 0
    assert summary["expected_face_visits"] > 0
    assert summary["expected_edge_visits"] > 0
    assert summary["expected_corner_visits"] > 0
    for row in packet["trials"]:
        assert row["walks"] == row["completed_walks"] == packet["expected"]["paths"]
        assert row["full_seconds"] > row["coefficients_seconds"] >= 0
        assert row["workspace_bytes"] > 0
    # A real budget failure must keep its cost and suppress a success-only mean.
    monkeypatch.setitem(profile.PLAN, "max_walks", 1)
    failed = profile.measure(2, 0, grouped)
    summary = profile.summarize(failed)
    assert summary["completed"] == 0
    assert summary["relative_l2_error_pct_mean"] is None
    assert summary["full_seconds_mean"] > 0
    assert all(row["walks"] == 1 for row in failed["trials"])
    assert all(row["status"] == "walk_budget" for row in failed["trials"])
