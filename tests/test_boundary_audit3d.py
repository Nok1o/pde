"""Independent stationary checks and holdout/report regressions for PDE-27."""

import json
from dataclasses import asdict

import numpy as np
import pytest

from pdelab import boundary_audit3d as audit
from pdelab.boundary3d import BoundaryOperator3D, face_indices
from pdelab.measurement_protocol import Trial, summarize


@pytest.mark.parametrize("variant", audit.VARIANTS)
@pytest.mark.parametrize("kind", ["sine", "polynomial"])
def test_stationary_snapshot_source_and_outward_data_independently(kind, variant):
    problem, exact = audit.stationary_snapshot(kind, 4, variant)
    z, y, x = np.meshgrid(problem.axis, problem.axis, problem.axis, indexing="ij")
    if kind == "sine":
        s = np.sin(np.pi * x) * np.sin(np.pi * y) * np.sin(np.pi * z)
        target = 1 + s
        expected_f = -3 * np.pi**2 * s
        gradients = [
            np.pi * np.cos(np.pi * x) * np.sin(np.pi * y) * np.sin(np.pi * z),
            np.pi * np.sin(np.pi * x) * np.cos(np.pi * y) * np.sin(np.pi * z),
            np.pi * np.sin(np.pi * x) * np.sin(np.pi * y) * np.cos(np.pi * z),
        ]
    else:
        target = (
            1 + (x + 2 * y + 3 * z + x * y + x * z + y * z + x * x + y * y + z * z) / 4
        )
        expected_f = np.full_like(x, 1.5)
        gradients = [
            (1 + 2 * x + y + z) / 4,
            (2 + x + 2 * y + z) / 4,
            (3 + x + y + 2 * z) / 4,
        ]
    np.testing.assert_allclose(exact, target, atol=1e-14)
    np.testing.assert_allclose(problem.f, expected_f, atol=1e-14)
    assert problem.exact is None  # working backends cannot read the reference
    for face, index in face_indices().items():
        condition = problem.faces[face]
        expected = target[index]
        if condition.kind == "robin":
            expected = (
                expected
                + (-1 if face[1] == "0" else 1) * gradients["xyz".index(face[0])][index]
            )
        np.testing.assert_allclose(condition.value, expected, atol=1e-14)


@pytest.mark.parametrize("variant", audit.VARIANTS)
def test_cross_polynomial_exact_discrete_equations_and_sparse_reference(variant):
    problem, exact = audit.stationary_snapshot("polynomial", 4, variant)
    operator = BoundaryOperator3D(problem)
    np.testing.assert_allclose(operator.apply(exact), operator.rhs(), atol=1e-12)
    np.testing.assert_allclose(operator.solve(), exact, atol=1e-12)


def test_volume_balance_and_markov_identity_on_arbitrary_field():
    problem, _ = audit.stationary_snapshot("polynomial", 4, "all_robin")
    operator = BoundaryOperator3D(problem)
    values = np.random.default_rng(52).normal(size=problem.f.shape)
    # Internal face fluxes cancel, including boundary/edge/corner volumes.
    assert np.sum(operator.volumes * operator.apply(values)) == pytest.approx(
        np.sum(operator.volumes * operator.reaction * values), abs=1e-13
    )
    probabilities, reward, killing = operator.mc_coefficients()
    np.testing.assert_allclose(probabilities.sum(axis=-1) + killing, 1)
    assert np.min(probabilities) >= 0 and np.min(killing) >= 0
    assert killing[2, 2, 2] == 0
    assert killing[0, 0, 0] == pytest.approx(24 / (96 + 24))
    expected = values - (operator.apply(values) - operator.rhs()) / operator.diagonal
    observed = reward.copy()
    for position in np.ndindex(values.shape):
        for direction in range(6):
            target = audit_step(position, 4, direction)
            observed[position] += (1 - killing[position]) / 6 * values[target]
    np.testing.assert_allclose(observed, expected, atol=1e-14)


@pytest.mark.parametrize("variant", audit.VARIANTS)
def test_sine_snapshot_discretization_converges_at_second_order(variant):
    errors = []
    for n in (8, 16):
        problem, exact = audit.stationary_snapshot("sine", n, variant)
        values = BoundaryOperator3D(problem).solve()
        errors.append(float(np.max(np.abs(values - exact))))
    assert 3.5 < errors[0] / errors[1] < 4.5


def audit_step(position, n, direction):
    axis, sign = direction // 2, 1 if direction % 2 == 0 else -1
    target = list(position)
    if target[axis] + sign < 0 or target[axis] + sign > n:
        sign = -sign
    target[axis] += sign
    return tuple(target)


def test_real_small_series_returns_full_cost_and_no_fabricated_rss():
    problem, exact = audit.stationary_snapshot("polynomial", 4, "y0_robin")
    points = np.array([[2, 2, 2]])
    result = audit.measure_series(
        "polynomial",
        4,
        "y0_robin",
        "lod_serial",
        1e-8,
        points,
        exact[tuple(points.T)],
        [51, 52],
        1,
        "control",
        60,
    )
    assert result["summary"]["status"] == "confirmed"
    assert result["summary"]["completed_repeats"] == 2
    assert result["summary"]["peak_rss_bytes"] is None
    assert all(t["preparation_seconds"] <= t["seconds"] for t in result["trials"])


def test_exception_retains_time_and_marks_unknown_counters(monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("path limit")

    monkeypatch.setattr(audit, "solve_boundary3d", fail)
    result = audit.measure_series(
        "polynomial",
        4,
        "all_robin",
        "mc",
        64,
        np.array([[2, 2, 2]]),
        [3],
        [1, 2],
        1,
        "control",
        60,
    )
    assert result["summary"]["status"] == "incomplete"
    assert result["summary"]["walks"] is None
    assert result["failures"][0]["reason"] == "path limit"
    assert result["summary"]["spent_seconds"] > 0


def test_control_failure_does_not_restart_pilot_and_selection_is_written_first(
    tmp_path, monkeypatch
):
    calls = []
    monkeypatch.setattr(audit, "warmup_boundary3d", lambda: None)
    # Keep source archiving real, but avoid external git process in this unit check.
    monkeypatch.setattr(
        audit,
        "provenance",
        lambda: dict(commit="synthetic", dirty=True, source_sha256={}, environment={}),
    )

    def fake(
        kind,
        n,
        variant,
        method,
        budget,
        points,
        reference,
        seeds,
        target,
        phase,
        remaining,
    ):
        calls.append((variant, method, phase))
        if phase == "control":
            choices = json.loads((tmp_path / "selection.json").read_text())
            assert choices[-1]["status"] == "frozen_before_control"
        trials = [
            Trial(
                r,
                seed,
                "completed",
                "",
                (reference * (1.004 if phase == "pilot" else 1.2)).tolist(),
                0.01,
                0.001,
                1024,
                None,
                int(budget) if method == "mc" else None,
                64 if method == "mc" else 0,
                100,
                0,
                0,
                5,
                None,
            )
            for r, seed in enumerate(seeds)
        ]
        return dict(
            summary=summarize(
                trials,
                reference,
                requested_repeats=len(seeds),
                target_pct=target,
                phase=phase,
                stochastic=method == "mc",
            ),
            trials=[asdict(t) for t in trials],
            failures=[],
            budget=budget,
            elapsed_seconds=0.1,
        )

    monkeypatch.setattr(audit, "measure_series", fake)
    rows = audit.run(tmp_path, n=4, repeats=2, kinds=("polynomial",))
    assert len(calls) == 12  # one pilot + one control for every method/variant
    assert all(
        calls[i][2:] == ("pilot",) and calls[i + 1][2:] == ("control",)
        for i in range(0, 12, 2)
    )
    assert all(r["status"] == "not_reached" for r in rows)
    assert all(
        r["lod_over_mc"] is None
        for r in json.loads((tmp_path / "results.json").read_text())["ratios"]
    )
    with pytest.raises(ValueError, match="empty"):
        audit.run(tmp_path, n=4, repeats=2, kinds=("polynomial",))
