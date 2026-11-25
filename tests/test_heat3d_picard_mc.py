from dataclasses import replace

import numpy as np
import pytest

from pdelab.boundary3d import FACES
from pdelab.heat3d_operator import freeze_heat3d
from pdelab.heat3d_problem import make_heat_problem3d
from pdelab.heat3d_solver import HeatSettings3D, solve_heat3d

SETTINGS = HeatSettings3D(
    tolerance=0.2, change_tolerance=0.12, walks=1024, max_outer=12
)
METHODS = ("mc", "mc_control")


def problem(n=5, alpha=0.5, faces=("x0",), case="polynomial"):
    return make_heat_problem3d(
        case, n, alpha, kinds={f: 3 if f in faces else 1 for f in FACES}
    )


@pytest.mark.parametrize("method", METHODS)
def test_advances_own_complete_field_and_validates_on_new_streams(method, monkeypatch):
    import pdelab.heat3d_control as control

    def no_reference(*args):
        pytest.fail("The full-history solver tried to read a reference")

    original = problem(faces=FACES)
    p = replace(original, exact=no_reference)
    line_solver = control.solve_lines3d

    def only_coarse(operator, initial, **kwargs):
        assert operator.n == SETTINGS.coarse_n < p.n
        return line_solver(operator, initial, **kwargs)

    monkeypatch.setattr(control, "solve_lines3d", only_coarse)
    previous_by_time = {}

    def freeze(p, prev, iterate, t, dt):
        previous_by_time.setdefault(t, prev.copy())
        np.testing.assert_array_equal(previous_by_time[t], prev)
        return freeze_heat3d(p, prev, iterate, t, dt)

    full = np.indices(p.shape).reshape(3, -1).T
    first_layer = solve_heat3d(
        p, full, method=method, dt=0.002, T=0.002, settings=SETTINGS
    )
    assert first_layer.status == "completed"
    result = solve_heat3d(
        p,
        [[0, 0, 0]],
        method=method,
        dt=0.002,
        T=0.003,
        settings=SETTINGS,
        operator_factory=freeze,
    )
    assert result.status == "completed" and result.physical_steps == 2
    np.testing.assert_array_equal(previous_by_time[0.002], p.initial_values())
    np.testing.assert_array_equal(
        previous_by_time[0.003], first_layer.values.reshape(p.shape)
    )
    repeated = solve_heat3d(
        p, full, method=method, dt=0.002, T=0.003, settings=SETTINGS
    )
    assert result.values[0] == repeated.values[0]
    assert result.walks == repeated.walks == result.completed_walks
    assert result.walks == sum(h["walks"] for h in result.history)
    assert result.transitions == sum(h["transitions"] for h in result.history)
    assert result.validation_walks == sum(
        h.get("validation_walks", 0) for h in result.history
    )
    assert result.validation_walks >= 2 * p.initial_values().size * SETTINGS.walks
    assert result.seconds >= result.preparation_seconds >= result.coarse_seconds
    assert (result.coarse_iterations > 0) == (method == "mc_control")
    assert result.coarse_iterations == result.inner_iterations
    streams = [
        h[k]
        for h in result.history
        for k in ("update_seed", "validation_seed")
        if k in h
    ]
    assert len(streams) == len(set(streams))
    accepted = [h for h in result.history if h["accepted"]]
    assert [h["dt"] for h in accepted] == pytest.approx([0.002, 0.001])
    for h in accepted:
        assert (
            max(h["nonlinear_relative_residual"], h["validation_residual"])
            <= SETTINGS.tolerance
        )
        assert (
            max(h["relative_change"], h["validation_change"])
            <= SETTINGS.change_tolerance
        )


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize(
    "n,alpha,faces",
    [(5, 0, ()), (5, 0.5, ("x0",)), (6, 0.5, ("x0", "y0", "z0")), (6, 0.5, FACES)],
)
def test_ensembles_of_whole_histories_agree_with_newton(method, n, alpha, faces):
    p = problem(n, alpha, faces)
    points = [[n // 2] * 3, [2, 2, 0], [2, 0, 0], [0, 0, 0]]
    args = dict(dt=0.002, T=0.003)
    ref = solve_heat3d(
        p, points, method="newton", settings=HeatSettings3D(tolerance=1e-12), **args
    )
    assert ref.status == "completed"
    values = []
    for seed in (631, 83, 275, 1039, 719, 853):
        run = solve_heat3d(
            p, points, method=method, settings=replace(SETTINGS, seed=seed), **args
        )
        assert run.status == "completed" and run.completed_until == args["T"]
        values.append(run.values)
    se = np.std(values, axis=0, ddof=1) / np.sqrt(len(values))
    # This measures finite-budget discrepancy including nonlinear and stopping
    # bias; it is not a proof that an individual stopped history is unbiased.
    np.testing.assert_array_less(
        abs(np.mean(values, axis=0) - ref.values), 6 * se + 0.001
    )


def test_multiple_picard_updates_and_independent_rejection(monkeypatch):
    import pdelab.heat3d_picard_mc as mc

    p = problem(6, faces=FACES)
    strict = replace(SETTINGS, walks=256, tolerance=0.01, change_tolerance=0.001)
    r = solve_heat3d(p, method="mc_control", dt=0.006, T=0.01, settings=strict)
    assert r.status == "completed" and r.outer_iterations > r.physical_steps == 2
    original = mc.PicardSampler3D.draw

    def reject_check(self, *args, check=False):
        field = original(self, *args, check=check)
        return field + 10 if check else field

    monkeypatch.setattr(mc.PicardSampler3D, "draw", reject_check)
    r = solve_heat3d(
        p,
        method="mc_control",
        dt=0.002,
        T=0.003,
        settings=replace(SETTINGS, max_outer=2),
    )
    assert r.status == "outer_limit" and r.values is None and r.physical_steps == 0
    assert all(h["validation_change"] > SETTINGS.change_tolerance for h in r.history)
    assert r.validation_walks > 0


@pytest.mark.parametrize("method", METHODS)
def test_run_limits_include_validation_and_preserve_only_accepted_time(method):
    p = problem(faces=FACES)
    allowance = p.initial_values().size * SETTINGS.walks
    settings = replace(SETTINGS, max_walks=3 * allowance + 1)
    r = solve_heat3d(p, method=method, dt=0.002, T=0.003, settings=settings)
    assert r.status == "walk_budget" and r.values is None
    assert r.completed_until == 0.002 and r.physical_steps == 1
    assert r.walks == settings.max_walks and r.completed_walks == r.walks
    assert r.validation_walks == allowance + 1
    assert "validation" in r.reason and r.diagnostic_time == 0.003
    limited = solve_heat3d(
        p,
        method=method,
        dt=0.002,
        T=0.003,
        settings=replace(SETTINGS, max_transitions=1),
    )
    assert limited.status == "walk_budget" and limited.transitions == 1
    assert limited.walks == limited.completed_walks + 1
    assert limited.values is None and limited.physical_steps == 0
    path = solve_heat3d(
        p,
        method=method,
        dt=0.002,
        T=0.003,
        settings=replace(SETTINGS, max_path_steps=1),
    )
    assert path.status == "path_limit" and path.walks == path.completed_walks + 1
    assert path.values is None


def test_coarse_failure_workspace_and_t_zero():
    p = problem()
    run = solve_heat3d(
        p,
        method="mc_control",
        dt=0.002,
        T=0.003,
        settings=replace(SETTINGS, max_inner=1),
    )
    assert run.status == "linear_limit" and run.values is None and run.walks == 0
    assert run.coarse_iterations == 1 and run.coarse_seconds > 0
    for method in METHODS:
        run = solve_heat3d(
            p, method=method, T=0.01, settings=replace(SETTINGS, max_workspace_bytes=1)
        )
        assert run.status == "memory_limit" and run.walks == 0 and run.values is None
        initial = solve_heat3d(p, [[0, 0, 0]], method=method, T=0, settings=SETTINGS)
        assert (
            initial.status == "completed"
            and initial.values[0] == p.initial_values()[0, 0, 0]
        )
        assert initial.physical_steps == initial.walks == initial.coarse_iterations == 0
    bad = solve_heat3d(p, method="mc_control", settings=replace(SETTINGS, coarse_n=p.n))
    assert bad.status == "unsupported" and bad.values is None
