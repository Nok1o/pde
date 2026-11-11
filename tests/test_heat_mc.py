import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from pdelab import (
    HeatProblem,
    heat_mc_step,
    make_heat_problem,
    solve_heat,
    solve_heat_mc,
)
from pdelab.cli import main
from pdelab.heat_mc_study import heat_mc_study
from pdelab.heat_mc_views import heat_mc_archive
from pdelab.operator import FrozenOperator


def test_one_step_uses_common_weights_source_and_new_boundary_time():
    p = make_heat_problem("manufactured", n=8, a2=0.3)
    dt, t = 0.037, 0.14
    previous = p.initial
    source = p.evaluate(p.source, t)
    boundary = p.evaluate(p.boundary, t)
    op = FrozenOperator.from_nodes(
        np.full_like(previous, p.a2 * dt), dt * source, boundary, sigma=1
    )
    probabilities, reward, temporal = op.mc_coefficients()
    np.testing.assert_allclose(probabilities.sum(axis=-1) + temporal, 1)
    full = FrozenOperator(op.wx, op.wy, op.sigma, previous + dt * source, boundary)
    reference = full.solve()
    neighbors = np.stack(
        (
            reference[2:, 1:-1],
            reference[:-2, 1:-1],
            reference[1:-1, 2:],
            reference[1:-1, :-2],
        ),
        axis=-1,
    )
    np.testing.assert_allclose(
        reward
        + (probabilities * neighbors).sum(axis=-1)
        + temporal * previous[1:-1, 1:-1],
        reference[1:-1, 1:-1],
    )
    points = [[4, 4], [2, 5]]
    result = heat_mc_step(p, previous, dt, t, points, walks=30000, seed=107)
    for i, (y, x) in enumerate(points):
        assert abs(result.values[i] - reference[y, x]) <= 5 * result.stderr[i]
    assert result.diagnostic_step and result.times.tolist() == [t - dt, t]


@pytest.mark.parametrize("preset", ["decay", "manufactured"])
@pytest.mark.parametrize("n", [8, 16])
def test_complete_space_time_paths_match_direct_and_analytic(preset, n):
    p = make_heat_problem(preset, n=n)
    result = solve_heat_mc(p, dt=0.013, T=0.04, walks=30000, seed=731)
    direct = solve_heat(p, dt=0.013, T=0.04, method="direct")
    y = x = n // 2
    reference = direct.fields[-1, y, x]
    assert abs(result.values[0] - reference) < 5 * result.stderr[0]
    exact = p.evaluate(p.exact, 0.04)[y, x]
    assert abs(result.values[0] - exact) < 5 * result.stderr[0] + abs(reference - exact)
    np.testing.assert_array_equal(result.times, direct.times)
    assert result.temporal_transitions > 0 and result.spatial_transitions > 0
    assert not result.diagnostic_step
    assert result.seconds == pytest.approx(
        result.preparation_seconds + result.path_seconds
    )
    assert result.memory_bytes > 0 and result.walks == 30000


def test_constant_data_is_exact_and_no_deterministic_solver_is_called(monkeypatch):
    import pdelab.heat

    def forbidden(*args, **kwargs):
        raise AssertionError("MC must not call the deterministic heat solver")

    monkeypatch.setattr(pdelab.heat, "solve_heat", forbidden)
    p = HeatProblem.from_functions(
        8, lambda x, y: 7, lambda x, y, t: 0, lambda x, y, t: 7
    )
    p.exact = forbidden
    result = solve_heat_mc(p, dt=0.01, T=0.1, walks=128)
    np.testing.assert_array_equal(result.values, 7)
    np.testing.assert_array_equal(result.stderr, 0)
    initial = solve_heat_mc(p, T=0, walks=8, max_transitions=1)
    np.testing.assert_array_equal(initial.values, 7)
    assert initial.spatial_transitions == initial.temporal_transitions == 0


def test_seed_reproducibility_and_complete_path_standard_error():
    p = make_heat_problem(n=8)
    a = solve_heat_mc(p, T=0.04, walks=512, seed=42)
    b = solve_heat_mc(p, T=0.04, walks=512, seed=42)
    np.testing.assert_array_equal(a.values, b.values)
    np.testing.assert_array_equal(a.stderr, b.stderr)
    assert a.spatial_transitions == b.spatial_transitions
    results = [solve_heat_mc(p, T=0.04, walks=1024, seed=seed) for seed in range(16)]
    observed = np.std([r.values[0] for r in results], ddof=1)
    predicted = np.sqrt(np.mean([r.stderr[0] ** 2 for r in results]))
    assert 0.5 < observed / predicted < 1.6


def test_transition_limit_never_returns_truncated_paths():
    with pytest.raises(RuntimeError, match="переходов"):
        solve_heat_mc(
            make_heat_problem(n=16), dt=0.01, T=0.1, walks=64, max_transitions=1
        )
    with pytest.raises(RuntimeError, match="переходов"):
        heat_mc_step(
            make_heat_problem(n=16),
            np.zeros((17, 17)),
            0.01,
            0.01,
            walks=64,
            max_transitions=1,
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(walks=1),
        dict(walks=2.5),
        dict(seed=-1),
        dict(seed=2**32),
        dict(max_transitions=0),
        dict(points=[[-1, 0]]),
        dict(dt=0),
        dict(T=-1),
    ],
)
def test_invalid_mc_settings(kwargs):
    with pytest.raises(ValueError):
        solve_heat_mc(make_heat_problem(n=8), **kwargs)


def test_bad_previous_and_nonfinite_data_are_rejected():
    p = make_heat_problem(n=8)
    with pytest.raises(ValueError):
        heat_mc_step(p, np.zeros((4, 4)), 0.01, 0.01)
    with pytest.raises(ValueError):
        heat_mc_step(p, p.initial, 0.1, 0.01)
    p.source = lambda x, y, t: np.nan
    with pytest.raises(ValueError):
        solve_heat_mc(p, T=0.01)


def test_research_uses_full_solves_and_independent_validation():
    p = make_heat_problem(n=8)
    study = heat_mc_study(
        p,
        dt=0.01,
        T=0.03,
        point=(3, 4),
        repeats=2,
        max_walks=256,
        seconds_per_method=10,
    )
    assert len(study["rows"]) == 6
    assert study["settings"]["T"] == 0.03 and study["settings"]["point_yx"] == [3, 4]
    for row in study["rows"]:
        if row["status"] == "reached":
            assert row["calibration_error_pct"] <= row["calibration_target_pct"]
            if row["calibration_margin_relaxed"]:
                assert row["budget"] == 256
                assert row["calibration_target_pct"] == row["target_pct"]
            assert row["validation_error_pct"] <= row["target_pct"]
            times = [
                r["seconds"]
                for r in study["trials"]
                if r["method"] == row["method"]
                and r["budget"] == row["budget"]
                and r["phase"] == "validation"
            ]
            assert row["seconds"] == pytest.approx(np.mean(times))
        else:
            assert row["seconds"] is None
    mc = [r for r in study["trials"] if r["method"] == "mc"]
    assert mc and all(r["transitions"] > 0 and r["preparation_seconds"] > 0 for r in mc)
    calibration = {r["seed"] for r in mc if r["phase"] == "calibration"}
    validation = {r["seed"] for r in mc if r["phase"] == "validation"}
    assert calibration.isdisjoint(validation)


def test_export_and_cli_include_costs_and_physical_times(tmp_path):
    p = make_heat_problem(n=8)
    r = solve_heat_mc(p, dt=0.013, T=0.04, walks=64)
    with zipfile.ZipFile(io.BytesIO(heat_mc_archive(r, p, 0.013))) as z:
        metadata = json.loads(z.read("experiment.json"))
        assert metadata["mode"] == "standalone_space_time"
        assert metadata["T"] == 0.04 and metadata["walks"] == 64
        assert metadata["seconds"] == r.seconds
        assert metadata["temporal_transitions"] == r.temporal_transitions
        with np.load(io.BytesIO(z.read("queries.npz"))) as data:
            np.testing.assert_array_equal(data["t"], r.times)
    main(
        ["heat-mc", "--n", "8", "--T", ".025", "--walks", "64", "--out", str(tmp_path)]
    )
    assert (tmp_path / "heat-mc-results.zip").is_file()


def test_ui_mc_and_research_work_without_grid_calculation():
    app = AppTest.from_file(
        str(Path(__file__).parents[1] / "app.py"), default_timeout=60
    ).run()
    app.radio(key="evolution").set_value("Теплопроводность").run()
    app.select_slider(key="heat_n").set_value(8)
    app.number_input(key="heat_T").set_value(0.025)
    app.select_slider(key="heat_mc_walks").set_value(64)
    app.run()
    app.button(key="heat_mc_run").click().run()
    assert not app.exception
    assert app.session_state["heat_mc_query"][0].times[-1] == 0.025
    assert "heat_result" not in app.session_state
    app.number_input(key="heat_study_repeats").set_value(2)
    app.select_slider(key="heat_study_walks").set_value(64)
    app.button(key="heat_study_run").click().run()
    assert not app.exception
    assert len(app.session_state["heat_mc_study"][0]["rows"]) == 6
    app.session_state["heat_mc_study"][0]["settings"].pop("validation_policy")
    app.run()
    assert "heat_mc_study" not in app.session_state
    assert any("Протокол проверки" in item.value for item in app.info)


def test_research_reports_transition_failure_without_publishing_times():
    study = heat_mc_study(
        make_heat_problem(n=8), T=0.04, repeats=2, max_walks=64, max_transitions=1
    )
    rows = [row for row in study["rows"] if row["method"] == "mc"]
    assert len(rows) == 3
    assert all(
        row["status"].startswith("solver_failed") and row["seconds"] is None
        for row in rows
    )


def test_failed_validation_retries_larger_budget(monkeypatch):
    from types import SimpleNamespace

    import pdelab.heat_mc_study as module

    p = HeatProblem.from_functions(
        8,
        lambda x, y: 1,
        lambda x, y, t: 0,
        lambda x, y, t: 1,
        exact=lambda x, y, t: 1,
    )
    budgets = []

    def estimate(*args, walks, **kwargs):
        budgets.append(walks)
        # Three pilot batches pass; validation fails at 64 and passes at 128.
        value = (1.005 if len(budgets) <= 6 else 1.12) if walks == 64 else 1.0
        return SimpleNamespace(
            values=np.array([value]),
            seconds=0.01,
            preparation_seconds=0.002,
            spatial_transitions=10,
            temporal_transitions=4,
            memory_bytes=100,
        )

    monkeypatch.setattr(module, "solve_heat_mc", estimate)
    study = module.heat_mc_study(p, dt=0.01, T=0.02, repeats=2, max_walks=256)
    rows = [row for row in study["rows"] if row["method"] == "mc"]
    assert budgets == [64] * 8 + [128] * 8
    assert len(rows) == 3
    assert all(row["status"] == "reached" and row["seconds"] == 0.01 for row in rows)
    assert all(row["budget"] == 128 for row in rows)
    assert all(row["validation_attempts"] == 2 for row in rows)
    assert all(row["calibration_error_pct"] == 0 for row in rows)
    assert all(row["validation_error_pct"] == 0 for row in rows)
