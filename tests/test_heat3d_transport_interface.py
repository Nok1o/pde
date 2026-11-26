"""Count transport and deterministic weights share full-history evidence."""

import csv
import io
import json
import shlex
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from pdelab.heat3d_cli import main
from pdelab.heat3d_experiment import (
    HeatConfig3D,
    compare_heat3d,
    csv_bytes,
    json_bytes,
    study_heat3d,
)
from pdelab.heat3d_problem import HeatFace3D, make_heat_problem3d
from pdelab.heat3d_solver import HeatSettings3D, solve_heat3d

COUNTERS = ("node_updates", "transport_sweeps", "edge_transfers")
METHODS = ("lod", "mc", "mc_counts", "mass_transport")
SETTINGS = HeatSettings3D(
    walks=1024,
    mc_tolerance=0.2,
    mc_change_tolerance=0.1,
    max_outer=20,
    max_seconds=20,
)


@pytest.mark.parametrize("kinds", [(1,) * 6, (3,) * 6])
def test_transport_reaches_physical_time_and_exports_actual_work(kinds):
    config = HeatConfig3D(
        case="polynomial",
        n=4,
        dt=0.001,
        T=0.002,
        kinds=kinds,
        capacity_gamma=0.4,
        reaction_sigma=0.25,
        methods=METHODS,
        newton=True,
    )
    result = compare_heat3d(config, SETTINGS)
    assert result["grid_diagnostic"]["status"] == "completed"
    records = {row["method"]: row for row in result["rows"]}
    assert all(row["status"] == "completed" for row in records.values())
    np.testing.assert_allclose(
        result["arrays"]["mass_transport"], result["arrays"]["newton"], rtol=1e-7
    )
    for method in ("mc_counts", "mass_transport"):
        row = records[method]
        assert row["physical_steps"] == 2 and row["completed_until"] == 0.002
        assert 0 < row["preparation_seconds"] <= row["seconds"]
        history = [r for r in result["history"] if r["method"] == method]
        for key in COUNTERS:
            assert row[key] > 0
            assert row[key] == sum(r[key] for r in history)
    counts = records["mc_counts"]
    assert counts["walks_applicable"] and counts["validation_walks"] > 0
    assert counts["walks"] == counts["completed_walks"]
    assert counts["edge_transfers"] <= counts["transitions"]
    mass = records["mass_transport"]
    assert not mass["walks_applicable"]
    assert mass["walks"] == mass["transitions"] == mass["validation_walks"] == 0
    assert "walks_per_free_node" not in mass and "seed" not in mass
    assert mass["batch_size"] is None
    assert mass["tolerance"] == SETTINGS.tolerance
    assert records["mc_counts"]["tolerance"] == SETTINGS.mc_tolerance
    encoded = json.loads(json_bytes(result))
    csv_rows = list(
        csv.DictReader(io.StringIO(csv_bytes(result["rows"]).decode("utf-8-sig")))
    )
    for row, json_row, csv_row in zip(
        result["rows"], encoded["rows"], csv_rows, strict=True
    ):
        for key in COUNTERS:
            assert row[key] == json_row[key] == int(csv_row[key])


@pytest.mark.parametrize("method", ["mc_counts", "mass_transport"])
def test_compact_transport_history_keeps_all_work(method):
    problem = make_heat_problem3d("polynomial", 4, 0.5)
    settings = replace(SETTINGS, conductivity_refresh_layers=2)
    full = solve_heat3d(problem, dt=0.001, T=0.003, method=method, settings=settings)
    compact = solve_heat3d(
        problem,
        dt=0.001,
        T=0.003,
        method=method,
        settings=replace(settings, history_mode="layers"),
    )
    assert full.status == compact.status == "completed"
    np.testing.assert_array_equal(full.values, compact.values)
    for key in (*COUNTERS, "walks", "transitions", "outer_iterations"):
        assert getattr(full, key) == getattr(compact, key)
    for key in COUNTERS:
        assert getattr(compact, key) == sum(row[key] for row in compact.history)


def test_study_keeps_independent_counts_and_deterministic_mass_evidence():
    config = HeatConfig3D(case="polynomial", n=4, dt=0.001, T=0.002, methods=METHODS)
    result = study_heat3d(
        config, SETTINGS, repeats=2, seed=91, budgets=[1e-3], walk_budgets=[1024]
    )
    assert result["budgets_by_method"] == {
        "lod": [1e-3],
        "mc": [1024],
        "mc_counts": [1024],
        "mass_transport": [1e-3],
    }
    for method in METHODS:
        trials = [r for r in result["trials"] if r["method"] == method]
        assert {r["phase"] for r in trials} == {"pilot", "final"}
        assert all(r["values"] is not None for r in trials)
        if method == "mass_transport":
            assert all(r["seed"] is None and r["walks"] == 0 for r in trials)
        if method == "mc_counts":
            assert len({r["seed"] for r in trials}) == len(trials)
            assert all(r["validation_walks"] > 0 for r in trials)
    for collection in ("rows", "series", "trials"):
        for row in result[collection]:
            if row["method"] in ("mass_transport", "mc_counts"):
                assert all(row[key] > 0 for key in COUNTERS)


@pytest.mark.parametrize("method", ["mc_counts", "mass_transport"])
def test_zero_reference_is_not_a_confirmed_relative_target(method, monkeypatch):
    def zero(x, y, z, t):
        return 0.0

    original = make_heat_problem3d(n=4)
    problem = replace(
        original,
        initial=zero,
        source=zero,
        exact=zero,
        faces={face: HeatFace3D(1, zero) for face in original.faces},
    )
    monkeypatch.setattr(
        "pdelab.heat3d_experiment.make_heat_problem3d", lambda *args, **kwargs: problem
    )
    result = study_heat3d(
        HeatConfig3D(n=4, dt=0.001, T=0.002, methods=(method,)),
        SETTINGS,
        repeats=2,
        budgets=[1e-3],
        walk_budgets=[32],
    )
    assert all(
        row["status"] == "relative_target_not_applicable" for row in result["rows"]
    )
    assert all(row["absolute_error"] == 0 for row in result["rows"])
    assert all(row["relative_error_pct"] is None for row in result["rows"])


@pytest.mark.parametrize(
    "method,limits,status",
    [
        ("mc_counts", {"max_walks": 1}, "walk_budget"),
        ("mass_transport", {"max_inner": 1}, "linear_limit"),
        ("mass_transport", {"max_workspace_bytes": 1}, "memory_limit"),
    ],
)
def test_limits_never_return_an_incomplete_transport_field_as_a_solution(
    method, limits, status
):
    problem = make_heat_problem3d("polynomial", 4, 0.5)
    result = solve_heat3d(
        problem,
        dt=0.001,
        T=0.002,
        method=method,
        settings=replace(SETTINGS, **limits),
    )
    assert result.status == status
    assert result.values is None and result.completed_until == 0
    if method == "mass_transport":
        assert result.walks == result.transitions == 0


def test_cli_reproduces_new_methods_without_inventing_a_mass_sample_size(tmp_path):
    args = [
        "run",
        "--case",
        "polynomial",
        "--n",
        "4",
        "--dt",
        ".001",
        "--T",
        ".002",
        "--methods",
        "mc_counts",
        "mass_transport",
        "--walks",
        "1024",
        "--out",
        str(tmp_path / "first"),
    ]
    assert main(args) == 0
    first = json.loads((tmp_path / "first/result.json").read_bytes())
    replay = shlex.split(first["metadata"]["reproduction"])[3:]
    replay[replay.index("--out") + 1] = str(tmp_path / "replay")
    assert main(replay) == 0
    second = json.loads((tmp_path / "replay/result.json").read_bytes())
    assert first["metadata"]["solver_settings"] == second["metadata"]["solver_settings"]
    for a, b in zip(first["rows"], second["rows"], strict=True):
        for key in (*COUNTERS, "relative_error_pct", "walks", "transitions", "status"):
            assert a[key] == b[key]
    mass = first["rows"][1]
    assert "walks_per_free_node" not in mass and "seed" not in mass


def test_ui_new_methods_run_study_and_preserve_saved_results():
    app = AppTest.from_file(
        Path(__file__).parents[1] / "app.py", default_timeout=60
    ).run()
    app.radio(key="evolution").set_value("Квазилинейная теплопроводность").run()
    app.radio(key="nonlinear_dimension").set_value("3D").run()
    app.multiselect(key="h3_methods").set_value(["mass_transport"]).run()
    assert not any(w.key == "h3_walks" for w in app.number_input)
    app.multiselect(key="h3_methods").set_value(["mc_counts", "mass_transport"]).run()
    assert not any(w.key == "h3_batch_size" for w in app.number_input)
    app.select_slider(key="h3_n").set_value(4)
    app.selectbox(key="h3_case").set_value("polynomial")
    app.number_input(key="h3_dt").set_value(0.001)
    app.number_input(key="h3_T").set_value(0.002)
    app.number_input(key="h3_walks").set_value(1024)
    app.button(key="h3_run").click().run()
    assert not app.error and not app.exception
    saved = app.session_state["h3_comparison"]
    assert all(row["status"] == "completed" for row in saved["rows"])
    app.number_input(key="h3_repeats").set_value(2)
    app.multiselect(key="h3_budgets").set_value([1e-3])
    app.multiselect(key="h3_walk_budgets").set_value([1024])
    app.button(key="h3_study_run").click().run()
    assert not app.error and not app.exception
    study = app.session_state["h3_study"]
    assert {row["method"] for row in study["rows"]} == {"mc_counts", "mass_transport"}
    app.number_input(key="h3_walks").set_value(32).run()
    assert app.session_state["h3_comparison"]["history"] == saved["history"]
    assert app.session_state["h3_study"]["metadata"]["walk_budgets"] == [1024]
