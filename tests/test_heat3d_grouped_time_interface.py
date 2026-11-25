"""Full-history grouped MC: UI, CLI, study budgets and portable results."""

import csv
import io
import json
import shlex
import zipfile
from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from pdelab.heat3d_cli import main
from pdelab.heat3d_picard_mc import CONTROL_MC_METHODS, GROUPED_MC_METHODS

METHODS = ("lod", "block", "mc", "mc_grouped", "mc_control", "mc_control_grouped")


@pytest.mark.parametrize("gamma,sigma", [(0.0, 0.0), (0.4, 0.25)])
def test_cli_all_six_methods_export_actual_group_settings_and_reproduce(
    tmp_path, gamma, sigma
):
    out = tmp_path / "run"
    assert (
        main(
            [
                "run",
                "--case",
                "polynomial",
                "--n",
                "5",
                "--dt",
                ".002",
                "--T",
                ".003",
                "--x0",
                "3",
                "--capacity-gamma",
                str(gamma),
                "--reaction-sigma",
                str(sigma),
                "--point",
                "0",
                "2",
                "2",
                "--methods",
                *METHODS,
                "--walks",
                "1024",
                "--batch-size",
                "7",
                "--coarse-n",
                "4",
                "--out",
                str(out),
            ]
        )
        == 0
    )
    saved = json.loads((out / "result.json").read_bytes())
    assert saved["metadata"]["solver_settings"]["batch_size"] == 7
    assert saved["metadata"]["configuration"]["capacity_gamma"] == gamma
    assert saved["metadata"]["configuration"]["reaction_sigma"] == sigma
    assert "c(u_new)" in saved["metadata"]["material_model"]["time_discretization"]
    with (out / "table.csv").open(encoding="utf-8-sig") as stream:
        csv_rows = list(csv.DictReader(stream))
    for row, csv_row in zip(saved["rows"], csv_rows, strict=True):
        method = row["method"]
        assert row["status"] == "completed" and row["physical_steps"] == 2
        assert row["completed_until"] == 0.003
        assert row["batch_size"] == (7 if method in GROUPED_MC_METHODS else None)
        assert csv_row["batch_size"] == ("7" if method in GROUPED_MC_METHODS else "")
        if row["walks_applicable"]:
            assert row["validation_walks"] > 0
            assert row["coarse_n"] == (4 if method in CONTROL_MC_METHODS else None)
    replay = shlex.split(saved["metadata"]["reproduction"])[3:]
    assert replay[replay.index("--batch-size") + 1] == "7"
    assert float(replay[replay.index("--capacity-gamma") + 1]) == gamma
    assert float(replay[replay.index("--reaction-sigma") + 1]) == sigma
    replay[replay.index("--out") + 1] = str(tmp_path / "replay")
    assert main(replay) == 0
    regenerated = json.loads((tmp_path / "replay/result.json").read_bytes())
    assert (
        saved["metadata"]["solver_settings"]
        == regenerated["metadata"]["solver_settings"]
    )
    for row, repeated in zip(saved["rows"], regenerated["rows"], strict=True):
        for key in (
            "method",
            "batch_size",
            "walks",
            "transitions",
            "relative_error_pct",
            "validation_walks",
        ):
            assert row[key] == repeated[key]


def test_grouped_study_keeps_per_method_settings_and_failed_attempts(tmp_path):
    with pytest.raises(SystemExit) as exc:
        main(
            [
                "study",
                "--case",
                "polynomial",
                "--n",
                "5",
                "--dt",
                ".002",
                "--T",
                ".003",
                "--methods",
                *METHODS,
                "--x0",
                "3",
                "--max-walks",
                "1",
                "--batch-size",
                "7",
                "--budgets",
                ".001",
                "--walk-budgets",
                "32",
                "--repeats",
                "2",
                "--out",
                str(tmp_path),
            ]
        )
    assert exc.value.code == 2
    study = json.loads((tmp_path / "result.json").read_bytes())
    assert study["budgets"] is None
    for method in METHODS:
        assert study["budgets_by_method"][method] == (
            [0.001] if method in ("lod", "block") else [32]
        )
    for collection in ("rows", "trials", "series"):
        for row in study[collection]:
            method = row["method"]
            assert row["batch_size"] == (7 if method in GROUPED_MC_METHODS else None)
            if method not in ("lod", "block"):
                assert row["coarse_n"] == (4 if method in CONTROL_MC_METHODS else None)
                assert row["budget_kind"] == "walks_per_free_node_per_solve"
    for trial in study["trials"]:
        if trial["method"] in GROUPED_MC_METHODS:
            assert trial["status"] == "walk_budget" and trial["values"] is None
            assert trial["walks"] == trial["completed_walks"] == 1
    for row in study["rows"]:
        if row["method"] in GROUPED_MC_METHODS:
            assert row["seconds"] is None and row["completed_repeats"] == 0
            assert row["total_walks"] == 2
    replay = shlex.split(study["metadata"]["reproduction"])[3:]
    replay[replay.index("--out") + 1] = str(tmp_path / "replay")
    with pytest.raises(SystemExit) as repeated:
        main(replay)
    assert repeated.value.code == 2
    regenerated = json.loads((tmp_path / "replay/result.json").read_bytes())
    assert (
        study["metadata"]["solver_settings"]
        == regenerated["metadata"]["solver_settings"]
    )
    for trial, repeated in zip(study["trials"], regenerated["trials"], strict=True):
        for key in ("method", "batch_size", "status", "seed", "walks", "transitions"):
            assert trial[key] == repeated[key]


def test_ui_groups_advance_time_study_and_preserve_saved_results(monkeypatch):
    downloads = {}
    download = st.download_button

    def capture(label, data, *args, **kwargs):
        downloads[kwargs.get("key")] = data
        return download(label, data, *args, **kwargs)

    monkeypatch.setattr(st, "download_button", capture)
    app = AppTest.from_file(
        Path(__file__).parents[1] / "app.py", default_timeout=60
    ).run()
    app.radio(key="evolution").set_value("Квазилинейная теплопроводность").run()
    app.radio(key="nonlinear_dimension").set_value("3D").run()
    assert not any(widget.key == "h3_batch_size" for widget in app.number_input)
    app.multiselect(key="h3_methods").set_value(
        ["mc_grouped", "mc_control_grouped"]
    ).run()
    app.select_slider(key="h3_n").set_value(6)
    app.selectbox(key="h3_case").set_value("polynomial")
    app.selectbox(key="h3_kind_x0").set_value(3)
    app.number_input(key="h3_x").set_value(0)
    app.number_input(key="h3_capacity_gamma").set_value(0.4)
    app.number_input(key="h3_reaction_sigma").set_value(0.25)
    app.number_input(key="h3_T").set_value(0.003)
    app.number_input(key="h3_dt").set_value(0.002)
    app.number_input(key="h3_walks").set_value(1024)
    app.number_input(key="h3_mc_seed").set_value(179)
    app.number_input(key="h3_batch_size").set_value(7)
    app.number_input(key="h3_coarse_n").set_value(4)
    app.button(key="h3_run").click().run()
    assert not app.error and not app.exception
    with zipfile.ZipFile(io.BytesIO(downloads["h3_comparison_zip"])) as archive:
        saved = json.loads(archive.read("result.json"))
        rows = list(
            csv.DictReader(io.StringIO(archive.read("table.csv").decode("utf-8-sig")))
        )
    for row, csv_row in zip(saved["rows"], rows, strict=True):
        assert row["status"] == "completed" and row["physical_steps"] == 2
        assert row["completed_until"] == 0.003 and row["validation_walks"] > 0
        assert row["batch_size"] == 7 and csv_row["batch_size"] == "7"
    before = app.session_state["h3_comparison"]
    assert before["metadata"]["configuration"]["capacity_gamma"] == 0.4
    assert before["metadata"]["configuration"]["reaction_sigma"] == 0.25
    app.number_input(key="h3_capacity_gamma").set_value(0.6).run()
    assert (
        app.session_state["h3_comparison"]["metadata"]["configuration"][
            "capacity_gamma"
        ]
        == 0.4
    )
    app.number_input(key="h3_batch_size").set_value(13).run()
    assert app.session_state["h3_comparison"]["history"] == before["history"]
    assert (
        app.session_state["h3_comparison"]["metadata"]["solver_settings"]["batch_size"]
        == 7
    )
    assert any("изменены" in entry.value for entry in app.info)
    app.number_input(key="h3_repeats").set_value(2)
    app.multiselect(key="h3_walk_budgets").set_value([1024])
    app.button(key="h3_study_run").click().run()
    assert not app.error and not app.exception
    study = json.loads(downloads["h3_study_json"])
    assert len(study["rows"]) == 6
    assert study["metadata"]["configuration"]["capacity_gamma"] == 0.6
    assert study["metadata"]["configuration"]["reaction_sigma"] == 0.25
    assert study["metadata"]["solver_settings"]["batch_size"] == 13
    assert study["budgets_by_method"] == {
        method: [1024] for method in GROUPED_MC_METHODS
    }
    for collection in ("rows", "trials", "series"):
        assert all(row["batch_size"] == 13 for row in study[collection])
    for method in GROUPED_MC_METHODS:
        trials = [trial for trial in study["trials"] if trial["method"] == method]
        assert {trial["phase"] for trial in trials} == {"pilot", "final"}
        assert len({trial["seed"] for trial in trials}) == len(trials)
        assert all(trial["validation_walks"] > 0 for trial in trials)
    app.run()
    assert app.session_state["h3_comparison"]["history"] == before["history"]
