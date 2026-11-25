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


def test_ui_executes_full_history_and_reproduces_through_cli(tmp_path, monkeypatch):
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
    app.multiselect(key="h3_methods").set_value(["mc", "mc_control"]).run()
    app.select_slider(key="h3_n").set_value(6)
    app.selectbox(key="h3_case").set_value("polynomial")
    app.selectbox(key="h3_kind_x0").set_value(3)
    app.number_input(key="h3_x").set_value(0)
    app.number_input(key="h3_T").set_value(0.003)
    app.number_input(key="h3_dt").set_value(0.002)
    app.number_input(key="h3_walks").set_value(1024)
    app.number_input(key="h3_mc_seed").set_value(179)
    app.button(key="h3_run").click().run()
    assert not app.error and not app.exception
    run = app.session_state["h3_comparison"]
    assert run["metadata"]["points_zyx"][0][2] == 0
    with zipfile.ZipFile(io.BytesIO(downloads["h3_comparison_zip"])) as archive:
        saved = json.loads(archive.read("result.json"))
        rows = list(
            csv.DictReader(io.StringIO(archive.read("table.csv").decode("utf-8-sig")))
        )
    for row, csv_row in zip(saved["rows"], rows, strict=True):
        assert row["method_name"] == csv_row["method_name"]
        assert row["status"] == "completed" and row["physical_steps"] == 2
        assert (
            row["completed_until"] == 0.003
            and row["walks"] == int(csv_row["walks"]) > 0
        )
        assert row["seed"] == 179 and row["walks_per_free_node"] == 1024
    assert (
        main(
            [
                *shlex.split(saved["metadata"]["reproduction"])[3:-2],
                "--out",
                str(tmp_path),
            ]
        )
        == 0
    )
    regenerated = json.loads((tmp_path / "result.json").read_bytes())
    for row, repeated in zip(saved["rows"], regenerated["rows"], strict=True):
        for key in ("seed", "coarse_n", "walks", "transitions", "relative_error_pct"):
            assert row[key] == repeated[key]

    app.number_input(key="h3_repeats").set_value(2)
    app.multiselect(key="h3_walk_budgets").set_value([1024])
    app.button(key="h3_study_run").click().run()
    assert not app.error and not app.exception
    study = json.loads(downloads["h3_study_json"])
    assert len(study["rows"]) == 6
    assert study["budgets_by_method"] == {"mc": [1024], "mc_control": [1024]}
    for trial in study["trials"]:
        assert (
            trial["walks"] == trial["completed_walks"] > trial["validation_walks"] > 0
        )
        assert trial["completed_until"] == 0.003 and trial["values"] is not None
    for method in ("mc", "mc_control"):
        trials = [t for t in study["trials"] if t["method"] == method]
        assert {t["phase"] for t in trials} == {"pilot", "final"}
        assert len({t["seed"] for t in trials}) == len(trials)
    csv_rows = list(
        csv.DictReader(io.StringIO(downloads["h3_study_csv"].decode("utf-8-sig")))
    )
    assert [r["status"] for r in csv_rows] == [r["status"] for r in study["rows"]]
    assert {r["target_pct"] for r in study["rows"]} == {1, 5, 10}
    app.run()
    assert app.session_state["h3_comparison"]["history"] == run["history"]


def test_mixed_method_study_keeps_units_and_refusals(tmp_path):
    with pytest.raises(SystemExit) as exc:
        main(
            [
                "study",
                "--methods",
                "lod",
                "mc",
                "mc_control",
                "--x0",
                "3",
                "--n",
                "5",
                "--dt",
                ".002",
                "--T",
                ".003",
                "--max-walks",
                "1",
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
    assert study["budgets_by_method"] == {
        "lod": [0.001],
        "mc": [32],
        "mc_control": [32],
    }
    for t in study["trials"]:
        if t["method"] == "lod":
            assert t["walks"] == 0 and t["status"] == "completed"
        else:
            assert t["status"] == "walk_budget" and t["values"] is None
            assert t["walks"] == t["completed_walks"] == 1
    for row in study["rows"]:
        if row["method"] != "lod":
            assert row["seconds"] is None and row["relative_error_pct"] is None
            assert row["completed_repeats"] == 0 and row["total_walks"] == 2
