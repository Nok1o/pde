"""Exercise the stationary reflection methods through the CLI and the app entry point."""

import csv
import io
import json
import zipfile
from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from pdelab.boundary_cli3d import main
from pdelab.boundary_solvers3d import CACHED_METHODS


def test_cli_comparison_and_budget_series(tmp_path, monkeypatch):
    import pdelab.boundary_experiment3d as experiment

    solve = experiment.solve_boundary3d

    def checked_solver(problem, *args, **options):
        assert problem.exact is None
        return solve(problem, *args, **options)

    monkeypatch.setattr(experiment, "solve_boundary3d", checked_solver)
    args = [
        "--n",
        "4",
        "--methods",
        *CACHED_METHODS,
        "--batch-size",
        "7",
        "--out",
        str(tmp_path),
    ]
    main(["run", "--walks", "17", *args])
    payload = json.loads((tmp_path / "experiment.json").read_text())
    assert payload["reference_kind"] == "analytic"
    with (tmp_path / "metrics.csv").open(encoding="utf-8-sig") as stream:
        for row, exported in zip(
            payload["metrics"], csv.DictReader(stream), strict=True
        ):
            assert exported["name"] == row["name"] == CACHED_METHODS[row["method"]]
            assert row["walks"] == row["started_walks"] == 17
            assert (
                float(exported["seconds"])
                == row["seconds"]
                >= row["preparation_seconds"]
            )
            assert int(exported["steps"]) == row["steps"] > 0
            assert (
                int(exported["reflection_table_bytes"])
                == row["reflection_table_bytes"]
                > 0
            )
    main(["benchmark", "--budgets", "17", "--repeats", "2", *args])
    with (tmp_path / "benchmark.csv").open(encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 4
    assert {r["method"] for r in rows} == set(CACHED_METHODS)
    assert all(
        r["batch_size"] == ("7" if r["method"] == "mc_grouped_cached" else "")
        for r in rows
    )


@pytest.mark.parametrize("method", CACHED_METHODS)
def test_cli_study_records_failed_paths_and_all_unconfirmed_targets(tmp_path, method):
    main(
        [
            "study",
            "--n",
            "4",
            "--methods",
            method,
            "--budgets",
            "17",
            "--repeats",
            "2",
            "--batch-size",
            "7",
            "--max-steps",
            "1",
            "--out",
            str(tmp_path),
        ]
    )
    study = json.loads((tmp_path / "study.json").read_text())
    started = 7 if method == "mc_grouped_cached" else 1
    assert len(study["trials"]) == 2
    for trial in study["trials"]:
        assert trial["status"] == "path_limit" and trial["values"] is None
        assert trial["walks"] == 0
        assert trial["transitions"] == trial["started_walks"] == started
    with (tmp_path / "timings.csv").open(encoding="utf-8-sig") as stream:
        for row, exported in zip(study["rows"], csv.DictReader(stream), strict=True):
            assert exported["status"] == row["status"] == "pilot_incomplete"
            assert row["seconds"] is None and row["relative_error_pct"] is None
            assert int(exported["started_walks"]) == row["started_walks"] == 2 * started
            assert int(exported["transitions"]) == 2 * started
            assert int(exported["total_walks"]) == 0


def test_app_runs_comparison_and_independent_study_with_downloads(monkeypatch):
    downloads = {}
    download = st.download_button

    def capture(label, data, *args, **kwargs):
        if key := kwargs.get("key"):
            downloads[key] = data
        return download(label, data, *args, **kwargs)

    monkeypatch.setattr(st, "download_button", capture)
    app = AppTest.from_file(
        Path(__file__).parents[1] / "app.py", default_timeout=90
    ).run()
    app.radio(key="dimension").set_value("3D").run()
    app.radio(key="cube_boundary_mode").set_value("Смешанные: 1, 2, 3 рода").run()
    methods = app.multiselect(key="bc3d_methods")
    assert set(CACHED_METHODS.values()) <= set(methods.options)
    assert not set(CACHED_METHODS) & set(methods.value)
    methods.set_value(list(CACHED_METHODS))
    app.select_slider(key="bc3d_n").set_value(8)
    app.select_slider(key="bc3d_walks").set_value(64)
    for axis, coordinate in zip("xyz", (0, 4, 4)):
        app.number_input(key="bc3d_" + axis).set_value(coordinate)
    app.number_input(key="bc3d_batch_size").set_value(7)
    app.button(key="bc3d_run").click().run()
    assert not app.error and not app.exception
    comparison = app.session_state["bc3d_result"]
    assert {r["method"] for r in comparison.metrics} == set(CACHED_METHODS)
    assert all(r["walks"] == 64 and r["steps"] > 0 for r in comparison.metrics)
    app.button(key="bc3d_export").click().run()
    with zipfile.ZipFile(io.BytesIO(downloads["bc3d_download"])) as archive:
        exported = json.loads(archive.read("experiment.json"))
    assert exported["metrics"] == comparison.metrics
    assert exported["config"]["batch_size"] == 7

    app.select_slider(key="bc3_research_n").set_value(8)
    for axis, coordinate in zip("xyz", (0, 4, 4)):
        app.number_input(key="bc3_research" + axis).set_value(coordinate)
    app.number_input(key="bc3_research_batch_size").set_value(5)
    app.multiselect(key="bc3_research_budgets").set_value([64, 256])
    app.number_input(key="bc3_research_repeats").set_value(2)
    app.button(key="bc3_research_run").click().run()
    assert not app.error and not app.exception
    study = app.session_state["bc3_research"][1]
    assert json.loads(json.dumps(study)) == json.loads(downloads["bc3_research_json"])
    assert study["config"]["batch_size"] == 5
    assert study["config"]["point"] == (0, 4, 4)
    records = list(csv.DictReader(io.StringIO(downloads["bc3_research_csv"])))
    assert len(records) == len(study["rows"]) == 6
    for row, exported in zip(study["rows"], records, strict=True):
        assert exported["name"] == CACHED_METHODS[row["method"]]
        assert exported["status"] == row["status"]
        if row["status"] == "confirmed":
            assert float(exported["seconds"]) == row["seconds"] > 0
        else:
            assert exported["seconds"] == "" and row["seconds"] is None
        assert exported["batch_size"] == (
            "5" if row["method"] == "mc_grouped_cached" else ""
        )
    assert {row["target_pct"] for row in study["rows"]} == {1, 5, 10}
    assert {t["phase"] for t in study["trials"]} == {"pilot", "final"}
    assert {t["method"] for t in study["trials"] if t["phase"] == "final"} == set(
        CACHED_METHODS
    )
    assert all(
        t["transitions"] >= t["walks"] == t["started_walks"] == t["budget"]
        for t in study["trials"]
    )
    assert len({t["seed"] for t in study["trials"]}) == len(study["trials"])
    assert app.session_state["bc3d_result"].metrics == comparison.metrics
    app.selectbox(key="bc3d_x0_kind").set_value("dirichlet").run()
    assert not app.error and not app.exception
    assert not set(CACHED_METHODS.values()) & set(
        app.multiselect(key="bc3d_methods").options
    )
    assert any("условие 3 рода" in c.value for c in app.caption)
