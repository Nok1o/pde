import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from pdelab.nonlinear import NonlinearSettings, make_nonlinear_problem
from pdelab.nonlinear_cli import main
from pdelab.nonlinear_io import comparison_archive
from pdelab.nonlinear_study import compare_nonlinear, nonlinear_targets


def test_cli_exports_three_iteration_levels_and_full_chains(tmp_path):
    main(
        [
            "run",
            "--n",
            "4",
            "--T",
            ".02",
            "--repeats",
            "2",
            "--walks",
            "32",
            "--out",
            str(tmp_path),
        ]
    )
    payload = json.loads((tmp_path / "experiment.json").read_text())
    assert payload["array_order"] == "yx" and payload["alpha"] == 0.5
    assert len(payload["runs"]) == 3
    assert (
        len({r["settings"]["seed"] for r in payload["runs"] if r["method"] == "mc"})
        == 2
    )
    assert all(r["completed"] for r in payload["runs"])
    assert payload["runs"][0]["layers"][0]["inner_iterations"] > 0
    with np.load(tmp_path / "fields.npz", allow_pickle=False) as arrays:
        assert arrays["times"][-1] == 0.02
        assert arrays["mc_0"].shape == arrays["newton"].shape
    for name in ("time.html", "field.html", "error.html", "residual.html"):
        text = (tmp_path / name).read_text()
        assert "Plotly.newPlot" in text and 'src="https://cdn.plot.ly' not in text


def test_incomplete_chains_export_reasons_without_claiming_target_error(tmp_path):
    with pytest.raises(SystemExit) as error:
        main(
            [
                "run",
                "--n",
                "4",
                "--T",
                ".02",
                "--repeats",
                "2",
                "--max-total-walks",
                "1",
                "--out",
                str(tmp_path),
            ]
        )
    assert error.value.code == 2
    payload = json.loads((tmp_path / "experiment.json").read_text())
    assert not payload["metrics"][1]["completed"]
    assert payload["metrics"][1]["rmse"] is None
    assert "walk_budget" in payload["runs"][1]["reason"]


def test_archive_and_zero_boundary_query():
    comparison = compare_nonlinear(
        make_nonlinear_problem(4), T=0.02, point=(0, 0), repeats=2
    )
    assert all(row["relative_error_pct"] == 0 for row in comparison.metrics)
    with zipfile.ZipFile(io.BytesIO(comparison_archive(comparison))) as archive:
        assert "iterations.csv" in archive.namelist()
        assert json.loads(archive.read("experiment.json"))["point_xy"] == [0, 0]


def test_failed_pilot_has_no_confirmed_mc_time():
    rows, evidence = nonlinear_targets(
        make_nonlinear_problem(4),
        T=0.02,
        repeats=2,
        budgets=(8,),
        settings=NonlinearSettings(max_total_walks=1),
    )
    mc = [row for row in rows if row["method"] == "mc"]
    assert all(row["status"] == "not_reached" and row["seconds"] is None for row in mc)
    assert evidence["validations"] == []


def test_ui_comparison_history_targets_export_and_old_modes():
    app = AppTest.from_file(
        Path(__file__).parents[1] / "app.py", default_timeout=60
    ).run()
    app.radio(key="evolution").set_value("Нелинейная теплопроводность").run()
    assert not app.exception
    app.select_slider(key="nl_n").set_value(4).run()
    app.number_input(key="nl_T").set_value(0.02)
    app.number_input(key="nl_repeats").set_value(2)
    app.button(key="nl_run").click().run()
    assert not app.exception and not app.error
    assert app.session_state["nl_result"].reference.times[-1] == 0.02
    assert len(app.latex) >= 4
    app.button(key="nl_export").click().run()
    assert app.session_state["nl_archive"]
    app.multiselect(key="nl_budgets").set_value([32])
    app.button(key="nl_targets_run").click().run()
    assert not app.exception and not app.error
    assert len(app.session_state["nl_targets"][1]) == 6
    app.number_input(key="nl_alpha").set_value(1.0).run()
    assert any("изменены" in item.value for item in app.info)
    app.radio(key="evolution").set_value("Теплопроводность").run()
    assert not app.exception and app.button(key="heat_run")
    app.radio(key="evolution").set_value("Стационарный").run()
    assert not app.exception and app.button(key="main_run")
