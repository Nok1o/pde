"""Two deterministic methods share the 3D workflow and fixed study protocol."""

import io
import json
import shlex
import zipfile
from dataclasses import replace

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from pdelab.heat3d_cli import main
from pdelab.heat3d_experiment import (
    METHOD_NAME,
    METHOD_NAMES,
    HeatConfig3D,
    compare_heat3d,
    result_archive,
    study_heat3d,
)
from pdelab.heat3d_solver import HeatSettings3D
from pdelab.study_protocol import METRIC_ID, PROTOCOL_ID


def heat_ui_app():
    from pdelab.heat3d_ui import render_heat3d_app

    render_heat3d_app()


@pytest.mark.parametrize("methods", [(), ("lod", "lod"), ("newton",), "lod"])
def test_service_rejects_empty_duplicate_or_diagnostic_methods(methods):
    with pytest.raises(ValueError, match="методы"):
        compare_heat3d(HeatConfig3D(n=4, T=0, methods=methods))


def test_two_method_cli_run_exports_results_and_separate_newton_diagnostic(tmp_path):
    assert (
        main(
            [
                "run",
                "--n",
                "4",
                "--case",
                "polynomial",
                "--T",
                ".015",
                "--dt",
                ".01",
                "--methods",
                "block",
                "lod",
                "--newton",
                "--query",
                "field",
                "--out",
                str(tmp_path),
            ]
        )
        == 0
    )
    payload = json.loads((tmp_path / "result.json").read_text())
    assert payload["metadata"]["methods"] == ["block", "lod"]
    assert payload["metadata"]["configuration"]["methods"] == ["block", "lod"]
    assert payload["metadata"]["method"] == "multiple"
    rows = payload["rows"]
    assert [row["method"] for row in rows] == ["block", "lod"]
    assert rows[1]["method_name"] == METHOD_NAME
    assert rows[0]["method_name"] == METHOD_NAMES["block"]
    assert all(
        row["status"] == "completed" and row["physical_steps"] == 2 for row in rows
    )
    assert rows[0]["factorizations"] > 0
    for key in ("assembly_seconds", "factorization_seconds", "solve_seconds"):
        assert 0 <= rows[0][key] <= rows[0]["seconds"]
    assert payload["grid_diagnostic"]["status"] == "completed"
    assert payload["grid_diagnostic"]["block_vs_grid"]["relative_error_pct"] < 0.001
    assert payload["grid_diagnostic"]["lod_vs_grid"]["relative_error_pct"] < 0.001
    assert set(row["method"] for row in payload["history"]) == {
        "lod",
        "block",
        "newton",
    }
    with np.load(tmp_path / "query-values.npz", allow_pickle=False) as arrays:
        assert set(arrays.files) == {"points_zyx", "analytic", "lod", "block", "newton"}
        np.testing.assert_allclose(
            arrays["lod"], arrays["block"], rtol=1e-8, atol=1e-10
        )
    command = shlex.split(payload["metadata"]["reproduction"])
    assert command[command.index("--methods") + 1 : command.index("--methods") + 3] == [
        "block",
        "lod",
    ]
    assert "block" in (tmp_path / "table.csv").read_text()
    with zipfile.ZipFile(tmp_path / "heat3d.zip") as archive:
        assert json.loads(archive.read("result.json")) == payload


def test_failed_methods_preserve_individual_candidates_and_no_target_solution():
    result = compare_heat3d(
        HeatConfig3D(n=4, methods=("lod", "block")),
        HeatSettings3D(max_outer=1),
    )
    assert [row["status"] for row in result["rows"]] == ["outer_limit", "outer_limit"]
    assert all(row["relative_error_pct"] is None for row in result["rows"])
    assert "lod" not in result["arrays"] and "block" not in result["arrays"]
    assert result["arrays"]["failed_candidate"].shape == (5, 5, 5)
    assert result["arrays"]["block_failed_candidate"].shape == (5, 5, 5)
    with zipfile.ZipFile(io.BytesIO(result_archive(result))) as archive:
        with np.load(
            io.BytesIO(archive.read("query-values.npz")), allow_pickle=False
        ) as arrays:
            assert "failed_candidate" in arrays and "block_failed_candidate" in arrays


def test_block_cap_refusal_is_exported_without_replacing_successful_lod(tmp_path):
    with pytest.raises(SystemExit) as exc:
        main(
            [
                "run",
                "--n",
                "24",
                "--T",
                "0",
                "--methods",
                "lod",
                "block",
                "--out",
                str(tmp_path),
            ]
        )
    assert exc.value.code == 2
    result = json.loads((tmp_path / "result.json").read_text())
    assert [row["status"] for row in result["rows"]] == ["completed", "unsupported"]
    assert "16" in result["rows"][1]["reason"]
    with np.load(tmp_path / "query-values.npz", allow_pickle=False) as arrays:
        assert "lod" in arrays and "block" not in arrays


def test_two_method_study_exports_method_tags_and_exact_protocol(tmp_path):
    assert (
        main(
            [
                "study",
                "--n",
                "4",
                "--T",
                "0",
                "--methods",
                "lod",
                "block",
                "--repeats",
                "2",
                "--budgets",
                "1e-3",
                "1e-7",
                "--out",
                str(tmp_path),
            ]
        )
        == 0
    )
    result = json.loads((tmp_path / "result.json").read_text())
    assert result["protocol"] == PROTOCOL_ID
    assert result["accuracy_metric"] == METRIC_ID
    assert len(result["rows"]) == 6
    assert all(row["status"] == "confirmed" for row in result["rows"])
    for method in ("lod", "block"):
        trials = [row for row in result["trials"] if row["method"] == method]
        assert [row["phase"] for row in trials] == ["pilot", "pilot", "final", "final"]
        assert all(row["method_name"] == METHOD_NAMES[method] for row in trials)
        assert all(
            row["method_name"] == METHOD_NAMES[method]
            for row in result["series"]
            if row["method"] == method
        )
    assert result["elapsed_seconds"] == pytest.approx(
        sum(row["seconds"] for row in result["trials"])
    )
    assert not (tmp_path / "query-values.npz").exists()
    assert "method" in (tmp_path / "trials.csv").read_text()


def test_each_method_finishes_pilots_before_final_and_never_retunes(monkeypatch):
    import pdelab.heat3d_solver as solver

    base = solver.solve_heat3d
    calls, counts = [], {}

    def controlled(problem, points, *, settings, method, **kwargs):
        budget = settings.tolerance
        key = (method, budget)
        repeat = counts.get(key, 0)
        counts[key] = repeat + 1
        calls.append(key)
        result = base(problem, points, settings=settings, method=method, **kwargs)
        if method == "lod" and budget == 1e-3:
            error = 0.06 if repeat < 2 else 0.12
        elif method == "block":
            error = 0.001 if repeat < 2 else 0.11
        else:
            error = 0.001
        return replace(result, values=result.values * (1 + error))

    monkeypatch.setattr(solver, "solve_heat3d", controlled)
    result = study_heat3d(
        HeatConfig3D(n=4, T=0, methods=("lod", "block")),
        repeats=2,
        budgets=(1e-3, 1e-7),
    )
    assert calls == (
        [("lod", 1e-3)] * 2
        + [("lod", 1e-7)] * 2
        + [("lod", 1e-3)] * 2
        + [("lod", 1e-7)] * 2
        + [("block", 1e-3)] * 4
    )
    lod_rows = [row for row in result["rows"] if row["method"] == "lod"]
    block_rows = [row for row in result["rows"] if row["method"] == "block"]
    assert [row["status"] for row in lod_rows] == [
        "confirmed",
        "confirmed",
        "validation_failed",
    ]
    assert all(
        row["status"] == "validation_failed" and row["seconds"] is None
        for row in block_rows
    )
    assert all(row["budget"] == 1e-3 for row in block_rows)


def test_ui_selected_methods_and_saved_results_share_one_workflow():
    app = AppTest.from_function(heat_ui_app, default_timeout=60).run()
    assert app.multiselect(key="h3_methods").value == ["lod"]
    app.select_slider(key="h3_n").set_value(4)
    app.number_input(key="h3_T").set_value(0.0)
    app.multiselect(key="h3_methods").set_value(["lod", "block"])
    app.selectbox(key="h3_mode").set_value("slice").run()
    app.button(key="h3_run").click().run()
    assert not app.exception and not app.error
    assert len(app.dataframe[0].value) == 2
    assert app.session_state["h3_comparison"]["metadata"]["methods"] == ["lod", "block"]
    app.selectbox(key="h3_value_method").set_value("block").run()
    app.selectbox(key="h3_view").set_value("Абсолютная ошибка").run()
    assert not app.exception
    app.number_input(key="h3_repeats").set_value(2)
    app.multiselect(key="h3_budgets").set_value([1e-3])
    app.button(key="h3_study_run").click().run()
    assert not app.exception and not app.error
    assert len(app.session_state["h3_study"]["rows"]) == 6
    app.multiselect(key="h3_methods").set_value(["block"]).run()
    assert not app.exception and any("изменены" in item.value for item in app.info)
    assert app.session_state["h3_comparison"]["metadata"]["methods"] == ["lod", "block"]
    assert app.session_state["h3_study"]["metadata"]["methods"] == ["lod", "block"]
