"""User workflow shares its settings, metrics and evidence with the CLI."""

import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from pdelab.heat3d_cli import main
from pdelab.heat3d_experiment import (
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


def test_cli_comparison_exports_query_history_and_independent_reference(tmp_path):
    assert (
        main(
            [
                "run",
                "--case",
                "polynomial",
                "--n",
                "4",
                "--T",
                ".015",
                "--dt",
                ".01",
                "--point",
                "1",
                "2",
                "3",
                "--newton",
                "--out",
                str(tmp_path),
            ]
        )
        == 0
    )
    payload = json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))
    metadata = payload["metadata"]
    row = payload["rows"][0]
    assert metadata["points_zyx"] == [[3, 2, 1]]
    assert (
        metadata["schema_version"] == "heat3d_v1"
        and metadata["accuracy_metric"] == METRIC_ID
    )
    assert all(kind == 1 for kind in metadata["boundary_kinds"].values())
    assert metadata["commit"] and metadata["versions"]["numpy"]
    assert row["status"] == "completed" and row["physical_steps"] == 2
    assert row["target_time"] == row["completed_until"] == 0.015
    assert payload["grid_diagnostic"]["status"] == "completed"
    assert payload["grid_diagnostic"]["lod_vs_grid"]["relative_error_pct"] < 0.001
    assert row["peak_rss_bytes"] is None
    assert "history.csv" in {path.name for path in tmp_path.iterdir()}
    with np.load(tmp_path / "query-values.npz", allow_pickle=False) as arrays:
        assert set(arrays.files) == {"points_zyx", "analytic", "lod", "newton"}
        assert arrays["lod"].shape == (1,)
    with zipfile.ZipFile(tmp_path / "heat3d.zip") as archive:
        assert json.loads(archive.read("result.json")) == payload


@pytest.mark.parametrize(
    "mode,kwargs,count",
    [
        ("point", dict(point=(1, 2, 3)), 1),
        ("patch", {}, 27),
        ("slice", dict(plane="yz", index=1), 9),
        ("field", {}, 27),
    ],
)
def test_comparison_query_modes_use_fixed_order_and_analytic_metric(
    mode, kwargs, count
):
    result = compare_heat3d(
        HeatConfig3D(case="polynomial", n=4, T=0, mode=mode, **kwargs)
    )
    assert result["metadata"]["query_size"] == count
    assert result["rows"][0]["relative_error_pct"] == 0
    assert len(result["arrays"]["analytic"]) == count
    if mode == "slice":
        assert all(point[2] == 1 for point in result["metadata"]["points_zyx"])


def test_failed_run_exports_status_and_actual_cost_without_target_values(tmp_path):
    with pytest.raises(SystemExit) as exc:
        main(["run", "--n", "4", "--max-outer", "1", "--out", str(tmp_path)])
    assert exc.value.code == 2
    payload = json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))
    row = payload["rows"][0]
    assert row["status"] == "outer_limit" and row["physical_steps"] == 0
    assert row["relative_error_pct"] is None
    assert row["seconds"] > 0 and row["outer_iterations"] == 1
    with np.load(tmp_path / "query-values.npz", allow_pickle=False) as arrays:
        assert "lod" not in arrays
        assert arrays["failed_candidate"].shape == (5, 5, 5)
    assert row["diagnostic_time"] == 0.01 and row["diagnostic_field_available"]
    assert payload["history"]


def test_repeated_exports_remove_only_obsolete_managed_artifacts(tmp_path):
    user_file = tmp_path / "notes.txt"
    user_file.write_text("keep", encoding="utf-8")
    assert main(["run", "--n", "4", "--T", "0", "--out", str(tmp_path)]) == 0
    assert (tmp_path / "query-values.npz").exists()
    assert (
        main(
            ["study", "--n", "6", "--T", "0", "--repeats", "2", "--out", str(tmp_path)]
        )
        == 0
    )
    assert not (tmp_path / "query-values.npz").exists()
    assert not (tmp_path / "history.csv").exists()
    assert (tmp_path / "trials.csv").exists()
    assert main(["run", "--n", "4", "--T", "0", "--out", str(tmp_path)]) == 0
    assert not (tmp_path / "trials.csv").exists()
    assert not (tmp_path / "series.csv").exists()
    assert user_file.read_text(encoding="utf-8") == "keep"


def test_cli_study_matches_service_and_exports_separate_final_trials(tmp_path):
    args = [
        "study",
        "--n",
        "4",
        "--T",
        "0",
        "--query",
        "slice",
        "--plane",
        "yz",
        "--index",
        "1",
        "--repeats",
        "2",
        "--seed",
        "123",
        "--budgets",
        "1e-3",
        "1e-7",
        "--margin",
        ".8",
        "--out",
        str(tmp_path),
    ]
    assert main(args) == 0
    payload = json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))
    expected = study_heat3d(
        HeatConfig3D(n=4, T=0, mode="slice", plane="yz", index=1),
        repeats=2,
        seed=123,
        budgets=(1e-3, 1e-7),
    )
    assert payload["protocol"] == PROTOCOL_ID
    assert payload["metadata"]["configuration"] == expected["metadata"]["configuration"]
    for actual, row in zip(payload["rows"], expected["rows"], strict=True):
        for key in (
            "target_pct",
            "status",
            "budget",
            "relative_error_pct",
            "completed_repeats",
            "requested_repeats",
            "accuracy_metric",
            "total_walks",
        ):
            assert actual[key] == row[key]
    assert [r["seed"] for r in payload["trials"]] == [
        r["seed"] for r in expected["trials"]
    ]
    assert [r["phase"] for r in payload["trials"]] == [
        "pilot",
        "pilot",
        "final",
        "final",
    ]
    assert all("values" not in r for r in payload["trials"])
    assert (tmp_path / "trials.csv").exists() and (tmp_path / "series.csv").exists()


def test_unreached_study_preserves_rows_and_returns_failure(tmp_path):
    with pytest.raises(SystemExit) as exc:
        main(
            [
                "study",
                "--n",
                "4",
                "--max-outer",
                "1",
                "--repeats",
                "2",
                "--budgets",
                "1e-7",
                "--out",
                str(tmp_path),
            ]
        )
    assert exc.value.code == 2
    payload = json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))
    assert all(
        row["status"] == "pilot_incomplete" and row["solver_status"] == "outer_limit"
        for row in payload["rows"]
    )
    assert len(payload["trials"]) == 2 and payload["elapsed_seconds"] > 0


def test_unsupported_newton_is_diagnostic_and_does_not_replace_lom():
    result = compare_heat3d(HeatConfig3D(n=24, T=0, newton=True))
    assert result["rows"][0]["method"] == "lod"
    assert result["rows"][0]["status"] == "completed"
    assert result["grid_diagnostic"]["status"] == "unsupported"
    assert "newton" not in result["arrays"]


def test_cli_requested_unavailable_diagnostic_returns_failure_after_export(tmp_path):
    with pytest.raises(SystemExit) as exc:
        main(["run", "--n", "24", "--T", "0", "--newton", "--out", str(tmp_path)])
    assert exc.value.code == 2
    result = json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))
    assert result["rows"][0]["status"] == "completed"
    assert result["grid_diagnostic"]["status"] == "unsupported"


@pytest.mark.parametrize(
    "arguments",
    [
        ["study", "--newton"],
        ["run", "--query", "field", "--point", "1", "2", "3"],
        ["run", "--plane", "yz"],
    ],
)
def test_cli_rejects_inapplicable_arguments(arguments):
    with pytest.raises(SystemExit) as exc:
        main(arguments)
    assert exc.value.code == 2


def test_ui_runs_comparison_and_study_and_preserves_saved_parameters(monkeypatch):
    import pdelab.heat3d_ui as ui

    calls = []
    real_compare, real_study = ui.compare_heat3d, ui.study_heat3d

    def comparison(*args, **kwargs):
        calls.append("run")
        return real_compare(*args, **kwargs)

    def study(*args, **kwargs):
        calls.append("study")
        return real_study(*args, **kwargs)

    monkeypatch.setattr(ui, "compare_heat3d", comparison)
    monkeypatch.setattr(ui, "study_heat3d", study)
    app = AppTest.from_function(heat_ui_app, default_timeout=60).run()
    app.select_slider(key="h3_tol").set_value(1e-6)
    app.select_slider(key="h3_change").set_value(1e-6)
    app.select_slider(key="h3_n").set_value(4)
    app.number_input(key="h3_T").set_value(0.0)
    app.selectbox(key="h3_mode").set_value("slice").run()
    app.button(key="h3_run").click().run()
    assert not app.exception and not app.error
    assert app.session_state["h3_sections"] == "Сравнение"
    saved = app.session_state["h3_comparison"]
    assert saved["metadata"]["configuration"]["n"] == 4
    assert saved["metadata"]["solver_settings"]["tolerance"] == 1e-6
    assert saved["metadata"]["solver_settings"]["change_tolerance"] == 1e-6
    app.selectbox(key="h3_view").set_value("Абсолютная ошибка").run()
    assert calls == ["run"]
    app.number_input(key="h3_repeats").set_value(2)
    app.multiselect(key="h3_budgets").set_value([1e-3, 1e-6])
    app.button(key="h3_study_run").click().run()
    assert not app.exception and not app.error
    assert app.session_state["h3_sections"] == "Исследование 1/5/10%"
    saved_study = app.session_state["h3_study"]
    assert saved_study["metadata"]["budgets"] == [1e-3, 1e-6]
    assert all(row["status"] == "confirmed" for row in saved_study["rows"])
    app.number_input(key="h3_alpha").set_value(1.0).run()
    assert not app.exception and calls == ["run", "study"]
    assert any("изменены" in item.value for item in app.info)
    assert (
        app.session_state["h3_comparison"]["metadata"]["configuration"]["alpha"] == 0.5
    )
    with zipfile.ZipFile(io.BytesIO(result_archive(saved_study))) as archive:
        payload = json.loads(archive.read("result.json"))
        assert payload["metadata"]["configuration"]["alpha"] == 0.5
        assert payload["requested_repeats"] == 2


def test_comparison_keeps_memory_refusal_as_evidence():
    result = compare_heat3d(HeatConfig3D(n=4), HeatSettings3D(max_workspace_bytes=1))
    assert result["rows"][0]["status"] == "memory_limit"
    assert result["rows"][0]["absolute_error"] is None


def test_ui_failed_series_shows_reasons_without_null_error_cells():
    app = AppTest.from_function(heat_ui_app, default_timeout=60).run()
    app.select_slider(key="h3_n").set_value(4)
    app.number_input(key="h3_outer").set_value(1)
    app.number_input(key="h3_repeats").set_value(2)
    app.multiselect(key="h3_budgets").set_value([1e-7])
    app.button(key="h3_study_run").click().run()
    assert not app.exception and not app.error
    row = app.dataframe[0].value.iloc[0]
    assert row["Статус"] == "Предварительная серия не завершена"
    assert "нет оценки" in row["Ошибка, %"]
    assert row["Повторы"] == "0/2"
    assert row["Время при достижении, с"] == "не подтверждено"


def test_main_app_routes_nonlinear_3d_and_preserves_other_modes():
    app = AppTest.from_file(
        str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=60
    ).run()
    app.radio(key="evolution").set_value("Квазилинейная теплопроводность").run()
    app.radio(key="nonlinear_dimension").set_value("3D").run()
    assert not app.exception
    assert app.title[0].value == "Квазилинейная теплопроводность · 3D"
    app.number_input(key="h3_T").set_value(0.0)
    app.button(key="h3_run").click().run()
    assert not app.exception and not app.error
    assert app.session_state["h3_comparison"]["rows"][0]["physical_steps"] == 0
    app.radio(key="nonlinear_dimension").set_value("2D").run()
    assert not app.exception and "3D" not in app.title[0].value
    app.radio(key="evolution").set_value("Теплопроводность").run()
    assert not app.exception
