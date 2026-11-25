"""Grouped MC reaches 3D comparison, saved settings and existing study controls."""

import csv
import io
import json

import numpy as np
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from pdelab.boundary import BoundaryOperator, make_boundary_problem
from pdelab.boundary_cli3d import main as boundary_main
from pdelab.boundary_experiment import BoundaryConfig, selected_boundary_methods
from pdelab.boundary_solvers import METHODS_BOUNDARY, solve_boundary
from pdelab.cli3d import main as cube_main
from pdelab.experiment import Config
from pdelab.problem import make_problem
from pdelab.solvers import METHODS, solve


@pytest.mark.parametrize("mixed", [False, True])
def test_cli_comparison_preserves_group_configuration_and_diagnostics(tmp_path, mixed):
    main = boundary_main if mixed else cube_main
    main(
        [
            "run",
            "--n",
            "4",
            "--methods",
            "mc",
            "mc_grouped",
            "--walks",
            "17",
            "--batch-size",
            "7",
            "--out",
            str(tmp_path),
        ]
    )
    payload = json.loads((tmp_path / "experiment.json").read_text(encoding="utf-8"))
    assert payload["config"]["batch_size"] == 7
    metrics = {row["method"]: row for row in payload["metrics"]}
    assert set(metrics) == {"mc", "mc_grouped"}
    grouped = metrics["mc_grouped"]
    assert grouped["name"] == "MC · группы частиц"
    assert grouped["batch_size"] == grouped["effective_batch_size"] == 7
    assert grouped["walks"] == grouped["started_walks"] == 17
    assert grouped["workspace_bytes"] > 0
    assert grouped["rng"] == "numba_mt19937_interleaved_groups_v1"
    assert 0 <= grouped["preparation_seconds"] <= grouped["seconds"]
    assert 0 <= grouped["path_seconds"] <= grouped["seconds"]
    assert metrics["mc"]["batch_size"] is None
    with (tmp_path / "metrics.csv").open(encoding="utf-8-sig", newline="") as stream:
        csv_metrics = {row["method"]: row for row in csv.DictReader(stream)}
    assert csv_metrics["mc_grouped"]["batch_size"] == "7"
    assert csv_metrics["mc_grouped"]["started_walks"] == "17"
    with np.load(tmp_path / "fields.npz", allow_pickle=False) as fields:
        assert np.isfinite(fields["mc_grouped"][2, 2, 2])
        assert np.isnan(fields["mc_grouped"][1, 1, 1])


def test_mixed_cli_budget_series_keeps_group_size_and_each_run_cost(tmp_path):
    boundary_main(
        [
            "benchmark",
            "--n",
            "4",
            "--methods",
            "mc_grouped",
            "--budgets",
            "17",
            "--repeats",
            "2",
            "--batch-size",
            "7",
            "--out",
            str(tmp_path),
        ]
    )
    settings = json.loads((tmp_path / "settings.json").read_text(encoding="utf-8"))
    assert settings["config"]["batch_size"] == 7
    assert settings["methods"] == ["mc_grouped"]
    with (tmp_path / "benchmark.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 2
    assert len({row["seed"] for row in rows}) == 2
    assert all(row["batch_size"] == "7" for row in rows)
    assert all(row["walks"] == row["started_walks"] == "17" for row in rows)


def test_2d_registry_and_solver_reject_grouped_method():
    assert "mc_grouped" not in METHODS
    assert "mc_grouped" not in METHODS_BOUNDARY
    assert "batch_size" not in Config.__dataclass_fields__
    assert "batch_size" not in BoundaryConfig.__dataclass_fields__
    boundary = make_boundary_problem(n=4)
    operator = BoundaryOperator(boundary)
    assert "mc_grouped" not in selected_boundary_methods(operator)
    with pytest.raises(ValueError):
        selected_boundary_methods(operator, ["mc_grouped"])
    with pytest.raises(ValueError):
        solve_boundary(boundary, [[2, 2]], "mc_grouped", walks=4)
    with pytest.raises(ValueError):
        solve(make_problem(n=4), [[2, 2]], "mc_grouped", walks=4)


def cube_app():
    from pdelab.ui3d import render_3d_app

    render_3d_app()


def mixed_app():
    from pdelab.boundary_ui3d import render_boundary_app3d

    render_boundary_app3d()


@pytest.mark.parametrize("mixed", [False, True])
def test_comparison_ui_runs_grouped_method_with_saved_batch_size(mixed):
    prefix = "bc3d" if mixed else "cube"
    app = AppTest.from_function(
        mixed_app if mixed else cube_app, default_timeout=60
    ).run()
    app.multiselect(key=prefix + "_methods").set_value(["mc_grouped"])
    app.select_slider(key=prefix + "_n").set_value(8)
    app.select_slider(key=prefix + "_walks").set_value(64)
    app.number_input(key=prefix + "_batch_size").set_value(7)
    app.button(key=prefix + "_run").click().run()
    assert not app.exception and not app.error
    experiment = app.session_state[prefix + "_result"]
    assert experiment.config.batch_size == 7
    assert list(experiment.results) == ["mc_grouped"]
    assert experiment.results["mc_grouped"].extras["batch_size"] == 7
    assert experiment.metrics[0]["batch_size"] == 7
    app.run()
    assert not app.exception and not app.error
    assert app.number_input(key=prefix + "_batch_size").value == 7


def study_app():
    from pdelab.experiment3d import Config3D
    from pdelab.ui3d import _research_tab

    _research_tab(Config3D(problem="constant", n=8, batch_size=11))


def test_existing_study_ui_preserves_its_group_size_and_independent_final():
    app = AppTest.from_function(study_app, default_timeout=60).run()
    assert app.number_input(key="study3d_batch_size").value == 11
    app.multiselect(key="study3d_methods").set_value(["mc_grouped"])
    app.number_input(key="study3d_batch_size").set_value(7)
    app.number_input(key="study3d_repeats").set_value(2)
    app.select_slider(key="study3d_walks").set_value(64)
    app.button(key="study3d_run").click().run()
    assert not app.exception and not app.error
    study = app.session_state["study3d_result"]
    assert study.settings.config.batch_size == 7
    assert {row["method"] for row in study.rows} == {"mc_grouped"}
    assert {row["target_pct"] for row in study.rows} == {1, 5, 10}
    assert all(row["seconds"] is not None for row in study.rows)
    assert {trial["phase"] for trial in study.trials} == {"pilot", "final"}
    assert all(trial["batch_size"] == 7 for trial in study.trials)
    assert "study3d_result" in app.session_state
    assert not any("три предварительные" in text.value for text in app.markdown)


def mixed_study_app():
    import streamlit as st

    from pdelab.boundary_experiment3d import BoundaryConfig3D
    from pdelab.boundary_research import render_boundary_research

    limit = st.session_state.get("test_path_limit")
    render_boundary_research(
        BoundaryConfig3D(
            problem="custom",
            n=4,
            kinds=(
                ("dirichlet",) * 6 if limit is None else ("robin",) + ("dirichlet",) * 5
            ),
            data=(7.0,) * 6,
            batch_size=11,
            max_steps=limit,
        ),
        ["mc_grouped"],
        "mixed_study",
    )


@pytest.mark.parametrize("path_limit", [None, 1])
def test_mixed_study_ui_and_downloads_preserve_complete_or_failed_runs(
    monkeypatch, path_limit
):
    downloads = {}
    download_button = st.download_button

    def capture_download(label, data, *args, **kwargs):
        downloads[kwargs["key"]] = data
        return download_button(label, data, *args, **kwargs)

    monkeypatch.setattr(st, "download_button", capture_download)
    app = AppTest.from_function(mixed_study_app, default_timeout=60)
    app.session_state["test_path_limit"] = path_limit
    app.run()
    assert app.number_input(key="mixed_study_batch_size").value == 11
    app.number_input(key="mixed_study_batch_size").set_value(7)
    app.number_input(key="mixed_study_repeats").set_value(2)
    app.multiselect(key="mixed_study_budgets").set_value([64])
    app.button(key="mixed_study_run").click().run()
    assert not app.exception and not app.error
    result = app.session_state["mixed_study"][1]
    assert result["config"]["batch_size"] == 7
    assert len(result["rows"]) == 3
    assert {row["name"] for row in result["rows"]} == {"MC · группы частиц"}
    assert all(not table.value.empty for table in app.dataframe)
    assert "MC · группы частиц" in app.dataframe[0].value.columns
    exported = json.loads(downloads["mixed_study_json"])
    assert exported["config"]["batch_size"] == 7
    assert exported["trials"] == result["trials"]
    assert exported["series"] == result["series"]
    csv_rows = list(csv.DictReader(io.StringIO(downloads["mixed_study_csv"])))
    assert len(csv_rows) == 3
    assert all(row["name"] == "MC · группы частиц" for row in csv_rows)
    if path_limit is None:
        assert {trial["phase"] for trial in exported["trials"]} == {"pilot", "final"}
        assert all(trial["status"] == "completed" for trial in exported["trials"])
        assert all(trial["values"] is not None for trial in exported["trials"])
        assert all(row["seconds"] is not None for row in exported["rows"])
    else:
        assert {trial["phase"] for trial in exported["trials"]} == {"pilot"}
        assert all(trial["values"] is None for trial in exported["trials"])
        assert all(trial["status"] == "path_limit" for trial in exported["trials"])
        assert all(trial["started_walks"] > 0 for trial in exported["trials"])
        assert all(row["seconds"] is None for row in exported["rows"])
        assert all(row["relative_error_pct"] is None for row in exported["rows"])
        assert any(
            "Подбор не завершён" in cell
            for cell in app.dataframe[0].value["MC · группы частиц"]
        )
