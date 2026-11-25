"""Mixed heat faces reach comparison, accuracy study and reproducible exports."""

import json
import shlex

import pytest
from streamlit.testing.v1 import AppTest

from pdelab.boundary3d import FACES
from pdelab.heat3d_cli import main
from pdelab.heat3d_experiment import HeatConfig3D, compare_heat3d
from pdelab.heat3d_mc_cli import main as frozen_main


@pytest.mark.parametrize("command", ["run", "study"])
def test_cli_all_kind3_compares_and_studies_same_selected_faces(tmp_path, command):
    args = [
        command,
        "--case",
        "polynomial",
        "--n",
        "4",
        "--alpha",
        "0",
        "--T",
        ".01",
        "--methods",
        "lod",
        "block",
        "--point",
        "0",
        "0",
        "0",
        "--beta",
        "2",
        "--beta-y1",
        "3",
        "--out",
        str(tmp_path),
    ]
    for face in FACES:
        args += ["--" + face, "3"]
    args += (
        ["--newton"]
        if command == "run"
        else ["--repeats", "2", "--budgets", "1e-3", "1e-7"]
    )
    assert main(args) == 0
    payload = json.loads((tmp_path / "result.json").read_text())
    metadata = payload["metadata"]
    assert metadata["boundary_kinds"] == dict.fromkeys(FACES, 3)
    assert metadata["boundary_betas"] == dict(zip(FACES, [2, 2, 2, 3, 2, 2]))
    assert metadata["configuration"]["betas"] == [2, 2, 2, 3, 2, 2]
    assert {row["method"] for row in payload["rows"]} == {"lod", "block"}
    if command == "run":
        assert all(row["status"] == "completed" for row in payload["rows"])
        assert payload["grid_diagnostic"]["status"] == "completed"
    else:
        assert all(row["status"] == "confirmed" for row in payload["rows"])
        assert {row["phase"] for row in payload["trials"]} == {"pilot", "final"}
    replay = shlex.split(metadata["reproduction"])[3:]
    replay[replay.index("--out") + 1] = str(tmp_path / "replay")
    assert main(replay) == 0
    replayed = json.loads((tmp_path / "replay/result.json").read_text())
    assert replayed["metadata"]["configuration"] == metadata["configuration"]


@pytest.mark.parametrize("method", ["mc", "mc_grouped"])
def test_frozen_cli_uses_new_heat_faces_and_keeps_diagnostic_scope(tmp_path, method):
    assert (
        frozen_main(
            [
                "--n",
                "4",
                "--method",
                method,
                "--walks",
                "128",
                "--batch-size",
                "7",
                "--x0",
                "3",
                "--z1",
                "3",
                "--beta-x0",
                "2.5",
                "--point",
                "0",
                "2",
                "2",
                "--direct",
                "--out",
                str(tmp_path),
            ]
        )
        == 0
    )
    payload = json.loads((tmp_path / "result.json").read_text())
    assert payload["metadata"]["configuration"]["kinds"] == [3, 1, 1, 1, 1, 3]
    assert payload["metadata"]["configuration"]["betas"][0] == 2.5
    assert payload["metadata"]["physical_layers_accepted"] == 0
    assert payload["result"]["status"] == "completed"
    assert payload["result"]["walks"] == 128  # The face is now a free query.


def heat_ui():
    from pdelab.heat3d_ui import render_heat3d_app

    render_heat3d_app()


def test_ui_boundary_controls_survive_run_study_and_parameter_change():
    app = AppTest.from_function(heat_ui, default_timeout=60).run()
    app.select_slider(key="h3_n").set_value(4)
    app.selectbox(key="h3_case").set_value("polynomial")
    app.number_input(key="h3_alpha").set_value(0.0)
    app.number_input(key="h3_T").set_value(0.01)
    for face in FACES:
        app.selectbox(key="h3_kind_" + face).set_value(3)
    app.run()
    app.number_input(key="h3_beta_x0").set_value(2.5)
    app.number_input(key="h3_x").set_value(0)
    app.button(key="h3_run").click().run()
    assert not app.exception and not app.error
    saved = app.session_state["h3_comparison"]
    assert saved["metadata"]["boundary_betas"]["x0"] == 2.5
    assert saved["rows"][0]["status"] == "completed"
    app.number_input(key="h3_repeats").set_value(2)
    app.multiselect(key="h3_budgets").set_value([1e-3, 1e-7])
    app.button(key="h3_study_run").click().run()
    assert not app.exception and not app.error
    study = app.session_state["h3_study"]
    assert app.session_state["h3_sections"] == "Исследование 1/5/10%"
    assert study["metadata"]["configuration"]["kinds"] == [3] * 6
    assert all(row["status"] == "confirmed" for row in study["rows"])
    app.number_input(key="h3_beta_x0").set_value(3.0).run()
    assert not app.exception
    assert app.session_state["h3_study"]["metadata"]["boundary_betas"]["x0"] == 2.5
    assert any("изменены" in entry.value for entry in app.info)


@pytest.mark.parametrize(
    "config",
    [
        HeatConfig3D(kinds=(1, 3)),
        HeatConfig3D(kinds=(2,) * 6),
        HeatConfig3D(betas=(1, 1)),
        HeatConfig3D(betas=(0,) * 6),
    ],
)
def test_invalid_boundary_configuration_is_rejected(config):
    with pytest.raises(ValueError):
        compare_heat3d(config)
