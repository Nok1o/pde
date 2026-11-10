import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from pdelab.cli3d import main
from pdelab.experiment3d import Config3D, export3d, run_experiment3d
from pdelab.ui3d import _archive
from pdelab.views3d import extract_slice, slice_figure3d


def test_point_export_preserves_dimension_unknown_nodes_and_exact_reference(tmp_path):
    experiment = run_experiment3d(Config3D(n=8, point=(2, 3, 5), walks=64))
    assert experiment.reference_kind == "analytic"
    assert experiment.points.tolist() == [[5, 3, 2]]
    assert experiment.metrics[0]["rmse"] > experiment.metrics[0]["rmse_grid"]
    export3d(experiment, tmp_path)
    payload = json.loads((tmp_path / "experiment.json").read_text())
    assert payload["metadata"]["dimension"] == 3
    assert payload["metadata"]["array_order"] == ["z", "y", "x"]
    arrays = np.load(tmp_path / "fields.npz")
    assert arrays["mc"].shape == (9, 9, 9)
    assert np.isnan(arrays["mc"][1, 1, 1])
    assert np.isfinite(arrays["mc"][5, 3, 2])
    assert "plotly.js" in (tmp_path / "slices.html").read_text()
    archive = zipfile.ZipFile(io.BytesIO(_archive(experiment, "xz", 3)))
    assert {
        "experiment.json",
        "fields.npz",
        "metrics.csv",
        "solution.html",
        "error.html",
    } == set(archive.namelist())


@pytest.mark.parametrize(
    "plane,index,expected",
    [("xy", 3, (3, 2, 1)), ("xz", 3, (2, 3, 1)), ("yz", 3, (2, 1, 3))],
)
def test_slices_follow_named_coordinate_order(plane, index, expected):
    z, y, x = np.indices((5, 5, 5))
    volume = 100 * z + 10 * y + x
    assert extract_slice(volume, plane, index)[2, 1] == volume[expected]


def test_plot_does_not_interpolate_single_mc_point():
    experiment = run_experiment3d(Config3D(n=8, walks=32))
    figure = slice_figure3d(experiment, "xy", 4)
    heatmaps = [trace for trace in figure.data if trace.type == "heatmap"]
    mc = np.asarray(heatmaps[-1].z)
    assert np.isnan(mc[1, 1]) and np.isfinite(mc[4, 4])
    assert np.isfinite(mc[1:-1, 1:-1]).sum() == 1


def test_large_grid_uses_analytic_reference_without_lu_and_cli_runs(tmp_path):
    experiment = run_experiment3d(Config3D(n=24, walks=8), methods=["mc"])
    assert experiment.grid_reference is None and experiment.reference_kind == "analytic"
    main(["run", "--n", "8", "--walks", "16", "--out", str(tmp_path)])
    assert (tmp_path / "fields.npz").exists()


def test_cube_ui_runs_queries_switches_planes_and_invalidates_export():
    app = AppTest.from_file(
        str(Path(__file__).parents[1] / "app.py"), default_timeout=60
    ).run()
    app.radio(key="dimension").set_value("3D").run()
    assert not app.exception
    assert app.multiselect(key="cube_methods").value == [
        "lod_serial",
        "mc",
        "graded",
        "wos",
    ]
    app.select_slider(key="cube_n").set_value(8)
    app.select_slider(key="cube_walks").set_value(64)
    app.run()
    app.number_input(key="cube_point_x").set_value(2)
    app.number_input(key="cube_point_y").set_value(3)
    app.number_input(key="cube_point_z").set_value(5)
    app.button(key="cube_run").click().run()
    assert not app.exception
    assert app.session_state["cube_result"].points.tolist() == [[5, 3, 2]]
    for plane in ("xz", "yz", "xy"):
        app.selectbox(key="cube_view_plane").set_value(plane).run()
        assert not app.exception
    app.button(key="cube_export").click().run()
    assert "cube_archive" in app.session_state
    app.number_input(key="cube_point_x").set_value(0)
    app.number_input(key="cube_point_y").set_value(0)
    app.number_input(key="cube_point_z").set_value(0)
    app.button(key="cube_run").click().run()
    assert not app.exception
    assert app.session_state["cube_result"].results["mc"].walks == 0
    assert "cube_archive" not in app.session_state
    app.radio(key="dimension").set_value("2D").run()
    assert not app.exception


def test_cube_ui_study_runs_without_main_calculation():
    app = AppTest.from_file(
        str(Path(__file__).parents[1] / "app.py"), default_timeout=60
    ).run()
    app.radio(key="dimension").set_value("3D").run()
    app.select_slider(key="study3d_grid").set_value(8)
    app.number_input(key="study3d_repeats").set_value(2)
    app.select_slider(key="study3d_walks").set_value(64)
    app.run()
    app.button(key="study3d_run").click().run()
    assert not app.exception
    assert "cube_result" not in app.session_state
    assert app.dataframe[0].value.shape == (3, 5)


def test_cli_study_respects_selected_method(tmp_path):
    main(
        [
            "study",
            "--problem",
            "constant",
            "--n",
            "4",
            "--methods",
            "mc",
            "--repeats",
            "2",
            "--max-walks",
            "2",
            "--out",
            str(tmp_path),
        ]
    )
    payload = json.loads((tmp_path / "settings.json").read_text())
    assert payload["settings"]["methods"] == ["mc"]
    import csv

    with (tmp_path / "trials.csv").open(encoding="utf-8-sig") as stream:
        trials = list(csv.DictReader(stream))
    assert trials and {trial["method"] for trial in trials} == {"mc"}
