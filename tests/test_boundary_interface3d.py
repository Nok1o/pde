import csv
import io
import json
import zipfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from pdelab.boundary import BoundaryCondition
from pdelab.boundary3d import FACES, BoundaryOperator3D, BoundaryProblem3D
from pdelab.boundary_cli3d import main
from pdelab.boundary_experiment3d import (
    BoundaryConfig3D,
    benchmark_boundary3d,
    boundary_archive3d,
    run_boundary_experiment3d,
)
from pdelab.boundary_workload3d import boundary_limit_message3d, boundary_workload3d


def test_cli_export_six_faces_coordinate_order_and_zip(tmp_path):
    main(
        [
            "run",
            "--n",
            "6",
            "--x0",
            "neumann",
            "--z1",
            "robin",
            "--point",
            "0",
            "2",
            "3",
            "--walks",
            "64",
            "--out",
            str(tmp_path),
        ]
    )
    payload = json.loads((tmp_path / "experiment.json").read_text())
    assert payload["dimension"] == 3 and payload["array_order"] == "zyx"
    assert payload["face_order"] == list(FACES)
    assert payload["boundary"]["x0"]["kind"] == "neumann"
    assert payload["boundary"]["z1"]["kind"] == "robin"
    assert np.shape(payload["boundary"]["z1"]["value"]) == (7, 7)
    assert payload["reference_kind"] == "analytic"
    with np.load(tmp_path / "fields.npz", allow_pickle=False) as arrays:
        np.testing.assert_array_equal(arrays["points_zyx"], [[3, 2, 0]])
    assert (tmp_path / "solution.html").is_file() and (
        tmp_path / "error.html"
    ).is_file()
    experiment = run_boundary_experiment3d(BoundaryConfig3D(n=4), ["lod_serial"])
    with zipfile.ZipFile(io.BytesIO(boundary_archive3d(experiment))) as archive:
        assert set(archive.namelist()) == {
            "experiment.json",
            "metrics.csv",
            "fields.npz",
            "solution.html",
            "error.html",
        }


def test_large_analytic_grid_avoids_sparse_lu_and_absent_reference_is_not_pickled(
    monkeypatch,
):
    def forbidden(*args):
        raise AssertionError("Sparse assembly must not run above n=16")

    monkeypatch.setattr(BoundaryOperator3D, "assemble", forbidden)
    experiment = run_boundary_experiment3d(
        BoundaryConfig3D(n=24, point=(0, 0, 0), walks=2), ["mc"]
    )
    assert (
        experiment.grid_reference is None and experiment.metrics[0]["rmse_grid"] is None
    )
    with zipfile.ZipFile(io.BytesIO(boundary_archive3d(experiment))) as archive:
        with np.load(
            io.BytesIO(archive.read("fields.npz")), allow_pickle=False
        ) as arrays:
            assert "grid_reference" not in arrays
            assert arrays["reference"].shape == (25, 25, 25)
    with pytest.raises(ValueError, match="прямой эталон"):
        run_boundary_experiment3d(BoundaryConfig3D(problem="custom", n=24))


@pytest.mark.parametrize(
    "method,limits", [("lod_serial", ["--max-steps", "1"]), ("mc", ["--max-iter", "1"])]
)
def test_cli_series_honors_selected_methods(tmp_path, method, limits):
    main(
        [
            "benchmark",
            "--n",
            "4",
            "--methods",
            method,
            "--budgets",
            "8",
            "--repeats",
            "2",
            "--out",
            str(tmp_path),
            *limits,
        ]
    )
    with (tmp_path / "benchmark.csv").open(encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    assert rows and {row["method"] for row in rows} == {method}
    assert json.loads((tmp_path / "settings.json").read_text())["methods"] == [method]


def test_pure_neumann_default_and_selected_methods_validation():
    config = BoundaryConfig3D(n=4, kinds=("neumann",) * 6)
    assert list(run_boundary_experiment3d(config).results) == ["lod_serial"]
    for methods in ([], ["unknown"], ["mc", "mc"]):
        with pytest.raises(ValueError, match="Выберите"):
            benchmark_boundary3d(config, methods=methods)
    with pytest.raises(ValueError, match="чистого Неймана"):
        benchmark_boundary3d(config, methods=["mc"])


@pytest.mark.parametrize("kind", ["dirichlet", "robin"])
@pytest.mark.parametrize("face", FACES)
def test_workload_matches_exact_expected_lifetime_for_one_absorbing_face(face, kind):
    n = 4
    kinds = dict.fromkeys(FACES, "neumann")
    kinds[face] = kind
    kappa = 0.5
    faces = {
        s: BoundaryCondition(kinds[s], 0.0, kappa if kinds[s] == "robin" else 0.0)
        for s in FACES
    }
    # A*T=diagonal is the Markov lifetime equation T=1+P*T on unknown nodes.
    operator = BoundaryOperator3D(BoundaryProblem3D(n, 0, faces))
    time_problem = BoundaryProblem3D(n, -operator.diagonal, faces)
    lifetime = BoundaryOperator3D(time_problem).solve()
    config = BoundaryConfig3D(
        n=n, kinds=tuple(kinds.values()), kappas=(kappa,) * 6, walks=2
    )
    paths, work = boundary_workload3d(config, ["mc"])
    assert paths == 2 and work == pytest.approx(2 * lifetime[2, 2, 2], rel=1e-12)


def test_workload_handles_reflecting_start_zero_cost_dirichlet_and_weak_robin():
    config = BoundaryConfig3D(n=64, mode="field", walks=16384)
    assert boundary_limit_message3d(config, ["mc"])
    assert boundary_limit_message3d(config, ["lod_serial"]) is None
    config = replace(config, mode="point", point=(0, 0, 0))
    assert boundary_workload3d(config, ["mc"]) == (0, 0)
    config = replace(config, n=8, walks=64, kinds=("neumann",) * 5 + ("dirichlet",))
    assert boundary_workload3d(config, ["mc"])[1] > 0
    config = replace(config, kinds=("robin",) * 6, kappas=(1e-8,) * 6)
    assert boundary_limit_message3d(config, ["mc"])


def _mixed_app():
    app = AppTest.from_file(
        Path(__file__).parents[1] / "app.py", default_timeout=60
    ).run()
    app.radio(key="dimension").set_value("3D").run()
    app.radio(key="cube_boundary_mode").set_value("Смешанные I / II / III").run()
    app.select_slider(key="bc3d_n").set_value(8).run()
    app.select_slider(key="bc3d_walks").set_value(64)
    return app


def test_ui_six_faces_unknown_corner_slice_trace_and_export():
    app = _mixed_app()
    for face in ("x0", "y0", "z0"):
        app.selectbox(key=f"bc3d_{face}_kind").set_value("neumann")
    for axis in "xyz":
        app.number_input(key=f"bc3d_{axis}").set_value(0)
    app.button(key="bc3d_run").click().run()
    assert not app.exception and not app.error
    experiment = app.session_state["bc3d_result"]
    assert experiment.points.tolist() == [[0, 0, 0]]
    assert experiment.results["mc"].sample_count[0, 0, 0] == 64
    assert len(app.latex) >= 8
    app.button(key="bc3d_trace").click().run()
    assert not app.exception and not app.error
    assert app.session_state["bc3d_trace_result"][1][-1]["event"] == "dirichlet"
    app.button(key="bc3d_export").click().run()
    assert app.session_state["bc3d_archive"][2]
    app.selectbox(key="bc3d_mode").set_value("slice").run()
    app.selectbox(key="bc3d_plane").set_value("yz")
    app.number_input(key="bc3d_index").set_value(0)
    app.button(key="bc3d_run").click().run()
    assert not app.exception and not app.error
    assert len(app.session_state["bc3d_result"].points) == 81
    assert np.all(app.session_state["bc3d_result"].points[:, 2] == 0)


def test_ui_pure_neumann_and_lod_only_series_do_not_run_mc():
    app = _mixed_app()
    app.multiselect(key="bc3d_methods").set_value(["lod_serial"])
    app.number_input(key="bc3d_max_steps").set_value(1)
    app.button(key="bc3d_benchmark").click().run()
    assert not app.exception and not app.error
    assert {row["method"] for row in app.session_state["bc3d_series"][1]} == {
        "lod_serial"
    }
    for face in FACES:
        app.selectbox(key=f"bc3d_{face}_kind").set_value("neumann")
    app.run()
    assert app.multiselect(key="bc3d_methods").options == ["ЛОМ"]
    app.button(key="bc3d_run").click().run()
    assert not app.exception and not app.error
    assert list(app.session_state["bc3d_result"].results) == ["lod_serial"]
