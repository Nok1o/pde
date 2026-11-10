import csv
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from pdelab.boundary_cli import main
from pdelab.boundary_experiment import (
    BoundaryConfig,
    benchmark_boundary,
    run_boundary_experiment,
)
from pdelab.boundary_experiment3d import BoundaryConfig3D, benchmark_boundary3d
from pdelab.boundary_ui import workload_message


@pytest.mark.parametrize(
    "config,benchmark",
    [
        (BoundaryConfig(n=4), benchmark_boundary),
        (BoundaryConfig3D(n=4), benchmark_boundary3d),
    ],
)
@pytest.mark.parametrize("methods", [["mc"], ["lod_serial", "mc"]])
def test_mc_series_requires_a_budget(config, benchmark, methods):
    with pytest.raises(ValueError, match="бюджет"):
        benchmark(config, budgets=[], methods=methods, repeats=2)


@pytest.mark.parametrize(
    "config,benchmark",
    [
        (BoundaryConfig(n=4), benchmark_boundary),
        (BoundaryConfig3D(n=4), benchmark_boundary3d),
    ],
)
def test_lod_only_series_does_not_require_mc_budgets(config, benchmark):
    rows = benchmark(
        config, budgets=[], methods=["lod_serial"], tolerances=[1e-4], repeats=2
    )
    assert len(rows) == 2 and all(row["method"] == "lod_serial" for row in rows)
    with pytest.raises(ValueError, match="допуск"):
        benchmark(config, methods=["lod_serial"], tolerances=[], repeats=2)


@pytest.mark.parametrize("dimension,prefix", [("2D", "bc"), ("3D", "bc3d")])
def test_ui_disables_series_with_mc_and_no_budgets(dimension, prefix):
    app = AppTest.from_file(
        Path(__file__).parents[1] / "app.py", default_timeout=60
    ).run()
    app.radio(key="dimension").set_value(dimension).run()
    mode_key = "boundary_mode" if dimension == "2D" else "cube_boundary_mode"
    app.radio(key=mode_key).set_value("Смешанные I / II / III").run()
    app.select_slider(key=f"{prefix}_n").set_value(8).run()
    app.multiselect(key=f"{prefix}_methods").set_value(["mc"])
    app.multiselect(key=f"{prefix}_budgets").set_value([]).run()
    assert app.button(key=f"{prefix}_benchmark").disabled
    assert not app.exception
    # A session that hit the old bug may still contain an empty saved series.
    result_key = "bc_benchmark_result" if dimension == "2D" else "bc3d_series"
    app.session_state[result_key] = ({}, [])
    app.run()
    assert not app.exception
    app.multiselect(key=f"{prefix}_methods").set_value(["lod_serial", "mc"]).run()
    assert app.button(key=f"{prefix}_benchmark").disabled
    app.multiselect(key=f"{prefix}_methods").set_value(["lod_serial"]).run()
    assert not app.button(key=f"{prefix}_benchmark").disabled
    app.button(key=f"{prefix}_benchmark").click().run()
    assert not app.exception and not app.error


def test_cli_exports_boundary_coefficients_and_reference(tmp_path):
    main(
        [
            "run",
            "--n",
            "8",
            "--left",
            "neumann",
            "--walks",
            "64",
            "--out",
            str(tmp_path),
        ]
    )
    payload = json.loads((tmp_path / "experiment.json").read_text())
    assert payload["boundary"]["left"]["kind"] == "neumann"
    assert payload["reference_kind"] == "analytic"
    assert len(payload["metrics"]) == 3
    fields = np.load(tmp_path / "fields.npz")
    assert fields["points_yx"].tolist() == [[4, 4]]
    assert (tmp_path / "field.html").is_file()


def test_benchmark_seeds_and_neumann_default():
    rows = benchmark_boundary(
        BoundaryConfig(n=8), budgets=(32, 64), repeats=2, tolerances=(1e-4,)
    )
    assert len(rows) == 10 and len({r["seed"] for r in rows}) == 2
    assert all(r["converged"] for r in rows)
    pure = run_boundary_experiment(BoundaryConfig(n=8, kinds=("neumann",) * 4))
    assert list(pure.results) == ["lod_serial"]


@pytest.mark.parametrize(
    "method,limits", [("lod_serial", ["--max-steps", "1"]), ("mc", ["--max-iter", "1"])]
)
def test_cli_benchmark_honors_method_selection(tmp_path, method, limits):
    main(
        [
            "benchmark",
            "--n",
            "8",
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
    assert rows and {r["method"] for r in rows} == {method}
    assert json.loads((tmp_path / "settings.json").read_text())["methods"] == [method]


@pytest.mark.parametrize("methods", [[], ["unknown"], ["mc", "mc"]])
def test_benchmark_rejects_invalid_methods(methods):
    with pytest.raises(ValueError, match="Выберите"):
        benchmark_boundary(BoundaryConfig(n=8), methods=methods)


def test_ui_workload_limit():
    config = BoundaryConfig(n=64, mode="field", walks=16384)
    assert workload_message(config, ["mc"])
    assert workload_message(config, ["lod_serial"]) is None
    assert (
        workload_message(replace(config, n=8, mode="point", walks=256), ["mc"]) is None
    )


def test_boundary_ui_calculation_and_boundary_start():
    app = AppTest.from_file(
        Path(__file__).parents[1] / "app.py", default_timeout=60
    ).run()
    app.radio(key="boundary_mode").set_value("Смешанные I / II / III").run()
    assert not app.exception
    app.select_slider(key="bc_n").set_value(8).run()
    app.select_slider(key="bc_walks").set_value(64)
    app.number_input(key="bc_x").set_value(0)
    app.number_input(key="bc_y").set_value(0)
    app.button(key="bc_run").click().run()
    assert not app.exception and not app.error
    assert app.session_state["bc_result"].points.tolist() == [[0, 0]]
    assert len(app.latex) >= 6
    for side in ("left", "right", "bottom", "top"):
        app.selectbox(key=f"bc_{side}_kind").set_value("neumann")
    app.run()
    assert not app.exception
    assert app.multiselect(key="bc_methods").options == ["ЛОМ"]
    app.button(key="bc_run").click().run()
    assert not app.exception and not app.error
