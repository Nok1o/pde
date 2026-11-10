import json
from dataclasses import replace
from pathlib import Path

import numpy as np
from streamlit.testing.v1 import AppTest

from pdelab.boundary_cli import main
from pdelab.boundary_experiment import (
    BoundaryConfig,
    benchmark_boundary,
    run_boundary_experiment,
)
from pdelab.boundary_ui import workload_message


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
    assert len(payload["metrics"]) == 2
    fields = np.load(tmp_path / "fields.npz")
    assert fields["points_yx"].tolist() == [[4, 4]]
    assert (tmp_path / "field.html").is_file()


def test_benchmark_seeds_and_neumann_default():
    rows = benchmark_boundary(
        BoundaryConfig(n=8), budgets=(32, 64), repeats=2, tolerances=(1e-4,)
    )
    assert len(rows) == 6 and len({r["seed"] for r in rows}) == 2
    assert all(r["converged"] for r in rows)
    pure = run_boundary_experiment(BoundaryConfig(n=8, kinds=("neumann",) * 4))
    assert list(pure.results) == ["lod_serial"]


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
