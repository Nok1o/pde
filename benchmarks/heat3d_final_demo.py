"""Exercise the real Streamlit workflow and replay its exported CLI commands.

This is functional acceptance, not a timing benchmark. Every UI/CLI artifact is
kept; no failed numerical run is discarded or retried to obtain a passing value.
"""

import argparse
import io
import json
import os
import shlex
import subprocess
import sys
import zipfile
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("lod", "block", "mc", "mc_control", "mc_grouped", "mc_control_grouped")
MATCH = (
    "method",
    "status",
    "phase",
    "candidate",
    "budget",
    "repeat",
    "seed",
    "target_time",
    "completed_until",
    "physical_steps",
    "outer_iterations",
    "inner_iterations",
    "walks",
    "total_walks",
    "completed_walks",
    "transitions",
    "validation_walks",
    "values",
    "relative_error_pct",
    "absolute_error",
    "history_mode",
    "conductivity_refresh_layers",
    "conductivity_reuses",
    "conductivity_fallbacks",
    "history_records_discarded",
    "batch_size",
    "coarse_n",
    "requested_repeats",
    "completed_repeats",
)


def assert_records(left, right, keys=MATCH):
    assert len(left) == len(right)
    for a, b in zip(left, right, strict=True):
        for key in keys:
            assert a.get(key) == b.get(key), (
                a.get("method"),
                key,
                a.get(key),
                b.get(key),
            )


def demonstrate(out):
    import numpy as np
    import streamlit as st
    from streamlit.testing.v1 import AppTest

    from pdelab.heat3d_experiment import export_result, json_bytes

    out.mkdir(parents=True, exist_ok=False)
    downloads, observations = {}, []
    download = st.download_button

    def capture(label, data, *args, **kwargs):
        downloads[kwargs.get("key")] = data
        return download(label, data, *args, **kwargs)

    def retain(app, name, study=False):
        assert not app.exception and not app.error
        key = "h3_study" if study else "h3_comparison"
        result = app.session_state[key]
        ui = out / name / "ui"
        export_result(result, ui)
        downloaded = json.loads(downloads[key + "_json"])
        saved = json.loads((ui / "result.json").read_bytes())
        assert downloaded == saved
        with zipfile.ZipFile(io.BytesIO(downloads[key + "_zip"])) as archive:
            assert json.loads(archive.read("result.json")) == saved
            assert archive.read("table.csv") == downloads[key + "_csv"]
        cli = out / name / "cli"
        command = shlex.split(result["metadata"]["reproduction"])
        command[0] = sys.executable
        command[command.index("--out") + 1] = str(cli.resolve())
        with (out / name / "cli.log").open("w") as log:
            replay = subprocess.run(
                command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=False
            )
        assert replay.returncode in (0, 2)
        repeated = json.loads((cli / "result.json").read_bytes())
        for field in ("configuration", "solver_settings", "query_size", "target_time"):
            assert saved["metadata"][field] == repeated["metadata"][field], field
        assert_records(saved["rows"], repeated["rows"])
        if study:
            assert_records(saved["trials"], repeated["trials"])
        else:
            with (
                np.load(ui / "query-values.npz") as a,
                np.load(cli / "query-values.npz") as b,
            ):
                assert set(a.files) == set(b.files)
                for field in a.files:
                    np.testing.assert_array_equal(a[field], b[field])
        observations.append(
            dict(
                scenario=name,
                cli_exit=replay.returncode,
                configuration=saved["metadata"]["configuration"],
                solver_settings=saved["metadata"]["solver_settings"],
                statuses=[
                    dict(method=r["method"], status=r["status"], reason=r.get("reason"))
                    for r in saved["rows"]
                ],
                ui_cli_values_and_counters_match=True,
                csv_json_zip_match=True,
            )
        )
        (out / "acceptance.json").write_bytes(
            json_bytes(
                dict(
                    purpose="functional UI/CLI/export acceptance; not timing evidence",
                    source_commit=result["metadata"]["commit"],
                    python=sys.version,
                    scenarios=observations,
                )
            )
        )
        print(name + ": UI/CLI/export agree", flush=True)
        return saved

    with patch.object(st, "download_button", capture):
        app = AppTest.from_file(ROOT / "app.py", default_timeout=120).run()
        app.radio(key="evolution").set_value("Квазилинейная теплопроводность").run()
        app.radio(key="nonlinear_dimension").set_value("3D").run()
        app.multiselect(key="h3_methods").set_value(list(METHODS)).run()
        for key, value in {
            "h3_capacity_gamma": 0.5,
            "h3_reaction_sigma": 1.0,
            "h3_dt": 0.002,
            "h3_T": 0.004,
            "h3_walks": 1024,
            "h3_mc_seed": 31037,
            "h3_batch_size": 128,
            "h3_coarse_n": 4,
            "h3_outer": 8,
            "h3_seconds": 5.0,
            "h3_max_walks": 12000000,
            "h3_max_moves": 40000000,
            "h3_max_path_steps": 100000,
        }.items():
            app.number_input(key=key).set_value(value)
        app.select_slider(key="h3_tol").set_value(1e-6)
        app.select_slider(key="h3_change").set_value(1e-6)
        app.checkbox(key="h3_newton").set_value(True)
        app.button(key="h3_run").click().run()
        before = retain(app, "sine-run")
        app.number_input(key="h3_capacity_gamma").set_value(0.7).run()
        assert (
            app.session_state["h3_comparison"]["metadata"]["configuration"][
                "capacity_gamma"
            ]
            == 0.5
        )
        assert any("Настройки изменены" in item.value for item in app.info)
        app.number_input(key="h3_capacity_gamma").set_value(0.5).run()
        assert json.loads(downloads["h3_comparison_json"]) == before
        assert any(expander.key == "h3_explanation" for expander in app.expander)
        app.number_input(key="h3_seed").set_value(31038)
        app.multiselect(key="h3_budgets").set_value([1e-3, 1e-6])
        app.multiselect(key="h3_walk_budgets").set_value([256, 1024])
        app.button(key="h3_study_run").click().run()
        retain(app, "sine-study", study=True)
        app.selectbox(key="h3_case").set_value("polynomial")
        for axis, value in zip("xyz", (2, 4, 6), strict=True):
            app.number_input(key="h3_" + axis).set_value(value)
        app.selectbox(key="h3_kind_x0").set_value(3)
        app.button(key="h3_run").click().run()
        retain(app, "polynomial-one-kind3")
        for face in ("x0", "x1", "y0", "y1", "z0", "z1"):
            app.selectbox(key="h3_kind_" + face).set_value(3)
        app.button(key="h3_run").click().run()
        full = retain(app, "polynomial-all-kind3-full")
        app.selectbox(key="h3_history_mode").set_value("layers")
        app.button(key="h3_run").click().run()
        layers = retain(app, "polynomial-all-kind3-layers")
        assert_records(
            full["rows"],
            layers["rows"],
            tuple(
                k
                for k in MATCH
                if k not in ("history_mode", "history_records_discarded")
            ),
        )
        app.number_input(key="h3_conductivity_refresh_layers").set_value(4)
        app.button(key="h3_run").click().run()
        retain(app, "polynomial-refresh4")
        app.multiselect(key="h3_methods").set_value(["lod", "mc_control"]).run()
        for mode in ("slice", "field"):
            app.selectbox(key="h3_mode").set_value(mode).run()
            app.button(key="h3_run").click().run()
            retain(app, "polynomial-" + mode)
        app.multiselect(key="h3_methods").set_value(list(METHODS)).run()
        app.number_input(key="h3_outer").set_value(1)
        app.number_input(key="h3_max_walks").set_value(1)
        app.button(key="h3_run").click().run()
        limited = retain(app, "limited-run")
        assert any(r["status"] != "completed" for r in limited["rows"])
        app.button(key="h3_study_run").click().run()
        limited = retain(app, "limited-study", study=True)
        assert all(
            r["status"] != "confirmed"
            for r in limited["rows"]
            if r["method"].startswith("mc")
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMBA_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = "1"
    demonstrate(args.out)


if __name__ == "__main__":
    main()
