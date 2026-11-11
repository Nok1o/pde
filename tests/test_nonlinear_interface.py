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
    assert all(
        row["status"] == "solver_failed" and row["seconds"] is None for row in mc
    )
    assert all(row["walks"] == 8 and "walk_budget" in row["reason"] for row in mc)
    assert evidence["validations"] == []


def test_nonlinear_targets_retry_failed_check_on_fresh_larger_budget(monkeypatch):
    from types import SimpleNamespace

    import pdelab.nonlinear_study as module

    baseline = SimpleNamespace(
        completed=True, seconds=0.001, reason="ok", settings=NonlinearSettings()
    )

    def comparison(problem, dt, final_time, point, repeats, deterministic, candidate):
        fails = candidate.walks == 8 and candidate.seed == module.child_seed(
            42, 300, 0, 1
        )
        runs = [
            SimpleNamespace(
                settings=SimpleNamespace(seed=module.child_seed(candidate.seed, r)),
                completed=True,
                reason="ok",
            )
            for r in range(repeats)
        ]
        return SimpleNamespace(
            deterministic=baseline,
            stochastic=runs,
            metrics=[
                dict(method="picard", relative_error_pct=0.0, completed=True),
                dict(
                    method="mc",
                    relative_error_pct=20.0 if fails else 0.1,
                    seconds=0.01 * candidate.walks,
                    completed=True,
                ),
            ],
        )

    monkeypatch.setattr(module, "compare_nonlinear", comparison)
    monkeypatch.setattr(
        module, "_target_comparison", lambda comparison, *args: comparison
    )
    monkeypatch.setattr(module, "solve_nonlinear", lambda *args: baseline)
    rows, evidence = module.nonlinear_targets(
        make_nonlinear_problem(4), repeats=2, budgets=(8, 32)
    )
    mc_rows = [row for row in rows if row["method"] == "mc"]
    assert all(row["status"] == "confirmed" and row["budget"] == 32 for row in mc_rows)
    assert all(row["validation_attempts"] == 2 for row in mc_rows)
    pilot_seeds = [s for item in evidence["pilots"] for s in item["seeds"]]
    validation_seeds = [s for item in evidence["validations"] for s in item["seeds"]]
    assert len(set(pilot_seeds + validation_seeds)) == 16
    assert len(evidence["validations"]) == 2


def test_nonlinear_initial_failure_preserves_solver_reason(monkeypatch):
    import pdelab.nonlinear_study as module

    def fail(*args):
        raise RuntimeError("reference did not converge")

    monkeypatch.setattr(module, "compare_nonlinear", fail)
    with pytest.raises(RuntimeError, match="reference did not converge"):
        module.nonlinear_targets(make_nonlinear_problem(4), budgets=(8,))


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


def test_fixed_study_rejects_ignored_arguments(tmp_path):
    for option, value in (("--max-total-walks", "1"), ("--n", "8"), ("--dt", ".01")):
        with pytest.raises(SystemExit) as error:
            main(["study", option, value, "--out", str(tmp_path)])
        assert error.value.code == 2
    assert not list(tmp_path.iterdir())


def test_fixed_study_records_actual_method_settings(monkeypatch, tmp_path):
    import pdelab.nonlinear_cli as cli
    import pdelab.nonlinear_study as study

    real_compare = study.compare_nonlinear

    def cheap_compare(problem, dt, T, **kwargs):
        return real_compare(problem, dt, 0, **kwargs)

    monkeypatch.setattr(study, "compare_nonlinear", cheap_compare)
    monkeypatch.setattr(cli, "nonlinear_accuracy_study", lambda: [])
    main(["study", "--repeats", "2", "--out", str(tmp_path)])
    data = json.loads((tmp_path / "settings.json").read_text())
    assert "n" not in data and "mc" not in data
    last = [row for row in data["cases"] if row["case"] == 30]
    lod, mc = last
    assert lod["method"] == "picard" and lod["tolerance"] == 1e-7
    assert lod["max_outer"] == 100 and lod["initial_walks"] is None
    assert mc["max_outer"] == 1 and mc["tolerance"] == 1e-10
    assert mc["max_total_walks"] == 20_000_000


def test_target_metric_changes_without_relabelling_comparison_rmse():
    from types import SimpleNamespace

    from pdelab.nonlinear_study import NonlinearComparison, _target_comparison

    runs = [SimpleNamespace(fields=np.array([[[value]]])) for value in (9.0, 9.0, 13.0)]
    deterministic = SimpleNamespace(fields=np.array([[[10.0]]]))
    comparison = NonlinearComparison(
        point=(0, 0),
        reference=deterministic,
        deterministic=deterministic,
        stochastic=runs,
        metrics=[
            {"method": "picard", "completed": True, "relative_error_pct": 0.0},
            {"method": "mc", "completed": True, "relative_error_pct": 19.148542155},
        ],
        matched_picard=deterministic,
    )
    converted = _target_comparison(comparison, SimpleNamespace(exact=None), 0.0)
    assert converted.metrics[1]["relative_error_pct"] == pytest.approx(100 / 6)
    assert comparison.metrics[1]["relative_error_pct"] == pytest.approx(19.148542155)
    assert (
        converted.metrics[1]["relative_rmse_pct"]
        == comparison.metrics[1]["relative_error_pct"]
    )
