"""Material parameters survive studies, independent checks and frozen exports."""

import json

import pytest

from pdelab.heat3d_cli import main
from pdelab.heat3d_experiment import METHOD_NAMES, HeatConfig3D, study_heat3d
from pdelab.heat3d_mc_cli import main as frozen_main
from pdelab.heat3d_solver import HeatSettings3D


def test_all_material_methods_enter_independent_accuracy_study():
    config = HeatConfig3D(
        case="polynomial",
        n=5,
        dt=0.002,
        T=0.003,
        capacity_gamma=0.4,
        reaction_sigma=0.25,
        kinds=(3, 1, 1, 1, 1, 1),
        methods=tuple(METHOD_NAMES),
    )
    settings = HeatSettings3D(
        history_mode="layers",
        conductivity_refresh_layers=4,
        mc_tolerance=0.2,
        mc_change_tolerance=0.1,
    )
    result = study_heat3d(
        config, settings, repeats=2, budgets=[0.001], walk_budgets=[1024]
    )
    assert result["metadata"]["configuration"]["capacity_gamma"] == 0.4
    assert result["metadata"]["configuration"]["reaction_sigma"] == 0.25
    assert len(result["rows"]) == 3 * len(METHOD_NAMES)
    for method in METHOD_NAMES:
        trials = [row for row in result["trials"] if row["method"] == method]
        assert {row["phase"] for row in trials} == {"pilot", "final"}
        if method == "mass_transport":
            assert all(row["seed"] is None for row in trials)
        else:
            assert len({row["seed"] for row in trials}) == len(trials)
        for row in trials:
            assert row["status"] == "completed"
            assert row["completed_until"] == config.T
            assert row["relative_error_pct"] >= 0
        rows = [row for row in result["rows"] if row["method"] == method]
        assert {row["target_pct"] for row in rows} == {1, 5, 10}
        # A missed target is evidence, not a solver failure or a hidden retry.
        assert all(
            row["status"] in {"confirmed", "not_reached", "validation_failed"}
            for row in rows
        )
    for collection in ("rows", "series"):
        for row in result[collection]:
            assert row["history_mode"] == "layers"
            assert row["conductivity_refresh_layers"] == 4
            matching = [
                trial
                for trial in result["trials"]
                if (trial["method"], trial["phase"], trial["candidate"])
                == (row["method"], row["phase"], row["candidate"])
            ]
            for key in (
                "conductivity_reuses",
                "conductivity_fallbacks",
                "history_records_discarded",
            ):
                assert row[key] == sum(trial[key] for trial in matching)
            assert row["conductivity_reuses"] == len(matching)


@pytest.mark.parametrize("method", ["mc", "mc_control_grouped"])
def test_frozen_diagnostic_saves_the_material_family(tmp_path, method):
    assert (
        frozen_main(
            [
                "--n",
                "5",
                "--method",
                method,
                "--dt",
                ".002",
                "--walks",
                "128",
                "--capacity-gamma",
                ".4",
                "--reaction-sigma",
                ".25",
                "--x0",
                "3",
                "--direct",
                "--out",
                str(tmp_path),
            ]
        )
        == 0
    )
    result = json.loads((tmp_path / "result.json").read_text())
    config = result["metadata"]["configuration"]
    assert config["capacity_gamma"] == 0.4
    assert config["reaction_sigma"] == 0.25
    assert result["result"]["status"] == "completed"


@pytest.mark.parametrize("argument", ["--capacity-gamma", "--reaction-sigma"])
def test_cli_refuses_unsupported_material_without_creating_result(tmp_path, argument):
    out = tmp_path / "invalid"
    with pytest.raises(SystemExit) as error:
        main(["run", argument, "-0.1", "--out", str(out)])
    assert error.value.code == 2
    assert not out.exists()
