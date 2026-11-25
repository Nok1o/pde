"""The frozen grouped diagnostic is explicit and is not a full heat solution."""

import csv
import json

import pytest

from pdelab.heat3d_mc_cli import main


@pytest.mark.parametrize("method", ["mc", "mc_grouped"])
def test_cli_exports_actual_backend_and_group_setting(tmp_path, method):
    main(
        [
            "--n",
            "4",
            "--method",
            method,
            "--walks",
            "17",
            "--batch-size",
            "7",
            "--direct",
            "--out",
            str(tmp_path),
        ]
    )
    payload = json.loads((tmp_path / "result.json").read_text())
    settings = payload["metadata"]["configuration"]
    result = payload["result"]
    assert result["method"] == settings["method"] == method
    assert (
        result["batch_size"]
        == settings["batch_size"]
        == (7 if method == "mc_grouped" else None)
    )
    assert payload["metadata"]["physical_layers_accepted"] == 0
    assert payload["metadata"]["scenario"] == "diagnostic_frozen_first_layer"
    assert "heat3d_grouped_mc.py" in payload["metadata"]["source_sha256"]
    assert result["status"] == "completed" and result["walks"] == 17
    assert result["completed_paths"] == [17]
    with (tmp_path / "points.csv").open(encoding="utf-8-sig") as stream:
        row = next(csv.DictReader(stream))
    assert float(row["value"]) == result["values"][0]


def test_cli_group_limit_exports_work_but_no_solution(tmp_path):
    with pytest.raises(SystemExit) as exc:
        main(
            [
                "--n",
                "4",
                "--method",
                "mc_grouped",
                "--walks",
                "17",
                "--batch-size",
                "7",
                "--max-walks",
                "3",
                "--out",
                str(tmp_path),
            ]
        )
    assert exc.value.code == 2
    payload = json.loads((tmp_path / "result.json").read_text())
    result = payload["result"]
    assert result["method"] == "mc_grouped" and result["batch_size"] == 7
    assert result["status"] == "walk_budget" and result["walks"] == 3
    assert result["values"] is None and payload["rows"][0]["value"] is None
    assert result["completed_paths"] == [3]
