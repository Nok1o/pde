"""Report regressions use synthetic observations, never new timing experiments."""

import json
from dataclasses import asdict

import pytest

from pdelab.boundary_audit3d import (
    AUDIT_SCHEMA_VERSION,
    canonical_series_files,
    json_write,
)
from pdelab.boundary_audit_report import render
from pdelab.measurement_protocol import Trial


@pytest.mark.parametrize(
    "status", ["confirmed", "not_reached", "incomplete", "not_selected"]
)
def test_report_uses_actual_configuration_and_control_status(tmp_path, status):
    completed = status in {"confirmed", "not_reached"}
    summary = dict(
        case="polynomial-all_robin",
        method="mc",
        target_pct=1,
        status=status,
        reason="synthetic reason",
        budget=64,
        mean_seconds=0.1 if completed else None,
        seconds_std=0.01 if completed else None,
        relative_error_pct=(
            (0.5 if status == "confirmed" else 13.2) if completed else None
        ),
    )
    protocol = dict(
        n=32,
        repeats=2,
        query_xyz=[[0.25, 0.5, 0.75]],
        root_seed=2,
        dirty=False,
        reproduction="synthetic command",
    )
    attempts = [
        dict(
            schema_version=AUDIT_SCHEMA_VERSION,
            case=summary["case"],
            method="mc",
            budget=64,
            summary=dict(summary, phase="control"),
            trials=[],
            failures=[],
        )
    ]
    json_write(
        tmp_path / "results.json", dict(protocol=protocol, summary=[summary], ratios=[])
    )
    json_write(tmp_path / "attempts.json", attempts)
    render(tmp_path)
    text = (tmp_path / "README.md").read_text(encoding="utf-8")
    assert "n=32, R=2" in text and "[[0.25, 0.5, 0.75]]" in text
    assert "Функции #24: polynomial;" in text
    assert "R=5" not in text and "Две функции" not in text
    assert "нецентральная точка ещё не измерены" not in text
    assert f"{status}: 1" in text
    if status == "confirmed":
        assert "Все строки целей подтверждены" in text
    else:
        assert "Все строки целей подтверждены" not in text
        assert "Есть неподтверждённые цели" in text and "synthetic reason" in text


def test_audit_adapter_requires_own_schema_and_preserves_known_trials():
    trial = Trial(
        0, 42, "completed", "", [1], 0.1, 0.01, 1024, None, None, 0, 0, 0, 0, 3, None
    )
    attempt = dict(
        schema_version=AUDIT_SCHEMA_VERSION,
        method="lod_serial",
        case="polynomial-all_robin",
        budget=0.01,
        budget_index=0,
        summary=dict(phase="control"),
        elapsed_seconds=0.1,
        trials=[asdict(trial)],
        failures=[],
        diagnostics=[],
        reference_values=[1],
    )
    protocol = dict(
        commit="synthetic",
        dirty=True,
        environment={},
        reference="analytic",
        points_zyx=[[1, 1, 1]],
        query_xyz=[[0.25, 0.25, 0.25]],
        timing="full",
        workspace_method="estimate",
        peak_rss_reason="not measured",
        repeats=1,
    )
    payload = json.loads(
        canonical_series_files(attempt, protocol, target_pct=1)["series.json"]
    )
    assert payload["schema_version"] == "measurement_v1"
    assert payload["metadata"]["diagnostics"]["source_schema"] == AUDIT_SCHEMA_VERSION
    assert payload["trials"] == attempt["trials"]
    assert payload["summary"]["stochastic"] is False
    with pytest.raises(ValueError, match="Expected boundary_audit_v1"):
        canonical_series_files(
            dict(attempt, schema_version="measurement_v1"), protocol, target_pct=1
        )
    with pytest.raises(ValueError, match="Unknown failure counters"):
        canonical_series_files(
            dict(attempt, failures=[dict(walks=None, transitions=None)]),
            protocol,
            target_pct=1,
        )
