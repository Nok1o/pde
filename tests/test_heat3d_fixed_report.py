from types import SimpleNamespace

import numpy as np
import pytest

from benchmarks import heat3d_fixed
from benchmarks.heat3d_six_report import group_rows
from pdelab.heat3d_solver import HeatSettings3D


@pytest.mark.parametrize("failed", [False, True])
def test_fixed_budget_retains_failed_work_without_average_of_successes(
    monkeypatch, failed
):
    seeds = []

    def compare(config, settings):
        seeds.append(settings.seed)
        last_failed = failed and len(seeds) == 2
        row = dict(
            method="mc",
            status="outer_limit" if last_failed else "completed",
            reason="Лимит итераций" if last_failed else "Завершён",
            seconds=2.0,
            preparation_seconds=0.5,
            coarse_seconds=0.0,
            walks=8,
            completed_walks=7 if last_failed else 8,
            transitions=12,
            workspace_bytes=1000,
            batch_size=None,
        )
        arrays = {"analytic": np.array([10.0])}
        if not last_failed:
            arrays["mc"] = np.array([9.0 if len(seeds) == 1 else 11.0])
        return dict(rows=[row], arrays=arrays, metadata={})

    monkeypatch.setattr(heat3d_fixed, "compare_heat3d", compare)
    result = heat3d_fixed.fixed_runs(
        {"seed": 31, "repeats": 2},
        SimpleNamespace(methods=("mc",)),
        HeatSettings3D(walks=8),
    )
    assert len(set(seeds)) == 2
    summary = result["summary"]
    assert summary["seconds_all_attempts"] == 4 and summary["walks"] == 16
    assert summary["completed_repeats"] == 2 - int(failed)
    assert summary["completed_walks"] == 16 - int(failed)
    if failed:
        assert summary["relative_error_pct"] is None
        assert summary["completed_mean_seconds"] is None
        assert result["trials"][-1]["values"] is None
    else:
        assert summary["relative_error_pct"] == pytest.approx(10.0)
        assert summary["completed_mean_seconds"] == 2.0


def test_group_ratio_requires_two_complete_series():
    def payload(complete, seconds):
        return {
            "fixed_budget": {
                "summary": {
                    "complete": complete,
                    "completed_mean_seconds": seconds,
                }
            },
            "process": {"peak_rss_bytes_before_export": 100},
        }

    rows = group_rows(
        [
            ("case--mc", payload(False, None), "fixed_budget"),
            ("case--mc_grouped-b32", payload(True, 2.0), "fixed_budget"),
        ]
    )
    assert all(row["time_ratio_to_ordinary"] is None for row in rows)
