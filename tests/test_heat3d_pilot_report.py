import pytest

from benchmarks.heat3d_pilot import jobs, summarize


def test_summary_keeps_failed_attempt_costs_and_counts_reused_series_once():
    trials = [
        dict(
            phase="pilot",
            candidate=0,
            seconds=1,
            preparation_seconds=0.2,
            walks=10,
            completed_walks=10,
            transitions=20,
            status="completed",
        ),
        dict(
            phase="pilot",
            candidate=0,
            seconds=3,
            preparation_seconds=0.4,
            walks=5,
            completed_walks=4,
            transitions=8,
            status="walk_budget",
        ),
    ]
    row = dict(
        phase="pilot",
        candidate=0,
        status="pilot_incomplete",
        seconds=None,
        relative_error_pct=None,
    )
    payload = dict(
        study=dict(
            trials=trials,
            rows=[dict(row, target_pct=t) for t in (1, 5, 10)],
            series=[row],
            pilot_seconds=4,
            final_seconds=0,
        ),
        process=dict(peak_rss_bytes_before_export=100),
    )
    targets, series, raw, costs = summarize([("case--mc", payload)])
    assert len(targets) == 3 and len(series) == 1 and len(raw) == 2
    assert all(
        t["seconds"] is None and t["relative_error_pct"] is None for t in targets
    )
    assert all(t["observed_preparation_mean"] == pytest.approx(0.3) for t in targets)
    assert len(costs) == 1 and costs[0]["pilot_seconds"] == 4
    assert costs[0]["paths_started"] == 15 and costs[0]["paths_completed"] == 14
    assert costs[0]["completed_runs"] == 1 and costs[0]["attempted_runs"] == 2


def test_plan_expansion_preserves_variants_and_coordinates():
    plan = dict(
        configuration=dict(n=8),
        settings=dict(coarse_n=4),
        cases=[dict(id="case", configuration=dict(point=[2, 4, 6]))],
        variants=[dict(id="control", method="mc_control", settings=dict(coarse_n=6))],
    )
    name, config, settings = next(jobs(plan))
    assert name == "case--control"
    assert config.methods == ("mc_control",) and config.point == [2, 4, 6]
    assert settings.coarse_n == 6
