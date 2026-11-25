"""Keep failed/unequal histories out of final-report speedup claims."""

from copy import deepcopy

import pytest

from benchmarks.heat3d_final_report import PARITY_FIELDS, job_rows, paired_history


def histories():
    fixed, trials = [], []
    for mode in ("full", "layers"):
        context = dict(
            stage="history-n8", scenario="point", method="mc_control", history_mode=mode
        )
        fixed.append(
            dict(
                **context,
                complete=True,
                completed_mean_seconds=1 if mode == "full" else 0.8,
                workspace_bytes=100 if mode == "full" else 50,
                process_peak_rss_bytes=200,
                history_records_discarded=0 if mode == "full" else 10,
            )
        )
        for repeat in range(2):
            trials.append(
                {
                    **dict.fromkeys(PARITY_FIELDS, 0),
                    **context,
                    "seed": repeat,
                    "status": "completed",
                    "values": [1.0],
                    "walks": 10,
                    "completed_walks": 10,
                }
            )
    return fixed, trials


def test_history_ratios_require_equal_complete_work():
    fixed, trials = histories()
    pair = paired_history(fixed, trials)[0]
    assert pair["same_results_and_work"] and pair["both_complete"]
    assert pair["time_ratio"] == 0.8 and pair["workspace_ratio"] == 0.5
    trials[-1]["walks"] = 11
    pair = paired_history(fixed, trials)[0]
    assert not pair["same_results_and_work"]
    assert (
        pair["time_ratio"]
        is pair["workspace_ratio"]
        is pair["process_rss_ratio"]
        is None
    )


def test_missing_counters_and_incomplete_runs_do_not_claim_parity_speedup():
    fixed, trials = histories()
    for trial in trials:
        trial.pop("validation_walks")
    assert not paired_history(fixed, trials)[0]["same_results_and_work"]
    fixed, trials = histories()
    for row in fixed:
        row.update(complete=False, completed_mean_seconds=None)
    for trial in trials:
        trial.update(status="time_limit", values=None)
    pair = paired_history(fixed, trials)[0]
    assert pair["same_results_and_work"] and not pair["both_complete"]
    assert pair["time_ratio"] is None


def test_report_rejects_confirmed_target_backed_by_failed_trials():
    config = dict(
        case="sine",
        n=8,
        alpha=0.5,
        capacity_gamma=0.5,
        reaction_sigma=1,
        dt=0.002,
        T=0.004,
        mode="point",
        kinds=[1] * 6,
    )
    metadata = dict(
        configuration=config,
        solver_settings=dict(
            history_mode="full", conductivity_refresh_layers=1, coarse_n=4
        ),
        query_size=1,
        commit="measured",
    )
    payload = dict(
        process=dict(peak_rss_bytes_before_export=1000, study_call_seconds=1),
        study=dict(
            metadata=metadata,
            rows=[
                dict(
                    phase="final", candidate=0, status="confirmed", requested_repeats=2
                )
            ],
            trials=[
                dict(phase="final", candidate=0, status="completed"),
                dict(phase="final", candidate=0, status="time_limit"),
            ],
        ),
    )
    original = deepcopy(payload)
    with pytest.raises(ValueError, match="incomplete/nonfinal"):
        job_rows("core-n8", "sine--mc", payload, "study")
    assert payload == original
