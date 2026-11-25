"""Issue #31, first increment: metrics and portable records, no solver/search.

The legacy study_policy module keeps its original L1/adaptive semantics.
All values here belong to a fixed query Q and independent complete runs.
"""

import csv
import io
import json
from dataclasses import asdict, dataclass

import numpy as np

SCHEMA_VERSION = "measurement_v1"
SELECTION_POLICY = "frozen_budget_holdout_v1"
ACCURACY_METRIC = "mean_run_relative_l2_v1"
PHASES = {"pilot": 0, "control": 1}


def run_seed(root_seed, case_index, method_index, attempt_index, phase, repeat):
    """Stable stream identity; indexes are assigned in the saved plan, not hashes."""
    indexes = (root_seed, case_index, method_index, attempt_index, repeat)
    if any(type(i) is not int or i < 0 for i in indexes):
        raise ValueError("Seed indexes must be nonnegative integers")
    return int(
        np.random.SeedSequence(
            root_seed,
            spawn_key=(case_index, method_index, attempt_index, PHASES[phase], repeat),
        ).generate_state(1, dtype=np.uint64)[0]
    )


def query_error(values, reference):
    """Return mean run L2 error and percentage, or None for zero reference.

    reference: [query]; values: [repeat, query], including a one-point query.
    Shapes are strict: broadcasting must never silently change the query.
    """
    reference = np.asarray(reference, dtype=float)
    values = np.asarray(values, dtype=float)
    if (
        reference.ndim != 1
        or not reference.size
        or values.ndim != 2
        or values.shape[1] != reference.size
        or not values.shape[0]
    ):
        raise ValueError("Expected reference[query] and values[repeat, query]")
    if not np.isfinite(reference).all() or not np.isfinite(values).all():
        raise ValueError("Values and reference must be finite")
    absolute = float(np.linalg.norm(values - reference, axis=1).mean())
    scale = float(np.linalg.norm(reference))
    return absolute, 100 * absolute / scale if scale else None


@dataclass(frozen=True)
class Trial:
    """One full solve; counters and costs include work before failure.

    status is a backend status (completed, time_limit, outer_limit, ...).
    paths_per_estimate is inapplicable for deterministic methods (None).
    Numerical diagnostics and the full method configuration live in JSON.
    """

    repeat: int
    seed: int
    status: str
    reason: str
    values: list[float] | None
    seconds: float
    preparation_seconds: float
    workspace_bytes: int
    peak_rss_bytes: int | None
    paths_per_estimate: int | None
    walks: int
    transitions: int
    physical_steps: int
    outer_iterations: int
    inner_iterations: int
    completed_until: float | None

    def __post_init__(self):
        for name in (
            "repeat",
            "seed",
            "workspace_bytes",
            "peak_rss_bytes",
            "paths_per_estimate",
            "walks",
            "transitions",
            "physical_steps",
            "outer_iterations",
            "inner_iterations",
        ):
            value = getattr(self, name)
            optional = name in {"peak_rss_bytes", "paths_per_estimate"}
            if (value is None and not optional) or (
                value is not None and (type(value) is not int or value < 0)
            ):
                raise ValueError(f"Invalid counter: {name}")
        if (
            not np.isfinite([self.seconds, self.preparation_seconds]).all()
            or not 0 <= self.preparation_seconds <= self.seconds
        ):
            raise ValueError("Preparation must be a subset of finite total time")
        if not self.status or (self.status != "completed" and not self.reason):
            raise ValueError("An incomplete trial needs a status and reason")
        if (self.status == "completed") != (self.values is not None):
            raise ValueError("Only completed trials may contain query values")
        if self.completed_until is not None and (
            not np.isfinite(self.completed_until) or self.completed_until < 0
        ):
            raise ValueError("Invalid completed time")


def summarize(trials, reference, *, requested_repeats, target_pct, phase):
    """Summarize one frozen budget, retaining partial/failing series costs.

    A pilot can never confirm a target. Caller must persist the choice before
    launching control. No retry, budget selection or success filtering occurs.
    """
    if phase not in PHASES:
        raise ValueError("Unknown phase")
    if type(requested_repeats) is not int or requested_repeats < 1:
        raise ValueError("Need a positive requested repeat count")
    if not np.isfinite(target_pct) or target_pct <= 0:
        raise ValueError("Need a positive finite percentage target")
    if len(trials) > requested_repeats or len({t.repeat for t in trials}) != len(
        trials
    ):
        raise ValueError("Duplicate or excessive repeats")
    if any(t.repeat >= requested_repeats for t in trials):
        raise ValueError("Repeat index outside the requested series")
    if len({t.paths_per_estimate for t in trials}) > 1:
        raise ValueError("A series must use one frozen paths-per-estimate budget")
    if any(t.paths_per_estimate is not None for t in trials) and len(
        {t.seed for t in trials}
    ) != len(trials):
        raise ValueError("Stochastic repeats need distinct seeds")
    reference = np.asarray(reference, dtype=float)
    # Validate Q even when no solve has completed.
    query_error(np.zeros((1, reference.size)), reference)
    completed = [t for t in trials if t.status == "completed"]
    for trial in completed:
        query_error([trial.values], reference)
    full = len(completed) == requested_repeats
    absolute, relative = (
        query_error([t.values for t in completed], reference) if full else (None, None)
    )
    seconds = sum(t.seconds for t in trials)
    if not full:
        status, reason = "incomplete", "Not all requested full runs completed"
    elif relative is None:
        status, reason = (
            "relative_not_applicable",
            "Zero reference norm; absolute L2 error only",
        )
    elif phase == "pilot":
        status, reason = "pilot_only", "Budget selection data cannot confirm a target"
    elif relative <= target_pct:
        status, reason = "confirmed", "Complete independent control series passed"
    else:
        status, reason = "not_reached", "Independent control error exceeds the target"
    return dict(
        accuracy_metric=ACCURACY_METRIC,
        selection_policy=SELECTION_POLICY,
        phase=phase,
        target_pct=target_pct,
        status=status,
        reason=reason,
        requested_repeats=requested_repeats,
        completed_repeats=len(completed),
        recorded_repeats=len(trials),
        absolute_l2_error=absolute,
        relative_error_pct=relative,
        spent_seconds=seconds,
        mean_seconds=seconds / requested_repeats if full else None,
        seconds_to_target=(
            seconds / requested_repeats if status == "confirmed" else None
        ),
        seconds_std=(
            float(np.std([t.seconds for t in trials], ddof=1))
            if full and len(trials) > 1
            else None
        ),
        walks=sum(t.walks for t in trials),
        transitions=sum(t.transitions for t in trials),
        workspace_bytes=max((t.workspace_bytes for t in trials), default=None),
        peak_rss_bytes=(
            max((t.peak_rss_bytes for t in trials), default=None)
            if trials and all(t.peak_rss_bytes is not None for t in trials)
            else None
        ),
    )


def export_series(metadata, trials, reference, **series):
    """Return JSON + flat per-trial CSV + flat summary CSV as UTF-8 bytes.

    This first-stage format requires explicit provenance supplied by the runner;
    it never invents a commit, a memory measurement or missing solver diagnostics.
    """
    required = {
        "series_id",
        "commit",
        "dirty",
        "environment",
        "configuration",
        "reference",
        "query",
        "budget",
        "timing",
        "memory",
        "diagnostics",
    }
    if required - metadata.keys():
        raise ValueError(f"Missing metadata: {sorted(required - metadata.keys())}")
    summary = summarize(trials, reference, **series)
    payload = dict(
        schema_version=SCHEMA_VERSION,
        metadata=metadata,
        reference_values=np.asarray(reference, dtype=float).tolist(),
        summary=summary,
        trials=[asdict(t) for t in trials],
    )
    encoded = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False).encode(
        "utf-8"
    )

    def csv_data(rows):
        buffer = io.StringIO(newline="")
        writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        return buffer.getvalue().encode("utf-8-sig")

    rows = []
    for trial in trials:
        row = {"series_id": metadata["series_id"], **asdict(trial)}
        values = row.pop("values")
        row["values_json"] = json.dumps(values, allow_nan=False)
        rows.append(row)
    return {
        "series.json": encoded,
        "trials.csv": csv_data(rows) if rows else b"",
        "summary.csv": csv_data([{"series_id": metadata["series_id"], **summary}]),
    }
