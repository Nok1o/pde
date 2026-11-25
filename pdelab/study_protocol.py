"""Small fixed-candidate accuracy protocol shared with future issue #31 adapters.

The runner performs a complete solve. All candidate selection finishes before
any final series starts. Final observations never feed back into selection.
Trial diagnostics are retained; stationary callers may additionally retain query
vectors. Full physical solution histories are never copied into this protocol.
"""

from time import perf_counter

import numpy as np

PROTOCOL_ID = "fixed_candidate_independent_final_v1"
METRIC_ID = "mean_run_relative_l2_v1"
ABSOLUTE_METRIC_ID = "mean_run_absolute_l2_v1"
METRIC_DESCRIPTION = (
    "100 * mean_runs(norm_Q(u_run-reference, 2) / norm_Q(reference, 2))"
)


def error_metrics(values, reference):
    """Average norms of individual runs; a zero reference has no percent error."""
    reference = np.atleast_1d(np.asarray(reference, dtype=float))
    values = np.asarray(values, dtype=float)
    if reference.ndim != 1 or not reference.size:
        raise ValueError("Эталон должен быть непустым вектором значений в Q")
    if values.ndim == 0:
        values = values.reshape(1, 1)
    elif values.ndim == 1:
        values = values.reshape(-1, 1) if reference.size == 1 else values[None, :]
    if values.ndim != 2 or values.shape[1] != reference.size or not len(values):
        raise ValueError("Нужны значения формы [повтор, точка Q]")
    if not np.isfinite(values).all() or not np.isfinite(reference).all():
        raise ValueError("Значения и эталон должны быть конечными")
    scale = float(np.linalg.norm(reference))
    absolute = float(np.mean(np.linalg.norm(values - reference, axis=1)))
    relative = 100 * absolute / scale if scale else None
    if not np.isfinite(absolute) or (
        relative is not None and not np.isfinite(relative)
    ):
        raise ValueError("Неконечная оценка ошибки")
    return dict(
        accuracy_metric=METRIC_ID,
        absolute_metric=ABSOLUTE_METRIC_ID,
        absolute_error=absolute,
        relative_error_pct=relative,
        reference_norm=scale,
        relative_applicable=scale != 0,
    )


def _seed(seed, phase, candidate, repeat):
    return int(
        np.random.SeedSequence([seed, 31025, phase, candidate, repeat]).generate_state(
            1
        )[0]
    )


def run_accuracy_study(
    runner,
    budgets,
    reference,
    *,
    repeats=5,
    seed=42,
    targets=(1.0, 5.0, 10.0),
    calibration_fraction=0.8,
    method="lod",
    budget_kind="relative_residual_tolerance",
    walks_applicable=False,
    progress=None,
    seconds_per_phase=None,
    clock=perf_counter,
    retain_values=False,
):
    """Select by pilots with a fixed margin, then check each frozen candidate once.

    ``runner(budget, seed)`` returns query values, status and full-run costs.
    Physical-time fields are optional for stationary observations. When supplied,
    completed_until must equal target_time. Existing heat callers keep their checks.
    The order of budgets is explicit, e.g. loose to strict deterministic tolerance.
    One final series may serve several targets selected at the same budget.
    Failure of any full run makes that entire series incomplete.
    """
    budgets, targets = list(budgets), tuple(targets)
    if not budgets or any(not np.isfinite(b) or b <= 0 for b in budgets):
        raise ValueError("Нужен непустой список положительных конечных бюджетов")
    if len(set(budgets)) != len(budgets):
        raise ValueError("Бюджеты не должны повторяться")
    if (
        isinstance(repeats, (bool, np.bool_))
        or int(repeats) != repeats
        or not 2 <= repeats <= 100
    ):
        raise ValueError("Нужно от 2 до 100 полных повторов")
    if isinstance(seed, (bool, np.bool_)) or int(seed) != seed or not 0 <= seed < 2**32:
        raise ValueError("Seed должен быть целым в [0, 2³²)")
    if not targets or any(not np.isfinite(t) or t <= 0 for t in targets):
        raise ValueError("Цели должны быть положительными и конечными")
    if len(set(targets)) != len(targets):
        raise ValueError("Цели не должны повторяться")
    if not np.isfinite(calibration_fraction) or not 0 < calibration_fraction < 1:
        raise ValueError("Доля цели при подборе должна быть строго между 0 и 1")
    if seconds_per_phase is not None and (
        not np.isfinite(seconds_per_phase) or seconds_per_phase <= 0
    ):
        raise ValueError("Лимит каждой фазы должен быть положительным")
    repeats, seed = int(repeats), int(seed)
    reference = np.atleast_1d(np.asarray(reference, dtype=float))
    reference_info = error_metrics(reference, reference)
    trials, series, chosen = [], [], {}
    maximum_runs = 2 * len(budgets) * repeats
    deadlines = {}

    def measure(index, phase):
        records = []
        timed_out = False
        if seconds_per_phase is not None and phase not in deadlines:
            deadlines[phase] = clock() + seconds_per_phase
        for repeat in range(repeats):
            if seconds_per_phase is not None and clock() >= deadlines[phase]:
                timed_out = True
                break
            trial_seed = _seed(seed, 0 if phase == "pilot" else 1, index, repeat)
            if progress:
                progress(
                    len(trials) / maximum_runs,
                    f"{'Подбор' if phase == 'pilot' else 'Финальная проверка'}: "
                    f"допуск/бюджет {budgets[index]:g}, повтор {repeat + 1}/{repeats}",
                )
            result = runner(budgets[index], trial_seed)
            record = dict(
                phase=phase,
                candidate=index,
                budget=budgets[index],
                repeat=repeat + 1,
                seed=trial_seed,
                status=result.status,
                reason=result.reason,
                target_time=getattr(result, "target_time", None),
                completed_until=getattr(result, "completed_until", None),
                seconds=float(result.seconds),
                preparation_seconds=float(result.preparation_seconds),
                workspace_bytes=int(result.workspace_bytes),
                peak_rss_bytes=result.peak_rss_bytes,
                physical_steps=int(getattr(result, "physical_steps", 0)),
                outer_iterations=int(getattr(result, "outer_iterations", 0)),
                inner_iterations=int(getattr(result, "inner_iterations", 0)),
                walks=int(result.walks),
                transitions=int(result.transitions),
                relative_error_pct=None,
                absolute_error=None,
            )
            if result.status == "completed":
                if result.values is None or (
                    record["target_time"] is not None
                    and record["completed_until"] != record["target_time"]
                ):
                    record.update(
                        status="invalid_result", reason="Нет решения в целевой момент T"
                    )
                else:
                    try:
                        metrics = error_metrics(
                            np.asarray(result.values)[None, :], reference
                        )
                        record.update(
                            relative_error_pct=metrics["relative_error_pct"],
                            absolute_error=metrics["absolute_error"],
                        )
                    except (ValueError, IndexError) as exc:
                        record.update(status="nonfinite", reason=str(exc))
            if retain_values:
                record["values"] = (
                    np.asarray(result.values).tolist()
                    if record["status"] == "completed"
                    else None
                )
            history = getattr(result, "history", None)
            if history:
                for key in (
                    "linear_relative_residual",
                    "nonlinear_relative_residual",
                    "residual_scale",
                ):
                    record[key] = history[-1].get(key)
            records.append(record)
            trials.append(record)
        completed = sum(record["status"] == "completed" for record in records)
        full = completed == repeats
        failed = [record for record in records if record["status"] != "completed"]
        result = dict(
            phase=phase,
            candidate=index,
            budget=budgets[index],
            requested_repeats=repeats,
            recorded_repeats=len(records),
            completed_repeats=completed,
            complete=full,
            relative_error_pct=(
                float(np.mean([r["relative_error_pct"] for r in records]))
                if full and reference_info["relative_applicable"]
                else None
            ),
            absolute_error=(
                float(np.mean([r["absolute_error"] for r in records])) if full else None
            ),
            elapsed_seconds=sum(r["seconds"] for r in records),
            mean_seconds=(
                float(np.mean([r["seconds"] for r in records])) if records else None
            ),
            preparation_seconds=sum(r["preparation_seconds"] for r in records),
            workspace_bytes=max((r["workspace_bytes"] for r in records), default=0),
            peak_rss_bytes=(
                max(r["peak_rss_bytes"] for r in records)
                if records and all(r["peak_rss_bytes"] is not None for r in records)
                else None
            ),
            physical_steps=sum(r["physical_steps"] for r in records),
            outer_iterations=sum(r["outer_iterations"] for r in records),
            inner_iterations=sum(r["inner_iterations"] for r in records),
            total_walks=sum(r["walks"] for r in records) if walks_applicable else None,
            transitions=sum(r["transitions"] for r in records),
            solver_status=(
                "completed"
                if full
                else ("time_limit" if timed_out else failed[0]["status"])
            ),
            reason=(
                ""
                if full
                else (
                    "Лимит времени фазы: не все запуски начаты"
                    if timed_out
                    else "; ".join(
                        dict.fromkeys(r["reason"] or r["status"] for r in failed)
                    )
                )
            ),
        )
        series.append(result)
        return result

    # With a zero reference, no percentage-based candidate can be selected.
    # Evaluate the first explicitly configured candidate once for absolute diagnostics.
    if not reference_info["relative_applicable"]:
        diagnostic = measure(0, "absolute_diagnostic")
        rows = [
            {
                **diagnostic,
                "target_pct": target,
                "status": "relative_target_not_applicable",
                "reason": "Норма эталона равна нулю: относительная цель неприменима. "
                + diagnostic["reason"],
                "seconds": None,
                "final_series": False,
            }
            for target in targets
        ]
    else:
        pilots = []
        for index in range(len(budgets)):
            pilot = measure(index, "pilot")
            pilots.append(pilot)
            if pilot["complete"]:
                for target in targets:
                    if (
                        target not in chosen
                        and pilot["relative_error_pct"] <= calibration_fraction * target
                    ):
                        chosen[target] = index
            if len(chosen) == len(targets):
                break
            if pilot["solver_status"] == "time_limit":
                break
        # No adaptation after this line: the entire candidate map is frozen.
        confirmations = {
            index: measure(index, "final") for index in sorted(set(chosen.values()))
        }
        rows = []
        for target in targets:
            index = chosen.get(target)
            final = confirmations.get(index)
            observed = final if final is not None else pilots[-1]
            if final is None:
                status = "not_reached" if observed["complete"] else "pilot_incomplete"
                reason = (
                    "В заданных бюджетах не достигнут фиксированный запас точности"
                    if observed["complete"]
                    else observed["reason"]
                )
            elif not final["complete"]:
                status, reason = "validation_incomplete", final["reason"]
            elif final["relative_error_pct"] > target:
                status, reason = (
                    "validation_failed",
                    "Финальная независимая серия не подтвердила цель; подбор завершён",
                )
            else:
                status, reason = (
                    "confirmed",
                    "Цель подтверждена полной независимой финальной серией",
                )
            rows.append(
                {
                    **observed,
                    "target_pct": target,
                    "status": status,
                    "reason": reason,
                    "seconds": final["mean_seconds"] if status == "confirmed" else None,
                    "final_series": final is not None,
                    "calibration_error_pct": (
                        pilots[index]["relative_error_pct"]
                        if index is not None
                        else observed["relative_error_pct"]
                    ),
                }
            )
    for row in rows:
        row.update(
            method=method,
            budget_kind=budget_kind,
            walks_applicable=walks_applicable,
            protocol=PROTOCOL_ID,
            accuracy_metric=METRIC_ID,
            absolute_metric=ABSOLUTE_METRIC_ID,
            calibration_fraction=calibration_fraction,
            calibration_target_pct=calibration_fraction * row["target_pct"],
        )
    if progress:
        progress(1.0, "Исследование завершено; все попытки сохранены")
    return dict(
        protocol=PROTOCOL_ID,
        accuracy_metric=METRIC_ID,
        absolute_metric=ABSOLUTE_METRIC_ID,
        accuracy=METRIC_DESCRIPTION,
        reference_norm=reference_info["reference_norm"],
        relative_applicable=reference_info["relative_applicable"],
        calibration_fraction=calibration_fraction,
        requested_repeats=repeats,
        seed=seed,
        budgets=budgets,
        targets=list(targets),
        rows=rows,
        trials=trials,
        series=series,
        pilot_seconds=sum(
            s["elapsed_seconds"] for s in series if s["phase"] == "pilot"
        ),
        final_seconds=sum(
            s["elapsed_seconds"] for s in series if s["phase"] == "final"
        ),
        elapsed_seconds=sum(s["elapsed_seconds"] for s in series),
    )
