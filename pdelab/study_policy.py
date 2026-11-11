"""Shared pilot margin, held-out confirmation and readable accuracy results."""

from time import perf_counter

import numpy as np

SELECTION_POLICY = "pilot_half_target_mean_relative_error_v3"
ACCURACY_METRIC = "mean_run_relative_l1_v1"
ACCURACY_DESCRIPTION = "100 * mean_runs(sum_points(abs(u_run-reference)) / sum_points(abs(reference))); independent validation"
CALIBRATION_FRACTION = 0.5
DETERMINISTIC_METHODS = {"lod", "lod_serial", "picard"}


def mean_relative_error(values, reference):
    """Mean of absolute relative errors, not RMS or error of the mean.

    For multiple query points use the relative L1 error of each whole query.
    Individual zero reference nodes are included; an all-zero reference is undefined.
    """
    reference = np.asarray(reference, dtype=float)
    values = np.asarray(values, dtype=float)
    if not reference.size or not values.size:
        raise ValueError("Для ошибки нужны непустые значения и эталон")
    if not np.isfinite(reference).all() or not np.isfinite(values).all():
        raise ValueError("Значения и эталон должны быть конечными")
    scale = np.mean(np.abs(reference))
    if scale == 0:
        raise ValueError("Нулевой эталон: относительная ошибка не определена")
    return float(100 * np.mean(np.abs(values - reference)) / scale)


def accuracy_metadata(rows):
    """Never relabel a saved RMS study with the new metric when exporting it."""
    row = rows[0] if rows else {}
    metric = row.get("accuracy_metric", "relative_rmse_v1")
    return dict(
        accuracy_metric=metric,
        accuracy=(
            ACCURACY_DESCRIPTION
            if metric == ACCURACY_METRIC
            else "100 * RMS(u_run-reference) / RMS(reference); individual runs"
        ),
        selection=row.get("selection_policy", "legacy"),
    )


def calibration_target(target, method):
    """Keep the deterministic baseline fair; reserve a margin for random errors."""
    return target if method in DETERMINISTIC_METHODS else target * CALIBRATION_FRACTION


def search_budgets(
    method,
    budgets,
    measure,
    seconds_per_phase,
    targets=(1.0, 5.0, 10.0),
    *,
    fastest=False,
    clock=perf_counter,
):
    """Choose only on pilot data, then confirm once, with a separate time allowance.

    measure(budget, index, phase, deadline) returns relative_error_pct and seconds.
    Partial/failed series must raise instead of reporting a success from fewer runs.
    A failed final check never returns its target to calibration.
    """
    rows = [
        dict(
            method=method,
            target_pct=target,
            seconds=None,
            validation_seconds=None,
            relative_error_pct=None,
            calibration_error_pct=None,
            budget=None,
            interrupted_budget=None,
            budget_kind=(
                "relative_residual_tolerance"
                if method in DETERMINISTIC_METHODS
                else "walks_per_start"
            ),
            calibration_target_pct=calibration_target(target, method),
            selection_policy=SELECTION_POLICY,
            accuracy_metric=ACCURACY_METRIC,
            status="Недостаточный запас точности в пределах бюджетов",
            status_code="not_reached",
        )
        for target in targets
    ]
    selected = {}
    deadline = clock() + seconds_per_phase
    for index, budget in enumerate(budgets):
        pending = [r for r in rows if fastest or r["target_pct"] not in selected]
        if not pending:
            break
        try:
            pilot = measure(budget, index, "calibration", deadline)
            if not np.isfinite(pilot["relative_error_pct"]):
                raise RuntimeError("Неконечная ошибка подбора")
        except (TimeoutError, RuntimeError) as exc:
            for row in pending:
                if row["target_pct"] not in selected:
                    row["interrupted_budget"] = budget
                    row["status"] = (
                        "Лимит времени поиска"
                        if isinstance(exc, TimeoutError)
                        else f"Ошибка решателя: {exc}"
                    )
                    row["status_code"] = (
                        "timeout" if isinstance(exc, TimeoutError) else "solver_failed"
                    )
            break
        for row in pending:
            target = row["target_pct"]
            # Record the last completed series even when the margin is not met.
            if target not in selected:
                row.update(
                    budget=budget, calibration_error_pct=pilot["relative_error_pct"]
                )
            if pilot["relative_error_pct"] <= row["calibration_target_pct"]:
                previous = selected.get(target)
                if previous is None or (
                    fastest and pilot["seconds"] < previous[2]["seconds"]
                ):
                    selected[target] = (budget, index, pilot)
                    row.update(
                        budget=budget, calibration_error_pct=pilot["relative_error_pct"]
                    )
    confirmations = {}
    deadline = clock() + seconds_per_phase
    for row in rows:
        chosen = selected.get(row["target_pct"])
        if chosen is None:
            continue
        budget, index, _ = chosen
        if index not in confirmations:
            try:
                confirmation = measure(budget, index, "validation", deadline)
                if not np.isfinite(confirmation["relative_error_pct"]):
                    raise RuntimeError("Неконечная ошибка проверки")
                confirmations[index] = confirmation
            except (TimeoutError, RuntimeError) as exc:
                confirmations[index] = exc
        confirmation = confirmations[index]
        if isinstance(confirmation, Exception):
            reason = (
                "лимит времени"
                if isinstance(confirmation, TimeoutError)
                else str(confirmation)
            )
            row.update(
                status=f"Бюджет найден, проверка не завершена: {reason}",
                status_code="validation_incomplete",
            )
            continue
        error = confirmation["relative_error_pct"]
        passed = np.isfinite(error) and error <= row["target_pct"]
        row.update(
            relative_error_pct=error,
            validation_seconds=confirmation["seconds"],
            seconds=confirmation["seconds"] if passed else None,
            status="Достигнута" if passed else "Не достигнута на контрольной серии",
            status_code="confirmed" if passed else "validation_failed",
        )
    return rows


def readable_status(row):
    status = row.get("status_code", row.get("status", ""))
    labels = {
        "confirmed": "Достигнута",
        "reached": "Достигнута",
        "validation_failed": "Проверка не прошла",
        "not_confirmed": "Проверка не прошла",
        "not_reached": "Нет запаса точности",
        "budget_exhausted": "Нет запаса точности",
        "timeout": "Лимит времени подбора",
        "time_limit": "Лимит времени подбора",
        "zero_reference": "Нулевой эталон: процент не определён",
        "solver_failed": "Решатель не завершён",
    }
    if status == "validation_incomplete":
        return row.get("status_detail", row["status"])
    if status == "solver_failed":
        reason = row.get("reason") or row.get("status_detail") or row["status"]
        for code, label in {
            "walk_budget": "лимит общего числа путей",
            "validation_budget": "лимит путей контрольной проверки",
            "max_outer": "лимит внешних итераций",
        }.items():
            reason = reason.replace(code, label)
        return reason
    return labels.get(status, row.get("status", str(status)))


def budget_label(row):
    budget = row.get("budget")
    if budget is None:
        budget = row.get("walks")
    if budget is None:
        interrupted = row.get("interrupted_budget")
        if interrupted is None:
            return "не выбран"
        label = (
            f"tol={interrupted:g}"
            if row["method"] in DETERMINISTIC_METHODS
            else f"N={int(interrupted)}"
        )
        return label + " (серия не завершена)"
    if row["method"] in DETERMINISTIC_METHODS:
        return f"tol={budget:g}"
    prefix = "N₀" if row.get("budget_kind") == "initial_walks_per_node" else "N"
    label = f"{prefix}={int(budget):,}".replace(",", " ")
    if prefix == "N₀":
        label += f"; Nmax={row['max_walks']}"
    return label


def error_label(row):
    error = row.get("relative_error_pct", row.get("validation_error_pct"))
    if error is not None:
        return f"проверка {error:.3g}%"
    error = row.get("calibration_error_pct")
    return f"подбор {error:.3g}%" if error is not None else "ошибка ещё не измерена"


def display_timing_rows(rows, names=None):
    """Numeric export stays unchanged; the on-screen matrix has no opaque nulls."""
    names = {} if names is None else names
    result = []
    for target in sorted({r["target_pct"] for r in rows}):
        line = {"Относительная ошибка, %": target}
        for row in rows:
            if row["target_pct"] != target:
                continue
            label = names.get(row["method"], row.get("name", row["method"]))
            if row["seconds"] is not None:
                cell = f"{row['seconds']:.6f} с"
            else:
                cell = (
                    f"{readable_status(row)}\n{budget_label(row)}\n{error_label(row)}"
                )
            line[label] = cell
        result.append(line)
    return result


def render_accuracy_results(rows, names=None, *, key):
    import pandas as pd
    import streamlit as st

    names = {} if names is None else names
    if any(r.get("selection_policy") != SELECTION_POLICY for r in rows):
        st.info(
            "Показано исследование по прежнему протоколу. Запустите заново для подбора с запасом."
        )
    else:
        st.caption(
            "Ошибка — среднее относительных ошибок отдельных запусков. MC подбирается до половины порога, затем независимо проверяется. Время — одного проверочного запуска."
        )
    timing = display_timing_rows(rows, names)
    for line in timing:
        for column, value in line.items():
            if isinstance(value, str) and "\n" in value:
                line[column] = value.split("\n", 1)[0]
    st.dataframe(
        pd.DataFrame(timing), hide_index=True, width="stretch", key=key + "_table"
    )
    targets = sorted({r["target_pct"] for r in rows})
    selected = st.segmented_control(
        "Бюджеты и ошибки для порога",
        targets,
        default=targets[0],
        format_func=lambda target: f"{target:g}%",
        key=key + "_target",
    )
    selected = targets[0] if selected is None else selected
    details = []
    for row in rows:
        if row["target_pct"] != selected:
            continue
        pilot = row.get("calibration_error_pct")
        check = row.get("relative_error_pct", row.get("validation_error_pct"))
        status = readable_status(row)
        interrupted = row.get("interrupted_budget")
        if interrupted is not None and row.get("budget") is not None:
            if row["method"] in DETERMINISTIC_METHODS:
                label = f"tol={interrupted:g}"
            else:
                label = f"N={int(interrupted)}"
            status += f"; серия {label} не завершена"
        details.append(
            {
                "Метод": names.get(row["method"], row.get("name", row["method"])),
                "Бюджет": budget_label(row),
                "Подбор, %": f"{pilot:.3g}" if pilot is not None else "не завершён",
                "Проверка, %": f"{check:.3g}" if check is not None else "не завершена",
                "Статус": status,
            }
        )
    st.dataframe(
        pd.DataFrame(details), hide_index=True, width="stretch", key=key + "_details"
    )
