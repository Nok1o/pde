"""Build final tables from preserved raw jobs; never run or retune a solver."""

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean, stdev

from benchmarks.heat3d_six_report import METHODS, fmt, read_jobs
from pdelab.heat3d_experiment import csv_bytes, json_bytes

STAGES = (
    "core-n8",
    "groups-n8",
    "history-n8",
    "query-slice-n8",
    "sensitivity-n8",
    "conductivity-n8",
    "core-n16",
    "history-n16",
    "coarse-n16",
    "history-long",
)
DIAGNOSTICS = (
    "history_records_discarded",
    "conductivity_reuses",
    "conductivity_fallbacks",
    "factorizations",
    "factor_reuses",
    "validation_walks",
)
PARITY_FIELDS = (
    "status",
    "seed",
    "values",
    "completed_until",
    "physical_steps",
    "outer_iterations",
    "inner_iterations",
    "walks",
    "completed_walks",
    "transitions",
    "validation_walks",
    "conductivity_reuses",
    "conductivity_fallbacks",
)


def context(stage, name, result, process):
    metadata = result["metadata"]
    config = metadata["configuration"]
    settings = metadata["solver_settings"]
    return dict(
        stage=stage,
        scenario=name.split("--")[0],
        variant=name.split("--")[1],
        **{
            k: config[k]
            for k in (
                "case",
                "n",
                "alpha",
                "capacity_gamma",
                "reaction_sigma",
                "dt",
                "T",
                "mode",
            )
        },
        kinds=json.dumps(config["kinds"]),
        query_size=metadata["query_size"],
        history_mode=settings["history_mode"],
        conductivity_refresh_layers=settings["conductivity_refresh_layers"],
        configured_coarse_n=settings["coarse_n"],
        process_peak_rss_bytes=process["peak_rss_bytes_before_export"],
        process_seconds=process["study_call_seconds"],
        source_commit=metadata["commit"],
    )


def job_rows(stage, name, payload, key):
    result = payload[key]
    base = context(stage, name, result, payload["process"])
    trials = [{**base, **trial} for trial in result["trials"]]
    targets, fixed = [], []
    if key == "study":
        for row in result["rows"]:
            selected = [
                t
                for t in trials
                if (t["phase"], t["candidate"]) == (row["phase"], row["candidate"])
            ]
            if row["status"] == "confirmed" and (
                len(selected) != row["requested_repeats"]
                or any(t["status"] != "completed" for t in selected)
                or row["phase"] != "final"
            ):
                raise ValueError(
                    f"Confirmed target has incomplete/nonfinal trials: {stage}/{name}"
                )
            targets.append(
                {
                    **base,
                    **row,
                    "seconds_std": (
                        stdev(t["seconds"] for t in selected)
                        if len(selected) > 1
                        else None
                    ),
                    "preparation_mean_seconds": (
                        mean(t["preparation_seconds"] for t in selected)
                        if selected
                        else None
                    ),
                    "study_pilot_seconds": result["pilot_seconds"],
                    "study_final_seconds": result["final_seconds"],
                }
            )
    else:
        summary = result["summary"]
        complete = len(trials) == summary["requested_repeats"] and all(
            t["status"] == "completed" for t in trials
        )
        if complete != summary["complete"]:
            raise ValueError(f"Summary/trial completeness differs for {stage}/{name}")
        if not complete and (
            summary["relative_error_pct"] is not None
            or summary["completed_mean_seconds"] is not None
        ):
            raise ValueError(f"Incomplete fixed series claims a mean in {stage}/{name}")
        fixed.append(
            {
                **base,
                **summary,
                **{
                    k: sum(t[k] for t in trials)
                    for k in DIAGNOSTICS
                    if all(k in t for t in trials)
                },
            }
        )
    return targets, fixed, trials


def collect(root):
    decisions_path = root / "decisions.json"
    decisions = (
        json.loads(decisions_path.read_bytes()) if decisions_path.exists() else {}
    )
    targets, fixed, trials, stages = [], [], [], []
    for stage in STAGES:
        folder = root / stage
        if not (folder / "manifest.json").exists():
            reason = decisions.get(stage)
            if not reason:
                raise ValueError(
                    f"Missing stage {stage}; record its reason in decisions.json"
                )
            stages.append(
                dict(stage=stage, status="not_run", reason=reason, jobs=0, attempts=0)
            )
            continue
        manifest = json.loads((folder / "manifest.json").read_bytes())
        if manifest["source_before"]["dirty"]:
            raise ValueError(f"Dirty measured source in {stage}")
        if manifest["completed_jobs"] != manifest["method_order"]:
            raise ValueError(f"Missing or reordered raw jobs in {stage}")
        key = (
            "fixed_budget"
            if manifest["plan"].get("experiment") == "fixed_budget"
            else "study"
        )
        jobs = read_jobs(folder, key)
        stage_trials = []
        for name, payload, key in jobs:
            if (
                payload[key]["metadata"]["commit"]
                != manifest["source_before"]["commit"]
            ):
                raise ValueError(
                    f"Raw job source differs from manifest: {stage}/{name}"
                )
            a, b, c = job_rows(stage, name, payload, key)
            targets.extend(a)
            fixed.extend(b)
            trials.extend(c)
            stage_trials.extend(c)
        stages.append(
            dict(
                stage=stage,
                status="measured",
                reason="",
                jobs=len(jobs),
                attempts=len(stage_trials),
                completed=sum(t["status"] == "completed" for t in stage_trials),
                solver_seconds=sum(t["seconds"] for t in stage_trials),
                source_commit=manifest["source_before"]["commit"],
                failures=dict(
                    Counter(
                        t["status"] for t in stage_trials if t["status"] != "completed"
                    )
                ),
            )
        )
    return targets, fixed, trials, stages


def paired_history(fixed, trials):
    """Do not count a time-limit-dependent work difference as a memory speedup."""
    lookup = {
        (r["stage"], r["scenario"], r["method"], r["history_mode"]): r for r in fixed
    }
    rows = []
    for key, full in lookup.items():
        stage, scenario, method, mode = key
        if mode != "full" or (stage, scenario, method, "layers") not in lookup:
            continue
        layers = lookup[(stage, scenario, method, "layers")]
        pair = [
            [
                t
                for t in trials
                if (t["stage"], t["scenario"], t["method"], t["history_mode"])
                == (stage, scenario, method, m)
            ]
            for m in ("full", "layers")
        ]
        equal = len(pair[0]) == len(pair[1]) and all(
            all(
                field in a and field in b and a[field] == b[field]
                for field in PARITY_FIELDS
            )
            for a, b in zip(*pair, strict=True)
        )
        comparable = equal and full["complete"] and layers["complete"]
        rows.append(
            dict(
                stage=stage,
                scenario=scenario,
                method=method,
                same_results_and_work=equal,
                both_complete=full["complete"] and layers["complete"],
                full_workspace_bytes=full["workspace_bytes"],
                layers_workspace_bytes=layers["workspace_bytes"],
                full_process_rss_bytes=full["process_peak_rss_bytes"],
                layers_process_rss_bytes=layers["process_peak_rss_bytes"],
                time_ratio=(
                    layers["completed_mean_seconds"] / full["completed_mean_seconds"]
                    if comparable
                    else None
                ),
                workspace_ratio=(
                    layers["workspace_bytes"] / full["workspace_bytes"]
                    if comparable
                    else None
                ),
                process_rss_ratio=(
                    layers["process_peak_rss_bytes"] / full["process_peak_rss_bytes"]
                    if comparable
                    else None
                ),
                history_records_discarded=layers.get("history_records_discarded"),
            )
        )
    return rows


def markdown(rows, columns):
    def cell(value):
        if value is None:
            return "—"
        if isinstance(value, float):
            return fmt(value)
        return str(value).replace("|", "\\|").replace("\n", " ")

    lines = ["| " + " | ".join(columns.values()) + " |", "|" + "---|" * len(columns)]
    for row in rows:
        lines.append("| " + " | ".join(cell(row.get(key)) for key in columns) + " |")
    return "\n".join(lines) + "\n"


def tables(root, targets, fixed, trials, stages):
    history = paired_history(fixed, trials)
    for name, rows in (
        ("targets", targets),
        ("fixed", fixed),
        ("trials", trials),
        ("stages", stages),
        ("history-pairs", history),
    ):
        (root / (name + ".csv")).write_bytes(csv_bytes(rows))
    (root / "totals.json").write_bytes(
        json_bytes(
            dict(
                targets=dict(Counter(r["status"] for r in targets)),
                trials=dict(Counter(r["status"] for r in trials)),
                solver_seconds=sum(r["seconds"] for r in trials),
                stages=stages,
            )
        )
    )
    (root / "targets.md").write_text(
        "# Все цели 1/5/10%\n\nОшибка — средняя ошибка отдельных полных запусков. Время достижения показано только для подтверждённой цели. Затраты одной серии повторяются в строках её целей; суммировать нужно trials.csv. RSS относится к процессу исследования, включая импорт/JIT и все серии. Подробные причины, подготовка и счётчики находятся в CSV.\n\n"
        + markdown(
            targets,
            {
                "stage": "Этап",
                "scenario": "Случай",
                "variant": "Вариант",
                "target_pct": "Цель, %",
                "budget": "Бюджет",
                "completed_repeats": "Завершено",
                "requested_repeats": "Запрошено",
                "relative_error_pct": "Ошибка, %",
                "seconds": "Время цели, с",
                "elapsed_seconds": "Затрачено, с",
                "workspace_bytes": "Буферы, байт",
                "process_peak_rss_bytes": "RSS процесса, байт",
                "total_walks": "Начато путей",
                "completed_walks": "Завершено путей",
                "status": "Статус",
                "reason": "Причина",
            },
        )
    )
    (root / "fixed.md").write_text(
        "# Фиксированные бюджеты\n\nВсе попытки сохранены. Средняя ошибка и время полного решения доступны только для полной серии; время попыток включает отказы. Эти строки не являются подтверждением целей 1/5/10%.\n\n"
        + markdown(
            fixed,
            {
                "stage": "Этап",
                "scenario": "Случай",
                "variant": "Вариант",
                "completed_repeats": "Завершено",
                "requested_repeats": "Запрошено",
                "relative_error_pct": "Ошибка, %",
                "completed_mean_seconds": "Полное время, с",
                "seconds_all_attempts": "Все попытки, с",
                "workspace_bytes": "Буферы, байт",
                "process_peak_rss_bytes": "RSS процесса, байт",
                "walks": "Начато путей",
                "completed_walks": "Завершено путей",
                "reason": "Причина",
            },
        )
    )
    (root / "history-pairs.md").write_text(
        "# Сопоставление full/layers\n\nОтношения времени и памяти показаны только для полностью завершённых пар с одинаковыми значениями, остановкой и работой. RSS измерен на процесс со всеми повторами и JIT.\n\n"
        + markdown(
            history,
            {
                "stage": "Этап",
                "scenario": "Случай",
                "method": "Метод",
                "same_results_and_work": "Одинаковые результат/работа",
                "both_complete": "Обе серии полны",
                "time_ratio": "Время layers/full",
                "workspace_ratio": "Буферы layers/full",
                "process_rss_ratio": "RSS layers/full",
                "history_records_discarded": "Сокращено записей",
            },
        )
    )


def plots(root, targets, fixed, history):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = dict(zip(METHODS, plt.get_cmap("tab10").colors, strict=False))
    for stage in ("core-n8", "core-n16", "conductivity-n8", "coarse-n16"):
        cases = list(
            dict.fromkeys(r["scenario"] for r in targets if r["stage"] == stage)
        )
        if not cases:
            continue
        fig, axes = plt.subplots(
            (len(cases) + 1) // 2,
            2,
            figsize=(13, 3.2 * ((len(cases) + 1) // 2)),
            squeeze=False,
        )
        for ax, case in zip(axes.flat, cases):
            selected = [
                r
                for r in targets
                if r["stage"] == stage
                and r["scenario"] == case
                and r["target_pct"] == 5
            ]
            for row in selected:
                if row["status"] != "confirmed":
                    continue
                ax.scatter(
                    row["seconds"],
                    row["relative_error_pct"],
                    color=colors[row["method"]],
                    label=row["variant"],
                )
            ax.set(
                title=case,
                xlabel="Полное время при цели 5%, с",
                ylabel="Средняя ошибка, %",
                xscale="log",
            )
            ax.grid(alpha=0.2)
            if ax.collections:
                ax.legend(fontsize=7)
            else:
                ax.text(
                    0.5,
                    0.5,
                    "Нет подтверждённых целей",
                    transform=ax.transAxes,
                    ha="center",
                )
        for ax in list(axes.flat)[len(cases) :]:
            ax.set_visible(False)
        fig.suptitle("Подтверждённые цели 5% · отказы сохранены в таблице")
        fig.tight_layout()
        fig.savefig(root / (stage + ".png"), dpi=150)
        plt.close(fig)
    # Group sizes are shown at fixed N, without selecting a winning size.
    group_cases = list(
        dict.fromkeys(r["scenario"] for r in fixed if r["stage"] == "groups-n8")
    )
    if group_cases:
        fig, axes = plt.subplots(1, len(group_cases), figsize=(12, 4), squeeze=False)
        for ax, case in zip(axes.flat, group_cases, strict=True):
            for method in ("mc_grouped", "mc_control_grouped"):
                rows = [
                    r
                    for r in fixed
                    if r["stage"] == "groups-n8"
                    and r["scenario"] == case
                    and r["method"] == method
                    and r["complete"]
                ]
                color = colors[method]
                ax.plot(
                    [r["batch_size"] for r in rows],
                    [r["completed_mean_seconds"] for r in rows],
                    "o-",
                    label=METHODS[method],
                    color=color,
                )
                ordinary = method.removesuffix("_grouped")
                baseline = [
                    r
                    for r in fixed
                    if r["stage"] == "groups-n8"
                    and r["scenario"] == case
                    and r["method"] == ordinary
                    and r["complete"]
                ]
                if baseline:
                    ax.axhline(
                        baseline[0]["completed_mean_seconds"],
                        color=color,
                        linestyle="--",
                        label=METHODS[ordinary],
                    )
            ax.set(
                title=case,
                xlabel="Размер группы B",
                ylabel="Среднее полное время, с",
                xscale="log",
            )
            ax.legend(fontsize=8)
            ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(root / "groups.png", dpi=150)
        plt.close(fig)
    comparable = [r for r in history if r["workspace_ratio"] is not None]
    if comparable:
        fig, axes = plt.subplots(1, 3, figsize=(15, max(5, 0.25 * len(comparable))))
        labels = [f"{r['stage']}/{r['scenario']}/{r['method']}" for r in comparable]
        for ax, key, title in zip(
            axes,
            ("workspace_ratio", "process_rss_ratio", "time_ratio"),
            ("Оценка буферов", "RSS процесса", "Полное время"),
            strict=True,
        ):
            ax.barh(
                range(len(comparable)), [r[key] for r in comparable], color="tab:blue"
            )
            ax.axvline(1, color="black", linewidth=0.8)
            ax.set(title=title, xlabel="layers / full", yticks=range(len(comparable)))
            ax.set_yticklabels(labels if key == "workspace_ratio" else [])
            ax.grid(axis="x", alpha=0.2)
        fig.suptitle("Только полные пары с одинаковыми результатами и работой")
        fig.tight_layout()
        fig.savefig(root / "history.png", dpi=150)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path("benchmarks/results/heat3d-final")
    )
    args = parser.parse_args()
    targets, fixed, trials, stages = collect(args.root)
    tables(args.root, targets, fixed, trials, stages)
    plots(args.root, targets, fixed, paired_history(fixed, trials))
    print(dict(Counter(t["status"] for t in trials)))


if __name__ == "__main__":
    main()
