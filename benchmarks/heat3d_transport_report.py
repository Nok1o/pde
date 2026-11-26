"""Summarize preserved transport studies; never start or retune a PDE solver."""

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import stdev

from benchmarks.heat3d_final_report import markdown
from benchmarks.heat3d_pilot import utc_now
from benchmarks.heat3d_six_report import read_jobs
from pdelab.heat3d_experiment import csv_bytes, json_bytes

METHODS = {
    "lod": "ЛОМ",
    "mc": "Обычный MC",
    "mc_counts": "MC: количества по узлам",
    "mass_transport": "Детерминированный перенос",
}
STAGES = ("frozen-n8", "full-n8", "full-n16")


def context(stage, name, metadata, process):
    method = metadata["method"]
    config = metadata["configuration"]
    frozen = stage.startswith("frozen")
    transport = method in ("mc_counts", "mass_transport")
    return dict(
        stage=stage,
        scope="one_frozen_system" if frozen else "complete_nonlinear_history",
        scenario=name.split("--")[0],
        method=method,
        n=config["n"],
        case=config["case"],
        point_xyz=config["point"],
        kinds=config["kinds"],
        alpha=config["alpha"],
        capacity_gamma=config["capacity_gamma"],
        reaction_sigma=config["reaction_sigma"],
        dt=config["dt"],
        T=None if frozen else config["T"],
        reference_kind=(
            "sparse_direct_same_frozen_operator"
            if frozen
            else metadata["reference_kind"]
        ),
        process_peak_rss_bytes=process["peak_rss_bytes_before_export"],
        study_call_seconds=process["study_call_seconds"],
        nodal_counters_applicable=transport,
        edge_work_kind=(
            "processed_nonzero_coefficients"
            if not frozen and method == "mass_transport"
            else "nonempty_flows" if transport else "not_applicable"
        ),
        sweep_kind=(
            "full_field_iterations"
            if not frozen and method == "mass_transport"
            else "generations_summed_over_starts" if transport else "not_applicable"
        ),
    )


def applicable(row, base):
    result = {**base, **row}
    stochastic = base["method"] in ("mc", "mc_counts")
    result["virtual_particle_moves"] = row.get("transitions") if stochastic else None
    if not base["nodal_counters_applicable"]:
        for key in ("node_updates", "transport_sweeps", "edge_transfers"):
            result[key] = None
    if not stochastic:
        result["seed"] = None
        for key in ("walks", "total_walks", "completed_walks", "walks_per_estimate"):
            if key in result:
                result[key] = None
    return result


def collect_stage(root, stage):
    folder = root / stage
    manifest = json.loads((folder / "manifest.json").read_bytes())
    if not manifest.get("complete") or manifest.get("source_after") != manifest.get(
        "source_before"
    ):
        raise ValueError(f"Incomplete or changed-source stage: {stage}")
    if manifest["source_before"]["dirty"]:
        raise ValueError(f"Dirty measured source: {stage}")
    if manifest["completed_jobs"] != manifest["method_order"]:
        raise ValueError(f"Missing or reordered jobs: {stage}")
    targets, trials = [], []
    for name, payload, _ in read_jobs(folder, "study"):
        study = payload["study"]
        metadata = study["metadata"]
        commit = (
            metadata["source"]["commit"]
            if stage.startswith("frozen")
            else metadata["commit"]
        )
        if commit != manifest["source_before"]["commit"]:
            raise ValueError(f"Job source differs from manifest: {stage}/{name}")
        base = context(stage, name, metadata, payload["process"])
        base["source_commit"] = commit
        current = [applicable(row, base) for row in study["trials"]]
        trials.extend(current)
        for row in study["rows"]:
            selected = [
                t
                for t in current
                if (t["phase"], t["candidate"]) == (row["phase"], row["candidate"])
            ]
            if row["status"] == "confirmed" and (
                row["phase"] != "final"
                or len(selected) != row["requested_repeats"]
                or any(t["status"] != "completed" for t in selected)
            ):
                raise ValueError(
                    f"Confirmed target lacks complete final trials: {name}"
                )
            targets.append(
                dict(
                    **applicable(row, base),
                    seconds_std=(
                        stdev(t["seconds"] for t in selected)
                        if len(selected) > 1
                        else None
                    ),
                    mean_preparation_seconds=(
                        row["preparation_seconds"] / len(selected) if selected else None
                    ),
                    mean_construction_seconds=(
                        row["construction_seconds"] / len(selected)
                        if selected and row.get("construction_seconds") is not None
                        else None
                    ),
                    study_pilot_seconds=study["pilot_seconds"],
                    study_final_seconds=study["final_seconds"],
                )
            )
    return targets, trials


def cost_gate(root):
    rows = []
    for stage in STAGES[:2]:
        _, trials = collect_stage(root, stage)
        rows.append(
            dict(
                stage=stage,
                attempts=len(trials),
                solver_seconds=sum(t["seconds"] for t in trials),
                statuses=dict(Counter(t["status"] for t in trials)),
                source_commit=trials[0]["source_commit"],
            )
        )
    if len({r["source_commit"] for r in rows}) != 1:
        raise ValueError("Small stages must use the same integrated source")
    total = sum(r["solver_seconds"] for r in rows)
    return dict(
        recorded_utc=utc_now(),
        stages=rows,
        solver_seconds_all_attempts=total,
        threshold_seconds=120,
        full_n16_allowed=total <= 120,
    )


def report(root):
    decision = json.loads((root / "cost-decision.json").read_bytes())
    targets, trials, stages = [], [], []
    for stage in STAGES:
        if stage == "full-n16" and not decision["full_n16_allowed"]:
            if (root / stage).exists():
                raise ValueError("Large stage was run despite rejected cost gate")
            stages.append(dict(stage=stage, status="not_run_due_to_cost"))
            continue
        rows, attempts = collect_stage(root, stage)
        targets.extend(rows)
        trials.extend(attempts)
        stages.append(
            dict(
                stage=stage,
                status="measured",
                attempts=len(attempts),
                solver_seconds=sum(t["seconds"] for t in attempts),
                statuses=dict(Counter(t["status"] for t in attempts)),
            )
        )
    if len({t["source_commit"] for t in trials}) != 1:
        raise ValueError("Stages must use one integrated source")
    actual_gate = cost_gate(root)
    for key in (
        "stages",
        "solver_seconds_all_attempts",
        "threshold_seconds",
        "full_n16_allowed",
    ):
        if actual_gate[key] != decision[key]:
            raise ValueError("Recorded gate differs from preserved small trials")
    for row in targets:
        for baseline in ("lod", "mc"):
            match = next(
                (
                    r
                    for r in targets
                    if (r["stage"], r["scenario"], r["target_pct"], r["method"])
                    == (row["stage"], row["scenario"], row["target_pct"], baseline)
                ),
                None,
            )
            row["time_ratio_to_" + baseline] = (
                row["seconds"] / match["seconds"]
                if match and row["status"] == match["status"] == "confirmed"
                else None
            )
    for name, rows in (("targets", targets), ("trials", trials), ("stages", stages)):
        (root / (name + ".csv")).write_bytes(csv_bytes(rows))
    (root / "totals.json").write_bytes(
        json_bytes(
            dict(
                attempts=len(trials),
                statuses=dict(Counter(t["status"] for t in trials)),
                targets=dict(Counter(r["status"] for r in targets)),
                solver_seconds=sum(t["seconds"] for t in trials),
                stages=stages,
            )
        )
    )
    for scope in ("one_frozen_system", "complete_nonlinear_history"):
        rows = [r for r in targets if r["scope"] == scope]
        title = (
            "Одна замороженная система: эталон — прямое сеточное решение"
            if scope == "one_frozen_system"
            else "Полная нелинейная история: эталон — аналитическое решение"
        )
        (root / (scope + ".md")).write_text(
            "# " + title + "\n\n"
            "Ошибка — средняя отдельных полных запусков; время цели только при её подтверждении. "
            "Затраты/пути относятся к выбранной серии и повторяются между целями. "
            "Для суммы используйте trials.csv. RSS — пик процесса со всеми повторами, не одного решения. "
            "Неприменимые/неизвестные показатели отмечены прочерком. "
            "В CSV preparation_seconds/construction_seconds — суммы серии; "
            "mean_preparation_seconds/mean_construction_seconds — средние на записанную попытку. "
            "Подробные фаза, разброс и единицы счётчиков сохранены в CSV.\n\n"
            + markdown(
                rows,
                {
                    "stage": "Этап",
                    "scenario": "Сценарий",
                    "method": "Метод",
                    "target_pct": "Цель, %",
                    "budget": "Бюджет",
                    "completed_repeats": "Завершено",
                    "requested_repeats": "Всего",
                    "relative_error_pct": "Ошибка, %",
                    "seconds": "Время цели, с",
                    "elapsed_seconds": "Затрачено, с",
                    "workspace_bytes": "Буферы, байт",
                    "process_peak_rss_bytes": "RSS процесса, байт",
                    "total_walks": "Начато путей",
                    "completed_walks": "Завершено путей",
                    "virtual_particle_moves": "Виртуальных перемещений",
                    "node_updates": "Обновлений узлов",
                    "edge_transfers": "Операций по рёбрам",
                    "edge_work_kind": "Смысл операций по рёбрам",
                    "status": "Статус",
                    "reason": "Причина",
                },
            )
        )
    plots(root, targets)


def plots(root, targets):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, LogLocator, NullLocator

    colors = dict(zip(METHODS, plt.get_cmap("tab10").colors, strict=False))
    for stage, target_pct in (
        ("frozen-n8", 1),
        ("frozen-n8", 5),
        ("full-n8", 5),
        ("full-n16", 5),
    ):
        frozen = stage.startswith("frozen")
        cases = list(
            dict.fromkeys(r["scenario"] for r in targets if r["stage"] == stage)
        )
        if not cases:
            continue
        columns = min(2, len(cases))
        rows = (len(cases) + columns - 1) // columns
        fig, axes = plt.subplots(
            rows, columns, figsize=(6.5 * columns, 3.5 * rows), squeeze=False
        )
        for ax, case in zip(axes.flat, cases):
            selected = [
                r
                for r in targets
                if r["stage"] == stage
                and r["scenario"] == case
                and r["target_pct"] == target_pct
            ]
            for row in selected:
                if row["status"] == "confirmed":
                    ax.scatter(
                        row["seconds"] * (1000 if frozen else 1),
                        row["relative_error_pct"],
                        color=colors[row["method"]],
                        label=METHODS[row["method"]],
                    )
            ax.set(
                title=case,
                xlabel=f"Полное время при цели {target_pct}%, "
                + ("мс" if frozen else "с"),
                ylabel="Средняя ошибка, %",
                xscale="linear" if frozen else "log",
            )
            if frozen:
                confirmed_times = [
                    r["seconds"] * 1000 for r in selected if r["status"] == "confirmed"
                ]
                ax.set_xlim(0, 1.1 * max(confirmed_times, default=1))
            else:
                ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
                ax.xaxis.set_major_formatter(
                    FuncFormatter(lambda value, _: f"{value:g}")
                )
            ax.xaxis.set_minor_locator(NullLocator())
            ax.grid(alpha=0.2)
            if ax.collections:
                ax.legend(fontsize=8)
            else:
                ax.text(
                    0.5,
                    0.5,
                    "Нет подтверждённых целей",
                    transform=ax.transAxes,
                    ha="center",
                )
        fig.suptitle(
            f"{stage} · цель {target_pct}% · полные подтверждающие серии; отказы в таблицах"
        )
        fig.tight_layout()
        suffix = "-target1" if target_pct == 1 else ""
        fig.savefig(root / (stage + suffix + ".png"), dpi=150)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--gate",
        action="store_true",
        help="Record n16 cost decision after both n8 stages",
    )
    args = parser.parse_args()
    if args.gate:
        path = args.root / "cost-decision.json"
        if path.exists() or (args.root / "full-n16").exists():
            raise ValueError("Record the decision once, before n16 starts")
        decision = cost_gate(args.root)
        path.write_bytes(json_bytes(decision))
        print(json.dumps(decision, indent=2))
    else:
        report(args.root)


if __name__ == "__main__":
    main()
