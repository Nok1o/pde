"""Rebuild combined tables and figures from preserved raw study JSON."""

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean, stdev

from pdelab.heat3d_experiment import csv_bytes

METHODS = {
    "lod": "ЛОМ",
    "block": "Матричная прогонка",
    "mc": "Обычный MC",
    "mc_control": "Контрольный MC",
    "mc_grouped": "Групповой MC",
    "mc_control_grouped": "Групповой контрольный MC",
}
CASES = {
    "sine-centre-kind1": "Синус, центр, все грани 1 рода",
    "polynomial-kind1": "Полином, все грани 1 рода",
    "polynomial-one-kind3": "Полином, x0 3 рода",
    "polynomial-all-kind3": "Полином, все грани 3 рода",
}
STATUS = {
    "confirmed": "Достигнута",
    "not_reached": "Нет запаса точности",
    "pilot_incomplete": "Неполная предварительная серия",
}


def read_jobs(folder, key):
    manifest = json.loads((folder / "manifest.json").read_bytes())
    if (
        not manifest["complete"]
        or manifest["source_before"] != manifest["source_after"]
    ):
        raise ValueError(f"Incomplete or changed-source measurement in {folder}")
    return [
        (name, json.loads((folder / (name + ".json")).read_bytes()), key)
        for name in manifest["completed_jobs"]
    ]


def comparison_rows(jobs):
    rows = []
    for name, payload, key in jobs:
        study = payload[key]
        for row in study["rows"]:
            selected = [
                t
                for t in study["trials"]
                if (t["phase"], t["candidate"]) == (row["phase"], row["candidate"])
            ]
            rows.append(
                {
                    "n": study["metadata"]["configuration"]["n"],
                    "case": name.split("--")[0],
                    **row,
                    "seconds_std": stdev(t["seconds"] for t in selected),
                    "observed_preparation_mean": mean(
                        t["preparation_seconds"] for t in selected
                    ),
                    "study_pilot_seconds": study["pilot_seconds"],
                    "study_final_seconds": study["final_seconds"],
                    "study_process_peak_rss_bytes": payload["process"][
                        "peak_rss_bytes_before_export"
                    ],
                    "source_commit": study["metadata"]["commit"],
                }
            )
    return rows


def group_rows(jobs):
    rows = []
    lookup = {name: payload[key] for name, payload, key in jobs}
    for name, payload, key in jobs:
        case, variant = name.split("--")
        result = payload[key]
        summary = result["summary"]
        base = "mc_control" if "control" in variant else "mc"
        baseline = lookup[case + "--" + base]["summary"]
        ratio = None
        if summary["complete"] and baseline["complete"]:
            ratio = (
                summary["completed_mean_seconds"] / baseline["completed_mean_seconds"]
            )
        rows.append(
            {
                "case": case,
                "variant": variant,
                **summary,
                "time_ratio_to_ordinary": ratio,
                "process_peak_rss_bytes": payload["process"][
                    "peak_rss_bytes_before_export"
                ],
            }
        )
    return rows


def fmt(value, scale=1):
    return "—" if value is None else f"{value * scale:.5g}"


def tables(rows, groups, root):
    (root / "comparison.csv").write_bytes(csv_bytes(rows))
    (root / "groups.csv").write_bytes(csv_bytes(groups))
    text = [
        "# Основная таблица 1/5/10%",
        "",
        "Ошибка — среднее относительных ошибок отдельных полных запусков. Время достижения",
        "приведено для подтверждённой цели. Затрачено — время всех попыток указанной серии.",
        "Пути считаются за всю серию, начатые и завершённые отдельно. Одна серия",
        "может обслуживать несколько целей, её затраты в этих строках повторяются.",
        "Буферы — оценка решателя. RSS — измеренный пик процесса исследования",
        "со всеми предварительными и финальными повторами, библиотеками и JIT.",
        "В CSV дополнительно записаны фаза, причина, подготовка, разброс времени",
        "и суммарные затраты подбора. Для групповых методов B=128.",
        "",
    ]
    for n, case in dict.fromkeys((r["n"], r["case"]) for r in rows):
        text.extend(
            [
                f"## n={n}, {CASES[case]}",
                "",
                "| Метод | Цель, % | Бюджет | Полных повторов | Ошибка, % | Время, мс | Затрачено, с | Буферы / RSS, МиБ | Пути начато / завершено | Статус |",
                "|---|---:|---:|---:|---:|---:|---:|---|---|---|",
            ]
        )
        for r in rows:
            if (r["n"], r["case"]) != (n, case):
                continue
            paths = (
                f"{r['total_walks']} / {r['completed_walks']}"
                if r["walks_applicable"]
                else "неприменимо"
            )
            memory = f"{fmt(r['workspace_bytes'], 1/2**20)} / {fmt(r['study_process_peak_rss_bytes'], 1/2**20)}"
            text.append(
                f"| {METHODS[r['method']]} | {r['target_pct']:g} | {r['budget']:g} | "
                f"{r['completed_repeats']}/{r['requested_repeats']} | {fmt(r['relative_error_pct'])} | "
                f"{fmt(r['seconds'], 1000)} | {fmt(r['elapsed_seconds'])} | {memory} | {paths} | "
                f"{STATUS.get(r['status'], r['status'])} |"
            )
        text.append("")
        failures = sorted(
            {
                r["reason"]
                for r in rows
                if (r["n"], r["case"]) == (n, case) and r["status"] != "confirmed"
            }
        )
        text.extend(f"- {reason}" for reason in failures)
        text.append("")
    (root / "comparison.md").write_text("\n".join(text).rstrip() + "\n")
    text = [
        "# Размеры групп при N=1024",
        "",
        "R=5, одинаковые seed между вариантами. Размеры B заданы заранее.",
        "Отношение времени больше 1 означает увеличение полного времени.",
        "Средняя ошибка и время полного решения доступны при завершении всех пяти повторов.",
        "Время попыток включает отказы. Разброс и детальные счётчики сохранены в CSV.",
        "",
        "| Случай | Вариант | Завершено | Ошибка, % | Полное время, мс | Отношение к обычным путям | Все попытки, с | Начато / завершено путей |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for r in groups:
        text.append(
            f"| {CASES[r['case']]} | {r['variant']} | {r['completed_repeats']}/5 | "
            f"{fmt(r['relative_error_pct'])} | {fmt(r['completed_mean_seconds'],1000)} | "
            f"{fmt(r['time_ratio_to_ordinary'])} | {fmt(r['seconds_all_attempts'])} | "
            f"{r['walks']} / {r['completed_walks']} |"
        )
    (root / "groups.md").write_text("\n".join(text) + "\n")


def plots(jobs, groups, root):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

    scenarios = list(
        dict.fromkeys(
            (p[k]["metadata"]["configuration"]["n"], name.split("--")[0])
            for name, p, k in jobs
        )
    )
    colors = dict(zip(METHODS, plt.get_cmap("tab10").colors, strict=False))
    fig, axes = plt.subplots(3, 2, figsize=(13, 12), squeeze=False)
    for ax, (n, case) in zip(axes.flat, scenarios, strict=True):
        for name, payload, key in jobs:
            s = payload[key]
            if (s["metadata"]["configuration"]["n"], name.split("--")[0]) != (n, case):
                continue
            method = s["metadata"]["method"]
            for phase, marker in (("pilot", "x"), ("final", "o")):
                complete = [
                    r for r in s["series"] if r["complete"] and r["phase"] == phase
                ]
                ax.scatter(
                    [r["mean_seconds"] for r in complete],
                    [r["relative_error_pct"] for r in complete],
                    color=colors[method],
                    marker=marker,
                    s=45,
                )
        ax.set(
            title=f"n={n} · {CASES[case]}",
            xlabel="Полное время, с",
            ylabel="Средняя ошибка, %",
        )
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.grid(alpha=0.2)
    handles = [
        Line2D([], [], color=colors[m], marker="o", linestyle="", label=label)
        for m, label in METHODS.items()
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3)
    fig.suptitle("Полные серии · × подбор, ● независимая финальная проверка")
    fig.tight_layout(rect=(0, 0.07, 1, 0.97))
    fig.savefig(root / "error-time.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for ax, case in zip(axes.flat, CASES, strict=True):
        for base, label in (
            ("mc_grouped", "Обычный MC"),
            ("mc_control_grouped", "Контрольный MC"),
        ):
            points = [
                r
                for r in groups
                if r["case"] == case and r["variant"].startswith(base + "-")
            ]
            ratios = [r["time_ratio_to_ordinary"] for r in points]
            if all(v is not None for v in ratios):
                ax.plot(
                    [str(r["batch_size"]) for r in points],
                    ratios,
                    "o-",
                    label=label,
                    color="tab:blue" if base == "mc_grouped" else "tab:orange",
                )
        ax.axhline(1, color="black", linewidth=0.8)
        ax.set(
            title=CASES[case],
            xlabel="Размер группы B",
            ylabel="Время группы / обычных путей",
        )
        ax.grid(alpha=0.2)
        ax.legend()
    fig.suptitle("N=1024, R=5 · отношения только для полностью завершённых серий")
    fig.tight_layout()
    fig.savefig(root / "group-sizes.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path("benchmarks/results/heat3d-six-methods")
    )
    args = parser.parse_args()
    jobs = read_jobs(args.root / "n8", "study") + read_jobs(args.root / "n16", "study")
    fixed = read_jobs(args.root / "groups", "fixed_budget")
    rows = comparison_rows(jobs)
    groups = group_rows(fixed)
    tables(rows, groups, args.root)
    plots(jobs, groups, args.root)
    print("Targets", dict(Counter(r["status"] for r in rows)))
    print(
        "Runs",
        dict(Counter(t["status"] for _, p, k in jobs + fixed for t in p[k]["trials"])),
    )


if __name__ == "__main__":
    main()
