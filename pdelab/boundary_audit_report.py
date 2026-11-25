"""Render saved PDE-27 observations without rerunning or selecting measurements."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

from .boundary_audit3d import csv_write

LABELS = {
    "all_dirichlet": "Все 1 рода",
    "y0_robin": "y0 — 3 рода",
    "all_robin": "Все 3 рода",
}


def render(folder):
    folder = Path(folder)
    results = json.loads((folder / "results.json").read_text(encoding="utf-8"))
    attempts = json.loads((folder / "attempts.json").read_text(encoding="utf-8"))
    trials = []
    for index, attempt in enumerate(attempts):
        for trial in attempt["trials"] + attempt["failures"]:
            row = dict(
                attempt_index=index,
                case=attempt["case"],
                method=attempt["method"],
                phase=attempt["summary"]["phase"],
                budget=attempt["budget"],
                **trial,
            )
            row["values_json"] = json.dumps(row.pop("values", None))
            trials.append(row)
    csv_write(folder / "trials.csv", trials)
    rows = results["summary"]
    figures, axes = plt.subplots(2, 3, figsize=(13, 7), squeeze=False)
    for axis, case in zip(axes.flat, dict.fromkeys(a["case"] for a in attempts)):
        for method, color in (("lod_serial", "tab:blue"), ("mc", "tab:orange")):
            for phase, marker in (("pilot", "x"), ("control", "o")):
                subset = [
                    a["summary"]
                    for a in attempts
                    if a["case"] == case
                    and a["method"] == method
                    and a["summary"]["phase"] == phase
                    and a["summary"]["relative_error_pct"] is not None
                ]
                axis.scatter(
                    [a["mean_seconds"] for a in subset],
                    [a["relative_error_pct"] for a in subset],
                    c=color,
                    marker=marker,
                    label=f"{method} / {phase}",
                )
        axis.set(
            xscale="log",
            yscale="log",
            title=case,
            xlabel="Полное время запуска, с",
            ylabel="Ошибка, %",
        )
        axis.grid(alpha=0.25)
        axis.xaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
        axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.3g}"))
        axis.xaxis.set_minor_formatter(NullFormatter())
        axis.tick_params(axis="x", labelsize=8)
    axes[0, 0].legend(fontsize=7)
    figures.tight_layout()
    figures.savefig(folder / "error-time.png", dpi=150)
    plt.close(figures)
    protocol = results["protocol"]
    lines = [
        "# Ранние результаты #27: стационарные 3D-границы",
        "",
        f"n={protocol['n']}, R={protocol['repeats']}, Q={protocol['query_xyz']}, seed={protocol['root_seed']}. "
        "Две функции #24 при t=0, k=1, beta=1. Ошибка — среднее относительных L2-ошибок полных запусков. "
        "Это стационарный опыт; перенос на нелинейную временную задачу ещё не выполнен.",
        "",
        "## Цель 1%",
        "",
        "Время включает новую подготовку данных и полный решатель, исключает прогрев/JIT и внешнюю оценку ошибки. "
        "Память — оценка рабочих буферов; измеренного RSS нет. Время подбора сохранено отдельно в CSV/JSON.",
        "",
        "| Функция | Границы | Метод | Ошибка, % | Время ± s, мс | Бюджет | Пути / итерации за повтор | Буферы, МиБ | Статус |",
        "|---|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        if row["target_pct"] != 1:
            continue
        kind, variant = row["case"].split("-", 1)
        time = row.get("mean_seconds")
        error = row.get("relative_error_pct")
        matching = next(
            (
                a
                for a in attempts
                if a["case"] == row["case"]
                and a["method"] == row["method"]
                and a["budget"] == row.get("budget")
                and a["summary"]["phase"] == "control"
            ),
            None,
        )
        counter = "—"
        if matching and matching["trials"]:
            name = "walks" if row["method"] == "mc" else "inner_iterations"
            counter = (
                f"{sum(t[name] for t in matching['trials'])/len(matching['trials']):g}"
            )
        time_label = (
            f"{1000*time:.2f} ± {1000*row.get('seconds_std', 0):.2f}"
            if time is not None
            else "—"
        )
        workspace = row.get("workspace_bytes")
        workspace_label = f"{workspace/2**20:.2f}" if workspace is not None else "—"
        lines.append(
            f"| {kind} | {LABELS[variant]} | {row['method']} | "
            f"{f'{error:.3f}' if error is not None else '—'} | "
            f"{time_label} | "
            f"{row.get('budget','—')} | {counter} | "
            f"{workspace_label} | {row['status']} |"
        )
    lines.extend(
        [
            "",
            "## Два разных отношения времени",
            "",
            "ЛОМ/MC > 1 означает меньшую измеренную цену MC для одинаковой подтверждённой цели. "
            "MC/MC с 1 родом показывает изменение цены самого MC при смене границ. "
            "В этот показатель входят и различия выбранных бюджетов, и время одной траектории; "
            "он не изолирует цену отражения. Цели 5/10% и все попытки находятся в CSV/JSON.",
            "",
            "| Функция | Границы | ЛОМ / MC при 1% | MC / MC с 1 родом при 1% |",
            "|---|---|---:|---:|",
        ]
    )
    for row in results["ratios"]:
        if row["target_pct"] == 1:
            kind, variant = row["case"].split("-", 1)
            ratios = [
                f"{row[k]:.3f}" if row[k] is not None else "не подтверждено"
                for k in ("lod_over_mc", "mc_over_dirichlet_mc")
            ]
            lines.append(f"| {kind} | {LABELS[variant]} | {' | '.join(ratios)} |")
    lines.extend(
        [
            "",
            "![Ошибка–время](error-time.png)",
            "",
            "## Ограничения",
            "",
            "R=5 — пилот без статистической гарантии. Одна сетка и одна внутренняя точка; "
            "n=32 и нецентральная точка ещё не измерены. Нет вывода об универсальном выигрыше. "
            "Все выбранные здесь контрольные серии прошли, но неподходящие бюджеты пилота также сохранены. "
            "Пределы времени и пути мягкие/явные согласно protocol.json. Отказы и неполные серии проверены отдельными тестами.",
            "",
            "В файлах protocol.json/source.zip сохранены commit, признак dirty, хеши и исходники исполнявшегося кода. "
            "Незакоммиченный runner не объявлен частью main. Отчёт построен из сохранённых наблюдений отдельно, без повторного подбора.",
            "",
            "## Воспроизведение",
            "",
            "```sh",
            protocol["reproduction"],
            "python -m pdelab.boundary_audit_report runs/boundary-audit-reproduction",
            "```",
            "",
        ]
    )
    (folder / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder")
    render(parser.parse_args(argv).folder)


if __name__ == "__main__":
    main()
