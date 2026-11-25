"""Render scientific figures from the saved audit, without rerunning solvers."""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "benchmarks/results/project-audit"
OUT = ROOT / "tmp/pdfs"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)
protocol = json.loads((DATA / "protocol.json").read_text())
rows = {}
for case in protocol["cases"]:
    for row in json.loads((DATA / (case["id"] + "-refined.json")).read_text())[
        "comparisons"
    ]:
        rows[row["case"], row["method"], row["target_pct"]] = row
names = {
    "lod": "ЛОМ",
    "mc": "MC",
    "mc_control": "MC-контроль",
    "reuse": "Хвосты",
    "layers": "Слои",
    "wos": "WoS",
    "graded": "Сгущение",
}


def save(fig, name):
    fig.savefig(OUT / name, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def equation(name, lines):
    fig = plt.figure(figsize=(8, 0.53 * len(lines) + 0.15))
    for i, line in enumerate(lines):
        fig.text(
            0.5,
            1 - (i + 0.7) / (len(lines) + 0.3),
            line,
            ha="center",
            va="center",
            fontsize=15,
            color="#173b57",
        )
    save(fig, name)


equation(
    "stationary-equations.png",
    [
        r"$r^{(m)}=(L_x+L_y)u^{(m)}-f$",
        r"$(I-\tau L_y)(I-\tau L_x)\,\delta^{(m)}=\tau r^{(m)},\quad u^{(m+1)}=u^{(m)}+\delta^{(m)}$",
    ],
)
equation(
    "time-equations.png",
    [
        r"$\frac{u^{k+1}-u^k}{\Delta t}=\nabla_h\!\cdot\!\left(k(u^{k+1})\nabla_h u^{k+1}\right)+F^{k+1}$",
        r"$k(u)=1+\alpha u^2,\qquad C=1,\qquad\alpha\geq 0$",
    ],
)
equation(
    "accuracy-equation.png",
    [
        r"$\varepsilon=100\sqrt{\frac{\frac{1}{R}\sum_{r=1}^{R}\frac{1}{M}\sum_{j=1}^{M}(u_{r,j}-u_j^*)^2}{\frac{1}{M}\sum_{j=1}^{M}(u_j^*)^2}}\,,\qquad S=\frac{\overline{t}_{\mathrm{LOM}}}{\overline{t}_{\mathrm{MC}}}$",
    ],
)

with (ROOT / "docs/nonlinear-study/accuracy.csv").open(encoding="utf-8-sig") as stream:
    convergence = list(csv.DictReader(stream))
fig, axes = plt.subplots(1, 2, figsize=(9, 3.3), layout="constrained")
for ax, study, xlabel in zip(
    axes, ["space_stationary", "time"], ["Шаг пространства h", "Шаг времени Δt"]
):
    for alpha in (0, 0.5, 2):
        selected = [
            r for r in convergence if r["study"] == study and float(r["alpha"]) == alpha
        ]
        x = [
            1 / int(r["n"]) if study == "space_stationary" else float(r["dt"])
            for r in selected
        ]
        ax.loglog(x, [float(r["error"]) for r in selected], "o-", label=f"α={alpha:g}")
    ax.set(
        xlabel=xlabel,
        ylabel="Ошибка",
        title=(
            "Пространство: порядок ≈ 2"
            if study == "space_stationary"
            else "Время: порядок ≈ 1"
        ),
    )
    ax.grid(alpha=0.25, which="both")
    ax.legend(fontsize=8)
save(fig, "convergence.png")

methods = [m for m in names if m != "lod"]
case_ids = [c["id"] for c in protocol["cases"]]
fig, axes = plt.subplots(1, 2, figsize=(10, 9.7), layout="constrained")
cmap = matplotlib.colormaps["RdYlGn"].copy()
cmap.set_bad("#e8edf1")
for ax, target in zip(axes, [1.0, 5.0]):
    values = np.full((len(case_ids), len(methods)), np.nan)
    for i, case_id in enumerate(case_ids):
        for j, method in enumerate(methods):
            row = rows.get((case_id, method, target))
            if row and row["speedup"] is not None:
                values[i, j] = np.log10(row["speedup"])
    ax.imshow(
        values,
        cmap=cmap,
        norm=TwoSlopeNorm(vmin=-3, vcenter=0, vmax=1.5),
        aspect="auto",
    )
    for i, case_id in enumerate(case_ids):
        for j, method in enumerate(methods):
            row = rows.get((case_id, method, target))
            if np.isfinite(values[i, j]):
                label = (
                    f'{row["speedup"]:.2f}'
                    if row["speedup"] >= 0.01
                    else f'{row["speedup"]:.1e}'
                )
                ax.text(
                    j,
                    i,
                    label + ("*" if row["robust_win"] else ""),
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="black",
                )
    ax.set_xticks(
        range(len(methods)), [names[m] for m in methods], rotation=50, ha="right"
    )
    ax.set_yticks(range(len(case_ids)), case_ids)
    ax.set_title(f"Допустимая ошибка {target:g}%")
    ax.tick_params(length=0)
save(fig, "speedup-map.png")

fig, ax = plt.subplots(figsize=(8.4, 3.4), layout="constrained")
selected = [
    rows["S06", m, 5.0] for m in names if rows["S06", m, 5.0]["status"] == "confirmed"
]
for i, row in enumerate(selected):
    value = row["seconds"] * 1000
    ax.barh(i, value, color="#173b57" if row["method"] == "lod" else "#087e8b")
    ax.text(
        value * 1.16,
        i,
        f'{value:.2f} мс; ε={row["error_pct"]:.2f}%',
        va="center",
        fontsize=8,
    )
ax.set_yticks(range(len(selected)), [names[r["method"]] for r in selected])
ax.invert_yaxis()
ax.set(
    xscale="log",
    xlim=(0.06, 50000),
    xlabel="Время, мс — логарифмическая шкала",
    title="S06: всё поле, n=16, порог 5%",
)
ax.grid(axis="x", alpha=0.2)
save(fig, "field-cost.png")

fig, axes = plt.subplots(1, 2, figsize=(9, 3.3), layout="constrained")
for ax, case_id, target in zip(axes, ["S10", "S11"], [5.0, 10.0]):
    selected = [
        rows[case_id, m, target]
        for m in ("lod", "mc", "graded")
        if rows[case_id, m, target]["status"] == "confirmed"
    ]
    for i, row in enumerate(selected):
        value = row["seconds"] * 1000
        ax.bar(i, value, color="#173b57" if row["method"] == "lod" else "#087e8b")
        ax.text(
            i,
            value + 0.02,
            f'{value:.2f} мс\nε={row["error_pct"]:.2f}%',
            va="bottom",
            ha="center",
            fontsize=8,
        )
    ax.set_xticks(range(len(selected)), [names[r["method"]] for r in selected])
    ax.set_ylim(0, max(r["seconds"] * 1000 for r in selected) * 1.45)
    ax.set(ylabel="Время, мс", title=f"{case_id}: 3D, порог {target:g}%")
    ax.grid(axis="y", alpha=0.2)
save(fig, "boundary-cost.png")
print("Rendered seven figures to", OUT)

equation(
    "analytic-stationary.png",
    [
        r"$u^*_{\mathrm{S01,S02,S05,S06,S12}}=100\sin(\pi x)\sin(\pi y),\quad f=-2\pi^2u^*$",
        r"$u^*_{\mathrm{S03}}=1000x(1-x)y(1-y)$",
        r"$u^*_{\mathrm{S07,S08}}=\sin(\pi x)\sin(\pi y)\sin(\pi z),\quad f=-3\pi^2u^*$",
        r"$u^*_{\mathrm{S09}}=2+x+2y+\cos(2\pi x)+\frac{1}{2}\cos(2\pi y)$",
        r"$u^*_{\mathrm{S10}}=3+x+2y+3z+\cos(2\pi x)+\frac{1}{2}\cos(2\pi y)+\frac{1}{4}\cos(2\pi z)$",
    ],
)
equation(
    "analytic-time.png",
    [
        r"$u^*_{\mathrm{H01}}=e^{-2\pi^2t}\sin(\pi x)\sin(\pi y)$",
        r"$u^*_{\mathrm{H02}}=1+x+y+t(1+x)+e^t x(1-x)y(1-y)$",
        r"$u^*_{\mathrm{Q01,Q02}}=1+e^{-t}\sin(\pi x)\sin(\pi y)$",
    ],
)
