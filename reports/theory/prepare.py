"""Render equations and explanatory scientific diagrams for the theory PDF."""

import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyBboxPatch

sys.path.insert(0, str(Path(__file__).parent))
from content import PAGES

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tmp/pdfs/theory/assets"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "mathtext.fontset": "dejavusans",
    }
)
BLUE = "#194c70"
TEAL = "#008b8b"
RED = "#d75b43"
GRAY = "#6b7c8b"


def fraction_braces(text):
    def token(pos):
        while text[pos].isspace():
            pos += 1
        start = pos
        if text[pos] == "{":
            depth = 1
            pos += 1
            while depth:
                depth += (text[pos] == "{") - (text[pos] == "}")
                pos += 1
            return text[start:pos], pos
        if text[pos] == "\\":
            pos += 1
            while pos < len(text) and text[pos].isalpha():
                pos += 1
        else:
            pos += 1
        return "{" + text[start:pos] + "}", pos

    result = ""
    pos = 0
    while pos < len(text):
        if text.startswith(r"\frac", pos):
            numerator, end = token(pos + 5)
            denominator, pos = token(end)
            result += r"\frac" + numerator + denominator
        else:
            result += text[pos]
            pos += 1
    return result


manifest = {}
for blocks in [b for _, b in PAGES]:
    for block in blocks:
        if block[0] != "e" or block[1] in manifest:
            continue
        key = f"equation-{len(manifest):03d}.png"
        fig = plt.figure(figsize=(11, 1))
        formula = re.sub(
            r"\\(overline|widehat|widetilde|bar|hat|sqrt)\s*([A-Za-z])",
            r"\\\1{\2}",
            block[1],
        )
        formula = formula.replace(r"\mathbb E", r"\mathbb{E}").replace(
            r"\tfrac", r"\frac"
        )
        formula = re.sub(r"\\le(?![A-Za-z])", r"\\leq", formula)
        formula = re.sub(r"\\ge(?![A-Za-z])", r"\\geq", formula)
        formula = formula.replace(r"\frac Rr", r"\frac{R}{r}").replace(
            r"\frac\pi", r"\frac{\pi}"
        )
        formula = re.sub(r"\\frac([0-9])", r"\\frac{\1}", formula)
        formula = re.sub(r"(\\frac\{[^{}]+\})([0-9A-Za-z])", r"\1{\2}", formula)
        formula = fraction_braces(formula)
        fig.text(0.01, 0.5, "$" + formula + "$", fontsize=15, va="center", color=BLUE)
        fig.savefig(
            OUT / key, dpi=240, bbox_inches="tight", pad_inches=0.04, facecolor="white"
        )
        plt.close(fig)
        manifest[block[1]] = key
(OUT / "equations.json").write_text(json.dumps(manifest, ensure_ascii=False))


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=220, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def square(ax, n=8):
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)
    ax.set_aspect("equal")
    for v in np.linspace(0, 1, n + 1):
        ax.plot([v, v], [0, 1], color="#d6e2ec", lw=0.6)
        ax.plot([0, 1], [v, v], color="#d6e2ec", lw=0.6)
    ax.plot([0, 1, 1, 0, 0], [0, 0, 1, 1, 0], color=BLUE, lw=1.4)
    ax.set_xlabel("x")
    ax.set_ylabel("y", rotation=0)


fig, ax = plt.subplots(figsize=(5.6, 3.5))
square(ax)
ax.scatter([0.375, 0.625, 0.5, 0.5], [0.5, 0.5, 0.375, 0.625], c=BLUE, s=55, zorder=5)
ax.scatter([0.5], [0.5], c=RED, s=80, zorder=6)
for xx, yy in [(0.375, 0.5), (0.625, 0.5), (0.5, 0.375), (0.5, 0.625)]:
    ax.plot([0.5, xx], [0.5, yy], c=BLUE, lw=2)
ax.annotate(
    "запрос uᵢ",
    (0.5, 0.5),
    (0.69, 0.72),
    arrowprops={"arrowstyle": "->", "color": RED},
    color=RED,
)
ax.text(0.08, 0.9, "граница: известное g", color=BLUE)
save(fig, "grid")

fig, axs = plt.subplots(1, 2, figsize=(8, 2.5))
for j, ax in enumerate(axs):
    square(ax, 6)
    for v in np.linspace(1 / 6, 5 / 6, 5):
        start, end = ((0.08, v), (0.92, v)) if j == 0 else ((v, 0.08), (v, 0.92))
        ax.annotate(
            "", end, start, arrowprops={"arrowstyle": "->", "color": TEAL, "lw": 1.5}
        )
    ax.set_title("1. Прогонки по x" if j == 0 else "2. Прогонки по y")
    ax.set_xticks([])
    ax.set_yticks([])
fig.tight_layout()
save(fig, "sweeps")

fig, ax = plt.subplots(figsize=(7.5, 2.7))
square(ax, 8)
path = (
    np.array(
        [[2, 3], [3, 3], [3, 4], [4, 4], [5, 4], [5, 5], [6, 5], [6, 6], [7, 6], [8, 6]]
    )
    / 8
)
ax.plot(path[:4, 0], path[:4, 1], color=GRAY, lw=3)
ax.plot(path[3:, 0], path[3:, 1], color=TEAL, lw=3)
ax.scatter(*path[0], s=65, c=RED, zorder=5)
ax.scatter(*path[3], s=65, c=TEAL, zorder=5)
ax.text(0.12, 0.25, "A", color=RED)
ax.text(0.49, 0.38, "B", color=TEAL)
ax.text(
    1.12,
    0.65,
    "Серый: префикс A → B\n\nБирюзовый: хвост B → граница\n\nПолная выплата минус префикс\nдаёт наблюдение из B.",
    transform=ax.transAxes,
    va="top",
    color=BLUE,
)
ax.set_xticks([])
ax.set_yticks([])
save(fig, "reuse")

fig, ax = plt.subplots(figsize=(6, 3.1))
y, x = np.indices((9, 9))
dep = np.minimum.reduce([x, y, 8 - x, 8 - y])
ax.scatter(x.ravel(), y.ravel(), c=dep.ravel(), cmap="YlGnBu", s=85, edgecolor="white")
for k in range(4):
    ax.plot(
        [k, 8 - k, 8 - k, k, k],
        [k, k, 8 - k, 8 - k, k],
        color=plt.cm.YlGnBu(0.2 + 0.18 * k),
        lw=1,
    )
ax.set_aspect("equal")
ax.axis("off")
ax.text(
    9,
    6.5,
    "0: точная граница\n\n1 → 2 → 3 → 4\n\nКаждый новый слой\nиспользует только\nготовые внешние.",
    va="top",
    color=BLUE,
)
save(fig, "layers")


def box(ax, x, y, w, h, label, color=BLUE):
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.015", ec=color, fc="#edf4f8", lw=1.4
        )
    )
    ax.text(
        x + w / 2, y + h / 2, label, ha="center", va="center", color=color, fontsize=11
    )


def arr(ax, start, end):
    ax.annotate(
        "", end, start, arrowprops={"arrowstyle": "->", "color": TEAL, "lw": 1.8}
    )


fig, ax = plt.subplots(figsize=(8, 2.2))
ax.axis("off")
box(ax, 0.01, 0.49, 0.27, 0.39, "Грубая система\nиз f и g")
box(ax, 0.36, 0.49, 0.27, 0.39, "Перенос v\nна исходную сетку")
box(ax, 0.71, 0.49, 0.27, 0.39, "MC для остатка\nf − Lₕv")
arr(ax, (0.28, 0.69), (0.36, 0.69))
arr(ax, (0.63, 0.69), (0.71, 0.69))
ax.text(
    0.5,
    0.16,
    "Ответ: v + средняя случайная поправка",
    ha="center",
    color=BLUE,
    fontsize=13,
)
save(fig, "control")

fig, ax = plt.subplots(figsize=(5.6, 3.5))
square(ax, 1)
pos = np.array([0.48, 0.5])
angles = [0.25, 1.9, 0.6]
for k, angle in enumerate(angles):
    radius = min(*pos, *(1 - pos))
    ax.add_patch(Circle(pos, radius, fill=False, color=TEAL, alpha=0.8, lw=1.4))
    nxt = pos + radius * np.array([np.cos(angle), np.sin(angle)])
    ax.annotate("", nxt, pos, arrowprops={"arrowstyle": "->", "color": RED, "lw": 1.7})
    ax.scatter(*pos, c=BLUE, s=25, zorder=5)
    if k == 0:
        ax.text(*(0.65 * pos + 0.35 * nxt + np.array([0, 0.04])), "R", color=RED)
    pos = nxt
ax.set_xticks([0, 0.5, 1])
ax.set_yticks([0, 0.5, 1])
save(fig, "wos")

fig, ax = plt.subplots(figsize=(8, 1.8))
s = np.linspace(0, 1, 17)
g = np.where(s <= 0.5, 2 * s * s, 1 - 2 * (1 - s) ** 2)
for yy, axis, label in [(1, s, "равномерная"), (0, g, "сгущённая")]:
    ax.plot([0, 1], [yy, yy], color=GRAY)
    ax.scatter(
        axis, np.full_like(axis, yy), c=TEAL if yy == 0 else BLUE, s=30, zorder=4
    )
    ax.text(-0.02, yy, label, ha="right", va="center")
ax.set_xlim(-0.01, 1.01)
ax.set_ylim(-0.5, 1.5)
ax.axis("off")
save(fig, "graded")

fig, ax = plt.subplots(figsize=(8, 2.4))
for m in range(4):
    ax.plot([0, 5], [m, m], color="#ccdbe6")
    ax.scatter(np.arange(6), np.full(6, m), s=12, c="#90a9bb")
path = np.array([[3, 3], [4, 3], [4, 2], [3, 2], [3, 1], [2, 1], [2, 0]])
for i in range(len(path) - 1):
    col = RED if path[i, 1] != path[i + 1, 1] else TEAL
    ax.annotate(
        "", path[i + 1], path[i], arrowprops={"arrowstyle": "->", "color": col, "lw": 2}
    )
ax.set_yticks(range(4), ["t₀=0", "t₁", "t₂", "t₃=T"])
ax.set_xlabel("пространственный узел (показана одна ось)")
ax.text(
    5.3,
    2.2,
    "→ по пространству\n↓ назад по времени\n\nНа t₀ выплачивается u₀.",
    color=BLUE,
    va="top",
)
ax.set_xlim(-0.1, 5.1)
save(fig, "spacetime")

fig, ax = plt.subplots(figsize=(8, 2.1))
ax.axis("off")
box(ax, 0.01, 0.33, 0.27, 0.5, "Новый слой tₘ\nпредыдущее поле uᵐ⁻¹")
box(ax, 0.36, 0.33, 0.27, 0.5, "Picard: k(vʲ)\nрешить → обновить")
box(ax, 0.71, 0.33, 0.27, 0.5, "Линейная система\nпрогонки или MC")
arr(ax, (0.28, 0.59), (0.36, 0.59))
arr(ax, (0.63, 0.59), (0.71, 0.59))
ax.text(
    0.5,
    0.06,
    "Проверки: линейная невязка → нелинейная невязка и изменение",
    ha="center",
    color=BLUE,
    fontsize=10,
)
save(fig, "nonlinear")

# Independently check the arithmetic examples used in prose.
a = (
    np.diag(np.full(3, 2.0))
    + np.diag(np.full(2, -1.0), 1)
    + np.diag(np.full(2, -1.0), -1)
)
assert np.allclose(np.linalg.solve(a, [1, 0, 1]), 1)
assert abs(np.exp(-2 * np.pi**2 * 0.01) - 0.82087) < 1e-5
assert abs(1 / (1 + 2 * np.pi**2 * 0.01) - 0.83515) < 1e-5
assert abs(100 * ((np.pi / 96) / np.sin(np.pi / 96)) ** 2 - 100 - 0.0357) < 1e-4
print(
    f"{len(PAGES)} chapters; {len(manifest)} equations; 9 diagrams; arithmetic checks passed"
)
