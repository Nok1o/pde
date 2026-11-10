"""Static figures, real solver animations and self-contained interactive plots."""

import csv
import html
import io
import json
from dataclasses import asdict
from pathlib import Path

import matplotlib
import numpy as np
import plotly.graph_objects as go
from PIL import Image

from .solvers import METHODS

# Выбираем backend без окна до импорта pyplot: графики сохраняются в файлы.
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

COLORS = {
    "wos": "#ec4899",
    "graded": "#14b8a6",
    "lod_serial": "#3156a3",
    "mc": "#168477",
    "mc_control": "#b74167",
    "reuse": "#d77923",
    "layers": "#8b53a3",
}
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "axes.titleweight": "bold",
        "savefig.dpi": 150,
    }
)


def _limits(exp):
    values = np.concatenate(
        [exp.comparison.field_values.ravel()]
        + [r.values[np.isfinite(r.values)] for r in exp.results.values()]
    )
    lo, hi = float(values.min()), float(values.max())
    return (lo, hi) if lo < hi else (lo - 0.5, hi + 0.5)


def _error(r, exp):
    error = np.full_like(exp.reference, np.nan)
    y, x = exp.points.T
    error[y, x] = r.values[y, x] - exp.comparison.values[y, x]
    return error


def field_plot(exp, kind="value"):
    ncols = 3
    nrows = 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(13.8, 8.2), layout="constrained")
    ref = exp.comparison
    items = [(ref.field_label, ref.field_values)] + [
        (METHODS[m], r.values if kind == "value" else np.abs(_error(r, exp)))
        for m, r in exp.results.items()
    ]
    lo, hi = (
        _limits(exp)
        if kind == "value"
        else (
            0,
            max(
                1e-12,
                max(np.nanmax(np.abs(_error(r, exp))) for r in exp.results.values()),
            ),
        )
    )
    cmap = plt.get_cmap("viridis" if kind == "value" else "magma").with_extremes(
        bad="#e8edf0"
    )
    if kind == "error":
        items[0] = (
            "Эталон: ошибка = 0",
            np.where(np.isfinite(ref.values), 0.0, np.nan),
        )
    for ax, (name, a) in zip(axes.ravel(), items):
        im = ax.imshow(
            a,
            origin="lower",
            extent=(
                -0.5 / exp.problem.n,
                1 + 0.5 / exp.problem.n,
                -0.5 / exp.problem.n,
                1 + 0.5 / exp.problem.n,
            ),
            vmin=lo,
            vmax=hi,
            cmap=cmap,
            interpolation="nearest",
        )
        ax.set(xlabel="x", ylabel="y", title=name, xlim=(0, 1), ylim=(0, 1))
        if len(exp.points) <= 9:
            y, x = exp.points.T
            ax.scatter(
                x / exp.problem.n,
                y / exp.problem.n,
                facecolors="none",
                edgecolors="white",
                s=95,
                linewidths=1.2,
            )
    for ax in axes.ravel()[len(items) :]:
        ax.set_visible(False)
    fig.colorbar(
        im,
        ax=list(axes.ravel()),
        shrink=0.8,
        label="Температура, усл. ед." if kind == "value" else "|u − эталон|",
    )
    fig.suptitle(
        ("Температурное поле" if kind == "value" else "Ошибка в запрошенных узлах")
        + f" · {exp.problem.name}",
        fontsize=15,
    )
    fig.supxlabel(
        "Серый = не вычислялось. Общая шкала. Эталон ошибок: " + ref.label + ".",
        fontsize=9,
    )
    return fig


def profile_plot(exp):
    fig, ax = plt.subplots(figsize=(10, 5), layout="constrained")
    y = int(exp.points[0, 0])
    ref = exp.comparison
    ax.plot(
        exp.problem.axis,
        ref.field_values[y],
        color="#172b40",
        label=ref.field_label,
        lw=2,
    )
    if ref.kind == "analytic_point":
        ax.scatter(
            [exp.points[0, 1] / exp.problem.n],
            [ref.values[tuple(exp.points[0])]],
            marker="*",
            s=110,
            color="#172b40",
            label=ref.label,
            zorder=5,
        )
    markers = ["o", "s", "D", "^", "v"]
    for (m, r), marker in zip(exp.results.items(), markers):
        mask = np.isfinite(r.values[y])
        mask[[0, -1]] = False
        se = np.nan_to_num(r.stderr[y, mask], nan=0.0)
        ax.errorbar(
            exp.problem.axis[mask],
            r.values[y, mask],
            yerr=1.96 * se,
            fmt=marker,
            markersize=5,
            capsize=3,
            color=COLORS[m],
            label=METHODS[m],
            alpha=0.85,
        )
    ax.set(
        xlabel="x",
        ylabel="Температура, усл. ед.",
        title=f"Горизонтальный срез y={y/exp.problem.n:.3f}",
    )
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)
    fig.supxlabel(
        "Показаны только рассчитанные узлы. Усы: приближённые ±1,96 SE; для слоёв SE по полным независимым запускам.",
        fontsize=8,
    )
    return fig


def metrics_plot(exp):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    rows = exp.metrics
    names = [r["name"].replace(" · ", "\n") for r in rows]
    colors = [COLORS[r["method"]] for r in rows]
    axes[0].bar(names, [r["seconds"] * 1000 for r in rows], color=colors)
    axes[1].bar(names, [max(r["rmse"], 1e-14) for r in rows], color=colors)
    axes[1].set_yscale("log")
    for ax in axes:
        ax.tick_params(axis="x", labelsize=8)
        ax.grid(axis="y", alpha=0.2)
    axes[0].set(ylabel="Время решателя, мс", title="Один запуск: время")
    axes[1].set(
        ylabel="RMSE по запрошенным узлам", title="Один запуск: ошибка к эталону"
    )
    fig.supxlabel(
        "Эталон: "
        + exp.comparison.label
        + ".\nОдинаковый бюджет путей не означает одинаковую точность; используйте серию повторений.",
        fontsize=9,
    )
    return fig


def convergence_plot(exp, interactive=False):
    if "mc" not in exp.results:
        return None
    samples = exp.results["mc"].extras["samples"]
    counts = np.arange(1, len(samples) + 1)
    means = np.cumsum(samples) / counts
    variances = np.maximum(
        0,
        (np.cumsum(samples * samples) - counts * means**2) / np.maximum(1, counts - 1),
    )
    se = np.sqrt(variances / counts)
    plot_indices = (
        np.unique(
            np.geomspace(
                min(16, len(samples)), len(samples), min(160, len(samples))
            ).astype(int)
        )
        - 1
    )
    reference_value = float(exp.comparison.values[tuple(exp.points[0])])
    sample_counts = counts[plot_indices]
    sample_means = means[plot_indices]
    sample_errors = se[plot_indices]
    if interactive:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=sample_counts,
                y=sample_means + 1.96 * sample_errors,
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=sample_counts,
                y=sample_means - 1.96 * sample_errors,
                line=dict(width=0),
                fill="tonexty",
                fillcolor="rgba(22,132,119,.16)",
                name="±1,96 SE",
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=sample_counts,
                y=sample_means,
                name="Среднее MК",
                line=dict(color=COLORS["mc"]),
            )
        )
        fig.add_hline(
            y=reference_value, line_dash="dash", annotation_text=exp.comparison.label
        )
        fig.update_layout(
            template="plotly_white",
            xaxis_type="log",
            xaxis_title="Число траекторий N",
            yaxis_title="Оценка u",
            title="Сходимость в первой запрошенной точке",
        )
        return fig
    fig, ax = plt.subplots(figsize=(10, 5), layout="constrained")
    ax.fill_between(
        sample_counts,
        sample_means - 1.96 * sample_errors,
        sample_means + 1.96 * sample_errors,
        color=COLORS["mc"],
        alpha=0.2,
        label="±1,96 SE",
    )
    ax.plot(sample_counts, sample_means, color=COLORS["mc"], label="Среднее MC")
    ax.axhline(reference_value, color="#172b40", ls="--", label=exp.comparison.label)
    ax.set(
        xscale="log",
        xlabel="Число траекторий N",
        ylabel="Оценка u",
        title="Накопление оценки в первой запрошенной точке",
    )
    ax.legend()
    ax.grid(alpha=0.2)
    fig.supxlabel(
        "Приближённая поточечная неопределённость. SE не включает погрешность сетки; полоса не одновременная.",
        fontsize=8,
    )
    return fig


def interactive_field(exp, kind="value"):
    entries = [(exp.comparison.field_label, exp.comparison.field_values)]
    if kind == "value":
        entries += [(METHODS[m], r.values) for m, r in exp.results.items()]
    elif kind == "error":
        entries = [(METHODS[m], np.abs(_error(r, exp))) for m, r in exp.results.items()]
    elif kind == "se":
        entries = [
            (METHODS[m], r.stderr)
            for m, r in exp.results.items()
            if not m.startswith("lod_")
        ]
    elif kind == "count":
        entries = [
            (
                METHODS[m],
                np.where(r.sample_count > 0, r.sample_count.astype(float), np.nan),
            )
            for m, r in exp.results.items()
            if not m.startswith("lod_")
        ]
    else:
        raise ValueError(kind)
    if not entries:
        return None
    lo, hi = (
        _limits(exp)
        if kind == "value"
        else (0, max(1e-12, max(float(np.nanmax(a)) for _, a in entries)))
    )
    fig = go.Figure()
    for i, (name, a) in enumerate(entries):
        fig.add_trace(
            go.Heatmap(
                x=exp.problem.axis,
                y=exp.problem.axis,
                z=a,
                zmin=lo,
                zmax=hi,
                colorscale="Viridis" if kind == "value" else "Magma",
                name=name,
                visible=i == 0,
                hoverongaps=False,
                colorbar=dict(
                    title={
                        "value": "u",
                        "error": "|u − эталон|",
                        "se": "SE",
                        "count": "Наблюдений",
                    }[kind]
                ),
                hovertemplate="x=%{x:.3f}<br>y=%{y:.3f}<br>значение=%{z:.5g}<extra>"
                + name
                + "</extra>",
            )
        )
    buttons = [
        dict(
            label=name,
            method="update",
            args=[{"visible": [i == j for j in range(len(entries))]}, {"title": name}],
        )
        for i, (name, _) in enumerate(entries)
    ]
    fig.update_layout(
        template="plotly_white",
        title=entries[0][0],
        updatemenus=[dict(buttons=buttons, x=0, y=1.17, xanchor="left")],
        margin=dict(t=95),
        xaxis_title="x",
        yaxis_title="y",
        yaxis_scaleanchor="x",
        plot_bgcolor="#e8edf0",
        height=530,
    )
    return fig


def surface_plot(exp):
    ref = exp.comparison
    fig = go.Figure(
        go.Surface(
            x=exp.problem.axis,
            y=exp.problem.axis,
            z=ref.field_values,
            colorscale="Viridis",
            opacity=0.7,
            name=ref.field_label,
            showscale=False,
        )
    )
    y, x = exp.points.T
    if ref.kind == "analytic_point":
        fig.add_trace(
            go.Scatter3d(
                x=x / exp.problem.n,
                y=y / exp.problem.n,
                z=ref.values[y, x],
                mode="markers",
                name=ref.label,
                marker=dict(size=7, color="#172b40", symbol="diamond"),
            )
        )
    for m, r in exp.results.items():
        fig.add_trace(
            go.Scatter3d(
                x=x / exp.problem.n,
                y=y / exp.problem.n,
                z=r.values[y, x],
                mode="markers",
                name=METHODS[m],
                marker=dict(size=4, color=COLORS[m]),
                error_z=dict(
                    type="data",
                    array=np.nan_to_num(1.96 * r.stderr[y, x], nan=0.0),
                    visible=not m.startswith("lod_"),
                ),
            )
        )
    fig.update_layout(
        template="plotly_white",
        title="Поверхность: " + ref.field_label,
        scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="u"),
        height=570,
        legend=dict(orientation="h", y=-0.08),
    )
    return fig


def residual_plot(exp, scale="linear"):
    """The same LOM residual history on a linear or logarithmic y axis."""
    if scale not in ("linear", "log"):
        raise ValueError(scale)
    baseline = exp.results.get("lod_serial")
    if baseline is None:
        return None
    residuals = np.asarray(baseline.extras["relative_residuals"], dtype=float)
    values = np.where(residuals > 0, residuals, np.nan) if scale == "log" else residuals
    fig = go.Figure(
        go.Scatter(
            x=np.arange(len(residuals)),
            y=values,
            mode="lines+markers",
            name="Относительная невязка",
            line=dict(color=COLORS[baseline.method]),
            marker=dict(size=4),
            connectgaps=False,
            hovertemplate="Итерация %{x}<br>Относительная невязка %{y:.6g}<extra></extra>",
        )
    )
    fig.add_hline(
        y=baseline.extras["tolerance"],
        line_dash="dash",
        line_color="#d77923",
        annotation_text=f'Допуск {baseline.extras["tolerance"]:g}',
    )
    label = "Линейная шкала" if scale == "linear" else "Логарифмическая шкала"
    fig.update_layout(
        template="plotly_white",
        title=label,
        xaxis_title="Итерация",
        xaxis_type="linear",
        yaxis_title="Относительная невязка",
        yaxis_type=scale,
        height=400,
    )
    if scale == "log" and np.any(residuals == 0):
        message = "Нулевые значения не отображаются на логарифмической шкале; они видны на линейной."
        if not np.any(residuals > 0):
            message = "Невязка равна 0. Для нуля логарифмическая шкала не определена."
        fig.add_annotation(
            text=message, xref="paper", yref="paper", x=0.5, y=1.08, showarrow=False
        )
    return fig


def trajectory_plot(exp):
    path = exp.trace["points"]
    idx = np.unique(np.linspace(0, len(path) - 1, min(65, len(path))).astype(int))
    n = exp.problem.n
    fig = go.Figure(
        [
            go.Heatmap(
                x=exp.problem.axis,
                y=exp.problem.axis,
                z=exp.comparison.field_values,
                colorscale="Viridis",
                opacity=0.45,
                showscale=False,
                name=exp.comparison.field_label,
            ),
            go.Scatter(
                x=[path[0, 1] / n],
                y=[path[0, 0] / n],
                mode="lines+markers",
                line=dict(color="#ce5724", width=3),
                marker=dict(size=6),
                name="Путь",
            ),
        ]
    )
    frames = []
    for i in idx:
        complete = i == len(path) - 1
        val = exp.trace["source"][i]
        title = f"Шаг {i}: вклад источника {val:.4g}"
        if complete:
            title += (
                f'; граница {exp.trace["boundary"]:.4g}; итог {exp.trace["reward"]:.4g}'
            )
        frames.append(
            go.Frame(
                name=str(i),
                data=[go.Scatter(x=path[: i + 1, 1] / n, y=path[: i + 1, 0] / n)],
                traces=[1],
                layout=dict(title=title),
            )
        )
    fig.frames = frames
    fig.update_layout(
        template="plotly_white",
        title="Одна отдельная траектория: расчёт вклада",
        xaxis=dict(title="x", range=[0, 1]),
        yaxis=dict(title="y", range=[0, 1], scaleanchor="x"),
        height=560,
        updatemenus=[
            dict(
                type="buttons",
                buttons=[
                    dict(
                        label="▶ Путь",
                        method="animate",
                        args=[
                            None,
                            dict(
                                frame=dict(duration=140, redraw=True),
                                fromcurrent=True,
                                transition=dict(duration=0),
                            ),
                        ],
                    ),
                    dict(
                        label="Пауза",
                        method="animate",
                        args=[
                            [None],
                            dict(
                                mode="immediate", frame=dict(duration=0, redraw=False)
                            ),
                        ],
                    ),
                ],
                x=0,
                y=1.12,
            )
        ],
        sliders=[
            dict(
                currentvalue=dict(prefix="Шаг "),
                steps=[
                    dict(
                        label=str(i),
                        method="animate",
                        args=[
                            [str(i)],
                            dict(
                                mode="immediate",
                                frame=dict(duration=0, redraw=True),
                                transition=dict(duration=0),
                            ),
                        ],
                    )
                    for i in idx
                ],
            )
        ],
    )
    return fig


def _frame_image(fig):
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=90)
    plt.close(fig)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB").copy()


def reuse_plot(exp):
    """First-visit suffixes of one illustrative path, not a fabricated ensemble."""
    path = exp.trace["points"]
    n = exp.problem.n
    seen = set()
    first = []
    for i, (y, x) in enumerate(path[:-1]):
        key = (int(y), int(x))
        if key not in seen:
            seen.add(key)
            first.append(i)
    starts = [
        first[j]
        for j in np.unique(
            np.linspace(0, len(first) - 1, min(7, len(first))).astype(int)
        )
    ]
    fig = go.Figure(
        [
            go.Heatmap(
                x=exp.problem.axis,
                y=exp.problem.axis,
                z=exp.comparison.field_values,
                colorscale="Viridis",
                opacity=0.25,
                showscale=False,
                hoverinfo="skip",
            ),
            go.Scatter(
                x=path[:, 1] / n,
                y=path[:, 0] / n,
                mode="lines",
                line=dict(color="#adb7be", width=2),
                name="Весь путь",
            ),
        ]
    )
    labels = []
    for j, i in enumerate(starts):
        value = exp.trace["reward"] - exp.trace["source"][i]
        label = (
            f"Узел ({path[i,1]}, {path[i,0]}) · первый визит {i} · вклад {value:.4g}"
        )
        labels.append(label)
        fig.add_trace(
            go.Scatter(
                x=path[i:, 1] / n,
                y=path[i:, 0] / n,
                mode="lines+markers",
                line=dict(color="#d77923", width=3),
                marker=dict(size=5),
                name="Пригодный хвост",
                visible=j == 0,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[path[i, 1] / n],
                y=[path[i, 0] / n],
                mode="markers",
                marker=dict(size=14, color="#172b40", symbol="diamond"),
                name="Начало хвоста",
                visible=j == 0,
            )
        )
    buttons = []
    for j, label in enumerate(labels):
        visible = [True, True] + [k // 2 == j for k in range(2 * len(starts))]
        buttons.append(
            dict(
                label=label,
                method="update",
                args=[{"visible": visible}, {"title": label}],
            )
        )
    fig.update_layout(
        template="plotly_white",
        height=530,
        title=labels[0],
        showlegend=False,
        xaxis=dict(title="x", range=[0, 1]),
        yaxis=dict(title="y", range=[0, 1], scaleanchor="x"),
        updatemenus=[dict(buttons=buttons, x=0, y=1.2, xanchor="left")],
        margin=dict(t=110),
    )
    return fig


def layer_plot(exp):
    if "layers" not in exp.results:
        return None
    r = exp.results["layers"]
    n = exp.problem.n
    depth = r.extras["maxdepth"]
    values = r.extras["prepared_mean"]
    yy, xx = np.indices(values.shape)
    distance = np.minimum.reduce([yy, xx, n - yy, n - xx])
    lo, hi = _limits(exp)

    def heat(d):
        return go.Heatmap(
            x=exp.problem.axis,
            y=exp.problem.axis,
            z=np.where(distance <= d, values, np.nan),
            zmin=lo,
            zmax=hi,
            colorscale="Viridis",
            hoverongaps=False,
            colorbar=dict(title="u"),
        )

    fig = go.Figure(heat(0))
    fig.frames = [
        go.Frame(
            name=str(d),
            data=[heat(d)],
            layout=dict(title=f"Готово слоёв: {d} из {depth}"),
        )
        for d in range(depth + 1)
    ]
    fig.update_layout(
        template="plotly_white",
        title="Начало: известна только граница",
        height=560,
        plot_bgcolor="#e8edf0",
        xaxis=dict(title="x", range=[0, 1]),
        yaxis=dict(title="y", range=[0, 1], scaleanchor="x"),
        updatemenus=[
            dict(
                type="buttons",
                x=0,
                y=1.12,
                buttons=[
                    dict(
                        label="▶ Слои",
                        method="animate",
                        args=[
                            None,
                            dict(
                                frame=dict(duration=400, redraw=True),
                                transition=dict(duration=0),
                                fromcurrent=True,
                            ),
                        ],
                    ),
                    dict(
                        label="Пауза",
                        method="animate",
                        args=[
                            [None],
                            dict(
                                mode="immediate", frame=dict(duration=0, redraw=False)
                            ),
                        ],
                    ),
                ],
            )
        ],
        sliders=[
            dict(
                currentvalue=dict(prefix="Слой "),
                steps=[
                    dict(
                        label=str(d),
                        method="animate",
                        args=[
                            [str(d)],
                            dict(
                                mode="immediate",
                                frame=dict(duration=0, redraw=True),
                                transition=dict(duration=0),
                            ),
                        ],
                    )
                    for d in range(depth + 1)
                ],
            )
        ],
    )
    return fig


def animations(exp, out):
    path = exp.trace["points"]
    n = exp.problem.n
    lo, hi = _limits(exp)
    images = []
    for i in np.unique(np.linspace(0, len(path) - 1, min(45, len(path))).astype(int)):
        fig, ax = plt.subplots(figsize=(8, 6), layout="constrained")
        ax.imshow(
            exp.comparison.field_values,
            origin="lower",
            extent=(-0.5 / n, 1 + 0.5 / n, -0.5 / n, 1 + 0.5 / n),
            vmin=lo,
            vmax=hi,
            cmap="viridis",
            alpha=0.55,
        )
        ax.plot(
            path[: i + 1, 1] / n,
            path[: i + 1, 0] / n,
            color="#dc5426",
            lw=2,
            marker="o",
            ms=2,
        )
        ax.scatter(path[0, 1] / n, path[0, 0] / n, c="black", s=55, zorder=3)
        label = f'Шаг {i} · накопленный источник: {exp.trace["source"][i]:.3f}'
        if i == len(path) - 1:
            label += f'\nИтог пути: {exp.trace["reward"]:.3f}'
        ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="x", ylabel="y", title=label)
        fig.supxlabel(
            "Фон: "
            + exp.comparison.field_label
            + ".\nОдин случайный путь не является готовым решением.",
            fontsize=9,
        )
        images.append(_frame_image(fig))
    images[0].save(
        out / "06-траектория.gif",
        save_all=True,
        append_images=images[1:],
        duration=[130] * (len(images) - 1) + [1800],
        loop=0,
    )
    baseline = exp.results.get("lod_serial")
    if baseline and baseline.extras["history"]:
        images = []
        for it, u in baseline.extras["history"]:
            fig, ax = plt.subplots(figsize=(8, 6), layout="constrained")
            im = ax.imshow(
                u,
                origin="lower",
                extent=(-0.5 / n, 1 + 0.5 / n, -0.5 / n, 1 + 0.5 / n),
                vmin=lo,
                vmax=hi,
                cmap="viridis",
            )
            fig.colorbar(im, ax=ax, label="u")
            ax.set(
                xlim=(0, 1),
                ylim=(0, 1),
                xlabel="x",
                ylabel="y",
                title=f"Факторизованный сеточный метод · итерация {it}",
            )
            fig.supxlabel(
                "Реальные состояния после полных шагов. Интервалы между кадрами не равны времени вычислений.",
                fontsize=8,
            )
            images.append(_frame_image(fig))
        images[0].save(
            out / "07-установление-сетки.gif",
            save_all=True,
            append_images=images[1:],
            duration=[450] * (len(images) - 1) + [1600],
            loop=0,
        )


def _jsonable(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.generic):
        return o.item()
    raise TypeError(type(o).__name__)


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def export(exp, out, with_gifs=True):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "metrics.csv", exp.metrics)
    meta = {
        "config": asdict(exp.config),
        "metadata": exp.metadata,
        "metrics": exp.metrics,
        "points_yx": exp.points.tolist(),
    }
    (out / "experiment.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2, default=_jsonable),
        encoding="utf-8",
    )
    ref = exp.comparison
    arrays = {
        "reference": ref.values,
        "grid_reference": exp.reference,
        "field_reference": ref.field_values,
        "source": exp.problem.f,
        "boundary": exp.problem.boundary,
        "points_yx": exp.points,
    }
    if exp.problem.exact is not None:
        arrays["exact"] = exp.problem.exact
    for m, r in exp.results.items():
        arrays[m] = r.values
        arrays[m + "_stderr"] = r.stderr
        arrays[m + "_count"] = r.sample_count
    np.savez_compressed(out / "fields.npz", **arrays)
    for name, fig in [
        ("01-поля.png", field_plot(exp)),
        ("02-ошибки.png", field_plot(exp, "error")),
        ("03-срез.png", profile_plot(exp)),
        ("04-время-и-точность.png", metrics_plot(exp)),
        ("05-сходимость.png", convergence_plot(exp)),
    ]:
        if fig is not None:
            fig.savefig(out / name)
            plt.close(fig)
    if with_gifs:
        animations(exp, out)
    charts = [
        (
            "Поле и выбранные точки",
            "Серый фон означает, что значение не вычислялось. В выпадающем списке можно выбрать метод.",
            interactive_field(exp),
        ),
        (
            "Ошибка",
            "Эталон: " + ref.label + ". " + ref.description,
            interactive_field(exp, "error"),
        ),
        (
            "Неопределённость",
            "SE обычного MК и хвостов: по независимым исходным путям; слои: по полным независимым запускам.",
            interactive_field(exp, "se"),
        ),
        (
            "Накопление ответа",
            "Среднее MК и приближённая поточечная полоса ±1,96 SE. SE не включает погрешность сетки.",
            convergence_plot(exp, True),
        ),
        (
            "Одна траектория",
            "Кнопка запуска и ползунок показывают путь и сумму источника. Фон: "
            + ref.field_label
            + ".",
            trajectory_plot(exp),
        ),
        (
            "Сходимость ЛОМ: линейная шкала",
            "Относительная невязка ‖b − Au‖∞ / ‖b‖∞; b включает вклад границы. Пунктир — допуск остановки.",
            residual_plot(exp),
        ),
        (
            "Сходимость ЛОМ: логарифмическая шкала",
            "Те же относительные невязки; меняется только вертикальная шкала.",
            residual_plot(exp, "log"),
        ),
        (
            "Повторное использование хвостов",
            "Выберите посещённый узел: оранжевый хвост даёт одно наблюдение для него. Вклад = полный итог минус источник до первого посещения. Узлы здесь поясняющие, не обязательно цели текущего расчёта.",
            reuse_plot(exp),
        ),
        (
            "Расчёт от границы к центру",
            "Реальные средние подготовленных слоёв. Каждый новый слой использует только ранее завершённые слои. Ползунок показывает порядок расчёта, а не затраченное время.",
            layer_plot(exp),
        ),
        (
            "Вращаемая поверхность",
            "Поверхность: "
            + ref.field_label
            + ". Маркеры — действительно вычисленные узлы; пропуски не интерполируются.",
            surface_plot(exp),
        ),
    ]
    body = []
    first = True
    for title, caption, fig in charts:
        if fig is None:
            continue
        chunk = fig.to_html(
            full_html=False,
            include_plotlyjs=True if first else False,
            auto_play=False,
            config={"responsive": True, "displaylogo": False},
        )
        first = False
        body.append(
            f"<section><h2>{html.escape(title)}</h2><p>{html.escape(caption)}</p>{chunk}</section>"
        )
    value_header = "<th>Значение u</th>" if len(exp.points) == 1 else ""
    table = (
        "<table><tr><th>Метод</th>"
        + value_header
        + "<th>Время, мс</th><th>RMSE к эталону</th><th>Относительная ошибка, %</th><th>Новых путей</th><th>Шагов</th><th>Итераций</th></tr>"
    )
    for r in exp.metrics:
        value_cell = (
            f'<td>{exp.results[r["method"]].values[tuple(exp.points[0])]:.6g}</td>'
            if len(exp.points) == 1
            else ""
        )
        relative = (
            "—" if r["relative_error_pct"] is None else f'{r["relative_error_pct"]:.5g}'
        )
        table += (
            f'<tr><td>{html.escape(r["name"])}</td>'
            + value_cell
            + f'<td>{r["seconds"]*1000:.3f}</td><td>{r["rmse"]:.5g}</td><td>{relative}</td><td>{r["walks"]}</td><td>{r["steps"]}</td><td>{r["iterations"]}</td></tr>'
        )
    table += "</table>"
    document = """<!doctype html><html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>PDE Lab — результаты</title><style>body{font:16px/1.5 system-ui,sans-serif;color:#193247;background:#f3f6f8;margin:0}main{max-width:1100px;margin:auto;padding:24px}section{background:white;padding:22px;margin:20px 0;border-radius:12px}h1{font-size:30px}h2{font-size:22px}p{color:#516579}table{border-collapse:collapse;width:100%;font-size:14px}td,th{border-bottom:1px solid #ddd;padding:10px;text-align:left}.scroll{overflow:auto}a{color:#2864b1}</style><main><h1>Сеточный метод и Монте-Карло</h1>"""
    document += f"<p>{html.escape(exp.problem.name)} · сетка {exp.problem.n+1}×{exp.problem.n+1} · запрошено точек: {len(exp.points)} · seed {exp.config.seed}</p>"
    document += (
        "<section><h2>Эталон: "
        + html.escape(ref.label)
        + "</h2><p>"
        + html.escape(ref.description)
        + "</p></section>"
    )
    document += (
        '<section><h2>Что посчитано</h2><p>Один запуск, не сравнение при гарантированно одинаковой точности. JIT, эталон и визуализации исключены из времени; подготовка слоёв включена. Для выводов о скорости выполните benchmark.</p><div class="scroll">'
        + table
        + "</div><p>Относительная ошибка = 100 × ‖u − эталон‖₂ / ‖эталон‖₂ по запрошенным точкам. Если все эталонные значения нулевые, процент не определён (—); используйте RMSE.</p></section>"
    )
    document += "".join(body) + "</main></html>"
    (out / "interactive.html").write_text(document, encoding="utf-8")
    return out


def benchmark_figure(rows):
    fig = go.Figure()
    for m in METHODS:
        subset = [r for r in rows if r["method"] == m]
        budgets = sorted(set(r["budget"] for r in subset), reverse=m.startswith("lod_"))
        ts = []
        es = []
        labels = []
        for b in budgets:
            group = [r for r in subset if r["budget"] == b]
            ts.append(np.median([r["seconds"] for r in group]))
            es.append(max(1e-14, np.sqrt(np.mean([r["rmse"] ** 2 for r in group]))))
            labels.append(("отн. tol=" if m.startswith("lod_") else "N=") + str(b))
        if ts:
            fig.add_trace(
                go.Scatter(
                    x=ts,
                    y=es,
                    text=labels,
                    mode="lines+markers",
                    name=METHODS[m],
                    line=dict(color=COLORS[m]),
                    hovertemplate="%{text}<br>время=%{x:.5g} с<br>ошибка=%{y:.5g}<extra>%{fullData.name}</extra>",
                )
            )
    label = rows[0]["reference_label"] if rows else "не задан"
    fig.update_layout(
        template="plotly_white",
        title="Время — точность<br><sup>Эталон: " + html.escape(label) + "</sup>",
        xaxis_type="log",
        yaxis_type="log",
        xaxis_title="Время, с: медиана",
        yaxis_title="RMSE к эталону: RMS по повторам",
    )
    return fig


def benchmark_plot(rows, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "benchmark.csv", rows)
    fig, ax = plt.subplots(figsize=(10, 6), layout="constrained")
    for m in METHODS:
        subset = [r for r in rows if r["method"] == m]
        if not subset:
            continue
        budgets = sorted(set(r["budget"] for r in subset), reverse=m.startswith("lod_"))
        ts = []
        es = []
        labels = []
        for b in budgets:
            group = [r for r in subset if r["budget"] == b]
            ts.append(np.median([r["seconds"] for r in group]))
            es.append(max(1e-14, np.sqrt(np.mean([r["rmse"] ** 2 for r in group]))))
            labels.append(("отн. tol=" if m.startswith("lod_") else "N=") + str(b))
        ax.loglog(ts, es, "o-", color=COLORS[m], label=METHODS[m])
        ax.scatter(
            [r["seconds"] for r in subset],
            [max(1e-14, r["rmse"]) for r in subset],
            s=12,
            alpha=0.3,
            color=COLORS[m],
        )
    ax.set(
        xlabel="Время решателя, с (медиана повторений)",
        ylabel="RMSE к эталону: корень из среднего квадрата по повторам",
        title="Время — точность: разные бюджеты, независимые повторения",
    )
    if rows:
        fig.supxlabel("Эталон: " + rows[0]["reference_label"], fontsize=9)
    ax.grid(alpha=0.2, which="both")
    ax.legend(fontsize=9)
    fig.savefig(out / "benchmark.png")
    plt.close(fig)
    benchmark_figure(rows).write_html(
        out / "benchmark.html", include_plotlyjs=True, auto_open=False
    )
