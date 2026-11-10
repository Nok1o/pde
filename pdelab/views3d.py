"""Срезы настоящих 3D-полей, без заполнения неизвестных узлов MC."""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .solvers3d import METHODS_3D

PLANES = {"xy": (0, "x", "y", "z"), "xz": (1, "x", "z", "y"), "yz": (2, "y", "z", "x")}


def extract_slice(values, plane, index):
    if plane not in PLANES:
        raise ValueError("Плоскость: xy, xz или yz")
    if (
        not np.isfinite(index)
        or int(index) != index
        or not 0 <= index < values.shape[0]
    ):
        raise ValueError("Индекс среза должен быть целым от 0 до n")
    return np.take(values, int(index), axis=PLANES[plane][0])


def slice_figure3d(experiment, plane="xy", index=None, error=False):
    index = experiment.config.n // 2 if index is None else index
    reference = experiment.reference
    entries = (
        []
        if error
        else [
            (
                (
                    "Точное решение"
                    if experiment.reference_kind == "analytic"
                    else "Сеточный эталон"
                ),
                reference,
            )
        ]
    )
    entries += [
        (
            METHODS_3D[method],
            np.abs(result.values - reference) if error else result.values,
        )
        for method, result in experiment.results.items()
    ]
    slices = [extract_slice(values, plane, index) for _, values in entries]
    finite = np.concatenate([values[np.isfinite(values)] for values in slices])
    minimum = 0.0 if error or not finite.size else float(finite.min())
    maximum = float(finite.max()) if finite.size else 1.0
    if maximum == minimum:
        maximum = minimum + max(1e-12, abs(minimum) * 1e-6)
    figure = make_subplots(
        rows=1, cols=len(entries), subplot_titles=[name for name, _ in entries]
    )
    _, horizontal, vertical, normal = PLANES[plane]
    axis = experiment.problem.axis
    for column, values in enumerate(slices, 1):
        figure.add_trace(
            go.Heatmap(
                x=axis, y=axis, z=values, coloraxis="coloraxis", hoverongaps=False
            ),
            row=1,
            col=column,
        )
        # Маркеры делают одиночные внутренние оценки видимыми на пустом срезе.
        if np.isnan(values).any():
            rows, columns = np.where(np.isfinite(values[1:-1, 1:-1]))
            rows, columns = rows + 1, columns + 1
            if len(rows):
                figure.add_trace(
                    go.Scatter(
                        x=axis[columns],
                        y=axis[rows],
                        mode="markers",
                        showlegend=False,
                        marker=dict(
                            size=7,
                            color=values[rows, columns],
                            coloraxis="coloraxis",
                            line=dict(width=1, color="white"),
                        ),
                    ),
                    row=1,
                    col=column,
                )
        figure.update_xaxes(
            title_text=horizontal,
            range=[0, 1],
            constrain="domain",
            dtick=0.5,
            row=1,
            col=column,
        )
        figure.update_yaxes(
            title_text=vertical if column == 1 else None,
            range=[0, 1],
            constrain="domain",
            dtick=0.5,
            showticklabels=column == 1,
            scaleanchor="x" if column == 1 else f"x{column}",
            row=1,
            col=column,
        )
    figure.update_layout(
        height=400,
        title=f"{'Абсолютная ошибка' if error else 'Решение'} · {plane}, {normal}={index / experiment.config.n:.4g}",
        coloraxis=dict(
            colorscale="Magma" if error else "Viridis",
            cmin=minimum,
            cmax=maximum,
            colorbar=dict(thickness=12, len=0.8),
        ),
        plot_bgcolor="#eeeeee",
        margin=dict(l=30, r=30, t=80, b=35),
    )
    return figure


def residual_figure3d(result):
    values = result.extras["relative_residuals"]
    figure = go.Figure(
        go.Scatter(
            x=np.arange(len(values)),
            y=np.where(values > 0, values, np.nan),
            mode="lines",
            name="ЛОМ",
        )
    )
    figure.add_hline(y=result.extras["tolerance"], line_dash="dash")
    figure.update_layout(
        xaxis_title="Итерация",
        yaxis_title="Относительная невязка",
        yaxis_type="log",
        height=350,
    )
    return figure
