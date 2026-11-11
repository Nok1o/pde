"""Interactive fields and explicit Neumann/Robin path events."""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .boundary_solvers import METHODS_BOUNDARY

EVENT_NAMES = {
    "interior": "Внутренний узел",
    "neumann": "2-й род: вклад производной",
    "robin": "3-й род: продолжение",
    "robin_absorption": "3-й род: поглощение",
    "dirichlet": "1-й род: выплата μ",
}
EVENT_COLORS = {
    "interior": "#3156a3",
    "neumann": "#d77923",
    "robin": "#8b53a3",
    "robin_absorption": "#b74167",
    "dirichlet": "#168477",
}


def boundary_field_figure(experiment, error=False):
    entries = [("Эталон", experiment.reference)] + [
        (METHODS_BOUNDARY[m], r.values) for m, r in experiment.results.items()
    ]
    figure = go.Figure()
    for index, (name, values) in enumerate(entries):
        z = np.abs(values - experiment.reference) if error else values
        figure.add_trace(
            go.Heatmap(
                x=experiment.problem.axis,
                y=experiment.problem.axis,
                z=z,
                name=name,
                visible=index == 0,
                colorscale="Magma" if error else "Viridis",
                colorbar={"title": "|u − эталон|" if error else "u"},
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
    figure.update_layout(
        title="Эталон",
        xaxis_title="x",
        yaxis_title="z",
        height=500,
        updatemenus=[dict(buttons=buttons, x=0, y=1.17)],
    )
    figure.update_yaxes(scaleanchor="x", scaleratio=1)
    return figure


def boundary_trace_figure(history, n):
    figure = make_subplots(
        rows=1, cols=2, subplot_titles=("Траектория", "Накопленный вклад")
    )
    figure.add_trace(
        go.Scatter(
            x=[e["x"] / n for e in history],
            y=[e["y"] / n for e in history],
            mode="lines",
            line={"color": "#b1beca"},
            name="Путь",
        ),
        row=1,
        col=1,
    )
    for event, label in EVENT_NAMES.items():
        selected = [e for e in history if e["event"] == event]
        if not selected:
            continue
        figure.add_trace(
            go.Scatter(
                x=[e["x"] / n for e in selected],
                y=[e["y"] / n for e in selected],
                mode="markers",
                name=label,
                marker={"color": EVENT_COLORS[event], "size": 7},
                text=[
                    f"Шаг {e['step']}; источник {e['source']:.4g}; граница {e['flux'] + e['boundary']:.4g}"
                    for e in selected
                ],
            ),
            row=1,
            col=1,
        )
    figure.add_trace(
        go.Scatter(
            x=[e["step"] for e in history],
            y=[e["payoff"] for e in history],
            mode="lines",
            name="Сумма",
        ),
        row=1,
        col=2,
    )
    figure.update_xaxes(range=[0, 1], title="x", row=1, col=1)
    figure.update_yaxes(range=[0, 1], title="z", row=1, col=1)
    figure.update_xaxes(title="Шаг", row=1, col=2)
    figure.update_yaxes(title="Вклад пути", row=1, col=2)
    figure.update_layout(height=480, legend={"orientation": "h", "y": -0.2})
    return figure
