"""An explanatory reflected/absorbed walk on the cube."""

import plotly.graph_objects as go
from plotly.subplots import make_subplots


def boundary_trace_figure3d(events, n):
    figure = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "scene"}, {"type": "xy"}]],
        subplot_titles=["Один случайный путь", "Накопленный вклад"],
    )
    figure.add_trace(
        go.Scatter3d(
            x=[e["x"] / n for e in events],
            y=[e["y"] / n for e in events],
            z=[e["z"] / n for e in events],
            mode="lines+markers",
            marker=dict(
                size=3, color=[e["step"] for e in events], colorscale="Viridis"
            ),
            text=[e["event"] + " · " + ", ".join(e["faces"]) for e in events],
            hovertemplate="(%{x:.3f}, %{y:.3f}, %{z:.3f})<br>%{text}<extra></extra>",
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Scatter(
            x=[e["step"] for e in events],
            y=[e["payoff"] for e in events],
            mode="lines+markers",
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    figure.update_layout(
        height=440,
        scene=dict(
            aspectmode="cube",
            xaxis=dict(range=[0, 1], title="x"),
            yaxis=dict(range=[0, 1], title="y"),
            zaxis=dict(range=[0, 1], title="z"),
        ),
    )
    figure.update_xaxes(title_text="Шаг", row=1, col=2)
    figure.update_yaxes(title_text="Вклад пути", row=1, col=2)
    return figure
