"""Portable nonlinear comparisons with full-chain uncertainty and stop reasons."""

import csv
import io
import json
import zipfile
from dataclasses import asdict
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def json_data(value):
    if isinstance(value, dict):
        return {key: json_data(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_data(item) for item in value]
    if isinstance(value, np.generic):
        return json_data(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def json_bytes(value):
    return json.dumps(
        json_data(value), ensure_ascii=False, indent=2, allow_nan=False
    ).encode("utf-8")


def csv_bytes(rows):
    buffer = io.StringIO(newline="")
    if rows:
        columns = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(buffer, columns)
        writer.writeheader()
        writer.writerows(json_data(rows))
    return buffer.getvalue().encode("utf-8-sig")


def history_rows(comparison):
    rows = []
    for repeat, result in [
        (-1, comparison.deterministic),
        *enumerate(comparison.stochastic),
    ]:
        for index, layer in enumerate(result.layers, 1):
            for entry in layer.history:
                rows.append(
                    dict(
                        method=result.method,
                        repeat=repeat,
                        layer=index,
                        time=float(result.times[index]),
                        layer_converged=layer.converged,
                        reason=layer.reason,
                        **entry,
                    )
                )
    return rows


def time_figure(comparison):
    x, y = comparison.point
    reference = comparison.reference
    figure = go.Figure()
    if reference.problem.exact is not None:
        exact = [
            reference.problem.evaluate(reference.problem.exact, t)[y, x]
            for t in reference.times
        ]
        figure.add_trace(go.Scatter(x=reference.times, y=exact, name="Точное решение"))
    figure.add_trace(
        go.Scatter(
            x=reference.times,
            y=reference.fields[:, y, x],
            name="Newton · сеточный эталон",
            line=dict(dash="dash"),
        )
    )
    lod = comparison.deterministic
    figure.add_trace(
        go.Scatter(
            x=lod.times,
            y=lod.fields[:, y, x],
            name="Picard + ЛОМ" + (" · не сошёлся" if not lod.completed else ""),
        )
    )
    if all(r.completed for r in comparison.stochastic):
        values = np.array([r.fields[:, y, x] for r in comparison.stochastic])
        mean, spread = np.mean(values, axis=0), np.std(values, axis=0, ddof=1)
        figure.add_trace(
            go.Scatter(
                x=reference.times,
                y=mean - spread,
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
            )
        )
        figure.add_trace(
            go.Scatter(
                x=reference.times,
                y=mean + spread,
                line=dict(width=0),
                fill="tonexty",
                name="MC · разброс полных цепочек ±s",
            )
        )
        figure.add_trace(
            go.Scatter(x=reference.times, y=mean, name="Picard + MC · среднее")
        )
    figure.update_layout(
        xaxis_title="Физическое время t", yaxis_title="u в выбранной точке", height=400
    )
    return figure


def field_figure(comparison, error=False):
    reference = comparison.reference
    problem = reference.problem
    exact = (
        problem.evaluate(problem.exact, reference.T)
        if problem.exact is not None
        else reference.fields[-1]
    )
    fields = [] if error else [("Эталон", exact)]
    if comparison.deterministic.completed:
        fields.append(("Picard + ЛОМ", comparison.deterministic.fields[-1]))
    if all(r.completed for r in comparison.stochastic):
        fields.append(
            (
                "Picard + MC · среднее",
                np.mean([r.fields[-1] for r in comparison.stochastic], axis=0),
            )
        )
    if not fields:
        fields = [("Нет сошедшихся расчётов", np.full_like(exact, np.nan))]
    if error:
        fields = [(name, np.abs(values - exact)) for name, values in fields]
    figure = make_subplots(
        rows=1, cols=len(fields), subplot_titles=[name for name, _ in fields]
    )
    for i, (_, values) in enumerate(fields, 1):
        figure.add_trace(
            go.Heatmap(x=problem.axis, y=problem.axis, z=values, coloraxis="coloraxis"),
            row=1,
            col=i,
        )
        figure.update_xaxes(title_text="x", row=1, col=i)
        figure.update_yaxes(
            title_text="y", scaleanchor="x" if i == 1 else f"x{i}", row=1, col=i
        )
    figure.update_layout(
        height=380,
        coloraxis=dict(colorscale="Magma" if error else "Viridis"),
        title=f"{'Ошибка' if error else 'Поле'} при t={reference.T:g}",
    )
    return figure


def residual_figure(comparison):
    figure = go.Figure()
    # One full MC chain is illustrative; uncertainty comes from all full chains.
    for result in (comparison.deterministic, comparison.stochastic[0]):
        for layer, outcome in enumerate(result.layers, 1):
            history = outcome.history
            if history:
                figure.add_trace(
                    go.Scatter(
                        x=[row["outer"] for row in history],
                        y=[row["nonlinear_residual"] for row in history],
                        mode="lines+markers",
                        name=f"{result.method} · слой {layer}",
                    )
                )
    figure.update_layout(
        xaxis_title="Внешняя итерация Picard",
        yaxis_title="Нелинейная относительная невязка",
        yaxis_type="log",
        height=380,
    )
    return figure


def comparison_files(comparison):
    reference = comparison.reference
    arrays = {
        "times": reference.times,
        "newton": reference.fields,
        "picard": comparison.deterministic.fields,
        "picard_times": comparison.deterministic.times,
        "matched_picard": comparison.matched_picard.fields,
        "matched_picard_times": comparison.matched_picard.times,
    }
    if reference.problem.exact is not None:
        arrays["exact"] = np.array(
            [
                reference.problem.evaluate(reference.problem.exact, t)
                for t in reference.times
            ]
        )
    for index, result in enumerate(comparison.stochastic):
        arrays[f"mc_{index}"] = result.fields
        arrays[f"mc_{index}_times"] = result.times
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    all_results = [comparison.deterministic, *comparison.stochastic]
    payload = dict(
        n=reference.problem.n,
        alpha=reference.problem.alpha,
        dt=reference.dt,
        T=reference.T,
        point_xy=comparison.point,
        array_order="yx",
        metrics=comparison.metrics,
        matched_picard=dict(
            settings=asdict(comparison.matched_picard.settings),
            completed=comparison.matched_picard.completed,
            reason=comparison.matched_picard.reason,
        ),
        runs=[
            dict(
                method=r.method,
                settings=asdict(r.settings),
                completed=r.completed,
                reason=r.reason,
                seconds=r.seconds,
                walks=r.walks,
                steps=r.steps,
                layers=[
                    dict(
                        converged=layer.converged,
                        reason=layer.reason,
                        inner_iterations=layer.inner_iterations,
                        walks=layer.walks,
                        steps=layer.steps,
                        history=layer.history,
                    )
                    for layer in r.layers
                ],
            )
            for r in all_results
        ],
        uncertainty="Independent complete time/Picard/MC chains; last inner SE is not used",
        timing="Includes all layers, freezes, MC validation and history storage; excludes JIT, reference and plots",
        workspace="Dense array/history reserve, excludes JIT/plots/sparse LU",
        reference="analytic if available, otherwise independent discrete Newton",
        mc_bias="Nonlinear plug-in and adaptive stopping need not be unbiased",
    )
    files = {
        "experiment.json": json_bytes(payload),
        "metrics.csv": csv_bytes(comparison.metrics),
        "iterations.csv": csv_bytes(history_rows(comparison)),
        "fields.npz": buffer.getvalue(),
    }
    for name, figure in (
        ("time", time_figure(comparison)),
        ("field", field_figure(comparison)),
        ("error", field_figure(comparison, True)),
        ("residual", residual_figure(comparison)),
    ):
        files[name + ".html"] = figure.to_html(include_plotlyjs=True).encode("utf-8")
    return files


def comparison_archive(comparison):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, content in comparison_files(comparison).items():
            archive.writestr(name, content)
    return buffer.getvalue()


def export_comparison(comparison, folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for name, content in comparison_files(comparison).items():
        (folder / name).write_bytes(content)


def study_figure(rows):
    figure = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=[
            "MC: ошибка к сеточному эталону",
            "Ошибка к точному решению и время",
        ],
    )
    groups = {}
    for row in rows:
        if row["method"] == "mc" and row["completed"]:
            key = (
                row["alpha"],
                row["n"],
                row["dt"],
                row["schedule"],
                row["relaxation"],
            )
            groups.setdefault(key, []).append(row)
    for key, group in groups.items():
        group.sort(key=lambda row: row["initial_walks"])
        figure.add_trace(
            go.Scatter(
                x=[row["initial_walks"] for row in group],
                y=[row["rmse_grid"] for row in group],
                mode="lines+markers",
                name=f"α={key[0]}, n={key[1]}, Δt={key[2]}, {key[3]}, ω={key[4]}",
            ),
            row=1,
            col=1,
        )
    for method in ("picard", "mc"):
        group = [row for row in rows if row["method"] == method and row["completed"]]
        figure.add_trace(
            go.Scatter(
                x=[row["seconds"] for row in group],
                y=[row["relative_error_pct"] for row in group],
                mode="markers",
                name=method,
                text=[
                    f"α={row['alpha']}, n={row['n']}, N={row['initial_walks']}, Δt={row['dt']}"
                    for row in group
                ],
            ),
            row=1,
            col=2,
        )
    figure.update_xaxes(title_text="Начальный бюджет N", type="log", row=1, col=1)
    figure.update_yaxes(title_text="RMSE к Newton", type="log", row=1, col=1)
    figure.update_xaxes(
        title_text="Полное время одной цепочки, с", type="log", row=1, col=2
    )
    figure.update_yaxes(title_text="Относительная ошибка, %", row=1, col=2)
    figure.update_layout(height=550)
    return figure
