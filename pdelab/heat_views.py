"""Physical-time profiles, field slices and reproducible heat exports."""

import csv
import io
import json
import platform
import zipfile
from pathlib import Path

import numpy as np
import plotly.graph_objects as go


def heat_profile(result, point=None):
    x, y = point or (result.problem.n // 2, result.problem.n // 2)
    figure = go.Figure(
        go.Scatter(x=result.times, y=result.fields[:, y, x], name="Сеточный расчёт")
    )
    if result.problem.exact:
        values = [
            result.problem.evaluate(result.problem.exact, t)[y, x] for t in result.times
        ]
        figure.add_scatter(
            x=result.times, y=values, name="Точное решение", line=dict(dash="dash")
        )
    figure.update_layout(
        title=f"Профиль во времени: x={x / result.problem.n:g}, z={y / result.problem.n:g}",
        xaxis_title="Физическое время t",
        yaxis_title="u(t)",
    )
    return figure


def heat_slice(result, index, error=False):
    field = result.fields[index]
    if error:
        if result.problem.exact is None:
            raise ValueError("Для ошибки требуется точное решение")
        field = field - result.problem.evaluate(
            result.problem.exact, result.times[index]
        )
    figure = go.Figure(
        go.Heatmap(
            x=result.problem.axis,
            y=result.problem.axis,
            z=field,
            colorbar=dict(title="u − u*" if error else "u"),
        )
    )
    figure.update_layout(
        title=f"{'Ошибка' if error else 'Поле'}, t={result.times[index]:.8g}",
        xaxis_title="x",
        yaxis_title="z",
        yaxis=dict(scaleanchor="x"),
    )
    return figure


def heat_archive(result):
    """ZIP with every physical layer, per-layer diagnostics and standalone plots."""
    rows = result.rows()
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    arrays = dict(
        t=result.times,
        time_index=np.arange(len(result.times)),
        fields=result.fields,
        axis=result.problem.axis,
        iterations=result.iterations,
        relative_residuals=result.relative_residuals,
    )
    if result.problem.exact:
        arrays["exact"] = np.array(
            [result.problem.evaluate(result.problem.exact, t) for t in result.times]
        )
    npz = io.BytesIO()
    np.savez_compressed(npz, **arrays)
    metadata = dict(
        equation="u_t = a2*Delta(u) + F",
        scheme="backward_euler",
        problem=result.problem.name,
        n=result.problem.n,
        a2=result.problem.a2,
        dt=result.dt,
        T=result.T,
        method=result.method,
        tolerance=result.tolerance,
        max_iter=result.max_iter,
        seconds=result.seconds,
        factorizations=result.factorizations,
        linear_stopping_rule="rhs_inf_including_dirichlet",
        python=platform.python_version(),
        numpy=np.__version__,
        timing="After sweep JIT warmup; includes matrix preparation; excludes plots/export.",
    )
    # The NPZ retains all layers; limit HTML frame count to keep it shareable.
    indices = np.unique(
        np.linspace(0, len(result.times) - 1, min(21, len(result.times)), dtype=int)
    )
    figure = heat_slice(result, int(indices[0]))
    figure.frames = [
        go.Frame(
            name=str(i),
            data=heat_slice(result, int(i)).data,
            layout=go.Layout(title=f"Поле, t={result.times[i]:.8g}"),
        )
        for i in indices
    ]
    figure.update_layout(
        sliders=[
            dict(
                currentvalue=dict(prefix="Физическое время t="),
                steps=[
                    dict(
                        label=f"{result.times[i]:.8g}",
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
                    for i in indices
                ],
            )
        ]
    )
    html = (
        "<!doctype html><html lang='ru'><meta charset='utf-8'><title>Теплопроводность</title>"
        + heat_profile(result).to_html(full_html=False, include_plotlyjs=True)
        + figure.to_html(full_html=False, include_plotlyjs=False)
        + "</html>"
    )
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("layers.csv", output.getvalue().encode("utf-8-sig"))
        z.writestr("fields.npz", npz.getvalue())
        z.writestr(
            "experiment.json", json.dumps(metadata, ensure_ascii=False, indent=2)
        )
        z.writestr("interactive.html", html)
    return archive.getvalue()


def export_heat(result, folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    archive = heat_archive(result)
    with zipfile.ZipFile(io.BytesIO(archive)) as z:
        z.extractall(folder)
    (folder / "heat-results.zip").write_bytes(archive)
