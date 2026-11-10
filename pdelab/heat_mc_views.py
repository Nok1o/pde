"""Export MC query values and full space-time costs separately from step diagnostics."""

import csv
import io
import json
import platform
import zipfile
from pathlib import Path

import numba
import numpy as np


def heat_mc_archive(result, problem, dt):
    rows = result.rows()
    text = io.StringIO(newline="")
    writer = csv.DictWriter(text, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    metadata = dict(
        problem=problem.name,
        n=problem.n,
        a2=problem.a2,
        dt=dt,
        T=float(result.times[-1]),
        times=result.times.tolist(),
        mode=(
            "given_previous_step" if result.diagnostic_step else "standalone_space_time"
        ),
        walks_per_point=result.walks_per_point,
        walks=result.walks,
        seed=result.seed,
        max_transitions=result.max_transitions,
        seconds=result.seconds,
        preparation_seconds=result.preparation_seconds,
        path_seconds=result.path_seconds,
        spatial_transitions=result.spatial_transitions,
        temporal_transitions=result.temporal_transitions,
        memory_bytes=result.memory_bytes,
        memory_scope="Main numerical buffers during run; excludes transient copies/RSS/JIT.",
        previous_solution_preparation=(
            "Not included; previous field supplied"
            if result.diagnostic_step
            else "None: paths go back to initial data"
        ),
        uncertainty="SE of independent complete paths; no estimated previous fields",
        python=platform.python_version(),
        numpy=np.__version__,
        numba=numba.__version__,
    )
    values = io.BytesIO()
    np.savez_compressed(
        values,
        points_yx=result.points,
        values=result.values,
        stderr=result.stderr,
        t=result.times,
    )
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("queries.csv", text.getvalue().encode("utf-8-sig"))
        z.writestr(
            "experiment.json", json.dumps(metadata, ensure_ascii=False, indent=2)
        )
        z.writestr("queries.npz", values.getvalue())
    return output.getvalue()


def export_heat_mc(result, problem, dt, folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    archive = heat_mc_archive(result, problem, dt)
    with zipfile.ZipFile(io.BytesIO(archive)) as z:
        z.extractall(folder)
    (folder / "heat-mc-results.zip").write_bytes(archive)
