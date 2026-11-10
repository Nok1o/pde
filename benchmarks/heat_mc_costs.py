"""python -m benchmarks.heat_mc_costs --out runs/heat-mc-costs.csv.

The one-step diagnostic is shown both with a free supplied previous field and
with explicit deterministic preparation. Neither is labelled standalone MC.
"""

import argparse
import csv
import json
import platform
from pathlib import Path
from time import perf_counter

import numba
import numpy as np

from pdelab import heat_mc_step, make_heat_problem, solve_heat, solve_heat_mc
from pdelab.heat import time_grid


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--out", type=Path, default=Path("runs/heat-mc-costs.csv"))
    args = parser.parse_args()
    rows = []
    for n, dt, T in ((16, 0.01, 0.04), (32, 0.001, 0.2)):
        p = make_heat_problem(n=n)
        times, steps = time_grid(dt, T)
        # Warm up both kernels before measuring preparation.
        solve_heat(p, dt, dt)
        solve_heat_mc(p, dt, dt, walks=2)
        for walks in (64, 4096):
            standalone = solve_heat_mc(p, dt, T, walks=walks)
            started = perf_counter()
            previous = solve_heat(p, dt, float(times[-2]), method="direct")
            previous_seconds = perf_counter() - started
            diagnostic = heat_mc_step(
                p, previous.fields[-1], float(steps[-1]), T, walks=walks
            )
            lod = solve_heat(p, dt, T)
            for label, result, prep in (
                ("standalone_space_time", standalone, 0.0),
                ("given_previous_step_only", diagnostic, 0.0),
                (
                    "diagnostic_with_direct_previous_preparation",
                    diagnostic,
                    previous_seconds,
                ),
            ):
                rows.append(
                    dict(
                        n=n,
                        dt=dt,
                        T=T,
                        walks=walks,
                        mode=label,
                        total_seconds=result.seconds + prep,
                        previous_preparation_seconds=prep,
                        known_data_preparation_seconds=result.preparation_seconds,
                        path_seconds=result.path_seconds,
                        transitions=result.spatial_transitions
                        + result.temporal_transitions,
                        explicit_buffer_bytes=result.memory_bytes
                        + (previous.fields.nbytes if prep else 0),
                        lod_full_seconds=lod.seconds,
                    )
                )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    args.out.with_suffix(".json").write_text(
        json.dumps(
            dict(
                python=platform.python_version(),
                platform=platform.platform(),
                numpy=np.__version__,
                numba=numba.__version__,
                note="Single illustrative runs after warmup, not a time-to-accuracy comparison. Buffers are explicit arrays, not process peak RSS.",
            ),
            indent=2,
        ),
        encoding="utf-8",
    )
    print(args.out)


if __name__ == "__main__":
    main()
