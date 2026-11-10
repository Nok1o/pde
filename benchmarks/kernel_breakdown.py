"""Isolated before/after kernels and candidate measurements, after JIT warmup."""

import importlib.util
import json
import subprocess
from pathlib import Path
from time import perf_counter

import numpy as np
from numba import njit

from pdelab import solvers
from pdelab.problem import laplacian, make_problem, select_points


@njit(cache=True)
def residual_into(u, f, h, out):
    for y in range(1, u.shape[0] - 1):
        for x in range(1, u.shape[1] - 1):
            out[y - 1, x - 1] = (
                u[y + 1, x] + u[y - 1, x] + u[y, x + 1] + u[y, x - 1] - 4 * u[y, x]
            ) / h**2 - f[y, x]


def measure(fn, repeats=15):
    fn()
    durations = []
    for _ in range(repeats):
        start = perf_counter()
        fn()
        durations.append(perf_counter() - start)
    return durations


def main():
    baseline_path = Path("runs/baseline_solver.py").resolve()
    baseline_path.parent.mkdir(exist_ok=True)
    baseline_path.write_bytes(
        subprocess.check_output(["git", "show", "2aec67e:pdelab/solvers.py"])
    )
    spec = importlib.util.spec_from_file_location(
        "pdelab.baseline_solver", baseline_path
    )
    baseline = importlib.util.module_from_spec(spec)
    import sys

    sys.modules[spec.name] = baseline
    spec.loader.exec_module(baseline)
    baseline.warmup()
    solvers.warmup()
    rows = []
    for n in (16, 32, 64):
        p = make_problem("sine", n)
        points = select_points(n)
        old = baseline.solve(p, points, "lod_serial", record=False)
        new = solvers.solve(p, points, "lod_serial", record=False)
        np.testing.assert_allclose(old.values, new.values, rtol=1e-12, atol=1e-12)
        rhs = np.ones((n - 1, n - 1))
        ratio = n / (2 * np.pi)
        alpha, denominators = solvers._line_coefficients(n - 1, ratio)
        beta = np.empty(n)
        out = np.empty_like(rhs)
        residual_into(new.values, p.f, p.h, out)
        np.testing.assert_allclose(
            out, laplacian(new.values, p.h) - p.f[1:-1, 1:-1], atol=1e-10
        )
        operations = {
            "lod_before": lambda: baseline.solve(p, points, "lod_serial", record=False),
            "lod_after": lambda: solvers.solve(p, points, "lod_serial", record=False),
            "line_before_y": lambda: baseline._sweep_serial(rhs, ratio, 1),
            "line_coefficients_per_sweep_y": lambda: solvers._sweep_serial(
                rhs, ratio, 1
            ),
            "line_reused_y": lambda: solvers._sweep_into(
                rhs, out, ratio, 1, alpha, denominators, beta
            ),
            "line_reused_x": lambda: solvers._sweep_into(
                rhs, out, ratio, 0, alpha, denominators, beta
            ),
            "line_transpose_y": lambda: solvers._sweep_serial(
                np.ascontiguousarray(rhs.T), ratio, 0
            ).T.copy(),
            "residual_numpy": lambda: laplacian(new.values, p.h) - p.f[1:-1, 1:-1],
            "residual_into_candidate": lambda: residual_into(new.values, p.f, p.h, out),
            "mc_512": lambda: solvers._mc(
                p.f, p.boundary, points, 512, 42, 100 * n * n
            ),
            "mc_four_batches": lambda: [
                solvers._mc(p.f, p.boundary, points, 128, 42 + i, 100 * n * n)
                for i in range(4)
            ],
            "mc_zero_source": lambda: solvers._mc(
                np.zeros_like(p.f), p.boundary, points, 512, 42, 100 * n * n
            ),
        }
        for name, operation in operations.items():
            rows.append(dict(n=n, operation=name, seconds=measure(operation)))
    Path("benchmarks/results/breakdown.json").write_text(
        json.dumps(rows), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
