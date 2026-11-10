"""Separate spatial, time and inner-linear errors for the sine eigenmode."""

import csv
from pathlib import Path

import numpy as np

from .heat import make_heat_problem, solve_heat


def heat_accuracy_study():
    rows = []
    # Tiny dt keeps time error below the spatial error, including n=32.
    previous = None
    for n in (8, 16, 32):
        problem = make_heat_problem(n=n)
        result = solve_heat(problem, dt=1e-6, T=0.002, method="direct")
        exact = problem.evaluate(problem.exact, result.T)
        error = float(np.max(abs(result.fields[-1] - exact)))
        order = None if previous is None else float(np.log2(previous / error))
        rows.append(
            dict(
                study="space",
                n=n,
                h=1 / n,
                dt=result.dt,
                T=result.T,
                tol=result.tolerance,
                error=error,
                order=order,
                reference="analytic",
                max_relative_residual=float(max(result.relative_residuals)),
            )
        )
        previous = error
    # Semidiscrete exact decay removes spatial error from the time-order study.
    problem = make_heat_problem(n=16)
    eigenvalue = 8 * problem.n**2 * np.sin(np.pi / (2 * problem.n)) ** 2
    previous = None
    for dt in (0.002, 0.001, 0.0005):
        result = solve_heat(problem, dt=dt, T=0.1, method="direct")
        exact = problem.initial * np.exp(-eigenvalue * problem.a2 * result.T)
        error = float(np.max(abs(result.fields[-1] - exact)))
        order = None if previous is None else float(np.log2(previous / error))
        rows.append(
            dict(
                study="time",
                n=problem.n,
                h=1 / problem.n,
                dt=dt,
                T=result.T,
                tol=result.tolerance,
                error=error,
                order=order,
                reference="semidiscrete_exact",
                max_relative_residual=float(max(result.relative_residuals)),
            )
        )
        previous = error
    problem = make_heat_problem(n=24)
    direct = solve_heat(problem, dt=0.01, T=0.04, method="direct")
    for tol in (1e-4, 1e-7, 1e-10):
        result = solve_heat(problem, dt=0.01, T=0.04, tol=tol)
        error = float(np.max(abs(result.fields - direct.fields)))
        rows.append(
            dict(
                study="linear",
                n=problem.n,
                h=1 / problem.n,
                dt=result.dt,
                T=result.T,
                tol=tol,
                error=error,
                order=None,
                reference="direct_same_steps",
                max_relative_residual=float(max(result.relative_residuals)),
            )
        )
    return rows


def export_heat_study(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
