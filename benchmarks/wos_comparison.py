"""Compare single-point WoS and grid MC estimates against a continuous reference."""

import argparse
import csv
import json
import platform
import subprocess
from functools import cache
from importlib.metadata import version
from math import pi, sqrt
from pathlib import Path
from statistics import fmean, median, stdev
from time import perf_counter

import numpy as np
from numba import njit

from pdelab.graded import estimate_graded, graded_axis, graded_operator
from pdelab.problem import make_problem, select_points
from pdelab.solvers import solve
from pdelab.wos import estimate_wos

METHODS = ("wos", "mc", "mc_control")
TARGETS = (0.01, 0.05, 0.10)
EPSILONS = (1e-2, 1e-3, 1e-4)
EPSILON = EPSILONS[-1]
WOS_MAX_STEPS = 1024
BASE_SEED = 2026


@njit
def _zero(x, y):
    return 0.0


@njit
def _affine(x, y):
    return 10.0 + 20.0 * x + 30.0 * y


@njit
def _polynomial(x, y):
    return 1000.0 * x * (1.0 - x) * y * (1.0 - y)


@njit
def _polynomial_source(x, y):
    return -2000.0 * (x * (1.0 - x) + y * (1.0 - y))


@njit
def _sine(x, y):
    return 100.0 * np.sin(pi * x) * np.sin(pi * y)


@njit
def _sine_source(x, y):
    return -2.0 * pi**2 * _sine(x, y)


# The exact functions also give the Dirichlet values when called on the boundary.
PRESETS = {
    "affine": (_zero, _affine),
    "polynomial": (_polynomial_source, _polynomial),
    "sine": (_sine_source, _sine),
}


def _queries(n):
    return {
        "center": (n // 2, n // 2),
        "offcenter": (n // 4, 3 * n // 8),
        "near_boundary": (1, 3 * n // 8),
    }


def _shared_queries(n):
    # These physical coordinates are nodes of both grids for all benchmark n.
    return {
        "center": (n // 2, n // 2),
        "offcenter": (n // 8, n // 2),
        "near_boundary": (n // 32, n // 2),
    }


def _graded_query(operator, point, walks, seed):
    value, stderr, steps = estimate_graded(operator, point, walks=walks, seed=seed)
    return value, stderr, steps, 0.0


def _seeds(phase, walks, repeats):
    phase_id = ("selection", "confirmation", "epsilon").index(phase)
    sequence = np.random.SeedSequence([BASE_SEED, phase_id, walks])
    return [int(seed) for seed in sequence.generate_state(repeats)]


def _trial(query, reference, seed):
    started = perf_counter()
    try:
        value, stderr, steps, preparation_seconds = query()
    except RuntimeError as exc:
        return dict(
            seed=seed,
            status="error",
            value=None,
            stderr=None,
            steps=None,
            seconds=perf_counter() - started,
            preparation_seconds=None,
            relative_error=None,
            error=str(exc),
        )
    return dict(
        seed=seed,
        status="ok",
        value=value,
        stderr=stderr,
        steps=steps,
        seconds=perf_counter() - started,
        preparation_seconds=preparation_seconds,
        relative_error=(value - reference) / abs(reference),
        error="",
    )


def _wos_query(point, source, boundary, walks, seed, epsilon):
    result = estimate_wos(
        point,
        source,
        boundary,
        walks=walks,
        seed=seed,
        epsilon=epsilon,
        max_steps=WOS_MAX_STEPS,
    )
    return result.value, result.stderr, result.steps, 0.0


def _grid_query(problem, points, method, walks, seed):
    result = solve(problem, points, method, walks=walks, seed=seed, record=False)
    index = tuple(points[0])
    return (
        float(result.values[index]),
        float(result.stderr[index]),
        result.steps,
        result.extras.get("preparation_seconds", 0.0),
    )


def _rms_error(rows):
    if any(row["status"] != "ok" for row in rows):
        return None
    return sqrt(fmean(row["relative_error"] ** 2 for row in rows))


def _thresholds(series, budgets):
    remaining = list(TARGETS)
    selections = {}
    had_error = False
    for walks in budgets:
        rms = _rms_error(series("selection", walks))
        if rms is None:
            had_error = True
            continue
        for target in remaining[:]:
            if rms <= target:
                selections[target] = (walks, rms)
                remaining.remove(target)
        if not remaining:
            break

    confirmations = {}
    for target in TARGETS:
        row = dict(
            target=target,
            walks=None,
            selection_rms=None,
            confirmation_rms=None,
            median_seconds=None,
            status="selection_error" if had_error else "not_selected",
        )
        if target in selections:
            walks, selection_rms = selections[target]
            if walks not in confirmations:
                confirmations[walks] = series("confirmation", walks)
            rows = confirmations[walks]
            rms = _rms_error(rows)
            row.update(walks=walks, selection_rms=selection_rms, confirmation_rms=rms)
            if rms is None:
                row["status"] = "confirmation_error"
            elif rms > target:
                row["status"] = "not_confirmed"
            else:
                row["status"] = "confirmed"
                row["median_seconds"] = median(item["seconds"] for item in rows)
        yield row


def _epsilon_summary(rows):
    rms = _rms_error(rows)
    errors = [row["relative_error"] for row in rows if row["status"] == "ok"]
    return dict(
        status="ok" if rms is not None else "error",
        rms_relative_error=rms,
        mean_signed_relative_error=fmean(errors) if rms is not None else None,
        mean_error_stderr=(
            stdev(errors) / sqrt(len(errors)) if rms is not None else None
        ),
    )


def _compare(grids, budgets, selection_repeats, confirmation_repeats, graded=False):
    trial_rows, thresholds, epsilon_rows = [], [], []
    methods = (*METHODS, "mc_graded") if graded else METHODS

    def run_series(query, reference, phase, walks):
        repeats = selection_repeats if phase == "selection" else confirmation_repeats
        return [
            _trial(lambda: query(seed), reference, seed)
            for seed in _seeds(phase, walks, repeats)
        ]

    @cache
    def wos_series(kind, point, phase, walks, epsilon):
        source, boundary = PRESETS[kind]
        return run_series(
            lambda seed: _wos_query(point, source, boundary, walks, seed, epsilon),
            boundary(*point),
            phase,
            walks,
        )

    # Compile both grid kernels and all WoS callback pairs before collecting times.
    warmup_problem = make_problem("affine", 4)
    warmup_points = select_points(4)
    for method in ("mc", "mc_control"):
        _grid_query(warmup_problem, warmup_points, method, 2, BASE_SEED)
    for source, boundary in PRESETS.values():
        _wos_query((0.5, 0.5), source, boundary, 2, BASE_SEED, EPSILON)
    if graded:
        operator = graded_operator(
            graded_axis(4), warmup_problem.f, warmup_problem.boundary
        )
        _graded_query(operator, (2, 2), 2, BASE_SEED)

    for kind, (source, exact) in PRESETS.items():
        epsilon_points = set()
        for n in grids:
            problem = make_problem(kind, n)
            queries = _shared_queries(n) if graded else _queries(n)
            if graded:
                axis = graded_axis(n)
                x, y = np.meshgrid(axis, axis)
                operator = graded_operator(
                    axis, np.broadcast_to(source(x, y), x.shape), exact(x, y)
                )
            for location, (ix, iy) in queries.items():
                point = (ix / n, iy / n)
                reference = exact(*point)
                points = select_points(n, point=(ix, iy))
                if graded:
                    graded_ix, graded_iy = np.searchsorted(axis, point)
                    graded_point = (graded_iy, graded_ix)
                case = dict(
                    problem=kind, n=n, location=location, x=point[0], y=point[1]
                )
                for method in methods:
                    epsilon = EPSILON if method == "wos" else None
                    context = dict(
                        **case, method=method, reference=reference, epsilon=epsilon
                    )

                    def series(phase, walks):
                        if method == "wos":
                            rows = wos_series(kind, point, phase, walks, epsilon)
                        elif method == "mc_graded":
                            rows = run_series(
                                lambda seed: _graded_query(
                                    operator, graded_point, walks, seed
                                ),
                                reference,
                                phase,
                                walks,
                            )
                        else:
                            rows = run_series(
                                lambda seed: _grid_query(
                                    problem, points, method, walks, seed
                                ),
                                reference,
                                phase,
                                walks,
                            )
                        trial_rows.extend(
                            dict(**context, phase=phase, walks=walks, **row)
                            for row in rows
                        )
                        return rows

                    thresholds.extend(
                        dict(**context, **row) for row in _thresholds(series, budgets)
                    )

                if point in epsilon_points:
                    continue
                epsilon_points.add(point)
                walks = budgets[-1]
                for epsilon in EPSILONS:
                    rows = wos_series(kind, point, "epsilon", walks, epsilon)
                    # No grid resolution is attached to the WoS-only sensitivity study.
                    context = case | dict(
                        n=None, method="wos", reference=reference, epsilon=epsilon
                    )
                    trial_rows.extend(
                        dict(**context, phase="epsilon", walks=walks, **row)
                        for row in rows
                    )
                    epsilon_rows.append(
                        dict(**context, walks=walks, **_epsilon_summary(rows))
                    )
            print(f"{kind} n={n}", flush=True)
    return trial_rows, thresholds, epsilon_rows


def _write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _git(*args):
    return subprocess.check_output(
        ["git", *args], cwd=Path(__file__).resolve().parents[1], text=True
    ).strip()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--smoke", action="store_true", help="Run a small comparison for verification."
    )
    parser.add_argument(
        "--graded", action="store_true", help="Include graded-grid MC at shared nodes."
    )
    args = parser.parse_args(argv)
    grids = (32,) if args.smoke else (32, 64, 128, 256)
    budgets = (64,) if args.smoke else (64, 256, 1024, 4096)
    selection_repeats, confirmation_repeats = (2, 2) if args.smoke else (3, 10)
    settings = dict(
        grids=grids,
        graded=args.graded,
        budgets=budgets,
        targets=TARGETS,
        epsilon=EPSILON,
        epsilon_sweep=EPSILONS,
        wos_max_steps=WOS_MAX_STEPS,
        seed=BASE_SEED,
        selection_repeats=selection_repeats,
        confirmation_repeats=confirmation_repeats,
        python=platform.python_version(),
        platform=platform.platform(),
        processor=platform.processor(),
        versions={name: version(name) for name in ("numpy", "scipy", "numba")},
        commit=_git("rev-parse", "HEAD"),
        working_tree_dirty=bool(_git("status", "--porcelain")),
    )
    args.out.mkdir(parents=True, exist_ok=True)
    trials, thresholds, epsilon_rows = _compare(
        grids,
        budgets,
        selection_repeats,
        confirmation_repeats,
        graded=args.graded,
    )
    for name, rows in (
        ("trials", trials),
        ("thresholds", thresholds),
        ("epsilon", epsilon_rows),
    ):
        _write_csv(args.out / f"{name}.csv", rows)
    (args.out / "settings.json").write_text(
        json.dumps(settings, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
