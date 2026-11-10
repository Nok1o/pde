"""Time-to-accuracy study for complete heat solves, never free previous layers."""

import csv
import json
import platform
from pathlib import Path
from time import perf_counter

import numba
import numpy as np

from .heat import solve_heat
from .heat_mc import _settings, _warmup, solve_heat_mc
from .problem import validate_points
from .research import TARGETS_PCT, _walk_budgets


def heat_mc_study(
    problem,
    dt=0.01,
    T=0.04,
    point=None,
    repeats=5,
    max_walks=16384,
    seed=42,
    seconds_per_method=60,
    max_transitions=1_000_000,
):
    if (
        isinstance(repeats, bool)
        or not np.isfinite(repeats)
        or int(repeats) != repeats
        or repeats < 2
    ):
        raise ValueError("Нужно не менее двух независимых повторов")
    if (
        isinstance(max_walks, bool)
        or not np.isfinite(max_walks)
        or int(max_walks) != max_walks
        or max_walks < 2
    ):
        raise ValueError("max_walks должно быть целым ≥2")
    if not np.isfinite(seconds_per_method) or seconds_per_method <= 0:
        raise ValueError("Лимит времени должен быть положительным")
    points = validate_points(problem.n, [point or (problem.n // 2, problem.n // 2)])
    _, max_walks, seed, max_transitions = _settings(
        problem.n, points, max_walks, seed, max_transitions
    )
    repeats = int(repeats)
    y, x = points[0]
    grid = solve_heat(problem, dt, T, method="direct")
    reference = (
        problem.evaluate(problem.exact, T)[y, x]
        if problem.exact
        else grid.fields[-1, y, x]
    )
    if reference == 0:
        raise ValueError("При нулевом эталоне относительная ошибка не определена")
    settings = dict(
        problem=problem.name,
        n=problem.n,
        a2=problem.a2,
        dt=dt,
        T=T,
        point_yx=[int(y), int(x)],
        repeats=int(repeats),
        max_walks=int(max_walks),
        seed=seed,
        seconds_per_method=seconds_per_method,
        max_transitions=max_transitions,
        python=platform.python_version(),
        platform=platform.platform(),
        numpy=np.__version__,
        numba=numba.__version__,
        reference=float(reference),
        reference_kind="analytic" if problem.exact else "discrete",
        grid_reference=float(grid.fields[-1, y, x]),
        timing="Full solve to T including all known-data preparation; JIT/reference excluded.",
    )
    trials, rows = [], []
    _warmup()
    solve_heat(problem, dt, min(dt, T), tol=1e-2)
    started = perf_counter()
    for method_index, method in enumerate(("lod", "mc")):
        deadline = perf_counter() + seconds_per_method
        budgets = (
            (1e-2, 1e-4, 1e-6, 1e-8, 1e-10)
            if method == "lod"
            else _walk_budgets(int(max_walks), 1)
        )
        pending = set(TARGETS_PCT)
        status = "budget_exhausted"

        def series(budget, budget_index, phase):
            values, times = [], []
            for repeat in range(int(repeats)):
                if perf_counter() >= deadline:
                    raise TimeoutError
                trial_seed = int(
                    np.random.SeedSequence(
                        [
                            seed,
                            method_index,
                            budget_index,
                            0 if phase == "calibration" else 1,
                            repeat,
                        ]
                    ).generate_state(1)[0]
                )
                if method == "mc":
                    result = solve_heat_mc(
                        problem,
                        dt,
                        T,
                        points,
                        walks=budget,
                        seed=trial_seed,
                        max_transitions=max_transitions,
                    )
                    value = float(result.values[0])
                    preparation = result.preparation_seconds
                    transitions = (
                        result.spatial_transitions + result.temporal_transitions
                    )
                    memory = result.memory_bytes
                else:
                    result = solve_heat(problem, dt, T, tol=budget)
                    value = float(result.fields[-1, y, x])
                    preparation, transitions, memory = None, 0, result.fields.nbytes
                values.append(value)
                times.append(result.seconds)
                trials.append(
                    dict(
                        method=method,
                        budget=float(budget),
                        phase=phase,
                        repeat=repeat,
                        seed=trial_seed if method == "mc" else None,
                        value=value,
                        seconds=result.seconds,
                        preparation_seconds=preparation,
                        transitions=transitions,
                        memory_bytes=memory,
                    )
                )
            error = (
                100
                * float(np.sqrt(np.mean((np.array(values) - reference) ** 2)))
                / abs(reference)
            )
            return error, float(np.mean(times))

        for budget_index, budget in enumerate(budgets):
            try:
                calibration, _ = series(budget, budget_index, "calibration")
                candidates = [target for target in pending if calibration <= target]
                if not candidates:
                    continue
                validation, seconds = series(budget, budget_index, "validation")
                for target in candidates:
                    if validation <= target:
                        rows.append(
                            dict(
                                method=method,
                                target_pct=target,
                                seconds=seconds,
                                budget=float(budget),
                                calibration_error_pct=calibration,
                                validation_error_pct=validation,
                                status="reached",
                            )
                        )
                        pending.remove(target)
                if not pending:
                    break
            except TimeoutError:
                status = "time_limit"
                break
            except RuntimeError as exc:
                status = "solver_failed: " + str(exc)
                break
        for target in pending:
            rows.append(
                dict(
                    method=method,
                    target_pct=target,
                    seconds=None,
                    budget=None,
                    calibration_error_pct=None,
                    validation_error_pct=None,
                    status=status,
                )
            )
    return dict(
        settings=settings,
        rows=sorted(rows, key=lambda r: (r["target_pct"], r["method"])),
        trials=trials,
        search_seconds=perf_counter() - started,
    )


def export_heat_mc_study(study, folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "settings.json").write_text(
        json.dumps(study["settings"], ensure_ascii=False, indent=2), encoding="utf-8"
    )
    for name, rows in (("timings", study["rows"]), ("trials", study["trials"])):
        with (folder / f"{name}.csv").open(
            "w", encoding="utf-8-sig", newline=""
        ) as stream:
            if rows:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
