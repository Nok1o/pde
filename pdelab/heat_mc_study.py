"""Time-to-accuracy study for complete heat solves, never free previous layers."""

import csv
import json
import platform
from pathlib import Path
from time import perf_counter

import numba
import numpy as np

from .heat import solve_heat
from .heat_mc import HEAT_MC_METHODS, _settings, _warmup, solve_heat_mc, validate_coarse
from .problem import validate_points
from .research import _walk_budgets
from .study_policy import (
    ACCURACY_DESCRIPTION,
    ACCURACY_METRIC,
    SELECTION_POLICY,
    phase_seed_index,
    search_budgets,
    study_error_metrics,
)

VALIDATION_POLICY = SELECTION_POLICY


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
    *,
    methods=("lod", "mc"),
    coarse_n=None,
):
    if (
        not methods
        or len(set(methods)) != len(methods)
        or any(m not in ("lod", *HEAT_MC_METHODS) for m in methods)
    ):
        raise ValueError("Выберите различные методы теплопроводности")
    if "mc_control" in methods:
        coarse_n = max(4, problem.n // 4) if coarse_n is None else coarse_n
        validate_coarse(problem.n, coarse_n)
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
    reference_info = study_error_metrics([reference], reference)
    settings = dict(
        methods=list(methods),
        coarse_n=coarse_n,
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
        seconds_per_phase=seconds_per_method,
        max_transitions=max_transitions,
        python=platform.python_version(),
        platform=platform.platform(),
        numpy=np.__version__,
        numba=numba.__version__,
        reference=float(reference),
        reference_kind="analytic" if problem.exact else "discrete",
        grid_reference=float(grid.fields[-1, y, x]),
        validation_policy=VALIDATION_POLICY,
        accuracy_metric=ACCURACY_METRIC,
        accuracy=ACCURACY_DESCRIPTION,
        absolute_metric=reference_info["absolute_metric"],
        relative_applicable=reference_info["relative_applicable"],
        reference_norm=reference_info["reference_norm"],
        timing="Full solve to T including all known-data preparation; JIT/reference excluded.",
        failure_accounting=(
            "Exceptions retain measured call time. Counters and memory unavailable "
            "from a failed legacy kernel remain null, never zero."
        ),
    )
    trials, rows = [], []
    _warmup()
    solve_heat(problem, dt, min(dt, T), tol=1e-2)
    started = perf_counter()
    for method_index, method in enumerate(methods):
        budgets = (
            (1e-2, 1e-4, 1e-6, 1e-8, 1e-10)
            if method == "lod"
            else _walk_budgets(int(max_walks), 1)
        )

        def series(budget, budget_index, phase, deadline):
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
                            phase_seed_index(phase),
                            repeat,
                        ]
                    ).generate_state(1)[0]
                )
                record = dict(
                    method=method,
                    budget=float(budget),
                    phase=phase,
                    repeat=repeat,
                    seed=trial_seed if method != "lod" else None,
                    status="completed",
                    reason="",
                    value=None,
                    relative_error_pct=None,
                    absolute_error=None,
                    seconds=None,
                    preparation_seconds=None,
                    walks=None,
                    transitions=None,
                    memory_bytes=None,
                )
                start = perf_counter()
                try:
                    if method != "lod":
                        result = solve_heat_mc(
                            problem,
                            dt,
                            T,
                            points,
                            walks=budget,
                            seed=trial_seed,
                            max_transitions=max_transitions,
                            method=method,
                            coarse_n=coarse_n if method == "mc_control" else None,
                        )
                        value = float(result.values[0])
                        record.update(
                            preparation_seconds=result.preparation_seconds,
                            walks=int(budget),
                            transitions=(
                                result.spatial_transitions + result.temporal_transitions
                            ),
                            memory_bytes=result.memory_bytes,
                        )
                    else:
                        result = solve_heat(problem, dt, T, tol=budget)
                        value = float(result.fields[-1, y, x])
                        record.update(transitions=0, memory_bytes=result.fields.nbytes)
                    record["seconds"] = result.seconds
                    metrics = study_error_metrics([value], reference)
                except (ValueError, RuntimeError) as exc:
                    if record["seconds"] is None:
                        record["seconds"] = perf_counter() - start
                    record.update(status="solver_failed", reason=str(exc))
                    trials.append(record)
                    raise RuntimeError(str(exc)) from exc
                record.update(
                    value=value,
                    relative_error_pct=metrics["relative_error_pct"],
                    absolute_error=metrics["absolute_error"],
                )
                trials.append(record)
                values.append(value)
                times.append(result.seconds)
            return dict(
                **study_error_metrics(values, reference),
                seconds=float(np.mean(times)),
                completed_repeats=len(values),
                requested_repeats=repeats,
            )

        method_rows = search_budgets(
            method,
            budgets,
            series,
            seconds_per_method,
            clock=perf_counter,
            relative_applicable=reference_info["relative_applicable"],
        )
        for row in method_rows:
            row["status_detail"] = row["status"]
            row["status"] = {
                "confirmed": "reached",
                "validation_failed": "not_confirmed",
                "timeout": "time_limit",
                "not_reached": "budget_exhausted",
            }.get(row["status_code"], row["status_code"])
            row["validation_error_pct"] = row["relative_error_pct"]
            if not reference_info["relative_applicable"]:
                recorded = [t for t in trials if t["method"] == method]
                row.update(
                    requested_repeats=repeats,
                    recorded_repeats=len(recorded),
                    completed_repeats=sum(t["status"] == "completed" for t in recorded),
                    spent_seconds=sum(t["seconds"] for t in recorded),
                    total_walks=(
                        sum(t["walks"] for t in recorded)
                        if method != "lod"
                        and all(t["walks"] is not None for t in recorded)
                        else None
                    ),
                )
        rows.extend(method_rows)
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
