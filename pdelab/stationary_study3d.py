"""Stationary observations adapted to the existing fixed-budget study protocol.

No fictitious physical time is attached to a stationary solve. Source/boundary
input fields and independent references are shared inputs; each timed backend
prepares its own operator, control approximation and particle buffers.
"""

from time import perf_counter
from types import SimpleNamespace

import numpy as np

from .boundary3d import BoundaryProblem3D, boundary_workspace_bytes3d
from .grouped_solver3d import GroupedWalkFailure
from .study_policy import readable_status
from .study_protocol import (
    METRIC_DESCRIPTION,
    METRIC_ID,
    PROTOCOL_ID,
    run_accuracy_study,
)


def study_stationary3d(
    problem,
    points,
    reference,
    methods,
    *,
    walk_budgets=(64, 256, 1024, 4096),
    tolerances=(0.1, 0.03, 0.01, 0.003, 0.001, 1e-4, 1e-6, 1e-8, 1e-10),
    repeats=5,
    seed=42,
    batch_size=128,
    max_iter=20000,
    max_steps=None,
    coarse_n=None,
    seconds_per_phase=60.0,
    solver=None,
    progress=None,
    clock=perf_counter,
):
    from .boundary_solvers3d import METHODS_BOUNDARY_3D, solve_boundary3d
    from .solvers3d import METHODS_3D, solve3d

    mixed = isinstance(problem, BoundaryProblem3D)
    names = METHODS_BOUNDARY_3D if mixed else METHODS_3D
    solver = (solve_boundary3d if mixed else solve3d) if solver is None else solver
    methods = list(methods)
    if (
        not methods
        or len(set(methods)) != len(methods)
        or any(m not in names for m in methods)
    ):
        raise ValueError("Выберите различные доступные методы 3D")
    if isinstance(seed, (bool, np.bool_)) or int(seed) != seed or not 0 <= seed < 2**32:
        raise ValueError("Seed должен быть целым в [0, 2³²)")
    walk_budgets, tolerances = list(walk_budgets), list(tolerances)
    if any(m != "lod_serial" for m in methods) and (
        not walk_budgets
        or any(
            isinstance(n, (bool, np.bool_))
            or not np.isfinite(n)
            or int(n) != n
            or n < 2
            for n in walk_budgets
        )
    ):
        raise ValueError("Бюджеты MC должны быть целыми >=2")
    if "lod_serial" in methods and (
        not tolerances or any(not np.isfinite(t) or not 0 < t < 1 for t in tolerances)
    ):
        raise ValueError("Нужны относительные допуски ЛОМ между 0 и 1")
    points = np.asarray(points, dtype=np.int64)
    reference = np.asarray(reference, dtype=float)
    output = dict(
        protocol=PROTOCOL_ID,
        selection_policy=PROTOCOL_ID,
        accuracy_metric=METRIC_ID,
        accuracy=METRIC_DESCRIPTION,
        calibration_fraction=0.8,
        rows=[],
        trials=[],
        series=[],
        seconds_per_phase=seconds_per_phase,
        query_values_retained=True,
        step_definition="Markov path events, including terminal absorption; not only spatial moves",
    )
    for method_index, method in enumerate(methods):
        diagnostics = []
        method_seed = int(
            np.random.SeedSequence(
                [int(seed), 29, list(names).index(method)]
            ).generate_state(1)[0]
        )

        def runner(budget, trial_seed):
            stochastic = method != "lod_serial"
            arguments = dict(
                walks=int(budget) if stochastic else 2,
                tol=1e-8 if stochastic else float(budget),
                seed=trial_seed,
                max_iter=max_iter,
                max_steps=max_steps,
                batch_size=batch_size,
            )
            if method == "mc_control":
                arguments["coarse_n"] = coarse_n
            started = clock()
            values, status, reason = None, "completed", ""
            work, walks, steps, iterations = {}, 0, 0, 0
            try:
                result = solver(problem, points, method, **arguments)
                seconds = result.elapsed
                work = result.extras
                walks, steps, iterations = result.walks, result.steps, result.iterations
                if result.converged:
                    values = result.values[tuple(points.T)]
                else:
                    status, reason = "linear_limit", "ЛОМ не достиг допуска невязки"
            except GroupedWalkFailure as exc:
                seconds = exc.seconds
                work = exc.diagnostics
                walks, steps = work["walks"], work["transitions"]
                status, reason = exc.status, str(exc)
            except (ValueError, RuntimeError, MemoryError) as exc:
                seconds = max(0.0, clock() - started)
                status, reason = "solver_failed", str(exc)
            # Existing scalar kernels throw on a path limit without counters.
            # Do not pretend those unknown counters are measured zero work.
            counters_known = status not in ("solver_failed",)
            workspace = work.get(
                "workspace_bytes",
                boundary_workspace_bytes3d(problem.n)
                + (int(budget) * 8 if stochastic else 0),
            )
            record = dict(
                batch_size=batch_size if method == "mc_grouped" else None,
                effective_batch_size=work.get("effective_batch_size"),
                started_walks=(
                    work.get("started_walks", walks if counters_known else None)
                    if stochastic
                    else None
                ),
                counters_known=counters_known,
                path_seconds=work.get("path_seconds"),
                coarse_n=work.get("coarse_n"),
                rng=work.get("rng"),
                scheme=work.get("scheme", "uniform_finite_difference"),
                preparation_measured="preparation_seconds" in work,
            )
            diagnostics.append(record)
            return SimpleNamespace(
                values=values,
                status=status,
                reason=reason,
                seconds=seconds,
                preparation_seconds=work.get("preparation_seconds", 0.0),
                workspace_bytes=workspace,
                peak_rss_bytes=None,
                inner_iterations=iterations,
                walks=walks,
                transitions=steps,
            )

        def report(fraction, message):
            if progress:
                progress(
                    (method_index + fraction) / len(methods),
                    names[method] + " · " + message,
                )

        study = run_accuracy_study(
            runner,
            walk_budgets if method != "lod_serial" else tolerances,
            reference,
            repeats=repeats,
            seed=method_seed,
            method=method,
            budget_kind=(
                "walks_per_start"
                if method != "lod_serial"
                else "relative_residual_tolerance"
            ),
            walks_applicable=method != "lod_serial",
            calibration_fraction=0.8,
            seconds_per_phase=seconds_per_phase,
            clock=clock,
            progress=report,
            retain_values=True,
        )
        for trial, extra in zip(study["trials"], diagnostics, strict=True):
            trial.update(
                extra,
                method=method,
                value=trial["values"],
                steps=trial["transitions"],
                iterations=trial["inner_iterations"],
            )
            if not extra["counters_known"]:
                trial["walks"] = trial["transitions"] = trial["steps"] = None
        for row in study["rows"]:
            code = row["status"]
            row.update(
                name=names[method],
                status_code=code,
                selection_policy=PROTOCOL_ID,
                status_detail=row["reason"],
                batch_size=batch_size if method == "mc_grouped" else None,
                calibration_seconds=study["pilot_seconds"],
                total_validation_seconds=study["final_seconds"],
                search_seconds=study["elapsed_seconds"],
                validation_attempts=int(row["final_series"]),
            )
            row["status"] = readable_status(row)
        for row in [*study["rows"], *study["series"]]:
            matching = [
                t
                for t in study["trials"]
                if (t["phase"], t["candidate"]) == (row["phase"], row["candidate"])
            ]
            row.update(
                method=method, batch_size=batch_size if method == "mc_grouped" else None
            )
            row["started_walks"] = (
                sum(t["started_walks"] for t in matching)
                if matching and all(t["started_walks"] is not None for t in matching)
                else None
            )
            if any(not t["counters_known"] for t in matching):
                row["total_walks"] = row["transitions"] = None
        for collection in ("rows", "trials", "series"):
            output[collection].extend(study[collection])
    return output
