"""Accuracy of one frozen system using public kernels and the shared protocol.

The sparse reference runs in a separate process. It is never passed to a kernel.
No Picard iteration or advance in physical time is performed by this benchmark.
"""

import argparse
import hashlib
import json
import os
import platform
import re
import resource
import subprocess
import sys
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

from benchmarks.heat3d_pilot import ROOT, THREAD_VARIABLES, source_state, utc_now

METHODS = ("lod", "mc", "mc_counts", "mass_transport")
TRANSPORT_COUNTERS = ("node_updates", "transport_sweeps", "edge_transfers")
OPERATOR_ARRAYS = ("diagonal", "weights", "rhs", "fixed", "fixed_values", "volumes")


def jobs(plan):
    seen = set()
    for case in plan["cases"]:
        config = {**plan["configuration"], **case["configuration"]}
        for variant in plan["variants"]:
            name = case["id"] + "--" + variant["id"]
            if not re.fullmatch(r"[a-zA-Z0-9_-]+", name) or name in seen:
                raise ValueError("Expected unique safe job identifiers")
            if variant["method"] not in METHODS:
                raise ValueError("Unsupported frozen benchmark method")
            seen.add(name)
            yield name, case["id"], config, variant["method"]


def snapshot(config):
    from dataclasses import replace

    import numpy as np

    from pdelab.boundary3d import FACES
    from pdelab.heat3d_operator import freeze_heat3d
    from pdelab.heat3d_problem import make_heat_problem3d

    problem = replace(
        make_heat_problem3d(
            config["case"],
            config["n"],
            config["alpha"],
            capacity_gamma=config["capacity_gamma"],
            reaction_sigma=config["reaction_sigma"],
            kinds=dict(zip(FACES, config["kinds"], strict=True)),
            beta=dict(zip(FACES, config["betas"], strict=True)),
        ),
        exact=None,
    )
    previous = problem.initial_values()
    x, y, z = problem.coordinates()
    iterate = previous + 0.1 * (x + 2 * y + 3 * z)
    operator = freeze_heat3d(problem, previous, iterate, config["dt"], config["dt"])
    points = np.asarray([config["point"][::-1]], dtype=np.int64)
    return operator, points


def signature(operator):
    digest = hashlib.sha256()
    for name in OPERATOR_ARRAYS:
        digest.update(getattr(operator, name).tobytes())
    return digest.hexdigest()


def rss_bytes():
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return raw if sys.platform == "darwin" else raw * 1024


def reference(config, plan):
    import numpy as np
    from scipy.sparse.linalg import spsolve

    start = perf_counter()
    operator, points = snapshot(config)
    construction_seconds = perf_counter() - start
    start = perf_counter()
    matrix, rhs = operator.assemble()
    assembly_seconds = perf_counter() - start
    start = perf_counter()
    field = spsolve(matrix, rhs)
    direct_seconds = perf_counter() - start
    scale = float(np.max(np.abs(rhs)))
    residual = float(np.max(np.abs(matrix @ field - rhs)))
    relative = residual / scale if scale else residual
    if not np.isfinite(field).all() or not np.isfinite(relative):
        raise ValueError("Nonfinite sparse reference")
    if relative > plan["reference"]["relative_residual_max"]:
        raise ValueError("Sparse reference did not meet its residual tolerance")
    return dict(
        kind="sparse_direct_same_frozen_operator",
        values=field.reshape(operator.shape)[tuple(points.T)],
        relative_residual=relative,
        construction_seconds=construction_seconds,
        matrix_assembly_seconds=assembly_seconds,
        direct_seconds=direct_seconds,
        process_peak_rss_bytes=rss_bytes(),
        operator_sha256=signature(operator),
        note="Independent linear solver for a fixed A*u=b; not an analytic reference in T",
    )


def warmup(method):
    if method == "lod":
        from pdelab.heat3d_linear import warmup_lines3d

        warmup_lines3d()
    elif method == "mc":
        from pdelab.heat3d_mc import warmup_mc3d

        warmup_mc3d()
    else:
        from pdelab.heat3d_transport import warmup_transport3d

        warmup_transport3d()


def run_trial(config, settings, method, budget, seed):
    import numpy as np

    start = perf_counter()
    operator, points = snapshot(config)
    construction_seconds = perf_counter() - start
    if method == "lod":
        from pdelab.heat3d_linear import solve_lines3d

        result = solve_lines3d(
            operator,
            np.zeros(operator.shape),
            tolerance=budget,
            max_iterations=settings["max_iterations"],
        )
        values = (
            result.values[tuple(points.T)] if result.status == "completed" else None
        )
        backend_preparation = 0.0
        # LinearResult3D has no workspace estimate. This required protocol field
        # is replaced with None in every exported row/series below.
        workspace = 0
    else:
        if method == "mc":
            from pdelab.heat3d_mc import solve_frozen_mc3d as solve
        elif method == "mc_counts":
            from pdelab.heat3d_transport import solve_frozen_counts3d as solve
        else:
            from pdelab.heat3d_transport import solve_frozen_mass3d as solve
        options = dict(max_workspace_bytes=settings["max_workspace_bytes"])
        if method == "mass_transport":
            options.update(tolerance=budget, max_iterations=settings["max_iterations"])
        else:
            options.update(
                walks=int(budget),
                seed=seed,
                **{
                    k: settings[k]
                    for k in ("max_steps", "max_walks", "max_transitions")
                },
            )
        result = solve(operator, points, **options)
        values = result.values if result.status == "completed" else None
        backend_preparation = result.preparation_seconds
        workspace = result.workspace_bytes
    seconds = perf_counter() - start
    stochastic = method in ("mc", "mc_counts")
    transport = method in ("mc_counts", "mass_transport")
    completed_paths = getattr(result, "completed_paths", None)
    diagnostics = dict(
        construction_seconds=construction_seconds,
        coefficients_seconds=backend_preparation,
        backend_and_result_seconds=seconds - construction_seconds,
        operator_sha256=signature(operator),
        workspace_kind="not_available" if method == "lod" else "backend_estimate",
        seed=seed if stochastic else None,
        walks_per_estimate=int(budget) if stochastic else None,
        completed_walks=(
            int(np.sum(completed_paths)) if completed_paths is not None else None
        ),
        virtual_particle_moves=int(result.transitions) if stochastic else None,
        **{
            k: int(getattr(result, k)) if transport else None
            for k in TRANSPORT_COUNTERS
        },
        tail_error_bound=(
            getattr(result, "tail_error_bound", None) if transport else None
        ),
        stderr=getattr(result, "stderr", None) if method == "mc" else None,
        iterations=getattr(result, "iterations", None),
        linear_relative_residual=getattr(result, "relative_residual", None),
    )
    observation = SimpleNamespace(
        values=values,
        status=result.status,
        reason=result.reason,
        seconds=seconds,
        preparation_seconds=construction_seconds + backend_preparation,
        workspace_bytes=workspace,
        peak_rss_bytes=rss_bytes(),
        walks=int(result.walks) if stochastic else 0,
        transitions=int(result.transitions) if stochastic else 0,
        inner_iterations=int(getattr(result, "iterations", 0)),
    )
    return observation, diagnostics


def enrich(study, details, method):
    """Attach measured diagnostics without changing protocol decisions."""
    for trial, detail in zip(study["trials"], details, strict=True):
        trial.update(detail)
        if method == "lod":
            trial["workspace_bytes"] = None
    for collection in ("series", "rows"):
        for row in study[collection]:
            selected = [
                t
                for t in study["trials"]
                if (t["phase"], t["candidate"]) == (row["phase"], row["candidate"])
            ]
            for key in (
                "construction_seconds",
                "coefficients_seconds",
                "completed_walks",
                "virtual_particle_moves",
                *TRANSPORT_COUNTERS,
            ):
                row[key] = (
                    sum(t[key] for t in selected)
                    if selected and all(t[key] is not None for t in selected)
                    else None
                )
            if method == "lod":
                row["workspace_bytes"] = None


def worker(plan, index, folder):
    from pdelab.heat3d_experiment import _versions, json_bytes
    from pdelab.study_protocol import run_accuracy_study

    name, case, config, method = list(jobs(plan))[index]
    target = json.loads((folder / (case + "--reference.json")).read_bytes())
    start = perf_counter()
    warmup(method)
    warmup_seconds = perf_counter() - start
    details = []
    journal = folder / (name + "--attempts.jsonl")

    def append_event(event):
        # Outside the solve timer; retain attempted work if a worker later fails.
        with journal.open("ab") as stream:
            stream.write(json_bytes(event).replace(b"\n", b" ") + b"\n")

    def runner(budget, seed):
        attempt = len(details) + 1
        append_event(
            dict(
                event="started",
                attempt=attempt,
                budget=budget,
                seed=seed if method in ("mc", "mc_counts") else None,
            )
        )
        observation, detail = run_trial(config, plan["settings"], method, budget, seed)
        append_event(
            dict(
                event="finished",
                attempt=attempt,
                budget=budget,
                **vars(observation),
                **detail,
            )
        )
        if detail["operator_sha256"] != target["operator_sha256"]:
            raise ValueError("Frozen operator changed between reference and trial")
        details.append(detail)
        return observation

    stochastic = method in ("mc", "mc_counts")
    start = perf_counter()
    study = run_accuracy_study(
        runner,
        plan["walk_budgets"] if stochastic else plan["budgets"],
        target["values"],
        repeats=plan["repeats"],
        seed=plan["seed"],
        calibration_fraction=plan["calibration_fraction"],
        method=method,
        walks_applicable=stochastic,
        budget_kind=(
            "walks_per_query"
            if stochastic
            else (
                "relative_linear_residual" if method == "lod" else "relative_tail_bound"
            )
        ),
        retain_values=True,
    )
    study_call_seconds = perf_counter() - start
    enrich(study, details, method)
    study["metadata"] = dict(
        scope="one_frozen_system_not_a_full_time_history",
        configuration=config,
        settings=plan["settings"],
        method=method,
        source=source_state(),
        python=platform.python_version(),
        platform=platform.platform(),
        versions=_versions(),
        warmup_seconds=warmup_seconds,
        reference_file=case + "--reference.json",
        reference_kind=target["kind"],
        operator_sha256=target["operator_sha256"],
        workspace="backend estimate; LOM unavailable; never a measured RSS",
        counter_semantics="virtual particle moves != occupied-node updates != nonzero edge flows",
        seed_applicable=stochastic,
    )
    (folder / (name + ".json")).write_bytes(
        json_bytes(
            dict(
                study=study,
                process=dict(
                    study_call_seconds=study_call_seconds,
                    peak_rss_bytes_before_export=rss_bytes(),
                    rss_scope="fresh worker, imports/JIT and all trials; sparse reference in another process",
                    thread_environment={k: os.environ.get(k) for k in THREAD_VARIABLES},
                ),
            )
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--worker", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--reference", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_bytes())
    if plan.get("experiment") != "frozen_accuracy":
        raise ValueError("Expected a frozen_accuracy plan")
    if args.worker is not None:
        worker(plan, args.worker, args.out)
        return
    if args.reference is not None:
        from pdelab.heat3d_experiment import json_bytes

        case = plan["cases"][args.reference]
        config = {**plan["configuration"], **case["configuration"]}
        result = reference(config, plan)
        (args.out / (case["id"] + "--reference.json")).write_bytes(json_bytes(result))
        return
    for name in THREAD_VARIABLES:
        os.environ[name] = "1"
    before = source_state()
    if before["dirty"]:
        raise RuntimeError("Commit integrated kernels and plans before measuring")
    job_list = list(jobs(plan))
    args.out.mkdir(parents=True, exist_ok=False)
    from pdelab.heat3d_experiment import csv_bytes, json_bytes

    (args.out / "plan.json").write_bytes(json_bytes(plan))
    manifest = dict(
        plan=plan,
        source_before=before,
        started_utc=utc_now(),
        method_order=[name for name, _, _, _ in job_list],
        completed_jobs=[],
        completed_references=[],
        complete=False,
    )

    def save():
        (args.out / "manifest.json").write_bytes(json_bytes(manifest))

    save()
    tasks = [("reference", i, c["id"]) for i, c in enumerate(plan["cases"])]
    tasks += [("worker", i, name) for i, (name, _, _, _) in enumerate(job_list)]
    for kind, index, name in tasks:
        with (args.out / (name + "--" + kind + ".log")).open("wb") as log:
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "benchmarks.heat3d_transport_frozen",
                    "--plan",
                    str((args.out / "plan.json").resolve()),
                    "--out",
                    str(args.out.resolve()),
                    "--" + kind,
                    str(index),
                ],
                cwd=ROOT,
                env=os.environ.copy(),
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if result.returncode:
            manifest.update(failed_job=name, worker_exit_code=result.returncode)
            save()
            raise RuntimeError("Worker failed; preserve manifest, journal and log")
        key = "completed_references" if kind == "reference" else "completed_jobs"
        manifest[key].append(name)
        save()
        print(kind, name, flush=True)
    after = source_state()
    if before != after:
        raise RuntimeError("Measured source changed")
    manifest.update(source_after=after, finished_utc=utc_now(), complete=True)
    save()
    for name in ("rows", "series", "trials"):
        records = []
        for job, _, _, _ in job_list:
            result = json.loads((args.out / (job + ".json")).read_bytes())["study"]
            records.extend(dict(scenario_variant=job, **row) for row in result[name])
        (args.out / (name + ".csv")).write_bytes(csv_bytes(records))


if __name__ == "__main__":
    main()
