"""Fixed stationary #29 study; run after committing the measured source.

Full: python -m benchmarks.grouped_mc3d --out runs/grouped-mc3d
Smoke: python -m benchmarks.grouped_mc3d --smoke --out /private/tmp/grouped-smoke

The existing study protocol selects candidates and independently confirms them.
RSS diagnostics run in separate, fresh, warmed worker processes and never enter
time-to-accuracy results. This benchmark does not implement a heat MC backend.
"""

import argparse
import csv
import hashlib
import inspect
import json
import os
import platform
import resource
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numba
import numpy as np
import scipy

from pdelab.boundary3d import (
    BoundaryOperator3D,
    boundary_workspace_bytes3d,
    make_boundary_problem3d,
)
from pdelab.boundary_solvers3d import (
    METHODS_BOUNDARY_3D,
    _walks,
    solve_boundary3d,
    warmup_boundary3d,
)
from pdelab.grouped_mc3d import grouped_workspace_bytes3d
from pdelab.grouped_solver3d import GroupedWalkFailure
from pdelab.problem3d import (
    DIRECT_MAX_N,
    make_problem3d,
    reference_solution3d,
    workspace_bytes,
)
from pdelab.solvers3d import METHODS_3D, _mc3d, solve3d, warmup3d
from pdelab.stationary_study3d import study_stationary3d
from pdelab.study_protocol import PROTOCOL_ID, _seed

ROOT = Path(__file__).resolve().parents[1]
MASTER_SEED = 20261125
WALK_BUDGETS = (64, 256, 1024, 4096, 16384, 65536)
TOLERANCES = (0.1, 0.01, 0.001, 1e-6)
BATCH_SIZES = (1, 32, 128, 512, 2048)
RSS_WALKS = 4096
RSS_TOLERANCE = 1e-6
MAX_ITER = 20000
PUBLISHED_COMMIT = "3bcd7836e4291f101794b9017fa72d254dae9d1a"
PUBLISHED_KERNELS = {
    "_mc3d": {
        "origin_commit": "5c7353e05fc9e9e8863ef49098c5c98b03f1c720",
        "sha256": "912a35d7e82a107208546584c86f2912d99ae5dc5c117b38d5728cd830b38380",
    },
    "_walks": {
        "origin_commit": "1267efb9d3d14b407d928395a321e5533d438951",
        "sha256": "f4b8ca46a1e90ab0027b93021c0037e8a95b48fffb45d1038ba3131315e4a44f",
    },
}
CASES = tuple(
    dict(
        id=f"{family}-{n}-{location}",
        seed_id=family_index * 10 + case_index,
        family=family,
        n=n,
        location=location,
    )
    for family_index, family in enumerate(("dirichlet", "mixed"))
    for case_index, (n, location) in enumerate(
        ((16, "center"), (32, "center"), (32, "offcenter"))
    )
)
VARIANTS = (
    dict(id="lod_serial", method="lod_serial", batch_size=None, seed_id=0),
    dict(id="mc", method="mc", batch_size=None, seed_id=1),
    *(
        dict(id=f"mc_grouped-b{b}", method="mc_grouped", batch_size=b, seed_id=i + 2)
        for i, b in enumerate(BATCH_SIZES)
    ),
)


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def write_json(path, value):
    path.write_text(
        json.dumps(
            value, ensure_ascii=False, indent=2, allow_nan=False, default=_json_default
        )
        + "\n",
        encoding="utf-8",
    )


def write_csv(path, rows):
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: (
                        json.dumps(value, ensure_ascii=False, default=_json_default)
                        if isinstance(value, (list, tuple, dict, np.ndarray))
                        else value
                    )
                    for key, value in row.items()
                }
            )


def _git(*arguments):
    return subprocess.check_output(["git", *arguments], cwd=ROOT, text=True).rstrip(
        "\n"
    )


def source_state():
    paths = sorted((ROOT / "pdelab").rglob("*.py")) + [Path(__file__).resolve()]
    hashes = {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths
    }
    return dict(
        commit=_git("rev-parse", "HEAD"),
        dirty_before_outputs=_git("status", "--porcelain=v1", "--untracked-files=all"),
        source_sha256=hashes,
        combined_sha256=hashlib.sha256(
            json.dumps(hashes, sort_keys=True).encode()
        ).hexdigest(),
    )


def baseline_provenance():
    kernels = {}
    for name, function in (("_mc3d", _mc3d), ("_walks", _walks)):
        current = hashlib.sha256(
            inspect.getsource(function.py_func).encode()
        ).hexdigest()
        kernels[name] = {
            **PUBLISHED_KERNELS[name],
            "current_sha256": current,
            "matches_published": current == PUBLISHED_KERNELS[name]["sha256"],
        }
    return dict(
        published_commit=PUBLISHED_COMMIT,
        kernels=kernels,
        prior_readonly_profile="benchmarks/results/grouped-mc3d/baseline-profile",
        prior_profile_is_separate_evidence=True,
    )


def problem_and_points(case):
    problem = (
        make_problem3d("sine", case["n"])
        if case["family"] == "dirichlet"
        else make_boundary_problem3d("polynomial", case["n"])
    )
    n = case["n"]
    point = (n // 2,) * 3 if case["location"] == "center" else (n // 4, n // 2, n // 4)
    return problem, np.array([point], dtype=np.int64)


def experiment_seed(case, variant, purpose=0):
    return int(
        np.random.SeedSequence(
            [MASTER_SEED, 2901, purpose, case["seed_id"], variant["seed_id"]]
        ).generate_state(1)[0]
    )


def seed_plan(cases, budgets, tolerances, repeats):
    """Enumerate the existing adapter/protocol seeds before observing any values."""
    records, seen = [], set()
    for case in cases:
        names = METHODS_3D if case["family"] == "dirichlet" else METHODS_BOUNDARY_3D
        for variant in VARIANTS:
            study_seed = experiment_seed(case, variant)
            method_seed = int(
                np.random.SeedSequence(
                    [study_seed, 29, list(names).index(variant["method"])]
                ).generate_state(1)[0]
            )
            candidates = tolerances if variant["method"] == "lod_serial" else budgets
            for phase_index, phase in enumerate(("pilot", "final")):
                for index, budget in enumerate(candidates):
                    for repeat in range(repeats):
                        trial_seed = _seed(method_seed, phase_index, index, repeat)
                        if trial_seed in seen:
                            raise RuntimeError(
                                "The fixed trial seed plan has a collision"
                            )
                        seen.add(trial_seed)
                        records.append(
                            dict(
                                case_id=case["id"],
                                variant=variant["id"],
                                study_seed=study_seed,
                                phase=phase,
                                candidate=index,
                                budget=budget,
                                repeat=repeat + 1,
                                seed=trial_seed,
                            )
                        )
    return records


def warmup():
    started = perf_counter()
    warmup3d()
    warmup_boundary3d()
    return perf_counter() - started


def peak_rss_bytes():
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return int(raw)
    if sys.platform.startswith("linux"):
        return int(raw * 1024)
    raise RuntimeError("RSS conversion is defined only for macOS bytes / Linux KiB")


def rss_worker(case, variant, *, budget=None, seed=None):
    """One full solve in a fresh process; high-water marks are never subtracted."""
    stochastic = variant["method"] != "lod_serial"
    walks_budget = RSS_WALKS if budget is None or not stochastic else int(budget)
    tolerance = RSS_TOLERANCE if budget is None or stochastic else float(budget)
    trial_seed = experiment_seed(case, variant, purpose=1) if seed is None else seed
    values = None
    prewarm = peak_rss_bytes()
    warming_seconds = warmup()
    postwarm = peak_rss_bytes()
    problem, points = problem_and_points(case)
    solve = solve3d if case["family"] == "dirichlet" else solve_boundary3d
    status, reason, extras = "completed", "", {}
    walks, events, iterations = None, None, None
    started = perf_counter()
    try:
        result = solve(
            problem,
            points,
            variant["method"],
            walks=walks_budget,
            tol=tolerance,
            seed=trial_seed,
            batch_size=variant["batch_size"] or 128,
            max_iter=MAX_ITER,
        )
        seconds, extras = result.elapsed, result.extras
        walks, events, iterations = result.walks, result.steps, result.iterations
        if not result.converged:
            status, reason = "linear_limit", "LOM tolerance was not reached"
        else:
            values = result.values[tuple(points.T)].tolist()
    except GroupedWalkFailure as exc:
        seconds, extras = exc.seconds, exc.diagnostics
        walks, events = extras["walks"], extras["transitions"]
        status, reason = "path_limit", str(exc)
    except (ValueError, RuntimeError, MemoryError) as exc:
        seconds = perf_counter() - started
        status, reason = "solver_failed", str(exc)
    aftersolve = peak_rss_bytes()
    reserve = (
        boundary_workspace_bytes3d if case["family"] == "mixed" else workspace_bytes
    )
    workspace_estimate = extras.get(
        "workspace_bytes", reserve(case["n"]) + (8 * walks_budget if stochastic else 0)
    )
    grouped = variant["method"] == "mc_grouped"
    return dict(
        case_id=case["id"],
        variant=variant["id"],
        method=variant["method"],
        batch_size=variant["batch_size"],
        n=case["n"],
        location=case["location"],
        fixed_walk_budget=walks_budget if stochastic else None,
        fixed_lod_tolerance=tolerance if not stochastic else None,
        seed=trial_seed,
        values=values,
        status=status,
        reason=reason,
        diagnostic_solve_seconds=seconds,
        preparation_seconds=extras.get("preparation_seconds"),
        path_seconds=extras.get("path_seconds"),
        completed_walks=walks,
        started_walks=extras.get("started_walks", walks),
        path_events=events,
        path_event_definition="reward/kill/proposal stages, including kill without movement",
        iterations=iterations,
        warmup_seconds=warming_seconds,
        prewarm_peak_rss_bytes=prewarm,
        postwarm_peak_rss_bytes=postwarm,
        aftersolve_peak_rss_bytes=aftersolve,
        process_peak_rss_bytes=aftersolve,
        rss_scope="whole worker through solve; includes imports, JIT/warmup, input and solve",
        rss_raw_units="bytes" if sys.platform == "darwin" else "KiB",
        workspace_estimate_bytes=workspace_estimate,
        workspace_estimate_kind="conservative numerical arrays; not measured RSS",
        workspace_estimate_source=(
            "Result.extras.workspace_bytes"
            if "workspace_bytes" in extras
            else f"{reserve.__module__}.{reserve.__name__} + first-point samples"
        ),
        batch_aux_exact_bytes=(
            48 * min(variant["batch_size"], walks_budget) if grouped else 0
        ),
        batch_aux_definition="six int64/float64 particle arrays; array payload only",
        kernel_array_payload_bytes=(
            grouped_workspace_bytes3d(walks_budget, variant["batch_size"], len(points))
            if grouped
            else None
        ),
        time_study_observation=False,
    )


def rss_diagnostics(case, variant, *, budget=None, seed=None):
    environment = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    command = [
        sys.executable,
        "-m",
        "benchmarks.grouped_mc3d",
        "--rss-worker",
        case["id"],
        variant["id"],
    ]
    if budget is not None:
        command.extend(["--rss-budget", str(budget)])
    if seed is not None:
        command.extend(["--rss-seed", str(seed)])
    try:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=environment,
            text=True,
            capture_output=True,
            timeout=180,
        )
    except subprocess.TimeoutExpired:
        return dict(
            case_id=case["id"],
            variant=variant["id"],
            status="worker_timeout",
            reason="Separate diagnostic worker exceeded 180 seconds",
            time_study_observation=False,
        )
    if completed.returncode:
        return dict(
            case_id=case["id"],
            variant=variant["id"],
            status="worker_failed",
            reason=completed.stderr.strip(),
            time_study_observation=False,
        )
    return json.loads(completed.stdout)


def reference_diagnostic(problem, points, case, use_csr):
    analytical = problem.exact[tuple(points.T)]
    record = dict(
        case_id=case["id"],
        reference_kind="analytical_continuous_manufactured_solution",
        reference_values=analytical.tolist(),
        points_zyx=points.tolist(),
        source=(
            "Problem3D.exact"
            if case["family"] == "dirichlet"
            else "BoundaryProblem3D.exact"
        ),
        analytical_reference_used_by_solver=False,
        csr_status="not_requested",
        diagnostic_seconds=0.0,
    )
    if use_csr:
        if problem.n > DIRECT_MAX_N:
            record["csr_status"] = "skipped_direct_grid_limit"
        else:
            started = perf_counter()
            field = (
                reference_solution3d(problem)
                if case["family"] == "dirichlet"
                else BoundaryOperator3D(problem).solve()
            )
            discrete = field[tuple(points.T)]
            record.update(
                csr_status="completed",
                csr_values=discrete.tolist(),
                csr_analytical_difference=(discrete - analytical).tolist(),
                diagnostic_seconds=perf_counter() - started,
            )
    return analytical, record


def run(args):
    source = source_state()  # Must precede creation of any output file/directory.
    provenance = baseline_provenance()
    if not args.smoke and source["dirty_before_outputs"]:
        raise ValueError(
            "Commit the measured source before a full run; use --smoke for development"
        )
    if not all(item["matches_published"] for item in provenance["kernels"].values()):
        raise ValueError(
            "Published baseline kernels changed; review baseline provenance first"
        )
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError(
            "Output directory must be new or empty; existing evidence is not overwritten"
        )
    cases = [case for case in CASES if not args.smoke or case["n"] == 16]
    budgets = (16, 64) if args.smoke else WALK_BUDGETS
    tolerances = (0.1, 1e-6) if args.smoke else TOLERANCES
    repeats = 2 if args.smoke else 5
    planned_seeds = seed_plan(cases, budgets, tolerances, repeats)
    expected_seeds = {
        (r["case_id"], r["variant"], r["phase"], r["candidate"], r["repeat"]): r["seed"]
        for r in planned_seeds
    }
    manifest = dict(
        experiment="stationary_grouped_mc3d_fixed_pilot_v1",
        scope="early published stationary uniform-k=1 MC; no heat #28 backend",
        mode="smoke_not_performance_evidence" if args.smoke else "full",
        started_utc=datetime.now(timezone.utc).isoformat(),
        run_state="running",
        source_before_outputs=source,
        baseline_provenance=provenance,
        grouping_control=(
            "B=1 is mandatory: grouped Dirichlet precomputes -f/(6*n^2), while "
            "published scalar Dirichlet divides each visit. A baseline speed ratio "
            "cannot all be attributed to batching. Mixed scalar already uses "
            "prepared rewards; B=1 reproduces its samples. Larger batches change "
            "MT19937 draw order while preserving the transition law."
        ),
        python=sys.version,
        executable=sys.executable,
        platform=platform.platform(),
        versions=dict(
            numpy=np.__version__, scipy=scipy.__version__, numba=numba.__version__
        ),
        thread_environment={
            key: os.environ.get(key)
            for key in (
                "OPENBLAS_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
                "NUMBA_NUM_THREADS",
            )
        },
        protocol=PROTOCOL_ID,
        calibration_fraction=0.8,
        targets_pct=[1.0, 5.0, 10.0],
        cases=cases,
        mixed_faces=dict(
            x0="robin",
            x1="dirichlet",
            y0="dirichlet",
            y1="dirichlet",
            z0="dirichlet",
            z1="dirichlet",
        ),
        mixed_kappa=1.0,
        variants=VARIANTS,
        walk_budgets=budgets,
        tolerances=tolerances,
        repeats=repeats,
        master_seed=MASTER_SEED,
        seed_derivation="SeedSequence([master,2901,purpose,case_id,variant_id]); existing adapter and protocol then derive method/phase/candidate/repeat seeds",
        max_iter=MAX_ITER,
        max_steps="solver defaults:100*n^2 Dirichlet;1000*n^2 mixed; per path",
        seconds_per_phase=60.0,
        reference="same analytical query values for every method and batch variant",
        csr_diagnostic_requested=args.csr_diagnostic,
        timed_cost="public Result.elapsed: preparation+kernel+outputs; shared sampled problem, analytical/CSR reference, JIT warmup, exports excluded",
        operator_reuse=False,
        rss=dict(
            walks=RSS_WALKS,
            lod_tolerance=RSS_TOLERANCE,
            separate_fresh_warmed_worker=True,
            high_water_marks_subtracted=False,
        ),
        settings_frozen_before_observations=True,
    )
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "manifest.json", manifest)
    write_csv(out / "planned_seeds.csv", planned_seeds)
    manifest["warmup_seconds"] = warmup()
    combined = dict(rows=[], trials=[], series=[], references=[], memory=[])
    try:
        for case in cases:
            problem, points = problem_and_points(case)
            reference, diagnostic = reference_diagnostic(
                problem, points, case, args.csr_diagnostic
            )
            combined["references"].append(diagnostic)
            for variant in VARIANTS:
                print(f"{case['id']} / {variant['id']}", flush=True)
                result = study_stationary3d(
                    problem,
                    points,
                    reference,
                    [variant["method"]],
                    walk_budgets=budgets,
                    tolerances=tolerances,
                    repeats=repeats,
                    seed=experiment_seed(case, variant),
                    batch_size=variant["batch_size"] or 128,
                    max_iter=MAX_ITER,
                    seconds_per_phase=60.0,
                )
                for trial in result["trials"]:
                    key = (
                        case["id"],
                        variant["id"],
                        trial["phase"],
                        trial["candidate"],
                        trial["repeat"],
                    )
                    if trial["seed"] != expected_seeds[key]:
                        raise RuntimeError(
                            "Study seed derivation changed after the plan was frozen"
                        )
                context = dict(
                    case_id=case["id"],
                    variant=variant["id"],
                    family=case["family"],
                    n=case["n"],
                    location=case["location"],
                    points_zyx=points.tolist(),
                    reference_values=reference.tolist(),
                )
                for collection in ("rows", "trials", "series"):
                    for row in result[collection]:
                        row.update(context)
                    combined[collection].extend(result[collection])
                # Checkpoint every completed study; retain all failed/final attempts.
                for collection, filename in (
                    ("rows", "results"),
                    ("trials", "trials"),
                    ("series", "series"),
                ):
                    write_csv(out / f"{filename}.csv", combined[collection])
                write_json(out / "results.json", combined)
        # Memory diagnostics cannot perturb the timed study in its parent process.
        for case in cases:
            for variant in VARIANTS:
                combined["memory"].append(rss_diagnostics(case, variant))
                write_json(out / "memory.json", combined["memory"])
                write_csv(out / "memory.csv", combined["memory"])
        after = source_state()
        manifest["source_after_run"] = after
        unchanged = (
            source["combined_sha256"] == after["combined_sha256"]
            and source["commit"] == after["commit"]
        )
        manifest["source_unchanged_during_run"] = unchanged
        manifest["run_state"] = "completed" if unchanged else "invalid_source_changed"
        write_json(out / "results.json", combined)
        if not unchanged and not args.smoke:
            raise RuntimeError("Measured source changed during the full run")
    except BaseException as exc:
        manifest.update(
            run_state="interrupted_or_failed", failure=f"{type(exc).__name__}: {exc}"
        )
        raise
    finally:
        manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(out / "manifest.json", manifest)
    print(out, flush=True)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--out", type=Path, default=Path("runs/grouped-mc3d"))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--csr-diagnostic", action="store_true")
    parser.add_argument(
        "--rss-worker", nargs=2, metavar=("CASE", "VARIANT"), help=argparse.SUPPRESS
    )
    parser.add_argument("--rss-budget", type=float, help=argparse.SUPPRESS)
    parser.add_argument("--rss-seed", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.rss_worker:
        case = next(c for c in CASES if c["id"] == args.rss_worker[0])
        variant = next(v for v in VARIANTS if v["id"] == args.rss_worker[1])
        print(
            json.dumps(
                rss_worker(case, variant, budget=args.rss_budget, seed=args.rss_seed),
                default=_json_default,
                allow_nan=False,
            )
        )
    else:
        run(args)


if __name__ == "__main__":
    main()
