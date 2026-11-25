"""PDE-27 stage 1: audit existing stationary backends, not a heat solver.

Run: python -m pdelab.boundary_audit3d --out runs/boundary-audit
"""

import argparse
import csv
import hashlib
import io
import json
import os
import platform
import subprocess
import sys
import zipfile
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import numba
import numpy as np
import scipy

from .boundary import BoundaryCondition
from .boundary3d import (
    FACES,
    BoundaryProblem3D,
    boundary_workspace_bytes3d,
)
from .boundary_solvers3d import solve_boundary3d, warmup_boundary3d
from .heat3d_problem import make_heat_problem3d
from .measurement_protocol import (
    ACCURACY_METRIC,
    SCHEMA_VERSION,
    SELECTION_POLICY,
    Trial,
    run_seed,
    summarize,
)

VARIANTS = ("all_dirichlet", "y0_robin", "all_robin")
TARGETS = (1.0, 5.0, 10.0)
METHODS = ("lod_serial", "mc")


def stationary_snapshot(kind, n, variant):
    """Manufactured t=0 snapshot, k=1: old Delta(u)=f convention.

    This is deliberately NOT the heat equation at alpha=0. The time derivative
    is subtracted from manufactured F: f=u_t-F=Delta(u*).
    """
    if variant not in VARIANTS:
        raise ValueError("Unknown boundary variant")
    kinds = {
        face: (
            3
            if variant == "all_robin" or (variant == "y0_robin" and face == "y0")
            else 1
        )
        for face in FACES
    }
    heat = make_heat_problem3d(kind, n, alpha=0, kinds=kinds, beta=1)
    exact = heat.initial_values()
    ut = 1 - exact if kind == "sine" else np.ones_like(exact)
    f = ut - heat.evaluate(heat.source, 0)
    faces = {
        face: BoundaryCondition(
            "dirichlet" if kinds[face] == 1 else "robin",
            heat.face_values(face, 0),
            0 if kinds[face] == 1 else 1,
        )
        for face in FACES
    }
    # Exact is kept ONLY by the experiment; neither backend receives it.
    return BoundaryProblem3D(n, f, faces, name=f"snapshot-{kind}-{variant}"), exact


def json_write(path, value):
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def csv_write(path, rows):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, list(dict.fromkeys(k for row in rows for k in row)))
    writer.writeheader()
    writer.writerows(rows)
    path.write_bytes(buffer.getvalue().encode("utf-8-sig"))


def provenance():
    root = Path(__file__).resolve().parents[1]

    def git(*args):
        return subprocess.check_output(
            ["git", "-C", str(root), *args], text=True
        ).strip()

    files = sorted(
        str(path.relative_to(root)).replace("\\", "/")
        for path in (root / "pdelab").glob("*.py")
    ) + ["pyproject.toml"]
    return dict(
        commit=git("rev-parse", "HEAD"),
        dirty=bool(git("status", "--porcelain")),
        source_sha256={
            name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in files
        },
        environment=dict(
            python=sys.version,
            os=platform.platform(),
            cpu=platform.processor(),
            numpy=np.__version__,
            scipy=scipy.__version__,
            numba=numba.__version__,
            threads={
                k: os.environ.get(k)
                for k in (
                    "OMP_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "NUMBA_NUM_THREADS",
                )
            },
        ),
    )


def measure_series(
    kind, n, variant, method, budget, points, reference, seeds, target, phase, remaining
):
    """Full fresh solves, including manufacture/sampling; stop between runs.

    Legacy kernels raise without partial counters. Preserve that failure and
    its wall time separately; never invent zero paths for interrupted work.
    """
    trials, failures, diagnostics = [], [], []
    phase_start = perf_counter()
    for repeat, seed in enumerate(seeds):
        if perf_counter() - phase_start >= remaining:
            break
        start = perf_counter()
        try:
            problem, _ = stationary_snapshot(kind, n, variant)
            prepared = perf_counter() - start
            result = solve_boundary3d(
                problem,
                points,
                method,
                walks=int(budget) if method == "mc" else 2,
                seed=seed,
                tol=float(budget) if method == "lod_serial" else 1e-8,
                max_iter=20000,
                max_steps=1000000,
            )
            seconds = perf_counter() - start
            completed = result.converged
            trials.append(
                Trial(
                    repeat=repeat,
                    seed=seed,
                    status="completed" if completed else "linear_limit",
                    reason="" if completed else "Iteration limit without convergence",
                    values=(
                        result.values[tuple(points.T)].tolist() if completed else None
                    ),
                    seconds=seconds,
                    preparation_seconds=prepared,
                    workspace_bytes=boundary_workspace_bytes3d(n)
                    + (int(budget) * 8 if method == "mc" else 0),
                    peak_rss_bytes=None,
                    paths_per_estimate=int(budget) if method == "mc" else None,
                    walks=int(result.walks),
                    transitions=int(result.steps),
                    physical_steps=0,
                    outer_iterations=0,
                    inner_iterations=int(result.iterations),
                    completed_until=None,
                )
            )
            diagnostics.append(
                dict(
                    repeat=repeat,
                    residual=result.residual,
                    relative_residual=result.relative_residual,
                    residual_scale=result.extras.get("residual_scale"),
                    stopping_rule=result.extras.get("stopping_rule"),
                )
            )
        except (RuntimeError, ValueError, MemoryError) as exc:
            failures.append(
                dict(
                    repeat=repeat,
                    seed=seed,
                    status="backend_exception",
                    reason=str(exc),
                    seconds=perf_counter() - start,
                    walks=None,
                    transitions=None,
                    counter_reason="Legacy exception exposes no partial counters",
                )
            )
            break
    summary = summarize(
        trials, reference, requested_repeats=len(seeds), target_pct=target, phase=phase
    )
    summary["spent_seconds"] += sum(f["seconds"] for f in failures)
    if failures:
        summary.update(walks=None, transitions=None, reason=failures[-1]["reason"])
    return dict(
        schema_version=SCHEMA_VERSION,
        summary=summary,
        trials=[asdict(t) for t in trials],
        failures=failures,
        diagnostics=diagnostics,
        reference_values=np.asarray(reference).tolist(),
        budget=budget,
        elapsed_seconds=perf_counter() - phase_start,
    )


def confirmed_ratio(numerator, denominator):
    if numerator["status"] != "confirmed" or denominator["status"] != "confirmed":
        return None
    return (
        numerator["seconds_to_target"] / denominator["seconds_to_target"]
        if denominator["seconds_to_target"] > 0
        else None
    )


def run(
    out,
    *,
    n=16,
    repeats=5,
    root_seed=20261125,
    seconds=120,
    kinds=("sine", "polynomial"),
    point="center",
):
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise ValueError("Output must be empty; do not overwrite earlier attempts")
    out.mkdir(parents=True, exist_ok=True)
    if n % 4 or repeats < 2 or not np.isfinite(seconds) or seconds <= 0:
        raise ValueError("Need n divisible by 4, repeats>=2 and a positive phase limit")
    points = np.array(
        [[n // 2] * 3] if point == "center" else [[3 * n // 4, n // 2, n // 4]]
    )
    budgets = {
        "mc": (64, 256, 1024, 4096, 16384, 65536),
        "lod_serial": (0.01, 0.001, 0.0001, 0.00001),
    }
    meta = dict(
        **provenance(),
        accuracy_metric=ACCURACY_METRIC,
        selection_policy=SELECTION_POLICY,
        n=n,
        repeats=repeats,
        root_seed=root_seed,
        points_zyx=points.tolist(),
        query_xyz=(points[:, ::-1] / n).tolist(),
        mode="stationary_3d_k1_snapshot_t0",
        beta=1,
        targets_pct=TARGETS,
        budgets=budgets,
        phase_seconds=seconds,
        max_iter=20000,
        max_steps=1000000,
        scenario="cold_solve",
        reference="analytic from #24, t=0",
        workspace_method="64 dense volumes plus MC samples; estimate, not measured peak",
        peak_rss_bytes=None,
        peak_rss_reason="No isolated process RSS measurement in this pilot",
        timing="Fresh data sampling plus solve/mandatory checks. JIT, external reference, metric, export excluded. Soft phase limit between full runs.",
        reproduction=f"python -m pdelab.boundary_audit3d --n {n} --repeats {repeats} --seed {root_seed} --seconds {seconds} --functions {' '.join(kinds)} --point {point} --out runs/boundary-audit-reproduction",
    )
    source_root = Path(__file__).resolve().parents[1]
    with zipfile.ZipFile(out / "source.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for name in meta["source_sha256"]:
            archive.write(source_root / name, name)
    json_write(out / "protocol.json", meta)
    started = perf_counter()
    warmup_boundary3d()
    meta["warmup_seconds"] = perf_counter() - started
    rows, attempts, selections = [], [], []
    for kind in kinds:
        for variant_index, variant in enumerate(VARIANTS):
            case_id = f"{kind}-{variant}"
            case_index = (0 if kind == "sine" else 1) * 3 + variant_index
            started = perf_counter()
            _, exact = stationary_snapshot(kind, n, variant)
            reference = exact[tuple(points.T)]
            reference_seconds = perf_counter() - started
            for method_index, method in enumerate(METHODS):
                case_rows, selected = {}, {}
                spent = {"pilot": 0.0, "control": 0.0}
                last_pilot = None
                for budget_index, budget in enumerate(budgets[method]):
                    if len(selected) == len(TARGETS) or spent["pilot"] >= seconds:
                        break
                    # Legacy NumPy RandomState backends require uint32. The saved
                    # seed is the actual input, not the untruncated uint64 source.
                    seeds = [
                        run_seed(
                            root_seed,
                            case_index,
                            method_index,
                            budget_index,
                            "pilot",
                            r,
                        )
                        % 2**32
                        for r in range(repeats)
                    ]
                    data = measure_series(
                        kind,
                        n,
                        variant,
                        method,
                        budget,
                        points,
                        reference,
                        seeds,
                        1,
                        "pilot",
                        seconds - spent["pilot"],
                    )
                    data.update(case=case_id, method=method, budget_index=budget_index)
                    attempts.append(data)
                    last_pilot = data
                    spent["pilot"] += data["elapsed_seconds"]
                    error = data["summary"]["relative_error_pct"]
                    for target in TARGETS:
                        margin = 0.5 if method == "mc" else 1.0
                        if (
                            target not in selected
                            and error is not None
                            and error <= target * margin
                        ):
                            selected[target] = dict(
                                budget=budget, budget_index=budget_index
                            )
                choice = dict(
                    case=case_id,
                    method=method,
                    selected={str(k): v for k, v in selected.items()},
                    status="frozen_before_control",
                    pilot_seconds=spent["pilot"],
                )
                selections.append(choice)
                # Persist every selection BEFORE any holdout for this case/method.
                json_write(out / "selection.json", selections)
                control = {}
                for target in TARGETS:
                    if target not in selected:
                        case_rows[target] = dict(
                            status="not_selected",
                            reason="No pilot candidate within fixed margin and limits",
                            relative_error_pct=None,
                            seconds_to_target=None,
                            last_pilot_budget=(
                                last_pilot["budget"] if last_pilot else None
                            ),
                            last_pilot_error_pct=(
                                last_pilot["summary"]["relative_error_pct"]
                                if last_pilot
                                else None
                            ),
                            requested_repeats=repeats,
                            completed_repeats=0,
                        )
                        continue
                    choice = selected[target]
                    budget, budget_index = choice["budget"], choice["budget_index"]
                    if budget_index not in control:
                        seeds = [
                            run_seed(
                                root_seed,
                                case_index,
                                method_index,
                                budget_index,
                                "control",
                                r,
                            )
                            % 2**32
                            for r in range(repeats)
                        ]
                        data = measure_series(
                            kind,
                            n,
                            variant,
                            method,
                            budget,
                            points,
                            reference,
                            seeds,
                            target,
                            "control",
                            max(0, seconds - spent["control"]),
                        )
                        data.update(
                            case=case_id, method=method, budget_index=budget_index
                        )
                        attempts.append(data)
                        spent["control"] += data["elapsed_seconds"]
                        control[budget_index] = data
                    data = control[budget_index]
                    # Reuse the SAME frozen control data for looser goals; no rerun.
                    trials = [Trial(**t) for t in data["trials"]]
                    result = summarize(
                        trials,
                        reference,
                        requested_repeats=repeats,
                        target_pct=target,
                        phase="control",
                    )
                    result["spent_seconds"] = data["summary"]["spent_seconds"]
                    if data["failures"]:
                        result.update(
                            walks=None,
                            transitions=None,
                            reason=data["failures"][-1]["reason"],
                        )
                    case_rows[target] = dict(**result, budget=budget)
                for target, result in case_rows.items():
                    rows.append(
                        dict(
                            case=case_id,
                            method=method,
                            target_pct=target,
                            pilot_seconds=spent["pilot"],
                            control_seconds=spent["control"],
                            reference_seconds=reference_seconds,
                            **{k: v for k, v in result.items() if k != "target_pct"},
                        )
                    )
                json_write(out / "attempts.json", attempts)
                csv_write(out / "summary.csv", rows)
                print(
                    f"{case_id} {method}: "
                    + ", ".join(f"{t:g}% {case_rows[t]['status']}" for t in TARGETS),
                    flush=True,
                )
    ratios = []
    for kind in kinds:
        for target in TARGETS:
            lookup = {
                (r["case"], r["method"]): r for r in rows if r["target_pct"] == target
            }
            baseline = lookup[(f"{kind}-all_dirichlet", "mc")]
            for variant in VARIANTS:
                lod, mc = [lookup[(f"{kind}-{variant}", m)] for m in METHODS]
                ratios.append(
                    dict(
                        case=f"{kind}-{variant}",
                        target_pct=target,
                        lod_over_mc=confirmed_ratio(lod, mc),
                        mc_over_dirichlet_mc=confirmed_ratio(mc, baseline),
                        reason="Ratios require both complete controls to confirm the same target",
                    )
                )
    csv_write(out / "ratios.csv", ratios)
    json_write(out / "protocol.json", meta)
    json_write(out / "results.json", dict(protocol=meta, summary=rows, ratios=ratios))
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--n", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20261125)
    parser.add_argument("--seconds", type=float, default=120)
    parser.add_argument(
        "--functions",
        choices=("sine", "polynomial"),
        nargs="+",
        default=["sine", "polynomial"],
    )
    parser.add_argument("--point", choices=("center", "offcenter"), default="center")
    args = parser.parse_args(argv)
    run(
        args.out,
        n=args.n,
        repeats=args.repeats,
        root_seed=args.seed,
        seconds=args.seconds,
        kinds=tuple(args.functions),
        point=args.point,
    )


if __name__ == "__main__":
    main()
