"""Reproduce with python -m benchmarks.heat3d_block; no plots in solver timings.

The default matrix uses n=4,8,16, alpha=0,.5, and five repeats per series.
Use --n 4 --repeats 2 --out /tmp/heat3d-block-smoke for a short smoke run.
Newton is a separate diagnostic, never a candidate in the accuracy study.
"""

import argparse
import hashlib
import os
import platform
import shlex
import subprocess
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from pdelab.heat3d_experiment import (
    HeatConfig3D,
    compare_heat3d,
    csv_bytes,
    json_bytes,
    study_heat3d,
)
from pdelab.heat3d_solver import HeatSettings3D

ROOT = Path(__file__).resolve().parents[1]
BUDGETS = (1e-3, 1e-5, 1e-7)
THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMBA_NUM_THREADS",
)


def _git(*arguments):
    try:
        return subprocess.check_output(
            ["git", *arguments], cwd=ROOT, stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _summary(n, alpha, run, study):
    rows = []
    diagnostic = run["grid_diagnostic"]
    for row in [*run["rows"], diagnostic]:
        method = row.get("method", "newton")
        rows.append(
            dict(
                n=n,
                alpha=alpha,
                phase="diagnostic" if method == "newton" else "strict",
                method=method,
                target_pct=None,
                status=row["status"],
                tolerance=1e-11 if method == "newton" else 1e-7,
                relative_error_pct=row.get("relative_error_pct"),
                seconds=row.get("seconds"),
                time_basis="single_full_run",
                pilot_seconds=None,
                final_seconds=None,
                completed_repeats=None,
                requested_repeats=None,
                preparation_seconds=row.get("preparation_seconds"),
                assembly_seconds=row.get("assembly_seconds"),
                factorization_seconds=row.get("factorization_seconds"),
                solve_seconds=row.get("solve_seconds"),
                factorizations=row.get("factorizations"),
                factor_reuses=row.get("factor_reuses"),
                workspace_bytes=row.get("workspace_bytes"),
                peak_rss_bytes=row.get("peak_rss_bytes"),
                relative_to_newton_pct=diagnostic.get(method + "_vs_grid", {}).get(
                    "relative_error_pct"
                ),
                reason=row.get("reason", ""),
            )
        )
    for row in study["rows"]:
        series = [s for s in study["series"] if s["method"] == row["method"]]
        rows.append(
            dict(
                n=n,
                alpha=alpha,
                phase="study",
                method=row["method"],
                target_pct=row["target_pct"],
                status=row["status"],
                tolerance=row["budget"],
                relative_error_pct=row["relative_error_pct"],
                seconds=row["seconds"],
                time_basis="mean_full_run_confirmed_final",
                pilot_seconds=sum(
                    s["elapsed_seconds"] for s in series if s["phase"] == "pilot"
                ),
                final_seconds=sum(
                    s["elapsed_seconds"] for s in series if s["phase"] == "final"
                ),
                completed_repeats=row["completed_repeats"],
                requested_repeats=row["requested_repeats"],
                workspace_bytes=row["workspace_bytes"],
                peak_rss_bytes=row["peak_rss_bytes"],
                reason=row["reason"],
            )
        )
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path, default=Path("benchmarks/results/heat3d-block")
    )
    parser.add_argument(
        "--n", nargs="+", type=int, choices=(4, 8, 16), default=[4, 8, 16]
    )
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--newton-max-n",
        type=int,
        choices=(0, 4, 8, 16),
        default=16,
        help="Largest grid for the separate Newton diagnostic; 0 disables it",
    )
    args = parser.parse_args(argv)
    if len(set(args.n)) != len(args.n):
        parser.error("--n must contain distinct grids")
    if not 2 <= args.repeats <= 100:
        parser.error("--repeats must be between 2 and 100")

    started = datetime.now(timezone.utc).isoformat()
    git_status = _git("status", "--porcelain")
    command = [
        sys.executable,
        "-m",
        "benchmarks.heat3d_block",
        "--n",
        *(str(n) for n in args.n),
        "--repeats",
        str(args.repeats),
        "--newton-max-n",
        str(args.newton_max_n),
        "--out",
        str(args.out),
    ]
    manifest = dict(
        schema_version="heat3d_block_benchmark_v1",
        started_utc=started,
        commit=_git("rev-parse", "HEAD"),
        working_tree_dirty=None if git_status is None else bool(git_status),
        git_status_porcelain=git_status,
        reproduction=shlex.join(command),
        run_from=str(ROOT),
        grids=args.n,
        alphas=[0.0, 0.5],
        case="sine",
        query="centre point (n//2,n//2,n//2), public xyz coordinates",
        dt=0.01,
        T=0.02,
        methods=["lod", "block"],
        strict_tolerance=1e-7,
        budgets=list(BUDGETS),
        repeats=args.repeats,
        seed=42,
        calibration_fraction=0.8,
        newton_max_n=args.newton_max_n,
        python=sys.version,
        executable=sys.executable,
        platform=platform.platform(),
        cpu_count=os.cpu_count(),
        thread_environment={key: os.environ.get(key) for key in THREAD_VARIABLES},
        source_sha256={
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [
                Path(__file__).resolve(),
                *sorted((ROOT / "pdelab").glob("*.py")),
            ]
        },
        notes=[
            "Each method uses the unchanged fixed-candidate protocol: all pilots precede its independent final series; final failure never retunes.",
            "Study seconds is a confirmed final mean, otherwise null. Errors average individual full-run relative L2 norms.",
            "pilot_seconds and final_seconds are per-method totals repeated for each target; do not sum them across target rows.",
            "Stage times and factor counters in summary.csv describe the strict single run only. All trial and series costs remain in the study JSON.",
            "Stage times are subsets of solver seconds; assembly is also included in preparation. workspace_bytes is estimated, not measured peak RSS.",
            "Newton is requested once per case up to newton_max_n with the same 60-second soft limit and 256 MiB workspace limit. Its cost is separate; failure remains diagnostic evidence.",
            "Solver timings exclude JIT warmup, analytical references, UI and export. Strict runs precede studies; method order is lod then block.",
            "Outputs are written only after all solves so this run's generated files do not change intermediate working-tree metadata.",
        ],
        results=[],
    )
    settings = HeatSettings3D(tolerance=1e-7, change_tolerance=1e-7)
    collected, summary = [], []
    for n in args.n:
        for alpha in (0.0, 0.5):
            config = HeatConfig3D(
                case="sine",
                n=n,
                alpha=alpha,
                dt=0.01,
                T=0.02,
                mode="point",
                point=(n // 2,) * 3,
                methods=("lod", "block"),
                newton=n <= args.newton_max_n,
            )
            run = compare_heat3d(config, settings)
            study = study_heat3d(
                replace(config, newton=False),
                settings,
                repeats=args.repeats,
                seed=42,
                budgets=BUDGETS,
                calibration_fraction=0.8,
            )
            name = f"n{n:02d}-alpha{alpha:g}"
            collected.extend([(name + "-run.json", run), (name + "-study.json", study)])
            manifest["results"].append(
                dict(
                    n=n, alpha=alpha, run=name + "-run.json", study=name + "-study.json"
                )
            )
            summary.extend(_summary(n, alpha, run, study))
            print(
                f"n={n}, alpha={alpha:g}: "
                + ", ".join(f"{row['method']}={row['status']}" for row in run["rows"])
                + f"; Newton={run['grid_diagnostic']['status']}; "
                + f"confirmed targets={sum(row['status'] == 'confirmed' for row in study['rows'])}/{len(study['rows'])}",
                flush=True,
            )
    manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
    args.out.mkdir(parents=True, exist_ok=True)
    for name, result in collected:
        (args.out / name).write_bytes(json_bytes(result))
    (args.out / "summary.csv").write_bytes(csv_bytes(summary))
    (args.out / "manifest.json").write_bytes(json_bytes(manifest))
    print(
        f"Saved {len(summary)} summary rows and {len(collected)} raw results to {args.out.resolve()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
