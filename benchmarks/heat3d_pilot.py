"""Run a small, sequential experiment through the published heat3d study API."""

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, stdev
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMBA_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def git(*arguments):
    return subprocess.check_output(["git", *arguments], cwd=ROOT, text=True).strip()


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def source_state():
    paths = [
        *sorted((ROOT / "benchmarks").glob("*.py")),
        *sorted((ROOT / "pdelab").glob("*.py")),
    ]
    return {
        "commit": git("rev-parse", "HEAD"),
        "dirty": bool(git("status", "--porcelain")),
        "sha256": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in paths
        },
    }


def jobs(plan):
    """Expand explicit scenarios, preserving each variant's optional settings."""
    from pdelab.heat3d_experiment import METHOD_NAMES, HeatConfig3D
    from pdelab.heat3d_solver import HeatSettings3D

    seen = set()
    for case in plan["cases"]:
        for variant in plan["variants"]:
            name = case["id"] + "--" + variant["id"]
            if not re.fullmatch(r"[a-zA-Z0-9_-]+", name) or name in seen:
                raise ValueError("Case/variant identifiers must be unique safe names")
            seen.add(name)
            method = variant["method"]
            if method not in METHOD_NAMES:
                raise ValueError(f"Method {method} is unavailable in this version")
            config = HeatConfig3D(
                **{
                    **plan["configuration"],
                    **case["configuration"],
                    "methods": (method,),
                }
            )
            settings = HeatSettings3D(
                **{**plan["settings"], **variant.get("settings", {})}
            )
            settings.for_method(method).validate()
            yield name, config, settings


def worker(plan, job, out):
    import resource

    from pdelab.heat3d_experiment import json_bytes, study_heat3d

    _, config, settings = list(jobs(plan))[job]
    start = perf_counter()
    if plan.get("experiment") == "fixed_budget":
        from benchmarks.heat3d_fixed import fixed_runs

        result = {"fixed_budget": fixed_runs(plan, config, settings)}
    else:
        result = {
            "study": study_heat3d(
                config,
                settings,
                repeats=plan["repeats"],
                seed=plan["seed"],
                budgets=plan["budgets"],
                walk_budgets=plan["walk_budgets"],
                calibration_fraction=plan["calibration_fraction"],
            )
        }
    wall = perf_counter() - start
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak = raw if sys.platform == "darwin" else raw * 1024
    payload = {
        **result,
        "process": {
            "study_call_seconds": wall,
            "peak_rss_bytes_before_export": peak,
            "rss_scope": "fresh process including imports, JIT and all study trials",
            "thread_environment": {k: os.environ.get(k) for k in THREAD_VARIABLES},
        },
    }
    out.write_bytes(json_bytes(payload))


def summarize(results):
    """Reuse protocol rows and retain costs of attempts that did not finish."""
    targets, series, trials, costs = [], [], [], []
    for name, payload in results:
        study = payload["study"]
        for collection, destination in (
            ("rows", targets),
            ("series", series),
            ("trials", trials),
        ):
            for row in study[collection]:
                destination.append({"scenario_variant": name, **row})
        costs.append(
            {
                "scenario_variant": name,
                "pilot_seconds": study["pilot_seconds"],
                "final_seconds": study["final_seconds"],
                "completed_runs": sum(
                    t["status"] == "completed" for t in study["trials"]
                ),
                "attempted_runs": len(study["trials"]),
                "paths_started": sum(t["walks"] for t in study["trials"]),
                "paths_completed": sum(
                    t.get("completed_walks", 0) for t in study["trials"]
                ),
                "transitions": sum(t["transitions"] for t in study["trials"]),
                **payload["process"],
            }
        )
        for row in targets[-len(study["rows"]) :]:
            matching = [
                t
                for t in study["trials"]
                if (t["phase"], t["candidate"]) == (row["phase"], row["candidate"])
            ]
            durations = [t["seconds"] for t in matching]
            row["observed_seconds_std"] = (
                stdev(durations) if len(durations) > 1 else None
            )
            row["observed_seconds_min"] = min(durations)
            row["observed_seconds_max"] = max(durations)
            row["observed_preparation_mean"] = mean(
                t["preparation_seconds"] for t in matching
            )
    return targets, series, trials, costs


def plot(series, out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(dict.fromkeys(r["scenario_variant"].split("--")[0] for r in series))
    fig, axes = plt.subplots(len(names), 1, figsize=(8, 3 * len(names)), squeeze=False)
    for name, ax in zip(names, axes[:, 0], strict=True):
        rows = [r for r in series if r["scenario_variant"].split("--")[0] == name]
        for variant in dict.fromkeys(r["scenario_variant"] for r in rows):
            complete = [
                r for r in rows if r["scenario_variant"] == variant and r["complete"]
            ]
            ax.scatter(
                [r["mean_seconds"] for r in complete],
                [r["relative_error_pct"] for r in complete],
                label=variant.split("--")[1],
                s=35,
            )
        ax.set(
            title=name,
            xlabel="Mean full solve time, s",
            ylabel="Mean relative L2 error, %",
        )
        ax.set_xscale("log")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "error-time.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plan", type=Path, default=ROOT / "benchmarks/heat3d-pilot.json"
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--worker", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    if args.worker is not None:
        worker(plan, args.worker, args.out)
        return
    # Apply before numpy/scipy imports and preserve the same settings in workers.
    for variable in THREAD_VARIABLES:
        os.environ[variable] = "1"
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
    job_list = list(jobs(plan))
    before = source_state()
    if before["dirty"]:
        raise RuntimeError("Commit the experiment plan and code before measurement")
    args.out.mkdir(parents=True, exist_ok=False)
    from pdelab.heat3d_experiment import csv_bytes, json_bytes

    manifest = {
        "plan": plan,
        "source_before": before,
        "started_utc": utc_now(),
        "platform": platform.platform(),
        "python": sys.version,
        "method_order": [name for name, _, _ in job_list],
        "timing_order": "sequential fresh processes, fixed order, no concurrent workers",
        "completed_jobs": [],
        "complete": False,
    }
    (args.out / "plan.json").write_bytes(json_bytes(plan))
    (args.out / "manifest.json").write_bytes(json_bytes(manifest))
    results = []
    for index, (name, _, _) in enumerate(job_list):
        print(f"{index + 1}/{len(job_list)} {name}", flush=True)
        raw_path = args.out / (name + ".json")
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "benchmarks.heat3d_pilot",
                "--plan",
                str((args.out / "plan.json").resolve()),
                "--out",
                str(raw_path.resolve()),
                "--worker",
                str(index),
            ],
            cwd=ROOT,
            env=os.environ.copy(),
            check=False,
        )
        if completed.returncode:
            manifest["failed_job"] = name
            manifest["worker_exit_code"] = completed.returncode
            (args.out / "manifest.json").write_bytes(json_bytes(manifest))
            raise RuntimeError(
                f"Worker {name} failed; completed raw jobs remain in output"
            )
        results.append((name, json.loads(raw_path.read_bytes())))
        manifest["completed_jobs"].append(name)
        (args.out / "manifest.json").write_bytes(json_bytes(manifest))
    after = source_state()
    if before != after:
        raise RuntimeError(
            "Source changed during measurement; series cannot be accepted"
        )
    manifest.update(source_after=after, finished_utc=utc_now(), complete=True)
    (args.out / "manifest.json").write_bytes(json_bytes(manifest))
    if plan.get("experiment") == "fixed_budget":
        from benchmarks.heat3d_fixed import summarize_fixed

        summarize_fixed(results, args.out)
        return
    targets, series, trials, costs = summarize(results)
    for name, rows in (
        ("targets", targets),
        ("series", series),
        ("trials", trials),
        ("costs", costs),
    ):
        (args.out / (name + ".csv")).write_bytes(csv_bytes(rows))
    plot(series, args.out)


if __name__ == "__main__":
    main()
