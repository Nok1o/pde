"""Repeat stationary 3D comparisons using the public, fixed-final protocol.

Run sequentially: python -m benchmarks.scenario_comparison --output runs/scenarios
The output directory must be new. No solver implementation is changed here.
"""

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from time import perf_counter

PLAN = Path("benchmarks/plans/scenario-comparison.json")


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))


def source():
    files = sorted(Path("pdelab").glob("*.py")) + [Path(__file__), PLAN]
    return {
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "dirty": bool(
            subprocess.check_output(["git", "status", "--porcelain"], text=True)
        ),
        "sha256": {
            str(p.relative_to(Path.cwd()) if p.is_absolute() else p): hashlib.sha256(
                p.read_bytes()
            ).hexdigest()
            for p in files
        },
    }


def run_case(plan, case, output):
    import numba
    import numpy as np
    import scipy

    from pdelab.problem3d import make_problem3d, select_points3d
    from pdelab.solvers3d import solve3d
    from pdelab.stationary_study3d import study_stationary3d

    before = source()
    if before["dirty"]:
        raise RuntimeError("Publish a clean source snapshot before measuring")
    warm_started = perf_counter()
    small = make_problem3d("sine", 4)
    for method in plan["methods"]:
        solve3d(small, [(2, 2, 2)], method, walks=4, max_iter=2)
    warmup_seconds = perf_counter() - warm_started
    problem = make_problem3d(case["problem"], case["n"])
    points = select_points3d(case["n"], case["mode"], case.get("point"))
    reference = problem.exact[tuple(points.T)]
    started = perf_counter()
    result = study_stationary3d(
        problem,
        points,
        reference,
        plan["methods"],
        walk_budgets=case.get("walk_budgets", plan["walk_budgets"]),
        tolerances=plan["tolerances"],
        repeats=plan["repeats"],
        seed=plan["seed"],
        batch_size=plan["batch_size"],
        coarse_n=case["coarse_n"],
        max_iter=plan["max_iter"],
        seconds_per_phase=plan["seconds_per_phase"],
    )
    study_seconds = perf_counter() - started
    after = source()
    if before != after:
        raise RuntimeError("Source changed during measurement")
    save(
        output / (case["id"] + ".json"),
        {
            "case": case,
            "source_before": before,
            "source_after": after,
            "environment": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "numpy": np.__version__,
                "scipy": scipy.__version__,
                "numba": numba.__version__,
            },
            "points_zyx": points.tolist(),
            "reference": reference.tolist(),
            "reference_kind": "analytic",
            "warmup_seconds": warmup_seconds,
            "study_seconds": study_seconds,
            **result,
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", type=int)
    args = parser.parse_args()
    plan = json.loads(PLAN.read_text())
    if args.case is not None:
        run_case(plan, plan["cases"][args.case], args.output)
        return
    args.output.mkdir(parents=True, exist_ok=False)
    before = source()
    if before["dirty"]:
        raise RuntimeError("Commit the plan and runner before measurement")
    save(args.output / "plan.json", plan)
    manifest = {"source_before": before, "complete": False, "completed": []}
    save(args.output / "manifest.json", manifest)
    env = dict(os.environ)
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMBA_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        env[name] = "1"
    for index, case in enumerate(plan["cases"]):
        print("START", case["id"], flush=True)
        with (args.output / (case["id"] + ".log")).open("w") as log:
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "benchmarks.scenario_comparison",
                    "--output",
                    str(args.output),
                    "--case",
                    str(index),
                ],
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        manifest["completed"].append(case["id"])
        save(args.output / "manifest.json", manifest)
        print("DONE", case["id"], flush=True)
    manifest["source_after"] = source()
    manifest["complete"] = manifest["source_after"] == before
    save(args.output / "manifest.json", manifest)
    if not manifest["complete"]:
        raise RuntimeError("Source changed during the study")


if __name__ == "__main__":
    main()
