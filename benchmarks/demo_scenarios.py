"""Reproduce presentation scenarios with the same study entry points as the UI.

Run sequentially: python -m benchmarks.demo_scenarios
Original measurements are retained; use a new --output for another run.
"""

import argparse
import hashlib
import json
import platform
import subprocess
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

from pdelab.boundary_experiment3d import BoundaryConfig3D
from pdelab.boundary_research import boundary_accuracy_study
from pdelab.experiment import Config
from pdelab.experiment3d import Config3D
from pdelab.heat import make_heat_problem
from pdelab.heat_mc_study import heat_mc_study
from pdelab.research import StudySettings, accuracy_study
from pdelab.research3d import StudySettings3D, accuracy_study3d
from pdelab.study_policy import ACCURACY_METRIC, SELECTION_POLICY

CASES = [
    dict(id="D01", family="2d", problem="sine", n=64, point=(32, 32), coarse_n=16),
    dict(id="D02", family="2d", problem="sine", n=48, point=(24, 24), coarse_n=12),
    dict(
        id="D03", family="2d", problem="polynomial", n=64, point=(16, 21), coarse_n=16
    ),
    dict(id="D04", family="2d", problem="hot_top", n=64, point=(32, 32), coarse_n=16),
    dict(id="D05", family="2d", problem="sine", n=64, point=(1, 24), coarse_n=16),
    dict(id="D06", family="2d", problem="sine", n=16, point=(8, 8), coarse_n=4),
    dict(id="D07", family="3d", problem="sine", n=8, point=(4, 4, 4)),
    dict(id="D08", family="3d", problem="sine", n=16, point=(8, 8, 8)),
    dict(
        id="D09",
        family="mixed3d",
        problem="smooth",
        n=16,
        point=(8, 8, 8),
        kinds=(
            "robin",
            "dirichlet",
            "dirichlet",
            "dirichlet",
            "dirichlet",
            "dirichlet",
        ),
    ),
    dict(
        id="D10",
        family="mixed3d",
        problem="polynomial",
        n=16,
        point=(4, 6, 8),
        kinds=("neumann", "dirichlet", "dirichlet", "dirichlet", "dirichlet", "robin"),
    ),
    dict(
        id="D11", family="heat", problem="decay", n=16, point=(8, 8), dt=0.001, T=0.01
    ),
]


def run_case(case, repeats=10, seed=42):
    common = dict(problem=case["problem"], n=case["n"], point=case["point"], seed=seed)
    if case["family"] == "2d":
        config = Config(**common, coarse_n=case["coarse_n"])
        return asdict(accuracy_study(StudySettings(config, repeats, 4096, 30)))
    if case["family"] == "3d":
        return asdict(
            accuracy_study3d(StudySettings3D(Config3D(**common), repeats, 4096, 30))
        )
    if case["family"] == "mixed3d":
        return boundary_accuracy_study(
            BoundaryConfig3D(**common, kinds=case["kinds"]),
            methods=["lod_serial", "mc", "graded"],
            repeats=repeats,
            budgets=[64, 256, 1024, 4096],
            seconds_per_method=30,
        )
    return heat_mc_study(
        make_heat_problem(case["problem"], case["n"], a2=1.0),
        dt=case["dt"],
        T=case["T"],
        point=case["point"][::-1],
        repeats=repeats,
        max_walks=4096,
        seed=seed,
        seconds_per_method=30,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("benchmarks/results/demo-20261111")
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    files = sorted(Path("pdelab").glob("*.py")) + [Path("app.py"), Path(__file__)]
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    metadata = dict(
        revision=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        platform=platform.platform(),
        python=platform.python_version(),
        source_sha256=hashes,
        cases=CASES,
        repeats=10,
        seed=42,
        maximum_walks=4096,
        seconds_per_phase=30,
        selection_policy=SELECTION_POLICY,
        accuracy_metric=ACCURACY_METRIC,
        scope="All previously winning scenario families, current UI study functions; no global optimum claim",
    )
    manifest = args.output / "protocol.json"
    if manifest.exists():
        assert json.loads(manifest.read_text())["source_sha256"] == hashes
    manifest.write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
    for case in CASES:
        path = args.output / (case["id"] + ".json")
        if path.exists():
            continue
        print("START", case["id"], case["family"], case["problem"], flush=True)
        started = perf_counter()
        data = run_case(case)
        data["case"] = case
        data["wall_seconds"] = perf_counter() - started
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False))
        rows = data["rows"]
        baseline = {
            r["target_pct"]: r for r in rows if r["method"] in ("lod", "lod_serial")
        }
        for r in rows:
            b = baseline[r["target_pct"]]
            if (
                r["method"] not in ("lod", "lod_serial")
                and r["seconds"]
                and b["seconds"]
            ):
                ratio = b["seconds"] / r["seconds"]
                if ratio > 1:
                    print(
                        case["id"],
                        r["method"],
                        r["target_pct"],
                        "gain",
                        round(ratio, 2),
                        "budget",
                        r["budget"],
                        "error",
                        round(r["relative_error_pct"], 3),
                        flush=True,
                    )
        print("DONE", case["id"], round(data["wall_seconds"], 2), "seconds", flush=True)
    assert all(
        hashlib.sha256(p.read_bytes()).hexdigest() == hashes[str(p)] for p in files
    )
    print("Source snapshot unchanged; all studies saved", flush=True)


if __name__ == "__main__":
    main()
