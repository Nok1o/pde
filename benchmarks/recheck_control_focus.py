"""Run after recheck_audit, serially, to avoid competing benchmark processes."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

from pdelab.experiment import Config
from pdelab.problem import make_problem, reference_solution, select_points
from pdelab.reference import choose_reference
from pdelab.research import (
    LOD_SEARCH_TOLERANCES,
    StudyResult,
    StudySettings,
    _measure_budget,
    _walk_budgets,
)
from pdelab.solvers import solve, warmup
from pdelab.study_policy import search_budgets

OUT = Path("benchmarks/results/audit-recheck-20261111")


def main(only=None):
    OUT.mkdir(parents=True, exist_ok=True)
    files = [
        "pdelab/research.py",
        "pdelab/study_policy.py",
        "pdelab/solvers.py",
        "pdelab/control.py",
    ]
    snapshot = {name: Path(name).read_bytes() for name in files}
    code_dir = OUT / "focus-code"
    code_dir.mkdir(exist_ok=True)
    metadata = dict(
        revision=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        source_sha256={
            name: hashlib.sha256(content).hexdigest()
            for name, content in snapshot.items()
        },
        repeats=5,
        seed=42,
        max_walks=16384,
        seconds_per_phase=600,
    )
    prior = OUT / "focus-protocol.json"
    if prior.exists():
        old = json.loads(prior.read_text())
        assert (
            old["source_sha256"] == metadata["source_sha256"]
        ), "Use a new directory for different source snapshots"
    for name, content in snapshot.items():
        (code_dir / Path(name).name).write_bytes(content)
    warmup()
    saved = OUT / "control-focus.json"
    cases = json.loads(saved.read_text()) if saved.exists() else []
    for n in ((only[0],) if only else (16, 24, 48, 64)):
        problem = make_problem("sine", n)
        points = select_points(n)
        grid = reference_solution(problem)
        reference = choose_reference(problem, points, grid)
        target = reference.values[tuple(points.T)]
        coarse_sizes = {4, max(4, n // 4), n}
        if n == 64:
            coarse_sizes.add(6)
        for coarse in ((only[1],) if only else sorted(coarse_sizes)):
            if any(case["n"] == n and case["coarse_n"] == coarse for case in cases):
                continue
            trials = []
            for repeat in range(50):
                rng = np.random.default_rng([202611114, n, coarse, repeat])
                for method in rng.permutation(["lod_serial", "mc_control"]):
                    seed = int(
                        np.random.SeedSequence([202611114, n, repeat]).generate_state(
                            1
                        )[0]
                    )
                    result = solve(
                        problem,
                        points,
                        method,
                        walks=64,
                        seed=seed,
                        tol=0.01,
                        record=False,
                        coarse_n=coarse,
                    )
                    trials.append(
                        dict(
                            method=str(method),
                            repeat=repeat,
                            seed=seed,
                            seconds=result.elapsed,
                            error_pct=float(
                                100
                                * np.mean(
                                    np.abs(result.values[tuple(points.T)] - target)
                                )
                                / np.mean(np.abs(target))
                            ),
                            preparation_seconds=result.extras.get(
                                "preparation_seconds", 0
                            ),
                            path_seconds=result.extras.get("path_seconds", 0),
                        )
                    )
            cfg = Config(problem="sine", n=n, coarse_n=coarse, walks=64, seed=42)
            settings = StudySettings(cfg, 5, 16384, 600)
            study = StudyResult(
                settings, reference.kind, reference.label, float(target[0])
            )
            rows = []
            for method in ("lod_serial", "mc_control"):
                budgets = (
                    LOD_SEARCH_TOLERANCES
                    if method == "lod_serial"
                    else _walk_budgets(settings.max_walks, cfg.batches)
                )

                def measure(budget, index, phase, deadline):
                    return _measure_budget(
                        study,
                        problem,
                        points,
                        method,
                        budget,
                        index,
                        phase,
                        deadline,
                        None,
                        grid[tuple(points.T)],
                    )

                rows.extend(
                    search_budgets(
                        method, budgets, measure, settings.seconds_per_method
                    )
                )
            cases.append(
                dict(
                    n=n,
                    coarse_n=coarse,
                    fixed_N64=trials,
                    ui_rows=rows,
                    ui_trials=study.trials,
                )
            )
            (OUT / "control-focus.json").write_text(
                json.dumps(cases, ensure_ascii=False, indent=2)
            )
            print(
                n,
                coarse,
                [
                    (r["method"], r["budget"], r["status"], r["seconds"])
                    for r in rows
                    if r["target_pct"] == 1
                ],
                flush=True,
            )
    metadata["source_unchanged_during_run"] = all(
        Path(name).read_bytes() == content for name, content in snapshot.items()
    )
    metadata["selection_policy"] = cases[0]["ui_rows"][0]["selection_policy"]
    (OUT / "focus-protocol.json").write_text(json.dumps(metadata, indent=2))
    assert metadata[
        "source_unchanged_during_run"
    ], "Source files changed while measuring"
    print("FOCUS COMPLETE", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--only", type=int, nargs=2, metavar=("N_GRID", "N_CONTROL"))
    args = parser.parse_args()
    OUT = args.output
    main(only=args.only)
