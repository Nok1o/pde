"""Recreate bundled figures and raw measurements: python generate_examples.py.

Run by itself, not alongside other benchmarks. Machine load affects timings.
"""

import json
import os
import platform
from dataclasses import asdict
from pathlib import Path

from pdelab.experiment import (
    LOD_TOLERANCES,
    METRICS_POLICY,
    Config,
    benchmark,
    run_experiment,
)
from pdelab.reference import REFERENCE_POLICY
from pdelab.solvers import LOD_STOPPING_RULE
from pdelab.views import benchmark_plot, export


def main():
    root = Path(__file__).parent / "examples"
    root.mkdir(exist_ok=True)
    for name, cfg in [
        ("center", Config()),
        ("patch", Config(mode="patch", walks=1024)),
        ("field", Config(problem="polynomial", n=16, mode="field", walks=512)),
    ]:
        print(f"Example: {name}", flush=True)
        exp = run_experiment(cfg)
        export(exp, root / name, with_gifs=name != "patch")
    for mode in ["point", "patch", "field"]:
        print(f"Benchmark: {mode}", flush=True)
        cfg = Config(n=24, mode=mode)
        out = root / f"benchmark-{mode}"
        rows = benchmark(cfg, budgets=[128, 512, 2048], repeats=5)
        benchmark_plot(rows, out)
        meta = {
            "config": asdict(cfg),
            "budgets": [128, 512, 2048],
            "repeats": 5,
            "tolerances": list(LOD_TOLERANCES),
            "lod_stopping_rule": LOD_STOPPING_RULE,
            "metrics_policy": METRICS_POLICY,
            "reference_policy": REFERENCE_POLICY,
            "reference_kind": rows[0]["reference_kind"],
            "reference_label": rows[0]["reference_label"],
            "python": platform.python_version(),
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
            "timing": "Solver wall time after JIT warmup; reference, plots excluded; all layers preparation included.",
        }
        (out / "benchmark-config.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    print("Examples ready:", root, flush=True)


if __name__ == "__main__":
    main()
