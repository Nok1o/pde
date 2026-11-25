"""Diagnostic frozen-system MC for PDE-28; does not advance a Picard/time chain."""

import argparse
import hashlib
import platform
import subprocess
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.sparse.linalg import spsolve

from .heat3d_experiment import csv_bytes, json_bytes
from .heat3d_mc import solve_frozen_mc3d, warmup_mc3d
from .heat3d_operator import freeze_heat3d
from .heat3d_problem import make_heat_problem3d
from .problem3d import select_points3d


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--case", choices=("sine", "polynomial"), default="sine")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--point", nargs=3, type=int, metavar=("X", "Y", "Z"))
    parser.add_argument("--method", choices=("mc", "mc_grouped"), default="mc")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--walks", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=1000000)
    parser.add_argument("--max-walks", type=int, default=1000000)
    parser.add_argument("--max-transitions", type=int, default=10000000)
    parser.add_argument("--max-workspace-bytes", type=int, default=256 * 1024**2)
    parser.add_argument(
        "--direct", action="store_true", help="Отдельная сеточная диагностика, n<=16"
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    batch_size = args.batch_size if args.method == "mc_grouped" else None
    if args.out.exists() and any(args.out.iterdir()):
        parser.error(
            "Папка результата должна быть пустой: предыдущие попытки не перезаписываются"
        )
    if args.direct and args.n > 16:
        parser.error("Прямая диагностика ограничена n<=16")
    try:
        warm_start = perf_counter()
        warmup_mc3d(batch_size=batch_size)
        warm_seconds = perf_counter() - warm_start
        start = perf_counter()
        problem = make_heat_problem3d(args.case, args.n, args.alpha)
        points = select_points3d(args.n, point=args.point)
        previous = problem.initial_values()
        operator = freeze_heat3d(problem, previous, previous, args.dt, args.dt)
        construction_seconds = perf_counter() - start
        result = solve_frozen_mc3d(
            operator,
            points,
            walks=args.walks,
            seed=args.seed,
            max_steps=args.max_steps,
            max_walks=args.max_walks,
            max_transitions=args.max_transitions,
            max_workspace_bytes=args.max_workspace_bytes,
            batch_size=batch_size,
        )
        full_seconds = perf_counter() - start
        reference, reference_seconds = None, 0.0
        if args.direct:
            reference_start = perf_counter()
            matrix, rhs = operator.assemble()
            reference = spsolve(matrix, rhs).reshape(operator.shape)[tuple(points.T)]
            reference_seconds = perf_counter() - reference_start
    except ValueError as exc:
        parser.error(str(exc))
    root = Path(__file__).resolve().parents[1]
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=root, text=True
            ).strip()
        )
    except (OSError, subprocess.SubprocessError):
        commit, dirty = None, None
    metadata = dict(
        schema_version="frozen_heat_mc3d_v1",
        scenario="diagnostic_frozen_first_layer",
        configuration={
            **{k: v for k, v in vars(args).items() if k != "out"},
            "batch_size": batch_size,
        },
        commit=commit,
        dirty=dirty,
        python=platform.python_version(),
        numpy=np.__version__,
        source_sha256={
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (
                Path(__file__),
                root / "pdelab/heat3d_mc.py",
                root / "pdelab/heat3d_grouped_mc.py",
                root / "pdelab/heat3d_operator.py",
                root / "pdelab/heat3d_problem.py",
            )
        },
        coordinate_order="xyz",
        array_order="zyx",
        physical_layers_accepted=0,
        timing=dict(
            full_seconds=full_seconds,
            operator_construction_seconds=construction_seconds,
            warmup_seconds=warm_seconds,
            reference_seconds=reference_seconds,
        ),
        note="One frozen A*w=b, not converged Picard or a full run to T. completed means fixed sample budget finished; stderr is diagnostic, no accuracy guarantee.",
    )
    records = []
    for index, (z, y, x) in enumerate(points):
        records.append(
            dict(
                x=int(x),
                y=int(y),
                z=int(z),
                status=result.status,
                value=None if result.values is None else result.values[index],
                stderr=None if result.stderr is None else result.stderr[index],
                direct_reference=None if reference is None else reference[index],
            )
        )
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "result.json").write_bytes(
        json_bytes(dict(metadata=metadata, result=asdict(result), rows=records))
    )
    (args.out / "points.csv").write_bytes(csv_bytes(records))
    print(
        f"{result.status}: {result.reason}; walks={result.walks}, transitions={result.transitions}"
    )
    if result.status != "completed":
        parser.exit(
            2, "Лимит/отказ: частичная оценка сохранена только как диагностика.\n"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
