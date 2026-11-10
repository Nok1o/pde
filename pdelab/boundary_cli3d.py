"""python -m pdelab.boundary_cli3d run --x0 neumann --z1 robin --n 16"""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from .boundary import KINDS
from .boundary3d import FACES, PRESETS_BOUNDARY3D
from .boundary_experiment3d import (
    BoundaryConfig3D,
    benchmark_boundary3d,
    export_boundary3d,
    run_boundary_experiment3d,
)
from .experiment3d import rows_csv
from .solvers3d import METHODS_3D


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="3D: Дирихле, Нейман и Робин на шести гранях"
    )
    parser.add_argument("command", choices=["run", "benchmark"])
    parser.add_argument(
        "--problem", choices=[*PRESETS_BOUNDARY3D, "custom"], default="polynomial"
    )
    parser.add_argument("--n", type=int, default=16)
    for face in FACES:
        parser.add_argument(
            f"--{face}", choices=KINDS, default="robin" if face == "x0" else "dirichlet"
        )
    parser.add_argument(
        "--kappas", nargs=6, type=float, default=[1.0] * 6, metavar=tuple(FACES)
    )
    parser.add_argument(
        "--data", nargs=6, type=float, default=[0.0] * 6, metavar=tuple(FACES)
    )
    parser.add_argument("--source", type=float, default=0.0)
    parser.add_argument(
        "--mode", choices=["point", "patch", "slice", "field"], default="point"
    )
    parser.add_argument("--point", nargs=3, type=int, metavar=("X", "Y", "Z"))
    parser.add_argument("--plane", choices=["xy", "xz", "yz"], default="xy")
    parser.add_argument("--index", type=int)
    parser.add_argument("--walks", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tol", type=float, default=1e-8)
    parser.add_argument("--max-iter", type=int, default=20000)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--methods", choices=METHODS_3D, nargs="+")
    parser.add_argument("--budgets", nargs="+", type=int, default=[128, 512, 2048])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--out", type=Path, default=Path("runs/boundary3d"))
    args = parser.parse_args(argv)
    config = BoundaryConfig3D(
        **{
            name: getattr(args, name)
            for name in BoundaryConfig3D.__dataclass_fields__
            if name != "kinds"
        },
        kinds=tuple(getattr(args, face) for face in FACES),
    )
    try:
        if args.command == "run":
            experiment = run_boundary_experiment3d(config, args.methods)
            export_boundary3d(experiment, args.out)
            print(rows_csv(experiment.metrics))
        else:
            rows = benchmark_boundary3d(
                config, args.budgets, args.repeats, methods=args.methods
            )
            args.out.mkdir(parents=True, exist_ok=True)
            (args.out / "benchmark.csv").write_bytes(rows_csv(rows).encode("utf-8-sig"))
            settings = {
                "config": asdict(config),
                "budgets": args.budgets,
                "repeats": args.repeats,
                "tolerances": [1e-3, 1e-6, 1e-9],
                "methods": list(dict.fromkeys(row["method"] for row in rows)),
            }
            (args.out / "settings.json").write_text(
                json.dumps(settings, ensure_ascii=False, indent=2, allow_nan=False),
                encoding="utf-8",
            )
            print(rows_csv(rows))
    except (ValueError, RuntimeError) as exc:
        parser.exit(2, f"Ошибка: {exc}\n")
    print(f"Сохранено: {args.out.resolve()}")


if __name__ == "__main__":
    main()
