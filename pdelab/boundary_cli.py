"""python -m pdelab.boundary_cli run|benchmark --left neumann --n 24"""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from .boundary import KINDS, PRESETS_BOUNDARY, SIDES
from .boundary_experiment import (
    BoundaryConfig,
    benchmark_boundary,
    export_boundary,
    run_boundary_experiment,
)
from .boundary_solvers import METHODS_BOUNDARY
from .experiment3d import rows_csv


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="2D: условия 1, 2 и 3 рода по сторонам"
    )
    parser.add_argument("command", choices=["run", "benchmark", "study"])
    parser.add_argument(
        "--problem", choices=[*PRESETS_BOUNDARY, "custom"], default="polynomial"
    )
    parser.add_argument("--n", type=int, default=24)
    for side in SIDES:
        parser.add_argument(
            f"--{side}",
            choices=KINDS,
            default="robin" if side == "left" else "dirichlet",
        )
    parser.add_argument(
        "--betas",
        "--kappas",
        dest="kappas",
        help="Коэффициенты β нормализованного условия ∂ₙu+βu=φ",
        nargs=4,
        type=float,
        default=[1.0] * 4,
        metavar=("LEFT", "RIGHT", "BOTTOM", "TOP"),
    )
    parser.add_argument(
        "--data",
        nargs=4,
        type=float,
        default=[0.0] * 4,
        metavar=("LEFT", "RIGHT", "BOTTOM", "TOP"),
    )
    parser.add_argument("--source", type=float, default=0.0)
    parser.add_argument("--mode", choices=["point", "patch", "field"], default="point")
    parser.add_argument("--point", nargs=2, type=int, metavar=("X", "Z"))
    parser.add_argument("--walks", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tol", type=float, default=1e-8)
    parser.add_argument("--max-iter", type=int, default=20000)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--methods", choices=METHODS_BOUNDARY, nargs="+")
    parser.add_argument("--budgets", nargs="+", type=int, default=[128, 512, 2048])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--out", type=Path, default=Path("runs/boundary"))
    args = parser.parse_args(argv)
    config = BoundaryConfig(
        **{
            name: getattr(args, name)
            for name in BoundaryConfig.__dataclass_fields__
            if name != "kinds"
        },
        kinds=tuple(getattr(args, side) for side in SIDES),
    )
    try:
        if args.command == "run":
            experiment = run_boundary_experiment(config, args.methods)
            export_boundary(experiment, args.out)
            print(rows_csv(experiment.metrics))
        elif args.command == "study":
            from .boundary_research import boundary_accuracy_study

            study = boundary_accuracy_study(
                config, args.methods, args.repeats, args.budgets
            )
            args.out.mkdir(parents=True, exist_ok=True)
            (args.out / "timings.csv").write_bytes(
                rows_csv(study["rows"]).encode("utf-8-sig")
            )
            (args.out / "study.json").write_text(
                json.dumps(study, ensure_ascii=False, allow_nan=False, indent=2),
                encoding="utf-8",
            )
            print(rows_csv(study["rows"]))
        else:
            rows = benchmark_boundary(
                config, args.budgets, args.repeats, methods=args.methods
            )
            args.out.mkdir(parents=True, exist_ok=True)
            (args.out / "benchmark.csv").write_text(
                rows_csv(rows), encoding="utf-8-sig"
            )
            (args.out / "settings.json").write_text(
                json.dumps(
                    {
                        "config": asdict(config),
                        "budgets": args.budgets,
                        "repeats": args.repeats,
                        "tolerances": [1e-3, 1e-6, 1e-9],
                        "methods": list(dict.fromkeys(row["method"] for row in rows)),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            print(rows_csv(rows))
    except (ValueError, RuntimeError) as exc:
        parser.exit(2, f"Ошибка: {exc}\n")
    print(f"Сохранено: {args.out.resolve()}")


if __name__ == "__main__":
    main()
