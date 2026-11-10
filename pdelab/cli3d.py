"""CLI для куба: python -m pdelab.cli3d run|study."""

import argparse
from pathlib import Path

from .experiment3d import Config3D, export3d, rows_csv, run_experiment3d
from .problem3d import PRESETS_3D
from .research3d import StudySettings3D, accuracy_study3d, export_study3d, timing_rows3d
from .solvers3d import METHODS_3D


def main(argv=None):
    parser = argparse.ArgumentParser(description="ЛОМ и Монте-Карло в единичном кубе")
    parser.add_argument("command", choices=["run", "study"])
    parser.add_argument("--problem", choices=PRESETS_3D, default="sine")
    parser.add_argument("--n", type=int, default=16)
    parser.add_argument("--point", type=int, nargs=3, metavar=("X", "Y", "Z"))
    parser.add_argument(
        "--mode", choices=["point", "patch", "slice", "field"], default="point"
    )
    parser.add_argument("--plane", choices=["xy", "xz", "yz"], default="xy")
    parser.add_argument("--index", type=int)
    parser.add_argument("--walks", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tol", type=float, default=1e-7)
    parser.add_argument("--max-iter", type=int, default=20000)
    parser.add_argument(
        "--methods", choices=METHODS_3D, nargs="+", default=list(METHODS_3D)
    )
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--max-walks", type=int, default=32768)
    parser.add_argument("--seconds-per-method", type=float, default=60)
    parser.add_argument("--out", type=Path, default=Path("runs") / "3d")
    args = parser.parse_args(argv)
    config = Config3D(
        **{name: getattr(args, name) for name in Config3D.__dataclass_fields__}
    )
    try:
        if args.command == "run":
            experiment = run_experiment3d(config, args.methods)
            export3d(experiment, args.out)
            print(rows_csv(experiment.metrics))
        else:
            settings = StudySettings3D(
                config,
                args.repeats,
                args.max_walks,
                args.seconds_per_method,
                methods=tuple(args.methods),
            )
            study = accuracy_study3d(
                settings,
                lambda fraction, text: print(f"{fraction:.0%} · {text}", flush=True),
            )
            export_study3d(study, args.out)
            print(rows_csv(timing_rows3d(study)))
    except (ValueError, RuntimeError) as exc:
        parser.exit(2, f"Ошибка: {exc}\n")
    print(f"Сохранено: {args.out.resolve()}")


if __name__ == "__main__":
    main()
