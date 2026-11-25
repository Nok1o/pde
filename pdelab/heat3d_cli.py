"""python -m pdelab.heat3d_cli run|study: per-face conditions of kind 1 or 3."""

import argparse
from pathlib import Path

from .heat3d_boundary_options import add_boundary_arguments, boundary_configuration
from .heat3d_experiment import (
    APPLICABILITY,
    METHOD_NAMES,
    HeatConfig3D,
    compare_heat3d,
    csv_bytes,
    export_result,
    study_heat3d,
)


def main(argv=None):
    from .heat3d_solver import HeatSettings3D

    parser = argparse.ArgumentParser(description=APPLICABILITY)
    parser.add_argument("command", choices=("run", "study"))
    parser.add_argument("--case", choices=("sine", "polynomial"), default="sine")
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=0.5)
    add_boundary_arguments(parser)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--T", type=float, default=0.02)
    parser.add_argument(
        "--methods", nargs="+", choices=tuple(METHOD_NAMES), default=["lod"]
    )
    parser.add_argument(
        "--query", choices=("point", "patch", "slice", "field"), default="point"
    )
    parser.add_argument("--point", nargs=3, type=int, metavar=("X", "Y", "Z"))
    parser.add_argument("--plane", choices=("xy", "xz", "yz"))
    parser.add_argument("--index", type=int)
    parser.add_argument(
        "--newton",
        action="store_true",
        help="Независимая сеточная диагностика для run, n ≤ 16",
    )
    parser.add_argument("--tol", type=float, default=1e-7)
    parser.add_argument("--change-tol", type=float, default=1e-7)
    parser.add_argument("--linear-tol", type=float, default=1e-10)
    parser.add_argument("--relaxation", type=float, default=1.0)
    parser.add_argument("--max-outer", type=int, default=100)
    parser.add_argument("--max-inner", type=int, default=20000)
    parser.add_argument("--max-seconds", type=float, default=60.0)
    parser.add_argument("--max-workspace-bytes", type=int, default=256 * 1024**2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--budgets", nargs="+", type=float, default=[1e-3, 1e-5, 1e-7, 1e-9]
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=0.8,
        help="Фиксированная доля цели для подбора: 0 < margin < 1",
    )
    parser.add_argument("--out", type=Path, default=Path("runs/heat3d"))
    args = parser.parse_args(argv)
    if args.point is not None and args.query != "point":
        parser.error("--point применяется только к --query point")
    if (args.plane is not None or args.index is not None) and args.query != "slice":
        parser.error("--plane и --index применяются только к --query slice")
    if args.newton and args.command != "run":
        parser.error(
            "--newton применяется только к run; исследование использует аналитический эталон"
        )
    config = HeatConfig3D(
        case=args.case,
        n=args.n,
        alpha=args.alpha,
        dt=args.dt,
        T=args.T,
        mode=args.query,
        point=None if args.point is None else tuple(args.point),
        plane=args.plane or "xy",
        index=args.index,
        newton=args.newton,
        methods=tuple(args.methods),
        **boundary_configuration(args),
    )
    settings = HeatSettings3D(
        tolerance=args.tol,
        change_tolerance=args.change_tol,
        linear_tolerance=args.linear_tol,
        relaxation=args.relaxation,
        max_outer=args.max_outer,
        max_inner=args.max_inner,
        max_seconds=args.max_seconds,
        max_workspace_bytes=args.max_workspace_bytes,
    )
    try:
        if args.command == "run":
            result = compare_heat3d(config, settings)
            success = all(row["status"] == "completed" for row in result["rows"])
            if args.newton:
                success = success and result["grid_diagnostic"]["status"] == "completed"
        else:
            result = study_heat3d(
                config,
                settings,
                repeats=args.repeats,
                seed=args.seed,
                budgets=args.budgets,
                calibration_fraction=args.margin,
            )
            success = all(row["status"] == "confirmed" for row in result["rows"])
        export_result(result, args.out)
    except (ValueError, RuntimeError) as exc:
        parser.exit(2, f"Ошибка: {exc}\n")
    print(csv_bytes(result["rows"]).decode("utf-8-sig"))
    print(f"Сохранено: {args.out.resolve()}")
    if not success:
        parser.exit(
            2,
            "Расчёт или цели исследования не завершены успешно; все причины и фактические затраты сохранены.\n",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
