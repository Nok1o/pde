"""python -m pdelab.nonlinear_cli run|study|targets."""

import argparse
import platform
import sys
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

from .nonlinear import NonlinearSettings, make_nonlinear_problem
from .nonlinear_io import csv_bytes, export_comparison, json_bytes, study_figure
from .nonlinear_study import (
    compare_nonlinear,
    nonlinear_accuracy_study,
    nonlinear_mc_study,
    nonlinear_targets,
)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Нелинейная теплопроводность: Picard, Newton и MC"
    )
    parser.add_argument("command", choices=["run", "study", "targets"])
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument(
        "--gamma",
        "--alpha",
        dest="alpha",
        help="γ в k(u)=1+γu²",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--tau",
        "--dt",
        dest="dt",
        help="Шаг физического времени τ",
        type=float,
        default=0.01,
    )
    parser.add_argument("--T", type=float, default=0.1)
    parser.add_argument("--point", nargs=2, type=int, metavar=("X", "Z"))
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--walks", type=int, default=128)
    parser.add_argument("--max-walks", type=int, default=2048)
    parser.add_argument(
        "--schedule", choices=["constant", "linear", "geometric"], default="geometric"
    )
    parser.add_argument(
        "--xi",
        "--relaxation",
        dest="relaxation",
        help="Релаксация ξ",
        type=float,
        default=0.7,
    )
    parser.add_argument("--tol", type=float, default=1e-7)
    parser.add_argument("--linear-tol", type=float, default=1e-10)
    parser.add_argument("--mc-tol", type=float, default=0.001)
    parser.add_argument("--max-outer", type=int, default=12)
    parser.add_argument("--max-inner", type=int, default=20000)
    parser.add_argument("--max-steps", type=int, default=100000)
    parser.add_argument("--max-total-walks", type=int, default=20_000_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--budgets", nargs="+", type=int, default=[32, 128, 512])
    parser.add_argument("--out", type=Path, default=Path("runs/nonlinear"))
    argv = sys.argv[1:] if argv is None else list(argv)
    args = parser.parse_args(argv)
    if args.command == "study":
        allowed = {"--repeats", "--seed", "--out"}
        rejected = [
            token.split("=")[0]
            for token in argv
            if token.startswith("--") and token.split("=")[0] not in allowed
        ]
        if rejected:
            parser.error(
                "study использует фиксированные случаи; неприменимые аргументы: "
                + ", ".join(rejected)
            )
    settings = NonlinearSettings(
        tolerance=args.mc_tol,
        change_tolerance=args.mc_tol,
        relaxation=args.relaxation,
        walks=args.walks,
        max_walks=args.max_walks,
        schedule=args.schedule,
        max_outer=args.max_outer,
        max_inner=args.max_inner,
        max_steps=args.max_steps,
        seed=args.seed,
        max_total_walks=args.max_total_walks,
    )
    deterministic = NonlinearSettings(
        tolerance=args.tol,
        change_tolerance=args.tol,
        linear_tolerance=args.linear_tol,
        relaxation=args.relaxation,
        max_outer=args.max_outer,
        max_inner=args.max_inner,
    )
    incomplete = False
    try:
        settings.validate()
        deterministic.validate()
        problem = make_nonlinear_problem(args.n, args.alpha)
        args.out.mkdir(parents=True, exist_ok=True)
        if args.command == "run":
            comparison = compare_nonlinear(
                problem,
                args.dt,
                args.T,
                args.point,
                args.repeats,
                deterministic,
                settings,
            )
            export_comparison(comparison, args.out)
            print(csv_bytes(comparison.metrics).decode("utf-8-sig"))
            incomplete = not all(row["completed"] for row in comparison.metrics)
        elif args.command == "targets":
            rows, evidence = nonlinear_targets(
                problem,
                args.dt,
                args.T,
                args.point,
                args.repeats,
                args.budgets,
                args.seed,
                settings=settings,
                deterministic=deterministic,
            )
            (args.out / "targets.csv").write_bytes(csv_bytes(rows))
            (args.out / "target-evidence.json").write_bytes(json_bytes(evidence))
            print(csv_bytes(rows).decode("utf-8-sig"))
        else:
            accuracy = nonlinear_accuracy_study()
            rows, runs = nonlinear_mc_study(args.repeats, args.seed)
            (args.out / "accuracy.csv").write_bytes(csv_bytes(accuracy))
            (args.out / "mc-study.csv").write_bytes(csv_bytes(rows))
            (args.out / "chains.json").write_bytes(json_bytes(runs))
            study_figure(rows).write_html(
                args.out / "study.html", include_plotlyjs=True
            )
            print(
                f"Исследование: {len(accuracy)} проверок ошибок, {len(rows)} строк сравнения."
            )
        study_cases = (
            [
                {
                    key: row[key]
                    for key in (
                        "case",
                        "method",
                        "n",
                        "alpha",
                        "dt",
                        "T",
                        "tolerance",
                        "change_tolerance",
                        "linear_tolerance",
                        "relaxation",
                        "max_outer",
                        "max_inner",
                        "initial_walks",
                        "max_walks",
                        "schedule",
                        "max_steps",
                        "max_total_walks",
                        "seed",
                    )
                }
                for row in rows
            ]
            if args.command == "study"
            else None
        )
        (args.out / "settings.json").write_bytes(
            json_bytes(
                dict(
                    command=args.command,
                    **(
                        dict(cases=study_cases, accuracy_cases=accuracy)
                        if args.command == "study"
                        else dict(
                            n=args.n,
                            alpha=args.alpha,
                            dt=args.dt,
                            T=args.T,
                            point=args.point,
                            mc=asdict(settings),
                            picard=asdict(deterministic),
                            budgets=args.budgets,
                        )
                    ),
                    repeats=args.repeats,
                    seed=args.seed,
                    python=platform.python_version(),
                    platform=platform.platform(),
                    versions={
                        name: version(name) for name in ("numpy", "scipy", "numba")
                    },
                )
            )
        )
    except (ValueError, RuntimeError) as exc:
        parser.exit(2, f"Ошибка: {exc}\n")
    if incomplete:
        parser.exit(
            2, "Нелинейная цепочка не сошлась; причины и неполная история сохранены.\n"
        )
    print(f"Сохранено: {args.out.resolve()}")


if __name__ == "__main__":
    main()
