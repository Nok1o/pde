import argparse
import json
import os
import platform
from dataclasses import asdict
from pathlib import Path

from .experiment import (
    LOD_TOLERANCES,
    METRICS_POLICY,
    Config,
    benchmark,
    run_experiment,
)
from .problem import PRESETS
from .reference import REFERENCE_POLICY
from .solvers import LOD_STOPPING_RULE
from .views import benchmark_plot, export


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="PDE Lab: Δu=f, сеточный метод и варианты Монте-Карло"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for cmd in ["run", "benchmark", "study"]:
        command_parser = subparsers.add_parser(cmd)
        command_parser.add_argument(
            "--problem", choices=list(PRESETS), default="hot_top"
        )
        command_parser.add_argument(
            "--n", type=int, default=24, help="Число интервалов; узлов n+1"
        )
        command_parser.add_argument(
            "--mode", choices=["point", "patch", "field"], default="point"
        )
        command_parser.add_argument(
            "--point",
            type=int,
            nargs=2,
            metavar=("X", "Y"),
            help="Индексы выбранного узла (по умолчанию центр)",
        )
        command_parser.add_argument(
            "--walks",
            type=int,
            default=2048,
            help="Новых путей на каждую стартовую точку",
        )
        command_parser.add_argument(
            "--batches",
            type=int,
            default=8,
            help="Независимых полных реализаций метода слоёв",
        )
        command_parser.add_argument("--coarse-n", type=int, default=None)
        command_parser.add_argument("--seed", type=int, default=42)
        command_parser.add_argument(
            "--tol",
            type=float,
            default=1e-7,
            help="Относительная невязка ||b-Au||∞/||b||∞; 0 < tol < 1",
        )
        command_parser.add_argument("--max-iter", type=int, default=20000)
        for side, default in [
            ("top", 100.0),
            ("bottom", 0.0),
            ("left", 0.0),
            ("right", 0.0),
            ("source", 0.0),
        ]:
            command_parser.add_argument("--" + side, type=float, default=default)
        command_parser.add_argument("--out", type=Path, default=Path("runs") / cmd)
    subparsers.choices["run"].add_argument("--no-gifs", action="store_true")
    subparsers.choices["benchmark"].add_argument(
        "--budgets", type=int, nargs="+", default=[256, 1024, 4096]
    )
    subparsers.choices["benchmark"].add_argument("--repeats", type=int, default=3)
    subparsers.choices["benchmark"].add_argument(
        "--tolerances",
        type=float,
        nargs="+",
        default=list(LOD_TOLERANCES),
        help="Относительные допуски ЛОМ",
    )
    study_parser = subparsers.choices["study"]
    study_parser.add_argument("--repeats", type=int, default=5)
    study_parser.add_argument("--max-walks", type=int, default=4096)
    study_parser.add_argument("--seconds-per-method", type=float, default=60)
    heat_parser = subparsers.add_parser(
        "heat", help="Линейная теплопроводность в физическом времени"
    )
    heat_parser.add_argument(
        "--problem", choices=["decay", "manufactured"], default="decay"
    )
    heat_parser.add_argument("--n", type=int, default=24)
    heat_parser.add_argument("--a2", type=float, default=1.0)
    heat_parser.add_argument("--dt", type=float, default=0.01)
    heat_parser.add_argument("--T", type=float, default=0.2)
    heat_parser.add_argument("--tol", type=float, default=1e-10)
    heat_parser.add_argument("--max-iter", type=int, default=20000)
    heat_parser.add_argument("--method", choices=["lod", "direct"], default="lod")
    heat_parser.add_argument("--out", type=Path, default=Path("runs/heat"))
    heat_study_parser = subparsers.add_parser(
        "heat-study", help="Порядки по h, dt и ошибка линейного решения"
    )
    heat_study_parser.add_argument(
        "--out", type=Path, default=Path("runs/heat-study.csv")
    )
    for name in ("heat-mc", "heat-mc-study"):
        command_parser = subparsers.add_parser(name)
        command_parser.add_argument(
            "--problem", choices=["decay", "manufactured"], default="decay"
        )
        command_parser.add_argument("--n", type=int, default=24)
        command_parser.add_argument("--a2", type=float, default=1.0)
        command_parser.add_argument("--dt", type=float, default=0.01)
        command_parser.add_argument("--T", type=float, default=0.04)
        command_parser.add_argument("--point", type=int, nargs=2, metavar=("X", "Y"))
        command_parser.add_argument("--seed", type=int, default=42)
        command_parser.add_argument("--max-transitions", type=int, default=1000000)
        command_parser.add_argument("--out", type=Path, default=Path("runs") / name)
    subparsers.choices["heat-mc"].add_argument("--walks", type=int, default=4096)
    subparsers.choices["heat-mc-study"].add_argument("--repeats", type=int, default=5)
    subparsers.choices["heat-mc-study"].add_argument(
        "--max-walks", type=int, default=16384
    )
    subparsers.choices["heat-mc-study"].add_argument(
        "--seconds-per-method", type=float, default=60.0
    )
    args = parser.parse_args(argv)
    if args.command in ("heat-mc", "heat-mc-study"):
        from .heat import make_heat_problem
        from .heat_mc import solve_heat_mc
        from .heat_mc_study import export_heat_mc_study, heat_mc_study
        from .heat_mc_views import export_heat_mc

        try:
            problem = make_heat_problem(args.problem, args.n, args.a2)
            point = None if args.point is None else (args.point[1], args.point[0])
            if args.command == "heat-mc":
                result = solve_heat_mc(
                    problem,
                    args.dt,
                    args.T,
                    None if point is None else [point],
                    args.walks,
                    args.seed,
                    args.max_transitions,
                )
                export_heat_mc(result, problem, args.dt, args.out)
                print(
                    f"MC до t={args.T:g}: {result.seconds:.4g} с, {result.spatial_transitions+result.temporal_transitions} переходов"
                )
            else:
                study = heat_mc_study(
                    problem,
                    args.dt,
                    args.T,
                    point,
                    args.repeats,
                    args.max_walks,
                    args.seed,
                    args.seconds_per_method,
                    args.max_transitions,
                )
                export_heat_mc_study(study, args.out)
            print(f"Результат: {args.out.resolve()}")
        except (ValueError, RuntimeError) as exc:
            parser.exit(2, f"Ошибка: {exc}\n")
        return
    if args.command in ("heat", "heat-study"):
        try:
            if args.command == "heat-study":
                from .heat_study import export_heat_study, heat_accuracy_study

                export_heat_study(heat_accuracy_study(), args.out)
            else:
                from .heat import make_heat_problem, solve_heat
                from .heat_views import export_heat

                result = solve_heat(
                    make_heat_problem(args.problem, args.n, args.a2),
                    args.dt,
                    args.T,
                    args.method,
                    args.tol,
                    args.max_iter,
                )
                export_heat(result, args.out)
                print(
                    f"Физическое время t={result.times[-1]:g}; слоёв: {len(result.times)}"
                )
            print(f"Результат: {args.out.resolve()}")
        except (ValueError, RuntimeError) as exc:
            parser.exit(2, f"Ошибка: {exc}\n")
        return
    config_fields = Config.__dataclass_fields__.keys()
    config = Config(**{k: getattr(args, k) for k in config_fields})

    def progress(frac, label):
        print(f"{frac:.0%} · {label}", flush=True)

    try:
        if args.command == "study":
            from .research import StudySettings, accuracy_study, export_study

            study = accuracy_study(
                StudySettings(
                    config, args.repeats, args.max_walks, args.seconds_per_method
                ),
                progress,
            )
            export_study(study, args.out)
        elif args.command == "run":
            exp = run_experiment(config, progress=progress)
            export(exp, args.out, with_gifs=not args.no_gifs)
            print("\nЭталон: " + exp.comparison.label)
            print(
                "\nМетод                          мс          RMSE к эталону    Отн. ошибка, %"
            )
            for row in exp.metrics:
                relative = (
                    "—"
                    if row["relative_error_pct"] is None
                    else f'{row["relative_error_pct"]:.6g}'
                )
                print(
                    f'{row["name"]:28s} {row["seconds"]*1000:10.3f} {row["rmse"]:16.6g} {relative:>17s}'
                )
            print(
                f'\nИнтерактивный результат: {(args.out/"interactive.html").resolve()}'
            )
        else:
            rows = benchmark(
                config, args.budgets, args.repeats, args.tolerances, progress
            )
            benchmark_plot(rows, args.out)
            (args.out / "benchmark-config.json").write_text(
                json.dumps(
                    {
                        "config": asdict(config),
                        "repeats": args.repeats,
                        "budgets": args.budgets,
                        "tolerances": args.tolerances,
                        "lod_stopping_rule": LOD_STOPPING_RULE,
                        "metrics_policy": METRICS_POLICY,
                        "reference_policy": REFERENCE_POLICY,
                        "reference_kind": rows[0]["reference_kind"],
                        "reference_label": rows[0]["reference_label"],
                        "python": platform.python_version(),
                        "platform": platform.platform(),
                        "cpu_count": os.cpu_count(),
                        "timing": "After JIT warm-up; reference and visualization excluded; all layer preparation included.",
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            print(f'\nРезультат: {(args.out/"benchmark.html").resolve()}')
    except (ValueError, RuntimeError) as exc:
        parser.exit(2, f"Ошибка: {exc}\n")


if __name__ == "__main__":
    main()
