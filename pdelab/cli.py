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
        description="PDE Lab: Δu=f, сеточный метод и три варианта Монте-Карло"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for cmd in ["run", "benchmark"]:
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
    args = parser.parse_args(argv)
    config_fields = Config.__dataclass_fields__.keys()
    config = Config(**{k: getattr(args, k) for k in config_fields})

    def progress(frac, label):
        print(f"{frac:.0%} · {label}", flush=True)

    try:
        if args.command == "run":
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
