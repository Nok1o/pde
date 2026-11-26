#!/usr/bin/env python3
"""Build historical stationary 3D report inputs from saved data only.

No solver imports, benchmark runs, subprocesses or repository writes occur.
Run: python pde_build_legacy3d.py --root /path/to/pde
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import fmean

CASE_IDS = ("S07", "S08", "S10", "S11")
FACES = ("x0", "x1", "y0", "y1", "z0", "z1")
KIND_NUMBERS = {"dirichlet": 1, "neumann": 2, "robin": 3}
METHOD_NAMES = {
    "lod": "ЛОМ",
    "mc": "Обычный MC",
    "wos": "WoS",
    "graded": "MC на сгущённой сетке",
}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def same_number(actual, expected):
    return math.isclose(actual, expected, rel_tol=1e-11, abs_tol=1e-12)


def reason_for(row):
    target = row["target_pct"]
    if row["status"] == "no_pilot_margin":
        fraction = 1.0 if row["method"] == "lod" else 0.5
        return (
            "Подбор не нашёл бюджета со средней пилотной ошибкой "
            f"≤ {fraction * target:g}%; финальная серия не запускалась."
        )
    if row["status"] == "validation_failed":
        return (
            f"Независимая проверка: ошибка {row['error_pct']:.3f}% "
            f"превысила порог {target:g}%; сохранённое время не является "
            "временем достижения цели."
        )
    if row["status"] == "solver_failed":
        return "Выбранная финальная серия содержит отказ или не завершена полностью."
    if row["method"] == "lod":
        return "Средняя ошибка прошла порог; базовое время сравнения."
    if row.get("repeat_win"):
        return "Порог пройден; верхняя граница ошибки ≤ цели и нижняя граница ускорения > 1."
    reasons = ["Средняя ошибка прошла порог"]
    if row.get("error_ci", [0, 0])[1] > target:
        reasons.append(
            f"верхняя 95%-граница ошибки {row['error_ci'][1]:.3f}% выше цели"
        )
    speedup = row.get("speedup")
    if speedup is not None and speedup <= 1:
        reasons.append("метод медленнее ЛОМ")
    elif row.get("speed_ci", [0])[0] <= 1:
        reasons.append("нижняя 95%-граница ускорения не превышает 1")
    return "; ".join(reasons) + "."


def build_row(row, raw, planned_repeats):
    parameters = row["parameters"]
    group = (
        []
        if parameters is None
        else [
            trial
            for trial in raw["trials"]
            if trial["phase"] == 1
            and trial["method"] == row["method"]
            and trial["parameters"] == parameters
        ]
    )
    successful = [trial for trial in group if trial["status"] == "ok"]
    if group:
        assert len({trial["candidate"] for trial in group}) == 1
        assert len({trial["repeat"] for trial in group}) == len(group)
    if row["status"] in ("confirmed", "validation_failed"):
        assert len(successful) == len(group) == planned_repeats
        for trial in successful:
            values, reference = trial["value"], raw["target"]
            error = 100 * sum(abs(v - r) for v, r in zip(values, reference))
            error /= sum(abs(r) for r in reference)
            assert same_number(error, trial["relative_error_pct"])
        assert same_number(
            fmean(t["relative_error_pct"] for t in group), row["error_pct"]
        )
        assert same_number(fmean(t["seconds"] for t in group), row["seconds"])
    walks = [trial.get("walks") for trial in group]
    return {
        "method": row["method"],
        "name": METHOD_NAMES[row["method"]],
        "target": row["target_pct"],
        "error_pct": row["error_pct"],
        "seconds": row["seconds"],
        "budget": None if parameters is None else next(iter(parameters.values())),
        "status": row["status"],
        "reason": reason_for(row),
        "repeats": planned_repeats,
        "completed": len(successful),
        "recorded_repeats": len(group),
        "phase": "validation" if group else "not_run",
        "speedup": row.get("speedup"),
        "total_walks": (
            sum(walks) if walks and all(v is not None for v in walks) else None
        ),
    }


def build_cases(root):
    folder = root / "benchmarks/results/audit-recheck-20261111"
    protocol_path = folder / "protocol.json"
    protocol = read_json(protocol_path)
    cases = []
    for case_id in CASE_IDS:
        path = folder / f"{case_id}.json"
        raw = read_json(path)
        case = raw["case"]
        n = case["n"]
        point = case.get("point", [n // 2] * 3)
        if case["family"] == "3d":
            kinds = ["dirichlet"] * 6
            label = "Синус, все грани 1 рода"
            formula = "u*=sin(πx)sin(πy)sin(πz)"
        else:
            kinds = case.get("kinds", ["robin"] + ["dirichlet"] * 5)
            if case["problem"] == "smooth":
                label = "Гладкое решение, 3 род на x=0"
                formula = "u*=3+x+2y+3z+cos(2πx)+0.5cos(2πy)+0.25cos(2πz)"
            else:
                label = "Полином, 2 род на x=0 и 3 род на z=1"
                formula = "u*=1+x+2y+3z+x²+y²+z²"
        assert raw["points"] == [point[::-1]]
        rows = [
            build_row(row, raw, protocol["validation"]) for row in raw["comparisons"]
        ]
        expected = {
            (method, target) for method in case["methods"] for target in (1, 5, 10)
        }
        assert {(row["method"], row["target"]) for row in rows} == expected
        cases.append(
            {
                "id": f"legacy3d-{case_id}",
                "title": f"{case_id}: {label}; 3D, n={n}, точка {tuple(point)}",
                "section": "Исторический контроль: стационарные задачи 3D",
                "settings": {
                    "archive_case": case_id,
                    "dimension": 3,
                    "equation": "Δu=f, [0,1]³",
                    "preset": case["problem"],
                    "n_intervals": n,
                    "nodes_per_axis": n + 1,
                    "point_xyz_indices": point,
                    "point_xyz_coordinates": [v / n for v in point],
                    "boundary_kind_by_face": dict(
                        zip(FACES, [KIND_NUMBERS[k] for k in kinds])
                    ),
                    "beta_on_third_kind_faces": 1.0,
                    "accuracy_metric": protocol["metric"],
                    "error_formula": "mean_r(100*|u_r-u*|/|u*|)",
                    "pilot_repeats": protocol["pilot"],
                    "final_repeats": protocol["validation"],
                    "mc_pilot_fraction": protocol["margin"],
                    "lod_pilot_fraction": 1.0,
                    "seed": protocol["seed"],
                    "python": protocol["python"],
                    "platform": protocol["platform"],
                    "walk_budgets": protocol["budgets"],
                    "lod_tolerances": protocol["tolerances"],
                    "max_iterations": 20000,
                    "max_steps_per_walk": 1000 * n**2,
                    "budget_kind": {
                        "lod": "relative_residual_tolerance",
                        "others": "walks_per_point",
                    },
                    "timing": protocol["timing"],
                    "uncertainty": protocol["uncertainty"],
                    "raw_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "protocol_source": str(protocol_path),
                    "current_version_measured": False,
                },
                "source": str(path),
                "commit": protocol["revision"],
                "protocol": "historical_mae_pilot5_margin0.5_fastest_candidate_independent_final50",
                "reference": {
                    "kind": "analytic",
                    "value": raw["target"][0],
                    "formula": formula,
                },
                "rows": rows,
            }
        )
    return {
        "schema": "pde.legacy3d.cases.v1",
        "cases": cases,
        "notes": [
            "Исторические результаты, не замеры текущей версии. commit — записанный HEAD; recheck не сохранял полный набор хешей исходников.",
            "Все 42 строки и все пороги 1/5/10% сохранены; status=confirmed означает достижение среднего порога, не автоматически выигрыш по времени.",
            "95%-интервалы без поправки на множественные сравнения. Устойчивость выигрыша указана в reason по сохранённому repeat_win.",
            "seconds при validation_failed — фактическое время неудачной финальной проверки, не время достижения цели.",
            "Средняя абсолютная относительная ошибка отдельных запусков; для одной точки совпадает с нынешней средней относительной L2-ошибкой. Это не RMS по запускам.",
            "budget для ЛОМ — допуск относительной невязки; для MC — пути на точку. total_walks — сумма фактических путей выбранной финальной серии, без подбора.",
            "Одна и та же финальная серия может подтверждать несколько порогов: числа повторов и путей таких строк нельзя складывать как независимые затраты.",
            "Для no_pilot_margin финальной серии нет: repeats — запланированные 50, completed=recorded_repeats=0; неизвестное/неприменимое не заменено нулевым временем.",
            "Не сравнивать эти абсолютные времена с текущими heat3d-замерами: другая задача, версия, среда и протокол.",
        ],
    }


def build_note(root):
    folder = root / "benchmarks/results/demo-20261111"
    raw = read_json(folder / "confirmation.json")
    rows = [
        row for row in raw["comparisons"] if row["case"] in ("D07", "D08", "D09", "D10")
    ]
    assert len(rows) == 16 and sum(row["confirmed"] for row in rows) == 14
    selected = {(row["case"], row["method"], row["target_pct"]): row for row in rows}
    d08_5 = selected["D08", "mc", 5]["speedup"]
    d08_10 = selected["D08", "mc", 10]["speedup"]
    d07_10 = selected["D07", "mc", 10]["speedup"]
    url = "https://github.com/Nok1o/pde/blob/3d5e0a22041d83674be42d23a3c870eadd0737fd/benchmarks/results/demo-20261111/confirmation.json"
    return (
        "Отдельная историческая проверка уже зафиксированных бюджетов на 50 новых запусках "
        f"подтвердила для обычного MC в центре синуса 3D: D08, n=16 — {d08_5:.2f}× "
        f"при пороге 5% и {d08_10:.2f}× при 10%; D07, n=8 — {d07_10:.2f}× при 10% "
        f"([исходные повторы и сравнения]({url})). Метрика — среднее абсолютных "
        "относительных ошибок отдельных запусков к точному решению; бюджеты после "
        "финальной проверки не меняли. Во всех четырёх 3D-сценариях сохранены 16 сравнений: "
        "14 подтверждены, два случая D10 на 1% (обычный MC и сгущённая сетка) не подтвердились "
        "из-за верхней границы ошибки выше цели; они не исключены из исходного архива. "
        "Это историческое свидетельство, не измерение нынешней версии: в protocol.json "
        "записан HEAD 7cca8de, но фактические SHA256 релевантных стационарных файлов "
        "соответствуют опубликованным версиям 3d5e0a2; нельзя приписывать замер чистому "
        "7cca8de. Современный протокол подбора и закреплённая версия Python отличаются.\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--out", type=Path, default=Path("/private/tmp/pde-legacy3d-cases.json")
    )
    parser.add_argument(
        "--note", type=Path, default=Path("/private/tmp/pde-legacy3d-note.md")
    )
    args = parser.parse_args()
    root = args.root.resolve()
    result = build_cases(root)
    assert len(result["cases"]) == 4
    assert sum(len(case["rows"]) for case in result["cases"]) == 42
    args.out.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    args.note.write_text(build_note(root), encoding="utf-8")
    print(f"4 cases, 42 rows; saved {args.out} and {args.note}. No solver calls.")


if __name__ == "__main__":
    main()
