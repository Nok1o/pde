"""Build one source-separated comparison document from saved JSON only.

No pdelab imports, numerical solvers, charts, or repository writes.  The public
build_existing(root) and write_outputs(data, prefix) functions can be reused to
append separately reviewed fresh cases before rendering the final report.
"""

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

METHOD_NAMES = {
    "lod": "ЛОМ",
    "lod_serial": "ЛОМ",
    "block": "Матричная прогонка",
    "mc": "Обычный MC",
    "mc_control": "MC с контрольным приближением",
    "mc_grouped": "MC: пакеты отдельных частиц",
    "mc_control_grouped": "Контрольный MC: пакеты отдельных частиц",
    "mc_counts": "MC: перенос количества частиц",
    "mass_transport": "Детерминированный перенос массы",
    "reuse": "MC: хвосты путей",
    "layers": "MC: слои от границы",
    "wos": "WoS",
    "graded": "MC на сгущённой сетке",
}
ORDER = {method: index for index, method in enumerate(METHOD_NAMES)}
PROBLEMS = {
    "affine": "Линейное поле",
    "constant": "Постоянное поле",
    "sine": "Синус",
    "polynomial": "Полином",
    "hot_top": "Нагретый край",
    "smooth": "Гладкое решение",
    "decay": "Затухание синуса",
    "manufactured": "Изготовленное решение",
}
SECTION_NAMES = {
    "final_core": "Полная нелинейная 3D-теплопроводность: шесть методов",
    "conductivity": "Полная 3D-история: частота обновления проводимости",
    "coarse": "Полная 3D-история: размер контрольной сетки",
    "transport_frozen": "Перенос частиц: одна замороженная 3D-система",
    "transport_full": "Перенос частиц: полная нелинейная 3D-история",
    "legacy_heat": "Историческая линейная и квазилинейная 2D-теплопроводность",
    "legacy_stationary": "Исторические стационарные 2D-задачи",
}
STATUS_NAMES = {
    "confirmed": "подтверждено",
    "not_reached": "нет запаса точности",
    "pilot_incomplete": "подбор не завершён",
    "validation_failed": "финал не подтвердил цель",
    "validation_incomplete": "финальная серия не завершена",
    "no_pilot_margin": "нет запаса точности на подборе",
    "solver_failed": "отказ решателя",
    "relative_target_not_applicable": "процентная цель неприменима",
}


def read(path):
    return json.loads(path.read_text())


def sum_saved(trials, key):
    if not trials:
        return 0
    values = [trial.get(key) for trial in trials]
    return sum(values) if all(value is not None for value in values) else None


def location(root, path, pointer=None):
    return {
        "path": str(path.relative_to(root)),
        "pointer": pointer,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def boundary_label(kinds):
    if not kinds:
        return "условия 1 рода"
    mapped = [{"dirichlet": 1, "neumann": 2, "robin": 3}.get(k, k) for k in kinds]
    if len(set(mapped)) == 1:
        return f"все границы {mapped[0]} рода"
    labels = ("x0", "x1", "y0", "y1", "z0", "z1")[: len(mapped)]
    return ", ".join(f"{face}: {kind} род" for face, kind in zip(labels, mapped))


def title(config, scenario):
    problem = PROBLEMS.get(
        config.get("case", config.get("problem")), "Квазилинейная задача"
    )
    mode = {"point": "точка", "patch": "участок", "field": "поле", "slice": "срез"}.get(
        config.get("mode", "point"), "точка"
    )
    text = f"{problem}; n={config['n']}; {mode}; {boundary_label(config.get('kinds'))}"
    if "capacity_gamma" in config:
        text += f"; γ={config['capacity_gamma']:g}, σ={config['reaction_sigma']:g}"
    if scenario.startswith("Q"):
        text += f"; α={config['alpha']:g}"
    return text


def validate_final(study, row):
    selected = [
        t
        for t in study["trials"]
        if (t["phase"], t["candidate"]) == (row["phase"], row["candidate"])
    ]
    assert len(selected) == row["recorded_repeats"]
    if row["status"] == "confirmed":
        assert row["phase"] == "final"
        assert len(selected) == row["requested_repeats"]
        assert all(t["status"] == "completed" for t in selected)
        assert row["relative_error_pct"] <= row["target_pct"]
        assert math.isclose(
            mean(t["seconds"] for t in selected), row["seconds"], rel_tol=1e-12
        )
    return selected


def load_modern(root, archive, stages):
    cases = {}
    for stage, section in stages:
        folder = root / "benchmarks/results" / archive / stage
        for path in sorted(folder.glob("*.json")):
            if path.name in ("manifest.json", "plan.json") or path.name.endswith(
                "--reference.json"
            ):
                continue
            payload = read(path)
            if "study" not in payload:
                continue
            study = payload["study"]
            meta = study["metadata"]
            scenario, variant = path.stem.split("--")
            config = dict(meta["configuration"])
            config.pop("methods", None)
            solver_settings = meta.get("solver_settings", meta.get("settings", {}))
            commit = meta.get("commit") or meta["source"]["commit"]
            key = f"{archive}/{stage}/{scenario}"
            if key not in cases:
                cases[key] = dict(
                    id=key,
                    title=title(config, scenario),
                    section=section,
                    section_title=SECTION_NAMES[section],
                    archive=archive,
                    stage=stage,
                    scenario=scenario,
                    settings={"configuration": config, "methods": {}},
                    source=[],
                    commit=commit,
                    protocol={
                        "id": study["protocol"],
                        "accuracy_metric": study["accuracy_metric"],
                        "repeats": study["requested_repeats"],
                        "calibration_fraction": study["calibration_fraction"],
                        "seed": study["seed"],
                        "independent_final": True,
                    },
                    reference={
                        "kind": meta["reference_kind"],
                        "norm": study["reference_norm"],
                        "scope": meta.get("scope", "complete_physical_time_history"),
                    },
                    environment={
                        k: meta.get(k) for k in ("python", "platform", "versions")
                    },
                    notes=[],
                    rows=[],
                )
            case = cases[key]
            assert case["commit"] == commit
            assert case["settings"]["configuration"] == config
            case["settings"]["methods"][variant] = solver_settings
            case["source"].append(location(root, path, "study"))
            if section == "conductivity":
                case["baseline_rule"] = (
                    "ЛОМ с периодом обновления проводимости 1 в этой же серии"
                )
            elif section == "coarse":
                case["baseline_rule"] = (
                    "Собственного ЛОМ нет; ускорение относительно ЛОМ не вычисляется"
                )
            else:
                case["baseline_rule"] = "ЛОМ того же сценария и той же серии"
            for index, raw in enumerate(study["rows"]):
                trials = validate_final(study, raw)
                method = raw["method"]
                name = METHOD_NAMES.get(method, method)
                if section == "conductivity":
                    name += (
                        f"; период k={solver_settings['conductivity_refresh_layers']}"
                    )
                if section == "coarse":
                    coarse_n = solver_settings["coarse_n"]
                    name += f"; H/h={config['n'] / coarse_n:g}"
                row = dict(
                    method=method,
                    variant=variant,
                    name=name,
                    target=raw["target_pct"],
                    error_pct=raw["relative_error_pct"],
                    seconds=raw["seconds"] if raw["status"] == "confirmed" else None,
                    budget=raw["budget"],
                    budget_kind=raw["budget_kind"],
                    status=raw["status"],
                    reason=raw["reason"],
                    repeats=raw["requested_repeats"],
                    completed=raw["completed_repeats"],
                    recorded=raw["recorded_repeats"],
                    recorded_repeats=raw["recorded_repeats"],
                    phase=raw["phase"],
                    speedup=None,
                    spent_seconds=raw["elapsed_seconds"],
                    elapsed_seconds=raw["elapsed_seconds"],
                    diagnostic_seconds=(
                        raw["elapsed_seconds"] if raw["status"] != "confirmed" else None
                    ),
                    phase_mean_seconds=raw["mean_seconds"],
                    actual_walks=raw.get("total_walks"),
                    total_walks=raw.get("total_walks"),
                    completed_walks=raw.get("completed_walks"),
                    transitions=raw.get("transitions"),
                    pilot_error_pct=raw.get("calibration_error_pct"),
                    source=location(root, path, f"study.rows[{index}]"),
                    raw=raw,
                    failure_reasons=dict(
                        Counter(
                            t["reason"] for t in trials if t["status"] != "completed"
                        )
                    ),
                )
                case["rows"].append(row)
    for case in cases.values():
        for row in case["rows"]:
            baseline = next(
                (
                    r
                    for r in case["rows"]
                    if r["method"] == "lod"
                    and r["target"] == row["target"]
                    and (
                        case["section"] != "conductivity"
                        or case["settings"]["methods"][r["variant"]][
                            "conductivity_refresh_layers"
                        ]
                        == 1
                    )
                ),
                None,
            )
            if (
                case["section"] != "coarse"
                and baseline
                and row["status"] == baseline["status"] == "confirmed"
            ):
                row["speedup"] = baseline["seconds"] / row["seconds"]
                row["baseline_variant"] = baseline["variant"]
        if case["section"] == "transport_frozen":
            case["notes"].append(
                "Сеточный эталон одной замороженной системы; не ошибка полной временной задачи. Подготовка оператора включена в время каждого запуска."
            )
        else:
            case["notes"].append(
                "Эталон — аналитическое решение в конечный момент. Время включает всю подготовку и физическую историю; JIT и эталон исключены."
            )
        if case["section"] == "final_core":
            case["notes"].append(
                "Групповые методы здесь — пакеты отдельных траекторий, не перенос количества частиц."
            )
    return list(cases.values())


def load_historical(root):
    folder = root / "benchmarks/results/audit-recheck-20261111"
    protocol_path = folder / "protocol.json"
    protocol = read(protocol_path)
    result = []
    for scenario in (
        "H01",
        "H02",
        "Q01",
        "Q02",
        "S01",
        "S02",
        "S03",
        "S04",
        "S05",
        "S06",
        "S09",
        "S12",
        "S13",
    ):
        path = folder / f"{scenario}.json"
        data = read(path)
        config = dict(data["case"])
        if config.get("family") == "mixed2d" and "kinds" not in config:
            # BoundaryConfig default at archived revision 47ced68a.
            config["kinds"] = ["robin", "dirichlet", "dirichlet", "dirichlet"]
        methods = config.pop("methods")
        section = (
            "legacy_heat" if scenario.startswith(("H", "Q")) else "legacy_stationary"
        )
        kind = "analytic"
        if scenario == "S04":
            kind = "analytic_point_by_symmetry"
        elif scenario == "S13":
            kind = "analytic_normalized_gauge"
        case = dict(
            id=f"audit-recheck-20261111/{scenario}",
            title=title(config, scenario),
            section=section,
            section_title=SECTION_NAMES[section],
            archive="audit-recheck-20261111",
            stage=None,
            scenario=scenario,
            settings={
                "configuration": config,
                "archived_configuration": data["case"],
                "method_list": methods,
                "points_yx": data["points"],
                "candidate_tolerances": protocol["tolerances"],
                "candidate_walk_budgets": protocol["budgets"],
            },
            source=[location(root, path), location(root, protocol_path)],
            commit=protocol["revision"],
            protocol={
                "id": "historical_fixed_final_fastest_eligible_pilot",
                "accuracy_metric": protocol["metric"],
                "repeats": 50,
                "pilot_repeats": 5,
                "calibration_fraction_mc": 0.5,
                "calibration_fraction_lod": 1,
                "seed": protocol["seed"],
                "independent_final": True,
            },
            reference={
                "kind": kind,
                "values": data["target"],
                "query_size": len(data["points"]),
                "construction": "benchmarks/project_audit.py:prepare",
            },
            environment={
                "python": protocol["python"],
                "platform": protocol["platform"],
                "versions": None,
            },
            historical=True,
            optional=section == "legacy_stationary",
            baseline_rule="ЛОМ той же исторической серии и цели",
            notes=[
                "Исторические данные, не повторный запуск текущей версии. Для нескольких точек используется относительная L1-ошибка, не современная L2. Для одной точки обе метрики совпадают.",
                "При выборе MC требовался запас 0.5 цели; у ЛОМ — сама цель. Финал после фиксации бюджета не использован для повторного подбора.",
            ],
            rows=[],
        )
        pilot_groups = defaultdict(list)
        for trial in data["trials"]:
            if trial["phase"] == 0:
                pilot_groups[trial["method"], trial["candidate"]].append(trial)
        best = {}
        for (method, _), group in pilot_groups.items():
            if len(group) == 5 and all(t["status"] == "ok" for t in group):
                item = {
                    "error_pct": mean(t["relative_error_pct"] for t in group),
                    "budget": group[0]["parameters"],
                    "phase": "pilot",
                    "repeats": 5,
                    "completed": 5,
                    "recorded_repeats": len(group),
                    "mean_seconds": mean(t["seconds"] for t in group),
                    "elapsed_seconds": sum(t["seconds"] for t in group),
                    "total_walks": sum_saved(group, "walks"),
                    "completed_walks": None,
                    "transitions": sum_saved(group, "steps"),
                    "trial_indices": [
                        i for i, t in enumerate(data["trials"]) if t in group
                    ],
                }
                if method not in best or item["error_pct"] < best[method]["error_pct"]:
                    best[method] = item
        for index, raw in enumerate(data["comparisons"]):
            trials = [
                t
                for t in data["trials"]
                if t["phase"] == 1
                and t["method"] == raw["method"]
                and t["parameters"] == raw["parameters"]
            ]
            if raw["status"] == "confirmed":
                assert len(trials) == 50 and all(t["status"] == "ok" for t in trials)
                assert raw["error_pct"] <= raw["target_pct"]
                assert math.isclose(
                    mean(t["relative_error_pct"] for t in trials),
                    raw["error_pct"],
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                )
                assert math.isclose(
                    mean(t["seconds"] for t in trials), raw["seconds"], rel_tol=1e-12
                )
            reasons = dict(Counter(t["status"] for t in trials if t["status"] != "ok"))
            reason = STATUS_NAMES.get(raw["status"], raw["status"])
            if raw["status"] == "no_pilot_margin":
                fraction = 1 if raw["method"] == "lod" else 0.5
                reason = f'Не найден полный подбор с ошибкой ≤ {raw["target_pct"] * fraction:g}%; финал не запускался'
            if reasons:
                reason += "; " + "; ".join(reasons)
            case["rows"].append(
                dict(
                    method=raw["method"],
                    variant=raw["method"],
                    name=METHOD_NAMES.get(raw["method"], raw["method"]),
                    target=raw["target_pct"],
                    error_pct=raw["error_pct"],
                    seconds=raw["seconds"] if raw["status"] == "confirmed" else None,
                    budget=raw["parameters"],
                    budget_kind="parameters",
                    status=raw["status"],
                    reason=reason,
                    repeats=50,
                    completed=sum(t["status"] == "ok" for t in trials),
                    recorded=len(trials),
                    recorded_repeats=len(trials),
                    phase="final" if trials else "not_started",
                    speedup=None,
                    spent_seconds=sum(t["seconds"] for t in trials),
                    elapsed_seconds=sum(t["seconds"] for t in trials),
                    diagnostic_seconds=(
                        sum(t["seconds"] for t in trials)
                        if raw["status"] != "confirmed"
                        else None
                    ),
                    total_walks=sum_saved(trials, "walks"),
                    completed_walks=None,
                    transitions=sum_saved(trials, "steps"),
                    search_costs={
                        "elapsed_seconds": sum(
                            t["seconds"]
                            for t in data["trials"]
                            if t["phase"] == 0 and t["method"] == raw["method"]
                        ),
                        "total_walks": sum_saved(
                            [
                                t
                                for t in data["trials"]
                                if t["phase"] == 0 and t["method"] == raw["method"]
                            ],
                            "walks",
                        ),
                        "recorded_repeats": sum(
                            t["phase"] == 0 and t["method"] == raw["method"]
                            for t in data["trials"]
                        ),
                    },
                    phase_mean_seconds=(
                        mean(t["seconds"] for t in trials) if trials else None
                    ),
                    diagnostic=(
                        best.get(raw["method"])
                        if raw["status"] == "no_pilot_margin"
                        else None
                    ),
                    speed_ci=raw.get("speed_ci"),
                    error_ci=raw.get("error_ci"),
                    archived_repeat_win=raw.get("repeat_win"),
                    source=location(root, path, f"comparisons[{index}]"),
                    raw=raw,
                )
            )
        for row in case["rows"]:
            baseline = next(
                r
                for r in case["rows"]
                if r["method"] == "lod" and r["target"] == row["target"]
            )
            if row["status"] == baseline["status"] == "confirmed":
                row["speedup"] = baseline["seconds"] / row["seconds"]
        result.append(case)
    return result


def build_existing(root):
    root = Path(root)
    cases = load_modern(
        root,
        "heat3d-final",
        [
            ("core-n8", "final_core"),
            ("core-n16", "final_core"),
            ("conductivity-n8", "conductivity"),
            ("coarse-n16", "coarse"),
        ],
    )
    cases += load_modern(
        root,
        "heat3d-transport",
        [
            ("frozen-n8", "transport_frozen"),
            ("full-n8", "transport_full"),
            ("full-n16", "transport_full"),
        ],
    )
    cases += load_historical(root)
    modern_final = sum(len(c["rows"]) for c in cases if c["archive"] == "heat3d-final")
    modern_transport = sum(
        len(c["rows"]) for c in cases if c["archive"] == "heat3d-transport"
    )
    historical = sum(
        len(c["rows"]) for c in cases if c["archive"] == "audit-recheck-20261111"
    )
    assert (modern_final, modern_transport, historical) == (300, 96, 159)
    return dict(
        schema="pde_scenario_comparisons_v1",
        source_root=str(root),
        cases=cases,
        interpretation=[
            "Цель означает верхнюю границу ошибки, а не строго одинаковое численное значение ошибки у разных методов.",
            "Среднее время подтверждённого финала показано только для полностью завершённой независимой серии. Подбор, отказы и затраты сохранены отдельно.",
            "Ошибка для точки — средняя абсолютная относительная ошибка отдельных полных запусков. Это не RMSE и не ошибка среднего решения.",
            "Архивы, версии, семейства, эталоны и протоколы не объединяются в одну измерительную серию.",
            "× = время ЛОМ / время метода. Значение вычислено только при подтверждении обеих целей в допустимой общей серии.",
            "Для проводимости baseline — ЛОМ с периодом 1 той же серии. Для H/h без собственного ЛОМ коэффициент не вычисляется.",
            "В fixed-budget исследованиях нет независимой приёмки 1/5/10; они не превращаются в строки таблицы целей.",
            "Числа разных целей могут ссылаться на одну финальную серию. Время/пути строк целей нельзя суммировать как стоимость всего исследования.",
        ],
        excluded=[
            "accuracy, project-audit и heat-mc: старый RMS; не переименовывается в MAE.",
            "heat3d-pilot и heat3d-six-methods: прежние серии, замещены heat3d-final для основной матрицы.",
            "demo-20261111: адаптивный протокол, не смешан с фиксированным финалом; D07-D10 при необходимости отдельным приложением.",
            "heat/convergence, heat3d-time-mc, heat-mc/costs, heat3d-final-acceptance: сходимость/диагностика/функциональная приёмка, не таблица достижения целей.",
        ],
    )


def number(value):
    return "—" if value is None else f"{value:.5g}"


def count_text(value):
    return "—" if value is None else f"{value:,}".replace(",", " ")


def budget_text(value, kind=""):
    if value is None:
        return "бюджет не выбран"
    if isinstance(value, dict):
        return ", ".join(
            ("N" if k == "walks" else "ε" if k == "tol" else k)
            + "="
            + (count_text(v) if k == "walks" else number(v))
            for k, v in value.items()
        )
    return ("N=" + count_text(value)) if "walk" in kind else ("ε=" + number(value))


def cell(row):
    if row["status"] == "confirmed":
        text = f"{number(row['seconds'] * 1000)} мс; E={number(row['error_pct'])}%; {budget_text(row['budget'], row.get('budget_kind', ''))}"
        if row["speedup"] is not None:
            ratio = f"×{number(row['speedup'])}"
            text += "; " + (
                f"**{ratio}**"
                if row["speedup"] > 1 and row["method"] != "lod"
                else ratio
            )
        return text
    text = STATUS_NAMES.get(row["status"], row["status"])
    if row["phase"] == "not_started":
        text += "; финал не запускался"
        diagnostic = row.get("diagnostic")
        if diagnostic:
            text += f"; лучший подбор: E={number(diagnostic['error_pct'])}%, {budget_text(diagnostic['budget'])}, {diagnostic['completed']}/{diagnostic['repeats']}"
    else:
        phase = "подбор" if row["phase"] == "pilot" else "финал"
        text += f"; {phase}: завершено {row['completed']}/{row['recorded_repeats']}, запрошено {row['repeats']}; E={number(row['error_pct'])}%"
        text += "; " + budget_text(row["budget"], row.get("budget_kind", ""))
        text += f"; потрачено {number(row['spent_seconds'] * 1000)} мс"
    text += "; " + row["reason"]
    return text.replace("|", "\\|").replace("\n", " ")


def settings_text(case):
    config = case["settings"]["configuration"]
    labels = {
        "n": "n",
        "alpha": "α",
        "capacity_gamma": "γ",
        "reaction_sigma": "σ",
        "dt": "dt",
        "T": "T",
        "point": "точка (x,y,z)",
        "mode": "запрос",
    }
    parts = [f"{label}={config[key]}" for key, label in labels.items() if key in config]
    if len(config.get("point", [])) == 2:
        parts = [x.replace("точка (x,y,z)", "точка (x,y)") for x in parts]
    if "points_yx" in case["settings"]:
        points = case["settings"]["points_yx"]
        parts.append(f"узлов запроса={len(points)}")
        if len(points) == 1:
            parts.append(f"точка (x,y)={list(reversed(points[0]))}")
    return "; ".join(parts)


def render_markdown(data):
    lines = ["# Сравнение методов при целях 1%, 5% и 10%", ""]
    lines += [f"- {text}" for text in data.get("interpretation", [])]
    lines += [
        "",
        "Ячейка: среднее время финала / фактическая ошибка / бюджет / отношение времени ЛОМ. У неудач указаны фаза, завершённые повторы и затраты; они не считаются временем достижения цели.",
        "",
    ]
    current = None
    for case in data["cases"]:
        section = case["section"]
        if section != current:
            lines += [
                "## " + case.get("section_title", SECTION_NAMES.get(section, section)),
                "",
            ]
            current = section
        lines += [
            "### " + case["title"],
            "",
            f"Идентификатор: `{case['id']}`. Версия: `{case['commit']}`. Метрика: `{case['protocol']['accuracy_metric']}`. Эталон: `{case['reference']['kind']}`.",
            "",
            settings_text(case) + ".",
            "",
            f"Повторов финала: {case['protocol']['repeats']}. {case['baseline_rule']}.",
            "",
        ]
        if case.get("historical"):
            lines += ["**Историческая серия; не измерение нынешней версии.**", ""]
        lines += ["| Метод | Цель 1% | Цель 5% | Цель 10% |", "|---|---|---|---|"]
        variants = {}
        for row in case["rows"]:
            variants.setdefault(row["variant"], []).append(row)
        ordered = sorted(
            variants.values(),
            key=lambda rows: (ORDER.get(rows[0]["method"], 99), rows[0]["variant"]),
        )
        for rows in ordered:
            by_target = {r["target"]: r for r in rows}
            lines.append(
                "| "
                + rows[0]["name"]
                + " | "
                + " | ".join(
                    cell(by_target[target]) if target in by_target else "Не измерено"
                    for target in (1, 5, 10)
                )
                + " |"
            )
        lines += [
            "",
            *case.get("notes", []),
            "",
            "Источники: "
            + ", ".join(
                f"[{Path(s['path']).name}](https://github.com/Nok1o/pde/blob/main/{s['path']})"
                for s in case["source"]
            )
            + ".",
            "",
        ]
    lines += [
        "## Сводка неподтверждённых целей",
        "",
        "Счётчики относятся к указанной диагностической серии; разные цели могут ссылаться на одну серию. Неприменимые или не сохранённые счётчики обозначены тире. У исторического отказа без финала показан лучший полный подбор, а не финальный результат.",
        "",
        "| Сценарий / метод / цель | Фаза; завершено / записано | Ошибка; бюджет | Запущено / завершено путей | Время серии, мс | Причина |",
        "|---|---|---|---|---|---|",
    ]
    for case in data["cases"]:
        for row in case["rows"]:
            if row["status"] == "confirmed":
                continue
            diagnostic = row.get("diagnostic") if row["phase"] == "not_started" else row
            diagnostic = diagnostic or row
            phase = (
                "лучший подбор; финал не запущен"
                if row["phase"] == "not_started"
                else ("подбор" if row["phase"] == "pilot" else "финал")
            )
            completed = diagnostic.get("completed", 0)
            recorded = diagnostic.get("recorded_repeats", 0)
            spent = diagnostic.get("elapsed_seconds")
            row_text = [
                f"`{case['id']}` / {row['name']} / {row['target']:g}%",
                f"{phase}; {completed}/{recorded}",
                f"E={number(diagnostic.get('error_pct'))}%; {budget_text(diagnostic.get('budget'), row.get('budget_kind', ''))}",
                f"{count_text(diagnostic.get('total_walks'))} / {count_text(diagnostic.get('completed_walks'))}",
                number(spent * 1000) if spent is not None else "—",
                row["reason"],
            ]
            lines.append(
                "| "
                + " | ".join(
                    text.replace("|", "\\|").replace("\n", " ") for text in row_text
                )
                + " |"
            )
    lines += ["", "## Не включено в сопоставимые таблицы", ""]
    lines += [f"- {text}" for text in data.get("excluded", [])]
    return "\n".join(lines) + "\n"


def write_outputs(data, prefix):
    prefix = Path(prefix)
    rows = [row for case in data["cases"] for row in case["rows"]]
    data["summary"] = {
        "cases": len(data["cases"]),
        "target_rows": len(rows),
        "statuses": dict(Counter(row["status"] for row in rows)),
        "by_section": {
            section: sum(
                len(c["rows"]) for c in data["cases"] if c["section"] == section
            )
            for section in dict.fromkeys(c["section"] for c in data["cases"])
        },
    }
    prefix.with_suffix(".json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    prefix.with_suffix(".md").write_text(render_markdown(data))
    return data["summary"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="Корень checkout с benchmarks/results",
    )
    parser.add_argument(
        "--output",
        "--out",
        dest="out",
        type=Path,
        default=Path("scenario-comparisons"),
        help="Префикс выходных JSON и Markdown",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            write_outputs(build_existing(args.root), args.out),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
