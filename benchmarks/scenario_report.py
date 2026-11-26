"""Build the combined comparison from saved data; never run a solver.

python -m benchmarks.scenario_report [--pdf]
The optional PDF renderer requires reportlab; the application does not.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import stdev
from xml.sax.saxutils import escape

from benchmarks import scenario_sources as sources

OUT = Path("benchmarks/results/scenario-comparison")
REPO_URL = "https://github.com/Nok1o/pde/blob/main/"


def historical_cases(root):
    from benchmarks.scenario_history3d import build_cases

    data = build_cases(root)
    for case in data["cases"]:
        settings = case["settings"]
        case["section_title"] = case["section"]
        case["section"] = "legacy_stationary3d"
        case["historical"] = True
        case["settings"] = {
            "configuration": {
                "n": settings["n_intervals"],
                "point": settings["point_xyz_indices"],
                "mode": "point",
            },
            "archive_settings": settings,
        }
        case["protocol"] = {
            "id": case["protocol"],
            "accuracy_metric": "mean_run_relative_l1_v1",
            "repeats": settings["final_repeats"],
        }
        case["source"] = [sources.location(root, Path(case["source"]))]
        case["baseline_rule"] = "ЛОМ той же исторической серии и цели"
        case["notes"] = [
            "Исторические результаты, записанный HEAD; полный снимок исходников не сохранён. "
            "Другие среда и протокол; это не замер текущей версии.",
            "Финал R=50 после подбора R=5; MC требует запаса 0,5 цели. "
            "Пройденная средняя ошибка не означает статистически устойчивый выигрыш по времени.",
            case["reference"]["formula"],
        ]
        for row in case["rows"]:
            row["variant"] = row["method"]
            row["recorded"] = row["recorded_repeats"]
            row["phase"] = "final" if row["phase"] == "validation" else "not_started"
            row["source"] = case["source"][0]
            row["spent_seconds"] = (row["seconds"] or 0) * row["recorded"]
            row["elapsed_seconds"] = row["spent_seconds"]
            row["phase_mean_seconds"] = row["seconds"]
            row["budget_kind"] = (
                "relative_residual_tolerance"
                if row["method"] == "lod"
                else "walks_per_start"
            )
            if row["status"] != "confirmed":
                row["seconds"] = None
                row["speedup"] = None
        for row in case["rows"]:
            baseline = next(
                r
                for r in case["rows"]
                if r["method"] == "lod" and r["target"] == row["target"]
            )
            if row["status"] == baseline["status"] == "confirmed":
                row["speedup"] = baseline["seconds"] / row["seconds"]
    return data["cases"]


def fresh_cases(root):
    folder = root / OUT / "raw"
    manifest = sources.read(folder / "manifest.json")
    assert manifest["complete"]
    assert manifest["source_before"] == manifest["source_after"]
    plan = sources.read(folder / "plan.json")
    cases = []
    for config in plan["cases"]:
        path = folder / (config["id"] + ".json")
        data = sources.read(path)
        assert (
            data["source_before"] == data["source_after"] == manifest["source_before"]
        )
        case = dict(
            id="stationary-current/" + config["id"],
            title=sources.title(config, config["id"]),
            section="stationary_current",
            section_title="Свежая стационарная 3D-задача: шесть методов",
            settings={"configuration": config, "plan": plan},
            source=[sources.location(root, path)],
            commit=data["source_before"]["commit"],
            protocol={
                "id": data["protocol"],
                "accuracy_metric": data["accuracy_metric"],
                "repeats": plan["repeats"],
                "calibration_fraction": 0.8,
            },
            reference={"kind": "analytic", "values": data["reference"]},
            environment=data["environment"],
            baseline_rule="ЛОМ этого же сценария и этой же серии",
            notes=[
                "N = 64/256/1024/4096/16384, для поля только 64/256/1024; B=128. "
                "Запас подбора 0,8; R=10 новых финальных запусков; лимит 20 с на каждую фазу каждого метода.",
                "Таймер включает подготовку метода, но не общие входные массивы, JIT, эталон и экспорт. "
                "WoS и сгущённая сетка имеют собственную погрешность аппроксимации.",
            ],
            rows=[],
        )
        baseline = {
            r["target_pct"]: r for r in data["rows"] if r["method"] == "lod_serial"
        }
        if config["n"] == 8 and config["problem"] == "sine":
            case["notes"].append(
                "На n=8 сеточное синусное решение имеет ошибку около 1,295%. Успех ЛОМ при 1% "
                "для этого запроса получен компенсацией сеточной и итерационной ошибок. Ужесточение "
                "внутреннего допуска не гарантирует уменьшения ошибки к аналитическому решению."
            )
        for raw in data["rows"]:
            selected = [
                t
                for t in data["trials"]
                if t["method"] == raw["method"]
                and (t["phase"], t["candidate"]) == (raw["phase"], raw["candidate"])
            ]
            base = baseline[raw["target_pct"]]
            status = raw["status_code"]
            speedup = (
                base["seconds"] / raw["seconds"]
                if status == base["status_code"] == "confirmed"
                else None
            )
            if status == "confirmed":
                assert len(selected) == plan["repeats"]
                assert all(t["status"] == "completed" for t in selected)
                assert raw["relative_error_pct"] <= raw["target_pct"]
            case["rows"].append(
                dict(
                    method=raw["method"],
                    variant=raw["method"],
                    name=sources.METHOD_NAMES[raw["method"]],
                    target=raw["target_pct"],
                    error_pct=raw["relative_error_pct"],
                    seconds=raw["seconds"],
                    budget=raw["budget"],
                    budget_kind=raw["budget_kind"],
                    status=status,
                    reason=raw["reason"],
                    repeats=raw["requested_repeats"],
                    recorded=raw["recorded_repeats"],
                    recorded_repeats=raw["recorded_repeats"],
                    completed=raw["completed_repeats"],
                    phase=raw["phase"],
                    speedup=speedup,
                    spent_seconds=raw["elapsed_seconds"],
                    elapsed_seconds=raw["elapsed_seconds"],
                    actual_walks=raw["total_walks"],
                    total_walks=raw["total_walks"],
                    phase_mean_seconds=raw["mean_seconds"],
                    transitions=raw["transitions"],
                    seconds_std=(
                        stdev(t["seconds"] for t in selected)
                        if len(selected) > 1
                        else None
                    ),
                    source=sources.location(root, path, "rows"),
                    raw=raw,
                )
            )
        cases.append(case)
    return cases


def summary_text(data):
    fresh = [c for c in data["cases"] if c["section"] == "stationary_current"]
    rows = [r for c in data["cases"] for r in c["rows"]]
    return [
        f"В одном файле {len(data['cases'])} сценариев и {len(rows)} строк целей. "
        "Для каждого метода показаны все пороги 1%, 5% и 10%, включая проигрыши и отказы.",
        "Выигрыш MC у ЛОМ есть в стационарной 3D-задаче с запросом одной точки. "
        "Он зависит от сетки, точности и объёма запроса; в полной нелинейной временной цепочке ЛОМ быстрее MC.",
        "Новые методы переноса частиц не меняют этот вывод: counts ускоряет обычный MC в измеренных "
        "полных историях, но проигрывает ЛОМ. Детерминированный перенос близок к ЛОМ; малые различия "
        "времени не доказывают устойчивого превосходства.",
        "Основная ошибка для точки: среднее по запускам 100·|u−u*|/|u*|. Это не RMSE и не ошибка "
        "усреднённого решения. Для современного запроса поля берётся средняя относительная L2-ошибка. "
        "Оба метода должны пройти один порог; их фактические ошибки могут различаться.",
        "Ускорение K = время ЛОМ / время метода вычисляется только внутри одной серии, при "
        "подтверждении цели обоими методами. K>1 — измеренный выигрыш, K<1 — проигрыш. "
        "Неудачи не заменяются нулями и не дают коэффициента ускорения.",
        "Свежая стационарная серия: 7 сценариев, 6 методов, R=10, commit "
        + fresh[0]["commit"][:7]
        + ". Подбор с запасом 0,8, затем независимый финал без перенастройки. "
        "Все расчёты времени на компьютере выполнялись последовательно, в один поток; аудит данных распараллелен.",
        "Архивы представлены отдельными разделами со своими версиями, R и протоколами. "
        "Старые RMS-результаты не переименованы в среднюю относительную ошибку. "
        "Между архивами не вычисляются ускорения; повторяющиеся задачи не считаются независимой статистикой.",
    ]


def mc_overview(data):
    lines = [
        "## Где обычный MC выигрывает и проигрывает ЛОМ сейчас",
        "",
        "Все строки ниже — новая стационарная 3D-серия. Ячейка: время MC / время ЛОМ, мс; K.",
        "",
        "| Сценарий | 1% | 5% | 10% |",
        "|---|---|---|---|",
    ]
    for case in data["cases"]:
        if case["section"] != "stationary_current":
            continue
        parts = []
        for target in (1, 5, 10):
            row = next(
                r for r in case["rows"] if r["method"] == "mc" and r["target"] == target
            )
            base = next(
                r
                for r in case["rows"]
                if r["method"] == "lod_serial" and r["target"] == target
            )
            if row["speedup"] is not None:
                parts.append(
                    f"{row['seconds']*1000:.3f} / {base['seconds']*1000:.3f}; K={row['speedup']:.2f}"
                )
            else:
                parts.append("Нет подтверждённой пары")
        lines.append(
            "| "
            + case["id"].split("/")[-1]
            + ": "
            + case["title"]
            + " | "
            + " | ".join(parts)
            + " |"
        )
    lines += [
        "",
        "У n=64 цель 1% ЛОМ ограничена временем подбора (20 с на фазу всех повторов), "
        "это не доказательство невозможности решить задачу ЛОМ. Для неё K не вычисляется. "
        "Новые попытки с увеличенными лимитами после просмотра результата не добавлялись.",
        "",
    ]
    return "\n".join(lines)


def instructions():
    return [
        "Для демонстрации: python main.py → Стационарный → 3D → Пуассон: синус в кубе → "
        "одна точка → центр → Исследование. Начните с n=16 или 32 и цели 5/10%; "
        "затем покажите 1% и всё поле, где выигрыш меняется или пропадает.",
        "Настройки новой серии: R=10, seed=42, максимальное N=16384, контрольная сетка "
        "4 для n=8/16, 8 для n=32, 16 для n=64, размер пакета 128. Для поля n=8 максимум N=1024. "
        "В интерфейсе список промежуточных бюджетов может быть плотнее; точное воспроизведение таблицы "
        "даёт python -m benchmarks.scenario_comparison --output runs/my-comparison. Каталог должен быть новым.",
        "Перенос количества частиц и массы подключён к квазилинейной теплопроводности 3D. "
        "В стационарной таблице mc_grouped означает пакеты отдельных траекторий, а не перенос количества. "
        "Неподдерживаемые сочетания не заполнены результатами другого решателя.",
        "В одной замороженной системе mass_transport переносит единичную вещественную массу "
        "из точки запроса. Для полной истории тот же линейный ответ для всех стартов вычисляется "
        "эквивалентной итерацией Якоби с начальным приближением. Оба варианта детерминированы; "
        "это не обычный MC и не одинаковая последовательность вычислительных операций.",
        "Синус стационарной задачи: u*=sin(πx)sin(πy)sin(πz), f=−3π²u*, "
        "все грани 1 рода. Линейное поле: u*=1+2x+3y+5z, f=0. "
        "Временная 3D-модель: c(u)=1+γu², k(u)=1+αu², источник F−σu³; "
        "параметры времени и границ приведены у каждого сценария.",
    ]


def make_pdf(data, path):
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import (
        PageBreak,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    fonts = Path("/System/Library/Fonts/Supplemental")
    pdfmetrics.registerFont(TTFont("Report", str(fonts / "Arial.ttf")))
    pdfmetrics.registerFont(TTFont("ReportBold", str(fonts / "Arial Bold.ttf")))
    pdfmetrics.registerFontFamily("Report", normal="Report", bold="ReportBold")
    body = ParagraphStyle(
        "body", fontName="Report", fontSize=10, leading=14, spaceAfter=8
    )
    small = ParagraphStyle("small", parent=body, fontSize=8, leading=10, spaceAfter=6)
    heading = ParagraphStyle(
        "heading",
        parent=body,
        fontName="ReportBold",
        fontSize=17,
        leading=21,
        spaceAfter=12,
    )
    cell_style = ParagraphStyle(
        "cell", parent=body, fontSize=8.5, leading=11, spaceAfter=0, alignment=TA_LEFT
    )
    story = []

    def para(text, style=body):
        return Paragraph(escape(str(text)), style)

    story.append(para("Где наши методы быстрее ЛОМ", heading))
    story.append(para("Сравнение при точности 1%, 5% и 10% • 26 ноября 2026", body))
    for text in summary_text(data):
        story.append(para(text))
    story.append(
        para(
            "Таблицы: зелёный — время меньше ЛОМ; красный — больше; серый — цель не подтверждена "
            "или нет сопоставимого ЛОМ. Это наблюдение данной серии, не статистическая гарантия. "
            "Единицы времени везде миллисекунды; N — бюджет на одну оценку, не сумма всех путей."
        )
    )
    story.append(PageBreak())
    story.append(para("Быстрый обзор: обычный MC против ЛОМ", heading))
    intro_rows = [
        [para(x, cell_style) for x in ("Сценарий", "Цель 1%", "Цель 5%", "Цель 10%")]
    ]
    for case in data["cases"]:
        if case["section"] != "stationary_current":
            continue
        cells = [para(case["id"].split("/")[-1] + "\n" + case["title"], cell_style)]
        for target in (1, 5, 10):
            row = next(
                r for r in case["rows"] if r["method"] == "mc" and r["target"] == target
            )
            base = next(
                r
                for r in case["rows"]
                if r["method"] == "lod_serial" and r["target"] == target
            )
            text = (
                f"MC {row['seconds']*1000:.3f} мс; ЛОМ {base['seconds']*1000:.3f} мс; K={row['speedup']:.2f}"
                if row["speedup"] is not None
                else "Нет подтверждённой пары; причина в подробной таблице"
            )
            cells.append(para(text, cell_style))
        intro_rows.append(cells)

    def table(rows):
        result = Table(
            rows, colWidths=[142, 124, 124, 124], repeatRows=1, hAlign="LEFT"
        )
        result.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e3eaf2")),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#bac4d0")),
                    ("LEFTPADDING", (0, 0), (-1, -1), 7),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                    ("TOPPADDING", (0, 0), (-1, -1), 5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ]
            )
        )
        return result

    story.append(table(intro_rows))
    story.append(Spacer(1, 10))
    story.append(
        para(
            "S01–S04: синус в центре n=8/16/32/64. S05: синус n=64 у границы. "
            "S06: линейное поле n=32. S07: всё внутреннее поле синуса n=8. "
            "У n=64 подбор ЛОМ для 1% ограничен временем фазы 20 с; это не предел применимости ЛОМ.",
            small,
        )
    )
    story.append(PageBreak())
    story.append(para("Как показать результат и читать таблицы", heading))
    for text in instructions():
        story.append(para(text))
    story.append(
        para(
            "Сначала идут новые стационарные результаты, затем новые переносы, полный временной расчёт "
            "с прежними шестью методами и чувствительность настроек. Исторические 2D/3D серии вынесены в конец. "
            "У каждого сценария записаны версия, число повторов, настройки и ссылки на исходные данные."
        )
    )
    for index, case in enumerate(data["cases"], 1):
        story.append(PageBreak())
        story.append(para(f"{index}. {case['title']}", heading))
        story.append(para(case["section_title"], body))
        config_text = sources.settings_text(case)
        config = case["settings"]["configuration"]
        metric_name = "L1" if "l1" in case["protocol"]["accuracy_metric"] else "L2"
        config_text += f"; средняя относительная {metric_name}-ошибка (для одной точки обе совпадают)"
        if "coarse_n" in config:
            config_text += f"; контрольная сетка={config['coarse_n']}"
        story.append(para(config_text, small))
        tag = "Историческая серия. " if case.get("historical") else ""
        story.append(
            para(
                tag + f"Версия {case['commit'][:12]}; R={case['protocol']['repeats']}; "
                f"эталон: {case['reference']['kind']}. {case['baseline_rule']}.",
                small,
            )
        )
        grouped = defaultdict(list)
        for row in case["rows"]:
            grouped[row["variant"]].append(row)
        groups = sorted(
            grouped.values(),
            key=lambda rs: (sources.ORDER.get(rs[0]["method"], 99), rs[0]["variant"]),
        )
        rows = [
            [para(x, cell_style) for x in ("Метод", "Цель 1%", "Цель 5%", "Цель 10%")]
        ]
        fills = []
        for row_index, group in enumerate(groups, 1):
            cells = [para(group[0]["name"], cell_style)]
            by_target = {r["target"]: r for r in group}
            for column, target in enumerate((1, 5, 10), 1):
                row = by_target.get(target)
                if row is None:
                    cells.append(para("Не измерено", cell_style))
                    continue
                budget = sources.budget_text(row["budget"], row.get("budget_kind", ""))
                error = (
                    "нет оценки"
                    if row.get("error_pct") is None
                    else f"ошибка {row['error_pct']:.4g}%"
                )
                if row["status"] == "confirmed":
                    content = f"<b>{row['seconds']*1000:.3f} мс</b><br/>{escape(error)}<br/>{escape(budget)}"
                    ratio = row.get("speedup")
                    content += "<br/>" + (
                        f"K={ratio:.3g}" if ratio is not None else "K не вычисляется"
                    )
                    shade = (
                        "#edf7ee"
                        if ratio and ratio > 1
                        else "#fff0ed" if ratio and ratio < 1 else "#f7f8fa"
                    )
                else:
                    label = sources.STATUS_NAMES.get(row["status"], row["status"])
                    diagnostic = row.get("diagnostic")
                    if row["phase"] == "not_started" and diagnostic:
                        error = f"подбор: ошибка {diagnostic['error_pct']:.4g}%"
                        budget = sources.budget_text(diagnostic["budget"])
                    phase_text = {
                        "pilot": "подбор",
                        "final": "финал",
                        "not_started": "финала нет",
                    }.get(row["phase"], row["phase"])
                    content = (
                        f"<b>{escape(label)}</b><br/>{escape(error)}<br/>{escape(budget)}"
                        f"<br/>{escape(phase_text)}: "
                        f"{row.get('completed',0)}/{row.get('recorded',0)} запусков"
                    )
                    shade = "#eef0f3"
                cells.append(Paragraph(content, cell_style))
                fills.append(
                    (
                        "BACKGROUND",
                        (column, row_index),
                        (column, row_index),
                        colors.HexColor(shade),
                    )
                )
            rows.append(cells)
        result = table(rows)
        result.setStyle(TableStyle(fills))
        story.append(result)
        story.append(Spacer(1, 8))
        failures = [r for r in case["rows"] if r["status"] != "confirmed"]
        reasons = list(dict.fromkeys(r["reason"] for r in failures))
        for reason in reasons:
            story.append(para("Причина неудачи: " + reason, small))
        unique_failures = {}
        for row in failures:
            key = (row["method"], row["phase"], str(row["budget"]), row["status"])
            unique_failures.setdefault(key, []).append(row)
        for same_rows in unique_failures.values():
            row = same_rows[0]
            targets_text = "/".join(f"{r['target']:g}" for r in same_rows)
            diagnostic = (
                row.get("diagnostic") if row["phase"] == "not_started" else None
            )
            evidence = diagnostic or row
            phase_text = (
                "лучший подбор, финал не запускался"
                if diagnostic
                else {
                    "pilot": "подбор",
                    "final": "финал",
                    "not_started": "финал не запускался",
                }.get(row["phase"], row["phase"])
            )
            if row["phase"] == "not_started" and not diagnostic:
                story.append(
                    para(
                        f"{row['name']}, {targets_text}%: финал не запускался; время достижения цели неизвестно.",
                        small,
                    )
                )
                continue
            total_walks = evidence.get("total_walks", evidence.get("actual_walks"))
            walks_text = (
                "не сохранены / неприменимы"
                if total_walks is None
                else str(total_walks)
            )
            spent = evidence.get("elapsed_seconds", evidence.get("spent_seconds"))
            time_text = "не измерено" if spent is None else f"{spent * 1000:.3f} мс"
            story.append(
                para(
                    f"{row['name']}, {targets_text}%: {phase_text}; "
                    f"запущено путей {walks_text}; время указанной серии {time_text}.",
                    small,
                )
            )
        for note in case.get("notes", []):
            if case.get("historical") and note.startswith(
                ("Исторические данные", "При выборе MC")
            ):
                continue
            story.append(para(note, small))
        if failures:
            story.append(
                para(
                    "Число запусков означает завершено/записано; ошибки pilot не являются финальным подтверждением. "
                    "Затраты неудачной серии не являются временем достижения цели. Несколько целей могут "
                    "ссылаться на одну серию, её затраты нельзя повторно суммировать.",
                    small,
                )
            )
        links = []
        for source_index, source in enumerate(case["source"], 1):
            url = REPO_URL + source["path"]
            label = Path(source["path"]).stem.split("--")[-1]
            links.append(
                f'<link href="{escape(url)}" color="#245c86">{escape(label)}</link>'
            )
        story.append(Paragraph("Исходные данные: " + " · ".join(links), small))
    story.append(PageBreak())
    story.append(para("Что эти результаты устанавливают", heading))
    for text in data.get("excluded", []):
        story.append(para(text))
    story.append(
        para(
            "Успешная цель означает среднюю ошибку данной конечной серии. "
            "Она не гарантирует точность каждого следующего случайного запуска. "
            "Бюджеты проверялись в заданном списке; глобально минимальное время среди всех настроек не найдено. "
            "Небольшие различия времени и многократный выбор удачного примера не являются доказательством устойчивого выигрыша."
        )
    )

    def footer(canvas, doc):
        canvas.setFont("Report", 8)
        canvas.setFillColor(colors.HexColor("#536273"))
        canvas.drawString(
            40, 24, "PDE • Сравнение методов при заданной точности • 26.11.2026"
        )
        canvas.drawRightString(A4[0] - 40, 24, str(doc.page))

    document = SimpleDocTemplate(
        str(path),
        pagesize=A4,
        leftMargin=40,
        rightMargin=40,
        topMargin=37,
        bottomMargin=42,
        title="Сравнение методов при точности 1%, 5%, 10%",
        author="PDE project",
    )
    bookmarks = []

    def bookmark(flowable):
        if isinstance(flowable, Paragraph) and flowable.style.name == "heading":
            key = f"section-{len(bookmarks)}"
            bookmarks.append(key)
            document.canv.bookmarkPage(key)
            document.canv.addOutlineEntry(flowable.getPlainText(), key, 0)

    document.afterFlowable = bookmark
    document.build(story, onFirstPage=footer, onLaterPages=footer)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--pdf", action="store_true")
    args = parser.parse_args()
    data = sources.build_existing(args.root)
    historical = historical_cases(args.root)
    order = {
        name: index
        for index, name in enumerate(
            (
                "stationary_current",
                "transport_frozen",
                "transport_full",
                "final_core",
                "conductivity",
                "coarse",
                "legacy_stationary",
                "legacy_stationary3d",
                "legacy_heat",
            )
        )
    }
    data["cases"] = fresh_cases(args.root) + data["cases"] + historical
    data["cases"].sort(key=lambda c: order.get(c["section"], 99))
    data["interpretation"] = summary_text(data)
    output = args.root / OUT
    sources.write_outputs(data, output / "comparison")
    text = (output / "comparison.md").read_text()
    position = text.index("\n## ")
    overview = (
        mc_overview(data)
        + "\n## Как воспроизвести\n\n"
        + "\n\n".join(instructions())
        + "\n"
    )
    text = text[:position] + "\n" + overview + text[position:]
    (output / "comparison.md").write_text(text)
    if args.pdf:
        make_pdf(data, output / "comparison.pdf")
    print(json.dumps(data["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
