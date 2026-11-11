"""Publish a complete recheck, retaining original experiments as historical data."""

import io
import json
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np
from pypdf import PdfReader, PdfWriter
from reportlab.lib import colors
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

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "benchmarks/results/audit-recheck-20261111"
OLD = ROOT / "benchmarks/results/project-audit"
OUT = ROOT / "output/pdf/pde-audit-recheck.pdf"
protocol = json.loads((DATA / "protocol.json").read_text())
archive = json.loads((DATA / "archive-check.json").read_text())
loaded = [
    json.loads((DATA / (case["id"] + ".json")).read_text())
    for case in protocol["cases"]
]
focus = json.loads((DATA / "control-focus.json").read_text())
focus_protocol = json.loads((DATA / "focus-protocol.json").read_text())
focus += json.loads((DATA / "user-case/control-focus.json").read_text())
focus.sort(key=lambda d: (d["n"], d["coarse_n"]))
font = next(
    (ROOT / ".venv/lib").glob("python*/site-packages/matplotlib/mpl-data/fonts/ttf")
)
for name, file in [("DV", "DejaVuSans.ttf"), ("DVB", "DejaVuSans-Bold.ttf")]:
    pdfmetrics.registerFont(TTFont(name, str(font / file)))
pdfmetrics.registerFontFamily(
    "DV", normal="DV", bold="DVB", italic="DV", boldItalic="DVB"
)
BLUE = colors.HexColor("#194c70")
TEAL = colors.HexColor("#008080")
W = A4[0] - 92
styles = {
    "body": ParagraphStyle(
        "body", fontName="DV", fontSize=10, leading=14, spaceAfter=8
    ),
    "title": ParagraphStyle(
        "title", fontName="DVB", fontSize=19, leading=24, spaceAfter=14, textColor=BLUE
    ),
    "sub": ParagraphStyle(
        "sub",
        fontName="DVB",
        fontSize=11,
        leading=15,
        spaceBefore=10,
        spaceAfter=7,
        textColor=TEAL,
    ),
    "small": ParagraphStyle(
        "small",
        fontName="DV",
        fontSize=8,
        leading=11,
        spaceAfter=6,
        textColor=colors.HexColor("#526575"),
    ),
    "cell": ParagraphStyle("cell", fontName="DV", fontSize=7.8, leading=10.6),
    "head": ParagraphStyle(
        "head", fontName="DVB", fontSize=7.8, leading=10.6, textColor=colors.white
    ),
}
flow = []


def p(s, style="body"):
    flow.append(Paragraph(s, styles[style]))


def title(s):
    if flow:
        flow.append(PageBreak())
    p(s, "title")


def table(headers, rows, widths=None):
    data = [[Paragraph(escape(str(x)), styles["head"]) for x in headers]]
    data += [[Paragraph(escape(str(x)), styles["cell"]) for x in row] for row in rows]
    t = Table(data, colWidths=widths or [W / len(headers)] * len(headers), repeatRows=1)
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), BLUE),
                (
                    "ROWBACKGROUNDS",
                    (0, 1),
                    (-1, -1),
                    [colors.HexColor("#edf4f8"), colors.white],
                ),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    flow.extend([t, Spacer(1, 10)])


def ms(x):
    return "-" if x is None else f"{x*1000:.3f}"


def number(x):
    return "-" if x is None else f"{x:.3f}"


names = {
    "lod": "ЛОМ",
    "mc": "Обычный MC",
    "mc_control": "Контрольный MC",
    "reuse": "Хвосты",
    "layers": "Слои",
    "wos": "WoS",
    "graded": "Сгущённая сетка",
}
rows = [r for d in loaded for r in d["comparisons"]]
lookup = {(r["case"], r["method"], r["target_pct"]): r for r in rows}
replayed = sum(len(d["replay"]) for d in loaded)
fresh = sum(len(d["trials"]) for d in loaded)
issues = [r for d in loaded for r in d["replay"] if r["status"] != "ok"]
maxdiff = max(d["replay_max_squared_error_difference"] for d in loaded)

p("Перепроверка таблиц сравнения", "title")
p("ЛОМ и Монте-Карло • 11 ноября 2026", "sub")
p(
    "Повод: при исследовании синуса в центре на сетке из 64 интервалов контрольный MC у пользователя оказался медленнее ЛОМ. Здесь отдельно проверены старые результаты, новый протокол ошибки и стоимость контрольной сетки."
)
p("Главное различие параметров", "sub")
table(
    ["Параметр", "Смысл"],
    [
        ["n=64", "64 интервала, 65 узлов по оси; центр (32,32)"],
        ["N=64", "64 случайных пути из каждой стартовой точки"],
        ["n_c=16", "16 интервалов грубой контрольной сетки; отдельный параметр"],
    ],
    [90, W - 90],
)
p(
    "Пользователь уточнил контрольную сетку: 6 интервалов. Проигрыш воспроизведён для 64/6. Ниже использованы seed=42 и пять усреднений; эти два значения для пользовательского запуска не подтверждены, поэтому приведены явно."
)
p("Проигрыш при 64/6 воспроизведён", "sub")
summary = []
for data in focus:
    if data["n"] != 64:
        continue
    pair = {r["method"]: r for r in data["ui_rows"] if r["target_pct"] == 1}
    a, b = pair["lod_serial"], pair["mc_control"]
    ratio = a["seconds"] / b["seconds"] if a["seconds"] and b["seconds"] else None
    summary.append(
        [
            64,
            data["coarse_n"],
            ms(a["seconds"]),
            ms(b["seconds"]),
            number(b["relative_error_pct"]),
            b["budget"],
            number(ratio),
        ]
    )
table(
    ["n", "n_c", "ЛОМ, мс", "MC, мс", "Ошибка MC, %", "Пути MC", "Tлом/Tmc"],
    summary,
    [30, 30, 65, 65, 75, 75, W - 340],
)
p(
    "При контрольной сетке 6 метод выбрал 1024 пути и оказался медленнее ЛОМ. При контрольной сетке 16 выбрано 256 путей, и MC быстрее. Причина - разное качество приближения и разброс остаточной поправки. Это разные конфигурации одного метода."
)
p(
    "Оба метода сравниваются с порогом 1%, а не с тождественными фактическими ошибками. Ошибка ЛОМ здесь 0.941%. Отношение Tлом/Tmc больше 1 означает преимущество MC. Размер контрольной сетки должен обязательно сопровождать вывод о выигрыше.",
    "small",
)

title("Что именно проверено")
table(
    ["Проверка", "Результат"],
    [
        [
            "Арифметика старых таблиц",
            f"{archive['checked_rows']} строк; обнаружено несоответствий: {len(archive['issues'])}",
        ],
        [
            "Повтор старой контрольной серии",
            f"{replayed} запусков с теми же параметрами и seed; неуспешных: {len(issues)}",
        ],
        [
            "Совпадение численных результатов",
            f"Максимальное отличие нормированных квадратов ошибок: {maxdiff:.3g}",
        ],
        [
            "Новая серия",
            f'{fresh} запусков, включая пилотные; {protocol["validation"]} независимых контрольных повторов кандидата',
        ],
        [
            "Постановки",
            f"{len(loaded)}: все 17 прежних и дополнительные центры n=24,48",
        ],
        [
            "Дополнительный разбор контроля",
            f"{len(focus)} сочетаний основной и контрольной сеток; фиксированные 64 пути и вызов протокола UI",
        ],
    ],
    [190, W - 190],
)
p("Что было в старом отчёте", "sub")
p(
    "Старая таблица использовала относительную RMS по отдельным запускам. Пример выигрыша 4.89× относился к синусу, n=64, центру, контрольной сетке n_c=16, N=64 путям и цели 1%. Это не утверждение о любой сетке, любой подготовке или каждом отдельном запуске. В старых таблицах малые сетки уже могли давать проигрыш контрольного MC."
)
p("Что изменено в новой проверке", "sub")
p(
    "Текущая метрика в одной точке: среднее по повторам величины 100·|u_r-u*|/|u*|. Для участка/поля: средняя относительная L1-ошибка всего запроса. Пилотный порог MC равен половине цели, для ЛОМ - полной цели. Финальные seed независимы. Провал контрольной серии не приводит к новому подбору по её ответам."
)
p(
    "Размер контрольной сетки по умолчанию max(4,n//4). Для каждого нового запуска она строится заново; время подготовки включено. Все остальные методы также включают свою подготовку. Общая дискретизация исходных данных, независимый эталон и JIT исключены из времени решателя."
)
p("Границы вывода", "sub")
p(
    "Уточнённая серия использует более плотный набор допусков ЛОМ, включая 0.15-0.9, и прежний набор MC-бюджетов 64/256/1024/4096/16384 (для участка/поля до 1024). Выбирается самый быстрый допустимый пилотный кандидат. Рабочая версия UI берёт бюджеты из удвоений, требует три успешные пилотные серии и допускает повторную попытку на большем бюджете после провала проверки. Поэтому отдельный точный вызов UI-протокола приведён далее, а не выдаётся за ту же процедуру."
)
p(
    "Интервалы - исследовательский bootstrap по 50 повторам, без поправки на множественные сравнения. «Подтверждено» означает прохождение этой контрольной серии, а не гарантию для всех запусков. Время также зависит от компьютера и нагрузки. Сохранились все кандидаты и отказы."
)

title("Размер контрольной сетки: цена подготовки")
p(
    "Ниже N=64 пути фиксировано, ЛОМ использует tol=0.01. По 50 повторов в случайном порядке. Ошибка - средняя относительная ошибка одного запуска. Здесь приведена диагностика фиксированных параметров, а не автоматически найденные бюджеты равной точности."
)
items = []
for data in focus:
    groups = {
        m: [r for r in data["fixed_N64"] if r["method"] == m]
        for m in ["lod_serial", "mc_control"]
    }
    a, b = groups["lod_serial"], groups["mc_control"]

    def av(group, key):
        return float(np.mean([r[key] for r in group]))

    items.append(
        [
            data["n"],
            data["coarse_n"],
            ms(av(a, "seconds")),
            ms(av(b, "seconds")),
            ms(av(b, "preparation_seconds")),
            number(av(b, "error_pct")),
            "да" if av(b, "error_pct") <= 1 else "нет",
        ]
    )
table(
    ["n", "n_c", "ЛОМ, мс", "MC, мс", "Подгот., мс", "Ошибка MC, %", "MC ≤1%"],
    items,
    [30, 30, 70, 70, 80, 80, W - 360],
)
p(
    "Если контрольная сетка равна основной, подготовка сама решает полную разреженную задачу. Поправка почти нулевая, но вычислительная работа не бесплатна. Слишком грубое контрольное поле, наоборот, может оставлять большой разброс поправки. Сравнивать разные n_c как одну настройку нельзя."
)

title("Точное правило интерфейса: синус, центр, 1%")
p(
    "Использованы функции _measure_budget и search_budgets из текущего интерфейса: seed=42, пять усреднений, максимум 16384 пути, отдельный лимит каждой фазы 600 с. Подбор идёт по удвоениям. Протокол stable_pilot_adaptive_retry_v4 требует три пилотные серии; после провала проверки может продолжить поиск. Поэтому эта таблица отделена от аудиторской серии с одной фиксированной независимой проверкой. Снимки использованного кода приложены."
)
items = []
status_names = {
    "confirmed": "достигнута",
    "validation_failed": "проверка не прошла",
    "not_reached": "нет запаса",
    "validation_incomplete": "проверка не завершена",
    "timeout": "лимит",
    "solver_failed": "сбой",
}
for data in focus:
    r = {x["method"]: x for x in data["ui_rows"] if x["target_pct"] == 1}
    a, b = r["lod_serial"], r["mc_control"]
    items.append(
        [
            data["n"],
            data["coarse_n"],
            ms(a["seconds"]),
            b["budget"],
            ms(b["seconds"]),
            number(b["relative_error_pct"]),
            status_names.get(b["status_code"], b["status"]),
        ]
    )
table(
    ["n", "n_c", "ЛОМ, мс", "Пути MC", "MC, мс", "Ошибка, %", "Статус MC"],
    items,
    [26, 26, 65, 55, 65, 65, W - 302],
)
p(
    "Пустое время означает отсутствие подтверждённого достижения 1%, а не нулевую стоимость. Пять повторов дают шумную оценку: другой seed или больше повторов могут выбрать другой бюджет. Для проверки пользовательского случая важно сравнить все настройки, а не только основную сетку."
)

old_wins = []
for data in loaded:
    key = data["case"]["id"]
    old_file = OLD / (key + "-refined.json")
    if not old_file.exists():
        continue
    original = json.loads(old_file.read_text())

    def replay_mean(row):
        ts = [
            r["seconds"]
            for r in data["replay"]
            if r["status"] == "ok"
            and r["method"] == row["method"]
            and r["parameters"] == row["parameters"]
        ]
        return float(np.mean(ts))

    for row in original["comparisons"]:
        if not row["robust_win"]:
            continue
        base = next(
            r
            for r in original["comparisons"]
            if r["method"] == "lod" and r["target_pct"] == row["target_pct"]
        )
        ratio = replay_mean(base) / replay_mean(row)
        freshrow = lookup[key, row["method"], row["target_pct"]]
        old_wins.append(
            [
                key,
                names[row["method"]],
                int(row["target_pct"]),
                number(row["speedup"]),
                number(ratio),
                number(freshrow.get("speedup")),
                "да" if freshrow["repeat_win"] else "нет",
            ]
        )
title("Все прежние отмеченные выигрыши: повтор")
p(
    "Столбец «повтор RMS» использует старые параметры и seed: численная точность воспроизведена, время измерено заново. Столбец «новая серия» относится к новой метрике, запасу и независимому выбору кандидата, поэтому это отдельное исследование."
)
table(
    [
        "Случай",
        "Метод",
        "Цель,%",
        "Старое ×",
        "Повтор RMS ×",
        "Новая серия ×",
        "Новый запас*",
    ],
    old_wins,
    [40, 100, 40, 60, 75, 80, W - 395],
)
p(
    "* «Да»: обе оценки проходят порог; верхняя 95%-граница ошибки MC не выше цели, нижняя граница отношения времён выше 1. Это исследовательский критерий, не поправка на все сравнения. Несколько целей могут использовать один и тот же кандидат и не являются независимыми победами.",
    "small",
)

for data in loaded:
    case = data["case"]
    title(
        f"{case['id']}. Полная таблица - {case['family']}, {case.get('problem','нелинейность')}"
    )
    coords = case.get(
        "point", [case["n"] // 2] * (3 if case["family"] in ("3d", "mixed3d") else 2)
    )
    details = f"Сетка n={case['n']}; запрос {case.get('mode','point')}; индексы точки/центра {coords}."
    if "mc_control" in case["methods"]:
        details += f" Контрольная сетка n_c={max(4,case['n']//4)}."
    if "dt" in case:
        details += f" Δt={case['dt']}, T={case['T']}."
    if "alpha" in case:
        details += f" α={case['alpha']}, релаксация 0.7."
    if case["family"].startswith("mixed"):
        kinds = case.get(
            "kinds",
            ["robin"] + ["dirichlet"] * (5 if case["family"] == "mixed3d" else 3),
        )
        details += " Границы: " + ", ".join(kinds) + ". κ=1 для Робина."
    p(details)
    p(
        "Время одного полного запуска, мс. Звёздочка: пройден дополнительный критерий запаса ошибки и интервала скорости. Значения времени при провале проверки показаны для диагностики и не означают достижения цели.",
        "small",
    )
    cells = []
    for row in sorted(
        data["comparisons"],
        key=lambda r: (r["target_pct"], case["methods"].index(r["method"])),
    ):
        params = row["parameters"]
        shown_error = row["error_pct"]
        shown_seconds = row["seconds"]
        pilot_note = ""
        if params is None:
            attempted = [
                t
                for t in data["trials"]
                if t["phase"] == 0 and t["method"] == row["method"]
            ]
            last_index = max(t["candidate"] for t in attempted)
            last = [t for t in attempted if t["candidate"] == last_index]
            params = last[0]["parameters"]
            if all(t["status"] == "ok" for t in last):
                shown_error = float(np.mean([t["relative_error_pct"] for t in last]))
                shown_seconds = float(np.mean([t["seconds"] for t in last]))
                pilot_note = " †"
        budget = (
            "-"
            if params is None
            else ", ".join(f"{k}={v:g}" for k, v in params.items())
        )
        status = {
            "confirmed": "достигнута",
            "no_pilot_margin": "нет запаса",
            "validation_failed": "проверка не прошла",
            "solver_failed": "сбой",
        }[row["status"]]
        if row["repeat_win"]:
            status += " *"
        cells.append(
            [
                int(row["target_pct"]),
                names[row["method"]],
                budget,
                number(shown_error) + pilot_note,
                ms(shown_seconds) + pilot_note,
                number(row.get("speedup")),
                status,
            ]
        )
    table(
        ["Цель,%", "Метод", "Параметры", "Ошибка,%", "Время,мс", "Tлом/Tmc", "Статус"],
        cells,
        [36, 88, 85, 59, 60, 62, W - 390],
    )
    p(
        "«Нет запаса» означает, что в объявленной пилотной сетке настроек не найден кандидат до половины цели MC (или полного порога ЛОМ). Это не доказательство невозможности достичь цели при других бюджетах. † Показаны параметры, ошибка и время последнего пилотного кандидата, не контрольной серии. Все испытанные настройки и отказы сохранены во вложенных JSON.",
        "small",
    )

title("Данные и воспроизведение")
p(
    f"Общая аудиторская серия: ревизия {protocol['revision']}, метрика {protocol['metric']}. Фокусный вызов UI выполнялся уже с локальными изменениями политики подбора: {focus_protocol['selection_policy']}. Хеши и снимки этих файлов сохранены; во время каждого фокусного запуска файлы не менялись. Уточнённый случай 64/6 сохранён в user-case; между фокусными сериями изменился только текст ACCURACY_DESCRIPTION, алгоритм подбора остался тем же."
)
p(
    "К этому PDF приложен архив audit-recheck-evidence.zip: протокол, все контрольные и пилотные запуски, повтор старых seed, полные таблицы, настройки фокусного эксперимента и скрипты. Исходные старые измерения не перезаписывались."
)
p(
    "Повтор полной серии: python -m benchmarks.recheck_audit --output /tmp/pde-audit-fresh. Пользовательский случай: python -m benchmarks.recheck_control_focus --output /tmp/pde-control-64-6 --only 64 6. Запускать последовательно, без конкурирующих измерений. Для нового измерения использовать новую папку; существующие результаты сохраняются. Точный алгоритм фокусной серии зафиксирован в приложенных снимках кода."
)
p("Как сравнить со своим экраном", "sub")
p(
    "Сверить функцию, число интервалов основной и контрольной сеток, координаты, число усреднений, seed, выбранный бюджет и фактическую ошибку. Сравнение обычного запуска с N путями и исследование достижения 1% отвечают на разные вопросы."
)
p("Что исправлено в интерпретации", "sub")
p(
    "64 интервала сетки не означают 64 пути. Выигрыш на одной конфигурации не относится ко всем размерам контрольного поля. Старые RMS-таблицы нельзя считать таблицами новой средней относительной ошибки. Быстрое время без подтверждённой точности не отмечается победой."
)


def footer(canvas, doc):
    canvas.setFont("DV", 8)
    canvas.setFillColor(colors.HexColor("#607487"))
    canvas.drawString(46, 27, "PDE / перепроверка измерений / 11.11.2026")
    canvas.drawRightString(A4[0] - 46, 27, str(doc.page))


OUT.parent.mkdir(parents=True, exist_ok=True)
SimpleDocTemplate(
    str(OUT),
    pagesize=A4,
    leftMargin=46,
    rightMargin=46,
    topMargin=42,
    bottomMargin=48,
    title="Перепроверка таблиц ЛОМ и Монте-Карло",
    author="Проект Nok1o/pde",
).build(flow, onFirstPage=footer, onLaterPages=footer)
archive_stream = io.BytesIO()
with zipfile.ZipFile(archive_stream, "w", zipfile.ZIP_DEFLATED) as z:
    for file in sorted(DATA.rglob("*")):
        if file.is_file():
            z.write(file, "data/" + str(file.relative_to(DATA)))
    for file in [
        "benchmarks/recheck_audit.py",
        "benchmarks/recheck_control_focus.py",
        "benchmarks/project_audit.py",
        "benchmarks/project_audit_refine.py",
    ]:
        z.write(ROOT / file, file)
    for file in sorted(OLD.glob("*.json")):
        if file.stem.startswith(("S", "H", "Q")) or file.name in (
            "protocol.json",
            "refinement-protocol.json",
        ):
            z.write(file, "historical/" + file.name)
reader = PdfReader(OUT)
writer = PdfWriter()
writer.clone_document_from_reader(reader)
writer.add_attachment("audit-recheck-evidence.zip", archive_stream.getvalue())
with OUT.open("wb") as f:
    writer.write(f)
print(
    OUT,
    "pages",
    len(reader.pages),
    "bytes",
    OUT.stat().st_size,
    "new wins",
    sum(r["repeat_win"] for r in rows),
)
