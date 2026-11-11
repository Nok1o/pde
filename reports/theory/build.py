"""Build a paginated, linked Russian theory PDF with source-code navigation."""

import ast
import json
import subprocess
import sys
from pathlib import Path
from xml.sax.saxutils import escape

from PIL import Image as PILImage
from pypdf import PdfReader
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

sys.path.insert(0, str(Path(__file__).parent))
from content import PAGES, ex, h, p, src, table

ROOT = Path(__file__).resolve().parents[2]
ASSET = ROOT / "tmp/pdfs/theory/assets"
OUT = ROOT / "output/pdf/pde-theory-guide.pdf"
FONT = next(
    (ROOT / ".venv/lib").glob("python*/site-packages/matplotlib/mpl-data/fonts/ttf")
)
for name, filename in [("DV", "DejaVuSans.ttf"), ("DVB", "DejaVuSans-Bold.ttf")]:
    pdfmetrics.registerFont(TTFont(name, str(FONT / filename)))
pdfmetrics.registerFontFamily(
    "DV", normal="DV", bold="DVB", italic="DV", boldItalic="DVB"
)
BLUE = colors.HexColor("#194c70")
TEAL = colors.HexColor("#008080")
GRAY = colors.HexColor("#607487")
LIGHT = colors.HexColor("#edf4f8")
WIDTH = A4[0] - 96
ST = {
    "p": ParagraphStyle(
        "body",
        fontName="DV",
        fontSize=10.1,
        leading=14.5,
        spaceAfter=8,
        textColor=colors.HexColor("#243746"),
    ),
    "h": ParagraphStyle(
        "sub",
        fontName="DVB",
        fontSize=11.1,
        leading=15,
        spaceBefore=7,
        spaceAfter=6,
        textColor=TEAL,
        keepWithNext=True,
    ),
    "title": ParagraphStyle(
        "chapter",
        fontName="DVB",
        fontSize=19,
        leading=24,
        spaceAfter=13,
        textColor=BLUE,
        keepWithNext=True,
    ),
    "src": ParagraphStyle(
        "source",
        fontName="DV",
        fontSize=8,
        leading=11,
        spaceBefore=9,
        spaceAfter=6,
        textColor=GRAY,
    ),
    "caption": ParagraphStyle(
        "caption",
        fontName="DV",
        fontSize=8.5,
        leading=11.5,
        spaceAfter=10,
        textColor=GRAY,
    ),
    "cell": ParagraphStyle(
        "cell",
        fontName="DV",
        fontSize=8.7,
        leading=12,
        textColor=colors.HexColor("#243746"),
    ),
    "th": ParagraphStyle(
        "th", fontName="DVB", fontSize=8.7, leading=12, textColor=colors.white
    ),
    "toc": ParagraphStyle(
        "toc", fontName="DV", fontSize=9.7, leading=13.5, textColor=BLUE
    ),
    "cover": ParagraphStyle(
        "cover", fontName="DVB", fontSize=31, leading=38, spaceAfter=22, textColor=BLUE
    ),
}
ST["ex"] = ParagraphStyle(
    "example",
    parent=ST["p"],
    fontSize=9.7,
    leading=14,
    backColor=LIGHT,
    borderPadding=9,
    spaceBefore=6,
    spaceAfter=14,
)
EQUATIONS = json.loads((ASSET / "equations.json").read_text())
REV = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
BASE = f"https://github.com/Nok1o/pde/blob/{REV}/"


def code_link(file, function):
    source = subprocess.check_output(
        ["git", "show", f"{REV}:pdelab/{file}"], cwd=ROOT, text=True
    )
    tree = ast.parse(source)
    node = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == function
    )
    url = f"{BASE}pdelab/{file}#L{node.lineno}"
    return f'<link href="{url}" color="#087e8b">{escape(file)}:{node.lineno}<br/>{escape(function)}</link>'


CODE = [
    ("Сетка 2D и задания", "problem.py", "make_problem"),
    ("Независимый сеточный эталон", "problem.py", "reference_solution"),
    ("Прогонка в стандартных обозначениях", "solvers.py", "solve_tridiagonal"),
    ("Переиспользуемые факторы прогонки", "solvers.py", "_line_coefficients"),
    ("Стационарный ЛОМ 2D", "solvers.py", "_lod"),
    ("Обычный сеточный MC", "solvers.py", "_mc"),
    ("Хвосты по первым посещениям", "solvers.py", "_reuse"),
    ("Одна полная реализация слоёв", "solvers.py", "_layers_batch"),
    ("Ансамбль реализаций слоёв", "solvers.py", "_solve_layers"),
    ("Подготовка контрольного поля", "control.py", "prepare_control"),
    ("3D-параметр по спектральным модам", "solvers3d.py", "splitting_parameter3d"),
    ("ЛОМ в кубе", "solvers3d.py", "_lod3d"),
    ("MC с шестью соседями", "solvers3d.py", "_mc3d"),
    ("Общий WoS 2D/3D", "stationary_extensions.py", "_wos"),
    ("Пути на сгущённой сетке", "stationary_extensions.py", "_graded_paths"),
    (
        "Перенос данных и новый оператор",
        "stationary_extensions.py",
        "graded_boundary_operator",
    ),
    ("Смешанные условия и баланс", "boundary.py", "BoundaryOperator"),
    ("Смешанные грани в 3D", "boundary3d.py", "BoundaryOperator3D"),
    ("Переменная проводимость", "operator.py", "FrozenOperator"),
    ("Блочные направленные проходы", "frozen_solvers.py", "_directional_lines"),
    ("Линейный MC с контрольным полем", "frozen_solvers.py", "solve_frozen_mc"),
    ("Факторы неявного временного слоя", "heat.py", "_factors"),
    ("История линейного тепла", "heat.py", "solve_heat"),
    ("Путь назад в пространстве-времени", "heat_mc.py", "_paths"),
    ("Picard, релаксация и проверка", "nonlinear.py", "picard_layer"),
    ("Независимый Jacobian Newton", "nonlinear.py", "newton_system"),
    ("История нелинейного решения", "nonlinear.py", "solve_nonlinear"),
    ("Текущая метрика исследования", "study_policy.py", "mean_relative_error"),
    ("Пилотный запас и контрольная серия", "study_policy.py", "search_budgets"),
]
EXTRA = []
for chunk_index in range(2):
    subset = CODE[chunk_index * 15 : (chunk_index + 1) * 15]
    EXTRA.append(
        (
            f"Карта кода - {chunk_index + 1}/2",
            (
                p(
                    "Ссылки ведут на зафиксированную версию GitHub. Так формулы можно сопоставить с алгоритмом, даже если основная ветка позже изменится."
                ),
                table(
                    ["Что искать", "Файл, строка и функция"],
                    [[name, code_link(file, fn)] for name, file, fn in subset],
                    [220, WIDTH - 220],
                ),
                src(
                    f"Ревизия {REV}. В документе разобраны численные ядра и протокол исследования этой версии. Обозначения документа согласованы с лекциями; исходные API-имена перечислены в главе 2. Ссылки и номера строк относятся к этой зафиксированной ревизии ядер."
                ),
            ),
        )
    )
EXTRA.append(
    (
        "Источники и как пользоваться этим конспектом",
        (
            h("Первоисточник реализации"),
            p(
                f'<link href="https://github.com/Nok1o/pde/tree/{REV}" color="#008080">Nok1o/pde, ревизия {REV[:7]}</link>. Основные выводы получены непосредственно из дискретных уравнений и сверены с функциями из карты кода. Дата подготовки: 11 ноября 2026.'
            ),
            h("Обозначения преподавателя"),
            p(
                "«Мир Градова.pdf»: лекции 7–8 (с. 38–49), 9–10 (с. 55–61), 14–15 (с. 92–101). Согласование обозначений и сохранённых имён API приведено в первых двух главах. Текст лекций не включён в репозиторий."
            ),
            h("Дополнительные источники"),
            p(
                '[1] <link href="https://mfem.org/fem_bc/" color="#008080">MFEM: Boundary Conditions</link>. Обозначения граничных условий 1, 2 и 3 рода. Проект использует собственную разностную схему, не FEM-дискретизацию MFEM.'
            ),
            p(
                '[2] J. S. Dokken. <link href="https://jsdokken.com/dolfinx-tutorial/v0.10/chapter2/nonlinpoisson_code.html" color="#008080">FEniCSx tutorial: Nonlinear Poisson, Implementation</link>. Изготовленные задачи с проводимостью 1+u². Наши разностные ядра и эталоны описаны отдельно.'
            ),
            p(
                '[3] M. E. Muller. <link href="https://doi.org/10.1214/aoms/1177728169" color="#008080">Исходная работа о непрерывных методах Монте-Карло</link>. Annals of Mathematical Statistics, 27(3), 569-589, 1956. Историческая ссылка на WoS; формулы объёмных весов в этом конспекте выведены явно.'
            ),
            p(
                '[4] K. K. Sabelfeld. <link href="https://ideas.repec.org/a/bpj/mcmeap/v25y2019i1p85-96n5.html" color="#008080">A global random walk on spheres algorithm for transient heat equation and some extensions</link>. Monte Carlo Methods and Applications, 25(1), 85-96, 2019. DOI: 10.1515/mcma-2019-2032. Сопоставление сделано по доступной аннотации, не по полному тексту.'
            ),
            p(
                '[5] P. Henry-Labordère, N. Oudjane, X. Tan, N. Touzi, X. Warin. <link href="https://arxiv.org/abs/1603.01727" color="#008080">Branching diffusion representation of semilinear PDEs and Monte Carlo approximation</link>, arXiv:1603.01727, 2016. Отдельное направление для полулинейных задач, не реализованный в проекте метод.'
            ),
            h("Маршрут чтения"),
            p(
                "Для первого понимания: сетка → граничные условия → трафарет → маленькая задача → прогонка → ЛОМ → обычный MC. Затем улучшения и 3D. После этого смешанные границы, время, Picard и Newton. Главы об ошибках и протоколе сравнения нужны перед любыми выводами о выигрыше."
            ),
            ex(
                "Файл объясняет математику текущей программы и её ограничения. Он не заменяет протокол измерений: численные примеры здесь учебные, а утверждения о скорости требуют сохранённых независимых экспериментальных серий."
            ),
        ),
    )
)
ALL = PAGES + EXTRA


class Guide(SimpleDocTemplate):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.locations = {}
        self.current_title = "Теория численных методов"

    def afterFlowable(self, flowable):
        if hasattr(flowable, "chapter_key"):
            key = flowable.chapter_key
            self.locations[key] = self.page
            self.current_title = flowable.getPlainText()
            self.canv.bookmarkPage(key)
            self.canv.addOutlineEntry(self.current_title, key, level=0, closed=False)


def footer(canvas, doc):
    canvas.saveState()
    w, h = A4
    canvas.setStrokeColor(colors.HexColor("#d5e0e8"))
    canvas.line(48, 40, w - 48, 40)
    canvas.setFont("DV", 8)
    canvas.setFillColor(GRAY)
    canvas.drawString(48, 27, "PDE / ЛОМ и Монте-Карло / теория")
    canvas.drawRightString(w - 48, 27, str(doc.page))
    if doc.page > 1:
        canvas.setFont("DV", 7.5)
        canvas.drawString(48, h - 29, "Nok1o/pde  ·  11.11.2026  ·  " + REV[:7])
    canvas.restoreState()


def para(text, style="p"):
    return Paragraph(
        text.replace("—", "-").replace("–", "-").replace("‑", "-"), ST[style]
    )


def make_table(headers, rows, widths=None):
    n = len(headers)
    if widths is None:
        widths = [WIDTH / n] * n
    data = [[para(x, "th") for x in headers]] + [
        [para(str(x), "cell") for x in row] for row in rows
    ]
    tab = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
    padding = 4 if widths == [220, WIDTH - 220] else 7
    tab.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), BLUE),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [LIGHT, colors.white]),
                ("LINEBELOW", (0, 0), (-1, 0), 0.5, BLUE),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), padding),
                ("BOTTOMPADDING", (0, 0), (-1, -1), padding),
            ]
        )
    )
    return [tab, Spacer(1, 10)]


def image(path, max_height=200, equation=False):
    with PILImage.open(path) as pic:
        width, height = pic.size
    if equation:
        scale = min(72 / 240, (WIDTH - 10) / width)
    else:
        scale = min(WIDTH / width, max_height / height)
    item = Image(str(path), width=width * scale, height=height * scale)
    item.hAlign = "CENTER"
    return item


def blocks(items):
    result = []
    for block in items:
        kind = block[0]
        if kind in ["p", "h", "ex", "src"]:
            result.append(para(block[1], kind))
        elif kind == "e":
            result.extend(
                [
                    Spacer(1, 2),
                    image(ASSET / EQUATIONS[block[1]], equation=True),
                    Spacer(1, 9),
                ]
            )
        elif kind == "fig":
            result.append(
                KeepTogether(
                    [
                        image(ASSET / (block[1] + ".png"), block[3]),
                        Spacer(1, 4),
                        para(block[2], "caption"),
                    ]
                )
            )
        elif kind == "table":
            result.extend(make_table(block[1], block[2], block[3]))
    return result


def story(locations):
    flow = [
        Spacer(1, 40),
        para("Численные методы<br/>для уравнений<br/>теплопроводности", "cover"),
        para("Полный теоретический конспект проекта PDE", "h"),
        Spacer(1, 12),
        para(
            "ЛОМ и прогонка · Монте-Карло и улучшения<br/>2D и 3D · Граничные условия 1, 2 и 3 рода<br/>Физическое время · Квазилинейность · Честное сравнение"
        ),
        Spacer(1, 23),
        image(ASSET / "sweeps.png", 150),
        Spacer(1, 20),
        para(
            "От локального уравнения в одном узле до полной нелинейной истории: выводы формул, пошаговые алгоритмы, числовые примеры и ссылки на реализацию.",
            "ex",
        ),
        Spacer(1, 12),
        para(
            f"{len(PAGES)} тематических раздела • {len(EQUATIONS)} формул • 10 схем<br/>Состояние численных ядер: {REV[:7]}<br/>11 ноября 2026",
            "src",
        ),
        PageBreak(),
    ]
    for part in range((len(ALL) + 20) // 21):
        flow.append(
            para("Оглавление" if part == 0 else "Оглавление / продолжение", "title")
        )
        if part == 0:
            flow.append(
                para(
                    "Названия разделов и номера страниц кликабельны. В приложении - переходы к конкретным функциям GitHub.",
                    "caption",
                )
            )
        subset = list(enumerate(ALL))[part * 21 : (part + 1) * 21]
        data = []
        for idx, (title, _) in subset:
            key = f"chapter-{idx + 1}"
            number = (
                f"{idx + 1:02d}"
                if idx < len(PAGES)
                else "П" + str(idx - len(PAGES) + 1)
            )
            data.append(
                [
                    para(
                        f'<link href="#{key}">{number}. {escape(title)}</link>', "toc"
                    ),
                    para(
                        f'<link href="#{key}">{locations.get(key, idx + 4)}</link>',
                        "toc",
                    ),
                ]
            )
        tab = Table(data, colWidths=[WIDTH - 28, 28])
        tab.setStyle(
            TableStyle(
                [
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 0),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 2),
                    ("TOPPADDING", (0, 0), (-1, -1), 5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                    ("LINEBELOW", (0, 0), (-1, -1), 0.3, colors.HexColor("#dce5ec")),
                ]
            )
        )
        flow.append(tab)
        flow.append(PageBreak())
    for idx, (title, items) in enumerate(ALL):
        number = (
            f"{idx + 1:02d}"
            if idx < len(PAGES)
            else "Приложение " + str(idx - len(PAGES) + 1)
        )
        flow.append(para(number, "h"))
        heading = para(title, "title")
        heading.chapter_key = f"chapter-{idx + 1}"
        flow.append(heading)
        flow.extend(blocks(items))
        if idx < len(ALL) - 1:
            flow.append(PageBreak())
    return flow


locations = {}
for build in range(2):
    doc = Guide(
        str(OUT),
        pagesize=A4,
        rightMargin=48,
        leftMargin=48,
        topMargin=50,
        bottomMargin=53,
        title="Теория PDE: ЛОМ, Монте-Карло, границы и квазилинейность",
        author="Проект Nok1o/pde",
        subject="Математические выводы, алгоритмы и учебные примеры",
    )
    doc.build(story(locations), onFirstPage=footer, onLaterPages=footer)
    locations = doc.locations
(ASSET.parent / "page-map.json").write_text(json.dumps(locations, indent=2))
reader = PdfReader(OUT)
print(f"PDF: {OUT}\nPages: {len(reader.pages)}; bytes: {OUT.stat().st_size}")
for i, (title, _) in enumerate(ALL):
    start = locations[f"chapter-{i + 1}"]
    end = locations.get(f"chapter-{i + 2}", len(reader.pages) + 1)
    if end - start != 1:
        print(f"Overflow: chapter {i + 1}, pages {start}-{end - 1}: {title}")
