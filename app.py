"""Run: python -m streamlit run app.py"""

import io
import zipfile
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from pdelab.experiment import (
    METRICS_POLICY,
    Config,
    benchmark,
    refresh_comparison,
    run_experiment,
)
from pdelab.problem import PRESETS
from pdelab.problem_info import render_problem_info
from pdelab.reference import REFERENCE_POLICY
from pdelab.research import export_study
from pdelab.research_ui import render_research_tab
from pdelab.solvers import LOD_STOPPING_RULE, METHODS
from pdelab.views import (
    benchmark_plot,
    convergence_plot,
    export,
    interactive_field,
    layer_plot,
    residual_plot,
    reuse_plot,
    surface_plot,
    trajectory_plot,
)

st.set_page_config(page_title="PDE Lab · Сеточный метод и Монте-Карло", layout="wide")
st.title("Сеточный метод и Монте-Карло")
with st.sidebar:
    st.header("Задача")
    problem = st.selectbox(
        "Уравнение и условия", list(PRESETS), format_func=PRESETS.get
    )
    n = st.select_slider(
        "Интервалов по каждой оси",
        options=[8, 12, 16, 24, 32, 48, 64, 96, 128],
        value=24,
    )
    st.caption(f"Узлов: {n+1} × {n+1}; шаг h={1/n:.4g}.")
    mode = st.selectbox(
        "Что нужно вычислить",
        ["point", "patch", "field"],
        format_func={
            "point": "Одна точка",
            "patch": "Участок 3 × 3",
            "field": "Всё поле",
        }.get,
    )
    point = None
    if mode == "point":
        ix = st.slider("Узел по x", 1, n - 1, n // 2)
        iy = st.slider("Узел по y", 1, n - 1, n // 2)
        point = (ix, iy)
        st.caption(f"Координаты: ({ix/n:.3f}, {iy/n:.3f})")
    top, bottom, left, right, source = 100.0, 0.0, 0.0, 0.0, 0.0
    if problem == "hot_top":
        with st.expander("Граница и источник"):
            top = st.number_input("Верх", value=100.0)
            bottom = st.number_input("Низ", value=0.0)
            left = st.number_input("Лево", value=0.0)
            right = st.number_input("Право", value=0.0)
            source = st.number_input("Постоянная правая часть f", value=0.0)
    st.header("Расчёт")
    methods = st.multiselect(
        "Методы", list(METHODS), default=list(METHODS), format_func=METHODS.get
    )
    walks = st.select_slider(
        "Новых путей на стартовую точку",
        options=[64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384],
        value=2048,
    )
    with st.expander("Точность и воспроизводимость"):
        tol = st.select_slider(
            "Относительный допуск ЛОМ",
            options=[1e-2, 1e-4, 1e-6, 1e-7, 1e-8],
            value=1e-7,
            format_func=lambda v: f"{v:g}",
        )
        st.latex(
            r"\frac{\lVert b-Au\rVert_\infty}{\lVert b\rVert_\infty}\leq\mathrm{tol}"
        )
        st.caption("b — правая часть сеточной системы с учётом границы.")
        batches = st.selectbox(
            "Независимых полных расчётов слоёв", [4, 8, 16, 32], index=1
        )
        coarse_n = st.number_input(
            "Интервалов контрольной сетки",
            min_value=4,
            max_value=n,
            value=max(4, n // 4),
            step=1,
        )
        seed = st.number_input(
            "Seed", min_value=0, max_value=2**32 - 1, value=42, step=1
        )
    cfg = Config(
        problem=problem,
        n=n,
        mode=mode,
        point=point,
        walks=walks,
        batches=batches,
        seed=seed,
        coarse_n=coarse_n,
        tol=tol,
        top=top,
        bottom=bottom,
        left=left,
        right=right,
        source=source,
    )
    run = st.button(
        "Рассчитать",
        type="primary",
        disabled=not methods,
        width="stretch",
        key="main_run",
    )

if run:
    progress = st.progress(0.0, text="Подготовка численных ядер…")
    try:
        with st.spinner("Расчёт…"):
            exp = run_experiment(
                cfg, methods, lambda f, t: progress.progress(f, text=t)
            )
        st.session_state["experiment"] = exp
        st.session_state.pop("archive", None)
        st.session_state.pop("benchmark_rows", None)
    except (ValueError, RuntimeError) as exc:
        st.error(str(exc))
    finally:
        progress.empty()

exp = st.session_state.get("experiment")
if (
    exp is not None
    and "lod_serial" in exp.results
    and exp.results["lod_serial"].extras.get("stopping_rule") != LOD_STOPPING_RULE
):
    for key in ("experiment", "archive", "benchmark_rows"):
        st.session_state.pop(key, None)
    exp = None
    st.info("Критерий остановки ЛОМ изменён. Требуется новый расчёт.")
if any(
    row.get("method") == "lod_serial" and row.get("stopping_rule") != LOD_STOPPING_RULE
    for row in st.session_state.get("benchmark_rows", [])
):
    st.session_state.pop("benchmark_rows", None)
    st.session_state.pop("archive", None)
if exp is not None and (
    exp.metadata.get("reference_policy") != REFERENCE_POLICY
    or exp.metadata.get("metrics_policy") != METRICS_POLICY
):
    exp = refresh_comparison(exp)
    st.session_state["experiment"] = exp
    st.session_state.pop("archive", None)
    st.session_state.pop("benchmark_rows", None)
    if not exp.results:
        st.session_state.pop("experiment", None)
        exp = None
render_problem_info(cfg)
if exp is not None and asdict(cfg) != asdict(exp.config):
    st.info("Параметры изменены. Нажмите «Рассчитать», чтобы обновить результаты.")
tabs = st.tabs(
    [
        "Обзор",
        "Поле и ошибка",
        "Как идёт расчёт",
        "Улучшения",
        "3D",
        "Серия экспериментов",
        "Исследование",
        "Сохранить",
    ]
)
with tabs[6]:
    render_research_tab(cfg)

if exp is None:
    with tabs[0]:
        st.info("Нажмите «Рассчитать».")
    st.stop()

ref = exp.comparison
with tabs[0]:
    df = pd.DataFrame(exp.metrics)
    cols = (
        ["name"]
        + (["query_mean"] if len(exp.points) == 1 else [])
        + ["seconds", "rmse", "relative_error_pct", "walks", "steps", "iterations"]
    )
    if len(exp.points) == 1:
        st.write(
            f"Эталон в выбранной точке: **{ref.values[tuple(exp.points[0])]:.6g}**"
        )
    st.dataframe(
        df[cols].rename(
            columns={
                "name": "Метод",
                "query_mean": "Значение u",
                "seconds": "Время, с",
                "rmse": "RMSE к эталону",
                "relative_error_pct": "Отн. ошибка, %",
                "walks": "Новых путей",
                "steps": "Шагов",
                "iterations": "Итераций",
            }
        ),
        hide_index=True,
        width="stretch",
        column_config={
            "Значение u": st.column_config.NumberColumn(format="%.5f"),
            "RMSE к эталону": st.column_config.NumberColumn(format="%.3g"),
            "Отн. ошибка, %": st.column_config.NumberColumn(
                format="%.3g",
                help="100 × ‖u − эталон‖₂ / ‖эталон‖₂ по запрошенным точкам. Не определена, если все эталонные значения равны нулю.",
            ),
        },
    )
    if "mc" in exp.results:
        st.plotly_chart(convergence_plot(exp, True), width="stretch")
with tabs[1]:
    kind = st.radio(
        "Показать",
        ["value", "error", "se", "count"],
        format_func={
            "value": "Значение",
            "error": "Ошибка к эталону",
            "se": "Стандартная ошибка",
            "count": "Число наблюдений",
        }.get,
        horizontal=True,
    )
    fig = interactive_field(exp, kind)
    if fig is not None:
        st.plotly_chart(fig, width="stretch")
with tabs[2]:
    st.plotly_chart(trajectory_plot(exp), width="stretch")
    st.latex(r"Z = g(X_{\tau}) - \frac{h^2}{4}\sum_{k=0}^{\tau-1} f(X_k)")
    baseline = exp.results.get("lod_serial")
    if baseline:
        st.subheader("Сходимость ЛОМ")
        linear_tab, log_tab = st.tabs(["Линейная шкала", "Логарифмическая шкала"])
        with linear_tab:
            st.plotly_chart(residual_plot(exp, "linear"), width="stretch")
        with log_tab:
            st.plotly_chart(residual_plot(exp, "log"), width="stretch")
with tabs[3]:
    st.subheader("Хвосты одного пути")
    st.plotly_chart(reuse_plot(exp), width="stretch")
    if "layers" in exp.results:
        st.subheader("Рассчитанные слои")
        st.plotly_chart(layer_plot(exp), width="stretch")
with tabs[4]:
    st.plotly_chart(surface_plot(exp), width="stretch")
with tabs[5]:
    repeats = st.selectbox("Повторений", [3, 5, 10])
    budgets = st.multiselect(
        "Бюджеты MC",
        [64, 128, 256, 512, 1024, 2048, 4096, 8192],
        default=[256, 1024, 4096],
    )
    if st.button("Провести серию", disabled=not budgets):
        prog = st.progress(0.0)
        try:
            rows = benchmark(
                exp.config,
                sorted(budgets),
                repeats,
                progress=lambda f, t: prog.progress(f, text=t),
            )
            st.session_state["benchmark_rows"] = rows
        except (ValueError, RuntimeError) as exc:
            st.error(str(exc))
        finally:
            prog.empty()
    if "benchmark_rows" in st.session_state:
        rows = st.session_state["benchmark_rows"]
        df = pd.DataFrame(rows)
        st.dataframe(
            df[
                [
                    "name",
                    "repeat",
                    "budget",
                    "budget_kind",
                    "seconds",
                    "rmse",
                    "relative_error_pct",
                ]
            ].rename(
                columns={
                    "name": "Метод",
                    "repeat": "Повтор",
                    "budget": "Бюджет",
                    "budget_kind": "Тип бюджета",
                    "seconds": "Время, с",
                    "rmse": "RMSE к эталону",
                    "relative_error_pct": "Отн. ошибка, %",
                }
            ),
            hide_index=True,
        )
        from pdelab.views import benchmark_figure

        st.plotly_chart(benchmark_figure(rows), width="stretch")
        st.download_button(
            "Скачать замеры CSV",
            df.to_csv(index=False).encode("utf-8-sig"),
            "benchmark.csv",
            "text/csv",
        )
with tabs[7]:
    if st.button("Подготовить все представления"):
        folder = (
            Path(__file__).parent / "runs" / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        )
        with st.spinner("Построение графиков и анимаций…"):
            export(exp, folder)
            if "benchmark_rows" in st.session_state:
                benchmark_plot(st.session_state["benchmark_rows"], folder / "benchmark")
            if "research_result" in st.session_state:
                export_study(st.session_state["research_result"], folder / "research")
            data = io.BytesIO()
            with zipfile.ZipFile(data, "w", zipfile.ZIP_DEFLATED) as z:
                for p in folder.rglob("*"):
                    if p.is_file():
                        z.write(p, p.relative_to(folder))
            st.session_state["archive"] = data.getvalue()
            st.session_state["export_path"] = str(folder)
    if "archive" in st.session_state:
        st.download_button(
            "Скачать результаты ZIP",
            st.session_state["archive"],
            "pde-results.zip",
            "application/zip",
        )
        st.success("Сохранено: " + st.session_state["export_path"])
