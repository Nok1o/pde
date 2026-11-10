"""Compact UI for individually selected boundary conditions on four sides."""

import json
from dataclasses import asdict

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from .boundary import KINDS, PRESETS_BOUNDARY, SIDES, BoundaryOperator
from .boundary_experiment import (
    BoundaryConfig,
    benchmark_boundary,
    boundary_payload,
    problem_from_config,
    run_boundary_experiment,
)
from .boundary_solvers import (
    METHODS_BOUNDARY,
    select_boundary_points,
    trace_boundary_walk,
)
from .boundary_views import boundary_field_figure, boundary_trace_figure
from .experiment3d import rows_csv

SIDE_LABELS = {"left": "Левая", "right": "Правая", "bottom": "Нижняя", "top": "Верхняя"}
KIND_LABELS = {
    "dirichlet": "I · Дирихле",
    "neumann": "II · Нейман",
    "robin": "III · Робин",
}


def workload_message(config, methods, walks=None, repeats=1):
    if not any(m != "lod_serial" for m in methods):
        return None
    points = select_boundary_points(config.n, config.mode, config.point)
    operator = BoundaryOperator(problem_from_config(config))
    point_count = int(np.count_nonzero(~operator.fixed[tuple(points.T)]))
    multiplier = 4.0
    if "dirichlet" not in config.kinds:
        positive = [
            k
            for kind, k in zip(config.kinds, config.kappas)
            if kind == "robin" and k > 0
        ]
        if positive:
            multiplier *= 1 + 1 / min(positive)
    estimate = (
        point_count
        * (config.walks if walks is None else walks)
        * config.n**2
        * multiplier
        * repeats
    )
    if estimate > 200_000_000:
        return "Этот запрос MC слишком велик для интерактивного запуска. Уменьшите сетку, число путей или область расчёта. Большие серии доступны через CLI."
    return None


def _formulas(config):
    with st.expander("Формулы и описание", expanded=True):
        st.latex(r"\Delta u=f,\quad (x,y)\in(0,1)^2,\qquad k=1")
        if config.problem == "polynomial":
            st.latex(r"u_*=1+x+2y+x^2+y^2,\qquad f=4")
            st.write("Полином: проверка знаков внешней нормали и граничных строк.")
        elif config.problem == "smooth":
            st.latex(r"u_*=2+x+2y+\cos(2\pi x)+\tfrac12\cos(2\pi y)")
            st.latex(r"f=-4\pi^2\cos(2\pi x)-2\pi^2\cos(2\pi y)")
            st.write("Гладкий пример: проверка уменьшения ошибки при уточнении сетки.")
        else:
            st.latex(rf"f={config.source:g}")
        normals = (-1, 3, -2, 4) if config.problem == "polynomial" else (-1, 1, -2, 2)
        locations = ("x=0", "x=1", "y=0", "y=1")
        derivatives = (r"-u_x", r"u_x", r"-u_y", r"u_y")
        for i, (side, kind, kappa) in enumerate(
            zip(SIDES, config.kinds, config.kappas)
        ):
            left = "u" if kind == "dirichlet" else derivatives[i]
            if kind == "robin":
                left += rf"+{kappa:g}u"
            if config.problem == "custom":
                right = f"{config.data[i]:g}"
            elif kind == "dirichlet":
                right = "u_*"
            else:
                right = f"{normals[i]:g}" + (
                    rf"+{kappa:g}u_*" if kind == "robin" else ""
                )
            st.latex(rf"{locations[i]}:\quad {left}={right}")
        st.caption(
            "Нейман задаёт внешнюю производную. Физический поток при k=1: q = −∂u/∂n."
        )


def render_boundary_app():
    st.title("Условия Неймана и Робина")
    with st.sidebar:
        names = {**PRESETS_BOUNDARY, "custom": "Свои данные"}
        problem = st.selectbox(
            "Задача со смешанной границей",
            list(names),
            format_func=names.get,
            key="bc_problem",
        )
        n = st.select_slider(
            "Интервалов по оси", [8, 12, 16, 24, 32, 48, 64], value=24, key="bc_n"
        )
        kinds, kappas, data = [], [], []
        for i, side in enumerate(SIDES):
            with st.expander(SIDE_LABELS[side] + " сторона"):
                kind = st.selectbox(
                    "Условие",
                    KINDS,
                    index=2 if i == 0 else 0,
                    format_func=KIND_LABELS.get,
                    key=f"bc_{side}_kind",
                )
                kinds.append(kind)
                kappas.append(
                    st.number_input(
                        "κ", min_value=0.0, value=1.0, key=f"bc_{side}_kappa"
                    )
                    if kind == "robin"
                    else 0.0
                )
                data.append(
                    st.number_input("g или r", value=0.0, key=f"bc_{side}_value")
                    if problem == "custom"
                    else 0.0
                )
        source = (
            st.number_input("Источник f", value=0.0, key="bc_source")
            if problem == "custom"
            else 0.0
        )
        mode = st.selectbox(
            "Что вычислить",
            ["point", "patch", "field"],
            format_func={
                "point": "Одна точка",
                "patch": "Участок",
                "field": "Всё поле",
            }.get,
            key="bc_mode",
        )
        x = st.number_input(
            "Узел x", min_value=0, max_value=n, value=n // 2, key="bc_x"
        )
        y = st.number_input(
            "Узел y", min_value=0, max_value=n, value=n // 2, key="bc_y"
        )
        st.caption(f"Точка ({x / n:g}, {y / n:g})")
        with st.expander("Точность и бюджет"):
            walks = st.select_slider(
                "Путей на точку",
                [64, 256, 1024, 2048, 4096, 16384, 65536],
                value=2048,
                key="bc_walks",
            )
            tol = st.select_slider(
                "Относительный допуск ЛОМ",
                [1e-3, 1e-5, 1e-7, 1e-8, 1e-10],
                value=1e-8,
                key="bc_tol",
            )
            seed = st.number_input(
                "Seed", min_value=0, max_value=2**32 - 1, value=42, key="bc_seed"
            )
            max_iter = st.number_input(
                "Максимум итераций ЛОМ",
                min_value=1,
                max_value=200000,
                value=20000,
                key="bc_max_iter",
            )
            max_steps = st.number_input(
                "Максимум шагов одного пути",
                min_value=1,
                value=1000 * n**2,
                key="bc_max_steps",
            )
        config = BoundaryConfig(
            problem,
            n,
            tuple(kinds),
            tuple(kappas),
            tuple(data),
            source,
            mode,
            (x, y),
            walks,
            seed,
            tol,
            max_iter,
            max_steps,
        )
    from .stationary_extensions import render_method_scope

    render_method_scope(2, mixed=True)
    _formulas(config)
    try:
        model = problem_from_config(config)
        operator = BoundaryOperator(model)
        operator.check_compatibility()
    except ValueError as exc:
        st.error(str(exc))
        return
    available = ["lod_serial"] if operator.pure_neumann else list(METHODS_BOUNDARY)
    if any(kind != "dirichlet" for kind in config.kinds):
        available = [m for m in available if m != "wos"]
    if operator.pure_neumann:
        st.caption(
            "Чистый Нейман: среднее с весами объёмов равно нулю. MC без поглощения пока не поддерживается."
        )
    with st.sidebar:
        methods = st.multiselect(
            "Методы",
            available,
            default=available,
            format_func=METHODS_BOUNDARY.get,
            key="bc_methods",
        )
        run = st.button("Рассчитать", type="primary", key="bc_run")
    if run:
        warning = workload_message(config, methods)
        if warning:
            st.error(warning)
        else:
            try:
                with st.spinner("Расчёт…"):
                    st.session_state["bc_result"] = run_boundary_experiment(
                        config, methods
                    )
            except (ValueError, RuntimeError) as exc:
                st.error(str(exc))
    comparison_tab, trace_tab, benchmark_tab, research_tab = st.tabs(
        ["Сравнение", "Траектория", "Серия измерений", "Исследование"]
    )
    with comparison_tab:
        experiment = st.session_state.get("bc_result")
        if experiment is not None:
            if asdict(config) != asdict(experiment.config) or set(methods) != set(
                experiment.results
            ):
                st.info("Настройки изменены. Запустите расчёт заново.")
            table = pd.DataFrame(experiment.metrics)
            st.dataframe(
                table[
                    [
                        "name",
                        "value",
                        "reference_value",
                        "seconds",
                        "rmse",
                        "relative_error_pct",
                    ]
                ].rename(
                    columns={
                        "name": "Метод",
                        "value": "Значение u",
                        "reference_value": "Эталон",
                        "seconds": "Время, с",
                        "rmse": "RMSE",
                        "relative_error_pct": "Отн. ошибка, %",
                    }
                ),
                hide_index=True,
                width="stretch",
            )
            error = st.toggle("Показать ошибку", key="bc_show_error")
            st.plotly_chart(boundary_field_figure(experiment, error), width="stretch")
            st.download_button(
                "Скачать CSV",
                rows_csv(experiment.metrics).encode("utf-8-sig"),
                "boundary-metrics.csv",
                "text/csv",
            )
            st.download_button(
                "Скачать настройки и результаты",
                json.dumps(boundary_payload(experiment), ensure_ascii=False, indent=2),
                "boundary-experiment.json",
                "application/json",
            )
    with trace_tab:
        if not operator.pure_neumann and st.button(
            "Показать траекторию",
            key="bc_trace",
            help="Поясняющий рисунок ограничен 20 000 шагами; незавершённый путь не показывается как результат.",
        ):
            try:
                st.session_state["bc_trace_result"] = (
                    asdict(config),
                    trace_boundary_walk(model, (x, y), seed, min(max_steps, 20000)),
                )
            except (ValueError, RuntimeError) as exc:
                st.error(str(exc))
        trace = st.session_state.get("bc_trace_result")
        if trace is not None and trace[0] == asdict(config):
            st.plotly_chart(boundary_trace_figure(trace[1], n), width="stretch")
            with st.expander("События на границе"):
                st.dataframe(
                    pd.DataFrame([e for e in trace[1] if e["sides"]]), hide_index=True
                )
    with benchmark_tab:
        repeats = st.number_input(
            "Независимых повторов", min_value=2, max_value=20, value=3, key="bc_repeats"
        )
        budgets = st.multiselect(
            "Бюджеты MC",
            [64, 256, 1024, 4096, 16384],
            default=[64, 256, 1024],
            key="bc_budgets",
        )
        benchmark_settings = {
            "config": asdict(config),
            "budgets": budgets,
            "repeats": repeats,
            "methods": methods,
        }
        disabled = not methods or (
            any(m != "lod_serial" for m in methods) and not budgets
        )
        if st.button("Запустить серию", key="bc_benchmark", disabled=disabled):
            warning = workload_message(
                config,
                methods,
                max(budgets, default=0),
                repeats * max(1, len(budgets)),
            )
            if warning:
                st.error(warning)
            else:
                try:
                    with st.spinner("Серия расчётов…"):
                        st.session_state["bc_benchmark_result"] = (
                            benchmark_settings,
                            benchmark_boundary(
                                config, budgets, repeats, methods=methods
                            ),
                        )
                except (ValueError, RuntimeError) as exc:
                    st.error(str(exc))
        saved = st.session_state.get("bc_benchmark_result")
        if saved is not None and saved[1]:
            if saved[0] != benchmark_settings:
                st.info("Показана серия с прежними настройками.")
            table = pd.DataFrame(saved[1])
            figure = px.scatter(
                table,
                x="seconds",
                y="rmse",
                color="name",
                hover_data=["budget", "repeat", "converged"],
                labels={"seconds": "Время, с", "rmse": "RMSE", "name": "Метод"},
            )
            st.plotly_chart(figure, width="stretch")
            st.download_button(
                "Скачать серию CSV",
                rows_csv(saved[1]).encode("utf-8-sig"),
                "boundary-benchmark.csv",
                "text/csv",
            )
            st.download_button(
                "Скачать настройки серии",
                json.dumps(saved[0], ensure_ascii=False, indent=2),
                "boundary-benchmark-settings.json",
                "application/json",
            )

    with research_tab:
        from .boundary_research import render_boundary_research

        render_boundary_research(config, methods, "bc_research")
