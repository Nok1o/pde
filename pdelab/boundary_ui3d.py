"""Six face controls, comparison, slices and reflected walks in the cube."""

import json
from dataclasses import asdict

import pandas as pd
import plotly.express as px
import streamlit as st

from .boundary import KINDS
from .boundary3d import FACES, PRESETS_BOUNDARY3D, BoundaryOperator3D
from .boundary_experiment3d import (
    BoundaryConfig3D,
    benchmark_boundary3d,
    boundary_archive3d,
    problem_from_config3d,
    run_boundary_experiment3d,
)
from .boundary_solvers3d import trace_boundary_walk3d
from .boundary_ui import KIND_LABELS
from .boundary_views3d import boundary_trace_figure3d
from .boundary_workload3d import boundary_limit_message3d
from .experiment3d import rows_csv
from .solvers3d import METHODS_3D
from .views3d import PLANES, residual_figure3d, slice_figure3d


def _node(label, n, value, key):
    if key in st.session_state:
        st.session_state[key] = min(n, max(0, int(st.session_state[key])))
    return st.number_input(label, min_value=0, max_value=n, value=value, key=key)


def _controls():
    names = {**PRESETS_BOUNDARY3D, "custom": "Свои данные"}
    problem = st.selectbox(
        "Задача", list(names), format_func=names.get, key="bc3d_problem"
    )
    grids = [8, 12, 16] if problem == "custom" else [8, 12, 16, 24, 32, 48, 64]
    if st.session_state.get("bc3d_n", 16) not in grids:
        st.session_state["bc3d_n"] = 16
    n = st.select_slider("Интервалов по оси", grids, value=16, key="bc3d_n")
    kinds, kappas, data = [], [], []
    for i, face in enumerate(FACES):
        with st.expander(f"Грань {face[0]} = {face[1]}"):
            kind = st.selectbox(
                "Условие",
                KINDS,
                index=2 if i == 0 else 0,
                format_func=KIND_LABELS.get,
                key=f"bc3d_{face}_kind",
            )
            kinds.append(kind)
            kappas.append(
                st.number_input("κ", min_value=0.0, value=1.0, key=f"bc3d_{face}_kappa")
                if kind == "robin"
                else 0.0
            )
            data.append(
                st.number_input("g или r", value=0.0, key=f"bc3d_{face}_value")
                if problem == "custom"
                else 0.0
            )
    source = (
        st.number_input("Источник f", value=0.0, key="bc3d_source")
        if problem == "custom"
        else 0.0
    )
    mode = st.selectbox(
        "Что вычислить",
        ["point", "patch", "slice", "field"],
        format_func={
            "point": "Одна точка",
            "patch": "Участок 3 × 3 × 3",
            "slice": "Срез",
            "field": "Всё поле",
        }.get,
        key="bc3d_mode",
    )
    point = tuple(_node(f"Узел {axis}", n, n // 2, f"bc3d_{axis}") for axis in "xyz")
    plane, index = "xy", point[2]
    if mode == "slice":
        plane = st.selectbox("Плоскость расчёта", list(PLANES), key="bc3d_plane")
        index = _node("Индекс среза", n, n // 2, "bc3d_index")
    with st.expander("Точность и бюджет"):
        walks = st.select_slider(
            "Путей на точку",
            [64, 256, 1024, 2048, 4096, 16384],
            value=2048,
            key="bc3d_walks",
        )
        tol = st.select_slider(
            "Относительный допуск ЛОМ",
            [1e-3, 1e-5, 1e-7, 1e-8, 1e-10],
            value=1e-8,
            key="bc3d_tol",
        )
        seed = st.number_input(
            "Seed", min_value=0, max_value=2**32 - 1, value=42, key="bc3d_seed"
        )
        max_iter = st.number_input(
            "Максимум итераций ЛОМ",
            min_value=1,
            max_value=200000,
            value=20000,
            key="bc3d_max_iter",
        )
        max_steps = st.number_input(
            "Максимум шагов одного пути",
            min_value=1,
            value=1000 * n**2,
            key="bc3d_max_steps",
        )
    return BoundaryConfig3D(
        problem=problem,
        n=n,
        kinds=tuple(kinds),
        kappas=tuple(kappas),
        data=tuple(data),
        source=source,
        mode=mode,
        point=point,
        walks=walks,
        seed=seed,
        tol=tol,
        max_iter=max_iter,
        max_steps=max_steps,
        plane=plane,
        index=index,
    )


def _formulas(config):
    with st.expander("Формулы и описание", expanded=True):
        st.latex(r"u_{xx}+u_{yy}+u_{zz}=f,\quad (x,y,z)\in(0,1)^3,\qquad k=1")
        if config.problem == "polynomial":
            st.latex(r"u_*=1+x+2y+3z+x^2+y^2+z^2,\qquad f=6")
            st.write(
                "Полином: проверка знаков нормали и совместной обработки граней на рёбрах и в углах."
            )
        elif config.problem == "smooth":
            st.latex(
                r"u_*=3+x+2y+3z+\cos(2\pi x)+\tfrac12\cos(2\pi y)+\tfrac14\cos(2\pi z)"
            )
            st.latex(
                r"f=-4\pi^2\left(\cos(2\pi x)+\tfrac12\cos(2\pi y)+\tfrac14\cos(2\pi z)\right)"
            )
            st.write("Гладкий пример: проверка уменьшения ошибки при уточнении сетки.")
        else:
            st.latex(rf"f={config.source:g}")
        normals = (
            (-1, 3, -2, 4, -3, 5)
            if config.problem == "polynomial"
            else (-1, 1, -2, 2, -3, 3)
        )
        for i, (face, kind, kappa) in enumerate(
            zip(FACES, config.kinds, config.kappas)
        ):
            derivative = ("-" if face[1] == "0" else "") + f"u_{face[0]}"
            left = "u" if kind == "dirichlet" else derivative
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
            st.latex(rf"{face[0]}={face[1]}:\quad {left}={right}")
        st.caption(
            "Нейман задаёт внешнюю производную. На рёбрах и в углах складываются вклады соседних граней; Дирихле имеет приоритет."
        )


def _comparison(experiment, config, methods):
    if asdict(config) != asdict(experiment.config) or set(methods) != set(
        experiment.results
    ):
        st.info("Настройки изменены. Запустите расчёт заново.")
    columns = {
        "name": "Метод",
        "value": "Значение u",
        "reference_value": "Эталон",
        "seconds": "Время, с",
        "rmse": "RMSE",
        "relative_error_pct": "Отн. ошибка, %",
    }
    st.dataframe(
        pd.DataFrame(experiment.metrics)[list(columns)].rename(columns=columns),
        hide_index=True,
        width="stretch",
    )
    plane = st.selectbox(
        "Плоскость просмотра",
        list(PLANES),
        index=list(PLANES).index(experiment.config.plane),
        key="bc3d_view_plane",
    )
    if experiment.config.mode == "slice":
        default_index = experiment.config.index
    else:
        default_index = experiment.config.point[2 - PLANES[plane][0]]
    index = _node("Слой", experiment.config.n, default_index, "bc3d_view_index")
    error = st.toggle("Показать ошибку", key="bc3d_show_error")
    st.plotly_chart(slice_figure3d(experiment, plane, index, error), width="stretch")
    if "lod_serial" in experiment.results:
        with st.expander("Сходимость ЛОМ"):
            st.plotly_chart(
                residual_figure3d(experiment.results["lod_serial"]), width="stretch"
            )
    if st.button("Подготовить архив", key="bc3d_export"):
        st.session_state["bc3d_archive"] = (
            plane,
            index,
            boundary_archive3d(experiment, plane, index),
        )
    archive = st.session_state.get("bc3d_archive")
    if archive is not None and archive[:2] == (plane, index):
        st.download_button(
            "Скачать результаты",
            archive[2],
            "boundary3d.zip",
            "application/zip",
            key="bc3d_download",
        )


def _benchmark(config, methods):
    repeats = st.number_input(
        "Независимых повторов", min_value=2, max_value=20, value=3, key="bc3d_repeats"
    )
    budgets = st.multiselect(
        "Бюджеты MC",
        [64, 256, 1024, 4096],
        default=[64, 256, 1024],
        key="bc3d_budgets",
        disabled="mc" not in methods,
    )
    settings = {
        "config": asdict(config),
        "methods": methods,
        "budgets": budgets,
        "repeats": repeats,
        "tolerances": [1e-3, 1e-6, 1e-9],
    }
    disabled = not methods or ("mc" in methods and not budgets)
    if st.button("Запустить серию", key="bc3d_benchmark", disabled=disabled):
        warning = boundary_limit_message3d(config, methods, sum(budgets), repeats)
        if warning:
            st.error(warning)
        else:
            try:
                with st.spinner("Серия расчётов…"):
                    st.session_state["bc3d_series"] = (
                        settings,
                        benchmark_boundary3d(config, budgets, repeats, methods=methods),
                    )
            except (ValueError, RuntimeError) as exc:
                st.error(str(exc))
    saved = st.session_state.get("bc3d_series")
    if saved is not None:
        if saved[0] != settings:
            st.info("Показана серия с прежними настройками.")
        figure = px.scatter(
            pd.DataFrame(saved[1]),
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
            "boundary3d-series.csv",
            "text/csv",
        )
        st.download_button(
            "Скачать настройки серии",
            json.dumps(saved[0], ensure_ascii=False, indent=2),
            "boundary3d-series.json",
            "application/json",
        )


def render_boundary_app3d():
    st.title("Куб · условия I, II и III рода")
    with st.sidebar:
        config = _controls()
    _formulas(config)
    try:
        model = problem_from_config3d(config)
        operator = BoundaryOperator3D(model)
        operator.check_compatibility()
    except ValueError as exc:
        st.error(str(exc))
        return
    pure_neumann = operator.pure_neumann
    del operator
    available = ["lod_serial"] if pure_neumann else list(METHODS_3D)
    if pure_neumann:
        st.caption(
            "Чистый Нейман: среднее с весами объёмов равно нулю. MC без поглощения пока не поддерживается."
        )
    with st.sidebar:
        methods = st.multiselect(
            "Методы",
            available,
            default=available,
            format_func=METHODS_3D.get,
            key="bc3d_methods",
        )
        run = st.button(
            "Рассчитать", type="primary", disabled=not methods, key="bc3d_run"
        )
    if run:
        warning = boundary_limit_message3d(config, methods)
        if warning:
            st.error(warning)
        else:
            try:
                with st.spinner("Расчёт…"):
                    st.session_state["bc3d_result"] = run_boundary_experiment3d(
                        config, methods
                    )
                for key in ("bc3d_archive", "bc3d_view_index", "bc3d_view_plane"):
                    st.session_state.pop(key, None)
            except (ValueError, RuntimeError) as exc:
                st.error(str(exc))
    comparison, trace_tab, series = st.tabs(
        ["Сравнение и срезы", "Траектория", "Серия измерений"]
    )
    with comparison:
        experiment = st.session_state.get("bc3d_result")
        if experiment is not None:
            _comparison(experiment, config, methods)
    with trace_tab:
        if st.button("Показать траекторию", key="bc3d_trace", disabled=pure_neumann):
            try:
                trace = trace_boundary_walk3d(
                    model, config.point, config.seed, min(config.max_steps, 20000)
                )
                st.session_state["bc3d_trace_result"] = (asdict(config), trace)
            except (ValueError, RuntimeError) as exc:
                st.error(str(exc))
        saved = st.session_state.get("bc3d_trace_result")
        if saved is not None and saved[0] == asdict(config):
            st.plotly_chart(
                boundary_trace_figure3d(saved[1], config.n), width="stretch"
            )
            with st.expander("События на границе"):
                st.dataframe(
                    pd.DataFrame([e for e in saved[1] if e["faces"]]), hide_index=True
                )
    with series:
        _benchmark(config, methods)
