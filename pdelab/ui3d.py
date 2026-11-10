"""Интерфейс настоящей трёхмерной задачи, отдельно от 2D-поверхности."""

import io
import json
import zipfile
from dataclasses import asdict

import numpy as np
import pandas as pd
import streamlit as st

from .experiment3d import Config3D, experiment_payload3d, rows_csv, run_experiment3d
from .problem3d import PRESETS_3D, workspace_bytes
from .research3d import (
    StudySettings3D,
    accuracy_study3d,
    study_payload3d,
    timing_rows3d,
)
from .solvers3d import METHODS_3D
from .views3d import residual_figure3d, slice_figure3d
from .workload3d import interactive_mc_limit_message

GRIDS = [8, 12, 16, 24, 32, 48, 64]


def _point_controls(n, prefix):
    previous_n = st.session_state.get(f"{prefix}_n", n)
    if previous_n != n:
        for axis in "xyz":
            key = f"{prefix}_{axis}"
            old = st.session_state.get(key, previous_n // 2)
            st.session_state[key] = min(n, max(0, round(old * n / previous_n)))
    st.session_state[f"{prefix}_n"] = n
    coordinates = []
    for axis, column in zip("xyz", st.columns(3)):
        with column:
            coordinates.append(
                st.number_input(
                    axis,
                    min_value=0,
                    max_value=n,
                    value=n // 2,
                    step=1,
                    key=f"{prefix}_{axis}",
                )
            )
    st.caption(
        "Координаты: (" + ", ".join(f"{value / n:.4g}" for value in coordinates) + ")"
    )
    return tuple(coordinates)


def _problem_info(kind):
    with st.expander("Формулы и описание", expanded=True):
        st.latex(
            r"u_{xx}+u_{yy}+u_{zz}=f,\quad (x,y,z)\in(0,1)^3,\qquad u|_{\partial\Omega}=g"
        )
        if kind == "sine":
            st.latex(
                r"u_*=\sin(\pi x)\sin(\pi y)\sin(\pi z),\qquad f=-3\pi^2u_*,\qquad g=0"
            )
            st.write(
                "Синус: проверка источника и уменьшения погрешности при уточнении сетки."
            )
        elif kind == "affine":
            st.latex(r"u_*=1+2x+3y+5z,\qquad f=0,\qquad g=u_*|_{\partial\Omega}")
            st.write(
                "Линейное поле: разные коэффициенты по осям помогают обнаружить перестановку координат."
            )
        else:
            st.latex(r"u_*=7,\qquad f=0,\qquad g=7")
            st.write("Константа: контрольный пример без случайного разброса MC.")
        st.latex(r"u_{ijk}=\frac{1}{6}\sum_{\text{6 соседей}}u-\frac{h^2f_{ijk}}{6}")


def _research_tab(config):
    grid_column, repeat_column = st.columns(2)
    with grid_column:
        n = st.select_slider(
            "Сетка исследования", options=GRIDS, value=config.n, key="study3d_grid"
        )
    with repeat_column:
        repeats = st.number_input(
            "Число усреднений",
            min_value=2,
            max_value=100,
            value=5,
            key="study3d_repeats",
        )
    with st.expander("Точка и лимиты"):
        point = _point_controls(n, "study3d_point")
        maximum = st.select_slider(
            "Максимум путей",
            options=[64, 256, 1024, 4096, 16384, 65536, 262144],
            value=16384,
            key="study3d_walks",
        )
        seconds = st.number_input(
            "Лимит поиска на метод, с",
            min_value=1.0,
            max_value=1800.0,
            value=60.0,
            key="study3d_seconds",
            help="Проверяется между запусками. Начатый запуск завершается полностью.",
        )
    settings = StudySettings3D(
        Config3D(
            problem=config.problem,
            n=n,
            point=point,
            seed=config.seed,
            max_iter=config.max_iter,
        ),
        int(repeats),
        maximum,
        seconds,
    )
    if st.button("Запустить исследование", key="study3d_run", type="primary"):
        progress = st.progress(0.0)
        try:
            st.session_state["study3d_result"] = accuracy_study3d(
                settings, lambda fraction, text: progress.progress(fraction, text=text)
            )
        except (ValueError, RuntimeError) as exc:
            st.error(str(exc))
        finally:
            progress.empty()
    study = st.session_state.get("study3d_result")
    if study is None:
        return
    if asdict(study.settings) != asdict(settings):
        st.info("Настройки изменены. Запустите исследование заново.")
    table = timing_rows3d(study)
    st.dataframe(pd.DataFrame(table), hide_index=True, width="stretch")
    st.download_button(
        "Скачать таблицу CSV",
        rows_csv(table).encode("utf-8-sig"),
        "3d-timings.csv",
        "text/csv",
    )
    with st.expander("Детали исследования"):
        st.write(
            "Бюджет фиксируется после подбора и проверяется новой серией. Прочерк означает недостигнутую точность; проверка не запускает повторный подбор."
        )
        st.dataframe(pd.DataFrame(study.rows), hide_index=True, width="stretch")
        payload = {
            **study_payload3d(study),
            "thresholds": study.rows,
            "trials": study.trials,
        }
        st.download_button(
            "Скачать все замеры JSON",
            json.dumps(payload, ensure_ascii=False, indent=2),
            "3d-study.json",
            "application/json",
        )


def _archive(experiment, plane, index):
    output = io.BytesIO()
    arrays = io.BytesIO()
    fields = {method: result.values for method, result in experiment.results.items()}
    fields.update(
        reference=experiment.reference,
        points_zyx=experiment.points,
        axis=experiment.problem.axis,
    )
    if experiment.grid_reference is not None:
        fields["grid_reference"] = experiment.grid_reference
    np.savez_compressed(arrays, **fields)
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "experiment.json",
            json.dumps(experiment_payload3d(experiment), ensure_ascii=False, indent=2),
        )
        archive.writestr(
            "metrics.csv", rows_csv(experiment.metrics).encode("utf-8-sig")
        )
        archive.writestr("fields.npz", arrays.getvalue())
        for error, name in ((False, "solution.html"), (True, "error.html")):
            archive.writestr(
                name,
                slice_figure3d(experiment, plane, index, error).to_html(
                    include_plotlyjs=True
                ),
            )
    return output.getvalue()


def render_3d_app():
    st.title("ЛОМ и Монте-Карло в кубе")
    with st.sidebar:
        problem = st.selectbox(
            "Задача", list(PRESETS_3D), format_func=PRESETS_3D.get, key="cube_problem"
        )
        n = st.select_slider("Интервалов по оси", options=GRIDS, value=16, key="cube_n")
        st.caption(
            f"{n + 1}³ узлов; резерв массивов {workspace_bytes(n) / 2**20:.1f} МиБ.",
            help="Расчётный резерв для плотных массивов. JIT, графики и sparse LU учитываются отдельно.",
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
            key="cube_mode",
        )
        point, plane, index = None, "xy", None
        if mode == "point":
            point = _point_controls(n, "cube_point")
        if mode == "slice":
            plane = st.selectbox(
                "Плоскость расчёта", ["xy", "xz", "yz"], key="cube_plane"
            )
            index = st.slider("Узел среза", 0, n, n // 2, key=f"cube_index_{n}")
        methods = st.multiselect(
            "Методы",
            list(METHODS_3D),
            default=list(METHODS_3D),
            format_func=METHODS_3D.get,
            key="cube_methods",
        )
        walks = st.select_slider(
            "Путей на точку",
            options=[64, 256, 1024, 2048, 4096, 16384, 65536],
            value=2048,
            key="cube_walks",
        )
        with st.expander("Точность и воспроизводимость"):
            tol = st.select_slider(
                "Относительный допуск ЛОМ",
                options=[1e-2, 1e-4, 1e-6, 1e-7, 1e-8],
                value=1e-7,
                key="cube_tol",
            )
            seed = st.number_input(
                "Seed", min_value=0, max_value=2**32 - 1, value=42, key="cube_seed"
            )
        config = Config3D(problem, n, mode, point, plane, index, walks, seed, tol)
        limit_message = interactive_mc_limit_message(config, methods)
        if limit_message:
            st.error(limit_message)
        run = st.button(
            "Рассчитать",
            key="cube_run",
            type="primary",
            disabled=not methods or limit_message is not None,
        )
    _problem_info(problem)
    if run and limit_message is None:
        try:
            with st.spinner("Расчёт в кубе…"):
                st.session_state["cube_result"] = run_experiment3d(config, methods)
                st.session_state.pop("cube_archive", None)
        except (ValueError, RuntimeError) as exc:
            st.error(str(exc))
    overview, slices, research, download = st.tabs(
        ["Результат", "Срезы", "Исследование", "Сохранить"]
    )
    with research:
        _research_tab(config)
    experiment = st.session_state.get("cube_result")
    if experiment is None:
        with overview:
            st.info("Нажмите «Рассчитать».")
        return
    if (
        asdict(experiment.config) != asdict(config)
        or list(experiment.results) != methods
    ):
        st.info("Параметры изменены. Нажмите «Рассчитать», чтобы обновить результаты.")
    with overview:
        columns = (
            ["name"]
            + (["value"] if len(experiment.points) == 1 else [])
            + ["seconds", "rmse", "relative_error_pct", "walks", "steps", "iterations"]
        )
        st.dataframe(
            pd.DataFrame(experiment.metrics)[columns].rename(
                columns={
                    "name": "Метод",
                    "value": "Значение u",
                    "seconds": "Время, с",
                    "rmse": "RMSE",
                    "relative_error_pct": "Отн. ошибка, %",
                    "walks": "Путей",
                    "steps": "Шагов",
                    "iterations": "Итераций",
                }
            ),
            hide_index=True,
            width="stretch",
        )
        if "lod_serial" in experiment.results:
            st.plotly_chart(
                residual_figure3d(experiment.results["lod_serial"]),
                width="stretch",
                key="cube_residual",
            )
    with slices:
        view_plane = st.selectbox(
            "Плоскость просмотра", ["xy", "xz", "yz"], key="cube_view_plane"
        )
        fixed_axis = {"xy": 0, "xz": 1, "yz": 2}[view_plane]
        default_index = (
            int(experiment.points[0, fixed_axis])
            if experiment.config.mode in ("point", "slice")
            else experiment.config.n // 2
        )
        view_index = st.slider(
            "Положение среза",
            0,
            experiment.config.n,
            default_index,
            key=f"cube_view_{experiment.config.n}_{view_plane}_{default_index}",
        )
        st.plotly_chart(
            slice_figure3d(experiment, view_plane, view_index),
            width="stretch",
            key="cube_values",
        )
        st.plotly_chart(
            slice_figure3d(experiment, view_plane, view_index, error=True),
            width="stretch",
            key="cube_errors",
        )
    with download:
        view = (view_plane, view_index)
        if st.session_state.get("cube_archive_view") != view:
            st.session_state.pop("cube_archive", None)
        if st.button("Подготовить ZIP", key="cube_export"):
            st.session_state["cube_archive"] = _archive(
                experiment, view_plane, view_index
            )
            st.session_state["cube_archive_view"] = view
        if "cube_archive" in st.session_state:
            st.download_button(
                "Скачать ZIP",
                st.session_state["cube_archive"],
                "pde-3d.zip",
                "application/zip",
            )
