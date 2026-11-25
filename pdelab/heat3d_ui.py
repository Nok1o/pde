"""Supported nonlinear 3D workflow; saved settings belong to saved results."""

from dataclasses import asdict, replace

import pandas as pd
import plotly.express as px
import streamlit as st

from .heat3d_experiment import (
    APPLICABILITY,
    METHOD_NAME,
    STATUS_LABELS,
    HeatConfig3D,
    compare_heat3d,
    csv_bytes,
    json_bytes,
    result_archive,
    study_heat3d,
)
from .ui_state import select_section


def _error_text(row, *, absolute=False):
    value = row.get("absolute_error" if absolute else "relative_error_pct")
    if value is not None:
        return f"{value:.6g}"
    if not absolute and (
        row.get("status") == "relative_target_not_applicable"
        or row.get("relative_applicable") is False
    ):
        return "неприменима: нулевой эталон"
    return "нет оценки: расчёт или серия не завершены"


def _downloads(result, key):
    st.download_button(
        "Скачать ZIP",
        result_archive(result),
        "heat3d-" + key + ".zip",
        "application/zip",
        key=key + "_zip",
        on_click="ignore",
    )
    st.download_button(
        "Скачать CSV",
        csv_bytes(result["rows"]),
        "heat3d-" + key + ".csv",
        "text/csv",
        key=key + "_csv",
        on_click="ignore",
    )
    st.download_button(
        "Скачать JSON",
        json_bytes({k: v for k, v in result.items() if k != "arrays"}),
        "heat3d-" + key + ".json",
        "application/json",
        key=key + "_json",
        on_click="ignore",
    )


def _saved_settings(result, current):
    saved = result["metadata"]
    if (
        saved["configuration"] != current["configuration"]
        or saved["solver_settings"] != current["solver_settings"]
    ):
        st.info("Настройки изменены. Показан сохранённый расчёт с параметрами ниже.")
    with st.expander(
        "Настройки выполненного запуска", key="h3_saved_" + saved["command"]
    ):
        st.json(
            {
                key: saved[key]
                for key in (
                    "configuration",
                    "solver_settings",
                    "query_size",
                    "target_time",
                    "boundary_kinds",
                )
            }
        )
        st.code(saved["reproduction"], language="bash")


def render_heat3d_app():
    from .heat3d_solver import HeatSettings3D

    st.title("Квазилинейная теплопроводность · 3D")
    st.caption(APPLICABILITY)
    with st.sidebar:
        case = st.selectbox(
            "Проверочная функция",
            ("sine", "polynomial"),
            format_func={"sine": "Синусоидальная", "polynomial": "Полиномиальная"}.get,
            key="h3_case",
        )
        n = st.select_slider(
            "Интервалов по оси", (4, 6, 8, 12, 16, 24, 32, 64), value=8, key="h3_n"
        )
        alpha = st.number_input(
            "α в k(u)=1+αu²", min_value=0.0, max_value=100.0, value=0.5, key="h3_alpha"
        )
        dt = st.number_input(
            "Физический шаг dt",
            min_value=0.0001,
            max_value=1.0,
            value=0.01,
            format="%.4f",
            key="h3_dt",
        )
        final = st.number_input(
            "Конечное время T",
            min_value=0.0,
            max_value=2.0,
            value=0.02,
            format="%.4f",
            key="h3_T",
        )
        mode = st.selectbox(
            "Область запроса",
            ("point", "patch", "slice", "field"),
            format_func={
                "point": "Одна точка",
                "patch": "Участок в центре",
                "slice": "Срез",
                "field": "Внутреннее поле",
            }.get,
            key="h3_mode",
        )
        point, plane, index = None, "xy", None
        if mode == "point":
            coordinates = []
            for axis in "xyz":
                key = "h3_" + axis
                if key in st.session_state and st.session_state[key] > n:
                    st.session_state[key] = n
                coordinates.append(
                    st.number_input(
                        "Узел " + axis, min_value=0, max_value=n, value=n // 2, key=key
                    )
                )
            point = tuple(coordinates)
        elif mode == "slice":
            plane = st.selectbox("Плоскость", ("xy", "xz", "yz"), key="h3_plane")
            if "h3_index" in st.session_state and st.session_state["h3_index"] > n:
                st.session_state["h3_index"] = n
            index = st.number_input(
                "Индекс среза", min_value=0, max_value=n, value=n // 2, key="h3_index"
            )
        newton = st.checkbox(
            "Независимая проверка Ньютоном (n ≤ 16)", value=False, key="h3_newton"
        )
        with st.expander("Параметры Пикара и прогонок", key="h3_solver_parameters"):
            tolerance = st.select_slider(
                "Допуск нелинейной невязки",
                (1e-3, 1e-5, 1e-7, 1e-9),
                value=1e-7,
                key="h3_tol",
            )
            change = st.select_slider(
                "Допуск изменения поля",
                (1e-3, 1e-5, 1e-7, 1e-9),
                value=1e-7,
                key="h3_change",
            )
            linear = st.select_slider(
                "Линейный допуск", (1e-8, 1e-10, 1e-12), value=1e-10, key="h3_linear"
            )
            relaxation = st.slider(
                "Релаксация",
                min_value=0.1,
                max_value=1.0,
                value=1.0,
                step=0.1,
                key="h3_relaxation",
            )
            outer = st.number_input(
                "Максимум итераций Пикара",
                min_value=1,
                max_value=1000,
                value=100,
                key="h3_outer",
            )
            inner = st.number_input(
                "Максимум циклов прогонок",
                min_value=1,
                max_value=100000,
                value=20000,
                key="h3_inner",
            )
            seconds = st.number_input(
                "Лимит секунд на полный запуск",
                min_value=0.01,
                max_value=3600.0,
                value=60.0,
                key="h3_seconds",
            )
            memory = st.number_input(
                "Лимит рабочей памяти, МиБ",
                min_value=1,
                max_value=4096,
                value=256,
                key="h3_memory",
            )
        run = st.button(
            "Рассчитать 3D",
            type="primary",
            key="h3_run",
            on_click=select_section,
            args=("h3_sections", "Сравнение"),
        )
    config = HeatConfig3D(
        case=case,
        n=n,
        alpha=alpha,
        dt=dt,
        T=final,
        mode=mode,
        point=point,
        plane=plane,
        index=index,
        newton=newton,
    )
    settings = HeatSettings3D(
        tolerance=tolerance,
        change_tolerance=change,
        linear_tolerance=linear,
        relaxation=relaxation,
        max_outer=outer,
        max_inner=inner,
        max_seconds=seconds,
        max_workspace_bytes=memory * 1024**2,
    )
    current = dict(configuration=asdict(config), solver_settings=asdict(settings))
    with st.expander("Уравнение, эталон и смысл ошибки", key="h3_explanation"):
        st.latex(
            r"u_t=\nabla\cdot((1+\alpha u^2)\nabla u)+F(x,y,z,t),\quad (x,y,z)\in(0,1)^3"
        )
        if case == "sine":
            st.latex(r"u^*(x,y,z,t)=1+e^{-t}\sin(\pi x)\sin(\pi y)\sin(\pi z)")
        else:
            st.latex(r"u^*(x,y,z,t)=1+t+\frac{x+2y+3z+xy+xz+yz+x^2+y^2+z^2}{4}")
        st.latex(r"F=u_t^*-(1+\alpha(u^*)^2)\Delta u^*-2\alpha u^*|\nabla u^*|^2")
        st.latex(
            r"u(x,y,z,0)=u^*(x,y,z,0),\qquad u|_{\partial\Omega}=g=u^*|_{\partial\Omega}"
        )
        st.write(
            "u — температура, k(u)=1+αu² — теплопроводность, F — источник. "
            "Параметр α задаёт силу нелинейности; при α=0 уравнение линейное. "
            "u* — известное проверочное решение, используемое для оценки ошибки."
        )
        st.write(
            "На всех шести гранях заданы условия 1 рода. Источник и границы согласованы с выбранным аналитическим решением. Неявный Эйлер использует фактический последний шаг до T; коэффициенты заморожены на всю линейную серию Пикара."
        )
        st.write(
            "Ошибка в области Q — относительная L2-норма к аналитическому решению. В исследовании усредняются ошибки отдельных полных запусков. Независимый Ньютон — отдельная сеточная диагностика; невязка не заменяет ошибку к аналитическому решению."
        )
        st.write(
            "Даже запрос одной точки оплачивает все необходимые пространственные поля и временные слои. Оценка рабочих массивов и измеренный пик памяти различаются."
        )
    if run:
        bar = st.progress(0.0, text="Расчёт всех временных слоёв…")
        try:
            st.session_state["h3_comparison"] = compare_heat3d(
                config,
                settings,
                progress=lambda fraction, text: bar.progress(
                    float(fraction), text=text
                ),
            )
        except (ValueError, RuntimeError) as exc:
            st.error(str(exc))
        finally:
            bar.empty()
    comparison_tab, study_tab = st.tabs(
        ["Сравнение", "Исследование 1/5/10%"], key="h3_sections", on_change="rerun"
    )
    with comparison_tab:
        result = st.session_state.get("h3_comparison")
        if result is None:
            st.info("Выберите параметры и нажмите «Рассчитать 3D».")
        else:
            _saved_settings(result, current)
            row = result["rows"][0]
            st.write(
                METHOD_NAME + ": " + STATUS_LABELS.get(row["status"], row["status"])
            )
            if row["status"] != "completed":
                st.warning(row["reason"])
                if "failed_candidate" in result["arrays"]:
                    with st.expander(
                        "Последнее приближение при остановке", key="h3_failed"
                    ):
                        st.caption(
                            f"Кандидат слоя t={row['diagnostic_time']:g}; это не решение в T."
                        )
                        points = result["arrays"]["points_zyx"]
                        diagnostic = pd.DataFrame(points, columns=["z", "y", "x"])
                        diagnostic["Последнее приближение"] = result["arrays"][
                            "failed_candidate"
                        ][tuple(points.T)]
                        st.dataframe(diagnostic, hide_index=True, width="stretch")
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "Ошибка, %": _error_text(row),
                            "Абсолютная L2-ошибка": _error_text(row, absolute=True),
                            "Полное время, с": row["seconds"],
                            "Достигнутое время": row["completed_until"],
                            "Принятых слоёв": row["physical_steps"],
                            "Итераций Пикара": row["outer_iterations"],
                            "Циклов прогонок": row["inner_iterations"],
                            "Рабочие массивы, байт": row["workspace_bytes"],
                        }
                    ]
                ),
                hide_index=True,
                width="stretch",
            )
            diagnostic = result["grid_diagnostic"]
            st.caption(
                "Ньютон: "
                + STATUS_LABELS.get(diagnostic["status"], diagnostic["status"])
                + ". "
                + diagnostic.get("reason", "")
            )
            if "lod_vs_grid" in diagnostic:
                st.write(
                    "Ошибка ЛОМ к независимому сеточному решению, %:",
                    diagnostic["lod_vs_grid"]["relative_error_pct"],
                )
            if "lod" in result["arrays"]:
                arrays = result["arrays"]
                values = pd.DataFrame(arrays["points_zyx"], columns=["z", "y", "x"])
                values["ЛОМ"] = arrays["lod"]
                values["Аналитическое решение"] = arrays["analytic"]
                values["Абсолютная ошибка"] = abs(arrays["lod"] - arrays["analytic"])
                st.dataframe(values, hide_index=True, width="stretch")
                if len(values) > 1:
                    view = st.selectbox(
                        "Показать в области Q",
                        ("ЛОМ", "Аналитическое решение", "Абсолютная ошибка"),
                        key="h3_view",
                    )
                    st.plotly_chart(
                        px.scatter_3d(values, x="x", y="y", z="z", color=view),
                        width="stretch",
                    )
            with st.expander("История невязок и итераций", key="h3_history"):
                st.dataframe(
                    pd.DataFrame(result["history"]), hide_index=True, width="stretch"
                )
            _downloads(result, "h3_comparison")
    with study_tab:
        st.caption(
            "Только "
            + METHOD_NAME
            + ". Подбор с фиксированным запасом → фиксация допуска → одна новая финальная серия. После её неудачи подбор не продолжается."
        )
        repeats = st.number_input(
            "Полных повторов в каждой серии",
            min_value=2,
            max_value=20,
            value=5,
            key="h3_repeats",
        )
        seed = st.number_input(
            "Seed протокола", min_value=0, max_value=2**32 - 1, value=42, key="h3_seed"
        )
        budgets = st.multiselect(
            "Допуски Пикара для подбора",
            (1e-3, 1e-5, 1e-7, 1e-9),
            default=[1e-3, 1e-5, 1e-7, 1e-9],
            key="h3_budgets",
        )
        margin = st.slider(
            "Доля цели на предварительной серии",
            min_value=0.1,
            max_value=0.9,
            value=0.8,
            step=0.1,
            key="h3_margin",
        )
        study_config = replace(config, newton=False)
        study_parameters = dict(
            repeats=repeats,
            seed=seed,
            budgets=sorted(budgets, reverse=True),
            calibration_fraction=margin,
        )
        if st.button(
            "Проверить 1/5/10% · 3D",
            disabled=not budgets,
            key="h3_study_run",
            on_click=select_section,
            args=("h3_sections", "Исследование 1/5/10%"),
        ):
            bar = st.progress(0.0, text="Подбор и независимая финальная серия…")
            try:
                st.session_state["h3_study"] = study_heat3d(
                    study_config,
                    settings,
                    **study_parameters,
                    progress=lambda fraction, text: bar.progress(
                        float(fraction), text=text
                    ),
                )
            except (ValueError, RuntimeError) as exc:
                st.error(str(exc))
            finally:
                bar.empty()
        study = st.session_state.get("h3_study")
        if study is not None:
            _saved_settings(
                study,
                dict(
                    configuration=asdict(study_config), solver_settings=asdict(settings)
                ),
            )
            if any(
                study["metadata"][key] != value
                for key, value in study_parameters.items()
            ):
                st.info(
                    "Параметры исследования изменены; таблица и выгрузки относятся к сохранённой серии."
                )
            st.caption(
                f"Выполненная серия: R={study['requested_repeats']}, запас={study['calibration_fraction']:g}, seed={study['seed']}. Пути: неприменимо для детерминированного метода."
            )
            display = []
            for row in study["rows"]:
                display.append(
                    {
                        "Цель, %": row["target_pct"],
                        "Статус": STATUS_LABELS.get(row["status"], row["status"]),
                        "Допуск": row["budget"],
                        "Серия": {
                            "pilot": "Предварительная",
                            "final": "Финальная",
                            "absolute_diagnostic": "Абсолютная диагностика",
                        }[row["phase"]],
                        "Ошибка, %": _error_text(row),
                        "Абсолютная L2-ошибка": _error_text(row, absolute=True),
                        "Повторы": f"{row['completed_repeats']}/{row['requested_repeats']}",
                        "Время при достижении, с": (
                            f"{row['seconds']:.6g}"
                            if row["seconds"] is not None
                            else "не подтверждено"
                        ),
                        "Затрачено на серию, с": row["elapsed_seconds"],
                        "Причина": row["reason"],
                    }
                )
            st.dataframe(pd.DataFrame(display), hide_index=True, width="stretch")
            st.caption(
                f"Подбор: {study['pilot_seconds']:.4g} с; финальные серии: {study['final_seconds']:.4g} с. Неподтверждённые цели не дают коэффициент ускорения."
            )
            attempts = pd.DataFrame(study["trials"])
            if not attempts.empty:
                st.plotly_chart(
                    px.scatter(
                        attempts,
                        x="seconds",
                        y="relative_error_pct",
                        color="phase",
                        hover_data=["budget", "status", "repeat"],
                        labels={
                            "seconds": "Полное время запуска, с",
                            "relative_error_pct": "Ошибка отдельного запуска, %",
                        },
                    ),
                    width="stretch",
                )
            _downloads(study, "h3_study")
