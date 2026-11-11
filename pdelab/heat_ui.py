"""Heat mode UI: physical time and inner iterations have separate controls."""

from dataclasses import asdict, dataclass

import pandas as pd
import streamlit as st

from .heat import HEAT_PRESETS, make_heat_problem, solve_heat
from .heat_views import heat_archive, heat_profile, heat_slice
from .ui_state import select_section


@dataclass
class HeatSettings:
    preset: str
    n: int
    a2: float
    dt: float
    T: float
    tol: float


def render_heat_app():
    st.title("Теплопроводность · физическое время")
    st.caption(
        "Лекция 15: u — непрерывное поле, y — сеточное; a — температуропроводность, τ — шаг времени, f — источник."
    )
    st.latex(r"u_t=a\Delta u+f,\qquad (I-\tau\,a L_h)y^{m+1}=y^m+\tau f^{m+1}")
    with st.sidebar:
        preset = st.selectbox(
            "Начальное поле и условия",
            list(HEAT_PRESETS),
            format_func=HEAT_PRESETS.get,
            key="heat_preset",
        )
        n = st.select_slider(
            "Интервалов по оси", [8, 12, 16, 24, 32, 48, 64], value=24, key="heat_n"
        )
        a2 = st.number_input(
            "Коэффициент a", min_value=0.001, max_value=100.0, value=1.0, key="heat_a2"
        )
        dt = st.number_input(
            "Шаг физического времени τ",
            min_value=0.0001,
            max_value=10.0,
            value=0.01,
            format="%.4f",
            key="heat_dt",
        )
        T = st.number_input(
            "Конечное физическое время T",
            min_value=0.0,
            max_value=10.0,
            value=0.2,
            key="heat_T",
        )
        tol = st.select_slider(
            "Допуск внутреннего линейного решения",
            [1e-6, 1e-8, 1e-10, 1e-12],
            value=1e-10,
            key="heat_tol",
        )
        run = st.button(
            "Рассчитать теплопроводность",
            type="primary",
            key="heat_run",
            on_click=select_section,
            args=("heat_sections", "Сеточный расчёт"),
        )
    settings = HeatSettings(preset, n, a2, dt, T, tol)
    if preset == "decay":
        st.latex(r"u_*=e^{-2\pi^2 at}\sin(\pi x)\sin(\pi z),\quad f=0,\quad \mu=0")
    else:
        st.latex("u_*=1+x+z+t(1+x)+e^t x(1-x)z(1-z)")
        st.latex(
            r"f=1+x+e^t[x(1-x)z(1-z)+2a(x(1-x)+z(1-z))],\quad \mu=u_*|_{\partial\Omega}"
        )
    st.caption(
        "Начальное поле μ(x,z)=u*(x,z,0). Итерации линейного решателя выполняются внутри каждого временного слоя."
    )
    from .heat_mc_ui import render_heat_mc, render_heat_mc_study

    tabs = st.tabs(
        ["Сеточный расчёт", "Монте-Карло", "Исследование по времени"],
        key="heat_sections",
        on_change="rerun",
    )
    with tabs[1]:
        render_heat_mc(settings)
    with tabs[2]:
        render_heat_mc_study(settings)
    with tabs[0]:
        _render_grid(settings, run)


def _render_grid(settings, run):
    preset, n, a2, dt, T, tol = (
        settings.preset,
        settings.n,
        settings.a2,
        settings.dt,
        settings.T,
        settings.tol,
    )
    if run:
        try:
            with st.spinner("Расчёт временных слоёв и независимая проверка…"):
                problem = make_heat_problem(preset, n, a2)
                result = solve_heat(problem, dt, T, tol=tol)
                direct = solve_heat(problem, dt, T, method="direct", tol=tol)
            st.session_state["heat_result"] = result
            st.session_state["heat_direct"] = direct
            st.session_state["heat_settings"] = asdict(settings)
            st.session_state.pop("heat_archive", None)
            st.session_state.pop("heat_slice", None)
        except (ValueError, RuntimeError) as exc:
            st.error(str(exc))
    result = st.session_state.get("heat_result")
    if result is None:
        st.info("Задайте физическое время и нажмите «Рассчитать теплопроводность».")
        return
    if st.session_state["heat_settings"] != asdict(settings):
        st.info("Параметры изменены. Выполните новый расчёт.")
    direct = st.session_state["heat_direct"]
    st.write(
        f"Временных шагов: {len(result.times) - 1}; физическое время: {result.times[-1]:g}; расчёт: {result.seconds:.3g} с."
    )
    rows = result.rows()
    for row, field, ref in zip(rows, result.fields, direct.fields):
        row["grid_error_inf"] = float(abs(field - ref).max())
    st.dataframe(
        pd.DataFrame(rows).rename(
            columns={
                "time_index": "Номер временного слоя",
                "t": "Физическое время t",
                "dt": "Шаг τ",
                "iterations": "Итераций внутри слоя",
                "relative_residual": "Относительная невязка слоя",
                "center": "u в центре",
                "rmse_exact": "RMSE к точному решению",
                "grid_error_inf": "Ошибка к прямому решателю слоя",
            }
        ),
        hide_index=True,
    )
    st.plotly_chart(heat_profile(result), width="stretch")
    index = st.select_slider(
        "Срез по физическому времени",
        options=list(range(len(result.times))),
        value=len(result.times) - 1,
        format_func=lambda i: f"t={result.times[i]:.8g}",
        key="heat_slice",
    )
    st.plotly_chart(heat_slice(result, index), width="stretch")
    st.plotly_chart(heat_slice(result, index, error=True), width="stretch")
    if st.button("Подготовить экспорт теплопроводности", key="heat_export"):
        st.session_state["heat_archive"] = heat_archive(result)
    if "heat_archive" in st.session_state:
        st.download_button(
            "Скачать теплопроводность ZIP",
            st.session_state["heat_archive"],
            "heat-results.zip",
            "application/zip",
            on_click="ignore",
        )
