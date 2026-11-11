"""Physical time, outer Picard and inner linear solves are separate controls."""

from dataclasses import asdict

import pandas as pd
import plotly.express as px
import streamlit as st

from .nonlinear import NonlinearSettings, make_nonlinear_problem
from .nonlinear_io import (
    comparison_archive,
    csv_bytes,
    field_figure,
    history_rows,
    json_bytes,
    residual_figure,
    time_figure,
)
from .nonlinear_study import compare_nonlinear, nonlinear_targets
from .study_policy import render_accuracy_results
from .ui_state import select_section


def _problem_info(alpha):
    with st.expander("Формулы и описание", expanded=False, key="nonlinear_formulas"):
        st.markdown("**1. Исходное уравнение**")
        st.latex(
            r"C(u)\frac{\partial u}{\partial t}"
            r"=\frac{\partial}{\partial x}\left(k(u)\frac{\partial u}{\partial x}\right)"
            r"+\frac{\partial}{\partial y}\left(k(u)\frac{\partial u}{\partial y}\right)"
            r"+f(x,y,t)"
        )
        st.write(
            "u — температура; C(u) — коэффициент теплоёмкости; "
            "k(u) — теплопроводность; f — заданный источник тепла."
        )

        st.markdown("**2. Какие коэффициенты принимаем**")
        st.latex(r"C(u)=1,\qquad k(u)=1+\alpha u^2")
        st.write(f"Сейчас α = {alpha:g}.")
        st.write(
            "При α = 0 получаем k(u) = 1: задача становится линейной."
            if alpha == 0
            else "Теплопроводность зависит от температуры, поэтому задача нелинейная."
        )

        st.markdown("**3. Как задаём источник f**")
        st.write("Для проверки методов выбираем известное точное решение:")
        st.latex(r"u_*(x,y,t)=1+e^{-t}\sin(\pi x)\sin(\pi y)")
        st.write(
            "Подбираем источник так, чтобы это решение удовлетворяло уравнению. "
            "В формуле источника u_* — известная функция, а не искомое численное поле."
        )
        st.latex(
            r"f=\frac{\partial u_*}{\partial t}"
            r"-\nabla\!\cdot\!\left(k(u_*)\nabla u_*\right)"
        )
        st.write("После вычисления производных:")
        st.latex(
            r"\begin{aligned}f(x,y,t)={}&-(u_*-1)"
            r"+2\pi^2 k(u_*)(u_*-1)\\"
            r"&-2\alpha u_*\left["
            r"\left(\frac{\partial u_*}{\partial x}\right)^2"
            r"+\left(\frac{\partial u_*}{\partial y}\right)^2\right].\end{aligned}"
        )

        st.markdown("**4. Итоговое уравнение**")
        st.write("Подставляем C(u) и k(u) в исходное уравнение:")
        st.latex(
            r"\begin{aligned}\frac{\partial u}{\partial t}"
            r"={}&\frac{\partial}{\partial x}"
            r"\left((1+\alpha u^2)\frac{\partial u}{\partial x}\right)\\"
            r"&+\frac{\partial}{\partial y}"
            r"\left((1+\alpha u^2)\frac{\partial u}{\partial y}\right)+f(x,y,t)."
            r"\end{aligned}"
        )

        st.markdown("**5. Область, начальное и граничное условия**")
        st.latex(r"\Omega=(0,1)^2,\qquad t\in[0,T]")
        st.write("В начальный момент:")
        st.latex(r"u(x,y,0)=1+\sin(\pi x)\sin(\pi y)")
        st.write("На всех четырёх сторонах квадрата поддерживаем температуру 1:")
        st.latex(r"u(0,y,t)=u(1,y,t)=u(x,0,t)=u(x,1,t)=1")

    with st.expander(
        "Как решаем: время → Пикар → ЛОМ или MC", key="nonlinear_method_formulas"
    ):
        st.write(
            "Сначала переходим к следующему моменту времени. "
            "На каждом временном слое уточняем поле итерациями Пикара; "
            "линейную задачу внутри каждой итерации решаем прогонками или MC."
        )
        st.write(
            "uᵐ — поле на предыдущем временном слое; vʲ — текущее приближение "
            "на новом слое. Фиксируем проводимость по vʲ:"
        )
        st.latex(r"k^{(j)}=1+\alpha(v^j)^2")
        st.write("Решаем линейную сеточную задачу для нового приближения:")
        st.latex(
            r"\left(\frac{I}{\Delta t}-\nabla_h\!\cdot k^{(j)}\nabla_h\right)"
            r"\widetilde v^{j+1}=\frac{u^m}{\Delta t}+f^{m+1}"
        )
        st.caption("I — единичный оператор; ∇ₕ обозначает производные на сетке.")
        st.write("Обновляем поле с релаксацией ω и повторяем до сходимости:")
        st.latex(r"v^{j+1}=(1-\omega)v^j+\omega\widetilde v^{j+1}")
        st.write(
            "Ньютон + прямой решатель служит отдельным сеточным эталоном. "
            "Основные сравниваемые варианты — Пикар + ЛОМ и Пикар + MC."
        )


def render_nonlinear_app():
    st.title("Нелинейная теплопроводность")
    with st.sidebar:
        n = st.select_slider(
            "Интервалов по оси", [4, 6, 8, 12, 16, 24, 32], value=8, key="nl_n"
        )
        alpha = st.number_input(
            "α", min_value=0.0, max_value=100.0, value=0.5, key="nl_alpha"
        )
        dt = st.number_input(
            "Физический шаг Δt",
            min_value=0.0001,
            max_value=1.0,
            value=0.01,
            format="%.4f",
            key="nl_dt",
        )
        final = st.number_input(
            "Конечное время T", min_value=0.0, max_value=2.0, value=0.1, key="nl_T"
        )
        coordinates = []
        for axis in "xy":
            key = f"nl_{axis}"
            if key in st.session_state:
                st.session_state[key] = min(n, st.session_state[key])
            coordinates.append(
                st.number_input(
                    f"Узел {axis}", min_value=0, max_value=n, value=n // 2, key=key
                )
            )
        point = tuple(coordinates)
        repeats = st.number_input(
            "Полных повторов MC", min_value=2, max_value=20, value=4, key="nl_repeats"
        )
        with st.expander("Нелинейные итерации", key="nonlinear_ui_expander_57"):
            relaxation = st.slider(
                "Релаксация", 0.1, 1.0, 0.7, 0.1, key="nl_relaxation"
            )
            tol = st.select_slider(
                "Допуск Picard + ЛОМ",
                [1e-3, 1e-5, 1e-7, 1e-9],
                value=1e-7,
                key="nl_tol",
            )
            mc_tol = st.select_slider(
                "Допуск Picard + MC",
                [0.03, 0.01, 0.003, 0.001, 0.0001],
                value=0.001,
                key="nl_mc_tol",
            )
            outer = st.number_input(
                "Максимум внешних итераций",
                min_value=1,
                max_value=100,
                value=12,
                key="nl_outer",
            )
        with st.expander("Линейный расчёт и бюджет MC", key="nonlinear_ui_expander_80"):
            linear_tol = st.select_slider(
                "Линейный допуск ЛОМ",
                [1e-8, 1e-10, 1e-12],
                value=1e-10,
                key="nl_linear_tol",
            )
            walks = st.select_slider(
                "Начальный бюджет на узел",
                [8, 32, 128, 512, 2048],
                value=128,
                key="nl_walks",
            )
            maximum = st.select_slider(
                "Максимальный бюджет на узел",
                [128, 512, 2048, 8192],
                value=2048,
                key="nl_max_walks",
            )
            schedule = st.selectbox(
                "Рост бюджета",
                ["constant", "linear", "geometric"],
                index=2,
                format_func={
                    "constant": "Постоянный",
                    "linear": "Линейный",
                    "geometric": "Удвоение",
                }.get,
                key="nl_schedule",
            )
            seed = st.number_input(
                "Seed", min_value=0, max_value=2**32 - 1, value=42, key="nl_seed"
            )
        run = st.button(
            "Рассчитать",
            type="primary",
            key="nl_run",
            on_click=select_section,
            args=("nonlinear_sections", "Сравнение"),
        )
    lod_settings = NonlinearSettings(
        tolerance=tol,
        change_tolerance=tol,
        linear_tolerance=linear_tol,
        relaxation=relaxation,
        max_outer=outer,
    )
    mc_settings = NonlinearSettings(
        tolerance=mc_tol,
        change_tolerance=mc_tol,
        relaxation=relaxation,
        max_outer=outer,
        walks=walks,
        max_walks=maximum,
        schedule=schedule,
        seed=seed,
        max_total_walks=2_000_000,
    )
    settings = dict(
        n=n,
        alpha=alpha,
        dt=dt,
        T=final,
        point=point,
        repeats=repeats,
        lod=asdict(lod_settings),
        mc=asdict(mc_settings),
    )
    _problem_info(alpha)
    try:
        lod_settings.validate()
        mc_settings.validate()
    except ValueError as exc:
        st.error(str(exc))
        return
    if run:
        try:
            with st.spinner("Временные слои, Picard и независимые цепочки MC…"):
                result = compare_nonlinear(
                    make_nonlinear_problem(n, alpha),
                    dt,
                    final,
                    point,
                    repeats,
                    lod_settings,
                    mc_settings,
                )
            st.session_state["nl_result"] = result
            st.session_state["nl_settings"] = settings
            st.session_state.pop("nl_archive", None)
        except (ValueError, RuntimeError) as exc:
            st.error(str(exc))
    comparison_tab, iterations_tab, research_tab = st.tabs(
        ["Сравнение", "Итерации", "Исследование 1/5/10%"],
        key="nonlinear_sections",
        on_change="rerun",
    )
    result = st.session_state.get("nl_result")
    with comparison_tab:
        if result is not None:
            if st.session_state["nl_settings"] != settings:
                st.info("Настройки изменены. Запустите расчёт заново.")
            labels = dict(
                method="Метод",
                completed="Завершён",
                seconds="Время, с",
                value="u",
                rmse="RMSE",
                relative_error_pct="Отн. ошибка, %",
            )
            st.dataframe(
                pd.DataFrame(result.metrics)[list(labels)].rename(columns=labels),
                hide_index=True,
                width="stretch",
            )
            for calculation in [result.deterministic, *result.stochastic]:
                if not calculation.completed:
                    st.warning(
                        f"{calculation.method}, seed={calculation.settings.seed}: {calculation.reason}"
                    )
            st.plotly_chart(time_figure(result), width="stretch")
            st.plotly_chart(
                field_figure(result, st.toggle("Показать ошибку", key="nl_error")),
                width="stretch",
            )
            with st.expander(
                "Статистика полных цепочек MC", key="nonlinear_ui_expander_208"
            ):
                st.dataframe(
                    pd.DataFrame([result.metrics[1]])[
                        [
                            "completed_runs",
                            "repeats",
                            "grid_bias",
                            "bias_ci_low",
                            "bias_ci_high",
                            "chain_std",
                            "conductivity_plugin_bias",
                        ]
                    ],
                    hide_index=True,
                )
            if st.button("Подготовить архив", key="nl_export"):
                st.session_state["nl_archive"] = comparison_archive(result)
            if "nl_archive" in st.session_state:
                st.download_button(
                    "Скачать ZIP",
                    st.session_state["nl_archive"],
                    "nonlinear.zip",
                    "application/zip",
                    on_click="ignore",
                )
    with iterations_tab:
        if result is not None:
            st.plotly_chart(residual_figure(result), width="stretch")
            st.dataframe(
                pd.DataFrame(history_rows(result)), hide_index=True, width="stretch"
            )
    with research_tab:
        budgets = st.multiselect(
            "Проверяемые начальные бюджеты MC",
            [8, 32, 128, 512, 2048],
            default=[32, 128, 512],
            key="nl_budgets",
        )
        target_settings = {**settings, "budgets": budgets}
        if st.button("Проверить 1/5/10%", key="nl_targets_run", disabled=not budgets):
            try:
                with st.spinner("Подбор бюджета и независимая проверка…"):
                    rows, evidence = nonlinear_targets(
                        make_nonlinear_problem(n, alpha),
                        dt,
                        final,
                        point,
                        repeats,
                        budgets,
                        seed,
                        settings=mc_settings,
                        deterministic=lod_settings,
                    )
                st.session_state["nl_targets"] = (target_settings, rows, evidence)
            except (ValueError, RuntimeError) as exc:
                st.error(str(exc))
        saved = st.session_state.get("nl_targets")
        if saved is not None:
            if saved[0] != target_settings:
                st.info("Показано исследование с прежними настройками.")
            pilots = pd.DataFrame(
                [
                    {**metric, "N": item["settings"]["walks"]}
                    for item in saved[2]["pilots"]
                    for metric in item["metrics"]
                ]
            )
            st.plotly_chart(
                px.scatter(
                    pilots,
                    x="seconds",
                    y="relative_error_pct",
                    color="method",
                    hover_data=["N", "completed"],
                    labels={
                        "seconds": "Полное время, с",
                        "relative_error_pct": "Отн. ошибка, %",
                        "method": "Метод",
                    },
                ),
                width="stretch",
            )
            render_accuracy_results(
                saved[1],
                {"picard": "Picard + ЛОМ", "mc": "Picard + MC"},
                key="nl_study_results",
            )
            st.download_button(
                "Скачать таблицу",
                csv_bytes(saved[1]),
                "nonlinear-targets.csv",
                "text/csv",
                on_click="ignore",
            )
            st.download_button(
                "Скачать проверку бюджетов",
                json_bytes(dict(settings=saved[0], evidence=saved[2])),
                "nonlinear-targets.json",
                "application/json",
                on_click="ignore",
            )
