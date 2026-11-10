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
        with st.expander("Нелинейные итерации"):
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
        with st.expander("Линейный расчёт и бюджет MC"):
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
        run = st.button("Рассчитать", type="primary", key="nl_run")
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
    with st.expander("Формулы и описание", expanded=True):
        st.latex(
            r"u_t=\nabla\cdot((1+\alpha u^2)\nabla u)+F,\quad C=1,\quad u|_{\partial\Omega}=1"
        )
        st.latex(r"u_*=1+e^{-t}\sin(\pi x)\sin(\pi y),\qquad u_0=u_*(x,y,0)")
        st.latex(r"F=-(u_*-1)+2\pi^2(1+\alpha u_*^2)(u_*-1)-2\alpha u_*|\nabla u_*|^2")
        st.latex(
            r"\left(\frac{I}{\Delta t}-\nabla_h\cdot k(v^j)\nabla_h\right)\widetilde v^{j+1}=\frac{u^m}{\Delta t}+F^{m+1},\quad v^{j+1}=(1-\omega)v^j+\omega\widetilde v^{j+1}"
        )
        st.caption(
            "Каждый повтор MC считает всю историю и все внутренние узлы. Newton — независимая проверка сеточной системы."
        )
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
        ["Сравнение", "Итерации", "Исследование 1/5/10%"]
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
            with st.expander("Статистика полных цепочек MC"):
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
            table = pd.DataFrame(saved[1])
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
            st.dataframe(
                table.pivot(
                    index="target_pct", columns="method", values="seconds"
                ).rename_axis("Точность, %"),
                width="stretch",
            )
            st.dataframe(
                table[
                    ["target_pct", "method", "status", "relative_error_pct", "walks"]
                ],
                hide_index=True,
            )
            st.download_button(
                "Скачать таблицу",
                csv_bytes(saved[1]),
                "nonlinear-targets.csv",
                "text/csv",
            )
            st.download_button(
                "Скачать проверку бюджетов",
                json_bytes(dict(settings=saved[0], evidence=saved[2])),
                "nonlinear-targets.json",
                "application/json",
            )
