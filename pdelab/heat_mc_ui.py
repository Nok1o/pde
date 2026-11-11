"""Queries and accuracy studies for standalone space-time Monte Carlo."""

import json
from dataclasses import asdict

import numpy as np
import pandas as pd
import streamlit as st

from .heat import make_heat_problem, solve_heat, time_grid
from .heat_mc import heat_mc_step, solve_heat_mc
from .heat_mc_study import VALIDATION_POLICY, heat_mc_study
from .heat_mc_views import heat_mc_archive
from .study_policy import render_accuracy_results


def _limit(settings, walks, diagnostic=False):
    _, steps = time_grid(settings.dt, settings.T)
    if diagnostic:
        steps = steps[:1]
    expected = float(np.sum(1 + 4 * settings.a2 * settings.n**2 * steps)) * walks
    if expected > 50_000_000:
        raise ValueError(
            "Интерактивный лимит: 50 млн ожидаемых переходов. Уменьшите T, сетку или число путей; для больших расчётов используйте CLI."
        )


def render_heat_mc(settings):
    st.subheader("Монте-Карло в пространстве и времени")
    st.caption(
        "Самостоятельные траектории идут назад до μ или границы. Источник и граница всех нужных времён входят в подготовку и полное время."
    )
    diagnostic = st.checkbox(
        "Диагностика только первого шага с заданным μ", key="heat_mc_diagnostic"
    )
    x = st.slider("Точка MC по x", 1, settings.n - 1, settings.n // 2, key="heat_mc_x")
    y = st.slider("Точка MC по z", 1, settings.n - 1, settings.n // 2, key="heat_mc_y")
    walks = st.select_slider(
        "Полных траекторий на точку",
        [64, 256, 1024, 4096, 16384, 65536],
        value=4096,
        key="heat_mc_walks",
    )
    seed = st.number_input(
        "Seed MC по времени",
        min_value=0,
        max_value=2**32 - 1,
        value=42,
        step=1,
        key="heat_mc_seed",
    )
    current = dict(
        **asdict(settings), x=x, y=y, walks=walks, seed=seed, diagnostic=diagnostic
    )
    if st.button("Рассчитать MC по времени", key="heat_mc_run"):
        try:
            _limit(settings, walks, diagnostic)
            p = make_heat_problem(settings.preset, settings.n, settings.a2)
            with st.spinner("Полные пространственно-временные траектории…"):
                if diagnostic:
                    t = min(settings.dt, settings.T)
                    if t <= 0:
                        raise ValueError("Для диагностики шага нужно T>0")
                    result = heat_mc_step(p, p.initial, t, t, [[y, x]], walks, seed)
                else:
                    result = solve_heat_mc(
                        p, settings.dt, settings.T, [[y, x]], walks, seed
                    )
                direct = solve_heat(
                    p,
                    min(settings.dt, result.times[-1]) if diagnostic else settings.dt,
                    result.times[-1],
                    method="direct",
                )
            st.session_state["heat_mc_query"] = (result, p, current, direct)
        except (ValueError, RuntimeError) as exc:
            st.error(str(exc))
    stored = st.session_state.get("heat_mc_query")
    if stored is None:
        return
    result, p, previous, direct = stored
    if previous != current:
        st.info("Настройки MC изменены. Выполните новый расчёт.")
    ref = float(direct.fields[-1, result.points[0, 0], result.points[0, 1]])
    row = result.rows()[0]
    row["direct_reference"] = ref
    row["error_to_grid"] = row["value"] - ref
    if p.exact:
        row["analytic_reference"] = float(
            p.evaluate(p.exact, result.times[-1])[tuple(result.points[0])]
        )
    st.dataframe(pd.DataFrame([row]), hide_index=True)
    st.write(
        f"Полное время MC: {result.seconds:.4g} с; подготовка: {result.preparation_seconds:.4g} с; траектории: {result.path_seconds:.4g} с. Прямой расчёт до того же t: {direct.seconds:.4g} с."
    )
    st.write(
        f"Переходы: {result.spatial_transitions} пространственных и {result.temporal_transitions} временных. Численные буферы: {result.memory_bytes / 2**20:.3g} МиБ (без памяти процесса и JIT)."
    )
    if result.diagnostic_step:
        st.info(
            "Диагностика первого шага: μ задано. Этот замер не является расчётом всей истории до выбранного T."
        )
    st.download_button(
        "Скачать MC по времени ZIP",
        heat_mc_archive(result, p, previous["dt"]),
        "heat-mc-results.zip",
        "application/zip",
        key="heat_mc_download",
        on_click="ignore",
    )


def render_heat_mc_study(settings):
    st.subheader("Время достижения 1%, 5% и 10%")
    st.caption(
        "Сравниваются полные расчёты до T. Порог проверяется по средней относительной ошибке независимых запусков, затем новой серией; подбор и эталон исключены из табличного времени."
    )
    x = st.slider(
        "Точка исследования по x",
        1,
        settings.n - 1,
        settings.n // 2,
        key="heat_study_x",
    )
    y = st.slider(
        "Точка исследования по z",
        1,
        settings.n - 1,
        settings.n // 2,
        key="heat_study_y",
    )
    repeats = st.number_input(
        "Независимых запусков в серии",
        min_value=2,
        max_value=20,
        value=3,
        step=1,
        key="heat_study_repeats",
    )
    max_walks = st.select_slider(
        "Лимит путей исследования",
        [64, 256, 1024, 4096, 16384, 65536],
        value=4096,
        key="heat_study_walks",
    )
    seconds = st.number_input(
        "Лимит каждой фазы на метод, с",
        min_value=1.0,
        max_value=300.0,
        value=10.0,
        key="heat_study_seconds",
    )
    current = dict(
        **asdict(settings),
        x=x,
        y=y,
        repeats=repeats,
        max_walks=max_walks,
        seconds=seconds,
    )
    if st.button("Исследовать MC по времени", key="heat_study_run"):
        try:
            _limit(settings, max_walks)
            with st.spinner("Подбор и независимая проверка…"):
                study = heat_mc_study(
                    make_heat_problem(settings.preset, settings.n, settings.a2),
                    settings.dt,
                    settings.T,
                    (y, x),
                    repeats,
                    max_walks,
                    seconds_per_method=seconds,
                )
            st.session_state["heat_mc_study"] = (study, current)
        except (ValueError, RuntimeError) as exc:
            st.error(str(exc))
    stored = st.session_state.get("heat_mc_study")
    if stored is None:
        return
    study, previous = stored
    if study["settings"].get("validation_policy") != VALIDATION_POLICY:
        st.session_state.pop("heat_mc_study", None)
        st.info("Протокол проверки точности изменён. Выполните новую серию.")
        return
    if current != previous:
        st.info("Настройки исследования изменены. Выполните новую серию.")
    table = pd.DataFrame(study["rows"])
    render_accuracy_results(
        study["rows"], {"lod": "ЛОМ", "mc": "Монте-Карло"}, key="heat_study_results"
    )
    with st.expander("Статусы и отдельные замеры", key="heat_mc_ui_expander_189"):
        st.dataframe(table, hide_index=True)
        st.dataframe(pd.DataFrame(study["trials"]), hide_index=True)
    st.download_button(
        "Скачать исследование MC (JSON)",
        json.dumps(study, ensure_ascii=False, indent=2),
        "heat-mc-study.json",
        "application/json",
        key="heat_study_download",
        on_click="ignore",
    )
