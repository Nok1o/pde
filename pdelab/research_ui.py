"""Вкладка сравнения времени при одинаковой относительной точности."""

from dataclasses import asdict

import pandas as pd
import streamlit as st

from .experiment import Config
from .research import StudySettings, accuracy_study, timing_rows


def render_research_tab(config):
    grid_column, repeats_column = st.columns(2)
    with grid_column:
        n = st.select_slider(
            "Сетка исследования: интервалов по оси",
            options=[8, 12, 16, 24, 32, 48, 64, 96, 128],
            value=config.n,
            key="research_n",
        )
    with repeats_column:
        repeats = st.number_input(
            "Число усреднений",
            min_value=2,
            max_value=100,
            value=5,
            step=1,
            help="Число независимых запусков в каждой серии. Усредняется время одного запуска; точность проверяется по RMS ошибок отдельных запусков.",
            key="research_repeats",
        )

    previous_n = st.session_state.get("research_point_grid", n)
    if previous_n != n:
        for key in ("research_x", "research_y"):
            previous_index = st.session_state.get(key, previous_n // 2)
            st.session_state[key] = min(
                n - 1, max(1, round(previous_index * n / previous_n))
            )
    st.session_state["research_point_grid"] = n

    with st.expander("Точка и лимиты поиска"):
        x_column, y_column = st.columns(2)
        with x_column:
            x = st.slider("Узел x", 1, n - 1, n // 2, key="research_x")
        with y_column:
            y = st.slider("Узел y", 1, n - 1, n // 2, key="research_y")
        st.caption(f"Точка: ({x / n:.4g}, {y / n:.4g})")
        max_walks = st.select_slider(
            "Максимум путей на точку",
            options=[
                64,
                128,
                256,
                512,
                1024,
                2048,
                4096,
                8192,
                16384,
                32768,
                65536,
                131072,
                262144,
                524288,
            ],
            value=131072,
            key="research_max_walks",
        )
        seconds_per_method = st.number_input(
            "Лимит поиска на метод, с",
            min_value=1.0,
            max_value=1800.0,
            value=60.0,
            step=10.0,
            help="Проверяется между запусками решателя. Уже начатый запуск завершается полностью.",
            key="research_seconds",
        )

    point_config = Config(
        problem=config.problem,
        n=n,
        mode="point",
        point=(x, y),
        batches=config.batches,
        seed=config.seed,
        max_iter=config.max_iter,
        top=config.top,
        bottom=config.bottom,
        left=config.left,
        right=config.right,
        source=config.source,
    )
    settings = StudySettings(point_config, int(repeats), max_walks, seconds_per_method)

    if st.button("Запустить исследование", type="primary", key="research_run"):
        progress = st.progress(0.0, text="Подготовка…")
        try:
            study = accuracy_study(
                settings,
                progress=lambda fraction, label: progress.progress(
                    fraction, text=label
                ),
            )
            st.session_state["research_result"] = study
            st.session_state.pop("archive", None)
        except (ValueError, RuntimeError) as exc:
            st.error(str(exc))
        finally:
            progress.empty()

    study = st.session_state.get("research_result")
    if study is None:
        return
    if asdict(settings) != asdict(study.settings):
        st.info("Настройки исследования изменены. Запустите его заново.")

    table = pd.DataFrame(timing_rows(study))
    columns = {
        name: st.column_config.NumberColumn(
            format="%.6f",
            help="Среднее время одного запуска проверочной серии. Подбор бюджета и эталон исключены; подготовка слоёв включена.",
        )
        for name in table.columns[1:]
    }
    columns["Относительная ошибка, %"] = st.column_config.NumberColumn(
        format="%.0f%%",
        help="Порог относительной RMSE по независимым запускам, подтверждённый на новой серии.",
    )
    st.dataframe(table, hide_index=True, width="stretch", column_config=columns)
    if any(row["seconds"] is None for row in study.rows):
        st.caption("—: точность не достигнута в пределах поиска. Причина — в деталях.")
    st.download_button(
        "Скачать таблицу CSV",
        table.to_csv(index=False).encode("utf-8-sig"),
        "accuracy-timings.csv",
        "text/csv",
        key="research_table_csv",
    )

    with st.expander("Детали исследования"):
        saved = study.settings.config
        saved_x, saved_y = saved.point
        st.write(
            f"n={saved.n}, точка ({saved_x / saved.n:.4g}, {saved_y / saved.n:.4g}), "
            f"усреднений: {study.settings.repeats}."
        )
        st.write(f"Эталон: {study.reference_label}; u = {study.reference_value:.6g}.")
        st.latex(
            r"\varepsilon_{\mathrm{rel}} = \frac{100\%}{|u_*|}\sqrt{\frac{1}{R}\sum_{r=1}^{R}(u_r-u_*)^2}"
        )
        st.write(
            "Бюджет проходит две серии: подбор и независимую проверку. В таблице — среднее время одного проверочного запуска. Это экспериментальная оценка точности."
        )
        details = pd.DataFrame(study.rows)
        details["budget_kind"] = details["budget_kind"].map(
            {
                "relative_residual_tolerance": "Допуск невязки",
                "walks_per_start": "Путей на точку",
            }
        )
        st.dataframe(
            details[
                [
                    "target_pct",
                    "name",
                    "relative_error_pct",
                    "calibration_error_pct",
                    "budget",
                    "budget_kind",
                    "status",
                ]
            ].rename(
                columns={
                    "target_pct": "Цель, %",
                    "name": "Метод",
                    "relative_error_pct": "Ошибка проверки, %",
                    "calibration_error_pct": "Ошибка подбора, %",
                    "budget": "Бюджет",
                    "budget_kind": "Тип бюджета",
                    "status": "Статус",
                }
            ),
            hide_index=True,
            width="stretch",
        )
        st.download_button(
            "Скачать все замеры CSV",
            pd.DataFrame(study.trials).to_csv(index=False).encode("utf-8-sig"),
            "accuracy-trials.csv",
            "text/csv",
            key="research_trials_csv",
        )
