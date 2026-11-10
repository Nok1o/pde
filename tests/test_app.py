from dataclasses import asdict, make_dataclass
from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_default_center_app_runs_without_debug_output():
    app = AppTest.from_file(
        str(Path(__file__).parents[1] / "app.py"), default_timeout=60
    ).run()
    assert not app.exception
    assert app.selectbox[1].value == "point"
    app.button(key="main_run").click().run()
    assert not app.exception
    exp = app.session_state["experiment"]
    assert exp.points.tolist() == [[12, 12]]
    assert len(exp.results) == 4
    assert len(app.dataframe) == 1
    columns = app.dataframe[0].value.columns
    assert "Значение u" in columns and "Отн. ошибка, %" in columns
    assert "Макс. ошибка" not in columns and "Средняя SE" not in columns
    assert (
        len(
            next(
                widget for widget in app.multiselect if widget.label == "Методы"
            ).options
        )
        == 4
    )
    assert not any("параллел" in slider.label.lower() for slider in app.slider)
    assert not any("Диагностика" in expander.label for expander in app.expander)
    assert not app.get("doc_string")
    assert any(
        widget.label == "Относительный допуск ЛОМ" for widget in app.select_slider
    )
    # A code reload can replace the Config class while preserving session values.
    values = asdict(exp.config)
    PreviousConfig = make_dataclass("PreviousConfig", [(key, object) for key in values])
    exp.config = PreviousConfig(**values)
    app.run()
    assert not app.exception
    assert not any("Параметры изменены" in item.value for item in app.info)
    # Old calculations must not be relabelled as satisfying the new stopping rule.
    exp.results["lod_serial"].extras.pop("stopping_rule")
    app.session_state["archive"] = b"old export"
    app.session_state["benchmark_rows"] = [{"method": "lod_serial"}]
    app.run()
    assert not app.exception and len(app.dataframe) == 0
    assert any("Критерий остановки ЛОМ изменён" in item.value for item in app.info)
    assert (
        "archive" not in app.session_state and "benchmark_rows" not in app.session_state
    )


def test_research_runs_without_main_calculation_and_detects_changed_settings():
    app = AppTest.from_file(
        str(Path(__file__).parents[1] / "app.py"), default_timeout=60
    ).run()
    assert not app.exception
    assert "Исследование" in [tab.label for tab in app.tabs]
    app.select_slider(key="research_n").set_value(8)
    app.number_input(key="research_repeats").set_value(2)
    app.select_slider(key="research_max_walks").set_value(64)
    app.run()
    app.button(key="research_run").click().run()
    assert not app.exception
    assert "experiment" not in app.session_state
    study = app.session_state["research_result"]
    assert study.settings.config.n == 8
    assert study.settings.config.point == (4, 4)
    assert study.settings.repeats == 2
    table = app.dataframe[0].value
    assert table.shape == (3, 5)
    assert table.iloc[:, 0].tolist() == [1, 5, 10]
    app.number_input(key="research_repeats").set_value(3).run()
    assert not app.exception
    assert any("Настройки исследования изменены" in item.value for item in app.info)
