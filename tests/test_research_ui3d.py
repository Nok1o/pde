import pytest
from streamlit.testing.v1 import AppTest


def research_app():
    from pdelab.experiment3d import Config3D
    from pdelab.ui3d import _research_tab

    _research_tab(Config3D(problem="constant", n=8))


@pytest.mark.parametrize("mode", ["patch", "slice", "field"])
def test_region_selection_runs_and_preserves_results(mode):
    app = AppTest.from_function(research_app, default_timeout=60).run()
    app.selectbox(key="study3d_mode").set_value(mode).run()
    if mode == "slice":
        app.selectbox(key="study3d_plane").set_value("yz").run()
        app.slider(key="study3d_index_8").set_value(3).run()
    app.number_input(key="study3d_repeats").set_value(2)
    app.select_slider(key="study3d_walks").set_value(64)
    app.button(key="study3d_run").click().run()
    assert not app.exception
    assert not app.error
    study = app.session_state["study3d_result"]
    assert study.settings.config.mode == mode
    if mode == "slice":
        assert study.settings.config.plane == "yz"
        assert study.settings.config.index == 3
    assert all(row["seconds"] is not None for row in study.rows)
    app.run()
    assert not app.exception
    assert app.selectbox(key="study3d_mode").value == mode
    assert app.session_state["study3d_result"].rows == study.rows
