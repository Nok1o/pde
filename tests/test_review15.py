import csv
import io
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from pdelab.experiment3d import Config3D, export3d, run_experiment3d
from pdelab.research3d import StudySettings3D, accuracy_study3d, export_study3d
from pdelab.workload3d import interactive_mc_limit_message, interactive_mc_workload


def test_csv_exports_disable_windows_translation(tmp_path, monkeypatch):
    original = Path.write_text

    def windows_write(path, data, encoding=None, errors=None, newline=None):
        if path.suffix == ".csv":
            # Эмулируем Windows даже на macOS/Linux.
            with path.open(
                "w", encoding=encoding, newline="\r\n" if newline is None else newline
            ) as stream:
                return stream.write(data)
        return original(path, data, encoding=encoding, errors=errors, newline=newline)

    monkeypatch.setattr(Path, "write_text", windows_write)
    experiment = run_experiment3d(Config3D(n=4, walks=4))
    export3d(experiment, tmp_path)
    study = accuracy_study3d(
        StudySettings3D(
            Config3D(n=4, problem="constant"), repeats=2, max_walks=2, methods=("mc",)
        )
    )
    export_study3d(study, tmp_path)
    for name, expected in (
        ("metrics", 4),
        ("timings", 3),
        ("thresholds", 3),
        ("trials", len(study.trials)),
    ):
        data = (tmp_path / f"{name}.csv").read_bytes()
        assert b"\r\r\n" not in data
        rows = list(csv.reader(io.StringIO(data.decode("utf-8-sig"), newline="")))
        assert len(rows) == expected + 1 and all(rows)


@pytest.mark.parametrize("mode", ["point", "patch", "slice", "field"])
def test_workload_counts_requested_interior_walks(mode):
    n, walks = 8, 64
    counts = {"point": 1, "patch": 27, "slice": 49, "field": 343}
    actual_walks, steps = interactive_mc_workload(
        Config3D(n=n, walks=walks, mode=mode), ["mc"]
    )
    assert actual_walks == counts[mode] * walks
    assert 0 < steps <= actual_walks * 3 * n**2 / 4


def test_large_volume_is_blocked_but_small_or_known_queries_are_allowed():
    assert interactive_mc_limit_message(Config3D(n=64, mode="field"), ["mc"])
    assert interactive_mc_limit_message(Config3D(n=64, walks=65536), ["mc"])
    assert interactive_mc_limit_message(Config3D(n=64), ["mc"]) is None
    assert (
        interactive_mc_limit_message(Config3D(n=64, mode="field"), ["lod_serial"])
        is None
    )
    assert interactive_mc_workload(Config3D(n=64, point=(0, 32, 32)), ["mc"]) == (0, 0)
    assert interactive_mc_workload(Config3D(n=64, mode="slice", index=0), ["mc"]) == (
        0,
        0,
    )


def test_ui_blocks_expensive_mc_before_solver_launch():
    app = AppTest.from_file(
        str(Path(__file__).parents[1] / "app.py"), default_timeout=60
    ).run()
    app.radio(key="dimension").set_value("3D").run()
    app.select_slider(key="cube_n").set_value(64)
    app.selectbox(key="cube_mode").set_value("field")
    app.run()
    assert not app.exception
    assert app.button(key="cube_run").disabled
    assert any("Слишком большой" in item.value for item in app.error)
    assert "cube_result" not in app.session_state
    app.multiselect(key="cube_methods").set_value(["lod_serial"]).run()
    assert not app.button(key="cube_run").disabled
