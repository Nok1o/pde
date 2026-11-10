import csv
import io
import json
import re
import zipfile
from pathlib import Path

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from pdelab import (
    HeatProblem,
    make_heat_problem,
    make_problem,
    reference_solution,
    solve_heat,
)
from pdelab.cli import main
from pdelab.heat import time_grid
from pdelab.heat_study import heat_accuracy_study
from pdelab.heat_views import heat_archive


@pytest.mark.parametrize("preset", ["decay", "manufactured"])
@pytest.mark.parametrize("a2", [0.1, 2.0])
def test_layers_match_independent_direct_solver(preset, a2):
    p = make_heat_problem(preset, n=12, a2=a2)
    lod = solve_heat(p, dt=0.013, T=0.04, tol=1e-12)
    direct = solve_heat(p, dt=0.013, T=0.04, method="direct")
    np.testing.assert_allclose(lod.fields, direct.fields, atol=2e-9, rtol=1e-9)
    assert np.all(lod.relative_residuals <= 1e-12)
    assert lod.times[-1] == 0.04
    assert lod.times[-1] - lod.times[-2] == pytest.approx(0.001)
    assert direct.factorizations == 2
    np.testing.assert_array_equal(lod.fields[0], p.initial)
    for t, field in zip(lod.times, lod.fields):
        boundary = p.evaluate(p.boundary, t)
        np.testing.assert_allclose(field[[0, -1], :], boundary[[0, -1], :], atol=1e-14)
        np.testing.assert_allclose(field[:, [0, -1]], boundary[:, [0, -1]], atol=1e-14)


def test_decay_and_exact_discrete_amplification():
    p = make_heat_problem(n=16)
    result = solve_heat(p, dt=0.01, T=0.1, tol=1e-12)
    eigenvalue = 8 * p.n**2 * np.sin(np.pi / (2 * p.n)) ** 2
    for index, u in enumerate(result.fields):
        np.testing.assert_allclose(
            u, p.initial / (1 + 0.01 * eigenvalue) ** index, atol=1e-10
        )
    assert np.all(np.diff(np.max(abs(result.fields), axis=(1, 2))) < 0)
    assert solve_heat(p, dt=0.1, T=0.3, method="direct").factorizations == 1


@pytest.mark.parametrize("method", ["lod", "direct"])
def test_nonzero_source_long_time_matches_stationary_sign(method):
    a2 = 0.4
    p = HeatProblem.from_functions(
        8, lambda x, y: 0, lambda x, y, t: 2, lambda x, y, t: 0, a2=a2
    )
    final = solve_heat(p, dt=0.25, T=8, method=method, tol=1e-12).fields[-1]
    stationary = make_problem(n=8, top=0, source=-2 / a2)
    np.testing.assert_allclose(final, reference_solution(stationary), atol=1e-9)


def test_three_error_sources_and_orders():
    rows = heat_accuracy_study()
    spatial = [row for row in rows if row["study"] == "space"]
    temporal = [row for row in rows if row["study"] == "time"]
    linear = [row for row in rows if row["study"] == "linear"]
    assert all(1.9 < row["order"] < 2.1 for row in spatial[1:])
    assert all(0.95 < row["order"] < 1.05 for row in temporal[1:])
    assert linear[-1]["error"] < linear[0]["error"] * 1e-4
    assert all(row["max_relative_residual"] <= row["tol"] for row in linear)


def test_manufactured_accuracy_refines_in_time_with_nonzero_boundary():
    p = make_heat_problem("manufactured", n=12)
    errors = []
    for dt in (0.02, 0.01, 0.005):
        r = solve_heat(p, dt=dt, T=0.1, tol=1e-12)
        errors.append(np.max(abs(r.fields[-1] - p.evaluate(p.exact, 0.1))))
    assert 1.9 < errors[0] / errors[1] < 2.1
    assert 1.9 < errors[1] / errors[2] < 2.1


def test_initial_only_and_short_single_step():
    p = make_heat_problem(n=8)
    initial = solve_heat(p, T=0)
    np.testing.assert_array_equal(initial.times, [0])
    np.testing.assert_array_equal(initial.fields[0], p.initial)
    short = solve_heat(p, dt=1, T=0.02, method="direct")
    np.testing.assert_array_equal(short.times, [0, 0.02])
    assert short.factorizations == 1
    times, steps = time_grid(1, 1e-20)
    np.testing.assert_array_equal(times, [0, 1e-20])
    np.testing.assert_array_equal(steps, [1e-20])


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(dt=0),
        dict(dt=-1),
        dict(dt=np.nan),
        dict(T=-1),
        dict(T=np.inf),
        dict(tol=0),
        dict(method="mc"),
        dict(max_iter=0),
        dict(dt=1e-8),
        dict(dt=0.001, T=10),
    ],
)
def test_invalid_solver_settings(kwargs):
    with pytest.raises(ValueError):
        solve_heat(make_heat_problem(n=64), **kwargs)


def test_failure_is_explicit_and_invalid_problem_data():
    with pytest.raises(RuntimeError, match="Слой 1"):
        solve_heat(make_heat_problem(n=16), dt=0.1, T=0.1, max_iter=1, tol=1e-14)
    with pytest.raises(ValueError, match="Начальное поле"):
        HeatProblem.from_functions(
            8, lambda x, y: 1, lambda x, y, t: 0, lambda x, y, t: 0
        )
    with pytest.raises(ValueError):
        make_heat_problem(a2=-1)
    p = make_heat_problem(n=8)
    p.source = lambda x, y, t: np.nan
    with pytest.raises(ValueError):
        solve_heat(p)


def test_export_and_cli_preserve_physical_times(tmp_path):
    result = solve_heat(make_heat_problem("manufactured", n=8), dt=0.013, T=0.04)
    with zipfile.ZipFile(io.BytesIO(heat_archive(result))) as archive:
        rows = list(
            csv.DictReader(io.StringIO(archive.read("layers.csv").decode("utf-8-sig")))
        )
        np.testing.assert_array_equal([float(row["t"]) for row in rows], result.times)
        with np.load(io.BytesIO(archive.read("fields.npz"))) as data:
            np.testing.assert_array_equal(data["t"], result.times)
            np.testing.assert_array_equal(data["fields"], result.fields)
            assert data["exact"].shape == result.fields.shape
        metadata = json.loads(archive.read("experiment.json"))
        assert metadata["T"] == 0.04 and metadata["dt"] == 0.013
        html = archive.read("interactive.html").decode()
        call = re.search(r'Plotly\.newPlot\(\s*"[^"]+",', html)
        assert call is not None
        decoder = json.JSONDecoder()
        arguments = html[call.end() :].lstrip()
        _, end = decoder.raw_decode(arguments)
        layout, _ = decoder.raw_decode(arguments[end:].lstrip()[1:].lstrip())
        assert layout["xaxis"]["title"]["text"] == "Физическое время t"
        assert 'src="https://cdn.plot.ly' not in html
    main(["heat", "--n", "8", "--T", ".015", "--out", str(tmp_path)])
    assert (tmp_path / "heat-results.zip").is_file()
    with np.load(tmp_path / "fields.npz") as data:
        assert data["t"][-1] == 0.015


def test_heat_ui_separates_time_and_iterations():
    app = AppTest.from_file(
        str(Path(__file__).parents[1] / "app.py"), default_timeout=60
    ).run()
    app.radio(key="evolution").set_value("Теплопроводность").run()
    assert not app.exception
    app.select_slider(key="heat_n").set_value(8)
    app.number_input(key="heat_T").set_value(0.025)
    app.button(key="heat_run").click().run()
    assert not app.exception
    assert app.session_state["heat_result"].times[-1] == 0.025
    assert "Физическое время t" in app.dataframe[0].value.columns
    assert "Итераций внутри слоя" in app.dataframe[0].value.columns
    app.number_input(key="heat_dt").set_value(0.005).run()
    assert any("Параметры изменены" in message.value for message in app.info)
    app.button(key="heat_run").click().run()
    assert not app.exception
    assert len(app.session_state["heat_result"].times) == 6
    app.number_input(key="heat_T").set_value(0.0)
    app.button(key="heat_run").click().run()
    assert not app.exception
    assert len(app.session_state["heat_result"].times) == 1
    app.radio(key="evolution").set_value("Стационарный").run()
    assert not app.exception
    assert app.button(key="main_run")


def test_random_modes_match_direct_without_splitting_bias():
    rng = np.random.default_rng(731)
    initial = np.zeros((9, 9))
    initial[1:-1, 1:-1] = rng.normal(size=(7, 7))
    source = rng.normal(size=(9, 9))
    p = HeatProblem(
        "Random modes",
        8,
        0.7,
        initial,
        lambda x, y, t: source * (1 + t),
        lambda x, y, t: 0,
    )
    a = solve_heat(p, dt=0.07, T=0.18, tol=1e-12)
    b = solve_heat(p, dt=0.07, T=0.18, method="direct")
    np.testing.assert_allclose(a.fields, b.fields, atol=1e-11)


@pytest.mark.parametrize("method", ["lod", "direct"])
def test_cancelled_rhs_resets_nonzero_previous_layer_without_iterations(method):
    p = HeatProblem.from_functions(
        32,
        initial=lambda x, y: x * (1 - x) * y * (1 - y),
        source=lambda x, y, t: -x * (1 - x) * y * (1 - y),
        boundary=lambda x, y, t: 0,
    )
    assert np.max(p.initial) > 0
    result = solve_heat(p, dt=1, T=1, method=method, max_iter=1)
    np.testing.assert_array_equal(result.fields[-1], 0)
    assert result.iterations[-1] == 0
    assert result.relative_residuals[-1] == 0
    assert result.factorizations == 0
