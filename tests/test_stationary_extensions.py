from dataclasses import replace

import numpy as np
import pytest
from scipy.interpolate import RegularGridInterpolator

from pdelab.boundary import BoundaryCondition, BoundaryOperator, BoundaryProblem
from pdelab.boundary3d import BoundaryOperator3D, BoundaryProblem3D
from pdelab.boundary_experiment import BoundaryConfig
from pdelab.boundary_experiment3d import BoundaryConfig3D
from pdelab.boundary_research import boundary_accuracy_study
from pdelab.problem import Problem
from pdelab.problem3d import Problem3D
from pdelab.stationary_extensions import graded_boundary_operator, solve_extended


@pytest.mark.parametrize("dimension", [2, 3])
def test_graded_mixed_operator_preserves_linear_field_and_balance(dimension):
    n = 8
    shape = (n + 1,) * dimension
    coordinates = np.meshgrid(*([np.linspace(0, 1, n + 1)] * dimension), indexing="ij")
    exact = 1 + sum((k + 1) * c for k, c in enumerate(coordinates))
    operator_type = BoundaryOperator if dimension == 2 else BoundaryOperator3D
    cls = BoundaryProblem if dimension == 2 else BoundaryProblem3D
    sides = {}
    for name, (axis, end) in operator_type.face_axes.items():
        indices = [slice(None)] * dimension
        indices[axis] = end
        normal = (axis + 1) * (-1 if end == 0 else 1)
        sides[name] = BoundaryCondition("robin", normal + 2 * exact[tuple(indices)], 2)
    problem = cls(n, np.zeros(shape), sides, exact)
    operator, axis = graded_boundary_operator(problem)
    mesh = np.meshgrid(*([axis] * dimension), indexing="ij")
    target = 1 + sum((k + 1) * c for k, c in enumerate(mesh))
    np.testing.assert_allclose(operator.apply(target), operator.rhs(), atol=1e-11)
    np.testing.assert_allclose(operator.solve(), target, atol=1e-10)
    assert operator.volumes.sum() == pytest.approx(1)
    # Source and boundaries, never exact, determine the stochastic solve.
    result = solve_extended(
        replace(problem, exact=None), [[3] * dimension], "graded", walks=20000
    )
    index = (3,) * dimension
    assert abs(result.values[index] - exact[index]) < 5 * result.stderr[index]


@pytest.mark.parametrize("dimension", [2, 3])
@pytest.mark.parametrize("method", ["wos", "graded"])
def test_extensions_constant_source_and_same_physical_point(dimension, method):
    cls = Problem if dimension == 2 else Problem3D
    problem = cls.from_functions(
        8, lambda *coords: 2 * dimension, lambda *coords: 1 + sum(c * c for c in coords)
    )
    points = np.array([[2] * dimension, [3] * dimension])
    result = solve_extended(problem, points, method, walks=24000)
    indices = tuple(points.T)
    if method == "graded":
        operator, axis = graded_boundary_operator(problem)
        grid = operator.solve()
        expected = RegularGridInterpolator((axis,) * dimension, grid)(
            points / problem.n
        )
    else:
        # Multilinear boundary reconstruction differs from the quadratic by O(h²).
        expected = 1 + np.sum((points / problem.n) ** 2, axis=1)
    tolerance = 5 * result.stderr[indices] + (
        dimension / problem.n**2 if method == "wos" else 0
    )
    assert np.all(abs(result.values[indices] - expected) <= tolerance)
    assert result.extras["query_grid"] == "original_uniform"
    with pytest.raises(RuntimeError):
        solve_extended(problem, points, method, walks=20, max_steps=1)


@pytest.mark.parametrize("config", [BoundaryConfig(n=4), BoundaryConfig3D(n=4)])
def test_boundary_targets_use_fresh_validation_seeds(config):
    result = boundary_accuracy_study(
        config, ["lod_serial", "mc"], repeats=2, budgets=[64, 256], tolerances=[1e-8]
    )
    pilot = {r["seed"] for r in result["trials"] if r["phase"] == 0}
    validation = {r["seed"] for r in result["trials"] if r["phase"] == 1}
    assert pilot.isdisjoint(validation) and validation
    assert len(result["rows"]) == 6
    assert all(
        r["seconds"] is None for r in result["rows"] if r["status"] != "confirmed"
    )
    boundary = boundary_accuracy_study(
        replace(config, point=(0,) * len(config.kinds[::2])), ["lod_serial"], repeats=2
    )
    assert len(boundary["rows"]) == 3


def test_all_dirichlet_data_are_supported_through_mixed_api():
    from pdelab.boundary_experiment import run_boundary_experiment
    from pdelab.boundary_experiment3d import run_boundary_experiment3d

    for config, run in (
        (
            BoundaryConfig(n=4, kinds=("dirichlet",) * 4, walks=64),
            run_boundary_experiment,
        ),
        (
            BoundaryConfig3D(n=4, kinds=("dirichlet",) * 6, walks=64),
            run_boundary_experiment3d,
        ),
    ):
        result = run(config)
        assert "wos" in result.results and "graded" in result.results
        with pytest.raises(ValueError, match="WoS"):
            run(replace(config, kinds=("robin",) + config.kinds[1:]), methods=["wos"])


def test_boundary_queries_do_not_take_random_steps():
    from pdelab.problem import make_problem
    from pdelab.solvers import METHODS, solve, trace_walk

    problem = make_problem("affine", 8)
    for method in METHODS:
        result = solve(problem, [[0, 0]], method, walks=64)
        assert result.values[0, 0] == problem.boundary[0, 0]
        assert result.steps == 0
    trace = trace_walk(problem, (0, 0))
    assert len(trace["points"]) == 1 and trace["reward"] == problem.boundary[0, 0]


@pytest.mark.parametrize(
    "dimension,prefix,research",
    [("2D", "bc", "bc_research"), ("3D", "bc3d", "bc3_research")],
)
def test_mixed_research_ui_runs_exports_and_invalidates(dimension, prefix, research):
    from pathlib import Path

    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(
        Path(__file__).parents[1] / "app.py", default_timeout=60
    ).run()
    app.radio(key="dimension").set_value(dimension).run()
    app.radio(
        key="boundary_mode" if dimension == "2D" else "cube_boundary_mode"
    ).set_value("Смешанные I / II / III").run()
    assert "Исследование" in [tab.label for tab in app.tabs]
    app.multiselect(key=prefix + "_methods").set_value(["lod_serial", "graded"])
    app.select_slider(key=research + "_n").set_value(4)
    app.number_input(key=research + "_repeats").set_value(2)
    app.multiselect(key=research + "_budgets").set_value([64])
    app.button(key=research + "_run").click().run()
    assert not app.exception and not app.error
    result = app.session_state[research][1]
    assert len(result["rows"]) == 6
    assert result["trials"]
    app.number_input(key=research + "x").set_value(0).run()
    assert any("изменены" in item.value for item in app.info)


@pytest.mark.parametrize("method", ["graded", "wos"])
def test_known_boundary_is_not_interpolated_twice(method):
    problem = Problem.from_functions(
        9, lambda x, y: 0 * x, lambda x, y: 1 + x * x + y * y
    )
    result = solve_extended(problem, [[0, 3], [9, 4]], method, walks=64)
    np.testing.assert_array_equal(
        result.values[[0, 9], [3, 4]], problem.boundary[[0, 9], [3, 4]]
    )
    assert result.walks == result.steps == 0
