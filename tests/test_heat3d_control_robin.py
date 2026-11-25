"""Coarse MC correction with the physical boundaries merged in PDE-27."""

import csv
import json
from dataclasses import replace

import numpy as np
import pytest
from scipy.sparse.linalg import spsolve

from pdelab import heat3d_control as control_module
from pdelab.boundary3d import FACES
from pdelab.heat3d_control import solve_controlled_frozen_mc3d
from pdelab.heat3d_mc_cli import main
from pdelab.heat3d_operator import freeze_heat3d
from pdelab.heat3d_problem import make_heat_problem3d


def boundary_problem(n, alpha, all_kind3):
    kinds = {face: 3 if all_kind3 or face.endswith("0") else 1 for face in FACES}
    betas = {face: 0.5 + index for index, face in enumerate(FACES)}
    return make_heat_problem3d("polynomial", n, alpha, kinds=kinds, beta=betas)


@pytest.mark.parametrize("n,coarse_n", [(8, 4), (9, 4)])
@pytest.mark.parametrize("alpha", [0.0, 0.5])
@pytest.mark.parametrize("all_kind3", [False, True])
def test_coarse_correction_matches_direct_at_faces_edges_and_corners(
    n, coarse_n, alpha, all_kind3
):
    problem = boundary_problem(n, alpha, all_kind3)
    previous = problem.initial_values()
    iterate = previous + 0.17
    operator = freeze_heat3d(problem, previous, iterate, 0.013, 0.013)
    points = np.array([[0, 0, 0], [2, 0, 0], [2, 2, 0], [n // 2] * 3, [n] * 3])
    matrix, rhs = operator.assemble()
    target = spsolve(matrix, rhs).reshape(operator.shape)[tuple(points.T)]
    runs = [
        solve_controlled_frozen_mc3d(
            problem,
            previous,
            iterate,
            0.013,
            0.013,
            points,
            coarse_n=coarse_n,
            walks=1024,
            seed=seed,
        )
        for seed in range(12)
    ]
    assert all(run.status == "completed" for run in runs)
    values = np.array([run.values for run in runs])
    standard_error = values.std(axis=0, ddof=1) / np.sqrt(len(runs))
    np.testing.assert_array_less(
        np.abs(values.mean(axis=0) - target), 5 * standard_error + 1e-10
    )
    free_queries = np.count_nonzero(~operator.fixed[tuple(points.T)])
    for run in runs:
        assert run.method == "mc_control" and run.batch_size is None
        assert run.walks == free_queries * 1024
        assert run.coarse_relative_residual <= 1e-8
        assert run.seconds >= run.preparation_seconds >= run.coarse_seconds > 0


def test_coarse_grid_preserves_face_data_and_does_not_read_exact(monkeypatch):
    problem = boundary_problem(9, 0.5, all_kind3=False)

    def forbidden(*args, **kwargs):
        raise AssertionError("Control must not read the analytic solution")

    problem = replace(problem, exact=forbidden)
    previous = problem.initial_values()
    original_freeze = control_module.freeze_heat3d
    coarse_problems = []

    def checked_freeze(candidate, *args, **kwargs):
        if candidate.n == 4:
            coarse_problems.append(candidate)
            assert candidate.faces == problem.faces
            assert candidate.exact is None
        return original_freeze(candidate, *args, **kwargs)

    monkeypatch.setattr(control_module, "freeze_heat3d", checked_freeze)
    result = solve_controlled_frozen_mc3d(
        problem, previous, previous, 0.013, 0.013, [[0, 0, 0]], walks=32
    )
    assert result.status == "completed" and len(coarse_problems) == 1


@pytest.mark.parametrize("method", ["mc", "mc_grouped", "mc_control"])
@pytest.mark.parametrize("all_kind3", [False, True])
def test_cli_exports_each_method_with_physical_boundary_settings(
    tmp_path, method, all_kind3
):
    args = [
        "--n",
        "8",
        "--coarse-n",
        "4",
        "--case",
        "polynomial",
        "--method",
        method,
        "--batch-size",
        "7",
        "--walks",
        "128",
        "--point",
        "0",
        "0",
        "0",
        "--beta",
        "2",
        "--beta-x0",
        "3.5",
        "--direct",
        "--out",
        str(tmp_path),
    ]
    kinds = [3 if all_kind3 or face.endswith("0") else 1 for face in FACES]
    for face, kind in zip(FACES, kinds, strict=True):
        args += ["--" + face, str(kind)]
    assert main(args) == 0
    payload = json.loads((tmp_path / "result.json").read_text())
    config = payload["metadata"]["configuration"]
    result = payload["result"]
    assert config["kinds"] == kinds
    assert config["betas"] == [3.5, 2, 2, 2, 2, 2]
    assert config["method"] == result["method"] == method
    assert result["status"] == "completed" and result["walks"] == 128
    assert result["batch_size"] == (7 if method == "mc_grouped" else None)
    assert payload["metadata"]["physical_layers_accepted"] == 0
    assert payload["rows"][0]["direct_reference"] is not None
    with (tmp_path / "points.csv").open(encoding="utf-8-sig") as stream:
        row = next(csv.DictReader(stream))
    assert row["method"] == method and row["status"] == "completed"
