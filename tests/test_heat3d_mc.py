"""PDE-28 frozen-system MC: identity, independent ensembles and honest limits."""

import json
from dataclasses import replace

import numpy as np
import pytest
from scipy.sparse.linalg import spsolve

from pdelab.heat3d_mc import frozen_mc_coefficients, solve_frozen_mc3d
from pdelab.heat3d_mc_cli import main
from pdelab.heat3d_operator import HeatOperator3D, freeze_heat3d
from pdelab.heat3d_problem import make_heat_problem3d


def operator(n=4, alpha=0.5, dt=0.013):
    problem = make_heat_problem3d("polynomial", n, alpha)
    previous = problem.initial_values()
    return freeze_heat3d(problem, previous, previous, dt, dt)


@pytest.mark.parametrize("alpha", [0, 0.5, 2])
def test_coefficients_are_the_same_eliminated_system(alpha):
    op = operator(alpha=alpha)
    p, r, q = frozen_mc_coefficients(op)
    np.testing.assert_allclose(p.sum(axis=-1) + q, 1, atol=1e-15)
    assert np.all(p >= 0) and np.all(q > 0)
    assert np.all(p[op.fixed] == 0) and np.all(q[op.fixed] == 1)
    field = np.random.default_rng(4).normal(size=op.shape)
    rhs = r.copy()
    for d, offset in enumerate(
        ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))
    ):
        rhs += p[..., d] * np.roll(field, tuple(-v for v in offset), axis=(0, 1, 2))
    np.testing.assert_allclose(
        rhs, field - (op.apply(field) - op.rhs) / op.diagonal, atol=2e-15
    )


@pytest.mark.parametrize("alpha", [0, 0.5, 2])
def test_independent_runs_match_direct_frozen_solution(alpha):
    op = operator(alpha=alpha)
    points = np.array([[1, 2, 3], [2, 2, 2], [0, 0, 0]])
    matrix, rhs = op.assemble()
    target = spsolve(matrix, rhs).reshape(op.shape)[tuple(points.T)]
    results = [
        solve_frozen_mc3d(op, points, walks=4096, seed=seed) for seed in range(16)
    ]
    assert all(r.status == "completed" for r in results)
    values = np.array([r.values for r in results])
    error = values.std(axis=0, ddof=1) / np.sqrt(len(results))
    assert np.all(np.abs(values.mean(axis=0) - target) <= 5 * error + 1e-12)
    observed = values[:, 1].std(ddof=1)
    predicted = np.sqrt(np.mean([r.stderr[1] ** 2 for r in results]))
    assert 0.4 < observed / predicted < 1.8


def test_known_queries_pay_once_without_paths_and_seed_is_reproducible(monkeypatch):
    op = operator()

    def forbidden(*args, **kwargs):
        raise AssertionError("No deterministic/reference solve inside MC")

    monkeypatch.setattr(HeatOperator3D, "assemble", forbidden)
    a = solve_frozen_mc3d(op, [[0, 0, 0], [2, 2, 2]], walks=128, seed=45)
    b = solve_frozen_mc3d(op, [[0, 0, 0], [2, 2, 2]], walks=128, seed=45)
    np.testing.assert_array_equal(a.values, b.values)
    np.testing.assert_array_equal(a.stderr, b.stderr)
    assert a.transitions == b.transitions and a.walks == b.walks == 128
    assert a.completed_paths.tolist() == [0, 128]
    assert a.values[0] == op.fixed_values[0, 0, 0] and a.stderr[0] == 0
    known = solve_frozen_mc3d(
        op, [[0, 0, 0], [4, 4, 4]], walks=128, max_walks=1, max_transitions=1
    )
    assert known.status == "completed" and known.walks == known.transitions == 0


def test_full_field_uses_input_operator_and_no_exact_previous_substitution():
    def forbidden(*args):
        raise AssertionError("Exact must not be read")

    problem = replace(make_heat_problem3d("sine", 4, 0), exact=forbidden)
    previous = np.full(problem.shape, 2.0)
    op = freeze_heat3d(problem, previous, previous, 0.01, 0.01)
    points = np.indices(op.shape).reshape(3, -1).T
    result = solve_frozen_mc3d(op, points, walks=4096, seed=12)
    matrix, rhs = op.assemble()
    target = spsolve(matrix, rhs)
    assert result.status == "completed"
    assert result.walks == 27 * 4096
    assert np.all(np.abs(result.values - target) < 6 * result.stderr + 1e-12)


def test_limits_count_executed_work_and_never_return_partial_as_solution():
    op = operator(dt=10)
    limited = solve_frozen_mc3d(op, walks=100, seed=3, max_steps=1)
    assert limited.status == "path_limit" and limited.reason == "max_steps"
    assert limited.values is None and limited.stderr is None
    assert limited.walks >= 1 and limited.transitions >= 1
    assert limited.walks == int(limited.completed_paths.sum()) + 1
    budget = solve_frozen_mc3d(op, walks=100, max_walks=3, seed=3)
    assert budget.status == "walk_budget" and budget.walks == 3
    assert budget.completed_paths.tolist() == [3]
    moves = solve_frozen_mc3d(op, walks=100, max_transitions=1, seed=3)
    assert moves.status == "walk_budget" and moves.transitions == 1
    assert moves.values is None
    memory = solve_frozen_mc3d(op, walks=100, max_workspace_bytes=1)
    assert memory.status == "memory_limit" and memory.walks == memory.transitions == 0
    assert memory.peak_rss_bytes is None and memory.diagnostic_values is None


def test_boundary_defect_via_with_rhs_and_no_double_known_payment():
    op = operator()
    field = np.random.default_rng(1).normal(size=op.shape)
    # Arbitrary manufactured discrete solution, including nonzero fixed rows.
    manufactured = op.with_rhs(op.apply(field))
    result = solve_frozen_mc3d(
        manufactured, [[0, 0, 0], [2, 2, 2]], walks=50000, seed=15
    )
    target = field[0, 0, 0], field[2, 2, 2]
    assert abs(result.values[0] - target[0]) < 1e-14
    assert abs(result.values[1] - target[1]) < 5 * result.stderr[1]


def test_nonabsorbing_rows_are_explicitly_unsupported():
    shape = (5, 5, 5)
    weights = np.zeros((*shape, 6))
    weights[2, 2, 2, 4] = 1
    weights[2, 2, 3, 5] = 1
    op = HeatOperator3D(
        np.ones(shape),
        weights,
        np.zeros(shape),
        np.zeros(shape, dtype=bool),
        np.zeros(shape),
        np.ones(shape),
    )
    with pytest.raises(ValueError, match="положительное поглощение"):
        solve_frozen_mc3d(op)


@pytest.mark.parametrize(
    "settings",
    [
        dict(walks=1),
        dict(walks=True),
        dict(seed=-1),
        dict(seed=2**32),
        dict(max_steps=0),
        dict(max_walks=0),
        dict(points=[[2, 2, 2], [2, 2, 2]]),
    ],
)
def test_invalid_inputs_are_explicit(settings):
    with pytest.raises(ValueError):
        solve_frozen_mc3d(operator(), **settings)


def test_cli_exports_diagnostic_scope_and_failures(tmp_path):
    out = tmp_path / "ok"
    assert main(["--n", "4", "--walks", "128", "--direct", "--out", str(out)]) == 0
    data = json.loads((out / "result.json").read_text(encoding="utf-8"))
    assert data["metadata"]["scenario"] == "diagnostic_frozen_first_layer"
    assert data["metadata"]["physical_layers_accepted"] == 0
    assert data["result"]["status"] == "completed"
    assert data["rows"][0]["direct_reference"] is not None
    assert data["result"]["preparation_seconds"] <= data["result"]["seconds"]
    out = tmp_path / "failed"
    with pytest.raises(SystemExit) as exc:
        main(["--n", "4", "--walks", "128", "--max-walks", "2", "--out", str(out)])
    assert exc.value.code == 2
    data = json.loads((out / "result.json").read_text(encoding="utf-8"))
    assert data["result"]["values"] is None
    assert data["rows"][0]["value"] is None and data["result"]["walks"] == 2
