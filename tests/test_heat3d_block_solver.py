"""Block Thomas as a backend of the unchanged nonlinear physical-time scheme."""

from dataclasses import replace

import numpy as np
import pytest

from pdelab.heat3d_block import BlockThomasSolver3D
from pdelab.heat3d_problem import make_heat_problem3d
from pdelab.heat3d_solver import HeatSettings3D, solve_heat3d

STRICT = HeatSettings3D(tolerance=1e-10, change_tolerance=1e-10, linear_tolerance=1e-12)


@pytest.mark.parametrize("case", ["sine", "polynomial"])
@pytest.mark.parametrize("alpha", [0.0, 0.5, 2.0])
def test_block_full_time_chain_matches_lom_and_independent_newton(case, alpha):
    problem = make_heat_problem3d(case, n=6, alpha=alpha)
    points = np.array([[3, 3, 3], [2, 3, 4], [0, 2, 4], [6, 6, 6]])
    runs = {
        method: solve_heat3d(
            problem, points, dt=0.013, T=0.04, settings=STRICT, method=method
        )
        for method in ("block", "lod", "newton")
    }
    result = runs["block"]
    assert result.status == runs["lod"].status == runs["newton"].status == "completed"
    np.testing.assert_allclose(result.values, runs["newton"].values, atol=3e-9, rtol=0)
    np.testing.assert_allclose(result.values, runs["lod"].values, atol=3e-9, rtol=0)
    assert result.physical_steps == 4 and result.completed_until == 0.04
    assert result.history[-1]["dt"] == pytest.approx(0.001)
    assert result.seconds >= result.preparation_seconds >= result.assembly_seconds > 0
    assert result.factorization_seconds > 0 and result.solve_seconds > 0
    assert result.factorizations >= 1
    assert result.factorizations + result.factor_reuses == result.inner_iterations
    for row in result.history:
        if row["accepted"]:
            assert row["nonlinear_relative_residual"] <= STRICT.tolerance
            assert row["relative_change"] <= STRICT.change_tolerance


def test_linear_time_chain_reuses_equal_matrix_and_rebuilds_short_step():
    result = solve_heat3d(
        make_heat_problem3d(n=6, alpha=0),
        dt=0.013,
        T=0.04,
        settings=STRICT,
        method="block",
    )
    assert result.status == "completed"
    # Three equal physical steps and one shortened step: exactly two matrices.
    assert result.factorizations == 2
    assert result.factor_reuses == 2
    assert result.inner_iterations == 4


def test_block_initial_only_does_not_factorize_or_enter_backend():
    def forbidden(*args, **kwargs):
        pytest.fail("T=0 requires no block solve")

    problem = make_heat_problem3d(n=6)
    result = solve_heat3d(problem, T=0, method="block", backend=forbidden)
    assert result.status == "completed"
    assert result.values[0] == problem.initial_values()[3, 3, 3]
    assert result.factorizations == result.factor_reuses == result.inner_iterations == 0


@pytest.mark.parametrize(
    "n,budget,status", [(20, 2**30, "unsupported"), (16, 1000, "memory_limit")]
)
def test_block_resource_checks_precede_backend_creation(monkeypatch, n, budget, status):
    import pdelab.heat3d_solver as module

    def forbidden(*args, **kwargs):
        pytest.fail("Refused configuration must not create a backend")

    monkeypatch.setattr(module, "BlockThomasSolver3D", forbidden)
    result = solve_heat3d(
        make_heat_problem3d(n=n),
        method="block",
        settings=replace(STRICT, max_workspace_bytes=budget),
    )
    assert result.status == status
    assert result.outer_iterations == result.factorizations == 0
    assert result.values is None and result.diagnostic_field is None


def test_failed_outer_layer_preserves_block_work_and_candidate():
    result = solve_heat3d(
        make_heat_problem3d(n=6), method="block", settings=replace(STRICT, max_outer=1)
    )
    assert result.status == "outer_limit"
    assert result.values is None and result.diagnostic_time == 0.01
    assert result.diagnostic_field.shape == (7, 7, 7)
    assert (
        result.factorizations == result.inner_iterations == result.outer_iterations == 1
    )
    assert result.factorization_seconds > 0 and result.solve_seconds > 0


def test_refused_block_backend_does_not_accept_a_picard_layer():
    result = solve_heat3d(
        make_heat_problem3d(n=4),
        method="block",
        settings=STRICT,
        backend=BlockThomasSolver3D(max_workspace_bytes=1),
    )
    assert result.status == "memory_limit"
    assert (
        result.physical_steps == result.inner_iterations == result.factorizations == 0
    )
    assert result.outer_iterations == 1 and result.values is None
    assert not result.history[-1]["accepted"]
