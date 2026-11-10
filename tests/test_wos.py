import csv
import json

import pytest
from numba import njit

from benchmarks.wos_comparison import main
from pdelab.wos import WosEstimate, WosWalkLimitError, estimate_wos


@njit
def _zero(x, y):
    return 0.0


@njit
def _constant(x, y):
    return 7.0


@njit
def _quadratic(x, y):
    return x**2 + y**2


@njit
def _four(x, y):
    return 4.0


def test_boundary_uses_exact_value_without_sampling():
    def unused_source(x, y):
        pytest.fail("A boundary query must not call the source")

    assert estimate_wos((0.0, 0.25), unused_source, _quadratic) == WosEstimate(
        0.25**2, 0.0, 0, 0
    )


def test_constant_harmonic_solution():
    result = estimate_wos((0.375, 0.25), _zero, _constant, walks=64)
    assert (result.value, result.stderr, result.walks) == (7.0, 0.0, 64)
    assert result.steps >= result.walks


def test_near_boundary_projects_onto_nearest_side():
    result = estimate_wos((0.25, 0.99999), _four, _quadratic, walks=2)
    assert result == WosEstimate(1.0 + 0.25**2, 0.0, 2, 0)


def test_quadratic_source_and_seed():
    point = (0.5, 0.375)
    epsilon = 1e-4
    options = dict(walks=8192, seed=2718, epsilon=epsilon)
    result = estimate_wos(point, _four, _quadratic, **options)
    assert result == estimate_wos(point, _four, _quadratic, **options)
    assert result.stderr > 0.0
    # Projection changes just one coordinate by at most epsilon, and |∂u| <= 2.
    assert abs(result.value - _quadratic(*point)) <= 5 * result.stderr + 2 * epsilon


def test_step_limit_rejects_incomplete_trajectory():
    with pytest.raises(WosWalkLimitError, match="max_steps"):
        estimate_wos((0.5, 0.5), _zero, _constant, walks=2, max_steps=1)


@pytest.mark.parametrize("graded", [False, True])
def test_benchmark_smoke(tmp_path, graded):
    main(["--out", str(tmp_path), "--smoke", *(["--graded"] if graded else [])])
    settings = json.loads((tmp_path / "settings.json").read_text(encoding="utf-8"))
    assert settings["grids"] == [32]
    with (tmp_path / "trials.csv").open(newline="", encoding="utf-8") as stream:
        trials = list(csv.DictReader(stream))
    with (tmp_path / "thresholds.csv").open(newline="", encoding="utf-8") as stream:
        thresholds = list(csv.DictReader(stream))
    with (tmp_path / "epsilon.csv").open(newline="", encoding="utf-8") as stream:
        epsilon = list(csv.DictReader(stream))

    methods = {"wos", "mc", "mc_control"}
    if graded:
        methods.add("mc_graded")
        assert {float(row["x"]) for row in trials} == {0.5, 0.125, 0.03125}
    assert {row["method"] for row in trials} == methods
    assert (
        len(thresholds) == 3 * 3 * len(methods) * 3
    )  # presets, points, methods, targets
    assert len(epsilon) == 3 * 3 * 3  # presets, points, stopping tolerances
    selection_seeds = {row["seed"] for row in trials if row["phase"] == "selection"}
    confirmation_seeds = {
        row["seed"] for row in trials if row["phase"] == "confirmation"
    }
    assert confirmation_seeds and selection_seeds.isdisjoint(confirmation_seeds)
    assert all(
        bool(row["median_seconds"]) == (row["status"] == "confirmed")
        for row in thresholds
    )
