"""Compaction changes diagnostics only; numerical work and RNG are preserved."""

from dataclasses import replace
from sys import getsizeof
from types import SimpleNamespace

import numpy as np
import pytest

from pdelab.boundary3d import FACES
from pdelab.heat3d_history import SEED_KEYS, WORK_COUNTERS, row_storage, store_record
from pdelab.heat3d_picard_mc import MC_METHODS, PicardSampler3D
from pdelab.heat3d_problem import make_heat_problem3d
from pdelab.heat3d_solver import HeatSettings3D, solve_heat3d


def test_layer_record_keeps_latest_metrics_and_sums_all_work():
    first = dict(
        layer=1,
        outer=1,
        accepted=False,
        reason="retry",
        walks=10,
        transitions=20,
        completed_walks=10,
        validation_walks=4,
        coarse_iterations=5,
        update_seed=7,
        validation_seed=8,
        validation_residual=0.5,
    )
    final = dict(
        layer=1,
        outer=2,
        accepted=False,
        reason="path_limit",
        walks=3,
        transitions=4,
        completed_walks=2,
        coarse_iterations=1,
        update_seed=9,
    )
    records = []
    size, discarded = store_record(records, first, "layers")
    assert discarded == 0
    delta, discarded = store_record(records, final, "layers")
    assert discarded == 1 and len(records) == 1
    assert size + delta == row_storage(records[0])
    assert records[0]["walks"] == 13 and records[0]["transitions"] == 24
    assert records[0]["completed_walks"] == 12
    assert records[0]["coarse_iterations"] == 6
    assert records[0]["validation_walks"] == 4
    assert records[0]["reason"] == "path_limit" and records[0]["update_seed"] == 9
    assert (
        "validation_seed" not in records[0] and "validation_residual" not in records[0]
    )
    assert records[0]["records_discarded"] == 1
    store_record(records, dict(layer=2, outer=0, accepted=True), "layers")
    assert len(records) == 2 and records[-1]["records_discarded"] == 0


def test_all_seed_memory_remains_charged_when_rows_are_compacted():
    p = make_heat_problem3d(n=4)
    sampler = PicardSampler3D(p, HeatSettings3D(seed=12), None)
    layer = SimpleNamespace(number=1, history_storage=0)
    initial_size = getsizeof(sampler.used_seeds)
    seed = sampler._batch_seed(layer, 1, False)
    assert layer.history_storage == (
        getsizeof(sampler.used_seeds) - initial_size + getsizeof(seed)
    )
    row = dict(layer=1, update_seed=seed)
    assert row_storage(row) == getsizeof(row) + getsizeof(row["layer"])
    old_size = layer.history_storage
    seed2 = sampler._batch_seed(layer, 2, True)
    assert seed2 != seed and layer.history_storage >= old_size + getsizeof(seed2)
    assert seed in sampler.used_seeds


def problem():
    return make_heat_problem3d("polynomial", 5, 0.5, kinds={f: 3 for f in FACES})


def assert_same_work(full, layers):
    for name in (
        "method",
        "status",
        "reason",
        "completed_until",
        "physical_steps",
        "outer_iterations",
        "inner_iterations",
        "walks",
        "completed_walks",
        "transitions",
        "validation_walks",
        "coarse_iterations",
        "diagnostic_time",
        "factorizations",
        "factor_reuses",
    ):
        assert getattr(full, name) == getattr(layers, name), name
    np.testing.assert_array_equal(full.values, layers.values)
    np.testing.assert_array_equal(full.diagnostic_field, layers.diagnostic_field)
    assert full.history_mode == "full" and layers.history_mode == "layers"
    assert full.history_records_discarded == 0
    assert layers.history_records_discarded == len(full.history) - len(layers.history)
    assert layers.history_records_discarded == sum(
        row["records_discarded"] for row in layers.history
    )
    for compact in layers.history:
        source = [row for row in full.history if row["layer"] == compact["layer"]]
        assert compact["outer"] == source[-1]["outer"]
        for key, value in source[-1].items():
            if key not in WORK_COUNTERS:
                assert compact[key] == value
        for key in WORK_COUNTERS:
            if key in compact:
                assert compact[key] == sum(row.get(key, 0) for row in source)
        for key in SEED_KEYS:
            assert (key in compact) == (key in source[-1])


@pytest.mark.parametrize("method", ["lod", "block", "newton", *MC_METHODS])
def test_all_methods_preserve_complete_shortened_history(method):
    settings = HeatSettings3D(
        tolerance=0.2 if method in MC_METHODS else 1e-9,
        change_tolerance=0.12 if method in MC_METHODS else 1e-9,
        walks=1024,
        batch_size=7,
    )
    p = problem()
    points = np.indices(p.shape).reshape(3, -1).T
    args = dict(points_zyx=points, method=method, dt=0.002, T=0.003)
    full = solve_heat3d(p, settings=settings, **args)
    layers = solve_heat3d(p, settings=replace(settings, history_mode="layers"), **args)
    assert full.status == "completed" and full.physical_steps == 2
    assert len(layers.history) == 2 and layers.history[-1]["dt"] == pytest.approx(0.001)
    assert_same_work(full, layers)


@pytest.mark.parametrize("method", MC_METHODS)
@pytest.mark.parametrize("seed", [42, 83, 631])
def test_compacted_mc_retries_keep_independent_streams_and_failed_work(
    method, seed, monkeypatch
):
    original = PicardSampler3D.draw
    streams = []

    def reject_check(self, layer, *args, check=False, **kwargs):
        estimate = original(self, layer, *args, check=check, **kwargs)
        streams.append(
            layer.mc_attempt[("validation" if check else "update") + "_seed"]
        )
        return estimate + 10 if check and estimate is not None else estimate

    monkeypatch.setattr(PicardSampler3D, "draw", reject_check)
    settings = HeatSettings3D(
        tolerance=0.2,
        change_tolerance=0.12,
        walks=1024,
        batch_size=7,
        max_outer=3,
        seed=seed,
    )
    full = solve_heat3d(problem(), method=method, dt=0.002, T=0.003, settings=settings)
    full_streams = streams.copy()
    streams.clear()
    layers = solve_heat3d(
        problem(),
        method=method,
        dt=0.002,
        T=0.003,
        settings=replace(settings, history_mode="layers"),
    )
    assert full.status == "outer_limit"
    assert len(full.history) == 3 and len(layers.history) == 1
    assert full_streams == streams and len(streams) == len(set(streams))
    assert_same_work(full, layers)


def test_layers_reduce_diagnostics_without_hiding_memory_limit():
    p = make_heat_problem3d("polynomial", 4, 0.5)
    settings = HeatSettings3D(max_seconds=60)
    full = solve_heat3d(p, dt=0.001, T=0.02, settings=settings)
    layers = solve_heat3d(
        p, dt=0.001, T=0.02, settings=replace(settings, history_mode="layers")
    )
    assert_same_work(full, layers)
    assert layers.workspace_bytes < full.workspace_bytes
    budget = (layers.workspace_bytes + full.workspace_bytes) // 2
    failed = solve_heat3d(
        p, dt=0.001, T=0.02, settings=replace(settings, max_workspace_bytes=budget)
    )
    reduced = solve_heat3d(
        p,
        dt=0.001,
        T=0.02,
        settings=replace(settings, history_mode="layers", max_workspace_bytes=budget),
    )
    assert failed.status == "memory_limit"
    assert reduced.status == "completed"
    np.testing.assert_array_equal(reduced.values, full.values)
    early = solve_heat3d(
        p, settings=replace(settings, history_mode="layers", max_workspace_bytes=1)
    )
    assert early.status == "memory_limit" and early.history_mode == "layers"
    assert (
        early.values is None
        and not early.history
        and early.history_records_discarded == 0
    )


def test_invalid_history_mode_and_initial_only():
    with pytest.raises(ValueError, match="history_mode"):
        solve_heat3d(problem(), settings=HeatSettings3D(history_mode="none"))
    result = solve_heat3d(
        problem(), T=0, settings=HeatSettings3D(history_mode="layers")
    )
    assert result.status == "completed" and not result.history
    assert result.history_mode == "layers" and result.history_records_discarded == 0
