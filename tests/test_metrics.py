from dataclasses import asdict, make_dataclass
from types import SimpleNamespace

import numpy as np
import pytest

from pdelab.experiment import (
    METRICS_POLICY,
    Config,
    metrics,
    refresh_comparison,
    run_experiment,
)
from pdelab.problem import Problem
from pdelab.solvers import METHODS, Result


@pytest.mark.parametrize(
    "target,answer,expected",
    [(10.0, 12.0, 20.0), (-10.0, -12.0, 20.0), (0.0, 0.0, None), (0.0, 1.0, None)],
)
def test_relative_error_at_one_point(target, answer, expected):
    p = Problem.from_functions(
        8, lambda x, y: 0.0, lambda x, y: target, lambda x, y: target
    )
    r = Result(
        "mc",
        np.full_like(p.f, answer),
        np.zeros_like(p.f),
        np.ones_like(p.f, dtype=int),
    )
    row = metrics(r, p, np.array([[4, 4]]), p.exact)
    assert row["relative_error_pct"] == expected
    assert row["rmse"] == abs(answer - target)
    assert "max_error" not in row and "mean_stderr" not in row


def test_relative_error_is_ratio_of_norms_including_zero_reference_nodes():
    p = Problem.from_functions(
        8, lambda x, y: 0.0, lambda x, y: x - 0.5, lambda x, y: x - 0.5
    )
    q = np.array([[4, 2], [4, 4], [4, 6]])
    y, x = q.T
    values = p.exact.copy()
    values[y, x] += [0.025, 0.1, -0.025]
    r = Result("mc", values, np.zeros_like(values), np.ones_like(values, dtype=int))
    row = metrics(r, p, q, p.exact)
    assert row["relative_error_pct"] == pytest.approx(30.0)


def test_upgrade_discards_retired_method_and_obsolete_config_fields():
    exp = run_experiment(Config(n=8), methods=["lod_serial"])
    legacy = SimpleNamespace(**vars(exp))
    values = asdict(exp.config)
    OldConfig = make_dataclass(
        "OldConfig", [(k, object) for k in values] + [("workers", int)]
    )
    legacy.config = OldConfig(**values, workers=4)
    legacy.results = {**exp.results, "retired_method": exp.results["lod_serial"]}
    legacy.metadata = {"workers": 4}
    upgraded = refresh_comparison(legacy)
    assert upgraded.config == exp.config
    assert list(upgraded.results) == ["lod_serial"]
    assert upgraded.results["lod_serial"] is exp.results["lod_serial"]
    assert "workers" not in upgraded.metadata
    assert upgraded.metadata["metrics_policy"] == METRICS_POLICY
    assert upgraded.metrics[0]["relative_error_pct"] is not None
    assert len(METHODS) == 5
