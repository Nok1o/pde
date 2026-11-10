import json

import numpy as np
from PIL import Image

from pdelab.experiment import Config, benchmark, run_experiment
from pdelab.views import (
    benchmark_figure,
    export,
    interactive_field,
    layer_plot,
    reuse_plot,
)


def test_export_and_real_animation(tmp_path):
    exp = run_experiment(Config(n=8, walks=32, batches=4))
    export(exp, tmp_path, with_gifs=True)
    stored = np.load(tmp_path / "fields.npz")
    assert np.isnan(stored["mc"][2, 2])
    assert np.isfinite(stored["mc"][4, 4])
    assert (
        json.loads((tmp_path / "experiment.json").read_text())["config"]["mode"]
        == "point"
    )
    page = (tmp_path / "interactive.html").read_text()
    assert "plotly.js" in page and "<script src=" not in page
    for name in ["06-траектория.gif", "07-установление-сетки.gif"]:
        assert Image.open(tmp_path / name).n_frames > 1


def test_visualizations_do_not_fill_unknown_nodes():
    exp = run_experiment(Config(n=8, point=(1, 1), walks=32, batches=4))
    counts = interactive_field(exp, "count")
    assert np.isnan(np.asarray(counts.data[0].z)[4, 4])
    layer = layer_plot(exp)
    assert len(layer.frames) == 2
    assert np.isnan(np.asarray(layer.frames[-1].data[0].z)[4, 4])
    reuse = reuse_plot(exp)
    assert len(reuse.layout.updatemenus[0].buttons) >= 1


def test_benchmark_varies_budgets_and_seeds():
    rows = benchmark(
        Config(n=8, walks=32, batches=4),
        budgets=[16, 32],
        repeats=2,
        tolerances=[1e-2, 1e-4],
    )
    assert len(rows) == 20
    assert set(r["repeat"] for r in rows) == {0, 1}
    assert all(r["converged"] for r in rows)
    assert len(benchmark_figure(rows).data) == 5
