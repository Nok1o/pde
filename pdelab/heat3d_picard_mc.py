"""Full-field MC updates and independent checks within physical-time Picard."""

from sys import getsizeof

import numpy as np

from .heat3d_control import _controlled_estimate
from .heat3d_mc import solve_frozen_mc3d

MC_METHODS = ("mc", "mc_grouped", "mc_control", "mc_control_grouped")
CONTROL_MC_METHODS = ("mc_control", "mc_control_grouped")
GROUPED_MC_METHODS = ("mc_grouped", "mc_control_grouped")


class PicardSampler3D:
    """One run owns its path/move allowance and disjoint per-update seed streams."""

    def __init__(self, problem, settings, result):
        self.problem = problem
        self.settings = settings
        self.result = result
        self.points = np.indices(problem.shape, dtype=np.int64).reshape(3, -1).T.copy()
        self.used_seeds = set()

    def _batch_seed(self, layer, outer, check):
        seed = int(
            np.random.SeedSequence(
                [int(self.settings.seed), layer.number, outer, int(check)]
            ).generate_state(1)[0]
        )
        # The frozen kernel takes a uint32. Distinct SeedSequences can yield
        # the same first word. Never restart an earlier stream in this run.
        while seed in self.used_seeds:
            seed = (seed + 1) % 2**32
        before = getsizeof(self.used_seeds)
        self.used_seeds.add(seed)
        # The int itself is retained and already charged in the history row.
        layer.history_storage += getsizeof(self.used_seeds) - before
        return seed

    def draw(self, layer, operator, previous, iterate, outer, *, check=False):
        result, settings = self.result, self.settings
        diagnostics = layer.mc_attempt
        phase = "validation" if check else "update"
        remaining_walks = settings.max_walks - result.walks
        remaining_moves = settings.max_transitions - result.transitions
        if remaining_walks <= 0:
            layer.status = "walk_budget"
            layer.reason = phase + "_max_walks"
            return None
        seed = self._batch_seed(layer, outer, check)
        diagnostics[phase + "_seed"] = seed
        options = dict(
            walks=settings.walks,
            seed=seed,
            max_steps=settings.max_path_steps,
            max_walks=remaining_walks,
            max_transitions=remaining_moves,
            max_workspace_bytes=settings.max_workspace_bytes,
        )
        if result.batch_size is not None:
            options["batch_size"] = result.batch_size
        if result.method in CONTROL_MC_METHODS:
            sampled = _controlled_estimate(
                self.problem,
                previous,
                iterate,
                layer.t,
                layer.dt,
                self.points,
                fine=operator,
                coarse_n=settings.coarse_n,
                coarse_tolerance=settings.linear_tolerance,
                coarse_max_iterations=settings.max_inner,
                settings=options,
            )
            result.coarse_iterations += sampled.coarse_iterations
            layer.inner_iterations += sampled.coarse_iterations
            result.coarse_seconds += sampled.coarse_seconds
            result.interpolation_seconds += sampled.interpolation_seconds
            result.defect_seconds += sampled.defect_seconds
            diagnostics["coarse_iterations"] = (
                diagnostics.get("coarse_iterations", 0) + sampled.coarse_iterations
            )
        else:
            sampled = solve_frozen_mc3d(operator, self.points, **options)
        layer.preparation_seconds += sampled.preparation_seconds
        completed = (
            int(sampled.completed_paths.sum())
            if sampled.completed_paths is not None
            else 0
        )
        result.walks += sampled.walks
        result.completed_walks += completed
        result.transitions += sampled.transitions
        for key, count in (
            ("walks", sampled.walks),
            ("completed_walks", completed),
            ("transitions", sampled.transitions),
        ):
            diagnostics[key] = diagnostics.get(key, 0) + count
        if check:
            result.validation_walks += sampled.walks
            diagnostics["validation_walks"] = sampled.walks
        if sampled.status != "completed":
            layer.status, layer.reason = sampled.status, phase + "_" + sampled.reason
            return None
        return sampled.values.reshape(self.problem.shape)
