"""Full-field MC updates and independent checks within physical-time Picard."""

import numpy as np

from .heat3d_control import _controlled_estimate
from .heat3d_mc import solve_frozen_mc3d

MC_METHODS = ("mc", "mc_control")


class PicardSampler3D:
    """One run owns its path/move allowance and disjoint per-update seed streams."""

    def __init__(self, problem, settings, result):
        self.problem = problem
        self.settings = settings
        self.result = result
        self.points = np.indices(problem.shape, dtype=np.int64).reshape(3, -1).T.copy()

    def draw(self, layer, operator, previous, iterate, outer, *, check=False):
        result, settings = self.result, self.settings
        diagnostics = layer.mc_attempt
        seed = int(
            np.random.SeedSequence(
                [int(settings.seed), layer.number, outer, int(check)]
            ).generate_state(1)[0]
        )
        phase = "validation" if check else "update"
        remaining_walks = settings.max_walks - result.walks
        remaining_moves = settings.max_transitions - result.transitions
        if remaining_walks <= 0 or remaining_moves <= 0:
            layer.status = "walk_budget"
            layer.reason = phase + (
                "_max_walks" if remaining_walks <= 0 else "_max_transitions"
            )
            return None
        diagnostics[phase + "_seed"] = seed
        options = dict(
            walks=settings.walks,
            seed=seed,
            max_steps=settings.max_path_steps,
            max_walks=remaining_walks,
            max_transitions=remaining_moves,
            max_workspace_bytes=settings.max_workspace_bytes,
        )
        if result.method == "mc_control":
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
