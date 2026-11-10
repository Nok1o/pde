"""Conservative expected walk work with reflecting faces; no solver allocations."""

import numpy as np

from .boundary3d import FACE_AXES, FACES, select_boundary_points3d
from .workload3d import MAX_INTERACTIVE_EXPECTED_STEPS, MAX_INTERACTIVE_WALKS


def boundary_workload3d(config, methods, walks=None, repeats=1):
    if not any(m != "lod_serial" for m in methods):
        return 0, 0.0
    points = select_boundary_points3d(
        config.n, config.mode, config.point, config.plane, config.index
    )
    fixed = np.zeros(len(points), dtype=bool)
    bounds = []
    kinds = dict(zip(FACES, config.kinds))
    for face, kind, kappa in zip(FACES, config.kinds, config.kappas):
        axis, end = FACE_AXES[face]
        distance = points[:, axis] if end == 0 else config.n - points[:, axis]
        if kind == "dirichlet":
            fixed |= distance == 0
        if kind == "robin" and np.min(kappa) > 0:
            # Only this face kills, all other faces reflect. This dominates
            # the actual lifetime, including the final absorption event.
            bounds.append(
                3 * distance * (2 * config.n - distance)
                + 6 * config.n**2 / np.min(kappa)
                + 1
            )
    for axis, prefix in ((2, "x"), (1, "y"), (0, "z")):
        low, high = (
            kinds[prefix + "0"] == "dirichlet",
            kinds[prefix + "1"] == "dirichlet",
        )
        distance = points[:, axis]
        if low and high:
            bounds.append(3 * distance * (config.n - distance))
        elif low:
            bounds.append(3 * distance * (2 * config.n - distance))
        elif high:
            bounds.append(3 * (config.n - distance) * (config.n + distance))
    count = int(np.count_nonzero(~fixed))
    paths = (config.walks if walks is None else walks) * repeats
    if not count:
        return 0, 0.0
    expected = np.min(bounds, axis=0) if bounds else np.full(len(points), np.inf)
    return count * paths, float(np.sum(expected[~fixed]) * paths)


def boundary_limit_message3d(config, methods, walks=None, repeats=1):
    count, steps = boundary_workload3d(config, methods, walks, repeats)
    if count > MAX_INTERACTIVE_WALKS or steps > MAX_INTERACTIVE_EXPECTED_STEPS:
        return "Запрос MC слишком велик для интерфейса. Уменьшите сетку, число путей или область расчёта. Большие серии доступны через CLI."
    return None
