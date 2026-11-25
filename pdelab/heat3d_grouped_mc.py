"""Packed active paths for a frozen, nonuniform 3D heat-system estimator.

Inputs are the already eliminated coefficients of ``heat3d_mc``. This kernel
uses one draw for killing and all six weighted directions, without reflection
or an additional boundary payoff. Its transition counter records spatial moves.
"""

import numpy as np
from numba import njit

from .solvers import NUMBA_LOCK


@njit(cache=True, nogil=True)
def _grouped_frozen_paths(
    probabilities,
    reward,
    killing,
    fixed,
    points,
    walks,
    seed,
    max_steps,
    max_walks,
    max_transitions,
    batch_size,
):
    """Return means, M2, completed counts, started paths, moves, status code.

    Status codes and failure ordering match the sequential frozen kernel:
    0=completed, 1=path limit, 2=path budget, 3=move budget, 4=nonfinite.
    Each active path advances once per sweep; a finished path is removed by
    swapping the final unvisited path into its slot. Paths are initialized only
    within the remaining global path budget. Five contiguous particle arrays
    consume exactly 40 * min(batch_size, walks) data bytes, excluding outputs.

    A failed attempt may expose partial means only as diagnostics. Started paths
    include every initialized particle, even those not yet advanced at failure.
    In particular, reward and possible killing precede both move-limit checks.
    ``batch_size=1`` reproduces the baseline's draw order and arithmetic exactly.
    """
    np.random.seed(seed)
    means = np.zeros(len(points))
    m2 = np.zeros(len(points))
    counts = np.zeros(len(points), dtype=np.int64)
    capacity = min(batch_size, walks)
    zs = np.empty(capacity, dtype=np.int64)
    ys = np.empty(capacity, dtype=np.int64)
    xs = np.empty(capacity, dtype=np.int64)
    payoffs = np.empty(capacity)
    path_steps = np.empty(capacity, dtype=np.int64)
    started = 0
    transitions = 0

    for index in range(len(points)):
        sz, sy, sx = points[index]
        if fixed[sz, sy, sx]:
            means[index] = reward[sz, sy, sx]
            continue
        batch_start = 0
        while batch_start < walks:
            if started >= max_walks:
                return means, m2, counts, started, transitions, 2
            group_size = min(capacity, walks - batch_start, max_walks - started)
            for slot in range(group_size):
                zs[slot], ys[slot], xs[slot] = sz, sy, sx
                payoffs[slot] = 0.0
                path_steps[slot] = 0
            started += group_size
            active = group_size

            while active:
                slot = 0
                while slot < active:
                    z, y, x = zs[slot], ys[slot], xs[slot]
                    payoff = payoffs[slot] + reward[z, y, x]
                    if not np.isfinite(payoff):
                        return means, m2, counts, started, transitions, 4
                    choice = np.random.random()
                    done = choice < killing[z, y, x]
                    if not done:
                        if path_steps[slot] >= max_steps:
                            return means, m2, counts, started, transitions, 1
                        if transitions >= max_transitions:
                            return means, m2, counts, started, transitions, 3
                        threshold = killing[z, y, x]
                        direction = -1
                        for axis_direction in range(6):
                            threshold += probabilities[z, y, x, axis_direction]
                            if choice < threshold:
                                direction = axis_direction
                                break
                        # Match the scalar kernel's cumulative-roundoff handling.
                        if direction == -1:
                            done = True
                        else:
                            if direction == 0:
                                z += 1
                            elif direction == 1:
                                z -= 1
                            elif direction == 2:
                                y += 1
                            elif direction == 3:
                                y -= 1
                            elif direction == 4:
                                x += 1
                            else:
                                x -= 1
                            path_steps[slot] += 1
                            transitions += 1

                    if done:
                        counts[index] += 1
                        delta = payoff - means[index]
                        means[index] += delta / counts[index]
                        m2[index] += delta * (payoff - means[index])
                        if not np.isfinite(means[index]) or not np.isfinite(m2[index]):
                            return means, m2, counts, started, transitions, 4
                        active -= 1
                        # Slots below this one already advanced this sweep;
                        # the final active slot is still waiting for its turn.
                        if slot < active:
                            zs[slot], ys[slot], xs[slot] = (
                                zs[active],
                                ys[active],
                                xs[active],
                            )
                            payoffs[slot] = payoffs[active]
                            path_steps[slot] = path_steps[active]
                    else:
                        zs[slot], ys[slot], xs[slot] = z, y, x
                        payoffs[slot] = payoff
                        slot += 1
            batch_start += group_size

    return means, m2, counts, started, transitions, 0


_warmed = False


def warmup_grouped_mc3d():
    """Compile the production array signature, outside the wrapper's timer."""
    global _warmed
    with NUMBA_LOCK:
        if _warmed:
            return
        shape = (5, 5, 5)
        _grouped_frozen_paths(
            np.zeros((*shape, 6)),
            np.zeros(shape),
            np.ones(shape),
            np.frombuffer(
                np.zeros(shape, dtype=np.bool_).tobytes(), dtype=np.bool_
            ).reshape(shape),
            np.array([[2, 2, 2]], dtype=np.int64),
            2,
            0,
            1,
            2,
            1,
            2,
        )
        _warmed = True
