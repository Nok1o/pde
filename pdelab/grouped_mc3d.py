"""Single-threaded batches of active six-neighbor stationary grid walks.

This low-level kernel expects validated constant-coefficient operator arrays and
settings. Its caller owns validation, the Numba RNG lock, and conversion of a
failed attempt into an exception. It does not implement variable-conductivity
transition probabilities.
"""

import numpy as np
from numba import njit


def grouped_workspace_bytes3d(walks, batch_size, point_count=1):
    """Bytes allocated by the kernel, excluding input operator/point arrays.

    Six eight-byte particle arrays use 48 * min(batch_size, walks) bytes.
    Returned means, standard errors and counts use 24 * point_count bytes;
    first-point samples use 8 * walks bytes, as in the sequential baseline.
    This counts array data, not Python/Numba allocator or compilation overhead.
    Arguments must already have passed the public solver's validation.
    """
    return 48 * min(batch_size, walks) + 24 * point_count + 8 * walks


@njit(inline="always")
def _grouped_reflected_step(z, y, x, n, direction):
    # Six equiprobable proposals; an outward proposal becomes an inward step.
    # The order agrees with the mixed-boundary baseline, not raw _mc3d.
    if direction == 0:
        z += 1 if z < n else -1
    elif direction == 1:
        z -= 1 if z > 0 else -1
    elif direction == 2:
        y += 1 if y < n else -1
    elif direction == 3:
        y -= 1 if y > 0 else -1
    elif direction == 4:
        x += 1 if x < n else -1
    else:
        x -= 1 if x > 0 else -1
    return z, y, x


@njit(cache=True, nogil=True)
def grouped_walks3d(
    fixed, boundary, reward, killing, points, walks, seed, max_steps, batch_size
):
    """Return means, SEs, counts, steps, samples, failed_point, started_walks.

    Each active particle advances once per sweep. Completed particles are
    removed by swapping the last still-unvisited particle into their slot.
    Groups are not replenished until all their particles finish; the final
    group contains only the remaining requested walks. Particle IDs preserve
    first-point sample order despite swaps and out-of-order completions.

    ``failed_point == -1`` means success. Otherwise a path reached max_steps
    before absorption: the caller must reject the whole attempt. The failed
    point and unvisited points retain NaN means/SEs; counts still report every
    completed path. Samples of unfinished first-point paths remain NaN.
    ``started_walks`` counts initialized particles, including unfinished ones.
    ``steps`` includes every reward/kill/proposal stage actually executed, also
    for unfinished paths and for a stage ending in killing without a move.
    Fixed starting points use no paths, stages, or random draws.

    Numba's MT19937 is seeded once for the entire call. Changing the batch size
    changes draw assignment to paths, and thus need not reproduce the baseline
    realization; the transition law and independent-path estimator agree.
    """
    np.random.seed(seed)
    n = fixed.shape[0] - 1
    means = np.full(len(points), np.nan)
    standard_errors = np.full(len(points), np.nan)
    counts = np.zeros(len(points), dtype=np.int64)
    samples = np.full(walks, np.nan)
    capacity = min(batch_size, walks)
    zs = np.empty(capacity, dtype=np.int64)
    ys = np.empty(capacity, dtype=np.int64)
    xs = np.empty(capacity, dtype=np.int64)
    payoffs = np.empty(capacity)
    path_steps = np.empty(capacity, dtype=np.int64)
    path_ids = np.empty(capacity, dtype=np.int64)
    total_steps = 0
    started_walks = 0

    for point_index in range(len(points)):
        start_z, start_y, start_x = points[point_index]
        if fixed[start_z, start_y, start_x]:
            means[point_index] = boundary[start_z, start_y, start_x]
            standard_errors[point_index] = 0.0
            if point_index == 0:
                samples[:] = means[point_index]
            continue

        mean = 0.0
        squared_deviations = 0.0
        batch_start = 0
        while batch_start < walks:
            group_size = min(capacity, walks - batch_start)
            active = group_size
            for slot in range(group_size):
                zs[slot], ys[slot], xs[slot] = start_z, start_y, start_x
                payoffs[slot] = 0.0
                path_steps[slot] = 0
                path_ids[slot] = batch_start + slot
            started_walks += group_size

            while active:
                slot = 0
                while slot < active:
                    if path_steps[slot] >= max_steps:
                        return (
                            means,
                            standard_errors,
                            counts,
                            total_steps,
                            samples,
                            point_index,
                            started_walks,
                        )
                    z, y, x = zs[slot], ys[slot], xs[slot]
                    payoff = payoffs[slot] + reward[z, y, x]
                    path_steps[slot] += 1
                    total_steps += 1
                    done = False
                    if killing[z, y, x] > 0:
                        done = np.random.random() < killing[z, y, x]
                    if not done:
                        z, y, x = _grouped_reflected_step(
                            z, y, x, n, np.random.randint(6)
                        )
                        if fixed[z, y, x]:
                            payoff += boundary[z, y, x]
                            done = True

                    if done:
                        counts[point_index] += 1
                        difference = payoff - mean
                        mean += difference / counts[point_index]
                        squared_deviations += difference * (payoff - mean)
                        if point_index == 0:
                            samples[path_ids[slot]] = payoff
                        active -= 1
                        # The last active slot has not advanced this sweep.
                        # Process its replacement before moving to the next slot.
                        if slot < active:
                            zs[slot], ys[slot], xs[slot] = (
                                zs[active],
                                ys[active],
                                xs[active],
                            )
                            payoffs[slot] = payoffs[active]
                            path_steps[slot] = path_steps[active]
                            path_ids[slot] = path_ids[active]
                    else:
                        zs[slot], ys[slot], xs[slot] = z, y, x
                        payoffs[slot] = payoff
                        slot += 1
            batch_start += group_size

        means[point_index] = mean
        standard_errors[point_index] = np.sqrt(
            max(squared_deviations, 0.0) / (walks * (walks - 1))
        )

    return (
        means,
        standard_errors,
        counts,
        total_steps,
        samples,
        -1,
        started_walks,
    )
