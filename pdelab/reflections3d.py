"""Single-stage reflection on the uniform stationary k=1 cube."""

import numpy as np
from numba import njit


def reflection_table(n):
    """Next coordinate for a positive or negative proposal, on any axis."""
    coordinates = np.arange(n + 1, dtype=np.intp)
    table = np.column_stack((coordinates + 1, coordinates - 1))
    table[n, 0] = n - 1
    table[0, 1] = 1
    return table


@njit(inline="always")
def reflected_step(z, y, x, n, direction, table=None):
    # Numba specializes the two representations. Both use z+,z-,y+,y-,x+,x-.
    if table is not None:
        sign = direction % 2
        if direction < 2:
            z = table[z, sign]
        elif direction < 4:
            y = table[y, sign]
        else:
            x = table[x, sign]
    elif direction == 0:
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


@njit(inline="always")
def advance_particle(z, y, x, fixed, boundary, reward, killing, payoff, table):
    """Pay, test killing, then propose exactly one neighbor on survival."""
    payoff += reward[z, y, x]
    probability = killing[z, y, x]
    if probability > 0 and np.random.random() < probability:
        return z, y, x, payoff, True
    z, y, x = reflected_step(z, y, x, fixed.shape[0] - 1, np.random.randint(6), table)
    if fixed[z, y, x]:
        return z, y, x, payoff + boundary[z, y, x], True
    return z, y, x, payoff, False


@njit(cache=True, nogil=True)
def cached_walks3d(fixed, boundary, reward, killing, points, walks, seed, limit, table):
    """One particle at a time, with the same work counters as grouped_walks3d."""
    np.random.seed(seed)
    means = np.full(len(points), np.nan)
    errors = np.full(len(points), np.nan)
    counts = np.zeros(len(points), dtype=np.int64)
    samples = np.full(walks, np.nan)
    stages = 0
    started = 0
    for index in range(len(points)):
        z0, y0, x0 = points[index]
        if fixed[z0, y0, x0]:
            means[index] = boundary[z0, y0, x0]
            errors[index] = 0.0
            if index == 0:
                samples[:] = means[index]
            continue
        mean = 0.0
        squared_deviations = 0.0
        for walk in range(walks):
            started += 1
            z, y, x = z0, y0, x0
            payoff = 0.0
            for step in range(limit):
                z, y, x, payoff, done = advance_particle(
                    z, y, x, fixed, boundary, reward, killing, payoff, table
                )
                if done:
                    stages += step + 1
                    break
            else:
                return means, errors, counts, stages + limit, samples, index, started
            counts[index] += 1
            difference = payoff - mean
            mean += difference / (walk + 1)
            squared_deviations += difference * (payoff - mean)
            if index == 0:
                samples[walk] = payoff
        means[index] = mean
        errors[index] = np.sqrt(max(squared_deviations, 0.0) / (walks * (walks - 1)))
    return means, errors, counts, stages, samples, -1, started
