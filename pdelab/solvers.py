"""Numerical kernels. Monte Carlo always stops on first known boundary hit.

The directional baseline is a factorized residual correction:
 (I-tau*L_y)(I-tau*L_x) delta = tau*(L_x u+L_y u-f), u <- u+delta.
Its fixed point is the ORIGINAL Poisson difference system (no splitting bias).
"""

import threading
from dataclasses import dataclass, field
from time import perf_counter

import numpy as np
from numba import njit

from .control import prepare_control
from .problem import Problem, blank_field, laplacian, validate_points

METHODS = {
    "lod_serial": "ЛОМ",
    "mc": "Монте-Карло",
    "mc_control": "MC с контрольным приближением",
    "reuse": "MК хвосты путей",
    "layers": "MК слои от границы",
    "wos": "WoS · 1-й род",
    "graded": "MC · сгущённая сетка",
}
LOD_STOPPING_RULE = "relative_rhs_inf_v1"
NUMBA_LOCK = threading.RLock()


@dataclass
class Result:
    method: str
    values: np.ndarray
    stderr: np.ndarray
    sample_count: np.ndarray
    elapsed: float = 0.0
    steps: int = 0
    walks: int = 0
    iterations: int = 0
    residual: float | None = None
    relative_residual: float | None = None
    converged: bool = True
    extras: dict = field(default_factory=dict)


@njit(inline="always")
def solve_tridiagonal(a, b, c, rhs):
    """Solve A[i]*v[i-1]+B[i]*v[i]+C[i]*v[i+1]=rhs[i].

    Endpoints outside the line are zero; nonzero Dirichlet values must first
    be moved into rhs. A[0] and C[-1] are unused.
    """
    size = len(rhs)
    if size == 0 or len(a) != size or len(b) != size or len(c) != size:
        raise ValueError("tridiagonal arrays must have the same nonzero length")
    alpha = np.zeros(size + 1)
    beta = np.zeros(size + 1)
    for index in range(size):
        denominator = a[index] * alpha[index] + b[index]
        if denominator == 0 or not np.isfinite(denominator):
            raise ValueError("invalid tridiagonal pivot")
        alpha[index + 1] = -c[index] / denominator
        beta[index + 1] = (rhs[index] - a[index] * beta[index]) / denominator
    solution = np.empty(size)
    solution[-1] = beta[size]
    for index in range(size - 2, -1, -1):
        solution[index] = alpha[index + 1] * solution[index + 1] + beta[index + 1]
    return solution


@njit(inline="always")
def _solve_line(
    rhs, solution, diffusion_ratio, axis, line_index, alpha, denominators, beta
):
    """Прогонка системы A*v[i-1] + B*v[i] + C*v[i+1] = F[i].

    Прогоночные коэффициенты задают v[i] = alpha[i+1]*v[i+1] + beta[i+1].
    Индексы внутренних узлов начинаются с 0. На обоих концах линии поправка
    равна нулю, поэтому alpha[0] = beta[0] = 0, а последний ответ = beta[size].
    """
    size = rhs.shape[0]
    a = -diffusion_ratio

    # Левая граница: поправка равна нулю.
    beta[0] = 0.0

    # Прямой ход: alpha[i+1] = -C / (A*alpha[i] + B),
    # beta[i+1] = (F[i] - A*beta[i]) / (A*alpha[i] + B).
    for index in range(size):
        rhs_value = rhs[line_index, index] if axis == 0 else rhs[index, line_index]
        beta[index + 1] = (rhs_value - a * beta[index]) / denominators[index]

    # Правая граница нулевая, поэтому v[size-1] = beta[size].
    if axis == 0:
        solution[line_index, size - 1] = beta[size]
    else:
        solution[size - 1, line_index] = beta[size]

    # Обратный ход: восстанавливаем решение от последнего узла к первому.
    for index in range(size - 2, -1, -1):
        if axis == 0:
            solution[line_index, index] = (
                alpha[index + 1] * solution[line_index, index + 1] + beta[index + 1]
            )
        else:
            solution[index, line_index] = (
                alpha[index + 1] * solution[index + 1, line_index] + beta[index + 1]
            )


@njit(cache=True)
def _line_coefficients(size, diffusion_ratio):
    alpha = np.zeros(size + 1)
    denominators = np.empty(size)
    for index in range(size):
        denominators[index] = 1 + 2 * diffusion_ratio - diffusion_ratio * alpha[index]
        alpha[index + 1] = diffusion_ratio / denominators[index]
    return alpha, denominators


@njit(cache=True, nogil=True)
def _sweep_into(rhs, solution, diffusion_ratio, axis, alpha, denominators, beta):
    for line_index in range(rhs.shape[0]):
        _solve_line(
            rhs, solution, diffusion_ratio, axis, line_index, alpha, denominators, beta
        )


@njit(cache=True, nogil=True)
def _sweep_serial(rhs, diffusion_ratio, axis):
    solution = np.empty_like(rhs)
    alpha, denominators = _line_coefficients(rhs.shape[0], diffusion_ratio)
    beta = np.empty(rhs.shape[0] + 1)
    _sweep_into(rhs, solution, diffusion_ratio, axis, alpha, denominators, beta)
    return solution


@njit(cache=True)
def _step(y, x):
    """Равновероятный переход в один из четырёх соседних узлов."""
    direction = np.random.randint(4)
    if direction == 0:
        y += 1
    elif direction == 1:
        y -= 1
    elif direction == 2:
        x += 1
    else:
        x -= 1
    return y, x


@njit(cache=True, nogil=True)
def _mc(f, boundary, points, walks, seed, max_steps):
    """Независимые пути из каждой целевой точки до первого выхода на границу."""
    np.random.seed(seed)
    n = f.shape[0] - 1
    source_weight = 1 / (4 * n * n)
    point_count = len(points)
    sums = np.zeros(point_count)
    squared_sums = np.zeros(point_count)
    total_steps = 0
    samples = np.empty(walks)

    for point_index in range(point_count):
        for walk_index in range(walks):
            y, x = points[point_index]
            reward = 0.0
            path_steps = 0

            # Внутри области накапливаем вклад источника -h²f/4.
            while 0 < y < n and 0 < x < n:
                if path_steps >= max_steps:
                    raise RuntimeError("Превышен предел шагов; результат не возвращён")
                reward -= source_weight * f[y, x]
                y, x = _step(y, x)
                path_steps += 1

            # На первом попадании на границу добавляем известное значение g.
            reward += boundary[y, x]
            sums[point_index] += reward
            squared_sums[point_index] += reward * reward
            total_steps += path_steps
            if point_index == 0:
                samples[walk_index] = reward
    return sums, squared_sums, np.full(point_count, walks), total_steps, samples


@njit(cache=True, nogil=True)
def _reuse(f, boundary, points, walks, seed, max_steps):
    """Дополнительно используем хвост после первого посещения каждой цели.

    Число новых путей и порядок случайных шагов совпадают с обычным MC.
    Каждый путь даёт не более одного наблюдения для каждой целевой точки.
    """
    np.random.seed(seed)
    n = f.shape[0] - 1
    source_weight = 1 / (4 * n * n)
    point_count = len(points)
    target_index = np.full(f.shape, -1, dtype=np.int64)
    for point_index in range(point_count):
        target_index[points[point_index, 0], points[point_index, 1]] = point_index
    sums = np.zeros(point_count)
    squared_sums = np.zeros(point_count)
    counts = np.zeros(point_count, dtype=np.int64)
    last_seen_walk = np.zeros(point_count, dtype=np.int64)
    source_before_visit = np.zeros(point_count)
    visited_targets = np.empty(point_count, dtype=np.int64)
    total_steps = 0
    walk_id = 0
    for start_index in range(point_count):
        for _ in range(walks):
            walk_id += 1
            y, x = points[start_index]
            reward = 0.0
            path_steps = 0
            visited_count = 0
            while 0 < y < n and 0 < x < n:
                if path_steps >= max_steps:
                    raise RuntimeError("Превышен предел шагов; результат не возвращён")
                point_index = target_index[y, x]
                # Запоминаем накопленную сумму до первого посещения цели.
                if point_index >= 0 and last_seen_walk[point_index] != walk_id:
                    last_seen_walk[point_index] = walk_id
                    source_before_visit[point_index] = reward
                    visited_targets[visited_count] = point_index
                    visited_count += 1
                reward -= source_weight * f[y, x]
                y, x = _step(y, x)
                path_steps += 1
            reward += boundary[y, x]
            total_steps += path_steps

            # Вклад хвоста равен полному вкладу за вычетом префикса пути.
            for visit_index in range(visited_count):
                point_index = visited_targets[visit_index]
                suffix_reward = reward - source_before_visit[point_index]
                sums[point_index] += suffix_reward
                squared_sums[point_index] += suffix_reward * suffix_reward
                counts[point_index] += 1
    return sums, squared_sums, counts, total_steps


@njit(cache=True, nogil=True)
def _layers_batch(f, boundary, max_depth, walks_per_node, seed, max_steps):
    """Один полный расчёт слоёв от внешней границы до глубины max_depth."""
    np.random.seed(seed)
    n = f.shape[0] - 1
    source_weight = 1 / (4 * n * n)
    values = boundary.copy()
    known = np.zeros(f.shape, dtype=np.bool_)
    known[0, :] = True
    known[-1, :] = True
    known[:, 0] = True
    known[:, -1] = True
    total_steps = 0
    total_walks = 0
    for depth in range(1, max_depth + 1):
        # Во время расчёта слоя известны только полностью готовые внешние слои.
        for start_y in range(1, n):
            for start_x in range(1, n):
                if min(start_y, start_x, n - start_y, n - start_x) != depth:
                    continue
                reward_sum = 0.0
                for _ in range(walks_per_node):
                    y = start_y
                    x = start_x
                    reward = 0.0
                    path_steps = 0
                    while not known[y, x]:
                        if path_steps >= max_steps:
                            raise RuntimeError(
                                "Превышен предел шагов; результат не возвращён"
                            )
                        reward -= source_weight * f[y, x]
                        y, x = _step(y, x)
                        path_steps += 1
                    reward_sum += reward + values[y, x]
                    total_steps += path_steps
                    total_walks += 1
                values[start_y, start_x] = reward_sum / walks_per_node

        # Открываем новый слой для блужданий только после расчёта всех его узлов.
        for start_y in range(1, n):
            for start_x in range(1, n):
                if min(start_y, start_x, n - start_y, n - start_x) == depth:
                    known[start_y, start_x] = True
    return values, total_steps, total_walks


def _put_stats(problem, points, sums, squared_sums, count):
    """Размещаем средние, стандартные ошибки и числа наблюдений на сетке."""
    values = blank_field(problem)
    standard_errors = np.full_like(values, np.nan)
    sample_counts = np.zeros_like(values, dtype=np.int64)
    mean = sums / count
    variance = np.maximum(0, (squared_sums - count * mean**2) / (count - 1))
    y, x = points.T
    values[y, x] = mean
    standard_errors[y, x] = np.sqrt(variance / count)
    sample_counts[y, x] = count
    return values, standard_errors, sample_counts


def _solve_layers(problem, points, walks, batches, seed, max_steps):
    """Усредняем независимые полные расчёты слоёв и оцениваем их разброс."""
    depth = max(min(y, x, problem.n - y, problem.n - x) for y, x in points)
    ensemble = []
    total_steps = 0
    total_walks = 0

    for batch_index in range(batches):
        walks_per_node = walks // batches + int(batch_index < walks % batches)
        batch_seed = (seed + 104729 * batch_index) % 2**32
        values, batch_steps, batch_walks = _layers_batch(
            problem.f,
            problem.boundary,
            depth,
            walks_per_node,
            batch_seed,
            int(max_steps),
        )
        ensemble.append(values)
        total_steps += batch_steps
        total_walks += batch_walks

    ensemble = np.array(ensemble)
    y, x = points.T
    values = blank_field(problem)
    values[y, x] = ensemble[:, y, x].mean(axis=0)
    standard_errors = np.full_like(values, np.nan)
    standard_errors[y, x] = ensemble[:, y, x].std(axis=0, ddof=1) / np.sqrt(batches)
    sample_counts = np.zeros_like(values, dtype=np.int64)
    sample_counts[y, x] = batches

    # Для анимации сохраняем подготовленные слои, скрывая невычисленную область.
    prepared = ensemble.mean(axis=0)
    grid_y, grid_x = np.indices(prepared.shape)
    depth_map = np.minimum.reduce(
        [grid_y, grid_x, problem.n - grid_y, problem.n - grid_x]
    )
    prepared[depth_map > depth] = np.nan

    return Result(
        method="layers",
        values=values,
        stderr=standard_errors,
        sample_count=sample_counts,
        steps=total_steps,
        walks=total_walks,
        extras={
            "replicates": ensemble[:, y, x],
            "prepared_mean": prepared,
            "batches": batches,
            "maxdepth": int(depth),
            "prepared_nodes": total_walks // walks,
        },
    )


def _lod(problem, method, tol, max_iter, record):
    """Поправки к решению через две прогонки до относительного допуска tol."""
    u = blank_field(problem)
    u[1:-1, 1:-1] = 0.0
    tau = problem.h / (2 * np.pi)
    diffusion_ratio = tau / problem.h**2
    alpha, denominators = _line_coefficients(problem.n - 1, diffusion_ratio)
    beta = np.empty(problem.n)
    x_correction = np.empty((problem.n - 1, problem.n - 1))
    correction = np.empty_like(x_correction)
    # При нулевом внутреннем поле начальная невязка равна правой части b.
    # В неё уже входят граничные значения, поэтому нормировка работает при f=0.
    initial_residual = laplacian(u, problem.h) - problem.f[1:-1, 1:-1]
    rhs_norm = float(np.max(np.abs(initial_residual)))
    history = []
    residuals = []
    relative_residuals = []
    iterations = 0
    for iteration in range(max_iter + 1):
        # 1. Считаем невязку текущего поля и нормируем её на ||b||∞.
        if iteration == 0:
            residual = initial_residual
        else:
            residual = laplacian(u, problem.h) - problem.f[1:-1, 1:-1]
        residual_norm = float(np.max(np.abs(residual)))
        residuals.append(residual_norm)
        if rhs_norm > 0:
            relative_residual = residual_norm / rhs_norm
        elif residual_norm == 0:
            relative_residual = 0.0
        else:
            relative_residual = float("inf")
        relative_residuals.append(relative_residual)

        # Кадры для анимации: начало и итерации 1, 2, 4, 8, ...
        if record and (iteration == 0 or iteration & (iteration - 1) == 0):
            history.append((iteration, u.copy()))

        # 2. Проверяем относительный допуск и ограничение числа итераций.
        if relative_residual <= tol or iteration == max_iter:
            iterations = iteration
            break

        # 3. Решаем линии по x, затем по y и добавляем полученную поправку.
        _sweep_into(
            tau * residual, x_correction, diffusion_ratio, 0, alpha, denominators, beta
        )
        _sweep_into(
            x_correction, correction, diffusion_ratio, 1, alpha, denominators, beta
        )
        u[1:-1, 1:-1] += correction

    if record and (not history or history[-1][0] != iterations):
        history.append((iterations, u.copy()))
    return Result(
        method=method,
        values=u,
        stderr=np.full_like(u, np.nan),
        sample_count=np.zeros_like(u, dtype=np.int64),
        iterations=iterations,
        residual=residual_norm,
        relative_residual=relative_residual,
        converged=relative_residual <= tol,
        extras={
            "history": history,
            "residuals": np.array(residuals),
            "relative_residuals": np.array(relative_residuals),
            "residual_scale": rhs_norm,
            "stopping_rule": LOD_STOPPING_RULE,
            "tolerance": tol,
            "tau": tau,
        },
    )


def solve(
    p: Problem,
    points,
    method="mc",
    walks=2048,
    seed=42,
    tol=1e-7,
    max_iter=20000,
    batches=8,
    max_steps=None,
    record=True,
    coarse_n=None,
    control=None,
):
    """Проверяем параметры, запускаем выбранный метод и измеряем его время.

    points содержит индексы (y, x). ЛОМ использует относительный допуск tol,
    варианты Монте-Карло — заданное число новых путей walks.
    """
    points = validate_points(p.n, points)
    if method in ("wos", "graded"):
        from .stationary_extensions import solve_extended

        return solve_extended(p, points, method, walks, seed, max_steps)
    if method not in METHODS:
        raise ValueError(f"Неизвестный метод: {method}")
    if int(walks) != walks or walks < 2:
        raise ValueError("walks должно быть целым >= 2")
    if not np.isfinite(tol) or not 0 < tol < 1:
        raise ValueError("Относительный допуск tol должен быть между 0 и 1")
    if int(max_iter) != max_iter or max_iter < 1:
        raise ValueError("max_iter должно быть целым >=1")
    if int(seed) != seed or not 0 <= seed < 2**32:
        raise ValueError("seed: целое в [0,2**32)")
    limit = 100 * p.n * p.n if max_steps is None else max_steps
    if int(limit) != limit or limit < 1:
        raise ValueError("max_steps должно быть целым >=1")
    if method == "layers" and (
        int(batches) != batches or batches < 2 or walks < 2 * batches
    ):
        raise ValueError("Для слоёв batches >=2 и walks >=2*batches")
    boundary_points = np.any((points == 0) | (points == p.n), axis=1)
    if boundary_points.any():
        started = perf_counter()
        interior = points[~boundary_points]
        if len(interior):
            result = solve(
                p,
                interior,
                method,
                walks,
                seed,
                tol,
                max_iter,
                batches,
                max_steps,
                record,
                coarse_n,
                control,
            )
        else:
            result = Result(
                method,
                blank_field(p),
                np.zeros_like(p.f),
                np.zeros_like(p.f, dtype=np.int64),
            )
        indices = tuple(points[boundary_points].T)
        result.values[indices] = p.boundary[indices]
        result.stderr[indices] = 0
        if boundary_points[0] and method in ("mc", "mc_control"):
            result.extras["samples"] = np.full(int(walks), p.boundary[tuple(points[0])])
        result.elapsed = perf_counter() - started
        return result
    # Блокировка изолирует генератор случайных чисел между сессиями интерфейса.
    with NUMBA_LOCK:
        started_at = perf_counter()
        if method == "lod_serial":
            result = _lod(p, method, tol, max_iter, record)
        elif method == "mc_control":
            prepared = control if control is not None else prepare_control(p, coarse_n)
            prepared.validate(p)
            path_started = perf_counter()
            sums, squares, counts, steps, samples = _mc(
                prepared.source,
                prepared.boundary,
                points,
                int(walks),
                int(seed),
                int(limit),
            )
            path_seconds = perf_counter() - path_started
            values, errors, counts = _put_stats(p, points, sums, squares, counts)
            y, x = points.T
            values[y, x] += prepared.values[y, x]
            result = Result(
                method,
                values,
                errors,
                counts,
                steps=steps,
                walks=len(points) * walks,
                extras={
                    "samples": samples + prepared.values[tuple(points[0])],
                    "preparation_seconds": (
                        prepared.preparation_seconds if control is None else 0.0
                    ),
                    "control_build_seconds": prepared.preparation_seconds,
                    "path_seconds": path_seconds,
                    "coarse_n": prepared.coarse_n,
                    "control_reused": control is not None,
                },
            )
        elif method == "mc":
            sums, squared_sums, observation_counts, steps, samples = _mc(
                p.f, p.boundary, points, int(walks), int(seed), int(limit)
            )
            values, standard_errors, sample_counts = _put_stats(
                p, points, sums, squared_sums, observation_counts
            )
            result = Result(
                method=method,
                values=values,
                stderr=standard_errors,
                sample_count=sample_counts,
                steps=steps,
                walks=len(points) * walks,
                extras={"samples": samples},
            )
        elif method == "reuse":
            sums, squared_sums, observation_counts, steps = _reuse(
                p.f, p.boundary, points, int(walks), int(seed), int(limit)
            )
            values, standard_errors, sample_counts = _put_stats(
                p, points, sums, squared_sums, observation_counts
            )
            result = Result(
                method=method,
                values=values,
                stderr=standard_errors,
                sample_count=sample_counts,
                steps=steps,
                walks=len(points) * walks,
            )
        else:
            result = _solve_layers(p, points, walks, batches, seed, limit)
        result.elapsed = perf_counter() - started_at
    return result


_warmed = False


def warmup():
    """Explicit untimed JIT compilation. Call before measurements, not inside them."""
    global _warmed
    with NUMBA_LOCK:
        if _warmed:
            return
        from .problem import make_problem, select_points

        problem = make_problem(n=4)
        points = select_points(4)
        for method in METHODS:
            if method in ("wos", "graded"):
                continue
            solve(problem, points, method, walks=8, batches=2, max_iter=2, record=False)
        _warmed = True


def trace_walk(p, point=None, seed=2026, max_steps=None):
    """Отдельная траектория с историей вклада источника для визуализации."""
    y, x = (p.n // 2, p.n // 2) if point is None else point
    validate_points(p.n, [(y, x)])
    if y in (0, p.n) or x in (0, p.n):
        value = float(p.boundary[y, x])
        return dict(
            points=np.array([(y, x)]),
            source=np.array([0.0]),
            boundary=value,
            reward=value,
        )
    rng = np.random.default_rng(seed)
    path = [(y, x)]
    source_history = [0.0]
    source_sum = 0.0
    limit = max_steps or 100 * p.n * p.n
    for _ in range(limit):
        source_sum -= p.h**2 * p.f[y, x] / 4
        dy, dx = ((1, 0), (-1, 0), (0, 1), (0, -1))[rng.integers(4)]
        y += dy
        x += dx
        path.append((y, x))
        source_history.append(source_sum)
        if y in (0, p.n) or x in (0, p.n):
            return {
                "points": np.array(path),
                "source": np.array(source_history),
                "boundary": float(p.boundary[y, x]),
                "reward": float(source_sum + p.boundary[y, x]),
            }
    raise RuntimeError("Траектория не достигла границы в заданный предел шагов")
