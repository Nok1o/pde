"""Ограничения синхронного MC в интерфейсе; CLI остаётся без этих лимитов."""

import numpy as np

from .problem3d import select_points3d

MAX_INTERACTIVE_WALKS = 2_000_000
MAX_INTERACTIVE_EXPECTED_STEPS = 50_000_000


def interactive_mc_workload(config, methods):
    """Число путей и верхняя оценка их ожидаемой суммарной длины.

    Для одной координаты V(i)=3*i*(n-i) уменьшается в среднем на 1 за
    трёхмерный шаг. Время выхода из куба не больше времени достижения
    границы любой координатой, поэтому E[T] <= min(V(x), V(y), V(z)).
    Это оценка математического ожидания, не предел фактического времени.
    """
    if not any(m != "lod_serial" for m in methods):
        return 0, 0
    points = select_points3d(
        config.n, config.mode, config.point, config.plane, config.index
    )
    interior = points[np.all((points > 0) & (points < config.n), axis=1)]
    walks = len(interior) * int(config.walks)
    steps = int(np.min(3 * interior * (config.n - interior), axis=1).sum())
    return walks, steps * int(config.walks)


def interactive_mc_limit_message(config, methods):
    walks, steps = interactive_mc_workload(config, methods)
    if walks <= MAX_INTERACTIVE_WALKS and steps <= MAX_INTERACTIVE_EXPECTED_STEPS:
        return None
    return (
        f"Слишком большой интерактивный MC: {walks:,} путей; "
        f"верхняя оценка ожидаемой работы — {steps:,} шагов. "
        "Лимиты интерфейса: 2 млн путей и 50 млн ожидаемых шагов. "
        "Уменьшите сетку или число путей, выберите точку/срез либо запустите через CLI."
    )
