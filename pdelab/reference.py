"""One reference-selection policy for metrics, plots and exported results."""

from dataclasses import dataclass

import numpy as np

REFERENCE_POLICY = "exact-first-v1"


@dataclass(frozen=True)
class ComparisonReference:
    values: np.ndarray
    kind: str
    label: str
    description: str
    field_values: np.ndarray
    field_label: str


def choose_reference(problem, points, grid_reference):
    if problem.exact is not None:
        label = "Точное аналитическое решение"
        return ComparisonReference(
            problem.exact,
            "analytic",
            label,
            "Ошибки считаются относительно известного непрерывного решения; "
            "они включают погрешность сетки и ошибку самого метода.",
            problem.exact,
            label,
        )

    # For constant edge values and zero source, symmetry gives the exact centre
    # value even though a full analytic field is not supplied by this preset.
    n = problem.n
    sides = (
        problem.boundary[0, 1:-1],
        problem.boundary[-1, 1:-1],
        problem.boundary[1:-1, 0],
        problem.boundary[1:-1, -1],
    )
    centre_only = (
        n % 2 == 0 and len(points) == 1 and np.array_equal(points[0], [n // 2, n // 2])
    )
    if (
        centre_only
        and np.all(problem.f[1:-1, 1:-1] == 0)
        and all(np.all(side == side[0]) for side in sides)
    ):
        values = np.full_like(grid_reference, np.nan)
        values[n // 2, n // 2] = sum(float(side[0]) / 4 for side in sides)
        return ComparisonReference(
            values,
            "analytic_point",
            "Точный ответ в центре (по симметрии)",
            "Ошибки в центре считаются относительно среднего значений четырёх сторон. "
            "Полное поле на фоновых графиках — прямое сеточное решение; "
            "точный ответ здесь задан только в центре.",
            grid_reference,
            "Прямое сеточное решение (фон)",
        )

    label = "Прямое сеточное решение"
    return ComparisonReference(
        grid_reference,
        "grid",
        label,
        "Точное решение для всего выбранного набора узлов в программе не задано. "
        "Эталон — прямое решение разностной системы; погрешность самой сетки в эту ошибку не входит.",
        grid_reference,
        label,
    )
