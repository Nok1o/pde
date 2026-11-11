"""Independent calibration and confirmation for mixed-boundary point queries."""

from dataclasses import asdict, replace
from time import perf_counter

import numpy as np

from .boundary import BoundaryOperator
from .boundary3d import BoundaryOperator3D
from .boundary_experiment import problem_from_config, selected_boundary_methods
from .boundary_experiment3d import BoundaryConfig3D, problem_from_config3d
from .boundary_solvers import select_boundary_points, solve_boundary, warmup_boundary
from .boundary_solvers3d import solve_boundary3d, warmup_boundary3d
from .problem3d import DIRECT_MAX_N
from .research import LOD_SEARCH_TOLERANCES, TARGETS_PCT
from .study_policy import SELECTION_POLICY, render_accuracy_results, search_budgets


def boundary_accuracy_study(
    config,
    methods=None,
    repeats=5,
    budgets=(64, 256, 1024, 4096),
    tolerances=LOD_SEARCH_TOLERANCES,
    seconds_per_method=60.0,
):
    """Select on pilot RMSE; confirm once on independent whole runs, never retune."""
    if isinstance(repeats, bool) or int(repeats) != repeats or repeats < 2:
        raise ValueError("Нужно минимум два независимых повтора")
    if not np.isfinite(seconds_per_method) or seconds_per_method <= 0:
        raise ValueError("Лимит времени должен быть положительным")
    config = replace(config, mode="point")
    is3d = isinstance(config, BoundaryConfig3D)
    make = problem_from_config3d if is3d else problem_from_config
    operator_type = BoundaryOperator3D if is3d else BoundaryOperator
    solver = solve_boundary3d if is3d else solve_boundary
    problem = make(config)
    operator = operator_type(problem)
    operator.check_compatibility()
    methods = selected_boundary_methods(operator, methods)
    if not tolerances and "lod_serial" in methods:
        raise ValueError("Выберите допуски ЛОМ")
    if not budgets and any(m != "lod_serial" for m in methods):
        raise ValueError("Выберите бюджеты MC")
    if any(int(b) != b or b < 2 for b in budgets):
        raise ValueError("Бюджеты MC должны быть целыми >= 2")
    if any(not 0 < t < 1 for t in tolerances):
        raise ValueError("Допуски должны лежать между 0 и 1")
    if is3d:
        from .boundary3d import select_boundary_points3d

        points = select_boundary_points3d(config.n, "point", config.point)
    else:
        points = select_boundary_points(config.n, "point", config.point)
    if problem.exact is None and is3d and problem.n > DIRECT_MAX_N:
        raise ValueError(f"Для сеточного эталона требуется n≤{DIRECT_MAX_N}")
    reference = (
        operator.solve() if problem.exact is None else operator.normalize(problem.exact)
    )
    target = float(reference[tuple(points[0])])
    (warmup_boundary3d if is3d else warmup_boundary)()
    trials, rows = [], []

    def measure(method, budget, budget_index, phase, deadline):
        values, times = [], []
        for repeat in range(int(repeats)):
            if perf_counter() >= deadline:
                return None, "timeout"
            seed = int(
                np.random.SeedSequence(
                    [config.seed, budget_index, phase, repeat]
                ).generate_state(1)[0]
            )
            start = perf_counter()
            try:
                result = solver(
                    problem,
                    points,
                    method,
                    seed=seed,
                    walks=config.walks if method == "lod_serial" else int(budget),
                    tol=float(budget) if method == "lod_serial" else config.tol,
                    max_iter=config.max_iter,
                    max_steps=config.max_steps,
                )
                value = float(result.values[tuple(points[0])])
                ok = result.converged and np.isfinite(value)
                reason = "ok" if ok else "not_converged"
                elapsed = result.elapsed
            except (ValueError, RuntimeError) as exc:
                value, ok, reason = None, False, str(exc)
                elapsed = perf_counter() - start
            trials.append(
                dict(
                    method=method,
                    budget=budget,
                    phase=phase,
                    repeat=repeat,
                    seed=seed,
                    seconds=elapsed,
                    value=value,
                    status=reason,
                )
            )
            if not ok:
                return None, reason
            values.append(value)
            times.append(elapsed)
        error = float(
            100 * np.sqrt(np.mean((np.array(values) - target) ** 2)) / abs(target)
        )
        return dict(error=error, seconds=float(np.mean(times))), "ok"

    for method in methods:
        if target == 0:
            rows.extend(
                dict(
                    method=method,
                    target_pct=t,
                    status="zero_reference",
                    selection_policy=SELECTION_POLICY,
                    seconds=None,
                    relative_error_pct=None,
                    budget=None,
                )
                for t in TARGETS_PCT
            )
            continue

        def series(budget, index, phase, deadline):
            measurement, reason = measure(
                method, budget, index, int(phase == "validation"), deadline
            )
            if measurement is None:
                if reason == "timeout":
                    raise TimeoutError("Лимит времени")
                raise RuntimeError(reason)
            return dict(
                relative_error_pct=measurement["error"], seconds=measurement["seconds"]
            )

        method_rows = search_budgets(
            method,
            tolerances if method == "lod_serial" else budgets,
            series,
            seconds_per_method,
            fastest=True,
            clock=perf_counter,
        )
        for row in method_rows:
            row["status_detail"] = row["status"]
            row["status"] = row["status_code"]
        rows.extend(method_rows)
    return dict(
        selection_policy=SELECTION_POLICY,
        config=asdict(config),
        repeats=repeats,
        budgets=list(budgets),
        tolerances=list(tolerances),
        seconds_per_phase=seconds_per_method,
        reference_kind="grid" if problem.exact is None else "analytic",
        reference_value=target,
        rows=rows,
        trials=trials,
    )


def render_boundary_research(config, methods, key):
    """Identical research controls and exports for the square and cube."""
    import json

    import pandas as pd
    import plotly.express as px
    import streamlit as st

    from .boundary_solvers import METHODS_BOUNDARY
    from .experiment3d import rows_csv

    dimension = 3 if isinstance(config, BoundaryConfig3D) else 2
    n = st.select_slider(
        "Сетка исследования",
        [4, 8, 12, 16, 24, 32, 48, 64],
        value=config.n,
        key=key + "_n",
    )
    for coordinate in "xyz"[:dimension]:
        if key + coordinate in st.session_state:
            st.session_state[key + coordinate] = min(
                n, max(0, st.session_state[key + coordinate])
            )
    point = tuple(
        st.number_input(
            f"Узел {axis}", min_value=0, max_value=n, value=n // 2, key=key + axis
        )
        for axis in "xyz"[:dimension]
    )
    repeats = st.number_input(
        "Число усреднений", min_value=2, max_value=100, value=5, key=key + "_repeats"
    )
    budgets = st.multiselect(
        "Бюджеты MC",
        [64, 256, 1024, 4096, 16384, 65536],
        default=[64, 256, 1024, 4096],
        key=key + "_budgets",
    )
    seconds = st.number_input(
        "Лимит одной фазы на метод, с",
        min_value=1.0,
        max_value=1800.0,
        value=60.0,
        key=key + "_seconds",
    )
    selected = replace(config, n=n, point=point, mode="point")
    signature = (asdict(selected), methods, repeats, budgets, seconds)
    if st.button("Запустить исследование", key=key + "_run", disabled=not methods):
        try:
            with st.spinner("Подбор и независимая проверка…"):
                result = boundary_accuracy_study(
                    selected, methods, repeats, budgets, seconds_per_method=seconds
                )
                st.session_state[key] = (signature, result)
        except (ValueError, RuntimeError) as exc:
            st.error(str(exc))
    saved = st.session_state.get(key)
    if saved is None:
        return
    if saved[0] != signature:
        st.info("Настройки изменены. Запустите исследование заново.")
    result = saved[1]
    rows = pd.DataFrame(result["rows"])
    rows["Метод"] = rows.method.map(METHODS_BOUNDARY)
    render_accuracy_results(result["rows"], METHODS_BOUNDARY, key=key + "_results")
    st.plotly_chart(
        px.scatter(
            rows,
            x="seconds",
            y="relative_error_pct",
            color="Метод",
            hover_data=["status", "budget"],
        ),
        key=key + "_plot",
    )
    with st.expander("Статусы и параметры", key="boundary_research_expander_227"):
        st.dataframe(rows, hide_index=True)
        st.caption(
            "Ошибка — RMS отдельных независимых запусков; время выводится только при подтверждённой точности. Нулевой эталон не допускает процентной ошибки."
        )
    st.download_button(
        "Скачать исследование CSV",
        rows_csv(result["rows"]),
        "accuracy.csv",
        key=key + "_csv",
        on_click="ignore",
    )
    st.download_button(
        "Скачать полные серии JSON",
        json.dumps(result, ensure_ascii=False, allow_nan=False),
        "accuracy.json",
        key=key + "_json",
        on_click="ignore",
    )
