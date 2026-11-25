"""Independent calibration and confirmation for mixed-boundary point queries."""

from dataclasses import asdict, replace
from time import perf_counter

import numpy as np

from .boundary import BoundaryOperator
from .boundary3d import BoundaryOperator3D
from .boundary_experiment import problem_from_config, selected_boundary_methods
from .boundary_experiment3d import (
    BoundaryConfig3D,
    problem_from_config3d,
    selected_boundary_methods3d,
)
from .boundary_solvers import select_boundary_points, solve_boundary, warmup_boundary
from .boundary_solvers3d import METHODS_BOUNDARY_3D, solve_boundary3d, warmup_boundary3d
from .problem3d import DIRECT_MAX_N
from .research import LOD_SEARCH_TOLERANCES
from .stationary_study3d import study_stationary3d
from .study_policy import (
    ACCURACY_DESCRIPTION,
    ACCURACY_METRIC,
    SELECTION_POLICY,
    phase_seed_index,
    render_accuracy_results,
    search_budgets,
    study_error_metrics,
)


def boundary_accuracy_study(
    config,
    methods=None,
    repeats=5,
    budgets=(64, 256, 1024, 4096),
    tolerances=LOD_SEARCH_TOLERANCES,
    seconds_per_method=60.0,
    progress=None,
):
    """Fixed pilot/final protocol in 3D; preserve legacy 2D study semantics."""
    if isinstance(repeats, bool) or int(repeats) != repeats or repeats < 2:
        raise ValueError("Нужно минимум два независимых повтора")
    if not np.isfinite(seconds_per_method) or seconds_per_method <= 0:
        raise ValueError("Лимит времени должен быть положительным")
    is3d = isinstance(config, BoundaryConfig3D)
    if not is3d:
        config = replace(config, mode="point")
    make = problem_from_config3d if is3d else problem_from_config
    operator_type = BoundaryOperator3D if is3d else BoundaryOperator
    solver = solve_boundary3d if is3d else solve_boundary
    problem = make(config)
    operator = operator_type(problem)
    operator.check_compatibility()
    select_methods = selected_boundary_methods3d if is3d else selected_boundary_methods
    methods = select_methods(operator, methods)
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

        points = select_boundary_points3d(
            config.n, config.mode, config.point, config.plane, config.index
        )
    else:
        points = select_boundary_points(config.n, "point", config.point)
    if problem.exact is None and is3d and problem.n > DIRECT_MAX_N:
        raise ValueError(f"Для сеточного эталона требуется n≤{DIRECT_MAX_N}")
    reference = (
        operator.solve() if problem.exact is None else operator.normalize(problem.exact)
    )
    if is3d:
        target_values = reference[tuple(points.T)]
        warmup_boundary3d(methods)
        study = study_stationary3d(
            problem,
            points,
            target_values,
            methods,
            repeats=repeats,
            seed=config.seed,
            batch_size=config.batch_size,
            walk_budgets=budgets,
            tolerances=tolerances,
            max_iter=config.max_iter,
            max_steps=config.max_steps,
            seconds_per_phase=seconds_per_method,
            solver=solver,
            clock=perf_counter,
            progress=progress,
        )
        for row in study["rows"]:
            row["status"] = row["status_code"]
        return dict(
            **study,
            config=asdict(config),
            repeats=repeats,
            budgets=list(budgets),
            tolerances=list(tolerances),
            reference_kind="grid" if problem.exact is None else "analytic",
            points_zyx=points.tolist(),
            reference_value=(
                float(target_values[0]) if len(points) == 1 else target_values.tolist()
            ),
        )
    warmup_boundary()
    target = float(reference[tuple(points[0])])
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
            result = None
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
                    absolute_error=(
                        study_error_metrics([value], target)["absolute_error"]
                        if ok
                        else None
                    ),
                    walks=(
                        result.walks
                        if result is not None and method != "lod_serial"
                        else None
                    ),
                    steps=result.steps if result is not None else None,
                )
            )
            if not ok:
                return None, reason
            values.append(value)
            times.append(elapsed)
        return (
            dict(
                **study_error_metrics(values, target),
                seconds=float(np.mean(times)),
                completed_repeats=len(values),
                requested_repeats=repeats,
            ),
            "ok",
        )

    for method in methods:

        def series(budget, index, phase, deadline):
            measurement, reason = measure(
                method, budget, index, phase_seed_index(phase), deadline
            )
            if measurement is None:
                if reason == "timeout":
                    raise TimeoutError("Лимит времени")
                raise RuntimeError(reason)
            return measurement

        method_rows = search_budgets(
            method,
            tolerances if method == "lod_serial" else budgets,
            series,
            seconds_per_method,
            fastest=True,
            clock=perf_counter,
            relative_applicable=bool(target != 0),
        )
        for row in method_rows:
            row["status_detail"] = row["status"]
            row["status"] = row["status_code"]
            if target == 0:
                recorded = [t for t in trials if t["method"] == method]
                row.update(
                    requested_repeats=repeats,
                    recorded_repeats=len(recorded),
                    completed_repeats=sum(t["status"] == "ok" for t in recorded),
                    spent_seconds=sum(t["seconds"] for t in recorded),
                )
        rows.extend(method_rows)
    return dict(
        selection_policy=SELECTION_POLICY,
        accuracy_metric=ACCURACY_METRIC,
        accuracy=ACCURACY_DESCRIPTION,
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
    names = METHODS_BOUNDARY_3D if dimension == 3 else METHODS_BOUNDARY
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
    if dimension == 3 and any(
        m in ("mc_grouped", "mc_grouped_cached") for m in methods
    ):
        batch_size = st.number_input(
            "Размер группы частиц исследования",
            min_value=1,
            max_value=65536,
            value=config.batch_size,
            key=key + "_batch_size",
        )
        selected = replace(selected, batch_size=int(batch_size))
    signature = (asdict(selected), methods, repeats, budgets, seconds)
    if st.button("Запустить исследование", key=key + "_run", disabled=not methods):
        progress_bar = st.progress(0.0)
        try:
            with st.spinner("Подбор и независимая проверка…"):
                result = boundary_accuracy_study(
                    selected,
                    methods,
                    repeats,
                    budgets,
                    seconds_per_method=seconds,
                    progress=lambda fraction, text: progress_bar.progress(
                        fraction, text=text
                    ),
                )
                st.session_state[key] = (signature, result)
        except (ValueError, RuntimeError) as exc:
            st.error(str(exc))
        finally:
            progress_bar.empty()
    saved = st.session_state.get(key)
    if saved is None:
        return
    if saved[0] != signature:
        st.info("Настройки изменены. Запустите исследование заново.")
    result = saved[1]
    rows = pd.DataFrame(result["rows"])
    rows["Метод"] = rows.method.map(names)
    render_accuracy_results(result["rows"], names, key=key + "_results")
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
            "Ошибка — среднее относительных ошибок отдельных независимых запусков; время выводится только при подтверждённой точности. Нулевой эталон не допускает процентной ошибки."
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
