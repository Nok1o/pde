"""Human-readable equations for the implemented problem presets."""

import streamlit as st

from .problem import PRESETS


def _number(value):
    """Format a configured float as a readable KaTeX number."""
    text = f"{value:.12g}"
    if "e" in text.lower():
        mantissa, exponent = text.lower().split("e")
        return rf"{mantissa}\cdot 10^{{{int(exponent)}}}"
    return text


def render_problem_info(cfg):
    """Render the currently selected conditions, before any calculation is run."""
    with st.expander(
        f"Формулы и описание: {PRESETS[cfg.problem]}",
        expanded=False,
        key="problem_info_expander_19"
        + str(f"Формулы и описание: {PRESETS[cfg.problem]}"),
    ):
        st.caption(
            "Все случаи — двумерные стационарные задачи в единичном квадрате: "
            "0 ≤ x ≤ 1, 0 ≤ y ≤ 1. Ищем поле u(x, y), например установившуюся температуру."
        )
        if cfg.problem == "hot_top":
            st.write(
                "**Наш стартовый простой случай.** Квадратная пластина с заданной температурой "
                "каждой стороны. По умолчанию верхняя сторона нагрета до 100, остальные — до 0, "
                "внутренних источников нет. Условия можно менять слева."
            )
        elif cfg.problem == "polynomial":
            st.write(
                "**Проверка на известном решении.** На всей границе значение равно нулю, "
                "а внутри задана неоднородная правая часть. Решение имеет форму гладкого купола."
            )
        elif cfg.problem == "sine":
            st.write(
                "**Проверка точности сетки.** На границе значение равно нулю. "
                "Внутри решение — гладкий синусоидальный купол с максимумом в центре."
            )
        elif cfg.problem == "affine":
            st.write(
                "**Проверка неоднородной границы.** Внутренних источников нет, "
                "но значения вдоль сторон меняются. Решение — наклонная плоскость."
            )

        st.markdown("**Уравнение внутри квадрата**")
        st.latex(
            r"\Delta u = \frac{\partial^2 u}{\partial x^2}"
            r" + \frac{\partial^2 u}{\partial y^2} = f(x,y)"
        )
        if cfg.problem == "hot_top":
            st.latex(rf"f(x,y) = {_number(cfg.source)}")
        elif cfg.problem == "polynomial":
            st.latex(r"f(x,y) = -2000\bigl[x(1-x)+y(1-y)\bigr]")
        elif cfg.problem == "sine":
            st.latex(r"f(x,y) = -200\pi^2\sin(\pi x)\sin(\pi y)")
        else:
            st.latex(r"f(x,y)=0")

        st.markdown("**Граничные условия — известные значения на сторонах**")
        if cfg.problem == "hot_top":
            st.latex(
                r"\begin{aligned}"
                rf"u(x,1)&={_number(cfg.top)} &&\text{{верх}}\\"
                rf"u(x,0)&={_number(cfg.bottom)} &&\text{{низ}}\\"
                rf"u(0,y)&={_number(cfg.left)} &&\text{{левая сторона}}\\"
                rf"u(1,y)&={_number(cfg.right)} &&\text{{правая сторона}}"
                r"\end{aligned}"
            )
        elif cfg.problem in ("polynomial", "sine"):
            st.latex(r"u(x,0)=u(x,1)=0,\qquad u(0,y)=u(1,y)=0")
        else:
            st.latex(
                r"\begin{aligned}"
                r"u(x,1)&=40+20x &&\text{верх}\\"
                r"u(x,0)&=10+20x &&\text{низ}\\"
                r"u(0,y)&=10+30y &&\text{левая сторона}\\"
                r"u(1,y)&=30+30y &&\text{правая сторона}"
                r"\end{aligned}"
            )

        if cfg.problem == "hot_top":
            if cfg.source == 0:
                st.markdown("**Проверка в центре**")
                center = sum(v / 4 for v in (cfg.top, cfg.bottom, cfg.left, cfg.right))
                st.latex(rf"u\!\left(\frac12,\frac12\right)={_number(center)}")
                st.write(
                    "При постоянных значениях сторон и f = 0 ответ в центре равен "
                    "среднему четырёх граничных значений. Это следует из симметрии квадрата."
                )
            else:
                st.write(
                    "При ненулевой правой части значение в центре уже не равно просто "
                    "среднему температур сторон. В программе проверяем его по прямому сеточному решению."
                )
            st.caption(
                "На каждой стороне условия заданы без углов. В углах берётся среднее двух "
                "соседних сторон; эти узлы не входят в уравнения внутренних точек пятиточечной схемы."
            )
            if cfg.source != 0:
                st.caption(
                    "Знак источника: в принятой записи Δu = f нагрев соответствует отрицательному f "
                    "(для теплопроводности с единичным коэффициентом)."
                )
        else:
            st.markdown("**Известное точное решение**")
            solutions = {
                "polynomial": r"u_*(x,y)=1000x(1-x)y(1-y)",
                "sine": r"u_*(x,y)=100\sin(\pi x)\sin(\pi y)",
                "affine": r"u_*(x,y)=10+20x+30y",
            }
            st.latex(solutions[cfg.problem])
            values = {"polynomial": "62.5", "sine": "100", "affine": "35"}
            st.latex(rf"u_*\!\left(\frac12,\frac12\right)={values[cfg.problem]}")
            descriptions = {
                "polynomial": "Этот пример проверяет знак и накопление правой части вдоль случайного пути. "
                "Для такого полинома центральные разности точно воспроизводят вторые производные "
                "в узлах: пространственной погрешности здесь нет, остаются ошибки решателей.",
                "sine": "Этот пример показывает погрешность самой сетки. Даже точно решённая разностная система "
                "отличается от непрерывного решения. При уменьшении шага вдвое пространственная ошибка "
                "асимптотически уменьшается примерно в четыре раза.",
                "affine": "Вторые производные линейной функции равны нулю, поэтому она удовлетворяет уравнению "
                "Лапласа. Пример проверяет обработку меняющихся граничных значений; "
                "центральные разности также воспроизводят это решение точно в узлах.",
            }
            st.write(descriptions[cfg.problem])
            st.caption(
                "u* — известное непрерывное решение: именно оно служит эталоном в таблицах "
                "и графиках ошибок. Относительная ошибка в таблице тоже считается к этому эталону."
            )
