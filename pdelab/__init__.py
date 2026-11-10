"""Reproducible laboratory for the Dirichlet problem Δu=f on a square."""

from .control import ControlApproximation, prepare_control
from .heat import HeatProblem, HeatResult, make_heat_problem, solve_heat
from .heat_mc import HeatMCResult, heat_mc_step, solve_heat_mc
from .operator import FrozenOperator
from .problem import Problem, make_problem, reference_solution, select_points
from .solvers import METHODS, solve, warmup

__all__ = [
    "ControlApproximation",
    "prepare_control",
    "HeatMCResult",
    "heat_mc_step",
    "solve_heat_mc",
    "HeatProblem",
    "HeatResult",
    "make_heat_problem",
    "solve_heat",
    "Problem",
    "FrozenOperator",
    "make_problem",
    "select_points",
    "reference_solution",
    "METHODS",
    "solve",
    "warmup",
]
