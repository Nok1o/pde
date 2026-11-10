"""Reproducible laboratory for the Dirichlet problem Δu=f on a square."""

from .control import ControlApproximation, prepare_control
from .problem import Problem, make_problem, reference_solution, select_points
from .solvers import METHODS, solve, warmup

__all__ = [
    "ControlApproximation",
    "prepare_control",
    "Problem",
    "make_problem",
    "select_points",
    "reference_solution",
    "METHODS",
    "solve",
    "warmup",
]
