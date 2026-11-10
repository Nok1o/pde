"""Reproducible laboratory for the Dirichlet problem Δu=f on a square."""

from .operator import FrozenOperator
from .problem import Problem, make_problem, reference_solution, select_points
from .solvers import METHODS, solve, warmup

__all__ = [
    "Problem",
    "FrozenOperator",
    "make_problem",
    "select_points",
    "reference_solution",
    "METHODS",
    "solve",
    "warmup",
]
