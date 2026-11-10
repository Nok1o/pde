"""Reproducible Poisson laboratory with grid solvers and Monte Carlo."""

from .boundary import BoundaryCondition, BoundaryOperator, BoundaryProblem
from .control import ControlApproximation, prepare_control
from .operator import FrozenOperator
from .problem import Problem, make_problem, reference_solution, select_points
from .solvers import METHODS, solve, warmup

__all__ = [
    "BoundaryCondition",
    "BoundaryOperator",
    "BoundaryProblem",
    "ControlApproximation",
    "prepare_control",
    "Problem",
    "FrozenOperator",
    "make_problem",
    "select_points",
    "reference_solution",
    "METHODS",
    "solve",
    "warmup",
]
