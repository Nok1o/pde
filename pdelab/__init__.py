"""Reproducible Poisson laboratory with grid solvers and Monte Carlo."""

from .boundary import BoundaryCondition, BoundaryOperator, BoundaryProblem
from .boundary3d import BoundaryOperator3D, BoundaryProblem3D
from .control import ControlApproximation, prepare_control
from .heat import HeatProblem, HeatResult, make_heat_problem, solve_heat
from .heat_mc import HeatMCResult, heat_mc_step, solve_heat_mc
from .nonlinear import (
    NonlinearProblem,
    NonlinearSettings,
    make_nonlinear_problem,
    solve_nonlinear,
)
from .operator import FrozenOperator
from .problem import Problem, make_problem, reference_solution, select_points
from .solvers import METHODS, solve, warmup

__all__ = [
    "BoundaryCondition",
    "BoundaryOperator",
    "BoundaryProblem",
    "BoundaryOperator3D",
    "BoundaryProblem3D",
    "ControlApproximation",
    "prepare_control",
    "HeatMCResult",
    "heat_mc_step",
    "solve_heat_mc",
    "HeatProblem",
    "HeatResult",
    "make_heat_problem",
    "solve_heat",
    "NonlinearProblem",
    "NonlinearSettings",
    "make_nonlinear_problem",
    "solve_nonlinear",
    "Problem",
    "FrozenOperator",
    "make_problem",
    "select_points",
    "reference_solution",
    "METHODS",
    "solve",
    "warmup",
]
