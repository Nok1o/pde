"""Backend interfaces for #25--#28; these are contracts, not implementations.

See docs/heat3d-contract.md for row conventions, stopping and timing rules.
Old 2D and stationary classes remain unchanged; future adapters are explicit.
"""

from dataclasses import dataclass
from typing import Literal, Protocol

import numpy as np
from scipy.sparse import csr_matrix

from .heat3d_problem import HeatProblem3D

# Match BoundaryOperator3D's array-axis neighbor order, not coordinate order.
NEIGHBOR_OFFSETS_ZYX = (
    (1, 0, 0),
    (-1, 0, 0),
    (0, 1, 0),
    (0, -1, 0),
    (0, 0, 1),
    (0, 0, -1),
)
Status3D = Literal[
    "completed",
    "linear_limit",
    "outer_limit",
    "walk_budget",
    "path_limit",
    "time_limit",
    "memory_limit",
    "nonfinite",
    "unsupported",
]


class FrozenHeatOperator3D(Protocol):
    """Immutable snapshot; full-grid rows with essential values eliminated.

    weights[..., direction] >= 0 couples only free neighboring nodes.
    fixed rows are identity; diagonal[fixed]=1, rhs[fixed]=fixed_values[fixed].
    Known boundary contributions are already in rhs. No double payment in MC.
    """

    diagonal: np.ndarray  # [z,y,x], positive
    weights: np.ndarray  # [z,y,x,6], zero outside cube and toward fixed nodes
    rhs: np.ndarray  # adjusted load, including known boundary contributions
    fixed: np.ndarray  # bool [z,y,x]
    fixed_values: np.ndarray  # [z,y,x], zero outside fixed nodes
    volumes: np.ndarray  # positive dual volumes, independent of fixed rows

    def apply(self, values: np.ndarray) -> np.ndarray:
        """Full-grid A*u, identity rows at fixed nodes."""
        ...

    def assemble(self) -> tuple[csr_matrix, np.ndarray]:
        """Full CSR and rhs, flattened in C order (x fastest); small grids only."""
        ...

    def with_rhs(self, rhs: np.ndarray) -> "FrozenHeatOperator3D":
        """New snapshot with the same matrix and fixed values rhs[fixed].

        The supplied rhs is already adjusted, e.g. rhs - apply(control).
        Do not add boundary loads again or mutate the original snapshot.
        """
        ...


class HeatOperatorFactory3D(Protocol):
    def __call__(
        self,
        problem: HeatProblem3D,
        previous: np.ndarray,
        iterate: np.ndarray,
        t: float,
        dt: float,
    ) -> FrozenHeatOperator3D:
        """Freeze k(iterate) for the layer ending at t, using actual final dt.

        No exact solution access. Includes physical face data at t.
        Implemented by #25 (all kind 1), then #27 (mixed boundaries).
        """
        ...


@dataclass
class LinearResult3D:
    values: np.ndarray  # full-grid candidate, not necessarily converged
    status: Status3D
    reason: str
    relative_residual: float
    iterations: int = 0  # complete xyz cycles; one full solve for a direct backend
    walks: int = 0  # paths started, including an unfinished path on failure
    transitions: int = 0  # executed transitions, including failed paths


class LinearBackend3D(Protocol):
    def __call__(
        self,
        operator: FrozenHeatOperator3D,
        initial: np.ndarray,
        *,
        tolerance: float,
        max_iterations: int,
    ) -> LinearResult3D:
        """Deterministic sweep/block backend; MC adds its explicit budget API in #28."""
        ...


@dataclass
class HeatRunResult3D:
    method: str
    status: Status3D
    reason: str
    target_time: float
    completed_until: float
    points_zyx: np.ndarray  # integer [query,3], including boundary nodes
    values: np.ndarray | None  # [query] only after reaching target_time
    physical_steps: int  # accepted time layers only
    outer_iterations: int  # attempted Picard updates, including rejected ones
    inner_iterations: int  # sum of LinearResult3D.iterations across all attempts
    walks: int
    transitions: int
    seconds: float  # total solve, including all preparation, excluding JIT/exact/UI
    preparation_seconds: float  # subset of seconds, never added to it again
    workspace_bytes: int  # estimate of numerical storage, NOT measured peak RSS
    peak_rss_bytes: int | None  # null unless measured; process baseline documented
    history: list[dict]  # per-layer/outer diagnostics, including failed candidate
    diagnostic_field: np.ndarray | None = (
        None  # full last candidate, never a solution at T
    )
    diagnostic_time: float | None = None  # attempted layer time of diagnostic_field
    assembly_seconds: float = 0.0  # block extraction, included in preparation_seconds
    factorization_seconds: float = 0.0  # block LU/Schur setup, included in seconds
    solve_seconds: float = 0.0  # block forward/back substitution, included in seconds
    factorizations: int = 0  # completed block factorizations of the whole matrix
    factor_reuses: int = 0  # solves reusing a previously prepared identical matrix
    completed_walks: int = 0
    warmup_seconds: float = 0.0  # excluded from seconds
    validation_walks: int = 0  # included in walks
    coarse_iterations: int = 0  # included in inner_iterations
    coarse_seconds: float = 0.0  # included in preparation_seconds
    interpolation_seconds: float = 0.0
    defect_seconds: float = 0.0
    batch_size: int | None = None  # only grouped MC, including failed/initial-only runs
    history_mode: str = "full"  # all outer records, or the latest record of each layer
    history_records_discarded: int = 0  # intermediate diagnostics removed, not work
