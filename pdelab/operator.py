"""Frozen conservative system -div(k grad u) + sigma*u = b on a square.

Arrays use [y, x]. Arithmetic face averages are second order for smooth k;
discontinuous materials require a separate interface model.
"""

from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve


@dataclass(frozen=True)
class FrozenOperator:
    """Shared coefficients: x faces (n+1,n), y faces (n,n+1), nodes (n+1,n+1).

    Face arrays contain w=k_face/h², including edges touching Dirichlet nodes.
    Construct a new instance when freezing a new conductivity field k(u).
    """

    wx: np.ndarray
    wy: np.ndarray
    sigma: np.ndarray
    b: np.ndarray
    boundary: np.ndarray

    def __post_init__(self):
        shape = np.shape(self.b)
        if len(shape) != 2 or shape[0] != shape[1] or shape[0] < 3:
            raise ValueError("b must be a square nodal array with interior nodes")
        n = shape[0] - 1
        shapes = dict(
            wx=(n + 1, n), wy=(n, n + 1), sigma=shape, b=shape, boundary=shape
        )
        for name, expected in shapes.items():
            a = np.array(getattr(self, name), dtype=float, copy=True)
            if a.shape != expected or not np.isfinite(a).all():
                raise ValueError(f"{name}: expected finite array of shape {expected}")
            if name in ("wx", "wy") and np.any(a <= 0):
                raise ValueError("conductivity weights must be positive")
            if name == "sigma" and np.any(a < 0):
                raise ValueError("sigma must be nonnegative")
            a.setflags(write=False)
            object.__setattr__(self, name, a)
        if not np.isfinite(self.diagonal).all():
            raise ValueError("operator diagonal must be finite")

    @classmethod
    def from_nodes(cls, k, b, boundary, sigma=0.0):
        """Freeze nodal k, using k_face=(k_left+k_right)/2 on every edge."""
        k = np.asarray(k, dtype=float)
        if k.ndim != 2 or k.shape[0] != k.shape[1] or k.shape[0] < 3:
            raise ValueError("k must be a square nodal array")
        if not np.isfinite(k).all() or np.any(k <= 0):
            raise ValueError("k must be finite and positive")
        scale = (k.shape[0] - 1) ** 2
        return cls(
            (k[:, :-1] / 2 + k[:, 1:] / 2) * scale,
            (k[:-1, :] / 2 + k[1:, :] / 2) * scale,
            np.broadcast_to(sigma, k.shape),
            np.broadcast_to(b, k.shape),
            np.broadcast_to(boundary, k.shape),
        )

    @classmethod
    def from_problem(cls, problem):
        """Adapter for the legacy Δu=f API: k=1, sigma=0, b=-f."""
        return cls.from_nodes(np.ones_like(problem.f), -problem.f, problem.boundary)

    @property
    def weights(self):
        """Interior neighbor weights, in order down, up, right, left."""
        return np.stack(
            (
                self.wy[1:, 1:-1],
                self.wy[:-1, 1:-1],
                self.wx[1:-1, 1:],
                self.wx[1:-1, :-1],
            ),
            axis=-1,
        )

    @property
    def diagonal(self):
        return self.sigma[1:-1, 1:-1] + self.weights.sum(axis=-1)

    def fluxes(self, u):
        """Oriented edge fluxes k*(u_left-u_right)/h; reverse flux is negative."""
        u = np.asarray(u, dtype=float)
        if u.shape != self.b.shape or not np.isfinite(u).all():
            raise ValueError("u must be a finite nodal field")
        h = 1 / (u.shape[0] - 1)
        return (
            self.wx * (u[:, :-1] - u[:, 1:]) * h,
            self.wy * (u[:-1, :] - u[1:, :]) * h,
        )

    def apply(self, u):
        """Apply the full operator, returning interior nodes only."""
        u = np.asarray(u, dtype=float)
        qx, qy = self.fluxes(u)
        n = self.b.shape[0] - 1
        return (
            n * (qx[1:-1, 1:] - qx[1:-1, :-1] + qy[1:, 1:-1] - qy[:-1, 1:-1])
            + self.sigma[1:-1, 1:-1] * u[1:-1, 1:-1]
        )

    def assemble(self):
        """Independently assemble interior CSR matrix and Dirichlet-adjusted RHS."""
        n = self.b.shape[0] - 1
        m = n - 1
        indices = np.arange(m * m).reshape(m, m)
        horizontal = -self.wx[1:-1, 1:-1].ravel()
        vertical = -self.wy[1:-1, 1:-1].ravel()
        diagonal = (
            self.sigma[1:-1, 1:-1]
            + self.wx[1:-1, :-1]
            + self.wx[1:-1, 1:]
            + self.wy[:-1, 1:-1]
            + self.wy[1:, 1:-1]
        ).ravel()
        rows = np.concatenate(
            (
                indices.ravel(),
                indices[:, :-1].ravel(),
                indices[:, 1:].ravel(),
                indices[:-1, :].ravel(),
                indices[1:, :].ravel(),
            )
        )
        cols = np.concatenate(
            (
                indices.ravel(),
                indices[:, 1:].ravel(),
                indices[:, :-1].ravel(),
                indices[1:, :].ravel(),
                indices[:-1, :].ravel(),
            )
        )
        data = np.concatenate((diagonal, horizontal, horizontal, vertical, vertical))
        rhs = self.b[1:-1, 1:-1].copy()
        rhs[0, :] += self.wy[0, 1:-1] * self.boundary[0, 1:-1]
        rhs[-1, :] += self.wy[-1, 1:-1] * self.boundary[-1, 1:-1]
        rhs[:, 0] += self.wx[1:-1, 0] * self.boundary[1:-1, 0]
        rhs[:, -1] += self.wx[1:-1, -1] * self.boundary[1:-1, -1]
        return (
            coo_matrix((data, (rows, cols)), shape=(m * m, m * m)).tocsr(),
            rhs.ravel(),
        )

    def solve(self):
        matrix, rhs = self.assemble()
        u = self.boundary.copy()
        u[1:-1, 1:-1] = spsolve(matrix, rhs).reshape(u.shape[0] - 2, -1)
        return u

    def mc_coefficients(self):
        """Return (P_down/up/right/left, source reward, zero-payoff killing P)."""
        d = self.diagonal
        return (
            self.weights / d[..., None],
            self.b[1:-1, 1:-1] / d,
            self.sigma[1:-1, 1:-1] / d,
        )
