"""Independent numerical checks of the manufactured data, not future solvers."""

from dataclasses import replace

import numpy as np
import pytest

from pdelab.boundary3d import FACES
from pdelab.heat import time_grid
from pdelab.heat3d_contract import NEIGHBOR_OFFSETS_ZYX
from pdelab.heat3d_problem import HeatFace3D, make_heat_problem3d


def shifted(coords, component, distance):
    result = list(coords)
    result[component] = result[component] + distance
    return tuple(result)


def derivative(fn, coords, t, component, step):
    return (
        fn(*shifted(coords, component, step), t)
        - fn(*shifted(coords, component, -step), t)
    ) / (2 * step)


@pytest.mark.parametrize("kind", ["sine", "polynomial"])
@pytest.mark.parametrize("alpha", [0.0, 0.5])
@pytest.mark.parametrize(
    "gamma,sigma", [(0.0, 0.0), (0.4, 0.0), (0.0, 0.3), (0.4, 0.3)]
)
def test_source_against_numerical_flux_divergence(kind, alpha, gamma, sigma):
    problem = make_heat_problem3d(
        kind, alpha=alpha, capacity_gamma=gamma, reaction_sigma=sigma
    )
    coords = (np.array([0.17, 0.43, 0.72]), np.array([0.31, 0.58, 0.81]), 0.37)
    t, h = 0.13, 2e-5
    ut = (problem.exact(*coords, t + h) - problem.exact(*coords, t - h)) / (2 * h)
    divergence = 0.0
    for component in range(3):

        def flux(x, y, z, time):
            point = (x, y, z)
            return problem.conductivity(problem.exact(x, y, z, time)) * derivative(
                problem.exact, point, time, component, h
            )

        divergence += derivative(flux, coords, t, component, h)
    np.testing.assert_allclose(
        problem.source(*coords, t),
        (1 + gamma * problem.exact(*coords, t) ** 2) * ut
        - divergence
        + sigma * problem.exact(*coords, t) ** 3,
        atol=4e-6,
        rtol=2e-6,
    )


@pytest.mark.parametrize("kind", ["sine", "polynomial"])
@pytest.mark.parametrize("alpha", [0.0, 0.5])
@pytest.mark.parametrize("face", FACES)
def test_all_face_normals_and_flux_signs(kind, alpha, face):
    problem = make_heat_problem3d(
        kind, n=8, alpha=alpha, kinds=dict.fromkeys(FACES, 3), beta=1.7
    )
    component = "xyz".index(face[0])
    coords = [0.23, 0.41, 0.67]
    coords[component] = float(face[1])
    t = 0.11
    sign = -1 if face[1] == "0" else 1
    normal = sign * derivative(problem.exact, tuple(coords), t, component, 1e-5)
    u = problem.exact(*coords, t)
    r = problem.faces[face].value(*coords, t)
    outward_flux = -problem.conductivity(u) * normal
    np.testing.assert_allclose(outward_flux, 1.7 * u - r, atol=1e-8, rtol=1e-8)


def test_array_axes_initial_field_and_mixed_faces():
    kinds = dict.fromkeys(FACES, 1)
    kinds["y0"] = 3
    problem = make_heat_problem3d("polynomial", n=8, kinds=kinds)
    field = problem.initial_values()
    z, y, x = 2, 3, 5
    np.testing.assert_allclose(field[z, y, x], problem.exact(x / 8, y / 8, z / 8, 0))
    assert field.shape == (9, 9, 9)
    assert problem.face_values("x1", 0.0).shape == (9, 9)
    np.testing.assert_allclose(problem.face_values("x1", 0.0), field[:, :, -1])
    np.testing.assert_allclose(problem.face_values("z0", 0.0), field[0, :, :])
    for t in (0, 0.01, 0.02):
        assert np.all(problem.conductivity(problem.evaluate(problem.exact, t)) >= 1)
    assert NEIGHBOR_OFFSETS_ZYX[0] == (1, 0, 0)
    assert NEIGHBOR_OFFSETS_ZYX[4] == (0, 0, 1)


@pytest.mark.parametrize("kind", ["sine", "polynomial"])
def test_linear_limit_and_initial_boundary_compatibility(kind):
    problem = make_heat_problem3d(kind, alpha=0.0)
    np.testing.assert_array_equal(problem.conductivity(problem.initial_values()), 1)
    x, y, z = problem.coordinates()
    actual = problem.evaluate(problem.source, 0.1)
    if kind == "polynomial":
        np.testing.assert_allclose(actual, -0.5)
    else:
        expected = (
            (3 * np.pi**2 - 1)
            * np.exp(-0.1)
            * np.sin(np.pi * x)
            * np.sin(np.pi * y)
            * np.sin(np.pi * z)
        )
        np.testing.assert_allclose(actual, expected)
    with pytest.raises(ValueError, match="совместимо"):
        replace(problem, initial=lambda x, y, z, t: 99)


@pytest.mark.parametrize("kind", ["sine", "polynomial"])
def test_manufactured_source_converges_under_interior_conservative_stencil(kind):
    errors = []
    for n in (16, 32):
        problem = make_heat_problem3d(kind, n=n)
        t = 0.12
        values = problem.evaluate(problem.exact, t)
        k = problem.conductivity(values)
        divergence = np.zeros((n - 1,) * 3)
        for axis in range(3):
            lower = [slice(None)] * 3
            upper = [slice(None)] * 3
            lower[axis] = slice(None, -1)
            upper[axis] = slice(1, None)
            flux = (
                (k[tuple(lower)] + k[tuple(upper)])
                / 2
                * np.diff(values, axis=axis)
                / problem.h
            )
            term = np.diff(flux, axis=axis) / problem.h
            interior = [slice(1, -1)] * 3
            interior[axis] = slice(None)
            divergence += term[tuple(interior)]
        ut = (
            problem.evaluate(problem.exact, t + 1e-5)
            - problem.evaluate(problem.exact, t - 1e-5)
        ) / (2e-5)
        errors.append(
            np.max(
                np.abs(
                    (ut - problem.evaluate(problem.source, t))[1:-1, 1:-1, 1:-1]
                    - divergence
                )
            )
        )
    assert 3.5 < errors[0] / errors[1] < 4.5


@pytest.mark.parametrize(
    "kwargs",
    [
        {"alpha": -1},
        {"alpha": np.nan},
        {"n": True},
        {"n": 0},
        {"kind": "unknown"},
        {"kinds": {"x0": 1}},
    ],
)
def test_reject_invalid_model(kwargs):
    with pytest.raises(ValueError):
        make_heat_problem3d(**kwargs)


@pytest.mark.parametrize(
    "kind,beta", [(True, 0), (2, 0), (1, 1), (3, 0), (3, -1), (3, np.inf)]
)
def test_reject_unsupported_face(kind, beta):
    with pytest.raises(ValueError):
        HeatFace3D(kind, lambda x, y, z, t: 0, beta)


def test_time_and_finite_values():
    problem = make_heat_problem3d()
    with pytest.raises(ValueError):
        problem.evaluate(problem.source, -0.1)
    with pytest.raises(ValueError):
        problem.evaluate(lambda x, y, z, t: np.inf, 0.1)
    with pytest.raises(ValueError):
        problem.conductivity(np.inf)
    with pytest.raises(TypeError):
        problem.faces["x0"] = HeatFace3D(1, problem.exact)
    times, steps = time_grid(0.03, 0.1)
    np.testing.assert_allclose(steps, [0.03, 0.03, 0.03, 0.01])
    assert times[-1] == 0.1
    assert len(time_grid(0.03, 0)[1]) == 0


def test_factory_rejects_boolean_boundary_kind():
    kinds = dict.fromkeys(FACES, 1)
    kinds["x0"] = True
    with pytest.raises(ValueError):
        make_heat_problem3d(kinds=kinds)


@pytest.mark.parametrize("alpha,expected", [(0.0, 1.0), (1e-300, 1e100)])
def test_conductivity_avoids_intermediate_overflow(alpha, expected):
    problem = make_heat_problem3d(alpha=alpha)
    np.testing.assert_allclose(problem.conductivity([1e200, -1e200]), expected)
    with pytest.raises(ValueError, match="конечной"):
        problem.conductivity(np.inf)
