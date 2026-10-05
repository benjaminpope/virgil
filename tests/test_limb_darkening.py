"""Tests for the limb-darkened disks and ``cvis_limb_darkened_disk``.

The visibilities are checked against direct numerical Hankel transforms of
each brightness profile (scipy), independent of the Bessel-function route,
and, when harmonix is installed, against harmonix for unspotted stars.
"""

import jax
import jax.numpy as np
import numpy as onp
import pytest
from scipy.integrate import quad
from scipy.special import j0

from virgil.models import (
    LimbDarkenedDisk,
    PointSource,
    QuadraticLimbDarkenedDisk,
    SquareRootLimbDarkenedDisk,
    System,
    UniformDisk,
    cvis_limb_darkened_disk,
)

# Independent reference constant (not imported from virgil).
_MAS2RAD_REF = onp.pi / 180.0 / 3600.0 / 1000.0

DIAM = 2.0  # mas
WAVEL = 1.6e-6
# baselines (m) out to the third lobe of a 2 mas star in the H band
BASELINES = onp.linspace(0.0, 600.0, 25)
ANGLES = onp.linspace(0.0, 2.5, 25)
U = BASELINES * onp.cos(ANGLES)
V = BASELINES * onp.sin(ANGLES)


def _hankel(profile):
    """Normalized visibility of brightness ``profile(mu)`` by quadrature."""
    x = onp.pi * DIAM * _MAS2RAD_REF * BASELINES / WAVEL
    vis = [
        quad(
            lambda r, xi=xi: profile(onp.sqrt(1 - r * r)) * j0(xi * r) * r,
            0.0,
            1.0,
            limit=400,
            epsabs=1e-13,
        )[0]
        for xi in x
    ]
    return onp.asarray(vis) / vis[0]


def _visibility(model):
    return onp.asarray(model.model(np.asarray(U), np.asarray(V), WAVEL))


def _quadratic(u1, u2):
    return lambda mu: 1 - u1 * (1 - mu) - u2 * (1 - mu) ** 2


def _square_root(c, d):
    return lambda mu: 1 - c * (1 - mu) - d * (1 - onp.sqrt(mu))


# Factories, so that each model is built inside the test's precision.
@pytest.mark.parametrize(
    "make, profile",
    [
        (lambda: LimbDarkenedDisk(DIAM), lambda mu: onp.ones_like(mu)),
        (
            lambda: LimbDarkenedDisk(DIAM, u=[0.6]),
            lambda mu: 1 - 0.6 * (1 - mu),
        ),
        (lambda: LimbDarkenedDisk(DIAM, u=[0.4, 0.25]), _quadratic(0.4, 0.25)),
        (
            lambda: LimbDarkenedDisk(DIAM, u=[0.3, 0.2, 0.1]),
            lambda mu: 1
            - 0.3 * (1 - mu)
            - 0.2 * (1 - mu) ** 2
            - 0.1 * (1 - mu) ** 3,
        ),
        # u1 = 2 sqrt(q1) q2, u2 = sqrt(q1) (1 - 2 q2)
        (
            lambda: QuadraticLimbDarkenedDisk(DIAM, q1=0.36, q2=0.25),
            _quadratic(0.3, 0.3),
        ),
        # c = sqrt(q1) (1 - 2 q2), d = 2 sqrt(q1) q2
        (
            lambda: SquareRootLimbDarkenedDisk(DIAM, q1=0.64, q2=0.375),
            _square_root(0.2, 0.6),
        ),
    ],
    ids=["uniform", "linear", "quadratic", "cubic", "kipping-quad", "sqrt"],
)
def test_matches_numerical_hankel_transform(make, profile):
    with jax.enable_x64(True):
        vis = _visibility(make())
    assert onp.allclose(vis.imag, 0.0, atol=1e-14)
    assert onp.allclose(vis.real, _hankel(profile), rtol=0, atol=1e-10)


def test_uniform_limit_is_uniform_disk():
    with jax.enable_x64(True):
        for model in (
            LimbDarkenedDisk(DIAM),
            QuadraticLimbDarkenedDisk(DIAM, q1=0.0, q2=0.5),
            SquareRootLimbDarkenedDisk(DIAM, q1=0.0, q2=0.5),
        ):
            assert onp.allclose(
                _visibility(model), _visibility(UniformDisk(DIAM)), atol=1e-14
            )


@pytest.mark.slow
def test_matches_harmonix_for_unspotted_stars():
    pytest.importorskip("harmonix")
    from harmonix.harmonix import Harmonix
    from jaxoplanet.starry import Surface

    from virgil.models import HarmonixModel

    with jax.enable_x64(True):
        for u in ([0.6], [0.4, 0.25], [0.3, 0.2, 0.1]):
            star = HarmonixModel(
                Harmonix(Surface(u=u), DIAM / 2), observation_time=0.0
            )
            expected = _visibility(star)
            assert onp.allclose(
                _visibility(LimbDarkenedDisk(DIAM, u=u)), expected, atol=1e-12
            )


def test_cvis_limb_darkened_disk_offsets_and_unit_flux():
    coeffs = np.asarray([0.2, 0.5, 0.3])
    centred = cvis_limb_darkened_disk(U, V, DIAM, coeffs, (0.0, 1.0, 0.5))
    shifted = cvis_limb_darkened_disk(
        U, V, DIAM, coeffs, (0.0, 1.0, 0.5), dra=1.0, ddec=-2.0
    )
    assert onp.isclose(centred[0], 1.0)
    assert onp.allclose(onp.abs(shifted), onp.abs(centred), atol=1e-6)
    assert not onp.allclose(shifted, centred)


def test_float32_matches_float64():
    # explicit, since importing jaxoplanet (harmonix test) turns on x64
    with jax.enable_x64(False):
        vis32 = _visibility(SquareRootLimbDarkenedDisk(DIAM, q1=0.5, q2=0.4))
    with jax.enable_x64(True):
        vis64 = _visibility(SquareRootLimbDarkenedDisk(DIAM, q1=0.5, q2=0.4))
    assert vis32.dtype == onp.complex64
    assert onp.allclose(vis32, vis64, atol=2e-6)


@pytest.mark.parametrize(
    "model",
    [
        LimbDarkenedDisk(DIAM, u=[0.4, 0.25]),
        QuadraticLimbDarkenedDisk(DIAM, q1=0.4, q2=0.3),
        SquareRootLimbDarkenedDisk(DIAM, q1=0.4, q2=0.3),
    ],
    ids=["polynomial", "quadratic", "sqrt"],
)
def test_gradients_are_finite_including_zero_baseline(model):
    params = (
        ["diam", "u"]
        if isinstance(model, LimbDarkenedDisk)
        else [
            "diam",
            "q1",
            "q2",
        ]
    )

    def loss(values):
        m = model.set(params, values)
        return np.sum(
            np.abs(m.model(np.asarray(U), np.asarray(V), WAVEL)) ** 2
        )

    grads = jax.grad(loss)([model.get(p) for p in params])
    for g in grads:
        assert onp.all(onp.isfinite(onp.asarray(g)))
    # diameter matters; at zero baseline alone the visibility is exactly 1
    assert onp.abs(grads[0]) > 0
    at_zero = jax.grad(
        lambda d: np.abs(model.set("diam", d).model(0.0, 0.0, WAVEL))
    )(model.diam)
    assert onp.isfinite(at_zero) and at_zero == 0.0


def test_kipping_quadratic_round_trip_and_definitions():
    star = QuadraticLimbDarkenedDisk.from_u(DIAM, u1=0.4, u2=0.25)
    assert onp.isclose(star.u1, 0.4) and onp.isclose(star.u2, 0.25)
    # Kipping (2013) eqs. 17-18
    assert onp.isclose(star.q1, 0.65**2)
    assert onp.isclose(star.q2, 0.4 / (2 * 0.65))
    zero = QuadraticLimbDarkenedDisk.from_u(DIAM, u1=0.0, u2=0.0)
    assert zero.q1 == 0.0 and zero.q2 == 0.0


def test_kipping_square_root_round_trip_and_definitions():
    star = SquareRootLimbDarkenedDisk.from_cd(DIAM, c=0.2, d=0.6)
    assert onp.isclose(star.c, 0.2) and onp.isclose(star.d, 0.6)
    # Kipping (2013) eqs. 23-24
    assert onp.isclose(star.q1, 0.8**2)
    assert onp.isclose(star.q2, 0.6 / (2 * 0.8))


@pytest.mark.parametrize(
    "cls, profile",
    [
        (QuadraticLimbDarkenedDisk, lambda s: _quadratic(s.u1, s.u2)),
        (SquareRootLimbDarkenedDisk, lambda s: _square_root(s.c, s.d)),
    ],
    ids=["quadratic", "sqrt"],
)
def test_unit_square_gives_exactly_the_physical_profiles(cls, profile):
    # Inside the unit square every profile is positive and decreases towards
    # the limb (Kipping 2013); just outside it, one of those fails.
    mu = onp.linspace(0.0, 1.0, 201)
    # On the square's corners the limb intensity is exactly 0, which float32
    # rounds to about -6e-8; the violations just outside are ~1e-3.
    tol = 1e-6

    def physical(q1, q2):
        intensity = profile(cls(DIAM, q1=q1, q2=q2))(mu)
        return onp.all(intensity >= -tol) and onp.all(
            onp.diff(intensity) >= -tol
        )

    grid = onp.linspace(0.0, 1.0, 11)
    assert all(physical(q1, q2) for q1 in grid for q2 in grid)
    assert not physical(1.0, -0.05)
    assert not physical(1.0, 1.05)
    assert not physical(1.2, 0.5)


def test_powers_outside_the_supported_range_are_refused():
    for powers in [(0.0, 23.0), (-2.0,)]:
        with pytest.raises(ValueError):
            cvis_limb_darkened_disk(U, V, DIAM, np.ones(len(powers)), powers)
    # order 22 is the highest polynomial law
    assert onp.isfinite(
        _visibility(LimbDarkenedDisk(DIAM, u=[0.0] * 22))
    ).all()


def test_is_physical():
    assert QuadraticLimbDarkenedDisk(DIAM, q1=0.3, q2=0.9).is_physical()
    assert not QuadraticLimbDarkenedDisk(DIAM, q1=1.3, q2=0.2).is_physical()
    assert not SquareRootLimbDarkenedDisk(DIAM, q1=0.3, q2=-0.1).is_physical()
    assert not LimbDarkenedDisk(-1.0, u=[0.5]).is_physical()
    assert LimbDarkenedDisk(DIAM, u=[0.5]).is_physical()
    # negative at the limb, and zero flux
    assert not LimbDarkenedDisk(DIAM, u=[2.0]).is_physical()
    assert not LimbDarkenedDisk(DIAM, u=[3.0]).is_physical()
    # limb brightening is allowed while the profile stays positive
    assert LimbDarkenedDisk(DIAM, u=[-0.3]).is_physical()
    jitted = jax.jit(lambda m: m.is_physical())
    assert not jitted(LimbDarkenedDisk(DIAM, u=[2.0]))


def test_render_darkens_the_limb():
    image = onp.asarray(
        QuadraticLimbDarkenedDisk(DIAM, q1=0.8, q2=0.4).render(
            npix=65, fov_mas=2.6
        )
    )
    uniform = onp.asarray(UniformDisk(DIAM).render(npix=65, fov_mas=2.6))
    assert onp.isclose(image.sum(), 1.0)
    # same footprint as the uniform disk, brighter centre
    assert onp.array_equal(image > 0, uniform > 0)
    centre, edge = image[32, 32], image[32, 32 + 23]
    assert centre > image.max() * 0.999 and edge < 0.7 * centre


def test_in_a_system_with_a_companion():
    star = SquareRootLimbDarkenedDisk(DIAM, q1=0.5, q2=0.4)
    system = System(star=star, comp=PointSource(flux=0.05, dra=10.0, ddec=5.0))
    vis = onp.asarray(system.model(np.asarray(U), np.asarray(V), WAVEL))
    assert onp.isclose(vis[0], 1.0)
    assert onp.all(onp.isfinite(vis))
