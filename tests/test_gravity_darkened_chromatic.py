"""Tests for the chromatic mode of GravityDarkenedStar, written from its spec.

In chromatic mode (``t_pole`` in kelvin) each surface triangle has
``T = t_pole * Teff_ratio / Teff_ratio(pole)``, intensity ``B_λ(T)`` and
weight ``projected area * B_λ(T)``, evaluated at every sample's own
wavelength. See "Surface-resolved chromatic components" in
``design/chromatic_sources.md``.
"""

import equinox as eqx
import jax
import jax.numpy as np
import matplotlib
import numpy as onp
import pytest

matplotlib.use("Agg")

from virgil import (  # noqa: E402
    BlackBody,
    GravityDarkenedStar,
    PointSource,
    PowerLaw,
    System,
    _elr,
)
from virgil._geometry import image_coordinates  # noqa: E402
from virgil.coverage import vlti_oidata  # noqa: E402
from virgil.likelihood import model_loglike  # noqa: E402
from tests._shared import float_leaves, planck  # noqa: E402

# Independent constants (not imported from virgil).
_MAS2RAD = onp.pi / 180.0 / 3600.0 / 1000.0

U = onp.array([20.0, -35.0, 50.0, 8.0, -60.0, 70.0])
V = onp.array([15.0, 40.0, -25.0, 62.0, 10.0, -45.0])
WAVES = onp.array([0.6e-6, 1.1e-6, 1.6e-6, 2.2e-6, 3.0e-6, 1.65e-6])


@eqx.filter_jit
def _model(model, u, v, wavel):
    return model.model(u, v, wavel)


def _hand_surface(omega, diam_eq, inc, pa, t_pole, n_lat=32):
    """Triangle x, y (mas), projected area and temperature, in float64."""
    x, y, _, teff, mesh = _elr.surface(
        omega,
        diam_eq / 2.0,
        onp.radians(90.0 - inc),
        onp.radians(pa),
        n_lat,
        return_mesh=True,
    )
    cosine = onp.asarray(mesh[2], dtype=float)
    area = onp.heaviside(cosine, 0.0) * cosine
    teff = onp.asarray(teff, dtype=float)
    pole = float(_elr.solve_ELR(omega, 0.0)[1])
    return (
        onp.asarray(x, float),
        onp.asarray(y, float),
        area,
        t_pole * teff / pole,
        teff,
    )


def _hand_vis(x, y, weights, u, v, wavel):
    """Hand DFT; weights has shape (n_samples, n_triangles)."""
    u, v, wavel = (onp.asarray(a, float) for a in (u, v, wavel))
    arg = (
        -2j
        * onp.pi
        * (u[:, None] * x[None] + v[:, None] * y[None])
        * _MAS2RAD
        / wavel[:, None]
    )
    return (weights * onp.exp(arg)).sum(1) / weights.sum(1)


def _photocentre_y(image, fov_mas):
    _, y = image_coordinates(image.shape[0], fov_mas)
    image = onp.asarray(image, dtype=float)
    image = image / image.sum()
    return float((image * onp.asarray(y)).sum())


# --- 1. grey unchanged ---------------------------------------------------


@pytest.mark.parametrize("omega", [0.0, 0.7])
def test_grey_mode_unchanged(omega):
    a = GravityDarkenedStar(1.0, omega, 50.0, 20.0, t_pole=None)
    b = GravityDarkenedStar(1.0, omega, 50.0, 20.0)
    for wavel in (1.0e-6, 2.2e-6):
        assert np.array_equal(_model(a, U, V, wavel), _model(b, U, V, wavel))
    assert float(a._weight(2.0e-6)) == float(a._weight(None)) == 1.0


# --- 2. omega = 0: uniform temperature -----------------------------------


@pytest.mark.parametrize("wavel", [0.6e-6, 1.6e-6, 2.2e-6])
def test_uniform_temperature_matches_grey_and_blackbody(wavel):
    with jax.enable_x64():
        grey = GravityDarkenedStar(1.5, 0.0, 70.0, 30.0)
        chrom = GravityDarkenedStar(
            1.5, 0.0, 70.0, 30.0, t_pole=7000.0, wavel0=1.3e-6
        )
        onp.testing.assert_allclose(
            _model(chrom, U, V, wavel),
            _model(grey, U, V, wavel),
            rtol=1e-10,
            atol=1e-12,
        )
        bb = BlackBody(1.0, 7000.0, 1.3e-6)
        onp.testing.assert_allclose(chrom._weight(wavel), bb(wavel), rtol=1e-5)
        onp.testing.assert_allclose(chrom._weight(1.3e-6), 1.0, rtol=1e-8)


def test_weight_none_is_flux_and_scales_with_flux():
    star = GravityDarkenedStar(1.0, 0.5, 60.0, t_pole=8000.0, flux=2.5)
    assert float(star._weight(None)) == pytest.approx(2.5)
    assert float(star._weight(star.wavel0)) == pytest.approx(2.5, rel=1e-5)


# --- 3. independent recomputation ----------------------------------------


@pytest.mark.parametrize(
    "omega, inc, pa", [(0.0, 90.0, 0.0), (0.6, 60.0, 25.0), (0.9, 45.0, 0.0)]
)
def test_matches_independent_recomputation(omega, inc, pa):
    t_pole, wavel0, diam, flux = 9000.0, 1.65e-6, 1.2, 1.7
    with jax.enable_x64():
        star = GravityDarkenedStar(
            diam, omega, inc, pa, flux=flux, t_pole=t_pole, wavel0=wavel0
        )
        x, y, area, temp, _ = _hand_surface(omega, diam, inc, pa, t_pole)
        weights = area[None] * planck(WAVES[:, None], temp[None])
        expected = _hand_vis(x, y, weights, U, V, WAVES)
        got = onp.asarray(_model(star, U, V, WAVES))
        onp.testing.assert_allclose(got, expected, rtol=1e-8, atol=1e-10)

        sed = lambda w: (area * planck(w, temp)).sum()  # noqa: E731
        expected_w = flux * onp.array([sed(w) for w in WAVES]) / sed(wavel0)
        onp.testing.assert_allclose(
            onp.asarray(star._weight(WAVES)), expected_w, rtol=1e-8
        )


# --- 4. physics ----------------------------------------------------------


def test_hot_pole_photocentre_moves_north_at_short_wavelength():
    fov = 3.0

    def centre(wavel0):
        star = GravityDarkenedStar(
            2.0, 0.9, 45.0, 0.0, t_pole=9000.0, wavel0=wavel0
        )
        return _photocentre_y(star.render(192, fov), fov)

    short, long = centre(0.5e-6), centre(2.2e-6)
    assert short > long > 0.0
    assert short - long > 1e-3 * 2.0


def test_phase_photocentre_shifts_with_wavelength():
    star = GravityDarkenedStar(2.0, 0.9, 45.0, 0.0, t_pole=9000.0)
    u, v = onp.array([0.0]), onp.array([2.0])

    def dec(wavel):
        phase = onp.angle(onp.asarray(_model(star, u, v, wavel)))[0]
        return -phase * wavel / (2 * onp.pi * _MAS2RAD * v[0])

    assert dec(0.5e-6) > dec(2.2e-6) > 0.0


def test_rayleigh_jeans_limit_weights_are_area_times_temperature():
    omega, inc, pa, diam = 0.9, 45.0, 0.0, 1.0
    with jax.enable_x64():
        star = GravityDarkenedStar(
            diam, omega, inc, pa, t_pole=1.0e9, wavel0=2.0e-6
        )
        x, y, area, _, teff = _hand_surface(omega, diam, inc, pa, 1.0e9)
        weights = onp.tile(area * teff, (len(WAVES), 1))
        expected = _hand_vis(x, y, weights, U, V, WAVES)
        onp.testing.assert_allclose(
            onp.asarray(_model(star, U, V, WAVES)),
            expected,
            rtol=1e-4,
            atol=1e-5,
        )


# --- 5. shapes and data --------------------------------------------------


def test_wavelength_shapes_agree():
    star = GravityDarkenedStar(1.0, 0.8, 55.0, 10.0, t_pole=8000.0)
    wavel = 1.7e-6
    ref = onp.asarray(_model(star, U, V, wavel))
    assert ref.shape == U.shape
    onp.testing.assert_allclose(
        _model(star, U, V, np.asarray([wavel])), ref, rtol=1e-5, atol=1e-6
    )
    onp.testing.assert_allclose(
        _model(star, U, V, np.full(U.shape, wavel)), ref, rtol=1e-5, atol=1e-6
    )
    # per-sample wavelengths: each sample equals the scalar-wavelength value
    mixed = onp.asarray(_model(star, U, V, WAVES))
    for i, w in enumerate(WAVES):
        single = onp.asarray(
            _model(star, U[i : i + 1], V[i : i + 1], float(w))
        )
        onp.testing.assert_allclose(mixed[i], single[0], rtol=1e-4, atol=1e-5)


def test_two_dimensional_samples():
    star = GravityDarkenedStar(1.0, 0.8, 55.0, 10.0, t_pole=8000.0)
    u2, v2, w2 = U.reshape(2, 3), V.reshape(2, 3), WAVES.reshape(2, 3)
    got = onp.asarray(_model(star, u2, v2, w2))
    assert got.shape == (2, 3)
    onp.testing.assert_allclose(
        got.ravel(),
        onp.asarray(_model(star, U, V, WAVES)),
        rtol=1e-5,
        atol=1e-6,
    )
    # a scalar wavelength on 2D u, v
    got = onp.asarray(_model(star, u2, v2, 1.7e-6))
    assert got.shape == (2, 3)
    onp.testing.assert_allclose(
        got.ravel(),
        onp.asarray(_model(star, U, V, 1.7e-6)),
        rtol=1e-5,
        atol=1e-6,
    )


def _data():
    return vlti_oidata(
        hour_angles_h=(-1.0, 1.0),
        wavelengths_m=onp.linspace(2.0e-6, 2.4e-6, 4),
    )


def test_multichannel_oidata_model_and_loglike_finite():
    data = _data()
    star = GravityDarkenedStar(1.0, 0.8, 55.0, 10.0, t_pole=8000.0)
    cvis = onp.asarray(data.model(star))
    assert cvis.size > 0 and onp.all(onp.isfinite(cvis))
    assert onp.isfinite(float(model_loglike(star, data)))


# --- 6. System -----------------------------------------------------------


def test_system_with_blackbody_companion_is_hand_mix():
    star = GravityDarkenedStar(
        1.0, 0.8, 55.0, 10.0, t_pole=8000.0, wavel0=1.65e-6
    )
    comp_flux = BlackBody(0.05, 3000.0)
    comp = PointSource(flux=comp_flux, dra=10.0)
    system = System(star=star, comp=comp)
    got = onp.asarray(_model(system, U, V, WAVES))
    w_s = onp.asarray(star._weight(WAVES))
    w_c = onp.asarray(comp_flux(WAVES))
    v_s = onp.asarray(_model(star, U, V, WAVES))
    v_c = onp.asarray(_model(comp, U, V, WAVES))
    expected = (w_s * v_s + w_c * v_c) / (w_s + w_c)
    onp.testing.assert_allclose(got, expected, rtol=1e-4, atol=1e-5)
    # at wavel0 the stellar weight is 1 and the companion's is its ratio
    onp.testing.assert_allclose(star._weight(1.65e-6), 1.0, rtol=1e-5)


# --- 7. gradients --------------------------------------------------------


def _check_grads(omega):
    data = _data()
    truth = GravityDarkenedStar(1.0, 0.5, 60.0, 20.0, t_pole=8000.0)
    data = data.with_model(truth, key=jax.random.PRNGKey(0))
    model = GravityDarkenedStar(1.0, omega, 55.0, 15.0, t_pole=8500.0)
    grads = eqx.filter_jit(eqx.filter_grad(model_loglike))(model, data)
    leaves = float_leaves(grads)
    assert len(leaves) >= 8
    for leaf in leaves:
        assert onp.all(onp.isfinite(onp.asarray(leaf)))
    assert onp.isfinite(onp.asarray(grads.t_pole))
    assert onp.isfinite(onp.asarray(grads.wavel0))


@pytest.mark.parametrize("omega", [0.0, 0.9])
def test_gradients_finite_float32(omega):
    _check_grads(omega)


@pytest.mark.parametrize("omega", [0.0, 0.9])
def test_gradients_finite_float64(omega):
    with jax.enable_x64():
        _check_grads(omega)


def test_system_gradients_finite_with_chromatic_star():
    data = _data()
    scene = System(
        star=GravityDarkenedStar(1.0, 0.6, 50.0, 0.0, t_pole=8000.0),
        comp=PointSource(flux=BlackBody(0.05, 3000.0), dra=10.0),
    )
    data = data.with_model(scene, key=jax.random.PRNGKey(1))
    grads = eqx.filter_jit(eqx.filter_grad(model_loglike))(scene, data)
    for leaf in float_leaves(grads):
        assert onp.all(onp.isfinite(onp.asarray(leaf)))


# --- 8. validation -------------------------------------------------------


@pytest.mark.parametrize("t_pole", [0.0, -100.0])
def test_non_positive_t_pole_raises(t_pole):
    with pytest.raises(ValueError):
        GravityDarkenedStar(1.0, t_pole=t_pole)


def test_spectrum_flux_with_t_pole_raises():
    with pytest.raises(ValueError):
        GravityDarkenedStar(
            1.0, t_pole=8000.0, flux=PowerLaw(1.0, -1.0, 1.65e-6)
        )
    with pytest.raises(ValueError):
        GravityDarkenedStar(1.0, t_pole=8000.0, flux=BlackBody(1.0, 5000.0))


def test_is_physical_requires_positive_t_pole():
    star = GravityDarkenedStar(1.0, 0.3, 60.0, t_pole=8000.0)
    assert bool(star.is_physical())
    assert not bool(star.set(["t_pole"], [-5.0]).is_physical())
    assert not bool(star.set(["t_pole"], [0.0]).is_physical())
    assert bool(star.set(["t_pole"], [6000.0]).is_physical())
