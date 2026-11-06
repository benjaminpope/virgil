"""LinearImage in the radio χ²: Σ pixels is the flux in Jy (fit_flux=False)."""

import jax
import jax.numpy as np
import numpy as onp
import pytest

from virgil import (
    CalibrationNuisance,
    ComplexVisData,
    Image,
    LinearImage,
    PointSource,
    System,
)
from virgil.imaging import MaxEntropy
from virgil.models import Rotated
from scipy.constants import c as C_LIGHT

from virgil._utils import mas2rad

PIXEL = 2.0  # mas
NPIX = 8
N = 200


@pytest.fixture(autouse=True)
def float64():
    with jax.enable_x64(True):
        yield


def _data(zero=0, seed=0):
    """Random points; ``zero`` extra points at u = v = 0."""
    rng = onp.random.default_rng(seed)
    q = 0.4 / (2.0 * mas2rad * PIXEL) * onp.sqrt(rng.uniform(0, 1, N))
    phi = rng.uniform(0, 2 * onp.pi, N)
    u = onp.concatenate([q * onp.cos(phi), onp.zeros(zero)])
    v = onp.concatenate([q * onp.sin(phi), onp.zeros(zero)])
    n = len(u)
    vis = rng.normal(size=n) + 1j * rng.normal(size=n)
    return ComplexVisData.from_arrays(
        u, v, vis, onp.full(n, 50.0), onp.full(n, 230e9)
    )


def _nuis(data, **kw):
    return CalibrationNuisance(
        data,
        gains=False,
        offsets=False,
        alpha=False,
        **{"sigmas": False, **kw},
    )


def _scene(parametrisation):
    def scene(values):
        return LinearImage(
            values["x"],
            PIXEL,
            ft="dft",
            parametrisation=parametrisation,
            reference_flux=1.0,
        )

    return scene


def _x(parametrisation, seed=1):
    flux = onp.random.default_rng(seed).uniform(0.01, 0.1, (NPIX, NPIX))
    x = onp.sqrt(flux) if parametrisation == "sqrt" else flux
    return np.asarray(x)


def _chi2(data, scene, nuis, values):
    return float(data.chi2_by_ddid(scene, values, nuis)[0][0])


@pytest.mark.parametrize("parametrisation", ["sqrt", "linear"])
def test_halving_the_pixels_changes_chi2(parametrisation):
    data, scene = _data(), _scene(parametrisation)
    nuis = _nuis(data, fit_flux=False)
    assert "cal.flux" not in nuis.sites
    x = _x(parametrisation)
    half = x / (onp.sqrt(2.0) if parametrisation == "sqrt" else 2.0)
    full = _chi2(data, scene, nuis, {"x": x})
    assert abs(_chi2(data, scene, nuis, {"x": half}) - full) > 1e-3 * full


@pytest.mark.parametrize("parametrisation", ["sqrt", "linear"])
@pytest.mark.parametrize("sigmas", [False, "profile"])
def test_toeplitz_matches_dense(parametrisation, sigmas):
    data, scene = _data(), _scene(parametrisation)
    nuis = _nuis(data, fit_flux=False, sigmas=sigmas)
    template = LinearImage(
        np.ones((NPIX, NPIX)), PIXEL, parametrisation=parametrisation
    )
    tdata = data.toeplitz(template, nuis)
    term = tdata.term(scene, nuis)
    direct = data.term(scene, nuis, chi2="direct")

    def toeplitz(x):
        chi2, log_norm = term.chi2_and_log_norm({"x": x})
        return 0.5 * chi2 + log_norm

    def dense(x):
        r, log_norm = direct.residuals_and_log_norm({"x": x})
        return 0.5 * np.sum(r**2) + log_norm

    x1, x2 = _x(parametrisation, 1), _x(parametrisation, 2)
    # the Toeplitz loss omits a constant: compare differences and gradients
    assert float(toeplitz(x1) - toeplitz(x2)) == pytest.approx(
        float(dense(x1) - dense(x2)), rel=1e-9
    )
    g, g_ref = jax.grad(toeplitz)(x1), jax.grad(dense)(x1)
    assert float(np.max(np.abs(g - g_ref)) / np.max(np.abs(g_ref))) < 1e-8
    if sigmas is False:
        chi2 = tdata.chi2_by_ddid(scene, {"x": x1}, nuis)[0]
        assert chi2[0] == pytest.approx(
            _chi2(data, scene, nuis, {"x": x1}), rel=1e-9
        )


@pytest.mark.parametrize("parametrisation", ["sqrt", "linear"])
def test_zero_spacing_visibility_is_the_pixel_sum(parametrisation):
    data, scene = _data(zero=3), _scene(parametrisation)
    nuis = _nuis(data, fit_flux=False)
    x = _x(parametrisation)
    image = scene({"x": x})
    model = onp.asarray(data.model(image, {"x": x}, nuis))
    assert model[-3:] == pytest.approx(float(np.sum(image.fluxes)), rel=1e-12)
    assert onp.allclose(model[-3:].imag, 0.0)


def test_images_keep_cal_flux():
    data = _data()
    nuis = _nuis(data)
    assert "cal.flux" in nuis.sites
    eta = np.asarray(onp.random.default_rng(3).normal(size=(NPIX, NPIX)))
    image = Image(eta, PIXEL, ft="dft")
    values = {**nuis.defaults(), "cal.flux": np.asarray(0.37)}
    lam = C_LIGHT / onp.asarray(data.freq)
    expected = 0.37 * onp.asarray(
        image.model(onp.asarray(data.u) * lam, onp.asarray(data.v) * lam, lam)
    )
    model = onp.asarray(data.model(image, values, nuis))
    assert onp.allclose(model, expected, rtol=1e-12, atol=0.0)


def test_flux_ownership_is_enforced():
    data = _data()
    x = _x("sqrt")
    lone = _scene("sqrt")({"x": x})
    with pytest.raises(TypeError, match="fit_flux=False"):
        data.model(lone, _nuis(data).defaults() | {"x": x}, _nuis(data))
    image = Image(np.zeros((NPIX, NPIX)), PIXEL, ft="dft")
    nuis = _nuis(data, fit_flux=False)
    with pytest.raises(ValueError, match="one LinearImage"):
        data.model(image, {}, nuis)
    with pytest.raises(ValueError, match="no flux site"):
        nuis.priors(flux=(1e-3, 1.0))


def test_maxentropy_gradient_through_the_radio_term_is_finite():
    data, scene = _data(), _scene("sqrt")
    nuis = _nuis(data, fit_flux=False)
    term = data.term(scene, nuis, chi2="direct")
    entropy = MaxEntropy(1.0)

    def loss(x):
        r, _ = term.residuals_and_log_norm({"x": x})
        return 0.5 * np.sum(r**2) + entropy.value(scene({"x": x}))

    g = jax.grad(loss)(_x("sqrt"))
    assert bool(np.all(np.isfinite(g)))
    assert float(np.max(np.abs(g))) > 0.0


def test_wrapped_scenes_equal_the_bare_image():
    data, scene = _data(), _scene("linear")
    nuis = _nuis(data, fit_flux=False)
    assert "cal.flux" not in nuis.defaults()
    values = {"x": _x("linear")}
    bare = _chi2(data, scene, nuis, values)
    wrapped = lambda v: System(env=scene(v))  # noqa: E731
    rotated = lambda v: Rotated(scene(v), 0.0)  # noqa: E731
    assert _chi2(data, wrapped, nuis, values) == pytest.approx(bare, rel=1e-12)
    assert _chi2(data, rotated, nuis, values) == pytest.approx(bare, rel=1e-12)
    template = LinearImage(np.ones((NPIX, NPIX)), PIXEL)
    tchi2 = data.toeplitz(template, nuis).chi2_by_ddid(wrapped, values, nuis)
    assert tchi2[0][0] == pytest.approx(bare, rel=1e-9)
    free = _nuis(data)
    with pytest.raises(TypeError, match="fit_flux=False"):
        data.model(rotated(values), {**free.defaults(), **values}, free)


def test_multi_part_and_toeplitz_refusals():
    data, x = _data(), _x("linear")
    scene = _scene("linear")({"x": x})
    mixed = System(star=PointSource(), env=scene)
    with pytest.raises(ValueError, match="one LinearImage"):
        data.model(mixed, {"x": x}, _nuis(data, fit_flux=False))
    template = LinearImage(np.ones((NPIX, NPIX)), PIXEL)
    nuis = _nuis(data)
    term = data.toeplitz(template, nuis).term(_scene("linear"), nuis)
    values = {**nuis.defaults(), "x": x}
    with pytest.raises(TypeError, match="fit_flux=False"):
        term.chi2_and_log_norm(values)


def test_toeplitz_curvature_is_that_of_the_pixels():
    data = _data()
    nuis = _nuis(data, fit_flux=False)
    template = LinearImage(np.ones((NPIX, NPIX)), PIXEL)
    term = data.toeplitz(template, nuis).term(_scene("linear"), nuis)
    x = _x("linear")
    image, _, g, _ = term.curvature_params({"x": x})
    assert onp.allclose(image, x) and onp.allclose(g, 1.0)


def test_zero_pixel_sum_raises():
    data = _data()
    nuis = _nuis(data, fit_flux=False)
    zeros = {"x": np.zeros((NPIX, NPIX))}
    with pytest.raises(Exception, match="positive"):
        data.model(_scene("linear")(zeros), zeros, nuis)


def test_mixtures_with_a_free_flux():
    data, x = _data(), _x("linear")
    free = _nuis(data)
    values = {**free.defaults(), "x": x}
    env = _scene("linear")({"x": x})
    ok = System(star=PointSource(), env=env)
    assert onp.all(onp.isfinite(onp.asarray(data.model(ok, values, free))))
    two = System(a=env, b=LinearImage(x, PIXEL, dra=3.0, reference_flux=1.0))
    with pytest.raises(TypeError, match="only of LinearImages"):
        data.model(two, values, free)


def test_zero_pixel_sum_raises_in_both_paths():
    data = _data()
    nuis = _nuis(data, fit_flux=False)
    template = LinearImage(np.ones((NPIX, NPIX)), PIXEL)
    term = data.toeplitz(template, nuis).term(_scene("linear"), nuis)
    zeros = {"x": np.zeros((NPIX, NPIX))}
    with pytest.raises(Exception, match="positive"):
        term.chi2_and_log_norm(zeros)
