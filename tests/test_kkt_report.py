"""kkt_report: convergence certificates of image fits (not a loss plateau)."""

import jax
import numpy as onp
import pytest

from virgil.coverage import ami_grid_record
from virgil.fitting import KKT_TOL, FitResult, fit, kkt_report
from virgil.imaging import TSV, MaxEntropy, image_priors
from virgil.models import Image, LinearImage, PointSource, System
from virgil.oidata import OIData

DATA = OIData(ami_grid_record(pitch_m=1.0))

pytestmark = pytest.mark.filterwarnings("ignore::RuntimeWarning")


@pytest.fixture(autouse=True)
def _x64():
    with jax.enable_x64(True):
        yield


def _disc(n, flux=1.0):
    yy, xx = onp.mgrid[:n, :n] - (n - 1) / 2
    disc = (xx**2 + yy**2 < (n / 8) ** 2).astype(float)
    return flux * disc / disc.sum()


def _scene(pixels, scale, parametrisation="linear"):
    env = LinearImage(
        pixels, scale, parametrisation=parametrisation, reference_flux=0.5
    )
    return System(star=PointSource(), env=env)


def _problem(n, scale, parametrisation="linear"):
    """A start scene, its priors and noisy data from a disc."""
    flat = onp.full((n, n), 0.5 / n**2)
    flat = onp.sqrt(flat) if parametrisation == "sqrt" else flat
    start = _scene(flat, scale, parametrisation)
    data = DATA.with_model(
        _scene(_disc(n, 0.5), scale),
        key=jax.random.PRNGKey(1),
        noise_scale=3.0,
    )
    return start, image_priors(start), data


def _polish(result, start, priors, data, regularizers, method, rounds=3):
    """Restart a fit: the stop tolerance is relative to the start gradient, so a
    restart from a converged point goes on to a tighter one."""
    for _ in range(rounds):
        result = fit(
            start,
            priors,
            data,
            regularizers,
            method=method,
            gtol=1e-4,
            init=result.values,
        )
    return result


def _report(
    n, scale, regularizers, parametrisation="linear", rounds=1, **options
):
    start, priors, data = _problem(n, scale, parametrisation)
    result = fit(start, priors, data, regularizers, method="lbfgsb", **options)
    result = _polish(
        result, start, priors, data, regularizers, "lbfgsb", rounds
    )
    return kkt_report(result, priors, data, regularizers)


TSV_ = [TSV(0.02, path="env", beam_mas=60.0)]


@pytest.mark.slow
def test_bounded_linear_fit_is_converged_with_active_bounds():
    report = _report(10, 24.0, TSV_, max_steps=5000)
    image = report["images"]["env"]
    assert report["converged"], report
    assert image["pg_rel"] < KKT_TOL
    assert (
        image["zero_violation"] == 0.0
    )  # bound-active zeros are not violations


def test_bounded_linear_fit_stopped_early_is_not_converged():
    report = _report(10, 24.0, TSV_, rounds=0, max_steps=5)
    assert not report["converged"]
    assert report["images"]["env"]["pg_rel"] > KKT_TOL


def test_zero_pixel_wanting_flux_is_a_violation():
    start, priors, data = _problem(10, 24.0)
    result = fit(start, priors, data, TSV_, method="lbfgsb", gtol=1e-6)
    result = _polish(result, start, priors, data, TSV_, "lbfgsb")
    pixels = onp.asarray(result.values["env.pixels"]).copy()
    pixels[_disc(10) > 0] = (
        0.0  # the lit pixels sit at a bound they want to leave
    )
    result = FitResult(start, {"env.pixels": pixels}, {})
    report = kkt_report(result, priors, data, TSV_)
    assert report["images"]["env"]["zero_violation"] > 0.0
    assert not report["converged"]


@pytest.mark.parametrize("steps, converged", [(5000, True), (5, False)])
def test_sqrt_maxentropy(steps, converged):
    reg = [MaxEntropy(0.1, path="env")]
    report = _report(
        32, 7.5, reg, "sqrt", rounds=2 * converged, max_steps=steps
    )
    image = report["images"]["env"]
    assert image["kind"] == "sqrt" and "band_limit" in image
    if converged:
        # Faint sqrt pixels set the want_grow noise floor: at most one pixel.
        assert image["kkt_rel"] < KKT_TOL and image["want_grow"] < 0.01
        assert image["band_limit"] < 2.0
    else:
        assert not report["converged"]


def _softmax_problem(n=32, scale=7.5):
    truth = Image(onp.log(_disc(n) + 1e-3), scale)
    data = DATA.with_model(truth, noise_scale=0.0)
    return truth, Image(onp.zeros((n, n)), scale), data


@pytest.fixture(scope="module")
def softmax_mem():
    with jax.enable_x64(True):
        _, start, data = _softmax_problem()
        priors, reg = image_priors(start), [MaxEntropy(30.0)]
        full = fit(start, priors, data, reg, method="lbfgs", gtol=1e-6)
        full = _polish(full, start, priors, data, reg, "lbfgs", 3)
    return start, priors, data, reg, full


def test_softmax_maxentropy_converges_and_stopped_fit_does_not(softmax_mem):
    start, priors, data, reg, full = softmax_mem
    short = fit(start, priors, data, reg, method="lbfgs", max_steps=5)
    done = kkt_report(full, priors, data, reg)
    assert done["converged"], done
    assert done["images"]["."]["band_limit"] < 2.0
    assert not kkt_report(short, priors, data, reg)["converged"]


def test_frozen_softmax_pixels_want_to_grow():
    # Fewer pixels than data, unregularised, with some starting at 1e-12: the
    # softmax cannot raise them, though the data would.
    n, scale = 6, 40.0
    truth = Image(onp.random.default_rng(3).normal(size=(n, n)), scale)
    data = DATA.with_model(truth, key=jax.random.PRNGKey(2), noise_scale=3.0)
    eta = onp.zeros((n, n))
    eta[onp.random.default_rng(0).random(eta.shape) < 0.2] = onp.log(1e-12)
    start = Image(eta, scale)
    priors = image_priors(start)
    result = fit(start, priors, data, (), method="lbfgs", gtol=1e-6)
    result = _polish(result, start, priors, data, (), "lbfgs", 1)
    report = kkt_report(result, priors, data, start={"log_brightness": eta})
    assert report["images"]["."]["want_grow"] > 0.0, report
    assert not report["converged"]
