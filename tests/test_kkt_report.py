"""kkt_report: convergence certificates of image fits (not a loss plateau)."""

import jax
import numpy as onp
import numpyro.distributions as dist
import pytest

from virgil.coverage import ami_grid_record
from virgil.fitting import (
    KKT_TOL,
    KKT_W_TOL,
    FitResult,
    _band_limit,
    _linear_certificate,
    fit,
    kkt_report,
)
from virgil.imaging import TSV, MaxEntropy, image_priors
from virgil.models import Image, LinearImage, PointSource, System
from virgil.oidata import OIData

MAS = onp.pi / 180 / 3600e3  # radians per milliarcsecond

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


def _polish(
    result, start, priors, data, regularizers, method, rounds=3, gtol=1e-4
):
    """Restart a fit: the stop tolerance is relative to the start gradient, so a
    restart from a converged point goes on to a tighter one."""
    for _ in range(rounds):
        result = fit(
            start,
            priors,
            data,
            regularizers,
            method=method,
            gtol=gtol,
            init=result.values,
        )
    return result


def _report(
    n,
    scale,
    regularizers,
    parametrisation="linear",
    rounds=1,
    polish_gtol=1e-4,
    **options,
):
    start, priors, data = _problem(n, scale, parametrisation)
    result = fit(start, priors, data, regularizers, method="lbfgsb", **options)
    result = _polish(
        result,
        start,
        priors,
        data,
        regularizers,
        "lbfgsb",
        rounds,
        polish_gtol,
    )
    return kkt_report(result, priors, data, regularizers)


TSV_ = [TSV(3.0, path="env", beam_mas=60.0)]


def test_bounded_linear_optimum_is_converged_with_active_bounds():
    # Noise-free data from a disc with zeros around it: the disc is a global
    # minimum of chi^2 (zero residual) on the box pixels >= 0, so its gradient
    # is zero up to rounding, and its zero pixels are weakly active bounds (see
    # the next test for strictly active ones). pg_rel is
    # then rounding against the flat start's gradient, far below KKT_TOL. Built
    # directly rather than fitted (#674): where L-BFGS-B stalls is
    # platform-dependent, and a stalled fit is rightly not converged.
    n, scale = 10, 24.0
    flat = onp.full((n, n), 0.5 / n**2)
    start, disc = _scene(flat, scale), _disc(n, 0.5)
    data = DATA.with_model(_scene(disc, scale), noise_scale=0.0)
    priors = image_priors(start)
    result = FitResult(start, {"env.pixels": disc}, {})
    assert (disc == 0).any()  # bounds active
    report = kkt_report(result, priors, data, start={"env.pixels": flat})
    image = report["images"]["env"]
    assert report["converged"], report
    assert image["pg_rel"] < KKT_TOL
    assert (
        image["zero_violation"] == 0.0
    )  # bound-active zeros are not violations


def test_strictly_active_bounds_are_projected_out():
    # Zero pixels whose gradient pushes outward (g > 0, multiplier > 0) are at a
    # KKT point: x - clip(x - g, 0, inf) is exactly zero there, so pg_rel and
    # kkt_w are zero and nothing is a violation. With the sign flipped every
    # zero pixel wants flux.
    n = 10
    image = _scene(onp.full((n, n), 0.5 / n**2), 24.0).get("env")
    x = _disc(n, 0.5).ravel()
    g = onp.where(x == 0.0, 1.0, 0.0)
    out, ok = _linear_certificate(x, g, image, 1.0, 3.0)
    assert ok and out["pg_rel"] == 0.0 and out["kkt_w"] == 0.0
    assert out["zero_violation"] == 0.0
    out, ok = _linear_certificate(x, -g, image, 1.0, 3.0)
    assert not ok and out["zero_violation"] == 1.0 and out["pg_rel"] == 1.0


def test_bounded_linear_fit_stopped_early_is_not_converged():
    report = _report(10, 24.0, TSV_, rounds=0, max_steps=5)
    assert not report["converged"]
    assert report["images"]["env"]["kkt_w"] > KKT_W_TOL


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
    reg = [MaxEntropy(5.0, path="env")]
    report = _report(
        32,
        7.5,
        reg,
        "sqrt",
        rounds=3 * converged,
        polish_gtol=1e-7,
        max_steps=steps,
    )
    image = report["images"]["env"]
    assert image["kind"] == "sqrt" and "band_limit" in image
    if converged:
        assert image["kkt_rel"] < KKT_TOL and image["want_grow"] == 0.0
        assert image["kkt_w"] < KKT_W_TOL
        assert report["converged"] is True, report
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
    assert done["images"]["."]["kkt_w"] < KKT_W_TOL
    assert not kkt_report(short, priors, data, reg)["converged"]


def test_frozen_softmax_pixels_want_to_grow():
    # Four disc pixels start at 1e-12: their eta-gradient is b times their flux
    # gradient, so the softmax cannot raise them and the fit converges on the
    # other pixels (kkt_rel ~ 1e-8, so the want_grow margin is its KKT_TOL floor).
    # Their flux gradient keeps the entropy's -alpha ln b ~ 30 x 27.6 against
    # the others', far beyond that margin, whatever the fit's path. (An
    # unregularised fit, #674, drives other pixels to b = 0 on some platforms.)
    _, _, data = _softmax_problem()
    eta = onp.zeros((32, 32))
    eta[15:17, 15:17] = onp.log(1e-12)
    start = Image(eta, 7.5)
    priors, reg = image_priors(start), [MaxEntropy(30.0)]
    result = fit(start, priors, data, reg, method="lbfgs", gtol=1e-6)
    result = _polish(result, start, priors, data, reg, "lbfgs", 3)
    report = kkt_report(result, priors, data, reg)
    image = report["images"]["."]
    assert image["want_grow"] == 4 / 32**2, report
    assert not report["converged"]
    # With start, the gradient scale is at least the start's rms (data and
    # penalty far from the optimum), so kkt_rel can only shrink.
    ref = kkt_report(result, priors, data, reg, start={"log_brightness": eta})
    assert ref["images"]["."]["kkt_rel"] <= image["kkt_rel"], ref


def test_noise_parameters_are_nuisances():
    start, priors, data = _problem(6, 40.0, "sqrt")
    noise, reg = (
        {"vis_scale": dist.Uniform(0.0, 10.0)},
        [MaxEntropy(5.0, path="env")],
    )
    result = fit(start, priors, data, reg, noise=noise)
    report = kkt_report(result, priors, data, reg, noise=noise)
    assert any(k.startswith("noise") for k in report["nuisances"])


def _gaussian_log_image(n, width):
    yy, xx = onp.mgrid[:n, :n] - (n - 1) / 2
    return -(xx**2 + yy**2) / (2 * width**2)


def test_band_limit_is_small_for_smooth_and_large_for_rough_images():
    n, scale, cut = 32, 7.5, 1.0 / (4 * 7.5 * MAS)
    smooth = onp.exp(_gaussian_log_image(n, 5.0))
    rough = smooth * onp.exp(
        0.5 * onp.random.default_rng(0).normal(size=(n, n))
    )
    assert _band_limit(smooth, scale, cut) < 1.5
    assert _band_limit(rough, scale, cut) > 2.0 * _band_limit(
        smooth, scale, cut
    )


def _reference_band_limit(eta, dx, cut):
    """The diagnosis' f_eta / f_floor (testAB/ab.py), windows in mas FWHM."""
    n, fwhm = eta.shape[0], 2 * onp.sqrt(2 * onp.log(2))

    def conv(x, fwhm_mas):
        s, fk = fwhm_mas / fwhm / dx, onp.fft.fftfreq(2 * n)
        g = onp.exp(
            -2 * onp.pi**2 * s**2 * (fk[:, None] ** 2 + fk[None, :] ** 2)
        )
        pad = onp.fft.irfft2(
            onp.fft.rfft2(x, s=(2 * n, 2 * n)) * g[:, : n + 1],
            s=(2 * n, 2 * n),
        )
        return pad[:n, :n]

    def frac(f):
        k = onp.fft.fftfreq(2 * n, d=dx * MAS)
        q = onp.hypot(k[:, None], k[None, :])
        p = onp.abs(onp.fft.fft2(f, s=(2 * n, 2 * n))) ** 2
        return p[q > cut].sum() / p.sum()

    b = onp.exp(eta - eta.max())
    mask = conv(b, 30.0) >= 1e-3 * conv(b, 30.0).max()
    win = onp.clip(conv(mask.astype(float), 20.0), 0, 1)
    k = onp.fft.fftfreq(n, d=dx * MAS)
    q = onp.hypot(k[:, None], k[None, :])
    low = onp.fft.ifft2(onp.fft.fft2(eta) * (q <= cut)).real
    return frac(win * (eta - eta[mask].mean())) / frac(
        win * (low - low[mask].mean())
    )


def test_band_limit_matches_the_diagnosis_reference():
    n, dx = 48, 6.0
    eta = _gaussian_log_image(n, 6.0) + 0.3 * onp.random.default_rng(1).normal(
        size=(n, n)
    )
    # One resolution element is 18 mas; nudged so no frequency bin sits on the cut.
    cut = 1.0000001 / (18.0 * MAS)
    expected = _reference_band_limit(eta, dx, cut)
    assert _band_limit(onp.exp(eta), dx, cut) == pytest.approx(
        expected, rel=1e-6
    )
