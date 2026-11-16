"""Image(blur_fwhm_mas=...): analytic-Gaussian visibilities, FFT pixels."""

import warnings

import jax
import jax.numpy as np
import numpy as onp
import pytest

from virgil._geometry import UVGrid, gaussian_taper, image_visibilities
from virgil._utils import mas2rad
from virgil.fitting import fit
from virgil.imaging import MaxEntropy, TSV, TV, LogSum, image_priors
from virgil.models import Image
from virgil.radio import CalibrationNuisance, ComplexVisData, _RadioTerm

# regulariser tests below penalize blurred images on purpose
pytestmark = pytest.mark.filterwarnings(
    "ignore:smoothed=True penalizes:UserWarning"
)

N = 32
PIXEL = 2.0


def _spike(i=16, j=16, n=N, **kw):
    b = onp.full((n, n), -40.0)
    b[i, j] = 0.0
    return Image(b, PIXEL, ft="dft", **kw)


def _blob(**kw):
    rng = onp.random.default_rng(0)
    eta = rng.normal(size=(N, N)) * 0.3
    yy, xx = onp.mgrid[:N, :N]
    eta -= ((yy - 15.5) ** 2 + (xx - 15.5) ** 2) / 8.0  # compact in the field
    return Image(eta, PIXEL, ft="dft", **kw)


def _uv(n=400, qmax=0.17, seed=0):
    """Frequencies (per radian) per axis up to qmax cycles per pixel (|q| <= 0.25)."""
    rng = onp.random.default_rng(seed)
    scale = qmax / (PIXEL * mas2rad)
    return rng.uniform(-scale, scale, n), rng.uniform(-scale, scale, n)


def test_smoothed_point_source_is_the_analytic_gaussian():
    uu, vv = _uv()
    fwhm = 9.0
    img = _spike(blur_fwhm_mas=fwhm)
    got = onp.asarray(img._centred_cvis(uu, vv))
    want = onp.asarray(gaussian_taper(uu**2 + vv**2, fwhm))
    # the spike sits half a pixel from the centre on each axis: phase only
    onp.testing.assert_allclose(onp.abs(got), want, rtol=1e-5)
    sigma = fwhm / (2 * onp.sqrt(2 * onp.log(2)))
    q2 = uu**2 + vv**2
    onp.testing.assert_allclose(
        want, onp.exp(-2 * onp.pi**2 * (sigma * mas2rad) ** 2 * q2), rtol=1e-6
    )


@pytest.mark.parametrize("fwhm", [6.0, 8.0, 10.0])
def test_fft_smoothed_pixels_match_the_taper(fwhm):
    uu, vv = _uv()
    img = _blob(blur_fwhm_mas=fwhm)
    raw = image_visibilities(img.brightness, uu, vv, PIXEL, "dft")
    want = onp.asarray(raw) * onp.asarray(gaussian_taper(uu**2 + vv**2, fwhm))
    got = image_visibilities(img.smoothed(), uu, vv, PIXEL, "dft")
    # float32 FFT round-off dominates at ~1e-7 of the flux
    assert onp.max(onp.abs(onp.asarray(got) - want)) < 1e-6
    onp.testing.assert_allclose(
        onp.asarray(img._centred_cvis(uu, vv)), want, rtol=1e-5, atol=1e-6
    )


def test_smoothing_conserves_flux():
    img = _blob(blur_fwhm_mas=8.0)
    assert float(np.sum(img.smoothed())) == pytest.approx(1.0, abs=1e-6)


def test_no_smoothing_is_unchanged():
    uu, vv = _uv()
    plain = _blob()
    assert plain.blur_fwhm_mas is None
    assert plain.smoothed() is not None
    onp.testing.assert_array_equal(plain.smoothed(), plain.brightness)
    ref = image_visibilities(plain.brightness, uu, vv, PIXEL, "dft")
    onp.testing.assert_array_equal(plain._centred_cvis(uu, vv), ref)


def test_grid_path_is_smoothed_too():
    img = _blob(blur_fwhm_mas=8.0)
    wavel = 4.8e-6
    u_axis = np.linspace(-6.0, 6.0, 5)
    v_axis = np.linspace(-6.0, 6.0, 4)
    grid = UVGrid(u_axis, v_axis, np.arange(20), 0.0)
    u = np.tile(u_axis, 4)
    v = np.repeat(v_axis, 5)
    got = img.model_on_grid(u, v, wavel, grid)
    want = img.model(u, v, wavel)
    onp.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-6)
    # ... and the smoothing is really in it
    raw = _blob().model_on_grid(u, v, wavel, grid)
    assert float(np.max(np.abs(got - raw))) > 1e-3


def test_invalid_fwhm():
    with pytest.raises(ValueError, match="blur_fwhm_mas"):
        Image(onp.zeros((8, 8)), PIXEL, blur_fwhm_mas=-1.0)


def test_regulariser_on_smoothed_pixels():
    img = _blob(blur_fwhm_mas=8.0)
    raw = float(TSV(1.0, beam_mas=PIXEL).value(img))
    smoothed = float(TSV(1.0, smoothed=True, beam_mas=PIXEL).value(img))
    assert smoothed < raw  # smoothing removes the roughness
    b = onp.pad(onp.asarray(img.smoothed(), dtype=float), 1)
    dx = b[:-1, 1:] - b[:-1, :-1]
    dy = b[1:, :-1] - b[:-1, :-1]
    hand = (onp.sum(dx**2) + onp.sum(dy**2)) / onp.sum(b) ** 2
    assert smoothed == pytest.approx(hand, rel=1e-4)
    # default: raw pixels, whatever the smoothing
    plain = _blob()
    assert float(TSV(1.0, beam_mas=PIXEL).value(plain)) == pytest.approx(
        raw, rel=1e-6
    )


def test_max_entropy_on_smoothed_pixels_matches_hand_computation():
    img = _blob(blur_fwhm_mas=8.0)
    b = onp.clip(onp.asarray(img.smoothed(), dtype=float), 0, None)
    q = 1.0 / b.size
    nz = b > 0
    hand = 3.0 * onp.sum(b[nz] * onp.log(b[nz] / q))
    got = float(MaxEntropy(3.0, smoothed=True).value(img))
    assert got == pytest.approx(hand, rel=1e-4)
    assert got != pytest.approx(float(MaxEntropy(3.0).value(img)), rel=1e-3)


@pytest.mark.parametrize(
    "reg",
    [
        TV(1.0, smoothed=True, beam_mas=PIXEL),
        LogSum(1.0, smoothed=True),
        MaxEntropy(1.0, smoothed=True),
    ],
)
def test_other_regularisers_accept_smoothed(reg):
    img = _blob(blur_fwhm_mas=8.0)
    value, grad = jax.value_and_grad(lambda m: reg.value(m))(img)
    assert onp.isfinite(float(value))
    assert onp.all(onp.isfinite(onp.asarray(grad.log_brightness)))


def _problem(smooth, taper=None, n=800):
    rng = onp.random.default_rng(3)
    u = rng.uniform(-3e6, 3e6, n)
    v = rng.uniform(-3e6, 3e6, n)
    truth = _blob(blur_fwhm_mas=smooth)
    freq = 230e9
    lam = 299792458.0 / freq
    vis = onp.asarray(truth.model(u * lam, v * lam, lam))
    data = ComplexVisData.from_arrays(
        u, v, vis, onp.full(n, 1e4), freq, ddid=onp.zeros(n, int)
    )
    nuis = CalibrationNuisance(
        data,
        gains=False,
        offsets=False,
        sigmas=False,
        alpha=False,
        blur_fwhm_mas=taper,
    )
    start = Image(onp.zeros((N, N)), PIXEL, ft="dft", blur_fwhm_mas=smooth)
    return data, nuis, start


def test_fit_through_smoothed_image_compiles_once(monkeypatch):
    data, nuis, start = _problem(8.0)
    traced = []
    original = _RadioTerm.__call__

    def counting(self, values):
        if isinstance(values["log_brightness"], jax.core.Tracer):
            traced.append(1)
        return original(self, values)

    monkeypatch.setattr(_RadioTerm, "__call__", counting)

    def scene(values):
        return start.set("log_brightness", values["log_brightness"])

    def model(**values):
        return scene(values)

    def run(weight):
        priors = nuis.priors()
        priors.update(image_priors(start))
        before = len(traced)
        result = fit(
            model,
            priors,
            data=(),
            likelihoods=[data.term(scene, nuis)],
            regularizers=[MaxEntropy(weight, smoothed=True)],
            init={**nuis.defaults(), "log_brightness": start.log_brightness},
            method="lbfgs",
            max_steps=3,
        )
        return len(traced) - before, result

    n_first, result = run(1.0)
    assert n_first > 0
    assert onp.all(onp.isfinite(onp.asarray(result.values["log_brightness"])))
    assert onp.isfinite(result.info["chi2"][-1])
    n_second, _ = run(2.0)
    assert n_second == 0  # same shapes: no retrace


def test_double_taper_warns():
    data, nuis, start = _problem(8.0, taper=10.0)
    values = {k: np.asarray(v) for k, v in nuis.defaults().items()}
    term = data.term(lambda v: start, nuis)
    with pytest.warns(UserWarning, match="compound"):
        term(values)
    data, nuis, start = _problem(8.0, taper=None)
    term = data.term(lambda v: start, nuis)
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        term(values)


def test_old_names_are_deprecated_aliases():
    with pytest.warns(FutureWarning, match="blur_fwhm_mas"):
        old = _blob(smooth_fwhm_mas=8.0)
    new = _blob(blur_fwhm_mas=8.0)
    assert old.blur_fwhm_mas == old.smooth_fwhm_mas == 8.0
    uu, vv = _uv(50)
    onp.testing.assert_array_equal(
        old.model(uu, vv, 1.0), new.model(uu, vv, 1.0)
    )
    data, nuis, start = _problem(None, taper=10.0)
    with pytest.warns(FutureWarning, match="blur_fwhm_mas"):
        old_nuis = CalibrationNuisance(
            data,
            gains=False,
            offsets=False,
            sigmas=False,
            alpha=False,
            taper_fwhm_mas=10.0,
        )
    with pytest.warns(FutureWarning, match="blur_fwhm_mas"):
        assert old_nuis.taper_fwhm_mas == old_nuis.blur_fwhm_mas == 10.0
    values = {k: np.asarray(v) for k, v in nuis.defaults().items()}
    loss = [data.term(lambda v: start, n)(values) for n in (nuis, old_nuis)]
    onp.testing.assert_array_equal(loss[0], loss[1])


def test_both_names_raise():
    with pytest.raises(TypeError, match="both blur_fwhm_mas="):
        Image(onp.zeros((8, 8)), PIXEL, blur_fwhm_mas=1.0, smooth_fwhm_mas=1.0)
    data, _, _ = _problem(None)
    with pytest.raises(TypeError, match="both blur_fwhm_mas="):
        CalibrationNuisance(data, blur_fwhm_mas=1.0, taper_fwhm_mas=1.0)


@pytest.mark.parametrize(
    "smoothed, blur, warns",
    [(True, 8.0, True), (True, None, False), (False, 8.0, False)],
)
def test_smoothed_regulariser_on_a_blurred_image_warns(smoothed, blur, warns):
    img = _blob(blur_fwhm_mas=blur)
    reg = MaxEntropy(1.0, smoothed=smoothed)
    with warnings.catch_warnings(record=True) as caught:
        warnings.filterwarnings("always")
        reg.value(img)
    hits = [w for w in caught if "unregularized unless" in str(w.message)]
    assert bool(hits) == warns
