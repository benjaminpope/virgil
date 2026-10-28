"""CLEAN of complex visibilities: point source, Gaussians, beam, weighting."""

import numpy as onp
import pytest

from virgil._utils import mas2rad
from virgil.radio import ComplexVisData
from virgil.radio_clean import (
    _auto_mask,
    _gaussian,
    _minor_cycle,
    clean,
    fit_beam,
)

S = 40.0  # mas per pixel
N = 97
C = (N - 1) / 2


def _uv(n=3000, umax=1.2e6, seed=0):
    rng = onp.random.default_rng(seed)
    r = umax * onp.sqrt(rng.uniform(0.02, 1.0, n))
    t = rng.uniform(0, 2 * onp.pi, n)
    return r * onp.cos(t), r * onp.sin(t)


def _vis(u, v, sources, blobs=()):
    """Point sources (flux, row, col) and circular Gaussians (flux, row,
    col, FWHM in pixels); V = Σ F exp(-2πi(u dra + v ddec))."""
    out = onp.zeros(len(u), complex)
    for flux, row, col in sources:
        dra, ddec = (C - col) * S * mas2rad, (C - row) * S * mas2rad
        out += flux * onp.exp(-2j * onp.pi * (u * dra + v * ddec))
    for flux, row, col, fwhm in blobs:
        dra, ddec = (C - col) * S * mas2rad, (C - row) * S * mas2rad
        t = fwhm * S * mas2rad
        taper = onp.exp(
            -((onp.pi * t) ** 2) * (u**2 + v**2) / (4 * onp.log(2))
        )
        out += flux * taper * onp.exp(-2j * onp.pi * (u * dra + v * ddec))
    return out


def _data(sources, blobs=(), seed=0):
    u, v = _uv(seed=seed)
    return ComplexVisData.from_arrays(
        u, v, _vis(u, v, sources, blobs), 1.0, 230e9
    )


@pytest.mark.parametrize("n", [96, 97])
@pytest.mark.parametrize("pa", [0.0, 30.0, 100.0, 165.0])
def test_fit_beam_recovers_a_gaussian(n, pa):
    psf = _gaussian((n, n), ((n - 1) / 2, (n - 1) / 2), 9.0, 5.0, pa)
    bmaj, bmin, pa_fit = fit_beam(psf, 2.0)
    assert bmaj == pytest.approx(18.0, rel=0.02)
    assert bmin == pytest.approx(10.0, rel=0.02)
    assert abs((pa_fit - pa + 90) % 180 - 90) < 1.0


def test_fit_beam_position_angle_from_explicit_sky_arithmetic():
    # elongated 30 deg East of North, built from (east, north) directly
    rr, cc = onp.indices((N, N), dtype=float)
    east, north = C - cc, C - rr  # column 0 is East, row 0 is North
    pa = onp.deg2rad(30.0)
    along = east * onp.sin(pa) + north * onp.cos(pa)
    across = east * onp.cos(pa) - north * onp.sin(pa)
    psf = onp.exp(-4 * onp.log(2) * ((along / 9.0) ** 2 + (across / 5.0) ** 2))
    assert fit_beam(psf, 1.0)[2] == pytest.approx(30.0, abs=1.0)
    # north-south baselines only: the beam is long in the East direction
    u = onp.zeros(40)
    v = onp.linspace(-3e5, 3e5, 40)
    one = ComplexVisData.from_arrays(u, v, onp.ones(40, complex), 1.0, 230e9)
    res = clean(one, N, S, niter=0)
    assert abs(res.beam[2] - 90.0) < 5.0


def test_point_source_is_recovered():
    data = _data([(2.0, C + 5, C - 7)])
    res = clean(data, N, S, niter=400, threshold=1e-3)
    assert res.stop == "threshold"
    row, col = onp.unravel_index(onp.argmax(res.model), res.model.shape)
    assert (row, col) == (C + 5, C - 7)
    assert res.model.sum() == pytest.approx(2.0, rel=0.02)
    assert res.image.max() == pytest.approx(2.0, rel=0.03)
    assert onp.abs(res.residual).max() < 1e-3
    assert res.psf[int(C), int(C)] == pytest.approx(1.0)
    # the model's visibilities reproduce the data
    miss = onp.asarray(data.vis) - res.model_visibilities(data)
    assert onp.abs(miss).max() < 5e-3


def test_two_sources_with_a_negative_residual_sidelobe_cleaned():
    data = _data([(1.0, C - 10, C + 12), (0.4, C + 14, C - 3)])
    res = clean(data, N, S, niter=2000, threshold=2e-3, cycleniter=100)
    assert res.history[-1]["peak_residual"] < 2e-3
    assert len(res.history) > 2  # several Cotton-Schwab major cycles
    assert res.model.sum() == pytest.approx(1.4, rel=0.03)


def test_multiscale_recovers_an_extended_source():
    blob = (1.0, C - 4, C + 6, 10.0)
    data = _data([], [blob])
    ms = clean(
        data,
        N,
        S,
        deconvolver="multiscale",
        scales_mas=[m * S for m in (0, 5, 10, 20)],
        niter=1500,
        threshold=3e-4,
    )
    hog = clean(data, N, S, niter=1500, threshold=3e-4)
    assert ms.model.sum() == pytest.approx(1.0, rel=0.05)
    # one extended component replaces many points: fewer iterations, and a
    # model that is smoother than the Hogbom point model
    assert ms.iterations < hog.iterations
    peak = ms.image[int(C - 4), int(C + 6)]
    assert peak == pytest.approx(_restored_peak(ms.beam), rel=0.1)


def _restored_peak(beam):
    """Peak (Jy/beam) of the unit-flux 10-pixel Gaussian seen with ``beam``:
    the beam area over the area of the blob convolved with it."""
    a2 = (10.0 * S) ** 2
    return beam[0] * beam[1] / (a2 + beam[0] * beam[1])


def test_weighting_and_taper_change_the_beam():
    data = _data([(1.0, C, C)])
    beams = {}
    for name, kw in {
        "natural": dict(weighting="natural"),
        "uniform": dict(weighting="uniform"),
        "taper": dict(weighting="natural", uvtaper=(400.0, 300.0, 20.0)),
    }.items():
        beams[name] = clean(data, N, S, niter=0, **kw).beam
    assert beams["uniform"][0] < beams["natural"][0]
    assert beams["taper"][0] > beams["natural"][0]
    assert beams["taper"][0] > beams["taper"][1]
    # a taper of FWHM t widens a Gaussian beam b towards √(b² + t²) (the
    # PSF is not Gaussian, so the fitted lobe falls a little short)
    b = beams["natural"]
    assert beams["taper"][0] == pytest.approx(onp.hypot(b[0], 400.0), rel=0.25)
    assert beams["taper"][2] == pytest.approx(20.0, abs=2.0)


def test_taper_beam_peaks_at_one_and_matches_dirty_beam():
    data = _data([(1.0, C, C)])
    res = clean(data, N, S, niter=0, uvtaper=300.0)
    ref = data.dirty_beam(N, S, uvtaper=300.0)
    onp.testing.assert_allclose(res.psf, ref, atol=1e-5)
    assert ref[int(C), int(C)] == pytest.approx(1.0)


def test_auto_mask_finds_the_source_and_nsigma_stops():
    data = _data([(1.0, C - 6, C + 8)])
    rng = onp.random.default_rng(1)
    noise = 0.02 * (
        rng.normal(size=data.vis.shape) + 1j * rng.normal(size=data.vis.shape)
    )
    data = ComplexVisData.from_arrays(
        data.u, data.v, onp.asarray(data.vis) + noise, 1.0, 230e9
    )
    res = clean(data, N, S, niter=500, usemask="auto-multithresh", nsigma=3.0)
    assert res.mask[int(C - 6), int(C + 8)]
    assert res.mask.sum() < 0.2 * N * N
    assert res.model.sum() == pytest.approx(1.0, rel=0.05)
    assert res.stop == "nsigma"


def test_auto_mask_seeds_the_peak_when_the_psf_sidelobe_is_high():
    # sidelobethreshold x sidelobe x peak = 1.8 x the peak: the rule alone
    # selects nothing, and clean() used to stop at once with no_progress
    rng = onp.random.default_rng(3)
    resid = _gaussian(
        (N, N), (C - 5, C + 7), 6.0, 6.0, 0.0
    ) + 0.01 * rng.normal(size=(N, N))
    mask = _auto_mask(resid, 5.0, 0.6, 3.0, 5.0, 1.5)
    assert mask[int(C - 5), int(C + 7)] and 0 < mask.sum() < 0.2 * N * N


def test_user_mask_limits_components():
    data = _data([(1.0, C - 6, C + 8), (1.0, C + 10, C - 10)])
    mask = onp.zeros((N, N), bool)
    mask[int(C - 12) : int(C), int(C + 2) : int(C + 14)] = True
    res = clean(data, N, S, niter=200, mask=mask)
    assert onp.all(res.model[~mask] == 0)
    assert res.model.sum() == pytest.approx(1.0, rel=0.1)


def test_restoring_beam_and_stop_codes():
    data = _data([(1.0, C, C)])
    res = clean(data, N, S, niter=5, restoringbeam=(300.0, 250.0, 10.0))
    assert res.beam == (300.0, 250.0, 10.0)
    assert res.stop == "niter" and res.iterations == 5


def test_write_fits_header_and_orientation(tmp_path):
    from astropy.io import fits
    from astropy.wcs import WCS

    row, col = int(C - 6), int(C + 8)  # North and West of the centre
    data = _data([(1.0, row, col)])
    res = clean(data, N, S, niter=300, threshold=1e-3)
    res.write_fits(tmp_path / "t", ra_deg=10.0, dec_deg=-30.0, freq_hz=230e9)
    hdul = fits.open(tmp_path / "t.image.fits")
    hdr = hdul[0].header
    assert hdr["BUNIT"] == "Jy/beam"
    assert hdr["BMAJ"] * 3.6e6 == pytest.approx(res.beam[0], rel=1e-5)
    assert hdr["RESTFRQ"] == 230e9
    assert fits.getheader(tmp_path / "t.model.fits")["BUNIT"] == "Jy/pixel"
    y, x = onp.unravel_index(onp.argmax(hdul[0].data), hdul[0].data.shape)
    ra, dec = WCS(hdr).all_pix2world([x], [y], 0)
    # col > centre is West (RA decreases), row < centre is North
    assert ra[0] < 10.0 and dec[0] > -30.0


def test_errors():
    data = _data([(1.0, C, C)])
    with pytest.raises(ValueError, match="odd"):
        clean(data, 96, S)
    with pytest.raises(ValueError, match="deconvolver"):
        clean(data, N, S, deconvolver="mtmfs")
    with pytest.raises(TypeError):
        clean(object(), N, S)


def _sparse_track_data(n=65, scale=4.0):
    """300 baselines x 80 hour angles (a poor, sidelobe-ridden beam) and three
    point sources summing to 1 Jy, one near an edge (virgil#551)."""
    rng = onp.random.default_rng(1)
    bl = onp.exp(rng.uniform(onp.log(100), onp.log(1e4), (300, 2)))
    bl *= rng.choice([-1, 1], (300, 2))
    ha = onp.deg2rad(onp.linspace(-45, 45, 80))
    dec, lat = onp.deg2rad(-37), onp.deg2rad(-23)
    u = bl[:, :1] * onp.cos(ha) - bl[:, 1:] * onp.sin(lat) * onp.sin(ha)
    v = bl[:, :1] * onp.sin(dec) * onp.sin(ha) + bl[:, 1:] * (
        onp.sin(lat) * onp.sin(dec) * onp.cos(ha) + onp.cos(lat) * onp.cos(dec)
    )
    u, v = (u / 1.3e-3).ravel(), (v / 1.3e-3).ravel()
    c = (n - 1) / 2
    vis = onp.zeros(len(u), complex)
    for flux, row, col in [
        (0.5, c + 16, c - 13),
        (0.3, c - 21, c + 21),
        (0.2, c, c),
    ]:
        phase = u * (c - col) + v * (c - row)
        vis += flux * onp.exp(-2j * onp.pi * phase * scale * mas2rad)
    noise = 0.01 * (rng.normal(size=u.size) + 1j * rng.normal(size=u.size))
    return ComplexVisData.from_arrays(u, v, vis + noise, 1e4, 230e9)


def test_clean_converges_when_the_psf_reaches_beyond_half_the_image():
    # the PSF sidelobes are 0.13 at lags 32-48 pixels of a 65-pixel image; a PSF
    # window clipped at 32 left them in the minor-cycle residual and the
    # components ran away (virgil#551)
    data = _sparse_track_data()
    sigma = 1 / onp.sqrt(onp.sum(onp.asarray(data.weight)))
    res = clean(data, 65, 4.0, niter=3000, threshold=3 * sigma)
    peaks = [h["peak_residual"] for h in res.history]
    assert res.stop == "threshold"
    assert all(b < a for a, b in zip(peaks, peaks[1:]))
    # compact sources, so the total flux is constrained despite the missing
    # short spacings
    assert res.model.sum() == pytest.approx(1.0, abs=0.01)


def test_clean_stops_with_diverged_when_the_psf_window_is_too_small():
    data = _sparse_track_data()
    res = clean(data, 65, 4.0, niter=3000, threshold=1e-4, psfpatch=6)
    assert res.stop == "diverged"
    assert (
        res.history[-1]["peak_residual"] < 10 * res.history[0]["peak_residual"]
    )


def test_the_returned_psf_and_restoring_beam_stay_image_sized():
    # only the minor cycle's PSF is larger than the image
    data = _sparse_track_data()
    res = clean(data, 65, 4.0, niter=50, threshold=1e-3)
    assert res.psf.shape == res.residual.shape == (65, 65)
    assert res.beam == fit_beam(res.psf, 4.0)


def test_one_component_clears_the_whole_residual_with_the_full_window():
    # a corner source: the PSF window must reach the opposite corner
    n, c = 65, 32
    data = _sparse_track_data(n)
    w = data._imaging_weights(
        "natural", 0.5, 1 / (n * 4.0 * mas2rad), None, None, uvtaper=None
    )
    ones = onp.ones(len(data), complex)
    phase = (data.u + data.v) * 25 * 4.0 * mas2rad
    resid = data._adjoint(onp.exp(2j * onp.pi * phase), w, n, 4.0, 10**6)
    full = data._adjoint(ones, w, 2 * n - 1, 4.0, 10**6)
    kw = dict(
        scales=[0],
        mask=onp.ones((n, n), bool),
        gain=1.0,
        niter=5,
        threshold=1e-4,
        bias=0.6,
        patch=256,
    )
    _, used, _ = _minor_cycle(resid, full, **kw)
    assert used == 1
    # the image-sized PSF leaves the far sidelobes behind (virgil#551)
    _, used, _ = _minor_cycle(resid, full[c : c + n, c : c + n], **kw)
    assert used > 1


@pytest.mark.parametrize(
    "kw",
    [
        dict(
            deconvolver="multiscale",
            scales_mas=[0, 16, 48],
            usemask="auto-multithresh",
        ),
        dict(weighting="briggs", robust=0.0),
    ],
    ids=["multiscale-auto-multithresh", "briggs"],
)
def test_convergent_runs_are_not_stopped_as_diverged(kw):
    data = _sparse_track_data()
    sigma = 1 / onp.sqrt(onp.sum(onp.asarray(data.weight)))
    res = clean(data, 65, 4.0, niter=3000, threshold=3 * sigma, **kw)
    assert res.stop == "threshold"
    assert res.model.sum() == pytest.approx(1.0, abs=0.01)
    # each major cycle that was followed by a minor cycle says why it ended
    assert all(
        h["minor_stop"] in ("threshold", "diverged", "niter")
        for h in res.history[:-1]
    )


def test_the_minor_cycle_psf_agrees_with_the_image_sized_one():
    # the 2N - 1 PSF is the same adjoint on a bigger grid: peak at its centre,
    # equal to the image-sized PSF in the overlap to the NUFFT's accuracy
    n = 65
    data = _sparse_track_data(n)
    w = data._imaging_weights(
        "natural", 0.5, 1 / (n * 4.0 * mas2rad), None, None, uvtaper=None
    )
    ones = onp.ones(len(data), complex)
    small = data._adjoint(ones, w, n, 4.0, 10**6)
    large = data._adjoint(ones, w, 2 * n - 1, 4.0, 10**6)
    assert onp.unravel_index(large.argmax(), large.shape) == (n - 1, n - 1)
    overlap = large[n // 2 : n // 2 + n, n // 2 : n // 2 + n]
    assert onp.abs(overlap - small).max() < 1e-5


def test_the_returned_psf_and_beam_match_stored_reference_values():
    # values from clean() on main, before the minor cycle got its own PSF
    # (rel 1e-5: the NUFFT differs by about 3e-8 across platforms)
    res = clean(_sparse_track_data(), 65, 4.0, niter=0)
    assert res.beam == pytest.approx(
        (49.0283201406404, 45.953434550067186, 130.89339973020668), rel=1e-5
    )
    assert res.psf.sum() == pytest.approx(570.4904774509284, rel=1e-5)
    assert (res.psf**2).sum() == pytest.approx(137.9294823556553, rel=1e-5)
    assert res.psf[32, 32] == pytest.approx(0.9999999550513774, rel=1e-5)
    assert res.psf[10, 50] == pytest.approx(0.11298123905795, rel=1e-5)
