"""virgil.radio: complex-visibility data, calibration nuisances, helpers."""

import os

import equinox as eqx
import jax
import jax.numpy as np
import numpy as onp
import pytest
from scipy import stats

import virgil
from virgil import (
    CalibrationNuisance,
    ComplexVisData,
    GaussianDisk,
    Image,
    PointSource,
    dartboard_folds,
    fit,
    primary_beam,
)
from virgil._geometry import _FT_DEFAULTS
from virgil._utils import mas2rad, rad2mas
from virgil.radio import C_LIGHT


@pytest.fixture(autouse=True)
def float64():
    """Tight tolerances need float64; opt in locally, as AGENTS.md asks."""
    with jax.enable_x64(True):
        yield


PIXEL = 2.0  # mas
NPIX = 64


def _data(
    n=1500, n_obs=2, ddids_per_obs=(2, 1), seed=0, noise=0.0, scene=None
):
    """Random uv points over ddids of several observations; vis = scene or noise."""
    rng = onp.random.default_rng(seed)
    ddid_obs = [e for e, k in enumerate(ddids_per_obs) for _ in range(k)]
    ddid = rng.integers(0, len(ddid_obs), n)
    obs = onp.asarray(ddid_obs)[ddid]
    freq = onp.asarray([230e9, 238e9, 245e9])[ddid]
    u = rng.uniform(-4e6, 4e6, n)
    v = rng.uniform(-4e6, 4e6, n)
    weight = rng.uniform(0.5, 2.0, n) * 1e4
    vis = rng.normal(size=n) + 1j * rng.normal(size=n)
    return ComplexVisData.from_arrays(
        u, v, vis, weight, freq, ddid=ddid + 10, obs=obs + 3
    )


def _image(seed=1, ft="dft", **kw):
    eta = onp.random.default_rng(seed).normal(size=(NPIX, NPIX)) * 0.5
    return Image(eta, PIXEL, ft=ft, **kw)


def _values(nuis, **over):
    values = {k: np.asarray(v) for k, v in nuis.defaults().items()}
    values.update({k: np.asarray(v) for k, v in over.items()})
    return values


def _expected(data, scene, flux, gain, offset, alpha, nu0):
    """The model written out directly, per point."""
    lam = C_LIGHT / onp.asarray(data.freq)
    u, v = onp.asarray(data.u), onp.asarray(data.v)
    vis = onp.asarray(scene.model(u * lam, v * lam, lam))
    obs = onp.asarray(data.obs)
    ramp = onp.exp(
        -2j * onp.pi * mas2rad * (offset[obs, 0] * u + offset[obs, 1] * v)
    )
    return (
        flux * gain[obs] * (onp.asarray(data.freq) / nu0) ** alpha * ramp * vis
    )


def test_model_matches_dft_and_nufft():
    data = _data()
    nuis = CalibrationNuisance(data, nu0_hz=239e9)
    values = _values(
        nuis,
        **{
            "cal.flux": 0.37,
            "cal.gain": [1.2],
            "cal.offset": [[7.0, -3.0]],
            "cal.alpha": 2.3,
        },
    )
    dft = _image(ft="dft")
    got = onp.asarray(data.model(dft, values, nuis))
    want = _expected(
        data,
        dft,
        0.37,
        onp.array([1.0, 1.2]),
        onp.array([[0.0, 0.0], [7.0, -3.0]]),
        2.3,
        239e9,
    )
    assert onp.max(onp.abs(got - want) / onp.abs(want)) < 1e-10

    nufft = _image(ft="nufft")
    got_n = onp.asarray(data.model(nufft, values, nuis))
    eps = _FT_DEFAULTS["eps_f64"]
    assert onp.linalg.norm(got_n - got) / onp.linalg.norm(got) <= 3 * eps


def test_offset_ramp_is_a_shift_of_the_scene():
    data = _data()
    nuis = CalibrationNuisance(
        data, gains=False, sigmas=False, alpha=False, nu0_hz=239e9
    )
    off = onp.array([[11.0, -6.0]])
    values = _values(nuis, **{"cal.flux": 1.0, "cal.offset": off})
    got = onp.asarray(data.model(_image(), values, nuis))
    shifted = _image(dra=11.0, ddec=-6.0)
    lam = C_LIGHT / onp.asarray(data.freq)
    u, v = onp.asarray(data.u), onp.asarray(data.v)
    want = onp.asarray(shifted.model(u * lam, v * lam, lam))
    sel = onp.asarray(data.obs) == 1
    onp.testing.assert_allclose(got[sel], want[sel], rtol=1e-9, atol=1e-12)
    # the reference observation is not shifted
    ref = onp.asarray(_image().model(u * lam, v * lam, lam))
    onp.testing.assert_allclose(got[~sel], ref[~sel], rtol=1e-9, atol=1e-12)


def _nll(data, scene, nuis, values, mask=None):
    """-Σ log N(Re) - Σ log N(Im) by scipy, with the scaled errors."""
    model = onp.asarray(data.model(scene, values, nuis))
    s = onp.asarray(nuis._unpack(values)[3])[onp.asarray(data.ddid)]
    sigma = s / onp.sqrt(onp.asarray(data.weight))
    keep = onp.ones(len(data), bool) if mask is None else mask
    r = onp.asarray(data.vis) - model
    return -(
        stats.norm.logpdf(r.real[keep], scale=sigma[keep]).sum()
        + stats.norm.logpdf(r.imag[keep], scale=sigma[keep]).sum()
    )


def test_normalization_matches_gaussian_density():
    data = _data()
    nuis = CalibrationNuisance(data)
    scene = _image()
    term = data.term(lambda v: scene, nuis, chunk_size=400)
    assert term.has_log_norm
    shifts = []
    for s in (1.0, 1.7):
        values = _values(nuis, **{"cal.flux": 0.2, "cal.sigma": [s, 0.8, s]})
        chi2 = float(np.sum(term(values) ** 2))
        loss = 0.5 * chi2 + float(term.log_norm(values))
        nll = _nll(data, scene, nuis, values)
        shifts.append(loss - nll)
        assert float(term.loglike(values)) == pytest.approx(-nll, rel=1e-10)
    # the difference is independent of the parameters (it is Σ ln w / 2 - N ln 2π ...)
    assert shifts[0] == pytest.approx(shifts[1], rel=1e-10)
    assert term(values).shape == (2 * len(data),)


def test_masked_points_do_not_count():
    data = _data()
    nuis = CalibrationNuisance(data)
    scene = _image()
    mask = onp.random.default_rng(3).random(len(data)) < 0.7
    values = _values(nuis, **{"cal.flux": 0.2, "cal.sigma": [1.2, 0.9, 1.1]})
    full = data.term(lambda v: scene, nuis, mask=mask, chunk_size=500)
    sub = data.subset(mask).term(lambda v: scene, nuis, chunk_size=500)
    chi2 = float(np.sum(full(values) ** 2))
    assert chi2 == pytest.approx(float(np.sum(sub(values) ** 2)), rel=1e-12)
    assert float(full.log_norm(values)) == pytest.approx(
        float(sub.log_norm(values)), rel=1e-12
    )
    # corrupting masked points changes nothing
    bad = data.vis.at[np.asarray(~mask)].set(1e6)
    data2 = eqx.tree_at(lambda d: d.vis, data, bad)
    full2 = data2.term(lambda v: scene, nuis, mask=mask, chunk_size=500)
    assert float(np.sum(full2(values) ** 2)) == pytest.approx(chi2, rel=1e-12)
    c, n = data.chi2_by_ddid(lambda v: scene, values, nuis, mask=mask)
    assert n.sum() == mask.sum()
    assert c.sum() == pytest.approx(chi2, rel=1e-10)
    assert list(n) == [
        int(((onp.asarray(data.ddid) == d) & mask).sum()) for d in range(3)
    ]


def test_gradients_match_finite_differences():
    data = _data(n=300)
    nuis = CalibrationNuisance(data)
    eta0 = onp.random.default_rng(5).normal(size=(NPIX, NPIX)) * 0.5

    def loss(eta, values):
        scene = Image(eta, PIXEL, ft="dft")
        term = data.term(lambda v: scene, nuis, chunk_size=128)
        return 0.5 * np.sum(term(values) ** 2) + term.log_norm(values)

    values = _values(
        nuis,
        **{
            "cal.flux": 0.3,
            "cal.gain": [1.1],
            "cal.offset": [[4.0, -2.0]],
            "cal.sigma": [1.1, 0.9, 1.3],
            "cal.alpha": 2.2,
        },
    )
    g_eta, g_val = jax.grad(loss, argnums=(0, 1))(np.asarray(eta0), values)
    rng = onp.random.default_rng(6)
    for _ in range(3):  # random directions in the image
        d = rng.normal(size=eta0.shape)
        h = 1e-5
        fd = (
            loss(np.asarray(eta0 + h * d), values)
            - loss(np.asarray(eta0 - h * d), values)
        ) / (2 * h)
        assert float(np.sum(g_eta * d)) == pytest.approx(float(fd), rel=1e-6)
    for key, val in values.items():
        flat = onp.asarray(val, dtype=float).ravel()
        for i in range(flat.size):

            def f(x, i=i, key=key, val=val):
                new = flat.copy()
                new[i] = x
                return loss(
                    np.asarray(eta0),
                    {**values, key: np.asarray(new.reshape(onp.shape(val)))},
                )

            h = 1e-6 * max(1.0, abs(flat[i]))
            fd = (f(flat[i] + h) - f(flat[i] - h)) / (2 * h)
            got = onp.asarray(g_val[key]).ravel()[i]
            assert got == pytest.approx(float(fd), rel=1e-5, abs=1e-6), (
                key,
                i,
            )


def test_fit_recovers_nuisances():
    rng = onp.random.default_rng(11)
    n = 2000
    truth = dict(
        flux=0.5, gain=1.15, off=(6.0, -4.0), s=(1.3, 0.8, 1.0), alpha=2.6
    )
    ddid = rng.integers(0, 3, n)
    ddid_obs = onp.array([0, 0, 1])
    obs = ddid_obs[ddid]
    freq = onp.array([230e9, 240e9, 245e9])[ddid]
    u = rng.uniform(-3e6, 3e6, n)
    v = rng.uniform(-3e6, 3e6, n)
    weight = onp.full(n, 4.0e4)
    sigma = onp.array(truth["s"])[ddid] / onp.sqrt(weight)
    template = Image.from_model(
        GaussianDisk(sigma=15.0), NPIX, PIXEL, ft="dft"
    )
    lam = C_LIGHT / freq
    clean = onp.asarray(template.model(u * lam, v * lam, lam))
    ramp = onp.where(
        obs == 1,
        onp.exp(
            -2j
            * onp.pi
            * mas2rad
            * (truth["off"][0] * u + truth["off"][1] * v)
        ),
        1.0,
    )
    vis = (
        truth["flux"]
        * onp.where(obs == 1, truth["gain"], 1.0)
        * (freq / 239e9) ** truth["alpha"]
        * ramp
        * clean
    )
    vis = vis + sigma * (rng.normal(size=n) + 1j * rng.normal(size=n))
    data = ComplexVisData.from_arrays(
        u, v, vis, weight, freq, ddid=ddid, obs=obs
    )
    nuis = CalibrationNuisance(data, nu0_hz=239e9)
    priors = nuis.priors(offset=(-100.0, 100.0))
    # vector-valued sites reach fit() as arrays
    assert priors["cal.gain"].batch_shape == (1,)
    term = data.term(lambda v: template, nuis, chunk_size=700)

    def model(**values):
        return template

    init = nuis.defaults()
    result = fit(
        model, priors, data=(), likelihoods=[term], init=init, method="lbfgs"
    )
    got = result.values
    # parameter uncertainties from the Gaussian approximation (Hessian)
    z_flux = float(got["cal.flux"])
    assert z_flux == pytest.approx(truth["flux"], rel=0.05)
    assert float(got["cal.gain"][0]) == pytest.approx(truth["gain"], rel=0.05)
    onp.testing.assert_allclose(
        onp.asarray(got["cal.offset"])[0], truth["off"], atol=1.0
    )
    onp.testing.assert_allclose(
        onp.asarray(got["cal.sigma"]), truth["s"], rtol=0.06
    )
    assert float(got["cal.alpha"]) == pytest.approx(truth["alpha"], abs=0.3)


def test_dartboard_folds():
    data = _data(n=3000)
    # add Hermitian partners
    u = onp.concatenate([onp.asarray(data.u), -onp.asarray(data.u)])
    v = onp.concatenate([onp.asarray(data.v), -onp.asarray(data.v)])
    n = len(data)
    full = ComplexVisData.from_arrays(
        u, v, onp.ones(2 * n), onp.ones(2 * n), 230e9
    )
    fold, annulus = dartboard_folds(full, k=5, n_q=12, n_phi=16, seed=2)
    assert onp.array_equal(fold[:n], fold[n:])
    assert onp.array_equal(annulus[:n], annulus[n:])
    assert set(fold) == set(range(5))
    assert annulus.min() == 0 and annulus.max() <= 11
    again, _ = dartboard_folds(full, k=5, n_q=12, n_phi=16, seed=2)
    assert onp.array_equal(fold, again)


def test_primary_beam_half_power():
    freq, dish, factor = 230e9, 12.0, 1.13
    fwhm = factor * C_LIGHT / (freq * dish) * rad2mas
    npix, scale = (
        257,
        0.5 * fwhm / 128,
    )  # odd grid: a pixel at r = FWHM/2 on the axis
    pb = primary_beam(npix, scale, freq, dish, factor)
    centre = npix // 2
    assert pb[centre, centre] == 0.0
    # half power at r = FWHM / 2, along either axis
    assert onp.exp(pb[centre, centre + 128]) == pytest.approx(0.5, rel=1e-12)
    assert onp.exp(pb[centre + 128, centre]) == pytest.approx(0.5, rel=1e-12)


def test_taper_is_a_gaussian_convolution():
    data = _data(n=200)
    theta = 30.0
    nuis = CalibrationNuisance(
        data,
        gains=False,
        offsets=False,
        sigmas=False,
        alpha=False,
        blur_fwhm_mas=theta,
    )
    values = _values(nuis, **{"cal.flux": 1.0})
    sigma = theta / (2 * onp.sqrt(2 * onp.log(2)))
    plain = PointSource()
    got = onp.asarray(data.model(plain, values, nuis))
    q2 = onp.asarray(data.u) ** 2 + onp.asarray(data.v) ** 2
    want = onp.exp(-2 * onp.pi**2 * (sigma * mas2rad) ** 2 * q2)
    onp.testing.assert_allclose(got.real, want, rtol=1e-12)


def test_from_arrays_validation():
    with pytest.raises(ValueError, match="weight"):
        ComplexVisData.from_arrays([1.0], [1.0], [1 + 0j], [0.0], 230e9)
    with pytest.raises(ValueError, match="finite"):
        ComplexVisData.from_arrays([1.0], [1.0], [onp.nan + 0j], [1.0], 230e9)


def test_select_keeps_labels():
    data = _data()
    sub = data.select(ddids=[11])
    assert set(onp.asarray(sub.ddid)) == {1}
    assert sub.ddid_ids == data.ddid_ids and sub.obs_ids == (3, 4)
    assert len(data.select(obs=[4])) == int((onp.asarray(data.obs) == 1).sum())


def test_empty_obs_and_ddid_groups_after_subset_produce_no_nan():
    """After ``subset`` drops all rows of an observation and a ddid,
    ``CalibrationNuisance`` priors, ``sigma_hat`` and ``term`` stay finite."""
    data = _data(n=600, n_obs=3, ddids_per_obs=(1, 1, 1), seed=7)
    # Drop observation 1 (and its ddid 1) completely via subset.
    sub = data.subset(onp.asarray(data.obs) != 1)
    assert sub.n_obs if hasattr(sub, "n_obs") else len(sub.obs_ids) == 3
    for sigmas in (True, "profile"):
        nuis = CalibrationNuisance(sub, sigmas=sigmas, alpha=0.7)
        assert onp.all(onp.isfinite(nuis.offset_unit_mas))
        for bounds in ({}, {"offset": (-20.0, 20.0)}):
            priors = nuis.priors(**bounds)
            assert onp.all(onp.isfinite(onp.asarray(priors["cal.offset"].low)))
            assert onp.all(
                onp.isfinite(onp.asarray(priors["cal.offset"].high))
            )
        values = _values(nuis)
        scene = PointSource()
        shat = sub.sigma_hat(scene, values, nuis)
        assert onp.all(onp.isfinite(shat))
        if sigmas == "profile":
            assert shat[1] == 1.0
        term = sub.term(lambda v: scene, nuis, chi2="direct")
        r, log_norm = term.residuals_and_log_norm(values)
        assert onp.all(onp.isfinite(r)) and onp.isfinite(log_norm)


# ----------------------------------------------------------------------
# A synthetic measurement set from CASA's simulator (needs casatools)
# ----------------------------------------------------------------------

# Phase centre and the CASA-predicted point source's offset (mas).
PHASE_CENTRE = (
    onp.deg2rad(15.0 * (15 + 56 / 60 + 9.20 / 3600)),
    -onp.deg2rad(37 + 56 / 60 + 6.42 / 3600),
)
POINT_MAS = (300.0, 100.0)  # East, North
SPW_FREQ = (230e9, 245e9)
NCHAN, CHAN_WIDTH = 4, 125e6

# Local East, North, Up (m) of 8 antennas: compact, so the w term and the
# curvature of the sky stay below the test's tolerance.
ANTENNAS = onp.array(
    [
        [0.0, 0.0, 0.0],
        [38.0, 11.0, 0.5],
        [-52.0, 64.0, -0.3],
        [75.0, -83.0, 0.2],
        [-17.0, -121.0, 0.8],
        [112.0, 45.0, -0.6],
        [-96.0, -22.0, 0.1],
        [23.0, 140.0, 0.4],
    ]
)


def _tangent_to_radec(ra0, dec0, east_mas, north_mas):
    """Gnomonic inverse: sky position of a tangent-plane offset."""
    xi, eta = east_mas * mas2rad, north_mas * mas2rad
    ra = ra0 + onp.arctan2(xi, onp.cos(dec0) - eta * onp.sin(dec0))
    dec = onp.arcsin(
        (onp.sin(dec0) + eta * onp.cos(dec0)) / onp.sqrt(1 + xi**2 + eta**2)
    )
    return ra, dec


@pytest.fixture(scope="module")
def synthetic_ms(tmp_path_factory):
    """(path, casatools version): a point source predicted by CASA."""
    measures = os.environ.get("VIRGIL_CASA_MEASURES")
    if measures:  # CASA's measures data, if not in ~/.casa
        from casaconfig import config

        config.measurespath = measures
        config.measures_auto_update = False
        config.data_auto_update = False
    try:
        import casatools

        me = casatools.measures()
        reference = me.observatory("ALMA")
    except ImportError:
        pytest.skip("casatools is not installed")
    except Exception as err:  # no measures data
        pytest.skip(f"casatools is unusable (measures data?): {err}")
    root = tmp_path_factory.mktemp("ms")
    casatools.logsink().setlogfile(str(root / "casa.log"))
    ms, cl = str(root / "sim.ms"), str(root / "point.cl")
    ra, dec = _tangent_to_radec(*PHASE_CENTRE, *POINT_MAS)
    comp = casatools.componentlist()
    comp.addcomponent(
        flux=[1.0, 0, 0, 0],
        fluxunit="Jy",
        polarization="Stokes",
        dir=f"J2000 {ra:.15f}rad {dec:.15f}rad",
        shape="point",
        freq="230GHz",
    )
    comp.rename(cl)
    comp.close()
    sm = casatools.simulator()
    sm.open(ms)
    sm.setconfig(
        telescopename="ALMA",
        x=list(ANTENNAS[:, 0]),
        y=list(ANTENNAS[:, 1]),
        z=list(ANTENNAS[:, 2]),
        dishdiameter=[12.0] * len(ANTENNAS),
        mount=["alt-az"] * len(ANTENNAS),
        antname=[f"A{i:02d}" for i in range(len(ANTENNAS))],
        coordsystem="local",
        referencelocation=reference,
    )
    sm.setfeed(mode="perfect X Y")
    sm.setfield(
        sourcename="src",
        sourcedirection=f"J2000 {PHASE_CENTRE[0]:.15f}rad {PHASE_CENTRE[1]:.15f}rad",
    )
    sm.setlimits(shadowlimit=0.001, elevationlimit="10deg")
    sm.setauto(autocorrwt=0.0)
    sm.settimes(
        integrationtime="30s",
        usehourangle=True,
        referencetime=me.epoch("utc", "2025/03/01"),
    )
    for i, f in enumerate(SPW_FREQ):
        sm.setspwindow(
            spwname=f"SPW{i}",
            freq=f"{f}Hz",
            deltafreq=f"{CHAN_WIDTH}Hz",
            freqresolution=f"{CHAN_WIDTH}Hz",
            nchannels=NCHAN,
            stokes="XX YY",
        )
    for i in range(len(SPW_FREQ)):
        sm.observe("src", f"SPW{i}", starttime="-0.25h", stoptime="0.25h")
    sm.setdata(spwid=list(range(len(SPW_FREQ))), fieldid=[0])
    sm.setvp(dovp=False)
    sm.predict(complist=cl)
    sm.close()
    return ms, casatools.version()


def _point(data, conjugate_model=False):
    values = {"cal.flux": 1.0}
    nuis = CalibrationNuisance(
        data, gains=False, offsets=False, sigmas=False, alpha=False
    )
    out = onp.asarray(
        data.model(
            PointSource(dra=POINT_MAS[0], ddec=POINT_MAS[1]), values, nuis
        )
    )
    return onp.conj(out) if conjugate_model else out


def test_read_ms_convention(synthetic_ms):
    """Counts, u against UVW and ν, and the DATA-column convention."""
    path, version = synthetic_ms
    from casatools import table

    data = ComplexVisData.read_ms(path)
    tb = table()
    tb.open(path)
    n_rows = tb.nrows()
    uvw = tb.getcol("UVW")
    ddid_col = tb.getcol("DATA_DESC_ID")
    tb.close()
    tb.open(f"{path}/SPECTRAL_WINDOW")
    chan_freq = [tb.getcell("CHAN_FREQ", i) for i in range(2)]
    tb.close()
    assert (
        len(data) == n_rows * NCHAN
    )  # nothing flagged, cross-correlations only
    assert data.ddid_ids == (0, 1)
    assert len(data.obs_ids) == 1
    # u, v from UVW and the channel frequency, per ddid
    for d in (0, 1):
        sub = data.select(ddids=[d])
        rows = onp.flatnonzero(ddid_col == d)
        chan = onp.asarray(sub.chan).reshape(-1)
        freq = onp.asarray(sub.freq)
        nrow = rows.size
        # samples are (channel, row) pairs, channel-major
        onp.testing.assert_allclose(
            onp.asarray(sub.u),
            uvw[0, rows[onp.tile(onp.arange(nrow), NCHAN)]] * freq / C_LIGHT,
            rtol=1e-12,
        )
        onp.testing.assert_allclose(
            onp.asarray(sub.v),
            uvw[1, rows[onp.tile(onp.arange(nrow), NCHAN)]] * freq / C_LIGHT,
            rtol=1e-12,
        )
        assert set(chan) == set(range(NCHAN))
        assert onp.allclose(onp.unique(freq), chan_freq[d], rtol=0, atol=1e-3)
    # the convention: the read data must equal the model, and its opposite not
    got = onp.asarray(data.vis)
    want = _point(data)
    err = onp.max(onp.abs(got - want))
    err_conj = onp.max(onp.abs(got - onp.conj(want)))
    print(
        f"casatools {version}: |data - model| = {err:.3e}, |data - model*| = {err_conj:.3e}"
    )
    assert err < 1e-6, (err, err_conj)
    assert err_conj > 1e-2
    # the opposite conjugation must fail
    wrong = ComplexVisData.read_ms(
        path, conjugate=not virgil.radio.MS_CONJUGATE
    )
    assert onp.max(onp.abs(onp.asarray(wrong.vis) - want)) > 1e-2


def test_defaults_come_from_the_data():
    data = _data()
    nuis = CalibrationNuisance(data)
    w, f = onp.asarray(data.weight), onp.asarray(data.freq)
    assert nuis.nu0_hz == pytest.approx(onp.sum(w * f) / onp.sum(w))
    q = onp.hypot(onp.asarray(data.u), onp.asarray(data.v))
    # default: three beams (2π b each) of the longest baseline of obs 1
    q_max = q[onp.asarray(data.obs) == 1].max()
    prior = nuis.priors()["cal.offset"]
    assert float(prior.high[0, 0]) == pytest.approx(3 * rad2mas / q_max)
    assert nuis.offset_bound_mas == pytest.approx(rad2mas / (2 * q.min()))
    custom = nuis.priors(offset=(-5.0, 5.0), alpha=(0.0, 3.0))
    assert float(custom["cal.offset"].high[0, 0]) == 5.0
    assert float(custom["cal.alpha"].high) == 3.0
    with pytest.raises(ValueError, match="unknown"):
        nuis.priors(bogus=(0, 1))


def test_model_and_term_in_float32():
    data = _data(n=400)
    nuis = CalibrationNuisance(data)
    values = _values(
        nuis,
        **{"cal.flux": 0.3, "cal.gain": [1.1], "cal.offset": [[4.0, -2.0]]},
    )
    scene = _image()
    ref = onp.asarray(data.model(scene, values, nuis))
    ref_res = onp.asarray(
        data.term(lambda v: scene, nuis, chunk_size=150)(values)
    )
    with jax.enable_x64(False):
        data32 = ComplexVisData.from_arrays(
            data.u,
            data.v,
            data.vis,
            data.weight,
            data.freq,
            ddid=data.ddid,
            obs=data.obs,
        )
        nuis32 = CalibrationNuisance(data32)
        values32 = {
            k: np.asarray(v, dtype=np.float32) for k, v in values.items()
        }
        scene32 = Image(
            onp.asarray(scene.log_brightness, dtype=onp.float32),
            PIXEL,
            ft="dft",
        )
        got = onp.asarray(data32.model(scene32, values32, nuis32))
        res = onp.asarray(
            data32.term(lambda v: scene32, nuis32, chunk_size=150)(values32)
        )
        assert got.dtype == onp.complex64 and res.dtype == onp.float32
    assert onp.linalg.norm(got - ref) / onp.linalg.norm(ref) < 1e-3
    assert onp.linalg.norm(res - ref_res) / onp.linalg.norm(ref_res) < 1e-2


def test_ddid_shared_by_two_observations():
    rng = onp.random.default_rng(4)
    n = 300
    obs = rng.integers(0, 2, n)
    data = ComplexVisData.from_arrays(
        rng.uniform(-1e6, 1e6, n),
        rng.uniform(-1e6, 1e6, n),
        onp.ones(n, complex),
        onp.ones(n),
        230e9,
        ddid=7,
        obs=obs + 20,
    )
    # one original ddid, two labels (one per observation)
    assert data.ddid_ids == (7, 7) and data.ddid_obs == (0, 1)
    assert data.obs_ids == (20, 21)
    nuis = CalibrationNuisance(data)
    assert nuis.priors()["cal.sigma"].batch_shape == (2,)
    assert len(data.select(ddids=[7])) == n
    assert len(data.select(ddids=[7], obs=[21])) == int(obs.sum())


def test_term_needs_a_function_of_the_values():
    data = _data(n=50)
    nuis = CalibrationNuisance(data)
    with pytest.raises(TypeError, match="function"):
        data.term(_image(), nuis)


class _FakeTable:
    """The columns of a CASA table query, as `read_ms` reads them."""

    def __init__(self, cols):
        self.cols = cols

    def getcol(self, name, start=0, nrow=-1, incr=1):
        x = self.cols[name]
        return x[..., ::incr]

    def colnames(self):
        return list(self.cols)

    def iscelldefined(self, name, row):
        return name in self.cols


def test_read_ddid_applies_the_casa_convention_without_casatools():
    """CASA's DATA is V* with u = +UVW[0] ν/c (pinned by test 9 with casatools).

    A fake table holds the DATA a point source gives under that rule; the
    reader must undo it, so the test runs in CI where casatools is missing.
    """
    from virgil.radio import _read_ddid

    rng = onp.random.default_rng(8)
    nrow, nchan = 40, 3
    freq = 230e9 + 1e8 * onp.arange(nchan)
    uvw = rng.uniform(-300, 300, (3, nrow))
    u = uvw[0][None] * freq[:, None] / C_LIGHT
    v = uvw[1][None] * freq[:, None] / C_LIGHT
    x, y = 300.0 * mas2rad, 100.0 * mas2rad
    true = onp.exp(-2j * onp.pi * (u * x + v * y))  # (nchan, nrow), virgil's V
    data = onp.stack([onp.conj(true), onp.conj(true)])  # CASA stores V*
    cols = {
        "DATA": data,
        "FLAG": onp.zeros(data.shape, bool),
        "FLAG_ROW": onp.zeros(nrow, bool),
        "UVW": uvw,
        "TIME": onp.arange(nrow, dtype=float),
        "ANTENNA1": onp.zeros(nrow, int),
        "ANTENNA2": onp.ones(nrow, int),
        "OBSERVATION_ID": onp.zeros(nrow, int),
        "SCAN_NUMBER": onp.full(nrow, 7),
        "WEIGHT": onp.full((2, nrow), 4.0),
    }
    part = _read_ddid(
        _FakeTable(cols),
        (0, 1),
        freq,
        1,
        "DATA",
        virgil.radio.MS_CONJUGATE,
        virgil.radio.MS_U_SIGN,
    )
    part["ddid"] = onp.zeros(part["u"].size, dtype=onp.int64)
    read = ComplexVisData.from_arrays(**part)
    nuis = CalibrationNuisance(
        read, gains=False, offsets=False, sigmas=False, alpha=False
    )
    model = onp.asarray(
        read.model(PointSource(dra=300.0, ddec=100.0), {"cal.flux": 1.0}, nuis)
    )
    onp.testing.assert_allclose(onp.asarray(read.vis), model, atol=1e-12)
    assert onp.all(onp.asarray(read.weight) == 8.0)


# ----------------------------------------------------------------------
# Diagnostics: dirty image, dirty beam, sigma_scale
# ----------------------------------------------------------------------


def _uv_data(n=4000, seed=0, vis=None, weight=None):
    rng = onp.random.default_rng(seed)
    u = rng.uniform(-3e6, 3e6, n)
    v = rng.uniform(-3e6, 3e6, n)
    if vis is None:
        vis = rng.normal(size=n) + 1j * rng.normal(size=n)
    if weight is None:
        weight = rng.uniform(1.0, 3.0, n)
    return ComplexVisData.from_arrays(u, v, vis, weight, 230e9)


@pytest.mark.parametrize("npix", [32, 31])
@pytest.mark.parametrize("weighting", ["natural", "uniform", "briggs"])
def test_dirty_image_matches_brute_force(npix, weighting):
    data = _uv_data(n=3000)
    s = 0.7
    img = data.dirty_image(npix, s, weighting=weighting)
    w = data._imaging_weights(
        weighting, 0.5, 1.0 / (npix * s * mas2rad), None, None
    )
    c = 0.5 * (npix - 1)
    x = (c - onp.arange(npix)) * s * mas2rad
    X, Y = onp.meshgrid(x, x, indexing="xy")
    u, v = onp.asarray(data.u), onp.asarray(data.v)
    vis = onp.asarray(data.vis)
    ph = 2 * onp.pi * (u[:, None, None] * X + v[:, None, None] * Y)
    ref = (
        w[:, None, None]
        * (
            vis.real[:, None, None] * onp.cos(ph)
            - vis.imag[:, None, None] * onp.sin(ph)
        )
    ).sum(0) / w.sum()
    assert onp.abs(img - ref).max() < 1e-10 * onp.abs(ref).max()


def test_dirty_image_point_source_position():
    data = _uv_data(n=3000)
    dra, ddec, flux = 6.0, -4.0, 2.5  # mas: East and South of the centre
    s, npix = 1.0, 33
    u, v = onp.asarray(data.u), onp.asarray(data.v)
    vis = flux * onp.exp(-2j * onp.pi * mas2rad * (dra * u + ddec * v))
    d = ComplexVisData.from_arrays(u, v, vis, data.weight, 230e9)
    img = d.dirty_image(npix, s)
    row, col = onp.unravel_index(onp.argmax(img), img.shape)
    c = 0.5 * (npix - 1)
    assert col == c - dra / s  # East is to the left
    assert row == c - ddec / s  # South is down
    assert img[row, col] == pytest.approx(flux, rel=0.05)
    assert img[row, col] <= flux * (1 + 1e-9)


def test_dirty_beam_peak_and_briggs_limits():
    data = _uv_data(n=3000)
    for wt in ("natural", "uniform", "briggs"):
        b = data.dirty_beam(33, 1.0, weighting=wt)
        assert b[16, 16] == pytest.approx(1.0, abs=1e-10)
        assert b.max() <= 1.0 + 1e-10
    rel = lambda a, b: onp.abs(a - b).max()  # noqa: E731
    kw = dict(npix=33, pixel_scale_mas=1.0, weighting="briggs")
    nat = data.dirty_beam(33, 1.0, weighting="natural")
    uni = data.dirty_beam(33, 1.0, weighting="uniform")
    hi = data.dirty_beam(robust=2.0, **kw)
    lo = data.dirty_beam(robust=-2.0, **kw)
    mid = data.dirty_beam(robust=0.5, **kw)
    assert rel(hi, nat) < 0.5 * rel(hi, uni)
    assert rel(lo, uni) < 0.5 * rel(lo, nat)
    assert rel(hi, nat) < rel(mid, nat) < rel(lo, nat)


def test_dirty_even_beam_peak_at_centre_is_one():
    data = _uv_data(n=500)
    b = data.dirty_beam(32, 0.2)  # small pixels: the sample is near the peak
    assert 0.99 < b.max() <= 1.0


def test_dirty_residual_of_true_model_is_zero():
    scene = _image()
    data0 = _data(n=1500, noise=0.0)
    nuis = CalibrationNuisance(data0, ref_obs=0)
    values = _values(nuis, **{"cal.flux": 1.3})
    model = onp.asarray(data0.model(scene, values, nuis))
    data = ComplexVisData.from_arrays(
        data0.u,
        data0.v,
        model,
        data0.weight,
        data0.freq,
        ddid=data0.ddid,
        obs=data0.obs,
    )
    resid = data.dirty_image(
        32, 2.0, scene=scene, values=values, nuisance=nuis
    )
    img = data.dirty_image(32, 2.0)
    assert onp.abs(resid).max() < 1e-10 * onp.abs(img).max()
    # a function of the values works as well, and use_sigma needs a fit
    resid2 = data.dirty_image(
        32,
        2.0,
        scene=lambda v: scene,
        values=values,
        nuisance=nuis,
        use_sigma=True,
    )
    assert onp.abs(resid2).max() < 1e-10 * onp.abs(img).max()
    with pytest.raises(ValueError):
        data.dirty_image(32, 2.0, scene=scene)


def _sigma_data(shuffle=False, seed=3, period=3000.0):
    rng = onp.random.default_rng(seed)
    nant, nint, nchan = 6, 40, 3
    scales = [1.0, 1.5, 2.0]  # one per ddid
    cols = {k: [] for k in "u v vis weight freq ddid time a1 a2 scan".split()}
    for d, s in enumerate(scales):
        for a1 in range(nant):
            for a2 in range(a1 + 1, nant):
                for ch in range(nchan):
                    for scan in (1, 2):
                        t = onp.arange(nint, dtype=float) * 6.0 + 1e4 * scan
                        w = rng.uniform(1.0, 4.0, nint) * 100.0
                        sky = (1.0 + 0.5 * onp.sin(t / period)) * onp.exp(
                            1j * 0.001 * (t - t[0])
                        )  # slowly varying
                        noise = (
                            (
                                rng.normal(size=nint)
                                + 1j * rng.normal(size=nint)
                            )
                            * s
                            / onp.sqrt(w)
                        )
                        cols["vis"].append(sky + noise)
                        cols["weight"].append(w)
                        cols["time"].append(t)
                        cols["u"].append(
                            onp.full(nint, 1e5 * (a1 + 1) + 7.0 * a2)
                        )
                        cols["v"].append(onp.full(nint, 3e4 * (a2 + 1)))
                        cols["freq"].append(onp.full(nint, 230e9 + 1e7 * ch))
                        cols["ddid"].append(onp.full(nint, d))
                        cols["a1"].append(onp.full(nint, a1))
                        cols["a2"].append(onp.full(nint, a2))
                        cols["scan"].append(onp.full(nint, scan))
    c = {k: onp.concatenate(x) for k, x in cols.items()}
    if shuffle:  # the time column no longer follows the data
        c["time"] = c["time"][
            onp.random.default_rng(9).permutation(c["time"].size)
        ]
    perm = onp.random.default_rng(1).permutation(
        c["u"].size
    )  # row order is irrelevant
    c = {k: x[perm] for k, x in c.items()}
    return scales, ComplexVisData.from_arrays(
        c["u"],
        c["v"],
        c["vis"],
        c["weight"],
        c["freq"],
        ddid=c["ddid"],
        time=c["time"],
        ant1=c["a1"],
        ant2=c["a2"],
        scan=c["scan"],
    )


def test_sigma_scale_recovers_known_scales():
    scales, data = _sigma_data()
    res = data.sigma_scale()
    assert res.labels == (0, 1, 2)
    for k, s in enumerate(scales):
        # the slow sky inflates by < 1e-3 here, far below the standard error
        assert abs(res.s[k] - s) < 3 * res.s_se[k]
        assert abs(res.s_re[k] - s) < 3 * res.s_re_se[k]
        assert abs(res.s_im[k] - s) < 3 * res.s_im_se[k]
        assert res.n_pairs[k] > 1000
    assert data.sigma_scale(by=None).labels == (None,)
    assert data.sigma_scale(by="obs").s.shape == (1,)
    # preserved by selection
    sub = data.select(ddids=[1]).sigma_scale()
    assert sub.n_pairs[1] == res.n_pairs[1] and sub.n_pairs[0] == 0


def test_sigma_scale_detects_scrambled_time():
    # a steeply varying sky makes adjacency matter; with the time column
    # scrambled the pairs are not adjacent, the estimate is far off and the
    # lag ratio drops to 1
    _, data = _sigma_data(period=60.0)
    _, bad = _sigma_data(shuffle=True, period=60.0)
    good_res, bad_res = data.sigma_scale(), bad.sigma_scale()
    assert onp.all(bad_res.s > 1.3 * good_res.s)
    assert onp.all(onp.abs(bad_res.lag_ratio - 1.0) < 0.2)
    assert onp.all(good_res.lag_ratio > 3.0)


def test_sigma_scale_needs_fields():
    data = _uv_data(n=50)
    with pytest.raises(ValueError, match="time, ant1, ant2, scan"):
        data.sigma_scale()
    with pytest.raises(ValueError, match="by must"):
        _sigma_data()[1].sigma_scale(by="bogus")


# -- the data stay jit arguments, not compile-time constants -----------------


def _constant_bytes(lowered):
    """Bytes of dense constants embedded in a lowered module's text."""
    import re

    return sum(
        len(m) for m in re.findall(r"dense<[^>]{200,}>", lowered.as_text())
    )


def _big_term_problem(n=200_000):
    rng = onp.random.default_rng(3)
    ddid = rng.integers(0, 3, n)
    data = ComplexVisData.from_arrays(
        rng.uniform(-4e6, 4e6, n),
        rng.uniform(-4e6, 4e6, n),
        rng.normal(size=n) + 1j * rng.normal(size=n),
        rng.uniform(0.5, 2.0, n) * 1e4,
        onp.array([230e9, 238e9, 245e9])[ddid],
        ddid=ddid,
        obs=onp.array([0, 0, 1])[ddid],
    )
    nuis = CalibrationNuisance(data, blur_fwhm_mas=50.0)
    values = _values(nuis)

    def scene(v):
        return Image(v["log_brightness"], PIXEL, ft="dft")

    values["log_brightness"] = np.zeros((8, 8))
    return data, nuis, scene, values


def test_term_data_are_arguments_not_constants():
    data, nuis, scene, values = _big_term_problem()
    term = data.term(scene, nuis, chunk_size=100_000)

    def loss(t, v):
        return jax.value_and_grad(
            lambda w: 0.5 * np.sum(t(w) ** 2) + t.log_norm(w)
        )(v)

    as_arg = eqx.filter_jit(loss).lower(term, values)
    closed = jax.jit(lambda v: loss(term, v)).lower(values)
    assert _constant_bytes(as_arg) < 1e6
    assert _constant_bytes(closed) > 1e7  # the documented pitfall


@pytest.mark.filterwarnings("ignore:fit.*did not converge")
def test_fit_compiles_once_across_masks_and_weights(monkeypatch):
    import numpyro.distributions as dist

    from virgil.imaging import MaxEntropy
    from virgil.radio import _RadioTerm

    data, nuis, scene, values = _big_term_problem(n=4000)
    fold, _ = dartboard_folds(data, k=3)
    priors = nuis.priors()
    priors["log_brightness"] = dist.Normal(0.0, 3.0)

    def model(**v):
        return scene(v)

    traced = []
    original = _RadioTerm.__call__

    def counting(self, v):
        if isinstance(v["cal.flux"], jax.core.Tracer):
            traced.append(1)
        return original(self, v)

    monkeypatch.setattr(_RadioTerm, "__call__", counting)

    def run(i, lam):
        term = data.term(scene, nuis, mask=fold != i, chunk_size=2000)
        n0 = len(traced)
        result = fit(
            model,
            priors,
            data=(),
            likelihoods=[term],
            regularizers=[MaxEntropy(lam)],
            init=dict(values),
            method="lbfgs",
            max_steps=2,
        )
        return len(traced) - n0, term, result

    n_first, _, first = run(0, 1e3)
    assert n_first > 0
    n_second, term1, second = run(1, 2e3)
    assert n_second == 0  # same shapes: no retrace, so no recompile
    # ... and fold 1's mask reached the computation: the reported chi2 is that
    # of an eager evaluation of fold 1's term at the fitted values
    eager = float(np.sum(term1(second.values) ** 2))
    assert second.info["chi2"][-1] == pytest.approx(eager, rel=1e-8)
    assert second.info["chi2"][-1] != pytest.approx(
        first.info["chi2"][-1], rel=1e-3
    )
