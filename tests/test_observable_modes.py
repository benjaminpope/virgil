"""Linear nuisance modes on DifferentialPhase and FluxSpectrum, and corrflux GainModes."""

import itertools

import equinox as eqx
import jax
import jax.numpy as np
import numpy as onp
import pytest
from scipy.linalg import block_diag
from scipy.stats import multivariate_normal

from tests._shared import PAIRS, TRIANGLES  # noqa: E402
from virgil._precision import cast_tree
from virgil.epochs import _Surface, marginal_loglike
from virgil.gains import gain_modes
from virgil.imaging import error_scale
from virgil.likelihood import (
    flux_scale_posterior,
    model_loglike,
)
from virgil.models import PointSource, System
from virgil.observables import (
    DifferentialPhase,
    FluxSpectrum,
)
from virgil.oidata import OIData, cp_indices
from virgil.oifits import build_hdulist, read_oifits
from virgil.spectra import GaussianLine, PowerLaw, Sum

STATIONS = onp.array([[0.0, 0.0], [32.0, 2.0], [14.0, 26.0], [-11.0, 18.0]])
WAVES = onp.linspace(2.154e-6, 2.178e-6, 8)
LINE = (2.162e-6, 2.170e-6)
CONT_RANGES = [(2.154e-6, 2.160e-6), (2.172e-6, 2.178e-6)]
SCALE_PRIOR = (3.7, 0.5)


def _baselines(pairs):
    delta = STATIONS[pairs[:, 1] - 1] - STATIONS[pairs[:, 0] - 1]
    return delta[:, 0], delta[:, 1]


def _scene(dra=0.5, ddec=0.2, amplitude=0.5):
    line = GaussianLine(amplitude=amplitude, line_wavel=2.166e-6, fwhm=1.2e-8)
    flux = Sum(cont=PowerLaw(0.1, 0.0, wavel0=2.166e-6), brg=line)
    return System(
        star=PointSource(), blob=PointSource(dra=dra, ddec=ddec, flux=flux)
    )


def _tables(
    scene=None,
    scale=3.7,
    amptyp="absolute",
    mjd=60000.0,
    pairs=PAIRS,
    wavel=WAVES,
):
    scene = _scene() if scene is None else scene
    u, v = _baselines(pairs)
    cvis = onp.asarray(scene.model(u[:, None], v[:, None], wavel[None, :]))
    flux = scale * onp.asarray(scene.total_spectrum(wavel))
    amp = onp.abs(cvis) * (flux if amptyp == "correlated flux" else 1.0)
    out = {
        "info": {"TARGET": "X", "INSTRUME": "TEST", "MJD": mjd},
        "OI_WAVELENGTH": {"EFF_WAVE": wavel, "EFF_BAND": 1e-9},
        "OI_VIS2": {
            "VIS2DATA": onp.abs(cvis) ** 2,
            "VIS2ERR": onp.full(cvis.shape, 0.22),
            "UCOORD": u,
            "VCOORD": v,
            "STA_INDEX": pairs,
        },
        "OI_VIS": {
            "VISAMP": amp,
            "VISAMPERR": onp.full(cvis.shape, 0.25),
            "VISPHI": onp.rad2deg(onp.angle(cvis)),
            "VISPHIERR": onp.full(cvis.shape, 18.0),
            "UCOORD": u,
            "VCOORD": v,
            "STA_INDEX": pairs,
            "AMPTYP": amptyp,
            "PHITYP": "differential",
        },
        "OI_FLUX": {
            "FLUXDATA": onp.tile(flux, (4, 1)),
            "FLUXERR": onp.full((4, wavel.size), 0.35),
            "STA_INDEX": onp.arange(1, 5),
        },
    }
    if pairs.shape[0] == PAIRS.shape[0]:
        i1, i2, i3 = cp_indices(pairs, TRIANGLES)
        cp = onp.angle(cvis[i1] * cvis[i2] / cvis[i3])
        t3amp = onp.abs(cvis[i1] * cvis[i2] * cvis[i3])
        u1, v1 = _baselines(TRIANGLES[:, [0, 1]])
        u2, v2 = _baselines(TRIANGLES[:, [1, 2]])
        out["OI_T3"] = {
            "T3AMP": t3amp,
            "T3AMPERR": onp.full(cp.shape, 0.22),
            "T3PHI": onp.rad2deg(cp),
            "T3PHIERR": onp.full(cp.shape, 10.0),
            "U1COORD": u1,
            "V1COORD": v1,
            "U2COORD": u2,
            "V2COORD": v2,
            "STA_INDEX": TRIANGLES,
        }
    return out


def _two_epoch_oidata(extras, amptyp="absolute", ragged=False):
    t0 = _tables(amptyp=amptyp, mjd=60000.0)
    pairs_1 = PAIRS[:-1] if ragged else PAIRS
    wavel_1 = WAVES[:-1] if ragged else WAVES
    t1 = _tables(amptyp=amptyp, mjd=60005.0, pairs=pairs_1, wavel=wavel_1)
    if ragged:
        del t0["OI_T3"]
    return OIData([build_hdulist(t0), build_hdulist(t1)], extras=extras)


def _zero_size_leaves(tree):
    return [
        leaf
        for leaf in jax.tree_util.tree_leaves(tree)
        if hasattr(leaf, "size") and leaf.size == 0
    ]


def _dense_diffphase_cov(block):
    """Exact projected covariance of a DifferentialPhase block in float64."""
    errors = onp.asarray(block.errors, float)
    var = onp.where(block.chan, errors[block.grid] ** 2, 0.0)
    q = onp.asarray(block.q, float)
    w = onp.asarray(block.w, float)
    a = onp.einsum("frc,fsc,fbc->fbrs", w, w, var)
    cov_f = onp.einsum("fkb,flb,fbrs->fkrls", q, q, a)
    n_f, k, r = block.valid.shape
    cov0_full = block_diag(*cov_f.reshape(n_f, k * r, k * r))
    keep = onp.asarray(block.keep, int)
    cov = cov0_full[onp.ix_(keep, keep)]

    if block.prior_width is not None:
        widths = onp.asarray(block.prior_width, float)
        u_frames = []
        for f in range(n_f):
            qf = q[f]
            bf = onp.asarray(block.basis[f], float) * widths
            uf = onp.kron(qf, bf)
            vf = onp.asarray(block.valid[f], bool).reshape(k * r)
            u_frames.append(onp.where(vf[:, None], uf, 0.0))
        u_cont = block_diag(*u_frames)[keep]
        cov = cov + u_cont @ u_cont.T

    if block.modes is not None:
        m_raw = onp.asarray(block.modes, float)
        m_grid = onp.where(block.chan, m_raw[:, block.grid], 0.0)
        m_proj = onp.einsum("fkb,mfbc,frc->fkrm", q, m_grid, w)
        u_modes = m_proj.reshape(n_f * k * r, m_raw.shape[0])[keep]
        cov = cov + u_modes @ u_modes.T
    return cov


def _dense_flux_cov(block, pred):
    """Exact covariance of an uncoupled FluxSpectrum block in float64."""
    pred = onp.asarray(pred, float)
    errs = onp.asarray(block.errors, float)
    cov = onp.diag(errs**2)
    if block.widths is not None:
        widths = onp.asarray(block.widths, float)
        poly = onp.asarray(block.poly, float)
        for members_g in block.members:
            idx = members_g[members_g < errs.size]
            u_g = onp.zeros((errs.size, widths.size))
            u_g[idx] = pred[idx, None] * poly[idx] * widths
            cov = cov + u_g @ u_g.T
    if block.modes is not None:
        u_m = pred[:, None] * onp.asarray(block.modes, float).T
        cov = cov + u_m @ u_m.T
    return cov


# === API INVARIANTS: PARTIAL UPDATES, KINDS GUARDS, AND CHROMATIC SELECT ===


def test_partial_updates_unset_clearing_and_multi_flux_kinds_guard():
    base_vp = _two_epoch_oidata(("visphi",)).select(observables=("visphi",))
    m_vp = onp.zeros((1, base_vp.u.size))
    m_vp[0, onp.asarray(base_vp.frame) == 0] = 0.06

    d_vp = base_vp.with_continuum(
        continuum=CONT_RANGES,
        lines=[LINE],
        order=1,
        prior_width=(0.2, 0.1),
        modes=m_vp,
    )
    # Omitted arguments preserve existing settings; explicit None clears them.
    d_ord0 = d_vp.with_continuum(order=0, prior_width=0.2)
    b_ord0 = d_ord0.extras[0]
    assert b_ord0.continuum == tuple(CONT_RANGES)
    assert b_ord0.lines == (LINE,)
    assert b_ord0.order == 0 and b_ord0.modes is not None

    d_proj = d_vp.with_continuum(prior_width=None)
    assert d_proj.extras[0].prior_width is None
    assert d_proj.extras[0].continuum == tuple(CONT_RANGES)
    assert d_proj.extras[0].modes is not None

    d_nomode = d_vp.with_continuum(modes=None)
    assert d_nomode.extras[0].modes is None
    assert d_nomode.extras[0].prior_width is not None

    d_allchan = d_proj.with_continuum(continuum=None, lines=None)
    assert d_allchan.extras[0].continuum is None
    assert d_allchan.extras[0].lines is None

    # Multiple FluxSpectrum blocks ("flux" + "corrflux") require explicit kinds=
    # when modes is supplied, and flux_scale_posterior skips coupled corrflux.
    both_raw = _two_epoch_oidata(("flux", "visamp"), amptyp="correlated flux")
    fb, cfb = both_raw.extras
    m_f = onp.full((1, fb.values.size), 0.03)
    m_cf = onp.full((1, cfb.values.size), 0.04)

    with pytest.raises(ValueError, match="kinds='flux' or kinds='corrflux'"):
        both_raw.with_flux_scale(scale=SCALE_PRIOR, modes=m_f)

    both = (
        both_raw.with_flux_scale(scale=SCALE_PRIOR)
        .with_flux_scale(modes=m_f, kinds="flux")
        .with_flux_scale(per="frame", modes=m_cf, kinds="corrflux")
        .with_gains(telescope=0.03)
    )
    assert both.extras[0].modes is not None
    assert both.extras[1].modes is not None

    post = flux_scale_posterior(_scene(), both)
    assert list(post) == ["flux"]
    assert list(flux_scale_posterior(_scene(), both, kinds="flux")) == ["flux"]
    with pytest.raises(NotImplementedError, match="corrflux"):
        flux_scale_posterior(_scene(), both, kinds="corrflux")
    with pytest.raises(ValueError, match="Unknown spectrum kind"):
        flux_scale_posterior(_scene(), both, kinds="visphi")

    # Clearing modes=None and scale=None on corrflux updates GainModes cleanly.
    cleared_m = both.with_flux_scale(modes=None, kinds="corrflux")
    assert cleared_m.extras[1].modes is None
    assert cleared_m.extras[1].scale is not None
    cleared_all = cleared_m.with_flux_scale(scale=None, kinds="corrflux")
    assert cleared_all.extras[1].scale is None
    assert "_flux" not in cleared_all.gains.groups


def test_with_corrflux_chromatic_selection_and_direct_gain_modes():
    raw = _two_epoch_oidata(("visamp",), amptyp="correlated flux").select(
        observables=("vis", "corrflux")
    )
    rng = onp.random.default_rng(141)
    m_u = rng.normal(0.0, 0.02, (2, raw.u.size))
    m_cf = rng.normal(0.0, 0.03, (2, raw.extras[0].values.size))
    asym_ranges = [(2.160e-6, 2.178e-6)]
    other = _scene(dra=0.61, ddec=0.15)

    with jax.enable_x64(True):
        r64 = cast_tree(raw, "float64")
        o64 = cast_tree(other, "float64")

        # 1. with_flux_scale and with_gains(..., chromatic=0.02) commute.
        d_fg = r64.with_flux_scale(
            scale=SCALE_PRIOR,
            per="frame",
            poly_order=1,
            modes=m_cf,
        ).with_gains(telescope=0.03, baseline=0.02, chromatic=0.02, modes=m_u)
        d_gf = r64.with_gains(
            telescope=0.03, baseline=0.02, chromatic=0.02, modes=m_u
        ).with_flux_scale(
            scale=SCALE_PRIOR,
            per="frame",
            poly_order=1,
            modes=m_cf,
        )
        assert float(model_loglike(o64, d_fg)) == pytest.approx(
            float(model_loglike(o64, d_gf)), abs=1e-12
        )

        # 2. Asymmetric wavelength select + observable select commute exactly
        # and match VIS2-only data (corrflux never alters VIS2 chromatic modes).
        vis_one_step = d_fg.select(ranges=asym_ranges, observables="vis")
        vis_two_step = d_fg.select(ranges=asym_ranges).select(
            observables="vis"
        )
        vis_pure = (
            r64.select(observables="vis")
            .with_gains(
                telescope=0.03, baseline=0.02, chromatic=0.02, modes=m_u
            )
            .select(ranges=asym_ranges)
        )
        ll_one = float(model_loglike(o64, vis_one_step))
        assert abs(ll_one - float(model_loglike(o64, vis_two_step))) < 1e-12
        assert abs(ll_one - float(model_loglike(o64, vis_pure))) < 1e-12
        assert "_flux" not in vis_two_step.gains.groups

        # 3. Direct gain_modes() attached via tree_at refreshes _flux across
        # both with_flux_scale and select(ranges=...).
        direct = eqx.tree_at(
            lambda d: d.gains,
            r64,
            gain_modes(
                r64, telescope=0.03, baseline=0.02, chromatic=0.02, modes=m_u
            ),
            is_leaf=lambda x: x is None,
        ).with_flux_scale(
            scale=SCALE_PRIOR, per="frame", poly_order=1, modes=m_cf
        )
        assert float(
            model_loglike(o64, direct.select(ranges=asym_ranges))
        ) == pytest.approx(
            float(model_loglike(o64, d_fg.select(ranges=asym_ranges))),
            abs=1e-12,
        )

    # 4. Dropping corrflux or clearing scale/modes on _flux-only GainModes
    # clears gains back to None.
    flux_only = raw.with_flux_scale(scale=SCALE_PRIOR, modes=m_cf).with_gains()
    assert flux_only.gains.groups == ("_flux",)
    assert flux_only.select(observables="vis").gains is None
    assert flux_only.with_flux_scale(scale=None, modes=None).gains is None


# === GATE 1a: DifferentialPhase(modes=...) ===


@pytest.mark.parametrize("prior_width", [None, (0.12, 0.08)])
@pytest.mark.parametrize("closure_free", [True, False])
@pytest.mark.parametrize("ragged", [False, True])
def test_diffphase_modes_dense_mvn_parity(prior_width, closure_free, ragged):
    if closure_free and ragged:
        pytest.skip("ragged fixture omits OI_T3")
    base = _two_epoch_oidata(("visphi",), ragged=ragged)
    if not closure_free and not ragged:
        base = base.select(observables=("visphi",))
        (b0,) = base.extras
        b0 = DifferentialPhase.build(
            b0.values,
            b0.errors,
            b0.sample,
            b0.wavel,
            b0.frame,
            b0.stations,
            b0.row,
            closure_free=False,
        )
        base = eqx.tree_at(lambda d: d.extras, base, (b0,))
    else:
        base = base.select(observables=("visphi",))

    rng = onp.random.default_rng(31)
    n_u = base.u.size
    frames = onp.asarray(base.frame)
    # Mode 0: local to frame 0; Mode 1: local to frame 1; Mode 2: spanning both.
    modes = onp.zeros((3, n_u))
    modes[0, frames == 0] = rng.normal(0.0, 0.08, int((frames == 0).sum()))
    modes[1, frames == 1] = rng.normal(0.0, 0.07, int((frames == 1).sum()))
    modes[2] = rng.normal(0.0, 0.05, n_u)

    noisy = base.with_continuum(
        lines=[LINE], prior_width=prior_width, modes=modes
    ).with_model(_scene(), key=jax.random.PRNGKey(7))
    assert _zero_size_leaves(noisy.extras[0]) == []

    other = _scene(dra=0.62, ddec=0.14)
    # Check float32 tolerance (< 5e-5).
    ll32 = float(model_loglike(other, noisy))

    with jax.enable_x64(True):
        d64 = cast_tree(noisy, "float64")
        o64 = cast_tree(other, "float64")
        ll64 = float(model_loglike(o64, d64))
        (block,) = d64.extras
        cvis = o64.model(d64.u, d64.v, d64.wavel)
        pred = onp.asarray(block.predict(o64, cvis), float)
        y_data = onp.asarray(block.data(), float)
        cov = _dense_diffphase_cov(block)
        expected = float(
            multivariate_normal.logpdf(y_data, mean=pred, cov=cov)
        )

    assert abs(ll64 - expected) < 1e-10
    assert abs(ll32 - expected) < 5e-5


def test_diffphase_modes_tangent_space_no_unwrap_and_guards():
    base = _two_epoch_oidata(("visphi",)).select(observables=("visphi",))
    n_u = base.u.size
    frames = onp.asarray(base.frame)

    # Large tangent-space step (> pi) must NOT be phase-wrapped/unwrapped:
    # scaling modes by alpha scales U U^T by alpha^2 at large amplitude.
    big_mode = onp.zeros((1, n_u))
    big_mode[0, :4] = [0.0, 4.5, -4.2, 3.9]
    with jax.enable_x64(True):
        d_big = cast_tree(
            base.with_continuum(lines=[LINE], modes=big_mode), "float64"
        )
        o64 = cast_tree(_scene(dra=0.58), "float64")
        (block,) = d_big.extras
        cvis = o64.model(d_big.u, d_big.v, d_big.wavel)
        pred = onp.asarray(block.predict(o64, cvis), float)
        y_data = onp.asarray(block.data(), float)
        cov = _dense_diffphase_cov(block)
        expected = float(
            multivariate_normal.logpdf(y_data, mean=pred, cov=cov)
        )
        assert abs(float(model_loglike(o64, d_big)) - expected) < 1e-10

    # Zero-size leaf absence for local-only and spanning-only modes.
    m_local = onp.zeros((1, n_u))
    m_local[0, frames == 0] = 0.05
    m_span = onp.full((1, n_u), 0.04)
    d_loc = base.with_continuum(lines=[LINE], modes=m_local)
    d_span = base.with_continuum(lines=[LINE], modes=m_span)
    assert _zero_size_leaves(d_loc.extras[0]) == []
    assert _zero_size_leaves(d_span.extras[0]) == []

    # Subset / select slices modes cleanly; with_continuum preserves modes.
    sub = d_loc.select(ranges=[(2.155e-6, 2.176e-6)])
    assert sub.extras[0].modes.shape == (1, sub.extras[0].values.size)
    kept = d_loc.with_continuum(prior_width=(0.2, 0.1))
    assert kept.extras[0].modes is not None

    # imaging.error_scale(..., by_observable=True) rejects DifferentialPhase modes.
    from virgil.fields import GaussianField
    from virgil.models import Image

    gp_field = GaussianField(
        onp.zeros((8, 8)), 1.0, 2.0, mean=onp.ones((8, 8)) / 64.0
    )
    gp_model = System(star=PointSource(), env=Image(gp_field, 1.0, flux=0.4))
    full = _two_epoch_oidata(("visphi",)).with_continuum(
        lines=[LINE], modes=onp.full((1, n_u), 0.05)
    )
    with pytest.raises(ValueError, match="nuisance modes"):
        error_scale(gp_model, full, by_observable=True)
    # ... and so does the scalar path, here and for a finite prior_width.
    with pytest.raises(ValueError, match="nuisance modes"):
        error_scale(gp_model, full)
    wide = _two_epoch_oidata(("visphi",)).with_continuum(
        lines=[LINE], prior_width=(0.3, 0.5)
    )
    with pytest.raises(ValueError, match="finite prior_width"):
        error_scale(gp_model, wide)

    # split_by_epoch succeeds for frame-local modes and refuses spanning modes.
    with jax.enable_x64(True):
        d_loc64 = cast_tree(d_loc, "float64")
        o64 = cast_tree(_scene(dra=0.55), "float64")
        parts = [cast_tree(p, "float64") for p in d_loc.split_by_epoch()]
        assert sum(
            float(model_loglike(o64, p)) for p in parts
        ) == pytest.approx(float(model_loglike(o64, d_loc64)), abs=1e-11)
    with pytest.raises(ValueError, match="spans several epochs"):
        d_span.split_by_epoch()

    # simulated wraps into (-pi, pi] and draws mode variance.
    draws = onp.stack(
        [
            onp.asarray(
                d_span.with_model(
                    _scene(), key=jax.random.PRNGKey(100 + i), noise_scale=0.0
                )
                .extras[0]
                .values
            )
            for i in range(64)
        ]
    )
    assert onp.all((draws > -onp.pi) & (draws <= onp.pi))
    assert float(onp.std(draws[:, 0])) > 0.01


# === GATE 1b: FluxSpectrum(modes=...) ===


@pytest.mark.parametrize("kind", ["flux", "corrflux"])
@pytest.mark.parametrize("per", ["frame", "station", "dataset"])
def test_flux_spectrum_modes_dense_mvn_and_posterior(kind, per):
    amptyp = "correlated flux" if kind == "corrflux" else "absolute"
    extra_name = "visamp" if kind == "corrflux" else "flux"
    obs_keep = "corrflux" if kind == "corrflux" else "flux"
    base = _two_epoch_oidata((extra_name,), amptyp=amptyp).select(
        observables=(obs_keep,)
    )
    (b0,) = base.extras
    n_b = b0.values.size
    rng = onp.random.default_rng(42)
    modes = onp.zeros((3, n_b))
    # Mode 0: local to frame 0; Mode 1: local to frame 1; Mode 2: spanning all.
    modes[0, b0.frame == 0] = rng.normal(0.0, 0.06, int((b0.frame == 0).sum()))
    modes[1, b0.frame == 1] = rng.normal(0.0, 0.05, int((b0.frame == 1).sum()))
    modes[2] = rng.normal(0.0, 0.04, n_b)

    data = base.with_flux_scale(
        scale=SCALE_PRIOR,
        per=per,
        poly_order=1,
        poly_width=0.08,
        modes=modes,
    ).with_model(_scene(), key=jax.random.PRNGKey(19))
    assert _zero_size_leaves(data.extras[0]) == []

    other = _scene(dra=0.6, ddec=0.12)
    ll32 = float(model_loglike(other, data))

    with jax.enable_x64(True):
        d64 = cast_tree(data, "float64")
        o64 = cast_tree(other, "float64")
        ll64 = float(model_loglike(o64, d64))
        (block,) = d64.extras
        cvis = (
            o64.model(d64.u, d64.v, d64.wavel) if kind == "corrflux" else None
        )
        pred = onp.asarray(block.predict(o64, cvis), float)
        y_data = onp.asarray(block.data(), float)
        cov = _dense_flux_cov(block, pred)
        expected = float(
            multivariate_normal.logpdf(y_data, mean=pred, cov=cov)
        )
        assert abs(ll64 - expected) < 1e-10
        assert abs(ll32 - expected) < 5e-5

        # Exact Gaussian Schur-complement conditioning for posterior().
        mean_got, cov_got = block.posterior(pred, y_data, block.errors)
        post_dict = flux_scale_posterior(o64, d64)[kind]
        onp.testing.assert_allclose(post_dict["mean"], mean_got, atol=1e-12)

        n_group, p = block.members.shape[0], block.poly.shape[1]
        mu = onp.asarray(block.mu, float)
        widths = onp.asarray(block.widths, float)
        # Parameter vector theta_g has prior mean [mu_g, 0, ...] and
        # diagonal std mu_g * widths, with y = A theta + pred * (M^T z) + eps.
        prior_mean = onp.zeros((n_group, p))
        prior_mean[:, 0] = mu
        prior_std = mu[:, None] * widths[None, :]
        a_mat = onp.zeros((n_b, n_group * p))
        t_base = pred / mu[block.group]
        for g in range(n_group):
            idx = block.members[g][block.members[g] < n_b]
            a_mat[idx, g * p : (g + 1) * p] = (
                t_base[idx, None] * block.poly[idx]
            )
        sigma_theta = onp.diag(prior_std.reshape(-1) ** 2)
        sigma_wy = sigma_theta @ a_mat.T
        # cov already equals A Sigma_theta A^T + U_modes U_modes^T + D!
        theta_mean = prior_mean.reshape(-1) + sigma_wy @ onp.linalg.solve(
            cov, y_data - pred
        )
        theta_cov = sigma_theta - sigma_wy @ onp.linalg.solve(cov, sigma_wy.T)
        onp.testing.assert_allclose(
            onp.asarray(mean_got),
            theta_mean.reshape(n_group, p),
            rtol=1e-11,
            atol=1e-11,
        )
        for g in range(n_group):
            sl = slice(g * p, (g + 1) * p)
            onp.testing.assert_allclose(
                onp.asarray(cov_got[g]),
                theta_cov[sl, sl],
                rtol=1e-11,
                atol=1e-11,
            )


def test_flux_spectrum_nflux_refusal_and_epoch_split():
    hdul = build_hdulist(_tables())
    nflux_data = OIData(read_oifits(hdul, extras=("nflux",)))
    (nb,) = nflux_data.extras
    with pytest.raises(ValueError, match="nflux"):
        nflux_data.with_flux_scale(modes=onp.ones((1, nb.values.size)))
    with pytest.raises(ValueError, match="nflux"):
        FluxSpectrum.build(
            "nflux",
            nb.values,
            nb.errors,
            nb.wavel,
            nb.row,
            nb.frame,
            nb.station,
            modes=onp.ones((1, nb.values.size)),
        )

    base = _two_epoch_oidata(("flux",)).select(observables=("flux",))
    (fb,) = base.extras
    m_loc = onp.zeros((2, fb.values.size))
    m_loc[0, fb.frame == 0] = 0.05
    m_loc[1, fb.frame == 1] = -0.04
    d_ok = base.with_flux_scale(scale=SCALE_PRIOR, per="frame", modes=m_loc)
    with jax.enable_x64(True):
        d64 = cast_tree(d_ok, "float64")
        o64 = cast_tree(_scene(dra=0.54), "float64")
        parts = [cast_tree(p, "float64") for p in d_ok.split_by_epoch()]
        assert sum(
            float(model_loglike(o64, p)) for p in parts
        ) == pytest.approx(float(model_loglike(o64, d64)), abs=1e-11)

    d_bad_mode = base.with_flux_scale(
        scale=SCALE_PRIOR,
        per="frame",
        modes=onp.full((1, fb.values.size), 0.03),
    )
    with pytest.raises(ValueError, match="spans several epochs"):
        d_bad_mode.split_by_epoch()

    d_bad_per = base.with_flux_scale(scale=SCALE_PRIOR, per="dataset")
    with pytest.raises(ValueError, match="spans several epochs"):
        d_bad_per.split_by_epoch()


# === GATE 1c: Joint VIS2 + corrflux + GainModes ===


def test_joint_vis2_corrflux_gain_modes_dense_mvn_and_zero_bleed():
    raw = _two_epoch_oidata(
        ("visamp", "t3amp"), amptyp="correlated flux"
    ).select(observables=("vis", "corrflux", "t3amp"))
    rng = onp.random.default_rng(77)
    n_u = raw.u.size
    cf_b = raw.extras[0]
    sample_modes = rng.normal(0.0, 0.02, (2, n_u))
    flux_modes = rng.normal(0.0, 0.03, (2, cf_b.values.size))

    configured = raw.with_flux_scale(
        scale=SCALE_PRIOR,
        per="frame",
        poly_order=1,
        poly_width=0.07,
        modes=flux_modes,
    ).with_gains(
        telescope=0.04, baseline=0.03, chromatic=0.02, modes=sample_modes
    )
    noisy = configured.with_model(_scene(), key=jax.random.PRNGKey(81))
    assert "_flux" in noisy.gains.groups
    assert _zero_size_leaves(noisy.gains) == []

    # widths_for pins _flux to 1.0 and refuses vis_gain__flux.
    w_mod = noisy.gains.widths_for(
        {"vis_gain_telescope": 0.09, "vis_gain_modes": 2.0}
    )
    flux_idx = noisy.gains.groups.index("_flux")
    assert float(w_mod[flux_idx]) == 1.0
    with pytest.raises(ValueError, match="vis_gain__flux"):
        noisy.gains.widths_for({"vis_gain__flux": 2.0})

    # Coupled corrflux refuses flux_scale_posterior with NotImplementedError.
    with pytest.raises(NotImplementedError, match="corrflux"):
        flux_scale_posterior(_scene(), noisy)

    other = _scene(dra=0.63, ddec=0.13)
    ll32 = float(model_loglike(other, noisy))

    with jax.enable_x64(True):
        # Rebuild GainModes in float64 so shapes/spanning have full f64 precision.
        d64 = (
            cast_tree(raw, "float64")
            .with_flux_scale(
                scale=SCALE_PRIOR,
                per="frame",
                poly_order=1,
                poly_width=0.07,
                modes=flux_modes,
            )
            .with_gains(
                telescope=0.04,
                baseline=0.03,
                chromatic=0.02,
                modes=sample_modes,
            )
        )
        # Copy the identical noisy realization into d64 for exact ll32 vs ll64 parity.
        d64 = eqx.tree_at(
            lambda d: (d.vis, d.extras[0].values, d.extras[1].values),
            d64,
            (
                np.asarray(noisy.vis, np.float64),
                np.asarray(noisy.extras[0].values, np.float64),
                np.asarray(noisy.extras[1].values, np.float64),
            ),
        )
        o64 = cast_tree(other, "float64")
        ll64 = float(model_loglike(o64, d64))

        # Independent sample-level log-|V| covariance over data.u.
        frames = onp.asarray(d64.frame)
        pairs = onp.sort(onp.asarray(d64.stations), axis=1)
        wavel = onp.broadcast_to(onp.asarray(d64.wavel, float), (n_u,))
        chrom = (onp.median(wavel) / wavel) ** 2
        cov_logv = sample_modes.T @ sample_modes
        for f in onp.unique(frames):
            in_f = onp.flatnonzero(frames == f)
            for t in onp.unique(pairs[in_f]):
                v = onp.zeros(n_u)
                v[in_f[(pairs[in_f] == t).any(axis=1)]] = 0.04
                cov_logv = cov_logv + onp.outer(v, v)
            for b_pair in onp.unique(pairs[in_f], axis=0):
                rows = in_f[(pairs[in_f] == b_pair).all(axis=1)]
                vb = onp.zeros(n_u)
                vb[rows] = 0.03
                vc = onp.zeros(n_u)
                vc[rows] = 0.02 * chrom[rows]
                cov_logv = cov_logv + onp.outer(vb, vb) + onp.outer(vc, vc)

        pred = onp.asarray(d64.model(o64), float)
        vals, errs = (onp.asarray(x, float) for x in d64.flatten_data())
        n_vis = d64.vis.size
        op_rows = []
        for r in range(n_vis):
            row = onp.zeros(n_u)
            row[r] = 2.0 * pred[r]
            op_rows.append(row)
        cf_block, t3_block = d64.extras
        n_cf = cf_block.values.size
        for k, s in enumerate(cf_block.sample):
            row = onp.zeros(n_u)
            row[int(s)] = pred[n_vis + k]
            op_rows.append(row)
        for k, (i1, i2, i3) in enumerate(
            zip(t3_block.i1, t3_block.i2, t3_block.i3)
        ):
            row = onp.zeros(n_u)
            for leg in (int(i1), int(i2), int(i3)):
                row[leg] += pred[n_vis + n_cf + k]
            op_rows.append(row)
        op = onp.stack(op_rows, axis=0)

        # Row-scoped corrflux covariance on [n_vis : n_vis + n_cf] ONLY.
        cov_joint = onp.diag(errs**2) + op @ cov_logv @ op.T
        cf_pred = pred[n_vis : n_vis + n_cf]
        cf_extra_cov = _dense_flux_cov(cf_block, cf_pred) - onp.diag(
            onp.asarray(cf_block.errors, float) ** 2
        )
        sl_cf = slice(n_vis, n_vis + n_cf)
        cov_joint[sl_cf, sl_cf] += cf_extra_cov

        j_vec = onp.concatenate([2.0 * pred[:n_vis], pred[n_vis:]])
        got_cov = onp.asarray(
            d64.gains.covariance(errs, j_vec, d64.gains.widths)
        )
        onp.testing.assert_allclose(got_cov, cov_joint, rtol=1e-12, atol=1e-12)

        expected = float(
            multivariate_normal.logpdf(vals, mean=pred, cov=cov_joint)
        )
        assert abs(ll64 - expected) < 1e-10
        assert abs(ll32 - expected) < 5e-5

        # Zero bleed onto VIS2 and T3AMP when corrflux scale/poly/modes change!
        alt = d64.with_flux_scale(
            scale=(3.7, 1.8),
            per="dataset",
            poly_order=2,
            poly_width=0.25,
            modes=3.0 * flux_modes,
        )
        alt_cov = onp.asarray(
            alt.gains.covariance(errs, j_vec, alt.gains.widths)
        )
        non_cf = onp.r_[:n_vis, n_vis + n_cf : errs.size]
        onp.testing.assert_allclose(
            alt_cov[onp.ix_(non_cf, non_cf)],
            got_cov[onp.ix_(non_cf, non_cf)],
            rtol=0.0,
            atol=1e-14,
        )
        ll_vis_a = float(model_loglike(o64, d64.select(observables="vis")))
        ll_vis_b = float(model_loglike(o64, alt.select(observables="vis")))
        assert abs(ll_vis_a - ll_vis_b) < 1e-14


def test_corrflux_only_flux_gains_and_exact_1x_simulated_covariance():
    raw = _two_epoch_oidata(("visamp",), amptyp="correlated flux").select(
        observables=("vis", "corrflux")
    )
    cf_b = raw.extras[0]
    rng = onp.random.default_rng(91)
    flux_modes = rng.normal(0.0, 0.05, (2, cf_b.values.size))

    # Empty sample-space with_gains() succeeds when corrflux has scale/modes!
    flux_only_gains = raw.with_flux_scale(
        scale=(3.7, 0.37), per="frame", modes=flux_modes
    ).with_gains()
    assert flux_only_gains.gains.groups == ("_flux",)

    # Exact 1x (never 2x double-drawn) covariance in OIData.with_model:
    template = raw.with_flux_scale(
        scale=(3.7, 0.37),
        per="frame",
        poly_order=1,
        poly_width=0.06,
        modes=flux_modes,
    ).with_gains(telescope=0.04, baseline=0.03)
    scene = _scene()
    pred = onp.asarray(template.model(scene), float)
    _, errs = (onp.asarray(x, float) for x in template.flatten_data())
    n_vis = template.vis.size
    j_vec = onp.concatenate([2.0 * pred[:n_vis], pred[n_vis:]])
    theo_cov = onp.asarray(
        template.gains.covariance(errs, j_vec, template.gains.widths), float
    )
    cf_theo_var = onp.diag(theo_cov)[n_vis:] - errs[n_vis:] ** 2

    n_mc = 400
    cf_draws = onp.stack(
        [
            onp.asarray(
                template.with_model(
                    scene, key=jax.random.PRNGKey(500 + i), noise_scale=0.0
                )
                .extras[0]
                .values,
                float,
            )
            for i in range(n_mc)
        ]
    )
    emp_var = onp.var(cf_draws, axis=0, ddof=1)
    ratio = float(onp.median(emp_var / cf_theo_var))
    # Must be ~1.0 (sampling sd ~0.07 for N=400), strictly ruling out 2.0x.
    assert 0.82 < ratio < 1.18


# === GATE 1d: Exact 6-Way Commutativity ===


def test_six_way_commutativity_flux_scale_gains_and_select():
    raw = _two_epoch_oidata(
        ("visamp", "visphi"), amptyp="correlated flux"
    ).with_continuum(lines=[LINE])
    noisy = raw.with_flux_scale(scale=SCALE_PRIOR).with_model(
        _scene(), key=jax.random.PRNGKey(123)
    )
    # Reset scale back to None so all 6 permutations start from identical unscaled data.
    unscaled = noisy.with_flux_scale(scale=None)

    rng = onp.random.default_rng(124)
    modes_u_full = rng.normal(0.0, 0.02, (2, unscaled.u.size))
    modes_cf_full = rng.normal(0.0, 0.03, (2, unscaled.extras[0].values.size))
    sel_ranges = [(2.160e-6, 2.178e-6)]

    from virgil.observables import in_ranges

    keep_u = in_ranges(unscaled.wavel, sel_ranges)
    keep_cf = keep_u[unscaled.extras[0].sample]

    def apply_flux(d):
        cf_len = d.extras[0].values.size
        m = (
            modes_cf_full
            if cf_len == modes_cf_full.shape[1]
            else modes_cf_full[:, keep_cf]
        )
        return d.with_flux_scale(
            scale=SCALE_PRIOR,
            per="frame",
            poly_order=1,
            poly_width=0.07,
            kinds="corrflux",
            modes=m,
        )

    def apply_gains(d):
        m = (
            modes_u_full
            if d.u.size == modes_u_full.shape[1]
            else modes_u_full[:, keep_u]
        )
        return d.with_gains(telescope=0.03, baseline=0.02, modes=m)

    def apply_select(d):
        return d.select(ranges=sel_ranges)

    ops = {"flux": apply_flux, "gains": apply_gains, "select": apply_select}
    other = _scene(dra=0.59, ddec=0.16)
    scores = {}
    with jax.enable_x64(True):
        u64 = cast_tree(unscaled, "float64")
        o64 = cast_tree(other, "float64")
        for perm in itertools.permutations(("flux", "gains", "select")):
            d = u64
            for name in perm:
                d = ops[name](d)
            scores[perm] = float(model_loglike(o64, cast_tree(d, "float64")))

    ref = next(iter(scores.values()))
    for perm, val in scores.items():
        assert abs(val - ref) < 1e-12, f"Mismatch for {perm}: {val} vs {ref}"


# === GATE 1e: epochs._Surface (marginal_loglike) 1D Quadrature Parity ===


def _dense_visphi_surface_integral(model, data, s_max=30.0, n_grid=3001):
    """Reference 1D integral over t = ln s for vis + visphi surface."""
    bound = onp.log(s_max)
    grid = onp.linspace(-bound, bound, n_grid)
    (block,) = data.extras
    cvis = model.model(data.u, data.v, data.wavel)
    pred_phi = onp.asarray(block.predict(model, cvis), float)
    y_phi = onp.asarray(block.data(), float)
    resid_phi = pred_phi - y_phi

    # White-noise covariance C_0 vs fixed prior + mode covariance U U^T.
    block_white = eqx.tree_at(
        lambda b: (b.prior_width, b.modes),
        block,
        (None, None),
        is_leaf=lambda x: x is None,
    )
    c0 = _dense_diffphase_cov(block_white)
    c_full = _dense_diffphase_cov(block)
    u_ut = c_full - c0
    eff_s1 = onp.sqrt(onp.diag(c0))

    # Integrate the visphi block term over t = ln s, matching _Surface._at's
    # additive normalization (+ sum(log(eff_s1))).
    log_f = []
    for t in grid:
        s2 = onp.exp(2.0 * t)
        cov_t = s2 * c0 + u_ut
        chol = onp.linalg.cholesky(cov_t)
        z = onp.linalg.solve(chol, resid_phi)
        logdet = 2.0 * onp.sum(onp.log(onp.diag(chol)))
        log_f.append(-0.5 * (z @ z) - 0.5 * logdet + onp.sum(onp.log(eff_s1)))
    log_f = onp.array(log_f)
    peak = log_f.max()
    phi_int = peak + onp.log(onp.trapezoid(onp.exp(log_f - peak), grid))

    vis_only = data.select(observables=("vis",))
    vis_int = float(marginal_loglike(model, vis_only, s_max=s_max))
    n_phi_ind = block.n_independent
    const = -0.5 * n_phi_ind * onp.log(2.0 * onp.pi) - onp.sum(onp.log(eff_s1))
    # Wait: let's compare model differences (log Bayes factor / relative surface
    # between two models), which cancels any model-independent normalization!
    return vis_int + phi_int + const


@pytest.mark.parametrize(
    "prior_width, use_modes",
    [
        ((0.12, 0.08), False),
        (None, True),
        ((0.12, 0.08), True),
        (None, False),
    ],
)
def test_surface_marginal_loglike_diffphase_quadrature_parity(
    prior_width, use_modes
):
    raw = _two_epoch_oidata(("visphi",)).select(observables=("vis", "visphi"))
    rng = onp.random.default_rng(202)
    n_u = raw.u.size
    frames = onp.asarray(raw.frame)
    modes = None
    if use_modes:
        modes = onp.zeros((2, n_u))
        modes[0, frames == 0] = rng.normal(0.0, 0.08, int((frames == 0).sum()))
        modes[1] = rng.normal(0.0, 0.05, n_u)

    data = raw.with_continuum(
        lines=[LINE], prior_width=prior_width, modes=modes
    )
    if prior_width is None and not use_modes:
        # Activate numeric surface via gains to exercise the c1 / s^2 branch!
        data = data.with_gains(telescope=0.04)
    noisy = data.with_model(
        _scene(), key=jax.random.PRNGKey(205), noise_scale=2.5
    )
    assert _Surface(noisy, 30.0).numeric

    m1 = _scene(dra=0.52, ddec=0.21)
    m2 = _scene(dra=0.68, ddec=0.09)

    with jax.enable_x64(True):
        d64 = cast_tree(noisy, "float64")
        m1_64 = cast_tree(m1, "float64")
        m2_64 = cast_tree(m2, "float64")
        got_diff = float(
            marginal_loglike(m1_64, d64, s_max=30.0)
            - marginal_loglike(m2_64, d64, s_max=30.0)
        )
        ref_diff = _dense_visphi_surface_integral(
            m1_64, d64, s_max=30.0
        ) - _dense_visphi_surface_integral(m2_64, d64, s_max=30.0)
        assert abs(got_diff - ref_diff) < 1e-4

        grad_dra = jax.grad(
            lambda dra: marginal_loglike(
                _scene(dra=dra, ddec=0.2), d64, s_max=30.0
            )
        )(np.asarray(0.55))
        assert onp.isfinite(float(grad_dra))


def test_night_summaries_diffphase_modes_and_prior_not_rescalable():
    pytest.importorskip("jaxoplanet")
    import types

    from virgil import night_summaries as ns

    t0 = _tables(mjd=60000.0)
    t1 = _tables(mjd=60000.05)
    t2 = _tables(mjd=60000.10)
    night = OIData(
        [build_hdulist(t) for t in (t0, t1, t2)], extras=("visphi",)
    )
    modes = onp.full((1, night.u.size), 0.06)
    pos = types.SimpleNamespace(
        names=("n",),
        peaks=(
            types.SimpleNamespace(
                dra=onp.array([0.5]),
                ddec=onp.array([0.2]),
                flux=onp.array([0.1]),
            ),
        ),
        primary_diam=None,
    )
    for configured in (
        night.with_continuum(lines=[LINE], modes=modes),
        night.with_continuum(lines=[LINE], prior_width=(0.15, 0.10)),
    ):
        peak = ns.night_summaries([configured], pos)[0].peaks[0]
        assert not peak.rescalable
        assert onp.isclose(peak.log_scale(1.0), 0.0)
        with pytest.raises(NotImplementedError):
            peak.covariance(scale=2.0)
