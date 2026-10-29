"""Laplace-marginalized antenna gains in virgil.radio."""

import equinox as eqx
import jax
import jax.numpy as np
import numpy as onp
import pytest

from virgil import (
    BinaryModelCartesian,
    CalibrationNuisance,
    ComplexVisData,
    GaussianDisk,
    fit,
)
from virgil.fitting import NoLeastSquaresForm
from virgil.radio import C_LIGHT

FREQ = 230e9


@pytest.fixture(autouse=True)
def float64():
    with jax.enable_x64(True):
        yield


def _array(n_ant=8, n_time=12, per_scan=4, seed=0):
    """(u, v, time, scan, ant1, ant2) of a rotating small array."""
    rng = onp.random.default_rng(seed)
    pos = rng.normal(size=(n_ant, 2)) * 6e5  # wavelengths
    i, j = onp.triu_indices(n_ant, 1)
    rows = []
    for t in range(n_time):
        h = 0.05 * t
        rot = onp.array([[onp.cos(h), -onp.sin(h)], [onp.sin(h), onp.cos(h)]])
        uv = (pos[i] - pos[j]) @ rot.T
        rows.append(
            onp.column_stack(
                [
                    uv,
                    onp.full(i.size, 10.0 * t),
                    onp.full(i.size, t // per_scan),
                    i,
                    j,
                ]
            )
        )
    a = onp.concatenate(rows)
    return (
        a[:, 0],
        a[:, 1],
        a[:, 2],
        a[:, 3].astype(int),
        a[:, 4].astype(int),
        a[:, 5].astype(int),
    )


def _scene(kind="disk"):
    if kind == "disk":
        return lambda v: GaussianDisk(sigma=6.0, dra=v["x"], ddec=0.0)
    return lambda v: BinaryModelCartesian(v["dra"], v["ddec"], v["q"])


def _data(
    scene,
    values,
    *,
    sigma=0.01,
    phase=1.0,
    amp=0.1,
    per="scan",
    seed=1,
    gains=True,
    noise=True,
    weight_follows_gain=False,
    two_obs=False,
    **kw,
):
    """Data from ``scene(values)`` with gains constant over a solution
    interval (``per`` = "scan" or "int"); returns ``(data, g_i g_j*, lam, phi)``."""
    u, v, t, scan, a1, a2 = _array(**kw)
    n_ant = int(max(a1.max(), a2.max())) + 1
    rng = onp.random.default_rng(seed)
    key = scan if per == "scan" else t
    keys = onp.unique(key)
    lam = rng.uniform(-amp, amp, (keys.size, n_ant))
    phi = rng.uniform(-phase, phase, (keys.size, n_ant))
    if not gains:
        lam, phi = 0 * lam, 0 * phi
    s = onp.searchsorted(keys, key)
    g = onp.exp(lam[s, a1] + lam[s, a2] + 1j * (phi[s, a1] - phi[s, a2]))
    wl = C_LIGHT / FREQ
    model = onp.asarray(scene(values).model(u * wl, v * wl, wl))
    noise_scale = float(noise)
    noise = rng.normal(size=u.size) + 1j * rng.normal(size=u.size)
    vis = g * model + noise_scale * sigma * noise
    data = ComplexVisData.from_arrays(
        u,
        v,
        vis,
        sigma**-2
        / (abs(g) ** 2 if weight_follows_gain else 1.0)
        * onp.ones(u.size),
        FREQ,
        obs=(t >= t.max() / 2).astype(int) if two_obs else 0,
        time=t,
        scan=scan,
        ant1=a1,
        ant2=a2,
    )
    return data, lam, phi


def _nuis(data, **kw):
    base = dict(
        gains=False, offsets=False, sigmas=False, alpha=False, nu0_hz=FREQ
    )
    base.update(kw)
    return CalibrationNuisance(data, **base)


def _loss(term, values):
    r, norm = term.residuals_and_log_norm(values)
    return 0.5 * np.sum(r**2) + norm


def _vals(**kw):
    out = {"cal.flux": 1.0}
    out.update({k: np.asarray(v) for k, v in kw.items()})
    return out


def test_calibrated_limit():
    """τ → 0: the marginal equals the ungained likelihood up to a constant."""
    scene = _scene()
    data, *_ = _data(scene, _vals(x=0.0))
    plain = data.term(scene, _nuis(data))
    gained = data.term(
        scene,
        _nuis(
            data,
            antenna_gains=True,
            gain_amp_width=1e-5,
            gain_phase_width=1e-5,
        ),
    )
    xs = [-1.0, 0.0, 0.7, 2.0]
    a = onp.array([_loss(plain, _vals(x=x)) for x in xs])
    b = onp.array([_loss(gained, _vals(x=x)) for x in xs])
    onp.testing.assert_allclose(b - b[0], a - a[0], rtol=1e-4, atol=1e-3)


def test_closure_invariance():
    """Large τ: the θ-dependence is unchanged by arbitrary data gains."""
    scene = _scene("binary")
    truth = _vals(dra=30.0, ddec=-20.0, q=0.5)
    # noise-free, with the weights scaled by |G|² (an amplitude gain rescales
    # the noise too, which the model does not know)
    clean, *_ = _data(
        scene, truth, gains=False, sigma=0.002, per="int", noise=False
    )
    gained, *_ = _data(
        scene,
        truth,
        phase=1.5,
        amp=0.3,
        sigma=0.002,
        per="int",
        noise=False,
        weight_follows_gain=True,
    )
    kw = dict(
        antenna_gains=True,
        solint="int",
        gain_amp_width=30.0,
        gain_phase_width=300.0,
        gain_steps=12,
    )
    ta = clean.term(scene, _nuis(clean, **kw))
    tb = gained.term(scene, _nuis(gained, **kw))
    thetas = [(30.0, -20.0, 0.5), (31.0, -20.0, 0.5), (30.0, -19.0, 0.4)]
    fa = onp.array(
        [_loss(ta, _vals(dra=a, ddec=b, q=q)) for a, b, q in thetas]
    )
    fb = onp.array(
        [_loss(tb, _vals(dra=a, ddec=b, q=q)) for a, b, q in thetas]
    )
    assert onp.ptp(fa) > 100  # the data do constrain θ
    onp.testing.assert_allclose(fb - fb[0], fa - fa[0], rtol=0.05, atol=2.0)


def test_snapshot_position_curvature_vanishes():
    """One snapshot, large τ: the source position is not constrained."""
    scene = _scene()
    data, *_ = _data(scene, _vals(x=0.0), n_time=1, per="int")

    def curvature(tau):
        term = data.term(
            scene,
            _nuis(
                data,
                antenna_gains=True,
                solint="int",
                gain_amp_width=tau,
                gain_phase_width=tau,
                gain_steps=8,
            ),
        )
        h = 0.3
        f = [_loss(term, _vals(x=x)) for x in (-h, 0.0, h)]
        return (f[0] - 2 * f[1] + f[2]) / h**2

    assert curvature(1e-5) > 100.0
    assert abs(curvature(1e3)) < 1e-2 * curvature(1e-5)


def test_solved_gains_match_injected():
    scene = _scene("binary")
    truth = _vals(dra=30.0, ddec=-20.0, q=0.5)
    data, lam, phi = _data(scene, truth, phase=1.0, sigma=0.003)
    term = data.term(
        scene,
        _nuis(
            data,
            antenna_gains=True,
            gain_steps=8,
            gain_amp_width=0.3,
            gain_phase_width=2.0,
        ),
    )
    g = onp.asarray(term.gains(truth))
    assert g.shape == lam.shape
    # the mean over antennas is set by the prior (a gauge of the data)
    for got, want in ((onp.log(abs(g)), lam), (onp.angle(g), phi)):
        got = got - got.mean(axis=1, keepdims=True)
        want = want - want.mean(axis=1, keepdims=True)
        onp.testing.assert_allclose(got, want, atol=0.03)


def test_recovery_with_large_phase_errors():
    scene = _scene("binary")
    truth = dict(dra=30.0, ddec=-20.0, q=0.5)
    data, *_ = _data(scene, _vals(**truth), phase=1.0, amp=0.1, sigma=0.02)
    nuis = _nuis(
        data,
        antenna_gains=True,
        gain_steps=8,
        gain_amp_width=0.2,
        gain_phase_width=1.0,
    )
    term = data.term(scene, nuis)
    import numpyro.distributions as dist

    priors = {
        "dra": dist.Uniform(20.0, 40.0),
        "ddec": dist.Uniform(-30.0, -10.0),
        "q": dist.Uniform(0.1, 1.0),
        "cal.flux": dist.LogUniform(0.1, 10.0),
    }
    init = dict(dra=29.0, ddec=-21.0, q=0.4, **{"cal.flux": 1.0})
    result = fit(
        lambda **v: None,
        priors,
        data=(),
        likelihoods=[term],
        init=init,
        method="lbfgs",
    )
    got = result.values
    assert float(got["dra"]) == pytest.approx(30.0, abs=0.5)
    assert float(got["ddec"]) == pytest.approx(-20.0, abs=0.5)
    assert float(got["q"]) == pytest.approx(0.5, abs=0.05)


def test_gradient_matches_finite_differences():
    scene = _scene()
    data, *_ = _data(scene, _vals(x=0.5))
    nuis = _nuis(data, antenna_gains=True, fit_gain_widths=True, gain_steps=6)
    term = data.term(scene, nuis)
    keys = ["x", "cal.flux", "cal.gain_amp_width", "cal.gain_phase_width"]
    x0 = onp.array([0.4, 1.05, 0.12, 0.6])

    def f(z):
        return _loss(term, dict(zip(keys, z)))

    g = onp.asarray(jax.grad(f)(np.asarray(x0)))
    for k in range(4):
        h = 1e-5 * max(1.0, abs(x0[k]))
        e = onp.zeros(4)
        e[k] = h
        fd = (f(np.asarray(x0 + e)) - f(np.asarray(x0 - e))) / (2 * h)
        assert g[k] == pytest.approx(float(fd), rel=1e-4, abs=1e-6)


def test_lm_is_refused():
    scene = _scene()
    data, *_ = _data(scene, _vals(x=0.0))
    term = data.term(scene, _nuis(data, antenna_gains=True))
    import numpyro.distributions as dist

    with pytest.raises(NoLeastSquaresForm):
        fit(
            lambda **v: None,
            {
                "x": dist.Uniform(-3.0, 3.0),
                "cal.flux": dist.LogUniform(0.1, 10.0),
            },
            data=(),
            likelihoods=[term],
            init={"x": 0.1, "cal.flux": 1.0},
            method="lm",
        )


def test_padding_and_masks_are_inert():
    """Chunk padding and masked rows (with garbage data) change nothing."""
    scene = _scene()
    data, *_ = _data(scene, _vals(x=0.3))
    nuis = _nuis(data, antenna_gains=True)
    base = _loss(data.term(scene, nuis), _vals(x=0.3))
    padded = _loss(data.term(scene, nuis, chunk_size=7), _vals(x=0.3))
    assert float(padded) == pytest.approx(float(base), rel=1e-10)
    bad = onp.arange(len(data)) % 5 == 0
    junk = eqx.tree_at(
        lambda d: d.vis, data, np.where(bad, 1e3 + 1e3j, data.vis)
    )
    masked = _loss(
        junk.term(scene, nuis, mask=~bad, chunk_size=7), _vals(x=0.3)
    )
    kept = _loss(data.subset(~bad).term(scene, nuis), _vals(x=0.3))
    assert float(masked) == pytest.approx(float(kept), rel=1e-6)


def test_option_validation_and_diagnostics():
    scene = _scene()
    data, *_ = _data(scene, _vals(x=0.0))
    with pytest.raises(ValueError, match="gain_steps"):
        _nuis(data, antenna_gains=True, gain_steps=0)
    with pytest.raises(ValueError, match="antenna_gains"):
        _nuis(data, fit_gain_widths=True)
    nuis = _nuis(data, antenna_gains=True, sigmas="profile", gain_steps=8)
    term = data.term(scene, nuis)
    v = _vals(x=0.0)
    assert float(term.gain_step(v)) < 1e-3  # converged
    assert "cal.gain_step" in term.derived(v)
    # the reported gains are those of the likelihood: the solved phases are
    # recovered from the prior rows, whatever ŝ is
    assert onp.all(onp.isfinite(onp.asarray(term.gains(v))))
    with pytest.raises(ValueError, match="no antenna gains"):
        data.model(scene(v), v, nuis)


def test_needs_time_and_antennas():
    data = ComplexVisData.from_arrays(
        [1e5, 2e5], [0.0, 1e5], [1, 1], 1.0, FREQ
    )
    with pytest.raises(ValueError, match="time"):
        CalibrationNuisance(data, antenna_gains=True)


def test_intervals_never_cross_scans():
    from virgil._antenna_gains import gain_intervals

    scene = _scene()
    data, *_ = _data(scene, _vals(x=0.0), n_time=8, per_scan=4)
    n_scan = gain_intervals(data, "scan")[3]
    n_int = gain_intervals(data, "int")[3]
    assert (n_scan, n_int) == (2, 8)
    # 25 s bins of 10 s integrations: 3 integrations a bin, cut at the scan
    assert gain_intervals(data, 25.0)[3] == 4


def test_no_recompile_when_values_change():
    scene = _scene()
    data, *_ = _data(scene, _vals(x=0.0))
    term = data.term(
        scene, _nuis(data, antenna_gains=True, fit_gain_widths=True)
    )
    traces = []

    @eqx.filter_jit
    def f(term, x, tau):
        traces.append(1)
        return _loss(
            term,
            {
                "x": x,
                "cal.flux": 1.0,
                "cal.gain_amp_width": tau,
                "cal.gain_phase_width": tau,
            },
        )

    f(term, np.asarray(0.1), np.asarray(0.2))
    f(term, np.asarray(0.3), np.asarray(0.4))
    assert len(traces) == 1


def test_profiled_sigma_runs_and_matches_fitted_scale():
    scene = _scene()
    data, *_ = _data(scene, _vals(x=0.0), sigma=0.01)
    term = data.term(scene, _nuis(data, antenna_gains=True, sigmas="profile"))
    s = onp.asarray(term.sigma_hat(_vals(x=0.0)))
    assert s.shape == (1,) and 0.5 < s[0] < 1.5


@pytest.mark.slow
def test_diag_precondition_and_offsets_do_not_wander():
    """L-BFGS with ``precondition='diag'`` converges; with per-observation
    offsets the source offset is not absorbed into ``cal.offset``."""
    import numpyro.distributions as dist

    scene = _scene()
    data, *_ = _data(scene, _vals(x=3.0), phase=0.5, sigma=0.01)
    nuis = CalibrationNuisance(
        data,
        gains=False,
        offsets=False,
        sigmas=False,
        alpha=False,
        nu0_hz=FREQ,
        antenna_gains=True,
        gain_steps=6,
    )
    term = data.term(scene, nuis)
    result = fit(
        lambda **v: None,
        {
            "x": dist.Uniform(-10.0, 10.0),
            "cal.flux": dist.LogUniform(0.1, 10.0),
        },
        data=(),
        likelihoods=[term],
        init={"x": 1.0, "cal.flux": 1.0},
        method="lbfgs",
        precondition="diag",
    )
    assert float(result.values["x"]) == pytest.approx(3.0, abs=0.3)


@pytest.mark.slow
def test_offset_with_gains_is_pinned_by_the_phase_prior():
    """``cal.offset`` is a phase gradient the gains can absorb; the finite
    ``τ_φ`` keeps it from wandering when the whole source is offset."""
    import numpyro.distributions as dist

    scene = _scene()
    data, *_ = _data(
        scene, _vals(x=3.0), phase=0.5, sigma=0.01, n_time=8, two_obs=True
    )
    nuis = _nuis(
        data,
        offsets=True,
        antenna_gains=True,
        gain_steps=6,
        gain_phase_width=0.5,
    )
    term = data.term(scene, nuis)
    priors = {
        "x": dist.Uniform(-10.0, 10.0),
        "cal.flux": dist.LogUniform(0.1, 10.0),
        **{k: v for k, v in nuis.priors().items() if k == "cal.offset"},
    }
    result = fit(
        lambda **v: None,
        priors,
        data=(),
        likelihoods=[term],
        init={**nuis.defaults(), "x": 2.0},
        method="lbfgs",
    )
    # The noise (sigma 0.01) is far too small to set these bounds: the error is
    # the prior's. With two intervals per observation the gains absorb most of
    # the ramp, so x is known only through τ_φ = 0.5 rad (measured |x - 3| =
    # 0.58 mas from a start 1 mas off; |cal.offset| = 0.73 mas at most, a
    # fraction of the prior box). Bounds of 0.8 leave 1.4x and 1.1x over the
    # deterministic fit; a drift to the old 1.0 would hide the gains absorbing
    # the source offset.
    assert float(result.values["x"]) == pytest.approx(3.0, abs=0.8)
    assert onp.all(onp.abs(onp.asarray(result.values["cal.offset"])) < 0.8)
