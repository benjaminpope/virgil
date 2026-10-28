"""Tempering the orbit evidence for correlated spectral channels (plan 1d).

The solved problem is the untempered evidence: with correlated channels the
best alias band got p = 1.000 at χ²/N 12, and the posteriors were far too
narrow. The effective-channel factor f = N / N_eff is estimated from the
residuals of neighbouring channels (``virgil.tempering``) and the likelihood
is tempered by 1/f in ``fit_orbit_aliases``.
"""

import dataclasses
import functools
import json
import types
import warnings

import jax
import numpy as onp
import pytest

pytest.importorskip("jaxoplanet")

from virgil import aliases, tempering  # noqa: E402
from virgil._precision import cast_tree  # noqa: E402
from virgil.aliases import AliasBandResult, AliasResult  # noqa: E402
from virgil.epochs import EpochPositions, _Surface  # noqa: E402
from virgil.models import OrbitalBinary  # noqa: E402
from virgil.oidata import OIData, cp_indices  # noqa: E402
from virgil.orbits import KeplerOrbit  # noqa: E402

T_REF = 60000.0


def _chain_factor(rho, n_channels):
    """The exact AR(1) factor ``1 + 2 Σ (1 - k/n) ρ^k`` of a chain of
    ``n_channels``, the oracle for ``estimate_factors``."""
    n = int(n_channels)
    k = onp.arange(1, n)
    return float(1.0 + 2.0 * onp.sum((1.0 - k / n) * rho**k))


TRUTH = dict(
    period=12.38,
    dt_peri=2.0,
    ecc=0.3,
    inc=50.0,
    omega=60.0,
    Omega=120.0,
    a_mas=10.0,
)
FLUX = 0.3
EPOCHS = T_REF + onp.array([0.0, 2.4, 5.1, 238.0, 240.6, 247.4])
STATIONS = onp.array([[0.0, 0.0], [60.0, 5.0], [25.0, 70.0], [-40.0, 45.0]])
PAIRS = onp.array([[1, 2], [1, 3], [1, 4], [2, 3], [2, 4], [3, 4]])
TRIANGLES = onp.array([[1, 2, 3], [1, 2, 4], [1, 3, 4], [2, 3, 4]])
WAVEL = onp.linspace(2.0e-6, 2.4e-6, 8)  # eight channels
SIGMA_V2, SIGMA_CP = 0.04, 0.08  # quoted errors
SCALE = 3.46  # the noise is this times the quoted errors: χ²/N = 12


def _night_dict(mjd, rotation=0.0, frames=3, vis=None, phi=None):
    """One night of V² and closure phases in eight channels (the layout is
    baseline-major with channels fastest, as OIData stores several
    channels), one time for every frame."""
    delta = STATIONS[PAIRS[:, 1] - 1] - STATIONS[PAIRS[:, 0] - 1]
    i1, i2, i3 = cp_indices(PAIRS, TRIANGLES)
    n = len(PAIRS)
    angles = onp.deg2rad(rotation + 15.0 * onp.arange(frames))
    c, s = onp.cos(angles)[:, None], onp.sin(angles)[:, None]
    u = (delta[:, 0] * c - delta[:, 1] * s).reshape(-1)
    v = (delta[:, 0] * s + delta[:, 1] * c).reshape(-1)

    def shift(i):
        return onp.concatenate([i + k * n for k in range(frames)])

    nb, nt, nw = u.size, len(i1) * frames, len(WAVEL)
    return dict(
        u=u,
        v=v,
        wavel=WAVEL,
        vis=onp.ones((nb, nw)) if vis is None else vis,
        d_vis=onp.full((nb, nw), SIGMA_V2),
        phi=onp.zeros((nt, nw)) if phi is None else phi,
        d_phi=onp.full((nt, nw), SIGMA_CP),
        i_cps1=shift(i1),
        i_cps2=shift(i2),
        i_cps3=shift(i3),
        mjd=onp.full(nb, mjd),
    )


def _ar1(rng, shape, rho):
    """Noise along the last axis with neighbour correlation ``rho``."""
    z = rng.standard_normal(shape)
    out = onp.empty(shape)
    out[..., 0] = z[..., 0]
    for k in range(1, shape[-1]):
        out[..., k] = rho * out[..., k - 1] + onp.sqrt(1 - rho**2) * z[..., k]
    return out


def _nights(
    rho, scale=SCALE, seed=0, n=len(EPOCHS), period=12.38, common=False
):
    """Nights of a binary orbit observed in eight channels, with noise of
    neighbour-channel correlation ``rho`` and ``scale`` times the quoted
    errors (``common``: the same correlation between every pair of channels,
    a calibration error shared by all of them), and the true per-night
    positions."""
    noise = _common if common else _ar1
    orbit = KeplerOrbit(**{**TRUTH, "period": period}, t_ref=T_REF)
    scene = OrbitalBinary(orbit, FLUX)
    rng = onp.random.default_rng(seed)
    data = []
    for k, mjd in enumerate(EPOCHS[:n]):
        shape = _night_dict(mjd, 7.0 * k)
        clean = OIData(shape)
        pred = onp.asarray(clean.model(scene))
        nv, nphi = clean.vis.size, clean.phi.size
        vis = pred[:nv].reshape(shape["vis"].shape)
        phi = pred[nv : nv + nphi].reshape(shape["phi"].shape)
        vis = vis + scale * SIGMA_V2 * noise(rng, vis.shape, rho)
        phi = phi + scale * SIGMA_CP * noise(rng, phi.shape, rho)
        data.append(OIData(_night_dict(mjd, 7.0 * k, vis=vis, phi=phi)))
    dra, ddec, _ = orbit.relative(EPOCHS[:n])
    return data, _positions(dra, ddec)


def _positions(dra, ddec):
    n = len(dra)
    return EpochPositions(
        names=tuple(f"n{k}" for k in range(n)),
        mjd=EPOCHS[:n],
        dra=onp.asarray(dra) + 0.2,  # off the truth: the pilot refines them
        ddec=onp.asarray(ddec) - 0.2,
        cov=onp.tile(onp.eye(2), (n, 1, 1)),
        flux=onp.full(n, FLUX),
        gap_marginal=onp.full(n, 20.0),
        chi2_raw=({},) * n,
        scale=({},) * n,
        flags=((),) * n,
    )


_cached = functools.cache(_nights)


def _pilot(rho, n=3):
    data, positions = _cached(rho, n=n)
    with jax.enable_x64(True):
        data = [cast_tree(d, "float64") for d in data]
        return aliases._temper_factors(
            data, positions, None, None, ["vis", "phi"]
        )


# --- the estimator -----------------------------------------------------------


def _chains_of(n_chains, length):
    """``n_chains`` chains of consecutive indices, and the flat layout."""
    idx = onp.arange(n_chains * length).reshape(n_chains, length)
    return list(idx)


def _common(rng, shape, c):
    """Noise with correlation ``c`` between every pair of channels."""
    return onp.sqrt(c) * rng.standard_normal(shape[:-1] + (1,)) + onp.sqrt(
        1 - c
    ) * rng.standard_normal(shape)


def test_white_noise_gives_f_within_five_percent_of_one():
    # Plain white residuals: f = 1 on average, scattering by sqrt(2/chains)
    # (0.01 at 20000 chains); max(f, 1) is the used value.
    for seed in range(5):
        z = onp.random.default_rng(seed).standard_normal((20000, 8))
        factors = tempering.estimate_factors(
            [{"vis": z.reshape(-1)}], [{"vis": _chains_of(20000, 8)}]
        )
        assert factors.f == pytest.approx(1.0, rel=0.05)
        assert factors.raw["vis"] == pytest.approx(1.0, rel=0.05)


def test_the_estimator_recovers_f_for_ar1_and_for_common_mode_noise():
    # AR(1), rho = 0.6, n = 8: f = _chain_factor. Common mode, c = 0.6: f =
    # 1 + (n - 1) c = 5.2, which a lag-1 AR(1) estimate puts at 3.2. The
    # inflation of the errors (chi2/N = 12) does not matter.
    rng = onp.random.default_rng(1)
    ar = 3.46 * _ar1(rng, (4000, 8), 0.6)
    cm = 3.46 * _common(rng, (4000, 8), 0.6)
    chains = [{"vis": _chains_of(4000, 8)}]
    f_ar = tempering.estimate_factors([{"vis": ar.reshape(-1)}], chains)
    f_cm = tempering.estimate_factors([{"vis": cm.reshape(-1)}], chains)
    assert f_ar.f == pytest.approx(_chain_factor(0.6, 8), rel=0.07)
    assert f_cm.f == pytest.approx(5.2, rel=0.07)
    assert f_cm.blocks["vis"] == pytest.approx(f_cm.f)
    # the AR(1) reading of the same lag-1 correlation misses the common mode
    ar1_guess = _chain_factor(f_cm.rho["vis"], 8)
    assert ar1_guess < 0.7 * f_cm.f
    # lag-1 and lag-2 are diagnostics: AR has rho2 = rho1^2, common mode not
    assert f_ar.rho2["vis"] == pytest.approx(f_ar.rho["vis"] ** 2, abs=0.05)
    assert not f_ar.flat and f_cm.flat == ("vis",)


def test_negative_correlation_is_floored_at_one_and_keeps_the_raw_value():
    # Alternating residuals (differenced data): the chain sum is smaller
    # than independent noise, f-hat is far below 1, tempering never sharpens.
    rng = onp.random.default_rng(2)
    z = _ar1(rng, (4000, 8), -0.7)
    factors = tempering.estimate_factors(
        [{"vis": z.reshape(-1)}], [{"vis": _chains_of(4000, 8)}]
    )
    assert factors.raw["vis"] < 0.5
    assert factors.f == 1.0 and factors.blocks["vis"] == 1.0
    assert factors.f_raw < 0.5 and tempering.out_of_range(
        factors.f, factors.f_raw
    )


def test_blocks_with_few_chains_take_the_pooled_factor():
    rng = onp.random.default_rng(0)
    z = 3.0 * _ar1(rng, (400, 10), 0.7).reshape(-1)  # the vis block
    few = _chains_of(5, 6)  # 5 chains of the phi block
    factors = tempering.estimate_factors(
        [{"vis": z, "phi": rng.standard_normal(30)}],
        [{"vis": _chains_of(400, 10), "phi": few}],
        names=["extra[0]"],
    )
    assert factors.blocks["phi"] == pytest.approx(factors.f)
    assert factors.blocks["extra[0]"] == pytest.approx(factors.f)
    assert factors.f > 3.0
    assert "phi" not in factors.rho and "vis" in factors.rho
    assert set(factors.fallback) == {"phi", "extra[0]"}


def test_the_public_dof_helper_inverts_the_blocks():
    blocks = {"vis": 4.0, "phi": 0.5, "extra[0]": 2.0}
    assert tempering.block_dof(blocks) == {
        "vis": 0.25, "phi": 1.0, "extra[0]": 0.5,
    }  # fmt: skip


def test_neighbouring_channels_are_found_from_the_data_layout():
    data = OIData(_night_dict(T_REF))
    pairs = tempering.adjacent_channels(data)
    nw = len(WAVEL)
    assert len(pairs["vis"][0]) == 18 * (nw - 1)  # 6 baselines x 3 frames
    assert len(pairs["phi"][0]) == 12 * (nw - 1)  # 4 triangles x 3 frames
    a, b = pairs["vis"]
    assert onp.all(b - a == 1)  # channels are fastest within a baseline
    assert onp.all(onp.asarray(data.u)[a] == onp.asarray(data.u)[b])
    assert onp.all(onp.asarray(data.wavel)[b] > onp.asarray(data.wavel)[a])


def test_data_with_one_wavelength_have_no_pairs_and_f_is_one():
    one = _night_dict(T_REF)
    one.update(
        wavel=2.2e-6,
        vis=onp.ones(one["u"].size),
        d_vis=onp.full(one["u"].size, 0.01),
    )
    one["phi"] = onp.zeros(len(one["i_cps1"]))
    one["d_phi"] = onp.full(len(one["i_cps1"]), 0.02)
    pairs = tempering.adjacent_channels(OIData(one))
    assert all(len(p[0]) == 0 for p in pairs.values())
    factors = tempering.estimate_factors([], [], names=["vis", "phi"])
    assert factors.f == 1.0 and factors.blocks == {"vis": 1.0, "phi": 1.0}


def test_the_pilot_on_white_channels_gives_f_of_one():
    # White noise at χ²/N = 12: the error scale absorbs the inflation and
    # the channels are independent, so no tempering.
    factors = _pilot(0.0, n=2)
    # sqrt(2 / chains) = 0.13 for the 120 chains of two nights, and the
    # floor at 1 biases it up a little: the safe direction
    assert 1.0 <= factors.f < 1.5
    assert factors.n_pairs["pooled"] > 300


def test_the_pilot_on_correlated_channels_gives_f_above_one():
    # ρ = 0.85 between neighbours of an eight-channel chain: the finite-chain
    # factor 1 + 2 Σ (1 - k/8) ρ^k = 5.5 (the asymptotic 1.85/0.15 = 12.3
    # would exceed the eight channels there are). The estimate is a little
    # low because the pilot fits the night's position and flux.
    factors = _pilot(0.85)
    assert 3.5 < factors.f < 8.0
    assert all(3.5 < v < 8.0 for v in factors.blocks.values())
    assert not factors.flat  # AR(1): rho2 = rho1^2
    assert factors.f >= 1.0 and factors.f_raw <= factors.f_max


# --- the tempered likelihood -------------------------------------------------


def _cfg():
    return (T_REF, None, False, None)


def test_temper_false_reproduces_the_untempered_evidence():
    # dof = 1 in every block is the old likelihood (temper=False).
    data, _ = _cached(0.85, n=2)
    with jax.enable_x64(True):
        data = [cast_tree(d, "float64") for d in data]
        surfaces = tuple(_Surface(d) for d in data)
        free = aliases._Bounds(onp.full(8, -onp.inf), onp.full(8, onp.inf))
        band = types.SimpleNamespace(p_lo=11.0, p_hi=14.0)
        x = aliases._start_vector(
            KeplerOrbit(**TRUTH, t_ref=T_REF), FLUX, T_REF, band, free
        )
        model = aliases._model(x, _cfg())
        old = sum(
            s.score(aliases._epoch_chi2(s, model, d, None)[0], 1.0, d)
            for s, d in zip(surfaces, data)
        )
        ones = tuple(onp.ones(len(s.names)) for s in surfaces)
        new = _ell_sum(surfaces, x, data, _cfg(), ones)
        assert float(new) == pytest.approx(float(old), rel=1e-12)
        # a dof of 1/f in every block divides the log likelihood by f
        f = 12.0
        dof = tuple(onp.full(len(s.names), 1.0 / f) for s in surfaces)
        temp = _ell_sum(surfaces, x, data, _cfg(), dof)
        assert float(temp) == pytest.approx(float(old) / f, rel=1e-12)


def _ell_sum(surfaces, x, data, cfg, dof):
    """The night terms' sum: the likelihood before the outlier mixture, which
    adds a rate-marginalized outlier component that tempering does not scale."""
    ell, _ = aliases._night_terms(
        surfaces, x, tuple(data), cfg, dof, onp.full(len(data), -50.0)
    )
    return float(ell.sum())


def test_the_problem_value_is_the_untempered_value_over_f():
    data, _ = _cached(0.85, n=2)

    def value(f):
        settings = _settings_stub(
            dof=() if f == 1 else (("phi", 1 / f), ("vis", 1 / f))
        )
        return aliases._Problem(
            data, dataclasses.replace(settings, log_broad=(-50.0,) * len(data))
        )

    with jax.enable_x64(True):
        data = [cast_tree(d, "float64") for d in data]
        band = types.SimpleNamespace(p_lo=11.0, p_hi=14.0)
        free = aliases._Bounds(onp.full(8, -onp.inf), onp.full(8, onp.inf))
        x = aliases._start_vector(
            KeplerOrbit(**TRUTH, t_ref=T_REF), FLUX, T_REF, band, free
        )
        a, b = value(1), value(4.0)
        va = _ell_sum(a.surfaces, x, a.data, a.cfg, a.dof)
        vb = _ell_sum(b.surfaces, x, b.data, b.cfg, b.dof)
        assert vb == pytest.approx(va / 4.0, rel=1e-10)


def test_the_dof_argument_is_f_one_or_a_fixed_f(monkeypatch):
    data, positions = _cached(0.0, n=3)

    class Stop(Exception):
        pass

    seen = {}

    def preflight(problem, settings, band):
        seen["settings"] = settings
        raise Stop

    monkeypatch.setattr(aliases, "_preflight", preflight)
    monkeypatch.setattr(
        aliases,
        "position_profile",
        lambda pos, bands, **kw: [dict(orbits=[]) for _ in bands],
    )
    monkeypatch.setattr(
        aliases,
        "_short_arc_profile",
        lambda data, seed, pos, bands, **kw: (
            [dict(orbits=[]) for _ in bands],
            None,
            None,
        ),
    )

    def run(**kw):
        with pytest.raises(Stop), warnings.catch_warnings():
            warnings.simplefilter("ignore")  # f = 4 is outside [0.5, 2]
            aliases.fit_orbit_aliases(
                data, (12.0, 13.0), positions=positions, t_ref=T_REF,
                primary_diam=None, n_random=5, **kw,
            )  # fmt: skip
        return seen["settings"]

    assert run(temper=False).dof == ()
    assert run(temper=1.0).dof == ()  # f = 1 is the untempered fit
    assert run(temper=4.0).dof == (("phi", 0.25), ("vis", 0.25))
    # white channels estimate f near 1 (floored at 1, so a little above:
    # the safe direction), not 4
    white = dict(run(temper=True).dof)
    assert all(0.6 < v <= 1.0 for v in white.values())
    with pytest.raises(ValueError, match="temper must be"):
        run(temper=-2.0)


def test_correlated_channels_are_tempered_by_default(monkeypatch):
    data, positions = _cached(0.85, n=3)
    seen = {}

    class Stop(Exception):
        pass

    def preflight(problem, settings, band):
        seen["settings"] = settings
        raise Stop

    monkeypatch.setattr(aliases, "_preflight", preflight)
    monkeypatch.setattr(
        aliases,
        "position_profile",
        lambda pos, bands, **kw: [dict(orbits=[]) for _ in bands],
    )
    monkeypatch.setattr(
        aliases,
        "_short_arc_profile",
        lambda data, seed, pos, bands, **kw: (
            [dict(orbits=[]) for _ in bands],
            None,
            None,
        ),
    )
    with pytest.warns(UserWarning, match="effective-channel factor"):
        with pytest.raises(Stop):
            aliases.fit_orbit_aliases(
                data, (12.0, 13.0), positions=positions, t_ref=T_REF,
                primary_diam=None, n_random=5,
            )  # fmt: skip
    dof = dict(seen["settings"].dof)
    assert set(dof) == {"vis", "phi"} and all(v < 0.5 for v in dof.values())


def test_several_channels_without_positions_warn_and_stay_untempered(
    monkeypatch,
):
    data, _ = _cached(0.85, n=3)

    class Stop(Exception):
        pass

    def preflight(problem, settings, band):
        raise Stop

    monkeypatch.setattr(aliases, "_preflight", preflight)
    with pytest.warns(UserWarning, match="needs EpochPositions"):
        with pytest.raises(Stop):
            aliases.fit_orbit_aliases(
                data, (12.0, 13.0), t_ref=T_REF, primary_diam=None,
                n_random=5,
            )  # fmt: skip


# --- the result --------------------------------------------------------------


def _band(n):
    return AliasBandResult(
        n=n, p_lo=1.0, p_hi=2.0, modes=[], log_z=-1.0, log_z_is=-1.0,
        ess=10.0, flagged=False, epochs=[],
    )  # fmt: skip


def test_the_factor_is_saved_and_loaded(tmp_path):
    result = AliasResult(
        T_REF, onp.zeros(2), 1.0, [_band(3)], {},
        primary_diam=None, vis_amp=None,
        temper_f=7.5, temper_blocks={"vis": 7.0, "phi": 8.0},
    )  # fmt: skip
    assert result.tempering  # 7.5 is outside [0.5, 2]
    result.save(tmp_path / "r.json")
    loaded = AliasResult.load(tmp_path / "r.json")
    assert loaded.temper_f == 7.5
    assert loaded.temper_blocks == {"vis": 7.0, "phi": 8.0}
    assert loaded.tempering


def test_an_old_file_loads_without_a_factor(tmp_path):
    result = AliasResult(
        T_REF, onp.zeros(2), 1.0, [_band(3)], {},
        primary_diam=None, vis_amp=None,
    )  # fmt: skip
    result.save(tmp_path / "r.json")
    payload = json.loads((tmp_path / "r.json").read_text())
    del payload["temper_f"], payload["temper_blocks"]  # before plan 1d
    (tmp_path / "old.json").write_text(json.dumps(payload))
    loaded = AliasResult.load(tmp_path / "old.json")
    assert loaded.temper_f is None and loaded.temper_blocks is None
    assert not loaded.tempering


@pytest.mark.parametrize(
    "f, flagged", [(1.0, False), (0.5, False), (2.0, False), (2.5, True),
                   (0.3, True), (None, False)]
)  # fmt: skip
def test_the_tempering_flag_is_outside_half_to_two(f, flagged):
    result = AliasResult(T_REF, onp.zeros(2), 1.0, [_band(3)], {}, temper_f=f)
    assert result.tempering is flagged


# --- the thresholds ----------------------------------------------------------


def _settings_stub(**kw):
    args = dict(
        p_range=(1.0, 2.0), a_range=(1.0, 30.0), flux_range=(0.01, 1.0),
        ecc_max=0.9, s_max=None, t_ref=T_REF, n_candidates=1, n_refine=1,
        n_random=1, pool_size=64, k=3.0, eccs=None, min_distance_mas=0.5,
        max_steps=1, n_is=1, flag_nats=1.0, ess_min=1.0, seed=0,
    )  # fmt: skip
    return aliases._Settings(**{**args, **kw})


@pytest.mark.slow
def test_the_resolution_threshold_is_on_the_tempered_score():
    # estimate_primary_diameter's gain over a point primary is dof times the
    # untempered one: the same 5-nat threshold then asks for f times the raw
    # gain, and is unchanged for f = 1.
    from virgil.epochs import RESOLVE_NATS, estimate_primary_diameter
    from virgil.models import PointSource, System, UniformDisk
    from virgil.simulate import simulate

    truth = (5.0, -3.0, 0.25)
    scene = System(
        primary=UniformDisk(1.05),
        companion=PointSource(truth[2], truth[0], truth[1]),
    )
    data = [
        simulate(scene, OIData(_single(k)), jax.random.PRNGKey(k))
        for k in range(4)
    ]
    start = _positions(onp.full(4, 5.3 - 0.2), onp.full(4, -2.8 + 0.2))
    start = EpochPositions(**{**start.__dict__, "flux": onp.full(4, 0.2)})
    full = estimate_primary_diameter(data, start)
    assert full.resolved and full.delta_nats > RESOLVE_NATS
    quarter = estimate_primary_diameter(data, start, dof=0.25)
    assert quarter.delta_nats == pytest.approx(full.delta_nats / 4, rel=0.05)
    blocks = estimate_primary_diameter(
        data, start, block_dof={"vis": 0.25, "phi": 0.25}
    )
    assert blocks.delta_nats == pytest.approx(quarter.delta_nats, rel=1e-3)


def _single(k):
    d = _night_dict(T_REF + k, 7.0 * k, frames=2)
    nb = d["u"].size
    nt = len(d["i_cps1"])
    d.update(
        wavel=2.2e-6,
        vis=onp.ones(nb),
        d_vis=onp.full(nb, 0.01),
        phi=onp.zeros(nt),
        d_phi=onp.full(nt, 0.02),
    )
    return d


# --- the engine --------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.filterwarnings("ignore:fit_orbit_aliases")  # f, poor-fit
def test_tempering_turns_an_overconfident_band_probability_into_a_calibrated_one():
    # Eight correlated channels (ρ = 0.85, f ≈ 5.5) with noise 3.5 times the
    # quoted errors (χ²/N = 11 at the best band). The untempered evidence
    # separates the bands by a gap of about f times the tempered one: p =
    # 1.000, which the data do not support. Tempered, the same bands are
    # f times closer together and p is below 1.
    data, positions = _cached(0.85, period=12.1)
    result = aliases.fit_orbit_aliases(
        data,
        (11.8, 12.9),
        positions=positions,
        t_ref=T_REF,
        n_candidates=10,
        n_refine=2,
        n_is=200,
        n_samples=20,
        primary_diam=None,
    )
    assert 4.0 < result.temper_f < 7.0
    best = result.bands[0]
    assert best.n == 20
    assert 9.0 < best.chi2_red < 14.0
    assert 0.5 < best.p < 1.0
    # temper=False is the old behaviour: p = 1.000
    old = aliases.fit_orbit_aliases(
        data,
        (11.8, 12.9),
        positions=positions,
        t_ref=T_REF,
        n_candidates=10,
        n_refine=2,
        n_is=200,
        n_samples=20,
        primary_diam=None,
        temper=False,
    )
    assert old.temper_f is None
    # the untempered band is certain to better than 1e-8 where the
    # tempered one is 1e-5 uncertain
    assert 1.0 - old.bands[0].p < 1e-8 < 1.0 - best.p


# --- calibration of the posterior against the truth --------------------------


def _offset(result, truth_xy, epoch):
    """The sky-plane offset of the posterior mean of the position at
    ``epoch`` from the truth, in standard deviations of the posterior (a
    2-D Mahalanobis distance: about 1.2 for a calibrated posterior)."""
    s = result.samples[result.bands[0].n]
    keys = ("period", "dt_peri", "ecc", "inc", "omega", "Omega", "a_mas")
    xy = onp.array(
        [
            onp.array(
                KeplerOrbit(
                    **{k: float(s[k][i]) for k in keys}, t_ref=T_REF
                ).relative(epoch)[:2]
            ).ravel()
            for i in range(len(s["period"]))
        ]
    )
    d = xy.mean(0) - truth_xy
    return float(onp.sqrt(d @ onp.linalg.solve(onp.cov(xy.T), d)))


def _median_offsets(common, rho, n_seeds=20):
    """Median offset over seeds of the tempered and the untempered fit of
    noise at chi2/N about 12 (eight channels, noise 3.5 times the errors)."""
    truth = onp.array(
        KeplerOrbit(**{**TRUTH, "period": 12.1}, t_ref=T_REF).relative(
            EPOCHS[0]
        )[:2]
    ).ravel()
    out, fs = {True: [], False: []}, []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for seed in range(n_seeds):
            data, positions = _nights(
                rho, seed=seed, period=12.1, common=common
            )
            for temper in out:
                result = aliases.fit_orbit_aliases(
                    data, (11.95, 12.25), positions=positions, t_ref=T_REF,
                    n_candidates=3, n_refine=1, n_is=100, n_samples=40,
                    primary_diam=None, temper=temper,
                )  # fmt: skip
                out[temper].append(_offset(result, truth, EPOCHS[0]))
                if temper:
                    fs.append(result.temper_f)
    med = {k: float(onp.median(v)) for k, v in out.items()}
    med["f"] = float(onp.median(fs))
    return med


@pytest.mark.slow
@pytest.mark.parametrize(
    "common, rho", [(False, 0.85), (True, 0.6)], ids=["ar1-0.85", "common-0.6"]
)
def test_the_tempered_posterior_is_calibrated_and_the_untempered_one_is_not(
    common, rho
):
    # 20 seeds of correlated noise at chi2/N about 12 (one AR(1) set at
    # rho = 0.85, one common-mode set at c = 0.6). The sky-plane offset of
    # the posterior mean from the truth, over the posterior spread, has a
    # median near 1.2 when calibrated. The untempered posterior is too
    # narrow by sqrt(f) (2.2 to 2.3): its median offset is well above 2.
    med = _median_offsets(common, rho)
    print(f"median offset/spread: tempered {med[True]:.2f}, "
          f"untempered {med[False]:.2f}")  # fmt: skip
    # the estimated f must be the true one (a lag-1 AR(1) estimate gives
    # 3.2 for common mode, truth 5.2, and a median offset near 1.5)
    truth_f = 1 + 7 * rho if common else _chain_factor(rho, 8)
    print(f"median f {med['f']:.2f}, truth {truth_f:.2f}")
    assert med["f"] == pytest.approx(truth_f, rel=0.25)
    assert med[True] < 1.45
    assert med[False] > 2.0


# --- the fit's use of the factors --------------------------------------------


class _Stop(Exception):
    pass


def _stub_fit(monkeypatch, data, positions, **kw):
    """Run ``fit_orbit_aliases`` up to the pre-flight check; the settings."""
    seen = {}

    def preflight(problem, settings, band):
        seen["settings"] = settings
        raise _Stop

    monkeypatch.setattr(aliases, "_preflight", preflight)
    monkeypatch.setattr(
        aliases,
        "position_profile",
        lambda pos, bands, **k: [dict(orbits=[]) for _ in bands],
    )
    monkeypatch.setattr(
        aliases,
        "_short_arc_profile",
        lambda data, seed, pos, bands, **k: (
            [dict(orbits=[]) for _ in bands],
            None,
            None,
        ),
    )
    with pytest.raises(_Stop), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        aliases.fit_orbit_aliases(
            data, (12.0, 13.0), positions=positions, t_ref=T_REF,
            n_random=5, **kw,
        )  # fmt: skip
    return seen["settings"]


def test_temper_must_be_a_finite_float_of_at_least_one():
    data, positions = _cached(0.0, n=3)
    for bad in (0.5, -1.0, 0.0, float("inf"), float("nan")):
        with pytest.raises(ValueError, match="temper must be"):
            aliases.fit_orbit_aliases(
                data, (12.0, 13.0), positions=positions, t_ref=T_REF,
                primary_diam=None, temper=bad,
            )  # fmt: skip


def test_a_fixed_f_tempers_the_resolution_test_and_does_not_warn(
    monkeypatch,
):
    data, positions = _cached(0.0, n=3)
    calls = []

    def auto(data, positions, vis_amp=None, block_dof=None):
        calls.append(block_dof)
        return None, positions, types.SimpleNamespace(resolved=False), {}

    monkeypatch.setattr(aliases, "_auto_primary", auto)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _stub_fit(
            monkeypatch, data, positions, primary_diam="auto", temper=8.0
        )
    assert calls == [{"phi": 0.125, "vis": 0.125}]
    # f = 8 is outside [0.5, 2], but the user chose it: no warning
    assert not [w for w in caught if "effective-channel" in str(w.message)]
    # untempered and estimated runs start from an untempered first pass
    calls.clear()
    _stub_fit(monkeypatch, data, positions, primary_diam="auto", temper=False)
    assert calls == [None]


def _scripted(monkeypatch, data, positions, resolved):
    """A fit whose ``_auto_primary`` resolves on the first (untempered)
    pass and answers ``resolved`` when rechecked on the tempered score, and
    whose factors are f = 4 at a disc and f = 3 at a point."""
    calls, pilots = [], []

    def auto(data, positions, vis_amp=None, block_dof=None):
        calls.append(block_dof)
        ok = block_dof is None or resolved
        est = types.SimpleNamespace(resolved=ok, diam=1.0)
        if not ok:  # a point: the positions are refined without a disc
            positions = EpochPositions(
                **{**positions.__dict__, "primary_diam": None}
            )
        return ("fit" if ok else None), positions, est, {}

    def factors(data, positions, diam, vis_amp, names):
        pilots.append(diam)
        f = 4.0 if diam else 3.0
        return tempering.ChannelFactors(
            f, dict.fromkeys(names, f), {}, {}, f_raw=f
        )

    monkeypatch.setattr(aliases, "_auto_primary", auto)
    monkeypatch.setattr(aliases, "_temper_factors", factors)
    settings = _stub_fit(monkeypatch, data, positions, primary_diam="auto")
    return calls, pilots, settings


def test_the_auto_primary_recheck_keeps_a_disc_that_survives_tempering(
    monkeypatch,
):
    data, positions = _cached(0.0, n=3)
    positions = EpochPositions(**{**positions.__dict__, "primary_diam": 1.0})
    calls, pilots, settings = _scripted(monkeypatch, data, positions, True)
    assert calls == [None, {"phi": 0.25, "vis": 0.25}]
    assert settings.primary_diam == "fit"
    assert dict(settings.dof) == {"phi": 0.25, "vis": 0.25}
    assert len(pilots) == 1 and pilots[0] == 1.0  # f estimated at the disc


def test_the_auto_primary_recheck_turns_a_tempered_away_disc_into_a_point(
    monkeypatch,
):
    data, positions = _cached(0.0, n=3)
    positions = EpochPositions(**{**positions.__dict__, "primary_diam": 1.0})
    calls, pilots, settings = _scripted(monkeypatch, data, positions, False)
    assert calls == [None, {"phi": 0.25, "vis": 0.25}]
    assert settings.primary_diam is None
    # f is re-estimated at the point primary (the disc's f no longer holds)
    assert len(pilots) == 2 and not pilots[1]
    assert dict(settings.dof) == pytest.approx(
        {"phi": 1 / 3, "vis": 1 / 3}, rel=1e-5
    )


def test_vis_and_phi_get_their_own_factors():
    rng = onp.random.default_rng(3)
    vis = 3.0 * _ar1(rng, (400, 8), 0.8).reshape(-1)
    phi = rng.standard_normal(400 * 8)
    chains = {"vis": _chains_of(400, 8), "phi": _chains_of(400, 8)}
    factors = tempering.estimate_factors([{"vis": vis, "phi": phi}], [chains])
    assert factors.blocks["vis"] > 3.0
    assert factors.blocks["phi"] == pytest.approx(1.0, abs=0.35)
    pairs = dict(aliases._dof_pairs(factors))
    assert pairs["vis"] < 0.35 < 0.7 < pairs["phi"]
    # six significant figures: an estimate's last digits do not matter
    assert all(v == float(f"{v:.6g}") for v in pairs.values())


def test_blocks_without_chains_of_their_own_warn_and_the_largest_block_flags():
    one = tempering.ChannelFactors(
        1.5, {"vis": 1.5, "phi": 3.0, "extra[0]": 1.5}, {}, {}, f_raw=1.0,
        fallback=("extra[0]",),
    )  # fmt: skip
    with pytest.warns(UserWarning) as caught:
        aliases._warn_tempering(one)
    text = " ".join(str(w.message) for w in caught)
    assert "block 'phi'" in text  # the largest block, not the pooled 1.5
    assert "extra[0]" in text and "pooled" in text
    # the pooled f alone is inside the range: nothing to warn about
    fine = tempering.ChannelFactors(
        1.5, {"vis": 1.5, "phi": 1.5}, {}, {}, f_raw=1.0
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        aliases._warn_tempering(fine)
    flat = tempering.ChannelFactors(
        1.5, {"vis": 1.5}, {}, {}, f_raw=1.0, flat=("vis",)
    )
    with pytest.warns(UserWarning, match="flat across channels"):
        aliases._warn_tempering(flat)


def test_large_f_reads_as_an_inadequate_noise_model_only_with_gains():
    big = tempering.ChannelFactors(5.0, {"vis": 5.0}, {}, {}, f_raw=5.0)
    with pytest.warns(UserWarning, match="noise model inadequate"):
        aliases._warn_tempering(big, calibrated=True)
    with pytest.warns(UserWarning, match="effective-channel factor") as caught:
        aliases._warn_tempering(big)
    assert "inadequate" not in " ".join(str(w.message) for w in caught)


def test_the_flag_follows_the_largest_block_and_the_flat_signature():
    def flagged(**kw):
        return AliasResult(
            T_REF, onp.zeros(2), 1.0, [_band(3)], {}, **kw
        ).tempering

    assert flagged(temper_f=1.5, temper_blocks={"vis": 1.4, "phi": 3.0})
    assert not flagged(temper_f=1.5, temper_blocks={"vis": 1.4, "phi": 1.6})
    assert flagged(
        temper_f=1.5, temper_blocks={"vis": 1.4},
        temper_rho={"vis": [0.6, 0.6, 500]},
    )  # fmt: skip
    assert not flagged(
        temper_f=1.5, temper_blocks={"vis": 1.4},
        temper_rho={"vis": [0.6, 0.36, 500]},
    )  # fmt: skip
    # AR(1) blocks of different rho: the pooled values alone would look flat
    assert not flagged(
        temper_f=1.5, temper_blocks={"vis": 1.4, "phi": 1.0},
        temper_rho={
            "vis": [0.85, 0.7225, 500], "phi": [0.0, 0.0, 500],
            "pooled": [0.425, 0.36, 1000],
        },
    )  # fmt: skip


def test_the_block_dof_reproduces_the_tempering_in_marginal_loglike():
    from virgil.epochs import marginal_loglike

    data, _ = _cached(0.85, n=1)
    result = AliasResult(
        T_REF, onp.zeros(2), 1.0, [_band(3)], {}, temper_f=4.0,
        temper_blocks={"vis": 4.0, "phi": 2.0},
    )  # fmt: skip
    assert result.block_dof() == {"vis": 0.25, "phi": 0.5}
    scene = OrbitalBinary(KeplerOrbit(**TRUTH, t_ref=T_REF), FLUX)
    with jax.enable_x64(True):
        d = cast_tree(data[0], "float64")
        plain = marginal_loglike(scene, d)
        both = marginal_loglike(scene, d, block_dof=result.block_dof())
        vis = marginal_loglike(scene, d, block_dof={"vis": 0.25, "phi": 1.0})
        phi = marginal_loglike(scene, d, block_dof={"vis": 1.0, "phi": 0.5})
        quarter = marginal_loglike(scene, d, dof=0.25)
        # blocks are additive, so the tempered sum is the sum of the parts
        assert float(both) == pytest.approx(
            float(vis) + float(phi) - float(plain), rel=1e-10
        )
        assert float(quarter) == pytest.approx(float(plain) / 4, rel=1e-10)
        assert float(both) != pytest.approx(float(plain))


@pytest.mark.slow
def test_a_checkpoint_with_tempering_on_roundtrips_and_depends_on_f(
    monkeypatch, tmp_path
):
    data, positions = _cached(0.85, period=12.1)
    calls = []
    real = aliases._fit_band

    def fit_band(problem, settings, band, orbits, times):
        calls.append(band.n)
        return real(problem, settings, band, orbits, times)

    monkeypatch.setattr(aliases, "_fit_band", fit_band)
    kw = dict(
        positions=positions, t_ref=T_REF, n_random=10, n_refine=1,
        max_steps=3, n_is=20, n_samples=5, primary_diam=None,
        checkpoint_dir=tmp_path,
    )  # fmt: skip
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        first = aliases.fit_orbit_aliases(data, (11.95, 12.25), **kw)
        assert calls and first.temper_f > 2.0
        n_first = len(calls)
        again = aliases.fit_orbit_aliases(data, (11.95, 12.25), **kw)
        assert len(calls) == n_first  # every band loaded from its checkpoint
        assert again.temper_f == first.temper_f
        # another f is another likelihood: the checkpoints do not apply
        aliases.fit_orbit_aliases(
            data, (11.95, 12.25), **{**kw, "temper": 3.0}
        )
        assert len(calls) == 2 * n_first
    assert aliases._CHECKPOINT_FORMAT >= 4


def test_the_short_arcs_see_the_tempered_gaps_and_errors(monkeypatch):
    # With a fixed f = 4, a night decisive by 12 nats untempered is only 3
    # nats decisive: the short-arc stage must not run on it (a spurious
    # peak, decisive only by the independent-channel surface, would cut the
    # true band), and when it runs it widens the errors by f.
    import dataclasses

    data, positions = _cached(0.0, n=3)
    seen = []

    def arcs_stage(data, seed, pos, bands, **kw):
        seen.append(kw["cov_scale"])
        raise _Stop

    monkeypatch.setattr(aliases, "_short_arc_profile", arcs_stage)
    monkeypatch.setattr(
        aliases,
        "position_profile",
        lambda pos, bands, **k: [dict(orbits=[]) for _ in bands],
    )
    monkeypatch.setattr(
        aliases, "_preflight", lambda *a: (_ for _ in ()).throw(_Stop)
    )

    def run(gap, f):
        found = dataclasses.replace(
            positions, gap_marginal=onp.full(len(positions.names), gap)
        )
        with pytest.raises(_Stop), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            aliases.fit_orbit_aliases(
                data, (12.0, 13.0), positions=found, t_ref=T_REF,
                primary_diam=None, n_random=5, temper=f,
            )  # fmt: skip

    run(12.0, 4.0)  # 12 / 4 = 3 nats < 5: no arcs stage
    assert seen == []
    run(40.0, 4.0)  # 10 nats tempered
    assert seen == [4.0]
    run(12.0, 1.0)  # untempered: as before
    assert seen == [4.0, 1.0]


def test_temper_factors_normalizes_by_night_scale():
    # Night 0 has white noise inflated 20x; night 1 has correlated channels
    # (rho = 0.85) at scale 1. Per-night scale normalization prevents the
    # high-chi2 white night from drowning out night 1's correlation.
    (d_white, _), _ = _nights(0.0, scale=20.0, seed=10, n=2)
    (_, d_corr), pos = _nights(0.85, scale=1.0, seed=11, n=2)
    with jax.enable_x64(True):
        data = [cast_tree(d_white, "float64"), cast_tree(d_corr, "float64")]
        factors = aliases._temper_factors(
            data, pos, None, None, ["vis", "phi"]
        )
    assert factors.f > 2.0
    assert factors.blocks["vis"] > 2.0 and factors.blocks["phi"] > 2.0
