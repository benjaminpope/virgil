"""Named epochs, per-dataset snapshots and multi-epoch orbit fits to
visibilities and closure phases."""

import warnings

import equinox as eqx
import jax
import jax.numpy as np
import numpy as onp
import numpyro.distributions as dist
import pytest

pytest.importorskip("jaxoplanet")

from virgil.epochs import (  # noqa: E402
    Epochs,
    chain_starts,
    rank_orbits,
    start_from_positions,
)
from virgil.fitting import fit  # noqa: E402
from virgil.likelihood import (  # noqa: E402
    chain_init_params,
    model_loglike,
    numpyro_model,
)
from virgil.models import (  # noqa: E402
    Attached,
    BinaryModelCartesian,
    OrbitalBinary,
    PointSource,
    System,
    TruncatedCone,
)
from virgil.oidata import OIData, cp_indices  # noqa: E402
from virgil.orbits import (  # noqa: E402
    KeplerOrbit,
    PositionData,
    starting_orbits,
)
from virgil.simulate import simulate  # noqa: E402

T_REF = 60500.0
TRUTH = dict(
    period=700.0,
    dt_peri=-120.0,
    ecc=0.3,
    inc=50.0,
    omega=60.0,
    Omega=120.0,
    a_mas=20.0,
)
FLUX = 0.1
STATIONS = onp.array([[0.0, 0.0], [60.0, 5.0], [25.0, 70.0], [-40.0, 45.0]])
PAIRS = onp.array([[1, 2], [1, 3], [1, 4], [2, 3], [2, 4], [3, 4]])
TRIANGLES = onp.array([[1, 2, 3], [1, 2, 4], [1, 3, 4], [2, 3, 4]])


def _night(mjd, frames=1, rotation=0.0, d_vis=0.01, d_phi=0.5):
    """VLTI-like data: ``frames`` four-telescope snapshots 1 h apart."""
    delta = STATIONS[PAIRS[:, 1] - 1] - STATIONS[PAIRS[:, 0] - 1]
    i1, i2, i3 = cp_indices(PAIRS, TRIANGLES)
    n = len(PAIRS)
    angles = onp.deg2rad(rotation + 15.0 * onp.arange(frames))
    u = onp.concatenate(
        [delta[:, 0] * onp.cos(a) - delta[:, 1] * onp.sin(a) for a in angles]
    )
    v = onp.concatenate(
        [delta[:, 0] * onp.sin(a) + delta[:, 1] * onp.cos(a) for a in angles]
    )
    shift = lambda i: onp.concatenate([i + k * n for k in range(frames)])  # noqa: E731
    return OIData(
        {
            "u": u,
            "v": v,
            "wavel": 2.2e-6,
            "vis": onp.ones(u.size),
            "d_vis": onp.full(u.size, d_vis),
            "phi": onp.zeros(len(i1) * frames),
            "d_phi": onp.full(len(i1) * frames, d_phi),
            "i_cps1": shift(i1),
            "i_cps2": shift(i2),
            "i_cps3": shift(i3),
            "mjd": onp.repeat(mjd + onp.arange(frames) / 24.0, n),
        }
    )


def _orbit(**changes):
    return KeplerOrbit(**{**TRUTH, **changes}, t_ref=T_REF)


# ---------------------------------------------------------------------------
# Names and times
# ---------------------------------------------------------------------------


def test_epochs_and_datasets_are_named_and_ordered_epoch_by_epoch():
    epochs = Epochs(
        {
            "a": {"ut": _night(60100.0), "at": _night(60101.0)},
            "b": [_night(60300.0), _night(60300.5)],
            "c": _night(60500.0),
        }
    )
    assert epochs.names == ("a", "b", "c")
    assert epochs.dataset_names == ("ut", "at", "b[0]", "b[1]", "c")
    assert epochs.epoch_of == ("a", "a", "b", "b", "c")
    assert len(epochs) == 5 and len(epochs.data) == 5
    assert epochs.index("b[1]") == 3
    with pytest.raises(KeyError, match="No dataset"):
        epochs.index("d")


@pytest.mark.parametrize(
    "groups",
    [
        {"a": {"x": None, "y": None}, "b": {"x": None}},  # two datasets
        {"a": {"b": None}, "b": None},  # a dataset named like another epoch
    ],
)
def test_names_must_be_unique(groups):
    groups = {
        e: (
            {k: _night(60000.0) for k in g}
            if isinstance(g, dict)
            else _night(60000.0)
        )
        for e, g in groups.items()
    }
    with pytest.raises(ValueError, match="unique"):
        Epochs(groups)


def test_snapshot_times_per_dataset_per_epoch_or_given():
    ut, at = _night(60100.0, frames=3), _night(60101.0, frames=3)
    by_dataset = Epochs({"a": {"ut": ut, "at": at}})
    onp.testing.assert_allclose(
        by_dataset.times, [60100.0 + 1 / 24, 60101.0 + 1 / 24], atol=1e-5
    )
    onp.testing.assert_allclose(by_dataset.spread_days, 1 / 24, atol=1e-5)
    by_epoch = Epochs({"a": {"ut": ut, "at": at}}, at="epoch")
    onp.testing.assert_allclose(by_epoch.times, 60100.5 + 1 / 24, atol=1e-5)
    given = Epochs({"a": {"ut": ut, "at": at}}, times={"a": 1.0, "at": 2.0})
    onp.testing.assert_array_equal(given.times, [1.0, 2.0])
    with pytest.raises(ValueError, match="unknown"):
        Epochs({"a": ut}, times={"b": 1.0})


def test_noise_is_keyed_by_name_with_datasets_overriding_epochs():
    epochs = Epochs(
        {
            "a": {"ut": _night(60100.0), "at": _night(60101.0)},
            "b": _night(60500.0),
        }
    )
    noise = epochs.noise(
        {"a": {"vis_scale": 2.0, "phi_scale": 1.5}, "at": {"vis_scale": 3.0}}
    )
    assert noise == [
        {"vis_scale": 2.0, "phi_scale": 1.5},
        {"vis_scale": 3.0, "phi_scale": 1.5},
        {},
    ]
    with pytest.raises(KeyError, match="unknown"):
        epochs.noise({"c": {"vis_scale": 2.0}})


# ---------------------------------------------------------------------------
# Snapshots
# ---------------------------------------------------------------------------


def test_an_orbital_binary_snapshot_is_the_static_binary_and_the_system():
    orbit = _orbit()
    mjd = T_REF + 150.0
    with jax.enable_x64(True):
        snap = OrbitalBinary(orbit, FLUX).at(mjd)
        assert isinstance(snap, BinaryModelCartesian)
        dra, ddec, _ = orbit.relative(mjd)
        onp.testing.assert_allclose([snap.dra, snap.ddec], [dra, ddec])
        system = System(
            primary=PointSource(), comp=Attached(PointSource(FLUX), orbit)
        ).at(mjd)
        data = _night(mjd)
        onp.testing.assert_allclose(
            data.model(snap), data.model(system), atol=1e-12
        )


def test_snapshots_match_the_per_sample_model_for_one_time_per_dataset():
    orbit = _orbit()
    epochs = Epochs(
        {f"e{k}": _night(T_REF + t) for k, t in enumerate([0, 200, 450])}
    )
    with jax.enable_x64(True):
        scene = OrbitalBinary(orbit, FLUX)
        per_sample = sum(model_loglike(scene, d) for d in epochs.data)
        onp.testing.assert_allclose(
            epochs.loglike(scene), per_sample, rtol=1e-6
        )


def test_snapshots_approximate_a_slowly_moving_scene_within_a_night():
    """The snapshot's loss of likelihood is bounded by the orbital motion.

    The data are noiseless, so the per-sample model fits them exactly and
    the snapshot loses ``0.5 * dchi2``. A companion of flux ratio f that
    moves by d (mas) shifts the fringe phase on a baseline of spatial
    frequency b (cycles/mas) by at most 2 pi b d, so its complex visibility
    by at most f/(1+f) of that, V^2 by at most 2|dV| + |dV|^2, and each
    baseline's phase by at most f/(1-f) of it (three in a closure phase).
    """
    orbit = _orbit()
    data = simulate(
        System(primary=PointSource(), comp=Attached(PointSource(FLUX), orbit)),
        _night(T_REF + 200.0, frames=3),
    )
    epochs = Epochs({"night": data})
    with jax.enable_x64(True):
        scene = OrbitalBinary(orbit, FLUX)
        lost = float(model_loglike(scene, data) - epochs.loglike(scene))
        dra, ddec, _ = orbit.relative(data.mjd)
        dra0, ddec0, _ = orbit.relative(epochs.times[0])
    moved = onp.hypot(onp.asarray(dra) - dra0, onp.asarray(ddec) - ddec0)
    mas = onp.pi / 180 / 3.6e6
    b = onp.hypot(data.u, data.v) / onp.asarray(data.wavel) * mas
    dpsi = 2 * onp.pi * b * moved
    dv = FLUX / (1 + FLUX) * dpsi
    dphase = FLUX / (1 - FLUX) * dpsi
    dcp = sum(
        dphase[onp.asarray(i)] for i in (data.i_cps1, data.i_cps2, data.i_cps3)
    )
    dchi2 = onp.sum(((2 * dv + dv**2) / onp.asarray(data.d_vis)) ** 2)
    dchi2 += onp.sum((dcp / onp.asarray(data.d_phi)) ** 2)
    assert epochs.spread_days[0] == pytest.approx(1 / 24, rel=1e-3)
    assert 0.0 <= lost <= 0.5 * dchi2
    assert 0.5 * dchi2 < 0.1  # the approximation is tight here


def test_mjd_times_with_a_default_t_ref_warn():
    orbit = KeplerOrbit(**TRUTH)  # t_ref = 0
    with jax.enable_x64(True):
        with pytest.warns(UserWarning, match="t_ref"):
            OrbitalBinary(orbit, FLUX).at(T_REF)
        with pytest.warns(UserWarning, match="t_ref"):
            orbit.relative(T_REF)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            _orbit().relative(T_REF)
            orbit.relative(300.0)  # times counted from 0 on purpose


# ---------------------------------------------------------------------------
# A cone bound to the orbit
# ---------------------------------------------------------------------------


def _bound_cone(Omega, skew):
    cone = TruncatedCone(
        tip=1.0,
        alpha=40.0,
        s0=1.0,
        length=3.0,
        width=0.5,
        flux=0.5,
        n_rings=8,
    )
    return Attached(
        cone,
        _orbit(Omega=Omega),
        anchor="primary",
        bind={"pa": "towards_primary", "tilt": "line_tilt"},
        offsets={"pa": skew},
    )


def test_a_bound_cone_follows_the_orbit_with_traced_angles_under_jit():
    mjd = T_REF + 150.0
    data = _night(mjd)
    with jax.enable_x64(True):
        snap = jax.jit(lambda O, s: _bound_cone(O, s).at(mjd))(120.0, 5.0)
        frame = _orbit().frame(mjd)
        assert float(snap.pa) == pytest.approx(
            float(frame["towards_primary"]) + 5.0
        )
        assert float(snap.tilt) == pytest.approx(float(frame["line_tilt"]))
        assert float(snap.dra) == 0.0 and float(snap.ddec) == 0.0
        # An optically thin cone is the same at +tilt and -tilt (mirror
        # images through the sky plane): a bound tilt sees |line_tilt|.
        assert abs(float(snap.tilt)) > 1.0
        mirror = eqx.tree_at(lambda c: c.tilt, snap, -snap.tilt)
        onp.testing.assert_allclose(
            data.model(System(star=PointSource(), cone=mirror)),
            data.model(System(star=PointSource(), cone=snap)),
            atol=1e-12,
        )

        def loglike(O, s):
            scene = System(
                star=PointSource(),
                cone=_bound_cone(O, s),
                comp=Attached(PointSource(FLUX), _orbit(Omega=O)),
            )
            return Epochs({"night": data}).loglike(scene)

        grad = jax.jit(jax.grad(loglike, argnums=(0, 1)))(120.0, 5.0)
        assert all(onp.isfinite(g) and g != 0.0 for g in grad)


# ---------------------------------------------------------------------------
# Recovery of a known orbit from multi-epoch visibilities
# ---------------------------------------------------------------------------

NAMES = tuple(TRUTH) + ("flux",)


def _scene(period, dt_peri, ecc, inc, omega, Omega, a_mas, flux):
    orbit = KeplerOrbit(
        period, dt_peri, ecc, inc, omega, Omega, a_mas, t_ref=T_REF
    )
    return OrbitalBinary(orbit, flux)


def test_a_known_orbit_is_recovered_from_multi_epoch_visibilities():
    times = [-300.0, -150.0, 0.0, 120.0, 260.0, 400.0]
    truth = {**TRUTH, "flux": FLUX}
    keys = jax.random.split(jax.random.PRNGKey(3), len(times))
    with jax.enable_x64(True):
        snapshots = Epochs(
            {
                f"e{k}": _night(T_REF + t, rotation=30.0 * k)
                for k, t in enumerate(times)
            }
        ).snapshots(_scene(**truth))
        epochs = Epochs(
            {
                f"e{k}": simulate(
                    snap, _night(T_REF + t, rotation=30.0 * k), key
                )
                for k, (t, snap, key) in enumerate(zip(times, snapshots, keys))
            }
        )
        priors = {
            "period": dist.LogUniform(300.0, 3000.0),
            "dt_peri": dist.Uniform(-600.0, 600.0),
            "ecc": dist.Uniform(0.0, 0.9),
            "inc": dist.Uniform(0.0, 180.0),
            "omega": dist.Uniform(-180.0, 360.0),
            "Omega": dist.Uniform(-180.0, 360.0),
            "a_mas": dist.LogUniform(2.0, 100.0),
            "flux": dist.LogUniform(1e-3, 1.0),
        }
        # Fringe aliases make the likelihood multimodal at the ~λ/B scale
        # (5 mas here), so start within a fraction of it, as a fit of
        # positions would.
        start = {
            "period": 705.0,
            "dt_peri": -117.0,
            "ecc": 0.29,
            "inc": 49.0,
            "omega": 61.0,
            "Omega": 119.5,
            "a_mas": 19.8,
            "flux": 0.095,
        }
        result = fit(
            epochs.model_fn(_scene),
            priors,
            epochs.data,
            init=start,
            method="lm",
        )
        best = onp.array([float(result.values[n]) for n in NAMES])

        def nll(x):
            return -epochs.loglike(_scene(*x))

        cov = onp.linalg.inv(onp.asarray(jax.hessian(nll)(np.asarray(best))))
        sigma = onp.sqrt(onp.diag(cov))
        assert onp.all(onp.isfinite(sigma)) and onp.all(sigma > 0)
        pulls = (best - onp.array([truth[n] for n in NAMES])) / sigma
        assert onp.all(onp.abs(pulls) < 4.0), dict(zip(NAMES, pulls))
        # A good fit: chi2 about the number of data.
        assert result.info["chi2_red"] < 2.0


def test_snapshots_work_in_float32():
    """Snapshot times are concrete float64 MJDs, so float32 keeps the phase."""
    epochs = Epochs({"a": _night(T_REF), "b": _night(T_REF + 300.0)})
    with jax.enable_x64(False):
        loglike = epochs.loglike(_scene(**TRUTH, flux=FLUX))
    with jax.enable_x64(True):
        exact = epochs.loglike(_scene(**TRUTH, flux=FLUX))
    assert float(loglike) == pytest.approx(float(exact), rel=1e-3, abs=0.1)


# ---------------------------------------------------------------------------
# Starting and sampling (stage 2)
# ---------------------------------------------------------------------------

STAGE2_TIMES = [-300.0, -150.0, 0.0, 120.0, 260.0, 400.0]


def _simulated_epochs(times=STAGE2_TIMES, seed=None, frames=1):
    """Epochs of the true binary, noiseless unless ``seed`` is given."""
    nights = [
        _night(T_REF + t, rotation=30.0 * k, frames=frames, d_phi=0.02)
        for k, t in enumerate(times)
    ]
    with jax.enable_x64(True):
        snapshots = Epochs(
            {f"e{k}": n for k, n in enumerate(nights)}
        ).snapshots(_scene(**TRUTH, flux=FLUX))
        keys = (
            [None] * len(times)
            if seed is None
            else list(jax.random.split(jax.random.PRNGKey(seed), len(times)))
        )
        return Epochs(
            {
                f"e{k}": simulate(snap, night, key)
                for k, (snap, night, key) in enumerate(
                    zip(snapshots, nights, keys)
                )
            }
        )


def _binary(orbit):
    return OrbitalBinary(orbit, FLUX)


def test_rank_orbits_puts_the_truth_first_among_starting_orbits():
    epochs = _simulated_epochs()
    with jax.enable_x64(True):
        dra, ddec, _ = _orbit().relative(epochs.times)
    offsets = onp.random.default_rng(1).normal(0.0, 0.3, (2, len(epochs)))
    positions = PositionData(
        epochs.times,
        onp.asarray(dra) + offsets[0],
        onp.asarray(ddec) + offsets[1],
        onp.tile(0.3**2 * onp.eye(2), (len(epochs), 1, 1)),
        t_ref=T_REF,
    )
    candidates = starting_orbits(
        positions, onp.geomspace(400.0, 2000.0, 20), n_phase=24, n_best=30
    )
    ranked = rank_orbits(_binary, epochs, candidates + [(_orbit(), 0.0)])
    assert len(ranked) == 31
    assert ranked.order[0] == 30  # the true orbit
    assert sorted(ranked.order) == list(range(31))
    assert onp.all(onp.diff(ranked.loglike) <= 0)
    with jax.enable_x64(True):
        exact = float(epochs.loglike(_binary(_orbit())))
    assert ranked.loglike[0] == pytest.approx(exact, rel=1e-6)
    orbit, loglike = ranked[0]
    assert orbit is ranked.best and loglike == ranked.loglike[0]
    # One orbit with a leading axis ranks like the list of its parts.
    batched = jax.tree_util.tree_map(
        lambda *x: np.stack(x), *[o for o, _ in candidates[:5]]
    )
    again = rank_orbits(_binary, epochs, batched)
    onp.testing.assert_allclose(
        again.loglike, onp.sort(ranked.loglike[ranked.order < 5])[::-1]
    )


def test_chain_starts_are_distinct_modes():
    epochs = _simulated_epochs(STAGE2_TIMES[:4])
    orbits = [
        _orbit(),
        _orbit(period=700.01),  # the same mode
        _orbit(omega=240.0, Omega=300.0),  # the mirror: the same sky motion
        _orbit(Omega=150.0),
        _orbit(a_mas=12.0),
    ]
    ranked = rank_orbits(_binary, epochs, orbits)
    starts = chain_starts(ranked, 3)
    assert len(starts) == 3
    assert starts.order[0] in (0, 2)
    assert set(starts.order[1:]) == {3, 4}
    positions = starts.positions()
    for i in range(3):
        for j in range(i):
            offset = positions[i] - positions[j]
            moved = onp.hypot(offset[:, 0], offset[:, 1]).max()
            assert moved > 0.5 * epochs.resolution_mas
    # Fewer distinct modes than chains: the modes repeat in turn.
    more = chain_starts(ranked, 5)
    onp.testing.assert_array_equal(more.order[3:], starts.order[:2])
    with pytest.raises(ValueError, match="n_chains"):
        chain_starts(ranked, 0)


STAGE2_PRIORS = {
    "period": dist.LogUniform(300.0, 3000.0),
    "dt_peri": dist.Uniform(-1500.0, 1500.0),
    "ecc": dist.Uniform(0.0, 0.9),
    "inc": dist.Uniform(0.0, 180.0),
    "omega": dist.Uniform(-180.0, 360.0),
    "Omega": dist.Uniform(-180.0, 360.0),
    "a_mas": dist.LogUniform(2.0, 100.0),
    "flux": dist.LogUniform(1e-3, 1.0),
}


def _start_values(orbit, flux):
    period = float(orbit.period)
    dt_peri = float(orbit.dt_peri)
    return {
        "period": period,
        "dt_peri": dt_peri - period * onp.round(dt_peri / period),
        "ecc": float(onp.clip(orbit.ecc, 0.01, 0.85)),
        "inc": float(onp.clip(orbit.inc, 1.0, 179.0)),
        "omega": float(onp.mod(orbit.omega, 360.0)),
        "Omega": float(onp.mod(orbit.Omega, 360.0)),
        "a_mas": float(orbit.a_mas),
        "flux": flux,
    }


def test_a_positions_fit_starts_where_a_default_start_fails():
    epochs = _simulated_epochs(STAGE2_TIMES[:5], seed=5, frames=4)
    axis = onp.arange(-30.0, 30.5, 2.0)
    start = start_from_positions(
        _scene,
        STAGE2_PRIORS,
        epochs,
        _start_values,
        grid={"dra": axis, "ddec": axis, "flux": [0.05, 0.1, 0.2]},
        periods=onp.geomspace(400.0, 2000.0, 10),
        t_ref=T_REF,
        n_phase=24,
        n_candidates=20,
        n_refine=2,
        method="lm",
    )
    with jax.enable_x64(True):
        true_dra, true_ddec, _ = _orbit().relative(epochs.times)
    found = onp.hypot(
        start.positions.dra - onp.asarray(true_dra),
        start.positions.ddec - onp.asarray(true_ddec),
    )
    assert onp.all(found < 0.5), found
    assert onp.all(start.positions.gap > 5.0)
    assert len(start.candidates) == 20
    assert start.best.info["chi2_red"] < 2.0
    with jax.enable_x64(True):
        best = _scene(**{k: start.best.values[k] for k in NAMES})
        fitted = onp.array(
            [onp.asarray(s.dra) for s in epochs.snapshots(best)]
        )
    onp.testing.assert_allclose(fitted, onp.asarray(true_dra), atol=0.3)
    # A start far from the truth, in the middle of the priors, sticks in
    # another minimum of the multimodal likelihood.
    default = fit(
        epochs.model_fn(_scene),
        STAGE2_PRIORS,
        epochs.data,
        init={
            "period": 1000.0,
            "dt_peri": 0.0,
            "ecc": 0.45,
            "inc": 90.0,
            "omega": 90.0,
            "Omega": 90.0,
            "a_mas": 10.0,
            "flux": 0.05,
        },
        method="lm",
    )
    assert default.info["loss"] > start.best.info["loss"] + 100.0
    values = start.chain_values(3)
    assert len(values) == 3 and set(NAMES) <= set(values[0])
    assert values[0] == dict(start.best.values)


def test_each_chain_starts_at_its_own_values():
    from numpyro.infer import MCMC, NUTS
    from numpyro.infer.util import constrain_fn

    epochs = _simulated_epochs(STAGE2_TIMES[:3], seed=7)
    truth = {**TRUTH, "flux": FLUX}
    starts = [truth, {**truth, "period": 710.0, "Omega": 121.0}]
    with jax.enable_x64(True):
        posterior = numpyro_model(
            epochs.model_fn(_scene), STAGE2_PRIORS, epochs.data
        )
        init = chain_init_params(posterior, starts, jax.random.PRNGKey(0))
        assert init["period"].shape == (2,)
        for k, values in enumerate(starts):
            back = constrain_fn(
                posterior, (), {}, {n: z[k] for n, z in init.items()}
            )
            for name in NAMES:
                assert float(back[name]) == pytest.approx(
                    values[name], rel=1e-6
                )
        # A 20-step smoke test of NUTS from the second start; one chain
        # keeps it cheap, and a real recovery run is an OzSTAR job.
        mcmc = MCMC(
            NUTS(posterior),
            num_warmup=10,
            num_samples=10,
            progress_bar=False,
        )
        mcmc.run(
            jax.random.PRNGKey(1),
            init_params=chain_init_params(posterior, starts[1:]),
        )
        samples = mcmc.get_samples()
    assert samples["period"].shape == (10,)
    assert all(onp.all(onp.isfinite(v)) for v in samples.values())
