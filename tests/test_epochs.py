"""Named epochs, per-dataset snapshots and multi-epoch orbit fits to
visibilities and closure phases."""

import dataclasses
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
    OrbitStart,
    _loss_order,
    _warn_if_coarse,
    chain_starts,
    epoch_positions,
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
    UniformDisk,
)
from virgil.orbits import (  # noqa: E402
    KeplerOrbit,
    PositionData,
    starting_orbits,
)
from virgil.simulate import simulate  # noqa: E402
from tests._shared import vlti_night  # noqa: E402

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

_night = vlti_night


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


def _per_epoch_loglike(epochs, scene):
    """The loglike with one ``at`` per dataset (the pre-vmap path)."""
    return sum(
        model_loglike(scene.at(float(t)), d)
        for t, d in zip(epochs.times, epochs.data)
    )


def _snapshot_scenes():
    orbit = _orbit()
    return {
        "binary": lambda **p: OrbitalBinary(
            orbit.set("period", p["period"]), FLUX
        ),
        "system": lambda **p: System(
            primary=PointSource(),
            comp=Attached(PointSource(FLUX), orbit.set("period", p["period"])),
        ),
    }


@pytest.mark.parametrize("name", ["binary", "system"])
def test_vmapped_snapshots_match_one_at_per_dataset(name):
    scene_fn = _snapshot_scenes()[name]
    epochs = Epochs(
        {f"e{k}": _night(T_REF + t) for k, t in enumerate([0, 200, 450])}
    )

    def new(period):
        return epochs.loglike(scene_fn(period=period))

    def old(period):
        return _per_epoch_loglike(epochs, scene_fn(period=period))

    with jax.enable_x64(True):
        period = np.asarray(TRUTH["period"] * 1.02)
        value_new, grad_new = jax.value_and_grad(new)(period)
        value_old, grad_old = jax.value_and_grad(old)(period)
        onp.testing.assert_allclose(value_new, value_old, rtol=1e-10)
        onp.testing.assert_allclose(grad_new, grad_old, rtol=1e-8)
        scene = scene_fn(period=period)
        for snap, t in zip(epochs.snapshots(scene), epochs.times):
            for a, b in zip(
                jax.tree_util.tree_leaves(snap),
                jax.tree_util.tree_leaves(scene.at(float(t))),
            ):
                onp.testing.assert_allclose(a, b, rtol=1e-10)


def test_vmapped_snapshots_still_warn_about_mjds_without_a_t_ref():
    epochs = Epochs({"a": _night(T_REF), "b": _night(T_REF + 50.0)})
    with jax.enable_x64(True):
        with pytest.warns(UserWarning, match="t_ref"):
            epochs.snapshots(OrbitalBinary(KeplerOrbit(**TRUTH), FLUX))


def test_vmapped_snapshots_shrink_the_gradient_graph():
    scene_fn = _snapshot_scenes()["binary"]
    epochs = Epochs(
        {f"e{k}": _night(T_REF + t) for k, t in enumerate([0, 100, 200, 300])}
    )
    new = jax.make_jaxpr(
        jax.value_and_grad(lambda p: epochs.loglike(scene_fn(period=p)))
    )(np.asarray(TRUTH["period"]))
    old = jax.make_jaxpr(
        jax.value_and_grad(
            lambda p: _per_epoch_loglike(epochs, scene_fn(period=p))
        )
    )(np.asarray(TRUTH["period"]))
    assert len(new.eqns) < len(old.eqns)


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


@pytest.mark.parametrize("scales", ["quoted", "marginal"])
def test_rank_orbits_compiles_once_per_layout(scales):
    from virgil import epochs as epochs_module

    kernel = epochs_module._loglike_kernel
    candidates = [_orbit(), _orbit(period=410.0), _orbit(ecc=0.5)]
    rank = lambda epochs, **kw: rank_orbits(  # noqa: E731
        _binary, epochs, candidates, scales=scales, **kw
    )
    first = rank(_simulated_epochs(seed=1))
    size = kernel._cache_size()
    # New data, or a new dof, of the same layout: no new compile, and the
    # reused trace still answers for them.
    second = rank(_simulated_epochs(seed=2))
    halved = rank(_simulated_epochs(seed=1), dof=0.5)
    again = rank(_simulated_epochs(seed=1))
    assert kernel._cache_size() == size
    assert not onp.allclose(first.loglike, second.loglike)
    assert scales == "quoted" or not onp.allclose(
        first.loglike, halved.loglike
    )
    onp.testing.assert_array_equal(first.loglike, again.loglike)
    if scales == "marginal":
        # A different s_max is a different layout: it recompiles.
        bounded = rank(_simulated_epochs(seed=1), s_max=3.0)
        assert kernel._cache_size() == size + 1
        assert not onp.allclose(first.loglike, bounded.loglike)


def test_start_from_positions_model_wrapper_shares_one_compile():
    from virgil import epochs as epochs_module

    from virgil.epochs import _stack

    def model(orbit):
        return _binary(orbit)

    stacked = _stack([_orbit(), _orbit(ecc=0.5)])
    epochs = _simulated_epochs(seed=1)
    kernel = epochs_module._loglike_kernel
    # start_from_positions wraps its model afresh on every call.
    run = lambda: epochs_module._loglikes(  # noqa: E731
        epochs_module._Kwargs(model), {"orbit": stacked}, epochs, None, None,
        "float64", scales="quoted",
    )  # fmt: skip
    run()
    size = kernel._cache_size()
    run()
    assert kernel._cache_size() == size


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
    ranked = rank_orbits(
        _binary, epochs, candidates + [(_orbit(), 0.0)], scales="quoted"
    )
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
    again = rank_orbits(_binary, epochs, batched, scales="quoted")
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
    ranked = rank_orbits(_binary, epochs, orbits, scales="quoted")
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


@pytest.mark.slow
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
        scales="quoted",
        method="lm",
    )
    with jax.enable_x64(True):
        true_dra, true_ddec, _ = _orbit().relative(epochs.times)
    found = onp.hypot(
        start.positions.dra - onp.asarray(true_dra),
        start.positions.ddec - onp.asarray(true_ddec),
    )
    assert onp.all(found < 0.5), found
    # On this coarse (2 mas) grid one night's marginal gap is about 3,
    # so it does not seed orbits; the others do.
    assert start.positions.decisive(5.0).sum() >= 4
    # Every dataset is either seeded or reported ambiguous, never dropped.
    decisive = start.positions.decisive(5.0)
    names = onp.array(start.positions.names)
    assert start.seeded == tuple(names[decisive])
    assert start.ambiguous == tuple(names[~decisive])
    assert len(start.positions.peaks) == len(names)
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


def test_coarse_period_grids_warn_and_failed_fits_sort_last():
    from virgil.fitting import FitResult
    from virgil.orbits import period_grid

    times = [0.0, 100.0, 410.0]
    with pytest.warns(UserWarning, match="period_grid"):
        _warn_if_coarse(onp.geomspace(5.0, 50.0, 100), times)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _warn_if_coarse(period_grid(times, 5.0, 50.0), times)
        _warn_if_coarse([12.0], times)
    fits = [
        FitResult(None, {}, {"loss": loss})
        for loss in (3.0, float("nan"), 1.0, float("inf"), 2.0)
    ]
    order = [r.info["loss"] for r in sorted(fits, key=_loss_order)]
    assert order[:3] == [1.0, 2.0, 3.0]
    assert not any(onp.isfinite(order[3:]))
    # modes() skips non-finite fits, and says when none is left.
    start = OrbitStart(None, None, (fits[1], fits[3]), None)
    with pytest.raises(ValueError, match="finite loss"):
        start.modes()


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
        # A 5-step smoke test of NUTS with one start per chain; a real
        # recovery run is an OzSTAR job.
        mcmc = MCMC(
            NUTS(posterior),
            num_warmup=3,
            num_samples=2,
            num_chains=2,
            chain_method="vectorized",
            progress_bar=False,
        )
        mcmc.run(jax.random.PRNGKey(1), init_params=init)
        samples = mcmc.get_samples(group_by_chain=True)
    assert samples["period"].shape == (2, 2)
    assert all(onp.all(onp.isfinite(v)) for v in samples.values())


# ---------------------------------------------------------------------------
# Night flags with reasons, and UD-aware seeds
# ---------------------------------------------------------------------------

FLAG_AXIS = onp.arange(-12.0, 12.01, 0.5)
FLAG_GRID = {"dra": FLAG_AXIS, "ddec": FLAG_AXIS, "flux": [0.25, 0.5, 1.0]}


def _binary_night(mjd, position, flux, key, **options):
    """A noisy night of a static binary (``position`` in mas)."""
    scene = BinaryModelCartesian(position[0], position[1], flux)
    return simulate(scene, _night(mjd, frames=3, **options), key)


def _flag_epochs():
    """Three nights: a clean one, a twin (flux 1, whose mirror peak ties:
    a perfect single-night alias) and a companion inside λ/2B_max."""
    keys = jax.random.split(jax.random.PRNGKey(4), 3)
    return Epochs(
        {
            "clean": _binary_night(
                T_REF, (5.0, -3.0), 0.25, keys[0], d_vis=0.01, d_phi=0.02
            ),
            "twin": _binary_night(
                T_REF + 1, (5.0, -3.0), 1.0, keys[1], d_vis=0.01, d_phi=0.02
            ),
            "close": _binary_night(
                T_REF + 2, (0.2, 0.1), 0.25, keys[2], d_vis=0.01, d_phi=0.02
            ),
        }
    )


@pytest.fixture(scope="module")
def flagged_positions():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return _flag_epochs(), epoch_positions(
            _flag_epochs(), FLAG_GRID, n_peaks=3
        )


@pytest.mark.slow
def test_epoch_positions_flags_each_night_with_a_code_and_a_reason(
    flagged_positions,
):
    _, found = flagged_positions
    codes = {
        n: {f.code for f in fl} for n, fl in zip(found.names, found.flags)
    }
    # Single-night alias peaks (Gl 229 2024-02-27): a twin's mirror peak
    # ties with it, and its flux sits at the top of the range.
    assert {"alias-peaks", "flux-at-bound"} <= codes["twin"]
    assert "sep-below-resolution" in codes["close"]
    assert codes["clean"] == set()
    # A sub-resolution separation is reported, never blocking (V^2 and
    # closure phases locate a companion below lambda/2B). Whether "close"
    # is flagged for another reason (a flux at the bound) depends on the
    # precision, so only the other nights are pinned.
    close = dict(zip(found.names, found.flags))["close"]
    assert [f.blocking for f in close if f.code == "sep-below-resolution"] == [
        False
    ]
    assert found.flagged[[0, 1]].tolist() == [False, True]
    assert set(found.flag_codes()) == {"twin", "close"}
    for fl in found.flags:
        for f in fl:
            assert f.reason and f.reason != f.code  # a human sentence
    close = dict(zip(found.names, found.flags))["close"]
    assert any(
        "mas" in f.reason for f in close if f.code == "sep-below-resolution"
    )


@pytest.mark.slow
def test_flag_thresholds_are_keywords_and_can_be_switched_off(
    flagged_positions,
):
    epochs, found = flagged_positions
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        off = epoch_positions(epochs, FLAG_GRID, n_peaks=3, flag_nights=False)
        wide = epoch_positions(
            epochs,
            FLAG_GRID,
            n_peaks=3,
            flag_min_gap=1e9,
            flag_alias_nats=None,
        )
    assert not off.flagged.any() and off.flag_codes() == {}
    # The clean night is decisive (no flag at the default 5 nats) but a
    # threshold above its gap flags it as low-gap; with the alias flag
    # disabled the twin is low-gap too.
    assert found.gap_marginal[0] > 5.0
    wide_codes = {
        n: [f.code for f in fl] for n, fl in zip(wide.names, wide.flags)
    }
    assert "low-gap" in wide_codes["clean"]
    assert "alias-peaks" not in wide_codes["twin"]
    assert "low-gap" in wide_codes["twin"]


@pytest.mark.slow
def test_skip_flagged_positions_drop_the_flagged_nights(flagged_positions):
    _, found = flagged_positions
    # Flags cannot be ignored silently: skipping is the default.
    kept = found.positions(t_ref=T_REF)
    keep = ~found.flagged
    assert kept.dra.shape == (int(keep.sum()),) and keep[0]
    onp.testing.assert_allclose(kept.dra, found.dra[keep])
    assert found.positions(t_ref=T_REF, skip_flagged=False).dra.shape == (3,)
    every = dataclasses.replace(
        found, flags=tuple(found.flags[1:2] * len(found.names))
    )
    with pytest.raises(ValueError, match="No dataset is decisive"):
        every.positions(t_ref=T_REF)


@pytest.mark.slow
def test_ud_seeds_are_unbiased_for_a_resolved_primary():
    # A resolved primary (iota Peg-like, 1.05 mas): a point primary
    # absorbs its V^2 deficit into the companion's flux and position and
    # fits badly; the UD seed recovers the companion.
    truth = (5.0, -3.0, 0.25)
    scene = System(
        primary=UniformDisk(1.05),
        companion=PointSource(truth[2], truth[0], truth[1]),
    )
    night = simulate(
        scene,
        _night(T_REF, frames=3, d_vis=0.005, d_phi=0.5),
        jax.random.PRNGKey(2),
    )
    axis = onp.arange(-10.0, 10.01, 1.0)
    grid = {"dra": axis, "ddec": axis, "flux": [0.1, 0.25, 0.5]}
    epochs = Epochs({"night": night})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        point = epoch_positions(epochs, grid)
        ud = epoch_positions(epochs, grid, primary_diam=1.05)
    assert ud.primary_diam == 1.05 and point.primary_diam is None
    assert onp.hypot(ud.dra[0] - truth[0], ud.ddec[0] - truth[1]) < 0.05
    assert abs(ud.flux[0] - truth[2]) < 0.02
    assert ud.chi2_raw[0]["all"] < 2.0
    assert point.chi2_raw[0]["all"] > 10 * ud.chi2_raw[0]["all"]
    assert abs(point.flux[0] - truth[2]) > 2 * abs(ud.flux[0] - truth[2])
    # Zero is a point primary, and a negative diameter is refused.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        zero = epoch_positions(epochs, grid, primary_diam=0.0)
    assert zero.primary_diam is None
    with pytest.raises(ValueError, match="primary_diam"):
        epoch_positions(epochs, grid, primary_diam=-1.0)


def test_a_fitted_primary_is_not_supported_by_epoch_positions_yet():
    epochs = Epochs({"n": _night(T_REF)})
    with pytest.raises(ValueError, match="only a fixed diameter"):
        epoch_positions(epochs, FLAG_GRID, primary_diam="fit")


# --- plan phase 1c: the primary's diameter from the data ----------------------------


def _start_positions(n, dra, ddec, flux):
    """``EpochPositions`` of ``n`` nights at a rough start, without a grid
    search: ``estimate_primary_diameter`` refines from each."""
    from virgil.epochs import EpochPositions

    return EpochPositions(
        names=tuple(f"n{k}" for k in range(n)),
        mjd=T_REF + onp.arange(n, dtype=float),
        dra=onp.full(n, dra),
        ddec=onp.full(n, ddec),
        cov=onp.tile(onp.eye(2), (n, 1, 1)),
        flux=onp.full(n, flux),
        gap_marginal=onp.full(n, 20.0),
        chi2_raw=({},) * n,
        scale=({},) * n,
        flags=((),) * n,
    )


@pytest.mark.slow
def test_a_resolved_primary_is_estimated_and_an_unresolved_one_is_a_point():
    # Resolved primary (problem 10): 1.05 mas (iota Peg) is resolved by a
    # 4-night VLTI-like set; the seeds come back refined with that disc.
    truth = (5.0, -3.0, 0.25)

    def nights(diam):
        primary = PointSource(1.0, 0.0, 0.0) if not diam else UniformDisk(diam)
        scene = System(
            primary=primary,
            companion=PointSource(truth[2], truth[0], truth[1]),
        )
        return [
            simulate(
                scene,
                _night(T_REF + k, frames=2, rotation=7.0 * k, d_vis=0.01),
                jax.random.PRNGKey(k),
            )
            for k in range(4)
        ]

    from virgil.epochs import RESOLVE_NATS, estimate_primary_diameter

    start = _start_positions(4, 5.3, -2.8, 0.2)
    ud = estimate_primary_diameter(nights(1.05), start)
    assert ud.resolved and ud.delta_nats > RESOLVE_NATS
    assert ud.diam == pytest.approx(1.05, rel=0.1)
    assert ud.sigma < 0.2  # mas
    seeds = ud.positions  # consistent with the diameter that was fitted
    assert seeds.primary_diam == pytest.approx(ud.diam)
    assert onp.all(
        onp.hypot(seeds.dra - truth[0], seeds.ddec - truth[1]) < 0.1
    )
    assert onp.all(abs(seeds.flux - truth[2]) < 0.03)
    point = estimate_primary_diameter(nights(None), start)
    assert not point.resolved and point.delta_nats < RESOLVE_NATS
    assert point.positions is start  # unchanged: a point primary


@pytest.mark.slow
def test_an_amplitude_loss_with_a_point_primary_is_not_a_resolved_disc():
    # V² amplitude (problem 13) in the first pass: a night that lost 15%
    # of its contrast looks like a disc to a fit without the amplitude,
    # and does not with it.
    from virgil.epochs import estimate_primary_diameter

    scene = System(
        primary=PointSource(1.0, 0.0, 0.0),
        companion=PointSource(0.25, 5.0, -3.0),
    )
    loss = (0.85, 0.80, 0.85, 0.80)
    data = []
    for k in range(4):
        night = simulate(
            scene,
            _night(T_REF + k, frames=2, rotation=7.0 * k, d_vis=0.01),
            jax.random.PRNGKey(k),
        )
        data.append(night.set("vis", night.vis * loss[k]))
    start = _start_positions(4, 5.3, -2.8, 0.2)
    with_amp = estimate_primary_diameter(data, start, vis_amp=(0.5, 1.2))
    assert not with_amp.resolved
    without = estimate_primary_diameter(data, start)
    assert without.delta_nats > with_amp.delta_nats


def test_the_diameter_estimate_needs_the_nights_of_the_positions():
    from virgil.epochs import estimate_primary_diameter

    with pytest.raises(ValueError, match="same nights"):
        estimate_primary_diameter(
            [_night(T_REF)], _start_positions(4, 5.0, -3.0, 0.2)
        )


def _catalogue(flux, marginal=(0.0, -50.0), sep=5.0):
    from virgil.epochs import EpochPeaks

    n = len(marginal)
    return EpochPeaks(
        dra=onp.full(n, sep),
        ddec=onp.zeros(n),
        flux=onp.full(n, flux),
        loglike=onp.zeros(n),
        marginal=onp.asarray(marginal, float),
        weight=onp.ones(n) / n,
        edge=onp.zeros(n, bool),
        laplace=onp.ones(n, bool),
    )


def _one_night_flags(catalogue, gap=20.0, **options):
    from virgil.epochs import _flag_one_night

    keys = dict(alias_nats=2.0, min_gap=5.0, sep_fraction=0.5, flux_tol=0.02)
    return _flag_one_night(
        catalogue, gap, (0.005, 1.0), 4.0, **{**keys, **options}
    )


def test_flux_at_the_lower_bound_is_flagged_but_the_grid_minimum_is_not():
    # The lower bound is a tenth of the grid's lowest flux (0.05 here), so a
    # faint companion fitted at the grid minimum is fine; one at the bound
    # is a night with no detected companion.
    assert _one_night_flags(_catalogue(0.05)) == []
    flags = _one_night_flags(_catalogue(0.0051))
    assert [f.code for f in flags] == ["flux-at-bound"]
    assert "lower bound" in flags[0].reason
    assert "upper bound" in _one_night_flags(_catalogue(1.0))[0].reason


def test_the_alias_reason_mentions_low_gap_only_when_it_applies():
    close = _catalogue(0.3, marginal=(0.0, -1.0))
    reason = _one_night_flags(close, gap=1.0)[0].reason
    assert "low-gap" in reason
    # Low-gap disabled, or an alias window wider than the gap limit.
    assert (
        "low-gap"
        not in _one_night_flags(close, gap=1.0, min_gap=None)[0].reason
    )
    wide = _catalogue(0.3, marginal=(0.0, -6.0))
    assert (
        "low-gap"
        not in _one_night_flags(wide, gap=6.0, alias_nats=8.0)[0].reason
    )
