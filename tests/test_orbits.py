"""Orbit conventions (design/orbit_scene_joint_fitting.md §2, §5.1–5.2)."""

import itertools

import jax
import jax.numpy as np
import numpy as onp
import pytest
from scipy.optimize import brentq

pytest.importorskip("jaxoplanet")

from virgil.orbits import (  # noqa: E402
    KeplerOrbit,
    PositionData,
    ThieleInnesOrbit,
    _phase_drift,
    period_grid,
    starting_orbits,
)

T_REF = 60500.0
ORBIT = dict(
    period=400.0, dt_peri=30.0, ecc=0.4, inc=60.0, omega=40.0, Omega=110.0
)


def _orbit(**changes):
    return KeplerOrbit(**{**ORBIT, "a_mas": 20.0, **changes}, t_ref=T_REF)


def _reference(mjd, period, dt_peri, ecc, inc, omega, Omega, a_mas, t_ref):
    """An independent NumPy ephemeris: Kepler's equation by root finding,
    then the orbit rotated onto the sky (no Thiele–Innes constants)."""
    out = []
    for t in onp.atleast_1d(mjd):
        mean = 2 * onp.pi * (t - t_ref - dt_peri) / period
        mean = onp.mod(mean, 2 * onp.pi)
        ecc_anomaly = brentq(
            lambda e: e - ecc * onp.sin(e) - mean, 0.0, 2 * onp.pi
        )
        # In the orbital plane: x towards periastron, y along the motion.
        x = a_mas * (onp.cos(ecc_anomaly) - ecc)
        y = a_mas * onp.sqrt(1 - ecc**2) * onp.sin(ecc_anomaly)
        w, n, i = onp.deg2rad([omega, Omega, inc])
        # Rotate by ω from the node, tilt by i about the line of nodes, then
        # turn the node to position angle Ω (North through East).
        u = x * onp.cos(w) - y * onp.sin(w)  # along the node
        v = x * onp.sin(w) + y * onp.cos(w)  # in the plane, ⟂ to the node
        north = u * onp.cos(n) - v * onp.cos(i) * onp.sin(n)
        east = u * onp.sin(n) + v * onp.cos(i) * onp.cos(n)
        away = v * onp.sin(i)
        out.append((east, north, away))
    return onp.array(out).T


def _angles(**changes):
    return {**ORBIT, "a_mas": 20.0, **changes, "t_ref": T_REF}


def test_positions_match_an_independent_ephemeris():
    mjd = T_REF + onp.linspace(-300.0, 500.0, 17)
    grid = itertools.product(
        (0.0, 0.3, 0.9), (5.0, 60.0, 120.0), (0.0, 140.0), (20.0, 250.0)
    )
    with jax.enable_x64(True):
        for ecc, inc, omega, Omega in grid:
            kw = _angles(ecc=ecc, inc=inc, omega=omega, Omega=Omega)
            ours = onp.array(KeplerOrbit(**kw).relative(mjd))
            assert onp.allclose(
                ours, _reference(mjd, **kw), rtol=0, atol=1e-10 * kw["a_mas"]
            )
    # float32, with times relative to t_ref computed on the host.
    kw = _angles(ecc=0.3, inc=60.0)
    ours = onp.array(KeplerOrbit(**kw).relative(mjd))
    assert onp.allclose(ours, _reference(mjd, **kw), rtol=0, atol=1e-5 * 20)


def test_inclination_below_90_turns_the_position_angle_forward():
    mjd = T_REF + onp.linspace(0.0, 400.0, 401)
    with jax.enable_x64(True):
        for inc, sign in ((30.0, 1), (150.0, -1)):
            _, pa = _orbit(inc=inc).separation_pa(mjd)
            steps = onp.diff(onp.unwrap(onp.deg2rad(onp.asarray(pa))))
            assert onp.all(sign * steps > 0)


def test_the_node_ambiguity_and_the_primary_swap():
    mjd = T_REF + onp.linspace(0.0, 400.0, 23)
    with jax.enable_x64(True):
        base = onp.array(_orbit().relative(mjd))
        both = onp.array(_orbit(omega=220.0, Omega=290.0).relative(mjd))
        swap = onp.array(_orbit(omega=220.0).relative(mjd))
    # (Ω + 180°, ω + 180°): the same sky, dz reversed.
    assert onp.allclose(both[:2], base[:2], atol=1e-10)
    assert onp.allclose(both[2], -base[2], atol=1e-10)
    # ω + 180° alone is r -> -r: the primary and secondary swapped.
    assert onp.allclose(swap, -base, atol=1e-10)


@pytest.mark.slow
def test_the_secondary_recedes_at_the_node_with_position_angle_omega():
    orbit = _orbit()
    with jax.enable_x64(True):

        def dz(t):
            return float(orbit.relative(T_REF + t)[2])

        # The ascending node: dz crosses zero going up.
        t = onp.linspace(0.0, 400.0, 4001)
        z = onp.array([dz(x) for x in t])
        up = onp.flatnonzero((z[:-1] < 0) & (z[1:] >= 0))[0]
        node = brentq(dz, t[up], t[up + 1])
        _, pa = orbit.separation_pa(T_REF + node)
        assert float(pa) == pytest.approx(ORBIT["Omega"], abs=1e-6)
        # The velocity is the exact derivative of the position.
        times = T_REF + onp.array([10.0, 123.4, 300.0])
        velocity = onp.array(orbit.relative_velocity(times))
        h = 1e-4
        step = (
            onp.array(orbit.relative(times + h))
            - onp.array(orbit.relative(times - h))
        ) / (2 * h)
        assert onp.allclose(velocity, step, rtol=1e-6, atol=1e-9)


@pytest.mark.parametrize(
    "changes",
    [{}, {"ecc": 0.0}, {"inc": 0.0}, {"inc": 150.0, "Omega": 300.0}],
)
def test_thiele_innes_round_trip(changes):
    mjd = T_REF + onp.linspace(-200.0, 600.0, 31)
    with jax.enable_x64(True):
        orbit = _orbit(**changes)
        ti = orbit.to_thiele_innes()
        assert isinstance(ti, ThieleInnesOrbit)
        assert onp.allclose(
            onp.array(ti.sky(mjd)), onp.array(orbit.relative(mjd))[:2]
        )
        back = ti.to_kepler()
        assert 0.0 <= float(back.Omega) < 180.0
        assert onp.allclose(
            onp.array(back.relative(mjd))[:2],
            onp.array(orbit.relative(mjd))[:2],
            atol=1e-9,
        )
        assert float(back.inc) == pytest.approx(float(orbit.inc), abs=1e-6)
        assert float(back.a_mas) == pytest.approx(20.0, rel=1e-10)


def test_jaxoplanet_round_trip_and_conventions():
    mjd = T_REF + onp.linspace(0.0, 400.0, 9)
    with jax.enable_x64(True):
        orbit = _orbit()
        body, scale = orbit.to_jaxoplanet()
        x, y, z = body.relative_position(mjd - T_REF)
        # (dra, ddec, dz) = (Y, X, -Z): jaxoplanet's Z points to us.
        assert onp.allclose(
            onp.array([y, x, -z]) * scale, onp.array(orbit.relative(mjd))
        )
        # Our ω is the secondary's, jaxoplanet's the primary's.
        assert float(
            onp.rad2deg(onp.arctan2(body.sin_omega_peri, body.cos_omega_peri))
        ) == pytest.approx(ORBIT["omega"] - 180.0)
        # The primary's radial velocity (redshift positive) is positive
        # while the secondary approaches (dz falling).
        vz = onp.array(orbit.relative_velocity(mjd)[2])
        rv = onp.array(body.radial_velocity(mjd - T_REF))
        assert onp.all(onp.sign(rv) == -onp.sign(vz))
        back = KeplerOrbit.from_jaxoplanet(body, a_mas=20.0, t_ref=T_REF)
        for name in ("period", "dt_peri", "ecc", "inc", "omega", "Omega"):
            assert float(getattr(back, name)) == pytest.approx(
                ORBIT[name], abs=1e-9
            )


def test_orbits_are_differentiable_under_jit():
    dt = onp.linspace(0.0, 400.0, 5)

    @jax.jit
    def separation(orbit, dt):
        dra, ddec, _ = orbit._relative(dt)
        return (dra**2 + ddec**2).sum()

    grads = jax.grad(separation)(_orbit(), dt)
    assert onp.all(onp.isfinite(onp.array([grads.ecc, grads.omega])))


def _positions(orbit, mjd, sigma=0.05, key=None):
    dra, ddec, _ = (onp.asarray(x) for x in orbit.relative(mjd))
    if key is not None:
        noise = sigma * onp.asarray(jax.random.normal(key, (2, mjd.size)))
        dra, ddec = dra + noise[0], ddec + noise[1]
    cov = onp.broadcast_to(sigma**2 * onp.eye(2), (mjd.size, 2, 2))
    return PositionData(mjd, dra, ddec, cov)


def test_starting_orbits_recover_the_true_grid_point():
    # §5.2.8: noiseless positions, with the truth on the grid.
    truth = _orbit(period=400.0, dt_peri=100.0, ecc=0.3)
    mjd = T_REF + onp.array([0.0, 37.0, 81.0, 150.0, 210.0, 299.0, 340.0])
    with jax.enable_x64(True):
        positions = _positions(truth, mjd)
        (best, chi2), *_ = starting_orbits(
            positions, periods=[300.0, 400.0, 500.0], n_phase=8
        )
        assert chi2 < 1e-6  # zero, up to the grid's rounding of e
        # Exact up to the grid's rounding of e (0.30000000000000004).
        assert float(best.Omega) == pytest.approx(110.0, abs=1e-4)
        for name, value in (("inc", 60.0), ("omega", 40.0), ("a_mas", 20.0)):
            assert float(getattr(best, name)) == pytest.approx(value, abs=1e-4)
        assert float(best.ecc) == pytest.approx(0.3)


def test_starting_orbits_find_a_noisy_orbit_off_the_grid():
    truth = _orbit(period=430.0, dt_peri=55.0, ecc=0.35)
    mjd = T_REF + onp.linspace(0.0, 420.0, 12)
    with jax.enable_x64(True):
        positions = _positions(truth, mjd, key=jax.random.PRNGKey(3))
        results = starting_orbits(
            positions, periods=onp.geomspace(200, 900, 40)
        )
        best, chi2 = results[0]
        assert [c for _, c in results] == sorted(c for _, c in results)
        assert float(best.period) == pytest.approx(430.0, rel=0.05)
        # Close enough to start a fit: positions within a few mas.
        offset = (
            onp.array(best.relative(mjd))[:2]
            - onp.array(truth.relative(mjd))[:2]
        )
        assert onp.max(onp.abs(offset)) < 2.0


def test_starting_orbits_in_float32():
    # JAX's default precision: the grid solve still finds the orbit.
    truth = _orbit(period=430.0, dt_peri=55.0, ecc=0.35)
    mjd = T_REF + onp.linspace(0.0, 420.0, 12)
    positions = _positions(truth, mjd, key=jax.random.PRNGKey(3))
    results = starting_orbits(positions, periods=onp.geomspace(200, 900, 40))
    best, _ = results[0]
    assert [c for _, c in results] == sorted(c for _, c in results)
    assert float(best.period) == pytest.approx(430.0, rel=0.05)
    offset = (
        onp.array(best.relative(mjd))[:2] - onp.array(truth.relative(mjd))[:2]
    )
    assert onp.max(onp.abs(offset)) < 2.0


def test_period_grid_stays_in_phase_across_the_baseline():
    # Gl 229 B-like: P ~ 12 d over a 410-day baseline. A grid that is
    # uniform in log P spends its points on long periods, and neighbouring
    # short periods drift apart by many cycles.
    times = 59000.0 + onp.array([0.0, 30.0, 200.0, 410.0])
    periods = period_grid(times, 5.0, 50.0)
    assert periods[0] == pytest.approx(5.0) and periods[-1] == pytest.approx(
        50.0
    )
    assert onp.all(onp.diff(periods) > 0)
    # δP <= P²/(kT): neighbouring periods drift by at most 1/k cycles.
    assert _phase_drift(periods, 410.0) <= 1 / 9 + 1e-12
    steps = onp.diff(periods)
    assert onp.all(steps <= periods[1:] ** 2 / (9 * 410.0) * (1 + 1e-12))
    # No more periods than needed (one more than the drift requires).
    assert len(periods) == int(onp.ceil((1 / 5 - 1 / 50) * 9 * 410)) + 1
    assert len(period_grid(times, 5.0, 50.0, k=3)) < len(periods) / 2
    assert _phase_drift(onp.geomspace(5.0, 50.0, 100), 410.0) > 1.0
    assert _phase_drift([12.0], 410.0) == 0.0
    with pytest.raises(ValueError):
        period_grid(times, 50.0, 5.0)
    with pytest.raises(ValueError):
        period_grid([59000.0], 5.0, 50.0)


def test_positions_at_the_origin_have_no_orbit():
    mjd = T_REF + onp.arange(4.0)
    zero = PositionData(
        mjd, onp.zeros(4), onp.zeros(4), onp.eye(2)[None] * onp.ones((4, 1, 1))
    )
    with pytest.raises(ValueError, match="no orbit"):
        starting_orbits(zero, periods=[100.0], n_phase=4)


def test_position_likelihood_matches_a_gaussian():
    from scipy.stats import multivariate_normal

    orbit = _orbit()
    mjd = T_REF + onp.array([10.0, 90.0, 200.0])
    cov = onp.array([[[0.04, 0.01], [0.01, 0.09]]] * 3)
    with jax.enable_x64(True):
        dra, ddec, _ = (onp.asarray(x) + 0.1 for x in orbit.relative(mjd))
        data = PositionData(mjd, dra, ddec, cov)
        expected = sum(
            multivariate_normal(onp.array([a, d]) - 0.1, c).logpdf([a, d])
            for a, d, c in zip(dra, ddec, cov)
        )
        assert float(data.loglike(orbit)) == pytest.approx(expected, rel=1e-10)
        # Separation and position angle with their errors give the same
        # positions and, to first order, the same covariance.
        sep, pa = (onp.asarray(x) for x in orbit.separation_pa(mjd))
        polar = PositionData.from_sep_pa(mjd, sep, pa, 0.2, 0.5)
        assert onp.allclose(polar.whitened_residuals(orbit), 0.0, atol=1e-9)


@pytest.mark.parametrize(
    "changes, match",
    [
        ({"period": 0.0}, "period"),
        ({"ecc": 1.0}, "ecc"),
        ({"inc": 200.0}, "inc"),
        ({"inc": 180.0}, "inc"),
        ({"a_mas": -1.0}, "a_mas"),
        ({"omega": float("nan")}, "omega"),
    ],
)
def test_out_of_domain_orbits_are_rejected(changes, match):
    with pytest.raises(ValueError, match=match):
        _orbit(**changes)
    # Traced values are not checked (they may be mid-optimization).
    jax.jit(lambda p: _orbit(period=p).period)(0.0)


def test_missing_jaxoplanet_names_the_extra(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "jaxoplanet", None)
    with pytest.raises(ImportError, match=r"virgil-astro\[orbits\]"):
        _orbit().to_jaxoplanet()
    with pytest.raises(ImportError, match=r"virgil-astro\[orbits\]"):
        _orbit().relative(T_REF)


def test_position_data_checks_its_shapes():
    mjd = T_REF + onp.arange(3.0)
    cov = onp.broadcast_to(onp.eye(2), (3, 2, 2))
    with pytest.raises(ValueError, match="dra has shape"):
        PositionData(mjd, onp.zeros(2), onp.zeros(3), cov)
    with pytest.raises(ValueError, match="ddec has shape"):
        PositionData(mjd, onp.zeros(3), 0.0, cov)


# --- State vectors for short arcs (design R4) ---


@pytest.mark.parametrize(
    "changes",
    [
        {},
        {"ecc": 0.9, "omega": 300.0, "Omega": 20.0},
        {"ecc": 1e-6},
        {"inc": 2.0},
        {"inc": 150.0, "Omega": 250.0},
    ],
)
def test_state_vector_round_trip(changes):
    from virgil.orbits import StateVectorOrbit

    mjd = T_REF + onp.linspace(-300.0, 500.0, 41)
    with jax.enable_x64(True):
        orbit = _orbit(**changes)
        state = StateVectorOrbit.from_kepler(orbit)
        back = state.to_kepler()
        # The orbit itself (all three axes, so the node is the true one).
        assert onp.allclose(
            onp.array(back.relative(mjd)),
            onp.array(orbit.relative(mjd)),
            atol=1e-7,
        )
        for name in ("period", "ecc", "a_mas"):
            assert float(getattr(back, name)) == pytest.approx(
                float(getattr(orbit, name)), rel=1e-7, abs=1e-9
            )
        if changes.get("ecc", 0.4) > 1e-3 and changes.get("inc", 60.0) > 5.0:
            for name in ("inc", "omega", "Omega"):
                assert float(getattr(back, name)) == pytest.approx(
                    float(getattr(orbit, name)), abs=1e-6
                )


def test_unbound_states_are_rejected_and_gradients_are_finite():
    import equinox as eqx

    from virgil.orbits import StateVectorOrbit

    with pytest.raises(ValueError, match="unbound"):
        StateVectorOrbit(10.0, 0.0, 0.0, 1e4, 0.0, 0.0, 1.0)
    with pytest.raises(ValueError, match="away from the primary"):
        StateVectorOrbit(0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    with pytest.raises(ValueError, match="no angular momentum"):
        StateVectorOrbit(10.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1e3)
    state = StateVectorOrbit.from_kepler(_orbit())

    def separation(vra):
        orbit = eqx.tree_at(lambda s: s.vra, state, vra).to_kepler()
        dra, ddec, _ = orbit._relative(onp.array([50.0, 100.0]))
        return (dra**2 + ddec**2).sum()

    grad = jax.jit(jax.grad(separation))(state.vra)
    assert onp.isfinite(float(grad))


# --- Radial velocities, masses and the axial prior (design R5, R6) ---


def test_rv_semi_amplitude_matches_keplers_laws():
    from virgil.orbits import RVData

    orbit = _orbit()  # P 400 d, e 0.4, i 60, a 20 mas
    q, distance = 0.5, 50.0
    mjd = T_REF + onp.linspace(0.0, 400.0, 4001)
    with jax.enable_x64(True):
        rv = onp.asarray(
            RVData(mjd, onp.zeros(mjd.size), 1.0).model(
                orbit, q, 3.0, distance
            )
        )
    # K1 = 2π a1 sin i / (P √(1 - e²)), a1 = a q / (1 + q) in km.
    a_km = 20e-3 * distance * 1.495978707e8
    k1 = (
        2
        * onp.pi
        * a_km
        * q
        / (1 + q)
        * onp.sin(onp.deg2rad(60.0))
        / (400.0 * 86400 * onp.sqrt(1 - 0.4**2))
    )
    assert (rv.max() - rv.min()) / 2 == pytest.approx(k1, rel=1e-4)
    assert (rv.max() + rv.min()) / 2 != pytest.approx(3.0)  # e ≠ 0: skewed
    # The primary recedes while the secondary approaches, and vice versa.
    with jax.enable_x64(True):
        vz = onp.asarray(orbit.relative_velocity(mjd)[2])
        secondary = RVData(mjd, onp.zeros(mjd.size), 1.0, star="secondary")
        rv2 = onp.asarray(secondary.model(orbit, q, 3.0, distance))
    assert onp.all(
        onp.sign(rv - 3.0)[onp.abs(vz) > 1e-6]
        == -onp.sign(vz)[onp.abs(vz) > 1e-6]
    )
    assert onp.allclose((rv - 3.0) * (1 + q) / q, -(rv2 - 3.0) * (1 + q))


def test_total_mass_and_distance_are_inverse():
    from virgil.orbits import distance_pc, total_mass

    # 1 au at 1 pc is 1000 mas; an orbit with P = 2π sqrt(au³ / GM☉) (the
    # Gaussian year, 365.256898 d) then has 1 solar mass.
    gm_sun, au, day = 1.3271244e20, 149597870700.0, 86400.0
    period = 2 * onp.pi * onp.sqrt(au**3 / gm_sun) / day
    assert period == pytest.approx(365.256898, abs=1e-6)
    with jax.enable_x64(True):
        earth = _orbit(period=period, a_mas=1000.0)
        assert float(total_mass(earth, 1.0)) == pytest.approx(1.0, abs=1e-12)
        assert float(distance_pc(earth, 1.0)) == pytest.approx(1.0, abs=1e-12)
        # At the same period M scales as a^3: 4 times larger a is 64 times the mass.
        wide = _orbit(period=period, a_mas=4000.0)
        assert float(total_mass(wide, 1.0)) == pytest.approx(64.0, rel=1e-12)
    # The old M = a³/P² with P in Julian years was 3.8e-5 low.
    julian = _orbit(period=365.25, a_mas=1000.0)
    assert float(total_mass(julian, 1.0)) == pytest.approx(1.0, rel=1e-4)
    orbit = _orbit()
    assert float(
        distance_pc(orbit, total_mass(orbit, 123.0))
    ) == pytest.approx(123.0)


def test_axial_von_mises_is_a_normalized_prior_with_period_180():
    from virgil.orbits import AxialVonMises

    prior = AxialVonMises(100.0, 4.0)
    theta = onp.linspace(0.0, 360.0, 36001)[:-1]
    density = onp.exp(onp.asarray(prior.log_prob(theta)))
    assert density.sum() * 0.01 == pytest.approx(1.0, rel=1e-6)
    assert float(prior.log_prob(100.0)) == pytest.approx(
        float(prior.log_prob(280.0))
    )
    draws = onp.asarray(prior.sample(jax.random.PRNGKey(0), (4000,)))
    assert draws.min() >= 0.0 and draws.max() < 360.0
    near = onp.abs((draws - 100.0 + 90.0) % 180.0 - 90.0) < 30.0
    assert near.mean() > 0.9
    assert (
        0.4
        < (onp.abs((draws - 100.0 + 180.0) % 360.0 - 180.0) < 90.0).mean()
        < 0.6
    )


def test_positions_and_rvs_fix_the_node_in_a_joint_fit():
    import numpyro.distributions as dist

    from virgil.fitting import fit
    from virgil.orbits import RVData

    truth = _orbit()
    mjd = T_REF + onp.linspace(0.0, 380.0, 10)
    with jax.enable_x64(True):
        positions = _positions(truth, mjd)
        rvs = RVData(
            mjd,
            onp.asarray(
                RVData(mjd, onp.zeros(10), 1.0).model(truth, 0.5, 3.0, 50.0)
            ),
            0.1,
        )
    names = ("period", "dt_peri", "ecc", "inc", "omega", "Omega", "a_mas")

    def orbit(v):
        return KeplerOrbit(*(v[k] for k in names), t_ref=T_REF)

    priors = {
        "period": dist.Uniform(300.0, 500.0),
        "dt_peri": dist.Uniform(-200.0, 200.0),
        "ecc": dist.Uniform(0.0, 0.9),
        "inc": dist.Uniform(0.0, 180.0),
        "omega": dist.Uniform(-360.0, 720.0),
        "Omega": dist.Uniform(-360.0, 720.0),
        "a_mas": dist.Uniform(1.0, 50.0),
        "gamma": dist.Uniform(-50.0, 50.0),
    }
    # Positions alone cannot tell the node from (Omega + 180, omega + 180):
    # fit from both, and the radial velocities pick the true one.
    terms = [
        positions.term(orbit),
        rvs.term(lambda v: (orbit(v), 0.5, v["gamma"], 50.0)),
    ]
    results = []
    for flip in (0.0, 180.0):
        start = {**{k: float(getattr(truth, k)) for k in names}, "gamma": 0.0}
        start["Omega"] += flip
        start["omega"] += flip
        results.append(
            fit(lambda **kw: None, priors, (), init=start, likelihoods=terms)
        )
    right, wrong = results
    assert right.info["converged"]
    assert right.info["ndata"] == [20, 10]
    assert float(right.values["Omega"]) == pytest.approx(110.0, abs=0.1)
    assert float(right.values["gamma"]) == pytest.approx(3.0, abs=0.01)
    # From the other node the velocities' signs are reversed: no orbit fits
    # both the positions and the velocities there.
    assert sum(right.info["chi2"]) < 1e-3
    assert sum(wrong.info["chi2"]) > 1e3


def test_a_plain_function_is_a_likelihood_term_and_axial_priors_work_in_fit():
    import numpyro.distributions as dist

    from virgil.fitting import fit
    from virgil.orbits import AxialVonMises, RVData

    truth = _orbit()
    mjd = T_REF + onp.linspace(0.0, 380.0, 10)
    with jax.enable_x64(True):
        positions = _positions(truth, mjd)
    fixed = {
        k: float(getattr(truth, k))
        for k in ("period", "dt_peri", "ecc", "inc", "omega", "a_mas")
    }

    def residuals(values):  # a plain function, as the docs allow
        orbit = KeplerOrbit(**fixed, Omega=values["Omega"], t_ref=T_REF)
        return positions.whitened_residuals(orbit)

    result = fit(
        lambda **kw: None,
        {"Omega": AxialVonMises(110.0, 2.0)},
        (),
        init={"Omega": 100.0},
        likelihoods=[residuals],
    )
    assert result.info["ndata"] == [20]
    assert float(result.values["Omega"]) == pytest.approx(110.0, abs=1e-3)
    with pytest.raises(ValueError, match="d_rv"):
        RVData(mjd, onp.zeros(10), 0.0)
    with pytest.raises(ValueError, match="d_rv"):
        RVData(mjd, onp.zeros(10), onp.nan)
    assert isinstance(AxialVonMises(1.0, 2.0), dist.Distribution)


def _numpyro_setup():
    import numpyro.distributions as dist

    truth = _orbit()
    mjd = T_REF + onp.linspace(0.0, 380.0, 6)
    with jax.enable_x64(True):
        positions = _positions(
            truth, mjd, sigma=0.3, key=jax.random.PRNGKey(2)
        )
    priors = {"a_mas": dist.Uniform(5.0, 50.0), "ecc": dist.Uniform(0.0, 0.9)}

    def orbit(v):
        return _orbit(a_mas=v["a_mas"], ecc=v["ecc"])

    return positions, priors, orbit


def _log_density(model, values):
    from numpyro.infer.util import log_density

    return float(log_density(model, (), {}, values)[0])


def test_numpyro_model_adds_a_position_term_to_the_prior():
    from virgil.likelihood import numpyro_model

    positions, priors, orbit = _numpyro_setup()
    model = numpyro_model(
        lambda **kw: None, priors, (), likelihoods=[positions.term(orbit)]
    )
    values = {"a_mas": 19.0, "ecc": 0.35}
    log_prior = sum(float(priors[k].log_prob(v)) for k, v in values.items())
    chi2 = float(onp.sum(positions.whitened_residuals(orbit(values)) ** 2))
    # Matching constant: the data's own normalization, from ``loglike``.
    constant = float(positions.loglike(orbit(values))) + 0.5 * chi2
    assert _log_density(model, values) == pytest.approx(
        log_prior - 0.5 * chi2 + constant, rel=1e-5
    )
    # A plain callable of whitened residuals adds -chi2/2 alone.
    plain = numpyro_model(
        lambda **kw: None,
        priors,
        (),
        likelihoods=[lambda v: positions.whitened_residuals(orbit(v))],
    )
    assert _log_density(plain, values) == pytest.approx(
        log_prior - 0.5 * chi2, rel=1e-5
    )


def test_term_loglike_is_the_datas_loglike():
    positions, _, orbit = _numpyro_setup()
    values = {"a_mas": 19.0, "ecc": 0.35}
    term = positions.term(orbit)
    assert float(term.loglike(values)) == pytest.approx(
        float(positions.loglike(orbit(values))), rel=1e-6
    )


def test_numpyro_model_sums_oidata_and_likelihood_terms():
    import numpyro.distributions as dist

    from virgil.coverage import vlti_oidata
    from virgil.likelihood import numpyro_model
    from virgil.models import PointSource

    positions, priors, orbit = _numpyro_setup()
    priors = priors | {"flux": dist.Uniform(0.5, 2.0)}
    data = vlti_oidata(hour_angles_h=(0.0,), wavelengths_m=[3.5e-6])

    def source(flux, **kw):
        return PointSource(flux=flux)

    term = [positions.term(orbit)]
    both = numpyro_model(source, priors, data, likelihoods=term)
    oidata = numpyro_model(source, priors, data)
    terms = numpyro_model(source, priors, (), likelihoods=term)
    values = {"a_mas": 19.0, "ecc": 0.35, "flux": 1.0}
    log_prior = sum(float(priors[k].log_prob(v)) for k, v in values.items())
    assert _log_density(both, values) == pytest.approx(
        _log_density(oidata, values) + _log_density(terms, values) - log_prior,
        rel=1e-5,
    )


def _rv_setup(n=30, jitter=1.0, d_rv=0.1, seed=3):
    from virgil.orbits import RVData

    truth = _orbit()
    mjd = T_REF + onp.linspace(0.0, 380.0, n)
    with jax.enable_x64(True):
        clean = RVData(mjd, onp.zeros(n), d_rv)
        model = onp.asarray(clean.model(truth, 0.5, 3.0, 50.0))
        noise = onp.random.default_rng(seed).normal(
            0.0, onp.hypot(d_rv, jitter), n
        )
        rvs = RVData(mjd, model + noise, d_rv)
    return truth, rvs


def _rv_term(truth, rvs, jitter="rv_jitter"):
    return rvs.term(lambda v: (truth, 0.5, v["gamma"], 50.0), jitter=jitter)


def test_rv_jitter_loglike_is_the_gaussian_with_inflated_errors():
    from scipy.stats import norm

    truth, rvs = _rv_setup(n=8)
    s = 0.7
    values = {"gamma": 2.5, "rv_jitter": s}
    with jax.enable_x64(True):
        model = onp.asarray(rvs.model(truth, 0.5, 2.5, 50.0))
        sigma = onp.sqrt(onp.asarray(rvs.d_rv) ** 2 + s**2)
        expected = norm.logpdf(onp.asarray(rvs.rv), model, sigma).sum()
        term = _rv_term(truth, rvs)
        assert float(term.loglike(values)) == pytest.approx(expected)
        assert float(term.log_norm(values)) == pytest.approx(
            onp.log(sigma).sum()
        )


def test_rv_jitter_zero_matches_the_plain_term():
    truth, rvs = _rv_setup(n=8)
    zero = {"gamma": 2.5, "rv_jitter": 0.0}
    with jax.enable_x64(True):
        term, plain = _rv_term(truth, rvs), _rv_term(truth, rvs, None)
        assert float(term.loglike(zero)) == pytest.approx(
            float(plain.loglike(zero))
        )
        onp.testing.assert_allclose(term(zero), plain(zero))
    assert term.has_log_norm and not plain.has_log_norm


def test_numpyro_model_includes_the_rv_jitter_normalization():
    import numpyro.distributions as dist

    from virgil.likelihood import numpyro_model

    truth, rvs = _rv_setup(n=8)
    priors = {
        "gamma": dist.Normal(0.0, 10.0),
        "rv_jitter": dist.HalfNormal(2.0),
    }
    values = {"gamma": 2.5, "rv_jitter": 0.7}
    with jax.enable_x64(True):
        term = _rv_term(truth, rvs)
        model = numpyro_model(
            lambda **kw: None, priors, (), likelihoods=[term]
        )
        expected = sum(
            float(priors[k].log_prob(v)) for k, v in values.items()
        ) + float(term.loglike(values))
        assert _log_density(model, values) == pytest.approx(expected)


def test_fit_with_rv_jitter_uses_lbfgs_and_rejects_lm():
    import numpyro.distributions as dist

    from virgil.fitting import fit

    truth, rvs = _rv_setup(n=10)
    priors = {
        "gamma": dist.Uniform(-50.0, 50.0),
        "rv_jitter": dist.HalfNormal(5.0),
    }
    init = {"gamma": 0.0, "rv_jitter": 0.5}
    with jax.enable_x64(True):
        term = _rv_term(truth, rvs)
        result = fit(
            lambda **k: None, priors, (), init=init, likelihoods=[term]
        )
        assert result.info["method"] == "lbfgs"
        with pytest.raises(TypeError, match="least-squares"):
            fit(
                lambda **k: None,
                priors,
                (),
                init=init,
                likelihoods=[term],
                method="lm",
            )


def test_fit_recovers_an_injected_rv_jitter():
    import numpyro.distributions as dist

    from virgil.fitting import fit

    truth, rvs = _rv_setup(n=30, jitter=1.0)
    priors = {
        "gamma": dist.Uniform(-50.0, 50.0),
        "rv_jitter": dist.HalfNormal(5.0),
    }
    with jax.enable_x64(True):
        result = fit(
            lambda **k: None,
            priors,
            (),
            init={"gamma": 0.0, "rv_jitter": 0.5},
            likelihoods=[_rv_term(truth, rvs)],
        )
    s = float(result.values["rv_jitter"])
    assert s == pytest.approx(1.0, abs=3 / onp.sqrt(2 * 30))
    assert float(result.values["gamma"]) == pytest.approx(3.0, abs=0.6)


# -- analytic marginalization of the instrument zero points ----------------


def _two_instruments(n=12, jitter=0.4, seed=5):
    """RVs from two spectrographs with zero points 3 and 7 km/s."""
    from virgil.orbits import RVData

    truth = _orbit()
    rng = onp.random.default_rng(seed)
    mjd = T_REF + onp.sort(rng.uniform(0.0, 380.0, n))
    inst = onp.where(onp.arange(n) % 3 == 0, "B", "A")
    d_rv = rng.uniform(0.05, 0.3, n)
    with jax.enable_x64(True):
        clean = RVData(mjd, onp.zeros(n), d_rv, instrument=inst)
        model = onp.asarray(clean.model(truth, 0.5, 0.0, 50.0))
        zero = onp.where(inst == "B", 7.0, 3.0)
        noise = rng.normal(0.0, onp.hypot(d_rv, jitter))
        rvs = RVData(mjd, model + zero + noise, d_rv, instrument=inst)
    return truth, rvs


def _dense_marginal(rvs, model, s, mean, sd):
    from scipy.stats import multivariate_normal

    A = onp.asarray(rvs._design())
    C = onp.diag(onp.asarray(rvs.d_rv) ** 2 + s**2)
    cov = C + A @ onp.diag(sd**2) @ A.T
    return multivariate_normal(model + A @ mean, cov).logpdf(
        onp.asarray(rvs.rv)
    ), (A, C)


def _marginal_term(truth, rvs, mean, sd):
    return rvs.term(
        lambda v: (truth, 0.5, 0.0, 50.0),
        jitter="rv_jitter",
        marginalize_offsets=(mean, sd),
    )


def test_marginal_zero_points_match_the_dense_gaussian():
    truth, rvs = _two_instruments()
    s, mean, sd = 0.4, onp.array([1.0, -2.0]), onp.array([5.0, 20.0])
    with jax.enable_x64(True):
        term = _marginal_term(truth, rvs, mean, sd)
        values = {"rv_jitter": s}
        model = onp.asarray(rvs.model(truth, 0.5, 0.0, 50.0))
        expected, _ = _dense_marginal(rvs, model, s, mean, sd)
        assert rvs.instruments == ("A", "B")
        assert float(term.loglike(values)) == pytest.approx(expected)
        u = onp.asarray(term(values))
        assert u.shape == (rvs.rv.size,)
        # the whitened residuals and log_norm give the same density
        logp = (
            -0.5 * u @ u
            - float(term.log_norm(values))
            - (u.size / 2) * onp.log(2 * onp.pi)
        )
        assert logp == pytest.approx(expected)


def test_zero_point_posterior_matches_dense_conditioning():
    truth, rvs = _two_instruments()
    s, mean, sd = 0.4, onp.array([1.0, -2.0]), onp.array([5.0, 20.0])
    with jax.enable_x64(True):
        term = _marginal_term(truth, rvs, mean, sd)
        post_mean, post_cov = term.posterior({"rv_jitter": s})
        model = onp.asarray(rvs.model(truth, 0.5, 0.0, 50.0))
        _, (A, C) = _dense_marginal(rvs, model, s, mean, sd)
        Lam = onp.diag(sd**2)
        gain = Lam @ A.T @ onp.linalg.inv(C + A @ Lam @ A.T)
        resid = onp.asarray(rvs.rv) - model - A @ mean
        assert onp.asarray(post_mean) == pytest.approx(
            mean + gain @ resid, rel=1e-8
        )
        assert onp.asarray(post_cov) == pytest.approx(
            Lam - gain @ A @ Lam, rel=1e-8, abs=1e-12
        )


def test_marginalized_fit_recovers_the_orbit_like_free_offsets():
    import numpyro.distributions as dist

    from virgil.fitting import fit

    truth, rvs = _two_instruments(n=16, jitter=0.0)
    is_b = rvs.inst == 1
    with jax.enable_x64(True):
        free = fit(
            lambda **k: None,
            {
                "gamma": dist.Uniform(-50.0, 50.0),
                "off": dist.Uniform(-50.0, 50.0),
                "dist": dist.Uniform(10.0, 200.0),
            },
            (),
            init={"gamma": 0.0, "off": 0.0, "dist": 40.0},
            likelihoods=[
                rvs.term(
                    lambda v: (
                        truth,
                        0.5,
                        v["gamma"] + v["off"] * is_b,
                        v["dist"],
                    )
                )
            ],
        )
        marg = fit(
            lambda **k: None,
            {"dist": dist.Uniform(10.0, 200.0)},
            (),
            init={"dist": 40.0},
            likelihoods=[
                rvs.term(
                    lambda v: (truth, 0.5, 0.0, v["dist"]),
                    marginalize_offsets=(0.0, 100.0),
                )
            ],
        )
    assert float(free.values["dist"]) == pytest.approx(50.0, rel=0.05)
    assert float(marg.values["dist"]) == pytest.approx(
        float(free.values["dist"]), rel=2e-3
    )


def test_single_instrument_tiny_prior_is_a_fixed_gamma():
    truth, rvs = _rv_setup(n=8)
    with jax.enable_x64(True):
        term = rvs.term(
            lambda v: (truth, 0.5, 0.0, 50.0),
            jitter="rv_jitter",
            marginalize_offsets=(2.5, 1e-6),
        )
        fixed = _rv_term(truth, rvs)
        values = {"rv_jitter": 0.7, "gamma": 2.5}
        # the prior sd adds a constant: compare the data part of the density
        assert float(term.loglike(values)) == pytest.approx(
            float(fixed.loglike(values)), abs=1e-5
        )
        assert onp.asarray(term(values)) == pytest.approx(
            onp.asarray(fixed(values)), abs=1e-5
        )


def test_true_zero_point_prior_is_an_error_and_flat_prior_is_rejected():
    truth, rvs = _rv_setup(n=8)
    with pytest.raises(ValueError, match=r"state the .*prior.*\(mean, sd\)"):
        rvs.term(lambda v: (truth, 0.5, 0.0, 50.0), marginalize_offsets=True)
    with pytest.raises(ValueError, match="finite sd"):
        rvs.term(
            lambda v: (truth, 0.5, 0.0, 50.0),
            marginalize_offsets=(0.0, onp.inf),
        )
    for bad in (onp.nan, onp.inf):
        with pytest.raises(ValueError, match="finite mean"):
            rvs.term(
                lambda v: (truth, 0.5, 0.0, 50.0),
                marginalize_offsets=(bad, 10.0),
            )


@pytest.mark.parametrize("positions_only", [True, False])
def test_varpi_orientation_round_trips_and_the_flip_collapses(
    positions_only,
):
    # design/orbit_prior_art.md §4.1 test 5: (Ω, ω) → (2Ω or Ω, ϖ) → back
    # gives the same sky orbit, and both members of the flip pair
    # (Ω + 180°, ω + 180°) map to one point when positions alone fit.
    from virgil.angles import AngleVector
    from virgil.orbits import orientation_from_varpi, orientation_priors

    priors = orientation_priors(positions_only, prefix="orbit.")
    node = "orbit.two_Omega" if positions_only else "orbit.Omega"
    assert set(priors) == {node, "orbit.varpi"}
    assert all(
        isinstance(p, AngleVector) and p.uniform for p in priors.values()
    )

    rng = onp.random.default_rng(3)
    mjd = T_REF + onp.linspace(0.0, 400.0, 9)
    for Omega, omega in rng.uniform(0.0, 360.0, (20, 2)):
        truth = _orbit(Omega=Omega, omega=omega)
        flipped = _orbit(Omega=Omega + 180.0, omega=omega + 180.0)
        varpi = onp.mod(Omega + omega, 360.0)
        node_value = (
            {"two_Omega": onp.mod(2 * Omega, 360.0)}
            if positions_only
            else {"Omega": Omega}
        )
        flipped_varpi = onp.mod(Omega + omega + 360.0, 360.0)
        flipped_node = (
            {"two_Omega": onp.mod(2 * (Omega + 180.0), 360.0)}
            if positions_only
            else {"Omega": Omega + 180.0}
        )
        if positions_only:  # one point for both modes
            assert flipped_varpi == pytest.approx(varpi)
            assert flipped_node["two_Omega"] == pytest.approx(
                node_value["two_Omega"]
            )
        rebuilt = KeplerOrbit.from_varpi(
            ORBIT["period"],
            ORBIT["dt_peri"],
            ORBIT["ecc"],
            ORBIT["inc"],
            varpi,
            20.0,
            t_ref=T_REF,
            **node_value,
        )
        omega_back, Omega_back = orientation_from_varpi(varpi, **node_value)
        assert float(rebuilt.omega) == pytest.approx(float(omega_back))
        sky = onp.asarray(rebuilt.relative(mjd)[:2])
        onp.testing.assert_allclose(
            sky, onp.asarray(truth.relative(mjd)[:2]), atol=1e-3
        )
        onp.testing.assert_allclose(
            sky, onp.asarray(flipped.relative(mjd)[:2]), atol=1e-3
        )
        if positions_only:
            assert 0.0 <= float(rebuilt.Omega) < 180.0
        else:  # the node, and so dz, is the truth's
            onp.testing.assert_allclose(
                onp.asarray(rebuilt.relative(mjd)[2]),
                onp.asarray(truth.relative(mjd)[2]),
                atol=1e-3,
            )
    with pytest.raises(ValueError, match="exactly one"):
        orientation_from_varpi(10.0)


def test_an_orbit_fit_in_two_omega_and_varpi_through_the_wrap():
    # A position-only fit samples 2Ω and ϖ as vectors, starting across the
    # wrap from the truth (ϖ = 350°, 2Ω = 20°).
    from virgil.fitting import fit
    from virgil.orbits import orientation_priors

    truth = _orbit(Omega=10.0, omega=340.0)
    mjd = T_REF + onp.linspace(0.0, 380.0, 10)
    with jax.enable_x64(True):
        positions = _positions(truth, mjd)

    def orbit_fn(v):
        return KeplerOrbit.from_varpi(
            ORBIT["period"],
            ORBIT["dt_peri"],
            ORBIT["ecc"],
            ORBIT["inc"],
            v["varpi"],
            20.0,
            two_Omega=v["two_Omega"],
            t_ref=T_REF,
        )

    result = fit(
        lambda **kw: None,
        orientation_priors(),
        (),
        init={"two_Omega": 340.0, "varpi": 15.0},
        likelihoods=[positions.term(orbit_fn)],
    )
    assert result.info["method"] == "lm"
    assert float(result.values["varpi"]) == pytest.approx(350.0, abs=1e-3)
    assert float(result.values["two_Omega"]) == pytest.approx(20.0, abs=1e-3)


def _random_elements(rng, n, edge_on_margin=0.05):
    """Random elements, prograde and retrograde, away from edge-on."""
    cos_i = rng.uniform(edge_on_margin, 1.0, n) * rng.choice([-1, 1], n)
    return dict(
        period=rng.uniform(50.0, 5000.0, n),
        ecc=rng.uniform(0.0, 0.95, n),
        inc=onp.rad2deg(onp.arccos(cos_i)),
        omega=rng.uniform(0.0, 360.0, n),
        Omega=rng.uniform(0.0, 360.0, n),
        a_mas=rng.uniform(1.0, 100.0, n),
    )


def test_position_angle_at_t_ref_round_trips_through_dt_peri():
    # design/orbit_prior_art.md §4.2: KeplerOrbit → PA at t_ref → back,
    # over random elements in float64, and the PA of the orbit built from
    # θ at t_ref is θ.
    rng = onp.random.default_rng(7)
    n = 200
    elements = _random_elements(rng, n)
    dt_peri = rng.uniform(-0.5, 0.5, n) * elements["period"]
    theta_in = rng.uniform(0.0, 360.0, n)
    mjd = T_REF + rng.uniform(-300.0, 300.0, 5)
    with jax.enable_x64(True):
        for k in range(n):
            el = {key: value[k] for key, value in elements.items()}
            orbit = KeplerOrbit(dt_peri=dt_peri[k], **el, t_ref=T_REF)
            _, theta = orbit.separation_pa(T_REF)
            back = KeplerOrbit.from_position_angle(
                theta=theta, **el, t_ref=T_REF
            )
            offset = (float(back.dt_peri) - dt_peri[k]) / el["period"]
            assert abs(offset - round(offset)) < 1e-12
            onp.testing.assert_allclose(
                onp.asarray(back.relative(mjd)),
                onp.asarray(orbit.relative(mjd)),
                atol=1e-9 * el["a_mas"],
            )
            from_theta = KeplerOrbit.from_position_angle(
                theta=theta_in[k], **el, t_ref=T_REF
            )
            _, pa = from_theta.separation_pa(T_REF)
            wrapped = (float(pa) - theta_in[k] + 180.0) % 360.0 - 180.0
            assert abs(wrapped) < 1e-9


def _mean_anomaly(theta, el):
    from virgil.orbits import _mean_anomaly_at_ref

    return onp.asarray(
        _mean_anomaly_at_ref(
            theta, el["ecc"], el["inc"], el["omega"], el["Omega"]
        )
    )


def test_position_angle_jacobian_matches_finite_differences():
    from virgil.orbits import position_angle_log_jacobian

    rng = onp.random.default_rng(11)
    elements = _random_elements(rng, 50)
    step = 1e-5  # degrees
    with jax.enable_x64(True):
        for k in range(50):
            el = {key: value[k] for key, value in elements.items()}
            theta = rng.uniform(0.0, 360.0)
            dm = _mean_anomaly(theta + step, el) - _mean_anomaly(
                theta - step, el
            )
            dm = (dm + onp.pi) % (2 * onp.pi) - onp.pi
            numeric = abs(dm / onp.deg2rad(2 * step))
            log_j = float(
                position_angle_log_jacobian(
                    theta, el["ecc"], el["inc"], el["omega"], el["Omega"]
                )
            )
            assert onp.exp(log_j) == pytest.approx(numeric, rel=1e-6)


def test_theta_weighted_by_the_jacobian_is_uniform_in_mean_anomaly():
    # Monte Carlo: θ uniform, weighted by |∂M/∂θ|, gives uniform M; θ
    # uniform alone does not (so the term matters), and the weights
    # average to 1 (the prior stays normalized).
    from virgil.orbits import position_angle_log_jacobian

    el = dict(ecc=0.7, inc=130.0, omega=75.0, Omega=200.0)
    theta = onp.random.default_rng(5).uniform(0.0, 360.0, 400_000)
    with jax.enable_x64(True):
        weights = onp.exp(
            onp.asarray(position_angle_log_jacobian(theta, **el))
        )
        mean = _mean_anomaly(theta, el)
    assert weights.mean() == pytest.approx(1.0, rel=0.02)
    edges = onp.linspace(-onp.pi, onp.pi, 13)
    weighted = onp.histogram(mean, edges, weights=weights)[0]
    onp.testing.assert_allclose(weighted / weights.sum(), 1 / 12, rtol=0.03)
    unweighted = onp.histogram(mean, edges)[0] / mean.size
    assert onp.max(onp.abs(unweighted * 12 - 1)) > 0.5


def test_position_angle_is_singular_edge_on():
    from virgil.orbits import position_angle_log_jacobian

    el = dict(period=400.0, ecc=0.3, omega=40.0, Omega=110.0, a_mas=20.0)
    with pytest.raises(ValueError, match="singular"):
        KeplerOrbit.from_position_angle(theta=150.0, inc=90.0, **el)
    # Near edge-on, θ away from the node line hardly moves the orbit's
    # phase: |∂M/∂θ| → 0 there, and diverges on the node line, where all
    # the phases crowd (density ∝ 1/|cos i| on a width ∝ |cos i|).
    with jax.enable_x64(True):
        log_j = [
            float(
                position_angle_log_jacobian(
                    theta, 0.3, inc, el["omega"], el["Omega"]
                )
            )
            for inc in (60.0, 89.99)
            for theta in (el["Omega"] + 60.0, el["Omega"])
        ]
    assert log_j[2] < log_j[0] - 7.0
    assert log_j[3] > log_j[1] + 7.0


def test_position_angle_prior_term_in_numpyro_and_fit():
    import numpyro.distributions as dist
    from numpyro.infer.util import log_density

    from virgil.angles import AngleVector
    from virgil.fitting import fit
    from virgil.likelihood import numpyro_model
    from virgil.orbits import position_angle_log_jacobian, position_angle_prior

    truth = _orbit()
    _, theta_true = truth.separation_pa(T_REF)
    el = {k: ORBIT[k] for k in ("period", "ecc", "inc", "omega", "Omega")}

    def orbit_fn(v):
        return KeplerOrbit.from_position_angle(
            theta=v["theta"], a_mas=20.0, t_ref=T_REF, **el
        )

    # Both terms see the fitted eccentricity (the numpyro check fixes it).
    prior_term = position_angle_prior(_ecc_free(orbit_fn))
    # numpyro: the vector's density plus the log-Jacobian at θ.
    model = numpyro_model(
        lambda **kw: None,
        {"theta": AngleVector(), "ecc": dist.Delta(el["ecc"])},
        (),
        likelihoods=[prior_term],
    )
    v = jax.numpy.array([0.3, -1.1])
    theta = float(onp.mod(onp.rad2deg(onp.arctan2(-1.1, 0.3)), 360.0))
    expected = float(AngleVector().log_prob(v)) + float(
        position_angle_log_jacobian(
            theta, el["ecc"], el["inc"], el["omega"], el["Omega"]
        )
    )
    value, _ = log_density(model, (), {}, {"theta_vec": v, "ecc": el["ecc"]})
    assert float(value) == pytest.approx(expected, rel=1e-5)
    # fit: L-BFGS by default (no least-squares form), through the wrap.
    mjd = T_REF + onp.linspace(-20.0, 20.0, 5)  # a short arc
    with jax.enable_x64(True):
        positions = _positions(truth, mjd)
    result = fit(
        lambda **kw: None,
        {"theta": AngleVector(), "ecc": dist.Uniform(0.0, 0.9)},
        (),
        init={"theta": float(theta_true) + 30.0, "ecc": 0.4},
        likelihoods=[positions.term(_ecc_free(orbit_fn)), prior_term],
    )
    assert result.info["method"] == "lbfgs"
    wrapped = (float(result.values["theta"]) - float(theta_true) + 180) % 360
    assert abs(wrapped - 180) < 0.05
    with pytest.raises(TypeError, match="log_norm"):
        fit(
            lambda **kw: None,
            {"theta": AngleVector(), "ecc": dist.Uniform(0.0, 0.9)},
            (),
            init={"theta": float(theta_true), "ecc": 0.4},
            likelihoods=[prior_term],
            method="lm",
        )


def _ecc_free(orbit_fn):
    """``orbit_fn`` with its eccentricity taken from ``values["ecc"]``."""

    def build(v):
        orbit = orbit_fn(v)
        return KeplerOrbit.from_position_angle(
            orbit.period,
            v["theta"],
            v["ecc"],
            orbit.inc,
            orbit.omega,
            orbit.Omega,
            orbit.a_mas,
            t_ref=orbit.t_ref,
        )

    return build


def test_gauss_newton_mass_includes_likelihood_terms():
    # Regression (virgil#211 review): with data=() and the positions as a
    # likelihoods= term, the mass matrix must see the term's residuals, or
    # the angular directions are singular. At noiseless truth the
    # Gauss–Newton matrix is the Hessian of the loss in the vectors.
    from virgil.angles import AngleVector
    from virgil.fitting import gauss_newton_mass
    from virgil.orbits import orientation_priors

    truth = _orbit(Omega=10.0, omega=340.0)
    mjd = T_REF + onp.linspace(0.0, 380.0, 10)
    with jax.enable_x64(True):
        positions = _positions(truth, mjd)

    def orbit_fn(v):
        return KeplerOrbit.from_varpi(
            ORBIT["period"],
            ORBIT["dt_peri"],
            ORBIT["ecc"],
            ORBIT["inc"],
            v["varpi"],
            20.0,
            two_Omega=v["two_Omega"],
            t_ref=T_REF,
        )

    priors = orientation_priors()
    values = {"two_Omega": 20.0, "varpi": 350.0}
    mass = gauss_newton_mass(
        lambda **kw: None,
        priors,
        (),
        values,
        likelihoods=[positions.term(orbit_fn)],
    )
    sites = ("two_Omega_vec", "varpi_vec")
    covariance = onp.asarray(mass["inverse_mass_matrix"][sites])
    assert mass["dense_mass"] == [sites]
    assert onp.all(onp.linalg.eigvalsh(covariance) > 0)

    # Independently: J of all the residuals (the term's and the rings')
    # with respect to the two vectors, and (JᵀJ)⁻¹.
    with jax.enable_x64(True):
        ring = AngleVector()

        def residuals(x):
            v = {"two_Omega": _angle(x[:2]), "varpi": _angle(x[2:])}
            return jax.numpy.concatenate(
                [
                    positions.whitened_residuals(orbit_fn(v)),
                    ring.residuals(x[:2]),
                    ring.residuals(x[2:]),
                ]
            )

        angles = onp.deg2rad([20.0, 20.0, 350.0, 350.0])
        x = jax.numpy.asarray(
            onp.where([1, 0, 1, 0], onp.cos(angles), onp.sin(angles))
        )
        jac = onp.asarray(jax.jacfwd(residuals)(x))
    onp.testing.assert_allclose(
        (jac.T @ jac) @ covariance, onp.eye(4), atol=1e-6
    )


def _angle(vector):
    from virgil.angles import vector_angle

    return vector_angle(vector)


def test_position_angle_keeps_a_small_mean_anomaly_in_float32():
    """Near periastron M is tiny; float32 must not round it to zero."""
    kw = dict(
        period=1000.0,
        theta=5.0,
        ecc=0.9999,
        inc=0.0,
        omega=0.0,
        Omega=0.0,
        a_mas=20.0,
        t_ref=T_REF,
    )
    with jax.enable_x64(True):
        expected = float(KeplerOrbit.from_position_angle(**kw).dt_peri)
    with jax.enable_x64(False):  # float32 even in the x64 CI job
        orbit = KeplerOrbit.from_position_angle(**kw)
    assert orbit.dt_peri.dtype == jax.numpy.float32
    assert expected != 0.0
    assert float(orbit.dt_peri) == pytest.approx(expected, rel=1e-3)


def test_plot_orbit_ensemble_draws_east_left_and_one_period_per_orbit():
    import matplotlib.pyplot as plt

    from virgil.plotting import plot_orbit_ensemble

    plt.switch_backend("Agg")
    # Two orbits as one batched KeplerOrbit, and the truth on its own.
    batch = KeplerOrbit(
        period=onp.array([400.0, 420.0]),
        dt_peri=onp.array([30.0, 35.0]),
        ecc=onp.array([0.4, 0.38]),
        inc=onp.array([60.0, 61.0]),
        omega=onp.array([40.0, 41.0]),
        Omega=onp.array([110.0, 111.0]),
        a_mas=onp.array([20.0, 20.5]),
        t_ref=T_REF,
    )
    truth = _orbit()
    mjd = T_REF + onp.array([0.0, 100.0, 200.0])
    dra, ddec, _ = (onp.asarray(x) for x in truth.relative(mjd))
    sigma = onp.array([0.5, 1.0, 2.0])
    cov = sigma[:, None, None] ** 2 * onp.eye(2)
    positions = PositionData(mjd, dra, ddec, cov)

    fig, ax = plot_orbit_ensemble(batch, positions, truth, n_points=50)

    # East (positive dra) to the left, North up.
    assert ax.get_xlim()[0] > ax.get_xlim()[1]
    assert ax.get_ylim()[0] < ax.get_ylim()[1]
    # Each track is one closed period that starts at periastron.
    tracks = ax.collections[0].get_segments()
    assert len(tracks) == 2
    for track in tracks:
        onp.testing.assert_allclose(track[0], track[-1], atol=1e-3)
    start = onp.array(truth.relative(T_REF + ORBIT["dt_peri"])[:2])
    onp.testing.assert_allclose(ax.lines[0].get_xydata()[0], start, atol=1e-3)
    # One n_sigma ellipse per epoch, centred on it and 2σ across.
    ellipses = ax.patches
    assert len(ellipses) == 3
    for ellipse, x, y, s in zip(ellipses, dra, ddec, sigma):
        assert ellipse.center == pytest.approx((x, y))
        assert ellipse.width == pytest.approx(2 * s, rel=1e-5)
        assert ellipse.height == pytest.approx(2 * s, rel=1e-5)
    plt.close(fig)


def test_thiele_innes_node_180_maps_to_the_twin_in_range():
    # Omega = 180 is outside [0, 180): the twin (0, omega + 180) is the same
    # sky orbit.
    mjd = T_REF + onp.linspace(-200.0, 500.0, 15)
    with jax.enable_x64(True):
        orbit = KeplerOrbit(1000, 0, 0.3, 60, 270, 180, 100)
        back = orbit.to_thiele_innes().to_kepler()
        assert float(back.Omega) == 0.0
        assert float(back.omega) == pytest.approx(90.0, abs=1e-9)
        assert float(back.inc) == pytest.approx(60.0, abs=1e-9)
        assert onp.allclose(
            onp.array(back.relative(mjd))[:2],
            onp.array(orbit.relative(mjd))[:2],
            atol=1e-9,
        )
        for Omega in (0.0, 1e-13, 90.0, 179.99999999, 180.0, 359.0):
            twin = (
                KeplerOrbit(1000, 0, 0.3, 60, 30, Omega, 100)
                .to_thiele_innes()
                .to_kepler()
            )
            assert 0.0 <= float(twin.Omega) < 180.0
            assert 0.0 <= float(twin.omega) < 360.0


@pytest.mark.parametrize("inc", [1e-4, 1e-3, 0.01, 0.1, 179.9, 179.99])
@pytest.mark.parametrize("ecc", [0.3, 1e-7])
def test_state_vector_round_trip_near_face_on(inc, ecc):
    from virgil.orbits import StateVectorOrbit

    mjd = T_REF + onp.linspace(-300.0, 900.0, 13)
    with jax.enable_x64(True):
        orbit = KeplerOrbit(1000.0, 100.0, ecc, inc, 30.0, 60.0, 100.0)
        back = StateVectorOrbit.from_kepler(orbit).to_kepler()
        assert float(back.inc) == pytest.approx(inc, rel=1e-9)
        assert float(back.Omega) == pytest.approx(60.0, abs=1e-9)
        assert onp.allclose(
            onp.array(back.relative(mjd)),
            onp.array(orbit.relative(mjd)),
            atol=1e-12 * 100.0,
        )
        if ecc > 1e-3:
            assert float(back.omega) == pytest.approx(30.0, abs=1e-9)


def test_orientation_priors_inclination_option_gives_the_haar_prior():
    from virgil.orbits import orientation_priors
    from virgil.priors import IsotropicInclination

    # Backward compatible: no inclination unless asked for.
    assert "inc" not in orientation_priors()
    priors = orientation_priors(False, prefix="orbit.", inclination=True)
    assert set(priors) == {"orbit.Omega", "orbit.varpi", "orbit.inc"}
    assert isinstance(priors["orbit.inc"], IsotropicInclination)
    # cos i uniform: a quarter of the mass is below i = 60 degrees.
    inc = onp.asarray(priors["orbit.inc"].sample(jax.random.key(0), (4000,)))
    assert onp.mean(inc < 60.0) == pytest.approx(0.25, abs=0.03)


@pytest.mark.parametrize("a_au, period_yr", [(100.0, 1000.0), (0.01, 0.001)])
def test_total_mass_and_distance_are_finite_in_float32(a_au, period_yr):
    from virgil.orbits import distance_pc, total_mass

    # a^3 / P^2 = 1 in au and Gaussian years: one solar mass (to 4e-5 for the
    # Julian year used here). SI-sized intermediates would overflow float32.
    gaussian_year = 365.256898
    with jax.enable_x64(False):
        f32 = np.float32
        orbit = _orbit(
            period=np.asarray(period_yr * gaussian_year, f32),
            a_mas=np.asarray(a_au * 1000.0, f32),
        )
        mass = total_mass(orbit, np.asarray(1.0, f32))
        assert mass.dtype == f32
        assert onp.isfinite(float(mass))
        assert float(mass) == pytest.approx(1.0, rel=1e-4)
        dist = distance_pc(orbit, mass)
        assert onp.isfinite(float(dist))
        assert float(dist) == pytest.approx(1.0, rel=1e-4)
