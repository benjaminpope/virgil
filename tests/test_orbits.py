"""Orbit conventions (design/orbit_scene_joint_fitting.md §2, §5.1–5.2)."""

import itertools

import jax
import numpy as onp
import pytest
from scipy.optimize import brentq

pytest.importorskip("jaxoplanet")

from virgil.orbits import (  # noqa: E402
    KeplerOrbit,
    PositionData,
    ThieleInnesOrbit,
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
    # Traced values are not checked (they may be mid-optimisation).
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

    # 1 au at 1 pc is 1000 mas; a 1-year orbit then has 1 solar mass.
    earth = _orbit(period=365.25, a_mas=1000.0)
    assert float(total_mass(earth, 1.0)) == pytest.approx(1.0)
    assert float(distance_pc(earth, 1.0)) == pytest.approx(1.0)
    orbit = _orbit()
    assert float(
        distance_pc(orbit, total_mass(orbit, 123.0))
    ) == pytest.approx(123.0)


def test_axial_von_mises_is_a_normalised_prior_with_period_180():
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
