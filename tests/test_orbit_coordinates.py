"""Smooth orbit sampling coordinates (TI′, z, zeta): maps, priors, smoothness."""

import jax
import jax.numpy as np
import numpy as onp
import pytest
from scipy import integrate, stats

from virgil import _orbit_coordinates as oc
from virgil.orbits import KeplerOrbit

A_MIN, A_MAX = 0.5, 50.0
T_ANCHOR = 10.0


@pytest.fixture(autouse=True)
def _x64():
    with jax.enable_x64(True):
        yield


def _orbit(seed=0, n=1):
    rng = onp.random.default_rng(seed)
    return KeplerOrbit(
        period=rng.uniform(50, 400, n),
        dt_peri=rng.uniform(-200, 200, n),
        ecc=rng.uniform(0.05, 0.9, n),
        inc=onp.degrees(onp.arccos(rng.uniform(-0.95, 0.95, n))),
        omega=rng.uniform(0, 360, n),
        Omega=rng.uniform(0, 360, n),
        a_mas=rng.uniform(1, 20, n),
    )


def test_anchor_rotation_round_trip():
    abfg = tuple(onp.random.default_rng(1).normal(size=(4, 5)))
    back = oc.anchor_unrotate(oc.anchor_rotate(abfg, 0.7), 0.7)
    onp.testing.assert_allclose(back, abfg, atol=1e-12)


def test_z_round_trips():
    z = oc.sample_z(jax.random.key(0), (200,), A_MIN, A_MAX)
    abfg = oc.z_to_ti(z)
    z2 = oc.ti_to_z(*abfg)
    assert onp.all(z2[:, 0] >= 0) and onp.all(z2[:, 2] >= 0)
    onp.testing.assert_allclose(oc.z_to_ti(z2), abfg, atol=1e-10)
    # the canonical z is one of the four preimages (±z_u, ±z_v)
    signs = onp.array(
        [[1, 1, 1, 1], [-1, -1, 1, 1], [1, 1, -1, -1], [-1, -1, -1, -1]]
    )
    for i in range(len(z)):
        assert any(onp.allclose(z2[i] * q, z[i], atol=1e-8) for q in signs)


def test_zeta_round_trip():
    zeta = oc.sample_zeta(jax.random.key(1), (200,))
    ecc, m = oc.zeta_to_eccentricity(zeta)
    z2 = oc.eccentricity_to_zeta(ecc, m)
    assert onp.all(z2[:, 0] >= 0)
    sign = onp.where(zeta[:, 0] < 0, -1.0, 1.0)[:, None]
    onp.testing.assert_allclose(z2, sign * zeta, atol=1e-10)


def test_campbell_round_trip():
    orbit = _orbit(2, 50)
    z, zeta = oc.campbell_to_coordinates(orbit, T_ANCHOR)
    back = oc.coordinates_to_orbit(orbit.period, z, zeta, T_ANCHOR)
    for got, want in zip(
        back.to_kepler().thiele_innes()[:4], orbit.thiele_innes()[:4]
    ):
        onp.testing.assert_allclose(got, want, atol=1e-9)
    onp.testing.assert_allclose(back.ecc, orbit.ecc, atol=1e-12)
    t = np.linspace(-300, 300, 17)[:, None] + 0 * orbit.period
    for got, want in zip(back.sky(t), orbit.relative(t)[:2]):
        onp.testing.assert_allclose(got, want, atol=1e-8)
    inv = oc.invariants(orbit.period, z, zeta)
    onp.testing.assert_allclose(inv["a"], orbit.a_mas, rtol=1e-10)
    onp.testing.assert_allclose(
        inv["cos_inc"], onp.cos(onp.radians(orbit.inc)), atol=1e-10
    )
    fold = onp.mod(orbit.Omega, 180.0)
    onp.testing.assert_allclose(inv["Omega"], fold, atol=1e-7)
    shift = onp.where(onp.mod(orbit.Omega, 360.0) >= 180.0, 180.0, 0.0)
    onp.testing.assert_allclose(
        inv["omega"], onp.mod(orbit.omega - shift, 360.0), atol=1e-7
    )


def test_jacobian_is_a3_sin2_i():
    # dA dB dF dG = a^3 sin^2 i da d(cos i) d(omega) d(Omega)
    rng = onp.random.default_rng(3)
    for _ in range(5):
        a, ci = rng.uniform(1, 10), rng.uniform(-0.9, 0.9)
        w, n = rng.uniform(0, 2 * onp.pi, 2)

        def abfg(p):
            orbit = KeplerOrbit(
                1.0,
                0.0,
                0.0,
                np.degrees(np.arccos(p[1])),
                np.degrees(p[2]),
                np.degrees(p[3]),
                p[0],
            )
            return np.stack(orbit.thiele_innes()[:4])

        p0 = np.array([a, ci, w, n])
        det = np.linalg.det(jax.jacfwd(abfg)(p0))
        onp.testing.assert_allclose(abs(det), a**3 * (1 - ci**2), rtol=1e-8)


def test_z_prior_recovery():
    # a and cos i check the sampler; omega and Omega go through the map.
    n = 20000
    z = oc.sample_z(jax.random.key(4), (n,), A_MIN, A_MAX)
    z = onp.asarray(z)
    zeta = np.zeros((n, 2)) + 0.5
    inv = oc.invariants(1.0, z, zeta)
    ci, a = onp.asarray(inv["cos_inc"]), onp.asarray(inv["a"])
    assert stats.kstest(ci, stats.uniform(-1, 2).cdf).pvalue > 0.01
    lna = onp.log(a)
    assert (
        stats.kstest(
            lna, stats.uniform(onp.log(A_MIN), onp.log(A_MAX / A_MIN)).cdf
        ).pvalue
        > 0.01
    )
    assert (
        stats.kstest(
            onp.asarray(inv["Omega"]), stats.uniform(0, 180).cdf
        ).pvalue
        > 0.01
    )
    assert (
        stats.kstest(
            onp.asarray(inv["omega"]), stats.uniform(0, 360).cdf
        ).pvalue
        > 0.01
    )


def test_z_log_prior_is_normalized():
    # radial quadrature of exp(log p) over the shell: ∫ p 2π² r³ dr = 1
    def density(r):
        z = np.array([r, 0.0, 0.0, 0.0])
        return (
            float(np.exp(oc.z_log_prior(z, A_MIN, A_MAX)))
            * 2
            * onp.pi**2
            * r**3
        )

    total, _ = integrate.quad(
        density, onp.sqrt(2 * A_MIN), onp.sqrt(2 * A_MAX)
    )
    assert total == pytest.approx(1.0, abs=1e-8)
    assert onp.isneginf(
        oc.z_log_prior(np.array([0.0, 0, 0, 0.01]), A_MIN, A_MAX)
    )


def test_z_density_follows_from_the_jacobians():
    # d⁴(ABFG) = 4 |z_u|² |z_v|² d⁴z, and with d⁴(ABFG) = a³ sin²i da dcos(i)
    # dω dΩ (above) the isotropic log-uniform prior is ∝ a⁻² ∝ |z|⁻⁴.
    z = np.array(onp.random.default_rng(11).normal(size=(5, 4)))
    det = jax.vmap(
        lambda x: np.linalg.det(
            jax.jacfwd(lambda y: np.stack(oc.z_to_ti(y)))(x)
        )
    )(z)
    zu2, zv2 = z[:, 0] ** 2 + z[:, 1] ** 2, z[:, 2] ** 2 + z[:, 3] ** 2
    onp.testing.assert_allclose(abs(det), 4 * zu2 * zv2, rtol=1e-10)


def test_zeta_prior_recovery():
    n = 20000
    zeta = oc.sample_zeta(jax.random.key(6), (n,))
    ecc, m = oc.zeta_to_eccentricity(zeta)
    assert (
        stats.kstest(onp.asarray(ecc), stats.uniform(0, 1).cdf).pvalue > 0.01
    )
    assert (
        stats.kstest(
            onp.asarray(m), stats.uniform(-onp.pi, 2 * onp.pi).cdf
        ).pvalue
        > 0.01
    )
    assert float(oc.zeta_log_prior(np.array([1.0, 0.5]))) == -onp.inf
    onp.testing.assert_allclose(
        oc.zeta_log_prior(np.array([0.3, 0.2])), -onp.log(onp.pi)
    )


def test_covers_give_identical_sky_orbits():
    z = oc.sample_z(jax.random.key(7), (), A_MIN, A_MAX)
    zeta = oc.sample_zeta(jax.random.key(8), ())
    t = np.linspace(-100, 100, 9)
    ref = oc.coordinates_to_orbit(200.0, z, zeta, T_ANCHOR).sky(t)
    for s in ([-1, -1, 1, 1], [1, 1, -1, -1], [-1, -1, -1, -1]):
        got = oc.coordinates_to_orbit(
            200.0, z * np.array(s), zeta, T_ANCHOR
        ).sky(t)
        onp.testing.assert_allclose(got, ref, atol=1e-10)
    got = oc.coordinates_to_orbit(200.0, z, -zeta, T_ANCHOR).sky(t)
    onp.testing.assert_allclose(got, ref, atol=1e-10)


def test_positions_match_kepler_orbit_at_random_epochs():
    rng = onp.random.default_rng(9)
    z = oc.sample_z(jax.random.key(9), (), A_MIN, A_MAX)
    zeta = oc.sample_zeta(jax.random.key(10), ())
    kepler = oc.to_kepler(150.0, z, zeta, T_ANCHOR)
    t = rng.uniform(-500, 500, 30)
    dra, ddec, _ = kepler.relative(t)
    sky = oc.coordinates_to_orbit(150.0, z, zeta, T_ANCHOR).sky(t)
    onp.testing.assert_allclose(sky[0], dra, atol=1e-8)
    onp.testing.assert_allclose(sky[1], ddec, atol=1e-8)


def test_anchor_rotation_direction_and_continuity_at_zero_eccentricity():
    z = np.array([1.2, 0.3, 0.4, -0.2])
    eps = 1e-4
    # (A′, F′) = R(-M_a)(A, F): the circular orbit does not depend on M_a,
    # so the sky is continuous as zeta -> 0 from every direction
    skies = [
        onp.array(
            oc.coordinates_to_orbit(
                150.0, z, eps * np.array([np.cos(p), np.sin(p)]), T_ANCHOR
            ).sky(np.linspace(-200, 200, 9))
        )
        for p in (0.0, 0.6, 2.0, -2.5)
    ]
    for other in skies[1:]:
        onp.testing.assert_allclose(other, skies[0], atol=1e-6)
    # at the anchor the position is (B′, A′), to O(e)
    m_a = 1.2
    zeta = eps * np.array([np.cos(m_a / 2), np.sin(m_a / 2)])  # M_a = 1.2
    sky = oc.coordinates_to_orbit(150.0, z, zeta, T_ANCHOR).sky(T_ANCHOR)
    a_t, b_t, _, _ = oc.z_to_ti(z)  # z is in TI′: (A′, B′)
    onp.testing.assert_allclose([sky[1], sky[0]], [a_t, b_t], atol=1e-6)


def test_coordinates_match_closed_form_campbell():
    # z_u = √(2a) cos(i/2) e^{i(ω+M_a+Ω)/2}, z_v = √(2a) sin(i/2) e^{i(ω+M_a-Ω)/2},
    # zeta = √e e^{iM_a/2}, built by hand from Campbell elements
    a, ecc, inc, omega, Omega, period, m_a = (
        7.0,
        0.4,
        1.1,
        0.8,
        2.3,
        120.0,
        0.9,
    )
    orbit = KeplerOrbit(
        period,
        T_ANCHOR - m_a * period / (2 * onp.pi),
        ecc,
        onp.degrees(inc),
        onp.degrees(omega),
        onp.degrees(Omega),
        a,
    )
    zu = (
        onp.sqrt(2 * a)
        * onp.cos(inc / 2)
        * onp.exp(0.5j * (omega + m_a + Omega))
    )
    zv = (
        onp.sqrt(2 * a)
        * onp.sin(inc / 2)
        * onp.exp(0.5j * (omega + m_a - Omega))
    )
    z = np.array([zu.real, zu.imag, zv.real, zv.imag])
    zeta_c = onp.sqrt(ecc) * onp.exp(0.5j * m_a)
    zeta = np.array([zeta_c.real, zeta_c.imag])
    t = onp.random.default_rng(12).uniform(-300, 300, 25)
    sky = oc.coordinates_to_orbit(period, z, zeta, T_ANCHOR).sky(t)
    dra, ddec, _ = orbit.relative(t)
    onp.testing.assert_allclose(sky[0], dra, atol=1e-8)
    onp.testing.assert_allclose(sky[1], ddec, atol=1e-8)


def _sky_sum(z, zeta):
    t = np.linspace(-100, 100, 7)
    dra, ddec = oc.coordinates_to_orbit(120.0, z, zeta, T_ANCHOR).sky(t)
    return np.sum(dra**2 + ddec**2)


def test_gradients_finite_face_on_and_near_circular():
    zeta = np.array([0.5, 0.3])
    # face-on: z_v = 0 (cos i = 1), where Campbell's omega and Omega merge
    z_face = np.array([1.5, 0.4, 0.0, 0.0])
    g = jax.grad(_sky_sum)(z_face, zeta)
    assert onp.all(onp.isfinite(g))
    # the maps themselves are smooth through z_v = 0 and zeta = 0
    assert onp.all(onp.isfinite(jax.jacfwd(oc.z_to_ti)(z_face)))
    assert onp.all(
        onp.isfinite(jax.jacfwd(oc.zeta_to_eccentricity)(np.array([0.3, 0.1])))
    )
    # nearly circular
    z = np.array([1.0, 0.7, 0.5, -0.3])
    g_zeta = jax.grad(_sky_sum, argnums=1)(z, np.array([1e-3, 2e-3]))
    assert onp.all(onp.isfinite(g_zeta))
    # cos i is exact and finite at face-on
    assert float(oc.cos_inclination(z_face)) == pytest.approx(1.0)
    assert onp.all(
        onp.isfinite(
            jax.grad(oc.cos_inclination)(np.array([1.0, 0.2, 1e-9, 0.0]))
        )
    )
