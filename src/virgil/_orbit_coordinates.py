"""Smooth sampling coordinates for binary sky orbits.

Experimental: not yet used by ``OrbitPipeline`` (see ``design/orbit_coordinates.md``).

An orbit is sampled as ``(period, z, zeta)`` at a fixed anchor epoch:

* ``zeta`` in the unit disc: the eccentricity vector
  ``k + ih = e exp(i M_a) = zeta**2``, with ``M_a`` the anchor mean anomaly.
  Uniform ``e`` and uniform ``M_a`` is uniform ``zeta``. Two ``zeta`` (±) are
  the same orbit.
* ``z`` in R^4: the Thiele–Innes constants rotated by ``M_a`` (TI′), through
  ``u = (A+G) + i(B-F) = z_u**2`` and ``V = (A-G) - i(B+F) = z_v**2``. An
  isotropic orientation with log-uniform ``a = |z|**2 / 2`` is ``|z|**-4``,
  smooth at face-on and for ``e = 0``, where Campbell's ``cos i``, ``omega``
  and ``Omega`` are not. Four ``z`` (±``z_u``, ±``z_v``) are the same orbit.

Campbell elements are for reporting only (:func:`invariants`).
``z`` and ``zeta`` are arrays with the coordinates on the last axis; the
Thiele–Innes constants are tuples ``(A, B, F, G)`` of arrays (also the
argument of the anchor rotations).

Source: Miles (1965), Moran (1975), Goldstein et al. (2002) and Ford (2006)
give the steps; using them as sampler coordinates is our exact change of
variables (derived, experimental, opt-in), checked by the prior-recovery tests.
"""

import jax
import jax.numpy as np

from .orbits import KeplerOrbit, ThieleInnesOrbit, _thiele_innes_angles


def anchor_rotate(abfg, mean_anomaly):
    """TI′ ``(A′, B′, F′, G′)`` of ``abfg = (A, B, F, G)``.

    ``(A′, F′) = R(-M_a)(A, F)`` and likewise ``(B′, G′)``: the basis
    ``(X, Y)`` is turned by the anchor mean anomaly, so that at the anchor
    ``(A′, B′)`` is roughly the position. The inverse is
    :func:`anchor_unrotate`.
    """
    a, b, f, g = abfg
    c, s = np.cos(mean_anomaly), np.sin(mean_anomaly)
    return a * c + f * s, b * c + g * s, f * c - a * s, g * c - b * s


def anchor_unrotate(abfg_prime, mean_anomaly):
    """The Thiele–Innes constants ``(A, B, F, G)`` of TI′ constants."""
    return anchor_rotate(abfg_prime, -mean_anomaly)


# The map, step by step (derivation: design/orbit_coordinates.md):
# (i) an isotropic orientation is Haar measure on SO(3) (Miles 1965): Omega and
#     omega uniform, cos i uniform.
# (ii) u = (A+G) + i(B-F) = 2a cos²(i/2) e^{i(omega+Omega)} and
#     V = (A-G) - i(B+F) = 2a sin²(i/2) e^{i(omega-Omega)}: the split of
#     ``_thiele_innes_angles``.
# (iii) z = (√u, √V) = √(2a) × the Cayley–Klein parameters
#     (cos(i/2) e^{i(omega+Omega)/2}, sin(i/2) e^{i(omega-Omega)/2})
#     (Goldstein et al. 2002, §4.5).
# (iv) a unit quaternion uniform on S³ is Haar on SO(3), covering it 2:1
#     (Moran 1975); the sign of each square root makes z a 4:1 cover.
# (v) log-uniform a, a = |z|²/2: with d⁴(A,B,F,G) = 4|z_u|²|z_v|² d⁴z and
#     |z_u|²|z_v|² = a² sin²i, the density 1/(a⁴ sin²i) in (A,B,F,G) becomes
#     a⁻² ∝ |z|⁻⁴ in z (change of variables; no approximation).
# (vi) zeta = √e e^{iM_a/2}: uniform on the disc ⇔ e uniform and M_a uniform,
#     since d²(k+ih) = 4e d²zeta. Ford (2006) uses (√e cos ω, √e sin ω); this
#     half-angle form, with the anchor mean anomaly M_a in place of ω, is
#     analytic at e = 0.
# Steps (i)-(iv) and Ford are from the literature; the sampler coordinates are
# our derived change of variables.


def z_to_ti(z):
    """``(A, B, F, G)`` of ``z = (Re z_u, Im z_u, Re z_v, Im z_v)``."""
    z = np.asarray(z)
    u_re = z[..., 0] ** 2 - z[..., 1] ** 2
    u_im = 2 * z[..., 0] * z[..., 1]
    v_re = z[..., 2] ** 2 - z[..., 3] ** 2
    v_im = 2 * z[..., 2] * z[..., 3]
    return (
        (u_re + v_re) / 2,
        (u_im - v_im) / 2,
        -(u_im + v_im) / 2,
        (u_re - v_re) / 2,
    )


def _sqrt_pair(re, im):
    """Principal square root ``(Re, Im)`` of ``re + i im``."""
    root, half = np.sqrt(np.hypot(re, im)), np.arctan2(im, re) / 2
    return root * np.cos(half), root * np.sin(half)


def ti_to_z(a, b, f, g):
    """The canonical ``z`` (``Re z_u >= 0``, ``Re z_v >= 0``) of ``(A, B, F, G)``.

    The other three are ``(±z_u, ±z_v)``.
    """
    zu = _sqrt_pair(a + g, b - f)
    zv = _sqrt_pair(a - g, -(b + f))
    return np.stack([*zu, *zv], axis=-1)


def semimajor(z):
    """``a = |z|² / 2`` (mas)."""
    return np.sum(np.asarray(z) ** 2, axis=-1) / 2


def cos_inclination(z):
    """``cos i = (|z_u|² - |z_v|²) / |z|²``, exact near face-on."""
    z2 = np.asarray(z) ** 2
    zu2, zv2 = z2[..., 0] + z2[..., 1], z2[..., 2] + z2[..., 3]
    return (zu2 - zv2) / (zu2 + zv2)


def z_log_normalization(a_min, a_max):
    """``ln ∫ |z|⁻⁴ d⁴z`` over ``a_min <= a <= a_max``: ``ln(π² ln(a_max/a_min))``."""
    return np.log(np.pi**2 * np.log(a_max / a_min))


def z_log_prior(z, a_min, a_max):
    """Normalized ``ln p(z) = -4 ln|z| - ln(π² ln(a_max/a_min))``.

    The density of an isotropic orientation and log-uniform ``a`` in
    ``[a_min, a_max]``, over all four preimages; ``-inf`` outside the range.
    """
    a = semimajor(z)
    log_p = -2.0 * np.log(2.0 * a) - z_log_normalization(a_min, a_max)
    return np.where((a >= a_min) & (a <= a_max), log_p, -np.inf)


def sample_z(key, shape, a_min, a_max):
    """Draws of ``z`` from the prior of :func:`z_log_prior`."""
    k_dir, k_a = jax.random.split(key)
    direction = jax.random.normal(k_dir, (*shape, 4))
    direction /= np.linalg.norm(direction, axis=-1, keepdims=True)
    log_a = jax.random.uniform(
        k_a, shape, minval=np.log(a_min), maxval=np.log(a_max)
    )
    return direction * np.sqrt(2.0 * np.exp(log_a))[..., None]


def zeta_to_eccentricity(zeta):
    """``(e, M_a)`` of ``zeta = (Re, Im)``: ``e = |zeta|²``, ``M_a`` in ``(-π, π]``.

    ``M_a`` is undefined, and its derivative not finite, at ``zeta = 0``.
    """
    x, y = np.asarray(zeta)[..., 0], np.asarray(zeta)[..., 1]
    return x**2 + y**2, np.arctan2(2 * x * y, x**2 - y**2)


def eccentricity_to_zeta(ecc, mean_anomaly):
    """The canonical ``zeta`` (``Re zeta >= 0``) of ``(e, M_a)``; ``-zeta`` is the twin."""
    root, half = np.sqrt(ecc), mean_anomaly / 2
    x, y = root * np.cos(half), root * np.sin(half)
    sign = np.where(x < 0, -1.0, 1.0)
    return np.stack([sign * x, sign * y], axis=-1)


def zeta_log_prior(zeta):
    """``ln p(zeta) = -ln π`` in the unit disc (uniform ``e`` and ``M_a``), else ``-inf``."""
    inside = np.sum(np.asarray(zeta) ** 2, axis=-1) < 1.0
    return np.where(inside, -np.log(np.pi), -np.inf)


def sample_zeta(key, shape):
    """Draws of ``zeta`` uniform on the unit disc."""
    k_r, k_t = jax.random.split(key)
    radius = np.sqrt(jax.random.uniform(k_r, shape))
    theta = jax.random.uniform(k_t, shape, maxval=2.0 * np.pi)
    return np.stack([radius * np.cos(theta), radius * np.sin(theta)], axis=-1)


def coordinates_to_orbit(period, z, zeta, t_anchor, t_ref=0.0):
    """The :class:`ThieleInnesOrbit` of ``(period, z, zeta)``.

    ``t_anchor`` is the anchor epoch in days since ``t_ref``; periastron is
    ``M_a P / 2π`` before it. At ``zeta = 0`` the phase is undefined and the
    gradient is not finite: use the Kepler equation in ``(k, h)`` there.
    """
    ecc, mean_anomaly = zeta_to_eccentricity(zeta)
    abfg = anchor_unrotate(z_to_ti(z), mean_anomaly)
    dt_peri = t_anchor - mean_anomaly * period / (2.0 * np.pi)
    return ThieleInnesOrbit(period, dt_peri, ecc, *abfg, t_ref=t_ref)


def campbell_to_coordinates(orbit, t_anchor):
    """``(z, zeta)`` of a :class:`KeplerOrbit` (canonical preimages).

    ``t_anchor`` is in days since ``orbit.t_ref``.
    """
    phase = 2 * np.pi * (t_anchor - orbit.dt_peri) / orbit.period
    mean_anomaly = np.arctan2(np.sin(phase), np.cos(phase))
    a, b, f, g, _, _ = orbit.thiele_innes()
    z = ti_to_z(*anchor_rotate((a, b, f, g), mean_anomaly))
    return z, eccentricity_to_zeta(orbit.ecc, mean_anomaly)


def invariants(period, z, zeta):
    """Reporting summaries, invariant under the covers.

    ``period``, ``a`` (mas), ``ecc``, ``mean_anomaly`` (``M_a`` at the
    anchor, radians), ``cos_inc`` and the folded Campbell
    angles ``omega`` in [0, 360) and ``Omega`` in [0, 180) (degrees), from
    :func:`virgil.orbits._thiele_innes_angles`.
    """
    ecc, mean_anomaly = zeta_to_eccentricity(zeta)
    abfg = anchor_unrotate(z_to_ti(z), mean_anomaly)
    _, omega, Omega, _ = _thiele_innes_angles(*abfg)
    return {
        "period": np.asarray(period) + 0 * ecc,
        "a": semimajor(z),
        "ecc": ecc,
        "mean_anomaly": mean_anomaly,
        "cos_inc": cos_inclination(z),
        "omega": omega,
        "Omega": Omega,
    }


def to_kepler(period, z, zeta, t_anchor, t_ref=0.0) -> KeplerOrbit:
    """The :class:`KeplerOrbit` (``0 <= Omega < 180``) of ``(period, z, zeta)``."""
    return coordinates_to_orbit(period, z, zeta, t_anchor, t_ref).to_kepler()
