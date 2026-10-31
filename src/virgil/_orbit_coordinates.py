"""Smooth sampling coordinates for binary sky orbits.

Experimental: not yet used by ``OrbitPipeline`` (see ``design/``).

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
Maps take and return arrays with the coordinates on the last axis.

Source: Thiele–Innes constants; the quaternion form of SO(3)'s Haar measure
(Shoemake 1992) applied to ``u``, ``V``. Derived here, checked by the
prior-recovery tests.
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


def z_to_ti(z):
    """``(A, B, F, G)`` of ``z = (Re z_u, Im z_u, Re z_v, Im z_v)``."""
    z = np.asarray(z)
    u = (z[..., 0] + 1j * z[..., 1]) ** 2
    v = (z[..., 2] + 1j * z[..., 3]) ** 2
    return (
        (u.real + v.real) / 2,
        (u.imag - v.imag) / 2,
        -(u.imag + v.imag) / 2,
        (u.real - v.real) / 2,
    )


def ti_to_z(a, b, f, g):
    """The canonical ``z`` (``Re z_u >= 0``, ``Re z_v >= 0``) of ``(A, B, F, G)``.

    The other three are ``(±z_u, ±z_v)``.
    """
    u, v = (a + g) + 1j * (b - f), (a - g) - 1j * (b + f)
    zu, zv = np.sqrt(u), np.sqrt(v)  # principal roots
    return np.stack([zu.real, zu.imag, zv.real, zv.imag], axis=-1)


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
    """``(e, M_a)`` of ``zeta = (Re, Im)``: ``e = |zeta|²``, ``M_a`` in ``(-π, π]``."""
    zeta = np.asarray(zeta)
    k_h = (zeta[..., 0] + 1j * zeta[..., 1]) ** 2
    return np.abs(k_h), np.angle(k_h)


def eccentricity_to_zeta(ecc, mean_anomaly):
    """The canonical ``zeta`` (``Re zeta >= 0``) of ``(e, M_a)``; ``-zeta`` is the twin."""
    zeta = np.sqrt(ecc + 0j) * np.exp(0.5j * mean_anomaly)
    zeta = np.where(zeta.real < 0, -zeta, zeta)
    return np.stack([zeta.real, zeta.imag], axis=-1)


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
    mean_anomaly = np.angle(
        np.exp(2j * np.pi * (t_anchor - orbit.dt_peri) / orbit.period)
    )
    a, b, f, g, _, _ = orbit.thiele_innes()
    z = ti_to_z(*anchor_rotate((a, b, f, g), mean_anomaly))
    return z, eccentricity_to_zeta(orbit.ecc, mean_anomaly)


def invariants(period, z, zeta):
    """Reporting summaries, invariant under the covers.

    ``period``, ``a`` (mas), ``ecc``, ``cos_inc`` and the folded Campbell
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
        "cos_inc": cos_inclination(z),
        "omega": omega,
        "Omega": Omega,
    }


def to_kepler(period, z, zeta, t_anchor, t_ref=0.0) -> KeplerOrbit:
    """The :class:`KeplerOrbit` (``0 <= Omega < 180``) of ``(period, z, zeta)``."""
    return coordinates_to_orbit(period, z, zeta, t_anchor, t_ref).to_kepler()
