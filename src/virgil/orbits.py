"""Keplerian orbits of a binary's secondary about its primary.

The position of the secondary relative to the primary is ``(dra, ddec, dz)``:
East and North in mas, and ``dz`` positive away from the observer. ``inc``,
``Omega`` and ``omega`` are the visual-binary angles (degrees; ``omega`` is
the secondary's, the spectroscopic one is ``omega - 180``), ``period`` is in
days, ``a_mas`` the angular semimajor axis, and ``dt_peri`` the time of
periastron minus the static ``t_ref`` (days). See
[conventions](../conventions.md#orbits) for the full definitions and
degeneracies, and [orbit methods](../orbit_methods.md) for the solvers and
references. Kepler's equation is solved by jaxoplanet, an optional
dependency (``pip install "virgil-astro[orbits]"``).
"""

import functools
import math
import warnings

import jax
import jax.numpy as np
import numpy as onp

import equinox as eqx
import numpyro.distributions as dist
import zodiax as zdx
from numpyro.distributions import constraints

from ._deprecate import renamed
from ._linear import LinearMarginal
from ._geometry import position_angle, separation_pa
from ._grid import MIN_BATCH_SIZE, batch_size_or_default, batch_visibilities
from ._utils import concrete
from .angles import AngleVector
from .priors import IsotropicInclination
from ._deprecate import external_stacklevel


__all__ = [
    "AxialVonMises",
    "KeplerOrbit",
    "PositionData",
    "RVData",
    "StateVectorOrbit",
    "ThieleInnesOrbit",
    "distance_pc",
    "orientation_from_varpi",
    "position_angle_log_jacobian",
    "position_angle_prior",
    "orientation_priors",
    "period_grid",
    "starting_orbits",
    "total_mass",
]


def _jaxoplanet():
    """The jaxoplanet package, with an install hint if it is missing."""
    try:
        import jaxoplanet.core
        import jaxoplanet.orbits.keplerian
    except ImportError as err:
        raise ImportError(
            'Orbits need jaxoplanet: pip install "virgil-astro[orbits]".'
        ) from err
    return jaxoplanet


def _check(cls, checks):
    """Reject concrete out-of-domain values; traced ones pass unchecked."""
    for name, value, ok, domain in checks:
        value = concrete(value)
        if value is not None and not onp.all(onp.isfinite(value) & ok(value)):
            raise ValueError(f"{cls}: {name} must be {domain}, not {value}.")


def _check_orbit(cls, orbit):
    _check(
        cls,
        (
            ("period", orbit.period, lambda x: x > 0, "positive"),
            ("ecc", orbit.ecc, lambda x: (x >= 0) & (x < 1), "in [0, 1)"),
            ("dt_peri", orbit.dt_peri, lambda x: True, "finite"),
        ),
    )


def _days_since(mjd, t_ref):
    """``mjd - t_ref`` in days, in float64 on the host when ``mjd`` is known.

    Under ``jit`` pass times already relative to ``t_ref`` (float32 resolves
    MJD 60000 only to about 0.004 d).
    """
    if isinstance(mjd, jax.core.Tracer):
        return mjd - t_ref
    return np.asarray(onp.asarray(mjd, dtype=onp.float64) - t_ref)


# Times this far (days) from an orbit's t_ref = 0 are MJDs (MJD 15000 is
# 1899), not offsets: the orbit was built without its t_ref.
_MJD_LIKE_DAYS = 15000.0


def _warn_if_mjd_without_t_ref(orbit_t_ref, dt):
    """Warn when an orbit with the default ``t_ref = 0`` is given MJDs."""
    if orbit_t_ref != 0.0:
        return
    days = concrete(dt)
    if (
        days is not None
        and onp.size(days)
        and (onp.max(onp.abs(days)) > _MJD_LIKE_DAYS)
    ):
        warnings.warn(
            "The orbit has t_ref = 0 but is evaluated at times that look "
            f"like MJDs (up to {onp.max(onp.abs(days)):.0f} d): its "
            "dt_peri is then counted from MJD 0. Give the orbit a t_ref "
            "near the data (e.g. KeplerOrbit(..., t_ref=60500.0)). "
            "This will become an error in virgil 0.5.",
            UserWarning,
            stacklevel=external_stacklevel(),
        )


# IAU 2015 nominal values (SI): the solar mass parameter, the astronomical
# unit and the day used throughout (periods are in days, a in au).
_GM_SUN = 1.3271244e20  # m³ s⁻²
_AU_M = 149597870700.0  # m
_DAY_S = 86400.0  # s
# M[M_sun] = _KEPLER_FACTOR a[au]^3 / P[day]^2, evaluated once in Python
# float64 so float32 arrays never see SI-sized intermediates (a_m^3 overflows).
_KEPLER_FACTOR = 4 * math.pi**2 * _AU_M**3 / (_GM_SUN * _DAY_S**2)


def _unit_orbit(dt, period, dt_peri, ecc):
    """Thiele–Innes ``X = cos E - e`` and ``Y = √(1 - e²) sin E``.

    Computed from the true anomaly: ``X = (r/a) cos f``, ``Y = (r/a) sin f``.
    """
    mean_anomaly = 2.0 * np.pi * (dt - dt_peri) / period
    dtype = np.result_type(mean_anomaly, ecc, float)
    eps = np.finfo(dtype).eps
    ecc = np.minimum(np.asarray(ecc, dtype=dtype), 1.0 - eps)
    sin_f, cos_f = _jaxoplanet().core.kepler(  # true anomaly
        mean_anomaly, ecc * np.ones_like(mean_anomaly)
    )
    radius = (1.0 - ecc**2) / np.maximum(1.0 + ecc * cos_f, eps)
    return radius * cos_f, radius * sin_f


def _velocity(position, dt):
    """Time derivative (per day) of ``position(dt)``, exactly, by a JVP."""
    return jax.jvp(position, (dt,), (np.ones_like(dt),))[1]


class KeplerOrbit(zdx.Base):
    """A Keplerian orbit of the secondary relative to the primary.

    Parameters
    ----------
    period, dt_peri : float
        Orbital period (days) and time of periastron minus ``t_ref`` (days).
    ecc : float
        Eccentricity, ``0 <= ecc < 1``.
    inc, omega, Omega : float
        Inclination (``0 <= inc < 180``; below 90 the position angle
        increases with time), argument of periastron and position angle of
        the ascending node, in degrees ([conventions](../conventions.md#orbits)).
    a_mas : float
        Angular semimajor axis (mas).
    t_ref : float, optional
        Reference time (MJD, float64, static).

    Notes
    -----
    (Omega + 180°, omega + 180°) gives the same sky positions with ``dz``
    reversed: positions alone cannot tell them apart.
    """

    period: jax.Array
    dt_peri: jax.Array
    ecc: jax.Array
    inc: jax.Array
    omega: jax.Array
    Omega: jax.Array
    a_mas: jax.Array
    t_ref: float = eqx.field(static=True)

    def __init__(
        self, period, dt_peri, ecc, inc, omega, Omega, a_mas, t_ref=0.0
    ):
        self.period = np.asarray(period, dtype=float)
        self.dt_peri = np.asarray(dt_peri, dtype=float)
        self.ecc = np.asarray(ecc, dtype=float)
        self.inc = np.asarray(inc, dtype=float)
        self.omega = np.asarray(omega, dtype=float)
        self.Omega = np.asarray(Omega, dtype=float)
        self.a_mas = np.asarray(a_mas, dtype=float)
        self.t_ref = float(t_ref)

    def __check_init__(self):
        _check_orbit("KeplerOrbit", self)
        _check(
            "KeplerOrbit",
            (
                (
                    "inc",
                    self.inc,
                    lambda x: (x >= 0) & (x < 180),
                    "in [0, 180)",
                ),
                ("a_mas", self.a_mas, lambda x: x >= 0, "non-negative"),
                ("omega", self.omega, lambda x: True, "finite"),
                ("Omega", self.Omega, lambda x: True, "finite"),
            ),
        )

    @classmethod
    def from_varpi(
        cls,
        period,
        dt_peri,
        ecc,
        inc,
        varpi,
        a_mas,
        *,
        Omega=None,
        two_Omega=None,
        t_ref=0.0,
    ):
        """The orbit with longitude of periastron ``varpi`` = Ω + ω.

        Give ``two_Omega`` (2Ω, degrees) for positions alone, which fix Ω
        only modulo 180°, or ``Omega`` when other data (RVs) fix the node;
        see [`orientation_priors`][virgil.orbits.orientation_priors],
        which samples them as angle vectors. ``omega`` is ``varpi - Omega``,
        the secondary's argument of periastron, as everywhere in virgil.
        """
        omega, Omega = orientation_from_varpi(
            varpi, Omega=Omega, two_Omega=two_Omega
        )
        return cls(period, dt_peri, ecc, inc, omega, Omega, a_mas, t_ref)

    @classmethod
    def from_position_angle(
        cls, period, theta, ecc, inc, omega, Omega, a_mas, t_ref=0.0
    ):
        """The orbit whose position angle at ``t_ref`` is ``theta``.

        An alternative to ``dt_peri`` for short arcs: astrometry measures the position
        angle at an epoch directly
        ([derivation](../orbit_methods.md#position-angle)). Sample ``theta`` as an
        [`AngleVector`][virgil.angles.AngleVector] and add
        [`position_angle_prior`][virgil.orbits.position_angle_prior] to the likelihood
        terms so the prior stays uniform in the time of periastron.
        **Singular at i = 90°** (a concrete ``inc`` of 90° is rejected) and badly
        conditioned near edge-on: keep ``dt_peri`` or use
        [`StateVectorOrbit`][virgil.orbits.StateVectorOrbit].

        Source: Thompson et al. (2023), [references](../references.md#thompson2023).

        Parameters
        ----------
        theta : float
            Position angle of the secondary at ``t_ref`` (degrees, North through East).
        period, ecc, inc, omega, Omega, a_mas, t_ref
            As for ``KeplerOrbit``.
        """
        inc_value = concrete(inc)
        if inc_value is not None and onp.any(
            onp.abs(onp.cos(onp.deg2rad(onp.asarray(inc_value, float))))
            < 1e-12
        ):
            raise ValueError(
                "KeplerOrbit.from_position_angle: inc = 90° is singular "
                "(the position angle takes only two values); use dt_peri "
                "or StateVectorOrbit for edge-on orbits."
            )
        f = _true_anomaly_at_ref(theta, inc, omega, Omega)
        ecc_ = np.asarray(ecc, float)
        ecc_anomaly = np.arctan2(
            np.sqrt(1.0 - ecc_**2) * np.sin(f), ecc_ + np.cos(f)
        )
        mean = ecc_anomaly - ecc_ * np.sin(ecc_anomaly)
        # E from atan2 is in (-π, π], so M is too: only M = π needs moving.
        # (Shifting by π before a modulo would round small M to zero in float32.)
        mean = np.where(mean >= np.pi, mean - 2.0 * np.pi, mean)
        dt_peri = -mean * np.asarray(period, float) / (2.0 * np.pi)
        return cls(period, dt_peri, ecc, inc, omega, Omega, a_mas, t_ref)

    def thiele_innes(self):
        """The Thiele–Innes constants ``(A, B, F, G, C, H)`` (mas).

        ``ddec = A X + F Y``, ``dra = B X + G Y`` and ``dz = C X + H Y``,
        with ``X = cos E - e`` and ``Y = √(1 - e²) sin E``.
        """
        i, w, n = (
            np.deg2rad(self.inc),
            np.deg2rad(self.omega),
            np.deg2rad(self.Omega),
        )
        a, ci = self.a_mas, np.cos(i)
        return (
            a * (np.cos(w) * np.cos(n) - np.sin(w) * np.sin(n) * ci),
            a * (np.cos(w) * np.sin(n) + np.sin(w) * np.cos(n) * ci),
            a * (-np.sin(w) * np.cos(n) - np.cos(w) * np.sin(n) * ci),
            a * (-np.sin(w) * np.sin(n) + np.cos(w) * np.cos(n) * ci),
            a * np.sin(w) * np.sin(i),
            a * np.cos(w) * np.sin(i),
        )

    def _relative(self, dt):
        x, y = _unit_orbit(dt, self.period, self.dt_peri, self.ecc)
        a, b, f, g, c, h = self.thiele_innes()
        return np.stack([b * x + g * y, a * x + f * y, c * x + h * y])

    def relative(self, mjd):
        """Position of the secondary relative to the primary (mas).

        Parameters
        ----------
        mjd : array-like
            Times (MJD). Concrete times are offset from ``t_ref`` in float64
            first. Under ``jit`` the offset is taken in the traced
            precision, which in float32 is good to only about 0.004 d near
            MJD 60000.

        Returns
        -------
        tuple of arrays
            ``(dra, ddec, dz)``, each shaped like ``mjd``.
        """
        dt = _days_since(mjd, self.t_ref)
        _warn_if_mjd_without_t_ref(self.t_ref, dt)
        return tuple(self._relative(dt))

    def relative_velocity(self, mjd):
        """``d(dra, ddec, dz)/dt`` (mas per day), exactly."""
        dt = _days_since(mjd, self.t_ref)
        return tuple(_velocity(self._relative, dt))

    def _frame(self, dt):  # deliberate: also called by models.py
        dra, ddec, dz = self._relative(dt)
        line_pa = position_angle(dra, ddec)
        constant = np.zeros_like(line_pa)
        opposite = line_pa + 180.0
        return {
            "line_pa": line_pa,
            "towards_primary": np.where(
                opposite >= 360.0, opposite - 360.0, opposite
            ),
            "line_tilt": np.rad2deg(np.arctan2(dz, np.hypot(dra, ddec))),
            "node_pa": self.Omega + constant,
            "inc": self.inc + constant,
            "apparent_inc": np.rad2deg(
                np.arccos(np.abs(np.cos(np.deg2rad(self.inc))))
            )
            + constant,
        }

    def frame(self, mjd):
        """Angles of the binary frame at ``mjd`` (degrees), by name.

        * ``line_pa``: position angle of the line of centres, primary to
          secondary; ``towards_primary`` is ``line_pa + 180``, wrapped to
          [0, 360).
        * ``line_tilt``: the line of centres' elevation out of the sky,
          ``arcsin(dz / |r|)``, positive when the secondary is farther.
        * ``node_pa``: ``Omega``, the line of nodes of the orbital plane.
        * ``inc``: the orbit's inclination (0–180); ``apparent_inc``:
          ``arccos|cos inc|`` (0–90), the projected tilt, for components
          whose ``inc`` is an apparent inclination.
        """
        dt = _days_since(mjd, self.t_ref)
        _warn_if_mjd_without_t_ref(self.t_ref, dt)
        return self._frame(dt)

    def separation_pa(self, mjd):
        """Separation (mas) and position angle (degrees, North through East,
        in [0, 360)) of the secondary from the primary."""
        dra, ddec, _ = self.relative(mjd)
        return separation_pa(dra, ddec)

    def to_thiele_innes(self):
        """The same sky orbit as a :class:`ThieleInnesOrbit`."""
        a, b, f, g, _, _ = self.thiele_innes()
        return ThieleInnesOrbit(
            self.period, self.dt_peri, self.ecc, a, b, f, g, self.t_ref
        )

    def to_jaxoplanet(self):
        """A jaxoplanet ``OrbitalBody`` for this orbit, and the factor that
        turns its relative positions into mas.

        jaxoplanet's times are days since ``t_ref``, its axes are
        (X, Y, Z) = (North, East, toward the observer), and its ω is the
        primary's: ``(dra, ddec, dz) = (Y, X, -Z) * scale``.
        """
        keplerian = _jaxoplanet().orbits.keplerian
        Body, Central, OrbitalBody = (
            keplerian.Body,
            keplerian.Central,
            keplerian.OrbitalBody,
        )
        body = OrbitalBody(
            Central(mass=1.0, radius=1.0),
            Body(
                period=self.period,
                time_peri=self.dt_peri,
                eccentricity=self.ecc,
                inclination=np.deg2rad(self.inc),
                omega_peri=np.deg2rad(self.omega) - np.pi,
                asc_node=np.deg2rad(self.Omega),
            ),
        )
        return body, self.a_mas / body.semimajor

    @classmethod
    def from_jaxoplanet(cls, body, a_mas, t_ref=0.0):
        """The orbit of a jaxoplanet ``OrbitalBody`` whose times are days
        since ``t_ref``, with angular semimajor axis ``a_mas``."""
        omega = np.rad2deg(
            np.arctan2(body.sin_omega_peri, body.cos_omega_peri)
        )
        Omega = np.rad2deg(np.arctan2(body.sin_asc_node, body.cos_asc_node))
        return cls(
            period=body.period,
            dt_peri=body.time_peri,
            ecc=body.eccentricity,
            inc=np.rad2deg(
                np.arctan2(body.sin_inclination, body.cos_inclination)
            ),
            omega=np.mod(omega + 180.0, 360.0),
            Omega=np.mod(Omega, 360.0),
            a_mas=a_mas,
            t_ref=t_ref,
        )


@jax.jit  # one dispatch per eager call; the maths is ~20 ns per draw under vmap
def _thiele_innes_angles(a, b, f, g):
    """``(inc, omega, Omega, a_mas)`` (degrees, mas) of Thiele–Innes constants."""
    total = np.arctan2(b - f, a + g)  # omega + Omega
    difference = np.arctan2(-(b + f), a - g)  # omega - Omega
    omega = np.rad2deg((total + difference) / 2)
    Omega = np.rad2deg((total - difference) / 2)
    # Move to the twin with 0 <= Omega < 180 (shifting both by 180° leaves
    # the sky orbit unchanged); omega ends in [0, 360).
    shift = np.floor(Omega / 180.0) * 180.0
    Omega = Omega - shift
    omega = omega - shift
    again = Omega >= 180.0  # Omega = -tiny: 180 - tiny rounds to 180
    Omega = np.where(again, Omega - 180.0, Omega)
    omega = np.where(again, omega - 180.0, omega)
    omega = np.mod(omega, 360.0)
    omega = np.where(omega >= 360.0, 0.0, omega)
    # The singular values of [[A, B], [F, G]] are a and a |cos i|, and
    # in terms of p = |(A - G, B + F)|, q = |(A + G, B - F)| they are
    # (q ± p) / 2 (p, q are differences of norms that cannot cancel).
    # So tan²(i/2) = p / q: exact near i = 0 and i = 180°, unlike arccos.
    p_norm = np.hypot(a - g, b + f)
    q_norm = np.hypot(a + g, b - f)
    a_mas = (q_norm + p_norm) / 2
    inc = 2.0 * np.rad2deg(np.arctan2(np.sqrt(p_norm), np.sqrt(q_norm)))
    return inc, omega, Omega, a_mas


class ThieleInnesOrbit(zdx.Base):
    """A sky orbit in Thiele–Innes form: linear in ``A``, ``B``, ``F``, ``G``.

    ``ddec = A X + F Y`` and ``dra = B X + G Y``, with ``X = cos E - e`` and
    ``Y = √(1 - e²) sin E``. For fixed ``(period, dt_peri, ecc)`` the
    positions are linear in the four constants, so a starting orbit is a
    linear least-squares solve on a grid of those three.

    Parameters
    ----------
    period, dt_peri, ecc : float
        As in :class:`KeplerOrbit`.
    A, B, F, G : float
        Thiele–Innes constants (mas).
    t_ref : float, optional
        Reference time (MJD, static float64).
    """

    period: jax.Array
    dt_peri: jax.Array
    ecc: jax.Array
    A: jax.Array
    B: jax.Array
    F: jax.Array
    G: jax.Array
    t_ref: float = eqx.field(static=True)

    def __init__(self, period, dt_peri, ecc, A, B, F, G, t_ref=0.0):
        self.period = np.asarray(period, dtype=float)
        self.dt_peri = np.asarray(dt_peri, dtype=float)
        self.ecc = np.asarray(ecc, dtype=float)
        self.A = np.asarray(A, dtype=float)
        self.B = np.asarray(B, dtype=float)
        self.F = np.asarray(F, dtype=float)
        self.G = np.asarray(G, dtype=float)
        self.t_ref = float(t_ref)

    def __check_init__(self):
        _check_orbit("ThieleInnesOrbit", self)
        finite = lambda x: True  # noqa: E731
        _check(
            "ThieleInnesOrbit",
            tuple((k, getattr(self, k), finite, "finite") for k in "ABFG"),
        )

    def sky(self, mjd):
        """``(dra, ddec)`` of the secondary from the primary (mas)."""
        dt = _days_since(mjd, self.t_ref)
        x, y = _unit_orbit(dt, self.period, self.dt_peri, self.ecc)
        return self.B * x + self.G * y, self.A * x + self.F * y

    def to_kepler(self):
        """The :class:`KeplerOrbit` with these sky positions.

        Positions fix ``Omega`` only modulo 180°: the result has
        ``0 <= Omega < 180``, and (``Omega + 180``, ``omega + 180``) is the
        other solution, with ``dz`` reversed.
        """
        a, b, f, g = self.A, self.B, self.F, self.G
        half = (a**2 + b**2 + f**2 + g**2) / 2
        if concrete(half) is not None and not onp.all(concrete(half) > 0):
            raise ValueError(
                "The Thiele–Innes constants are all zero: the positions "
                "carry no orbit (all at the primary)."
            )
        inc, omega, Omega, a_mas = _thiele_innes_angles(a, b, f, g)
        return KeplerOrbit(
            period=self.period,
            dt_peri=self.dt_peri,
            ecc=self.ecc,
            inc=inc,
            omega=omega,
            Omega=Omega,
            a_mas=a_mas,
            t_ref=self.t_ref,
        )


class PositionData(zdx.Base):
    """Measured positions of the secondary relative to the primary.

    For starting orbits from per-epoch binary fits, and for published
    positions with no raw data. **It ignores the scene:** when the source
    is more than two point stars, positions fitted at about λ/D resolution
    can be biased, and the orbit should be fitted to the visibilities.

    Parameters
    ----------
    mjd : array-like
        Time of each position (MJD).
    dra, ddec : array-like
        Positions (mas), East and North.
    cov : array-like
        Covariance of ``(dra, ddec)`` at each epoch, shape ``(n, 2, 2)``
        (mas²), e.g. from [`laplace_cov`][virgil.inference.laplace_cov].
    t_ref : float, optional
        Reference time (MJD, static float64); by default the first epoch.
    """

    dt: jax.Array
    dra: jax.Array
    ddec: jax.Array
    whitener: jax.Array  # (n, 2, 2): L⁻¹ with cov = L Lᵀ
    t_ref: float = eqx.field(static=True)

    def __init__(self, mjd, dra, ddec, cov, t_ref=None):
        mjd = onp.atleast_1d(onp.asarray(mjd, dtype=onp.float64))
        cov = onp.asarray(cov, dtype=float).reshape(-1, 2, 2)
        if cov.shape[0] != mjd.size:
            raise ValueError(
                f"cov has {cov.shape[0]} epochs but there are {mjd.size}."
            )
        dra = onp.atleast_1d(onp.asarray(dra, dtype=float))
        ddec = onp.atleast_1d(onp.asarray(ddec, dtype=float))
        for name, values in (("dra", dra), ("ddec", ddec)):
            if values.shape != mjd.shape:
                raise ValueError(
                    f"{name} has shape {values.shape} but there are "
                    f"{mjd.size} epochs."
                )
        self.t_ref = float(mjd.min() if t_ref is None else t_ref)
        self.dt = np.asarray(mjd - self.t_ref)
        self.dra = np.asarray(dra)
        self.ddec = np.asarray(ddec)
        self.whitener = np.asarray(onp.linalg.inv(onp.linalg.cholesky(cov)))

    @classmethod
    def from_sep_pa(cls, mjd, sep, pa, sep_err, pa_err, t_ref=None):
        """Positions given as separation (mas) and position angle (degrees,
        North through East), with independent errors on each."""
        sep, pa_rad = onp.asarray(sep, float), onp.deg2rad(pa)
        dra, ddec = sep * onp.sin(pa_rad), sep * onp.cos(pa_rad)
        # The Jacobian of (dra, ddec) with respect to (sep, pa).
        jac = onp.stack(
            [
                onp.stack([onp.sin(pa_rad), sep * onp.cos(pa_rad)], -1),
                onp.stack([onp.cos(pa_rad), -sep * onp.sin(pa_rad)], -1),
            ],
            -2,
        )
        errors = onp.stack(
            [onp.asarray(sep_err, float), onp.deg2rad(pa_err)], -1
        )
        cov = jac * errors[..., None, :] ** 2 @ jac.swapaxes(-1, -2)
        return cls(mjd, dra, ddec, cov, t_ref)

    def model(self, orbit, north_angle=None, plate_scale=None):
        """The positions ``(dra, ddec)`` these data would measure (mas).

        The orbit's sky positions, seen through this dataset's astrometric
        calibration: ``m R(δ) (dra, ddec)`` with ``δ = north_angle``
        (degrees), ``m = plate_scale`` and
        ``R(δ) = [[cos δ, sin δ], [-sin δ, cos δ]]``. So a companion at
        true PA θ and separation ρ is measured at PA θ + δ and separation
        m ρ: ``north_angle`` is the error *added* to every measured
        position angle, as in
        [`OIData.with_north_angle`][virgil.oidata.OIData.with_north_angle].
        ``None`` (the default) leaves the positions untouched.
        """
        dt = self.dt + (self.t_ref - orbit.t_ref)
        dra, ddec, _ = orbit._relative(dt)
        if north_angle is not None:
            c = np.cos(np.deg2rad(north_angle))
            s = np.sin(np.deg2rad(north_angle))
            dra, ddec = c * dra + s * ddec, -s * dra + c * ddec
        if plate_scale is not None:
            dra, ddec = plate_scale * dra, plate_scale * ddec
        return dra, ddec

    def whitened_residuals(self, orbit, north_angle=None, plate_scale=None):
        """``L⁻¹ (data - model)`` for every epoch, flattened (2n,).

        ``north_angle`` and ``plate_scale`` are this dataset's calibration
        terms (see [`model`][virgil.orbits.PositionData.model]).
        """
        dra, ddec = self.model(orbit, north_angle, plate_scale)
        resid = np.stack([self.dra - dra, self.ddec - ddec], -1)
        return np.einsum("nij,nj->ni", self.whitener, resid).reshape(-1)

    def loglike(self, orbit, north_angle=None, plate_scale=None):
        """Gaussian log-likelihood of the positions under ``orbit``."""
        resid = self.whitened_residuals(orbit, north_angle, plate_scale)
        log_det = np.sum(np.log(np.abs(np.diagonal(self.whitener, 0, 1, 2))))
        return (
            -0.5 * resid @ resid + log_det - resid.size / 2 * np.log(2 * np.pi)
        )

    def term(self, orbit, north_angle=None, plate_scale=None):
        """A likelihood term for [`fit`][virgil.fitting.fit]'s ``likelihoods``.

        Source: Thompson et al. (2023), [references](../references.md#thompson2023).

        Parameters
        ----------
        orbit : callable
            Maps the fitted values (a dict, by path or keyword) to a
            [`KeplerOrbit`][virgil.orbits.KeplerOrbit].
        north_angle, plate_scale : str, optional
            Keys of ``priors`` holding this dataset's North angle (degrees, added to
            every position angle) and plate scale (a factor), as in
            [`model`][virgil.orbits.PositionData.model]; each dataset needs its own
            names, and the priors must be stated (no default; see
            [calibration terms](../orbit_methods.md#position-term)).
        """
        if north_angle is None and plate_scale is None:
            return _Term(self, lambda values: (orbit(values),))

        def build(values):
            return (
                orbit(values),
                None if north_angle is None else values[north_angle],
                None if plate_scale is None else values[plate_scale],
            )

        return _Term(self, build)


@jax.jit
def _thiele_innes_fit(  # deliberate: jit boundary vmapped by _fit_batch
    dt, dra, ddec, whitener, period, dt_peri, ecc
):
    """Weighted least-squares ``(A, B, F, G)`` and χ² at one grid point."""
    x, y = _unit_orbit(dt, period, dt_peri, ecc)
    zero = np.zeros_like(x)
    # Rows (dra, ddec) per epoch; columns A, B, F, G.
    design = np.stack(
        [
            np.stack([zero, x, zero, y], -1),
            np.stack([x, zero, y, zero], -1),
        ],
        -2,
    )
    data = np.stack([dra, ddec], -1)
    design = np.einsum("nij,njk->nik", whitener, design).reshape(-1, 4)
    data = np.einsum("nij,nj->ni", whitener, data).reshape(-1)
    params = np.linalg.lstsq(design, data)[0]
    resid = data - design @ params
    return params, resid @ resid


_N_SIGN_PAIRS = 16  # epoch pairs tried per grid point by the sign-free fit


@jax.jit
def _thiele_innes_fit_sign_free(  # deliberate: jit/vmap boundary, 59 lines
    dt, dra, ddec, whitener, period, dt_peri, ecc
):
    """Thiele–Innes fit when each position is known only up to a sign.

    For each of the best-conditioned pairs of epochs and both relative signs,
    solve for the constants, fix every sign, and refit; return the lowest χ².
    """
    x, y = _unit_orbit(dt, period, dt_peri, ecc)
    n = x.shape[0]
    zero = np.zeros_like(x)
    design = np.stack(
        [np.stack([zero, x, zero, y], -1), np.stack([x, zero, y, zero], -1)],
        -2,
    )
    wd = np.einsum("nij,njk->nik", whitener, design).reshape(-1, 4)
    r = np.stack([dra, ddec], -1)

    def refit(signed):
        data = np.einsum("nij,nj->ni", whitener, signed).reshape(-1)
        params = np.linalg.lstsq(wd, data)[0]
        resid = data - wd @ params
        return params, resid @ resid

    def signs_for(
        params,
    ):  # params (A, B, F, G): ddec = A x + F y, dra = B x + G y
        model = np.stack(
            [params[1] * x + params[3] * y, params[0] * x + params[2] * y], -1
        )
        return np.where(np.sum(model * r, -1) >= 0, 1.0, -1.0)

    # Pairs: every epoch with its neighbours at a few lags; the best-conditioned
    # _N_SIGN_PAIRS by |det [u_a u_b]| are used.
    lags = onp.unique((onp.array([1, 2, 3, 5, 8, 13]) - 1) % max(n - 1, 1) + 1)
    a = onp.tile(onp.arange(n), len(lags))
    b = (a + onp.repeat(lags, n)) % n
    a, b = np.asarray(a), np.asarray(b)
    det = np.abs(x[a] * y[b] - x[b] * y[a])
    best = np.argsort(-det)[:_N_SIGN_PAIRS]
    ia, ib = a[best], b[best]

    def candidate(i, j, rel):
        u = np.array([[x[i], x[j]], [y[i], y[j]]])
        u = u + 1e-12 * np.eye(2)
        rr = np.stack([r[i], rel * r[j]], -1)
        m = rr @ np.linalg.inv(u)  # rows (dra, ddec): [[B, G], [A, F]]
        params = np.array([m[1, 0], m[0, 0], m[1, 1], m[0, 1]])
        params, c2 = refit(signs_for(params)[:, None] * r)
        params, c2 = refit(signs_for(params)[:, None] * r)
        return params, c2

    fit = jax.vmap(
        lambda i, j: jax.vmap(lambda rel: candidate(i, j, rel))(
            np.array([1.0, -1.0])
        )
    )
    params, chi2 = fit(ia, ib)
    params, chi2 = params.reshape(-1, 4), chi2.reshape(-1)
    k = np.argmin(np.where(np.isfinite(chi2), chi2, np.inf))
    return params[k], chi2[k]


# Default k of ``period_grid`` (see docs/orbit_methods.md#period-grid).
_PHASE_COHERENCE = 9.0


def period_grid(times, p_min, p_max, k=_PHASE_COHERENCE):
    """Trial periods, uniform in 1/P, that stay in phase over the observations.

    See [period grids](../orbit_methods.md#period-grid).

    Parameters
    ----------
    times : array-like
        Epoch times (days); their range is the baseline T.
    p_min, p_max : float
        The shortest and longest trial periods (days).
    k : float, optional
        Neighbouring periods drift apart by at most ``1/k`` of a cycle over
        T (default 9).

    Returns
    -------
    numpy.ndarray
        Periods (days), increasing from ``p_min`` to ``p_max``.

    Raises
    ------
    ValueError
        If ``0 < p_min < p_max`` fails, or the times span no baseline.
    """
    times = onp.asarray(times, float)
    baseline = float(onp.ptp(times)) if times.size else 0.0
    if not 0 < p_min < p_max:
        raise ValueError(
            f"Need 0 < p_min < p_max, not p_min={p_min}, p_max={p_max}."
        )
    if baseline <= 0 or not k > 0:
        raise ValueError(
            "period_grid needs times spanning a positive baseline and k > 0."
        )
    span = 1.0 / p_min - 1.0 / p_max
    n = int(onp.ceil(span * k * baseline)) + 1
    return 1.0 / onp.linspace(1.0 / p_min, 1.0 / p_max, max(n, 2))


def _phase_drift(periods, baseline):  # deliberate: imported by epochs.py
    """The largest phase drift (cycles) between neighbouring trial periods
    over ``baseline`` days: ``T |1/P_i - 1/P_(i+1)|``; 0 for one period."""
    frequencies = onp.sort(1.0 / onp.asarray(periods, float).ravel())
    if frequencies.size < 2:
        return 0.0
    return float(baseline * onp.max(onp.diff(frequencies)))


@functools.partial(jax.jit, static_argnums=0)
def _fit_batch(  # deliberate: jit boundary, fitter static
    fitter, dt, dra, ddec, whitener, period, dt_peri, ecc
):
    """``fitter`` at a batch of grid points (positions traced)."""
    return jax.vmap(fitter, in_axes=(None,) * 4 + (0, 0, 0))(
        dt, dra, ddec, whitener, period, dt_peri, ecc
    )


def starting_orbits(
    positions,
    periods,
    eccs=None,
    n_phase=36,
    n_best=5,
    sign_free=False,
    batch_size=None,
):
    """Good starting orbits for a set of positions, from a grid search.

    Each grid point is an exact weighted least-squares solve; see
    [starting orbits](../orbit_methods.md#starting-orbits).

    Parameters
    ----------
    positions : PositionData
        The measured positions.
    periods : array-like
        Trial periods (days); see [`period_grid`][virgil.orbits.period_grid].
    eccs : array-like, optional
        Trial eccentricities (default 0 to 0.9 in steps of 0.05).
    n_phase, n_best : int, optional
        Trial times of periastron per period; orbits returned.
    sign_free : bool, optional
        Positions known only up to a sign (nearly equal fluxes); the orbit
        returned has the global sign of its best fit, and its reflection
        (``omega`` + 180°) fits equally well.
    batch_size : int, optional
        Grid points solved together; bounds memory, not the result. Capped
        at the grid size (rounded up to a power of two) even when given.

    Returns
    -------
    list of (KeplerOrbit, float)
        The best orbits and χ², best first, each with ``0 <= Omega < 180``.
    """
    eccs = onp.arange(0.0, 0.91, 0.05) if eccs is None else onp.asarray(eccs)
    phases = onp.arange(n_phase) / n_phase
    periods = onp.asarray(periods)
    p, e, f = onp.meshgrid(periods, eccs, phases, indexing="ij")
    grid = onp.stack([p.ravel(), (f * p).ravel(), e.ravel()], axis=1)
    fitter = _thiele_innes_fit_sign_free if sign_free else _thiele_innes_fit
    n_grid = len(grid)
    # Never pad past the grid itself (a power of two, to share compiles):
    # the memory bound is for grids too large to solve at once.
    if batch_size is None:  # about batch_visibilities() position-epochs in all
        n = max(int(onp.size(positions.dt)), 1)
        batch = max(MIN_BATCH_SIZE, batch_visibilities() // n)
    else:
        batch = batch_size_or_default(batch_size, None)
    size = min(batch, 1 << (n_grid - 1).bit_length())
    # Fixed-size batches, the last padded: grids share a compile per number
    # of positions and power-of-two size (up to the batch), and memory is
    # bounded by ``size``.
    padded = onp.concatenate(
        [grid, onp.repeat(grid[-1:], -n_grid % size, axis=0)]
    )
    out = [
        _fit_batch(
            fitter,
            positions.dt,
            positions.dra,
            positions.ddec,
            positions.whitener,
            *(np.asarray(padded[i : i + size, k]) for k in range(3)),
        )
        for i in range(0, len(padded), size)
    ]
    params = onp.concatenate([onp.asarray(x[0]) for x in out])[:n_grid]
    chi2 = onp.concatenate([onp.asarray(x[1]) for x in out])[:n_grid]
    best = onp.argsort(onp.asarray(chi2))[:n_best]
    return [
        (
            ThieleInnesOrbit(
                *grid[k], *onp.asarray(params[k]), t_ref=positions.t_ref
            ).to_kepler(),
            float(chi2[k]),
        )
        for k in best
    ]


@jax.jit  # one dispatch per eager call, as for ``_thiele_innes_angles``
def _state_elements(  # deliberate: jit boundary
    dra, ddec, dz, vra, vdec, vz, mu
):
    """``(period, dt_peri, ecc, inc, omega, Omega, a_mas)`` of a state vector."""
    r = np.stack([dra, ddec, dz])  # (East, North, away)
    v = np.stack([vra, vdec, vz])
    r_norm = np.linalg.norm(r)
    h = np.cross(r, v)
    e_vec = np.cross(v, h) / mu - r / r_norm
    ecc = np.linalg.norm(e_vec)
    a_mas = 1.0 / (2.0 / r_norm - np.dot(v, v) / mu)
    period = 2 * np.pi * np.sqrt(a_mas**3 / mu) * 365.25  # days
    circular = ecc < 1e-12
    p_hat = np.where(
        circular, r / r_norm, e_vec / np.where(circular, 1.0, ecc)
    )
    # Eccentric anomaly at t_ref, then the time of periastron. It goes
    # through the true anomaly measured from ``p_hat`` itself, so for a
    # near-circular orbit, where ``p_hat`` is uncertain by about
    # eps / e, the error moves ω and the anomaly oppositely and cancels
    # in the positions (ω + M is what the sky fixes).
    q_hat = np.cross(h / np.linalg.norm(h), p_hat)
    nu = np.arctan2(np.dot(r, q_hat), np.dot(r, p_hat))
    ecc_anomaly = 2.0 * np.arctan2(
        np.sqrt(1.0 - ecc) * np.sin(nu / 2),
        np.sqrt(1.0 + ecc) * np.cos(nu / 2),
    )
    mean_anomaly = ecc_anomaly - ecc * np.sin(ecc_anomaly)
    dt_peri = -mean_anomaly / (2 * np.pi) * period
    # The angles from the plane's orientation, all by atan2 so that they
    # stay accurate near face-on and edge-on. With P, Q the unit vectors
    # of the Thiele–Innes constants, h ∝ P × Q = (sin i cos Ω,
    # -sin i sin Ω, -cos i) in (East, North, away), which fixes i and
    # the true node Ω (in [0, 360), not just modulo 180°) directly.
    h_xy = np.hypot(h[0], h[1])
    inc = np.arctan2(h_xy, -h[2])
    Omega = np.arctan2(-h[1], h[0])
    sin_i, cos_i = np.sin(inc), np.cos(inc)
    sin_n, cos_n = np.sin(Omega), np.cos(Omega)
    # P = cos ω u + sin ω w, with u the node direction at ω = 0 and w
    # the in-plane direction 90° on from it.
    u_hat = np.stack([sin_n, cos_n, np.zeros_like(sin_n)])
    w_hat = np.stack([cos_n * cos_i, -sin_n * cos_i, sin_i])
    omega = np.arctan2(np.dot(p_hat, w_hat), np.dot(p_hat, u_hat))
    Omega_deg = np.mod(np.rad2deg(Omega), 360.0)
    omega_deg = np.mod(np.rad2deg(omega), 360.0)
    return (
        period,
        dt_peri,
        ecc,
        np.rad2deg(inc),
        np.where(omega_deg >= 360.0, 0.0, omega_deg),
        np.where(Omega_deg >= 360.0, 0.0, Omega_deg),
        a_mas,
    )


class StateVectorOrbit(zdx.Base):
    """An orbit given by the relative position and velocity at ``t_ref``.

    For short arcs, where the Keplerian elements are poorly determined.

    Parameters
    ----------
    dra, ddec : float
        Position of the secondary at ``t_ref`` (mas).
    vra, vdec : float
        Its velocity at ``t_ref`` (mas/yr).
    dz, vz : float
        Line-of-sight position (mas) and velocity (mas/yr), positive away
        from the observer; they and ``mu`` carry the physical priors.
    mu : float
        Gravitational parameter ``4π² a_mas³ / P²`` (mas³/yr², P in years);
        free of distance.
    t_ref : float, optional
        The epoch of the state (MJD, static float64).

    Notes
    -----
    Only bound states (negative energy) are orbits; see
    [`to_kepler`][virgil.orbits.StateVectorOrbit.to_kepler].
    """

    dra: jax.Array
    ddec: jax.Array
    vra: jax.Array
    vdec: jax.Array
    dz: jax.Array
    vz: jax.Array
    mu: jax.Array
    t_ref: float = eqx.field(static=True)

    def __init__(self, dra, ddec, vra, vdec, dz, vz, mu, t_ref=0.0):
        self.dra = np.asarray(dra, dtype=float)
        self.ddec = np.asarray(ddec, dtype=float)
        self.vra = np.asarray(vra, dtype=float)
        self.vdec = np.asarray(vdec, dtype=float)
        self.dz = np.asarray(dz, dtype=float)
        self.vz = np.asarray(vz, dtype=float)
        self.mu = np.asarray(mu, dtype=float)
        self.t_ref = float(t_ref)

    def __check_init__(self):
        _check(
            "StateVectorOrbit",
            (("mu", self.mu, lambda x: x > 0, "positive"),),
        )
        r = np.stack([self.dra, self.ddec, self.dz])
        v = np.stack([self.vra, self.vdec, self.vz])
        radius = concrete(np.linalg.norm(r))
        momentum = concrete(np.linalg.norm(np.cross(r, v)))
        if radius is not None and not (onp.isfinite(radius) and radius > 0):
            raise ValueError(
                "StateVectorOrbit: the position must be finite and away from "
                "the primary."
            )
        if momentum is not None and not (
            onp.isfinite(momentum) and momentum > 0
        ):
            raise ValueError(
                "StateVectorOrbit: the velocity is along the line to the "
                "primary (no angular momentum), a radial fall with no orbital "
                "plane."
            )
        r_len = np.sqrt(self.dra**2 + self.ddec**2 + self.dz**2)
        v2 = self.vra**2 + self.vdec**2 + self.vz**2
        energy = concrete(v2 / 2 - self.mu / r_len)
        if energy is not None and not onp.all(energy < 0):
            raise ValueError(
                "StateVectorOrbit: the state is unbound (non-negative "
                "energy), so it is not an orbit."
            )

    @classmethod
    def from_kepler(cls, orbit):
        """The state of ``orbit`` at its ``t_ref``."""
        dra, ddec, dz = orbit._relative(np.asarray(0.0))
        vra, vdec, vz = _velocity(orbit._relative, np.asarray(0.0))
        per_year = 365.25
        mu = 4 * np.pi**2 * orbit.a_mas**3 / (orbit.period / per_year) ** 2
        return cls(
            dra,
            ddec,
            vra * per_year,
            vdec * per_year,
            dz,
            vz * per_year,
            mu,
            t_ref=orbit.t_ref,
        )

    def to_kepler(self):
        """The [`KeplerOrbit`][virgil.orbits.KeplerOrbit] of this state.

        The periastron and the direction of motion there give the
        Thiele–Innes constants directly, which fixes the angles in virgil's
        conventions; the line-of-sight components then pick the node, which
        positions alone leave ambiguous by 180°. A circular orbit (e = 0)
        has no periastron: the position at ``t_ref`` is used instead.
        """
        elements = _state_elements(
            self.dra, self.ddec, self.dz, self.vra, self.vdec, self.vz, self.mu
        )
        return KeplerOrbit(*elements, t_ref=self.t_ref)

    def relative(self, mjd):
        """``(dra, ddec, dz)`` (mas), as for
        [`KeplerOrbit.relative`][virgil.orbits.KeplerOrbit.relative]."""
        return self.to_kepler().relative(mjd)


# km/s per (mas/day at 1 pc): 1 mas at 1 pc is 1e-3 au.
_KMS_PER_MAS_DAY_PC = _AU_M * 1e-6 / _DAY_S  # 1 mas at 1 pc is 1e-3 au


class RVData(zdx.Base):
    """Radial velocities of one star of the binary.

    Radial velocities are the only data that fix the node absolutely: from
    positions alone (Omega + 180, omega + 180) fits equally well. They need
    a physical scale, the distance, to turn the orbit's angular velocities
    into km/s, and the mass ratio to share the motion between the stars.

    Parameters
    ----------
    mjd : array-like
        Times (MJD).
    rv, d_rv : array-like
        Radial velocities and their errors (km/s, positive receding).
    star : {"primary", "secondary"}, optional
        Which star they are of.
    t_ref : float, optional
        Reference time (MJD, static float64); by default the first epoch.
    instrument : array-like, optional
        One label per epoch naming the spectrograph. Only used when
        [`term`][virgil.orbits.RVData.term] marginalizes the zero points;
        by default there is a single instrument.
    """

    dt: jax.Array
    rv: jax.Array
    d_rv: jax.Array
    inst: jax.Array
    t_ref: float = eqx.field(static=True)
    star: str = eqx.field(static=True)
    instruments: tuple = eqx.field(static=True)

    def __init__(
        self, mjd, rv, d_rv, star="primary", t_ref=None, instrument=None
    ):
        if star not in ("primary", "secondary"):
            raise ValueError(
                f"star must be 'primary' or 'secondary', not {star!r}."
            )
        mjd = onp.atleast_1d(onp.asarray(mjd, dtype=onp.float64))
        rv = onp.atleast_1d(onp.asarray(rv, dtype=float))
        d_rv = onp.broadcast_to(onp.asarray(d_rv, dtype=float), mjd.shape)
        if rv.shape != mjd.shape:
            raise ValueError(
                f"rv has shape {rv.shape} but there are {mjd.size} epochs."
            )
        if not onp.all(onp.isfinite(d_rv) & (d_rv > 0)):
            raise ValueError("d_rv must be positive and finite.")
        self.t_ref = float(mjd.min() if t_ref is None else t_ref)
        self.dt = np.asarray(mjd - self.t_ref)
        self.rv = np.asarray(rv)
        self.d_rv = np.asarray(d_rv)
        self.star = star
        if instrument is None:
            labels, codes = (0,), onp.zeros(mjd.shape, dtype=int)
        else:
            instrument = onp.broadcast_to(onp.asarray(instrument), mjd.shape)
            uniq, codes = onp.unique(instrument, return_inverse=True)
            labels = tuple(u.item() for u in uniq)
        self.instruments = labels
        self.inst = np.asarray(codes.reshape(mjd.shape))

    def model(self, orbit, q, gamma, distance_pc):
        """Predicted radial velocities (km/s).

        Parameters
        ----------
        orbit : KeplerOrbit
            The relative orbit (secondary about primary).
        q : float
            Mass ratio, secondary / primary.
        gamma : float
            Systemic velocity (km/s).
        distance_pc : float
            Distance (pc), which turns mas/day into km/s.
        """
        dt = self.dt + (self.t_ref - orbit.t_ref)
        vz = _velocity(orbit._relative, dt)[2]  # mas/day, positive receding
        v_rel = vz * distance_pc * _KMS_PER_MAS_DAY_PC
        share = -q / (1.0 + q) if self.star == "primary" else 1.0 / (1.0 + q)
        return gamma + share * v_rel

    def errors(self, jitter=0.0):
        """Effective errors ``sqrt(d_rv² + jitter²)`` (km/s)."""
        return np.sqrt(self.d_rv**2 + jitter**2)

    def whitened_residuals(self, orbit, q, gamma, distance_pc, jitter=0.0):
        """``(rv - model) / sqrt(d_rv² + jitter²)`` for every epoch."""
        model = self.model(orbit, q, gamma, distance_pc)
        return (self.rv - model) / self.errors(jitter)

    def loglike(self, orbit, q, gamma, distance_pc, jitter=0.0):
        """Gaussian log-likelihood of the velocities.

        ``jitter`` (km/s) adds an extra scatter in quadrature to every error;
        the normalization then depends on it.
        """
        resid = self.whitened_residuals(orbit, q, gamma, distance_pc, jitter)
        return (
            -0.5 * resid @ resid
            - np.sum(np.log(self.errors(jitter)))
            - resid.size / 2 * np.log(2 * np.pi)
        )

    # -- analytic marginalization of the instrument zero points ----------

    def _marginal(self, prior):
        """The zero points as a [`LinearMarginal`][virgil._linear.LinearMarginal]."""
        mean, sd = prior
        k = len(self.instruments)
        design = (self.inst[:, None] == np.arange(k)[None, :]).astype(
            self.rv.dtype
        )
        return LinearMarginal(design, mean, prior_sd=sd)

    def marginal_whitened_residuals(
        self, orbit, q, gamma, distance_pc, jitter=0.0, prior=None
    ):
        """Whitened residuals of the zero-point-marginalized Gaussian.

        The data are ``d ~ N(m + Aμ, C + AΛAᵀ)`` with ``m`` the Keplerian
        model (``gamma`` included), ``A`` the indicator matrix of the
        instruments and ``w ~ N(μ, Λ)`` the zero points. Returns ``u`` of
        length N with ``uᵀu = rᵀ(C + AΛAᵀ)⁻¹r``, ``r = d - m - Aμ``, by the
        dense Woodbury whitening of
        [`virgil._linear`][virgil._linear] (an exact square root, with no
        eigendecomposition).

        ``prior`` is ``(mean, sd)`` per instrument, as
        [`term`][virgil.orbits.RVData.term] builds it.
        """
        resid = self.rv - self.model(orbit, q, gamma, distance_pc)
        return self._marginal(prior).whiten(resid, self.errors(jitter))[0]

    def marginal_log_norm(self, jitter=0.0, prior=None):
        """``½ log det(C + AΛAᵀ)``.

        This is the normalization that depends on the jitter, in the same
        convention as the plain ``Σ log σ_eff``.
        """
        sigma = self.errors(jitter)
        return self._marginal(prior).whiten(np.zeros_like(sigma), sigma)[1]

    def marginal_loglike(
        self, orbit, q, gamma, distance_pc, jitter=0.0, prior=None
    ):
        """Normalized log density of the zero-point-marginalized Gaussian.

        Equal to a dense ``N(m + Aμ, C + AΛAᵀ)`` log density, computed in
        O(N k²). A flat prior is the limit ``Λ → ∞`` only up to a constant
        (``-½ Σ log Λ``); the finite prior width is part of the model.
        """
        resid = self.rv - self.model(orbit, q, gamma, distance_pc)
        return self._marginal(prior).loglike(resid, self.errors(jitter))

    def zero_point_posterior(
        self, orbit, q, gamma, distance_pc, jitter=0.0, prior=None
    ):
        """Mean and covariance of the zero points ``w`` given the orbit.

        ``w | d, θ ~ N(S⁻¹(Λ⁻¹μ + AᵀC⁻¹(d - m)), S⁻¹)`` with
        ``S = Λ⁻¹ + AᵀC⁻¹A``. Entry ``j`` is the velocity zero point of
        ``self.instruments[j]``; ``m`` includes ``gamma``, so with
        ``gamma = 0`` they are the systemic velocity seen by each
        instrument, and differences between them are the offsets.
        """
        resid = self.rv - self.model(orbit, q, gamma, distance_pc)
        return self._marginal(prior).posterior(resid, self.errors(jitter))

    @renamed()
    def term(self, params, jitter=None, marginalize_offsets=None):
        """A likelihood term for [`fit`][virgil.fitting.fit]'s ``likelihoods``.

        See [radial-velocity terms](../orbit_methods.md#rv-term) for the jitter
        and marginalized zero-point likelihoods.

        Parameters
        ----------
        params : callable
            Maps the fitted values to ``(orbit, q, gamma, distance_pc)``.
        jitter : str, optional
            Path of a fitted value holding an RV jitter ``s`` (km/s), which
            inflates the errors to ``sqrt(d_rv² + s²)``. The term then
            reports ``log_norm``, so ``fit`` defaults to L-BFGS. Give it a
            positive prior, by default ``dist.LogUniform(lo, hi)``.
        marginalize_offsets : (mean, sd), optional
            Analytically marginalize one velocity zero point per instrument,
            with prior ``N(mean, sd²)`` in km/s (scalars, or one per
            instrument in ``RVData.instruments`` order). Return ``gamma = 0``
            from ``params``.
            ``True`` is an error: the prior must be stated.
        """
        if marginalize_offsets not in (None, False):
            if marginalize_offsets is True:
                raise ValueError(
                    "marginalize_offsets=True is not supported: state the "
                    "zero-point prior explicitly as (mean, sd) in km/s "
                    "(scalars or one per instrument), e.g. "
                    "marginalize_offsets=(0.0, 50.0)."
                )
            k = len(self.instruments)
            mean, sd = marginalize_offsets
            mean = onp.broadcast_to(onp.asarray(mean, dtype=float), (k,))
            sd = onp.broadcast_to(onp.asarray(sd, dtype=float), (k,))
            if not onp.all(onp.isfinite(sd) & (sd > 0)):
                raise ValueError(
                    "the zero-point prior needs a positive, finite sd: an "
                    "infinite (flat) prior is not supported."
                )
            if not onp.all(onp.isfinite(mean)):
                raise ValueError("the zero-point prior needs a finite mean.")
            marginalize_offsets = np.asarray(mean), np.asarray(sd)
        else:
            marginalize_offsets = None
        return _Term(self, params, jitter, marginalize_offsets)


class _Term(eqx.Module):
    """``data.whitened_residuals(*build(values))``, for ``fit``."""

    data: object
    build: object = eqx.field(static=True)
    jitter: object = eqx.field(static=True, default=None)
    prior: object = None

    def _args(self, values):
        args = tuple(self.build(values))
        if self.jitter is not None:
            args += (values[self.jitter],)
        return args

    def __call__(self, values):
        if self.prior is not None:
            return self.data.marginal_whitened_residuals(
                *self._args(values), prior=self.prior
            )
        return np.ravel(self.data.whitened_residuals(*self._args(values)))

    def posterior(self, values):
        """``(mean, cov)`` of the marginalized zero points given ``values``."""
        if self.prior is None:
            raise ValueError("the term does not marginalize zero points.")
        return self.data.zero_point_posterior(
            *self._args(values), prior=self.prior
        )

    @property
    def has_log_norm(self):
        """Whether the normalization depends on fitted values."""
        return self.jitter is not None

    def log_norm(self, values):
        """``Σ log σ_eff`` (``½ log det Σ`` when marginalizing), which
        ``fit`` adds to the loss with a fitted jitter."""
        if self.prior is not None:
            jit = values[self.jitter] if self.jitter is not None else 0.0
            return self.data.marginal_log_norm(jit, self.prior)
        return np.sum(np.log(self.data.errors(values[self.jitter])))

    def loglike(self, values):
        """The data's normalized Gaussian log density, for ``numpyro_model``."""
        if self.prior is not None:
            return self.data.marginal_loglike(
                *self._args(values), prior=self.prior
            )
        return self.data.loglike(*self._args(values))


def orientation_priors(
    positions_only=True, *, prefix="", ring_width=0.25, inclination=False
):
    """Angle-vector priors for an orbit's node and periastron.

    With ``positions_only`` the keys are ``"two_Omega"`` (2Ω) and ``"varpi"``
    (ϖ = Ω + ω), merging the twin modes of a positions-only fit; otherwise
    ``"Omega"`` and ``"varpi"``. Build the orbit with
    [`KeplerOrbit.from_varpi`][virgil.orbits.KeplerOrbit.from_varpi]; see
    [orientation priors](../orbit_methods.md#orientation-priors).

    Parameters
    ----------
    positions_only : bool, optional
        Whether only positions constrain the orbit (default ``True``).
    prefix : str, optional
        Prepended to the keys, e.g. ``"orbit."``.
    ring_width : float, optional
        Passed to [`AngleVector`][virgil.angles.AngleVector].
    inclination : bool, optional
        Also return ``IsotropicInclination()`` under ``prefix + "inc"``.

    Returns
    -------
    dict
    """
    node = "two_Omega" if positions_only else "Omega"
    priors = {
        prefix + name: AngleVector(ring_width=ring_width)
        for name in (node, "varpi")
    }
    if inclination:
        priors[prefix + "inc"] = IsotropicInclination()
    return priors


def orientation_from_varpi(varpi, *, Omega=None, two_Omega=None):
    """``(omega, Omega)`` (degrees) from ϖ = Ω + ω and the node.

    Give exactly one of ``Omega`` and ``two_Omega``. From ``two_Omega``,
    Ω is reported in [0°, 180°), as by
    [`starting_orbits`][virgil.orbits.starting_orbits]; ω is in
    [0°, 360°).
    """
    if (Omega is None) == (two_Omega is None):
        raise ValueError("Give exactly one of Omega and two_Omega.")
    if Omega is None:
        Omega = 0.5 * np.mod(two_Omega, 360.0)
    return np.mod(varpi - Omega, 360.0), Omega


def _true_anomaly_at_ref(theta, inc, omega, Omega):
    """f (radians) at the epoch where the position angle is ``theta``
    (degrees), from u = atan2(sin φ / cos i, cos φ), φ = θ - Ω."""
    phi = np.deg2rad(theta - Omega)
    u = np.arctan2(np.sin(phi) / np.cos(np.deg2rad(inc)), np.cos(phi))
    return u - np.deg2rad(omega)


def position_angle_log_jacobian(theta, ecc, inc, omega, Omega):
    """log|∂M/∂θ| at fixed (e, i, ω, Ω), for a prior uniform in t_peri.

    The invariant prior on the epoch is uniform in the time of periastron,
    i.e. in the mean anomaly M at ``t_ref`` (a translation), not in the
    position angle θ there. Sampling θ uniformly (an
    [`AngleVector`][virgil.angles.AngleVector]) and adding this term gives
    back the uniform prior in M:

        log|∂M/∂θ| = 3/2 log(1 - e²) - 2 log(1 + e cos f) + log|cos i|
                     - log(cos²φ cos²i + sin²φ),  φ = θ - Ω,

    from dM/df = (1 - e²)^{3/2}/(1 + e cos f)² and du/dφ = cos i /
    (cos²φ cos²i + sin²φ). Over a full turn of θ it integrates to 2π, so
    the prior stays normalized. It diverges at i = 90° (see
    [`KeplerOrbit.from_position_angle`][virgil.orbits.KeplerOrbit.from_position_angle]).
    All angles in degrees.
    """
    f = _true_anomaly_at_ref(theta, inc, omega, Omega)
    phi = np.deg2rad(theta - Omega)
    cos_i = np.cos(np.deg2rad(inc))
    ecc = np.asarray(ecc, float)
    return (
        1.5 * np.log1p(-(ecc**2))
        - 2.0 * np.log1p(ecc * np.cos(f))
        + np.log(np.abs(cos_i))
        - np.log(np.cos(phi) ** 2 * cos_i**2 + np.sin(phi) ** 2)
    )


class _PositionAnglePrior(eqx.Module):
    """The log-Jacobian of a θ-sampled orbit, as a ``likelihoods=`` term.

    It has no residuals; ``fit`` adds ``log_norm`` (= -log|∂M/∂θ|) to its
    loss and ``numpyro_model`` adds ``loglike`` (= log|∂M/∂θ|).
    """

    orbit_fn: object = eqx.field(static=True)
    has_log_norm = True

    def _log_jacobian(self, values):
        orbit = self.orbit_fn(values)
        dra, ddec, _ = orbit._relative(np.zeros(()))
        theta = position_angle(dra, ddec)
        return position_angle_log_jacobian(
            theta, orbit.ecc, orbit.inc, orbit.omega, orbit.Omega
        )

    def __call__(self, values):
        return np.zeros((0,))

    def log_norm(self, values):
        return -self._log_jacobian(values)

    def loglike(self, values):
        return self._log_jacobian(values)


def position_angle_prior(orbit_fn):
    """The prior term that keeps a θ-sampled orbit uniform in t_peri.

    Pass it in ``likelihoods=`` to [`fit`][virgil.fitting.fit] or
    [`numpyro_model`][virgil.likelihood.numpyro_model], beside the data
    terms, when ``orbit_fn(values)`` builds its orbit with
    [`KeplerOrbit.from_position_angle`][virgil.orbits.KeplerOrbit.from_position_angle].
    It adds [`position_angle_log_jacobian`][virgil.orbits.position_angle_log_jacobian]
    at the orbit's position angle at ``t_ref`` (its θ) to the log
    posterior. It has no least-squares form, so ``fit`` then defaults to
    L-BFGS; its χ² in ``info`` is 0 over 0 points.

    Examples
    --------
    >>> priors = {"theta": AngleVector(), "ecc": dist.Uniform(0.0, 0.9)}
    >>> def orbit_fn(v):
    ...     return KeplerOrbit.from_position_angle(
    ...         400.0, v["theta"], v["ecc"], 60.0, 40.0, 110.0, 20.0,
    ...         t_ref=60500.0,
    ...     )
    >>> terms = [positions.term(orbit_fn), position_angle_prior(orbit_fn)]  # doctest: +SKIP
    """
    return _PositionAnglePrior(orbit_fn)


def total_mass(orbit, distance_pc):
    """Total mass (solar masses) from the orbit at a distance (pc).

    Kepler's third law, M = 4π² a³ / (G P²), with a in au
    (``a_mas · D / 1000``) and P in days, and the IAU 2015 nominal
    G M☉ = 1.3271244e20 m³ s⁻², au = 149597870700 m and day = 86400 s. (The
    shortcut M = a³ / P² with P in Julian years holds only for the Gaussian
    year, 365.256898 d, and is 4e-5 off in M.) Report it as a function of
    distance, or with the distance's uncertainty: positions alone do not
    fix it.
    """
    a_au = orbit.a_mas * 1e-3 * distance_pc
    return _KEPLER_FACTOR * a_au**3 / orbit.period**2


def distance_pc(orbit, total_mass):
    """The distance (pc) at which ``orbit`` has this total mass (M☉): the
    dynamical parallax, the inverse of [`total_mass`][virgil.orbits.total_mass]
    (same constants)."""
    a_au = (total_mass * orbit.period**2 / _KEPLER_FACTOR) ** (1.0 / 3.0)
    return a_au / (orbit.a_mas * 1e-3)


class AxialVonMises(dist.Distribution):
    """A prior on an angle (degrees) known only modulo 180°.

    For a node position angle from a source whose convention is in doubt:
    the density is a von Mises in 2θ, so θ and θ + 180° are equally likely,
    ``exp(kappa cos 2(θ - mean))``, normalized over [0, 360).

    Parameters
    ----------
    mean : float
        Mean angle (degrees); ``mean + 180`` is equivalent.
    kappa : float
        Concentration (of the doubled angle); larger is tighter, with a
        width of about ``28.6 / sqrt(kappa)`` degrees.
    """

    arg_constraints = {
        "mean_deg": constraints.real,
        "kappa": constraints.positive,
    }
    support = constraints.interval(0.0, 360.0)

    def __init__(self, mean, kappa, *, validate_args=None):
        self.mean_deg = np.asarray(mean, dtype=float)
        self.kappa = np.asarray(kappa, dtype=float)
        super().__init__(
            batch_shape=np.broadcast_shapes(np.shape(mean), np.shape(kappa)),
            validate_args=validate_args,
        )

    def log_prob(self, value):
        doubled = 2.0 * np.deg2rad(value - self.mean_deg)
        return self.kappa * (np.cos(doubled) - 1.0) - np.log(
            360.0 * jax.scipy.special.i0e(self.kappa)
        )

    def sample(self, key, sample_shape=()):
        shape = sample_shape + self.batch_shape
        key_vm, key_flip = jax.random.split(key)
        doubled = dist.VonMises(0.0, self.kappa).sample(key_vm, sample_shape)
        flip = 180.0 * jax.random.bernoulli(key_flip, 0.5, shape)
        return np.mod(self.mean_deg + np.rad2deg(doubled) / 2.0 + flip, 360.0)
