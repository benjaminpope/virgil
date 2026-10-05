"""Keplerian orbits of a binary's secondary about its primary.

The conventions are those of ``design/orbit_scene_joint_fitting.md`` §2.1:

* ``dra`` is positive East and ``ddec`` positive North (mas), as everywhere
  in virgil, and ``dz`` is positive **away from** the observer, so that
  (dra, ddec, dz) is right-handed and ``dz`` grows while the secondary
  recedes. The vector runs from the primary (the scene's reference
  component) to the secondary.
* ``inc`` in [0°, 180°): below 90° the position angle increases with time
  (counterclockwise on the sky, North through East).
* ``Omega``: the position angle of the ascending node, the node where the
  secondary recedes. Positions alone fix it only modulo 180°.
* ``omega``: the **secondary's** argument of periastron (the visual-binary
  convention); the spectroscopic ω of the primary is ``omega - 180°``.
* ``dt_peri``: the time of periastron minus the static float64 ``t_ref``
  (days), so that float32 keeps it precise; ``period`` in days; ``a_mas``
  the angular semimajor axis of the relative orbit.

Kepler's equation is solved by jaxoplanet (an optional dependency,
``pip install "virgil-astro[orbits]"``), with its exact derivatives; the
positions follow from the Thiele–Innes constants (§2.4). jaxoplanet's own
conventions differ (its ω is the primary's, and its third axis points
toward the observer) and stay inside :meth:`KeplerOrbit.to_jaxoplanet` and
:meth:`KeplerOrbit.from_jaxoplanet`.
"""

import jax
import jax.numpy as np
import numpy as onp

import equinox as eqx
import numpyro.distributions as dist
import zodiax as zx
from numpyro.distributions import constraints

from ._utils import concrete


__all__ = [
    "AxialVonMises",
    "KeplerOrbit",
    "PositionData",
    "RVData",
    "StateVectorOrbit",
    "ThieleInnesOrbit",
    "distance_pc",
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


def _kepler(mean_anomaly, ecc):
    """``(sin f, cos f)`` of the true anomaly, from jaxoplanet's solver."""
    return _jaxoplanet().core.kepler(mean_anomaly, ecc)


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

    Under ``jit`` the subtraction happens in the traced precision, so pass
    times already relative to ``t_ref`` there (float32 resolves MJD ≈ 60000
    to only about 0.004 d).
    """
    if isinstance(mjd, jax.core.Tracer):
        return mjd - t_ref
    return np.asarray(onp.asarray(mjd, dtype=onp.float64) - t_ref)


def _unit_orbit(dt, period, dt_peri, ecc):
    """Thiele–Innes ``X = cos E - e`` and ``Y = √(1 - e²) sin E``.

    Computed from the true anomaly: ``X = (r/a) cos f``, ``Y = (r/a) sin f``.
    """
    mean_anomaly = 2.0 * np.pi * (dt - dt_peri) / period
    sin_f, cos_f = _kepler(mean_anomaly, ecc * np.ones_like(mean_anomaly))
    radius = (1.0 - ecc**2) / (1.0 + ecc * cos_f)
    return radius * cos_f, radius * sin_f


def _thiele_innes(a_mas, inc, omega, Omega):
    """``(A, B, F, G, C, H)`` for angles in degrees (§2.4)."""
    i, w, n = np.deg2rad(inc), np.deg2rad(omega), np.deg2rad(Omega)
    ci = np.cos(i)
    return (
        a_mas * (np.cos(w) * np.cos(n) - np.sin(w) * np.sin(n) * ci),
        a_mas * (np.cos(w) * np.sin(n) + np.sin(w) * np.cos(n) * ci),
        a_mas * (-np.sin(w) * np.cos(n) - np.cos(w) * np.sin(n) * ci),
        a_mas * (-np.sin(w) * np.sin(n) + np.cos(w) * np.cos(n) * ci),
        a_mas * np.sin(w) * np.sin(i),
        a_mas * np.cos(w) * np.sin(i),
    )


def _velocity(position, dt):
    """Time derivative (per day) of ``position(dt)``, exactly, by a JVP."""
    return jax.jvp(position, (dt,), (np.ones_like(dt),))[1]


class KeplerOrbit(zx.Base):
    """A Keplerian orbit of the secondary relative to the primary.

    Parameters
    ----------
    period : float
        Orbital period (days).
    dt_peri : float
        Time of periastron minus ``t_ref`` (days).
    ecc : float
        Eccentricity, ``0 <= ecc < 1``.
    inc : float
        Inclination (degrees, ``0 <= inc < 180``; below 90 the position
        angle increases with time).
    omega : float
        The secondary's argument of periastron (degrees), measured from the
        ascending node in the direction of motion.
    Omega : float
        Position angle of the ascending node, where the secondary recedes
        (degrees, North through East).
    a_mas : float
        Angular semimajor axis of the relative orbit (mas).
    t_ref : float, optional
        Reference time (MJD, float64, static). Times are measured from it,
        so that float32 keeps them precise.

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

    def thiele_innes(self):
        """The Thiele–Innes constants ``(A, B, F, G, C, H)`` (mas).

        ``ddec = A X + F Y``, ``dra = B X + G Y`` and ``dz = C X + H Y``,
        with ``X = cos E - e`` and ``Y = √(1 - e²) sin E``.
        """
        return _thiele_innes(self.a_mas, self.inc, self.omega, self.Omega)

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
        return tuple(self._relative(_days_since(mjd, self.t_ref)))

    def relative_velocity(self, mjd):
        """``d(dra, ddec, dz)/dt`` (mas per day), exactly."""
        dt = _days_since(mjd, self.t_ref)
        return tuple(_velocity(self._relative, dt))

    def _frame(self, dt):
        dra, ddec, dz = self._relative(dt)
        line_pa = np.rad2deg(np.arctan2(dra, ddec))
        constant = np.zeros_like(line_pa)
        return {
            "line_pa": line_pa,
            "towards_primary": line_pa + 180.0,
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
          secondary; ``towards_primary`` is ``line_pa + 180``.
        * ``line_tilt``: the line of centres' elevation out of the sky,
          ``arcsin(dz / |r|)``, positive when the secondary is farther.
        * ``node_pa``: ``Omega``, the line of nodes of the orbital plane.
        * ``inc``: the orbit's inclination (0–180); ``apparent_inc``:
          ``arccos|cos inc|`` (0–90), the projected tilt, for components
          whose ``inc`` is an apparent inclination.
        """
        return self._frame(_days_since(mjd, self.t_ref))

    def separation_pa(self, mjd):
        """Separation (mas) and position angle (degrees, North through East,
        in [0, 360)) of the secondary from the primary."""
        dra, ddec, _ = self.relative(mjd)
        pa = np.mod(np.rad2deg(np.arctan2(dra, ddec)), 360.0)
        return np.hypot(dra, ddec), pa

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


class ThieleInnesOrbit(zx.Base):
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
        total = np.arctan2(b - f, a + g)  # omega + Omega
        difference = np.arctan2(-(b + f), a - g)  # omega - Omega
        omega, Omega = (total + difference) / 2, (total - difference) / 2
        # Omega into [0, π): shifting both angles by π keeps the sky orbit.
        shift = np.floor(Omega / np.pi) * np.pi
        omega, Omega = omega - shift, Omega - shift
        half = (a**2 + b**2 + f**2 + g**2) / 2
        if concrete(half) is not None and not onp.all(concrete(half) > 0):
            raise ValueError(
                "The Thiele–Innes constants are all zero: the positions "
                "carry no orbit (all at the primary)."
            )
        cos_term = a * g - b * f  # a² cos i
        a_sq = half + np.sqrt(np.maximum(half**2 - cos_term**2, 0.0))
        return KeplerOrbit(
            period=self.period,
            dt_peri=self.dt_peri,
            ecc=self.ecc,
            inc=np.rad2deg(np.arccos(np.clip(cos_term / a_sq, -1.0, 1.0))),
            omega=np.mod(np.rad2deg(omega), 360.0),
            Omega=np.rad2deg(Omega),
            a_mas=np.sqrt(a_sq),
            t_ref=self.t_ref,
        )


class PositionData(zx.Base):
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

    def whitened_residuals(self, orbit):
        """``L⁻¹ (data - orbit)`` for every epoch, flattened (2n,)."""
        dt = self.dt + (self.t_ref - orbit.t_ref)
        dra, ddec, _ = orbit._relative(dt)
        resid = np.stack([self.dra - dra, self.ddec - ddec], -1)
        return np.einsum("nij,nj->ni", self.whitener, resid).reshape(-1)

    def loglike(self, orbit):
        """Gaussian log-likelihood of the positions under ``orbit``."""
        resid = self.whitened_residuals(orbit)
        log_det = np.sum(np.log(np.abs(np.diagonal(self.whitener, 0, 1, 2))))
        return (
            -0.5 * resid @ resid + log_det - resid.size / 2 * np.log(2 * np.pi)
        )

    def term(self, orbit):
        """A likelihood term for [`fit`][virgil.fitting.fit]'s ``likelihoods``.

        Parameters
        ----------
        orbit : callable
            Maps the fitted values (a dict, by path or keyword) to a
            [`KeplerOrbit`][virgil.orbits.KeplerOrbit].
        """
        return _Term(self, lambda values: (orbit(values),))


@jax.jit
def _thiele_innes_fit(dt, dra, ddec, whitener, period, dt_peri, ecc):
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


def starting_orbits(positions, periods, eccs=None, n_phase=36, n_best=5):
    """Good starting orbits for a set of positions, from a grid search.

    At fixed period, eccentricity and time of periastron the positions are
    linear in the Thiele–Innes constants, so each grid point is an exact
    weighted least-squares solve. This is the classical way to start an
    orbit fit: it needs no random restarts and handles the several minima
    of a short arc. Refine the best orbits with a fit to the visibilities
    or to the positions.

    Parameters
    ----------
    positions : PositionData
        The measured positions.
    periods : array-like
        Trial periods (days), e.g. log-spaced over the plausible range.
    eccs : array-like, optional
        Trial eccentricities; by default 0 to 0.9 in steps of 0.05.
    n_phase : int, optional
        Number of trial times of periastron, spread over each period.
    n_best : int, optional
        Number of orbits to return.

    Returns
    -------
    list of (KeplerOrbit, float)
        The best orbits and their χ², best first. Each has
        ``0 <= Omega < 180``; (Omega + 180, omega + 180) fits equally well.
    """
    eccs = onp.arange(0.0, 0.91, 0.05) if eccs is None else onp.asarray(eccs)
    phases = onp.arange(n_phase) / n_phase
    grid = onp.array(
        [
            (p, f * p, e)
            for p in onp.asarray(periods)
            for e in eccs
            for f in phases
        ]
    )
    fit_all = jax.vmap(_thiele_innes_fit, in_axes=(None,) * 4 + (0, 0, 0))
    params, chi2 = fit_all(
        positions.dt,
        positions.dra,
        positions.ddec,
        positions.whitener,
        *(np.asarray(grid[:, k]) for k in range(3)),
    )
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


class StateVectorOrbit(zx.Base):
    """An orbit given by the relative position and velocity at ``t_ref``.

    For short arcs, where the measured quantities (the position and its
    rate of change) are well determined but the Keplerian elements are not:
    sampling these instead of the elements avoids long curved degeneracies.
    The line-of-sight position and velocity, and the gravitational
    parameter, carry the physical priors.

    Parameters
    ----------
    dra, ddec : float
        Position of the secondary from the primary at ``t_ref`` (mas).
    vra, vdec : float
        Its velocity at ``t_ref`` (mas/yr).
    dz, vz : float
        Line-of-sight position (mas) and velocity (mas/yr), positive away
        from the observer.
    mu : float
        Gravitational parameter in angular units, ``4π² a_mas³ / P²`` with
        P in years (mas³/yr²). It is free of distance.
    t_ref : float, optional
        The epoch of the state (MJD, static float64).

    Notes
    -----
    Use [`to_kepler`][virgil.orbits.StateVectorOrbit.to_kepler] for the
    elements. Only bound states (negative energy) are orbits.
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

    def _energy(self):
        r = np.sqrt(self.dra**2 + self.ddec**2 + self.dz**2)
        v2 = self.vra**2 + self.vdec**2 + self.vz**2
        return v2 / 2 - self.mu / r

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
        energy = concrete(self._energy())
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
        r = np.stack([self.dra, self.ddec, self.dz])  # (East, North, away)
        v = np.stack([self.vra, self.vdec, self.vz])
        r_norm = np.linalg.norm(r)
        h = np.cross(r, v)
        e_vec = np.cross(v, h) / self.mu - r / r_norm
        ecc = np.linalg.norm(e_vec)
        a_mas = 1.0 / (2.0 / r_norm - np.dot(v, v) / self.mu)
        period = 2 * np.pi * np.sqrt(a_mas**3 / self.mu) * 365.25  # days
        circular = ecc < 1e-12
        p_hat = np.where(
            circular, r / r_norm, e_vec / np.where(circular, 1.0, ecc)
        )
        q_hat = np.cross(h / np.linalg.norm(h), p_hat)
        # Eccentric anomaly at t_ref, then the time of periastron.
        cos_e = np.where(
            circular, 1.0, (1 - r_norm / a_mas) / np.where(circular, 1.0, ecc)
        )
        sin_e = np.where(
            circular,
            0.0,
            np.dot(r, v)
            / (np.where(circular, 1.0, ecc) * np.sqrt(self.mu * a_mas)),
        )
        ecc_anomaly = np.arctan2(sin_e, cos_e)
        mean_anomaly = ecc_anomaly - ecc * np.sin(ecc_anomaly)
        dt_peri = -mean_anomaly / (2 * np.pi) * period
        # Thiele–Innes: (ddec, dra, dz) = a (P X + Q Y) per component.
        a_ti, b_ti, c_ti = a_mas * p_hat[1], a_mas * p_hat[0], a_mas * p_hat[2]
        f_ti, g_ti, h_ti = a_mas * q_hat[1], a_mas * q_hat[0], a_mas * q_hat[2]
        sky = ThieleInnesOrbit(
            period, dt_peri, ecc, a_ti, b_ti, f_ti, g_ti, t_ref=self.t_ref
        ).to_kepler()
        # to_kepler picks Omega in [0, 180); the line of sight decides.
        _, _, c_sky, h_sky = sky.thiele_innes()[2:]
        flip = (c_sky * c_ti + h_sky * h_ti) < 0
        shift = np.where(flip, 180.0, 0.0)
        return KeplerOrbit(
            period,
            dt_peri,
            ecc,
            sky.inc,
            np.mod(sky.omega + shift, 360.0),
            np.mod(sky.Omega + shift, 360.0),
            a_mas,
            t_ref=self.t_ref,
        )

    def relative(self, mjd):
        """``(dra, ddec, dz)`` (mas), as for
        [`KeplerOrbit.relative`][virgil.orbits.KeplerOrbit.relative]."""
        return self.to_kepler().relative(mjd)


# km/s per (mas/day at 1 pc): 1 mas at 1 pc is 1e-3 au.
_KMS_PER_MAS_DAY_PC = 1.495978707e8 * 1e-3 / 86400.0


class RVData(zx.Base):
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
    """

    dt: jax.Array
    rv: jax.Array
    d_rv: jax.Array
    t_ref: float = eqx.field(static=True)
    star: str = eqx.field(static=True)

    def __init__(self, mjd, rv, d_rv, star="primary", t_ref=None):
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

    def whitened_residuals(self, orbit, q, gamma, distance_pc):
        """``(rv - model) / d_rv`` for every epoch."""
        return (self.rv - self.model(orbit, q, gamma, distance_pc)) / self.d_rv

    def loglike(self, orbit, q, gamma, distance_pc):
        """Gaussian log-likelihood of the velocities."""
        resid = self.whitened_residuals(orbit, q, gamma, distance_pc)
        return (
            -0.5 * resid @ resid
            - np.sum(np.log(self.d_rv))
            - resid.size / 2 * np.log(2 * np.pi)
        )

    def term(self, params):
        """A likelihood term for [`fit`][virgil.fitting.fit]'s ``likelihoods``.

        Parameters
        ----------
        params : callable
            Maps the fitted values to ``(orbit, q, gamma, distance_pc)``.
        """
        return _Term(self, params)


class _Term(eqx.Module):
    """``data.whitened_residuals(*build(values))``, for ``fit``."""

    data: object
    build: object = eqx.field(static=True)

    def __call__(self, values):
        return np.ravel(self.data.whitened_residuals(*self.build(values)))


def total_mass(orbit, distance_pc):
    """Total mass (solar masses) from the orbit at a distance (pc).

    Kepler's third law, M = a³ / P², with a in au (``a_mas · D / 1000``)
    and P in years. Report it as a function of distance, or with the
    distance's uncertainty: positions alone do not fix it.
    """
    a_au = orbit.a_mas * 1e-3 * distance_pc
    return a_au**3 / (orbit.period / 365.25) ** 2


def distance_pc(orbit, total_mass):
    """The distance (pc) at which ``orbit`` has this total mass (M☉): the
    dynamical parallax, the inverse of [`total_mass`][virgil.orbits.total_mass]."""
    a_au = (total_mass * (orbit.period / 365.25) ** 2) ** (1.0 / 3.0)
    return a_au / (orbit.a_mas * 1e-3)


class AxialVonMises(dist.Distribution):
    """A prior on an angle (degrees) known only modulo 180°.

    For a node position angle from a source whose convention is in doubt:
    the density is a von Mises in 2θ, so θ and θ + 180° are equally likely,
    ``exp(kappa cos 2(θ - mean))``, normalised over [0, 360).

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
