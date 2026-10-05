"""Priors that are invariant measures but not plain numpyro priors.

virgil's default priors are the Jeffreys priors under the groups that act on
each parameter (see the design notes): uniform for locations, log-uniform
(``numpyro.distributions.LogUniform``) for scales, and the Haar measure for
orientations. The Haar measure on rotations is uniform in the direction of
an axis on the sphere, which is *not* uniform in the angles virgil uses:

* an orbit's inclination ``inc`` (degrees, 0 to 180) has density ∝ sin i, so
  that cos i is uniform; use
  [`IsotropicInclination`][virgil.priors.IsotropicInclination];
* a position on a sphere, such as a spot's latitude (radians), has density
  ∝ cos(lat), so that sin(lat) is uniform; use
  [`IsotropicLatitude`][virgil.priors.IsotropicLatitude].

The longitude of the node, argument of periastron and spot longitude are
uniform in their angles, so ``numpyro.distributions.Uniform`` is right for
them.

Both classes are numpyro distributions, so they work as entries of the
``priors`` of [`fit`][virgil.fitting.fit] and
[`numpyro_model`][virgil.likelihood.numpyro_model]: their support is an
interval, which both map to unconstrained coordinates with numpyro's
``biject_to``. These priors have no least-squares form, so ``fit`` uses
L-BFGS (its automatic choice) or Adam, not ``method="lm"``.
"""

import jax
import jax.numpy as jnp
import numpy as onp
from numpyro.distributions import Distribution, constraints
from numpyro.distributions.util import promote_shapes

__all__ = ["IsotropicInclination", "IsotropicLatitude"]


def _check_range(name, low, high, lo_limit, hi_limit):
    """Validate bounds that are concrete numbers (tracers pass unchecked)."""
    try:
        low_v, high_v = onp.asarray(low, float), onp.asarray(high, float)
    except (TypeError, jax.errors.TracerArrayConversionError):
        return
    if onp.any(low_v >= high_v):
        raise ValueError(f"{name}: need low < high, got {low} and {high}.")
    if onp.any(low_v < lo_limit) or onp.any(high_v > hi_limit):
        raise ValueError(
            f"{name}: the range must lie within [{lo_limit}, {hi_limit}], "
            f"got [{low}, {high}]."
        )


class _InverseCDFPrior(Distribution):
    """Shared machinery: an interval prior sampled by its inverse CDF."""

    arg_constraints = {"low": constraints.real, "high": constraints.real}
    reparametrized_params = ["low", "high"]

    @constraints.dependent_property(is_discrete=False, event_dim=0)
    def support(self):
        return constraints.interval(self.low, self.high)

    def _init(self, low, high, validate_args):
        low, high = promote_shapes(low, high)
        self.low, self.high = low, high
        batch_shape = jax.lax.broadcast_shapes(jnp.shape(low), jnp.shape(high))
        super().__init__(batch_shape=batch_shape, validate_args=validate_args)

    def sample(self, key, sample_shape=()):
        shape = sample_shape + self.batch_shape
        return self.icdf(jax.random.uniform(key, shape))


class IsotropicInclination(_InverseCDFPrior):
    """The inclination of an isotropically oriented axis: density ∝ sin i.

    The Haar measure on rotations gives an orbit normal (or a spin axis)
    that is uniform on the sphere, so its inclination ``i`` has density
    ``sin(i) / (cos(low) - cos(high))`` on ``[low, high]``, that is, cos i
    is uniform. This is the default prior on ``inc``; ``Uniform(0, 180)``
    is not isotropic (it favours face-on orbits).

    Parameters
    ----------
    low, high : float
        Bounds in **degrees**, the unit of ``inc`` in virgil (0 to 180, see
        the conventions page). Defaults: the full range.

    Notes
    -----
    Use ``IsotropicInclination()`` (0 to 180) whenever the data distinguish
    ``i`` from ``180 - i``: astrometry or interferometric closure phases
    give the sense of motion on the sky, so the orbit is prograde
    (``i < 90``) or retrograde (``i > 90``). Use
    ``IsotropicInclination(0, 90)`` when only ``|cos i|`` is identifiable,
    for example with radial velocities alone, or with the sky-projected
    image of a rotating star: the data cannot tell ``i`` from ``180 - i``,
    and the density is the same shape on half the range.

    Examples
    --------
    >>> import jax
    >>> from virgil.priors import IsotropicInclination
    >>> prior = IsotropicInclination(0.0, 90.0)
    >>> samples = prior.sample(jax.random.key(0), (1000,))
    >>> bool(((samples >= 0.0) & (samples <= 90.0)).all())
    True
    """

    def __init__(self, low=0.0, high=180.0, *, validate_args=None):
        _check_range("IsotropicInclination", low, high, 0.0, 180.0)
        self._init(low, high, validate_args)

    @property
    def _cos_bounds(self):
        return (
            jnp.cos(jnp.deg2rad(self.low)),
            jnp.cos(jnp.deg2rad(self.high)),
        )

    def icdf(self, q):
        """Quantile function: cos i is uniform, so invert through arccos."""
        c_lo, c_hi = self._cos_bounds
        return jnp.rad2deg(jnp.arccos(c_lo - q * (c_lo - c_hi)))

    def cdf(self, value):
        c_lo, c_hi = self._cos_bounds
        c = jnp.cos(jnp.deg2rad(value))
        return jnp.clip((c_lo - c) / (c_lo - c_hi), 0.0, 1.0)

    def log_prob(self, value):
        c_lo, c_hi = self._cos_bounds
        # The density is per degree, so d(cos i)/di carries pi/180.
        log_norm = jnp.log((c_lo - c_hi) * 180.0 / jnp.pi)
        inside = (value >= self.low) & (value <= self.high)
        log_sin = jnp.log(jnp.sin(jnp.deg2rad(value)))
        return jnp.where(inside, log_sin - log_norm, -jnp.inf)

    @property
    def mean(self):
        lo, hi = jnp.deg2rad(self.low), jnp.deg2rad(self.high)
        c_lo, c_hi = self._cos_bounds
        num = (jnp.sin(hi) - hi * c_hi) - (jnp.sin(lo) - lo * c_lo)
        return jnp.rad2deg(num / (c_lo - c_hi))


class IsotropicLatitude(_InverseCDFPrior):
    """The latitude of an isotropically placed point: density ∝ cos(lat).

    A point uniform on the sphere has sin(lat) uniform, so its latitude has
    density ``cos(lat) / (sin(high) - sin(low))``. Use it for the position
    of a star spot (harmonix's ``lat`` argument), together with
    ``Uniform(-pi, pi)`` for the longitude.

    Parameters
    ----------
    low, high : float
        Bounds in **radians**, the unit harmonix's ``ylm_spot`` uses.
        Defaults: ``-pi/2`` to ``pi/2``, the whole sphere. Convert degrees
        with ``numpy.deg2rad``; the values sampled are then in radians too.

    Examples
    --------
    >>> import jax
    >>> from virgil.priors import IsotropicLatitude
    >>> samples = IsotropicLatitude().sample(jax.random.key(0), (1000,))
    >>> bool((abs(samples) <= 3.1416 / 2).all())
    True
    """

    def __init__(
        self, low=-jnp.pi / 2, high=jnp.pi / 2, *, validate_args=None
    ):
        half_pi = onp.pi / 2 + 1e-12
        _check_range("IsotropicLatitude", low, high, -half_pi, half_pi)
        self._init(low, high, validate_args)

    def icdf(self, q):
        s_lo, s_hi = jnp.sin(self.low), jnp.sin(self.high)
        return jnp.arcsin(s_lo + q * (s_hi - s_lo))

    def cdf(self, value):
        s_lo, s_hi = jnp.sin(self.low), jnp.sin(self.high)
        return jnp.clip((jnp.sin(value) - s_lo) / (s_hi - s_lo), 0.0, 1.0)

    def log_prob(self, value):
        s_lo, s_hi = jnp.sin(self.low), jnp.sin(self.high)
        inside = (value >= self.low) & (value <= self.high)
        return jnp.where(
            inside, jnp.log(jnp.cos(value)) - jnp.log(s_hi - s_lo), -jnp.inf
        )

    @property
    def mean(self):
        lo, hi = self.low, self.high
        num = (jnp.cos(hi) + hi * jnp.sin(hi)) - (
            jnp.cos(lo) + lo * jnp.sin(lo)
        )
        return num / (jnp.sin(hi) - jnp.sin(lo))
