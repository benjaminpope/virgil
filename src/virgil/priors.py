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
``biject_to``. ``fit`` optimises them in their flat coordinate (cos i or
sin(lat), through ``flat_coordinate()``), where the prior is constant, so
Levenberg–Marquardt applies and is ``fit``'s automatic choice.
"""

import jax
import jax.numpy as jnp
import numpy as onp
from numpyro.distributions import Distribution, constraints
from numpyro.distributions.util import promote_shapes

__all__ = ["IsotropicInclination", "IsotropicLatitude"]


def _check_range(name, low, high, limit, unit):
    """Validate concrete bounds (tracers pass unchecked).

    ``limit`` is the pole angle (180 or pi/2) in the bounds' unit. It is
    rounded to each bound's own dtype, so float32 full-range bounds, whose
    pole rounds past the true value, are accepted.
    """
    try:
        low_v, high_v = onp.asarray(low, float), onp.asarray(high, float)
        limits = []
        for bound in (low, high):
            dtype = jnp.asarray(bound).dtype
            limits.append(float(onp.asarray(jnp.asarray(limit, dtype))))
    except (TypeError, jax.errors.TracerArrayConversionError):
        return
    lim_low, lim_high = limits
    lo_limit = 0.0 if unit == "inc" else -lim_low
    if onp.any(low_v >= high_v):
        raise ValueError(f"{name}: need low < high, got {low} and {high}.")
    if onp.any(low_v < lo_limit) or onp.any(high_v > lim_high):
        raise ValueError(
            f"{name}: the range must lie within [{lo_limit}, {lim_high}], "
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

    def flat_coordinate(self):
        """The coordinate in which the prior is uniform, for ``fit``.

        Returns ``(cdf, icdf, 0, 1)``: the CDF is uniform on [0, 1] and
        affine in cos i (inclination) or sin(lat) (latitude), so it is that
        flat coordinate, rescaled. Using the CDF rather than cos i itself
        keeps the cancellation-resistant formulas above, which matter for
        narrow ranges and ranges at a pole. ``fit`` optimises the angle in
        this coordinate, where the prior adds nothing to the loss (see
        "Priors and the MAP" in the conventions).
        """
        return self.cdf, self.icdf, 0.0, 1.0


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
        _check_range("IsotropicInclination", low, high, 180.0, "inc")
        self._init(low, high, validate_args)

    # All formulas below avoid cos a - cos b and 1 - cos x, which cancel
    # for narrow ranges or ranges at a pole (every float32 range at 0 or 180
    # degrees of a few thousandths of a degree). With a, b the bounds in
    # radians, cos a - cos b = 2 S, S = sin((a+b)/2) sin((b-a)/2), and
    # 1 - cos x = 2 sin^2(x/2), 1 + cos x = 2 cos^2(x/2).

    @property
    def _radians(self):
        return jnp.deg2rad(self.low), jnp.deg2rad(self.high)

    @property
    def _half_norm(self):
        a, b = self._radians
        return jnp.sin(0.5 * (a + b)) * jnp.sin(0.5 * (b - a))

    def icdf(self, q):
        """Quantile function: cos i is uniform, inverted near the nearer pole."""
        a, b = self._radians
        qs = q * self._half_norm
        south = jnp.sin(0.5 * a) ** 2 + qs  # sin^2(i/2)
        north = jnp.cos(0.5 * a) ** 2 - qs  # cos^2(i/2)
        near_zero = jnp.clip(south, 0.0, 1.0) <= 0.5
        half = jnp.where(
            near_zero,
            jnp.arcsin(jnp.sqrt(jnp.clip(south, 0.0, 1.0))),
            jnp.arccos(jnp.sqrt(jnp.clip(north, 0.0, 1.0))),
        )
        return jnp.clip(jnp.rad2deg(2.0 * half), self.low, self.high)

    def cdf(self, value):
        a, b = self._radians
        x = jnp.deg2rad(jnp.clip(value, self.low, self.high))
        num = jnp.sin(0.5 * (a + x)) * jnp.sin(0.5 * (x - a))
        return jnp.clip(num / self._half_norm, 0.0, 1.0)

    def log_prob(self, value):
        # The density is per degree: d(cos i)/di carries pi/180. sin i is
        # evaluated from the distance to the nearer pole, so that it is
        # exactly 0 at 0 and 180 degrees even if the angle rounds past pi.
        distance = jnp.clip(jnp.minimum(value, 180.0 - value), 0.0, None)
        log_sin = jnp.log(jnp.sin(jnp.deg2rad(distance)))
        log_norm = jnp.log(2.0 * self._half_norm * 180.0 / jnp.pi)
        inside = (value >= self.low) & (value <= self.high)
        return jnp.where(inside, log_sin - log_norm, -jnp.inf)

    @property
    def mean(self):
        a, b = self._radians
        c_a, c_b = jnp.cos(a), jnp.cos(b)
        num = (jnp.sin(b) - b * c_b) - (jnp.sin(a) - a * c_a)
        mean = jnp.rad2deg(num / (2.0 * self._half_norm))
        return jnp.clip(mean, self.low, self.high)


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
        _check_range("IsotropicLatitude", low, high, onp.pi / 2, "lat")
        self._init(low, high, validate_args)

    # Cancellation-resistant, as for the inclination: sin b - sin a =
    # 2 cos((a+b)/2) sin((b-a)/2) =: 2 S, and near a pole 1 -/+ sin x =
    # 2 sin^2(delta/2) with delta the colatitude from that pole.

    @property
    def _half_norm(self):
        return jnp.cos(0.5 * (self.low + self.high)) * jnp.sin(
            0.5 * (self.high - self.low)
        )

    def icdf(self, q):
        qs = q * self._half_norm
        quarter = 0.25 * jnp.pi
        # North pole: sin^2((pi/2 - x)/2) = sin^2(pi/4 - a/2) - qs.
        north = jnp.sin(quarter - 0.5 * self.low) ** 2 - qs
        # South pole: sin^2((pi/2 + x)/2) = sin^2(pi/4 + a/2) + qs.
        south = jnp.sin(quarter + 0.5 * self.low) ** 2 + qs
        use_north = jnp.sin(self.low) + 2.0 * qs >= 0.0
        half_pi = 0.5 * jnp.pi
        x = jnp.where(
            use_north,
            half_pi - 2.0 * jnp.arcsin(jnp.sqrt(jnp.clip(north, 0.0, 1.0))),
            2.0 * jnp.arcsin(jnp.sqrt(jnp.clip(south, 0.0, 1.0))) - half_pi,
        )
        return jnp.clip(x, self.low, self.high)

    def cdf(self, value):
        a = self.low
        x = jnp.clip(value, self.low, self.high)
        num = jnp.sin(0.5 * (x - a)) * jnp.cos(0.5 * (x + a))
        return jnp.clip(num / self._half_norm, 0.0, 1.0)

    def log_prob(self, value):
        # cos(lat) from the distance to the nearer pole, in the dtype of
        # value, so it is exactly 0 at the (rounded) endpoints, not negative.
        half_pi = jnp.asarray(jnp.pi / 2, jnp.result_type(value, float))
        distance = jnp.clip(half_pi - jnp.abs(value), 0.0, None)
        log_cos = jnp.log(jnp.sin(distance))
        log_norm = jnp.log(2.0 * self._half_norm)
        inside = (value >= self.low) & (value <= self.high)
        return jnp.where(inside, log_cos - log_norm, -jnp.inf)

    @property
    def mean(self):
        lo, hi = self.low, self.high
        num = (jnp.cos(hi) + hi * jnp.sin(hi)) - (
            jnp.cos(lo) + lo * jnp.sin(lo)
        )
        mean = num / (jnp.sin(hi) - jnp.sin(lo))
        return jnp.clip(mean, lo, hi)
