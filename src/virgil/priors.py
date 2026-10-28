"""Priors that are invariant measures but not plain numpyro priors.

The Haar measure on rotations is uniform in the direction of an axis on the
sphere, which is not uniform in the angles virgil uses:
[`IsotropicInclination`][virgil.priors.IsotropicInclination] (density
∝ sin i, so cos i is uniform) is the prior on an orbit's inclination in
degrees, and [`IsotropicLatitude`][virgil.priors.IsotropicLatitude] (density
∝ cos lat) that of a position on a sphere, in radians. Longitudes and nodes
are uniform in their angles, so ``Uniform`` is right for them.
[`hierarchical_scales`][virgil.priors.hierarchical_scales] gives a population
of scale factors. All are numpyro distributions that work in the ``priors``
of [`fit`][virgil.fitting.fit] and
[`numpyro_model`][virgil.likelihood.numpyro_model]; the isotropic priors
optimize in their flat coordinate (see
[which priors](../conventions.md#which-priors)).
"""

import dataclasses

import jax
import jax.numpy as np
import numpy as onp
import numpyro.distributions as dist
from numpyro.distributions import Distribution, constraints
from numpyro.distributions.util import promote_shapes

__all__ = [
    "IsotropicInclination",
    "IsotropicLatitude",
    "PopulationScale",
    "hierarchical_scales",
]


def _check_range(name, low, high, limit, unit):
    """Validate concrete bounds (tracers pass unchecked).

    ``limit`` is the pole angle in the bounds' unit, rounded to each bound's dtype.
    """
    try:
        low_v, high_v = onp.asarray(low, float), onp.asarray(high, float)
        limits = []
        for bound in (low, high):
            dtype = np.asarray(bound).dtype
            limits.append(float(onp.asarray(np.asarray(limit, dtype))))
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


def _clip_through(x, low, high):
    """``x`` clipped to ``[low, high]`` with the derivative of ``x``."""
    return x + jax.lax.stop_gradient(np.clip(x, low, high) - x)


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
        batch_shape = jax.lax.broadcast_shapes(np.shape(low), np.shape(high))
        super().__init__(batch_shape=batch_shape, validate_args=validate_args)

    def sample(self, key, sample_shape=()):
        shape = sample_shape + self.batch_shape
        return self.icdf(jax.random.uniform(key, shape))

    def flat_coordinate(self):
        """The coordinate in which the prior is uniform, for ``fit``.

        Returns ``(cdf, icdf, 0, 1)``: the CDF is affine in cos i or sin(lat). See
        [priors and the MAP](../conventions.md#priors-and-the-map).
        """
        return self.cdf, self.icdf, 0.0, 1.0


class IsotropicInclination(_InverseCDFPrior):
    """The inclination of an isotropically oriented axis: density ∝ sin i.

    The density is ``sin(i) / (cos(low) - cos(high))`` on ``[low, high]``, so
    cos i is uniform; this is the default prior on ``inc``. ``Uniform(0, 180)``
    is not isotropic (it favours face-on orbits).

    Parameters
    ----------
    low, high : float
        Bounds in **degrees**, the unit of ``inc`` in virgil (0 to 180).
        Defaults: the full range.

    Notes
    -----
    Use the full range when the data distinguish ``i`` from ``180 - i``
    (astrometry, closure phases). Use ``IsotropicInclination(0, 90)`` when only
    ``|cos i|`` is identifiable, as with radial velocities alone.

    Examples
    --------
    >>> import jax
    >>> from virgil.priors import IsotropicInclination
    >>> prior = IsotropicInclination(0.0, 90.0)
    >>> samples = prior.sample(jax.random.key(0), (1000,))
    >>> bool(((samples >= 0.0) & (samples <= 90.0)).all())
    True
    """

    _poles = (0.0, 180.0)

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
        return np.deg2rad(self.low), np.deg2rad(self.high)

    @property
    def _half_norm(self):
        a, b = self._radians
        return np.sin(0.5 * (a + b)) * np.sin(0.5 * (b - a))

    def icdf(self, q):
        """Quantile function: cos i is uniform, inverted near the nearer pole."""
        a, b = self._radians
        qs = q * self._half_norm
        south = np.sin(0.5 * a) ** 2 + qs  # sin^2(i/2)
        north = np.cos(0.5 * a) ** 2 - qs  # cos^2(i/2)
        near_zero = np.clip(south, 0.0, 1.0) <= 0.5
        # The branch not taken gets a safe argument (0.25), so that its
        # infinite derivative at a pole does not make the gradient NaN
        # (samplers differentiate this map; see virgil._flat).
        south = np.where(near_zero, south, 0.25)
        north = np.where(near_zero, 0.25, north)
        half = np.where(
            near_zero,
            np.arcsin(np.sqrt(np.clip(south, 0.0, 1.0))),
            np.arccos(np.sqrt(np.clip(north, 0.0, 1.0))),
        )
        return _clip_through(np.rad2deg(2.0 * half), self.low, self.high)

    def cdf(self, value):
        a, b = self._radians
        x = np.deg2rad(np.clip(value, self.low, self.high))
        num = np.sin(0.5 * (a + x)) * np.sin(0.5 * (x - a))
        return np.clip(num / self._half_norm, 0.0, 1.0)

    def log_prob(self, value):
        # The density is per degree: d(cos i)/di carries pi/180. sin i is
        # evaluated from the distance to the nearer pole, so that it is
        # exactly 0 at 0 and 180 degrees even if the angle rounds past pi.
        distance = np.clip(np.minimum(value, 180.0 - value), 0.0, None)
        log_sin = np.log(np.sin(np.deg2rad(distance)))
        log_norm = np.log(2.0 * self._half_norm * 180.0 / np.pi)
        inside = (value >= self.low) & (value <= self.high)
        return np.where(inside, log_sin - log_norm, -np.inf)

    @property
    def mean(self):
        a, b = self._radians
        c_a, c_b = np.cos(a), np.cos(b)
        num = (np.sin(b) - b * c_b) - (np.sin(a) - a * c_a)
        mean = np.rad2deg(num / (2.0 * self._half_norm))
        return np.clip(mean, self.low, self.high)


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

    _poles = (-onp.pi / 2, onp.pi / 2)

    def __init__(self, low=-np.pi / 2, high=np.pi / 2, *, validate_args=None):
        _check_range("IsotropicLatitude", low, high, onp.pi / 2, "lat")
        self._init(low, high, validate_args)

    # Cancellation-resistant, as for the inclination: sin b - sin a =
    # 2 cos((a+b)/2) sin((b-a)/2) =: 2 S, and near a pole 1 -/+ sin x =
    # 2 sin^2(delta/2) with delta the colatitude from that pole.

    @property
    def _half_norm(self):
        return np.cos(0.5 * (self.low + self.high)) * np.sin(
            0.5 * (self.high - self.low)
        )

    def icdf(self, q):
        qs = q * self._half_norm
        quarter = 0.25 * np.pi
        # North pole: sin^2((pi/2 - x)/2) = sin^2(pi/4 - a/2) - qs.
        north = np.sin(quarter - 0.5 * self.low) ** 2 - qs
        # South pole: sin^2((pi/2 + x)/2) = sin^2(pi/4 + a/2) + qs.
        south = np.sin(quarter + 0.5 * self.low) ** 2 + qs
        use_north = np.sin(self.low) + 2.0 * qs >= 0.0
        # A safe argument for the branch not taken, as in
        # IsotropicInclination.icdf.
        north = np.where(use_north, north, 0.25)
        south = np.where(use_north, 0.25, south)
        half_pi = 0.5 * np.pi
        x = np.where(
            use_north,
            half_pi - 2.0 * np.arcsin(np.sqrt(np.clip(north, 0.0, 1.0))),
            2.0 * np.arcsin(np.sqrt(np.clip(south, 0.0, 1.0))) - half_pi,
        )
        return _clip_through(x, self.low, self.high)

    def cdf(self, value):
        a = self.low
        x = np.clip(value, self.low, self.high)
        num = np.sin(0.5 * (x - a)) * np.cos(0.5 * (x + a))
        return np.clip(num / self._half_norm, 0.0, 1.0)

    def log_prob(self, value):
        # cos(lat) from the distance to the nearer pole, in the dtype of
        # value, so it is exactly 0 at the (rounded) endpoints, not negative.
        half_pi = np.asarray(np.pi / 2, np.result_type(value, float))
        distance = np.clip(half_pi - np.abs(value), 0.0, None)
        log_cos = np.log(np.sin(distance))
        log_norm = np.log(2.0 * self._half_norm)
        inside = (value >= self.low) & (value <= self.high)
        return np.where(inside, log_cos - log_norm, -np.inf)

    @property
    def mean(self):
        lo, hi = self.low, self.high
        num = (np.cos(hi) + hi * np.sin(hi)) - (np.cos(lo) + lo * np.sin(lo))
        mean = num / (np.sin(hi) - np.sin(lo))
        return np.clip(mean, lo, hi)


@dataclasses.dataclass(frozen=True)
class PopulationScale:
    """Member ``index`` of a log-normal population of scales.

    Called with the dict of sampled values, it returns the member's scale s_k;
    see [`hierarchical_scales`][virgil.priors.hierarchical_scales]. In the
    centred form [`log_prior`][virgil.priors.PopulationScale.log_prior] is the
    population density, which ``numpyro_model`` and ``fit`` add once per member
    used as a ``noise`` term. It is a frozen dataclass, so equal members compare
    (and hash) equal: repeated fits do not recompile.
    """

    name: str
    index: int
    centred: bool = True

    def _hyper(self, values):
        return values[f"{self.name}_median"], values[f"{self.name}_spread"]

    def __call__(self, values):
        if self.centred:
            return np.exp(values[f"{self.name}_log"][..., self.index])
        median, spread = self._hyper(values)
        z = values[f"{self.name}_z"][..., self.index]
        return median * np.exp(spread * z)

    def log_prior(self, values):
        """The member's population log density (zero when non-centred,
        where the standard normal prior on z_k carries it)."""
        if not self.centred:
            return np.zeros(())
        median, spread = self._hyper(values)
        log_scale = values[f"{self.name}_log"][..., self.index]
        return dist.Normal(np.log(median), spread).log_prob(log_scale)

    def residuals(self, values):
        """Residuals whose half sum of squares is ``-log_prior(values)`` + const."""
        if not self.centred:
            return np.zeros(0)
        median, spread = self._hyper(values)
        log_scale = values[f"{self.name}_log"][..., self.index]
        shift = 100.0  # keeps log(spread) + shift > 0 across float range
        return np.array(
            [
                (log_scale - np.log(median)) / spread,
                np.sqrt(2.0 * np.maximum(0.0, np.log(spread) + shift)),
            ]
        )

    def curvature_rows(self, values):
        """Mean and normalizer curvature rows of the centred population prior."""
        if not self.centred:
            return np.zeros(0)
        median, spread = self._hyper(values)
        log_scale = values[f"{self.name}_log"][..., self.index]
        return np.array(
            [
                (log_scale - np.log(median)) / jax.lax.stop_gradient(spread),
                np.sqrt(2.0) * np.log(spread),
            ]
        )


def hierarchical_scales(name, n, median=None, spread=None, centred=True):
    """``n`` positive scales drawn from a log-normal population.

    ``log s_k`` is normal about ``log(median)`` with standard deviation
    ``spread``; both are sampled, with log-uniform priors. Use it for
    nuisances that differ between epochs but share an instrument, passed as
    ``noise`` terms (one dict per dataset, in order). Use every member: an
    unused centred member has an improper flat prior. When to centre:
    [hierarchical scales](../conventions.md#hierarchical-scales).

    Parameters
    ----------
    name : str
        Prefix of the parameters ``"<name>_median"``, ``"<name>_spread"`` and
        the length-``n`` ``"<name>_log"`` (centred) or ``"<name>_z"``.
    n : int
        Number of scales, e.g. the number of epochs.
    median : numpyro.distributions.Distribution, optional
        Prior on the population median (default ``LogUniform(0.1, 10)``,
        for error scales whose neutral value is 1).
    spread : numpyro.distributions.Distribution, optional
        Prior on the standard deviation of log s (default
        ``LogUniform(0.01, 1)``: from 1% to a factor of e).
    centred : bool, optional
        Sample ``log s_k`` (default) or a standard normal ``z_k`` with
        ``s_k = median exp(spread z_k)``.

    Returns
    -------
    priors : dict
        The three priors, to merge into a ``priors`` dict.
    scales : list of PopulationScale
        One callable per member, each mapping the sampled values to s_k.

    Examples
    --------
    >>> import numpyro.distributions as dist
    >>> from virgil.priors import hierarchical_scales
    >>> priors, scales = hierarchical_scales("cp_scale", 3)
    >>> sorted(priors)
    ['cp_scale_log', 'cp_scale_median', 'cp_scale_spread']
    >>> noise = [{"phi_scale": s} for s in scales]
    >>> values = {"cp_scale_median": 2.0, "cp_scale_spread": 0.5,
    ...           "cp_scale_log": np.log(np.array([1.0, 2.0, 4.0]))}
    >>> round(float(scales[1](values)), 6)
    2.0
    """
    n = int(n)
    if n < 1:
        raise ValueError("hierarchical_scales: n must be at least 1.")
    if centred:
        members = dist.ImproperUniform(constraints.real, (), (n,))
        key = f"{name}_log"
    else:
        members = dist.Normal(0.0, 1.0).expand([n]).to_event(1)
        key = f"{name}_z"
    priors = {
        f"{name}_median": dist.LogUniform(0.1, 10.0)
        if median is None
        else median,
        f"{name}_spread": dist.LogUniform(0.01, 1.0)
        if spread is None
        else spread,
        key: members,
    }
    scales = [PopulationScale(name, k, bool(centred)) for k in range(n)]
    return priors, scales
