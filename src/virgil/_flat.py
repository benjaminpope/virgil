"""Flat coordinates of priors, shared by ``fit`` and the samplers.

A prior that is an invariant measure written in a coordinate where it is
not uniform (a log-uniform scale, an isotropic inclination) is uniform in
some other coordinate of its parameter, its *flat coordinate*:
``_flat_coordinate`` finds it. ``_FlatBijection`` maps the real line onto
the parameter through that coordinate. [`fit`][virgil.fitting.fit]
optimizes in it, and [`numpyro_model`][virgil.likelihood.numpyro_model]
samples in it when asked (``flat_coordinates=True``; off by default, as the
SBC showed no gain): ``flat_sampled`` wraps a prior so that numpyro's
``biject_to`` of its support is the same bijection, with its Jacobian,
while the site keeps its name, its value (the parameter) and its density.
See ``design/sampler_flat_coordinates.md``.

Private; imports nothing from virgil.
"""

import jax
import jax.numpy as np
import numpyro.distributions as dist
from numpyro.distributions import constraints
from numpyro.distributions.transforms import Transform, biject_to


def _base(distribution):
    """``distribution`` without ``.expand(...)`` and ``.to_event(...)``.

    They change the shape, not the density's form (a flat prior stays flat,
    a Normal stays Normal).
    """
    while isinstance(
        distribution, (dist.Independent, dist.ExpandedDistribution)
    ):
        distribution = distribution.base_dist
    return distribution


def _flat_coordinate(distribution):
    """The coordinate in which a prior is uniform, or None.

    Returns ``(to_flat, from_flat, low, high)``: a monotonic map ``u =
    to_flat(x)`` of the parameter, its inverse, and the interval ``[low,
    high]`` of ``u`` on which the prior's density is constant. Such a prior
    is an invariant measure written in a coordinate where it is not flat:

    * ``LogUniform(a, b)``: ``u = log x`` on ``[log a, log b]``;
    * any prior with a ``flat_coordinate()`` method returning that tuple,
      such as an isotropic inclination (``u = cos i``) or latitude (``u =
      sin(lat)``).

    ``Uniform`` is its own flat coordinate (the identity) and keeps
    numpyro's bijection; Normal and other priors have none and are
    evaluated in the model's own parameters.
    """
    base = _base(distribution)
    if isinstance(base, _FlatPrior):
        return _flat_coordinate(base.prior)
    if isinstance(base, dist.LogUniform):
        low, high = base.low, base.high

        def from_flat(u):
            # exp(log a) can round below a (float32): clip the value into
            # the support, keeping exp's derivative (straight through).
            x = np.exp(u)
            return x + jax.lax.stop_gradient(np.clip(x, low, high) - x)

        return np.log, from_flat, np.log(low), np.log(high)
    declared = getattr(base, "flat_coordinate", None)
    if callable(declared):
        return declared()
    return None


class _FlatBijection:
    """Unconstrained z to a parameter uniform in ``u = to_flat(x)``.

    ``x = from_flat(lo + (hi - lo) sigmoid(z))``: numpyro's bijection of the
    flat coordinate's interval, followed by the map back to the parameter.
    """

    def __init__(self, to_flat, from_flat, low, high):
        self.to_flat, self.from_flat = to_flat, from_flat
        self.interval = biject_to(constraints.interval(low, high))

    def __call__(self, z):
        return self.from_flat(self.interval(z))

    def inv(self, x):
        return self.interval.inv(self.to_flat(x))

    def log_abs_det_jacobian(self, z):
        """``log |dx/dz|`` = ``log |du/dz|`` + ``log |dx/du|``, elementwise.

        ``from_flat`` acts elementwise, so its derivative is one
        forward-mode pass with a tangent of ones.
        """
        u = self.interval(z)
        log_du = self.interval.log_abs_det_jacobian(z, u)
        _, dx_du = jax.jvp(self.from_flat, (u,), (np.ones_like(u),))
        return log_du + np.log(np.abs(dx_du))


class _FlatSupport(constraints.Constraint):
    """The support of a prior, tagged so that ``biject_to`` maps it through
    the prior's flat coordinate (``_FlatTransform``). Membership is the
    prior's own support's."""

    event_dim = 0

    def __init__(self, prior):
        self.prior = prior

    def __call__(self, x):
        return _base(self.prior).support(x)

    def feasible_like(self, prototype):
        return _base(self.prior).support.feasible_like(prototype)

    def tree_flatten(self):
        return (self.prior,), (("prior",), {})


class _FlatTransform(Transform):
    """numpyro's form of ``_FlatBijection``, for NUTS and its init."""

    domain = constraints.real

    def __init__(self, prior):
        self.prior = prior

    @property
    def codomain(self):
        return _FlatSupport(self.prior)

    @property
    def _bijection(self):
        return _FlatBijection(*_flat_coordinate(self.prior))

    def __call__(self, z):
        return self._bijection(z)

    def _inverse(self, x):
        return self._bijection.inv(x)

    def log_abs_det_jacobian(self, z, x, intermediates=None):
        return self._bijection.log_abs_det_jacobian(z)

    def tree_flatten(self):
        return (self.prior,), (("prior",), {})


@biject_to.register(_FlatSupport)
def _(constraint):
    return _FlatTransform(constraint.prior)


class _FlatPrior(dist.Distribution):
    """``prior`` with a support whose bijection is its flat coordinate.

    The density, samples, shapes and moments are ``prior``'s; only the
    unconstrained coordinates in which numpyro's NUTS and initialization
    work change, to ``_FlatBijection``'s (the same as ``fit``'s).
    """

    pytree_data_fields = ("prior",)

    def __init__(self, prior):
        self.prior = prior
        super().__init__(
            batch_shape=prior.batch_shape,
            event_shape=prior.event_shape,
            validate_args=False,
        )

    @constraints.dependent_property(is_discrete=False)
    def support(self):
        support = _FlatSupport(_base(self.prior))
        if self.prior.event_dim:
            return constraints.independent(support, self.prior.event_dim)
        return support

    def sample(self, key, sample_shape=()):
        return self.prior.sample(key, sample_shape)

    def log_prob(self, value):
        return self.prior.log_prob(value)

    @property
    def mean(self):
        return self.prior.mean

    @property
    def variance(self):
        return self.prior.variance


def flat_sampled(prior):
    """``prior``, sampled in its flat coordinate if it has one.

    Priors without a flat coordinate (Uniform, Normal, angle vectors,
    ...) are returned unchanged, so they keep numpyro's own bijection.
    """
    if isinstance(prior, _FlatPrior) or _flat_coordinate(prior) is None:
        return prior
    return _FlatPrior(prior)
