"""Angles sampled as 2-D vectors, so that no prior has a wrap boundary.

A prior on an angle θ (degrees) with support on an interval, such as
``Uniform(0, 360)``, numpyro's ``VonMises`` or
[`AxialVonMises`][virgil.orbits.AxialVonMises], reaches samplers and
[`fit`][virgil.fitting.fit] through a bijection onto the real line, so an
angle whose posterior straddles the end of the interval meets a wall there.
[`AngleVector`][virgil.angles.AngleVector] removes the wall: put it in a
``priors`` dict under the angle's path, and a vector v = r(cos θ, sin θ) in
ℝ² is sampled at the site ``"<path>_vec"`` instead. The angle itself, in
degrees in [0°, 360°), is what the model receives, and what
[`numpyro_model`][virgil.likelihood.numpyro_model] records as a deterministic
site ``"<path>"`` and ``fit`` reports under ``"<path>"`` (with the vector
under ``"<path>_vec"``).

The radius r is a nuisance with a ring prior, ∝ exp(-(r - 1)²/2s²), so the
mode is the unit circle and a maximum a posteriori fit keeps away from the
origin, where θ is undefined. The prior is rotationally symmetric, so θ is
exactly uniform unless a von Mises (or axial von Mises) prior is given;
that one enters as a chord, √κ (v̂ - m̂), the same form as virgil's
unprojected phase residuals (see
[`whitened_residuals`][virgil.likelihood.whitened_residuals]). Every term has
a least-squares form, so Levenberg–Marquardt and
[`gauss_newton_mass`][virgil.fitting.gauss_newton_mass] take it unchanged,
and the density is normalised in ℝ², so evidences stay normalised.

The construction follows Octofitter's ``UniformCircular`` (Thompson et al.
2023, AJ 166, 164) and exoplanet's ``Angle`` (Foreman-Mackey et al. 2021,
JOSS 6, 3285), which sample v ~ N(0, I); the ring and the chord priors are
virgil's.
"""

import jax
import jax.numpy as np
import numpyro.distributions as dist
from jax.scipy.special import i0e, ndtr
from numpyro.distributions import constraints

__all__ = ["AngleVector", "is_angle_vector", "vector_angle", "vector_site"]


def vector_site(path):
    """The numpyro site of the vector that carries the angle at ``path``."""
    return f"{path}_vec"


def is_angle_vector(prior):
    """Whether ``prior`` is an [`AngleVector`][virgil.angles.AngleVector]."""
    return isinstance(prior, AngleVector)


def vector_angle(vector):
    """The direction of ``vector`` (last axis of length 2), in degrees in
    [0°, 360°), measured from its first component towards its second."""
    vector = np.asarray(vector)
    return np.mod(np.rad2deg(np.arctan2(vector[..., 1], vector[..., 0])), 360)


def _log_radial_norm(width):
    """log ∫₀^∞ r exp(-(r - 1)²/2s²) dr, the ring's normaliser."""
    return np.log(
        width**2 * np.exp(-0.5 / width**2)
        + width * np.sqrt(2.0 * np.pi) * ndtr(1.0 / width)
    )


class AngleVector(dist.Distribution):
    """A prior on an angle (degrees), sampled as a 2-D vector.

    Put it in a ``priors`` dict, keyed by the angle's path, for
    [`fit`][virgil.fitting.fit],
    [`gauss_newton_mass`][virgil.fitting.gauss_newton_mass] or
    [`numpyro_model`][virgil.likelihood.numpyro_model]. They sample the
    vector v at the site ``"<path>_vec"`` and give the model its direction
    θ = atan2(v₂, v₁) in degrees, in [0°, 360°). A ``fit`` starts from the
    unit vector of the angle's starting value (in degrees, from ``init`` or
    the template), or from ``init["<path>_vec"]``.

    The density of v = r(cos θ, sin θ) is

        p(v) = exp(-(r - 1)²/2s²) / Z_s × p(θ),

    with Z_s = ∫₀^∞ r exp(-(r - 1)²/2s²) dr, so that r and θ are independent
    and θ has exactly the density p(θ) (per radian):

    * uniform, 1/2π, by default;
    * a von Mises, exp(κ(cos(θ - μ) - 1)) / (2π i0e(κ)), given ``mean`` and
      ``kappa``;
    * with ``axial=True``, the von Mises of 2θ, so that θ and θ + 180° are
      equally likely (as [`AxialVonMises`][virgil.orbits.AxialVonMises]).

    Its least-squares residuals (``residuals``) are the ring, (r - 1)/s,
    and for a von Mises the chord √κ (v̂ - m̂), with v̂ = v/r and m̂ =
    (cos μ, sin μ): half its square is κ(1 - cos(θ - μ)), the von Mises
    exponent, just as virgil's phase residuals 2 sin(Δ/2)/σ are chords with
    κ = 1/σ². The axial chord is the same on the doubled direction,
    (cos 2θ, sin 2θ) = (x² - y², 2xy)/r².

    The default, uniform θ, is the invariant prior for an angle. A von Mises
    prior is strong information, from an external measurement.

    The construction follows Octofitter's ``UniformCircular`` (Thompson et
    al. 2023, AJ 166, 164) and exoplanet's ``Angle`` (Foreman-Mackey et al.
    2021, JOSS 6, 3285), whose v ~ N(0, I) peaks at the origin, where θ is
    undefined: a maximum a posteriori fit would drive r → 0. The ring's mode
    is the unit circle instead, and every term has a least-squares form.

    Parameters
    ----------
    mean : float, optional
        Mean angle μ (degrees) of a von Mises prior; with ``axial=True``,
        ``mean + 180`` is equivalent.
    kappa : float, optional
        Concentration κ of the von Mises prior (of 2θ when ``axial``), whose
        width is about ``57.3 / sqrt(kappa)`` degrees (half that if
        ``axial``). Give both ``mean`` and ``kappa``, or neither.
    axial : bool, optional
        Whether the von Mises prior is on 2θ (an axis known modulo 180°).
    ring_width : float, optional
        The ring's radial width s (default 0.25). It does not change the
        prior on θ.
    """

    arg_constraints = {"ring_width": constraints.positive}
    pytree_data_fields = ("mean_deg", "kappa", "ring_width")
    pytree_aux_fields = ("axial",)
    support = constraints.real_vector

    def __init__(
        self,
        mean=None,
        kappa=None,
        *,
        axial=False,
        ring_width=0.25,
        validate_args=None,
    ):
        if (mean is None) != (kappa is None):
            raise ValueError(
                "AngleVector: give both mean and kappa for a von Mises "
                "prior, or neither for a uniform angle."
            )
        if axial and kappa is None:
            raise ValueError("AngleVector: axial=True needs mean and kappa.")
        for name, value in (
            ("mean", mean),
            ("kappa", kappa),
            ("ring_width", ring_width),
        ):
            if value is not None and np.ndim(value) != 0:
                raise ValueError(f"AngleVector: {name} must be a scalar.")
        if kappa is not None and not bool(np.all(np.asarray(kappa) >= 0)):
            raise ValueError("AngleVector: kappa must be non-negative.")
        self.mean_deg = None if mean is None else np.asarray(mean, float)
        self.kappa = None if kappa is None else np.asarray(kappa, float)
        self.ring_width = np.asarray(ring_width, float)
        self.axial = bool(axial)
        super().__init__(
            batch_shape=(), event_shape=(2,), validate_args=validate_args
        )

    @property
    def uniform(self):
        """Whether the prior on the angle is uniform."""
        return self.kappa is None

    def _directions(self, vector):
        """(v̂ or its doubled direction, m̂ or its doubled mean)."""
        x, y = vector[..., 0], vector[..., 1]
        r2 = x**2 + y**2
        mean = np.deg2rad(self.mean_deg)
        if self.axial:
            unit = np.stack([(x**2 - y**2) / r2, 2.0 * x * y / r2], -1)
            mean = 2.0 * mean
        else:
            unit = vector / np.sqrt(r2)[..., None]
        return unit, np.stack([np.cos(mean), np.sin(mean)])

    def residuals(self, vector):
        """Residuals whose half sum of squares is -log p(vector) + const.

        The ring, (r - 1)/s, then, for a von Mises prior, the chord
        √κ (v̂ - m̂) (two more rows).
        """
        vector = np.asarray(vector)
        r = np.sqrt(np.sum(vector**2, axis=-1))
        ring = np.atleast_1d((r - 1.0) / self.ring_width)
        if self.uniform:
            return ring
        unit, mean = self._directions(vector)
        return np.concatenate([ring, np.sqrt(self.kappa) * (unit - mean)])

    def log_prob(self, value):
        value = np.asarray(value)
        r = np.sqrt(np.sum(value**2, axis=-1))
        log_p = (
            -0.5 * ((r - 1.0) / self.ring_width) ** 2
            - _log_radial_norm(self.ring_width)
            - np.log(2.0 * np.pi)
        )
        if self.uniform:
            return log_p
        unit, mean = self._directions(value)
        chord2 = np.sum((unit - mean) ** 2, axis=-1)
        return log_p - 0.5 * self.kappa * chord2 - np.log(i0e(self.kappa))

    def _sample_radius(self, key, shape):
        """Draws of r ∝ r exp(-(r - 1)²/2s²), by rejection from a Normal
        at its mode r* = (1 + √(1 + 4s²))/2 (the ratio is log-concave)."""
        s = self.ring_width
        mode = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * s**2))

        def log_ratio(r):
            return np.log(r) - (mode - 1.0) * (2.0 * r - 1.0 - mode) / (
                2.0 * s**2
            )

        top = log_ratio(mode)

        def draw(key, r, done):
            key, k_r, k_u = jax.random.split(key, 3)
            trial = mode + s * jax.random.normal(k_r, shape)
            u = jax.random.uniform(k_u, shape)
            safe = np.where(trial > 0, trial, 1.0)
            accept = (trial > 0) & (np.log(u) < log_ratio(safe) - top)
            take = accept & ~done
            return key, np.where(take, trial, r), done | accept

        def cond(carry):
            return ~np.all(carry[2])

        def body(carry):
            return draw(*carry)

        start = (key, np.ones(shape), np.zeros(shape, bool))
        return jax.lax.while_loop(cond, body, start)[1]

    def sample(self, key, sample_shape=()):
        shape = tuple(sample_shape)
        k_r, k_theta, k_flip = jax.random.split(key, 3)
        r = self._sample_radius(k_r, shape)
        if self.uniform:
            theta = jax.random.uniform(k_theta, shape, maxval=2.0 * np.pi)
        else:
            mean = np.deg2rad(self.mean_deg)
            if self.axial:
                doubled = dist.VonMises(0.0, self.kappa).sample(k_theta, shape)
                flip = np.pi * jax.random.bernoulli(k_flip, 0.5, shape)
                theta = mean + 0.5 * doubled + flip
            else:
                theta = dist.VonMises(mean, self.kappa).sample(k_theta, shape)
        return np.stack([r * np.cos(theta), r * np.sin(theta)], axis=-1)
