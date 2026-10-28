"""Angles sampled as 2-D vectors, so that no prior has a wrap boundary.

[`AngleVector`][virgil.angles.AngleVector] in a ``priors`` dict, under an
angle's path, samples a vector ``v = r(cos θ, sin θ)`` at the site
``"<path>_vec"`` instead of the angle; the model receives θ in degrees in
[0°, 360°), recorded (by [`numpyro_model`][virgil.likelihood.numpyro_model])
and reported (by [`fit`][virgil.fitting.fit]) under ``"<path>"``. See
[angle priors without a wrap](../conventions.md#angle-vectors).
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


class AngleVector(dist.Distribution):
    """A prior on an angle (degrees), sampled as a 2-D vector.

    Put it in a ``priors`` dict, keyed by the angle's path, for
    [`fit`][virgil.fitting.fit],
    [`gauss_newton_mass`][virgil.fitting.gauss_newton_mass] or
    [`numpyro_model`][virgil.likelihood.numpyro_model]. The vector is
    sampled at ``"<path>_vec"``, giving θ = atan2(v₂, v₁) in [0°, 360°); a
    ``fit`` starts from the starting angle or ``init["<path>_vec"]``. The
    prior on θ is uniform unless ``mean`` and ``kappa`` give a von Mises
    prior; see [angle priors](../conventions.md#angle-vectors).

    Parameters
    ----------
    mean, kappa : float, optional
        Mean angle (degrees) and concentration of a von Mises prior (of 2θ
        when ``axial``), of width about ``57.3 / sqrt(kappa)`` degrees (half
        if ``axial``). Give both, or neither. It is strong information from
        an external measurement; with ``axial``, ``mean + 180`` is equivalent.
    axial : bool, optional
        Whether the von Mises prior is on 2θ (an axis known modulo 180°).
    ring_width : float, optional
        The ring's radial width (default 0.25); it does not change the
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
        self.mean_deg = (
            None
            if mean is None
            else (mean if isinstance(mean, jax.Array) else float(mean))
        )
        self.kappa = (
            None
            if kappa is None
            else (kappa if isinstance(kappa, jax.Array) else float(kappa))
        )
        self.ring_width = (
            ring_width
            if isinstance(ring_width, jax.Array)
            else float(ring_width)
        )
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
        vector = np.asarray(vector, dtype=np.result_type(vector, float))
        dtype = vector.dtype
        x, y = vector[..., 0], vector[..., 1]
        r2 = x**2 + y**2
        mean = np.deg2rad(np.asarray(self.mean_deg, dtype))
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
        vector = np.asarray(vector, dtype=np.result_type(vector, float))
        dtype = vector.dtype
        r = np.sqrt(np.sum(vector**2, axis=-1))
        ring = np.atleast_1d((r - 1.0) / np.asarray(self.ring_width, dtype))
        if self.uniform:
            return ring
        unit, mean = self._directions(vector)
        kappa = np.asarray(self.kappa, dtype)
        return np.concatenate([ring, np.sqrt(kappa) * (unit - mean)])

    def log_prob(self, value):
        value = np.asarray(value, dtype=np.result_type(value, float))
        dtype = value.dtype
        r = np.sqrt(np.sum(value**2, axis=-1))
        w = np.asarray(self.ring_width, dtype)
        # log ∫₀^∞ r exp(-(r - 1)²/2s²) dr, the ring's normalizer.
        log_norm = np.log(
            w**2 * np.exp(-0.5 / w**2)
            + w * np.sqrt(2.0 * np.pi) * ndtr(1.0 / w)
        )
        log_p = -0.5 * ((r - 1.0) / w) ** 2 - log_norm - np.log(2.0 * np.pi)
        if self.uniform:
            return log_p
        unit, mean = self._directions(value)
        kappa = np.asarray(self.kappa, dtype)
        chord2 = np.sum((unit - mean) ** 2, axis=-1)
        return log_p - 0.5 * kappa * chord2 - np.log(i0e(kappa))

    def _sample_radius(
        self, key, shape
    ):  # deliberate: 32-line sampler with nested loop fns; inlined it would bury sample
        """Draws of r ∝ r exp(-(r - 1)²/2s²), by rejection from a Normal
        at its mode r* = (1 + √(1 + 4s²))/2 (the ratio is log-concave)."""
        s = np.asarray(self.ring_width, float)
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
            mean = np.deg2rad(np.asarray(self.mean_deg, float))
            kappa = np.asarray(self.kappa, float)
            if self.axial:
                doubled = dist.VonMises(0.0, kappa).sample(k_theta, shape)
                flip = np.pi * jax.random.bernoulli(k_flip, 0.5, shape)
                theta = mean + 0.5 * doubled + flip
            else:
                theta = dist.VonMises(mean, kappa).sample(k_theta, shape)
        return np.stack([r * np.cos(theta), r * np.sin(theta)], axis=-1)
