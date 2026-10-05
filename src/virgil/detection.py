"""Detection statistics for companion searches, for ROC curves.

[`detection_statistics`][virgil.detection.detection_statistics] reduces a
companion search over a grid to three numbers that measure how strongly the
data prefer a companion to none:

- ``delta_chi2``, the profile likelihood ratio 2 [max log L − log L₀] over
  the grid, with the companion flux constrained to be non-negative;
- ``log_bayes_factor``, the log evidence ratio of "a companion somewhere on
  the grid" to "no companion", marginalised over the grid;
- ``max_snr``, the largest best-fit flux over its Laplace uncertainty, the
  significance map of the composition tutorial.

The null hypothesis is the same model with the companion flux set to zero
and every other parameter as given, so fixed known components (a resolved
star, a disk, a companion already found) stay in both hypotheses.

The function is traceable in the data: inside ``jax.jit`` or
``jax.lax.map`` over simulated observations it compiles once, which is what
Monte Carlo estimates of false-alarm rates and completeness need. The
design is described in ``design/detection_roc.md``.

[`local_nsigma`][virgil.detection.local_nsigma] converts ``delta_chi2`` to
Wilks's Gaussian-equivalent significance at a single position. It ignores
the look-elsewhere effect of searching a grid, so it overstates the
significance of the best of many positions; calibrate with simulations.

The simulations:

- [`gaussian_null`][virgil.detection.gaussian_null] and
  [`bootstrap_null`][virgil.detection.bootstrap_null] build simulators,
  ``simulate(key, scene=None) -> OIData``: Gaussian noise from a template's
  errors, or a residual bootstrap of real data about the null scene.
  [`rescale_errors`][virgil.detection.rescale_errors] first scales errors so
  that the null scene has χ²_r = 1.
- [`injection_grid`][virgil.detection.injection_grid] lays out companions
  to inject, and
  [`injection_recovery`][virgil.detection.injection_recovery] runs the
  statistics on null and injected simulations with one compiled kernel.
- Its result, a [`DetectionMC`][virgil.detection.DetectionMC], holds plain
  NumPy arrays and gives empirical false-alarm probabilities, thresholds,
  ROC curves, completeness maps and contrast curves; it saves, loads and
  concatenates, so that array jobs can be merged.
"""

import dataclasses
import hashlib
import json
import math
import types
import warnings

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from ._grid import (
    batch_size_or_default,
    coordinate_points,
    ordered_values,
    resolve_grid_keys,
    warn_unconverged,
)
from ._utils import concrete
from .grid_fit import (
    _best_grid_flux,
    _laplace_flux_uncertainty_grid,
    _likelihood_grid,
    _refine_flux_grid,
)
from .likelihood import build_model, loglike, whitened_residuals
from .limits import nsigma


__all__ = [
    "DetectionMC",
    "bootstrap_null",
    "detection_statistics",
    "gaussian_null",
    "injection_grid",
    "injection_recovery",
    "local_nsigma",
    "rescale_errors",
]

STATISTICS = ("delta_chi2", "log_bayes_factor", "max_snr")

# The flux axis should span the likelihood peak's full width at half
# maximum with at least this many steps. A grid sum over a Gaussian of
# width sigma with spacing s is off by about 2 exp(-2 pi^2 sigma^2 / s^2)
# relative: ~1e-6 at two steps per FWHM (s = 1.18 sigma), but 20% at
# s = 3 sigma.
MIN_PEAK_STEPS = 2.0
FWHM_PER_SIGMA = 2.0 * np.sqrt(2.0 * np.log(2.0))


def detection_statistics(
    data, model, samples_dict, *, flux_param=None, batch_size=None
):
    """Detection statistics of a companion search over a grid.

    Parameters
    ----------
    data : OIData
        Data to search. It may be traced (e.g. a simulation built inside
        ``jax.lax.map`` with
        [`OIData.with_model`][virgil.oidata.OIData.with_model]); the
        function then compiles once for all of them.
    model : SourceModel or class
        Companion model, as for
        [`likelihood_grid`][virgil.grid_fit.likelihood_grid]: a template
        whose parameters at the paths in ``samples_dict`` are varied (e.g.
        a [System][virgil.models.System] with ``"comp.dra"``), or a class
        called with ``samples_dict``'s keys (e.g. ``BinaryModelCartesian``).
        The null hypothesis is this model with ``flux_param`` set to zero
        and everything else as given.
    samples_dict : dict[str, array-like]
        Grid axes, as for
        [`optimized_flux_grid`][virgil.grid_fit.optimized_flux_grid]: the
        coordinate keys (e.g. ``dra``, ``ddec`` in mas) and a non-negative
        flux axis. The flux axis is both the starting grid of the flux
        optimizer and the flux prior of ``log_bayes_factor``.
    flux_param : str, optional
        The key of ``samples_dict`` holding the companion flux. By default,
        the one key whose last part is ``flux``.
    batch_size : int, optional
        Number of grid points evaluated at once, as for
        [`likelihood_grid`][virgil.grid_fit.likelihood_grid].

    Returns
    -------
    dict[str, jax.Array]
        Scalars:

        - ``delta_chi2``: 2 [max over positions and fluxes ≥ 0 of log L −
          log L₀], always ≥ 0. At each position the flux is the best grid
          flux refined by BFGS (as in
          [`optimized_flux_grid`][virgil.grid_fit.optimized_flux_grid]);
          where the unconstrained best flux is negative the constrained
          one is 0, so that position adds nothing.
        - ``log_bayes_factor``: log of the prior-weighted mean of L / L₀
          over the full grid, ``logsumexp(log L − log L₀ + log w)``. The
          prior weights ``w`` sum to 1 and follow each axis's own spacing:
          trapezoid weights in the grid index (1 inside, ½ at the two
          ends) along every axis. So the prior is uniform over the
          searched box for evenly spaced coordinates, and uniform in log
          flux between the axis's ends for a log-spaced flux axis (uniform
          in flux for a linear one). Positive values favour a companion.
        - ``max_snr``: the largest unconstrained best flux over its Laplace
          uncertainty, over positions (NaN-safe).
        - one entry per key of ``samples_dict`` (e.g. ``dra``, ``ddec``,
          ``flux``): the position where ``delta_chi2`` is reached and the
          non-negative best flux there. If ``delta_chi2`` is 0 the flux is
          0 and the position is the first grid position.
        - ``flux_peak_steps``: a diagnostic, the full width at half
          maximum of the likelihood peak in flux at that position (2.355
          times the Laplace uncertainty) divided by the local spacing of
          the flux axis. Below about 2 the flux axis does not resolve the
          peak and ``log_bayes_factor`` is inaccurate.

    Warns
    -----
    RuntimeWarning
        With concrete (untraced) data only: if the flux axis does not
        resolve the likelihood peak (``flux_peak_steps`` < 2), if the
        best flux lies above the flux axis, or if the flux optimizer did not
        converge at some positions. Under ``jit``/``lax.map`` nothing is
        checked; inspect ``flux_peak_steps`` instead.

    Notes
    -----
    Every statistic uses the full likelihood grid, evaluated once. The flux
    axis must resolve the likelihood peak in flux for ``log_bayes_factor``
    to approximate the evidence integral, and the coordinate spacing must
    resolve it in position (not checked); a coarse grid still gives a
    valid test statistic when its null distribution is simulated, but not
    an accurate evidence. A strong detection has a narrow peak, so a
    log-spaced axis needs many points per decade: the peak's relative width
    (FWHM over flux) is about 2.4/SNR, so two steps across it at SNR 10
    need a step of 0.12 in ln(flux), about 20 points per decade.

    The trapezoid weights make ``log_bayes_factor`` a trapezoid-rule
    integral over a prior with fixed bounds (the axes' ends), so it
    converges as the grid is refined. Equal weights per point would put
    the prior's edges half a step beyond the axes' ends, and, since L / L₀
    stays near 1 far from a companion, would change the result at first
    order in the step.

    ``delta_chi2`` at one fixed position follows ½δ₀ + ½χ²₁ under the null
    (Chernoff 1954, the flux being on its boundary at 0); searching a grid
    makes it larger (the look-elsewhere effect). See
    [`local_nsigma`][virgil.detection.local_nsigma].

    Examples
    --------
    Statistics for many simulated null observations, compiled once:

    >>> keys = jax.random.split(jax.random.PRNGKey(0), 100)
    >>> stats = jax.lax.map(
    ...     lambda key: detection_statistics(
    ...         template.with_model(null_scene, key=key), model, grid
    ...     ),
    ...     keys,
    ... )  # doctest: +SKIP
    """
    params, coord_keys, flux_key = _resolve_keys(samples_dict, flux_param)
    stats, success = _detection_statistics(
        data,
        model,
        samples_dict,
        params=params,
        coord_keys=coord_keys,
        flux_key=flux_key,
        batch_size=batch_size_or_default(batch_size, data),
    )
    _warn_if_concrete(stats, success, samples_dict, flux_key)
    return stats


def local_nsigma(delta_chi2):
    """Local (Wilks) significance of ``delta_chi2``, in Gaussian sigma.

    The significance a ``delta_chi2`` would have **at one position fixed in
    advance**: under the null hypothesis it then follows ½δ₀ + ½χ²₁ (one
    flux, bounded at zero), whose upper tail at ``delta_chi2`` is the
    one-sided Gaussian tail at ``sqrt(delta_chi2)``. This is
    [`nsigma`][virgil.limits.nsigma] with one degree of freedom.

    It is local: it does not account for the look-elsewhere effect of
    taking the best of many grid positions, so for a search it overstates
    the significance. The global false-alarm probability needs simulations
    of the null over the same grid.

    Parameters
    ----------
    delta_chi2 : float or array-like
        Profile likelihood ratio from
        [`detection_statistics`][virgil.detection.detection_statistics].

    Returns
    -------
    array-like
        Local significance in Gaussian sigma (``sqrt(delta_chi2)`` up to
        rounding; it saturates at about 13σ in float32).
    """
    delta_chi2 = jnp.maximum(jnp.asarray(delta_chi2, dtype=float), 0.0)
    return nsigma(delta_chi2, 1.0, 1)


def _resolve_keys(samples_dict, flux_param):
    """Grid keys as ``resolve_grid_keys``, refusing statistic names."""
    params, coord_keys, flux_key = resolve_grid_keys(samples_dict, flux_param)
    clash = set(params) & (set(STATISTICS) | {"flux_peak_steps"})
    if clash:
        raise ValueError(
            f"samples_dict keys {sorted(clash)} clash with the names of the "
            "returned statistics."
        )
    return params, coord_keys, flux_key


@eqx.filter_jit
def _detection_statistics(
    data, model, samples_dict, params, coord_keys, flux_key, batch_size
):
    """Jitted implementation of :func:`detection_statistics`.

    Returns ``(stats, converged)``, where ``converged`` is the flux
    optimizer's convergence at every position.
    """
    loglike_im = _likelihood_grid(
        data, model, samples_dict, params, batch_size
    )
    grid_flux, grid_loglike = _best_grid_flux(
        loglike_im, samples_dict, params, flux_key
    )
    opt_flux, opt_loglike, converged = _refine_flux_grid(
        data,
        model,
        samples_dict,
        params,
        coord_keys,
        flux_key,
        batch_size,
        grid_flux,
        grid_loglike,
    )
    coords, shape = coordinate_points(samples_dict, coord_keys)
    null_values = ordered_values(0.0, coords[0], params, coord_keys, flux_key)
    loglike0 = loglike(null_values, params, data, model)

    profile, profile_flux = _constrained_profile(
        grid_flux, grid_loglike, opt_flux, opt_loglike, loglike0
    )
    best = jnp.argmax(profile.reshape(-1))
    delta_chi2 = 2.0 * (profile.reshape(-1)[best] - loglike0)
    best_flux = profile_flux.reshape(-1)[best]

    # Grid-marginalised evidence ratio over the trapezoid-weighted prior.
    log_ratio = jnp.where(
        jnp.isnan(loglike_im), -jnp.inf, loglike_im - loglike0
    )
    log_bayes_factor = jax.nn.logsumexp(
        log_ratio + _log_prior_weights(loglike_im.shape, log_ratio.dtype)
    )

    sigma = _laplace_flux_uncertainty_grid(
        opt_flux,
        data,
        model,
        samples_dict,
        params,
        coord_keys,
        flux_key,
        batch_size,
    )
    max_snr = jnp.nanmax(opt_flux / sigma)

    stats = {
        "delta_chi2": delta_chi2,
        "log_bayes_factor": log_bayes_factor,
        "max_snr": max_snr,
    }
    for key in params:
        if key == flux_key:
            stats[key] = best_flux
        else:
            stats[key] = coords[best, coord_keys.index(key)]
    stats["flux_peak_steps"] = (
        FWHM_PER_SIGMA
        * sigma.reshape(-1)[best]
        / _flux_step(samples_dict[flux_key], best_flux)
    )
    return stats, converged


def _constrained_profile(
    grid_flux, grid_loglike, opt_flux, opt_loglike, loglike0
):
    """Profile log likelihood and flux with flux >= 0 at every position.

    The refined optimum is used only where its flux is non-negative and it
    improves on the best grid point; otherwise (a negative or failed, e.g.
    NaN, refinement) the grid candidate is kept, whose flux is non-negative
    because flux axes are. A position whose candidate does not improve on
    the null gets the null's log likelihood and flux 0.
    """
    use_opt = (opt_flux >= 0.0) & (opt_loglike >= grid_loglike)
    alt_loglike = jnp.where(use_opt, opt_loglike, grid_loglike)
    alt_flux = jnp.where(use_opt, opt_flux, grid_flux)
    improves = (alt_flux > 0.0) & (alt_loglike > loglike0)
    return (
        jnp.where(improves, alt_loglike, loglike0),
        jnp.where(improves, alt_flux, 0.0),
    )


def _log_prior_weights(shape, dtype):
    """Log trapezoid weights in the grid index, normalised to sum to 1.

    Along each axis the weights are 1 inside and ½ at the two ends (1 for
    an axis of one point); the grid's weight is their outer product.
    """
    log_w = jnp.zeros(shape, dtype)
    for axis, n in enumerate(shape):
        w = np.ones(n)
        if n > 1:
            w[[0, -1]] = 0.5
        w = np.log(w / w.sum())
        log_w = log_w + w.reshape((-1,) + (1,) * (len(shape) - axis - 1))
    return log_w


def _flux_step(flux_axis, flux):
    """Spacing of ``flux_axis`` around ``flux`` (0 for a single point)."""
    axis = jnp.sort(jnp.asarray(flux_axis).reshape(-1))
    if axis.size < 2:
        return jnp.zeros((), axis.dtype)
    i = jnp.clip(jnp.searchsorted(axis, flux), 1, axis.size - 1)
    return axis[i] - axis[i - 1]


def _warn_if_concrete(stats, success, samples_dict, flux_key):
    """The checks of :func:`detection_statistics` that need numbers."""
    steps = concrete(stats["flux_peak_steps"])
    if steps is None:
        return
    warn_unconverged(success, "detection_statistics")
    flux = float(concrete(stats[flux_key]))
    top = float(np.max(np.asarray(samples_dict[flux_key])))
    if flux > top:
        warnings.warn(
            f"detection_statistics(): the best flux {flux:.3g} lies above "
            f"the flux axis (max {top:.3g}), so log_bayes_factor misses the "
            "likelihood peak; extend the axis.",
            RuntimeWarning,
            stacklevel=3,
        )
    elif float(steps) < MIN_PEAK_STEPS:
        warnings.warn(
            "detection_statistics(): the flux axis does not resolve the "
            f"likelihood peak ({float(steps):.2g} steps across its FWHM, "
            f"under {MIN_PEAK_STEPS:g}), so log_bayes_factor is "
            "inaccurate; refine the flux axis near the best flux.",
            RuntimeWarning,
            stacklevel=3,
        )


# ---------------------------------------------------------------------------
# Simulators of the null hypothesis (and of injections)
# ---------------------------------------------------------------------------


class _GaussianNull(eqx.Module):
    """Gaussian noise from a template's errors (see :func:`gaussian_null`)."""

    template: object
    null_scene: object
    error_scale: float = eqx.field(static=True)

    def __call__(self, key, scene=None):
        scene = self.null_scene if scene is None else scene
        return self.template.with_model(
            scene, key=key, noise_scale=self.error_scale
        )

    def describe(self):
        return {"kind": "gaussian", "error_scale": self.error_scale}


class _BootstrapNull(eqx.Module):
    """Residual bootstrap about the null scene (see :func:`bootstrap_null`)."""

    data: object
    null_scene: object
    vis_white: jax.Array  # whitened visibility residuals, one per sample
    phi_white: jax.Array  # whitened phase residuals (independent combos)
    method: str = eqx.field(static=True)

    def __call__(self, key, scene=None):
        scene = self.null_scene if scene is None else scene
        data = self.data
        vis_key, phi_key = jax.random.split(key)
        r_vis = self._shuffle(vis_key, self.vis_white) * data.d_vis
        w_phi = self._shuffle(phi_key, self.phi_white)
        if data._phases_wrap and data.cp_noise is not None:
            r_phi = data.cp_noise.colour(w_phi, data.d_phi, data.phi.size)
        else:
            r_phi = w_phi * data.d_phi
        prediction = data.model(scene)
        n_vis = data.vis.size
        n_phase = n_vis + data.phi.size
        return data.set(
            ["vis", "phi"],
            [prediction[:n_vis] + r_vis, prediction[n_vis:n_phase] + r_phi],
        )

    def _shuffle(self, key, white):
        """Sign-flipped or resampled (with replacement) residuals."""
        n = white.shape[0]
        if n == 0:
            return white
        if self.method == "sign_flip":
            return white * jax.random.rademacher(key, (n,), white.dtype)
        return white[jax.random.randint(key, (n,), 0, n)]

    def describe(self):
        return {"kind": "bootstrap", "method": self.method}


BOOTSTRAP_METHODS = ("sign_flip", "resample")


def gaussian_null(template, null_scene, *, error_scale=1.0):
    """A simulator of the null hypothesis with Gaussian noise.

    Each draw is ``template.with_model(scene, key=key)``
    ([`OIData.with_model`][virgil.oidata.OIData.with_model]): the scene's
    prediction plus Gaussian noise from the template's errors, closure
    phases correlated through ``cp_noise``, and draws of the template's
    gains and closure-phase offsets if it has them.

    Parameters
    ----------
    template : OIData
        Sampling and errors to simulate. Its values are not used.
    null_scene : SourceModel
        The scene without a companion.
    error_scale : float, optional
        The true noise as a multiple of the template's errors. The draws
        still carry the template's errors, as real data analysed with
        mis-estimated errors would, so ``error_scale`` ≠ 1 tests how the
        statistics fare when the errors are wrong. (To change the errors
        themselves, pass ``template.with_error_scale(...)`` or the result
        of [`rescale_errors`][virgil.detection.rescale_errors].) It is a
        static Python float: a new value compiles anew.

    Returns
    -------
    callable
        ``simulate(key, scene=None) -> OIData``, observing ``scene`` (by
        default ``null_scene``) with fresh noise for each ``key``. It is
        an equinox Module, a pytree whose arrays ``jax.jit`` traces, and is
        traceable in ``key`` and ``scene``, so it runs under ``jax.lax.map``
        over keys.

    Examples
    --------
    >>> simulate = gaussian_null(template, BinaryModelCartesian(0, 0, 0))  # doctest: +SKIP
    >>> data = simulate(jax.random.PRNGKey(0))  # doctest: +SKIP
    """
    error_scale = float(error_scale)
    if not (np.isfinite(error_scale) and error_scale >= 0.0):
        raise ValueError(
            f"error_scale must be finite and non-negative, not {error_scale}."
        )
    return _GaussianNull(template, null_scene, error_scale)


def rescale_errors(data, null_scene):
    """Scale the errors so that the null scene has χ²_r = 1.

    The visibility errors and the phase errors are scaled separately, each
    by ``s = sqrt(χ² / n)``: χ² is the sum of squares of that block of
    [`whitened_residuals`][virgil.likelihood.whitened_residuals] of
    ``null_scene`` (including the periodic penalty rows of correlated
    closure phases), and ``n`` its independent observables, as in
    [`n_independent`][virgil.oidata.OIData.n_independent] (the
    independent closure-phase combinations, not the triangles). Without
    gains, the whitened residuals scale as 1 / s, so afterwards each block
    has χ² / n = 1 exactly; with gains, approximately.

    Parameters
    ----------
    data : OIData
        Data to rescale, concrete (not traced).
    null_scene : SourceModel
        The scene without a companion, fitted to ``data``.

    Returns
    -------
    data : OIData
        A copy with ``d_vis`` and ``d_phi`` scaled. Extra observables
        (``data.extras``) keep their errors.
    factors : dict[str, float]
        The scale factors, ``{"vis": s_vis, "phi": s_phi}`` (``s_phi`` is
        1 for data without phases).

    Notes
    -----
    ``n`` does not subtract the parameters fitted to obtain
    ``null_scene``; with few data and several fitted parameters, the
    factors are biased low by about ``p / (2n)``.
    """
    r = np.asarray(whitened_residuals(null_scene, data))
    n_vis = int(np.size(data.vis))
    n_phi = int(np.size(data.phi))
    if data._phases_wrap and data.cp_noise is not None:
        rows_phi, dof_phi = data.cp_noise.size + n_phi, data.cp_noise.size
    else:
        rows_phi, dof_phi = n_phi, n_phi
    s_vis = math.sqrt(float(np.sum(r[:n_vis] ** 2)) / n_vis) if n_vis else 1.0
    s_phi = 1.0
    if dof_phi:
        chi2_phi = float(np.sum(r[n_vis : n_vis + rows_phi] ** 2))
        s_phi = math.sqrt(chi2_phi / dof_phi)
    for name, s in (("vis", s_vis), ("phi", s_phi)):
        if not (np.isfinite(s) and s > 0.0):
            raise ValueError(
                f"rescale_errors(): the {name} scale factor is {s}; the "
                "null scene fits that block perfectly or not at all."
            )
    scaled = eqx.tree_at(
        lambda d: (d.d_vis, d.d_phi),
        data,
        (data.d_vis * s_vis, data.d_phi * s_phi),
    )
    return scaled, {"vis": s_vis, "phi": s_phi}


def bootstrap_null(data, null_scene, *, method="sign_flip"):
    """A simulator of the null hypothesis by residual bootstrap.

    The residuals of ``data`` about ``null_scene``'s prediction are
    whitened, sign-flipped (``"sign_flip"``, a wild bootstrap) or resampled
    with replacement (``"resample"``), re-coloured and added back to the
    prediction of the scene being simulated:

    - visibilities are independent per sample: whitened as r / σ;
    - unprojected phases are wrapped into [−π, π) first, as in
      [`OIData.residuals`][virgil.oidata.OIData.residuals]. Uncorrelated
      phases are whitened as Δ / σ, which equals the likelihood's chord
      2 sin(Δ/2) / σ to O(Δ³); both are odd in Δ, so a sign flip flips
      the likelihood's whitened residual exactly;
    - closure phases from four or more telescopes are whitened with
      ``data.cp_noise`` (``ClosureNoise.whiten``) into independent
      combinations, which are flipped or resampled and re-coloured with
      ``ClosureNoise.colour``, so the triangles' correlations survive;
    - projected phases (kernel or DISCO phases) are linear and not
      wrapped.

    Sign flipping keeps each whitened residual's magnitude, so a
    heteroscedastic or mis-estimated error survives into every draw, and it
    removes any mean offset; it is the default. Resampling mixes the
    residuals of different samples (within the visibilities, and within the
    phases).

    Parameters
    ----------
    data : OIData
        The real data, concrete. Data with gains, closure-phase offsets or
        extra observables are not supported.
    null_scene : SourceModel
        The scene without a companion, fitted to ``data``.
    method : {"sign_flip", "resample"}, optional
        How to draw new whitened residuals.

    Returns
    -------
    callable
        ``simulate(key, scene=None) -> OIData``: ``scene``'s prediction (by
        default ``null_scene``'s) plus bootstrapped residuals, with
        ``data``'s errors. Like
        [`gaussian_null`][virgil.detection.gaussian_null] it is an equinox
        Module, traceable in ``key`` and ``scene``.

    Notes
    -----
    Caveats:

    - It needs a null that fits: residuals from a poor fit carry the
      misfit into every draw.
    - A real companion in the data is part of the residuals, so it
      appears, scrambled, in every null draw and biases the null
      distribution upward (and the injections' too).
    - Only the closure-phase covariance model is kept: correlations
      between visibilities, between baselines beyond the closure
      triangles, or between frames are lost.
    - That covariance model is Kammerer et al.'s equal-baseline-noise
      approximation (see ``virgil._closure``): with unequal errors within
      a group, the part of a real residual outside its column space is
      not recovered, and the re-coloured residuals are not exact closures
      of baseline phases.
    """
    if method not in BOOTSTRAP_METHODS:
        raise ValueError(
            f"method must be one of {BOOTSTRAP_METHODS}, not {method!r}."
        )
    if data.observable_kind != "split":
        raise ValueError(
            "bootstrap_null() supports visibility and phase observables "
            f"('split'), not {data.observable_kind!r}."
        )
    unsupported = [
        name
        for name, value in (
            ("gains", data.gains),
            ("closure-phase offsets", data.phase_offsets),
        )
        if value is not None
    ] + (["extra observables"] if data.extras else [])
    if unsupported:
        raise ValueError(
            f"bootstrap_null() does not support data with "
            f"{' or '.join(unsupported)}: their noise is not independent "
            "per sample. Use gaussian_null instead."
        )
    prediction = data.model(null_scene)
    n_vis = int(np.size(data.vis))
    n_phase = n_vis + int(np.size(data.phi))
    resid = data.flatten_data()[0][:n_phase] - prediction[:n_phase]
    vis_white = resid[:n_vis] / data.d_vis
    phi = resid[n_vis:]
    if data._phases_wrap:
        phi = jnp.mod(phi + jnp.pi, 2.0 * jnp.pi) - jnp.pi
        if data.cp_noise is not None:
            phi_white = data.cp_noise.whiten(phi, data.d_phi)[0]
        else:
            phi_white = phi / data.d_phi
    else:
        phi_white = phi / data.d_phi
    return _BootstrapNull(data, null_scene, vis_white, phi_white, method)


_SIMULATORS = (_GaussianNull, _BootstrapNull)


# ---------------------------------------------------------------------------
# Injections and the Monte Carlo driver
# ---------------------------------------------------------------------------


def injection_grid(separations, fluxes, n_pa, key):
    """Companions to inject: every separation and flux at random PAs.

    Parameters
    ----------
    separations : array-like
        Separations in mas (non-negative).
    fluxes : array-like
        Companion fluxes relative to the primary (non-negative; 0 gives
        null draws among the injections).
    n_pa : int
        Number of position angles per (separation, flux), each drawn
        uniformly in [0°, 360°).
    key : jax.Array or int
        Random key (or integer seed) for the position angles.

    Returns
    -------
    dict[str, numpy.ndarray]
        ``dra``, ``ddec`` (mas) and ``flux``, each of length
        ``len(separations) * len(fluxes) * n_pa``, ordered by separation,
        then flux, then draw. Positions follow virgil's convention: PA
        from North through East, ``dra = sep sin PA`` (East positive),
        ``ddec = sep cos PA`` (North positive).
    """
    sep = np.asarray(separations, dtype=float).reshape(-1)
    flux = np.asarray(fluxes, dtype=float).reshape(-1)
    n_pa = int(n_pa)
    if n_pa < 1:
        raise ValueError(f"n_pa must be positive; got {n_pa}.")
    if np.any(sep < 0.0) or np.any(flux < 0.0):
        raise ValueError("separations and fluxes must be non-negative.")
    shape = (sep.size, flux.size, n_pa)
    pa = jax.random.uniform(_as_key(key), shape, maxval=360.0)
    pa = np.radians(np.asarray(pa, dtype=float))
    sep_grid = np.broadcast_to(sep[:, None, None], shape)
    return {
        "dra": (sep_grid * np.sin(pa)).reshape(-1),
        "ddec": (sep_grid * np.cos(pa)).reshape(-1),
        "flux": np.broadcast_to(flux[None, :, None], shape).reshape(-1),
    }


def injection_recovery(
    template,
    null_scene,
    model,
    samples_dict,
    key,
    *,
    n_null,
    injections=None,
    noise="gaussian",
    match_radius=None,
    flux_param=None,
    draw_batch=1,
    chunk_size=None,
    batch_size=None,
    progress=True,
):
    """Detection statistics of simulated null and injected observations.

    Every draw simulates an observation (null, or with a companion
    injected), runs the companion search of
    [`detection_statistics`][virgil.detection.detection_statistics] on it,
    and keeps the statistics and the best position. One compiled kernel,
    ``(key, injection) -> statistics``, serves every draw, null and
    injected, mapped with ``jax.lax.map(..., batch_size=draw_batch)``, so
    memory stays bounded (there is no vmap over grid × draws).

    Parameters
    ----------
    template : OIData
        For ``noise="gaussian"``, the sampling and errors to simulate; for
        ``noise="bootstrap"``, the real data.
    null_scene : SourceModel
        The scene without a companion. It must predict the same data as
        ``model`` with zero companion flux, the null hypothesis that the
        statistics test (checked on ``template``).
    model : SourceModel or class
        Companion model, as for
        [`detection_statistics`][virgil.detection.detection_statistics]:
        a template with dotted paths or a class called with
        ``samples_dict``'s keys. Injections are this model at the injected
        coordinates and flux; null draws are this model at zero flux.
    samples_dict : dict[str, array-like]
        The search grid.
    key : jax.Array or int
        Random key, or an integer seed. Draw ``i`` of the null (or of the
        injections) uses ``fold_in(fold_in(key, 0 or 1), i)``, so results
        do not depend on ``draw_batch`` or ``chunk_size``.
    n_null : int
        Number of null draws (may be 0).
    injections : dict[str, array-like], optional
        Companions to inject, e.g. from
        [`injection_grid`][virgil.detection.injection_grid]: one array per
        grid key, found by its full name (``"comp.dra"``) or its last part
        (``"dra"``), all of one length. None for no injections.
    noise : {"gaussian", "bootstrap"} or simulator, optional
        Null noise model: [`gaussian_null`][virgil.detection.gaussian_null]
        or [`bootstrap_null`][virgil.detection.bootstrap_null] of
        ``template`` and ``null_scene`` with their defaults, or a simulator
        built by either (for ``error_scale`` or ``method="resample"``)
        from ``template`` itself (checked by fingerprint), whose null
        scene must predict the same data as ``null_scene``.
    match_radius : float, optional
        In mas, finite and non-negative. When set, an injection counts as
        detected only if its best position lies within ``match_radius`` of
        the injected one. Needs coordinate keys ending in ``dra`` and
        ``ddec`` (Cartesian, mas) or ``sep`` and ``pa`` (mas and degrees,
        as in ``BinaryModelAngular``). Stored in ``meta``; the best
        positions are stored either way.
    flux_param : str, optional
        The flux key of ``samples_dict``, as for the grid tools.
    draw_batch : int, optional
        Draws evaluated together (vectorised) within ``jax.lax.map``.
    chunk_size : int, optional
        Draws per call of the compiled kernel, rounded up to a multiple of
        ``draw_batch``. The null and the injected draws run in chunks of
        this size (the last one padded), with a progress bar over chunks.
        By default the smaller of 64 and the larger number of draws. Fix
        it to reuse the compilation across calls with different numbers
        of draws.
    batch_size : int, optional
        Grid points evaluated at once within each search, as for
        [`detection_statistics`][virgil.detection.detection_statistics].
    progress : bool, optional
        Show a ``tqdm.auto`` progress bar, if tqdm is installed.

    Returns
    -------
    DetectionMC
        The statistics of every draw, the injections, and metadata.

    Examples
    --------
    >>> grid = {"dra": jnp.linspace(-150, 150, 16),
    ...         "ddec": jnp.linspace(-150, 150, 16),
    ...         "flux": jnp.geomspace(1e-4, 0.03, 32)}
    >>> injections = injection_grid([60, 100], [1e-3, 3e-3], 100, 1)
    >>> mc = injection_recovery(
    ...     template, BinaryModelCartesian(0, 0, 0), BinaryModelCartesian,
    ...     grid, 0, n_null=1000, injections=injections,
    ... )  # doctest: +SKIP
    >>> mc.threshold("delta_chi2", 1.35e-3)  # doctest: +SKIP
    """
    params, coord_keys, flux_key = _resolve_keys(samples_dict, flux_param)
    names = _canonical_names(params, flux_key)
    n_null = int(n_null)
    if n_null < 0:
        raise ValueError(f"n_null must be non-negative; got {n_null}.")
    draw_batch = int(draw_batch)
    if draw_batch < 1:
        raise ValueError(f"draw_batch must be positive; got {draw_batch}.")
    if match_radius is not None:
        match_radius = float(match_radius)
        if not (np.isfinite(match_radius) and match_radius >= 0.0):
            raise ValueError(
                "match_radius must be finite and non-negative, not "
                f"{match_radius}."
            )
        short = set(names.values())
        if not any(set(pair) <= short for pair in _POSITION_KEYS):
            raise ValueError(
                "match_radius needs coordinate keys ending in 'dra' and "
                f"'ddec', or 'sep' and 'pa'; the grid has {list(coord_keys)}."
            )
    simulator = _simulator(noise, template, null_scene)
    _check_same_prediction(
        template,
        build_model(model, params, [0.0] * len(params)),
        null_scene,
        "the model at zero companion flux",
        "null_scene",
    )
    inj_values = _injection_values(injections, params, names)
    n_inj = inj_values.shape[0]
    chunk = _chunk_size(chunk_size, draw_batch, max(n_null, n_inj))
    seed = _seed_record(key)
    base = _raw_key(_as_key(key))
    static = {
        "params": params,
        "coord_keys": coord_keys,
        "flux_key": flux_key,
        "batch_size": batch_size_or_default(batch_size, template),
        "draw_batch": draw_batch if draw_batch > 1 else None,
    }
    bar = _progress_bar(-(-n_null // chunk) - (-n_inj // chunk), progress)

    def run(stream, values):
        base_key = jax.random.fold_in(base, stream)
        out = []
        for start in range(0, values.shape[0], chunk):
            part = values[start : start + chunk]
            padded = np.zeros((chunk, len(params)), dtype=np.float32)
            padded[: part.shape[0]] = part
            stats = _simulated_statistics(
                simulator,
                model,
                samples_dict,
                base_key,
                np.arange(start, start + chunk, dtype=np.int32),
                padded,
                **static,
            )
            out.append(
                {k: np.asarray(v)[: part.shape[0]] for k, v in stats.items()}
            )
            if bar is not None:
                bar.update(1)
        return _gather(out, params, names)

    try:
        null = run(0, np.zeros((n_null, len(params)), dtype=np.float32))
        injected = run(1, inj_values)
    finally:
        if bar is not None:
            bar.close()
    for k in params:
        # The injected values at full precision.
        injected[names[k]] = (
            np.zeros(0)
            if injections is None
            else np.asarray(
                _lookup(injections, k, names[k]), dtype=float
            ).reshape(-1)
        )

    from . import __version__

    meta = {
        "format": _FORMAT,
        "virgil_version": __version__,
        "params": list(params),
        "flux_param": flux_key,
        "names": {k: names[k] for k in params},
        "grid": {
            k: np.asarray(v, dtype=float).reshape(-1).tolist()
            for k, v in samples_dict.items()
        },
        "model": _describe(model),
        "null_scene": _describe(null_scene),
        "template": _fingerprint_data(template),
        "noise": simulator.describe(),
        "match_radius": match_radius,
        "n_null": n_null,
        "n_injected": n_inj,
        "seeds": [seed],
    }
    return DetectionMC(null=null, injected=injected, meta=meta)


@eqx.filter_jit
def _simulated_statistics(
    simulator,
    model,
    samples_dict,
    base_key,
    index,
    values,
    params,
    coord_keys,
    flux_key,
    batch_size,
    draw_batch,
):
    """The Monte Carlo kernel: statistics of one chunk of draws."""

    def one(args):
        i, row = args
        scene = build_model(model, params, [row[j] for j in range(row.size)])
        data = simulator(jax.random.fold_in(base_key, i), scene)
        stats, converged = _detection_statistics(
            data,
            model,
            samples_dict,
            params=params,
            coord_keys=coord_keys,
            flux_key=flux_key,
            batch_size=batch_size,
        )
        stats = dict(stats)
        stats["converged_fraction"] = jnp.mean(converged.astype(row.dtype))
        return stats

    return jax.lax.map(one, (index, values), batch_size=draw_batch)


_FORMAT = 1
# Statistics stored per draw, besides the best position (``best_<name>``).
_STORED = STATISTICS + ("flux_peak_steps", "converged_fraction")
# Coordinate names that give positions on the sky: Cartesian (mas), or
# separation (mas) and position angle (degrees, North through East).
_POSITION_KEYS = (("dra", "ddec"), ("sep", "pa"))
# Rows of null draws per bootstrap batch in DetectionMC.threshold, to bound
# its memory (about 2**22 values, 32 MB).
_BOOT_VALUES = 2**22
# Metadata that must agree for results to be concatenated.
_COMPATIBLE = (
    "format",
    "params",
    "flux_param",
    "names",
    "grid",
    "model",
    "null_scene",
    "template",
    "noise",
    "match_radius",
)


def _as_key(key):
    """A PRNG key from a key or an integer seed."""
    if isinstance(key, (int, np.integer)):
        return jax.random.PRNGKey(int(key))
    return key


def _raw_key(key):
    """Raw uint32 key data (typed keys are unwrapped)."""
    if jax.dtypes.issubdtype(getattr(key, "dtype", None), jax.dtypes.prng_key):
        return jax.random.key_data(key)
    return jnp.asarray(key)


def _seed_record(key):
    """JSON description of ``key``, to tell array jobs apart."""
    if isinstance(key, (int, np.integer)):
        return {"seed": int(key)}
    return {"key": np.asarray(_raw_key(key)).reshape(-1).tolist()}


def _canonical_names(params, flux_key):
    """Short names of the grid keys: their last dotted part.

    The outputs are stored under ``_STORED``, ``best_<name>`` and (for the
    injected values) ``<name>``; all of them must be distinct, or one
    would overwrite another.
    """
    names = {k: k.rsplit(".", 1)[-1] for k in params}
    names[flux_key] = "flux"
    short = list(names.values())
    fields = list(_STORED) + short + [f"best_{n}" for n in short]
    if len(set(fields)) != len(fields):
        raise ValueError(
            f"The grid keys {list(params)} have last parts {short}, which "
            "repeat or clash with the stored outputs (statistics, "
            "diagnostics and best_<name>); rename them."
        )
    return names


def _lookup(injections, key, name):
    if key in injections:
        return injections[key]
    if name in injections:
        return injections[name]
    raise KeyError(
        f"injections has no {key!r} (or {name!r}); it has "
        f"{sorted(injections)}."
    )


def _injection_values(injections, params, names):
    """``(n, len(params))`` float32 injection values in ``params`` order."""
    if injections is None:
        return np.zeros((0, len(params)), dtype=np.float32)
    columns = {
        k: np.asarray(_lookup(injections, k, names[k]), dtype=float).ravel()
        for k in params
    }
    if len({c.size for c in columns.values()}) != 1:
        raise ValueError("The injection arrays must all have one length.")
    flux = [c for k, c in columns.items() if names[k] == "flux"][0]
    if np.any(flux < 0.0):
        raise ValueError("Injected fluxes must be non-negative.")
    return np.stack([columns[k] for k in params], axis=1).astype(np.float32)


def _chunk_size(chunk_size, draw_batch, n_max):
    if chunk_size is None:
        chunk = min(64, max(n_max, 1))
    else:
        chunk = int(chunk_size)
        if chunk < 1:
            raise ValueError(f"chunk_size must be positive; got {chunk}.")
    return -(-chunk // draw_batch) * draw_batch


def _simulator(noise, template, null_scene):
    if isinstance(noise, _SIMULATORS):
        # The null check, batch sizing and metadata use ``template``, so
        # the simulator must observe through the same data.
        source = (
            noise.template if isinstance(noise, _GaussianNull) else noise.data
        )
        if source is not template:
            mine, theirs = _fingerprint(source), _fingerprint(template)
            if mine is None or mine != theirs:
                raise ValueError(
                    "The simulator was built from different data than "
                    "template (or from data that cannot be fingerprinted); "
                    "pass the simulator's own data as template."
                )
        _check_same_prediction(
            template,
            noise.null_scene,
            null_scene,
            "the simulator's null scene",
            "null_scene",
        )
        return noise
    if noise == "gaussian":
        return gaussian_null(template, null_scene)
    if noise == "bootstrap":
        return bootstrap_null(template, null_scene)
    raise ValueError(
        "noise must be 'gaussian', 'bootstrap' or a simulator from "
        f"gaussian_null or bootstrap_null, not {noise!r}."
    )


def _check_same_prediction(template, a, b, name_a, name_b, tol=1e-3):
    """Refuse two scenes whose predictions differ by more than tol σ."""
    errors = template.flatten_data()[1]
    diff = template.residuals(template.model(a), template.model(b)) / errors
    worst = float(np.max(np.abs(np.asarray(diff))))
    if not worst <= tol:
        raise ValueError(
            f"{name_a} and {name_b} predict different data (by up to "
            f"{worst:.3g}σ). Null draws simulate the model at zero flux, "
            "the null hypothesis the statistics test, so the two must "
            "agree."
        )


def _progress_bar(total, enabled):
    if not enabled or total == 0:
        return None
    try:
        from tqdm.auto import tqdm
    except ImportError:
        return None
    return tqdm(total=total, desc="injection_recovery", leave=False)


def _gather(chunks, params, names):
    """Concatenate per-chunk statistics; best positions as best_<name>."""
    out = {stat: _concat([c[stat] for c in chunks]) for stat in _STORED}
    for k in params:
        out[f"best_{names[k]}"] = _concat([c[k] for c in chunks])
    return out


def _concat(arrays):
    if not arrays:
        return np.zeros(0)
    return np.concatenate(arrays).astype(float)


class _Unfingerprintable(Exception):
    """An object whose content cannot be hashed reproducibly."""


def _fingerprint(obj):
    """Hash of a whole object, or None if part of it cannot be hashed.

    Equinox Modules (``OIData``, models, ``ClosureNoise``, ...) are
    dataclasses: every field is hashed, static or not, with the class of
    every node, so two objects hash equally only if they hold the same
    arrays (dtype, shape, bytes), the same Python values and the same
    structure. Classes and module-level functions are hashed by name;
    anything else (a lambda, a closure, an arbitrary object) makes the
    fingerprint None, so that results built on it are never merged.
    """
    h = hashlib.sha256()
    try:
        _feed(h, obj)
    except _Unfingerprintable:
        return None
    return h.hexdigest()[:16]


def _feed(h, obj):
    """Add ``obj`` to the hash ``h`` (see :func:`_fingerprint`)."""
    if obj is None or isinstance(obj, (bool, int, float, complex, str)):
        h.update(f"{type(obj).__name__}:{obj!r};".encode())
    elif isinstance(obj, (jax.Array, np.ndarray, np.generic)):
        try:
            a = np.ascontiguousarray(np.asarray(obj))
        except TypeError as err:  # e.g. typed PRNG keys
            raise _Unfingerprintable from err
        if a.dtype == object:
            raise _Unfingerprintable
        h.update(f"array:{a.dtype}:{a.shape};".encode())
        h.update(a.tobytes())
    elif isinstance(obj, (tuple, list)):
        h.update(f"{type(obj).__name__}:{len(obj)}(".encode())
        for item in obj:
            _feed(h, item)
        h.update(b")")
    elif isinstance(obj, dict):
        h.update(f"dict:{len(obj)}(".encode())
        for key in sorted(obj, key=repr):
            _feed(h, key)
            _feed(h, obj[key])
        h.update(b")")
    elif dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        h.update(f"{_qualname(type(obj))}(".encode())
        for field in dataclasses.fields(obj):
            h.update(f"{field.name}=".encode())
            _feed(h, getattr(obj, field.name, "<unset>"))
        h.update(b")")
    elif isinstance(obj, (type, types.FunctionType)) and "<" not in (
        obj.__qualname__
    ):
        h.update(f"{type(obj).__name__}:{_qualname(obj)};".encode())
    else:
        raise _Unfingerprintable


def _qualname(obj):
    return f"{getattr(obj, '__module__', '')}.{obj.__qualname__}"


def _fingerprint_data(data):
    """Shape and hash of the whole template (see :func:`_fingerprint`)."""
    return {
        "n_vis": int(np.size(data.vis)),
        "n_phi": int(np.size(data.phi)),
        "hash": _fingerprint(data),
    }


def _describe(model):
    """A JSON description of a model class, function or template."""
    named = isinstance(model, (type, types.FunctionType))
    name = _qualname(model if named else type(model))
    return {"type": name, "hash": _fingerprint(model)}


# ---------------------------------------------------------------------------
# The result container
# ---------------------------------------------------------------------------


@dataclasses.dataclass(eq=False)
class DetectionMC:
    """Statistics of simulated null and injected companion searches.

    Made by [`injection_recovery`][virgil.detection.injection_recovery];
    plain NumPy, with no JAX inside.

    Attributes
    ----------
    null : dict[str, numpy.ndarray]
        Per null draw: ``delta_chi2``, ``log_bayes_factor``, ``max_snr``,
        the diagnostics ``flux_peak_steps`` and ``converged_fraction`` (of
        the flux optimizer over grid positions), and the best position and
        flux as ``best_<name>`` (e.g. ``best_dra``, ``best_flux``), where
        ``name`` is the last dotted part of each grid key.
    injected : dict[str, numpy.ndarray]
        The same per injected draw, plus the injected values under their
        names (e.g. ``dra``, ``ddec``, ``flux``).
    meta : dict
        JSON-serialisable: the grid, fingerprints (hashes of every field,
        static or not) of the model, the null scene and the template, the
        noise model,
        ``match_radius``, the numbers of draws, the seeds and the virgil
        version.

    Notes
    -----
    Conventions: the false-alarm probability of a value is the fraction of
    null draws at or above it; at a threshold ``t`` a draw is detected
    when its statistic exceeds ``t`` (and, with ``match_radius``, its best
    position matches the injection). Separations are ``hypot(dra, ddec)``
    of the injections, or their ``sep`` for angular grids (``sep`` in mas,
    ``pa`` in degrees), in mas; fluxes are relative to the primary, as in
    [`absil_limits`][virgil.limits.absil_limits].

    The single-position null of ``delta_chi2`` is ½δ₀ + ½χ²₁, so a local
    3σ (``local_nsigma`` = 3, ``delta_chi2`` = 9) is a one-sided FAP of
    0.135% (``scipy.stats.norm.sf(3)``), not the two-sided 0.27%; over a
    grid the empirical FAP of that value is larger.
    """

    null: dict
    injected: dict
    meta: dict

    def __repr__(self):
        return (
            f"DetectionMC(n_null={self.n_null}, "
            f"n_injected={self.n_injected}, "
            f"noise={self.meta.get('noise')})"
        )

    @property
    def n_null(self):
        """Number of null draws."""
        return int(np.size(self.null["delta_chi2"]))

    @property
    def n_injected(self):
        """Number of injected draws."""
        return int(np.size(self.injected["delta_chi2"]))

    def _scores(self, part, stat):
        if stat not in STATISTICS:
            raise ValueError(
                f"stat must be one of {STATISTICS}, not {stat!r}."
            )
        x = np.asarray(part[stat], dtype=float)
        return np.where(np.isnan(x), -np.inf, x)

    def _null_scores(self, stat):
        x = self._scores(self.null, stat)
        if x.size == 0:
            raise ValueError("There are no null draws.")
        return x

    def _positions(self, prefix=""):
        """Injected (or best, with prefix "best_") ``(dra, ddec)`` in mas."""
        part = self.injected
        if f"{prefix}dra" in part and f"{prefix}ddec" in part:
            return part[f"{prefix}dra"], part[f"{prefix}ddec"]
        if f"{prefix}sep" in part and f"{prefix}pa" in part:
            sep = np.asarray(part[f"{prefix}sep"], dtype=float)
            pa = np.radians(np.asarray(part[f"{prefix}pa"], dtype=float))
            return sep * np.sin(pa), sep * np.cos(pa)
        raise ValueError(
            "The injections have neither 'dra' and 'ddec' nor 'sep' and "
            f"'pa'; they have {sorted(part)}."
        )

    def separations(self):
        """Separations of the injections in mas.

        ``hypot(dra, ddec)`` for Cartesian grids, or the injected ``sep``
        for angular ones (``sep``, ``pa`` in degrees).
        """
        if "dra" not in self.injected and "sep" in self.injected:
            return np.asarray(self.injected["sep"], dtype=float)
        return np.hypot(*self._positions())

    def matched(self):
        """Whether each injection's best position is within match_radius.

        All True when ``meta["match_radius"]`` is None. Angular positions
        (``sep``, ``pa`` in degrees) are compared on the sky.
        """
        radius = self.meta.get("match_radius")
        if radius is None:
            return np.ones(self.n_injected, dtype=bool)
        (x, y), (bx, by) = self._positions(), self._positions("best_")
        return np.hypot(bx - x, by - y) <= radius

    def false_alarm_probability(self, stat, value, *, confidence=0.95):
        """Empirical false-alarm probability of ``value``, with an interval.

        Parameters
        ----------
        stat : str
            ``"delta_chi2"``, ``"log_bayes_factor"`` or ``"max_snr"``.
        value : float or array-like
            Observed statistic(s).
        confidence : float, optional
            Confidence of the binomial interval.

        Returns
        -------
        fap, lower, upper : numpy.ndarray
            ``fap = (k + 1) / (n + 1)`` for ``k`` of the ``n`` null draws
            at or above ``value``: the Monte Carlo p-value, never 0, and
            conservative. ``lower`` and ``upper`` bound the true
            probability P(null ≥ value) given ``k`` of ``n``, with the
            exact (Clopper–Pearson) binomial interval.
        """
        from scipy import stats as sstats

        null = np.sort(self._null_scores(stat))
        n = null.size
        value = np.asarray(value, dtype=float)
        k = n - np.searchsorted(null, value, side="left")
        alpha = 1.0 - float(confidence)
        with np.errstate(invalid="ignore"):
            lower = np.where(
                k > 0, sstats.beta.ppf(alpha / 2, k, n - k + 1), 0.0
            )
            upper = np.where(
                k < n, sstats.beta.ppf(1 - alpha / 2, k + 1, n - k), 1.0
            )
        return (k + 1.0) / (n + 1.0), lower, upper

    def threshold(self, stat, fap, *, n_boot=200, seed=0):
        """Threshold of ``stat`` at false-alarm probability ``fap``.

        Parameters
        ----------
        stat : str
            The statistic.
        fap : float
            False-alarm probability (e.g. 1.35e-3, the one-sided Gaussian
            tail at 3σ).
        n_boot : int, optional
            Bootstrap resamples of the null draws, for the error.
        seed : int, optional
            Seed of the bootstrap.

        Returns
        -------
        threshold, error : float
            The ``1 - fap`` quantile of the null draws (linear
            interpolation), and its bootstrap standard deviation.

        Warns
        -----
        RuntimeWarning
            If fewer than one null draw is expected above the threshold
            (``fap * n < 1``): it is then about the largest null value and
            underestimates the true threshold.

        Notes
        -----
        The bootstrap runs in batches of resamples, holding about 2²²
        values at a time, so its memory does not grow with
        ``n_boot × n``.
        """
        null, value = self._threshold(stat, fap)
        q = 1.0 - float(fap)
        rng = np.random.default_rng(seed)
        n_boot = int(n_boot)
        rows = max(1, _BOOT_VALUES // null.size)
        quantiles = []
        for start in range(0, n_boot, rows):
            boot = rng.choice(
                null, size=(min(rows, n_boot - start), null.size)
            )
            quantiles.append(np.quantile(boot, q, axis=1))
        error = float(np.std(np.concatenate(quantiles))) if n_boot else np.nan
        return value, error

    def _threshold(self, stat, fap):
        """The null scores and their ``1 - fap`` quantile, without error."""
        null = self._null_scores(stat)
        fap = float(fap)
        if not 0.0 < fap < 1.0:
            raise ValueError(f"fap must lie in (0, 1); got {fap}.")
        if fap * null.size < 1.0:
            warnings.warn(
                f"threshold(): {null.size} null draws cannot resolve a FAP "
                f"of {fap:.3g}; the threshold is about the largest null "
                "value. Simulate more null draws.",
                RuntimeWarning,
                stacklevel=3,
            )
        return null, float(np.quantile(null, 1.0 - fap))

    def detected(self, stat, fap):
        """Whether each injection is detected at false-alarm probability fap.

        Its statistic exceeds :meth:`threshold` and, with ``match_radius``,
        its best position matches the injection.
        """
        return self._detected(stat, self._threshold(stat, fap)[1])

    def _detected(self, stat, threshold):
        scores = self._scores(self.injected, stat)
        return (scores > threshold) & self.matched()

    def _select(self, flux=None, sep_bin=None):
        keep = np.ones(self.n_injected, dtype=bool)
        if flux is not None:
            f = self.injected["flux"]
            if np.ndim(flux) == 0:
                keep &= np.isclose(f, float(flux), rtol=1e-6, atol=0.0)
            else:
                lo, hi = flux
                keep &= (f >= lo) & (f < hi)
        if sep_bin is not None:
            lo, hi = sep_bin
            sep = self.separations()
            keep &= (sep >= lo) & (sep < hi)
        return keep

    def roc(self, stat, flux=None, sep_bin=None):
        """ROC curve: true- against false-positive rate over thresholds.

        Parameters
        ----------
        stat : str
            The statistic.
        flux : float or (float, float), optional
            Only the injections of this flux (to relative precision 1e-6),
            or with ``lo <= flux < hi``.
        sep_bin : (float, float), optional
            Only the injections with ``lo <= separation < hi`` (mas).

        Returns
        -------
        fpr, tpr, thresholds : numpy.ndarray
            For each threshold ``t`` (``+inf``, then every distinct value
            of the statistic in decreasing order), the fraction of null
            draws with a statistic ≥ ``t``, and of the selected injections
            with a statistic ≥ ``t`` (and matched, with
            ``match_radius``). ``fpr`` rises from 0 to 1.
        """
        null = np.sort(self._null_scores(stat))
        keep = self._select(flux, sep_bin)
        if not np.any(keep):
            raise ValueError("No injections match the selection.")
        scores = self._scores(self.injected, stat)[keep]
        hits = np.sort(scores[self.matched()[keep]])
        values = np.unique(np.concatenate([null, scores]))[::-1]
        thresholds = np.concatenate([[np.inf], values])
        fpr = 1.0 - np.searchsorted(null, thresholds, "left") / null.size
        tpr = (hits.size - np.searchsorted(hits, thresholds, "left")) / (
            scores.size
        )
        return fpr, tpr, thresholds

    def auc(self, stat, flux=None, sep_bin=None):
        """Area under the :meth:`roc` curve (0.5 for no discrimination)."""
        fpr, tpr, _ = self.roc(stat, flux=flux, sep_bin=sep_bin)
        return float(np.trapezoid(tpr, fpr))

    def completeness(self, stat, fap, sep_bins=None, flux_bins=None):
        """Detection fraction of the injections by separation and flux.

        Parameters
        ----------
        stat : str
            The statistic.
        fap : float
            False-alarm probability of the detection threshold.
        sep_bins, flux_bins : array-like, optional
            Bin edges (mas, and flux). By default each distinct injected
            separation or flux (to six significant figures) is its own
            bin, which suits an
            [`injection_grid`][virgil.detection.injection_grid].

        Returns
        -------
        dict
            ``completeness`` (n_sep × n_flux, NaN in empty bins), ``n``
            (the injections per bin), ``sep`` and ``flux`` (the distinct
            values, or the bin centres: arithmetic, except geometric for
            flux bins whose two edges are positive), and ``threshold``.
        """
        threshold = self._threshold(stat, fap)[1]
        detected = self._detected(stat, threshold).astype(float)
        sep_idx, sep = _bin(self.separations(), sep_bins, geometric=False)
        flux_idx, flux = _bin(self.injected["flux"], flux_bins, True)
        shape = (sep.size, flux.size)
        ok = (sep_idx >= 0) & (flux_idx >= 0)
        flat = np.ravel_multi_index((sep_idx[ok], flux_idx[ok]), shape)
        size = sep.size * flux.size
        n = np.bincount(flat, minlength=size).reshape(shape)
        hits = np.bincount(flat, detected[ok], minlength=size).reshape(shape)
        with np.errstate(invalid="ignore", divide="ignore"):
            fraction = np.where(n > 0, hits / n, np.nan)
        return {
            "completeness": fraction,
            "n": n,
            "sep": sep,
            "flux": flux,
            "threshold": threshold,
        }

    def contrast_curve(
        self, stat, fap, completeness=0.5, sep_bins=None, flux_bins=None
    ):
        """Flux detected with a given completeness, against separation.

        In each separation bin of :meth:`completeness`, the detection
        fraction is made non-decreasing in flux (a running maximum) and
        interpolated linearly in log flux to where it first reaches
        ``completeness``.

        Parameters
        ----------
        stat : str
            The statistic.
        fap : float
            False-alarm probability of the detection threshold.
        completeness : float, optional
            Target detection fraction, e.g. 0.5 or 0.9.
        sep_bins, flux_bins : array-like, optional
            As for :meth:`completeness`.

        Returns
        -------
        sep, flux : numpy.ndarray
            Separations (mas) and the companion flux relative to the
            primary, the units of
            [`absil_limits`][virgil.limits.absil_limits] (convert with
            [`flux_to_contrast`][virgil.limits.flux_to_contrast] or
            [`flux_to_delta_mag`][virgil.limits.flux_to_delta_mag]). NaN
            where the positive injected fluxes do not bracket the target:
            it is never reached, or already exceeded at the lowest flux.
        """
        target = float(completeness)
        result = self.completeness(stat, fap, sep_bins, flux_bins)
        flux = result["flux"]
        out = np.full(result["sep"].size, np.nan)
        for i, row in enumerate(result["completeness"]):
            ok = (flux > 0.0) & np.isfinite(row)
            f, c = flux[ok], np.maximum.accumulate(row[ok])
            above = np.flatnonzero(c >= target)
            if above.size == 0:
                continue
            j = above[0]
            if j == 0:
                if c[0] == target:
                    out[i] = f[0]
                continue
            t = (target - c[j - 1]) / (c[j] - c[j - 1])
            out[i] = np.exp(np.log(f[j - 1]) + t * np.log(f[j] / f[j - 1]))
        return result["sep"], out

    def save(self, path):
        """Write to ``path``, an ``.npz`` with the metadata as JSON."""
        arrays = {f"null__{k}": np.asarray(v) for k, v in self.null.items()}
        for k, v in self.injected.items():
            arrays[f"injected__{k}"] = np.asarray(v)
        arrays["meta"] = np.array(json.dumps(self.meta))
        np.savez(path, **arrays)

    @classmethod
    def load(cls, path):
        """Read a file written by :meth:`save`."""
        with np.load(path, allow_pickle=False) as f:
            meta = json.loads(str(f["meta"]))
            parts = {"null": {}, "injected": {}}
            for name in f.files:
                part, _, key = name.partition("__")
                if part in parts:
                    parts[part][key] = f[name]
        return cls(meta=meta, **parts)

    @classmethod
    def concatenate(cls, results):
        """Merge results of one experiment run with different seeds.

        Parameters
        ----------
        results : sequence of DetectionMC
            E.g. one per array job. They must agree on the grid, the model,
            the null scene, the template, the noise model and
            ``match_radius``, and their seeds must all differ (one seed
            repeats the same draws).

        Returns
        -------
        DetectionMC
            All the null and injected draws, with every seed in ``meta``.

        Raises
        ------
        ValueError
            If the metadata are incompatible or a seed repeats.
        """
        results = list(results)
        if not results:
            raise ValueError("Nothing to concatenate.")
        first = results[0].meta
        unhashed = [
            k
            for k in ("model", "null_scene", "template")
            if isinstance(first.get(k), dict) and first[k]["hash"] is None
        ]
        if len(results) > 1 and unhashed:
            raise ValueError(
                f"Cannot concatenate: the {unhashed} could not be "
                "fingerprinted (they hold a lambda, a closure or another "
                "object without a reproducible hash), so the runs cannot be "
                "shown to match."
            )
        for other in results[1:]:
            bad = [k for k in _COMPATIBLE if other.meta.get(k) != first.get(k)]
            if bad:
                raise ValueError(
                    f"Cannot concatenate results that differ in {bad}."
                )
            if other.meta.get("virgil_version") != first.get("virgil_version"):
                warnings.warn(
                    "concatenate(): the results come from different virgil "
                    "versions.",
                    RuntimeWarning,
                    stacklevel=2,
                )
        seeds = [s for r in results for s in r.meta["seeds"]]
        if len({json.dumps(s, sort_keys=True) for s in seeds}) < len(seeds):
            raise ValueError(
                "Cannot concatenate results that share a seed: they repeat "
                "the same draws."
            )

        def merge(part):
            keys = getattr(results[0], part).keys()
            if any(getattr(r, part).keys() != keys for r in results):
                raise ValueError(
                    f"Cannot concatenate: the {part} fields differ."
                )
            return {
                k: np.concatenate([getattr(r, part)[k] for r in results])
                for k in keys
            }

        meta = dict(first)
        meta["seeds"] = seeds
        meta["n_null"] = sum(r.n_null for r in results)
        meta["n_injected"] = sum(r.n_injected for r in results)
        return cls(null=merge("null"), injected=merge("injected"), meta=meta)


def _round_sig(x, digits=6):
    """``x`` rounded to ``digits`` significant figures (0 stays 0)."""
    x = np.asarray(x, dtype=float)
    with np.errstate(divide="ignore"):
        mag = np.where(x == 0.0, 0.0, np.floor(np.log10(np.abs(x))))
    scale = 10.0 ** (digits - 1 - mag)
    return np.round(x * scale) / scale


def _bin(values, edges, geometric):
    """Bin indices (-1 outside) and labels: distinct values or centres."""
    if edges is None:
        labels, index = np.unique(_round_sig(values), return_inverse=True)
        return index.reshape(-1), labels
    edges = np.asarray(edges, dtype=float)
    index = np.searchsorted(edges, values, side="right") - 1
    index = np.where((index >= 0) & (index < edges.size - 1), index, -1)
    lo, hi = edges[:-1], edges[1:]
    labels = 0.5 * (lo + hi)
    if geometric:
        positive = (lo > 0.0) & (hi > 0.0)
        labels = np.where(positive, np.sqrt(np.abs(lo * hi)), labels)
    return index, labels
