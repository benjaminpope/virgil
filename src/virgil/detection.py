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
design, and the simulation tools still to come, are described in
``design/detection_roc.md``.

[`local_nsigma`][virgil.detection.local_nsigma] converts ``delta_chi2`` to
Wilks's Gaussian-equivalent significance at a single position. It ignores
the look-elsewhere effect of searching a grid, so it overstates the
significance of the best of many positions; calibrate with simulations.
"""

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
from .likelihood import loglike
from .limits import nsigma


__all__ = ["detection_statistics", "local_nsigma"]

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
    params, coord_keys, flux_key = resolve_grid_keys(samples_dict, flux_param)
    clash = set(params) & (set(STATISTICS) | {"flux_peak_steps"})
    if clash:
        raise ValueError(
            f"samples_dict keys {sorted(clash)} clash with the names of the "
            "returned statistics."
        )
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

    # Profile likelihood with flux >= 0. The refined optimum is used where
    # it improves on the best grid point (it may fail, e.g. to NaN); where
    # the unconstrained best flux is negative, the constrained optimum is
    # flux 0, which is the null itself.
    use_opt = opt_loglike >= grid_loglike
    alt_loglike = jnp.where(use_opt, opt_loglike, grid_loglike)
    alt_flux = jnp.where(use_opt, opt_flux, grid_flux)
    sign = jnp.where(jnp.isnan(opt_flux), alt_flux, opt_flux)
    positive = (sign > 0.0) & (alt_loglike > loglike0)
    profile = jnp.where(positive, alt_loglike, loglike0)
    best = jnp.argmax(profile.reshape(-1))
    delta_chi2 = 2.0 * (profile.reshape(-1)[best] - loglike0)
    best_flux = jnp.where(positive, alt_flux, 0.0).reshape(-1)[best]

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
