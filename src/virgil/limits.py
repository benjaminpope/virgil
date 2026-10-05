"""Contrast limits, significance, and flux/contrast/Δmag conversions.

virgil parameterizes a companion by its **flux** relative to the primary
(companion/primary, so 0.01 for a companion 100 times fainter). Results are
usually reported instead as a **contrast**, primary/companion (100 here), or
as a magnitude difference ``Δmag = 2.5 log10(contrast)`` (5 mag here), which
:func:`flux_to_contrast` and :func:`flux_to_delta_mag` compute.

* :func:`ruffio_upperlimit`: Bayesian upper limits with a positive-flux prior
  (Ruffio et al. 2018).
* `absil_limits`: frequentist limits from the chi-squared ratio to the
  no-companion model (Absil et al. 2011), using :func:`nsigma`.
* `injection_limits`: limits by companion injection (Gallenne et al. 2015,
  as in CANDID): the flux at which an injected companion would be detected
  at ``sigma``.
* :func:`radial_profile`: azimuthal statistics of a limit map, for contrast
  curves.
"""

import warnings

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np

from ._utils import concrete
from ._grid import (
    batch_size_or_default,
    coordinate_points,
    map_points,
    meshgrid_vectors,
    ordered_values,
    resolve_grid_keys,
    warn_unconverged,
)
from .likelihood import build_model, whitened_residuals


__all__ = [
    "absil_limits",
    "chi2ppf",
    "delta_mag_to_flux",
    "contrast_to_flux",
    "flux_to_contrast",
    "flux_to_delta_mag",
    "injection_limits",
    "nsigma",
    "radial_profile",
    "ruffio_upperlimit",
]


# === UNITS ===

# Most decades the starting flux of `absil_limits` may be moved to bracket the
# target significance.
_MAX_BRACKET_DECADES = 40
_BISECTIONS = 14

# Smallest flux used in conversions, so that zero or negative limits map to a
# large but finite contrast instead of infinity or NaN.
_TINY_FLUX = 1e-30


def flux_to_contrast(flux):
    """Contrast (primary/companion) of a companion/primary flux ratio.

    >>> float(flux_to_contrast(0.01))
    100.0
    """
    return 1.0 / np.maximum(np.asarray(flux, dtype=float), _TINY_FLUX)


def contrast_to_flux(contrast):
    """Companion/primary flux ratio of a contrast (primary/companion)."""
    return 1.0 / np.asarray(contrast, dtype=float)


def flux_to_delta_mag(flux):
    """Magnitude difference ``2.5 log10(primary/companion)`` of a flux ratio.

    >>> float(flux_to_delta_mag(0.01))
    5.0
    """
    return 2.5 * np.log10(flux_to_contrast(flux))


def delta_mag_to_flux(delta_mag):
    """Companion/primary flux ratio of a magnitude difference."""
    return 10.0 ** (-0.4 * np.asarray(delta_mag, dtype=float))


# === RADIAL PROFILES ===


def radial_profile(values, dra, ddec, center=(0.0, 0.0), r_max=None, bins=20):
    """Azimuthal statistics of a map in bins of separation.

    Parameters
    ----------
    values : array-like
        Map with shape ``(len(dra), len(ddec))`` (axis 0 is ``dra``), as
        returned by the grid and limit functions, e.g. flux limits.
    dra, ddec : array-like
        Grid axes in milliarcseconds.
    center : tuple[float, float], optional
        Centre ``(dra, ddec)`` of the annuli in milliarcseconds.
    r_max : float, optional
        Outer radius in milliarcseconds (default: the largest separation on
        the grid).
    bins : int, optional
        Number of annuli.

    Returns
    -------
    dict
        ``r`` (annulus centres, mas), ``mean``, ``std``, ``median``,
        ``q16``, ``q84`` and ``count`` per annulus. Non-finite values are
        ignored; empty annuli are NaN.
    """
    xx, yy = np.meshgrid(np.asarray(dra), np.asarray(ddec), indexing="ij")
    values = np.asarray(values, dtype=float)
    if values.shape != xx.shape:
        raise ValueError(
            f"values has shape {values.shape}; expected "
            f"(len(dra), len(ddec)) = {xx.shape}."
        )
    rr = np.hypot(xx - float(center[0]), yy - float(center[1]))
    if r_max is None:
        r_max = float(rr.max())
    edges = np.linspace(0.0, float(r_max), int(bins) + 1)
    stats = {key: [] for key in ("mean", "std", "median", "q16", "q84")}
    counts = []
    for k, (low, high) in enumerate(zip(edges[:-1], edges[1:])):
        # Annuli are [low, high), except the last, which includes r_max.
        upper = rr <= high if k == len(edges) - 2 else rr < high
        inside = (rr >= low) & upper & np.isfinite(values)
        vals = values[inside]
        counts.append(vals.size)
        if vals.size == 0:
            for key in stats:
                stats[key].append(np.nan)
            continue
        stats["mean"].append(vals.mean())
        stats["std"].append(vals.std())
        stats["median"].append(np.median(vals))
        stats["q16"].append(np.percentile(vals, 16))
        stats["q84"].append(np.percentile(vals, 84))
    return {
        "r": 0.5 * (edges[:-1] + edges[1:]),
        **{key: np.asarray(val) for key, val in stats.items()},
        "count": np.asarray(counts),
    }


# === SIGNIFICANCE ===


def chi2ppf(p, df):
    """
    Percentile function for chi-square.

    For ``df=1`` (the path used in ``nsigma``), use the closed-form identity
    based on the standard normal quantile, i.e. square ``norm.ppf((p+1)/2)``.
    This remains JAX-native, differentiable, and fast.

    For ``df != 1``, this falls back to numpyro's gammaincinv backend.

    Parameters
    ----------
    p : array-like
        Percentile value.
    df : array-like
        Degrees of freedom.

    Returns
    -------
    array-like
        Corresponding chi2 value to the percentile.

    Notes
    -----
    ``p`` is clipped to ``[eps, 1 - eps]`` of its own floating-point type, so
    the result stays finite. Near ``p = 1`` this loses precision; to convert
    small tail probabilities, use [`nsigma`][virgil.limits.nsigma], which works with the upper
    tail directly.
    """
    p = jnp.asarray(p, dtype=float)
    eps = jnp.finfo(p.dtype).eps
    p = jnp.clip(p, eps, 1.0 - eps)

    df_value = concrete(df)
    if df_value is not None and df_value.size == 1 and float(df_value) == 1.0:
        z = jax.scipy.stats.norm.ppf((p + 1.0) / 2.0)
        return z**2

    from numpyro.distributions.util import gammaincinv

    return jnp.asarray(gammaincinv(df / 2.0, p), dtype=float) * 2.0


def nsigma(chi2r_test, chi2r_true, ndof):
    """
    Convert a reduced-chi-squared ratio to a Gaussian-equivalent significance.

    The statistic ``x = ndof * chi2r_test / chi2r_true`` is compared with a
    chi-squared distribution of ``ndof`` degrees of freedom, and its
    upper-tail probability is expressed as the equivalent two-sided Gaussian
    significance (as in Absil et al. 2011).

    Parameters
    ----------
    chi2r_test: float
        Reduced chi-squared of test model.
    chi2r_true: float
        Reduced chi-squared of true model.
    ndof: int
        Number of degrees of freedom.

    Returns
    -------
    nsigma: float
        Detection significance in Gaussian sigma.

    Notes
    -----
    The upper tail is computed directly (with the regularized incomplete
    gamma function) rather than as ``1 - cdf``, so significances stay finite
    and accurate far beyond 8σ, including in float32 (up to about 13σ).
    """
    x = ndof * chi2r_test / chi2r_true
    half_tail = 0.5 * jax.scipy.special.gammaincc(ndof / 2.0, x / 2.0)
    # Floor at the smallest normal number, so the result saturates (about
    # 13σ in float32, 37σ in float64) instead of becoming infinite.
    half_tail = jnp.maximum(half_tail, jnp.finfo(half_tail.dtype).tiny)
    return -jax.scipy.special.ndtri(half_tail)


# === CONTRAST LIMITS ===


def ruffio_upperlimit(mean, sigma, percentile):
    """
    Percentile of a flux posterior truncated to non-negative values.

    Following Ruffio et al. (2018, eqn 8), the flux posterior at each
    position is the Laplace Gaussian ``N(mean, sigma)`` of the
    unconstrained (possibly negative) best-fit flux, truncated to
    ``flux >= 0`` by the positivity prior. This returns its ``percentile``
    quantile, so ``percentile = norm.cdf(2)`` gives a 2-sigma-equivalent
    upper limit and ``0.16, 0.5, 0.84`` give a median and credible interval.

    Parameters
    ----------
    mean : float or array-like
        Unconstrained best-fit flux, e.g. from [`optimized_flux_grid`][virgil.grid_fit.optimized_flux_grid].
        It may be negative.
    sigma : float or array-like
        Laplace uncertainty of the flux, e.g. from
        [`laplace_flux_uncertainty_grid`][virgil.grid_fit.laplace_flux_uncertainty_grid], broadcastable to ``mean``.
    percentile : float or array-like
        Quantile(s) of the truncated posterior to return, between 0 and 1.

    Returns
    -------
    array-like
        Non-negative flux at each percentile, with shape
        ``broadcast(mean, sigma).shape + percentile.shape``.

    Notes
    -----
    The flat ``flux >= 0`` prior is the published convention (Ruffio et al.
    2018) and is deliberately not the Jeffreys prior for a scale: a
    log-uniform prior on the flux makes the posterior improper as
    ``flux -> 0``, so there would be no finite upper limit to quote.

    The quantile is computed from the upper tail,
    ``mean + sigma * z`` with ``Q(z) = (1 - percentile) Q(-mean / sigma)``
    and ``Q`` the standard normal survival function, evaluated in log space
    so that it stays accurate when the best fit is many sigma below zero.
    Where the tail underflows, the large-deviation limit
    ``z = sqrt(a**2 - 2 log(1 - percentile))`` (``a = -mean / sigma``) is the
    starting point, and two Newton steps on ``log Q(z)`` polish the result.
    """
    mean, sigma = jnp.broadcast_arrays(jnp.asarray(mean), jnp.asarray(sigma))
    percentile = jnp.asarray(percentile)
    expand = (...,) + (None,) * percentile.ndim
    mean, sigma = mean[expand], sigma[expand]

    a = -mean / sigma
    log_tail = jnp.log1p(-percentile) + jsp.special.log_ndtr(-a)
    z_tail = -jsp.special.ndtri(jnp.exp(log_tail))
    z_asymptotic = jnp.sqrt(a**2 - 2.0 * jnp.log1p(-percentile))
    z = jnp.where(jnp.isfinite(z_tail), z_tail, z_asymptotic)

    # Newton steps on log Q(z) = log_tail polish either starting point.
    def newton(z, _):
        log_q = jsp.special.log_ndtr(-z)
        log_pdf = -0.5 * z**2 - 0.5 * jnp.log(2.0 * jnp.pi)
        return z + (log_q - log_tail) * jnp.exp(log_q - log_pdf), None

    z, _ = jax.lax.scan(newton, z, None, length=2)
    return jnp.maximum(mean + sigma * z, 0.0)


def absil_limits(
    data_obj,
    model,
    samples_dict,
    sigma,
    flux_param=None,
    flux_bounds=(1e-6, 1.0),
    batch_size=None,
):
    """Flux above which a companion is ruled out at ``sigma`` significance.

    Following Absil et al. (2011), at each grid position this finds the flux
    at which the model fits the data worse than the no-companion model by a
    chi-squared ratio corresponding to ``sigma`` (see
    [nsigma][virgil.limits.nsigma]). Brighter companions at that
    position are excluded at ``sigma``: with ``sigma=3`` the result is a
    3-sigma upper limit on the flux.

    Parameters
    ----------
    data_obj : OIData
        Data to fit.
    model : SourceModel or class
        Template model or model class, as for
        [`likelihood_grid`][virgil.grid_fit.likelihood_grid]. The
        no-companion model sets every parameter in ``samples_dict`` to zero.
    samples_dict : dict[str, array-like]
        Grid axes, as a mapping from parameter name or path to 1D values
        (e.g. ``dra``/``ddec`` in milliarcseconds). The flux axis is only
        used for the starting guess (a single value is enough, and the limit
        may lie decades away from it), and must contain at least one
        positive value.
    sigma : float
        Exclusion significance. It must exceed the significance of a
        chi-squared ratio of 1 (about 0.67 for many degrees of freedom).
    flux_param : str, optional
        The key of ``samples_dict`` holding the flux optimized at each grid
        position. By default, the one key whose last part is ``flux``.
    flux_bounds : tuple[float, float] or None, optional
        Limits are clipped to this range (default ``(1e-6, 1.0)``), and a
        ``RuntimeWarning`` reports how many were clipped. Pass ``None`` to
        return them unclipped, e.g. for [`System`][virgil.models.System]
        weights that may exceed 1.
    batch_size : int, optional
        Number of grid points evaluated at once. By default, enough for
        about 2**20 model visibilities on a CPU and 2**23 on other backends
        (GPU, TPU), and at least 256. Larger can be faster for small data;
        smaller bounds memory for large models.

    Returns
    -------
    array-like
        Flux limit (companion/primary), with one axis per coordinate key;
        see :func:`flux_to_contrast` and :func:`flux_to_delta_mag`.

    Notes
    -----
    The number of degrees of freedom is the number of data points; the
    fitted parameters are not subtracted.
    """
    params, coord_keys, flux_key = resolve_grid_keys(samples_dict, flux_param)
    if not np.any(np.asarray(samples_dict[flux_key]) > 0.0):
        raise ValueError(
            f"The flux axis {flux_key!r} needs at least one positive value "
            "to start the limit search from."
        )
    ndof = data_obj.n_independent
    floor = float(nsigma(1.0, 1.0, ndof))
    if not float(sigma) > floor:
        raise ValueError(
            f"sigma={sigma} cannot be reached: with {ndof} degrees of "
            f"freedom a chi-squared ratio of 1 is already {floor:.3g} sigma."
        )
    limits, success = _absil_limits(
        samples_dict,
        data_obj,
        model,
        jnp.asarray(sigma, dtype=float),
        params=params,
        coord_keys=coord_keys,
        flux_key=flux_key,
        batch_size=batch_size_or_default(batch_size, data_obj),
    )
    warn_unconverged(success, "absil_limits")
    if flux_bounds is None:
        return limits
    low, high = flux_bounds
    clipped = int(
        np.sum((np.asarray(limits) < low) | (np.asarray(limits) > high))
    )
    if clipped:
        warnings.warn(
            f"absil_limits(): {clipped} limits fell outside flux_bounds="
            f"{tuple(flux_bounds)} and were clipped; pass flux_bounds=None "
            "to keep them.",
            RuntimeWarning,
            stacklevel=2,
        )
    return jnp.clip(limits, low, high)


@eqx.filter_jit
def _absil_limits(
    samples_dict,
    data_obj,
    model,
    sigma,
    params,
    coord_keys,
    flux_key,
    batch_size,
):
    """Jitted implementation of `absil_limits`.

    Returns the unclipped limits and whether each reaches ``sigma``.
    """
    ndof = data_obj.n_independent

    def reduced_chi2(values):
        source = build_model(model, params, values)
        return jnp.sum(whitened_residuals(source, data_obj) ** 2) / ndof

    null_values = [0.0] * len(params)
    chi2_null = reduced_chi2(null_values)

    def loss(values):
        significance = nsigma(reduced_chi2(values), chi2_null, ndof)
        return (significance - sigma) ** 2

    vals_vec, grid_shape = meshgrid_vectors(samples_dict, params)
    loss_grid = map_points(loss, vals_vec, batch_size=batch_size).reshape(
        grid_shape
    )
    flux_axis = params.index(flux_key)
    best_flux_indices = jnp.nanargmin(loss_grid, axis=flux_axis)

    coords, shape = coordinate_points(samples_dict, coord_keys)
    # A zero flux would start the log-flux search at -inf; start from the
    # smallest positive flux on the grid instead.
    flux_axis_vals = jnp.asarray(samples_dict[flux_key])
    smallest_positive = jnp.min(
        jnp.where(flux_axis_vals > 0.0, flux_axis_vals, jnp.inf)
    )
    start_flux = flux_axis_vals[best_flux_indices].reshape(-1)
    start_flux = jnp.where(start_flux > 0.0, start_flux, smallest_positive)

    def significance_at(log_flux, coord_vals):
        flux = 10.0**log_flux
        values = ordered_values(flux, coord_vals, params, coord_keys, flux_key)
        return nsigma(reduced_chi2(values), chi2_null, ndof)

    def bracketed_limit(flux0, coord_vals):
        # The significance saturates (about 37 sigma in float64) for bright
        # companions, where the loss is flat and BFGS cannot move. Step the
        # starting flux by decades until it crosses the target, then bisect
        # that decade in log flux. The axis thus only has to give a rough
        # starting point, and no gradient is needed. Returns log10 of the
        # limit.
        log0 = jnp.log10(flux0)
        above = significance_at(log0, coord_vals) > sigma
        direction = jnp.where(above, -1.0, 1.0)

        def keep_going(state):
            log_flux, n = state
            same_side = (
                significance_at(log_flux, coord_vals) > sigma
            ) == above
            return same_side & (n < _MAX_BRACKET_DECADES)

        def step(state):
            log_flux, n = state
            return log_flux + direction, n + 1

        log_end, _ = jax.lax.while_loop(keep_going, step, (log0, 0))
        # [log_end, log_end - direction] brackets the target.

        def bisect(_, edges):
            near, far = edges
            mid = 0.5 * (near + far)
            same_side = (significance_at(mid, coord_vals) > sigma) == above
            return jnp.where(same_side, mid, near), jnp.where(
                same_side, far, mid
            )

        # `near` is on the starting side of the target; `far` on the other.
        near, far = jax.lax.fori_loop(
            0, _BISECTIONS, bisect, (log_end - direction, log_end)
        )
        return 0.5 * (near + far)

    def best_flux(flux0, coord_vals):
        limit = 10.0 ** bracketed_limit(flux0, coord_vals)
        # Converged if the target significance is reached to 0.01 sigma.
        reached = jnp.abs(
            significance_at(jnp.log10(limit), coord_vals) - sigma
        )
        return limit, reached < 1e-2

    limits, success = map_points(
        best_flux, start_flux, coords, batch_size=batch_size
    )
    return limits.reshape(shape), success.reshape(shape)


# Bisection steps in log flux: the default bracket (1e-6 to 1) spans 6
# decades, so 40 steps resolve the limit to about 1e-11 decades.
_BISECTION_STEPS = 40
# Search range used when ``flux_bounds=None``: wide enough for System weights.
_UNBOUNDED_BRACKET = (1e-8, 1e3)


def injection_limits(
    data_obj,
    model,
    samples_dict,
    sigma,
    flux_param=None,
    flux_bounds=(1e-6, 1.0),
    batch_size=None,
):
    """Flux at which an injected companion would be detected at ``sigma``.

    This is the injection method of Gallenne et al. (2015, section 3.2),
    as in CANDID's ``detectionLimit(methods=["injection"])``. At each grid
    position a companion of flux ``f`` is added to the data (to every
    observable: the observables of the model with the companion minus those
    of the no-companion model), and the no-companion model is fitted to the
    result. The limit is the flux at which the null model fits the injected
    data worse than the companion model does, by a chi-squared ratio
    corresponding to ``sigma`` (see [nsigma][virgil.limits.nsigma]). The
    companion model fits the injected data exactly as the null model fits
    the original data, so the ratio is
    ``chi2(data + signal(f)) / chi2(data)`` for the null model.

    Compare [`absil_limits`][virgil.limits.absil_limits], which uses
    ``chi2(data - signal(f)) / chi2(data)``. The two differ by the sign of
    the cross term between the data's residuals and the signal.

    Parameters
    ----------
    data_obj : OIData
        Data to fit.
    model : SourceModel or class
        Template model or model class, as for
        [`likelihood_grid`][virgil.grid_fit.likelihood_grid]. The
        no-companion model sets every parameter in ``samples_dict`` to zero.
    samples_dict : dict[str, array-like]
        Grid axes, as a mapping from parameter name or path to 1D values
        (e.g. ``dra``/``ddec`` in milliarcseconds). The values on the flux
        axis are not used, because the flux is solved for at each position.
    sigma : float
        Detection significance. It must exceed the significance of a
        chi-squared ratio of 1 (about 0.67 for many degrees of freedom).
    flux_param : str, optional
        The key of ``samples_dict`` holding the flux that is solved for.
        By default, the one key whose last part is ``flux``.
    flux_bounds : tuple[float, float] or None, optional
        The flux is searched in this range (default ``(1e-6, 1.0)``), and
        limits outside it are clipped to it, with a ``RuntimeWarning``
        reporting how many. Pass ``None`` to search 1e-8 to 1e3 instead,
        e.g. for [`System`][virgil.models.System] weights that may exceed 1.
    batch_size : int, optional
        Number of grid points evaluated at once; see
        [`absil_limits`][virgil.limits.absil_limits].

    Returns
    -------
    array-like
        Flux limit (companion/primary), with one axis per coordinate key;
        see :func:`flux_to_contrast` and :func:`flux_to_delta_mag`.

    Notes
    -----
    The flux is found by bisection in log flux. The significance rises with
    flux once the signal exceeds the noise, but the cross term can make it
    dip at very faint fluxes; bisection then returns one of the crossings.

    Differences from CANDID: CANDID refits the primary's diameter to the
    injected data before computing the null chi-squared (for V² and T3),
    which virgil does not; the null model is exactly the one with every
    grid parameter set to zero. CANDID also brackets the limit in steps of
    1.4 in flux and interpolates linearly in significance, where virgil
    solves the criterion by bisection. The chi-squared and the number of
    degrees of freedom are virgil's, as for ``absil_limits``.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from virgil import PointSource, System, UniformDisk, injection_limits
    >>> template = System(star=UniformDisk(0.8), comp=PointSource(0.01))
    >>> samples = {
    ...     "comp.dra": jnp.linspace(-10, 10, 21),
    ...     "comp.ddec": jnp.linspace(-10, 10, 21),
    ...     "comp.flux": jnp.array([0.01]),
    ... }
    >>> limits = injection_limits(data, template, samples, 3.0)  # doctest: +SKIP
    >>> limits.shape  # doctest: +SKIP
    (21, 21)
    """
    params, coord_keys, flux_key = resolve_grid_keys(samples_dict, flux_param)
    ndof = data_obj.n_independent
    floor = float(nsigma(1.0, 1.0, ndof))
    if not float(sigma) > floor:
        raise ValueError(
            f"sigma={sigma} cannot be reached: with {ndof} degrees of "
            f"freedom a chi-squared ratio of 1 is already {floor:.3g} sigma."
        )
    bracket = _UNBOUNDED_BRACKET if flux_bounds is None else flux_bounds
    low, high = (float(b) for b in bracket)
    if not (np.isfinite(low) and np.isfinite(high) and 0.0 < low < high):
        raise ValueError(
            "flux_bounds must be finite with 0 < low < high (the search is "
            f"in log flux), got {tuple(bracket)}."
        )
    limits, outside = _injection_limits(
        samples_dict,
        data_obj,
        model,
        jnp.asarray(sigma, dtype=float),
        jnp.asarray([low, high], dtype=float),
        params=params,
        coord_keys=coord_keys,
        flux_key=flux_key,
        batch_size=batch_size_or_default(batch_size, data_obj),
    )
    if flux_bounds is not None:
        clipped = int(np.sum(np.asarray(outside)))
        if clipped:
            warnings.warn(
                f"injection_limits(): {clipped} limits fell outside "
                f"flux_bounds={tuple(flux_bounds)} and were clipped; pass "
                "flux_bounds=None to search a wider range.",
                RuntimeWarning,
                stacklevel=2,
            )
    return limits


@eqx.filter_jit
def _injection_limits(
    samples_dict,
    data_obj,
    model,
    sigma,
    bracket,
    params,
    coord_keys,
    flux_key,
    batch_size,
):
    """Jitted implementation of `injection_limits`.

    Returns the limits and whether each sits at an end of the bracket
    without reaching ``sigma`` inside it.
    """
    ndof = data_obj.n_independent
    n_vis = data_obj.vis.shape[0]

    null_source = build_model(model, params, [0.0] * len(params))
    null_prediction = data_obj.model(null_source)
    chi2_null = jnp.sum(whitened_residuals(null_source, data_obj) ** 2) / ndof

    def significance(values):
        # Add the companion's signal to the data, and fit the null model.
        source = build_model(model, params, values)
        signal = data_obj.model(source) - null_prediction
        injected = eqx.tree_at(
            lambda d: (d.vis, d.phi),
            data_obj,
            (data_obj.vis + signal[:n_vis], data_obj.phi + signal[n_vis:]),
        )
        chi2_injected = (
            jnp.sum(whitened_residuals(null_source, injected) ** 2) / ndof
        )
        # The companion model fits the injected data as the null model fits
        # the original data, with chi-squared chi2_null.
        return nsigma(chi2_injected, chi2_null, ndof)

    coords, shape = coordinate_points(samples_dict, coord_keys)
    log_low, log_high = jnp.log10(bracket[0]), jnp.log10(bracket[1])

    def excess(log_flux, coord_vals):
        values = ordered_values(
            10.0**log_flux, coord_vals, params, coord_keys, flux_key
        )
        return significance(values) - sigma

    def solve(coord_vals):
        def step(_, state):
            lo, hi = state
            mid = 0.5 * (lo + hi)
            reached = excess(mid, coord_vals) >= 0.0
            return jnp.where(reached, lo, mid), jnp.where(reached, mid, hi)

        lo, hi = jax.lax.fori_loop(
            0, _BISECTION_STEPS, step, (log_low, log_high)
        )
        below = excess(log_low, coord_vals) >= 0.0
        above = excess(log_high, coord_vals) < 0.0
        log_limit = jnp.where(
            below, log_low, jnp.where(above, log_high, 0.5 * (lo + hi))
        )
        return 10.0**log_limit, below | above

    limits, outside = map_points(solve, coords, batch_size=batch_size)
    return limits.reshape(shape), outside.reshape(shape)
