"""Contrast limits, significance, and flux/contrast/Δmag conversions.

Fluxes are companion/primary; contrast is primary/companion and
``Δmag = 2.5 log10(contrast)``. See
[detection limits](../detection_limits.md) for the methods and which to use.

* `ruffio_upperlimit`: Bayesian upper limits.
* `absil_limits`: frequentist chi-squared limits.
* `injection_limits`: limits by companion injection.
* `radial_profile`: azimuthal statistics of a limit map.
"""

import warnings

import equinox as eqx
import jax
import jax.numpy as np
import jax.scipy as jsp
import numpy as onp

from ._deprecate import old_order
from ._grid import (
    batch_size_or_default,
    coordinate_points,
    first_crossing,
    map_points,
    ordered_values,
    resolve_grid_keys,
)
from .likelihood import (
    _whiten,
    build_model,
    inflated_errors,
    whitened_residuals,
)
from ._deprecate import external_stacklevel


__all__ = [
    "absil_limits",
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

# Smallest flux used in conversions, so that zero or negative limits map to a
# large but finite contrast instead of infinity or NaN.
_TINY_FLUX = 1e-30


def flux_to_contrast(flux):
    """Contrast (primary/companion) of a companion/primary flux ratio.

    >>> float(flux_to_contrast(0.01))
    100.0
    """
    return 1.0 / onp.maximum(onp.asarray(flux, dtype=float), _TINY_FLUX)


def contrast_to_flux(contrast):
    """Companion/primary flux ratio of a contrast (primary/companion)."""
    return 1.0 / onp.asarray(contrast, dtype=float)


def flux_to_delta_mag(flux):
    """Magnitude difference ``2.5 log10(primary/companion)`` of a flux ratio.

    >>> float(flux_to_delta_mag(0.01))
    5.0
    """
    return 2.5 * onp.log10(flux_to_contrast(flux))


def delta_mag_to_flux(delta_mag):
    """Companion/primary flux ratio of a magnitude difference."""
    return 10.0 ** (-0.4 * onp.asarray(delta_mag, dtype=float))


# === RADIAL PROFILES ===


def radial_profile(values, dra, ddec, center=(0.0, 0.0), r_max=None, bins=20):
    """Azimuthal statistics of a map in bins of separation.

    Parameters
    ----------
    values : array-like
        Map of shape ``(len(dra), len(ddec))`` (axis 0 is ``dra``), e.g. flux limits.
    dra, ddec : array-like
        Grid axes in mas.
    center : tuple[float, float], optional
        Centre ``(dra, ddec)`` of the annuli in mas.
    r_max : float, optional
        Outer radius in mas (default: the largest separation on the grid).
    bins : int, optional
        Number of annuli.

    Returns
    -------
    dict
        ``r`` (annulus centres, mas), ``mean``, ``std``, ``median``, ``q16``,
        ``q84`` and ``count``. Non-finite values are ignored; empty annuli are NaN.
    """
    xx, yy = onp.meshgrid(onp.asarray(dra), onp.asarray(ddec), indexing="ij")
    values = onp.asarray(values, dtype=float)
    if values.shape != xx.shape:
        raise ValueError(
            f"values has shape {values.shape}; expected "
            f"(len(dra), len(ddec)) = {xx.shape}."
        )
    rr = onp.hypot(xx - float(center[0]), yy - float(center[1]))
    if r_max is None:
        r_max = float(rr.max())
    edges = onp.linspace(0.0, float(r_max), int(bins) + 1)
    stats = {key: [] for key in ("mean", "std", "median", "q16", "q84")}
    counts = []
    for k, (low, high) in enumerate(zip(edges[:-1], edges[1:])):
        # Annuli are [low, high), except the last, which includes r_max.
        upper = rr <= high if k == len(edges) - 2 else rr < high
        inside = (rr >= low) & upper & onp.isfinite(values)
        vals = values[inside]
        counts.append(vals.size)
        if vals.size == 0:
            for key in stats:
                stats[key].append(onp.nan)
            continue
        stats["mean"].append(vals.mean())
        stats["std"].append(vals.std())
        stats["median"].append(onp.median(vals))
        stats["q16"].append(onp.percentile(vals, 16))
        stats["q84"].append(onp.percentile(vals, 84))
    return {
        "r": 0.5 * (edges[:-1] + edges[1:]),
        **{key: onp.asarray(val) for key, val in stats.items()},
        "count": onp.asarray(counts),
    }


# === SIGNIFICANCE ===


def chi2ppf(p, df):
    """Percentile function (inverse CDF) of the chi-squared distribution.

    Accurate to about 1e-14 relative in float64 for ``p`` in [1e-10, 1 - 1e-10];
    differentiable and ``jit``-able. ``p`` is clipped to ``[eps, 1 - eps]``; for
    small tail probabilities use [`nsigma`][virgil.limits.nsigma]. See
    [detection limits](../detection_limits.md#significance).

    Parameters
    ----------
    p, df : array-like
        Percentile and degrees of freedom.

    Returns
    -------
    array-like
        The chi-squared value at the percentile.
    """
    p = np.asarray(p, dtype=float)
    eps = np.finfo(p.dtype).eps
    p = np.clip(p, eps, 1.0 - eps)

    a = np.asarray(df, dtype=p.dtype) / 2.0
    x = _gammaincinv(np.where(a > 0.0, a, 1.0), p)
    return np.where(a > 0.0, 2.0 * x, np.nan)


@jax.custom_jvp
def _gammaincinv(a, p):
    """``x`` with ``gammainc(a, x) = p``, by Halley's method (implicit JVP)."""
    a, p = np.broadcast_arrays(a, p)
    q = 1.0 - p
    upper_tail = p > 0.5
    log_gamma_a = jsp.special.gammaln(a)

    def step(_, x):
        err = np.where(
            upper_tail,
            q - jsp.special.gammaincc(a, x),
            jsp.special.gammainc(a, x) - p,
        )
        log_x = np.log(x)
        density = np.exp((a - 1.0) * log_x - x - log_gamma_a)
        u = err / density
        halley = u / (1.0 - 0.5 * np.minimum(1.0, u * ((a - 1.0) / x - 1.0)))
        new = x - halley
        new = np.where(new <= 0.0, 0.5 * x, new)
        converged = (density == 0.0) | ~np.isfinite(u)
        return np.where(converged, x, new)

    # starting point: Numerical Recipes 3rd ed. 6.2.1
    tiny = np.finfo(p.dtype).tiny
    pp = np.where(p < 0.5, p, 1.0 - p)
    t = np.sqrt(-2.0 * np.log(pp))
    z = (2.30753 + t * 0.27061) / (1.0 + t * (0.99229 + t * 0.04481)) - t
    z = np.where(p < 0.5, -z, z)
    a_big = np.maximum(a, 1.0)
    wh = a_big * (1.0 - 1.0 / (9.0 * a_big) - z / (3.0 * np.sqrt(a_big))) ** 3
    wh = np.maximum(1e-3, wh)

    a_small = np.clip(a, tiny, 1.0)
    t = 1.0 - a_small * (0.253 + a_small * 0.12)
    lower = np.exp(np.log(np.maximum(p, tiny) / t) / a_small)
    upper = 1.0 - np.log1p(-np.minimum((p - t) / (1.0 - t), 1.0 - 1e-16))
    small = np.where(p < t, lower, upper)
    guess = np.where(a > 1.0, wh, small)

    return jax.lax.fori_loop(0, 20, step, guess)


# deliberate: custom_jvp rule, an implicit-function derivative
@_gammaincinv.defjvp
def _gammaincinv_jvp(primals, tangents):
    a, p = primals
    da, dp = tangents
    x = _gammaincinv(a, p)
    density = np.exp((a - 1.0) * np.log(x) - x - jsp.special.gammaln(a))
    # gammainc(a, x(a, p)) = p, so dx = (dp - ∂_a gammainc da) / density.
    _, dpda = jax.jvp(lambda a_: jsp.special.gammainc(a_, x), (a,), (da,))
    return x, (dp - dpda) / density


def nsigma(chi2r_test, chi2r_true, ndof):
    """Gaussian-equivalent significance of a reduced-chi-squared ratio.

    ``ndof * chi2r_test / chi2r_true`` is compared with a chi-squared
    distribution of ``ndof`` degrees of freedom. Accurate far beyond 8σ; it
    saturates near 13σ in float32 and 37σ in float64. See
    [detection limits](../detection_limits.md#significance).

    Source: Absil et al. (2011), [references](../references.md#absil2011).

    Parameters
    ----------
    chi2r_test, chi2r_true : float
        Reduced chi-squared of the test and true models.
    ndof : int
        Degrees of freedom.

    Returns
    -------
    float
        Detection significance in Gaussian sigma.
    """
    x = ndof * chi2r_test / chi2r_true
    half_tail = 0.5 * jax.scipy.special.gammaincc(ndof / 2.0, x / 2.0)
    # Floor at the smallest normal number, so the result saturates (about
    # 13σ in float32, 37σ in float64) instead of becoming infinite.
    half_tail = np.maximum(half_tail, np.finfo(half_tail.dtype).tiny)
    return -jax.scipy.special.ndtri(half_tail)


# === CONTRAST LIMITS ===


def ruffio_upperlimit(mean, sigma, percentile):
    """Percentile of a flux posterior truncated to non-negative values.

    The posterior is the Laplace Gaussian ``N(mean, sigma)`` truncated to
    ``flux >= 0``; ``percentile = norm.cdf(2)`` gives a 2σ-equivalent upper
    limit. See [detection limits](../detection_limits.md#ruffio).

    Source: Ruffio et al. (2018), [references](../references.md#ruffio2018).

    Parameters
    ----------
    mean : float or array-like
        Unconstrained best-fit flux (may be negative), e.g. from
        [`optimized_flux_grid`][virgil.grid_fit.optimized_flux_grid].
    sigma : float or array-like
        Laplace uncertainty of the flux, e.g. from
        [`laplace_flux_uncertainty_grid`][virgil.grid_fit.laplace_flux_uncertainty_grid].
    percentile : float or array-like
        Quantile(s) of the truncated posterior, between 0 and 1.

    Returns
    -------
    array-like
        Non-negative flux, of shape ``broadcast(mean, sigma).shape + percentile.shape``.
    """
    mean, sigma = np.broadcast_arrays(np.asarray(mean), np.asarray(sigma))
    percentile = np.asarray(percentile)
    expand = (...,) + (None,) * percentile.ndim
    mean, sigma = mean[expand], sigma[expand]

    a = -mean / sigma
    log_tail = np.log1p(-percentile) + jsp.special.log_ndtr(-a)
    z_tail = -jsp.special.ndtri(np.exp(log_tail))
    z_asymptotic = np.sqrt(a**2 - 2.0 * np.log1p(-percentile))
    z = np.where(np.isfinite(z_tail), z_tail, z_asymptotic)

    # Newton steps on log Q(z) = log_tail polish either starting point.
    def newton(z, _):
        log_q = jsp.special.log_ndtr(-z)
        log_pdf = -0.5 * z**2 - 0.5 * np.log(2.0 * np.pi)
        return z + (log_q - log_tail) * np.exp(log_q - log_pdf), None

    z, _ = jax.lax.scan(newton, z, None, length=2)
    return np.maximum(mean + sigma * z, 0.0)


@old_order("data", "model", "grid", data="data")
def absil_limits(
    model,
    data,
    grid,
    sigma,
    *,
    flux_param=None,
    flux_bounds=(1e-6, 1.0),
    batch_size=None,
):
    """Flux above which a companion is ruled out at ``sigma`` significance.

    At each grid position, the flux at which the model fits worse than the
    no-companion model by a chi-squared ratio of significance ``sigma`` (see
    [`nsigma`][virgil.limits.nsigma]); the degrees of freedom are the data
    points, with fitted parameters not subtracted. See
    [detection limits](../detection_limits.md#absil-injection).

    Source: Absil et al. (2011), [references](../references.md#absil2011).

    Parameters
    ----------
    model : SourceModel or class
        As for [`likelihood_grid`][virgil.grid_fit.likelihood_grid]; the
        no-companion model sets every parameter in ``grid`` to zero.
    data : OIData
        Data to fit.
    grid : dict[str, array-like]
        Grid axes (e.g. ``dra``/``ddec`` in mas). The flux axis matters only
        with ``flux_bounds=None``, where its smallest positive value starts the
        search.
    sigma : float
        Exclusion significance in Gaussian-equivalent σ, above that of a ratio of 1 (about
        0.67) and below the float ceiling (12.9 in float32, 37 in float64).
    flux_param : str, optional
        Key of ``grid`` holding the flux (default: the key ending in ``flux``).
    flux_bounds : tuple[float, float] or None, optional
        Search range, searched upward from its lower end; limits outside it are
        clipped with a ``RuntimeWarning``. ``None`` searches without bounds
        (e.g. for [`System`][virgil.models.System] weights above 1).
    batch_size : int, optional
        Grid points evaluated at once.

    Returns
    -------
    array-like
        Flux limit (companion/primary), one axis per coordinate key; convert
        with `flux_to_contrast` or `flux_to_delta_mag`.
    """
    return _limits(
        "absil_limits",
        _absil_limits,
        data,
        model,
        grid,
        sigma,
        flux_param,
        flux_bounds,
        batch_size,
    )


@old_order("data", "model", "grid", data="data")
def injection_limits(
    model,
    data,
    grid,
    sigma,
    *,
    flux_param=None,
    flux_bounds=(1e-6, 1.0),
    batch_size=None,
):
    """Flux at which an injected companion would be detected at ``sigma``.

    A companion of flux ``f`` is added to every observable and the no-companion
    model is fitted to the result; the limit is where it fits worse than the
    companion model by a chi-squared ratio of significance ``sigma``. Compare
    [`absil_limits`][virgil.limits.absil_limits]; the search and the differences
    from CANDID are in [detection limits](../detection_limits.md#absil-injection).

    Source: Gallenne et al. (2015), [references](../references.md#gallenne2015).

    Parameters
    ----------
    model, data, grid, flux_param, flux_bounds, batch_size
        As for [`absil_limits`][virgil.limits.absil_limits].
    sigma : float
        Detection significance, with the same bounds as for ``absil_limits``.

    Returns
    -------
    array-like
        Flux limit (companion/primary), one axis per coordinate key.

    Examples
    --------
    >>> import jax.numpy as np
    >>> from virgil import PointSource, System, UniformDisk, injection_limits
    >>> template = System(star=UniformDisk(0.8), comp=PointSource(0.01))
    >>> grid = {
    ...     "comp.dra": np.linspace(-10, 10, 21),
    ...     "comp.ddec": np.linspace(-10, 10, 21),
    ...     "comp.flux": np.array([0.01]),
    ... }
    >>> limits = injection_limits(template, data, grid, 3.0)  # doctest: +SKIP
    """
    return _limits(
        "injection_limits",
        _injection_limits,
        data,
        model,
        grid,
        sigma,
        flux_param,
        flux_bounds,
        batch_size,
    )


# The search brackets the target significance in steps of a quarter decade
# (a factor of 1.78; CANDID uses 1.4), so a significance that rises above
# sigma and falls back within one decade is not stepped over, then bisects
# the last step: 2**-22 of a quarter decade is below float32 resolution in
# log flux. An unbounded search (``flux_bounds=None``) goes at most
# _MAX_BRACKET_DECADES from its start.
_STEP_DECADES = 0.25
_BISECTIONS = 22
_MAX_BRACKET_DECADES = 40


# deliberate: test seam: used by test_grid_and_limits
def _significance_ceiling():
    """Largest significance `nsigma` represents in the default float type."""
    tiny = np.finfo(np.result_type(float)).tiny
    return float(-jax.scipy.special.ndtri(np.asarray(tiny)))


def _limits(
    caller,
    solver,
    data,
    model,
    grid,
    sigma,
    flux_param,
    flux_bounds,
    batch_size,
):
    """Validate the inputs of a limit function, solve, and report."""
    params, coord_keys, flux_key = resolve_grid_keys(grid, flux_param)
    ndof = data.n_independent
    floor = float(nsigma(1.0, 1.0, ndof))
    if not float(sigma) > floor:
        raise ValueError(
            f"sigma={sigma} cannot be reached: with {ndof} degrees of "
            f"freedom a chi-squared ratio of 1 is already {floor:.3g} sigma."
        )
    ceiling = _significance_ceiling()
    if not float(sigma) < ceiling:
        dtype = np.result_type(float).name
        raise ValueError(
            f"sigma={sigma} exceeds the largest significance {dtype} can "
            f"represent ({ceiling:.4g} sigma), where nsigma saturates. Enable "
            "float64 with jax.config.update('jax_enable_x64', True) (up to "
            "about 37 sigma), or use a lower sigma."
        )
    if flux_bounds is None:
        fluxes = onp.asarray(grid[flux_key], dtype=float)
        if not onp.any(fluxes > 0.0):
            raise ValueError(
                f"With flux_bounds=None, the flux axis {flux_key!r} needs at "
                "least one positive value to start the limit search from."
            )
        start = float(onp.min(fluxes[fluxes > 0.0]))
        max_steps = int(onp.ceil(_MAX_BRACKET_DECADES / _STEP_DECADES))
        search = (start, -onp.inf, onp.inf, max_steps)
    else:
        low, high = (float(b) for b in flux_bounds)
        if not (onp.isfinite(low) and onp.isfinite(high) and 0.0 < low < high):
            raise ValueError(
                "flux_bounds must be finite with 0 < low < high (the search "
                f"is in log flux), got {tuple(flux_bounds)}."
            )
        # Every step but the last moves a whole step.
        steps = int(onp.ceil(onp.log10(high / low) / _STEP_DECADES))
        search = (low, onp.log10(low), onp.log10(high), steps)
    start, log_low, log_high, max_steps = search
    limits, crossed = solver(
        grid,
        data,
        model,
        np.asarray(sigma, dtype=float),
        np.asarray([onp.log10(start), log_low, log_high], dtype=float),
        params=params,
        coord_keys=coord_keys,
        flux_key=flux_key,
        max_steps=max_steps,
        batch_size=batch_size_or_default(batch_size, data),
    )
    missed = int(onp.sum(~onp.asarray(crossed, dtype=bool)))
    if missed and flux_bounds is not None:
        warnings.warn(
            f"{caller}(): {missed} limits fell outside flux_bounds="
            f"{tuple(flux_bounds)} and were clipped to the nearer bound; "
            "pass wider flux_bounds, or None, to search further.",
            RuntimeWarning,
            stacklevel=external_stacklevel(),
        )
    elif missed:
        warnings.warn(
            f"{caller}(): the significance did not cross sigma={sigma} "
            f"within {_MAX_BRACKET_DECADES} decades of the starting flux "
            f"{start:g} at {missed} of {onp.size(crossed)} grid positions; "
            "the limits there are where the search stopped.",
            RuntimeWarning,
            stacklevel=external_stacklevel(),
        )
    return limits


def _solve_limits(
    significance,
    sigma,
    search,
    coord_keys,
    grid,
    max_steps,
    batch_size,
    flux_key,
    params,
):
    """Limit at every grid position, from ``significance(values)``.

    ``search`` holds the log10 of the starting flux and of the bounds.
    Returns the limits and whether each was bracketed.
    """
    coords, shape = coordinate_points(grid, coord_keys)
    log_start, log_low, log_high = search[0], search[1], search[2]

    def solve(coord_vals):
        def reached(log_flux):
            values = ordered_values(
                10.0**log_flux, coord_vals, params, coord_keys, flux_key
            )
            return significance(values) >= sigma

        log_limit, crossed = first_crossing(
            reached,
            log_start,
            log_low,
            log_high,
            max_steps,
            _BISECTIONS,
            _STEP_DECADES,
        )
        return 10.0**log_limit, crossed

    limits, crossed = map_points(solve, coords, batch_size=batch_size)
    return limits.reshape(shape), crossed.reshape(shape)


# deliberate: jit boundary
@eqx.filter_jit
def _absil_limits(
    grid,
    data,
    model,
    sigma,
    search,
    params,
    coord_keys,
    flux_key,
    max_steps,
    batch_size,
):
    """Jitted implementation of `absil_limits`.

    Returns the limits and whether each was bracketed.
    """
    ndof = data.n_independent
    coords, _ = coordinate_points(grid, coord_keys)
    null_values = ordered_values(0.0, coords[0], params, coord_keys, flux_key)

    def reduced_chi2(values):
        source = build_model(model, params, values)
        return np.sum(whitened_residuals(source, data) ** 2) / ndof

    chi2_null = reduced_chi2(null_values)

    def significance(values):
        return nsigma(reduced_chi2(values), chi2_null, ndof)

    return _solve_limits(
        significance,
        sigma,
        search,
        coord_keys,
        grid,
        max_steps,
        batch_size,
        flux_key,
        params,
    )


# deliberate: jit boundary
@eqx.filter_jit
def _injection_limits(
    grid,
    data,
    model,
    sigma,
    search,
    params,
    coord_keys,
    flux_key,
    max_steps,
    batch_size,
):
    """Jitted implementation of `injection_limits`.

    Returns the limits and whether each was bracketed.
    """
    ndof = data.n_independent
    coords, _ = coordinate_points(grid, coord_keys)
    null_values = ordered_values(0.0, coords[0], params, coord_keys, flux_key)
    null_source = build_model(model, params, null_values)
    null_prediction = data.model(null_source)
    data_vector = data.flatten_data()[0]
    errors = inflated_errors(data, null_prediction)

    def reduced_chi2(prediction, reference):
        # ``prediction`` against ``reference`` in place of the data vector,
        # each block (visibilities, phases, every extra) whitened as by
        # `whitened_residuals`.
        whitened, _ = _whiten(data, prediction, reference, errors, {}, {})
        return np.sum(whitened**2) / ndof

    def significance(values):
        # Add the companion's signal to every observable, and compare the
        # null model's fit to the result with the companion model's. With
        # whitening independent of the model the latter is the null model's
        # chi-squared on the original data; it is computed in full because
        # gains and OI_FLUX blocks whiten with the model's own prediction.
        source = build_model(model, params, values)
        prediction = data.model(source)
        injected = data_vector + (prediction - null_prediction)
        return nsigma(
            reduced_chi2(null_prediction, injected),
            reduced_chi2(prediction, injected),
            ndof,
        )

    return _solve_limits(
        significance,
        sigma,
        search,
        coord_keys,
        grid,
        max_steps,
        batch_size,
        flux_key,
        params,
    )
