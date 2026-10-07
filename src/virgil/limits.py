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

    It is ``2·gammaincinv(df/2, p)``, inverting
    ``jax.scipy.special.gammainc`` by Halley's method from a
    Wilson–Hilferty start, with the residual taken in the upper tail
    (``gammaincc``) for ``p > 1/2``. It is accurate to about 1e-14
    relative in float64 for every ``df`` and ``p`` in [1e-10, 1 - 1e-10],
    needs no optional dependency, and is differentiable (by the implicit
    function theorem) and ``jit``-able in both arguments.

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

    a = jnp.asarray(df, dtype=p.dtype) / 2.0
    return 2.0 * _gammaincinv(a, p)


def _gammaincinv_guess(a, p):
    """Starting point for :func:`_gammaincinv` (Numerical Recipes, 3rd ed.,
    §6.2.1): Wilson–Hilferty for ``a > 1``, a power law in the lower tail
    and an exponential in the upper tail otherwise."""
    tiny = jnp.finfo(p.dtype).tiny
    pp = jnp.where(p < 0.5, p, 1.0 - p)
    t = jnp.sqrt(-2.0 * jnp.log(pp))
    z = (2.30753 + t * 0.27061) / (1.0 + t * (0.99229 + t * 0.04481)) - t
    z = jnp.where(p < 0.5, -z, z)
    a_big = jnp.maximum(a, 1.0)
    wh = a_big * (1.0 - 1.0 / (9.0 * a_big) - z / (3.0 * jnp.sqrt(a_big))) ** 3
    wh = jnp.maximum(1e-3, wh)

    a_small = jnp.clip(a, tiny, 1.0)
    t = 1.0 - a_small * (0.253 + a_small * 0.12)
    lower = jnp.exp(jnp.log(jnp.maximum(p, tiny) / t) / a_small)
    upper = 1.0 - jnp.log1p(-jnp.minimum((p - t) / (1.0 - t), 1.0 - 1e-16))
    small = jnp.where(p < t, lower, upper)
    return jnp.where(a > 1.0, wh, small)


@jax.custom_jvp
def _gammaincinv(a, p):
    """``x`` with ``gammainc(a, x) = p``, by Halley's method.

    Residuals use the upper tail ``gammaincc(a, x) = 1 - p`` when
    ``p > 1/2``, so the result keeps full relative precision at both ends.
    The derivative is from the implicit function theorem, so this is
    differentiable in ``a`` and ``p`` and does not unroll the iterations.
    """
    a, p = jnp.broadcast_arrays(a, p)
    q = 1.0 - p
    upper_tail = p > 0.5
    log_gamma_a = jsp.special.gammaln(a)

    def step(_, x):
        err = jnp.where(
            upper_tail,
            q - jsp.special.gammaincc(a, x),
            jsp.special.gammainc(a, x) - p,
        )
        log_x = jnp.log(x)
        density = jnp.exp((a - 1.0) * log_x - x - log_gamma_a)
        u = err / density
        halley = u / (1.0 - 0.5 * jnp.minimum(1.0, u * ((a - 1.0) / x - 1.0)))
        new = x - halley
        new = jnp.where(new <= 0.0, 0.5 * x, new)
        converged = (density == 0.0) | ~jnp.isfinite(u)
        return jnp.where(converged, x, new)

    return jax.lax.fori_loop(0, 20, step, _gammaincinv_guess(a, p))


@_gammaincinv.defjvp
def _gammaincinv_jvp(primals, tangents):
    a, p = primals
    da, dp = tangents
    x = _gammaincinv(a, p)
    density = jnp.exp((a - 1.0) * jnp.log(x) - x - jsp.special.gammaln(a))
    # gammainc(a, x(a, p)) = p, so dx = (dp - ∂_a gammainc da) / density.
    _, dpda = jax.jvp(lambda a_: jsp.special.gammainc(a_, x), (a,), (da,))
    return x, (dp - dpda) / density


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

    Following Absil et al. (2011), at each grid position this finds the flux
    at which the model fits the data worse than the no-companion model by a
    chi-squared ratio corresponding to ``sigma`` (see
    [nsigma][virgil.limits.nsigma]). Brighter companions at that
    position are excluded at ``sigma``: with ``sigma=3`` the result is a
    3-sigma upper limit on the flux.

    Parameters
    ----------
    model : SourceModel or class
        Template model or model class, as for
        [`likelihood_grid`][virgil.grid_fit.likelihood_grid]. The
        no-companion model sets every parameter in ``grid`` to zero.
    data : OIData
        Data to fit.
    grid : dict[str, array-like]
        Grid axes, as a mapping from parameter name or path to 1D values
        (e.g. ``dra``/``ddec`` in milliarcseconds). The flux axis is only
        used with ``flux_bounds=None``, where its smallest positive value
        starts the search (a single value is enough).
    sigma : float
        Exclusion significance. It must exceed the significance of a
        chi-squared ratio of 1 (about 0.67 for many degrees of freedom),
        and be below the largest significance the floating-point type can
        represent (about 12.9 in float32, 37 in float64).
    flux_param : str, optional
        The key of ``grid`` holding the flux solved for at each grid
        position. By default, the one key whose last part is ``flux``.
    flux_bounds : tuple[float, float] or None, optional
        Search range of the flux (default ``(1e-6, 1.0)``), searched upward
        from its lower end, as in
        [`injection_limits`][virgil.limits.injection_limits]. Limits outside
        it are returned at the nearer bound, and a ``RuntimeWarning`` reports
        how many. Pass ``None`` to search without bounds, from the smallest
        positive value of the flux axis, e.g. for
        [`System`][virgil.models.System] weights that may exceed 1.
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

    The limit is the first flux, going up from the start of the search, at
    which the significance reaches ``sigma``: the flux is stepped by
    decades until it does, and that decade is bisected in log flux. For a
    normalised scene the significance falls again once the companion
    outshines the primary (flux well above 1), so an unbounded search
    should start below that.
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

    This is the injection method of Gallenne et al. (2015, section 3.2),
    as in CANDID's ``detectionLimit(methods=["injection"])``. At each grid
    position a companion of flux ``f`` is added to the data (to every
    observable, including the extras: the observables of the model with the
    companion minus those of the no-companion model), and the no-companion
    model is fitted to the result. The limit is the flux at which the null
    model fits the injected data worse than the companion model does, by a
    chi-squared ratio corresponding to ``sigma`` (see
    [nsigma][virgil.limits.nsigma]). The companion model fits the injected
    data exactly as the null model fits the original data, so the ratio is
    ``chi2(data + signal(f)) / chi2(data)`` for the null model. (With gains
    or OI_FLUX data, whose whitening depends on the model, the companion
    model's chi-squared on the injected data is computed in full.)

    Compare [`absil_limits`][virgil.limits.absil_limits], which uses
    ``chi2(data - signal(f)) / chi2(data)``. The two differ by the sign of
    the cross term between the data's residuals and the signal.

    Parameters
    ----------
    model : SourceModel or class
        Template model or model class, as for
        [`likelihood_grid`][virgil.grid_fit.likelihood_grid]. The
        no-companion model sets every parameter in ``grid`` to zero.
    data : OIData
        Data to fit.
    grid : dict[str, array-like]
        Grid axes, as a mapping from parameter name or path to 1D values
        (e.g. ``dra``/``ddec`` in milliarcseconds). The flux axis is only
        used with ``flux_bounds=None``, where its smallest positive value
        starts the search (a single value is enough).
    sigma : float
        Detection significance; see
        [`absil_limits`][virgil.limits.absil_limits] for its range.
    flux_param : str, optional
        The key of ``grid`` holding the flux that is solved for.
        By default, the one key whose last part is ``flux``.
    flux_bounds : tuple[float, float] or None, optional
        Search range of the flux (default ``(1e-6, 1.0)``), searched upward
        from its lower end. Limits outside it are returned at the nearer
        bound, and a ``RuntimeWarning`` reports how many. Pass ``None`` to
        search without bounds, from the smallest positive value of the flux
        axis, e.g. for [`System`][virgil.models.System] weights that may
        exceed 1.
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
    The limit is the first flux, going up from the start of the search, at
    which the significance reaches ``sigma``, found as in ``absil_limits``.
    The significance rises with flux once the signal exceeds the noise, but
    the cross term can make it dip at very faint fluxes, and for a
    normalised scene it falls again once the companion outshines the
    primary (flux well above 1), so an unbounded search should start below
    that.

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
    >>> grid = {
    ...     "comp.dra": jnp.linspace(-10, 10, 21),
    ...     "comp.ddec": jnp.linspace(-10, 10, 21),
    ...     "comp.flux": jnp.array([0.01]),
    ... }
    >>> limits = injection_limits(template, data, grid, 3.0)  # doctest: +SKIP
    >>> limits.shape  # doctest: +SKIP
    (21, 21)
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


def _significance_ceiling():
    """Largest significance `nsigma` represents in the default float type."""
    tiny = jnp.finfo(jnp.result_type(float)).tiny
    return float(-jax.scipy.special.ndtri(jnp.asarray(tiny)))


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
        dtype = jnp.result_type(float).name
        raise ValueError(
            f"sigma={sigma} exceeds the largest significance {dtype} can "
            f"represent ({ceiling:.4g} sigma), where nsigma saturates. Enable "
            "float64 with jax.config.update('jax_enable_x64', True) (up to "
            "about 37 sigma), or use a lower sigma."
        )
    if flux_bounds is None:
        fluxes = np.asarray(grid[flux_key], dtype=float)
        if not np.any(fluxes > 0.0):
            raise ValueError(
                f"With flux_bounds=None, the flux axis {flux_key!r} needs at "
                "least one positive value to start the limit search from."
            )
        start = float(np.min(fluxes[fluxes > 0.0]))
        max_steps = int(np.ceil(_MAX_BRACKET_DECADES / _STEP_DECADES))
        search = (start, -np.inf, np.inf, max_steps)
    else:
        low, high = (float(b) for b in flux_bounds)
        if not (np.isfinite(low) and np.isfinite(high) and 0.0 < low < high):
            raise ValueError(
                "flux_bounds must be finite with 0 < low < high (the search "
                f"is in log flux), got {tuple(flux_bounds)}."
            )
        # Every step but the last moves a whole step.
        steps = int(np.ceil(np.log10(high / low) / _STEP_DECADES))
        search = (low, np.log10(low), np.log10(high), steps)
    start, log_low, log_high, max_steps = search
    limits, crossed = solver(
        grid,
        data,
        model,
        jnp.asarray(sigma, dtype=float),
        jnp.asarray([np.log10(start), log_low, log_high], dtype=float),
        params=params,
        coord_keys=coord_keys,
        flux_key=flux_key,
        max_steps=max_steps,
        batch_size=batch_size_or_default(batch_size, data),
    )
    missed = int(np.sum(~np.asarray(crossed, dtype=bool)))
    if missed and flux_bounds is not None:
        warnings.warn(
            f"{caller}(): {missed} limits fell outside flux_bounds="
            f"{tuple(flux_bounds)} and were clipped to the nearer bound; "
            "pass wider flux_bounds, or None, to search further.",
            RuntimeWarning,
            stacklevel=3,
        )
    elif missed:
        warnings.warn(
            f"{caller}(): the significance did not cross sigma={sigma} "
            f"within {_MAX_BRACKET_DECADES} decades of the starting flux "
            f"{start:g} at {missed} of {np.size(crossed)} grid positions; "
            "the limits there are where the search stopped.",
            RuntimeWarning,
            stacklevel=3,
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

    def reduced_chi2(values):
        source = build_model(model, params, values)
        return jnp.sum(whitened_residuals(source, data) ** 2) / ndof

    chi2_null = reduced_chi2([0.0] * len(params))

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
    null_source = build_model(model, params, [0.0] * len(params))
    null_prediction = data.model(null_source)
    data_vector = data.flatten_data()[0]
    errors = inflated_errors(data, null_prediction)

    def reduced_chi2(prediction, reference):
        # ``prediction`` against ``reference`` in place of the data vector,
        # each block (visibilities, phases, every extra) whitened as by
        # `whitened_residuals`.
        whitened, _ = _whiten(data, prediction, reference, errors, {}, {})
        return jnp.sum(whitened**2) / ndof

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
