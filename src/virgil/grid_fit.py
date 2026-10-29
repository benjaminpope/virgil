"""Grid searches over model parameters.

Grids use ``indexing="ij"``: one output axis per grid key, in the order of
``grid`` (for ``dra``, ``ddec``, axis 0 is East, axis 1 North, so transpose
a 2D map to show it North up; [`plot_grid_map`][virgil.plotting.plot_grid_map]
does). The flux optimized at each position is the one key ending in ``flux``,
unless ``flux_param`` says otherwise; fluxes are companion/primary (see
[`virgil.limits`][virgil.limits] for contrast and Δmag, and for contrast limits).
See [Grid searches](../grid_searches.md).
"""

import math
from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as np
import numpy as onp
import numpyro.distributions as dist
import optimistix as optx

from ._grid import (
    batch_size_or_default,
    check_flux_axes,
    coordinate_points,
    map_points,
    ordered_values,
    resolve_grid_keys,
    warn_unconverged,
)
from ._deprecate import old_order, renamed
from ._marginal import gauss_legendre
from .inference import _ambient_dtype, laplace_parameter_uncertainty
from .likelihood import build_model, loglike, whitened_residuals


def _best_grid_flux(loglike_im, grid, params, flux_key):
    """Best flux on a full likelihood grid, and its log likelihood, at every
    position (``loglike_im`` has one axis per key of ``params``)."""
    flux_axis = params.index(flux_key)
    best_index = np.nanargmax(loglike_im, axis=flux_axis)
    best_flux = np.asarray(grid[flux_key])[best_index]
    return best_flux, np.nanmax(loglike_im, axis=flux_axis)


@eqx.filter_jit
def _optimize_flux_grid(
    data, model, grid, params, coord_keys, flux_key, batch_size
):
    """Refine the best grid flux at every position with BFGS.

    Returns ``(flux, loglike, converged)``, each with one axis per coordinate
    key; a point has converged when within a quarter sigma of the likelihood
    maximum along the flux. Starts from the best fluxes of the full likelihood
    grid, in units of the starting flux.
    """
    loglike_im = _likelihood_grid(data, model, grid, params, batch_size)
    start_flux, start_loglike = _best_grid_flux(
        loglike_im, grid, params, flux_key
    )
    return _refine_flux_grid(
        data,
        model,
        grid,
        params,
        coord_keys,
        flux_key,
        batch_size,
        start_flux,
        start_loglike,
    )


def _refine_flux_grid(
    data,
    model,
    grid,
    params,
    coord_keys,
    flux_key,
    batch_size,
    start_flux,
    start_loglike,
):
    """BFGS refinement of :func:`_optimize_flux_grid` from given starts.

    ``start_flux`` and ``start_loglike`` have one axis per coordinate key. Not
    jitted itself: traced inside ``_optimize_flux_grid`` and the detection
    statistics.
    """
    coords, shape = coordinate_points(grid, coord_keys)

    def objective(x, coord_vals, scale, loglike0):
        values = ordered_values(
            x * scale, coord_vals, params, coord_keys, flux_key
        )
        return loglike0 - loglike(values, params, model, data)

    def flux_loglike(flux, coord_vals):
        values = ordered_values(flux, coord_vals, params, coord_keys, flux_key)
        return loglike(values, params, model, data)

    def newton_step(flux, coord_vals):
        grad = jax.grad(flux_loglike)(flux, coord_vals)
        curvature = jax.grad(jax.grad(flux_loglike))(flux, coord_vals)
        step = np.where(curvature < 0.0, grad / curvature, 0.0)
        # Keep the step only if it improves the fit: far from quadratic
        # regions a Newton step can overshoot.
        trial = flux - step
        improved = flux_loglike(trial, coord_vals) >= flux_loglike(
            flux, coord_vals
        )
        return np.where(improved, trial, flux)

    def refine(flux0, coord_vals, loglike0):
        scale = np.where(np.abs(flux0) > 0.0, np.abs(flux0), 1.0)
        result = optx.compat.minimize(
            objective,
            x0=np.array([flux0 / scale]),
            args=(coord_vals, scale, loglike0),
            method="BFGS",
            options={"maxiter": 100},
        )
        # BFGS stops once the change in log likelihood is below its
        # tolerance, which in float32 is comparable to rounding noise. Two
        # Newton steps on the analytic gradient pin down the maximum.
        flux = result.x[0] * scale
        for _ in range(2):
            flux = newton_step(flux, coord_vals)
        # Converged if the remaining distance to the maximum, estimated from
        # the gradient and curvature, is under a quarter sigma of the flux.
        # (Float32 rounding alone leaves offsets of up to ~0.1 sigma.)
        grad = jax.grad(flux_loglike)(flux, coord_vals)
        curvature = jax.grad(jax.grad(flux_loglike))(flux, coord_vals)
        converged = (curvature < 0.0) & (
            np.abs(grad) < 0.25 * np.sqrt(np.abs(curvature))
        )
        return flux, flux_loglike(flux, coord_vals), converged

    flux, best_loglike, success = map_points(
        refine,
        start_flux.reshape(-1),
        coords,
        start_loglike.reshape(-1),
        batch_size=batch_size,
    )
    return (
        flux.reshape(shape),
        best_loglike.reshape(shape),
        success.reshape(shape),
    )


@old_order("data", "model", "grid", data="data", keywords=("data", "grid"))
def likelihood_grid(model, *, data, grid, batch_size=None):
    """Evaluate the log likelihood at every point of a parameter grid.

    Only ``model`` may be positional: ``likelihood_grid(model, data=data, grid=grid)``
    (the 0.3 order warns until 0.5).

    Parameters
    ----------
    model : SourceModel or class
        Template model whose parameters at the paths in ``grid`` are varied
        (e.g. a [System][virgil.models.System] with ``"comp.dra"``), or a model
        class called with ``grid``'s keys as keywords.
    data : OIData
        Data to fit.
    grid : dict[str, array-like]
        Parameter name or path to 1D values (``dra``/``ddec`` in mas, ``flux``
        as companion/primary); the output has one axis per key, in this order.
    batch_size : int, optional
        Grid points evaluated at once (default about 2**20 model visibilities
        on a CPU, 2**23 elsewhere, at least 256); smaller bounds memory.

    Returns
    -------
    array-like
        Log likelihood of shape ``tuple(len(v) for v in grid.values())``.
    """
    params = tuple(grid.keys())
    check_flux_axes(grid)
    return _likelihood_grid(
        data,
        model,
        grid,
        params=params,
        batch_size=batch_size_or_default(batch_size, data),
    )


@eqx.filter_jit
def _likelihood_grid(data, model, grid, params, batch_size):
    """Jitted implementation of [`likelihood_grid`][virgil.grid_fit.likelihood_grid]."""

    vals_vec, grid_shape = coordinate_points(grid, params)

    return map_points(
        lambda values: loglike(values, params, model, data),
        vals_vec,
        batch_size=batch_size,
    ).reshape(grid_shape)


_OPTIMIZED_PARAMS_DOC = """
    Parameters
    ----------
    model : SourceModel or class
        As for [`likelihood_grid`][virgil.grid_fit.likelihood_grid].
    data : OIData
        Data to fit.
    grid : dict[str, array-like]
        As for ``likelihood_grid``; the flux axis only sets the optimizer's
        starting points, and the output has one axis per other key.
    flux_param : str, optional
        Key of ``grid`` holding the optimized flux (default: the key ending in
        ``flux``).
    batch_size : int, optional
        Grid points evaluated at once.
"""


@old_order("data", "model", "grid", data="data")
def optimized_likelihood_grid(
    model, data, grid, *, flux_param=None, batch_size=None
):
    params, coord_keys, flux_key = resolve_grid_keys(grid, flux_param)
    _, best_loglike, success = _optimize_flux_grid(
        data,
        model,
        grid,
        params=params,
        coord_keys=coord_keys,
        flux_key=flux_key,
        batch_size=batch_size_or_default(batch_size, data),
    )
    warn_unconverged(success, "optimized_likelihood_grid")
    return best_loglike


optimized_likelihood_grid.__doc__ = (
    """Find the maximum log likelihood over flux at every grid position.

    The best grid flux is refined by BFGS with the coordinates held fixed;
    a ``RuntimeWarning`` flags non-convergence.
"""
    + _OPTIMIZED_PARAMS_DOC
    + """
    Returns
    -------
    array-like
        Log likelihood at the optimized flux, one axis per coordinate key.
    """
)


@old_order("data", "model", "grid", data="data")
def optimized_flux_grid(
    model, data, grid, *, flux_param=None, batch_size=None
):
    params, coord_keys, flux_key = resolve_grid_keys(grid, flux_param)
    best_flux, _, success = _optimize_flux_grid(
        data,
        model,
        grid,
        params=params,
        coord_keys=coord_keys,
        flux_key=flux_key,
        batch_size=batch_size_or_default(batch_size, data),
    )
    warn_unconverged(success, "optimized_flux_grid")
    return best_flux


optimized_flux_grid.__doc__ = (
    """Find the best-fit flux at every grid position.

    The best grid flux is refined by BFGS with the coordinates held fixed. The
    flux is not constrained to be positive, as
    [`ruffio_upperlimit`][virgil.limits.ruffio_upperlimit] expects; a
    ``RuntimeWarning`` flags non-convergence.
"""
    + _OPTIMIZED_PARAMS_DOC
    + """
    Returns
    -------
    array-like
        Best-fit ``flux_param``, one axis per coordinate key.
    """
)


class Gaussian(NamedTuple):
    """Gaussian prior ``N(mean, sd**2)`` on the companion flux ratio.

    Its support includes negative flux: a closed form only. Prefer
    [`LogUniform`][virgil.grid_fit.LogUniform].
    """

    mean: float
    sd: float


class LogUniform(NamedTuple):
    """Log-uniform (scale-invariant) prior on the companion flux ratio.

    ``p(f) = 1 / (f ln(f_max / f_min))`` on ``f_min <= f <= f_max``, with
    ``0 < f_min < f_max`` finite, in units of the companion/primary flux. See
    [Grid searches](../grid_searches.md#linear-flux-prior).
    """

    f_min: float
    f_max: float


_N_NODES = 256  # Gauss-Legendre nodes in ln f for the log-uniform evidence
_WINDOW_SIGMA = 12.0  # half-width of the integration window, in sigma_f
_TAIL_SIGMA = 40.0  # decay lengths kept when f_hat is outside the bounds


def _scalar_params(d, names):
    """Scalar float parameters of a numpyro distribution, or a clear error."""
    out = []
    for name in names:
        v = onp.asarray(getattr(d, name))
        if v.ndim != 0:
            raise ValueError(
                f"linear_flux_grid needs scalar {type(d).__name__} "
                f"parameters, but {name} has shape {v.shape}."
            )
        out.append(float(v))
    return out


class LinearFluxGrid(NamedTuple):
    """Result of [`linear_flux_grid`][virgil.grid_fit.linear_flux_grid].

    The last three fields are ``None`` unless a ``prior`` was given.
    """

    flux: np.ndarray
    flux_error: np.ndarray
    snr: np.ndarray
    posterior_mean: np.ndarray | None = None
    posterior_sd: np.ndarray | None = None
    log_bayes_factor: np.ndarray | None = None


@eqx.filter_jit
def _linear_flux_grid(
    data,
    model,
    grid,
    params,
    coord_keys,
    flux_key,
    batch_size,
    n_iter,
    prior,
):
    """Jitted implementation of [`linear_flux_grid`][virgil.grid_fit.linear_flux_grid]."""
    coords, shape = coordinate_points(grid, coord_keys)

    def residuals(flux, coord_vals):
        values = ordered_values(flux, coord_vals, params, coord_keys, flux_key)
        return whitened_residuals(build_model(model, params, values), data)

    def solve(coord_vals):
        def linearize(flux):
            # r(f_n) and g = dr/df at f_n, exactly (forward mode).
            r, g = jax.jvp(
                lambda f: residuals(f, coord_vals),
                (flux,),
                (np.ones(()),),
            )
            curvature = np.sum(g * g)
            # Gauss-Newton step: the minimizer of the model linear about f_n.
            return flux - np.sum(g * r) / curvature, curvature

        # Start at f = 0: the closed-form result is exact only for faint companions.
        flux, curvature = linearize(np.zeros(()))
        # Relinearize at the current estimate; a fixed trip count keeps this
        # jit- and vmap-friendly. The last g is the one sigma_f comes from.
        flux, curvature = jax.lax.fori_loop(
            0, n_iter, lambda _, c: linearize(c[0]), (flux, curvature)
        )
        ok = curvature > 0.0
        flux = np.where(ok, flux, np.nan)
        sigma = np.where(ok, 1.0 / np.sqrt(curvature), np.nan)
        if prior is None:
            return flux, sigma, None, None, None
        if isinstance(prior, LogUniform):
            safe = np.where(ok, curvature, 1.0)
            # Evidence, posterior mean and sd of f under LogUniform(f_min,
            # f_max): the Gaussian likelihood of f (mean flux, sd sig)
            # relative to f = 0, integrated by Gauss-Legendre quadrature in
            # ln f over the part of the bounds where it is not negligible.
            sig = 1.0 / np.sqrt(safe)
            gap = np.maximum(
                prior.f_min - flux, flux - prior.f_max
            )  # > 0 outside the bounds
            outside = gap > sig
            width = np.where(
                outside,
                _TAIL_SIGMA * sig * (sig / np.maximum(gap, sig)),
                _WINDOW_SIGMA * sig,
            )
            above = outside & (flux > prior.f_max)
            centre = np.clip(flux, prior.f_min, prior.f_max)
            lo = np.clip(centre - width, prior.f_min, prior.f_max)
            hi = np.clip(centre + width, prior.f_min, prior.f_max)
            # The window is f = f0 * exp(sgn * delta), delta in [0, span]: anchored at
            # the bound nearest flux when flux is outside, so that a window much
            # narrower than f (float32!) is still resolved, and f - flux is formed
            # from offsets rather than from two nearly equal fluxes.
            tail = np.minimum(width, prior.f_max - prior.f_min)
            f0 = np.where(above, prior.f_max, lo)
            sgn = np.where(above, -1.0, 1.0)
            span = np.where(
                outside,
                np.where(
                    above,
                    -np.log1p(-tail / prior.f_max),
                    np.log1p(tail / prior.f_min),
                ),
                np.log(hi) - np.log(lo),
            )
            x, w = gauss_legendre(_N_NODES)
            delta = 0.5 * span * (x + 1.0)
            wd = 0.5 * span * w
            offset = f0 * np.expm1(sgn * delta)  # f - f0
            # (f - flux)^2 - flux^2 with f = f0 + offset, expanded so that nothing
            # large cancels: f0 (f0 - 2 flux) + 2 (f0 - flux) offset + offset^2.
            h = -(
                f0 * (f0 - 2.0 * flux) + 2.0 * (f0 - flux) * offset + offset**2
            ) / (2.0 * sig**2)
            m = np.max(h)
            wt = wd * np.exp(h - m)
            s0 = np.sum(wt)
            mean_offset = np.sum(wt * offset) / s0
            var = np.sum(wt * (offset - mean_offset) ** 2) / s0
            post_mean = f0 + mean_offset
            post_sd = np.sqrt(var)
            log_bf = (
                m
                + np.log(s0)
                - np.log(np.log(prior.f_max) - np.log(prior.f_min))
            )
            return (
                flux,
                sigma,
                np.where(ok, post_mean, np.nan),
                np.where(ok, post_sd, np.nan),
                np.where(ok, log_bf, np.nan),
            )
        # Gaussian: the likelihood is exp(-curvature (f - flux)^2 / 2) up to a
        # constant in the linear model about the final point. With a
        # N(mean, sd^2) prior the precisions add (Luger et al. 2017).
        mean, sd = prior
        precision = curvature + 1.0 / sd**2
        b = curvature * flux  # g . (g flux), whitened
        post_mean = (b + mean / sd**2) / precision
        log_bf = (
            -0.5 * np.log(sd**2 * precision)
            + (b + mean / sd**2) ** 2 / (2.0 * precision)
            - mean**2 / (2.0 * sd**2)
        )
        return (
            flux,
            sigma,
            np.where(ok, post_mean, np.nan),
            np.where(ok, precision**-0.5, np.nan),
            np.where(ok, log_bf, np.nan),
        )

    out = map_points(solve, coords, batch_size=batch_size)
    out = tuple(None if o is None else o.reshape(shape) for o in out)
    flux, sigma = out[:2]
    return LinearFluxGrid(flux, sigma, flux / sigma, *out[2:])


@old_order("data", "model", "grid", data="data")
def linear_flux_grid(
    model,
    data,
    grid,
    *,
    flux_param=None,
    batch_size=None,
    n_iter=0,
    prior=None,
):
    """Linearized best-fit companion flux at every grid position, in closed form.

    A fast first pass beside
    [`optimized_flux_grid`][virgil.grid_fit.optimized_flux_grid]: for a faint
    companion the residuals are linear in the flux, so one model evaluation
    per position gives the flux (either sign) and its error. Raise ``n_iter``
    for a bright companion; see [Grid searches](../grid_searches.md#linear-flux-grid).

    Source: Luger et al. (2017), [references](../references.md#luger2017linear);
    compare with [fouriever](../references.md#fouriever)'s ``lincmap`` (closure
    or kernel phases only).

    Parameters
    ----------
    model, data, grid, flux_param, batch_size
        As for ``optimized_flux_grid``. At zero flux ``model`` must reduce to
        the primary alone. The flux axis of ``grid`` is ignored (the flux is
        solved for) but must name the parameter.
    n_iter : int, optional
        Gauss-Newton relinearization steps after the first at ``f = 0``.
    prior : numpyro LogUniform or Normal, optional
        Prior on ``f``, filling the posterior and Bayes-factor fields. It must
        be a scalar distribution with both bounds (``LogUniform``) or both
        parameters (``Normal``), and the Bayes factor depends on the bounds, so
        choose them (see [Grid searches](../grid_searches.md#linear-flux-prior)).

    Returns
    -------
    LinearFluxGrid
        Named tuple of arrays with one axis per coordinate key: ``flux``,
        ``flux_error`` (NaN where the model ignores the flux), ``snr`` (flux over error), and
        with a ``prior`` ``posterior_mean``, ``posterior_sd`` and
        ``log_bayes_factor`` (per position; > 0 favours a companion; else ``None``).

    Examples
    --------
    >>> axis = np.linspace(-300.0, 300.0, 61)
    >>> grid = {"dra": axis, "ddec": axis, "flux": np.array([1e-3])}
    >>> res = linear_flux_grid(BinaryModelCartesian, data, grid)  # doctest: +SKIP
    """
    params, coord_keys, flux_key = resolve_grid_keys(grid, flux_param)
    # numpyro's dist.LogUniform / dist.Normal (scalar parameters) become the
    # NamedTuples; anything else is validated
    if prior is not None:
        if isinstance(prior, dist.LogUniform):
            prior = LogUniform(*_scalar_params(prior, ("low", "high")))
        elif isinstance(prior, dist.Normal):
            prior = Gaussian(*_scalar_params(prior, ("loc", "scale")))
        elif not isinstance(prior, (Gaussian, LogUniform)):
            hint = (
                " A bare (mean, sd) tuple is not accepted: use dist.Normal."
                if isinstance(prior, tuple)
                else ""
            )
            raise TypeError(
                "prior must be numpyro's dist.LogUniform(low, high) or "
                "dist.Normal(mean, sd) (or grid_fit.LogUniform / Gaussian), "
                f"got {prior!r}.{hint}"
            )
        a, b = (float(x) for x in prior)
        if not (math.isfinite(a) and math.isfinite(b)):
            raise ValueError(
                f"{type(prior).__name__} needs finite parameters, got {prior}"
            )
        if isinstance(prior, LogUniform) and not 0.0 < a < b:
            raise ValueError(
                f"LogUniform needs 0 < f_min < f_max, got ({a}, {b})"
            )
        if isinstance(prior, Gaussian) and not b > 0.0:
            raise ValueError(f"Gaussian needs sd > 0, got sd = {b}")
        prior = type(prior)(np.asarray(a), np.asarray(b))
    return _linear_flux_grid(
        data,
        model,
        grid,
        params=params,
        coord_keys=coord_keys,
        flux_key=flux_key,
        batch_size=batch_size_or_default(batch_size, data),
        n_iter=int(n_iter),
        prior=prior,
    )


@old_order("data", "model", "grid", data="data")
def laplace_flux_uncertainty_grid(
    model,
    data,
    grid,
    flux=None,
    *,
    flux_param=None,
    batch_size=None,
):
    """Laplace uncertainty of the flux at every grid position.

    The inverse square root of the curvature of the negative log likelihood
    along the flux, with the coordinates held fixed.

    Parameters
    ----------
    model, data, grid, flux_param, batch_size
        As for [`optimized_flux_grid`][virgil.grid_fit.optimized_flux_grid].
    flux : array-like, optional
        Flux at which to evaluate the curvature, one axis per coordinate key
        (default: the ``optimized_flux_grid`` best fit, the mean
        [`ruffio_upperlimit`][virgil.limits.ruffio_upperlimit] expects).

    Returns
    -------
    array-like
        One-sigma flux uncertainty per coordinate-key point; NaN where the
        curvature is not positive.
    """
    params, coord_keys, flux_key = resolve_grid_keys(grid, flux_param)
    if flux is None:
        flux = optimized_flux_grid(
            model,
            data,
            grid,
            flux_param=flux_key,
            batch_size=batch_size,
        )
    return _laplace_flux_uncertainty_grid(
        np.asarray(flux),
        data,
        model,
        grid,
        params=params,
        coord_keys=coord_keys,
        flux_key=flux_key,
        batch_size=batch_size_or_default(batch_size, data),
    )


@eqx.filter_jit
def _laplace_flux_uncertainty_grid(
    flux_values,
    data,
    model,
    grid,
    params,
    coord_keys,
    flux_key,
    batch_size,
):
    """Jitted implementation of :func:`laplace_flux_uncertainty_grid`."""
    coords, shape = coordinate_points(grid, coord_keys)

    def sigma(flux, coord_vals):
        values = np.stack(
            ordered_values(flux, coord_vals, params, coord_keys, flux_key)
        )
        return laplace_parameter_uncertainty(
            values=values,
            params=params,
            data=data,
            model=model,
            target_param=flux_key,
            # Inside this jit: stay in the caller's precision.
            dtype=_ambient_dtype(),
        )

    return map_points(
        sigma, flux_values.reshape(-1), coords, batch_size=batch_size
    ).reshape(shape)


@renamed()
def best_grid_point(loglike_grid, grid):
    """Return the grid point with the highest log likelihood.

    Parameters
    ----------
    loglike_grid : array-like
        Output of [`likelihood_grid`][virgil.grid_fit.likelihood_grid] for
        ``grid``, with one axis per key. NaNs are ignored.
    grid : dict[str, array-like]
        The grid axes used to compute ``loglike_grid``.

    Returns
    -------
    dict[str, float]
        ``{name: value}`` at the maximum, in the order of ``grid``.
    """
    shape = np.shape(loglike_grid)
    if len(shape) != len(grid):
        raise ValueError(
            f"loglike_grid has {len(shape)} axes but grid has "
            f"{len(grid)} keys; pass the full likelihood_grid output."
        )
    index = onp.unravel_index(int(np.nanargmax(loglike_grid)), shape)
    return {
        key: float(values[i]) for (key, values), i in zip(grid.items(), index)
    }
