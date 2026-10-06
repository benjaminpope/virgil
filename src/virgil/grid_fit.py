"""Grid searches over model parameters.

Grids are built with ``indexing="ij"``: every output has one axis per grid
key, in the order of ``grid``. For ``{"dra", "ddec", ...}`` axis 0 is
``dra`` (East offset) and axis 1 is ``ddec`` (North offset), so 2D maps need
a transpose to be shown as images with North up;
[`plot_grid_map`][virgil.plotting.plot_grid_map] handles this.

The functions that optimize a flux at every grid position find it as the one
key whose last part is ``flux`` (``flux``, ``comp.flux``, ...), unless
``flux_param`` says otherwise. Fluxes are relative to the primary (at flux
1), so for a companion the flux is its companion/primary flux ratio; see
[`virgil.limits`][virgil.limits] for converting to contrast or Δmag.
Contrast limits (Ruffio, Absil) are in [`virgil.limits`][virgil.limits].
"""

import math
from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import optimistix as optx

from ._grid import (
    batch_size_or_default,
    check_flux_axes,
    coordinate_points,
    map_points,
    meshgrid_vectors,
    ordered_values,
    resolve_grid_keys,
    warn_unconverged,
)
from ._deprecate import old_order, renamed
from .inference import laplace_parameter_uncertainty
from .likelihood import build_model, loglike, whitened_residuals


def _best_grid_flux(loglike_im, grid, params, flux_key):
    """Best flux on a full likelihood grid, and its log likelihood, at every
    position (``loglike_im`` has one axis per key of ``params``)."""
    flux_axis = params.index(flux_key)
    best_index = jnp.nanargmax(loglike_im, axis=flux_axis)
    best_flux = jnp.asarray(grid[flux_key])[best_index]
    return best_flux, jnp.nanmax(loglike_im, axis=flux_axis)


@eqx.filter_jit
def _optimize_flux_grid(
    data, model, grid, params, coord_keys, flux_key, batch_size
):
    """Refine the best grid flux at every position with BFGS.

    Returns ``(flux, loglike, converged)``, each with one axis per
    coordinate key; a point has converged when it is within a quarter sigma
    of the likelihood maximum along the flux. The optimizer works in units of the starting flux, and on the log
    likelihood relative to its starting value, so its default tolerances are
    relative to the problem's own scale. The starting points are the best
    fluxes of the full likelihood grid.
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

    ``start_flux`` and ``start_loglike`` have one axis per coordinate key,
    e.g. from :func:`_best_grid_flux`. Not jitted itself: it is traced
    inside :func:`_optimize_flux_grid` and the detection statistics, which
    reuse a full likelihood grid they have already computed.
    """
    coords, shape = coordinate_points(grid, coord_keys)

    def objective(x, coord_vals, scale, loglike0):
        values = ordered_values(
            x * scale, coord_vals, params, coord_keys, flux_key
        )
        return loglike0 - loglike(values, params, data, model)

    def flux_loglike(flux, coord_vals):
        values = ordered_values(flux, coord_vals, params, coord_keys, flux_key)
        return loglike(values, params, data, model)

    def newton_step(flux, coord_vals):
        grad = jax.grad(flux_loglike)(flux, coord_vals)
        curvature = jax.grad(jax.grad(flux_loglike))(flux, coord_vals)
        step = jnp.where(curvature < 0.0, grad / curvature, 0.0)
        # Keep the step only if it improves the fit: far from quadratic
        # regions a Newton step can overshoot.
        trial = flux - step
        improved = flux_loglike(trial, coord_vals) >= flux_loglike(
            flux, coord_vals
        )
        return jnp.where(improved, trial, flux)

    def refine(flux0, coord_vals, loglike0):
        scale = jnp.where(jnp.abs(flux0) > 0.0, jnp.abs(flux0), 1.0)
        result = optx.compat.minimize(
            objective,
            x0=jnp.array([flux0 / scale]),
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
            jnp.abs(grad) < 0.25 * jnp.sqrt(jnp.abs(curvature))
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


@old_order("data", "model", "grid", data="data")
def likelihood_grid(model, data, grid, *, batch_size=None):
    """Evaluate the log likelihood at every point of a parameter grid.

    Parameters
    ----------
    model : SourceModel or class
        Template model whose parameters at the paths in ``grid`` are
        varied (e.g. a [System][virgil.models.System] with paths such as
        ``"comp.dra"``), or a model class called with ``grid``'s keys
        as keyword arguments (e.g. ``BinaryModelCartesian``).
    data : OIData
        Data to fit.
    grid : dict[str, array-like]
        Grid axes, as a mapping from parameter name or path to 1D values
        (e.g. ``dra``/``ddec`` in milliarcseconds and ``flux`` as a
        companion/primary flux ratio). The output has one axis per key, in
        this order.
    batch_size : int, optional
        Number of grid points evaluated at once. By default, enough for
        about 2**20 model visibilities on a CPU and 2**23 on other backends
        (GPU, TPU), and at least 256. Larger can be faster for small data;
        smaller bounds memory for large models.

    Returns
    -------
    array-like
        Log likelihood with shape
        ``tuple(len(v) for v in grid.values())``. Axis ``k`` follows
        the ``k``-th key (``indexing="ij"``), so for
        ``{"dra", "ddec", "flux"}`` axis 0 is ``dra``: transpose a 2D slice
        before showing it as an image with North up.
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

    vals_vec, grid_shape = meshgrid_vectors(grid, params)

    return map_points(
        lambda values: loglike(values, params, data, model),
        vals_vec,
        batch_size=batch_size,
    ).reshape(grid_shape)


_OPTIMIZED_PARAMS_DOC = """
    Parameters
    ----------
    model : SourceModel or class
        Template model whose parameters at the paths in ``grid`` are
        varied (e.g. a [System][virgil.models.System] with paths such as
        ``"comp.dra"``), or a model class called with ``grid``'s keys
        as keyword arguments (e.g. ``BinaryModelCartesian``).
    data : OIData
        Data to fit.
    grid : dict[str, array-like]
        Grid axes, as a mapping from parameter name or path to 1D values
        (e.g. ``dra``/``ddec`` in milliarcseconds and ``flux`` as a
        companion/primary flux ratio). The output has one axis per
        coordinate key (every key except ``flux_param``), in this order;
        the flux axis only sets the optimizer's starting points.
    flux_param : str, optional
        The key of ``grid`` holding the flux optimized at each grid
        position, e.g. ``"comp.flux"``. By default, the one key whose last
        part is ``flux``.
    batch_size : int, optional
        Number of grid points evaluated at once. By default, enough for
        about 2**20 model visibilities on a CPU and 2**23 on other backends
        (GPU, TPU), and at least 256. Larger can be faster for small data;
        smaller bounds memory for large models.
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

    A grid search over ``flux_param`` gives the starting point, which BFGS
    then refines with the coordinates held fixed. A ``RuntimeWarning`` is
    raised if the optimizer fails to converge anywhere.
"""
    + _OPTIMIZED_PARAMS_DOC
    + """
    Returns
    -------
    array-like
        Log likelihood at the optimized flux, with one axis per coordinate
        key (axis 0 is the first coordinate key, e.g. ``dra``).
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

    A grid search over ``flux_param`` gives the starting point, which BFGS
    then refines with the coordinates held fixed. The flux is not
    constrained to be positive, as
    [`ruffio_upperlimit`][virgil.limits.ruffio_upperlimit] expects. A
    ``RuntimeWarning`` is raised if the optimizer fails to converge
    anywhere.
"""
    + _OPTIMIZED_PARAMS_DOC
    + """
    Returns
    -------
    array-like
        Best-fit value of ``flux_param`` (for a companion, its
        companion/primary flux ratio), with one axis per coordinate key
        (axis 0 is the first coordinate key, e.g. ``dra``).
    """
)


class Gaussian(NamedTuple):
    """Gaussian prior ``N(mean, sd**2)`` on the companion flux ratio.

    Gaussian-prior evidence (a computational approximation; ``f`` may go
    negative): the prior has support on negative flux, so its evidence is
    only a convenient closed form. Prefer [`LogUniform`][virgil.grid_fit.LogUniform].
    """

    mean: float
    sd: float


class LogUniform(NamedTuple):
    """Log-uniform (scale-invariant) prior on the companion flux ratio.

    ``p(f) = 1 / (f ln(f_max / f_min))`` on ``f_min <= f <= f_max``. The flux
    ratio is a scale parameter spanning decades, so this is the invariant
    measure of the scaling group (the Jeffreys prior under that group
    action), not the root-Fisher-information prior of the linearised
    likelihood, which has constant Fisher information in ``f`` and so would
    be flat. It is improper without bounds, so the evidence needs
    ``0 < f_min < f_max`` (finite), in the units of
    the flux (companion/primary).
    """

    f_min: float
    f_max: float


_N_NODES = 256  # Gauss-Legendre nodes in ln f for the log-uniform evidence
_WINDOW_SIGMA = 12.0  # half-width of the integration window, in sigma_f
_TAIL_SIGMA = 40.0  # decay lengths kept when f_hat is outside the bounds


def _log_uniform_evidence(f_hat, sigma, f_min, f_max):
    """Evidence, posterior mean and sd of f under ``LogUniform(f_min, f_max)``.

    In the linear model the likelihood relative to ``f = 0`` is
    ``h(f) = -((f - f_hat)**2 - f_hat**2) / (2 sigma**2)``, and the evidence
    is ``(1 / ln(f_max / f_min)) * integral h-exponential d ln f``. It is
    computed by fixed-node Gauss-Legendre quadrature in ``ln f`` over the
    part of ``[f_min, f_max]`` where the likelihood is not negligible (within
    ``_WINDOW_SIGMA`` sigma of ``f_hat``, or, with ``f_hat`` outside the
    bounds, within the decay length of the nearer bound), so a narrow peak
    inside wide bounds is resolved. Returns ``(log_B, mean, sd)``.
    """
    d = jnp.maximum(f_min - f_hat, f_hat - f_max)  # > 0 outside the bounds
    outside = d > sigma
    width = jnp.where(
        outside,
        _TAIL_SIGMA * sigma * (sigma / jnp.maximum(d, sigma)),
        _WINDOW_SIGMA * sigma,
    )
    above = outside & (f_hat > f_max)
    centre = jnp.clip(f_hat, f_min, f_max)
    lo = jnp.clip(centre - width, f_min, f_max)
    hi = jnp.clip(centre + width, f_min, f_max)
    # The window is f = f0 * exp(sgn * delta), delta in [0, span]: anchored at
    # the bound nearest f_hat when f_hat is outside, so that a window much
    # narrower than f (float32!) is still resolved, and f - f_hat is formed
    # from offsets rather than from two nearly equal fluxes.
    tail = jnp.minimum(width, f_max - f_min)
    f0 = jnp.where(above, f_max, lo)
    sgn = jnp.where(above, -1.0, 1.0)
    span = jnp.where(
        outside,
        jnp.where(above, -jnp.log1p(-tail / f_max), jnp.log1p(tail / f_min)),
        jnp.log(hi) - jnp.log(lo),
    )
    x, w = np.polynomial.legendre.leggauss(_N_NODES)
    delta = 0.5 * span * (jnp.asarray(x) + 1.0)
    wd = 0.5 * span * jnp.asarray(w)
    offset = f0 * jnp.expm1(sgn * delta)  # f - f0
    # (f - f_hat)^2 - f_hat^2 with f = f0 + offset, expanded so that nothing
    # large cancels: f0 (f0 - 2 f_hat) + 2 (f0 - f_hat) offset + offset^2.
    h = -(
        f0 * (f0 - 2.0 * f_hat) + 2.0 * (f0 - f_hat) * offset + offset**2
    ) / (2.0 * sigma**2)
    m = jnp.max(h)
    wt = wd * jnp.exp(h - m)
    s0 = jnp.sum(wt)
    mean_offset = jnp.sum(wt * offset) / s0
    var = jnp.sum(wt * (offset - mean_offset) ** 2) / s0
    mean = f0 + mean_offset
    log_b = m + jnp.log(s0) - jnp.log(jnp.log(f_max) - jnp.log(f_min))
    return log_b, mean, jnp.sqrt(var)


def _as_prior(prior):
    """Validate ``prior`` and return it as ``Gaussian``, ``LogUniform`` or None.

    Accepts numpyro's ``dist.LogUniform(low, high)`` and ``dist.Normal(mean,
    sd)`` (scalar parameters), converting them to the NamedTuples, as well as
    the NamedTuples themselves.
    """
    if prior is None:
        return None
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
        raise ValueError(f"LogUniform needs 0 < f_min < f_max, got ({a}, {b})")
    if isinstance(prior, Gaussian) and not b > 0.0:
        raise ValueError(f"Gaussian needs sd > 0, got sd = {b}")
    return type(prior)(jnp.asarray(a), jnp.asarray(b))


def _scalar_params(d, names):
    """Scalar float parameters of a numpyro distribution, or a clear error."""
    out = []
    for name in names:
        v = np.asarray(getattr(d, name))
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

    flux: jnp.ndarray
    flux_error: jnp.ndarray
    snr: jnp.ndarray
    posterior_mean: jnp.ndarray | None = None
    posterior_sd: jnp.ndarray | None = None
    log_bayes_factor: jnp.ndarray | None = None


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
        def linearise(flux):
            # r(f_n) and g = dr/df at f_n, exactly (forward mode).
            r, g = jax.jvp(
                lambda f: residuals(f, coord_vals),
                (flux,),
                (jnp.ones(()),),
            )
            curvature = jnp.sum(g * g)
            # Gauss-Newton step: the minimiser of the model linear about f_n.
            return flux - jnp.sum(g * r) / curvature, curvature

        flux, curvature = linearise(jnp.zeros(()))
        # Relinearise at the current estimate; a fixed trip count keeps this
        # jit- and vmap-friendly. The last g is the one sigma_f comes from.
        flux, curvature = jax.lax.fori_loop(
            0, n_iter, lambda _, c: linearise(c[0]), (flux, curvature)
        )
        ok = curvature > 0.0
        flux = jnp.where(ok, flux, jnp.nan)
        sigma = jnp.where(ok, 1.0 / jnp.sqrt(curvature), jnp.nan)
        if prior is None:
            return flux, sigma, None, None, None
        if isinstance(prior, LogUniform):
            safe = jnp.where(ok, curvature, 1.0)
            log_bf, post_mean, post_sd = _log_uniform_evidence(
                flux, 1.0 / jnp.sqrt(safe), prior.f_min, prior.f_max
            )
            return (
                flux,
                sigma,
                jnp.where(ok, post_mean, jnp.nan),
                jnp.where(ok, post_sd, jnp.nan),
                jnp.where(ok, log_bf, jnp.nan),
            )
        # Gaussian: the likelihood is exp(-curvature (f - flux)^2 / 2) up to a
        # constant in the linear model about the final point. With a
        # N(mean, sd^2) prior the precisions add (Luger et al. 2017).
        mean, sd = prior
        precision = curvature + 1.0 / sd**2
        b = curvature * flux  # g . (g flux), whitened
        post_mean = (b + mean / sd**2) / precision
        log_bf = (
            -0.5 * jnp.log(sd**2 * precision)
            + (b + mean / sd**2) ** 2 / (2.0 * precision)
            - mean**2 / (2.0 * sd**2)
        )
        return (
            flux,
            sigma,
            jnp.where(ok, post_mean, jnp.nan),
            jnp.where(ok, precision**-0.5, jnp.nan),
            jnp.where(ok, log_bf, jnp.nan),
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
    """Linearised best-fit companion flux at every grid position, in closed form.

    A fast first pass for companion searches, beside the iterative
    [`optimized_flux_grid`][virgil.grid_fit.optimized_flux_grid], and the
    equivalent of fouriever's linear contrast map (``lincmap``). For a faint
    companion (flux ``f`` much smaller than 1) the whitened residuals
    ``r`` of virgil's likelihood are linear in ``f`` at fixed position:
    ``r(f) = r(0) + f g``, where ``g = dr/df`` at ``f = 0`` is computed
    exactly by forward-mode automatic differentiation of the model (no
    finite difference, no hand-derived closure-phase derivative). The
    weighted least-squares solution is then

    ``f_hat = -(g . r(0)) / (g . g)``, ``sigma_f = (g . g) ** -0.5``,

    which is ``(gᵀ C⁻¹ (d - m₀)) / (gᵀ C⁻¹ g)`` with ``C`` the data
    covariance, because ``r`` is already whitened: correlated closure phases
    are handled exactly as in the likelihood
    ([`whitened_residuals`][virgil.likelihood.whitened_residuals]), and
    every observable in the data (|V| or V², and phases) contributes.
    One model evaluation with its derivative per grid position, and no
    optimizer.

    ``f_hat`` is not constrained to be positive (as with
    ``optimized_flux_grid`` and fouriever's ``lincmap``), so noise gives
    negative values with SNR of either sign. Unlike fouriever, whose
    ``lincmap`` returns the *variance* ``1 / (g . g)`` (and warns not to
    trust it), ``sigma_f`` here is the standard deviation.

    **Limitation, and ``n_iter``.** With ``n_iter=0`` the linearisation
    holds only for ``f`` much smaller than 1. The
    closure phase of a binary scales as ``f`` only to first order, with
    corrections of order ``f**2`` (and ``f`` times the |V| change for
    amplitudes), so for a bright companion (for example ``f ~ 0.3``)
    ``f_hat`` is biased, by tens of percent, and ``sigma_f`` is
    unreliable. ``n_iter`` Gauss–Newton steps soften this: each relinearises
    at the current ``f_hat`` per pixel (``g = dr/df`` at ``f_hat``, then
    ``f_hat <- f_hat - (g . r(f_hat)) / (g . g)``), and ``sigma_f`` comes
    from the final ``g``, so a few steps (3 at ``f ~ 0.3``) reach
    the optimizer's ``f_hat`` and the Laplace ``sigma_f``, at
    ``n_iter + 1`` model evaluations per pixel. The steps are not
    safeguarded, so for companions far brighter than the primary or
    strongly non-linear residuals they can fail to converge; use
    [`optimized_flux_grid`][virgil.grid_fit.optimized_flux_grid] to refine
    candidates. At
    Δ-phase residuals of order 1 rad the phase wrapping is not
    linear either. A companion at a position where ``g`` is nearly zero
    (for example at a null of the baselines) has a large ``sigma_f``
    and so a small SNR.

    Parameters
    ----------
    model : SourceModel or class
        Template model whose parameters at the paths in ``grid`` are
        varied (e.g. a [System][virgil.models.System] with paths such as
        ``"comp.dra"``), or a model class called with ``grid``'s keys
        as keyword arguments (e.g. ``BinaryModelCartesian``). At zero flux
        it must reduce to the primary alone (flux 1).
    data : OIData
        Data to fit.
    grid : dict[str, array-like]
        Grid axes, as for
        [`optimized_flux_grid`][virgil.grid_fit.optimized_flux_grid]:
        ``dra``/``ddec`` in milliarcseconds, plus a flux key, whose values
        are ignored (the flux is solved for, at ``f = 0``), but which must
        be present to name the parameter. The output has one axis per
        coordinate key (every key except ``flux_param``), in this order.
    flux_param : str, optional
        The key of ``grid`` holding the flux, e.g. ``"comp.flux"``.
        By default, the one key whose last part is ``flux``.
    batch_size : int, optional
        Number of grid points evaluated at once. By default, enough for
        about 2**20 model visibilities on a CPU and 2**23 on other backends
        (GPU, TPU), and at least 256.
    n_iter : int, optional
        Number of Gauss–Newton refinement steps after the first
        linearisation at ``f = 0`` (default 0, the closed-form result).
    prior : numpyro LogUniform or Normal, optional
        Prior on the flux ratio ``f``. By default none, and the posterior
        and Bayes-factor fields of the result are ``None``. With a prior
        they hold the posterior mean and sd and the marginal-likelihood
        detection map ``log_bayes_factor`` (see Returns); ``log B > 0``
        favours a companion at that pixel. All of
        these hold in the linear model about the final linearisation point,
        i.e. exactly only where the residuals are linear in ``f`` over the
        posterior (``f`` much smaller than 1, or after enough ``n_iter`` for
        the point to sit near the posterior); the position is not
        marginalised. Give numpyro's ``dist.LogUniform(low, high)`` or
        ``dist.Normal(mean, sd)`` with scalar parameters (the ``LogUniform``
        and ``Gaussian`` named tuples of this module are still accepted); a
        bare ``(mean, sd)`` tuple is an error.

        **Recommended: ``dist.LogUniform(f_min, f_max)``**, the scale-invariant
        (Jeffreys, under the scaling group) prior for a flux ratio,
        ``p(f) = 1 / (f ln(f_max / f_min))``. The flux ratio is a scale
        parameter spanning decades, so the prior is the invariant measure of
        the scaling group, not the root-Fisher prior of the linearised
        likelihood (which would be flat). It is improper without bounds, and the evidence needs a
        proper prior, so both bounds are required (``0 < f_min < f_max``).
        The Bayes factor depends on them, as it must for a scale prior: for
        ``f_hat`` well inside the bounds, widening them by a factor changes
        ``log B`` by about ``-Δ ln(ln(f_max / f_min))`` (the Occam
        factor). Choose them from the physics or the data, for example
        ``f_max`` the brightest companion you would entertain and ``f_min``
        a little below the faintest contrast the data can reach (its
        dynamic range, e.g. the smallest ``flux_error`` over the grid). The likelihood in ``f`` is Gaussian with
        mean ``f_hat`` and sd ``sigma_f``, so ``Z = ∫ N(f; f_hat,
        sigma_f**2) p(f) df / N(0; f_hat, sigma_f**2)``, computed by fixed
        256-node Gauss–Legendre quadrature in ``ln f`` over the part of the
        bounds where the likelihood is not negligible; the posterior mean
        and sd come from the same quadrature. It is vmappable and
        jit-compatible.

        ``dist.Normal(mean, sd)`` gives a Gaussian-prior evidence (a
        computational approximation; ``f`` may go negative). With ``P = g . g
        + 1 / sd**2`` (``g`` the final whitened derivative) the posterior is
        Gaussian with mean ``(g . (g f_hat) + mean / sd**2) / P`` and sd
        ``P ** -0.5``, and the log Bayes factor against ``f = 0`` is the
        closed-form Gaussian evidence ratio

        ``log B = -0.5 log(sd**2 P) + (g.g f_hat + mean/sd**2)**2 / (2P)
        - mean**2 / (2 sd**2)``

        (Luger, Foreman-Mackey & Hogg 2017, arXiv:1710.11136).

    Returns
    -------
    LinearFluxGrid
        A named tuple, always of the same type, whose fields are arrays with
        one axis per coordinate key (axis 0 is the first, e.g. ``dra``):

        - ``flux``: best-fit flux ratio (companion/primary), unconstrained
          in sign.
        - ``flux_error``: one-sigma uncertainty on ``flux``, NaN where the
          model does not depend on the flux.
        - ``snr``: ``flux / flux_error``, the detection significance map.
        - ``posterior_mean``, ``posterior_sd``, ``log_bayes_factor``: the
          posterior under the given ``prior`` and the log evidence ratio
          against ``f = 0``; ``None`` if ``prior`` is not given, whatever
          the kind of prior.

        Because there are six fields, unpack by attribute (``res.flux``) or
        take the first three with ``flux, error, snr = res[:3]``; unpacking
        the result directly into three names fails.

    Examples
    --------
    >>> grid = {
    ...     "dra": jnp.linspace(-300.0, 300.0, 61),
    ...     "ddec": jnp.linspace(-300.0, 300.0, 61),
    ...     "flux": jnp.array([1e-3]),  # ignored: only names the parameter
    ... }
    >>> res = linear_flux_grid(
    ...     BinaryModelCartesian, data, grid
    ... )  # doctest: +SKIP
    >>> i, j = jnp.unravel_index(
    ...     jnp.nanargmax(res.snr), res.snr.shape
    ... )  # doctest: +SKIP
    """
    params, coord_keys, flux_key = resolve_grid_keys(grid, flux_param)
    return _linear_flux_grid(
        data,
        model,
        grid,
        params=params,
        coord_keys=coord_keys,
        flux_key=flux_key,
        batch_size=batch_size_or_default(batch_size, data),
        n_iter=int(n_iter),
        prior=_as_prior(prior),
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

    At each position the coordinates are held fixed and the uncertainty is
    the inverse square root of the curvature of the negative log likelihood
    along the flux.

    Parameters
    ----------
    model : SourceModel or class
        Template model or model class, as for :func:`likelihood_grid`.
    data : OIData
        Data to fit.
    grid : dict[str, array-like]
        Grid axes, as for :func:`optimized_flux_grid`. The output has one
        axis per coordinate key (every key except the flux), in this order.
    flux : array-like, optional
        Flux at which to evaluate the curvature, with one axis per
        coordinate key. By default this is the best fit from
        :func:`optimized_flux_grid`, which is also the mean that
        [`ruffio_upperlimit`][virgil.limits.ruffio_upperlimit] expects;
        pass it if you have already computed it.
    flux_param : str, optional
        The key of ``grid`` holding the flux. By default, the one key
        whose last part is ``flux``.
    batch_size : int, optional
        Number of grid points evaluated at once. By default, enough for
        about 2**20 model visibilities on a CPU and 2**23 on other backends
        (GPU, TPU), and at least 256. Larger can be faster for small data;
        smaller bounds memory for large models.

    Returns
    -------
    array-like
        One-sigma flux uncertainty, with one axis per coordinate key. It is
        NaN where the curvature is not positive (the flux is not at a
        likelihood maximum).
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
        jnp.asarray(flux),
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
        values = jnp.stack(
            ordered_values(flux, coord_vals, params, coord_keys, flux_key)
        )
        return laplace_parameter_uncertainty(
            values=values,
            params=params,
            data_obj=data,
            model=model,
            target_param=flux_key,
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
        Output of [`likelihood_grid`][virgil.grid_fit.likelihood_grid] for ``grid``, with one axis
        per key. NaNs are ignored.
    grid : dict[str, array-like]
        The grid axes used to compute ``loglike_grid``.

    Returns
    -------
    dict[str, float]
        ``{name: value}`` at the maximum, in the order of ``grid``.
    """
    shape = jnp.shape(loglike_grid)
    if len(shape) != len(grid):
        raise ValueError(
            f"loglike_grid has {len(shape)} axes but grid has "
            f"{len(grid)} keys; pass the full likelihood_grid output."
        )
    index = np.unravel_index(int(jnp.nanargmax(loglike_grid)), shape)
    return {
        key: float(values[i]) for (key, values), i in zip(grid.items(), index)
    }
