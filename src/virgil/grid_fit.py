"""Grid searches over model parameters.

Grids are built with ``indexing="ij"``: every output has one axis per grid
key, in the order of ``samples_dict``. For ``{"dra", "ddec", ...}`` axis 0 is
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

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
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
from .inference import laplace_parameter_uncertainty
from .likelihood import build_model, loglike, whitened_residuals


def _best_grid_flux(loglike_im, samples_dict, params, flux_key):
    """Best flux on a full likelihood grid, and its log likelihood, at every
    position (``loglike_im`` has one axis per key of ``params``)."""
    flux_axis = params.index(flux_key)
    best_index = jnp.nanargmax(loglike_im, axis=flux_axis)
    best_flux = jnp.asarray(samples_dict[flux_key])[best_index]
    return best_flux, jnp.nanmax(loglike_im, axis=flux_axis)


@eqx.filter_jit
def _optimize_flux_grid(
    data_obj, model, samples_dict, params, coord_keys, flux_key, batch_size
):
    """Refine the best grid flux at every position with BFGS.

    Returns ``(flux, loglike, converged)``, each with one axis per
    coordinate key; a point has converged when it is within a quarter sigma
    of the likelihood maximum along the flux. The optimizer works in units of the starting flux, and on the log
    likelihood relative to its starting value, so its default tolerances are
    relative to the problem's own scale. The starting points are the best
    fluxes of the full likelihood grid.
    """
    loglike_im = _likelihood_grid(
        data_obj, model, samples_dict, params, batch_size
    )
    start_flux, start_loglike = _best_grid_flux(
        loglike_im, samples_dict, params, flux_key
    )
    return _refine_flux_grid(
        data_obj,
        model,
        samples_dict,
        params,
        coord_keys,
        flux_key,
        batch_size,
        start_flux,
        start_loglike,
    )


def _refine_flux_grid(
    data_obj,
    model,
    samples_dict,
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
    coords, shape = coordinate_points(samples_dict, coord_keys)

    def objective(x, coord_vals, scale, loglike0):
        values = ordered_values(
            x * scale, coord_vals, params, coord_keys, flux_key
        )
        return loglike0 - loglike(values, params, data_obj, model)

    def flux_loglike(flux, coord_vals):
        values = ordered_values(flux, coord_vals, params, coord_keys, flux_key)
        return loglike(values, params, data_obj, model)

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


def likelihood_grid(data_obj, model, samples_dict, batch_size=None):
    """Evaluate the log likelihood at every point of a parameter grid.

    Parameters
    ----------
    data_obj : OIData
        Data to fit.
    model : SourceModel or class
        Template model whose parameters at the paths in ``samples_dict`` are
        varied (e.g. a [System][virgil.models.System] with paths such as
        ``"comp.dra"``), or a model class called with ``samples_dict``'s keys
        as keyword arguments (e.g. ``BinaryModelCartesian``).
    samples_dict : dict[str, array-like]
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
        ``tuple(len(v) for v in samples_dict.values())``. Axis ``k`` follows
        the ``k``-th key (``indexing="ij"``), so for
        ``{"dra", "ddec", "flux"}`` axis 0 is ``dra``: transpose a 2D slice
        before showing it as an image with North up.
    """
    params = tuple(samples_dict.keys())
    check_flux_axes(samples_dict)
    return _likelihood_grid(
        data_obj,
        model,
        samples_dict,
        params=params,
        batch_size=batch_size_or_default(batch_size, data_obj),
    )


@eqx.filter_jit
def _likelihood_grid(data_obj, model, samples_dict, params, batch_size):
    """Jitted implementation of [`likelihood_grid`][virgil.grid_fit.likelihood_grid]."""

    vals_vec, grid_shape = meshgrid_vectors(samples_dict, params)

    return map_points(
        lambda values: loglike(values, params, data_obj, model),
        vals_vec,
        batch_size=batch_size,
    ).reshape(grid_shape)


_OPTIMIZED_PARAMS_DOC = """
    Parameters
    ----------
    data_obj : OIData
        Data to fit.
    model : SourceModel or class
        Template model whose parameters at the paths in ``samples_dict`` are
        varied (e.g. a [System][virgil.models.System] with paths such as
        ``"comp.dra"``), or a model class called with ``samples_dict``'s keys
        as keyword arguments (e.g. ``BinaryModelCartesian``).
    samples_dict : dict[str, array-like]
        Grid axes, as a mapping from parameter name or path to 1D values
        (e.g. ``dra``/``ddec`` in milliarcseconds and ``flux`` as a
        companion/primary flux ratio). The output has one axis per
        coordinate key (every key except ``flux_param``), in this order;
        the flux axis only sets the optimizer's starting points.
    flux_param : str, optional
        The key of ``samples_dict`` holding the flux optimized at each grid
        position, e.g. ``"comp.flux"``. By default, the one key whose last
        part is ``flux``.
    batch_size : int, optional
        Number of grid points evaluated at once. By default, enough for
        about 2**20 model visibilities on a CPU and 2**23 on other backends
        (GPU, TPU), and at least 256. Larger can be faster for small data;
        smaller bounds memory for large models.
"""


def optimized_likelihood_grid(
    data_obj, model, samples_dict, flux_param=None, batch_size=None
):
    params, coord_keys, flux_key = resolve_grid_keys(samples_dict, flux_param)
    _, best_loglike, success = _optimize_flux_grid(
        data_obj,
        model,
        samples_dict,
        params=params,
        coord_keys=coord_keys,
        flux_key=flux_key,
        batch_size=batch_size_or_default(batch_size, data_obj),
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


def optimized_flux_grid(
    data_obj, model, samples_dict, flux_param=None, batch_size=None
):
    params, coord_keys, flux_key = resolve_grid_keys(samples_dict, flux_param)
    best_flux, _, success = _optimize_flux_grid(
        data_obj,
        model,
        samples_dict,
        params=params,
        coord_keys=coord_keys,
        flux_key=flux_key,
        batch_size=batch_size_or_default(batch_size, data_obj),
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


@eqx.filter_jit
def _linear_flux_grid(
    data_obj,
    model,
    samples_dict,
    params,
    coord_keys,
    flux_key,
    batch_size,
    n_iter,
    prior,
):
    """Jitted implementation of [`linear_flux_grid`][virgil.grid_fit.linear_flux_grid]."""
    coords, shape = coordinate_points(samples_dict, coord_keys)

    def residuals(flux, coord_vals):
        values = ordered_values(flux, coord_vals, params, coord_keys, flux_key)
        return whitened_residuals(build_model(model, params, values), data_obj)

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
            return flux, sigma
        # The likelihood is exp(-curvature (f - flux)^2 / 2) up to a constant
        # in the linear model about the final point. With a N(mean, sd^2)
        # prior the precisions add (Luger et al. 2017).
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
    out = tuple(o.reshape(shape) for o in out)
    flux, sigma = out[:2]
    if prior is None:
        return flux, sigma, flux / sigma
    return {
        "flux": flux,
        "flux_error": sigma,
        "snr": flux / sigma,
        "posterior_mean": out[2],
        "posterior_sd": out[3],
        "log_bayes_factor": out[4],
    }


def linear_flux_grid(
    data_obj,
    model,
    samples_dict,
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
    data_obj : OIData
        Data to fit.
    model : SourceModel or class
        Template model whose parameters at the paths in ``samples_dict`` are
        varied (e.g. a [System][virgil.models.System] with paths such as
        ``"comp.dra"``), or a model class called with ``samples_dict``'s keys
        as keyword arguments (e.g. ``BinaryModelCartesian``). At zero flux
        it must reduce to the primary alone (flux 1).
    samples_dict : dict[str, array-like]
        Grid axes, as for
        [`optimized_flux_grid`][virgil.grid_fit.optimized_flux_grid]:
        ``dra``/``ddec`` in milliarcseconds, plus a flux key, whose values
        are ignored (the flux is solved for, at ``f = 0``), but which must
        be present to name the parameter. The output has one axis per
        coordinate key (every key except ``flux_param``), in this order.
    flux_param : str, optional
        The key of ``samples_dict`` holding the flux, e.g. ``"comp.flux"``.
        By default, the one key whose last part is ``flux``.
    batch_size : int, optional
        Number of grid points evaluated at once. By default, enough for
        about 2**20 model visibilities on a CPU and 2**23 on other backends
        (GPU, TPU), and at least 256.
    n_iter : int, optional
        Number of Gauss–Newton refinement steps after the first
        linearisation at ``f = 0`` (default 0, the closed-form result).
    prior : tuple of float, optional
        ``(mean, sd)`` of a Gaussian prior on the flux ratio (same units as
        ``flux``). If given, the result is a dict, with the posterior and
        the marginal-likelihood detection map added (see Returns). With
        ``P = g . g + 1 / sd**2`` (``g`` the final whitened derivative) the
        posterior is Gaussian with mean ``(g . (g f_hat) + mean / sd**2) / P``
        and sd ``P ** -0.5``, and the log Bayes factor against ``f = 0``
        is the closed-form Gaussian evidence ratio

        ``log B = -0.5 log(sd**2 P) + (g.g f_hat + mean/sd**2)**2 / (2P)
        - mean**2 / (2 sd**2)``

        (Luger, Foreman-Mackey & Hogg 2017, arXiv:1710.11136), so
        ``log B > 0`` favours a companion at that pixel. These hold in the
        linear model about the final linearisation point, i.e. exactly
        only where the residuals are linear in ``f`` over the posterior
        (``f`` much smaller than 1, or after enough ``n_iter`` for the
        point to sit near the posterior); the position is not marginalised.

    Returns
    -------
    flux : array-like
        Best-fit flux ratio (companion/primary), unconstrained in sign, with
        one axis per coordinate key (axis 0 is the first, e.g. ``dra``).
        With ``prior``, all outputs come as a dict with keys ``flux``,
        ``flux_error``, ``snr``, ``posterior_mean``, ``posterior_sd`` and
        ``log_bayes_factor`` instead of the tuple.
    flux_error : array-like
        One-sigma uncertainty on ``flux``, same shape, NaN where the model
        does not depend on the flux.
    snr : array-like
        ``flux / flux_error``, same shape: the detection significance map.

    Examples
    --------
    >>> grid = {
    ...     "dra": jnp.linspace(-300.0, 300.0, 61),
    ...     "ddec": jnp.linspace(-300.0, 300.0, 61),
    ...     "flux": jnp.array([1e-3]),  # ignored: only names the parameter
    ... }
    >>> flux, flux_error, snr = linear_flux_grid(
    ...     data, BinaryModelCartesian, grid
    ... )  # doctest: +SKIP
    >>> i, j = jnp.unravel_index(jnp.nanargmax(snr), snr.shape)  # doctest: +SKIP
    """
    params, coord_keys, flux_key = resolve_grid_keys(samples_dict, flux_param)
    return _linear_flux_grid(
        data_obj,
        model,
        samples_dict,
        params=params,
        coord_keys=coord_keys,
        flux_key=flux_key,
        batch_size=batch_size_or_default(batch_size, data_obj),
        n_iter=int(n_iter),
        prior=None
        if prior is None
        else tuple(jnp.asarray(float(x)) for x in prior),
    )


def laplace_flux_uncertainty_grid(
    data_obj, model, samples_dict, flux=None, flux_param=None, batch_size=None
):
    """Laplace uncertainty of the flux at every grid position.

    At each position the coordinates are held fixed and the uncertainty is
    the inverse square root of the curvature of the negative log likelihood
    along the flux.

    Parameters
    ----------
    data_obj : OIData
        Data to fit.
    model : SourceModel or class
        Template model or model class, as for :func:`likelihood_grid`.
    samples_dict : dict[str, array-like]
        Grid axes, as for :func:`optimized_flux_grid`. The output has one
        axis per coordinate key (every key except the flux), in this order.
    flux : array-like, optional
        Flux at which to evaluate the curvature, with one axis per
        coordinate key. By default this is the best fit from
        :func:`optimized_flux_grid`, which is also the mean that
        [`ruffio_upperlimit`][virgil.limits.ruffio_upperlimit] expects;
        pass it if you have already computed it.
    flux_param : str, optional
        The key of ``samples_dict`` holding the flux. By default, the one key
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
    params, coord_keys, flux_key = resolve_grid_keys(samples_dict, flux_param)
    if flux is None:
        flux = optimized_flux_grid(
            data_obj,
            model,
            samples_dict,
            flux_param=flux_key,
            batch_size=batch_size,
        )
    return _laplace_flux_uncertainty_grid(
        jnp.asarray(flux),
        data_obj,
        model,
        samples_dict,
        params=params,
        coord_keys=coord_keys,
        flux_key=flux_key,
        batch_size=batch_size_or_default(batch_size, data_obj),
    )


@eqx.filter_jit
def _laplace_flux_uncertainty_grid(
    flux_values,
    data_obj,
    model,
    samples_dict,
    params,
    coord_keys,
    flux_key,
    batch_size,
):
    """Jitted implementation of :func:`laplace_flux_uncertainty_grid`."""
    coords, shape = coordinate_points(samples_dict, coord_keys)

    def sigma(flux, coord_vals):
        values = jnp.stack(
            ordered_values(flux, coord_vals, params, coord_keys, flux_key)
        )
        return laplace_parameter_uncertainty(
            values=values,
            params=params,
            data_obj=data_obj,
            model=model,
            target_param=flux_key,
        )

    return map_points(
        sigma, flux_values.reshape(-1), coords, batch_size=batch_size
    ).reshape(shape)


def best_grid_point(loglike_grid, samples_dict):
    """Return the grid point with the highest log likelihood.

    Parameters
    ----------
    loglike_grid : array-like
        Output of [`likelihood_grid`][virgil.grid_fit.likelihood_grid] for ``samples_dict``, with one axis
        per key. NaNs are ignored.
    samples_dict : dict[str, array-like]
        The grid axes used to compute ``loglike_grid``.

    Returns
    -------
    dict[str, float]
        ``{name: value}`` at the maximum, in the order of ``samples_dict``.
    """
    shape = jnp.shape(loglike_grid)
    if len(shape) != len(samples_dict):
        raise ValueError(
            f"loglike_grid has {len(shape)} axes but samples_dict has "
            f"{len(samples_dict)} keys; pass the full likelihood_grid output."
        )
    index = np.unravel_index(int(jnp.nanargmax(loglike_grid)), shape)
    return {
        key: float(values[i])
        for (key, values), i in zip(samples_dict.items(), index)
    }
