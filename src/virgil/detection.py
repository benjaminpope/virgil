"""Detection statistics, null simulators and injection-recovery for ROC curves.

[`detection_statistics`][virgil.detection.detection_statistics] reduces a
companion search over a grid to ``delta_chi2``, ``log_bayes_factor`` and
``max_snr``; it is traceable in the data, so it compiles once under
``jax.lax.map``. [`gaussian_null`][virgil.detection.gaussian_null] and
[`bootstrap_null`][virgil.detection.bootstrap_null] simulate the null,
[`injection_recovery`][virgil.detection.injection_recovery] runs the statistics
on null and injected draws, and its [`DetectionMC`][virgil.detection.DetectionMC]
gives false-alarm probabilities, thresholds, ROC curves and contrast curves.

See [Detection statistics](../detection_statistics.md) and the
[Detection ROC curves](../detection_roc.md) tutorial.
"""

import dataclasses
import hashlib
import json
import math
import types
import warnings

import equinox as eqx
import jax
import jax.numpy as np
import numpy as onp

from ._deprecate import old_order
from ._grid import (
    batch_size_or_default,
    coordinate_points,
    ordered_values,
    resolve_grid_keys,
    warn_unconverged,
)
from ._utils import FWHM_PER_SIGMA, concrete, wrap_phase
from .grid_fit import (
    _best_grid_flux,
    _laplace_flux_uncertainty_grid,
    _likelihood_grid,
    _refine_flux_grid,
)
from .likelihood import build_model, loglike, whitened_residuals
from .limits import nsigma
from ._deprecate import external_stacklevel


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


@old_order("data", "model", "grid", data="data")
def detection_statistics(
    model, data, grid, *, flux_param=None, batch_size=None
):
    """Detection statistics of a companion search over a grid.

    The null hypothesis is ``model`` with the companion flux at zero; see
    [Detection statistics](../detection_statistics.md#statistics).

    Parameters
    ----------
    model : SourceModel or class
        Companion model, as for
        [`likelihood_grid`][virgil.grid_fit.likelihood_grid].
    data : OIData
        Data to search; may be traced (a simulation built under ``jax.lax.map``).
    grid : dict[str, array-like]
        Coordinate axes (``dra``/``ddec`` in mas, or ``sep`` in mas and ``pa``
        in degrees) and a non-negative, usually log-spaced flux axis, which
        must [resolve the peak](../detection_statistics.md#grid-resolution).
    flux_param, batch_size : optional
        Key of ``grid`` holding the flux (default: the key ending in ``flux``),
        and grid points evaluated at once, as for ``likelihood_grid``.

    Returns
    -------
    dict[str, jax.Array]
        Scalars ``delta_chi2`` (profile likelihood ratio, flux >= 0),
        ``log_bayes_factor`` (companion somewhere on the grid against none),
        ``max_snr`` (best flux over its Laplace uncertainty), one entry per
        ``grid`` key (where ``delta_chi2`` peaks) and ``flux_peak_steps``
        (peak FWHM over flux spacing; below about 2 the Bayes factor is
        inaccurate).

    Warns
    -----
    RuntimeWarning
        With concrete data, if the flux axis does not resolve the peak, the
        best flux lies above it, or the optimizer did not converge. Nothing is
        checked under ``jit`` or ``lax.map``: test ``flux_peak_steps`` yourself.

    Examples
    --------
    >>> stats = jax.lax.map(
    ...     lambda key: detection_statistics(
    ...         model, template.with_model(null_scene, key=key), grid
    ...     ),
    ...     jax.random.split(jax.random.PRNGKey(0), 100),
    ... )  # doctest: +SKIP
    """
    params, coord_keys, flux_key = _resolve_keys(grid, flux_param)
    stats, success = _detection_statistics(
        data,
        model,
        grid,
        params=params,
        coord_keys=coord_keys,
        flux_key=flux_key,
        batch_size=batch_size_or_default(batch_size, data),
    )
    # the checks that need numbers
    steps = concrete(stats["flux_peak_steps"])
    if steps is None:  # caveat: traced under jit, so none of these checks run
        return stats
    warn_unconverged(success, "detection_statistics")
    flux = float(concrete(stats[flux_key]))
    top = float(onp.max(onp.asarray(grid[flux_key])))
    if flux > top:
        warnings.warn(
            f"detection_statistics(): the best flux {flux:.3g} lies above "
            f"the flux axis (max {top:.3g}), so log_bayes_factor misses the "
            "likelihood peak; extend the axis.",
            RuntimeWarning,
            stacklevel=external_stacklevel(),
        )
    elif float(steps) < MIN_PEAK_STEPS:
        warnings.warn(
            "detection_statistics(): the flux axis does not resolve the "
            f"likelihood peak ({float(steps):.2g} steps across its FWHM, "
            f"under {MIN_PEAK_STEPS:g}), so log_bayes_factor is "
            "inaccurate; refine the flux axis near the best flux.",
            RuntimeWarning,
            stacklevel=external_stacklevel(),
        )
    return stats


def local_nsigma(delta_chi2):
    """Local significance of ``delta_chi2``, in Gaussian sigma.

    The significance at one position fixed in advance:
    [`nsigma`][virgil.limits.nsigma] with one degree of freedom. It ignores the
    look-elsewhere effect, so for a search it overstates; see
    [Detection statistics](../detection_statistics.md#local).

    Source: Wilks (1938), Chernoff (1954), [references](../references.md#chernoff1954).

    Parameters
    ----------
    delta_chi2 : float or array-like
        Profile likelihood ratio from
        [`detection_statistics`][virgil.detection.detection_statistics].

    Returns
    -------
    array-like
        ``sqrt(delta_chi2)`` up to rounding (saturates near 13σ in float32).
    """
    delta_chi2 = np.maximum(np.asarray(delta_chi2, dtype=float), 0.0)
    return nsigma(delta_chi2, 1.0, 1)


def _resolve_keys(grid, flux_param):
    """Grid keys as ``resolve_grid_keys``, refusing statistic names."""
    params, coord_keys, flux_key = resolve_grid_keys(grid, flux_param)
    clash = set(params) & (set(STATISTICS) | {"flux_peak_steps"})
    if clash:
        raise ValueError(
            f"grid keys {sorted(clash)} clash with the names of the "
            "returned statistics."
        )
    return params, coord_keys, flux_key


@eqx.filter_jit
def _detection_statistics(
    data, model, grid, params, coord_keys, flux_key, batch_size
):
    """Jitted implementation of :func:`detection_statistics`.

    Returns ``(stats, converged)``, where ``converged`` is the flux
    optimizer's convergence at every position.
    """
    loglike_im = _likelihood_grid(data, model, grid, params, batch_size)
    grid_flux, grid_loglike = _best_grid_flux(
        loglike_im, grid, params, flux_key
    )
    opt_flux, opt_loglike, converged = _refine_flux_grid(
        data,
        model,
        grid,
        params,
        coord_keys,
        flux_key,
        batch_size,
        grid_flux,
        grid_loglike,
    )
    coords, shape = coordinate_points(grid, coord_keys)
    null_values = ordered_values(0.0, coords[0], params, coord_keys, flux_key)
    loglike0 = loglike(null_values, params, model, data)

    profile, profile_flux = _constrained_profile(
        grid_flux, grid_loglike, opt_flux, opt_loglike, loglike0
    )
    best = np.argmax(profile.reshape(-1))
    delta_chi2 = 2.0 * (profile.reshape(-1)[best] - loglike0)
    best_flux = profile_flux.reshape(-1)[best]

    # Grid-marginalized evidence ratio over the trapezoid-weighted prior.
    # A grid point whose evaluation failed (NaN) carries no evidence.
    log_ratio = np.where(np.isnan(loglike_im), -np.inf, loglike_im - loglike0)
    log_bayes_factor = jax.nn.logsumexp(
        log_ratio + _log_prior_weights(loglike_im.shape, log_ratio.dtype)
    )

    sigma = _laplace_flux_uncertainty_grid(
        opt_flux,
        data,
        model,
        grid,
        params,
        coord_keys,
        flux_key,
        batch_size,
    )
    # The unconstrained best flux: a negative flux gives a negative SNR.
    max_snr = np.nanmax(opt_flux / sigma)

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
    # spacing of the flux axis around the best flux (0 for a single point)
    axis = np.sort(np.asarray(grid[flux_key]).reshape(-1))
    if axis.size < 2:
        flux_step = np.zeros((), axis.dtype)
    else:
        i = np.clip(np.searchsorted(axis, best_flux), 1, axis.size - 1)
        flux_step = axis[i] - axis[i - 1]
    stats["flux_peak_steps"] = (
        FWHM_PER_SIGMA * sigma.reshape(-1)[best] / flux_step
    )
    return stats, converged


# deliberate: profile over positive-flux grid points only; test_constrained_profile_keeps_positive_grid_points feeds it a bare grid
def _constrained_profile(
    grid_flux, grid_loglike, opt_flux, opt_loglike, loglike0
):
    """Profile log likelihood and flux with flux >= 0 at every position.

    The refined optimum is used only where its flux is non-negative and it
    improves on the best grid point; a position whose candidate does not
    improve on the null gets the null's log likelihood and flux 0.
    """
    use_opt = (opt_flux >= 0.0) & (opt_loglike >= grid_loglike)
    alt_loglike = np.where(use_opt, opt_loglike, grid_loglike)
    alt_flux = np.where(use_opt, opt_flux, grid_flux)
    improves = (alt_flux > 0.0) & (alt_loglike > loglike0)
    return (
        np.where(improves, alt_loglike, loglike0),
        np.where(improves, alt_flux, 0.0),
    )


# deliberate: grid prior weights; test_grid_prior_weights_integrate_to_one checks they sum to one on a bare shape
def _log_prior_weights(shape, dtype):
    """Log trapezoid weights in the grid index, normalized to sum to 1.

    Along each axis the weights are 1 inside and ½ at the two ends (1 for
    an axis of one point); the grid's weight is their outer product.
    """
    log_w = np.zeros(shape, dtype)
    for axis, n in enumerate(shape):
        w = onp.ones(n)
        if n > 1:
            w[[0, -1]] = 0.5
        w = onp.log(w / w.sum())
        log_w = log_w + w.reshape((-1,) + (1,) * (len(shape) - axis - 1))
    return log_w


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


@old_order("template", "null_scene", data="template")
def gaussian_null(null_scene, template, *, error_scale=1.0):
    """A simulator of the null hypothesis with Gaussian noise.

    Each draw is ``template.with_model(scene, key=key)``: the scene's prediction
    plus Gaussian noise from the template's errors. See
    [Detection statistics](../detection_statistics.md#null).

    Parameters
    ----------
    null_scene : SourceModel
        The scene without a companion.
    template : OIData
        Sampling and errors to simulate; its values are not used.
    error_scale : float, optional
        True noise as a multiple of the template's errors, to test statistics
        under wrong errors. Static: a new value compiles anew.

    Returns
    -------
    callable
        ``simulate(key, scene=None) -> OIData``, observing ``scene`` (default
        ``null_scene``) with fresh noise per ``key``; an equinox Module,
        traceable in both.

    Examples
    --------
    >>> simulate = gaussian_null(BinaryModelCartesian(0, 0, 0), template)  # doctest: +SKIP
    >>> data = simulate(jax.random.PRNGKey(0))  # doctest: +SKIP
    """
    error_scale = float(error_scale)
    if not (onp.isfinite(error_scale) and error_scale >= 0.0):
        raise ValueError(
            f"error_scale must be finite and non-negative, not {error_scale}."
        )
    return _GaussianNull(template, null_scene, error_scale)


@old_order("data", "null_scene", data="data")
def rescale_errors(null_scene, data):
    """Scale the errors so that the null scene has χ²_r = 1.

    Visibility and phase errors are scaled separately, each by
    ``sqrt(χ² / n)`` of the whitened residuals of ``null_scene``; see
    [Detection statistics](../detection_statistics.md#null).

    Parameters
    ----------
    null_scene : SourceModel
        The scene without a companion, fitted to ``data``.
    data : OIData
        Data to rescale; concrete.

    Returns
    -------
    data : OIData
        A copy with ``d_vis`` and ``d_phi`` scaled (the factors are already
        applied); ``data.extras`` unchanged.
    factors : dict[str, float]
        ``{"vis": s_vis, "phi": s_phi}`` (``s_phi`` is 1 without phases).

    Raises
    ------
    ValueError
        If a block fits perfectly or not at all.
    """
    r = onp.asarray(whitened_residuals(null_scene, data))
    n_vis = int(onp.size(data.vis))
    n_phi = int(onp.size(data.phi))
    if data._phases_wrap and data.cp_noise is not None:
        rows_phi, dof_phi = data.cp_noise.size + n_phi, data.cp_noise.size
    else:
        rows_phi, dof_phi = n_phi, n_phi
    s_vis = math.sqrt(float(onp.sum(r[:n_vis] ** 2)) / n_vis) if n_vis else 1.0
    s_phi = 1.0
    if dof_phi:
        chi2_phi = float(onp.sum(r[n_vis : n_vis + rows_phi] ** 2))
        s_phi = math.sqrt(chi2_phi / dof_phi)
    for name, s in (("vis", s_vis), ("phi", s_phi)):
        if not (onp.isfinite(s) and s > 0.0):
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


@old_order("data", "null_scene", data="data")
def bootstrap_null(null_scene, data, *, method="sign_flip"):
    """A simulator of the null hypothesis by residual bootstrap.

    The residuals of ``data`` about ``null_scene`` are whitened, sign-flipped or
    resampled, re-coloured (keeping closure-phase correlations) and added to the
    simulated scene's prediction. The null must fit, and a companion in the data
    biases the null distribution upward; see
    [Detection statistics](../detection_statistics.md#null).

    Source: Efron (1979), [references](../references.md#efron1979).

    Parameters
    ----------
    null_scene : SourceModel
        The scene without a companion, fitted to ``data``.
    data : OIData
        The real data, concrete, without gains, closure-phase offsets or extras.
    method : {"sign_flip", "resample"}, optional
        A wild bootstrap keeping each residual's magnitude, or resampling with
        replacement.

    Returns
    -------
    callable
        ``simulate(key, scene=None) -> OIData``, as for
        [`gaussian_null`][virgil.detection.gaussian_null], with ``data``'s
        errors.
    """
    if method not in BOOTSTRAP_METHODS:
        raise ValueError(
            f"method must be one of {BOOTSTRAP_METHODS}, not {method!r}."
        )
    if data.vis_encoding != "split":
        raise ValueError(
            "bootstrap_null() supports visibility and phase observables "
            f"('split'), not {data.vis_encoding!r}."
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
    n_vis = int(onp.size(data.vis))
    n_phase = n_vis + int(onp.size(data.phi))
    resid = data.flatten_data()[0][:n_phase] - prediction[:n_phase]
    vis_white = resid[:n_vis] / data.d_vis
    phi = resid[n_vis:]
    if data._phases_wrap:
        phi = wrap_phase(phi)
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
        Fluxes relative to the primary (non-negative; 0 gives null draws).
    n_pa : int
        Position angles per (separation, flux), uniform in [0°, 360°).
    key : jax.Array or int
        Random key or integer seed.

    Returns
    -------
    dict[str, numpy.ndarray]
        ``dra``, ``ddec`` (mas, East and North positive) and ``flux``, each of
        length ``len(separations) * len(fluxes) * n_pa``, ordered by separation,
        flux, then draw.
    """
    sep = onp.asarray(separations, dtype=float).reshape(-1)
    flux = onp.asarray(fluxes, dtype=float).reshape(-1)
    n_pa = int(n_pa)
    if n_pa < 1:
        raise ValueError(f"n_pa must be positive; got {n_pa}.")
    if onp.any(sep < 0.0) or onp.any(flux < 0.0):
        raise ValueError("separations and fluxes must be non-negative.")
    shape = (sep.size, flux.size, n_pa)
    pa = jax.random.uniform(_as_key(key), shape, maxval=360.0)
    pa = onp.radians(onp.asarray(pa, dtype=float))
    sep_grid = onp.broadcast_to(sep[:, None, None], shape)
    return {
        "dra": (sep_grid * onp.sin(pa)).reshape(-1),
        "ddec": (sep_grid * onp.cos(pa)).reshape(-1),
        "flux": onp.broadcast_to(flux[None, :, None], shape).reshape(-1),
    }


@old_order("template", "null_scene", "model", "grid", "key", data="template")
def injection_recovery(
    model,
    null_scene,
    template,
    grid,
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

    Each draw simulates an observation (null, or with a companion injected) and
    runs [`detection_statistics`][virgil.detection.detection_statistics] on it
    in one compiled kernel; see
    [Detection statistics](../detection_statistics.md#injection).

    Compare: Mawet et al. (2014), whose small-sample false-alarm correction is
    not applied here, [references](../references.md#mawet2014).

    Parameters
    ----------
    model : SourceModel or class
        Companion model: injected at the injected position and flux, null
        draws at zero flux.
    null_scene, template : SourceModel, OIData
        The scene without a companion (it must equal ``model`` at zero flux),
        and the sampling and errors to simulate (the data for ``"bootstrap"``).
    grid, flux_param
        The search grid and its flux key, as for ``detection_statistics``.
    key, n_null : jax.Array or int, int
        Random key or integer seed, and the number of null draws (may be 0).
    injections : dict[str, array-like], optional
        Companions to inject, e.g. from
        [`injection_grid`][virgil.detection.injection_grid]: one equal-length
        array per grid key, by full name or last part (``"dra"``).
    noise : {"gaussian", "bootstrap"} or simulator, optional
        [`gaussian_null`][virgil.detection.gaussian_null],
        [`bootstrap_null`][virgil.detection.bootstrap_null], or a simulator
        built by either from ``template``.
    match_radius : float, optional
        In mas: an injection counts as detected only if its best position is
        within this of the injected one (needs position keys).
    draw_batch, chunk_size, batch_size : int, optional
        Draws evaluated together in ``jax.lax.map``; draws per kernel call
        (default min(64, draws)); grid points per search.
    progress : bool, optional
        Show a ``tqdm`` bar.

    Returns
    -------
    DetectionMC
        The statistics of every draw, the injections, and metadata.

    Examples
    --------
    >>> mc = injection_recovery(
    ...     BinaryModelCartesian, BinaryModelCartesian(0, 0, 0), template,
    ...     grid, 0, n_null=1000,
    ... )  # doctest: +SKIP
    """
    params, coord_keys, flux_key = _resolve_keys(grid, flux_param)
    # short names of the grid keys (their last dotted part); the outputs are
    # stored under ``_STORED``, ``best_<name>`` and (for the injected values)
    # ``<name>``, all of which must be distinct, or one would overwrite another
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
    n_null = int(n_null)
    if n_null < 0:
        raise ValueError(f"n_null must be non-negative; got {n_null}.")
    draw_batch = int(draw_batch)
    if draw_batch < 1:
        raise ValueError(f"draw_batch must be positive; got {draw_batch}.")
    if match_radius is not None:
        match_radius = float(match_radius)
        if not (onp.isfinite(match_radius) and match_radius >= 0.0):
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
    # ``(n, len(params))`` float32 injection values in ``params`` order
    if injections is None:
        inj_values = onp.zeros((0, len(params)), dtype=onp.float32)
    else:
        columns = {
            k: onp.asarray(
                _lookup(injections, k, names[k]), dtype=float
            ).ravel()
            for k in params
        }
        if len({c.size for c in columns.values()}) != 1:
            raise ValueError("The injection arrays must all have one length.")
        flux = [c for k, c in columns.items() if names[k] == "flux"][0]
        if onp.any(flux < 0.0):
            raise ValueError("Injected fluxes must be non-negative.")
        inj_values = onp.stack([columns[k] for k in params], axis=1).astype(
            onp.float32
        )
    n_inj = inj_values.shape[0]
    draw_batch, grid_batch = _batch_sizes(batch_size, template, draw_batch)
    if chunk_size is None:
        chunk = min(64, max(n_null, n_inj, 1))
    else:
        chunk = int(chunk_size)
        if chunk < 1:
            raise ValueError(f"chunk_size must be positive; got {chunk}.")
    chunk = -(-chunk // draw_batch) * draw_batch
    # JSON description of ``key``, to tell array jobs apart.
    if isinstance(key, (int, onp.integer)):
        seed = {"seed": int(key)}
    else:
        seed = {"key": onp.asarray(_raw_key(key)).reshape(-1).tolist()}
    base = _raw_key(_as_key(key))
    static = {
        "params": params,
        "coord_keys": coord_keys,
        "flux_key": flux_key,
        "batch_size": grid_batch,
        "draw_batch": draw_batch if draw_batch > 1 else None,
    }
    bar = None
    if progress and n_null + n_inj > 0:
        try:
            from tqdm.auto import tqdm

            bar = tqdm(
                total=-(-n_null // chunk) - (-n_inj // chunk),
                desc="injection_recovery",
                leave=False,
            )
        except ImportError:
            pass

    def run(stream, values):
        # Stream 0 is the null, 1 the injections; draw i folds in i, so
        # results do not depend on chunking.
        base_key = jax.random.fold_in(base, stream)
        out = []
        for start in range(0, values.shape[0], chunk):
            part = values[start : start + chunk]
            # An explicit chunk_size keeps one compiled shape (padding the
            # last chunk); by default the last chunk is compiled at its own
            # length rather than padded with discarded draws.
            width = chunk if chunk_size is not None else part.shape[0]
            padded = onp.zeros((width, len(params)), dtype=onp.float32)
            padded[: part.shape[0]] = part
            stats = _simulated_statistics(
                simulator,
                model,
                grid,
                base_key,
                onp.arange(start, start + width, dtype=onp.int32),
                padded,
                **static,
            )
            out.append(
                {k: onp.asarray(v)[: part.shape[0]] for k, v in stats.items()}
            )
            if bar is not None:
                bar.update(1)
        # per-chunk statistics concatenated; best positions as best_<name>
        gathered = {stat: _concat([c[stat] for c in out]) for stat in _STORED}
        for k in params:
            gathered[f"best_{names[k]}"] = _concat([c[k] for c in out])
        return gathered

    try:
        null = run(0, onp.zeros((n_null, len(params)), dtype=onp.float32))
        injected = run(1, inj_values)
    finally:
        if bar is not None:
            bar.close()
    for k in params:
        # The injected values at full precision.
        injected[names[k]] = (
            onp.zeros(0)
            if injections is None
            else onp.asarray(
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
            k: onp.asarray(v, dtype=float).reshape(-1).tolist()
            for k, v in grid.items()
        },
        "model": _describe(model),
        "null_scene": _describe(null_scene),
        "template": {  # shape and hash of the whole template
            "n_vis": int(onp.size(template.vis)),
            "n_phi": int(onp.size(template.phi)),
            "hash": _fingerprint(template),
        },
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
    grid,
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
            grid,
            params=params,
            coord_keys=coord_keys,
            flux_key=flux_key,
            batch_size=batch_size,
        )
        stats = dict(stats)
        stats["converged_fraction"] = np.mean(converged.astype(row.dtype))
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
    if isinstance(key, (int, onp.integer)):
        return jax.random.PRNGKey(int(key))
    return key


def _raw_key(key):
    """Raw uint32 key data (typed keys are unwrapped)."""
    if jax.dtypes.issubdtype(getattr(key, "dtype", None), jax.dtypes.prng_key):
        return jax.random.key_data(key)
    return np.asarray(key)


def _lookup(injections, key, name):
    if key in injections:
        return injections[key]
    if name in injections:
        return injections[name]
    raise KeyError(
        f"injections has no {key!r} (or {name!r}); it has "
        f"{sorted(injections)}."
    )


# deliberate: splits the grid batch size among draw batches; pinned by test_default_grid_batch_size_is_split_among_draw_batch
def _batch_sizes(batch_size, template, draw_batch):
    """``(draw_batch, grid batch size)`` for ``injection_recovery``.

    An explicit ``batch_size`` is used as given. Otherwise the default grid
    batch is split among the draws evaluated together, and ``draw_batch``
    is capped at that default, so that their product never exceeds it.
    """
    default = batch_size_or_default(batch_size, template)
    if batch_size is not None:
        return draw_batch, default
    draw_batch = min(draw_batch, default)
    return draw_batch, default // draw_batch


# deliberate: injection_recovery is already past the 120-line cap, so growing it fails code_shape
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
        return gaussian_null(null_scene, template)
    if noise == "bootstrap":
        return bootstrap_null(null_scene, template)
    raise ValueError(
        "noise must be 'gaussian', 'bootstrap' or a simulator from "
        f"gaussian_null or bootstrap_null, not {noise!r}."
    )


def _check_same_prediction(template, a, b, name_a, name_b, tol=1e-3):
    """Refuse two scenes whose predictions differ by more than tol σ."""
    errors = template.flatten_data()[1]
    diff = template.residuals(template.model(a), template.model(b)) / errors
    worst = float(onp.max(onp.abs(onp.asarray(diff))))
    if not worst <= tol:
        raise ValueError(
            f"{name_a} and {name_b} predict different data (by up to "
            f"{worst:.3g}σ). Null draws simulate the model at zero flux, "
            "the null hypothesis the statistics test, so the two must "
            "agree."
        )


def _concat(arrays):
    if not arrays:
        return onp.zeros(0)
    return onp.concatenate(arrays).astype(float)


class _Unfingerprintable(Exception):
    """An object whose content cannot be hashed reproducibly."""


def _fingerprint(obj):
    """Hash of a whole object (classes and module-level functions by name), or
    None if part of it cannot be hashed, so that results built on it are never
    merged.
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
    elif isinstance(obj, (jax.Array, onp.ndarray, onp.generic)):
        try:
            a = onp.ascontiguousarray(onp.asarray(obj))
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

    Made by [`injection_recovery`][virgil.detection.injection_recovery]; plain
    NumPy. Conventions are in
    [Detection statistics](../detection_statistics.md#result).

    Source: Fawcett (2006), [references](../references.md#fawcett2006).

    Attributes
    ----------
    null : dict[str, numpy.ndarray]
        Per null draw: the three statistics, ``flux_peak_steps``,
        ``converged_fraction`` and the best position and flux as ``best_<name>``
        (``name`` is the last dotted part of each grid key).
    injected : dict[str, numpy.ndarray]
        The same per injected draw, plus the injected values by name.
    meta : dict
        JSON-serializable: grid, fingerprints, noise model, ``match_radius``,
        draw counts, seeds and virgil version.
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
        return int(onp.size(self.null["delta_chi2"]))

    @property
    def n_injected(self):
        """Number of injected draws."""
        return int(onp.size(self.injected["delta_chi2"]))

    def _scores(self, part, stat):
        if stat not in STATISTICS:
            raise ValueError(
                f"stat must be one of {STATISTICS}, not {stat!r}."
            )
        x = onp.asarray(part[stat], dtype=float)
        return onp.where(onp.isnan(x), -onp.inf, x)

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
            sep = onp.asarray(part[f"{prefix}sep"], dtype=float)
            pa = onp.radians(onp.asarray(part[f"{prefix}pa"], dtype=float))
            return sep * onp.sin(pa), sep * onp.cos(pa)
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
            return onp.asarray(self.injected["sep"], dtype=float)
        return onp.hypot(*self._positions())

    def matched(self):
        """Whether each injection's best position is within match_radius.

        All True when ``meta["match_radius"]`` is None. Angular positions
        (``sep``, ``pa`` in degrees) are compared on the sky.
        """
        radius = self.meta.get("match_radius")
        if radius is None:
            return onp.ones(self.n_injected, dtype=bool)
        (x, y), (bx, by) = self._positions(), self._positions("best_")
        return onp.hypot(bx - x, by - y) <= radius

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
            ``fap = (k + 1) / (n + 1)`` for ``k`` of ``n`` null draws at or
            above ``value`` (never 0); ``lower`` and ``upper`` are the exact
            Clopper–Pearson bounds on P(null >= value).
        """
        from scipy import stats as sstats

        null = onp.sort(self._null_scores(stat))
        n = null.size
        value = onp.asarray(value, dtype=float)
        k = n - onp.searchsorted(null, value, side="left")
        alpha = 1.0 - float(confidence)
        with onp.errstate(invalid="ignore"):
            lower = onp.where(
                k > 0, sstats.beta.ppf(alpha / 2, k, n - k + 1), 0.0
            )
            upper = onp.where(
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
            False-alarm probability (e.g. 1.35e-3 for one-sided 3σ).
        n_boot : int, optional
            Bootstrap resamples of the null draws, for the error.
        seed : int, optional
            Seed of the bootstrap.

        Returns
        -------
        threshold, error : float
            The ``1 - fap`` quantile of the null draws and its bootstrap
            standard deviation.

        Warns
        -----
        RuntimeWarning
            If ``fap * n < 1``: the threshold is then about the largest null
            value and underestimates the true one.
        """
        null, value = self._threshold(stat, fap)
        q = 1.0 - float(fap)
        rng = onp.random.default_rng(seed)
        n_boot = int(n_boot)
        rows = max(1, _BOOT_VALUES // null.size)
        quantiles = []
        for start in range(0, n_boot, rows):
            boot = rng.choice(
                null, size=(min(rows, n_boot - start), null.size)
            )
            quantiles.append(onp.quantile(boot, q, axis=1))
        error = (
            float(onp.std(onp.concatenate(quantiles))) if n_boot else onp.nan
        )
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
                stacklevel=external_stacklevel(),
            )
        return null, float(onp.quantile(null, 1.0 - fap))

    def detected(self, stat, fap):
        """Whether each injection is detected at false-alarm probability fap.

        Its statistic exceeds :meth:`threshold` and, with ``match_radius``,
        its best position matches the injection.
        """
        return self._detected(stat, self._threshold(stat, fap)[1])

    def _detected(self, stat, threshold):
        scores = self._scores(self.injected, stat)
        return (scores > threshold) & self.matched()

    def roc(self, stat, flux=None, sep_bin=None):
        """ROC curve: true- against false-positive rate over thresholds.

        Parameters
        ----------
        stat : str
            The statistic.
        flux : float or (float, float), optional
            Only injections of this flux (to 1e-6), or with ``lo <= flux < hi``.
        sep_bin : (float, float), optional
            Only injections with ``lo <= separation < hi`` (mas).

        Returns
        -------
        fpr, tpr, thresholds : numpy.ndarray
            At each threshold (``+inf``, then every distinct statistic value,
            decreasing): the fraction of null draws and of selected injections
            (matched, with ``match_radius``) at or above it.
        """
        null = onp.sort(self._null_scores(stat))
        keep = onp.ones(self.n_injected, dtype=bool)
        if flux is not None:
            f = self.injected["flux"]
            if onp.ndim(flux) == 0:
                keep &= onp.isclose(f, float(flux), rtol=1e-6, atol=0.0)
            else:
                lo, hi = flux
                keep &= (f >= lo) & (f < hi)
        if sep_bin is not None:
            lo, hi = sep_bin
            sep = self.separations()
            keep &= (sep >= lo) & (sep < hi)
        if not onp.any(keep):
            raise ValueError("No injections match the selection.")
        scores = self._scores(self.injected, stat)[keep]
        hits = onp.sort(scores[self.matched()[keep]])
        values = onp.unique(onp.concatenate([null, scores]))[::-1]
        thresholds = onp.concatenate([[onp.inf], values])
        fpr = 1.0 - onp.searchsorted(null, thresholds, "left") / null.size
        tpr = (hits.size - onp.searchsorted(hits, thresholds, "left")) / (
            scores.size
        )
        return fpr, tpr, thresholds

    def auc(self, stat, flux=None, sep_bin=None):
        """Area under the :meth:`roc` curve (0.5 for no discrimination)."""
        fpr, tpr, _ = self.roc(stat, flux=flux, sep_bin=sep_bin)
        return float(onp.trapezoid(tpr, fpr))

    def completeness(self, stat, fap, sep_bins=None, flux_bins=None):
        """Detection fraction of the injections by separation and flux.

        Parameters
        ----------
        stat : str
            The statistic.
        fap : float
            False-alarm probability of the detection threshold.
        sep_bins, flux_bins : array-like, optional
            Bin edges (mas, and flux); by default each distinct injected value
            (six significant figures) is a bin, which suits
            [`injection_grid`][virgil.detection.injection_grid].

        Returns
        -------
        dict
            ``completeness`` (n_sep × n_flux, NaN where empty), ``n`` (injections
            per bin), ``sep`` and ``flux`` (distinct values or bin centres;
            geometric for positive flux bins) and ``threshold``.
        """
        threshold = self._threshold(stat, fap)[1]
        detected = self._detected(stat, threshold).astype(float)
        sep_idx, sep = _bin(self.separations(), sep_bins, geometric=False)
        flux_idx, flux = _bin(self.injected["flux"], flux_bins, True)
        shape = (sep.size, flux.size)
        ok = (sep_idx >= 0) & (flux_idx >= 0)
        flat = onp.ravel_multi_index((sep_idx[ok], flux_idx[ok]), shape)
        size = sep.size * flux.size
        n = onp.bincount(flat, minlength=size).reshape(shape)
        hits = onp.bincount(flat, detected[ok], minlength=size).reshape(shape)
        with onp.errstate(invalid="ignore", divide="ignore"):
            fraction = onp.where(n > 0, hits / n, onp.nan)
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

        Per separation bin of :meth:`completeness`, the detection fraction is
        made non-decreasing in flux and interpolated in log flux to where it
        first reaches ``completeness``.

        Parameters
        ----------
        stat : str
            The statistic.
        fap : float
            False-alarm probability of the detection threshold.
        completeness : float, optional
            Target detection fraction.
        sep_bins, flux_bins : array-like, optional
            As for :meth:`completeness`.

        Returns
        -------
        sep, flux : numpy.ndarray
            Separations (mas) and flux relative to the primary (convert with
            [`flux_to_contrast`][virgil.limits.flux_to_contrast]); NaN where the
            positive injected fluxes do not bracket the target.
        """
        target = float(completeness)
        result = self.completeness(stat, fap, sep_bins, flux_bins)
        flux = result["flux"]
        out = onp.full(result["sep"].size, onp.nan)
        for i, row in enumerate(result["completeness"]):
            ok = (flux > 0.0) & onp.isfinite(row)
            f, c = flux[ok], onp.maximum.accumulate(row[ok])
            above = onp.flatnonzero(c >= target)
            if above.size == 0:
                continue
            j = above[0]
            if j == 0:
                if c[0] == target:
                    out[i] = f[0]
                continue
            t = (target - c[j - 1]) / (c[j] - c[j - 1])
            out[i] = onp.exp(onp.log(f[j - 1]) + t * onp.log(f[j] / f[j - 1]))
        return result["sep"], out

    def save(self, path):
        """Write to ``path``, an ``.npz`` with the metadata as JSON.

        ``meta["format"]`` is the file-format version (1).
        """
        arrays = {f"null__{k}": onp.asarray(v) for k, v in self.null.items()}
        for k, v in self.injected.items():
            arrays[f"injected__{k}"] = onp.asarray(v)
        arrays["meta"] = onp.array(json.dumps(self.meta))
        onp.savez(path, **arrays)

    @classmethod
    def load(cls, path):
        """Read a file written by :meth:`save`.

        A file without ``meta["format"]`` is read as version 1. Any other
        version than the one this virgil writes raises a ``ValueError``,
        rather than being read in a layout it may not have.
        """
        with onp.load(path, allow_pickle=False) as f:
            meta = json.loads(str(f["meta"]))
            version = meta.setdefault("format", 1)
            if version != _FORMAT:
                raise ValueError(
                    f"{path} is a DetectionMC file of format {version!r}; "
                    f"this virgil reads format {_FORMAT}. Upgrade virgil, "
                    "or regenerate the file."
                )
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
            E.g. one per array job. They must agree on grid, model, null scene,
            template, noise and ``match_radius``, with distinct seeds.

        Returns
        -------
        DetectionMC
            All draws, with every seed in ``meta``.

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
                    stacklevel=external_stacklevel(),
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
                k: onp.concatenate([getattr(r, part)[k] for r in results])
                for k in keys
            }

        meta = dict(first)
        meta["seeds"] = seeds
        meta["n_null"] = sum(r.n_null for r in results)
        meta["n_injected"] = sum(r.n_injected for r in results)
        return cls(null=merge("null"), injected=merge("injected"), meta=meta)


def _bin(values, edges, geometric):
    """Bin indices (-1 outside) and labels: distinct values or centres."""
    if edges is None:
        # distinct values, to 6 significant figures (0 stays 0)
        x = onp.asarray(values, dtype=float)
        with onp.errstate(divide="ignore"):
            mag = onp.where(x == 0.0, 0.0, onp.floor(onp.log10(onp.abs(x))))
        scale = 10.0 ** (5 - mag)
        labels, index = onp.unique(
            onp.round(x * scale) / scale, return_inverse=True
        )
        return index.reshape(-1), labels
    edges = onp.asarray(edges, dtype=float)
    index = onp.searchsorted(edges, values, side="right") - 1
    index = onp.where((index >= 0) & (index < edges.size - 1), index, -1)
    lo, hi = edges[:-1], edges[1:]
    labels = 0.5 * (lo + hi)
    if geometric:
        positive = (lo > 0.0) & (hi > 0.0)
        labels = onp.where(positive, onp.sqrt(onp.abs(lo * hi)), labels)
    return index, labels
