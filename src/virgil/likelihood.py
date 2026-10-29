"""Likelihoods of source models given interferometric data.

A model is given either as a template [`SourceModel`][virgil.models.SourceModel],
whose parameters at dot-separated zodiax paths (e.g. ``"comp.flux"``) are
replaced, or as a class/callable called with the parameters as keyword
arguments (see [`build_model`][virgil.likelihood.build_model]).

Every likelihood, grid, limit and fit goes through one residual vector,
[`whitened_residuals`][virgil.likelihood.whitened_residuals]: the
residuals divided by their uncertainties, with unprojected phases measured
as a chord, 2 sin(Δ/2), so that the likelihood is smooth where phases
wrap at ±π. Closure phases from four or more telescopes are correlated:
their independent combinations are whitened together, using the sine
sin Δ (not the chord) of each residual, with a periodic penalty
2 sin²(Δ/2) / σ added per closure phase. Both terms repeat every 2π and
are smooth, so the likelihood is continuous everywhere.
"""

import os
import re
import warnings

import jax
import jax.numpy as np
import numpy as onp
from jax.scipy.special import i0e

from ._deprecate import old_order, renamed
from ._flat import flat_sampled
from ._utils import (
    _per_dataset,
    _reference,
    concrete,
    inflate_errors,
    is_flux_param,
)
from .angles import is_angle_vector, vector_angle, vector_site
from .gains import GAIN_GROUPS, OFFSET_GROUPS
from .models import SourceModel
from ._deprecate import external_stacklevel


def _gain_jacobian(data, vis_prediction):
    """dObs/dlog|V| of the model's visibility observables."""
    if data.vis_mode == "v2":
        return 2.0 * vis_prediction
    if data.vis_mode == "amp":
        return vis_prediction
    return np.ones_like(vis_prediction)


def _whiten(
    data, prediction, reference, errors, gain_terms=None, offset_terms=None
):
    """Whitened residuals, and the errors that normalize their likelihood.

    Returns ``(whitened, errors_out)``; see ``_whiten_with_log_norm``.
    """
    return _whiten_with_log_norm(
        data, prediction, reference, errors, gain_terms, offset_terms
    )[:2]


def _whiten_with_log_norm(
    data,
    prediction,
    reference,
    errors,
    gain_terms=None,
    offset_terms=None,
    frozen=False,
    split_norm=False,
):
    """Whitened residuals, effective errors, and the marginal log-normalizer.

    Returns ``(whitened, errors_out, log_norm)``. ``log_norm`` is the part of
    ``Σ log errors_out`` from marginalized linear nuisances (gains, closure
    offsets, extra blocks), which may depend on the model; the rest of the
    normalizer depends only on the data and error terms. Residuals are
    ``(prediction - reference) / errors``, except for gains, unprojected phases
    and correlated closure phases (chords, sines and the periodic penalty), which
    follow ``docs/conventions.md`` (Observables; Closure phases with four or more
    telescopes; Likelihood normalization). Extra observables follow the phases,
    each block whitened by itself. With ``frozen=True`` the diagonal ``errors``
    pass through ``jax.lax.stop_gradient`` for ``_curvature_rows``. With
    ``split_norm=True`` ``log_norm`` is instead ``[gains', offsets']`` (the
    visibility and phase parts; extra blocks are left out), as the
    scale-marginalized surface needs them block by block.
    """
    prediction = np.asarray(prediction)
    resid = prediction - np.asarray(reference)
    errors = np.asarray(errors)
    if frozen:
        errors = jax.lax.stop_gradient(errors)
        gain_terms = jax.lax.stop_gradient(gain_terms)
        offset_terms = jax.lax.stop_gradient(offset_terms)
    n_vis = np.asarray(data.vis).size
    n_phase = n_vis + np.asarray(data.phi).size
    vis, vis_errors = resid[:n_vis] / errors[:n_vis], errors[:n_vis]
    vis_norm = np.zeros((), errors.dtype)
    if data.gains is not None:  # marginalize the visibility gains
        gains = data.gains
        jacobian = _gain_jacobian(data, prediction[:n_vis]) / vis_errors
        vis, extra = gains.whiten(vis, jacobian, gains.widths_for(gain_terms))
        vis_errors, vis_norm = vis_errors * np.exp(extra), np.sum(extra)
    phase, phase_errors, phase_norm = _whiten_phases(
        data, resid[n_vis:n_phase], errors[n_vis:n_phase], offset_terms
    )
    whitened, effective = [vis, phase], [vis_errors, phase_errors]
    log_norm = vis_norm + phase_norm
    reference = np.asarray(reference)
    offset = n_phase
    for block in data.extras:
        # Extra observables (OI_FLUX, T3AMP, VISAMP, VISPHI) whiten their
        # own blocks: see virgil.observables.
        end = offset + int(block.data().size)
        w, e = block.whiten(
            prediction[offset:end], reference[offset:end], errors[offset:end]
        )
        whitened.append(w)
        effective.append(e)
        log_norm = log_norm + np.sum(np.log(e) - np.log(errors[offset:end]))
        offset = end
    if split_norm:
        log_norm = np.stack([vis_norm, phase_norm])
    return np.concatenate(whitened), np.concatenate(effective), log_norm


# deliberate: three exits (wrapped-off, von Mises, correlated with offsets) that flags would replace in _whiten_with_log_norm
def _whiten_phases(data, resid, errors, offset_terms=None):
    """Whitened phase residuals, their effective errors and the closure
    offsets' log normalization (see ``_whiten_with_log_norm``)."""
    zero = np.zeros((), errors.dtype)
    if not data._phases_wrap:
        return resid / errors, errors, zero
    if data.cp_noise is None:
        von_mises = np.sqrt(2.0 * np.pi) * i0e(1.0 / errors**2)
        return 2.0 * np.sin(0.5 * resid) / errors, von_mises, zero
    # Correlated closure phases mix their residuals, so each sign matters,
    # and a chord's sign flips under 2π. Whiten the (smooth, periodic) sines
    # and append the periodic penalty 2 sin²(Δ/2)/σ = (1 - cos Δ)/σ, which
    # removes the false minimum at Δ = π.
    whitened, whitened_errors = data.cp_noise.whiten(np.sin(resid), errors)
    offsets = data.phase_offsets
    log_norm = zero
    if offsets is not None:
        whitened, extra = offsets.whiten(
            data.cp_noise,
            whitened,
            errors,
            offsets.widths_for(offset_terms),
        )
        whitened_errors = whitened_errors * np.exp(extra)
        log_norm = np.sum(extra)
    penalty = 2.0 * np.sin(0.5 * resid) ** 2 / errors
    penalty_errors = np.full(penalty.shape, 1.0 / np.sqrt(2.0 * np.pi))
    return (
        np.concatenate([whitened, penalty]),
        np.concatenate([whitened_errors, penalty_errors.astype(errors.dtype)]),
        log_norm,
    )


def _gaussian_loglike(whitened, errors):
    """Gaussian log density of whitened residuals with uncertainties ``errors``."""
    return (
        -0.5 * np.sum(whitened**2)
        - np.sum(np.log(errors))
        - 0.5 * whitened.size * np.log(2.0 * np.pi)
    )


# Nuisance terms, as accepted by the likelihoods and the ``noise`` argument
# of ``fit`` and ``numpyro_model``: error inflation (``inflated_errors``),
# the widths of gains correlated across channels (``OIData.with_gains``) and
# of closure-phase offsets (``OIData.with_closure_offsets``), the
# wavelength scale (``OIData.with_wavelength_scale``) and the North angle
# (``OIData.with_north_angle``).
GAIN_TERMS = tuple(f"vis_gain_{group}" for group in GAIN_GROUPS)
OFFSET_TERMS = tuple(f"phi_offset_{group}" for group in OFFSET_GROUPS)
WAVEL_TERMS = ("wavel_scale", "wavel_offset")
NORTH_TERMS = ("north_angle",)
NOISE_TERMS = (
    (
        "vis_scale",
        "phi_scale",
        "vis_error_rel",
        "phi_error",
        "vis_error",
    )
    + GAIN_TERMS
    + OFFSET_TERMS
    + WAVEL_TERMS
    + NORTH_TERMS
)


@renamed()
def inflated_errors(
    data,
    prediction,
    vis_error_rel=None,
    phi_error=None,
    vis_scale=None,
    phi_scale=None,
    vis_error=None,
    *,
    where="model",
    combine="quadrature",
):
    """The data uncertainties, scaled and with extra terms in quadrature.

    The visibility errors become
    ``hypot(vis_scale σ, vis_error, vis_error_rel V)``, for the model
    visibility observable ``V``, and the phase errors
    ``hypot(phi_scale σ, phi_error)``. Terms left as ``None`` are not
    applied. Extra observables (``OIData.extras``) are left unchanged;
    give them floors with
    [`OIData.with_error_floor`][virgil.oidata.OIData.with_error_floor].

    Parameters
    ----------
    data : OIData
        Data whose uncertainties are inflated.
    prediction : array-like
        Model vector, e.g. from [`OIData.model`][virgil.oidata.OIData.model].
    vis_error_rel : float, optional
        Extra visibility error, as a fraction of the *model* visibility
        observable (e.g. of the model V² for squared visibilities): a
        calibration error.
    phi_error : float, optional
        Extra phase error in radians.
    vis_scale, phi_scale : float, optional
        Factors multiplying the visibility and phase uncertainties.
    vis_error : float, optional
        Extra absolute visibility error, in the units of the observable.
    where : {"model", "data"}, optional
        What ``vis_error_rel`` is relative to: the model (default, the
        right choice for a fitted term, as it does not reward the model for
        low data points) or the data (as error floors are).
    combine : {"quadrature", "max"}, optional
        Add the terms in quadrature (default), or take the largest (a
        floor, as ``with_error_floor`` does; both use
        ``_utils.inflate_errors``).

    Returns
    -------
    array-like
        Uncertainties matching [`flatten_data`][virgil.oidata.OIData.flatten_data].
    """
    values, errors = data.flatten_data()
    terms = (vis_error_rel, phi_error, vis_scale, phi_scale, vis_error)
    if all(term is None for term in terms):
        return errors
    projected = data.vis_mat is not None or data.phi_mat is not None
    added = (vis_error_rel, phi_error, vis_error)
    if projected and any(term is not None for term in added):
        raise ValueError(
            "Extra error terms are defined for the observed visibilities and "
            "phases, not for projected (vis_mat/phi_mat) observables; scale "
            "their errors with vis_scale/phi_scale instead."
        )
    if where not in ("model", "data"):
        raise ValueError(f"where must be 'model' or 'data', not {where!r}.")
    reference = values if where == "data" else np.asarray(prediction)
    n_vis = np.asarray(data.vis).size
    n_phase = n_vis + np.asarray(data.phi).size
    d_vis = inflate_errors(
        errors[:n_vis],
        reference[:n_vis],
        absolute=vis_error,
        relative=vis_error_rel,
        scale=vis_scale,
        combine=combine,
    )
    d_phi = inflate_errors(
        errors[n_vis:n_phase],
        absolute=phi_error,
        scale=phi_scale,
        combine=combine,
    )
    return np.concatenate([d_vis, d_phi, errors[n_phase:]])


def is_tied(spec):
    """Whether a ``noise`` entry is tied: a function of the sampled
    parameters (rather than a prior), e.g. one epoch's error scale drawn
    from a population (see
    [`hierarchical_scales`][virgil.priors.hierarchical_scales])."""
    return callable(spec) and not hasattr(spec, "log_prob")


def tied_log_prior(sites, values):
    """The summed ``log_prior(values)`` of the distinct tied ``noise``
    terms that have one, such as the population density of a centred
    [`hierarchical_scales`][virgil.priors.hierarchical_scales] member
    (a term used twice counts once)."""
    seen = []
    for spec, _, _ in sites.values():
        if is_tied(spec) and hasattr(spec, "log_prior") and spec not in seen:
            seen.append(spec)
    return sum((spec.log_prior(values) for spec in seen), np.zeros(()))


def noise_sites(noise, n_datasets):
    """Expand a ``noise`` specification into named sites.

    ``noise`` maps error-inflation terms (``NOISE_TERMS``) to priors and
    applies to every dataset, giving sites ``"noise.<term>"``; a list of such
    dicts, one per dataset, gives sites ``"noise[i].<term>"``. An entry may
    also be *tied*: a function of the dict of sampled parameter values
    (keyed like ``priors``) that returns the term's value, which then has
    no prior of its own (see [`is_tied`][virgil.likelihood.is_tied]),
    unless it has a ``log_prior(values)`` method (see
    [`tied_log_prior`][virgil.likelihood.tied_log_prior]).

    Returns
    -------
    dict
        ``{site: (prior, datasets, term)}``, ``datasets`` being the indices
        of the datasets the term applies to; ``prior`` is the function for
        a tied term.
    """
    if noise is None:
        return {}
    if isinstance(noise, dict):
        specs = [("noise", tuple(range(n_datasets)), noise)]
    else:
        noise = list(noise)
        if len(noise) != n_datasets:
            raise ValueError(
                f"noise has {len(noise)} entries for {n_datasets} datasets; "
                "pass one dict per dataset, or one dict for all of them."
            )
        specs = [(f"noise[{i}]", (i,), n) for i, n in enumerate(noise)]
    sites = {}
    for prefix, datasets, terms in specs:
        for term, prior in terms.items():
            if term not in NOISE_TERMS:
                raise ValueError(
                    f"Unknown noise term {term!r}; use one of {NOISE_TERMS}."
                )
            if is_tied(prior) or term in WAVEL_TERMS + NORTH_TERMS:
                # Tied to the parameters (no prior of its own), or not
                # errors: a scale near 1, an offset or an angle of either
                # sign.
                sites[f"{prefix}.{term}"] = (prior, datasets, term)
                continue
            lower = getattr(prior.support, "lower_bound", None)
            value = None if lower is None else concrete(lower)
            if value is None or onp.any(value < 0.0):
                raise ValueError(
                    f"The prior on noise term {term!r} must have "
                    "non-negative support. It is a scale parameter, so its "
                    "default (Jeffreys) prior is log-uniform on stated "
                    "bounds that contain its plausible values, e.g. "
                    "dist.LogUniform(0.1, 10.0) for vis_scale or phi_scale "
                    "(neutral value 1) and dist.LogUniform(1e-4, 0.3) for "
                    "added errors and widths."
                )
            sites[f"{prefix}.{term}"] = (prior, datasets, term)
    return sites


def noise_for(sites, values, index):
    """The error-inflation terms of dataset ``index``, from site values."""
    return {
        term: values[site]
        for site, (_, datasets, term) in sites.items()
        if index in datasets
    }


def _whitened_and_errors(model, data, noise):
    """Whitened residuals and effective errors (see ``_whiten``)."""
    return _whitened_errors_and_log_norm(model, data, noise)[:2]


def _prediction_and_errors(model, data, noise):
    """The model prediction and the diagonal errors of ``data``.

    Returns ``(prediction, errors, values, gain_terms, offset_terms)``: the
    prediction with the wavelength and North-angle terms applied, the errors of
    ``inflated_errors``, the data vector, and the gain and offset widths.
    """
    unknown = set(noise) - set(NOISE_TERMS)
    if unknown:
        raise TypeError(
            f"Unknown error terms {sorted(unknown)}; use {NOISE_TERMS}."
        )
    gain_terms = {k: v for k, v in noise.items() if k in GAIN_TERMS}
    if gain_terms and data.gains is None:
        raise ValueError(
            f"Error terms {sorted(gain_terms)} need gain modes: add them with "
            "OIData.with_gains."
        )
    offset_terms = {k: v for k, v in noise.items() if k in OFFSET_TERMS}
    if offset_terms and data.phase_offsets is None:
        raise ValueError(
            f"Error terms {sorted(offset_terms)} need closure-phase offsets: "
            "add them with OIData.with_closure_offsets."
        )
    wavel_terms = {
        k.removeprefix("wavel_"): v
        for k, v in noise.items()
        if k in WAVEL_TERMS
    }
    inflation = {
        k: v
        for k, v in noise.items()
        if k not in GAIN_TERMS
        and k not in OFFSET_TERMS
        and k not in WAVEL_TERMS
        and k not in NORTH_TERMS
    }
    observed = data
    if wavel_terms:
        data = data.with_wavelength_scale(**wavel_terms)
    if "north_angle" in noise:
        data = data.with_north_angle(noise["north_angle"])
    prediction = data.model(model)
    data = observed
    errors = inflated_errors(data, prediction, **inflation)
    values = data.flatten_data()[0]
    return prediction, errors, values, gain_terms, offset_terms


def _whitened_errors_and_log_norm(model, data, noise):
    prediction, errors, values, gain_terms, offset_terms = (
        _prediction_and_errors(model, data, noise)
    )
    return _whiten_with_log_norm(
        data, prediction, values, errors, gain_terms, offset_terms
    )


def _curvature_rows(model, data, noise):
    """The curvature rows of one dataset (see ``_Objective.curvature_rows``).

    Their Jacobian J gives the Gauss–Newton curvature JᵀJ of the negative log
    likelihood: *mean rows* (``_whiten_with_log_norm(frozen=True)``) plus, when
    error terms are fitted, *normalizer rows* ``√2 log σ_k`` for the diagonal
    errors of the visibility and phase observables. See
    ``docs/gauss_newton_mass.md``.
    """
    prediction, errors, values, gain_terms, offset_terms = (
        _prediction_and_errors(model, data, noise)
    )
    mean = _whiten_with_log_norm(
        data,
        prediction,
        values,
        errors,
        gain_terms,
        offset_terms,
        frozen=True,
    )[0]
    rows = [mean]
    root2 = np.sqrt(2.0)
    n_vis = np.asarray(data.vis).size
    n_phase = n_vis + np.asarray(data.phi).size
    resid = np.asarray(prediction) - values
    fitted = any(
        k not in GAIN_TERMS + OFFSET_TERMS + WAVEL_TERMS + NORTH_TERMS
        for k in noise
    )
    if fitted:
        rows.append(root2 * np.log(errors[:n_vis]))
        phase_errors = errors[n_vis:n_phase]
        if data._phases_wrap and data.cp_noise is not None:
            phase_errors = data.cp_noise.whiten(
                np.sin(resid[n_vis:n_phase]), phase_errors
            )[1]
        rows.append(root2 * np.log(phase_errors))
    if gain_terms:
        pred_vis = jax.lax.stop_gradient(prediction[:n_vis])
        err_vis = jax.lax.stop_gradient(errors[:n_vis])
        jacobian = _gain_jacobian(data, pred_vis) / err_vis
        extra_vis = data.gains.whiten(
            jax.lax.stop_gradient(resid[:n_vis]) / err_vis,
            jacobian,
            data.gains.widths_for(gain_terms),
        )[1]
        rows.append(root2 * extra_vis)
    if offset_terms:
        err_phi = jax.lax.stop_gradient(errors[n_vis:n_phase])
        whitened_phi = data.cp_noise.whiten(
            jax.lax.stop_gradient(np.sin(resid[n_vis:n_phase])), err_phi
        )[0]
        extra_phi = data.phase_offsets.whiten(
            data.cp_noise,
            whitened_phi,
            err_phi,
            data.phase_offsets.widths_for(offset_terms),
        )[1]
        rows.append(root2 * extra_phi)
    return np.concatenate(rows)


@renamed()
def whitened_residuals(model, data, **noise):
    """Residuals of a model divided by the data uncertainties.

    The one residual vector behind every likelihood in virgil: ``model_loglike``
    is ``-0.5 * sum(whitened_residuals**2)`` plus the normalization, and
    least-squares fits minimize its sum of squares. Visibility and projected-phase
    residuals are ``(model - data) / σ``; an unprojected phase residual Δ is the
    chord ``2 sin(Δ/2) / σ``; correlated closure phases are replaced by whitened
    independent combinations plus periodic penalty rows. See
    [conventions](../conventions.md#observables) and
    [closure phases with four or more telescopes](../conventions.md#closure-phases-with-four-or-more-telescopes).

    Parameters
    ----------
    model : SourceModel
        Model to evaluate.
    data : OIData
        Data to compare with.
    **noise
        Error terms (see [common arguments](../common_arguments.md#noise)).

    Returns
    -------
    array-like
        One dimensionless residual per independent observable, in the order of
        [`flatten_data`][virgil.oidata.OIData.flatten_data]; the length is
        [`n_residuals`][virgil.oidata.OIData.n_residuals].
    """
    return _whitened_and_errors(model, data, noise)[0]


@renamed()
def model_loglike(model, data, *, reject_unphysical=False, **noise):
    """Evaluate the log likelihood for an instantiated model object.

    This is ``-0.5 * sum(r**2)`` for the residuals ``r`` of
    [`whitened_residuals`][virgil.likelihood.whitened_residuals], plus each
    density's normalization (Gaussian, or von Mises for unprojected phases; see
    [likelihood normalization](../conventions.md#likelihood-normalization)).

    Parameters
    ----------
    model : SourceModel
        Model to evaluate.
    data : OIData
        Data to compare with.
    reject_unphysical : bool, optional
        If True, return ``-inf`` when
        [`is_physical`][virgil.models.SourceModel.is_physical] is false, e.g. for a
        negative flux. Works inside ``jax.jit``, as a hard prior boundary.
    **noise
        Error terms (see [common arguments](../common_arguments.md#noise)). The
        normalization uses the inflated errors; for data with gains it includes
        the log-determinant of the marginalized covariance.
    """
    whitened, errors = _whitened_and_errors(model, data, noise)
    logl = _gaussian_loglike(whitened, errors)
    if reject_unphysical:
        logl = np.where(model.is_physical(), logl, -np.inf)
    return logl


@old_order("params", "data", "model", data="data")
def joint_prediction(params, model, data):
    """Concatenate predictions for a parameter pytree and multiple data.

    ``model(params, index)`` defines which parameters are shared and which
    are specific to each observation.
    """
    return np.concatenate(
        [
            observation.model(model(params, index))
            for index, observation in enumerate(data)
        ]
    )


@renamed()
def joint_data(data):
    """Concatenate observed vectors in the same order as ``joint_prediction``."""
    return np.concatenate(
        [observation.flatten_data()[0] for observation in data]
    )


@renamed()
def joint_errors(data):
    """Concatenate uncertainty vectors in the same order as ``joint_prediction``."""
    return np.concatenate(
        [observation.flatten_data()[1] for observation in data]
    )


@old_order("params", "data", "model", data="data")
def joint_loglike(params, model, data, **options):
    """Sum independent Gaussian log likelihoods over multiple data.

    ``options`` (error terms and ``reject_unphysical``) are
    passed to [`model_loglike`][virgil.likelihood.model_loglike].
    """
    return sum(
        model_loglike(model(params, index), observation, **options)
        for index, observation in enumerate(data)
    )


def build_model(model, params, values):
    """Build a model from parameter names and values.

    ``model`` is either a class/callable, called as ``model(**dict(zip(params,
    values)))``, or a [`SourceModel`][virgil.models.SourceModel] instance used as a template whose
    leaves at the (dot-separated) paths ``params`` are replaced by ``values``.
    """
    if isinstance(model, SourceModel):
        return model.set(list(params), list(values))
    return model(**dict(zip(params, values)))


@old_order("values", "params", "data", "model", data="data")
def loglike(values, params, model, data, **options):
    """
    Gaussian log-likelihood of a model with the given parameter values, assuming Gaussian errors.

    Parameters
    ----------
    values : array-like
        Values of the model parameters.
    params : list
        List of parameter names.
    model : SourceModel or callable
        Template model whose parameters at the dot-separated paths ``params``
        are replaced by ``values``, or a class/callable called as
        ``model(**dict(zip(params, values)))`` (see [`build_model`][virgil.likelihood.build_model]).
    data : OIData
        Object containing the data to be fitted.
    **options
        Error terms and ``reject_unphysical``, passed to
        [`model_loglike`][virgil.likelihood.model_loglike].

    Returns
    -------
    float
        Log-likelihood value.
    """

    return model_loglike(build_model(model, params, values), data, **options)


def _check_positive_flux_prior(name, distribution):
    """Reject flux priors whose support includes negative values."""
    from numpyro.distributions import constraints

    support = distribution.support
    lower = getattr(support, "lower_bound", None)
    if lower is None:
        unbounded = support in (constraints.real, constraints.real_vector)
    else:
        value = concrete(lower)
        unbounded = value is not None and bool(onp.any(value < 0.0))
    if unbounded:
        raise ValueError(
            f"The prior on {name!r} allows negative values, but fluxes must "
            "be non-negative. Use a prior with non-negative support: "
            "dist.LogUniform(lo, hi) (scale invariant) by default, or "
            "dist.Uniform(0, ...) as the exception, when a flat flux "
            "prior is really what you mean."
        )


def _host_devices():
    """The number of local devices, without initializing the backend (by reading
    ``XLA_FLAGS`` first; several GPUs are not counted before the backend starts).
    """
    try:
        from jax._src import xla_bridge

        if xla_bridge.backends_are_initialized():
            return jax.local_device_count()
    except (ImportError, AttributeError):
        pass  # private API moved: fall back to the flag
    flag = re.search(
        r"xla_force_host_platform_device_count=(\d+)",
        os.environ.get("XLA_FLAGS", ""),
    )
    return int(flag.group(1)) if flag else 1


# deliberate: named check, keeps numpyro_model short
def _warn_zero_size_for_parallel_chains(observations):
    """Warn if parallel chains would crash on zero-size arrays in the data (XLA's
    Shardy pass segfaults compiling a ``jax.pmap`` that captures one).
    """
    if _host_devices() < 2:
        return
    empty = [
        jax.tree_util.keystr(path)
        for path, leaf in jax.tree_util.tree_leaves_with_path(observations)
        if isinstance(leaf, (jax.Array, onp.ndarray)) and leaf.size == 0
    ]
    if empty:
        warnings.warn(
            "The data hold zero-size arrays ("
            + ", ".join(empty[:4])
            + ("" if len(empty) <= 4 else ", ...")
            + "), on which jax.pmap segfaults while compiling (JAX 0.11.2). "
            'Run numpyro chains with chain_method="vectorized", not '
            '"parallel".',
            RuntimeWarning,
            stacklevel=external_stacklevel(),
        )


@renamed()
def numpyro_model(
    model,
    priors,
    data,
    regularizers=(),
    noise=None,
    likelihoods=(),
    *,
    flat_coordinates=False,
    **options,
):
    """Return a numpyro model sampling the parameters in ``priors``.

    Arguments are shared with [`fit`][virgil.fitting.fit]; see
    [common arguments](../common_arguments.md).

    Parameters
    ----------
    model : SourceModel or callable
        A template whose leaves at the paths in ``priors`` are sampled, or a
        function of the sampled values returning a
        [`SourceModel`][virgil.models.SourceModel] or a list of them, one per
        dataset (see [`build_model`][virgil.likelihood.build_model]).
    priors : dict[str, numpyro.distributions.Distribution]
        Parameter path (or function argument name) to prior; each key is also
        the sample-site name. An [`AngleVector`][virgil.angles.AngleVector]
        prior samples a 2-D vector at ``"<path>_vec"`` and records the angle at
        ``"<path>"``.
    data : OIData or sequence of OIData
        Gaussian log likelihood, added with ``numpyro.factor``; may be ``()``
        when ``likelihoods`` holds all the data.
    regularizers : sequence, optional
        Log-prior terms, e.g. [`Centroid`][virgil.imaging.Centroid].
    noise : dict or list of dict, optional
        Priors on error and calibration terms, sites ``"noise.<term>"``
        (``"noise[i].<term>"`` for a list); see
        [common arguments](../common_arguments.md#noise).
    likelihoods : sequence, optional
        Extra data terms, sites ``"likelihood_<i>"``; see
        [common arguments](../common_arguments.md#likelihoods).
    flat_coordinates : bool, optional
        Sample each prior in its flat coordinate (default ``False``; see
        [common arguments](../common_arguments.md#flat_coordinates)). Pass the
        same value to [`gauss_newton_mass`][virgil.fitting.gauss_newton_mass].
    **options
        Fixed error terms and ``reject_unphysical``, passed to
        [`model_loglike`][virgil.likelihood.model_loglike].

    Returns
    -------
    callable
        Zero-argument numpyro model, e.g. for ``numpyro.infer.NUTS``.

    Examples
    --------
    >>> posterior = numpyro_model(model, priors, data)  # doctest: +SKIP
    >>> mcmc = MCMC(NUTS(posterior), num_warmup=500, num_samples=500)  # doctest: +SKIP
    """
    import numpyro

    paths = list(priors)
    for path in paths:
        if is_flux_param(path):
            _check_positive_flux_prior(path, priors[path])
    penalties = [type(r).__name__ for r in regularizers if not r.probabilistic]
    if penalties:
        raise ValueError(
            f"{', '.join(penalties)} are penalties, not log prior densities, "
            "so they cannot be sampled; use fit for regularized MAP images."
        )
    observations = tuple(data) if isinstance(data, (list, tuple)) else (data,)
    _warn_zero_size_for_parallel_chains(observations)

    sites = noise_sites(noise, len(observations))
    likelihoods = tuple(likelihoods)

    def numpyro_fn():
        values = []
        for path in paths:
            prior = priors[path]
            if is_angle_vector(prior):  # the vector at ``<path>_vec``
                vector = numpyro.sample(vector_site(path), prior)
                values.append(
                    numpyro.deterministic(path, vector_angle(vector))
                )
            else:  # with ``flat_coordinates``, in the flat coordinate
                values.append(
                    numpyro.sample(
                        path,
                        flat_sampled(prior) if flat_coordinates else prior,
                    )
                )
        source = build_model(model, paths, values)
        sources = _per_dataset(source, len(observations))
        fitted = dict(zip(paths, values))
        terms = {
            site: (
                numpyro.deterministic(site, spec(fitted))
                if is_tied(spec)
                else numpyro.sample(
                    site, flat_sampled(spec) if flat_coordinates else spec
                )
            )
            for site, (spec, _, _) in sites.items()
        }
        if any(is_tied(spec) for spec, _, _ in sites.values()):
            numpyro.factor("noise_prior", tied_log_prior(sites, fitted))
        if observations:
            numpyro.factor(
                "loglike",
                sum(
                    model_loglike(
                        src, obs, **options, **noise_for(sites, terms, i)
                    )
                    for i, (src, obs) in enumerate(zip(sources, observations))
                ),
            )
        for i, term in enumerate(likelihoods):
            # A term's own normalized ``loglike`` (``PositionData.term``,
            # ``RVData.term``), otherwise ``-0.5 * sum(r**2)`` only.
            numpyro.factor(
                f"likelihood_{i}",
                term.loglike(fitted)
                if hasattr(term, "loglike")
                else -0.5 * np.sum(np.ravel(term(fitted)) ** 2),
            )
        for i, regularizer in enumerate(regularizers):
            numpyro.factor(
                f"regularizer_{i}", -regularizer.value(_reference(source))
            )

    numpyro_fn.flat_coordinates = bool(flat_coordinates)
    return numpyro_fn


def chain_init_params(model, starts, key=None, mass=None):
    """Initial parameters for numpyro's ``MCMC``, one chain per start.

    Converts one dict of values per chain (e.g. the distinct fits of
    [`OrbitStart.chain_values`][virgil.epochs.OrbitStart.chain_values]) into
    numpyro's ``init_params``: unconstrained, with a leading axis over chains.

    Parameters
    ----------
    model : callable
        The numpyro model, e.g. from
        [`numpyro_model`][virgil.likelihood.numpyro_model].
    starts : sequence of dict
        One dict of values per chain, keyed by sample site as ``init_to_value``
        takes them (e.g. ``FitResult.values``). Missing sites start uniformly at
        random.
    key : jax.Array, optional
        Random key for those sites (default ``PRNGKey(0)``).
    mass : dict, optional
        The result of [`gauss_newton_mass`][virgil.fitting.gauss_newton_mass]; a
        ``ValueError`` is raised if it was computed with another
        ``flat_coordinates`` than ``model``.

    Returns
    -------
    dict
        Unconstrained parameters by site, with a leading axis of ``len(starts)``
        (none for a single start), for ``mcmc.run(key, init_params=...)`` with
        ``num_chains=len(starts)``.

    Examples
    --------
    >>> posterior = numpyro_model(model, priors, data)  # doctest: +SKIP
    >>> init = chain_init_params(posterior, start.chain_values(4))  # doctest: +SKIP
    >>> mcmc = MCMC(NUTS(posterior), num_warmup=500, num_samples=500,
    ...             num_chains=4, chain_method="vectorized")  # doctest: +SKIP
    >>> mcmc.run(jax.random.PRNGKey(1), init_params=init)  # doctest: +SKIP
    """
    from numpyro.infer.initialization import init_to_value
    from numpyro.infer.util import initialize_model

    wanted = getattr(model, "flat_coordinates", None)
    given = getattr(mass, "flat_coordinates", None)
    if wanted is not None and given is not None and wanted != given:
        raise ValueError(
            f"The mass matrix was computed with flat_coordinates={given}, "
            f"but the numpyro model samples with flat_coordinates={wanted}: "
            "its matrix is in the wrong coordinates, and with "
            "adapt_mass_matrix=False it is never corrected. Pass the same "
            "flat_coordinates to numpyro_model and gauss_newton_mass."
        )
    starts = list(starts)
    if not starts:
        raise ValueError("starts must hold at least one dict of values.")
    keys = jax.random.split(
        jax.random.PRNGKey(0) if key is None else key, len(starts)
    )
    unconstrained = [
        initialize_model(
            k, model, init_strategy=init_to_value(values=dict(values))
        )[0].z
        for k, values in zip(keys, starts)
    ]
    if len(unconstrained) == 1:
        return unconstrained[0]
    return jax.tree_util.tree_map(lambda *z: np.stack(z), *unconstrained)


@renamed()
def posterior_predictive_summary(samples, model, data, params=None):
    """Mean and spread of the model observables over posterior samples.

    Parameters
    ----------
    samples : dict[str, array-like]
        Posterior samples, as equal-length 1D arrays keyed by parameter
        name or path (e.g. from ``mcmc.get_samples()``).
    model : SourceModel or callable
        Template model or class, as for :func:`loglike`.
    data : OIData
        Data defining the observables.
    params : list[str], optional
        Which keys of ``samples`` to use (default: all of them).

    Returns
    -------
    dict
        ``vis_mean``, ``vis_std``, ``phi_mean`` and ``phi_std``: the mean
        and standard deviation over the samples of each visibility and
        phase observable.
    """
    params = list(samples) if params is None else list(params)
    values = np.stack(
        [np.asarray(samples[param], dtype=float) for param in params], axis=1
    )
    predictions = jax.vmap(
        lambda row: data.model(build_model(model, params, list(row)))
    )(values)
    n_vis = np.asarray(data.vis).size
    n_phase = n_vis + np.asarray(data.phi).size
    vis, phi = predictions[:, :n_vis], predictions[:, n_vis:n_phase]
    return {
        "vis_mean": vis.mean(axis=0),
        "vis_std": vis.std(axis=0),
        "phi_mean": phi.mean(axis=0),
        "phi_std": phi.std(axis=0),
    }


@renamed()
def flux_scale_posterior(model, data):
    """The grey scales of the extra spectra, given a model.

    The scales (and polynomial coefficients) of OI_FLUX spectra and
    correlated fluxes are marginalized in the likelihood (see
    [`FluxSpectrum`][virgil.observables.FluxSpectrum]); this is their
    Gaussian posterior conditional on ``model``, for reporting.

    Parameters
    ----------
    model : SourceModel
        E.g. the best fit.
    data : OIData
        Data with extra spectra (``extras=("flux",)`` and the like).

    Returns
    -------
    dict
        Per kind (``"flux"``, ``"nflux"``, ``"corrflux"``): ``mean``
        ``(n_group, p)`` and ``cov`` ``(n_group, p, p)`` of the weights,
        whose first is the scale k multiplying the model's template
        (normalized to a mean of 1 per group); and ``scale``, the scale of
        ``total_spectrum`` itself (k over the template's normalization),
        for ``"flux"`` and ``"corrflux"``: data ≈ scale × model. ``groups``
        gives each sample's group.
    """
    from .observables import FluxSpectrum

    cvis = data._cvis(model)
    out = {}
    for block in data.extras:
        if not isinstance(block, FluxSpectrum):
            continue
        prediction = block.predict(model, cvis)
        mean, cov = block.posterior(prediction, block.values, block.errors)
        entry = {"mean": mean, "cov": cov, "groups": block.group}
        if block.kind != "nflux":
            base = model.total_spectrum(np.asarray(block.wavel))
            base = np.broadcast_to(base, block.wavel.shape)
            if block.kind == "corrflux":
                base = base * np.abs(cvis)[block.sample]
            n_group = block.members.shape[0]
            norm = jax.ops.segment_sum(base, block.group, n_group)
            norm = norm / np.asarray(block.count, norm.dtype)
            entry["scale"] = mean[:, 0] / norm
        out[block.kind] = entry
    return out
