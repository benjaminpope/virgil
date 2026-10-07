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


def _gain_jacobian(data_obj, vis_prediction):
    """dObs/dlog|V| of the model's visibility observables."""
    if data_obj.vis_mode == "v2":
        return 2.0 * vis_prediction
    if data_obj.vis_mode == "amp":
        return vis_prediction
    return np.ones_like(vis_prediction)


def _whiten_vis(data_obj, prediction, resid, errors, gain_terms):
    """Whitened visibility residuals, their effective errors, and the log
    normalisation of the marginalised gains (``Σ extra``, 0 without)."""
    whitened = resid / errors
    if data_obj.gains is None:
        return whitened, errors, np.zeros((), errors.dtype)
    gains = data_obj.gains
    jacobian = _gain_jacobian(data_obj, prediction) / errors
    whitened, extra = gains.whiten(
        whitened, jacobian, gains.widths_for(gain_terms)
    )
    return whitened, errors * np.exp(extra), np.sum(extra)


def _whiten(
    data_obj, prediction, reference, errors, gain_terms=None, offset_terms=None
):
    """Whitened residuals, and the errors that normalise their likelihood.

    Returns ``(whitened, errors_out)``; see ``_whiten_with_log_norm``.
    """
    return _whiten_with_log_norm(
        data_obj, prediction, reference, errors, gain_terms, offset_terms
    )[:2]


def _whiten_with_log_norm(
    data_obj, prediction, reference, errors, gain_terms=None, offset_terms=None
):
    """Whitened residuals, effective errors, and the marginal log-normaliser.

    Returns ``(whitened, errors_out, log_norm)``. ``log_norm`` is the part
    of ``Σ log errors_out`` that comes from marginalised linear nuisances
    and so may depend on the model: ½ log det of the gains' and the
    closure offsets' covariance factors (``Σ extra`` of
    [`GainModes.whiten`][virgil.gains.GainModes.whiten] and
    [`ClosureOffsets.whiten`][virgil.gains.ClosureOffsets.whiten]), and
    for each extra observable block ``Σ log(errors_out / errors)``. It is
    zero for visibilities and phases without gains or offsets. The rest of
    the normaliser (the quoted errors, the von Mises and correlated
    closure-phase terms) depends only on the data and error terms.

    Residuals are
    ``(prediction - reference) / errors``, except:

    - With gains (``OIData.gains``), the visibility residuals are whitened
      by their covariance with the gains marginalised (see
      [`virgil.gains`][virgil.gains]), with widths from ``gain_terms``
      (``vis_gain_<group>``) or the defaults. ``errors_out`` then holds
      effective errors whose log-sum is ½ log of that covariance's
      determinant.

    - An unprojected phase residual Δ becomes 2 sin(Δ/2). Its square,
      2(1 - cos Δ), equals Δ² to fourth order, repeats every 2π and is
      smooth at ±π, so the Gaussian likelihood built on it is a von Mises
      likelihood with concentration κ = 1/σ². Its exact normaliser,
      -log(2π I₀(κ)) + κ = -log 2π - log i0e(κ), is returned as the
      effective error √(2π) i0e(κ), so that ``_gaussian_loglike`` gives a
      density normalised on the circle; for σ ≪ 1 this is σ. Projected
      (kernel or DISCO) phases are linear combinations that are not
      wrapped, and are left as Δ.
    - Closure phases from four or more telescopes are correlated, and only
      some are independent. A chord changes sign under Δ → Δ + 2π, which
      is harmless in a single square but not in correlated cross terms
      (wrapping the residuals first just moves the discontinuity to
      ±π). So the sines s = sin Δ, which are smooth and 2π-periodic, are
      mapped to the independent combinations and whitened with the
      covariance (``OIData.cp_noise``): ``k`` residuals, fewer than the
      closure phases. Then, for each closure phase, a periodic penalty
      q/σ = 2 sin²(Δ/2)/σ = (1 - cos Δ)/σ is appended, uncorrelated, so
      there are ``n_phase`` more residuals (``OIData.n_residuals`` in all).
      Near Δ = 0, sin Δ = Δ - Δ³/6 and q/σ = Δ²/(2σ), so the χ² is the
      correlated Gaussian ΔᵀC⁻¹Δ to O(Δ³), with the penalty adding only
      O(Δ⁴/σ²). At Δ = π the sines vanish, which alone would be a false
      minimum; the penalty there is (2/σ)², which removes it. The
      penalty rows carry the effective error 1/√(2π), so that they add
      nothing to the normalisation (their ``-log σ - ½ log 2π`` is zero);
      ``errors_out`` for the whitened rows holds effective errors whose
      log-sum is ½ log of the covariance's pseudo-determinant, keeping the
      Gaussian normaliser (a correlated von Mises density has no
      closed-form one, and the Gaussian is its limit for σ ≪ 1).
      With closure-phase offsets (``OIData.phase_offsets``), the whitened
      sines are whitened again for the offsets' covariance (see
      [`ClosureOffsets`][virgil.gains.ClosureOffsets]), with widths from
      ``offset_terms`` (``phi_offset_<group>``) or the defaults; the
      penalty rows are unchanged.
    - Extra observables (``OIData.extras``) follow the phases, each block
      whitened by itself (see [`virgil.observables`][virgil.observables]).
    """
    prediction = np.asarray(prediction)
    resid = prediction - np.asarray(reference)
    errors = np.asarray(errors)
    n_vis = np.asarray(data_obj.vis).size
    n_phase = n_vis + np.asarray(data_obj.phi).size
    vis, vis_errors, vis_norm = _whiten_vis(
        data_obj, prediction[:n_vis], resid[:n_vis], errors[:n_vis], gain_terms
    )
    phase, phase_errors, phase_norm = _whiten_phases(
        data_obj, resid[n_vis:n_phase], errors[n_vis:n_phase], offset_terms
    )
    whitened, effective = [vis, phase], [vis_errors, phase_errors]
    log_norm = vis_norm + phase_norm
    reference = np.asarray(reference)
    offset = n_phase
    for block in data_obj.extras:
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
    return np.concatenate(whitened), np.concatenate(effective), log_norm


def _whiten_phases(data_obj, resid, errors, offset_terms=None):
    """Whitened phase residuals, their effective errors and the closure
    offsets' log normalisation (see ``_whiten_with_log_norm``)."""
    zero = np.zeros((), errors.dtype)
    if not data_obj._phases_wrap:
        return resid / errors, errors, zero
    if data_obj.cp_noise is None:
        von_mises = np.sqrt(2.0 * np.pi) * i0e(1.0 / errors**2)
        return 2.0 * np.sin(0.5 * resid) / errors, von_mises, zero
    # Correlated closure phases mix their residuals, so each sign matters,
    # and a chord's sign flips under 2π. Whiten the (smooth, periodic) sines
    # and append the periodic penalty 2 sin²(Δ/2)/σ = (1 - cos Δ)/σ, which
    # removes the false minimum at Δ = π.
    whitened, whitened_errors = data_obj.cp_noise.whiten(np.sin(resid), errors)
    offsets = data_obj.phase_offsets
    log_norm = zero
    if offsets is not None:
        whitened, extra = offsets.whiten(
            data_obj.cp_noise,
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


def inflated_errors(
    data_obj,
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
    data_obj : OIData
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
    data, errors = data_obj.flatten_data()
    terms = (vis_error_rel, phi_error, vis_scale, phi_scale, vis_error)
    if all(term is None for term in terms):
        return errors
    projected = data_obj.vis_mat is not None or data_obj.phi_mat is not None
    added = (vis_error_rel, phi_error, vis_error)
    if projected and any(term is not None for term in added):
        raise ValueError(
            "Extra error terms are defined for the observed visibilities and "
            "phases, not for projected (vis_mat/phi_mat) observables; scale "
            "their errors with vis_scale/phi_scale instead."
        )
    if where not in ("model", "data"):
        raise ValueError(f"where must be 'model' or 'data', not {where!r}.")
    reference = data if where == "data" else np.asarray(prediction)
    n_vis = np.asarray(data_obj.vis).size
    n_phase = n_vis + np.asarray(data_obj.phi).size
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


def _whitened_and_errors(model_object, data_obj, noise):
    """Whitened residuals and effective errors (see ``_whiten``)."""
    return _whitened_errors_and_log_norm(model_object, data_obj, noise)[:2]


def _whitened_and_log_norm(model_object, data_obj, noise=None):
    """Whitened residuals and the marginal nuisances' log-normaliser.

    ``model_loglike`` is ``-½ Σ r² - log_norm`` plus a term that depends
    only on the data and the error terms (``-Σ log σ - ½ n log 2π`` and
    its von Mises and correlated-closure-phase forms), where ``log_norm``
    is ½ log det of the covariance factors of marginalised gains and
    closure offsets, which may depend on the model (see
    ``_whiten_with_log_norm``). Without them it is zero.
    """
    whitened, _, log_norm = _whitened_errors_and_log_norm(
        model_object, data_obj, {} if noise is None else noise
    )
    return whitened, log_norm


def _whitened_errors_and_log_norm(model_object, data_obj, noise):
    unknown = set(noise) - set(NOISE_TERMS)
    if unknown:
        raise TypeError(
            f"Unknown error terms {sorted(unknown)}; use {NOISE_TERMS}."
        )
    gain_terms = {k: v for k, v in noise.items() if k in GAIN_TERMS}
    if gain_terms and data_obj.gains is None:
        raise ValueError(
            f"Error terms {sorted(gain_terms)} need gain modes: add them with "
            "OIData.with_gains."
        )
    offset_terms = {k: v for k, v in noise.items() if k in OFFSET_TERMS}
    if offset_terms and data_obj.phase_offsets is None:
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
    observed = data_obj
    if wavel_terms:
        data_obj = data_obj.with_wavelength_scale(**wavel_terms)
    if "north_angle" in noise:
        data_obj = data_obj.with_north_angle(noise["north_angle"])
    prediction = data_obj.model(model_object)
    data_obj = observed
    errors = inflated_errors(data_obj, prediction, **inflation)
    data = data_obj.flatten_data()[0]
    return _whiten_with_log_norm(
        data_obj, prediction, data, errors, gain_terms, offset_terms
    )


@renamed()
def whitened_residuals(model, data, **noise):
    """Residuals of a model divided by the data uncertainties.

    This is the one residual vector behind every likelihood in virgil:
    ``model_loglike`` is ``-0.5 * sum(whitened_residuals**2)`` plus the
    Gaussian normalisation, and least-squares fits minimise its sum of
    squares.

    Visibility and projected-phase (kernel or DISCO) residuals are
    ``(model - data) / σ``. Unprojected phase residuals Δ are
    ``2 sin(Δ/2) / σ``: equal to Δ/σ for small Δ, but smooth where Δ wraps
    at ±π, so that a χ² surface has no kinks there. The resulting
    likelihood is a von Mises distribution with concentration 1/σ²,
    normalised exactly on the circle.

    Closure phases from four or more telescopes are the exception: they are
    correlated, so the sines sin Δ of the residuals are whitened together
    and replaced by their independent combinations, and a periodic penalty
    2 sin²(Δ/2)/σ per closure phase is appended (chords would flip sign at
    Δ → Δ + 2π, making correlated combinations discontinuous). The χ² is
    continuous and smooth everywhere, equals the correlated Gaussian
    ΔᵀC⁻¹Δ to O(Δ³) for small residuals, and has no false minimum at
    Δ = π. There are then ``n_independent`` + (number of closure phases)
    residuals, namely ``OIData.n_residuals``.

    Parameters
    ----------
    model : SourceModel
        Model to evaluate.
    data : OIData
        Data to compare with.
    **noise
        Error-inflation terms, ``vis_scale``, ``phi_scale``,
        ``vis_error_rel`` and ``phi_error`` (see
        [`inflated_errors`][virgil.likelihood.inflated_errors]), and the
        widths of the data's gains, ``vis_gain_<group>`` (see
        [`OIData.with_gains`][virgil.oidata.OIData.with_gains]),
        closure-phase offsets, ``phi_offset_<group>`` (see
        [`OIData.with_closure_offsets`][virgil.oidata.OIData.with_closure_offsets]), the
        wavelength scale, ``wavel_scale`` and ``wavel_offset`` (see
        [`OIData.with_wavelength_scale`][virgil.oidata.OIData.with_wavelength_scale]),
        and the North angle, ``north_angle`` (degrees; see
        [`OIData.with_north_angle`][virgil.oidata.OIData.with_north_angle]).

    Returns
    -------
    array-like
        One dimensionless residual per independent observable
        ([`n_independent`][virgil.oidata.OIData.n_independent] of
        them), in the order of
        [`flatten_data`][virgil.oidata.OIData.flatten_data]; correlated
        closure phases are replaced by their whitened independent
        combinations, followed by one periodic penalty residual per
        closure phase, so the length is
        [`n_residuals`][virgil.oidata.OIData.n_residuals].
    """
    return _whitened_and_errors(model, data, noise)[0]


@renamed()
def model_loglike(model, data, *, reject_unphysical=False, **noise):
    """Evaluate the log likelihood for an instantiated model object.

    This is ``-0.5 * sum(r**2)`` for the residuals ``r`` of
    [`whitened_residuals`][virgil.likelihood.whitened_residuals], plus
    each density's normalisation: Gaussian in visibilities and projected
    phases, ``-log σ - ½ log 2π``; von Mises with concentration
    κ = 1/σ² in uncorrelated unprojected phases, ``-log 2π - log i0e(κ)``,
    which is the Gaussian one for σ ≪ 1 but stays a normalised density
    on the circle when σ is large (e.g. a fitted ``phi_error``).
    Correlated closure phases (``OIData.cp_noise``) keep the Gaussian
    normalisation, their small-σ limit, since a correlated circular
    density has no closed-form normaliser; their periodic penalty
    residuals add nothing to it.

    Parameters
    ----------
    model : SourceModel
        Model to evaluate.
    data : OIData
        Data to compare with.
    reject_unphysical : bool, optional
        If True, return ``-inf`` when
        [`is_physical`][virgil.models.SourceModel.is_physical] is false,
        e.g. for a negative flux or a rim whose brightness goes negative.
        This works inside ``jax.jit``, so samplers can use it as a hard prior
        boundary.
    **noise
        Error-inflation terms, e.g. fitted as nuisance parameters:
        ``vis_scale`` and ``phi_scale`` multiply the uncertainties, and
        ``vis_error_rel`` (relative to the model visibility) and
        ``phi_error`` (radians) are added in quadrature (see
        [`inflated_errors`][virgil.likelihood.inflated_errors]). The
        Gaussian normalization uses the inflated errors. For data with
        gains ([`OIData.with_gains`][virgil.oidata.OIData.with_gains]),
        ``vis_gain_<group>`` sets the width of a group of gains, which are
        marginalised: the normalisation then includes the log-determinant
        of the visibility covariance. ``phi_offset_<group>`` does the same
        for closure-phase offsets ([`OIData.with_closure_offsets`][virgil.oidata.OIData.with_closure_offsets]).
        ``wavel_scale`` and ``wavel_offset`` evaluate the model at corrected
        wavelengths (see [`OIData.with_wavelength_scale`][virgil.oidata.OIData.with_wavelength_scale]),
        and ``north_angle`` on a rotated sky (see
        [`OIData.with_north_angle`][virgil.oidata.OIData.with_north_angle]).
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


def joint_data(observations):
    """Concatenate observed vectors in the same order as ``joint_prediction``."""
    return np.concatenate(
        [observation.flatten_data()[0] for observation in observations]
    )


def joint_errors(observations):
    """Concatenate uncertainty vectors in the same order as ``joint_prediction``."""
    return np.concatenate(
        [observation.flatten_data()[1] for observation in observations]
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


def _term_loglike(term, values):
    """Log density of one ``likelihoods=`` term at the fitted ``values``.

    A term built by ``PositionData.term`` or ``RVData.term`` wraps its data,
    so its full normalised ``loglike`` is used, matching the OIData terms.
    A plain callable returning whitened residuals has no known
    normalisation, so it contributes ``-0.5 * sum(r**2)`` only.
    """
    if hasattr(term, "loglike"):
        return term.loglike(values)
    return -0.5 * np.sum(np.ravel(term(values)) ** 2)


def _sample(numpyro, path, prior, flat=True):
    """Sample the parameter at ``path``: an angle vector at ``<path>_vec``,
    with the angle (degrees) recorded as the deterministic site ``path``.

    With ``flat``, a prior with a flat coordinate (LogUniform, isotropic
    angles) is sampled in it (``_flat.flat_sampled``): the site, its value
    and its density are unchanged, only NUTS's unconstrained coordinate.
    """
    if is_angle_vector(prior):
        vector = numpyro.sample(vector_site(path), prior)
        return numpyro.deterministic(path, vector_angle(vector))
    return numpyro.sample(path, flat_sampled(prior) if flat else prior)


def _host_devices():
    """The number of local devices, without initialising the backend.

    Once JAX's backend starts, ``XLA_FLAGS`` can no longer set the number
    of host devices, so before that read the flag instead. Before the
    backend starts, several GPUs are not counted: the crash is only
    confirmed on CPU.
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


def _warn_zero_size_for_parallel_chains(observations):
    """Warn if parallel chains would crash on zero-size arrays in the data.

    XLA's Shardy pass segfaults compiling a ``jax.pmap`` (numpyro's
    ``chain_method="parallel"``) that captures a zero-size array (JAX
    0.11.2, CPU). Data without closure phases, for example, carry some.
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
            stacklevel=4,
        )


@renamed()
def numpyro_model(
    model,
    priors,
    data,
    regularisers=(),
    noise=None,
    likelihoods=(),
    *,
    flat_coordinates=True,
    **options,
):
    """Return a numpyro model sampling the parameters in ``priors``.

    Parameters
    ----------
    model : SourceModel or callable
        Either a template model whose leaves at the paths in ``priors`` are
        sampled, or a function called with the sampled values as keyword
        arguments that returns a [`SourceModel`][virgil.models.SourceModel]. A function lets you
        sample parameters that are not leaves of the model, such as a
        separation and position angle, or one inclination shared by two
        components (see [`build_model`][virgil.likelihood.build_model]).
        As for [`fit`][virgil.fitting.fit], the function may return a
        list of models, one per dataset in ``data``, sharing
        parameters: for example binaries at two epochs with one flux
        ratio and a position each. Model ``i`` is compared with dataset
        ``i``, and regularisers act on the first model only.
    priors : dict[str, numpyro.distributions.Distribution]
        Mapping from parameter path (e.g. ``"comp.flux"``) or function
        argument name to prior; each key is also used as the numpyro
        sample-site name. Priors on fluxes (keys named ``flux`` or ending
        in ``.flux``) must have non-negative support. An angle (degrees)
        with an [`AngleVector`][virgil.angles.AngleVector] prior is
        sampled as a 2-D vector at the site ``"<path>_vec"``, with the
        angle recorded as the deterministic site ``"<path>"``: there is no
        wrap boundary at 0°/360°.
    data : OIData or sequence of OIData
        Data whose Gaussian log likelihood is added with ``numpyro.factor``.
        May be ``()`` when ``likelihoods`` holds all the data.
    regularisers : sequence, optional
        Log-prior terms on the model, e.g. a
        [`Centroid`][virgil.imaging.Centroid] prior, added with
        ``numpyro.factor``. Only genuine prior densities
        (``probabilistic``) are allowed: penalties such as maximum entropy
        are for [`fit`][virgil.fitting.fit].
    noise : dict or list of dict, optional
        Priors on error-inflation terms (``vis_scale``, ``phi_scale``,
        ``vis_error_rel``, ``phi_error``; see
        [`inflated_errors`][virgil.likelihood.inflated_errors]) and on
        gain widths (``vis_gain_<group>``; see
        [`OIData.with_gains`][virgil.oidata.OIData.with_gains]), on
        closure-offset widths (``phi_offset_<group>``; see
        [`OIData.with_closure_offsets`][virgil.oidata.OIData.with_closure_offsets]), on the
        wavelength scale (``wavel_scale``, ``wavel_offset``; see
        [`OIData.with_wavelength_scale`][virgil.oidata.OIData.with_wavelength_scale])
        and on the North angle (``north_angle``, degrees; see
        [`OIData.with_north_angle`][virgil.oidata.OIData.with_north_angle]),
        sampled as sites ``"noise.<term>"``. A list gives each dataset its
        own terms, as sites ``"noise[i].<term>"``. An entry may instead be
        a function of the dict of sampled values (keyed like ``priors``),
        recorded as a deterministic site: a term *tied* to parameters in
        ``priors``, as in a hierarchical model where each epoch's error
        scale is drawn from a population with fitted hyperparameters (see
        [`hierarchical_scales`][virgil.priors.hierarchical_scales]). A
        tied term's ``log_prior(values)``, if it has one, is added once
        (site ``"noise_prior"``).
        **Priors.** These terms are scale parameters, so their default
        (Jeffreys) prior is log-uniform on stated bounds; a
        ``Uniform(0, ...)`` favours large values. The bounds must contain
        the plausible values: for the factors ``vis_scale`` and
        ``phi_scale``, whose neutral value is 1, e.g.
        ``dist.LogUniform(0.1, 10.0)``; for the added errors and widths
        (``vis_error_rel``, ``phi_error``, ``vis_gain_<group>``,
        ``phi_offset_<group>``), e.g. ``dist.LogUniform(1e-4, 0.3)``.
        ``wavel_scale`` is a scale too: log-uniform about 1 unless a
        calibration gives a Gaussian (``Normal(1, 2e-4)`` for GRAVITY is
        such information).
    likelihoods : sequence, optional
        Extra data terms, as for [`fit`][virgil.fitting.fit]: callables of
        the sampled values (a dict keyed like ``priors``) returning whitened
        residuals, such as ``PositionData.term(orbit_fn)`` or
        ``RVData.term(params_fn)``. Term ``i`` is added with
        ``numpyro.factor`` as site ``"likelihood_<i>"``. Those two built-in
        terms add their full normalised Gaussian log density, like the OIData
        terms; a plain callable adds ``-0.5 * sum(r**2)`` only.
    flat_coordinates : bool, optional
        Sample each prior that is uniform in some coordinate of its
        parameter in that coordinate, as [`fit`][virgil.fitting.fit]
        optimises it (default ``True``): a ``LogUniform(a, b)`` in the
        logit of log x on [log a, log b], an
        [`IsotropicInclination`][virgil.priors.IsotropicInclination] in
        cos i, an [`IsotropicLatitude`][virgil.priors.IsotropicLatitude]
        in sin(lat), for parameters and error terms alike. There the
        prior's potential is exactly logistic in NUTS's unconstrained
        coordinate, and the coordinates match those of
        [`fit`][virgil.fitting.fit] and
        [`gauss_newton_mass`][virgil.fitting.gauss_newton_mass]. This
        does not cure hierarchical funnels, which come from the
        dependence between a group scale and its members: use the
        non-centred form or the mass matrix for those. Only the
        coordinate NUTS moves in changes: the sites keep
        their names, their values are the parameters, and the posterior
        is the same. ``False`` restores numpyro's bijection of each
        prior's support (virgil 0.3), e.g. to reuse unconstrained
        ``init_params`` or a mass matrix computed that way.
    **options
        Fixed error terms and ``reject_unphysical``, passed to
        [`model_loglike`][virgil.likelihood.model_loglike].

    Returns
    -------
    callable
        Zero-argument numpyro model, e.g. for ``numpyro.infer.NUTS``.

    Examples
    --------
    Sample an orbit from measured positions alone, with no OIData:

    >>> import numpy as np
    >>> import numpyro.distributions as dist
    >>> from virgil.likelihood import numpyro_model
    >>> from virgil.orbits import KeplerOrbit, PositionData
    >>> mjd = 60500.0 + np.array([0.0, 100.0, 200.0, 300.0])
    >>> truth = KeplerOrbit(400.0, 30.0, 0.4, 60.0, 40.0, 110.0, 20.0, t_ref=60500.0)
    >>> dra, ddec, _ = (np.asarray(x) for x in truth.relative(mjd))
    >>> cov = np.broadcast_to(0.05**2 * np.eye(2), (4, 2, 2))
    >>> positions = PositionData(mjd, dra, ddec, cov)
    >>> priors = {"a_mas": dist.LogUniform(5.0, 50.0), "ecc": dist.Uniform(0.0, 0.9)}
    >>> def orbit_fn(v):
    ...     return KeplerOrbit(
    ...         400.0, 30.0, v["ecc"], 60.0, 40.0, 110.0, v["a_mas"], t_ref=60500.0
    ...     )
    >>> model = numpyro_model(
    ...     lambda **kw: None, priors, (), likelihoods=[positions.term(orbit_fn)]
    ... )
    >>> from numpyro.infer.util import log_density
    >>> values = {"a_mas": 20.0, "ecc": 0.4}
    >>> bool(np.isfinite(float(log_density(model, (), {}, values)[0])))
    True
    """
    import numpyro

    paths = list(priors)
    for path in paths:
        if is_flux_param(path):
            _check_positive_flux_prior(path, priors[path])
    penalties = [type(r).__name__ for r in regularisers if not r.probabilistic]
    if penalties:
        raise ValueError(
            f"{', '.join(penalties)} are penalties, not log prior densities, "
            "so they cannot be sampled; use fit for regularised MAP images."
        )
    observations = tuple(data) if isinstance(data, (list, tuple)) else (data,)
    _warn_zero_size_for_parallel_chains(observations)

    sites = noise_sites(noise, len(observations))
    likelihoods = tuple(likelihoods)

    def numpyro_fn():
        values = [
            _sample(numpyro, path, priors[path], flat_coordinates)
            for path in paths
        ]
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
            numpyro.factor(f"likelihood_{i}", _term_loglike(term, fitted))
        for i, regulariser in enumerate(regularisers):
            numpyro.factor(
                f"regulariser_{i}", -regulariser.value(_reference(source))
            )

    return numpyro_fn


def chain_init_params(model, starts, key=None):
    """Initial parameters for numpyro's ``MCMC``, one chain per start.

    ``init_to_value`` starts every chain at one point. To start each
    chain in its own mode (e.g. the distinct fits of
    [`OrbitStart.chain_values`][virgil.epochs.OrbitStart.chain_values]),
    numpyro instead takes ``init_params``, in its unconstrained
    coordinates and with a leading axis over chains; this converts one
    dict of values per chain into them.

    Parameters
    ----------
    model : callable
        The numpyro model, e.g. from
        [`numpyro_model`][virgil.likelihood.numpyro_model].
    starts : sequence of dict
        One dict of values per chain, keyed by sample site as
        ``init_to_value`` takes them (e.g. ``FitResult.values``, which
        holds the ``"<path>_vec"`` sites of angle vectors). Sites missing
        from a dict start uniformly at random, as ``init_to_value`` does.
    key : jax.Array, optional
        Random key for those missing sites (default ``PRNGKey(0)``).

    Returns
    -------
    dict
        Unconstrained parameters by site, with a leading axis of
        ``len(starts)`` (none for a single start). Pass them as
        ``mcmc.run(key, init_params=...)`` with
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
    correlated fluxes are marginalised in the likelihood (see
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
        (normalised to a mean of 1 per group); and ``scale``, the scale of
        ``total_spectrum`` itself (k over the template's normalisation),
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
