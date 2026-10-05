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

import jax
import jax.numpy as np
import numpy as onp
from jax.scipy.special import i0e

from ._utils import _per_dataset, _reference, concrete, is_flux_param
from .gains import GAIN_GROUPS
from .models import SourceModel


def _gain_jacobian(data_obj, vis_prediction):
    """dObs/dlog|V| of the model's visibility observables."""
    if data_obj.vis_mode == "v2":
        return 2.0 * vis_prediction
    if data_obj.vis_mode == "amp":
        return vis_prediction
    return np.ones_like(vis_prediction)


def _whiten_vis(data_obj, prediction, resid, errors, gain_terms):
    """Whitened visibility residuals and their effective errors."""
    whitened = resid / errors
    if data_obj.gains is None:
        return whitened, errors
    gains = data_obj.gains
    jacobian = _gain_jacobian(data_obj, prediction) / errors
    whitened, extra = gains.whiten(
        whitened, jacobian, gains.widths_for(gain_terms)
    )
    return whitened, errors * np.exp(extra)


def _whiten(data_obj, prediction, reference, errors, gain_terms=None):
    """Whitened residuals, and the errors that normalise their likelihood.

    Returns ``(whitened, errors_out)``. Residuals are
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
    """
    prediction = np.asarray(prediction)
    resid = prediction - np.asarray(reference)
    errors = np.asarray(errors)
    n_vis = np.asarray(data_obj.vis).size
    vis, vis_errors = _whiten_vis(
        data_obj, prediction[:n_vis], resid[:n_vis], errors[:n_vis], gain_terms
    )
    phase, phase_errors = _whiten_phases(
        data_obj, resid[n_vis:], errors[n_vis:]
    )
    return (
        np.concatenate([vis, phase]),
        np.concatenate([vis_errors, phase_errors]),
    )


def _whiten_phases(data_obj, resid, errors):
    """Whitened phase residuals and their effective errors (see ``_whiten``)."""
    if not data_obj._phases_wrap:
        return resid / errors, errors
    if data_obj.cp_noise is None:
        von_mises = np.sqrt(2.0 * np.pi) * i0e(1.0 / errors**2)
        return 2.0 * np.sin(0.5 * resid) / errors, von_mises
    # Correlated closure phases mix their residuals, so each sign matters,
    # and a chord's sign flips under 2π. Whiten the (smooth, periodic) sines
    # and append the periodic penalty 2 sin²(Δ/2)/σ = (1 - cos Δ)/σ, which
    # removes the false minimum at Δ = π.
    whitened, whitened_errors = data_obj.cp_noise.whiten(np.sin(resid), errors)
    penalty = 2.0 * np.sin(0.5 * resid) ** 2 / errors
    penalty_errors = np.full(penalty.shape, 1.0 / np.sqrt(2.0 * np.pi))
    return (
        np.concatenate([whitened, penalty]),
        np.concatenate([whitened_errors, penalty_errors.astype(errors.dtype)]),
    )


def _gaussian_loglike(whitened, errors):
    """Gaussian log density of whitened residuals with uncertainties ``errors``."""
    return (
        -0.5 * np.sum(whitened**2)
        - np.sum(np.log(errors))
        - 0.5 * whitened.size * np.log(2.0 * np.pi)
    )


# Error terms, as accepted by the likelihoods and the ``noise`` argument of
# ``fit`` and ``numpyro_model``: error inflation (``inflated_errors``), and
# the widths of gains correlated across channels (``OIData.with_gains``).
GAIN_TERMS = tuple(f"vis_gain_{group}" for group in GAIN_GROUPS)
NOISE_TERMS = (
    "vis_scale",
    "phi_scale",
    "vis_error_rel",
    "phi_error",
) + GAIN_TERMS


def inflated_errors(
    data_obj,
    prediction,
    vis_error_rel=None,
    phi_error=None,
    vis_scale=None,
    phi_scale=None,
):
    """The data uncertainties, scaled and with extra terms in quadrature.

    The visibility errors become ``hypot(vis_scale σ, vis_error_rel V)``,
    for the model visibility observable ``V``, and the phase errors
    ``hypot(phi_scale σ, phi_error)``. Terms left as ``None`` are not
    applied.

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

    Returns
    -------
    array-like
        Uncertainties matching [`flatten_data`][virgil.oidata.OIData.flatten_data].
    """
    _, errors = data_obj.flatten_data()
    terms = (vis_error_rel, phi_error, vis_scale, phi_scale)
    if all(term is None for term in terms):
        return errors
    projected = data_obj.vis_mat is not None or data_obj.phi_mat is not None
    if projected and (vis_error_rel is not None or phi_error is not None):
        raise ValueError(
            "Extra error terms are defined for the observed visibilities and "
            "phases, not for projected (vis_mat/phi_mat) observables; scale "
            "their errors with vis_scale/phi_scale instead."
        )
    n_vis = np.asarray(data_obj.vis).size
    d_vis, d_phi = errors[:n_vis], errors[n_vis:]
    if vis_scale is not None:
        d_vis = vis_scale * d_vis
    if phi_scale is not None:
        d_phi = phi_scale * d_phi
    if vis_error_rel is not None:
        d_vis = np.hypot(d_vis, vis_error_rel * np.asarray(prediction)[:n_vis])
    if phi_error is not None:
        d_phi = np.hypot(d_phi, phi_error)
    return np.concatenate([d_vis, d_phi])


def noise_sites(noise, n_datasets):
    """Expand a ``noise`` specification into named sites.

    ``noise`` maps error-inflation terms (``NOISE_TERMS``) to priors and
    applies to every dataset, giving sites ``"noise.<term>"``; a list of such
    dicts, one per dataset, gives sites ``"noise[i].<term>"``.

    Returns
    -------
    dict
        ``{site: (prior, datasets, term)}``, ``datasets`` being the indices
        of the datasets the term applies to.
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
            lower = getattr(prior.support, "lower_bound", None)
            value = None if lower is None else concrete(lower)
            if value is None or onp.any(value < 0.0):
                raise ValueError(
                    f"The prior on noise term {term!r} must have "
                    "non-negative support, e.g. dist.Uniform(0, ...)."
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
    inflation = {k: v for k, v in noise.items() if k not in GAIN_TERMS}
    prediction = data_obj.model(model_object)
    errors = inflated_errors(data_obj, prediction, **inflation)
    data = data_obj.flatten_data()[0]
    return _whiten(data_obj, prediction, data, errors, gain_terms)


def whitened_residuals(model_object, data_obj, **noise):
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
    model_object : SourceModel
        Model to evaluate.
    data_obj : OIData
        Data to compare with.
    **noise
        Error-inflation terms, ``vis_scale``, ``phi_scale``,
        ``vis_error_rel`` and ``phi_error`` (see
        [`inflated_errors`][virgil.likelihood.inflated_errors]), and the
        widths of the data's gains, ``vis_gain_<group>`` (see
        [`OIData.with_gains`][virgil.oidata.OIData.with_gains]).

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
    return _whitened_and_errors(model_object, data_obj, noise)[0]


def model_loglike(model_object, data_obj, *, reject_unphysical=False, **noise):
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
    model_object : SourceModel
        Model to evaluate.
    data_obj : OIData
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
        of the visibility covariance.
    """
    whitened, errors = _whitened_and_errors(model_object, data_obj, noise)
    logl = _gaussian_loglike(whitened, errors)
    if reject_unphysical:
        logl = np.where(model_object.is_physical(), logl, -np.inf)
    return logl


def joint_prediction(params, observations, model_fn):
    """Concatenate predictions for a parameter pytree and multiple observations.

    ``model_fn(params, index)`` defines which parameters are shared and which
    are specific to each observation.
    """
    return np.concatenate(
        [
            observation.model(model_fn(params, index))
            for index, observation in enumerate(observations)
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


def joint_loglike(params, observations, model_fn, **options):
    """Sum independent Gaussian log likelihoods over multiple observations.

    ``options`` (error terms and ``reject_unphysical``) are
    passed to [`model_loglike`][virgil.likelihood.model_loglike].
    """
    return sum(
        model_loglike(model_fn(params, index), observation, **options)
        for index, observation in enumerate(observations)
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


def loglike(values, params, data_obj, model, **options):
    """
    Gaussian log-likelihood of a model with the given parameter values, assuming Gaussian errors.

    Parameters
    ----------
    values : array-like
        Values of the model parameters.
    params : list
        List of parameter names.
    data_obj : OIData
        Object containing the data to be fitted.
    model : SourceModel or callable
        Template model whose parameters at the dot-separated paths ``params``
        are replaced by ``values``, or a class/callable called as
        ``model(**dict(zip(params, values)))`` (see [`build_model`][virgil.likelihood.build_model]).
    **options
        Error terms and ``reject_unphysical``, passed to
        [`model_loglike`][virgil.likelihood.model_loglike].

    Returns
    -------
    float
        Log-likelihood value.
    """

    return model_loglike(
        build_model(model, params, values), data_obj, **options
    )


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
            "be non-negative. Use a prior with non-negative support, e.g. "
            "dist.LogUniform or dist.Uniform(0, ...)."
        )


def numpyro_model(
    model, priors, data_obj, regularisers=(), noise=None, **options
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
        list of models, one per dataset in ``data_obj``, sharing
        parameters: for example binaries at two epochs with one flux
        ratio and a position each. Model ``i`` is compared with dataset
        ``i``, and regularisers act on the first model only.
    priors : dict[str, numpyro.distributions.Distribution]
        Mapping from parameter path (e.g. ``"comp.flux"``) or function
        argument name to prior; each key is also used as the numpyro
        sample-site name. Priors on fluxes (keys named ``flux`` or ending
        in ``.flux``) must have non-negative support.
    data_obj : OIData or sequence of OIData
        Data whose Gaussian log likelihood is added with ``numpyro.factor``.
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
        [`OIData.with_gains`][virgil.oidata.OIData.with_gains]),
        sampled as sites ``"noise.<term>"``. A list gives each dataset its
        own terms, as sites ``"noise[i].<term>"``.
    **options
        Fixed error terms and ``reject_unphysical``, passed to
        [`model_loglike`][virgil.likelihood.model_loglike].

    Returns
    -------
    callable
        Zero-argument numpyro model, e.g. for ``numpyro.infer.NUTS``.
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
    observations = (
        tuple(data_obj) if isinstance(data_obj, (list, tuple)) else (data_obj,)
    )

    sites = noise_sites(noise, len(observations))

    def numpyro_fn():
        values = [numpyro.sample(path, priors[path]) for path in paths]
        source = build_model(model, paths, values)
        sources = _per_dataset(source, len(observations))
        terms = {site: numpyro.sample(site, sites[site][0]) for site in sites}
        numpyro.factor(
            "loglike",
            sum(
                model_loglike(
                    src, obs, **options, **noise_for(sites, terms, i)
                )
                for i, (src, obs) in enumerate(zip(sources, observations))
            ),
        )
        for i, regulariser in enumerate(regularisers):
            numpyro.factor(
                f"regulariser_{i}", -regulariser.value(_reference(source))
            )

    return numpyro_fn


def posterior_predictive_summary(samples, model, data_obj, params=None):
    """Mean and spread of the model observables over posterior samples.

    Parameters
    ----------
    samples : dict[str, array-like]
        Posterior samples, as equal-length 1D arrays keyed by parameter
        name or path (e.g. from ``mcmc.get_samples()``).
    model : SourceModel or callable
        Template model or class, as for :func:`loglike`.
    data_obj : OIData
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
        lambda row: data_obj.model(build_model(model, params, list(row)))
    )(values)
    n_vis = np.asarray(data_obj.vis).size
    vis, phi = predictions[:, :n_vis], predictions[:, n_vis:]
    return {
        "vis_mean": vis.mean(axis=0),
        "vis_std": vis.std(axis=0),
        "phi_mean": phi.mean(axis=0),
        "phi_std": phi.std(axis=0),
    }
