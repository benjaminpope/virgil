"""Maximum a posteriori fits of models, including images.

[`fit`][virgil.fitting.fit] takes the same arguments as
[`numpyro_model`][virgil.likelihood.numpyro_model]: a model (a template,
or a function of the parameters), a dict of numpyro priors whose keys are the
free parameters, and the data, plus optional regularisers (see
[`virgil.imaging`][virgil.imaging]). It finds the maximum a
posteriori parameters with Levenberg–Marquardt, L-BFGS or Adam, optimising
each parameter in unconstrained coordinates through the bijection to its
prior's support, in float64 by default. A parameter whose prior is uniform in
some coordinate (a log-uniform scale, an isotropic inclination) is fitted in
that flat coordinate, where its prior adds nothing to the loss, so that
Levenberg–Marquardt works with the Jeffreys priors. To sample the same
posterior, pass the same arguments to ``numpyro_model``.
"""

import dataclasses
import math
import warnings

import equinox as eqx
import jax
import jax.numpy as np
import lineax as lx
import numpy as onp
import optax
import optimistix as optx

from ._precision import cast_tree, run_in
from ._utils import _per_dataset, _reference, is_flux_param
from .fields import GaussianField
from .likelihood import (
    _check_positive_flux_prior,
    _whitened_and_errors,
    build_model,
    noise_for,
    noise_sites,
)
from .models import SourceModel


def _bijection(distribution):
    from numpyro.distributions.transforms import biject_to

    return biject_to(distribution.support)


def _base(distribution):
    """``distribution`` without ``.expand(...)`` and ``.to_event(...)``.

    They change the shape, not the density's form (a flat prior stays flat,
    a Normal stays Normal).
    """
    import numpyro.distributions as dist

    while isinstance(
        distribution, (dist.Independent, dist.ExpandedDistribution)
    ):
        distribution = distribution.base_dist
    return distribution


def _flat_coordinate(distribution):
    """The coordinate in which a prior is uniform, or None.

    Returns ``(to_flat, from_flat, low, high)``: a monotonic map ``u =
    to_flat(x)`` of the parameter, its inverse, and the interval ``[low,
    high]`` of ``u`` on which the prior's density is constant. Such a prior
    is an invariant measure written in a coordinate where it is not flat:

    * ``LogUniform(a, b)``: ``u = log x`` on ``[log a, log b]``;
    * any prior with a ``flat_coordinate()`` method returning that tuple,
      such as an isotropic inclination (``u = cos i``) or latitude (``u =
      sin(lat)``).

    ``Uniform`` is its own flat coordinate (the identity) and keeps
    numpyro's bijection; Normal and other priors have none and are
    evaluated in the model's own parameters.
    """
    import numpyro.distributions as dist

    base = _base(distribution)
    if isinstance(base, dist.LogUniform):
        return np.log, np.exp, np.log(base.low), np.log(base.high)
    declared = getattr(base, "flat_coordinate", None)
    if callable(declared):
        return declared()
    return None


class _FlatBijection:
    """Unconstrained z to a parameter uniform in ``u = to_flat(x)``.

    ``x = from_flat(lo + (hi - lo) sigmoid(z))``: numpyro's bijection of the
    flat coordinate's interval, followed by the map back to the parameter.
    """

    def __init__(self, to_flat, from_flat, low, high):
        from numpyro.distributions import constraints
        from numpyro.distributions.transforms import biject_to

        self.to_flat, self.from_flat = to_flat, from_flat
        self.interval = biject_to(constraints.interval(low, high))

    def __call__(self, z):
        return self.from_flat(self.interval(z))

    def inv(self, x):
        return self.interval.inv(self.to_flat(x))


def _prior_residuals(path, distribution, value):
    """Residuals ``r`` with ``-log p(value) = 0.5 sum r² + const``, or None.

    ``None`` means the prior adds nothing to a least-squares objective:
    it is flat on its support (Uniform, ImproperUniform), or flat in the
    coordinate ``fit`` optimises it in (LogUniform and the other priors with
    a flat coordinate; see ``_flat_coordinate``).
    """
    import numpyro.distributions as dist

    if _flat_coordinate(distribution) is not None:
        return None
    distribution = _base(distribution)
    if isinstance(distribution, (dist.Uniform, dist.ImproperUniform)):
        return None
    if isinstance(distribution, dist.Normal):
        return np.ravel((value - distribution.loc) / distribution.scale)
    raise TypeError(
        f"The {type(distribution).__name__} prior on {path!r} has no "
        "least-squares form; use Normal, Uniform, ImproperUniform or "
        "LogUniform priors (or one with a flat coordinate, such as an "
        "isotropic inclination), or fit with method='lbfgs' or 'adam'."
    )


def _traced(prior):
    """``prior`` with its Python-number parameters as JAX arrays.

    The jitted solvers take the problem as an argument, and equinox treats
    Python numbers in it as static, so that every new prior bound (say
    ``Uniform(lo, 50.0)`` for a new ``lo``) would recompile the fit. Arrays
    are traced instead.
    """

    def to_array(leaf):
        if isinstance(leaf, (int, float)) and not isinstance(leaf, bool):
            return np.asarray(float(leaf))
        return leaf

    return jax.tree.map(to_array, prior)


def _warn_if_field_hyperparameter(model, path):
    """Warn when a GaussianField's sigma or length is fitted by MAP."""
    parent, _, name = path.rpartition(".")
    if name in ("sigma", "length_mas") and parent:
        if isinstance(model.get(parent), GaussianField):
            warnings.warn(
                f"Fitting {path!r} by MAP: the hyperparameters of a Gaussian "
                "field are biased at the MAP (towards a flat field). Fix "
                "them, choose them from the evidence, or sample them.",
                UserWarning,
                stacklevel=4,
            )


class _Objective(eqx.Module):
    """The negative log posterior of a fit, as a loss and as residuals.

    ``residuals(z)`` is a vector whose half sum of squares is ``loss(z)``
    (up to a constant), for least-squares solvers; it raises ``TypeError``
    if a regulariser or prior has no least-squares form, or if error terms
    are fitted. ``z`` are the unconstrained coordinates of the parameters
    and of any error terms (keyed by their ``noise`` sites).

    With ``flat=True`` (for ``fit``), a parameter whose prior has a flat
    coordinate (``_flat_coordinate``: LogUniform, isotropic angles) is
    unconstrained through that coordinate, where its prior is constant and
    adds nothing to the loss. With ``flat=False`` (for
    ``gauss_newton_mass``), every parameter uses numpyro's bijection of its
    prior's support, the coordinates NUTS samples.
    """

    model: object
    data: tuple
    priors: dict
    regularisers: tuple
    noise: dict
    likelihoods: tuple
    flat: bool = eqx.field(static=True)

    def __init__(
        self,
        model,
        priors,
        data,
        regularisers=(),
        noise=None,
        likelihoods=(),
        flat=True,
    ):
        self.model = model
        self.flat = flat
        self.data = tuple(data) if isinstance(data, (list, tuple)) else (data,)
        self.likelihoods = tuple(likelihoods)
        self.priors = {path: _traced(p) for path, p in priors.items()}
        self.regularisers = tuple(regularisers)
        self.noise = {
            site: (_traced(prior), datasets, term)
            for site, (prior, datasets, term) in noise_sites(
                noise, len(self.data)
            ).items()
        }
        if not self.priors:
            raise ValueError("priors must name at least one free parameter.")
        for path, prior in self.priors.items():
            if isinstance(model, SourceModel):
                model.get(path)  # raises for an unknown path
                _warn_if_field_hyperparameter(model, path)
            if is_flux_param(path):
                _check_positive_flux_prior(path, prior)

    @property
    def _has_gains(self):
        """Whether any dataset's covariance depends on the model: gains, or
        marginalised flux scales (``OIData.has_model_covariance``)."""
        return any(
            getattr(d, "has_model_covariance", False) for d in self.data
        )

    @property
    def paths(self):
        """The free parameters' paths, in the order of ``priors``."""
        return tuple(self.priors)

    def _in_flat_coordinate(self, prior):
        """Whether ``prior``'s parameter is fitted in its flat coordinate."""
        return self.flat and _flat_coordinate(prior) is not None

    def bijection(self, prior):
        """The map from unconstrained coordinates to ``prior``'s support."""
        if self._in_flat_coordinate(prior):
            return _FlatBijection(*_flat_coordinate(prior))
        return _bijection(prior)

    def init(self, values=None):
        """Unconstrained coordinates of ``values`` (default: the template's).

        Error terms start at 1 (scales, including ``wavel_scale``, and the
        width of supplied gain modes), 0 (``wavel_offset``, ``north_angle``)
        or 0.01 (added
        errors and other gain widths), or at their prior's mean if that is
        outside the prior's support or on its boundary.
        """
        values = {} if values is None else dict(values)
        z = {}
        for site, (prior, _, term) in self.noise.items():
            unit = term.endswith("scale") or term.endswith("_modes")
            zero = term in ("wavel_offset", "north_angle")
            default = 0.0 if zero else 1.0 if unit else 0.01
            start = values.get(site, default)
            if not bool(prior.support(np.asarray(start, float))):
                start = prior.mean
            z[site] = self.bijection(prior).inv(np.asarray(start, float))
            if not bool(np.all(np.isfinite(z[site]))):
                # On the boundary of a bounded prior (e.g. 0 for
                # Uniform(0, ...)), the unconstrained coordinate is
                # infinite: start inside it, at the prior's mean.
                mean = np.asarray(prior.mean, float)
                z[site] = self.bijection(prior).inv(mean)
        for path, prior in self.priors.items():
            if path not in values:
                if not isinstance(self.model, SourceModel):
                    raise ValueError(
                        f"No starting value for {path!r}: pass init= when "
                        "the model is a function."
                    )
                values[path] = self.model.get(path)  # raises if unknown
            z[path] = self.bijection(prior).inv(
                np.asarray(values[path], float)
            )
        return z

    def constrain(self, z):
        """Map unconstrained coordinates to parameter and error-term values."""
        values = {
            path: self.bijection(prior)(z[path])
            for path, prior in self.priors.items()
        }
        for site, (prior, _, _) in self.noise.items():
            values[site] = self.bijection(prior)(z[site])
        return values

    def build(self, z):
        """The model with the parameters at unconstrained coordinates ``z``."""
        values = self.constrain(z)
        return build_model(
            self.model, self.paths, [values[p] for p in self.paths]
        )

    @property
    def _has_term_norms(self):
        return any(getattr(t, "has_log_norm", False) for t in self.likelihoods)

    def data_residuals(self, model, values=None):
        """Whitened residuals of ``model`` for each dataset, as a list.

        ``model`` is one model for all the datasets, or a list of them, one
        per dataset; ``values`` holds the error terms, if any are fitted.
        """
        return [w for w, _ in self._whitened(model, values)]

    def _whitened(self, model, values=None):
        """(whitened residuals, inflated errors) for each dataset."""
        models = _per_dataset(model, len(self.data))
        values = {} if values is None else values
        return [
            _whitened_and_errors(m, d, noise_for(self.noise, values, i))
            for i, (m, d) in enumerate(zip(models, self.data))
        ]

    def residuals(self, z):
        """Residual vector whose half sum of squares is ``loss(z)`` + const.

        Raises ``TypeError`` if a regulariser or prior has no least-squares
        form (then use L-BFGS or Adam).
        """
        if self.noise:
            raise TypeError(
                "Fitted error terms have no least-squares form (the "
                "likelihood's normalisation depends on them); fit with "
                "method='lbfgs' or 'adam'."
            )
        if self._has_gains:
            raise TypeError(
                "Data with gains or marginalised flux scales have no "
                "least-squares form (their covariance, and its determinant, "
                "depend on the model); fit "
                "with method='lbfgs' or 'adam'."
            )
        model = self.build(z)
        values = self.constrain(z)
        if self._has_term_norms:
            raise TypeError(
                "Likelihood terms with fitted error terms (an RV jitter) "
                "have no least-squares form (their normalisation depends on "
                "them); fit with method='lbfgs' or 'adam'."
            )
        parts = self.data_residuals(model)
        parts += [term(values) for term in self.likelihoods]
        for regulariser in self.regularisers:
            if not hasattr(regulariser, "residuals"):
                raise TypeError(
                    f"{type(regulariser).__name__} has no least-squares "
                    "form; fit with method='lbfgs' or 'adam'."
                )
            parts.append(np.ravel(regulariser.residuals(_reference(model))))
        for path, prior in self.priors.items():
            r = _prior_residuals(path, prior, values[path])
            if r is not None:
                parts.append(r)
        return np.concatenate(parts)

    def loss(self, z):
        """Negative log posterior (up to a constant) at ``z``.

        A prior with a flat coordinate (LogUniform, isotropic angles) is
        constant in it, and adds nothing: the minimum is the maximum a
        posteriori in that coordinate. Other priors are evaluated at the
        constrained values, without the Jacobian of the bijection, so for
        them the minimum is the maximum a posteriori in the model's own
        parameters.
        """
        model = self.build(z)
        values = self.constrain(z)
        whitened = self._whitened(model, values)
        chi2 = sum(np.sum(w**2) for w, _ in whitened)
        chi2 = chi2 + sum(np.sum(t(values) ** 2) for t in self.likelihoods)
        # With fitted error terms, the normalisation of the Gaussian
        # likelihood, sum(log σ), is no longer a constant.
        log_norm = sum(np.sum(np.log(e)) for _, e in whitened)
        log_norm = log_norm if self.noise or self._has_gains else 0.0
        # A likelihood term with a fitted jitter reports its own Σ log σ.
        log_norm = log_norm + sum(
            t.log_norm(values)
            for t in self.likelihoods
            if getattr(t, "has_log_norm", False)
        )
        penalty = sum(r.value(_reference(model)) for r in self.regularisers)
        log_prior = sum(
            np.sum(prior.log_prob(values[path]))
            for path, prior in self.priors.items()
            if not self._in_flat_coordinate(prior)
        )
        for site, (prior, _, _) in self.noise.items():
            if not self._in_flat_coordinate(prior):
                log_prior = log_prior + np.sum(prior.log_prob(values[site]))
        return 0.5 * chi2 + log_norm + penalty - log_prior


@dataclasses.dataclass(frozen=True)
class FitResult:
    """The result of [`fit`][virgil.fitting.fit].

    Attributes
    ----------
    model : SourceModel or list
        The fitted model, or models (one per dataset) if the model function
        returned a list.
    values : dict
        The fitted parameter values, keyed by path, in the model's own
        parameters (not the flat coordinates some are fitted in).
    info : dict
        ``method``; ``converged`` (``None`` for Adam, which has no
        convergence test); ``steps``; ``loss`` (the unscaled negative log
        posterior); ``chi2`` and ``ndata``, per dataset and then per
        likelihood term; and ``chi2_red``,
        the total χ² per data point. With fitted error terms, χ² uses the
        inflated errors, and ``values`` holds the terms too.
    """

    model: object
    values: dict
    info: dict


def fit(
    model,
    priors,
    data,
    regularisers=(),
    *,
    noise=None,
    init=None,
    method=None,
    max_steps=None,
    gtol=1e-4,
    max_step_size=2.0,
    learning_rate=1e-2,
    cg_steps=50,
    dtype="float64",
    likelihoods=(),
):
    """Find the maximum a posteriori parameters of a model given data.

    Parameters
    ----------
    model : SourceModel or callable
        A template model whose leaves at the paths in ``priors`` are fitted
        (their values are the starting point), or a function called with the
        parameters as keyword arguments, as for
        [`numpyro_model`][virgil.likelihood.numpyro_model]. The function
        may return a list of models, one per dataset, sharing parameters:
        for example a scene and a [`Rotated`][virgil.models.Rotated]
        copy of it, for two epochs between which it turns. Regularisers
        then act on the first model only: an Image that appears only in a
        later model is not regularised.
    priors : dict[str, numpyro.distributions.Distribution]
        A prior for each free parameter, keyed by its path (e.g.
        ``"comp.flux"`` or ``"env.log_brightness"``; see
        [`image_priors`][virgil.imaging.image_priors]). Priors on
        fluxes must have non-negative support.

        A prior that is uniform in some coordinate of its parameter has a
        *flat coordinate*, and ``fit`` optimises the parameter in it:
        ``LogUniform(a, b)`` is uniform in ``log x`` on ``[log a, log b]``,
        an isotropic inclination in ``cos i`` and an isotropic latitude in
        ``sin(lat)`` (any prior with a ``flat_coordinate()`` method), and
        ``Uniform`` in the parameter itself. The prior is then constant and
        adds nothing to the loss, so the fit is the maximum of the
        likelihood (times the other priors) inside the prior's range, and
        Levenberg–Marquardt applies. This is the maximum a posteriori in the
        coordinates in which the invariant (Jeffreys) prior is uniform, the
        choice consistent with virgil's prior rule; a mode in ``x`` itself
        would depend on the parametrisation (for ``LogUniform``, the density
        ``1/x`` would pull every scale towards ``a``). Other priors, such as
        ``Normal``, ``Beta`` or ``HalfNormal``, are evaluated in the model's
        own parameters, without the Jacobian of the bijection, as before.
        Either way the returned ``values`` are in the model's own
        parameters.
    data : OIData or sequence of OIData
        The data, fitted jointly. May be empty (``()``) when
        ``likelihoods`` holds all the data.
    regularisers : sequence, optional
        Penalties added to the loss, e.g. from
        [`virgil.imaging`][virgil.imaging].
    noise : dict or list of dict, optional
        Priors on error-inflation terms to fit with the parameters:
        ``vis_scale`` and ``phi_scale`` multiply the uncertainties, and
        ``vis_error_rel`` (a fraction of the model visibility) and
        ``phi_error`` (radians) are added in quadrature (see
        [`inflated_errors`][virgil.likelihood.inflated_errors]), and the
        widths of gains correlated across channels, ``vis_gain_<group>``
        (see [`OIData.with_gains`][virgil.oidata.OIData.with_gains]), and
        of closure-phase offsets, ``phi_offset_<group>`` (see
        [`OIData.with_closure_offsets`][virgil.oidata.OIData.with_closure_offsets]), and
        the wavelength scale, ``wavel_scale`` and ``wavel_offset`` (see
        [`OIData.with_wavelength_scale`][virgil.oidata.OIData.with_wavelength_scale]), and the North
        angle, ``north_angle`` in degrees (see
        [`OIData.with_north_angle`][virgil.oidata.OIData.with_north_angle]),
        whose priors may be of either sign (e.g. ``Normal(1, 2e-4)``). A dict
        applies to every dataset (values ``"noise.<term>"``); a list gives
        each dataset its own (``"noise[i].<term>"``). The loss is then the
        full Gaussian negative log likelihood, including ``Σ log σ``, so the
        default method is L-BFGS. So it is for data with gains, whose
        covariance depends on the model. Fitting error terms with an image is
        degenerate (a smoother image with larger errors fits as well):
        estimate them with a parametric model first.
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
    init : dict, optional
        Starting values by path (or ``noise`` site), overriding the
        template's (required for a function model).
    method : {"lm", "lbfgs", "adam"}, optional
        ``"lm"``: Levenberg–Marquardt (optimistix) on the residuals. Its
        inner solve is a dense QR of the Jacobian for up to 200 unconstrained
        coordinates, and otherwise matrix-free (``cg_steps``
        conjugate-gradient steps on the normal equations), so that the
        Jacobian of a larger image is never formed. The default when the whole
        objective has a least-squares form: Normal priors, and priors that
        are flat in their fitted coordinate (Uniform, ImproperUniform,
        LogUniform and the other priors with a flat coordinate).
        ``"lbfgs"``: L-BFGS (optax) on the loss, for penalties
        such as maximum entropy and total variation; the default otherwise.
        No unconstrained coordinate moves by more than ``max_step_size``
        per step, so that log-brightness pixels cannot be switched off by
        one long step.
        ``"adam"``: Adam with ``learning_rate``, run for ``max_steps``.
    max_steps : int, optional
        Step limit (defaults: 1000 for LM, 20000 for L-BFGS, 2000 for Adam).
        It sets the length of LM's and Adam's loops, so a new value
        recompiles them (not L-BFGS); other numbers do not.
    gtol : float, optional
        LM and L-BFGS stop when no component of the gradient of the loss
        per data point exceeds ``gtol``, nor 1/1000 of its largest starting
        value (so that a fit started near a solution, e.g. along an
        L-curve, still converges). The tolerance is never below √eps of the
        dtype times that starting value (3.5e-4 times it in float32), which
        rounding errors in the gradient would not let the fit reach, nor
        below ``1e-6 * gtol``, so that a fit started exactly at a
        zero-residual optimum, whose gradient is rounding noise, is
        converged at once.
    max_step_size : float, optional
        L-BFGS moves no unconstrained coordinate by more than this per step
        (for a log-brightness pixel, a factor ``exp(max_step_size)``).
        Uncapped, its line search accepts steps that drive pixels so dark
        that their gradient vanishes and they never recover, leaving the fit
        at a spurious stationary point far above the minimum. Coordinates
        with real support (e.g. a position under a Normal prior) are in
        their own units, so raise this if they must move far.
    learning_rate : float, optional
        Adam's learning rate, in unconstrained coordinates.
    cg_steps : int, optional
        Conjugate-gradient steps per LM step, for more than 200 coordinates.
        The inner solve runs for exactly this many steps, or as many as
        there are coordinates if fewer: its tolerances are zero, because an
        inner solve that stops at a step limit would abort the outer one.
    dtype : {"float64", "float32"}, optional
        Precision of the fit. The default runs in float64 inside a local
        ``jax.enable_x64`` context. The returned model and values are cast
        back to JAX's precision outside the fit (float32, unless x64 is
        enabled), so that they work with the rest of your code.
    likelihoods : sequence, optional
        Further Gaussian likelihood terms that are not visibilities: each is
        a callable of the fitted values (a dict, by path or keyword) that
        returns whitened residuals, such as
        [`PositionData.term`][virgil.orbits.PositionData.term] or
        [`RVData.term`][virgil.orbits.RVData.term] for an orbit's
        positions and radial velocities. Their χ² follow the datasets' in
        ``info``.

    Returns
    -------
    FitResult
        The fitted model, parameter values and diagnostics. A warning is
        raised if LM or L-BFGS did not converge; for L-BFGS it says whether
        the fit reached ``max_steps``, met a non-finite gradient, or stopped
        earlier because a step no longer changed the parameters (its
        precision ran out, as can happen in float32).
    """
    if not math.isfinite(max_step_size) or max_step_size <= 0:
        raise ValueError(
            f"max_step_size must be finite and positive, not {max_step_size}."
        )
    with run_in(dtype):
        problem = cast_tree(
            _Objective(model, priors, data, regularisers, noise, likelihoods),
            dtype,
        )
        z0 = problem.init(cast_tree(init, dtype))
        method = method or ("lm" if _has_residuals(problem, z0) else "lbfgs")
        # Optimisers see the loss per data point, so step sizes and
        # tolerances do not depend on the size of the dataset.
        ndata = [d.n_independent for d in problem.data]
        # A term's size is the length of its residuals at the start.
        start = problem.constrain(z0)
        ndata += [int(np.size(term(start))) for term in problem.likelihoods]
        scale = float(max(sum(ndata), 1))
        # Numbers go to the jitted solvers as arrays, which are traced, so
        # that new values do not recompile. The step limits of LM and Adam
        # set a loop's length, so they stay static.
        traced_scale, gtol = np.asarray(scale), np.asarray(gtol)
        if method == "lm":
            z, steps, converged = _lm(
                problem, z0, traced_scale, max_steps or 1000, gtol, cg_steps
            )
        elif method == "lbfgs":
            limit = max_steps or 20_000
            z, steps, converged, stop = _lbfgs(
                problem,
                z0,
                traced_scale,
                np.asarray(limit),
                gtol,
                np.asarray(max_step_size),
            )
        elif method == "adam":
            steps = max_steps or 2000
            z, converged = _adam(
                problem, z0, traced_scale, np.asarray(learning_rate), steps
            )
        else:
            raise ValueError(
                f"method must be 'lm', 'lbfgs' or 'adam', not {method!r}."
            )
        if converged is False:
            warnings.warn(
                _lbfgs_not_converged(stop, steps, limit, dtype)
                if method == "lbfgs"
                else f"fit(method={method!r}) did not converge in {steps} "
                "steps.",
                RuntimeWarning,
                stacklevel=2,
            )
        model = problem.build(z)
        values = problem.constrain(z)
        loss, chi2 = _summary(problem, z)
        chi2 = [float(c) for c in chi2]
        info = {
            "method": method,
            "converged": converged,
            "steps": steps,
            "loss": float(loss),
            "chi2": chi2,
            "ndata": ndata,
            "chi2_red": sum(chi2) / scale,
        }
    ambient = "float64" if jax.config.jax_enable_x64 else "float32"
    return FitResult(
        cast_tree(model, ambient), cast_tree(values, ambient), info
    )


def gauss_newton_mass(model, priors, data, values):
    """A dense NUTS mass matrix from the Gauss–Newton curvature at a fit.

    Near the maximum a posteriori, the posterior is close to a Gaussian
    whose precision, in the unconstrained coordinates z that numpyro samples,
    is the Gauss–Newton matrix JᵀJ. Here J is the Jacobian of the
    whitened residuals with respect to z, including the residuals of the
    priors (so a standard-normal prior adds the identity). Giving NUTS the
    inverse, (JᵀJ)⁻¹, as its inverse mass matrix whitens that Gaussian.
    Directions the data fix tightly then take the same step size as those
    left to the prior. Without it, the step size shrinks to suit the
    tightest direction, and NUTS needs its full tree depth (1023 leapfrog
    steps) per draw. On a 62² Gaussian-field image fitted to 588 AMI
    observables, it cut the cost to 63 steps per draw.

    Use it at fixed field hyperparameters (σ and ℓ, chosen for example by
    [`log_evidence`][virgil.imaging.log_evidence]). The curvature
    depends on them, so a matrix computed at one σ and ℓ is wrong when
    they move. Every sampled parameter must be in ``priors``: a
    tightly constrained one left out (such as an image's flux) keeps its
    identity mass and its tiny step size.

    Parameters
    ----------
    model, priors, data
        As for [`fit`][virgil.fitting.fit]. The priors must have a
        least-squares form, as for ``fit``'s Levenberg–Marquardt: Normal
        priors, and flat ones (Uniform, ImproperUniform, LogUniform and the
        other priors with a flat coordinate). Flat priors add no curvature,
        as for Uniform. The matrix is in numpyro's unconstrained
        coordinates (``biject_to`` of each prior's support), which NUTS
        samples, not in ``fit``'s flat coordinates: for ``LogUniform(a,
        b)`` that is the logit of ``(x - a) / (b - a)``, not of ``log x``.
    values : dict
        The parameter values at which to take the curvature, normally
        ``fit(model, priors, data).values``.

    Returns
    -------
    dict
        Keyword arguments for ``numpyro.infer.NUTS``:
        ``inverse_mass_matrix``, ``dense_mass``, and
        ``adapt_mass_matrix=False``. Warmup adaptation is switched off
        because a dense covariance estimated from a few hundred draws in
        thousands of dimensions is far worse than this matrix.

    Examples
    --------
    >>> result = fit(scene, priors, data)
    >>> kernel = NUTS(numpyro_model(result.model, priors, data),
    ...               init_strategy=init_to_value(values=result.values),
    ...               **gauss_newton_mass(scene, priors, data, result.values))
    """
    with run_in("float64"):
        # NUTS samples numpyro's unconstrained coordinates, so the
        # curvature is taken in those, not in fit's flat coordinates.
        problem = cast_tree(
            _Objective(model, priors, data, flat=False), "float64"
        )
        z = problem.init(cast_tree(values, "float64"))
        covariance, ok = _gauss_newton_covariance(problem, z)
        covariance = onp.asarray(covariance)
    if not ok:
        raise ValueError(
            "The Gauss–Newton curvature is singular: a parameter in priors "
            "is constrained by neither the data nor a Normal prior."
        )
    paths = problem.paths
    return {
        "inverse_mass_matrix": {paths: covariance},
        "dense_mass": [paths],
        "adapt_mass_matrix": False,
    }


@eqx.filter_jit
def _gauss_newton_covariance(problem, z):
    """(JᵀJ)⁻¹ over the parameters, in the order of ``problem.paths``.

    J is the Jacobian of the residuals, the data's and the priors', with
    respect to the unconstrained coordinates ``z``. The priors' residuals
    are elementwise, so they add a diagonal D to JᵀJ, and only the data's
    rows are differentiated (in reverse mode, one pass per datum). Returns
    the covariance and whether JᵀJ + D was positive definite.
    """
    paths = problem.paths
    shapes = [np.shape(z[p]) for p in paths]
    sizes = [math.prod(shape) for shape in shapes]

    def unflatten(x):
        pieces = np.split(x, onp.cumsum(sizes)[:-1])
        return {p: v.reshape(s) for p, v, s in zip(paths, pieces, shapes)}

    def data_residuals(x):
        model = problem.build(unflatten(x))
        residuals = problem.data_residuals(model)
        # With data=() there are no residuals: an empty vector.
        return np.concatenate(residuals) if residuals else np.zeros(0)

    def prior_curvature(path):
        prior = problem.priors[path]
        if _prior_residuals(path, prior, z[path]) is None:
            return None  # a flat prior
        bijection = _bijection(prior)

        def residuals(v):
            return _prior_residuals(path, prior, bijection(v))

        ones = np.ones_like(z[path])
        return jax.jvp(residuals, (z[path],), (ones,))[1] ** 2

    curvatures = [prior_curvature(p) for p in paths]
    # Which coordinates have flat priors is known when tracing.
    flat_prior = onp.concatenate(
        [onp.full(n, c is None) for n, c in zip(sizes, curvatures)]
    )
    d = np.concatenate(
        [np.zeros(n) if c is None else c for n, c in zip(sizes, curvatures)]
    )
    flat = np.concatenate([np.ravel(z[p]) for p in paths])
    # Forward mode costs a pass per parameter, reverse a pass per residual.
    n, n_data = flat.size, jax.eval_shape(data_residuals, flat).size
    jacobian = jax.jacfwd if n_data >= n else jax.jacrev
    jac = jacobian(data_residuals)(flat)
    if n_data >= n:
        factor = jax.scipy.linalg.cho_factor(jac.T @ jac + np.diag(d))
        covariance = jax.scipy.linalg.cho_solve(factor, np.eye(n))
        ok = np.all(np.isfinite(factor[0])) & np.all(np.diag(factor[0]) > 0)
        return covariance, ok
    # Fewer data than parameters (an image's latents): by Woodbury,
    # (D + JᵀJ)⁻¹ = W (I - Kᵀ(I + KKᵀ)⁻¹K) W with W = D^-1/2 and K = JW,
    # which solves a system the size of the data, not of the parameters.
    # Coordinates with flat priors (a few, e.g. a flux) have no D: give
    # them D = 1, then take that back off, (A - EᵀE)⁻¹ = A⁻¹ +
    # A⁻¹Eᵀ(I - EA⁻¹Eᵀ)⁻¹EA⁻¹ with E picking them out of A = D + JᵀJ.
    w = 1 / np.sqrt(np.where(flat_prior, 1.0, d))
    k = jac * w
    inner = jax.scipy.linalg.cho_factor(np.eye(n_data) + k @ k.T)
    shrink = k.T @ jax.scipy.linalg.cho_solve(inner, k)
    a_inv = w[:, None] * (np.eye(n) - shrink) * w[None, :]
    flats = onp.flatnonzero(flat_prior)
    if flats.size == 0:
        return a_inv, np.asarray(True)  # D + JᵀJ is positive definite
    columns = a_inv[:, flats]
    schur = jax.scipy.linalg.cho_factor(np.eye(flats.size) - columns[flats])
    covariance = a_inv + columns @ jax.scipy.linalg.cho_solve(schur, columns.T)
    ok = np.all(np.isfinite(schur[0])) & np.all(np.diag(schur[0]) > 0)
    return covariance, ok


def _has_residuals(problem, z):
    """Whether the whole objective has a least-squares form."""
    try:
        jax.eval_shape(problem.residuals, z)
    except TypeError:
        return False
    return True


# The solvers below are jitted at module level, with the problem as an
# argument, so that repeated fits of problems with the same structure (an
# L-curve, a grid of hyperparameters, a fit per dataset) reuse one
# compilation. A function defined inside the call would be a new function
# each time and recompile.


@eqx.filter_jit
def _summary(problem, z):
    """The loss and each dataset's chi-squared at ``z``.

    With fitted error terms, chi-squared uses the inflated errors.
    """
    values = problem.constrain(z)
    residuals = problem.data_residuals(problem.build(z), values)
    residuals += [term(values) for term in problem.likelihoods]
    chi2 = [np.sum(r**2) for r in residuals]
    return problem.loss(z), chi2


def _largest(tree):
    """The largest absolute value in a pytree of arrays."""
    return np.max(np.stack([np.max(np.abs(x)) for x in jax.tree.leaves(tree)]))


def _scaled_loss(problem, scale):
    return lambda z: problem.loss(z) / scale


def _tolerance(gradient, gtol):
    """The gradient tolerance of LM and L-BFGS, from the starting gradient.

    It is ``gtol``, or 1/1000 of the starting gradient's largest component
    if that is smaller (so that a warm start still converges), but no less
    than √eps times it. The gradient is a sum of terms that cancel at the
    minimum, and rounding them leaves it a floor which, relative to the
    starting gradient, is about eps times the data's signal-to-noise ratio
    over the starting residuals: up to ~1e-4 in float32 (eps ≈ 1.2e-7),
    where a fixed ``gtol = 1e-4`` was out of reach and LM ran all its
    steps. The √eps floor (3.5e-4 in float32, 1.5e-8 in float64) clears
    it, and is the usual stopping limit for finite-precision optimisers.
    Neither part is allowed below ``1e-6 * gtol``.
    """
    largest = _largest(gradient)
    floor = np.sqrt(np.finfo(largest.dtype).eps) * largest
    # Both floors scale with the starting gradient, which at an exact,
    # zero-residual optimum is itself rounding noise; 1e-6 * gtol keeps
    # such a start reachable (converged at once).
    return np.maximum(
        np.minimum(gtol, 1e-3 * largest), np.maximum(floor, 1e-6 * gtol)
    )


class _GradientStoppedLM(optx.LevenbergMarquardt):
    """Levenberg-Marquardt that stops when the gradient of the loss is small.

    optimistix's own test, on the change of the residuals, passes at once
    from a warm start, and a test on the parameters never passes for
    log-brightness pixels that should be dark (they drift towards -inf
    without changing the image). This stops like the L-BFGS path instead.
    """

    tolerance: jax.Array

    def __init__(self, tolerance, linear_solver):
        super().__init__(rtol=0.0, atol=0.0, linear_solver=linear_solver)
        self.tolerance = tolerance

    def terminate(self, fn, y, args, options, state, tags):
        problem, scale = args
        gradient = jax.grad(_scaled_loss(problem, scale))(y)
        return _largest(gradient) <= self.tolerance, optx.RESULTS.successful


def _lm_residuals(z, args):
    problem, scale = args
    return problem.residuals(z) / np.sqrt(scale)


@eqx.filter_jit
def _lm_tolerance(problem, z0, scale, gtol):
    gradient = jax.grad(_scaled_loss(problem, scale))(z0)
    return _tolerance(gradient, gtol)


# Up to this many unconstrained coordinates, LM's inner solve is a dense QR
# of the Jacobian, which costs one Jacobian-vector product per coordinate.
_DENSE_LM = 200


def _lm_linear_solver(z0, cg_steps):
    """QR for a few coordinates, else CG on the normal equations.

    CG runs for exactly ``cg_steps`` steps (its tolerances are zero),
    because an inner solve that stops at a step limit would abort the outer
    one; it is exact after as many steps as there are coordinates, so it
    runs no more than that.
    """
    n = sum(int(np.size(x)) for x in jax.tree.leaves(z0))
    if n <= _DENSE_LM:
        return lx.QR()
    return lx.Normal(lx.CG(rtol=0.0, atol=0.0, max_steps=min(cg_steps, n)))


def _lm(problem, z0, scale, max_steps, gtol, cg_steps):
    tolerance = _lm_tolerance(problem, z0, scale, gtol)
    solver = _GradientStoppedLM(tolerance, _lm_linear_solver(z0, cg_steps))
    solution = optx.least_squares(
        _lm_residuals,
        solver,
        z0,
        args=(problem, scale),
        max_steps=max_steps,
        throw=False,
    )
    converged = bool(solution.result == optx.RESULTS.successful)
    return solution.value, int(solution.stats["num_steps"]), converged


def _capped_lbfgs(max_step_size):
    """optax's L-BFGS, with no coordinate moving more than ``max_step_size``.

    The L-BFGS direction is scaled down to the cap before the zoom line
    search, whose step is then at most 1, so the line search evaluates the
    points actually taken (as ``optax.value_and_grad_from_state`` needs).
    Without the cap, a softmax image can take a step of ~20 in its
    log-brightnesses that sends most pixels to ~exp(-20): their gradient,
    proportional to their brightness, then vanishes, and the fit stalls.
    Scaling the whole step, rather than clipping each coordinate, keeps the
    quasi-Newton direction.
    """

    def cap(updates, state, params=None):
        factor = np.minimum(1.0, max_step_size / _largest(updates))
        return jax.tree.map(lambda u: u * factor, updates), state

    return optax.chain(
        optax.scale_by_lbfgs(),
        optax.scale(-1.0),
        optax.GradientTransformation(lambda params: optax.EmptyState(), cap),
        optax.scale_by_zoom_linesearch(
            max_linesearch_steps=20,
            max_learning_rate=1.0,
            initial_guess_strategy="one",
        ),
    )


@eqx.filter_jit
def _lbfgs_run(problem, z0, scale, max_steps, gtol, max_step_size):
    # optax's L-BFGS (with a zoom line search, and capped steps; see
    # _capped_lbfgs), stopped on the gradient.
    # A gradient test suits log-brightness pixels: the gradient for a pixel
    # that should be dark vanishes with its flux, while its value keeps
    # drifting, and a test on the loss change can stop at the first short
    # line-search step. The gradient must also fall by a factor of 1000 from
    # where it started, so that a warm start (e.g. along an L-curve, whose
    # gradient is small from the outset) still converges rather than
    # stopping at once. The fit also stops, unconverged, when a step no
    # longer changes the parameters (the line search has run out of
    # precision, as can happen in float32).
    optimiser = _capped_lbfgs(max_step_size)
    loss = _scaled_loss(problem, scale)
    value_and_grad = optax.value_and_grad_from_state(loss)
    tolerance = _tolerance(jax.grad(loss)(z0), gtol)

    def keep_going(carry):
        step, _, _, gradient, moved = carry
        return (step < max_steps) & (gradient > tolerance) & moved

    def step(carry):
        count, z, state, _, _ = carry
        value, grad = value_and_grad(z, state=state)
        updates, state = optimiser.update(
            grad, state, z, value=value, grad=grad, value_fn=loss
        )
        new = optax.apply_updates(z, updates)
        moved = jax.tree.reduce(
            np.logical_or,
            jax.tree.map(lambda a, b: np.any(a != b), new, z),
        )
        return count + 1, new, state, _largest(grad), moved

    # The start itself may already be a stationary point.
    start_gradient = _largest(jax.grad(loss)(z0))
    start = (0, z0, optimiser.init(z0), start_gradient, np.asarray(True))
    count, z, _, gradient, moved = jax.lax.while_loop(keep_going, step, start)
    return z, count, gradient, tolerance, moved


def _lbfgs_not_converged(stop, steps, limit, dtype):
    """The warning for an unconverged L-BFGS fit, saying why it stopped."""
    head = "fit(method='lbfgs') did not converge"
    if stop == "limit":
        return f"{head} in {steps} steps, the step limit; raise max_steps."
    if stop == "non-finite":
        return (
            f"{head}: its gradient was not finite (NaN or inf) after "
            f"{steps} steps; check the model and priors for invalid values."
        )
    hint = "; dtype='float64' can go further" if dtype == "float32" else ""
    return (
        f"{head}: it stopped after {steps} of {limit} steps, when a step no "
        f"longer changed the parameters (the line search ran out of {dtype} "
        f"precision{hint})."
    )


def _lbfgs(problem, z0, scale, max_steps, gtol, max_step_size):
    """L-BFGS, returning ``(z, steps, converged, stop)``.

    ``stop`` says why an unconverged fit ended: ``"limit"`` (``max_steps``),
    ``"stalled"`` (a step no longer moved the parameters) or
    ``"non-finite"`` (a NaN or infinite gradient); it is ``None`` when the
    fit converged.
    """
    z, count, gradient, tolerance, moved = _lbfgs_run(
        problem, z0, scale, max_steps, gtol, max_step_size
    )
    count, gradient, tolerance = int(count), float(gradient), float(tolerance)
    if gradient <= tolerance:
        return z, count, True, None
    # The loop starts from an infinite gradient, so only a gradient seen
    # after a step (or a non-finite starting tolerance) is a real failure.
    if not math.isfinite(tolerance) or (
        count > 0 and not math.isfinite(gradient)
    ):
        return z, count, False, "non-finite"
    return z, count, False, "limit" if bool(moved) else "stalled"


@eqx.filter_jit
def _adam_run(problem, z0, scale, learning_rate, steps):
    optimiser = optax.adam(learning_rate)
    grad = jax.grad(_scaled_loss(problem, scale))

    def step(carry, _):
        z, state = carry
        updates, state = optimiser.update(grad(z), state, z)
        return (optax.apply_updates(z, updates), state), None

    (z, _), _ = jax.lax.scan(
        step, (z0, optimiser.init(z0)), None, length=steps
    )
    return z


def _adam(problem, z0, scale, learning_rate, steps):
    return _adam_run(problem, z0, scale, learning_rate, steps), None
