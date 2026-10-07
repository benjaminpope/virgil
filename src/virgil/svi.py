"""Variational inference with numpyro's SVI, started from a fit.

[`variational`][virgil.svi.variational] takes the same arguments as
[`fit`][virgil.fitting.fit] and
[`numpyro_model`][virgil.likelihood.numpyro_model], and fits a guide (an
approximate posterior) to ``numpyro_model``'s posterior by maximising the
evidence lower bound (ELBO) with numpyro's ``SVI``. The guide works in the
unconstrained coordinates NUTS samples (the priors' flat coordinates by
default), and starts at the Laplace approximation: its centre at a
[`FitResult`][virgil.fitting.FitResult] and its width from
[`gauss_newton_mass`][virgil.fitting.gauss_newton_mass].

A Gaussian guide (``guide="mvn"``) is close to that Laplace approximation
for a well-behaved posterior: use
[`laplace_cov`][virgil.inference.laplace_cov] or the Gauss–Newton
covariance if a Gaussian is all you need. The default guide is a
normalising flow (a block neural autoregressive flow) after the Laplace
affine map, which can bend and skew the Gaussian to follow curved ridges,
skewed scales and parameters pressed against a prior bound. Every guide
covers one mode.
"""

import dataclasses
import math
import time
import warnings

import jax
import jax.numpy as np
import numpy as onp
import numpyro
import numpyro.distributions as dist
from numpyro.distributions import constraints
from numpyro.distributions.transforms import LowerCholeskyAffine
from numpyro.infer import autoguide

from ._precision import cast_tree, run_in
from .fitting import FitResult, gauss_newton_mass
from .likelihood import numpyro_model

GUIDES = ("bnaf", "iaf", "mvn", "laplace")


@dataclasses.dataclass
class VariationalResult:
    """The result of [`variational`][virgil.svi.variational].

    Attributes
    ----------
    samples : dict
        Draws from the guide, ``num_samples`` per site, in the model's own
        parameters and under the site names NUTS uses
        (``MCMC.get_samples()``), deterministic sites (an angle with an
        [`AngleVector`][virgil.angles.AngleVector] prior, in degrees)
        included.
    losses : numpy.ndarray
        The loss (the negative ELBO, a Monte Carlo estimate) at each step.
    guide : numpyro.infer.autoguide.AutoGuide
        The fitted guide; ``guide.sample_posterior(key, params,
        sample_shape=(n,))`` draws more.
    params : dict
        The guide's fitted parameters.
    converged : bool
        Whether the loss stopped falling: the mean loss over the last
        ``window`` steps is not below that of the ``window`` steps before it
        by more than twice their combined standard error. A plateau is
        necessary, not sufficient: compare with NUTS on a test problem.
    info : dict
        ``guide`` (its name), ``steps``, ``elbo`` (minus the mean loss over
        the last window), ``dense_start`` (whether the guide's width
        started from the Gauss–Newton covariance), ``khat`` (the Pareto
        k̂ of the importance weights p/q of ``num_samples`` draws, PSIS;
        below 0.7 the guide is good enough to reweight, above it the guide
        misses mass the posterior has; ``None`` if not computed), ``time``
        (seconds, compilation included) and ``dtype``.
    """

    samples: dict
    losses: onp.ndarray
    guide: object
    params: dict
    converged: bool
    info: dict


class _NoFlow(autoguide.AutoContinuous):
    """A standard normal in the latent space: with the Laplace frame of
    ``_laplace_started``, a dense Gaussian guide."""

    def get_base_dist(self):
        return dist.Normal(np.zeros(self.latent_dim), 1).to_event(1)


def _laplace_started(base):
    """``base`` (a numpyro flow autoguide, or none) in the Laplace frame.

    numpyro's flow guides start near a standard normal in the
    unconstrained coordinates z, far from a fit, and its Gaussian guides
    learn their centre and width in z itself, whose scales differ by
    orders of magnitude between parameters (a position known to 1 mas on
    a 1000 mas prior range, a log flux known to 10%), so that one step
    size cannot suit them all. Here the guide is

        z = z0 + L0 (shift + L u),   u = flow(ε),   ε ~ N(0, I),

    with z0 and L0 fixed at the fit and the Cholesky factor of its
    Gauss–Newton covariance (``start_tril``), and ``shift`` (from zero),
    the lower-triangular ``L`` (from the identity) and the flow learnt.
    Every learnt parameter is then in units of the Laplace widths, and the
    guide starts at the Laplace approximation (exactly, without a flow).
    """

    class Guide(base):
        start_tril = None

        def _get_posterior(self):
            if getattr(self, "_hidden_dims", ()) is None:
                # numpyro's IAF default, two layers of latent_dim units, is
                # too narrow to bend a banana in a few dimensions.
                self._hidden_dims = [max(16, self.latent_dim)] * 2
            if base is _NoFlow:
                flows = []
            else:
                flows = list(super()._get_posterior().transforms)
            shift = numpyro.param(
                f"{self.prefix}_shift", np.zeros(self.latent_dim)
            )
            scale_tril = numpyro.param(
                f"{self.prefix}_scale_tril",
                np.identity(self.latent_dim),
                constraint=constraints.lower_cholesky,
            )
            return dist.TransformedDistribution(
                self.get_base_dist(),
                flows
                + [
                    LowerCholeskyAffine(shift, scale_tril),
                    LowerCholeskyAffine(self._init_latent, self._start()),
                ],
            )

        def _start(self):
            if self.start_tril is None:
                return np.identity(self.latent_dim)
            return np.asarray(self.start_tril, self._init_latent.dtype)

        def gaussian(self, params):
            """The affine part (centre and Cholesky factor in z): the whole
            guide when there is no flow."""
            start = (
                onp.identity(self.latent_dim)
                if self.start_tril is None
                else onp.asarray(self.start_tril)
            )
            shift = onp.asarray(params[f"{self.prefix}_shift"])
            tril = onp.asarray(params[f"{self.prefix}_scale_tril"])
            return onp.asarray(self._init_latent) + start @ shift, start @ tril

    name = "Gaussian" if base is _NoFlow else base.__name__
    Guide.__name__ = Guide.__qualname__ = f"LaplaceStarted{name}"
    return Guide


def _make_guide(guide, posterior, init_loc_fn):
    if callable(guide) and not isinstance(guide, str):
        return guide(posterior, init_loc_fn)
    bases = {
        "bnaf": autoguide.AutoBNAFNormal,
        "iaf": autoguide.AutoIAFNormal,
        "mvn": _NoFlow,
    }
    if guide in bases:
        return _laplace_started(bases[guide])(
            posterior, init_loc_fn=init_loc_fn
        )
    if guide == "laplace":
        return autoguide.AutoLaplaceApproximation(
            posterior, init_loc_fn=init_loc_fn
        )
    raise ValueError(
        f"guide must be one of {GUIDES} or a callable, not {guide!r}."
    )


def _start_scale(
    guide_obj, model, priors, data, values, likelihoods, flat, init_scale
):
    """The Cholesky factor of the guide's starting covariance.

    The Gauss–Newton covariance at ``values``, reordered into the guide's
    latent vector, for the sites it covers; ``init_scale`` times the
    identity for any others (error terms). Returns ``(L, dense)``.
    """
    shapes = {k: np.shape(v) for k, v in guide_obj._init_locs.items()}
    sizes = {k: math.prod(s) for k, s in shapes.items()}
    n = sum(sizes.values())
    covariance = onp.eye(n) * init_scale**2
    if values is None:
        return onp.linalg.cholesky(covariance), False
    try:
        mass = gauss_newton_mass(
            model,
            priors,
            data,
            values,
            likelihoods=likelihoods,
            flat_coordinates=flat,
        )
    except (ValueError, TypeError, NotImplementedError) as error:
        warnings.warn(
            f"No Gauss–Newton starting width ({error}); the guide starts "
            f"with width init_scale={init_scale}.",
            stacklevel=3,
        )
        return onp.linalg.cholesky(covariance), False
    ((sites, gn),) = mass["inverse_mass_matrix"].items()
    gn = onp.asarray(gn)
    # Offsets of each site in the guide's latent vector and in GN's.
    offsets, start = {}, 0
    for k in shapes:
        offsets[k] = start
        start += sizes[k]
    gn_index, start = {}, 0
    for site in sites:
        size = sizes.get(site, math.prod(onp.shape(values.get(site, 0.0))))
        gn_index[site] = onp.arange(start, start + size)
        start += size
    common = [s for s in sites if s in shapes]
    latent = onp.concatenate(
        [offsets[s] + onp.arange(sizes[s]) for s in common]
    )
    source = onp.concatenate([gn_index[s] for s in common])
    covariance[onp.ix_(latent, latent)] = gn[onp.ix_(source, source)]
    return onp.linalg.cholesky(covariance), True


def _converged(losses, window):
    """Whether the mean loss of the last window is not significantly below
    that of the window before it."""
    if losses.size < 2 * window or not onp.all(onp.isfinite(losses[-window:])):
        return False
    last, before = losses[-window:], losses[-2 * window : -window]
    drop = before.mean() - last.mean()
    noise = math.sqrt((last.var() + before.var()) / window)
    return bool(drop <= 2 * noise)


def variational(
    model,
    priors,
    data,
    regularisers=(),
    *,
    noise=None,
    likelihoods=(),
    start=None,
    guide="bnaf",
    steps=3000,
    optimizer=None,
    learning_rate=3e-3,
    num_particles=8,
    num_samples=2000,
    key=None,
    flat_coordinates=True,
    dense_start=True,
    init_scale=0.1,
    window=None,
    psis=True,
    dtype="float64",
    **options,
):
    """Approximate the posterior with a guide fitted by numpyro's SVI.

    The guide is fitted to the posterior of
    [`numpyro_model`][virgil.likelihood.numpyro_model] with the same
    arguments, by maximising the ELBO (``Trace_ELBO``) with Adam. It works
    in the unconstrained coordinates NUTS samples, and starts at the Laplace
    approximation at ``start``: its centre at the fitted values and, by
    default, its width from
    [`gauss_newton_mass`][virgil.fitting.gauss_newton_mass].

    **Which guide.** ``"mvn"`` is a Gaussian with a dense covariance. For a
    well-behaved posterior it is close to the Laplace approximation that
    ``start`` and ``gauss_newton_mass`` already give (SVI minimises the
    KL divergence over the whole posterior, Laplace uses the curvature at
    the mode, and for a near-Gaussian posterior the two agree), so if a
    Gaussian is all you need, use
    [`laplace_cov`][virgil.inference.laplace_cov] or
    [`laplace_samples`][virgil.imaging.laplace_samples].
    ``"laplace"`` is numpyro's ``AutoLaplaceApproximation``, the MAP and
    the Hessian of the log posterior there. The flows ``"bnaf"`` (the
    default; a block neural autoregressive flow) and ``"iaf"`` (an
    inverse autoregressive flow) bend the Laplace Gaussian into a curved,
    skewed or truncated shape: a banana between a separation and a flux,
    or a scale pressed against its prior bound. They are worth their cost
    when the posterior is not Gaussian in the flat coordinates and NUTS is
    too slow. On small test posteriors (a banana, a curved ridge, a scale
    with a tail to its prior bound) BNAF came closest to NUTS's means,
    widths and 5–95% quantiles, where the Gaussian guides missed the
    banana's curvature and the tail. A flow's cost grows with the square of the
    number of parameters, so for images use ``"mvn"`` or the Laplace
    approximation. A callable ``guide(model, init_loc_fn)``
    returning any numpyro autoguide may be passed instead. Flows need at
    least two parameters.

    **One mode.** Every guide here covers one mode of the posterior. For
    multimodal posteriors (orbits with Ω/ω flips or period aliases,
    images with a mirrored solution), run it from several starts, such as
    the fits of
    [`OrbitStart.chain_values`][virgil.epochs.OrbitStart.chain_values],
    and compare their ELBOs, or reweight the draws by p/q and check
    ``info["khat"]``.

    Parameters
    ----------
    model, priors, data, regularisers, noise, likelihoods
        As for [`numpyro_model`][virgil.likelihood.numpyro_model]
        (and [`fit`][virgil.fitting.fit]).
    start : FitResult or dict, optional
        Where the guide starts: a fit, normally ``fit(model, priors,
        data)``, or its ``values``. ``None`` starts at the priors' medians
        with width ``init_scale``.
    guide : str or callable, optional
        ``"bnaf"`` (default), ``"iaf"``, ``"mvn"`` or ``"laplace"``; see
        above.
    steps : int, optional
        Optimisation steps (default 3000).
    optimizer : numpyro or optax optimiser, optional
        Default ``optax.adam(learning_rate)``.
    learning_rate : float, optional
        Adam's step size when ``optimizer`` is not given (default 3e-3).
        Started at the Laplace approximation, the guide is already close,
        so a small step suffices.
    num_particles : int, optional
        Draws per step in the ELBO estimate (default 8): more give a
        smoother loss and gradient, at proportional cost.
    num_samples : int, optional
        Draws returned in ``samples`` (default 2000), also used for PSIS.
    key : jax.Array, optional
        Random key (default ``PRNGKey(0)``).
    flat_coordinates : bool, optional
        Passed to ``numpyro_model`` and ``gauss_newton_mass`` (default
        ``True``): the coordinates the guide is defined in. A Gaussian guide
        is Gaussian in them, e.g. in the logit of ``log x`` on ``[log a,
        log b]`` for a ``LogUniform(a, b)``.
    dense_start : bool, optional
        Start the guide's width from the Gauss–Newton covariance at
        ``start`` (default ``True``). It needs priors with a least-squares
        form (see ``gauss_newton_mass``); otherwise, or for error terms in
        ``noise``, which it does not cover, the width starts at
        ``init_scale``.
    init_scale : float, optional
        Starting width, in the unconstrained coordinates, of the
        coordinates not covered by the Gauss–Newton covariance (default
        0.1).
    window : int, optional
        Steps per window of the convergence test (default ``steps // 10``).
    psis : bool, optional
        Compute the Pareto k̂ of the importance weights (default ``True``).
    dtype : {"float64", "float32"}, optional
        Precision of the SVI run (default ``"float64"``, in a local
        ``jax.enable_x64`` context, as for ``fit``). The returned samples
        are in this precision.
    **options
        Fixed error terms and ``reject_unphysical``, as for
        ``numpyro_model``.

    Returns
    -------
    VariationalResult
        The draws in the model's parameters (keyed as NUTS's), the loss
        history, the guide and its parameters, and diagnostics. A warning
        is raised if the loss had not settled.

    Examples
    --------
    >>> result = fit(scene, priors, data)  # doctest: +SKIP
    >>> vi = variational(scene, priors, data, start=result)  # doctest: +SKIP
    >>> vi.samples["companion.flux"].std()  # doctest: +SKIP
    """
    import optax
    from numpyro.infer import SVI, Trace_ELBO
    from numpyro.infer.initialization import init_to_median, init_to_value

    if steps < 1:
        raise ValueError(f"steps must be positive, not {steps}.")
    window = max(1, steps // 10) if window is None else int(window)
    key = jax.random.PRNGKey(0) if key is None else key
    values = start.values if isinstance(start, FitResult) else start
    tic = time.perf_counter()
    with run_in(dtype):
        model_c, priors_c, data_c, noise_c = cast_tree(
            (model, priors, data, noise), dtype
        )
        values_c = None if values is None else cast_tree(dict(values), dtype)
        posterior = numpyro_model(
            model_c,
            priors_c,
            data_c,
            regularisers,
            noise_c,
            likelihoods,
            flat_coordinates=flat_coordinates,
            **options,
        )
        init_loc_fn = (
            init_to_median
            if values_c is None
            else init_to_value(values=values_c)
        )
        guide_obj = _make_guide(guide, posterior, init_loc_fn)
        if optimizer is None:
            optimizer = optax.adam(learning_rate)
        svi = SVI(
            posterior,
            guide_obj,
            optimizer,
            Trace_ELBO(num_particles=num_particles),
        )
        k_init, k_run, k_draw, k_psis = jax.random.split(key, 4)
        # Set up the guide's prototype, then its starting width.
        svi.init(k_init)
        dense = False
        name = guide if isinstance(guide, str) else type(guide_obj).__name__
        if hasattr(guide_obj, "start_tril"):  # started at the Laplace frame
            guide_obj.start_tril, dense = _start_scale(
                guide_obj,
                model_c,
                priors_c,
                data_c,
                values_c if dense_start else None,
                likelihoods,
                flat_coordinates,
                init_scale,
            )
        run = svi.run(k_run, steps, progress_bar=False)
        params = run.params
        samples = guide_obj.sample_posterior(
            k_draw, params, sample_shape=(num_samples,)
        )
        samples = {k: onp.asarray(v) for k, v in samples.items()}
        khat = None
        if psis:
            khat = _khat(
                k_psis, params, posterior, guide_obj, num_samples, name
            )
    losses = onp.asarray(run.losses)
    converged = _converged(losses, window)
    if not converged:
        warnings.warn(
            f"The SVI loss had not settled after {steps} steps; raise "
            "steps or lower learning_rate, or check info and the losses.",
            stacklevel=2,
        )
    info = {
        "guide": name,
        "steps": steps,
        "elbo": float(-losses[-window:].mean()),
        "dense_start": dense,
        "khat": khat,
        "time": time.perf_counter() - tic,
        "dtype": dtype,
    }
    return VariationalResult(
        samples, losses, guide_obj, params, converged, info
    )


def _khat(key, params, posterior, guide, num_samples, name):
    """PSIS k̂ of the guide's importance weights, or None.

    Undefined for the Laplace guide (a point mass while it is optimised)
    and for numpyro releases without ``psis_diagnostic``.
    """
    if name == "laplace":
        return None
    try:
        from numpyro.infer.importance import psis_diagnostic
    except ImportError:  # numpyro < 0.20
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        k = psis_diagnostic(
            key, params, posterior, guide, num_particles=num_samples
        )
    return float(k)
