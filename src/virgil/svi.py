"""Variational inference with numpyro's SVI, started from a fit.

!!! warning "Provisional"
    May change in any 0.4.x release without a deprecation cycle.

[`variational`][virgil.svi.variational] takes the same arguments as
[`fit`][virgil.fitting.fit] and
[`numpyro_model`][virgil.likelihood.numpyro_model], and fits a guide (an
approximate posterior) to the posterior by maximizing the evidence lower
bound (ELBO) with numpyro's ``SVI``. The guide works in the unconstrained
coordinates NUTS samples, and starts at the Laplace approximation: its
centre at a [`FitResult`][virgil.fitting.FitResult] and its width from
[`gauss_newton_mass`][virgil.fitting.gauss_newton_mass]. The default guide
is a normalizing flow that bends that Gaussian to follow curved ridges and
skewed scales; a Gaussian guide is close to the Laplace approximation, for
which see [`laplace_cov`][virgil.inference.laplace_cov]. Every guide covers
one mode.

See [Variational inference](../variational_inference.md) for the choice of
guide and how to check the result. References: ADVI (Kucukelbir et al.
2017, https://arxiv.org/abs/1603.00788), the block neural autoregressive
flow (De Cao, Titov & Aziz 2019, https://arxiv.org/abs/1904.04676), the
inverse autoregressive flow (Kingma et al. 2016,
https://arxiv.org/abs/1606.04934) and Pareto-smoothed importance sampling
(Vehtari et al. 2024, https://arxiv.org/abs/1507.02646).
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
from .fitting import (
    FitResult,
    NoLeastSquaresForm,
    NoStartingValue,
    SingularCurvature,
    _check_dense_size,
    gauss_newton_mass,
)
from .likelihood import numpyro_model
from ._deprecate import external_stacklevel

__provisional__ = ("*",)

GUIDES = ("bnaf", "iaf", "mvn", "laplace")

_DENSE_HINT = (
    "Pass guide=autoguide.AutoNormal (a diagonal guide), or use "
    "imaging.laplace_samples."
)


class _SizedLaplace(autoguide.AutoLaplaceApproximation):
    """numpyro's Laplace guide, refusing a Hessian above the size limit."""

    def _setup_prototype(self, *args, **kwargs):
        super()._setup_prototype(*args, **kwargs)
        _check_dense_size(self.latent_dim, "The Laplace guide", _DENSE_HINT)


@dataclasses.dataclass
class VariationalResult:
    """The result of [`variational`][virgil.svi.variational].

    Attributes
    ----------
    samples : dict
        Draws from the guide, ``num_samples`` per site, in the model's own
        parameters and under the site names NUTS uses
        (``MCMC.get_samples()``), deterministic sites included.
    losses : numpy.ndarray
        The loss (the negative ELBO, a Monte Carlo estimate) at each step.
    guide : numpyro.infer.autoguide.AutoGuide
        The fitted guide; ``guide.sample_posterior(key, params,
        sample_shape=(n,))`` draws more.
    params : dict
        The guide's fitted parameters.
    converged : bool
        Whether the loss stopped falling: the mean loss over the last ``window``
        steps is not below that of the ``window`` steps before it by more than
        twice their combined standard error. Necessary, not sufficient.
    info : dict
        ``guide`` (its name), ``steps``, ``elbo`` (minus the mean loss over the
        last window; ``None`` for a point-mass guide, whose loss is given as
        ``log_joint``), ``dense_start`` (whether the width started from the
        Gauss–Newton covariance), ``khat`` (the Pareto k̂ of the importance
        weights p/q of ``num_samples`` draws: below 0.7 the guide is good enough
        to reweight; ``None`` if not computed or for a point-mass guide),
        ``time`` (seconds, compilation included) and ``dtype``.
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

    The guide is ``z = z0 + L0 (shift + L u)``, ``u = flow(ε)``, with ``z0`` and
    ``L0`` fixed at the fit and its Gauss–Newton Cholesky factor
    (``start_tril``), and ``shift``, ``L`` and the flow learnt. Every learnt
    parameter is then in units of the Laplace widths, and the guide starts at
    the Laplace approximation (exactly, without a flow).
    """

    class Guide(base):
        start_tril = None

        def _get_posterior(self):
            _check_dense_size(
                self.latent_dim, "A Laplace-started guide", _DENSE_HINT
            )
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


# deliberate: maps the guide name or callable to its autoguide class, Laplace-started for the flow and mvn families
def _make_guide(guide, posterior, init_loc_fn):
    if callable(guide) and not isinstance(guide, str):
        return guide(posterior, init_loc_fn=init_loc_fn)
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
        return _SizedLaplace(posterior, init_loc_fn=init_loc_fn)
    raise ValueError(
        f"guide must be one of {GUIDES} or a callable, not {guide!r}."
    )


# deliberate: starting guide scales from the Gauss-Newton curvature on active latent sites, init_scale widths on the rest
def _start_scale(
    guide_obj,
    model,
    priors,
    data,
    values,
    likelihoods,
    noise,
    flat,
    init_scale,
):
    """``(L, dense)``: the Cholesky factor of the guide's starting covariance
    (Gauss–Newton at ``values`` for the sites the guide covers, ``init_scale``
    times the identity elsewhere) and whether it is dense."""
    n = guide_obj.latent_dim
    _check_dense_size(n, "The guide's starting width", _DENSE_HINT)
    # Each site's indices in the latent vector, in numpyro's packing order.
    index = guide_obj._unpack_latent(np.arange(n, dtype=float))
    latent_index = {k: onp.asarray(v, int).ravel() for k, v in index.items()}
    sizes = {k: v.size for k, v in latent_index.items()}
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
            noise=noise,
            flat_coordinates=flat,
        )
    except (NoStartingValue, SingularCurvature, NoLeastSquaresForm) as error:
        warnings.warn(
            f"No Gauss–Newton starting width ({error}); the guide starts "
            f"with width init_scale={init_scale}.",
            RuntimeWarning,
            stacklevel=external_stacklevel(),
        )
        return onp.linalg.cholesky(covariance), False
    ((sites, gn),) = mass["inverse_mass_matrix"].items()
    gn = onp.asarray(gn)
    if gn.ndim == 1:  # a diagonal mass
        gn = onp.diag(gn)
    # Indices of each site in GN's matrix (in the order of ``sites``).
    gn_index, start = {}, 0
    for site in sites:
        size = sizes.get(site, math.prod(onp.shape(values.get(site, 0.0))))
        gn_index[site] = onp.arange(start, start + size)
        start += size
    common = [s for s in sites if s in latent_index]
    latent = onp.concatenate([latent_index[s] for s in common])
    source = onp.concatenate([gn_index[s] for s in common])
    covariance[onp.ix_(latent, latent)] = gn[onp.ix_(source, source)]
    return onp.linalg.cholesky(covariance), True


def variational(
    model,
    priors,
    data,
    regularizers=(),
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
    flat_coordinates=False,
    dense_start=True,
    init_scale=0.1,
    window=None,
    psis=True,
    dtype="float64",
    **options,
):
    """Approximate the posterior with a guide fitted by numpyro's SVI.

    Maximizes the ELBO with Adam on the posterior of
    [`numpyro_model`][virgil.likelihood.numpyro_model]. The guide starts at
    ``start`` with the Gauss–Newton width, or width ``init_scale`` given
    ``regularizers``, ``dense_start=False`` or ``start=None`` (then at the
    priors' medians). Guide choice and checks:
    [Variational inference](../variational_inference.md).

    Parameters
    ----------
    model, priors, data
        Scene, priors and data as for
        [`numpyro_model`][virgil.likelihood.numpyro_model].
    regularizers, noise, likelihoods : optional
        Regularizers, noise priors and extra likelihoods; see
        [Common arguments](../common_arguments.md).
    start : FitResult or dict, optional
        Where the guide starts: a fit or its ``values`` (default: the priors'
        medians, width ``init_scale``).
    guide : str or callable, optional
        ``"bnaf"`` (default), ``"iaf"``, ``"mvn"`` or ``"laplace"``; or a
        callable ``guide(model, init_loc_fn=...)`` returning a numpyro
        autoguide. The flows need two or more parameters; the dense guides
        raise ``ValueError`` above 4096 coordinates (use
        [`laplace_samples`][virgil.imaging.laplace_samples] for images).
    steps, num_particles, num_samples : int, optional
        Steps (3000), draws per ELBO estimate (8), draws returned (2000).
    optimizer, learning_rate, key : optional
        Optimizer (default ``optax.adam``), its step size (3e-3), and the
        PRNG key (default ``PRNGKey(0)``).
    flat_coordinates, dtype : optional
        ``flat_coordinates`` as for ``numpyro_model``, and the precision
        (default ``"float64"``, as for ``fit``).
    dense_start, init_scale : optional
        Start the width at the Gauss–Newton covariance (default ``True``;
        skipped, with a warning, given ``regularizers``). ``init_scale``
        (default 0.1) is the width of the coordinates it does not cover, or of
        all of them without it.
    window, psis : optional
        Convergence-test window (default ``steps // 10``); whether to compute
        the Pareto k̂.
    **options
        Fixed error terms and ``reject_unphysical``, as for ``numpyro_model``.

    Returns
    -------
    VariationalResult
        ``samples`` (dict of draws), ``losses``, ``guide``, ``params``,
        ``converged`` and ``info``; warns if the loss had not settled.

    Examples
    --------
    >>> vi = variational(scene, priors, data, start=result)  # doctest: +SKIP
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
        (
            model_c,
            priors_c,
            data_c,
            noise_c,
            regularizers_c,
            likelihoods_c,
        ) = cast_tree(
            (model, priors, data, noise, regularizers, likelihoods), dtype
        )
        values_c = None if values is None else cast_tree(dict(values), dtype)
        posterior = numpyro_model(
            model_c,
            priors_c,
            data_c,
            regularizers_c,
            noise_c,
            likelihoods_c,
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
        point_mass = isinstance(
            guide_obj,
            (autoguide.AutoDelta, autoguide.AutoLaplaceApproximation),
        )
        if dense_start and regularizers and values_c is not None:
            # caveat: gauss_newton_mass takes no regularizers (an image's curvature
            # is singular without them), so with them the start is init_scale wide.
            warnings.warn(
                "The Gauss–Newton starting width leaves out regularizers; "
                f"with them, the guide starts with width "
                f"init_scale={init_scale}.",
                UserWarning,
                stacklevel=external_stacklevel(),
            )
        use_dense = dense_start and not regularizers
        if hasattr(guide_obj, "start_tril"):  # started at the Laplace frame
            guide_obj.start_tril, dense = _start_scale(
                guide_obj,
                model_c,
                priors_c,
                data_c,
                values_c if use_dense else None,
                likelihoods_c,
                noise_c,
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
        if psis and not point_mass:  # undefined for a point-mass guide
            try:
                from numpyro.infer.importance import psis_diagnostic
            except ImportError:  # numpyro < 0.20
                pass
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    khat = float(
                        psis_diagnostic(
                            k_psis,
                            params,
                            posterior,
                            guide_obj,
                            num_particles=num_samples,
                        )
                    )
    losses = onp.asarray(run.losses)
    # converged: the mean loss of the last window is not significantly below
    # that of the window before it
    last, before = losses[-window:], losses[-2 * window : -window]
    if losses.size < 2 * window or not onp.all(onp.isfinite(last)):
        converged = False
    else:
        drop = before.mean() - last.mean()
        noise = math.sqrt((last.var() + before.var()) / window)
        converged = bool(drop <= 2 * noise)
    if not converged:
        warnings.warn(
            f"The SVI loss had not settled after {steps} steps; raise "
            "steps or lower learning_rate, or check info and the losses.",
            RuntimeWarning,
            stacklevel=external_stacklevel(),
        )
    info = {
        "guide": name,
        "steps": steps,
        # A point-mass guide's loss is minus the log joint, not an ELBO.
        "elbo": None if point_mass else float(-losses[-window:].mean()),
        "log_joint": (float(-losses[-window:].mean()) if point_mass else None),
        "dense_start": dense,
        "khat": khat,
        "time": time.perf_counter() - tic,
        "dtype": dtype,
    }
    return VariationalResult(
        samples, losses, guide_obj, params, converged, info
    )
