"""Adaptive tempered sequential Monte Carlo on BlackJAX.

!!! warning "Provisional"
    May change in any 0.4.x release without a deprecation cycle.

[`tempered_smc`][virgil.smc.tempered_smc] tempers a prior to prior × L;
[`bridge_smc`][virgil.smc.bridge_smc] tempers prior × L₁ to prior × L₂ from
given particles, ``π_β ∝ prior · L₁^(1-β) · L₂^β``. Both return the
particles, their weights, ``log Z`` (the log evidence, or the log ratio
``Z₂/Z₁`` for a bridge) with a Monte Carlo error from independent replicates,
and the schedule. The next β solves for an ESS of one half of the incremental
weights of the (resampled) particles, as in BlackJAX's adaptive tempering
(Jasra et al. 2011; Zhou, Johansen & Aston 2016). The mutation is fixed-length HMC with the
step size and trajectory length tuned per stage and the mass matrix set to the
particle covariance (Buchholz, Chopin & Jacob 2021). A run that reaches the
time or stage cap has not reached β = 1: it returns ``stopped=True`` and no
particles. Parameters are a flat vector; the log densities take one.
With x64 on, use float64 particles: BlackJAX draws the HMC momentum in the
default float width.

Source: BlackJAX (Cabezas et al. 2024), Del Moral, Doucet & Jasra 2006,
Jasra et al. 2011 (see ``references/provenance.toml``).
"""

import dataclasses
import time
from typing import NamedTuple

import blackjax
import jax
import jax.numpy as np
import numpy as onp
from blackjax.mcmc import hmc
from blackjax.smc import resampling
from blackjax.smc.pretuning import build_pretune, esjd

__all__ = ["SMCResult", "bridge_smc", "tempered_smc"]

__provisional__ = ("*",)


@dataclasses.dataclass(frozen=True)
class SMCResult:
    """The outcome of a tempered SMC run.

    Attributes
    ----------
    particles : array or None
        ``(replicates · n_particles, d)`` particles at β = 1; ``None`` if
        ``stopped``.
    weights : array or None
        Normalized weights of ``particles`` (sum to one).
    log_z : float
        Log evidence (log ratio for a bridge): the log of the mean of the
        replicates' unbiased ``Ẑ``. ``nan`` if ``stopped``.
    log_z_error : float
        Standard error of ``log_z`` from the replicates; ``nan`` for one.
    log_z_replicates : array
        Each replicate's ``log Ẑ``.
    betas : list of array
        Per replicate, β after each stage.
    ess : list of array
        Per replicate, the ESS of the weights after each stage.
    stopped : bool
        A replicate reached the time or stage cap before β = 1.
    undersampled : bool
        A replicate's final ESS is below ``ess_target · n_particles``.
    """

    particles: object
    weights: object
    log_z: float
    log_z_error: float
    log_z_replicates: object
    betas: list
    ess: list
    stopped: bool
    undersampled: bool


def _block(particles):
    """The continuous block HMC moves: ``particles["x"]``, or the array."""
    return particles["x"] if isinstance(particles, dict) else particles


class _Position(NamedTuple):
    position: object


def _build(
    logprior, loglike, dtype, d, n_particles, target_ess, steps, tune, moves,
    composite,
):  # fmt: skip
    """Jitted one-stage function: tune, mutate, reweight."""
    step_size, leapfrogs, alpha = tune
    if composite:
        # Tuned HMC on the continuous block, then each of ``moves``.
        hmc_kernel = hmc.build_kernel()

        def init(position, logdensity_fn):
            return _Position(position)

        def step(key, state, logdensity_fn, step_size, num_integration_steps,
                 inverse_mass_matrix):  # fmt: skip
            keys = jax.random.split(key, len(moves) + 1)
            p = state.position

            def on_block(x):
                return logdensity_fn(
                    {**p, "x": x} if isinstance(p, dict) else x
                )

            new, info = hmc_kernel(
                keys[0],
                hmc.init(_block(p), on_block),
                on_block,
                step_size=step_size,
                inverse_mass_matrix=inverse_mass_matrix,
                num_integration_steps=num_integration_steps,
            )
            p = (
                {**p, "x": new.position}
                if isinstance(p, dict)
                else new.position
            )
            for k, move in zip(keys[1:], moves):
                p, _ = move(k, p, logdensity_fn)
            return _Position(p), info

        def measure(state):
            inner = esjd(state.parameter_override["inverse_mass_matrix"][0])
            return lambda prev, nxt, acc: inner(_block(prev), _block(nxt), acc)

        mcmc_init, mcmc_step = init, step
        extra = {"performance_of_chain_measure_factory": measure}
    else:
        mcmc_init, mcmc_step, extra = hmc.init, hmc.build_kernel(), {}
    initial = {
        "step_size": np.full((n_particles,), step_size, dtype),
        "num_integration_steps": np.full((n_particles,), leapfrogs, int),
        "inverse_mass_matrix": np.eye(d, dtype=dtype)[None],
    }
    sigma = {
        "step_size": np.asarray(step_size / 4, dtype),
        "num_integration_steps": np.asarray(1.0, dtype),
    }
    pretune = build_pretune(
        mcmc_init,
        mcmc_step,
        alpha=alpha,
        sigma_parameters=sigma,
        n_particles=n_particles,
        natural_parameters=["num_integration_steps"],
        positive_parameters=["step_size"],
        **extra,
    )
    algorithm = blackjax.pretuning(
        blackjax.adaptive_tempered_smc,
        logprior,
        loglike,
        mcmc_step,
        mcmc_init,
        resampling.systematic,
        steps,
        initial,
        pretune,
        target_ess=target_ess,
    )

    @jax.jit
    def stage(key, state):
        particles = _block(state.sampler_state.particles)
        weights = state.sampler_state.weights
        centred = particles - weights @ particles
        cov = (centred * weights[:, None]).T @ centred
        # The jitter keeps the Cholesky in the ESJD finite on a degenerate cloud.
        jitter = np.finfo(cov.dtype).eps * (np.trace(cov) + 1.0)
        cov = (cov + jitter * np.eye(d, dtype=cov.dtype)).astype(
            particles.dtype
        )
        parameters = {
            **state.parameter_override,
            "inverse_mass_matrix": cov[None],
        }
        state = type(state)(state.sampler_state, parameters)
        return algorithm.step(key, state)

    return algorithm.init, stage


def _run(key, init, stage, particles, max_stages, deadline):
    """One replicate: ``(particles, weights, log Z, betas, ess, stopped)``."""
    state = init(particles)
    tol = 8 * float(np.finfo(_block(particles).dtype).eps)
    log_z, betas, ess = 0.0, [], []
    for key in jax.random.split(key, max_stages):
        if time.monotonic() > deadline:
            return None, None, onp.nan, betas, ess, True
        state, info = stage(key, state)
        beta = float(state.sampler_state.tempering_param)
        log_z += float(info.log_likelihood_increment)
        betas.append(beta)
        ess.append(float(1.0 / np.sum(state.sampler_state.weights**2)))
        if beta >= 1.0 - tol:
            return (
                state.sampler_state.particles,
                state.sampler_state.weights,
                log_z,
                betas,
                ess,
                False,
            )
    return None, None, onp.nan, betas, ess, True


def _smc(
    logprior, loglike, starts, key, n_particles, replicates, ess_target,
    steps, step_size, leapfrogs, alpha, max_stages, max_seconds, moves,
):  # fmt: skip
    deadline = time.monotonic() + max_seconds
    # Built once: every replicate shares the compiled stage.
    init, stage = _build(
        logprior,
        loglike,
        _block(starts[0]).dtype,
        _block(starts[0]).shape[1],
        n_particles,
        0.5,
        steps,
        (step_size, leapfrogs, alpha),
        moves,
        bool(moves) or isinstance(starts[0], dict),
    )
    runs = [
        _run(k, init, stage, start, max_stages, deadline)
        for k, start in zip(jax.random.split(key, replicates), starts)
    ]
    stopped = any(run[5] for run in runs)
    betas = [onp.asarray(run[3]) for run in runs]
    ess = [onp.asarray(run[4]) for run in runs]
    if stopped:
        return SMCResult(
            None, None, onp.nan, onp.nan, onp.full(len(runs), onp.nan),
            betas, ess, True, False,
        )  # fmt: skip
    log_zs = onp.array([run[2] for run in runs])
    shift = log_zs.max()
    log_z = float(shift + onp.log(onp.mean(onp.exp(log_zs - shift))))
    # Delta method for log of the mean of the unbiased Z estimates.
    u = onp.exp(log_zs - shift)
    error = (
        float(onp.std(u, ddof=1) / (onp.mean(u) * onp.sqrt(len(runs))))
        if len(runs) > 1
        else onp.nan
    )
    return SMCResult(
        jax.tree.map(lambda *a: np.concatenate(a), *[run[0] for run in runs]),
        np.concatenate([run[1] for run in runs]) / len(runs),
        log_z,
        error,
        log_zs,
        betas,
        ess,
        False,
        any(e[-1] < ess_target * n_particles for e in ess),
    )


def _default_step(d):
    # Optimal HMC step for a Gaussian target after whitening scales as d^(-1/4).
    return float(d) ** -0.25


def tempered_smc(
    logprior,
    loglike,
    init_particles,
    *,
    key,
    n_particles,
    replicates=2,
    ess_target=0.25,
    steps=5,
    leapfrogs=4,
    step_size=None,
    alpha=1.0,
    max_stages=200,
    max_seconds=float("inf"),
    moves=(),
):
    """Temper the prior to prior × likelihood with adaptive SMC.

    Parameters
    ----------
    logprior, loglike : callable
        Normalized log prior and log likelihood of one flat parameter vector.
    init_particles : callable
        ``init_particles(key, n)`` returns ``n`` exact prior draws: an array
        ``(n, d)``, or a dict with the continuous block under ``"x"`` and
        auxiliary arrays (for example discrete labels) under other keys.
        ``logprior`` and ``loglike`` then take the dict.
    key : jax.random.PRNGKey
    n_particles : int
        Particles per replicate.
    replicates : int
        Independent runs; ``log_z_error`` needs at least two and has one
        degree of freedom at two, so it is then only an order of magnitude.
    ess_target : float
        Fraction of ``n_particles`` below which the final ESS sets
        ``undersampled``. It is below the schedule's one half so that a
        healthy run, whose final ESS scatters about one half, is not flagged.
    steps : int
        HMC mutations per stage.
    leapfrogs, step_size, alpha
        Initial trajectory length, initial step size (default ``d**-0.25``)
        and the ESJD offset of the per-stage tuning (Buchholz et al. 2021).
    max_stages, max_seconds
        Caps; reaching either returns ``stopped=True``.
    moves : sequence of callable
        Extra mutations run after the HMC steps of each round, each
        ``move(key, particle, logdensity) -> (particle, info)`` on one
        particle, where ``logdensity`` is the tempered log density at the
        current β. A move must leave that density invariant (a Gibbs update,
        or Metropolis-Hastings with its own acceptance); HMC moves only the
        continuous block.

    Returns
    -------
    SMCResult

    Source: Jasra et al. 2011; Buchholz, Chopin & Jacob 2021.
    """
    keys = jax.random.split(key, replicates + 1)
    starts = [init_particles(k, n_particles) for k in keys[1:]]
    step = (
        _default_step(_block(starts[0]).shape[1])
        if step_size is None
        else step_size
    )
    return _smc(
        logprior, loglike, starts, keys[0], n_particles, replicates, ess_target,
        steps, step, leapfrogs, alpha, max_stages, max_seconds, moves,
    )  # fmt: skip


def bridge_smc(
    logprior,
    loglike1,
    loglike2,
    particles,
    *,
    key,
    weights=None,
    n_particles=None,
    replicates=2,
    ess_target=0.25,
    steps=5,
    leapfrogs=4,
    step_size=None,
    alpha=1.0,
    max_stages=200,
    max_seconds=float("inf"),
    moves=(),
):
    """Bridge prior × L₁ to prior × L₂: ``π_β ∝ prior · L₁^(1-β) · L₂^β``.

    ``particles`` (``(m, d)``, with optional ``weights``) sample prior × L₁;
    each replicate resamples ``n_particles`` of them. ``log_z`` estimates
    ``log(Z₂ / Z₁)``. The replicates share the input cloud, so ``log_z_error``
    excludes the variance of that cloud. Other arguments are those of
    [`tempered_smc`][virgil.smc.tempered_smc].

    Returns
    -------
    SMCResult

    Source: Del Moral, Doucet & Jasra 2006; Zhou, Johansen & Aston 2016.
    """
    particles = jax.tree.map(np.asarray, particles)
    block = _block(particles)
    m = block.shape[0]
    n_particles = m if n_particles is None else n_particles
    w = (
        np.full((m,), 1.0 / m, block.dtype)
        if weights is None
        else np.asarray(weights)
    )
    keys = jax.random.split(key, replicates + 1)
    starts = [
        (lambda i: jax.tree.map(lambda a: a[i], particles))(
            resampling.systematic(k, w / np.sum(w), n_particles)
        )
        for k in keys[1:]
    ]
    step = _default_step(block.shape[1]) if step_size is None else step_size
    # prior · L1 · (L2 / L1)^β is a tempering from prior · L1 to prior · L2.
    return _smc(
        lambda x: logprior(x) + loglike1(x),
        lambda x: loglike2(x) - loglike1(x),
        starts, keys[0], n_particles, replicates, ess_target,
        steps, step, leapfrogs, alpha, max_stages, max_seconds, moves,
    )  # fmt: skip
