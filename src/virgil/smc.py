"""Adaptive tempered sequential Monte Carlo on BlackJAX.

!!! warning "Provisional"
    May change in any 0.4.x release without a deprecation cycle.

[`tempered_smc`][virgil.smc.tempered_smc] tempers a prior to prior × L;
[`bridge_smc`][virgil.smc.bridge_smc] tempers prior × L₁ to prior × L₂ from
given particles, ``π_β ∝ prior · L₁^(1-β) · L₂^β``. Both return the
particles, their weights, ``log Z`` (the log evidence, or the log ratio
``Z₂/Z₁`` for a bridge) with a Monte Carlo error from a single run, and the
schedule. The error is the genealogy-based estimator of Lee & Whiteley (2018),
extended to adaptive SMC by Du & Guyader (2021); it needs multinomial
resampling, which is what is used. When the genealogy has collapsed to one
ancestor the error is ``nan`` and ``variance_degenerate`` is set. The next β solves for an ESS of one half of the incremental
weights of the (resampled) particles, as in BlackJAX's adaptive tempering
(Jasra et al. 2011; Zhou, Johansen & Aston 2016). The mutation is fixed-length HMC with the
step size and trajectory length tuned per stage and the mass matrix set to the
particle covariance (Buchholz, Chopin & Jacob 2021). A run that reaches the
time or stage cap has not reached β = 1: it returns ``stopped=True`` and no
particles. Parameters are a flat vector; the log densities take one.
With x64 on, use float64 particles: BlackJAX draws the HMC momentum in the
default float width.

Source: BlackJAX (Cabezas et al. 2024), Del Moral, Doucet & Jasra 2006,
Jasra et al. 2011, Lee & Whiteley 2018, Du & Guyader 2021 (see
``references/provenance.toml``).
"""

import dataclasses
import time
from typing import NamedTuple

import blackjax
import jax
import jax.numpy as np
import numpy as onp
from scipy import stats
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
        Standard error of ``log_z`` from the genealogy-based variance
        estimator; ``nan`` if ``variance_degenerate``.
    log_z_variance : float
        The unclipped variance estimate behind ``log_z_error``; it is unbiased
        and so can be negative. ``nan`` if every particle of a replicate
        shares one ancestor, or if ``stopped``.
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
    variance_degenerate : bool
        The error estimate is unusable: all particles of a replicate share
        one ancestor, or the estimate is not positive. Use more particles.
    log_z_floor_variance : float
        ``Σ_r u_r² Σ_t (1/ESS_t − 1/N)`` over the replicates (``u_r`` the
        replicates' normalized ``Ẑ_r``): the variance of ``Ẑ`` for kernels
        that mix perfectly (Chopin 2004; Bérard, Del Moral & Doucet 2014).
    log_z_rep_variance : float
        ``(t₃(0.975) / z(0.975))² s² / R``, with ``s²`` the sample variance of
        the replicates' ``log Ẑ_r`` (``nan`` for fewer than three): the
        scatter of the replicates, widened for its ``R − 1`` degrees of
        freedom.
    mixing_z : float
        With replicates at two different ``steps``, the pooled ``log Ẑ`` at the
        larger ``steps`` minus that at the smaller, over its standard error
        (``max`` of the genealogy and floor variances of each half): the
        convergence check against the mutation length (Chopin &
        Papaspiliopoulos 2020). ``nan`` otherwise.
    steps_per_replicate : tuple of int
        The HMC mutations per stage of each replicate.
    """

    particles: object
    weights: object
    log_z: float
    log_z_error: float
    log_z_variance: float
    log_z_replicates: object
    betas: list
    ess: list
    stopped: bool
    undersampled: bool
    variance_degenerate: bool
    log_z_floor_variance: float = float("nan")
    log_z_rep_variance: float = float("nan")
    mixing_z: float = float("nan")
    steps_per_replicate: tuple = ()


def _block(particles):
    """The continuous block HMC moves: ``particles["x"]``, or the array."""
    return particles["x"] if isinstance(particles, dict) else particles


class _Position(NamedTuple):
    position: object


def _build(
    logprior, loglike, dtype, d, n_particles, target_ess, steps, tune, moves,
    composite, masses,
):  # fmt: skip
    """Jitted one-stage function: tune, mutate, reweight.

    ``masses`` (``None``, or one covariance per label ``particle["c"]``) are the
    fixed inverse mass matrices of the HMC; ``None`` sets it to the cloud's.
    """
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

            if masses is not None:  # this label's mass: it never depends on x
                inverse_mass_matrix = inverse_mass_matrix[p["c"]]

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
            if masses is not None:  # each particle's jump in its own metric
                chol = np.linalg.cholesky(masses)

                def own(prev, nxt, acc):
                    r = jax.scipy.linalg.solve_triangular(
                        chol[prev["c"]], prev["x"] - nxt["x"], lower=True
                    )
                    return acc * np.sum(r**2)

                return jax.vmap(own)
            inner = esjd(state.parameter_override["inverse_mass_matrix"][0])
            return lambda prev, nxt, acc: inner(_block(prev), _block(nxt), acc)

        mcmc_init, mcmc_step = init, step
        extra = {"performance_of_chain_measure_factory": measure}
    else:
        mcmc_init, mcmc_step, extra = hmc.init, hmc.build_kernel(), {}
    initial = {
        "step_size": np.full((n_particles,), step_size, dtype),
        "num_integration_steps": np.full((n_particles,), leapfrogs, int),
        "inverse_mass_matrix": (
            np.eye(d, dtype=dtype) if masses is None else masses
        )[None],
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
        # Multinomial, as the variance estimator of Lee & Whiteley requires.
        resampling.multinomial,
        steps,
        initial,
        pretune,
        target_ess=target_ess,
    )

    @jax.jit
    def stage(key, state):
        if masses is not None:
            return algorithm.step(key, state)
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
    """One replicate: ``(particles, weights, log Z, betas, ess, stopped, v)``.

    ``v`` estimates ``var(Ẑ) / Z²`` (``nan`` if the genealogy collapsed).
    """
    state = init(particles)
    n = _block(particles).shape[0]
    tol = 8 * float(np.finfo(_block(particles).dtype).eps)
    log_z, betas, ess = 0.0, [], []
    # Eve indices (Lee & Whiteley 2018, Algorithm 1): the time-0 ancestor of
    # each particle, carried through BlackJAX's resampling ancestors.
    eve = onp.arange(n)
    *keys, final_key = jax.random.split(key, max_stages + 1)
    for key in keys:
        if time.monotonic() > deadline:
            return None, None, onp.nan, betas, ess, True, onp.nan
        state, info = stage(key, state)
        eve = eve[onp.asarray(info.ancestors)]
        beta = float(state.sampler_state.tempering_param)
        log_z += float(info.log_likelihood_increment)
        betas.append(beta)
        ess.append(float(1.0 / np.sum(state.sampler_state.weights**2)))
        if beta >= 1.0 - tol:
            weights = state.sampler_state.weights
            # Ẑ is a product over the potentials of every stage including the
            # last, so the estimator's final time is the cloud resampled by
            # the last weights (Lee & Whiteley eq. 4 at time n = stages + 1).
            final = onp.asarray(resampling.multinomial(final_key, weights, n))
            sizes = onp.bincount(eve[final], minlength=n)
            pairs = n * n - int((sizes**2).sum())  # ordered pairs, E^i != E^j
            # v = 1 - prod_p N/(N-1) * pairs / (N (N-1)), for phi = 1; the
            # unbiasedness E[Ẑ² v] = var Ẑ (their Theorem 1) holds with
            # multinomial resampling, adaptively by Du & Guyader (2021).
            v = (
                1.0 - (n / (n - 1)) ** (len(betas) + 1) * pairs / (n * (n - 1))
                if pairs > 0
                else onp.nan
            )
            return (
                state.sampler_state.particles, weights, log_z, betas, ess,
                False, v,
            )  # fmt: skip
    return None, None, onp.nan, betas, ess, True, onp.nan


def _pool(log_zs, vs, fs):
    """``(ln mean Ẑ_r, genealogy variance, floor variance)`` of ``ln`` of the
    mean of the unbiased ``Ẑ_r``: the delta method with weights
    ``u_r = Ẑ_r / Σ Ẑ`` and per-replicate relative variances ``vs`` (Lee &
    Whiteley 2018, Theorem 1: ``var(Ẑ_r) ≈ Ẑ_r² v_r``) and ``fs``."""
    shift = log_zs.max()
    u = onp.exp(log_zs - shift)
    u = u / u.sum()
    log_z = float(shift + onp.log(onp.mean(onp.exp(log_zs - shift))))
    return log_z, float(onp.sum(u**2 * vs)), float(onp.sum(u**2 * fs))


def _smc(
    logprior, loglike, starts, key, n_particles, replicates, ess_target,
    steps, step_size, leapfrogs, alpha, max_stages, max_seconds, moves,
    masses=None,
):  # fmt: skip
    deadline = time.monotonic() + max_seconds
    per_run = [steps] * replicates if isinstance(steps, int) else list(steps)
    if len(per_run) != replicates:
        raise ValueError("steps needs one entry per replicate")
    # Built once per distinct ``steps``: replicates share the compiled stage.
    stages = {
        n_steps: _build(
            logprior,
            loglike,
            _block(starts[0]).dtype,
            _block(starts[0]).shape[1],
            n_particles,
            0.5,
            n_steps,
            (step_size, leapfrogs, alpha),
            moves,
            bool(moves) or masses is not None or isinstance(starts[0], dict),
            masses,
        )
        for n_steps in sorted(set(per_run))
    }
    runs = [
        _run(k, *stages[n_steps], start, max_stages, deadline)
        for k, start, n_steps in zip(
            jax.random.split(key, replicates), starts, per_run
        )
    ]
    stopped = any(run[5] for run in runs)
    betas = [onp.asarray(run[3]) for run in runs]
    ess = [onp.asarray(run[4]) for run in runs]
    if stopped:
        return SMCResult(
            None, None, onp.nan, onp.nan, onp.nan, onp.full(len(runs), onp.nan),
            betas, ess, True, False, False, steps_per_replicate=tuple(per_run),
        )  # fmt: skip
    log_zs = onp.array([run[2] for run in runs])
    vs = onp.array([run[6] for run in runs])
    # Σ_t (1/ESS_t − 1/N): the variance of Ẑ_r / Z of a perfectly mixing
    # tempered SMC (Chopin 2004; Bérard, Del Moral & Doucet 2014).
    fs = onp.array([onp.sum(1.0 / e - 1.0 / n_particles) for e in ess])
    log_z, variance, floor = _pool(log_zs, vs, fs)
    degenerate = not variance > 0
    r = replicates
    # The scatter of ln Ẑ_r, widened from the normal to the Student-t interval
    # of its r − 1 degrees of freedom (ratio of the 97.5% points).
    rep = (
        float(
            (stats.t.ppf(0.975, r - 1) / stats.norm.ppf(0.975)) ** 2
            * onp.var(log_zs, ddof=1)
            / r
        )
        if r > 1
        else onp.nan
    )
    mixing = onp.nan
    lo, hi = min(per_run), max(per_run)
    if lo < hi:  # doubled mutation raises a deficit shared by the replicates
        halves = []
        for n_steps in (lo, hi):
            keep = onp.array([k == n_steps for k in per_run])
            m, v, f = _pool(log_zs[keep], vs[keep], fs[keep])
            halves.append((m, max(v if v == v else 0.0, f)))
        total = halves[0][1] + halves[1][1]
        mixing = (halves[1][0] - halves[0][0]) / onp.sqrt(total)
    return SMCResult(
        jax.tree.map(lambda *a: np.concatenate(a), *[run[0] for run in runs]),
        np.concatenate([run[1] for run in runs]) / len(runs),
        log_z,
        onp.nan if degenerate else float(onp.sqrt(variance)),
        variance,
        log_zs,
        betas,
        ess,
        False,
        any(e[-1] < ess_target * n_particles for e in ess),
        degenerate,
        floor,
        rep,
        float(mixing),
        tuple(per_run),
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
    replicates=1,
    ess_target=0.25,
    steps=5,
    leapfrogs=4,
    step_size=None,
    alpha=1.0,
    max_stages=200,
    max_seconds=float("inf"),
    moves=(),
    masses=None,
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
        Independent runs, pooled into one evidence and one particle set. The
        error comes from each run's genealogy, so one run suffices; more are
        for cross-checks (compare ``log_z_replicates``).
    ess_target : float
        Fraction of ``n_particles`` below which the final ESS sets
        ``undersampled``. It is below the schedule's one half so that a
        healthy run, whose final ESS scatters about one half, is not flagged.
    steps : int or sequence of int
        HMC mutations per stage; a sequence gives each replicate its own (the
        mixing check ``SMCResult.mixing_z`` compares the largest with the
        smallest).
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
    masses : array, optional
        ``(K, d, d)`` fixed inverse mass matrices (covariances), one for each
        label ``particle["c"] ∈ [0, K)`` of dict particles. HMC moves ``x``
        given the label with that label's mass: it depends on the auxiliary
        label and never on ``x``, so each move leaves the target invariant
        (Neal 2011, §5.2). The step size and trajectory are tuned on the jump
        in each particle's own metric. ``None``: the mass is the cloud's
        covariance. ``moves`` must leave the label's conditional invariant (a
        Gibbs update of the label).

    Returns
    -------
    SMCResult

    Source: Jasra et al. 2011; Buchholz, Chopin & Jacob 2021; Pompe, Holmes &
    Łatuszyński 2020; Neal 2011.
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
        steps, step, leapfrogs, alpha, max_stages, max_seconds, moves, masses,
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
    replicates=1,
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
