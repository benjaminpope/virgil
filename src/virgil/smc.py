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
(Jasra et al. 2011; Zhou, Johansen & Aston 2016). The mutation is HMC with the
step size and trajectory length tuned per stage (Buchholz, Chopin & Jacob 2021)
and the mass matrix set each stage to the weighted covariance of the cloud, one
per label when particles carry labels (Beskos et al. 2016). The number of
mutation steps adapts: it stops once the cloud has moved a fraction ``1 - TOL``
of the distance between independent draws (Chopin & Papaspiliopoulos 2020), and
reaching the cap sets ``mixing_short``. A run that reaches the
time or stage cap has not reached β = 1: it returns ``stopped=True`` and no
particles. Parameters are a flat vector; the log densities take one.
With x64 on, use float64 particles: BlackJAX draws the HMC momentum in the
default float width.

Source: BlackJAX (Cabezas et al. 2024), Del Moral, Doucet & Jasra 2006,
Jasra et al. 2011, Lee & Whiteley 2018, Du & Guyader 2021, Beskos et al. 2016,
Chopin & Papaspiliopoulos 2020 (see ``references/provenance.toml``).
"""

import dataclasses
import functools
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

TOL = 0.1  # stop at 1 - TOL of the squared distance of independent draws
ESS_PER_DIM = 10  # a label's own mass needs this many effective particles / d


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
    mixing_short : bool
        Some stage reached ``max_steps`` before every label's cloud had moved
        ``1 - TOL`` of the distance between independent draws.
    steps : list of array
        Per replicate, the mutation steps used at each stage.
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
    mixing_short: bool = False
    steps: list = dataclasses.field(default_factory=list)


def _block(particles):
    """The continuous block HMC moves: ``particles["x"]``, or the array."""
    return particles["x"] if isinstance(particles, dict) else particles


class _Position(NamedTuple):
    position: object


def _algorithm(
    logprior, loglike, labelled, dtype, d, n_particles, target_ess, tune,
    moves, composite, lo, hi, args, mass, strategy,
):  # fmt: skip
    """The BlackJAX algorithm of one SMC stage.

    ``logprior(p, *args)``, ``loglike(p, *args)`` and each ``move(key, p,
    logdensity, *args)`` get ``args``. ``mass`` is the stage's inverse mass
    matrices (covariances): ``(K, d, d)``, one per label ``particle["c"]``, if
    ``labelled``, else ``(1, d, d)`` shared; ``strategy`` is the mutation
    (``update_strategy`` of BlackJAX's pretuning; built by ``_stage``).
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

            if labelled:  # this label's mass: it never depends on x
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
                p, _ = move(k, p, logdensity_fn, *args)
            return _Position(p), info

        def measure(state):
            chol = np.linalg.cholesky(
                state.parameter_override["inverse_mass_matrix"][0]
            )
            if labelled:  # each particle's jump in its own metric

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
        "inverse_mass_matrix": mass[None] if labelled else mass,
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
        lambda p: logprior(p, *args),
        lambda p: loglike(p, *args),
        mcmc_step,
        mcmc_init,
        # Multinomial, as the variance estimator of Lee & Whiteley requires.
        resampling.multinomial,
        lo,
        initial,
        pretune,
        target_ess=target_ess,
        update_strategy=strategy,
    )

    return algorithm


_STATIC = (
    "logprior", "loglike", "labelled", "dtype", "d", "n_particles",
    "target_ess", "tune", "moves", "composite", "lo", "hi",
)  # fmt: skip


@functools.partial(jax.jit, static_argnames=_STATIC)
def _stage(key, state, args, masses, pool, **static):
    """One stage: set the mass from the cloud, tune, mutate, reweight. The
    data are arguments, not closed over, so the compiled stage is reused by
    every run with the same functions and shapes. ``masses`` and ``pool`` are
    those of ``tempered_smc`` (``None`` unless the particles are labelled).

    Mass. A label's pool is the particles of its group, each mapped back by
    its label's ``maps⁻¹`` into the group's frame; the label gets
    ``map Σ̂ mapᵀ`` of the pool's weighted covariance ``Σ̂``, unless the pool
    has fewer than ``ESS_PER_DIM · d`` effective particles (then the fixed
    ``masses`` stay and the label does not count towards the stopping rule).

    Steps. All particles step together until, for every adapted group, the
    mean ``‖L⁻¹(xₖ − x₀)‖²`` of the particles that started in it (``L`` the
    Cholesky factor of their label's mass; the norm is the same for every
    image of a mode, so the images pool, which keeps the mean steady) reaches ``(1 - TOL) · 2d``, the squared
    distance of two independent draws; at least ``lo``, at most ``hi``."""
    cloud = state.sampler_state
    x, w = _block(cloud.particles), cloud.weights
    k, d, lo, hi = 1, static["d"], static["lo"], static["hi"]
    eps = np.finfo(x.dtype).eps
    if static["labelled"]:
        c, k = cloud.particles["c"], len(masses)
        group, maps, inv = pool
        g = group[c]
        seg = functools.partial(
            jax.ops.segment_sum, segment_ids=g, num_segments=k
        )
        total = np.maximum(seg(w), np.finfo(x.dtype).tiny)
        xp = np.einsum("nij,nj->ni", inv[c], x)
        r = xp - (seg(w[:, None] * xp) / total[:, None])[g]
        cov = seg(w[:, None, None] * r[:, :, None] * r[:, None, :])
        cov = cov / total[:, None, None]
        ess = total**2 / np.maximum(seg(w**2), np.finfo(x.dtype).tiny)
        own = np.einsum("kij,kjl,kml->kim", maps, cov[group], maps)
        # The jitter keeps the Cholesky in the ESJD finite on a degenerate pool.
        jitter = eps * (np.trace(own, axis1=1, axis2=2) + 1.0)
        own = own + jitter[:, None, None] * np.eye(d, dtype=x.dtype)
        active = ess[group] >= ESS_PER_DIM * d
        mass = np.where(active[:, None, None], own.astype(x.dtype), masses)
        shared = mass[None]
    else:
        centred = x - w @ x
        cov = (centred * w[:, None]).T @ centred
        # The jitter keeps the Cholesky in the ESJD finite on a degenerate cloud.
        cov = cov + eps * (np.trace(cov) + 1.0) * np.eye(d, dtype=cov.dtype)
        shared = mass = cov[None].astype(x.dtype)
        active = np.ones(1, bool)
        group = np.zeros(1, int)
    chol = np.linalg.cholesky(mass)
    target = (1.0 - TOL) * 2.0 * d

    def strategy(mcmc_init, logdensity, shared_step, n_particles,
                 num_mcmc_steps):  # fmt: skip
        def update(keys, particles, unshared):
            states = jax.vmap(lambda p: mcmc_init(p, logdensity))(particles)
            x0 = _block(particles)
            c0 = (
                particles["c"]
                if static["labelled"]
                else np.zeros(len(x0), int)
            )
            g0 = group[
                c0
            ]  # the metric is invariant under the maps: pool the images
            seen = np.bincount(g0, length=k)
            count = np.maximum(seen, 1)
            live = jax.ops.segment_max(active.astype(int), group, k) > 0

            def moved(xn):
                r = jax.vmap(
                    lambda m, v: jax.scipy.linalg.solve_triangular(
                        m, v, lower=True
                    )
                )(chol[c0], xn - x0)
                return jax.ops.segment_sum(np.sum(r**2, -1), g0, k) / count

            def body(carry):
                n, states, _ = carry
                step_keys = jax.vmap(jax.random.fold_in, (0, None))(keys, n)
                states, _ = jax.vmap(
                    lambda key, s, u: shared_step(key, s, logdensity, **u)
                )(step_keys, states, unshared)
                reached = np.all(
                    ~live
                    | (seen == 0)
                    | (moved(_block(states.position)) >= target)
                )
                return n + 1, states, (n + 1 >= lo) & reached

            n, states, done = jax.lax.while_loop(
                lambda carry: (carry[0] < hi) & ~carry[2],
                body,
                (np.zeros((), int), states, np.zeros((), bool)),
            )
            return states.position, {"steps": n, "capped": ~done}

        return update, n_particles

    algorithm = _algorithm(*[static[s] for s in _STATIC], args, mass, strategy)
    parameters = {**state.parameter_override, "inverse_mass_matrix": shared}
    return algorithm.step(key, type(state)(cloud, parameters))


def _run(key, init, stage, particles, max_stages, deadline):
    """One replicate: ``(particles, weights, log Z, betas, ess, stopped, v,
    steps, capped)``.

    ``v`` estimates ``var(Ẑ) / Z²`` (``nan`` if the genealogy collapsed);
    ``steps`` are the mutation steps of each stage and ``capped`` is whether
    any stage hit the cap.
    """
    state = init(particles)
    n = _block(particles).shape[0]
    tol = 8 * float(np.finfo(_block(particles).dtype).eps)
    log_z, betas, ess, steps, capped = 0.0, [], [], [], False
    # Eve indices (Lee & Whiteley 2018, Algorithm 1): the time-0 ancestor of
    # each particle, carried through BlackJAX's resampling ancestors.
    eve = onp.arange(n)
    *keys, final_key = jax.random.split(key, max_stages + 1)
    for key in keys:
        if time.monotonic() > deadline:
            return (
                None,
                None,
                onp.nan,
                betas,
                ess,
                True,
                onp.nan,
                steps,
                capped,
            )
        state, info = stage(key, state)
        steps.append(int(info.update_info["steps"]))
        capped = capped or bool(info.update_info["capped"])
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
                False, v, steps, capped,
            )  # fmt: skip
    return None, None, onp.nan, betas, ess, True, onp.nan, steps, capped


def _smc(
    logprior, loglike, starts, key, n_particles, replicates, ess_target,
    lo, hi, step_size, leapfrogs, alpha, max_stages, max_seconds, moves,
    masses=None, pool=None, args=(),
):  # fmt: skip
    deadline = time.monotonic() + max_seconds
    # One compiled stage, reused by every replicate (and every run with the
    # same functions and shapes: see ``_stage``).
    dtype, d = _block(starts[0]).dtype, _block(starts[0]).shape[1]
    labelled = masses is not None
    composite = bool(moves) or labelled or isinstance(starts[0], dict)
    if labelled:
        if pool is None:  # every label its own group, no map
            pool = (
                np.arange(len(masses)),
                np.tile(np.eye(d), (len(masses), 1, 1)),
            )
        group, maps = np.asarray(pool[0]), np.asarray(pool[1], dtype)
        pool = (group, maps, np.linalg.inv(maps))
    static = dict(
        logprior=logprior, loglike=loglike, labelled=labelled, dtype=dtype,
        d=d, n_particles=n_particles, target_ess=0.5,
        tune=(step_size, leapfrogs, alpha), moves=tuple(moves),
        composite=composite, lo=lo, hi=hi,
    )  # fmt: skip
    fixed = masses if labelled else np.eye(d, dtype=dtype)[None]
    init = _algorithm(*[static[k] for k in _STATIC], args, fixed, None).init
    stage = functools.partial(
        _stage, args=args, masses=masses, pool=pool, **static
    )
    runs = [
        _run(k, init, stage, start, max_stages, deadline)
        for k, start in zip(jax.random.split(key, replicates), starts)
    ]
    stopped = any(run[5] for run in runs)
    betas = [onp.asarray(run[3]) for run in runs]
    ess = [onp.asarray(run[4]) for run in runs]
    steps = [onp.asarray(run[7]) for run in runs]
    capped = any(run[8] for run in runs)
    if stopped:
        return SMCResult(
            None, None, onp.nan, onp.nan, onp.nan, onp.full(len(runs), onp.nan),
            betas, ess, True, False, False, mixing_short=capped, steps=steps,
        )  # fmt: skip
    log_zs = onp.array([run[2] for run in runs])
    vs = onp.array([run[6] for run in runs])
    # Σ_t (1/ESS_t − 1/N): the variance of Ẑ_r / Z of a perfectly mixing
    # tempered SMC (Chopin 2004; Bérard, Del Moral & Doucet 2014).
    fs = onp.array([onp.sum(1.0 / e - 1.0 / n_particles) for e in ess])
    # ln of the mean of the unbiased Ẑ_r: the delta method with weights
    # u_r = Ẑ_r / Σ Ẑ and per-replicate relative variances ``vs`` (Lee &
    # Whiteley 2018, Theorem 1: var(Ẑ_r) ≈ Ẑ_r² v_r) and ``fs``.
    shift = log_zs.max()
    u = onp.exp(log_zs - shift)
    u = u / u.sum()
    log_z = float(shift + onp.log(onp.mean(onp.exp(log_zs - shift))))
    variance, floor = float(onp.sum(u**2 * vs)), float(onp.sum(u**2 * fs))
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
        capped,
        steps,
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
    min_steps=5,
    max_steps=50,
    leapfrogs=4,
    step_size=None,
    alpha=1.0,
    max_stages=200,
    max_seconds=float("inf"),
    moves=(),
    masses=None,
    pool=None,
    args=(),
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
    min_steps, max_steps : int
        Bounds on the HMC mutation steps of a stage. The mutation stops at the
        first step at or after ``min_steps`` where, for every label whose mass was
        adapted, the mean squared displacement from the stage start, whitened
        by that mass, reaches ``(1 - TOL) · 2d`` (``TOL = 0.1``); reaching
        ``max_steps`` first sets ``SMCResult.mixing_short``.
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
        (Neal 2011, §5.2). Each stage replaces a label's mass by the weighted
        covariance of the cloud's particles with that label (of its ``pool``
        group), if they have at least ``10 d`` effective particles; otherwise
        the fixed mass stays. The step size and trajectory are tuned on the
        jump in each particle's own metric. ``None``: one mass, the cloud's
        covariance. ``moves`` must then update the label (Gibbs).
    pool : tuple of array, optional
        ``(group, maps)`` for labelled particles: ``group`` ``(K,)`` ints, the
        labels sharing a group are images ``x = maps[c] x'`` of one
        distribution, pooled after mapping back by ``maps[c]⁻¹`` (``maps``
        ``(K, d, d)``). Default: every label its own group.
    args : tuple of pytrees
        Extra arguments of ``logprior``, ``loglike`` and each move
        (``logprior(p, *args)``, ``move(key, p, logdensity, *args)``). They are
        traced, so module-level functions and equal shapes reuse the compiled
        stage across calls.

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
        min_steps, max_steps, step, leapfrogs, alpha, max_stages, max_seconds,
        moves, masses, pool, tuple(args),
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
    min_steps=5,
    max_steps=50,
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
        min_steps, max_steps, step, leapfrogs, alpha, max_stages, max_seconds,
        moves,
    )  # fmt: skip
