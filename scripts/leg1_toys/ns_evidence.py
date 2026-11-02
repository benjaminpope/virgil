"""Candidate (d), nested sampling, on the leg-1 toy (design/orbit_alias_cycles.md §6.4).

BlackJAX nested slice sampling (Yallup, Kroupa & Handley 2026, arXiv:2601.23252; blackjax >= 1.7)
on the same target as search_evidence.py: live points drawn from the prior (no search), labels summed
in the likelihood. Reports ln Z against the exact reference, the information H = KL(post || prior),
the likelihood evaluations, and whether the posterior mass sits on the search's best mode.

Needs blackjax >= 1.7, which the worktree venv lacks until #623: run with a scratch venv.

usage: ns_evidence.py PATTERN REL [REPS] [N_LIVE] [MIN_NATS] [seed]
"""

import math
import sys
import time

import blackjax
import jax
import jax.numpy as np
import numpy as onp
from blackjax.ns.utils import finalise, log_weights

import search_evidence as S
from jax.scipy.special import gammaln
from toy import SW, Problem

FRAC_DELETE = 0.1  # particles replaced per step, as a fraction of the live set (BlackJAX examples)
INNER_PER_DIM = (
    5  # num_inner_steps = 5 dim, the NSS docstring's floor for unbiased mixing
)
STOP = -3.0  # stop when the live points hold < e^-3 of the evidence so far (Skilling 2006)
MIN_NATS = 0.0  # optional floor on the compression -ln X before stopping (plateaus: Fowlie, Handley & Su 2021)


def _logprior(w):
    lw0 = -jax.nn.softplus(-w[0]) - jax.nn.softplus(w[0])
    return (
        lw0 - 0.5 * np.sum(w[1:] ** 2) / SW**2 - 2 * np.log(2 * np.pi * SW**2)
    )


def _log_q(q):
    """JAX log density of the search's defensive mixture (search_evidence.Mixture)."""
    d, lw = q.d, np.log(q.w)
    const = (
        gammaln(0.5 * (S.DF + d))
        - gammaln(0.5 * S.DF)
        - 0.5 * d * math.log(S.DF * math.pi)
    )

    def log_q(w):
        z = np.einsum("mij,mj->mi", q.linv, w[None] - q.mu)
        lt = (
            -0.5 * (S.DF + d) * np.log1p((z**2).sum(-1) / S.DF)
            - 0.5 * q.logdet
            + const
        )
        return np.logaddexp(
            math.log(S.ALPHA) + _logprior(w),
            math.log(1 - S.ALPHA) + jax.scipy.special.logsumexp(lw + lt),
        )

    return log_q


def run(prob, n_live, key, q=None):
    """Nested sampling from the prior, or (q given) from the search's mixture by posterior
    repartitioning (Chen, Feroz & Hobson 2019): reference q, likelihood p L / q, same Z."""
    logprior = _logprior if q is None else _log_q(q)

    def loglike(w):
        return prob._logpost_one(w) - logprior(w)

    dim = 5
    algo = blackjax.nss(
        logprior,
        loglike,
        num_inner_steps=INNER_PER_DIM * dim,
        num_delete=max(1, int(FRAC_DELETE * n_live)),
    )
    k0, key = jax.random.split(key)
    if q is None:
        w0 = np.column_stack(
            [
                jax.random.logistic(k0, (n_live,)),
                SW * jax.random.normal(jax.random.fold_in(k0, 1), (n_live, 4)),
            ]
        )
    else:
        w0 = np.asarray(q.draw(n_live, onp.random.default_rng(int(k0[-1]))))
    state = algo.init(w0)
    step = jax.jit(algo.step)
    dead, evals = [], float(n_live)
    while (
        state.integrator.logZ_live - state.integrator.logZ > STOP
        or -state.integrator.logX < MIN_NATS
    ):
        key, k = jax.random.split(key)
        state, info = step(k, state)
        dead.append(info)
        ui = info.update_info
        evals += float(np.sum(ui.num_expansions + ui.num_shrink + 2))
    fin = finalise(state, dead)
    lw = log_weights(
        jax.random.PRNGKey(0), fin, shape=100
    )  # (n, 100) simulated shrinkages
    lz_draws = jax.scipy.special.logsumexp(lw, axis=0)
    lz = float(
        np.log(np.mean(np.exp(lz_draws - lz_draws.max()))) + lz_draws.max()
    )
    p = onp.exp(
        onp.asarray(lw.mean(1)) - jax.scipy.special.logsumexp(lw.mean(1))
    )
    ll = onp.asarray(fin.particles.loglikelihood)
    h = float(onp.sum(p * ll) - lz)
    w_post = onp.asarray(fin.particles.position)
    return lz, float(lz_draws.std()), h, evals, len(dead), w_post, p


if __name__ == "__main__":
    pattern, rel = sys.argv[1], float(sys.argv[2])
    reps = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    n_live = int(sys.argv[4]) if len(sys.argv) > 4 else 500
    MIN_NATS = float(sys.argv[5]) if len(sys.argv) > 5 else 0.0
    seeded = len(sys.argv) > 6 and sys.argv[6] == "seed"
    prob = Problem(pattern, rel)
    ex, _ = prob.exact()
    lz_exact = ex[prob.n_lo]
    x_true = prob.T / 12.1
    print(
        f"{pattern} sigma/a={rel:g} N={prob.n_lo} exact lnZ={lz_exact:.3f} floor={prob.K * math.log(0.1 / (math.pi * 900)):.2f} "
        f"needle(x)={prob.needle_width() * prob.T / 12.1:.2e} n_live={n_live}",
        flush=True,
    )
    q, s_cost = None, 0.0
    if seeded:
        modes, s_cost, _ = S.search(prob)
        q = S.Mixture(prob, modes)
        print(
            f"  seeded by the search: modes={len(modes)} search cost={s_cost:.2e}",
            flush=True,
        )
    z, cost = [], []
    for r in range(reps):
        t0 = time.time()
        lz, sd_in, h, ev, steps, w_post, p = run(
            prob, n_live, jax.random.PRNGKey(r), q
        )
        x = prob.T / onp.exp(prob.u_of_w(w_post[:, 0]))
        near = float(
            p[onp.abs(x - x_true) < 50 * prob.needle_width() * x_true].sum()
        )
        z.append(lz), cost.append(ev + s_cost)
        print(
            f"  rep {r}: lnZ={lz:9.3f} bias={lz - lz_exact:8.3f} sd_internal={sd_in:.3f} H={h:6.1f} "
            f"evals={ev:.2e} steps={steps} p(needle)={near:.3f} wall={time.time() - t0:.0f}s",
            flush=True,
        )
    z = onp.array(z)
    var = z.var(ddof=1) if reps > 1 else onp.nan
    print(
        f"  N: lnZ={z.mean():9.3f} bias={z.mean() - lz_exact:8.3f} sd={math.sqrt(var):7.3f} "
        f"cost={onp.mean(cost):.2e} var*cost={var * onp.mean(cost):.2e}",
        flush=True,
    )
