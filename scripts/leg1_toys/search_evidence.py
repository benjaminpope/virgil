"""Leg-1 search, within-cell HMC and the evidence estimators against the exact reference
(design/orbit_alias_cycles.md §6.1-6.5).

Search (§6.1): a deterministic grid in nu = 1/P at phase coherence k, with TI' solved linearly at
every grid point and the night labels assigned robustly (every pair of nights as a minimal set,
then C-steps, Rousseeuw 1984; Fischler & Bolles 1981); every local maximum of the grid profile is
refined by Newton on the label-summed target. Coverage: the oracle's peaks (the exact collapsed
profile split at its minima) whose basin holds a search mode, as a share of the cell's mass.

Evidence (§6.4), each from the search modes, no tempering:
  L : Laplace at the modes (baseline).
  I : defensive importance sampling (Hesterberg 1995) from q = ALPHA prior + (1 - ALPHA) sum_m
      w_m t_DF(mode m, Laplace covariance), w_m from Laplace.
  B : bridge sampling (Meng & Wong 1996; Gronau et al. 2017) per mode, on the balance-heuristic
      partition of unity r_m = (1 - ALPHA) w_m t_m / q (Veach & Guibas 1995), from HMC draws of
      p L r_m (whitened by the mode's Laplace covariance) and draws of t_m; plus the prior part by IS.

usage: search_evidence.py PATTERN REL [REPS] [K_COH]  [METHODS, default IB]
"""

import itertools
import math
import sys
import time

import jax
import jax.numpy as np
import numpy as onp
from scipy import optimize
from scipy.signal import find_peaks
from scipy.special import gammaln

from toy import LN_IN, LNB, SW, Problem, features

ALPHA, DF = (
    0.1,
    5.0,
)  # defensive weight (budget); Student-t degrees of freedom (as aliases._mode_evidence)
DELTA = 1e-3  # reporting tolerance (design §10)
GRAD_COST = (
    3.0  # a gradient costs about 3 likelihood evaluations (reverse mode)
)
N_IS = 4000
N_CHAINS, N_WARM, N_DRAW, N_PROP, N_PRIOR = 4, 150, 250, 1000, 1000


# ------------------------------------------------------------------------------------- search
def solve_theta(c0, c1, d, mask, sig):
    """Weighted least squares for TI' (with the N(0, SW^2) prior as a ridge) on the nights in
    ``mask``; c0, c1, mask (..., K). Returns theta (..., 4)."""
    s2 = sig**2
    A = (mask * c0 * c0).sum(-1) / s2 + 1 / SW**2
    B = (mask * c0 * c1).sum(-1) / s2
    C = (mask * c1 * c1).sum(-1) / s2 + 1 / SW**2
    det = A * C - B * B
    th = []
    for j in (0, 1):
        r0 = (mask * c0 * d[:, j]).sum(-1) / s2
        r1 = (mask * c1 * d[:, j]).sum(-1) / s2
        th.append(((C * r0 - B * r1) / det, (A * r1 - B * r0) / det))
    return onp.stack([th[0][0], th[1][0], th[0][1], th[1][1]], -1)


def night_ell(c0, c1, th, d, sig):
    dec = th[..., 0:1] * c0 + th[..., 2:3] * c1
    ra = th[..., 1:2] * c0 + th[..., 3:4] * c1
    r2 = (d[:, 0] - dec) ** 2 + (d[:, 1] - ra) ** 2
    return LN_IN - onp.log(2 * onp.pi * sig**2) - r2 / (2 * sig**2)


def triplet_candidates(prob, k_coh):
    """Minimal sets (Fischler & Bolles 1981; every triplet, as elemental sets, Rousseeuw 1984):
    for each triplet of nights, TI' by weighted least squares given rho (variable projection,
    Golub & Pereyra 1973) and the triplet's residual R(u) minimized in ln P over the cell from
    every local minimum of a grid at coherence k_coh over the triplet's own spread. Returns
    candidate (u, theta) and the cost in likelihood evaluations."""
    K, t, s2 = prob.K, prob.t, prob.sig**2
    nu_lo, nu_hi = math.exp(-prob.LHI), math.exp(-prob.LLO)
    cands, cost = [], 0.0
    for tri in itertools.combinations(range(K), 3):
        tri = onp.array(tri)
        spread = t[tri].max() - t[tri].min()
        n = max(int(math.ceil((nu_hi - nu_lo) * k_coh * spread)), 2)
        nu = nu_lo + (nu_hi - nu_lo) * (onp.arange(n) + 0.5) / n
        d = prob.d[tri]

        def resid(u):
            c0, c1 = features(onp.exp(u), t[tri])
            th = solve_theta(c0, c1, d, onp.ones(3), prob.sig)
            dec = th[:, 0:1] * c0 + th[:, 2:3] * c1
            ra = th[:, 1:2] * c0 + th[:, 3:4] * c1
            r = ((d[:, 0] - dec) ** 2 + (d[:, 1] - ra) ** 2).sum(1) / s2 + (
                th**2
            ).sum(1) / SW**2
            return r, th

        u = -onp.log(nu)
        r, _ = resid(u)
        cost += n * 3 / K
        loc = [
            i
            for i in range(n)
            if r[i] <= r[max(i - 1, 0)] and r[i] <= r[min(i + 1, n - 1)]
        ]
        du = (u[0] - u[-1]) / max(n - 1, 1)
        for i in loc:
            lo, hi = max(u[i] - du, prob.LLO), min(u[i] + du, prob.LHI)
            res = optimize.minimize_scalar(
                lambda x: resid(onp.array([x]))[0][0],
                bounds=(lo, hi),
                method="bounded",
                options=dict(xatol=1e-12),
            )
            cost += res.nfev * 3 / K
            cands.append((res.x, resid(onp.array([res.x]))[1][0]))
    return cands, cost


def c_steps(prob, w, n_max=4):
    """Concentration steps (Rousseeuw & Van Driessen 1999): labels from the night terms at w,
    then Newton on the labelled target, until the labels repeat. Returns w, labels, cost."""
    lab, cost = None, 0.0
    for _ in range(n_max):
        u = prob.u_of_w(w[0:1])
        c0, c1 = features(onp.exp(u), prob.t)
        new = (
            night_ell(c0, c1, w[None, 1:], prob.d, prob.sig)[0] > LNB
        ).astype(float)
        cost += 1
        if lab is not None and onp.array_equal(new, lab):
            break
        lab = new
        f = lambda x: tuple(
            -onp.asarray(v)
            for v in prob._vg_lab(np.asarray(x), np.asarray(lab))
        )  # noqa: E731
        h = lambda x: -onp.asarray(
            prob._hess_lab(np.asarray(x), np.asarray(lab))
        )  # noqa: E731
        res = optimize.minimize(
            f,
            w,
            jac=True,
            hess=h,
            method="trust-exact",
            options=dict(gtol=1e-8),
        )
        w = res.x
        cost += res.nfev + res.njev * GRAD_COST + res.nhev * GRAD_COST * len(w)
    return w, lab, cost


def refine(prob, w0):
    """Newton (trust region, exact Hessian) on the label-summed log target from w0."""
    f = lambda w: tuple(-onp.asarray(v) for v in prob._vg(np.asarray(w)))  # noqa: E731
    h = lambda w: -onp.asarray(prob._hess(np.asarray(w)))  # noqa: E731
    res = optimize.minimize(
        f, w0, jac=True, hess=h, method="trust-exact", options=dict(gtol=1e-8)
    )
    return (
        res.x,
        -res.fun,
        res.nfev + res.njev * GRAD_COST + res.nhev * GRAD_COST * len(w0),
    )


def search(prob, k_coh=4.0):
    """Modes [(w, log target, covariance)] of the cell, the search cost and the candidate count."""
    cands, cost = triplet_candidates(prob, k_coh)
    seen, modes = set(), []
    for u, th in cands:
        w0 = onp.concatenate([[prob.w_of_u(u)], th])
        if not onp.all(onp.isfinite(w0)):
            continue
        w, lab, c = c_steps(prob, w0)
        cost += c
        key = (
            tuple(lab.astype(int)),
            round(float(prob.u_of_w(w[0])) / (5 * prob.needle_width())),
        )
        if key in seen:
            continue  # this label set's mode, already refined
        seen.add(key)
        w, val, c = refine(prob, w)
        cost += c
        H = -onp.asarray(prob._hess(np.asarray(w)))
        cost += GRAD_COST * len(w)
        ev = onp.linalg.eigvalsh(0.5 * (H + H.T))
        if not (onp.all(onp.isfinite(ev)) and ev.min() > 0):
            continue
        cov = onp.linalg.inv(H)
        if any(
            max(_mahal(w, m[0], onp.linalg.inv(m[2])), _mahal(w, m[0], H))
            < 1.0
            for m in modes
        ):
            continue  # the same mode again (close under both covariances: a broad mode never hides a needle)
        modes.append((w, val, 0.5 * (cov + cov.T)))
    modes.sort(key=lambda m: -m[1])
    return modes, cost, len(cands)


def _mahal(x, mu, prec):
    r = x - mu
    return float(r @ prec @ r)


def coverage(prob, modes, profile):
    """Share of the exact cell mass in oracle basins that hold a search mode, and the oracle
    peaks with mass >= DELTA (found / all). Oracle peaks: local maxima of the exact collapsed
    profile with prominence >= 1 nat (shallower wiggles are one basin); basins are cut at the
    lowest point between neighbouring peaks."""
    u, f = profile
    peaks, _ = find_peaks(
        onp.concatenate([[-onp.inf], f, [-onp.inf]]), prominence=1.0
    )
    peaks = peaks - 1
    cuts = (
        [0]
        + [a + int(onp.argmin(f[a:b])) for a, b in zip(peaks[:-1], peaks[1:])]
        + [len(u) - 1]
    )
    g = onp.exp(f - f.max())
    mass = onp.array(
        [
            onp.trapezoid(g[a : b + 1], u[a : b + 1])
            for a, b in zip(cuts[:-1], cuts[1:])
        ]
    )
    mass /= mass.sum()
    um = [prob.u_of_w(m[0][0]) for m in modes]
    found = onp.array(
        [
            any(u[a] <= x <= u[b] for x in um)
            for a, b in zip(cuts[:-1], cuts[1:])
        ]
    )
    big = mass >= DELTA
    return float(mass[found].sum()), int(big.sum()), int((found & big).sum())


# --------------------------------------------------------------------------------- proposal q
class Mixture:
    """q = ALPHA prior + (1 - ALPHA) sum_m w_m t_DF(mu_m, cov_m)."""

    def __init__(self, prob, modes):
        self.prob = prob
        # no modes: q is the prior alone, so the IS estimate stays valid (and the coverage tests,
        # not the sentinel, catch the missed mode)
        self.alpha = ALPHA if modes else 1.0
        if not modes:
            self.w = onp.zeros(0)
            return
        lz = onp.array([laplace_one(m) for m in modes])
        self.w = onp.exp(lz - lz.max())
        self.w /= self.w.sum()
        self.mu = onp.array([m[0] for m in modes])
        self.chol = onp.array([onp.linalg.cholesky(m[2]) for m in modes])
        self.linv = onp.array([onp.linalg.inv(c) for c in self.chol])
        self.logdet = onp.array(
            [2 * onp.log(onp.diag(c)).sum() for c in self.chol]
        )
        self.d = self.mu.shape[1]

    def log_t(self, x, m):
        z = (x - self.mu[m]) @ self.linv[m].T
        q = (z**2).sum(-1)
        d = self.d
        return (
            -0.5 * (DF + d) * onp.log1p(q / DF)
            - 0.5 * self.logdet[m]
            + gammaln(0.5 * (DF + d))
            - gammaln(0.5 * DF)
            - 0.5 * d * onp.log(DF * onp.pi)
        )

    def draw_t(self, m, n, rng):
        z = rng.standard_normal((n, self.d))
        g = rng.chisquare(DF, n) / DF
        return self.mu[m] + (z @ self.chol[m].T) / onp.sqrt(g)[:, None]

    def log_q(self, x):
        parts = [onp.log(self.alpha) + self.prob.logprior(x)]
        parts += [
            onp.log(1 - self.alpha) + onp.log(self.w[m]) + self.log_t(x, m)
            for m in range(len(self.w))
        ]
        return onp.logaddexp.reduce(onp.stack(parts), 0)

    def draw(self, n, rng):
        comp = rng.choice(
            len(self.w) + 1,
            size=n,
            p=onp.concatenate([[self.alpha], (1 - self.alpha) * self.w]),
        )
        x = self.prob.prior_draw(n, rng)
        for m in range(len(self.w)):
            sel = comp == m + 1
            x[sel] = self.draw_t(m, sel.sum(), rng)
        return x


def laplace_one(mode):
    w, val, cov = mode
    return (
        val
        + 0.5 * len(w) * onp.log(2 * onp.pi)
        + 0.5 * onp.linalg.slogdet(cov)[1]
    )


def laplace(modes):
    return float(onp.logaddexp.reduce([laplace_one(m) for m in modes]))


def logmeanexp(a):
    a = onp.asarray(a)
    m = a.max()
    return float(m + onp.log(onp.mean(onp.exp(a - m))))


def importance(prob, q, n, rng):
    x = q.draw(n, rng)
    lw = prob.logpost(x) - q.log_q(x)
    return logmeanexp(lw), float(n), lw


# --------------------------------------------------------------------------------------- HMC
def make_hmc(prob, q, m):
    """HMC on log p L r_m in coordinates whitened by mode m's Laplace covariance; fixed L per
    transition = ceil(pi / (2 eps)), eps by dual averaging (Hoffman & Gelman 2014) to 0.8."""
    mu, chol = np.asarray(q.mu[m]), np.asarray(q.chol[m])
    jmu, jlinv, jlogdet, jw = (
        np.asarray(a) for a in (q.mu, q.linv, q.logdet, q.w)
    )
    d = q.d

    def log_t_all(x):
        z = np.einsum("mij,mj->mi", jlinv, x[None] - jmu)
        qq = (z**2).sum(-1)
        return (
            -0.5 * (DF + d) * np.log1p(qq / DF)
            - 0.5 * jlogdet
            + math.lgamma(0.5 * (DF + d))
            - math.lgamma(0.5 * DF)
            - 0.5 * d * math.log(DF * math.pi)
        )

    def logprior(x):
        return (
            -jax.nn.softplus(-x[0])
            - jax.nn.softplus(x[0])
            - 0.5 * np.sum(x[1:] ** 2) / SW**2
            - 2 * np.log(2 * np.pi * SW**2)
        )

    def target(y):
        x = mu + chol @ y
        lt = np.log(1 - ALPHA) + np.log(jw) + log_t_all(x)
        lq = np.logaddexp(
            np.log(ALPHA) + logprior(x), jax.scipy.special.logsumexp(lt)
        )
        return prob._logpost_one(x) + lt[m] - lq

    vg = jax.value_and_grad(target)

    def transition(state, key, eps, n_leap):
        y, lp, g = state
        k1, k2, k3 = jax.random.split(key, 3)
        e = eps * jax.random.uniform(k3, minval=0.8, maxval=1.2)
        p = jax.random.normal(k1, y.shape)
        h0 = lp - 0.5 * p @ p

        def body(_, c):
            y_, p_, g_, _l = c
            p_ = p_ + 0.5 * e * g_
            y_ = y_ + e * p_
            l_, g_ = vg(y_)
            return y_, p_ + 0.5 * e * g_, g_, l_

        y1, p1, g1, l1 = jax.lax.fori_loop(0, n_leap, body, (y, p, g, lp))
        h1 = l1 - 0.5 * p1 @ p1
        acc = np.minimum(
            1.0, np.exp(np.where(np.isfinite(h1), h1 - h0, -np.inf))
        )
        take = jax.random.uniform(k2) < acc
        new = (
            np.where(take, y1, y),
            np.where(take, l1, lp),
            np.where(take, g1, g),
        )
        return new, acc

    @jax.jit
    def run(key, eps0):
        keys = jax.random.split(key, N_CHAINS)

        def chain(key):
            y0 = np.zeros(d)
            l0, g0 = vg(y0)

            def warm(c, i):
                state, log_eps, hbar, log_eps_bar, k = c
                eps = np.exp(log_eps)
                n_leap = np.clip(np.ceil(np.pi / 2 / eps), 1, 256).astype(int)
                state, acc = transition(
                    state, jax.random.fold_in(k, i), eps, n_leap
                )
                t = i + 1.0
                hbar = (1 - 1 / (t + 10)) * hbar + (0.8 - acc) / (t + 10)
                log_eps = np.log(10 * eps0) - np.sqrt(t) / 0.05 * hbar
                eta = t**-0.75
                log_eps_bar = eta * log_eps + (1 - eta) * log_eps_bar
                return (state, log_eps, hbar, log_eps_bar, k), n_leap

            init = ((y0, l0, g0), np.log(eps0), 0.0, 0.0, key)
            (state, _, _, log_eps_bar, k), nw = jax.lax.scan(
                warm, init, np.arange(N_WARM)
            )
            eps = np.exp(log_eps_bar)
            n_leap = np.clip(np.ceil(np.pi / 2 / eps), 1, 256).astype(int)

            def draw(state, i):
                state, acc = transition(
                    state, jax.random.fold_in(k, 10_000 + i), eps, n_leap
                )
                return state, (state[0], acc)

            _, (ys, accs) = jax.lax.scan(draw, state, np.arange(N_DRAW))
            return ys, accs, nw.sum() + N_DRAW * n_leap

        ys, accs, n_grad = jax.vmap(chain)(keys)
        return mu + ys.reshape(-1, d) @ chol.T, accs.mean(), n_grad.sum()

    return run, target


def bridge(prob, q, rng, key):
    """Sum over modes of the per-mode bridge estimates, plus the prior part by IS."""
    logz, cost = [], 0.0
    for m in range(len(q.w)):
        run, _ = make_hmc(prob, q, m)
        xs, acc, n_grad = run(jax.random.fold_in(key, m), 0.5 / math.sqrt(q.d))
        xs = onp.asarray(xs)
        cost += float(n_grad) * GRAD_COST
        xp = q.draw_t(m, N_PROP, rng)
        lq_all = lambda x: q.log_q(x)  # noqa: E731
        lr = (
            lambda x: onp.log(1 - ALPHA)
            + onp.log(q.w[m])
            + q.log_t(x, m)
            - lq_all(x)
        )  # noqa: E731
        l1 = prob.logpost(xs) + lr(xs) - q.log_t(xs, m)
        l2 = prob.logpost(xp) + lr(xp) - q.log_t(xp, m)
        cost += N_PROP + len(xs)
        n1, n2 = len(l1), len(l2)
        s1, s2 = n1 / (n1 + n2), n2 / (n1 + n2)
        r = logmeanexp(l2)  # start from the IS estimate
        for _ in range(200):  # Meng & Wong (1996) iteration, in logs
            num = logmeanexp(
                l2 - onp.logaddexp(onp.log(s1) + l2, onp.log(s2) + r)
            )
            den = logmeanexp(-onp.logaddexp(onp.log(s1) + l1, onp.log(s2) + r))
            new = num - den
            if abs(new - r) < 1e-10:
                break
            r = new
        logz.append(r)
    x0 = prob.prior_draw(N_PRIOR, rng)
    logz.append(onp.log(ALPHA) + logmeanexp(prob.logpost(x0) - q.log_q(x0)))
    cost += N_PRIOR
    return float(onp.logaddexp.reduce(logz)), cost


def sentinel(prob, x, lw, modes):
    """Fails (True) when the weighted draws sit at the all-outlier floor (fewer than 2 inlier
    nights expected) while the search found a mode with >= 2 inliers W nats above the floor."""
    W = math.log(1 / DELTA)
    u = prob.u_of_w(x[:, 0])
    c0, c1 = features(onp.exp(u), prob.t)
    ell = night_ell(c0, c1, x[:, 1:], prob.d, prob.sig)
    resp = 1 / (1 + onp.exp(LNB - ell))
    w = onp.exp(lw - lw.max())
    n_in = float((w * resp.sum(1)).sum() / w.sum())
    if (
        not modes
    ):  # nothing to compare against: a missed mode is the coverage tests' job
        return False, n_in
    best = modes[0][0]
    uc0, uc1 = features(onp.exp(prob.u_of_w(best[0:1])), prob.t)
    lb = night_ell(uc0, uc1, best[None, 1:], prob.d, prob.sig)[0]
    above = onp.logaddexp(lb, LNB).sum() - prob.K * LNB
    return bool(n_in < 2 and (lb > LNB).sum() >= 2 and above > W), n_in


if __name__ == "__main__":
    pattern, rel = sys.argv[1], float(sys.argv[2])
    reps = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    k_coh = float(sys.argv[4]) if len(sys.argv) > 4 else 4.0
    prob = Problem(pattern, rel)
    t0 = time.time()
    ex, profile = prob.exact()
    lz_exact = ex[prob.n_lo]
    t_ex = time.time() - t0
    t0 = time.time()
    modes, s_cost, n_grid = search(prob, k_coh)
    t_s = time.time() - t0
    cov_mass, n_or, n_found = coverage(prob, modes, profile)
    print(
        f"{pattern} sigma/a={rel:g} N={prob.n_lo} exact lnZ={lz_exact:.3f} floor={prob.K * LNB:.2f} "
        f"[{t_ex:.0f}s] | search k={k_coh:g}: grid={n_grid} modes={len(modes)} cost={s_cost:.2e} "
        f"wall={t_s:.1f}s coverage={cov_mass:.6f} oracle peaks>=delta {n_found}/{n_or}",
        flush=True,
    )
    q = Mixture(prob, modes)
    lz_lap = laplace(modes)
    print(f"  L: lnZ={lz_lap:9.3f} bias={lz_lap - lz_exact:8.3f}", flush=True)
    for meth in sys.argv[5] if len(sys.argv) > 5 else "IB":
        z, co, wt, sen = [], [], [], []
        for r in range(reps):
            rng = onp.random.default_rng(1000 + r)
            t0 = time.time()
            if meth == "I":
                lz, c, lw = importance(prob, q, N_IS, rng)
                x = None
            else:
                lz, c = bridge(prob, q, rng, jax.random.PRNGKey(r))
            wt.append(time.time() - t0)
            z.append(lz), co.append(c)
            if meth == "I" and r == 0:
                rng2 = onp.random.default_rng(1000 + r)
                x = q.draw(N_IS, rng2)
                lw = prob.logpost(x) - q.log_q(x)
                fail, n_in = sentinel(prob, x, lw, modes)
                sen.append(
                    f"sentinel={'FAIL' if fail else 'ok'} E[n_in]={n_in:.2f}"
                )
        z = onp.array(z)
        var = z.var(ddof=1)
        print(
            f"  {meth}: lnZ={z.mean():9.3f} bias={z.mean() - lz_exact:8.3f} sd={z.std(ddof=1):7.3f} "
            f"cost={onp.mean(co):.2e} var*cost={var * onp.mean(co):.2e} wall={onp.mean(wt):.1f}s "
            f"{' '.join(sen)} all={onp.round(z, 3).tolist()}",
            flush=True,
        )
