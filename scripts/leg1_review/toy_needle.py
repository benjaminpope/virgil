"""Leg-1 toy at realistic precision: circular orbit, P sampled inside one alias
cell, TI' Gaussian prior N(0, s_w^2 I) (so the reference is exact), clustered
epochs plus a long gap, per-night defensive mixture (eps fixed, uniform-disc
outlier). Methods (all adaptive tempered SMC, CESS = 1/2, same M, same moves):

  W : #629's path: (whole mixture)^beta; Luger move with labels drawn at beta=1
      and MH ratio L^(beta-1); RW-MH on P with theta fixed.
  C : D 6.3 / toy-6.5 path: per-component tempering sum_z [w_z L(theta|z)]^beta;
      tempered label Gibbs + exact tempered Gaussian theta draw; RW-MH on P.
  J : Joker-style: labels as particle state, theta collapsed for the P move;
      per-component tempering; theta|P,z then z|theta,P Gibbs.
  X : labels and theta both summed exactly (2^K label sets); RW-MH on P only.

Reference: grid over ln P of the exact collapsed integrand.
usage: toy_needle.py K rel_sigma reps
"""

import sys
import time
import numpy as onp

EPS, R_OUT, SW = 0.1, 30.0, 6.6
LNB = onp.log(EPS) - onp.log(onp.pi * R_OUT**2)
TRUTH = onp.array([4.0, -2.5, 1.5, 3.0])  # (A, B, F, G); a ~ 4.7
A_TRUE = 4.7
EPOCHS = {
    6: [0, 2.4, 5.1, 238, 240.6, 247.4],
    10: [0, 1.2, 2.4, 3.7, 5.1, 238, 239.3, 240.6, 244, 247.4],
}
P_TRUE = 12.1
PLO, PHI = 240 / 20.5, 240 / 19.5  # one alias cell (N = 20 cycles)
LLO, LHI = onp.log(PLO), onp.log(PHI)
M, MOVES = 1000, 4


def feats(P, t):
    ph = 2 * onp.pi * t[None] / P[:, None]
    return onp.cos(ph), onp.sin(ph)  # (n, K)


def positions(theta, c0, c1):
    dec = theta[:, None, 0] * c0 + theta[:, None, 2] * c1
    ra = theta[:, None, 1] * c0 + theta[:, None, 3] * c1
    return dec, ra


class Prob:
    def __init__(self, K, rel, seed):
        rng = onp.random.default_rng(seed)
        self.t = onp.array(EPOCHS[K], float)
        self.K, self.sig = K, rel * A_TRUE
        c0, c1 = feats(onp.array([P_TRUE]), self.t)
        dec, ra = positions(TRUTH[None], c0, c1)
        self.d = onp.stack([dec[0], ra[0]], 1) + self.sig * rng.normal(
            size=(K, 2)
        )
        if K == 10:
            self.d[7] = rng.uniform(-8, 8, 2)  # one outlier night
        self.masks = ((onp.arange(2**K)[:, None] >> onp.arange(K)) & 1).astype(
            float
        )
        self.cost = 0

    # ---- per-night, explicit theta
    def ell(self, theta, P):
        c0, c1 = feats(P, self.t)
        dec, ra = positions(theta, c0, c1)
        r2 = (self.d[None, :, 0] - dec) ** 2 + (self.d[None, :, 1] - ra) ** 2
        self.cost += len(P) * self.K
        return (
            onp.log(1 - EPS)
            - onp.log(2 * onp.pi * self.sig**2)
            - r2 / (2 * self.sig**2)
        )

    def ll_whole(self, theta, P):
        return onp.logaddexp(self.ell(theta, P), LNB).sum(1)

    def ll_comp(self, theta, P, beta):
        return onp.logaddexp(beta * self.ell(theta, P), beta * LNB).sum(1)

    # ---- collapsed over theta for label sets m (n, Mk, K) or (Mk, K)
    def coll(self, P, m, beta, draw=None, rng=None):
        if beta == 0.0:
            return onp.zeros(
                m.shape[:-1] if m.ndim == 3 else (len(P), m.shape[0])
            )
        c0, c1 = feats(P, self.t)
        s2b = self.sig**2 / beta
        m = onp.broadcast_to(m, (len(P),) + m.shape[-2:]) if m.ndim == 2 else m
        self.cost += m.shape[0] * m.shape[1] * self.K
        A = onp.einsum("nmk,nk->nm", m, c0 * c0) / s2b + 1 / SW**2
        B = onp.einsum("nmk,nk->nm", m, c0 * c1) / s2b
        C = onp.einsum("nmk,nk->nm", m, c1 * c1) / s2b + 1 / SW**2
        det = A * C - B * B
        quad, means = 0.0, []
        for j in (0, 1):
            r0 = onp.einsum("nmk,nk->nm", m, c0 * self.d[None, :, j]) / s2b
            r1 = onp.einsum("nmk,nk->nm", m, c1 * self.d[None, :, j]) / s2b
            quad = quad + (C * r0**2 - 2 * B * r0 * r1 + A * r1**2) / det
            means.append(((C * r0 - B * r1) / det, (A * r1 - B * r0) / det))
        Q = onp.einsum("nmk,k->nm", m, (self.d**2).sum(1)) / s2b
        nin = m.sum(-1)
        out = (
            -beta * nin * onp.log(2 * onp.pi * self.sig**2)
            - 2 * onp.log(SW**2)
            - onp.log(det)
            - 0.5 * (Q - quad)
            + beta * (nin * onp.log(1 - EPS) + (self.K - nin) * LNB)
        )
        if draw is None:
            return out
        # draw theta | P, m (single mask per particle: Mk = 1)
        cov = (
            onp.stack([onp.stack([C, -B], -1), onp.stack([-B, A], -1)], -2)
            / det[..., None, None]
        )
        L = onp.linalg.cholesky(cov[:, 0])
        th = onp.zeros((len(P), 4))
        for j, (mu0, mu1) in enumerate(means):
            z = L @ rng.normal(size=(len(P), 2, 1))
            th[:, j] = mu0[:, 0] + z[:, 0, 0]
            th[:, j + 2] = mu1[:, 0] + z[:, 1, 0]
        return out, th

    def coll_all(self, P, beta):
        if beta == 0.0:
            return onp.full(len(P), self.K * onp.log(2.0))
        out = []
        for i in range(0, len(P), 200):
            out.append(
                onp.logaddexp.reduce(
                    self.coll(P[i : i + 200], self.masks, beta), axis=1
                )
            )
        return onp.concatenate(out)

    def exact(self):
        tot = []
        for n in (200_000, 400_000):
            lp = onp.linspace(LLO, LHI, n)
            f = self.coll_all(onp.exp(lp), 1.0)
            i = onp.argmax(f)
            fine = onp.linspace(
                lp[max(i - 200, 0)], lp[min(i + 200, n - 1)], 100_000
            )
            lp2 = onp.sort(onp.concatenate([lp, fine]))
            f2 = self.coll_all(onp.exp(lp2), 1.0)
            g = onp.exp(f2 - f2.max())
            tot.append(
                f2.max() + onp.log(onp.trapezoid(g, lp2)) - onp.log(LHI - LLO)
            )
        return tot


def u_to_P(u):
    return onp.exp(LLO + (LHI - LLO) / (1 + onp.exp(-u)))


def log_logistic(u):
    return -onp.logaddexp(0, -u) - onp.logaddexp(0, u)


def next_beta(inc, beta):
    def cess(d):
        w = inc(d)
        w = onp.exp(w - w.max())
        return w.sum() ** 2 / (w**2).sum() / len(w) - 0.5

    hi = 1.0 - beta
    if FIRST:  # first crossing from below: CESS(d) need not be monotone
        ds = hi * onp.logspace(-14, 0, 141)
        bad = [i for i, x in enumerate(ds) if cess(x) < 0]
        if not bad:
            return hi
        if bad[0] == 0:
            return ds[0]
        lo, hi = ds[bad[0] - 1], ds[bad[0]]
        for _ in range(30):
            mid = onp.sqrt(lo * hi)
            if cess(mid) >= 0:
                lo = mid
            else:
                hi = mid
        return lo
    if cess(hi) >= 0:
        return hi
    lo = hi
    while cess(lo) < 0 and lo > 1e-16:
        lo *= 1e-2
    hi = min(lo * 100, 1.0 - beta)
    for _ in range(40):
        mid = onp.sqrt(lo * hi)
        if cess(mid) >= 0:
            lo = mid
        else:
            hi = mid
    return max(lo, 1e-16)


FIRST = False
GRID = 4000
SCAN = 200000
DEBUG = False


def griddy(prob, u, m, beta, rng):
    """Griddy Gibbs (Ritter & Tanner 1992) on ln P | labels, theta collapsed,
    used as an independence MH proposal (exact for any grid)."""
    edges = onp.linspace(LLO, LHI, GRID + 1)
    mids, wd = 0.5 * (edges[1:] + edges[:-1]), edges[1] - edges[0]
    newu = u.copy()
    for i in range(len(u)):
        lg = prob.coll(
            onp.exp(mids),
            onp.broadcast_to(m[i], (GRID, 1, prob.K)).copy(),
            beta,
        )[:, 0]
        lq = lg - onp.logaddexp.reduce(lg)
        g = rng.choice(GRID, p=onp.exp(lq))
        lpn = edges[g] + rng.uniform() * wd
        lpo = onp.log(u_to_P(u[i : i + 1]))[0]
        bo = min(max(int((lpo - LLO) / wd), 0), GRID - 1)
        tn = prob.coll(onp.exp([lpn]), m[i][None, None], beta)[0, 0]
        to = prob.coll(onp.exp([lpo]), m[i][None, None], beta)[0, 0]
        if onp.log(rng.uniform()) < (tn - lq[g]) - (to - lq[bo]):
            x = (lpn - LLO) / (LHI - LLO)
            newu[i] = onp.log(x) - onp.log1p(-x)
    return newu


def build_q(prob, alpha=0.1):
    """Defensive reference over ln P (Hesterberg 1995): alpha x prior + (1 - alpha)
    x Student-t3 at the scan's best period, scale 3x its Laplace width. The scan
    is a search (any heuristic is allowed: it only sets q)."""
    x = onp.linspace(LLO, LHI, SCAN)
    f = prob.coll_all(onp.exp(x), 1.0)
    i = onp.argmax(f)
    xf = onp.linspace(x[max(i - 3, 0)], x[min(i + 3, len(x) - 1)], 4001)
    ff = prob.coll_all(onp.exp(xf), 1.0)
    j = onp.argmax(ff)
    xh = xf[j]
    w = onp.abs(
        xf[ff > ff[j] - 0.5] - xh
    ).max()  # half-width at -0.5 nats ~ 1 sd
    s = 3 * max(w, (xf[1] - xf[0]))
    edges = onp.unique(
        onp.concatenate(
            [
                onp.linspace(LLO, LHI, 200001),
                xh + s * onp.linspace(-50, 50, 20001),
            ]
        )
    )
    edges = edges[(edges >= LLO) & (edges <= LHI)]
    mid = 0.5 * (edges[1:] + edges[:-1])
    dens = (
        alpha / (LHI - LLO)
        + (1 - alpha) * (1 + ((mid - xh) / s) ** 2 / 3) ** -2 / s
    )
    mass = dens * onp.diff(edges)
    mass /= mass.sum()
    return edges, mass, onp.log(mass / onp.diff(edges))


def run_R(prob, rng):
    edges, mass, lqb = build_q(prob)
    cdf = onp.cumsum(mass)

    def q_draw(n):
        b = onp.minimum(
            onp.searchsorted(cdf, rng.uniform(size=n) * cdf[-1]), len(mass) - 1
        )
        return edges[b] + rng.uniform(size=n) * (edges[b + 1] - edges[b])

    def lq(x):
        return lqb[onp.clip(onp.searchsorted(edges, x) - 1, 0, len(lqb) - 1)]

    lp = -onp.log(LHI - LLO)
    x = q_draw(M)
    m = (rng.uniform(size=(M, prob.K)) < 0.5).astype(float)
    beta, lnz, stages = 0.0, prob.K * onp.log(2.0), 0

    def tgt(x, m, b):
        return (
            (1 - b) * lq(x)
            + b * lp
            + prob.coll(onp.exp(x), m[:, None], b)[:, 0]
        )

    while beta < 1.0:
        l0 = tgt(x, m, beta)
        d = next_beta(lambda s_: tgt(x, m, beta + s_) - l0, beta)
        inc = tgt(x, m, beta + d) - l0
        lnz += onp.logaddexp.reduce(inc) - onp.log(M)
        w = onp.exp(inc - inc.max())
        idx = rng.choice(M, M, p=w / w.sum())
        x, m = x[idx], m[idx]
        beta = min(1.0, beta + d)
        stages += 1
        if DEBUG:
            print(
                f"   b={beta:.3e} inl={m.mean():.2f} near={onp.mean(onp.abs(x - onp.log(P_TRUE)) < 1e-4):.2f} lnz={lnz:.2f}"
            )
        for _ in range(MOVES):
            P = onp.exp(x)
            _, theta = prob.coll(P, m[:, None], beta, draw=True, rng=rng)
            e = prob.ell(theta, P)
            p_in = 1 / (1 + onp.exp(-beta * (e - LNB)))
            m = (rng.uniform(size=p_in.shape) < p_in).astype(float)
            # independence MH from q, then RW in ln P
            xp = q_draw(M)
            ok = onp.log(rng.uniform(size=M)) < (tgt(xp, m, beta) - lq(xp)) - (
                tgt(x, m, beta) - lq(x)
            )
            x = onp.where(ok, xp, x)
            mad = onp.median(onp.abs(x - onp.median(x))) * 1.4826
            xp = x + 2.38 * max(mad, 1e-12) * rng.normal(size=M)
            inside = (xp > LLO) & (xp < LHI)
            ok = inside & (
                onp.log(rng.uniform(size=M))
                < onp.where(
                    inside, tgt(onp.clip(xp, LLO, LHI), m, beta), -onp.inf
                )
                - tgt(x, m, beta)
            )
            x = onp.where(ok, xp, x)
        if stages > 400:
            return onp.nan, stages
    return lnz, stages


def run(prob, method, rng):
    global FIRST
    if method == "R":
        FIRST = True
        return run_R(prob, rng)
    FIRST = method.islower() or method == "G"
    method = method.upper()
    u = rng.logistic(size=M)
    theta = rng.normal(size=(M, 4)) * SW
    m = (rng.uniform(size=(M, prob.K)) < 0.5).astype(float)
    beta, lnz, stages = (
        0.0,
        (0.0 if method == "W" else prob.K * onp.log(2.0)),
        0,
    )

    def target(u, theta, m, b):
        P = u_to_P(u)
        if method == "W":
            return b * prob.ll_whole(theta, P)
        if method == "C":
            return prob.ll_comp(theta, P, b)
        if method in ("J", "G"):
            return prob.coll(P, m[:, None], b)[:, 0]
        return prob.coll_all(P, b)

    while beta < 1.0:
        l0 = target(u, theta, m, beta)
        d = next_beta(lambda s: target(u, theta, m, beta + s) - l0, beta)
        inc = target(u, theta, m, beta + d) - l0
        lnz += onp.logaddexp.reduce(inc) - onp.log(M)
        w = onp.exp(inc - inc.max())
        idx = rng.choice(M, M, p=w / w.sum())
        u, theta, m = u[idx], theta[idx], m[idx]
        beta = min(1.0, beta + d)
        stages += 1
        if method == "G":
            u = griddy(prob, u, m, beta, rng)
        for _ in range(MOVES):
            P = u_to_P(u)
            if method in (
                "C",
                "J",
                "G",
            ):  # tempered label Gibbs + exact theta | labels
                if method in ("J", "G"):
                    _, theta = prob.coll(
                        P, m[:, None], beta, draw=True, rng=rng
                    )
                e = prob.ell(theta, P)
                p_in = 1 / (1 + onp.exp(-beta * (e - LNB)))
                m = (rng.uniform(size=p_in.shape) < p_in).astype(float)
                _, theta = prob.coll(P, m[:, None], beta, draw=True, rng=rng)
            if (
                method == "W"
            ):  # #629 Luger: labels at beta = 1, MH with L^(beta-1)
                e = prob.ell(theta, P)
                p_in = 1 / (1 + onp.exp(-(e - LNB)))
                mm = (rng.uniform(size=p_in.shape) < p_in).astype(float)
                _, prop = prob.coll(P, mm[:, None], 1.0, draw=True, rng=rng)
                lr = (beta - 1) * (
                    prob.ll_whole(prop, P) - prob.ll_whole(theta, P)
                )
                ok = onp.log(rng.uniform(size=M)) < lr
                theta = onp.where(ok[:, None], prop, theta)
            # RW-MH on u (theta fixed for W, C; collapsed for J, X)
            step = 2.38 * max(u.std(), 1e-10)
            up = u + step * rng.normal(size=M)
            if method in ("J", "G"):
                lc, lp = (
                    prob.coll(P, m[:, None], beta)[:, 0],
                    prob.coll(u_to_P(up), m[:, None], beta)[:, 0],
                )
            else:
                lc, lp = target(u, theta, m, beta), target(up, theta, m, beta)
            ok = onp.log(rng.uniform(size=M)) < lp + log_logistic(
                up
            ) - lc - log_logistic(u)
            u = onp.where(ok, up, u)
        if stages > 400:
            return onp.nan, stages
    return lnz, stages


_run_old = run


def run(prob, method, rng):  # noqa: F811
    global FIRST
    if method == "Q":
        FIRST = True
        from r2 import run_R2  # lazy: r2 imports this module

        return run_R2(prob, rng)
    return _run_old(prob, method, rng)


if __name__ == "__main__":
    K, rel, reps = int(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3])
    methods = sys.argv[4] if len(sys.argv) > 4 else "WCJX"
    GRID = int(min(20000, max(2000, 0.05 / (0.1 * rel))))
    prob = Prob(K, rel, seed=0)
    t0 = time.time()
    ex = prob.exact()
    print(
        f"K={K} sigma/a={rel:g} exact lnZ={ex[0]:.3f} (grid x2: {ex[1]:.3f}) "
        f"floor={K * LNB:.3f} [{time.time() - t0:.0f}s]",
        flush=True,
    )
    for meth in methods:
        z, st, co, wt = [], [], [], []
        for r in range(reps):
            prob.cost = 0
            t0 = time.time()
            lz, s = run(prob, meth, onp.random.default_rng(100 + r))
            (
                z.append(lz),
                st.append(s),
                co.append(prob.cost / M),
                wt.append(time.time() - t0),
            )
        z = onp.array(z)
        print(
            f"  {meth}: lnZ={onp.nanmean(z):9.3f} bias={onp.nanmean(z) - ex[1]:8.3f} "
            f"sd={onp.nanstd(z, ddof=1):7.3f} stages={onp.mean(st):5.1f} "
            f"cost/particle={onp.mean(co):.2e} var*cost={onp.nanvar(z, ddof=1) * onp.mean(co):.2e} "
            f"wall={onp.mean(wt):.1f}s  all={onp.round(z, 2).tolist()}",
            flush=True,
        )
