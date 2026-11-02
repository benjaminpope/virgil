"""Q: reference-distribution path on (ln P, theta); driver for toy_needle."""

import sys
import numpy as onp
import toy_needle as T
from toy_needle import feats, SW, LNB, LLO, LHI


def run_R2(prob, rng, alpha=0.1, infl=2.0):
    M = T.M
    T.FIRST = True
    edges, mass, lqb = T.build_q(prob, alpha)
    cdf = onp.cumsum(mass)
    lp = -onp.log(LHI - LLO)
    xh = 0.5 * (edges[:-1] + edges[1:])[onp.argmax(lqb)]
    P1 = onp.exp(onp.array([xh]))
    zhat = prob.masks[
        onp.argmax(prob.coll(P1, prob.masks, 1.0)[0])
    ]  # best label set at the scan peak

    def cond(x):
        n = len(x)
        c0, c1 = feats(onp.exp(x), prob.t)
        s2 = prob.sig**2
        A = (zhat * c0 * c0).sum(1) / s2 + 1 / SW**2
        B = (zhat * c0 * c1).sum(1) / s2
        C = (zhat * c1 * c1).sum(1) / s2 + 1 / SW**2
        det = A * C - B * B
        mu = onp.zeros((n, 4))
        for j in (0, 1):
            r0 = (zhat * c0 * prob.d[None, :, j]).sum(1) / s2
            r1 = (zhat * c1 * prob.d[None, :, j]).sum(1) / s2
            mu[:, j] = (C * r0 - B * r1) / det
            mu[:, j + 2] = (A * r1 - B * r0) / det
        cov2 = (
            onp.stack([onp.stack([C, -B], -1), onp.stack([-B, A], -1)], -2)
            / det[:, None, None]
        )
        return mu, cov2

    def lq_theta(x, th):
        mu, cov2 = cond(x)
        cov2 = cov2 * infl**2
        inv, ld = onp.linalg.inv(cov2), onp.linalg.slogdet(cov2)[1]
        out = 0.0
        for j in (0, 1):
            r = onp.stack(
                [th[:, j] - mu[:, j], th[:, j + 2] - mu[:, j + 2]], 1
            )
            out = (
                out
                - 0.5 * onp.einsum("ni,nij,nj->n", r, inv, r)
                - 0.5 * ld
                - onp.log(2 * onp.pi)
            )
        lw = -(th**2).sum(1) / (2 * SW**2) - 2 * onp.log(2 * onp.pi * SW**2)
        return onp.logaddexp(onp.log(1 - alpha) + out, onp.log(alpha) + lw)

    def draw(n):
        b = onp.minimum(
            onp.searchsorted(cdf, rng.uniform(size=n) * cdf[-1]), len(mass) - 1
        )
        x = edges[b] + rng.uniform(size=n) * (edges[b + 1] - edges[b])
        mu, cov2 = cond(x)
        L = onp.linalg.cholesky(cov2 * infl**2)
        th = onp.zeros((n, 4))
        for j in (0, 1):
            e = (L @ rng.normal(size=(n, 2, 1)))[:, :, 0]
            th[:, j], th[:, j + 2] = mu[:, j] + e[:, 0], mu[:, j + 2] + e[:, 1]
        wide = rng.uniform(size=n) < alpha
        th[wide] = rng.normal(size=(wide.sum(), 4)) * SW
        return x, th

    def lq(x, th):
        return lqb[
            onp.clip(onp.searchsorted(edges, x) - 1, 0, len(lqb) - 1)
        ] + lq_theta(x, th)

    def lpost(x, th):
        return (
            lp
            - (th**2).sum(1) / (2 * SW**2)
            - 2 * onp.log(2 * onp.pi * SW**2)
            + prob.ll_whole(th, onp.exp(x))
        )

    def tgt(x, th, b):
        return (1 - b) * lq(x, th) + b * lpost(x, th)

    x, th = draw(M)
    beta, lnz, stages = 0.0, 0.0, 0
    while beta < 1.0:
        l0 = tgt(x, th, beta)
        d = T.next_beta(lambda s_: tgt(x, th, beta + s_) - l0, beta)
        inc = tgt(x, th, beta + d) - l0
        lnz += onp.logaddexp.reduce(inc) - onp.log(M)
        w = onp.exp(inc - inc.max())
        idx = rng.choice(M, M, p=w / w.sum())
        x, th = x[idx], th[idx]
        beta = min(1.0, beta + d)
        stages += 1
        for _ in range(T.MOVES):
            xp, tp = draw(M)
            ok = onp.log(rng.uniform(size=M)) < (
                tgt(xp, tp, beta) - lq(xp, tp)
            ) - (tgt(x, th, beta) - lq(x, th))
            x, th = onp.where(ok, xp, x), onp.where(ok[:, None], tp, th)
            P = onp.exp(x)
            e = prob.ell(th, P)
            p_in = 1 / (1 + onp.exp(-(e - LNB)))
            mm = (rng.uniform(size=p_in.shape) < p_in).astype(float)
            _, tp = prob.coll(P, mm[:, None], 1.0, draw=True, rng=rng)
            lr = (tgt(x, tp, beta) - lpost(x, tp)) - (
                tgt(x, th, beta) - lpost(x, th)
            )
            ok = onp.log(rng.uniform(size=M)) < lr
            th = onp.where(ok[:, None], tp, th)
        if stages > 400:
            return onp.nan, stages
    return lnz, stages


if __name__ == "__main__":
    K, rel, reps = int(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3])
    prob = T.Prob(K, rel, seed=0)
    ex = prob.exact()
    z, st, co = [], [], []
    for r in range(reps):
        prob.cost = 0
        lz, s = run_R2(prob, onp.random.default_rng(100 + r))
        z.append(lz), st.append(s), co.append(prob.cost / T.M)
    z = onp.array(z)
    print(
        f"K={K} sigma/a={rel:g} exact={ex[1]:.3f}  Q: lnZ={z.mean():.3f} bias={z.mean() - ex[1]:.3f} "
        f"sd={z.std(ddof=1):.3f} stages={onp.mean(st):.1f} cost/particle={onp.mean(co):.2e} "
        f"var*cost={z.var(ddof=1) * onp.mean(co):.2e} all={onp.round(z, 2).tolist()}",
        flush=True,
    )
