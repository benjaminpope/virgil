"""Toy for design/orbit_alias_cycles.md: Luger-move leg against the Joker-style leg.

Fixed circular core (period known), anchor-frame Thiele-Innes constants
theta = (A', B', F', G') with the isotropic, log-uniform-a prior, K nights
of 2-D positions, each night an inlier (Gaussian) or an outlier (uniform
disc), with labels in the particle state.

  L : SMC from the isotropic prior, likelihood tempering beta 0 -> 1, moves
      = label Gibbs + Metropolis-Hastings independence move on theta from
      its Gaussian conditional under a working prior (the Luger move).
  J : SMC from the Gaussian working prior (exact conditional moves, no
      rejection), tempering beta 0 -> 1, then the prior ratio
      (iso / working)^gamma tempered in, gamma 0 -> 1, same moves.

Both choose the next step by a conditional ESS of 1/2 and estimate
ln Z = ln int p_iso(theta) prod_k L_k(theta) d theta. Cost is the number of
per-particle theta conditionals (one 4x4 solve) plus likelihood evaluations.
Prints ln Z mean/sd over replicates, steps, cost and variance x cost.

    .venv/bin/python scripts/toy_tiprime_legs.py 6.6 15

The arguments are the working scales s_w; ``--faceon`` uses a nearly face-on truth,
``--zmove`` adds a z-space random-walk step to each move, ``--moves N`` sets the moves per stage (default 2). Each data seed prints the exact ln Z.
"""

import sys

import numpy as onp
from scipy.optimize import brentq

K, SIGMA, EPS, A_LO, A_HI = 8, 0.3, 0.1, 1.0, 30.0
R_OUT = A_HI  # outlier density: uniform on a disc of radius a_hi
LN_B = -onp.log(onp.pi * R_OUT**2)
N, REPS, MOVES = 1000, 10, 2
TRUTH = [
    4.0,
    -2.5,
    1.5,
    3.0,
]  # --faceon: a nearly face-on orbit (|V| -> 0, the cusp)
NORM = onp.log(onp.pi**2 * onp.log(A_HI / A_LO))  # ln of the p_iso constant


def design(phi):
    """H_k (2 x 4) of the position (dec, ra) in theta = (A, B, F, G)."""
    h = onp.zeros((len(phi), 2, 4))
    h[:, 0, 0], h[:, 0, 2] = onp.cos(phi), onp.sin(phi)
    h[:, 1, 1], h[:, 1, 3] = onp.cos(phi), onp.sin(phi)
    return h


def sample_iso(n, rng):
    lnr = rng.uniform(
        onp.log(onp.sqrt(2 * A_LO)), onp.log(onp.sqrt(2 * A_HI)), n
    )
    w = rng.normal(size=(n, 4))
    z = onp.exp(lnr)[:, None] * w / onp.linalg.norm(w, axis=1)[:, None]
    u, v = (z[:, 0] + 1j * z[:, 1]) ** 2, (z[:, 2] + 1j * z[:, 3]) ** 2
    return onp.stack(
        [(u.real + v.real) / 2, (u.imag - v.imag) / 2,
         (-v.imag - u.imag) / 2, (u.real - v.real) / 2], axis=1)  # fmt: skip


def log_iso(theta):
    """ln p_iso(TI'), -inf outside the a range."""
    a, b, f, g = theta.T
    u, v = onp.hypot(a + g, b - f), onp.hypot(a - g, b + f)
    ok = (u + v >= 2 * A_LO) & (u + v <= 2 * A_HI)
    with onp.errstate(divide="ignore"):
        out = -NORM - 2 * onp.log(u + v) - onp.log(u) - onp.log(v)
    return onp.where(ok, out, -onp.inf)


class Toy:
    def __init__(self, rng, s_w):
        self.rng, self.s_w = rng, s_w
        t = onp.sort(rng.uniform(0, 1, K))
        self.h = design(2 * onp.pi * 3.37 * t)  # a few cycles, irregular
        truth = onp.array(TRUTH)
        self.d = self.h @ truth + SIGMA * rng.normal(size=(K, 2))
        bad = rng.choice(K, 2, replace=False)  # two outlier nights
        self.d[bad] = rng.uniform(-8, 8, (2, 2))
        self.cost = 0

    def comp(self, theta):
        """ln of the two components per night: (n, K) inlier, outlier."""
        r = self.d[None] - onp.einsum("kij,nj->nki", self.h, theta)
        inl = (onp.log(1 - EPS) - (r**2).sum(-1) / (2 * SIGMA**2)
               - onp.log(2 * onp.pi * SIGMA**2))  # fmt: skip
        self.cost += theta.shape[0] * K
        return inl, onp.full_like(inl, onp.log(EPS) + LN_B)

    def log_mix(self, theta, beta):
        i, o = self.comp(theta)
        return onp.logaddexp(beta * i, beta * o).sum(1)

    def sample_labels(self, theta, beta):
        i, o = self.comp(theta)
        p_out = 1 / (1 + onp.exp(beta * (i - o)))
        return self.rng.uniform(size=p_out.shape) < p_out  # True = outlier

    def conditional(self, labels, beta, scale_work):
        """Gaussian conditional of theta | labels (inliers only) at power beta
        under the working prior N(0, s_w^2 I): mean (n,4), cholesky of cov."""
        w_in = (~labels).astype(float)  # (n, K)
        hh = onp.einsum("kij,kil->kjl", self.h, self.h)  # (K, 4, 4)
        hd = onp.einsum("kij,ki->kj", self.h, self.d)  # (K, 4)
        lam = (beta / SIGMA**2) * onp.einsum("nk,kjl->njl", w_in, hh)
        lam += onp.eye(4)[None] / scale_work**2
        rhs = (beta / SIGMA**2) * (w_in @ hd)
        cov = onp.linalg.inv(lam)
        self.cost += len(labels)
        return onp.einsum("njl,nl->nj", cov, rhs), onp.linalg.cholesky(cov)

    def log_work(self, theta):
        return -(theta**2).sum(1) / (2 * self.s_w**2) - 4 * onp.log(
            self.s_w) - 2 * onp.log(2 * onp.pi)  # fmt: skip


def ti_to_z(theta):
    a, b, f, g = theta.T
    zu, zv = (
        onp.sqrt((a + g) + 1j * (b - f) + 0j),
        onp.sqrt((a - g) - 1j * (b + f) + 0j),
    )
    return onp.stack([zu.real, zu.imag, zv.real, zv.imag], axis=1)


def z_to_ti(z):
    u, v = (z[:, 0] + 1j * z[:, 1]) ** 2, (z[:, 2] + 1j * z[:, 3]) ** 2
    return onp.stack(
        [
            (u.real + v.real) / 2,
            (u.imag - v.imag) / 2,
            (-v.imag - u.imag) / 2,
            (u.real - v.real) / 2,
        ],
        axis=1,
    )


def next_step(log_inc, lo, hi=1.0, target=0.5):
    """Largest step d in (0, hi - lo] with conditional ESS >= target * n.
    ``log_inc(d)`` is the per-particle log incremental weight for a step d."""
    n = len(log_inc(1e-9))

    def cess(d):
        w = log_inc(d)
        w = onp.exp(w - w.max())
        return w.sum() ** 2 / (w**2).sum() / n - target

    if cess(hi - lo) >= 0:
        return hi - lo
    return brentq(cess, 1e-9, hi - lo, xtol=1e-6)


def resample(rng, w, *arrays):
    idx = rng.choice(len(w), len(w), p=w / w.sum())
    return [a[idx] for a in arrays]


def smc(toy, prior):
    """Returns (ln Z estimate, steps). ``prior``: 'iso' (L) or 'work' (J)."""
    rng, s_w = toy.rng, toy.s_w
    use_iso = prior == "iso"
    theta = sample_iso(N, rng) if use_iso else rng.normal(size=(N, 4)) * s_w
    beta, gamma = 0.0, 0.0 if not use_iso else 1.0
    labels = toy.sample_labels(theta, 0.0)
    ln_z = K * onp.log(2.0)  # beta = 0: two labels per night, normalized prior
    steps = 0

    def prior_ratio(th):
        return log_iso(th) - toy.log_work(th)

    def move(theta, labels, beta, gamma):
        """Gibbs on labels, MH independence on theta; target prior
        p_work^(1-gamma) p_iso^gamma (gamma = 1 for L)."""
        for _ in range(MOVES):
            labels = toy.sample_labels(theta, beta)
            mean, chol = toy.conditional(labels, beta, s_w)
            prop = mean + onp.einsum(
                "njl,nl->nj", chol, rng.normal(size=(len(theta), 4))
            )
            if gamma > 0:
                with onp.errstate(invalid="ignore"):
                    ratio = gamma * (prior_ratio(prop) - prior_ratio(theta))
                ok = onp.log(rng.uniform(size=len(theta))) < onp.nan_to_num(
                    ratio, nan=-onp.inf
                )
                theta = onp.where(ok[:, None], prop, theta)
            else:
                theta = prop
            if (
                ZMOVE and gamma == 1.0
            ):  # symmetric random walk in z: target |z|^-4 x likelihood, the 4:1 cover is exact
                z = ti_to_z(theta)
                zp = z + (2.38 / 2.0) * z.std(0).mean() * rng.normal(
                    size=z.shape
                )
                shell = (zp**2).sum(1)
                ok_shell = (shell >= 2 * A_LO) & (shell <= 2 * A_HI)
                prop = z_to_ti(zp)
                lp = toy.log_mix(prop, beta) - 2 * onp.log(shell)
                lc = toy.log_mix(theta, beta) - 2 * onp.log((z**2).sum(1))
                ok = ok_shell & (
                    onp.log(rng.uniform(size=len(theta))) < lp - lc
                )
                theta = onp.where(ok[:, None], prop, theta)
        return theta, labels

    # stage 1: likelihood tempering (L: under the iso prior via MH; J: under
    # the working prior, exact moves)
    g1 = 1.0 if use_iso else 0.0
    while beta < 1.0:
        l0 = toy.log_mix(theta, beta)
        d = next_step(lambda s: toy.log_mix(theta, beta + s) - l0, beta, 1.0)
        ln_inc = toy.log_mix(theta, beta + d) - l0
        ln_z += onp.logaddexp.reduce(ln_inc) - onp.log(len(ln_inc))
        w = onp.exp(ln_inc - ln_inc.max())
        theta, labels = resample(rng, w, theta, labels)
        beta += d
        theta, labels = move(theta, labels, beta, g1)
        steps += 1
        if steps > 400:
            raise RuntimeError("no convergence")
    if use_iso:
        return ln_z, steps
    # stage 2 (J): swap working -> iso prior, gamma 0 -> 1 at beta = 1
    while gamma < 1.0:
        pr = prior_ratio(theta)
        d = next_step(lambda s: s * pr, gamma, 1.0)
        ln_inc = d * pr
        ln_z += onp.logaddexp.reduce(ln_inc) - onp.log(len(ln_inc))
        w = onp.exp(ln_inc - ln_inc.max())
        theta, labels = resample(rng, w, theta, labels)
        gamma += d
        theta, labels = move(theta, labels, 1.0, gamma)
        steps += 1
        if steps > 800:
            raise RuntimeError("no convergence")
    return ln_z, steps


def exact_ln_z(toy, n_mc=200_000, seed=0):
    """Reference ln Z by enumerating the night labels (no SMC).

    Given the labels, the inlier nights are Gaussian in theta, so
    Z = sum_z (eps b)^n_out (1-eps)^n_in c_z E_{N(m_z, Lam_z^-1)}[p_iso];
    the expectation is a plain Monte Carlo average (finite variance: each
    posterior sits far from the cusp). Label sets with fewer than two
    inliers leave theta unconstrained and carry weight <~ (eps b)^(K-1),
    negligible here, and are dropped.
    """
    rng = onp.random.default_rng(seed)
    hh = onp.einsum("kij,kil->kjl", toy.h, toy.h)
    hd = onp.einsum("kij,ki->kj", toy.h, toy.d)
    terms = []
    for mask in range(2**K):
        inl = onp.array([(mask >> k) & 1 for k in range(K)], bool)
        n_in = int(inl.sum())
        if n_in < 2:
            continue
        lam = hh[inl].sum(0) / SIGMA**2
        m = onp.linalg.solve(lam, hd[inl].sum(0) / SIGMA**2)
        s_min = ((toy.d[inl] - toy.h[inl] @ m) ** 2).sum()
        ln_c = (-n_in * onp.log(2 * onp.pi * SIGMA**2) - s_min / (2 * SIGMA**2)
                + 2 * onp.log(2 * onp.pi) - 0.5 * onp.linalg.slogdet(lam)[1])  # fmt: skip
        theta = (
            m
            + rng.normal(size=(n_mc, 4))
            @ onp.linalg.cholesky(onp.linalg.inv(lam)).T
        )
        ln_e = onp.logaddexp.reduce(log_iso(theta)) - onp.log(n_mc)
        ln_w = (K - n_in) * (onp.log(EPS) + LN_B) + n_in * onp.log(1 - EPS)
        terms.append(ln_w + ln_c + ln_e)
    return float(onp.logaddexp.reduce(terms))


def main():
    global MOVES, TRUTH, ZMOVE
    args = sys.argv[1:]
    # --zmove: one random-walk Metropolis step in the z coordinates after the Luger move
    ZMOVE = "--zmove" in args
    if "--faceon" in args:
        TRUTH = [4.0, -2.5, 2.5, 4.0]
    if "--moves" in args:
        MOVES = int(args[args.index("--moves") + 1])
    nums = [
        a
        for i, a in enumerate(args)
        if not a.startswith("--") and (i == 0 or args[i - 1] != "--moves")
    ]
    s_w_values = [float(s) for s in nums] or [6.6, 15.0]
    for data_seed in range(3):
        print(f"data seed {data_seed}")
        ref_toy = Toy(onp.random.default_rng(1000 * data_seed), 1.0)
        print(f"  exact ln Z = {exact_ln_z(ref_toy):9.3f}")
        for name, prior in (("L Luger", "iso"), ("J Joker", "work")):
            for s_w in s_w_values:
                lz, st, cost = [], [], []
                for rep in range(REPS):
                    toy = Toy(onp.random.default_rng(1000 * data_seed), s_w)
                    toy.rng = onp.random.default_rng(10_000 + rep)
                    z, steps = smc(toy, prior)
                    lz.append(z), st.append(steps), cost.append(toy.cost)
                lz = onp.array(lz)
                print(
                    f"  {name:8s} s_w={s_w:5.1f}  lnZ={lz.mean():9.3f}  "
                    f"sd={lz.std(ddof=1):6.3f}  steps={onp.mean(st):5.1f}  "
                    f"cost={onp.mean(cost):.2e}  var*cost={lz.var(ddof=1) * onp.mean(cost):.2e}"
                )


if __name__ == "__main__":
    main()
