"""The leg-1 toy of design/orbit_alias_cycles.md section 6, for tests.

A circular orbit with night positions only, ``P`` inside alias cells, ``TI'``
with a Gaussian prior, and every night a mixture of an inlier Gaussian and a
uniform-disc outlier. The reference is exact: ``TI'`` is integrated in closed
form given the night labels, and the ``2^K`` labels and ``ln P`` are summed
exactly. The solver (minimal triplets, variable projection, C-steps) is the
toy's own, independent of ``virgil.orbit_cells``.

Run the target inside ``jax.enable_x64(True)``.
"""

import itertools
import math

import jax
import jax.numpy as np
import numpy as onp
from scipy import optimize
from scipy.signal import find_peaks

EPS, R_OUT, SW = 0.1, 30.0, 6.6
LNB = math.log(EPS) - math.log(math.pi * R_OUT**2)  # the floor, per night
LN_IN = math.log(1 - EPS)
TRUTH = onp.array([4.0, -2.5, 1.5, 3.0])  # (A, B, F, G)
A_TRUE = 4.7
P_TRUE = 12.1
PATTERNS = {  # name: (epochs in days, outlier nights)
    "K6_2s": ([0, 2.4, 5.1, 238, 240.6, 247.4], []),
    "K10_2s": ([0, 1.2, 2.4, 3.7, 5.1, 238, 239.3, 240.6, 244, 247.4], [7]),
    "K10_3s": (
        [0, 1.2, 2.4, 3.7, 151.0, 152.2, 154.9, 300.0, 301.3, 304.6],
        [5],
    ),
    "K14_4s": (
        [
            0,
            1.1,
            2.3,
            3.5,
            121,
            122.4,
            124,
            330,
            331.2,
            333.1,
            335,
            480,
            481.5,
            483,
        ],
        [5, 9],
    ),
}


def features(P, t):
    """cos and sin of the mean anomaly 2 pi t / P (circular, anchor at t = 0): (n, K) each."""
    ph = 2 * onp.pi * onp.asarray(t)[None] / onp.atleast_1d(P)[:, None]
    return onp.cos(ph), onp.sin(ph)


class Problem:
    """Data, cell and target of one toy data set."""

    def __init__(
        self,
        pattern,
        rel,
        seed=0,
        n_cell=None,
        cells=0,
        truth=TRUTH,
        p_true=P_TRUE,
    ):
        t, bad = PATTERNS[pattern] if isinstance(pattern, str) else pattern
        rng = onp.random.default_rng(seed)
        self.t = onp.asarray(t, float)
        self.K = len(self.t)
        self.sig = rel * A_TRUE
        self.T = float(self.t[-1] - self.t[0])  # anchors: first and last night
        c0, c1 = features(p_true, self.t)
        dec = truth[0] * c0[0] + truth[2] * c1[0]
        ra = truth[1] * c0[0] + truth[3] * c1[0]
        self.d = onp.stack([dec, ra], 1) + self.sig * rng.normal(
            size=(self.K, 2)
        )
        for k in bad:
            self.d[k] = rng.uniform(
                -8, 8, 2
            )  # an outlier night: anywhere in the field
        n = round(self.T / p_true) if n_cell is None else n_cell
        self.set_cell(n - cells, n + cells)
        self._lp = jax.jit(jax.vmap(self._logpost_one))
        self._vg = jax.jit(jax.value_and_grad(self._logpost_one))
        self._hess = jax.jit(jax.hessian(self._logpost_one))
        self._vg_lab = jax.jit(jax.value_and_grad(self._logpost_labels))
        self._hess_lab = jax.jit(jax.hessian(self._logpost_labels))

    def set_cell(self, n_lo, n_hi):
        """The period domain: cells n_lo..n_hi, x = T/P in [n_lo - 1/2, n_hi + 1/2]."""
        self.n_lo, self.n_hi = n_lo, n_hi
        self.LLO = math.log(self.T / (n_hi + 0.5))
        self.LHI = math.log(self.T / (n_lo - 0.5))

    # ---- the target in w = (logit of ln P within the domain, theta): log prior + log likelihood
    def _u(self, w0):
        return self.LLO + (self.LHI - self.LLO) * jax.nn.sigmoid(w0)

    def _logpost_one(self, w):
        return self._logpost_labels(w, None)

    def _logpost_labels(self, w, mask):
        """The target with each night's label fixed by ``mask`` (1 inlier, 0 outlier), or summed
        (``mask`` None)."""
        u = self._u(w[0])
        th = w[1:]
        ph = 2 * np.pi * self.t / np.exp(u)
        c0, c1 = np.cos(ph), np.sin(ph)
        dec = th[0] * c0 + th[2] * c1
        ra = th[1] * c0 + th[3] * c1
        r2 = (self.d[:, 0] - dec) ** 2 + (self.d[:, 1] - ra) ** 2
        ell = LN_IN - np.log(2 * np.pi * self.sig**2) - r2 / (2 * self.sig**2)
        if mask is None:
            loglike = np.sum(np.logaddexp(ell, LNB))
        else:
            loglike = np.sum(np.where(mask > 0, ell, LNB))
        # ln P uniform on the domain, mapped by a logistic: log density of w0 is the logistic's
        lw0 = -jax.nn.softplus(-w[0]) - jax.nn.softplus(w[0])
        lth = -0.5 * np.sum(th**2) / SW**2 - 2 * np.log(2 * np.pi * SW**2)
        return loglike + lw0 + lth

    def logpost(self, w):
        return onp.asarray(self._lp(np.asarray(onp.atleast_2d(w))))

    def logprior(self, w):
        w = onp.atleast_2d(w)
        lw0 = -onp.logaddexp(0, -w[:, 0]) - onp.logaddexp(0, w[:, 0])
        return (
            lw0
            - 0.5 * (w[:, 1:] ** 2).sum(1) / SW**2
            - 2 * onp.log(2 * onp.pi * SW**2)
        )

    def prior_draw(self, n, rng):
        return onp.column_stack(
            [rng.logistic(size=n), SW * rng.normal(size=(n, 4))]
        )

    def w_of_u(self, u):
        x = (onp.asarray(u) - self.LLO) / (self.LHI - self.LLO)
        return onp.log(x) - onp.log1p(-x)

    def u_of_w(self, w0):
        return self.LLO + (self.LHI - self.LLO) / (
            1 + onp.exp(-onp.asarray(w0))
        )

    # ---- exact reference: labels and theta summed exactly; ln P by adaptive quadrature
    def collapsed(self, u, chunk=128):
        """ln of the label-summed, theta-integrated likelihood at each ln P in ``u``
        (theta prior N(0, SW^2 I)), i.e. the density of ln P before its own prior."""
        masks = (
            (onp.arange(2**self.K)[:, None] >> onp.arange(self.K)) & 1
        ).astype(float)
        nin = masks.sum(1)
        s2 = self.sig**2
        out = []
        for i in range(0, len(u), chunk):
            c0, c1 = features(onp.exp(u[i : i + chunk]), self.t)
            A = c0 * c0 @ masks.T / s2 + 1 / SW**2
            B = c0 * c1 @ masks.T / s2
            C = c1 * c1 @ masks.T / s2 + 1 / SW**2
            det = A * C - B * B
            quad = 0.0
            for j in (0, 1):
                r0 = (c0 * self.d[None, :, j]) @ masks.T / s2
                r1 = (c1 * self.d[None, :, j]) @ masks.T / s2
                quad = quad + (C * r0**2 - 2 * B * r0 * r1 + A * r1**2) / det
            Q = (masks @ (self.d**2).sum(1))[None] / s2
            lg = (
                -nin * onp.log(2 * onp.pi * s2)
                - 2 * onp.log(SW**2)
                - onp.log(det)
                - 0.5 * (Q - quad)
                + nin * LN_IN
                + (self.K - nin) * LNB
            )
            m = lg.max(1, keepdims=True)
            out.append(
                (m + onp.log(onp.exp(lg - m).sum(1, keepdims=True)))[:, 0]
            )
        return onp.concatenate(out)

    def needle_width(self):
        """The ln P scale of the narrowest feature: one sigma of phase at the last night."""
        return self.sig / (A_TRUE * 2 * onp.pi * self.T / P_TRUE)

    def exact(self, n0=None, cells=None):
        """ln Z per cell (dict n -> ln Z_n), by trapezoid on a grid fine enough for the needle
        (>= 4 points per sigma) plus a 20x refinement around every local maximum within 40 nats of
        its own cell's maximum, so cells far below the best one are refined too."""
        w = self.needle_width()
        n0 = n0 or int(min(4e5, max(2e4, 4 * (self.LHI - self.LLO) / w)))
        if n0 < 4 * (self.LHI - self.LLO) / w:
            print(
                f"exact: {n0} points is under 4 per needle sigma; check convergence in n0"
            )
        u = onp.linspace(self.LLO, self.LHI, n0)
        f = self.collapsed(u)
        cell = onp.rint(self.T / onp.exp(u)).astype(int)
        f_cell = onp.full(cell.max() + 1, -onp.inf)
        onp.maximum.at(f_cell, cell, f)
        loc = (
            onp.flatnonzero(
                (f[1:-1] >= f[:-2])
                & (f[1:-1] >= f[2:])
                & (f[1:-1] > f_cell[cell[1:-1]] - 40)
            )
            + 1
        )
        extra = [
            onp.linspace(u[max(i - 3, 0)], u[min(i + 3, n0 - 1)], 121)
            for i in loc
        ]
        if extra:
            ue = onp.concatenate(extra)
            u = onp.concatenate([u, ue])
            f = onp.concatenate([f, self.collapsed(ue)])
            o = onp.argsort(u)
            u, f = u[o], f[o]
        # ln P uniform over the whole domain: each cell's Z is its integral / total ln P width
        x = self.T / onp.exp(u)
        res = {}
        for n in range(self.n_lo, self.n_hi + 1):
            sel = (x >= n - 0.5) & (x <= n + 0.5)
            g = onp.exp(f[sel] - f[sel].max())
            res[n] = (
                f[sel].max()
                + onp.log(onp.trapezoid(g, u[sel]))
                - onp.log(self.LHI - self.LLO)
            )
        return res, (u, f)


GRAD_COST = 3.0


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
    peaks with mass >= 1e-3 (found / all). Oracle peaks: local maxima of the exact collapsed
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
    big = mass >= 1e-3
    return float(mass[found].sum()), int(big.sum()), int((found & big).sum())
