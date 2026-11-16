"""Leg-1 toy shared by the search, evidence and alias-grid scripts (design/orbit_alias_cycles.md §6).

Circular orbit, night positions only, P inside alias cells, TI' with a Gaussian prior N(0, SW^2 I) so
that the reference is exact: for each period the TI' integral is closed form given the night labels,
and the 2^K labels are summed exactly. Each night's likelihood is the defensive mixture
(1 - EPS) N(d_k | p_k(x), sigma^2 I) + EPS / (pi R_OUT^2) (uniform-disc outlier, EPS fixed).

The epochs are clustered seasons (the review's toy_needle epochs extended to 3 and 4 seasons).
Everything is float64 numpy except the target's gradient and Hessian (JAX, x64).
"""

import math

import jax
import jax.numpy as np
import numpy as onp

jax.config.update("jax_enable_x64", True)

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
