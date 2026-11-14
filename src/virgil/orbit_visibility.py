"""Visibility-level leg, exact outlier floor and labels of the alias-resolved orbit fit.

!!! warning "Provisional"
    May change in any 0.4.x release without a deprecation cycle.

Stage 3c of the plan in ``design/orbit_alias_cycles.md``. It adds to
[`virgil.orbit_cells`][virgil.orbit_cells]:

- the **outlier floor** ``b_k`` of design §2, exact: night ``k``'s per-exposure
  likelihood, with within-night motion and its own error scales, integrated against the
  joint prior predictive ``π(p, v) π(f)`` of the night state by adaptive Genz–Malik
  cubature around the night's peaks
  ([`outlier_exact`][virgil.orbit_visibility.outlier_exact]). This is #618's integral with
  the velocity added, in the coordinates where the prior is smooth;
- the **visibility leg** (§6.2): each position-level mode of a cell is refined by Newton
  on the per-exposure target, with one global inlier scale per error block in the sampled
  vector, gains and closure offsets marginalized analytically at that scale and the
  outlier floor ``b_k`` shared with the position leg; a defensive mixture on the refined
  modes gives ``ln Z`` by importance sampling and an HMC resample-move gives the draws
  ([`visibility_leg`][virgil.orbit_visibility.visibility_leg],
  [`visibility_cells`][virgil.orbit_visibility.visibility_cells]), as
  [`mode_search`][virgil.mode_search] does at position level. No tempering;
- the **labels**: the brighter/fainter label and the antipodal swap ``TI′ → −TI′`` are one
  symmetry, [`swap_label`][virgil.orbit_visibility.swap_label]; posteriors are reported in
  invariants ([`report_invariants`][virgil.orbit_visibility.report_invariants]);
- [`fit_cells`][virgil.orbit_visibility.fit_cells], the whole new path on prepared data.

Nothing here touches the old search path. The primary diameter is held at the value of
``positions`` and no visibility amplitude is fitted.

Source: Genz & Malik (1980), Hesterberg (1995), Gilks & Berzuini (2001), Luger,
Foreman-Mackey & Hogg (2017), [references](../references.md); the contract and the table
are derived in the design note.
"""

import copy
import dataclasses
import functools
import math
import warnings

import jax
import jax.numpy as np
import numpy as onp
from jax.scipy.special import logsumexp
from scipy import ndimage
from scipy.special import logsumexp as onp_logsumexp

from . import _orbit_coordinates as oc
from . import orbit_cells as cells_mod
from ._precision import cast_tree, run_in
from .epochs import Epochs, _Surface
from .mode_search import (
    DefensiveMixture,
    ESS_FAIL,
    N_IS,
    W,
    Mode,
    distinct_modes,
    importance_evidence,
    laplace_log_evidence,
    resample_move,
)
from .night_summaries import (
    DELTA,
    EPS_PRIOR,
    _model,
    keep_peaks,
    night_summaries,
)
from .orbit_cells import (
    A_RANGE,
    E_MAX,
    FLUX_FLOOR,
    S_MAX,
    Cell,
    _Z,
    outlier_prior,
    period_domain,
    position_cells,
)

__all__ = [
    "CellsFit",
    "OutlierTable",
    "VisibilityCell",
    "fit_cells",
    "outlier_exact",
    "outlier_table",
    "report_invariants",
    "swap_label",
    "visibility_cells",
    "visibility_leg",
]

__provisional__ = ("*",)

# Tolerance of the cubature: the integral's relative error, 4 δ (δ is the reporting
# precision of a probability, design §10). #618's value.
_TOL = 4.0 * DELTA
# Half width of a peak's cubature box in standard deviations of the night's own
# (own-scale marginalized) peak. A t distribution with ν ≥ 9 degrees of freedom per
# scale block has under 1e-4 of its mass beyond five; the box is cut to the prior's support.
_WINDOW = 5.0
# Budget: cubature points per night. A night that reaches it is reported, not hidden.
_MAX_EVALS = 400_000
# Points per compiled call: one shape for every night (padded).
_CHUNK = 1024
_LOG_TINY = -1e30


# ---------------------------------------------------------------------------
# The prior predictive on a grid
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class OutlierTable:
    """The Gaussian KDE of design §2.2 on a grid, so that the cubature of ``b_k`` can
    evaluate it at 10⁵ points per night.

    The 2²⁰ samples are binned (linear bins of half a bandwidth) and smoothed by a
    Gaussian filter of the KDE's bandwidth; the binning widens the kernel's variance by
    1/48 (a 1 % bandwidth change). ``dens`` is the density in
    ``(ln|p|, ln|v|, ψ)`` and ``dens_p`` the one in ``ln|p|`` alone (the velocity
    integrated out, for a night with a single exposure); they are interpolated linearly
    as densities, so a zero cell never turns a neighbour into ``nan``. Outside the grid, which pads the
    samples by four bandwidths, the density is zero.
    """

    lo: onp.ndarray  # (2,) grid origin of (ln|p|, ln|v|)
    step: onp.ndarray  # (3,) bin widths of (ln|p|, ln|v|, ψ)
    dens: (
        onp.ndarray
    )  # (n_p, n_v, n_ψ + 2) with the ψ axis wrapped by one cell
    dens_p: onp.ndarray  # (n_p,)
    ln_f_floor: float

    def _inside(self, lp, lv=None):
        n_p = self.dens.shape[0]
        ok = (lp >= self.lo[0]) & (lp <= self.lo[0] + n_p * self.step[0])
        if lv is not None:
            n_v = self.dens.shape[1]
            ok &= (lv >= self.lo[1]) & (lv <= self.lo[1] + n_v * self.step[1])
        return ok

    def log_density(self, lp, lv, psi):
        """``ln g`` at ``(ln|p|, ln|v|, ψ)`` (arrays), ``-inf`` off the grid."""
        ok = self._inside(lp, lv)
        psi = (psi + onp.pi) % (2 * onp.pi) - onp.pi
        coords = onp.stack(
            [
                (lp - self.lo[0]) / self.step[0] - 0.5,
                (lv - self.lo[1]) / self.step[1] - 0.5,
                (psi + onp.pi) / self.step[2] - 0.5 + 1.0,
            ]
        )
        out = ndimage.map_coordinates(
            self.dens, onp.where(ok, coords, 0.0), order=1, mode="nearest"
        )
        return onp.where(ok, _safe_log(out), -onp.inf)

    def log_density_p(self, lp):
        """``ln g_p`` at ``ln|p|``: the velocity integrated out."""
        ok = self._inside(lp)
        coords = (lp - self.lo[0]) / self.step[0] - 0.5
        out = ndimage.map_coordinates(
            self.dens_p, [onp.where(ok, coords, 0.0)], order=1, mode="nearest"
        )
        return onp.where(ok, _safe_log(out), -onp.inf)

    def bounds(self):
        """``((ln|p|_lo, ln|p|_hi), (ln|v|_lo, ln|v|_hi))`` of the grid."""
        n_p, n_v = self.dens.shape[:2]
        return (
            (self.lo[0], self.lo[0] + n_p * self.step[0]),
            (self.lo[1], self.lo[1] + n_v * self.step[1]),
        )


def _safe_log(x):
    with onp.errstate(divide="ignore"):
        return onp.where(x > 1e-300, onp.log(onp.maximum(x, 1e-300)), -onp.inf)


def outlier_table(prior):
    """The grid form of an [`OutlierPrior`][virgil.orbit_cells.OutlierPrior].

    Parameters
    ----------
    prior : OutlierPrior
        The tabulated prior predictive of ``(ln|p|, ln|v|, ψ)`` and its Scott bandwidths.

    Returns
    -------
    OutlierTable
    """
    s, h = (
        onp.asarray(prior.samples, float),
        onp.asarray(prior.bandwidth, float),
    )
    step = h / 2.0
    n_psi = int(math.ceil(2 * math.pi / step[2]))
    step[2] = 2 * math.pi / n_psi
    lo = s[:, :2].min(0) - 4.0 * h[:2]
    hi = s[:, :2].max(0) + 4.0 * h[:2]
    n_pv = onp.ceil((hi - lo) / step[:2]).astype(int)
    edges = [
        lo[0] + step[0] * onp.arange(n_pv[0] + 1),
        lo[1] + step[1] * onp.arange(n_pv[1] + 1),
        -math.pi + step[2] * onp.arange(n_psi + 1),
    ]
    hist, _ = onp.histogramdd(s, bins=edges)
    sigma = h / step
    dens = ndimage.gaussian_filter(
        hist, sigma, mode=("constant", "constant", "wrap"), truncate=4.0
    ) / (len(s) * onp.prod(step))
    dens = onp.concatenate(
        [dens[..., -1:], dens, dens[..., :1]], axis=2
    )  # wrap ψ by one cell for the interpolation
    marg = ndimage.gaussian_filter1d(hist.sum((1, 2)), sigma[0], truncate=4.0)
    return OutlierTable(
        lo, step, dens, marg / (len(s) * step[0]), prior.ln_f_floor
    )


# ---------------------------------------------------------------------------
# The nights' visibility likelihood at a linear-motion companion
# ---------------------------------------------------------------------------


def _centred(d, t_ref, dtype):
    """``d`` with its sample times relative to ``t_ref`` (float64 on the host, so that
    float32 holds them) and its static epoch zero: one compiled kernel serves every
    night of one shape, and the model's epoch is the night's own."""
    out = copy.copy(d)
    dt = onp.asarray(d.mjd, onp.float64) - float(t_ref)
    object.__setattr__(out, "dt", np.asarray(dt))
    object.__setattr__(out, "t_ref", 0.0)
    return cast_tree(out, dtype)


@dataclasses.dataclass
class _Group:
    surface: object
    data: object  # nights of one shape, stacked on a leading axis
    index: onp.ndarray


@jax.tree_util.register_pytree_node_class
class _Nights:
    """The per-exposure log likelihood of every night for a companion in linear motion
    ``p + v (t - t̄)`` (design §2.1: the same data model for the inlier and the outlier
    term), on the normalization of ``NightPeak.loglike``.

    ``ell`` is the log likelihood at fixed error scales (one per block, the global scales
    of the inlier); ``own`` integrates each block's scale over ``ln s`` uniform on
    ``[-ln s_max, ln s_max]``, as the outlier's own scales (§2.2). Gains and closure
    offsets are marginalized at each scale by Woodbury (``_Surface``).
    """

    def __init__(self, data, summaries, diam, dtype, s_max=S_MAX):
        if len(data) != len(summaries):
            raise ValueError(
                f"{len(data)} datasets but {len(summaries)} night summaries."
            )
        self.dtype, self.s_max = dtype, float(s_max)
        self.diam = None if not diam else float(diam)
        self.n = len(data)
        self.blocks = summaries[0].blocks
        groups, sig = [], {}
        with run_in("float64"):
            consts = onp.zeros(self.n)
            for k, (d, s) in enumerate(zip(data, summaries)):
                surface = _Surface(d, self.s_max)
                if surface.names != tuple(s.blocks):
                    raise ValueError(
                        f"Night {k}: data blocks {surface.names} differ from the "
                        f"summary's {tuple(s.blocks)}."
                    )
                d64 = _centred(d, s.t_ref, "float64")
                zero = np.zeros(len(surface.names))
                null = _model(
                    np.asarray([0.0, 0.0, 0.0, 0.0, -np.inf]),
                    0.0,
                    None if not self.diam else np.asarray(self.diam),
                )
                # the data-only constant between `_Surface.fixed_score` and
                # `NightPeak.loglike`: the no-companion likelihood fixes it
                consts[k] = float(s.loglike_null) - float(
                    surface.fixed_score(null, d64, None, None, zero)
                )
                leaves, tree = jax.tree_util.tree_flatten(d64)
                key = (surface, tree, tuple(onp.shape(x) for x in leaves))
                if key not in sig:
                    sig[key] = _Group(surface, [], [])
                    groups.append(sig[key])
                sig[key].data.append(_centred(d, s.t_ref, dtype))
                sig[key].index.append(k)
        self.groups = [
            _Group(
                g.surface,
                jax.tree_util.tree_map(
                    lambda *xs: np.stack([np.asarray(x) for x in xs]), *g.data
                ),
                onp.asarray(g.index),
            )
            for g in groups
        ]
        self.const = np.asarray(consts, dtype)

    def tree_flatten(self):
        children = ([g.data for g in self.groups], self.const)
        aux = (
            tuple((g.surface, tuple(g.index.tolist())) for g in self.groups),
            self.dtype, self.s_max, self.diam, self.n, self.blocks,
        )  # fmt: skip
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        new = object.__new__(cls)
        groups, new.dtype, new.s_max, new.diam, new.n, new.blocks = aux
        new.groups = [
            _Group(s, d, onp.asarray(i, int))
            for (s, i), d in zip(groups, children[0])
        ]
        new.const = children[1]
        return new

    def _model_of(self, z):
        return _model(
            z, 0.0, None if not self.diam else np.asarray(self.diam, z.dtype)
        )

    def ell(self, pv, ln_f, ln_s):
        """``(K,)`` per-night log likelihood at states ``pv`` ``(K, 4)``, ``ln f`` and
        block scales ``ln_s``."""
        out = np.zeros(self.n, pv.dtype)
        for g in self.groups:
            z = np.concatenate(
                [pv[g.index], np.full((len(g.index), 1), ln_f, pv.dtype)], 1
            )

            def one(d, zk, g=g):
                return g.surface.fixed_score(
                    self._model_of(zk), d, None, None, ln_s
                )

            out = out.at[g.index].set(jax.vmap(one)(g.data, z))
        return out + self.const

    def own(self, k, z):
        """Night ``k``'s log likelihood with its own scales integrated, at states ``z``
        ``(m, 5)`` (``p, v, ln f``), minus the scale prior's normalization
        ``G ln(2 ln s_max)``."""
        for g in self.groups:
            if k in g.index:
                break
        else:
            raise IndexError(k)
        j = int(onp.flatnonzero(g.index == k)[0])
        d = jax.tree_util.tree_map(lambda x: x[j], g.data)
        # chunks of one shape, padded, so that every night shares one compilation
        z = onp.asarray(z, float)
        out = onp.empty(len(z))
        for i in range(0, len(z), _CHUNK):
            part = z[i : i + _CHUNK]
            pad = onp.concatenate(
                [part, onp.repeat(part[:1], _CHUNK - len(part), axis=0)]
            )
            res = _own_core(self.hashable, g.surface, d, np.asarray(pad))
            out[i : i + len(part)] = onp.asarray(res)[: len(part)]
        return (
            out
            + float(self.const[k])
            - len(g.surface.names) * math.log(2 * math.log(self.s_max))
        )


@functools.partial(jax.jit, static_argnums=(0, 1))
def _own_core(nights, surface, d, z):
    ones = np.ones(len(surface.names), z.dtype)

    def one(zk):
        model = nights._model_of(zk)
        chi2 = surface.chi2(model, d, None, ones)
        return surface.score(chi2, ones, d)

    return jax.vmap(one)(z)


# jit needs a hashable static stand-in for the nights object: the only things the
# compiled kernel reads from it are the primary diameter and how the model is built
class _Static:
    def __init__(self, diam):
        self.diam = diam

    def __hash__(self):
        return hash(self.diam)

    def __eq__(self, other):
        return isinstance(other, _Static) and self.diam == other.diam

    def _model_of(self, z):
        return _model(
            z, 0.0, None if not self.diam else np.asarray(self.diam, z.dtype)
        )


_Nights.hashable = property(lambda self: _Static(self.diam))


# ---------------------------------------------------------------------------
# Adaptive cubature (Genz & Malik 1980)
# ---------------------------------------------------------------------------


@functools.cache
def _genz_malik(n):
    """The degree-7 rule of Genz & Malik (1980) on ``[-1, 1]^n`` and its embedded
    degree-5 rule: the offsets, the two weight vectors (of volume 1), the point class
    and the indices of the points that give each axis's fourth difference."""
    l2, l3, l5 = math.sqrt(9 / 70), math.sqrt(9 / 10), math.sqrt(9 / 19)
    axes = onp.eye(n)
    offsets, kind = [onp.zeros(n)], [0]
    for k, lam in ((1, l2), (2, l3)):
        for sign in (-1.0, 1.0):
            offsets += [sign * lam * axes[i] for i in range(n)]
            kind += [k] * n
    for i in range(n):
        for j in range(i + 1, n):
            for si in (-1.0, 1.0):
                for sj in (-1.0, 1.0):
                    offsets.append(l3 * (si * axes[i] + sj * axes[j]))
                    kind.append(3)
    for corner in onp.array(onp.meshgrid(*[(-1.0, 1.0)] * n)).reshape(n, -1).T:
        offsets.append(l5 * corner)
        kind.append(4)
    kind = onp.array(kind)
    w7 = onp.array(
        [
            (12824 - 9120 * n + 400 * n**2) / 19683,
            980 / 6561,
            (1820 - 400 * n) / 19683,
            200 / 19683,
            6859 / 19683 / 2**n,
        ]
    )
    w5 = onp.array(
        [
            (729 - 950 * n + 50 * n**2) / 729,
            245 / 486,
            (265 - 100 * n) / 1458,
            25 / 729,
            0.0,
        ]
    )
    # fourth differences: point 1 + i and 1 + n + i are ±λ2 e_i, then ±λ3 e_i
    plus2, minus2 = 1 + onp.arange(n), 1 + n + onp.arange(n)
    plus3, minus3 = 1 + 2 * n + onp.arange(n), 1 + 3 * n + onp.arange(n)
    return (
        onp.array(offsets),
        w7[kind],
        w5[kind],
        (plus2, minus2, plus3, minus3, l2 / l3),
    )


def _cubature(log_f, lo, hi, mref, *, tol=_TOL, max_evals=_MAX_EVALS):
    """``∫ exp(log_f - mref)`` over the boxes ``lo`` to ``hi`` ``(m, n)`` by adaptive
    cubature with the rule of Genz & Malik (1980).

    ``log_f`` maps rows of points to the log integrand (``-inf`` outside the support).
    A cell carries the degree-7 value, the difference from the embedded degree-5 value as
    its error and the axis of the largest fourth difference. The cells that carry half of
    the total error bisect along that axis, until the errors total at most ``tol`` of
    the integral or the budget of evaluations is spent.

    Returns ``(integral, error, evaluations, converged)``.
    """
    lo, hi = onp.atleast_2d(lo), onp.atleast_2d(hi)
    n = lo.shape[1]
    points, w7, w5, (p2, m2, p3, m3, ratio) = _genz_malik(n)
    c, h = 0.5 * (lo + hi), 0.5 * (hi - lo)
    n_eval = 0

    def rule(c, h):
        nonlocal n_eval
        z = c[:, None, :] + h[:, None, :] * points[None]
        f = onp.exp(log_f(z.reshape(-1, n)) - mref).reshape(len(c), -1)
        n_eval += f.size
        vol = 2.0**n * onp.prod(h, axis=1)
        diff = onp.abs(
            (f[:, p2] + f[:, m2] - 2 * f[:, :1])
            - ratio**2 * (f[:, p3] + f[:, m3] - 2 * f[:, :1])
        )
        s = vol * (f @ w7)
        return s, onp.abs(s - vol * (f @ w5)), onp.argmax(diff, axis=1)

    s, e, axis = rule(c, h)
    cells = dict(c=c, h=h, s=s, e=e, a=axis)
    while True:
        total, err = cells["s"].sum(), cells["e"].sum()
        if err <= tol * abs(total):
            return float(total), float(err), n_eval, True
        if n_eval >= max_evals:
            return float(total), float(err), n_eval, False
        order = onp.argsort(-cells["e"])
        worst = onp.searchsorted(onp.cumsum(cells["e"][order]), 0.5 * err) + 1
        split = onp.zeros(len(order), bool)
        split[order[:worst]] = True
        sc, sh, sa = cells["c"][split], cells["h"][split], cells["a"][split]
        rows = onp.arange(len(sc))
        shift = onp.zeros_like(sc)
        shift[rows, sa] = 0.5 * sh[rows, sa]
        half = sh.copy()
        half[rows, sa] *= 0.5
        child_c = onp.concatenate([sc - shift, sc + shift])
        child_h = onp.concatenate([half, half])
        s, e, axis = rule(child_c, child_h)
        keep = {k: v[~split] for k, v in cells.items()}
        new = dict(c=child_c, h=child_h, s=s, e=e, a=axis)
        cells = {k: onp.concatenate([keep[k], new[k]]) for k in keep}


# ---------------------------------------------------------------------------
# The exact outlier floor
# ---------------------------------------------------------------------------


def _polar_box(peak, scale, table, velocity, ln_f_floor):
    """The cubature box of a peak in ``(ln|p|, φ_p[, ln|v|, φ_v], ln f)``: the peak ±
    ``_WINDOW`` standard deviations of the peak's own covariance (at the scales ``scale``),
    pushed through the Jacobian of the polar maps, cut to the prior's support."""
    cov = peak.covariance(scale)
    (p_lo, p_hi), (v_lo, v_hi) = table.bounds()
    centre, half = [], []
    pieces = [(0, p_lo, p_hi)] + ([(2, v_lo, v_hi)] if velocity else [])
    for i, lo, hi in pieces:
        x, y = peak.mean[i], peak.mean[i + 1]
        r2 = x * x + y * y
        jac = onp.array([[x, y], [-y, x]]) / r2
        sd = onp.sqrt(onp.diag(jac @ cov[i : i + 2, i : i + 2] @ jac.T))
        lr, ph = 0.5 * math.log(r2), math.atan2(y, x)
        centre += [lr, ph]
        half += [min(_WINDOW * sd[0], hi - lo), min(_WINDOW * sd[1], math.pi)]
        # a radial box is cut to the grid
        centre[-2] = float(onp.clip(lr, lo, hi))
    centre.append(float(peak.mean[4]))
    half.append(_WINDOW * math.sqrt(cov[4, 4]))
    centre, half = onp.array(centre), onp.array(half)
    lo, hi = centre - half, centre + half
    lo[-1], hi[-1] = max(lo[-1], ln_f_floor), min(hi[-1], 0.0)
    i = 0
    for _, a, b in pieces:  # the radial range is cut to the prior's support
        lo[i], hi[i] = max(lo[i], a), min(hi[i], b)
        i += 2
    return lo, hi


def _merge_boxes(lo, hi):
    """The disjoint boxes that cover ``lo`` to ``hi``: overlapping boxes are replaced by
    their bounding box until none overlap."""
    lo, hi = list(lo), list(hi)
    again = True
    while again:
        again = False
        for i in range(len(lo)):
            for j in range(i + 1, len(lo)):
                if onp.all(lo[i] < hi[j]) and onp.all(lo[j] < hi[i]):
                    lo[i], hi[i] = (
                        onp.minimum(lo[i], lo[j]),
                        onp.maximum(hi[i], hi[j]),
                    )
                    del lo[j], hi[j]
                    again = True
                    break
            if again:
                break
    return onp.array(lo), onp.array(hi)


def _night_log_f(nights, table, k, velocity):
    """``u -> ln[π_out(u) · L_k(own scales)]`` in the polar coordinates of ``_polar_box``
    (design §2.1–2.2); ``π_out`` is the table's density in ``(ln|p|, φ_p, ln|v|, φ_v,
    ln f)`` (``g / 2π × 1/|ln f_lo|``: in these coordinates the Cartesian ``|p|⁻²|v|⁻²``
    is the Jacobian), with the velocity integrated out for a night of one exposure."""
    ln_f_floor = table.ln_f_floor
    norm = -math.log(2 * math.pi) - math.log(-ln_f_floor)

    def log_f(u):
        out = onp.full(len(u), -onp.inf)
        flux = u[:, -1]
        ok = (flux >= ln_f_floor) & (flux <= 0.0)
        lp = u[:, 0]
        if velocity:
            lv, psi = u[:, 2], u[:, 3] - u[:, 1]
            log_pi = table.log_density(lp, lv, psi)
        else:
            log_pi = table.log_density_p(lp)
        ok &= onp.isfinite(log_pi)
        if not ok.any():
            return out
        v = u[ok]
        r = onp.exp(v[:, 0])
        cart = onp.zeros((len(v), 5))
        cart[:, 0], cart[:, 1] = r * onp.cos(v[:, 1]), r * onp.sin(v[:, 1])
        if velocity:
            q = onp.exp(v[:, 2])
            cart[:, 2], cart[:, 3] = q * onp.cos(v[:, 3]), q * onp.sin(v[:, 3])
        cart[:, 4] = v[:, -1]
        out[ok] = log_pi[ok] + norm + nights.own(k, cart)
        return out

    return log_f


def outlier_exact(
    data,
    summaries,
    table,
    *,
    primary_diam=None,
    s_max=S_MAX,
    tol=_TOL,
    max_evals=_MAX_EVALS,
):
    """Each night's log outlier density ``b_k``, exact (design §2).

    ``b_k = ln ∫ π_out(p, v, f) ∫ L_k(p, v, f; s) π(s) ds dp dv df``: the night's
    per-exposure likelihood for a companion in linear motion, with its own error scales
    ``s`` (``ln s`` uniform on ``[-ln s_max, ln s_max]`` per block, gains and offsets
    marginalized at each), against the joint prior predictive of the night state. A
    night with a single exposure has no velocity and takes ``π(p, f)``. The integral is
    adaptive Genz–Malik cubature over the boxes of the night's catalogued peaks; the
    catalogue is the one that makes the position leg's likelihood complete, so a mode
    outside it is outside both. A night without a peak takes its no-companion likelihood.

    Parameters
    ----------
    data : sequence of OIData
        One per night, with sample times (``mjd``).
    summaries : sequence of NightSummary
        The nights' peaks, in the same order.
    table : OutlierTable
        The prior predictive on a grid ([`outlier_table`][virgil.orbit_visibility.outlier_table]).
    primary_diam, s_max : optional
        The primary's diameter (mas) of ``positions`` and the bound of the scale prior.
    tol, max_evals : optional
        Relative tolerance of the integral (4 δ) and the budget of evaluations per night.

    Returns
    -------
    numpy.ndarray
        ``b_k``, on the normalization of ``NightPeak.loglike``.

    Warns
    -----
    UserWarning
        If the cubature of a night reached its budget before its tolerance.

    Source: Genz & Malik (1980).
    """
    with run_in("float64"):
        nights = _Nights(list(data), summaries, primary_diam, "float64", s_max)
        out, loose = [], []
        for k, s in enumerate(summaries):
            if not s.peaks:
                out.append(float(s.loglike_null))
                continue
            velocity = s.t_ss > 0
            boxes = []
            for p in s.peaks:
                ok = (p.n > 0) & p.rescalable
                scale = onp.where(
                    ok,
                    onp.clip(
                        onp.sqrt(p.chi2 / onp.where(ok, p.n, 1.0)),
                        1 / s_max,
                        s_max,
                    ),
                    1.0,
                )
                boxes.append(
                    _polar_box(p, scale, table, velocity, table.ln_f_floor)
                )
            # peaks closer than the angle's period share one representative
            # (unwrap each angle to within π of the best peak's)
            lo, hi = (
                onp.array([b[0] for b in boxes]),
                onp.array([b[1] for b in boxes]),
            )
            for col in (1, 3) if velocity else (1,):
                shift = (
                    onp.round(
                        (
                            (lo[:, col] + hi[:, col]) / 2
                            - (lo[0, col] + hi[0, col]) / 2
                        )
                        / (2 * math.pi)
                    )
                    * 2
                    * math.pi
                )
                lo[:, col] -= shift
                hi[:, col] -= shift
            keep = ~(hi <= lo).any(axis=1)
            if not keep.any():
                out.append(_LOG_TINY)
                continue
            lo, hi = lo[keep], hi[keep]
            # the scale of the integrand: its largest value at a catalogued peak
            # (a box's centre is its peak), where the likelihood is near its mode
            log_f = _night_log_f(nights, table, k, velocity)
            mref = float(onp.max(log_f(0.5 * (lo + hi))))
            lo, hi = _merge_boxes(lo, hi)
            for col in (1, 3) if velocity else (1,):
                # a merged box must not wrap an angle twice (the integrand is 2π periodic)
                hi[:, col] = onp.minimum(
                    hi[:, col], lo[:, col] + 2.0 * math.pi
                )
            if not onp.isfinite(mref):
                out.append(_LOG_TINY)
                continue
            mass, _, _, converged = _cubature(
                log_f, lo, hi, mref, tol=tol, max_evals=max_evals
            )
            loose += [k] * (not converged)
            out.append(mref + math.log(mass) if mass > 0 else _LOG_TINY)
    if loose:
        warnings.warn(
            f"The outlier integral of nights {loose} reached its budget of "
            f"{max_evals} evaluations before a relative error of {tol:g}.",
            UserWarning,
            stacklevel=2,
        )
    return onp.array(out)


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------


def swap_label(x):
    """The twin of chart vectors ``x`` under the label swap: the same core with
    ``TI′ → −TI′``.

    The companion sits at ``±p(t)`` according to which star the orbit describes (the
    brighter or the fainter): ``(u, V) → −(u, V)``, so ``z_u → i z_u`` and
    ``z_v → i z_v``, which leaves ``|z|`` and the rest of the chart alone; the
    likelihood summed over the label (``½ Σ_σ``) is therefore symmetric under it, and
    so is every invariant of [`report_invariants`][virgil.orbit_visibility.report_invariants].
    ``x`` is the flat chart of ``virgil.orbit_cells`` ``(w_P, y[4], ...)``.
    """
    x = np.asarray(x)
    y = x[..., 1:5]
    # multiplication by i on both complex coordinates (re, im) -> (-im, re)
    rot = np.stack([-y[..., 1], y[..., 0], -y[..., 3], y[..., 2]], axis=-1)
    return x.at[..., 1:5].set(rot)


def report_invariants(result):
    """A leg's particles in quantities unchanged by the covers of ``z`` and ``ζ`` and by
    the label swap.

    ``period``, ``a``, ``ecc``, ``mean_anomaly``, ``cos_inc``, ``Omega`` (the
    ascending node, a position angle folded to ``[0, 180)`` as ``virgil.orbit_cells``) and
    ``omega_mod_pi``: the argument of periastron mod 180°, since the swap shifts
    ``ω`` by 180° and nothing else (design §3.6). ``p_cut`` of the label is not
    reported: which twin a particle is in has no meaning.
    """
    inv = cells_mod.decode_particles(result)
    keys = ("period", "a", "ecc", "mean_anomaly", "cos_inc", "Omega")
    out = {k: inv[k] for k in keys}
    out["omega_mod_pi"] = onp.mod(onp.asarray(inv["omega"]), 180.0)
    return out


# ---------------------------------------------------------------------------
# The visibility leg
# ---------------------------------------------------------------------------


@jax.tree_util.register_pytree_node_class
class _VisSpace:
    """The chart, prior and sampler of the position leg with the visibility-level
    likelihood: the mixture of design §1.1 over the nights' per-exposure likelihood,
    the global scales live and the outlier floor of the position leg.

    A pytree (the position leg's space and the nights' data are traced), so one
    compilation serves every cell of a system.
    """

    def __init__(self, base, nights):
        self.base, self.nights = base, nights

    def tree_flatten(self):
        return (self.base, self.nights), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(*children)

    @property
    def d(self):
        return self.base.d

    @property
    def dtype(self):
        return self.base.dtype

    def logprior(self, x):
        return self.base.logprior(x)

    def decode(self, x):
        return self.base.decode(x)

    def sample(self, key, n):
        return self.base.sample(key, n)

    def terms(self, x):
        """``(ln L_vis, expected inlier nights)`` at chart vector ``x``."""
        base = self.base
        c = base.decode(x)
        pv = base.states(c["period"], oc.z_to_ti(c["z"]), c["zeta"])
        ln_s = math.log(S_MAX) * (2.0 * jax.nn.sigmoid(x[9:]) - 1.0)
        eps = np.asarray(base.cat.eps, base.dtype)
        log_w = np.asarray(base.cat.log_w_eps, base.dtype)
        per_sign, resp = [], []
        for sign in (1.0, -1.0):
            ell = self.nights.ell(sign * pv, c["ln_f"], ln_s)
            inside = np.log1p(-eps)[:, None] + ell[None]
            night = np.logaddexp(inside, np.log(eps)[:, None] + base.cat.log_b)
            per_sign.append(night.sum(axis=1))
            resp.append(np.exp(inside - night).sum(axis=1))
        total = np.stack(per_sign)
        out = logsumexp(total + log_w) - np.log(2.0)
        weight = np.exp(total + log_w - logsumexp(total + log_w))
        return out, np.sum(weight * np.stack(resp))

    def loglike(self, x):
        return self.terms(x)[0]


def _vis_target(x, space):
    """The label-summed log target at visibility level (log prior + log likelihood)."""
    return space.loglike(x) + space.logprior(x)


def _vis_inliers(x, space):
    return space.terms(x)[1]


def _vis_objective(x, space):
    """The search objective at visibility level: the target with the chi-4 radial
    prior of ``y_z``, as [`orbit_cells._objective`][virgil.orbit_cells]."""
    return _vis_target(x, space) + 3.0 * np.log(np.linalg.norm(x[_Z]))


@functools.partial(jax.jit, static_argnames=("n_iter",))
def _vis_refine(space, x, n_iter):
    """Newton on the visibility objective from ``x``: the result, minus its Hessian,
    the objective and the expected inlier count."""
    x, _, _ = cells_mod._lm(_vis_objective, x, (space,), n_iter)
    return (
        x,
        -jax.hessian(_vis_objective)(x, space),
        _vis_objective(x, space),
        _vis_inliers(x, space),
    )


@dataclasses.dataclass
class VisibilityCell:
    """The visibility-level fit of one cell (refined modes, importance sampling, draws).

    Attributes
    ----------
    cell : Cell
    log_z, log_z_error : float
        ``ln Ẑ_N^vis`` of the cell's normalized prior (the cell's mass is applied in
        [`CellsFit`][virgil.orbit_visibility.CellsFit]) and its delta-method error.
    modes : list of Mode
        The refined visibility-level modes.
    draws : array or None
        Resample-move posterior draws (chart vectors); ``None`` until made.
    at_floor, undersampled, unreliable, nonfinite : bool
        As for [`CellFit`][virgil.orbit_cells.CellFit]: ``cell-at-floor`` and
        ``nonfinite`` fail the cell; ``cell-undersampled`` (ESS below ``ESS_TARGET``)
        is information; ``unreliable`` is ESS below ``ESS_FAIL``.
    space : _VisSpace
        For [`report_invariants`][virgil.orbit_visibility.report_invariants].
    evidence : Evidence
    move : Move or None
    """

    cell: Cell
    log_z: float
    log_z_error: float
    modes: list
    draws: object
    at_floor: bool
    undersampled: bool
    unreliable: bool
    nonfinite: bool
    space: object = dataclasses.field(repr=False, default=None)
    evidence: object = dataclasses.field(repr=False, default=None)
    move: object = dataclasses.field(repr=False, default=None)

    @property
    def failed(self):
        """The cell failed on its own: ``at_floor`` or ``nonfinite``."""
        return bool(self.at_floor or self.nonfinite)

    @property
    def log_z_upper(self):
        """The larger of ``ln Ẑ`` and the best mode's Laplace ``ln Z``."""
        return max(
            [self.log_z] + [laplace_log_evidence(m) for m in self.modes]
        )


def visibility_leg(fit, nights, key, *, n_is=N_IS, posterior_draws=True):
    """Visibility-level evidence and draws of one cell from its position-level modes.

    Each mode of ``fit`` is refined by Newton on the per-exposure visibility target
    (gains and offsets marginalized, global scales live), a defensive mixture is built
    on the refined modes and their symmetry images, ``ln Z`` is its importance-sampling
    estimate and the draws an HMC resample-move of the weighted draws
    (`mode_search`; design §6.2). The sentinel and flags are those of the position leg.

    Parameters
    ----------
    fit : CellFit
        The cell's position leg ([`position_leg`][virgil.orbit_cells.position_leg]).
    nights : _Nights
        The nights' visibility likelihood.
    key : jax.random.PRNGKey
    n_is : int, optional
        Importance draws.
    posterior_draws : bool, optional
        Make the resample-move draws (not for a cell that failed).

    Returns
    -------
    VisibilityCell

    Source: Hesterberg (1995), Gilks & Berzuini (2001), Hoffman & Gelman (2014).
    """
    k_is, k_move = jax.random.split(key)
    base = fit.space
    with run_in(onp.dtype(base.dtype).name):
        space = _VisSpace(base, nights)
        # each mode refined by Newton on the visibility objective, with the
        # saddle-free covariance of its Hessian, as ``search_cell``
        refined = []
        for m in fit.modes:
            x, h, t, n = (
                onp.asarray(a)
                for a in _vis_refine(
                    space, np.asarray(m.x, space.dtype), cells_mod.N_NEWTON
                )
            )
            h = 0.5 * (h + h.T)
            if (
                onp.isfinite(x).all()
                and onp.isfinite(h).all()
                and onp.isfinite(t)
            ):
                e, vec = onp.linalg.eigh(h)
                e = onp.maximum(
                    onp.abs(e),
                    len(e) * onp.finfo(h.dtype).eps * onp.abs(e).max(),
                )
                refined.append(Mode(x, float(t), (vec / e) @ vec.T, float(n)))
        modes = distinct_modes(refined)
        mixture = DefensiveMixture(
            modes, cells_mod._log_prior,
            cells_mod._prior_sampler(space.d, space.dtype),
            images=cells_mod._symmetry_images(space.d), dtype=space.dtype,
            dim=space.d,
        )  # fmt: skip
        n_k = len(base.cat.log_b)
        # ln Z of the all-outlier target: Σ b_k + ln E[ε^K] = ln B(a+K, b) - ln B(a, b)
        a, b = EPS_PRIOR
        floor = float(onp.sum(base.cat.log_b)) + (
            cells_mod._ln_beta(a + n_k, b) - cells_mod._ln_beta(a, b)
        )
        ev = importance_evidence(
            _vis_target, space, mixture, k_is, n_is, inliers_fn=_vis_inliers,
            log_floor=floor,
        )  # fmt: skip
        out = VisibilityCell(
            fit.cell, ev.log_z, ev.log_z_error, modes, None, ev.at_floor,
            ev.undersampled, bool(ev.ess < ESS_FAIL), bool(ev.n_nonfinite > 0),
            space, ev,
        )  # fmt: skip
        if posterior_draws and not out.failed:
            cov = max(modes, key=laplace_log_evidence).cov if modes else None
            out.move = resample_move(_vis_target, space, ev, k_move, cov=cov)
            out.draws = out.move.draws
            if out.move.stuck:
                warnings.warn(
                    f"cell {fit.cell}: HMC accepted {out.move.acceptance:.3g} of "
                    "moves (< 1 per particle): the draws are not rejuvenated",
                    RuntimeWarning,
                    stacklevel=2,
                )
    return out


# ---------------------------------------------------------------------------
# The whole path
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class CellsFit:
    """The visibility-level posterior over the cells.

    Attributes
    ----------
    positions : CellsResult
        The position leg over every cell ([`position_cells`][virgil.orbit_cells.position_cells]).
    legs : list of VisibilityCell
        The visibility leg of each kept cell (``None`` for a cell the position leg
        dropped below ``δ / N_cells``, or when the position leg failed).
    log_z : numpy.ndarray
        ``ln Ẑ_N^vis`` per cell (``-inf`` where dropped).
    p : numpy.ndarray
        ``p(N) ∝ mass_N Ẑ_N^vis`` over the kept cells, summing to one; all ``nan`` when
        any cell failed (never renormalized over the others).
    failed : numpy.ndarray
        Cells that failed: the position leg's, and the visibility leg's ``at_floor`` or
        ``nonfinite``, or unreliable and within ``W`` of the best.
    log_outlier : numpy.ndarray
        The exact ``b_k``.
    summaries : tuple of NightSummary
        The nights' peaks kept by the rule of design §5.
    """

    positions: object
    legs: list
    log_z: onp.ndarray
    p: onp.ndarray
    failed: onp.ndarray
    log_outlier: onp.ndarray
    summaries: tuple


def visibility_cells(positions, nights, key, *, delta=DELTA, n_is=N_IS):
    """The visibility leg in every cell that leg 1 keeps (``p_pos ≥ δ / N_cells``).

    Parameters
    ----------
    positions : CellsResult
        From [`position_cells`][virgil.orbit_cells.position_cells]. If its leg failed,
        no cell is refined and ``p`` is undefined.
    nights : _Nights
    key : jax.random.PRNGKey
    delta : float, optional
        Reporting precision.
    n_is : int, optional
        Importance draws per cell.

    Returns
    -------
    legs, log_z, p, failed : list, numpy.ndarray, numpy.ndarray, numpy.ndarray
    """
    n = len(positions.cells)
    legs = [None] * n
    failed = onp.array(positions.failed, bool)
    if not failed.any():
        for i, k in enumerate(jax.random.split(key, n)):
            if positions.p_pos[i] >= delta / n:
                legs[i] = visibility_leg(
                    positions.fits[i], nights, k, n_is=n_is
                )
    log_z = onp.array([-onp.inf if v is None else v.log_z for v in legs])
    log_p = onp.array([c.log_mass for c in positions.cells]) + log_z
    upper = onp.array(
        [
            -onp.inf if v is None else c.log_mass + v.log_z_upper
            for c, v in zip(positions.cells, legs)
        ]
    )
    matters = upper >= upper.max() - W
    for i, v in enumerate(legs):
        if v is not None:
            failed[i] |= v.failed or (v.unreliable and bool(matters[i]))
    if failed.any() or not onp.isfinite(log_p).any():
        p = onp.full(n, onp.nan)
    else:
        p = onp.exp(log_p - onp_logsumexp(log_p))
    return legs, log_z, p, failed


def fit_cells(
    data,
    positions,
    key,
    *,
    p_range=None,
    a_range=A_RANGE,
    e_max=E_MAX,
    flux_floor=FLUX_FLOOR,
    s_max=S_MAX,
    n_is=N_IS,
    prune=False,
    dtype="float64",
    outlier_samples=2**20,
):
    """The new orbit path on prepared data: night summaries, domain, exact outlier floor,
    position leg, visibility leg, ``p(N)``.

    Parameters
    ----------
    data : Epochs or sequence of OIData
        The nights, with sample times.
    positions : EpochPositions
        Their peak catalogues and the primary diameter.
    key : jax.random.PRNGKey
    p_range : (float, float), optional
        Override the period domain ``[P_cut, P_max]``.
    a_range, e_max, flux_floor, s_max : optional
        Declared prior bounds (design §10).
    n_is : int, optional
        Importance draws per cell, in both legs.
    prune : bool, optional
        As for [`position_cells`][virgil.orbit_cells.position_cells]. The visibility
        leg runs only in the cells the position leg keeps either way.
    dtype : str, optional
        Precision of the legs.
    outlier_samples : int, optional
        Draws of the tabulated prior predictive (default 2²⁰).

    Returns
    -------
    CellsFit
    """
    sets = list(data.data) if isinstance(data, Epochs) else list(data)
    summaries = night_summaries(sets, positions)
    domain = period_domain(
        summaries,
        a_range=a_range,
        p_range=p_range,
    )
    prior = outlier_prior(
        domain, a_range, e_max, flux_floor, n=outlier_samples
    )
    table = outlier_table(prior)
    diam = positions.primary_diam
    b = outlier_exact(sets, summaries, table, primary_diam=diam, s_max=s_max)
    summaries = keep_peaks(summaries, b)
    k_pos, k_vis = jax.random.split(key)
    pos = position_cells(
        summaries,
        domain,
        k_pos,
        log_outlier=b,
        n_is=n_is,
        prune=prune,
        a_range=a_range,
        e_max=e_max,
        flux_floor=flux_floor,
        dtype=dtype,
    )
    with run_in(dtype):
        nights = _Nights(sets, summaries, diam, dtype, s_max)
    legs, log_z, p, failed = visibility_cells(pos, nights, k_vis, n_is=n_is)
    return CellsFit(pos, legs, log_z, p, failed, b, summaries)
