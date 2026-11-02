"""Cells, period domain and the position-level search of the alias-resolved orbit fit.

!!! warning "Provisional"
    May change in any 0.4.x release without a deprecation cycle.

The period prior is log-uniform on ``[P_cut, P_max]``
([`period_domain`][virgil.orbit_cells.period_domain]). Two anchor nights set
``T``, and ``T / P = N + ρ`` splits the domain into cells, an exact partition
for any edges ([`make_cells`][virgil.orbit_cells.make_cells]). Each cell is
searched for the modes of the night-summary likelihood of
[`virgil.night_summaries`][virgil.night_summaries], in the ``(ln P, z, ζ)``
coordinates of ``virgil._orbit_coordinates``: variable projection of the
Thiele–Innes constants on minimal sets of nights, concentration steps and
Newton refinement. A defensive mixture on the modes (with the images of the
exact symmetries of the target) gives the evidence by importance sampling, and
an HMC resample-move gives posterior draws
([`position_leg`][virgil.orbit_cells.position_leg]). The generic parts are in
[`virgil.mode_search`][virgil.mode_search].
[`position_cells`][virgil.orbit_cells.position_cells] runs every cell and gives
``p(N) ∝ prior mass × Ẑ_N``, with an opt-in pruning that records the dropped
mass; a cell whose search missed the dominant mode (``cell-at-floor``) is a
failure, never an evidence. Design: ``design/orbit_alias_cycles.md`` §4–§7.

Source: Fischler & Bolles (1981), Rousseeuw (1984), Rousseeuw & Van Driessen
(1999), Golub & Pereyra (1973), Hesterberg (1995), Gilks & Berzuini (2001),
[references](../references.md); the cells, outlier floor and search are
derived in the design note.
"""

import dataclasses
import functools
import itertools
import math
import time

import jax
import jax.numpy as np
import numpy as onp
from jax.scipy.special import logsumexp
from scipy import stats
from scipy.special import logsumexp as onp_logsumexp
from scipy.stats import qmc

from . import _orbit_coordinates as oc
from ._precision import run_in
from .aliases import _P_FLOOR, alias_bands, period_cut, position_profile
from .mode_search import (
    DefensiveMixture,
    M_MIN,
    W,
    N_IS,
    Mode,
    distinct_modes,
    importance_evidence,
    resample_move,
)
from .night_summaries import DELTA
from .orbits import KeplerOrbit, PositionData, _unit_orbit

__all__ = [
    "Cell",
    "CellFit",
    "CellsResult",
    "OutlierPrior",
    "PeriodDomain",
    "choose_anchors",
    "decode_particles",
    "fit_jitter",
    "make_cells",
    "outlier_log_density",
    "outlier_prior",
    "period_domain",
    "position_cells",
    "position_leg",
    "search_cell",
    "search_plan",
]

__provisional__ = ("*",)

# Declared prior bounds, inherited from the engine (design §10).
S_MAX = 10.0  # inlier scale: ln s uniform on ±ln S_MAX (covers χ²/N <= 100)
A_RANGE = (1.0, 30.0)  # mas
E_MAX = 0.9
FLUX_FLOOR = 1e-4  # engine's flux floor (arcs._half_position)
_N_PV = 4  # (p, v) of the linear Gaussian block; ln f is the fifth dimension

# The search (design §6.1, §10; the importance-sampling constants are in
# virgil.mode_search).
K_COH = (
    4  # ln P grid points per cycle of a set's span (> 2 brackets a minimum)
)
ZETA_E = (0.0, 0.3, 0.6)  # ζ starts: eccentricities, 8 phases each (17 points)
ZETA_PHASES = 8  # anchor mean anomalies M_a per eccentricity, from 0 to 2π
N_SET_MAX = 2000  # enumerate every set below this, a random subset above
EPS_HI = 0.3  # declared outlier fraction of the RANSAC count (design §10)
N_C = 10  # C-step cap (labels repeat in 2-3 steps on the toys)
N_VARPRO = 12  # damped Newton steps of a set's (ln P, ζ) refinement (budget)
N_NEWTON = 30  # damped Newton steps per C-step and for the refinement (budget)
TOL_GAIN = (
    1e-5  # Newton stops at a predicted gain below this (nats: DELTA / 100)
)
W_PRUNE = 2.0 * W  # nats below the best Laplace ln Z that drop a candidate
N_PRE = 4  # Newton steps of the first C-step, before the candidates merge
CHUNK_C = 256  # candidates per batched C-step call (budget)
CHUNK_VP = 1024  # candidates per batched set refinement call (budget)
R0 = math.sqrt(3.0)  # radius of y_z at a mode: the mode of its chi-4 prior


# ---------------------------------------------------------------------------
# Period domain, anchors, cells
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class PeriodDomain:
    """The period prior ``log-uniform [p_cut, p_max]`` (days), declared.

    ``overridden`` is true when ``p_range`` replaced the derived limits.
    """

    p_cut: float
    p_max: float
    overridden: bool


def _sigma_p(summary):
    """Position error (mas) of a night's best peak; ``inf`` if it has none."""
    if not summary.peaks:
        return onp.inf
    return float(onp.sqrt(onp.trace(summary.peaks[0].cov[:2, :2]) / 2))


def period_domain(
    summaries,
    *,
    night_length,
    resolution,
    a_range=A_RANGE,
    p_range=None,
    motion_fraction=0.5,
    p_floor=_P_FLOOR,
):
    """The declared period domain ``[P_cut, P_max]``.

    ``P_cut = max(2 Δt_min, P_night)``
    ([`period_cut`][virgil.aliases.period_cut]) and
    ``P_max = 2π a_hi · span / σ_min``, with ``σ_min`` the smallest night position
    error: beyond it the motion over the span is below that error.

    Parameters
    ----------
    summaries : sequence of NightSummary
        One per night.
    night_length, resolution : float
        The longest night (days) and λ/B_max (mas), as for
        [`period_window`][virgil.aliases.period_window].
    a_range : (float, float)
        Prior range of the semimajor axis (mas); ``a_hi`` is the field.
    p_range : (float, float), optional
        Override ``(P_cut, P_max)``.
    motion_fraction, p_floor : float, optional
        As for [`period_window`][virgil.aliases.period_window].

    Returns
    -------
    PeriodDomain
    """
    if p_range is not None:
        lo, hi = map(float, p_range)
        overridden = True
    else:
        times = onp.array([s.t_ref for s in summaries])
        sigma = onp.array([_sigma_p(s) for s in summaries])
        if onp.isfinite(sigma).sum() < 2:
            raise ValueError(
                "Need two nights with a peak, or pass p_range, to set the "
                "period domain."
            )
        lo = period_cut(
            times,
            a_range[1],
            night_length,
            resolution=resolution,
            motion_fraction=motion_fraction,
            p_floor=p_floor,
        )
        hi = 2 * math.pi * a_range[1] * float(onp.ptp(times)) / sigma.min()
        overridden = False
    if not 0 < lo < hi:
        raise ValueError(f"The period domain [{lo:.4g}, {hi:.4g}] d is empty.")
    return PeriodDomain(lo, hi, overridden)


def choose_anchors(summaries):
    """Indices ``(first, last)`` of the two anchor nights.

    The night with the smallest best-peak position error among those in the
    first quarter of the span, and likewise in the last; an empty quarter
    falls back to the first (last) night with a peak, and if both choices are
    one night, to the earliest and latest nights with a peak. They always differ.
    """
    t = onp.array([s.t_ref for s in summaries])
    sigma = onp.array([_sigma_p(s) for s in summaries])
    ok = onp.isfinite(sigma)
    if ok.sum() < 2 or not onp.ptp(t[ok]) > 0:
        raise ValueError(
            "Anchors need two nights with a peak at different times."
        )
    quarter = onp.ptp(t) / 4

    def best(mask, default):
        pool = onp.flatnonzero(mask & onp.isfinite(sigma))
        return int(pool[onp.argmin(sigma[pool])]) if pool.size else default

    first = best(
        t <= t.min() + quarter, int(onp.argmin(onp.where(ok, t, onp.inf)))
    )
    last = best(
        t >= t.max() - quarter, int(onp.argmax(onp.where(ok, t, -onp.inf)))
    )
    if first == last:  # one night is best in both quarters: span the peaks
        first = int(onp.argmin(onp.where(ok, t, onp.inf)))
        last = int(onp.argmax(onp.where(ok, t, -onp.inf)))
    return first, last


@dataclasses.dataclass(frozen=True)
class Cell:
    """One cell: ``p_lo <= P <= p_hi`` (days), ``N = round(T / P)``.

    ``log_mass`` is the cell's share of the period prior, ``ln[ln(p_hi / p_lo) /
    ln(P_max / P_cut)]`` (the period prior is log-uniform). Cell ``N = 0`` is the long-period band (``P > T``, no
    cycle count), which ends at ``P_max``.
    """

    n: int
    p_lo: float
    p_hi: float
    log_mass: float


def make_cells(t_a, t_b, domain, p_edges=None):
    """The cells of ``domain`` for anchors at ``t_a`` and ``t_b`` (days).

    The default edges are those of [`alias_bands`][virgil.aliases.alias_bands]
    (``T / P`` half-integers). ``p_edges`` replace them: one period between
    each pair of consecutive cells, increasing with ``N`` decreasing.

    Returns
    -------
    list of Cell
        Ordered by ``N``; their masses sum to one.
    """
    bands = alias_bands([t_a, t_b], (domain.p_cut, domain.p_max))
    default = onp.array([b.p_lo for b in bands[:-1]])
    edges = default if p_edges is None else onp.asarray(p_edges, float)
    if edges.shape != default.shape:
        raise ValueError(f"Need {len(default)} edges, got {edges.shape}.")
    # bands run from long to short period: boundaries decrease with N
    bounds = onp.r_[domain.p_max, edges, domain.p_cut]
    if not onp.all(onp.diff(bounds) < 0):
        raise ValueError("The edges must lie in the domain, in order.")
    total = math.log(domain.p_max / domain.p_cut)
    return [
        Cell(b.n, float(bounds[i + 1]), float(bounds[i]),
             math.log(math.log(bounds[i] / bounds[i + 1]) / total))
        for i, b in enumerate(bands)
    ]  # fmt: skip


# ---------------------------------------------------------------------------
# The outlier floor: the prior predictive of a night's (p, v, ln f)
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class OutlierPrior:
    """The tabulated joint prior predictive of a night state (design §2.2).

    ``samples`` are ``(ln|p|, ln|v|, ψ)`` of prior draws (``ψ`` the angle
    between ``v`` and ``p``), ``bandwidth`` the Scott bandwidths and
    ``ln_f_floor`` the lower end of the flux prior.
    """

    samples: onp.ndarray
    bandwidth: onp.ndarray
    ln_f_floor: float

    def log_density(self, mean, velocity=True):
        """``ln π(p, v) π(f)`` at ``mean = (dra, ddec, vra, vdec, ln f)``,
        a density in ``(p, v, ln f)``; ``-inf`` outside the flux range. With
        ``velocity=False`` the velocity is integrated out: the density in
        ``(p, ln f)`` (``mean[2:4]`` unused)."""
        mean = onp.atleast_2d(onp.asarray(mean, float))
        p = mean[:, :2]
        v = mean[:, 2:4] if velocity else onp.ones_like(p)  # unused if not
        lp, lv = onp.log(onp.hypot(*p.T)), onp.log(onp.hypot(*v.T))
        psi = onp.arctan2(v[:, 1], v[:, 0]) - onp.arctan2(p[:, 1], p[:, 0])
        query = onp.stack([lp, lv, psi], axis=1)
        if not velocity:  # the flat kernels ignore these; keep them finite
            query[:, 1:] = 0.0
        bandwidth = self.bandwidth if velocity else self.bandwidth.copy()
        if not velocity:
            bandwidth[1:] = (
                onp.inf
            )  # a flat kernel in ln|v| and ψ: integrates out
        norm = onp.sum(
            onp.log(
                bandwidth[: 1 if not velocity else 3] * math.sqrt(2 * math.pi)
            )
        )
        log_g = (
            onp.asarray(
                _kde_core(
                    np.asarray(self.samples, np.float32),
                    np.asarray(bandwidth, np.float32),
                    np.asarray(query, np.float32),
                ),
                float,
            )
            - norm
        )  # Gaussian KDE of the wrapped-angle samples
        log_pv = log_g - math.log(2 * math.pi) - 2 * lp
        if velocity:
            log_pv = log_pv - 2 * lv
        in_f = (mean[:, 4] >= self.ln_f_floor) & (mean[:, 4] <= 0.0)
        log_f = onp.where(in_f, -math.log(-self.ln_f_floor), -onp.inf)
        return log_pv + log_f


@jax.jit
def _kde_core(samples, bandwidth, query):
    def one(q):
        r = (q - samples) / bandwidth
        wrapped = (q[2] - samples[:, 2] + np.pi) % (2 * np.pi) - np.pi
        r = r.at[:, 2].set(wrapped / bandwidth[2])
        return logsumexp(-0.5 * np.sum(r**2, axis=1)) - np.log(
            samples.shape[0]
        )

    return jax.lax.map(one, query)


def outlier_prior(
    domain,
    a_range=A_RANGE,
    e_max=E_MAX,
    flux_floor=FLUX_FLOOR,
    n=2**20,
    seed=0,
):
    """Tabulate the prior predictive of ``(p, v)`` at a random epoch.

    ``n`` scrambled-Sobol draws of the orbit prior (log-uniform ``P`` on the
    domain and ``a``, uniform ``e`` and ``M``, isotropic orientation) are
    mapped to ``(ln|p|, ln|v|, ψ)``; the density is a Gaussian KDE with Scott's
    bandwidth, smooth for the night cubature.

    Returns
    -------
    OutlierPrior
    """
    u = qmc.Sobol(7, scramble=True, seed=seed).random(n)
    lo, hi = onp.log(domain.p_cut), onp.log(domain.p_max)
    chunks = []
    for part in onp.array_split(u, max(1, n // 2**16)):
        period = onp.exp(lo + (hi - lo) * part[:, 0])
        a = onp.exp(math.log(a_range[0])
                    + math.log(a_range[1] / a_range[0]) * part[:, 1])  # fmt: skip
        ecc = e_max * part[:, 2]
        orbit = KeplerOrbit(
            period, 0.0 * period, ecc,
            onp.degrees(onp.arccos(2 * part[:, 3] - 1)),
            360 * part[:, 4], 360 * part[:, 5], a,
        )  # fmt: skip
        dt = part[:, 6] * period  # a uniformly random epoch
        pos, vel = jax.jvp(
            lambda t: orbit._relative(t)[:2],
            (np.asarray(dt),),
            (np.ones_like(np.asarray(dt)),),
        )
        pos, vel = onp.asarray(pos, float), onp.asarray(vel, float)
        psi = onp.arctan2(vel[1], vel[0]) - onp.arctan2(pos[1], pos[0])
        chunks.append(
            onp.stack(
                [
                    onp.log(onp.hypot(*pos)),
                    onp.log(onp.hypot(*vel)),
                    (psi + onp.pi) % (2 * onp.pi) - onp.pi,
                ],
                axis=1,
            )
        )
    samples = onp.concatenate(chunks)
    return OutlierPrior(
        samples,
        samples.std(0) * n ** (-1.0 / 7.0),  # Scott, three dimensions
        float(onp.log(flux_floor)),
    )


def outlier_log_density(summaries, prior, s_max=S_MAX):
    """Each night's log outlier density ``b_k`` (Laplace form, design §2.3).

    ``b_k = ln Σ_peaks L_j (2π)^{5/2} |Σ_j|^{1/2} π_out(m_j)`` with each peak's
    own error scales integrated (log-uniform on ``[1/s_max, s_max]``, by
    Laplace; fixed at 1 for a peak that is not ``rescalable``). On a
    single-exposure night the velocity is integrated out of ``π_out``, leaving
    the ``(p, ln f)`` Laplace form. A night without a peak gets its
    no-companion likelihood.

    Parameters
    ----------
    summaries : sequence of NightSummary
    prior : OutlierPrior
    s_max : float, optional
        Bound of the scale prior (default 10).

    Returns
    -------
    numpy.ndarray
        ``b_k``, on the normalization of ``NightPeak.loglike``.
    """
    out = []
    for s in summaries:
        terms = []
        for p in s.peaks:
            # a scale does not factor out of gains or closure offsets: unit scale
            ok = (p.n > 0) & p.rescalable
            hat = onp.clip(
                onp.sqrt(p.chi2 / onp.where(ok, p.n, 1.0)), 1 / s_max, s_max
            )
            hat = onp.where(ok, hat, 1.0)
            occam = onp.sum(
                onp.where(
                    ok,
                    0.5 * onp.log(onp.pi / onp.where(ok, p.n, 1.0))
                    - onp.log(2 * onp.log(s_max)),
                    0.0,
                )
            )
            cov = p.covariance(hat)
            # a single exposure leaves the velocity unconstrained: integrate
            # it against its prior instead of taking its (huge) Laplace width
            dims = [0, 1, 4] if s.t_ss == 0 else [0, 1, 2, 3, 4]
            _, logdet = onp.linalg.slogdet(cov[onp.ix_(dims, dims)])
            terms.append(
                p.loglike
                + p.log_scale(hat)
                + occam
                + 0.5 * len(dims) * math.log(2 * math.pi)
                + 0.5 * logdet
                + float(prior.log_density(p.mean, velocity=len(dims) == 5)[0])
            )
        out.append(float(onp_logsumexp(terms)) if terms else s.loglike_null)
    return onp.array(out)


# ---------------------------------------------------------------------------
# The position-level likelihood
# ---------------------------------------------------------------------------


@jax.tree_util.register_pytree_node_class
class _Catalogue:
    """The nights' peaks as padded arrays (a pytree of constants)."""

    def __init__(self, summaries, log_outlier):
        blocks = {s.blocks for s in summaries}
        if len(blocks) != 1:
            raise ValueError(f"The nights' error blocks differ: {blocks}.")
        self.blocks = blocks.pop()
        k, g = len(summaries), len(self.blocks)
        j = max(1, max(len(s.peaks) for s in summaries))
        self.mean = onp.zeros((k, j, 5))
        self.hess = onp.tile(onp.eye(5), (k, j, g, 1, 1))
        self.n = onp.ones((k, j, g))
        self.chi2 = onp.zeros((k, j, g))
        self.loglike = onp.zeros((k, j))
        self.valid = onp.zeros((k, j), bool)
        for i, s in enumerate(summaries):
            for q, p in enumerate(s.peaks):
                self.mean[i, q], self.hess[i, q] = p.mean, p.hessian
                self.n[i, q], self.chi2[i, q] = p.n, p.chi2
                self.loglike[i, q], self.valid[i, q] = p.loglike, True
        self.rescalable = all(p.rescalable for s in summaries for p in s.peaks)
        self.t_ss = onp.array([s.t_ss for s in summaries])
        self.n_frames = onp.array([s.n_frames for s in summaries], float)
        self.t_ref = onp.array([s.t_ref for s in summaries])
        self.log_b = onp.broadcast_to(
            onp.asarray(log_outlier, float), (k,)
        ).copy()
        self.n_eps = (k + 2) // 2  # ⌈(K + 1) / 2⌉: exact for degree K

    _ARRAYS = (
        "mean", "hess", "n", "chi2", "loglike", "valid", "t_ss", "n_frames",
        "t_ref", "log_b",
    )  # fmt: skip

    def tree_flatten(self):
        return tuple(getattr(self, k) for k in self._ARRAYS), (
            self.blocks,
            self.rescalable,
            self.n_eps,
        )

    @classmethod
    def tree_unflatten(cls, aux, children):
        new = object.__new__(cls)
        new.__dict__.update(zip(cls._ARRAYS, children))
        new.blocks, new.rescalable, new.n_eps = aux
        return new

    def cast(self, dtype):
        """The same arrays as ``dtype`` JAX arrays."""
        new = object.__new__(_Catalogue)
        new.__dict__ = {
            k: np.asarray(v, dtype if v.dtype.kind == "f" else v.dtype)
            if isinstance(v, onp.ndarray)
            else v
            for k, v in self.__dict__.items()
        }
        return new


def _peak_precision(cat, ln_s, jitter, dtype):
    """``(Σ_data, Σ_total, Σ_total⁻¹)`` of every peak at scales ``exp(ln_s)``,
    with an independent per-frame ``jitter`` (mas) added in quadrature."""
    prec = np.einsum("g,kjgab->kjab", np.exp(-2 * ln_s), cat.hess)
    eps = np.finfo(dtype).eps
    # An unconstrained direction gets a huge variance, as NightPeak.covariance
    reg = eps * np.trace(prec, axis1=-2, axis2=-1)[..., None, None]
    cov = np.linalg.inv(prec + reg * np.eye(5, dtype=dtype))
    add = np.stack(
        [
            jitter**2 / cat.n_frames,
            jitter**2 / cat.n_frames,
            jitter**2 / np.maximum(cat.t_ss, eps),
            jitter**2 / np.maximum(cat.t_ss, eps),
            np.zeros_like(cat.t_ss),
        ],
        axis=-1,
    )
    total = cov + add[:, None, :, None] * np.eye(5, dtype=dtype)
    return cov, total, np.linalg.inv(total)


def _peak_terms(cat, ln_s, jitter, state, dtype):
    """``ln L_j(s) + ½ ln(|Σ_data| / |Σ_total|) - ½ dᵀ Σ_total⁻¹ d`` per peak,
    ``(K, J)``; ``-inf`` for padding. ``state`` is ``(K, 5)``."""
    cov, total, inv = _peak_precision(cat, ln_s, jitter, dtype)
    s = np.exp(ln_s)
    scaled = cat.loglike - np.sum(
        cat.n * ln_s + 0.5 * cat.chi2 * (s**-2 - 1.0), axis=-1
    )
    d = state[:, None, :] - cat.mean
    quad = np.einsum("kja,kjab,kjb->kj", d, inv, d)
    log_det = 0.5 * (np.linalg.slogdet(cov)[1] - np.linalg.slogdet(total)[1])
    # a large negative number, not -inf: no NaN in a gradient through a
    # night with no peak
    return np.where(cat.valid, scaled + log_det - 0.5 * quad, -1e30)


def _eps_nodes(n):
    """Gauss–Chebyshev nodes of the Beta(½, ½) integral of ε (equal weights)."""
    return np.sin((2 * np.arange(1, n + 1) - 1) * np.pi / (4 * n)) ** 2


def _log_mixture(cat, ln_s, jitter, pv, ln_f, dtype, inliers=False):
    """``ln ∫ dε Beta(ε) ½ Σ_σ Π_k [(1-ε) e^{ℓ̃_k} + ε e^{b_k}]``, with the
    brighter/fainter sign ``σ`` flipping ``(p, v)``. ``pv`` is ``(K, 4)``.

    With ``inliers``, also the expected number of inlier nights: the night
    responsibilities ``(1-ε) e^{ℓ̃_k} / [(1-ε) e^{ℓ̃_k} + ε e^{b_k}]``, summed
    over nights and averaged over ``(σ, ε)`` with their posterior weights.
    """
    eps = _eps_nodes(cat.n_eps).astype(dtype)
    per_sign, resp = [], []
    for sign in (1.0, -1.0):
        state = np.concatenate(
            [sign * pv, np.full((pv.shape[0], 1), ln_f, dtype)], axis=1
        )
        a = logsumexp(_peak_terms(cat, ln_s, jitter, state, dtype), axis=1)
        inside = np.log1p(-eps)[:, None] + a[None]
        night = np.logaddexp(inside, np.log(eps)[:, None] + cat.log_b)
        per_sign.append(night.sum(axis=1))
        resp.append(np.exp(inside - night).sum(axis=1))
    total = np.stack(per_sign)  # (2, n_eps)
    out = logsumexp(total) - np.log(2.0 * cat.n_eps)
    if not inliers:
        return out
    weight = np.exp(total - logsumexp(total))
    return out, np.sum(weight * np.stack(resp))


# ---------------------------------------------------------------------------
# The sampling space of one cell
# ---------------------------------------------------------------------------

_Z = slice(1, 5)  # y of z (4), then the logistic of ln a, then ζ's y (2),
#                   the logistic of ln f, one logistic per error scale


def _logistic_log_pdf(w):
    return jax.nn.log_sigmoid(w) + jax.nn.log_sigmoid(-w)


def _log_prior(x):
    """The cell prior in the chart (independent of the cell)."""
    yz = x[6:8]
    return (
        _logistic_log_pdf(x[0])
        + _logistic_log_pdf(x[5])
        + _logistic_log_pdf(x[8])
        + np.sum(_logistic_log_pdf(x[9:]))
        + np.sum(-0.5 * x[_Z] ** 2 - 0.5 * math.log(2 * math.pi))
        - math.log(math.pi)
        - 2.0 * np.log1p(np.sum(yz**2))
    )


@functools.lru_cache(maxsize=None)
def _prior_sampler(d, dtype):
    """``sample(key, n)``: exact draws of the chart prior (a stable function)."""

    def sample(key, n):
        k = jax.random.split(key, 6)
        zeta = oc.sample_zeta(k[3], (n,)).astype(dtype)  # uniform on the disc
        return np.concatenate(
            [
                jax.random.logistic(k[0], (n, 1), dtype),
                jax.random.normal(k[1], (n, 4), dtype),
                jax.random.logistic(k[2], (n, 1), dtype),
                zeta / np.sqrt(1.0 - np.sum(zeta**2, -1, keepdims=True)),
                jax.random.logistic(k[4], (n, 1), dtype),
                jax.random.logistic(k[5], (n, d - 9), dtype),
            ],
            axis=1,
        )

    return sample


@jax.tree_util.register_pytree_node_class
class _Space:
    """Coordinates, prior and likelihood of one cell: ``x`` is the flat chart
    ``(w_P, y_z[4], w_a, y_ζ[2], w_f, w_s[G])``.

    A pytree: the cell bounds, the nights and the catalogue are traced, so one
    compilation serves every cell of a system.
    """

    def __init__(
        self, cat, cell, t_anchor, t_k, a_range, e_max, flux_floor, jitter,
        dtype,
    ):  # fmt: skip
        self.cat, self.cell, self.dtype = cat.cast(dtype), cell, dtype
        self.t_anchor, self.t_k = t_anchor, np.asarray(t_k, dtype)
        self.a_range, self.e_max = tuple(a_range), float(e_max)
        self.jitter = np.asarray(jitter, dtype)
        self.ln_lo = np.asarray(math.log(cell.p_lo), dtype)
        self.ln_hi = np.asarray(math.log(cell.p_hi), dtype)
        self.ln_f_floor = math.log(flux_floor)
        self.d = 9 + len(cat.blocks)
        self.s_w2 = (a_range[1] ** 2 - a_range[0] ** 2) / (
            2 * math.log(a_range[1] / a_range[0]) * 3
        )

    def tree_flatten(self):
        children = (self.cat, self.t_k, self.jitter, self.ln_lo, self.ln_hi)
        aux = (
            self.dtype, self.t_anchor, self.a_range, self.e_max,
            self.ln_f_floor, self.d, self.s_w2,
        )  # fmt: skip
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        new = object.__new__(cls)
        new.cat, new.t_k, new.jitter, new.ln_lo, new.ln_hi = children
        (new.dtype, new.t_anchor, new.a_range, new.e_max, new.ln_f_floor,
         new.d, new.s_w2) = aux  # fmt: skip
        new.cell = None
        return new

    def period(self, w):
        """The period of the logistic coordinate ``w`` (days)."""
        return np.exp(
            self.ln_lo + (self.ln_hi - self.ln_lo) * jax.nn.sigmoid(w)
        )

    def zeta(self, y):
        """The ζ of its chart coordinate ``y`` (last axis, 2)."""
        r2 = np.sum(y**2, axis=-1, keepdims=True)
        return math.sqrt(self.e_max) * y / np.sqrt(1.0 + r2)

    def decode(self, x):
        """The orbit coordinates of chart vector(s) ``x`` (last axis)."""
        la = math.log(self.a_range[0])
        da = math.log(self.a_range[1] / self.a_range[0])
        y = x[..., _Z]
        direction = y / np.linalg.norm(y, axis=-1, keepdims=True)
        a = np.exp(la + da * jax.nn.sigmoid(x[..., 5]))
        return {
            "period": self.period(x[..., 0]),
            "z": direction * np.sqrt(2.0 * a)[..., None],
            "zeta": self.zeta(x[..., 6:8]),
            "ln_f": self.ln_f_floor * (1.0 - jax.nn.sigmoid(x[..., 8])),
            # error scales stay at 1 where a scale does not factor out (gains,
            # closure offsets): the coordinates remain, with no likelihood
            "ln_s": float(self.cat.rescalable)
            * math.log(S_MAX)
            * (2.0 * jax.nn.sigmoid(x[..., 9:]) - 1.0),
        }

    def states_at(self, t, period, ti, zeta):
        """``(p, v)`` of the Thiele–Innes constants ``ti`` (the anchor-frame
        TI′) at times ``t`` (days from the anchor): ``(len(t), 4)``, in mas and
        mas/day."""
        ecc, anomaly = oc.zeta_to_eccentricity(zeta)
        a, b, f, g = oc.anchor_unrotate(tuple(ti), anomaly)
        dt_peri = -anomaly * period / (2.0 * np.pi)  # anchor at time 0

        def position(dt):
            x, y = _unit_orbit(dt, period, dt_peri, ecc)
            return np.stack([b * x + g * y, a * x + f * y])

        pos, vel = jax.jvp(position, (t,), (np.ones_like(t),))
        return np.concatenate([pos.T, vel.T], axis=1)

    def states(self, period, ti, zeta):
        """``(p, v)`` of ``ti`` at the nights: ``(K, 4)``."""
        return self.states_at(self.t_k, period, ti, zeta)

    def loglike(self, x):
        c = self.decode(x)
        pv = self.states(c["period"], oc.z_to_ti(c["z"]), c["zeta"])
        return _log_mixture(
            self.cat, c["ln_s"], self.jitter, pv, c["ln_f"], self.dtype
        )

    def logprior(self, x):
        return _log_prior(x)

    def sample(self, key, n):
        """``n`` exact draws from the cell prior."""
        return _prior_sampler(self.d, self.dtype)(key, n)


# ---------------------------------------------------------------------------
# The search
# ---------------------------------------------------------------------------


def _target(x, space):
    """The label-summed log target (log prior + log likelihood) at ``x``."""
    return space.loglike(x) + space.logprior(x)


def _inliers(x, space):
    """Expected number of inlier nights at ``x`` (see ``_log_mixture``)."""
    c = space.decode(x)
    pv = space.states(c["period"], oc.z_to_ti(c["z"]), c["zeta"])
    return _log_mixture(
        space.cat, c["ln_s"], space.jitter, pv, c["ln_f"], space.dtype,
        inliers=True,
    )[1]  # fmt: skip


def _objective(x, space):
    """The search objective: the target with the chi-4 radial prior of ``y_z``.

    The likelihood depends on the direction of ``y_z`` only, so the target has
    no mode in its radius; ``+ 3 ln |y_z|`` (the radial volume) puts it at
    ``√3`` with curvature 2, and leaves the angular curvature at a mode as it
    is. It places modes and their covariances; the evidence uses the target.
    """
    return _target(x, space) + 3.0 * np.log(np.linalg.norm(x[_Z]))


def _night_terms(x, space):
    """``(K, J)`` peak terms at ``x`` for the brighter-star sign ``σ = +1``."""
    c = space.decode(x)
    pv = space.states(c["period"], oc.z_to_ti(c["z"]), c["zeta"])
    state = np.concatenate(
        [pv, np.full((pv.shape[0], 1), c["ln_f"], space.dtype)], axis=1
    )
    return _peak_terms(space.cat, c["ln_s"], space.jitter, state, space.dtype)


def _labels(x, space):
    """Each night's best peak, or -1 (outlier) where the floor is higher.

    At the prior mean ``ε̄ = ½`` the inlier and outlier weights ``1 - ε̄`` and
    ``ε̄`` are equal, so the comparison is of ``ℓ̃_k`` and ``b_k``.
    """
    terms = _night_terms(x, space)
    j = np.argmax(terms, axis=1)
    best = np.take_along_axis(terms, j[:, None], axis=1)[:, 0]
    return np.where(best > space.cat.log_b, j, -1)


def _lm(f, x, args, n_iter, step_max=10.0):
    """Saddle-free Levenberg–Marquardt ascent of ``f(x, *args)``.

    Newton with ``|eigenvalues|`` of the Hessian plus a damping ``λ`` that
    falls on an accepted step and rises on a rejected one (a step is accepted
    only if ``f`` is finite and rises). Stops after ``n_iter`` steps or when
    the predicted gain ``g·step`` is below ``TOL_GAIN``. vmap-able.
    Returns ``(x, f(x))``.
    """
    grad, hess = jax.grad(f), jax.hessian(f)

    def body(c):
        i, x, fx, lam, _ = c
        g, h = grad(x, *args), -hess(x, *args)
        e, v = np.linalg.eigh(0.5 * (h + h.T))
        step = v @ ((v.T @ g) / (np.abs(e) + lam))
        step = step * np.minimum(
            1.0, step_max / (np.linalg.norm(step) + 1e-30)
        )
        fn = f(x + step, *args)
        ok = np.isfinite(fn) & (fn > fx)
        gain = np.where(np.isfinite(g @ step), g @ step, 0.0)
        return (
            i + 1,
            np.where(ok, x + step, x),
            np.where(ok, fn, fx),
            np.where(
                ok, np.maximum(lam / 3.0, 1e-9), np.minimum(lam * 4.0, 1e8)
            ),
            np.where(ok, gain, np.inf),
        )

    fx = f(x, *args)
    fx = np.where(np.isfinite(fx), fx, -np.inf)
    init = (0, x, fx, np.asarray(1e-2, x.dtype), np.asarray(np.inf, x.dtype))
    _, x, fx, _, _ = jax.lax.while_loop(
        lambda c: (c[0] < n_iter) & (c[4] > TOL_GAIN), body, init
    )
    return x, fx


def _design(space, nights, v):
    """``states`` at a set's three nights as a linear map of TI′: ``(3, 4, 4)``.

    ``v = (w_P, y_ζ[2])`` is the chart's period and ζ.
    """
    period, zeta = space.period(v[0]), space.zeta(v[1:3])
    t = space.t_k[nights]
    zero = np.zeros(4, space.dtype)
    return jax.jacfwd(lambda ti: space.states_at(t, period, ti, zeta))(zero)


def _solve_set(g, mean, w, s_w2):
    """TI′ and the residual of one set at design ``g`` (variable projection).

    Weighted least squares of the set's ``(p, v)`` peak means on TI′ with the
    precisions ``w`` and a ridge ``1 / s_w²`` (the working prior), so the solve
    is defined on degenerate sets. Returns ``(R, ti)``, ``R = Σ rᵀ W r +
    |ti|²/s_w²``; the ``χ²`` cut applies to ``R - |ti|²/s_w²``, the data
    residual (the ridge term grows with the orbit and is no part of the null
    ``χ²``).

    Source: Golub & Pereyra (1973).
    """
    a = np.einsum("mia,mij,mjb->ab", g, w, g) + np.eye(4, dtype=g.dtype) / s_w2
    b = np.einsum("mia,mij,mj->a", g, w, mean)
    ti = np.linalg.solve(a, b)
    r = mean - np.einsum("mia,a->mi", g, ti)
    return np.einsum("mi,mij,mj->", r, w, r) + ti @ ti / s_w2, ti


@jax.jit
def _grid_stage(space, plan):
    """Every set's residual on a ``ln P`` grid of the cell, at every ζ start.

    ``K_COH + 1`` points equally spaced in ``ν = 1 / P`` over the cell: a cell
    is ``1 / T`` wide in ``ν``, so a set of span ``Δ`` needs
    ``⌈K_COH Δ / T⌉ + 1 <= K_COH + 1``. Returns the residuals ``(S, Z, n)``,
    their local minima along the grid (grid ends count) and the grid's ``ln P``.
    """
    nu = np.linspace(
        1.0 / np.exp(space.ln_hi), 1.0 / np.exp(space.ln_lo), K_COH + 1
    )
    period, zeta = 1.0 / nu, space.zeta(plan["y0"])
    zero = np.zeros(4, space.dtype)

    def design(p, z):
        return jax.jacfwd(lambda ti: space.states(p, ti, z))(zero)

    g = jax.vmap(lambda z: jax.vmap(lambda p: design(p, z))(period))(zeta)

    def per_set(nights, w, mean):
        return jax.vmap(
            jax.vmap(lambda gg: _solve_set(gg, mean, w, space.s_w2)[0])
        )(g[:, :, nights])

    r = jax.vmap(per_set)(plan["nights"], plan["w"], plan["mean"])
    left = np.concatenate([r[..., :1], r[..., :-1]], -1)
    right = np.concatenate([r[..., 1:], r[..., -1:]], -1)
    return r, (r <= left) & (r <= right) & np.isfinite(r), np.log(period)


def _varpro_objective(v, nights, w, mean, space):
    return -0.5 * _solve_set(_design(space, nights, v), mean, w, space.s_w2)[0]


def _chart_from(space, v, ti, ln_f):
    """The chart vector of ``(w_P, y_ζ) = v``, TI′ and ``ln f``, with ``y_z``
    at the radius ``R0`` and the error scales at one."""
    z = oc.ti_to_z(*tuple(ti))
    la = math.log(space.a_range[0])
    da = math.log(space.a_range[1] / space.a_range[0])
    frac = np.clip((np.log(oc.semimajor(z)) - la) / da, 1e-6, 1.0 - 1e-6)
    ff = np.clip(1.0 - ln_f / space.ln_f_floor, 1e-6, 1.0 - 1e-6)
    return np.concatenate(
        [
            v[:1],
            z / np.linalg.norm(z) * R0,
            (np.log(frac) - np.log1p(-frac))[None],
            v[1:3],
            (np.log(ff) - np.log1p(-ff))[None],
            np.zeros(space.d - 9, space.dtype),
        ]
    )


@functools.partial(jax.jit, static_argnames=("n_iter",))
def _varpro_chunk(space, plan, s, v0, n_iter):
    """Refine ``(ln P, ζ)`` of each set ``s`` from ``v0``; return the chart
    vector, labels, the *data* residual (without the ridge term: the quantity
    the ``χ²`` cut applies to), ``v`` and the Hessian of ``R / 2`` in ``v``."""

    def one(si, v):
        args = (plan["nights"][si], plan["w"][si], plan["mean"][si], space)
        v, fv = _lm(_varpro_objective, v, args, n_iter)
        ti = _solve_set(
            _design(space, args[0], v), args[2], args[1], space.s_w2
        )[1]
        x = _chart_from(space, v, ti, plan["lnf"][si])
        h = -jax.hessian(_varpro_objective)(v, *args)
        return x, _labels(x, space), -2.0 * fv - ti @ ti / space.s_w2, v, h

    return jax.vmap(one)(s, v0)


def _labelled_objective(v, lab, wk, mk, space):
    """Minus half the variable-projection residual of every night labelled an
    inlier, at ``(w_P, y_ζ) = v``: the C-step's labelled target (``s = 1``,
    full ``(p, v)`` precisions, outlier nights constant)."""
    k = lab.shape[0]
    j = np.maximum(lab, 0)
    w = np.where((lab >= 0)[:, None, None], wk[np.arange(k), j], 0.0)
    g = _design(space, np.arange(k), v)
    return -0.5 * _solve_set(g, mk[np.arange(k), j], w, space.s_w2)[0]


@functools.partial(jax.jit, static_argnames=("n_iter",))
def _cstep_chunk(space, plan, v, lab, n_iter):
    """One C-step per candidate: Newton in ``(ln P, ζ)`` on the labelled
    target (TI′ and ``ln f`` solved linearly), then relabel at the result.
    Returns ``v``, the labels, the chart vector, the residual and the Hessian
    of ``R / 2`` in ``v``."""

    def one(v, lab):
        args = (lab, plan["wk"], plan["mk"], space)
        v, fv = _lm(_labelled_objective, v, args, n_iter)
        k = lab.shape[0]
        j = np.maximum(lab, 0)
        w = np.where(
            (lab >= 0)[:, None, None], plan["wk"][np.arange(k), j], 0.0
        )
        ti = _solve_set(
            _design(space, np.arange(k), v),
            plan["mk"][np.arange(k), j],
            w,
            space.s_w2,
        )[1]
        wf = np.where(lab >= 0, plan["wf"][np.arange(k), j], 0.0)
        ln_f = np.sum(wf * plan["fm"][np.arange(k), j]) / np.maximum(
            wf.sum(), 1e-30
        )
        x = _chart_from(space, v, ti, ln_f)
        h = -jax.hessian(_labelled_objective)(v, *args)
        return v, _labels(x, space), x, -2.0 * fv, h

    return jax.vmap(one)(v, lab)


@functools.partial(jax.jit, static_argnames=("n_iter",))
def _refine(space, x, n_iter):
    """Newton on the label-summed objective from one ``x``: the result, minus
    its Hessian, the target and the expected inlier count. One candidate per
    call: a batch would run every member as long as the slowest."""
    x, _ = _lm(_objective, x, (space,), n_iter)
    return (
        x,
        -jax.hessian(_objective)(x, space),
        _target(x, space),
        _inliers(x, space),
    )


@jax.jit
def _scores(space, x):
    """Laplace ``ln Z`` at each ``x`` of a batch, from the saddle-free Hessian
    (``|eigenvalues|``) of the objective: the start's estimate of its mode's."""

    def one(x):
        h = -jax.hessian(_objective)(x, space)
        e = np.linalg.eigvalsh(0.5 * (h + h.T))
        return (
            _objective(x, space)
            + 0.5 * x.shape[0] * math.log(2.0 * math.pi)
            - 0.5 * np.sum(np.log(np.maximum(np.abs(e), 1e-12)))
        )

    return jax.vmap(one)(x)


def _prune(space, x):
    """Indices of the candidates within ``W_PRUNE`` of the best Laplace ``ln Z``.

    A candidate that far below the best is covered by the prior share of the
    mixture, whose weights are bounded by ``e^W / α``; dropping it cannot bias
    the evidence. The estimate is at the start ``x`` (not its mode): it agreed
    with the refined one to 1-2 nats where measured, and the margin is twice
    ``W`` for that. A non-finite score drops the candidate.
    """
    if not len(x):
        return onp.zeros(0, int)
    score = onp.concatenate(
        [
            onp.asarray(
                _scores(
                    space, np.asarray(_padded(x[sl], CHUNK_C), space.dtype)
                )
            )[: sl.stop - sl.start]
            for sl in _chunks(len(x), CHUNK_C)
        ]
    )
    score = onp.where(onp.isfinite(score), score, -onp.inf)
    return onp.flatnonzero(score >= score.max() - W_PRUNE)


def _chunks(n, size):
    """Slices of ``range(n)`` of ``size`` (the last is padded by the caller)."""
    return [slice(i, min(i + size, n)) for i in range(0, n, size)]


def _padded(a, size):
    """``a`` with its first row repeated to ``size`` rows."""
    return onp.concatenate([a, onp.repeat(a[:1], size - len(a), axis=0)])


def _ransac_count(m):
    """Random minimal sets of ``m`` nights, one all-inlier with prob ``1 - δ``.

    ``n = ⌈ln δ / ln(1 - w^m)⌉`` with inlier fraction ``w = 1 - EPS_HI``.

    Source: Fischler & Bolles (1981).
    """
    return math.ceil(math.log(DELTA) / math.log(1.0 - (1.0 - EPS_HI) ** m))


def search_plan(space, key=None):
    """The minimal sets of a system, as arrays (the same for every cell).

    Pairs of nights (positions and velocities: 8 equations for the 4 TI′ at
    fixed ``(P, ζ)``) and triplets (positions only: 6), with every combination
    of their peaks. Above ``N_SET_MAX`` a random subset is drawn from ``key``,
    split over the two sizes by their counts and at least
    [`_ransac_count`][virgil.orbit_cells._ransac_count] each (the count
    guarantees an all-inlier set, not the set that spans the inliers: see
    design §6.1).

    Each set carries its precisions ``W`` (the ``(p, v)`` block of the peak's
    total covariance at ``ln s = 0`` and the fitted jitter), the peak means, the
    precision-weighted ``ln f`` and the cut on its residual: the ``1 - δ``
    quantile of ``χ²`` with the set's equations less the 4 TI′, times
    ``S_MAX²`` where the error scale is free (``(P, ζ)`` are fitted too,
    which only lowers the residual: the cut is conservative). It applies to the
    data residual ``Σ rᵀ W r`` at the ridge solution, not to the ridge-penalized
    ``R``: the ridge term ``|TI′|²/s_w²`` grows with the orbit (the old cut
    removed every true set at ``a = 25`` mas). The ridge solution's data
    residual exceeds the least-squares one by at most ``|TI′|² / (4 s_w²)``
    (per eigendirection ``λ² d / (d + λ)²`` is at most ``λ / 4``), negligible
    once the sets are informative (``σ`` well below ``s_w``).
    """
    cat, dt = space.cat, space.dtype
    zero = np.zeros(len(cat.blocks), dt)
    total = onp.asarray(_peak_precision(cat, zero, space.jitter, dt)[1], float)
    w_pv = onp.linalg.inv(total[..., :4, :4])
    w_pos = onp.zeros_like(w_pv)
    w_pos[..., :2, :2] = onp.linalg.inv(total[..., :2, :2])
    w_f = 1.0 / total[..., 4, 4]
    mean_all, valid = onp.asarray(cat.mean, float), onp.asarray(cat.valid)
    peaks = [onp.flatnonzero(v) for v in valid]
    seed = 0 if key is None else int(jax.random.randint(key, (), 0, 2**31 - 1))
    rng = onp.random.default_rng(seed)
    sets = {2: [], 3: []}
    for m in (2, 3):
        for nights in itertools.combinations(range(len(peaks)), m):
            if all(len(peaks[k]) for k in nights):
                for pk in itertools.product(*[peaks[k] for k in nights]):
                    sets[m].append((nights, pk))
    total_sets = sum(len(v) for v in sets.values())
    if total_sets > N_SET_MAX:
        for m in (2, 3):
            n = max(
                _ransac_count(m), round(N_SET_MAX * len(sets[m]) / total_sets)
            )
            if n < len(sets[m]):
                pick = rng.choice(len(sets[m]), n, replace=False)
                sets[m] = [sets[m][i] for i in sorted(pick)]
    chosen = sets[2] + sets[3]
    s = len(chosen)
    nights = onp.zeros((s, 3), int)
    w = onp.zeros((s, 3, 4, 4))
    mean = onp.zeros((s, 3, 4))
    lnf, cut = onp.zeros(s), onp.zeros(s)
    scale = S_MAX**2 if cat.rescalable else 1.0
    for i, (nt, pk) in enumerate(chosen):
        wt = w_pv if len(nt) == 2 else w_pos
        nights[i, :] = nt[0]
        nights[i, : len(nt)] = nt
        for j, (k, p) in enumerate(zip(nt, pk)):
            w[i, j], mean[i, j] = wt[k, p], mean_all[k, p, :4]
        wf = onp.array([w_f[k, p] for k, p in zip(nt, pk)])
        lnf[i] = (
            wf
            @ onp.array([mean_all[k, p, 4] for k, p in zip(nt, pk)])
            / wf.sum()
        )
        cut[i] = scale * stats.chi2.isf(DELTA, 4 if len(nt) == 2 else 2)
    ecc = onp.minimum(onp.array(ZETA_E), 0.5 * space.e_max)
    ecc = onp.maximum(ecc, min(1e-4, 0.5 * space.e_max))  # ζ = 0 has no M_a
    starts = [(ecc[0], 0.0)] + [
        (e, 2.0 * math.pi * i / ZETA_PHASES)
        for e in ecc[1:]
        for i in range(ZETA_PHASES)
    ]
    zeta = onp.array(
        [
            [math.sqrt(e) * math.cos(a / 2), math.sqrt(e) * math.sin(a / 2)]
            for e, a in starts
        ]
    )
    y0 = zeta / onp.sqrt(space.e_max - (zeta**2).sum(1, keepdims=True))
    return {
        "nights": np.asarray(nights), "w": np.asarray(w, dt), "mean": np.asarray(mean, dt),
        "lnf": np.asarray(lnf, dt), "cut": onp.asarray(cut), "y0": np.asarray(y0, dt),
        "wk": np.asarray(w_pv, dt), "mk": np.asarray(mean_all[..., :4], dt),
        "wf": np.asarray(w_f, dt), "fm": np.asarray(mean_all[..., 4], dt),
    }  # fmt: skip


def _unique_starts(lab, v, h, r):
    """Candidates (best residual first) not within ``MAHAL`` of a kept one.

    Two candidates with the same labels are the same when their
    squared Mahalanobis distance in ``(w_P, y_ζ)`` is below one under *both*
    Hessians (the symmetric rule of ``distinct_modes``).
    """
    prec = 0.5 * (h + h.swapaxes(1, 2))
    e, u = onp.linalg.eigh(prec)
    h[:] = (u * onp.maximum(e, 1e-12)[:, None, :]) @ u.swapaxes(1, 2)
    keys = [row.tobytes() for row in lab]
    kept, groups = [], {}
    for i in onp.argsort(r):
        near = groups.get(keys[i])
        if near:
            idx = onp.array(near)
            d = v[i] - v[idx]
            a = onp.einsum("mi,mij,mj->m", d, h[idx], d)
            b = onp.einsum("mi,ij,mj->m", d, h[i], d)
            if (onp.maximum(a, b) < 1.0).any():
                continue
        kept.append(i)
        groups.setdefault(keys[i], []).append(i)
    return onp.array(kept, int)


def search_cell(space, key=None, *, plan=None, info=None):
    """The modes of the label-summed target in one cell, best first.

    1. Every minimal set of [`search_plan`][virgil.orbit_cells.search_plan], at
       each of the 17 ζ starts, on a ``ln P`` grid of the cell; at every grid
       minimum a damped Newton in ``(ln P, ζ)`` on the set's variable-projection
       residual.
    2. Candidates whose set residual exceeds its derived ``χ²`` cut are dropped,
       the rest deduplicated by labels and ``(ln P, ζ)``.
    3. C-steps (at most ``N_C``): Newton on the labelled target in the full
       chart, relabel, until the labels repeat.
    4. Newton on the label-summed objective; the covariance is the inverse of
       its saddle-free Hessian (``|eigenvalues|``); ``distinct_modes``.

    ``info``, a dict, receives the counts of each step and ``evaluations``, the
    likelihood-equivalent cost (a gradient is 3, a Hessian 3 per dimension).

    Source: Fischler & Bolles (1981), Rousseeuw (1984), Rousseeuw & Van Driessen
    (1999), Golub & Pereyra (1973).
    """
    plan = search_plan(space, key) if plan is None else plan
    info = {} if info is None else info
    n_sets, d, k = len(plan["nights"]), space.d, len(space.cat.log_b)
    newton = 4.0 + 3.0 * d  # one Newton step, in likelihood evaluations
    info.update(sets=n_sets, candidates=0, survivors=0, unique=0, modes=0)
    info["evaluations"] = 0.0
    if n_sets == 0:
        return []
    r, is_min, ln_p = _grid_stage(space, plan)
    s_i, z_i, p_i = onp.nonzero(onp.asarray(is_min))
    info["candidates"] = len(s_i)
    info["evaluations"] += float(r.size * 2 / max(k, 1))
    frac = onp.clip(
        (onp.asarray(ln_p)[p_i] - float(space.ln_lo))
        / float(space.ln_hi - space.ln_lo),
        0.01,
        0.99,
    )
    v0 = onp.stack(
        [onp.log(frac) - onp.log1p(-frac)]
        + [onp.asarray(plan["y0"])[z_i, 0], onp.asarray(plan["y0"])[z_i, 1]],
        1,
    )
    out = []
    for sl in _chunks(len(s_i), CHUNK_VP):
        n = sl.stop - sl.start
        res = _varpro_chunk(
            space, plan, _padded(s_i[sl], CHUNK_VP),
            np.asarray(_padded(v0[sl], CHUNK_VP), space.dtype), N_VARPRO,
        )  # fmt: skip
        out.append([onp.asarray(a)[:n] for a in res])
    info["evaluations"] += len(s_i) * N_VARPRO * (4.0 + 9.0) * 3 / max(k, 1)
    if not out:
        return []
    x, lab, rr, v, h = (onp.concatenate([o[i] for o in out]) for i in range(5))
    ok = onp.isfinite(x).all(1) & onp.isfinite(rr) & (rr <= plan["cut"][s_i])
    info["survivors"] = int(ok.sum())
    if not ok.any():
        return []
    x, lab, rr, v, h = x[ok], lab[ok], rr[ok], v[ok], h[ok]
    keep = _unique_starts(lab, v, h, rr)
    v, lab = v[keep], lab[keep]
    info["unique"] = len(v)
    x = onp.zeros((len(v), d))
    r2, h2 = onp.zeros(len(v)), onp.zeros((len(v), 3, 3))
    active, steps = onp.arange(len(v)), 0
    for n_pass, n_iter in enumerate((N_PRE,) + (N_NEWTON,) * N_C):
        changed = []
        for sl in _chunks(len(active), CHUNK_C):
            idx = active[sl]
            res = _cstep_chunk(
                space, plan,
                np.asarray(_padded(v[idx], CHUNK_C), space.dtype),
                np.asarray(_padded(lab[idx], CHUNK_C)), n_iter,
            )  # fmt: skip
            vs, ls, xs, rs, hs = (onp.asarray(a)[: len(idx)] for a in res)
            changed.append(idx[(ls != lab[idx]).any(1)])
            v[idx], lab[idx], x[idx], r2[idx], h2[idx] = vs, ls, xs, rs, hs
            steps += len(idx) * n_iter
        if n_pass == 0:
            # a few steps merge most candidates onto their modes
            ok = onp.isfinite(x).all(1) & onp.isfinite(v).all(1)
            keep = onp.flatnonzero(ok)[
                _unique_starts(lab[ok], v[ok], h2[ok], r2[ok])
            ]
            keep = keep[_prune(space, x[keep])]
            if not len(keep):
                return []
            v, lab, x, r2, h2 = v[keep], lab[keep], x[keep], r2[keep], h2[keep]
            active = onp.arange(len(v))
            continue
        active = onp.concatenate(changed) if changed else active[:0]
        if not len(active):
            break
    info["evaluations"] += steps * (1.0 + 3.0 + 9.0)
    ok = onp.isfinite(x).all(1) & onp.isfinite(v).all(1)
    x, lab, v, r2, h2 = x[ok], lab[ok], v[ok], r2[ok], h2[ok]
    keep = _unique_starts(lab, v, h2, r2)
    x = x[keep][_prune(space, x[keep])]
    modes = []
    for xi in x:
        xi, hi, ti, ni = (
            onp.asarray(a)
            for a in _refine(space, np.asarray(xi, space.dtype), N_NEWTON)
        )
        info["evaluations"] += N_NEWTON * newton
        hi = 0.5 * (hi + hi.T)
        if (
            onp.isfinite(xi).all()
            and onp.isfinite(hi).all()
            and onp.isfinite(ti)
        ):
            # saddle-free: |eigenvalues|, floored at the numerical rank (a
            # saturated coordinate has none); a saddle's t is still a proposal
            e, vec = onp.linalg.eigh(hi)
            e = onp.maximum(
                onp.abs(e), len(e) * onp.finfo(hi.dtype).eps * onp.abs(e).max()
            )
            modes.append(Mode(xi, float(ti), (vec / e) @ vec.T, float(ni)))
    modes = distinct_modes(modes)
    info["modes"] = len(modes)
    return modes


def _symmetry_images(d):
    """The 15 non-identity linear maps of the chart that leave the target
    invariant (the group of order 16, identity implicit).

    The ``z`` cover (``±z_u``, ``±z_v``: ``y[0:2]`` and ``y[2:4]`` change sign
    independently), the ``ζ`` twin (``y_ζ → -y_ζ``) and the label swap
    ``TI′ → -TI′`` (``z → i z``: ``y → (-y1, y0, -y3, y2)``, a symmetry
    because the target sums the brighter/fainter sign).
    """
    eye = onp.eye(d)

    def diag(*idx):
        m = eye.copy()
        for i in idx:
            m[i, i] = -1.0
        return m

    swap = eye.copy()
    swap[1:5, 1:5] = [[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 0, -1], [0, 0, 1, 0]]
    gens = [diag(1, 2), diag(3, 4), diag(6, 7), swap]
    group = {eye.tobytes(): eye}
    frontier = [eye]
    while frontier:
        nxt = []
        for g in frontier:
            for h in gens:
                m = h @ g
                if m.tobytes() not in group:
                    group[m.tobytes()] = m
                    nxt.append(m)
        frontier = nxt
    return [
        m for key, m in sorted(group.items()) if not onp.array_equal(m, eye)
    ]


@dataclasses.dataclass
class CellFit:
    """The position-level fit of one cell (search, importance sampling, draws).

    Attributes
    ----------
    cell : Cell
    log_z, log_z_error : float
        ``ln Ẑ_N^pos`` of the cell's normalized prior (the cell's mass is
        applied in [`CellsResult`][virgil.orbit_cells.CellsResult]) and its
        delta-method error ``√(1/ESS - 1/n)``.
    modes : list of Mode
    draws : array or None
        Resample-move posterior draws (chart vectors); ``None`` until made.
    log_w, x_is : array
        The importance draws and their log weights (exact, not resampled).
    at_floor : bool
        ``cell-at-floor``: the draws sit at the all-outlier floor although the
        search found a mode above it, so the search missed the dominant mode.
        The cell *failed*: never report its ``log_z`` as an evidence.
    undersampled : bool
        ``cell-undersampled``: the ESS is below ``ESS_TARGET``; information only.
    search_evaluations, is_evaluations : int
        Likelihood-equivalent evaluations of the search and the sampling.
    """

    cell: Cell
    log_z: float
    log_z_error: float
    modes: list
    draws: object
    log_w: object
    x_is: object
    at_floor: bool
    undersampled: bool
    search_evaluations: int
    is_evaluations: int
    evidence: object = dataclasses.field(repr=False, default=None)
    plan: object = dataclasses.field(repr=False, default=None)
    space: object = dataclasses.field(repr=False, default=None)


def decode_particles(result):
    """Orbit coordinates of a fit's posterior draws.

    Returns a dict of arrays with ``period``, ``z``, ``zeta``, ``ln_f``,
    ``ln_s`` and the reporting invariants ``a``, ``ecc``, ``cos_inc``,
    ``omega`` and ``Omega`` (invariant under the covers of ``z`` and ``ζ``).
    """
    if result.draws is None:
        raise ValueError("The fit has no draws (failed, or not drawn yet).")
    c = result.space.decode(np.asarray(result.draws))
    return {**c, **oc.invariants(c["period"], c["z"], c["zeta"])}


def _draw(fit, key):
    """Resample-move posterior draws of a fit, with the mass matrix of its
    dominant mode."""
    with run_in(onp.dtype(fit.space.dtype).name):
        return resample_move(
            _target, fit.space, fit.evidence, key,
            cov=fit.modes[0].cov if fit.modes else None,
        )  # fmt: skip


def position_leg(
    summaries,
    log_outlier,
    cell,
    key,
    *,
    t_anchor,
    t_nights,
    a_range=A_RANGE,
    e_max=E_MAX,
    flux_floor=FLUX_FLOOR,
    jitter=0.0,
    n_is=N_IS,
    posterior_draws=True,
    dtype="float64",
    plan=None,
):
    """Search, defensive importance sampling and posterior draws of one cell.

    The modes of the cell ([`search_cell`][virgil.orbit_cells.search_cell]) with
    the images of the target's symmetries are the components of a defensive
    mixture; ``ln Ẑ`` is its importance-sampling estimate and the draws come
    from an HMC resample-move of the weighted draws.

    Parameters
    ----------
    summaries : sequence of NightSummary
        The nights.
    log_outlier : array-like
        Each night's ``b_k``
        (from [`outlier_log_density`][virgil.orbit_cells.outlier_log_density]).
    cell : Cell
    key : jax.random.PRNGKey
    t_anchor, t_nights : float, array-like
        The anchor time and each night's reference time (days, any common
        origin).
    a_range, e_max, flux_floor : declared prior bounds
        Semimajor axis (mas), maximum eccentricity and flux-ratio floor.
    jitter : float, optional
        Position jitter (mas) added in quadrature, from
        [`fit_jitter`][virgil.orbit_cells.fit_jitter] (default 0).
    n_is : int, optional
        Importance draws.
    posterior_draws : bool, optional
        Make the resample-move draws (not for a cell that failed).
    dtype : {"float64", "float32"}, optional
        Precision of the run (default float64, as ``fit``).
    plan : dict, optional
        The sets of [`search_plan`][virgil.orbit_cells.search_plan], reused
        across the cells of a system.

    Returns
    -------
    CellFit

    Source: Hesterberg (1995), Gilks & Berzuini (2001), Hoffman & Gelman (2014).
    """
    cat = _Catalogue(summaries, log_outlier)
    name = onp.dtype(dtype).name
    k_plan, k_is, k_move = jax.random.split(key, 3)
    with run_in(name):
        space = _Space(
            cat, cell, 0.0, onp.asarray(t_nights, float) - t_anchor, a_range,
            e_max, flux_floor, jitter, getattr(np, name),
        )  # fmt: skip
        plan = search_plan(space, k_plan) if plan is None else plan
        info = {}
        modes = search_cell(space, plan=plan, info=info)
        mixture = DefensiveMixture(
            modes, _log_prior, _prior_sampler(space.d, space.dtype),
            images=_symmetry_images(space.d), dtype=space.dtype,
        )  # fmt: skip
        n_k = len(cat.log_b)
        # ln Z of the all-outlier target: Σ b_k + ln E[ε^K] (Beta(½, ½))
        floor = float(onp.sum(cat.log_b)) + (
            math.lgamma(n_k + 0.5)
            - 0.5 * math.log(math.pi)
            - math.lgamma(n_k + 1)
        )
        ev = importance_evidence(
            _target, space, mixture, k_is, n_is, inliers_fn=_inliers,
            log_floor=floor, m_min=M_MIN,
        )  # fmt: skip
    fit = CellFit(
        cell, ev.log_z, ev.log_z_error, modes, None, ev.log_w, ev.x,
        ev.at_floor or ev.n_nonfinite > 0, ev.undersampled,
        int(info["evaluations"]), n_is, ev, plan, space,
    )  # fmt: skip
    if posterior_draws and not fit.at_floor:
        fit.draws = _draw(fit, k_move)
    return fit


def fit_jitter(
    summaries,
    log_outlier,
    domain,
    t_ab,
    *,
    n_grid=21,
    n_best=3,
):
    """The position jitter ``σ_J`` (mas) by type-II maximum likelihood.

    ``σ_J`` maximizes ``ln Σ_n S_n(σ_J)`` on a log grid from 0.1 to 10 times the
    median night position error, with ``S_n`` the position-level mixture
    likelihood of the best seed orbit of cell ``n``
    ([`position_profile`][virgil.aliases.position_profile], best peaks as
    positions): the Laplace errors of a night are overconfident wherever the
    summary model is imperfect.

    Parameters
    ----------
    summaries : sequence of NightSummary
    log_outlier : array-like
        Each night's ``b_k``.
    domain : PeriodDomain
    t_ab : (float, float)
        The anchor times (days).
    n_grid, n_best : int, optional
        Grid size and seed orbits per cell (budgets, 21 and 3).

    Returns
    -------
    float
    """
    have = [i for i, s in enumerate(summaries) if s.peaks]
    t = onp.array([summaries[i].t_ref for i in have])
    best = [summaries[i].peaks[0] for i in have]
    data = PositionData(
        t,
        onp.array([p.mean[0] for p in best]),
        onp.array([p.mean[1] for p in best]),
        onp.array([p.cov[:2, :2] for p in best]),
        t_ref=float(t.min()),
    )
    bands = alias_bands([t_ab[0], t_ab[1]], (domain.p_cut, domain.p_max))
    seeds = [
        o
        for row in position_profile(data, bands, n_best=n_best)
        for o, _ in row["orbits"]
    ]
    sigma = float(onp.median([_sigma_p(s) for s in summaries if s.peaks]))
    ln_f = float(onp.median([p.mean[4] for p in best]))
    with run_in("float64"):
        cat = _Catalogue(summaries, log_outlier).cast(np.float64)
        # each seed's (p, v) at the nights, once
        states = np.stack(
            [
                np.concatenate(
                    [
                        np.stack(o.relative(cat.t_ref)[:2], axis=1),
                        np.stack(o.relative_velocity(cat.t_ref)[:2], axis=1),
                    ],
                    axis=1,
                )
                for o in seeds
            ]
        )
        zero = np.zeros(len(cat.blocks))
        score = jax.jit(
            jax.vmap(
                lambda pv, j: _log_mixture(cat, zero, j, pv, ln_f, np.float64),
                in_axes=(0, None),
            )
        )
        grid = sigma * onp.logspace(-1, 1, n_grid)
        out = [float(logsumexp(score(states, j))) for j in grid]
    return float(grid[int(onp.argmax(out))])


# ---------------------------------------------------------------------------
# All cells
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class CellsResult:
    """The position-level fit over every cell.

    Attributes
    ----------
    domain : PeriodDomain
    anchors : (int, int)
        Indices of the anchor nights.
    t_ab : float
        ``T``, the time between the anchors (days).
    cells : list of Cell
    fits : list of CellFit
    p_pos : numpy.ndarray
        ``p(N) ∝ mass_N Ẑ_N^pos`` over the cells, summing to one; all ``nan``
        when a cell failed (never renormalized over the others).
    below_delta : numpy.ndarray
        Cells with ``p_pos < δ / N_cells``: the ones pruning would drop.
    pruned : numpy.ndarray
        Cells actually dropped (all false unless ``prune=True``).
    pruned_mass : float
        ``Σ p_pos`` over ``below_delta`` (< δ by construction): the position-level
        mass pruning drops, for Stage 4 to compare with the visibility level.
    jitter : float
        The position jitter used (mas).
    failed : numpy.ndarray
        Cells at the floor (``cell-at-floor``): the search missed the dominant
        mode. Any true entry makes ``p_pos`` undefined.
    """

    domain: PeriodDomain
    anchors: tuple
    t_ab: float
    cells: list
    fits: list
    p_pos: onp.ndarray
    below_delta: onp.ndarray
    pruned: onp.ndarray
    pruned_mass: float
    jitter: float
    failed: onp.ndarray


def position_cells(
    summaries,
    domain,
    key,
    *,
    log_outlier=None,
    outlier=None,
    jitter=None,
    prune=False,
    delta=DELTA,
    n_is=N_IS,
    max_seconds=float("inf"),
    **leg_kwargs,
):
    """The position-level fit of every cell of ``domain``.

    Chooses the anchors, makes the cells (edges at the half-integers of
    ``T / P``: edges change labels only) and runs the search and the importance
    sampling in each; ``p(N) ∝ mass_N Ẑ_N``. Resample-move draws are made only
    where ``p(N) >= δ / N_cells``. The cells are independent, so a system
    compiles once and the sets are drawn once.

    TODO: share one search and one set of importance draws across the cells
    (exact, unbiased for every ``N`` at once; design §6.7). Not implemented: it
    is unmeasured (``scripts/leg1_toys/alias_grid.py`` has not run).

    Parameters
    ----------
    summaries : sequence of NightSummary
        From [`night_summaries`][virgil.night_summaries.night_summaries].
    domain : PeriodDomain
    key : jax.random.PRNGKey
    log_outlier : array-like, optional
        Each night's ``b_k``; default computed from ``outlier``.
    outlier : OutlierPrior, optional
        The prior predictive for ``b_k`` (default tabulated for ``domain``).
    jitter : float, optional
        Position jitter (mas); default fitted by
        [`fit_jitter`][virgil.orbit_cells.fit_jitter]. Pass 0 for none.
    prune : bool, optional
        Drop cells below ``δ / N_cells`` (False: the dropped mass is recorded
        but nothing is dropped).
    delta : float, optional
        Reporting precision (default ``DELTA``).
    n_is : int, optional
        Importance draws per cell.
    max_seconds : float, optional
        Time cap on the whole run.
    **leg_kwargs
        Other arguments of [`position_leg`][virgil.orbit_cells.position_leg].

    Returns
    -------
    CellsResult

    Raises
    ------
    RuntimeError
        If ``max_seconds`` ran out.
    """
    if log_outlier is None:
        outlier = outlier or outlier_prior(domain)
        log_outlier = outlier_log_density(summaries, outlier)
    first, last = choose_anchors(summaries)
    t_all = [s.t_ref for s in summaries]
    t_a, t_b = t_all[first], t_all[last]
    t_ab = t_b - t_a
    if jitter is None:
        jitter = fit_jitter(summaries, log_outlier, domain, (t_a, t_b))
    cells = make_cells(t_a, t_b, domain)
    start, fits, plan = time.time(), [], None
    for cell, k in zip(cells, jax.random.split(key, len(cells))):
        if time.time() - start > max_seconds:
            raise RuntimeError(
                f"position_cells reached max_seconds before cell N={cell.n}."
            )
        fits.append(
            position_leg(
                summaries, log_outlier, cell, k, t_anchor=t_a, t_nights=t_all,
                jitter=jitter, n_is=n_is, posterior_draws=False, plan=plan,
                **leg_kwargs,
            )
        )  # fmt: skip
        plan = fits[0].plan
    failed = onp.array([f.at_floor for f in fits])
    log_p = onp.array([c.log_mass + f.log_z for c, f in zip(cells, fits)])
    if failed.any():
        p = onp.full(len(cells), onp.nan)
    else:
        p = onp.exp(log_p - onp_logsumexp(log_p))
    below = p < delta / len(cells)
    for f, pi, k in zip(
        fits, p, jax.random.split(jax.random.fold_in(key, 1), len(cells))
    ):
        if pi >= delta / len(cells):
            f.draws = _draw(f, k)
    return CellsResult(
        domain, (first, last), t_ab, cells, fits, p, below,
        below if prune else onp.zeros_like(below), float(p[below].sum()),
        float(jitter), failed,
    )  # fmt: skip
