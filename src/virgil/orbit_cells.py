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
failure, never an evidence.

Source: Fischler & Bolles (1981), Rousseeuw (1984), Rousseeuw & Van Driessen
(1999), Golub & Pereyra (1973), Hesterberg (1995), Gilks & Berzuini (2001),
[references](../references.md); the cells, outlier floor and search are
derived in the design note.
"""

import concurrent.futures
import dataclasses
import functools
import itertools
import math
import time
import warnings

import jax
import jax.numpy as np
import numpy as onp
from jax.scipy.special import logsumexp
from scipy import stats
from scipy.special import log_ndtr
from scipy.special import logsumexp as onp_logsumexp
from scipy.stats import qmc

from . import _orbit_coordinates as oc
from ._precision import run_in
from .aliases import _P_FLOOR, alias_bands, position_profile
from .mode_search import (
    DefensiveMixture,
    ESS_FAIL,
    M_MIN,
    MAHAL,
    W,
    N_IS,
    N_POST,
    Mode,
    distinct_modes,
    importance_evidence,
    laplace_log_evidence,
    resample_move,
)
from .night_summaries import DELTA
from .smc import tempered_smc
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
    "smc_cell",
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
GAP_DAYS = 0.5  # nights closer than this merge (design §10; OIData.epochs)
K_COH = (
    4  # ln P grid points per cycle of a set's span (> 2 brackets a minimum)
)
ZETA_E = (0.0, 0.3, 0.6)  # ζ starts: eccentricities, 8 phases each (17 points)
ZETA_PHASES = 8  # anchor mean anomalies M_a per eccentricity, from 0 to 2π
N_SET_MAX = 2000  # enumerate every set up to this; above it, every span pair
#                   (all peaks) and random interior fills of each (design §6.1)
EPS_HI = 0.3  # declared outlier fraction of the RANSAC count (design §10)
N_C = 10  # C-step cap (labels repeat in 2-3 steps on the toys)
N_VARPRO = 12  # damped Newton steps of a set's (ln P, ζ) refinement (budget)
N_SCALE = 3  # _lm steps of the scale with a jitter (budget)
N_NEWTON = 30  # damped Newton steps per C-step and for the refinement (budget)
TOL_GAIN = (
    1e-5  # Newton stops at a predicted gain below this (nats: DELTA / 100)
)
W_PRUNE = 2.0 * W  # nats below the best Laplace ln Z that drop a candidate
#                    (at least; raised to ln(n / δ) for n candidates)
N_PRE = 4  # Newton steps of the first C-step, before the candidates merge
CHUNK_C = 256  # candidates per batched C-step call (budget)
CHUNK_VP = 1024  # candidates per batched set refinement call (budget)
EDGE_STOP = W  # |w_P| beyond which the logistic chart holds less than δ of the
#                cell's ρ-interval: where a shared neighbour mode's proposal ends
N_WORKERS = (
    1  # cells run at once; 2 threads ran K = 14 two-season 1.24x faster
)
#                (design §9), at +0.3 GB: opt in with ``workers``
R0 = math.sqrt(3.0)  # radius of y_z at a mode: the mode of its chi-4 prior


# ---------------------------------------------------------------------------
# Period domain, anchors, cells
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class PeriodDomain:
    """The period prior ``log-uniform [p_cut, p_max]`` (days), declared.

    Attributes
    ----------
    overridden : bool
        True when ``p_range`` replaced the derived limits.
    binding : {"2dt_min", "p_floor", "p_range"}
        The term that sets ``p_cut``.
    a_hi, span, sigma_min : float
        The inputs of ``p_max = 2π a_hi span / sigma_min`` (mas, days, mas);
        ``sigma_min`` is ``nan`` without a night with a peak.
    n0_mass : float
        Prior mass of the long-period band ``N = 0`` (``P > span``).
    """

    p_cut: float
    p_max: float
    overridden: bool
    binding: str = "p_range"
    a_hi: float = math.nan
    span: float = math.nan
    sigma_min: float = math.nan
    n0_mass: float = math.nan


def _sigma_p(summary):
    """Position error (mas) of a night's best peak; ``inf`` if it has none."""
    if not summary.peaks:
        return onp.inf
    return float(onp.sqrt(onp.trace(summary.peaks[0].cov[:2, :2]) / 2))


def period_domain(
    summaries,
    *,
    a_range=A_RANGE,
    p_range=None,
    gap_days=GAP_DAYS,
    p_floor=_P_FLOOR,
):
    """The declared period domain ``[P_cut, P_max]`` (design §1.3).

    ``P_cut = max(2 Δt_min, p_floor)``, ``Δt_min`` the shortest gap between
    consecutive nights after merging nights less than ``gap_days`` apart:
    above it the closest pair of nights advances by under half a cycle, so the
    regular-sampling alias lies outside the domain. There is no within-night
    bound; the per-exposure target and the exact-projection summaries model
    the motion within a night. A prior choice, not a Nyquist limit.
    ``P_max = 2π a_hi · span / σ_min``, with ``σ_min`` the smallest night
    position error: beyond it the motion over the span is below that error.

    Parameters
    ----------
    summaries : sequence of NightSummary
        One per night.
    a_range : (float, float)
        Prior range of the semimajor axis (mas); ``a_hi`` is the field.
    p_range : (float, float), optional
        Override ``(P_cut, P_max)``.
    gap_days : float, optional
        Nights closer than this merge (default 0.5 d, as ``OIData.epochs``).
    p_floor : float, optional
        The only other lower bound (days).

    Returns
    -------
    PeriodDomain
        With the binding term of ``P_cut``, the inputs of ``P_max`` and the
        prior mass of the ``N = 0`` band.
    """
    times = onp.array([s.t_ref for s in summaries])
    sigma = onp.array([_sigma_p(s) for s in summaries])
    span = float(onp.ptp(times)) if times.size else math.nan
    sigma_min = float(sigma.min()) if onp.isfinite(sigma).any() else math.nan
    if p_range is not None:
        lo, hi = map(float, p_range)
        overridden, binding = True, "p_range"
    else:
        if onp.isfinite(sigma).sum() < 2:
            raise ValueError(
                "Need two nights with a peak, or pass p_range, to set the "
                "period domain."
            )
        # nights: mean time of each group of times at most gap_days apart
        # (a new night starts on a gap > gap_days, as OIData.epochs)
        t = onp.sort(times)
        label = onp.r_[0, onp.cumsum(onp.diff(t) > gap_days)]
        nights = onp.bincount(label, weights=t) / onp.bincount(label)
        if nights.size < 2:
            raise ValueError(
                "Need two nights more than gap_days apart, or pass p_range."
            )
        two_dt = 2.0 * float(onp.min(onp.diff(nights)))
        lo, binding = (
            max(two_dt, p_floor),
            ("2dt_min" if two_dt >= p_floor else "p_floor"),
        )
        hi = 2 * math.pi * a_range[1] * span / sigma_min
        overridden = False
    if not 0 < lo < hi:
        raise ValueError(f"The period domain [{lo:.4g}, {hi:.4g}] d is empty.")
    n0 = math.log(hi / span) / math.log(hi / lo) if 0 < span < hi else 0.0
    return PeriodDomain(
        lo, hi, overridden, binding, float(a_range[1]), span, sigma_min,
        min(1.0, max(0.0, n0)),
    )  # fmt: skip


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

    def log_density(self, mean, velocity=True, sigma_f=None):
        """``ln π(p, v) π(f)`` at ``mean = (dra, ddec, vra, vdec, ln f)``,
        a density in ``(p, v, ln f)``; ``-inf`` outside the flux range. With
        ``velocity=False`` the velocity is integrated out: the density in
        ``(p, ln f)`` (``mean[2:4]`` unused). With ``sigma_f`` the flux box
        is integrated over a Gaussian ``ln f`` error of that width
        (:meth:`log_flux_box`) instead of evaluated at ``mean[:, 4]``."""
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
        if sigma_f is None:
            in_f = (mean[:, 4] >= self.ln_f_floor) & (mean[:, 4] <= 0.0)
            log_f = onp.where(in_f, -math.log(-self.ln_f_floor), -onp.inf)
        else:
            log_f = self.log_flux_box(mean[:, 4], sigma_f)
        return log_pv + log_f

    def log_flux_box(self, m_f, sigma_f):
        """``ln E[π(ln f)]`` for ``ln f ~ N(m_f, σ_f²)``: the uniform flux
        prior on ``[ℓ, 0]`` (``ℓ = ln_f_floor``) integrated over the Gaussian
        measurement error, ``[Φ(-m/σ) - Φ((ℓ-m)/σ)] / |ℓ|``.
        Finite for every ``m_f``, including ``m_f > 0``."""
        m_f = onp.asarray(m_f, float)
        sigma_f = onp.asarray(sigma_f, float)
        hi = -m_f / sigma_f  # Φ(hi) - Φ(lo), hi > lo
        lo = (self.ln_f_floor - m_f) / sigma_f
        # reflect when both limits sit in the upper tail, where the two
        # log-CDFs are ~0 and the difference would cancel
        flip = lo > 0
        a = log_ndtr(onp.where(flip, -lo, hi))
        b = log_ndtr(onp.where(flip, -hi, lo))
        with onp.errstate(divide="ignore"):
            log_diff = a + onp.log1p(-onp.exp(b - a))
        return log_diff - math.log(-self.ln_f_floor)


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
                + float(
                    prior.log_density(
                        p.mean,
                        velocity=len(dims) == 5,
                        sigma_f=math.sqrt(cov[4, 4]),
                    )[0]
                )
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
        # frame offsets and the (p̂, v̂) projection, padded with zeros
        f = max(s.n_frames for s in summaries)
        self.offsets = onp.zeros((k, f))
        self.proj = onp.zeros((k, 2, f))
        for i, s in enumerate(summaries):
            self.offsets[i, : s.n_frames] = s.frame_offsets()
            self.proj[i, :, : s.n_frames] = s.projection()
        self.single = self.t_ss <= 0  # one frame: the velocity is the state's
        self.log_b = onp.broadcast_to(
            onp.asarray(log_outlier, float), (k,)
        ).copy()
        self.n_eps = (k + 2) // 2  # ⌈(K + 1) / 2⌉: exact for degree K

    _ARRAYS = (
        "mean", "hess", "n", "chi2", "loglike", "valid", "t_ss", "n_frames",
        "t_ref", "log_b", "offsets", "proj", "single",
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


def _peak_terms(cat, ln_s, jitter, state, dtype, extra=None):
    """``ln L_j(s) + ½ ln(|Σ_data| / |Σ_total|) - ½ dᵀ Σ_total⁻¹ d`` per peak,
    ``(K, J)``; ``-inf`` for padding. ``state`` is ``(K, 5)``. With ``extra``
    ``(K, 5, 5)``, the state's covariance, it is added to ``Σ_total``: the
    peak term marginalised over the state's uncertainty (predictive term)."""
    cov, total, inv = _peak_precision(cat, ln_s, jitter, dtype)
    if extra is not None:
        total = total + extra[:, None]
        inv = np.linalg.inv(total)
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

    def night_states(self, nights, period, ti, zeta):
        """The summaries' prediction ``(p̂, v̂)`` of ``ti`` at ``nights``
        (indices): ``(len(nights), 4)``.

        The exact least-squares projection of the orbit's positions at the
        stored frame offsets onto linear motion, the fit the summaries made
        (design §5); it removes the curvature bias of the instantaneous state.
        A one-frame night has no velocity fit and takes the state's.
        """
        cat = self.cat
        off, proj = cat.offsets[nights], cat.proj[nights]
        m, f = off.shape
        s = self.states_at(
            (self.t_k[nights][:, None] + off).reshape(-1), period, ti, zeta
        ).reshape(m, f, 4)
        fit = np.einsum("mcf,mfx->mcx", proj, s[..., :2])
        v = np.where(cat.single[nights][:, None], s[:, 0, 2:], fit[:, 1])
        return np.concatenate([fit[:, 0], v], axis=1)

    def states(self, period, ti, zeta):
        """``(p̂, v̂)`` of ``ti`` for every night: ``(K, 4)``."""
        return self.night_states(np.arange(len(self.t_k)), period, ti, zeta)

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

    The ``+ 3 ln |y_z|`` is the Cartesian volume element of the chi-4 radial
    prior, so the Laplace ``ln Z`` of this objective is that of the target: the
    radial factor is ``√π e^-3/2 r0³`` against the exact ``2`` (``r0 = √3``),
    an offset of 0.03 nats, common to every mode. A [`Mode`][virgil.mode_search.Mode]
    records the objective at the mode, so that its Laplace ``ln Z`` and the
    pruning scores are on one footing.
    """
    return _target(x, space) + 3.0 * np.log(np.linalg.norm(x[_Z]))


def _labels(x, space, extra=None):
    """Each night's best peak, or -1 (outlier) where the floor is higher.

    At the prior mean ``ε̄ = ½`` the inlier and outlier weights ``1 - ε̄`` and
    ``ε̄`` are equal, so the comparison is of ``ℓ̃_k`` and ``b_k``. With
    ``extra``, the predictive term of ``_peak_terms``.
    """
    with run_in(onp.dtype(space.dtype).name):
        c = space.decode(x)
        pv = space.states(c["period"], oc.z_to_ti(c["z"]), c["zeta"])
        state = np.concatenate(
            [pv, np.full((pv.shape[0], 1), c["ln_f"], space.dtype)], axis=1
        )
        terms = _peak_terms(
            space.cat, c["ln_s"], space.jitter, state, space.dtype, extra
        )
        j = np.argmax(terms, axis=1)
        best = np.take_along_axis(terms, j[:, None], axis=1)[:, 0]
        return np.where(best > space.cat.log_b, j, -1)


def _peak_state(x, space):
    """``(decoded x, (K, 5) state)`` of the chart vector ``x``."""
    c = space.decode(x)
    pv = space.states(c["period"], oc.z_to_ti(c["z"]), c["zeta"])
    state = np.concatenate(
        [pv, np.full((pv.shape[0], 1), c["ln_f"], space.dtype)], axis=1
    )
    return c, state


def _scale_sums(cat, state):
    """Per peak and block ``χ² + q`` and ``n``, ``(K, J, G)`` each: ``q`` is
    the peak's Laplace quadratic ``dᵀ H_g d`` at the state."""
    d = state[:, None, :] - cat.mean
    q = np.einsum("kja,kjgab,kjb->kjg", d, cat.hess, d)
    return cat.chi2 + q, cat.n


def _initial_labels(x, space):
    """Each night's best peak where it beats the floor at the night's own best
    scale, else -1: ``I₀ = {k : max_s ℓ_k(x, s) > b_k}`` (the comparison ``b_k``
    makes; more permissive than any fixed ``s``). The own scale is the closed
    form of ``_scale_step`` per peak, at zero jitter."""
    cat = space.cat
    if not cat.rescalable:
        return _labels(x, space)
    _, state = _peak_state(x, space)
    a, n = _scale_sums(cat, state)
    lim = math.log(S_MAX)
    ln_s = np.clip(0.5 * np.log(np.maximum(a, 1e-300) / n), -lim, lim)
    peak = cat.loglike - np.sum(
        n * ln_s + 0.5 * a * np.exp(-2 * ln_s) - 0.5 * cat.chi2, axis=-1
    )
    peak = np.where(cat.valid, peak, -1e30)
    j = np.argmax(peak, axis=1)
    best = np.take_along_axis(peak, j[:, None], axis=1)[:, 0]
    return np.where(best > cat.log_b, j, -1)


def _scale_step(x, space, lab):
    """``x`` with the inlier scale at its maximum given the labels ``lab``.

    Per block, ``ŝ² = Σ_I (χ² + q) / Σ_I n`` with ``ŝ`` clipped to ``[1/S_MAX, S_MAX]``
    (where the prior is flat); with jitter, ``N_SCALE`` steps of ``_lm`` in
    ``ln s`` (trust radius one e-fold) from there. The labels fixed, this does not decrease the labelled objective.
    Data that do not rescale, or no inlier, keep ``s = 1``.

    Source: Celeux & Govaert (1992), [references](../references.md#celeux1992).
    """
    cat = space.cat
    if not cat.rescalable:
        return x
    _, state = _peak_state(x, space)
    j = np.maximum(lab, 0)
    k = np.arange(lab.shape[0])
    a, n = _scale_sums(cat, state)
    on = (lab >= 0)[:, None]
    num = np.sum(np.where(on, a[k, j], 0.0), axis=0)
    den = np.sum(np.where(on, n[k, j], 0.0), axis=0)
    lim = math.log(S_MAX)
    ln_s = np.where(
        den > 0,
        np.clip(
            0.5 * np.log(np.maximum(num, 1e-300) / np.maximum(den, 1e-300)),
            -lim,
            lim,
        ),
        0.0,
    )

    def objective(ls):
        t = _peak_terms(cat, ls, space.jitter, state, space.dtype)
        return np.sum(np.where(lab >= 0, t[k, j], 0.0))

    ln_s = jax.lax.cond(
        space.jitter > 0,
        lambda ls: np.clip(
            _lm(objective, ls, (), N_SCALE, step_max=1.0)[0], -lim, lim
        ),
        lambda ls: ls,
        ln_s,
    )
    u = np.clip(0.5 * (ln_s / lim + 1.0), 1e-6, 1.0 - 1e-6)  # decode's inverse
    return x.at[9:].set(np.log(u) - np.log1p(-u))


def _relabel(x, space, lab):
    """Scale step given ``lab``, then the labels at that scale: ``(x, labels)``."""
    x = _scale_step(x, space, lab)
    return x, _labels(x, space)


def _lm(f, x, args, n_iter, step_max=10.0):
    """Saddle-free Levenberg–Marquardt ascent of ``f(x, *args)``.

    Newton with ``|eigenvalues|`` of the Hessian plus a damping ``λ`` that
    falls on an accepted step and rises on a rejected one (a step is accepted
    only if ``f`` is finite and rises). Stops after ``n_iter`` steps or when
    the predicted gain ``g·step`` is below ``TOL_GAIN``. vmap-able.
    Returns ``(x, f(x), capped)``, ``capped`` true if the step cap ended it.
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
    i, x, fx, _, gain = jax.lax.while_loop(
        lambda c: (c[0] < n_iter) & (c[4] > TOL_GAIN), body, init
    )
    return x, fx, (i >= n_iter) & (gain > TOL_GAIN)


def _design(space, nights, v):
    """``states`` at a set's three nights as a linear map of TI′: ``(3, 4, 4)``.

    ``v = (w_P, y_ζ[2])`` is the chart's period and ζ.
    """
    period, zeta = space.period(v[0]), space.zeta(v[1:3])
    zero = np.zeros(4, space.dtype)
    return jax.jacfwd(lambda ti: space.night_states(nights, period, ti, zeta))(
        zero
    )


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


def _prediction_cost(gg, nights, w, chol, valid):
    """``λ_k = λ_max(Σ_k⁻¹ C_k)`` per night ``(K,)`` for a pair solved on
    ``nights`` (precisions ``w``) at the designs ``gg`` ``(K, 4, 4)``:
    ``C_k = G_k F⁻¹ G_kᵀ``, ``F`` the pair's Fisher information; the
    largest over the night's peaks (``chol`` ``(K, J, 4, 4)``, ``Σ_k⁻¹ = L Lᵀ``).
    Scaling every ``Σ`` by ``c²`` leaves it unchanged."""
    gs = gg[nights]
    f = np.einsum("mia,mij,mjb->ab", gs, w, gs)
    f = f + np.finfo(gg.dtype).eps * np.trace(f) * np.eye(4, dtype=gg.dtype)
    c = np.einsum("kia,ab,kjb->kij", gg, np.linalg.inv(f), gg)
    m = np.einsum("kjai,kab,kjbc->kjic", chol, c, chol)
    lam = np.linalg.eigvalsh(m)[..., -1]
    return np.where(valid, lam, 0.0).max(axis=1)


@jax.jit
def _grid_stage(space, plan):
    """Every set's residual on a ``ln P`` grid of the cell, at every ζ start.

    The nodes ``plan["grid"]`` (in ``[0, 1]``) are equally spaced in
    ``ν = 1 / P`` over the cell; their number is sized by ``search_plan`` to
    the widest set. Returns the residuals ``(S, Z, n)``,
    their local minima along the grid (grid ends count) and the grid's ``ln P``.
    """
    nu_lo, nu_hi = 1.0 / np.exp(space.ln_hi), 1.0 / np.exp(space.ln_lo)
    nu = nu_lo + plan["grid"] * (nu_hi - nu_lo)
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
    is_min = (r <= left) & (r <= right) & np.isfinite(r)
    if len(plan["pairs"]) == 0:
        return r, is_min, np.log(period), np.zeros((), int)
    # A span pair solved on its own nights predicts night k between them with
    # the error cost ½ λ_k χ²₄(1 - δ) in ℓ_k at most, with probability 1 - δ
    # (_prediction_cost). It covers its span when that is below the night's
    # height Δ_k over its floor for every interior night: its triplet fills
    # then add nothing (design §6.11, fix 4).
    chi4 = stats.chi2.isf(DELTA, 4)
    chol = np.linalg.cholesky(plan["wk"])  # (K, J, 4, 4), precision = L Lᵀ

    def pair_covers(nights, w, inner, gg):
        lam = _prediction_cost(gg, nights, w, chol, space.cat.valid)
        return (
            ((0.5 * chi4 * lam < plan["gap"]) | ~inner).all()
        ) & inner.any()

    pr = plan["pairs"]
    covered = jax.vmap(
        lambda nt, w, inner: jax.vmap(
            jax.vmap(lambda gg: pair_covers(nt, w, inner, gg))
        )(g)
    )(plan["nights"][pr], plan["w"][pr], plan["inner"])  # (Q, Z, n)
    skip = (plan["parent"] >= 0)[:, None, None] & covered[plan["parent"]]
    return r, is_min & ~skip, np.log(period), (is_min & skip).sum()


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
            z / np.maximum(np.linalg.norm(z), 1e-30) * R0,
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
    the ``χ²`` cut applies to), ``v``, the Hessian of ``R / 2`` in ``v`` and
    whether the step cap ended it."""

    def one(si, v):
        args = (plan["nights"][si], plan["w"][si], plan["mean"][si], space)
        v, fv, capped = _lm(_varpro_objective, v, args, n_iter)
        ti = _solve_set(
            _design(space, args[0], v), args[2], args[1], space.s_w2
        )[1]
        x = _chart_from(space, v, ti, plan["lnf"][si])
        x, lab = _relabel(x, space, _initial_labels(x, space))
        h = -jax.hessian(_varpro_objective)(v, *args)
        return (
            x, lab, -2.0 * fv - ti @ ti / space.s_w2, v, h,
            capped,
        )  # fmt: skip

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
    target (TI′ and ``ln f`` solved linearly), then relabel at the result by
    each night's predictive term (design §6.14): the labels only seed the
    search, the target is unchanged.
    Returns ``v``, the labels, the chart vector, the residual, the Hessian
    of ``R / 2`` in ``v`` and whether the step cap ended it."""

    def one(v, lab):
        args = (lab, plan["wk"], plan["mk"], space)
        v, fv, capped = _lm(_labelled_objective, v, args, n_iter)
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
        x = _scale_step(x, space, lab)
        # C_k = J_k H⁻¹ J_kᵀ with H the posterior curvature of the labelled
        # fit in the full chart (w_P, y_ζ, TI′): the likelihood's at the
        # profiled scale (the largest if several, which keeps more nights),
        # the chart prior's in (w_P, y_ζ), the ridge on TI′. J_k is the
        # Jacobian of night k's (p, v); zero in ln f. Saddle-free.
        idx = np.arange(k)
        mk = plan["mk"][idx, j]

        def states(th):
            return np.einsum("kia,a->ki", _design(space, idx, th[:3]), th[3:])

        def likelihood(th):
            r = mk - states(th)
            return -0.5 * np.einsum("ki,kij,kj->", r, w, r)

        def log_prior(vv):
            return _log_prior(x.at[0].set(vv[0]).at[6:8].set(vv[1:3]))

        th = np.concatenate([v, ti])
        hf = -jax.hessian(likelihood)(th) * np.exp(
            -2.0 * np.max(space.decode(x)["ln_s"])
        )
        hf = hf.at[:3, :3].add(-jax.hessian(log_prior)(v))
        hf = hf + np.diag(np.r_[np.zeros(3), np.ones(4) / space.s_w2])
        e, u = np.linalg.eigh(0.5 * (hf + hf.T))
        e = np.abs(e)
        c = np.einsum(
            "kia,ab,kjb->kij",
            jax.jacfwd(states)(th),
            (u / np.maximum(e, np.finfo(space.dtype).eps * e.max())) @ u.T,
            jax.jacfwd(states)(th),
        )
        extra = np.zeros((k, 5, 5), space.dtype).at[:, :4, :4].set(c)
        new = _labels(x, space, extra)
        h = -jax.hessian(_labelled_objective)(v, *args)
        return v, new, x, -2.0 * fv, h, capped

    return jax.vmap(one)(v, lab)


@functools.partial(jax.jit, static_argnames=("n_iter",))
def _refine(space, x, n_iter):
    """Newton on the label-summed objective from one ``x``: the result, minus
    its Hessian, the objective and the expected inlier count. One candidate per
    call: a batch would run every member as long as the slowest."""
    x, _, _ = _lm(_objective, x, (space,), n_iter)
    return (
        x,
        -jax.hessian(_objective)(x, space),
        _objective(x, space),
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
    """Indices of the candidates within ``max(W_PRUNE, ln(n / δ))`` of the best
    Laplace ``ln Z`` (``n`` candidates scored).

    A dropped candidate carries at most ``e^-threshold`` of the best mode's
    mass if its score is accurate, so the ``n`` of them together less than
    ``δ``. The score is at the current ``x`` (not its mode): it can sit tens of
    nats from the refined one, but a candidate scored that far below the best
    is a duplicate of a better mode as far as measured (a K = 8 cell kept 3 of
    95, the dropped ones refined onto the best mode: ``test_the_prune_...``
    compares the evidence with and without it). A non-finite score drops the
    candidate.
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
    margin = max(W_PRUNE, math.log(len(x) / DELTA))
    return onp.flatnonzero(score >= score.max() - margin)


def _chunks(n, size):
    """Slices of ``range(n)`` of ``size`` (the last is padded by the caller)."""
    return [slice(i, min(i + size, n)) for i in range(0, n, size)]


def _padded(a, size):
    """``a`` with its first row repeated to ``size`` rows."""
    return onp.concatenate([a, onp.repeat(a[:1], size - len(a), axis=0)])


def search_plan(space, key=None, nu_width=None):
    """The minimal sets of a system, as arrays (the same for every cell).

    Pairs of nights (positions and velocities: 8 equations for the 4 TI′ at
    fixed ``(P, ζ)``) and triplets (positions only: 6), with every combination
    of their peaks. Above ``N_SET_MAX`` sets the triplets are the random
    interior fills of every pair (per pair of nights ``(a, b)`` with a night
    between them, ``⌈ln δ / ln(1 - w)⌉`` triplets ``(a, c, b)``, ``c`` an
    interior night with a peak, the three peaks random; design §6.1; drawn
    from ``key``) and the pairs stay whole, so the pair that spans the
    inliers (it pins the period to ``σ / (a Δ)``) is always there.

    The ``ln P`` grid has ``⌈K_COH Δ ν_w⌉ + 1`` nodes (at least ``K_COH + 1``)
    equally spaced in ``ν = 1 / P`` over a cell, ``Δ`` the largest set span and
    ``ν_w`` the largest cell width in ``ν`` (``1 / T``; default the width of
    ``space.cell``): at least ``K_COH`` per cycle of the widest set.

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
    if nu_width is None:
        nu_width = 1.0 / space.cell.p_lo - 1.0 / space.cell.p_hi
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
    parent = {}  # fill triplet -> the set of its span pair
    for m in (2, 3):
        for nights in itertools.combinations(range(len(peaks)), m):
            if all(len(peaks[k]) for k in nights):
                for pk in itertools.product(*[peaks[k] for k in nights]):
                    sets[m].append((nights, pk))
    if sum(len(v) for v in sets.values()) > N_SET_MAX:
        n_fill = math.ceil(math.log(DELTA) / math.log(1.0 - (1.0 - EPS_HI)))
        fills = set()
        for a, b in sorted({nt for nt, _ in sets[2]}):
            inner = [c for c in range(a + 1, b) if len(peaks[c])]
            for _ in range(n_fill if inner else 0):
                c = inner[rng.integers(len(inner))]
                pk = tuple(int(rng.choice(peaks[k])) for k in (a, c, b))
                fills.add(((a, c, b), pk))
        sets[3] = sorted(fills)
        index = {st: i for i, st in enumerate(sets[2])}
        parent = {
            st: index[((st[0][0], st[0][2]), (st[1][0], st[1][2]))]
            for st in sets[3]
        }
    chosen = sets[2] + sets[3]
    s = len(chosen)
    pairs = sorted(set(parent.values()))
    local = {q: i for i, q in enumerate(pairs)}
    parent_of = onp.array(
        [local[parent[st]] if st in parent else -1 for st in chosen], int
    )
    inner = onp.zeros((len(pairs), len(peaks)), bool)
    for i, q in enumerate(pairs):
        a, b = chosen[q][0]
        inner[i, a + 1 : b] = [len(peaks[c]) > 0 for c in range(a + 1, b)]
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
    t_k = onp.asarray(space.t_k, float)
    span = max((t_k[max(nt)] - t_k[min(nt)] for nt, _ in chosen), default=0.0)
    n_grid = max(K_COH + 1, math.ceil(K_COH * span * nu_width) + 1)
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
    # Δ_k: night k's best peak over its outlier floor, at the error scale that
    # maximizes it (9 values of ln s over ±ln S_MAX if free, else s = 1)
    heights = []
    for ln_s in (
        onp.linspace(-1.0, 1.0, 9) * math.log(S_MAX)
        if cat.rescalable
        else [0.0]
    ):
        cov, total, _ = _peak_precision(
            cat, np.full(len(cat.blocks), ln_s, dt), space.jitter, dt
        )
        scaled = cat.loglike - np.sum(
            cat.n * ln_s + 0.5 * cat.chi2 * (math.exp(-2 * ln_s) - 1.0),
            axis=-1,
        )
        log_det = 0.5 * (
            np.linalg.slogdet(cov)[1] - np.linalg.slogdet(total)[1]
        )
        heights.append(
            np.where(cat.valid, scaled + log_det, -1e30).max(axis=1)
        )
    gap = np.max(np.stack(heights), axis=0) - cat.log_b
    return {
        "nights": np.asarray(nights), "w": np.asarray(w, dt), "mean": np.asarray(mean, dt),
        "lnf": np.asarray(lnf, dt), "cut": onp.asarray(cut), "y0": np.asarray(y0, dt),
        "grid": np.linspace(0.0, 1.0, n_grid, dtype=dt),
        "wk": np.asarray(w_pv, dt), "mk": np.asarray(mean_all[..., :4], dt),
        "wf": np.asarray(w_f, dt), "fm": np.asarray(mean_all[..., 4], dt),
        "parent": np.asarray(parent_of), "pairs": np.asarray(pairs, int),
        "inner": np.asarray(inner),
        "gap": np.asarray(gap, dt),
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


# deliberate: polishes each C-step survivor on the label-summed target and takes its saddle-free Laplace covariance, a whole step that keeps search_cell under the function-length gate
def _laplace_modes(space, x, info):
    """The ``Mode`` of each start in ``x``: Newton on the label-summed target,
    saddle-free Hessian covariance; adds the cost to ``info["evaluations"]``."""
    newton = 4.0 + 3.0 * space.d  # one Newton step, in likelihood evaluations
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
    return modes


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

    ``info``, a dict, receives the counts of each step, the runs ended by the
    step cap in each of the VarPro, pre-pass and C-step stages
    (``varpro_capped`` of ``varpro_runs``, ``pre_capped`` of ``pre_runs``,
    ``c_capped`` of ``c_runs``; only the C-steps are meant to converge) and
    ``evaluations``, the likelihood-equivalent cost (a gradient is 3, a Hessian 3 per dimension).

    Source: Fischler & Bolles (1981), Rousseeuw (1984), Rousseeuw & Van Driessen
    (1999), Golub & Pereyra (1973).
    """
    plan = search_plan(space, key) if plan is None else plan
    info = {} if info is None else info
    n_sets, d, k = len(plan["nights"]), space.d, len(space.cat.log_b)
    info.update(sets=n_sets, candidates=0, survivors=0, unique=0, modes=0)
    info.update(varpro_runs=0, varpro_capped=0, pre_runs=0, pre_capped=0)
    info.update(c_runs=0, c_capped=0)
    info["evaluations"] = 0.0
    if n_sets == 0:
        return []
    r, is_min, ln_p, covered = _grid_stage(space, plan)
    s_i, z_i, p_i = onp.nonzero(onp.asarray(is_min))
    info["candidates"] = len(s_i)
    info["covered"] = int(covered)
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
    x, lab, rr, v, h, capped = (
        onp.concatenate([o[i] for o in out]) for i in range(6)
    )
    info["varpro_runs"], info["varpro_capped"] = len(x), int(capped.sum())
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
    cap = onp.zeros(len(v), bool)
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
            vs, ls, xs, rs, hs, cs = (onp.asarray(a)[: len(idx)] for a in res)
            changed.append(idx[(ls != lab[idx]).any(1)])
            v[idx], lab[idx], x[idx], r2[idx], h2[idx] = vs, ls, xs, rs, hs
            cap[idx] = cs
            steps += len(idx) * n_iter
        stage = "pre" if n_pass == 0 else "c"
        info[f"{stage}_runs"] += len(active)
        info[f"{stage}_capped"] += int(cap[active].sum())
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
            cap = onp.zeros(len(v), bool)
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
    modes = distinct_modes(_laplace_modes(space, x, info))
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
        Never report the cell's ``log_z`` as an evidence. It fails the *system*
        only if the cell could matter: see ``CellsResult.failed``.
    undersampled : bool
        ``cell-undersampled``: the ESS is below ``ESS_TARGET``; information only.
    unreliable : bool
        The ESS is below ``ESS_FAIL``: the estimate is a few draws and its
        error meaningless (a search that found only a partial-outlier mode had
        ESS about 1 and an evidence 100 nats low). It fails the *system* only
        if the cell could matter: see ``CellsResult.failed``. The error
        ``log_z_error`` is a lower bound (first order in ``1/ESS``) and is not
        propagated into ``p_pos``.
    nonfinite : bool
        Some importance draws had a non-finite target: the cell *failed*.
    smc : virgil.smc.SMCResult or None
        The tempered SMC from the search mixture of a cell whose importance
        sampling was unreliable ([`smc_cell`][virgil.orbit_cells.smc_cell]);
        ``log_z``, ``log_z_error``, ``unreliable`` and ``draws`` then come
        from it. ``None`` for every other cell.
    smc_seconds : float
        The SMC's wall time (0 if it did not run).
    search_evaluations, is_evaluations : int
        Likelihood-equivalent evaluations of the search and the sampling.
    move : virgil.mode_search.Move or None
        The resample-move of ``draws`` (HMC acceptance and step size).
    shared : list of Mode
        Neighbour modes that straddle a shared edge, in this cell's chart: extra
        components of the mixture (not in ``modes``).
    info : dict
        The search's counts (``c_capped``, ``c_runs``, ...:
        [`search_cell`][virgil.orbit_cells.search_cell]).
    log_l_sup : float
        The largest ``ln L`` evaluated in the cell (at every found mode and
        importance draw): a *lower* bound on the cell's ``sup ln L``, which
        itself bounds ``ln Z_N`` (the cell prior is normalized). It is a bound
        on ``Z_N`` only if the search found the cell's top mode (design
        §6.10); ``nan`` (not set) counts as mattering.
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
    unreliable: bool = False
    nonfinite: bool = False
    move: object = dataclasses.field(repr=False, default=None)
    shared: list = dataclasses.field(repr=False, default_factory=list)
    info: dict = dataclasses.field(repr=False, default_factory=dict)
    log_l_sup: float = float("nan")
    smc: object = dataclasses.field(repr=False, default=None)
    smc_seconds: float = 0.0

    @property
    def failed(self):
        """The cell failed on its own: ``at_floor`` or ``nonfinite``."""
        return bool(self.at_floor or self.nonfinite)


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
    """Resample-move posterior draws of a fit, each draw with the mass of its
    mixture component (§6.15). The
    draws of a cell with a finished SMC are its particles, resampled by their
    weights (§6.15)."""
    if fit.smc is not None and not fit.smc.stopped:
        w = onp.asarray(fit.smc.weights, float)
        u = float(jax.random.uniform(key))
        pos = (u + onp.arange(N_POST)) / N_POST
        idx = onp.minimum(
            onp.searchsorted(onp.cumsum(w / w.sum()), pos), len(w) - 1
        )
        return onp.asarray(fit.smc.particles["x"])[idx]
    cov, mixture = None, None
    if fit.modes:  # each draw moves with the mass of its mixture component
        mixture = _mixture(fit.space, [*fit.modes, *fit.shared])
        cov = mixture.masses(_prior_scale(fit.space.d))
    with run_in(onp.dtype(fit.space.dtype).name):
        move = resample_move(
            _target, fit.space, fit.evidence, key, cov=cov, mixture=mixture
        )
    fit.move = move
    if move.stuck:
        warnings.warn(
            f"cell {fit.cell}: HMC accepted {move.acceptance:.3g} of moves "
            "(< 1 per particle): the draws are not rejuvenated",
            RuntimeWarning,
            stacklevel=2,
        )
    return move.draws


def _mixture(space, modes):
    """The defensive mixture of ``modes`` (and the images of the target's
    symmetries)."""
    return DefensiveMixture(
        modes, _log_prior, _prior_sampler(space.d, space.dtype),
        images=_symmetry_images(space.d), dtype=space.dtype, dim=space.d,
    )  # fmt: skip


def _cell_evidence(space, modes, key, n_is):
    """Defensive importance sampling of a cell on ``modes`` (and the images of
    the target's symmetries)."""
    mixture = _mixture(space, modes)
    n_k = len(space.cat.log_b)
    # ln Z of the all-outlier target: Σ b_k + ln E[ε^K] (Beta(½, ½))
    floor = float(onp.sum(space.cat.log_b)) + (
        math.lgamma(n_k + 0.5) - 0.5 * math.log(math.pi) - math.lgamma(n_k + 1)
    )
    return importance_evidence(
        _target, space, mixture, key, n_is, inliers_fn=_inliers,
        log_floor=floor, m_min=M_MIN,
    )  # fmt: skip


SMC_ERROR = ESS_FAIL**-0.5  # sd(ln Ẑ) ≈ ESS^(-1/2) at ESS_FAIL (§6.15)
SMC_STEPS = (5, 5, 10, 10)  # HMC steps per stage of each replicate (§6.15)
Z_MIXING = 2.326  # z(0.99): the per-cell false-flag rate of the mixing check


def _smc_error(res):
    """``√max(V_LW, V_floor, V_rep)`` of the replicated SMC: the genealogy,
    ESS-floor and replicate variances of ``ln Ẑ`` (§6.15)."""
    terms = [
        res.log_z_variance,
        res.log_z_floor_variance,
        res.log_z_rep_variance,
    ]
    valid = [v for v in terms if v == v and v > 0]
    return math.sqrt(max(valid)) if valid else math.nan


def _smc_unreliable(res):
    """A SMC run that stopped, has no usable genealogy error, whose error
    (``_smc_error``) is above the sd that ``ESS_FAIL`` implies, or whose
    doubled-mutation replicates sit significantly above the others (the
    mixing check, ``Z_MIXING``)."""
    return bool(
        res.stopped
        or res.variance_degenerate
        or not _smc_error(res) <= SMC_ERROR
        or res.mixing_z > Z_MIXING
    )


def _prior_scale(d):
    """The variance of each chart coordinate under ``_log_prior``: the
    logistic's π²/3 for ``w_P, w_a, w_f, w_s``, 1 for ``y_z`` (standard normal)
    and for ``y_ζ`` (no variance: its median ``|y|²`` is 1)."""
    v = onp.ones(d)
    for i in (0, 5, 8, *range(9, d)):
        if i < d:
            v[i] = math.pi**2 / 3.0
    return v


def _smc_logprior(p, mixture, space):
    return mixture.log_components(p["x"])[p["c"]]


def _smc_loglike(p, mixture, space):
    return _target(p["x"], space) - mixture.log_q(p["x"])


def _smc_label_gibbs(key, p, logdensity, mixture, space):
    """``c | x ~ q_c(x) / q(x)``: the label's conditional, so any ``β``."""
    return {
        **p,
        "c": jax.random.categorical(key, mixture.log_components(p["x"])),
    }, None


def _run_smc(space, mixture, key, n, max_seconds, prior_scale):
    """Tempered SMC from ``mixture`` (normalized, sampleable) to the target,
    ``π_β ∝ q^(1-β) (prior·L)^β``, on particles ``(x, c)`` augmented with their
    mixture component: ``π̃_β(x, c) ∝ q_c(x) (p̃/q)^β`` has the marginal ``π_β``
    and the normalizer ``Z_β``, so ``Ẑ`` still estimates ``Z_N``. HMC runs on
    ``x`` given ``c`` with the mass of component ``c``; a Gibbs step redraws
    ``c`` (§6.15). ``len(SMC_STEPS)`` replicates of ``n`` particles. The
    mixture and the space are arguments of module-level functions, so the
    compiled stage is reused by every cell of the same shape."""
    with run_in(onp.dtype(space.dtype).name):
        return tempered_smc(
            _smc_logprior, _smc_loglike, mixture.sample_labelled, key=key,
            n_particles=n, replicates=len(SMC_STEPS), steps=SMC_STEPS,
            max_seconds=max_seconds, moves=(_smc_label_gibbs,),
            masses=mixture.masses(prior_scale), args=(mixture, space),
        )  # fmt: skip


def smc_cell(fit, key, *, max_seconds=float("inf")):
    """Replace a cell's importance-sampling evidence by a tempered SMC from its
    search-built defensive mixture (design §6.15).

    ``fit`` is changed in place: ``log_z``, ``log_z_error``, ``unreliable``,
    ``undersampled`` and ``smc``. The SMC's key must be independent of the
    cell's importance draws. A run that stops (``max_seconds`` or the stage
    cap) keeps the importance ``ln Ẑ`` and is ``unreliable``. ``at_floor``
    keeps reading the importance draws.

    Source: Del Moral, Doucet & Jasra (2006), Hesterberg (1995).
    """
    start = time.time()
    res = _run_smc(
        fit.space, _mixture(fit.space, [*fit.modes, *fit.shared]), key,
        len(fit.evidence.log_w) // 2, max_seconds, _prior_scale(fit.space.d),
    )  # fmt: skip
    fit.smc, fit.smc_seconds = res, time.time() - start
    if not res.stopped:
        fit.log_z, fit.log_z_error = res.log_z, _smc_error(res)
        fit.undersampled = bool(res.undersampled)
    fit.unreliable = _smc_unreliable(res)
    return fit


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
    nu_width=None,
    shared=(),
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
    nu_width : float, optional
        The widest cell in ``1 / P`` (``1 / T``), to size the plan's ``ln P``
        grid when ``plan`` is made here (default the width of ``cell``).
    shared : sequence of Mode, optional
        Neighbour modes in this cell's chart, extra mixture components
        ([`position_cells`][virgil.orbit_cells.position_cells] makes them).

    Returns
    -------
    CellFit

    Source: Hesterberg (1995), Gilks & Berzuini (2001), Hoffman & Gelman (2014).
    """
    cat = _Catalogue(summaries, log_outlier)
    name = onp.dtype(dtype).name
    k_plan, k_is, k_move = jax.random.split(key, 3)
    if not onp.all(onp.isfinite(cat.log_b)):
        raise ValueError(
            "A night has a non-finite outlier log density b_k (a "
            "non-finite peak mean or covariance): the all-outlier floor is "
            "undefined."
        )
    with run_in(name):
        space = _Space(
            cat, cell, 0.0, onp.asarray(t_nights, float) - t_anchor, a_range,
            e_max, flux_floor, jitter, getattr(np, name),
        )  # fmt: skip
        plan = search_plan(space, k_plan, nu_width) if plan is None else plan
        info = {}
        modes = search_cell(space, plan=plan, info=info)
        ev = _cell_evidence(space, [*modes, *shared], k_is, n_is)
        sup = [float(space.loglike(m.x)) for m in [*modes, *shared]]
        if ev.log_q is not None:
            ll = (
                onp.asarray(ev.log_w)
                + ev.log_q
                - onp.asarray(
                    jax.vmap(space.logprior)(np.asarray(ev.x)), float
                )
            )
            sup.append(float(onp.max(ll[onp.isfinite(ll)], initial=-onp.inf)))
    fit = CellFit(
        cell, ev.log_z, ev.log_z_error, modes, None, ev.log_w, ev.x,
        ev.at_floor, ev.undersampled, int(info["evaluations"]), n_is, ev,
        plan, space, bool(ev.ess < ESS_FAIL), bool(ev.n_nonfinite > 0),
        shared=list(shared), info=info,
        log_l_sup=max(sup, default=float("-inf")),
    )  # fmt: skip
    if posterior_draws and not fit.failed:
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
        Cells with a non-finite target, or at the floor (``cell-at-floor``: the
        search missed the dominant mode) or unreliable (ESS below ``ESS_FAIL``)
        *and* it ``matters`` (below). A flagged cell that cannot matter is
        information only: it stays flagged on its fit (``CellFit.at_floor``,
        ``unreliable``) but does not fail the system. Any true entry makes
        ``p_pos`` undefined.
    upper, matters : numpy.ndarray
        ``upper = ln mass + ln sup L`` bounds the cell's ``ln (mass Z_N)``.
        With ``R`` the best ``ln mass + ln Ẑ`` among unflagged cells, a cell
        ``matters`` iff it has no found mode or ``upper >= R - ln(n_cells/δ)``
        (every flagged cell matters if none is unflagged).
    omitted_bound : float
        The summed ``exp(upper)`` of flagged cells that do not matter, over
        the total mass. A diagnostic: below ``δ`` by the threshold, given the
        search found each cell's top mode (a lower ``sup L`` would understate
        it).
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
    upper: onp.ndarray = None
    matters: onp.ndarray = None
    omitted_bound: float = None


def _matters(cells, fits, log_p, delta):
    """``(upper, matters, flagged, omitted_bound)`` of the cells.

    ``upper = ln mass + ln sup L`` bounds a cell's ``ln (mass Z_N)``. With
    ``R`` the largest ``ln mass + ln Ẑ`` of a cell that is neither failed nor
    unreliable, a flagged cell is negligible iff ``upper < R - ln(n/δ)``: the
    negligible cells hold at most ``δ`` of the mass in all (up to the
    reference cell's own error). A cell without a found mode always matters,
    as does every flagged cell if no cell is unflagged.

    Source: design/orbit_alias_cycles.md §6.6, §6.10 (union bound).
    """
    n = len(cells)
    upper = onp.array([c.log_mass + f.log_l_sup for c, f in zip(cells, fits)])
    flagged = onp.array([bool(f.at_floor or f.unreliable) for f in fits])
    no_mode = onp.array([len(f.modes) == 0 for f in fits])
    ok = onp.array([not (f.failed or f.unreliable) for f in fits])
    ok &= onp.isfinite(log_p)
    r = log_p[ok].max() if ok.any() else -onp.inf
    # a nan bound compares false: it matters
    matters = no_mode | ~(upper < r - math.log(n / delta))
    neg = flagged & ~matters
    total = onp_logsumexp(log_p[onp.isfinite(log_p)])
    omitted = float(onp.exp(upper[neg] - total).sum()) if neg.any() else 0.0
    return upper, matters, flagged, omitted


def _map(fn, items, workers):
    """``[fn(i) for i in items]`` in ``workers`` threads, in order; the first
    exception (or interrupt) cancels the queued items before it is raised.
    One worker runs inline."""
    if workers <= 1:
        return [fn(i) for i in items]
    pool = concurrent.futures.ThreadPoolExecutor(workers)
    try:
        return [f.result() for f in [pool.submit(fn, i) for i in items]]
    except BaseException:
        pool.shutdown(wait=False, cancel_futures=True)
        raise
    finally:
        pool.shutdown(wait=True)


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
    workers=N_WORKERS,
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
    workers : int, optional
        Cells run at once, in threads (default ``N_WORKERS``, one: inline);
        the first cell runs alone, as it compiles the kernels and draws the
        sets. The result does not depend on it: each cell has its own key.
        Two threads cost about 0.3 GB more.
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
    workers = N_WORKERS if workers is None else workers
    start = time.time()
    keys = jax.random.split(key, len(cells))

    def run(i, plan):
        if time.time() - start > max_seconds:
            raise RuntimeError(
                f"position_cells reached max_seconds before cell N={cells[i].n}."
            )
        return position_leg(
            summaries, log_outlier, cells[i], keys[i], t_anchor=t_a,
            t_nights=t_all, jitter=jitter, n_is=n_is, posterior_draws=False,
            plan=plan, nu_width=1.0 / t_ab, **leg_kwargs,
        )  # fmt: skip

    fits = [run(0, None)]
    fits += _map(lambda i: run(i, fits[0].plan), range(1, len(cells)), workers)
    # a neighbour's mode with more than δ of its ρ mass across the shared edge
    # is a component here (its Laplace in ρ: a mode never leaves its own chart)
    for i, (cell, fit) in enumerate(zip(cells, fits)):
        shared = []
        for j, side in ((i - 1, 1.0), (i + 1, -1.0)):
            if not 0 <= j < len(cells):
                continue
            nu_edge = 1.0 / (cell.p_hi if side > 0 else cell.p_lo)
            d_nu = 1.0 / cell.p_lo - 1.0 / cell.p_hi
            for m in fits[j].modes:
                lo, hi = float(fits[j].space.ln_lo), float(fits[j].space.ln_hi)
                u = 1.0 / (1.0 + math.exp(-float(m.x[0])))
                nu = math.exp(-(lo + (hi - lo) * u))
                sd = nu * (hi - lo) * u * (1.0 - u) * math.sqrt(m.cov[0, 0])
                across = float(stats.norm.sf(abs(nu - nu_edge) / sd))
                if not across > delta:
                    continue
                # centred on the tail inside this cell, from one σ_ρ to the
                # edge stop, in this cell's chart
                r = min(sd / d_nu, 0.5)
                near = min(-math.log(r / (1.0 - r)), EDGE_STOP)
                mid, half = 0.5 * (EDGE_STOP + near), 0.5 * (EDGE_STOP - near)
                half = max(half, 1.0)
                scale = onp.ones(len(m.x))
                scale[0] = half / math.sqrt(m.cov[0, 0])
                cov = scale[:, None] * m.cov * scale
                x = onp.array(m.x, float)
                x[0] = side * mid
                # cell j's Laplace is under j's conditional prior: restate it
                # in this cell's (the ratio of the cells' prior masses)
                lap = (
                    laplace_log_evidence(m) + math.log(across)
                    + cells[j].log_mass - cell.log_mass
                )  # fmt: skip
                shift = 0.5 * (
                    len(x) * math.log(2.0 * math.pi)
                    + onp.linalg.slogdet(cov)[1]
                )
                shared.append(Mode(x, lap - shift, cov, m.n_inliers))
        if shared:
            with run_in(onp.dtype(fit.space.dtype).name):
                ev = _cell_evidence(
                    fit.space, [*fit.modes, *shared],
                    jax.random.split(keys[i], 3)[1], n_is,
                )  # fmt: skip
            fit.shared, fit.evidence = shared, ev
            fit.is_evaluations += n_is  # the first run's draws were discarded
            fit.log_z, fit.log_z_error = ev.log_z, ev.log_z_error
            fit.log_w, fit.x_is = ev.log_w, ev.x
            fit.at_floor, fit.undersampled = ev.at_floor, ev.undersampled
            fit.unreliable = bool(ev.ess < ESS_FAIL)
            fit.nonfinite = bool(ev.n_nonfinite > 0)
    # The unreliable cells that matter get the SMC (§6.15). Whether a cell
    # matters depends on every estimate, so repeat until none is left; each
    # cell is SMC'd at most once.
    tried = set()
    smc_keys = jax.random.split(jax.random.fold_in(key, 3), len(cells))
    while True:
        log_p = onp.array([c.log_mass + f.log_z for c, f in zip(cells, fits)])
        upper, matters, flagged, omitted = _matters(cells, fits, log_p, delta)
        todo = [
            i
            for i, (f, m) in enumerate(zip(fits, matters))
            if m and i not in tried and not f.failed and f.unreliable
        ]
        if not todo:
            break
        tried.update(todo)
        left = max(max_seconds - (time.time() - start), 0.0)
        _map(lambda i: smc_cell(fits[i], smc_keys[i], max_seconds=left), todo, workers)  # fmt: skip
    failed = onp.array(
        [f.nonfinite or (fl and m) for f, fl, m in zip(fits, flagged, matters)]
    )
    p = onp.full(len(cells), onp.nan) if failed.any() else onp.exp(log_p - onp_logsumexp(log_p))  # fmt: skip
    below = p < delta / len(cells)
    draw_keys = jax.random.split(jax.random.fold_in(key, 1), len(cells))
    drawn = [
        (f, k)
        for f, pi, k in zip(fits, p, draw_keys)
        if pi >= delta / len(cells)
    ]
    for (f, _), draws in zip(
        drawn, _map(lambda fk: _draw(*fk), drawn, workers)
    ):
        f.draws = draws
    return CellsResult(
        domain, (first, last), t_ab, cells, fits, p, below,
        below if prune else onp.zeros_like(below), float(p[below].sum()),
        float(jitter), failed, upper, matters, omitted,
    )  # fmt: skip


# ---------------------------------------------------------------------------
# Census diagnostics: they read a fit and never change it
# ---------------------------------------------------------------------------


def truth_chart(space, orbit, t_anchor, ln_f, ln_s=0.0):
    """The chart vector of a known orbit in the cell ``space`` (a diagnostic of
    the population census, never an input of a fit).

    ``t_anchor`` is the anchor time in days since ``orbit.t_ref``. ``y_z`` is
    placed at the radius ``R0`` (the likelihood does not depend on it); the
    label sign is not encoded: it is a symmetry of the target (the swap among
    ``_symmetry_images``).
    """
    z, zeta = oc.campbell_to_coordinates(orbit, t_anchor)
    z, zeta = onp.asarray(z, float), onp.asarray(zeta, float)

    def logit(frac):
        frac = onp.clip(frac, 1e-6, 1.0 - 1e-6)
        return onp.log(frac) - onp.log1p(-frac)

    la = math.log(space.a_range[0])
    da = math.log(space.a_range[1] / space.a_range[0])
    ln_lo, ln_hi = float(space.ln_lo), float(space.ln_hi)
    u = zeta / math.sqrt(space.e_max)  # the disc point of the ζ chart
    w_s = onp.zeros(space.d - 9)
    if space.cat.rescalable:
        w_s += logit((ln_s / math.log(S_MAX) + 1.0) / 2.0)
    return onp.concatenate(
        [
            [logit((math.log(float(orbit.period)) - ln_lo) / (ln_hi - ln_lo))],
            z / onp.linalg.norm(z) * R0,
            [logit((math.log(float(onp.sum(z**2) / 2.0)) - la) / da)],
            u / onp.sqrt(max(1.0 - onp.sum(u**2), 1e-12)),
            [logit(1.0 - ln_f / space.ln_f_floor)],
            w_s,
        ]
    )


def _maha2(r, cov):
    return float(r @ onp.linalg.solve(onp.asarray(cov, float), r))


def _group(d):
    return [onp.eye(d), *_symmetry_images(d)]


def truth_distance2(space, mode, x_true):
    """Squared Mahalanobis distance of ``mode`` to the truth, under the mode's
    Laplace covariance, minimized over the images of the target's symmetries
    (design §14 assertion 5). The radius of ``y_z`` is set to the mode's: the
    likelihood is blind to it."""
    best = onp.inf
    x = onp.asarray(mode.x, float)
    for g in _group(space.d):
        t = g @ x_true
        t[_Z] *= onp.linalg.norm(x[_Z]) / onp.linalg.norm(t[_Z])
        best = min(best, _maha2(t - x, mode.cov))
    return best


def oracle_evidence(fit, x_true, key, n_is=N_IS):
    """The mass oracle of a cell: its defensive IS rerun with the
    truth-started optimum added as a mixture component.

    The optimum is Newton (the search's own ``_refine``) on the label-summed
    objective from the best of the symmetry images of ``x_true``
    (``truth_chart``). Adding a component keeps the estimate unbiased. Nothing
    of ``fit`` changes: the result is a dict of diagnostic columns.

    Returns
    -------
    dict
        ``oracle_log_z`` and ``oracle_log_z_error`` (the augmented IS),
        ``oracle_ess`` (of the IS), ``oracle_smc`` (1 if the augmented cell's
        IS was unreliable and its ``oracle_log_z`` is the SMC's, which then
        was not itself unreliable), ``oracle_laplace`` (the optimum's
        Laplace ``ln Z``),
        ``oracle_gap`` (augmented minus unaided ``ln Ẑ``), ``oracle_sd`` (the
        two errors in quadrature) and ``oracle_basin`` (the number of found
        modes whose Newton run ends at the optimum); ``oracle_log_z`` is
        ``nan`` and ``oracle_basin`` -1 if the optimum is not finite.
    """
    space = fit.space
    with run_in(onp.dtype(space.dtype).name):
        starts = onp.array([g @ x_true for g in _group(space.d)])
        value = onp.array(
            [
                float(_objective(np.asarray(s, space.dtype), space))
                for s in starts
            ]
        )
        value = onp.where(onp.isfinite(value), value, -onp.inf)
        found = _laplace_modes(
            space, starts[[int(onp.argmax(value))]], {"evaluations": 0.0}
        )
        if not found:
            return dict(oracle_log_z=onp.nan, oracle_basin=-1)
        mode = found[0]
        ev = _cell_evidence(space, [*fit.modes, mode], key, n_is)
        log_z, log_z_error, smc = ev.log_z, ev.log_z_error, False
        if ev.ess < ESS_FAIL and ev.n_nonfinite == 0:  # the same rule (§6.15)
            res = _run_smc(
                space, _mixture(space, [*fit.modes, mode]),
                jax.random.fold_in(key, 1), n_is // 2, float("inf"),
                _prior_scale(space.d),
            )  # fmt: skip
            if not _smc_unreliable(res):
                log_z, log_z_error, smc = res.log_z, _smc_error(res), True
        # a found mode shares the optimum's basin when Newton from it ends at
        # the optimum or a symmetry image, by the distinct-mode rule (both
        # covariances, ``MAHAL``)
        images = [
            (g @ onp.asarray(mode.x, float), g @ mode.cov @ g.T)
            for g in _group(space.d)
        ]
        basin = 0
        for m in fit.modes:
            x = onp.asarray(
                _refine(space, np.asarray(m.x, space.dtype), N_NEWTON)[0]
            )
            basin += any(
                max(_maha2(x - gx, gc), _maha2(x - gx, m.cov)) < MAHAL
                for gx, gc in images
            )
    return dict(
        oracle_log_z=log_z, oracle_log_z_error=log_z_error,
        oracle_ess=ev.ess, oracle_laplace=laplace_log_evidence(mode),
        oracle_gap=log_z - fit.log_z,
        oracle_sd=math.hypot(log_z_error, fit.log_z_error),
        oracle_basin=int(basin), oracle_smc=int(smc),
    )  # fmt: skip


N_BLOCK = 200  # redraws per block of ``block_variance`` (a fixed diagnostic)
_Z_BLOCK = slice(1, 6)  # (y_z[4], w_a): the Thiele–Innes block of the chart


def block_variance(fit, key, n=N_BLOCK):
    """Variance of ``ln w`` of a cell's importance sampling by block.

    Take the highest-weight draw ``x*`` of the cell's IS and the mixture
    component ``m*`` it most belongs to. **z block**: freeze θ = (w_P, y_ζ,
    w_f, w_s) at ``x*`` and redraw ``(y_z, w_a)`` from the conditional of the
    multivariate t of ``m*``; **θ block**: freeze z and redraw θ. ``ln w`` is
    ``ln target - ln q`` of the full mixture. Diagnostics only: nothing of
    ``fit`` changes.

    Returns
    -------
    dict
        ``var_lnw_z`` and ``var_lnw_theta`` (``nan`` for a cell without modes
        or draws).
    """
    nan = dict(var_lnw_z=onp.nan, var_lnw_theta=onp.nan)
    ev, space = fit.evidence, fit.space
    if not fit.modes or ev is None or not onp.isfinite(ev.log_w).any():
        return nan
    with run_in(onp.dtype(space.dtype).name):
        mix = _mixture(space, fit.modes)
        x_star = onp.asarray(ev.x, float)[int(onp.argmax(ev.log_w))]
        z = onp.einsum(
            "mij,mj->mi",
            onp.asarray(mix.linv, float),
            x_star - onp.asarray(mix.mu, float),
        )
        df, d = mix.df, space.d
        lt = -0.5 * (df + d) * onp.log1p(
            (z**2).sum(-1) / df
        ) - 0.5 * onp.asarray(mix.logdet, float)
        m = int(onp.argmax(onp.asarray(mix.log_w, float) + lt))
        mu = onp.asarray(mix.mu, float)[m]
        chol = onp.asarray(mix.chol, float)[m]
        cov = chol @ chol.T
        in_z = onp.zeros(d, bool)
        in_z[_Z_BLOCK] = True
        k_z, k_t = jax.random.split(key)
        out = {}
        for name, free, k in (
            ("var_lnw_z", in_z, k_z),
            ("var_lnw_theta", ~in_z, k_t),
        ):
            fix = ~free
            s_cc = cov[onp.ix_(fix, fix)]
            gain = cov[onp.ix_(free, fix)] @ onp.linalg.inv(s_cc)
            r = x_star[fix] - mu[fix]
            delta = float(r @ onp.linalg.solve(s_cc, r))
            nu = df + fix.sum()
            c_f = (
                (df + delta)
                / nu
                * (cov[onp.ix_(free, free)] - gain @ cov[onp.ix_(fix, free)])
            )
            l_f = onp.linalg.cholesky(0.5 * (c_f + c_f.T))
            k_n, k_g = jax.random.split(k)
            e = onp.asarray(jax.random.normal(k_n, (n, int(free.sum()))))
            g = onp.asarray(jax.random.gamma(k_g, 0.5 * nu, (n,))) / (0.5 * nu)
            x = onp.tile(x_star, (n, 1))
            x[:, free] = (mu[free] + gain @ r) + e @ l_f.T / onp.sqrt(g)[
                :, None
            ]
            xj = np.asarray(x, space.dtype)
            lw = onp.asarray(
                jax.vmap(lambda v: _target(v, space) - mix.log_q(v))(xj),
                float,
            )
            lw = lw[onp.isfinite(lw)]
            out[name] = float(onp.var(lw)) if len(lw) > 1 else onp.nan
    return out


def persist_arrays(fit, key):
    """The arrays that rebuild a cell's baseline: every mode (chart ``x``,
    Laplace covariance, ``log_target``, inlier count, night labels, ``-1`` for
    an outlier, and the inlier set), the IS draws and their ``ln w`` and the
    cell's JAX key (data)."""
    space = fit.space
    n_k, d = len(space.cat.log_b), space.d
    with run_in(onp.dtype(space.dtype).name):
        labels = onp.array(
            [
                onp.asarray(_labels(np.asarray(m.x, space.dtype), space))
                for m in fit.modes
            ],
            int,
        ).reshape(len(fit.modes), n_k)
    arrays = dict(
        mode_x=onp.array([onp.asarray(m.x, float) for m in fit.modes]).reshape(-1, d),
        mode_cov=onp.array([onp.asarray(m.cov, float) for m in fit.modes]).reshape(-1, d, d),
        mode_log_target=onp.array([float(m.log_target) for m in fit.modes]),
        mode_n_inliers=onp.array([float(m.n_inliers) for m in fit.modes]),
        mode_labels=labels, mode_inliers=labels >= 0,
        key=onp.asarray(jax.random.key_data(key)),
    )  # fmt: skip
    if fit.evidence is not None:
        arrays.update(
            is_x=onp.asarray(fit.evidence.x, float),
            is_log_w=onp.asarray(fit.evidence.log_w, float),
        )
    return arrays
