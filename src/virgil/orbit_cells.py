"""Cells, period domain and the position-level SMC leg of the alias-resolved orbit fit.

!!! warning "Provisional"
    May change in any 0.4.x release without a deprecation cycle.

The period prior is log-uniform on ``[P_cut, P_max]``
([`period_domain`][virgil.orbit_cells.period_domain]). Two anchor nights set
``T``, and ``T / P = N + ρ`` splits the domain into cells, an exact partition for
any edges ([`make_cells`][virgil.orbit_cells.make_cells]): the edges sit at valleys
of a pilot run's marginal of ``T / P`` and change labels only. Each cell is
sampled by the tempered SMC of [`virgil.smc`][virgil.smc], from the cell prior to
the night-summary likelihood of [`virgil.night_summaries`][virgil.night_summaries],
in the ``(ln P, z, ζ)`` coordinates of ``virgil._orbit_coordinates``, with
HMC and the Luger Gibbs move on the Thiele–Innes constants
([`position_leg`][virgil.orbit_cells.position_leg]).
[`position_cells`][virgil.orbit_cells.position_cells] runs every cell and gives
``p(N) ∝ prior mass × Ẑ_N``, with an opt-in pruning that records the dropped mass.
Design: ``design/orbit_alias_cycles.md`` §4–§7.

Source: Luger, Foreman-Mackey & Hogg (2017), Del Moral, Doucet & Jasra (2006),
[references](../references.md); the cells, outlier floor and leg are derived
in the design note.
"""

import dataclasses
import math

import jax
import jax.numpy as np
import numpy as onp
from jax.scipy.special import logsumexp
from scipy.special import logsumexp as onp_logsumexp
from scipy.stats import qmc

from . import _orbit_coordinates as oc
from . import smc
from ._precision import run_in
from .aliases import _P_FLOOR, alias_bands, period_cut, position_profile
from .night_summaries import DELTA
from .orbits import KeplerOrbit, PositionData, _unit_orbit

__all__ = [
    "Cell",
    "CellsResult",
    "LegResult",
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
    "valley_edges",
]

__provisional__ = ("*",)

# Declared prior bounds, inherited from the engine (design §10).
S_MAX = 10.0  # inlier scale: ln s uniform on ±ln S_MAX (covers χ²/N <= 100)
A_RANGE = (1.0, 30.0)  # mas
E_MAX = 0.9
FLUX_FLOOR = 1e-4  # engine's flux floor (arcs._half_position)
_N_PV = 4  # (p, v) of the linear Gaussian block; ln f is the fifth dimension


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


def valley_edges(x, weights, cells, t_ab, width=0.25):
    """Cell edges (periods) at the valleys of a pilot marginal of ``x = T / P``.

    For each pair of neighbouring cells the edge is the minimum of the
    weighted, Gaussian-smoothed marginal of ``x`` between the two integers
    around their default edge; with no mass there, the default.

    Parameters
    ----------
    x, weights : array-like
        ``T / P`` of the pilot particles and their weights.
    cells : sequence of Cell
        The default cells.
    t_ab : float
        ``T``, the time between the anchors (days).
    width : float, optional
        Smoothing in ``x`` (default 1/4).

    Returns
    -------
    numpy.ndarray
        One edge per pair of neighbouring cells, as for ``make_cells``.
    """
    x, w = onp.asarray(x, float), onp.asarray(weights, float)
    x_min, x_max = t_ab / cells[0].p_hi, t_ab / cells[-1].p_lo
    edges = []
    for upper in cells[:-1]:
        default = t_ab / upper.p_lo
        lo, hi = (
            max(math.floor(default), x_min),
            min(math.floor(default) + 1, x_max),
        )
        # an interior grid: an edge never touches the domain's end
        grid = onp.linspace(lo, hi, 101)[1:-1]
        if not w[(x >= lo) & (x <= hi)].sum() > 0:
            edges.append(upper.p_lo)
            continue
        density = (
            w[None] * onp.exp(-0.5 * ((grid[:, None] - x[None]) / width) ** 2)
        ).sum(1)
        edges.append(t_ab / grid[onp.argmin(density)])
    return onp.array(edges)


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


def _log_mixture(cat, ln_s, jitter, pv, ln_f, dtype):
    """``ln ∫ dε Beta(ε) ½ Σ_σ Π_k [(1-ε) e^{ℓ̃_k} + ε e^{b_k}]``, with the
    brighter/fainter sign ``σ`` flipping ``(p, v)``. ``pv`` is ``(K, 4)``."""
    eps = _eps_nodes(cat.n_eps).astype(dtype)
    per_sign = []
    for sign in (1.0, -1.0):
        state = np.concatenate(
            [sign * pv, np.full((pv.shape[0], 1), ln_f, dtype)], axis=1
        )
        a = logsumexp(_peak_terms(cat, ln_s, jitter, state, dtype), axis=1)
        night = np.logaddexp(
            np.log1p(-eps)[:, None] + a[None], np.log(eps)[:, None] + cat.log_b
        )
        per_sign.append(night.sum(axis=1))
    return logsumexp(np.stack(per_sign)) - np.log(2.0 * cat.n_eps)


# ---------------------------------------------------------------------------
# The sampling space of one cell
# ---------------------------------------------------------------------------

_Z = slice(1, 5)  # y of z (4), then the logistic of ln a, then ζ's y (2),
#                   the logistic of ln f, one logistic per error scale


def _logistic_log_pdf(w):
    return jax.nn.log_sigmoid(w) + jax.nn.log_sigmoid(-w)


class _Space:
    """Coordinates, prior and likelihood of one cell: ``x`` is the flat chart
    ``(w_P, y_z[4], w_a, y_ζ[2], w_f, w_s[G])``."""

    def __init__(
        self, cat, cell, t_anchor, t_k, a_range, e_max, flux_floor, jitter,
        dtype,
    ):  # fmt: skip
        self.cat, self.cell, self.dtype = cat.cast(dtype), cell, dtype
        self.t_anchor, self.t_k = t_anchor, np.asarray(t_k, dtype)
        self.a_range, self.e_max, self.jitter = a_range, e_max, jitter
        self.ln_f_floor = math.log(flux_floor)
        self.d = 9 + len(cat.blocks)
        self.s_w2 = (a_range[1] ** 2 - a_range[0] ** 2) / (
            2 * math.log(a_range[1] / a_range[0]) * 3
        )

    def decode(self, x):
        """The orbit coordinates of chart vector(s) ``x`` (last axis)."""
        lo, hi = math.log(self.cell.p_lo), math.log(self.cell.p_hi)
        la = math.log(self.a_range[0])
        da = math.log(self.a_range[1] / self.a_range[0])
        y = x[..., _Z]
        direction = y / np.linalg.norm(y, axis=-1, keepdims=True)
        a = np.exp(la + da * jax.nn.sigmoid(x[..., 5]))
        yz = x[..., 6:8]
        zeta = (
            math.sqrt(self.e_max)
            * yz
            / np.sqrt(1.0 + np.sum(yz**2, axis=-1, keepdims=True))
        )
        return {
            "period": np.exp(lo + (hi - lo) * jax.nn.sigmoid(x[..., 0])),
            "z": direction * np.sqrt(2.0 * a)[..., None],
            "zeta": zeta,
            "ln_f": self.ln_f_floor * (1.0 - jax.nn.sigmoid(x[..., 8])),
            # error scales stay at 1 where a scale does not factor out (gains,
            # closure offsets): the coordinates remain, with no likelihood
            "ln_s": float(self.cat.rescalable)
            * math.log(S_MAX)
            * (2.0 * jax.nn.sigmoid(x[..., 9:]) - 1.0),
        }

    def states(self, period, ti, zeta):
        """``(p, v)`` of the Thiele–Innes constants ``ti`` (the anchor-frame
        TI′) at the nights: ``(K, 4)``, in mas and mas/day."""
        ecc, anomaly = oc.zeta_to_eccentricity(zeta)
        a, b, f, g = oc.anchor_unrotate(tuple(ti), anomaly)
        dt_peri = -anomaly * period / (2.0 * np.pi)  # anchor at time 0

        def position(dt):
            x, y = _unit_orbit(dt, period, dt_peri, ecc)
            return np.stack([b * x + g * y, a * x + f * y])

        pos, vel = jax.jvp(position, (self.t_k,), (np.ones_like(self.t_k),))
        return np.concatenate([pos.T, vel.T], axis=1)

    def loglike(self, x):
        c = self.decode(x)
        pv = self.states(c["period"], oc.z_to_ti(c["z"]), c["zeta"])
        return _log_mixture(
            self.cat, c["ln_s"], self.jitter, pv, c["ln_f"], self.dtype
        )

    def logprior(self, x):
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

    def sample(self, key, n):
        """``n`` exact draws from the cell prior."""
        k = jax.random.split(key, 6)
        dt = self.dtype
        zeta = oc.sample_zeta(k[3], (n,)).astype(dt)  # uniform on the disc
        return np.concatenate(
            [
                jax.random.logistic(k[0], (n, 1), dt),
                jax.random.normal(k[1], (n, 4), dt),
                jax.random.logistic(k[2], (n, 1), dt),
                zeta / np.sqrt(1.0 - np.sum(zeta**2, -1, keepdims=True)),
                jax.random.logistic(k[4], (n, 1), dt),
                jax.random.logistic(k[5], (n, self.d - 9), dt),
            ],
            axis=1,
        )

    def with_ti(self, x, theta, key):
        """``x`` with its ``z`` replaced by a preimage of ``theta`` (TI′), the
        radius of ``y`` kept; ``None`` where ``a`` leaves its prior range."""
        z = oc.ti_to_z(*theta)
        # the four preimages are equivalent: pick one uniformly
        z = z * (2.0 * jax.random.bernoulli(key, 0.5, (2,)) - 1.0).repeat(2)
        a = oc.semimajor(z)
        la = math.log(self.a_range[0])
        da = math.log(self.a_range[1] / self.a_range[0])
        frac = (np.log(a) - la) / da
        ok = (frac > 0.0) & (frac < 1.0)
        frac = np.clip(frac, 1e-12, 1.0 - 1e-12)
        y = z / np.linalg.norm(z) * np.linalg.norm(x[_Z])
        new = x.at[_Z].set(y).at[5].set(np.log(frac) - np.log1p(-frac))
        return new, ok


def _luger_move(space):
    """The Luger Gibbs-then-Metropolis move on TI′ (design §6.3).

    Given the core ``(P, ζ, ln f, ln s)`` the night states are linear in
    ``TI′``. Draw ``(ε node, σ, peak labels)`` from their conditional, propose
    ``TI′`` from the Gaussian conditional under the working prior
    ``N(0, s_w² I)`` and accept by the independence-Metropolis ratio of the
    augmented target; it leaves the tempered density invariant at any β.
    """
    cat, dtype = space.cat, space.dtype

    def move(key, x, logdensity):
        k_node, k_label, k_prop, k_acc, k_sign = jax.random.split(key, 5)
        c = space.decode(x)
        theta = np.stack(oc.z_to_ti(c["z"]))
        jac = jax.jacfwd(lambda t: space.states(c["period"], t, c["zeta"]))
        lin = jac(theta)  # (K, 4, 4): states are linear in TI′
        eps = _eps_nodes(cat.n_eps).astype(dtype)
        pv = np.einsum("kab,b->ka", lin, theta)
        signs = np.array([1.0, -1.0], dtype)
        terms = np.stack(
            [
                _peak_terms(
                    cat,
                    c["ln_s"],
                    space.jitter,
                    np.concatenate(
                        [s * pv, np.full((pv.shape[0], 1), c["ln_f"], dtype)],
                        1,
                    ),
                    dtype,
                )
                for s in signs
            ]
        )  # (2, K, J)
        a = logsumexp(terms, axis=2)  # (2, K)
        night = np.logaddexp(
            np.log1p(-eps)[:, None, None] + a[None],
            np.log(eps)[:, None, None] + cat.log_b,
        )  # (n, 2, K)
        joint = night.sum(axis=2)
        pick = jax.random.categorical(k_node, joint.ravel())
        i, si = pick // 2, pick % 2
        sign = signs[si]
        labels_logit = np.concatenate(
            [
                np.log1p(-eps[i]) + terms[si],
                (np.log(eps[i]) + cat.log_b)[:, None],
            ],
            axis=1,
        )  # (K, J + 1): the last column is the outlier
        j = jax.random.categorical(k_label, labels_logit, axis=1)  # (K,)
        peak = j < terms.shape[2]
        j = np.minimum(j, terms.shape[2] - 1)
        _, _, inv = _peak_precision(cat, c["ln_s"], space.jitter, dtype)
        rows = np.arange(j.shape[0])
        lam = inv[rows, j][:, :_N_PV, :_N_PV] * peak[:, None, None]
        cross = inv[rows, j][:, :_N_PV, _N_PV] * peak[:, None]
        mean = cat.mean[rows, j]
        d_f = c["ln_f"] - mean[:, 4]
        prec = np.eye(4, dtype=dtype) / space.s_w2 + np.einsum(
            "kba,kbc,kcd->ad", lin, lam, lin
        )
        rhs = sign * np.einsum(
            "kba,kb->a", lin,
            np.einsum("kab,kb->ka", lam, mean[:, :_N_PV]) - cross * d_f[:, None],
        )  # fmt: skip
        chol = np.linalg.cholesky(prec)
        mu = np.linalg.solve(prec, rhs)
        noise = jax.random.normal(k_prop, (4,), dtype)
        proposal = mu + jax.scipy.linalg.solve_triangular(
            chol.T, noise, lower=False
        )
        new, ok = space.with_ti(x, proposal, k_sign)

        def log_target(point, ti):
            """``ln τ(θ) - ln N_w(θ) - ln L(θ)`` with τ the tempered density in
            TI′: the chart prior drops out, leaving β ln L and ``p(TI′)``."""
            u = np.hypot(ti[0] + ti[3], ti[1] - ti[2])
            v = np.hypot(ti[0] - ti[3], ti[1] + ti[2])
            a = (u + v) / 2
            # ln p(TI′): isotropic orientation, log-uniform a (design §3.2)
            log_p = (
                -np.log(
                    np.pi**2 * math.log(space.a_range[1] / space.a_range[0])
                )
                - 2.0 * np.log(u + v)
                - np.log(u)
                - np.log(v)
            )
            log_p = np.where(
                (a >= space.a_range[0]) & (a <= space.a_range[1]),
                log_p,
                -np.inf,
            )
            return (
                log_p
                + logdensity(point)
                - space.logprior(point)
                - space.loglike(point)
                + 0.5 * np.sum(ti**2) / space.s_w2
            )

        log_ratio = log_target(new, proposal) - log_target(x, theta)
        take = ok & (np.log(jax.random.uniform(k_acc, (), dtype)) < log_ratio)
        return np.where(take, new, x), take

    return move


# ---------------------------------------------------------------------------
# The leg
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class LegResult:
    """The position-level leg of one cell.

    Attributes
    ----------
    cell : Cell
    log_z, log_z_error, log_z_variance : float
        ``ln Ẑ_N^pos`` of the cell's normalized prior (not multiplied by the
        cell's mass), the standard error and the single-run variance from
        [`SMCResult`][virgil.smc.SMCResult]; ``nan`` if ``stopped``.
    particles, weights
        Chart vectors at β = 1 and their weights (``None`` if ``stopped``);
        ``decode_particles`` turns them into orbit coordinates.
    stopped : bool
        The time or stage cap was reached before β = 1. This is a failure: there
        is no posterior at β < 1.
    undersampled : bool
        ``band-undersampled``: the final ESS is below target. Information only.
    variance_degenerate : bool
        The variance estimate is unusable (see ``SMCResult``).
    """

    cell: Cell
    log_z: float
    log_z_error: float
    log_z_variance: float
    particles: object
    weights: object
    stopped: bool
    undersampled: bool
    variance_degenerate: bool
    space: object = dataclasses.field(repr=False, default=None)


def decode_particles(result):
    """Orbit coordinates of a leg's particles.

    Returns a dict of arrays with ``period``, ``z``, ``zeta``, ``ln_f``,
    ``ln_s`` and the reporting invariants ``a``, ``ecc``, ``cos_inc``,
    ``omega`` and ``Omega`` (invariant under the covers of ``z`` and ``ζ``).
    """
    if result.particles is None:
        raise ValueError("The leg stopped before β = 1: it has no particles.")
    c = result.space.decode(result.particles)
    return {**c, **oc.invariants(c["period"], c["z"], c["zeta"])}


def position_leg(
    summaries,
    log_outlier,
    cell,
    key,
    *,
    t_anchor,
    t_nights,
    n_particles=2048,
    a_range=A_RANGE,
    e_max=E_MAX,
    flux_floor=FLUX_FLOOR,
    jitter=0.0,
    luger=True,
    steps=4,
    replicates=1,
    max_seconds=float("inf"),
    dtype="float64",
):
    """The tempered SMC of one cell from its prior to the night-summary likelihood.

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
    n_particles, steps, replicates, max_seconds
        As for [`tempered_smc`][virgil.smc.tempered_smc]; defaults are the budgets
        of the design note (steps = HMC moves per stage).
    a_range, e_max, flux_floor : declared prior bounds
        Semimajor axis (mas), maximum eccentricity and flux-ratio floor.
    jitter : float, optional
        Position jitter (mas) added in quadrature, from
        [`fit_jitter`][virgil.orbit_cells.fit_jitter] (default 0).
    luger : bool, optional
        Add the Luger move on TI′ to the HMC steps (default True).
    dtype : {"float64", "float32"}, optional
        Precision of the run (default float64, as ``fit``).

    Returns
    -------
    LegResult

    Source: Luger, Foreman-Mackey & Hogg (2017); Del Moral, Doucet & Jasra (2006).
    """
    cat = _Catalogue(summaries, log_outlier)
    with run_in(dtype):
        space = _Space(
            cat, cell, 0.0, onp.asarray(t_nights, float) - t_anchor, a_range,
            e_max, flux_floor, jitter, np.float64 if dtype == "float64" else np.float32,
        )  # fmt: skip
        result = smc.tempered_smc(
            space.logprior,
            space.loglike,
            space.sample,
            key=key,
            n_particles=n_particles,
            replicates=replicates,
            steps=steps,
            max_seconds=max_seconds,
            moves=(_luger_move(space),) if luger else (),
        )
    return LegResult(
        cell,
        result.log_z,
        result.log_z_error,
        result.log_z_variance,
        result.particles,
        result.weights,
        result.stopped,
        result.undersampled,
        result.variance_degenerate,
        space,
    )


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
    """The position-level leg over every cell.

    Attributes
    ----------
    domain : PeriodDomain
    anchors : (int, int)
        Indices of the anchor nights.
    t_ab : float
        ``T``, the time between the anchors (days).
    cells : list of Cell
    legs : list of LegResult
    p_pos : numpy.ndarray
        ``p(N) ∝ mass_N Ẑ_N^pos`` over the cells, summing to one.
    below_delta : numpy.ndarray
        Cells with ``p_pos < δ / N_cells``: the ones pruning would drop.
    pruned : numpy.ndarray
        Cells actually dropped (all false unless ``prune=True``).
    pruned_mass : float
        ``Σ p_pos`` over ``below_delta`` (< δ by construction): the position-level
        mass pruning drops, for Stage 4 to compare with the visibility level.
    jitter : float
        The position jitter used (mas).
    edges : numpy.ndarray
        The cell edges (periods, days).
    """

    domain: PeriodDomain
    anchors: tuple
    t_ab: float
    cells: list
    legs: list
    p_pos: onp.ndarray
    below_delta: onp.ndarray
    pruned: onp.ndarray
    pruned_mass: float
    jitter: float
    edges: onp.ndarray


def position_cells(
    summaries,
    domain,
    key,
    *,
    log_outlier=None,
    outlier=None,
    jitter=None,
    pilot=True,
    prune=False,
    delta=DELTA,
    n_particles=2048,
    max_seconds=float("inf"),
    **leg_kwargs,
):
    """The position-level leg over every cell of ``domain``.

    Chooses the anchors, runs a pilot leg over the whole domain at a quarter of
    the particles to place the cell edges at valleys (``pilot=False`` keeps the
    default edges), then runs one leg per cell. ``p(N) ∝ mass_N Ẑ_N``. A leg that
    reaches ``max_seconds`` is a failure and raises.

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
    pilot, prune : bool, optional
        Place edges by the pilot (True); drop cells below ``δ / N_cells``
        (False: the dropped mass is recorded but nothing is dropped).
    delta : float, optional
        Reporting precision (default ``DELTA``).
    n_particles : int, optional
        Particles per cell.
    max_seconds : float, optional
        Time cap per leg.
    **leg_kwargs
        Other arguments of [`position_leg`][virgil.orbit_cells.position_leg].

    Returns
    -------
    CellsResult

    Raises
    ------
    RuntimeError
        If a leg reached its time or stage cap.
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
    kwargs = dict(
        t_anchor=t_a, t_nights=t_all, jitter=jitter, max_seconds=max_seconds,
        **leg_kwargs,
    )  # fmt: skip
    keys = jax.random.split(key, 2)
    cells = make_cells(t_a, t_b, domain)
    if pilot and len(cells) > 1:
        whole = Cell(0, domain.p_cut, domain.p_max, 0.0)
        run = position_leg(
            summaries, log_outlier, whole, keys[0],
            n_particles=max(n_particles // 4, 8), **kwargs,
        )  # fmt: skip
        _check(run)
        period = run.space.decode(run.particles)["period"]
        cells = make_cells(
            t_a, t_b, domain,
            valley_edges(t_ab / onp.asarray(period), run.weights, cells, t_ab),
        )  # fmt: skip
    legs = []
    for cell, k in zip(cells, jax.random.split(keys[1], len(cells))):
        legs.append(
            position_leg(
                summaries,
                log_outlier,
                cell,
                k,
                n_particles=n_particles,
                **kwargs,
            )
        )
        _check(legs[-1])
    log_p = onp.array([c.log_mass + g.log_z for c, g in zip(cells, legs)])
    p = onp.exp(log_p - onp_logsumexp(log_p))
    below = p < delta / len(cells)
    return CellsResult(
        domain, (first, last), t_ab, cells, legs, p, below,
        below if prune else onp.zeros_like(below), float(p[below].sum()),
        float(jitter),
        onp.array([c.p_hi for c in cells[1:]]),
    )  # fmt: skip


def _check(leg):
    if leg.stopped:
        raise RuntimeError(
            f"The leg of cell N={leg.cell.n} reached its time or stage cap "
            "before β = 1; there is no posterior. Raise max_seconds, or use "
            "more particles."
        )
    if not onp.isfinite(leg.log_z):
        raise RuntimeError(
            f"The leg of cell N={leg.cell.n} returned a non-finite evidence."
        )
