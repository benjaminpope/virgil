"""Population test of the position-level cell fit (search, defensive IS, HMC).

Simulation-based calibration (Talts et al. 2018) is exact only when the
generating prior is the fitted prior, so each system is drawn from the fit's
own prior:

* ``P`` log-uniform on ``[2, 60]`` d, kept when it lies in the data-derived
  default domain ``[P_cut, P_max]``, which the fit then uses intersected with
  ``[2, 60]`` (``p_range``). The domain depends on the nights and the quoted
  errors only, drawn independently of ``P``, so ``P`` given the selection is
  log-uniform on the fit domain. A truth outside the default domain is
  excluded and counted (stratum B');
* ``e`` uniform on [0, ``E_MAX``], isotropic orientation, ``a`` log-uniform over
  ``A_RANGE``, ``ln f`` uniform from ``ln FLUX_FLOOR`` to 0, the brighter or
  fainter sign at random, ``ln s`` uniform on ±ln ``S_MAX`` (the quoted
  covariance is scaled by ``s²``);
* each night an outlier with probability ``ε ~ Beta(*EPS_PRIOR)`` (Beta(1, 9)); an outlier's
  state is that of an independent orbit from the same prior at its night (the
  prior predictive of ``outlier_prior``);
* 5-20 nights in one season of about 30 d or two seasons a year apart;
  sigma / a log-uniform 1e-4 to 1e-2.

The census is paired: system ``i`` of seed ``s`` is drawn from
``SeedSequence([s, i])`` (a child stream for the simulation, another for the
PIT draw, which is consumed whatever the status), so a seed list draws the
same systems in every run, whatever the fits do. Every system keeps a full
record (ok, failed or error), and every failed system's PITs are labelled
``pit_diagnostic`` and never enter a gated SBC.

Verdict (INCONCLUSIVE, never PASS, below ``MIN_FITTED`` fitted systems: the
criteria cannot fail on so few). Design §14, assertions 1-8, thresholds as
committed in ``design/orbit_census_criteria.md``:

The detectable stratum is defined from the truth and the sampling, never from
a fit: the true period is Nyquist-sampled (at least twice the median
within-season gap between nights) and the true orbit's noise-free chi-square
excess over the best static model on the inlier nights is at least the excess
at which a 1 % test rejects that model with power 0.9 (columns ``nyquist_ok``,
``dchi2_true``, ``dchi2_thresh``, ``detectable``). Assertions 1, 2 and 5 gate
there only; elsewhere they are counted and tabled.

1. sky-plane retrieval: the truth's sky test quantity f (the time-averaged
   squared distance of a physical sky track, ``σ_i · track(θ_i)`` with the label
   ``σ_i`` drawn from its exact conditional given the draw, from the posterior-weighted mean
   track, ``SKY_GRID`` uniform times over the nights; the truth's track carries the simulated
   sign) above the 99 % quantile of f
   among the draws is a miss: FAIL if ``P(Binomial(n_detectable, 0.01) >=
   misses)`` is below ``P_MISS``. The element-space test of N (``in_99``) is
   reported only;
2. no cell at the floor (no failed system) in the detectable stratum;
3. SBC, chi-square over 10 bins: the randomized rank of f(true) among the
   draws (``pit_sky``) over all fitted in-domain systems, failing at
   p < ``P_FAIL`` (f depends on the data, which SBC allows: Modrák et al.
   2023; a truth-selected stratum breaks the rank guarantee, so the
   stratum's own uniformity is reported only). The PITs of N and ln P and the
   ranks of the invariants a, e, cos i, Omega and omega (folded) are reported
   only;
4. the coverage contract needs circular, positions-only truths against the
   exact toy reference: not drawn here, reported as not applicable;
5. e > 0.3: the true cell's best mode within chi2_d(0.99) of the truth
   (detectable stratum; outside it, reported only);
6. scale equivariance on every ``--equiv-every``-th system;
7. no dependence of failure on ln s at fixed effective noise (CMH, detectable
   stratum);
8. informativeness, reported only.

Strata: all nights outliers (B''), at most ``M_MIN`` inliers, the rest; the detectable stratum
and its two complements (report only); and a report-only stratum of at most
two outlier nights. With ``--oracle`` the true
cell is also rerun with the truth-started optimum as an extra mixture
component (diagnostic columns only: the fit, ``p_pos`` and the status are
untouched).

Writes ``<out>.csv`` (one row per system) and ``<out>.md`` (the summary).

usage: orbit_population.py [--n-sys 200] [--seed 0] [--out orbit_population]
                           [--oracle] [--equiv-every 10]
"""

import argparse
import csv
import dataclasses
import json
import math
import os
import resource
import subprocess
import time

os.environ.setdefault("OMP_NUM_THREADS", "2")

import jax
import jax.numpy as np
import numpy as onp
from scipy import optimize, stats

from virgil import _orbit_coordinates as oc
from virgil import orbit_cells as oc_cells
from virgil._precision import run_in
from virgil.mode_search import ESS_FAIL, M_MIN, W
from virgil.night_summaries import NightPeak, NightSummary
from virgil.orbits import KeplerOrbit

BLOCKS = ("vis",)
N_OBS = 40.0
N_PAR = 5  # fitted per night (p, v, ln f): chi2_min ~ s^2 chi2_{N_OBS - N_PAR}
FRAMES = onp.arange(6) * 0.02
T_SS = float(onp.sum((FRAMES - FRAMES.mean()) ** 2))
BINS = 10
P_FAIL = 1e-3  # SBC chi-square p-value that fails
P_MISS = 5e-3  # binomial tail of the misses of the 99 % set that fails
MIN_FITTED = 100  # fitted systems for a conclusive verdict
N_FULL = 200
P_TRUE = (2.0, 60.0)  # d, the truth prior's period range
SIGMA_F = 0.05  # quoted error of ln f
SD_TOL = (4.0, 0.01)  # §14 assertion 4: |gap| <= 4 sd + 0.01
E_ECC = 0.3  # assertion 5's eccentric stratum
CONFIDENCE = 0.99  # assertion 5's chi-square region
SCALES = (math.exp(-1.0), math.exp(1.5))  # assertion 6's c
STEP_COUNTS = (  # runs ended by the step cap, by stage; only the C-steps converge
    "varpro_runs", "varpro_capped", "pre_runs", "pre_capped", "c_runs", "c_capped",
)  # fmt: skip
INVARIANTS = ("a", "ecc", "cos_inc", "omega", "Omega")
STRATA = ("B2", "le_m", "rest")  # B'' (all outliers), <= m inliers, the rest
CLASSES = ("search_miss", "proposal_miss", "sentinel_fp", "support")
ALPHA_DET = (
    0.01  # false-alarm rate of the signal test (the plan's no-orbit verdict)
)
POWER_DET = 0.9  # power of the signal test at the threshold excess
SEASON_LEN = 30.0  # d: a simulated season's span
SEASON_SPACING = (
    365.25  # d: the second season starts this long after the first
)
# nights further apart are in different seasons: the midpoint of the longest
# in-season gap (SEASON_LEN) and the shortest between-season gap
# (SEASON_SPACING - SEASON_LEN)
SEASON_GAP = 0.5 * (SEASON_LEN + (SEASON_SPACING - SEASON_LEN))
SKY_GRID = 64  # declared budget: points of the uniform time grid of the sky test quantity


# ---------------------------------------------------------------------------
# The simulation: paired streams, independent of every fit
# ---------------------------------------------------------------------------


def streams(seed, i):
    """``(simulation, PIT)`` generators of system ``i`` of ``seed``: children
    of ``SeedSequence([seed, i])``, so the draws of a system depend on neither
    the other systems nor any fit."""
    sim, pit = onp.random.SeedSequence([seed, i]).spawn(2)
    return onp.random.default_rng(sim), onp.random.default_rng(pit)


def label_stream(seed, i):
    """The generator of the label draws ``σ_i`` of system ``i`` of ``seed``: a
    third child of ``SeedSequence([seed, i])``. Spawning a child leaves the
    others unchanged, so the simulation and PIT streams are those of `streams`."""
    return onp.random.default_rng(
        onp.random.SeedSequence([seed, i]).spawn(3)[2]
    )


def _loguniform(rng, lo, hi):
    return float(onp.exp(rng.uniform(onp.log(lo), onp.log(hi))))


def _orbit(rng, p_lo, p_hi, a):
    """An orbit from the fit's prior: P log-uniform on ``[p_lo, p_hi]``, e
    uniform, isotropic orientation, a random epoch of periastron."""
    period = _loguniform(rng, p_lo, p_hi)
    return KeplerOrbit(
        period, rng.uniform(0.0, period), rng.uniform(0.0, oc_cells.E_MAX),
        float(onp.degrees(onp.arccos(rng.uniform(-1.0, 1.0)))),
        rng.uniform(0.0, 360.0), rng.uniform(0.0, 360.0), a,
    )  # fmt: skip


def _state(orbit, times):
    """``(K, 4)`` position and velocity (mas, mas/d) of ``orbit``."""
    pos = onp.stack([onp.asarray(c, float) for c in orbit.relative(times)[:2]])
    vel = onp.stack(
        [onp.asarray(c, float) for c in orbit.relative_velocity(times)[:2]]
    )
    return onp.concatenate([pos.T, vel.T], axis=1)


def _night_state(orbit, times):
    """``(K, 4)`` ``(p̂, v̂)`` of ``orbit``: the least-squares line through its
    positions at the six frames of each night (design §5), as the summaries fit
    it; not the instantaneous state."""
    off = FRAMES - FRAMES.mean()
    times = onp.asarray(times, float)
    pos = _state(orbit, (times[:, None] + off).ravel())[:, :2].reshape(
        len(times), len(off), 2
    )
    return onp.concatenate(
        [pos.mean(1), onp.einsum("f,kfc->kc", off / T_SS, pos)], axis=1
    )


def _cov(sigma):
    return onp.diag(
        [sigma**2, sigma**2, sigma**2 / T_SS, sigma**2 / T_SS, SIGMA_F**2]
    )


def _summaries(times, mean, sigma, chi2=None):
    """One-peak night summaries with the means ``mean`` ``(K, 5)`` and the
    within-night minimum χ² ``chi2`` ``(K,)`` (default: its mean for unit
    scale, ``N_OBS - N_PAR``)."""
    cov = _cov(sigma)
    if chi2 is None:
        chi2 = onp.full(len(times), N_OBS - N_PAR)
    out = []
    for t, m, c2 in zip(times, mean, chi2):
        peak = NightPeak(
            m, cov, -30.0, onp.array([N_OBS]), onp.array([float(c2)]),
            onp.linalg.inv(cov)[None],
        )  # fmt: skip
        out.append(NightSummary(float(t), T_SS, 6, BLOCKS, (peak,), -200.0))
    return tuple(out)


def draw_nights(rng):
    """Nights (5 to 20; one season or two a year apart), ``a`` and the quoted
    error: everything the default period domain depends on."""
    while True:
        k = int(rng.integers(5, 21))
        two = rng.uniform() < 0.5
        if two:
            k1 = int(rng.integers(2, k - 1))
            times = onp.r_[
                rng.uniform(0.0, SEASON_LEN, k1),
                SEASON_SPACING + rng.uniform(0.0, SEASON_LEN, k - k1),
            ]
        else:
            times = rng.uniform(0.0, SEASON_LEN, k)
        times = onp.unique(onp.sort(onp.round(times, 3)))
        if len(times) >= 5:
            break
    a = _loguniform(rng, *oc_cells.A_RANGE)
    rel = _loguniform(rng, 1e-4, 1e-2)
    return dict(times=times, a=a, rel=rel, two_seasons=two)


def fit_domain(nights):
    """``(default domain, fit domain)`` of ``nights``: the fit prior is the
    default domain cut to ``P_TRUE`` (``None`` if they do not overlap)."""
    sigma = nights["rel"] * nights["a"]
    blank = _summaries(
        nights["times"], onp.zeros((len(nights["times"]), 5)), sigma
    )
    default = oc_cells.period_domain(blank)
    lo, hi = max(default.p_cut, P_TRUE[0]), min(default.p_max, P_TRUE[1])
    if not lo < hi:
        return default, None
    return default, oc_cells.period_domain(blank, p_range=(lo, hi))


def _noise_outliers_and_chi2(rng, mean, outlier, ln_s, sigma):
    """``(mean, chi2)`` of the nights: ``chi2`` is s² χ²_{N_OBS - N_PAR} (the
    fit's summary model). An outlier night has its own scale s_k, log-uniform
    on [1/S_MAX, S_MAX], and its mean is scattered by s_k times the night's
    noise. Drawn after everything else in ``draw_system``, so the truth and
    the inlier means of a stream are as before the χ² draw; the extra draws
    only follow them."""
    k = len(mean)
    floor = math.log(oc_cells.FLUX_FLOOR)
    mean = mean.copy()
    scale = onp.full(k, math.exp(ln_s))
    smax = math.log(oc_cells.S_MAX)
    for i in onp.flatnonzero(outlier):
        scale[i] = math.exp(rng.uniform(-smax, smax))
        z = rng.multivariate_normal(onp.zeros(5), _cov(sigma))
        while not floor <= mean[i, 4] + scale[i] * z[4] <= 0.0:
            z[4] = SIGMA_F * rng.standard_normal()
        mean[i] += scale[i] * z
    return mean, scale**2 * rng.chisquare(N_OBS - N_PAR, size=k)


def draw_system(rng, nights, fit, jitter=False):
    """The truth (from the fit's prior) and the summaries, given the nights
    and the fit domain. With ``jitter`` (the census arm), a between-night
    jitter ``σ_J`` is drawn log-uniform on ``oc_cells.JITTER_RANGE`` (the fit's
    own prior, so SBC stays exact) after everything else, so the paired truth
    and means of a stream are those of the no-jitter arm until it is added to
    the inlier nights' means (``σ_J² / n_frames`` on ``p``, ``σ_J² / T_ss`` on
    ``v``, as the likelihood adds it)."""
    times, a, rel = nights["times"], nights["a"], nights["rel"]
    sigma = rel * a
    orbit = _orbit(rng, fit.p_cut, fit.p_max, a)
    period = float(orbit.period)
    k = len(times)
    sign = rng.choice([-1.0, 1.0])
    ln_f = onp.log(oc_cells.FLUX_FLOOR) * rng.uniform()
    ln_s = rng.uniform(-math.log(oc_cells.S_MAX), math.log(oc_cells.S_MAX))
    eps = rng.beta(*oc_cells.EPS_PRIOR)
    outlier = rng.uniform(size=k) < eps
    truth = onp.append(
        sign * _night_state(orbit, times), onp.full((k, 1), ln_f), 1
    )
    noise = rng.multivariate_normal(onp.zeros(5), _cov(sigma), size=k)
    mean = truth + math.exp(ln_s) * noise
    for i in onp.flatnonzero(outlier):  # an independent orbit, the prior
        other = _orbit(
            rng, fit.p_cut, fit.p_max, _loguniform(rng, *oc_cells.A_RANGE)
        )
        mean[i, :4] = _night_state(other, times[i : i + 1])[0]
        mean[i, 4] = onp.log(oc_cells.FLUX_FLOOR) * rng.uniform()
    mean, chi2 = _noise_outliers_and_chi2(rng, mean, outlier, ln_s, sigma)
    extra = {}
    if jitter:
        lo, hi = oc_cells.JITTER_RANGE
        sj = _loguniform(rng, lo, hi)
        z = rng.standard_normal((k, 4))
        scale = onp.array([1 / math.sqrt(len(FRAMES))] * 2 + [1 / math.sqrt(T_SS)] * 2)
        mean[~outlier, :4] += sj * scale * z[~outlier]
        extra = dict(jitter=float(sj))
    return dict(
        **extra,
        **nights, period=period, orbit=orbit, sign=float(sign), ln_s=ln_s,
        ln_f=float(ln_f), eps=float(eps), outliers=onp.flatnonzero(outlier),
        summaries=_summaries(times, mean, sigma, chi2),
    )  # fmt: skip


def stratum(k, n_outliers):
    """``B2`` (every night an outlier: the answer is the prior), ``le_m`` (at
    most ``M_MIN`` inliers left) or ``rest`` (informative)."""
    n_in = k - n_outliers
    return "B2" if n_in == 0 else "le_m" if n_in <= M_MIN else "rest"


def nyquist_ok(period, times):
    """Whether ``period`` is at least twice the median gap between consecutive
    nights of a season (gaps above ``SEASON_GAP`` are between seasons); false
    with no within-season gap. Not the smallest gap, which is too lenient
    (Eyer & Bartholdi 1999 on irregular sampling)."""
    gaps = onp.diff(onp.sort(onp.asarray(times, float)))
    gaps = gaps[gaps <= SEASON_GAP]
    return bool(len(gaps) and period >= 2.0 * onp.median(gaps))


def signal_excess(orbit, times, inliers, sigma):
    """``(dchi2, dof)`` of the best static model on the inlier nights.

    ``dchi2`` is the noncentrality: the noise-free χ² of the model of a
    constant sky position (fitted by weighted least squares) and zero
    velocity, to the true orbit's night states ``(p̂, v̂)`` at the inlier
    nights, each position coordinate with error ``sigma`` and each velocity
    coordinate ``sigma / sqrt(T_SS)`` (the simulator's noise). ``dof`` is
    that model's: 4 numbers a night minus the 2 fitted coordinates."""
    m = len(inliers)
    if m == 0:
        return math.nan, 0
    st = _night_state(orbit, onp.asarray(times, float)[inliers])
    pos, vel = st[:, :2], st[:, 2:]
    lam = onp.sum((pos - pos.mean(0)) ** 2) + T_SS * onp.sum(vel**2)
    return float(lam / sigma**2), 4 * m - 2


def required_excess(dof):
    """The noncentrality λ at which the best static model's χ² (``dof`` degrees
    of freedom) exceeds its ``1 - ALPHA_DET`` quantile with probability
    ``POWER_DET``: the root of the noncentral-χ² power equation."""
    cut = stats.chi2.ppf(1.0 - ALPHA_DET, dof)

    def gap(lam):
        return stats.ncx2.sf(cut, dof, lam) - POWER_DET

    hi = float(dof)
    while gap(hi) < 0.0:
        hi *= 2.0
    return float(optimize.brentq(gap, 0.0, hi))


def detectability(sysd):
    """The a priori detectable stratum, from the truth and the sampling only:
    the period is Nyquist-sampled (``nyquist_ok``) and the true orbit's
    noise-free χ² excess over the best static model on the inlier nights
    (``dchi2_true``) reaches ``dchi2_thresh``, the excess at which a
    ``ALPHA_DET`` test of that model rejects it with power ``POWER_DET``."""
    times = sysd["times"]
    inl = onp.setdiff1d(onp.arange(len(times)), sysd["outliers"])
    sigma = math.exp(sysd["ln_s"]) * sysd["rel"] * sysd["a"]
    d2, dof = signal_excess(sysd["orbit"], times, inl, sigma)
    thresh = required_excess(dof) if dof else math.nan
    nyq = nyquist_ok(sysd["period"], times)
    return dict(
        nyquist_ok=int(nyq), dchi2_true=d2, dchi2_thresh=thresh,
        detectable=int(nyq and d2 >= thresh),
    )  # fmt: skip


def describe(sysd, fit):
    """The columns every row carries, fitted or not."""
    k, n_out = len(sysd["times"]), len(sysd["outliers"])
    return dict(
        **detectability(sysd),
        period=sysd["period"], a=sysd["a"], rel=sysd["rel"], k=k,
        two_seasons=int(sysd["two_seasons"]), n_outliers=n_out,
        p_cut=fit.p_cut, p_max=fit.p_max, ln_s=sysd["ln_s"],
        eff_noise=math.exp(sysd["ln_s"]) * sysd["rel"], eps=sysd["eps"],
        ecc=float(sysd["orbit"].ecc), stratum=stratum(k, n_out),
        le2_outliers=int(n_out <= 2),
    )  # fmt: skip


# ---------------------------------------------------------------------------
# One system: the fit and its full record
# ---------------------------------------------------------------------------


def _pit_p(p, cells, fits, truth_p):
    """Mixture PIT of ln P: the cells' resample-move draws, or the cell's
    position where it has none."""
    ln_p, pit = math.log(truth_p), 0.0
    for pi, c, f in zip(p, cells, fits):
        if f.draws is not None:
            d = oc_cells.decode_particles(f)["period"]
            pit += pi * float(onp.mean(onp.log(onp.asarray(d)) < ln_p))
        else:
            pit += pi * float(c.p_hi < truth_p)
    return pit


def _pit_lns(p, fits, truth_ln_s):
    """Mixture PIT of ln s: the cells' resample-move draws weighted by ``p``
    (cells without draws left out, the rest renormalized)."""
    num, den = 0.0, 0.0
    for pi, f in zip(p, fits):
        if f.draws is None or not pi > 0:
            continue
        d = oc_cells.decode_particles(f)["ln_s"]
        num += pi * float(onp.mean(onp.asarray(d) < truth_ln_s))
        den += pi
    return num / den if den > 0 else math.nan


def invariant_pits(p, fits, truth, distinct=False):
    """Mixture PITs of the invariants of ``truth`` (a dict): each cell's
    resample-move draws, equally weighted (a draw repeated by a rejected move
    carries its mass), weighted by the cell's ``p``; ``distinct`` keeps each
    distinct draw once instead (a separate diagnostic, columns ``pitd_*``).
    Cells without draws (below ``δ / N_cells``) are left out and the rest
    renormalized."""
    num, den = dict.fromkeys(INVARIANTS, 0.0), 0.0
    for pi, f in zip(p, fits):
        if f.draws is None or not pi > 0:
            continue
        if distinct:
            f = dataclasses.replace(
                f, draws=onp.unique(onp.asarray(f.draws), axis=0)
            )
        dec = oc_cells.decode_particles(f)
        for name in INVARIANTS:
            draws = onp.asarray(dec[name])
            if name == "omega":  # orbit <-> -orbit is omega -> omega + 180
                draws = draws % 180.0
            num[name] += pi * float(onp.mean(draws < truth[name]))
        den += pi
    return {
        f"{'pitd' if distinct else 'pit'}_{n}": (
            num[n] / den if den > 0 else math.nan
        )
        for n in INVARIANTS
    }


def truth_invariants(sysd, t_anchor):
    """The folded invariants of the true orbit, through the fit's own map.
    ``omega`` is folded to [0, 180): a fit is right up to the orbit <->
    -orbit swap (omega -> omega + 180), the orientation degeneracy."""
    z, zeta = oc.campbell_to_coordinates(sysd["orbit"], t_anchor)
    inv = oc.invariants(sysd["period"], z, zeta)
    out = {n: float(inv[n]) for n in INVARIANTS}
    out["omega"] %= 180.0
    return out


def sky_features(space, c, t_rel):
    """``(n, G, 2)`` sky positions ``(x, y)`` (mas) of each draw of ``c`` (orbit
    coordinates, as ``decode_particles``) at ``t_rel`` (days from the
    anchor): the track of the orbit, before the brighter/fainter label."""
    t = np.asarray(t_rel, space.dtype)
    pos = jax.vmap(
        lambda p, z, zt: space.states_at(t, p, oc.z_to_ti(z), zt)[:, :2]
    )(c["period"], c["z"], c["zeta"])
    return onp.asarray(pos, float)


def label_probability(fit):
    """``(n,)`` ``P(σ = +1 | θ_i, D)`` of each posterior draw of ``fit``: the
    softmax of the two terms of the label sum in the target (the exact
    conditional of the label given the draw)."""
    ll = jax.vmap(fit.space.sign_log_likelihood)(np.asarray(fit.draws))
    return onp.asarray(jax.nn.sigmoid(ll[:, 0] - ll[:, 1]), float)


def sky_distance(feat, mean):
    """``(n,)`` time-averaged squared sky distance of the tracks ``feat``
    ``(n, G, 2)`` from the mean track ``mean`` ``(G, 2)``: the test quantity
    f."""
    return ((feat - mean) ** 2).sum(-1).mean(-1)


def weighted_rank(f, w, f_true, u):
    """``(pit, miss, q99)``: the randomized PIT of ``f_true`` among the
    weighted draws ``f`` (``u`` breaks ties), whether it exceeds their 99 %
    quantile ``q99``."""
    f, w = onp.asarray(f, float), onp.asarray(w, float)
    w = w / w.sum()
    pit = float(w[f < f_true].sum() + u * w[f == f_true].sum())
    order = onp.argsort(f)
    cum = onp.cumsum(w[order])
    q99 = float(f[order][min(onp.searchsorted(cum, CONFIDENCE), len(f) - 1)])
    return pit, int(f_true > q99), q99


def sky_columns(sysd, out, p, u, t_anchor, true_i, rng=None):
    """``pit_sky``, ``sky_miss`` (and ``sky_f``, ``sky_q99``): the rank of the
    truth's sky test quantity f among the cells' draws, weighted by ``p`` (each
    cell's draws equally; cells without draws left out). f is the distance of a
    track from the posterior-weighted mean track on ``SKY_GRID`` uniform times
    over the span of the nights. The target sums out the brighter/fainter label
    ``σ``, so each draw θ_i is given ``σ_i`` from its exact conditional
    (``label_probability``, drawn from ``rng``); the physical track is
    ``σ_i · track(θ_i)`` and the truth's is the simulator's ``sign · track``.
    No draws at all: no PIT and a miss."""
    rng = onp.random.default_rng(0) if rng is None else rng
    nan = dict(pit_sky=math.nan, sky_miss=1, sky_f=math.nan, sky_q99=math.nan)
    cells = [
        (pi, f) for pi, f in zip(p, out.fits) if f.draws is not None and pi > 0
    ]
    if not cells:
        return nan
    times = sysd["times"]
    t_rel = onp.linspace(times[0], times[-1], SKY_GRID) - t_anchor
    feats, weights = [], []
    for pi, f in cells:
        with run_in(onp.dtype(f.space.dtype).name):
            feat_f = sky_features(f.space, oc_cells.decode_particles(f), t_rel)
            sigma = onp.where(
                rng.uniform(size=len(feat_f)) < label_probability(f), 1.0, -1.0
            )
        feats.append(sigma[:, None, None] * feat_f)
        weights.append(onp.full(len(feats[-1]), pi / len(feats[-1])))
    feat, w = onp.concatenate(feats), onp.concatenate(weights)
    mean = (w[:, None, None] * feat).sum(0) / w.sum()
    space = out.fits[true_i].space
    z, zeta = oc.campbell_to_coordinates(sysd["orbit"], t_anchor)
    truth = dict(
        period=np.asarray([sysd["period"]]), z=np.asarray(z)[None],
        zeta=np.asarray(zeta)[None],
    )  # fmt: skip
    with run_in(onp.dtype(space.dtype).name):
        track = sysd.get("sign", 1.0) * sky_features(space, truth, t_rel)
        f_true = float(sky_distance(track, mean)[0])
    pit, miss, q99 = weighted_rank(sky_distance(feat, mean), w, f_true, u)
    return dict(pit_sky=pit, sky_miss=miss, sky_f=f_true, sky_q99=q99)


def classify(rec, ess_fail=ESS_FAIL):
    """The class of a cell (the criteria file, precedence search miss >
    proposal miss > sentinel false positive > support), or ``"unresolved"``
    (flagged, with no oracle: not the true cell, or the oracle off), or
    ``"none"``. ``rec`` carries the cell's columns."""
    if rec.get("oracle_gap") is not None and not math.isnan(rec["oracle_gap"]):
        tol = SD_TOL[0] * rec["oracle_sd"] + SD_TOL[1]
        if rec["oracle_gap"] > tol:
            return "search_miss"
        if rec["oracle_basin"] > 0 and rec["ess"] < ess_fail:
            return "proposal_miss"
        if rec["laplace"] - rec["log_z"] > W and abs(rec["oracle_gap"]) <= tol:
            return "sentinel_fp"
    if rec["nonfinite"]:
        return "support"
    return "unresolved" if rec["flagged"] else "none"


def matched_cell(out, p):
    """The matched sample of the criteria file: the unflagged cell of largest
    ``p`` (``None`` if every cell is flagged)."""
    free = [
        i for i, f in enumerate(out.fits)
        if not (f.at_floor or f.unreliable or f.nonfinite)
    ]  # fmt: skip
    return max(free, key=lambda i: p[i]) if free else None


def old_rule(out):
    """``(matters_old, status_old)`` of the rule before the bounded
    ``matters`` (``max(ln Ẑ, Laplace)`` within ``W`` of the best): diagnostic
    columns from the same fit, never binding."""
    up = onp.array(
        [
            c.log_mass
            + max(
                [f.log_z] + [oc_cells.laplace_log_evidence(m) for m in f.modes]
            )
            for c, f in zip(out.cells, out.fits)
        ]
    )
    m = up >= up.max() - W
    bad = any(
        f.nonfinite or ((f.at_floor or f.unreliable) and mi)
        for f, mi in zip(out.fits, m)
    )
    return m, "failed" if bad else "ok"


def cell_records(out, true_i, oracle, matched=None):
    """One dict per flagged cell and for the true cell: truth, rank in
    ``upper``, ESS, ln Ẑ, best Laplace, the gap, the modes' inlier counts, the
    capped Newton runs, ``matters`` and the class."""
    rank = onp.argsort(onp.argsort(-out.upper))
    m_old = old_rule(out)[0]
    recs = []
    for i, (c, f) in enumerate(zip(out.cells, out.fits)):
        flagged = bool(f.at_floor or f.unreliable or f.nonfinite)
        if not (flagged or i == true_i or i == matched):
            continue
        lap = max(
            [oc_cells.laplace_log_evidence(m) for m in f.modes] or [-math.inf]
        )
        rec = dict(
            n=int(c.n), cell=i, true=int(i == true_i), flagged=flagged,
            matched=int(i == matched), rank=int(rank[i]),
            at_floor=bool(f.at_floor), unreliable=bool(f.unreliable),
            nonfinite=bool(f.nonfinite), ess=float(f.evidence.ess),
            log_z=float(f.log_z), laplace=float(lap),
            gap=float(lap - f.log_z), matters=bool(out.matters[i]),
            matters_old=bool(m_old[i]),
            n_inliers=[round(float(m.n_inliers), 2) for m in f.modes],
            **{k: int(f.info.get(k, 0)) for k in STEP_COUNTS},
        )  # fmt: skip
        if i == true_i and oracle:
            rec.update(oracle)
        rec["class"] = classify(rec)
        recs.append(rec)
    return recs


def _p_diagnostic(out):
    """``p(N)`` for the record: ``p_pos``, or for a failed system the same
    normalization over the cells' estimates (a diagnostic, never reported)."""
    if not out.failed.any():
        return out.p_pos
    log_p = onp.array(
        [c.log_mass + f.log_z for c, f in zip(out.cells, out.fits)]
    )
    log_p = onp.where(onp.isfinite(log_p), log_p, -onp.inf)
    if not onp.isfinite(log_p).any():
        return onp.full(len(log_p), onp.nan)
    return onp.exp(log_p - onp.logaddexp.reduce(log_p))


def sbc_columns(sysd, out, p, u, true_i, t_anchor):
    """The record of the posterior given the truth: ``p_true``, the 99 % set,
    the PITs of N, ln P and the invariants, the KL divergence from the prior."""
    ok = bool(onp.isfinite(p).all())
    p = onp.nan_to_num(p)  # no estimate at all: every record is empty
    order = onp.argsort(-p)
    keep = order[: int(onp.searchsorted(onp.cumsum(p[order]), 0.99)) + 1]
    prior = onp.exp([c.log_mass for c in out.cells])
    row = dict(
        p_true=float(p[true_i]), in_99=int(true_i in keep) if ok else 0,
        n_set=len(keep) if ok else len(p),
        pit_diagnostic=int(out.failed.any()),
        kl_prior=float(onp.sum(p * onp.log(onp.maximum(p, 1e-300) / prior)))
        if ok else math.nan,
        pit_n=float(p[:true_i].sum() + u * p[true_i]),
        pit_lnp=_pit_p(p, out.cells, out.fits, sysd["period"]),
        pit_lns=_pit_lns(p, out.fits, sysd["ln_s"]),
    )  # fmt: skip
    truth = truth_invariants(sysd, t_anchor)
    row.update(invariant_pits(p, out.fits, truth))
    row.update(invariant_pits(p, out.fits, truth, distinct=True))
    return row


def truth_columns(sysd, out, true_i, t_anchor, oracle, key):
    """The truth-aware diagnostics of the true cell: the Mahalanobis distance of
    its best mode (assertion 5) and, with ``oracle``, the mass oracle."""
    fit = out.fits[true_i]
    x_true = oc_cells.truth_chart(
        fit.space, sysd["orbit"], t_anchor, sysd["ln_f"], sysd["ln_s"],
        jitter=sysd.get("jitter"),
    )  # fmt: skip
    dim = fit.space.d
    maha = math.inf
    if fit.modes:
        best = max(fit.modes, key=oc_cells.laplace_log_evidence)
        maha = oc_cells.truth_distance2(fit.space, best, x_true)
    cols = dict(maha2=maha, maha_df=dim)
    orc = oc_cells.oracle_evidence(fit, x_true, key) if oracle else {}
    return cols, orc


def rescaled(summaries, c):
    """``summaries`` with every quoted error multiplied by ``c``: the same data
    under a different ``σ``, so the target is exactly equivariant."""
    out = []
    for s in summaries:
        peaks = tuple(
            dataclasses.replace(
                pk,
                cov=pk.cov * c**2,
                chi2=pk.chi2 / c**2,
                hessian=pk.hessian / c**2,
                loglike=pk.loglike
                - float(
                    onp.sum(pk.n * math.log(c) + 0.5 * pk.chi2 * (c**-2 - 1.0))
                ),
            )
            for pk in s.peaks
        )
        out.append(dataclasses.replace(s, peaks=peaks))
    return tuple(out)


def _inlier_set(fit):
    """Nights labelled inlier at the fit's best mode."""
    if not fit.modes:
        return None
    best = max(fit.modes, key=oc_cells.laplace_log_evidence)
    lab = oc_cells._labels(onp.asarray(best.x, fit.space.dtype), fit.space)
    return tuple(int(i) for i in onp.flatnonzero(onp.asarray(lab) >= 0))


def jitter_columns(sysd, out):
    """Report-only columns of the jitter arm (empty otherwise): the injected
    ``σ_J`` (``jitter_true``), its posterior median (``jitter_hat``), whether it
    lies in the central 98 % of the ``p(N)``-weighted importance draws
    (``jitter_in_99``), and the raw ``χ²/N`` of the dominant cell's best mode on
    the quoted errors (``chi2_quoted``) and on ``σ² + σ_J²`` (``chi2_total``).
    The no-jitter arm still reports the chi-squares (``jitter_hat`` 0)."""
    cols = {}
    try:
        best = int(onp.nanargmax(out.p_pos))
        chi2 = oc_cells.fit_chi2(out.fits[best])
        cols.update(chi2_quoted=chi2["quoted"], chi2_total=chi2["total"])
        if "jitter" not in sysd:
            return cols
        draws = oc_cells.jitter_draws(out.fits, out.p_pos)
        cols.update(jitter_true=sysd["jitter"], jitter_hat=out.jitter)
        if draws is not None:
            lo, hi = (oc_cells.jitter_quantile(draws, q) for q in (0.01, 0.99))
            cols["jitter_in_99"] = int(lo <= sysd["jitter"] <= hi)
    except Exception as err:  # a diagnostic never fails a system
        cols["jitter_diag_error"] = repr(err)[:200]
    return cols


def equivariance_columns(sysd, fit_dom, out, key, true_i):
    """Assertion 6: rerun with every ``σ`` times ``c``, when the true ``ln s / c``
    stays in the prior's bounds; compare ``ln p̂(N)`` over the cells with
    ``p >= δ`` (tolerance ``4 sqrt(sd_c² + sd_1²) + 0.01``) and the true cell's
    best-mode inlier set."""
    cols = {}
    ln_max = math.log(oc_cells.S_MAX)
    for j, c in enumerate(SCALES):
        if abs(sysd["ln_s"] - math.log(c)) >= ln_max:
            continue
        alt = oc_cells.position_cells(
            rescaled(sysd["summaries"], c),
            fit_dom,
            jax.random.fold_in(key, 10 + j),
            jitter=0.0,
            e_max=oc_cells.E_MAX,
        )
        if len(alt.cells) != len(out.cells) or alt.failed.any():
            cols[f"eqv{j}_viol"] = 1
            continue
        sd = [
            math.hypot(a.log_z_error, b.log_z_error)
            for a, b in zip(alt.fits, out.fits)
        ]
        big = out.p_pos >= oc_cells.DELTA
        dev = onp.abs(onp.log(alt.p_pos) - onp.log(out.p_pos))
        tol = SD_TOL[0] * onp.array(sd) + SD_TOL[1]
        same = _inlier_set(alt.fits[true_i]) == _inlier_set(out.fits[true_i])
        cols[f"eqv{j}_viol"] = int(bool((dev > tol)[big].any()) or not same)
        cols[f"eqv{j}_dev"] = float(onp.max(dev[big] - tol[big]))
    return cols


def persist_cells(out, recs, key, store):
    """Block variances of ``ln w`` (into the records) and the baseline arrays
    of every recorded cell (into ``store``, keyed ``c<N>_<name>``); a
    diagnostic that never fails a system."""
    cell_keys = jax.random.split(key, len(out.cells))  # as position_cells
    for rec in recs:
        f = out.fits[rec["cell"]]
        try:
            k = jax.random.fold_in(key, 200 + rec["cell"])
            rec.update(oc_cells.block_variance(f, k))
            for name, a in oc_cells.persist_arrays(
                f, cell_keys[rec["cell"]]
            ).items():
                store[f"c{rec['n']}_{name}"] = a
        except Exception as err:
            rec["persist_error"] = repr(err)[:200]


def run_one(
    sysd, fit, key, u, *, oracle=False, equiv=False, store=None, u_sky=None,
    label_rng=None,
):  # fmt: skip
    """Fit one system; returns its record (a dict). Every system, failed or
    not, keeps its truth, strata, ``p_true``, PITs (``pit_diagnostic`` if the
    system failed) and its flagged cells. ``u_sky`` (default ``u``) breaks
    ties in the sky PIT; ``label_rng`` draws the brighter/fainter labels of the
    sky draws."""
    s = sysd["summaries"]
    start = time.time()
    row = describe(sysd, fit)
    arm = "jitter" in sysd  # the jitter arm marginalizes σ_J, else none
    out = oc_cells.position_cells(
        s, fit, key, jitter=None if arm else 0.0, e_max=oc_cells.E_MAX
    )
    cells = out.cells
    true_i = next(
        i for i, c in enumerate(cells) if c.p_lo <= sysd["period"] <= c.p_hi
    )
    fits = out.fits
    row.update(
        status="failed" if out.failed.any() else "ok",
        status_old=old_rule(out)[1],
        seconds=time.time() - start, n_cells=len(cells),
        true_n=cells[true_i].n, no_mode=int(all(not f.modes for f in fits)),
        n_at_floor=sum(f.at_floor for f in fits),
        n_unreliable=sum(f.unreliable for f in fits),
        n_nonfinite=sum(f.nonfinite for f in fits),
        **{k: sum(f.info.get(k, 0) for f in fits) for k in STEP_COUNTS},
        min_ess=min(float(getattr(f.evidence, "ess", onp.nan)) for f in fits),
        n_undersampled=sum(f.undersampled for f in fits),
        **jitter_columns(sysd, out),
    )  # fmt: skip
    t_a = float(s[out.anchors[0]].t_ref)
    extra = time.time()
    p = _p_diagnostic(out)
    if out.failed.any():  # the draws a failed system never made, for its PITs
        for j, (f, pi) in enumerate(zip(fits, p)):
            if pi >= oc_cells.DELTA / len(cells) and not f.failed:
                try:
                    f.draws = oc_cells._draw(
                        f, jax.random.fold_in(key, 100 + j)
                    )
                except Exception as err:  # a diagnostic never fails a system
                    row["diag_error"] = repr(err)[:200]
    row.update(sbc_columns(sysd, out, p, u, true_i, t_a))
    try:
        row.update(
            sky_columns(
                sysd,
                out,
                p,
                u if u_sky is None else u_sky,
                t_a,
                true_i,
                label_rng,
            )
        )
    except (
        Exception
    ) as err:  # a diagnostic never fails a system; no PIT, a miss
        row.update(pit_sky=math.nan, sky_miss=1, diag_error=repr(err)[:200])
    try:
        cols, orc = truth_columns(
            sysd, out, true_i, t_a, oracle, jax.random.fold_in(key, 2)
        )
    except Exception as err:  # likewise
        cols, orc = dict(maha2=math.nan), {}
        row["diag_error"] = repr(err)[:200]
    row.update(cols)
    row.update(orc)
    matched = matched_cell(out, p) if store is not None else None
    recs = cell_records(out, true_i, orc, matched)
    if store is not None:
        persist_cells(out, recs, key, store)
    row.update(
        cells_json=json.dumps(recs),
        true_class=next(r["class"] for r in recs if r["true"]),
    )
    if equiv and row["status"] == "ok":
        row.update(equivariance_columns(sysd, fit, out, key, true_i))
    row["seconds_extra"] = time.time() - extra
    return row


# ---------------------------------------------------------------------------
# The verdict
# ---------------------------------------------------------------------------


def chi2_uniform(pit):
    """Chi-square p-value of ``pit`` against uniform on ``BINS`` bins (``nan``
    with fewer than ``5 BINS`` values: the expected counts are below 5)."""
    pit = [x for x in pit if x == x]
    if len(pit) < 5 * BINS:
        return float("nan")
    counts, _ = onp.histogram(pit, bins=BINS, range=(0.0, 1.0))
    return float(stats.chisquare(counts).pvalue)


def miss_tail(n_fitted, n_miss, rate=0.01):
    """``P(Binomial(n_fitted, rate) >= n_miss)``."""
    return float(stats.binom.sf(n_miss - 1, n_fitted, rate))


def terciles(values):
    """0, 1, 2: the tercile of each value."""
    values = onp.asarray(values, float)
    if len(values) == 0:
        return onp.zeros(0, int)
    edges = onp.quantile(values, [1 / 3, 2 / 3])
    return onp.searchsorted(edges, values, side="right")


def cmh_p(fail, group, layer):
    """Generalized Cochran-Mantel-Haenszel p-value of ``fail`` (0/1) against
    ``group`` (rows) across the strata ``layer``: 1 where no stratum
    has both outcomes."""
    fail, group, layer = map(onp.asarray, (fail, group, layer))
    n_g = int(group.max()) + 1 if len(group) else 0
    if n_g < 2:
        return 1.0
    stat, cov = onp.zeros(n_g - 1), onp.zeros((n_g - 1, n_g - 1))
    for h in onp.unique(layer):
        m = layer == h
        n, f = m.sum(), fail[m].sum()
        if n < 2 or f in (0, n):
            continue
        rows = onp.array([(group[m] == g).sum() for g in range(n_g)], float)
        obs = onp.array(
            [fail[m][group[m] == g].sum() for g in range(n_g)], float
        )
        stat += (obs - rows * f / n)[:-1]
        v = (f * (n - f) / (n**2 * (n - 1))) * (
            n * onp.diag(rows) - onp.outer(rows, rows)
        )
        cov += v[:-1, :-1]
    if not onp.any(cov):
        return 1.0
    q = float(stat @ onp.linalg.pinv(cov) @ stat)
    return float(stats.chi2.sf(q, n_g - 1))


def _a5(rows, detectable=True):
    """Assertion 5: ``(n, misses)`` over the e > ``E_ECC`` systems that have a
    distance (everything but errors) in the detectable stratum, or outside it
    (``detectable=False``, report only)."""
    sel = [
        r for r in rows
        if r.get("maha2") is not None and r["ecc"] > E_ECC
        and bool(r.get("detectable", 0)) == detectable
    ]  # fmt: skip
    # a row whose distance failed carries no df: a miss
    miss = sum(
        not r["maha2"] <= stats.chi2.ppf(CONFIDENCE, r.get("maha_df", 0))
        for r in sel
    )
    return len(sel), miss


def _a6(rows):
    """Assertion 6: ``(reruns, violations)``."""
    runs = [r[k] for r in rows for k in ("eqv0_viol", "eqv1_viol") if k in r]
    return len(runs), int(sum(runs))


def _a7(rows):
    """Assertion 7: the CMH p-value of failure against the terciles of the true
    ``ln s`` within the terciles of ``s σ / a`` (the detectable stratum, where
    failure gates)."""
    sel = [
        r for r in rows
        if r["status"] in ("ok", "failed") and "ln_s" in r
        and r.get("detectable", 0)
    ]  # fmt: skip
    if len(sel) < 9:
        return math.nan
    fail = [int(r["status"] == "failed" or r.get("no_mode", 0)) for r in sel]
    return cmh_p(
        fail,
        terciles([r["ln_s"] for r in sel]),
        terciles([r["eff_noise"] for r in sel]),
    )


def stratum_lines(rows):
    """Per-stratum counts and cell classes: the data strata, the detectable
    stratum (the gated one) and its complements, and the report-only stratum."""
    lines = [
        "",
        "| stratum | drawn | ok | failed | error | missed (N, report only) | sky miss | search miss | proposal miss | sentinel fp | support | unresolved |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    groups = {s: [r for r in rows if r.get("stratum") == s] for s in STRATA}
    groups["<=2 outlier nights (report only)"] = [
        r for r in rows if r.get("le2_outliers")
    ]
    scored = [r for r in rows if "detectable" in r]
    groups["detectable (gated)"] = [r for r in scored if r["detectable"]]
    groups["not detectable: period not Nyquist-sampled (report only)"] = [
        r for r in scored if not r["nyquist_ok"]
    ]
    groups["not detectable: signal below threshold (report only)"] = [
        r for r in scored if r["nyquist_ok"] and not r["detectable"]
    ]
    for name, g in groups.items():
        cls = [
            c["class"] for r in g if "cells_json" in r
            for c in json.loads(r["cells_json"]) if c["flagged"] or c["true"]
        ]  # fmt: skip
        n = {
            s: sum(r["status"] == s for r in g)
            for s in ("ok", "failed", "error")
        }
        miss = sum(r["status"] == "ok" and not r["in_99"] for r in g)
        sky = sum(r["status"] == "ok" and bool(r.get("sky_miss")) for r in g)
        lines.append(
            f"| {name} | {len(g)} | {n['ok']} | {n['failed']} | {n['error']} | {miss} | {sky} | "
            + " | ".join(str(cls.count(c)) for c in (*CLASSES, "unresolved"))
            + " |"
        )
    return lines


def informativeness_lines(rows):
    """Assertion 8: the median 99 % set size and KL from the prior per
    ln s and sigma/a tercile (ok systems; reported, not gated)."""
    ok = [r for r in rows if r["status"] == "ok" and "kl_prior" in r]
    if len(ok) < 9:
        return []
    t_s = terciles([r["ln_s"] for r in ok])
    t_r = terciles([r["rel"] for r in ok])
    lines = [
        "",
        "| ln s | sigma/a | n | median 99 % set | median KL (nats) |",
        "|---|---|---|---|---|",
    ]
    for i in range(3):
        for j in range(3):
            g = [r for r, a, b in zip(ok, t_s, t_r) if a == i and b == j]
            if g:
                lines.append(
                    f"| {i} | {j} | {len(g)} | {onp.median([r['n_set'] for r in g]):.0f} | "
                    f"{onp.median([r['kl_prior'] for r in g]):.2f} |"
                )
    return lines


def jitter_lines(ok):
    """Report-only lines of the jitter arm (none without ``jitter_true``)."""
    arm = [r for r in ok if "jitter_in_99" in r]
    if not arm:
        return []
    ratio = onp.array([math.log10(r["jitter_hat"] / r["jitter_true"]) for r in arm])
    hit = onp.mean([r["jitter_in_99"] for r in arm])
    q = onp.array([r.get("chi2_quoted", math.nan) for r in arm], float)
    t = onp.array([r.get("chi2_total", math.nan) for r in arm], float)
    return [
        f"- jitter arm ({len(arm)} fitted systems; report only): log10(sigma_J hat / "
        f"true) median {onp.median(ratio):+.2f}, 16-84% [{onp.quantile(ratio, 0.16):+.2f}, "
        f"{onp.quantile(ratio, 0.84):+.2f}]; truth in the central 98%: {hit:.3f}",
        f"- jitter arm raw chi2/N of the best mode, quoted errors: median "
        f"{onp.nanmedian(q):.3g}; sigma^2 + sigma_J^2: median {onp.nanmedian(t):.3g}",
    ]


def summarize(rows, n_sys):
    """The markdown summary and the verdict: ``"PASS"``, ``"FAIL"`` or
    ``"INCONCLUSIVE"`` (too few fitted systems for the criteria to bite)."""
    ok = [r for r in rows if r["status"] == "ok"]
    outside = [r for r in rows if r["status"] == "outside_domain"]
    failed = [r for r in rows if r["status"] == "failed"]
    errors = [r for r in rows if r["status"] == "error"]
    miss = [r for r in ok if not r["in_99"]]
    p_n = chi2_uniform([r["pit_n"] for r in ok])
    p_p = chi2_uniform([r["pit_lnp"] for r in ok])
    p_s = chi2_uniform([r.get("pit_lns", math.nan) for r in ok])
    p_inv = {
        n: chi2_uniform([r.get(f"pit_{n}", math.nan) for r in ok])
        for n in INVARIANTS
    }
    tail = miss_tail(len(ok), len(miss))
    det = [r for r in ok if r.get("detectable", 0)]
    sky_miss = [r for r in det if r.get("sky_miss", 1)]
    sky_all = [r for r in ok if r.get("sky_miss", 1)]
    failed_det = [r for r in failed if r.get("detectable", 0)]
    tail_sky = miss_tail(len(det), len(sky_miss))
    p_sky = chi2_uniform([r.get("pit_sky", math.nan) for r in ok])
    p_sky_det = chi2_uniform([r.get("pit_sky", math.nan) for r in det])
    legacy = any(
        "detectable" not in r or ("pit_sky" not in r and r["status"] == "ok")
        for r in (*ok, *failed)
    )
    n5, m5 = _a5(rows)
    n5o, m5o = _a5(rows, detectable=False)
    n6, v6 = _a6(rows)
    p7 = _a7(rows)
    bad = (
        tail_sky < P_MISS
        or bool(failed_det)
        or bool(errors)
        or p_sky < P_FAIL
        or (n5 > 0 and miss_tail(n5, m5) < P_MISS)
        or v6 > 0
        or p7 < P_FAIL
    )
    enough = len(ok) >= MIN_FITTED and not legacy
    verdict = "FAIL" if bad else "PASS" if enough else "INCONCLUSIVE"
    sec = onp.array([r["seconds"] for r in ok]) if ok else onp.zeros(1)
    lines = [
        "# Orbit population test",
        "",
        f"- systems drawn: {n_sys}; fitted: {len(ok)}",
        f"- truth outside the default period domain (stratum B'): {len(outside)}",
        f"- detectable stratum (true period Nyquist-sampled and signal at 90 % power for a "
        f"{ALPHA_DET} test): {len(det)} of {len(ok)} fitted",
        f"- cell at floor (failed), detectable stratum (gated): {len(failed_det)}; "
        f"outside it (report only): {len(failed) - len(failed_det)}",
        f"- errors: {len(errors)}",
        f"- sky-plane miss, detectable stratum (gated; f(true) above the 99 % "
        f"quantile of f): {len(sky_miss)} of {len(det)} "
        f"(P(Binomial >= misses at 1 %) = {tail_sky:.3g}; fails below {P_MISS})",
        f"- sky-plane miss, all fitted (report only): {len(sky_all)} of {len(ok)}",
        f"- sky-plane SBC chi-square p, all fitted in-domain systems (gated; "
        f"randomized rank of f(true), {BINS} bins): {p_sky:.3g}",
        f"- sky-plane SBC chi-square p, detectable stratum (report only: the "
        f"stratum is selected on the truth, so uniformity is approximate): "
        f"{p_sky_det:.3g}",
        f"- true N outside the 99 % set (report only): {len(miss)} of {len(ok)} "
        f"(P(Binomial >= misses at 1 %) = {tail:.3g})",
        f"- SBC chi-square p, N (randomized PIT, {BINS} bins; report only): {p_n:.3g}",
        f"- SBC chi-square p, ln P (resample-move draws; report only): {p_p:.3g}",
        f"- SBC chi-square p, ln s (resample-move draws; reported, not yet "
        f"gated): {p_s:.3g}",
        "- SBC chi-square p, invariants (distinct weighted draws; "
        + ", ".join(f"{n} {v:.3g}" for n, v in p_inv.items())
        + ")",
        "- assertion 4 (exact toy reference, circular positions-only truths): "
        "not applicable to this population, evaluated by the CI exact-reference tests",
        f"- assertion 5, detectable stratum (gated; e > {E_ECC}: true cell's best mode within chi2_d({CONFIDENCE})): "
        f"{m5} misses of {n5}"
        + (f" (P = {miss_tail(n5, m5):.3g})" if n5 else ""),
        f"- assertion 5, outside the detectable stratum (report only): "
        f"{m5o} misses of {n5o}",
        f"- assertion 6 (scale equivariance): {v6} violations in {n6} reruns",
        f"- assertion 7 (CMH, failure vs ln s at fixed effective noise): p = {p7:.3g}",
        *jitter_lines(ok),
        f"- seconds per system: median {onp.median(sec):.1f}, max {sec.max():.1f}",
        f"- peak RSS: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30:.2f} GB",
        "",
        f"Verdict: {verdict}"
        + (
            " (rows lack the detectability or sky-plane columns: rescore with a rerun)"
            if verdict == "INCONCLUSIVE" and legacy
            else f" (fewer than {MIN_FITTED} systems fitted)"
            if verdict == "INCONCLUSIVE"
            else ""
        ),
        "",
        "Failed and error systems' PITs are diagnostic (`pit_diagnostic`) and "
        "never enter the SBC above.",
    ]
    lines += stratum_lines(rows) + informativeness_lines(rows)
    if miss:
        lines += ["", "Missed (period, k, rel, p_true): "] + [
            f"- {r['period']:.3f} d, K={r['k']}, sigma/a={r['rel']:.2e}, p={r['p_true']:.2e}"
            for r in miss
        ]
    return "\n".join(lines) + "\n", verdict


def git_hash():
    """The full hash of ``HEAD`` (``""`` outside a checkout)."""
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
            check=True, cwd=os.path.dirname(os.path.abspath(__file__)),
        ).stdout.strip()  # fmt: skip
    except (OSError, subprocess.CalledProcessError):
        return ""


def run_population(
    seed, start, n_sys, oracle, equiv_every, persist=None, jitter=False
):
    """The records of systems ``start .. start + n_sys - 1`` of ``seed``."""
    root = jax.random.key(seed)
    rows = []
    # float64 throughout: the simulator and the inlier set run outside the
    # fit, and a float32 mean anomaly is off by ~σ/a for short periods
    with run_in("float64"):
        for i in range(start, start + n_sys):
            rng, pit_rng = streams(seed, i)
            u = float(pit_rng.uniform())  # consumed whatever the status
            u_sky = float(pit_rng.uniform())  # drawn after u: u is unchanged
            label_rng = label_stream(
                seed, i
            )  # a new stream: the rest unchanged
            key = jax.random.fold_in(root, i)
            nights = draw_nights(rng)
            default, fit = fit_domain(nights)
            # the selection event: a log-uniform P on P_TRUE inside the default
            # domain; given it, P is log-uniform on the fit domain (drawn in
            # draw_system, the domain being a function of the nights only)
            p_try = _loguniform(rng, *P_TRUE)
            base = dict(
                period=p_try, a=nights["a"], rel=nights["rel"],
                k=len(nights["times"]), p_cut=default.p_cut, p_max=default.p_max,
            )  # fmt: skip
            if fit is None or not default.p_cut <= p_try <= default.p_max:
                row = {**base, "status": "outside_domain"}
            else:
                sysd = None
                try:
                    sysd = draw_system(rng, nights, fit, jitter)
                    store = {} if persist else None
                    row = run_one(
                        sysd, fit, key, u, oracle=oracle,
                        equiv=bool(equiv_every) and not jitter
                        and i % equiv_every == 0,
                        store=store, u_sky=u_sky,
                        label_rng=label_rng,
                    )  # fmt: skip
                    if store:
                        onp.savez_compressed(
                            f"{persist}_s{seed}_{i}.npz",
                            git_hash=git_hash(),
                            seed=seed,
                            sys=i,
                            **store,
                        )
                except Exception as err:  # an error is a recorded failure
                    row = {**base, "status": "error", "error": repr(err)[:200]}
                    if sysd is not None:
                        row.update(describe(sysd, fit))
            rows.append(dict(seed=seed, sys=i, **row))
            print(
                i,
                row.get("status"),
                f"{row.get('seconds', 0):.0f}s",
                flush=True,
            )
    return rows


def read_rows(paths):
    """The rows of part files written by ``main`` (empty cells dropped,
    numbers parsed), for ``--merge``."""
    rows = []
    for path in paths:
        with open(path) as fh:
            for raw in csv.DictReader(fh):
                row = {}
                for k, v in raw.items():
                    if v == "":
                        continue
                    try:
                        row[k] = float(v)
                    except ValueError:
                        row[k] = v
                rows.append(row)
    return rows


def finish(rows, n_sys, out):
    """Write ``out.csv`` and ``out.md``; the process exit code is the verdict."""
    names = sorted({k for r in rows for k in r})
    with open(f"{out}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=names)
        w.writeheader()
        w.writerows(rows)
    text, verdict = summarize(rows, n_sys)
    with open(f"{out}.md", "w") as fh:
        fh.write(text)
    print(text)
    raise SystemExit(0 if verdict == "PASS" else 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--n-sys", type=int, default=N_FULL)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="orbit_population")
    ap.add_argument(
        "--oracle", action="store_true",
        help="rerun the true cell with the truth-started optimum (diagnostic columns only)",
    )  # fmt: skip
    ap.add_argument(
        "--equiv-every", type=int, default=10,
        help="assertion 6 reruns every this many systems (0: none)",
    )  # fmt: skip
    ap.add_argument(
        "--persist", action="store_true",
        help="per-system npz of the modes and IS draws of the flagged, true and matched cells, and their block variances of ln w",
    )  # fmt: skip
    ap.add_argument(
        "--jitter", action="store_true",
        help="the jitter arm: inject a between-night jitter drawn from the fit's prior (log-uniform 1 uas to 3 mas) and marginalize it in the fit; scale equivariance is skipped",
    )  # fmt: skip
    ap.add_argument("--start", type=int, default=0, help="first system index")
    ap.add_argument(
        "--merge", nargs="+", metavar="CSV",
        help="summarize the part files of several tasks instead of running",
    )  # fmt: skip
    args = ap.parse_args()
    if args.merge:
        rows = read_rows(args.merge)
        finish(
            sorted(rows, key=lambda r: (r["seed"], r["sys"])),
            len(rows),
            args.out,
        )
    rows = run_population(
        args.seed,
        args.start,
        args.n_sys,
        args.oracle,
        args.equiv_every,
        args.out if args.persist else None,
        args.jitter,
    )
    finish(rows, args.n_sys, args.out)


if __name__ == "__main__":
    main()
