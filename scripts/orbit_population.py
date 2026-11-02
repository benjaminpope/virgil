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
* each night an outlier with probability ``ε ~ Beta(1/2, 1/2)``; an outlier's
  state is that of an independent orbit from the same prior at its night (the
  prior predictive of ``outlier_prior``);
* 5-20 nights in one season of about 30 d or two seasons a year apart;
  sigma / a log-uniform 1e-4 to 1e-2.

Verdict (INCONCLUSIVE, never PASS, below ``MIN_FITTED`` fitted systems: the
criteria cannot fail on so few):

* the true N in the 99 % set: FAIL if ``P(Binomial(n_fitted, 0.01) >= misses)``
  is below ``P_MISS``;
* no cell is at the floor;
* chi-square over 10 bins: the randomized PIT of N and the PIT of ln P from the
  resample-move draws, each failing at p < ``P_FAIL``.

Writes ``<out>.csv`` (one row per system) and ``<out>.md`` (the summary).

usage: orbit_population.py [--n-sys 200] [--seed 0] [--out orbit_population]
"""

import argparse
import csv
import math
import os
import resource
import time

os.environ.setdefault("OMP_NUM_THREADS", "2")

import jax
import numpy as onp
from scipy import stats

from virgil import orbit_cells as oc_cells
from virgil.night_summaries import NightPeak, NightSummary
from virgil.orbits import KeplerOrbit

BLOCKS = ("vis",)
N_OBS = 40.0
FRAMES = onp.arange(6) * 0.02
T_SS = float(onp.sum((FRAMES - FRAMES.mean()) ** 2))
BINS = 10
P_FAIL = 1e-3  # SBC chi-square p-value that fails
P_MISS = 5e-3  # binomial tail of the misses of the 99 % set that fails
MIN_FITTED = 100  # fitted systems for a conclusive verdict
N_FULL = 200
P_TRUE = (2.0, 60.0)  # d, the truth prior's period range
SIGMA_F = 0.05  # quoted error of ln f


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


def _cov(sigma):
    return onp.diag(
        [sigma**2, sigma**2, sigma**2 / T_SS, sigma**2 / T_SS, SIGMA_F**2]
    )


def _summaries(times, mean, sigma, scale=1.0):
    """One-peak night summaries with the means ``mean`` ``(K, 5)``; ``chi2``
    is that of data scattered by ``scale`` times the quoted errors."""
    cov = _cov(sigma)
    out = []
    for t, m in zip(times, mean):
        peak = NightPeak(
            m, cov, -30.0, onp.array([N_OBS]), onp.array([N_OBS * scale**2]),
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
                rng.uniform(0.0, 30.0, k1),
                365.25 + rng.uniform(0.0, 30.0, k - k1),
            ]
        else:
            times = rng.uniform(0.0, 30.0, k)
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
    default = oc_cells.period_domain(blank, night_length=0.1, resolution=3.0)
    lo, hi = max(default.p_cut, P_TRUE[0]), min(default.p_max, P_TRUE[1])
    if not lo < hi:
        return default, None
    return default, oc_cells.period_domain(
        blank, night_length=0.1, resolution=3.0, p_range=(lo, hi)
    )


def draw_system(rng, nights, fit):
    """The truth (from the fit's prior) and the summaries, given the nights
    and the fit domain."""
    times, a, rel = nights["times"], nights["a"], nights["rel"]
    sigma = rel * a
    orbit = _orbit(rng, fit.p_cut, fit.p_max, a)
    period = float(orbit.period)
    k = len(times)
    sign = rng.choice([-1.0, 1.0])
    ln_f = onp.log(oc_cells.FLUX_FLOOR) * rng.uniform()
    ln_s = rng.uniform(-math.log(oc_cells.S_MAX), math.log(oc_cells.S_MAX))
    eps = rng.beta(0.5, 0.5)
    outlier = rng.uniform(size=k) < eps
    truth = onp.append(sign * _state(orbit, times), onp.full((k, 1), ln_f), 1)
    mean = truth + math.exp(ln_s) * rng.multivariate_normal(
        onp.zeros(5), _cov(sigma), size=k
    )
    for i in onp.flatnonzero(outlier):  # an independent orbit, the prior
        other = _orbit(
            rng, fit.p_cut, fit.p_max, _loguniform(rng, *oc_cells.A_RANGE)
        )
        mean[i, :4] = _state(other, times[i : i + 1])[0]
        mean[i, 4] = onp.log(oc_cells.FLUX_FLOOR) * rng.uniform()
    return dict(
        **nights, period=period, orbit=orbit, ln_s=ln_s, eps=float(eps),
        outliers=onp.flatnonzero(outlier),
        summaries=_summaries(times, mean, sigma, math.exp(ln_s)),
    )  # fmt: skip


def run_one(sysd, fit, key, rng):
    """Fit one system; returns its record (a dict)."""
    s = sysd["summaries"]
    start = time.time()
    row = dict(
        period=sysd["period"], a=sysd["a"], rel=sysd["rel"],
        k=len(sysd["times"]), two_seasons=int(sysd["two_seasons"]),
        n_outliers=len(sysd["outliers"]), p_cut=fit.p_cut, p_max=fit.p_max,
    )  # fmt: skip
    out = oc_cells.position_cells(
        s, fit, key, jitter=0.0, e_max=oc_cells.E_MAX
    )
    cells = out.cells
    true_i = next(
        i for i, c in enumerate(cells) if c.p_lo <= sysd["period"] <= c.p_hi
    )
    row.update(
        status="failed" if out.failed.any() else "ok",
        seconds=time.time() - start, n_cells=len(cells),
        true_n=cells[true_i].n,
    )  # fmt: skip
    if out.failed.any():
        return row
    p = out.p_pos
    order = onp.argsort(-p)
    keep = order[: int(onp.searchsorted(onp.cumsum(p[order]), 0.99)) + 1]
    # cells run with N decreasing; the cumulative is over cell index
    below = p[:true_i].sum()
    pit_n = below + rng.uniform() * p[true_i]
    ln_p = math.log(sysd["period"])
    pit_p = 0.0
    for i, (c, f) in enumerate(zip(cells, out.fits)):
        if f.draws is not None:
            d = oc_cells.decode_particles(f)["period"]
            pit_p += p[i] * float(onp.mean(onp.log(onp.asarray(d)) < ln_p))
        else:
            pit_p += p[i] * float(c.p_hi < sysd["period"])
    row.update(
        in_99=int(true_i in keep), p_true=float(p[true_i]),
        n_set=len(keep), pit_n=float(pit_n), pit_lnp=float(pit_p),
        min_ess=min(
            float(getattr(f.evidence, "ess", onp.nan)) for f in out.fits
        ),
        n_undersampled=sum(f.undersampled for f in out.fits),
    )  # fmt: skip
    return row


def chi2_uniform(pit):
    """Chi-square p-value of ``pit`` against uniform on ``BINS`` bins (``nan``
    with fewer than ``5 BINS`` values: the expected counts are below 5)."""
    if len(pit) < 5 * BINS:
        return float("nan")
    counts, _ = onp.histogram(pit, bins=BINS, range=(0.0, 1.0))
    return float(stats.chisquare(counts).pvalue)


def miss_tail(n_fitted, n_miss, rate=0.01):
    """``P(Binomial(n_fitted, rate) >= n_miss)``."""
    return float(stats.binom.sf(n_miss - 1, n_fitted, rate))


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
    tail = miss_tail(len(ok), len(miss))
    bad = (
        tail < P_MISS
        or bool(failed)
        or bool(errors)
        or p_n < P_FAIL
        or p_p < P_FAIL
    )
    verdict = (
        "FAIL" if bad else "PASS" if len(ok) >= MIN_FITTED else "INCONCLUSIVE"
    )
    sec = onp.array([r["seconds"] for r in ok]) if ok else onp.zeros(1)
    lines = [
        "# Orbit population test",
        "",
        f"- systems drawn: {n_sys}; fitted: {len(ok)}",
        f"- truth outside the default period domain (stratum B'): {len(outside)}",
        f"- cell at floor (failed): {len(failed)}",
        f"- errors: {len(errors)}",
        f"- true N outside the 99 % set: {len(miss)} of {len(ok)} "
        f"(P(Binomial >= misses at 1 %) = {tail:.3g}; fails below {P_MISS})",
        f"- SBC chi-square p, N (randomized PIT, {BINS} bins): {p_n:.3g}",
        f"- SBC chi-square p, ln P (resample-move draws): {p_p:.3g}",
        f"- seconds per system: median {onp.median(sec):.1f}, max {sec.max():.1f}",
        f"- peak RSS: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30:.2f} GB",
        "",
        f"Verdict: {verdict}"
        + (
            f" (fewer than {MIN_FITTED} systems fitted)"
            if verdict == "INCONCLUSIVE"
            else ""
        ),
        "",
    ]
    if miss:
        lines += ["Missed (period, k, rel, p_true): "] + [
            f"- {r['period']:.3f} d, K={r['k']}, sigma/a={r['rel']:.2e}, p={r['p_true']:.2e}"
            for r in miss
        ]
    return "\n".join(lines) + "\n", verdict


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--n-sys", type=int, default=N_FULL)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="orbit_population")
    args = ap.parse_args()
    rng = onp.random.default_rng(args.seed)
    keys = jax.random.split(jax.random.key(args.seed), args.n_sys)
    rows = []
    for i in range(args.n_sys):
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
            try:
                sysd = draw_system(rng, nights, fit)
                row = run_one(sysd, fit, keys[i], rng)
            except Exception as err:  # an error is a recorded failure
                row = {**base, "status": "error", "error": repr(err)[:200]}
        rows.append(dict(sys=i, **row))
        print(
            i, row.get("status"), f"{row.get('seconds', 0):.0f}s", flush=True
        )
    names = sorted({k for r in rows for k in r})
    with open(f"{args.out}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=names)
        w.writeheader()
        w.writerows(rows)
    text, verdict = summarize(rows, args.n_sys)
    with open(f"{args.out}.md", "w") as fh:
        fh.write(text)
    print(text)
    raise SystemExit(0 if verdict == "PASS" else 1)


if __name__ == "__main__":
    main()
