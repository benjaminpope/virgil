"""Population test of the position-level cell fit (search, defensive IS, HMC).

Draws ``--n-sys`` simulated binaries (P log-uniform 2-60 d; e uniform 0-0.8;
isotropic orientation; a log-uniform over ``A_RANGE``; 5-20 nights in one
season of about 30 d or two seasons a year apart; sigma / a log-uniform
1e-4 to 1e-2; 0-2 outlier nights) and runs ``position_cells`` on each. A
truth outside the period domain is excluded and counted (stratum B').

Pass criteria at ``n_sys >= 200``:

* the true N is in the 99 % set (fails at 7 or more misses);
* no cell is at the floor;
* simulation-based calibration (Talts et al. 2018), chi-square over 10 bins:
  the randomized PIT of N, and the PIT of ln P from the resample-move draws,
  each failing at p < 0.001.

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
MISS_FAIL = 7  # misses of the 99 % set at n_sys = 200 that fail the run
P_FAIL = 1e-3  # SBC chi-square p-value that fails
N_FULL = 200


def draw_system(rng):
    """One simulated system: orbit, nights, error and outlier nights."""
    period = float(onp.exp(rng.uniform(onp.log(2.0), onp.log(60.0))))
    a = float(
        onp.exp(
            rng.uniform(
                onp.log(oc_cells.A_RANGE[0]), onp.log(oc_cells.A_RANGE[1])
            )
        )
    )
    orbit = KeplerOrbit(
        period, rng.uniform(0.0, period), rng.uniform(0.0, 0.8),
        float(onp.degrees(onp.arccos(rng.uniform(-1.0, 1.0)))),
        rng.uniform(0.0, 360.0), rng.uniform(0.0, 360.0), a,
    )  # fmt: skip
    k = int(rng.integers(5, 21))
    two = rng.uniform() < 0.5
    if two:
        k1 = int(rng.integers(2, k - 1))
        times = onp.r_[
            rng.uniform(0.0, 30.0, k1), 365.25 + rng.uniform(0.0, 30.0, k - k1)
        ]
    else:
        times = rng.uniform(0.0, 30.0, k)
    times = onp.unique(onp.sort(onp.round(times, 3)))
    rel = float(onp.exp(rng.uniform(onp.log(1e-4), onp.log(1e-2))))
    n_out = int(rng.integers(0, 3))
    outliers = rng.choice(
        len(times), size=min(n_out, len(times)), replace=False
    )
    return dict(
        period=period, a=a, orbit=orbit, times=times, rel=rel,
        two_seasons=two, outliers=onp.sort(outliers),
    )  # fmt: skip


def make_summaries(sysd, rng):
    """Night summaries: the truth plus a draw from the quoted covariance;
    an outlier night's peak sits at a random position and velocity."""
    orbit, times = sysd["orbit"], sysd["times"]
    sigma = sysd["rel"] * sysd["a"]
    pos = onp.stack([onp.asarray(c, float) for c in orbit.relative(times)[:2]])
    vel = onp.stack(
        [onp.asarray(c, float) for c in orbit.relative_velocity(times)[:2]]
    )
    truth = onp.concatenate([pos.T, vel.T], axis=1)
    cov = onp.diag(
        [sigma**2, sigma**2, sigma**2 / T_SS, sigma**2 / T_SS, 0.05**2]
    )
    out = []
    for i, (t, s) in enumerate(zip(times, truth)):
        mean = onp.append(s, -2.0) + rng.multivariate_normal(onp.zeros(5), cov)
        if i in sysd["outliers"]:
            r = oc_cells.A_RANGE[1]
            mean[:2] = rng.uniform(-r, r, 2)
            mean[2:4] = rng.normal(size=2) * onp.abs(truth[:, 2:4]).mean()
        peak = NightPeak(
            mean, cov, -30.0, onp.array([N_OBS]), onp.array([N_OBS]),
            onp.linalg.inv(cov)[None],
        )  # fmt: skip
        out.append(NightSummary(float(t), T_SS, 6, BLOCKS, (peak,), -200.0))
    return tuple(out)


def run_one(sysd, key, rng):
    """Fit one system; returns its record (a dict)."""
    s = make_summaries(sysd, rng)
    start = time.time()
    domain = oc_cells.period_domain(s, night_length=0.1, resolution=3.0)
    row = dict(
        period=sysd["period"], a=sysd["a"], rel=sysd["rel"],
        k=len(sysd["times"]), two_seasons=int(sysd["two_seasons"]),
        n_outliers=len(sysd["outliers"]), p_cut=domain.p_cut,
        p_max=domain.p_max,
    )  # fmt: skip
    if not domain.p_cut <= sysd["period"] <= domain.p_max:
        return {**row, "status": "outside_domain"}
    out = oc_cells.position_cells(s, domain, key, jitter=0.0)
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
    """Chi-square p-value of ``pit`` against uniform on ``BINS`` bins."""
    counts, _ = onp.histogram(pit, bins=BINS, range=(0.0, 1.0))
    return float(stats.chisquare(counts).pvalue) if len(pit) else float("nan")


def summarize(rows, n_sys):
    """The markdown summary and the pass flag."""
    ok = [r for r in rows if r["status"] == "ok"]
    outside = [r for r in rows if r["status"] == "outside_domain"]
    failed = [r for r in rows if r["status"] == "failed"]
    errors = [r for r in rows if r["status"] == "error"]
    miss = [r for r in ok if not r["in_99"]]
    p_n = chi2_uniform([r["pit_n"] for r in ok])
    p_p = chi2_uniform([r["pit_lnp"] for r in ok])
    full = n_sys >= N_FULL
    verdict = (
        len(miss) < MISS_FAIL
        and not failed
        and not errors
        and not (p_n < P_FAIL)
        and not (p_p < P_FAIL)
    )
    sec = onp.array([r["seconds"] for r in ok]) if ok else onp.zeros(1)
    lines = [
        "# Orbit population test",
        "",
        f"- systems drawn: {n_sys}; fitted: {len(ok)}",
        f"- truth outside the period domain (stratum B'): {len(outside)}",
        f"- cell at floor (failed): {len(failed)}",
        f"- errors: {len(errors)}",
        f"- true N outside the 99 % set: {len(miss)} (fails at {MISS_FAIL}+ of {N_FULL})",
        f"- SBC chi-square p, N (randomized PIT, {BINS} bins): {p_n:.3g}",
        f"- SBC chi-square p, ln P (resample-move draws): {p_p:.3g}",
        f"- seconds per system: median {onp.median(sec):.1f}, max {sec.max():.1f}",
        f"- peak RSS: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30:.2f} GB",
        "",
        f"Verdict: {'PASS' if verdict else 'FAIL'}"
        + (
            ""
            if full
            else f" (smoke run: the criteria are defined at n_sys = {N_FULL})"
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
        sysd = draw_system(rng)
        try:
            row = run_one(sysd, keys[i], rng)
        except Exception as err:  # keep going: an error is a recorded failure
            row = dict(
                period=sysd["period"], a=sysd["a"], rel=sysd["rel"],
                k=len(sysd["times"]), status="error", error=repr(err)[:200],
            )  # fmt: skip
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
    raise SystemExit(0 if verdict else 1)


if __name__ == "__main__":
    main()
