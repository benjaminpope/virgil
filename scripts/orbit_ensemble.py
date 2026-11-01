"""Simulate an ensemble of orbits to evaluate position-level cell recovery."""

import argparse
import csv
import math
import os
import resource
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=true")

import jax
import numpy as onp
from virgil import orbit_cells as oc_cells
from virgil.night_summaries import NightPeak, NightSummary
from virgil.orbits import KeplerOrbit

BLOCKS = ("vis",)
N_OBS = 40.0
FRAMES = onp.arange(6) * 0.02
T_SS = float(onp.sum((FRAMES - FRAMES.mean()) ** 2))
LOG_B = -55.0


def _state(orbit, times):
    """The position and velocity vectors of ``orbit`` at ``times``."""
    pos = onp.stack([onp.asarray(c, float) for c in orbit.relative(times)[:2]])
    vel = onp.stack(
        [onp.asarray(c, float) for c in orbit.relative_velocity(times)[:2]]
    )
    return onp.concatenate([pos.T, vel.T], axis=1)


def _summaries(
    orbit, times, sigma=0.05, ln_f=-2.0, seed=0, sign=1.0, sigma_f=0.05
):
    """Night summaries with Gaussian peaks at truth plus covariance noise."""
    rng = onp.random.default_rng(seed)
    truth = _state(orbit, onp.asarray(times, float))
    cov = onp.diag(
        [sigma**2, sigma**2, sigma**2 / T_SS, sigma**2 / T_SS, sigma_f**2]
    )
    inv_cov = onp.linalg.inv(cov)[None]
    n_obs = onp.array([N_OBS])
    out = []
    for t, s in zip(times, truth):
        noise = rng.multivariate_normal(onp.zeros(5), cov)
        mean = onp.append(sign * s, ln_f) + noise
        peak = NightPeak(mean, cov, -30.0, n_obs, n_obs, inv_cov)
        out.append(NightSummary(float(t), T_SS, 6, BLOCKS, (peak,), -200.0))
    return tuple(out)


def draw_population(rng, n):
    """Draw a random population of binary orbits with realistic night sampling.

    Parameters
    ----------
    rng : numpy.random.Generator
        Random number generator.
    n : int
        Number of orbits to draw.

    Returns
    -------
    list of dict
        Drawn orbit parameters and observation times.
    """
    rows = []
    for i in range(n):
        period = float(onp.exp(rng.uniform(onp.log(2.0), onp.log(60.0))))
        ecc = float(rng.uniform(0.0, 0.8))
        cos_i = float(rng.uniform(-1.0, 1.0))
        # Keep inclination strictly in [0, 180) degrees
        inc_rad = onp.arccos(onp.clip(cos_i, -1.0 + 1e-12, 1.0 - 1e-12))
        inc_deg = float(onp.rad2deg(inc_rad))
        omega_deg = float(rng.uniform(0.0, 360.0))
        Omega_deg = float(rng.uniform(0.0, 360.0))
        t_peri = float(rng.uniform(0.0, period))
        a_mas = float(onp.exp(rng.uniform(onp.log(2.0), onp.log(30.0))))
        sigma = float(onp.exp(rng.uniform(onp.log(0.01), onp.log(0.1))))
        K = int(rng.integers(5, 21))

        if rng.uniform() < 0.5:
            layout = "one_season"
            times_set = set()
            while len(times_set) < K:
                draws = onp.round(
                    rng.uniform(0.0, 30.0, size=K - len(times_set)), 3
                )
                times_set.update(draws.tolist())
            times = onp.sort(onp.array(sorted(times_set), dtype=float))
        else:
            layout = "two_seasons"
            while True:
                mask = rng.uniform(size=K) < 0.5
                k1 = int(onp.sum(mask))
                k2 = K - k1
                if k1 >= 2 and k2 >= 2:
                    break
            t1_set = set()
            while len(t1_set) < k1:
                draws = onp.round(
                    rng.uniform(0.0, 30.0, size=k1 - len(t1_set)), 3
                )
                t1_set.update(draws.tolist())
            t2_set = set()
            while len(t2_set) < k2:
                draws = onp.round(
                    rng.uniform(0.0, 30.0, size=k2 - len(t2_set)), 3
                )
                t2_set.update(draws.tolist())
            t1 = onp.array(sorted(t1_set), dtype=float)
            t2 = onp.array(sorted(t2_set), dtype=float) + 365.25
            times = onp.sort(onp.round(onp.concatenate([t1, t2]), 3))

        span = float(times[-1] - times[0])
        rows.append(
            {
                "orbit_id": i,
                "period": period,
                "ecc": ecc,
                "cos_i": cos_i,
                "inc_deg": inc_deg,
                "omega_deg": omega_deg,
                "Omega_deg": Omega_deg,
                "t_peri": t_peri,
                "a_mas": a_mas,
                "sigma": sigma,
                "K": K,
                "layout": layout,
                "times": times,
                "span": span,
                "T": span,
            }
        )
    return rows


def run_one(params, key, *, n_particles, max_seconds, pilot):
    """Run position-level SMC on one simulated orbit and record results.

    Parameters
    ----------
    params : dict
        Drawn orbit parameters from ``draw_population``.
    key : jax.random.PRNGKey
        PRNG key for JAX.
    n_particles : int
        Number of particles per cell.
    max_seconds : float
        Maximum seconds per SMC leg.
    pilot : bool
        Whether to place cell edges using pilot run marginal valleys.

    Returns
    -------
    dict
        Results row recording truth recovery, cell statistics, and diagnostics.
    """
    t_start = time.perf_counter()
    orbit_id = params.get("orbit_id", 0)
    period = params["period"]
    times = params["times"]
    K = params["K"]
    span = params["span"]
    layout = params["layout"]

    base_row = {
        "orbit_id": orbit_id,
        "period": period,
        "ecc": params["ecc"],
        "cos_i": params["cos_i"],
        "inc_deg": params["inc_deg"],
        "omega_deg": params["omega_deg"],
        "Omega_deg": params["Omega_deg"],
        "t_peri": params["t_peri"],
        "a_mas": params["a_mas"],
        "sigma": params["sigma"],
        "K": K,
        "layout": layout,
        "span": span,
        "T": span,
        "times": times,
    }

    try:
        orbit = KeplerOrbit(
            period=period,
            dt_peri=params["t_peri"],
            ecc=params["ecc"],
            inc=params["inc_deg"],
            omega=params["omega_deg"],
            Omega=params["Omega_deg"],
            a_mas=params["a_mas"],
            t_ref=0.0,
        )
        key, subkey = jax.random.split(key)
        seed = int(jax.random.randint(subkey, (), 0, 2**31 - 1))
        summaries = _summaries(orbit, times, sigma=params["sigma"], seed=seed)
        domain = oc_cells.period_domain(
            summaries, night_length=0.1, resolution=3.0
        )
        log_outlier = onp.full(K, LOG_B)
        out = oc_cells.position_cells(
            summaries,
            domain,
            key,
            log_outlier=log_outlier,
            jitter=0.0,
            n_particles=n_particles,
            pilot=pilot,
            max_seconds=max_seconds,
        )

        n_cells = len(out.cells)
        matches = [
            i for i, c in enumerate(out.cells) if c.p_lo <= period <= c.p_hi
        ]
        if matches:
            true_n = int(matches[0])
            outside_domain = False
        else:
            true_n = -1
            outside_domain = True

        order = onp.argsort(-out.p_pos)
        cum_p = onp.cumsum(out.p_pos[order])
        cut_idx = int(onp.searchsorted(cum_p, 0.99))
        keep = order[: cut_idx + 1]

        if not outside_domain:
            in_99 = bool(true_n in keep)
            p_pos_true = float(out.p_pos[true_n])
            rank_true = int(onp.where(order == true_n)[0][0])
        else:
            in_99 = False
            p_pos_true = 0.0
            rank_true = -1

        top_cell = out.cells[int(order[0])]
        top_cell_n = int(top_cell.n)
        top_cell_p_lo = float(top_cell.p_lo)
        top_cell_p_hi = float(top_cell.p_hi)

        prior_mass = onp.array([math.exp(c.log_mass) for c in out.cells])
        flat = bool(onp.all(onp.abs(out.p_pos - prior_mass) < 1e-6))

        log_z_values = [g.log_z for g in out.legs]
        spread_ln_z = float(max(log_z_values) - min(log_z_values))
        exception_text = ""
        p_pos = out.p_pos

    except Exception as exc:
        n_cells = 0
        true_n = -1
        outside_domain = False
        in_99 = False
        p_pos_true = 0.0
        rank_true = -1
        top_cell_n = -1
        top_cell_p_lo = float("nan")
        top_cell_p_hi = float("nan")
        flat = False
        spread_ln_z = float("nan")
        exception_text = f"{type(exc).__name__}: {exc}"
        p_pos = onp.array([])

    wall_time = float(time.perf_counter() - t_start)
    rusage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_rss = float(
        rusage / (1024 * 1024) if sys.platform == "darwin" else rusage / 1024.0
    )

    result = dict(base_row)
    result.update(
        {
            "n_cells": n_cells,
            "number_of_cells": n_cells,
            "true_n": true_n,
            "outside_domain": outside_domain,
            "p_pos_true": p_pos_true,
            "p_pos_true_n": p_pos_true,
            "in_99": in_99,
            "in_99_set": in_99,
            "rank_true": rank_true,
            "top_cell_n": top_cell_n,
            "top_cell_p_lo": top_cell_p_lo,
            "top_cell_p_hi": top_cell_p_hi,
            "flat": flat,
            "is_flat": flat,
            "spread_ln_z": spread_ln_z,
            "wall_time": wall_time,
            "peak_rss": peak_rss,
            "exception": exception_text,
            "p_pos": p_pos,
        }
    )
    return result


def build_summary_markdown(rows):
    """Format overall and stratified ensemble statistics as Markdown."""
    n_total = len(rows)
    if n_total == 0:
        return "# Orbit Ensemble Simulation Summary\n\nNo orbits processed.\n"

    hits = sum(1 for r in rows if r["in_99"])
    outside = sum(1 for r in rows if r["outside_domain"])
    flats = sum(1 for r in rows if r["flat"])
    exceptions = sum(1 for r in rows if r["exception"])

    wall_times = [r["wall_time"] for r in rows]
    rss_values = [r["peak_rss"] for r in rows]

    med_wall = float(onp.median(wall_times))
    p90_wall = float(onp.percentile(wall_times, 90))
    med_rss = float(onp.median(rss_values))
    p90_rss = float(onp.percentile(rss_values, 90))

    lines = [
        "# Orbit Ensemble Simulation Summary",
        "",
        "## Overall Results",
        "",
        f"- **Total orbits:** {n_total}",
        f"- **Exceptions:** {exceptions}",
        f"- **Overall 99% credible set coverage:** "
        f"{hits / n_total:.1%} ({hits}/{n_total})",
        f"- **Outside domain:** {outside / n_total:.1%} ({outside}/{n_total})",
        f"- **Flagged flat (defect exposed):** "
        f"{flats / n_total:.1%} ({flats}/{n_total})",
        f"- **Wall time:** median {med_wall:.1f} s, 90th percentile {p90_wall:.1f} s",
        f"- **Peak RSS:** median {med_rss:.1f} MB, 90th percentile {p90_rss:.1f} MB",
        "",
        "## Coverage by Layout",
        "",
        "| Layout | Orbits | 99% Coverage | Flagged Flat | Outside Domain |",
        "| --- | --- | --- | --- | --- |",
    ]

    for layout in ("one_season", "two_seasons"):
        sub = [r for r in rows if r["layout"] == layout]
        n_sub = len(sub)
        if n_sub > 0:
            sub_hits = sum(1 for r in sub if r["in_99"])
            sub_flat = sum(1 for r in sub if r["flat"])
            sub_out = sum(1 for r in sub if r["outside_domain"])
            label = "One season" if layout == "one_season" else "Two seasons"
            lines.append(
                f"| {label} | {n_sub} | {sub_hits / n_sub:.1%} ({sub_hits}/{n_sub}) "
                f"| {sub_flat / n_sub:.1%} ({sub_flat}/{n_sub}) "
                f"| {sub_out / n_sub:.1%} ({sub_out}/{n_sub}) |"
            )

    lines.extend(
        [
            "",
            "## Coverage by Number of Nights K (Terciles)",
            "",
            "| Tercile | Range (K) | Orbits | 99% Coverage |",
            "| --- | --- | --- | --- |",
        ]
    )
    k_vals = onp.array([r["K"] for r in rows])
    q1_k, q2_k = onp.quantile(k_vals, [1 / 3, 2 / 3])
    k_bins = [
        ("Tercile 1", f"K <= {q1_k:.1f}", [r for r in rows if r["K"] <= q1_k]),
        (
            "Tercile 2",
            f"{q1_k:.1f} < K <= {q2_k:.1f}",
            [r for r in rows if q1_k < r["K"] <= q2_k],
        ),
        ("Tercile 3", f"K > {q2_k:.1f}", [r for r in rows if r["K"] > q2_k]),
    ]
    for name, rng_str, sub in k_bins:
        n_sub = len(sub)
        sub_hits = sum(1 for r in sub if r["in_99"])
        cov_str = (
            f"{sub_hits / n_sub:.1%} ({sub_hits}/{n_sub})" if n_sub else "N/A"
        )
        lines.append(f"| {name} | {rng_str} | {n_sub} | {cov_str} |")

    lines.extend(
        [
            "",
            "## Coverage by Precision σ (Terciles)",
            "",
            "| Tercile | Range (σ, mas) | Orbits | 99% Coverage |",
            "| --- | --- | --- | --- |",
        ]
    )
    s_vals = onp.array([r["sigma"] for r in rows])
    q1_s, q2_s = onp.quantile(s_vals, [1 / 3, 2 / 3])
    s_bins = [
        (
            "Tercile 1",
            f"σ <= {q1_s:.4f}",
            [r for r in rows if r["sigma"] <= q1_s],
        ),
        (
            "Tercile 2",
            f"{q1_s:.4f} < σ <= {q2_s:.4f}",
            [r for r in rows if q1_s < r["sigma"] <= q2_s],
        ),
        (
            "Tercile 3",
            f"σ > {q2_s:.4f}",
            [r for r in rows if r["sigma"] > q2_s],
        ),
    ]
    for name, rng_str, sub in s_bins:
        n_sub = len(sub)
        sub_hits = sum(1 for r in sub if r["in_99"])
        cov_str = (
            f"{sub_hits / n_sub:.1%} ({sub_hits}/{n_sub})" if n_sub else "N/A"
        )
        lines.append(f"| {name} | {rng_str} | {n_sub} | {cov_str} |")

    lines.extend(
        [
            "",
            "## Coverage by Eccentricity",
            "",
            "| Eccentricity | Orbits | 99% Coverage |",
            "| --- | --- | --- |",
        ]
    )
    e_bins = [
        ("e < 0.4", [r for r in rows if r["ecc"] < 0.4]),
        ("e >= 0.4", [r for r in rows if r["ecc"] >= 0.4]),
    ]
    for name, sub in e_bins:
        n_sub = len(sub)
        sub_hits = sum(1 for r in sub if r["in_99"])
        cov_str = (
            f"{sub_hits / n_sub:.1%} ({sub_hits}/{n_sub})" if n_sub else "N/A"
        )
        lines.append(f"| {name} | {n_sub} | {cov_str} |")

    lines.append("")
    return "\n".join(lines)


def main(argv=None):
    """Run simulated orbit ensemble from command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Simulated orbit ensemble for position-level cells."
    )
    parser.add_argument(
        "--n-orbits",
        type=int,
        default=50,
        help="Number of orbits to simulate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Random seed for population and SMC.",
    )
    parser.add_argument(
        "--n-particles",
        type=int,
        default=512,
        help="Particles per cell (default 512).",
    )
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=900.0,
        help="Maximum seconds per leg (default 900).",
    )
    parser.add_argument(
        "--pilot",
        action="store_true",
        default=False,
        help="Place cell edges at valleys of pilot run marginal.",
    )
    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="Output directory for ensemble.csv and summary.md.",
    )
    args = parser.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)
    rng = onp.random.default_rng(args.seed)
    population = draw_population(rng, args.n_orbits)

    csv_path = os.path.join(args.out, "ensemble.csv")
    summary_path = os.path.join(args.out, "summary.md")

    key = jax.random.key(args.seed)
    keys = jax.random.split(key, args.n_orbits)

    rows = []
    csv_fields = [
        "orbit_id",
        "period",
        "ecc",
        "cos_i",
        "inc_deg",
        "omega_deg",
        "Omega_deg",
        "t_peri",
        "a_mas",
        "sigma",
        "K",
        "layout",
        "span",
        "n_cells",
        "true_n",
        "outside_domain",
        "p_pos_true",
        "in_99",
        "rank_true",
        "top_cell_n",
        "top_cell_p_lo",
        "top_cell_p_hi",
        "flat",
        "spread_ln_z",
        "wall_time",
        "peak_rss",
        "exception",
        "times",
    ]

    with open(csv_path, "w", newline="") as f_csv:
        writer = csv.DictWriter(
            f_csv, fieldnames=csv_fields, extrasaction="ignore"
        )
        writer.writeheader()

        for i, (params, k) in enumerate(zip(population, keys)):
            row = run_one(
                params,
                k,
                n_particles=args.n_particles,
                max_seconds=args.max_seconds,
                pilot=args.pilot,
            )
            rows.append(row)

            row_for_csv = dict(row)
            row_for_csv["times"] = ";".join(
                f"{t:.3f}" for t in row_for_csv["times"]
            )
            writer.writerow(row_for_csv)
            f_csv.flush()

            status_str = "HIT" if row["in_99"] else "MISS"
            if row["outside_domain"]:
                status_str = "OUTSIDE"
            elif row["flat"]:
                status_str = "FLAT"

            print(
                f"[{i + 1:2d}/{args.n_orbits}] orbit={i:2d} "
                f"layout={row['layout']:11s} K={row['K']:2d} "
                f"P={row['period']:5.2f}d cells={row['n_cells']:2d} "
                f"status={status_str:7s} wall={row['wall_time']:5.1f}s "
                f"rss={row['peak_rss']:5.1f}MB",
                flush=True,
            )

    summary_content = build_summary_markdown(rows)
    with open(summary_path, "w") as f_sum:
        f_sum.write(summary_content)

    print(f"\nSaved ensemble results to {csv_path} and {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
