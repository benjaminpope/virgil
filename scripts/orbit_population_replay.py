"""Replay the systems of an old population run, without fitting anything.

The population run at ``aea931a`` (200 systems, 20 tasks of 10) drew each
task's systems from one ``default_rng(seed)`` stream, and consumed the PIT
uniform only for systems whose fit was ok, so the draws after a failed system
depend on the fit. This script re-simulates every task with that *old* stream
(the script as of ``aea931a``, loaded with ``git show``), consuming the PIT
uniform only for the rows that are ok in the run's CSV, and so recovers
the systems that run fitted. It counts the systems, and which failed ones,
with a night whose observed ``ln f`` lies outside ``[ln FLUX_FLOOR, 0]``: a
night there has an undefined outlier density ``b_k`` (the fit's
``log_outlier`` is ``-inf``), the defect #641 hid by truncating the simulated
noise. Each replayed period is checked against the CSV's.

numpy only (the old script's simulator; no fit is run).

usage: orbit_population_replay.py COMBINED_CSV [--commit aea931a]
"""

import argparse
import csv
import math
import subprocess
import types

import numpy as onp


def load_old(commit):
    """The population script of ``commit`` as a module."""
    src = subprocess.run(
        ["git", "show", f"{commit}:scripts/orbit_population.py"],
        check=True, capture_output=True, text=True,
    ).stdout  # fmt: skip
    mod = types.ModuleType("orbit_population_old")
    exec(
        compile(src, f"{commit}:scripts/orbit_population.py", "exec"),
        mod.__dict__,
    )
    return mod


def replay_task(old, seed, statuses):
    """The systems of ``--seed seed --n-sys len(statuses)`` as run at the old
    commit: ``(index, simulated system or None, status)`` per index."""
    rng = onp.random.default_rng(seed)
    out = []
    for i, status in enumerate(statuses):
        nights = old.draw_nights(rng)
        default, fit = old.fit_domain(nights)
        p_try = old._loguniform(rng, *old.P_TRUE)
        if fit is None or not default.p_cut <= p_try <= default.p_max:
            out.append((i, None, status))
            continue
        sysd = old.draw_system(rng, nights, fit)
        if status == "ok":
            rng.uniform()  # the PIT draw, consumed only by an ok system
        out.append((i, sysd, status))
    return out


def outside_support(old, sysd):
    """Nights whose observed ln f lies outside ``[ln FLUX_FLOOR, 0]``."""
    lo = math.log(old.oc_cells.FLUX_FLOOR)
    ln_f = onp.array([s.peaks[0].mean[4] for s in sysd["summaries"]])
    return int(onp.sum((ln_f < lo) | (ln_f > 0.0)))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("csv")
    ap.add_argument("--commit", default="aea931a")
    args = ap.parse_args()
    old = load_old(args.commit)
    with open(args.csv) as fh:
        rows = list(csv.DictReader(fh))
    by_task = {}
    for r in rows:
        by_task.setdefault(int(float(r["seed"])), {})[
            int(float(r["sys_in_task"]))
        ] = r
    table, bad_period, flagged = {}, 0, []
    for seed, task in sorted(by_task.items()):
        statuses = [task[i]["status"] for i in range(len(task))]
        for i, sysd, status in replay_task(old, seed, statuses):
            if sysd is None:
                continue
            period = float(task[i]["period"])
            if not math.isclose(sysd["period"], period, rel_tol=1e-5):
                bad_period += 1
            n_bad = outside_support(old, sysd)
            cell = table.setdefault(status, [0, 0])
            cell[0] += 1
            cell[1] += n_bad > 0
            if n_bad:
                flagged.append((status, seed, i, n_bad))
    print(f"replayed periods that differ from the CSV: {bad_period}")
    print(
        "| status | systems simulated | with a night outside [ln floor, 0] |"
    )
    print("|---|---|---|")
    for status in ("ok", "failed"):
        n, k = table.get(status, [0, 0])
        print(f"| {status} | {n} | {k} |")
    n = sum(v[0] for v in table.values())
    k = sum(v[1] for v in table.values())
    print(f"| all | {n} | {k} |")
    print("outside support (status, seed, system, nights):", flagged)


if __name__ == "__main__":
    main()
