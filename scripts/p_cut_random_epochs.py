"""P_cut = max(2 dt_min, P_night) on random epoch patterns (design/orbit_alias_cycles.md §1.3).

Schedules as the gate's stratum B (S1: K nights uniform in 90 d; S2: two 60-day
seasons 365 d apart; S3: half the nights within 10 d, the rest over 2 years),
K in {4, 6, 8, 12}. Night length, resolution and field are drawn over the stated
illustrative ranges, not tuned to any target. Prints how often each term sets
P_cut and the quantiles of P_cut / span.

    .venv/bin/python scripts/p_cut_random_epochs.py
"""

import numpy as onp

rng = onp.random.default_rng(0)
N = 20_000


def nights(kind, k):
    if kind == "S1":
        return rng.uniform(0, 90, k)
    if kind == "S2":
        a = k // 2
        return onp.r_[rng.uniform(0, 60, a), 365 + rng.uniform(0, 60, k - a)]
    a = k // 2
    return onp.r_[rng.uniform(0, 10, a), rng.uniform(0, 730, k - a)]


rows = []
for _ in range(N):
    kind = ("S1", "S2", "S3")[rng.integers(3)]
    k = int(rng.choice([4, 6, 8, 12]))
    t = onp.sort(nights(kind, k))
    night = rng.uniform(0.05, 0.3)  # d, the longest night
    res = rng.uniform(2.0, 5.0)  # mas, lambda / B_max
    field = rng.choice([10.0, 30.0])  # mas
    p_night = max(2 * onp.pi * field * night / (0.5 * res), night, 0.2)
    dt2 = 2 * onp.min(onp.diff(t))
    rows.append((kind, p_night, dt2, max(p_night, dt2), onp.ptp(t)))

kind = onp.array([r[0] for r in rows])
p_night, dt2, cut, span = (
    onp.array([r[i] for r in rows], float) for i in (1, 2, 3, 4)
)
print(
    f"{'schedule':<9}{'P_night sets P_cut':>20}{'median P_cut (d)':>18}{'P_cut/span q10, q50, q90':>28}"
)
for s in ("S1", "S2", "S3", "all"):
    m = onp.ones(N, bool) if s == "all" else kind == s
    q = onp.quantile(cut[m] / span[m], [0.1, 0.5, 0.9])
    print(
        f"{s:<9}{onp.mean(p_night[m] >= dt2[m]):>20.2f}{onp.median(cut[m]):>18.1f}{q[0]:>10.3f}{q[1]:>9.3f}{q[2]:>9.3f}"
    )
