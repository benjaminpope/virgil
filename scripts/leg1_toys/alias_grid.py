"""Across the integer alias grid N (design/orbit_alias_cycles.md §6.7): per-N runs against one
shared run over all cells, both against the exact per-cell reference.

  per-N  : for each cell N, its own search (minimal sets) and defensive IS (search_evidence.py).
  shared : one search over the whole period domain (each minimal set's grid spans every cell at
           that set's own coherence), one defensive mixture q over all modes, one set of IS draws;
           Z_N is the IS mean of p L 1{x in cell N} / q. Unbiased for every N at once.

Prints ln Z_N (exact, per-N, shared), the posterior p(N), and the costs.

usage: alias_grid.py PATTERN REL [CELLS_EACH_SIDE] [REPS]
"""

import math
import sys
import time

import numpy as onp

import search_evidence as S
from toy import Problem


def cell_of(prob, x):
    return onp.rint(prob.T / onp.exp(prob.u_of_w(x[:, 0]))).astype(int)


if __name__ == "__main__":
    pattern, rel = sys.argv[1], float(sys.argv[2])
    side = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    reps = int(sys.argv[4]) if len(sys.argv) > 4 else 5
    whole = Problem(pattern, rel, cells=side)
    ns = list(range(whole.n_lo, whole.n_hi + 1))
    # the whole domain at the default 4e5-point cap is coarser than the outer cells' peaks
    # (N = 22 of K10_2s was 0.57 nats low); 1.6e6 points agree with 4e5 to 0.001 elsewhere
    ex, _ = whole.exact(n0=1_600_000)
    w_tot = whole.LHI - whole.LLO

    # per-N
    per, per_cost, per_modes = {n: [] for n in ns}, 0.0, 0
    for n in ns:
        prob = Problem(pattern, rel, n_cell=n)
        modes, c, _ = S.search(prob)
        per_cost += c
        per_modes += len(modes)
        q = S.Mixture(prob, modes) if modes else None
        for r in range(reps):
            rng = onp.random.default_rng(1000 + r)
            lz, c, _ = S.importance(prob, q, S.N_IS, rng)
            per[n].append(lz + math.log((prob.LHI - prob.LLO) / w_tot))
            per_cost += c / reps

    # shared
    t0 = time.time()
    modes, sh_cost, _ = S.search(whole)
    q = S.Mixture(whole, modes)
    n_sh = S.N_IS * len(ns)  # the same total number of draws as the per-N runs
    sh = {n: [] for n in ns}
    for r in range(reps):
        rng = onp.random.default_rng(1000 + r)
        x = q.draw(n_sh, rng)
        lw = whole.logpost(x) - q.log_q(x)
        cell = cell_of(whole, x)
        for n in ns:
            lwn = onp.where(cell == n, lw, -onp.inf)
            sh[n].append(
                S.logmeanexp(lwn) if onp.isfinite(lwn).any() else -onp.inf
            )
    sh_cost += n_sh
    print(
        f"{pattern} sigma/a={rel:g} cells {ns[0]}..{ns[-1]} | per-N: modes={per_modes} "
        f"cost={per_cost:.2e} | shared: modes={len(modes)} cost={sh_cost:.2e} [{time.time() - t0:.0f}s]"
    )
    lz_ex = onp.array([ex[n] for n in ns])
    p_ex = onp.exp(lz_ex - onp.logaddexp.reduce(lz_ex))
    for name, res in (("per-N", per), ("shared", sh)):
        z = onp.array([res[n] for n in ns])  # (cells, reps)
        pz = onp.exp(z - onp.logaddexp.reduce(z, 0))
        print(f"  {name}:")
        for i, n in enumerate(ns):
            zi = z[i][onp.isfinite(z[i])]
            sd = zi.std(ddof=1) if len(zi) > 1 else onp.nan
            print(
                f"    N={n}: exact lnZ={lz_ex[i]:9.3f} p={p_ex[i]:.3e} | lnZ={zi.mean() if len(zi) else -onp.inf:9.3f} "
                f"bias={(zi.mean() - lz_ex[i]) if len(zi) else onp.nan:8.3f} sd={sd:6.3f} "
                f"p={pz[i].mean():.3e} sd(p)={pz[i].std(ddof=1):.1e} finite={len(zi)}/{reps}"
            )
        print(
            f"    total-variation distance of p(N) from exact: {0.5 * onp.abs(pz.mean(1) - p_ex).sum():.2e}"
        )
