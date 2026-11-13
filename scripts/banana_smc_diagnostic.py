"""Per-key diagnostics of the replicated SMC on the banana toy.

Usage: ``python scripts/banana_smc_diagnostic.py [--s1 0.2] [--n 200]
[--out banana_smc_s1_0.2.json]``; ``VIRGIL_TEST_SEED`` replays a seed. Runs the
same keys (``seed + i``) as the 200-key banana test and writes one row per key:
ln Ẑ, truth, error, which variance term set the error, the replicate spread,
the steps used, why the run is flagged, stages and minimum ESS. Data only: it
asserts nothing and changes no tolerance.
"""

import argparse
import json
import math
import os
import secrets
import sys

import jax
import numpy as onp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))
import test_orbit_cells as t  # noqa: E402

from virgil import orbit_cells as oc  # noqa: E402

TERMS = ("V_LW", "V_floor", "V_rep")


def row(i, seed, res, truth):
    terms = [
        res.log_z_variance,
        res.log_z_floor_variance,
        res.log_z_rep_variance,
    ]
    ok = [v if v == v and v > 0 else -math.inf for v in terms]
    err = oc._smc_error(res)
    dev = float(res.log_z - truth)
    return dict(
        i=i, seed=seed, truth=truth, log_z=float(res.log_z), err=err, dev=dev,
        dev_over_err=dev / err if err == err and err > 0 else math.nan,
        err_term=TERMS[int(onp.argmax(ok))],
        terms=dict(zip(TERMS, map(float, terms))),
        replicate_sd=float(onp.std(res.log_z_replicates, ddof=1)),
        replicates=[float(z) for z in res.log_z_replicates],
        steps=[[int(s) for s in r] for r in res.steps],
        mixing_short=bool(res.mixing_short),
        stopped=bool(res.stopped),
        degenerate=bool(res.variance_degenerate),
        err_high=not err <= oc.SMC_ERROR,
        unreliable=oc._smc_unreliable(res),
        stages=[len(b) for b in res.betas],
        min_ess=float(min(onp.min(e) for e in res.ess)),
        final_ess=float(min(e[-1] for e in res.ess)),
    )  # fmt: skip


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--s1", type=float, default=0.2)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    seed = int(os.environ.get("VIRGIL_TEST_SEED", secrets.randbits(31)))
    print("seed", seed, flush=True)
    rows = []
    with jax.enable_x64(True):
        for i in range(args.n):
            res, truth = t._banana_run(seed + i, args.s1)
            rows.append(row(i, seed + i, res, truth))
    out = args.out or f"banana_smc_s1_{args.s1}.json"
    with open(out, "w") as fh:
        json.dump(dict(seed=seed, s1=args.s1, rows=rows), fh, indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main()
