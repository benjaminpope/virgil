import sys
import numpy as onp

sys.argv = [sys.argv[0]] + sys.argv[1:]
import toy_needle as T

K, rel = int(sys.argv[1]), float(sys.argv[2])
prob = T.Prob(K, rel, 0)
# consistency: E_{N_w}[exp ll_comp] vs exp coll_all at one P, beta
rng = onp.random.default_rng(1)
P = onp.array([12.0])
b = 1e-3
th = rng.normal(size=(400000, 4)) * T.SW
lc = prob.ll_comp(th, onp.repeat(P, len(th)), b)
print(
    "MC check beta=1e-3:",
    onp.logaddexp.reduce(lc) - onp.log(len(lc)),
    prob.coll_all(P, b)[0],
)
lp = onp.linspace(T.LLO, T.LHI, 100000)
f1 = prob.coll_all(onp.exp(lp), 1.0)
i = onp.argmax(f1)
lp = onp.sort(
    onp.concatenate(
        [
            lp,
            onp.linspace(
                lp[max(i - 100, 0)], lp[min(i + 100, len(lp) - 1)], 40000
            ),
        ]
    )
)
out = []
for b in onp.concatenate(
    [[1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3], onp.logspace(-2.9, 0, 30)]
):
    f = prob.coll_all(onp.exp(lp), b)
    g = onp.exp(f - f.max())
    lz = f.max() + onp.log(onp.trapezoid(g, lp)) - onp.log(T.LHI - T.LLO)
    # spike share: mass within +-0.001 in lnP of the global max at beta=1
    c = onp.abs(lp - onp.log(T.P_TRUE)) < 2e-3
    share = onp.trapezoid(g * c, lp) / onp.trapezoid(g, lp)
    out.append((b, lz, share))
for b, lz, s in out:
    print(f"beta={b:.2e} lnZ={lz:9.3f} share(|lnP-lnPtrue|<2e-3)={s:.3f}")
