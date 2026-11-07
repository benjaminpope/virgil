import sys; sys.path.insert(0,'tests')
import jax, numpy as onp, warnings
warnings.simplefilter("ignore")
jax.config.update("jax_enable_x64", True)
import virgil.fitting as F
from test_kkt_report import *
from test_kkt_report import _problem
start, priors, data = _problem(10, 24.0)
r = fit(start, priors, data, TSV_, method="lbfgsb", max_steps=5000)
print("fit1", {k: r.info[k] for k in ('steps','stop','loss','message')}, "nevals", len(F.TRACE))
prev = None
for i, (v, g, x) in enumerate(F.TRACE[:12]):
    msg = ""
    if prev is not None:
        d = x - prev[2]
        msg = "dx=%.3e  g.d_prev=%.4e  dv=%.4e  pred(dv)~%.4e" % (onp.linalg.norm(d), prev[1] @ d, v - prev[0], prev[1] @ d)
    print("T", i, "v=%.10g" % v, "gmax=%.4g" % onp.abs(g).max(), "sumx=%.6g" % x.sum(), "nzero=%d" % (x == 0).sum(), "nan=%s" % (not onp.isfinite(g).all()), msg)
    prev = (v, g, x)
