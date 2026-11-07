import sys; sys.path.insert(0,'tests')
import jax, numpy as onp, warnings
warnings.simplefilter("ignore")
jax.config.update("jax_enable_x64", True)
import virgil.fitting as F
from test_kkt_report import *
from test_kkt_report import _problem,_polish,_disc
orig = F._value_and_gradient
for name, bad in (("inf", onp.inf), ("max", onp.finfo(float).max)):
    def w(problem, template, scale, x, bad=bad):
        v, g = orig(problem, template, scale, x)
        if not (onp.isfinite(float(v)) and onp.isfinite(onp.asarray(g)).all()):
            return onp.float64(bad), onp.zeros_like(onp.asarray(g))
        return v, g
    F._value_and_gradient = w
    start, priors, data = _problem(10, 24.0)
    r = fit(start, priors, data, TSV_, method="lbfgsb", max_steps=5000)
    print(name, "test1 fit", r.info['steps'], r.info['stop'], r.info['loss'], r.info['message'])
    r = _polish(r, start, priors, data, TSV_, "lbfgsb", 1)
    print(name, "test1 report", kkt_report(r, priors, data, TSV_))
    r = fit(start, priors, data, TSV_, method="lbfgsb", gtol=1e-6)
    print(name, "test3 fit", r.info['steps'], r.info['stop'], r.info['loss'], r.info['message'])
    r = _polish(r, start, priors, data, TSV_, "lbfgsb")
    p = onp.asarray(r.values["env.pixels"]).copy()
    print(name, "test3 nonzero", (p>0).sum(), "lit nonzero", ((p>0)&(_disc(10)>0)).sum())
    p[_disc(10)>0]=0
    print(name, "test3 report", kkt_report(FitResult(start, {"env.pixels": p}, {}), priors, data, TSV_))
