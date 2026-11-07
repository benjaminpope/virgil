import sys; sys.path.insert(0,'tests')
import jax, numpy as onp, warnings
warnings.simplefilter("ignore")
jax.config.update("jax_enable_x64", True)
from test_kkt_report import *
from test_kkt_report import _problem,_polish,_disc
start, priors, data = _problem(10, 24.0)
r = fit(start, priors, data, TSV_, method="lbfgsb", gtol=1e-6)
r = _polish(r, start, priors, data, TSV_, "lbfgsb")
p = onp.asarray(r.values["env.pixels"]).copy()
print(r.info['steps'], r.info['stop'], (p>0).sum(), ((p>0)&(_disc(10)>0)).sum(), (_disc(10)>0).sum(), p.sum())
p[_disc(10)>0]=0
print("nonzero after", (p>0).sum(), p.sum())
import jax.numpy as jnp
from virgil.fitting import kkt_report
res = FitResult(start, {"env.pixels": p}, {})
print(kkt_report(res, priors, data, TSV_))
print("pixels", onp.round(onp.asarray(r.values["env.pixels"]),4).tolist())
