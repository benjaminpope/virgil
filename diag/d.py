import sys; sys.path.insert(0,'tests')
import jax, numpy as onp, warnings
warnings.simplefilter("ignore")
jax.config.update("jax_enable_x64", True)
from test_kkt_report import *
from test_kkt_report import _problem, _polish
start, priors, data = _problem(10, 24.0)
for i in range(1):
    r = fit(start, priors, data, TSV_, method="lbfgsb", max_steps=5000)
    print("fit1", {k: r.info[k] for k in ('steps','stop','loss','message')})
r = _polish(r, start, priors, data, TSV_, "lbfgsb", 1)
print(kkt_report(r, priors, data, TSV_))
