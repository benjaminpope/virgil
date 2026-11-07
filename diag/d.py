import sys; sys.path.insert(0,'tests')
import jax, numpy as onp, warnings
warnings.simplefilter("ignore")
jax.config.update("jax_enable_x64", True)
import scipy; print("scipy", scipy.__version__, "jax", jax.__version__, "np", onp.__version__)
from test_kkt_report import *
from test_kkt_report import _problem
start, priors, data = _problem(10, 24.0)
r = fit(start, priors, data, TSV_, method="lbfgsb", max_steps=5000)
print("fit1", {k: r.info[k] for k in ('steps','stop','loss','grad_norm','message')})
r = fit(start, priors, data, TSV_, method="lbfgsb", gtol=1e-4, init=r.values)
print("polish", {k: r.info[k] for k in ('steps','stop','loss','grad_norm','message')})
rep = kkt_report(r, priors, data, TSV_)
print(rep)
x = onp.asarray(r.values["env.pixels"]).ravel()
from virgil.fitting import _Objective, cast_tree, run_in
import jax.numpy as jnp
prob = _Objective(r.model, priors, data, TSV_, None, (), boxed=True)
z = prob.init(r.values)
g = onp.asarray(jax.grad(prob.loss)(z)["env.pixels"]).ravel()
onp.set_printoptions(linewidth=200, precision=4)
print("sum x", x.sum(), "max x", x.max())
print("x", x); print("g", g)
