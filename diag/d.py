import sys; sys.path.insert(0,'tests')
import jax, numpy as onp, warnings
warnings.simplefilter("ignore")
jax.config.update("jax_enable_x64", True)
import scipy; print("scipy", scipy.__version__, "jax", jax.__version__, "np", onp.__version__)
import virgil.fitting as F
from test_kkt_report import *
from test_kkt_report import _problem
orig = F._value_and_gradient
log = []
def wrapped(problem, template, scale, x):
    v, g = orig(problem, template, scale, x)
    xx = onp.asarray(x); gg = onp.asarray(g)
    log.append((float(v), float(onp.abs(gg).max()), float(xx.sum()), int((xx==0).sum()), bool(onp.isfinite(gg).all())))
    return v, g
F._value_and_gradient = wrapped
start, priors, data = _problem(10, 24.0)
r = fit(start, priors, data, TSV_, method="lbfgsb", max_steps=5000)
print("fit1", {k: r.info[k] for k in ('steps','stop','loss','grad_norm','message','chi2')})
print("nevals", len(log))
for i, l in enumerate(log[:60]): print(i, l)
# finite-difference check of directional derivative at the start
import jax.numpy as jnp
prob = F._Objective(start, priors, data, TSV_, None, (), boxed=True)
z0 = prob.init(start.get_values() if hasattr(start,'get_values') else {"env.pixels": start.get("env").pixels})
x0, un = prob.ravel(z0)
f = lambda x: float(prob.loss(un(jnp.asarray(x))))
g = onp.asarray(jax.grad(lambda x: prob.loss(un(x)))(jnp.asarray(x0)))
d = -g/onp.linalg.norm(g)
for h in (1e-9,1e-7,1e-5):
    print("FD h", h, (f(onp.asarray(x0)+h*d)-f(onp.asarray(x0)-h*d))/(2*h), "analytic", g@d)
