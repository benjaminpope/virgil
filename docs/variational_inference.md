# Variational inference

!!! warning "Provisional"
    May change in any 0.4.x release without a deprecation cycle.

NUTS (see [Binary search](binary_search.md)) draws from the posterior itself, but each draw costs many likelihood gradients. Variational inference (VI) instead fits a distribution of a chosen form, the *guide*, to the posterior by maximizing the evidence lower bound (ELBO), equivalently minimizing the KL divergence from guide to posterior. Once fitted, the guide gives draws almost for free. [`variational`][virgil.svi.variational] runs numpyro's stochastic VI (SVI) on the posterior of [`numpyro_model`][virgil.likelihood.numpyro_model], with the same arguments as [`fit`][virgil.fitting.fit].

## Which guide

For a well-behaved posterior, a Gaussian guide (`guide="mvn"`) is nearly the **Laplace approximation**, the Gaussian at the fit with the covariance from the curvature there. SVI fits the whole posterior where Laplace uses only the curvature at the mode, but for a nearly Gaussian posterior they agree, and virgil gives the Laplace approximation directly through [`gauss_newton_mass`][virgil.fitting.gauss_newton_mass], [`laplace_cov`][virgil.inference.laplace_cov] and [`laplace_samples`][virgil.imaging.laplace_samples]. **If a Gaussian is all you need, use those.**

Run VI when the posterior is *not* Gaussian in the sampler's coordinates: a curved degeneracy (a banana between two parameters), a skewed scale, or a posterior that runs into a prior bound, such as a faint companion's flux with a tail towards zero. For these the default guide is a **normalizing flow**, which passes the Laplace Gaussian through a learnt, invertible neural map that can bend and skew it. Two flows are available:

| `guide` | Form | Use it for |
| --- | --- | --- |
| `"bnaf"` (default) | block neural autoregressive flow | curved, skewed or truncated posteriors in a few to tens of parameters |
| `"iaf"` | inverse autoregressive flow | the same, slightly cheaper and less flexible |
| `"mvn"` | dense Gaussian | ≈ Laplace; a check, or when the flow is too costly |
| `"laplace"` | numpyro's `AutoLaplaceApproximation` | the MAP and Hessian, found by numpyro |

On small test posteriors (a banana, a curved ridge, and a scale with a tail down to its prior bound), the BNAF guide came closest to NUTS's means, widths and 5–95% quantiles. On the banana the Gaussian guides missed the curvature (half the width in one parameter, a third of the shift in the other's mean), and on the bounded scale they missed the tail. No guide reproduced the heavy tail along the ridge, where NUTS's width was about 1.5 times the guides'. A flow's cost grows with the square of the number of parameters, so for an image of thousands of pixels use [`laplace_samples`][virgil.imaging.laplace_samples] (see [Sampling the posterior](imaging_sampling.md)) or a diagonal guide.

Every guide starts at the Laplace approximation, centred on a [`FitResult`][virgil.fitting.FitResult] passed as `start`, with the Gauss–Newton covariance there as its width. That covariance includes fitted error terms in `noise` (each with its own width), data with gains and any elementwise prior, but not regularizers or fixed error terms in `**options`; with regularizers (as for an image) the guide starts at width `init_scale` instead. The dense guides (`"bnaf"`, `"iaf"`, `"mvn"`) keep an n × n Cholesky factor and `"laplace"` an n × n Hessian, so all four raise a `ValueError` above 4096 coordinates; use [`laplace_samples`][virgil.imaging.laplace_samples] or a diagonal guide such as `autoguide.AutoNormal` there. The guide learns its parameters in units of those widths, so one step size suits a position known to a milliarcsecond and a flux known to a few per cent. With `flat_coordinates=True` (off by default, as for `numpyro_model`; see [Conventions](conventions.md)) the guide works in the priors' flat coordinates, as `fit` does, and returns draws in the model's own parameters under the site names NUTS uses.

## A binary

```python
import jax
import numpy as np
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, init_to_value

from virgil.coverage import nrm_oidata
from virgil.fitting import fit
from virgil.likelihood import numpyro_model
from virgil.models import BinaryModelCartesian
from virgil.svi import variational

truth = BinaryModelCartesian(120.0, -80.0, 0.02)
data = nrm_oidata().with_model(truth, key=jax.random.PRNGKey(0))

priors = {
    "dra": dist.Uniform(-400.0, 400.0),
    "ddec": dist.Uniform(-400.0, 400.0),
    "flux": dist.LogUniform(1e-4, 1.0),
}
template = BinaryModelCartesian(110.0, -70.0, 0.01)
result = fit(template, priors, data)

vi = variational(template, priors, data, start=result)
print(vi.converged, vi.info["elbo"], vi.info["khat"])

mcmc = MCMC(
    NUTS(
        numpyro_model(template, priors, data),
        init_strategy=init_to_value(values=result.values),
    ),
    num_warmup=500,
    num_samples=2000,
)
mcmc.run(jax.random.PRNGKey(1))
nuts = mcmc.get_samples()

for name in priors:
    print(
        f"{name:5s} VI {np.mean(vi.samples[name]):9.4g} ± {np.std(vi.samples[name]):.3g}"
        f"   NUTS {np.mean(nuts[name]):9.4g} ± {np.std(nuts[name]):.3g}"
    )
```

`vi.samples` is keyed like `mcmc.get_samples()`, so the same corner plots and summaries work on both. This binary's posterior is close to Gaussian: the guides and NUTS agree to a few tenths of a standard deviation in the means and 10% in the widths, and `"mvn"` matches the flow.

## Checking the result

- **Convergence.** `vi.converged` is `True` when the upper 2σ bound on the loss's fall between the last two windows (a tenth of the steps each) is under one nat; `vi.losses` holds its history. A noisy loss cannot pass, so raise `steps`, `num_particles` or `window` if it warns on a plateau. A plateau is necessary, not sufficient.
- **PSIS k̂.** `vi.info["khat"]` is the Pareto k̂ of the importance weights p/q of the guide's draws. Below about 0.7 the guide covers the posterior well enough to reweight its draws towards it; above that it misses posterior mass, and `variational` warns.
- **One mode.** Every guide here covers a single mode. A multimodal posterior (an orbit with Ω/ω flips or period aliases, an image with a mirrored solution) needs several starts, for example the distinct fits of [`OrbitStart.chain_values`][virgil.epochs.OrbitStart.chain_values]. Compare their ELBOs, or reweight their draws and check k̂.
- **Precision.** SVI runs in float64 by default, as `fit` does (`dtype="float32"` also works).
