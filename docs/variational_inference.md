# Variational inference

NUTS (see [Binary search](binary_search.md)) draws from the posterior itself, but each draw costs many likelihood gradients. Variational inference (VI) instead fits a distribution of a chosen form, the *guide*, to the posterior, by maximising the evidence lower bound (ELBO), which is the same as minimising the KL divergence from the guide to the posterior. Once the guide is fitted, draws from it cost almost nothing. [`variational`][virgil.svi.variational] runs numpyro's stochastic VI (SVI) on the posterior of [`numpyro_model`][virgil.likelihood.numpyro_model], with the same arguments as [`fit`][virgil.fitting.fit].

## Which guide

A Gaussian guide (`guide="mvn"`) is, for a well-behaved posterior, nearly the **Laplace approximation**: the Gaussian centred on the fit, with the covariance given by the curvature there. The two are not identical. SVI fits the whole posterior, while Laplace uses only the curvature at the mode. For a posterior close to Gaussian, though, they agree, and virgil already gives the Laplace approximation directly through [`gauss_newton_mass`][virgil.fitting.gauss_newton_mass], [`laplace_cov`][virgil.inference.laplace_cov] and [`laplace_samples`][virgil.imaging.laplace_samples]. **If a Gaussian is all you need, use those.**

VI is worth running when the posterior is *not* Gaussian in the coordinates the sampler works in. That happens with a curved degeneracy (a banana between two parameters), a skewed scale, or a parameter whose posterior runs into a prior bound, such as a faint companion's flux with a tail towards zero. For these, the default guide is a **normalising flow**: the Laplace Gaussian is passed through a learnt, invertible neural map that can bend and skew it. Two flows are available:

| `guide` | Form | Use it for |
| --- | --- | --- |
| `"bnaf"` (default) | block neural autoregressive flow | curved, skewed or truncated posteriors in a few to tens of parameters |
| `"iaf"` | inverse autoregressive flow | the same, slightly cheaper and less flexible |
| `"mvn"` | dense Gaussian | ≈ Laplace; a check, or when the flow is too costly |
| `"laplace"` | numpyro's `AutoLaplaceApproximation` | the MAP and Hessian, found by numpyro |

On small test posteriors (a banana, a curved ridge, and a scale with a tail down to its prior bound), the BNAF guide came closest to NUTS's means, widths and 5–95% quantiles. On the banana, the Gaussian guides missed the curvature (half the width in one parameter, and a third of the shift in the other's mean); on the bounded scale they missed the tail towards the bound. No guide reproduced the heavy tail along the ridge, where NUTS's width was about 1.5 times the guides'. A flow's cost grows with the square of the number of parameters, so for an image of thousands of pixels use `"mvn"` with care, or better the Laplace approximation (see [Sampling the posterior](imaging_sampling.md)).

Every guide starts at the Laplace approximation. Its centre is at a [`FitResult`][virgil.fitting.FitResult] passed as `start`, and its width is the Gauss–Newton covariance there. The guide's parameters are then learnt in units of those widths, so one step size suits a position known to a milliarcsecond and a flux known to a few per cent. Like NUTS, the guide works in the priors' flat coordinates (`flat_coordinates=True`, see [Conventions](conventions.md)), and its draws are returned in the model's own parameters, under the site names NUTS uses.

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
start = BinaryModelCartesian(110.0, -70.0, 0.01)
result = fit(start, priors, data)

vi = variational(start, priors, data, start=result)
print(vi.converged, vi.info["elbo"], vi.info["khat"])

mcmc = MCMC(
    NUTS(
        numpyro_model(start, priors, data),
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

`vi.samples` is keyed like `mcmc.get_samples()`, so the same corner plots and summaries work on both. On this binary the posterior is close to Gaussian. The guides and NUTS agree to within a few tenths of a standard deviation in the means and 10% in the widths, and `"mvn"` gives the same answer as the flow.

## Checking the result

- **Convergence.** `vi.converged` is `True` when the loss (the negative ELBO) has stopped falling over the last tenth of the steps, and `vi.losses` holds its history. A plateau is necessary, not sufficient.
- **PSIS k̂.** `vi.info["khat"]` is the Pareto k̂ of the importance weights p/q of the guide's draws. Below about 0.7, the guide covers the posterior well enough for the draws to be reweighted towards it. Above that, it misses mass that the posterior has.
- **One mode.** Every guide here covers a single mode. A multimodal posterior, such as an orbit with Ω/ω flips or period aliases, or an image with a mirrored solution, needs several starts (for example the distinct fits of [`OrbitStart.chain_values`][virgil.epochs.OrbitStart.chain_values]). Compare their ELBOs, or reweight their draws and check k̂.
- **Precision.** SVI runs in float64 by default, as `fit` does (`dtype="float32"` also works).
