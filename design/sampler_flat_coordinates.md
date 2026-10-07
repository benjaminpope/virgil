# Sampling in the priors' flat coordinates

Status: implemented on `sampler-flat-coords` (0.4.0 backlog, "sampler
health"). Follows the 2026-10-05 decision in `imaging_plan.md` (standing
decision 9) that `fit` optimises each prior in its flat coordinate.

## Problem

A Jeffreys prior on a scale (`LogUniform(a, b)`) or an orientation
(`IsotropicInclination`, `IsotropicLatitude`) is uniform in some coordinate
u of its parameter x: u = log x, cos i, sin(lat). `fit` has optimised in u
since #226. `numpyro_model` still let numpyro map each site to NUTS's
unconstrained z with `biject_to(prior.support)`, the logit of the
*linear* position of x in its interval, so `fit`, `gauss_newton_mass` and
NUTS worked in different coordinates, and the sampler's prior in z was not
the clean logistic density that `fit`'s coordinate gives.

## Audit of the sampling entry points (before this change)

| Entry point | Where | Coordinates before | After |
| --- | --- | --- | --- |
| `numpyro_model` parameter sites | `src/virgil/likelihood.py:708` (`_sample`), called at `:928` | numpyro's `biject_to(support)` (logit of x on [a, b]) | flat (`_flat.flat_sampled`) |
| `numpyro_model` `noise=` sites (error scales, gain widths, `wavel_scale`, ...) | `src/virgil/likelihood.py:938` | `biject_to(support)` | flat |
| `numpyro_model` angle vectors (`AngleVector`) | `src/virgil/likelihood.py:717` | a 2-D vector (no wrap boundary) | unchanged: already its own coordinate |
| `numpyro_model` tied noise terms, `hierarchical_scales` members | `src/virgil/likelihood.py:937`, `src/virgil/priors.py:349` | deterministic; hyperparameters are ordinary `priors` sites | hyperparameters (LogUniform median, spread) now flat |
| `chain_init_params` | `src/virgil/likelihood.py:966` | whatever the sites' `biject_to` is | follows the sites: flat |
| `gauss_newton_mass` | `src/virgil/fitting.py:824` | `_Objective(flat=False)`: numpyro's coordinates | `flat=flat_coordinates` (default flat), matching NUTS |
| `fit`, `bias_test`, `OrbitStart` (MAP, not sampling) | `src/virgil/fitting.py:432`, `src/virgil/simulate.py:72`, `src/virgil/epochs.py:1662` | flat (since #226) | unchanged |
| `laplace_samples` | `src/virgil/imaging.py:1936` | Gaussian in a field's latents (Normal priors) | unchanged: no flat-coordinate prior |
| grid marginalisation (`linear_flux_grid(prior=LogUniform)`, detection log Bayes factor) | `src/virgil/grid_fit.py:556`, `src/virgil/detection.py:358` | closed-form/quadrature in log f | unchanged: not a sampler |

## Reparametrization

`_flat.py` holds the one implementation of the flat coordinate, used by
both `fit` and the samplers (it moved out of `fitting.py`):

- `_flat_coordinate(prior)` returns `(to_flat, from_flat, low, high)`:
  LogUniform gives `(log, from_flat, log a, log b)`, where `from_flat` is the straight-through exp clipped into [a, b]; a prior with a
  `flat_coordinate()` method (the isotropic priors) gives its own, the CDF
  and quantile function on [0, 1].
- `_FlatBijection` is z → u = low + (high − low) σ(z) → x = from_flat(u),
  with `log_abs_det_jacobian(z)` = log|du/dz| + log|dx/du|. The second term
  is one forward-mode pass of `from_flat` with a tangent of ones
  (`from_flat` is elementwise), so no prior needs a hand-written Jacobian.
- `flat_sampled(prior)` wraps a prior that has a flat coordinate in
  `_FlatPrior`, a distribution with the prior's density, samples, shapes
  and moments, whose support is a tagged constraint `_FlatSupport`.
  `biject_to` is registered for `_FlatSupport` and returns `_FlatTransform`,
  numpyro's form of `_FlatBijection`. Expanded and `to_event` priors keep
  their event dimensions (`constraints.independent`).

Every numpyro tool that asks a site for its unconstrained coordinates
(NUTS's potential, `initialize_model`, `init_to_value`, `chain_init_params`,
AutoGuides) gets the flat bijection through `biject_to`, with no change to
numpyro.

## Jacobian

numpyro's potential is −[log p(x) + log|dx/dz|]. With x uniform in u on
[low, high], log p(x) + log|dx/du| = −log(high − low), so the potential is
−log σ(z) − log σ(−z) + const: exactly the logistic density, for every
flat-coordinate prior. The tests check this identity to |z| = 24 in
float32 and float64, and the Jacobian against autodiff of the transform.

Two float32 details made it exact at the ends:

- `exp(log a)` can round below a; LogUniform's `from_flat` clips into the
  support with a straight-through gradient (the value is clipped, the
  derivative is exp's).
- The isotropic quantile functions used `jnp.where` between two branches
  whose untaken one had an infinite slope near a pole, giving NaN
  gradients for |z| ≳ 18 in float32, and a final clip whose slope at a
  bound is 1/2. Both now keep the true derivative (`_clip_through` and a
  safe argument for the untaken branch).

## What users see

- Sites keep their names (`"comp.flux"`, `"noise.vis_scale"`), and
  `mcmc.get_samples()` returns the model's own parameters, never u or z.
- `log_density`, `Predictive` and `posterior_predictive_summary` are
  unchanged, because the site's density is the prior's.
- `FitResult.values` starts NUTS with `init_to_value` as before.
- `numpyro_model(..., flat_coordinates=False)` and
  `gauss_newton_mass(..., flat_coordinates=False)` restore numpyro's
  bijection of the support. Pass the same value to both.

## Backwards compatibility

The posterior is the same; only the coordinate NUTS moves in differs. What
changes for a user:

- unconstrained `init_params` saved from 0.3 (or computed with
  `initialize_model` there) are in the old coordinates: recompute them
  (`chain_init_params`) or pass `flat_coordinates=False`;
- a `gauss_newton_mass` matrix from 0.3 is in the old coordinates for
  LogUniform and isotropic sites; recompute it;
- chains differ draw by draw (different z), so seeded results change, but
  not in distribution.

## Testing

`tests/test_sampler_flat_coordinates.py` (tiny, local):

- round trips z → x → z of the numpyro transform, and that it equals
  `fit`'s `_FlatBijection` (float32 and x64);
- `log_abs_det_jacobian` against autodiff;
- the potential of every flat prior is the logistic density, with finite
  gradients, to |z| = 24 (float32 and x64);
- sites, values and `log_density` are the same with and without flat
  coordinates; `init_to_value`/`chain_init_params` give `fit`'s z;
  expanded priors; other priors are not wrapped;
- no-data NUTS reproduces each flat prior (KS on ESS-thinned draws, float32
  and x64), parameters and an error term.

Existing tests updated: the no-data potential test now takes each site's
own bijection, and `gauss_newton_mass` is checked in both coordinates.

With data, the check is simulation-based calibration: virgil-validation's
`sbc_numpyro` job (binary diameter, flux ratio and V² error scale from
LogUniform priors, NUTS on `numpyro_model`) run on this branch's commit,
compared with its 0.3 run.

## Evidence so far (laptop, float32, single chains of 500 + 2000)

The change makes the coordinates consistent and the prior's potential
exactly logistic; it did not measurably change sampling efficiency on the
toy problems tried, because numpyro's logit of x is already close to log x
away from the bounds:

| Problem | numpyro's bijection | flat coordinates |
| --- | --- | --- |
| No data: LogUniform(1e-6, 1), LogUniform(0.1, 10), IsotropicInclination; min ESS per 1000 gradients, 4 seeds | 118–147 | 82–125 |
| Ridge log(a³/P²) known to 10%, a, P log-uniform; min ESS per 1000 gradients, 3 seeds | 3.1–5.8, no divergences | 3.7–4.4, no divergences |
| Centred hierarchical funnel, 8 weakly measured log-scales, `hierarchical_scales` priors; divergences in 2000, 11 seeds | 0–578, total 2312 | 0–825, total 2672 |
| Binary, faint companion (flux 3e-3) at 7-hole SNR; divergences in 2000 | 142 | 215 |

KS checks of the prior marginals passed in both. The funnel and the faint
companion are hard in both coordinates: they are correlations between
parameters (a scale against its members, a flux against the position it
makes measurable), which no per-parameter reparametrization removes; the
non-centred form and the Gauss–Newton mass matrix are the tools for them.
The gains are consistency (`gauss_newton_mass`, `fit` and NUTS share
coordinates, so the mass matrix computed at the MAP is the one NUTS
needs) and a well-defined prior potential at the ends of the range,
including the poles of the isotropic priors.

## Variational inference (numpyro SVI): assessment for virgil#12

**What already works.** `numpyro_model` is an ordinary numpyro model, so
SVI runs on it today with no virgil code. A tiny test (binary on a 7-hole
mask, float32, `optax.adam(1e-2)`, 2000 steps, guides started with
`init_loc_fn=init_to_value(values=fit(...).values)`) ran in under a second
per guide after compilation:

| Guide | σ(dra) | σ(ddec) | σ(flux) |
| --- | --- | --- | --- |
| `AutoNormal` | 1.03 | 0.94 | 0.0010 |
| `AutoMultivariateNormal` | 1.49 | 1.24 | 0.0017 |
| `AutoLaplaceApproximation` | 1.71 | 1.25 | 0.0021 |

The posterior means agree with the fit; the mean-field guide underestimates
the widths of correlated parameters, as expected. `guide.sample_posterior`
returns the model's own parameters under the same site names.

**What is missing.**

- *Guides in the flat coordinates*: none needed. AutoGuides build their
  Gaussian in `biject_to(site.support)`, so with this change they are
  Gaussian in the flat coordinate z = logit((log x − log a)/(log b −
  log a)), not in log x. Away from the bounds z is close to an affine
  function of log x, so a LogUniform scale's posterior is close to
  log-normal in x there; near a bound the logit stretches it. This was
  the main obstacle.
- *Start from a fit*: `init_to_value(values=FitResult.values)` works now.
  A helper could also set the initial scale of an `AutoMultivariateNormal`
  from `gauss_newton_mass`'s covariance, which is in the same coordinates
  (`init_scale` takes one number; a dense start needs a custom guide or
  setting the `scale_tril` parameter).
- *Angle vectors*: an `AngleVector` site is a 2-D vector on a ring; a
  Gaussian guide on it works but is a poor approximation for wide angles.
- *Outputs*: a thin `virgil` wrapper (say `variational(model, priors, data,
  guide="mvn", start=result)`) returning draws keyed like NUTS's, plus
  the ELBO, would make it one call. Optional.
- *Tests*: no-data SVI with `AutoMultivariateNormal` reproduces nothing
  exactly (a logistic is not Gaussian), so test (i) that a posterior
  that is Gaussian in z is recovered by `AutoNormal` (a likelihood
  Gaussian in z, so that the logistic prior is not in the way, or a
  narrow posterior far from the bounds, where the prior is locally flat
  in z up to a small tilt, so the match is approximate, not exact), (ii) that
  `AutoLaplaceApproximation` matches `gauss_newton_mass` at the MAP, and
  (iii) agreement of means and widths with a short NUTS run on a binary,
  in float32 and x64.

**Effort.** S for documentation and a tutorial cell (it works now); M for
a wrapper with fit-started guides, a dense initial scale from
`gauss_newton_mass` and the tests above.

**Risks.** Multimodality: Gaussian guides find one mode, so orbits
(Ω/ω degeneracies, period aliases) and images need several starts
(`OrbitStart.chain_values` gives them) or importance weighting of the
guide's draws with PSIS k̂ as the check, as the priors rule already asks
for reweighting. High-dimensional images: an `AutoMultivariateNormal` over
thousands of pixels is a dense Cholesky per step; `AutoLowRankMultivariateNormal`
or the Laplace approximation (`laplace_samples`) fit better. float32: the
ELBO's Monte Carlo gradient is noisy and the Cholesky of a badly
conditioned scale can fail; the flat coordinates and the logistic prior
potential help, but SVI should run in x64 like `fit` by default.

**Does the flat-coordinate work help?** Yes: it is what makes the
standard AutoGuides sensible for virgil's Jeffreys priors (Gaussian in the logit of the
flat coordinate, log x or cos i, rather than in the logit of x itself), and it puts `fit`, the
Gauss–Newton covariance, NUTS and any guide in one coordinate system, so a
guide can be started from the MAP and its curvature directly.
