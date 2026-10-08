# Sampling in the priors' flat coordinates

Status: implemented on `sampler-flat-coords` (0.4.0 backlog, "sampler
health"). Follows the 2026-10-05 decision in `imaging_plan.md` (standing
decision 9) that `fit` optimizes each prior in its flat coordinate.

## Problem

A Jeffreys prior on a scale (`LogUniform(a, b)`) or an orientation
(`IsotropicInclination`, `IsotropicLatitude`) is uniform in some coordinate
u of its parameter x: u = log x, cos i, sin(lat). `fit` has optimized in u
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
| grid marginalization (`linear_flux_grid(prior=LogUniform)`, detection log Bayes factor) | `src/virgil/grid_fit.py:556`, `src/virgil/detection.py:358` | closed-form/quadrature in log f | unchanged: not a sampler |

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

## Variational inference (numpyro SVI): implemented (virgil#12)

**Status: implemented** as `virgil.svi.variational(model, priors, data,
regularizers=(), *, noise, likelihoods, start, guide="bnaf", steps=3000,
optimizer, learning_rate=3e-3, num_particles=8, num_samples=2000, key,
flat_coordinates=True, dense_start=True, init_scale=0.1, window, psis=True,
dtype="float64", **options)`, returning a `VariationalResult` (`samples`
keyed like NUTS's, `losses`, `guide`, `params`, `converged`, `info` with
the ELBO, PSIS k̂, time and whether the start was dense). Tests:
`tests/test_svi.py`.

**Guides in the flat coordinates.** AutoGuides build their Gaussian in
`biject_to(site.support)`, so with this change they are Gaussian in the
flat coordinate z = logit((log x − log a)/(log b − log a)) for a
LogUniform, not in log x. `fit`, the Gauss–Newton covariance, NUTS and
the guide share one coordinate system, so a guide can start from the MAP
and its curvature directly. `flat_coordinates` is passed through to
`numpyro_model` and `gauss_newton_mass` and tested both ways.

**A Gaussian guide is ≈ Laplace.** For a well-behaved posterior, a dense
Gaussian guide (reverse KL over the whole posterior) and the Laplace
approximation (curvature at the mode) nearly agree, and virgil already
has the latter (`gauss_newton_mass`, `laplace_cov`, `laplace_samples`).
SVI earns its keep only with a guide that can capture non-Gaussian
shape, so the default is a normalizing flow; `"mvn"` is kept and
documented as ≈ Laplace.

**Starting at the Laplace approximation (the Laplace frame).** numpyro's
flow guides (`AutoIAFNormal`, `AutoBNAFNormal`) start near a standard
normal in z and ignore `init_loc_fn` beyond the prototype, and its
Gaussian guides learn loc and scale in raw z, whose scales differ by
orders of magnitude (a position known to 3 mas on an 800 mas Uniform
range has σ_z ≈ 0.016). Adam's step is absolute, so a learning rate of
3e-3 jittered a binary's means by 0.2σ. Every virgil guide is therefore
z = z0 + L0 (shift + L · flow(ε)), with z0 and L0 fixed at the fit and the
Cholesky factor of its Gauss–Newton covariance, and `shift` (from 0), `L`
(from I) and the flow learnt: everything is learnt in units of the
Laplace widths. Without a flow this is exactly the GN Laplace Gaussian at
step 0. After the change the binary's means agreed with NUTS to 0.07σ
(`"mvn"`) and 0.16σ (`"bnaf"`). numpyro's IAF default hidden width
(latent_dim) is too narrow in 2–3 dimensions to bend a banana; virgil uses
max(16, latent_dim).

**Guide comparison** (laptop, x64, 3000 steps, Adam 3e-3, 8 particles,
20 000 draws; NUTS 2000 warmup + 20 000 draws, one chain; times include
compilation). Each cell is mean ± σ [5%, 95%]. `GN-Laplace` is the
Gauss–Newton Gaussian at the fit, without SVI.

*Banana*: x ~ N(0, 1), y | x ~ N(x²/2, 0.3²) (Uniform priors).

| Method | time | k̂ | x | y |
| --- | --- | --- | --- | --- |
| NUTS | 1.2 s | | 0.02 ± 0.96 [−1.57, 1.61] | 0.46 ± 0.67 [−0.34, 1.87] |
| GN-Laplace | 0.1 s | | −0.01 ± 0.99 [−1.63, 1.63] | 0.00 ± 0.30 [−0.49, 0.49] |
| mvn | 1.7 s | 0.70 | 0.03 ± 0.52 [−0.83, 0.88] | 0.13 ± 0.30 [−0.36, 0.62] |
| iaf | 1.7 s | 0.41 | 0.04 ± 0.80 [−1.30, 1.32] | 0.32 ± 0.58 [−0.37, 1.40] |
| bnaf | 3.7 s | 0.63 | 0.01 ± 0.88 [−1.42, 1.54] | 0.46 ± 0.57 [−0.31, 1.58] |
| laplace | 1.1 s | | 0.00 ± 0.98 [−1.62, 1.59] | 0.00 ± 0.30 [−0.50, 0.49] |

*Curved ridge*: a, b ~ LogUniform(0.01, 100), a·b = 1 ± 0.05, log a = 0 ± 1.5.

| Method | time | k̂ | a | b |
| --- | --- | --- | --- | --- |
| NUTS | 2.0 s | | 2.92 ± 6.21 [0.082, 11.9] | 3.00 ± 6.68 [0.084, 12.0] |
| GN-Laplace | 0.1 s | | 2.39 ± 3.84 [0.104, 9.5] | 2.42 ± 3.87 [0.106, 9.6] |
| mvn | 0.5 s | 0.75 | 2.59 ± 4.17 [0.099, 10.5] | 2.45 ± 4.04 [0.096, 10.1] |
| iaf | 0.5 s | 2.25 | 2.86 ± 4.82 [0.087, 11.8] | 2.66 ± 4.72 [0.085, 11.4] |
| bnaf | 0.6 s | 2.18 | 2.86 ± 4.21 [0.111, 11.6] | 2.14 ± 3.34 [0.087, 9.0] |
| laplace | 0.4 s | | 1.92 ± 2.66 [0.142, 6.8] | 1.95 ± 2.73 [0.147, 7.0] |

*Bounded scale*: s ~ LogUniform(1e-3, 1) measured as 0.03 ± 0.03 (a
tail in log s down to the bound), m ~ Uniform(−1, 1) with m + 3s = 0.3 ± 0.1.

| Method | time | k̂ | s | m |
| --- | --- | --- | --- | --- |
| NUTS | 0.7 s | | 0.0183 ± 0.0190 [0.0013, 0.058] | 0.246 ± 0.113 [0.056, 0.429] |
| GN-Laplace | 0.0 s | | 0.0461 ± 0.0477 [0.0066, 0.140] | 0.205 ± 0.133 [−0.021, 0.419] |
| mvn | 0.3 s | 0.89 | 0.0177 ± 0.0196 [0.0034, 0.054] | 0.251 ± 0.114 [0.056, 0.432] |
| iaf | 0.6 s | 0.50 | 0.0164 ± 0.0173 [0.0013, 0.053] | 0.243 ± 0.122 [0.049, 0.449] |
| bnaf | 0.6 s | 0.65 | 0.0178 ± 0.0193 [0.0013, 0.058] | 0.256 ± 0.117 [0.058, 0.445] |
| laplace | 0.2 s | | 0.0439 ± 0.0406 [0.0080, 0.124] | 0.198 ± 0.126 [−0.016, 0.398] |

**Choice of default: `"bnaf"`.** It is the only guide that reproduces the
banana (y's mean, width and 95% quantile), and the bounded scale's 5%
quantile (the Gaussians stop at 0.003, a factor 2.5 above NUTS's 0.0013).
On the ridge, whose a·b tail is heavy (NUTS's σ is 6, dominated by rare
draws), every guide underestimates σ; the flows' 95% quantiles are
closest. IAF is close behind and cheaper, but bent the banana only
partly. At a learning rate of 1e-2 the flows were noisier on the ridge
and the bounded scale, and with numpyro's IAF width (2 units) IAF failed
on the banana (σ_x = 0.41) at every setting tried (1e-3 to 1e-2, 8 or 32
particles, 2000 to 10 000 steps). Settings follow H. McDougall's
numpyro SVI notes (Adam ~1e-3, several particles per step, check that the
loss plateaus). AutoDAIS was not tried: those notes found it ~10 times
slower with fragile convergence. On a 7-hole-mask binary (a near-Gaussian
posterior) `"mvn"` and `"bnaf"` both agreed with NUTS (0.07σ and 0.16σ in
the means, widths within 10%), as expected when Gaussian ≈ Laplace.

**PSIS k̂** (`numpyro.infer.importance.psis_diagnostic`, numpyro ≥ 0.20;
`None` otherwise) is reported but is a rough guide on these toys: on the
bounded scale BNAF matched NUTS's quantiles while k̂ ranged 0.6–1.5 between
runs, and on the ridge the flows' k̂ ≈ 2 flags the missed tail.

**Risks and open items.** Multimodality: every guide covers one mode, so
orbits (Ω/ω flips, period aliases) and images need several starts
(`OrbitStart.chain_values`) or importance reweighting with k̂ as the check.
High-dimensional images: a flow costs O(d²) per layer, and a dense
Gaussian a d × d Cholesky; use the Laplace approximation
(`laplace_samples`) there. A fair comparison at tens of parameters (a
binary with error terms, a short orbit) is a candidate OzSTAR job; the
comparison script is not in the repository. Angle vectors: a Gaussian or
flow guide on a ring works but approximates wide angles poorly.
float32 runs (tested on the binary) but x64 is the default, as for `fit`.
