# Common arguments

[`fit`][virgil.fitting.fit], [`numpyro_model`][virgil.likelihood.numpyro_model],
[`gauss_newton_mass`][virgil.fitting.gauss_newton_mass] and the other fitting
entry points share these arguments. Each is documented once here; docstrings
cover only what is specific to a function.

## `model` {#model}

Either a template model whose leaves at the paths in `priors` are free (for
`fit`, their values are the starting point), or a function that takes the
parameter values as keyword arguments and returns a
[`SourceModel`][virgil.models.SourceModel]. Use a function for parameters that
are not leaves of the model, such as a separation and position angle, or one
inclination shared by two components (see
[`build_model`][virgil.likelihood.build_model]).

The function may return a list of models, one per dataset in `data`, sharing
parameters: a scene and a [`Rotated`][virgil.models.Rotated] copy of it for two
epochs between which it turns, say, or binaries at two epochs with one flux
ratio and a position each. Model `i` is compared with dataset `i`.
Regularizers cannot be combined with a list of models: that raises a
`ValueError`, since they would act on the first model only.

## `priors` {#priors}

A dict from parameter path (e.g. `"comp.flux"` or `"env.log_brightness"`; see
[`image_priors`][virgil.imaging.image_priors]), or function argument name, to
a numpyro distribution. In `numpyro_model` each key is also the sample-site
name. [Which priors?](conventions.md#which-priors) helps you choose.

- Priors on fluxes (keys named `flux` or ending in `.flux`) must have
  non-negative support.
- An angle (degrees) with an [`AngleVector`][virgil.angles.AngleVector] prior is
  sampled as a 2-D vector, with no wrap at 0°/360°. `numpyro_model` samples
  `"<path>_vec"` and records the angle as the deterministic site `"<path>"`;
  `fit` gives its von Mises prior, if any, a least-squares form.
- `fit` optimizes a prior that is uniform in some coordinate in that *flat
  coordinate* (see [`flat_coordinates`](#flat_coordinates)). Other priors
  (`Normal`, `Beta`, `HalfNormal`, ...) are evaluated in the model's own
  parameters, without the Jacobian of the bijection. The `values` that `fit`
  returns are always in the model's own parameters; see
  [Priors and the MAP](conventions.md#priors-and-the-map) for why.

## `data` {#data}

An `OIData` or a sequence of them, fitted jointly. It may be empty (`()`) when
`likelihoods` holds all the data. In `numpyro_model` their Gaussian log
likelihood is added with `numpyro.factor`.

## `noise` {#noise}

Priors on error terms fitted with the parameters, as `{term: prior}`. A dict
applies to every dataset (numpyro sites and `fit` values `"noise.<term>"`); a
list of dicts gives each dataset its own (`"noise[i].<term>"`).

| Term | Meaning | See |
|---|---|---|
| `vis_scale`, `phi_scale` | multiply the visibility and phase uncertainties | [`inflated_errors`][virgil.likelihood.inflated_errors] |
| `vis_error_rel`, `phi_error` | added in quadrature: a fraction of the model visibility; radians | [`inflated_errors`][virgil.likelihood.inflated_errors] |
| `vis_gain_<group>` | width of gains correlated across channels | [`OIData.with_gains`][virgil.oidata.OIData.with_gains] |
| `phi_offset_<group>` | width of closure-phase offsets | [`OIData.with_closure_offsets`][virgil.oidata.OIData.with_closure_offsets] |
| `wavel_scale`, `wavel_offset` | wavelength scale and offset | [`OIData.with_wavelength_scale`][virgil.oidata.OIData.with_wavelength_scale] |
| `north_angle` | the North angle, in degrees | [`OIData.with_north_angle`][virgil.oidata.OIData.with_north_angle] |

The priors of `wavel_scale`, `wavel_offset` and `north_angle` may take either
sign (e.g. `Normal(1, 2e-4)`).

**Priors.** Error terms are scales, so give them log-uniform priors on stated
bounds; a `Uniform(0, ...)` favours large values. Choose bounds that contain
the plausible values: `dist.LogUniform(0.1, 10.0)` for the factors
`vis_scale` and `phi_scale`, whose neutral value is 1, and
`dist.LogUniform(1e-4, 0.3)` for the added errors and widths (`vis_error_rel`,
`phi_error`, `vis_gain_<group>`, `phi_offset_<group>`). `wavel_scale` is a
scale too: log-uniform about 1, unless a calibration supplies a Gaussian.

**Tied terms.** An entry may be a function of the dict of fitted or sampled
values (keyed like `priors`) instead of a prior. This ties the term to
parameters in `priors`, as in a hierarchical model where each epoch's error
scale comes from a population with fitted hyperparameters (see
[`hierarchical_scales`][virgil.priors.hierarchical_scales]). `numpyro_model`
records it as a deterministic site and adds the term's `log_prior(values)`,
if it has one, once (site `"noise_prior"`).

**Loss.** With fitted error terms the loss is the full Gaussian negative log
likelihood, including $\sum\log\sigma$, so `fit` defaults to L-BFGS, as it
does for data with gains, whose covariance depends on the model. Don't fit
error terms together with an image: a smoother image with larger errors fits
as well. Estimate them with a parametric model first.

## `likelihoods` {#likelihoods}

Further Gaussian likelihood terms beyond the visibilities: callables of the
parameter values (a dict keyed like `priors`) that return whitened residuals.
The built-in ones, [`PositionData.term`][virgil.orbits.PositionData.term]
and [`RVData.term`][virgil.orbits.RVData.term], cover an orbit's positions and
radial velocities and add their full normalized Gaussian log density, like
the `OIData` terms; a plain callable adds only $-\tfrac12\sum r^2$. In
`numpyro_model`, term `i` is added with `numpyro.factor` as site
`"likelihood_<i>"`; in `fit` their χ² follow the datasets' in `info`.

## `regularizers` {#regularizers}

Penalties or log-prior terms on the model, e.g. from
[`virgil.imaging`][virgil.imaging]. `fit` takes any of them. `numpyro_model`
takes only genuine prior densities (those with `probabilistic` true, such as
a [`Centroid`][virgil.imaging.Centroid] prior), added with `numpyro.factor`;
penalties such as maximum entropy are for `fit`.

## `flat_coordinates` {#flat_coordinates}

`fit` always optimizes a prior that is uniform in some coordinate in that
*flat coordinate*: $\log x$ for `LogUniform(a, b)` on $[\log a, \log b]$,
$\cos i$ for an
[`IsotropicInclination`][virgil.priors.IsotropicInclination], $\sin(\mathrm{lat})$
for an [`IsotropicLatitude`][virgil.priors.IsotropicLatitude] (any prior with a
`flat_coordinate()` method), and the parameter itself for `Uniform`.

For `numpyro_model` and `gauss_newton_mass`, `flat_coordinates` (default
`False`) chooses the coordinates NUTS moves in. `False` uses numpyro's
bijection of each prior's support. `True` uses `fit`'s flat coordinates, for
parameters and error terms alike: a `LogUniform(a, b)` moves in the logit of
$\log x$ on $[\log a, \log b]$, where the prior's potential is exactly
logistic. Site names, values and the posterior are unchanged; only the
coordinates NUTS moves in differ.

- **Pass the same value to both functions.** With `adapt_mass_matrix=False` a
  matrix in the wrong coordinates is never corrected, and sampling is silently
  slow. `numpyro_model` records the choice as its `flat_coordinates`
  attribute, and [`chain_init_params`][virgil.likelihood.chain_init_params]
  (given `mass=`) raises on a mismatch. Unconstrained `init_params` computed in
  numpyro's coordinates do not apply with `True`.
- It does not cure hierarchical funnels, which come from the dependence
  between a group scale and its members: use the non-centred form or the mass
  matrix for those.
- It is off by default because simulation-based calibration (1000 replicates)
  showed no gain over numpyro's bijection, in effective samples per second or
  in divergences.

## `dtype` {#dtype}

Precision of a fit: `"float64"` (the default) runs in a local
`jax.enable_x64` context; `"float32"` is faster and less precise. The returned
model and values are cast back to JAX's precision outside the fit (float32,
unless x64 is enabled) so that they work with the rest of your code. See
[Precision](conventions.md#precision).

## Stopping rules of `fit` {#fit-stopping}

LM and L-BFGS stop when no component of the gradient of the loss per data point exceeds `gtol`, nor 1/1000 of its largest starting value (floored at √eps of the `dtype` times that value, and at `1e-6 * gtol`). `max_step_size` limits L-BFGS to a change of at most that much in any unconstrained coordinate per step, a factor `exp(max_step_size)` for a log-brightness pixel. Adam converges if the final gradient meets the same test (`ftol_abs` is its own, optional, plateau stop). `time_limit` stops after the first chunk that ends past it, compilation included. LM is the default when the objective has a least-squares form (Normal priors, and priors flat in their fitted coordinate); above 200 coordinates it is matrix-free (`cg_steps`).
