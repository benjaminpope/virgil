# Changelog

All notable changes to this project are recorded here, in the style of
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/). The project follows
[semantic versioning](https://semver.org/), with the usual caveat that
anything before 1.0 may change between minor versions.

## Unreleased

### Migration from 0.3.0

- **Model before data in the grid, limit and detection tools.**
  `likelihood_grid`, `optimized_likelihood_grid`, `optimized_flux_grid`,
  `linear_flux_grid`, `laplace_flux_uncertainty_grid`, `absil_limits`,
  `injection_limits` and `detection_statistics` are now `f(model, data,
  grid, ...)`, as `fit` already was; `gaussian_null`, `rescale_errors`
  and `bootstrap_null` take the null scene first; `injection_recovery`
  is `(model, null_scene, template, grid, key, ...)`. Calls in the 0.3
  order still work, with a `FutureWarning` naming the new call; from 0.5
  they raise a `TypeError` naming it.
- **Model before data in the likelihoods.** `loglike`, `laplace_cov`,
  `laplace_parameter_uncertainty` and `fisher` are now `f(values, params,
  model, data, ...)`, and `joint_prediction` and `joint_loglike` are
  `f(params, model, data)`, with the same `FutureWarning` for the 0.3
  order until 0.5.
- **One name for each argument.** The model is `model` and the data
  `data` everywhere: `whitened_residuals`, `model_loglike`,
  `flux_scale_posterior`, `numpyro_model` and
  `posterior_predictive_summary` (whose order was already right) and the
  reordered functions accept `model_object=`, `model_fn=`, `data_obj=`
  and `observations=` with a `FutureWarning` until 0.5.
- **`samples_dict` is now `grid`** in these functions and in
  `best_grid_point`, `plot_grid_map` and `plot_contrast_curve`, and their
  data argument is `data`. `samples_dict=` and `data_obj=` still work
  with a `FutureWarning` until 0.5.
- To find the remaining 0.3-order calls in your own code before 0.5, add
  `"error:.*stops working in virgil 0.5:FutureWarning"` to your pytest
  `filterwarnings` (as virgil's own test suite now does), or run
  `pytest -W error::FutureWarning`. A message regex given to `-W` on the
  command line is matched literally, so it would not catch them.

### Changed

- **NUTS samples each prior in its flat coordinate, as `fit` optimises
  it.** `numpyro_model` now maps a `LogUniform` scale to NUTS's
  unconstrained coordinate through log x, an `IsotropicInclination`
  through cos i and an `IsotropicLatitude` through sin(lat), for
  parameters and `noise=` terms alike, where the prior is a logistic
  density. Sites keep their names, samples are the model's own
  parameters, and the posterior is unchanged; only the coordinate NUTS
  moves in differs, so it now matches `fit`'s and `gauss_newton_mass`
  (which defaults to the same coordinates). Pass
  `flat_coordinates=False` to both for the 0.3 coordinates (numpyro's
  bijection of each prior's support), e.g. to reuse saved unconstrained
  `init_params`. See `design/sampler_flat_coordinates.md`.
- **Orbit starts are scale-aware (#268).** `epoch_positions` now scores
  its grid on the scale-marginalized surface m = -Σ_b (ν_b/2) ln χ²_b,
  in which each dataset's V² and closure-phase error scales are
  integrated out under their Jeffreys priors (new
  `epochs.marginal_loglike`; Gaussian normalization in every block, with
  `s_max=` for a bounded numerical marginal where closure phases are
  weak, and `dof=` for an effective-dof fraction). The refinement
  maximizes m (equivalently, a fit with free `vis_scale` and
  `phi_scale`), and the covariance is the curvature of m, so positions,
  covariances and the new `EpochPositions.gap_marginal` do not change
  when one dataset's errors, or only its closure-phase errors, are
  rescaled. `gap` keeps its old value on the quoted errors for one
  release and will then switch to the marginal value; `decisive`,
  `positions(min_gap=...)` and `start_from_positions(min_gap=...)` now
  compare `gap_marginal`. **Behaviour change:** a night whose errors are
  underestimated by s used to look s² times more decisive than it is, and
  is no longer used to seed orbits unless its marginal gap passes
  `min_gap`. New fields `chi2_raw` (χ²/N on the quoted errors, per block
  and in all, N = `n_independent`) and `scale` (ŝ per block) are
  recorded, and a `UserWarning` names the datasets with raw χ²/N > 4.
  Grid fluxes must be at most 1 (and the refinement keeps f ≤ 1): f > 1
  at r is the same binary as 1/f at -r.
  Data with gains, closure-phase offsets or a model-dependent covariance
  raise a `NotImplementedError` on the marginalized surface, since their
  nuisance covariance is not multiplied by an error scale.
- `rank_orbits` and `start_from_positions` take `scales="quoted" |
  "marginal"`. `"marginal"` ranks with each dataset's error scales
  integrated out, so the worst-calibrated nights no longer dominate the
  ranking. The default stays `"quoted"` in 0.4, with a `FutureWarning`
  when `scales` is not given; it will become `"marginal"`.

### Fixed

- **Isotropic priors have finite gradients at the poles in float32.**
  `IsotropicInclination.icdf` and `IsotropicLatitude.icdf` gave a NaN
  derivative within about 1e-8 of a pole (the branch not taken of a
  `where` had an infinite slope), and their final clip a slope of 1/2 at
  the bounds; both now keep the true derivative.

- **`read_oifits` reads closure triangles whose leg has no visibility
  row** ([#299](https://github.com/benjaminpope/virgil/issues/299)). A
  triangle leg stored in no `OI_VIS2`/`OI_VIS` row in either orientation
  raised a `ValueError`, so MIRC-X files and some ESO phase-3 PIONIER
  products (which omit a baseline's V²) could not be read. `OI_T3` is
  self-contained in the OIFITS standard, so such a leg now gets a flagged
  sample at the T3 row's own coordinates (`(U1COORD, V1COORD)`,
  `(U2COORD, V2COORD)` or their sum), shared by later triangles of the
  same frame, as reversed legs already were. One `UserWarning` per file
  gives the number of legs placed this way. The `ValueError` remains when
  those coordinates are NaN or zero, or disagree with the triangle's other
  legs (a table written with the opposite baseline direction).

- **`epoch_positions` no longer commits to a peak before refining it.**
  It refined only the best grid point, which is not always the best peak
  once refined: fringe peaks are often narrower than the grid step, and
  on Gl 229 (GRAVITY) two nights committed to a wrong peak with a
  `gap_marginal` above `min_gap`, which then seeded the orbit search in a
  wrong mode. It now refines the top `n_peaks=5` distinct local maxima of
  the grid (more than `gap_mas` apart), takes the best refined one as the
  position, and measures `gap_marginal` against its best refined rival
  (with the grid as a floor). The new `EpochPositions.peaks` holds each
  dataset's catalogue (`EpochPeaks`: positions, fluxes, quoted and
  marginal scores, weights, an `edge` flag and a `laplace` flag; also
  `EpochPositions.edge`). The weights are each peak's Laplace mass
  exp(m) det(H)^(-1/2) over (dra, ddec, flux), normalized, falling back
  to the height exp(m) where `laplace` is false (no refinement, or a
  Hessian that is not positive definite). Refined peaks within `gap_mas`
  of a higher one are dropped, so two grid cells that refine to one
  maximum are counted once. The quoted `gap` stays the grid quantity for
  its last release. The cost of the refinement grows linearly with
  `n_peaks`, which must be an integer of at least 1 (a `ValueError`
  otherwise). The edge warning now fires when either the best grid point
  or the best refined peak's starting cell is on the grid edge, with a
  message for each case; before, only the best grid point was tested.
  `start_from_positions(n_peaks=)` passes it on, and
  `OrbitStart.seeded` and `OrbitStart.ambiguous` name the datasets that
  seeded the orbits and those left out (with a `UserWarning`), so that no
  night is dropped silently.
- `ensemble.combine` drops members whose fit diverged (a non-finite χ²),
  with `reason="diverged"`, instead of rejecting every member with a
  `ValueError`, and `LCurve.corner` ignores non-finite points so that a
  diverged fit cannot move the corner.
- Parallel numpyro chains (`chain_method="parallel"`) on data with
  `with_gains` no longer segfault. `GainModes` held zero-size arrays when
  no mode spans frames, and XLA's Shardy pass crashes compiling a
  `jax.pmap` that captures one (JAX 0.11.2). `GainModes.spanning` and
  `spanning_group` are now `None` then. `numpyro_model` warns, suggesting
  `chain_method="vectorized"`, when the data still hold zero-size arrays
  (e.g. no closure phases) and more than one device is visible.
- `metrics.rms_convolved` no longer needs `pixel_scale_mas` for array
  inputs when no `beam` is given (the docstring already said so); a beam
  with arrays and no pixel scale still raises a clear `ValueError`.
- The `lawson_sigma_over_peak` docstring now states that both images are
  normalised over the whole array, so flux in empty sky does affect σ.
  Values are unchanged.
- `limits.chi2ppf` works for any number of degrees of freedom on a
  standard install. For `df != 1` it called numpyro's `gammaincinv`,
  which needs `tensorflow_probability`, not a virgil dependency; it now
  inverts `jax.scipy.special.gammainc` by Halley's method for every `df`,
  matching `scipy.stats.chi2.ppf` to about 1e-14 relative in float64
  for p in [1e-10, 1 - 1e-10], and stays differentiable and `jit`-able.
  `df = 1` also uses it, gaining precision at small p. It returns NaN
  for `df <= 0`, and no longer needs `tensorflow_probability`. This
  supersedes the removal of `chi2ppf` in #293, which is reverted. Found
  by virgil-validation (F17).
- `epochs.marginal_loglike(..., s_max=...)` (and `scales="marginal"` in
  `epoch_positions` and `rank_orbits` with `s_max`) integrates each
  block's error scale adaptively in ln s instead of on a fixed 257-node
  grid, which missed by up to 0.7 in `m` where the likelihood falls
  steeply from a bound (a model with s ≈ 5 against `s_max = 1.2`). It now
  matches `scipy.integrate.quad` and the closed form in incomplete gamma
  functions to about 1e-11 in float64. `m` is now the log of the integral
  over ln s itself, so bounded values shift by a constant per block
  (differences between models are unaffected by it), and `gap_marginal`
  values computed with `s_max` may shift slightly. Found by
  virgil-validation (F18).

### Added

- **Variational inference** (virgil#12). `virgil.svi.variational(model,
  priors, data, ..., start=fit_result)` fits a guide to `numpyro_model`'s
  posterior with numpyro's SVI and returns a `VariationalResult`: draws in
  the model's parameters under NUTS's site names, the loss history, the
  guide and its parameters, a convergence flag and the PSIS k̂. Every guide
  starts at the Laplace approximation (the fit and its Gauss–Newton
  covariance, at the nominal errors and without regularisers; with
  regularisers it starts at `init_scale` instead) and learns in units of
  its widths. The default is a block
  neural autoregressive flow (`"bnaf"`), which follows curved, skewed and
  bounded posteriors; `"iaf"`, `"mvn"` (a Gaussian, ≈ Laplace) and
  `"laplace"` are the alternatives. float64 by default, as for `fit`;
  `flat_coordinates` is passed through. New page: Variational inference.

- **Fit budgets and guards.** `fit(..., time_limit=, progress=)`: LM and
  L-BFGS run in chunks of steps, check the wall clock between them and
  stop, unconverged, with `info["stop"] == "time"` (L-BFGS carries its
  whole state across chunks, so its path is unchanged; LM restarts each
  chunk with its damping reset); `progress` reports steps, loss, gradient
  norm and elapsed time between chunks. `info["stop"]` gives every
  unconverged fit's reason (`"limit"`, `"time"`, `"stalled"`,
  `"non-finite"`, `"failed"`), a non-finite loss now stops L-BFGS, a fit
  by any method (Adam included) that ends on a non-finite loss is now
  marked unconverged, and `info["at_bound"]` lists parameters that ended
  at a genuine edge of a Uniform, LogUniform or other interval prior
  (not at the poles of isotropic inclinations or latitudes).
  `start_from_positions` forwards `time_limit`, keeps going when one
  refinement fit raises (recorded in `OrbitStart.failed`; a `TypeError`
  or `ValueError`, such as a misspelt option, still raises at once, and
  if every fit fails the `RuntimeError` is chained to the last error),
  sorts non-finite losses last and leaves them out of `modes()`.
- `orbits.period_grid(times, p_min, p_max, k=9)`: trial periods uniform
  in frequency with δP ≤ P²/(kT) over the baseline T, finer than
  ARMADA's and The Joker's rules; `start_from_positions` warns when its
  `periods` are coarser than that over the seeding datasets.
- A private `virgil._deprecate` module for the 0.4 change to a single
  model-before-data argument order. `old_order` lets a function written
  in the new order accept calls in the 0.3 positional order, recognized
  by which argument is the `OIData`, with a `FutureWarning` that names the
  new call; with `removed=True` (for 0.5) such a call is a `TypeError`
  naming the new call. `old_order` and `renamed` also accept the old
  keyword names `data_obj=`, `observations=`, `model_object=`,
  `model_fn=` and `samples_dict=` for `data`, `model` and `grid`, with a
  `FutureWarning`; passing an old name with its new one is a `TypeError`.
  No public function uses it yet, so nothing changes for users.
- Starting and sampling orbit fits to multi-epoch visibilities
  (`design/visibility_orbits.md`, stage 2), all model first:
  `rank_orbits(model, data, orbits)` ranks trial orbits (e.g. from
  `starting_orbits`) by the log likelihood of all the epochs, one snapshot
  per dataset; `chain_starts` picks the best distinct orbits (modes judged
  by the companion's positions at the epochs), one per chain;
  `epoch_positions` fits the companion's position and covariance in each
  dataset; `start_from_positions(model, priors, data, start_values, ...)`
  chains these into a start (positions, starting orbits, ranking, `fit`
  from distinct orbits) returned as an `OrbitStart`, whose `chain_values`
  give one start per distinct mode; and `likelihood.chain_init_params`
  turns one start per chain into numpyro's `init_params`.
  `Epochs.resolution_mas` gives the data's finest λ/B_max.
- The orbit tutorial now fits one snapshot per night with `Epochs` and
  `OrbitalBinary` (the default for orbits whose motion within a night is
  negligible), starts from `start_from_positions`, and starts one NUTS
  chain per distinct mode.

## 0.3.0 (2026-10-06)

Everything listed as new was added after 0.2.0.

### Migration from 0.2.0

Most of 0.3.0 is additive. These are the changes that can alter an existing
analysis or warn:

- **`fit` with a `LogUniform` prior now returns the MAP in `log x`.** This
  covers priors on parameters and on `noise=` terms (#226). Each prior is
  optimised in its flat coordinate (`LogUniform(a, b)` in `log x`; isotropic
  priors in `cos i` or `sin(lat)`), where it is constant, so the fit is the
  maximum of the likelihood inside the prior's range. In 0.2.0 it was the mode
  of likelihood × `1/x` in `x`, which pulled scales towards small values, so
  fitted values of scales (fluxes, diameters, noise terms) can shift.
  Levenberg–Marquardt is now chosen automatically for these priors, where
  0.2.0 used L-BFGS and `method="lm"` raised. Uniform, Normal and other
  priors are unchanged. See "Priors and the MAP" in the conventions.
- **Frames are grouped by `MJD` and `TIME` (#203), which changes χ² for
  OIFITS v1 files** that keep the snapshot in `TIME` rather than `MJD`
  (OYSTER output, the 2004 Interferometry Beauty Contest files). Snapshots
  that share an `MJD` used to be one frame, and closure phases from
  different snapshots were whitened as one correlated group. They are now
  separate frames: for the 2004 contest files the independent closure phases
  go from 10/130 to 130/130, and every χ² and uncertainty from them
  changes. Files with a constant `TIME`, and the other files we tested
  (AMBER, CHARA, GRAVITY, MIRC), are unchanged.
- **`starting_image` uses `LogUniform` priors** on the envelope width and
  flux (flux bounds 1e-4 to 100), so the starting point, and a fit that
  begins from it, can differ slightly from 0.2.0.
- **Keyword-only grid and limit arguments.** `flux_param` and `batch_size`
  must be passed by keyword to `likelihood_grid`, `optimized_likelihood_grid`,
  `optimized_flux_grid` and `linear_flux_grid`, and `flux_param`,
  `flux_bounds` and `batch_size` to `absil_limits` and `injection_limits`;
  positional calls from 0.2.0 raise `TypeError`. A bare `(mean, sd)` tuple
  as `linear_flux_grid(prior=)` is a `TypeError` (use `dist.Normal`), and
  `RVData.term(marginalise_offsets=True)` needs a stated zero-point prior.
- **`injection_limits(flux_bounds=None)` limits change** where 0.2.0
  returned the top of its search range (1000); both limit functions now
  find the first crossing from below. Bounded results agree with 0.2.0 to
  about 1e-4.
- **L-BFGS keeps 50 past steps** (`fit(lbfgs_memory=50)`, was optax's 10),
  so L-BFGS fits, regularised images especially, change slightly.
- **`total_mass` and `distance_pc` use GM☉** instead of a³/P² in Julian
  years, which raises masses by 3.8e-5 (relative).
- **`Tabulated` is deprecated** in favour of `virgil.spectra.Nodes`: it
  now raises a `DeprecationWarning`, behaviour is unchanged, and removal is
  planned for 0.4.0.

### Added

- **Per-observable error scales.** `error_scale(model, data,
  by_observable=True)` re-estimates MacKay's noise scale separately for each
  kind of observable (`"vis"`, `"phi"`, and each kind in `extras`, pooled over
  datasets), solving the coupled equations `1/β_b = χ²_b/(N_b − γ_b)` with
  `γ_b` from the hat matrix. Use it when blocks are mis-calibrated by
  different factors (in a MATISSE N-band contest file, V² gave χ² per point
  0.005 and closure phases 0.49). `OIData.with_error_scale` accepts the
  returned dictionary of factors per kind. Data whose covariance includes
  nuisance terms that do not scale with the quoted errors (gains, closure
  offsets, marginalised flux scales, differential phases with a finite
  `prior_width`) raise a `ValueError`. The default single scale is unchanged.

- **Hierarchical error scales and tied `noise=` terms.** A `noise=` entry of
  `fit` and `numpyro_model` may now be a function of the sampled parameters
  instead of a prior (`likelihood.is_tied`), recorded as a deterministic site;
  a tied term's `log_prior(values)`, if any, is added once
  (`likelihood.tied_log_prior`). `priors.hierarchical_scales(name, n,
  median=, spread=, centred=True)` builds n scales (e.g. one closure-phase
  error scale per epoch) from a log-normal population with log-uniform
  hyperpriors: centred (log s sampled, for well-measured members; the
  non-centred form diverged on such members in a test) or non-centred.

- **`fit` reports `info["grad_norm"]`**, the infinity norm of the gradient
  of the loss per data point at the result, in the unconstrained
  coordinates of the convergence test, for every method (LM, L-BFGS, Adam).

- **`imaging.laplace_samples`** draws Gaussian-field images from the
  Laplace (Gauss–Newton) posterior of the whitened latents about the MAP,
  `N(z_MAP, (I + JᵀJ)⁻¹)`, and optionally renders the whole model for each
  draw. Other fitted parameters are held at the MAP.

- **`clean(base_priors=...)`** fits named parameters of the base (e.g. a
  companion's position) together with the components' fluxes at every major
  cycle. CLEAN also ends with a final major cycle when it stops at the target
  or at `max_iterations`, so its fluxes are refitted even when the data start
  close to the target and no cycle has run.

- **Multi-scale CLEAN**: `clean(scales_mas=(0.0, ...))` adds components
  that are pixels convolved with circular Gaussians of the given FWHM
  (Cornwell 2008), so extended emission takes a few broad components
  instead of many points. The search and the major cycles run over every
  (scale, pixel). `CleanResult` gains `components_by_scale` and
  `scales_mas`; `components` is still the total image. The default
  `scales_mas=(0.0,)` gives exactly the previous point-only results.
  An optional `scale_bias` favours small scales; it is off by default.

- **`fit(lbfgs_memory=50)`.** L-BFGS now keeps 50 past steps (optax's default
  is 10). On regularised images, 10 stopped short of the optimum at many
  weights; 50 reached lower losses (lower χ² and lower penalty together) and
  converged in 2–3× fewer steps for StarletL1, at a higher cost per step.
  L-BFGS fits therefore change slightly from earlier versions.

- **Isotropic-orientation priors.** `virgil.priors` has
  `IsotropicInclination(low=0, high=180)` (degrees, density ∝ sin i, so cos i
  is uniform; use `(0, 90)` when only |cos i| is identifiable) and
  `IsotropicLatitude(low=-pi/2, high=pi/2)` (radians, density ∝ cos lat), as
  numpyro distributions for `fit` and `numpyro_model`. Docstrings and error
  text now recommend Jeffreys-consistent priors: `LogUniform` for scales
  (RV jitter, fluxes, diameters, separations), with `ruffio_upperlimit`
  documenting why its flat flux prior is deliberate.

- **Log-uniform (scale-invariant) detection prior for `linear_flux_grid`.**
  `prior=LogUniform(f_min, f_max)` puts the scale-invariant (Jeffreys, under
  the scaling group) prior, `1 / f`, on the companion flux ratio, and is now
  the documented recommendation. The flux ratio is a scale parameter
  spanning decades, so the prior is the invariant measure of the scaling
  group, not the root-Fisher prior of the linearised likelihood (which has
  constant Fisher information and would be flat). The evidence needs a proper prior, so both bounds are
  required, and the Bayes factor depends on them as it must (widening them
  changes `log_bayes_factor` by about `-Δ ln ln(f_max / f_min)`). The
  evidence, posterior mean and sd come from 256-node Gauss-Legendre
  quadrature in `ln f` over the part of the bounds the likelihood occupies,
  inside `jit` and `vmap`; they agree with a dense-grid quadrature to
  better than 1e-4. A Gaussian prior is `dist.Normal(mean, sd)` (a
  Gaussian-prior evidence, a computational approximation where `f` may go
  negative). With no `prior`, the posterior and Bayes-factor fields are
  `None`, whatever the prior kind.

- **Angle vectors (`virgil.angles.AngleVector`).** A prior for any angle
  (degrees), sampled as a 2-D vector at the site `"<path>_vec"` with the
  angle as the deterministic `"<path>"`, so there is no wall at 0°/360°.
  A ring prior on the radius keeps MAP fits off the origin; von Mises and
  axial von Mises priors are chords √κ (v̂ − m̂), the same form as the phase
  residuals, so `fit`'s Levenberg–Marquardt and `gauss_newton_mass` take
  them (`fit` used to reject `VonMises`). The density is normalised in the
  plane. For orbits, `orientation_priors` samples 2Ω and ϖ = Ω + ω (or Ω
  and ϖ with RVs), and `KeplerOrbit.from_varpi` builds the orbit. After
  Octofitter's `UniformCircular` and exoplanet's `Angle`.

- **`KeplerOrbit.from_position_angle`.** The position angle θ at `t_ref`
  as an alternative to `dt_peri`, for short arcs (after Thompson et al.
  2023). `position_angle_prior(orbit_fn)` is a `likelihoods=` term adding
  log|∂M/∂θ| (`position_angle_log_jacobian`), so a uniform θ gives the
  invariant prior, uniform in the time of periastron. Singular at i = 90°.

- **Spectro-interferometric observables (Stage 6a, PR B).**
  `read_oifits(..., extras=...)` and `OIData(path, extras=...)` read OI_FLUX
  (`"flux"` or `"nflux"`), T3AMP, VISAMP beside V² (absolute, or correlated
  flux as `AMPTYP` declares) and VISPHI as a differential phase beside
  closure phases. They are off by default, so existing analyses are
  unchanged. The new `virgil.observables` blocks follow the phases in the data
  vector. Spectra are predicted from `total_spectrum`, with the grey scale
  (and optionally a polynomial in λ) marginalised analytically under a broad
  Gaussian prior; `likelihood.flux_scale_posterior` reports it. Differential
  phases use the exact arg V, with the continuum normalisation (offset and
  delay over continuum channels) applied as a linear operator and its
  propagated covariance N D Nᵀ. Beside closure phases, only their
  closure-free part in the line windows is used, so nothing is counted
  twice. New methods: `OIData.with_continuum`, `with_flux_scale` and
  `with_error_floor` (PMOIRED-style floors per observable). The writer
  gains OI_FLUX and the `AMPTYP`/`PHITYP`/`CALSTAT` keywords. Guide:
  "Spectro-interferometric observables".
- **`vis_error`, and `where=`/`combine=` in `inflated_errors`.** A fitted
  absolute visibility error term. Floors and fitted terms share one rule,
  `_utils.inflate_errors`.

- **Detection statistics for ROC curves (`virgil.detection`).**
  `detection_statistics(data, model, samples_dict)` returns, from one
  companion grid search, the profile likelihood ratio `delta_chi2` (flux >= 0),
  a grid-marginalised `log_bayes_factor` against no companion (trapezoid prior
  weights, uniform in position and following the flux axis's spacing), the
  largest flux/σ `max_snr`, and the best position and flux. It is traceable in
  the data, so it compiles once under `jax.lax.map` over simulated datasets;
  `local_nsigma` gives the single-position (Wilks) significance. A new tutorial,
  "Detection ROC curves", calibrates the statistics with simulations. Design
  and later stages in `design/detection_roc.md` (virgil#2).
- **Detection Monte Carlo (`virgil.detection`, stage 2 of virgil#2).**
  Null simulators `gaussian_null` (noise from the template's errors, with an
  `error_scale` for mis-estimated errors) and `bootstrap_null` (a sign-flip or
  resampling residual bootstrap about the null scene; correlated closure
  phases are whitened and re-coloured with the new `ClosureNoise.colour`),
  and `rescale_errors` (visibility and phase errors scaled separately so the
  null has χ²_r = 1). `injection_grid` lays out companions at random PAs, and
  `injection_recovery` runs the search on null and injected draws with one
  compiled kernel (`jax.lax.map`, chunked, with a progress bar). It returns a
  NumPy `DetectionMC` with empirical false-alarm probabilities and their
  Clopper–Pearson intervals, thresholds with bootstrap errors, ROC curves,
  AUC, completeness maps, contrast curves in the units of `absil_limits`, an
  optional `match_radius` (Cartesian `dra`/`ddec` or angular `sep`/`pa`
  grids), and `save`/`load`/`concatenate` for array jobs; `concatenate`
  compares fingerprints of the whole model, null scene and template (every
  field, static or not) and refuses runs that cannot be fingerprinted.
- **Detection plots (`virgil.plotting`, stage 3 of virgil#2).**
  `plot_null_distribution(mc, stat, observed=, fap=)` draws the empirical
  false-alarm probability of every threshold on a log scale, with the
  single-position reference (½χ²₁ for Δχ², the Gaussian tail for max SNR),
  the threshold at a FAP, and an observed value with its FAP and 95%
  interval; its legend sits outside the axes so it never hides the tail.
  `plot_roc(mc, stat, flux=, sep_bin=)` draws ROC curves for one or several
  statistics, fluxes or separation bins (distinct colours and line styles),
  on a log false-positive axis by default with the chance curve TPR = FPR
  drawn as a curve, and marks the one-position FAP of local 3σ and 5σ
  (0.135%, 2.9×10⁻⁷) next to where each curve's threshold equals them, to
  show the look-elsewhere effect. `plot_completeness(mc, stat, fap,
  units=)` draws the completeness map against separation and Δmag, contrast
  or flux, with the 50% and 90% `contrast_curve`s, on axes that
  `plot_contrast_curve` can draw Absil or Ruffio limits onto.

- **Gauss-Newton and a marginal-likelihood map in `linear_flux_grid`.**
  `n_iter=k` relinearises the whitened residuals at the current flux per pixel
  for `k` extra steps, removing the bias for bright companions (at f = 0.3,
  `n_iter=3` agrees with `optimized_flux_grid` to 3e-7 and with the Laplace
  `sigma_f` to 2e-5, where `n_iter=0` is 34% low). `prior=Gaussian(mean, sd)` puts a
  Gaussian prior on the flux and fills the `posterior_mean`,
  `posterior_sd` and `log_bayes_factor` fields (closed-form evidence ratio
  against f = 0, in the linearised model about the final point). The result
  is a `LinearFluxGrid` named tuple (those three fields are `None`
  without a prior), so unpack it by attribute, or take the first three
  with `flux, error, snr = res[:3]`; unpacking it into three names directly
  fails.

- **Fitted RV jitter.** `RVData.term(params, jitter="rv_jitter")` inflates the
  errors to `sqrt(d_rv² + s²)` with `s` a fitted value (km/s; give it a
  half-normal or log-uniform prior). The term protocol gains an optional
  `log_norm(values)`, the `Σ log σ_eff` that a fitted error makes
  non-constant: `fit` adds it to the loss, defaults to L-BFGS (as for
  `noise=`), and raises `TypeError` for `method="lm"`. `numpyro_model` needs
  no change, since the term's `loglike` is already normalised.

- **Marginalised RV zero points.** `RVData(..., instrument=labels)` and
  `RVData.term(params, marginalise_offsets=(mean, sd))` marginalise one
  velocity zero point per instrument analytically (Luger, Foreman-Mackey &
  Hogg 2017), in O(N k²) by the Woodbury identity and the matrix-determinant
  lemma, with the jitter-dependent log-determinant in `log_norm`.
  `term.posterior(values)` gives the zero points' conditional mean and
  covariance after a fit. Only a finite prior width is supported (`True` is
  N(0, 1000²) km/s and warns).

- **North-angle and plate-scale nuisances (#212).**
  `OIData.with_north_angle(angle)`, and the `noise=` term `north_angle`
  (degrees) in `fit`, `numpyro_model`, `model_loglike` and
  `whitened_residuals`, rotate a dataset's sky by a fitted angle: every
  position angle it measures is the true one plus the angle (North through
  East). The plate scale is the existing `wavel_scale`, as 1/m for a
  magnification m. For positions, `PositionData.model(orbit, north_angle=,
  plate_scale=)` and `PositionData.term(orbit, north_angle="path",
  plate_scale="path")` take the paths of fitted values. Both are off by
  default, there are no default widths, and a `uv_grid` is dropped when the
  rotation is applied. After Octofitter (Thompson et al. 2023).

- **Closure-phase offsets per frame** (Stage 6d).
  `OIData.with_closure_offsets(baseline=, triangle=, modes=)` adds closure-phase
  offsets common to a frame's channels (per baseline, as T·e; per triangle; or
  supplied modes), marginalised analytically on the whitened closure phases,
  with widths `phi_offset_baseline`, `phi_offset_triangle` and
  `phi_offset_modes`. Four or more telescopes; off by default.

- **`injection_limits`: injection-method detection limits.** Like
  `absil_limits`, with the same grid, inputs and return format, but each
  limit is the flux at which a companion injected into the data (Gallenne et
  al. 2015, section 3.2, as in CANDID's `detectionLimit(methods=["injection"])`)
  would be detected at the requested significance. It solves for the flux by
  bisection in log flux, vmapped over the grid. It agrees with CANDID's
  criterion to 1e-4; unlike CANDID it does not refit the primary's diameter
  to the injected data. It equals `absil_limits` on the data reflected about
  the null model, and neither is uniformly more sensitive.


- **`linear_flux_grid`.** A closed-form, fouriever-style (`lincmap`)
  linearised companion flux map for fast first-pass searches, beside
  `optimized_flux_grid`. The derivative of virgil's whitened residuals with
  respect to the flux at f = 0 is computed exactly with `jax.jvp`, so
  correlated closure phases are whitened as in the likelihood, and
  `f_hat = -(g . r0) / (g . g)`, `sigma_f = (g . g)**-0.5` and SNR are
  returned per pixel (f_hat unconstrained in sign). Valid only for
  f much smaller than 1: a bright companion (f ~ 0.3) is biased low.
- **Wavelength-scale nuisance** (Stage 6d). `OIData.with_wavelength_scale`
  evaluates models at scale·λ + offset, and the noise terms `wavel_scale` and
  `wavel_offset` fit or sample it (e.g. `Normal(1, 2e-4)` for GRAVITY).

- **Calibration gains correlated across channels** (Stage 6d).
  `OIData.with_gains(telescope=, baseline=, chromatic=, modes=)` adds gains
  on log |V| per frame: per telescope, per baseline, a chromatic coherence
  loss, or supplied modes such as a calibrator PCA's. The likelihood
  marginalises them analytically (`virgil.gains`), and their widths can be
  fitted or sampled with the noise terms `vis_gain_telescope`,
  `vis_gain_baseline`, `vis_gain_chromatic` and `vis_gain_modes`.
  `OIData.stations` holds each sample's station pair, read from `STA_INDEX`.

- **Line and node spectra (Stage 6a, spectra).** `GaussianLine` and
  `LorentzianLine` (amplitude is the peak flux; negative for absorption),
  `Nodes` (linear or natural cubic spline through free fluxes, constant or a
  fixed `outside` value beyond the nodes, e.g. `outside=0.0` for an excess)
  and `Sum` (named parts, so continuum plus lines is one flux). Every
  spectrum's reference flux is its value at `wavel0`, and positivity is
  checked on the total. `Spectrum()` with no argument evaluates at `wavel0`.
  `SourceModel.total_spectrum(wavel)` (a `System`'s sum over its parts) and
  `OIData.select(wavel_min, wavel_max)`. `Tabulated` is deprecated in
  favour of `Nodes` (a `DeprecationWarning`; behaviour unchanged). Removal is
  planned for 0.4.0.

- **`numpyro_model(..., likelihoods=[...])`.** The extra data terms that `fit`
  takes (`PositionData.term`, `RVData.term`, or a callable returning whitened
  residuals) can now be sampled: term `i` is added as the site
  `likelihood_<i>`, and `data_obj` may be `()`. Only `PositionData` and
  `RVData` terms are fully normalised Gaussian log densities (like the OIData
  terms); a plain callable of whitened residuals adds `-0.5 * sum(r**2)` only.

- **`TruncatedCone`.** A thin, optically thin conical shell truncated near its
  apex (e.g. the dust cone of a colliding-wind binary), with an analytic
  visibility (a stack of projected rings), a tilt out of the sky plane, an
  optional elliptical cross-section, and a rendered image that matches its
  visibilities.
- **`EllipticalLimbDarkenedDisk(diam, ratio, pa, u)`.** A limb-darkened disk
  with an elliptical outline (minor/major axis `ratio`, major axis at `pa`
  North to East as for `EllipticalGaussian`), with the polynomial law and
  `u` of `LimbDarkenedDisk` and analytic visibilities.
- **Short-arc orbits.** `StateVectorOrbit` parameterises an orbit by the
  relative position and velocity at `t_ref` and the gravitational parameter,
  which a short arc constrains well where the elements are degenerate; it
  converts exactly to a `KeplerOrbit` (`to_kepler`, `from_kepler`).
- **Radial velocities and masses.** `RVData` fits radial velocities of either
  star (with the mass ratio, systemic velocity and distance), which fix the
  node that positions leave ambiguous by 180°; `total_mass` and `distance_pc`
  convert between them by Kepler's third law; `AxialVonMises` is a prior on an
  angle known only modulo 180°. `fit(..., likelihoods=[...])` adds such terms
  (`PositionData.term`, `RVData.term`) to a fit, with or without visibilities.

- **Orbit example.** `notebooks/mwe/mwe_orbit.ipynb`: a companion and an
  attached disc observed on six VLTI nights, recovered through per-night
  positions, Thiele–Innes starting orbits and a joint fit to all the
  visibilities. `coverage.vlti_oidata(nights_mjd=...)` gives synthetic
  coverage with times.

- **Simulation.** `virgil.simulate.simulate(scene, template)` observes a scene
  with a template's sampling, errors and times (each sample at its own time
  for a moving scene, with `shift_days` to move the epochs), and
  `bias_test` fits a model to many noise draws to show biases and spreads.

- **Times and frames.** `OIData` keeps each sample's time (`mjd`, stored as
  `dt` days since a float64 `t_ref`) and exposure (`frame`) from OIFITS, and
  dict input may give `mjd` and `frame`. A frame is the baselines that
  closure phases tie together; by default all its samples get the frame's
  mean time (`read_oifits(frame_mjd="row")` keeps each row's).
  `OIData.epochs(gap_days=0.5)` labels nights, and `split_by_epoch()` returns
  one `OIData` per night. This is the groundwork for orbits and per-frame
  calibration terms.
- **Orbits.** `KeplerOrbit` (period, time of periastron, eccentricity,
  inclination, the secondary's ω, the receding node's Ω, angular semimajor
  axis) gives the secondary's `relative` position and exact
  `relative_velocity` in virgil's sky conventions, and `ThieleInnesOrbit` the
  linear form used for starting orbits, with converters between them and to
  jaxoplanet, which solves Kepler's equation (the new `[orbits]` extra).
  `PositionData` holds measured positions with their covariances (or
  separations and position angles), and `starting_orbits` finds good
  starting orbits for them by an exact Thiele–Innes least-squares solve on a
  grid of period, eccentricity and time of periastron.
- **Scenes that move.** `SourceModel.at(mjd)` gives a model at a time, and
  `Attached(component, orbit, anchor, bind, offsets)` places a component on a
  binary's orbit and binds its angles to the binary frame (line of centres,
  line of nodes, inclination, the side facing the primary). `OIData.model`
  evaluates a time-dependent model at each sample's own time; static models
  keep their fast path.
- **A model per dataset in sampling.** `numpyro_model`, like `fit`, accepts
  a model function returning a list of models, one per dataset, sharing
  parameters (e.g. a binary at several epochs with one flux ratio).
  Regularisers act on the first model.

### Changed

- **API consistency before 0.3.0.**
  - `linear_flux_grid(prior=)` takes numpyro's `dist.LogUniform(low, high)`
    and `dist.Normal(mean, sd)` (scalar parameters), the same classes as
    `fit` and `noise=`, so passing the numpyro `LogUniform` no longer raises
    `TypeError`. The `virgil.grid_fit.LogUniform` and `Gaussian` named tuples
    still work. A bare `(mean, sd)` tuple, whose deprecation path never
    shipped, is now a `TypeError` with a message saying so.
  - `RVData.term(marginalise_offsets=True)` raises `ValueError` instead of
    silently using N(0, 1000²) km/s: state the zero-point prior as
    `(mean, sd)`.
  - `flux_param` and `batch_size` are keyword-only in `likelihood_grid`,
    `optimized_likelihood_grid`, `optimized_flux_grid`, `linear_flux_grid`
    and `laplace_flux_uncertainty_grid` (in the last, after `flux`).
    The first four are from 0.2.0, so a positional `flux_param` or
    `batch_size` there now raises `TypeError`.
  - `injection_recovery(draw_batch > 1)` divides the default grid
    `batch_size` by `draw_batch` and caps `draw_batch` at that default,
    so the working set stays within the documented bound, and by default
    the last chunk is compiled at its own length instead of being padded
    with discarded draws (an explicit `chunk_size` still pads, to reuse one
    compilation).

- **Frames are grouped by `MJD` and `TIME` (#203).** OIFITS v1 gives each row
  a `TIME` (UTC seconds) and an `MJD`, and some writers (OYSTER, the 2004
  Interferometry Beauty Contest files) set `MJD` to the night's date and put
  the snapshot in `TIME`. The reader grouped frames, and matched
  closure-phase legs to baselines, by `MJD` alone, so every snapshot of such
  a file became one frame and closure phases from different snapshots were
  whitened as one correlated group. Rows are now one frame, and a leg matches
  a baseline row, only if `MJD` and `TIME` both agree within the existing
  tolerance. **Behaviour change:** for the 2004 contest files the independent
  closure phases go from 10/130 to 130/130, and every χ² from them changes.
  Files with a constant `TIME` group as before, and epochs and model times
  still come from `MJD`.
- **`fit` optimises each prior in its flat coordinate.** A prior that is
  uniform in some coordinate of its parameter, `LogUniform(a, b)` in
  `log x`, or any prior with a `flat_coordinate()` method (isotropic
  inclinations in `cos i`, latitudes in `sin(lat)`), is fitted in that
  coordinate, where it is constant and adds nothing to the loss. So
  Levenberg–Marquardt now runs, and is chosen automatically, with these
  Jeffreys priors (it used to fall back to L-BFGS, and `method="lm"`
  raised), and `gauss_newton_mass` accepts them. **Behaviour change:** a
  fit with a `LogUniform` prior (on a parameter or a `noise=` term) is now
  the maximum of the likelihood inside the prior's range, the MAP in
  `log x`; before, it was the mode of likelihood × `1/x` in `x`, which
  pulled scales towards small values. Uniform, Normal and other priors are
  unchanged. See "Priors and the MAP" in the conventions.
- `starting_image`'s internal fit uses `LogUniform` priors on the envelope
  width and flux (flux bounds 1e-4 to 100), so the starting point may differ
  slightly.

- **One home for analytic marginalisation of linear parameters.**
  `virgil._linear` holds the shared algebra: `LinearMarginal(design,
  prior_mean, prior_sd | prior_cov, method)`, the successive rank-one and
  dense-Cholesky whitenings, and the conditional posterior. The gains,
  closure-phase offsets, VISPHI continuum terms, flux grey scales and RV
  zero points use it.
- **The OI_FLUX / correlated-flux grey-scale prior is stated, not taken from
  the data.** `with_flux_scale(scale=(mean, sd))` gives it in the
  data's units. It is required for `"flux"` and
  `"corrflux"`, and the likelihood raises until it is given; `"nflux"`
  defaults to `(1, 0.1)`. The Gaussian is documented as a proposal for the
  Jeffreys 1/k prior.

- **`TruncatedCone.n_rings` guidance.** The docstring now states the measured
  `1 / n_rings**2` error scale (about 8e-4 in |V| at the default 32 for a
  13.8 mas cone), and recommends doubling `n_rings` and checking Δχ² at the
  best fit; well-measured data may need 64 or more. A convergence test was
  added.

### Fixed

- **`clean` no longer hangs when `nnls` gives up in a major cycle.** The
  bounded least-squares fallback now uses BVLS: scipy's default TRF solver
  can loop forever in its line search once the step underflows, which stalled
  the py3.11 lowest-dependency CI job on scipy 1.13.
- **Contrast limits: one search, from below.** Found in the pre-0.3.0 review
  (`design/codebase_review_2026-10b.md`, B1, B2, S2 and S5).
  - `injection_limits(flux_bounds=None)` returned 1000, the top of its search
    range, at every position whose limit was above about 1e-3. For a
    normalised scene the significance falls again once the companion
    outshines the primary, and the bisection ended at that top. Both limit
    functions now find the *first* crossing from below: they step up by
    quarter decades (CANDID steps by 1.4) and then bisect the last step in
    log flux (one shared helper, `_grid.first_crossing`). Unbounded results
    now match bounded ones, for model classes and for `System` templates.
  - `injection_limits` raised `TypeError` on data with extra observables
    (T3AMP, VISAMP, VISPHI, OI_FLUX). The companion's signal is now injected
    into every block of the data vector, extras included. The companion
    model's chi-squared on the injected data is computed in full, since the
    OI_FLUX blocks and gains whiten with the model's own prediction.
  - Both functions raise a `ValueError` up front for a `sigma` beyond what
    `nsigma` can represent in the float type (about 12.95 in float32, 37 in
    float64). Before, `absil_limits` returned about 1e37 or silently clipped,
    with a stale "optimizer did not converge" warning. The new warnings say
    what happened: limits clipped to `flux_bounds`, or no crossing within 40
    decades of the start.
  - `flux_bounds` now means the same in both functions: the range searched,
    upward from its lower end, with limits outside it set to the nearer
    bound and a `RuntimeWarning`. With `flux_bounds=None`, the search starts
    at the flux axis's smallest positive value, and only then must the axis
    have one. `absil_limits` no longer evaluates the whole positions × fluxes
    loss grid to choose a start. Results agree with the old ones to about
    1e-4 relative or better.
  - `flux_param`, `flux_bounds` and `batch_size` are keyword-only in
    `absil_limits` and `injection_limits`, as in `grid_fit` and `detection`.

- **Orbits: GM☉ in `total_mass`, Ω range, face-on inclination (F14–F16).**
  `total_mass` and `distance_pc` now use Kepler's third law with the IAU 2015
  nominal GM☉, au and the 86400 s day instead of a³/P² with P in Julian years
  (masses were 3.8e-5 low). `ThieleInnesOrbit.to_kepler` no longer returns
  Ω = 180° exactly: Ω is in [0°, 180°) with ω paired to keep the sky orbit.
  `StateVectorOrbit.to_kepler` computes i, Ω and ω by atan2 from the orbit
  normal, so nearly face-on orbits keep their inclination to float64
  precision. `orientation_priors(..., inclination=True)` also returns
  `IsotropicInclination` under `"inc"`: the full Haar orientation prior in
  one call (default off, so existing callers are unchanged).

- **`log_evidence` with calibration gains or closure offsets** now includes
  the likelihood's model-dependent normalisation, ½ log det of the
  marginalised nuisances' covariance factor (and, for extra observable
  blocks, their effective-error normaliser), exactly as `model_loglike`
  evaluates it. Before, it used the whitened χ² alone, so wider gains
  always looked better. Values for plain data are unchanged; the evidence
  still omits the data-only normalisation (`-Σ log σ - ½ n log 2π`).

- **`log_evidence` and `clean` on high signal-to-noise data (#214).** The
  evidence takes `log det(I + JᵀJ)` from the singular values of the Jacobian,
  instead of a Cholesky factor of `I + J Jᵀ`, which was numerically
  indefinite and raised `LinAlgError` when `J Jᵀ` is rank deficient;
  `classic_maxent`'s curvature and `error_scale`'s λ use the squared singular
  values too. The CLEAN major-cycle refit survives an `nnls` failure: it
  gets a larger iteration limit and falls back to bounded `lsq_linear`, so
  no-base CLEAN runs on the 2004 contest data instead of raising.
- **Components build under `jax.jit` from concrete shape parameters.**
  `TruncatedCone` validated `tilt` with a `jax.numpy` call on the concrete
  array, which inside `jit` became a tracer and raised
  `TracerBoolConversionError` when a fit's model function built a cone from
  fixed shapes and a traced flux. The check now uses NumPy. The other
  concrete checks (components, spectra, orbits) were audited and need no
  change; a regression test builds each checked component inside `jit`.
  Found in a real-data OzSTAR run.
- **`absil_limits` with a far-off or single-value flux axis.** The
  significance saturates (about 37 sigma in float64) for bright companions,
  so starting the optimizer on such a flux, e.g. `flux=[0.01]`, gave a flat
  loss and returned the starting flux with a non-convergence warning. The
  flux axis now only gives a rough starting point: the limit is bracketed by
  decades and bisected in log flux, replacing the BFGS search, so the result
  no longer depends on the axis.
- `gauss_newton_mass(model, priors, (), values)` no longer raises a
  `ValueError` on an empty residual list: with `data=()` the curvature comes
  from the priors alone, as `fit` and `numpyro_model` already allow.

### Removed

- `examples/elr_pavo/`, the PAVO re-analysis scripts, moved to the private paper repository; the golden-fixture generator is now `scripts/make_elr_golden.py`.

### Docs

- **`GravityDarkenedStar` docstring: "Choosing `n_lat`"** gives the
  measured mesh error from an independent ELR11 reference (virgil-validation),
  the second-order convergence, and the doubling check (|Δχ²| ≳ 1 per
  dataset).
- **"Detection ROC curves" rewritten on `injection_recovery`** (Binaries,
  `notebooks/detection_roc.ipynb`). A candidate companion in a simulated
  NIRISS AMI observation, one Monte Carlo call (10⁴ null and 1280 injected
  searches), the look-elsewhere effect on the null distribution, the
  empirical threshold and FAP to quote for a detection, ROC curves of the
  three statistics, the completeness map and 50%/90% contrast curves to quote
  for a non-detection against Absil and Ruffio limits, and what wrong error
  bars do to a Gaussian null and how `rescale_errors` and the bootstrap fix
  it.
- **New tutorial: "Orbits from interferometric data"**, a joint fit to
  the interferometric data (Binaries, `notebooks/orbit_fitting.ipynb`).
  One `KeplerOrbit` model is fitted to the V² and closure phases of all
  eight simulated VLTI epochs at once, each sample at its own time, with
  per-epoch V² and closure-phase error scales drawn from fitted log-normal
  populations (`hierarchical_scales`). Priors are Jeffreys throughout
  (`IsotropicInclination`, `orientation_priors` and `AngleVector` angles,
  log-uniform scales, interim uniform e), with a no-data check.
  Initialisation is kept separate from the inference: coarse per-epoch grids
  seed `starting_orbits`, the candidates are ranked by the joint likelihood
  of all the data, and the best four are refined with `fit`. Outputs are NUTS
  diagnostics, a corner plot, an ensemble of posterior orbits on the sky with
  the implied per-epoch positions, posterior-predictive closure-phase
  checks, the inferred calibration, separation and position angle in time,
  and a comparison with the two-step posterior.
- **`plotting.plot_orbit_ensemble`.** Draws a batch of `KeplerOrbit`s on the
  sky (East left, North up) as thin lines, one period each, with measured
  `PositionData` positions coloured by epoch with their error ellipses, a
  reference orbit and the primary.
- **Conventions.** Dropped the stale "Not yet in this version" note from the
  orbit conventions: `virgil.orbits` is on main.
- The `virgil._linear` API page is labelled internal.
- Docs pages no longer point to the internal design notes.
- Docstrings no longer point to the internal design notes.
- **Closure-phase whitening** (`virgil._closure`): the docstring states that the correlation, built from the per-triangle errors, is exact only when the baseline phase errors are equal; with unequal errors the whitened χ² comes out slightly low (about 2% with one baseline three times noisier). OIFITS carries no per-baseline phase errors.

## 0.2.0 (2026-10-03)

### Renamed: drpangloss is now virgil

- The import package is `virgil` (`import virgil`) and the PyPI distribution
  is `virgil-astro` (`pip install virgil-astro`). `pip install virgil` installs
  an unrelated package.
- The GitHub repository is `benjaminpope/virgil` and the docs are at
  <https://benjaminpope.github.io/virgil/>.
- A final `drpangloss` 0.2.0 on PyPI depends on `virgil-astro` and forwards
  to it, with a `FutureWarning` on import. Replace `import drpangloss` with
  `import virgil` and `drpangloss.x` with `virgil.x`; nothing else was renamed.
- There is no `nufft` extra: the NUFFT backend was removed (see below).

### Changed: closure phases (every four-telescope chi-squared changes)

- Closure phases of triangles that share a baseline are correlated. For four or
  more telescopes, `virgil` now keeps only the independent combinations of the
  closure phases of each frame and channel (three of the four triangles for
  four telescopes) and whitens them with the covariance of Kammerer et al.
  (2020, A&A 644, A110): the reported errors on the diagonal and correlations
  of +/-1/3 between triangles that share a baseline. Before, all triangles were
  treated as independent, which counted the closure phases 4/3 times for four
  telescopes.
- As a result every chi-squared, likelihood, evidence, grid and fit that uses
  four or more telescopes changes, and the number of closure phases in the
  likelihood (`OIData.n_independent`) is smaller than the number stored.
  Three-telescope data (e.g. JWST/NIRISS AMI with a single triangle per
  frame) are unchanged. Numbers from earlier versions, including some logged in
  the design notes, are marked "pre-6.0" and have not been re-measured.
- Unprojected closure phases enter the likelihood as the chord 2 sin(delta/2)
  over sigma (a von Mises likelihood for independent phases), which is
  smooth across +/-pi.

### Added

- **Readers.** `read_oifits(insname=...)` selects an instrument's tables. A
  GRAVITY file holding both fringe-tracker (FT) and science (SC) tables must
  be read with `insname=`: the two are never merged. `PHITYP` is checked, so a
  differential VISPHI is never read as an absolute phase. Closure phases are
  matched to visibilities of the same exposure, within twice the longest
  `INT_TIME`, since GRAVITY's pipeline averages different frames for each
  table (before, 50 of 136 reads of archival GRAVITY files failed).
- **Image reconstruction.** An `Image` component (a pixel map in a `System`,
  with an exact DFT, and an exact matrix Fourier transform on the uv lattices of
  AMIGO DISCO data), `fit` (Levenberg-Marquardt, L-BFGS or Adam, in float64 by
  default), the regularisers `MaxEntropy`, `TSV`, `TV` and `Centroid`,
  `l_curve` with corner, discrepancy and classic MaxEnt weights,
  `GaussianField` Gaussian-process pixels, `log_evidence`, `error_scale`
  (MacKay's re-estimate of the error bars), `dirty_image`, `beam`,
  `convolve_beam`, `diagnose`, and a Gauss-Newton mass matrix for NUTS. See
  the imaging tutorials in the docs.
- **Synthetic data.** `virgil.coverage` (`ami_grid_record`, `nrm_oidata`,
  `vlti_oidata`, `mask_transfer`) and `virgil.scenes` for simulations and tests.
- **Chromatic scenes.** Wavelength-dependent fluxes (`PowerLaw`, `BlackBody`)
  following SPARCO, a `Resolved` background, and multi-channel and multi-file
  `OIData`.
- **Source models.** Composable components and `System`, explicit `flux`
  parameters, `FlaredDisk` (Blakely et al.), `ModulatedGaussianRim`,
  `UniformDisk`, `EllipticalGaussian` and `GaussianArc`, an anisotropic
  `GaussianField`, fitted error inflation (`noise=`), the rapid-rotator
  `GravityDarkenedStar` (ported from Shashank Dholakia's ELR model, grey and
  chromatic), and an interface to harmonix for spotted stars.
- **Packaging.** Bessel functions now come from the separate `jaxbessel`
  package. Releases are published to PyPI from GitHub releases.

### Fixed and improved

- One likelihood (`likelihood.whitened_residuals`) for fitting, grids, limits
  and sampling, with normalised Gaussian log-likelihoods.
- The image coordinate convention (East left, North up, position angle North
  to East) is applied everywhere, including a position-angle bug in
  `BinaryModelAngular` and the orientation of plots.
- Maximum-entropy L-BFGS fits no longer collapse onto a few pixels, and
  stalled image fits are capped.
- Solvers and curvature functions compile once rather than on every call, grid
  tools size their batches by data size, and the test suite is faster.

### Fixed after the October 2026 codebase review

- The rim's non-negativity check (`ModulatedGaussianRim`, `is_physical`) no
  longer passes negative brightness when the top azimuthal order is zero.
- Independent phases use the exact von Mises normaliser, so fitted phase
  errors are no longer biased at large sigma (a true 1.8 rad used to fit as
  1.3 rad). Small-sigma likelihoods are unchanged.
- Converting V^2 to amplitudes or log-amplitudes floors the data at their own
  error, so points near or below zero no longer get enormous or tiny errors.
- Levenberg-Marquardt and L-BFGS in float32 converge instead of running to
  `max_steps`: the gradient tolerance is floored at sqrt(eps) of the starting
  gradient.
- `laplace_cov` and `fisher` compute in float64, like `fit`.
- `error_scale` solves MacKay's self-consistent equation instead of
  evaluating it at beta = 1; `log_evidence`, `error_scale` and
  `classic_maxent` accept a `FitResult` and refuse fits with fitted noise
  terms or per-dataset model lists, which they cannot handle.
- Index arrays cached under one x64 mode no longer break the other on JAX
  0.10 (closure-phase noise and the gravity-darkened star mesh).
- Corrected citations: MacKay (1992) eq. 4.10; classic MaxEnt is Gull (1989)
  with Skilling (1989).
- `virgil.__version__`; true minimum dependency versions (`jax>=0.8`), tested
  in CI; pandas, ChainConsumer and astroquery moved to the `plots` and
  `legacy` extras; CI on macOS with Python 3.11, Python 3.13, the lowest
  versions, and a wheel build.

### Removed

- The experimental NUFFT backend (`backend="nufft"`, the `nufft` extra): on a
  GPU it was slower than the DFT. Its code is in the history of PR #71.
- `amigo.simulated_disco_record`, replaced by `coverage.ami_grid_record`.
- Before the first release, these names were removed from the public API:
  `GaussianDiskModel` (use `System(star=PointSource(), disk=GaussianDisk(...))`),
  `cvis_gaussian_disk`, `cvis_binary_angular` (now private),
  `loglike_nosignal`, `fisher_matrix` and `observed_information` (use
  `fisher` and `inference.hessian_matrix`), `plotting.plot_trace_panels`,
  `plotting.plot_recovery_residuals`, the `legacy.savefits` module and
  `legacy.oifits_implaneia.load_oifits`.
- `Tabulated` and `load_oi_data` are no longer top-level names: they are
  `virgil.spectra.Tabulated` (provisional, to be replaced) and
  `virgil.amigo.load_oi_data`. `pixel_offsets` and `HarmonixModel` are new
  top-level names.
- The unused `termcolor` dependency and `sampling` extra, and the unused
  `data/chi2_ppf*.npy` tables.
