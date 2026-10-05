# Changelog

All notable changes to this project are recorded here, in the style of
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/). The project follows
[semantic versioning](https://semver.org/), with the usual caveat that
anything before 1.0 may change between minor versions.

## Unreleased

The release commit will set the version to 0.3.0 and date this section; until
then the notes below are under "Unreleased". Everything listed as new was
added after 0.2.0, which is on PyPI.

The release commit must set `version = "0.3.0"` in `pyproject.toml` and refresh
`uv.lock`, which records the editable virgil-astro version: run `uv lock` in a
worktree, never in `~/code/drpangloss`.

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
- **`Tabulated` is deprecated** in favour of `virgil.spectra.Nodes`: it
  now raises a `DeprecationWarning`, behaviour is unchanged, and removal is
  planned for 0.4.0.

### Added

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
  better than 1e-4. A Gaussian prior is `Gaussian(mean, sd)` (a
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

### Docs

- **New tutorial: "Orbits from interferometric epochs"** (Binaries,
  `notebooks/orbit_fitting.ipynb`). Eight epochs of simulated VLTI
  (UT) V² and closure phases of a three-year binary: per-epoch astrometry
  with a grid, a fit and the Laplace covariance into `PositionData`,
  Thiele–Innes starting orbits, and a NUTS posterior under Jeffreys priors
  (log-uniform P and a, uniform cos i, ω, Ω and phase as 2-vector
  directions, uniform e) with a no-data prior check, a corner plot, and an
  ensemble of posterior orbits on the sky and in time.
- **`plotting.plot_orbit_ensemble`.** Draws a batch of `KeplerOrbit`s on the
  sky (East left, North up) as thin lines, one period each, with measured
  `PositionData` positions coloured by epoch with their error ellipses, a
  reference orbit and the primary.
- **Conventions.** Dropped the stale "Not yet in this version" note from the
  orbit conventions: `virgil.orbits` is on main.

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
