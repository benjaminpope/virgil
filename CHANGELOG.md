# Changelog

All notable changes to this project are recorded here, in the style of
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/). The project follows
[semantic versioning](https://semver.org/), with the usual caveat that
anything before 1.0 may change between minor versions.

<!-- towncrier release notes start -->

## 0.4.2 (2026-10-24)

### Changes to default behaviour

- `CalibrationNuisance(alpha=1)` (a non-zero integer) raises. It used to fit α
  (`bool(1)`), and a float now fixes it. Pass `alpha=True` to fit or
  `alpha=1.0` to fix; `alpha=0` still gives α = 0. ([#516](https://github.com/benjaminpope/virgil/pull/516))

### Added

- **`virgil.radio_toeplitz`** (provisional): `ComplexVisData.toeplitz`
  precomputes the dirty beams and images of a one-`Image` radio fit, after
  Fessler et al. (2005). Its `term` gives the loss, gradient, profiled `ŝ` and
  `derived` of `ComplexVisData.term` at a cost independent of the number of
  visibilities. See [Toeplitz χ²](docs/radio_interferometry.md#toeplitz-chi2).
  - Supported: per-observation gains, flux, taper, smoothing, a mask, and
    fixed, fitted or profiled error scales; float64 only. The template's
    fixed `dra`, `ddec` (such as a half-pixel map convention) are folded into
    the visibilities at the precompute.
  - Refused: antenna gains, a fitted spectral index and other components.
  - `crosscheck` compares it with the direct term (about 1e-13 of the loss).
    `rebuild` repeats the precompute for new weights without recompiling.
  - Extension, opt-in: `offsets="spread"` fits the per-observation offsets
    by evaluating the stored nufftax spread grids (nufftax ≥ 0.6.1) at
    shifted positions. This is a type-3 evaluation (Lee & Greengard 2005;
    Barnett et al. 2019); we know of no prior use of this combination. It
    refuses baselines past the kernel's Nyquist margin. Tests:
    `test_matches_direct_term`, `test_matches_direct_term_at_scale`,
    `test_refuses_what_it_cannot_do`, `test_runtime_checks` and
    `test_spread_grid_against_nufftax_and_direct_sums`.

  ([#516](https://github.com/benjaminpope/virgil/pull/516))
- `fit` accepts likelihood terms with a scalar χ² (`chi2_and_log_norm`, no
  residual vector). L-BFGS and Adam use them; LM and the Gauss–Newton
  covariance raise `NoLeastSquaresForm`; `precondition="diag"` takes their
  Gauss–Newton diagonal through `curvature_params` and `curvature_loss`. ([#516](https://github.com/benjaminpope/virgil/pull/516))
- `ComplexVisData.chi2_by_ddid` takes `chunk_size` (that of `term`). ([#516](https://github.com/benjaminpope/virgil/pull/516))
- `CalibrationNuisance(alpha=<float>)` fixes the spectral index at that value. ([#516](https://github.com/benjaminpope/virgil/pull/516))
- The README development-carbon badge is now a shields.io endpoint badge reading
  `generated/dev_carbon_badge.json` on the docs site, so the PyPI README shows
  the current figure, not the one frozen at release. `scripts/dev_carbon.py`
  writes both. ([#529](https://github.com/benjaminpope/virgil/pull/529))
- CI fails a PR that adds a private function called exactly once in the package
  (`scripts/code_shape.py --since`, rule 8 in AGENTS.md; exemptions: `# deliberate:`
  comment, jit kernels, custom derivatives, callbacks, names used in tests). The size
  comment also prints before/after code-shape metrics (`scripts/code_shape_metrics.py`). ([#537](https://github.com/benjaminpope/virgil/pull/537))

### Changed

- `MultipoleField` is single-valued at the disk centre when `r_min_mas=0` (the default; #517): order `m` of the field is scaled by `tanh(r / spacing)^|m|` (the `r^|m|` regularity of a smooth 2D field), so only `m = 0` survives at `r = 0`. This changes the default prior: within about five node spacings of the centre the `m >= 1` variance is below `sigma**2`, so MAP fits and evidences move; farther out, and for `r_min_mas > 0`, it is unchanged. ([#518](https://github.com/benjaminpope/virgil/pull/518))
- The release checklist no longer requires a mutation-testing kill rate. ([#523](https://github.com/benjaminpope/virgil/pull/523))
- `fit(method="adam")` now runs in jitted chunks of 200 steps (same trajectory, step for step), so `time_limit` and `progress` apply to it, and it reports convergence: `info["converged"]` is `True` if the `ftol_abs` plateau stop fired or `grad_norm <= gtol`, else `False` with `info["stop"]` `"limit"`, `"time"` or `"non-finite"` (it was `None`, so an Adam fit that hits `max_steps` now warns). New keywords: `adam_eps` (`optax.adam`'s `eps`), `ftol_abs` and `ftol_window` (stop when the loss fell by less than `ftol_abs` over the last `ftol_window` steps); `info["loss_history"]` is the unscaled loss before each Adam step. `ftol_window` below 1 raises `ValueError`. ([#534](https://github.com/benjaminpope/virgil/pull/534))
- Radio innovations register audited (virgil#531): B5 and B7 updated to what is on `main`, a defaults audit
  and the D13 finding added, six references added to the provenance file with `Source:` lines. The docstrings
  of `fit(precondition=...)`, `gauss_newton_diag` and `MaxEntropy.metric_rows` now give the entropy metric's
  diagonal as `weight * b * (1 - b)`, as the code computes it; no behaviour changes. ([#538](https://github.com/benjaminpope/virgil/pull/538))
- The disk, image and quicklook pipeline modules lose a dozen single-use private helpers and a few docstring lines; no behaviour change. ([#540](https://github.com/benjaminpope/virgil/pull/540))
- `ComplexVisData.term` takes `chi2="auto"|"direct"|"toeplitz"` (default `"auto"`). `fit` then uses the exact Toeplitz (dirty-beam) χ² where it applies: one fixed `Image`, a fixed `alpha`, `offsets=False`, no `antenna_gains`, float64, `method="lbfgs"` or `"adam"`, and at least `toeplitz_min_points` points and `toeplitz_points_per_pixel` per pixel and ddid (`set_ft_defaults`). Otherwise the direct term is used; `"toeplitz"` raises a `ValueError` naming the unmet condition. The beams take about 24 N² bytes of device memory per ddid; auto stays direct above `toeplitz_max_bytes` (4 GB), and `chi2="direct"` opts out. `FitResult.info["likelihood_forms"]` says what ran (and why auto fell back), and a Toeplitz term's `info["chi2"]` is re-evaluated directly at the result. `numpyro_model` and the training fits of `cross_validate` stay direct. ([#541](https://github.com/benjaminpope/virgil/pull/541))
- Code shape (#536, batch I): `oifits`, `oidata`, `amigo` and `legacy` inline 23 single-use private helpers in `oifits.py` and 3 elsewhere, and trim long docstrings to the calling contract. Behaviour and OIFITS round trips are unchanged. ([#542](https://github.com/benjaminpope/virgil/pull/542))
- Code shape: nine refactor batches ([#540](https://github.com/benjaminpope/virgil/pull/540), [#542](https://github.com/benjaminpope/virgil/pull/542), [#543](https://github.com/benjaminpope/virgil/pull/543), [#544](https://github.com/benjaminpope/virgil/pull/544), [#545](https://github.com/benjaminpope/virgil/pull/545), [#546](https://github.com/benjaminpope/virgil/pull/546), [#547](https://github.com/benjaminpope/virgil/pull/547), [#549](https://github.com/benjaminpope/virgil/pull/549), [#550](https://github.com/benjaminpope/virgil/pull/550)) inline single-use private helpers and trim docstrings to the calling contract, with no behaviour change. Measured on `src/virgil` against v0.4.1: private helpers called from one place 418 → 225 (−46%), docstring lines 14,040 → 13,647 (−3%), docstrings over 40 lines 30 → 22, none over 60; source lines 56,790 → 57,813 (+2%, the radio and Adam work).

### Fixed

- `fit` gauge-fixes the log-brightness of a free-pixel `Image`. A constant
  added to it changes no image, and LM's QR solve drifted along that direction
  by rounding noise (to 4e5 or 3.6e10, depending on platform). The fitted
  `log_brightness` is now `log` of the pixel fluxes (`Σ exp = 1`), the same
  on every platform; images, χ² and penalties are unchanged to rounding.
  Pixels whose optimum is zero flux still fit towards `-inf`. ([#519](https://github.com/benjaminpope/virgil/pull/519))
- `test_the_fitted_log_brightness_is_gauge_fixed_for_every_method` holds `logsumexp` to 1e-9 only in float64, and to a few float32 eps otherwise (one ulp was 2.4e-7). ([#539](https://github.com/benjaminpope/virgil/pull/539))
- `test_fit_reaches_the_direct_optimum` no longer fails on some CI runner CPUs
  (#528): the fitted `cal.sigma` of two optimiser paths is compared at 1e-6,
  and the two terms' `sigma` at the same parameters at 1e-12. ([#539](https://github.com/benjaminpope/virgil/pull/539))
- The `precondition="diag"` example in `docs/conventions.md` is self-contained. ([#539](https://github.com/benjaminpope/virgil/pull/539))
- `radio_clean.clean` diverged on sparse-coverage problems (#551): the minor cycle used a PSF the size of the image, so its window was clipped at half the image and sidelobes beyond it were never subtracted. The minor cycle's PSF is now computed out to `psfpatch` or the far edge of the image from any component, as in CASA; results on well-behaved problems are unchanged (to 1e-9). `clean` also ends a minor cycle at a 10 per cent rise above its running minimum (per scale; `history[i]["minor_stop"]` records why) and the run with `stop="diverged"` at a relative rise above 3.0 of the whole-image peak residual over the previous major cycle or the minimum so far, as CASA's iteration control does. ([#553](https://github.com/benjaminpope/virgil/pull/553))

### Docs

- New simulated radio imaging tutorial (`docs/radio_tutorial.md`, from `notebooks/radio_tutorial.ipynb`): `ComplexVisData` from arrays, CLEAN start, `fit` of an `Image` with the Toeplitz χ², and diagnostics; a Performance table skeleton in the radio guide. ([#552](https://github.com/benjaminpope/virgil/pull/552))


## 0.4.1 (2026-10-23)

0.4.1 is a consolidation release. It adds no new model families. It makes the
existing code shorter to read, safer by default on aperture-masking and DISCO
data, and able to write what it reads.

### Highlights

- **Write OIFITS from an `OIData`.** `write_oifits(data, path)` round-trips
  through `read_oifits`, closure-only (AMI) data included. The pipelines now
  write `data/processed.oifits` for closure-only data and for data with flagged
  visibilities too; data whose channel widths differ between baselines of one
  channel are skipped, with the reason in `processed_oifits_skipped` (0.4.0
  wrote them with `EFF_BAND = 0`).
- **No more alias-copy detections by default.** On lattice uv coverage (aperture
  masks, DISCO), `BinaryPipeline` and `OrbitPipeline` cap the default search box
  at half an alias cell. An explicit `max_sep_mas` is never capped.
- **`OrbitPipeline` checks its own verdict.** It drops the night with the worst
  χ²/N and refits, and records the result in `modes.robustness`.
- **Better-conditioned fits.** `fit(precondition="diag")` (provisional) and an
  L-BFGS default history of 100, which was 50. Maximum-entropy imaging needs
  30–40% fewer steps.
- **Radio CLEAN** (`virgil.radio_clean`, provisional). This is a first piece;
  the full radio work is planned for 0.4.2.
- **Shorter docstrings and fewer internal functions** (details below), and a
  public [References](https://benjaminpope.github.io/virgil/references/) page
  that lists every cited source.

### Smaller and easier to read

Measured on `src/virgil`, from v0.4.0 to 0.4.1:

| | 0.4.0 | 0.4.1 |
|---|---|---|
| Docstrings over 60 lines | 19 | **0** (CI gate) |
| Docstrings over 40 lines | 63 | 30 |
| Docstring lines in total | 15,503 | 14,040 (−9%) |
| Private helpers called from one place | 456 | 418 (−8%) |
| Source lines in total | 56,804 | 56,790 |

- **Docstrings.**
  - Every public docstring is now 60 lines or fewer, and
    `tests/lint/test_docstring_length.py` enforces it.
  - The longest docstrings (`DiskPipeline`, the other pipelines, `epoch_positions`,
    `imaging.clean`, `fit_orbit_aliases`) were cut to their calling contract.
  - Derivations moved to `docs/orbit_methods.md` and the guides.
  - No caveat that a caller needs was dropped. Each one is either a single line
    in the docstring or a `# deliberate:` comment at the line it explains.
  - Each docstring cites its source in one `Source:` line that links to the
    References page.
- **Private helpers.**
  - About 40 single-use private helpers were inlined into their callers across
    the models, fitting, imaging, plotting, detection, orbit, OIFITS and
    pipeline modules, and duplicate helpers were merged.
  - Results and `summary.json` are unchanged.
  - The source is no longer than in 0.4.0, although the release adds about
    1,400 lines of features (radio CLEAN, the OIFITS writer, preconditioning).
  - The lint size comment now lists any new single-use private function a PR
    adds.

### Changes to default behaviour

- **`CalibrationNuisance` default priors are in natural units.**
  - Gains and error scales use Jeffreys priors; the fit coordinate is `ln x`.
  - Offsets are uniform over `offset_beams` beams (default 3). This box is
    narrower than before, and the bound is now enforced: a fit that stops on
    it is listed in `info["at_bound"]`.
  - Pass `priors(gain=, sigma=, offset=)` to restore the old boxes.
- **L-BFGS history.** `fit(lbfgs_memory=None)` now uses 100 vectors, cut to
  keep the history under 1 GiB, with a warning. LM ignores it.
- **Alias cap.** The default search box is capped on lattice coverage, as
  described in the Highlights. The cap is recorded in
  `summary["search"]["alias_cap_mas"]`.
- **Lazier import.** `import virgil` no longer imports matplotlib.

### Added


- `plot_field_draws`: three draws and an ensemble of cuts (principal axes, disk axes, or the azimuthal harmonics of a `MultipoleField`) for a Gaussian-process image prior or posterior, with an optional per-pixel S/N map. Figures in the imaging parts 3 and 5 and the multipole-disk pages.
- Docs: the API index lists Radio CLEAN, Cross-validation and Short arcs; the fitting page documents `GaussNewtonMass` and its exceptions; the README names the `[radio]` and `[pipeline]` extras; the OIFITS writing sections are merged. Nine references checked against ADS (Högbom, Clark, Schwab, Briggs, Offringa, Quirrenbach, Thiele, Bekas, Hutchinson).
- `to_kepler` (Thiele–Innes and state-vector orbits) runs its maths as one jitted kernel: an eager call takes 0.16 ms (was 0.31) and 0.10 ms (was 0.57); under `jit` + `vmap` it was already about 20 ns per draw, so results are unchanged. The orbit docstrings are now 45 lines or fewer.
- `write_oifits` takes an `OIData` as well as a dictionary of tables, and `read_oifits` reads the file back to the same observables, errors, flags, wavelengths and (u, v), and to the times if there are any. Closure phases without station indices get placeholder stations, so the pipelines now write `data/processed.oifits` for closure-only data (e.g. AMI) and for data with flagged visibilities (flagged closure phases are still skipped); `processed_oifits_skipped` keeps an honest reason for the rest. The channel widths are now written as `EFF_BAND` (they were written as 0; an unknown width is estimated from the channel spacing and marked `BANDEST`, since OIFITS2 needs it positive), and data whose widths differ between baselines of one channel are skipped with that reason. The written file now has `MJD` in `d`, and 32-character `TARGET` and `SPECTYP`, as OIFITS2 requires.
- **`virgil.radio_clean`** (provisional): `clean`, a Cotton–Schwab CLEAN of
  `ComplexVisData` named after CASA's `tclean` (Högbom and multi-scale minor
  cycles with a PSF window and cycle threshold, `niter`, `gain`, `threshold`,
  `nsigma`, `cyclefactor`, masks and a simplified `auto-multithresh`, a fitted
  restoring beam), with `RadioCleanResult` writing FITS and the model as
  visibilities, `fit_beam` and `restore`. `ComplexVisData.dirty_image` and
  `dirty_beam` gain `uvtaper`. It is not `virgil.imaging.clean` (which is
  unchanged) and claims no parity with CASA or WSClean. See the
  [guide](docs/radio_clean.md).
- `fit(..., precondition="diag")` rescales each coordinate by the square root of the diagonal Gauss–Newton curvature in L-BFGS (refreshed every restart round, with `gtol`, `max_step_size` and `info["grad_norm"]` still in the unscaled coordinates, and the softmax gauge of an `Image` projected out) and Adam. Off by default; LM ignores it. `info["precondition_rounds"]` records, per round, the spread of the scaling (`decades`) and the fraction of coordinates at the floor (`floored`). `gauss_newton_diag` returns the diagonal (exact up to 1024 coordinates, else probes). It takes 30–40% fewer L-BFGS steps in total along a maximum-entropy L-curve; it does not make weak-weight fits converge in float32.
- Docs: a public [References](https://benjaminpope.github.io/virgil/references/) page, generated from `references/provenance.toml` by `scripts/sync_references_page.py`; `tests/lint/test_provenance.py` fails on a source cited in `src/` or `docs/` with no entry. A docstring-length gate (`tests/lint/test_docstring_length.py`, 60 lines) and terse docstrings for `aliases`, `epochs`, `orbit_search` and `orbits`; the derivations are in `docs/orbit_methods.md`.
- Docs: terse docstrings for the data and pipeline modules (`oidata`, `observables`, `virgil.pipeline.*`): the pipeline classes' settings are now a table per pipeline in [Pipelines](https://benjaminpope.github.io/virgil/pipeline/).
- `OrbitPipeline` checks that its verdict survives dropping the most outlying
  night (`night_dropped`): the `modes` stage refits the search without the
  night of highest raw χ²/N and compares the orbit-or-not verdict and the
  winning alias band, reported in `modes.robustness` (a second search; skipped
  below 3 nights or with `drop_outlier_night=False`).
- `CalibrationNuisance(sigmas="profile")` removes the error-scale sites and sets
  `ŝ_d² = χ²_d(1) / (2 N_d)` at each evaluation, the exact optimum over `ln s`
  under the Jeffreys prior; `fit` reports it in `info["derived"]`,
  `ComplexVisData.sigma_hat` computes it, and `term(sigma=)` fixes it for scoring. New `offset_beams` argument.
- `write_oifits(OIData)`:
  - Errors, flags, wavelengths, (u, v) and times survive a round trip.
  - Closure phases without station indices get placeholder stations.
  - Channel widths are written as `EFF_BAND`. If they are unknown, they are
    estimated and marked `BANDEST`.
  - It follows the OIFITS2 field rules.
  - `tables=` is now `data`. The old name works with a `FutureWarning` until 0.5.
- `OrbitPipeline` drop-a-night check (`night_dropped`). It is skipped below 3
  nights or with `drop_outlier_night=False`.
- `fit(precondition="diag")` and `gauss_newton_diag` (both provisional). The
  `MaxEntropy` entropy metric and `MaxEntropy.residuals` let LM fit an image
  (float64 only). Preconditioning does not make weak-weight fits converge in
  float32.
- `virgil.radio_clean` (provisional): a Cotton–Schwab CLEAN of
  `ComplexVisData` with Högbom and multi-scale minor cycles, masks, and a
  fitted restoring beam. See the [guide](docs/radio_clean.md).
- `CalibrationNuisance(sigmas="profile")` and `ComplexVisData.sigma_hat`.
- `lattice_alias_limit` works per channel.
- The public References page, generated from `references/provenance.toml`. A
  lint test fails on any cited source that has no entry.

### Fixed

- `fit_orbit_aliases` no longer runs out of memory on data with gains and many
  nights. It accumulates the Hessian in groups of nights that fit
  `memory_budget`, and results agree to rounding.
- `ImproperUniform` priors with an `event_shape` no longer raise in
  `gauss_newton_mass`.

### Other changes

- `OrbitPipeline(epochs, output=...)`: `output` is now keyword-only. Positional
  use still works until 0.5, with a `FutureWarning`.
- `ComplexVisData.dirty_beam` takes `values` before `nuisance` (provisional,
  no shim).
- The orbit classes' `t_ref = 0` warning now says it becomes an error in 0.5.
  `virgil.name` is removed; use `virgil.__version__`.
- `to_kepler` is one jitted kernel, 2–6× faster for an eager call.
- CI is faster: the orbit tests are sharded, timings come from CI runs, and the
  example pages run lighter.

## 0.4.0 (2026-10-22)

Everything listed as new was added after 0.3.0, which is on PyPI.

### Highlights

- **Stability tiers.** virgil now says what may change. Everything public is
  **stable** (a change needs a deprecation cycle) except the features listed
  under [Provisional](#provisional), which may change in any 0.4.x release.
  See the [Stability page](docs/stability.md).
- **One argument order: model before data.** The grid, limit, detection and
  likelihood tools are all `f(model, data, ...)`, as `fit` already was, and
  each argument has one name (`model`, `data`, `grid`). Calls in the 0.3
  order work with a `FutureWarning` until 0.5.
- **Orbit fitting to multi-epoch visibilities** (provisional). `Epochs` and
  `OrbitalBinary` model a binary on its orbit with one snapshot per dataset;
  `epoch_positions`, `start_from_positions`, `score_orbits` and
  `fit_orbit_aliases` find the orbit, including aliased periods, with
  evidences tempered for correlated spectral channels.
- **The `MultipoleDisk` family** (provisional). Inclined, rotated disks built
  from radial profiles (rings, power laws, tails, gaps and Gaussian
  processes), azimuthal multipoles (positive `Harmonics`, `VonMises`),
  height, a spectral index and a residual `MultipoleField`.
- **Pipelines** (provisional). `BinaryPipeline`, `StarPipeline`,
  `DiskPipeline`, `OrbitPipeline` and `ImagingPipeline` run named stages on
  public virgil functions and write a resumable run folder.
- **Imaging that sizes itself.** The image field comes from the data, the
  reconstruction ensemble (`virgil.ensemble`) sweeps it, and
  `gauss_newton_mass` and `variational` (SVI, `virgil.svi`) start NUTS and
  guides from the data's curvature.
- **Faster and steadier.** Orbit scoring, the evidence helpers, `dirty_image`,
  `fit(method="lm")` and several builders compile once or avoid quadratic
  passes; results are unchanged up to rounding.
- **Correctness fixes**: projected correlated closure phases in float32, a
  zero-error sample, `GaussianArc` and `TruncatedCone` quadrature, the
  `SharedFlux` grid, differential-phase wrapping, and parallel NUTS chains
  with gains.

### Migration from 0.3

Each 0.3 call below still works until 0.5 unless noted, with a
`FutureWarning` that names the new call. To find the remaining 0.3-order
calls in your own code, add `"error:.*stops working in virgil
0.5:FutureWarning"` to your pytest `filterwarnings` (as virgil's own tests do),
or run `pytest -W error::FutureWarning`. A message regex given to `-W` on
the command line is matched literally, so it would not catch them.

**Argument order and names**

- `likelihood_grid(data, model, grid)` → `likelihood_grid(model, data=...,
  grid=...)`. Only the model is positional; any positional call other than the
  0.3 order raises a `TypeError` naming the keyword call. The shim recognizes
  data by type and rebinds the arguments.
- `optimized_likelihood_grid`, `optimized_flux_grid`, `linear_flux_grid`,
  `laplace_flux_uncertainty_grid`, `absil_limits`, `injection_limits` and
  `detection_statistics` take `(model, data, grid, ...)`; `gaussian_null`,
  `rescale_errors` and `bootstrap_null` take the null scene first;
  `injection_recovery` is `(model, null_scene, template, grid, key, ...)`.
- `loglike`, `laplace_cov`, `laplace_parameter_uncertainty` and `fisher` are
  `f(values, params, model, data, ...)`; `joint_prediction` and `joint_loglike`
  are `f(params, model, data)`.
- `model_object=`, `model_fn=`, `data_obj=`, `observations=` → `model=` and
  `data=` (`whitened_residuals`, `model_loglike`, `flux_scale_posterior`,
  `numpyro_model`, `posterior_predictive_summary`, `OIData.model`,
  `OIData.with_model`, every `observables.*.predict`, `inflated_errors`,
  `joint_data`, `joint_errors`).
- `samples_dict=` → `grid=` in these functions and in `best_grid_point`,
  `plot_grid_map` and `plot_contrast_curve`.
- `starting_image(data, star, oversample, largest_mas, start, hole_mas)` →
  everything after `data` by keyword (`oversample=`, `largest_mas=`, ...).
  `field_fwhms`, `inner_elements` and `aperture_mas` are keyword-only
  outright.
- `cvis_uniform_disk(..., ud=)` → `diam=`.
- Oxford spelling: `regularisers=` → `regularizers=` (`fit`,
  `numpyro_model`, `diagnose`), `regulariser=` → `regularizer=`
  (`imaging.l_curve`), `marginalise_offsets=` → `marginalize_offsets=`
  (`RVData.term`). The numpyro factor sites are now `regularizer_<i>`.

**Renamed**

- `virgil.inference`: `fisher` → `observed_information` (also
  `virgil.observed_information`; the Hessian of -log L), `gaussian_fisher` →
  `expected_fisher`, `laplace_covariance` → `objective_laplace_covariance`.
  `laplace_cov` keeps its name. `virgil.inference` has an `__all__`.
- `OIData.observable_kind` → `OIData.vis_encoding` (`"split"` or
  `"mixed_log_complex"`), so it is not mistaken for the record key
  `vis_mode`. The old attribute still reads, with a `FutureWarning` outside
  virgil.
- `legacy.oifits_implaneia.GetWavelength`, `Format_STAINDEX_V2`,
  `Format_STAINDEX_T3`, `rad2mas` and the `cp_indices` re-export are
  deprecated (`FutureWarning`; removal in 0.6).

**Results that change for the same call**

- **`GaussianArc` has a different shape.** The weight is now a von Mises arc
  rather than a Gaussian in arc length. Refits of long arcs change (about
  1e-2 in V at a fifth of the semicircle, more for longer arcs); no warning
  is given. See Changed.
- **`TruncatedCone` shifts numerically.** Its default `n_rings` goes from 32
  to 48 and its quadrature is Gauss–Legendre, so a 0.3 fit rerun gives a
  different χ² and best fit.
- **`fit_orbit_aliases` tempers by default** (`temper=True`), so evidences,
  band probabilities and posteriors are not comparable with untempered 0.3
  runs on correlated channels; `temper=False` reproduces them. Its other new
  defaults (`p_range`, `sign_free`, `primary_diam="auto"`, `vis_amp`,
  `skip_flagged`, `flip_flux`) are listed under Changed; `p_range=`,
  `sign_free=False`, `primary_diam=None` or `vis_amp=None` restores the old
  behaviour. Checkpoints of earlier versions are refitted.
- **Image fields and flagged samples.** The default `starting_image` field
  now comes from the data (`image_field`), the ensemble sweeps `(0.75, 1.0,
  1.5)` times it, and `beam`, `fringe_scales`, `nyquist_pixel_scale`,
  `field_of_view` and `Epochs.resolution_mas` ignore flagged samples. Data
  without flags are unchanged.
- **Dense mass limit.** `gauss_newton_mass` returns the diagonal, with a
  `UserWarning`, above 4096 coordinates (was 8192); `variational` with
  `"bnaf"`, `"iaf"`, `"mvn"` or `"laplace"` raises `ValueError` above it. Use
  `laplace_samples` or a diagonal guide such as `autoguide.AutoNormal`.
- **`laplace_samples` with a given `key`** returns different draws (the
  singular vectors' signs differ between host and device SVDs); the
  distribution is the same.
- **`laplace_parameter_uncertainty`** computes in float64 by default.
- **Zero errors.** `OIData` drops V² and phase samples with a zero or negative
  error (a zero error gave non-finite likelihoods and fits).
- **Array fields.** The numeric prior fields of `FluxSpectrum` (`widths`,
  `scale`, `poly_width`) and `DifferentialPhase.prior_width` are NumPy arrays;
  read them with `tuple(...)` where you compared them with a tuple.
- **`line_pa`.** `KeplerOrbit.frame(...)["line_pa"]` is in `[0, 360)` (was
  `(-180, 180]`).
- **`DetectionMC.load`** raises `ValueError` for a file format other than 1
  (a file without `format` reads as 1).
- **Warnings.** Four warnings that set no category now say `UserWarning`. A
  pipeline no longer emits a `FutureWarning` (it is provisional).

**`Tabulated` is removed in 0.4.0**, as the 0.3.0 entry promised:
`Tabulated(ratio, wavel)` → `Nodes(ratio, wavel)`, with the free values at
`flux.values` (see Removed). The other deprecations wait for 0.5.

### Added

**Orbits** (provisional; see the orbit-fitting page)

- **`Epochs` and `OrbitalBinary`**: a binary on its orbit, observed as one
  snapshot per dataset (the default for orbits whose motion within a night
  is negligible). `Epochs.resolution_mas` gives the data's finest λ/B.
- **`epoch_positions`** fits the companion's position and covariance in each
  dataset on the scale-marginalized surface `m = -Σ_b (ν_b/2) ln χ²_b`, in
  which each dataset's V² and closure-phase error scales are integrated out
  under their Jeffreys priors (`epochs.marginal_loglike`; `s_max=` bounds the
  scale, matching `scipy.integrate.quad` to about 1e-11, and `dof=` sets an
  effective-dof fraction). Positions, covariances and
  `EpochPositions.gap_marginal` therefore do not change when one dataset's
  errors are rescaled. It refines the top `n_peaks=5` distinct local maxima
  (`EpochPositions.peaks`, a catalogue of `EpochPeaks`), measures
  `gap_marginal` against the best refined rival, and records raw χ²/N and the
  profiled scales (`chi2_raw`, `scale`), with a `UserWarning` for datasets
  with raw χ²/N > 4. Data with gains, closure-phase offsets or a
  model-dependent covariance raise `NotImplementedError`. `decisive`,
  `positions(min_gap=...)` and `start_from_positions(min_gap=...)` compare
  `gap_marginal`.
- **Night flags and UD-aware seeds.** `epoch_positions` flags nights that
  should not seed an orbit (`alias-peaks`, `low-gap`, `chi2-high`,
  `sep-below-resolution`, `flux-at-bound`), each with a reason (`NightFlag`,
  `EpochPositions.flags`, `.flagged`, `.flag_codes()`), with the keywords
  `flag_nights`, `flag_alias_nats`, `flag_min_gap`, `flag_chi2_factor`,
  `flag_sep_fraction` and `flag_flux_tol`. `primary_diam=` fits the
  per-night companion beside a uniform-disc primary (also
  `start_from_positions(primary_diam=, flag_options=)`).
- **Starting orbit fits**, all model first: `rank_orbits(model, data, orbits)`
  ranks trial orbits (e.g. from `starting_orbits`) by the log likelihood of
  all epochs; `chain_starts` picks the best distinct orbits, one per chain;
  `start_from_positions(model, priors, data, start_values, ...)` chains
  positions, starting orbits, ranking and `fit` into an `OrbitStart`, whose
  `chain_values` give one start per distinct mode, `.seeded` and `.ambiguous`
  name the datasets that seeded the orbits and those left out (with a
  `UserWarning`), and `.failed` records refinements that raised;
  `likelihood.chain_init_params` turns one start per chain into numpyro's
  `init_params`. `rank_orbits` and `start_from_positions` take `scales="quoted"
  | "marginal"` (default `"marginal"`, as in `score_orbits`); rankings use
  `orbit_search.rank_scores` (ties within 1e-3 nat broken by index,
  non-finite scores last). `start_from_positions` forwards `time_limit` and
  keeps going when one refinement fit raises (a `TypeError` or `ValueError`,
  such as a misspelt option, still raises at once).
- `starting_orbits(batch_size=)` bounds the memory of the grid solve, and
  `sign_free=True` seeds from positions known only up to a sign.
  `orbits.period_grid(times, p_min, p_max, k=9)` gives trial periods uniform
  in frequency with δP ≤ P²/(kT) over the baseline T, finer than ARMADA's
  and The Joker's rules; `start_from_positions` warns when its `periods` are
  coarser than that over the seeding datasets.
- **`score_orbits`** (`virgil.orbit_search`): `score_orbits(model, epochs,
  orbits, shared=SharedFlux(...), terms=(...), scales="marginal")` scores a
  stack of candidate orbits on all epochs in one compiled, `lax.map`-batched
  kernel (`model(orbit, flux)`; `rank_orbits` takes `model(orbit)`). The
  companion flux is shared (one per band, optionally f₀(λ/λ₀)^β, at most 1 in a
  reference band), integrated out under a log-uniform prior and also reported
  profiled with an uncertainty from the curvature. The coarse grid seeds
  Newton steps to the profiled peak, and Gauss–Legendre nodes then integrate
  it, so the marginal is within 0.05 nat of brute-force quadrature (a
  coarse-grid fallback is flagged in `OrbitScores.fallback`). Each dataset's
  error scales are marginalized as in `marginal_loglike`, bounded by default
  to [1/5, 5] (`s_max=5`; `None` for the unbounded closed form), and the
  scale needed at the profiled flux is reported in `OrbitScores.scale`
  (`scale_at_bound` flags those pressing on `s_max`). Gains and
  closure-phase offsets are profiled analytically; `RVData` and
  `PositionData` terms or callables are added. `max_evaluations` refuses a run
  whose predicted cost is over budget, and `rank_scores` /
  `OrbitScores.order` rank with ties broken on quantized scores.
- **`fit_orbit_aliases`** (`virgil.aliases`) fits orbits whose epochs
  undersample the period: alias bands `N = round(T/P)`, each fitted to the
  closure phases of all epochs with per-epoch error scales integrated out,
  with Laplace and importance-sampling evidences, band probabilities and
  posterior samples. `AliasResult.table()` gives one row per band;
  `AliasResult.save` writes a `format` field (1) and `load` refuses a newer
  one. `checkpoint_dir=` writes each band as soon as it is fitted and a rerun
  with the same data, positions and settings loads it (each band has its own
  random stream, so a resumed run reproduces an uninterrupted one);
  `on_band=` is called with each `AliasBandResult`. Failures are rows, not
  crashes: `band-failed`, `hessian-singular`, `no-valid-mode`, `starts-failed`
  (`AliasBandResult.error`, `.n_starts_failed`); a `RuntimeError` is raised
  only when every band fails, and the likelihood and gradient are evaluated
  once before any band, so a misconfiguration raises there. `AliasResult`
  also records `labels_free`, `period_bound`, `period_at_lower_bound`,
  `window_capped`, `p_range`, `primary_fit` and `unseeded`.
  - **Near-equal binaries.** `sign_free=True` adds sign-free seeds beside the
    signed ones (closure phases cannot tell `(dra, ddec)` from `(-dra,
    -ddec)` epoch by epoch at flux ratios of 0.9–1, and the wrong sign mix
    landed on the wrong alias); fluxes are also seeded near the top of
    `flux_range`.
  - **Resolved primary, contrast loss, bright companion.** `primary_diam` gives
    the primary a uniform disk (fixed, or `"fit"`/`"auto"` for one shared
    log-uniform diameter); `vis_amp=(lo, hi)` gives every epoch's V² its own
    profiled amplitude; `flip_flux` refits a mode whose flux ratio is pinned at
    the top of `flux_range` with the stars' labels swapped
    (`AliasBandResult.flipped`); `init=` is a warm start always refined in the
    band holding its period.
  - **Tempered evidence for correlated spectral channels.** `temper=True`
    divides every block's log likelihood by the effective-channel factor
    `f = N / N_eff`, estimated as the empirical within-chain covariance factor
    `sum (sum_chain z)^2 / sum z^2` of the residuals of each night's best static
    fit, over the chains of consecutive channels of a baseline or triangle.
    It is exact for AR(1) and common-mode correlation alike, is 1 for white
    noise, at most the channel count, used as `max(f, 1)` (tempering never
    sharpens). On 20 seeds of strongly correlated noise the median sky-plane
    offset/spread is 1.1 and 1.3 tempered against 2.6 and 3.0 untempered.
    `temper=False` is the old behaviour and a finite float ≥ 1 a fixed `f`
    (which also tempers the `primary_diam="auto"` resolution test).
    `AliasResult.temper_f`, `.temper_blocks`, `.temper_rho` and `.tempering`
    record it; `block_dof()` gives the block-keyed `dof` for later stages. New
    module `virgil.tempering` (`channel_chains`, `estimate_factors`,
    `ChannelFactors`), with `marginal_loglike(block_dof=)` and
    `estimate_primary_diameter(block_dof=)`.
  - `period_window` computes a default `p_range` from the data (a Nyquist-style
    sampling limit at the bottom, 4 times the span at the top, capped at
    `n_bands_max=200` bands with a warning); `estimate_primary_diameter` and
    `PrimaryDiameter` are new. L-BFGS is followed by Levenberg–Marquardt
    polishing, aimed at the `not-converged` and
    `hessian-not-positive-definite` flags.

- **Short arcs narrow the alias bands** (`virgil.arcs`, provisional).
  `fit_orbit_aliases(short_arcs=True)`, the default with `EpochPositions`
  and `skip_flagged=True` when every seeding night beats its point
  reflection by `arc_min_gap` (5 nats of `gap_marginal / f`, tempered
  like the evidence, with the cut's position and velocity covariances
  times `f`; a twin's reflections tie, and other positions seed every
  band as before),
  measures each night's
  velocity from its two halves (`night_motion`), joins the nights shortest
  gap first and keeps the periods whose Thiele–Innes fit to the positions
  and velocities, profiled over eccentricity, inclination and orientation,
  is within `arc_delta_chi2` (25) of the best (`unwrap_cycles`), so a long
  gap admits only the cycle counts the shorter arcs allow. Each band is
  seeded deterministically from the same fit (`arc_profile`). An excluded
  band whose seeds beat every kept band's is fitted after all, with a
  warning, and too few data keep the whole window. Positions, and each
  half night behind a velocity, get a floor of `arc_jitter` (0.1) of
  λ/B_max. `AliasResult.short_arc` records the
  admitted periods and bands and the cheap-check scores; band
  probabilities `p` are now conditional on the bands fitted, not the whole
  window. The arc seeds are added beside each kept band's position seeds, not in
  their place, and a half-night velocity is accepted only from a converged
  refinement. `short_arcs=False` is the previous behaviour. Checkpoints from
  earlier versions are not reused.
- **Tutorial**: the orbit tutorial leads with `OrbitPipeline`. It simulates
  two observing seasons of a binary whose period is aliased, runs
  `OrbitPipeline.from_oifits(files).run()`, reads the raw χ²/N, the checks,
  the alias bands and the orbit, and then shows the functions the pipeline
  calls (`epoch_positions`, `fit_orbit_aliases`, tempering) for readers who
  need control. `start_from_positions` is marked superseded by
  `fit_orbit_aliases` and `OrbitPipeline` for an unknown period (docs only;
  the function is unchanged).
- **Orbit fitting with gains.** The scale-marginalized surface
  (`marginal_loglike`, `epoch_positions`, `rank_orbits`, `fit_orbit_aliases`)
  accepts data with gains or closure-phase offsets, which it refused. Each
  affected block has covariance s²D + UUᵀ, so ln s is integrated numerically
  (32-node Gauss–Legendre on a window around the integrand's peak, exact log
  determinant at every node). Such data take `s_max = 10` when none is given.
  `vis_amp` is refused for them; `vis_amp=True` (the default) then means none,
  logged at INFO. For data with gains or offsets, an estimated tempering
  factor above 3 warns "noise model inadequate".
  Grid batches for such data come from a memory budget (`memory_budget`,
  1 GiB on a CPU by default) that counts the observables and the gains' rank,
  so no data set runs out of memory by default; `epoch_positions`,
  `fit_orbit_aliases` and `OrbitPipeline` take `batch_size` and
  `memory_budget`. `epoch_positions` scores such grids at one fixed scale per
  block (one whitening per point) and integrates ln s only within
  `coarse_nats` (30) of the best point and its neighbours; the peaks are
  refined with the full integral and `EpochPositions.exact_fraction` reports
  the share. On a calibrated GRAVITY night (882 V², 588 closure phases) the
  peaks are unchanged and the grid stage is about 6× faster. `gap_marginal`
  rivals more than `coarse_nats` below the best, and the stored surface
  there, are approximate; `coarse_nats=None` restores the full integral.
- **Night flags inform instead of excluding.** `sep-below-resolution` is
  reported but no longer keeps a night from seeding an orbit
  (`NightFlag.blocking`; `epoch_positions(flag_sep_blocks=True)` restores
  it). `chi2-high` compares each night with the median of its own instrument
  mode (`flag_chi2_groups`, by default the spectral channels and wavelength
  range). `EpochPositions.log_broad` is each night's mean likelihood over the
  grid, and `broad_loglike` is in `virgil.epochs.__all__`. `AliasResult.unseeded`
  lists only nights with a blocking flag, and the `night_flags` check says
  that non-blocking flags are reported only.
- **Outlier nights.** `fit_orbit_aliases(outlier_eps=...)` mixes a broad
  outlier component into each night's likelihood, so one bad (sidelobe) night
  cannot move the orbit; `AliasResult.outlier` gives each night's posterior
  outlier probability and `AliasResult.night_flags` the post-fit
  `orbit-prefers-secondary` flag (`sidelobe_flags`). It is off by default in
  `fit_orbit_aliases`; **`OrbitPipeline` sets `outlier_eps=0.02`**, so a
  pipeline run down-weights a night the orbit cannot explain instead of
  following it. Pass `outlier_eps=0` for the plain likelihood.
  `OrbitPipeline` rejects an `outlier_eps` outside [0, 1), and its summary's
  `search` section gains `outliers` and `night_flags`.

**`MultipoleDisk` family** (provisional; follows the ideas of frank, Jennings
et al. 2020, and vis-r, Kennedy 2025, credited in the docstrings and on the
contributors page; no code was copied)

- **`MultipoleDisk`**: an inclined, rotated disk with a radial profile
  (`ThinRing`, `GaussianRing`, `RadialPowerLaw` with sharp or soft edges,
  `ExpTail`, `Gap`, `RadialSum`) and an optional Gaussian blur, built on
  `multipole_disk_cvis`, a Hankel transform by quadrature (Gauss–Legendre, or
  Gauss–Laguerre for an exponential tail). A weight belongs to a term of the
  sum, as `RadialSum(a, (b, 0.3))` or `RadialSum(core=ring, halo=(tail,
  0.2))`. `ExpTail(cutoff=)` uses a Gauss–Legendre rule on a finite tail.
- **Height and harmonics.** `height=` takes `Cone(alpha, tip)` or
  `PowerLawHeight(h_ref, r_ref, psi)` (each ring is lifted along the disk's
  normal), and `harmonics=` takes `Harmonics(c=, s=)` of Cartesian
  coefficients, constants or functions of the radius (`RadialLinear`,
  `RadialPower`). `min_intensity()` reports where the image would go
  negative. `multipole_disk_cvis` takes `c`, `s`, `height` and `stretch`.
- **Positive azimuthal harmonics, `VonMises` and `virgil.harmonics`.**
  `Harmonics(degree=M)` (or `w=`, `re=`, `im=`) is the *positive* form by
  default: brightness `|h(e^{i theta})|^2` normalized to the azimuthal mean,
  with `h` the minimum-phase polynomial of complex reflection coefficients
  from unconstrained latents. Every non-negative image of degree `M` is
  reachable (Fejér–Riesz), the image is positive for every parameter value,
  and the root-flip multimodality is removed. The linear form (`c=`, `s=`) is
  unchanged and chosen whenever `c` or `s` is given. `VonMises(kappa, theta0,
  degree=M)` is an arc with analytic harmonics (`truncation_power()` reports
  the dropped power). `virgil.harmonics` converts every form to and from the
  image harmonics (`to_image_harmonics`, `from_image_harmonics`,
  `Harmonics.convert`, and the legacy `az_amps`/`az_pas` conventions via
  `from_legacy` and `to_legacy`). Credited to Dumitrescu 2007 and
  Barndorff-Nielsen & Schou 1973.
- **Gaussian-process profiles.** `RadialGP(r_in, r_out, ...)` is a whitened
  Matérn profile (positive, $\exp g$) and `RadialGPFunction` the same process
  as a radial function $g(r)$ (a spectral index or a harmonic coefficient);
  `MultipoleKernel` is a prior over the harmonic coefficients
  (`Harmonics(degree=M, prior=...)`); `MultipoleDisk` takes `spectral_index=`
  (number, function or `RadialGPFunction`) with `wavel0=`; `gp_priors` builds
  the priors, and `fit`, `numpyro_model` (NUTS, flat coordinates) and
  `imaging.log_evidence` (Laplace over the latents) support them.
  `freeze_geometry(data)` caches the Bessel matrices. After FRAP (Yoshida et
  al. 2026), credited in the docstrings, the contributors page and the
  bibliography.
- **`MultipoleField`**: a residual field for an `Image`, a Gaussian process
  on a polar grid in a disk's frame (`MultipoleField.from_disk` copies the
  geometry, or `geometry_from="disk"` ties it to the sibling disk;
  `System.resolved()` bakes the ties in), with the Matérn kernel of
  `RadialGP` in radius and the azimuthal spectrum of `MultipoleKernel`
  (`nu` up to infinity, `nu_theta`). Whitened latents, so `image_priors`,
  `fit`, `log_evidence` and `numpyro_model` work as for `GaussianField`;
  `multipole_field_priors` gives bounded log-uniform priors on `sigma`,
  `length_r` and `length_theta`. After FRAP and frank's polar representation.
  `FlaredDisk*` is unchanged.
- **Names** follow the stable conventions: `radius`, `sigma` (a standard
  deviation), `inc`/`pa` in degrees, `r_ref`/`h_ref` for reference values.
  `Harmonics(c, s, *, degree, form, re, im, w, prior)` takes everything after
  `s` by keyword, and `ExpTail(r_in, scale_length, *, cutoff)` is
  keyword-only for `cutoff`.
- **Tutorial**: *a planet in a gap between inclined rings*
  (`notebooks/multipole_disk_rings.ipynb`) simulates chromatic VLTI-like V²
  and closure phases of a star, three coplanar blackbody rings (one
  lopsided) and a planet in the gap; fits the parametric family with
  invariant priors (MAP, Laplace covariance and evidence); then recovers the
  rings, their colours and the planet with one `MultipoleDisk` whose
  profile, spectral index and harmonics are Gaussian processes.
- **`TruncatedCone`, `ModulatedGaussianRim` and `GaussianArc` run on the
  `MultipoleDisk` engine** with the same names and parameters (see Changed
  for what shifts). `TruncatedCone(..., az_amps=, az_pas=)` multiplies every
  ring's brightness by `1 + Σ A_m cos(m (φ - φ_m))` in its own azimuth
  (analytic visibilities, matching render); unmodulated cones are unchanged.
  Credits carry over from the earlier cone code, `frank` and `vis-r`.

**Pipelines** (`virgil.pipeline`, the `pipeline` extra; provisional)

- **`ImagingPipeline`** (`virgil.pipeline`, provisional; `virgil-pipeline
  image`): a model-independent image of a dataset.
  `ImagingPipeline.from_oifits(files).run()` chooses a Gaussian-process
  prior's hyperparameters by evidence, decides by ΔlogZ (5 nats) whether a
  central star is present, fits `members` (default 8) images at those shared
  hyperparameters, varying only the start, the pixel scale and the field,
  keeps the members that fit V² and closure phases, and writes the mean and
  the spread of the members as sky-oriented FITS images (`Result.image()`, a
  `SkyImage`), the fitted models and a quicklook whose first cell says that a
  good fit does not promise the morphology. Twelve checks report the raw χ²/N
  split into V² and closure phases; the spread of the members excludes
  hyperparameter uncertainty and any error every member shares. `run_member(k)`
  fits one member, for cluster array tasks. `wavel_range` and `error_floor`
  are validated before the data load. CLEAN starts, classical
  regularizers, `anneal` and the channels are not in this version.
- **`OrbitPipeline`** (`virgil.pipeline`, provisional; `virgil-pipeline
  orbit`): the default multi-epoch binary-orbit run is a blind period-alias
  search. `OrbitPipeline.from_oifits(files).run()` groups the files into
  nights, finds the companion's position on each, searches every alias band
  of the period with the engine's defaults (no published period, window,
  warm start or prior), ranks the bands, reports the winner's margin over the
  runner-up and a `no-orbit` test against a single star, and samples the
  winning band (and any band that is not negligible) with NUTS. New checks:
  `night_flags`, `labels_free`, `alias_ambiguity`, `tempering`,
  `primary_resolved`, `no_orbit` and `period_bound`. `Result.aliases()`
  reloads the search, and `AliasResult` records its prior ranges
  (`AliasResult.prior`).
- `BinaryPipeline` (companion search, contrast limits, fit and NUTS
  posterior) and `StarPipeline` (angular diameter and limb darkening, with
  a registry of models by short name) run fixed, named stages on public
  virgil functions and write a run folder: `run.json`, `summary.json`, HDF5
  grids and samples, plots, the processed OIFITS and an executed quicklook
  notebook, with deterministic quality checks (`Check`). Runs resume; a run
  folder whose settings, data or template differ raises
  `ConfigMismatchError`; re-running a stage deletes every later stage's
  outputs. `load` reloads a run as a `Result`, which serves only complete
  stages; `run.json` records provenance per stage; and the `virgil-pipeline`
  command runs a pipeline and prints, validates (a run left `running` fails)
  or re-renders the quicklook. `import virgil` does not import the
  pipelines. See the Pipelines page.
- `StarPipeline` compares diameter lobes and models on −2 log L
  (`model_loglike`, so with `error_scale="fit"` each candidate is scored at
  its own maximum-likelihood error scales), walks the registry's parent chain
  with a BIC penalty (`summary["comparison"]` decides, `summary["comparisons"]`
  lists all), and keeps the uniform disk's diameter scan with its scales
  profiled. The look-elsewhere significance is computed in log space, so it
  stays finite for large searches.
- **`DiskPipeline`** fits a `MultipoleDisk` around one star. `model="gp"`
  (the default) fits one `RadialGP` profile with low-degree harmonics and,
  with enough channels, a Gaussian-process spectral index, keeping the
  length scale of largest Laplace evidence from `ell_halffringes`;
  `model="rings"` adds `GaussianRing` profiles that share the inclination
  and position angle one at a time until the BIC stops improving, re-placing
  each ring after every fit (a count with two rings too close to tell apart
  is one ring fitted as two and is left out of the comparison, a ring whose
  removal lowers the BIC is pruned, and the `ring_count` check warns when
  such a count had the lower BIC). The geometry is found from a single-ring grid
  over inclination (uniform in cos *i*) and position angle, never from one
  local start; `companion=True` searches the residuals for a point source
  and fits a detection jointly with the disk (the residuals are always
  searched, to check that the disk has not absorbed one). Priors are
  group-invariant. Stages are `geometry`, `profile`, `compare`, an opt-in
  `posterior` (NUTS with the geometry released) and `quicklook`. Defaults
  come from imaging nine PIONIER post-AGB discs: for `model="gp"`,
  `error_scale="auto"` (the default; `"rescale"` for the Gaussian process,
  `"quoted"` for the rings) estimates one MacKay error scale from a baseline
  fit and evaluates every fit and Laplace evidence on the rescaled data
  (`chi2.error_scale_factor`; χ²/N stays on the quoted errors), and
  `imaging.error_scale` accepts a `MultipoleDisk` with Gaussian-process parts
  as `log_evidence` does. The ℓ grid cannot choose its own edge: the best ℓ on
  an edge extends the grid once (`ell_floor_halffringes`, 0.5, is the lowest
  ℓ/√3), `on_grid_edge` is recorded and `length_scale` warns if it persists.
  The star can be a uniform disk (`star_diameter_mas`, which also floors the
  inner radius at its radius) with a power-law spectrum (`star_index`), and
  the summary's `model.star` records it. New checks: `structure_unconstrained`
  (flat evidence and unstable radial profiles across ℓ), `render_consistency`
  (the rendered image against the model's χ²) and `at_bound` (parameters within
  0.1% of a prior bound). See the Pipelines page.

**Imaging, inference and data**

- `Image` and the `FlaredDisk` family take `ft="auto" | "dft" | "nufft"`
  (virgil#75), and `virgil.set_ft_defaults` changes the defaults. nufftax is
  now a dependency. **The default changes results slightly**: on a CPU an image
  of at least 64² pixels with at least max(500, 2N) points (N the larger side),
  or on a GPU one of at least 128² pixels with 3000 or more points, uses a
  non-uniform FFT, accurate to 1e-5 (float32) or 1e-12 (float64) of the
  zero-baseline visibility rather than exact. Pass `ft="dft"`, or call `set_ft_defaults(ft="dft")`, for the
  exact sum; CLEAN (`_CleanScene`) has no `ft` and follows the global default.
  `jax.hessian` and nested `vmap` through the NUFFT fail in nufftax 0.6.1; use
  `ft="dft"` there. `ft` and `set_ft_defaults` are provisional.
- **`starting_image(pad_to_mas=...)`** pads the start image with empty pixels
  out to a field wider than the interferometric limit λ/B_min, which
  `largest_mas` (a cap) cannot do: `largest_mas=54` used to give HR 4049 a
  ±17.8 mas grid whatever was asked. `largest_mas` above λ/B_min now
  warns, naming `pad_to_mas`. Defaults are unchanged.

- **`image_field`, `second_moment_fwhm` and `OIData.unflagged`.**
  `image_field(data)` is `field_fwhms` (6) times the source's FWHM, at least
  `inner_elements` (4) resolution elements λ/(2 B_max) across, and at most
  `field_of_view(data)`; the FWHM is `second_moment_fwhm(data)`, a
  straight-line fit of −ln V² against (B/λ)² on the unflagged samples with
  V² ≥ 0.3 (or the FWHM you give). `OIData.unflagged` is the mask of the
  samples an observable uses. `field_of_view` takes `aperture_mas` and stops
  at the bandwidth-smearing field R λ/B_max when the data carry channel
  widths. `starting_image` takes `field_fwhms`, `inner_elements` and
  `aperture_mas`; `EnsembleSpec` takes `field_mas` and `aperture_mas`;
  `reference_starts` takes `aperture_mas`.
- **`virgil.ensemble`**: PYRA/MYTHRA-style reconstruction ensembles.
  `EnsembleSpec`, `draw_groups`, `run_group` (one L-curve per geometry group),
  `combine` and `ensemble` return an `Ensemble` (mean `Image`, per-pixel σ,
  raw χ²/N per dataset). The L-curve window keeps the weights from
  `window_dex` below the corner up to it; the default weight ranges (`tsv`
  0.1–10⁴, the others 10⁻⁴–10 per data point) were set from the corners of the
  60 datasets of virgil-validation's contest bench, and each sweep draws one
  log-uniform weight per equal bin of `log w`. The iterative mean admits a
  member if every dataset's χ² stays within `mean_rtol` (by default each
  dataset's χ²/N noise √(2/N)) of the best member's; diverged members are
  dropped with `reason="diverged"`, and `combine` warns when fewer than
  `min_kept` (3) are kept. A field too large for `max_npix` keeps its size
  with coarser pixels.
- **`virgil.metrics`**: scores for a reconstructed image against a known
  truth from the imaging contests' published definitions:
  `lawson_sigma_over_peak`, `rms_convolved` (after a Gaussian beam;
  `pixel_scale_mas` is needed only with a beam), `l1_score` and `ncc`.
- **Variational inference.** `virgil.svi.variational(model, priors, data,
  ..., start=fit_result)` fits a guide to `numpyro_model`'s posterior with
  numpyro's SVI and returns a `VariationalResult`: draws under NUTS's site
  names, the loss history, the guide and its parameters, a convergence flag
  and the PSIS k̂. Every guide starts at the Laplace approximation and
  learns in units of its widths. The default is a block neural autoregressive
  flow (`"bnaf"`), which follows curved, skewed and bounded posteriors;
  `"iaf"`, `"mvn"` (a Gaussian, ≈ Laplace) and `"laplace"` are the
  alternatives. float64 by default, as for `fit`; `flat_coordinates` is passed
  through. New page: Variational inference.
- **`gauss_newton_mass(..., noise=...)`** takes the fitted error terms of
  `fit` and `numpyro_model` under their numpyro site names
  (`"noise.vis_scale"`); for Gaussian data the matrix is the Fisher
  information with rows √2 log σ for each visibility and phase error. It
  accepts data with gains, any elementwise prior (HalfNormal, Beta, Uniform
  and so on, each with the curvature of the sampler's transform), and
  likelihood terms with a `log_norm`. Above 4096 coordinates it returns the
  diagonal with a `UserWarning`. The result's `condition` is the condition
  number of its correlation matrix. `SingularCurvature` now means a
  direction that neither the data nor a proper prior constrains, or a matrix
  that is not positive definite: a parameter under a bounded prior no
  longer raises. A prior with no diagonal curvature (`MultivariateNormal`,
  `Dirichlet`) raises `NoLeastSquaresForm`.
- **NUTS in flat coordinates (opt-in).** `numpyro_model(...,
  flat_coordinates=True)` maps a `LogUniform` scale to NUTS's unconstrained
  coordinate through log x, an `IsotropicInclination` through cos i and an
  `IsotropicLatitude` through sin(lat), for parameters and `noise=` terms
  alike, as `fit` optimizes them. Sites keep their names, samples are the
  model's own parameters, and the posterior is unchanged. It is off by
  default: simulation-based calibration (1000 replicates) showed no gain in
  effective sample size per second or in divergences. The 62² imaging
  notebook, with `gauss_newton_mass`, ran in 67 s instead of 550 s with it, so
  the tutorials that sample large images pass it. Pass the same value to
  `gauss_newton_mass`, `variational` and `chain_init_params`' model;
  `chain_init_params(..., mass=)` raises if they differ.
- **Evidence helpers take several field Images.** `log_evidence`, `error_scale`
  and `laplace_samples` accept `path=None` (every Image whose log-brightness
  is a `GaussianField`) or a tuple of paths, with the Jacobian over all those
  latents together, so a compact and an extended GP Image in one `System`
  get a joint Laplace evidence. `laplace_samples` then returns `"latents"` as
  a dict keyed by Image path. The default `path="env"` is unchanged.
- **Fit budgets and guards.** `fit(..., time_limit=, progress=)`: LM and L-BFGS
  run in chunks of steps, check the wall clock between them and stop,
  unconverged, with `info["stop"] == "time"`; `progress` reports steps, loss,
  gradient norm and elapsed time between chunks and when the fit finishes.
  `info["stop"]` gives every unconverged fit's reason (`"limit"`, `"time"`,
  `"stalled"`, `"non-finite"`, `"failed"`), a fit by any method that ends on
  a non-finite loss is marked unconverged, and `info["at_bound"]` lists
  parameters that ended at a genuine edge of an interval prior.
- **Opt-in bandwidth smearing.** `OIData(path, bandwidth=True)` (or
  `read_oifits(..., bandwidth=True)`) keeps `EFF_BAND` as `OIData.bandwidth`
  (`with_bandwidth(dlambda)` sets it for simulated data); models are then
  smeared over boxcar channels by a `sinc` envelope on each component offset
  (`models.smeared_model`). Off by default.
- **`spectra.Scaled(template, ratio)`**: a fixed spectral shape with one free
  reference flux (`ratio` at the template's `wavel0`), so any spectrum (a
  library SED, a `Sum` with lines) can be a companion in the grids, limits and
  detection statistics, with `comp.flux.ratio` as the flux axis. Regression
  tests run every grid, limit and detection tool on a WR + OB system, and
  show a differential-phase-only detection of a 0.1 mas companion that a grey
  V²/T3 fit misses. Tutorial: *Chromatic companion search*.
- `OIData(..., insname=)` selects the instrument table(s) when reading an
  OIFITS file, as `read_oifits` does; the GRAVITY fringe-tracker/science
  error message already told users to pass it.
- `OIData.select` takes `ranges=` (a union of wavelength windows), `exclude=`
  (windows to drop, such as a line) and `observables=` (`"vis"`, `"phi"` and
  the extra observables' kinds), so closure-phase-only or line-free subsets
  need no hand-flagged records. Dropping the closure phases rebuilds
  differential phases with their closure part, and gains go with the
  visibilities.
- **Image smoothing** (provisional). `Image(smooth_fwhm_mas=...)` carries a
  fixed Gaussian smoothing kernel: its visibilities are the pixel transform
  times the analytic Gaussian, and `Image.smoothed()` returns the convolved
  pixels. `TSV`, `TV`, `MaxEntropy`, `Laplacian`, `StarletL1` and `LogSum`
  take `smoothed=True` to penalize them, as MPoL does. Defaults are
  unchanged. `CalibrationNuisance.taper_fwhm_mas` stays; setting both warns.
- **Matrix Fourier transform for rendered disks on DISCO lattices**
  (provisional). `FlaredDisk` (and `FlaredDiskHG`, `FlaredDiskGaussian`,
  `FlaredDiskPowerLaw`) use the exact two-sided matrix Fourier transform on
  data with a `uv_grid`, as `Image` does. A new `rotation_deg` orients the
  disk's pixel grid; the default, `"grid"`, renders it in the frame of the
  data's lattice (North up without one), so DISCO data always take the fast
  path. On the old PDS 70 product this changes ln L by under 0.005 (the pixel
  discretization) and runs 3–4× faster for a 96² disk at 1300 uv points. With
  `"grid"`, `disk.model(u, v, wavel)` and data without a lattice keep North-up
  pixels, so they differ from the lattice path by that discretization.
  Ensembles on drpangloss DISCO dictionaries now cap every field factor at 1,
  as on other lattice data (a larger field aliases), so the same seed draws
  smaller fields than before. Dictionaries with `disco_vis_mat` or
  `disco_phi_mat` (the drpangloss DISCO format) get a `uv_grid` when their
  single-wavelength samples lie on a lattice to 1e-5 cells, as the mixed-DISCO
  loader already did; both share one lattice check, and `find_uv_grid` fits
  the lattice by least squares, so float32 coordinates no longer defeat it.
- **`OIData.with_closure_offsets` accepts three-telescope data** (one
  triangle per frame and channel): triangle and `modes` offsets work, baseline
  offsets are folded into the triangle group (width √3 × baseline) with a
  warning. The result carries `cp_noise`, so the likelihood uses the
  sine-plus-penalty form of four-telescope data, whose normalization differs
  slightly from the von Mises form of plain three-telescope data.
- **`virgil.crossval`** (provisional): k-fold cross-validation of a
  hyperparameter for any likelihood term that takes a mask, as one process or
  as array tasks with JSON records. `aggregate` ranks a setting with a NaN
  score last (with a warning) instead of choosing it as `best`, raises
  `ValueError` if every setting is NaN, and works with a single fold; a group
  whose scores are all NaN has `group_best = -1`. It is provisional and may
  change in any 0.4.x release; it claims no parity with other imaging
  packages.
- **`virgil.radio`** (provisional): fits calibrated complex visibilities from
  CASA measurement sets, with per-observation flux scales, phase-centre
  offsets and error scales fitted with the sky (`ComplexVisData`,
  `CalibrationNuisance`). It is provisional, not final: its design, defaults
  and names may change in any 0.4.x release, and it claims no parity with
  MPoL or with published DSHARP analyses. A comparison with MPoL on ALMA
  IM Lup data is in progress in virgil-validation. Reading a measurement set
  needs the `radio` extra, `pip install "virgil-astro[radio]"`, which brings
  `casatools>=6.6`. On some systems casatools 6.7.6 fails to import with a
  missing `OPENSSL_3.0.1` symbol; casatools 6.7.0.31 works. virgil does not
  pin it.

### Changed

- **`GaussianArc` is a von Mises arc (a change of shape).** The weight along
  the arc is `exp(kappa cos(angle - pa))` with `kappa = (2 sqrt(2 ln 2)
  radius / length)^2`, where it was a Gaussian in arc length taken once round
  the circle and cut off at the antipode of `pa`. `length` is still the FWHM
  along the arc and the parameter names are unchanged; the two shapes agree
  to about 1e-3 in V for arcs with `length` below a tenth of the semicircle
  (1e-2 at a fifth) but differ for long arcs. The visibility is analytic (a
  `ThinRing` with `VonMises` harmonics and a blur); `degree` (default chosen
  from the starting `radius` and `length`) replaces `nodes` for the
  visibility, and `nodes` only samples the rendered image. Arcs narrower than
  256 harmonics allow (`length / radius` below about 0.05) are integrated
  along the arc at `nodes` points, as before (`quadrature=True`);
  `is_physical` is false when the series drops more than 1e-4 of its power.
- **`TruncatedCone` changes numerically.** Its default `n_rings` goes from 32
  to 48, since too few Gauss–Legendre nodes fail sharply at long baselines,
  and its visibilities at fixed parameters shift by the old midpoint rule's
  error (about 8e-4 at the old default, worth Δ log L of about 4 per GRAVITY
  epoch). Gauss–Legendre converges spectrally: for the Apep-like cone
  (edge-on, baselines to 130 m at 2.2 µm) 40 nodes agree with a converged
  midpoint stack to 1e-6, where 64 midpoint rings are off by 5e-4. The cost
  per call at the default rises (48 nodes against 32 rings); it is lower
  only at equal accuracy. `rings()` and `render()` use the same nodes as
  `model()`. `ModulatedGaussianRim` agrees with the old code to rounding.
- **`fit_orbit_aliases` needs no per-system arguments.** The defaults
  changed, so results change: `p_range` is optional (computed by
  `period_window`); `sign_free=True`; `primary_diam="auto"` (a first pass fits
  a shared uniform-disc diameter, resolved at a gain of 5 nats over a point
  primary, then fits it and refines the seeds; a point primary otherwise);
  `vis_amp=True` (a per-night V² amplitude in 0.5 to 1.2 for squared-visibility
  data); `flip_flux=True`, which can add a mode to a band's evidence;
  `temper=True` (see Migration). It warns when data with several channels
  have no `EpochPositions` to estimate `f` from, when a block takes the
  pooled `f` for want of chains of its own, and when `f` is outside the
  range or flat across channels (`AliasResult.tempering`). These settings
  are in the checkpoint key, so existing `checkpoint_dir`s are refit
  (format 4).
- **Seeding skips flagged nights by default** (`skip_flagged=True` in
  `fit_orbit_aliases`, `EpochPositions.positions` and `start_from_positions`;
  `False` restores the old behaviour). A flagged night's data still count in
  the likelihood; fewer than two seeding nights fall back to random starts
  with a warning, and seeds from three or fewer nights keep random
  candidates beside them.
- **`fit_orbit_aliases` and `epoch_positions` optimize with damped Newton
  steps.** Each start and each peak refinement runs Levenberg–Marquardt on the
  iteratively-reweighted curvature of the scale-marginalized likelihood,
  instead of L-BFGS-B with restarts and a Newton polish; L-BFGS-B remains the
  fallback. It reaches the same optima (positions agree to 1e-4 mas) and is
  faster, though which local optimum a start far from the ridge reaches can
  differ. `max_steps=` caps the kernel's iterations (rejected steps count) and
  the fallback's, and a refinement of `epoch_positions` that does not
  converge keeps its grid point.
- **The default image field comes from the data, not from λ/B_min.**
  `starting_image` took `min(max(500 mas, 6 FWHM), λ/B_min)`; the 500 mas
  was a holdover from one system, and since λ/B_min is 14 to 80 mas for every
  VLTI or CHARA dataset, the source's size made no difference (2.9 mas
  sources on 58 mas fields). It is now six FWHMs of the envelope (or of the
  short baselines' second moment, if larger), between four resolution
  elements across and the interferometric field of view; for a 2.9 mas
  Gaussian seen by 130 m baselines at 2.2 µm that is 17 mas. Pass
  `field_fwhms` to change the six.
- **Resolution and field use unflagged samples only.** `beam`,
  `fringe_scales`, `nyquist_pixel_scale`, `field_of_view`,
  `Epochs.resolution_mas` and the pipelines' geometry report no longer count
  flagged samples, whose u, v stay in the data (the provider flags 59 to 95% of
  V² of the 2024 contest's Obj1 GRAVITY at SNR < 1, which gave a beam of 4.3 ×
  3.3 mas against 11.6 × 8.1 mas from the unflagged data). Data without
  flags, and projected (kernel, DISCO) data, are unchanged.
- **The ensemble sweeps the data's field.** `EnsembleSpec.field_factors`
  defaults to `(0.75, 1.0, 1.5)` times `image_field(data)` (or `field_mas`),
  never beyond `field_of_view(data)`, instead of `(1, 2, 4)` times λ/B_min,
  which gave fields beyond what the data constrain. The field is sized from
  the envelope fit of `starting_image` (`envelope_fwhm`), so a bright star
  does not shrink it; every field is capped at `field_of_view(data)`, so
  `field_factors` above 1 only reach the ceiling.
- **`variational` starts error terms and gains data at their Gauss–Newton
  width** (`noise=` is passed to `gauss_newton_mass`), and priors without a
  least-squares form (HalfNormal, Beta) get a Gauss–Newton start rather than
  `init_scale`. The posterior is unchanged; only the starting width.
- **Bounded elementwise priors** (`Uniform`, `LogUniform` and so on) add the
  curvature of the sampler's transform to `gauss_newton_mass`'s matrix,
  where `Uniform` added none. The matrix is a preconditioner, so this
  changes the efficiency of NUTS, not its samples.
- **Dense-matrix limit.** `fitting._DENSE_MASS_MAX` is 4096 coordinates (was
  8192; a dense 8192 × 8192 matrix is 537 MB in float64, and its
  eigendecomposition is slower still). For 4096 < n ≤ 8192 (images of 65² to
  90²) `gauss_newton_mass` returns the diagonal, with `condition=None` and a
  `UserWarning`. `variational` with `"bnaf"`, `"iaf"`, `"mvn"` or
  `"laplace"` raises `ValueError` above the limit, before allocating.
- **Performance** (results unchanged up to rounding unless noted):
  - `FlaredDiskHG` and `FlaredDiskPowerLaw` form cos θ as y/ρ instead of
    calling `arctan2` and `cos` (cos(θ/2)ⁿ = ((1 + cos θ)/2)ⁿ/²), roughly
    halving the value-and-gradient time of a 96×96 render on CPU in float64.
    The gradient at a pixel exactly on the star is now finite;
    `FlaredDiskGaussian` still needs θ.
  - `score_orbits`, `rank_orbits` and `chain_starts` compile their kernel once
    per layout of the data, and repeated refinements in
    `start_from_positions` compile once. `epoch_positions(batch_size=)`
    defaults from the data's size (it was 4096) and `batch_size=0` raises.
  - `Epochs.snapshots` evaluates a single scene at all the datasets' times with
    one `vmap`, so the gradient graph no longer grows with the number of
    datasets (about half the jaxpr equations for an `OrbitalBinary`), and the
    `DiskPipeline` ring stages reuse one compiled model instead of recompiling
    every fit.
  - **Fit summary performance.** `fit` no longer compiles a second gradient of
    the whole model to report `info["chi2"]` and `info["grad_norm"]`. The
    chi-squared is a forward evaluation, and for L-BFGS the gradient norm is
    read from the optimizer's final state, which already holds the gradient at
    the returned point. Only LM and Adam, which have none, still compute it
    afresh. The `info` keys and values are unchanged, as is the API; the
    progress report and L-BFGS restarts also skip the gradient where they do
    not need it. For a model with many datasets this saves a large share of
    the compile time of a fit.
  - The evidence helpers (`log_evidence`, `error_scale`, `laplace_samples`,
    `chi2_by_observable`, `classic_maxent`) take the Jacobian's SVD on the
    device in float64, QR-reduced when it is far from square, instead of
    NumPy's SVD on the host. `laplace_samples` with a given `key` returns
    different draws (same distribution).
  - `dirty_image` sums the visibilities in chunks of samples, so its memory no
    longer grows as samples × pixels; `metrics.align` sweeps its shift table in
    one compiled `lax.map`.
  - `fit(method="lm")` compiles its step kernel once for every chunk and
    every `max_steps`: the step bound is traced.
  - The builders of `gain_modes`, `ClosureNoise`, `DifferentialPhase` and the
    OIFITS reader's frame grouping sort once instead of passing over all
    samples per group (quadratic before); the numeric prior fields of
    `FluxSpectrum` and `DifferentialPhase.prior_width` are NumPy arrays that
    jit traces, so scanning a prior width does not recompile.
- **Models simplification (no numerical change).** The long derivations and
  accuracy notes in the docstrings of `virgil.models` and
  `virgil.fields.MultipoleField` moved to the new
  [model notes page](docs/api/models/notes.md); the docstrings keep the
  summary, parameters and an example. The class shape checks and the ring
  splatting and in-plane blur of the disk models each share one definition.
  Results are unchanged to rounding.
- **`fit` falls back to L-BFGS only when the objective has no least-squares
  form** (`fitting.NoLeastSquaresForm`), and `variational` starts at
  `init_scale` only for that, a singular curvature
  (`fitting.SingularCurvature`) or a parameter with no starting value
  (`fitting.NoStartingValue`, a `ValueError`); other errors propagate
  instead of being taken for one of these. `gauss_newton_mass` raises
  `NoLeastSquaresForm` for data with gains, as `fit`'s Gauss–Newton path does.
- **Warnings name the caller's line** and say their category. Deprecation
  `FutureWarning`s and virgil's other warnings are attributed to the first
  frame outside virgil, not to a fixed depth that pointed into equinox's
  method wrappers or the shim; frames of JAX, numpyro and the optimizers are
  looked through. Shim messages name methods as `Class.method`.
- **Precision.** The limb-darkening coefficient matmul (binomials up to C(22,
  11) with alternating signs), the `RadialGP` and `MultipoleKernel`
  contractions, the DISCO operator application, `Centroid` and the two
  pixel-sum matmuls in `GaussianArc` and `TruncatedCone` request
  `Precision.HIGHEST`, as the other forward-model matmuls do (TF32 on A100/H100
  loses the cancellation otherwise). The AMIGO covariance check takes its
  tiny-number floor from the array's dtype.
- **Geometry.** `_geometry.position_angle` never returns exactly 360, and
  `KeplerOrbit.frame()["towards_primary"]` is wrapped to `[0, 360)`; the
  geometry helpers compute lists and array-likes (such as a pandas Series) in
  NumPy float64. One `position_angle` / `separation_pa` (North through East,
  `[0, 360)`) serves the models, orbits and plotting.
- **One field-of-view computation.** `field_of_view`, `nyquist_pixel_scale`,
  `Epochs.resolution_mas` and the pipelines use per-sample `B/λ` over non-zero
  baselines, rather than a longest wavelength and a shortest baseline that
  are not observed together when the uv sampling differs between channels.
- Differential-phase steps and anchors use one wrap, `[-π, π)`, in data and
  model alike. Phase units go through one parser with one error message; a
  blank `TUNIT` or `phi_unit` means the default unit in both.
- The `oifits` module no longer imports JAX itself (`import virgil` still
  does). `FWHM_PER_SIGMA` replaces the hand-written `2.3548` (a 1e-5
  relative change in an envelope-fit starting width).
- The orbit API settled before release: `score_orbits(model, epochs, orbits,
  ...)` is model first (the order `score_orbits(epochs, model, ...)` works
  with a `FutureWarning` for one release), `EpochPositions.gap` is gone
  (`gap_marginal` is the gap), `UNRECORDED` is no longer in
  `virgil.aliases.__all__`, and `aliases.spectral_window` is removed (unused;
  `|mean exp(2πi f t)|²` is one line of numpy).
- A private `virgil._deprecate` module implements the 0.4 change to a single
  argument order and the renamed keywords: `old_order` accepts calls in the
  0.3 positional order, recognized by which argument is the `OIData`, with a
  `FutureWarning` that names the new call (a `TypeError` once `removed=True`,
  in 0.5); `old_order` and `renamed` accept the old keyword names with a
  `FutureWarning`, and passing an old name with its new one is a
  `TypeError`.
- **The `optimistix` floor is 0.1.0** (was 0.0.11): `fit(method="lm")` needs a
  loop that stops on an unsuccessful result, which 0.0.11 did not.
- Tests and tooling: `import jax.numpy as np` and `import numpy as onp`
  everywhere in `src`, `tests`, `examples` and `scripts`, enforced by Ruff's
  `ICN001` (six `src` modules had the two swapped, so code moved between
  files silently changed backend); CI is faster and steadier, with the
  slowest tests sharing fixtures, one pull-request job running the slow
  tests, and the `dev` extra including `orbits`; releases are gated on a
  full-tier test run of the tagged commit.

### Fixed

- **`nrm_oidata` carries station indices.** It now gives the hole pair of each
  baseline as its station indices (numbered from 1), so the data can be
  written to OIFITS and the pipelines' `processed.oifits` is no longer
  skipped for it.
- **`fit_orbit_aliases(outlier_eps=...)` ranks and refines on the plain
  likelihood.** Ranking and refining under the outlier mixture lost the true
  alias on a cold search: the mixture's weight on a night vanishes when a start
  misses it, so a start that is off by a little settled on the nights it
  already fit, tens to hundreds of nats short of the optimum that the plain
  likelihood reaches from the same start (on Gl 229, the N=34 band's best mode
  was 108 nats below the plain one). The search now ranks and refines on the
  plain likelihood and applies the mixture as a polish of each mode found, so
  the mixture, and so `OrbitPipeline` with its default `outlier_eps=0.02`, can
  down-weight a night of a basin but cannot hide the basin.
- **`BinaryPipeline` starts its fit inside the position priors.** A search
  peak on the grid's edge is on the prior's bound (the grid and the priors
  share their half-width), where the unconstrained start is infinite: the fit
  returned NaN in float32 and the run crashed in the summary with `TypeError:
  where requires ndarray or scalar arguments, got NoneType`. A fit that still
  returns a non-finite parameter now raises a `ValueError` naming it at the
  fit stage.
- **`separation_pa` has a zero gradient at the origin.** `jnp.hypot` has a NaN
  gradient at (0, 0) on JAX 0.12.0, so an angular model that started on the
  origin (a non-detection's best fit) had non-finite derivatives and NUTS
  could not initialize there.
- **`clean`'s `"stalled"` stop is platform-independent.** It compares the
  predicted decrease with `eps` times the χ², not with exactly zero. At the
  optimum the sign of the gradient is rounding noise, so whether CLEAN
  stopped depended on the platform and on thread scheduling, and it sometimes
  ran to `max_iterations` instead.
- **Pipeline robustness (`virgil.pipeline`, provisional; valid runs and
  their results are unchanged).**
  - `from_oifits([])` raises a `ValueError` naming the pipeline ("needs at
    least one OIFITS file") instead of a bare `IndexError`.
  - Checks no longer pass silently on a value that is not finite:
    `error_scale` (a NaN scale at any position), `divergences` (a NaN
    fraction) and `prior_bound` (a NaN fraction) now fail, fail and warn
    respectively; `field_of_view` and `mode_converged` warn on a NaN.
    `ring_count` treats a ring count whose BIC margin is not finite as not
    clearly preferred, warns when the chosen count is missing from its rows
    instead of raising `KeyError`, and `inclination(None, ...)` warns ("the
    inclination is unknown") instead of raising `TypeError`.
    `OrbitPipeline`'s `_optimum_gap` is NaN, not 0, when no draw can be
    scored and its worst gap and bound fractions keep a NaN from any band.
  - The fit report's `priors` entries read `LogUniform(low=1e-05, high=0.5)`
    instead of numpyro's `repr` with a memory address, so a rerun writes the
    same report.
  - `at_bound` and `start_at_bound` are one function in
    `virgil.pipeline._checks` (`at_bound(names, start=True)`; `start_at_bound`
    is gone). Check names, values and messages are unchanged.
  - Integer and positive settings reject `True` and `False` in every pipeline
    (`OrbitPipeline` already did).
  - `OrbitPipeline` uses the shared validation, posterior summary, sampler
    checks and `samples.h5` and figure writers: its band summaries take the
    worst R-hat and ESS NaN-aware, and its corner and trace figures become a
    placeholder with a warning when the chains never move. The search, its
    numbers, the summary keys and the run folder are unchanged.
- **`StarPipeline` reports a scanned (uniform-disk) fit that sits on its diameter bound.** Only gradient fits listed `at_bound`, so a uniform fit at the edge of `diam_range_mas` passed the fit stage silently; it now lists `diam` in `at_bound` and adds the `start_at_bound` warning, as the limb-darkened fit does.

- **Pipelines no longer crash in the posterior figures when the chains never move.** Draws with zero width made chainconsumer raise "Axis limits cannot be NaN or Inf", failing the run in the plotting step instead of reporting the non-convergence. The corner and trace figures are now placeholders with a recorded warning, and the R-hat and ESS checks fail as they should.

- **Pipeline sampler reports no longer hide a stuck parameter.** The worst R-hat and smallest ESS in `posterior` reports were taken with Python `max`/`min`, so a NaN at one site was dropped when a finite site came first and the R-hat and ESS checks passed. A NaN at any site now makes the worst value NaN, which those checks fail.

- **Pipeline setting validation is uniform.** `BinaryPipeline`, `StarPipeline` and `DiskPipeline` validate the shared sampler and data settings with one helper: `DiskPipeline` now rejects a non-positive `batch_size`, a non-number where a positive number is required is a `ValueError` (it was a `TypeError`), and messages for positive settings read "X must be positive; got V." (provisional pipelines; valid runs and results are unchanged). A missing or corrupt `models/best/info.json` now makes `DiskPipeline`'s `compare` stage raise instead of reporting an empty `at_bound`.

- **`epoch_positions` no longer discards a refinement that does not
  converge.** The damped-Newton refinement of a grid peak returned its start
  point unless it stopped as `CONVERGED`, although the kernel only accepts
  steps that lower the loss. A run that stalled (the damping limit) or
  reached its 200-step limit therefore left the peak at its grid cell. In the
  CI job on the oldest dependencies that left the true peak of a benchmark
  night at marginal -469 (-154 refined), below its point reflection (-361):
  a 21 mas error. `_refine_marginal` now keeps the best point reached and
  returns the kernel's stop code; only a start that is not finite falls
  back to the start. `epoch_positions` warns when the refinement of a night's
  best peak did not converge, and `estimate_primary_diameter` when one of the
  fits that set the diameter and its gain did not (the marginal score at that
  point is a lower bound; the warning names the stop). The refinements
  behind `fit_orbit_aliases` also keep an improved point where they used to
  fall back to the start, and since its default `primary_diam="auto"` calls
  `estimate_primary_diameter`, it can emit that warning too.
  Two things the fix exposed are corrected with it. `estimate_primary_diameter`
  took the point-primary score from one fit started at the grid's positions,
  which can end at a poor local maximum (33 nats below what the same model
  reaches from a disc's positions) and read as a resolved disc once refinements
  were no longer discarded; the baseline is now the better of that fit and a
  refit from the best diameter's positions. And `epoch_positions` no longer
  raises `LinAlgError` when the Hessian at a kept point is not finite: the
  covariance falls back to the step size.

- **The limb-darkening tutorial printed an understated χ²/N.** It divided the
  sum of squared whitened residuals by their number (234 rows), but
  four-telescope closure phases give more residual rows than independent data
  (162), so the printed values were 451.32 and 0.79 instead of about 652 and
  1.14. The tutorial now uses `virgil.imaging.chi2_by_observable`, which counts
  independent data, and its outputs were rerun.

- **The pipelines no longer fail with numpyro's "Cannot find valid initial
  parameters" on a non-detection.** With no companion the fitted flux runs to
  the lower end of its prior (and an unresolved star's diameter to the edge of
  `diam_range_mas`); NUTS then started exactly on the bound, where the
  unconstrained value is infinite. `BinaryPipeline`, `StarPipeline` and
  `DiskPipeline` now start the chains at least 1e-3 of the prior's width
  inside any bound (so a start within 0.1% of a bound, which `fit` reports in
  `at_bound`, moves slightly and gives different draws for the same seed); a
  start that is still not finite raises a `ValueError` naming the start
  values instead of numpyro's `RuntimeError`; and `BinaryPipeline` and
  `StarPipeline` add a `start_at_bound` warning check when the best fit sits
  at a bound.
- **Projected correlated closure phases keep only their real directions in
  float32.** When an operator (a kernel-phase or DISCO phase projection) is
  applied to closure phases with four or more telescopes, their correlation
  matrix is singular, and `OIData` rotates the operator onto the
  eigenvectors of the projected covariance with a cut at 1e-10 of the
  largest eigenvalue. That is below float32 eps, so the numerically zero
  directions were kept (33 to 36 outputs where 30 are real, for ten
  four-telescope frames), with a weight 10⁶ times that of a real one. The
  covariance and its `eigh` are now computed on the host in float64.
  Operators that already whiten the covariance (as DISCO operators do) are
  unchanged.
- **`OIData` drops samples with a zero error** (see Migration): a zero error
  gave an infinite whitened residual and made every likelihood and fit
  non-finite. A zero *projected* error raises a `ValueError`, also when a
  correlating operator would otherwise drop that direction silently.
- **`MultipoleDisk.is_physical` works with a `VonMises` azimuthal model.** It
  read `harmonics.prior`, so `model_loglike(..., reject_unphysical=True)` and
  `System.is_physical` raised `AttributeError` on any scene containing one.
- **`SharedFlux` grids are no longer misread as `(lo, hi, n)`.** A three-value
  tuple was a range whenever its last element was at least 1, so
  `SharedFlux(grid=(0.01, 0.1, 1.0))` became the single flux 0.01. The
  `(lo, hi, n)` form now needs an integer `n`; a list, an array or a tuple
  whose last element is a float is a list of flux values (the same for
  `slope`).
- **Bandwidth smearing for binaries, rotated scenes and resolved
  backgrounds.** With `bandwidth` set, `BinaryModelCartesian`,
  `BinaryModelAngular`, `Rotated` scenes and `System(star, bg=Resolved())`
  raised a `TypeError` saying only that Systems of Components are supported.
  They are now smeared, and an unsupported model, such as an `OrbitalBinary`
  that is not a snapshot, raises a `TypeError` that names it.
- **`fit(method="lbfgs")` restarts a round that crawls.** L-BFGS runs in
  rounds of 2000 steps and restarts from the last point with fresh curvature
  memory while a round that used all its steps still gains in the loss, instead
  of spending all 20 000 steps with a memory spoilt by a parameter running
  into a bound. On macOS the limb-darkened `StarPipeline(error_scale="fit")`
  fit crawled to the step limit with χ² at the fitted scales 94 against N =
  81; Linux converged in under 700 steps. Fits that converge, stall or time
  out within 2000 steps are unchanged, and `info["steps"]` is the total.
- **`fit(method="lm")` no longer accepts a step that raises the loss.**
  optimistix accepts a Levenberg–Marquardt step when the loss change is at
  most 1% of the *predicted* change, so a step whose predicted change is
  positive (a damped solve that came out wrong, which on macOS happened in
  about half of the fits of a noise-free limb-darkened star with a companion,
  started 5% from the truth) was accepted whatever it did to the loss: the
  companion's flux fell to 0, its offset ran to the prior bound, and the fit
  stopped there at χ² of order 10⁶ against 10⁻¹⁵ at the truth. Such a step is
  now rejected and the damping raised, so an LM fit never ends above its
  starting loss. Found by the independent validation suite
  ([virgil-validation#138](https://github.com/benjaminpope/virgil-validation/pull/138)).
- **Alias fits reach the likelihood ridge.** The L-BFGS-B refinement stopped on
  its relative-reduction test 50–800 nats short of the optimum on a
  many-cycle orbit's narrow ridge, so about one noise draw in four settled in
  a neighbouring alias. Each start now restarts from its own optimum until
  it stops improving, and a winning band with χ²/ν above 10 on the quoted
  errors (a missed ridge gives 65–800) is flagged `poor-fit` with a
  `UserWarning`; errors underestimated by more than 3× also trip it.
- **Differential-phase steps near ±π wrap the same in data and model.** A step
  of exactly π wrapped to -π in the data but, one ulp lower from `arctan2` on
  macOS, to +π in the model, a 2π disagreement. The model's steps are now
  wrapped to within π of the data's.
- **Isotropic priors have finite gradients at the poles in float32.**
  `IsotropicInclination.icdf` and `IsotropicLatitude.icdf` gave a NaN
  derivative within about 1e-8 of a pole, and their final clip a slope of 1/2
  at the bounds; both now keep the true derivative.
- **`read_oifits` reads closure triangles whose leg has no visibility row**
  ([#299](https://github.com/benjaminpope/virgil/issues/299)). A triangle
  leg stored in no `OI_VIS2`/`OI_VIS` row in either orientation raised a
  `ValueError`, so MIRC-X files and some ESO phase-3 PIONIER products could
  not be read. `OI_T3` is self-contained in the OIFITS standard, so such a
  leg now gets a flagged sample at the T3 row's own coordinates, shared by
  later triangles of the same frame. One `UserWarning` per file gives the
  number of legs placed this way. The `ValueError` remains when those
  coordinates are NaN or zero, or disagree with the triangle's other legs.
- **Parallel numpyro chains** (`chain_method="parallel"`) on data with
  `with_gains` no longer segfault: `GainModes` held zero-size arrays when no
  mode spans frames, and XLA's Shardy pass crashes compiling a `jax.pmap` that
  captures one (JAX 0.11.2). `GainModes.spanning` and `spanning_group` are
  `None` then. `numpyro_model` warns, suggesting `chain_method="vectorized"`,
  when the data still hold zero-size arrays (such as no closure phases) and
  more than one device is visible.
- **`limits.chi2ppf` works for any number of degrees of freedom on a standard
  install.** For `df != 1` it called numpyro's `gammaincinv`, which needs
  `tensorflow_probability`, not a virgil dependency. It now inverts
  `jax.scipy.special.gammainc` by Halley's method for every `df`, matching
  `scipy.stats.chi2.ppf` to about 1e-14 relative in float64 for p in [1e-10, 1
  - 1e-10], and stays differentiable and `jit`-able. It returns NaN for `df <=
  0`. Found by virgil-validation.
- **`DetectionMC.load` checks the file format** (see Migration).

### Removed

- **Unused helpers, with no caller in virgil or its analyses.**
  `virgil.tempering.lag_correlation`, `lag1_correlation` and
  `factor_from_correlation` (`estimate_factors` computes the lag statistics
  itself) and `chain_factor` (the exact AR(1) factor, now a helper of the
  tests); `virgil.plotting.plot_hmc_fisher_chainconsumer` (call
  `plot_chainconsumer_diagnostics` for each coordinate system); and
  `virgil.harmonics.image_harmonics_from_latents`
  (`image_harmonics_from_h(step_up(reflection_coefficients(re, im)))`). No
  result changes.

- **`virgil.spectra.Tabulated`.** The 0.3.0 entry promised its removal in
  0.4.0. Replace `Tabulated(ratio, wavel)` with `Nodes(ratio, wavel)`: the
  flux at every wavelength is the same (identical between the end nodes,
  within an ulp beyond them). Two things change. The free values are
  `flux.values`, not `flux.ratio` (fit and prior paths such as
  `"comp.flux.ratio"` become `"comp.flux.values"`), and the reference flux
  `spectrum()` (used by `render` and the zero-baseline normalization) is the
  value at `wavel0`, the first node, not the mean over the nodes; pass
  `wavel0=` to set it. Negative values now pass the constructor and are
  checked on the total flux, as for every other spectrum. Priors on
  `comp.flux.values` are therefore not checked for non-negative support (as
  `comp.flux.ratio` was): keep a non-negative prior (such as a log-uniform
  scale) if every channel must stay positive, and pass `flux_param=` where a
  grid or plot auto-detected the old key.

Nothing else is removed from the released 0.3 API in 0.4: every other
deprecation above works with a `FutureWarning` until 0.5 (the
`legacy.oifits_implaneia` helpers until 0.6). Dead and duplicated internals
are gone with no behaviour change.

### Provisional

virgil 0.4 has two tiers (see the [Stability page](docs/stability.md)).
**Stable**: a change needs a deprecation cycle. **Provisional**: may change in
any 0.4.x release without a deprecation, so that new features can settle.
Everything public that is not listed here is stable. Nothing warns at
runtime; provisional pages and API entries carry a "Provisional" notice, and
each module lists its provisional names in `__provisional__`, which
`tests/test_stability.py` checks. New public features start provisional.

Provisional in 0.4:

- the modules `virgil.epochs`, `virgil.aliases`, `virgil.arcs`,
  `virgil.orbit_search` and `virgil.tempering` (the orbit fitting, aliases and
  search; `orbit_search` ships provisional because orbit search is a core
  feature whose names are still settling);
- the `MultipoleDisk` family in `virgil.models`: `MultipoleDisk`, `ThinRing`,
  `GaussianRing`, `RadialPowerLaw`, `ExpTail`, `Gap`, `RadialSum`,
  `RadialLinear`, `RadialPower`, `RadialGP`, `RadialGPFunction`,
  `MultipoleKernel`, `Harmonics`, `VonMises`, `Cone`, `PowerLawHeight`,
  `multipole_disk_cvis` and `gp_priors`; and in `virgil.fields`,
  `MultipoleField` and `multipole_field_priors`;
- the Gaussian-process members of `virgil.ensemble`: `evidence_grid`,
  `gp_members`, `GPFit` and `EvidenceGrid`;
- `virgil.svi` (`variational`);
- all of `virgil.pipeline`: `BinaryPipeline`, `StarPipeline` and
  `DiskPipeline`. `BinaryPipeline` and `StarPipeline` were documented as
  stable and are now provisional, to be promoted after a release of use
  (`global_nsigma` is to be settled first). The run schema
  `virgil-pipeline-run-v1` is unchanged;
- `virgil.imaging.clean` and `CleanResult`;
- `virgil.metrics`;
- the `ft` argument of `Image`, `FlaredDisk*` and `image_visibilities`,
  `set_ft_defaults`, and the automatic thresholds and tolerances;
- the `rotation_deg` argument of `FlaredDisk*` (and its `"grid"` default) and
  the lattice tolerance for DISCO dictionaries;
- image smoothing: `Image(smooth_fwhm_mas=...)`, `Image.smoothed()` and the
  `smoothed` option of the regularizers;
- `virgil.crossval`;
- `virgil.fitting.gauss_newton_diag` and the `precondition` argument of `fit`;
- `virgil.plotting.plot_field_draws`;
- the module `virgil.radio_clean` (new in 0.4.1: `clean`, `RadioCleanResult`,
  `fit_beam`, `restore`), with the `uvtaper` argument of
  `ComplexVisData.dirty_image` and `dirty_beam`, `ComplexVisData.sigma_hat`,
  `CalibrationNuisance(offset_beams=)` and `term(sigma=)`;
- the module `virgil.radio` (new: complex visibilities from CASA measurement
  sets with per-observation calibration nuisances; instrument-neutral, with
  the spectral-index prior `Uniform(0, 5)` and the data-derived reference
  frequency and offset bound among its defaults).

## 0.3.0 (2026-10-06)

Everything listed as new was added after 0.2.0.

### Migration from 0.2.0

Most of 0.3.0 is additive. These are the changes that can alter an existing
analysis or warn:

- **`fit` with a `LogUniform` prior now returns the MAP in `log x`.** This
  covers priors on parameters and on `noise=` terms (#226). Each prior is
  optimized in its flat coordinate (`LogUniform(a, b)` in `log x`; isotropic
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
  so L-BFGS fits, regularized images especially, change slightly.
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
  offsets, marginalized flux scales, differential phases with a finite
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
  is 10). On regularized images, 10 stopped short of the optimum at many
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
  them (`fit` used to reject `VonMises`). The density is normalized in the
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
  (and optionally a polynomial in λ) marginalized analytically under a broad
  Gaussian prior; `likelihood.flux_scale_posterior` reports it. Differential
  phases use the exact arg V, with the continuum normalization (offset and
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
  a grid-marginalized `log_bayes_factor` against no companion (trapezoid prior
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
  no change, since the term's `loglike` is already normalized.

- **Marginalised RV zero points.** `RVData(..., instrument=labels)` and
  `RVData.term(params, marginalise_offsets=(mean, sd))` marginalize one
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
  supplied modes), marginalized analytically on the whitened closure phases,
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
  marginalizes them analytically (`virgil.gains`), and their widths can be
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
  `RVData` terms are fully normalized Gaussian log densities (like the OIData
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
- **`fit` optimizes each prior in its flat coordinate.** A prior that is
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

- **One home for analytic marginalization of linear parameters.**
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
    normalized scene the significance falls again once the companion
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
  the likelihood's model-dependent normalization, ½ log det of the
  marginalized nuisances' covariance factor (and, for extra observable
  blocks, their effective-error normalizer), exactly as `model_loglike`
  evaluates it. Before, it used the whitened χ² alone, so wider gains
  always looked better. Values for plain data are unchanged; the evidence
  still omits the data-only normalization (`-Σ log σ - ½ n log 2π`).

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
  default), the regularizers `MaxEntropy`, `TSV`, `TV` and `Centroid`,
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
  and sampling, with normalized Gaussian log-likelihoods.
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
- Independent phases use the exact von Mises normalizer, so fitted phase
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
