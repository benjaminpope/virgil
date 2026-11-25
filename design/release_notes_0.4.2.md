# virgil 0.4.2

```bash
pip install --upgrade virgil-astro==0.4.2
```

> **Radio imaging is experimental in this release.** The IM Lup regularizer-weight gate failed its convergence
> criterion (see [Radio](#radio)). The gate carries to 0.4.3 with its criteria unchanged.

0.4.2 brings the radio work of 0.4.1's plan (an exact Toeplitz χ² that `fit` picks on its own where it applies, a
CLEAN that no longer diverges on sparse coverage, measurement-set input), a large audit of numerical and verdict
bugs, and code that is shorter to read. Everything new ships **provisional**: no feature is promoted to stable in
this release. Default changes are in the CHANGELOG and the tables below.

## Headlines

- **Toeplitz χ² is automatic.** `ComplexVisData.term(chi2="auto")` (the default) lets `fit` use the exact dirty-beam
  χ² of [`virgil.radio_toeplitz`](https://benjaminpope.github.io/virgil/radio_interferometry/#toeplitz-chi2) when the
  fit has one fixed `Image`, no antenna gains and float64, and the data are large enough to pay for it. Its cost
  does not depend on the number of visibilities. `chi2="direct"` and `chi2="toeplitz"` choose explicitly.
- **`radio_clean`** converges on sparse coverage: the minor cycle's PSF reaches `psfpatch` or the far edge of the
  image, as in CASA, and `clean` ends on a diverging residual.
- **`ComplexVisData` and `read_ms`**: radio visibilities from a measurement set, with `CalibrationNuisance`
  (per-antenna gains, spectral index) and `fit_staged_nuisances` (image, nuisances, then both, in three stages).
- **`kkt_report`** checks a fit's first-order optimality, bound by bound, and `fit` reports a parameter stuck at a
  prior bound (`stop="bound"`).
- **`LinearImage`**, a pixel image whose brightness is linear in its parameters, and **`DofDiscrepancy`**
  (`LCurve.dof_discrepancy`), a regularization weight chosen by the degrees-of-freedom discrepancy principle.
- **`group_outlier_mixture`**, a per-group (night) outlier-mixture likelihood.
- **`ArchimedeanSpiral`**, a pinwheel spiral model, and **`rotation_deg="grid"`**, which renders pixel images on a
  rotated uv lattice (see [Rendered images and CLEAN outputs change](#rendered-images-and-clean-outputs-change)).
- **Adam reports convergence.** `fit(method="adam")` runs in jitted chunks, so `time_limit` and `progress` apply to
  it, and `info["converged"]` and `info["stop"]` are set as for LM and L-BFGS.

## Added (not headlines)

Provisional, with no independent check yet:

- `Crescent` and `Blurred(source, fwhm)`; `noise_spec` (instrument noise specs); the observable nuisance modes
  (`DifferentialPhase`, `FluxSpectrum`, the correlated-flux marginalization of `GainModes`, closure-phase modes
  spanning frames).
- `fit(method="lbfgsb")`, `fit(post_step=)`, `virgil.enable_compilation_cache`.
- `ComplexVisData.term(chi2=...)`, `set_ft_defaults(toeplitz_*)`, `fit(adam_eps=, ftol_abs=, ftol_window=)`,
  `info["loss_history"]`, `virgil.radio_toeplitz` (opt-in `offsets="spread"`), `CalibrationNuisance(alpha=<float>)`.
- A radio imaging tutorial on simulated data, and a pull-request gate on new single-use private helpers.
- The **new orbit path** (`orbit_cells`, `mode_search`, `night_summaries`, `smc`) is provisional and **under
  census**. `OrbitPipeline` does not use it (see [Known issues](#known-issues)).

## Rendered images and CLEAN outputs change

- **Rotated grids.** Pixel renderings of `GravityDarkenedStar` on a rotated grid
  (`Image.from_model(..., rotation_deg=θ)`, θ ≠ 0) and under `Rotated(...)` were drawn unrotated, shifted and
  shrunk; the in-plane blur of `ModulatedGaussianRim` and `MultipoleDisk` now follows the grid rotation too.
  Analytic `model()` visibilities are unchanged; pixelized versions, and fits that use them, change. Unrotated grids
  are bit-identical. `clean` with no components now reports the grid's `rotation_deg`.
- **CLEAN restore and mask (D7).** `restore` centred its kernel half a pixel off on even-sized images, shifting
  restored components and lowering the peak by about 4 per cent; it now sits on pixel `n // 2` (odd sizes, all that
  `clean` accepts, are unchanged). `usemask="auto-multithresh"` follows Kepley et al. (2020, arXiv:1912.04970,
  Eqs. 1–4): seeds are `median + max(noisethreshold × rms, sidelobethreshold × sidelobe × |peak|)` of the residual,
  and source-free noise maps no longer seed masks. Masks and cleaned images change.
- **Departures from Kepley et al. (2020), documented in #762.** Three remain: the 0.5 × peak cap on the sidelobe
  level (#551); a percolated sidelobe level instead of Eq. 2; region growth on the beam-smoothed residual. Making
  the masking paper-exact, with a census, is planned for 0.4.3 (#763).

## Behaviour changes that now raise or warn

| If you | In 0.4.2 | Do this |
|---|---|---|
| call `error_scale(by_observable=False)` on data with gains, closure offsets, a finite `prior_width` or nuisance modes (D6) | raises `ValueError` (it silently mis-scaled) | Estimate the scale on data without them; the weighting is [#777](https://github.com/benjaminpope/virgil/issues/777) (0.4.3). |
| run `DiskPipeline` with `error_scale="auto"` on such data | falls back to the quoted errors **with a warning**; an explicit `error_scale="rescale"` raises | Check the warning; pass `"quoted"` to silence it. |
| call `cv_task` or `cross_validate` with `sigmas="profile"` and no fixed `sigma` (D5) | raises `ValueError` (the held-out fold was scored at its own σ) | Pass a fixed `sigma` or `sigmas=True`; scoring at the training σ̂ is 0.4.3. |
| call `apply_noise_spec` with a spec that names an instrument (D11) | requires `insname=` and refuses data whose `FluxSpectrum` carries nuisance modes | Pass `insname=`. |
| call a `DetectionMC` method with `stat="max_snr"` on an `injection_recovery(uncertainty=False)` run (F32) | raises `ValueError` (it returned `1/(n+1)`, `(nan, nan)` or 0.0); `concatenate` refuses to mix `uncertainty` settings | Run with `uncertainty=True`, or use another statistic. |
| call `detection.rescale_errors` on data with gains, phase offsets or extras | raises `ValueError` ([#776](https://github.com/benjaminpope/virgil/pull/776); it ignored them) | Rescale data without them; the weighting is [#777](https://github.com/benjaminpope/virgil/issues/777) (0.4.3). |
| `fit` a noise term that is a centred `hierarchical_scales` member (S5) | warns: the MAP has no finite optimum (Neal's funnel) | Use a non-centred parameterization; the default changes in 0.4.3. |
| call `flux_scale_posterior` with only a coupled `corrflux` spectrum | the spectrum is skipped unless requested (it raised `NotImplementedError`) | Request it explicitly. |

Verdicts that can change: `error_scale` judges each scale on its own N-aware limits; `OrbitPipeline` has a
`sampler_budget` check; a failed check of a `StarPipeline` model that was not selected is a warning. `ClosureNoise`
now correlates triangles that share a reversed `OI_T3` leg (D1). Grid-fit `converged` flags can change (flux
convergence is judged against −½χ²).

## Upgrading from 0.4.1

| If your code… | In 0.4.2… | Do this |
|---|---|---|
| fits an Adam model that stops at `max_steps` | `info["converged"]` is `False` and the fit warns (it was `None`, silent). | Raise `max_steps`, set `ftol_abs`, or filter the warning. |
| fits one radio `Image` with enough data | `fit` uses the Toeplitz χ² (exact; agrees with the direct term to about 1e-13 of the loss). | Pass `chi2="direct"` to `term` for the old path. |
| passes `CalibrationNuisance(alpha=1)` | it raises: an integer used to fit α. | Write `alpha=True` to fit or `alpha=1.0` to fix. |
| uses `MultipoleField` with the default `r_min_mas=0` | order `m` is scaled by `tanh(r/length_r)^|m|` (`length_r` is five node spacings by default), so only `m = 0` survives at the centre. The default prior is suppressed for orders m ≥ 1 within about five spacings of the centre, whatever `n_r`, so MAP fits and evidences move. | Refit; nothing to change in code. |
| passes `Image(smooth_fwhm_mas=)` or `CalibrationNuisance(taper_fwhm_mas=)` ([#565](https://github.com/benjaminpope/virgil/pull/565)) | both are now `blur_fwhm_mas=`. The old names are deprecated aliases until 0.5 and raise a `FutureWarning`. Results are bit-identical. | Rename the keyword. |
| reads `fit(...).values["...log_brightness"]` of a free-pixel `Image` | it is the log of the pixel fluxes (`Σ exp = 1`), the same on every platform. Images, χ² and penalties are unchanged to rounding. | Nothing, unless you compared raw log-brightness across runs. |

Other rows (all in the CHANGELOG's "Changes to default behaviour"):

| If you | What changes | What to do |
|---|---|---|
| call `fit_orbit_aliases` or `OrbitPipeline` | `temper=False` is now the default, and results change: default orbit results can cut the true cycle count. | Pass `temper=True` for the old behaviour. |
| use `TSV(weight=w)` or `TV` | `beam_mas` is required, so the old call raises `TypeError`. | Pass `beam_mas`. |
| use `BinaryPipeline` | `flux_range` is `[1e-5, 1]`; `global_nsigma` comes from simulated nulls, is capped at 3.09σ and `n_trials` is removed; limits use Ruffio et al. by default. | `limit_method="absil"` restores the old limits. |
| pass `outlier_eps` or `drop_outlier_night`, or read `night_dropped` or `modes.robustness` | the keywords are ignored with a `FutureWarning`; the check and fields are removed. | Drop them. |
| fit with `regularizers` and a model returning one model per dataset | it raises. | Fit one dataset, or fold the regularizers into the model. |
| install virgil | `blackjax>=1.7.1` is a new hard dependency and the floor is `jax>=0.9`; environments pinned to jax 0.8 will not install. | Upgrade jax. |

## Radio

**Radio imaging is experimental in 0.4.2:** `radio_clean`, `ComplexVisData` / `read_ms`, and the radio path that
chooses the regularization weight by the degrees-of-freedom discrepancy (`DofDiscrepancy`). They are marked in
[`docs/stability.md`](https://benjaminpope.github.io/virgil/stability/). Use them, but check results against an
independent imager.

Radio imaging (`virgil.imaging` MaxEnt on ALMA data) ships as experimental in 0.4.2. Its pre-registered
regularizer-weight rule (virgil-validation, criteria commit `432aa3a`) failed its convergence criterion on the IM Lup
visibilities: the fast-loop fits at w/s² ≤ 316 stopped at `kkt_w` of 1.8e-4 to 7.8e-4 against the required 1e-4, so no
weight was chosen and the criteria were not changed. The gate carries to 0.4.3
([virgil#534](https://github.com/benjaminpope/virgil/issues/534),
[virgil-validation#292](https://github.com/benjaminpope/virgil-validation/issues/292)).

Trust status of each piece, from the radio register:

| Item | Status | Independent check |
|---|---|---|
| Full-batch L-BFGS, float64 | established | virgil-validation (IM Lup MaxEnt, MPoL reference) |
| Diagonal Gauss–Newton and entropy metric | derived | `test_radio_innovations.py`; not yet in pinned evidence |
| Nuisance parameterization | established | none yet |
| Cold starts: CLEAN start and staging | established | CLEAN against CASA; staging has none |
| Cold starts: coarse-to-fine | novel, experimental | none |
| Toeplitz likelihood | established | Toeplitz against direct term; the IM Lup A100 run is not yet in evidence |
| Free offsets (`offsets="spread"`, opt-in) | derived (type-3 NUFFT; the combination is not in the literature we found) | none |
| Marginalized gains, Stage 1 (opt-in) | established | none yet |

## Known issues

- **`OrbitPipeline` still uses the old orbit path**, with its tuning history. The new orbit modules
  (`orbit_cells`, `mode_search`, `night_summaries`, `smc`) are under census and are not used by it; the census gate
  moves to 0.4.3 ([#743](https://github.com/benjaminpope/virgil/issues/743)). **The Gl 229 orbit-gate passes at 0.4.0
  and 0.4.1 are unverified.**
- **Sampler issue [#580](https://github.com/benjaminpope/virgil/issues/580)** is open.
- **Open xfails.** Strict: [#747](https://github.com/benjaminpope/virgil/issues/747) (tempering), two in
  `test_night_flags`, [#751](https://github.com/benjaminpope/virgil/issues/751) (the `kkt_report` band-limit gate at
  10 mas pixels), [#580](https://github.com/benjaminpope/virgil/issues/580), and two for `nufftax` 0.6.1 under nested
  `vmap`. [#745](https://github.com/benjaminpope/virgil/issues/745) is not an xfail: the orbit-cell
  `ln Z` check is restricted to the cells that matter, because `ln Z` of negligible-mass cells disagrees across error
  scales.
- **`position_cells` can return NaN `p_pos` (with `failed` set)** for a system whose alias cell next to the true period
  has an importance-sampling ESS below 10 and a likelihood-peak bound too loose to prove it negligible. Seen on the
  two-season, 5-sigma-jitter test, seed 0, where that cell is 42 nats below the true one. The result is flagged, not
  silently wrong. Tracked for 0.4.3 (orbit audit, census-gated).
- **Nightly [#574](https://github.com/benjaminpope/virgil/issues/574) is red** on two lowest-dependency-pin tests,
  `test_the_shape_gate_decides_on_both_sides_of_the_threshold[data2-4.0-True]` and
  `test_optimized_flux_grid_converges_everywhere_in_float32`; the fix is on `main`, not yet in this candidate.
- **A staged fit can report success with an unconverged stage 3.** `fit_staged_nuisances`'s `failed` ignores
  stage-3 convergence; the fix is [#771](https://github.com/benjaminpope/virgil/pull/771), planned for 0.4.3.
- **Antenna-gain phase prior (D4).** The prior on antenna-gain phase and its test tolerances are a known limitation;
  the review is planned for 0.4.3.
- **Radio imaging is experimental**, as above.
- **Closure phases near a visibility null** ([#578](https://github.com/benjaminpope/virgil/issues/578), which
  supersedes the closed #309). A closure
  phase jumps by π where a model visibility passes through zero, so its gradient is unbounded there. An image fit
  that drifts to such a null can stall.

## Smaller and easier to read

Measured on `src/virgil` (v0.4.1 against the release candidate) with `scripts/code_shape_metrics.py` and a count of
**total** docstring lines (sections such as Parameters and Returns included). The CI cap in
`tests/lint/test_docstring_length.py` counts prose lines only, and no docstring exceeds it:

| | 0.4.1 | 0.4.2 |
|---|---|---|
| Docstrings over 60 lines in total | 0 | 3 (`fit`, `variational`, `dof_discrepancy`) |
| Docstrings over 40 lines | 30 | 42 |
| Docstring lines in total | 14,040 | 16,210 (+15%) |
| Private helpers referenced once (raw text count) | 418 | 234 (−44%) |
| Source lines in total | 56,790 | 68,681 (+21%) |

The package is longer in total: the radio, orbit-cell and audit work added more than the refactor batches removed,
and three docstrings now pass 60 lines in total. Numbers are re-run at the tag.

Full details are in [CHANGELOG.md](https://github.com/benjaminpope/virgil/blob/v0.4.2/CHANGELOG.md).
