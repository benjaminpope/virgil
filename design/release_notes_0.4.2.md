# virgil 0.4.2

```bash
pip install --upgrade virgil-astro==0.4.2
```

> **Draft.** The IM Lup cold-start gate has not passed (see [Radio](#radio)). This release is not tagged until
> it does.

0.4.2 brings the radio work of 0.4.1's plan: an exact Toeplitz χ² that `fit` picks on its own where it applies,
Adam that reports convergence, and a CLEAN that no longer diverges on sparse coverage. Most of the rest is code
shape: the package is shorter to read, with 193 fewer single-use private helpers.

## What's new

- **Toeplitz χ² is automatic.** `ComplexVisData.term(chi2="auto")` (the default) lets `fit` use the exact dirty-beam
  χ² of [`virgil.radio_toeplitz`](https://benjaminpope.github.io/virgil/radio_interferometry/#toeplitz-chi2) when the
  fit has one fixed `Image`, no antenna gains and float64, and the data are large enough to pay for it. Its cost
  does not depend on the number of visibilities. `chi2="direct"` and `chi2="toeplitz"` choose explicitly.
- **Adam reports convergence.** `fit(method="adam")` runs in jitted chunks, so `time_limit` and `progress` apply to
  it, and `info["converged"]` and `info["stop"]` are set as for LM and L-BFGS. New keywords `adam_eps`,
  `ftol_abs` and `ftol_window` (a plateau stop), and `info["loss_history"]`. The trajectory is unchanged step for
  step.
- **`radio_clean` converges on sparse coverage.** The minor cycle's PSF reaches `psfpatch` or the far edge of the
  image, as in CASA, and `clean` ends on a diverging residual instead of running on.
- **`virgil.radio_toeplitz`** (provisional): the precomputed dirty beams and images of a one-`Image` radio fit,
  after Fessler et al. (2005), with opt-in free per-observation offsets.
- **A radio imaging tutorial on simulated data.**
- **Code shape.** Single-use private helpers are inlined across the package, and CI fails a pull request that adds
  one.

## Upgrading from 0.4.1

| If your code… | In 0.4.2… | Do this |
|---|---|---|
| fits an Adam model that stops at `max_steps` | `info["converged"]` is `False` and the fit warns (it was `None`, silent). | Raise `max_steps`, set `ftol_abs`, or filter the warning. |
| fits one radio `Image` with enough data | `fit` uses the Toeplitz χ² (exact; agrees with the direct term to about 1e-13 of the loss). | Pass `chi2="direct"` to `term` for the old path. |
| passes `CalibrationNuisance(alpha=1)` | it raises: an integer used to fit α. | Write `alpha=True` to fit or `alpha=1.0` to fix. |
| uses `MultipoleField` with the default `r_min_mas=0` | order `m` is scaled by `tanh(r/spacing)^|m|`, so only `m = 0` survives at the centre. The default prior changes near the centre, so MAP fits and evidences move. | Refit; nothing to change in code. |
| passes `Image(smooth_fwhm_mas=)` or `CalibrationNuisance(taper_fwhm_mas=)` ([#565](https://github.com/benjaminpope/virgil/pull/565)) | both are now `blur_fwhm_mas=`. The old names are deprecated aliases until 0.5 and raise a `DeprecationWarning`. Results are bit-identical. | Rename the keyword. |
| reads `fit(...).values["...log_brightness"]` of a free-pixel `Image` | it is the log of the pixel fluxes (`Σ exp = 1`), the same on every platform. Images, χ² and penalties are unchanged to rounding. | Nothing, unless you compared raw log-brightness across runs. |

Everything else runs unchanged.

## Added

- `ComplexVisData.term(chi2=...)`, `set_ft_defaults(toeplitz_min_points=, toeplitz_points_per_pixel=,
  toeplitz_max_bytes=)`.
- `fit(adam_eps=, ftol_abs=, ftol_window=)`, `info["loss_history"]`, and `time_limit` and `progress` for Adam.
- `virgil.radio_toeplitz` (provisional): `ComplexVisData.toeplitz`, `crosscheck`, `rebuild`, and the opt-in
  `offsets="spread"`.
- `CalibrationNuisance(alpha=<float>)` fixes the spectral index. `ComplexVisData.chi2_by_ddid` takes
  `chunk_size`.
- `fit` accepts likelihood terms with a scalar χ² and no residual vector, for L-BFGS and Adam.
- A pull-request gate on new single-use private helpers (`scripts/code_shape.py --since`), and code-shape metrics in
  the size comment.

## Fixed

- `radio_clean.clean` no longer diverges on sparse coverage (#551).
- `fit` gauge-fixes the log-brightness of a free-pixel `Image`: LM's QR solve used to drift along the constant
  direction by rounding noise.
- Two tests that failed on some CI runner CPUs (#528) compare at tolerances the platforms agree on.

## Radio

All radio code stays **provisional**. Trust status of each piece, from the radio register
(`design/radio_imaging_innovations.md`, section B):

| Item | Status | Independent check |
|---|---|---|
| Full-batch L-BFGS, float64 | established | virgil-validation (IM Lup MaxEnt, MPoL reference) |
| Diagonal Gauss–Newton and entropy metric | derived | `test_radio_innovations.py`; not yet in pinned evidence |
| Nuisance parameterization | established | none yet |
| Cold starts: CLEAN start and staging | established | CLEAN against CASA; staging has none |
| Cold starts: coarse-to-fine | novel, experimental | none |
| Toeplitz likelihood | established | Toeplitz against direct term; the IM Lup A100 run is not yet in evidence |
| Free offsets (`offsets="spread"`, opt-in) | derived (type-3 NUFFT; the combination is not in the literature we found) | none |
| float32 δV loss | derived, experimental; no code on `main` | the method alone |
| Marginalized gains, Stage 1 (opt-in) | established | none yet |
| Ordered subsets | established; not built | none |

**IM Lup cold-start gate: not passed.** The gate asks that a cold start beat MPoL's objective and raw χ²/N, converge,
leave under 1% dead pixels, and recover the CLEAN flux to 10%. The last run reached raw χ²/N 2.7046 against MPoL's
2.7062, but its image fails: dead pixels left by the Gaussian start leave edge arcs and a ripple across the disk.
The Adam arms are pending.

The dead-pixel criterion is judged on the 30 mas tapered image, the image the regulariser acts on and the one MPoL
exports. The untapered base image's dead-pixel count is logged alongside it. The base image can collapse into dots
because the taper multiplies only the model, so lowering the regularization weight deconvolves the taper under
positivity.

> Gate result: _to be filled in before the tag._

## Known issues

- **Closure phases near a visibility null** ([#309](https://github.com/benjaminpope/virgil/issues/309)). Unchanged
  from 0.4.1: a closure phase jumps by π where a model visibility passes through zero, so its gradient is unbounded
  there. An image fit that drifts to such a null can stall.
- **Radio is provisional**, as above, and the cold-start gate has not passed.

## Smaller and easier to read

Measured on `src/virgil` with `design/code_shape_metrics.py`:

| | 0.4.1 | 0.4.2 |
|---|---|---|
| Docstrings over 60 lines | 0 | 0 |
| Docstrings over 40 lines | 30 | 22 |
| Docstring lines in total | 14,040 | 13,647 (−3%) |
| Private helpers called from one place | 418 | 225 (−46%) |
| Source lines in total | 56,790 | 57,813 (+2%) |

The source is longer in total because of the radio work (Toeplitz, CLEAN) and the new Adam driver; the refactor
batches shorten what they touch. Numbers are from `origin/main` at the time of writing and are re-run at the tag.

Full details are in [CHANGELOG.md](https://github.com/benjaminpope/virgil/blob/v0.4.2/CHANGELOG.md).
