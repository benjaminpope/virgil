# Detection statistics, false-alarm rates and ROC curves

Status: **Stage 1 built**, 2026-10-05. Tracks the last open item of
[virgil#2](https://github.com/benjaminpope/virgil/issues/2), "ROC curves from
evidence / injection recovery".

## Goal

A companion search over a grid ends with a number: the best Δχ², a peak
SNR, an evidence ratio. On its own the number does not say how often noise
alone would give a value that large, nor how often a real companion of a
given flux would be found. Both are needed:

- **after a detection**, the empirical false-alarm probability (FAP) of the
  observed statistic, which includes the look-elsewhere effect of searching
  many positions, next to the local Wilks significance;
- **after a non-detection**, the completeness at a fixed FAP, e.g. the flux
  detected 50% or 90% of the time at 0.27% FAP, next to the Absil and Ruffio
  limits that virgil already computes.

virgil will measure both by Monte Carlo for a given observation (a template
`OIData`) and search grid:

- the **false-positive rate** of each statistic on companion-free
  simulations;
- the **true-positive rate** on simulations with injected companions, as a
  function of flux and separation;

and combine them into ROC curves (true- against false-positive rate as the
threshold varies), thresholds at a chosen FAP, completeness maps and
empirical contrast curves.

## Decisions

### Statistics

Three statistics, all from one grid evaluation, so the ROC curves can show
which discriminates best:

1. **Δχ²**, the profile likelihood ratio,
   Δχ² = 2 [max over the grid of log L(position, flux ≥ 0) − log L₀].
   - The flux at each position is the best grid flux refined by BFGS, the
     `optimized_flux_grid` machinery, constrained to be non-negative: where
     the unconstrained best flux is negative, the constrained one is 0 and
     the position adds nothing. So Δχ² ≥ 0.
   - At one position fixed in advance, Δχ² follows ½δ₀ + ½χ²₁ under the
     null (Chernoff 1954; the flux sits on its boundary at zero). Its upper
     tail is the one-sided Gaussian tail at √Δχ², which is
     `limits.nsigma` with one degree of freedom: the **local Wilks
     significance**. It is always labelled local, because searching a grid
     makes the maximum larger (the look-elsewhere effect).
   - The **empirical FAP** from simulated nulls over the same grid (Stage
     2) replaces it as the global significance.
2. **log B**, a grid-marginalised log Bayes factor of "one companion
   somewhere on the grid" against "no companion":
   log B = logsumexp over the full (coordinates × flux) grid of
   [log L − log L₀ + log w].
   - The prior weights w sum to 1 and follow each axis's own spacing:
     trapezoid weights in the grid index along every axis (1 inside, ½ at
     the two ends). The prior is then uniform over the searched box for
     evenly spaced coordinates, and uniform in log flux between the axis's
     ends for a log-spaced flux axis (uniform in flux for a linear one).
   - The flux axis must resolve the likelihood peak for the sum to
     approximate the integral. A diagnostic, `flux_peak_steps`, gives the
     peak's full width at half maximum (2.355 Laplace σ) at the best
     position in local flux-axis steps, and a warning fires below 2 (a grid
     sum over a Gaussian with spacing s is off by about
     2 exp(−2π²σ²/s²) relative, ~10⁻⁶ at two steps per FWHM). The
     coordinate spacing must resolve the peak too; that is not checked.
   - A coarse grid still gives a valid *test statistic*, calibrated by its
     simulated null distribution, but not an accurate evidence.
3. **max SNR**, the largest unconstrained best flux over its Laplace
   uncertainty (`laplace_flux_uncertainty_grid`), over positions: the
   significance map of the composition tutorial. It is cheap and is a
   useful comparison in the ROC curves.

### Null hypothesis and noise models

- The null is the companion model with its flux set to zero and every
  other parameter as given. Fixed known components (a resolved star, a
  disk, a companion already found) therefore sit in both hypotheses, so the
  same tool answers "is there anything else?".
- Two null noise models (Stage 2):
  - **Gaussian**, from the template's errors (`OIData.with_model`, closure
    phases correlated through `cp_noise`), with an optional rescaling of
    the errors so that the null's χ²_r = 1, separately for visibilities and
    closure phases;
  - **residual bootstrap**: residuals of the data about the null's
    prediction, whitened (closure phases with `ClosureNoise.whiten`, so the
    triangle correlations survive), resampled or sign-flipped, and
    re-coloured. Sign flipping keeps each residual's own variance and is the
    default. It needs the null to fit the data, is wrong when a real
    companion is present, and loses correlations between baselines beyond
    the closure-phase covariance.
- Injections add a companion at (position, flux) to the null scene and
  observe it with either noise model.

### Nuisances

Nuisance parameters (the null scene's parameters, error inflation) are held
at their fit to the real data for now. Refitting them for every draw comes
later (Stage 5).

## API

### Built (Stage 1): `virgil.detection`

```python
detection_statistics(data, model, samples_dict, *, flux_param=None,
                     batch_size=None) -> dict
local_nsigma(delta_chi2) -> array
```

- `model` and `samples_dict` are as for the grid tools: a class called with
  the grid keys (`BinaryModelCartesian`) or a template with dotted paths
  (`"comp.dra"`), and a non-negative flux axis found by the usual
  `flux_param` rule.
- The result holds scalars: `delta_chi2`, `log_bayes_factor`, `max_snr`;
  one entry per grid key (e.g. `dra`, `ddec`, `flux`) giving the position
  where Δχ² is reached and the non-negative best flux there (flux 0 and the
  first grid position when Δχ² = 0); and the diagnostic `flux_peak_steps`.
- It is traceable in `data`: under `jax.jit` or `jax.lax.map` over
  simulated `OIData` it compiles once. Value checks that need concrete
  numbers (the flux-resolution warning, a best flux above the axis, the
  optimizer's convergence) run only on concrete data; under tracing, use
  `flux_peak_steps`.
- Internally the full likelihood grid is evaluated once
  (`grid_fit._likelihood_grid`); its best flux per position seeds the BFGS
  refinement (`grid_fit._refine_flux_grid`, factored out of
  `_optimize_flux_grid` without changing the public grid functions), and
  the Laplace uncertainties come from
  `grid_fit._laplace_flux_uncertainty_grid`.

### Planned

Stage 2, simulators and the Monte Carlo driver:

```python
gaussian_null(template, null_scene, *, error_scale=1.0)
bootstrap_null(data, null_scene, *, method="sign_flip" | "resample")
rescale_errors(data, null_scene)
injection_grid(separations, fluxes, n_pa, key)  # dict of dra, ddec, flux
injection_recovery(template, null_scene, model, samples_dict, key, *,
                   n_null, injections, noise="gaussian" | "bootstrap",
                   match_radius=None, draw_batch=..., batch_size=None)
    -> DetectionMC
```

- One jitted kernel, key and injection → statistics, run with
  `jax.lax.map(..., batch_size=draw_batch)`: one compilation, bounded
  memory (no vmap of grid × draws).
- `match_radius` (mas, optional): an injection counts as recovered only if
  the best position is within it. By default anything detected counts, the
  usual ROC definition; the best positions are stored either way.

`DetectionMC`, a plain-NumPy result container:

- fields `null`, `injected` (statistics plus injection parameters) and
  `meta` (grid, noise model, number of draws, seeds, virgil version);
- `false_alarm_probability(stat, value)`, the empirical (k + 1)/(n + 1)
  with a binomial interval;
- `threshold(stat, fap)`, the null quantile with a bootstrap error;
- `roc(stat, ...)` → `(fpr, tpr, thresholds)` and `auc(...)`;
- `completeness(stat, fap, sep_bins, flux_bins)`;
- `contrast_curve(stat, fap, completeness=0.5 | 0.9)`, in the units of
  `absil_limits` so the two overplot;
- `save` / `load` (npz plus JSON metadata) and `concatenate`, so array jobs
  can be merged.

Stage 3, plotting in `plotting.py`: `plot_roc` (log FPR option, marking the
FAP of Wilks 3σ and 5σ to show the look-elsewhere offset),
`plot_completeness` (with Absil/Ruffio curves), `plot_null_distribution`
(with the χ²₁ reference and the observed value's FAP).

## Stages

Stacked PRs into main.

0. This design note.
1. **Statistics**: `detection_statistics` and `local_nsigma`, with tests:
   Δχ² ≥ 0; with a one-point grid, Gaussian nulls follow ½δ₀ + ½χ²₁; log B
   rises with injected flux, is stable under grid refinement and matches a
   brute-force marginalisation; one compilation across draws.
2. **Simulators, Monte Carlo driver and `DetectionMC`**, with tests: one
   compilation across draws; seeded reproducibility; `concatenate` and the
   save/load round trip; the empirical FAP threshold exceeds Wilks's on a
   multi-point grid; AUC ≈ 0.5 for zero-flux injections and → 1 at high
   flux; the bootstrap preserves the closure-phase covariance on average.
3. **Plotting and a tutorial** (`notebooks/detection_roc.ipynb` →
   `docs/detection_roc.md`), run on a cluster rather than a laptop: a
   detection (observed Δχ² and log B, empirical FAP against Wilks nσ) and a
   non-detection (completeness map and empirical contrast curve against
   Absil/Ruffio, for both nulls). Link it from the binary-search and
   contrast-limits tutorials, then close virgil#2.
4. **Campaign and independent check**: a cluster array job over seeds that
   writes `DetectionMC` files and merges them; an issue on
   [virgil-validation](https://github.com/benjaminpope/virgil-validation)
   for an independent FAP and completeness check (its planned binomial
   test at the nominal rate), not built inside virgil.
5. **Later**: refitting nuisances for every draw; a Laplace-at-peak
   evidence as a further statistic if wanted.

Example (aperture masking): for a 7-hole mask with 21 V² and 35 closure
phases, a companion at 70 mas with flux 5×10⁻³ (about 5σ in flux) gives
Δχ² ≈ 30 and log B ≈ 11 on a 120 × 120 mas box with a linear flux axis; at
one fixed position Wilks would call that 5.4σ, and the look-elsewhere
correction over the box is what Stage 2 measures.

## Log

### Stage 0 and 1 (2026-10-05)

- Built `virgil.detection` with `detection_statistics` and `local_nsigma`;
  factored `_refine_flux_grid` out of `grid_fit._optimize_flux_grid` and
  made `_best_grid_flux` take a computed grid, so the detection kernel
  evaluates the full grid once. The public grid functions are unchanged.
- Two refinements of the plan:
  - **Trapezoid prior weights** instead of equal weights per grid point.
    L/L₀ stays near 1 far from a companion, so equal weights, which put the
    prior's edges half a step beyond the axes' ends, changed log B at first
    order in the step (10.61, 10.72, 10.78, 10.81 for 7 to 49 positions per
    axis in a test case). With trapezoid weights the prior's bounds are the
    axes' ends and the same case gives 10.845, 10.842, 10.844, 10.844.
  - **The resolution criterion** is two flux steps across the peak's FWHM
    (σ ≳ 0.85 steps), not two steps per σ, since a grid sum over a
    Gaussian is already accurate to ~10⁻⁶ at that spacing.
- Tests (`tests/test_detection.py`): at one position, 200 Gaussian nulls
  gave 51.5% zeros and a KS p ≈ 0.2 for the positive part against χ²₁.
  Each call compiles once per grid shape; a 200-draw `lax.map` over nulls
  on a one-point grid runs in a few seconds on a laptop CPU.
