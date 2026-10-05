# Detection statistics, false-alarm rates and ROC curves

Status: **Stages 1 and 2 built**, 2026-10-05. Tracks the last open item of
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
  detected 50% or 90% of the time at 0.135% FAP (a one-sided 3σ; see the
  FAP convention below), next to the Absil and Ruffio limits that virgil
  already computes.

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
   - **FAP convention.** Because the flux is bounded at zero, only upward
     fluctuations count, and the tail of ½δ₀ + ½χ²₁ above Δχ² is
     ½ P(χ²₁ ≥ Δχ²), the one-sided Gaussian tail at √Δχ². So a local "3σ"
     (`local_nsigma` = 3, Δχ² = 9) is a FAP of 0.135%
     (`scipy.stats.norm.sf(3)`), and 5σ (Δχ² = 25) is 2.9×10⁻⁷. The often
     quoted 0.27% is the two-sided value, P(χ²₁ ≥ 9), and is not the FAP of
     a non-negative flux. Everywhere in virgil "nσ" means the one-sided
     tail, and a FAP at "3σ" means 0.135%.
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
    phases correlated through `cp_noise`, and the template's gains and
    closure-phase offsets drawn if it has them). `error_scale` scales the
    noise but not the errors the draws carry, to test mis-estimated
    errors; `rescale_errors` instead scales the errors themselves so that
    the null's χ²_r = 1, separately for visibilities and phases;
  - **residual bootstrap**: residuals of the data about the null's
    prediction (phases wrapped into [−π, π)), whitened (visibilities by
    σ, uncorrelated phases by σ, closure phases from four or more
    telescopes with `ClosureNoise.whiten`), sign-flipped or resampled with
    replacement, re-coloured (`ClosureNoise.colour`) so that the
    triangles' correlations survive, and added to the prediction of the
    scene being simulated. Sign flipping keeps each whitened residual's
    magnitude, so heteroscedastic or mis-estimated errors survive, and is
    the default. It needs the null to fit the data, is biased when a real
    companion is present, loses correlations beyond the closure-phase
    covariance, and inherits that covariance's equal-baseline-noise
    approximation (`_closure.py`). Data with gains, closure-phase offsets
    or extra observables are refused, since their noise is not
    independent per sample.
- Injections are the companion model at the injected (position, flux),
  observed with either noise model; null draws are the same model at zero
  flux, which must predict what `null_scene` predicts (checked).

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

### Built (Stage 2): simulators, Monte Carlo and `DetectionMC`

```python
gaussian_null(template, null_scene, *, error_scale=1.0)
bootstrap_null(data, null_scene, *, method="sign_flip" | "resample")
    # each -> simulate(key, scene=None) -> OIData
rescale_errors(data, null_scene) -> (data, {"vis": s_vis, "phi": s_phi})
injection_grid(separations, fluxes, n_pa, key)  # dict of dra, ddec, flux
injection_recovery(template, null_scene, model, samples_dict, key, *,
                   n_null, injections=None,
                   noise="gaussian" | "bootstrap" | simulator,
                   match_radius=None, flux_param=None, draw_batch=1,
                   chunk_size=None, batch_size=None, progress=True)
    -> DetectionMC
```

- The simulators are equinox Modules, so `jax.jit` traces their arrays
  (the template's included) rather than baking them in as constants; they
  are traceable in the key and the scene.
- One jitted kernel, (draw index, injection) → statistics, serves null and
  injected draws alike (a null draw is an injection of zero flux). Within
  a call of the kernel, `jax.lax.map(..., batch_size=draw_batch)` maps
  over the draws; draws run in Python chunks of `chunk_size`, the last one
  padded, so the kernel compiles once for both kinds of draws, and for
  later calls with other numbers of draws when `chunk_size` is fixed. A
  `tqdm.auto` bar counts the chunks when tqdm is installed. Draw `i` uses
  `fold_in(fold_in(key, 0 for null or 1 for injected), i)`, so results do
  not depend on the chunking.
- The best position of every draw is stored (`best_dra`, `best_ddec`,
  `best_flux`, named by the last dotted part of each grid key), with the
  diagnostics `flux_peak_steps` and `converged_fraction`. `match_radius`
  (mas) is kept in the metadata and applied by `DetectionMC`.

`DetectionMC`, plain NumPy:

- fields `null`, `injected` (statistics, best positions and the injected
  values) and `meta` (grid, model, null-scene and template fingerprints,
  noise model, `match_radius`, numbers of draws, seeds, virgil version);
- `false_alarm_probability(stat, value, *, confidence=0.95)` → the
  empirical (k + 1)/(n + 1) with the Clopper–Pearson interval for
  P(null ≥ value);
- `threshold(stat, fap, *, n_boot=200, seed=0)` → the null's 1 − fap
  quantile and its bootstrap error, warning when fap × n < 1;
- `detected(stat, fap)`: statistic above the threshold (and matched);
- `roc(stat, flux=None, sep_bin=None)` → `(fpr, tpr, thresholds)` and
  `auc(...)`;
- `completeness(stat, fap, sep_bins=None, flux_bins=None)` → a dict of
  the detection-fraction map, counts and bin labels (by default each
  distinct injected separation and flux is a bin);
- `contrast_curve(stat, fap, completeness=0.5, ...)` → `(sep, flux)`,
  interpolating the running maximum of completeness linearly in log flux,
  in flux relative to the primary as `absil_limits` gives;
- `save` / `load` (one `.npz`, with the metadata as a JSON string) and
  `concatenate`, which refuses results that differ in grid, model, null
  scene, template, noise model or `match_radius`, or share a seed.

### Planned

Stage 3, plotting in `plotting.py`: `plot_roc` (log FPR option, marking the
FAP of Wilks 3σ and 5σ, 0.135% and 2.9×10⁻⁷, to show the look-elsewhere
offset), `plot_completeness` (with Absil/Ruffio curves),
`plot_null_distribution` (with the ½χ²₁ reference and the observed value's
FAP).

A candidate fourth statistic, or a cross-check of `log_bayes_factor`:
`grid_fit.linear_flux_grid(..., prior=...)` now returns a closed-form
linearised log Bayes factor per position. Not used yet.

## Stages

Stacked PRs into main.

0. This design note.
1. **Statistics**: `detection_statistics` and `local_nsigma`, with tests:
   Δχ² ≥ 0; with a one-point grid, Gaussian nulls follow ½δ₀ + ½χ²₁; log B
   rises with injected flux, is stable under grid refinement and matches a
   brute-force marginalisation; one compilation across draws.
2. **Simulators, Monte Carlo driver and `DetectionMC`** (built), with
   tests: one compilation across draws; seeded reproducibility; `concatenate` and the
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

### Stage 2 (2026-10-05)

- Built `gaussian_null`, `bootstrap_null`, `rescale_errors`,
  `injection_grid`, `injection_recovery` and `DetectionMC`, and added
  `ClosureNoise.colour`, the inverse of `whiten` on the covariance's
  column space (exact round trip for any whitened vector, and for any
  draw of `sample`).
- Deviations from the plan:
  - The simulators take `(key, scene=None)`, not just a key, so that the
    same simulator serves null and injected draws, and they are equinox
    Modules, so their arrays are traced arguments of the kernel.
  - Null draws are the companion model at zero flux, not `null_scene`
    itself, so null and injected draws share one kernel; the two must
    predict the same data to 10⁻³σ on the template, or
    `injection_recovery` raises.
  - `error_scale` scales the simulated noise (`with_model`'s
    `noise_scale`), not the errors the draws carry: scaling both would
    leave the statistics nearly unchanged, and the errors can already be
    changed on the template (`with_error_scale`, `rescale_errors`).
  - Added `chunk_size`: the kernel runs over fixed-size chunks of draws
    in a Python loop (a `tqdm.auto` bar), because a `lax.map` over all
    the draws would recompile for every number of draws, so null and
    injected runs of different sizes would compile twice.
  - `noise` also accepts a simulator, to pass `error_scale` or
    `method="resample"` without more keywords.
  - `rescale_errors` returns the scaled data and the factors; extra
    observables keep their errors. Its degrees of freedom are
    `n_independent`'s, not reduced by the null's fitted parameters.
  - The bootstrap whitens wrapped residuals Δ linearly (Δ/σ, and
    `ClosureNoise.whiten` of Δ), which equals the likelihood's chord and
    sine forms to O(Δ³); a sign flip flips the likelihood's whitened
    residual exactly for uncorrelated phases.
- Corrected the FAP convention in this note: a local 3σ is 0.135%, the
  one-sided tail of ½δ₀ + ½χ²₁, not the two-sided 0.27%.
- Tests (`tests/test_detection.py`, small grids on a laptop): over a 7 × 7
  grid, 64 nulls give a Δχ² threshold of 4.7 at 10% FAP against Wilks's
  1.64, whose empirical FAP is 55%; 96 nulls against 96 zero-flux
  injections on a 3 × 3 grid give AUC 0.5 ± 0.15 for all three
  statistics, and a 20σ companion gives AUC 1; the Clopper–Pearson
  interval covers a known 5% tail in ≥ 93% of 400 synthetic sets; the
  bootstrapped closure phases of 150 datasets with unequal errors whiten
  to unit covariance.
