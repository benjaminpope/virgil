# Detection statistics: definitions and simulation methods

[`virgil.detection`](api/detection.md) turns a companion search over a grid
into statistics whose null distribution you simulate, so that a candidate gets
a false-alarm probability and a non-detection a completeness. The
[Detection ROC curves](detection_roc.md) tutorial runs the whole recipe; this
page defines the quantities and explains the choices.

## The statistics {#statistics}

[`detection_statistics`][virgil.detection.detection_statistics] reduces a
search to three numbers measuring how strongly the data prefer a companion to
none:

- **`delta_chi2`**, the profile likelihood ratio
  $2[\max \log L - \log L_0]$ over positions and fluxes $\ge 0$ (so never
  negative). At each position the best grid flux is refined by BFGS, as in
  [`optimized_flux_grid`][virgil.grid_fit.optimized_flux_grid]; where the
  unconstrained best flux is negative, the constrained one is 0 and the
  position adds nothing.
- **`log_bayes_factor`**, the log evidence ratio of "a companion somewhere on
  the grid" to "no companion": the prior-weighted mean of $L/L_0$ over the
  grid. Positive values favour a companion.
- **`max_snr`**, the largest ratio of unconstrained best flux to its Laplace
  uncertainty, over positions.

The null hypothesis is the same model with the companion flux at zero and
everything else as given, so fixed known components (a resolved star, a disk,
a companion already found) stay in both. The function is traceable in the
data: inside `jax.jit` or `jax.lax.map` over simulated observations it
compiles once, as Monte Carlo false-alarm rates need.

### The prior of `log_bayes_factor` {#bayes-prior}

The prior weights sum to 1 and follow each axis's spacing, with trapezoid
weights in the grid index (1 inside, ½ at the ends). So the prior is uniform
over the searched box for evenly spaced coordinates, and uniform in log flux
for a log-spaced flux axis (in flux for a linear one). The trapezoid weights
make the Bayes factor converge as the grid is refined; equal weights would put
the prior's edges half a step beyond the axes and, since $L/L_0 \approx 1$
far from a companion, change the result at first order in the step.

### Grid resolution {#grid-resolution}

Every statistic uses the full likelihood grid, evaluated once. For
`log_bayes_factor` to approximate the evidence, the flux axis must resolve the
likelihood peak in flux and the coordinate spacing must resolve it in
position (not checked). A coarse grid still gives a valid test statistic once
its null distribution is simulated, but not an accurate evidence.

A strong detection has a narrow peak, of relative width (FWHM over flux)
about $2.4/\mathrm{SNR}$. Two steps across it at SNR 10 need a step of 0.12 in
$\ln f$, about 20 points per decade. Check the diagnostic `flux_peak_steps`,
that FWHM (2.355 times the Laplace uncertainty) over the local flux spacing:
below about 2 the axis does not resolve the peak and `log_bayes_factor` is
inaccurate. A grid sum over a Gaussian of width $\sigma$ with spacing $s$ is
off by about $2\exp(-2\pi^2\sigma^2/s^2)$ relative: $10^{-6}$ at
$s = 1.18\sigma$, but 20% at $s = 3\sigma$.

## Local significance and the look-elsewhere effect {#local}

At one position fixed in advance, `delta_chi2` follows $\tfrac12\delta_0 +
\tfrac12\chi^2_1$ under the null (Chernoff 1954, because the flux sits on its
boundary at 0; the $\chi^2$ law of a likelihood ratio is Wilks's theorem), so
its upper tail is the one-sided Gaussian tail at $\sqrt{\Delta\chi^2}$.
[`local_nsigma`][virgil.detection.local_nsigma] is
[`nsigma`][virgil.limits.nsigma] with one degree of freedom. A local 3σ
(`delta_chi2` = 9) is a one-sided false-alarm probability of 0.135%, not the
two-sided 0.27%.

Taking the best of many positions inflates the statistic (the look-elsewhere
effect; Gross & Vitells 2010, arXiv:1005.1891), so for a search `local_nsigma`
overstates the significance. The global false-alarm probability needs
simulations of the null over the same grid.

## Simulating the null {#null}

Simulators have the signature `simulate(key, scene=None) -> OIData` and are
traceable in `key` and `scene`.

**Gaussian** ([`gaussian_null`][virgil.detection.gaussian_null]). Each draw is
`template.with_model(scene, key=key)`: the scene's prediction plus Gaussian
noise from the template's errors, with closure phases correlated through
`cp_noise` and draws of the template's gains and closure-phase offsets if it
has them. `error_scale` makes the true noise a multiple of the template's
errors while the draws still carry the template's errors, like real data
analysed with mis-estimated errors, to test how the statistics fare then.

**Error rescaling** ([`rescale_errors`][virgil.detection.rescale_errors]).
Visibility and phase errors are scaled separately, each by
$s = \sqrt{\chi^2/n}$, where $\chi^2$ is that block's sum of squared whitened
residuals about the null scene (including the periodic penalty rows of
correlated closure phases) and $n$ its independent observables, as in
[`n_independent`][virgil.oidata.OIData.n_independent]. Each block then has
$\chi^2/n = 1$, exactly without gains and approximately with them. $n$ does
not subtract the parameters fitted to get the null scene, so with few data
and several fitted parameters the factors are biased low by about $p/(2n)$.

**Residual bootstrap** ([`bootstrap_null`][virgil.detection.bootstrap_null];
Efron 1979). The residuals about the null scene's prediction are whitened,
sign-flipped (`"sign_flip"`, a wild bootstrap, the default) or resampled with
replacement (`"resample"`), re-coloured, and added to the prediction of the
scene being simulated:

- visibilities are independent per sample, whitened as $r/\sigma$;
- unprojected phases are first wrapped into $[-\pi, \pi)$. Uncorrelated phases
  are whitened as $\Delta/\sigma$, which equals the likelihood's chord
  $2\sin(\Delta/2)/\sigma$ to $O(\Delta^3)$; both are odd in $\Delta$, so a
  sign flip flips the likelihood's whitened residual exactly;
- closure phases from four or more telescopes are whitened into independent
  combinations with `ClosureNoise.whiten`, flipped or resampled, and
  re-coloured with `ClosureNoise.colour`, keeping the triangles' correlations;
- projected phases (kernel or DISCO phases) are linear and not wrapped.

Sign flipping keeps each whitened residual's magnitude, so heteroscedastic or
mis-estimated errors survive into every draw, and it removes any mean offset.
Resampling mixes residuals of different samples (within the visibilities, and
within the phases). Caveats:

- The null must fit: residuals from a poor fit carry the misfit into every
  draw.
- A real companion in the data is part of the residuals, so it appears,
  scrambled, in every null draw and biases the null distribution (and the
  injections) upward.
- Only the closure-phase covariance model survives: correlations between
  visibilities, between baselines beyond the closure triangles, or between
  frames are lost. That model is the equal-baseline-noise approximation of
  Kammerer et al. (2020), so with unequal errors in a group the re-coloured
  residuals are not exact closures of baseline phases.
- Data with gains, closure-phase offsets or extra observables are not
  supported.

## Injection and recovery {#injection}

[`injection_recovery`][virgil.detection.injection_recovery] simulates
observations (null, or with an injected companion), searches each and keeps
the statistics and the best position. One compiled kernel,
`(key, injection) -> statistics`, serves every draw through
`jax.lax.map(..., batch_size=draw_batch)`, so memory stays bounded (there is
no `vmap` over grid × draws). The default grid `batch_size` is divided by
`draw_batch`, which is capped at that default, so the working set stays
within one search's usual budget. An explicit `batch_size` applies to each
of the `draw_batch` searches evaluated together, multiplying the working set
by `draw_batch`.

Draw `i` of the null (or of the injections) uses
`fold_in(fold_in(key, 0 or 1), i)`, so results do not depend on `draw_batch`
or `chunk_size`. `chunk_size`, the draws per kernel call, is rounded up to a
multiple of `draw_batch`. By default it is the smaller of 64 and the larger
number of draws, and the last chunk is exactly as long as the draws left (one
extra compilation, no waste). If you set it, the last chunk is padded
instead, so the compilation is reused across calls with different numbers of
draws, at the cost of up to `chunk_size - 1` discarded draws.

With `match_radius`, an injection counts as detected only if its best
position lies within that many mas of the injected one. The grid then needs
coordinate keys ending in `dra` and `ddec` (Cartesian, mas) or `sep` and `pa`
(mas and degrees, as in `BinaryModelAngular`).

## The result {#result}

A [`DetectionMC`][virgil.detection.DetectionMC] holds plain NumPy arrays. A
value's false-alarm probability is the fraction of null draws at or above it,
`(k + 1)/(n + 1)` for `k` of `n` (the Monte Carlo p-value, never 0 and
conservative), with an exact Clopper–Pearson interval. At a threshold `t` a
draw is detected when its statistic exceeds `t` (and, with `match_radius`, its
best position matches). Separations are in mas: `hypot(dra, ddec)` of the
injections, or their `sep` for angular grids. Fluxes are relative to the
primary, as in [`absil_limits`][virgil.limits.absil_limits]. Fawcett (2006)
reviews the ROC curve and its area. The bootstrap for a threshold's error
works in batches of about $2^{22}$ values, so its memory does not grow with
`n_boot × n`.

## Reading the detection plots {#detection-plots}

[Detection ROC curves](detection_roc.md) describes the three plots of a
[`DetectionMC`][virgil.detection.DetectionMC] (simulated searches from
[`injection_recovery`][virgil.detection.injection_recovery]); here are their
conventions and what to look at.

**Null distribution** ([`plot_null_distribution`][virgil.plotting.plot_null_distribution]).
The curve is the fraction of companion-free simulations at or above each
threshold, that threshold's false-alarm probability, on a log scale so the
tail that sets thresholds is visible. The reference curve is the statistic at
one position fixed in advance ([local significance](#local)) for `delta_chi2`
and `max_snr`, and none for `log_bayes_factor`. The gap between the two is
the look-elsewhere effect of searching a grid. The legend sits outside the
axes so it never hides the tail; leave room for it in subplot grids.

**ROC curves** ([`plot_roc`][virgil.plotting.plot_roc]). Each curve follows a
threshold from high to low, with its false-positive rate (the false-alarm
probability) on the x-axis and its true-positive rate (the completeness) on
the y-axis. A statistic that cannot tell companions from noise lies on the
chance curve TPR = FPR; a better one lies above it. The x-axis is logarithmic
by default, from $0.5/n_\text{null}$ to 1, since detections are claimed at
small false-alarm probabilities, so the chance curve is curved. `mark_wilks`
marks [local significances](#local) (default 3σ and 5σ): a vertical line at
the false-alarm probability of each, and a circle on each `delta_chi2`
or `max_snr` curve where the threshold equals that significance
(`delta_chi2` = 9 or `max_snr` = 3 at 3σ). The circle's horizontal distance
from the line is the look-elsewhere effect. With several statistics, line
style distinguishes the statistics and colour the flux or separation
selections.

**Completeness map** ([`plot_completeness`][virgil.plotting.plot_completeness]).
Each cell is the fraction of injected companions of that separation and flux
detected at the false-alarm probability `fap`
([`DetectionMC.completeness`][virgil.detection.DetectionMC.completeness]).
Cells span halfway to their neighbours in separation and log flux; zero-flux
injections have no Δmag or contrast and are left out. The contours are the
empirical contrast curves of
[`DetectionMC.contrast_curve`][virgil.detection.DetectionMC.contrast_curve]:
in each separation bin the completeness is made non-decreasing in flux and
interpolated linearly in log flux (linearly in Δmag), and a contour is absent
where the injected fluxes do not bracket its level. The map fixes the axis
limits to its cells, so curves drawn afterwards, such as Absil limits from
`plot_contrast_curve`, do not widen them; `plot_contrast_curve` redraws the
legend, so place it again at the end.

## References

- Chernoff (1954), *Ann. Math. Statist.* 25, 573, doi:10.1214/aoms/1177728725.
- Efron (1979), *Ann. Statist.* 7, 1, doi:10.1214/aos/1176344552.
- Fawcett (2006), *Pattern Recognit. Lett.* 27, 861, doi:10.1016/j.patrec.2005.10.010.
- Gallenne et al. (2015), *A&A* 579, A68, arXiv:1505.02715.
- Gross & Vitells (2010), *Eur. Phys. J. C* 70, 525, arXiv:1005.1891.
- Kammerer et al. (2020), *A&A* 644, A110.
- Mawet et al. (2014), *ApJ* 792, 97, arXiv:1407.2247.
- Wilks (1938), *Ann. Math. Statist.* 9, 60, doi:10.1214/aoms/1177732360.
