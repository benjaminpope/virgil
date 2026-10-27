# Radio interferometry

!!! warning "Provisional"
    May change in any 0.4.x release without a deprecation cycle.

This module is not final and claims no parity with MPoL or with published
DSHARP analyses.

[`virgil.radio`](api/radio.md) fits calibrated complex visibilities at
arbitrary (u, v) points, the data of any radio or millimetre array whose
visibilities are in a CASA measurement set. Any [source model](source_models.md) works, including a pixel
[`Image`](api/models/classes/image.md) whose Fourier transform runs through
the NUFFT backend. Each observation (observation) may have its own flux scale,
phase-centre offset and error scale, fitted together with the sky.

## Install

Reading a CASA measurement set needs `casatools`:

```bash
pip install "virgil-astro[radio]"
```

`casatools` publishes wheels for Python 3.12 and 3.13 in its newest releases
(older releases for 3.11) and none for 3.14, so the extra is skipped on
Python 3.14. `ComplexVisData.from_arrays` needs nothing extra.
`casatools` also needs CASA's measures data (see CASA's `casaconfig`) on
first use.

The extra brings `casatools>=6.6`. On some systems casatools 6.7.6 fails to
import with a missing `OPENSSL_3.0.1` symbol; casatools 6.7.0.31 works
(`pip install "casatools==6.7.0.31"`). virgil does not pin the version.

## Conventions

* The visibility is `V(u, v) = Σ I(x, y) exp(−2πi (u x + v y))`, with `x`
  towards East, `y` towards North and `u`, `v` in wavelengths: virgil's
  convention everywhere.
* The CASA `DATA` column is the **conjugate** of this `V`, with
  `u = UVW[0] ν / c` and `v = UVW[1] ν / c`. `read_ms` conjugates it for you.
  This was decided with a point source predicted by CASA's simulator
  (`tests/test_radio.py::test_read_ms_convention`, casatools 6.7.6.14): the
  data match virgil's model to 5×10⁻⁷ with the conjugation and differ by
  order 1 without it. (The equivalent statement without conjugation is
  `u = −UVW[0] ν / c`, `v = −UVW[1] ν / c`.)
* Only cross-correlations are read. A sample is one (row, channel), kept
  when the row and both parallel-hand polarizations (XX and YY, or RR and
  LL) are unflagged and both weights positive. The two polarizations are
  averaged with their weights, `V = (w₁V₁ + w₂V₂)/(w₁ + w₂)`, with weight
  `w₁ + w₂`.
* A weight is `1/σ²` for each of the real and imaginary parts (CASA's
  meaning). `WEIGHT_SPECTRUM` is used if present, otherwise `WEIGHT`.
* `obs` (`OBSERVATION_ID`) is relabelled 0..n−1 in sorted order, the
  originals kept in `obs_ids`. `ddid` numbers the sorted
  (`DATA_DESC_ID`, observation) pairs, so a concatenated set in which one
  `DATA_DESC_ID` spans several observations gets a label, and an error
  scale, for each; `ddid_ids` holds each label's original `DATA_DESC_ID`.

## Reading data

```python
from virgil import ComplexVisData

data = ComplexVisData.read_ms("target.ms", stride=100)  # every 100th row
```

`read_ms` raises if the selection holds more than one field (a mosaic); pass `field=` to pick one. It handles one ddid at a time, so memory is that of the largest ddid.
`select(ddids=..., obs=...)` and `subset(mask)` keep the labels, so a
calibration model built for the full data still fits.

## A model with calibration nuisances

The model at a point `k` of observation `e` is

`M_k = F g_e (ν_k/ν₀)^α exp(−2πi(δα_e u_k + δδ_e v_k)) T(u_k, v_k) V_scene(u_k, v_k)`

with the total flux `F` (Jy), relative gains `g_e`, phase-centre offsets
`(δα_e, δδ_e)` in mas, a spectral index `α` and an optional Gaussian taper
`T` (a convolution of the image). The reference observation has `g = 1` and no
offset. Each ddid label `d` has an error scale `s_d` multiplying its errors, fitted
with the Jeffreys prior `1/s` (or profiled out, below). The spectral index `α` has the prior
`Uniform(0, 5)` by default, which excludes the falling spectra of
synchrotron or free-free sources: pass `priors(alpha=(-3, 5))` for those, or
`alpha=False` to fix it at 0 (or `alpha=2.0` at that value). `CalibrationNuisance` holds this configuration
and the default priors:

```python
import numpyro.distributions as dist
from virgil import CalibrationNuisance, Image, fit

nuis = CalibrationNuisance(data, ref_obs=0)  # nu0_hz: the data's mean frequency
priors = nuis.priors()  # cal.flux, cal.gain, cal.offset, cal.sigma, cal.alpha
init = nuis.defaults()

def scene(values):
    return Image(values["log_brightness"], pixel_scale_mas=2.0)

def model(**values):  # the likelihood term holds the data
    return scene(values)

priors["log_brightness"] = dist.Normal(0.0, 3.0)  # per pixel, shape (64, 64)
init["log_brightness"] = ...                      # a starting image
term = data.term(scene, nuis, chunk_size=1_000_000)
result = fit(model, priors, data=(), likelihoods=[term], init=init, method="lbfgs")
```

Vector-valued sites (`cal.gain` has one entry per non-reference observation) are
ordinary array-valued parameters. Because the error scales are fitted, the
term reports a log normalization and `fit` uses L-BFGS.

**Conditioning.** `fit` optimizes in unconstrained coordinates, and a poorly
scaled coordinate makes L-BFGS crawl. The gains and error scales have the
improper Jeffreys prior `1/x` by default (flat in `ln x`, no box), so the
coordinate is `ln x`. The offset of observation `k` is uniform over three
beams (`offset_beams=3`; one beam is `2π b_k`, `b_k = 1/(2π q_max,k)`, `q_max,k` the
longest baseline of that observation), so a unit step of the coordinate is a few
phase turns at the longest baseline of each observation. An offset known to a
microarcsecond from long baselines and one known to a tenth of a milliarcsecond
from short ones then have curvatures of the same order, where the old
`±1/(2 q_min)` box (about 10⁴ mas) did not. The bound is a prior: an offset beyond it
is excluded and appears in `info["at_bound"]`, so widen `offset_beams` if it does.
The site values stay physical (`cal.gain`, `cal.offset` in mas, `cal.sigma`).
Explicit bounds, `priors(offset=(lo, hi))` in mas, `priors(gain=(0.5, 2))` or
`sigma=(0.1, 10)`, set the scale again.

**Loss.** Per unmasked point the term is `½|r|²/s_d² + 2 ln s_d` for
`r = (V − M)√w`, the two real parts counting as one complex point, so the total
is `½χ²` plus `log_norm = Σ_k m_k 2 ln s_d(k)`. The taper is
`T = exp(−(πθq)²/(4 ln 2))`, `θ` the FWHM in radians and `q = √(u² + v²)`; the
offset ramp has the sign of a component's `dra`, `ddec`.

**Profiled error scales.** `CalibrationNuisance(data, sigmas="profile")` has no
`cal.sigma` site. At every evaluation `s_d² = χ²_d(s=1) / (2 N_d)`, `N_d` the number
of unmasked complex points of ddid `d`. The loss of a point is
`½|r|²/s² + 2 ln s`, and for the Jeffreys prior this `s_d` is exactly the
maximum over `ln s_d`, so the profiled optimum of the other parameters is the
joint optimum, found with fewer coordinates and without the degeneracy
between `s` and the amplitude. The gradient is the partial derivative at `ŝ`
(the envelope theorem; `ŝ` is held by `stop_gradient`). `fit` returns `ŝ` in
`result.info["derived"]["cal.sigma"]` and
[`data.sigma_hat`][virgil.radio.ComplexVisData.sigma_hat] computes it for any
values. Held-out data are not scored with their own `ŝ`: pass the training
`ŝ` as `data.term(..., sigma=ŝ)`, and use `sigmas=True` with `cross_validate`. The loss it returns is `½χ²(ŝ) + log_norm(ŝ) = Σ_d N_d (1 + 2 ln ŝ_d)`, the profile of the joint loss, so losses compare across fits of the same data, profiled or not. The reported χ² is at `ŝ` (so it is `2 N_d` per ddid by construction): quote the
χ² at the quoted errors, `Σ_d 2 N_d ŝ_d²`, instead.

The term evaluates the points in chunks of `chunk_size` with `jax.lax.map`,
padded with zero weight, so memory is bounded by the chunk rather than the
number of visibilities.

The data live in the term as array leaves, so `fit` passes them to its jitted
functions as arguments: they are never compile-time constants, and fits that
differ only in the mask (`mask=fold != i`) or in a regularizer weight reuse one
compiled loss. Keep it that way in your own jitted code by passing the term as
an argument:

```python
import equinox as eqx
import jax.numpy as jnp

@eqx.filter_jit
def loss_and_grad(term, values):
    return eqx.filter_value_and_grad(
        lambda v: 0.5 * jnp.sum(term(v) ** 2) + term.log_norm(v)
    )(values)

loss, grad = loss_and_grad(term, values)
```

Do not write `jax.jit(lambda values: ... term(values) ...)`: a closure over the
term embeds every visibility in the compiled program (gigabytes at ALMA scale,
with JAX's "large amount of constants were captured" warning, slow compiles and
executables too big for the persistent cache). The same holds for a
`jax.jit(jax.value_and_grad(f))` where `f` closes over `term`.

For a fit with the scene on the model side only, `data.model(scene, values,
nuis)` returns `M_k` and `data.chi2_by_ddid(scene, values, nuis)` the χ² and
number of points of each ddid.

## Antenna-based gains

Self-calibration solves for the antenna gains and applies them; closure-only
imaging discards them. `CalibrationNuisance(data, antenna_gains=True,
solint="scan")` instead **marginalizes** one complex gain
`g = exp(λ + iφ)` per antenna and solution interval, so the posterior of the
image carries the calibration uncertainty. The model is
`V_k = g_i conj(g_j) M_k + noise`, with priors `λ ~ N(0, τ_a²)` and
`φ ~ N(0, τ_φ²)`, independent between antennas and intervals.

`solint` is `"int"` (each integration), `"scan"` or a number of seconds;
intervals never cross a scan. The data need `time`, `ant1` and `ant2` (and
`scan` for `"scan"`; `read_ms` fills them).

```python
nuis = CalibrationNuisance(data, antenna_gains=True, solint="scan")
term = data.term(scene, nuis)  # then fit(..., likelihoods=[term], method="lbfgs")
```

**Method.** At every evaluation the model `M_k` is computed once, then a few
(`gain_steps`, default 5) Gauss–Newton steps from `g = 1` give the gain
solution, and the integral over the gains is taken by a Laplace approximation
about it. The Hessian splits into an amplitude block and a phase block
(a weighted graph Laplacian), because the cross terms cancel; each adds
`½ log det`, from a batched Cholesky factorization. The gradient passes
through the unrolled steps. Centring on the solution, not on `g = 1`, is what
lets phase errors of about a radian be handled. Larger errors (about two
radians or more) can leave the solution in a local minimum; raise
`gain_steps` or calibrate first.

**The widths are choices.** `cal.gain_amp_width` (`τ_a`, default 0.1) and
`cal.gain_phase_width` (`τ_φ`, default 0.5 rad) are not measurements; the
default amplitude width is of the order of ALMA's absolute flux accuracy
(5–20%, by band), and no phase specification exists. Fix them, or fit them
(`fit_gain_widths=True`: log-uniform priors on 1e-3–10, overridable in
`priors`; an unbounded `1/τ` prior would be improper, since the marginal tends
to a constant as `τ → 0`).

- With `τ → 0` the likelihood is the ungained one (up to a constant); with
  `τ → ∞` it tends to closure-only inference. The finite phase prior is
  essential: without it the absolute position is lost, as with closure phases.
  A snapshot then constrains the source position only through `τ_φ`.
- The mean `λ` of an interval is degenerate with `cal.flux` and `cal.gain`;
  only `τ_a` identifies it, so `cal.gain` is pinned by the mean over
  intervals.
- Within an interval a per-observation `cal.offset` is a phase gradient that
  the gains can absorb: with antenna gains it is identified through `τ_φ`
  alone.
- An amplitude gain rescales the noise too; the model does not know it, so
  only the phase part of the closure equivalence is exact.
- There is no least-squares form (`fit(method="lm")` raises
  `NoLeastSquaresForm`; use `lbfgs` or `adam`, or `numpyro_model`). The
  `residuals` hold the data and the solved gains over their prior widths,
  and `log_norm` the log-determinants. `precondition="diag"` sees those rows
  and omits the log-determinant, so it is approximate.
- `data.model` and `dirty_image(scene=...)` apply no gains and raise with
  antenna gains; `term.gains(values)` returns the solved gains, and
  `chi2_by_ddid` is taken at the solution. `fit(...).info["derived"]` holds
  `cal.gain_step`, the size of the last Gauss–Newton step: if it is not small,
  raise `gain_steps`.
- `sigmas="profile"` takes `ŝ` at the gain solution and holds it fixed.
- Memory is `n_interval × n_ant²` floats per Hessian block; independent
  intervals only (gains correlated in time are a later stage).

Source: Blackburn et al. (2020), [references](references.md#blackburn2020);
Broderick et al. (2020), [references](references.md#broderick2020); Arras et
al. (2019), [references](references.md#arras2019).

## Blurring kernel: `blur_fwhm_mas`

`Image(..., blur_fwhm_mas=...)` and `CalibrationNuisance(blur_fwhm_mas=...)`
put a fixed Gaussian blurring kernel of that FWHM (mas) **in the forward
model**: the data are compared with the blurred model, not with a blurred
copy afterwards. The fit then recovers the *unblurred* sky, and a kernel
wider than the true structure is deconvolved, which can leave dots. The
default is `None`, which is recommended: fit the underlying image, regularize
it, and blur for display after the reconstruction. (Formerly `smooth_fwhm_mas`
and `taper_fwhm_mas`, deprecated aliases until 0.5. This is not CASA's
`uvtaper`, the data weighting of `dirty_image`.)

- `CalibrationNuisance` multiplies the visibilities of **any** scene (a
  parametric model, an `Image`, a `System`) by `exp(-π² θ² q² / (4 ln 2))`.
  It is a uv-plane factor, so a regulariser, which acts on pixels, never
  sees it.
- `Image` declares that the sky the data see is its pixels convolved with the
  Gaussian, as in MPoL's `GaussConvFourier` (Czekala et al. 2025,
  [references](references.md#mpol)). The visibilities are the same factor
  times the pixel transform (exact at any uv point for every `ft`), and
  `Image.smoothed()` returns the convolved pixels by an FFT on a
  zero-padded grid. A regulariser's `smoothed=True` option penalizes those
  pixels instead of the raw ones; the base image is then unregularized, so
  `fit` warns (`UserWarning`). MPoL does this; virgil's default does not.

```python
env = Image(eta, 5.0, blur_fwhm_mas=30.0)
reg = MaxEntropy(w, smoothed=True)  # entropy of the blurred image
```

`smoothed=True` clips FFT round-off at zero; `MaxEntropy` and `LogSum` then
use a flat default over all pixels, since blurring spreads flux beyond the
`support`. Setting both blurs the model twice, so `fit` and `data.term` warn
(`UserWarning`) when they meet.

**Accuracy.** `smoothed()` crops its result to the image, so flux spreading
past the edges is lost (keep the emission clear of them). For a kernel of at
least 3 pixels FWHM the transform of the smoothed pixels equals the analytic
Gaussian times the raw transform to about `exp(-π² θ² (1/p - q)² / (4 ln 2))`
of the flux (an upper bound, times the raw transform at the aliased
frequency), p the pixel size: about 1.5e-8 at FWHM 3 px and `q = 0.25 / p`.
Measured on a compact image in float64 with `|q| <= 0.25 / p`, the error is
3e-11 of the flux at 3 px. A narrower kernel is not band-limited on the pixel grid, and
the discretization error grows quickly; the visibilities, which use the
analytic factor, stay exact.

## Primary beam and cross-validation

`primary_beam(npix, pixel_scale_mas, freq_hz, dish_m, factor)` is the log
attenuation `−4 ln 2 r²/FWHM²` of a Gaussian primary beam with
`FWHM = factor × λ/dish_m`. Nothing is assumed about the telescope: take the
dish diameter and factor from its documentation. For example, ALMA's 12 m
antennas have `factor = 1.13` (ALMA Technical Handbook), and ATCA's antennas
are 22 m in diameter (ATNF). Add the result to an `Image`'s log-brightness so
that the image is the sky times the beam.

`dartboard_folds(data, k=5)` splits the data into `k` folds for
cross-validation by cells of the uv plane (annuli in q, sectors in the
azimuth mod π), so that a point and its Hermitian partner never straddle a
train/test split and most neighbouring points share a fold (points either
side of a cell edge can still fall in different folds, so scores can be
slightly optimistic). Fit with
`term(..., mask=fold != i)` and score the held-out points with
`chi2_by_ddid(..., mask=fold == i)`.

## Toeplitz χ² {#toeplitz-chi2}

A worked example, on simulated data, is the [radio imaging tutorial](radio_tutorial.md).

`fit` picks this form itself (`ComplexVisData.term(..., chi2="auto")`) when it
is exact: one fixed `Image`, `offsets=False`, a fixed `alpha`, no
`antenna_gains`, float64, `method="lbfgs"` or `"adam"`, and enough points per pixel
and ddid (`set_ft_defaults`; about 24 N² bytes of device memory per ddid, capped by `toeplitz_max_bytes`; `fit(...).info["likelihood_forms"]` says what ran). Otherwise it stays direct; `chi2="toeplitz"` demands it
and `chi2="direct"` forbids it. `numpyro_model` and the fits of `cross_validate` are always
direct. Build it by hand, as below, for fitted offsets.

When the scene is one [`Image`][virgil.models.Image], the χ² of each ddid
`d` (observation `e`) is a quadratic in the pixels,

`χ²_d = (C_d − 2 G_e Iᵀ D_d(δ_e) + G_e² Iᵀ B_d I) / s_d²`,

with `G_e = F g_e`, `C_d = Σ w|V|²`, the dirty image `D_d` at the
observation's offset `δ_e` and the dirty beam `B_d`, a Toeplitz matrix
applied by FFT on a 2N × 2N grid (Fessler et al. 2005). Precompute them once
and an evaluation costs a few FFTs of the image, whatever the number of
visibilities:

```python
import jax

jax.config.update("jax_enable_x64", True)  # float64 is required

nuis = CalibrationNuisance(data, ref_obs=0, alpha=False, offsets=False)  # or alpha=2.0
tdata = data.toeplitz(Image(init["log_brightness"], pixel_scale_mas=2.0), nuis)
term = tdata.term(scene, nuis)
result = fit(model, priors, data=(), likelihoods=[term], init=init,
             method="lbfgs", precondition="diag")
```

The term has the loss, gradient, profiled `ŝ` and `info["derived"]` of
`data.term(scene, nuis)` for the same points, but no residual vector: fit
with `"lbfgs"` or `"adam"` (`"lm"`, `gauss_newton_mass` and the Gauss–Newton
covariance raise `NoLeastSquaresForm`). With `precondition="diag"` the pixel,
flux and gain diagonal is the exact Gauss–Newton one of the direct term; the
offsets and error scales take the diagonal of the loss Hessian.

- **What it covers.** Per-observation gains, the flux, the nuisance blur, `Image(blur_fwhm_mas=...)`, a mask, and fixed, fitted or
  profiled error scales. The image layout (shape, pixel scale, rotation,
  smoothing) is fixed at the precompute; the scene must return that layout,
  alone or as the only part of a `System`, at the template's `dra`, `ddec`
  (a fixed shift, folded into the visibilities at the precompute).
- **What it refuses.** A fitted spectral index (pass `alpha=False` or a
  number), antenna gains, other components, float32, and fitted offsets
  unless asked for (below).
- **Checks.** `tdata.crosscheck(scene, values, nuis)` compares the χ² per
  ddid and the loss with the direct term; `tdata.direct_term(scene, nuis)` is
  that term. They agree to about 1e-13 of the loss.
- **New weights.** `tdata.rebuild(vis, weight)` repeats the precompute for
  new visibilities or weights without recompiling (bootstraps, reweighting).
  Weights must stay finite and positive; drop points through `mask` instead.

**Fitted offsets** (`offsets="spread"`, an extension). The
offset ramp `e^{−2πi u·δ_e}` cancels in `|M|²`, so `B_d` does not depend on
the offsets and only `D_d` moves. `tdata` stores each ddid's type-1 spread
grid `b` (nufftax's exponential-of-semicircle kernel `ψ`, Barnett et al.
2019), and Poisson summation gives the dirty image at any real position,

`D(x + δ) = Σ_l b_l e^{2πil(x+δ)/n_f} / ψ̂(x + δ)`,

a phase ramp on the stored grid divided by the kernel's transform at a
non-integer position (by quadrature, as in a type-3 NUFFT; Lee & Greengard
2005). It is exact to the kernel tolerance while no point's kernel wraps,
`max |t| ≤ π(1 − w/n_f)`, with `t` the baseline in radians per pixel. This
builds on type-3 NUFFT evaluation; we know of no prior use of this
combination, so it is opt-in:

```python
nuis = CalibrationNuisance(data, ref_obs=0, alpha=False)  # offsets fitted
tdata = data.toeplitz(template, nuis, offsets="spread")
```

It refuses baselines past `tdata.margin` of the pixel Nyquist limit
(`tdata.nyquist_fraction` is `max |t|/π`; use smaller pixels or mask the
longest baselines), and offsets beyond the precompute's bound
(`max_offset_mas`, by default that of `nuis.priors()`) raise at run time.
Without it, `B_d` and `D_d` need only integer lags and are exact at any
baseline.

### Performance

Time of one value-and-gradient evaluation of χ² for a pixel `Image`, by path. The full table
comes from a benchmark job (the simulated problem of the [radio imaging tutorial](radio_tutorial.md)
at several sizes, CPU and GPU, exact path with `offsets=False`); only the cells with a sourced
measurement are filled, and they do not all share one setup.

| Path | CPU | GPU (A100) |
| --- | --- | --- |
| direct, exact DFT (`ft="dft"`) | — | — |
| direct, nufftax backend (`ft="nufft"`, eps 1e-6) | — | 735 ms |
| Toeplitz (`chi2="toeplitz"`), exact path | — | — |
| Toeplitz with free offsets (`offsets="spread"`) | — | 21.8 ms (34× the nufftax cell) |

The two GPU cells are from one independent check on a real ALMA data set (virgil-validation, float64): 42.4 M points,
26 ddids, 1028² pixels of 5 mas, with gain and offset gradients.

## Diagnostics

Both diagnostics are model-free displays or checks: neither is part of the
likelihood.

### Error scale from time-adjacent samples

`data.sigma_scale(by="ddid")` estimates the error scale `s` (the factor on
the quoted errors `1/√w`) without a sky model. Samples of one baseline,
channel and scan are sorted by time and paired (1st with 2nd, 3rd with
4th, …); a pair has `Var(ΔRe) = s²(1/wᵢ + 1/wⱼ)`, so

`ŝ² = mean over pairs of ΔRe² / (1/wᵢ + 1/wⱼ)`,

the same for Im, and `ŝ` combines the two. Standard errors are the sample
errors of those means.

```python
res = data.sigma_scale()            # per ddid label; by="obs" or None also work
for label, s, se in zip(res.labels, res.s, res.s_se):
    print(f"ddid {label}: s = {s:.3f} ± {se:.3f}")
```

Caveats:

* **It is an upper bound.** A change of the sky, gain or phase between the
  two integrations adds to `|ΔV|²`, so `ŝ` is biased high where the data vary
  quickly (atmospheric phase, long integrations). Compare it with the fitted
  `cal.sigma`: a fit well below `ŝ` is expected, a fit above it is not.
* It needs `time`, `ant1`, `ant2` and `scan`, which `read_ms` fills (otherwise
  `from_arrays(..., time=, ant1=, ant2=, scan=)`); it raises if they are
  missing. With `read_ms(stride=k>1)` the rows are no longer adjacent.
* `lag_ratio` compares the mean-square difference of far pairs (half a run
  apart) to adjacent pairs. A ratio well above 1 means the pairing does what
  it should; near 1 means constant data or a time order that does not follow
  the data (for example a scrambled time column), so check `ŝ`.
* It sorts one ddid at a time with NumPy and no Python loop over points:
  memory is a few arrays of the largest ddid's length (see the PR notes for
  numbers).

### Dirty and residual images

```python
beam = data.dirty_beam(256, 0.5, weighting="briggs", robust=0.5)
dirty = data.dirty_image(256, 0.5, weighting="briggs", robust=0.5)  # Jy/beam
resid = data.dirty_image(
    256, 0.5, weighting="briggs", scene=scene, values=values, nuisance=nuis,
    use_sigma=True,
)  # image of data - model, in Jy/beam
```

The image is `Σ wₖ Re[Vₖ e^{+2πi(uₖx + vₖy)}] / Σ wₖ`, computed with the
adjoint (type-1) NUFFT of nufftax at the raw (u, v) (nothing is gridded onto
cell centres; the accuracy is that of
[`set_ft_defaults`][virgil._geometry.set_ft_defaults]'s tolerance); the real
part adds the Hermitian conjugates. Dividing by `Σ w` makes the dirty beam
`Σ w cos(2π(ux + vy)) / Σ w` peak at 1 at the phase centre, so the image is in
Jy/beam and a point source of flux `F` at the centre peaks at `F`.

* **Orientation** is that of `Image`: East to the left, North up, the phase
  centre at pixel `((npix − 1)/2, (npix − 1)/2)` (between pixels for an even
  `npix`, where the sampled beam peak is a little below 1: use an odd `npix` for a
  centred pixel), pixel `(row, col)` at `dra = (c − col)s`, `ddec = (c − row)s`.
* **Weighting**: `"natural"` is `w`; `"uniform"` is `w/W_c` with `W_c` the
  weight in the point's uv cell; `"briggs"` is `w/(1 + W_c f²)` with
  `f² = (5·10^−R)² / (Σ_c W_c² / Σ_k w_k)`, natural as `R → +∞` and uniform as
  `R → −∞`. The density is counted on a uv grid (cell `1/FoV` by default;
  Hermitian partners share a cell) and only reweights the points. `uvtaper`
  multiplies the weights by the Fourier transform of a Gaussian, so the beam is
  the weighted beam convolved with it.
* **Residuals**: with `scene`, `values` and `nuisance` the data minus the
  model (with the calibration nuisances applied, as in `term`) is imaged;
  `use_sigma=True` divides the weights by the fitted `s_d²`. `dirty_beam` has
  no `scene` or `use_sigma`: passing `values` and `nuisance` always divides its
  weights by `s_d²`, so match it to an image made with `use_sigma=True`.
* The beam is not deconvolved: this is a display and a check on a fit, never
  a likelihood. Large data are transformed in chunks of `chunk_size` points.
