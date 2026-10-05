# Design note: sparse image reconstruction

## Purpose and scope

This note plans support for **sparse** reconstructions in virgil: images
made of a few compact features, or of structure that is compact in some
basis. Sparse imaging suits scenes that the smoothness priors of
`design/image_reconstruction.md` (maximum entropy, TSV, the Gaussian-process
field) handle badly: point-like companions, knots and sharp edges next to
extended emission.

Two existing codes set the scope:
- **SQUEEZE** (Baron, Monnier & Kloppenborg; github.com/fabienbaron/squeeze),
  whose L0 regulariser is the standard sparsity prior in optical
  interferometry;
- **CLEAN** (Högbom 1974), the basic image-restoration algorithm of radio
  astronomy.

## The obstacle: an L1 penalty on the pixels is constant

An [`Image`][virgil.models.Image]'s pixels are ``b = softmax(η)`` over its
support: positive and summing to one. So ``‖b‖₁ = 1`` for every image, and
an L1 penalty on the pixels does nothing. Positivity and a fixed flux
already contain the L1 constraint; that is why SQUEEZE uses L0, not L1.
The softmax also means pixels approach zero but never reach it, so a MAP
image is never exactly sparse.

Sparsity therefore has to come from one of:
1. **L1 in another basis**, such as the starlet (isotropic undecimated
   wavelet) coefficients of the image. TV is already this: an L1 norm of the
   image's gradient, smoothed by ε.
2. **A non-convex penalty on the pixels**, a smooth surrogate for L0, such as
   the log-sum penalty Σ log(1 + b/ε) (Candès, Wakin & Boyd 2008).
3. **A linear parameterisation of the pixels** (b ≥ 0, directly), with a
   solver that handles a non-smooth penalty (proximal gradient). This is the
   only route to exact zeros.
4. **A greedy algorithm** that adds point-like components one at a time:
   CLEAN.

## SQUEEZE: what it has, and what virgil does with it

SQUEEZE 3.0's README lists the regularisers L0, total variation, Laplacian,
maximum entropy and "dark energy". Its engine is parallel simulated
annealing (or parallel tempering) with Metropolis–Hastings moves of discrete
flux elements between pixels. It fits SPARCO and binary-plus-image models,
images polychromatically, models bandwidth smearing and computes marginal
likelihoods. Its papers describe wavelet regularisation in a compressed
sensing framework.

| SQUEEZE feature | virgil | Plan |
| --- | --- | --- |
| Total variation, maximum entropy | `TV`, `MaxEntropy` | nothing |
| L0 | none | `LogSum`, a smooth surrogate (S1) |
| Laplacian | none | `Laplacian`, quadratic so LM can fit it (S1) |
| Wavelet sparsity | none | `StarletL1` and `starlet` (S1) |
| Dark energy | none | deferred: its definition is not in the README; check the source first |
| SPARCO, binary + image | `System` + `Image` + `spectra.PowerLaw` | nothing |
| Marginal likelihood | `log_evidence` for Gaussian forms | not for L1/L0, which have no Laplace-approximation evidence |
| Bandwidth smearing | deferred in `image_reconstruction.md` | separate item |
| Annealing / tempering of flux elements | NUTS and MAP | not ported: discrete moves get nothing from gradients or JAX |

The non-convex penalties have local minima. SQUEEZE escapes them by
annealing; virgil uses starting points instead: a parametric fit, the dirty
image, or a CLEAN model (S2).

## CLEAN

**Classic Högbom CLEAN** repeatedly finds the peak of the residual dirty
image and subtracts a fraction (the loop gain) of the dirty beam there. It
needs data that are linear in the image: complex visibilities with calibrated
phases. With closure phases it needs hybrid mapping (self-calibration loops),
which fits virgil badly and is fragile on sparse optical and AMI coverage.
virgil's [`dirty_image`][virgil.imaging.dirty_image] is a least-squares
estimate, so its beam is not the plain uv point-spread function either. We do
not implement classic CLEAN.

**Gradient CLEAN** is what we build. For linear data, the residual dirty image
is proportional to −∂χ²/∂c, the gradient of χ² with respect to the flux of a
point added at each pixel, and Högbom's peak search picks its most negative
pixel. Written that way, CLEAN needs only the gradient of virgil's likelihood,
which JAX gives in one backward pass for any data: closure phases, kernel
phases, DISCOs, V², or a mix. Each iteration:
1. computes ``g = ∂χ²/∂c`` over the pixels, where ``c ≥ 0`` are the
   components' fluxes;
2. picks the pixel ``p`` in the support whose Gauss–Newton step lowers χ²
   most, the largest ``g_p² / ‖J e_p‖²`` with ``g_p < 0``, and stops if no
   pixel lowers χ²;
3. adds ``gain × Δ`` to ``c_p``, where ``Δ = −g_p / (2‖J e_p‖²)`` is the
   Gauss–Newton step, from one Jacobian–vector product;
4. stops when χ² per data point reaches a target (default 1, the discrepancy
   principle), or after ``max_iterations``.

The atom norms ``‖J e_p‖²`` are computed once, at the start, with one
Jacobian–vector product per pixel (a row of pixels at a time, so the Jacobian
is never held whole).

For linear data ``‖J e_p‖`` is the same at every pixel, so steps 2 and 3 are
exactly Högbom's: the residual peak, divided by the beam's peak.
Mathematically this is matching pursuit with normalised atoms.

The components are fluxes relative to a fixed **base scene** (usually an
analytic star at flux 1), so the model is
``V = (w V_base + Σ c_p e_p) / (w + Σ c_p)``, smooth in ``c`` even at
``c = 0``. Without a base scene, the components alone make the image, and
CLEAN starts from one component at the centre (closure phases cannot fix the
position, so this is also the anchor).

The result is an ordinary virgil model: ``System(base=..., clean=Image(...))``
whose Image has support only where ``c > 0``. So it can be polished with
``fit`` on that support, used as a starting image for RML, or convolved with
the beam (``convolve_beam``) to give the "restored" image of radio astronomy.

Limitations, accepted for S2:
- Components are only added, never removed. A wrong early component stays
  (Högbom CLEAN can subtract negative components; we keep positivity). A
  final ``fit`` on the support can move flux among the components.
- The base scene's parameters are fixed during CLEAN; fit them first.
- The components are grey: their flux is a fixed fraction of the base
  scene's at every wavelength.

## Stages

Each stage is a PR with tests and an entry in the stage log below. S1 and S2
share one MWE notebook, `notebooks/mwe/mwe_sparse_imaging.ipynb`, added with
S2, because `LogSum` is best started from a CLEAN model.

### S1: smooth sparsity regularisers

- `starlet(b, scales)`: the isotropic undecimated wavelet transform
  (à-trous, B3-spline kernel; Starck, Murtagh & Fadili 2010), with zeros
  beyond the edges, as in `TV`. Its detail planes and coarse plane sum back to
  the image.
- `StarletL1(weight, scales=4, epsilon=1e-2)`: ``weight × Σ √(w² + ε²)``
  over the detail coefficients, ε a fraction of the mean pixel flux, like
  `TV`.
- `LogSum(weight, epsilon=1e-2)`: ``weight × Σ log(1 + b / (ε b̄))``, a
  smooth surrogate for L0, where b̄ is the mean pixel flux over the support.
- `Laplacian(weight)`: ``weight × Σ (∇²b)²`` with the five-point stencil;
  it has residuals, so LM can fit it.
- The pixel L1 identity is documented in the `virgil.imaging` docstring.

All three plug into `fit`, `l_curve` and `diagnose` unchanged.

### S2: gradient CLEAN

`clean(data, npix, pixel_scale_mas, base=None, *, gain=0.1,
max_iterations=1000, target_chi2_red=1.0, support=None, init=None,
rotation_deg=0.0, dtype="float64")` returning a `CleanResult` with the
model, the component fluxes, the χ² history and why it stopped, and a
`restored(beam)` method.

### S3: proximal solver (only if S1 is not sparse enough)

A linear-brightness Image option; `fit(method="fista")` for proximal gradient
with projection onto the simplex (optax's `projection_simplex`) and gradient
steps on the analytic parameters; regularisers gain `prox()`. Starlet L1
has no closed-form prox for a redundant transform, so it needs a primal–dual
method (Condat–Vũ). This is the only stage that touches `Image` and `fit`.

### S4: sparse sampling (deferred)

Horseshoe or Laplace priors on starlet coefficients through `numpyro_model`.
Deferred: the posterior mean under a Laplace prior is not sparse (the
"Bayesian lasso"), and the Gaussian-process prior already covers sampling.

## Status and next steps (paused 2026-10-05)

S1 and S2 are merged (#165, #166). Users have `StarletL1`, `LogSum`,
`Laplacian`, `starlet`, `clean` and `CleanResult`. The docs have imaging part 6,
"Sparse images and CLEAN" (`notebooks/imaging_clean.ipynb`), and there are two
MWEs: `notebooks/mwe/mwe_clean.ipynb` (five cells) and
`notebooks/mwe/mwe_sparse_imaging.ipynb` (CLEAN against MEM, StarletL1 and
LogSum). Work is paused here.

### What the runs taught us

- **Normalise CLEAN's pixel selection.** Picking the most negative raw
  gradient piles flux next to an analytic star (S2 log). The matching-pursuit
  score `g_p² / |J e_p|²` fixes it, and is exactly Högbom's rule for linear
  data.
- **CLEAN stops slightly short of the flux.** At the default gain of 0.1 and
  the discrepancy target, it recovered 90–99% of the flux: 0.0297 of 0.030 (AMI),
  0.0341 of 0.035 (AMI, extended) and 0.062 of 0.070 (VLTI). Refitting the
  component fluxes with `fit` on the support recovers the rest.
- **A large gain stalls.** At a gain of 0.5, CLEAN reached χ²/N ≈ 1.1 in
  20 iterations and then ran to `max_iterations`, because flux is never
  removed. CLEAN also runs to the limit when noise puts the truth's own
  χ²/N above the target (about √(2/N) scatter). Neither case is detected.
- **Warm-started sweeps fail for penalties that switch pixels off.** Under
  `l_curve` (strong to weak, warm starts), `LogSum` collapsed to one pixel at
  every weight, because log-brightness pixels driven to ~0 never recover.
  Fitted per weight from the CLEAN start, it reached χ²/N = 0.97 and was as
  compact as CLEAN. The same path dependence could affect any penalty strong
  enough to darken pixels, including MEM with a peaked prior.
- **L-BFGS hits the step limit.** MEM and StarletL1 did not converge
  within 50 000 steps at some weights on 62² pixels, although they reached
  χ²/N ≈ 1. Not yet investigated.
- **StarletL1 barely changes how sparse the image is.** At the discrepancy
  weight, its brightest 2% of pixels held 0.78 of the flux, against 0.80 for MEM
  and 1.00 for CLEAN and LogSum. Its benefit, if any, is multi-scale
  structure, which these scenes do not test. So far this is no case for S3:
  LogSum started from CLEAN already gives sparse images.
- **Cost.** The atom norms cost one JVP per pixel, once. That is negligible at
  40–62² (the tutorial runs in 31 s on a CPU node), but grows as N² pixels times
  the cost of one JVP. The norms are also frozen at the start, which is only
  approximate for non-linear data.

### Next steps, in order

1. **Detect stalls in `clean`.** Stop when χ² falls by less than a tolerance
   over the last k iterations, with `stop="stalled"`. Report the closest
   approach to the target. Cheap, and it removes both silent runs to
   `max_iterations`.
2. **Remove flux.** Every k iterations, refit the component fluxes on the
   support: a few LM steps of `fit`, or projected non-negative least squares
   on the Gauss–Newton model. These are the "major cycles" of Clark and
   Cotton–Schwab CLEAN, and they let flux move off early mistakes. The
   alternative is Frank–Wolfe "away steps". Either should fix the gain-0.5
   stall.
3. **Sweeps without warm starts.** Give `l_curve` a `warm_start=False` option,
   or a weak-to-strong order, so that `LogSum` (and anything like it) can be
   swept with the standard tool. Then the MWE's hand-written loop can go.
4. **Real data.** Try CLEAN on an AMI or NRM dataset with a known companion,
   and on PIONIER closure phases (e.g. the HR 4049 or IW Car data used
   elsewhere). Compare with the parametric fits.
5. **Independent validation.** Request a virgil-validation check (as an
   Issue there; see AGENTS.md): for linear complex-visibility data, `clean`
   must reproduce an independent Högbom CLEAN component for component.
6. **Refresh the atom norms** every so often for non-linear data, and
   compute them in chunks for large grids, if real data need ≥128² pixels.
7. **Multi-frequency CLEAN** (Rau & Cornwell 2011): a spectral index per
   component, instead of grey components, for polychromatic data.

S3 (proximal solver) and S4 (sparse sampling) stay deferred; nothing so far
needs them.

### Running the notebooks

The MWEs and tutorial run on OzSTAR, never on the laptop. The new generic
runner `ozstar_scripts/scripts/_notebook_template` takes the notebook list as
`--notebooks=a,b`. `scripts/sparse_mwe` (one task per notebook, as listed in
its `tasks.txt`) still works, but can be retired in favour of the runner.
The full `mwe_sparse_imaging` takes about 28 min on a CPU node, mostly in the
MEM and StarletL1 sweeps. The tutorial and the five-cell MWE take under a minute each.

## Deferred items

| Item | Revisit when |
| --- | --- |
| Dark-energy regulariser | Someone needs it; read its definition in SQUEEZE's source first |
| Removing CLEAN components (major cycles or away steps) | Next step 2 above; the gain-0.5 stall shows the need |
| Fitting the base scene during CLEAN | A base parameter cannot be fitted beforehand |
| Residual map in flux units (per-pixel curvature) | Restored images need residuals added |
| S3, proximal solver | S1 images are not sparse enough to matter scientifically |
| S4, sparse sampling | A science case needs uncertainties on a sparse image |

## Stage log

### S1

`Laplacian`, `starlet`, `StarletL1` and `LogSum` in `virgil.imaging`, with
tests in `tests/test_sparse_imaging.py`. Like `TV`, their ε is an array, so a
new value does not recompile a fit. Zero padding means a flat image has
starlet details at its edges, as it has differences for TV: flux at the edge
of the field is penalised.

### S2

`clean` and `CleanResult` in `virgil.imaging`. `Image`'s matrix-Fourier
path is now a shared helper, `models._pixel_visibilities`, used by the
private `_CleanScene` too.

The first version picked the pixel with the most negative gradient, as
Högbom's peak search does. On AMI DISCO data of a star with a 5% companion,
it put 0.45 of the star's flux in the pixel next to the star: there, both
the gradient and ``‖J e_p‖`` are small, so the Gauss–Newton step is huge.
With the selection normalised by ``‖J e_p‖²`` (the predicted fall in χ²),
CLEAN put its brightest component on the companion's pixel and reached
χ²/N = 1 after 28 iterations, with 0.034 of the 0.05 companion flux
(stopping at the discrepancy point under-recovers flux).

### First MWE run (OzSTAR job 17999081, commit 05d5da8)

Three knots and an extended blob (3.5% of the flux) beside a star, AMI
DISCOs, 62² pixels of 20 mas:
- CLEAN reached χ²/N = 1 in 131 iterations with 64 components and 0.0341 of
  the 0.035 flux. Convolved with the beam it matches the truth (NCC 1.00);
  at native resolution the blob is a scatter of points (NCC 0.70).
- MEM and StarletL1, swept with `l_curve` from the CLEAN start, also match
  the truth after convolution (NCC 1.00). Their native NCCs are 0.91 and
  0.83, and neither converged within 50 000 L-BFGS steps at every weight.
- **LogSum failed when swept with `l_curve`**: every weight ended at one
  bright pixel, χ²/N = 58. A strong weight switches most pixels off, and in
  log-brightness coordinates they cannot recover, so each warm-started
  weaker fit inherits the collapse. The `LogSum` docstring now says so, and
  the MWE fits each `LogSum` weight from the CLEAN start instead.

### MWE rerun (OzSTAR job 18019130, commit 3f3c0b2)

With each `LogSum` weight fitted from the CLEAN start, `LogSum` reached
χ²/N = 0.974 at w = 3.2e-3 and matches the truth after convolution (NCC
1.00); it is as compact as CLEAN (all its flux in the brightest 2% of
pixels) and, like CLEAN, breaks the extended blob into points (native NCC
0.68). The other results were unchanged. The CLEAN tutorial
(`notebooks/imaging_clean.ipynb`, imaging part 6) ran in 32 s.

## References

- Baron, Monnier & Kloppenborg 2010, Proc. SPIE 7734: SQUEEZE (MCMC
  imaging with L0 and multiple regularisers).
- Candès, Wakin & Boyd 2008, J. Fourier Anal. Appl. 14, 877: reweighted L1
  and the log-sum penalty.
- Högbom 1974, A&AS 15, 417: CLEAN.
- Rau & Cornwell 2011, A&A 532, A71: multi-scale multi-frequency synthesis
  (multi-frequency CLEAN).
- Starck, Murtagh & Fadili 2010, *Sparse Image and Signal Processing*
  (Cambridge): the starlet transform.
- Thiébaut & Young 2017, JOSA A 34, 904 (arXiv:1708.08390): principles of
  image reconstruction in optical interferometry.
