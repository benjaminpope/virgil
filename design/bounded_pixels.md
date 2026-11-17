# Design: bounded linear pixels and KKT convergence certificates for image fits

Status: B1 (§2–§3 item 1, `LinearImage`) implemented; §1, §3 items 2–3 and §4 are later changes (L-BFGS-B, preconditioner
fix, KKT certificates). From the IM Lup dots diagnosis. Public docs must not link this note (design notes stay internal).

Implementation notes for B1 (deviations from the draft):
- `LinearImage` is a sibling of `Image` (shared `_BrightnessImage` base), not a subclass, so `fit`'s `Image`-only code
  (`_scene_images`, `log_brightness`) is untouched. The free leaf is `pixels`.
- `flux` stays a field fixed at 1 (the `Component` weight slot); the component's System weight (`_weight`) is Σ pixels,
  exposed as `total_flux`. Do not fit `flux`.
- Regularisers act on `fluxes / reference_flux`, with their flux normaliser fixed at 1 (convex). MaxEntropy on
  linear pixels uses Skilling's un-normalised form (value `Σ b ln(b/q) − b + q`, residuals unchanged). It raises for
  `parametrization="linear"` until the L-BFGS-B change supplies the positive floor; use `"sqrt"`.
- `image_priors` gives `pixels` a real (sqrt) or non-negative (linear) improper prior; the linear form is only
  properly bounded once `method="lbfgsb"` exists.
- No external source exists for the `"sqrt"` form: its support is design §3 (test P) and the tests in
  `tests/test_linear_image.py`. §5.5's linear-MEM comparison waits for L-BFGS-B.

## Why
- Softmax/log pixels (`Image`) cannot reach a minimiser with exact zero pixels, and plain gradients there scale as I², so
  faint pixels freeze. Maximum entropy always has an interior minimiser, so it keeps softmax pixels, preconditioned with the
  entropy metric. TSV, TV, StarletL1, Laplacian and the unregularised fit can have zeros at their minimiser and need
  pixels that can reach zero.
- Convergence is judged today by a loss plateau (vv C8, virgil's `converged`). A plateau does not show that the faint or
  poorly constrained pixels are converged. Replace it with first-order optimality (KKT) certificates.

## 1. Positivity by bounds, not by a map
Enforce positivity in the solver with box constraints. Do not use a ReLU, softplus or squared map.
- **Solver: L-BFGS-B** (Byrd, Lu, Nocedal & Zhu 1995, SIAM J. Sci. Comput. 16, 1190, ADS 1995SJSC...16.1190B; Zhu et al.
  1997, ACM TOMS 23, 550). It is the solver of eht-imaging (Chael et al. 2016, 2016ApJ...829...11C; 2018, 2018ApJ...857...23C)
  and SMILI (Akiyama et al. 2017, 2017ApJ...838....1A; 2017AJ....153..159A). Its active set (generalised Cauchy point)
  keeps the curvature pairs valid. Projecting after an L-BFGS step does not, which is why `fit` already refuses
  `post_step` with L-BFGS (`fitting.py` ~760).
- **Implementation: `scipy.optimize.minimize(method="L-BFGS-B")`** driven by a jitted
  `jax.value_and_grad` of the existing `_Objective` loss. Use float64 (the Fortran code is float64), bounds `(0, None)` per
  pixel, and `(0, 0)` outside `support`. The host loop costs one device-to-host copy of the gradient per evaluation
  (≤ 2.4M floats). That is negligible next to the χ² evaluation on a GPU, and the Toeplitz χ² keeps working unchanged.
  Rejected options:
  - `jaxopt.LBFGSB`: jaxopt is in maintenance mode, and it would be a new dependency.
  - optax projections after `optax.lbfgs`: not a convergent method.
  - Spectral projected gradient (Birgin, Martínez & Raydan 2000): a jittable fallback, but slower on stiff problems. Mention
    it in the design as future work only.
- **API:** `fit(..., method="lbfgsb")`. It is valid only when every free leaf is either a bounded-pixel image or a parameter
  with a box prior. Other bounded priors (`Uniform`) map to the same bounds, so no bijection is needed. Unbounded leaves get
  `(None, None)`. If the problem has other constrained parameters that cannot be boxed, raise `TypeError`.
  Stop rules: `pgtol` (projected-gradient ∞-norm, relative to the starting value; default 1e-6), `ftol`, `max_steps` and
  `time_limit` as for `lbfgs`. `info["converged"]` comes from the KKT certificate below, not from scipy's message alone.

## 2. Bounded-pixel image and total flux
New `LinearImage(pixels, pixel_scale_mas, support=None, dra=0, ddec=0, rotation_deg=0, ft="auto", blur_fwhm_mas=None)`
(name open), sharing `Image`'s transform code (`_PixelRendered`/`Image` helpers; no copy of the FT paths):
- **Free leaf:** `pixels ≥ 0`, the flux per pixel in Jy (or in the scene's flux unit).
- **Flux:** the total flux is Σ pixels. There is **no separate free `flux`**, because a free flux together with free pixel
  amplitudes is an exact scale degeneracy. `flux` is a read-only property. `brightness = pixels / Σ pixels` gives
  `Image`-compatible unit-sum pixels for display and for code that expects them. In a composite scene (star + envelope)
  the envelope's Σ pixels is its flux, and the flux ratio follows from it.
- **Regularisers on linear pixels act on `pixels / F0`**, where `F0` is a fixed reference flux (a constructor argument;
  default Σ of the starting pixels, recorded). This keeps TSV, TV, Laplacian and StarletL1 convex quadratic or convex in the
  free variables, and keeps their #614 normalisation. Dividing by the fitted Σ pixels would make them non-convex. Document
  that the weight's meaning drifts by (F/F0)² if the fitted flux moves far from F0.
- **Maximum entropy on linear pixels** (not the recommended path, but well defined): use Skilling's un-normalised form
  Σ [I ln(I/m) − I + m] with m = F0·q (Skilling 1989, *Classic maximum entropy*, in Maximum Entropy and Bayesian Methods,
  Kluwer; not on ADS). It is convex, and its minimiser is interior. Its gradient is −∞ at I = 0, so the lower bound must be
  `tiny·F0`, not 0. Raise if a user asks for MaxEntropy on `LinearImage` without that floor.

## 3. Maximum entropy: √I pixels (revised after test P, 2026-11-03)
The earlier draft kept softmax `Image` + `precondition="diag"`. Test P (`diagnosis/testP/`, same J for all, IM Lup-like
synthetic, w = 1257) rejects that path at data-dominated weights:
- `precondition="diag"`: χ²/N 3.78 after 4400 steps, 84% of faint pixels still want to grow. Cause:
  `_fisher_diagonal` (`fitting.py` ~1686) deliberately *replaces* the data JᵀJ with the entropy metric w·b on MEM pixels.
  That is right only when w dominates the data curvature (the 8×8 AMI test in its comment). With 42M visibilities the
  data curvature in bright pixels exceeds w·b by orders of magnitude, so the metric mis-scales exactly the pixels that matter.
- √I pixels (x = √I, unbounded L-BFGS; metric 4I, i.e. exponentiated-gradient/entropic mirror descent) reached the lowest
  J (41208.7) in 680 evaluations; bounded linear L-BFGS-B 41212.0 in 1466; softmax plain 41378.0 in 6687 steps. The √I and
  linear solutions agree to 0.7% of the peak (ψ correlation 0.99), and √I meets the band-limit check (f_eta = f_floor).
Changes:
1. `LinearImage(..., parametrization="linear" | "sqrt")`. With "sqrt" the free leaf is x, pixels = x², no bounds needed,
   any `fit` method works. x = 0 is a stationary point, so start strictly positive (flat or a CLEAN/blurred start).
   For MaxEntropy x → 0 is never a minimiser (the entropy gradient pushes it up); for other regularisers a fit started
   positive escapes the saddle unless a pixel is exactly 0.
2. Fix `_fisher_diagonal`: add the entropy metric to the data diagonal instead of replacing it (full Gauss–Newton
   diagonal in η). Keep the replacement only behind an explicit option if the AMI speed-up must be kept. Re-run test P2
   with the fix before deciding whether softmax + diag remains a supported MEM path.
3. Do not make `precondition="diag"` the default for MaxEntropy (drop the earlier proposal).

## 4. KKT certificates (new `virgil.fitting.kkt_report(result or model, priors, data, regularizers)`)
- **Softmax `Image` (simplex):** g = ∂J/∂b (the gradient with respect to the unit-sum pixels; flux and nuisances at their
  values), μ = Σ b g.
  - `kkt_rel` = rms over pixels with b > 1e-3·max b of (g_j − μ)/(|μ| + scale), where scale is the rms of g over the same
    pixels.
  - `want_grow` = fraction of pixels with b < 1e-6·max b and g_j < μ − 3·scale·`kkt_rel`. These are faint pixels the
    objective would raise, i.e. frozen ones.
  - For MaxEntropy alone add `band_limit` = (power of the windowed ln b beyond the data's largest |uv|) / the same for its
    hard low-pass. ≈ 1 at the optimum (diagnosis §1.2, Test A).
- **`LinearImage` (orthant):** projected gradient P(I − g) − I; `pg_rel` = its ∞-norm / ∞-norm of g at the start. Also
  report the fraction of zero pixels with g_j < 0 (must be 0 at the optimum).
- **Nuisances:** ∞-norm of the gradient in their unconstrained coordinates, relative to the start.
- `converged` is true when `kkt_rel` (or `pg_rel`) < 1e-3, `want_grow` = 0, and for MEM `band_limit` < 2. Record the
  thresholds as constants with this design as their source. vv's gate uses this report instead of C8 for image fits.

## 5. Tests (fast, CI)
1. **NNLS equivalence:** a 16² image, 300 random uv points, noise, unregularised `LinearImage` via `method="lbfgsb"`
   against `scipy.optimize.nnls` on the explicit real DFT matrix. Pixels agree to 1e-6 of the peak, and the zero set is
   the same.
2. **Zeros reached:** a point source on an empty field with weak TSV. More than 50% of pixels are exactly 0, and every
   zero pixel has g ≥ 0 (KKT).
3. **Flux:** Σ pixels matches the zero-spacing flux of noiseless data to 1e-8, and `flux`/`brightness` round-trip.
4. **Support:** pixels outside `support` stay exactly 0.
5. **Softmax vs linear agree on an interior problem:** MEM (Skilling form on linear pixels with the tiny floor) and
   `Image` + MaxEntropy + `precondition="diag"` reach the same J to 1e-8 relative and the same image to 1e-5 of the peak
   (flat q, F0 = true flux).
6. **Certificates:** `kkt_report` gives converged on the outputs of tests 1, 2 and 5, and not converged on the same fits
   stopped after 5 steps. On a softmax unregularised fit started with 10% of pixels at 1e-12, `want_grow` > 0.
7. **Refusals:** `method="lbfgsb"` with a non-boxable prior raises `TypeError`; float32 gets a warning and computes in
   float64.
Slow set (not CI): the 384² fast-loop TSV fit by `lbfgsb` against softmax `lbfgs`. Report J, `pg_rel`, steps and wall time.

## 6. Docs
- `docs/imaging_rml.md`: which pixel type goes with which regulariser (MEM → `LinearImage(parametrization="sqrt")`;
  TSV/TV/StarletL1/none → `LinearImage` with `lbfgsb`), and why (one paragraph, with Byrd et al. 1995 and Skilling &
  Bryan 1984). Mention that MPoL's softplus pixels with entropy on the blurred image leave the base image unregularised.
- Add the references to `references/provenance.toml` and `docs/references.md` as for other sources.

## Register entry: √I pixels (methods beyond published practice)
- **Claim:** `LinearImage(parametrization="sqrt")` fits x = √I with a plain unconstrained optimiser. The metric is 4I,
  that of entropic mirror descent, so faint pixels converge as fast as bright ones. It reached the lowest objective
  fastest in test P (§3).
- **References:** bounded linear pixels, Akiyama et al. 2017 (`akiyama2017`); L-BFGS-B, Byrd et al. 1995 (`byrd1995`);
  Skilling's un-normalised entropy, Skilling 1989 (`skilling1989`), all in `references/provenance.toml`. No source
  for the square-root parametrization was found on ADS. Biraud 1969 (1969A&A.....1..124B, "A New Approach for
  Increasing the Resolving Power by Data Processing") is candidate prior art that has not been read: unverified.
- **Tests:** `tests/test_linear_image.py`. **vv check:** none yet.
- **Status:** novel / experimental. It is opt-in (the default is `"linear"`) and provisional.
