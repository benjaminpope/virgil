# Conditioning of full-batch fits: natural units, profiled scales, preconditioning

Status: A and C built; B pieces 1–2 built (opt-in `precondition="diag"`), pieces 3–4 pending; the Stage 7 success test is *not* met by a diagonal alone (see "B as built"). General note: the worked example
is a multi-observation radio fit, but nothing here is specific to it.

## The problem

A full-batch L-BFGS fit of an image (10⁶ pixels) and a few calibration
nuisances to 10⁷–10⁸ visibilities is badly conditioned when the Fisher
information, **in the coordinates the optimizer sees**, spans many decades.
L-BFGS memory (tens of steps) cannot learn a condition number of 10¹⁰. For a
radio fit with several observations the diagonal of the Gauss–Newton Fisher in
the old coordinates was, per coordinate:

| Coordinate | Origin | Fisher scale |
| --- | --- | --- |
| offset of an observation | `Uniform(±1/(2 q_min))` ≈ ±10⁴ mas through a sigmoid: `dx/dz` ≈ 5·10³ mas, while a long-baseline offset is known to ~1 µas | ~3·10¹³ |
| gains, ln s | logit box on a log range, `dz/d ln x` of order 1 | ~10⁷ (number of points) |
| bright image pixel (`z = ln I`) | | ~50 |

(Measured per-observation offset information, in the example: `1/(2π q_max)` =
13.7, 15.1, 61.5, 50.9, 18.4 mas for five short-baseline observations and 0.87
and 0.81 mas for two long-baseline ones; offset σ from 0.155 to 0.0009 mas.)
Three separate defects: (1) the offset coordinate was not in units of what the
data resolve; (2) gains and σ used boxes, not plain log coordinates; (3) σ was
fitted jointly although it has a closed-form optimum.

## A. Natural-unit coordinates (built)

`CalibrationNuisance.priors()` now gives, by default:

* **offsets**: for observation `k`, `b_k = 1/(2π q_max,k)` (rad → mas), the
  offset that turns the phase by one radian at the longest baseline. The prior
  is a plain `Uniform(±θ_k)`, `θ_k = offset_beams · 2π b_k` (default 3
  beams: a beam is `≈ 1/q_max = 2π b_k`; three beams is generous for a phase
  centre error after correlation, wide enough that the optimum is interior, and
  narrow enough that the sigmoid's slope at the centre, `2θ_k/4 ≈ 9 b_k`, stays O(`b_k`)). The width of the box sets the scale: a sigmoid on `±θ_k` has `dx/dz = θ_k/2 ≈ 9.4 b_k` at `z = 0`, O(`b_k`) for every observation, whatever the baselines. (An earlier version also divided by `b_k` in a custom flat coordinate; that cancels exactly and was removed.)
  The old box remains available as `priors(offset=(-1/(2 q_min), 1/(2 q_min)))`
  (`offset_bound_mas`).
* **gains, σ scales**: the improper Jeffreys prior `1/x` (`_LogImproper`), flat
  in `ln x`. Its flat coordinate has infinite ends, which `_FlatBijection` maps
  to the identity, so `z = ln x`, no sigmoid. Explicit `gain=(lo, hi)` or
  `sigma=(lo, hi)` bounds give the old `LogUniform` box.

Site names, shapes and physical values (`cal.gain`, `cal.offset` in mas,
`cal.sigma`) are unchanged, so warm starts from saved physical values and
downstream code keep working. `at_bound` still reports an offset on its (now narrower) box; the
gain and σ priors have no edge. Flux keeps `LogUniform(1e-4, 1e2)`: its
logit slope in `ln F` is 4, harmless.

On the synthetic test (a compact Gaussian seen by a short- and a long-baseline
observation, 1500 points), the diagonal of the Hessian of the loss at the truth
spans 8.0 decades over (gain, offset, σ) with the old priors and 2.7 with the
new; over all sites (flux, α included) 9.2 and 4.0 decades.

## C. Profiled σ (built)

With the per-point loss `½|r|²/s² + 2 ln s` (`r = (V − M)√w`, one complex
point), the joint minimum over `ln s_d` is at `s_d² = χ²_d(1)/(2 N_d)` under the
flat-in-`ln s` prior, **exactly**: `d/ds [½χ²₁/s² + 2N ln s] = 0`. So
`CalibrationNuisance(sigmas="profile")` drops the sites and evaluates `ŝ` inside
the term. Envelope theorem: `∇_θ L(θ, ŝ(θ)) = ∂_θ L(θ, ŝ)`, so `ŝ` is held by
`stop_gradient`; the value of the loss is the profile
`Σ_d N_d (1 + 2 ln ŝ_d)`. The term exposes `residuals_and_log_norm` so that `fit`
needs one forward pass (a separate `log_norm` call would need a second one to
find `ŝ`). `ŝ` is reported in `info["derived"]["cal.sigma"]`. Tests check the
gradient against finite differences of the profiled loss and that the profile
and joint optima (and loss values) agree.

The reported χ² of a profiled fit is at `ŝ`, so `χ²_d = 2 N_d` by construction;
quote the raw χ² at the stated errors, `Σ 2 N_d ŝ_d²`.

**Not done: profiling per-observation gains.** A gain is linear,
`g_k = Re⟨M,V⟩_k/⟨M,M⟩_k` with weights `w/s²`, but with σ profiled the weights
depend on `ŝ`, which depends on `g`: a fixed-point coupling, and the gain
couples to flux, α and the image through `M`. Variable projection over `(g, s)`
jointly would need an inner iteration or a closed form for the pair, and the
joint conditioning problem is already fixed by `z = ln g`. Left out.

## B. Diagonal Fisher preconditioning in `virgil.fit` (pieces 1–2 built)

This is the plan for finishing the **Stage 7 maximum-entropy preconditioning
item** of [`imaging_plan.md`](imaging_plan.md#stage-7-hardening-and-release-about-8-11-h-of-which-3-4-h-is-release-chores-and-the-rest-mem-preconditioning)
(Ben, 2026-10-03), not a new idea: its option "precondition L-BFGS in the
entropy metric (diag 1/b, Skilling & Bryan 1984), or in a basis whitened by the
default image" is B, and B generalizes it from the entropy to the data term.
**Success test (adopted from Stage 7):** an L-curve from `w = 0.1` to `10⁴` that
converges at every weight within the default step limit, in float32 and float64.
It applies to image reconstruction in general, not to radio data alone (see
"Image reconstruction in general" below).

Measured on the synthetic radio test after the fact: with the gains and σ left
in their old logit-on-`ln x` boxes and only the offsets changed, the
(gain, offset, σ) spread is 2.6 decades; with `ln x` coordinates, 2.7. As the
slope of that sigmoid is already O(1), the gain/σ change is not needed for
conditioning (it removes the box and the saturation near its edges, not a decade
of spread); the offsets carry the whole improvement.

### Idea

Optimize `z̃ = D^{½}(z − z_ref)` instead of `z`, with `D = diag(F)`, `F` the
Gauss–Newton curvature `JᵀJ + diag(P)` of `fit`'s loss in the unconstrained
coordinates. It is a change of variables: the minimum and the loss are the
same, the gradient is `D^{-½}∇_z L`, and L-BFGS in `z̃` is L-BFGS with a
diagonal initial Hessian `D^{-1}`. It removes the scale mismatch among
coordinates (nuisances vs pixels) but not off-diagonal correlation (a pixel
and its neighbours, flux and gain).

### Where `D` comes from

1. `gauss_newton_mass` already forms the diagonal (exact below 4096
   coordinates; above, the Hutchinson estimator `diag(F) ≈ mean_r (v_r ⊙ F v_r)`
   with Rademacher probes `v_r`).
   Reuse its code path (`curvature_rows`, `curvature_diag`) and expose
   `gauss_newton_diag(problem, z, n_probes)`.
2. For a radio image of `z = ln I` pixels the diagonal is analytic and cost-free:
   `D_p = I_p² Σ_k w_k g_k² T_k²`, since a pixel's Fourier kernel has unit modulus: a
   uv-weight sum times the pixel flux squared, plus the entropy/regularizer Hessian diagonal (known
   per regularizer: the maximum-entropy `λ/I_p` pieces, TV approximated by its
   smoothing). It is `O(N)` once, `O(1)` per refresh for fixed weights.
3. The calibration sites use the diagonal of `JᵀJ` from the (small) dense block, exact.

`D` is refreshed once per restart round (the fit already restarts L-BFGS in
chunks for `time_limit`/`progress`): a frozen change of variables within a round, as
a changing metric inside L-BFGS invalidates its secant pairs.

### Wiring

* optax L-BFGS: wrap the loss as `f̃(z̃) = loss(z_ref + D^{-½} z̃)` and
  `jax.value_and_grad` it; or equivalently scale the gradient/updates
  (`optax.scale_by_...` is not enough because the line search and the `max_step_size`
  cap act on the step norm in whichever coordinates it runs in). Cleanest: the
  change of variables in `_lbfgs_round`, un-scaled on exit. Adam:
  `optax.scale_by_rms` is already a diagonal preconditioner; use `D^{-1/2}`
  as the initial `nu` or precondition the same way.
* The `Budget`/`progress` code reports `z`, unchanged.

### What breaks

* **Convergence criteria**: `gtol` tests the gradient per data point in `z`. In `z̃` the gradient
  is `D^{-½}∇L`; keep the test in `z` (compute `∇_z = D^{½}∇_{z̃}`) so tolerances
  and `info["grad_norm"]` keep their meaning.
* **Step cap** (`max_step_size`, "a factor exp(2) per pixel"): defined in `z`; map
  to `z̃` per coordinate as `max_step · D^{½}`, or apply the cap after the
  un-scaling (breaks L-BFGS's line-search assumptions slightly).
* **Warm starts**: `z_ref` is the start; saved physical values are unaffected
  (`D` is recomputed, never stored with the result).
* **Floor pixels** (`I → 0`, `D_p ∝ I_p² → 0`): `z̃` scales as `1/I_p`, so a pixel
  at the floor takes huge `z` steps. Clip `D_p ≥ D_min` (the floor of the
  pixel's own regularizer curvature, or `ε · max D`), which is the usual
  Levenberg-like damping.
* **LM**: already uses `JᵀJ` and needs no preconditioner; leave unchanged.
* Float32 fits: `D` spans fewer decades, which helps.

### Tests

(1) Quadratic with a known diagonal metric: preconditioned L-BFGS steps ≈ unit
condition; (2) the optimum and loss are unchanged to tolerance on a small
radio problem with and without preconditioning; (3) iteration count falls on
the synthetic multi-observation problem of A; (4) refresh each round
reproduces the un-preconditioned answer; (5) `grad_norm` and `converged`
agree.

### Effort (PR-sized pieces)

1. `gauss_newton_diag` + tests (exact and Hutchinson), reusing existing code — small (~1 day).
2. Frozen change of variables in `_lbfgs_round` with the `z`-space convergence
   test and step cap; `precondition="diag"` argument, off by default — medium (~2–3 days).
3. Analytic radio image diagonal and regularizer Hessian diagonals; round refresh and floor damping; docs — medium (~2–3 days).
4. Benchmark on an imaging problem (OzSTAR) and decide the default — small
   (~1 day plus a job).

Total roughly 1–1.5 weeks; piece 2 carries the risk.

### B as built (pieces 1–2)

`fit(..., precondition="diag")` (default off) and
`gauss_newton_diag(model, priors, data, values, ...)` in `virgil.fitting`.

* **The diagonal.** `D = diag(JᵀJ) + P + max(0, diag ∇²penalty)`, in `fit`'s own
  (flat) coordinates: `J` from `curvature_rows`, `P` from `curvature_diag`, the
  regularizers' Hessian diagonal from Hessian-vector products. Exact up to 1024
  coordinates (one Jacobian-vector product each, with `lax.map`, never
  materializing `J`), not the 4096 of the NUTS mass: a refresh costs `n` model
  evaluations, so the threshold is lower. Above it, `diag(JᵀJ)` is the mean of
  `(Jᵀu)²` over 16 Rademacher `u` (±1) in *row* space (2026-10-22; they were
  Gaussian): the estimate is never negative and unbiased, and exact for a
  diagonal `J`, which the entropy metric's rows nearly are; the
  regularizer's diagonal uses 16 Rademacher Hessian-vector probes. The probe
  keys are fixed, so a fit is reproducible and does not depend on the JAX version.
  The result sets scale only: the optimum does not depend on it.
* **The change of variables.** `z = z_ref + P(w ⊙ z̃)` with `w = √(N / D)`,
  `N` the number of data points (the loss the optimizers see is per point, so
  `z̃` has unit curvature and the first L-BFGS step is a diagonal Newton step).
  It is frozen within a round and refreshed at each L-BFGS restart (every 2000
  steps). `gtol`, `info["grad_norm"]` and the stopping test stay in `z`
  (`∇_z = ∇_z̃ / w`); the step cap acts on `P(w ⊙ Δz̃)`, a step in `z`, as one
  factor on the whole direction as before.
* **Floor.** `D ≥ 10⁻³ max D` within each coordinate block. Smaller floors
  (10⁻⁶, 10⁻⁸) let dead pixels (`I → 0`, `D → 0`) have huge `w`; the cap then
  shrinks every step to nothing and the line search stalls. Sweep over floors
  10⁻³…10⁻¹² (AMI 32² and VLTI-like 41², float64/float32; steps, `*` = not
  converged): at 10⁻³ every weight converges in float64 on both (AMI
  40 27 36 188 1357 1326; 41²: 18 56 121 963 4911 9236); at 10⁻⁶ the weakest
  weight stalls or hits the 20000-step limit; at ≤10⁻⁹ the weak weights stop
  after 1–2 steps. So 10⁻³ is the only floor tested that converges every
  weight, chosen from this data, not from the rank argument.
* **Softmax gauge.** `P` removes the mean over the support from the plain
  `log_brightness` block of an `Image` (`flux × softmax`): the loss does not see
  it, but the diagonal scaling would let it drift. (`GaussianField` latents get
  no projection.)
* **Adam.** `scale_by_rms` is kept. It follows the *size of the gradient*, the
  metric the *width of the coordinate*; Adam's step is then `lr · width` per
  coordinate, rather than `lr` for all (a faint pixel's minimum is wide in `z`,
  a nuisance's narrow). `D` is computed once at the start (Adam has no rounds)
  and its widths normalized to a minimum of one, so the best-determined
  coordinate keeps the learning rate `lr`. Not benchmarked: the question is
  whether the gain over plain Adam is real (see (c)).
* **LM** ignores the option (it already uses `JᵀJ`).
* `ImproperUniform` priors with an `event_shape` (an image's) are now accepted by
  the curvature code: they add none (previously they raised, as the check on the
  event shape came first).

**Measured** (L-curves, descending `w = 10⁴ … 0.1` in six log steps, warm-started,
default `max_steps = 20000`, `MaxEntropy`, flat prior; steps per weight):

| Case | dtype | plain | `precondition="diag"` |
| --- | --- | --- | --- |
| 2 VLTI-like nights, 495 + 396 points, 41² image | float64 | 19, 57, 360, 2042, 8844, **20000 (limit)** | 19, 80, 127, 850, 6092, 14645; all converge |
| same | float32 | 18, 53, 286, 1309, 6860, 16002 (stalled) | 18, 60, 124, 720, 5680, **20000 (limit)** |
| AMI 7-hole grid, 32² image | float64 | 206, 81, 137, 465, 2066, 2088 | 40, 27, 36, 188, 1357, 1326 |
| same | float32 | converges to w = 100; stalls from w = 10 | the same: converges to w = 100; stalls from w = 10 |

At a tight `gtol` the gain is larger: 16² AMI image at `w = 10`, `gtol = 10⁻⁸`:
17607 steps plain, 4020 preconditioned. At the default `gtol = 10⁻⁴` both stop
0.1–0.3% above the minimum in the loss, in a flat direction (the loss at
`gtol = 10⁻⁸` is 129.2282, at `10⁻⁴` 129.28 plain and 129.53 preconditioned),
so that stopping rule, not the optimizer, limits how well two runs agree.

**Accuracy of the generic diagonal (decides piece 3).** On the 41² case at the
w = 1 solution (75 pixels above 10⁻³ of the peak), 16 probes give a median
per-pixel error of 27% (90th percentile 55%; 56% of pixels within 30%), 64
probes 15%; the exact diagonal of the 1681 coordinates costs about 0.4 s
including compilation. That is at the 30% bar and the cost is negligible, so
**no analytic per-data-type overrides are built**: one generic path (data terms
via `curvature_rows`, regularizers via Hessian probes) serves OI, DISCO and
radio alike. `info["precondition_rounds"]` records, per round, the spread of D (`decades`) and the floored fraction (`floored`).

**The Stage 7 success test is not met by a diagonal alone.** Float64 passes
on the 41² case at 14645 of 20000 steps (plain fails), but float32 fails for both:
at float32 resolution the line search stalls ("a step no longer changes the
parameters") before the gradient test is met, and preconditioning does not
change that. Why a diagonal cannot do more, at the `w = 1` solution of the 41²
case (dense `F = JᵀJ + ∇²penalty` over the 1644 supported pixels):

* `JᵀJ` has numerical rank ≈ 60 of 1644. The other directions have curvature
  only from the entropy, `∇²S` in `z = ln I`, which is `I(1 + ln(I/b))` per pixel,
  **negative** for `I < b/e` and tiny for the faint pixels (the image has 155
  pixels above 10⁻³ of its peak, whose brightness runs to 10⁻¹⁶).
* `F` has 1568 non-positive eigenvalues; its positive spread is `1.7·10⁸`.
  Scaling by `D^{-½}` cuts that to `5.6·10⁶` (30×), not to something L-BFGS
  can absorb. The remaining spread is between the 60 data-fixed directions and
  the entropy-only ones, which is off-diagonal.

**Next options (Stage 7 list), in the order I would try them:**

1. **A least-squares / Newton–CG form of the entropy**, in the plan's words.
   `F = JᵀJ + diag(R)` is a rank-≈60 matrix plus a diagonal, so a Newton step
   is cheap by Woodbury or by CG on `(diag(R) + JᵀJ)`, preconditioned by `D`.
   virgil already has matrix-free LM (`method="lm"`, CG on the normal
   equations); the missing piece is a residual form for the entropy
   (`r_p = √(2 w (I ln(I/b) − I + b))`, whose half squared norm is the entropy
   penalty, defined for `I ≥ 0`), so that LM applies to `MaxEntropy` fits.
2. **The Skilling–Bryan subspace solver** in `I`-space (entropy metric `diag(b)`),
   which handles the non-convexity of the entropy in `ln I` (the image stays
   positive by construction).
3. **A better stopping rule** for the flat directions (see above), independent of both.

### Worth doing after A–C?

A and C remove the ~10⁶ spread between nuisance and image coordinates, which was
the dominant defect. What remains is image pixels among themselves (`D_p ∝ I_p²`
spans several decades from bright to faint) and the uv-sampling correlation.
Diagonal `D` helps the first, not the second. Do piece 1–2 if a fit after A–C still needs many thousands of
steps; otherwise defer.

## D. A Fourier (dirty-beam) preconditioner on the image

The Gauss–Newton curvature of the data term on the *linear* brightness `I` is
`P_I⁻¹ = Aᵀ W A`, with `A` the (type-2) Fourier operator; it is a
convolution with the dirty beam `B` (the transform of the weights on the uv
sampling), so `P_I ≈ B⁻¹` can be applied with FFTs, or approximated by a
uv-density (uniform-weight) factor `1/(ρ(q))`. A preconditioner for `I` is
natural; virgil's images are `I = exp(z)`, so the curvature in `z` is
`P_z⁻¹ = diag(I) (Aᵀ W A) diag(I) + diag(g)` (`g` the gradient term, dropped
in Gauss–Newton), i.e. `P_z ≈ diag(1/I) P_I diag(1/I)`. That depends on the
current image: for L-BFGS it must be **a frozen change of variables per round**
(`I_ref`, rebuilt at each restart), like B's `D`, never a changing metric within
a round. It also ill-behaves where `I → 0` (the floor pixels), the same
`1/I²` blow-up as in B, so it needs the same clip, and the dirty beam's
negative sidelobes make `B⁻¹` indefinite in poorly sampled directions (deconvolution is
ill-posed: it would need a Wiener-type regularization `(B + εI)⁻¹`, where `ε`
is the regularizer's curvature, which is exactly what CLEAN-like residual
solvers avoid by working in the image domain).

Assessment: **not after A–C.** It addresses the (uv-coverage) correlation that
the diagonal cannot, and would be the real fix for full-batch convergence at
10⁶ pixels, but it needs the frozen-round machinery of B (so do B piece 2
first), an FFT per step (the MFT/nufft operators are already the cost, so the
overhead is modest), regularization of `B⁻¹`, and a floor treatment, and its
benefit is hard to predict because the entropy/GP prior curvature is of the
same size as the data curvature on faint pixels. Decide with a measurement:
after A–C and B pieces 1–2, compute the eigenvalue spread of `diag(D^{-½}) F
diag(D^{-½})` on a small image (Lanczos on a 128² problem); if the remaining
condition number is ≫ 10³, build D as a follow-up. Otherwise skip.

## Image reconstruction in general

(a) **Analytic image diagonal in log-brightness coordinates.** For a pixel `z_p
= ln I_p` and data linear in `I`, `V_k = Σ_p T_k F_{kp} I_p`,
`D_p = I_p² · c + R_pp`, with `c = Σ_k w_k |T_k|²` (`|F_{kp}| = 1`) the same
for every pixel, and `R_pp` the regularizer's curvature. For squared
visibilities, closure phases and kernel phases (`OIData`) the data are not
linear in `I`: the Gauss–Newton row of datum `m` is `∂m/∂I_p`, which for
`V²_k = |V_k|²` is `2 Re(V_k* F_{kp}) T_k`, so
`D_p = I_p² Σ_k w_k (∂m_k/∂I_p)²`. Averaging over the unknown phase of `V_k`
gives `≈ I_p² · 2 Σ_k w_k |V_k|² |T_k|²` (the weights of V² data scale with
`|V|²`: a `|V|`-weighted average of the same constant), and for a closure phase
the sum of the three baselines' `|V_k|^{-2} w` factors (a phase datum has
sensitivity `1/|V_k|`). These are used for scale only, so the constant
`c` may be taken as a one-off sum, and can be refreshed from the model's
current `|V_k|` each round.

(b) **How to apply it.** Either as a diagonal initial inverse Hessian `H0 =
D^{-1}` in L-BFGS (replacing optax's scalar `γ = sᵀy / yᵀy`, the two-loop
recursion starts from `H0` instead of `γ I`), or as a frozen change of
variables per restart round (above). `H0` is the more direct, keeps
step caps and `gtol` in `z`, and lets `D` track the model inside a round
because `H0` need not be constant over the secant pairs; its cost is the
custom two-loop recursion (optax's `scale_by_lbfgs` fixes `γ`; a small own
implementation or a wrapper around its `precondition_by_...` hooks is needed).
The frozen change of variables works with optax unchanged. Start with the
change of variables (piece 2), evaluate `H0` only if the secant pairs
degrade across a refresh.

(c) **Adam** already divides by the root of an exponential average of squared
gradients, an empirical diagonal Fisher (`E[g²]` equals the diagonal of the
Fisher at the optimum for a log-likelihood). This is likely why Adam does
well on MPoL-style problems, and why a diagonal `D` is expected to bring L-BFGS
to where Adam already is, not beyond it: the gain over Adam must be
measured, not assumed, and the success test should include an Adam baseline.

(d) **The softmax gauge.** A constant added to every `z_p` rescales the whole
image, so with a fixed-flux (softmax) parametrization it is an exact null
direction of the loss. A diagonal `D` weights it, so a preconditioned step can
move along it by a lot without changing the loss, drifting the coordinates.
Project out the mean of the step (or of `z̃`) every iteration, or fit the flux
as a separate site.

*Implemented for every method (`fitting._recentre`).* The step projection above
keeps L-BFGS and Adam off the direction, but LM's QR solve steps along it by
rounding noise (the l_curve of virgil-validation, 8 × 8, TSV: the lit pixels
at 4e5 on jax 0.12 and 3.6e10 on 0.11, by platform). So LM is given one extra
residual, `log Σ exp x`, which makes the constant direction well conditioned
(the data's Jacobian does not see it; the row is zero at the gauge, so the
loss and stopping test do not see it), and `fit` recentres the start, each LM
chunk and the result to `x = log(flux)`, `Σ exp x = 1`, the standard
normalization of log-probabilities, which keeps the lit pixels near 0 whatever
the dark ones do. Dark pixels, whose optimum is zero flux, run towards `-inf`
(to -1e3 or below, by rounding) and are not clamped: that is a boundary optimum.
No loss, image or χ² changes beyond rounding; only the reported `log_brightness`.

(e) **`GaussianField` latents are already prior-whitened** (standard-normal
`η`; LM converges in 20–50 steps, per Stage 7), so B matters mostly for plain
`ImproperUniform` log-brightness with an entropy or TV regularizer. Do not add
`D` to the latent parametrization without a measurement.

**Relation to the entropy metric.** The maximum-entropy penalty
`λ Σ (I ln(I/b) − I + b)` has curvature `λ/I` in `I`, so the Skilling–Bryan
metric is `diag(1/b)` at the default image `b`. In `z = ln I`:
`dS/dz = I ln(I/b)`, `d²S/dz² = I (1 + ln(I/b))`, which is `b` at `I = b`, so the
`1/b` metric becomes `diag(b)`: the entropy's part of `R_pp`. The data
part `I_p² c` and this `λ b_p` together are `D_p`, with the entropy dominating on faint
pixels and weak data, where Stage 7 reports that the L-BFGS fits stall. The
analytic `D` of (a) therefore contains the Stage 7 preconditioner and adds the
data term; the entropy part is the pixel's `λ b_p` and the data part `I_p² c`, so
the data part dominates only where `I_p² c > λ b_p`.

(f) **Profiling generalized (follow-up, not this PR).** C is variable
projection of the scale parameters of one Gaussian likelihood. The same
applies to every data type with a fitted error scale (a per-group
`ŝ² = χ²_g(1)/(2 N_g)` for real and imaginary parts, `N_g` for real data), and
to the linear amplitudes (flux, per-observation gains, flux ratios for complex
visibilities and DISCO): solve the weighted least squares in closed form
(`a = ⟨M,V⟩_w/⟨M,M⟩_w`) inside the loss with `stop_gradient`. Where a
scale and an amplitude are both profiled, the pair is a coupled fixed point (see
C), to be solved with an inner iteration. Scope as a separate PR per data
type after B piece 2.

## Status (2026-10-22)

`precondition="diag"` is merged (virgil#474) and documented in
`docs/conventions.md`. **PR 2 (analytic per-data-type diagonals) is dropped**:
the Hutchinson probes are accurate enough (see "Accuracy of the generic diagonal"), so the generic
path serves OI, DISCO and radio alike. The IM Lup benchmark (float64, with and
without `precondition`, same step budget, pinned commits) is pending. Newton–CG
for the entropy term is not started; it waits on the benchmark.

## Entropy metric, LM residuals (2026-10-22)

Experiments on the 32² AMI and 41² VLTI-like L-curves (w = 1e4 → 0.1):

- A maximum-entropy pixel's curvature in `ln I` is `w·b` (the Skilling–Bryan
  metric); using it instead of the floored Hessian diagonal gives 1.5–3× fewer
  L-BFGS steps at weak weights. `MaxEntropy.metric_rows` exposes it as a
  Gauss–Newton form; probes cost about a third of the gain at 1681
  coordinates, so the diagonal is now exact up to 4096. *Superseded below:* the
  cutoff is back at 1024 and the probes are Rademacher.
- L-BFGS memory 100 helps image fits (the default stays 50 here; see the memory sweep).
- 41² VLTI-like, memory 50, steps at w = 1e4 … 0.1 (Rademacher probes at
  1024, one run, laptop under load): float64 18 33 86 300 2919 4527, float32
  18 31 96 385 2456 19524 (before: 18 50 147 792 4382 12265 and
  18 52 111 585 4004 20000, not converged). Memory 100 converges both faster
  (see the memory sweep).
- `MaxEntropy.residuals` (r = t√(2wqφ(t)), t = (b−q)/q) lets LM fit entropy;
  it converges every weight in float64 (11–1133 steps) but not in float32, so
  L-BFGS stays the default for entropy.
- float32 on AMI 32² still stalls at w ≤ 10 for every variant: the line search
  meets the noise of the float32 model evaluation (χ² biased 1–4 %); float64
  master parameters, float64 loss sums, re-centred parameters and a loss
  offset to the round's reference did not help. float32 is documented as
  limited at weak weights.

### Ensemble corners at memory 100 (12² AMI grid of `tests/test_ensemble.py`)

Corner (weight per data point) of an 11-weight L-curve, by L-BFGS memory and
step budget; the ensemble itself runs `fit`'s default 20000 steps, not 200.

| family | range | mem 50, 200 steps | mem 50, converged | mem 100, 200 steps | mem 100, converged |
| --- | --- | --- | --- | --- | --- |
| tv | 1e-4 … 10 | 0.32 | 1.0 | 1.0 | 1.0 |
| maxent | 1e-4 … 10 | 0.32 | 0.32 | 0.32 | 0.32 |
| starlet | 1e-4 … 10 | 0.1 | 0.32 | **3e-4** | 0.32 |
| tsv (LM) | 0.1 … 1e4 | – | – | 32 | 32 |

All corners lie in the default ranges once converged, so the ranges are
unchanged. The corner of a truncated starlet curve is not stable (3e-4 at
memory 100 and 200 steps), so `test_default_weight_ranges_hold_the_corner_with_room_below`
now fits to convergence. The starlet corner also moves with the sweep's range
(converged: 0.32 for 1e-4 … 10, 0.16 for 1e-5, 5e-6 for 1e-6): its L-curve has
no sharp corner, so a corner at the edge of a sweep is not a corner. Re-deriving
the ranges on the 60-dataset contest bench needs an OzSTAR job.

## Rademacher probes, exact cutoff back at 1024 (2026-10-22)

The exact diagonal at 4096 removed the probe error on the entropy metric, but
that error was the Gaussian probes' own: for a nearly diagonal `J`,
`(Jᵀu)_p² ≈ J_pp² u_p²` has relative variance 2 for Gaussian `u` and 0 for
Rademacher `u`. So `_gn_diagonal` uses Rademacher probes and `_EXACT_DIAG` is 1024
again (above it exact costs `n` Jacobian-vector products per refresh, wasted on
entropy pixels, whose data term is dropped). 41² VLTI-like L-curve, memory 100,
steps for w = 1e4 → 0.1 (identical χ²):

| dtype | Rademacher, 1024 | Gaussian, exact (4096) |
|---|---|---|
| float64 | 18 33 79 236 1348 2837 (15 s) | 17 33 77 317 1453 2877 (18 s) |
| float32 | 18 31 80 266 1355 3416 (9 s) | 18 32 73 252 1529 2971 (10 s) |

Wall times are one run each on a loaded laptop. Float64 is at least as good;
float32 is within the line search's run-to-run variation at the last weight.

**2026-11-02, float32 L-curve under ambient x64.** The Stage 7 test
(`test_the_l_curve_converges_at_every_weight[float32]`) stalled at w = 0.1
(3358 steps, unconverged, deterministic across five CI runs) in the x64-ambient
job only, once #615 asserted it. The fit itself runs float32 inside `run_in`, so
the difference is the inputs: noise draws, scenes and the starting image built
in float64 and cast down are another realisation of the problem. The test now
builds a float32 case under `jax.enable_x64(False)`, so it is the same problem in
every job; float32 at the weakest weight remains noisy, as above.

## L-BFGS memory sweep (2026-10-22)

Steps to converge at the weakest weight (w = 0.1) of the L-curves with
`precondition="diag"` and `MaxEntropy` (entropy metric, Rademacher probes), by
L-BFGS memory `m`; `*` is not converged in 20 000. One run each, laptop under
load; float32 at the weakest weight is noisy (line-search noise floor).

| case | 5 | 10 | 20 | 30 | 50 | 60 | 75 | 100 | 150 | 200 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 41² VLTI-like, float64 | 20000* | 13748 | 17502 | 8999 | 4527 | 4296 | 3695 | 2837 | 2316 | 2644 |
| 41² VLTI-like, float32 | 20000* | 20000* | 20000* | 20000* | 19524 | 14480 | 7225 | 3416 | 9401 | 9065 |
| 41² VLTI-like, float64, no `precondition` | 20000* | 20000* | 20000* | 20000* | 20000* | 12597 | 8471 | 6541 | – | – |
| 32² AMI, float64 | 3072 | 2882 | 2222 | 1546 | 725 | 667 | 724 | 754 | – | – |
| 32² AMI, float32 | fails at w ≤ 10 for every m | | | | | | | | | |

Wall time follows the steps (41², float64: 39 s at m = 5, 19 s at 50 and
100). The history is `2·m·n` floats: 2.7 MB at m = 100 for these images, 1.6 GB
for 10⁶ float64 pixels. Other cases:

- 12² ensemble corner fits (`tv`, `maxent`, `starlet`, 144 coordinates,
  converged): 547 965 steps (88 s) at m = 20, 269 727 (42 s) at 50, 74 256
  (19 s) at 100.
- A 3-parameter binary fit by L-BFGS: 14 steps for every m from 5 to 100.
- `StarPipeline`, `BinaryPipeline`, `ImagePipeline` and `DiskPipeline` (the
  test fixtures; `OrbitPipeline` makes no `fit` call): every `fit` is LM, which
  ignores the memory, so the pipelines do not need a default of their own.

Which cases need more than 50: the 41² image (float64 and float32) and the
ensemble corner fits; 32² AMI and parametric fits do not (50–60 is as good as
100). The smallest tested `m` within 10 % of the best everywhere it matters
is not one number (float64 41² keeps improving to 150, float32 41² is best at
100, 32² AMI at 60), so the default is 100, with one rule: it is cut so that the
history stays under 1 GiB (never below 5 steps, with a `UserWarning`), which binds
only above 6.7 × 10⁵ float64 coordinates (1.3 × 10⁶ in float32); none of the
cases above is affected.
An explicit `lbfgs_memory` is used as given.
