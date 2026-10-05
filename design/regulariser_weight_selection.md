# Choosing the regularisation weight w

Research note for Stage 3 of `design/imaging_plan.md` (`Problem`, `fit`, regularisers).
The objective is `0.5 chi2(image) + w R(image)`, with R one of maximum entropy (MEM),
total variation (TV) or total squared variation (TSV), minimised by LM or L-BFGS in JAX.
Status of the citations: papers marked (read) were opened; others were confirmed only by
title, authors and abstract in search results, so check details before quoting them.

## Decision (2026-10-01)

Implemented in Stage 3: `l_curve` with `LCurve.corner()` and `LCurve.discrepancy()` (target χ²/N = 1, the worst-fitted dataset deciding). Planned for Stage 5: Laplace evidence for quadratic and GP priors, and Gull–Skilling classic MaxEnt. Not planned for now: cross-validation, GCV/SURE, Deep Probabilistic Imaging, and Top-Set surveys over synthetic truths (the latter are being built separately for PDS 70). The research below is kept as background.

## Recommendation

1. **Implement now** (all in `imaging.py`, about 150 lines, no new dependencies):
   - `weight_sweep(make_problem, weights, ...)`: fit at a logarithmic grid of weights
     (default 10 to 12 points spanning about 6 decades), warm-starting from the
     previous weight, going from strong to weak regularisation. Returns a small
     `WeightSweep` record: `weights`, `chi2_red` per data block, `R`, the images, `info`.
     The name `l_curve` can be an alias, since the L-curve is just a plot of this record.
   - `WeightSweep.corner()`: maximum-curvature corner of (log chi2, log R), with the
     curvature computed on a smoothing spline of the sweep.
   - `WeightSweep.discrepancy(target=1.0)`: the weight at which chi2_red crosses the
     target, by interpolation in log w. It needs no extra fits.
   - Both criteria are read off the same sweep, so together they cost **N_w fits**
     (about 10 to 12, cheaper with warm starts) and **no Jacobian traces**.
2. **Optional third criterion, also cheap to write:** `cross_validate(make_problem,
   weights, folds)` with hold-out by baseline (see section 5). Cost is **K x N_w
   fits** (K = 4 or 5), no traces, and it needs only a mask on the whitened residuals.
   Ship it if time allows; label it experimental for sparse AMI coverage.
3. **Always plot** the sweep and show chi2_red per block next to the corner and the
   discrepancy weights. Tell users to choose a weight from the flat region, not a
   single point, and to inspect the images on either side (dorito does exactly this).
4. **Defer to Stage 5 (GP/sampling):** Bayesian evidence and Laplace log-determinants,
   hierarchical sampling of w or GP amplitude and length scale, and Hutchinson-trace
   GCV/SURE. They all need Jacobian traces or Hessian log-determinants, which are
   worth building once for the GP stage, not twice.
5. Keep w a plain float argument to the regulariser. Automatic selection is a
   *helper that returns a float*, never hidden inside `fit`.

## 1. What Charles et al. 2025 do (dorito)

Paper: Charles et al., "Image reconstruction with the JWST Interferometer",
[arXiv:2510.10924](https://arxiv.org/abs/2510.10924) (PASA). (read; the HTML text was
fetched and summarised, and the appendix figures could not be inspected.)

- Section 1.5.2 defines the L-curve as "plotting the log-likelihood term against the
  log-prior term of an optimised image for a range of lambda values". Figure 2 shows it
  for Io, with reconstructions at several points along the curve.
- Axes: log-likelihood term on one axis, regulariser (log-prior) term on the other,
  on logarithmic scales (this is how the figure is presented; the text does not say
  more).
- The text does **not** say that the likelihood is a reduced chi2, nor whether it is
  split by data type. It is described only as the Gaussian log-likelihood. Treat it as
  the total unreduced chi2 until the code (`wr137_disco.py`) confirms otherwise.
- Corner: chosen "from the region of the elbow", "the largest effect of the regulariser
  without incurring an excessive data penalty". This is by eye. There is no curvature
  criterion and no stated number of weights.
- Caveats they note: visual inspection is almost always enough because under- and
  over-regularisation are not subtle, and there is usually a wide interval of good
  weights; the method works in a well-behaved loss space.
- Regularisers: TV for Io and NGC 1068, MEM for WR 137. Additional L-curves for
  NGC 1068 and WR 137 are in Appendix A (not read).
- Note: the fetched text gave no numerical weights. Take these from `wr137_disco.py`
  (our reference notes list a MEM weight of 1e5 with a phase-centre prior).

## 2. Criteria for choosing w

| Criterion | Extra cost beyond a sweep | Needs | Main weakness |
|---|---|---|---|
| L-curve corner | none | sweep of N_w fits | corner ill-defined for smooth curves; can fail to converge (Vogel) |
| Discrepancy (chi2_red = 1) | none | sweep, correct error bars | biased by wrong sigma; ignores fitted dof |
| GCV / randomised GCV | 10 to 30 CG solves per weight | Jacobian-vector products | linearisation; needs Hessian of R |
| UPRE / SURE / MC-SURE | as GCV | known sigma, Jacobian products | needs known noise; risk on data, not on image |
| K-fold / leave-baseline-out CV | K x N_w fits | masking of data | sparse coverage; correlated closure phases |
| Evidence (Laplace), Gull-Skilling alpha | Hessian logdet per weight | quadratic or well-approximated R | not defined for TV/MEM without approximations |
| Hierarchical sampling | full posterior | probabilistic prior | Stage 5 only |
| Top-set survey (EHT) | 10^3 to 10^4 fits | synthetic data and truth | expensive; needs simulated truth |

### 2.1 L-curve and corner detection
- Hansen, "Analysis of discrete ill-posed problems by means of the L-curve", SIAM Review
  34 (1992), and Hansen and O'Leary, SIAM J. Sci. Comput. 14 (1993): log residual norm
  against log solution norm; corner at maximum curvature. Overview at
  [sintef.no/.../lcurve.pdf](https://www.sintef.no/globalassets/project/evitameeting/2005/lcurve.pdf).
- Castellanos, Gomez and Guerra, "The triangle method for finding the corner of the
  L-curve", Applied Numerical Mathematics 43 (2002) (abstract seen, no arXiv).
  Selects the vertex of the smallest-angle triangle on the log-log curve; robust to a
  noisy curve.
- Cultrera and Callegaro, "A simple algorithm to find the L-curve corner in the
  regularisation of ill-posed inverse problems",
  [arXiv:1608.04571](https://arxiv.org/abs/1608.04571): Menger curvature of three-point
  circles plus golden-section search. Needs O(10) fits, not a dense grid, but each
  step is a fit so it is sequential.
- Limits: Vogel, "Non-convergence of the L-curve regularization parameter selection
  method", Inverse Problems 12 (1996) 535,
  [doi](https://iopscience.iop.org/article/10.1088/0266-5611/12/4/013), shows the corner
  can fail for smooth solutions. Hanke, "Limitations of the L-curve method in ill-posed
  problems", BIT 36 (1996),
  [doi](https://link.springer.com/article/10.1007/BF01731984). For nonlinear
  imaging with local minima the curve can be non-monotonic, so warm-start carefully and
  report non-converged fits.
- Practical: put the two terms on the same footing (chi2 not chi2_red is fine, since
  a constant factor is a shift in log space), and fit the spline to the log-log points.

### 2.2 Discrepancy principle
- Morozov: choose the largest w with chi2 <= N_data (chi2_red about 1). Simple, no
  Jacobian, natural for interferometric data whose sigma we already carry.
- Sensitivity: if the error bars are wrong by a factor f, the chosen weight moves
  strongly, because chi2 is proportional to f^-2 (calibration errors in AMI/DISCO make
  this likely; `diagnose` already reports chi2_red per block). The fitted model also
  absorbs dof, so chi2_red at the optimum is below 1 for the true w when the number of
  effective parameters is not small: the correction is chi2 about N - df, with df from
  section 2.3. Using N_data is the standard, slightly over-smoothing approximation.
- Use per-block targets (CP, V2, kernel phase separately) and report which block
  binds. Do not mix blocks with very different error credibility into one target.
- Hansen's review and tutorial treatments cover this; for interferometry see the
  tutorial by Thiébaut and Young,
  [arXiv:1708.08390](https://arxiv.org/abs/1708.08390) (abstract seen).

### 2.3 GCV, UPRE and SURE for nonlinear problems
- Golub, Heath and Wahba, "Generalized cross-validation as a method for choosing a good
  ridge parameter", Technometrics 21 (1979). For a linear smoother y_hat = A(w) y,
  GCV(w) = N ||y - y_hat||^2 / (N - tr A)^2, with no need for sigma.
- Nonlinear extension: Ramani, Liu, Rosen, Nielsen and Fessler, "Regularization parameter
  selection for nonlinear iterative image restoration and MRI reconstruction using GCV
  and SURE-based methods", IEEE Trans. Image Process. 21 (2012) 3659,
  [PDF](https://web.eecs.umich.edu/~fessler/papers/files/jour/12/pre/ramani-12-rps.pdf).
  Both need the Jacobian of the reconstruction with respect to the data, and they
  approximate its trace by Monte Carlo.
- Monte-Carlo SURE: Ramani, Blu and Unser, IEEE Trans. Image Process. 17 (2008) 1540,
  [preprint](https://bigwww.epfl.ch/preprints/ramani0803p.pdf). Estimates the risk from
  a finite difference of the reconstruction at y and y + eps b (b random), so it is
  **black-box: two fits per probe, no Jacobian**. It needs sigma known, and the
  identity assumes the data are Gaussian and the map y -> x is weakly differentiable.
- For a Gauss-Newton fit at the optimum, df(w) = tr[J (J^T J + w H_R)^-1 J^T]
  (whitened J, H_R the Hessian of the regulariser). Hutchinson: df is about the
  average over k = 10 to 30 Rademacher probes z of z^T J (...)^-1 J^T z, each needing
  one CG solve using `jvp`/`vjp`. This is matrix-free and about the cost of a few LM
  steps, so it is affordable per weight in JAX. Gauge directions (flux, centroid) make
  J^T J singular; add a small ridge.
- Pitfalls: JAX autodiff through the optimiser is not needed (differentiate the
  model at the solution only); MEM and TV have poor Hessians near zero pixels (use the
  eps-smoothed TV); closure phases are nonlinear and correlated, so the linearised
  df is only a guide. Lucka et al., "Risk estimators for choosing regularization
  parameters in ill-posed problems: properties and limitations",
  [arXiv:1701.04970](https://arxiv.org/abs/1701.04970), compare these estimators and
  give their failure modes (abstract seen only).
- SUGAR (Deledalle et al., [arXiv:1405.1164](https://arxiv.org/abs/1405.1164)) gives
  the gradient of SURE for multiple weights. It is useful if we later have several
  regularisers, but it is not needed for a single R.

### 2.4 Cross-validation on interferometric data
- Hold out data, fit on the rest, score the held-out chi2 (or negative log predictive
  density) against w. It needs no sigma calibration beyond relative errors, and works
  with any nonlinear R. Cost: K x N_w fits.
- Split by **baseline or by hole**, not by random closure phase: closure phases share
  baselines, so a random split leaks information and picks under-regularised weights.
  For kernel-phase and DISCO data the same holds for the correlated linear
  combinations. In JAX this is a boolean mask in `whitened_residuals`.
- Warning: for a 7-hole AMI mask there are only 21 baselines and 15 independent
  closure phases. Dropping a fold removes a large part of the uv coverage, and the
  score is noisy. It is better suited to VLTI/CHARA multi-epoch or MATISSE/GRAVITY
  data (hundreds of baselines-channels). Report the fold-to-fold scatter and refuse to
  pick a weight when a single fold dominates.
- Data-splitting for interferometry appears in the EHT-era imaging work
  (see 2.6); I did not find a standard cross-validation implementation in the
  optical interferometry codes.

### 2.5 Bayesian evidence and Gull-Skilling
- For a quadratic prior (Gaussian, or a quadratic R such as TSV), the evidence in w is
  analytic in the linear-data case: log p(d|w) = -0.5 chi2(x*) - w R(x*) -
  0.5 log det(J^T J + w H_R) + 0.5 M log w + const, where M is the number of pixels
  the prior constrains. MacKay's evidence framework, "Bayesian interpolation", Neural
  Comput. 4 (1992) 415 ([MIT Press](https://direct.mit.edu/neco/article/4/3/415/5639/Bayesian-Interpolation)),
  optimises this to find w; the derivative gives gamma = the effective number of
  well-determined parameters, and the fixed point is 2 w R = gamma. This is the Gull
  (1989) and Skilling (1989, "Classic Maximum Entropy", in *Maximum Entropy and Bayesian
  Methods*, Kluwer, 45-52) "classic MaxEnt" recipe, generalised.
- Historic MaxEnt (chi2 = N) and its algorithm: Skilling and Bryan, "Maximum entropy image reconstruction: general
  algorithm", MNRAS 211 (1984) 111 ([PDF](https://files.batistalab.com/teaching/attachments/chem572/Skilling_Bryan1984.pdf)),
  implemented in MEMSYS. BSMEM (Buscher)
  uses it: alpha is picked by maximising the evidence (search-result summaries of BSMEM
  documentation, and of [arXiv:1007.4473](https://arxiv.org/abs/1007.4473)).
- Cost: needs log det of a pixel-space Hessian. For 10^3 to 10^5 pixels use
  stochastic Lanczos quadrature or a low-rank plus diagonal approximation (virgil's
  `log_evidence` uses a dense Cholesky factor instead, which is fine up to about 10^4
  data or pixels; a stochastic log-determinant is not built). It is
  exact for the GP prior with a linear-Gaussian model, which is why it belongs in
  Stage 5. For TV/MEM the Laplace approximation is uncontrolled: TV is not
  differentiable at zero, and MEM is on positive pixels only.
- Historic caveat: classic MaxEnt has been criticised for over-fitting with alpha
  from the evidence when the noise is under-estimated. It is only as good as sigma.

### 2.5a The error bars are a hyperparameter too (implemented: `imaging.error_scale`)

Every criterion above assumes the quoted error bars are right. On real PIONIER data they are not: χ² per point is 0.5–0.8 even for smooth images, which makes the discrepancy principle, classic MaxEnt and the evidence all over-regularise.

MacKay's evidence framework treats the noise precision β = 1/s² as one more hyperparameter, alongside the prior's weight, or its σ and ℓ. Maximising the Laplace evidence over β gives the **re-estimation formula**

    s² = χ² / (N − γ),   γ = Σ λᵢ / (1 + λᵢ),

where λᵢ are the eigenvalues of the likelihood's Gauss–Newton curvature in whitened prior coordinates. γ, the effective number of well-measured parameters, is the same quantity that appears in classic MaxEnt's fixed point, 2wR = γ.

The intuition: each well-measured parameter absorbs one datum's worth of scatter, so honest errors give χ² ≈ N − γ rather than N. Using χ²/N would underestimate s², as dividing by N rather than N − 1 does for a sample variance. The formula is MacKay 1992 (Neural Comput. 4, 415, [doi:10.1162/neco.1992.4.3.415](https://doi.org/10.1162/neco.1992.4.3.415)), eq. 4.10, and Bishop 2006, *PRML*, §3.5.2.

In virgil:
- `imaging.error_scale(model, data)` computes s at a `GaussianField` MAP, from the same Jacobian as `log_evidence`.
- `OIData.with_error_scale(s)` rescales the data, for a refit.

One fixed-point step usually suffices. The estimate assumes the model is adequate, because unmodelled structure inflates s.

### 2.6 What the codes actually do
- **BSMEM**: evidence-based automatic alpha as above (see 2.5).
- **MiRA** (Thiébaut 2008, SPIE 7013, 70131I; note that arXiv:0807.3020, which an
  earlier draft cited here, is Renard et al. 2008 and not MiRA): weight is a user-set hyperparameter, with a
  suggested procedure of scanning it. The tutorial
  [arXiv:1708.08390](https://arxiv.org/abs/1708.08390) states the same.
- **WISARD, SQUEEZE, OITOOLS.jl**: the user sets the weights. The Beauty Contests
  ([arXiv:1007.4473](https://arxiv.org/abs/1007.4473),
  [arXiv:1207.7141](https://arxiv.org/abs/1207.7141)) compare codes with the
  hyperparameters chosen by each team, with no shared criterion.
- **eht-imaging, SMILI**: Chael et al. 2018 use closure-only chi2 terms with several
  regularisers and weights chosen by hand
  ([arXiv:1803.07088](https://arxiv.org/abs/1803.07088)). The M87* imaging paper
  ([EHT IV, ApJL 875 L4](https://iopscience.iop.org/article/10.3847/2041-8213/ab0e85))
  runs a **parameter survey** of 10^3 to 10^4 images per pipeline on synthetic
  geometric data (ring, crescent, disk, double Gaussian) and keeps a **Top-Set**:
  parameter combinations that fit the real data (chi2 within a bound) and recover the
  synthetic truths. The Sgr A* paper
  ([arXiv:2311.09479](https://arxiv.org/abs/2311.09479), ApJL 930, L14, 2022) repeats the recipe. It is the
  most defensible published procedure, but it needs simulated truths and a large
  compute budget.
- **Comrade / THEMIS** (Bayesian sampling; Tiede, JOSS 2022): weights are not tuned;
  they are hyperparameters with priors and are sampled with the image (see
  Tiede et al.'s hierarchical interferometric Bayesian imaging, HIBI,
  [arXiv:2511.17706](https://arxiv.org/abs/2511.17706), for an application; the
  hierarchical CHIBI idea is at [arXiv:2606.04094](https://arxiv.org/abs/2606.04094),
  seen in search results only).

## 3. What suits virgil

- Data: 10^2 to 10^5 points, mostly closure-type, with correlated errors in
  DISCO/kernel phases. Discrepancy and L-curve work with the residuals we already have
  from `data_residuals`; both are insensitive to whether the model is nonlinear.
- Precision: the sweep uses the same `fit` and dtype policy as everything else; no
  new precision issues.
- Warm-started continuation makes a 10 to 12 point sweep only a few times the cost of
  one fit, because later fits start near a solution.
- Only the chi2 and R values enter L-curve and discrepancy. Store them, not the whole
  `FitResult`, unless the user asks for images (memory at 10^5 pixels x 12 weights is
  fine but wasteful).
- For a single regulariser there is one knob. With a centroid prior and a MEM prior
  together (dorito's recipe) fix the secondary weight and sweep the primary one.

## 4. The API as built (Stages 3 and 5)

The sketch originally proposed here, `weight_sweep`/`WeightSweep` with a spline corner and cross-validation, became the following.

```python
curve = l_curve(start, priors, data, MaxEntropy(1.0, path="env"),
                weights=jnp.logspace(3.5, 1.0, 11))      # strong to weak, warm-started
curve.discrepancy()          # χ²/N = 1, on the binding dataset; None if not crossed
curve.corner()               # maximum curvature of (log χ², log R), by finite differences
curve.classic_maxent(data)   # Gull–Skilling: −2wS = Σ λ/(λ + w); MaxEntropy sweeps only

log_evidence(gp_fit.model, data)   # Laplace evidence of a GaussianField MAP; compare over σ, ℓ
error_scale(gp_fit.model, data)    # MacKay's s = √(χ²/(N − γ)); then data.with_error_scale(s)
```

- Every criterion is a helper that returns a number; `fit` never chooses a weight.
- Cross-validation was dropped by decision (2026-10-01).
- The tutorials `imaging_rml` and `imaging_gp` show the workflow.

## 5. Costs summary

| Method | Fits | Jacobian products | Notes |
|---|---|---|---|
| L-curve corner | N_w = 10 to 12 (warm-started) | none | Stage 3 |
| Discrepancy | same sweep | none | Stage 3 |
| Cross-validation | K x N_w = 40 to 60 | none | dropped (see Deferred in `imaging_plan.md`) |
| Randomised GCV / MC-SURE | N_w (+ 2 per probe for MC-SURE) | k = 10 to 30 CG solves per weight (GCV) | Stage 5 with the evidence machinery |
| Laplace evidence | N_w | dense Jacobian, Cholesky log det (Lanczos probes not built) | Stage 5 (GP) |
| Sampling w | one long run | gradients only | Stage 5 (GP); Comrade-style |

## 6. Open questions

- Does the sweep use chi2 (total) or chi2_red per block on the L-curve axes? Recommend
  total chi2 for the corner (shift-invariant in log space) and per-block chi2_red for
  the discrepancy criterion.
- Should the discrepancy target subtract the effective dof? Not at first; add it when
  the Stage 5 trace machinery exists.
- Are the AMI/DISCO error bars trustworthy enough to make discrepancy the default?
  Test in MWE-B by inflating sigma by 1.5x and seeing how the chosen weight and the
  recovery metrics move. If they move a lot, prefer the corner and say so in the docs.
