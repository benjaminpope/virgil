# Automatic binary orbits from multi-epoch interferometry

Status: **design**, 2026-10-07. Nothing here is implemented. It extends
`design/visibility_orbits.md` ("V" below; requirements V13–V17 and the
stage-2 start tools), `design/thiele_innes_marginalisation.md` ("TI";
decisions D1–D6) and `design/detection_roc.md` (null simulators and
detection statistics), and answers two questions:

1. Can an orbit fit be initialized from the multimodal per-epoch
   binary-search grids?
2. Otherwise, what is the best way to get a binary's orbit
   **automatically and robustly** from multi-epoch V², closure phases,
   kernel phases and AMI/SAM data, with no published orbit to start
   from, so that it runs unattended on large volumes of data?

Science targets appear only as worked examples. Spelling is Oxford
(-ize).

> **Revised 2026-10-07 (§14).** A rerun after the peak catalogue and an
> adversarial review showed that per-epoch flux freedom makes false
> decisive peaks, so a mixture of Gaussians built from per-epoch peaks
> excludes the truth by construction. §14 replaces the search engine of
> §3.2–§4 (shared nuisances by default, exact scoring, plug-in candidate
> generators with automatic steps and budgets) and extends §9.3. Where
> §14 and earlier sections disagree, §14 wins.

## 0. Summary

* **Yes, and more than that.** For a binary of two point sources, the
  snapshot likelihood of epoch *e* depends on the orbit only through the
  companion's position and flux at that epoch. The multi-epoch
  likelihood is therefore *exactly* a sum of per-epoch likelihood maps
  evaluated along the orbit (§3.2). The per-epoch grid is not merely a
  seed: it is a sufficient summary of each epoch for orbit search, up
  to interpolation error. An orbit search should go **through the maps**
  and never commit to one peak per night.
* **The failure that prompted this** (§1) has two separate causes. The
  per-night gap is computed with the quoted errors and so is inflated by
  s². The pipeline also commits each night to a single peak. The quick
  win (§12, PR A) fixes the first by marginalizing each night's error
  scale analytically under its Jeffreys prior, which gives the surface
  −(ν/2) ln χ²(θ). That surface is invariant to a constant rescaling of
  the errors. The redesign fixes the second.
* **Recommendation (§4):** a five-step pipeline:
  1. per-epoch maps with scale-marginalized likelihoods and a **peak
     catalogue** (every peak within a calibrated Δ, each with a Laplace
     Gaussian);
  2. a dense, deterministic **search over (P, e, T₀)** in which the
     Thiele–Innes constants are solved by EM on each epoch's
     mixture of Gaussians plus a floor, and each candidate is rescored
     exactly on the maps (GPU, vmapped);
  3. clustering into **modes** by predicted sky positions;
  4. per-mode refinement of the full model on the visibilities;
  5. NUTS **within each mode** (at least two chains per mode), with mode
     weights from per-mode evidence, and stacking as a check for
     misspecification.

  Nested sampling on the map surrogate is the independent cross-check
  in validation. It is not the default.
* **Why not a global sampler on the visibilities?** Nested sampling and
  SMC on 20–30 dimensions of narrow fringe-alias modes cost about 10⁷
  full-likelihood calls per system (§3.3). They also fail stochastically
  when they miss a narrow mode, which is the failure we need to rule
  out. Deterministic search on the maps costs about 10⁶–10⁷ *cheap*
  evaluations and is reproducible.
* **Benchmarking and reliability are deliverables** (§9). Every step is
  timed against its alternatives on M4, OzSTAR CPU and A100, with a JAX
  profiling run. A fixed adversarial suite of simulated systems is
  kept as a regression test, small in CI and in full on OzSTAR, with a
  target of **zero silent failures**.

## 1. The failure, diagnosed

Worked example (Gl 229 Ba–Bb, GRAVITY closure phases, validation repo).
Every orbit fit landed in an i ≈ 110° mode, against the published
i ≈ 31°. The per-night closure-phase likelihoods have several peaks
within Δloss < 1 of each other once the noise is scaled correctly. Raw
χ²/N on the quoted errors is 4–25 per night, so the error scale s has
s² ≈ 2–20.

**The gap is inflated by s².** `epoch_positions` computes each night's
`gap` as the log-likelihood difference between the best peak and the
best rival more than λ/B_max away, with the quoted σ. If the true errors
are sσ, the log-likelihood difference is Δχ²/(2s²), not Δχ²/2. A
true gap of 0.5 is reported as 1–10, so `min_gap=5` treats ambiguous
nights as decisive. `start_from_positions` then seeds `starting_orbits`
with whichever peak happened to be highest, which on at least one night
was the wrong one.

**Three further effects compound it:**

* **The covariance is wrong.** The covariance from the Hessian with the
  quoted errors is too small by s² on each night. Nights with different
  s are mis-weighted against each other in `starting_orbits`.
* **The ranking is wrong.** `start_from_positions` ranks candidates with
  `noise=None`, i.e. the quoted errors. The nights with the largest s,
  the worst-calibrated, dominate the ranking.
* **The per-night ranking disagrees with an independent fit.** The
  validation session's per-night fits (`fit_nights`: a binary with
  linear motion and a fitted `phi_scale`) rank the peaks differently
  from `epoch_positions`:
  * on one night (2023-12-29), `epoch_positions` picked a peak that the
    independent fit puts 8.9 behind the best, and reported a gap of
    22.6 for it;
  * on another (2024-02-27), its position was not among the
    independent fit's top five.

  The 180° mirror hypothesis (one night taken from the point
  reflection −r of its true peak) is **ruled out**. Besides the quoted
  errors, the candidate causes are:
  * the grid's position or flux range (a peak cut off by the flux axis,
    or a refinement leaving the grid cell);
  * static versus within-night motion: `epoch_positions` fits a static
    binary, while the independent fit includes the companion's motion
    over the night, which for a short period moves it by a fraction of
    λ/B.

  PR A tests for this directly (§12).

**The scale-marginalized surface.** Each night has an unknown error
scale s_b per observable block b (V², closure phase). Give s_b its
Jeffreys prior 1/s_b and integrate:

  ∫ s^−ν exp(−χ²_b/2s²) ds/s ∝ (χ²_b)^(−ν/2),

so that

  m(θ) = −Σ_b (ν_b/2) ln χ²_b(θ),   ν_b = n_independent of block b.

Throughout this note **N = ν = `n_independent`**: raw χ²/N, ŝ² = χ²/ν
and m all use it. For correlated closure phases from four or more
telescopes, `whitened_residuals` returns `n_residuals` > `n_independent`
entries; the periodic-penalty terms add to χ² but not to ν. At the best
fit, E[χ²_b] = (ν_b − p_b)s_b² with p_b fitted parameters, so "χ²/N ≈ s²"
holds to O(p/ν).

**m is a search surface with Gaussian normalization.** The integral
assumes a normalization s^−ν for every block. That is true for V²,
kernel phases, DISCOs and correlated closure phases, but not for
uncorrelated unprojected closure phases, which `model_loglike`
normalizes as a von Mises density (−ln 2π − ln I₀(κ) with κ = 1/(sσ)²).
That density tends to the uniform 1/2π as s → ∞, so the likelihood
tends to a constant there and the exact Jeffreys marginal is improper.
m therefore uses the Gaussian (small-σ) normalization for every block,
on the same chord residuals 2 sin(Δ/2)/σ, as the correlated closure
phases already do. It is only a search surface: refinement and sampling
(§6) use virgil's exact likelihood with fitted scales. Where sσ ≳ 0.5 rad
(weak closure phases with large s), m and the exact profile diverge.
There, the option `s_max` bounds the scale and the marginal is
integrated numerically on a 1-D quadrature in ln s per block; PR A tests
that regime.

The profile over s gives the same function as m up to a constant. Two
properties make m the right surface for search:

* m is **invariant to a constant rescaling of the errors**, which is
  exactly the property the gap lacked.
* Near a peak, m ≈ −χ²/(2ŝ²), so its gaps and curvatures are the
  quoted ones divided by ŝ² = χ²_min/ν.

**Caveat: correlated errors.** When s² ≫ 1 the excess is usually
correlated (calibration, wavelength-correlated closure phases), and the
effective number of degrees of freedom ν_eff < ν. Then m is still
over-confident by about ν/ν_eff. §6 calibrates this.

## 2. Requirements for unattended orbits

* **A1 No single-peak commitment.** No step may reduce an epoch to one
  position unless that epoch is decisive on the scale-marginalized
  surface, with a calibrated threshold.
* **A2 Scale-aware everywhere.** Search, gaps, covariances and rankings
  use the scale-marginalized likelihood (or fitted scales), never the
  quoted errors alone. Every report leads with **raw χ²/N on the quoted
  errors** and the fitted s per dataset and block.
* **A3 All modes, weighted.** The output is a set of modes with weights
  and per-mode posteriors, never "the best fit". Modes are identified by
  predicted sky positions (as `chain_starts` does), not by elements.
* **A4 Deterministic and reproducible.** The same inputs and keys give
  the same modes on CPU and GPU, and across JAX versions to within
  documented tolerances: mode identity exactly, weights to 10⁻², posterior
  quantiles within Monte Carlo error. Nothing depends on a JAX version.
* **A5 Flag, never silently fail.** Every known failure mode (§7.2)
  raises a structured flag in the result. "No flag" must mean that the
  checks passed, not that they were not run.
* **A6 Jeffreys priors.** Log-uniform P, a (or total mass with a
  distance) and flux. Isotropic orientation. Uniform phase. Uniform e
  (interim, TI D2). Log-uniform error scales. Gaussian ψ priors are
  proposals only, reweighted per TI D1.
* **A7 Serializable.** Every intermediate (maps, peak catalogue, search
  result, modes) saves to disk and loads in another process (#268 item 4),
  so that search and sampling can be separate jobs.
* **A8 Batched.** One compiled kernel serves many systems with the same
  shapes. Shapes are bucketed so that a survey compiles a handful of
  times, not once per system.
* **A9 Observable-agnostic.** V², closure phases, kernel phases, AMI/SAM
  DISCOs and correlated closure phases all enter only through
  `model_loglike` / `whitened_residuals` and `n_independent`.
* **A10 Extended scenes degrade gracefully.** For scenes beyond two
  point sources the maps are a proposal, not the likelihood (§3.2). The
  pipeline must still find the mode and say that the map and the
  visibilities disagree.

## 3. Options

### 3.1 Fix the current pipeline

Fit a per-night scale at each peak before computing gaps (or use m of
§1), and keep every peak within a calibrated Δ instead of only the best.
`starting_orbits` then needs a position per night, so ambiguous nights
must either be dropped from seeding (as now, with a correct gap) or
enumerated: one `starting_orbits` call per combination of peaks. With K
peaks on each of n ambiguous nights that is Kⁿ calls (3⁶ = 729). It is
feasible for a few nights and explodes for surveys.

* **Pros:** small changes, and the quick win (§12 PR A) is exactly
  this, minus enumeration.
* **Cons:** dropping ambiguous nights throws away information that
  would break period aliases. Enumeration does not scale. The final
  ranking still sees only the candidates seeded from the decisive
  nights.

**Verdict:** land the scale-aware gap as the quick win. Do not build
enumeration.

### 3.2 Orbits through the likelihood maps

**The key fact: the likelihood factorizes through the maps.** For a
two-point-source binary, the snapshot model of dataset e is
`BinaryModelCartesian(dra_e, ddec_e, f_e)`, so

  ln L(orbit, f, s) = Σ_e ℓ_e(r_e(orbit), f_e, s_e),

where ℓ_e is the per-epoch log likelihood on the (dra, ddec, flux) grid,
and m_e is ℓ_e with the scales marginalized. The orbit enters only
through r_e = (dra, ddec) at t_e. A map over position and flux is
therefore a *sufficient* summary of epoch e for any orbit, up to
interpolation. Scoring an orbit means evaluating n interpolants: no
visibility is touched after the maps are built.

**Representation.** Fringe peaks have width about (λ/B)/SNR, which is
often well below a practical grid step. A raw grid interpolant blurs
them. Store instead:

* **A peak catalogue.** All local maxima of the profile M_e(r) = max_f
  m_e(r, f) within Δ_keep of the best, each refined by `fit` and
  carrying its mean μ_ek, its Laplace covariance Σ_ek (from the Hessian
  of −m at the maximum of m, §12 PR A), its height h_ek, its flux and its
  log-determinant.
* **A floor.** The coarse grid M_e itself, interpolated
  (bilinear/bicubic), which covers positions far from all catalogued
  peaks. This matters for epochs with no detection, whose map is
  nearly flat. These epochs then contribute (correctly) almost nothing,
  instead of being forced onto a noise peak.

The epoch score is

  S_e(r) = max( logsumexp_k [h_ek − ½ d²_ek(r)],  M̃_e(r) ),

where d²_ek is the Mahalanobis distance from r to μ_ek under Σ_ek, and
M̃_e is the floor M_e with every point within d_ek < 3 of a catalogued
peak masked to −∞. This is a mixture of Gaussians (MoG) with a floor.
The combination is a **hard maximum, not a log-sum-exp**: a smooth
maximum would overshoot each peak by up to ln 2 wherever the coarse
floor comes close to the peak height. With the mask, S_e(μ_ek) = h_ek
exactly when the peaks are well separated (the other components' tails
add at most e^(−d²/2)). Near a peak S_e is the Laplace approximation;
far away it is the grid.

**The flux is bounded, f ≤ 1, in the grid and in every refinement
prior.** The flip f ↔ 1/f with r ↔ −r is exact for every f: with the
companion at r and flux f, V = (1 + f e^(−2πiu·r))/(1 + f), and with the
companion at −r and flux 1/f, V′ = e^(2πiu·r) V, which is the same
visibility times a pure phase gradient. V², closure phases and kernel
phases are therefore identical. Without the bound every epoch would
carry an exact twin peak, the catalogue would double, and EM would split
responsibility between twins in every epoch. With f ≤ 1 the flip
survives only as a near-twin when f is close to 1.

**Flux.** By default the flux is profiled per epoch, which is robust to
different filters. As an option, flux is tied per instrument/filter:
keep the flux axis of the map and evaluate Σ_e m_e(r_e, f) on the shared
flux axis, then take the maximum (or log-sum-exp under the log-uniform
prior) over f. This costs a factor n_flux and helps near-equal-flux
pairs, where the near-twin at f ≈ 1 is otherwise almost free per
epoch.

**Search option (a): (P, e, T₀) with linear Thiele–Innes constants.**
At fixed φ = (P, e, T₀), r_e = K_e(φ)ψ is linear in ψ = (A, B, F, G).
`starting_orbits` exploits this with one Gaussian per epoch. With a MoG
per epoch, the profile over ψ is no longer one least-squares solve, but
EM makes it a few:

* **E-step:** responsibilities γ_ek ∝ exp(h_ek) N(K_eψ; μ_ek, Σ_ek),
  plus an outlier component per epoch with a **constant** density
  exp(F_e), where F_e is the epoch's floor level (the maximum of M̃_e).
  This keeps outlier epochs from dominating. The outlier component
  does not depend on ψ, so this is the standard Gaussian-plus-uniform
  mixture and the iteration is genuine EM: it increases the surrogate
  objective monotonically. The interpolated floor M̃_e, which depends on
  ψ non-quadratically, enters only the rescoring below, never the
  M-step.
* **M-step:** a weighted linear solve for ψ:
  ψ = (Σ_ek γ_ek K_eᵀΣ_ek⁻¹K_e + εI)⁻¹ Σ_ek γ_ek K_eᵀΣ_ek⁻¹μ_ek,
  with a tiny ridge ε for conditioning only, so that this is the profile
  over ψ and no prior.
* **Starts per φ:** a few. Use the best component of every epoch, and
  the best component of the decisive epochs with the ambiguous ones
  left free (γ uniform), plus R random assignments. EM converges in
  3–10 iterations, because peaks are separated by many σ.
* **Rescore** each converged ψ on the exact S_e (not the EM bound).
  **Pruning uses this prior-free profile score only.** The TI Gaussian
  prior on ψ is not isotropic (TI §2.2), and pruning on a score that
  includes it could discard face-on modes because of a proposal that TI
  D1 rejects. This keeps TI D3 (starting points rank by fit, not by
  posterior).
* **Mode evidence, after pruning.** For each surviving mode, draw ψ from
  its conditional Gaussian under the TI proposal prior (width
  s₀(P/P₀)^(2/3), TI D6), importance-reweight the draws to the
  invariant target prior (TI D1), and report PSIS k̂. These map
  evidences are a proxy for mode weights. Where k̂ > 0.7 the mode is
  kept, with its weight left to the visibility evidence of §5.3.

The cost per φ is (starts) × (iterations) × n × K solves of 4×4
systems. It is near-linear and vmaps over φ.

**Search option (b): dense quasi-random search over all elements.** A
Sobol (or stratified) set over (ln P, e, phase, cos i, Ω, ω, ln a),
scored on S_e directly. It needs no linear structure, so it also
handles models where the positions are not linear in ψ (e.g. a fitted
North angle or plate scale per epoch, which rotate or scale r_e). It
costs about 10³–10⁴ times more points than option (a) for the same
coverage in four extra dimensions. That is affordable on a GPU only
because each point costs n interpolations.

**Period sampling.** Phase coherence across a time span T requires
δP ≲ P²/(k T) with k ≈ 10, so a log-P axis needs about kT/P_min
points. For T = 5 yr and P_min = 10 d that is about 2 × 10³ periods;
with 20 eccentricities and 36 phases, about 10⁶ φ. Short periods
dominate the count. A strong external period (an RV solution) collapses
this axis.

**Verdict:** option (a) is the search engine. Option (b) is the
fallback when ψ is not linear, and a cross-check in validation.

### 3.3 Global samplers

| Method | On the visibilities (20–30-D) | On the map surrogate (3-D or 7-D) | Mode weights | Failure mode |
|---|---|---|---|---|
| Nested sampling (jaxns, blackjax NS) | n_live × H × slice steps ≈ 10³ × 60 × 30 ≈ 10⁶–10⁷ likelihood calls, each a full visibility model; hours per system on GPU | seconds to minutes | evidence per mode, directly | misses a narrow mode if n_live is too small; stochastic |
| Tempered SMC (blackjax) | ≈ 10⁴ particles × 30–100 temperatures × MCMC steps; similar to NS | seconds | particle mass, noisy for small modes | weight collapse; tuning of the tempering ladder |
| Parallel tempering | many chains × long runs; poor scaling with the number of modes | cheap | no direct weights (needs thermodynamic integration) | swaps stall between narrow modes |
| Grid peaks → many local fits → NUTS per mode | 10³–10⁴ cheap map evaluations, then a few fits and NUTS per mode | — | per-mode evidence (Laplace, then importance-sampled) | a mode absent from the search is never found, which is why the search must be dense and deterministic |

The information H (ln of prior volume over posterior volume) is large
here: seven elements each pinned to a fraction of their prior range
gives about 7 × 8 ≈ 56 nats. NS cost grows linearly with H, and with
the number of modes it must keep populated.

**Verdict:** global samplers on the *visibilities* are a validation tool
at best. On the *map surrogate* they are cheap, and NS gives evidences
and mode masses independently of our search. It belongs in the
benchmark (§9) and in SBC as the second root for mode weights, not in
the default path. blackjax and jaxns are not dependencies today; the
benchmark decides whether one becomes an optional extra.

### 3.4 Comparison

| | 3.1 fix | 3.2(a) maps + TI EM | 3.2(b) maps, dense | 3.3 NS on visibilities |
|---|---|---|---|---|
| Uses ambiguous epochs | no (drops them) | yes | yes | yes |
| Deterministic | yes | yes | yes (fixed Sobol set) | no |
| Cost per system (est.) | minutes | minutes CPU, < 1 min GPU | GPU minutes | GPU hours |
| Extended scenes | proposal only | proposal only | proposal only | exact |
| Non-linear nuisances (North, plate scale) | no | at fixed nuisance | yes | yes |
| Mode weights | no | map evidence (proxy) | map evidence (proxy) | yes |

## 4. Recommended pipeline

```
data (Epochs) ──► 1. maps + peak catalogue (per dataset; scale-marginalized; Δ_keep)
             ──► 2. orbit search on maps: (P, e, T₀) grid × TI-EM over each epoch's MoG
             ──► 3. modes: cluster candidates by predicted positions (≤ ½ λ/B_max)
             ──► 4. per-mode refinement: fit() of the full model on the visibilities
             ──► 5. per-mode NUTS (≥ 2 chains per mode) → per-mode evidence → weights
             ──► OrbitResult: modes, weights, draws, raw χ²/N, s, flags
```

* **Step 1** is the existing grid machinery plus `fit` and a Hessian of m
  per peak. Δ_keep defaults to 10 nats on m and is calibrated by
  simulation (§6). Grid steps must be ≤ λ/(3B_max), checked and flagged.
* **Step 2** keeps the top N_keep (default 10⁴) φ by marginal score, and
  their ψ.
* **Step 3** reuses `_distinct` (positions at every snapshot within
  ½ λ/B_max make one mode). Every mode within Δ_mode (default 20 nats on
  the map score) of the best is returned in the catalogue of modes,
  however many there are. The contract on `n_modes_max` (default 8) is:
  all modes are refined in step 4 (fits are cheap), and NUTS runs on
  the n_modes_max modes with the largest Laplace evidence. The omitted
  mass is bounded by the Laplace evidences of the unsampled modes. If
  that bound exceeds 0.01 of the total, the pipeline stops before
  sampling with the flag "mode overflow" and returns the refined modes,
  marked incomplete. It never truncates silently.
* **Step 4** fits the full scene (extended components, hierarchical
  scales, North angles) from each mode's best orbit. A fit that leaves
  its mode (its positions move by more than ½ λ/B_max) is recorded as
  "mode merged" and deduplicated. **The map check compares like with
  like:** at the refined orbit's positions, Σ_e S_e (the interpolated
  map score) is compared with Σ_e m_e recomputed directly from the
  visibility residuals of a point-source snapshot, with the same
  scale marginalization and flux profile. It is not compared with the
  refinement's own log likelihood, which uses hierarchical scales and
  the exact normalization. For point-source binaries the two agree to
  within the interpolation tolerance measured in PR C. A larger
  difference flags a map that is too coarse. A separate comparison of
  the point-source m_e with the full scene's m_e at the same positions
  flags extended emission or a third body.
* **Step 5** runs NUTS per mode with `chain_init_params`, at least two
  chains per mode, so that R̂ is meaningful within a mode. A chain that
  leaves its mode is detected by its positions and reported. The weight
  of mode j is its posterior mass Z_j/ΣZ (§5.3).

**What the maps buy over the current tools:** ambiguous nights constrain
the search instead of being dropped; no night is seeded from a single
peak; every candidate is scored on all epochs in microseconds; and the
whole search is one compiled kernel, batchable across systems.

## 5. Modes

### 5.1 Which ambiguities exist

* **Fringe aliases.** Per-epoch peaks spaced by about λ/B. They are
  handled by the peak catalogue and by mode identity at ½ λ/B_max.
* **Period aliases.** P versus P/k or orbits that differ by an integer
  number of cycles between epochs. They are distinct modes, found by the
  dense period axis and reported with their weights. V17's prior
  configurations (literature priors with aliases) become alternative
  prior weights over the same modes.
* **(Ω, ω) → (Ω + 180°, ω + 180°).** Exact for positions without RVs:
  the same sky motion. It is one mode by the positional identity, and
  reported as Ω mod 180°, as V §6 requires. It is never counted twice
  in the weights.
* **ω → ω + 180° alone (r → −r at every epoch).** Exact for V²-only data,
  broken by closure phases. With V² only it is reported as one mode
  with a "primary/secondary unresolved" flag; with weak closure phases
  it is two modes with weights.
* **Per-epoch flips (f ↔ 1/f, r ↔ −r).** Exact for every f (§3.2), and
  removed by the bound f ≤ 1 except near f = 1, where a near-twin peak
  remains on single epochs. The catalogue keeps both, and the orbit
  decides. With flux tied across epochs this ambiguity is mostly
  broken.
* **Inclination sense (i ↔ 180° − i).** Determined by the sense of
  motion; a single flipped epoch can swap it (§1). It is handled by
  scoring all peaks.
* **Sign conventions.** These are virgil's, from
  `orbit_scene_joint_fitting.md` §2. Every reported mode carries its
  elements in those conventions and also its predicted positions, which
  are convention-free.

### 5.2 Reporting

Each mode reports:

* weight with its uncertainty;
* the per-mode posterior summary (quantiles; Ω mod 180°);
* predicted (sep, PA) at each epoch and at a requested forecast date;
* raw χ²/N per dataset on the quoted errors, and the fitted scales;
* per-mode diagnostics: R̂, ESS, divergences, k̂.

A system with two modes of comparable weight is reported as bimodal. It
is never collapsed. A forecast is a mixture over modes.

### 5.3 Weighting: evidence, LOO and stacking

Within one model, the correct weight of a mode is its posterior mass,
the integral of the posterior over its basin. Mode weights are therefore
**per-mode evidences** Z_j:

1. **Laplace evidence** at each refined fit (cheap; available after
   step 4).
2. **Importance-sampled evidence** from each mode's NUTS draws, with a
   Student-t proposal fitted to the draws (or a bridge sampler). PSIS k̂
   is the diagnostic; if k̂ > 0.7 the Laplace value is reported with a
   flag.
3. **Cross-check:** nested sampling on the map surrogate gives
   independent mode masses for point-source binaries (in validation and
   as an optional mode).

**Stacking** (Yao et al. 2022, stacking for non-mixing computations)
weights modes by leave-one-epoch-out predictive performance (PSIS-LOO
over epochs). It is not a posterior mass. It is the right tool when the
model is misspecified, which raw χ²/N ≫ 1 signals. It is reported beside
the evidence weights. When they disagree strongly (e.g. a mode whose
evidence weight exceeds 0.9 has a stacking weight below 0.1), the result
is flagged "misspecified: weights unreliable". PSIS-LOO by epoch is also
the per-epoch outlier diagnostic (k̂ > 0.7 for an epoch means the orbit
depends on that epoch alone).

## 6. Noise

* **In the search (steps 1–3):** independent per-dataset, per-block
  Jeffreys scales, marginalized analytically (m of §1). This is
  conservative relative to the hierarchical prior and needs no
  hyperparameters. It is the only noise model that keeps the maps
  independent per epoch.
* **In refinement and sampling (steps 4–5):** the tutorial's hierarchical
  scales (`hierarchical_scales`, centred by default) per block across
  epochs. This is where population information about the instrument
  enters.
* **Effective degrees of freedom.** ν_eff per dataset defaults to
  `n_independent`. It is calibrated, not assumed. The current
  `detection.bootstrap_null` cannot do this: it sign-flips each whitened
  residual independently, which destroys exactly the wavelength, frame
  and baseline correlations that make ν_eff < ν. The calibration uses a
  **grouped (block) sign-flip bootstrap** instead: one sign per group
  of residuals that share a calibration error, i.e. per exposure (or
  frame) and per baseline or triangle, across all wavelengths. This is
  a new `groups=` option of `bootstrap_null`, added in PR C. Applied at
  the best peak, it gives the null distribution of the gap between the
  true peak and its best alias. Δ_keep and the "decisive"
  threshold are set to its 99.9th percentile. If that percentile exceeds
  the Gaussian expectation by a factor of more than 2, ν_eff is reduced
  so that it does not, and the reduction is reported. This is the same
  machinery as the detection ROC work, reused.
* **Raw χ²/N** on the quoted errors leads every report. s > 2 (raw
  χ²/N > 4) on a dataset is flagged "calibration problem". The orbit is
  still reported, because a rescaled χ²/N ≈ 1 is tautological and must
  not be read as a good fit.

## 7. Automation

### 7.1 Diagnostics, with no human in the loop

| Stage | Check | Pass |
|---|---|---|
| Maps | grid step ≤ λ/(3B_max); best peak not on the grid edge; flux not at its bound | all |
| Maps | raw χ²/N and ŝ per block; ν_eff calibration | recorded; s ≤ 2 or flag |
| Maps | per-epoch detection Δχ² against the null threshold | below threshold: epoch marked "no detection" (it still contributes its flat map) |
| Search | each surviving mode found from ≥ 2 independent search seeds (two Sobol scrambles / two EM start sets) | identical mode set |
| Search | best period not within 5% of the prior bounds | or flag |
| Search | map score converged in the period step (halving δP changes no mode) | in validation and on a sample of systems |
| Refinement | fit converged; stays in its mode; map–visibility score difference within tolerance | or flag |
| NUTS | per mode: R̂ < 1.01, bulk ESS > 400, 0 divergences, E-BFMI > 0.3, no chain left its mode | or flag |
| Weights | k̂ < 0.7; evidence and stacking weights consistent | or flag |
| Snapshot | companion motion over `spread_days` < 0.1 λ/B_max (V §6) | or flag "use per-sample model" |

### 7.2 Failure modes that must be flagged, never silent

* **Too few constraining epochs.** Fewer than four detected epochs
  cannot fix seven elements. The result is an "arc only" set of
  constraints (prior-dominated P and e), flagged.
* **Unresolved epochs.** Separation < λ/(2B_max): position and flux are
  degenerate and the map is a ridge. Marked per epoch.
* **Companion at the grid edge or beyond the field.** Bandwidth smearing
  and the field of view are checked from the data.
* **Inflated errors** (s > 2).
* **More modes than n_modes_max**, or none within Δ_mode.
* **Period at a prior bound.**
* **Within-night motion** too large for snapshots.
* **Map–visibility disagreement:** extended emission, a third body, or a
  map that is too coarse.
* **Non-mixing within a mode, or a chain hopping modes.**
* **Unstable weights:** k̂ > 0.7, or evidence and stacking disagree.
* **Search not reproducible** between seeds.

Each flag is a code in `OrbitResult.flags`, with a message and the
evidence (numbers) behind it. The summary table has one row per system
and one column per flag.

### 7.3 Compute (estimates, to be replaced by §9 measurements)

* **Maps:** about 200 × 200 × 30 points × 10³–10⁴ visibilities per
  dataset. On an A100, a 81 × 81 × 60 grid took 0.02 s warm
  (2026-10-03 profile), so about 0.1–1 s per dataset. On a CPU, 1–30 s.
* **Peak refinement:** about 5–20 peaks per dataset × one binary `fit`
  (0.04 s warm) and a 3 × 3 Hessian. This is launch-bound on GPU, so
  vmap the fits over peaks and datasets (all have the same 3-parameter
  shape).
* **Search:** 10⁶ φ × 4 starts × 8 EM iterations × n = 10 epochs ×
  K = 5 components of 4 × 4 work is about 10¹⁰ flop: seconds on A100,
  minutes on CPU.
* **Refinement:** ≤ 8 modes × one full `fit`: seconds to minutes.
* **NUTS:** dominant. The tutorial's 4 chains × 2000 steps on 8 nights
  is of order an hour on a CPU node; launch-bound on GPU. NUTS on small
  models stays on CPU (2026-10-03 profile).
* **Per system:** about 10–30 min on GPU + CPU, 1–3 h CPU only; NUTS is
  most of either.
* **Surveys:** steps 1–3 batch across systems on GPU (§9.3); step 5 runs
  as CPU array tasks.

## 8. Validation

* **SBC v3** (virgil-validation). It starts from this pipeline's modes,
  one chain set per mode, replacing the single start that made v2 fail
  through mode trapping. It needs simulated multi-epoch sets with
  realistic multimodality: few baselines, weak closure phases, s from
  the observed distribution (1–5), near-equal fluxes and sparse epochs.
  Ranks are computed on the mode-weighted mixture. Weights come from
  §5.3, and the NS-on-maps weights are the independent check.
* **Injection–recovery.** Over (P, e, separation/λ·B⁻¹, flux, n_epochs,
  s), measure the fraction of systems whose true mode is found, ranked
  and weighted correctly (§9.2 metrics).
* **Gl 229 Ba–Bb rerun** (worked example). The pipeline must report the
  published i ≈ 31° mode with the dominant weight, or report i ≈ 110° as
  a weighted alternative with the ambiguous nights named. It must never
  report i ≈ 110° alone. The quick win (PR A) alone is checked on the
  same data first.
* **Published-orbit binaries** (virgil-validation#69 OiDB collections).
  Every system with a published visual orbit is run blind. Pass means
  the published orbit lies in a reported mode with weight above 0.05 and
  inside that mode's 99% interval. The independence rule holds:
  crosscheck orbits come from the validation repo's own evaluator.

## 9. Benchmarking and reliability

The pipeline must be efficient and bulletproof, and both are measured,
not asserted. The benchmark suite is a deliverable of its own (stages B1
and B2 in §12), kept as a regression test: small in CI and in full on
OzSTAR.

### 9.1 Efficiency benchmarks

Each step is benchmarked separately, against the alternatives of §3:

| Step | Variants compared |
|---|---|
| Per-epoch maps | `epoch_positions` grid (now) vs `epoch_maps` (batched over datasets) vs `linear_flux_grid` (analytic flux marginal) |
| Per-epoch noise scale | quoted errors vs scale-marginalized m vs a per-peak `fit` with noise terms |
| Candidate scoring | rank on visibilities (`rank_orbits`) vs interpolated map sum (S_e) vs MoG/TI-EM linear solve |
| Search | `starting_orbits` (decisive epochs only) vs TI-EM over (P, e, T₀) vs dense 7-D Sobol on maps |
| Multi-start local fits | sequential `fit` loop vs vmapped `fit` over starts |
| NUTS per mode | per-mode chains, vectorized vs parallel chains; CPU vs A100 |
| Global samplers | NS (jaxns, blackjax), tempered SMC (blackjax) and PT, on the map surrogate and on the visibilities (small cases only) |

**Measured for each:**

* wall time, split into compile time (first call) and run time (warm
  median of ≥ 5 calls, `block_until_ready`);
* peak memory (host RSS from `resource.getrusage`, device from
  `jax.devices()[0].memory_stats()`, and `XLA_PYTHON_CLIENT_PREALLOCATE=false`);
* throughput in **systems per hour** at steady state;
* the number of compilations (`jax.log_compiles` count);
* compile time and compile memory (§9.5).

**Platforms:**

* Apple M4 CPU, small cases only, one process, a few minutes at most;
* OzSTAR CPU node (8 cores);
* OzSTAR A100 (1 GPU).

**Scaling axes**, each varied with the others at the reference values:

* epochs n ∈ {4, 8, 16, 32};
* observables per dataset ∈ {10², 10³, 10⁴};
* map grid ∈ {64², 128², 256²} × {16, 32} fluxes;
* candidates N_φ ∈ {10⁴, 10⁵, 10⁶, 10⁷};
* peaks per epoch K ∈ {1, 3, 10}.

Each variant also records its result quality (true mode found? score
error against the exact visibility likelihood), so that a faster variant
is never preferred at the cost of reliability.

**Profiling run** (with the jax-profiling procedure: an opt-in
`jaxprof.py` helper, `StepTraceAnnotation` per step, compile outside the
trace, `jax_trace_summary.py` for headless summaries, and an XLA HLO dump
for the search kernel). It must answer:

* **Recompiles.** Does any step recompile per epoch, per mode or per
  system? `fit` once did, before #104. Count compiles per system across
  10 systems of the same bucket; the target is zero after the first.
* **Launch-bound kernels.** Device busy fraction per step. Earlier
  profiles found grids GPU-friendly but about 50% busy at small batch
  sizes, and NUTS and solvers launch-bound (8% busy). The target for
  steps 1–3 is above 70% busy on A100, by raising batch sizes and
  vmapping over datasets and systems.
* **Host–device transfers.** Per-step H2D/D2H bytes. Maps and peak
  catalogues stay on device between steps 1 and 3; only the top N_keep
  candidates come back.
* **The jit/vmap boundaries.** Confirm that one jitted kernel covers
  "all datasets of a bucket" (step 1) and "all φ × all systems" (step 2),
  and that Python loops remain only over modes (step 4) and systems on
  CPU (step 5).

### 9.2 Reliability metrics and targets

Measured on the adversarial suite (§9.3) and on injection–recovery
draws:

| Metric | Definition | Target |
|---|---|---|
| Mode recall | fraction of systems whose true mode is among the reported modes | ≥ 0.99 (≥ 0.999 in the full suite) |
| Weight calibration | among modes reported with weight w, the fraction that are true is ≈ w (reliability diagram, 10 bins) | within binomial 2σ in every bin |
| Top-mode accuracy | true mode has the largest weight | reported, no target (it is not the goal on ambiguous data) |
| SBC rank uniformity | per-parameter ranks of the truth in the mode-weighted posterior; χ² test and ECDF bands | p > 0.01 for every parameter, after multiplicity correction |
| Flagged failure rate | fraction of systems with ≥ 1 flag | reported per case |
| **Silent failure rate** | true mode missing **and** no flag raised, or truth outside the 99.9% interval of its mode with no flag | **0** (any occurrence fails the suite) |
| Reproducibility | identical mode sets across seeds, CPU vs GPU and latest JAX vs the previous release | 100% mode sets; weights within 0.02 |

### 9.3 The adversarial benchmark suite

Simulated with `simulate` and the `coverage` builders (`vlti_oidata` four
telescopes; `nrm_oidata` 7-hole SAM; `ami_grid_record` AMI kernel
phases), fixed seeds, stored as a manifest of seeds and parameters (not
data):

| Case | What it tests |
|---|---|
| A1 clean | many epochs, s = 1; baseline |
| A2 period alias | epochs sampled near multiples of P/2 |
| A3 (Ω, ω) mirror | no RVs, positions only; must give one mode, Ω mod 180° |
| A4 ω + 180° | V² only (no closure phases); must flag primary/secondary |
| A5 sparse epochs | 3, 4 and 5 epochs; arc-only flag for 3 |
| A6 inflated errors | s = 3–5, with correlated closure-phase errors; the Gl 229 failure |
| A7 near-equal flux | f = 0.8–1.0; the f ↔ 1/f, r ↔ −r flip is exact, removed by f ≤ 1 except near f = 1, where near-twin peaks remain per epoch |
| A8 unresolved epochs | some epochs at separation < λ/(2B) |
| A9 grid edge | companion at or beyond the grid edge in some epochs |
| A10 non-detection | some epochs below the detection threshold |
| A11 extended scene | binary + disc; maps as proposal; must flag map–visibility disagreement |
| A12 short arc | arc ≪ P; prior-dominated P, e; must not invent a sharp P |
| A13 within-night motion | P of days with hours-long nights; snapshot flag |

* **CI (`tests/test_orbit_benchmarks.py`):** one system per case at the
  smallest size (4–6 epochs, ≤ 50 observables, 32² × 8 grids, 10⁴ φ), no
  NUTS (steps 1–4 only), each case under about 10 s. Asserts mode recall
  and flags. Marked `slow` where needed (run on main and weekly).
* **Full suite (OzSTAR):** 200 systems per case, steps 1–5, all metrics
  of §9.2, written to `results.jsonl` with the pinned commit. A summary
  script fails if any target is missed. The results file is the
  regression baseline for the next run.

### 9.4 Batching over systems

* **Shape buckets.** Systems are grouped by (n_epochs, padded observables
  per dataset, grid shape, K_max). Datasets are padded with zero-weight
  observables, and epochs and peaks are masked, so that one compiled
  kernel serves a bucket.
* **vmap over systems** for steps 1–3 on GPU. Steps 4–5 run per system,
  with a vmapped `fit` over modes, and NUTS on CPU array tasks.
* **Per-system jobs** stay the fallback and the comparison. The benchmark
  measures systems/hour for (vmap within one GPU job) vs (one array task
  per system), and the default follows the measurement.

### 9.5 Compile cost and execution robustness

An unattended pipeline fails if compilation fails, however fast the run
is. Example (an orbit + extended-scene fit elsewhere: about 48
parameters, 7 datasets, about 6300 V² and 3200 closure phases,
linear-marginal gains and closure offsets on 3 epochs, JAX 0.11.2 on
CPU): the Jupyter kernel died silently about a minute in, at the first
jit or warm-up compile. It was not out of memory (peak RSS was 102 MB).
The same code ran cleanly as a plain script with one chain. Whether
four chains with `chain_method="parallel"` over forced host devices
trigger it is still open. The pipeline treats this as a robustness item:

* **Unattended work runs as scripts, never notebooks.** Every OzSTAR job
  of §9.6 calls a Python entry point (`python -m ...` or a script), with
  `faulthandler` enabled (`python -X faulthandler`), so that a crash
  leaves a traceback. Notebooks are only for tutorials, executed after
  the fact.
* **Parallel chains are tested for crashes.** The benchmark runs NUTS
  per mode with `chain_method` ∈ {"sequential", "vectorized",
  "parallel"} (the last with `XLA_FLAGS=--xla_force_host_platform_device_count=4`
  on CPU). Each runs in a subprocess, and the exit status and signal are
  recorded. A crash is a failed benchmark, not a skipped one. The
  default `chain_method` follows the result.
* **Compile time and compile memory are tracked metrics**, for every
  step, beside run time:
  * the first-call wall time minus the warm median;
  * `jax.jit(f).lower(...).compile()` timed separately;
  * peak RSS during compilation, sampled by a watcher thread, since
    `getrusage` alone cannot separate it from the run;
  * HLO size: the instruction count of the optimized module, from the
    XLA dump;
  * the number of compilations.

  Each is measured against the number of epochs, the number of gain and
  offset parameters (0, 10, 50, 200) and the model size (point binary;
  binary + disc; binary + 64-ring cone).
* **Design rules that keep compilation bounded** (checked by the HLO size
  growing at most logarithmically with n_epochs):
  * Epochs and datasets of one shape bucket are **stacked and mapped**
    (`vmap`, or `lax.map`/`lax.scan` when memory matters), not unrolled
    by a Python loop over datasets. `Epochs.loglike` and `_loglikes` sum
    over datasets in Python today, so their HLO grows linearly with the
    number of datasets. PR C adds a stacked path for one bucket, and the
    benchmark compares the two.
  * **Static shapes.** Pad datasets to a bucket size with zero-weight
    observables; mask peaks and epochs rather than changing array
    lengths; pass mode and system counts as data, not as Python
    integers that end up in shapes.
  * **Nothing traced becomes static.** No Python floats from traced
    values (the `start_values` rule of V already says this). Grid
    axes, periods and keys are arguments, not closure constants, so
    that a new system reuses the compiled kernel.
  * **Loops over steps go inside the jit.** The EM iterations and the
    peak-refinement fits use `lax.fori_loop`/`while_loop`, as `fit`'s
    solvers already do.
  * **Persistent compilation cache.** Jobs set `JAX_COMPILATION_CACHE_DIR`
    on `/fred`, keyed by the pinned commit, so that array tasks of one
    bucket compile once.

The adversarial suite gains a case for this:

| Case | What it tests |
|---|---|
| A14 compile stress | orbit + disc with per-baseline linear-marginal gains and closure offsets, 3–16 epochs, about 10⁴ observables; NUTS with each `chain_method`; must compile within a stated budget (time and RSS), with HLO size sublinear in n_epochs, and no crash in any chain method |

### 9.6 OzSTAR jobs

All jobs live in `~/code/ozstar_scripts/scripts/<job>/` (copied from
`_template`) and are submitted by Ben with `bin/oz push <job>` and
`bin/oz submit <job> --ref=<full 40-character virgil commit>`. The ref is
taken with `git rev-parse` on the stage's PR head, never shortened and
never from memory. `submit.sh` pins the snapshot (`PIN_COMMIT`,
`PIN_TAG`), results go to `out/$PIN_TAG/`, and `bin/oz pull <job>`
brings them back. Environment: conda env `virgil` (latest JAX), with
`pip install -e ".[dev,orbits]"` of the pinned snapshot. The CPU jobs set
`JAX_PLATFORMS=cpu` (the template does this when no GPU is present). The
GPU jobs set `XLA_PYTHON_CLIENT_PREALLOCATE=false` and record
`JAX_DEFAULT_MATMUL_PRECISION`.

| Job | Runs | Resources | When |
|---|---|---|---|
| `orbit_bench_baseline` | §9.1 benchmarks of the **current** tools (`epoch_positions`, `starting_orbits`, `rank_orbits`, `start_from_positions`, NUTS from `chain_values`) on cases A1, A6, A7; pinned to main `27aa60119b508bdf2cbe1e3ee9127d825b52b997` (or main at submission, by full hash) | array 0–1: task 0 CPU (8 cores, 32 GB, 4 h); task 1 A100 (1 GPU, 8 cores, 64 GB, 2 h) | B0, before any new code |
| `orbit_bench` | §9.1 for every step and variant on the scaling axes; writes `bench.jsonl` (step, variant, platform, compile s, run s, peak host/device MB, systems/h, n_compiles) | array 0–1 as above; 6 h CPU, 3 h GPU | after each of PRs C–F |
| `orbit_prof` | the profiling run of §9.1: `JAX_PROFILE_DIR=$OUT/prof`, `XLA_FLAGS=--xla_dump_to=$OUT/hlo --xla_dump_hlo_as_text` for the search kernel only, `nvidia-smi` telemetry at 1 Hz; summarized with `jax_trace_summary.py` (copied from `drpangloss_prof`) | A100, 8 cores, 64 GB, 1 h; plus a CPU task (8 cores, 32 GB, 1 h) | after PR D, and after any change to steps 1–3 |
| `orbit_reliability` | the full adversarial suite (§9.3): 14 cases × 200 systems, steps 1–5; `SEED=SEED_BASE+SLURM_ARRAY_TASK_ID`, one task per (case, block of 20 systems) | array 0–139, CPU, 8 cores, 16 GB, 4 h per task (NUTS dominant); steps 1–3 optionally on a 1-GPU task per case | after PR F; then weekly on main and before releases |
| `orbit_compile` | §9.5: compile time, compile RSS, HLO size and crash status per step against n_epochs × gain parameters × model size, plus case A14 with each `chain_method`, every configuration in its own subprocess under `python -X faulthandler` | CPU, 8 cores, 32 GB, 2 h; plus an A100 task, 1 h | with B0 against main, then after PRs C and F |
| `orbit_nested_check` | NS (and SMC) on the map surrogate for the A1, A2, A6 and A7 systems of `orbit_reliability`, for independent mode weights | A100, 8 cores, 32 GB, 2 h | with PR G |
| `orbit_gl229` (validation repo) | the Gl 229 rerun: PR A alone, then the full pipeline | CPU, 8 cores, 32 GB, 4 h | after PR A; after PR F |

Requests are first estimates. Each job's README records the measured
elapsed time and `sacct` MaxRSS of its first run, and the requests are
reset to about twice those. No heavy JAX runs on the laptop: local runs
are limited to the CI-sized tests.

## 10. API sketch

```python
from virgil.epochs import Epochs
from virgil.orbit_search import (
    epoch_maps, search_orbits, orbit_modes, fit_modes, sample_modes,
    automatic_orbit,
)

maps = epoch_maps(epochs, grid, *, scales="marginal", dof=None, s_max=None,
                  keep_delta=None, refine=True, batch_size=None)
#  EpochMaps: per dataset the profile map M_e(dra, ddec) (+ flux axis),
#  PeakCatalogue (mu, cov, height, flux, log_det), raw chi2/N and s-hat per
#  block, nu_eff, flags; save()/load().  keep_delta=None -> calibrated (§6).

found = search_orbits(maps, *, periods, eccs=None, n_phase=36, t_ref,
                      prior=OrbitPrior.jeffreys(...), n_starts=4,
                      n_keep=10_000, tie_flux=None, key)
#  OrbitSearch: top phi, psi, map score, marginal score; save()/load().

modes = orbit_modes(found, *, max_delta=20.0, n_max=8)
#  OrbitModes: one representative orbit per mode (positional identity),
#  map evidence per mode.

refined = fit_modes(model, priors, epochs, modes, start_values, *,
                    noise=None, **fit_options)
#  per-mode FitResult, Laplace evidence, map-vs-visibility check.

result = sample_modes(model, priors, epochs, refined, *, noise=None,
                      chains_per_mode=2, num_warmup=1000, num_samples=1000,
                      key)
#  OrbitResult: modes, weights (evidence, stacking), draws per mode,
#  diagnostics, raw chi2/N, scales, flags; summary() -> one table row.

result = automatic_orbit(model, priors, epochs, start_values, *, grid,
                         periods, t_ref, key, stop_after=None)
#  all five steps; stop_after in {"maps", "search", "modes", "fit"}.
```

The module is new (`orbit_search.py`), imported by nothing in the core,
and imports `epochs`, `orbits`, `likelihood`, `fitting`, `detection` (for
the null) and `_grid`. It keeps `epochs.py` small. Point-source binaries
need no `start_values`; a default maps an orbit and flux to an
`OrbitalBinary`.

## 11. Changes to the current orbit-start tools

| Tool | Change |
|---|---|
| `epoch_positions` | PR A: a new field `gap_marginal` (the gap on the scale-marginalized surface) beside the unchanged `gap` for one release, then `gap` switches to the marginal value (decision 6). Covariance and ranking on the scale-marginalized surface; new fields `chi2_raw` (χ²/N per dataset on the quoted errors) and `scale` (ŝ per block). Later it becomes a thin view of `epoch_maps` (the best peak of each catalogue). |
| `EpochPositions.decisive`, `min_gap` | Unchanged API. In PR A they compare `gap_marginal` (CHANGELOG: a behaviour change of `start_from_positions`); `gap` keeps its old value until the switch. |
| `start_from_positions`, `OrbitStart` | Kept through 0.4, with PR A's fixes. Deprecated with a FutureWarning when `automatic_orbit` lands, then removed (decision 5). `OrbitStart.modes` switches from χ² on quoted errors to the positional identity used by `chain_starts`, so modes mean one thing everywhere. |
| `OrbitStart.chain_values` | Superseded by `sample_modes` (chains per mode, not cycling). Kept until removal. |
| `rank_orbits`, `start_from_positions` | Both gain `scales="quoted" \| "marginal"` in PR A, with one staged default for both: "quoted" in 0.4 (omitting `scales` gives a FutureWarning), then "marginal" (decision 5). The `start_from_positions` docstring ("The ranking uses the quoted errors") is updated. |
| `chain_starts` | Unchanged; used inside `orbit_modes`. |
| `starting_orbits` | Unchanged. It is the K = 1, no-floor special case of the TI-EM search and is tested to agree with it. |
| `Epochs.noise` | Docs fixed or fixed values accepted (#268 item 3), in PR B. |
| #268 item 1 (time-dependent refinement) | `fit_modes` takes the full scene, snapshot or per-sample; `start_from_positions` gains `refine_model=` in PR B. |
| #268 item 4 (serialization) | `save`/`load` on every result object (npz for arrays plus JSON for metadata and the pinned virgil version), in PR B for `OrbitStart` and in C–F for the new objects. |

## 12. Staged implementation plan

Each PR is small, has tests and stacks on the previous one where noted.
Benchmarks are staged alongside the code.

* **PR A, the #268 quick win (lands first, off main).** In
  `epoch_positions`:
  * compute per-block raw χ² at every grid point (`whitened_residuals`
    split by block) and the scale-marginalized m = −Σ_b (ν_b/2) ln χ²_b
    with ν_b = `n_independent`, using the Gaussian normalization for
    every block (§1), with the optional `s_max` numerical marginal for
    weak closure phases;
  * add `gap_marginal`, the gap on m, beside the unchanged `gap` for
    one release (decision 6); `decisive` and `min_gap` compare
    `gap_marginal`;
  * **refine on m itself**, not on the quoted errors: fit
    `BinaryModelCartesian` with per-block `vis_scale` and `phi_scale`
    free (their profile has the same maximum as m). With V² and closure
    phases together, the quoted-error maximum of Σ_b χ²_b and the
    maximum of m differ whenever ŝ_V2 ≠ ŝ_CP, and can sit on different
    fringe aliases;
  * take the covariance from the Hessian of −m at that maximum,
    Σ_b [H_b/(2ŝ_b²) − (ν_b/2) g_b g_bᵀ/χ_b⁴], where H_b and g_b are the
    Hessian and gradient of χ²_b. The per-block gradients do not vanish
    individually, so the second term is not a small correction, and the
    result is not the quoted-error covariance × ŝ²;
  * **bound the flux, f ≤ 1**, in the grid and in the refinement prior
    (today `LogUniform(0.1·min, max(1, 2·max))`), which removes the
    exact f ↔ 1/f, r ↔ −r twin (§3.2). This is also one of the candidate
    causes of the Gl 229 disagreement (§1);
  * record `chi2_raw` (χ²/N with N = `n_independent`, per dataset and
    block) and `scale` (ŝ per block);
  * warn when raw χ²/N > 4.

  `rank_orbits` and `start_from_positions` gain `scales="quoted" |
  "marginal"`, with the staged default of §11 ("quoted" in 0.4 with a
  FutureWarning when omitted). With "marginal", `start_from_positions`
  ranks candidates with each dataset's per-block ŝ. The docstring and the
  CHANGELOG record this as a behaviour change of the default to come.

  **Tests** (each on a 3-hole `nrm_oidata` case and a 4-telescope
  `vlti_oidata` case with correlated closure phases):
  1. **Per-block invariance.** Multiplying one dataset's errors by 3, and
     separately multiplying only its closure-phase errors by 3, leaves
     the grid quantities of m (`gap_marginal`, the best grid point and
     its rival) exactly unchanged and the refined positions and `cov`
     unchanged to the fit tolerance. With all blocks scaled, `gap` is
     divided by 9, with the test asserting that the best peak and rival
     grid point are the same in both runs. A pooled per-dataset scale
     or a quoted-error refinement fails the closure-phase-only case.
  2. **The fix, pinned.** A simulated night with two peaks
     0.6 < Δm < 1 and errors quoted 3× too small: assert `gap > 5`
     (decisive on main) and `gap_marginal < 1`, so `decisive(5.0)` is
     False and `start_from_positions` does not seed from that night.
  3. `chi2_raw` ≈ s² (N = `n_independent`) for a simulated s, within
     the tolerance set by E[χ²] = (ν − 3)s² and its scatter.
  4. The ranking is unchanged when one dataset's errors are rescaled
     (by 3) and the others are not; on main (`scales="quoted"`) it
     changes.
  5. **Weak closure phases.** sσ ≈ 0.7 rad: the `s_max` numerical
     marginal is finite, and the peak order from m agrees with the exact
     von Mises profile over s.
  6. **Per-night ranking against an independent fit.** On simulated
     nights with several near-equal peaks, the order of the peaks from
     `epoch_positions` on the scale-marginalized surface matches a
     brute-force reference: a `fit` with fitted `phi_scale` (and
     `vis_scale`) started from every grid peak, then ranked by its
     marginal score. Variants test the other candidate causes of the
     Gl 229 disagreement (§1): a peak whose flux lies near the grid's
     flux bound, a peak near the grid edge, and a night whose companion
     moves by 0.2 λ/B within the night, where the static fit must
     either agree or flag the motion (`spread_days`).

  CHANGELOG entry; the design note V gets a pointer. **Validation:** the
  `orbit_gl229` job, which reruns the Gl 229 fits.
* **PR B, #268 remainder.** `Epochs.noise` fixed values (or docs);
  `refine_model=`; `OrbitStart.save/load`. Tests: a round trip through
  disk gives identical `chain_values`; a fixed scale value works without a
  tied lambda.
* **PR B0, the benchmark harness (parallel to B).** `benchmarks/orbits/`:
  case generator for the adversarial suite (manifests of seeds and
  parameters), the timing/memory/compile-count harness, `jaxprof.py`,
  and the `orbit_bench_baseline` and `orbit_compile` jobs against main. Test: the harness runs
  one tiny case in CI and writes a valid `bench.jsonl` row.
* **PR C, `epoch_maps` and the peak catalogue.** Batched over datasets;
  scale-marginalized; peak finding (local maxima on the profile map
  above the floor, within Δ_keep); vmapped peak `fit`s and Hessians;
  floor interpolant with the peak mask; f ≤ 1; the grouped sign-flip
  bootstrap (`bootstrap_null(groups=...)`) for Δ_keep and ν_eff;
  `save/load`. Tests: every simulated alias within Δ is catalogued; no
  exact f ↔ 1/f twins appear; the peak Gaussians match a brute-force
  fine grid near each peak (KL < 0.05); S_e equals m computed directly
  from the visibilities at the catalogued peaks (point-source binary)
  to 10⁻³ nats, and between peaks to an interpolation tolerance that the
  test measures and records against a fine grid; a non-detected epoch
  gives a flat map and no peaks above the floor; the grouped bootstrap
  recovers a known ν_eff from simulated correlated closure phases
  within 20%.
* **PR D, `search_orbits` (TI-EM over (P, e, T₀)).** Tests: with K = 1 and
  no floor it reproduces `starting_orbits` exactly; EM increases the
  surrogate objective monotonically; a face-on true orbit survives
  pruning (the prior-free score); the reweighted mode evidence reports
  k̂; the true orbit is in
  the top modes for cases A1, A2, A6 and A7 at CI size; a seed change
  gives the same mode set; a compile count of 1 for two systems of one
  bucket. Benchmark: `orbit_bench` and `orbit_prof`.
* **PR E, `orbit_modes` and `fit_modes`.** Positional mode identity;
  Laplace evidence; the map–visibility check. Tests: the (Ω, ω) mirror
  is one mode; ω + 180° is one mode with a flag for V² only and two
  modes with closure phases; an extended-scene case raises the
  disagreement flag.
* **PR F, `sample_modes`, weights and `OrbitResult`.** Chains per mode,
  mode-exit detection, per-mode evidence by IS with k̂, stacking by
  PSIS-LOO over epochs, flags and summary. Tests: a two-mode toy with
  known masses (analytic Gaussians) gets weights within 0.02; a chain
  forced out of its mode is flagged; a 20-step NUTS smoke test. The full
  NUTS recovery runs in `orbit_reliability`.
* **PR G, `automatic_orbit`, the tutorial and deprecations.** End-to-end
  driver; the orbit tutorial adds an "automatic" section (run on OzSTAR,
  figures counted); `start_from_positions` deprecated; AGENTS.md layout
  and import flow; API pages. The `orbit_nested_check` job runs.
* **PR H, the regression suite.** `tests/test_orbit_benchmarks.py` (CI
  sizes, §9.3), the `orbit_reliability` job and its pass/fail summary
  script, and the targets of §9.2 enforced. Linked to virgil-validation
  for SBC v3 (an Issue there with the request, per AGENTS.md).
* **Later:** dense 7-D search for non-linear nuisances (North angle,
  plate scale); tied flux by instrument; optional NS extra if the
  benchmark justifies it; RV-assisted period priors.

Order: A → B (and B0) → C → D → E → F → G → H. A and B are independent of
the rest and of each other.

## 13. Decisions (2026-10-07)

Ben accepted the recommendations of this note, with these decisions:

1. **Thresholds.** Δ_keep and Δ_mode are calibrated by bootstrap (§6),
   with 10 and 20 nats as the defaults when no calibration is run.
2. **Correlated errors.** Reduce the effective degrees of freedom ν_eff
   from the bootstrap now; model correlated systematics explicitly
   (e.g. wavelength-correlated closure-phase offsets) later.
3. **Weights.** Per-mode evidence weights are the default; stacking is
   reported as a misspecification check.
4. **Dependencies.** jaxns and blackjax are allowed as optional extras
   only, never core dependencies.
5. **Defaults.** Ranking with marginalized scales is the default.
   `start_from_positions` is deprecated with a FutureWarning, then
   removed.
6. **PR A.** It adds `gap_marginal` beside the unchanged `gap` for one
   release, then `gap` switches to the marginal value.

## 14. Revision after the peak-catalogue rerun (2026-10-07)

### 14.1 What the rerun showed

Worked example (Gl 229 Ba–Bb, seven GRAVITY nights, closure phases
only, after `epoch_positions` refined and ranked its top peaks, with the
period prior widened to 10.5–13 d):

* One night that had committed to the wrong peak (2023-12-29) now takes
  its refined best, which is the true peak.
* Two others still commit to wrong peaks with decisive marginal gaps
  (9.2 and 7.5): one at the grid edge with flux 0.21, one with flux
  0.95. The flux fixed by the RVs is 0.47.
* At the orbit confirmed by RVs, three of seven nights lie 8–13 mas from
  their best peak, yet fit with raw χ²/N 3.3–8.1. On two of them the
  true position is not among the top five refined peaks at all.
* Not yet measured: on how many nights the truth is a peak (within
  Δ_keep) **at the shared flux** of 0.47. This number decides whether
  any anchor-based generator can find the worked example; a fixed-flux
  per-night job measures it before D2's defaults are set.
* The free search therefore still lands in a wrong mode (i ≈ 128°, Δloss
  ≈ +1100), and with three training nights it finds the 11.0 d period
  alias.

The cause is **per-epoch nuisance freedom**, not only multimodality.
With the flux free per epoch (fitted values 0.15–0.95), each epoch's
peaks move by about 1 mas and new peaks appear that no orbit with one
flux visits. §3.2 profiled the flux per epoch by default; that default
is wrong.

### 14.2 Consequences for the design

1. **Shared nuisances are the default.** A nuisance shared across
   epochs in the model is shared in every score that ranks, clusters or
   weights candidates (R11). Concretely:
   * **Flux.** One flux per band, with a chromatic slope where the band
     is dispersed: f(λ) = f₀(λ/λ₀)^β. The bound f ≤ 1 applies in one
     reference band only, so that the r ↔ −r labelling is the same in
     every band.
   * **Separation-dependent throughput** (fibre field-of-view
     attenuation, bandwidth smearing) is part of the forward model, not
     a per-epoch flux, so that wide orbits do not break the sharing.
   * **Per-epoch calibration gains** (e.g. a V² gain), which are truly
     per epoch and degenerate with f, are **profiled** analytically
     inside χ² (they enter linearly; Luger et al. 2017), not
     marginalized: a marginal gain covariance does not scale with s, so
     χ² would no longer be ∝ 1/s². Each profiled gain costs one degree of
     freedom (ν → ν − 1). `marginal_loglike` and `epoch_positions`
     currently refuse `with_gains` and `with_closure_offsets`
     (`NotImplementedError`); D1 replaces that refusal and tests it.
     Because a V² gain is partly degenerate with f, it can hand
     V²-dominated data back part of the per-epoch flux freedom of §14.1;
     a V²-only variant of A17 checks that it does not recreate decoys.
     Gains and the per-epoch error scale s of the scale marginal m (§1)
     are the only per-epoch nuisances allowed in ranking.
   * **Variability** is a hierarchical per-epoch δf about the shared
     flux, with a Laplace correction; it is off by default.
   * **Cost structure.** The expensive term per candidate and epoch is
     g = exp(−2πi u·r). Flux, the slope and a primary's uniform-disc
     visibility (cached per epoch) are cheap arithmetic on g. At most
     two nuisances are grid-integrated (≤ 32 points each); the rest are
     profiled by Gauss–Newton with a Laplace term, at the polish stage
     only.
   * The per-epoch flux profile is a strictly looser model. It may
     propose candidates; it never ranks them.
2. **Exact scoring replaces the MoG surrogate.** A candidate is scored
   with the scale-marginal likelihood m (§1), computed from the
   visibilities at its predicted positions, with the shared nuisances
   handled as in item 1. This is `_loglikes` generalized and needs no
   interpolation tolerance. A mixture of Gaussians from per-epoch peaks
   puts the truth on the floor whenever the truth is not a per-epoch
   peak (§14.1), so the MoG and TI-EM of §3.2(a) are demoted to a
   diagnostic. Maps survive only as an optional cache, with their flux
   axis, if the benchmark shows exact scoring is too slow. Two
   properties of m are documented: being Student-t-like, it is robust to
   an outlier epoch, and it compresses the gap of a very sharp epoch.
3. **Extra terms enter the score**, each with its own Jeffreys scale
   (RV jitter, an astrometric error scale), so a search with RVs uses
   them from the start. The (Ω, ω) flip is folded only when no term
   breaks it.
4. **Positions are evaluated at sample times, always.** Kepler solves
   per sample cost nothing next to g, so no "within-night motion"
   threshold is needed. Times are stored relative to t_ref in float64
   (an MJD in float32 has a 0.004 d resolution, fatal for periods of
   days).
5. **Candidate generators are plug-ins**, each estimating its cost in
   work units before it runs and stopping at its budget (R12):
   * **(a) Anchor pairs.** Two epochs fix the four Thiele–Innes
     constants exactly at each (P, e, T₀).
     * **Peaks:** every peak within Δ_keep (calibrated, §6) at each
       value of a coarse shared-flux grid, not "the top two" (in §14.1
       the truth was not in the top five on two epochs).
     * **Pairs:** sized by the number w of wrong (or truth-less)
       epochs to tolerate, not by n. One perfect matching of n epochs
       has ⌊n/2⌋ disjoint pairs and tolerates only w ≤ ⌊n/2⌋ − 1; all
       C(n, 2) pairs tolerate w = n − 2. The design is the first k
       rounds of a round-robin schedule (each round a perfect
       matching, n − 1 rounds giving all pairs), with k the smallest
       that guarantees a clean pair for the required w. Default: all
       pairs when C(n, 2) is within budget (n = 7 gives 21 pairs, w ≤ 5),
       since the worked example needs w ≥ 3. Pairs whose regularized
       solve still fails (e.g. identical times) are dropped, which
       breaks the design, so the guaranteed w is recomputed after
       that and recorded; if it falls below the
       required w the search says so.
     * **Conditioning, by regularization.** The exactly determined
       2-epoch Thiele–Innes solve is ill conditioned when the
       eccentric anomalies are ≈ 0 or π apart or Δt ≪ P, and then
       proposes huge a. Instead of hand-set exclusions, the solve is
       ridge regularized: a Gaussian prior on (A, B, F, G) with width
       σ_a(P) ∝ P^(2/3) from Kepler's law (the total-mass prior and
       parallax where known, otherwise the field bound of §14.2.6), the
       astrometric analogue of The Joker's period-scaled prior on K
       (Price-Whelan et al. 2020). This prior shapes only what is
       proposed, never the ranking, so it does not conflict with the
       default Jeffreys priors of the fit. The condition number and the
       ridge's log Occam term are recorded per candidate. The width's
       scale factor is set once on the ill-conditioned pairs of A17 and
       A18 (R13).
     * **Usable epoch:** detected (§7.1), resolved (separation >
       λ/(2B_max)) and with at least one peak within Δ_keep.
     * Anchor χ² never ranks: with three or more anchors it leaves a
       valley of perfect fits.
   * **(b) Anchor-free fallback.** The dense quasi-random search of
     §3.2(b) on the exact score, for systems with fewer than two usable
     epochs, or with non-linear nuisances.
   * **(c) Arcs**, for periods above about 2T (T the time span). Exact
     rejection sampling fails (peaks are about 0.1 mas wide), so arcs use
     OFTI's scale-and-rotate from one anchor epoch's peak Gaussians, then
     the exact score. The draws are deterministic: a fixed Sobol index
     range, recorded in the result (§14.2.10). A period prior spanning
     decades is split: (a) or (b) for P ≲ 2T, (c) above.
   * **Merging generators.** Generators only propose; they never weigh
     their own candidates. All candidates are polished and clustered
     together (item 7), and modes are compared on the refined
     scale-marginal score plus the Laplace log volume under the one
     common prior. Different prior volumes of the generators' P ranges
     therefore do not enter.

   Which generator runs by default, and the hand-off thresholds, are
   measured on the benchmark (§14.4), not assumed.
6. **Steps follow the data and the polisher.** Neighbouring candidates
   must lie within the polisher's **capture radius** ρ of each other in
   predicted position at every epoch. ρ is measured on the benchmark
   (expected about λ/(4B_max)).
   * **Period:** δP ≤ P² ρ / (2π a_max(P) Δt_max · v_e), with Δt_max the
     largest |t − t_ref| (half the span when t_ref is centred) and
     v_e = √((1 + e)/(1 − e)) the periastron speed factor (4.4 at
     e = 0.9).
   * **T₀ and e:** δT₀ ≤ P ρ / (2π a_max v_e) and δe ≤ ρ √(1 − e²) / a_max (at fixed mean anomaly, ∂y/∂e reaches
     a/√(1 − e²) near E = π/2: 2.3× at e = 0.9, 7× at e = 0.99), or
     equivalently uniform steps in a √e-type parameter.
   * **a_max(P) is coupled to P by Kepler's law** from a total-mass prior
     and the parallax where known, a_max ∝ P^(2/3). Without that
     coupling the step count explodes: for a fixed a_max = 100 mas with
     GRAVITY UT baselines, a 5 yr span and P_min = 10 d it reaches about
     10¹¹ (P, T₀, e) points.
   * **Without a distance or mass prior** (decision 4), the search does
     not refuse. a_max defaults to the instrument's field: the fibre
     field of view and the bandwidth-smearing limit for GRAVITY, the
     equivalent for aperture masking (the interferogram's field and the
     smearing limit of its spectral channels), or an explicit maximum
     separation if given. P and a are then independent: Kepler's law
     constrains only the total mass a³/(ϖ³P²), which is reported, with a
     flag when it is implausible for the parallax where known.
   * **The period range** then comes from the period prior alone, which
     must be proper. The default is log-uniform from P_min, twice the
     longest single-epoch sample span (shorter periods are not
     separable from within-epoch motion at the sampling), to P_max = 100
     × the time span T, with P ≳ 2T handled by the arc generator (c).
   * The count is estimated first and recorded, and the search never
     coarsens silently. Over budget, with a mass prior, it refuses with a
     flag. Without one, it runs the anchor-pair generator (a), whose
     Thiele–Innes solve leaves a free, in budget order, and reports the
     fraction of the (P, T₀, e) step grid covered with an "incomplete
     coverage" flag.
7. **Suppress, then polish.** Raw scores at grid points rank candidates
   almost at random, because peaks (0.05–0.2 mas) are narrower than any
   affordable step. So:
   1. non-maximum suppression on predicted positions: one candidate per
      cell of radius ρ in position space at every epoch, the one with
      the best exact score in the cell (so every grid point is scored
      before suppression; ties by index);
   2. polish every surviving cell's candidate by L-BFGS on the score
      (elements and the shared nuisances). The number polished is the
      cell count, not a constant, and it is **not truncated**: since raw
      scores rank cells almost at random, nothing principled could pick
      a subset. The cost (cells × polish steps, §14.5) is estimated
      before scoring; over budget, the search refuses with a flag, as in
      item 6. An ordering proxy (e.g. the score at an inflated error
      scale) may replace the refusal only once validated on A17 and
      A18;
   3. cluster the polished candidates into modes (§4 step 3);
   4. refine on the full model under a work-unit budget, rejecting modes
      at a prior bound with a flag.
8. **Grid domain.** Peak finding covers the prior's largest separation
   (or flags that it does not). A refined peak leaving the grid is kept
   and flagged, never clipped.
9. **Comparing modes.** Modes are compared on the scale-marginal score
   with shared nuisances, in units scaled by ν_eff/ν from the grouped
   bootstrap (§6), because a fixed number of nats means different things
   for 50 closure phases and 10⁴ V². Every reported mode shows raw χ²/N
   per epoch. "Comparable" is a recorded rule: within Δ_mode on that
   score, at most n_sample (default 3) modes, and modes within the
   bootstrap tolerance of the boundary are flagged "borderline".
   **Sampling runs only from the best mode, or the comparable few.**
   This keeps §4 step 3's overflow rule: up to n_modes_max = 8 modes are
   refined and reported with Laplace weights, and if the Laplace weight
   of comparable modes left unsampled exceeds 0.01 the result is flagged
   "mode overflow", so §9.2's weight calibration stays testable.
10. **Reproducibility.**
    * Enumeration everywhere, with deterministic covering designs; no
      random draws in the search.
    * Budgets in deterministic work units (candidates, L-BFGS steps,
      likelihood evaluations), with the predicted cost checked up front.
      Wall-clock time is only a kill, which marks the run failed
      (decision 3).
    * Ties are broken on quantized scores, then by index, never by
      floating-point accident (a CI test was lost to two mirror grid
      cells tying to rounding error).
    * Every setting, step rule, budget, platform, precision and version
      is recorded in the result.
    * "Identical across seeds" (§7.1, §9.2) becomes a contract across
      CPU and GPU, float32 and float64, and changes of step size and
      search domain: the same top mode, unless its gap is below the
      score quantum or the bootstrap tolerance, in which case the result
      is flagged "borderline" (a score near a quantum boundary can round
      either way).

### 14.3 Requirements added

Requirements are numbered R11–R13 to avoid the benchmark cases A11–A13
of §9.3.

* **R11 No per-epoch nuisances in ranking**, except the per-epoch error
  scale s of the scale marginal m and per-epoch calibration gains
  profiled analytically (§14.2.1).
* **R12 Work-unit budgets.** Every stage estimates its cost and runs
  under a budget in deterministic work units; it reports `stop` and
  `at_bound` (as `fit` does). Wall-clock limits are only a kill, which marks the run
  failed.
* **R13 Generality.** No default may be tuned on one system. Every
  default (ρ, Δ_keep, Δ_mode, budgets, hand-off thresholds) is set on the
  benchmark suite of §14.4, across instruments and bands, before the
  sealed set is run.

### 14.4 Benchmark suite: additions

The cases of §9.3 are simulated with white noise, under which the truth
is always a per-epoch peak. They cannot reproduce the failure of §14.1,
and simulating with virgil's own `simulate` alone is an inverse crime.

**New cases:**

| Case | What it tests |
|---|---|
| A15 hang start | a retrograde start that crawls to the bounds; must stop within its budget, flag `at_bound`, and rank below the true mode |
| A16 realistic noise | residuals block-bootstrapped (by exposure and triangle) from real data, one residual source per instrument family, plus injected per-epoch closure-phase offsets |
| A17 per-epoch flux scatter | best per-epoch fluxes scattered over 0.15–0.95 by the noise; the per-epoch catalogue misses the truth on some epochs. Variant A17-V²: V²-dominated data with a profiled per-epoch gain, which must not recreate the decoys |
| A18 wrong decisive epoch | one epoch with a decoy at a decisive gap; the covering design must still recover the truth |
| A19 null | a single star; no confident mode |
| A20 wide orbit | separations beyond the default grid, with field-of-view attenuation and smearing |
| A21 mixed instruments | two bands or instruments, different fluxes, scales and a chromatic slope; a companion brighter in one band; a plate-scale or wavelength miscalibration between them |
| A22 extreme geometry | e > 0.9; exactly face-on and edge-on |
| A23 cadence aliases | P near 1 d and 1 yr |
| A24 triples | a resolved third body; an unresolved photocentre wobble |
| A25 disc + orbit | a circumbinary disc with a binary orbit |
| A26 single epoch | one epoch (e.g. AMI): position recovery only (decision 6); with RVs or Gaia astrometry later |

**Statistics.** Zero failures in 200 trials bounds the failure rate only
at 1.5% (95%, the rule of three). The recall target of 0.999 (§9.2)
needs about 3000 systems per case; the full suite's size is set from
these bounds and fixed, with the sealed manifest, before any default is
chosen (decision 5).

**Against gaming.**
* Defaults are frozen on a development manifest. A **sealed** manifest,
  generated after the freeze, is then run once.
* Part of the simulation uses an independent forward model from
  virgil-validation.

**Real data with known orbits** run blind, in virgil-validation (its
independence rule): the Gl 229 worked example with and without RVs; the
SPIE imaging-contest data; PIONIER binaries with published orbits.
Single-epoch calibrator binaries (NACO SAM, with published positions or
a non-detection) test per-epoch recovery and detection limits only. The
proposed multi-epoch set is GRAVITY SB2 binaries with published relative
positions at many epochs.

**Invariance metrics:** the top mode must not change with the step
sizes, the search domain, the platform or the precision.

### 14.5 Cost, re-estimated

Exact scoring costs, per candidate, (visibilities) × (epochs)
evaluations of g, the complex exponential, plus (visibilities) ×
(epochs) × (grid nuisance points) multiply-adds for the flux grid, which
is cheap arithmetic on g (§14.2.1). With 10⁴ visibilities per epoch, 10
epochs and 16 flux points, that is 10⁵ g evaluations and 1.6 × 10⁶
multiply-adds per candidate; the figures below count the latter, as the
larger.
* **Scoring** happens before suppression (§14.2.7), on every grid point.
  After Kepler coupling, 10⁶ grid candidates cost about 10¹²
  evaluations: minutes on an A100, hours on a CPU node.
* **Polishing** costs about 50 L-BFGS steps per cell, each with a
  gradient (a few score evaluations), so roughly 100–200 candidate
  scores per cell. 10⁴ surviving cells then cost as much as scoring the
  whole grid, and 10⁵ cells about 10× more. Polishing therefore
  dominates unless suppression leaves ≲ 10⁴ cells; the up-front estimate
  (§14.2.7) counts both.
* 10⁹ candidates (wide, long-period searches without a mass prior) are
  out of reach, hence the refusal of §14.2.6.

§3.2's "about 10⁶ φ" and §7.3's search costs are superseded by these
figures, to be replaced by measurements (§9.1).

### 14.6 Superseded sections and open decisions

**Superseded by §14:**

| Section | What changes |
|---|---|
| §0 summary, §4 steps 1–3 | maps and TI-EM replaced by exact scoring, plug-in generators, suppression and polishing; §4 step 3 keeps n_modes_max = 8 and the 0.01 "mode overflow" rule, but samples at most n_sample comparable modes (§14.2.9) |
| §3.2 | flux default (per-epoch profile → shared); TI-EM demoted to a diagnostic; the period-sampling paragraph replaced by §14.2.6 |
| §7.1 | the grid step λ/(3B_max) replaced by the capture radius; "two Sobol scrambles" replaced by the invariance checks of §14.2.10 |
| §7.3 | search costs replaced by §14.5 |
| §9.2, §9.3 | "across seeds" replaced (§14.2.10); sample sizes per §14.4 |
| §9.6 | the PR D jobs follow §14.7 |
| §10 | `epoch_maps` optional; `tie_flux=None` replaced by shared nuisances by default |
| §11 | `starting_orbits` is no longer "the K = 1 case of TI-EM"; it is the inner solve of the anchor-pair generator |
| §12 PR C/D tests, "Later: tied flux" | replaced by §14.7 |
| §2 A10 | requirements on maps hold only where the optional map cache is used |
| §3.3, §5.3 item 3 | nested sampling on the map surrogate is no longer the independent weight check; the check is NS (or Laplace) on the exact score |
| §4 step 4 | the map–visibility check applies only with the map cache; otherwise scoring is already on the visibilities |
| §5.1 | per-epoch flips are not searched separately: positions come from whole orbits, and the (Ω, ω) flip is folded per §14.2.3 |
| §6, first bullet | maps are not kept independent per epoch by default; the shared nuisances couple the epochs |
| §7.1 Snapshot and Refinement rows; §7.2 "within-night motion", "map too coarse" | positions are always evaluated at sample times (§14.2.4), so the snapshot flag and the map-resolution warning apply only to the map cache |
| §9.1 "Candidate scoring", "Search" | timed on the exact scorer and polishing (§14.5) |
| §9.3 A11, A13 | A11 tests the map cache only when it is used; A13 tests sample-time evaluation, not the snapshot flag |

**Decisions** (Ben, 2026-10-07):

1. **Shared flux** (and chromatic slope): integrated out wherever
   possible (marginal), **and** the fitted (profiled) value with its
   uncertainty is reported as an output.
2. **Bootstrap calibration is mandatory for unattended runs.** A run
   counts as unattended only if Δ_keep, Δ_mode and ν_eff were
   calibrated by the grouped bootstrap, cached per dataset (the
   correlations differ by night). Without it, the result is flagged
   "needs review".
3. **Work-unit budgets.** Budgets count likelihood evaluations or
   optimizer steps, with the predicted cost checked before the run.
   Wall-clock time is only a kill, which marks the run failed.
4. **No distance or mass prior required.** With neither and no explicit
   maximum, the separation bound defaults to the instrument's field and
   the search does not refuse (§14.2.6, which also says how the period
   range is bounded).
5. **Preregistration: yes.** Sample sizes per case and the sealed
   held-out manifest are fixed before any default is chosen.
6. **Single epochs are "position only"** in this design. Joint
   single-epoch fits with RVs or Gaia astrometry come later; A26 tests
   position recovery only until then.

### 14.7 Staging, revised

PR letters below are this note's (§12), not GitHub numbers. Already
done: PR A (`gap_marginal`), a budget and `period_grid` for `fit`, and
the refined peak catalogue (`EpochPeaks`).

* **PR D1, the scorer.** `score_orbits(epochs, model, orbits, *,
  shared=..., terms=(), scales="marginal", batch_size)` in
  `orbit_search.py`, generalizing `_loglikes`:
  * shared flux per band with an optional chromatic slope, on a grid
    (marginalized, with the profiled value and its uncertainty also
    reported, per decision 1);
  * analytically profiled per-epoch gains;
  * extra terms with their own scales;
  * positions at sample times, times relative to t_ref in float64;
  * quantized, index-broken ties.

  Tests:
  * equals the sum of `marginal_loglike` at a fixed flux, on data with
    one time per epoch;
  * on data with several sample times per epoch, equals the sum of
    `marginal_loglike` over the same data split into one dataset per
    time (with the error scale tied across the split), so that
    within-night motion is exercised;
  * the flux marginal matches quadrature;
  * an RV term adds its scale-marginal `loglike`;
  * independent of `batch_size`;
  * agrees between float32 and float64 and is tie-deterministic;
  * a per-epoch-flux decoy loses to the truth under shared flux.
* **PR H (brought forward), the suite.** A15–A26 at CI size, the harness
  writing each row as it finishes, and the development and sealed
  manifests; the full runs on OzSTAR. It lands beside D1, so that D2's
  defaults are chosen on it. Cases that need D2's generators (A15, A18,
  and any others that fail at CI size with D1 alone) are marked
  `xfail(strict=True)` until D2, so the suite does not land red.
* **PR D2, generators and modes.** Peaks at a coarse shared-flux grid;
  the anchor-pair covering design with conditioning; the step rules and
  cost estimate; suppression and polishing; clustering; refinement
  under a budget; the mode report and the sampling rule. Then the
  fallback (b) and the arcs (c). `start_from_positions` becomes a thin
  deprecated wrapper.
* PRs E–G of §12 follow, except that NUTS starts only from the best mode
  or the comparable few.

## References

* Luger, Foreman-Mackey & Hogg (2017), linear marginalization.
* Price-Whelan et al. (2017), The Joker: rejection sampling of
  Keplerian orbits.
* Blunt et al. (2017), OFTI (orbits for the impatient).
* Price-Whelan, A. M., et al. (2017, 2020), The Joker: rejection sampling for
  Keplerian orbits with linear parameters marginalized.
* Lucy (2014), dense grids over (P, e, T₀) with linear Thiele–Innes
  constants.
* Yao, Vehtari & Gelman (2022), stacking for non-mixing Bayesian
  computations.
* Vehtari, Gelman & Gabry (2017), PSIS-LOO; Vehtari et al. (2024),
  Pareto-smoothed importance sampling.
* Talts et al. (2018), simulation-based calibration.
* Skilling (2006), nested sampling.
