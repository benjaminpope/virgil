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

The profile over s gives the same function up to a constant. Two
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
  of m, so already scaled by ŝ²), its height h_ek, its flux and its
  log-determinant.
* **A floor.** The coarse grid M_e itself, interpolated
  (bilinear/bicubic), which covers positions far from all catalogued
  peaks. This matters for epochs with no detection, whose map is
  nearly flat. These epochs then contribute (correctly) almost nothing,
  instead of being forced onto a noise peak.

The epoch score is

  S_e(r) = logsumexp_k [h_ek + ln N(r; μ_ek, Σ_ek) + c_ek] ⊕ M_e(r),

a mixture of Gaussians (MoG) with a floor, where ⊕ is a smooth maximum
and c_ek normalizes each component to its peak height. Near a peak S_e
is the Laplace approximation; far away it is the grid.

**Flux.** By default the flux is profiled per epoch, which is robust to
different filters. As an option, flux is tied per instrument/filter:
keep the flux axis of the map and evaluate Σ_e m_e(r_e, f) on the shared
flux axis, then take the maximum (or log-sum-exp under the log-uniform
prior) over f. This costs a factor n_flux and helps near-equal-flux
pairs, where a per-epoch flux flip (f ↔ 1/f with r ↔ −r) is otherwise
free.

**Search option (a): (P, e, T₀) with linear Thiele–Innes constants.**
At fixed φ = (P, e, T₀), r_e = K_e(φ)ψ is linear in ψ = (A, B, F, G).
`starting_orbits` exploits this with one Gaussian per epoch. With a MoG
per epoch, the profile over ψ is no longer one least-squares solve, but
EM makes it a few:

* **E-step:** responsibilities γ_ek ∝ exp(h_ek) N(K_eψ; μ_ek, Σ_ek), with
  the floor as an extra broad component (weight from M_e at K_eψ, which
  keeps outlier epochs from dominating).
* **M-step:** a weighted linear solve for ψ, with the TI Gaussian
  proposal prior Λ (width s₀(P/P₀)^(2/3), TI D6):
  ψ = (Σ_ek γ_ek K_eᵀΣ_ek⁻¹K_e + Λ⁻¹)⁻¹ Σ_ek γ_ek K_eᵀΣ_ek⁻¹μ_ek.
* **Starts per φ:** a few. Use the best component of every epoch, and
  the best component of the decisive epochs with the ambiguous ones
  left free (γ uniform), plus R random assignments. EM converges in
  3–10 iterations, because peaks are separated by many σ.
* **Rescore** each converged ψ on the exact S_e (not the EM bound), and
  keep the marginal over ψ by the Laplace/TI formula at the EM optimum.
  The result is a 3-D map of the marginal score over φ, as in TI §4
  ("exhaustive 3-D maps").

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
  the map score) of the best goes on, up to n_modes_max (default 8;
  exceeding it is a flag, not a truncation).
* **Step 4** fits the full scene (extended components, hierarchical
  scales, North angles) from each mode's best orbit. A fit that leaves
  its mode (its positions move by more than ½ λ/B_max) is recorded as
  "mode merged" and deduplicated. The map score and the visibility
  log likelihood at the refined point are compared: for point-source
  binaries they agree to within interpolation tolerance, and a larger
  difference flags model misspecification or a map that is too coarse.
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
* **Per-epoch flips for near-equal fluxes.** f ↔ 1/f with r ↔ −r on a
  single epoch. The catalogue keeps both peaks, and the orbit decides.
  With flux tied across epochs this ambiguity is mostly broken.
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
  `n_independent`. It is calibrated, not assumed: the residual bootstrap
  of `detection.bootstrap_null` (sign-flip, closure phases whitened)
  applied at the best peak gives the null distribution of the gap
  between the true peak and its best alias. Δ_keep and the "decisive"
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
| A7 near-equal flux | f = 0.8–1.0; per-epoch 180° flips |
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

maps = epoch_maps(epochs, grid, *, scales="marginal", dof=None,
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
| `rank_orbits` | Gains `scales="quoted" \| "marginal"`, with "marginal" the default (decision 5). Until the default changes, omitting `scales` gives a FutureWarning. |
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
    with ν_b = `n_independent`;
  * add `gap_marginal`, the gap on m, beside the unchanged `gap` for
    one release (decision 6); `decisive` and `min_gap` compare
    `gap_marginal`;
  * take the covariance from the Hessian of −m at the refined position,
    which is the quoted-error covariance × ŝ² to first order;
  * record `chi2_raw` and `scale`;
  * warn when raw χ²/N > 4.

  In `start_from_positions`, rank candidates with each dataset's ŝ
  (passed as `noise` values) instead of the quoted errors.

  **Tests:**
  1. Multiplying one dataset's errors by 3 leaves `gap_marginal`,
     positions and `cov` unchanged (rtol 1e-6) and divides `gap` by 9.
  2. On a simulated night with two peaks Δm < 1 and errors quoted 3× too
     small, `decisive(5.0)` is False (it is True on main), and
     `start_from_positions` does not seed from that night.
  3. `chi2_raw` ≈ s² for a simulated s.
  4. The ranking is unchanged under a common error rescaling.
  5. **Per-night ranking against an independent fit.** On simulated
     nights with several near-equal peaks, the order of the peaks from
     `epoch_positions` on the scale-marginalized surface matches a
     brute-force reference: a `fit` with a fitted `phi_scale` (and
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
  floor interpolant; `save/load`. Tests: every simulated alias within Δ
  is catalogued; the peak Gaussians match a brute-force fine grid near
  each peak (KL < 0.05); S_e equals the exact snapshot likelihood at the
  catalogued peaks (point-source binary) to 10⁻³ nats; a non-detected
  epoch gives a flat map and no peaks above the floor.
* **PR D, `search_orbits` (TI-EM over (P, e, T₀)).** Tests: with K = 1 and
  no floor it reproduces `starting_orbits` exactly; the true orbit is in
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

## References

* Luger, Foreman-Mackey & Hogg (2017), linear marginalization.
* Price-Whelan et al. (2017), The Joker: rejection sampling of
  Keplerian orbits.
* Blunt et al. (2017), OFTI (orbits for the impatient).
* Lucy (2014), dense grids over (P, e, T₀) with linear Thiele–Innes
  constants.
* Yao, Vehtari & Gelman (2022), stacking for non-mixing Bayesian
  computations.
* Vehtari, Gelman & Gabry (2017), PSIS-LOO; Vehtari et al. (2024),
  Pareto-smoothed importance sampling.
* Talts et al. (2018), simulation-based calibration.
* Skilling (2006), nested sampling.
