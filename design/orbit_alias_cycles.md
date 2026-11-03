# Alias-resolved orbit fits: model, coordinates, cells, search and evidence (Stage D)

Status: **design for Ben's review**, 2026-11-01 (Opus-reviewed plan; Sonnet draft of this note). Nothing here is
implemented. §6–7, §9–10, §13–14 were rewritten on 2026-11-02 (Opus) after the leg-1 review, on Ben's decisions of 2026-11-01. It is the note every later stage codes against (2a coordinates, 2b night summaries, 3b′
search, evidence and position leg, 3c visibility leg and noise contract, 3d no-orbit verdict, 4 validation).

Companions: [orbit_mixture_v2.md](orbit_mixture_v2.md) (approved 2026-10-29: the noise model),
[orbit_dehack.md](orbit_dehack.md) (decisions 4, 5, 10, 12; §2I flux floor), [linear_marginalisation.md](linear_marginalisation.md),
[thiele_innes_marginalisation.md](thiele_innes_marginalisation.md), [orbit_gate_042.md](orbit_gate_042.md) (§3.3, the G1 gate).
Checks: [`scripts/check_orbit_coordinates.py`](../scripts/check_orbit_coordinates.py) (every Jacobian below) and
[`scripts/toy_tiprime_legs.py`](../scripts/toy_tiprime_legs.py) (the superseded tempered comparison), [`scripts/leg1_review/`](../scripts/leg1_review/) (the review that retired it) and [`scripts/leg1_toys/`](../scripts/leg1_toys/) (§6.4–6.7).

Design notes stay general: gallery targets appear only as worked examples. No constant below was chosen by looking at
vv or gallery results (§10).

## 0. Summary

1. **Model (§1).** Mixture v2 with global inlier scales; isotropic orientation, log-uniform *a* and *P*, uniform *e*;
   period domain [P_cut, P_max] with P_cut = max(2Δt_min, p_floor) as a **declared prior** (P_night dropped, 2026-11-02).
2. **Outlier contract (§2).** b_k is the prior-predictive density of night *k* under the *same* per-exposure
   likelihood as the inlier term, with the companion's night state (p, v, f) drawn from π_out instead of from the
   orbit. #618's b_k is the uniform-disc, σ_v → ∞ special case. Measured on #618's fixture: at the true orbit b_k − ℓ_k is
   −27 to −58 nats, so **no choice of π(v) or scales restores #618's six failing tests**: they fail because the optimizer
   returns partial-orbit optima at least 54 nats below the truth (data precision pinned; a float32 rerun passed). The contract is still fixed (it is a modelling error), and the
   tests are the acceptance test of the search (§2.5, §6).
3. **Coordinates (§3).** Anchor Thiele–Innes TI′; z ∈ ℝ⁴ with u = z_u², V = z_v² (prior ∝ |z|⁻⁴, 4:1 cover); ζ with
   k + ih = ζ² (uniform prior, 2:1 cover). Both are *derived* departures; all Jacobians checked to 6 × 10⁻¹⁴.
4. **Cells (§4).** T/P = N + ρ, π(N, ρ) ∝ 1/(N + ρ); exact partition for any edges; edges and anchors affect efficiency
   and labels only.
5. **Search, evidence, sampling (§5–§7; rewritten 2026-11-02).** There is no tempering: it fails at the first-order
   transition onto the all-outlier floor (−47 to −156 nats on the review toys).
   - **Search.** Per cell, a deterministic search in the low-dimensional core (ρ, ζ): minimal sets of nights with TI′
     solved by variable projection, then concentration steps, then Newton on the label-summed target. Its coverage is
     derived for ρ, tested for ζ, and gated, with a `cell-at-floor` sentinel.
   - **Evidence.** Z_N comes from defensive importance sampling from the search-built t₅ mixture. It is unbiased on every
     toy at 4000 evaluations, with var × cost 500–2000× below bridge sampling. Laplace is a baseline. Nested sampling
     fails unseeded as tempering does.
   - **Sampling.** Posterior draws come by resample-move with HMC, in cells that carry mass.
   - **Across N.** Cells run per-N; shared draws across N were measured and rejected (§6.7). Modes, not draws, are
     shared: neighbour-cell modes that straddle an edge and short-span band modes (§6.11).
   - **Design pass 2026-11-03 (§6.9–6.11).** (A) The search profiles the inlier scale jointly with the labels, so it is
     scale-free. At s = 1 it found no mode at large s, which also explains the gallery's silent floor-everywhere fit.
     (B) m-inlier modes are real mass. One-season ones are ridges in ρ, proposed by an INLA-style conditional
     component, and negligibility uses the bound π(cell)·sup L. (C) Edge candidates stop at |w_P| = W, short-span sets
     are searched once, and triplets are kept only where a pair fails a velocity-error coverage criterion.
   - **Testing.** A population test (§14) replaces Stage 3b's slow tests.
6. **Verdict (§8), cost (§9), constants (§10), literature (§11), draft G1 criteria change (§12, awaiting Ben).**

## 1. The model

### 1.1 Data, likelihood, noise

Nights k = 1…K, each a set of exposures j with the per-exposure likelihood of `epochs._Surface` (V², closure phases,
extras; gains and closure offsets marginalized analytically, as now). Calibration groups g (observable kind ×
instrument) each carry one global inlier scale s_g with ln s_g uniform on [−ln s_max, ln s_max] and s_max = 10
(orbit_mixture_v2 §2.2), appended to the sampled vector. Night *k* contributes

  (1 − ε) e^{ℓ_k(x, s)} + ε e^{b_k},  ε ~ Beta(½, ½) integrated (Gauss–Chebyshev, exact),

with ℓ_k the tempered inlier log likelihood at orbit x and b_k the outlier component of §2. Tempering factors f_b
(`virgil.tempering`) multiply each block's log likelihood in ℓ_k **and** b_k.

Mixture v2 has two outlier candidates, chosen on stratum D by the pre-registered rule (v2 §5.3, gate §3.6, dehack
decision 2), and both are carried through this note:

- **(ii), a companion anywhere:** the outlier term is the constant e^{b_k} of §2 (own scales integrated inside b_k).
- **(iii′), inflation relative to the global scale** (v2 §5.2): the outlier term is
  e^{o_k(x, s)}, o_k = ln ∫₁^{s_max} Π_g L_kg(x; s_g t) dt/(t ln s_max), the night's *own* likelihood at the orbit with
  every group's scale inflated by one shared t (the group product is inside the log, so t_k is a single per-night label). It depends on (x, s), so there is **no constant floor**: an outlier night still pulls
  x (weight ≈ 1/s_max²), and the position-level analogue is õ_k(x, s) of §5. Wherever this note says "the floor
  ε e^{b_k}", read ε e^{o_k(x, s)} under (iii′); §5 and §6 give both forms. The search's C-step labels (§6.1) carry
  t_k as one more per-night label (conditionally Gaussian in TI′ with the inflated variance t_k² σ²).

### 1.2 Prior

| quantity | prior | status |
|---|---|---|
| orientation | isotropic: cos i ~ U(−1, 1), (ω, Ω) uniform | by definition |
| a | log-uniform on [a_lo, a_hi] | declared (§10) |
| P | log-uniform on [P_cut, P_max] | declared (§1.3) |
| e | uniform on [0, e_max] | declared |
| phase | M_a (anchor mean anomaly) uniform on [0, 2π) | by definition |
| flux ratio f | log-uniform on (lo, 1], lo from the data's 2σ contrast curve (orbit_dehack §2I, decision 5), with the brighter/fainter **label** λ ∈ {0, 1}, prior ½ each | decision 5 |
| θ (primary diameter), disc label, vis_amp, flux floor | as in `OrbitPipeline` today | unchanged |
| ln s_g | uniform on ±ln s_max | v2 |

Isotropy and log-uniform *a* together are exactly "uniform direction on S³ × log-uniform |z|" in the coordinates of §3.2.

### 1.3 The period domain is a declared prior

P ∈ [P_cut, P_max], **P_cut = max(2·Δt_min, p_floor)** (Ben, 2026-11-02; replaces max(2Δt_min, P_night)):

- **Δt_min** is the shortest gap between consecutive nights, after merging exposures less than `gap_days` = 0.5 d apart
  into one night.
- **p_floor** (0.2 d, inherited) is the only other lower bound. The orbit-cells path drops P_night, the within-night
  motion bound of `period_window`. P_night belongs to the snapshot-era likelihood; the per-exposure target of §2 models
  the motion within a night, so it needs no within-night bound. The only bound left is within one integration, about
  0.015 d, which is below p_floor.
- **The alias argument.** Above P_cut the closest pair of nights advances by less than half a cycle, so the
  regular-sampling alias (prograde, retrograde, ν) ≡ (retrograde, prograde, 1/Δt − ν) lies outside the domain. The
  factor 2 holds exactly at every e and inclination, given that Δt_min is taken over merged nights.
- **Not a Nyquist limit.** For irregular sampling it is not one (VanderPlas 2018): below Δt_min, irregular gaps and
  within-night velocities can still identify P. It is a prior choice, reported in every summary and overridden by
  `p_range`. There is no `period-floor` flag and no κ (orbit_dehack decision 4: no nats anywhere).
- **P_max** follows orbit_dehack decision 13 (`span_factor` deleted). Periods beyond the span form one long-period band
  [span, P_max], with P_max = 2π a_hi · span/σ_min, where σ_min is the smallest per-night position error (the motion
  over the span falls below it). It is a data-derived, reported domain limit. The gate draws truths up to 3 × span, far
  inside it.
- **Reporting.** `PeriodDomain` reports which term binds P_cut (2Δt_min or p_floor), the inputs to P_max (a_hi, span,
  σ_min) and the prior mass of the long-period band N = 0. With P_max at a_hi and σ_min, that mass is 63–91 % and
  depends on the data, so it is shown, not hidden.
- **Curvature inside a night.** This is no longer covered by a bound. It is modelled in two places:
  - the per-exposure target (§2), whose inlier term moves the companion through the night;
  - the night summaries of the cheap leg (§5), which predict each night's (p̂, v̂) as the exact least-squares
    projection of the orbit at the stored frame offsets, not as the instantaneous state.

  The instantaneous-state prediction carried a curvature bias of ½·a(2π/P)²·T_ss/n, up to about 15σ for eccentric
  short-period orbits. Because the bias depends on the parameters, it is modelled rather than cut.
- **Why it changed.** Argon's simulated ensemble on #629 (#632) found 52 % of true periods outside the old domain.
  `period_domain` passed field = A_RANGE[1] = 30 mas to `aliases.period_cut`. With L = 0.1 d and res = 3 mas, that gave
  P_night = 2π·30·0.1/(0.5·3) = 12.566 d, which bound in 50 of 50 draws, while 2·Δt_min was ≤ 2.6 d (median 0.25 d).
  The old note's random-epoch study (`scripts/p_cut_random_epochs.py`) had already found P_night binding in 81 % of
  patterns.
- **Cost.** Two seasons a year apart give about 250 cells (§9).
- Consequence for the gate: §12.

### 1.4 Linear parameters (Luger, Foreman-Mackey & Hogg 2017)

| parameter | linear in | use |
|---|---|---|
| gains, closure offsets | observables (linearized) | already marginalized analytically (Stage 1 makes it cheap) |
| ln s_g | no (scale) | sampled; one Woodbury solve per night at the current s_g |
| outlier own scale (per group, per night) | no | integrated inside b_k only (closed-form Gamma integral, or 1-D quadrature with gains) |
| TI′ = (A′, B′, F′, G′) | night positions *and* velocities | exact Gaussian conditional given the nonlinear core and the night labels; used by the search's variable-projection solve (§6.1) and sampled by HMC (§6.3), **not** used as a closed-form marginal: the isotropic prior is not Gaussian in TI′ and peaks and outliers make the marginal a sum over ∏(J_k + 1) assignments |
| RV γ per instrument, RV amplitude | RVs | marginalize when RVs enter; C and H are functions of TI′ so TI′ stops being linear with RV (thiele_innes_marginalisation §3) |
| flux ratio f | closure phases only for f ≪ 1 | proposal only |
| P, e, M_a, θ, vis_amp | no | sampled |
| outlier labels, peak labels, disc label, brighter/fainter label | discrete | summed exactly per night in the target; Gibbs where carried as state |

## 2. The outlier contract

### 2.1 Definition

The inlier hypothesis for night *k* says the companion's night state **s_k = (p, v, f)** (sky position at the night's
reference time, sky velocity, flux ratio) is the orbit's: s_k(x). The outlier hypothesis says s_k is drawn from π_out
instead, the data are scored by the *same* function, and the night has its own scale:

  ℓ_k(x, s) = L_k(s_k(x); s_g),  b_k = ln ∫ π_out(p, v, f) · [ ∫ L_k(p, v, f; s_k) π(s_k) ds_k ] dp dv df,

where L_k(p, v, f; ·) is the per-exposure likelihood of a companion moving linearly through the night, p(t_j) = p + v
(t_j − t̄_k), exactly the within-night motion the inlier term has (`arcs.night_motion`). The orbit's curvature inside a
night is in the inlier term through the per-exposure positions. The cheap leg's night summaries model it through the
exact least-squares projection (§1.3, §5). The own scale s_k has one
log-uniform entry per calibration group on [1/s_max, s_max] (orbit_mixture_v2 §2.4), so b_k is constant in (x, s) and
a night with q → 1 has zero pull on x. This is the contract; the rest of the section fixes π_out and checks it.

### 2.2 π_out and the scales

π_out(p, v, f) = π(p, v) π(f): the **joint prior predictive** of the night state under §1.2 at a uniformly random epoch,

  π(p, v) = ∫ δ(p − p(x)) δ(v − v(x)) π(x) dx,

with π(f) the prior of §1.2 (log-uniform on (lo, 1]). (p, v) are *not* independent (the periastron couples them) and |p|
is *not* uniform on a disc: under the orbit prior |p| concentrates at small separations, about 3 nats above 1/(π r_max²)
at 5 mas. The contract has no constants of its own: the same data model on both sides, the same prior for the state.

- **Tabulation, once per domain.** A scrambled-Sobol sample of 2²⁰ prior draws (§1.2: a, P, e, orientation, phase) is
  mapped to (ln|p|, ln|v|, φ, the angle between p and v) and smoothed by a Gaussian kernel density with Scott's bandwidth,
  a C^∞ density that the Genz–Malik cubature of b_k can integrate. **Tolerance:** b_k changes by less than 0.05 nats
  between 2²⁰ and 2²¹ draws (§10). A histogram is not used: it has a bin width and is not smooth.
- To first order ln|v| = ln a − ln P + ln 2π, a trapezoid (convolution of two uniforms) of width ln(a_hi/a_lo) +
  ln(P_max/P_cut), smeared by the eccentricity factor; a check of the tabulation, not its definition.
- v = 0 (#618) is **not** a prior: it is the limit of infinitely well-measured stationarity and gives the outlier a
  likelihood the inlier does not have to pay for. #618's b_k is recovered by replacing π_out(p) with the uniform disc
  1/(π r_max²), r_max = a_hi(1 + e_max), and taking σ_v → ∞; that is the Stage 3c unit-test oracle (§2.6). The joint prior
  predictive differs from it by the shape of π(p) and the v-Occam.
- Scales: as v2 (outlier own scales for (ii), inlier global). Their Occam cost is Σ_g ln(2 ln s_max √(ν_g/π)), 2.0–2.4
  nats per group at ν = 9–18 and s_max = 10, and moves b_k by −J ln 2 between s_max = 10 and 100 (orbit_mixture_v2 §2.2).

### 2.3 Laplace form and the fair-comparison identity

Near each peak of the night's likelihood (peak state ŝ, covariance Σ over (p, v, ln f)):

  b_k ≈ ln Σ_peaks L_peak · (2π)^{5/2} |Σ|^{1/2} π_out(ŝ),

while for the true orbit ℓ_k(truth) ≈ ln L_peak. So at the truth

  Δ_k^truth = b_k − ℓ_k(truth) ≈ ln[(2π)^{5/2} |Σ|^{1/2} π_out(ŝ)] + (own-scale Occam)
            = −(Occam of p) − (Occam of v) − (Occam of f) − Σ_g (2.0 to 2.4 nats),

the last term from §2.2's integral over the outlier's own scales, which the inlier does not pay.

With a velocity that the night resolves (σ_v² = σ_p²/Σ_j (t_j − t̄)² ≪ the scale of π(v)), the v-Occam is
ln(2π σ_v² π(v̂)), tens of nats; with a velocity it does not resolve (σ_v larger than π(v)'s scale) the v integral is
1 and b_k reduces continuously to #618's static value. No threshold on span or SNR enters: a short or faint night's
velocity carries no weight by itself.

### 2.4 Measurement on #618's fixture (2026-11-01, laptop, float64, 2 threads)

[`#618`](https://github.com/benjaminpope/virgil/pull/618) at 77d1d7d, `tests/test_aliases.py` six-night fixture, P = 12.1 d
truth, b_k from `broad_loglike` (the PR's exact static integral; its numbers reproduce the PR's float64 b_k table
for the 12.38 d data exactly). ℓ at the truth orbit is evaluated through the fit's own `_night_terms` (per-exposure,
with motion); "static" is the same score at the truth position at the night's reference time, as in
`broad_loglike`:

| night | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| b_k (static, #618) | −40.8 | −75.7 | −54.1 | −55.3 | −46.2 | −47.9 |
| ℓ_k(truth), with within-night motion | −8.5 | −17.0 | −15.0 | −11.6 | −15.4 | −4.2 |
| ℓ_k(truth position), static | −42.6 | −69.0 | −46.1 | −48.5 | −34.4 | −40.2 |
| b_k − ℓ_k, motion | −32.3 | −58.6 | −39.2 | −43.7 | −30.9 | −43.7 |
| b_k − ℓ_k, static | +1.8 | −6.7 | −8.1 | −6.8 | −11.8 | −7.7 |

(12.38 d data: b_k = [−42.4, −74.8, −54.6, −69.5, −51.7, −45.7], Σℓ_k(truth) = −71.7 in this run.)

An independent review reproduced b_k exactly and found Σℓ_k(truth) = −70.6 on float64 data and −77.6 on float32 data
(truth mixture lnL −72.1 and −79.0). The ℓ_k row above comes from one run whose data precision was not pinned
(`_synthetic` caches the precision of its first caller), so it is good to about 1 nat per night, not 0.1. The
conclusion stands: b_k − ℓ_k(truth) is −34 to −58 (float64) and −27 to −58 (float32).

Reading it:

- The static row scores ℓ at the truth *position*, not at the static peak, so it does not test whether #618's b_k is a
  fair static density: on moving data (≈ 5 mas/d over a ≈ 0.01 mas peak) the static model is misspecified and its peak
  lies above its score at the truth position by an amount not measured here (the Laplace p- and f-Occam factors alone are
  ≈ −23 nats, below the −7 ± 5 seen). What the table does show is that the fit's inlier ℓ_k has the within-night motion and
  is 20–40 nats better than its own static score at the truth. The outlier component lacks the velocity the inlier has:
  **the two sides are scored on different data models**. That is the inconsistency the contract repairs.
- Adding v with π(v) of §2.2: b_k rises by the motion improvement and falls by the v-Occam. For the fixture
  (σ_p ≈ 0.01 mas, three frames 1 h apart so σ_v ≈ 0.17 mas/d, v̂ ≈ 5 mas/d, π(ln v) ≈ 0.2 over a width of about 8) the
  Laplace estimate is Δ_k^truth ≈ −17 (p) − 8 (v) − 6 (f) ≈ **−31 nats**, against −31 to −59 now. *Estimate, not run:* b_k
  with velocity needs the 5-D cubature of Stage 3c, whose test (§2.6) checks this number.

### 2.5 Which choice restores #618's six failing tests

None of π(v), the scales or v = 0 does by itself:

1. At the truth every night has Δ_k ≈ −31 to −59, so q_k ≈ e^{−30} / (2K − 1): the true orbit is never an outlier under
   any choice in the family, and the contract choice moves b_k by ≈ +10 nats per night at most (−41 → −31 mean).
2. The mixture log likelihood at the truth is Σℓ_k + ln[B(½, K + ½)/B(½, ½)] = −70.6 − 1.5 = **−72.1** (float64 data;
   −79.0 on float32 data). #618 reports best-fit optima of **−132.8 / −132.9** (float32, n = 20 / 21; q = [1 0 0 0 0 1])
   and, in float64, a band-20 optimum of −238.9 with q = [1 0 0 1 1 1], whose four outlier nights alone cost 209 nats.
   The gap to the truth is ≥ 54 nats on consistent float32 data and 167 on float64. A float32 rerun of the `[12.38]` case
   in the review **passed** (best n = 20, p = 1.000, all q = 0): #618's float32 failure did not reproduce, and the data
   precision must be pinned in the acceptance test (§2.6).
3. Raising b_k by the contract's ≈ +10 nats per outlier night makes a partial optimum with m outlier nights cheaper by
   ≈ 10 m: the float32 optimum (m = 2 outlier nights) gap shrinks from ≥ 54 to ≈ 34 nats, and the float64 optimum (m = 4) gap from
   167 to ≈ 127; both stay positive.
4. So the failing tests are **search failures** (graduated non-convexity's cliff: a night absorbed at q = 1 stops
   pulling, and the optimizer settles). #618's own diagnosis ("a poor fit is absorbed at cost b_k, so the alias contest
   loses discriminating power") is true of the *search*, not of the target: the target still prefers the truth by ≥ 34 nats (float32) and ≈ 127 (float64).
   Restoring the tests is the job of the search and IS of §6, which keep every mode with its weight and need no
   optimizer basin.

**Decision:** the contract of §2.1–2.2 is the target in Stage 3c (it is consistent; #618's v = 0 integral is its
special case and is the Stage 3c unit-test oracle). **Acceptance:** #618's six tests, unedited (including the float32 `[12.1]`
case), pass on the 3c path; any that fails is reported raw and the failure investigated, never loosened. This design does
**not** claim the contract restores them.

### 2.6 Tests registered for Stage 3c

- b_k with π(v) → 5-D cubature against a fine-grid reference (the #618 oracle in 5-D on a toy night), ≤ 4δ.
- b_k with π_out(p) replaced by the uniform disc and σ_v → ∞ (an unresolved night) equals #618's b_k^static: the
  continuity limit and the unit-test oracle.
- Δ_k^truth on the six-night fixture lies within 3 nats of the Laplace estimate of §2.3.
- E[b_k − ℓ_k(truth)] ≤ 0 over noise draws of a *moving* night (the #618 calibration test, extended with v).
- The six #618 tests, unchanged, with the data precision **pinned** (`_synthetic_as`) and both precisions reported: the
  review found `_synthetic` caches whichever precision calls it first.

## 3. Coordinates

All Jacobians below are checked by `scripts/check_orbit_coordinates.py` (autodiff; errors ≤ 6 × 10⁻¹⁴).

### 3.1 Thiele–Innes and the circle amplitudes

Δdec = A X + F Y, ΔRA = B X + G Y with X = cos E − e, Y = √(1 − e²) sin E. In terms of w = X + iY,

  ΔRA + iΔdec = (i/2) (V w + ū w̄),  u = (A + G) + i(B − F),  V = (A − G) − i(B + F),

the split `orbits._thiele_innes_angles` already uses. From the Campbell elements, **u = a(1 + cos i) e^{i(ω+Ω)}**,
**V = a(1 − cos i) e^{i(ω−Ω)}**: |u| and |V| are the prograde and retrograde circle amplitudes, a = (|u| + |V|)/2,
tan²(i/2) = |V|/|u|.

### 3.2 The z coordinates (4:1 cover of TI)

Set u = z_u², V = z_v² with z = (Re z_u, Im z_u, Re z_v, Im z_v) ∈ ℝ⁴. Then |z|² = |u| + |V| = 2a, so a = |z|²/2, and
z/|z| = (cos(i/2) e^{iα}, sin(i/2) e^{iβ}) with α = (ω + Ω)/2, β = (ω − Ω)/2 (Hopf coordinates on S³).

**Prior.** The S³ area element is sin(i/2) cos(i/2) d(i/2) dα dβ = (1/16) d(cos i) d(ω+Ω) d(ω−Ω): uniform in cos i and the
phases, i.e. isotropic. With d ln a = 2 dr/r (r = |z|): isotropic × log-uniform *a* is

  p(z) d⁴z ∝ (dr / r) dΩ₃ = |z|⁻⁴ d⁴z,

on the shell 2a_lo ≤ |z|² ≤ 2a_hi, normalizer π² ln(a_hi/a_lo). Jacobian (checked): |∂(ln a, cos i, ω+Ω, ω−Ω)/∂z| = 32/|z|⁴.
This is the unit quaternion description of SO(3) Haar measure (Shoemake 1992), with the radius carrying a. It is smooth
everywhere except the origin, which a_lo excludes. It has **no 1/r face-on cusp** (3.4% of isotropic orbits lie within 15° of face-on, in either sense: i < 15° or i > 165°), no cos i bound and no Ω fold.

**Cover.** (z_u, z_v) → (±z_u, ±z_v) leaves (u, V), hence TI, unchanged: 4:1. The map is an exact symmetry of the
likelihood; posteriors are reported in invariants (a, cos i, ω ± Ω folded as now), never in z.

**Density in TI′.** With d(A, B, F, G) = ¼ d(u, V) and d²u = 4|z_u|² d²z_u, the normalized density of TI′ (all four preimages
summed) is

  p(TI′) = 1 / [π² ln(a_hi/a_lo) · (|u| + |V|)² · |u| · |V|]  on a_lo ≤ (|u| + |V|)/2 ≤ a_hi,

checked to integrate to 1 (3 × 10⁻⁴, grid). The 1/(|u||V|) factor is the face-on cusp in TI′: it is what a Gaussian
working prior would have to be reweighted by, and why TI′ is sampled rather than marginalized (§6.3).

### 3.3 The ζ map (2:1 cover of the eccentricity vector)

k + ih = e e^{iM_a} is the eccentricity vector in the anchor frame (§3.4). Set k + ih = ζ², ζ uniform on the disc of
radius √e_max. Then e = |ζ|² is uniform on [0, e_max] and M_a = 2 arg ζ uniform, **with a uniform density in ζ** (the
Cartesian density of (k, h) is 1/(2π e)·(1/e_max), a cusp at e = 0; the Jacobian |∂(k, h)/∂ζ| = 4|ζ|² removes it).
The 2:1 cover (ζ → −ζ) is exact. The anchor-frame positions are polynomial in (k + ih, k − ih) (§3.4), hence
real-analytic in ζ (the expansion contains ζ̄²): there is no coordinate singularity at e = 0, where (e, M_a) polar
coordinates have one. **That is a statement about the maths.** #622 defers the Kepler solve in (k, h) to Stage 2b, and its
gradient is not finite at ζ = 0 until that solve exists; the (k, h) solve is a Stage 2b deliverable (§13). #622's disc is
the unit disc, ζ_code = ζ/√e_max, the same map with e_max entering as a scale.

### 3.4 Anchor TI′

Choose an anchor night t_a (§4.2) with mean anomaly M_a = 2π(t_a − τ)/P at the anchor. Rotating the Kepler-plane basis,
w = e^{iM_a} w′, gives p = (i/2)(V′ w′ + ū′ w̄′) with **V′ = e^{iM_a} V, u′ = e^{iM_a} u**: both circle amplitudes keep their
moduli and shift phase by M_a. At e = 0, w′ = e^{in(t − t_a)} does not depend on M_a, so the anchor position is (A′, B′)
(up to the e-corrections) and **M_a survives only in the eccentricity vector**: the Fourier–Bessel series of the
eccentric orbit gives w′ = Σ_m c_m(e) e^{i(m−1)M_a} e^{i m n (t − t_a)} with c_m = O(e^{|m−1|}), so each term is
(e e^{iM_a})^{m−1} (or its conjugate), a polynomial in k ± ih.

**Jacobian.** For fixed P, τ ↔ M_a is affine (|dM_a/dτ| = 2π/P) and the phase rotation acts on (u, V) with unit
determinant (checked), so the prior is unchanged: u′, V′ keep |z|⁻⁴ and M_a keeps the uniform prior. Reporting converts
back to Campbell elements.

### 3.5 Unconstrained charts

HMC needs an unconstrained chart, and a 4-D shell is not diffeomorphic to ℝ⁴, so no map of the form z = g(|y|) y/|y| avoids a
singularity at y = 0 (its density ∝ ρ⁻³ puts 9% of the prior mass inside ρ < 0.1 and a force 3/ρ on HMC). Two charts
replace it:

- **z:** the direction ẑ = y/|y| of an auxiliary y ∈ ℝ⁴ with density N(0, I) (the `unit_vector` construction: |y| is a
  nuisance coordinate with no effect on z), and the radius by its own coordinate ℓ ∈ ℝ,
  ln a = ln a_lo + (ln a_hi − ln a_lo) · logistic(ℓ), with the logistic density on ℓ (so ln a is uniform). The sampled
  dimension is 5 for z, with no singularity: the prior is smooth in (y, ℓ).
- **ζ:** ζ = √e_max · y/√(1 + |y|²), y ∈ ℝ². Then |ζ|² = e_max |y|²/(1 + |y|²) < e_max, the map is smooth (no kink at 0), and
  its Jacobian is e_max (1 + |y|²)⁻², so the density in y is ∝ (1 + |y|²)⁻² for ζ uniform on the disc (checked).

The charts have no statistical meaning: they only decide HMC geometry, and their log-densities are analytic.

### 3.6 Labels, Campbell, status

- **Label swap** (which star is the primary) is TI′ → −TI′ (antipodal) at the same core, with the disc label swapped when
  it applies; the isotropic prior is symmetric under it. The reference holds both modes for twins, with a transform for
  log f ≤ 0. The two labels are carried as discrete particle state with Gibbs flips.
- **Campbell's fold.** With the fold Campbell maps 1:1, so no spurious ln 2 enters against the null. Campbell elements
  are for reporting only.
- **Status: both maps are derived departures** (no source paper for z or ζ; the algebra is §3.1–3.4 and the check
  script). They replace orbit_dehack decision 10 (Campbell (h, k, λ_ref) → TI′ in z and ζ). Stage 2a registers them as
  *derived* in the methods register, with prior-recovery tests: cos i, ln a, ω ± Ω, e uniform (the check script's KS
  tests are the first draft), the cell partition giving equal Ẑ in Campbell and the new coordinates on toys, and the TI′
  density above.

## 4. Cells

### 4.1 The partition is exact

With anchors t_a < t_b and T = t_b − t_a, write T/P = N + ρ, N integer, ρ ∈ [0, 1). Log-uniform P gives

  π(N, ρ) ∝ 1/(N + ρ).

Each cell is a band of x = T/P between its edges, x ∈ [N − ½, N + ½] for the cells of §4.4 (ρ ∈ [0, 1) for the default
edges at integers; with edges x_j < x_{j+1} the cell's ρ range is x − N and its normalized prior density is
(N + ρ)⁻¹/ln(x_{j+1}/x_j)). **Each band starts from its normalized cell prior, and Ẑ_N is multiplied by the cell's prior
mass** ln(x_{j+1}/x_j)/ln(x_max/x_min) before p(N) is formed. The long-period band (§4.5) has x < x_low.

The cells (N, ρ ∈ [0, 1)) partition the period domain for **any** choice of edges, so Z = Σ_N Z_N and the prior mass of
a cell with edges x_j < x_{j+1} (x = T/P) is ln(x_{j+1}/x_j)/ln(x_max/x_min), x_min = T/P_max, x_max = T/P_cut. Edges
and anchors therefore change efficiency and labels, never the answer. Within a cell the sampling coordinate is ρ, with
density ∝ 1/(N + ρ) (an analytic log-prior term).

### 4.2 Anchors

The two anchor nights must differ, so that T > 0 on every data set. First anchor: the night with the smallest σ_p
of its best peak (the peak of largest L_j; a night's σ_p is a peak property) among nights in the first quarter of the span;
last anchor: the same in the last quarter; if a quarter holds no night, the first (last) night. Then T ≥ span/2. One-season
data are not special: the quarters of the span differ. With T < span, gaps longer than T exist and a cell can hold
sub-ridges spaced Δx = T/span < 1, so the cell structure is approximate there; edges (§4.4) and the exact partition
(§4.1) keep the answer unaffected. A poor anchor costs efficiency only (§4.1). The quarter is a budget (§10).

### 4.3 What N changes

Anchor *positions* do not depend on N (they are the TI′ amplitudes); velocities scale with (N + ρ)/T. Aliasing is
approximately the spectral window of x_k = (t_k − t_a)/T, and exactly so only for circular orbits without within-night motion.

### 4.4 Edges

Edges are fixed at x = N ± ½ (`make_cells`' default). The pilot pass and valley edges of the previous draft are
retired (2026-11-02). They only moved labels, and the search and IS of §6 do not need the posterior to sit inside one
cell: a mode straddling an edge is found from both sides, and the IS indicator splits its mass exactly.

**Modes on an edge.** In the cell's logistic period chart, the half of such a mode that lies inside the cell has its
maximum pushed inward by the log-Jacobian, so its Laplace covariance is distorted. This costs efficiency, not
correctness:

- IS weights use the exact target and the exact q, so the estimate stays unbiased. Each cell's evidence integrates
  only over that cell. The defensive share α bounds the weights, and a poor t5 fit shows up as low ESS (flag, §6.6).
- HMC in the logistic chart samples the truncated target exactly.

Overlapping cells (e.g. N ± 0.6) are not used. The cells must partition x, or their evidences double-count. The edge
case is measured in §6.7's grid, where every neighbouring cell agrees with the reference.

### 4.5 The long-period band

Everything with x < x_low (below the lowest cell: P longer than the lowest cell's upper edge, up to P_max) is one band,
sampled in ln P with π(ln P) uniform; for P > T it carries no cycle count. Its mass is per band, never pooled.

## 5. Night summaries (position level)

Ben's rule: velocity only with adequate span and SNR, and no threshold. This is the interface Stage 2b codes against.

**What is computed, and at what.** Each night k is fitted **per peak at visibility level** with linear motion over the
**whole night** (no half-night split, so offsets and spanning gain modes are kept; `OIData._subset` raises on offsets, and
hd_9312 had 0 velocities; `_MAX_SHIFT` is deleted). The frame time offsets t_j − t̄_k are stored with each summary.

**What the orbit predicts for a summary** (Ben, 2026-11-02). It predicts the exact least-squares projection of the orbit's
positions at the stored frame offsets onto (p, v), not the instantaneous state at t̄_k. (`NightSummary.offsets` and
`.weights` store the offsets and per-frame scalar weights, equal for now; `_Space.night_states` applies the map. The
visibility fit weights each frame by its own information (unflagged rows, errors, baseline orientation: a 2 × 2 matrix
per frame), so the map is the exact projection under the stored scalar weights and matches the fit only to first order
in the track's departure from a line. Open for Ben: weights from the per-frame Fisher information. A one-frame night has no velocity fit and takes the instantaneous velocity.) The projection is a fixed linear
map of the per-frame positions, so it costs one small matrix product per night. It removes the curvature bias
½·a(2π/P)²·T_ss/n of §1.3, which reached about 15σ for eccentric short-period orbits. This is what lets P_cut drop
P_night.

The fit runs at

- **global scale s_g = 1** for every group (the tempered errors of §1.1), the summary being stored so that any s_g is a
  closed-form rescale: per peak j and group g the data count n_gj, the minimum χ²_gj and the Hessian H_gj of ½χ², giving
  L_j(s) = L_j exp{−Σ_g [n_gj ln s_g + (χ²_gj/2)(s_g⁻² − 1)]} and Σ_j⁻¹(s) = Σ_g s_g⁻² H_gj. The mean m_j is held at its
  s = 1 value (exact for one group; a second-order error otherwise, repaired by leg 2, which is exact);
- **θ (primary diameter), disc label and vis_amp at the pipeline defaults, with a per-epoch amplitude profiled** by
  variable projection exactly as the engine does now (`aliases._epoch_residuals`, `amp_range`). This **departs from v2 §2.3**
  ("marginalize, not profile"): the profile is what the summary's Hessian already contains. Leg 2 samples θ and
  marginalizes the amplitude as v2 prescribes, so the departure is confined to leg 1, and it is flagged for Stage 2b.
- **Jitter σ_J** (orbit_dehack §F): one jitter added in quadrature to every σ_p, fitted by type-II maximum likelihood over a
  1-D grid, maximizing log Σ_n S_n(σ_J) on the seed profile (grid in §10). It exists only at the position level, because
  a Laplace σ_p (≈ 0.01 mas on #618's fixture) is overconfident wherever the summary model is imperfect; leg 2 has none.

Each night's catalogue is (m_j, Σ_j, ln L_j, n_gj, χ²_gj, H_gj) over its kept peaks, plus ln L_k⁰, the night's visibility
likelihood with no companion (used in §8).

- **Variance of v:** σ_v² = σ_p² / Σ_j (t_j − t̄)² (12σ_p²/span² for uniform sampling), with σ_p² including σ_J²: a short
  or faint night's velocity carries no weight automatically.
- **Peak keep rule (derived):** the night factor is (1 − ε) e^{ℓ_k} + ε e^{b_k}, and an orbit through peak j contributes at
  best (1 − ε) L_j with *no* π_out or Occam factor. A peak is therefore kept iff **(1 − ε_lo) L_j ≥ δ ε_lo e^{b_k}**
  under (ii) (under (iii′): e^{b_k} replaced by e^{o_k} at the peak's own state), ε_lo the smallest Gauss–Chebyshev node of
  the ε integral (§10). Comparing the Laplace-penalized outlier share instead (the first draft) would drop a true-companion
  peak 8 nats below a sidelobe, which is still ≈ 23 nats above the floor. J_k is not bounded by 1/δ: its distribution is
  reported, and its cost enters T_summ (§9).
- **Position-level likelihood** of night k at orbit x, at scale s:
  ℓ̃_k(x, s) = ln Σ_j L_j(s) exp(−½ (s_k(x) − m_j)ᵀ Σ_j(s)⁻¹ (s_k(x) − m_j)), s_k(x) = (p_k(x), v_k(x), ln f(x)). Under (ii)
  the outlier term is the **same b_k** (§2), so the two legs share the floor that makes the bridge's ratio L_vis/L_pos
  finite wherever the catalogue is complete. Under (iii′) it is
  õ_k(x, s) = ln ∫₁^{s_max} [peak sum at scales s_g t, all g jointly] dt/(t ln s_max), the same inflation integral on the catalogue,
  evaluated by Gauss–Legendre in ln t (node count in §10), so the two legs again share the structure.
- **Completeness assumption:** a visibility-level mode absent from the catalogue gets position-level mass only from the
  outlier term and the bridge cannot find it. The planted-peak test of #618 (`test_a_planted_narrow_peak_is_found`) is the
  check; the catalogue is the same one that makes b_k exact.
- Summaries feed only leg 1; the target of leg 2 is the per-exposure likelihood.

## 6. Search, sampling and evidence per cell

**Rewritten 2026-11-01 after the leg-1 review** (Ben's decisions, quoted where they are his):

1. The evidence rests on the search finding every mode, so **search coverage is a gated, tested component** (§6.1), with a
   sentinel that fails a cell ending at the all-outlier floor.
2. **No tempering** from the band prior: "I don't think we need to temper unless there is a strong reason to".
3. Laplace is allowed as a baseline but not preferred: "reduce it to a low-dim continuous search space (ideal for monte
   carlo methods) and analytically propagating that information through a grid of N integer aliases" (§6.1, §6.7).
4. The within-cell sampler is **HMC** ("I'd have gone with hmc because why not"; §6.3).
5. Stage 3b's slow tests are replaced by a **population test** (§14).

**Why the tempered SMC was dropped.** At σ/a ≲ 10⁻³ the band prior's mass on the needle that holds the posterior is
≲ e⁻⁴⁰. Tempering from it meets a first-order transition: the β-path stays on the all-outlier floor and then jumps, and
no ESS rule sees the jump coming. Skilling (2006) named this failure of thermodynamic paths and built nested sampling
for it. The review toys ([`scripts/leg1_review/`](../scripts/leg1_review/), `table.log`) measured it. Whole-mixture
tempering (#629's leg) was biased by −47 to −156 nats, per-label tempering by −27 to −156, and the Joker-style path by
−7 to −113. At σ/a = 10⁻³ and 10⁻⁴ each of these ended with a single stage at the floor (ln Ẑ = K·b). A reference path
started from a search-built proposal was unbiased. The cure is in the start, not the path: find the modes first, then
integrate from a proposal built on them.

The structure per cell N (§4) is therefore:

**search (§6.1) → defensive mixture q → evidence Ẑ_N by importance sampling (§6.4) → posterior draws by
resample-move with HMC (§6.3)**

The sentinel and ESS flags are in §6.6. The same draws serve every N where that is exact (§6.7).

### 6.1 Search: minimal sets in a low-dimensional core

**The space.** Given the nonlinear core c = (ρ, ζ) (§3.3–3.4; three real dimensions), the night states are linear in
TI′, and ln f is a weighted mean of the nights' flux summaries. For fixed c and fixed night labels, TI′ and ln f are
therefore solved by weighted least squares: variable projection (Golub & Pereyra 1973), as `aliases._epoch_residuals`
already does for amplitudes. The search runs only over c and the labels.

**Why a fixed coherence grid fails.** Each night's term is the defensive mixture ln[(1 − ε)e^{ℓ_k} + ε e^{b_k}]. Its
basin in ρ around a mode is the set of ρ over which the inlier nights stay within about σ√(2Δ_k) of the orbit, where
Δ_k = ℓ_k^max − b_k is the night's height above its floor. Beyond that a night becomes an outlier and stops pulling. The
basin therefore shrinks ∝ σ/a, while a grid at fixed coherence (`orbits.period_grid`, k points per cycle) does not. On
the toy, a k = 4 grid followed by Newton missed the needle at σ/a = 10⁻⁴ (coverage 0, IS bias −84 nats). At 10⁻² it
converged to partial-outlier maxima. Both runs were measured before the rewrite.

**Minimal sets (RANSAC; Fischler & Bolles 1981; elemental sets, Rousseeuw 1984).** A minimal set is the smallest set of
nights whose summaries determine (ρ, TI′) at a fixed ζ:

| data | equations per night | unknowns at fixed ζ | m (nights per set) |
|---|---|---|---|
| positions and velocities | 4 | 5 (ρ, TI′) | 2 |
| positions only | 2 | 5 | 3 |

For each set S and each ζ start (below):

1. Grid ρ at coherence k_coh over the set's own span Δ_S, so n_ρ(S) = ⌈k_coh Δ_S/T⌉ + 1 points per cell.
2. From each local minimum of the set's VarPro residual on that grid, run a bounded 1-D minimization (Brent) in ρ.
3. With velocities, or with m + 1 nights in the set, the solve is over (ρ, ζ) jointly, started at the grid node.

**Concentration steps (Rousseeuw & Van Driessen 1999).** From each set's solution:

1. Label each night inlier if ℓ_k > b_k, i.e. if its inlier component is the larger one, at the jointly profiled
   inlier scale ŝ (§6.9: labels and ln s are chosen together given the parameters).
2. Maximize the labelled target (inliers Gaussian, outliers at b_k) by Newton in (c, TI′, ln f).
3. Relabel, and repeat until the labels repeat (at most n_C steps, §10).

Each step is coordinate ascent on max_labels of the labelled target: the labels maximize it given the parameters, and
the parameters given the labels. The objective therefore never decreases, and the steps end in finitely many moves
(the C-step argument of Rousseeuw & Van Driessen). Finally, Newton on the **label-summed** target gives the mode, and its
Hessian gives the Laplace covariance.

**Distinct modes.** A new mode is the same as a kept one if their Mahalanobis distance is < 1 under **both**
covariances. A one-sided test lets a broad partial-outlier mode absorb the needle inside it. On the toy at σ/a = 10⁻⁴
that error cost 62 nats in Laplace and was fixed by the symmetric test.

**Coverage argument (derived for ρ, independent of σ/a).** Take any posterior mode whose inlier set I contains a
minimal set.

- Among the enumerated sets is one drawn from I that spans I's earliest and latest nights.
- That set's residual has a minimum inside the mode's needle, because its span equals the needle's coherence length,
  and the grid at k_coh > 2 points per cycle brackets it.
- The other inliers of I lie between the set's nights. They are interpolated with error about σ, which is below the
  inlier threshold σ√(2Δ_k).
- The first C-step therefore labels I (or a superset) inlier and converges to the mode.

Nothing in this argument depends on σ/a, nor (with σ read as ŝσ, §6.9) on the inlier scale. The ζ starts have no such argument. On the circular toy ζ = 0 is exact; for
eccentric orbits the start set is a declared budget (the `starting_orbits` grid in e and phase, §10), and its coverage
is a Stage 3b test on eccentric toys.

**Number of sets.** All C(K, m) sets are enumerated while C(K, m) ≤ n_set_max (§10): 91 pairs or 364 triplets at
K = 14. Above that budget the sampling is stratified so that the coverage argument still holds. Plain RANSAC
sampling does not suffice: the chance that a random set contains the two outer inliers falls as 1/C(K, 2).

- All C(K, 2) span pairs (t_i, t_j) are enumerated. There are 1770 at K = 60, so the pair holding I's outer nights
  is always among them, and pairs (m = 2) never need sampling.
- For m > 2, each span pair takes n = ln δ / ln(1 − w^(m−2)) random fills of m − 2 nights from between its ends
  (Fischler & Bolles' count applied to the interior nights only). Here w = 1 − ε_hi is the declared inlier fraction.
- The cost is C(K, 2)·n sets. At m = 3, ε_hi = 0.3 and δ = 10⁻³, n = 6.

**The coverage contract (gated).** The search is a component with its own test, not an implementation detail:

- On every toy with an exact reference (§6.5, §14), the modes found must hold ≥ 1 − δ of the exact posterior mass of
  the cell. Mass is attributed to basins cut at the profile minima between the oracle's peaks (prominence ≥ 1 nat).
- **Sentinel (Ben's decision 1).** A cell fails, with flag `cell-at-floor`, when both of the following hold:
  - under the IS draws the posterior mean of the number of inlier nights is < m (the evidence came from the floor);
  - the search's best mode has ≥ m inliers and lies more than W = ln(1/δ) above the floor K·b.

  This is the signature of every review failure (the evidence ended at K·b while a mode existed). It is cheap: one
  weighted mean over draws the estimator already has.

  The sentinel catches an estimator that falls to the floor when the search has found the mode. It cannot catch a
  mode the search missed, because condition 2 then has no mode to point at. Missed modes are the coverage contract's
  job (above, §6.5, §14).

  If the search returns no mode at all, q is the prior alone (α = 1). The IS estimate stays unbiased, the sentinel
  cannot fire, and the cell carries the ESS flag of §6.6.

### 6.2 Leg 2: the visibility level (changed; flag for Ben)

The tempered bridge from L_pos to L_vis is withdrawn for the same reason as leg 1. Its efficiency rested on
L_pos ≈ L_vis, with no defensive bound: e³⁰ to e⁶⁰ per night at §2.4's numbers. The replacement has the same shape as
leg 1:

1. Refine each leg-1 mode by Newton on the per-exposure target (§1.1), with labels summed per night.
2. Build the defensive mixture from the visibility-level Laplace covariances.
3. Estimate Z_N^vis by IS (§6.4).

The catalogue-completeness assumption (§5) stays: a visibility mode absent from the catalogue can be reached only
through the defensive prior share. The sentinel applies here too, on the visibility-level inlier count. This changes
#630 (§13).

### 6.3 Within-cell sampler: HMC by resample-move

**Coordinates.** All are unconstrained, and each log-Jacobian is analytic:

| coordinate | chart | dimension |
|---|---|---|
| ρ | logistic chart on the cell's edges, ρ = x_lo + (x_hi − x_lo)·logistic(r) − N, with log-density ln(1/(N + ρ)) + Jacobian | 1 |
| ζ | §3.5 chart | 2 |
| z | §3.5: four Gaussian direction coordinates plus the logistic ln a | 5 |
| ln f | | 1 |
| ln s_g | one per group | 1 per group |
| θ, vis_amp | leg 2 only | |

The leg-1 dimension is 9 plus the number of groups. **TI′ is sampled, not marginalized.** Its prior p_iso ∝ |z|⁻⁴ is
not Gaussian, and with labels summed the likelihood in TI′ is a 2^K-term Gaussian mixture. A closed form exists only
per label set under a Gaussian working prior, which is the Joker-style route J of the previous draft. That route was
biased by 0.2–4.7 nats at a broad working scale in the old toy and grows exponentially with K. **Labels are summed
exactly in the target**, per night: there is no discrete state and no label sweep.

**Resample-move (Gilks & Berzuini 2001).**

1. Draw n_IS points from q (§6.4) and resample them by their IS weights; the weights already give Ẑ_N.
2. Move each resampled point by n_HMC HMC transitions that leave the cell posterior invariant.

The weighted draws are exact before the moves. The moves only repair the duplicates that resampling creates, so a
weak sampler costs efficiency, never correctness.

**HMC settings (Neal 2011):**

- Coordinates are whitened by the Cholesky factor of the dominant mode's Laplace covariance (the mass matrix).
- The step size ε is tuned by dual averaging to acceptance 0.8 (Hoffman & Gelman 2014) during n_warm warm-up
  transitions, and jittered ±20 % per transition.
- The trajectory length is L = ⌈π/(2ε)⌉, a quarter period of a unit Gaussian in whitened coordinates.

There is no tempering. Each draw starts inside a found mode, and the only multimodality HMC must cross is within a mode
the search already split. Moves between modes are not needed, because the mode weights come from IS.

**HMC runs only where it is needed:** in cells with p̂(N) ≥ δ/B, for posterior draws. Ẑ_N never needs it (§6.4).

### 6.4 Evidence per cell: four candidates, measured

All four candidates target the same cell posterior, with labels and TI′ in the target as in §6.3:

- **(a) Bridge sampling** (Meng & Wong 1996; Gronau et al. 2017).
  - Per-mode bridges between q_m and the posterior restricted to mode m by the balance-heuristic partition of unity
    r_m = w_m q_m/Σ_j w_j q_j (Veach & Guibas 1995).
  - The posterior side is drawn by HMC (§6.3) on p·L·r_m.
  - The prior share is done by IS.
- **(b) Defensive importance sampling** (Hesterberg 1995): q = α·prior + (1 − α)·Σ_m w_m t₅(ĉ_m, Σ_m).
  - Student-t components with 5 degrees of freedom sit at each mode's Laplace covariance, with w_m ∝ the Laplace
    evidence (`aliases._Proposal`).
  - In the chart the target is exactly invariant under 16 linear maps per mode, and each found mode enters q with all
    16 images, its weight split equally. The maps are the z cover (y_z[0:2] and y_z[2:4] sign flips, 4), the ζ cover
    (y_ζ → −y_ζ, 2) and the label swap TI′ → −TI′ (z_u → i z_u, z_v → i z_v, 2), which is a symmetry because the target
    sums the brighter/fainter sign. Without the images, IS would still be unbiased through the prior share, but its
    variance would be large. Stage 3b′ tests the invariance numerically.
  - The weight is p·L/q ≤ L_max/α, which is bounded, so its variance is finite whatever the modes look like (Hesterberg's
    defensive argument). The prior share also covers any mode the search missed, at a cost the sentinel and the ESS
    expose.
- **(c) Laplace** (Tierney & Kadane 1986): Σ_m of each mode's Laplace evidence. Baseline only.
- **(d) Nested sampling** (Skilling 2006), as BlackJAX's nested slice sampler (Yallup, Kroupa & Handley 2026;
  `blackjax.nss`, ≥ 1.7; jaxns, Albert 2020, is the alternative).
  - Live points start from the cell prior, with no search.
  - Run with 5·d inner slice steps, 10 % of the live set replaced per step, and stop when the live points hold
    < e⁻³ of Ẑ.

Toys: [`scripts/leg1_toys/`](../scripts/leg1_toys/).

- **Model.** Circular orbit, positions only, TI′ with a Gaussian prior (so the reference is exact); ε = 0.1 fixed;
  uniform-disc floor; the cell N = round(T/P_true).
- **Patterns.**

  | pattern | nights | seasons | outlier nights |
  |---|---|---|---|
  | K6_2s | 6 | 2 | 0 |
  | K10_2s | 10 | 2 | 1 |
  | K10_3s | 10 | 3 | 1 |
  | K14_4s | 14 | 4 | 2 |

- **Reference.** ln Z_N by exact sums over the 2^K label sets, with TI′ integrated in closed form, and adaptive quadrature
  in ln P that resolves the needle (`toy.Problem.exact`).
- **Cost** is in likelihood evaluations, with a gradient counted as 3. Variance × cost is the figure of merit.
- **Settings.** IS uses n_IS = 4000. Bridge uses 4 chains of 150 warm-up and 250 draws per mode, with 1000 proposal and
  1000 prior draws.

**(a)–(c), with the search** (`search_evidence.py PATTERN REL 4`). Bias is the mean of 4 replicates minus exact; sd is
over replicates. Cost counts the estimator only; the search cost, which every method needs, is in §6.5.

| pattern | σ/a | exact ln Z_N | (c) L bias | (b) I bias | I sd | I cost | I var × cost | (a) B bias | B sd | B cost | B var × cost |
|---|---|---|---|---|---|---|---|---|---|---|---|
| K6_2s | 1e-2 | −5.213 | −0.001 | −0.003 | 0.012 | 4.0e3 | 0.58 | +0.002 | 0.013 | 1.8e6 | 315 |
| K6_2s | 1e-3 | 10.910 | −0.002 | −0.005 | 0.012 | 4.0e3 | 0.57 | −0.000 | 0.013 | 2.1e6 | 358 |
| K6_2s | 1e-4 | 27.028 | −0.002 | −0.005 | 0.012 | 4.0e3 | 0.56 | −0.000 | 0.011 | 2.6e6 | 326 |
| K10_2s | 1e-2 | −6.024 | −0.001 | −0.003 | 0.012 | 4.0e3 | 0.57 | +0.002 | 0.014 | 6.3e6 | 1260 |
| K10_2s | 1e-3 | 23.912 | −0.004 | −0.006 | 0.012 | 4.0e3 | 0.56 | −0.002 | 0.012 | 6.4e6 | 897 |
| K10_2s | 1e-4 | 53.845 | −0.004 | −0.006 | 0.012 | 4.0e3 | 0.56 | −0.003 | 0.013 | 4.6e6 | 757 |
| K10_3s | 1e-2 | −5.178 | −0.000 | | | | | | | | |
| K10_3s | 1e-3 | 24.762 | −0.003 | −0.005 | 0.012 | 4.0e3 | 0.56 | | | | |
| K10_3s | 1e-4 | 54.696 | −0.003 | −0.006 | 0.012 | 4.0e3 | 0.56 | | | | |
| K14_4s | 1e-2 | −7.447 | −0.000 | −0.003 | 0.012 | 4.0e3 | 0.56 | | | | |
| K14_4s | 1e-4 | 80.055 | −0.004 | −0.006 | 0.012 | 4.0e3 | 0.56 | | | | |

Blank (a) cells were not run. The bridge costs 10³ times IS per run, and the K6/K10_2s rows already separate the two
by 500–2000× in var × cost. The K10_3s 1e-2 IS row was lost to an interrupted run.

The IS biases (−0.003 to −0.006) are a quarter to a half of one replicate's sd. All four replicates share draw seeds
across rows, which is why the sd column repeats. The Laplace bias is ≤ 0.004 here because the toy's modes are close to
Gaussian once the needle is found. It is a baseline only, because nothing bounds its error when a mode is not Gaussian.

**(d) Nested sampling** (`ns_evidence.py`; BlackJAX 1.7.1 in a scratch venv, since the worktree venv lacks it until #623).
Runs are unseeded, starting from the cell prior:

| pattern | σ/a | n_live | stop rule | ln Ẑ bias (per run) | H (nats) | evaluations per run |
|---|---|---|---|---|---|---|
| K6_2s | 1e-2 | 200 | standard (live < e⁻³ of Ẑ) | −0.06 | 25 | 9.4e5 |
| K6_2s | 1e-3 | 500 | standard | −66.6, −72.4, −67.1 | 0–13 | 1.2e6–1.4e6 |
| K6_2s | 1e-4 | 500 | standard | −88.5, −88.5, −78.7 | 0–13 | 1.2e6–2.0e6 |
| K10_3s | 1e-3 | 500 | standard | −119.8, −113.2, −119.1 | 10–22 | 1.2e6–2.2e6 |
| K10_3s | 1e-4 | 500 | standard | −157.2 | 0 | 2.6e6 |
| K6_2s | 1e-3 | 500 | forced to −ln X ≥ 100 | −66.6, −72.4 | 0–12 | 8.3e7–8.5e7 |
| K6_2s | 1e-4 | 500 | forced | −88.5, −88.5 | 0 | 3.4e7–8.3e7 |
| K10_3s | 1e-3 | 500 | forced | −119.8, −113.2 | 10–22 | 2.4e7–8.6e7 |
| K10_3s | 1e-4 | 500 | forced | −157.2, −157.2 | 0 | 2.2e7–8.3e7 |

**Unseeded nested sampling fails here exactly as tempering does.** The all-outlier floor is a likelihood plateau
(Fowlie, Handley & Su 2021). The live points sit on it, and the standard rule stops after about 30 compression steps
with ln Ẑ = K·b (H = 0). Forcing 100 nats of compression (950 steps, 30–60× the cost) changes nothing: the slice
moves never enter the needle, whose prior volume is ≈ e⁻⁴¹ at 1e-3 and ≈ e⁻⁵³ at 1e-4 in K6_2s (ln P width over the cell, times the four TI′ widths σ/√K over the prior SW). Runs that left the plateau
(H = 10–22) found partial-outlier modes, not the needle. At 1e-2, where the needle is wide, NS is unbiased (−0.06),
but its variance × cost (≈ 0.18 × 9.4e5 ≈ 1.7e5) is about 3 × 10⁵ times IS's. More live points push the threshold down
but cannot remove it. The live set must sample a volume of ≈ e⁻⁴⁰ to e⁻⁵⁰ before the likelihood shows any slope
towards it. That is the structure Kroupa & Handley (2026) report for aliased time-delay likelihoods.

**Seeded nested sampling** is posterior repartitioning (Chen, Feroz & Hobson 2023): reference q (the search mixture),
likelihood π·L/q, same Z. Its information is H = KL(posterior ‖ q) ≈ 1 nat, so it would be cheap and valid. It is
implemented (`ns_evidence.py … seed`) but was not run before this PR. Once the search has built q it duplicates (b) at
higher cost, so it is a cross-check option, not a candidate.

**Choice: (b) defensive IS from the search-built mixture.** It was unbiased on every row, at a var × cost
500–2000 times below bridge sampling, and its weights are bounded by construction (L_max/α). Its error comes from its
own ESS (§6.6), and its draws are the posterior once resampled (§6.3). (a) needs HMC per mode for no gain. (c) is
reported beside Ẑ as a diagnostic: a Laplace–IS gap above 1 nat flags a non-Gaussian mode. (d) needs the search as
much as (b) does. Unseeded, it fails; seeded, it is (b) at higher cost.

### 6.5 Search coverage on the toys

The search is the triplet minimal-set search of §6.1, with k_coh = 4 and ζ = 0 on these circular toys. "Sets × grid"
counts the candidate starts after the per-set ln P grid. Coverage is the share of the exact cell posterior mass in
the basins of the modes found. The oracle peaks are the profile's local maxima of prominence ≥ 1 nat holding
≥ δ of the mass.

| pattern | σ/a | candidate starts | modes kept | search evaluations | wall (s) | coverage | oracle peaks found |
|---|---|---|---|---|---|---|---|
| K6_2s | 1e-2 | 37 | 6 | 1.6e4 | 2.5 | 1.000000 | 1/1 |
| K6_2s | 1e-3 | 37 | 6 | 3.0e4 | 5.7 | 1.000000 | 1/1 |
| K6_2s | 1e-4 | 37 | 6 | 4.4e4 | 5.6 | 1.000000 | 1/1 |
| K10_2s | 1e-2 | 211 | 21 | 1.4e5 | 19 | 1.000000 | 1/1 |
| K10_2s | 1e-3 | 212 | 14 | 2.1e5 | 22 | 1.000000 | 1/1 |
| K10_2s | 1e-4 | 212 | 10 | 1.8e5 | 22 | 1.000000 | 1/1 |
| K10_3s | 1e-2 | 147 | 12 | 5.3e4 | 8 | 1.000000 | 1/1 |
| K10_3s | 1e-3 | 147 | 8 | 9.5e4 | 2 | 1.000000 | 1/1 |
| K10_3s | 1e-4 | 147 | 4 | 2.0e5 | 3 | 1.000000 | 1/1 |
| K14_4s | 1e-2 | 398 | 54 | 1.5e5 | 2 | 1.000000 | 1/1 |
| K14_4s | 1e-4 | 398 | 11 | 1.7e5 | 3 | 1.000000 | 1/1 |

The search costs 4–50 times the IS (1.6e4–2.1e5 evaluations against 4e3), so it dominates the cell. Before the rewrite
the k = 4 coherence grid had coverage 0 at 1e-4 on K6_2s (IS bias −84 nats). The sentinel raised `cell-at-floor`
there, which is the case it exists for. The modes kept beyond the needle are partial-outlier maxima. They cost
mixture components and nothing else: their IS weight is what it is. The ζ starts are not exercised by these toys.
Their coverage is the eccentric test of Stage 3b (§14).

### 6.6 Errors, flags, failure

- **Error of Ẑ_N.** By the delta method, sd(ln Ẑ_N) ≈ √(1/ESS − 1/n_IS), with ESS = (Σw)²/Σw² the Kish effective sample
  size of the IS weights. One run per cell suffices, so the two-replicate rule of the previous draft is retired.
- **Flags.** One diagnostic and one flag each, with no retry and no cascade:
  - `cell-at-floor`: the sentinel of §6.1. The cell **fails** and is reported as a failure, never as an evidence.
  - `cell-undersampled`: ESS < ESS_target. This is information only.
  - `cell-unreliable`: ESS < ESS_FAIL while the bound of §6.10 admits p(N) ≥ δ. The cell fails.
  - `no-mode` (per system): no mode in any cell and every ln Ẑ_N at the all-floor value. Information; the system goes
    to the §8 null test before any p̂(N) is reported (§6.9).
- The mass is per cell, never pooled. Ridge modes carry their quadrature mass, never a Laplace mass (§6.10).

### 6.7 Across the alias grid N (Ben's idea)

Ben's suggestion is to reduce the problem to a low-dimensional continuous search and to propagate it analytically
across the integer aliases N. LITMUS (McDougall, Pope & Davis 2026) is the closest prior art. Its `hessian_scan` scans an
aliased lag, optimizes and takes a Laplace evidence per slice, then integrates across the grid. Here is what is exact
across N, and what is not.

**Exact across N:**

- The prior factor, π(N, ρ) ∝ 1/(N + ρ).
- The anchor nights' positions, which are the TI′ amplitudes and independent of N (§4.3).
- The normal equations of TI′ on the anchor nights. With velocities they are polynomial in (N + ρ), of degree 2.

**Not exact across N.** Every other night enters through its phase 2π(N + ρ)(t_k − t_a)/T. For two-season data the
in-season offsets enter only through P = T/(N + ρ):

  L_N(ρ, θ) = L̃(ρ, θ, P = T/(N + ρ)).

This is smooth in N across neighbouring cells, so one search over (ρ, P) covers them all. Each intermediate season s
adds a phase 2π(N + ρ)x_s that is not smooth in N. Propagating in an extended space then costs about k^{S−1}·Δ_max/T
of a per-N search, with S the number of seasons, so it pays only for S ≤ 2–3 seasons.

**What can be shared exactly** is the draws:

- One defensive mixture over the modes of all cells, sampled once.
- Ẑ_N is the IS mean of p·L·1{x ∈ cell N}/q, which is unbiased for every N at once.
- Draws are allocated by mass, so a cell with negligible mass gets few draws. Its estimate is still unbiased in
  expectation, but in practice it is skewed (measured below).

The per-N work that remains is the search's cross-season minimal sets and the evaluations of non-anchor nights.

**Measured** (`alias_grid.py K10_2s 1e-2 2 4`: cells 18–22, 4 replicates, the same total number of draws each way):

| N | exact ln Z_N | per-N bias | per-N sd | shared bias | shared sd |
|---|---|---|---|---|---|
| 18 | −61.139 | −0.005 | 0.009 | −40.0 | 1.6 |
| 19 | −50.312 | −0.001 | 0.013 | −52.9 | 0.8 |
| 20 (true) | −7.638 | −0.003 | 0.012 | +0.001 | 0.003 |
| 21 | −51.969 | −0.003 | 0.017 | −45.8 | 4.0 |
| 22 | −64.784 | −0.007 | 0.006 | −34.8 | 2.0 |

Cost: per-N 1.0e6 evaluations, shared 1.8e6. The shared search over five cells cost more than five per-N searches,
because each set's grid spans every cell. The total-variation distance of p(N) from exact is 3e-22 per-N and 2e-19
shared. The reference uses 1.6e6 grid points over the five cells. At the default 4e5, the narrow peak of N = 22 fell
between grid points and its ln Z was 0.57 nats low. The other cells agree to 0.001 between the two grids.

**Conclusion: per-N is the default.**

- Shared draws are unbiased in expectation, and p(N) comes out right because the cells they get wrong carry ~e⁻⁴⁰ of
  the mass.
- But the mixture allocates draws by Laplace mass, so a negligible cell sees only prior draws on the floor. Its ln Ẑ_N
  is then 34–53 nats low in practice: a heavily skewed estimator, not a usable one.
- The shared search did not save work either.

What propagates analytically across N is the prior factor and the anchor terms (above), which per-N runs use for free.
Ben's idea pays as a shared *search* only where a single (ρ, P) search covers several cells, i.e. two-season data. That
is the extension to measure on the §14 population, not a default.

### 6.8 Why not tempering, and where nested sampling fits

- **Tempered SMC and parallel tempering** (Pigeons, Octofitter; Thompson et al. 2023, Surjanovic et al. 2023) both
  cross the first-order transition of §6. Both can be run *from* the search-built mixture instead of the prior. That is
  a stepping-stone path, and its first stone is (b).
- **Nested sampling** (Skilling 2006) is the method designed for such transitions, and it needs no search. Its cost
  scales as n_live·H, with H = KL(posterior ‖ prior), which grows by about ln(a/σ) per dimension. Kroupa & Handley (2026)
  show that on aliased time-delay likelihoods it needs many live points to avoid spurious modes. Its role here is
  measured in §6.4. Unseeded it stays on the floor plateau, as tempering does; seeded by repartitioning onto q it is a
  valid but costlier duplicate of (b). It is kept as an optional cross-check (BlackJAX is a dependency already), not as
  the estimator.
- **A grid** survives only as the test oracle on toys.

### 6.9 The search must be scale-free (design pass 2026-11-03, problem A)

**Symptom.** On the §14 population (`orbit_population_20261102`, 187 systems), 38 of 60 systems with ln s > 1 fail,
against 35 of 127 below. Every large-s failure has no mode in any cell. Several large-s systems counted as passes are
floor fits with no mode: p̂(N) is the prior, so "N in the 99 % set" holds trivially. The single miss in 114 therefore
partly measures this.

**Where the search assumes s = 1.** The target (§1.1) carries a global inlier scale ln s_g in the chart (x[9:]). The
floor b_k integrates the night's own scale (Laplace at ŝ_k = √(χ²_k/n_k), clipped to [1/s_max, s_max], with its Occam
term). The search, however, works at ln s = 0 throughout:

- `_chart_from` sets w_s = 0.
- `_labels` compares ℓ_k at that s against b_k.
- `_scores` and `_prune` score at it.
- `_refine` starts Newton there.
- `fit_jitter` seeds from best-peak profiles at s = 1.

At s = 1 a night whose quoted errors are s times too small sits below its own floor by

  ℓ_k(x, ŝ) − ℓ_k(x, 1) = n_k·g(ŝ),  g(s) = ½(s² − 1) − ln s ≥ 0,

whatever the orbit (rescalable summaries, jitter 0, where the peak term is ℓ_k(x, 1) − n_k ln s − ½χ²_k(s⁻² − 1)). A
night is labelled inlier only if its height over the floor, Δ_k = ℓ_k^max(ŝ) − b_k, exceeds n_k·g(ŝ). On the
population, n_k = 40:

- at ln s = 1, n_k·g ≈ 88 nats well above the heights Δ_k of a few tens of nats, so every label is outlier;
- at ln s = −1, n_k·g ≈ 23 nats, so systems with s < 1 fail too, less often.

With every label an outlier, the labelled objective is constant, the C-steps do not move, and `_refine` starts where the
label-summed gradient in w_s is weighted by e^{ℓ−b} ≈ 0. No mode is returned, q is the prior, and the sentinel cannot
fire (§6.1). Only the IS ESS can flag the cell, and on a flat floor every weight is equal, so ESS = n_IS.

**Derivation of a scale-free search.** The target is equivariant. Hold the data fixed and quote every σ as cσ. Then
ℓ_k^new(x, s) = ℓ_k^old(x, c·s) exactly, which is a translation in ln s. The same holds for each night's own scale
inside b_k, and under (iii′) for the inflation integral, which is relative to s. The prior on ln s is uniform (§1.1),
so the target is unchanged once ln s is integrated, provided the translated support stays inside ±ln s_max. That
proviso applies to the inlier s_g and to every night's own ŝ_k, not just the true s. Then p(N | d) is unchanged. A search inherits that equivariance only if its labels, scores and starts depend on
s·σ, never on σ alone. The labelled objective of §6.1 is therefore maximized over (labels, x, ln s) jointly:

  F(x, s, I) = Σ_{k∈I} ℓ_k(x, s) + Σ_{k∉I} o_k(x, s) + ln π(s),

where o_k is the outlier term. Under candidate (ii) of §1.1, which the code implements, o_k = b_k is a constant. Under
(iii′), o_k(x, s) moves with s. This is block coordinate ascent, the classification-EM scheme of Celeux & Govaert
(1992), with the scale as a parameter of the classification step (Rousseeuw & Van Driessen's C-steps are its trimming
special case). Each block step does not decrease F, so the argument of §6.1 for finite termination is unchanged:

1. **Labels and scale jointly, given x:**

     (I, s) ← argmax_{I, s} F = argmax_s [Σ_k max(ℓ_k(x, s), o_k(x, s)) + ln π(s)],

   with each night inlier iff ℓ_k > o_k at that s. The maximization is 1-D in ln s over the prior range
   [−ln s_max, ln s_max]. It uses a grid at spacing h = 1/√n_tot (n_tot the total residual count), then Newton in ln s
   within the winning label set. Near its optimum F has curvature about −2n_I in ln s, so the grid's loss is at most
   n_I·(h/2)² ≤ ¼ nat. Under (ii), rescalable data and jitter 0, the Newton step has the closed form
   ŝ_g² = Σ_{k∈I}(χ²_gk + q_gk)/Σ_{k∈I} n_gk per inlier group g, with q_gk = d_kᵀH_gk d_k the peak's Laplace quadratic,
   clipped to [1/s_max, s_max]. Under (iii′), or with jitter, it is a 1-D Newton over all nights, since the outliers'
   o_k pull s too.
2. **Core given (labels, s):** the WLS/VarPro Newton of §6.1. A global scale cancels in the WLS solve, so only the labels
   and the Laplace terms see it.

**Why the labels and the scale move together.** A scale step at fixed labels (a pooled ŝ, then relabel) is not robust.
Suppose one outlier with (χ² + q)/n = r·s² is labelled inlier. It raises the pooled ŝ² by about (r − 1)/|I|. Each true
inlier then loses n_k·g(ŝ/s), and once that exceeds Δ_k the relabel drops the inliers and keeps the outlier. With four
40-residual inliers and r = 10, the loss is about 35 nats per inlier. F still rises, but to the wrong mode. The joint
step cannot do this: each night enters as max(ℓ_k, o_k), so an outlier's contribution is capped at its floor whatever s
is, and the maximum over s is set by the nights that fit. No start labels are needed either. A minimal set fits its own
nights exactly and carries no information on s, but step 1 needs only x.

**Coverage argument, carried over.** Replace σ by sσ throughout the argument of §6.1. The set spanning I's outer nights
still brackets the needle, since the needle's width in ρ is set by sσ/a and the grid's coherence by the span, which is
independent of both. An interpolated inlier then sits within about sσ of the orbit, below its threshold sσ√(2Δ_k).
At the true s, the true inliers contribute Σ_{k∈I}ℓ_k(x, s), which exceeds the all-floor value by Σ_{k∈I}Δ_k minus their
interpolation cost. So the joint maximum over s is at the true s (to within 1/√(2n_I) in ln s), unless a different
label set fits better, which would make it the larger mode. The first full C-step labels I or a superset inlier. The
argument is free of σ/a and of s.

**What must change (and nothing else).** `_labels`, `_cstep_chunk`, `_scores`, `_prune`, `_refine` and `fit_jitter`
all use the profiled ŝ. Non-rescalable data (gains or closure offsets) keep s pinned at 1 on both sides, as now (§1.1).
The scale step adds one pass over the inlier nights' χ² per C-step, which is negligible.

**Toy check.** `scalefix.py` is kept with the pass's session notes, not in the repository. It patches `_labels`,
`_refine` and `_scores` in-process. Each call runs step 1 (the joint max of Σ_k max(ℓ_k, b_k) over ln s) on an
81-node grid, on replayed population systems, with no src change:

| system (seed:index) | K | outliers | true ln s | base: modes, p̂(true), failed cells | scale-profiled: mode ln s, p̂(true), failed cells, ESS (true cell) |
|---|---|---|---|---|---|
| 8:9 | 5 | 0 | 1.88 | none, 0.18 (99 % set 6 of 7 cells), 1 | 1.93, 1.00 (set 1), 0, 197 |
| 12:3 | 16 | 0 | 2.01 | none, 2e-4 (**truth outside the 99 % set**), 2 | 2.06, 1.00, 0, 239 |
| 4:8 | 16 | 0 | 1.74 | none, 0.02 (set 14 of 15), 4 | 1.80, 1.00, 1, **1** |
| 12:1 | 10 | 10 (all) | 1.92 | none, 0.15 (set 12 of 12), 1 | none, 0.18 (set 12 of 12), 0, 47 |
| 16:2 | 9 | 0 | −1.11 | −1.07, 1.00, 2 | −1.07, 1.00, 0, 1013 |
| 1:4 | 10 | 1 | 2.17 | 2.14 (2 of 9 inliers), 9e-5 (**truth outside the 99 % set**), 1 | 2.22 (9 of 9 inliers), 1.00, 1, 288 |
| 14:4 | 8 | 1 | 1.83 | none, 0.31 (set 4 of 5), 1 | 1.86 (7 of 7 inliers), 1.00, 1, **2** |

The profiled search finds the mode at the true scale (to 0.05 in ln s) in every system that has inliers. In the two
systems with an outlier night it recovers exactly the true inlier set. The ln Ẑ of the true cell rises by 40–208 nats
over the base fit. The all-outlier system correctly stays at the prior. Two things remain:

- 4:8 and 14:4 have ESS 1–2 in the true cell, and one failed cell remains in each of 4:8, 1:4 and 14:4. The mode is
  right, but at n_I = 280–640 its t₅ component is too narrow or misplaced for the draws. This is the proposal problem
  of §6.10, not the search.
- The toy profiles s on an 81-node grid inside every call, so it ran 2–6× slower. The closed-form step above costs one
  pass over χ².

**gl229b_blind is the same mechanism, not a closure-phase-summary defect.** Its summaries are rescalable (no gains, no
closure offsets). The nights' closure-phase χ²/n is 3.6–10 with n = 238–3087, so at s = 1 each night sits ≈ 10³ nats
below its floor. Every cell returns ln Ẑ = −11826.6989 (equal to 12 digits, the all-floor value), with full ESS and no
mode. It is unflagged for the reason above: the sentinel needs a mode, and ESS is maximal on a flat floor. omi_leo works
because its data carry gains, so s is pinned at 1 on both sides and its mode has 9 of 9 inliers.

**A silent-failure flag.** A system with no mode in any cell and every ln Ẑ_N within its error of the all-floor value
K·b is reported as `no-mode` (information, not failure). Its p̂(N) is the prior, which is the correct answer under the
model only when the data have no orbit. It is therefore routed to the §8 null test before any p̂(N) is reported. Under
the scale-free search this flag should appear only on data without a companion, and §14 asserts that (assertion 7).

**Side issue (not in this pass).** Non-rescalable data with χ²/n ≈ 2 (omi_leo's nights: 1.5–2.3) are overconfident at
s = 1. That needs a scale model for data with gains (refit, §5 Stage 2b note), and is a follow-up.

### 6.10 Two-inlier ridge modes are mass to compute, not cells to bound away (problem B)

**What the failures are.** Classifying the population's non-A failures by their fit tables gives three kinds:

1. **Genuinely weak systems** (2–3 inliers left after the outliers): modes with exactly m inliers in every cell, ESS 2–7,
   Laplace ≈ IS. These are the true B.
2. **The sentinel on an inflated Laplace**: `cell-at-floor` fires on a 2–3-inlier mode whose Laplace lies 12–47 nats
   above the cell's IS estimate.
3. **Low ESS in the true cell** (one system, ESS 8.4).

**Are m-inlier modes real?** Yes. Under the model they are a term of the cell's integrand and carry mass of the same
order as the floor. For nights i, j labelled inlier and the rest at the floor, the mode's mass relative to the all-floor
term is the ratio of the pair's marginal likelihood under the orbit prior to its density under π_out. Averaged over the
pair's data drawn from π_out (the floor's own generative model) and summed over cells,

  E_{π_out}[ Σ_N p_orb,N(d_ij) / p_out(d_ij) ] = Σ_N π(N) = 1,

so the expected ratio per cell is the cell's prior mass. The ratio is heavy-tailed: its mean is 1 and its median is
typically ≪ 1. So this does not say that every such mode matters on given data. It says that no data-independent rule
can drop them: the data-dependent bound below may. Raising m_min above m would drop mass the model assigns (and would
be tuning to counts). **m_min stays m** (2 with velocities, 3
positions only): it is the identifiability count of §6.1, and the sentinel keeps it.

**Why they are ridges.** An m-inlier mode does not resolve the alias when its inliers lie in one season. The phase
between two inliers is 2π(N + ρ)Δ_I/T, so across one cell (Δρ = 1) it changes by only 2πΔ_I/T. For Δ_I ≪ T the
likelihood is nearly flat in ρ across the cell, and the mode is a ridge with its maximum on (or beyond) the cell edge.
The Laplace approximation in the logistic chart then fails in two ways:

- the ρ-variance is set by the chart's curvature, not the likelihood's, so Laplace inflates the mass (kind 2);
- a t₅ component centred at |w_P| ≫ 1 covers the ridge badly, so ESS collapses (kinds 1 and 3).

**Design: a ridge component (INLA-style).** A mode is a *ridge* if its Laplace ρ-marginal (in ρ, not w_P) puts more than
δ of its mass outside the cell, or if its set-Newton ended at |w_P| ≥ W (§6.11). Its proposal component is replaced by
a truncated conditional mixture, following the integrated nested Laplace approximation (Rue, Martino & Chopin 2009):

1. Start with nodes ρ_j across the cell, n_ρ = max(2, ⌈K_COH·Δ_I/T⌉ + 1), at the same coherence as the set grid (§6.1).
2. At each node, find the conditional mode of the other coordinates and its Laplace: log evidence
   h_j = ln L̂_j + ½ln|2πΣ_j|, with covariance Σ_j.
3. Bisect any interval whose |h_{j+1} − h_j| > 1 nat, and repeat until none remains. Phase coherence does not bound how
   much h changes across the cell. That change grows with the information, ∝ n_I·(2πΔ_I/T·a/sσ)², so two nodes can
   differ by tens of nats at large n_I. The 1-nat step is a tolerance: a log-linear interpolant through such nodes has
   a relative mass error of order (1 nat)²/8 per interval.
4. Set the component's ρ-density to exp of the piecewise-linear interpolant of h, which is piecewise exponential and
   normalizable in closed form over the cell. Given ρ, the other coordinates are t₅ about the conditional mode and
   covariance, interpolated linearly between nodes.
5. In the chart, ρ = ρ_lo + σ(w_P) across the cell, so log q_w = log q_ρ + ln σ(w_P) + ln(1 − σ(w_P)). The other
   coordinates keep their own chart Jacobians as for any component.

The component is a proper density with a closed-form log q, so the IS of §6.4 stays exact and unbiased. Its quadrature
sum (the closed-form integral of e^{h(ρ)} over the cell) replaces the Laplace mass wherever a mass is used: in the
mixture weights, in `_prune`, and in the sentinel's "W above the floor" test. Kind 2 disappears because no inflated
Laplace remains to trigger on.

**Negligibility by a bound, not by Laplace.** `position_cells` now calls a cell negligible when its Laplace upper estimate
is W below the best cell. A ridge's Laplace is not an upper bound, so this can call a real cell negligible or a
negligible one matters. The bound that holds is

  Z_N ≤ π(cell N)·sup_{x∈cell} L(x),

with sup L evaluated as the label-summed target at the best mode found. A cell is negligible iff
ln π(cell) + ln sup L < max_N' ln Ẑ_N' − W.

The best mode found is a *lower* bound on sup L. So the rule is conservative (more cells kept than by the exact
integral, since the bound omits the Occam factor) only *conditional on the search's coverage*. When the search misses
the cell's top mode, which the coverage contract allows with probability ≤ δ, a real cell can be called negligible.
Two consequences follow:

- A cell in which the search found no mode is never negligible by this rule. It runs its IS.
- A dropped real cell shows up as a miss of N in assertion 1, or as a non-uniform PIT in assertion 3 of §14, on the
  whole population, and in assertion 4 where the exact reference exists. The failure rate is the search's δ, not a
  new tolerance.

**Unreliable flag by relative error.** ESS < ESS_FAIL stays the trigger, because Kish's ESS is the available diagnostic.
A cell fails only if its estimate could change p̂(N) at δ, i.e. ESS < ESS_FAIL and the bound above admits p(N) ≥ δ.
The ridge component is what makes ESS ≥ ESS_FAIL in kinds 1 and 3. Retries and cascades stay excluded (§6.6).

**The population stays a prior draw.** The population draws ε ~ Beta(½,½) per system, which is the model's prior
(§1.1, §10). §14's spec says 0–2 outlier nights, but switching to that would make the population no longer a prior draw.
SBC (assertion 3) would then lose its uniformity, and the change would remove exactly the kind-1 systems above after
they failed. The population therefore keeps Beta(½,½). Results are reported by stratum: all nights outliers (B″, where
the correct answer is the prior), ≤ m inliers left, and the rest. §14 flags the mismatch with the spec's
"0–2 outlier nights" for Ben. Making the model's ε prior match the spec instead would be a model change, and
would need his decision.

### 6.11 Speed on two-season data (problem C)

**Measurement.** The K = 14 two-season case takes 877 s against a 5-min target. Replaying a K = 7 two-season system
(`edge.py`, with the same notes) covers the true cell and two neighbours, about 2000 candidates per cell. Splitting the
candidates by whether the set's nights lie in one season or two gives:

| set class | share | at the edge after 12 → 48 Newton steps | median \|w_P\|, 12 → 48 steps |
|---|---|---|---|
| one season | 18–45 % | 0.41 → 0.51, 0.84 → 0.95, 0.68 → 0.85 | 2.8 → 4.5, 7.8 → 13.3, 5.8 → 12.4 |
| two seasons | 55–82 % | 0.39 → 0.40, 0.46 → 0.47, 0.36 → 0.39 | stable (2.0–3.5) |

**Why Newton exhausts its steps, and why candidates run to the edge.** The cell edges sit at w_P = ±∞ in the logistic
chart. Two distinct cases put a candidate's maximum there:

- **One-season sets** are ridges (§6.10). The likelihood is nearly flat across the cell, so the maximum is on the
  boundary, and Newton follows it towards |w_P| → ∞. More steps take it further (median |w_P| 7.8 → 13.3) and never
  converge.
- **Two-season sets at the edge** are images of a mode of the neighbouring cell: the likelihood rises towards the shared
  edge. Their share at the edge is stable with more steps, because they stop at the first chart point where the
  gradient underflows.

Neither has a stationary point inside the cell, so neither meets the convergence test, and both use every step
(N_VARPRO, then N_NEWTON). About 35 % of the *interior* candidates also use all 12 VarPro steps, which is not explained
by either case; the first work item measures the gradient at stop.

**Fix 1: stop at the edge, and share the neighbour's mode.** A candidate is stopped once |w_P| ≥ W = ln(1/δ). Beyond that
point the logistic chart leaves less than a fraction δ of the cell's ρ-interval (1/(1 + e^{W}) < δ), so it is derived,
not tuned. A stopped two-season candidate whose ρ-gradient points outward is dropped here: the neighbour's own search
finds it as an interior mode, under the coverage argument of §6.1. That holds only when the neighbour exists and is
searched. At the first and last cells of the N grid, or when the neighbour is not searched (pruned, or negligible by
the §6.10 bound), the stopped candidate is kept here as a ridge component (§6.10) instead of being dropped. Otherwise
the mode would be found nowhere, and neither the sentinel nor `no-mode` would fire. The neighbour's mode enters this
cell's mixture as a component when its ρ-space (delta-method) Laplace marginal puts more than δ of its mass across the
shared edge. In its own logistic chart a Laplace never leaves the cell, so the test is made in ρ. This shares *modes* across cells, not
draws: each cell keeps its own n_IS draws and its own estimate, so §6.7's objection (a negligible cell sees only prior
draws and its ln Ẑ is skewed low) does not apply.

**Fix 2: search short-span sets once.** A set S is *short-span* if K_COH·Δ_S/T < 1, i.e. its residual changes by less
than a quarter cycle across one cell. Its likelihood depends on P only through the phase over Δ_S, smoothly across every
cell. It is searched once over the whole ν-domain (unbounded ln P), and its band modes become ridge components
(§6.10) in every cell they overlap. Each cell then runs only the long-span sets.

This is Ben's §6.7 idea restricted to the case where it pays. §6.7 measured a shared search over all sets, where each
set's grid spanned every cell and the shared search cost 1.8× the per-N searches. For a short-span set the whole-domain
grid has about K_COH·Δ_S·Δν_domain + 1 points against B·(K_COH·Δ_S/T + 1) per-N. The shared grid saves the
per-cell +1 and, more importantly, the B duplicate candidates and their C-steps.

**Fix 3: per-set grids, as specified.** §6.1 sizes each set's grid by its own span, n_ρ(S) = ⌈K_COH·Δ_S/T⌉ + 1. The code
sizes every set's grid by the widest set (n_grid = max(K_COH + 1, ⌈K_COH·span·ν_width⌉ + 1)). The design stands; the
code is to follow it. This matters less after fix 2, which removes most short sets from the per-cell search.

**Fix 4: triplets only where a pair does not cover (the velocity criterion).** The proposal is to drop triplets when the
velocities are informative. Its coverage condition is the interpolation step of the §6.1 argument, which can be made
quantitative per span pair. For a pair S = (i, j) solved at a ζ start, night k between them is predicted with covariance

  C_k = G_k F_S⁻¹ G_kᵀ,

where G_k is the Jacobian of night k's state (p, v) in the parameters and F_S the pair's Fisher information. The
C-step compares ℓ_k at the pair's prediction p̂_k with ℓ_k^max, which is taken at night k's own peak μ_k, not at the
true state. The offset is therefore p̂_k − μ_k = e_pred − e_noise ~ N(0, C_k + Σ_k), with Σ_k the night's summary
covariance at the profiled scale. It costs ½eᵀΣ_k⁻¹e in ℓ_k. With λ_k = λ_max(Σ_k⁻¹C_k),

  eᵀΣ_k⁻¹e ≤ (1 + λ_k)·|z|²,  |z|² ~ χ²_4,

so P(½eᵀΣ_k⁻¹e > ½(1 + λ_k)·χ²_4(1 − δ)) ≤ δ. The pair then labels night k inlier at its first C-step with probability
≥ 1 − δ when

  ½(1 + λ_k)·χ²_4(1 − δ) < Δ_k  for every night k between i and j,

with Δ_k = ℓ_k^max(ŝ_k) − b_k the night's height over its floor, known from its summary before the search. The night's
own noise term ½χ²_4(1 − δ) = 9.2 nats at δ = 10⁻³ dominates when the velocities are informative (λ_k ≪ 1). A night
lower than that over its floor keeps its pair's triplet fills whatever the velocities. For a position-only night the
state has 2 dimensions, and χ²_4 is conservative. If every
interior night passes, the pair alone covers that span, and its triplet fills add nothing to coverage. Otherwise the
pair keeps its stratified triplet fills (§6.1). The blanket proposal is the special case in which every pair passes.
The criterion keeps coverage to first order, since it is the coverage argument's own condition. It rests on two
linearizations: the pair's solution must be in its Gaussian regime at the needle, and the ζ start must be close enough
for G_k to hold. Eccentric orbits can break either, so the gate is the coverage test (§6.5, §14 assertion 5) on the
reduced set list, not the derivation. A pair that fails the criterion keeps its fills, so a wrong λ_k can only cost
time when it is too large. When it is too small, it costs coverage, and the test catches that. Ben accepted the
blanket proposal in principle (2026-11-03), and this criterion is its derived form. It is scale-free:
C_k ∝ s² and Σ_k ∝ s², so λ_k does not depend on s. It costs one small solve per span pair.

**What is not shared.** Shared IS draws across cells stay rejected (§6.7). The N-independent work that can be shared
exactly is the prior factor, the anchors' normal equations (§6.7), the floors b_k (already per night), and the
short-span band modes (fix 2).

**Expected effect.** This is an estimate, to be measured by the work items. On the K = 14 two-season case, triplets are
364 of 455 sets. Fix 4 removes those whose pairs pass. Fix 1 cuts the edge candidates' Newton cost (30–40 % of
candidates, each at the step cap) to the steps needed to reach |w_P| = W. Fix 2 moves the 18–45 % short-span
candidates out of the per-cell loop. Together these are a 3–10× reduction, which brings 877 s to the 5-min target. The
acceptance test is the measured time, not this estimate.


## 7. Pruning

The evidence of every cell comes from the search and IS, which are cheap at position level (§9). Pruning therefore saves
only the expensive parts: the visibility-level refinement and IS (leg 2), and the HMC posterior draws.

- A cell enters leg 2 iff its position-level mass p̂_pos(N) ≥ δ/B.
- It gets HMC draws iff p̂_vis(N) ≥ δ/B.

By construction the position-level mass dropped is < δ. The visibility-level mass of dropped cells can differ, so leg-2
pruning stays **opt-in until the population test (§14) measures the mean pruned posterior mass**: report
E[Σ_dropped p_vis(N)] with its standard error against δ. Pruning becomes the default only if the mean plus 3 SE ≤ δ.

## 8. The no-orbit verdict

An empirical false-alarm probability at α = 0.01 (orbit_dehack decision 4, cheap tier; `detection.DetectionMC`, #621). The
statistic is **T = max_n S_n − S_null**, where S_n is the seed-profile evidence of the best orbit seed in band n at position
level and S_null is the same quantity for the single-star model. Both are computed identically on the data and on
simulated single-star nulls. **Single-star position-level likelihood:** the night likelihood with no companion, L_k⁰
(§5, at visibility level), is an absolute density like the peaks' L_j and b_k. S_n is the seed-profile sum of
ln[(1 − ε) L_j + ε e^{b_k}] (peak j of the seed's orbit) and **S_null = Σ_k ln[(1 − ε) L_k⁰ + ε e^{b_k}]**, ε-integrated as in §2;
orbit statistic and null share the same b_k. Under (iii′) the floor is e^{o_k} for the orbit and the single-star analogue
o_k⁰ = ln ∫₁^{s_max} L_k⁰(s_g t) dt/(t ln s_max) for the null.

**FAP** = (1 + #{T_null ≥ T_obs}) / (1 + n_null) (Phipson & Smyth 2010). The null count is **raised step by step**: 100 → 200 →
400 → 800, to a cap of 1000 (= 10/α); stop when the 95 % interval of the FAP excludes α (verdict: FAP < α or ≥ α), or at
the cap (verdict: **inconclusive**, reported as such). Each null is a simulated single-star data set, summarised once per
night (T_summ), with its own b_k (nulls change the data, so b_k is **per null**, K·T_b each) and a seed profile
(B·n_seed·K·J·c_pos, §9). The profile sums over all B bands, so the null cost scales with B. The visibility leg informs
the posterior, not the verdict, so that the null calibrates the statistic actually thresholded.

## 9. Cost formula

Per system, with every symbol a measured quantity (Stage 0 `run.json` counters, Stage 1 timings). Costs are in
likelihood evaluations c (no gradient) or gradients g ≈ 3c:

  T_sys = T_compile + T_epochs + T_summ + K·T_b
          + B·(T_search + N_IS·c_pos)
          + B_vis·(n_modes·T_refine,vis + N_IS·c_vis)
          + B_post·N_POST·N_HMC·(L + 1)·g_vis
          + n_null·T_null

  T_search = Σ_S Σ_ζ n_P(S)·m·c_pos + n_cand·N_C·T_Newton,pos + n_modes·T_Newton,pos,  with n_P(S) = ⌈K_COH·Δ_S/T⌉ + 1

  T_null = T_summ + K·T_b + B·n_seed·K·J·c_pos.

| symbol | meaning | source |
|---|---|---|
| T_compile, T_epochs, T_summ, T_b, K | as before | Stage 0, 2b, #618 |
| B | cells, T/P_cut − T/P_max | data |
| sets S, m | minimal sets (all C(K, m) × peak choices up to N_SET_MAX, else all span pairs × stratified interior fills) | §6.1 |
| n_cand, n_modes | candidates after the grid; distinct modes | measured |
| N_IS | IS draws per cell (4000) | budget |
| B_vis | cells entering leg 2 (B unless pruning, §7) | data |
| B_post | cells with p̂ ≥ δ/B that get HMC draws | data, usually 1–3 |
| N_POST, N_HMC, L | resample-move draws, moves per draw, leapfrog steps | budget, measured |
| c_pos, c_vis, g_vis | evaluation costs, position and visibility level | Stage 0, Stage 1 |
| n_seed, J, n_null | as §8 | Stage 3d |

**Measured on the toys:**

- T_search = 1.6e4–2.1e5 evaluations for K = 6–10 (§6.5), plus N_IS = 4e3.
- The cell is search-dominated. The search grows with the number of sets, C(K, 3) = 20 to 364 for K = 6 to 14.

**Two seasons, measured (K = 14, 7 + 7 nights a year apart, P ∈ [2.2, 60] d, 161 cells, float64, 2 threads, a laptop at
load ≈ 26, so wall times are inflated; compare only within the table).** 455 sets (91 pairs + 364 triplets) × 17 ζ
starts give ≈ 1.5 × 10⁴ candidates per cell, 30–40 % of which pass the χ² cut. Per cell, search ≈ 93 % of the time:

| stage | share of the search |
|---|---|
| grid | 0.4 % |
| variable-projection Newton on the sets (`_varpro_chunk`) | 63 % |
| first C-step pass, 4 steps on ≈ 4 × 10³ unique starts | 24 % |
| Laplace scores, per-mode refinement | 4 % + 5 % |

Importance sampling is 0.3 s per cell and an HMC resample-move 150 s per drawn cell (usually one to three cells). Cost is
flat in N: every cell has the same ν width 1/T, so the same grid, sets and candidates. The kernels compile once (≈ 90
compiles, 60–75 s, none after the first cell). In the set Newton, 98 % of candidates use all `N_VARPRO = 12` steps and
63 % end with |w_P| > 6, at the cell edge: the minimum of their residual lies outside the cell (a grid end counts as a
minimum, and a straddling needle needs it), so the steps are not wasted for coverage, and no cheaper stopping rule
applies. Cells can run in threads (`workers=2`; exact, bit-identical `ln Z` and `p(N)` to one worker on this system; K = 14: 1088 s to 877 s, K = 8: 79.7 s to 67.8 s, +0.3 GB; virgil#638). Tried and reverted: the gradient and Hessian of each Newton
step in one pass (about 10 % faster) moved the K = 3 prior-importance check of `test_the_position_evidence_of_a_tiny_case_matches_prior_importance`
by 1.3 nats (4 sigma), so the search is sensitive to floating-point reassociation there. Not done, because they change which
candidates are kept: dropping triplets where
velocities are informative (pairs suffice for the coverage argument, 5× fewer sets), and a pre-Newton cut on the grid
residual (it has no derived bound: a candidate far above the cut at a node can refine below it).

**A worked example** at gallery scale:

- Position level: B ≈ 250 cells × 2 × 10⁵ evaluations ≈ 5 × 10⁷.
- Visibility level, unpruned: 250 × (4000 + refinement) ≈ 10⁶ evaluations.
- HMC: a few cells × 1000 × 5 × ~20 ≈ 10⁵ gradients per cell.

The old tempered design needed 9 × 10⁸ position-level and 2 × 10⁸ visibility-level gradients.

**Laptop verdict (revised).**

- A single system is now feasible on the laptop. The position level is minutes at a few μs per evaluation. The
  visibility level is ~10³ s at c_vis ≈ 1 ms unpruned, and less with pruning.
- The §14 population of 200 systems has B ≈ 15 cells for one 30-d season at P ≥ 2 d. It is ≈ 200 × 15 × 2 × 10⁵ ≈
  6 × 10⁸ position-level evaluations, which is hours on 2 threads. It goes to OzSTAR as a stage; a 10-system smoke
  run stays local.
- Falsifiable: Stage 0/1 fill the table below, and any of c_pos, c_vis, n_cand or B changes the verdict.

| quantity | M4 4 threads | M4 10 threads | A100 | source |
|---|---|---|---|---|
| T_compile, T_epochs, c_pos, c_vis, g_vis, T_b, T_summ, T_null | _to fill_ | _to fill_ | _to fill_ | Stage 0 / 1 |
| T_search, n_cand, n_modes, ESS, B | _to fill_ | | | Stage 3b′ on the §14 population |

## 10. Constants

Every constant is **derived**, a **tolerance** with a meaning, or a declared **budget**. None was chosen by looking at vv
or gallery targets. Provisional budgets are fixed in the 3b′ PR from the §14 population before any vv
run; changing one afterwards is a recorded decision.

| constant | value | kind | basis |
|---|---|---|---|
| s_max | 10 | declared prior bound | inherited `_NUMERIC_S_MAX`; covers χ²/N ≤ 100 (v2 §2.2) |
| ε prior | Beta(½, ½) | derived | Jeffreys for a rate |
| P_cut | max(2Δt_min over nights merged at gap_days, p_floor) | derived (a declared prior; Ben, 2026-11-02) | §1.3 |
| P_night | dropped from the orbit-cells path (snapshot-era likelihood only) | n/a | §1.3 |
| motion_fraction | ½ | tolerance | P_night only, which the orbit-cells path no longer uses (§1.3); inherited |
| p_floor | 0.2 d | declared | inherited |
| P_max | 2π·a_hi·span/σ_min, one long-period band [span, P_max] | declared prior bound | orbit_dehack decision 13 (the inherited 4 × span is retired) |
| a_lo, a_hi | 1, 30 mas | declared prior bound | inherited default; to move to resolution units (decision 12) |
| e_max | 0.9 | declared prior bound | inherited default |
| flux floor lo | data's 2σ contrast curve | derived | dehack §2I |
| π_out | joint prior predictive π(p, v)π(f), tabulated from a 2²⁰ scrambled-Sobol sample, KDE (Scott bandwidth) in (ln|p|, ln|v|, φ) | derived | §2.2; b_k changes < 0.05 nats from 2²⁰ to 2²¹ (tolerance) |
| σ_J grid | type-II ML on a 1-D log grid, 0.1–10 × the median σ_p | budget | §5 |
| gap_days | 0.5 d | tolerance | night separation (inherited) |
| flux floor | 2σ of the data's contrast curve | derived | dehack §2I |
| lattice, cubature box widths | ±4σ of π_out and of the per-night posterior | tolerance | #618 |
| ε_lo | sin²(π/(4 n_GC)), n_GC = ⌈(K + 1)/2⌉ | derived | smallest Gauss–Chebyshev node (§5) |
| anchor quarter | first/last ¼ of span | budget (labels only) | §4.2 |
| pruning margin | mean + 3 SE ≤ δ | tolerance | §7 |
| FAP stepping | 100, 200, 400, 800; cap 1000 = 10/α | budget | §8 |
| δ | 10⁻³ | tolerance | reporting precision (dehack §2) |
| W = ln(1/δ) | 6.9 | derived | from δ |
| peak keep rule | (1 − ε_lo)L_j ≥ δ·ε_lo·e^{b_k} | derived | §5 |
| season gap | P_cut | derived | §4.2 |
| edges | x = N ± ½ (`make_cells` default) | labels only | §4.4; the pilot and valley edges are retired |
| z chart | auxiliary Gaussian direction in ℝ⁴ plus a logistic ln a coordinate | chart, no statistical meaning | §3.5 |
| ζ chart | ζ = √e_max·y/√(1 + \|y\|²) | chart, no statistical meaning | §3.5 |
| m (minimal set) | 2 with velocities, 3 positions only | derived | equations ≥ unknowns at fixed ζ (§6.1) |
| K_COH | 4 | budget (derived > 2) | ln P grid points per cycle of a set's span; > 2 brackets every minimum (§6.1) |
| ζ starts | e ∈ {0, 0.3, 0.6} × 8 anchor phases (17) | budget | coverage tested on eccentric toys, not derived (§6.1, §14) |
| N_SET_MAX | 2000 | budget | enumerate all sets below; above it, all span pairs with stratified interior fills (§6.1) |
| ε_hi | 0.3 | declared | outlier fraction in the RANSAC count n = ln δ/ln(1 − (1 − ε_hi)^m) |
| N_C | 10 | budget | C-step cap; labels repeat within 2–3 steps on the toys |
| inlier rule | ℓ_k(x, ŝ) > o_k at the jointly profiled ŝ | derived | the larger component (C-step labels; §6.9) |
| scale step | joint (labels, ln s) max of Σ_k max(ℓ_k, o_k): grid at h = 1/√n_tot, then Newton (closed form ŝ_g² = Σ_I(χ² + q)/Σ_I n under (ii)) | derived; h a tolerance (≤ ¼ nat) | CEM classification step (Celeux & Govaert 1992; §6.9) |
| edge stop | \|w_P\| ≥ W | derived | the logistic chart leaves < δ of the cell beyond it (§6.11) |
| short-span set | K_COH·Δ_S/T < 1 | derived | from K_COH: under a quarter cycle across a cell (§6.11) |
| ridge mode | Laplace ρ-marginal > δ outside the cell, or set-Newton stopped at the edge | derived | from δ and the edge stop (§6.10) |
| ridge nodes | start n_ρ = max(2, ⌈K_COH·Δ_I/T⌉ + 1); bisect until \|Δh\| ≤ 1 nat between nodes | derived start, tolerance step | the set grid's coherence; log-linear interpolation error (§6.10) |
| triplet criterion | keep a span pair's fills iff some interior night has ½(1 + λ_max(Σ_k⁻¹C_k))·χ²_4(1 − δ) ≥ Δ_k | derived | the §6.1 interpolation step at level δ (§6.11) |
| negligible cell | ln π(cell) + ln sup L < max ln Ẑ − W | derived | Z_N ≤ π(cell)·sup L (§6.10) |
| distinct-mode radius | Mahalanobis 1 under both covariances | tolerance | §6.1; the one-sided test cost 62 nats |
| VarPro ridge | 1/s_w², s_w² = ⟨a²⟩/3 | derived | §6.1; the per-component TI′ variance of the prior |
| α (IS) | 0.1 | inherited | `aliases`; defensive share, w ≤ L_max/α (Hesterberg 1995) |
| t degrees of freedom | 5 | inherited | `aliases._Proposal` |
| N_IS | 4000 | budget | sd(ln Ẑ) ≈ 0.012 measured (§6.4) |
| ESS_target | 400 | derived | sd(ln Ẑ) ≈ ESS^{−1/2} ≤ 0.05 (flag only) |
| sentinel | E_w[n_inliers] < m while the best mode has ≥ m inliers and its mass (ridge quadrature for a ridge, §6.10) lies W above K·b | derived | §6.1 |
| ESS_FAIL | 10 | inherited | `mode_search`; a cell fails only when also not negligible by the §6.10 bound |
| HMC acceptance target | 0.8 | tolerance | dual averaging (Hoffman & Gelman 2014) |
| HMC warm-up, moves per draw, L cap | 150, 5, 64 | budget | §6.3 |
| step jitter | ±20 % | budget | Neal 2011 |
| N_POST | 1000 | budget | resample-move draws per cell with p̂ ≥ δ/B |
| α (FAP) | 0.01 | declared | dehack decision 4 (§8) |

P_cut: P_night is dropped (Ben, 2026-11-02). It had set P_cut in 81 % of random epoch patterns
(`scripts/p_cut_random_epochs.py`) and in all 50 draws of #632's ensemble. P_cut is now max(2Δt_min, p_floor), and
within-night curvature is modelled by the exact-projection summaries (§1.3).

## 11. Literature and provenance

Owen & Zhou 2000 is withdrawn: its defensive bound is vacuous here (§6.2). Entries are in `references/provenance.toml`; verified against ADS (bibcodes) or Crossref (the statistics journals
ADS does not index).

| source | role here | verified |
|---|---|---|
| Luger, Foreman-Mackey & Hogg 2017, RNAAS 1, 7 | linear parameters given the core (§1.4; the VarPro solve of §6.1) | ADS 2017RNAAS...1....7L (existing entry) |
| Price-Whelan, Hogg & Foreman-Mackey 2017, ApJ 837, 20; Price-Whelan et al. 2020, ApJ 895, 2 | The Joker: collapsed linear parameters (the superseded J path; §6.3) | ADS 2017ApJ...837...20P (existing), 2020ApJ...895....2P |
| Lucy 2014, A&A 563, A126 | mass estimates for incomplete visual-binary orbits (prior art on short arcs; not the source of the alias structure) | ADS 2014A&A...563A.126L (existing) |
| Del Moral, Doucet & Jasra 2006, JRSSB 68, 411 | SMC samplers, Ẑ (superseded tempered design) | Crossref 10.1111/j.1467-9868.2006.00553.x |
| Jasra, Stephens, Doucet & Tsagaris 2011, Scand. J. Stat. 38, 1 | ESS-adaptive tempering (superseded) | Crossref 10.1111/j.1467-9469.2010.00723.x |
| Zhou, Johansen & Aston 2016, JCGS 25, 701 | conditional ESS rule (superseded) | Crossref 10.1080/10618600.2015.1060885 |
| Buchholz, Chopin & Jacob 2021, Bayesian Anal. 16, 745 | fixed-L HMC within SMC (superseded) | Crossref 10.1214/20-ba1222, arXiv:1808.07730 |
| Mengersen & Tweedie 1996, Ann. Stat. 24, 101 | independence MH is uniformly ergodic iff the weight is bounded (why the Luger move stuck face-on; superseded) | Crossref 10.1214/aos/1033066201 |
| Kustaanheimo & Stiefel 1965, J. reine angew. Math. 218, 204; Levi-Civita 1920, Acta Math. 42, 99 | regularizing maps (z, ζ are analogues; §3.2) | Crossref 10.1515/crll.1965.218.204; 10.1007/BF02404416 |
| Blunt et al. 2017, AJ 153, 229 | OFTI: rejection sampling of short arcs (prior art; prior-proposal IS fails at the needle, §6) | ADS 2017AJ....153..229B |
| Phipson & Smyth 2010, Stat. Appl. Genet. Mol. Biol. 9, 39 | permutation p-values (1 + #)/(1 + n) (§8) | Crossref 10.2202/1544-6115.1585 |
| VanderPlas 2018, ApJS 236, 16 | Nyquist limit does not apply to irregular sampling (§1.3) | ADS 2018ApJS..236...16V (existing) |
| Shoemake 1992, Graphics Gems III, 124 | unit quaternions, SO(3) Haar measure (§3.2) | Crossref 10.1016/b978-0-08-050755-2.50036-1 |
| Thompson et al. 2023, AJ 166, 164; Surjanovic et al. 2023, arXiv:2308.09769 | Octofitter and Pigeons: parallel-tempering prior art (§6.8) | ADS 2023AJ....166..164T (existing); 2023arXiv230809769S |
| Genz & Malik 1980; Blake & Zisserman 1987 | b_k cubature (#618); GNC (superseded) | in #618 / mixture v2 |
| Rasmussen & Williams 2006, A.9 | Woodbury and determinant lemma (Stage 1) | textbook |
| Skilling 2006, Bayesian Anal. 1, 833 | nested sampling; thermodynamic paths fail at first-order transitions (§6, candidate (d) of §6.4) | Crossref 10.1214/06-BA127 |
| Fischler & Bolles 1981, Commun. ACM 24, 381 | RANSAC: minimal sets, and the count of random sets (§6.1) | Crossref 10.1145/358669.358692 |
| Rousseeuw 1984, JASA 79, 871 | least median of squares; elemental sets (§6.1) | Crossref 10.1080/01621459.1984.10477105 |
| Rousseeuw & Van Driessen 1999, Technometrics 41, 212 | concentration steps, monotone in the objective (§6.1) | Crossref 10.1080/00401706.1999.10485670 |
| Celeux & Govaert 1992, Comput. Stat. Data Anal. 14, 315 | classification EM: block ascent over labels and parameters, here with the inlier scale as an M-step (§6.9) | Crossref 10.1016/0167-9473(92)90042-E |
| Rue, Martino & Chopin 2009, JRSS B 71, 319 | INLA: quadrature in a low-dimensional parameter with conditional Laplace, here the ridge component in ρ (§6.10) | Crossref 10.1111/j.1467-9868.2008.00700.x |
| Golub & Pereyra 1973, SIAM J. Numer. Anal. 10, 413 | variable projection for the linear TI′ (§6.1) | Crossref 10.1137/0710036 |
| Hesterberg 1995, Technometrics 37, 185 | defensive mixture importance sampling, bounded weights (§6.4, the estimator) | Crossref 10.1080/00401706.1995.10484303 |
| Meng & Wong 1996, Stat. Sinica 6, 831; Gronau et al. 2017, J. Math. Psychol. 81, 80 | bridge sampling, candidate (a) (§6.4) | publisher page; Crossref 10.1016/j.jmp.2017.09.005 |
| Veach & Guibas 1995, SIGGRAPH '95, 419 | balance heuristic: the per-mode partition of unity in (a) (§6.4) | Crossref 10.1145/218380.218498 |
| Tierney & Kadane 1986, JASA 81, 82 | Laplace evidence, candidate (c) and the mixture weights (§6.4) | Crossref 10.1080/01621459.1986.10478240 |
| Gilks & Berzuini 2001, JRSSB 63, 127 | resample-move (§6.3) | Crossref 10.1111/1467-9868.00280 |
| Neal 2011, Handbook of MCMC ch. 5 | HMC, mass matrix, step jitter (§6.3) | arXiv:1206.1901 |
| Hoffman & Gelman 2014, JMLR 15, 1593 | dual averaging of the step size (§6.3) | existing entry |
| Yallup, Kroupa & Handley 2026 | BlackJAX nested slice sampling, measured as (d) (§6.4) | arXiv:2601.23252 |
| Albert 2020 | jaxns, the alternative NS implementation (not measured; §6.4) | arXiv:2012.15286 |
| Chen, Feroz & Hobson 2023, Bayesian Anal. 18, 695 | posterior repartitioning: NS seeded by q (§6.4) | Crossref 10.1214/22-BA1323 |
| Fowlie, Handley & Su 2021, MNRAS 503, 1199 | NS on likelihood plateaus: the all-outlier floor (§6.4) | Crossref 10.1093/mnras/stab590 |
| Kroupa & Handley 2026 | NS on aliased (time-delay) likelihoods needs many live points (§6.4, §6.8) | arXiv:2602.22307 |
| McDougall, Pope & Davis 2026, PASA 43 | LITMUS `hessian_scan`: a scan over an aliased variable with Laplace per slice; prior art for §6.7 | Crossref 10.1017/pasa.2026.10149 |
| Talts et al. 2018 | simulation-based calibration (§14) | arXiv:1804.06788 |

**Ben's suggestions (labelled as his, no source):** reducing the problem to a low-dimensional search and propagating across N (§6.7, measured and partly adopted: the anchor and prior terms are exact; shared draws were rejected); HMC within cells (§6.3); the population test (§14).

**Derived departures (no source):** the z map (§3.2), the ζ map (§3.3), the anchor TI′ rotation (§3.4), the sentinel and the symmetric distinct-mode test (§6.1), the edge stop, short-span band search and triplet criterion (§6.11), the contract
for b_k with velocity (§2) and the peak keep rule (§5). Each is flagged *derived* in the methods register by its stage and
checked by a vv-independent test (the check script, the prior-recovery tests, the #618 oracle in 5-D).

**Stage 2b status (`virgil.night_summaries`, provisional): *derived*.** Per-peak whole-night linear-motion summaries at
visibility level (§5) and the peak keep rule are not published practice; both are derived here. Tests pinning them
(`tests/test_night_summaries.py`): unit pulls over seeds, the velocity covariance ∝ 1/span² (exact under time stretching)
and ∝ σ², the keep-rule threshold at K = 1 and K = 3, merge-to-best, no peak reported on a bound, the null profiled like the
peaks. Per-frame jitter σ_J enters as σ_J²/N at the night position and σ_J²/Σ(t_j − t̄)² in the velocity (§5: σ_J is added to
every per-exposure σ_p). Closed-form error rescaling is exact only without gains and closure offsets; with them
`NightPeak` refuses it (refit instead). vv check: with 3b.

## 12. G1 criteria change (separate PR; awaiting Ben's sign-off; the gate is **not** changed by this PR)

`design/orbit_gate_042.md` §3.3 draws stratum B truths with *P* log-uniform on [1 d, 3 × span]. Under the declared prior
of §1.3, truths with *P* < P_cut = max(2Δt_min, p_floor) fall outside the default prior. The change is in
[#625](https://github.com/benjaminpope/virgil/pull/625), not here: the draw is unchanged, truths with *P* < P_cut are
excluded and counted (stratum B′, reported, not scored; excluding rather than redrawing keeps the generator stream and
the before/after pairs identical), the baseline stays reported under its own criteria commit, and the rule is
`aliases.period_cut`, shared by the engine and the vv harness. M4's `init=` oracle is unchanged. Stratum A (explicit `p_range`) is unaffected. The engine's existing redraw rule used
P_night at the truth's a. With P_night dropped (§1.3), that rule changes to the 2Δt_min and p_floor terms, so far fewer
truths fall in B′ than under the old rule.

## 13. Stage consequences

| stage | takes from this note |
|---|---|
| 2a | §3 (maps, chart, TI′ density, invariants, Campbell conversion); prior-recovery tests; register z, ζ as derived |
| 2b | §5 (night summaries, keep rule, v variance, σ_J, (iii′) analogue); the (k, h) Kepler solve (§3.3) |
| 3a | **retired for orbits**: the generic search and IS module of §6.1–6.4 and §6.6 replaces it (see #623 below) |
| 3b′ | §4, §5 position-level target, §6.1–6.6 at position level, §7, the §14 population test |
| 3c | §2 contract (5-D b_k, joint π_out), §6.2 refine-and-IS at visibility level, §1.1 noise; #618's six tests as acceptance at both pinned precisions |
| 3d | §8 (T statistic, stepwise FAP) |
| 4 | the vv criteria file (committed before any run; awaiting Ben's sign-off), §7 measurement, §9 table |

### 13.1 Changes to the open PRs (flag for Ben)

- **#623 (generic tempered SMC on BlackJAX).** The orbit legs no longer use it, and the composite-mutation hook it was
  to grow is not needed. It can land as a generic tool, or close. Ben decides. Nothing in this note depends on it. BlackJAX ≥ 1.7.1 stays a
  core dependency, which #629 already declares: it supplies the HMC kernel and the dual-averaging step-size adaptation of
  §6.3, so the resample-move does not fork a sampler.
- **#629 (cells and position leg).** The brief in the coordinating session's notes is the implementation plan.
  - **Kept:** `PeriodDomain`, `period_domain`, `choose_anchors`, `Cell`, `make_cells`, `OutlierPrior` and its functions,
    `_Catalogue`, `_log_mixture` and its helpers, `_Space` (minus `with_ti`), `fit_jitter`.
  - **Deleted:** the tempered `position_leg` body, `_luger_move`, `valley_edges`, the pilot, and the slow tests of the
    tempered leg.
  - **Added:** a generic `mode_search` module (minimal-set search, C-steps, distinct modes, defensive mixture with the
    chart's 16 symmetry images, IS evidence, sentinel, HMC resample-move) and its orbit adapter. The fast tests port the
    toys' exact references, and §14 replaces the slow tests.
  - **Stacking:** a new branch on #629's head.
- **#630 (visibility leg).**
  - **Withdrawn:** the tempered bridge `visibility_leg`.
  - **Kept:** `OutlierTable`, `outlier_table`, `_genz_malik`, `_cubature`, `outlier_exact`, `swap_label`,
    `report_invariants` and `_VisSpace`.
  - **Replaced:** `visibility_leg` and `fit_cells` become a per-mode visibility-level Newton refine, then the same
    defensive IS (§6.2). Work on it stays paused until 3b′ lands.
- **#632 (Argon's simulated ensemble harness).** It is reused by §14: `draw_population`, `_summaries` and `run_one`,
  with `run_one` calling the new `position_cells`. Its first run found 52 % of the true periods outside the default
  period domain. That is the open domain question of §1.3. It is not a property of the search.

## 14. The population test (replaces Stage 3b's slow tests)

This is Ben's decision 5: a simulated population, the truths drawn from the prior, and the pipeline asserted against
calibration, not against one hand-picked case. Toys and simulations only. No gallery or vv target enters it, and no
constant is tuned to it after a failure.

**The population (Ben's simulated-ensemble spec, 2026-11-02):**

- Orbits: P log-uniform on 2–60 d; e uniform on 0–0.8; isotropic orientation; a log-uniform on [a_lo, a_hi].
- Epochs: 5–20 nights, uniform. Either one season of about 30 d, or two seasons a year apart. Nights are clustered
  within each season.
- Noise: σ/a log-uniform on 10⁻⁴–10⁻². 0–2 outlier nights, each with its summary drawn from π_out. **Flag for Ben
  (2026-11-03):** the population script instead draws ε ~ Beta(½, ½) per system, which is the model's prior. The
  design pass recommends keeping that draw, so the population stays a prior draw and SBC stays valid (§6.10). The
  alternative is to change the model's ε prior to match the spec. Either choice is Ben's.
- Data: night summaries simulated as in #632's `_summaries`.
- Truths with P outside the declared period domain (§1.3) are excluded and counted (stratum B′, as §12). Under the 2026-11-02
  domain (P_cut = max(2Δt_min, p_floor)), B′ holds only truths below twice the shortest gap between nights or below
  p_floor. The count is reported.

**Assertions:**

1. **N in the 99 % set.** The true cell N is in the smallest set of cells holding ≥ 99 % of p̂(N). At n_sys = 200 the
   test fails if misses ≥ 7, since P(≥ 7 | Binomial(200, 0.01)) = 0.005.
2. **No `cell-at-floor` failures** in any system.
3. **SBC** (Talts et al. 2018).
   - For N, use the randomized PIT u = F̂(N_true − 1) + U·p̂(N_true), with U uniform.
   - For ln P, use the rank of the truth among the resample-move draws of the true cell, thinned to their effective
     size.
   - Each is uniform under a calibrated pipeline. The test is χ² on 10 bins, failing at p < 0.001 per statistic.
4. **Coverage contract.** On the subset with circular truths, K ≤ 14 and positions only, the exact collapsed reference
   of `scripts/leg1_toys/toy.py` applies with a Gaussian TI′ prior. There, |ln Ẑ_N − ln Z_N| ≤ 4 sd(ln Ẑ_N) + 0.01 for
   every cell with p(N) ≥ δ.
5. **Eccentric coverage of the ζ starts.** In the e > 0.3 stratum the true cell's best mode has squared Mahalanobis
   distance to the truth, under its Laplace covariance, within χ²_d(0.99) (d the chart dimension) in ≥ 99 % of systems.
   A system misses when it exceeds that bound, and misses are judged by the binomial rule of assertion 1 at the 1 % rate
   expected under a correct Laplace approximation plus the search's δ. (The Mahalanobis-1 rule of §6.1 is a
   duplicate test, not a confidence region.)
6. **Scale equivariance (§6.9).** For a subset of systems, rerun with every quoted σ multiplied by c ∈ {e⁻¹, e^{1.5}},
   keeping every night's ŝ_k/c and the true s/c inside the prior bounds. The target is exactly equivariant, so for every cell with
   p(N) ≥ δ, |ln p̂_c(N) − ln p̂_1(N)| ≤ 4·√(sd_c² + sd_1²) + 0.01, and the true cell's best mode has the same inlier set.
7. **No failure depends on ln s at fixed effective noise.** Equivariance says only s·σ/a matters. Within strata of the
   effective noise s·σ/a, the failure rate (any flag that fails a system, plus `no-mode` on a system with an orbit) is
   the same across the terciles of the true ln s. The test is Cochran–Mantel–Haenszel across strata, failing at
   p < 0.001, so it needs no tuned threshold.
8. **Informativeness is reported, not gated.** Per ln s and σ/a stratum, the median size of the 99 % set over B and the
   KL divergence of p̂(N) from the prior. A pass in assertion 1 with p̂(N) equal to the prior is visible here.

Results are reported by stratum: all nights outliers (B″, where the correct answer is the prior and assertion 7
does not apply), ≤ m inliers left, and the rest.

**Tiers:**

- n_sys = 10 as a local smoke run under pyslot.
- n_sys = 200 as an OzSTAR stage that Ben submits.
- CI keeps only the fast exact-reference tests of 3b′.

The thresholds are fixed in the criteria file before the first n_sys = 200 run. Changing one afterwards is a recorded
decision, never a loosening after a failure.

**Fix 4 implemented (virgil#638).** `search_plan` records each fill's span pair and `_grid_stage` drops the fill's grid minima at `(ζ start, ln P)` nodes where `_covered` holds (λ_k from `_prediction_cost`; Δ_k the night's best peak height at its best scale over its floor; the largest λ over a night's peaks). It applies above `N_SET_MAX` only. Coverage gate: `test_the_velocity_criterion_keeps_the_true_mode_with_fewer_triplets` (K = 8, circular and e = 0.5, fill regime forced) passes. On the K = 14 two-season case (night length 0.007 d, σ = 6 mas·10⁻³ · a) it dropped **no** candidates in any cell: the prediction error of a pair at a random grid node exceeds the interior nights' heights, so the criterion never passes there. Results are bit-identical to before (max |Δ ln Z| = 0, max |Δp| = 0), so it saves nothing on this system; it can only help where velocities are much better than σ/t_ss here.
