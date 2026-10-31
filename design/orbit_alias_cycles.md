# Alias-resolved orbit fits: model, coordinates, cells, tempered SMC (Stage D)

Status: **design for Ben's review**, 2026-11-01 (Opus-reviewed plan; Sonnet draft of this note). Nothing here is
implemented. It is the note every later stage codes against (2a coordinates, 2b night summaries, 3a generic SMC, 3b
cells and position leg, 3c visibility leg and noise contract, 3d no-orbit verdict, 4 validation).

Companions: [orbit_mixture_v2.md](orbit_mixture_v2.md) (approved 2026-10-29: the noise model),
[orbit_dehack.md](orbit_dehack.md) (decisions 4, 5, 10, 12; §2I flux floor), [linear_marginalisation.md](linear_marginalisation.md),
[thiele_innes_marginalisation.md](thiele_innes_marginalisation.md), [orbit_gate_042.md](orbit_gate_042.md) (§3.3, the G1 gate).
Checks: [`scripts/check_orbit_coordinates.py`](../scripts/check_orbit_coordinates.py) (every Jacobian below) and
[`scripts/toy_tiprime_legs.py`](../scripts/toy_tiprime_legs.py) (§6.5).

Design notes stay general: gallery targets appear only as worked examples. No constant below was chosen by looking at
vv or gallery results (§10).

## 0. Summary

1. **Model (§1).** Mixture v2 with global inlier scales; isotropic orientation, log-uniform *a* and *P*, uniform *e*;
   period domain [P_cut, P_max] with P_cut = max(2Δt_min, P_night) as a **declared prior**.
2. **Outlier contract (§2).** b_k is the prior-predictive density of night *k* under the *same* per-exposure
   likelihood as the inlier term, with the companion's night state (p, v, f) drawn from π_out instead of from the
   orbit. #618's b_k is the uniform-disc, σ_v → ∞ special case. Measured on #618's fixture: at the true orbit b_k − ℓ_k is
   −27 to −58 nats, so **no choice of π(v) or scales restores #618's six failing tests**: they fail because the optimizer
   returns partial-orbit optima at least 54 nats below the truth (data precision pinned; a float32 rerun passed). The contract is still fixed (it is a modelling error), and the
   tests are the acceptance test of the tempered search (§2.5).
3. **Coordinates (§3).** Anchor Thiele–Innes TI′; z ∈ ℝ⁴ with u = z_u², V = z_v² (prior ∝ |z|⁻⁴, 4:1 cover); ζ with
   k + ih = ζ² (uniform prior, 2:1 cover). Both are *derived* departures; all Jacobians checked to 6 × 10⁻¹⁴.
4. **Cells (§4).** T/P = N + ρ, π(N, ρ) ∝ 1/(N + ρ); exact partition for any edges; edges and anchors affect efficiency
   and labels only.
5. **Search (§5–§7).** Per band, an adaptive tempered SMC: leg 1 at position level on per-peak (p, v, f) night summaries,
   leg 2 a bridge to the visibility level. TI′ moves are the Luger conditional with an MH correction (leg 1; leg 2
   per §6.3); the Joker-style collapsed alternative (J) is the cross-check. Neither dominates in the toy (§6.5).
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
  e^{o_k(x, s)}, o_k = Σ_g ln ∫₁^{s_max} L_kg(x; s_g t) dt/(t ln s_max), the night's *own* likelihood at the orbit with
  every group's scale inflated by t. It depends on (x, s), so there is **no constant floor**: an outlier night still pulls
  x (weight ≈ 1/s_max²), and the position-level analogue is õ_k(x, s) of §5. Wherever this note says "the floor
  ε e^{b_k}", read ε e^{o_k(x, s)} under (iii′); §5 and §6 give both forms. The Luger move and the label Gibbs step carry
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

P ∈ [P_cut, P_max], **P_cut = max(2·Δt_min, P_night)**:

- Δt_min: the shortest gap between consecutive nights. P_night: the within-night motion bound of `period_window`
  (the companion moves less than `motion_fraction` of a resolution element in the longest night, floor `p_floor`).
- Above P_cut the closest pair of nights advances by less than half a cycle, so the regular-sampling alias
  (prograde, retrograde, ν) ≡ (retrograde, prograde, 1/Δt − ν) lies outside the domain. The factor 2 holds at every
  inclination.
- It is **not** a Nyquist limit for irregular sampling (VanderPlas 2018): below Δt_min, irregular gaps and within-night
  velocities can still identify P. It is a prior choice (Ben, 2026-11-01), reported in every summary, overridden by
  `p_range`. There is no `period-floor` flag and no κ (orbit_dehack decision 4: no nats anywhere).
- **P_max** follows orbit_dehack decision 13 (`span_factor` deleted): periods beyond the span form one long-period band
  [span, P_max] with P_max = 2π a_hi · span/σ_min, σ_min the smallest per-night position error (the motion over the span
  falls below it); a data-derived, reported domain limit. The gate draws truths up to 3 × span, far inside it.
- **In practice P_night usually sets P_cut.** On random epoch patterns (gate stratum B's schedules S1–S3, K ∈ {4, 6, 8, 12};
  night length U(0.05, 0.3) d, resolution U(2, 5) mas, field ∈ {10, 30} mas: illustrative ranges, tuned to no target;
  `scripts/p_cut_random_epochs.py`), P_night ≥ 2Δt_min in 81% of draws (S1 0.81, S2 0.70, S3 0.91), the median P_cut is
  11.8 d, and P_cut/span has quartiles 0.011, 0.047, 0.30 (10th, 50th, 90th percentile). The prior's lower end therefore
  depends on a_hi (through `field`), `motion_fraction` and `gap_days` = 0.5 of `_night_length` (all in §10), not mainly on
  Δt_min. The contract's "curvature inside a night is ignored" rests on P_night: if P_night is ever dropped, that
  dependency must be restated.
- Consequence for the gate: §12.

### 1.4 Linear parameters (Luger, Foreman-Mackey & Hogg 2017)

| parameter | linear in | use |
|---|---|---|
| gains, closure offsets | observables (linearized) | already marginalized analytically (Stage 1 makes it cheap) |
| ln s_g | no (scale) | sampled; one Woodbury solve per night at the current s_g |
| outlier own scale (per group, per night) | no | integrated inside b_k only (closed-form Gamma integral, or 1-D quadrature with gains) |
| TI′ = (A′, B′, F′, G′) | night positions *and* velocities | exact Gaussian conditional given the nonlinear core and the night labels; used as an MH-corrected Gibbs move (§6.3), **not** as a closed-form marginal: the isotropic prior is not Gaussian in TI′ and peaks and outliers make the marginal a sum over ∏(J_k + 1) assignments |
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
(t_j − t̄_k), exactly the within-night motion the inlier term has (`arcs.night_motion`; the orbit's curvature inside a
night is below the half-resolution motion bound of P_night and is ignored in both). The own scale s_k has one
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
   Restoring the tests is the job of the tempered search (§5–6), which keeps every mode with its weight and needs no
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
working prior has to be reweighted by (§6.3–6.4).

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

One cheap pilot pass of leg 1 (§5) over the whole period domain at n_pilot = M/4 particles, one replicate, run to β = 1,
gives the pooled position-level marginal of x = T/P. For each integer N, the edge x_N is the minimum of the
Gaussian-smoothed marginal (width ¼) inside [N − ½, N + ½]; with no mass in the window it is N. Labels by the nearest
integer are unchanged for any edge in that window. Constants: ±½, ¼, ¼M, all affecting labels only (§10).

### 4.5 The long-period band

Everything with x < x_low (below the lowest cell: P longer than the lowest cell's upper edge, up to P_max) is one band,
sampled in ln P with π(ln P) uniform; for P > T it carries no cycle count. Its mass is per band, never pooled.

## 5. Night summaries (position level)

Ben's rule: velocity only with adequate span and SNR, and no threshold. This is the interface Stage 2b codes against.

**What is computed, and at what.** Each night k is fitted **per peak at visibility level** with linear motion over the
**whole night** (no half-night split, so offsets and spanning gain modes are kept; `OIData._subset` raises on offsets, and
hd_9312 had 0 velocities; `_MAX_SHIFT` is deleted), at

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
  õ_k(x, s) = Σ_g ln ∫₁^{s_max} [peak sum at scale s_g t] dt/(t ln s_max), the same inflation integral on the catalogue,
  evaluated by Gauss–Legendre in ln t (node count in §10), so the two legs again share the structure.
- **Completeness assumption:** a visibility-level mode absent from the catalogue gets position-level mass only from the
  outlier term and the bridge cannot find it. The planted-peak test of #618 (`test_a_planted_narrow_peak_is_found`) is the
  check; the catalogue is the same one that makes b_k exact.
- Summaries feed only leg 1; the target of leg 2 is the per-exposure likelihood.

## 6. The tempered SMC

One adaptive tempered SMC per band, started from the band prior. Particles are the sampled vector x (10 + J chart
coordinates: ρ or ln P, ζ (2), y (4), logit ln f, θ, vis_amp, ln s_g) plus discrete labels (peak labels per night,
brighter/fainter, disc).

### 6.1 Leg 1: position level

Temper from the band prior to the night-summary likelihood, π_β ∝ prior · L_pos^β, with each night's outlier term under
§2's contract ((ii): the constant e^{b_k}; (iii′): õ_k of §5), so the leg is defensive night by night. The next β is the largest for which the conditional ESS of the
incremental weights is ≥ ½ (Zhou, Johansen & Aston 2016; ESS-adaptive tempering is Jasra et al. 2011; the SMC sampler is
Del Moral, Doucet & Jasra 2006). Leg 1 does nearly all the compression and gives Z_N^pos.

### 6.2 Leg 2: visibility level

Bridge π_β ∝ prior · L_pos^{1−β} · L_vis^β, β: 0 → 1, from leg 1's particles with the same adaptive rule. The estimate
of Z_N is the product of the two legs' normalizing-constant ratios, unbiased in the SMC sense (Del Moral et al. 2006).
The bridge's ratio is finite when the catalogue is complete (§5). **Its efficiency rests on L_pos ≈ L_vis on the retained
peaks, not on a defensive bound:** per night, L_vis/L_pos ≤ 1 + (1 − ε) e^{ℓ_vis − b_k}/ε, which is e^{30} to e^{60} at §2.4's
numbers, and ε is integrated jointly so nights do not factor. The bridge's conditional-ESS steps S₂ measure it (Stage 3c).

### 6.3 Moves

- **Fixed-L HMC** on the chart coordinates, as `virgil.smc` (#623, read at 525ba54) implements it: BlackJAX pretuning, a
  per-particle step size and trajectory length perturbed with σ_ε = ε/4, 5 steps, ε₀ = d^{−1/4}, L₀ = 4, a dense
  particle-covariance mass matrix, at most 200 stages (Buchholz, Chopin & Jacob 2021 for fixed-L HMC within SMC). This
  note's own targets (acceptance 0.651, Beskos et al. 2013; trajectory-length candidates {½, 1, 2}×) are not what #623
  implements, so they are dropped from §10 in favour of #623's constants; vmapped NUTS would run every particle to the
  slowest tree.
- **Luger TI′ move** (Luger, Foreman-Mackey & Hogg 2017). Given the nonlinear core c and the night labels z, the night states
  are linear in θ = TI′ (p and v are linear in TI′; ln f is in the core; the brighter/fainter sign is linear). A Gaussian
  working prior N(0, s_w² I), s_w² = ⟨a²⟩/3 (the per-component variance of TI′ under the isotropic prior:
  E[A² + B² + F² + G²] = ⟨a²⟩ E[1 + cos² i] = 4⟨a²⟩/3, ⟨a²⟩ = (a_hi² − a_lo²)/(2 ln(a_hi/a_lo))), gives an exact Gaussian
  conditional G(θ | c, z, β) ∝ N_w(θ) L_G(θ; z)^β, with L_G the labelled position-level Gaussian. Propose from it
  (independence proposal).
  - **Leg 1** (target p_iso L_G^β): accept with min{1, [p_iso/N_w](θ′)/[p_iso/N_w](θ)}, p_iso the §3.2 density. This is
    exact only because the tempered likelihood is exactly Gaussian in θ given (c, z).
  - **Leg 2** (target p_iso L_pos^{1−β} L_vis^β, proposal from G(θ | c, z, 1)): the ratio is
    [p_iso/N_w](θ′)/[p_iso/N_w](θ) × {[L_vis/L_G](θ′)/[L_vis/L_G](θ)}^β, one extra visibility evaluation per proposal.
    Without the second factor the move would sample the wrong distribution. **The Luger move is used in leg 1 only by
    default;** the leg-2 form is the exact option, to be measured in Stage 3b (its acceptance falls as β → 1 wherever
    L_vis/L_G varies), and leg 2 otherwise uses HMC and the label Gibbs step.
  - A wrong s_w costs acceptance, never correctness. The independence proposal is not uniformly ergodic: the weight
    p_iso/N_w ∝ 1/(|u||V|) is unbounded at |V| → 0 (Mengersen & Tweedie 1996), so it can stick face-on (§6.5, measured).
- **Labels are summed in the target, drawn only inside the move.** Every discrete label (peak, outlier and t_k under (iii′),
  brighter/fainter, disc) is summed exactly per night in the target, so particles stay flat vectors. The Luger move draws
  the labels from their conditional given θ, proposes θ′ from G(θ | c, z), and accepts on the augmented target; the
  θ-marginal is invariant (a Gibbs-then-Metropolis step, as in the toy §6.5). #623's kernel therefore needs only a
  composite-mutation hook (custom MH and Gibbs moves), which Stage 3a adds; one label sweep and one Luger move per stage (§10).

### 6.4 The Joker-style collapsed alternative

Carry the night labels as particle state, so that **given the labels, TI′ marginalizes in closed form** under the
Gaussian working prior (The Joker; Price-Whelan et al. 2017, 2020). Temper likelihood under the working prior (exact moves,
no rejection), then swap to the isotropic prior by tempering the ratio (p_iso/N_w)^γ with γ: 0 → 1. The face-on ratio
p_iso/N_w ∝ 1/(|u||V|) has an infinite second moment (∫ d²u/|u|²: logarithmic divergence), so a single step has infinite
weight variance. After a step to γ_prev the particles follow π_γ ∝ N_w^{1−γ} p_iso^γ L and the next weight is
(p_iso/N_w)^{Δγ}, so E[w²] ∝ ∫ N_w^{1−γ_prev−2Δγ} p_iso^{γ_prev+2Δγ} L, whose face-on tail is ∝ ∫ |V|^{−(γ_prev+2Δγ)} d²V:
finite iff γ_prev + 2Δγ < 2. Only the first step (γ_prev = 0) is limited to Δγ < 1; later steps may be larger, and it
matters only if L puts non-negligible mass near |V| = 0. The conditional-ESS rule sees none of this tail (a finite particle
set cannot).

### 6.5 Toy comparison and the choice

[`scripts/toy_tiprime_legs.py`](../scripts/toy_tiprime_legs.py): fixed circular core, K = 8 nights of 2-D positions
(σ = 0.3, two outlier nights), a ∈ [1, 30], M = 1000 particles, 2 moves per stage, 10 replicates, three data seeds; ε =
0.1 fixed. **L** is the Luger-move SMC from the isotropic prior; **J** is the Joker-style path above with the exact
conditional moves. Both estimate ln Z = ln ∫ p_iso(θ) ∏_k L_k(θ) dθ and use the same adaptive rule. Results, raw:

ln Ẑ mean and sd over 10 replicates (cost = per-particle night evaluations + θ conditionals; the two methods cost the same
to within 25%). s_w = 6.6 is the derived working scale ⟨a²⟩/3 of §6.3 (a ∈ [1, 30]); 15 is a deliberately broad one.

| data seed | method | s_w | ln Ẑ | sd | stages | cost | variance × cost |
|---|---|---|---|---|---|---|---|
| 0 | L | 6.6 | -33.346 | 0.279 | 8.0 | 1.01e6 | 7.8e4 |
| 0 | L | 15 | -33.343 | 0.34 | 8.0 | 1.01e6 | 1.2e5 |
| 0 | J | 6.6 | -33.349 | 0.096 | 9.0 | 1.03e6 | 9.5e3 |
| 0 | J | 15 | -36.494 | 2.166 | 7.8 | 8.4e5 | 4.0e6 |
| 1 | L | 6.6 | -34.096 | 0.308 | 8.0 | 1.00e6 | 9.5e4 |
| 1 | L | 15 | -34.166 | 0.443 | 8.0 | 1.00e6 | 2.0e5 |
| 1 | J | 6.6 | -34.39 | 0.311 | 9.0 | 1.02e6 | 9.9e4 |
| 1 | J | 15 | -37.058 | 1.754 | 7.4 | 7.9e5 | 2.4e6 |
| 2 | L | 6.6 | -35.372 | 0.261 | 8.0 | 1.00e6 | 6.9e4 |
| 2 | L | 15 | -35.512 | 0.419 | 8.0 | 1.00e6 | 1.8e5 |
| 2 | J | 6.6 | -35.612 | 0.288 | 8.9 | 1.00e6 | 8.3e4 |
| 2 | J | 15 | -40.121 | 2.578 | 7.3 | 7.7e5 | 5.1e6 |

Reference ln Z, by enumerating the 2⁸ label sets (the inlier nights are Gaussian in θ given the labels; the prior
expectation of each Gaussian is a Monte Carlo average of 2 × 10⁵ draws; `exact_ln_z` in the script; two MC seeds agree to
0.002): **−33.342, −34.124, −35.389** for the three data seeds. L agrees with it within its error at both s_w (|Δ| ≤ 0.07 at
s_w = 6.6, ≤ 0.14 at 15, against a standard error of the mean of 0.1); J at s_w = 6.6 is low by 0.01, 0.27 and 0.22, and at
15 by 3.2, 2.9 and 4.7. So the J bias is real, not a bug in one path.

More moves (10 per stage, s_w = 15, generic truth): L stays unbiased with sd 0.10–0.20 (ln Ẑ −33.327, −34.133, −35.342 against
−33.342, −34.124, −35.389); J falls by 1.6, 1.1 and 2.5 and has sd 0.9–1.4. J's bias shrinks with moves, so it is a stage-1
mixing failure at a broad working prior, not the infinite-variance cusp tail of §6.4.

**Face-on truth** (`--faceon`, truth TI′ = [4, −2.5, 2.5, 4]; exact ln Z −31.743, −32.951, −34.137), 2 moves, s_w = 6.6:

| data seed | method | ln Ẑ | sd |
|---|---|---|---|
| 0 | L | −31.997 | 0.90 |
| 0 | J | −31.780 | 0.17 |
| 1 | L | −33.409 | 0.70 |
| 1 | J | −33.124 | 0.36 |
| 2 | L | −34.466 | 0.78 |
| 2 | J | −34.676 | 0.60 |

L has 3–5× the generic-truth sd and is low by 0.25–0.46 on seeds 0–1. The reason is the independence-MH weight
p_iso/N_w ∝ 1/(|u||V|), unbounded as |V| → 0 (§6.3, Mengersen & Tweedie 1996): the chain sticks near the face-on cusp. J, which
integrates θ out, is steadier here. With **one z-space random-walk step added** (`--zmove`, step (2.38/2)·mean sd(z), target
|z|⁻⁴ × likelihood, applied at β = 1) L's face-on sd falls to 0.37, 0.59 and 0.56 and its bias to −0.35, −0.28 and −0.05,
and the generic-truth result is unchanged (L at 2 moves: sd 0.22, 0.27, 0.22; |bias| ≤ 0.11).

**Decision: neither dominates.** L keeps the default structure with the z-space step added to its mutation; J is the cross-check
and the fall-back where L's z-step does not reach J's face-on sd. Stage 3b's acceptance: L + HMC (with the z step) must have an
sd within 1.5× J's on the face-on toy (today: 2.2×, 1.7×, 0.9× on the three seeds), else J or a hybrid is the default. J's
sensitivity to the working scale remains a defect (a wide s_w costs it 3–5 nats at 2 moves). **Either way the SMC core (#623)
needs custom MH and Gibbs moves** (§6.3); that is a Stage 3a deliverable independent of this choice. The toy fixes the core and
the labels' enumerability is used by neither method; a real fit has a nonlinear core and ∏(J_k + 1) labels, which only raises J's
collapsed label cost.

### 6.6 Replicates, errors, failure

- Two independent replicates per band; the band reports their mean ln Ẑ, and p(N) ∝ Ẑ_N uses it. The replicate
  difference |Δ ln Ẑ| is a diagnostic only: |Δ|/√2 estimates the sd of one replicate (|Δ|/2 that of the mean), with one degree
  of freedom, so its 95 % interval spans a factor of about 30. It is never propagated into p(N) or the pruning rule, and is
  reported beside the band. A per-run, genealogy-based variance estimator is a Stage 3a option to replace it. The SMC particles give the posterior.
- **Hitting the time cap is a failure**: a posterior at β < 1 is never reported. One flag, `band-undersampled`, marks a
  band whose final ESS is below ESS_target; it is information, with no retry. No cascade: one diagnostic, one action, one
  flag.
- The mass is per band, never pooled.

### 6.7 Why SMC

Nested sampling would do the whole prior-to-posterior compression on whichever likelihood it targets, and its cost
multiplies by the number of bands. Parallel tempering (Pigeons, as in Octofitter; Thompson et al. 2023, Surjanovic
et al. 2023) handles multimodality well, but every chain pays the target likelihood and Z needs stepping stones. A grid has
a resolution parameter, covers only a surrogate and gives no particles to continue from; it survives only as a test
oracle on low-dimensional toys. Tempered SMC lets the cheap likelihood do the compression, carries its particles straight
into the expensive leg, gives Z directly and vectorizes.

## 7. Pruning

Optional after leg 1: drop bands whose position-level mass is below δ/N_bands. By construction the position-level mass
dropped is < δ. The visibility-level mass of dropped bands can differ, and Ẑ_N carries Monte Carlo noise, so pruning stays
**opt-in until Stage 4 measures the mean pruned posterior mass**: run the unpruned two-leg posterior on simulated
populations (random orbits on the gallery's real epoch patterns, never the gallery's own data) and report
E[Σ_dropped p_vis(N)] with its standard error against δ. Pruning becomes default only if the mean plus 3 SE ≤ δ. The
replicate difference of §6.6 is not used to prune.

## 8. The no-orbit verdict

An empirical false-alarm probability at α = 0.01 (orbit_dehack decision 4, cheap tier; `detection.DetectionMC`, #621). The
statistic is **T = max_n S_n − S_null**, where S_n is the seed-profile evidence of the best orbit seed in band n at position
level and S_null is the same quantity for the single-star model. Both are computed identically on the data and on
simulated single-star nulls. **Single-star position-level likelihood:** the night likelihood with no companion, L_k⁰
(§5, at visibility level), enters as the reference, so a peak's gain is ln L_j − ln L_k⁰. S_null is the mixture sum over
nights with those gains set to zero; orbit statistic and null share the same b_k, so T is the evidence of the gains alone.

**FAP** = (1 + #{T_null ≥ T_obs}) / (1 + n_null) (Phipson & Smyth 2010). The null count is **raised step by step**: 100 → 200 →
400 → 800, to a cap of 1000 (= 10/α); stop when the 95 % interval of the FAP excludes α (verdict: FAP < α or ≥ α), or at
the cap (verdict: **inconclusive**, reported as such). Each null is a simulated single-star data set, summarised once per
night (T_summ), with its own b_k (nulls change the data, so b_k is **per null**, K·T_b each) and a seed profile
(B·n_seed·K·J·c_pos, §9). The profile sums over all B bands, so the null cost scales with B. The visibility leg informs
the posterior, not the verdict, so that the null calibrates the statistic actually thresholded.

## 9. Cost formula

Per system, with every symbol a measured quantity (Stage 0 `run.json` counters, Stage 1 timings):

  T_sys = T_compile + T_epochs + T_summ + K·T_b + T_pilot + f_tune·[B·R·M·S₁·m₁·(L₁ + 1)·g_pos
          + B_keep·R·M·S₂·m₂·(L₂ + 1)·g_vis] + n_null·T_null

  T_null = T_summ + K·T_b + B·n_seed·K·J·c_pos.

| symbol | meaning | source |
|---|---|---|
| T_compile | XLA compile of the unrolled `_night_terms` (once per shape) | Stage 0 |
| T_epochs | epochs stage (`epoch_positions`) | Stage 0, per system |
| T_summ | per-night peak summaries (includes the keep rule's J_k cost, §5) | Stage 2b |
| T_b, K | b_k per night: 5-D (p, v, f) adaptive cubature, cached per fingerprint | #618 / Stage 3c |
| B, B_keep | bands, bands kept after pruning (B_keep = B while pruning is opt-in) | data |
| R | replicates per band (2) | budget |
| M | particles | budget |
| S₁, S₂ | leg-1 and leg-2 stages | measured; set by the conditional-ESS rule |
| m₁, m₂ | HMC moves per stage | budget |
| L₁, L₂ | leapfrog steps per move | measured |
| f_tune | step-size and trajectory pretuning factor (BlackJAX), 1–3 | measured |
| g_pos, g_vis | gradient cost, position- and visibility-level | Stage 0, Stage 1 |
| T_pilot | edge pilot ≈ (M/4)·S₁·m₁·(L₁ + 1)·g_pos | as above |
| n_seed, J, c_pos | seeds per band, peaks per night, per-peak position-level cost | Stage 3d |
| n_null | stepwise 100–1000 (§8) | budget |

g_vis loses Stage 1's per-night quadrature under v2's global scales (a cost up to 32× per gradient); B scales as
T/P_cut − T/P_max cells. **A worked example** at the order of magnitude of the gallery: B ≈ 250, R = 2, M = 2048, S₁ ≈ 20,
m = 4, L ≈ 10 gives B·R·M·S₁·m·(L + 1) ≈ 9 × 10⁸ position-level gradients, and with B_keep = B about 2 × 10⁸
visibility-level gradients at S₂ ≈ 5.

**Laptop verdict (revised).** At these defaults and without pruning the formula does **not** support population runs on the
laptop: 9 × 10⁸ position-level gradients is hours at a few μs, and the visibility term at a measured g_vis ≳ 1 ms is
2 × 10⁵ s. A single system is feasible only with pruning (B_keep ≪ B) or a measured g_vis ≲ 1 ms; populations go to
OzSTAR or a GPU. This is falsifiable: Stage 0/1 fill the table below, and any of g_pos, g_vis, S, B changes the verdict.

| quantity | M4 4 threads | M4 10 threads | A100 | source |
|---|---|---|---|---|
| T_compile, T_epochs, g_pos, g_vis, T_b, T_summ, T_null | _to fill_ | _to fill_ | _to fill_ | Stage 0 / 1 |
| S₁, S₂, L₁, L₂, B, ESS, f_tune | _to fill_ | | | Stage 3 on simulated populations |

## 10. Constants

Every constant is **derived**, a **tolerance** with a meaning, or a declared **budget**. None was chosen by looking at vv
or gallery targets. Provisional budgets are fixed in the 3a/3b PRs from simulated populations (stratum D) before any vv
run; changing one afterwards is a recorded decision.

| constant | value | kind | basis |
|---|---|---|---|
| s_max | 10 | declared prior bound | inherited `_NUMERIC_S_MAX`; covers χ²/N ≤ 100 (v2 §2.2) |
| ε prior | Beta(½, ½) | derived | Jeffreys for a rate |
| P_cut | max(2Δt_min, P_night) | derived (a declared prior) | §1.3 |
| P_night | 2π·field·night/(motion_fraction·res), ≥ night, ≥ p_floor | derived | `period_window` |
| motion_fraction | ½ | tolerance | a companion moving < ½ resolution element in a night; inherited |
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
| edge window, smoothing, pilot size | ±½, ¼, M/4 | tolerance / budget (labels only) | §4.4 |
| z chart | auxiliary Gaussian direction in ℝ⁴ plus a logistic ln a coordinate | chart, no statistical meaning | §3.5 |
| ζ chart | ζ = √e_max·y/√(1 + \|y\|²) | chart, no statistical meaning | §3.5 |
| s_w² (Luger move) | ⟨a²⟩/3 | derived | §6.3; a wrong value costs acceptance only |
| conditional ESS target | ½ | tolerance | Zhou et al. 2016; the adaptive rule |
| #623 HMC: α, σ_ε, steps, ε₀, L₀, mass, max stages | 1, ε/4, 5, d^{−1/4}, 4, dense, 200 | inherited | `virgil.smc` at 525ba54; §6.3 |
| label sweeps / Luger moves per stage | 1 / 1 | budget | §6.3 (set in 3a/3b) |
| Luger move in leg 2 | off (leg 1 only) | budget | §6.3; measured in 3b |
| particles M | provisional 2048 | budget | set on simulated populations in 3a |
| moves per stage m | provisional 4 | budget | idem |
| replicates R | 2 | budget | the minimum that gives an error |
| ESS_target | provisional M/4 at β = 1 | budget | flag only |
| wall-time cap per band | declared per run | budget | failure, never a posterior at β < 1 |
| α | 0.01 | declared | dehack decision 4 |
| n_pilot for edges | M/4 (the single pilot size) | budget | §4.4 |

P_cut caveat: in practice the within-night motion bound P_night usually sets P_cut (81 % of random epoch patterns, median
P_cut/span ≈ 0.05; `scripts/p_cut_random_epochs.py`), so the 2Δt_min term seldom binds. With (p, v) night summaries
P_night's role is only inside P_cut; the max(2Δt_min, P_night) rule stands (Ben's decision); P_night could now be dropped,
since within-night motion is modelled. Not changed here.

## 11. Literature and provenance

Owen & Zhou 2000 is withdrawn: its defensive bound is vacuous here (§6.2). Entries are in `references/provenance.toml`; verified against ADS (bibcodes) or Crossref (the statistics journals
ADS does not index).

| source | role here | verified |
|---|---|---|
| Luger, Foreman-Mackey & Hogg 2017, RNAAS 1, 7 | exact Gaussian conditional for linear parameters (§1.4, §6.3) | ADS 2017RNAAS...1....7L (existing entry) |
| Price-Whelan, Hogg & Foreman-Mackey 2017, ApJ 837, 20; Price-Whelan et al. 2020, ApJ 895, 2 | The Joker: collapsed linear parameters, label sampling (§6.4) | ADS 2017ApJ...837...20P (existing), 2020ApJ...895....2P |
| Lucy 2014, A&A 563, A126 | mass estimates for incomplete visual-binary orbits (prior art on short arcs; not the source of the alias structure) | ADS 2014A&A...563A.126L (existing) |
| Del Moral, Doucet & Jasra 2006, JRSSB 68, 411 | SMC samplers, Ẑ (§6) | Crossref 10.1111/j.1467-9868.2006.00553.x |
| Jasra, Stephens, Doucet & Tsagaris 2011, Scand. J. Stat. 38, 1 | ESS-adaptive tempering (§6.1) | Crossref 10.1111/j.1467-9469.2010.00723.x |
| Zhou, Johansen & Aston 2016, JCGS 25, 701 | conditional ESS rule (§6.1); not the source of the two-replicate error (§6.6) | Crossref 10.1080/10618600.2015.1060885 |
| Buchholz, Chopin & Jacob 2021, Bayesian Anal. 16, 745 | fixed-L HMC within SMC (§6.3) | Crossref 10.1214/20-ba1222, arXiv:1808.07730 |
| Mengersen & Tweedie 1996, Ann. Stat. 24, 101 | independence MH is uniformly ergodic iff the weight is bounded (§6.3) | Crossref 10.1214/aos/1033066201 |
| Kustaanheimo & Stiefel 1965, J. reine angew. Math. 218, 204; Levi-Civita 1920, Acta Math. 42, 99 | regularizing maps (z, ζ are analogues; §3.2) | Crossref 10.1515/crll.1965.218.204; 10.1007/BF02404416 |
| Blunt et al. 2017, AJ 153, 229 | OFTI: rejection sampling of short arcs (§6.4) | ADS 2017AJ....153..229B |
| Phipson & Smyth 2010, Stat. Appl. Genet. Mol. Biol. 9, 39 | permutation p-values (1 + #)/(1 + n) (§8) | Crossref 10.2202/1544-6115.1585 |
| VanderPlas 2018, ApJS 236, 16 | Nyquist limit does not apply to irregular sampling (§1.3) | ADS 2018ApJS..236...16V (existing) |
| Shoemake 1992, Graphics Gems III, 124 | unit quaternions, SO(3) Haar measure (§3.2) | Crossref 10.1016/b978-0-08-050755-2.50036-1 |
| Thompson et al. 2023, AJ 166, 164; Surjanovic et al. 2023, arXiv:2308.09769 | Octofitter and Pigeons: parallel-tempering prior art (§6.7) | ADS 2023AJ....166..164T (existing); 2023arXiv230809769S |
| Genz & Malik 1980; Blake & Zisserman 1987 | b_k cubature (#618); GNC (superseded) | in #618 / mixture v2 |
| Rasmussen & Williams 2006, A.9 | Woodbury and determinant lemma (Stage 1) | textbook |

**Derived departures (no source):** the z map (§3.2), the ζ map (§3.3), the anchor TI′ rotation (§3.4), the contract
for b_k with velocity (§2) and the peak keep rule (§5). Each is flagged *derived* in the methods register by its stage and
checked by a vv-independent test (the check script, the prior-recovery tests, the #618 oracle in 5-D).

## 12. G1 criteria change (separate PR; awaiting Ben's sign-off; the gate is **not** changed by this PR)

`design/orbit_gate_042.md` §3.3 draws stratum B truths with *P* log-uniform on [1 d, 3 × span]. Under the declared prior
of §1.3, truths with *P* < P_cut = max(2Δt_min, P_night) fall outside the default prior. The change is in
[#625](https://github.com/benjaminpope/virgil/pull/625), not here: the draw is unchanged, truths with *P* < P_cut are
excluded and counted (stratum B′, reported, not scored; excluding rather than redrawing keeps the generator stream and
the before/after pairs identical), the baseline stays reported under its own criteria commit, and the rule is
`aliases.period_cut`, shared by the engine and the vv harness. M4's `init=` oracle is unchanged. Stratum A (explicit `p_range`) is unaffected. The engine's existing redraw rule already
uses P_night at the truth's a, so only the 2Δt_min part of P_cut is new in practice.

## 13. Stage consequences

| stage | takes from this note |
|---|---|
| 2a | §3 (maps, chart, TI′ density, invariants, Campbell conversion); prior-recovery tests; register z, ζ as derived |
| 2b | §5 (night summaries, keep rule, v variance, σ_J, (iii′) analogue); the (k, h) Kepler solve (§3.3) |
| 3a | §6.1–6.3, 6.6 as generic, non-orbit SMC on BlackJAX (#623) plus the composite mutation hook (custom MH/Gibbs); budgets of §10 |
| 3b | §4, §5 position-level target, §7 pruning measurement |
| 3c | §2 contract (5-D b_k, joint π_out), §6.2 bridge, §1.1 noise; #618's six tests as acceptance at both pinned precisions |
| 3d | §8 (T statistic, stepwise FAP) |
| 4 | the vv criteria file (committed before any run; awaiting Ben's sign-off), §7 measurement, §9 table |
