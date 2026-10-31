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
   orbit. #618's b_k is the v = 0 special case. Measured on #618's fixture: at the true orbit b_k − ℓ_k is −31 to −59
   nats, so **no choice of π(v) or scales restores #618's six failing tests**: they fail because the optimizer returns
   partial-orbit optima at least 59 nats below the truth. The contract is still fixed (it is a modelling error), and the
   tests are the acceptance test of the tempered search (§2.5).
3. **Coordinates (§3).** Anchor Thiele–Innes TI′; z ∈ ℝ⁴ with u = z_u², V = z_v² (prior ∝ |z|⁻⁴, 4:1 cover); ζ with
   k + ih = ζ² (uniform prior, 2:1 cover). Both are *derived* departures; all Jacobians checked to 10⁻¹⁵.
4. **Cells (§4).** T/P = N + ρ, π(N, ρ) ∝ 1/(N + ρ); exact partition for any edges; edges and anchors affect efficiency
   and labels only.
5. **Search (§5–§7).** Per band, an adaptive tempered SMC: leg 1 at position level on per-peak (p, v, f) night summaries,
   leg 2 a bridge to the visibility level. TI′ moves are the Luger conditional with an MH correction; the Joker-style
   collapsed alternative is **not chosen** (toy, §6.5).
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
- P_max = `span_factor` × span (4, inherited from `period_window`; §10).
- Worked example: one pair of nights 1.99 d apart sets P_cut ≈ 4 d.
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

π_out(p, v, f) = π(p) π(v) π(f), independent (a modelling choice, flagged):

- π(p): uniform on the disc |p| ≤ r_max = a_hi(1 + e_max), density 1/(π r_max²).
- π(f): the prior of §1.2, log-uniform on (lo, 1].
- **π(v): the projected-velocity density of the orbit prior at a random epoch**, isotropic in direction, with the
  speed density tabulated once from the prior: draw (a, P, e, orientation, phase) from §1.2, compute the sky speed,
  histogram ln|v|. It is a deterministic function of the declared prior, so it has no constants of its own. To first
  order ln|v| = ln a − ln P + ln 2π, a trapezoid (convolution of two uniforms) of width ln(a_hi/a_lo) + ln(P_max/P_cut),
  smeared by the eccentricity factor (up to (1+e)/(1−e)). v = 0 (#618) is **not** a prior: it is the limit of
  infinitely well-measured stationarity, and gives the outlier a likelihood the inlier does not have to pay for.
- Scales: as v2 (outlier own scales, inlier global). Their Occam cost is Σ_g ln(2 ln s_max √(ν_g/π)), 2.0–2.4 nats per group
  at ν = 9–18 and s_max = 10, and moves b_k by −J ln 2 between s_max = 10 and 100 (orbit_mixture_v2 §2.2).

### 2.3 Laplace form and the fair-comparison identity

Near each peak of the night's likelihood (peak state ŝ, covariance Σ over (p, v, ln f)):

  b_k ≈ ln Σ_peaks L_peak · (2π)^{5/2} |Σ|^{1/2} π_out(ŝ),

while for the true orbit ℓ_k(truth) ≈ ln L_peak. So at the truth

  Δ_k^truth = b_k − ℓ_k(truth) ≈ ln[(2π)^{5/2} |Σ|^{1/2} π_out(ŝ)] = −(Occam of p) − (Occam of v) − (Occam of f).

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

(12.38 d data: b_k = [−42.4, −74.8, −54.6, −69.5, −51.7, −45.7], ℓ_k(truth) as above to 0.1, Σℓ_k = −71.7.)

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
2. The mixture log likelihood at the truth is Σℓ_k + ln[B(½, K + ½)/B(½, ½)] = −71.7 − 1.5 = **−73.2**; #618 reports
   best-fit optima of **−132.8 / −132.9** (float32, n = 20 / 21; q = [1 0 0 0 0 1]) and, in float64, an optimum with
   q = [1 0 0 1 1 1] whose four outlier nights alone cost 209 nats. The reported optima are ≥ 59 nats below the truth
   (float32), ≈ 160 (float64).
3. Raising b_k by the contract's ≈ +10 nats per outlier night makes a partial optimum with m outlier nights cheaper by
   ≈ 10 m: the float32 optimum (m = 2 outlier nights) gap shrinks from 59 to ≈ 39 nats, and the float64 optimum (m = 4) gap from
   ≈ 160 to ≈ 120; both stay positive.
4. So the failing tests are **search failures** (graduated non-convexity's cliff: a night absorbed at q = 1 stops
   pulling, and the optimizer settles). #618's own diagnosis ("a poor fit is absorbed at cost b_k, so the alias contest
   loses discriminating power") is true of the *search*, not of the target: the target still prefers the truth by ≥ 39 nats (float32) and ≈ 120 (float64).
   Restoring the tests is the job of the tempered search (§5–6), which keeps every mode with its weight and needs no
   optimizer basin.

**Decision:** the contract of §2.1–2.2 is the target in Stage 3c (it is consistent; #618's v = 0 integral is its
special case and is the Stage 3c unit-test oracle). **Acceptance:** #618's six tests, unedited (including the float32 `[12.1]`
case), pass on the 3c path; any that fails is reported raw and the failure investigated, never loosened. This design does
**not** claim the contract restores them.

### 2.6 Tests registered for Stage 3c

- b_k with π(v) → 5-D cubature against a fine-grid reference (the #618 oracle in 5-D on a toy night), ≤ 4δ.
- b_k(π(v)) → b_k^static (#618) as σ_v → ∞ (an unresolved night): the continuity limit.
- Δ_k^truth on the six-night fixture lies within 3 nats of the Laplace estimate of §2.3.
- E[b_k − ℓ_k(truth)] ≤ 0 over noise draws of a *moving* night (the #618 calibration test, extended with v).
- The six #618 tests, unchanged.

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
everywhere except the origin, which a_lo excludes. It has **no 1/r face-on cusp** (about 3.4% of isotropic orbits lie
within 15° of face-on), no cos i bound and no Ω fold.

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
The 2:1 cover (ζ → −ζ) is exact. The anchor-frame positions are polynomial in (k + ih, k − ih) (§3.4), hence analytic in
ζ: the sampler sees no coordinate singularity at e = 0, where (e, M_a) polar coordinates have one.

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

### 3.5 The radial chart

HMC on the shell needs an unconstrained chart. For y ∈ ℝ⁴, z = g(|y|) y/|y| with g increasing from r_lo to r_hi; then
|∂z/∂y| = g³ g′/ρ³ (checked, ρ = |y|) and the prior density in y is g′/(g ρ³). Stage 2a takes the logistic chart,
ln g = ln r_lo + (ln r_hi − ln r_lo) · ρ/(1 + ρ), and the radial logit for ζ's radius √e_max likewise. The chart
has no statistical meaning: it only decides HMC geometry, and its log-Jacobian is analytic.

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

The cells (N, ρ ∈ [0, 1)) partition the period domain for **any** choice of edges, so Z = Σ_N Z_N and the prior mass of
a cell with edges x_j < x_{j+1} (x = T/P) is ln(x_{j+1}/x_j)/ln(x_max/x_min), x_min = T/P_max, x_max = T/P_cut. Edges
and anchors therefore change efficiency and labels, never the answer. Within a cell the sampling coordinate is ρ, with
density ∝ 1/(N + ρ) (an analytic log-prior term).

### 4.2 Anchors

First anchor: the night with the smallest position error σ_p in the first season; last anchor: the same in the last
season. A season is a maximal run of nights with consecutive gaps ≤ P_cut (derived: the shortest period in the domain).
T is then ≈ the full span and no cell holds sub-ridges. A poor anchor costs efficiency only (§4.1).

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

Ben's rule: velocity only with adequate span and SNR, and no threshold. Each night is fitted **per peak at visibility
level** with linear motion over the **whole night**, giving a peak catalogue of (m_j, Σ_j, ln L_j): state mean over
(p, v, ln f), covariance, and peak log likelihood. No half-night split, so offsets and spanning gain modes are kept
(`OIData._subset` raises on offsets, and hd_9312 had 0 velocities); `_MAX_SHIFT` is deleted.

- **Variance of v:** σ_v² = σ_p² / Σ_j (t_j − t̄)² (12σ_p²/span² for uniform sampling): a short or faint night's
  velocity carries no weight automatically.
- **Peak keep rule (derived):** keep a peak iff its share of b_k is ≥ δ, i.e. its Laplace mass
  L_j (2π)^{5/2} |Σ_j|^{1/2} π_out(m_j) ≥ δ e^{b_k}. A peak below δ of the outlier floor cannot move the mixture by more
  than δ. J_k is then bounded by 1/δ and typically 1–3.
- **Position-level likelihood** of night k at orbit x: ℓ̃_k(x) = ln Σ_j L_j exp(−½ (s_k(x) − m_j)ᵀ Σ_j⁻¹ (s_k(x) − m_j)),
  s_k(x) = (p_k(x), v_k(x), ln f(x)); the outlier term is the **same b_k** (§2). The two legs therefore share the
  floor that makes the bridge's ratio L_vis/L_pos finite wherever the catalogue is complete.
- **Completeness assumption:** a visibility-level mode absent from the catalogue gets position-level mass only from the
  outlier floor and the bridge cannot find it. The planted-peak test of #618 (`test_a_planted_narrow_peak_is_found`) is the
  check; the catalogue is the same one that makes b_k exact.
- Summaries feed only leg 1; the target of leg 2 is the per-exposure likelihood.

## 6. The tempered SMC

One adaptive tempered SMC per band, started from the band prior. Particles are the sampled vector x (10 + J chart
coordinates: ρ or ln P, ζ (2), y (4), logit ln f, θ, vis_amp, ln s_g) plus discrete labels (peak labels per night,
brighter/fainter, disc).

### 6.1 Leg 1: position level

Temper from the band prior to the night-summary likelihood, π_β ∝ prior · L_pos^β, with each night's outlier term under
§2's contract (so the leg is defensive night by night). The next β is the largest for which the conditional ESS of the
incremental weights is ≥ ½ (Zhou, Johansen & Aston 2016; ESS-adaptive tempering is Jasra et al. 2011; the SMC sampler is
Del Moral, Doucet & Jasra 2006). Leg 1 does nearly all the compression and gives Z_N^pos.

### 6.2 Leg 2: visibility level

Bridge π_β ∝ prior · L_pos^{1−β} · L_vis^β, β: 0 → 1, from leg 1's particles with the same adaptive rule. The estimate
of Z_N is the product of the two legs' normalizing-constant ratios, unbiased in the SMC sense (Del Moral et al. 2006).
The bridge's ratio is finite when the catalogue is complete (§5). Weights are bounded by the defensive component of the
outlier floor in the sense of Owen & Zhou (2000): the floor ε e^{b_k} is a defensive mixture component per night.

### 6.3 Moves

- **Fixed-L HMC** on the chart coordinates, ε and L tuned per stage and per band (Buchholz, Chopin & Jacob 2021).
  Acceptance target 0.651 (Beskos et al. 2013, the d → ∞ optimum, also the Buchholz et al. default); trajectory length
  ε L chosen from {½, 1, 2} × the previous stage's by the best mean ESJD per gradient on the previous stage. vmapped NUTS
  would run every particle to the slowest tree.
- **Luger TI′ move** (Luger, Foreman-Mackey & Hogg 2017). Given the nonlinear core c and the current night labels, the
  night states are linear in θ = TI′. A Gaussian working prior N(0, s_w² I) with s_w² = ⟨a²⟩/3 (the per-component
  variance of TI′ under the isotropic prior: E[A² + B² + F² + G²] = ⟨a²⟩ E[1 + cos² i] = 4⟨a²⟩/3, ⟨a²⟩ =
  (a_hi² − a_lo²)/(2 ln(a_hi/a_lo))) gives an exact Gaussian conditional given c and the labels. Propose from it
  (independence proposal), accept with min{1, [p_iso/N_w](θ′) / [p_iso/N_w](θ)} where p_iso is the §3.2 density. The
  acceptance depends on the working scale only through that ratio; a wrong s_w costs acceptance, never correctness.
- Label Gibbs per night (peak labels, outlier, brighter/fainter, disc), at the tempered target; the tempered marginal
  Σ_z c_z^β per night keeps the augmented target consistent (as in the toy, §6.5).

### 6.4 The Joker-style collapsed alternative

Carry the night labels as particle state, so that **given the labels, TI′ marginalizes in closed form** under the
Gaussian working prior (The Joker; Price-Whelan et al. 2017, 2020). Temper likelihood under the working prior (exact moves,
no rejection), then swap to the isotropic prior by tempering the ratio (p_iso/N_w)^γ with γ: 0 → 1. The face-on ratio
p_iso/N_w ∝ 1/(|u||V|) has an infinite second moment (∫ d²u/|u|²: logarithmic divergence), so a single step has infinite
weight variance, and a step γ < 1 has E[w²] ∝ ∫ |u|^{−2γ} d²u, finite iff γ < 1: **the steps must stay below one**, which
the conditional-ESS rule enforces only empirically (the empirical ESS of finitely many particles cannot see the tail).

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

(An earlier run at s_w = 5 and 15 gave the same picture: J at 5 agreed with L to 0.2, J at 15 was biased low by 3 to 5.)

Choice: **L (the Luger move), as default**; J stays an opt-in possibility to measure on the full model in Stage 3b.
Reasons: J's accuracy depends on a working scale the data decide, and when the working prior is broad relative to the
posterior (s_w = 15 here) it is biased low by ≫ its reported error, which is the infinite-variance tail of §6.4 seen
through a finite particle set; L is insensitive to s_w (its sd barely moves). At the derived s_w, J's variance × cost is 0.1×, 1.0× and 1.2× L's on
the three seeds, and its mean ln Ẑ sits 0.00, 0.29 and 0.24 below L's (standard error of each difference ≈ 0.13), so J
is no better where it works and possibly slightly low. The toy fixes the core and the
labels' enumerability is not used by either method; a real fit has a nonlinear core and ∏(J_k + 1) labels, which only
increases the cost of J's collapsed label sampling.

### 6.6 Replicates, errors, failure

- Two independent replicates per band give the Z error (the difference of ln Ẑ over √2) and p(N) ∝ Ẑ_N, with its error
  propagated; the SMC particles give the posterior.
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
dropped is < δ. The visibility-level mass of dropped bands can differ, so pruning stays **opt-in until Stage 4 measures
the mean pruned posterior mass**: run the unpruned two-leg posterior on simulated populations (random orbits on the
gallery's real epoch patterns, never the gallery's own data) and report E[Σ_dropped p_vis(N)] with its standard error
against δ. Pruning becomes default only if the mean plus 3 SE ≤ δ.

## 8. The no-orbit verdict

An empirical false-alarm probability at α = 0.01 from `detection.DetectionMC` (#621; orbit_dehack §6, the detection-v3
prerequisite), replacing the 100-diameter null: the statistic is the position-level evidence (leg 1) of the orbit model
over the single-star model, computed identically on the data and on simulated single-star nulls; the verdict is
FAP < α. A null needs night summaries of simulated single-star data (a noise-peak catalogue per night), which is its
dominant cost (§9); the summary-level null simulation is specified in Stage 3d. The visibility leg informs the
posterior, not the verdict, so that the null calibrates the statistic actually thresholded.

## 9. Cost formula

Per system, with every symbol a measured quantity (Stage 0 `run.json` counters, Stage 1 timings):

  T_sys = T_epochs + T_summ + K·T_b + T_pilot + B·R·M·S₁·m₁·(L₁ + 1)·g_pos
          + B_keep·R·M·S₂·m₂·(L₂ + 1)·g_vis + n_null·T_null

| symbol | meaning | source |
|---|---|---|
| T_epochs | epochs stage (`epoch_positions`) | Stage 0, per system |
| T_summ | per-night peak summaries | Stage 2b |
| T_b, K | b_k per night (adaptive cubature, cached per fingerprint) | #618 / Stage 3c |
| B, B_keep | bands, bands kept after pruning (B_keep = B while pruning is opt-in) | data |
| R | replicates per band (2) | budget |
| M | particles | budget |
| S₁, S₂ | leg-1 and leg-2 stages | measured; set by the conditional-ESS rule |
| m₁, m₂ | HMC moves per stage | budget |
| L₁, L₂ | leapfrog steps per move | measured (tuned per stage) |
| g_pos, g_vis | gradient cost, position- and visibility-level | Stage 0, Stage 1 |
| T_pilot | edge pilot: ≈ (M/4)·S₁·m₁·(L₁ + 1)·g_pos | as above |
| n_null, T_null | FAP nulls, each = noise-peak summaries + position-level statistic | ≈ 1/(α·0.09) ≈ 10³ at 30% relative error; Stage 3d |

Structural remarks: g_vis loses Stage 1's per-night quadrature (up to 32× per gradient) under v2's global scales; B scales
as T/P_cut − T/P_max cells and dominates unless pruning holds; the null term is likely the largest single item and does
not multiply by B. **The laptop answer is this formula with Stage 0/1 numbers**, not an estimate here. Table to fill:

| quantity | M4 4 threads | M4 10 threads | A100 | source |
|---|---|---|---|---|
| T_epochs, g_pos, g_vis, T_b, T_summ, T_null | _to fill_ | _to fill_ | _to fill_ | Stage 0 / 1 |
| S₁, S₂, L₁, L₂, B, ESS | _to fill_ | | | Stage 3 on simulated populations |

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
| P_max | 4 × span | declared prior bound | inherited `span_factor` |
| a_lo, a_hi | 1, 30 mas | declared prior bound | inherited default; to move to resolution units (decision 12) |
| e_max | 0.9 | declared prior bound | inherited default |
| flux floor lo | data's 2σ contrast curve | derived | dehack §2I |
| π_out(p) radius | a_hi(1 + e_max) | derived | the prior's reach |
| π_out(v) | speed density of the prior at a random epoch | derived | §2.2, no constants |
| δ | 10⁻³ | tolerance | reporting precision (dehack §2) |
| W = ln(1/δ) | 6.9 | derived | from δ |
| peak keep rule | share of b_k ≥ δ | derived | §5 |
| season gap | P_cut | derived | §4.2 |
| edge window, smoothing, pilot size | ±½, ¼, M/4 | tolerance / budget (labels only) | §4.4 |
| radial chart | logistic in ln ρ | chart, no statistical meaning | §3.5 |
| s_w² (Luger move) | ⟨a²⟩/3 | derived | §6.3; a wrong value costs acceptance only |
| conditional ESS target | ½ | tolerance | Zhou et al. 2016; the adaptive rule |
| HMC acceptance target | 0.651 | tolerance | Beskos et al. 2013 |
| trajectory-length candidates | {½, 1, 2}× | tolerance | ESJD per gradient, Buchholz et al. 2021 |
| particles M | provisional 2048 | budget | set on simulated populations in 3a |
| moves per stage m | provisional 4 | budget | idem |
| replicates R | 2 | budget | the minimum that gives an error |
| ESS_target | provisional M/4 at β = 1 | budget | flag only |
| wall-time cap per band | declared per run | budget | failure, never a posterior at β < 1 |
| n_null | ≈ 1100 | derived from α = 0.01 and 30% relative error | §9 |
| α | 0.01 | declared | dehack decision 4 |
| n_pilot for edges | M/4 | budget | §4.4 |

Open for Ben: with (p, v) night summaries, P_night's role is only inside P_cut; the max(2Δt_min, P_night) rule stands
(Ben's decision) but P_night could now be dropped, since within-night motion is modelled. Not changed here.

## 11. Literature and provenance

Entries are in `references/provenance.toml`; verified against ADS (bibcodes) or Crossref (the statistics journals
ADS does not index).

| source | role here | verified |
|---|---|---|
| Luger, Foreman-Mackey & Hogg 2017, RNAAS 1, 7 | exact Gaussian conditional for linear parameters (§1.4, §6.3) | ADS 2017RNAAS...1....7L (existing entry) |
| Price-Whelan, Hogg & Foreman-Mackey 2017, ApJ 837, 20; Price-Whelan et al. 2020, ApJ 895, 2 | The Joker: collapsed linear parameters, label sampling (§6.4) | ADS 2017ApJ...837...20P (existing), 2020ApJ...895....2P |
| Lucy 2014, A&A 563, A126 | aliasing of sparse visual-binary arcs (§1.3, §4) | ADS 2014A&A...563A.126L (existing) |
| Del Moral, Doucet & Jasra 2006, JRSSB 68, 411 | SMC samplers, Ẑ (§6) | Crossref 10.1111/j.1467-9868.2006.00553.x |
| Jasra, Stephens, Doucet & Tsagaris 2011, Scand. J. Stat. 38, 1 | ESS-adaptive tempering (§6.1) | Crossref 10.1111/j.1467-9469.2010.00723.x |
| Zhou, Johansen & Aston 2016, JCGS 25, 701 | conditional ESS rule and Z error (§6.1) | Crossref 10.1080/10618600.2015.1060885 |
| Buchholz, Chopin & Jacob 2021, Bayesian Anal. 16, 745 | fixed-L HMC within SMC (§6.3) | Crossref 10.1214/20-ba1222, arXiv:1808.07730 |
| Beskos et al. 2013, Bernoulli 19, 1501 | HMC acceptance 0.651 (§6.3) | Crossref 10.3150/12-bej414 |
| Owen & Zhou 2000, JASA 95, 135 | defensive mixture bound on weights (§6.2) | Crossref 10.1080/01621459.2000.10473909 |
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
`aliases.period_cut`, shared by the engine and the vv harness. M4's `init=` oracle is unchanged.

## 13. Stage consequences

| stage | takes from this note |
|---|---|
| 2a | §3 (maps, chart, TI′ density, invariants, Campbell conversion); prior-recovery tests; register z, ζ as derived |
| 2b | §5 (night summaries, keep rule, v variance) |
| 3a | §6.1–6.3, 6.6 as generic, non-orbit SMC on BlackJAX; budgets of §10 |
| 3b | §4, §5 position-level target, §7 pruning measurement |
| 3c | §2 contract (5-D b_k), §6.2 bridge, §1.1 noise; #618's six tests as acceptance |
| 3d | §8 |
| 4 | the vv criteria file (committed before any run), §7 measurement, §9 table |
