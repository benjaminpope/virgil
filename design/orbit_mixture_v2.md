# Orbit night-outlier mixture, v2: global scale, continuation, exact outlier integral

Status: **approved by Ben 2026-10-29** (in chat, with option (a) of §7 PR D as recommended). Written 2026-10-29 (Opus design
pass). Coding may start.
Sonnet pr-workers do the coding (§7).

Base: `main` at 99067b6. Stage 2 (#586, merged at 2011949) introduced the
mixture. Companions: [orbit_dehack.md](orbit_dehack.md) §2A (the mixture
as designed) and [orbit_gate_042.md](orbit_gate_042.md) (the
pre-registered gate; §8 below lists the criteria changes this note asks
for).

## 0. Summary

Stage 2's honest test results showed five defects (brief of 2026-10-29).
Before designing anything I ran three small diagnostics (§1). They change
the diagnosis in two places:

- **Per-night noise scales do the outliers' job.** The inlier likelihood
  ℓ_k marginalizes a free noise scale *per block per night*. A night the
  orbit misses can inflate its own errors, so the miss costs
  (ν/2) ln(χ²_miss/χ²_fit) nats, not χ²/2. On the sidelobe fixture the bad
  night has χ² = 26 806 on ν = 27 and costs only 105 nats. That is why
  q stalls at 0.83 (defect 4) and why the inflation candidate cannot
  separate a night (defect 3): its support lies inside the inlier's own
  scale prior.
- **The outlier integral b_k is not converged.** The disc grid is far
  coarser than the per-night likelihood peak, so b_k is a lottery over
  where the nearest grid point falls. At the default step it is off by
  8–26 nats on the sidelobe fixture (§1.2). The docstring's "converged to
  about 1 nat" covers only the flux axis.
- **Defect 2 is mostly not a σ-level problem.** Per-night scales already
  absorb a σ that is 3.5× too small. The untempered arm fails because its
  correlated channels are left untempered (f ≈ 5): the orbit misses every
  night by the correlated error, and the untempered likelihood overstates
  each miss f-fold. On seed 0 the orbit scores ℓ ≈ −785 a night against
  b_k ≈ −784 to −815. Every night sits on the outlier boundary, the
  Hessian is not positive definite and no mode is returned (§1.3). A
  global scale does not change this. What v2 can do is report it honestly
  and stop the vacuous pass (§6).

The v2 design:

1. **Global noise scales.** One scale s_g per calibration group (observable
   kind × instrument), shared by all nights. Jeffreys prior (log-uniform)
   on [1/s_max, s_max]. **Marginalized**: ln s_g is appended to the orbit
   vector, so the optimizer, the Laplace proposals, importance-sampled
   evidence and NUTS integrate it (§2). The inlier loses its per-night
   scales. The outlier keeps them, because an outlier night's calibration
   is unknown as well as its companion's position.
2. **Deterministic ε-continuation** (graduated non-convexity). Every start
   is fitted first at ε = 0. A fixed ε then rises along an event-driven
   schedule in the threshold T = ln((1 − ε)/ε), in 1-nat steps while some
   night is in transition, until ε* = 1/(2K). The ε-marginalized target
   comes last. Only the target's value ever ranks or selects (§3).
3. **Sidelobe q derived**: q = 1/(1 + (2K − 1)e^{−Δ}) in the single-outlier
   limit, with Δ = b_k − ℓ_k. It reproduces the measured 0.833 (predicted
   0.826). Under v2, Δ ≈ 1.5 × 10⁴ nats on the fixture, so
   q = 1.000. The post-hoc q > 0.5 and 0.02 tolerances are replaced by
   derived bounds (§4).
4. **Inflation candidate repaired, not removed**: (iii′) inflates relative
   to the global scale, which makes it separable. This is a criteria
   change for Ben (§5, gate §3.6).
5. **Cost**: b_k by adaptive cubature, which converges where the present
   grid does not. Its level-0 sum is folded into the `epoch_positions`
   grid pass the pipeline already makes, and it is cached per
   (data, dof, s_max). Wall-clock targets are in §6.
6. **Four stacked PRs** for Sonnet workers (§7).

## 1. Diagnostics (run 2026-10-29, laptop, pyslot, 2 threads)

The scripts are in the session scratchpad. They are not committed: these
are reported measurements, not tests.

### 1.1 Sidelobe fixture (`tests/test_night_flags.py::_with_bad_night(3.0)`)

Four nights, each with ν = 27 (18 V², 9 CP). The mixture optimum (outlier
kind `mixture`, `s_max = 50`, b_k from the test helper at step 1.0 mas)
gives:

| night | χ²(V²) | χ²(CP) | ℓ_k | b_k (step 1.0) | q_k |
|---|---|---|---|---|---|
| 0 | 17.8 | 7.3 | −13.9 | −91.4 | 0 |
| 1 | 10.3 | 8.6 | −9.7 | −107.5 | 0 |
| 2 | 13.7 | 16.3 | −15.1 | −85.9 | 0 |
| 3 (bad) | 24 095.9 | 2 710.4 | −105.4 | −101.9 | 0.833 |

Night 3 misses by about 1000× its χ² budget, yet ℓ_3 is only 90 nats
below a good night: its free scale absorbs the miss. The closed form of
§4.1 with Δ = −101.9 + 105.4 = 3.5 and K = 4 gives
q = 1/(1 + 7e^{−3.5}) = 0.826, against 0.833 measured. The mixture behaves
as derived. The likelihood it is given is what makes the night ambiguous.

### 1.2 Convergence of b_k in the grid step

`broad_loglike` on the same four nights, with the 201-point axis cap
lifted, a_max = 12 mas (the true positions lie within 10 mas), 13 flux
nodes, resolution 4.21 mas:

| step (mas) | night 0 | night 1 | night 2 | night 3 |
|---|---|---|---|---|
| 2.0 | −101.81 | −102.98 | −118.21 | −119.95 |
| 1.0 | −88.25 | −104.34 | −82.80 | −119.57 |
| 0.5 | −78.73 | −97.83 | −84.20 | −114.19 |
| 0.25 | −67.73 | −90.37 | −80.40 | −109.20 |
| 0.125 | −63.30 | −90.14 | −75.56 | −107.16 |
| 0.0625 | −61.97 | −90.23 | −74.68 | −107.27 |
| 0.04 | −62.05 | −90.23 | −74.61 | −107.30 |

The value settles to about 0.1 nat only at step ≲ res/67. The last two
rows still differ by 0.07–0.08 nats, non-monotonically, so convergence
below 0.1 nat is not shown. The default step,
`_BROAD_STEP` × res ≈ 1.05 mas, is wrong by 26, 14, 8 and 12 nats, and the
error is not monotone (night 2 at steps 2.0 and 1.0). The flux axis was
held at 13 nodes, so flux convergence is not tested by this table. At
the production a_max = 57 mas a converged uniform grid would need about
1800² × 13 points a night, so a uniform grid is not an option (§6).

### 1.3 Untempered correlated noise (`tests/test_tempering.py::_nights`, ρ = 0.85, seed 0)

| arm | f | χ²/N | mode log L (6 nights) | b_k per night | q | result |
|---|---|---|---|---|---|---|
| tempered | 4.81 | 10.0 | −942 (−157/night) | −172 … −178 | 0 (all) | offset 1.35 |
| untempered | — | — | −4707 (−785/night) | −784 … −815 | not computed | `hessian-not-positive-definite`, `no-valid-mode`, offset inf |

The untempered orbit and the outlier component tie night by night, so the
mixture flattens the likelihood and the band loses its mode.
`med[False] > 2` then passes because inf > 2.

### 1.4 Cost

A warm `fit_orbit_aliases` call on seed 1 of the same fixture (settings
as in `_median_offsets`) takes 1.8 s, of which `broad_loglike` takes
1.44 s (80%). The calibration test makes 40 such calls per parametrization.

## 2. Global noise scales (brief item 1)

### 2.1 Model

Night k's inlier likelihood at orbit x and scales s:

  ℓ_k(x, s) = Σ_b dof_b · ln N(d_kb | m_kb(x), s_{g(b)}² Σ_kb),

with Σ_kb the quoted covariance of block b (V², CP, extras), dof_b = 1/f_b
the tempering, and g(b) the block's **calibration group**. CP blocks use
the von Mises form already in `_Surface.score`, with κ = 1/(sσ)². Blocks
with gains or offsets use C = s²D + UUᵀ, whitened once at the given s
(`_Surface._table`'s `at(log_s)` without the node loop).

- **Calibration group** = (observable kind, instrument name), so V² and CP
  get their own scales. Their errors come from different pipeline steps:
  V² from the calibrator transfer function, CP from phase noise. One shared
  s would settle between them and leave one kind's nights all looking bad,
  which is defect 2 in another form. Nights from different instruments
  (say GRAVITY and PIONIER) get their own groups, since miscalibration is
  a property of an instrument's error model. When `INSNAME` is absent the
  group is the kind alone. A worked case has two to four groups.
- **Per-night variation** in conditions is carried by the quoted errors,
  which follow seeing night by night, and by the outlier component. The
  inlier no longer has a per-night scale.

### 2.2 Prior: Jeffreys, proper, s_max

ln s_g is uniform on [−ln s_max, ln s_max]: the Jeffreys prior for a scale
(Jeffreys 1946), made proper by the bound. The bound's normalization,
−ln(2 ln s_max) per group, is the same in every band and in the null, so
it cancels in band probabilities. It also cancels in the detection FAP,
which simulates its nulls through the same fit. So s_max matters to the
global scale only through its support: a posterior that reaches the bound
is a reported failure (`scale-at-bound` when the posterior median of ln s_g
is within 3 posterior sd of ±ln s_max), never silent.

Default: s_max = 10, the engine's existing bound (`_NUMERIC_S_MAX`), so
there is no new constant. Today both `fit_orbit_aliases` and
`OrbitPipeline` default to `s_max=None`, so PR A changes both defaults,
and the seeding pass with them, and records the change in the CHANGELOG. It covers χ²/N up to 100. The user can raise it.

**Where s_max does matter: the outlier's own scales.** Under (ii) the
outlier component keeps one free scale per group per night, on the same
log-uniform prior (§2.4). That prior's width does not cancel between
inlier and outlier. It costs the outlier
≈ Σ_g ln(2 ln s_max · √(ν_g/π)) nats a night, which is 2.0 to 2.4 nats per
group at ν = 9–18 and s_max = 10. Raising s_max to 100 moves each night's
Δ by −J ln 2 (−1.4 nats at J = 2). That shift changes q only for nights
already within a few nats of the boundary. It is stated, tested
analytically (§7, PR B) and reported, not tuned.

### 2.3 Marginalize, not profile

**Marginalize.** ln s_g are appended to the orbit vector x (8 + J
coordinates, flat prior in these sampled coordinates like the rest):

- The **optimizer** maximizes jointly over (x, ln s): the joint posterior
  mode. That is the only sense in which s is "profiled", and only to find
  modes.
- **Laplace covariances, IS evidence and NUTS** include ln s, so the band
  evidence and the posterior integrate s exactly in expectation. The code
  needs nothing beyond longer vectors.
- **Why not profile:** (a) an evidence is an integral, and a profiled
  scale leaves out its Laplace width, which differs between models with
  different residual structure (orbit against null); (b) a profile inside
  the likelihood needs a nested optimizer and gradients through an argmax
  at every evaluation; (c) the posterior on x should carry the uncertainty
  in s, small as it is (sd of ln s ≈ 1/√(2ν_g,tot)).
- **Why not quadrature in the likelihood:** that is exact and keeps x
  short, but with J groups coupled through the per-night mixture terms it
  is a J-dimensional tensor rule (32^J nodes). Appended coordinates scale
  linearly in J.

Tempering composes cleanly. L^{1/f} with errors s σ peaks at
s² = χ²/N whatever f is, and its width in ln s grows by √f. So f carries
the correlation and s the level, and they do not compete.

### 2.4 The two outlier components with a global scale

- **(ii) companion anywhere:** b_k = ln of the mean over the prior disc
  |p| ≤ a_max(1 + e_max) and over log-uniform flux of the night's
  likelihood. It is evaluated with **the night's own free scales** (per
  group, log-uniform on [1/s_max, s_max]), i.e. today's per-night
  scale-marginalized surface. An outlier night is one the model knows
  nothing about: neither where the companion is nor how well calibrated
  the night is. As a result b_k is constant in (x, s): it is computed once
  per (data, dof, s_max), and a night with q → 1 has exactly zero pull on
  x. **s_max may no longer be None**: the closed-form −(ν/2) ln χ² profile
  is improper, and its constant would no longer cancel against a proper
  inlier density.
- **(iii′) inflation relative to the global scale:** see §5.

### 2.5 Reporting (house rule: raw χ²/N first)

`AliasResult` and the pipeline summary report, in this order, per band
and per group:

1. χ²/N on the **quoted** errors at the best orbit (over all nights, and
   over nights with q < ½);
2. ŝ_g: posterior median and 68% interval, from the NUTS draws or, without
   them, the Laplace mode ± sd;
3. the tempering factors f_b;
4. q_k per night.

On white data, ŝ_g² ≈ χ²_g/N_g over the inlier nights. A docs sentence says
that s ≫ 1 means the quoted errors are wrong, and that a rescaled χ² ≈ 1 is
then tautological.

## 3. Deterministic ε-continuation (brief item 2)

### 3.1 The family

One function with two node sets, so there is no second likelihood path:

  M(x, s; nodes) = logsumexp_j [ ln w_j + Σ_k logaddexp(ln(1 − ε_j) + ℓ_k, ln ε_j + b_k) ].

- **Target**: the Gauss–Chebyshev nodes of `group_outlier_mixture`
  (Beta(½, ½), exact).
- **Fixed ε**: one node, ε, with w = 1. ε = 0 gives Σ_k ℓ_k exactly. The
  code takes the ε = 0 node as that sum, not as logaddexp with −∞, so its
  gradient is finite.

`likelihood._outlier_terms` gains an optional `(log_eps, log_w)` node
argument, defaulting to the Gauss–Chebyshev set. ε enters as a traced
scalar, so one compilation serves every step.

### 3.2 Schedule (per start, deterministic)

Parametrize by the half-weight threshold T(ε) = ln((1 − ε)/ε). At fixed ε,
night k's outlier responsibility is r_k = σ(Δ_k − T), with
Δ_k = b_k − ℓ_k(x, s) and σ the logistic function.

1. **ε = 0**: optimize from the seed to (x₀, s₀). This is a plain Gaussian
   fit with global scales. Its residuals are not redescending, so the
   basin is the least-squares basin.
2. **Enter**: T₁ = max(max_k Δ_k(x₀, s₀) + W, T_end), where
   W = ln(1/δ) = 6.9 nats, δ = 10⁻³ is the reporting precision
   (orbit_dehack §2), and T_end is defined in step 5. At T₁ every
   r_k ≤ δ, so the first fixed-ε objective differs from ε = 0 by less than
   the reported precision. When every night fits (all Δ_k + W < T_end),
   T₁ = T_end and the schedule goes straight to step 5.
3. **Step**: while some night has |Δ_k(x_i, s_i) − T_i| ≤ W, set
   T_{i+1} = max(T_i − h, T_end) with **h = 1 nat**, and re-optimize from
   (x_i, s_i).
4. **Jump**: if no night is within W of T_i, set
   T_{i+1} = max(max{Δ_k : Δ_k < T_i − W} + W, T_end): the next night to
   enter transition, or T_end if there is none. No responsibility changes
   by more than δ across a jump.
5. **End**: when T_i = T_end = ln(2K − 1), i.e. ε* = 1/(2K), optimize the
   target (marginal) once, warm-started.

T therefore never increases and never passes T_end.

**Why h = 1 nat.** dr/dT ≤ ¼, so one step changes any night's weight by at
most ¼. Taking a night from inlier (r ≤ ¼) to outlier (r ≥ ¾) needs
ΔT ≥ 2 ln 3 = 2.2 nats, so every transition spans at least two
warm-started solves. No night flips within one solve, and the optimizer
never meets the cliff that stalled #586's starts.

**Why ε* = 1/(2K).** With one outlier among K nights, the marginal
mixture's posterior odds that the night is an outlier are
e^{Δ}/(2K − 1) (§4.1). The fixed-ε odds are ε/(1 − ε) · e^{Δ}, and they
are equal at ε = 1/(2K). The last fixed-ε stage therefore has the
target's half-weight point. When there are more outliers, the final
marginal solve moves the effective ε.

**Bound on work, and the cap.** If each Δ_k were fixed, every night's
transition window (width 2W) would take at most ⌈2W/h⌉ + 1 = 15 steps,
so there would be at most 15K steps in all. But Δ_k is recomputed after
every solve. A night whose Δ_k falls with T (for example a good night
while ŝ relaxes) can stay in its window. In that case no bound follows
from the schedule alone, and the worst case is (T₁ − T_end)/h, which is
about 10⁴ for a sidelobe night.

Therefore **N_cap = 15K steps**, the static bound itself. A start that
reaches N_cap with T > T_end jumps to T_end, runs step 5, and is flagged
`continuation-capped`. The flag is counted in the band's result and in the
summary, and is never silent. The cap is not tuned: exceeding the static
bound means a night's status is still tracking the fit, i.e.
continuation has not settled, and that is exactly what the flag reports.
M4 measures whether capped starts cost search completeness.

Typical cost is two to three solves (ε = 0, one jump, the target). Each
step is a warm-started Gauss–Newton solve on an already compiled kernel,
measured against the targets in §6.

**Worked case (the sidelobe fixture).** At ε = 0 the orbit follows the bad
night partway, and the global scale inflates to fit it: ŝ ≈ √(26 806/108)
≈ 16, clipped at s_max = 10. The good nights' Δ rise towards 0,
but the bad night's Δ is still the largest by far, so it enters
transition first. As it detaches, ŝ returns to about 1 and the good nights'
Δ fall back to ≪ 0. Because Δ_k is recomputed at every step, this
ordering is what the schedule does, not an assumption.

### 3.3 What continuation is not

- It is **not the deleted plain-then-polish path**. That path ranked and
  discarded candidates under the wrong (no-outlier) model and then polished
  the survivors. Here every start runs the whole schedule, and the modes
  are ranked, deduplicated and kept **only by the target's value** at the
  end. No ε < target value is ever read for selection. A test checks this
  (§7, PR C).
- It is not random restarts, and has no fallback or `except`. A start whose
  ε = 0 solve is not finite returns −inf, as today.
- Its residual risk is that of graduated non-convexity in general: a bad
  night that drags the ε = 0 fit so far that a good night becomes the worst
  misfit. M4 (search misses against the oracle) measures that. It is never
  patched with a heuristic.

Sources: graduated non-convexity, Blake & Zisserman (1987); its use with
robust mixtures, Yang et al. (2020), IEEE RA-L 5, 1127 (GNC). Both need
`references/provenance.toml` entries in PR C.

## 4. The sidelobe night: derived q and tolerances (brief item 3)

### 4.1 q in the single-outlier limit

Let night o have Δ = b_o − ℓ_o, and let the other K − 1 nights have
Δ_k ≪ 0. Then their factors are (1 − ε)e^{ℓ_k}, and

  P(o outlier) / P(o inlier) = e^{Δ} · E[ε(1 − ε)^{K−1}] / E[(1 − ε)^K]
                             = e^{Δ} · B(3/2, K − ½) / B(½, K + ½)
                             = e^{Δ} / (2K − 1),

with the expectations over Beta(½, ½). So

  **q = 1 / (1 + (2K − 1) e^{−Δ}).**

This is an analytic oracle that does not depend on the Gauss–Chebyshev
quadrature. In general (good nights not ≪ 0) the oracle is the 2^K
enumeration with Beta moments B(½ + n_out, ½ + K − n_out)/B(½, ½), already
in `tests/test_group_outlier.py`.

q > 1 − δ needs Δ > ln((2K − 1)/δ), which is 8.9 nats at K = 4.

### 4.2 What q does a sidelobe night reach under v2?

The global scale is fixed by the good nights: χ²/ν = 74/81, so ŝ ≈ 0.95.
At ŝ the bad night's inlier term is ≈ −26 806/(2 × 0.91) ≈ −14 700 nats,
while its own-scale b_o is ≈ −110 (§1.2 converged −107.3 at a_max = 12,
less ln(57²/12²) = 3.1 for the production disc). So Δ ≈ 1.46 × 10⁴ and
q = 1 − 7e^{−14 600}, i.e. **1.000 to double precision**. The night's pull
on x is (1 − q) times its inlier pull: zero.

The same holds for any sidelobe night whose miss exceeds its χ² budget by
more than ln((2K − 1)/δ) + Occam(b) nats, a few tens of nats. That is
the generic case, because the miss of a sidelobe is a fringe spacing,
while the budget is ν.

### 4.3 Tolerances: decisions

| assertion | at #586 | v2 | derivation | changed after seeing a result |
|---|---|---|---|---|
| bad night q | > 0.99 (pre-#586) → **> 0.5** | **> 1 − δ = 0.999**, and equal to the closed form of §4.1 at the fit's Δ to 10⁻⁹ relative | δ is the reporting precision; §4.2 predicts Δ ≈ 10⁴ | **yes** (the q = 0.69–0.83 results and §1.1 were seen) |
| good nights q | < 10⁻⁶ | < 10⁻⁶, unchanged, plus the enumeration oracle | §1.1: Δ_good ≈ −70 to −100, so q ≈ 10⁻³¹ (measured 10⁻³¹) | no |
| near vs far orbit | 10⁻³ (pre-#586) → **0.02** (raw units) | **\|x_near − x_far\|_i ≤ 10⁻³ σ_i**, with σ_i the Laplace posterior sd at x_clean, for every coordinate including ln s | with q = 1 the objectives of near, far and clean differ by constants in x, so their argmaxes agree to the optimizer precision (_DTOL = 10⁻⁹ nats ⇒ ≤ 4.5 × 10⁻⁵ σ_i). 10⁻³ σ_i shifts a 95% interval's coverage by < 10⁻³ (δ again) | **yes** (0.02 was set after a measured 8 × 10⁻³) |
| near vs clean | (none) | the same 10⁻³ σ_i bound | as above | no (new) |
| near vs off | < 0.5 × off | unchanged | the comparative check that the night does not lead | no |

The raw-unit 10⁻³ is not reverted literally. The coordinates have
different scales (days, degrees, mas, log flux), so posterior sd is the
scale-free unit.

## 5. Replacing the shared-noise-inflation candidate (brief item 4)

**Criteria change: needs Ben's approval before any D run.**

### 5.1 Why (iii) failed

As registered, (iii) inflates each block's noise by s ∈ [1, s_max] on top
of an inlier that already integrates its own s over [1/s_max, s_max]. The
outlier's support lies inside the inlier's, so B/L ≤ 2^{n_blocks}
(the `inflated` twin adds ln 2 per block), and no night can be separated.
That is the #586 review finding, and the strict xfail confirms it.

### 5.2 (iii′): inflation relative to the global scale

The outlier component of night k is that model's own night likelihood with
each group's scale inflated beyond the global one:

  ℓ_k^out(x, s) = Σ_g ln ∫₁^{s_max} L_kg(x; s_g t) dt / (t ln s_max).

This is the registered (iii), with t log-uniform on [1, s_max] (the
existing `_Surface.inflated` twin), applied to the v2 inlier, which no
longer has per-night scales. No new constant enters. The null stays a
single star, and an outlier night costs both models alike.

**Separability, derived.** For a good night (χ² ≈ ν s²), the integrand
at t = 1 has slope ≈ 0 in ln t and curvature ≈ −2ν_g. The integral is a
half-Gaussian, so Δ_good ≈ Σ_g ln(½√(π/ν_g) / ln s_max): −2.4 and −2.1
nats for ν = 18 and 9 at s_max = 10. That gives q_good ≈ 1.7 × 10⁻³ on the
K = 4 fixture. An inflation component always grants a good night some
weight. That is a property of (iii′), not a defect, and its tests use the
enumeration oracle instead of the (ii) bound of 10⁻⁶. For the sidelobe
night at t = s_max the inflated term is ≈ −195 nats against −14 700, so
q = 1.000. Unlike (ii), the outlier night still pulls x, with weight
≈ max(1/s_max², ν s²/χ²) ≈ 0.01. The near/far bound of §4.3 therefore does
not apply to (iii′). Its test keeps the comparative near-vs-off check and
reports the measured pull against the predicted 1/s_max².

### 5.3 Gate rule

§3.6 is unchanged in structure: run both (ii) and (iii′) on D at the
commit that has both, require M6 and M8, and break an exact tie towards
(iii′). Only the definition of candidate (iii) changes, along with the YAML
name `shared_noise_inflation` → `global_scale_inflation`. The gate change
log records it (gate §9).

Removing (iii) altogether and adopting (ii) without the D comparison was
considered and **not recommended**. The D rule exists because (ii)'s null
("a companion anywhere every night" as ε → 1) is the M8 risk at K = 4–5,
and only a comparison measures that.

## 6. Cost (brief item 5)

### 6.1 Exact b_k by adaptive cubature

The integrand has peaks of width σ_p ≈ res/(2π · SNR), often below res/100,
on a disc of radius a_max(1 + e_max), 57 mas by default. A uniform grid
cannot integrate it (§1.2). The replacement:

- **Level 0**: the existing grid (step res/4, the fringe-envelope rule of
  `_BROAD_STEP`, with 13 log-flux nodes) over the disc, as boxes in
  (x, y, ln f). The 201-per-axis cap is **deleted**: on long baselines it
  coarsened the step past the fringe Nyquist rate.
- **Find every peak first.** A midpoint value is not a bound on a box's
  integral: a peak of width res/100 can sit about 680 nats above the
  midpoint of its own res/4 box (§1.2 shows the effect). Refinement is
  therefore seeded from peaks, not from midpoints. The χ² of each block
  is smooth on the fringe scale: V² of a point companion is a sum of
  cos(2π u·p) with |u| ≤ 1/res, so it is band-limited, and the res/4
  step samples it at twice its Nyquist rate. Every local minimum of χ²
  then lies within one cell of a local minimum of the level-0 grid
  (3 × 3 × 3 neighbourhood). Newton on the night's score, started from
  each such grid minimum (the peak catalogue `epoch_positions` already
  builds), finds every peak. For CP, which is smooth but not strictly
  band-limited, this is checked by a planted-peak test (§7, PR A), not
  assumed.
- **Integrate each peak** by adaptive midpoint cubature in its
  Laplace-whitened coordinates. The cubature box is ±9 Laplace sd, the
  `_WINDOW` already used for the scale integral. Boxes are split 2 × 2 × 2
  until the difference between a box's midpoint rule and its children's
  sum, totalled over the night, is ≤ 4δ = 0.004 nats.
- **Background**: the level-0 boxes outside every peak's ±9 sd box, by
  their midpoints. The χ² gradient bound (Bernstein's inequality for the
  band-limited V² part: |∇χ²| ≤ (2π/res)·max|χ² − χ̄²| over the box's
  neighbours) bounds each box's maximum. A background box whose bound lies
  within W_b = 30 nats of the largest peak is also refined.
- **Why 4δ:** |dq/dΔ| ≤ ¼, so a b_k error of 4δ moves any q by at most δ.
  **Why 30 nats:** the level-0 grid has at most
  (2 × 57/1.05)² × 13 ≈ 1.6 × 10⁵ ≈ e^{12} boxes. Background boxes whose
  whole-box bound is more than 30 nats down sum to at most
  e^{12 − 30} = e^{−18} of the peak, far below 4δ.
- **The guarantee is conditional** on the peak catalogue being complete.
  The planted-peak test checks completeness and reports any miss raw.
- **Source:** adaptive subdivision cubature (Genz & Malik 1980, J. Comput.
  Appl. Math. 6, 295), restricted to midpoint rules so it runs on the
  existing batched scorer (`_grid_scores`). It needs a provenance entry.

### 6.2 Same answers, cheaper

1. **Fold level 0 into `epoch_positions`.** The pipeline already scores
   this exact grid (same step, same 13 flux nodes, same surface, same dof)
   for seeding. One extra masked logsumexp in that pass returns the
   level-0 disc sum per night, stored on `EpochPositions` with a
   fingerprint (data hash, dof, s_max, a_max, flux_range, step).
   `fit_orbit_aliases` reuses it when the fingerprint matches and
   otherwise computes it. The seeding grid must cover the outlier disc:
   seeding must cover the prior's support anyway, since a companion
   outside the grid cannot seed. PR A makes the pipeline's field at least
   a_max(1 + e_max).
2. **Cache b_k per fingerprint** within a process, so a resume, an oracle
   run and the null's (ii) component do not recompute it.
3. **Closed-form own-scale marginal** for Gaussian blocks of b_k. This
   replaces the numeric `log_integral` per grid point; von Mises and
   numeric blocks keep theirs. For a block tempered by d = dof_b (so
   L^d has ν′ = dν and χ′² = dχ²), with the prior normalized on
   [a, b] = [1/s_max, s_max], u = χ′²/(2s²) and x_a = χ′²/(2a²),
   x_b = χ′²/(2b²):

     ln ∫ L^d π(s) ds = −d(Σ ln σ + (ν/2) ln 2π) − ln(2 ln s_max)
                        + ln ½ + ln Γ(ν′/2) + (ν′/2) ln(2/χ′²)
                        + ln[Q(ν′/2, x_b) − Q(ν′/2, x_a)].

   The last term is evaluated without cancellation:
   - when x_b > ν′/2, as ln Q_b + log1p(−Q_a/Q_b);
   - otherwise as ln P_a + log1p(−P_b/P_a).

   For a badly missed grid point (the sidelobe V²: χ² = 24 096, ν = 18,
   x_b ≈ 120 ≫ 9) both P round to 1 and the naive difference is −∞. If
   Q_b underflows (float32), ln Q comes from its large-x asymptotic
   series (Abramowitz & Stegun 6.5.32).

### 6.3 Pre-registered targets

Measured on Ben's laptop with 2 threads, warm, by
`scripts/time_orbit_mixture.py` (new in PR A, which prints a table and
writes no files). The baseline is main at 2011949:

| quantity | baseline | target |
|---|---|---|
| b_k accuracy against a Richardson-extrapolated fine-grid oracle (§7, PR A) | 8–26 nats off | ≤ 4δ = 0.004 nats |
| b_k time per warm fit, positions from `epoch_positions` | 1.44 s (80%) | ≤ 5% of the warm fit |
| b_k time per warm fit, hand-built positions (the tempering tests) | 1.44 s | ≤ 1.44 s, converged (level 0 unchanged; refinement ≤ 0.2 s) |
| `test_the_tempered_posterior_is_calibrated_and_the_untempered_one_is_not`, per parametrization, warm, including continuation | measured in PR A | ≤ 1.25 × baseline |
| `test_the_mixture_keeps_the_optimum_from_starts_near_the_truth` | measured in PR A | ≤ 2 × baseline (continuation adds solves) |

A miss is reported raw in the PR, never met by coarsening the cubature
tolerance or the schedule.

## 7. Staged PRs for Sonnet workers

Every PR branches from current `main`, merges `main` (never rebases),
runs the touched tests plus the fast suite with
`~/.claude/bin/pyslot .venv/bin/python -m pytest …`, puts long tests in
the slow set, uses Oxford -ize spelling and terse docstrings with one
`Source:` line, adds provenance entries, and answers the AGENTS.md
reviewer checklist item in its description. Every PR description
**quotes these pre-registered criteria verbatim and reports the results
raw**, failures included. A test that fails is reported, never loosened
or xfailed without Ben's sign-off.

### PR A: exact outlier integral (independent; can start first)

- **Files:** `src/virgil/epochs.py` (`broad_loglike` → adaptive cubature;
  delete `_MAX_AXIS`; closed-form own-scale marginal; level-0 disc sum in
  the `epoch_positions` pass with its fingerprint); `src/virgil/aliases.py`
  (`_broad_components` reuses or caches by fingerprint; the default
  `s_max` of `fit_orbit_aliases` becomes `_NUMERIC_S_MAX`, and an explicit
  `s_max=None` raises for `outlier="mixture"`);
  `src/virgil/pipeline/orbit.py` (default `s_max` = `_NUMERIC_S_MAX`, also
  for the seeding `epoch_positions` pass, so the level-0 fingerprint
  matches; grid field ≥ a_max(1 + e_max)); `CHANGELOG.md` (seeding scores
  and summaries change for every caller who does not pass `s_max`);
  `scripts/time_orbit_mixture.py` (new); `references/provenance.toml`
  (Genz & Malik 1980).
- **Tests to add:**
  - `test_the_broad_component_converges_to_a_fine_grid`: on a small disc
    (a_max = 6 mas, one night simulated with its companion at 3 mas), the
    oracle is written in the test with numpy from the surface's public
    score. It uses midpoint grids at h, h/2 and h/4 (h = res/64), with the
    flux axis integrated by Gauss–Legendre in ln f, doubling nodes until a
    change < δ/10. The test first asserts the oracle's own Richardson error
    estimate |R(h/2, h/4) − R(h, h/2)| ≤ δ, where
    R(h, h/2) = (4I_{h/2} − I_h)/3. Only then does it assert
    |b_k − R(h/2, h/4)| ≤ 4δ = 0.004 nats. If the oracle check fails, the
    test fails and is reported, never loosened. (§1.2's last rows differ by
    0.07–0.08 nats non-monotonically, so a two-level oracle is not enough.)
  - `test_a_planted_narrow_peak_is_found`: a night whose companion sits at
    the corner of a level-0 box (the worst case), at SNR giving
    σ_p ≈ res/200. Assert that the catalogue contains it and that b_k
    matches the oracle above.
  - `test_default_calls_use_a_bounded_scale`: `fit_orbit_aliases` and
    `OrbitPipeline` with no `s_max` run with `_NUMERIC_S_MAX`, and
    `s_max=None` with the mixture raises.
  - `test_the_broad_component_does_not_depend_on_the_grid_origin`: shift
    the level-0 grid by half a step and assert the same b_k to 0.004 nats.
    This is the aliasing lottery of §1.2, closed.
  - `test_the_own_scale_marginal_matches_quadrature`: the incomplete-gamma
    form against `scipy.integrate.quad` over ln s, to 10⁻¹⁰ relative, with
    cases χ² ≈ ν, χ² ≫ ν s_max² (finite, not −∞), and dof ≠ 1; plus a
    float32 case at the underflow end.
  - `test_the_level_zero_sum_is_reused_only_on_a_matching_fingerprint`.
- **Unchanged and must pass:** `tests/test_group_outlier.py` (all three);
  `test_the_broad_component_is_a_prior_average_of_the_night_likelihood`
  (its flux-refinement check); `test_the_outlier_component_is_built_per_kind_and_must_be_finite`;
  all of `tests/test_orbit_search.py`.
- **Report:** §1.2's table recomputed by the new code, and the §6.3 timing
  rows.

### PR B: global scales and (iii′) (after A)

- **Files:** `src/virgil/epochs.py` (`_Surface.score_at(chi2, log_s, dof,
  data)`: the existing integrand at a given ln s, Gaussian, von Mises and
  numeric); `src/virgil/aliases.py` (calibration groups; ln s_g appended
  to x with bounds ±ln s_max; `_night_terms` uses `score_at` for the
  inlier and, for `inflation`, the inflated twin relative to s; Laplace,
  IS and NUTS over the longer vector; `AliasResult.scale` and
  `scale-at-bound`; summary order of §2.5); `src/virgil/pipeline/orbit.py`
  (summary); `docs/conventions.md` (`noise`: one sentence on global scales).
  Delete the per-night inlier scale path in `aliases.py`. Per-night scales
  stay in `epoch_positions` and the tempering pilots, where each night is
  fitted alone.
- **Tests to add:**
  - `test_the_global_scale_posterior_is_the_inverse_gamma` (ε = 0, white
    Gaussian, J = 1): the NUTS or Laplace posterior of s² against the
    analytic inverse-gamma of χ² at the fixed best orbit, median to 1%.
  - `test_miscalibrated_white_errors_inflate_the_scale_not_the_outliers`:
    white noise at 3.5 × the quoted σ on every night, f = 1. Assert ŝ
    within its 99% interval of 3.5, and max_k q_k < 10⁻⁶. The v2 version of
    the brief's item 1, on data where it applies.
  - `test_a_v2_or_cp_miscalibration_moves_only_its_own_scale` (V² at 3 ×,
    CP at 1 ×).
  - `test_the_outlier_own_scale_prior_costs_its_closed_form` (§2.2
    sensitivity: b_k shifts by −J ln(ln s′/ln s) between s_max = 10 and 100
    when the own-scale posterior is interior, to 0.01 nats).
  - `test_scale_at_bound_is_reported`.
- **Tests changed (Ben signs off in the PR):**
  - `test_one_sidelobe_night_cannot_move_the_orbit`: the §4.3 assertions.
    `_broad` calls the production b_k (PR A), not a step-1.0 grid.
  - `test_one_sidelobe_night_cannot_move_the_orbit_with_inflation`:
    **un-xfail**, with the (iii′) assertions of §5.2 (enumeration oracle
    for every q, bad night q > 0.999, near-vs-off check unchanged, pull
    reported against 1/s_max²).
  - `test_band_evidences_are_no_lower_than_with_the_old_optimizer`
    (`_OLD_LOG_Z`): the pinned values belong to the per-night-scale
    likelihood, a different model, and pinning outputs from a past
    virgil run is the oracle that AGENTS.md rule 3 forbids. Replace it
    with `test_band_evidences_are_no_lower_than_from_the_truth`: per band,
    the blind log Z ≥ the log Z of the same fit seeded at the truth
    − (1 nat + 3√(SE₁² + SE₂²)). That is gate M4's rule in miniature.
  - `_problem` and `_optimum(off=True)`: the −10⁶ outlier level becomes the
    ε = 0 node set (PR C), or, in PR B, ε fixed at 0 through the same
    node argument of §3.1, which PR B introduces for this purpose.
- **Unchanged and must pass** (fixtures may append ln s = 0 to hand-built
  x vectors, with no other edit): `tests/test_group_outlier.py`;
  `test_tempered_nights_the_orbit_fits_are_not_outliers`;
  `test_fit_orbit_aliases_reports_each_nights_outlier_probability`;
  `test_outlier_eps_is_deprecated_and_ignored`;
  `test_the_problem_value_is_the_untempered_value_over_f`;
  `test_temper_false_reproduces_the_untempered_evidence`;
  `test_tempering_turns_an_overconfident_band_probability_into_a_calibrated_one`;
  the tempering estimator tests (`test_white_noise_gives_f_within_five_percent_of_one`
  through `test_temper_factors_normalizes_by_night_scale`);
  `tests/test_pipeline_orbit.py::test_summary_search_section_round_trips_outliers_and_night_flags`.

### PR C: ε-continuation (after B)

- **Files:** `src/virgil/likelihood.py` (`_outlier_terms` node argument,
  §3.1); `src/virgil/aliases.py` (`_continue(problem, bounds, x0)`
  implementing §3.2, called for every start where `_optimize` is called
  today; Δ_k from `_night_terms`); `references/provenance.toml`
  (Blake & Zisserman 1987; Yang et al. 2020).
- **Tests to add:**
  - `test_the_fixed_eps_mixture_is_the_one_node_case`: ε = 0 equals Σℓ_k
    with a finite gradient; fixed ε against direct evaluation.
  - `test_the_schedule_is_deterministic_and_bounded`: on random static
    (ℓ, b), including the all-good case where T₁ = T_end, assert that T is
    non-increasing, never below T_end, ends at ln(2K − 1), takes at most
    15K steps, and repeats identically.
  - `test_a_drifting_night_hits_the_cap_and_is_flagged`: a constructed
    problem whose Δ_k falls by h per solve stops at N_cap = 15K, runs the
    target, and carries `continuation-capped`.
  - `test_eps_star_matches_the_marginal_single_outlier_odds`: at
    ε = 1/(2K), the fixed-ε q equals §4.1's closed form for every Δ.
  - `test_only_the_target_value_ranks_modes`: a mode's reported log L
    equals `group_outlier_mixture` at its x.
- **Tests changed:** `test_the_mixture_keeps_the_optimum_from_starts_near_the_truth`:
  **un-xfail**, and the tolerance goes from 10 nats to **1 nat**. Why 1 nat:
  gate M4's threshold, the smallest shortfall that moves a reported band
  probability noticeably. Run it for both (ii) and (iii′). Changed after
  seeing a result: **yes** (the −309/−169 xfail was seen); this is a
  tightening.
- **Unchanged and must pass:** everything listed for PR B.
- **Prediction, registered here:** all four perturbed starts reach the
  truth-start optimum to within 1 nat. If any does not, the PR reports the
  values raw and stops. The fix is never extra starts.

### PR D: the calibration test made meaningful, and timings (after C)

- **Files:** `tests/test_tempering.py`; `tests/test_night_flags.py`
  (helper cleanup); the `scripts/time_orbit_mixture.py` report.
- **`test_the_tempered_posterior_is_calibrated_and_the_untempered_one_is_not`:**
  - **Must hold:** every seed returns a finite offset in **both** arms. A
    missing mode is a failure, never a pass, which closes the vacuous
    `inf > 2`. Tempered: `med[True] < 1.45` and the f check, unchanged.
  - **Untempered arm (criteria change: Ben signs off).** §1.3 shows that
    with the mixture, an untempered fit on correlated noise does not make
    the posterior too narrow, the behaviour `med > 2` tests. It turns
    every night into an outlier. Two pre-registered options:
    - **(a), recommended:** a two-way negative control. The untempered fit
      is visibly wrong if `med[False] > 2` (overconfident) **or** the
      median over seeds of the fraction of nights with q > ½ is ≥ ½ (the
      misfit is flagged). The test prints which held. Both outcomes are
      failures a user sees in the summary, and neither is vacuous once
      finiteness is required. The threshold 2 is unchanged: the calibrated
      median Mahalanobis offset in 2-D is √(2 ln 2) = 1.18, and √f ≈ 2.3
      times that is 2.7.
    - **(b):** drop the untempered arm, since `temper=False` on correlated
      data is a misuse that the tempering flag already reports.
  - **Prediction, registered here:** with v2, the finiteness requirement
    holds in the tempered arm. In the untempered arm it may fail: when
    every night ties with b_k, the likelihood is flat and the band may have
    no mode (§1.3). If so, the PR reports it raw and Ben chooses. Giving a
    flat band an evidence by prior sampling instead of a mode is a separate
    proposal (§9), not part of this PR.
  - Changed after seeing a result: **yes** (§1.3).
- **Report:** every §6.3 row against its baseline.

### Order

A → B → C → D, stacked. A touches only the outlier integral and can run
alone. B, C and D all touch `aliases.py` and must be sequential (two
writers never share files).

## 8. Criteria changes requested in orbit_gate_042.md

Each needs Ben's approval before any D or scored run. All are entered in
the gate change log (§9 there) in this PR, marked as proposed.

1. **§3.6 candidate (iii) → (iii′)**, as in §5 here. Changed after seeing
   a result: **yes** (the #586 xfail).
2. **YAML block repair**: `null_choice` sat between two `must_hold`
   entries at top-level indentation, so the block did not parse
   (`yaml.safe_load` fails at `M4_search_misses`). The harness hashes and
   reads this block, so it could not have run. Moved to top level, and the
   candidate renamed. Changed after seeing a result: **no** (structural).
3. **Stratum A draws the global scale from its prior.** SBC is exact only
   when every parameter the fit infers is drawn from the fit's prior.
   v2's fit infers s_g with ln s_g uniform on ±ln s_max, so A's true noise
   becomes s_g × the quoted σ, with s_g drawn from that prior, one per
   group. s_g is reported in the SBC ranks (not a tested p-value, so the
   Holm family is unchanged). Changed after seeing a result: **no** (it
   follows from the model change; no gate run exists).
4. **Showcases and summaries report ŝ_g after raw χ²/N** (§2.5). Changed
   after seeing a result: **no**.

## 9. Open items (not in this plan)

- **A band with a flat likelihood** (every night tied with the outlier
  component) has no Laplace mode, and its evidence is reported as failed.
  Its evidence could come from importance sampling with the prior as
  proposal. That would be a proposal to Ben once PR D's result is known.
- **Stratum A and the outlier component.** A has no contaminants because
  a bad night has no generative model inside the prior. Under (iii′) an
  inflated night *is* generative (t ~ log-uniform on [1, s_max] with rate
  ε ~ Beta(½, ½)). If (iii′) wins on D, A could draw from it, which would
  make the mixture itself SBC-testable. Decide after the D run.
- The `_NUMERIC_S_MAX` = 10 default is the engine's existing bound, kept
  to avoid a new constant. A data set with χ²/N > 100 hits it and is
  flagged.

## References

- Blake & Zisserman 1987, *Visual Reconstruction*, MIT Press (graduated
  non-convexity).
- Genz & Malik 1980, J. Comput. Appl. Math. 6, 295 (adaptive cubature).
- Hogg, Bovy & Lang 2010, arXiv:1008.4686 (mixture models for outliers).
- Jeffreys 1946, Proc. R. Soc. A 186, 453 (Beta(½, ½) for a rate; 1/s for a
  scale).
- Talts et al. 2018, arXiv:1804.06788 (SBC).
- Yang, Antonante, Tzoumas & Carlone 2020, IEEE Robot. Autom. Lett. 5,
  1127 (GNC with robust costs).
