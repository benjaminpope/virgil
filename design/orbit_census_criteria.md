# Orbit census criteria (committed before any census run)

Pre-registration for `scripts/orbit_population.py` (the paired, complete census of the position-level cell fit). The
code assigns the classes below exactly as written. Thresholds come only from existing constants: `W`, `ESS_FAIL` and
the §14 assertion 4 tolerance `4·sd + 0.01` (`design/orbit_alias_cycles.md`). A change after a run is a recorded
decision in git, never a loosening after a failure.

Source: the plan of 2026-11-04 ("critical review and a measure-first plan"), taxonomy section, reproduced verbatim.

## Two-level taxonomy (precedence error > search miss > proposal miss > sentinel false positive > support)

- **Per system:** retrieved / missed / failed / error. A truth that loses to an alias is a *miss*, and the binomial
  99% rule absorbs it. There is no "correct behaviour" class.
- **Per flagged cell, true or other cell:**
  - *search miss*: the oracle-augmented ln Ẑ exceeds the unaided ln Ẑ by more than 4·sd + 0.01, the §14 assertion 4
    tolerance;
  - *proposal miss*: a found mode is in the truth's basin, meaning LM from it converges to the truth-started optimum,
    yet the IS ESS is below ESS_FAIL;
  - *sentinel false positive*: the Laplace exceeds ln Ẑ by more than W while the oracle-augmented ln Ẑ agrees with the
    unaided one (§6.10 kind 2);
  - *support*: a non-finite b_k or target.

## Strata (§14)

B″ (every night an outlier); ≤ m inliers left; the rest (informative). Report-only: "≤ 2 outlier nights".

## Gating

§14 assertions 1–8 stand as committed, assertion 2 included (no cell-at-floor failures), except as amended on
2026-11-16 below: assertions 1, 2, 5 and 7 gate in the detectable stratum only (assertion 5 outside it is report-only),
and assertions 1 and 3 are scored in the sky plane. Rows of failed systems carry `pit_diagnostic=1` and never enter a gated SBC. The oracle columns are diagnostics:
they never touch `p_pos` or the status.

## Persistence and the matched sample (added 2026-11-04, before the baseline run)

Paired seeds reproduce each system's data, not the search, which changes as fixes merge. With `--persist` the driver
writes, per system, one npz next to the CSV holding, for each persisted cell: every mode (chart `x`, Laplace
covariance, `log_target`, inlier count, the night labels and the inlier set at `x`), the importance-sampling draws `x`
and their `ln w`, the cell's JAX key and the git hash.

**Persisted cells (rule fixed here, from the plan, not from data):** every flagged cell; the true cell; and the
unflagged cell of largest p̂. The last is the matched sample: same data, same `K`, same `two_seasons` by construction.

## Block variance of ln w (diagnostic columns, per persisted cell)

Take the highest-weight draw `x*` of the cell's IS and its mixture component `m*` (largest responsibility). With a
fixed 200 redraws each: **z block** freeze θ = (w_P, y_ζ, w_f, w_s) at `x*` and redraw (y_z, w_a) from the conditional
of component `m*`; **θ block** freeze z and redraw θ. Columns `var_lnw_z`, `var_lnw_theta` (`ln w = ln target − ln q`,
the full mixture). Diagnostics only: they never enter `p_pos`, the status or the verdict.

## Decision rule for the collapsed Thiele–Innes plan (approved by Ben 2026-11-04)

Build the collapsed-TI′ proposal only if the TI′ (z) block carries ≥ 50% of Var(ln w) in a failure class with ≥ ~5%
of informative systems, or the banana measure (‖μ(ρ) − linear fit‖ in conditional-sd units along a flagged mode's ρ
extent) exceeds 1 there; otherwise the fix belongs to the θ-chart and ridge work (§6.10, §6.12).

## Quadrature references

Any quadrature reference must show its integrand at the box edge ≤ e⁻¹⁰ of its peak, a deterministic inner integral,
and tolerance 4·√(sd² + sd_ref²) + 0.01.

## Platform drift (added 2026-11-05)

Mode counts drift across platforms (macOS 9 vs Linux runner 10, seed 0 system 13); census arms are paired only within
one platform (OzSTAR), never across.

---
## Arm: predictive C-step labels (#667, stacked on W1 #642), added 2026-11-07 before any run of this arm

The held-out W1 result (paired with baseline f71294b, systems 200–399) was read in aggregate on 2026-11-07, including a post-hoc split by ln s. Systems 200–399 are therefore no longer clean for deciding W1 or this arm. The decision set is a fresh paired set, systems 400–599 of seed 0 (`START_BASE=400`), which nobody has inspected.

- **Arms on 400–599:** baseline f71294b; W1 2673182; this arm (its pinned head). All three run on OzSTAR with the same image.
- **Primary, for each of W1 and this arm against the baseline:** failure (status failed or error) in the informative stratum "rest", exact one-sided McNemar on discordant pairs, α = 0.01.
- **Harm:** failure in each other stratum, and misses (ok with the truth outside the 99% set). Exact two-sided McNemar; harm if the arm is worse at p < 0.05.
- **Merge rule:** primary passes and no harm.
- **Secondary, recorded as a hypothesis from the W1 analysis, report-only:** this arm against W1, rest stratum, baseline ln s < −0.5, one-sided McNemar at α = 0.05.
- **Report-only:** this arm on 200–399 against the baseline and W1 there.
- **Systems 400–599 are aggregate-only,** like 200–399 before them.

**W1 merge rule (added 2026-11-09, before the 400–599 baseline is complete).**

- **W1's first test.** W1's first primary, on 200–399, was written in an uncommitted local script (`~/data/orbit_census_local/diag/heldout_delta.py`), not in git, and it failed (p = 0.036). The 400–599 primary is therefore W1's second try.
- **W1 alone.** W1 merges alone only if its 400–599 primary passes at α = 0.005 (Bonferroni over its two tries), with no harm.
- **#667.** #667 is judged as W1 + #667 against the baseline, at α = 0.01 as already committed. If #667 passes, that does not mean W1 passed.
- **Comparator.** #667 against W1 stays report-only. It may not replace the baseline comparison.
- **Partial look.** An aggregate look was taken at the partial baseline (82 of ~190 systems) during an independent review on 2026-11-08. No decision uses it. The primary is scored only on the complete set, exactly as written.
- **Simulator caveat.** The census simulator sets each night's χ² to exactly n·s² (noise-free) and truncates ln f (#641), so these results are not performance on real data.

**Floor sentinel gated by mass (#641): decision 2026-11-09.**

- **What #641 changed.** It was merged on 2026-11-03, after the first census (aea931a, 73/200 failed). It changed `CellsResult.failed` so that a cell at the floor fails the system only if it `matters`. This loosens assertion 2 as implemented.
- **The decision.** Ben accepts this rule, on the condition that `matters` gets the M2 fix from the 2026-11-08 code review:
  - a genuine upper bound on each cell's evidence, in place of max(ln Ẑ, best Laplace);
  - the threshold ln(n_cells/δ).

  The fix will be a separate PR, judged on a fresh census arm.
- **Until then.** Census results under the current `matters` rule carry this caveat.
- **Update 2026-11-12.** The M2 fix merged as #686 (`0750be5`), into main and so into W1 (`570e13e`) and this branch. Only runs pinned before it (baseline f71294b, W1 2673182) carry the caveat.
---

---
## Arm: tempered SMC from the search mixture where the IS is unreliable (design §6.15), added 2026-11-08 before any run of this arm

- **Arm:** the head of the SMC PR (stacked on #667), frozen once this file is committed; later pushes are CI or review fixes only, each declared as changing results or not.
- **Paired comparator:** #667 head `9204b787bd62b15e60d19f06c7982e2858f11800`. Same image, same platform (OzSTAR).
- **Decision set:** fresh systems 600–799 of seed 0 (`START_BASE=600`). Nobody has inspected them. Aggregate only.
- **Primary:** failure (status failed or error) in the "rest" stratum, exact one-sided McNemar on the discordant pairs (arm better), α = 0.01.
- **Harm (exact two-sided McNemar, harm if the arm is worse at p < 0.05):** failures in each other stratum, and misses (ok with the truth outside the 99% set).
- **Calibration, harm if violated (arm only, ok systems):** misses are not significantly above 1% (binomial, p < 0.01); the PIT KS tests for N, ln P, a, e and cos i are not rejected at 0.01, judged with a Bonferroni correction over those five.
- **Cost, report only:** median and maximum `seconds` per stratum, and the `n_smc`, `n_smc_failed`, `smc_stages`, `smc_seconds` columns.
- **Merge rule:** primary passes and no harm.
- **Review fix, 2026-11-08, before any run of this arm (changes results):** the SMC runs two replicates and its error is √max(genealogy variance, Δ²/4) (design §6.15), commit `0814e1c5d9d126b46f74029dd6b34070ad0646f2`. The decision set stays 600–799 and nothing above is moved.
- **Freeze breach, declared 2026-11-09 (before the 600–799 results are downloaded):** commit `80babea8c0e17c5373f24ce5b0bac7dc6231a7af` **changes results**. The SMC trigger excludes at-floor cells, the per-cell time budget changed, and the oracle accepts the SMC only if it is not unreliable. It was pushed after the freeze and after the 600–799 census was submitted, so that census measures `fce671cc48fd73d7c35e8bd958b10faaf23ddbc0`, not the PR head. Its results are **report-only**: a diagnostic of the old SMC, not a decision.
- **Report-only oracle check, to be scored when those results arrive:** for every true cell that got an SMC and has an oracle, report the signed gap `oracle_gap` and z = (ln Ẑ_SMC − ln Z_oracle)/err. Harm if the count with |gap| > 4·err + 0.01 exceeds the binomial 1% expectation at p < 0.01, or if the mean signed gap is below −0.1.
- **Redesign:** the SMC error and mass matrix are being redesigned (plan to follow, implemented on this PR). The redesigned SMC gets fresh criteria and fresh systems before any decision census. The 600–799 set is not a decision set for it.
---
## Detectable stratum and sky-plane scoring (added 2026-11-16, before any run on it)

Signed off by Ben with amendments (a)–(d), 2026-11-16: (a) the signal test is at 90 % power, not the expected data;
(b) the sky test quantity is the plain, sign-aware track; (c) assertion 5 gates in the detectable stratum only;
(d) `SEASON_GAP` is derived from the simulator's season constants. All four are incorporated below.

Direction (Ben, 2026-11-16): some orbits should not be detectable and the census should not worry about them: those
whose time sampling does not Nyquist-sample the true period, or with many outliers or poor SNR. Comparison is made in
the sky plane only. This section is the pre-registration: it is committed before any census run that scores with it. No
threshold below was chosen or adjusted from a census CSV, and held-out systems (200–399) are reported in aggregate
only.

### The detectable stratum (columns `nyquist_ok`, `dchi2_true`, `dchi2_thresh`, `detectable`)

Defined from the truth and the night times only, never from a fit. A system is detectable iff (a) and (b) hold.

**(a) Sampling.** `P_true >= 2 · median(gaps)`, the gaps being those between consecutive nights of the same season
(consecutive nights further apart than `SEASON_GAP` are in different seasons). `SEASON_GAP` is derived from the
simulator's own constants, `SEASON_LEN` = 30 d (a season's span) and `SEASON_SPACING` = 365.25 d (the second season
starts this long after the first): the longest in-season gap is at most `SEASON_LEN` and the shortest between-season
gap is at least `SEASON_SPACING − SEASON_LEN` = 335.25 d, and `SEASON_GAP` is their midpoint,
`(SEASON_LEN + (SEASON_SPACING − SEASON_LEN)) / 2` = `SEASON_SPACING / 2` ≈ 182.6 d. Any value strictly between the two
separates the simulator's seasons exactly; the midpoint is the neutral one. The inter-season gap is never a sampling
gap. A system with no within-season gap fails.
Irregular sampling has no sharp Nyquist limit; the median gap is the typical cadence, so (a) removes the systems whose
cadence plainly cannot resolve the period and nothing finer. The smallest gap was considered and rejected: the usable
Nyquist frequency of irregular sampling can far exceed that of the mean gap (Eyer & Bartholdi 1999), so the smallest
gap would call detectable a period sampled only by a few close pairs. The domain rule already excludes truths below
2·Δt_min (stratum B′, `P_cut`); (a) is the stricter, typical-cadence version of it.

**(b) Signal.** The static model is a constant sky position (fitted by weighted least squares) and zero velocity, on the
`m` true inlier nights. Each inlier night's summary is `(p̂, v̂)` = the true night state plus noise, as the simulator
draws it: each position coordinate with standard deviation `σ_eff = s σ` (`s = e^{ln s}` the true error scale, `σ` the
quoted error) and each velocity coordinate with `σ_eff / √T_SS` (the least-squares slope over the six frames). The flux
coordinate is the same in both models and is left out. With the true states `μ_k, u_k` (position, velocity) at the
inlier nights,

- `χ²_static = Σ_k |x_k − x̄|² / σ_eff² + T_SS Σ_k |v_k|² / σ_eff²`, where `x̄` is the weighted-least-squares position
  (the mean, as the weights are equal);
- degrees of freedom `ν = 4m − 2`: 4 numbers a night (2 position, 2 velocity), minus the 2 fitted position
  coordinates; the velocity has no free parameter;
- under the true orbit `χ²_static` is noncentral χ² with `ν` degrees of freedom and noncentrality
  `λ = [Σ_k |μ_k − μ̄|² + T_SS Σ_k |u_k|²] / σ_eff²`, the noise-free Δχ² of the static model against the true orbit
  (the true orbit fits the noise-free data with χ² = 0). This is `dchi2_true`;
- under a static truth `χ²_static` is central χ²_ν, and the 1 % false-alarm threshold is `c = χ²_ν,0.99`
  (`α = 0.01` matches the plan's no-orbit verdict);
- the **power** of that test against the true orbit is `P(χ²'_ν(λ) > c)`, the noncentral χ² survival function
  (`scipy.stats.ncx2.sf(c, ν, λ)`), increasing in `λ`. `dchi2_thresh` is the `λ*` at which the power is 0.9, the root
  of `ncx2.sf(c, ν, λ*) = 0.9` found by Brent's method (`required_excess`). A system is detectable iff
  `dchi2_true >= dchi2_thresh`, i.e. iff the 1 % test of a static model has power ≥ 0.9 against the true orbit.

This one number covers SNR, the number of inlier nights and the outlier fraction (through `m`, hence `ν` and `c`), so
there are no hand-set cuts on any of them. A system with no inlier night (B″) has no signal: not detectable. A Monte
Carlo test (`tests/test_orbit_population.py`) checks the derivation: mean `χ²_static` is `ν + λ`, the false-alarm rate
under a static truth is 1 %, and the power at `λ = λ*` is 0.9.

### What gates in the stratum

Assertion 2 (no flagged or failed cell), the retrieval-miss rule and assertion 5 (e > 0.3: the true cell's best mode
within χ²_d(0.99) of the truth; otherwise unchanged) gate in the detectable stratum only. Everything outside it (not
Nyquist-sampled; Nyquist-sampled with the signal below the threshold) is counted and tabled by stratum, never gated;
assertion 5 outside the stratum is reported. Assertion 7 (CMH on failure vs ln s) uses the detectable stratum, where
failure gates. Errors (a crash) still fail the run wherever they occur. Assertion 6 is unchanged.

### Sky-plane retrieval (columns `pit_sky`, `sky_miss`, `sky_f`, `sky_q99`)

**Test quantity.** `f` is the time-averaged squared sky-plane distance between a *physical* relative-position track
and the posterior-weighted mean physical track, on `SKY_GRID` = 64 uniform times over `[first night, last night]` (a
declared budget, not tuned). SBC is valid for any test quantity, so the choice only sets the sensitivity.

*The label.* The target sums out the brighter/fainter label `σ = ±1` (`½ Σ_σ`, with `σ` flipping `(p, v)`), so the
posterior over the orbit `θ` is exactly symmetric under `TI′ → −TI′` (`z → iz`), which negates the orbit's track, for
every system. The physical label is `σ`, not the sign of `TI′`. Each draw `θ_i` is therefore given `σ_i` from its exact
conditional, `P(σ = +1 | θ_i, D) ∝ L_{+1}(θ_i)` with `L_σ(θ) = ∫ dε Π_k [(1−ε) e^{ℓ̃_k(σ·state)} + ε e^{b_k}]` (the two
terms of the target's label sum, exposed as `sign_log_likelihood`; the target value is unchanged), so that `(θ_i, σ_i)`
are exact draws of the joint posterior. The physical track of draw `i` is `σ_i · track(θ_i)`. The truth's is the
simulator's, `sign · track(θ_true)` (`σ = +1` is the simulator's convention, companion relative to primary with `ln f ≤ 0`).
The `σ_i` are drawn from a new child stream of the system's `SeedSequence`, so the simulation, PIT and tie-break draws
are unchanged. The image `θ → θ′` has `P(σ = +1 | θ′) = P(σ = −1 | θ)`, so `σ · track` is invariant under it.

- For a non-twin the catalogue's peaks sit at one sign of `(p, v)` with that `ln f`, so one `σ` fits the data and
  dominates: the posterior is one physical track, the test has full power and flags a wrong-label fit;
- for twins (peaks at both signs) `σ` is about 50/50, the posterior is bimodal and the mean track shrinks towards zero:
  power is lost, validity is not. That is the accepted twin degeneracy.

*Correction.* The first version of amendment (b) said that the per-night flux fixes the TI′ sign. That was wrong: the
target is symmetric in TI′ and the label is carried by `σ`, which the target sums out. The mechanism above is the
correction. It keeps the intent of (b) (a sign-aware test quantity that catches a wrong label for non-twins) and was
decided by the coordinator within Ben's sign-off, with Ben informed.

Cells are weighted by `p` and each cell's resample-move draws equally, as for the other PITs.

**SBC.** `pit_sky` is the randomized rank of `f(θ_true)` among the weighted draws (ties broken by a uniform drawn after
the existing PIT's uniform in the system's PIT stream, so the paired draws are unchanged), tested for uniformity with
the existing chi-square over 10 bins (gate `p < P_FAIL`). **Sky miss:** `f(θ_true)` above the posterior 99 % quantile of
`f`, counted in the detectable stratum with the existing `Binomial(n, 0.01)` rule (gate `P_MISS`).
Test quantities may depend on the data and still give a valid SBC (Modrák et al. 2023, arXiv:2211.02383).

**Statistics caveat.** The SBC rank guarantee holds over the joint prior, so the gated uniformity test runs on all fitted
in-domain systems, with no selection on a truth-based quantity. (Rows of failed systems are already excluded from the gate
by the existing rule; that selection on a fit outcome is not new.) The truth-based detectable stratum is used only for the
failure gate and the sky-miss count. There the coverage is approximate: the stratum keeps the data-dominated systems,
where the posterior is close to the likelihood and its calibration under selection is close to the unselected case,
but the 1 % miss rate is not exact. The SBC uniformity of the stratum itself is reported, not gated.

**Report only.** `in_99` on N, the SBC of N and ln P, the invariants' ranks and the element-space miss counts stay as
columns and report lines, so arms stay comparable with earlier runs.

### Sentinel

`at_floor` (`virgil.mode_search.importance_evidence`) fired when the weighted mean inlier count was below `m_min`; a
healthy mode with exactly `m_min` inliers fired it whenever a draw lost one. It now fires only when more than half of
the importance weight is on draws with fewer than `m_min` inliers (the count rounded to an integer), and the found-mode
Laplace `ln Z > floor + W` condition is unchanged.

---
## Stack decision: W1 #642 + #667 + #678, added 2026-11-17 before any run of this arm (approved by Ben)

- **Arms:** the stack, at the head pinned for its census (this commit or a later one with no `src`/`scripts` change), against main `8dd7264461162f2f076b7b734c021d961e560dd7`. Both run on OzSTAR in the same image, on seed 0, systems 400–599 (`START_BASE=400`). They are scored with the criteria committed by #728 (detectable stratum, sky-plane retrieval, majority-weight `at_floor` sentinel).
- **Unit:** the stack merges as one unit. W1 and #667 are not decided alone. Their earlier sub-rules in this file (the W1 Bonferroni rule and the #667 primary against f71294b) are kept on record but superseded for the merge decision.
- **Primary:** failure (status failed or error) in the detectable stratum. Exact one-sided McNemar on discordant pairs, α = 0.01.
- **Harm**, judged by exact two-sided McNemar (p < 0.05, arm worse), on each of:
  - sky misses in the detectable stratum;
  - failure in each non-detectable stratum.

  Also harm: the sky-plane SBC uniformity test (all fitted in-domain systems) fails for the arm where it passes for main.
- **Merge rule:** the primary passes and there is no harm.
- **Data handling:** systems 400–599 are aggregate-only. Nobody traces them per seed.
