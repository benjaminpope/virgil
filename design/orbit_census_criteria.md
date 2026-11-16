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
2026-11-16 below: assertions 1 and 2 gate in the detectable stratum only, and assertions 1 and 3 are scored in the sky
plane. Rows of failed systems carry `pit_diagnostic=1` and never enter a gated SBC. The oracle columns are diagnostics:
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

## Detectable stratum and sky-plane scoring (added 2026-11-16, before any run on it)

Direction (Ben, 2026-11-16): some orbits should not be detectable and the census should not worry about them: those
whose time sampling does not Nyquist-sample the true period, or with many outliers or poor SNR. Comparison is made in
the sky plane only. This section is the pre-registration: it is committed before any census run that scores with it. No
threshold below was chosen or adjusted from a census CSV, and held-out systems (200–399) are reported in aggregate
only.

### The detectable stratum (columns `nyquist_ok`, `dchi2_true`, `dchi2_thresh`, `detectable`)

Defined from the truth and the night times only, never from a fit. A system is detectable iff (a) and (b) hold.

**(a) Sampling.** `P_true >= 2 · median(gaps)`, the gaps being those between consecutive nights of the same season
(consecutive nights further apart than `SEASON_GAP` = 100 d are in different seasons; the simulator's seasons span 30 d
and are 365 d apart, so the inter-season gap is never a sampling gap). A system with no within-season gap fails.
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
- under a static truth `χ²_static` is central χ²_ν, and the 1 % false-alarm threshold is `c = χ²_ν,0.99`. The expected
  data reject the static model at that level iff `ν + λ >= c`, so `dchi2_thresh = c − ν` and the test is
  `dchi2_true >= dchi2_thresh`.

`α = 0.01` matches the plan's no-orbit verdict. The test is the expected-data one (about even odds of rejecting
a static model, not a high-power one): the stratum is the systems whose noise-free data already reject a static source at
1 %. This one number covers SNR, the number of inlier nights and the outlier fraction, so there are no hand-set cuts on
any of them. A system with no inlier night (B″) has no signal: not detectable.

### What gates in the stratum

Assertion 2 (no flagged or failed cell) and the retrieval-miss rule gate in the detectable stratum only. Everything
outside it (not Nyquist-sampled; Nyquist-sampled with the signal below the threshold) is counted and tabled by stratum,
never gated. Assertion 7 (CMH on failure vs ln s) uses the detectable stratum, where failure gates. Errors (a crash) still
fail the run wherever they occur. Assertions 5 and 6 are unchanged.

### Sky-plane retrieval (columns `pit_sky`, `sky_miss`, `sky_f`, `sky_q99`)

**Test quantity.** `f(θ)` is the time-averaged squared sky-plane distance between θ's relative-position track and the
posterior-weighted mean track, on `SKY_GRID` = 64 uniform times over `[first night, last night]` (a declared budget,
not tuned). Position-only data fix the track only up to the point reflection through the origin (the label swap
`TI′ → −TI′`, a symmetry of the target; the simulator's brighter/fainter sign), and the posterior has equal mass in both
images, so the mean track itself would vanish. The track is therefore represented by its outer product
`(x², xy, y²)`, which is invariant under the reflection, and `f` is the squared Frobenius distance of the draw's
features from the posterior-weighted mean features. Cells are weighted by `p` and each cell's resample-move draws
equally, as for the other PITs.

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
