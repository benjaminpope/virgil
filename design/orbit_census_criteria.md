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

§14 assertions 1–8 stand as committed, assertion 2 included (no cell-at-floor failures). Rows of failed systems carry
`pit_diagnostic=1` and never enter a gated SBC. The oracle columns are diagnostics: they never touch `p_pos` or the
status.

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
