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

**W1 merge rule (added 2026-11-09, before the 400–599 baseline is complete).**

- **W1's first test.** W1's first primary, on 200–399, was written in an uncommitted local script (`~/data/orbit_census_local/diag/heldout_delta.py`), not in git, and it failed (p = 0.036). The 400–599 primary is therefore W1's second try.
- **W1 alone.** W1 merges alone only if its 400–599 primary passes at α = 0.005 (Bonferroni over its two tries), with no harm.
- **#667.** #667 is judged as W1 + #667 against the baseline, at α = 0.01 as already committed. If #667 passes, that does not mean W1 passed.
- **Comparator.** #667 against W1 stays report-only. It may not replace the baseline comparison.
- **Partial look.** An aggregate look was taken at the partial baseline (82 of ~190 systems) during an independent review on 2026-11-08. No decision uses it. The primary is scored only on the complete set, exactly as written.
- **Simulator caveat.** The census simulator sets each night's χ² to exactly n·s² (noise-free) and truncates ln f (#641), so these results are not performance on real data.
---
