# Orbit library: de-hacking design (Track 2, Stage D)

Status: **draft for Ben's approval.** No stage starts before he approves.
Scope: `aliases.py`, `epochs.py`, `arcs.py`, `orbit_search.py`,
`orbits.py`, `tempering.py`, `_marginal.py`, `pipeline/orbit.py` and the
orbit checks in `pipeline/_checks.py`, at 884aa7e (11,952 lines).
Companion: [orbit_gate_042.md](orbit_gate_042.md), the pre-registered
population gate that every stage below reports against.

Ben's brief (2026-10-28): the orbit code must be lean, efficient and
above all principled, with no dodgy heuristics or hard-coded cheats.
Priorities, in order: principled; reliable for orbits in general; lean
(one path per job, no unused code); efficient.

## 0. Summary

- **No constant in `src/` names a target.** Target tuning entered by two
  other routes: (i) commits made to pass a Gl 229 gate, and (ii) tests
  that pin Gl 229, ι Peg and AK For measurements as the generic fixture.
- **Past Gl 229 passes depended on heuristics.** Said plainly:
  - 32acb33 (2026-10-19) divided the short-arc sign gap by the tempering
    factor *f*. The commit body says the 0.4.0 Gl 229 gate failed because
    three seeding nights' gaps (44, 8.6, 140 nats at *f* = 41) passed the
    gate. Dividing by *f* = 41 switched short arcs off for Gl 229.
  - 52c7013 (2026-10-21) split the search into a plain-likelihood search
    plus a mixture "polish", and added the flux-flip swap-back queue,
    because "the N=34 band's best mode was 108 nats short".
  - 0e3ce54 gated short arcs on median flux < 0.8; 6a2a7b4 replaced that
    with a 5-nat gap gate the same day.
  - Gl 229's *f* = 41 is far above the code's own `_INADEQUATE_F` = 3,
    beyond which, per the code, "the noise model does not describe the
    data". The pass was on data the code itself calls inadequate.
  - The one non-blind pass (job 19004328) used a 10–14 d window centred
    on the published period.

  The 0.4.0/0.4.1 Gl 229 passes are **unverified** until the de-hacked
  code reproduces them cold (gate §7).
- **Blind defaults fail exactly where the model misfits.** The gallery
  session's blind runs:

  | system | χ²/N | our *P* | published *P* | right alias? |
  |---|---|---|---|---|
  | ι Peg | 27 | 9.861 d | 10.213 d | no |
  | ο Leo | 13 | 9.518 d | ~14.5 d | no |
  | Gl 229 B (blind) | 12 | 8.79 d | 12.13 d | no (ΔlnZ 1) |
  | HD 210763, HD 9312, κ Vel | 2–3 | — | — | yes |
  | HR 6819 | 5.6 | — | — | 6 of 10 nights outliers |

  This is the strongest evidence in hand. The engine's machinery of
  gates, re-admission, polish and flips does not rescue misfit data. It
  only hides that the evidence is overconfident when no model in the
  family fits.
- **The redesign replaces six families of heuristics** with five
  formulations, each with a stated derivation and a test against an
  analytic or brute-force reference:
  - (A) one night-outlier mixture, with ε marginalised;
  - (B) sign handled at the seeding level by per-night peak mixtures,
    with no sign gate;
  - (C) band evidence by truncated importance sampling at bounds, with a
    reported standard error;
  - (D) the primary diameter always fitted linearly, with a
    Savage–Dickey verdict;
  - (E) one distance unit;
  - (F) one cycle-count rule with a stated pruned-mass bound and no
    re-admission.

  No nats threshold survives: alias, label and resolved verdicts are
  probabilities with 95% credible sets, and detection (orbit against no
  orbit) is an empirical false-alarm probability below α = 0.01.
- **Expected size:** the orbit files go from about 11,480 to about 5,350
  lines; the whole package changes by about −5,100 (11,950 → 6,870 for
  the scope above), because about 1,060 generic lines move to or are
  added in core modules (§1.5, §3.2). `orbit_search.py` is retired and
  `arcs.py` is folded into seeding.
- **No orbit-local forks of generic fitting code** (Ben, 2026-10-28):
  optimiser, evidence, importance sampling, NUTS, marginals, mixtures,
  covariance and I/O live in core modules usable by any fit (§1.5). `fit_orbit_aliases`
  becomes about 60 lines over seven named stages.

## 1. Inventory

Verdicts:
- **DELETE**: goes, with nothing in its place;
- **MERGE**: one copy survives;
- **REPLACE**: a principled formulation in §2 takes its place;
- **KEEP**: stays, with the derivation written here and in the code.

File keys: a = aliases.py, e = epochs.py, r = arcs.py, s = orbit_search.py,
o = orbits.py, t = tempering.py, p = pipeline/orbit.py, c = pipeline/_checks.py.
"Target evidence" is what links an item to a named system, if anything.

### 1.1 Heuristics, cascades and gates

| item | where | verdict | replaced by | target evidence |
|---|---|---|---|---|
| plain search → mixture polish, `_plain`, `except` fallback to plain | a:1169-1239 | REPLACE | A: one mixture everywhere | 52c7013 "Gl 229 gate regression … 108 nats short"; test_night_flags.py:190 |
| `outlier_eps` (engine 0.0, pipeline 0.02), `_NO_OUTLIER` −1e30, `_with_outlier` | a:2896, a:428, a:2697, p:~143 | REPLACE | A: ε ~ Beta(½,½), marginalised | engine default kept 0 "so direct calls keep their results": results preservation, not science |
| `_signs_trusted`, `ARC_MIN_GAP` 5, tempered by *f* | a:2239, r:74, a:3146-3170 | DELETE | B: no sign gate; the sign is in the peak mixture | 32acb33; test_aliases.py:2331 pins Gl 229 gaps and *f* = 41.1 |
| `sign_free` seed doubling, `_thiele_innes_fit_sign_free`, `_N_SIGN_PAIRS` 16 | a:3234-3240, o:709, o:705 | REPLACE | B: peak-mixture EM | none (twin fixtures) |
| flux seeds 0.7, 0.95 × top; `_N_FLUX` 4 | a:786, a:111 | DELETE | B: flux seed from the assigned peaks | none |
| `flip_flux`, `_flip_modes`, `_flipped` (0.9·top, ±1e-3), pin test 0.02, swap-back queue ×2, `_flip_is_relabelling` | a:1129-1148, a:1323-1366, a:1370 | DELETE | B: flux ∈ (0, 1] is a fundamental domain; seeds are already in it | swap-back from 52c7013 |
| `LABELS_FREE_NATS` 3, `_LABELS_FREE_RES_FRACTION` 0.25 | a:120-122 | REPLACE | B: report p_label = Z_x/(Z_x+Z_Rx) | none |
| short-arc re-admission, `_seed_scores`, `_N_CHECK` 3 | a:2307-2385, a:2304 | DELETE | F: one rule, no re-admission | none |
| `ARC_DELTA_CHI2` 25, `ARC_JITTER` 0.1 | r:66, r:62 | REPLACE | F: pruned-mass bound; jitter fitted by ML | test_arcs.py fixture is Gl 229 Ba–Bb |
| `_auto_primary`, `_pilot_diameter`, re-temper retry, `diam_seed`, ×1.01/0.99 clip, `2*estimate.diam` | a:2440, a:2527, a:3056-3110 | REPLACE | D: θ always fitted, one pass | ι Peg 1.05 mas in tests |
| `estimate_primary_diameter` (224 lines), `RESOLVE_NATS` 5, `_DIAM_FRACTIONS`, 0.15 curvature step | e:2497, e:2453, e:2457, e:2652 | DELETE | D | ι Peg fixtures test_epochs.py:877, 958 |
| `_DIAM_MIN_AUTO` 1e-3, `_DIAM_TOP_AUTO` 1.5, `diam_range` (0.05, 5) | a:151-152, a:746 | REPLACE | D: θ ∈ [0, 1.2197 λ/B_max] | none |
| `_N_DIAM_NULL` 100 | p:65 | REPLACE | D: Gauss–Legendre over θ (accuracy-tested) | none |
| near-bound IS/Laplace switch, `ess_min` 50, `flag_nats` 1, `bound_sigma` 3, "laplace-is-differ", "low-ess" | a:1263-1299, a:975 | REPLACE | C: always truncated IS with SE | none |
| #569's on-bound Hessian surgery and 1e-6 p_in floor | PR #569 | REPLACE | C | written for the gallery's bound modes |
| `_PERIOD_BOUND_FRACTION` 0.9, `_LOWER_BOUND_FACTOR` 1.1, `_PERIOD_SIGMA_FRACTION` 0.25 | a:127-131 | REPLACE | C: window-edge truncated mass > δ | none |
| `min_distance_mas` 0.5 mas, #569's min(0.5, 0.5·res), `_MODE_TOL` | a:2872, e:1888 | REPLACE | E: ½ λ/B_max | none |
| random starts e ≤ 0.6·e_max, cos *i* ±0.95; `_FEW_SEED_NIGHTS`, `_N_RANDOM_FEW` 200, `_N_RANDOM_NONE` 1000 | a:1086-1100, a:2163-2165 | REPLACE | draws from the prior; one `n_random` | none |
| `_flag_chi2` (factor 3, floor 1, < 3-night exemption, per-mode median), `_mode_keys` | e:1995-2037, e:1891 | DELETE | A gives each night an outlier probability | AK For: test_night_flags.py:90 |
| `sidelobe_flags`; flux-at-bound, alias-peaks, low-gap, edge night flags; `skip_flagged` | e:2039, e:1801-1885 | DELETE | B: each night's peaks and weights enter the seeding mixture | Gl 229 2024-02-27 night: test_epochs.py:806 |
| `drop_outlier_night`, `_robustness`, `_outlying_night` | p:1042-1150 | DELETE | A: a jackknife that costs a full search, made redundant by the mixture | none |
| `_POOR_FIT_CHI2_RED` 10 ("χ²/ν of 65–800", uncited) and `_INADEQUATE_F` 3 | a:2159, a:2596 | MERGE | `tempering.FLAG_RANGE`, plus raw χ²/N reported first | Gl 229 *f* = 41 passed regardless |
| `_WIDEN` 1.5, `_MAX_WIDEN` 5 box-widening loop | p:48-49 | REPLACE | prior support a_max(1+e_max) sets the box once | none |
| `alias_nats` 10, `no_orbit_nats` 10, `LABELS_FREE_NATS` 3, `RESOLVE_NATS` 5 | p, a:120, e:2453 | DELETE | G: probabilities and 95% credible sets; detection by empirical FAP < α = 0.01 | none |
| `posterior_p_min` 0.05, `max_modes` 3, engine `p_min_samples` 0.01 | p, a | MERGE | sample bands until the remaining mass < 0.01 | none |
| `_WARN_P_MIN_NIGHTS` 5 | a:141 | DELETE | the window's lower end is reported; the gate's data-sufficiency table covers the rest | none |
| `n_refine × 4` keep in `_refine` | a:1169 | DELETE | ranking by mixture likelihood, top `n_refine` distinct | none |

### 1.2 Constants kept, with derivations

| item | where | value | derivation |
|---|---|---|---|
| `_P_FLOOR` | a:135 | 0.2 d | Contact limit. Two main-sequence stars touch below about 5 h (W UMa period cut-off near 0.22 d; Rucinski 1992). The method models detached point/disc pairs. |
| position grid step | p:45 | λ/(4 B_max) | Twice Nyquist: the shortest fringe period is λ/B_max, so Nyquist is ½ λ/B_max. At this step every true peak lies within λ/(8 B_max) of a grid point on each axis, i.e. inside its basin (radius ½ res, E), so refinement from the nearest grid point reaches it. Efficiency-only: M4 measures misses. |
| `_MAX_SHIFT` | r:77 | ½ λ/B_max | E: half the shortest fringe period. A larger shift between half-nights is a different peak, not motion. |
| `MIN_CHAINS` | t:38 | 10 | The relative scatter of a block's *f* is √(2/C), which is 0.45 at C = 10. Fewer chains pool. |
| `FLAG_RANGE` | t:35 | (0.5, 2) | *f* < ½: anticorrelated residuals. *f* > 2: fewer than half the channels independent. Absorbs `_INADEQUATE_F` and `_POOR_FIT_CHI2_RED`. |
| `FLAT_EXCESS` | t:40 | 0.15, and 3 SE | Diagnostic only; reported, never gating. |
| `_N_NODES`, `_WINDOW`, `_N_CENTRE` | e:444-446 | 32, 9σ, 4 | Quadrature accuracy. Stage 5 adds a test: relative error < 10⁻⁶ against `scipy.integrate.quad` on the bounded scale integral over the documented s range. |
| `_DTOL` | a:869 | 1e-9 nats | Newton-decrement convergence, far below any reported precision (the gate resolves 1 nat). |
| `k` (phase coherence) | o:773 | 9 | Efficiency. The phase drift across a band is at most 1/k cycle, while a χ² basin's half-width is about ¼ cycle, so k ≥ 4 keeps every basin sampled. Nine is 2× margin; M4 measures any miss. |
| `_N_PHASE` 36, *e* step 0.05 | r, o:835 | | Efficiency, as for k: 10° in phase and 0.05 in *e* are well inside a basin (¼ cycle; Δe ≈ 0.1 at the default span). M4 measures misses. |
| Student-t df 5 in the IS proposal | a:975 | 5 | Efficiency only. Any proposal positive on the support gives an unbiased estimator; heavier tails than the target give finite variance. The reported SE bounds the effect (C). |
| memory constants (`_HESSIAN_COPIES`, `_TABLE_COPIES`, budgets) | a:533, e:1415 | measured | Efficiency only. The two planners merge (§1.3). |
| `period_window` | a:211 | | The method's domain statement. Lower end: the companion moves less than ½ λ/B_max (E) within the longest night. Upper end (Ben, 2026-10-28): `span_factor` is deleted; periods beyond the span form **one long-period band** [span, P_max] whose evidence is computed like any other. P_max is where the orbit stops being distinguishable from a static companion: the motion over the span, ≈ 2π a_max · span / P, falls below the smallest per-night position error σ_min, so P_max = 2π a_max · span / σ_min (a data-derived domain limit, reported). `n_bands_max` stays a reported resource cap. |
| `_VIS_AMP_DEFAULT` | a:144 | (0.5, 1.2) | **Moves to the instrument layer** (Ben, 2026-10-28): the per-epoch V² amplitude range is supplied with the prepared data by virgil-vlti and the other instrument packages (pipeline/instrument boundary). The engine has no default; with none supplied, the amplitude is fixed at 1. No citation is invented. |
| `n_bands_max` | a:138 | 200 | A resource limit, stated as one, with `window_capped` reported. |

### 1.3 Duplicates (one path per job)

| job | copies now | survivor |
|---|---|---|
| Thiele–Innes least squares | o:683, o:709, r:361 (+ velocity rows), plus the grids in `starting_orbits` and `_pv_grid` | one `thiele_innes(design, w)` in `orbits.py`, with optional velocity rows |
| band profile over (P, e, τ) | `aliases.position_profile`, `arcs.arc_profile` | one `seed_profile` (stage 3 of §3) |
| search and rank orbits | `fit_orbit_aliases`, `rank_orbits` + `start_from_positions`, `orbit_search.score_orbits` | `fit_orbit_aliases`; the other two are deprecated (§5) |
| scale marginal | `_Surface` closed form, `_Surface.numeric`, `_marginal.log_integral`, `orbit_search._band_marginal`, `grid_fit._log_uniform_evidence` | `_Surface` + `_marginal.log_integral`; `orbit_search` goes; gains are integrated, never profiled |
| von Mises block | e:642-655 and e:718-728 | one helper |
| scene → score | aliases `_Problem`/`_KernelProblem`/`_epoch_residuals`, `epochs._loglikes`, `_grid_scores` | one `night_loglikes(scene, nights)` |
| UD + point binary | `epochs._binary`, `aliases._UDBinary`, `aliases._model` | `aliases._model` (`OrbitalBinary` with a UD primary) |
| track distance and dedup | `_distinct_tracks`, a:1354, a:1467, `OrbitStart.modes`, `_mode_keys` | `_distinct_tracks` with the E tolerance |
| resolution λ/B_max | five sites, three styles | computed once in `prepare`, passed down |
| orbit vector layout | about nine index sites plus `_SITES` and `_numpyro_order` | one `_Layout` (names ↔ slices ↔ periodic periods) used by engine and pipeline |
| PD / Hessian → covariance | seven `eigvalsh` checks, three Hessian→cov routines | one `_covariance(H)` returning (cov, ok) |
| memory planners | `_hessian_groups`, `_grid_batch`, `_candidate_batch` | one `_plan_batches(per_item, budget)` |
| night grouping | `OIData.epochs`, `_night_length` (ignores `gap_days`), `_read_nights` | `OIData.epochs(gap_days)` only; `gap_days` is threaded through |
| night-outlier logic | `_flag_chi2`, `_outlying_night`, `_with_outlier`, `_flag_one_night` | the mixture's posterior outlier probability (A) |
| JSON | `AliasResult.save` (not atomic), `_checkpoint`, `pipeline._io` | `_io.write_json` (atomic, strict) |

### 1.5 Generic operations: one core home (no orbit-local forks)

Ben (2026-10-28): "we want this orbits code to be clean and to
consolidate as many generic fitting functions into one place as
possible, rather than it having a whole bunch of ad hoc forks or
reimplementations of the same idea." Orbit files keep only orbit
physics (Kepler and Thiele–Innes, epochs, band and alias enumeration,
seeding) plus the stage driver. Everything else uses, or moves to, a
core module.

| generic operation | orbit-side implementation(s) now | existing core home | verdict |
|---|---|---|---|
| optimiser loops, restarts, convergence | `aliases._optimize` (Gauss–Newton + L-BFGS-B fallback), `_refine`, `_flip_modes`; `epochs._refine_marginal`; `start_from_positions` refinements; `orbits._fit_batch` | `fitting.fit` (LM, L-BFGS, trust region, budgets), `_gauss_newton.solve` | **use core**: one batched multi-start wrapper `fitting.multistart(objective, starts, …)` (new, in `fitting`) over `_gauss_newton.solve`; the try/except-count-failure pattern lives there once |
| Laplace evidence and covariance | `aliases._mode_evidence`; `pipeline/orbit._laplace_mass`; three Hessian→cov routines | `inference.laplace_cov`, `regularized_inverse`, `objective_laplace_covariance`; `fitting.gauss_newton_mass` | **use core** (`inference`); `_laplace_mass` becomes `fitting.gauss_newton_mass` plus the `_Layout` transform |
| importance-sampled evidence (truncated, defensive mixture, adaptive n, SE) | `aliases._mode_evidence` IS branch, `_band_evidence` | none (`svi` has PSIS diagnostics only) | **move to core**: new `virgil/evidence.py` (`importance_evidence(logpost, modes, covs, in_support, …)` → (ln Ẑ, SE, ESS, p̂_in)) |
| Savage–Dickey ratio | `epochs.estimate_primary_diameter` (gain test) | none | **move to core**: `evidence.savage_dickey(draws, cond_loglike, grid)` |
| NUTS setup, mass matrix, chains, deadline | `pipeline/orbit._sample_band`, `_laplace_mass`, `_numpyro_order`, `_SITES` | `pipeline/_common._run_chains`, `_interior_start`, `_diagnostics`; `fitting` NUTS kwargs | **use core** (`pipeline/_common`); orbit supplies only its `_Layout` sites |
| tempering | `aliases._temper_factors`, `_dof_pairs` | `tempering.estimate_factors`, `block_dof` | **use core**; `_temper_factors` keeps only the per-night fit call |
| scale (noise) marginal | `epochs._Surface` (closed form, numeric, gain table), `orbit_search._Scorer`/`_band_marginal`/`_newton` | `_marginal.log_integral`, `_linear.LinearMarginal`/`whiten_*`, `likelihood._whiten*` | **move to core**: `_Surface` → `_marginal.BlockScaleMarginal` (generic block-scaled residuals); `orbit_search` copies deleted; gains integrated via `_linear` |
| flux marginal (log-uniform) | `orbit_search._band_marginal`; `epoch_positions` third grid axis | `grid_fit._log_uniform_evidence`, `LinearFluxGrid` | **use core** (`grid_fit`) |
| PD checks, covariance regularisation | seven `eigvalsh` sites (epochs ×3, arcs, aliases, …) | `inference.regularized_inverse`, `_add_ridge` | **use core**: one `inference.psd_inverse(H) -> (cov, ok)` replaces them (extend `regularized_inverse`) |
| prior transforms, flat coordinates | `aliases._Bounds`, `_fold`, `_PERIODIC`, `_start_vector`; pipeline `_SITES`/`_numpyro_order`/hard-coded periodic half-widths | `_flat` (`_FlatBijection`, `flat_sampled`), `priors` | **use core**: `_Layout` is a thin orbit table of names, bounds and periods built on `_flat`; the (h, k, λ_ref) map is orbit-specific, because it is orbital geometry |
| mixture likelihood (per-group outliers, ε marginal) | `aliases._with_outlier`, `_night_diagnostics`; `epochs._flag_chi2`; pipeline `_outlying_night` | none (`ensemble._mixture` mixes images, unrelated) | **move to core**: `likelihood.group_outlier_mixture(loglikes, outlier_loglikes)` with the exact Gauss–Chebyshev ε marginal and posterior outlier probabilities |
| empirical FAP / null simulation | none yet (`no_orbit_nats`) | `detection` (`gaussian_null`, `bootstrap_null`, `rescale_errors`, `injection_recovery`, `DetectionMC`) | **use core**; one new hook: a user-supplied statistic in `injection_recovery` |
| contrast-curve floor | none yet | `grid_fit.optimized_flux_grid`, `laplace_flux_uncertainty_grid`; `limits.ruffio_upperlimit`, `radial_profile` | **use core** (`limits.flux_floor`, a ~20-line composition, new in `limits`) |
| grid search, batching, memory planning | `epochs._grid_scores`, own `meshgrid`s, `_grid_batch`, `aliases._hessian_groups`, `_candidate_batch`, `arcs._BATCH`, pad-batch ×6 | `_grid.map_points`, `coordinate_points`, `batch_size_or_default` | **use core** (`_grid`); one `_grid.plan_batches(per_item, budget)` replaces the planners |
| mode dedup | `epochs._distinct_tracks`, `_distinct`, `aliases._distinct`, inlined ×2, `OrbitStart.modes`, `_mode_keys` | none | **orbit-specific, because** the distance is between sky tracks over the observed epochs (§2E); one function in `epochs` |
| EM over per-night peak mixtures | `arcs` (none yet) | none | **orbit-specific, because** the M-step is the Thiele–Innes linear fit; lives in `_seeding` |
| JSON I/O, atomic writes, checkpoints | `AliasResult.to_json/save/load`, `_checkpoint`, `pipeline/orbit._positions_json` | `pipeline/_io.write_json`, `read_json`, `atomic_path` | **use core** (`pipeline/_io`); checkpoints via `write_json` + `atomic_path` |
| metrics, image scores | none in orbit code | `metrics` | n/a |
| precision, stability | `aliases` float64 wrappers | `_precision.in_float64`, `run_in`; `_stability` | **use core** (`_precision`) |

New generic code goes only in: `virgil/evidence.py` (new: importance
evidence, Savage–Dickey), `likelihood.py` (group outlier mixture),
`fitting.py` (multistart), `inference.py` (`psd_inverse`),
`_marginal.py` (`BlockScaleMarginal`), `_grid.py` (`plan_batches`),
`limits.py` (`flux_floor`) and `detection.py` (statistic hook). Each
gets its own unit tests independent of orbits, and at least one
non-orbit caller or a written reason (e.g. `evidence.importance_evidence`
is also the natural evidence for `pipeline/star` and `grid_fit` modes;
`group_outlier_mixture` for any multi-epoch fit).

### 1.4 Dead and constant paths

- **`_Settings` fields read only through the checkpoint key:** `k`,
  `eccs`, `n_candidates`, `short_arcs`, `arc_*` (C1). They go; the
  checkpoint key hashes the real inputs.
- **`_band_target` builds a dummy `_Settings`** (C2). The likelihood
  object needs no search settings once `_Problem` is split from the
  search.
- **Kwargs never named by any caller:**
  - `fit_orbit_aliases`: `k`, `min_distance_mas`, `p_min_samples`,
    `flag_nats`, `ess_min`, `diam_range`, `arc_jitter`,
    `arc_delta_chi2`, `arc_min_gap`;
  - `epoch_positions`: `coarse_margin`, `flag_chi2_factor`,
    `flag_sep_fraction`, `flag_flux_tol`, `flag_chi2_groups`;
  - `period_window`: `motion_fraction`, `p_floor`;
  - `broad_loglike`: `coarse_*`;
  - `sidelobe_flags`: all three;
  - `night_motion`: `max_shift`;
  - `arc_profile`/`unwrap_cycles`: `ecc_max`, `k`.

  They are removed without deprecation, since no caller names them. The
  rule is "deprecate public kwargs with external callers only"; vv and
  `~/data` were searched.
- **`scales="quoted"` branches** (C6) go with `rank_orbits` and
  `score_orbits`.
- **The `old_order` shim on `score_orbits`** (C10) goes with it.

## 2. Principled formulations

Notation:
- *K* nights *k*; ℓ_k(x) is the tempered, scale-marginalised log-likelihood
  of night *k* under orbit vector x (`_night_loglikes`, block dof = 1/f_b);
- res = λ_min/B_max = 1/max|u|, the finest fringe scale over all
  channels (what `Epochs.resolution_mas` already returns; `prepare` uses
  that property, never a central or reference λ);
- δ = 10⁻³ is the precision to which band probabilities are reported
  (three decimals in the summary table).

### A. One night-outlier mixture, ε marginalised

**Model.** Each night is either described by the orbit, or it is an
outlier: a night whose data come from the same binary but whose
companion position is not predicted by the orbit. A miscalibrated or
mis-signed night, or a night with a blend, falls under the outlier case.
The outlier likelihood of night *k* is

  b_k = ln ∫ L_k(p, f_c) π(p) π(f_c) dp df_c,

where:
- p ranges over the disc |p| ≤ a_max(1 + e_max), which is the support of
  the orbit prior's positions;
- f_c is log-uniform on `flux_range`.

The uniform density on the disc is a choice, not derived: the orbit prior
induces a non-uniform density on p (it piles up towards a, set by the
eccentricity and phase), which would need an integral over the orbit prior
per night. The outlier component stands for "somewhere in the prior's
support", and a flat density there is the plain reading of that. The grid is
capped at 201 points per axis (the step coarsens on a long baseline), and
the flux average uses 13 nodes: ln B_k is converged to about 1 nat, small
against the tens of nats that separate an outlier night.

This is `broad_loglike` with its extent taken from the prior. The
`_WIDEN`/`_MAX_WIDEN` loop goes, because the extent is no longer
data-chosen.

The night rate ε has the Jeffreys prior Beta(½, ½). The data's
likelihood is

  L(x) = ∫₀¹ Π_k [(1 − ε) e^{ℓ_k(x)} + ε e^{b_k}] π(ε) dε.

**Exact evaluation.** The integrand is a polynomial of degree *K* in ε.
With ε = (1 − cos θ)/2, the Beta(½, ½) measure becomes dθ/π on [0, π], and
the integrand becomes a polynomial of degree *K* in cos θ.
Gauss–Chebyshev quadrature of the first kind integrates such polynomials
exactly when the node count n ≥ ⌈(K + 1)/2⌉. The nodes and weights are

  θ_j = (2j − 1)π/(2n), ε_j = (1 − cos θ_j)/2, w_j = 1/n.

So

  ln L(x) = logsumexp_j [ −ln n + Σ_k logaddexp(ln(1 − ε_j) + ℓ_k, ln ε_j + b_k) ].

Every term is positive, so there is no cancellation. n = ⌈(K + 1)/2⌉ is
exact; use n = K + 1 for headroom. The cost is (K + 1) × K logaddexps
per orbit, which is negligible beside ℓ_k.

**Consequences.**
- **Occam factor.** With all nights good (b_k ≪ ℓ_k), L = e^{Σℓ} ·
  Γ(K + ½)/(Γ(K + 1)√π) ≈ e^{Σℓ}/√(πK): the mixture costs ½ ln(πK) nats,
  1.6 nats at K = 8. It applies identically to every band, so it cancels in band
  ratios; for the null see below.
- **Posterior outlier probability of night *k*:**

  q_k(x) = Σ_j W_j ε_j e^{b_k} / [(1 − ε_j) e^{ℓ_k} + ε_j e^{b_k}],

  where W_j ∝ w_j Π_k[…]_j are the node posterior weights. This
  replaces `_night_diagnostics`, `_flag_chi2`, `_outlying_night` and
  `drop_outlier_night`.
- **One likelihood for every stage:** search, refinement, evidence, NUTS
  and the null. No "plain" variant exists, and nothing has to be
  polished.
- **The null** (no companion). A null with no mixture is rejected (Ben,
  2026-10-28): on a single star with one bad night the orbit model can
  call that night an outlier while the null pays its full χ², so bad
  nights make false orbits. Two candidates, chosen between by the rule
  pre-registered in orbit_gate_042.md §3.6 (run on unscored stratum D
  before any scored run; must pass both M6 and M8):
  - (ii) shared mixture: the null carries the same mixture with b_k as
    its outlier component. Risk: as ε → 1 the null becomes "a companion
    at a free position every night", which can beat a real orbit at
    K = 4–5 (M8 counts this).
  - (iii) shared noise inflation: in both models the outlier component
    is the model's own night likelihood with its noise scale inflated,
    ln ∫ L_k(σ s) π(s) ds, *s* log-uniform on [1, s_max] (the engine's
    existing scale bound). The null stays a single star; an outlier
    night costs both models alike. Under (iii), b_k is used only by the
    seeding mixture (§2B), not by the likelihood.

  Stage 2 implements both behind a setting; the PR that sets the
  default per the rule deletes the other.

**The local-optimum risk.** 52c7013 split search from mixture because
the mixture flattened a wrong start's bad nights, so L-BFGS stopped 108
nats short. Under A this cannot be patched by a second path. Instead:
- the seeds come from B, which already assigns nights;
- M4 (search misses against the oracle) measures whatever remains.

If M4 fails, the remedy goes back to Ben as a proposal; continuation in
a temperature on ε is the candidate. The failure is never fixed by
reinstating the plain path.

**Tests (generic):**
- the quadrature against `scipy.integrate.quad` on random ℓ, b for
  K ∈ {1, 3, 8, 20}, to 10⁻¹⁰ relative;
- the Occam factor against the closed form;
- q_k against brute-force enumeration of the 2^K inlier/outlier
  assignments, for K ≤ 8;
- a constructed data set with one corrupted night: q_k > 0.9 for that
  night, and the period posterior matches the fit with that night
  removed, to within its SE.

### B. Sign, labels and seeding

**Level 1: the observed-data likelihood has no sign latent.** The
closure-phase sign is physics: the convention is fixed by the OIFITS
standard. A night whose sign is wrong is a night the orbit does not
predict, so it is an outlier under A. An explicit sign-flip component
with its own Dirichlet rate is possible, but **not recommended for
0.4.2**: it adds a parameter for a calibration error that A already
absorbs, and the gate reports how A does on injected sign-error nights.
**Accepted (Ben, 2026-10-28), conditional** on the gate reporting the
sign-error-night ROC (outlier probability of injected sign-error nights
against the rest) and on M5 passing. If either fails, an explicit sign
component comes back to Ben as a proposal.

**Level 2: seeding uses per-night peak mixtures.** `epoch_positions`
already produces, for each night, refined peaks j with:
- position p_kj;
- Laplace covariance C_kj, inflated by the night's tempering factor
  (×f);
- height h_kj;
- a velocity v_kj from the two half-nights, when the shift is under
  `_MAX_SHIFT`.

The night's likelihood as a function of the companion position is
approximated, in likelihood units like A (no per-night normalisation, so
nothing depends on the length unit), by its peaks:

  e^{ℓ_k(p)} ≈ G_k(p) = Σ_j e^{h_kj} exp(−½ (p − p_kj)ᵀ C_kj⁻¹ (p − p_kj)).

Seeding then scores an orbit with A's own mixture, G_k in place of
e^{ℓ_k} and the same e^{b_k} outlier term and ε nodes:

  ln L_seed = logsumexp_j [ −ln n + Σ_k logaddexp(ln(1 − ε_j) + ln G_k, ln ε_j + b_k) ],

so seeding and the full likelihood judge a night's outlier status the
same way. For a near-equal pair, the reflected peak is one of the j,
with nearly equal height. That is the whole sign and label story at this level: no
`sign_free` doubling and no sign gate.

**The fit.** For fixed (P, e, τ), the position is linear in the
Thiele–Innes constants θ = (A, B, F, G): p(t) = M(t) θ. Velocity rows
add Ṁ(t) θ = v. Seeding maximises

  S(P, e, τ) = max_θ ln L_seed(M θ)    (+ velocity terms where present)

by EM (Dempster, Laird & Rubin 1977), treating the peak index j and the
inlier/outlier label as the latent assignment. The E-step computes
responsibilities r_kj (and the outlier responsibility). The M-step is one weighted least squares in θ.
EM increases S monotonically. It stops when the gain is below 10⁻³ nats,
i.e. δ, the reporting precision; a non-converged count is reported.

**Starts** are vectorised over the (P, e, τ) grid:
1. each night's top peak;
2. at the best `n_refine` grid points per band, RANSAC-style minimal
   samples: two nights × their top-J peaks give exactly four equations
   for θ (Fischler & Bolles 1981). All pairs are enumerated when
   K(K − 1)J²/2 ≤ 2000; otherwise draws come from a fixed-seed
   generator.

**What this replaces:**
- the sign-free fit and `_N_SIGN_PAIRS` (`_thiele_innes_fit_sign_free`);
- `_N_FLUX` and the 0.7/0.95 flux seeds. The flux seed is the
  responsibility-weighted mean of the assigned peaks' fluxes.
- `skip_flagged` and the night flags. A night with ambiguous peaks
  simply has spread weights; a bad night lands in π_k0.

**Labels and flux flips.** For a point pair with no differential phase,
(x, f) ↔ (Rx, 1/f) is exact, where R adds 180° to ω. With
`flux_range` ⊂ (0, 1], exactly one of each pair lies in the domain, so
the domain is a fundamental domain. Seeds come from per-night fits in
the same domain. So `flip_flux`, `_flip_modes`, `_flipped` and
`_flip_is_relabelling` are deleted.

With a resolved primary the label is physical (the disc belongs to one
star). **Decision 5 (Ben, 2026-10-28): no upper number.** Flux stays the
faint-to-bright ratio in (lo, 1]. When θ is fitted, one discrete label
d ∈ {disc on the brighter star, disc on the fainter star} is added and
**both are enumerated** (prior ½ each), instead of widening the flux
range. Each band's evidence is Z_n = ½ (Z_n,bright + Z_n,faint); the
summary reports p(disc on the brighter star) with its IS SE. At θ = 0
the two labels are the same model, so the label is enumerated only when
θ is fitted, and the Savage–Dickey verdict (§2D) decides whether it is
reported. The flux lower bound lo is §2I.

`labels_free` asks a different question from the exact symmetry above
(which would make any such ratio identically ½). Let M x be the orbit
with the companion reflected through the primary (ω + 180°) and the
**flux kept**, as `labels_free` compares today. M x differs from x in
closure-phase sign for a contrast pair and coincides with Rx for a
twin. The verdict is p_label = Z_x/(Z_x + Z_Mx), Z_Mx being the
evidence of the mode refined from M x. The label is decided when its
95% credible set has one member (p_label ≥ 0.95 or ≤ 0.05); otherwise
it is reported as `labels_free` with p_label. A contrast pair is
decided, a twin is not. This replaces `LABELS_FREE_NATS` and
`_LABELS_FREE_RES_FRACTION`.

**Tests:**
- EM's final S against brute-force enumeration of all (J + 1)^K hard
  assignments, for K ≤ 6, J = 2. EM must reach the best hard assignment's
  mixture likelihood to within 10⁻³ nats.
- A constructed twin (Δmag 0.05) with one night's peaks swapped: the
  true orbit is seeded, and p_label ≈ ½ ± SE; a contrast pair
  (Δmag 2) is not `labels_free`.
- A constructed sign-error night: it is assigned to π_k0 or to its
  mirror peak, and the period is unchanged.

### C. Band evidence at bounds

The band evidence is the integral over the band's box (and disc, below):

  Z_n = ∫_{box_n} L(x) π(x) dx.

**Estimator.** Use truncated importance sampling with a defensive
mixture proposal q = Σ_m α_m t₅(μ_m, Σ_m) over the band's distinct
modes m:
- μ_m: the mode;
- Σ_m: its Laplace covariance;
- α_m ∝ the mode's Laplace mass.

Then

  Ẑ = (1/n) Σ_i L(x_i) π(x_i) 1[x_i ∈ box] / q(x_i).

Ẑ is unbiased whatever the bounds. One mixture proposal per band also
stops overlapping modes being counted twice, which per-mode
Laplace-then-logsumexp does. The variance-based standard error is

  SE(ln Ẑ) ≈ √(1/ESS − 1/n), where ESS = (Σw)²/Σw².

n starts at `n_is` and doubles until SE ≤ 0.2 nat, up to `n_is_max`.
Two independent estimates (blind and oracle, or two bands) then differ
by an SD of at most √2 × 0.2 ≈ 0.28 nat, so the gate's 1-nat miss
threshold sits beyond 3σ; the gate also adds the reported SEs to the
threshold (gate M4). ln Ẑ is biased low at small ESS (Jensen), another
reason for the tight target. A band still above 0.2 at `n_is_max` is
reported with its SE and flagged `evidence-noisy`; no Laplace
substitution is made. When every draw falls outside the box (p̂_in = 0)
the band is recorded as `evidence-failed`, with no number and a
warning, and is listed separately from the band probabilities; it is
never silently given p = 0.

**What this deletes:**
- `ess_min` and `flag_nats`;
- the laplace-is-differ switch and `bound_sigma`;
- #569's on-bound Hessian surgery and its 1e-6 floor;
- `_PERIOD_*`. The flag "period at the window edge" becomes "the
  window-edge cut removes more than δ of the mode's mass", estimated as
  the fraction of IS draws that fall beyond the edge.

**Reparameterisation.** Sample (h, k) = (√e cos ω, √e sin ω) in the
disc h² + k² ≤ e_max instead of (e, ω). The map has Jacobian
|∂(h,k)/∂(e,ω)| = ½, so uniform on the disc is uniform in e on
[0, e_max] times uniform ω: the prior is unchanged. The e = 0 boundary,
where most circular orbits pile up against a bound, disappears. The disc
edge at e_max is handled by the indicator.

(h, k) alone is not enough: near e = 0 the likelihood depends only on
ω + 2π(t − τ)/P, so with the periastron time τ kept, (h, k, τ) has a
curved ridge along ω + 2πτ/P = const as √e → 0. The standard choice
(Ford 2006; Eastman et al. 2013) pairs (h, k) with the mean longitude at
a reference epoch, λ_ref = M(t_ref) + ω, which replaces τ (uniform on
[0, 2π), the same prior as uniform τ for fixed P and e, since the map
τ ↦ λ_ref is a shift). **Accepted (Ben, 2026-10-28)**: (h, k, λ_ref) together; it
changes the packed vector and the NUTS sites.

**Tests:**
- a 2-D Gaussian truncated by a box with the mode on the bound and off
  it: Ẑ against the closed form (erf products), to within 3 SE;
- SE calibration over 200 seeds: the empirical z-scores have standard
  deviation in [0.8, 1.25];
- a mode at e = 0: Ẑ in (h, k, λ_ref) equals Ẑ in (e, ω, τ) by
  quadrature, to within 3 SE, with ESS/n no worse than at e = 0.5.

### D. Primary diameter

**Always fit θ**, with a uniform prior on [0, θ_max]:
- θ_max = 1.2197 res, the uniform disc's first visibility null at the
  longest baseline (2J₁(z)/z = 0 at z = 3.8317, so θ = 3.8317/π · λ/B;
  `FIRST_NULL` in `_checks`);
- below θ_max, V² falls monotonically with θ on every baseline, so the
  diameter is identified from V² alone;
- a uniform prior includes θ = 0, so the point primary is nested.

**Departure: uniform rather than log-uniform.** A log-uniform prior,
the Jeffreys default for scales, needs an arbitrary lower bound and does
not nest the point star. **Accepted (Ben, 2026-10-28)**, and logged in
the methods register (§2H) as a departure from the Jeffreys default.

**Verdict.** The "resolved" verdict is the Savage–Dickey density ratio
(Dickey 1971; Verdinelli & Wasserman 1995):

  B_point/disc = p(θ = 0 | D) / π(θ = 0) = θ_max · E_post[ L(0, ψ) / ∫₀^{θ_max} L(θ, ψ) dθ ].

The conditional ∫dθ is a 1-D Gauss–Legendre integral per posterior draw
(Rao–Blackwellised). With prior odds 1 between point and disc, the
summary reports p(disc) = B_disc/point / (1 + B_disc/point); the primary
is "resolved" when p(disc) ≥ 0.95, "point" when ≤ 0.05, and otherwise
both are reported.

**Tempering.** *f* comes from the per-night fits with θ free, which is
the nesting model. This is one pass, with no point/disc ping-pong.

**What this deletes:**
- `_auto_primary`, `_pilot_diameter` and the re-temper retry;
- `estimate_primary_diameter` with `RESOLVE_NATS`, `_DIAM_FRACTIONS`
  and the curvature step;
- `_DIAM_*_AUTO`, `diam_seed` and the clip factors;
- `_N_DIAM_NULL`: the null integrates θ by Gauss–Legendre.

`primary_diam="auto"` becomes a deprecated alias of `"fit"`;
`primary_diam=<float>` (fixed) and `None` (a point) remain.

**Tests:**
- Savage–Dickey against the direct evidence ratio, Z_disc/Z_point by
  quadrature, on a constructed 1-parameter V² problem, to 10⁻⁶;
- an injected θ = 0.5 res is resolved; θ = 0 gives ln B < 0 in 95% of
  50 seeds.

### E. Distances

**One unit and one tolerance.** The unit is res = λ/B_max, computed once
in `prepare`. The tolerance is ½ res: half the shortest fringe period,
the Nyquist spacing that already sets the grid. Two optima closer than
this are not separable maxima of a surface band-limited at B_max/λ.

**Uses:**
- mode and candidate dedup, using the maximum over nights of the track
  separation;
- the reflection match;
- `_MAX_SHIFT`;
- the motion limit of `period_window`.

**Replaces:** `min_distance_mas` (0.5 mas absolute), #569's
min(0.5, 0.5 res), `_LABELS_FREE_RES_FRACTION` 0.25 and
`flag_sep_fraction`.

`_REFINE_WINDOW` = 3 res becomes the refinement box: basin radius (½ res)
plus one grid step plus the 3σ position error. It is computed, not
fixed.

**Test:** two synthetic tracks ¼ res apart merge, and two 1 res apart
stay distinct, across res ∈ {0.5, 5, 50} mas.

### F. Cycle counts (which bands get a full fit)

**Score.** The seed profile S_n is the best S of B within band *n*, plus
ln Δln P_n, the band's prior mass under the log-uniform period prior. It
is a Laplace-level surrogate of ln Z_n.

**Jitter.** One jitter σ_J is added in quadrature to every C_kj and
fitted by maximising logsumexp_n S_n(σ_J), a type-II maximum
likelihood over a 1-D grid. This replaces `ARC_JITTER` = 0.1 and the
χ²/dof rescaling.

**Rule.** Fit band *n* fully if and only if

  S_best − S_n ≤ ln(N_bands/δ).

Under the surrogate, the pruned posterior mass is then

  Σ_pruned e^{S_n − S_best} ≤ N_bands · δ/N_bands = δ

relative to the best band. Nothing is re-admitted, which deletes the
re-admission step, `_seed_scores` and `_N_CHECK`. `ARC_DELTA_CHI2` goes.

The bound is only as good as the surrogate. **F is opt-in and
experimental (Ben, 2026-10-28): short arcs default to off**, so every
band in the window is fitted by default. F stays behind its setting,
reported against the all-bands search on stratum D, until vv validates
it.

The pruned count and the bound are reported in the summary.

### G. Thresholds and posterior sampling

Decision 4 (Ben, 2026-10-28). `alias_nats` 10, `no_orbit_nats` 10,
`LABELS_FREE_NATS` 3 and `RESOLVE_NATS` 5 are all deleted, and no
`DECISIVE_NATS` replaces them.

- **Alias, label and resolved verdicts: no threshold in nats.** They are
  reported as probabilities with their 95% credible sets. "Decisive"
  means only that the 95% credible set has one member. Whether these
  probabilities are honest is what the gate's M1 and M9 check by
  injection–recovery.
- **Detection (orbit against no orbit): an empirical false-alarm
  probability.** Theory is unreliable here: the detection tutorial
  found the empirical 3σ threshold at Δχ² 15.2 ± 0.9 where Wilks gives
  9. So for each data set:
  1. the statistic T is the cheap-tier detection statistic,
     T = max_n S_n − S_null (the seed profile of §2B/§2F against the
     single-star fit, no search over full fits);
  2. null data sets are simulated on the target's own coverage and
     errors with the existing `virgil.detection` simulators
     (`gaussian_null` with `rescale_errors`, or `bootstrap_null`),
     including bad nights and the misspecified single-star kinds drawn
     as in gate stratum N (envelopes with chromatic line features, an
     elongated primary, a faint background source);
  3. FAP = (1 + #{T_null ≥ T_obs}) / (1 + n_null), the `DetectionMC`
     convention, with its binomial MC error reported. n_null is raised
     until the FAP's 95% interval excludes α or reaches a stated cap,
     in which case the result is "inconclusive" (n_null ≥ 1/α · 10 =
     1000 at α = 0.01 resolves FAP to ±0.003);
  4. an orbit is reported only if FAP < α.

  Any statistic gives a valid FAP; T is chosen because one value costs
  one seeding pass, not a full search. The full search runs only once,
  on the real data. How powerful T is gets checked by M8 (lost
  detections). The `DetectionMC` hook needed is a user statistic in
  place of the grid statistic; nothing else in `virgil.detection` is
  new.
- **α = 0.01 (Ben)** is the single user-facing number, stated as a
  false-alarm probability (1e-3 and 3σ judged too conservative).
- **Multimodal results are a first-class outcome.** The summary
  never forces a single winner. It lists the 95% credible set of bands
  C(0.95) (smallest set by decreasing *p* with total ≥ 0.95), each with
  *p*, its IS standard error, *P* and its interval. When |C(0.95)| = 1
  the verdict is the band; otherwise
  it is "ambiguous: *k* modes", with the note that RV data or
  astrometry at other epochs is needed to break the tie. The gate's
  headline criterion (M9) is the calibration of exactly this set.
- **Posterior sampling:** sample bands in decreasing *p* until the
  remaining mass is below δ_post = 0.01. This merges
  `posterior_p_min`, `max_modes` and `p_min_samples`. 0.01 is below the
  binomial resolution of the gate's coverage tests: 1/√240 ≈ 0.065 at
  the stratum size.
- **Random starts** are drawn from the prior, as one `n_random`.

### I. Flux prior bounds (decision 5, Ben 2026-10-28)

The flux ratio prior is log-uniform on (lo, 1]. The upper end is 1 by
construction (§2B, with the disc label for a resolved primary). The
lower end is a meaningful prior bound — the faintest companion the
search is for — and it has two sources:

- **Default: a contrast-curve floor from the data.** Using existing
  code only (no new contrast machinery):
  1. per night *k*, the single-companion grid fit
     (`grid_fit.optimized_flux_grid` and
     `grid_fit.laplace_flux_uncertainty_grid`) gives the Laplace flux
     error σ_k(p) at each grid position p;
  2. a coherent orbit fit accumulates S/N over nights, so the combined
     error at separation ρ is σ(ρ) = (Σ_k ⟨σ_k(p)⁻²⟩_{|p|=ρ})^{-1/2},
     the azimuthal mean taken over the grid ring at ρ;
  3. the 2σ-equivalent limit is
     `limits.ruffio_upperlimit(0, σ(ρ), 0.9545)` (Ruffio et al. 2018,
     already in `references/provenance.toml`; verified on ADS:
     2018AJ....156..196R). The mean is set to 0, the no-detection case,
     so a detected companion never raises its own floor;
  4. lo is the geometric mean of that limit over the separation grid
     inside the prior's support (|p| ≤ a_max(1 + e_max), a ≥ a_min).
     Geometric, because the flux prior is log-uniform.
- **Override: the user's search domain.** An explicit `flux_range` (or
  `flux_min`) states the domain: e.g. the faintest companion reported in
  the literature for this class, rounded down a decade, or orders of
  magnitude lower when searching for planets. The docs call this a
  science choice. The summary records which source set lo and its value.

**Why it is principled.** Below the floor the likelihood is flat in
flux (the companion is undetectable), so the posterior shape does not
depend on lo; only the Occam term, ln(1/lo) per unit of log-flux prior
width, does. The floor therefore sets the prior volume of "a companion
the data could see", which is what the orbit-against-null comparison
should weigh.

**Consequences, stated because the prior depends on the data:**
- gate stratum A keeps a fixed, explicit floor (SBC requires the prior
  not to depend on the data);
- strata B and N, and the per-data-set null simulations of the
  detection FAP (§2G), use the default rule;
- B truths below their system's floor are counted and reported, and
  count as misses in M9 (no exclusion).

**Tests:** on constructed single-star data the floor matches
`ruffio_upperlimit` on the analytically combined σ to 10⁻⁶; doubling the
number of identical nights lowers lo by √2; an explicit `flux_range`
overrides it and is recorded.

### H. Methods register and opt-in

AGENTS.md ("Methods beyond published practice") requires every
departure to be registered with a status, references and pinning tests,
and keeps novel or experimental methods opt-in until virgil-validation
validates them. The register entries (to be added to AGENTS.md's table
by the stage that introduces each):

| method | status | references | pinned by |
|---|---|---|---|
| night-outlier mixture, ε ~ Beta(½, ½), exact Gauss–Chebyshev | derived (mixture: Hogg, Bovy & Lang 2010; quadrature: exact for polynomials) | Jeffreys 1946; Hogg et al. 2010 | §2A tests |
| peak-mixture EM seeding with RANSAC starts | novel / experimental | Dempster et al. 1977; Fischler & Bolles 1981 | §2B brute-force test |
| ln(N/δ) band-pruning rule | novel / experimental | — (derived bound, §2F) | §2F test, gate M4 |
| truncated defensive-mixture IS with adaptive n | established | Owen & Zhou 2000 | §2C tests |
| uniform-θ primary with Savage–Dickey verdict | derived | Dickey 1971; Verdinelli & Wasserman 1995 | §2D tests |
| (h, k, λ_ref) parameterisation | established | Ford 2006; Eastman et al. 2013 | §2C test |
| flux floor from the data's contrast curve (2σ Ruffio limit, geometric mean over separations) | derived, default (accepted by Ben, 2026-10-28). Ben's framing: it represents the range of stellar companion fluxes; for a large enough data set it is an approximate empirical-Bayes (hierarchical) prior, "a bit of a dodgy hierarchical inference". Justification: below the floor the likelihood is flat, so only the Occam term ln(1/lo) depends on it | Ruffio et al. 2018 | §2I tests; gate (c) of §2I |
| detection by per-data-set empirical FAP < α = 0.01 | established (Monte Carlo test; the `virgil.detection` convention) | Fawcett 2006 (as `DetectionMC`) | gate M6, M8 |
| disc-label enumeration for a resolved primary | derived (exact enumeration of a binary latent) | — | §2D, §2B tests |
| uniform θ on [0, θ_max] for the primary diameter | departure from the Jeffreys log-uniform default (nests the point star) | Dickey 1971 | §2D tests |
| null-model outlier component (ii or iii) | novel / experimental until chosen on D | orbit_gate_042 §3.6 | gate M6, M8 |

**Rollout:** all of Stages 0–6 ship in 0.4.2 (Ben, 2026-10-28). Each
stage lands its method behind a setting, off by default, plus its
register entry. The gate is run with it on. The default flips in a
separate PR only after that stage's gate passes against the baseline
and a vv check covers it, and **the PR that flips a default deletes the
old path in that same PR** (no period with two live paths after a
flip). F is the exception: it stays opt-in and its flip is not planned
for 0.4.2; for the two
novel entries that check is the gate's stratum-D comparison against
the brute-force (all-bands) search. Each cited reference goes into
`references/provenance.toml` with the code that cites it
(`test_provenance` does not scan `design/`).

## 3. Target architecture

### 3.1 The engine as seven named stages

`fit_orbit_aliases` becomes a short driver (about 60 lines) over seven
stages. Each stage is at most 80 lines, none over 120, and each has a
contract.

| stage | in → out | contract |
|---|---|---|
| `prepare` | data, settings → `Prepared` (nights via `OIData.epochs(gap_days)`, res, `_Layout`, prior box, θ_max) | pure; no fitting |
| `night_summaries` | `Prepared` → per night: peaks (p, C, h, v), b_k, tempering *f*_b | one `epoch_positions` call per night, θ free; *f* from these fits |
| `seed_profile` | summaries → S_n per band, seeds per band, σ_J | §2B EM + §2F; deterministic given the seed |
| `select_bands` | S_n → bands to fit, pruned-mass bound | §2F rule; no re-admission |
| `fit_band` | band, seeds, random prior draws → distinct modes, Laplace covariances | mixture likelihood (A) only; dedup by E |
| `band_evidence` | modes → Ẑ_n, SE, edge mass, flags | §2C |
| `assemble` | bands → `AliasResult` (p_n with SE, the 95% credible set of bands, verdicts by G incl. "ambiguous: k modes", q_k, p_label, Savage–Dickey) | no fitting; JSON via `_io.write_json` |

`OrbitPipeline` keeps its I/O, checks, figures and NUTS; its own search
loop, widening, robustness jackknife and band-sampling rules go.

### 3.2 Module layout (after Stage 5)

Orbit files (orbit physics and the driver only):

| module | holds | lines now → after |
|---|---|---|
| `orbits.py` | orbit classes, `thiele_innes` (one), `period_grid`, `_Layout`, (h, k, λ_ref) map | 1,568 → 1,450 |
| `epochs.py` | `Epochs`, `epoch_positions` (peaks only, no flags), `night_loglikes`, `broad_loglike`, track dedup | 3,016 → 1,200 |
| `aliases.py` | band enumeration, the seven-stage driver, `AliasResult` | 3,459 → 1,400 |
| `_seeding.py` (new, from `arcs.py`) | peak mixtures, EM, `seed_profile`, `select_bands` | 645 → 300 |
| `orbit_search.py` | retired (deprecated in 0.4.2, removed in 0.5) | 1,110 → 0 (0.5) |
| `pipeline/orbit.py` | I/O, checks, figures; NUTS via `pipeline/_common` | 1,678 → 1,000 |
| **orbit total** | | **11,476 → 5,350** |

Core modules (generic, §1.5):

| module | change | lines |
|---|---|---|
| `evidence.py` (new) | importance evidence, Savage–Dickey | +300 |
| `_marginal.py` | `BlockScaleMarginal` (from `epochs._Surface`) | 77 → 680 |
| `likelihood.py` | `group_outlier_mixture` | +120 |
| `fitting.py`, `inference.py`, `_grid.py`, `limits.py`, `detection.py` | multistart, `psd_inverse`, `plan_batches`, `flux_floor`, statistic hook | about +60 net (each replaces orbit copies) |
| `tempering.py` | callers only | 399 → 380 |

Package total for the scope: about 11,950 → 6,870 (−5,080). Tests
shrink with the code, because named-target tests are replaced by
constructed ones (§4), and the moved core pieces get their own
orbit-independent tests.

### 3.3 Efficiency

- **Seeding becomes the cheap tier.** The gallery's two-tier idea
  (cheap per-night screening of the top-K peaks, then full fits for the
  top few bands) is §2B + §2F. With F opt-in and off by default, every
  band in the window is fitted, seeded by §2B, so the default's cost is
  the all-bands cost; F (opt-in) cuts it to bands within ln(N/δ) ≈ 12
  nats of the best.
- **No polish pass and no flip refits.** That roughly halves the
  optimiser calls per band. There is also no jackknife (`_robustness`
  re-ran a full search per outlying night).
- **Adaptive IS** spends samples only on bands that need them.
- **One `night_loglikes`** compiles once per data set, instead of three
  scene→score paths compiling separately.
- **Measure, don't assume.** Every stage PR reports wall time and peak
  memory on stratum D (gate §3.5), against the baseline.

## 4. Tests to replace (named targets out, constructed cases in)

| test | now | replacement |
|---|---|---|
| test_aliases.py:2331-2344 | Gl 229 gaps and *f* = 41.1 pin `_signs_trusted` | deleted with the function |
| test_aliases.py:2347 | seasons "spaced like Gl 229" | a generic two-season schedule drawn from gate template S2 with a fixed seed; asserts the true band is within the fitted set and p_true ≥ the gate's calibration, not rank 1 |
| test_aliases.py:34-56 (12.1 d, N = 34 fixture) | Gl 229 B-like numbers used everywhere | `_fixture(seed)`: period, *e* and schedule drawn from gate stratum D's generator; tests assert relations (true band fitted, coverage), not specific numbers |
| test_arcs.py:22-27 | "Gl 229 Ba–Bb-like" | stratum-D fixture |
| test_orbits.py:370 | Gl 229 B baseline pins k = 9 | phase drift ≤ 1/k for random (span, P-range) draws |
| test_night_flags.py:90 | AK For χ² per mode | deleted with `_flag_chi2` |
| test_night_flags.py:190 | "Gl 229 gate regression" polish | deleted with the polish; A's tests cover the mixture |
| test_night_flags.py:343 | "AK For-like" | a sub-resolution constructed binary (a = 0.3 res) |
| test_orbit_positions.py:671 | "the Gl 229 failure" | a constructed night with two near-equal peaks, both entering the seed mixture |
| test_epochs.py:806 | Gl 229 2024-02-27 night flags | deleted with the flags |
| test_epochs.py:877, 958; test_tempering.py:525, 569, 913; test_pipeline_orbit_resolved.py | ι Peg 1.05 mas | θ drawn uniform in [0.2, 0.8] res; assert Savage–Dickey verdict and coverage |

Constructed tests assert inequalities with stated slack (SE multiples,
binomial quantiles), never a number measured on a named target.

## 5. Public API changes and external callers

Deprecations are only for public names with callers outside the
library. Each takes a `FutureWarning` in 0.4.2 and is removed in 0.5.

| name | external callers | 0.4.2 | migration |
|---|---|---|---|
| `fit_orbit_aliases(sign_free, flip_flux, short_arcs, outlier_eps, skip_flagged)` | vv `oidb_fit.py:564,666`, `reanalyse_orbits.py:785`; tests | accepted and ignored, with a `FutureWarning` naming the replacement (A/B/F) | drop the kwarg |
| `fit_orbit_aliases(primary_diam="auto")` | pipeline default; vv | alias of `"fit"`, warns | `"fit"` |
| `OrbitPipeline` settings `drop_outlier_night`, `alias_nats`, `no_orbit_nats`, `posterior_p_min`, `max_modes`, `outlier_eps` | vv configs | warn and ignore; `alpha` (FAP, default 0.01) replaces the two nats | |
| `estimate_primary_diameter` | tests, notebook only | removed (no external caller) | `fit_orbit_aliases(primary_diam="fit")` |
| `start_from_positions`, `OrbitStart`, `rank_orbits`, `chain_starts` | vv `test_epochs.py`, `oidb_fit.py:456`; `~/data/eso_binaries/gravity/scripts/fit_orbit.py`; `benchmarks/orbits/run_bench.py`; tutorial notebook | deprecated | `fit_orbit_aliases`, then `AliasResult` modes; notes for each script in the Stage 5 PR |
| `orbit_search.score_orbits`, `SharedFlux`, `OrbitScores` | `~/data/eso_binaries/gravity/scripts/{score_modes,refine_modes,nested_modes}.py` | deprecated (not exported; warn on import) | `fit_orbit_aliases(p_range=…, init=…)` |
| `epoch_positions(flag_*)` | vv (9 calls); `~/data/.../fit_orbit.py:71` | `flag_*` warn and are ignored; `.flags` returns empty | read peaks and weights |
| `AliasResult.to_json(n_samples=0)` | vv `reanalyse_orbits.py:810`; `~/data/oidb_census/write_census.py:24` | unchanged | — |

`~/data` scripts are Ben's. The Stage 5 PR lists the edits each needs;
it does not edit them.

## 6. Stages (PR-sized Sonnet briefs)

Every stage:
- branches from current `main` and merges `main` (never rebases);
- runs `.venv/bin/python -m pytest` on the touched tests, plus the fast
  suite;
- quotes the gate summary (Ben runs the array) at its head beside the
  baseline, plus the raw showcases;
- uses Oxford -ize spelling and terse docstrings, with one `Source:` line
  per formula;
- adds each new reference to `references/provenance.toml`;
- gets an Opus equivalence review where it is a refactor (Stage 5).

**Stage 0: settle the open PRs.** Land #569's tempering fix: per-night
scale normalisation, whitened CP through `cp.basis`, and its angle wrap.
Drop its on-bound Hessian surgery, its p_in floor and its
`min_distance` change, since C and E supersede them. Close #572: it adds
ω-reflection doubling, 0.7/0.95 seeds, a pooled χ² fallback and a
seed-picked SEED_ALIAS_NIGHT. Its realistic flipped-night test idea
moves into Stage 3 as a constructed case. The decision on both PRs is
Ben's.

**Stage 1: harness and baseline (vv).** `scripts/orbit_population.py` as
in gate §8, **built on the existing injection–recovery infrastructure,
not a new stack** (Ben, 2026-10-28):
- the hashed criteria file and Holm aggregate pattern of
  `scripts/detection_campaign.py` (`criteria_hash`, `design_hash`,
  `virgil_commit`, `aggregate`);
- the SBC rank machinery of `scripts/sbc_v2.py` (`analyse`,
  `unhealthy`, chain-health rules) for stratum A and M2;
- `virgil.detection` (`gaussian_null`, `rescale_errors`,
  `bootstrap_null`, `injection_recovery`, `DetectionMC` with its
  fingerprints and save/load/concatenate) for the per-data-set no-orbit
  FAP;
- `scripts/pull_campaign.py` and `benchmarks/polar_disk/v4_injection_recovery.py`
  as models for pulling and per-system array tasks.

Genuinely new, with reasons: the orbit population draws (strata A, B,
N, D, templates T1–T3, contaminants), since no existing generator
draws multi-epoch orbits; the oracle (`init=` truth) runs and M4/M8,
since no campaign has a search-completeness metric; the band
credible-set metrics M1/M9, since existing campaigns score single
detections. Tests check the hash, the seeding and the aggregator on a
toy summary. Ben runs the baseline on `main`. No virgil changes.

**Prerequisite (before Stage 4's detection rule):** pull and report
detection v3 (OzSTAR job 18467928, submitted 2026-10-12; criteria
registered in vv 099b418/82e6e43, with the tail primary
`crosscheck.tails`; `~/code/ozstar_scripts/results/detection_v3` is
still empty). The no-orbit FAP relies on `virgil.detection` being
trusted, including in its tails. Ben pulls the results; an agent then
aggregates and reports them raw.

Every default flip deletes the path it replaces in the same PR (§2H).

**Stage 2: mixture (A).** Includes both null candidates (§2A) behind a
setting; Ben runs gate §3.6 on D, and the PR that sets the chosen
default deletes the other.
- Files: `aliases.py` (`_with_outlier` → `_mixture_loglike`; delete
  `_plain`, the polish and `_NO_OUTLIER`), `epochs.broad_loglike`
  (extent from the prior), `pipeline/orbit.py` (delete `_robustness`,
  `_outlying_night`, `drop_outlier_night`, `_WIDEN`), `epochs._flag_chi2`
  (including its mode grouping, which disables flagging for a mode with
  `len(idx) < 3` at `epochs.py:2009`, and its AK For-shaped test at
  `tests/test_night_flags.py:90`); the broad `except Exception` clauses
  of `aliases.py` at 1211, 1228, 1352, 1543, 2799 and 3356 (at `0c20064`).
- Tests: §2A's four.
- Default change: every fit carries the mixture; `outlier_eps`
  deprecated.

**Stage 3: seeding (B, E).**
- Files: new `_seeding.py` from `arcs.py`; `orbits.thiele_innes` (one
  copy); delete `_thiele_innes_fit_sign_free`, `_signs_trusted`, flips,
  the sign doubling, `_N_FLUX`, the flag machinery in `epoch_positions`,
  `sidelobe_flags` and `skip_flagged`.
- Tests: §2B and §2E, plus the constructed flipped night. The
  seed-scanned `AMBIGUOUS` fixtures (`tests/test_orbit_positions.py:86-91`,
  seeds 27 and 65, "found by a scan of seeds 0-140") become constructed
  cases.
- Default changes: `sign_free`, `flip_flux` and `skip_flagged`
  deprecated; `labels_free` becomes p_label.

**Stage 4: bands and evidence (C, F, D, G).**
- Files: `aliases._band_evidence` (truncated IS, adaptive n), the (h, k)
  layout, `select_bands`, the diameter (delete `_auto_primary`,
  `_pilot_diameter` and `estimate_primary_diameter`), deletion of every
  nats threshold (§2G), the empirical-FAP detection with a user-statistic
  hook in `DetectionMC`, and pipeline sampling by remaining mass.
- Tests: §2C, §2D, §2F and §2G.
- Also the absolute default `fit_orbit_aliases(min_distance_mas=0.5)`
  (`aliases.py:2891`), which becomes resolution-scaled as in
  `epochs.py:1168` (E, distances).
- Also §2I: flux floor from the contrast curve (reusing `grid_fit` and
  `limits.ruffio_upperlimit`) and the disc label.
- Default changes: "auto" → "fit"; flux floor from the data; band sampling; short arcs off (F
  opt-in); `span_factor` deleted (long-period band); `vis_amp` range
  taken from the data's instrument package; verdicts as probabilities
  and credible sets; detection at FAP < α = 0.01 (the `alpha` setting
  replaces `alias_nats`/`no_orbit_nats`, which warn and are ignored).

**Stage 5: structure (§1.3, §1.4, §1.5, §3).** Generic pieces move to
their core homes (§1.5); stages 2–4 already write new generic code
(mixture, IS evidence, Savage–Dickey, flux floor) directly in core,
never in an orbit file. The seven stages, `_Layout`,
the one scene→score path, one PD/covariance helper, one batch planner,
atomic JSON, `gap_days` threaded through, and the `orbit_search` and
`start_from_positions` deprecations with migration notes. It must not
change behaviour: the gate summary and the stratum-D fits must be
bit-for-bit or within IS SE of Stage 4. Opus equivalence review.

**Stage 6: docs.**
- Update `docs/orbit_fitting.md`, `docs/orbit_methods.md`,
  `which_orbit_tool.md` and AGENTS.md (the "Label swap" row).
- State that a multimodal result is reported, not resolved: the
  summary gives the 95% credible set of bands with *p*, its SE, *P* and
  its interval, and only RV data or astrometry at other epochs breaks
  the tie.
- Explain M9 plainly, **Bayesian language first** (Ben found it
  "reasonable, if confusingly stated in frequentist terms"): what a
  band probability means (the posterior probability that the true
  period lies in that alias band, given the data and the priors); what
  "the true orbit lies in the 95% credible set" means; and why
  injection–recovery checks that these probabilities are honest (over
  many simulated systems, sets that claim 95% must contain the truth
  about 95% of the time). Only then the frequentist coverage reading.
- Explain detection as a false-alarm probability against simulated
  single stars like this one, at α = 0.01, and why it is simulated
  rather than taken from Wilks' theorem.
- Add a Known issues entry: evidence is overconfident when χ²/N on the
  quoted errors is well above *f*. Lead every summary with raw χ²/N.
- The tutorial notebook. Public docs never link `design/`.

## 7. Adversarial-review checklist (every stage PR)

The reviewer answers each with evidence (file:line, a test, or a
number):

1. **New constants.** Does any new number lack a derivation or citation
   in the code and here? Is an efficiency-only constant shown not to
   change a result beyond its SE?
2. **Targets.** Does any test, comment, constant or commit message name
   or numerically match a real system (12.1 d, 34 cycles, 1.05 mas,
   41.1, …)? Grep for them.
3. **Second paths.** Is there any fallback, `except Exception`, retry,
   re-admission, "polish" or second likelihood? Each needs a stated
   reason; a failure must be recorded, never silently replaced.
4. **Gate and tests.** Was any gate threshold, population, test slack or
   fixture changed after seeing a result? The gate change log must say
   "no".
5. **Baseline deltas.** Gate summary at head against baseline: did any
   metric get worse? A worse showcase is reported, never fixed by
   tuning.
6. **One path.** Does the PR add a second way to do something §1.3 says
   is done once?
7. **Deletions.** Are the deleted lines really gone (`git diff --stat`),
   with no dead helper left behind (grep callers in src, tests, vv and
   `~/data`)?
8. **Deprecations.** Does every public kwarg or function with an
   external caller warn (`FutureWarning`, naming the replacement), and
   does nothing without external callers carry a shim?
9. **Maths.** Is each formula tested against an analytic or brute-force
   reference at the tolerance §2 states?
10. **Efficiency.** Wall time and memory on stratum D against baseline;
    any slowdown is above 20% explained.
11. **No orbit-local generic code.** This grep returns nothing (run on
    the orbit modules after Stage 5):
    `grep -nE "def .*(optimi[sz]e|minimi[sz]e|lbfgs|newton|evidence|importance|laplace|nuts|mcmc|hessian_inv|cholesky|logsumexp_mixture|write_json)" src/virgil/{orbits,epochs,aliases,_seeding}.py src/virgil/pipeline/orbit.py`;
    and no `NUTS(`, `MCMC(` or `json.dump(` appears in those files.
12. **Moved functions have callers.** The Stage 5 Opus equivalence
    review checks that each function moved to core (§1.5) is used by
    both orbit and non-orbit callers, or carries a written reason, and
    that its behaviour is unchanged (same outputs within tolerance on
    stratum D).

## 8. Decisions (Ben, 2026-10-28)

Accepted:
1. **Uniform θ** on [0, 1.2197 λ_min/B_max]; registered as a departure
   from the Jeffreys default (§2H).
3. **No explicit sign component**, conditional on the sign-error ROC
   being reported and M5 passing (§2B).
6. **Close #572, trim #569** to its tempering fix.
8. **Delete `drop_outlier_night`** and the jackknife.
9. **Deprecation timeline:** `estimate_primary_diameter` removed now;
   `start_from_positions`, `rank_orbits`, `chain_starts` and
   `orbit_search` deprecated in 0.4.2, removed in 0.5.
10. **(√e cos ω, √e sin ω, λ_ref) parameterisation.**
12. **`a_range` in resolution units.**

Changed:
2. **The null:** a no-mixture null is rejected. Candidates (ii) shared
   mixture and (iii) shared noise inflation are chosen between on
   stratum D by the pre-registered rule (gate §3.6).
7. **Short arcs default off;** F is opt-in and experimental.
11. **`_VIS_AMP_DEFAULT` moves to the instrument layer**; no invented
    citation.
13. **`span_factor` deleted;** one long-period band takes everything
    beyond the span (§1.2).
14. **Gate approved** with: misspecified nulls in N; B's misspecification
    rate 0.3 and M3b must-hold on that subset; the null-choice rule.
15. **All stages (0–6) in 0.4.2;** the PR that flips a default deletes
    the old path in the same PR.

5. **Flux bounds:** no upper number; flux in (lo, 1] with the
   disc-on-brighter/fainter label enumerated for a resolved primary.
   Lower bound lo from the data's 2σ contrast curve by default, or the
   user's stated search domain (§2I).

4. **Thresholds:** no nats anywhere. Alias, label and resolved verdicts
   are probabilities with 95% credible sets (calibrated by M1 and M9);
   detection is an empirical per-data-set FAP < α = 0.01 using the
   `virgil.detection` simulators with stratum-N-like nulls (§2G).

All 15 decisions are settled.

## References

- Dempster, Laird & Rubin 1977, JRSS B 39, 1 (EM).
- Dickey 1971, Ann. Math. Stat. 42, 204; Verdinelli & Wasserman 1995,
  JASA 90, 614 (Savage–Dickey ratio).
- Eastman, Gaudi & Agol 2013, PASP 125, 83; Ford 2006, ApJ 642, 505
  ((√e cos ω, √e sin ω) parameterisation).
- Fischler & Bolles 1981, Comm. ACM 24, 381 (RANSAC).
- Hogg, Bovy & Lang 2010, arXiv:1008.4686 (mixture models for
  outliers).
- Jeffreys 1946, Proc. R. Soc. A 186, 453 (Beta(½, ½) for a rate).
- Owen & Zhou 2000, JASA 95, 135 (defensive mixture importance
  sampling).
- Rucinski 1992, AJ 103, 960 (contact-binary short-period limit).
- Ruffio et al. 2018, AJ 156, 196 (2018AJ....156..196R; Bayesian
  upper limits for non-detections).
- Fawcett 2006, Pattern Recognit. Lett. 27, 861 (ROC analysis; the
  `DetectionMC` source).
