# In-flight work vs. the virgil 0.4.0 tag (review, read-only)

Snapshot: origin/main 420d1b7 (2026-10-15), pyproject still 0.3.0, CHANGELOG "Unreleased" holds everything since 0.3.0.
Ratings: none / mark-provisional / should-land-first / API-conflict.

## Summary table

| # | Thread | State | Touches on main | Marked provisional? | Rating |
|---|--------|-------|-----------------|---------------------|--------|
| 1 | `likelihood_grid` keyword-only (branch `likelihood-grid-kwonly`, a415d24, no PR) | finished, 20 files, never PR'd | `grid_fit.likelihood_grid(model, data, grid, *, batch_size)` -> `(model, *, data, grid, batch_size)`; `_deprecate.old_order(keywords=)` | n/a (a signature) | **should-land-first** (API-conflict if tagged without it) |
| 2 | Orbit consolidation, phase 1d tempering (a local worktree, uncommitted) | WIP, no tests, new `src/virgil/tempering.py` | `fit_orbit_aliases(temper=True)` new kwarg, `AliasResult.temper_f/.temper_blocks/.tempering`, `AliasResult.log_z` and band `p` change meaning, `LABELS_FREE_NATS` semantics, `_Settings.cfg` tuple grows, `epochs._Surface.score/weights` per-block `dof`, `estimate_primary_diameter(dof=)`, saved JSON gains keys | no | **should-land-first** |
| 3 | Orbit phase 1e `OrbitPipeline` + 1f docs | not started (plan of 2026-10-15, now `design/orbit_pipeline.md`) | adds `virgil.pipeline.orbit`, changes the `_core.py` base class (multi-night inputs), CLI `_PIPELINES`, `_quicklook._SECTIONS`; marks `start_from_positions` superseded; supersedes `design/automatic_orbits.md` and `orbit_aliases.md` | planned provisional | mark-provisional; see item 8 |
| 4 | DiskPipeline PR #369 | open, MERGEABLE, ozstar-sync checks red (notebook outputs not stamped, awaiting an OzSTAR run) | new `virgil.pipeline.DiskPipeline`, CLI `disk`, `_io.Result` disk branches, `_common._quantiles(period=)`, check names, `virgil-pipeline` | yes (`STABILITY="provisional"`, FutureWarning once, tier written to run.json) | mark-provisional (fine); decide before merge, not a 0.4 blocker |
| 5 | Ensemble / ImagePipeline series | A1 (#370) and A2 (#371) merged; later stages unwritten in repo; `ensemble-window` WIP branch (b876d0f, old, no PR) | `EnsembleSpec` (`window_dex` 1.0 -> 2.0, window direction, `weight_ranges` -> per-group `weight_scale`, `field_factors` (1,2,4)), `draw_groups(..., star=, starts=)`, new `weight_scale`; `gp_members`, `evidence_grid`, `EnsembleSpec.family="gp"`, `gp_*` fields; `diagnose` extra keys | **no** | mark-provisional |
| 6 | Gauss-Newton series (GN1, GN2, GN4 merged; GN3, GN5, GN6 not started) | design/gauss_newton_plan.md | GN3: `fit` default solver rule (kernel up to 50 coords; behind `_GN_DEFAULT_ON=False`), `docs/gauss_newton.md`; GN5: pipeline setting `mass` (additive); GN6: MEM `lm_default` | n/a | none for API; numerics in `fit` may change later. Known: GN4 regression in the alias slow test (band 22 -9.7 nats) |
| 7 | PolarDisk (P1-P6, #335..#357 merged) | complete; DiskPipeline waits for "PolarDisk API unchanged for a release" | `PolarDisk`, profiles, `Harmonics`, `VonMises`, `RadialGP`, `PolarKernel`, `freeze_geometry`, `PolarField(geometry_from=)`, `virgil.harmonics`; subsumed `TruncatedCone`, `ModulatedGaussianRim`, `GaussianArc` (shape change, in CHANGELOG) | no (the design note itself says the API is not yet settled) | mark-provisional (docs/CHANGELOG sentence); not a blocker |
| 8 | Superseded orbit API still exported | main | `virgil.start_from_positions`, `rank_orbits`, `Epochs` at top level; `scales="quoted"` default with FutureWarning (staged flip to `"marginal"`); `EpochPositions.gap` vs `gap_marginal` (gap to switch "after one release"); `OrbitStart`; `orbit_search.score_orbits/SharedFlux/OrbitScores/rank_scores`; `EpochPeaks` | no | mark-provisional / decide |
| 9 | Pipeline contract: StarPipeline/BinaryPipeline declared "stable" in 0.4.0 | main | frozen keys of `run.json`/`summary.json`, check order, settings | stable | **decide before tag** (see below) |
| 10 | Issues #309, #297, #268, #328, #329, #39 | open | see section 10 | no | mostly none; #309, #268 touch public names |
| 11 | Release evidence gate (`release-evidence-gate`, 3b8c49a, no PR) | docs only | CONTRIBUTING.md/AGENTS.md: tier A + single pinned evidence commit + no open `bug` findings + mutation kill rate before tagging | n/a | should-land-first (it is the checklist for this release) |

## Risky items first

### 1. `likelihood_grid` keyword-only is finished but unmerged
`origin/likelihood-grid-kwonly` (one commit a415d24, based on 0c1be69 of 2026-10-12, no PR): only `model` stays positional, `data` and `grid` become keyword-only; a 0.3-order call keeps working with a FutureWarning until 0.5; any other positional call raises `TypeError` naming the keyword call. Also gives `old_order` a `keywords=` argument (private `_deprecate`) and touches `pipeline/star.py`, 20 files in all (3-dot diff; the 2-dot diff against main is huge only because the branch is stale). Rationale in the commit: the 0.4 order change was silent for swapped calls (virgil-validation ledger F21). main still has `likelihood_grid(model, data, grid, *, batch_size=None)` (grid_fit.py:162). If 0.4.0 is tagged without it, narrowing the signature in 0.4.1 is itself a break. Land (needs rebase/merge with main and a PR) before tagging. The same F21 argument applies to any other three-positional model/data/grid function; the branch only fixes this one.

### 2. Orbit phase 1d (tempering) changes the meaning of the evidence the API already returns
A local worktree (branch `orbit-1d-tempering` = main, uncommitted edits to `aliases.py`, `epochs.py`, new untracked `tempering.py`, no tests yet). Changes to public surface:
- `fit_orbit_aliases(..., temper=True)`: new default-on kwarg; every `log_z`, band probability `p`, `labels_free` verdict (`LABELS_FREE_NATS` now "on the tempered evidence"), `LABELS_FREE`, posterior widths change by f at the same inputs. Results from 0.4.0 and 0.4.x would differ.
- `AliasResult` gains `temper_f`, `temper_blocks` and a `tempering` bool property (naming clash with the `temper` kwarg to check); `to_json/save/load` gain keys (loader uses `p.get`, so old files still load).
- `epochs._Surface.score(chi2, dof, data)` / `weights` / `weighted_rows`: per-block `dof` vector (`dof_vector`); `estimate_primary_diameter(..., dof=)`; `_auto_primary(dof=)`; private `_Settings.dof` and `cfg[4]`. The `cfg` tuple grows, so cached/pickled band checkpoints (`aliases._save_band` pickles `(key, done)`) from a pre-1d run are incompatible unless the key includes `temper`.
- New module `virgil.tempering` (`FLAG_RANGE`, `ChannelFactors`, `estimate_factors`, `adjacent_channels`, `lag1_correlation`, `factor_from_correlation`, `MIN_PAIRS`).
The 1c PR (#368, merged) already changed `fit_orbit_aliases` defaults ("results change"); 1c and 1d together are one behaviour change. Tagging between them gives users two different evidence conventions in 0.4.0 vs 0.4.1. Either land 1d first (it needs tests and review) or state in the 0.4.0 notes that alias evidences are untempered and will be tempered in 0.4.x.

### 3. Result and file schemas without a version field
- `AliasResult.save/to_json` JSON (aliases.py:1925-1960): no schema or virgil-version key. It has already gained `unseeded`, `primary_fit`, `window_capped`, `period_at_lower_bound`, `primary_diam`, `vis_amp` since 0.3.0, and gains `temper_f/temper_blocks` in 1d. Loading is tolerant (`p.get`), so adding keys is safe, but renames are not detectable. `design/automatic_orbits.md` §11 says new `save/load` should carry "the pinned virgil version"; `AliasResult` does not.
- `AliasResult` band checkpoints (`_save_band`): a pickle of a dataclass plus a JSON; fingerprint-keyed, no version.
- `detection.DetectionMC.save/load` (.npz with a `meta` JSON): no version.
- `OrbitStart`: not serializable at all (#268 item 4); the plan wants npz+JSON with version when built.
- Pipeline folders have `schema = "virgil-pipeline-run-v1"` (run.json, summary.json, h5 attrs, quicklook metadata) and `load` refuses unknown schemas. Good. Models saved by `save_model` use a spec hash/manifest. Provisional DiskPipeline reuses `-v1` with `stability: provisional`; the schema string does not say provisional, only the `stability` field does.
Suggest a one-line `"format_version"`/`"virgil_version"` in `AliasResult` JSON and `DetectionMC.meta` before the tag; it is cheap and makes later renames detectable.

### 4. StarPipeline/BinaryPipeline become "stable" in this release (first release with `virgil.pipeline`)
`virgil.pipeline` is in "Unreleased", not in 0.3.0, so 0.4.0 is the first public release of it, and `docs/pipeline.md` + `design/pipeline.md` declare Binary and Star stable: frozen constructor/run signatures, stage names/order, setting names/defaults, every `run.json`/`summary.json` key, check names/order/thresholds, file names. In-flight items that conflict with that contract:
- `design/pipeline.md` "Open questions": "Before the first stable release, settle `global_nsigma`" (a Sidak estimate with a nominal trial count; either calibrate by simulation or drop it from the frozen summary keys and check). Still `"global_nsigma"` + `"global_nsigma_method"` in `binary.py:549-550`, quicklook uses it. This is the one frozen key the design itself says may be wrong. Decide now (calibrate, drop, or accept as "approximate" and say so in release notes).
- GN5 will add a `mass` setting (additive, defaults to `"adapted"`; plan handles old `run.json`), fine.
- Issue #309 (closure phases near visibility nulls: `closure="bispectrum"` in `_whiten_phases`, lobe evidence) would change StarPipeline's `multimodal` check/float64 scan strategy (#307 workaround); the pinned thresholds there may move.
- OrbitPipeline (1e) requires a change to the `_Pipeline` base-class input handling (flat input list -> per-night groups). That touches `run.json` "inputs" inventory; should be additive but it is the stable `run.json` schema.
- `design/pipeline.md` calls the imaging pipeline `ImagingPipeline` (CLI `imaging`); PRs #370/#371 and the A1/A2 stages call it "ImagePipeline". Pick the name before anything ships in a table or docs.

## Other in-flight threads (detail)

### 4a. DiskPipeline (#369, branch claude/angry-carson-edc41c, 7 commits ahead)
- Adds `src/virgil/pipeline/disk.py` (2181 lines), `_checks.py` (+347: `chi2_reduced`, `length_scale`, `ring_count`, `inner_edge`, `outer_edge`, `inclination`, `absorbed_companion`, posterior checks), `docs/polar_disk_rings.md`, notebook section.
- Registry `model="rings"|"gp"`; `model=None` means `"gp"` (pinned in a snapshot test) while `design/pipeline.md` (main) still proposes "an error that asks for a choice" as the default; the design note should be reconciled.
- NUTS hyperparameter sampling for the GP is explicitly "a later provisional upgrade"; ring/GP stage and summary keys will change. Companion search opt-in.
- Notebook assertions unverified until the OzSTAR run; ozstar-sync checks fail at head until outputs are stamped.
- Design: "stay provisional until the PolarDisk API is unchanged for a release." So 0.4.0 can include it provisional; it must not be promoted in 0.4.0.
- Shared-code edits: `_io.Result` branches on `pipeline == "disk"`, `_common._quantiles(x, angle=False, period=360.0)` (private).

### 5. Imaging pipeline / ensemble
- A1 (#370, merged): `imaging.chi2_by_observable`, `short_baseline_residual`, `residual_image`; `diagnose` gains `chi2_red_by_observable`, `short_baseline_mean` ("additively").
- A2 (#371, merged): `ensemble.evidence_grid`, `gp_members`, `EnsembleSpec` family `"gp"` and `gp_*` fields, `combine` with GP groups. The PR notes one test failing on main (`test_default_weight_ranges_hold_the_corner_with_room_below[maxent]`), and main CI after #371 is red on x64 shard 1/3 (two alias tests, see below).
- `ensemble-window` (b876d0f, 2026-10-08, one WIP commit, no PR; stale base 5efdb83): data-driven weight scales, `weight_scale(data, draw, reference, star=True)`, `window_dex` default 1 -> 2, `n_weights` 8, `field_factors` (1,2,4), `draw_groups(..., star=, starts=)`, warnings for edge corners. That is exactly the area of the failing test; it would change `EnsembleSpec` defaults and field semantics (partly overtaken by #361, which sizes the field from the data).
- Plan doc for the ImagePipeline is not in the repo (only the name in PR bodies). `virgil.ensemble` and the new GP/ensemble helpers are not marked provisional anywhere. Recommend a "provisional" docstring/CHANGELOG line.
- imaging contest work (virgil-validation) drove #361 (`image_field`, `second_moment_fwhm`, `OIData.unflagged`, `beam/fringe_scales` on unflagged samples); the plan wanted no further virgil API beyond that. Issues #328 (SVD on CPU in `log_evidence`, no API change) and #329 (OIData cannot read the 2014 R Car file: closure triangle with no V2 baselines; would relax the reader, possibly new warning) are open.

### 6. Gauss-Newton
`design/gauss_newton_plan.md`: D1 (Ben 2026-10-14): GN3 merges with the kernel opt-in (`method="lm"`), default flip gated on an OzSTAR comparison, `_GN_DEFAULT_ON = False`. `fit` method rule will change (additive). GN6 (MEM `lm_default`) conditional. Nothing blocks 0.4.0, but the `fit` default solver may change in 0.4.x; say so in docs if `method=None` behaviour is documented. GN4 (#350) reported a 9.7-nat drop of one band's evidence on the slow test; `test_injected_alias_wins_in_closure_phases[12.38]` fails on main per #323 (best band 22, truth 20).

### 7. PolarDisk
All six stages merged (#335-#357) plus docs #356/#367; `FlaredDisk*` stays (decision 2026-10-14, #348); deprecation of `TruncatedCone`/`ModulatedGaussianRim` was to happen "at the next minor release" per the design note but the changelog instead records them running on the engine without deprecation, and `GaussianArc` as a shape change with `FutureWarning` for one release. Open decision in design §9: profile parametrization (positive I0 via exp of GP vs positive parametric forms; `Harmonics(form="log")` deferred). Branch `docs-polar-pilot` is a throwaway pilot script, never to merge. `pds70-disk` (ecd7941, Oct 1, edits `src/drpangloss/...`, pre-rename) is dead: `FlaredDisk*` is on main.

### 8. Orbit API: superseded-but-exported, staged defaults
- Plan (2026-10-15, now `design/orbit_pipeline.md`): single `OrbitPipeline` grown out of `aliases.fit_orbit_aliases` (engine keeps its name), `start_from_positions` and `design/automatic_orbits.md` marked superseded, new `design/orbit_solver.md`. Phases: 1a #359, 1b #362/#366, 1c #368 merged; 1d WIP; 1e, 1f, 1g (conditional) unbuilt. vv phases 2-4 consume `virgil.pipeline.load(run)`.
- Planned `OrbitPipeline` defaults not yet in an API: `period_range=None` (data-computed window via `period_window`), `ecc_max=0.95`, `sqrt_e=True`, `flux_range=(1e-3, 2)`, `flip_flux`, `sign_free`, `primary_diam="auto"`, `vis_amp`, `gap_days`, `flag_nights`, `exclude`, `temper`, `error_scale="fit"`, `n_bands_max`, `n_candidates`, `num_chains`, `num_samples`, `seed`; summary keys and check names (`chi2`, `night_flags`, `labels_free`, `alias_ambiguity`, `tempering`, `primary_resolved`, `no_orbit`, `period_bound`, ...). The engine will need to expose matching kwarg names; keep them consistent with what 1c/1d already ship (`sign_free`, `flip_flux`, `vis_amp`, `primary_diam="auto"`, `p_range`, `n_bands_max`, `temper`) to avoid a later rename: the plan says `period_range`, the engine says `p_range`; the plan says `error_scale="fit"`, check the engine's spelling.
- Staged default in main: `rank_orbits`/`start_from_positions` `scales` (omitting it gives a FutureWarning; default "quoted" in 0.4, to become "marginal") and `EpochPositions.gap` -> `gap_marginal` switch. `automatic_orbits.md` decision 5/6 timeline conflicts with the new "supersede" plan. If `start_from_positions` is superseded in 0.4.x, deprecating a function that was just given new keywords is awkward; decide whether 0.4.0 documents these as legacy.
- `virgil.orbit_search` (`score_orbits`, `SharedFlux`, `OrbitScores`, `rank_scores`) and `EpochPeaks` are public (documented in docs/api) but the new plan routes through `aliases`/GN kernel instead; their future is undecided.
- #268 follow-ups still open: (1) `start_from_positions` refine hook for time-dependent models, (2) decisive-night gap ignores noise scale (done via `gap_marginal`), (3) `Epochs.noise` docs vs fixed values, (4) `OrbitStart` not serializable.
- Main CI (run 38723896496, head 420d1b7): `ubuntu py3.12 x64 1/3` fails `test_a_single_night_alias_is_flagged_and_the_alias_fit_finds_the_band` and `test_skip_flagged_can_be_switched_off_and_unseeded_survives_save_load` (alias tests from 1b). Not an API item, but the alias/epoch engine is unsettled.

### 9. Other branches (all with commits ahead of origin/main)
| Branch | What | Verdict |
|---|---|---|
| `claude/angry-carson-edc41c` | DiskPipeline #369 | see above |
| `likelihood-grid-kwonly` | see item 1 | land first |
| `ensemble-window` | see item 5 | WIP, stale |
| `release-evidence-gate` | release checklist | land first (process) |
| `docs-polar-pilot` | throwaway script | ignore |
| `orbit-1a-result-api` | only a merge commit (#359 already merged) | none |
| `linear-marginal` (#217), `stage6d-jeffreys-priors` (#205), `stage6a*`, `pds70-disk` | already merged or pre-rename | none (stale) |
| `shashank-dev` (16 ahead, 2026-09-20), `copilot/*`, `develop`, `benjaminpope-patch-1`, `feat/fisher-reparam-docs-20260303`, `kernel_phase`, `gh-pages` | old; `GravityDarkenedStar` is on main | none; not examined line by line |
### 10. Open issues
| # | Title | API impact on 0.4.0 |
|---|-------|---------------------|
| 309 | Closure phases near visibility nulls (`closure="bispectrum"`, lobe evidence) | would add an option to `_whiten_phases`/likelihood and possibly change StarPipeline checks; additive; none for tagging, but StarPipeline's `multimodal` thresholds are frozen |
| 297 | `with_closure_offsets(modes=)` across frames | relaxes an error; additive |
| 268 | Visibility orbits stage 2 follow-ups | `Epochs.noise` fixed values vs docs (docs statement may be wrong in 0.4.0: fix the docs or the code), `OrbitStart` serialization, `refine_model=` |
| 328 | `log_evidence` SVD runs on NumPy CPU | internal; none |
| 329 | OIData cannot read 2014 R Car (triangle without V2 baselines) | reader behaviour/warning; none |
| 39 | Split notebooks/tutorials into a separate repo | repo layout; does not block, but touches `docs`/`notebooks` sync scripts and the CI notebook-output gate; do not move before the tag |

### 11. Design-note follow-ups with API weight
- `design/pipeline.md`: `OrbitPipeline` and `ImagingPipeline` planned provisional; `DiskPipeline` open question on default model (now answered in code); `info`/`validate` should avoid importing JAX (needs lazy `virgil/__init__`, which would change top-level imports); multi-file inputs with different instruments (additive); `global_nsigma`.
- `design/spectro_interferometry_workflow.md` "Next": bandwidth smearing (opt-in `EFF_BAND` landed, #345), closure offsets for three telescopes (`gains.closure_offsets` refuses them today), model-comparison helper (χ²/N on uninflated errors), OI_FLUX wavelength rescale, primary beam (coupled flux), `OI_CORR`; open: `noise=` becomes the general nuisance argument (name kept), reference flux at `wavel0`.
- `design/chromatic_sources.md`: `render()` optional `wavel`; bandwidth smearing as a scene wrapper (done differently in #345: opt-in `EFF_BAND`/boxcar channels), grid tools' `flux_param=` for chromatic companions.
- `design/composite_scenes.md` "Open items": `Joint`, Occam-corrected multi-image evidence, parametric envelope component, `SumField` (PR #330 states "plan stage 2: `SumField`; the Laplace evidence still holds the flux ratio at its MAP" is not built).
- `design/sparse_imaging.md`: S3/S4 deferred; L-BFGS step limit for MEM/StarletL1 not investigated; `clean(..)` signatures (`stall_tolerance`, `refit_every`, `spectrum=`) are new in 0.4 and unversioned.
- `design/visibility_orbits.md`: later stages proposed; `design/orbit_scene_joint_fitting.md` real anchor binaries not chosen.
- `design/imaging_plan.md`: Stage 7 MFT-matrix cache, Stage 3c PDS 70 deferred.
- `design/radial_profiles.md` §9: profile parametrization open; stage V (virgil-validation benchmarks vs frank/vis-r/FRAP) not done: i.e. PolarDisk has no independent validation yet (relevant to the `release-evidence-gate`).

## Recommendations
1. Land `likelihood-grid-kwonly` (PR from a merge of main) and `release-evidence-gate` before tagging.
2. Either land 1d with tests, or state in the 0.4.0 notes that alias evidences are untempered and `fit_orbit_aliases` defaults/evidences will change in 0.4.x; mark `fit_orbit_aliases`, `epoch_positions`, `start_from_positions`, `virgil.orbit_search` as provisional in docstrings and CHANGELOG.
3. Decide `global_nsigma` (calibrate, drop from frozen keys, or document) before declaring Binary/Star pipelines stable; settle `ImagingPipeline` vs `ImagePipeline` naming.
4. Add a version key to `AliasResult` JSON and `DetectionMC.meta`.
5. Mark `virgil.ensemble` additions (`evidence_grid`, `gp_members`, `EnsembleSpec` GP fields), `PolarDisk` family, and `DiskPipeline` provisional in the CHANGELOG/docs; keep DiskPipeline provisional.
6. Check main CI (alias tests failing on x64 shard 1/3 at 420d1b7) before tagging.
