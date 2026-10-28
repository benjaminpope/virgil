# Binary orbits: one blind, automatic, user-friendly pipeline

Status (2026-10-20): **the canonical orbit plan.** It supersedes
`history/automatic_orbits.md`, `orbit_aliases.md` and `thiele_innes_marginalisation.md`
(see their headers for what stays valid).
- Phases 1a to 1f are merged: the engine result API (1a, #359), night flags
  (1b, #366), engine defaults (1c, #368), tempering (1d, #372, with
  review follow-up #380), `OrbitPipeline` (1e, #422) and the docs (1f:
  `orbit_fitting.md` leads with the pipeline and shows the engine
  underneath; the pipeline page and API entries; `start_from_positions`
  marked superseded in the docs). No `design/orbit_solver.md`: the v2 solver
  internals were not worth keeping (the prototype LM is never merged).
- Next: phase 2 (vv consolidation) and Phase 3 (the OzSTAR golden table).
- `orbit_search.score_orbits` ships as **provisional** in 0.4.0, with the rest
  of `epochs`, `aliases` and `tempering` (`docs/stability.md`).
- Release gate: the unattended run on the Gl 229 data was the gate for 0.4.0
  (Ben's decision, 2026-10-16) and through 0.4.1; retired 2026-10-23. The release review's proposal to move the gate
  to the release that promotes `OrbitPipeline` was not adopted.

## Context
We have found and fixed the same orbit problems three to five times. A census on 2026-10-15 found five orbit "tracks" in virgil-validation (vv):
- **T1**, `reanalyse_orbits.py`: the alias track.
- **T2**, `oidb_fit.py`: the gallery fits.
- **T3**, `oidb_compare.py` and `orbit_gallery.py`: comparison and rendering.
- **T4**, `trackb_fit.py`: Track B per-epoch positions.
- **T5**, `orbit_diagnose.py`: diagnostics.

A v2 design plans a sixth: a vv `solve_orbits.py` with its own hand-written LM. virgil has four fitting entry points, built or planned. Twenty-one problems were catalogued, and **every fix reached only one or two tracks**:
- **Integer orbit-count degeneracy:** solved in T1, missing from T2 until #124.
- **Thiele–Innes sky-plane comparison:** written twice in `crosscheck`, unused by T3.
- **UD primary, flux flip, label swap, V² amplitude:** virgil options that only T1 passes.
- **Per-night seeds:** point-binary in every track.
- **Published orbits still leak into fits:** warm starts, period windows centred on P_ref, eccentricity caps per system, twin lists, UD diameters from the paper under test.
- **Untempered evidence:** band p = 1.000 even at χ²/N 12.
- **Stale verdicts** on the aliases page.

Agents briefed on one track never saw the others' fixes.

**Goal:** a single `OrbitPipeline` in virgil's existing `virgil.pipeline` package. It takes a list of OIFITS files plus optional metadata and robustly fits an orbit with sensible defaults. Every option stays available, and every solved problem is explained in comments and docstrings and pinned by a test. vv keeps only a thin driver, the independent comparison and rendering, and the duplicates are deleted.

**Ben's decisions (2026-10-15):**
- The fitter lives in **virgil**, grown out of `aliases.py`.
- The costly v2 parts are built **only if a test system fails**.
- **HR 6819:** keep the 10-07 ruling (overlay, no verdict).
- **#124:** fix its blocking threads and merge.
- **A clean pipeline abstraction** whose API matches the existing pipelines, with sensible defaults, then docs and vv updated to match.

Earlier rulings still hold:
- no published starts, windows or priors;
- a tempered likelihood (effective channels), checked by injection;
- report both Gl 229 data selections;
- compare in sky coordinates, Thiele–Innes without RV;
- design on Opus, implement on Sonnet;
- heavy runs go to OzSTAR, and Ben runs every cluster command.

## Principles
1. **virgil fits; vv checks.** vv holds data lists, metadata with sources, rulings, the independent comparison (`crosscheck`, which never imports virgil), injection checks and rendering.
2. **Two layers in virgil, no more:**
   - the **engine**, `aliases.fit_orbit_aliases` (same name, so no rename churn), extended with the new defaults, for power users;
   - the **user-facing `OrbitPipeline`**, which calls it.

   `start_from_positions` and `design/history/automatic_orbits.md` are marked superseded. The v2 prototype LM (`exp3_lm.py`) is never merged; everything uses `_gauss_newton.solve`.
3. **Sensible defaults, every option exposed.** Each default is data-driven or physical, never per-system. Each is a one-line settings bullet in the docstring; why it was chosen goes in the docs page.
4. **Published information reaches only the comparison stage**, enforced by tests.
5. **Self-documenting fixes, under virgil's growth rules (2026-10-16).** Every solved problem has:
   - a **named test** that pins it;
   - **one row** in the "Binary orbits: solved problems" table (`design/orbit_solved_problems.md` here, `AGENTS.md` in virgil-validation), naming that test. Named targets (Gl 229 nights, AL Dor, ι Peg) and design history live here or in design notes, never in public docstrings;
   - a **one-line comment** at the code that handles it, pointing at the row (e.g. `# Solved problem: single-night alias peaks (orbit_solved_problems.md)`), not an essay.

   Public docstrings say only what a function does: a summary, parameters, returns and one short example, about 40 lines at most (over 60 counts as a defect). Derivations and the reasons behind defaults go in docs pages (`docs/orbit_fitting.md`, `docs/pipeline.md`).

## `OrbitPipeline` design (virgil `src/virgil/pipeline/orbit.py`, tier **provisional**)
It follows the conventions of `BinaryPipeline` and `StarPipeline` exactly:
- the base `_Pipeline` in `pipeline/_core.py`, with `NAME`, `STABILITY = "provisional"`, `STAGES`, `_DEFAULTS`, `_stages()`, `_validate()` and `_summarize()`;
- the `Stage(name, fn, outputs)` records;
- the run folder (`run.json`, `summary.json`, `stages/*/report.json`, `samples.h5`, `plots/`, `quicklook.ipynb`);
- `Result` and `load`;
- `Check(name, status, value, threshold, message)`;
- resume with `ConfigMismatchError`;
- the `virgil-pipeline orbit` CLI;
- NumPy docstrings with the "Stages: a → b" line and settings bullets.

**Inputs.** There are two equivalent ways in:
```python
OrbitPipeline.from_oifits(paths, metadata=None, output="run", **settings)
OrbitPipeline(epochs, output="run", **settings)
```
`from_oifits` reads the files and groups them into nights with `OIData.epochs(gap_days)`. `epochs` may be an `Epochs` object or a `{night: OIData | [paths]}` mapping.

`metadata` is optional, as `{path_or_night: {...}}`. It can give:
- an `epoch` label override;
- `exclude` with a reason: rulings only, recorded in `run.json`;
- `instrument`;
- a time stamp, when the file's MJD is wrong.

`run.json` records every file's sha256, plus the night it was assigned to. This needs one generalisation of the base class's flat input list, kept small in `_core.py`.

**Stages:** `load → epochs → search → modes → posterior → quicklook`.

| Stage | What it does | Reuses |
|---|---|---|
| load | read the files, group them into nights, apply `wavel_range` and `error_floor`, record provenance | `_common._load_report`, `OIData.epochs` |
| epochs | per-night peak catalogue with **automatic flags and reasons**: multiple peaks within Δ, low `gap_marginal`, χ² far above the median night, separation below λ/2B_max, flux at the bound. Seeds are UD-aware. A flagged night seeds nothing, but its data still count | `epochs.epoch_positions` (extended), `decisive` |
| search | alias bands over a **period window computed from the data**; direct-to-data band fits with the label swap, flux flip, V² amplitude and UD primary; tempered evidences | `aliases.fit_orbit_aliases` (extended), `alias_bands`, the GN kernel |
| modes | distinct modes by sky track; a refit per mode; mode weights; the `no-orbit` comparison (orbit against free per-night positions and a single star) | `aliases._distinct`, `epochs.chain_starts` |
| posterior | NUTS per significant mode with `gauss_newton_mass`, weighted by mode | `_common._mcmc`, `fitting.gauss_newton_mass` |
| quicklook | sky track with epochs and error ellipses, band evidence plot, per-night residuals, mode table | `_quicklook` sections |

**Settings and defaults.** Each default exists because a problem was solved (problem number from the catalogue):
- `period_range=None`: computed from the data by `period_window(times, field, night_length)`. The lower end is where a static snapshot within a night stays valid; the upper end is 4× the span. Problem 5.
- `ecc_max=0.9`: eccentric orbits. Like every default below, it equals the engine's (`fit_orbit_aliases`), so the pipeline and the engine never carry different priors; a change to one is a change to both, with a CHANGELOG line. A √e coordinate is only an option if a test system needs it, and then as a named engine argument.
- `flux_range=(0.01, 1.0)` and `flip_flux=True`: problem 9. The flux ratio stays at most 1: the flip keeps the brighter star the primary, so a range above 1 would count the label-swapped mode twice and bias the mode weights and the `labels_free` and `alias_ambiguity` odds.
- `sign_free=True` (the label swap is always explored): problem 8.
- `primary_diam="auto"`: fitted when the data resolve it, lower bound 0. Problem 10.
- `vis_amp=True`: problem 13.
- `gap_days=0.5`.
- `flag_nights=True`: problem 2.
- `exclude={}`: rulings only.
- `temper=True`, with an effective-channel factor estimated from the residuals: problem 14.
- `error_scale="fit"`, then `n_bands_max`, `n_candidates`, `num_chains`, `num_samples` and `seed`.

**Checks**, in order:
- `chi2` (raw, always first)
- `night_flags`
- `labels_free`
- `alias_ambiguity` (the runner-up band's tempered odds)
- `tempering` (f outside [0.5, 2])
- `primary_resolved`
- `no_orbit`
- `period_bound` (short arc)
- `prior_bound`
- `r_hat`, `ess`, `divergences`
- the stage-warning checks.

**Summary keys:** best mode elements with intervals, band table, mode weights, per-night flags with reasons, f, raw χ²/N per night, flags.

## Phase 0: triage in-flight work (don't polish code that will be deleted)
- Freeze notice in vv AGENTS.md: "T1–T5 orbit code may only be deleted, never extended." Add first versions of the solved-problems tables (marking rows "test pending" where needed) and point the project notes at them.
- **#124:** fix only the blocking threads:
  - the T3 control test that can't fail;
  - the 0.01 d tolerance;
  - `b.n == best.n`;
  - nights with no loadable files.

  Then label the 10–14 d window "reference-informed", answer the rest "superseded by OrbitPipeline", and merge. Golden value: Gl 229 N = 34, P 12.134 d.
- **#120:** merge after #124 (resolving towards #124); it ships the CHARA workshop summary.
- **#118, #119:** merge as they are.
- **#125:** reduce it to recomputing verdicts from stored statistics under the current `skyplane` rule.
- **Ben's reruns** of the frozen T2, unmodified, at #124's merged head, in one `oidb_o1_fits` submit:
  - task 4, the A-star companions (a single-epoch recipe, which survives);
  - tasks 2 and 3, the workshop and ι Peg.
- **Leave running:** the Gl 229 NUTS job, the ι Peg diagnosis and the v2 prototype (which decides whether the screen is needed).

## Phase 1: virgil (one worker per PR, in order)

Status: 1a to 1f done (see the top of this note); 1g only if a golden case fails.
| PR | Change | Exit test (virgil CI, seconds, synthetic) |
|---|---|---|
| 1a | Engine result API: `AliasResult.load`, public `scene()`, `residuals()`, per-night χ² (`aliases.py`) | round trip; vv's `best_scene`, `block_chi2` and `residual_stats` become dead code |
| 1b | `epoch_positions`: night flags with reasons, relative thresholds, UD-aware seeds (`primary_diam=`); engine: "flagged nights seed nothing" (`epochs.py`, `aliases.py`) | single-night alias flagged; 1.05 mas primary gives unbiased positions |
| 1c | Engine defaults: `sign_free`, `flip_flux`, `vis_amp`, `primary_diam="auto"`; `labels-free` flag; `period_window` when `p_range=None` | twin, flip, UD and short-arc cases pass with no per-system arguments |
| 1d | Tempering: per-block dof plus a pooled effective-channel f at runtime; remove `dof=1.0` from `_loglike` | white noise gives f within 5% of 1; p < 1 at χ²/N ≈ 12 |
| 1e | **`OrbitPipeline`**:<ul><li>`pipeline/orbit.py`</li><li>registration in `__init__._LAZY`, `__all__`, `cli._PIPELINES` and `_quicklook._SECTIONS`</li><li>the base-class change for multi-night inputs</li><li>`no-orbit` and modes, per-mode NUTS with `gauss_newton_mass`</li></ul> | `tests/test_pipeline_orbit.py`, modelled on `test_pipeline_binary.py`: API snapshot (signatures, `STAGES`, `DEFAULTS`), summary keys and check order, settings validation, folder layout and round trip, resume/fresh, CLI, quicklook; golden synthetic cases: gap alias, twin, UD, null → `no-orbit`, e = 0.8 |
| 1f | Docs:<ul><li>rewrite `docs/orbit_fitting.md` to lead with `OrbitPipeline.from_oifits(paths)` and show the engine underneath, in the same style as `binary_search.md`'s "Everything above can be run in one call" section</li><li>add the orbit row and its stage table to `docs/pipeline.md`</li><li>add `docs/api/pipeline.md` entries</li><li>mark `start_from_positions` superseded</li><li>if the v2 solver internals are worth keeping, put them in virgil `design/orbit_solver.md` under this plan (this note stays the single orbit plan; `history/automatic_orbits.md` and `orbit_aliases.md` are already marked superseded by it)</li></ul> | docs build with `--strict`; doctest-skip examples run in the notebook |
| 1g | *Only if Phase 3 fails a golden case:* the v2 screen (P4) and hierarchical gains (P1), inside the engine | the failing case passes |

Every PR must have short NumPy docstrings (about 40 lines at most) on every function it touches, and follow Principle 5: a named test, one table row, a one-line pointer comment. `OrbitPipeline` stays thin (about 500 lines): it subclasses `_core`, reuses `_common` and shares one NUTS wrapper.

## Phase 2: vv consolidation (one worker per PR)
- **2a, crosscheck:**
  - one `thiele_innes`, in `crosscheck/orbits.py`;
  - delete `skyplane.py`'s copy and the unused `compare_orbits`;
  - add `crosscheck/references.py`, which loads `oidb/references/*.json` and normalises ω using a **required** `omega_of: primary|secondary` field, enforced by a schema test;
  - add `VERDICT_RULE_VERSION`;
  - add a `REFERENCE-DISAGREES` verdict;
  - accept the sign-free reflection only when the run reports `labels-free`.

  This can go in parallel with Phase 1.
- **2b, driver:**
  - `scripts/orbits/systems.py`: frozen dataclasses giving, per system, the OIFITS path list, the `metadata`, and rulings as (night, reason, ruling reference). No P, e, diameter or twin values.
  - `scripts/orbits/fit.py`: one `OrbitPipeline.from_oifits(...).run()` per system, with run folders under `results/orbits/<system>/`. The pipeline's resume and `ConfigMismatchError` give provenance and staleness for free.
  - `scripts/orbits/compare.py`: the **only** module that imports `crosscheck.references`. It reads `virgil.pipeline.load(run)`.
  - Leak tests: an import test; a canary test (perturbed references leave the pipeline's stored config and inputs unchanged); a grep test.
  - Both Gl 229 selections are separate systems.
- **2c, renderer:** `scripts/orbits/report.py` writes the gallery and aliases pages from run folders. It reuses the `orbit_gallery` templates and front end, recomputes verdicts with the current rule, and badges a run STALE when its virgil commit, rule version or config differs. It replaces `build_aliases_page.py`, the data side of `orbit_gallery.py` and the reporting in `oidb_compare`.
- **2d, deletions:**
  - `reanalyse_orbits.py`;
  - the orbit recipes and the `alias_bands` copy in `oidb_fit.py`;
  - `orbit_diagnose.py`;
  - `oidb_compare.orbit_rows`;
  - `trackb_fit`'s own peak picker (routed through `epoch_positions`);
  - `TWINS`, `NEAR_EQUAL` and `PRIMARY_OMEGA` from every fit path.

  Exit: the grep test finds no `published=`, `init=`, `P_ref`, `P_WINDOW` or `TWINS` on any fit path.
- Update vv docs: `docs/orbits/*.md` describe the pipeline runs, and the method page links virgil's orbit pipeline docs.

## Phase 3: one OzSTAR job
`ozstar_scripts/scripts/orbits/` runs `scripts/orbits/fit.py`, one array task per system, with full commits pinned. It retires `orbit_aliases`, the orbit tasks of `oidb_o1_fits`, `orbit_v2_proto` and `orbit_diagnose`.

**Golden table** (the exit criterion):
- Gl 229, both selections: N = 34 found blind, from a data-only window.
- AL Dor: `labels-free`, right period.
- ι Peg: UD fitted, raw χ²/N per night.
- Null: `no-orbit`.
- Track B: on one page, none STALE.

A failure goes to 1g or a virgil fix, never to a per-system knob.

## Phase 4: calibration (vv)
`scripts/orbits/inject.py` generates data with `crosscheck.simulate` (never virgil's forward model), anchored on the blind top mode. It runs null and orbit injections through `OrbitPipeline`, and produces a reliability diagram plus completeness, on OzSTAR within the v2 §13 Q1 budget. When coverage is off, the fix is a virgil PR with a test.

## Failure modes and how each is detected automatically
| Failure | Detection (check, or flag in `summary.json`) |
|---|---|
| Integer orbit count / aliases | bands over the data window; tempered odds → `alias_ambiguity`; leave-one-night-out |
| Single-night alias peak | `night_flags` (seeds nothing, data still count) |
| Twin / Ω–ω mirror | label swap always explored → `labels_free`; sky-plane comparison only |
| Resolved primary | `primary_diam="auto"`, UD seeds → `primary_resolved`; V² residuals at long baselines |
| V² calibration | per-block scales, `vis_amp`; `tempering`, `error_scale` checks |
| Poor fit / no convergence | `chi2`, `poor-fit`, `not-converged`, `laplace-is-differ`, `r_hat` |
| No orbit / noise | `no_orbit`, against free positions and a single star; thresholds from null injections |
| Short arc | window reaches 4× span → `period_bound` read as "P unconstrained" |
| High eccentricity | `ecc_max=0.9`; dense-mass fallback on `divergences` |
| Triple / extra light | f ≫ 1 on specific nights in `night_flags`; optional third-light term |
| Wrong reference (vv) | `REFERENCE-DISAGREES`; `omega_of` schema; reference positions checked against the reference's own orbit |
| Stale result (vv) | renderer badge from commit, rule version and config |

## Delivery
- **Design:** this plan. Each PR goes to one worker session that owns it end to end.
- **Every brief includes:**
  - the solved-problems table and `orbit-solved-toolkit`;
  - Principle 5 (comments, docstrings, tests);
  - virgil's AGENTS.md rules;
  - the laptop rules: targeted tests, two threads, one process;
  - "never run cluster commands".
- **Order:**
  1. Phase 0.
  2. 1a → 1f, with 2a in parallel.
  3. 2b–2d once 1e is merged.
  4. Phase 3, then Phase 4.
- **Review gate:** an adversarial Sonnet review of each phase's diff before merge.

## Verification
- **virgil:** `tests/test_pipeline_orbit.py`, `tests/test_aliases.py` and `tests/test_epochs.py` on CI. Locally run targeted files only, with `OMP_NUM_THREADS=2`. Run `virgil-pipeline orbit <files> -o run --through epochs` on the Gl 229 OIFITS as a laptop smoke test, and the full run on OzSTAR.
- **docs:** `zensical build --strict` in both repos. Check the orbit tutorial, the pipeline page, the gallery and the aliases pages in the browser pane, in light and dark themes at phone width.
- **vv:** the leak tests (import, canary, grep), the reference schema test, `test_skyplane.py`, `test_orbit_gallery.py`.
- **OzSTAR:** the Phase 3 golden table, compared with the papers in the sky plane, leading with raw χ²/N.
- **Done when:**
  - a single `OrbitPipeline.from_oifits(paths).run()` reproduces every golden case with default settings;
  - no published information appears on any fit path;
  - T1–T5 are deleted;
  - every solved-problem row has a passing test.
