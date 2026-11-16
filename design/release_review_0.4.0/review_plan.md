# Plan-fidelity review of virgil main (420d1b7) for 0.4.0

Read-only audit. Checkout: origin/main 420d1b7. Paths below are relative to that checkout. Note that `pyproject.toml:3` and `CITATION.cff:6` are still 0.3.0 and CHANGELOG has a very large `## Unreleased` section (lines 8-667); the version bump and a dated 0.4.0 heading are still to do.

## 0. Headline findings

1. **The orbit-search release criterion is NOT met in the repo.** `automatic_orbits.md` §14.6 decision 8 (trap/degeneracy/release test on Gl 229 Ba-Bb, unattended) has no evidence anywhere: no Gl 229 result in CHANGELOG, docs, tests or benchmarks (the only mentions of Gl 229 are comments in tests and design notes). Of the pipeline's steps, built: PR A (`gap_marginal`, scale-marginal ranking), the peak catalogue (`EpochPeaks`), D1 `score_orbits` (`src/virgil/orbit_search.py`, `__all__` = OrbitScores, SharedFlux, rank_scores, score_orbits). Not built: D2 (generators, suppression/polishing, clustering, mode report), `orbit_modes`, `fit_modes`, `sample_modes`, `automatic_orbit`, grouped-bootstrap calibration (`bootstrap_null` has no `groups=`; `detection.py:601`), the regression suite A15-A26 (`tests/test_orbit_benchmarks.py` does not exist), `OrbitPipeline`. The shipping unattended-ish tool is `aliases.fit_orbit_aliases` (alias bands, per-epoch scales), which the design note itself (§14.1) says still lands in a wrong mode on Gl 229 (i ~ 128 deg, delta loss ~ +1100) with three training nights. `automatic_orbits.md:3` still says "Nothing here is implemented" (stale). If the maintainer's rule "0.4.0 waits for Gl 229 found unattended" stands, 0.4.0 is blocked on D2 + E-G + the OzSTAR `orbit_gl229` job; nothing in the repo shows this is close.
2. **Standing decisions of imaging_plan.md are mostly faithfully implemented**; exceptions are small (section 1 below).
3. **Three promised-for-0.4 deprecation/removal items are unresolved**: `Tabulated` ("removal planned for 0.4.0", CHANGELOG:715; still present with a FutureWarning, `spectra.py:624-671`); the `TruncatedCone` / `ModulatedGaussianRim` deprecation ("at the next minor release", radial_profiles.md §5; none made, §14 says "No deprecations were made"); and the t_ref warning message does not mention the 0.5 requirement.
4. Several plan texts are stale and should be edited (section 3).

## 1. imaging_plan.md (revision 4): standing decisions vs code

| # | Decision | Status | Evidence / exception |
|---|---|---|---|
| 1a | `Precision.HIGHEST` on every FT/matmul | **partial (done for FTs)** | FT paths: `_geometry.py:82,114-116` (DFT/MFT), `models.py:727,1115,4672,5279`, `fields.py:670`, `imaging.py:443-449,1210`, `_elr.py:271-274,441-454`. **Not HIGHEST:** the DISCO operator application `oidata.py:1240` (`vis_mat @ log_cvis.real + phi_mat @ log_cvis.imag`, the AMI path the plan targets), the RadialGP/GP factor matmuls `models.py:3597,3620,3820`, and the Whitening/Woodbury matmuls in `_linear.py`, `observables.py`, `orbits.py`. Only matters for float32 on A100/H100 (TF32); `fit`/`laplace_cov`/`svi` run float64 by default so fits are unaffected, but pipeline NUTS runs in ambient precision (pipeline.md). No test asserts HIGHEST (only `tests/test_polar_disk_height.py:90` uses it as a reference). AGENTS.md:93 narrows the rule to "matmuls over pixels", so the plan wording ("every ... matmul") is stronger than the policy. Fix: either put HIGHEST on `oidata.py:1240` or narrow the plan sentence to AGENTS.md's wording; add one A100 float32 test/benchmark. |
| 1b | `fit` default float64 in local x64; x64 never global | **done** | `fitting.py:653` `dtype="float64"`, `:851` `run_in(dtype)`; `svi.py:303`; `inference.py` `dtype="float64"`; `_precision.py`. No global `jax_enable_x64` in src/tests/examples (grep found only reads). |
| 2 | One likelihood, chord `2 sin(D/2)/sigma`; `model_loglike` on top | **done** | `likelihood.py:193,198,210`; whitened_residuals shared by grids/limits/numpyro/fit. 4+-telescope caveat documented. |
| 3 | `fit` and `numpyro_model` same spec; numpyro rejects MEM/TV/TSV | **done** | `likelihood.py:1009-1016` rejects any regulariser with `probabilistic=False` (base class `imaging.py:99`; only `Centroid` is True, `:431`); test `tests/test_fitting.py:157` (TSV). MEM/TV rely on the same flag but have no dedicated rejection test. Spelling is now `regularizers` (plan says `regularisers`; alias accepted until 0.5). |
| 4 | optimisers `lm`/`lbfgs`/`adam`; LM "never materialises a Jacobian" | **done, wording stale** | `fitting.py:743,857-875`. `_normal_solver` (`fitting.py:1494-1504`) uses dense `lx.QR()` for few coordinates and `lx.Normal(lx.CG)` above, so "never materialises" is wrong for small problems. LSMR not used. |
| 5 | `Image` signature, softmax, `GaussianField` via `evaluate` | **done** | `models.py:2191-2290`. |
| 6a | Separable DFT default; exact MFT on uv lattice | **done** | `_geometry.py`; `models.py:2168-2190,2340-2358` (`uv_grid`, `rotation_deg`). |
| 6b | NUFFT gone | **done** | no `nufft`/`finufft` in src, pyproject, tests; only design/MIGRATION history and `design/figures/bench_ft_*`. |
| 6c | Padded FFT + interpolation never used | done | none found. |
| 7 | Default FOV = min(500 mas, lambda/B_min); beam shaded; residual map diverging, z-scores with uncertainties | **done, FOV statement stale** | `imaging.py:517` `field_of_view`; `plotting.py:333-409,432-480` (`beam=`, `RdBu_r`, z-score label). **Stale:** Unreleased CHANGELOG:24-36,89-98: the default field now comes from the data (`image_field`, six FWHMs, between 4 resolution elements and `field_of_view`), not "smaller of 500 mas and lambda/B_min". |
| 8 | optax+lineax required; blackjax `[sampling]` extra | **contradicted (plan stale)** | `pyproject.toml:25-30` has optax, lineax, optimistix; there is no `sampling` extra (removed in 0.2.0 review, CHANGELOG:1358; no import of blackjax anywhere in src/docs/tests). Plan lines 49, 289, 337, 770 are stale. Also zodiax floor is `>=0.4.1`. |
| 9 | Jeffreys / group-invariant priors by default | **done in code; some documentation conflicts** | LogUniform scales and Uniform locations: `imaging.py:833-834`, `ensemble.py:625,755-761`, `pipeline/binary.py:293-295`, `pipeline/star.py:164-177`, `pipeline/_core.py:266-268`, `orbits.py:1362,1448 (orientation_priors)`, `priors.py` (IsotropicInclination/Latitude), `grid_fit.py:656` (LogUniform recommended). Gaussians only as GP latents (`imaging.py:485`, proposal in whitened coordinates) or stated calibration priors. No-data test rule implemented: `tests/test_no_data_priors.py` (numpyro_model, fit, gauss_newton_mass, noise sites, GP latents, RV terms, flux scale, detection grid weights) plus `tests/test_sampler_flat_coordinates.py`. **Gaps:** (i) no no-data test for `fit_orbit_aliases`/epoch priors, `PolarDisk`/`RadialGP` hyperpriors or `AngleVector` in combination with orbit priors (angles has its own tests); (ii) "interim prior + importance reweighting + PSIS k-hat" has no general helper (PSIS exists only in `svi.py`; `aliases.py` has its own mixture IS); (iii) the VISPHI continuum projection default is the flat-prior limit and the plan's own log (line 530) flags the conflict with "finite lambda only" as "for Ben to settle", still unrecorded; (iv) OI_FLUX/NFLUX scale uses a stated Gaussian (documented as a proposal for 1/k) - consistent with the rule only if reweighted, which nothing does. |
| 9b | `fit` optimises each prior in its flat coordinate; NUTS | **done; NUTS default differs** | `_flat.py`; `fit` flat since #226. NUTS uses numpyro's bijection by default, flat is opt-in (`flat_coordinates=True`, `sampler_flat_coordinates.md` header, SBC showed no gain). The plan's 2026-10-05 log is consistent with this; `sampler_flat_coordinates.md`'s body still says "default"/"now" in many places (the header warns to read them as opt-in). Rewrite the body or move to history. |

### Stage table (imaging_plan.md)

| Stage | Status on main | Evidence |
|---|---|---|
| 0 groundwork, whitened_residuals | done | `likelihood.py`; `design/image_reconstruction.md`; AGENTS.md precision policy (:89-96). |
| 1 `Image`, DFT | done | `models.py:2191`; `tests/test_image_model.py`. |
| 2 NUFFT | **abandoned (by design)** | removed after Stage 3 (#75). Plan keeps its full Stage 2 build list; that text should be condensed to the outcome. |
| 2b MFT | done | `models.py:2168`, `notebooks/mwe/mwe_uv_lattice_mft.ipynb`. |
| 3 `fit`, regularisers, `diagnose` | done (differences already logged) | `fitting.py`, `imaging.py` (MaxEntropy, TSV, TV, Centroid, plus Laplacian, StarletL1, LogSum), `diagnose` `imaging.py:3078`. |
| 4, 4b, 4c | done | `coverage.vlti_oidata/nrm_oidata`, `Rotated` `models.py:5805`, `BlackBody` `spectra.py:154`; MWEs in `notebooks/mwe/`. 4c real-data notebooks live outside the repo (by design). |
| 5, 5a-5e | done | `fields.py`, `imaging.log_evidence` `:2261`, `classic_maxent`, `error_scale`, `laplace_samples`, `fitting.gauss_newton_mass`; `imaging_gp/imaging_sampling.md`. `gauss_newton_diagonal` deliberately not built. **Open debt never closed:** "TSV evidence" (log_evidence needs a GaussianField) and the discrepancy target N - gamma for `l_curve` (`imaging.py:1906` still target=1.0). |
| 6.0 reader/likelihood fixes | done | `OIData.cp_noise`, `read_oifits(insname=)`, `PHITYP` (`oifits.py:58-104`). TODO(Stage 6a.0) for closure-phase legs from OI_T3 still open (`oifits.py:44`). |
| 6a.0 times/frames | done | `OIData.dt`, `frame`, `split_by_epoch` `oidata.py:504`. |
| 6a.1 orbits | done | `KeplerOrbit/ThieleInnesOrbit/StateVectorOrbit/PositionData/RVData/AxialVonMises/orientation_priors/position_angle_prior` (`orbits.py`), `Attached`, `OrbitalBinary`, `TruncatedCone`, `simulate`, `bias_test`, `with_north_angle`. "Later" items (physical skew, TI marginalisation) not started. AMICAL writer PA round trip untested (plan says so). |
| 6a spectra/observables | **mostly done** | `GaussianLine/LorentzianLine/Nodes/Sum`, extras readers, VISPHI continuum, `with_error_floor`, `with_flux_scale`, `with_bandwidth` (new, opt-in). **Not built:** spectral-resolution kernel, primary beam (fibre coupling), differential VISAMP, VISAMP debiasing, fitted `noise=` terms for extras. |
| 6d nuisances | done | `gains.py`, `with_gains`, `with_closure_offsets`, `with_wavelength_scale`; `notebooks/mwe/mwe_gains.ipynb`. |
| 6e closure amplitudes | **not started** (not estimated) | no `camp`/`logcamp` anywhere. |
| 6b joint multi-filter AMI, 6c `ImageCube` | **not started** | no `ImageCube`; only the per-dataset-model-function machinery (`fit` takes a function returning one model per dataset). Milestone 2 not reached. |
| 6.5 0.2.0 release | done (history) | tag v0.3.0 exists. |
| 7 hardening | **mostly not started** | MEM preconditioning / L-curve success test, multi-start, K-fold CV, MFT caching, "choosing a regulariser and prior" guide: none present. API page exists (`docs/api/`), version bump pending. |
| 8 PMOIRED parity | **partial** | `RadialProfile` realized as `PolarDisk` (see radial_profiles). `Projected` and `bootstrap_fit` not built (grep: none). |
| Binary-detection parity | done | `injection_limits`, `linear_flux_grid`, `detection.py`. |

## 2. Other design notes

### design/pipeline.md (header: BinaryPipeline and StarPipeline built; Orbit/Imaging provisional follow; DiskPipeline design only)

| Item | Status | Evidence |
|---|---|---|
| `_Pipeline` contract, run folder, resume, checks, `load`, CLI | done | `src/virgil/pipeline/_core.py`, `cli.py`; `tests/test_pipeline_*.py`. |
| `BinaryPipeline`, `StarPipeline` stable | done | `binary.py:165`, `star.py:326`. |
| `OrbitPipeline` (provisional) | **not started** | blocked on orbit search; note `fit_orbit_aliases` checkpoints as pickles (`band_<N>.pkl`, CHANGELOG:296-298) which conflicts with the contract "no pickles". |
| `ImagingPipeline` | not started | |
| `DiskPipeline` | not started; **its gating condition is met**: note says it waits for virgil#356, which merged (8909ec1). Plan text stale; Ben's open question (default `model=`) unresolved. |
| "Priors" paragraph | consistent with code | |
| Open: JAX-free `info`/`validate`, `global_nsigma` calibration | open; `global_nsigma` is **in the frozen stable summary keys and a stable check** while the note says to settle it "before the first stable release" - the pipelines are already tagged stable. Decide for 0.4.0: calibrate or drop/rename (breaking the schema later costs more). |

### design/radial_profiles.md (stages P1-P6, V)

| Stage | Status | Evidence |
|---|---|---|
| P1 `PolarDisk` engine + profiles | done | `models.py:2762 ThinRing, 2806 GaussianRing, 2864 RadialPowerLaw, 2943 ExpTail, 3033 Gap, 3114 RadialSum, 4737 PolarDisk`. Note: the note names `PowerLaw` and `Sum`; code uses `RadialPowerLaw`/`RadialSum` (names collide with spectra `PowerLaw`/`Sum`). |
| P2 height, Cone, subsume TruncatedCone | done | `Cone 4385`, `PowerLawHeight 4419`, `TruncatedCone 757` on engine (CHANGELOG:240-248,503). |
| P3 positive Harmonics, VonMises, ModulatedGaussianRim, GaussianArc | done | `Harmonics 3952`, `VonMises 4254`, `virgil/harmonics.py`; GaussianArc FutureWarning (`models.py:549,623`, pyproject filterwarnings). |
| P4 RadialGP, PolarKernel, spectral_index, freeze_geometry | done | `models.py:3362,3659,5121`; `log_evidence` support. |
| P5 tutorial | done | `docs/polar_disk_rings.md`, `notebooks/polar_disk_rings.ipynb` (#347/#356/#367). |
| P6 PolarField, FlaredDisk comparison | done | `fields.py:342`; `tests/test_flared_vs_polar_disk.py`. |
| V (virgil-validation) | outside repo; not verifiable here | |
| §5 deprecation rule | **open for 0.4.0** | "At the next minor release each other subsumed class gets a deprecation note and a changelog migration line, removed one release later." `TruncatedCone` and `ModulatedGaussianRim` are subsumed and reproducible; none deprecated. Decide: deprecate in 0.4.0 or amend the rule. |
| Status line | stale | still "scoping note (2026-10-14)"; §13-14 (execution rules for an autonomous run) are history and should be archived. §9 "Open decisions" still lists a decision (profile parametrization) that was settled. |

### design/visibility_orbits.md (header: "plan for 0.4. Stages 1 and 2 are implemented")

| Stage / req | Status | Evidence |
|---|---|---|
| 1 Epochs, OrbitalBinary, t_ref warning (V1,V2,V3,V5,V8,V10) | done | `epochs.py:94`, `models.py:6039`, `orbits.py:135-155` warns when t_ref=0 and |t| > 15000 d (test `tests/test_epochs.py:247-251`). Message does not say "will be required in 0.5"; nothing makes it required yet (`KeplerOrbit(..., t_ref=0.0)` `orbits.py:258`, `at(mjd, t_ref=0.0)` `models.py:293`). `start_from_positions` / `EpochPositions.positions` already require t_ref. |
| 2 rank_orbits, start_from_positions, chain_starts (V13-V16) | done | `epochs.py:889,1008,2358`; scales default staged with FutureWarning (`epochs.py:703-713`). |
| 3 curvature (V18, V19, two-stage Laplace->NUTS->PSIS) | **partial** | `gauss_newton_mass(noise=)` done (CHANGELOG:37-55); `laplace_cov` exists (`inference.py:243`) but no `FitResult.covariance()` and the two-stage workflow is not built. |
| 4 forecasts (V20) | **not started** | no `forecast`. |
| 5 conventions (V6, V7, V9) | **partial** | docs/conventions.md "Orbits" section covers conventions; no conversion table (orbitize!, orvara, Gaia NSS, 6th catalog) or per-row regression tests; no PA unwrapping helper; White et al. numbers still "to be checked". |
| 6 calibration and shape (V4, V11, V12, V17) | **partial** | `with_north_angle`, `wavel_scale`, `with_bandwidth` done; plate-scale prior helper (lambda_eff), `epoch_offsets`, spectral tied flux scales, evidence over prior configurations: not built. |
| Decisions (§7) | all consistent with code, except item 6 (two-stage workflow) unbuilt. | |

### design/automatic_orbits.md (1500 lines; header "Nothing here is implemented", §14 "wins over earlier sections")

| Piece | Status | Evidence |
|---|---|---|
| PR A (`gap_marginal`, `chi2_raw`, scale-marginal ranking, f <= 1, `scales=` FutureWarning) | done | `epochs.py:1210,703`. |
| Peak catalogue (`EpochPeaks`), flags/UD seeds, period window | done | `epochs.py:1049,1096,1619`; `aliases.py:212`. |
| PR B (`Epochs.noise` fixed values, `refine_model=`, `OrbitStart.save/load`) | not verified done; CHANGELOG mentions serialization only for alias results | check before release. |
| PR B0 benchmark harness | done | `benchmarks/orbits/`, `tests/test_benchmark_harness.py`; cases A1, A2, A5, A6, A7, A9, A10 only (suite wants A15-A26). |
| D1 `score_orbits` | done | `orbit_search.py:918`. |
| PR H suite (A15-A26, sealed manifests) | not built | no `tests/test_orbit_benchmarks.py`. |
| D2 generators/modes; E `orbit_modes`/`fit_modes`; F `sample_modes`; G `automatic_orbit`, tutorial section, `start_from_positions` deprecation | **not built** | none of the names exist in src. |
| Mandatory grouped-bootstrap calibration (decision 2) | not built | `bootstrap_null(groups=)` missing. |
| Release criterion (decision 8) | **not met** | see headline. |
| Divergences / stale | | (a) header and §12 describe PR C/D as "TI-EM over maps"; §14 superseded them but §0, §3.2, §4, §10-12 still read as the plan (the superseded table in §14.6 is a 24-row patch list; the note is unreadable as a plan). (b) §11 says `start_from_positions` is "deprecated with a FutureWarning when automatic_orbit lands": not yet. (c) `fit_orbit_aliases` (orbit_aliases.md, "implemented") is a second, parallel unattended method with its own priors, scale marginal and evidence; the note never says whether `automatic_orbit` replaces it or wraps it. (d) `thiele_innes_marginalisation.md` (TI-EM, "approved ... nothing implemented") is superseded by §14 ("TI-EM demoted to a diagnostic"). |

### design/orbit_aliases.md
Status "implemented" is correct (`aliases.py`, `fit_orbit_aliases`), and the file is compact. Its "Extensions" now overlap with CHANGELOG's later changes (auto `p_range`, `sign_free=True` default, `primary_diam="auto"`, `vis_amp=True`, night flags): update the Method/Extensions text to reflect that the defaults changed in Unreleased (CHANGELOG:59-79). Its Gl 229 "worked example" is a result from a prototype, not a regression test in the repo.

### design/sampler_flat_coordinates.md
Implemented as opt-in (header correct). Body text uses "default"/"now"/"After" for the flat coordinates; should be edited to say opt-in. The SVI section duplicates `docs/variational_inference.md`; mark as implemented and shorten.

### design/detection_roc.md
Stages 1-3 built; header says tutorial "outputs await a cluster run" - **stale**: `notebooks/detection_roc.ipynb` has outputs in all 9 code cells and `docs/detection_roc.md` has 3 figures. Stage 4 (cluster campaign + virgil-validation independent FAP check) not in repo; Stage 5 later. virgil#2 closure should be confirmed.

### design/sparse_imaging.md
S1, S2 done and more (multi-scale CLEAN, final major cycle, `StarletL1`, `LogSum`, `Laplacian`, `clean`, `CleanResult`); S3/S4 deliberately deferred; "paused 2026-10-05" status is now misleading (work continued, "Still open": multi-frequency CLEAN). Fine to keep; update the status line.

### design/api_argument_order.md
**Done and verified**: no public function in `src/virgil` takes data before model (AST scan of public defs for data-before-model: zero hits); `samples_dict` survives only inside `_deprecate.py`; `old_order(removed=True)` path exists for 0.5; pytest filterwarnings turn the shim warnings into errors in the test suite (`pyproject.toml:161-163`). Remaining: the 0.5 removal. The survey tables (line numbers at c064b82) are history and could be archived once 0.5 ships. Note `inflated_errors(data, prediction, ...)` stays data-first (declared out of scope, arrays).

### Other notes that act as plans
- `gauss_newton_plan.md` ("not built", 2026-10-14): GN1 kernel is built (`_gauss_newton.py`, used by `aliases.py:47` and `epochs.py:53`), GN2 partly (`gauss_newton_mass(noise=)`); GN3 (`fit` on the kernel), GN4 (aliases: partly), GN5 (pipeline `mass="gauss_newton"`: pipelines use `dense_mass`, not the kernel), GN6 (residual MEM) not built. Status line is wrong.
- `image_reconstruction.md` header says "planned. Stage 0 ... under way": stale (it also carries the Stage log).
- `orbit_scene_joint_fitting.md`, `orbit_prior_art.md`: decisions implemented in `orbits.py`; fine as reference.

## 3. Statements in the plans that are stale (edit the plan, not the code)

1. imaging_plan.md:49, 289, 337, 770: `[sampling]` extra / blackjax (extra removed).
2. imaging_plan.md:44 and standing decision 7: default FOV is now data-driven (`image_field`).
3. imaging_plan.md:32: LM "never materialises a Jacobian" (QR for few coordinates).
4. imaging_plan.md:72-82 and the whole "Implementation orchestration" section (lines 792-855): `imaging` branch in the old `drpangloss` folder, Fable/Opus allocation, stacked-PR mechanics; entirely history. `Branching and workflow` and `Stage 6.5` are history too.
5. imaging_plan.md:765: notebook names (`imaging_ami`, `imaging_long_baseline`, `imaging_gp_sampling`, `imaging_chromatic`) do not match the real ones (`imaging_ami`, `imaging_rml`, `imaging_gp`, `imaging_composite`, `imaging_sampling`, `imaging_clean`, `imaging_ensemble`); `imaging_chromatic` never existed.
6. imaging_plan.md Stage 2: the build and test lists describe code that was removed; replace by the outcome.
7. imaging_plan.md Stage 6 and 6a: "Left for PR C" list still correct; `Tabulated` "becomes Nodes in 6a" done (deprecated, not yet removed).
8. "Order of work after Stage 5" (lines 392-413) is a to-do list that is finished except 6b/6c; replace by a status table (the 0.2.0 codebase review already suggested this, line 695).
9. Standing decision 9 table: eccentricity "uniform e as interim" fine, but orbit-search notes (`aliases`) use flat coordinates in (log P, T0/P, e, cos i, omega, Omega, log a, log flux) - consistent; just say so once.
10. automatic_orbits.md:3 "Nothing here is implemented"; visibility_orbits.md header; pipeline.md DiskPipeline gating; detection_roc.md outputs pending; gauss_newton_plan.md "not built"; image_reconstruction.md "planned".
11. Naming: plans spell `regularisers`, code and docs now `regularizers` (OUP -ize); `design/*.md` should follow (low priority).

## 4. Promised-for-0.4 items that are missing

Block 0.4.0 (if the stated criteria are applied literally):
- **Unattended Gl 229 orbit search** (D2, E, F, G, calibration, regression suite, OzSTAR `orbit_gl229` run). This is the maintainer's own gate. If it is going to take long, relax the gate explicitly (see ordering below) rather than leaving it implied.
- **Decision on `Tabulated`** (remove now, as the 0.3.0 changelog promised, or re-promise removal in 0.5 and say so in the warning text). Small.
- **Decision on `TruncatedCone`/`ModulatedGaussianRim` deprecation** per radial_profiles §5 (deprecate in 0.4.0 or change the rule). Small.
- **t_ref warning text**: add "will be required in 0.5" (and a one-line CHANGELOG migration entry). Trivial. Also confirm CHANGELOG's Unreleased "Migration from 0.3.0" lists it (currently only argument order).
- **`global_nsigma` in the frozen stable pipeline schema** (pipeline.md "before the first stable release"): decide now.
- Release chores: version bump (`pyproject.toml`, `CITATION.cff`, docs), dated heading, re-execute stamped notebooks (`scripts/check_notebook_outputs.py`), changelog "Migration" completeness.

Can move to 0.4.x (or later):
- visibility_orbits stages 3-6 (FitResult.covariance, two-stage Laplace/NUTS/PSIS, `forecast`, conversions table, PA unwrapping, plate-scale helper, `epoch_offsets`).
- `OrbitPipeline`, `DiskPipeline`, `ImagingPipeline` (all provisional by design).
- Stage 6b/6c (joint multi-filter AMI, `ImageCube`), 6e closure amplitudes, primary beam, resolution kernel, differential VISAMP.
- Stage 7 (MEM preconditioning, multistart, K-fold CV, MFT caching, regulariser guide), Stage 8 (`Projected`, `bootstrap_fit`).
- GN3-GN6, TSV-evidence and N - gamma discrepancy (propose: record "won't do" for TSV evidence).
- Interim-prior reweighting helper with PSIS (decision 9); finite-prior VISPHI default decision.
- HIGHEST on `oidata.py:1240` (small code change; arguably should go in 0.4.0 since it is a one-line correctness fix for float32 GPU AMI users).

## 5. Coherence of the plans

Duplications and contradictions:
- The Jeffreys-prior rule is restated in imaging_plan §9, AGENTS.md, pipeline.md "Priors", visibility_orbits "Priors", automatic_orbits, orbit_aliases, thiele_innes_marginalisation, sampler_flat_coordinates. They agree, but the canonical text should be one place (conventions doc or AGENTS.md), with the others linking.
- Two "unattended orbit" plans coexist: orbit_aliases.md/`fit_orbit_aliases` (shipped) and automatic_orbits.md (§14, unbuilt), plus thiele_innes_marginalisation.md (superseded) and visibility_orbits stage 2 (`start_from_positions`, slated for deprecation). The relationship between `fit_orbit_aliases` and `automatic_orbit` is undefined.
- Two LM paths (optimistix and `_gauss_newton` kernel) are accepted in gauss_newton_plan D3; imaging_plan Stage 7 still lists "a least-squares (LM) form of the entropy term" as a Stage 7 option while gauss_newton_plan says it supersedes it (GN6).
- Release-criterion text exists only in automatic_orbits.md §14.6 decision 8 and the maintainer's notes; it should be copied into a short release checklist so the gate is visible.
- The plans' "Open decisions" never get closed: radial_profiles §9, pipeline.md open questions, VISPHI finite-prior default, TSV evidence.

Notes that are now history and could go to `design/archive/` (keep rationale, stop being read as plans): imaging_plan.md orchestration/branching/Stage 2/6.5 sections (or the whole note after extracting a status table), image_reconstruction.md Stage log, codebase_review_2026-10.md and -10b.md, orbit_fitting_review.md, radial_profiles.md §13-14, api_argument_order.md survey tables (after 0.5), pmoired_parity.md once Stage 8 is decided, flared_disk_vs_polar_disk.md, sampler_flat_coordinates.md SVI section, examples_backlog.md, orbit_prior_art.md (decided), thiele_innes_marginalisation.md (superseded by automatic_orbits §14, or explicitly kept as the 0.5+ item).

## 6. Suggested revised ordering for 0.4.x

0. **Pre-0.4.0 (days, not weeks)**: close the three small decisions (Tabulated, TruncatedCone/Rim deprecation, `global_nsigma`), add the 0.5 sentence to the t_ref warning, HIGHEST on the DISCO operator (or narrow the plan), fix the stale design headers, bump versions, re-stamp notebooks, write the CHANGELOG Migration section for t_ref/deprecations.
1. **Decide the orbit gate.** Either (a) keep it: finish D2 -> E -> F -> G and the A15-A26 suite, run `orbit_gl229` on OzSTAR and commit the result summary (modes, weights, raw chi2/N, the trap test) to docs or CHANGELOG as the evidence; or (b) ship 0.4.0 with `fit_orbit_aliases` + `score_orbits` documented as "assisted, not unattended" and move automatic_orbit to 0.5 (recommended if D2 is more than a few weeks away, since 0.4.0 already carries a very large unreleased change set and argument-order migration).
2. **0.4.x**: `OrbitPipeline` on whatever orbit stack ships; `DiskPipeline` (gate met); visibility_orbits stage 3 (FitResult.covariance) then stage 5 (conversions table with regression tests; cheap, high user value), stage 4 forecasts.
3. **Imaging**: Stage 7 hardening (MEM preconditioning/GN6, multistart, K-fold CV, regulariser guide), then 6b joint multi-filter AMI, then 6c/6e only if a dataset demands them.
4. **Later**: Projected / bootstrap_fit (Stage 8), primary beam and resolution kernel, interim-prior reweighting helper, TI marginalisation.
5. **0.5**: remove argument-order shims and `samples_dict`, make `t_ref` required, remove deprecated classes, then archive the argument-order survey.

## 7. Things I could not verify (read-only, no heavy runs)
- That the test suite passes on main; whether PR B (`refine_model=`, `OrbitStart.save/load`) is complete.
- Result of any OzSTAR job (the Gl 229 `orbit_gl229` run lives in virgil-validation and the cluster scripts, outside this checkout).
- Whether the numbers quoted in the plan logs still reproduce (the plan itself notes drift in the composite tutorial).
