# Adversarial pre-0.3.0 review, October 2026 (main at `8562a39`)

This review covers everything merged between `v0.2.0` (released on PyPI on
2026-10-03) and `8562a39`, about 330 commits. That work added:

- orbits;
- spectro-interferometric observables;
- the Stage 6d calibration nuisances;
- `virgil._linear`;
- the detection tools;
- Jeffreys priors, with `fit` running in flat coordinates;
- `AngleVector`.

The review looks for simplicity, maintainability, legibility and performance, in the code, the docs and the examples. It follows up the
earlier review ([`codebase_review_2026-10.md`](codebase_review_2026-10.md)):
its fixed findings are not repeated, but they were checked for regressions
(none found; see [Performance](#performance)).

**Method.** The core modules were read by hand: `limits`, `_linear`, `fitting`, `detection`, `grid_fit`, `observables`, `spectra` and `priors`. Four parallel sweeps (docs, notebooks, API consistency, performance) went over the rest. Every finding from the sweeps that is used here was re-checked against the code. Bugs are shown with minimal reproductions run locally in float32 (jax 0.11.2). No heavy JAX was run on the laptop. The profiling jobs are prepared, not run.

## Executive summary

- **Not ready to tag as it stands. Four blockers, about a day of work plus one OzSTAR notebook run.**
  - Two are real bugs in `injection_limits`, both new since 0.2.0:
    - `flux_bounds=None` silently returns the top of the search bracket (flux 1000) at every position (B1).
    - It crashes on data with extra observables (B2).
  - The other two are release hygiene:
    - 22 notebooks, 11 of them rendered into the docs, show results from priors that their own source no longer uses (B3).
    - `CHANGELOG.md` is wrong about what was released and misses three changes, one of them a behaviour change to χ² for OIFITS v1 files (B4).
- **The new numerical code is sound.**
  - `_linear`'s two whitenings agree with dense algebra in float32 and float64.
  - The grid, limit and fit tools do not recompile when values change. The perf sweep counted XLA compiles: 0 on new prior bounds, σ, `flux_bounds`, grid values or template leaves.
  - Neither earlier performance finding has regressed.
- **The main cost to maintainability is inconsistency, not size.**
  - Priors are passed in five different ways.
  - There are two unrelated classes called `LogUniform`.
  - Grid tools take `(data, model)` while `fit` takes `(model, priors, data)`.
  - The two bisection loops in `limits.py` disagree about what `flux_bounds` means.

  All of these are cheap to fix before 0.3.0 freezes the new names. Once released, each one needs a deprecation.
- **Performance: one measured hot spot, one memory hazard.**
  - `DifferentialPhase.whiten` refactors a dense per-frame covariance on every call. It grows from 0.5 to 91 ms per gradient between 32 and 512 channels, 10× slower than its own `prior_width` path.
  - `injection_recovery(draw_batch>1)` multiplies the grid's memory budget by `draw_batch`.

  Three OzSTAR profiles are prepared in `~/code/ozstar_scripts/scripts/virgil_prof03/`.

## Findings by severity

Effort is for one agent, including tests. "Blocks" means it must be fixed before tagging 0.3.0.

### Release blockers

**B1. `injection_limits(flux_bounds=None)` returns 1000 everywhere.** Blocks; 1–2 h.
- **Where:** `src/virgil/limits.py:512-516` (`_UNBOUNDED_BRACKET = (1e-8, 1e3)`) and `:695-711` (`solve`).
- **Evidence.** One position, NIRISS-like data, 3σ:

  | `flux_bounds` | `absil_limits` | `injection_limits` |
  |---|---|---|
  | `(1e-6, 1)`, `(1e-6, 10)`, `(1e-6, 100)` | 0.00613 | 0.00590 |
  | `None` | 0.00613 | **1000** |

  The same happens for a `System` template (`comp.flux`), which is the case the docstring recommends `None` for.
- **Cause.** For a normalised scene, the significance is not monotonic in flux. At f = 1000 the "companion" is the primary, the scene looks like an offset point source, and the significance falls below σ. So `above = excess(log_high) < 0` holds, and the function returns `log_high`. The `RuntimeWarning` is only issued when `flux_bounds is not None`, so the wrong answer is silent.
- **Fix.** Find the first crossing from below, as `_absil_limits` already does: step up by decades from `low`, then bisect. Then share one helper (S5). Add a test that `None` and `(1e-6, 100)` agree.
- **Repro** (float32, a few seconds):

  ```python
  D = nrm_oidata().with_model(BinaryModelCartesian(0., 0., 0.), key=jax.random.PRNGKey(0))
  P = {"dra": jnp.array([60.0]), "ddec": jnp.array([-40.0]), "flux": jnp.array([1e-3])}
  injection_limits(D, BinaryModelCartesian, P, 3.0)                    # [[0.0059]]
  injection_limits(D, BinaryModelCartesian, P, 3.0, flux_bounds=None)  # [[1000.]]
  ```

**B2. `injection_limits` crashes on data with extra observables.** Blocks; 1 h.
- **Where:** `limits.py:672-678`.
- **The bug.** The code adds `signal[n_vis:]` to `data_obj.phi`. The model vector also carries the `extras` blocks (T3AMP, OI_FLUX, VISPHI), so the shapes differ:

  ```text
  OIData(read_oifits(..., extras=("t3amp",)))  -> n_vis 72, n_phi 48, extras [TripleAmplitude]
  injection_limits(data, System(...), grid, 3.0)
  TypeError: add got incompatible shapes for broadcasting: (48,), (96,).
  ```

  `absil_limits` runs on the same data.
- **Fix.** Inject into every block: add the signal to `flatten_data()` and rebuild, or give the extras a `with_values`. Failing that, refuse with the same clear message that `bootstrap_null` gives (`detection.py:603-615`).
- **Tests.** Add a test with extras to `tests/test_grid_and_limits.py`. Gains and closure offsets have the same exposure.

**B3. Notebooks with stale outputs (#227), and a flagship tutorial that contradicts the library.** Blocks; about 2 h of agent time plus one OzSTAR `virgil_notebooks` run.
- **What #227 did.** Commit `926fb0f` (#227) changed priors in 22 notebooks and did not execute them; its message says so. The sync script then regenerated their docs pages from the old outputs. `tests/test_tutorial_docs_sync.py` passes, but that only proves the pages match stale notebooks.
- **Docs pages with old-prior results beside new-prior code:**
  - `composition`, `gravity_darkened_star`, `harmonix`, `limb_darkening`;
  - `imaging_clean`, `imaging_composite`, `imaging_gp`, `imaging_rml`, `imaging_sampling`;
  - possibly `binary_search`, `contrast_limits` and `hierarchical_inference`, whose sources changed after their outputs.
- **MWEs that also need re-running:**
  - `mwe_brg_disk`, `mwe_clean`, `mwe_elr_chara`, `mwe_evidence`, `mwe_fit`, `mwe_gains`, `mwe_gaussian_field`, `mwe_orbit`, `mwe_rotating_epochs`, `mwe_sam_v2_cp`, `mwe_sparse_imaging`, `mwe_stress_test`, `mwe_vlti`.
  - `mwe_gains` is a special case. Its logged OzSTAR rerun (#222) still had `Uniform` priors on `comp.flux`, `disk.flux` and `disk.sigma`, which #227 changed afterwards. The numbers in `design/imaging_plan.md` §6d therefore describe neither the old source nor the new one.
- **`orbit_fitting` contradicts the library.** It is the 0.3.0 headline tutorial (`docs/orbit_fitting.md:298`, notebook markdown cell 15).
  - It says "virgil has no helper for an isotropic inclination yet" and "virgil will provide this prior as `AngleVector` (virgil#211)". Both now exist.
  - It hand-rolls `Ring`, `cos_inc` and `phase_vec`.
  - Rewrite it with `IsotropicInclination` and `AngleVector` / `orientation_priors`, then re-run it.
- **Saved outputs leak local paths and warnings.**
  - `/Users/benpope/code/virgil/src/...` in `mwe_l_curve`, `mwe_recovery_sweep` and `mwe_rotating_epochs`.
  - `/var/folders/...` in `contrast_limits`.
  - Saved `fit(method='lbfgs') did not converge` warnings in nine MWEs.

  Re-running fixes the paths. The prose should say which non-convergences are expected, which is Stage 7's MEM item.
- **Leave alone:** `detection_roc` and `mwe_uv_lattice_mft` are current.

**B4. `CHANGELOG.md` is wrong about what was released, and incomplete.** Blocks; 1 h.
- **The 0.2.0 heading is wrong.** `CHANGELOG.md:325` reads `## 0.2.0 (not yet released)`, but 0.2.0 is on PyPI and GitHub (2026-10-03). `drpangloss` 0.2.0 is out too, while `README.md:92` still says it "will depend on" virgil-astro.
- **Unreleased has two `### Added` sections** (lines 10 and 210), split by Changed, Fixed and Docs. Merge them.
- **Missing entries:**
  - **#203, frames matched on `MJD` *and* `TIME`.** This is a behaviour change. For OIFITS v1 files that keep the snapshot in `TIME` (OYSTER, the 2004 contest), independent closure phases go from 10/130 to 130/130, and every χ² changes. Users must be told.
  - **#212, `OIData.with_north_angle` and the plate-scale nuisances.** `grep -c with_north_angle CHANGELOG.md` gives 0.
  - **#214, evidence from singular values, and the CLEAN refit surviving an NNLS failure.**
- **Mislabelled as breaking.** Several items are called "breaking" or "deprecated" but were never released: `with_flux_scale(scale=)` replacing `width=`, the `LinearFluxGrid` return type, and the bare `(mean, sd)` tuple for `linear_flux_grid(prior=)`. `git grep` at `v0.2.0` finds none of `with_flux_scale`, `linear_flux_grid`, `injection_limits` or `orbits.py`. Say "new in 0.3.0", and delete the tuple deprecation path rather than ship it (S1).
- **What the migration notes need, and only this:**
  - **The flat-coordinate MAP.** `LogUniform` priors on parameters and on `noise=` terms (which existed in 0.2.0) now give the MAP in log x, and LM is chosen automatically.
  - **`Tabulated` is deprecated** (it was in `virgil.spectra.__all__` in 0.2.0), with a removal version, e.g. "removed in 0.5.0".
  - **#203 and `starting_image`'s new priors.**
- **Version.** Bump `pyproject.toml:3` to 0.3.0. It is the only version string, and `__version__` comes from metadata.

### Should fix before 0.3.0

**S1. One prior convention, and no second `LogUniform`.** About 2 h.
- **Two different `LogUniform`s.** `virgil.grid_fit.LogUniform(f_min, f_max)` and `Gaussian(mean, sd)` (`grid_fit.py:321-346`) are `NamedTuple`s. Everywhere else, `LogUniform` means `numpyro.distributions.LogUniform(low, high)`: `fit`, `noise=`, the conventions page, and the error text in `fitting.py:131`. Passing the numpyro one to `linear_flux_grid(prior=)` raises `TypeError` (`grid_fit.py:424`).
- **The other prior forms:**
  - bare tuples: `with_flux_scale(scale=(mean, sd))` and `RVData.term(marginalise_offsets=(mean, sd) | True)`;
  - a width float or tuple: `with_continuum(prior_width=)`;
  - width floats: `with_gains(telescope=...)`;
  - numpyro distributions: `priors=` and `noise=`.
- **Fix:**
  - Accept `dist.LogUniform` and `dist.Normal` in `_as_prior`, and drop the NamedTuples, or rename them if they must stay.
  - Delete the never-released tuple deprecation path (`grid_fit.py:415-421`).
  - Make `marginalise_offsets=True` an error. Its silent N(0, 1000²) default (`orbits.py:~1001`) contradicts the module's own "state the prior" rule, and orbits are new, so nothing breaks.
  - Document the remaining `(mean, sd)` shorthand once, in `docs/conventions.md`.

**S2. `absil_limits` above the float32 significance ceiling.** About 30 min.
- **What happens.** `nsigma` saturates at 12.95σ in float32 (37σ in float64). With `sigma=14`, `absil_limits` returns 9.999e36 (with `flux_bounds=None`) or silently clips to 1. The only explanation is "the optimizer did not converge". There has been no optimizer since the bisection rewrite, and `warn_unconverged`'s text is stale.
- **Fix.** Raise up front in the wrapper when `sigma >= nsigma(huge)` for the data's dtype, saying "use float64 (jax.enable_x64) or a lower sigma". Reword the warning.

**S3. `DifferentialPhase.whiten` factors a dense covariance on every call.** 2–3 h.
- **Where:** `observables.py:921-937`.
- **Cost.** The default (projection) path builds the per-frame (K·R)² covariance and its Cholesky inside every likelihood and gradient. Laptop CPU, 4 frames, `grad(model_loglike)`:

  | Line channels | Projection | `prior_width` |
  |---|---|---|
  | 32 | 0.5 ms | 1.5 ms |
  | 128 | 7.8 ms | 3.0 ms |
  | 512 | 91 ms | 9.8 ms |

- **Fix.** The covariance depends only on the errors, which are fixed (extras take no `noise=` terms). Factor it once in `build` and `with_errors`, and store the factor.
- **Before the fix,** profile P2 confirms the size of the win on GRAVITY-like shapes.

**S4. `injection_recovery` memory and padding.** About 1 h.
- **Memory.** `_simulated_statistics` (`detection.py:947-980`) runs `lax.map(one, batch_size=draw_batch)`, which vmaps `draw_batch` full grid searches. Each one uses the default grid `batch_size` (`batch_size_or_default`, about 2²⁰ visibilities on CPU and 2²³ on GPU, `detection.py:878`). The working set is therefore draw_batch × the documented bound, which contradicts the docstring's "memory stays bounded" (`detection.py:762`). Fix: divide the default by `draw_batch`.
- **Padding.** Every chunk is padded to `chunk` draws (`detection.py:887-890`), so `n_null=65` computes 128. Fix: choose `chunk` to divide the work, or compile one short final chunk.
- Profile P1 measures both.

**S5. `limits.py` should have one bisection, one `flux_bounds` meaning, and no throw-away grid.** About 2 h.
- **Two loops.** `_absil_limits` (`:459-496`, decade bracketing plus 14 bisections) and `_injection_limits` (`:695-711`, a fixed 40-step bracket) are two implementations of "first crossing in log flux". Extract one helper into `_grid` (it also fixes B1).
- **Two meanings of `flux_bounds`.**
  - In `absil_limits`, it is clipping applied *after* an unbounded search.
  - In `injection_limits`, it is the *search range*.
  - The docstrings say different things about the same keyword. Pick one, the search range plus a clipping warning.
- **A grid computed only to be discarded.** `_absil_limits` still evaluates the full positions × fluxes loss grid (`:440-446`), only to pick a starting flux. Since bracketing makes the result independent of the start (the CHANGELOG says so), one start per position suffices, and the work drops by the length of the flux axis. Profile P3 measures it.
- **A stale comment.** `_BISECTION_STEPS = 40` claims a "6 decade" default bracket (`:512-513`), while `_UNBOUNDED_BRACKET` spans 11. About 28 steps reach float32 resolution.

**S6. Docs.** About 2 h, excluding notebook reruns. Generated pages are fixed in their notebooks.
- **README and `docs/index.md`:**
  - They don't link the three new tutorials (`spectro_observables`, `detection_roc`, `orbit_fitting`).
  - The Imaging bullet is missing a comma.
  - The README never mentions `pip install "virgil-astro[orbits]"`, and neither do `docs/orbit_fitting.md` and `docs/api/orbits.md`. The import error does give the hint (`orbits.py:68`), which is good.
- **`docs/detection_roc.md:8,244`** say `DetectionMC` and the null simulators come "in the next stage". They shipped in #221.
- **`docs/spectro_observables.md`** carries contributor jargon:
  - "Stage 6d gains" (line 58);
  - "design note S §2.3" (line 148);
  - a "Not yet" list that "come[s] with Stage 6a PR C" (209-214).

  Rephrase all three for users, and check each "not yet" item against the code.
- **`docs/conventions.md:3-12`:** the table of contents omits "Which priors?" and "Priors and the MAP".
- **Links into `design/*.md`.** `conventions.md:153`, `detection_roc.md` and `imaging_rml.md:240` link into `design/`, which is not part of the site. Make those sentences stand alone.
- **Stale harmonix notes.**
  - `pyproject.toml:62-64` says harmonix is not on PyPI, but harmonix 0.1.0 is. The comment also sits above the `orbits` extra, so it reads as if it described it.
  - CI still installs harmonix from a git pin (`tests.yml:92-95`).
  - Decide on a `[harmonix]` extra.
- **`docs/api/linear.md`** publishes the private `virgil._linear` in the nav. Either make it public (`virgil.linear`) or label the page as internal.
- **The docs build passes.** `mkdocs build --strict` passes with no warnings. Zensical could not be tested without writing `site/` into the repo; CI runs it.

**S7. Doctests cover two modules.** About 1 h.
- **Current coverage.** `tests/test_docstrings.py` runs `doctest` on `virgil.models` and `virgil.grid_fit` only.
- **Running every module locally:** spectra, priors, limits, oidata and likelihood pass, with 49 examples between them. Four examples fail:
  - `orbits.position_angle_prior` (`positions` undefined);
  - `detection.gaussian_null` (`template` undefined);
  - `fitting.gauss_newton_mass` (×2: `scene`, `NUTS`).

  Each is a sketch without `# doctest: +SKIP`.
- **Fix.** Mark the sketches, and parametrise the test over every public module, so new examples are checked.

**S8. Make `flux_param` and `batch_size` keyword-only in the grid and limit tools.** About 30 min.
- `laplace_flux_uncertainty_grid(data, model, grid, flux=None, flux_param=None, ...)` puts `flux` before `flux_param`, unlike its siblings, so positional calls mix them up.
- `detection_statistics` already makes both keyword-only.
- Doing it now costs nothing for the new functions. For the 0.2.0 ones, it needs a deprecation or a CHANGELOG line.

**S9. Add `--durations=30` to the CI pytest line.** 5 min.
- There are 1129 tests, and no timing data anywhere.
- The macOS py3.11 job takes 22.5 min against about 5 min on Ubuntu.
- The likely hot spots are the NUTS no-data tests (`test_no_data_priors.py:42-47`, 1500 + 500 draws; `:328`, 4000) and `test_detection.py`'s Monte Carlo. Measure before acting.

### Later

- **Argument order.**
  - Grid, limit and detection tools take `(data, model, samples_dict)`.
  - `fit`, `numpyro_model`, `whitened_residuals` and `model_loglike` take model first.
  - `loglike`, `laplace_cov` and `fisher` take `(values, params, data, model)`.

  This predates 0.2.0, so changing it is a breaking API change with deprecations, not a 0.3.0 fix. Document the rule in `conventions.md` meanwhile, and consider renaming `samples_dict` to `grid`.
- **Overloaded names.**
  - `sigma` is a significance in `absil_limits` and `injection_limits` but an uncertainty in `ruffio_upperlimit`.
  - `noise=` means fitted error terms in `fit` but a simulator in `injection_recovery`. `noise_scale`, `error_scale` and `with_error_scale` are three spellings of one idea.
  - `contrast_curve` returns flux.
  - Spelling is mixed: British `marginalise` and `regulariser`, American `optimized_*` and `regularized_inverse`.
- **Return containers vary.**
  - Bare arrays: most grid tools.
  - A NamedTuple: `LinearFluxGrid`, whose three trailing fields may be `None`, so it can't be unpacked into three names.
  - Dicts: `detection_statistics`, `best_grid_point`, `radial_profile`, `injection_grid`.
  - A dataclass: `DetectionMC`.

  At minimum, list `LinearFluxGrid` in `docs/api/grid_fit.md` and give `grid_fit` an `__all__`.
- **Top-level exports.** `virgil/__init__.py` exports `detection_statistics` but not the rest of `detection`, the priors or `AngleVector`. Decide on a rule: the top level holds what a tutorial imports.
- **Static fields recompile per value:**
  - `OIData.t_ref` and the orbit classes' `t_ref` (`orbits.py:178, 436, 521, 762, 923`), deliberately static for float64 time;
  - `FluxSpectrum.scale`, `widths` and `poly_width` (`observables.py:313-315`);
  - `DifferentialPhase.prior_width` (`:662`);
  - `_GaussianNull.error_scale`;
  - `Nodes.outside`.

  A loop over epochs, or over prior widths, recompiles each time. Add a sentence to AGENTS.md beside the "traced templates" rule, and make the prior hyperparameters traced leaves.
- **Host-side builders are quadratic.**
  - `DifferentialPhase.build` scans `frame == f` and `row == r` (`observables.py:733-745`).
  - The gains builders do the same per frame (`gains.py:303-315`).

  Group with one `argsort`, as `_closure._groups` does.
- **Long functions:**
  - `OIData.__init__`: 210 lines of code (`oidata.py:126`).
  - `imaging.clean`: 147.
  - `oidata._build_extras`: 120.
  - `DifferentialPhase.build`: 114.
  - `injection_recovery`: 104.
  - `OIData._subset`: 102.

  `OIData.__init__` is the one worth splitting (reader dict → arrays → flags → operators). `DetectionMC` (about 470 lines) could be its own module.
- **`gauss_newton_mass` builds an n × n dense matrix** (`fitting.py:855-876`). A 256² image would need 34 GB. Add a size guard that says to use a diagonal mass instead.
- **`starting_orbits` uses `np.linalg.lstsq` (SVD) on a 2n × 4 design per grid point** (`orbits.py:645-662`). The 4 × 4 normal equations, or `_linear.posterior`, are cheaper.
- **`_linear` in float32.** χ² is off by 7e-5 relative at a prior sd of 200σ and by 2e-5 at 2 × 10⁴σ, against dense float64. The Cholesky and rank-one paths agree, and the log-determinants are exact. Fine, but a sentence in the module notes would stop someone "fixing" it.
- **The Jeffreys rule in `examples/elr_pavo`.**
  - `fit_pavo.py:93` and `robustness/common.py:57-60` use `Uniform` on the diameter and a non-isotropic `Uniform(0, 90)` on the inclination.
  - `pa` is `Uniform` in `gravity_darkened_star` and `mwe_elr_chara`. An axial `AngleVector` fits a pole PA.
  - See also "Only Ben can decide".
- **Weak error messages.** The worst are:
  - `oidata.py:1553` ("noise_scale must be non-negative.");
  - `observables.py:729` ("prior_width must be positive.");
  - `detection.py:1315`, `:1507` and `:1661` ("There are no null draws.", "No injections match the selection.", "Nothing to concatenate.");
  - `gains.py:270`;
  - `spectra.py:79` (a bare `NotImplementedError`).

  Each should say what was passed and what to do.
- **Five `mwe/` notebooks import private names:** `_geometry`, `_precision.cast_tree` and `plotting._enforce_sky_orientation`. Promote `cast_tree`, or accept that the MWEs are internal.
- **`docs/orbit_fitting.md` has a 119-line stretch with no heading or figure** (from line 418), in a 613-line page. Split it.

## Correctness spot checks

| Check | Result |
|---|---|
| `injection_limits` vs `absil_limits`, bounded | agree to 4% (0.0059 vs 0.0061), as expected for the cross-term sign |
| `injection_limits(flux_bounds=None)` | **bug B1**: returns 1000 |
| `injection_limits` with T3AMP extras | **bug B2**: TypeError |
| `absil_limits` at σ = 14, float32 | **S2**: 1e37 or a silent clip, with a misleading message |
| `_linear.LinearMarginal`, Cholesky and rank-one, 200 data in 3 groups, sd 10 and 1000 | χ² and log-det match dense `solve`/`slogdet` to ≤ 2e-5 (f32) and exactly (f64) |
| Recompilation: `linear_flux_grid` prior bounds, `absil_limits`/`injection_limits` σ and `flux_bounds`, `fit` LogUniform bounds, `gtol`, starts | 0 compiles after the first (perf sweep, `tests/_compiles.py`) |
| `IsotropicInclination.flat_coordinate` | the CDF in [0, 1], so `fit`'s bijection is well ordered for any `(low, high)` |
| Doctests outside CI | 4 sketches fail (S7); no wrong outputs |
| `injection_recovery(draw_batch=4)` vs `draw_batch=1` | identical statistics (seeds are per draw, as documented) |

## Performance

**Earlier findings: no regressions.**
- LM picks QR for n ≤ 200 and a capped CG otherwise (`fitting.py:979-992`).
- `_gauss_newton_covariance` is jitted at module level, with a size-chosen jacfwd/jacrev and Woodbury (`fitting.py:795`).
- `ClosureNoise` groups by argsort and factorises each pattern once.
- Every grid goes through `_grid.map_points` (`lax.map` with a batch size).

**New since 0.2.0: what is good.**
- `_linear` is O(n k²), and `gains`, `observables` and `orbits.RVData` reuse it.
- Both bisections run inside `jit` (`fori_loop` / `while_loop`), so there are no host round-trips per step.
- `linear_flux_grid`'s 256-node quadrature is traced and does not recompile on new bounds.
- `injection_recovery` syncs once per chunk.

**Hot spots and hazards:** S3 (the `DifferentialPhase` Cholesky, measured), S4 (detection Monte Carlo memory and padding), S5 (the absil start grid), plus the "Later" items on static fields, quadratic builders and the dense mass matrix.

### Does anything merit a JAX profile? Yes, three questions

None of them can be answered on the laptop, because the question is GPU memory and launch overhead. They are prepared as the ozstar_scripts job `virgil_prof03`: `prof.sbatch`, `submit.sh` (pins virgil `main`), `profile_{detection,visphi,limits}.py`, plus `jaxprof.py` and `jax_trace_summary.py` from `drpangloss_prof`. A README is included. The files are **uncommitted** in `~/code/ozstar_scripts`. The API calls each script makes were smoke-tested locally on tiny problems.

| Profile | Runs | Answers | Decides |
|---|---|---|---|
| **P1 `detection`** (A100) | `injection_recovery` on the NIRISS template and the `detection_roc` grid (19×19×32), 64 null + 64 injected draws, `draw_batch` 1/4/16 × grid `batch_size` auto/4096. Timings, XLA compile counts, a device-memory snapshot per case; traces of one chunk at `draw_batch` 1 and 4 | Is `draw_batch > 1` faster at all (the inner `lax.map`s may serialise under the vmap)? Where does it run out of memory? | S4: the default `batch_size / draw_batch`, and whether `draw_batch` should stay public |
| **P2 `visphi`** (A100 and CPU) | `grad(model_loglike)` on synthetic 4-UT differential phases, line channels 128/512/1024 × frames 8/32, with and without `prior_width`; the largest projection case traced | Does the Cholesky (`cholesky` / `triangular-solve` in `hlo_stats`) dominate at GRAVITY-like sizes, on both devices? | S3: cache the factor at construction |
| **P3 `limits`** (A100) | `absil_limits` with a 60-flux and a 1-flux axis, `injection_limits`, and `linear_flux_grid` (no prior, LogUniform, `n_iter=3`) on a 101×101 grid | Relative cost; the A100 busy fraction (launch-bound bisection?); what the start grid and the quadrature cost | S5: drop the start grid, shorten the bisection |

**Commands** (run on the Mac; `bin/oz` reaches the cluster itself):

```bash
cd ~/code/ozstar_scripts && git add scripts/virgil_prof03 && git commit -m "virgil_prof03: pre-0.3.0 profiles" && ~/code/ozstar_scripts/bin/oz push virgil_prof03 && ~/code/ozstar_scripts/bin/oz submit virgil_prof03 --var PROF=detection && ~/code/ozstar_scripts/bin/oz submit virgil_prof03 --var PROF=visphi && ~/code/ozstar_scripts/bin/oz submit virgil_prof03 --var PROF=limits && ~/code/ozstar_scripts/bin/oz submit virgil_prof03 --var PROF=visphi --var CPU=1 --gres=none && ssh nt 'squeue --me'
```

Afterwards (once `bin/oz status virgil_prof03` shows nothing running):

```bash
~/code/ozstar_scripts/bin/oz pull virgil_prof03 && ls ~/code/ozstar_scripts/results/virgil_prof03/out/*/ && cat ~/code/ozstar_scripts/results/virgil_prof03/out/*/*/*/proc0/summary.txt | head -120
```

The resource requests (1 h, 8 CPUs, 32 GB, one A100) are guesses; set them from `jobinfo` after the first run. `--var CPU=1` only stops `submit.sh` from treating the CPU run as a duplicate of the GPU `visphi` submission.

## Release checklist for 0.3.0

1. **Fix B1 and B2** in `limits.py`, with tests, and share the bisection helper (S5).
2. **Rewrite `orbit_fitting`** with `IsotropicInclination` and `AngleVector`.
3. **Re-execute on OzSTAR** (`virgil_notebooks`) the 22 notebooks #227 touched, plus `orbit_fitting`, `binary_search`, `contrast_limits` and `hierarchical_inference`. Then run `scripts/sync_tutorial_docs.py` and update the §6d MWE numbers in `imaging_plan.md`.
4. **CHANGELOG (B4):**
   - a `## 0.3.0 (date)` heading;
   - one `Added` section;
   - entries for #203, #212 and #214;
   - correct "breaking" labels;
   - a removal version for `Tabulated`;
   - mark 0.2.0 as released;
   - a "Migrating from 0.2" box with the flat-coordinate MAP, #203, `Tabulated` and `starting_image`.
5. **Decide the API names before they freeze:**
   - the prior convention, the `LogUniform` clash and the tuple path (S1);
   - `marginalise_offsets=True`;
   - keyword-only `flux_param` and `batch_size` (S8).
6. **Small fixes:** the S2 guard, the S6 docs fixes, and the S7 doctests across every module.
7. **Version bump** to 0.3.0 (`pyproject.toml:3`). `uv lock --check` passes in CI.
8. **CI:** add `--durations=30` (S9). Optionally install harmonix from PyPI, now that 0.1.0 is out.
9. **Full suite** under float32 and x64 (CI does both). `mkdocs build --strict` and `zensical build --clean` (CI). The wheel smoke job.
10. **Run virgil-validation** against the release candidate. #203 changes the reader's frame grouping, which AGENTS.md says needs it.
11. **Tag `v0.3.0` on main.** `publish.yml` checks the tag against the version.

## Only Ben can decide

- **The prior-passing convention (S1).** Numpyro distributions everywhere? Or keep `(mean, sd)` tuples for analytically marginalised Gaussians?
- **Argument order across the API (Later).** Change it in 0.4 with deprecations, or document it and live with it?
- **`virgil._linear`'s docs page.** Make the module public, or hide the page?
- **`examples/elr_pavo/`.** The PAVO results are kept in the private paper repo, but `reference_posteriors.json` and the fitting and robustness scripts are still in public virgil. Is that intended?
- **`Tabulated`'s removal version.**
- **`draw_batch` in `injection_recovery`.** Keep it public after P1?
