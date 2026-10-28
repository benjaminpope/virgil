# AGENTS.md

> The import package is `virgil`; the PyPI distribution is `virgil-astro` (`pip install virgil-astro`; `pip install virgil` installs an unrelated package).

Guidance for AI coding agents (Copilot, Claude Code, and similar) working in this
repository. Humans should read [CONTRIBUTING.md](CONTRIBUTING.md) instead.

## Project overview

`virgil` fits interferometric (AMI/OIFITS) data with JAX- and zodiax-based models.
Library code lives in `src/virgil/`, tutorials in `notebooks/`, reusable end-to-end
scripts in `examples/`, and tests in `tests/`.

## Setup

```bash
uv python install 3.12
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e ".[dev]"
pre-commit install
```

Cloud agents get this environment from `.github/workflows/copilot-setup-steps.yml`.
CI tests against the newest JAX: it installs from `uv.lock`, which
`.github/workflows/lock-upgrade.yml` upgrades daily through a pull request of its
own, and a nightly canary runs ahead of the lock. A stale JAX locally can hide
failures that only appear in newer releases. Bring an existing `.venv`
up to date with `uv pip install --python .venv/bin/python --upgrade -e ".[dev]"`;
upgrading `jax` alone can leave optax, equinox etc. too old for it.

## Commands

| Task | Command |
| --- | --- |
| Run relevant tests | `.venv/bin/python -m pytest tests/test_models_core.py` |
| Run one test | `.venv/bin/python -m pytest tests/test_models_core.py::test_name` |
| Lint (check only) | `bash scripts/lint_local.sh` |
| Lint (apply fixes) | `bash scripts/lint_local.sh --fix` |
| Lint changed files only | `bash scripts/lint_local.sh --changed --fix` |
| Full pre-commit pass | `.venv/bin/python -m pre_commit run --all-files` |
| Regenerate tutorial docs | `.venv/bin/python scripts/sync_tutorial_docs.py` |
| Regenerate the references page (after editing `references/provenance.toml`) | `.venv/bin/python scripts/sync_references_page.py` |
| Build docs | `.venv/bin/python -m zensical build --clean` |
| Refresh the development-carbon page and badge (local only; needs [claude-code-carbon-dashboard](https://github.com/benjaminpope/claude-code-carbon-dashboard) installed) | `python3 scripts/dev_carbon.py` |

Call the venv interpreter directly: `uv run` resyncs the venv to the lock, so use `.venv/bin/python -m <cmd>`.

## Definition of done

Before committing:

1. `bash scripts/lint_local.sh --fix`
2. `.venv/bin/python -m pytest` on the relevant test files or tests only. Do not
   run the full suite locally: heavy JAX work overloads the laptop, and CI runs the full suite.
3. If you touched a tutorial notebook: rerun `scripts/sync_tutorial_docs.py` and
   `pytest tests/lint/test_tutorial_docs_sync.py`.
4. If you touched docstrings or `docs/`: `.venv/bin/python -m zensical build --clean`.
5. If you touched `references/provenance.toml`: rerun `scripts/sync_references_page.py`.

CI checks linting and formatting (`.github/workflows/lint.yml`) but does not fix
them, so lint before you push.

## Independent validation

Tests in this repository were largely written by AI agents and mostly compare
virgil with itself. Independent checks live in
[virgil-validation](https://github.com/benjaminpope/virgil-validation), which
simulates data with its own first-principles code and dLux and must not import
or copy virgil. When you change a model's conventions, its Fourier transform or
the OIFITS reader, run that suite against your branch. If someone wants a
validation that is not covered there, ask them to open an Issue on
virgil-validation (https://github.com/benjaminpope/virgil-validation/issues)
detailing the request: the model or function, the independent result it should
match, and the precision expected. Before a release the validation evidence
must be pinned to the release commit (see the evidence gate under Releasing in
`CONTRIBUTING.md`).

## Methods beyond published practice

Until 1.0, virgil implements the state of the art and makes no unsupported
innovations; the community will not yet trust LLM-driven development at this scale.
Search the literature (ADS, arXiv) before calling anything new. Every method that goes
beyond published practice gets an entry in a design-note register (the radio one is
`design/radio_imaging_innovations.md`) giving:

- the claim and a short derivation;
- ADS-verified references, also in `references/provenance.toml`, with a one-line
  `Source:` in the docstring;
- the unit tests that pin it and the virgil-validation check;
- a status: *established*, *derived* or *novel / experimental*.

Novel or experimental methods are opt-in, never defaults, until virgil-validation has
validated them. The rule binds new entries; gaps in existing ones are listed in the register
as to-do. Public docs do not link `design/` notes.

## Anti-reward-hacking & scientific integrity (mandatory for all AI agents)

AI agents (Claude, Argon/Antigravity, Copilot, Codex) must never game release
gates, test assertions, or linter rules at the expense of mathematical
validity, generalizability, or code legibility:

1. **No target-specific heuristics or rescue cascades.** Never add a heuristic,
   magic threshold, fallback pass, or mode-grouping rule to push a specific
   benchmark target (e.g. Gl 229, ι Peg, AL Dor, AK For, HD 9312, HD 142527,
   IW Car, PS Gem, or an Imaging Beauty Contest dataset) across a gate. If a
   benchmark fails:
   - First check whether the posterior is genuinely multimodal or
     underdetermined in the published literature (e.g. Gl 229 without RVs).
   - If a heuristic you added breaks another case, **delete or redesign the
     heuristic**—never stack a second compensating heuristic (such as post-hoc
     polish, readmission passes for excluded bands, or dividing a gate by
     `temper_f` to disable the feature on the target it was built for).
2. **No tautological, self-referential, or weakened tests.**
   - Never copy-paste the implementation formula (or call private helpers under
     test) into a test to compute the expected value. Test against analytical
     solutions, dense numerical quadrature (`scipy.integrate`), finite
     differences, or `virgil-validation`.
   - Never pin floating-point outputs copied from running `virgil` itself as a
     ground-truth oracle, and never delete a regressing case (e.g. a failing
     alias band) from a test dictionary.
   - Never widen a test tolerance (`atol`, `rtol`, eigenvalue bounds) or mark a
     failing test `@pytest.mark.xfail` to make CI green without explicit
     sign-off from Ben in the PR description.
   - Never build unphysical synthetic test data (such as negating closure
     phases while keeping V², collapsing frame timestamps to hide orbital
     motion, or shrinking an injected source because default pipeline settings
     truncate it).
   - Never assert vacuous conditions guaranteed by parameter transforms (such
     as `0 <= val < 0.2` on a `Uniform(0, 0.2)` sigmoid-mapped parameter).
3. **Never silence sampler or optimizer failures.** Never add code that
   automatically downgrades MCMC divergences, non-convergence, or high R-hat
   from `"fail"` to `"warn"` (such as probing random directions for steep
   walls).
4. **Never game code-shape or docstring gates.**
   - Do not inline multi-step logic into >120-line monolithic functions to pass
     `scripts/code_shape.py`, and do not stamp generic
     `# deliberate: test seam`, `# deliberate: a named step`, or
     `# deliberate: stage callable` on single-use functions. Extract genuinely
     reusable helpers or cohesive domain stages.
   - Never delete load-bearing parameter contracts (coordinate spaces,
     incompatibilities, units) or cram >4 unrelated parameters onto one
     numpydoc header line to stay under the docstring line limit.

## Instrument-specific work

Calibration and error models specific to the VLTI instruments (GRAVITY,
PIONIER, MATISSE) live in [virgil-vlti](https://github.com/benjaminpope/virgil-vlti),
together with the GRAVITY calibration review. virgil keeps the generic
machinery (readers, likelihoods, nuisance modes) and never imports virgil-vlti;
see that repository's `PLAN.md` for the boundary.

## Conventions

- Ruff is pinned to **0.11.0**; `[tool.ruff] required-version`, the `.pre-commit-config.yaml`
  rev, and `RUFF_VERSION` in the workflows must always match. A different ruff version will
  reformat files differently and fail CI.
- Line length 79, double quotes, rules `E` + `F` + `ICN` + `W605` (see `pyproject.toml` for ignores).
- Import `jax.numpy as np` and `numpy as onp` (ruff ICN001; notebooks are exempt until re-executed).
- virgil does not enable float64: library code must work in JAX's default float32
  (e.g. use `np.finfo(x.dtype)`, not `onp.finfo(float)`). Tests run in float32 unless
  they opt in locally with `with jax.enable_x64(True):` (as `tests/test_utils.py` does);
  never set `jax_enable_x64` globally at import time in a test module.
  Forward-model code must pass in both float32 and float64. Fourier transforms and
  matmuls over pixels use `precision=jax.lax.Precision.HIGHEST`, since on A100/H100
  GPUs the default is TF32 (~1e-3 relative error). Fitting entry points (`fit`) default
  to float64 inside a local `jax.enable_x64(True)` context (`_precision.run_in` and
  `cast_tree`), with float32 as an option.
- Every likelihood, grid, limit and fit uses one residual vector,
  `likelihood.whitened_residuals` (unprojected phases as the chord 2 sin(Δ/2)/σ,
  a von Mises likelihood; correlated closure phases from four or more
  telescopes use whitened sin Δ plus a periodic penalty 2 sin²(Δ/2)/σ, so
  `OIData.n_residuals` exceeds `n_independent`). Do not recompute χ² from `OIData.residuals`, which is for
  display.
- `OIData.model` calls `model_on_grid` when the data carry a `uv_grid` (a regular uv
  lattice, e.g. AMIGO DISCOs); it defaults to `model`. A model that overrides
  `model_on_grid` must return exactly what `model` would, only faster. The
  one exception is a rendered component (`FlaredDisk*`) with
  `rotation_deg="grid"`, its default: on a lattice its pixels follow the
  lattice's frame, elsewhere North up, so the two differ by the pixel
  discretisation alone.
- New model code goes in `virgil.models`.
- New public features start **provisional**: list the name in the module's
  `__provisional__` tuple (`("*",)` for a whole new module), put the notice
  from `virgil._stability.ADMONITION` in its docstring, and add it to
  `docs/stability.md` and the CHANGELOG's Stability section;
  `tests/test_stability.py` checks the first two. Promotion to stable is
  Ben's decision at a release.
- Bessel functions come from [jaxbessel](https://github.com/benjaminpope/jaxbessel),
  shared with harmonix; fix or extend them there, not in virgil.
- Old exploratory notebooks live in `notebooks/archive/`, which is git-ignored
  and unmaintained: do not read, edit, lint or cite them.

## Keeping the code small

`src/` doubled in the ten days after 0.3.0. The growth was features, not
duplication: it came from building ahead of need, keeping what was superseded
and writing history into docstrings. So:

1. **Delete before you add.** A feature PR names what it supersedes and removes
   it in the same PR, or opens a removal issue for the next minor release.
2. **Build on demand.** A plan stage lands in `src/` when a test system,
   validation case or user needs it. Keep exploratory engines in
   virgil-validation or a notebook until then, and land them provisional.
3. **Docstrings say what, docs say why** (next section).
4. **Reuse before writing a numerical helper.** Check `_linear`, `_grid`,
   `_geometry`, `_utils`, `_gauss_newton`, `_flat` and `_precision`, and
   `pipeline/_common`, before adding a quadrature, solver or linear-algebra routine.
5. **Tests share helpers.** Put shared fixtures in `tests/conftest.py`, and
   shared constants and helpers in `tests/_shared.py` or data in `tests/_test_data.py`
   (extend these, don't add a second module, except one `tests/_<pipeline>.py` for the
   files a split pipeline test shares, since `_shared` must not import optional extras); parametrize near-identical tests, and never redefine a
   `_scene`/`_data` helper that exists.
6. **Keep pipelines thin.** A new pipeline subclasses `pipeline/_core` and reuses
   `_common`; aim at about 500 lines.
7. **Make growth visible.** CI comments each PR's net `src`/`tests` line change.
   Above +1,000 `src` lines, the PR description says what the size buys and what
   it supersedes.
8. **No single-use private helpers or oversized monoliths.** CI fails a PR that
   adds a private def (module-level, nested or method) called exactly once in
   the package, at any size: inline it or share it across callers. Exempt: a
   specific `# deliberate: <reason>` comment on the def line or the line above
   (generic rubber-stamp reasons such as `# deliberate: named step`,
   `# deliberate: test seam`, and `# deliberate: stage callable` are rejected);
   `jit`/`filter_jit` kernels; `custom_jvp`/`custom_vjp` functions and their
   `.defjvp`/`.defvjp` rules; `singledispatch` `.register` functions; callbacks
   (a def passed as a value, e.g. to `lax.scan`, `vmap`, `grad`, `map`); and
   names imported from that module in `tests/`. Newly added or grown functions
   over `MAX_FUNC_LINES = 120` body lines (excluding the docstring) also fail
   `python scripts/code_shape.py --since origin/main`, preventing destructive
   inlining into monoliths. The base branch's copy of the script decides, so a
   PR cannot weaken the gate.

## Docstring style

Ben is surprised how long some docstrings are. Keep them short.

- **Public docstring:** a one-line summary; at most about five lines of
  description; numpydoc `Parameters`, `Returns` and `Raises`; at most one short
  `Example`. Aim for 40 lines or fewer. Over 60 is a defect (80 for functions
  or classes with >= 10 documented parameters).
- **Notes:** only the formula a user needs to call the function correctly.
- **Leave out:** derivations, design history, plan or "problem N" numbers,
  PR/issue numbers and named science targets. Put a derivation in a docs page
  and link it with a markdown link; put history in `design/` notes.
- **Shared parameters** (`noise`, `priors`, `flat_coordinates`, `dtype`,
  `regularizers`, ...) are documented once, in `docs/conventions.md` (see "Which
  priors?", "Priors and the MAP" and "Precision"; add `noise` and `regularizers`
  there rather than creating a page), and each docstring says one line and links it.
- **Private helpers:** one or two lines.
- **Deliberate choices:** a surprising or deliberate choice gets a short inline
  `#` comment at the line that makes it, not a docstring paragraph, e.g.
  `mixed=False  # deliberate: the null must not get the broad companion`.
- **Sources:** a docstring cites a source in one line, never in a paragraph:
  `Source: Kammerer et al. (2020), [references](../references.md#kammerer2020).`
  (the link resolves from the API pages). Any new idea or port adds or extends
  its entry in `references/provenance.toml` (`used_in`, and `how = "ported"`
  when code was adapted) in the same PR, and `scripts/sync_references_page.py`
  regenerates the public `docs/references.md`; `tests/lint/test_provenance.py`
  fails on a cited arXiv ID, DOI, ADS bibcode, GitHub/GitLab/Bitbucket
  repository or author-year citation ("Smith (2019)", "Smith et al. 2019") with
  no entry, in the source, docs, notebooks, scripts and top-level Markdown. A
  citation matches the first author and `year`; when the journal year differs
  from the arXiv year, add `alt_years` with an `alt_years_note`. An entry
  with no `used_in` must be cited somewhere or have `role = "bibliography"`. Never invent an arXiv ID: leave the entry `verified = false` until
  it is checked on the arXiv abstract page (`scripts/fetch_references.py`
  resolves entries against ADS). `scripts/sync_references_page.py --by-module`
  and `--unverified` give the census.
- **The gate:** `tests/lint/test_docstring_length.py` allows up to 80 lines for
  functions or classes with >= 10 documented parameters (60 lines otherwise)
  and forbids cramming > 4 parameter names onto a single numpydoc header line:
  shorten the docstring, and put derivations in `docs/` and design history in
  `design/`.
- **Comprehension check:** every docstring-rewrite PR runs one. Give a fresh
  subagent only the signature and the new docstring of each rewritten public
  function. It writes a correct call for a stated task and lists anything it
  had to guess. Each guess restores one line of the docstring. The PR
  description records the results.

## Package layout

| Module | Contents |
| --- | --- |
| `oidata.py` | `OIData` (observables, flags, operators, residuals; `unflagged`, the samples an observable uses, which set every resolution and field), `closure_phases`, `cp_indices` |
| `gains.py` | `GainModes`: calibration gains correlated across channels (and closure-phase offsets), marginalized analytically; used by `OIData.with_gains`/`with_closure_offsets` and `likelihood` |
| `_closure.py` | `ClosureNoise`: independent, whitened closure phases with Kammerer et al.'s (2020) correlations, used by `whitened_residuals`; `colour` inverts `whiten` for the residual bootstrap (private) |
| `oifits.py` | `read_oifits` / `write_oifits` (an `OIData` or a dict of tables, via `_oidata_tables`) / `build_hdulist`, astropy only |
| `amigo.py` | AMIGO mixed-DISCO records and `load_oi_data` |
| `models.py` | source models (`SourceModel`, components including the pixel `Image` and the radial-profile `MultipoleDisk` with its profiles, `Harmonics` and `VonMises`, `System`, binaries, `HarmonixModel`) and the analytic `cvis_*` functions, and `multipole_disk_cvis`, the quadrature engine of `MultipoleDisk` |
| `harmonics.py` | azimuthal harmonics of `MultipoleDisk`: the image-harmonic convention, the positive form's reflection-coefficient machinery (`reflection_coefficients`, `step_up`, `step_down`, `spectral_factor`), the von Mises harmonics, legacy `az_amps`/`az_pas` conversions, and `to_image_harmonics`/`from_image_harmonics`; imports `models` only lazily |
| `likelihood.py` | `whitened_residuals`, `build_model`, `loglike`, `model_loglike`, `joint_*`, `numpyro_model`, `chain_init_params` (numpyro `init_params`, one start per chain), `posterior_predictive_summary` |
| `fitting.py` | `fit(model, priors, data, regularizers)`: MAP fits with `lm`, `lbfgs` or `adam`, float64 by default via `_precision` (sampling uses `likelihood.numpyro_model` with the same arguments), and `gauss_newton_mass`, the Gauss–Newton preconditioner |
| `svi.py` | `variational`: numpyro SVI on `numpyro_model`'s posterior, guides (BNAF/IAF flows, Gaussian, Laplace) started at a fit and its `gauss_newton_mass` covariance, `VariationalResult` (draws keyed like NUTS's, losses, PSIS k̂); see `design/sampler_flat_coordinates.md` |
| `imaging.py` | regularizers (`TSV`, `TV`, `MaxEntropy`, `Centroid`), `starting_image`, `image_priors`, `nyquist_pixel_scale`, `field_of_view`, `image_field` (the data-driven field: k FWHMs from `second_moment_fwhm`, between an inner working angle and the interferometric field), `dirty_image`, `beam`, `convolve_beam`, `l_curve`, `log_evidence`, `laplace_samples`, `error_scale`, `diagnose` |
| `priors.py` | invariant priors that are not plain numpyro priors: `IsotropicInclination`, `IsotropicLatitude`, hierarchical (population) scales; imports nothing from virgil |
| `inference.py` | Hessian/Laplace/Fisher tools, and the model-level `laplace_cov`, `laplace_parameter_uncertainty`, `observed_information` (and `expected_fisher` for a fixed-error prediction) |
| `grid_fit.py` | grid searches: `likelihood_grid`, `optimized_*_grid`, `laplace_flux_uncertainty_grid`, `best_grid_point` |
| `limits.py` | `ruffio_upperlimit`, `absil_limits`, `injection_limits`, `nsigma`, `radial_profile`, flux/contrast/Δmag conversions |
| `detection.py` | `detection_statistics` (Δχ², grid-marginalized log Bayes factor, max SNR; traceable in the data for `lax.map` over simulations) and `local_nsigma`; null simulators (`gaussian_null`, `bootstrap_null`, `rescale_errors`), `injection_grid`, the Monte Carlo driver `injection_recovery` (one compiled kernel over draws) and its NumPy result `DetectionMC` (FAP, thresholds, ROC/AUC, completeness, contrast curves, save/load/concatenate); see `design/detection_roc.md` |
| `fields.py` | Gaussian-process log-brightness for an `Image` (`GaussianField`, a DCT field with a Matérn-like spectrum; `MultipoleField`, a multipole-kernel residual field aligned with a disk, sharing the radial kernel of `RadialGP`/`RadialGPFunction` and `MultipoleKernel`'s azimuthal spectrum, with optional `geometry_from=` tying to a sibling disk, resolved by `System`) |
| `observables.py` | extra observable blocks after `vis`/`phi` in `OIData`: OI_FLUX spectra with marginalized grey scales (`FluxSpectrum`), \|V\|, T3AMP, and continuum-normalized differential phases (`DifferentialPhase`, closure-free beside closure phases); `continuum_operator` |
| `spectra.py` | wavelength-dependent fluxes (`PowerLaw`, `BlackBody`, `GaussianLine`, `LorentzianLine`, `Nodes` for a free flux per channel, and `Sum`) accepted as a component's `flux` (SPARCO) |
| `angles.py` | `AngleVector`: an angle prior sampled as a 2-D vector (site `<path>_vec`, ring and von Mises chord residuals), recognized by `fit`, `gauss_newton_mass` and `numpyro_model`; imports nothing from virgil |
| `orbits.py` | Keplerian orbits in virgil's conventions (`KeplerOrbit`, `ThieleInnesOrbit`), solved with jaxoplanet (the optional `[orbits]` extra, imported lazily); see `design/orbit_scene_joint_fitting.md` |
| `epochs.py` | `Epochs`: datasets grouped into named epochs, one snapshot of a time-dependent scene per dataset (or epoch), name-keyed per-dataset `noise`, and the model function, data and summed log likelihood for multi-epoch orbit fits; starting them: `rank_orbits` (trial orbits ranked by the data), `chain_starts` (distinct modes, one per chain), `epoch_positions` (per-dataset positions on the scale-marginalized surface, `marginal_loglike`; `gap_marginal`, `chi2_raw`, `scale`; night `flags` with reasons, `NightFlag`; a uniform-disc `primary_diam`) and `start_from_positions` (positions → `starting_orbits` → ranking with `scales=` → `fit` from distinct starts, as an `OrbitStart`); see `design/visibility_orbits.md` |
| `aliases.py` | `fit_orbit_aliases`: orbits whose epochs undersample the period; alias bands `N = round(T/P)` fitted to the observables of all epochs (error scales marginalized), Laplace and importance-sampling evidences per band, band probabilities and posterior samples; optional uniform-disk primary, per-epoch V² amplitude and flux-flip refits; nights with a blocking flag (`EpochPositions.flags`) seed nothing but still count (`skip_flagged`, `AliasResult.unseeded`); the result API (`AliasResult.save`/`load`, and `scene()`, `residuals(data)` and `chi2_per_night(data)` on bands and the result, which rebuild the fit's scene and raw per-night χ²/N); imports `_gauss_newton`, `detection`, `epochs`, `likelihood`, `models`, `orbits` and `tempering`; `temper=True` tempers every evidence by the effective-channel factor (`AliasResult.temper_f`); see `design/orbit_aliases.md` |
| `tempering.py` | tempering the orbit evidence for correlated spectral channels: `channel_chains`/`adjacent_channels` (the chains of consecutive channels of a dataset), `estimate_factors` (the effective-channel factors f = N/N_eff as the empirical within-chain covariance factor of residuals, per block and pooled, as `ChannelFactors`; lag-1/lag-2 correlations are diagnostics, `flat_signature`), `block_dof`; imports nothing from virgil; used by `aliases` |
| `radio.py` | `ComplexVisData` (arrays or a CASA measurement set via the optional `casatools`; `term` for `fit`, chunked with `lax.map`), `CalibrationNuisance` (flux, per-observation gains and offsets, per-ddid error scales, spectral index, taper), `primary_beam`, `dartboard_folds`; instrument-neutral; provisional |
| `radio_clean.py` | `clean`: Cotton–Schwab CLEAN of `ComplexVisData` (Högbom/multi-scale minor cycles, `tclean`-style arguments, auto-mask), `RadioCleanResult` (FITS, model visibilities, primary-beam correction), `fit_beam`, `restore`; imports `radio`, `_geometry` and `_utils` |
| `radio_toeplitz.py` | `ToeplitzVisData` (from `ComplexVisData.toeplitz`): the χ² of a one-Image radio fit through precomputed dirty beams and images, a scalar-χ² term for `fit` (`chi2_and_log_norm`, `curvature_params`, `curvature_loss`); `crosscheck` against the direct term; imports `radio`, `_nufft_spread`, `_geometry`, `_utils` and `models` |
| `_nufft_spread.py` | the only user of nufftax's private spreader (version-checked, ≥ 0.6.1): spread grids and their exact evaluation at real positions; imports nothing from virgil |
| `orbit_search.py` | `score_orbits`: candidate orbits scored exactly on all epochs with shared nuisances (one flux per band with an optional chromatic slope, integrated on a log-uniform grid and also profiled; scale-marginalized per dataset; gains and closure offsets profiled; extra `RVData`/`PositionData` terms), positions at sample times, a work-unit budget; `SharedFlux`, `OrbitScores`, `rank_scores` (quantized, index-broken ties) |
| `_stability.py` | the stability policy and the provisional notice (`ADMONITION`); imports nothing; each module's `__provisional__` tuple lists its provisional names (private) |
| `simulate.py` | `simulate` (a scene observed with a template's sampling, errors and times, optionally shifted in time) and `bias_test` (fits to many noise draws) |
| `coverage.py` | synthetic coverage for simulations: `ami_grid_record` (AMIGO-style uv grid with a splodge-weighted mode basis), `nrm_oidata` (V² and closure phases), `vlti_oidata` (Earth-rotation tracks, channels), `mask_transfer` |
| `metrics.py` | image-recovery scores against a known truth (`lawson_sigma_over_peak`, `rms_convolved`, `l1_score`, `ncc`), from the imaging contests' published definitions |
| `ensemble.py` | PYRA/MYTHRA-style reconstruction ensembles: `EnsembleSpec`, `draw_groups`, `run_group` (one L-curve per geometry group), `evidence_grid` and `gp_members` (Gaussian-process groups: `GPFit`, `EvidenceGrid`), `combine` (selection and the iterative mean) and `ensemble`, returning an `Ensemble` (mean `Image`, per-pixel σ, raw χ²/N per dataset) |
| `scenes.py` | synthetic truth images for imaging tests (`ring`, `spiral`, `gaussian_blob`); imports only `_geometry` and `_utils` |
| `plotting.py` | figures, notably `plot_grid_map(kind=...)` and `plot_contrast_curve`; for a `DetectionMC`, `plot_null_distribution`, `plot_roc` and `plot_completeness` (duck-typed: `plotting` does not import `detection`) |
| `_elr.py` | Espinosa Lara & Rieutord (2011) Roche shape and gravity darkening on a triangle mesh, ported from S. Dholakia's jax-interferometry (private; used by the gravity-darkened star model) |
| `_deprecate.py` | the 0.4 argument-order shim: `old_order` (accepts the 0.3 model/data order with a `FutureWarning`; `removed=True` makes it a `TypeError` for 0.5; `keywords=` names the parameters a function declares keyword-only, e.g. `likelihood_grid`'s `data` and `grid`, and any other positional call raises a `TypeError` naming the keyword call) and `renamed` (old keyword names `data_obj`, `observations`, `model_object`, `model_fn`, `samples_dict`); imports `oidata` lazily; see `design/api_argument_order.md` (private) |
| `_flat.py` | flat coordinates of priors (`_flat_coordinate`, `_FlatBijection`), shared by `fit` and `numpyro_model`; `flat_sampled` makes numpyro's `biject_to` use them, so NUTS samples in them with `flat_coordinates=True` (opt-in; private; imports nothing from virgil); see `design/sampler_flat_coordinates.md` |
| `_geometry.py`, `_utils.py`, `_grid.py` | shared geometry, constants and helpers, and the grid machinery used by both `grid_fit` and `limits` (private) |
| `_linear.py` | the shared algebra for parameters that enter linearly and are marginalized analytically (`LinearMarginal`), used by `gains`, `observables` and `orbits` (private) |
| `_gauss_newton.py` | the dense damped-Newton (Levenberg–Marquardt) kernel for objectives that are not plain sums of squares: kernel problems (`loss`, `rows`, `diag`), `init`/`step`/`solve`, ε-active box bounds, integer stop codes (private; imports nothing from virgil); see `design/history/gauss_newton_plan.md` |
| `_precision.py` | local float32/float64 for entry points: `run_in`, `cast_tree` and the `in_float64` decorator (private) |
| `pipeline/` | `virgil.pipeline`, loaded lazily and needing the `pipeline` extra: scriptable pipelines (`BinaryPipeline`, `StarPipeline`, `DiskPipeline`, `ImagingPipeline` for a model-independent image: `image`, and `OrbitPipeline` for a multi-epoch binary orbit, a blind alias search: `orbit`) that run fixed stages, resume, and write a run folder (`run.json`, `summary.json`, HDF5 grids and samples, plots, processed OIFITS, an executed quicklook notebook) with deterministic quality checks; `_core` (`_Pipeline`, `Stage`, `ConfigMismatchError`), `_common` (helpers both pipelines share: χ², residual moments, quantiles, data geometry, prior-bound fractions, figure saving), `_io` (atomic writes, `Result`, `load`, model specs without pickles), `_checks` (`Check`), `_quicklook`, `binary`, `star` (with a registry of models by short name), `disk`, `image`, `orbit`, `cli` (`virgil-pipeline`); public docs in `docs/pipeline.md`, design in `design/pipeline.md` |
| `legacy/` | ImPlaneIA-derived OIFITS tools, not imported by `import virgil` |

Module-level imports flow one way. `_utils` is at the bottom; `_geometry`, `_linear`, `_grid`, `spectra`, `scenes` and `_elr` import only `_utils` (and `scenes` also `_geometry`); `_precision`, `_flat`, `_gauss_newton`, `angles`, `priors` and `_closure` import nothing from virgil; `_deprecate` is imported everywhere and imports `oidata` only lazily. Then `oifits`, `amigo`, `_closure`, `gains` and `observables` (both on `_linear`) → `oidata` → `coverage`. `orbits` imports `_utils`, `_geometry`, `_linear`, `angles` and `priors` (and jaxoplanet lazily); `_elr`, `spectra` and `orbits` → `models` → `fields` (which also imports `_utils` and `_geometry`) and `likelihood` (which also imports `angles`, `_flat` and `gains`) → `fitting` (also `fields`) → `imaging` (also `fields` and `spectra`). `likelihood` → `inference` → `grid_fit`, and `likelihood` → `limits`; `grid_fit` and `limits` both use `_grid` and do not import each other; `limits` → `plotting`. `detection` imports `grid_fit`, `limits`, `_grid` and `likelihood`. `metrics` imports `imaging`; `ensemble` imports `fields`, `fitting`, `imaging`, `metrics` and `models`. `epochs` imports `_gauss_newton`, `likelihood`, `fitting`, `models` and `orbits`; `orbit_search` imports `epochs`, `likelihood`, `models` and `orbits`; `aliases` imports `_gauss_newton`, `detection`, `epochs`, `likelihood`, `models`, `orbits` and `tempering` (which imports nothing from virgil). `svi` imports `fitting` and `likelihood`; `radio` imports `_geometry`, `_utils` and `models`; `radio_clean` imports `radio`, `_geometry` and `_utils`; `radio_toeplitz` imports `radio`, `_nufft_spread`, `_geometry`, `_utils` and `models`; `simulate` imports `fitting`.

Six imports run inside functions, to break what would be cycles: `amigo` → `oidata`, `_deprecate` → `oidata`, `epochs` → `orbit_search`, `likelihood` → `observables`, `plotting` → `imaging` and `radio` → `radio_toeplitz`. Keep new cross-module imports at module level and in the order above; a new lazy import needs a comment saying which cycle it breaks.

## Binary orbits: solved problems

Each row was solved before; never re-derive one or report its symptom as an
open problem. A fix has a comment at the code that handles it, a docstring
paragraph and a test, and a row here. Rows without a pinning test say which
phase of the orbit-consolidation plan adds it (virgil-validation's
`AGENTS.md` has the matching table for the comparison side).

| Problem | Fix | Where | Test |
|---|---|---|---|
| Integer orbit count (alias band N = round(T/P)) | band scan over the period range, mode-keeping, evidence per band | `aliases.fit_orbit_aliases`, `alias_bands` | `test_aliases.py::test_injected_alias_wins_in_closure_phases` |
| Single-night alias peaks (Gl 229 2024-02-27, 2024-03-28) | flag the night (`alias-peaks`, `low-gap`, `chi2-high`, ...): it seeds no position, its data still count | `epochs.epoch_positions` (`EpochPositions.flags`), `aliases.fit_orbit_aliases(skip_flagged=True)`, `start_from_positions`, `AliasResult.unseeded` | `test_epochs.py::test_epoch_positions_flags_each_night_with_a_code_and_a_reason`, `test_aliases.py::test_a_single_night_alias_is_flagged_and_the_alias_fit_finds_the_band` |
| Separation below λ/2B_max | flag (`sep-below-resolution`), reported only (seeds unless `flag_sep_blocks`) | `epochs.epoch_positions` | `test_epochs.py::test_epoch_positions_flags_each_night_with_a_code_and_a_reason` |
| Flux at its bound (twin / label swap) | flag (`flux-at-bound`) | `epochs.epoch_positions` | same test |
| Resolved primary (ι Peg, 1.05 mas) | uniform-disc primary in the per-night seeds | `epochs.epoch_positions(primary_diam=)` (a fixed diameter or None) | `test_epochs.py::test_ud_seeds_are_unbiased_for_a_resolved_primary` |
| Resolved primary in the orbit fit, automatically (problem 10) | `primary_diam="auto"` (default): a first pass fits a shared UD diameter to the per-night positions; resolved when it gains 5 nats over a point primary, then the engine fits the diameter and the seeds are refined with it | `epochs.estimate_primary_diameter`, `aliases._auto_primary`, `AliasResult.primary_fit` | `test_epochs.py::test_a_resolved_primary_is_estimated_and_an_unresolved_one_is_a_point`, `test_aliases.py::test_auto_primary_reseeds_with_the_diameter_it_fits`, `test_aliases.py::test_a_resolved_primary_is_fitted_and_an_unresolved_one_is_a_point`, `test_aliases.py::test_a_fixed_diameter_seeds_are_kept_with_a_warning_when_unresolved`, `test_aliases.py::test_a_diameter_estimate_at_the_grid_edge_is_recorded` |
| Resolved primary in the orbit fit | `primary_diam` fixed or `"fit"` | `aliases.fit_orbit_aliases` | `test_aliases.py::test_a_fitted_diameter_is_recovered` |
| Cycle counts from short arcs | `short_arcs`: night velocities from the two halves of each night; nights joined shortest gap first, each step keeping the periods whose Thiele–Innes fit to positions + velocities (profiled over e, τ and the constants) is within `arc_delta_chi2` of the best, with a position floor `arc_jitter` λ/B_max; arc seeds are *added* to a kept band's position-profile seeds, never replace them; too few data keep the whole window | `arcs.night_motion`, `arcs.unwrap_cycles`, `arcs.arc_profile`, `aliases._short_arc_bands`, `AliasResult.short_arc` | `test_arcs.py::test_the_true_cycle_count_survives`, `test_arcs.py::test_too_few_data_keep_the_whole_window`, `test_arcs.py::test_errors_underestimated_sixfold_keep_the_true_cycle_count`, `test_arcs.py::test_unwrapping_and_seeds_are_deterministic`, `test_aliases.py::test_short_arc_seeds_come_beside_the_position_seeds` |
| Label swap (twins, AL Dor; problem 8) | `sign_free=True` by default: sign-free seeds are added to the signed ones; `labels_free` when a point-reflected mode is within 3 nats of the best and the tracks match within a fraction of the resolution | `aliases.fit_orbit_aliases`, `aliases._labels_free`, `LABELS_FREE_NATS`, `AliasResult.labels_free` | `test_aliases.py::test_a_twin_is_labels_free_and_a_contrast_pair_is_not`, `test_aliases.py::test_a_twin_fit_is_labels_free_with_default_arguments`, `test_aliases.py::test_sign_free_seeds_add_to_the_signed_ones_and_keep_the_true_orbit`, `test_aliases.py::test_labels_free_tolerance_is_a_fraction_of_the_resolution` |
| Period window from the data (problem 5) | `p_range=None` (default): `period_window`: a Nyquist-style sampling limit at the bottom (the companion moves less than half the resolution within the longest night, field = median unflagged separation, a documented λ/B_min without positions, never a published period) and 4 x the span at the top; capped at `n_bands_max` bands with a warning and `window_capped`; `period_bound` when the best period is at the top or σ(log P) is wide (short arc: P unconstrained), `period_at_lower_bound` when it is at the bottom (the limit may have cut the truth) | `aliases.period_window`, `aliases._night_length`, `aliases._field_mas`, `aliases._cap_bands`, `AliasResult.period_bound`, `AliasResult.period_at_lower_bound`, `AliasResult.window_capped` | `test_aliases.py::test_period_window_limits_on_simple_inputs`, `test_aliases.py::test_without_p_range_the_window_comes_from_the_data`, `test_aliases.py::test_a_best_period_at_the_top_of_the_window_is_period_bound`, `test_aliases.py::test_a_short_arc_fit_is_period_bound_with_default_arguments`, `test_aliases.py::test_the_night_is_measured_from_frame_gaps_not_the_dataset_span`, `test_aliases.py::test_the_window_field_is_the_median_unflagged_separation`, `test_aliases.py::test_a_default_window_with_too_many_bands_is_capped_and_says_so`, `test_aliases.py::test_a_flat_period_likelihood_is_period_bound_wherever_the_best_is`, `test_aliases.py::test_a_best_period_just_above_the_bottom_is_at_the_lower_bound` |
| Flux flip (companion brighter than the primary; problem 9) | `flip_flux` refit with the labels swapped (the default) | `aliases.fit_orbit_aliases` | `test_aliases.py::test_the_flip_acts_only_on_a_flux_pinned_at_the_top`, `test_aliases.py::test_a_brighter_companion_is_found_by_the_flux_flip` |
| V² amplitude (contrast loss; problem 13) | per-night amplitude, profiled; on by default (range 0.5 to 1.2) for squared visibilities | `aliases.fit_orbit_aliases(vis_amp=True)` | `test_aliases.py::test_the_epoch_v2_amplitude_is_profiled`, `test_aliases.py::test_default_vis_amp_is_off_for_data_without_squared_visibilities`, `test_epochs.py::test_an_amplitude_loss_with_a_point_primary_is_not_a_resolved_disc`, `test_aliases.py::test_the_engine_defaults_need_no_per_system_arguments` |
| Result must be reloadable and checkable | `AliasResult.save/load` (JSON with a `format` field; a file without it loads as legacy, a newer one is refused), `scene`, `residuals`, `chi2_per_night` | `aliases.AliasResult` | `test_aliases.py::test_a_saved_result_loads_back_equal`, `test_aliases.py::test_a_saved_result_records_its_format_and_refuses_a_newer_one` |
| Too few seeding nights (one unflagged night, all flagged) | fewer than two seeds fall back to random starts with a warning; seeds from three or fewer nights keep random candidates | `aliases._seeding_positions` | `test_aliases.py::test_fewer_than_two_seeding_nights_fall_back_to_random_starts` |
| Untempered evidence (correlated channels: band p = 1.000 at χ²/N 12, posteriors far too narrow) | `temper=True` (default): per-block `dof = 1/f_b` in the likelihood, f = Σ(Σ_chain z)²/Σz² (the empirical within-chain covariance, exact for AR(1) and common-mode correlation) of the residuals of each night's best static fit, used as max(f, 1) (white noise gives f near 1); ρ1/ρ2 are diagnostics and a flat signature warns; `temper=False` is the old evidence, a float a fixed f; `LABELS_FREE_NATS` and `RESOLVE_NATS` are in tempered nats | `tempering.estimate_factors`, `aliases._temper_factors`, `aliases._loglike`, `epochs._Surface.score`, `AliasResult.temper_f`, `.temper_blocks`, `.temper_rho`, `.tempering` (largest block f above 2, raw f below 0.5, or flat), `AliasResult.block_dof()` | `test_tempering.py::test_tempering_turns_an_overconfident_band_probability_into_a_calibrated_one`, `test_tempering.py::test_the_tempered_posterior_is_calibrated_and_the_untempered_one_is_not`, `test_tempering.py::test_the_estimator_recovers_f_for_ar1_and_for_common_mode_noise`, `test_tempering.py::test_temper_false_reproduces_the_untempered_evidence`, `test_tempering.py::test_the_factor_is_saved_and_loaded` |
| Default multi-epoch run must be blind (Gl 229 Ba–Bb: a two-season gap leaves many integer orbit counts; the paper period, a window or a warm start must never be needed) | `OrbitPipeline` (`virgil.pipeline`): the default `search` calls `fit_orbit_aliases(p_range=None, primary_diam="auto", ...)` with the engine's defaults (`init=None`, `skip_flagged=True`, pinned by `test_the_default_run_is_blind`); the per-night positions come from a grid widened until no best peak is on its edge; bands are checkpointed in the run folder | `pipeline.orbit.OrbitPipeline`, `orbit._night_positions`, `orbit._search`, `orbit._modes` | `test_pipeline_orbit.py::test_blind_search_resolves_the_two_season_gap_alias`, `test_pipeline_orbit.py::test_the_position_grid_widens_until_no_peak_is_on_its_edge`, `test_pipeline_orbit.py::test_the_default_run_is_blind` |
| Twin through the pipeline (labels undetermined) | the summary carries `modes.labels_free` and the `labels_free` check warns | `orbit._modes`, `_checks.labels_free` | `test_pipeline_orbit.py::test_a_twin_is_labels_free` |
| Resolved primary through the pipeline | the engine's `primary_diam="auto"` estimate, reported by `search.primary_resolved` and the `primary_resolved` check | `orbit._search`, `_checks.primary_resolved` | `test_pipeline_orbit.py::test_a_resolved_primary_is_fitted` |
| A single star must not be reported as an orbit | `modes.no_orbit`: the best orbit's evidence against the star alone (the engine's likelihood with an invisible companion); `no-orbit` skips the posterior; the `no_orbit` check fails below 0 nats and warns below 10 | `orbit._null_verdict`, `orbit._modes`, `_checks.no_orbit` | `test_pipeline_orbit.py::test_a_single_star_is_no_orbit` |
| High eccentricity (e = 0.8) through the pipeline | `ecc_max` is a setting; the posterior samples e on its band prior | `orbit._sample_band` | `test_pipeline_orbit.py::test_a_highly_eccentric_orbit_is_recovered` |
| An interrupted alias search must not refit finished bands | the band checkpoints live in `stages/search/bands`, outside the stage's declared outputs, so a re-run keeps them and `resume=False` removes them | `orbit._band_checkpoints`, `_core._discard` | `test_pipeline_orbit.py::test_band_checkpoints_survive_a_rerun_but_not_a_fresh_run` |
| One outlying night must not decide the verdict | `modes.robustness` refits the search without the night of highest raw χ²/N (`drop_outlier_night`, skipped below 3 nights); the `night_dropped` check warns when the orbit-or-not verdict or the winning alias band changes | `orbit._robustness`, `_checks.night_dropped` | `test_pipeline_orbit.py::test_the_most_outlying_night_is_dropped_and_the_verdicts_compared` |
| A failed alias band must not hide behind a decisive winner; files of one night split by 0h UT are one night | `search.failed_bands` and the `bands_failed` and `winner_flags` checks; night labels chain pieces within `gap_days`; the single-star null refines its diameter grid | `orbit._search`, `orbit._read_nights`, `orbit._null_verdict`, `_checks.bands_failed` | `test_pipeline_orbit.py::test_orbit_checks_have_the_documented_thresholds`, `test_pipeline_orbit.py::test_from_oifits_groups_files_into_nights` (and the `mode_converged` check: a search that stops short of the optimum, with the effort defaults read from the engine, `orbit._engine_defaults`) |


## Flux and contrast

`flux` always means a flux *relative to the primary* (companion/primary, so
0.01 for a companion 100 times fainter). Reports and plots follow the
astronomical convention instead: **contrast** is primary/companion (100) and
**Δmag** is `2.5 log10(contrast)` (5 mag). Convert with
`virgil.limits.flux_to_contrast` / `flux_to_delta_mag`, and plot with
`units="flux" | "contrast" | "delta_mag"`. Never name a model parameter
`contrast`.

## Image coordinate convention

The user-facing version of the conventions (coordinates, Fourier sign, observables, fluxes, times) is `docs/conventions.md`; keep it in step with this section.

**This convention must never be violated.** It has been the direct cause of real
bugs (see below), and any new coordinate, rendering, or plotting code must be
verified against it with a direct orientation test — not just a
finiteness/normalization check.

- Rendered images and `(dra, ddec)` grids are 2D arrays using standard array
  indexing: the first index (row) runs top-to-bottom, the second index (column)
  runs left-to-right. That part is generic.
- The **physical sky coordinates** assigned to those array positions follow
  astronomical convention, not a generic image convention: **East points left,
  North points up**.
  - `x` (a right-ascension-like offset, e.g. `dra`) **increases to the left**
    (East is positive `x`).
  - `y` (a declination-like offset, e.g. `ddec`) **increases upward** (North
    is positive `y`).
  - Consequently: as the **row index increases, `y` decreases** (row 0 =
    North, the top row). As the **column index increases, `x` decreases**
    (column 0 = East, the leftmost column).
- The coordinate origin `(x, y) = (0, 0)` sits at the **geometric center of
  the pixel grid** — the center of the center pixel(s), not a pixel edge.
- **Position angles** (binary separations, an ellipse's projected major axis,
  azimuthal modulation phases) are measured **counter-clockwise from the top**,
  i.e. **North-to-East**: PA=0° points North, PA=90° points East.

The canonical, tested reference implementation is `image_coordinates` in
`src/virgil/_geometry.py` (image-plane pixel coordinates) and the elliptical
rotation/stretch helpers in the same module (`undo_`/`apply_elliptical_transf_coord`
and `..._spat_freq`). Every geometric `SourceModel`'s `model()`/`render()` must
be dimensionally consistent with these, and should have a direct regression
test analogous to `test_gaussian_disk_render_uses_interferometric_image_orientation`,
`test_uniform_disk_render_uses_interferometric_image_orientation`, or
`test_modulated_gaussian_rim_render_follows_north_to_east_pa_convention` in
`tests/test_models_sources.py` — i.e. one that asserts flux/argmax lands at the
*correct* pixel for a known offset or PA, not just that the output is finite.

**Plotting rule:** any `imshow`-based display of a sky-coordinate image or grid
must end up with the x-axis increasing toward the left (East) and the y-axis
increasing toward the top (North) *regardless of how the underlying
array/extent/origin was constructed*. Do not assume a particular `dra`/`ddec`
axis ordering (ascending vs. descending) — different parts of this codebase
build these axes both ways. Use `_enforce_sky_orientation` in
`src/virgil/plotting.py`, which corrects an `Axes`' final displayed limits
regardless of the plotted array's construction, rather than hand-tuning
`origin`/`extent` per call site.

Historical motivation: this exact convention was violated in three independent
places found in one audit — `BinaryModelAngular`'s position angle was mirrored
about the North-South axis (a real bug in `model()`/`render()`, not just
display); `notebooks/source_models.ipynb`'s render plots used `origin="lower"`
with a non-reversed extent; and two of `plotting.py`'s five grid-plotting
functions displayed `dra` backwards. None of these were caught by existing
tests because none of them asserted orientation directly — see
`add_modulated_gaussian_rim_and_uniform_disk`'s commit history for the fixes.

Every model's `render()` must also be consistent with its `model()`: the
Fourier transform of the rendered image must reproduce the visibilities
(`test_render_fourier_transform_matches_model_visibilities`). Add new models
to that test.

## Model composition

- There are two kinds of `SourceModel`. **Components** (`Component`
  subclasses: `PointSource`, `GaussianDisk`, `EllipticalGaussian`,
  `GaussianArc`, `UniformDisk`, `ModulatedGaussianRim`, and others) are pure shapes normalized to unit flux, with
  `flux`, `dra`, `ddec`; their `flux` is a relative weight. **Scenes**
  (`System`, the binaries, `HarmonixModel`) are whole normalized skies with
  weight 1 inside a `System`, unless they carry their own `flux` weight as
  `System` does.
- `flux` means a relative weight. The binaries' companion/primary `flux` is a legacy exception; no new model may use `flux` as a ratio.
- Components never contain a built-in star; compose one with
  `System(star=PointSource(), ...)`. New shapes subclass `Component` and
  implement `_centred_cvis` and `_centred_image`. Anything that can be drawn
  implements `_image`; `SourceModel.render` handles the grid and
  normalization.
- `System(**named)` mixes components as `sum(f_i V_i) / sum(f_i)`. Only flux
  ratios are identifiable: keep one reference component (usually the star) at
  `flux=1`. Components are stored as ordered tuples (`names` is static), so
  their order survives pytree operations. Names must be identifiers that do
  not clash with `System` attributes.
- There is deliberately no `+`/`*` operator sugar: parameter paths are the
  public fitting interface and must depend only on the names the user chose.
- Parameters are addressed by zodiax dot-paths through component names
  (`"comp.flux"`); tools accept a template model plus paths anywhere they
  accept a model class (`build_model`). The argument is called `model`.
- Grid tools that optimize a flux (`optimized_likelihood_grid`,
  `optimized_flux_grid`, `laplace_flux_uncertainty_grid`, `absil_limits`, `injection_limits`)
  use the one key whose last part is `flux` (`_utils.resolve_flux_param`);
  if there is none or more than one, the caller must pass `flux_param=`.
  Plotting uses the same rule.
- Fluxes are non-negative. `Component`/`System` reject concrete negative
  fluxes in `__check_init__` (traced values cannot be checked),
  `numpyro_model` rejects flux priors with negative support, and grid tools
  reject negative flux axes. The optimizers in `optimized_flux_grid`
  deliberately stay unconstrained, because Ruffio upper limits need the
  unconstrained estimate.
- `SourceModel._weight(wavel)` takes the wavelength (or `None` for the
  reference flux, e.g. when rendering), so chromatic fluxes can be added
  without changing every subclass.
- To fit derived or tied parameters, pass a function returning a model
  wherever a template is accepted (`build_model`, `numpyro_model`,
  `laplace_cov`, grid tools). Positions are tied by nesting.
- Grid tools are `eqx.filter_jit`-compiled with the template's arrays
  **traced**, so new parameter values never recompile and JAX's persistent
  compilation cache can hit across sessions. Do not reintroduce static
  templates or value-dependent code paths (e.g. skipping work when a value
  is a known zero): they bake constants into the HLO, which recompiles for
  every value and defeats the persistent cache.
- `BinaryModelCartesian`/`BinaryModelAngular` stay dedicated classes — the
  core binary-fitting path must not change or slow down. `to_system()` gives
  the equivalent `System` for extension and rendering.

## Do not modify

- `site/` — committed build output, regenerated by the docs tooling.
- `docs/generated/` and `data/*.npy` — generated or fixture data.
- `docs/*.md` pages listed in `scripts/sync_tutorial_docs.py::MAPPINGS` — generated from
  notebooks (see below).
- `docs/references.md` — generated from `references/provenance.toml` by
  `scripts/sync_references_page.py`; edit the TOML (and `references/references_intro.md`).
- `docs/index.md` — generated from `README.md` by `scripts/sync_tutorial_docs.py`; edit
  `README.md`, then run the script.
- `docs/dev_carbon.md` and the README badge between the `dev-carbon-badge` markers —
  generated by `scripts/dev_carbon.py`, which Ben runs locally.
- `notebooks/archive/` — old exploratory notebooks, git-ignored; do not read or edit.
- `.venv/`, `.lint-logs/`.

## Notebooks and docs are coupled

The notebooks listed in `MAPPINGS` in `scripts/sync_tutorial_docs.py` are the source of
truth for the corresponding `docs/*.md` pages. Edit the notebook, execute it so outputs
are current (the sync embeds text and PNG outputs), then run the sync script.
`tests/lint/test_tutorial_docs_sync.py` fails if the markdown is stale (it runs in
`lint.yml`, with the other checks in `tests/lint/` that need no JAX compiles).

Execute notebooks with the repo's own kernel, `virgil`, which lives in `.venv`. The
user-level `python3` kernel may point at another environment.

```bash
.venv/bin/jupyter-nbconvert --to notebook --execute --inplace NOTEBOOK.ipynb \
  --ExecutePreprocessor.kernel_name=virgil
```

If the kernel is missing, recreate it with
`.venv/bin/python -m ipykernel install --sys-prefix --name virgil --display-name "virgil (.venv)"`.

## OzSTAR outputs and PRs

The `OzSTAR outputs synced` check passes with a warning (and a job-summary note) while a PR carries the `awaiting-ozstar` label: waiting is not a failure. The `Notebook outputs not stale` check fails on a stale notebook.
When you hand the user a submit block for a PR, add the label and put a line
`Awaiting OzSTAR: <job ids>` in the PR body. When you commit the pulled outputs, remove
the label and stamp each executed notebook with
`python3 scripts/check_notebook_outputs.py --stamp NOTEBOOK`. Stamp last, after
`lint_local.sh --fix` and any other edit to the notebook. That records
`metadata.virgil.source_hash`, a hash of the code-cell sources only (prose edits cannot
change outputs, so they need no re-run or re-stamp). The `Notebook outputs not stale` check
fails if a code cell changes without a re-execution. It covers every notebook under
`notebooks/`, recursively, skipping the git-ignored `archive/`. Notebooks with outputs but
no hash only produce a warning.

## Testing notes

`pytest` is preconfigured with `-q` and `testpaths = ["tests"]`. The JAX suites are slow to
warm up, so iterate with a single test id, then run the relevant test files. Leave the full suite to CI.

## Pull requests

- Keep diffs small and focused on the request.
- Never edit `CHANGELOG.md` (CI fails it outside a `release*` branch). Add a fragment to `changelog.d/` instead: one file per change, `<PR>.<type>.md`; see `changelog.d/README.md`.
- Do not commit notebook output churn unrelated to your change.
- Do not add runtime dependencies to `[project].dependencies` without asking.
- The pre-push hook (`.githooks/pre-push`, via `scripts/lint_local.sh`) runs the pinned ruff through `uvx ruff@<version>`, reading the version from `[tool.ruff] required-version`, so it works in fresh worktrees without a `.venv`. Never bypass it.
- Before the first push of a branch that may predate a tooling fix on main, merge `origin/main` into it (a merge, never a rebase). The pre-push hook runs the branch's own copy of `scripts/lint_local.sh`, so an old branch can carry an old, broken hook. If the hook fails because of the environment rather than the code (e.g. "ruff 0.12 does not match 0.11.0"), merging main is the fix, not `--no-verify`.
- Never use `--no-verify`, never rewrite published history, never commit secrets.
