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
uv python install 3.11
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -e ".[dev]"
pre-commit install
```

Cloud agents get this environment from `.github/workflows/copilot-setup-steps.yml`.
CI tests against the newest JAX (`.github/workflows/tests.yml`), and a stale JAX
can hide failures that only appear in newer releases. Bring an existing `.venv`
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
| Build docs | `.venv/bin/python -m zensical build --clean` |
| Refresh the development-carbon page and badge (local only; needs [claude-code-carbon-dashboard](https://github.com/benjaminpope/claude-code-carbon-dashboard) installed) | `python3 scripts/dev_carbon.py` |

Call the venv interpreter directly: `uv run` resyncs the venv to the lock, so use `.venv/bin/python -m <cmd>`.

## Definition of done

Before committing:

1. `bash scripts/lint_local.sh --fix`
2. `.venv/bin/python -m pytest` on the relevant test files or tests only. Do not
   run the full suite locally: heavy JAX work overloads the laptop, and CI runs the full suite.
3. If you touched a tutorial notebook: rerun `scripts/sync_tutorial_docs.py` and
   `pytest tests/test_tutorial_docs_sync.py`.
4. If you touched docstrings or `docs/`: `.venv/bin/python -m zensical build --clean`.

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
match, and the precision expected.

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
- Line length 79, double quotes, rules `E` + `F` + `ICN` (see `pyproject.toml` for ignores).
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
  `model_on_grid` must return exactly what `model` would, only faster.
- New model code goes in `src/virgil/models.py`.
- Bessel functions come from [jaxbessel](https://github.com/benjaminpope/jaxbessel),
  shared with harmonix; fix or extend them there, not in virgil.
- Old exploratory notebooks live in `notebooks/archive/`, which is git-ignored
  and unmaintained: do not read, edit, lint or cite them.

## Package layout

| Module | Contents |
| --- | --- |
| `oidata.py` | `OIData` (observables, flags, operators, residuals), `closure_phases`, `cp_indices` |
| `_closure.py` | `ClosureNoise`: independent, whitened closure phases with Kammerer et al.'s (2020) correlations, used by `whitened_residuals`; `colour` inverts `whiten` for the residual bootstrap (private) |
| `oifits.py` | `read_oifits` / `write_oifits` / `build_hdulist`, astropy only |
| `amigo.py` | AMIGO mixed-DISCO records and `load_oi_data` |
| `models.py` | source models (`SourceModel`, components including the pixel `Image`, `System`, binaries, `HarmonixModel`) and the analytic `cvis_*` functions |
| `likelihood.py` | `whitened_residuals`, `build_model`, `loglike`, `model_loglike`, `joint_*`, `numpyro_model`, `chain_init_params` (numpyro `init_params`, one start per chain), `posterior_predictive_summary` |
| `fitting.py` | `fit(model, priors, data, regularizers)`: MAP fits with `lm`, `lbfgs` or `adam`, float64 by default via `_precision` (sampling uses `likelihood.numpyro_model` with the same arguments), and `gauss_newton_mass`, the Gauss–Newton preconditioner |
| `imaging.py` | regularizers (`TSV`, `TV`, `MaxEntropy`, `Centroid`), `starting_image`, `image_priors`, `nyquist_pixel_scale`, `field_of_view`, `dirty_image`, `beam`, `convolve_beam`, `l_curve`, `log_evidence`, `laplace_samples`, `error_scale`, `diagnose` |
| `inference.py` | Hessian/Laplace/Fisher tools, and the model-level `laplace_cov`, `laplace_parameter_uncertainty`, `fisher` |
| `grid_fit.py` | grid searches: `likelihood_grid`, `optimized_*_grid`, `laplace_flux_uncertainty_grid`, `best_grid_point` |
| `limits.py` | `ruffio_upperlimit`, `absil_limits`, `injection_limits`, `nsigma`, `radial_profile`, flux/contrast/Δmag conversions |
| `detection.py` | `detection_statistics` (Δχ², grid-marginalized log Bayes factor, max SNR; traceable in the data for `lax.map` over simulations) and `local_nsigma`; null simulators (`gaussian_null`, `bootstrap_null`, `rescale_errors`), `injection_grid`, the Monte Carlo driver `injection_recovery` (one compiled kernel over draws) and its NumPy result `DetectionMC` (FAP, thresholds, ROC/AUC, completeness, contrast curves, save/load/concatenate); see `design/detection_roc.md` |
| `fields.py` | Gaussian-process log-brightness for an `Image` (`GaussianField`, a DCT field with a Matérn-like spectrum) |
| `observables.py` | extra observable blocks after `vis`/`phi` in `OIData`: OI_FLUX spectra with marginalized grey scales (`FluxSpectrum`), \|V\|, T3AMP, and continuum-normalized differential phases (`DifferentialPhase`, closure-free beside closure phases); `continuum_operator` |
| `spectra.py` | wavelength-dependent fluxes (`PowerLaw`, `BlackBody`, `GaussianLine`, `LorentzianLine`, `Nodes` for a free flux per channel, and `Sum`; `Tabulated` is deprecated for `Nodes`) accepted as a component's `flux` (SPARCO) |
| `angles.py` | `AngleVector`: an angle prior sampled as a 2-D vector (site `<path>_vec`, ring and von Mises chord residuals), recognised by `fit`, `gauss_newton_mass` and `numpyro_model`; imports nothing from virgil |
| `orbits.py` | Keplerian orbits in virgil's conventions (`KeplerOrbit`, `ThieleInnesOrbit`), solved with jaxoplanet (the optional `[orbits]` extra, imported lazily); see `design/orbit_scene_joint_fitting.md` |
| `epochs.py` | `Epochs`: datasets grouped into named epochs, one snapshot of a time-dependent scene per dataset (or epoch), name-keyed per-dataset `noise`, and the model function, data and summed log likelihood for multi-epoch orbit fits; starting them: `rank_orbits` (trial orbits ranked by the data), `chain_starts` (distinct modes, one per chain), `epoch_positions` (per-dataset positions on the scale-marginalized surface, `marginal_loglike`; `gap_marginal`, `chi2_raw`, `scale`) and `start_from_positions` (positions → `starting_orbits` → ranking with `scales=` → `fit` from distinct starts, as an `OrbitStart`); see `design/visibility_orbits.md` |
| `orbit_search.py` | `score_orbits`: candidate orbits scored exactly on all epochs with shared nuisances (one flux per band with an optional chromatic slope, integrated on a log-uniform grid and also profiled; scale-marginalized per dataset; gains and closure offsets profiled; extra `RVData`/`PositionData` terms), positions at sample times, a work-unit budget; `SharedFlux`, `OrbitScores`, `rank_scores` (quantized, index-broken ties) |
| `simulate.py` | `simulate` (a scene observed with a template's sampling, errors and times, optionally shifted in time) and `bias_test` (fits to many noise draws) |
| `coverage.py` | synthetic coverage for simulations: `ami_grid_record` (AMIGO-style uv grid with a splodge-weighted mode basis), `nrm_oidata` (V² and closure phases), `vlti_oidata` (Earth-rotation tracks, channels), `mask_transfer` |
| `ensemble.py` | PYRA/MYTHRA-style reconstruction ensembles: `EnsembleSpec`, `draw_groups`, `run_group` (one L-curve per geometry group), `combine` (selection and the iterative mean) and `ensemble`, returning an `Ensemble` (mean `Image`, per-pixel σ, raw χ²/N per dataset) |
| `scenes.py` | synthetic truth images for imaging tests (`ring`, `spiral`, `gaussian_blob`); imports only `_geometry` and `_utils` |
| `plotting.py` | figures, notably `plot_grid_map(kind=...)` and `plot_contrast_curve`; for a `DetectionMC`, `plot_null_distribution`, `plot_roc` and `plot_completeness` (duck-typed: `plotting` does not import `detection`) |
| `_elr.py` | Espinosa Lara & Rieutord (2011) Roche shape and gravity darkening on a triangle mesh, ported from S. Dholakia's jax-interferometry (private; used by the gravity-darkened star model) |
| `_deprecate.py` | the 0.4 argument-order shim: `old_order` (accepts the 0.3 model/data order with a `FutureWarning`; `removed=True` makes it a `TypeError` for 0.5) and `renamed` (old keyword names `data_obj`, `observations`, `model_object`, `model_fn`, `samples_dict`); imports `oidata` lazily; see `design/api_argument_order.md` (private) |
| `_geometry.py`, `_utils.py`, `_grid.py` | shared geometry, constants and helpers, and the grid machinery used by both `grid_fit` and `limits` (private) |
| `pipeline/` | `virgil.pipeline`, loaded lazily and needing the `pipeline` extra: stable, scriptable pipelines (`BinaryPipeline`) that run fixed stages, resume, and write a run folder (`run.json`, `summary.json`, HDF5 grids and samples, plots, processed OIFITS, an executed quicklook notebook) with deterministic quality checks; `_core` (`_Pipeline`, `Stage`, `ConfigMismatchError`), `_io` (atomic writes, `Result`, `load`, model specs without pickles), `_checks` (`Check`), `_quicklook`, `binary`, `cli` (`virgil-pipeline`); public docs in `docs/pipeline.md`, design in `design/pipeline.md` |
| `legacy/` | ImPlaneIA-derived OIFITS tools, not imported by `import virgil` |

Imports flow one way: `_utils`/`_geometry`/`_precision` → `oifits`/`amigo`/`_closure`
→ `oidata` → `coverage`; `gains` → `observables` → `oidata`. Separately, `_utils` → `spectra` and `fields`, and
`_elr`/`spectra` → `models` → `likelihood` → `fitting` (which also imports `fields`) →
`imaging` (which imports `fitting`, `fields`, `likelihood` and `models`). `likelihood` →
`inference` → `grid_fit` and `likelihood` → `limits`; `grid_fit` and `limits` also use
`_grid`, which imports only `_utils`, and do not import each other; `limits` →
`plotting`. `detection` imports `grid_fit`, `limits`, `_grid` and `likelihood`. `scenes` imports only `_geometry` and `_utils`. `ensemble` imports `imaging`, `metrics` and `models`. `angles` imports nothing from virgil, and
`likelihood`, `fitting` and `orbits` import it. `orbits` imports only `_utils` and `angles`
(and jaxoplanet lazily), and `models` imports `orbits` (for `Attached` and `OrbitalBinary`). `epochs` imports `likelihood`, `fitting`, `models` and `orbits`; `orbit_search` imports `epochs`, `likelihood`, `models` and `orbits`. `simulate` imports `fitting`.

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
- `flux` means a relative weight. The binaries' companion/primary `flux` (and
  `contrast`) is a legacy exception; no new model may use `flux` as a ratio.
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
`tests/test_tutorial_docs_sync.py` fails if the markdown is stale.

Execute notebooks with the repo's own kernel, `virgil`, which lives in `.venv`. The
user-level `python3` kernel may point at another environment.

```bash
.venv/bin/jupyter-nbconvert --to notebook --execute --inplace NOTEBOOK.ipynb \
  --ExecutePreprocessor.kernel_name=virgil
```

If the kernel is missing, recreate it with
`.venv/bin/python -m ipykernel install --sys-prefix --name virgil --display-name "virgil (.venv)"`.

## OzSTAR outputs and PRs

The `OzSTAR outputs synced` check fails while a PR carries the `awaiting-ozstar` label.
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
- Do not commit notebook output churn unrelated to your change.
- Do not add runtime dependencies to `[project].dependencies` without asking.
- The pre-push hook (`.githooks/pre-push`, via `scripts/lint_local.sh`) runs the pinned ruff through `uvx ruff@<version>`, reading the version from `[tool.ruff] required-version`, so it works in fresh worktrees without a `.venv`. Never bypass it.
- Before the first push of a branch that may predate a tooling fix on main, merge `origin/main` into it (a merge, never a rebase). The pre-push hook runs the branch's own copy of `scripts/lint_local.sh`, so an old branch can carry an old, broken hook. If the hook fails because of the environment rather than the code (e.g. "ruff 0.12 does not match 0.11.0"), merging main is the fix, not `--no-verify`.
- Never use `--no-verify`, never rewrite published history, never commit secrets.
