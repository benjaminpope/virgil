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
8. **No single-use private helpers.** CI fails a PR that adds a private def
   (module-level, nested or method) called exactly once in the package, at any
   size: inline it. Exempt: a `# deliberate: <reason>` comment on the def line or
   the line above; `jit`/`filter_jit` kernels; `custom_jvp`/`custom_vjp` functions
   and their `.defjvp`/`.defvjp` rules; `singledispatch` `.register` functions;
   callbacks (a def passed as a value, e.g. to `lax.scan`, `vmap`, `grad`,
   `map`); and names used in `tests/`. Only additions fail, measured against the
   PR base (`python scripts/code_shape.py --since origin/main`). The base branch's
   copy of the script decides, so a PR cannot weaken the gate.

## Docstring style

Ben is surprised how long some docstrings are. Keep them short.

- **Public docstring:** a one-line summary; at most about five lines of
  description; numpydoc `Parameters`, `Returns` and `Raises`; at most one short
  `Example`. Aim for 40 lines or fewer. Over 60 is a defect.
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
- **The gate:** `tests/lint/test_docstring_length.py` fails on any docstring
  over 60 lines, with no allowlist: shorten the docstring, and put
  derivations in `docs/` and design history in `design/`.
- **Comprehension check:** every docstring-rewrite PR runs one. Give a fresh
  subagent only the signature and the new docstring of each rewritten public
  function. It writes a correct call for a stated task and lists anything it
  had to guess. Each guess restores one line of the docstring. The PR
  description records the results.

## Package layout

Library code is in `src/virgil/`; `virgil.pipeline` (the `pipeline` extra) holds the
scriptable pipelines and `legacy/` the ImPlaneIA tools that `import virgil` does not load.
The module-by-module table and the import order are in
[design/package_layout.md](design/package_layout.md): **read it before adding a module,
moving code between modules or adding a cross-module import**, and update its row in the
same PR when a module's contents change.

- Module-level imports flow one way, in the order given there; `_utils` is at the bottom.
- Keep new cross-module imports at module level and in that order. A new lazy import
  (inside a function) needs a comment saying which cycle it breaks.

## Binary orbits: solved problems

The table of orbit problems already solved (alias bands, flagged nights, resolved
primaries, short arcs, label swaps, period windows, flux flips, tempering, the blind
pipeline and more), each with its fix, code and pinning test, is in
[design/orbit_solved_problems.md](design/orbit_solved_problems.md). **Read it before
working on `aliases`, `arcs`, `epochs`, `tempering`, `orbit_search`, `pipeline/orbit`
or the orbit checks in `pipeline/_checks`, or before reporting an orbit-fitting symptom.** Never re-derive a
listed problem or report its symptom as an open one. A new fix gets a comment at the
code that handles it, a docstring paragraph, a test and a row in that table
(virgil-validation's `AGENTS.md` has the matching table for the comparison side).

## Flux and contrast

`flux` always means a flux *relative to the primary* (companion/primary, so
0.01 for a companion 100 times fainter). Reports and plots follow the
astronomical convention instead: **contrast** is primary/companion (100) and
**Δmag** is `2.5 log10(contrast)` (5 mag). Convert with
`virgil.limits.flux_to_contrast` / `flux_to_delta_mag`, and plot with
`units="flux" | "contrast" | "delta_mag"`. Never name a model parameter
`contrast`.

## Image coordinate convention

**This convention must never be violated**; it has caused real bugs. East points
left and North up: `x` (`dra`) increases to the left, `y` (`ddec`) upward, so row 0
is North and column 0 is East; the origin is the centre of the pixel grid; position
angles run North-to-East (PA 0° North, 90° East). Every `imshow` of a sky image goes
through `_enforce_sky_orientation` in `plotting.py`, and every model's `render()` must
Fourier-transform to its `model()`. The full rules, the reference implementation in
`_geometry.py` and the orientation tests to copy are in
[design/image_coordinates.md](design/image_coordinates.md): **read it before writing
or changing any coordinate, rendering, plotting or position-angle code**, and give new
code a direct orientation test, not just a finiteness check. The user-facing version
is `docs/conventions.md`; keep the two in step.

## Model composition

**Components** (`Component` subclasses) are unit-flux shapes with `flux`, `dra` and
`ddec`, where `flux` is a relative weight; **scenes** (`System`, the binaries,
`HarmonixModel`) are whole normalized skies. Parameters are zodiax dot-paths through the
names the user chose (`"comp.flux"`), and there is deliberately no `+`/`*` sugar. No
new model may use `flux` as a ratio (the binaries' companion/primary `flux` is a legacy
exception), and new models are added to
`test_render_fourier_transform_matches_model_visibilities`. Jitted kernels keep the
template's arrays traced: never add static templates or value-dependent paths (such as
skipping work when a value is zero), which bake constants into the HLO and recompile for
every value. `BinaryModelCartesian`/`BinaryModelAngular` stay dedicated classes; the core
binary-fitting path must not change or slow down. The
rules on stars, fluxes, `flux_param=`, tied parameters, traced grid templates and the
binary classes are in [design/model_composition.md](design/model_composition.md):
**read it before adding or changing a `SourceModel`, `System` or a grid tool's
handling of templates**.

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
