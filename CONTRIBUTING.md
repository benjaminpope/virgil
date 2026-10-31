# Contributing Guide

virgil is an open-source package and welcomes contributions via pull requests.

---

## Getting Started

Firstly, you will need to fork the repository to your own GitHub account. This will allow you to make changes to the code and then submit a pull request to the main repository. To do this, click the fork button in the top right of the repository page. This will create a copy of the repository in your own account that you can make changes to and then request to merge with the main repository.

Next, you will need to clone the repository to your local machine. To do this, open a terminal and navigate to the directory you would like to clone the repository to. Then run the following command:

```bash
git clone --filter=blob:none https://github.com/your-username-here/virgil.git
cd virgil
uv python install 3.12
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e ".[dev,notebooks]"
```

`--filter=blob:none` makes a partial clone: it downloads every commit, so `git log`,
`git blame` and pushing work as usual, but fetches old versions of files only when you
look at them. The history holds large data files that are no longer used, so this cuts
the download from about 280 MB to about 15 MB.

CI tests against the newest JAX, so if you reuse an existing `.venv`, upgrade it first
with `uv pip install --python .venv/bin/python --upgrade -e ".[dev,notebooks]"`. Upgrading
`jax` on its own can leave optax, equinox etc. too old for it.

The extras are `plots` (pandas and ChainConsumer, for the corner-plot helpers in
`virgil.plotting`), `legacy` (astroquery, for `virgil.legacy`), `orbits` (jaxoplanet,
for `virgil.orbits`, `epochs` and `aliases`), `integrations` (the `orbits` extra again,
for `HarmonixModel`), `pipeline` (h5py and the notebook tools, for `virgil.pipeline`),
`test`, `docs`, `notebooks` (everything the tutorials use) and `dev` (`test`, `docs`
and `orbits` plus the tools). harmonix itself is not an extra; to run its tests, install it
from PyPI as CI does:
`uv pip install --python .venv/bin/python "harmonix>=0.1.0"`.

Ruff is pinned to an exact version in `pyproject.toml` (`required-version`) so that local
runs and CI format identically; installing the `dev` extra gives you the right one. If you
have another ruff on your `PATH`, call the one in `.venv` explicitly.

Then you will need to install the pre-commit hooks. These run Ruff (lint and formatting) on the files you commit; they do not run the unit tests. To do this, run the following command:

```bash
pre-commit install
```

This will ensure that any changes you make will adhere to the code style and formatting guidelines of the rest of the package! Optionally, enable the pre-push hook as well, which refuses to push code that fails the Ruff checks:

```bash
git config core.hooksPath .githooks
```

You can also run linting and formatting manually at any time:

```bash
ruff check . --fix
ruff format src tests examples scripts
```

(Notebooks are linted but not reformatted, matching CI.)

For lower-noise local runs (especially when notebooks are involved), use:

```bash
bash scripts/lint_local.sh
```

This command writes full diagnostics to `.lint-logs/` and keeps terminal output compact.
Useful flags:

```bash
bash scripts/lint_local.sh --fix
bash scripts/lint_local.sh --changed
bash scripts/lint_local.sh --no-notebooks
```

---

## Making Changes

Next you can start to make any changes you desire!

**Unit Tests**

It is important that any changes you make are tested to ensure that they work as intended and do not break any existing functionality. If you are creating _new_ functionality you will need to create some new unit tests, otherwise you should be able to modify the existing tests.

Locally, run the lint checks plus the tests relevant to your change, for example:

```bash
.venv/bin/python -m pytest tests/test_file.py
```

Do not run the full suite locally: the JAX tests are heavy, and GitHub Actions runs the full suite on every pull request. Call the venv interpreter directly rather than `uv run`, which resyncs the venv to the lock.

Note that passing locally does not guarantee cross-platform compatibility. GitHub Actions
(`.github/workflows/tests.yml`, with the tiers tabulated in `.github/scripts/ci_matrix.py`)
runs the suite in tiers. Every pull request runs the tests not marked `slow` on Linux with
Python 3.12, in float32 (split across three jobs) and in float64 (three jobs, skipping the
files listed in `DTYPE_INDEPENDENT` in `tests/conftest.py`, whose results cannot depend on
the float width), with the versions in `uv.lock`; a pull request that edits `pyproject.toml`
also runs once with every dependency at the oldest version it allows. A pull request that
edits `aliases`, `epochs` or `orbit*` (or their test files) also runs the slow tests of
`test_aliases.py`, `test_epochs.py` and `test_orbit*.py` in those jobs. Pushes to main run
the same jobs with all the slow tests. Pull requests labelled `full-ci` (always the release
pull request) and manual dispatches add macOS with Python 3.11, Python 3.13 and the
dependency floors (the last two in two shards each, to stay under the 30-minute pull-request limit). A daily run does all of that, plus a canary with dependencies newer than
the lock, and opens an issue labelled `nightly-failure` when it fails. A failing push to main opens one
issue labelled `red-main`, which the next green push closes. Each pull request also
builds the wheel and sdist, checks them with `twine check --strict`, and imports the wheel in
a clean environment.

Each test not marked `slow` has a budget of 30 s for its call, and its setup has the same
budget (with a warm JAX cache); CI fails a test that exceeds either. Setup is charged to the
file, since the first test of a module on each worker pays for the module's fixtures. Shrink
the test or fixture (a smaller problem, fewer draws, a stub run) or mark it `slow`.
`tests/over_budget.txt` lists the tests and files that were already over budget; remove
lines from it as they are fixed, and never add to it. `pytest-timeout`
fails any test that runs past 300 s. The float32 jobs pack whole test files by the timings
in `tests/test_durations.json`; refresh them occasionally with
`scripts/update_test_durations.py`.

The lint workflow only checks: fix what it reports with
`bash scripts/lint_local.sh --fix` and push again.

**Keep it small**

Prefer deleting to adding, build only what a test system or user needs now, and keep docstrings short (a summary, parameters, returns and at most one example), putting derivations and design history in the docs or `design/` notes instead. A docstring over 60 lines fails `tests/lint/test_docstring_length.py`; and a source cited in the code or docs needs an entry in `references/provenance.toml` (`scripts/sync_references_page.py` regenerates `docs/references.md`). The full rules, including the docstring style, are in [AGENTS.md](AGENTS.md); each pull request also gets a comment showing its net change in `src/` and `tests/` lines and code-shape metrics. CI fails a PR that adds a private function (module-level, nested or method) called exactly once in the package: inline it, or mark it `# deliberate: <reason>` anywhere from the line above the def through the line ending its signature (vacuous reasons are rejected; see AGENTS.md rule 8 and `tests/lint/stamp_allowlist.txt`). Exempt without a comment: `jit` kernels, `custom_jvp`/`custom_vjp` functions and their `defjvp`/`defvjp` rules, `singledispatch` registrations, callbacks passed as values (to `lax.scan`, `vmap`, ...) and names used in `tests/`. Check locally with `python scripts/code_shape.py --since origin/main`.

**Documentation**

Any changes you make should also be appropriately documented! For small API changes this shouldn't require any changes, however if you are adding new functionality you will need to add some documentation. This can be done by modifying the appropriates files in the `docs` directory.

Tutorial pages are notebook-synced: for every notebook listed in `MAPPINGS` in `scripts/sync_tutorial_docs.py`, edit the notebook first, then regenerate the corresponding docs markdown with:

```bash
.venv/bin/python scripts/sync_tutorial_docs.py
```

Notebook style conventions for tutorials:

- Keep explanatory markdown between major code blocks (zodiax sandbox style).
- Keep imports in the top import cell; avoid repeated imports in later cells.
- Prefer shared plotting helpers from `src/virgil/plotting.py` over long notebook-local plotting scripts.
- When a plotting helper is missing, add/extend it in `src/virgil/plotting.py` first, then call it from the notebook.
- Keep notebook plotting cells short and declarative (prepare inputs, call helper, show figure).

Typical helper usage patterns:

- Grid and map visuals: `plot_grid_map(values, grid, kind=...)` (kinds `loglike`, `flux`, `sigma`, `snr`, `limit`) and `plot_contrast_curve`
- Data: `plot_oidata_overview`, `plot_model`
- Posterior diagnostics: `plot_chainconsumer_diagnostics`
- Correlation summaries: `plot_data_model_correlation`

The documentation is built with Zensical. To build it locally and make sure everything is working correctly, run:

```bash
.venv/bin/python -m zensical build --clean
```

To preview it on a local server, run:

```bash
.venv/bin/python -m zensical serve
```

---

## Contributing the Changes

After these steps have been completed, you can commit your changes and push them to your forked repository. These changes should have its formatting and linting checked by the pre-commit hooks. If there are any issues, you will need to fix them before you can commit your changes. Do not edit `CHANGELOG.md`: describe a user-visible change in a fragment file in `changelog.d/` (see its README), so parallel pull requests never conflict. Once you have pushed your changes to your forked repository, you can submit a pull request to the main repository. This will allow the maintainers to review your changes and merge them into the main repository!

---

## Releasing

Releases go to PyPI as `virgil-astro` (the import name is `virgil`) through `.github/workflows/publish.yml`, which uses PyPI trusted publishing. No tokens are involved.

**One-time setup** (maintainer):
1. On pypi.org, under *Your account → Publishing*, add a pending trusted publisher with these settings:
   - PyPI project name `virgil-astro`;
   - owner `benjaminpope`;
   - repository `virgil`;
   - workflow `publish.yml`;
   - environment `pypi`.
2. On GitHub, under *Settings → Environments*, create the environment `pypi`. Adding yourself as a required reviewer makes every upload wait for your approval.

**Each release** (a short list; every item must hold for the commit you tag):
1. **Versions.** `version` in `pyproject.toml` and `CITATION.cff` agree, `uv lock` is run, and the release PR (head branch `release*`, the only kind CI lets edit `CHANGELOG.md`) collects the fragments with `towncrier build --yes --version X.Y.Z`, which writes the dated `## X.Y.Z (YYYY-MM-DD)` section and deletes the fragments. Review the stability-tier list (`docs/stability.md` and the CHANGELOG's Stability section): promotion to stable is a decision for the maintainer.
2. **Main is green on the tagged commit, in the full tier.** Label the release PR `full-ci`, merge it, then dispatch the full tier on the merge commit and wait for it to pass (a push run lacks macOS, py3.13 and the floors, and a newer push may have cancelled it):
   ```bash
   gh workflow run tests.yml --ref main
   ```
   `publish.yml` refuses to build unless `tests.yml` succeeded in the full tier (macOS, py3.13 and the dependency floors) on exactly the tagged commit, so a release can never ship untested code. A red `main` opens an issue labelled `red-main`; close it first.
3. **Evidence gate** ([virgil-validation](https://github.com/benjaminpope/virgil-validation), its method page "Release gate"). All three must hold for the release commit:
   - virgil-validation's tier A passes against that commit (`pytest -m "not slow"` there, with virgil installed from the commit);
   - its evidence (`trust/evidence/*.jsonl`) is regenerated and pinned to that single commit: `scripts/check_evidence_pin.py --release --expect <sha>` passes there;
   - no finding in its ledger ruled `virgil` with severity `bug` is open without a recorded decision (fixed, or accepted with a reason in the release notes).
4. **GPU and long runs** (maintainer, on OzSTAR): the docs-examples run on a GPU passes (ozstar_scripts job `docs_examples_prof`). The Gl 229 Ba–Bb period-alias search was a release gate through 0.4.1 and was retired on 2026-10-23: the alias-search regressions are guarded by the CI tests listed in the "Binary orbits: solved problems" table (linked from AGENTS.md), and a maintainer may still run the cold search (ozstar_scripts job `orbit_aliases`, `virgil.aliases.fit_orbit_aliases` with `--no-init`; the N = 34 band, P ≈ 12.134 d, should rank first) when the search code changes.
5. Check the build locally:
   ```bash
   uv build && uvx twine check --strict dist/*
   ```
6. On GitHub, draft a release with a new tag `v<version>` (e.g. `v0.2.0`) on that commit, write the notes, and publish it. The workflow first checks the gate in item 2, then that the tag matches the version, builds and checks the distributions, and uploads them.
