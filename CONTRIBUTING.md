# Contributing Guide

virgil is an open-source package and welcomes contributions via pull requests.

---

## Getting Started

Firstly, you will need to fork the repository to your own GitHub account. This will allow you to make changes to the code and then submit a pull request to the main repository. To do this, click the fork button in the top right of the repository page. This will create a copy of the repository in your own account that you can make changes to and then request to merge with the main repository.

Next, you will need to clone the repository to your local machine. To do this, open a terminal and navigate to the directory you would like to clone the repository to. Then run the following command:

```bash
git clone --filter=blob:none https://github.com/your-username-here/virgil.git
cd virgil
uv python install 3.11
uv venv --python 3.11 .venv
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
`virgil.plotting`), `legacy` (astroquery, for `virgil.legacy`), `integrations`
(jaxoplanet, for `HarmonixModel`), `test`, `docs`, `notebooks` and `dev` (`test` and
`docs` plus the tools). harmonix itself is not an extra; to run its tests, install it
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

Note that passing locally does not guarantee cross-platform compatibility. On every pull
request, GitHub Actions (`.github/workflows/tests.yml`) runs the suite on Linux with
Python 3.12 (in float32 and float64) and 3.13, on macOS with Python 3.11, and once with
every dependency at the oldest version `pyproject.toml` allows; it also builds the wheel
and sdist, checks them with `twine check --strict`, and imports the wheel in a clean
environment. Raising a dependency floor means editing `pyproject.toml`, which that last
job then tests. The lint workflow only checks: fix what it reports with
`bash scripts/lint_local.sh --fix` and push again.

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

After these steps have been completed, you can commit your changes and push them to your forked repository. These changes should have its formatting and linting checked by the pre-commit hooks. If there are any issues, you will need to fix them before you can commit your changes. Once you have pushed your changes to your forked repository, you can submit a pull request to the main repository. This will allow the maintainers to review your changes and merge them into the main repository!

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

**Each release:**
1. Bump `version` in `pyproject.toml`, run `uv lock`, and merge to `main` with CI green.
2. Check the build locally:
   ```bash
   uv build && uvx twine check --strict dist/*
   ```
3. On GitHub, draft a release with a new tag `v<version>` (e.g. `v0.2.0`) on `main`, write the notes, and publish it. The workflow checks that the tag matches the version, builds and checks the distributions, and uploads them.
