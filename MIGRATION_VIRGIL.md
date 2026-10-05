# Migration: drpangloss → virgil

**VIRGIL**: Versatile Interferometric Reconstruction and Gradient-based Inference Library.

| | Old | New |
|---|---|---|
| GitHub repo | `benjaminpope/drpangloss` | `benjaminpope/virgil` |
| Import package | `import drpangloss` | `import virgil` |
| Source directory | `src/drpangloss/` | `src/virgil/` |
| PyPI distribution | `drpangloss` | `virgil-astro` (`virgil` is taken on PyPI) |
| Install | `pip install drpangloss` | `pip install virgil-astro` |
| Extras | `drpangloss[nufft]` | none: the NUFFT backend was removed (issue #75) |
| Docs site | `benjaminpope.github.io/drpangloss` | `benjaminpope.github.io/virgil` |

This file replaces any earlier instructions for a rename to `sibylla`. That rename never landed and must not be started.

The refactor is done by one **coordinator**, following the final section of this file. Every other agent follows Steps 1–4.

## Hard rules (all agents, at all times)

- **Never** `git push --force` / `--force-with-lease`. Never rebase a branch that has been pushed. Integrate with `git merge` only.
- **Never** rename, delete or recreate branches, tags, PRs or issues. Never close and reopen PRs.
- **Never** touch `main`, `gh-pages`, or another agent's branch. Work only on your own feature branch.
- **Never** create a repository, branch or tag named `drpangloss`. **Never** publish to PyPI.
- **Never** do the rename by hand, or with your own sed/regex. Use `scripts/rename_to_virgil.py` only. It knows which occurrences are the PyPI name (`virgil-astro`) and which are the import name (`virgil`). Do not edit that script or this file.
- If any step prints `STOP`, or does something you did not expect, stop and report to the human. Do not improvise.

## Step 1: point your clone at the new URL (after the GitHub rename)

GitHub redirects the old URL, so nothing breaks if you skip this. Do it anyway so nothing depends on the redirect:

```bash
git remote set-url origin "$(git remote get-url origin | sed 's/drpangloss/virgil/')"
git fetch origin
```

Branch names, PR numbers and commit SHAs are unchanged. Keep working as normal.

## Step 2: wait for the code rename to land

**Status: the rename has landed** (merged 3 October 2026, PR #126; the tags below exist), so new work uses `virgil`. The rest of this step is the historical instruction. While the rename was pending, agents kept using `drpangloss` everywhere (imports, docs, new files under `src/drpangloss/`) and did not rename anything pre-emptively.

The rename has landed when both tags exist:

```bash
git fetch origin --tags
git rev-parse --verify virgil-rename-base virgil-rename
```

| Tag | Meaning |
|---|---|
| `virgil-rename-base` | last commit on `main` before the rename; adds the migration script |
| `virgil-rename` | the rename commit; byte-for-byte output of the script |

## Step 3: migrate your feature branch (after the tags exist)

If your branch was created from `main` *after* the rename, skip this step: it is already on `virgil`.

Otherwise, with a clean working tree on your feature branch (**including no untracked files**: the copy of the script you get from `virgil-rename-base` stages everything, so move untracked files out of the repository or `git stash -u` them first):

```bash
git fetch origin --tags
git merge --no-edit virgil-rename-base      # ordinary merge; brings in the script
python3 scripts/rename_to_virgil.py --migrate-branch
```

If the first merge conflicts, those are ordinary content conflicts with pre-rename `main` that your branch would hit anyway. Resolve them, run the tests, `git commit`, then run the script. For a conflict in `uv.lock`, run `git checkout --theirs uv.lock && uv lock`. This takes `main`'s lockfile and re-resolves it.

`--migrate-branch` then:

1. checks that `virgil-rename` is exactly what the script produces, and stops otherwise;
2. runs the same script on your branch: `git mv src/drpangloss src/virgil` (including files only your branch added), plus the text rewrite, then commits;
3. records the rename with `git merge -s ours virgil-rename`. Its content is already on your branch, so this avoids conflicts on lines both sides renamed;
4. merges `origin/main` normally, to pick up anything that landed after the rename;
5. checks that no `drpangloss` references remain.

**Known false alarm.** If step 5 prints `STOP: still mentions drpangloss: ['README.md', 'docs/index.md']`, those are `main`'s intentional "Formerly drpangloss" note. The copy of the script running from `virgil-rename-base` doesn't yet know `main` skips them. Run `python3 scripts/rename_to_virgil.py --check`, which uses the script just merged from `main`. If it exits 0, the migration is complete. Later copies of the script (on `main`) stage only tracked files and run this check themselves.

If step 4 prints `STOP: conflicts merging origin/main`, they are ordinary conflicts. Resolve them, `git commit`, and rerun `--migrate-branch`. It is idempotent.

This was rehearsed on 3 October 2026 against a copy of `main` and the live branches:

- `elr-s0` … `elr-s5` migrated with no conflicts. Their diff against `main` matched the original diff exactly, with only the name changed.
- `imaging`, `plan-spectro-orbits` and `elr-s6-chara-notebook` already conflict with current `main`, independent of the rename. They go through the "first merge conflicts" path above, then migrate cleanly.

## Step 4: rebuild the environment and verify

```bash
rm -rf src/drpangloss src/drpangloss.egg-info   # leftovers, if any
uv sync --extra dev                             # or: pip install -e ".[dev]"
python3 scripts/rename_to_virgil.py --check     # must exit 0
python3 -c "import virgil, importlib.util as u, importlib.metadata as m; assert u.find_spec('drpangloss') is None; print(m.version('virgil-astro'))"
uv run ruff check .
uv run pytest
```

Restart any Python kernels or long-running processes: they still have `drpangloss` imported. Then push with a plain `git push`. An open PR updates itself; do not open a new PR for the same work.

## After migration

- Code: `import virgil`, `from virgil.models import ...`. Doc cross-references use `[name][virgil.module.name]`.
- Packaging: the distribution is `virgil-astro` (`pip install virgil-astro`, `importlib.metadata.version("virgil-astro")`; there is no `nufft` extra). Never write `pip install virgil`: that installs an unrelated package.
- New modules go in `src/virgil/`.
- Leave these intentional leftovers alone:
  - the string `drpangloss-synthetic-mixed-disco-v1`: the tag stored inside `data/calibrated_visibility.npy`;
  - historical `.log` files and binary data (FITS headers in the bundled `.oifits` files);
  - this file and `scripts/rename_to_virgil.py`.
  - `docs/contributors.md`, whose history names the original package (its present-tense mentions were edited by hand);
  - `README.md` and `docs/index.md` (generated from it), which say what the package used to be called.

## For the coordinator only

Do this once, while feature agents are paused at Step 2.

### 0. Prerequisites (human, GitHub Settings)

1. `benjaminpope/virgil` already exists: a 2017–18 TensorFlow maximum-entropy imaging repo. Rename it to `virgil-legacy` and archive it. Until then the name is not free.
2. Rename `benjaminpope/drpangloss` → `benjaminpope/virgil`.
3. Tell agents to do Step 1 and pause at Step 2.

### 1. The rename PR

1. Branch `rename-to-virgil` from up-to-date `main`.
2. **Commit A0:** add `scripts/rename_to_virgil.py` and this file (`MIGRATION_VIRGIL.md`). Nothing else.
3. **Commit A:** `python3 scripts/rename_to_virgil.py && git add -A && git commit -m "Rename drpangloss -> virgil (mechanical, script output)"`. Nothing else may go in this commit. Do not touch its output by hand. In the rehearsal on current `main`, the script:
   - moved 22 files and rewrote 120;
   - set `name = "virgil-astro"` in `pyproject.toml` and `uv.lock`;
   - changed `pip install drpangloss` to `pip install virgil-astro` in `README.md` and `docs/index.md`, and `drpangloss[nufft]` to `virgil-astro[nufft]` in `design/` (that extra never existed on `main`: the NUFFT backend had been removed);
   - left everything else as `virgil`;
   - passed `ruff check .` and the full `pytest` suite (390 passed, 5 skipped).
4. **Commit B and later:** hand edits only. Keep this small:
   - `pyproject.toml`: bump `version` (e.g. `0.2.0`). Update `description` and `keywords` if wanted.
   - `mkdocs.yml`: `site_name` currently reads "virgil - the best of all possible interferometry models", which is the Pangloss joke after the mechanical rename. Change it to e.g. `VIRGIL: Versatile Interferometric Reconstruction and Gradient-based Inference Library`.
   - Replace the `## Name` section of `README.md` (and the matching paragraph in `docs/index.md`) with the draft text below.
   - Read `AGENTS.md`, `CLAUDE.md` and `.github/copilot-instructions.md` and fix any wording the mechanical rename made odd. Add one line saying the PyPI distribution is `virgil-astro`.
   - Optional: run `uv lock` to refresh the lockfile. It was already stale before the rename. Agents resolve `uv.lock` conflicts as described in Step 3.
5. Run `uv run ruff check .` and the full `uv run pytest`.
6. Open the PR and merge it with a **merge commit**, not squash or rebase, so that A0 and A keep their SHAs on `main`.
7. Tag and push the tags:
   ```bash
   git fetch origin
   git tag virgil-rename-base <A0-sha>
   git tag virgil-rename <A-sha>
   git push origin virgil-rename-base virgil-rename
   ```
8. On `main`: `python3 scripts/rename_to_virgil.py --check` must exit 0. Then tell the agents to start Step 3.

Publishing `virgil-astro` to PyPI is the human's job, not the coordinator's.

### Draft text for the `## Name` section

> ## Name
>
> Why is it called virgil?
>
> VIRGIL is the **V**ersatile **I**nterferometric **R**econstruction and **G**radient-based **I**nference **L**ibrary. In Dante's *Divine Comedy*, the poet Virgil is Dante's guide through the Inferno and Purgatory. In Virgil's own *Aeneid*, when Aeneas enters the underworld he draws his sword on the monsters crowding its threshold. His guide, the Cumaean Sibyl, warns him that they are only thin, bodiless lives flitting in a hollow semblance of form (*Aeneid* VI.292–294). Image reconstruction from sparse interferometric data is full of false visions like these: artefacts that look like structure but have no substance in the data. VIRGIL aims to help you tell the difference.

For `docs/index.md`, the same text without the heading and the "Why is it called virgil?" line.