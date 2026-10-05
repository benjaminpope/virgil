#!/usr/bin/env bash

set -eo pipefail

show_help() {
  cat <<'EOF'
Low-noise local lint runner for virgil.

Usage:
  scripts/lint_local.sh [--fix] [--no-notebooks] [--changed]

Options:
  --fix           Apply Ruff fixes/formatting changes.
  --no-notebooks  Skip notebook lint pass.
  --changed       Lint only files changed vs origin/main, including
                  uncommitted and untracked files.
  -h, --help      Show this help message.

Defaults:
  - Checks src/, tests/, examples/ and scripts/ (same scope as CI)
  - Includes the tracked notebooks/*.ipynb
  - Writes full logs to .lint-logs/
EOF
}

fix_mode=0
include_notebooks=1
changed_only=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --fix)
      fix_mode=1
      ;;
    --no-notebooks)
      include_notebooks=0
      ;;
    --changed)
      changed_only=1
      ;;
    -h|--help)
      show_help
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      show_help
      exit 2
      ;;
  esac
  shift
done

# Single source of truth for the ruff version: [tool.ruff] required-version
# in pyproject.toml (e.g. required-version = "==0.11.0").
ruff_version="$(sed -n 's/^required-version *= *"==\([^"]*\)".*/\1/p' pyproject.toml | head -n 1)"

if command -v uvx >/dev/null 2>&1 && [[ -n "$ruff_version" ]]; then
  # Preferred: run the pinned ruff in uv's tool cache. Needs no .venv, so it
  # works in fresh worktrees and never picks up a stray ruff from PATH.
  ruff_cmd=(uvx "ruff@${ruff_version}")
elif command -v uv >/dev/null 2>&1 && [[ -x ".venv/bin/python" ]]; then
  ruff_cmd=(uv run --python .venv/bin/python ruff)
elif command -v uv >/dev/null 2>&1; then
  ruff_cmd=(uv run ruff)
elif command -v ruff >/dev/null 2>&1; then
  ruff_cmd=(ruff)
else
  echo "Could not find 'ruff'. Install with 'uv pip install --python .venv/bin/python ruff' or ensure 'ruff' is on PATH." >&2
  exit 127
fi

mkdir -p .lint-logs
timestamp="$(date +"%Y%m%d_%H%M%S")"
log_file=".lint-logs/lint_${timestamp}.log"

failed=0

# Run one lint command, logging its output. A failure is recorded (and its
# log tail printed) instead of exiting, so every phase runs and the log path
# is always reported.
run_phase() {
  local label="$1"
  shift
  {
    echo
    echo "===== ${label} ====="
    echo "Command: $*"
  } >>"$log_file"
  if ! "$@" >>"$log_file" 2>&1; then
    failed=1
    echo "FAILED: ${label}" >&2
    tail -n 20 "$log_file" >&2
  fi
}

count_python=0
count_notebooks=0
py_targets=()
ipynb_targets=()

if [[ "$changed_only" -eq 1 ]]; then
  if git rev-parse --verify origin/main >/dev/null 2>&1; then
    base_ref="origin/main"
  else
    base_ref="$(git rev-list --max-parents=0 HEAD | tail -n 1)"
  fi

  # Committed changes since the merge base, plus uncommitted and untracked
  # files, so that the script is useful before committing.
  changed_files() {
    local pattern="$1"
    {
      git diff --name-only --diff-filter=ACMRTUXB "${base_ref}"...HEAD -- "$pattern"
      git diff --name-only --diff-filter=ACMRTUXB HEAD -- "$pattern"
      git ls-files --others --exclude-standard -- "$pattern"
    } | sort -u
  }

  changed_py=()
  while IFS= read -r file; do
    [[ -n "$file" && -f "$file" ]] && changed_py+=("$file")
  done < <(changed_files "*.py")

  changed_ipynb=()
  while IFS= read -r file; do
    [[ -n "$file" && -f "$file" ]] && changed_ipynb+=("$file")
  done < <(changed_files "*.ipynb")

  for file in "${changed_py[@]}"; do
    case "$file" in
      src/*|tests/*|examples/*|scripts/*)
        py_targets+=("$file")
        ;;
    esac
  done

  for file in "${changed_ipynb[@]}"; do
    if [[ "$file" == notebooks/* ]]; then
      ipynb_targets+=("$file")
    fi
  done
else
  py_targets=(src tests examples scripts)
  # Tracked notebooks only, so local scratch notebooks are skipped.
  ipynb_targets=()
  while IFS= read -r file; do
    [[ -n "$file" ]] && ipynb_targets+=("$file")
  done < <(git ls-files -- "notebooks/*.ipynb")
fi

if [[ "${#py_targets[@]}" -gt 0 ]]; then
  count_python="${#py_targets[@]}"
  if [[ "$fix_mode" -eq 1 ]]; then
    run_phase "ruff check (python, --fix)" "${ruff_cmd[@]}" check --force-exclude --fix "${py_targets[@]}"
    run_phase "ruff format (python)" "${ruff_cmd[@]}" format --force-exclude "${py_targets[@]}"
  else
    run_phase "ruff check (python)" "${ruff_cmd[@]}" check --force-exclude "${py_targets[@]}"
    run_phase "ruff format --check (python)" "${ruff_cmd[@]}" format --force-exclude --check "${py_targets[@]}"
  fi
fi

if [[ "$include_notebooks" -eq 1 && "${#ipynb_targets[@]}" -gt 0 ]]; then
  count_notebooks="${#ipynb_targets[@]}"
  if [[ "$fix_mode" -eq 1 ]]; then
    run_phase "ruff check (notebooks, --fix)" "${ruff_cmd[@]}" check --force-exclude --fix "${ipynb_targets[@]}"
  else
    run_phase "ruff check (notebooks)" "${ruff_cmd[@]}" check --force-exclude "${ipynb_targets[@]}"
  fi
fi

if [[ "$count_python" -eq 0 && ( "$include_notebooks" -eq 0 || "$count_notebooks" -eq 0 ) ]]; then
  echo "No lint targets found for requested scope."
  echo "Log: ${log_file}"
  exit 0
fi

if [[ "$failed" -ne 0 ]]; then
  echo "Lint failed; full log: ${log_file}" >&2
  exit 1
fi

echo "Lint completed successfully."
echo "Python targets: ${count_python}"
if [[ "$include_notebooks" -eq 1 ]]; then
  echo "Notebook targets: ${count_notebooks}"
else
  echo "Notebook targets: skipped"
fi
echo "Full log: ${log_file}"