"""Refuse a release (or a version bump) whose carbon page was not refreshed.

Release (publish.yml gate)::

    GITHUB_REF_NAME=v0.5.0 RELEASE_BODY=... python3 .github/scripts/check_carbon_fresh.py

Passes when ``docs/generated/dev_carbon.json`` carries the ``release`` stamp
written by ``scripts/dev_carbon.py --release``, for the tagged version, made
at most ``CARBON_MAX_AGE_DAYS`` (default 7) before the tagged commit. A line
``carbon-gate: skip <reason>`` in the release notes passes with a warning.

PR mode (lint.yml)::

    PR_BODY=... python3 .github/scripts/check_carbon_fresh.py --pr origin/main

Passes unless the PR changes ``project.version`` without a matching stamp
(the same skip line in the PR description bypasses it).
The standard library only.
"""

import json
import os
import re
import subprocess
import sys
import tomllib
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SUMMARY = ROOT / "docs" / "generated" / "dev_carbon.json"
SKIP = re.compile(r"^\s*carbon-gate:\s*skip\b", re.M | re.I)


def problems(stamp, version, commit_time, max_age_days=7):
    """Why ``stamp`` does not clear the gate for ``version``; [] if it does.

    ``commit_time`` is the tagged commit's date; ``None`` skips the age check
    (pull requests).
    """
    if not stamp:
        return [
            "no release stamp in docs/generated/dev_carbon.json: run "
            "`python3 scripts/dev_carbon.py --release` in the release PR"
        ]
    out = []
    if stamp.get("version") != version:
        out.append(
            f"carbon page was refreshed for v{stamp.get('version')}, "
            f"not v{version}"
        )
    if commit_time is not None:
        age = commit_time - datetime.fromisoformat(stamp["generated"])
        if age > timedelta(days=max_age_days):
            out.append(
                f"carbon refresh is {age.days} days older than the tagged "
                f"commit (limit {max_age_days})"
            )
    return out


def skipped(body):
    return bool(SKIP.search(body or ""))


def version_of(text):
    return tomllib.loads(text)["project"]["version"]


def git(*args):
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def main(argv):
    stamp = json.loads(SUMMARY.read_text("utf-8")).get("release")
    max_age = int(os.environ.get("CARBON_MAX_AGE_DAYS", "7"))
    if argv[:1] == ["--pr"]:
        new = version_of((ROOT / "pyproject.toml").read_text("utf-8"))
        if new == version_of(git("show", f"{argv[1]}:pyproject.toml")):
            print("version unchanged: carbon page not required")
            return 0
        found, body, label = problems(stamp, new, None), "PR_BODY", "PR"
    else:
        tag = os.environ["GITHUB_REF_NAME"]
        commit = datetime.fromisoformat(
            git("show", "-s", "--format=%cI", "HEAD")
        )
        found = problems(stamp, tag.removeprefix("v"), commit, max_age)
        body, label = "RELEASE_BODY", "release notes"
    if found and skipped(os.environ.get(body)):
        print(
            f"::warning::carbon gate skipped by the {label}: {'; '.join(found)}"
        )
        return 0
    for p in found:
        print(f"::error::{p}")
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
