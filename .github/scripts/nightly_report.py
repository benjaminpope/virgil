"""Open, update or close the one failure issue of a tests.yml tier.

    RESULT=failure RUN_URL=... python3 .github/scripts/nightly_report.py DIR

DIR holds the run's JUnit reports (one ``junit-*`` artifact per job).
On failure, the failing test ids, the run and the commits since the last
green run go into the open issue labelled ``nightly-failure``, or a new
one. On success, an open issue is closed. Either way the nightly then
deletes the JAX caches of earlier weeks, keeping the store under GitHub's
10 GB. Uses the ``gh`` CLI and the standard library only.

``REPORT=main`` reports a push to main instead (label ``red-main``, the
last green push, no cache cleaning).
"""

import json
import os
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

TIERS = {
    "nightly": {
        "label": "nightly-failure",
        "event": "schedule",
        "name": "Nightly",
        "title": "Nightly CI is failing",
        "description": "The nightly CI run is failing",
    },
    "main": {
        "label": "red-main",
        "event": "push",
        "name": "Main",
        "title": "main is red",
        "description": "Tests on main are failing",
    },
}
TIER = TIERS[os.environ.get("REPORT", "nightly")]
LABEL = TIER["label"]


def gh(*args):
    return subprocess.run(
        ["gh", *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def failures(directory):
    """{job: [failing test ids]} from the JUnit reports."""
    out = {}
    for report in sorted(Path(directory).glob("*/*.xml")):
        failed = [
            f"{case.get('classname')}::{case.get('name')}"
            for case in ET.parse(report).getroot().iter("testcase")
            if case.find("failure") is not None
            or case.find("error") is not None
        ]
        if failed:
            out[report.parent.name.removeprefix("junit-")] = failed
    return out


def unsuccessful_jobs():
    """[(job name, conclusion)] of the run's finished jobs that did not pass.

    A job cancelled at its ``timeout-minutes`` has no failing test in any
    JUnit report, so the issue names it from the run itself. Jobs still
    running (this reporter) and skipped ones do not count.
    """
    run = os.environ.get("GITHUB_RUN_ID")
    if not run:
        return []
    jobs = json.loads(gh("run", "view", run, "--json", "jobs"))["jobs"]
    return [
        (job["name"], job["conclusion"])
        for job in jobs
        if job["conclusion"] not in ("success", "skipped", "", None)
    ]


def last_green():
    runs = json.loads(
        gh(
            "run",
            "list",
            "--workflow",
            "tests.yml",
            "--event",
            TIER["event"],
            "--status",
            "success",
            "--limit",
            "1",
            "--json",
            "headSha",
        )
    )
    return runs[0]["headSha"] if runs else None


def body(directory):
    repo = os.environ["GITHUB_REPOSITORY"]
    sha = os.environ["GITHUB_SHA"]
    lines = [
        f"{TIER['name']} run {os.environ['RUN_URL']} failed at {sha[:12]}.",
        "",
    ]
    found = failures(directory)
    bad = unsuccessful_jobs()
    if bad:
        lines.append("Jobs that did not pass:")
        lines += [f"- {name}: **{how}**" for name, how in bad]
        lines.append("")
    if not found:
        lines.append(
            "No failing tests in the JUnit reports: a job failed or was "
            "cancelled outside pytest (environment, timeout, packaging). "
            "See the run."
        )
    for job, ids in found.items():
        lines.append(f"**{job}** ({len(ids)})")
        lines += [f"- `{i}`" for i in ids[:30]]
        if len(ids) > 30:
            lines.append(f"- ... and {len(ids) - 30} more")
        lines.append("")
    green = last_green()
    if green and green != sha:
        lines.append(
            f"Commits since the last green run: "
            f"https://github.com/{repo}/compare/{green[:12]}...{sha[:12]}"
        )
    elif not green:
        lines.append("No green run on record yet.")
    return "\n".join(lines)


def clean_caches():
    """Delete every ``jax-`` cache not of the current ISO week."""
    week = time.strftime("%G-W%V", time.gmtime())
    caches = json.loads(
        gh(
            "cache",
            "list",
            "--key",
            "jax-",
            "--limit",
            "500",
            "--json",
            "id,key",
        )
    )
    for cache in caches:
        if not cache["key"].endswith(f"-{week}"):
            gh("cache", "delete", str(cache["id"]))


def report(directory):
    open_issue = gh(
        "issue",
        "list",
        "--label",
        LABEL,
        "--state",
        "open",
        "--json",
        "number",
        "--jq",
        ".[0].number // empty",
    )
    if os.environ["RESULT"] == "success":
        if open_issue:
            gh(
                "issue",
                "close",
                open_issue,
                "--comment",
                f"{TIER['name']} run {os.environ['RUN_URL']} is green.",
            )
        return
    text = body(directory)
    print(text)
    if open_issue:
        gh("issue", "comment", open_issue, "--body", text)
    else:
        gh(
            "label",
            "create",
            LABEL,
            "--force",
            "--color",
            "B60205",
            "--description",
            TIER["description"],
        )
        gh(
            "issue",
            "create",
            "--title",
            TIER["title"],
            "--label",
            LABEL,
            "--body",
            text,
        )


def main():
    try:
        report(sys.argv[1])
    finally:
        if TIER is TIERS["nightly"]:
            clean_caches()


if __name__ == "__main__":
    main()
