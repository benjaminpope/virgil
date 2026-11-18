"""The committed development-carbon files carry aggregates only.

``docs/dev_carbon.md`` and ``docs/generated/dev_carbon*.json`` are public and
are rewritten by a local script that reads private transcripts, ledgers and
cluster accounting. This lint fails if any of them contains an absolute home
or cluster path, a transcript, database or job-log file name, a session id, an
email address, a host name or address, a long hex digest, or a credential-shaped
token, or if a JSON file holds raw per-record tables. The patterns are generic;
they name no person, machine or project.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

URL = re.compile(r"https?://\S+")
_HEX = r"(?=[0-9a-fA-F]*[a-fA-F])(?=[0-9a-fA-F]*\d)[0-9a-fA-F]"
_NOT_SSH_WORD = (
    r"(?!(?:to|into|in|the|a|an|and|or|for|from|key|keys|access|agent|config"
    r"|connection|via|with|on|is|as)\b)"
)

# Applied to text with URLs removed (public links are fine).
PATTERNS: dict[str, re.Pattern[str]] = {
    "absolute home path": re.compile(
        r"(?:/Users/|/home/|/root/|/private/var/|/var/folders/|/fred/|/scratch/"
        r"|/g/data/|[A-Za-z]:\\{1,2}Users\\{1,2})[^\s/\\\"']+|file://"
    ),
    "Claude project slug": re.compile(
        r"(?<![\w-])(?:[A-Za-z]-)?-(?:Users|home|root)-[\w.]+"
    ),
    "home-relative path": re.compile(r"(?<![\w.])~/[\w.-]"),
    "transcript, database or job log": re.compile(
        r"\.(?:jsonl|sqlite3?|vscdb|db)\b|\bsubmissions\.tsv\b"
        r"|[-_.]\d+(?:_\d+)?\.(?:out|err)\b"
        r"|(?:[/\\][\w.-]*|\w*[_\d-])\.log\b"
    ),
    "uuid (session id)": re.compile(
        r"(?<![0-9a-fA-F-])[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}"
        r"-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}(?![0-9a-fA-F-])"
    ),
    "email address": re.compile(
        r"[\w.+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}\b"
    ),
    "ssh target or user@host": re.compile(
        rf"\bssh\s+{_NOT_SSH_WORD}[\w@.-]+"
        r"|\b\w+@(?!v?\d|(?:main|master|latest|stable|nightly|dev)\b)[A-Za-z][\w-]{2,}\b"
    ),
    "private host name": re.compile(
        r"\b[\w-]+(?:\.[\w-]+)*\.(?:local|lan|internal|home|edu\.au|ac\.uk)\b"
    ),
    "ip address": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
    # 16+ hex digits with a letter and a digit: commit, workspace or session digests.
    "long hex id": re.compile(rf"(?<![\w.-]){_HEX}{{16,}}(?![\w.+-])"),
    "token": re.compile(
        r"\b(?:gh[pousr]_\w{20,}|github_pat_\w{20,}|sk-[A-Za-z0-9_-]{20,}"
        r"|xox[abprs]-[\w-]{10,}|AKIA[0-9A-Z]{16}|AIza[0-9A-Za-z_-]{35}"
        r"|ya29\.[0-9A-Za-z_-]{20,}|eyJ[\w-]{10,}\.[\w-]{10,}\.[\w-]{5,})"
    ),
    "authorization header": re.compile(
        r"(?i)\b(?:bearer|token)\s+[A-Za-z0-9._-]{20,}"
    ),
}

# Public PR titles and the release stamp may quote a few short (7-12 hex)
# shas; a run of them on one line is a commit listing the page does not need.
# Float exponents such as ``1234567e-05`` are not hashes.
SHORT_SHA = re.compile(rf"(?<![\w.]){_HEX}{{7,12}}(?![\w.+-])")
MAX_SHORT_SHAS_PER_LINE = 3

# Keys of raw per-record tables; the committed files hold aggregates only.
RECORD_KEYS = frozenset(
    "sources sessions records items sid session cwd project_dir folders workspace"
    " workspace_uris conversation response requestId jobid master sbatch log"
    " log_dir branch sha".split()
)


def carbon_files() -> list[Path]:
    files = [ROOT / "docs" / "dev_carbon.md"]
    files += sorted((ROOT / "docs" / "generated").glob("dev_carbon*.json"))
    return files


def violations(
    text: str, max_short: int = MAX_SHORT_SHAS_PER_LINE
) -> list[str]:
    """Names of the private-data patterns found in ``text``."""
    text = URL.sub("", text)
    found = [name for name, pat in PATTERNS.items() if pat.search(text)]
    if any(
        len(SHORT_SHA.findall(line)) > max_short for line in text.splitlines()
    ):
        found.append("run of short commit hashes")
    return found


def record_keys(obj: object) -> set[str]:
    """Forbidden per-record keys anywhere in parsed JSON."""
    if isinstance(obj, dict):
        found = {k for k in obj if k in RECORD_KEYS}
        for v in obj.values():
            found |= record_keys(v)
        return found
    if isinstance(obj, list):
        return set().union(*(record_keys(v) for v in obj)) if obj else set()
    return set()


def structure_violations(data: dict) -> list[str]:
    found = [f"raw record key {k!r}" for k in sorted(record_keys(data))]
    # Science items are shown only as anonymised "<source>|excluded" rows.
    found += [
        f"science key {k!r}"
        for k in data.get("science", {})
        if not str(k).endswith("|excluded")
    ]
    return found


@pytest.mark.parametrize("path", carbon_files(), ids=lambda p: p.name)
def test_committed_carbon_file_is_private_data_free(path: Path) -> None:
    assert path.exists(), f"{path} is expected to be committed"
    text = path.read_text(encoding="utf-8")
    # The archive strips PR titles, so it never quotes a short hash.
    max_short = (
        0
        if path.name == "dev_carbon_archive.json"
        else MAX_SHORT_SHAS_PER_LINE
    )
    assert violations(text, max_short) == []
    if path.suffix == ".json":
        assert structure_violations(json.loads(text)) == []


def test_carbon_files_found() -> None:
    names = {p.name for p in carbon_files()}
    assert {
        "dev_carbon.md",
        "dev_carbon.json",
        "dev_carbon_archive.json",
        "dev_carbon_badge.json",
    } <= names


# Synthetic fixtures: each positive case must trip the named pattern.
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ('"path": "/Users/alice/code/proj"', "absolute home path"),
        ("see /home/bob/data", "absolute home path"),
        ("C:\\Users\\carol\\x", "absolute home path"),
        (json.dumps("C:\\Users\\carol\\x"), "absolute home path"),
        ("/fred/proj01/alice/jobs", "absolute home path"),
        ("/g/data/ab12/x", "absolute home path"),
        ("file:///workspace/thing", "absolute home path"),
        ("-Users-alice-code-virgil", "Claude project slug"),
        ("-home-bob", "Claude project slug"),
        ("C--Users-carol-proj", "Claude project slug"),
        ("read ~/.claude/state", "home-relative path"),
        ("session abc.jsonl", "transcript, database or job log"),
        ("conversations/summary.db", "transcript, database or job log"),
        ("submissions.tsv", "transcript, database or job log"),
        ("target_fit_12345.out", "transcript, database or job log"),
        ("logs/run.log", "transcript, database or job log"),
        ("id 123e4567-e89b-12d3-a456-426614174000", "uuid (session id)"),
        (
            '"request_123e4567-e89b-12d3-a456-426614174000"',
            "uuid (session id)",
        ),
        ("mail alice@example.org", "email address"),
        ("ran ssh gpunode", "ssh target or user@host"),
        ("login alice@gpunode", "ssh target or user@host"),
        ("host node01.cluster.local", "private host name"),
        ("addr 10.0.0.12", "ip address"),
        ("head " + "a1b2c3d4" * 5, "long hex id"),
        ("sid a1b2c3d4e5f60718", "long hex id"),
        ("ws " + "0123456789ABCDEF" * 2, "long hex id"),
        ("key ghp_" + "A1b2C3d4E5" * 3, "token"),
        ("key AIza" + "A1b2C3d4E5" * 3 + "abcde", "token"),
        ("tok ya29." + "A1b2C3d4E5" * 3, "token"),
        (
            "Authorization: Bearer " + "abcDEF0123456789" * 2,
            "authorization header",
        ),
        (" ".join(["abc1234"] * 4), "run of short commit hashes"),
    ],
)
def test_private_data_is_flagged(text: str, expected: str) -> None:
    assert expected in violations(text)


@pytest.mark.parametrize(
    "text",
    [
        '{"generated": "2026-10-30T12:54+11:00", "total": {"kg": [1.5, 2.5, 3.5]}}',
        "Refreshed for v0.4.2 on 2026-11-20; head 1a2b3c4.",
        "Source: https://github.com/example/project/pull/12 (public repository).",
        "See https://www.example.edu.au/~someone/paper.pdf for the method.",
        "Slurm accounting from the cluster `nt`; 1.234 kg CO2e per 100 kWh.",
        '"2026-03-04|copilot|Copilot cloud agent|cloud agent|dev|#30"',
        "version 1.2.3 and numbers 20260304 and 1234567",
        "CI: pin uvx ruff@0.11.0 and actions/checkout@v4",
        "Docs: forbid ssh to the cluster from agents",
        "Use np.log and jnp.log1p in the likelihood",
        "[1.23456789012345e-05, 2.5e+03]",
        "#285 Orbit tutorial: outputs from rerun at 33e1390 (and 5ca0bc9)",
    ],
)
def test_aggregates_are_not_flagged(text: str) -> None:
    assert violations(text) == []


def test_raw_record_tables_are_flagged() -> None:
    raw = {"sessions": [{"sid": "x", "calls": 1}], "17938486": {"jobid": 1}}
    assert record_keys(raw) == {"sessions", "sid", "jobid"}
    assert structure_violations({"science": {"Secret job": {}}}) == [
        "science key 'Secret job'"
    ]
    assert (
        structure_violations({"science": {"Claude Code|excluded": {}}}) == []
    )
