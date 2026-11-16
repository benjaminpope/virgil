"""The committed development-carbon files carry aggregates only.

``docs/dev_carbon.md`` and ``docs/generated/dev_carbon*.json`` are public and
are rewritten by a local script that reads private transcripts, ledgers and
cluster accounting. This lint fails if any of them contains an absolute home
path, a transcript file name, a session UUID, an email address, a host name
or address, a full commit hash, or a credential-shaped token. The patterns
are generic; they name no person, machine or project.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

PATTERNS: dict[str, re.Pattern[str]] = {
    "absolute home path": re.compile(
        r"(?:/Users/|/home/|/root/|/private/var/|/var/folders/|[A-Za-z]:\\{1,2}Users\\{1,2})[^\s/\\\"']+"
    ),
    "Claude project slug": re.compile(
        r"(?<![\w-])-(?:Users|home|root)-[\w.]+-"
    ),
    "home-relative path": re.compile(r"(?<![\w.])~/[\w.-]"),
    "transcript or log file": re.compile(r"\.(?:jsonl|log|sqlite3?|vscdb)\b"),
    "uuid (session id)": re.compile(
        r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
    ),
    "email address": re.compile(r"[\w.+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+"),
    "ssh target or user@host": re.compile(r"\bssh\s+[\w@.-]+|\b\w+@[\w-]+\b"),
    "private host name": re.compile(
        r"\b[\w-]+(?:\.[\w-]+)*\.(?:local|lan|internal|home|edu\.au|ac\.uk)\b"
    ),
    "ip address": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
    "full commit hash": re.compile(r"\b(?:[0-9a-f]{40}|[0-9a-f]{64})\b"),
    "token": re.compile(
        r"\b(?:gh[pousr]_\w{20,}|github_pat_\w{20,}|sk-[A-Za-z0-9_-]{20,}"
        r"|xox[abprs]-[\w-]{10,}|AKIA[0-9A-Z]{16}|eyJ[\w-]{10,}\.[\w-]{10,}\.[\w-]{5,})"
    ),
    "authorization header": re.compile(
        r"(?i)\b(?:bearer|token)\s+[A-Za-z0-9._-]{20,}"
    ),
}

# Public PR titles and the release stamp may quote a few short (7-12 hex)
# shas; a long run of them is a commit listing the page does not need.
# Float exponents such as ``1234567e-05`` are not hashes.
SHORT_SHA = re.compile(
    r"(?<![\d.\w])(?=[0-9a-f]*\d)(?=[0-9a-f]*[a-f])[0-9a-f]{7,12}(?![\w.+-])"
)
MAX_SHORT_SHAS = 8


def carbon_files() -> list[Path]:
    files = [ROOT / "docs" / "dev_carbon.md"]
    files += sorted((ROOT / "docs" / "generated").glob("dev_carbon*.json"))
    return files


def violations(text: str) -> list[str]:
    """Names of the private-data patterns found in ``text``."""
    found = [name for name, pat in PATTERNS.items() if pat.search(text)]
    if len(SHORT_SHA.findall(text)) > MAX_SHORT_SHAS:
        found.append("list of short commit hashes")
    return found


@pytest.mark.parametrize("path", carbon_files(), ids=lambda p: p.name)
def test_committed_carbon_file_is_private_data_free(path: Path) -> None:
    assert path.exists(), f"{path} is expected to be committed"
    assert violations(path.read_text(encoding="utf-8")) == []


def test_carbon_files_found() -> None:
    names = {p.name for p in carbon_files()}
    assert {
        "dev_carbon.md",
        "dev_carbon.json",
        "dev_carbon_archive.json",
    } <= names


# Synthetic fixtures: each positive case must trip the named pattern.
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ('"path": "/Users/alice/code/proj"', "absolute home path"),
        ("see /home/bob/data", "absolute home path"),
        ("C:\\Users\\carol\\x", "absolute home path"),
        (json.dumps("C:\\Users\\carol\\x"), "absolute home path"),
        ("-Users-alice-code-virgil", "Claude project slug"),
        ("-home-bob-proj", "Claude project slug"),
        ("read ~/.claude/state", "home-relative path"),
        ("session abc.jsonl", "transcript or log file"),
        ("id 123e4567-e89b-12d3-a456-426614174000", "uuid (session id)"),
        ("mail alice@example.org", "email address"),
        ("ran ssh gpunode", "ssh target or user@host"),
        ("host node01.cluster.local", "private host name"),
        ("addr 10.0.0.12", "ip address"),
        ("head " + "a1b2c3d4" * 5, "full commit hash"),
        ("key ghp_" + "A1b2C3d4E5" * 3, "token"),
        (
            "Authorization: Bearer " + "abcDEF0123456789" * 2,
            "authorization header",
        ),
        (" ".join(["abc1234"] * 9), "list of short commit hashes"),
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
        "Slurm accounting from the cluster `nt`; 1.234 kg CO2e per 100 kWh.",
        '"2026-03-04|copilot|Copilot cloud agent|cloud agent|dev|#30"',
        "version 1.2.3 and numbers 20260304 and 1234567",
    ],
)
def test_aggregates_are_not_flagged(text: str) -> None:
    assert violations(text) == []
