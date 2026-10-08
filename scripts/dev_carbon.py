#!/usr/bin/env python3
"""Refresh the development-carbon page, its archive and the README badge.

A thin wrapper around ``carbon_report.py`` from
claude-code-carbon-dashboard (https://github.com/benjaminpope/
claude-code-carbon-dashboard), found in ``$CLAUDE_CARBON_HOME`` (default
``~/.claude/carbon``). It runs locally only: it reads Claude Code
transcripts, VS Code chat logs, ``gh`` and ``ssh nt``, none of which CI has.

``carbon_report.py`` collects every source into its private, per-record
ledger (``~/.claude/carbon/ledger-virgil.json``) and compacts the records
whose source has expired, or is about to, into
``docs/generated/dev_carbon_archive.json``: daily totals by source, model
or workflow, class and feature, with no PR titles and no names of
data-analysis work. That archive is the permanent copy in git; live
records are regenerated from their sources on each run. The report
combines both into ``docs/generated/dev_carbon.json`` and
``docs/dev_carbon.md``. This script then rewrites the README badge between
``<!-- dev-carbon-badge -->`` markers and runs
``scripts/sync_tutorial_docs.py`` for ``docs/index.md``.

Usage::

    python3 scripts/dev_carbon.py             # fetch, archive, write
    python3 scripts/dev_carbon.py --offline   # from the private ledger only
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "scripts" / "dev_carbon.toml"
ARCHIVE = ROOT / "docs" / "generated" / "dev_carbon_archive.json"
SUMMARY = ROOT / "docs" / "generated" / "dev_carbon.json"
PAGE = ROOT / "docs" / "dev_carbon.md"
README = ROOT / "README.md"
PAGE_URL = "https://benjaminpope.github.io/virgil/dev_carbon/"
BADGE = re.compile(
    r"<!-- dev-carbon-badge -->.*?<!-- /dev-carbon-badge -->", re.S
)


def carbon_home():
    home = Path(
        os.environ.get("CLAUDE_CARBON_HOME", Path.home() / ".claude/carbon")
    ).expanduser()
    if not (home / "carbon_report.py").exists():
        sys.exit(
            f"carbon_report.py not found in {home}: install "
            "claude-code-carbon-dashboard (./install.sh) or set "
            "CLAUDE_CARBON_HOME"
        )
    return home


def science_section(summary):
    sci = summary["by_class"].get("science")
    if not sci:
        return ""
    kwh, kg = sci["kwh"], sci["kg"]
    return "\n".join(
        [
            "## Excluded data analysis",
            "",
            "Compute for science with virgil (fits to observations, data "
            "reduction and archive downloads), and anything whose purpose "
            "was unclear, is costed the same way but kept out of the total. "
            "It is shown only in aggregate.",
            "",
            "| Item | Records | kWh | kg CO₂e | kg range |",
            "|---|---:|---:|---:|---:|",
            f"| Excluded data analysis | {sci['n']:,} | {kwh[1]:.3g} "
            f"| {kg[1]:.3g} | {kg[0]:.3g}–{kg[2]:.3g} |",
            "",
            "",
        ]
    )


def caveats(summary):
    cal = summary.get("copilot_calibration") or {}
    per = cal.get("credits_per_mtok", 0)
    return "\n".join(
        [
            "## Scope and caveats",
            "",
            "- **What is counted.** Development and validation of virgil: "
            "AI coding assistants, GitHub Actions CI, and jobs on Swinburne's "
            "OzSTAR/NT cluster. Validation jobs (simulation-based "
            "calibration, the imaging contests, detection false-alarm and "
            "orbit checks) are counted. Data analysis is not (see above).",
            "- **Claude Code** was used from 2026-09-28, and is counted "
            "from its local transcripts on one machine. Sessions run in the "
            "cloud (claude.ai, cloud agents) or on other machines are not "
            "stored locally and are not captured.",
            "- **Before that, VS Code Copilot Chat** was the main assistant "
            "(from March 2026). Its logs give prompt and output tokens per "
            "request but no cache split, so prompts are costed as uncached "
            "input, which errs high.",
            "- **GPT and other non-Claude models** have no published "
            "per-token energy. Each is assigned an assumed Claude size "
            "class (e.g. GPT-5.6 Sol and Terra and GPT-5.5 as Opus class, "
            "GPT-5.6 Luna and GPT-5.3-Codex as Sonnet class), with a range "
            "across the neighbouring classes.",
            "- **Copilot cloud agent and code review** expose no token "
            "counts. Their billed AI Credits are converted to tokens at the "
            f"rate measured on the local chat logs ({per:,.0f} credits per "
            "million tokens), with a factor-of-two range; months without "
            "credits use run time times a token rate.",
            "- **Data centres and grids.** Model inference uses "
            "TokenClimate's PUE (1.14) and grid factor; GitHub runners "
            "assume a hyperscale PUE of 1.18 and the US average grid; NT "
            "assumes Green Algorithms' default PUE of 1.67 and the "
            "Victorian grid. None of these is published for the specific "
            "facilities.",
            "- **Cache reads dominate the uncertainty** of the Claude "
            "figure. They are almost all of its tokens, costed at 0.08 times "
            "the input energy (range 0.05–0.20); at 1.0 the Claude figure "
            "would be several times larger.",
            "- Model training, local hardware and the embodied carbon of "
            "the cluster are not included.",
            "",
            "",
        ]
    )


def page(fragment, summary):
    fragment = re.sub(
        r"## Data analysis runs \(not in the total\)\n.*?(?=\n## )",
        "",
        fragment,
        flags=re.S,
    )
    head, sep, tail = fragment.partition("## Methodology")
    return (
        "<!-- AUTO-GENERATED by scripts/dev_carbon.py from "
        "scripts/dev_carbon.toml. Do not edit by hand. -->\n"
        "# Development carbon\n\n"
        "An estimate of the energy and carbon emitted in developing and "
        "validating virgil: AI coding assistants, continuous integration "
        "and cluster jobs. It is refreshed by hand with "
        "`python3 scripts/dev_carbon.py`, so it lags the repository.\n\n"
        + head.rstrip()
        + "\n\n"
        + science_section(summary)
        + caveats(summary)
        + sep
        + tail
    )


def write_badge(summary):
    # Markers go on their own lines: a line starting with an HTML comment is
    # raw HTML to Markdown, so the badge on that line would render as text.
    badge = (
        "<!-- dev-carbon-badge -->\n"
        f"[![{summary['badge']['text']}]({summary['badge']['url']})]"
        f"({PAGE_URL})\n"
        "<!-- /dev-carbon-badge -->"
    )
    text = README.read_text("utf-8")
    if BADGE.search(text):
        text = BADGE.sub(lambda _: badge, text)
    else:
        lines = text.splitlines(keepends=True)
        last = max(i for i, s in enumerate(lines[:12]) if s.startswith("[!["))
        lines.insert(last + 1, badge + "\n")
        text = "".join(lines)
    README.write_text(text, "utf-8")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--offline",
        action="store_true",
        help="skip fetching; report from the private ledger and the archive",
    )
    ap.add_argument(
        "--sources", help="comma-separated subset of sources to fetch"
    )
    args = ap.parse_args(argv)

    home = carbon_home()
    cmd = [sys.executable, str(home / "carbon_report.py")]
    cmd += ["--config", str(CONFIG), "--archive", str(ARCHIVE)]
    if args.offline:
        cmd.append("--offline")
    if args.sources:
        cmd += ["--sources", args.sources]
    with tempfile.TemporaryDirectory() as tmp:
        fragment = Path(tmp) / "page.md"
        cmd += ["--markdown", str(fragment), "--summary", str(SUMMARY)]
        subprocess.run(cmd, check=True)
        summary = json.loads(SUMMARY.read_text())
        PAGE.write_text(page(fragment.read_text(), summary), "utf-8")
    write_badge(summary)
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "sync_tutorial_docs.py")],
        check=True,
    )
    print(summary["badge"]["text"])


if __name__ == "__main__":
    main()
