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
``<!-- dev-carbon-badge -->`` markers (an endpoint badge reading
``docs/generated/dev_carbon_badge.json``, which the docs site serves), adds everyday equivalents from the
tool's ``token_carbon.COMPARISONS`` and runs
``scripts/sync_tutorial_docs.py`` for ``docs/index.md``.

Usage::

    python3 scripts/dev_carbon.py             # fetch, archive, write
    python3 scripts/dev_carbon.py --offline   # from the private ledger only
"""

import argparse
import json
import math
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "scripts" / "dev_carbon.toml"
ARCHIVE = ROOT / "docs" / "generated" / "dev_carbon_archive.json"
SUMMARY = ROOT / "docs" / "generated" / "dev_carbon.json"
PAGE = ROOT / "docs" / "dev_carbon.md"
BADGE_JSON = ROOT / "docs" / "generated" / "dev_carbon_badge.json"
BADGE_JSON_URL = (
    "https://benjaminpope.github.io/virgil/generated/dev_carbon_badge.json"
)
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


def comparisons_section(summary, home):
    """The total against the nearest everyday items in the tool's table.

    The three items nearest the mid total on a log scale, in order of
    size, each as a count with the range from the low and high totals.
    """
    sys.path.insert(0, str(home))
    import token_carbon as tc

    items = sorted(tc.COMPARISONS, key=lambda c: c["kg"])
    lo, mid, hi = summary["total"]["kg"]
    near = sorted(
        range(len(items)),
        key=lambda j: abs(math.log(mid / items[j]["kg"])),
    )[:3]
    rows = [
        f"| {items[j]['one'][0].upper()}{items[j]['one'][1:]} "
        f"| {items[j]['kg']:,g} "
        f"| {tc.format_count(mid / items[j]['kg'])} "
        f"| {tc.format_count(lo / items[j]['kg'])}–"
        f"{tc.format_count(hi / items[j]['kg'])} |"
        for j in sorted(near)
    ]
    return "\n".join(
        [
            "## In everyday terms",
            "",
            f"The total of {mid:.3g} kg CO₂e set against the three nearest "
            "items, on a log scale, in the "
            "[ALPLA CO₂ Comparison Tool]"
            "(https://www.alpla.com/en/sustainability/co2-comparison-tool): "
            "the table the carbon tool uses for its everyday equivalents. "
            "The per-item figures are ALPLA's own, read from the calculator "
            "on 2026-10-01 (ALPLA is a plastic packaging maker), and have "
            "not been checked independently. The range follows the low and "
            "high totals.",
            "",
            "| Item | kg CO₂e each | Equivalent to | Range |",
            "|---|---:|---:|---:|",
            *rows,
            "",
            "",
        ]
    )


EXCLUDED_ROWS = {
    "science": "Excluded data analysis",
    "other": "Gemini work on other projects",
    "unattributed": "Gemini work with no project",
}


def science_section(summary):
    rows = [
        f"| {label} | {g['n']:,} | {g['kwh'][1]:.3g} | {g['kg'][1]:.3g} "
        f"| {g['kg'][0]:.3g}–{g['kg'][2]:.3g} |"
        for cls, label in EXCLUDED_ROWS.items()
        if (g := summary["by_class"].get(cls))
    ]
    if not rows:
        return ""
    return "\n".join(
        [
            "## Excluded data analysis",
            "",
            "Compute for science with virgil (fits to observations, data "
            "reduction and archive downloads), and anything whose purpose "
            "was unclear, is costed the same way but kept out of the total. "
            "It is shown only in aggregate. So are Gemini conversations "
            "about other projects (the website, cluster scripts, tool "
            "set-up) and the few that name no project.",
            "",
            "| Item | Records | kWh | kg CO₂e | kg range |",
            "|---|---:|---:|---:|---:|",
            *rows,
            "",
            "",
        ]
    )


GEMINI_CAVEAT = (
    "- **Gemini (Google Antigravity)** has been used since 2026-10-27 for "
    "pull-request reviews and code audits (Gemini 4 Argon, with Gemini Flash "
    "for some calls). Token counts per model call come from Antigravity's "
    "local conversation databases. Reviews and audits of virgil and "
    "virgil-validation count; each conversation is attributed by the "
    "repository its review or audit names, else its workspace. Google "
    "publishes no per-token energy, so Argon is assumed Opus class and Flash "
    "Haiku class, with a range across the neighbouring classes. Google's "
    "measured 0.24 Wh for the median Gemini Apps text prompt (Elsworth et al. "
    "2025) is far below the per-call figure here, as review calls carry "
    "contexts of 10⁴–10⁵ tokens."
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
            "from its local transcripts on one machine: sessions in the "
            "virgil repository and its worktrees, but not in the separate "
            "virgil-validation and virgil-vlti repositories. Sessions run in "
            "the cloud (claude.ai, cloud agents) or on other machines are not "
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
            *(
                [GEMINI_CAVEAT]
                if "gemini" in summary.get("by_source", {})
                else []
            ),
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


def page(fragment, summary, home):
    fragment = re.sub(
        r"## Data analysis runs \(not in the total\)\n.*?(?=\n## )",
        "",
        fragment,
        flags=re.S,
    )
    head, sep, tail = fragment.partition("## Methodology")
    lead, by, rest = head.partition("## By source")
    head = lead + comparisons_section(summary, home) + by + rest
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
    """Write the shields.io endpoint JSON and point the README badge at it.

    The README is frozen into each PyPI release, so the badge is an endpoint
    badge: it reads the JSON the docs site serves, which is always current.
    """
    kg = summary["total"]["kg"][1]
    BADGE_JSON.write_text(
        json.dumps(
            {
                "schemaVersion": 1,
                "label": "dev carbon",
                "message": f"{kg:.0f} kg CO₂e",
                "color": "2e7d32",
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        "utf-8",
    )
    url = "https://img.shields.io/endpoint?url=" + quote(
        BADGE_JSON_URL, safe=""
    )
    # Markers go on their own lines: a line starting with an HTML comment is
    # raw HTML to Markdown, so the badge on that line would render as text.
    badge = (
        "<!-- dev-carbon-badge -->\n"
        f"[![{summary['badge']['text']}]({url})]"
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
        PAGE.write_text(page(fragment.read_text(), summary, home), "utf-8")
    write_badge(summary)
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "sync_tutorial_docs.py")],
        check=True,
    )
    print(summary["badge"]["text"])


if __name__ == "__main__":
    main()
