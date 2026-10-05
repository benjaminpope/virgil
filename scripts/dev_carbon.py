#!/usr/bin/env python3
"""Refresh the development-carbon page, its ledger and the README badge.

A thin wrapper around ``carbon_report.py`` from
claude-code-carbon-dashboard (https://github.com/benjaminpope/
claude-code-carbon-dashboard), found in ``$CLAUDE_CARBON_HOME`` (default
``~/.claude/carbon``). It runs locally only: it reads Claude Code
transcripts, VS Code chat logs, ``gh`` and ``ssh nt``, none of which CI has.

1. Collect every source into the private ledger
   (``~/.claude/carbon/ledger-virgil.json``), which keeps full detail.
2. Redact that ledger and max-merge it into the committed copy,
   ``docs/generated/dev_carbon_ledger.json``: home directories become
   ``~``, log paths and job variables are dropped, and data-analysis
   records lose their names, branches and folders, so no science target
   appears in the repository.
3. Cost the committed ledger (offline) into
   ``docs/generated/dev_carbon.json`` and ``docs/dev_carbon.md``.
4. Rewrite the README badge between ``<!-- dev-carbon-badge -->`` markers
   and run ``scripts/sync_tutorial_docs.py`` for ``docs/index.md``.

Usage::

    python3 scripts/dev_carbon.py             # fetch, redact, write
    python3 scripts/dev_carbon.py --offline   # rewrite from the committed ledger
"""

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "scripts" / "dev_carbon.toml"
LEDGER = ROOT / "docs" / "generated" / "dev_carbon_ledger.json"
SUMMARY = ROOT / "docs" / "generated" / "dev_carbon.json"
PAGE = ROOT / "docs" / "dev_carbon.md"
README = ROOT / "README.md"
PAGE_URL = "https://benjaminpope.github.io/virgil/dev_carbon/"
BADGE = re.compile(
    r"<!-- dev-carbon-badge -->.*?<!-- /dev-carbon-badge -->", re.S
)
SCIENCE = "data-analysis"
HOME = str(Path.home())


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


def report(home, *args):
    cmd = [sys.executable, str(home / "carbon_report.py")]
    cmd += ["--config", str(CONFIG), *args]
    subprocess.run(cmd, check=True)


def tag(text):
    """A stable, opaque stand-in for a redacted name."""
    digest = hashlib.sha1(text.encode()).hexdigest()[:8]
    return f"{SCIENCE}-{digest}"


def tilde(path):
    path = path or ""
    return "~" + path[len(HOME) :] if path.startswith(HOME) else path


def redact(sources, cfg, cc):
    """A copy of the private ledger's sources that is safe to commit."""
    classes = cfg.get("classes", {})
    slug = cfg["repo"]
    out = {}

    claude = {}
    for r in sources.get("claude", {}).values():
        r = {k: v for k, v in r.items() if k not in ("project", "project_dir")}
        r["cwd"] = tilde(r.get("cwd"))
        science = (
            cc.classify(classes, branch=r.get("branch"), dir=r["cwd"])
            == "science"
        )
        if science:
            r["branch"] = tag(r.get("branch") or "?")
            r["cwd"] = SCIENCE
        key = f"{r['session']}|{r['day']}|{r['model']}|{r.get('branch', '?')}"
        claude[key] = r
    out["claude"] = claude

    vscode = {}
    for rid, r in sources.get("vscode_copilot", {}).items():
        r = dict(r, folders=[tilde(f) for f in r.get("folders", [])])
        if any(cc.classify(classes, dir=f) == "science" for f in r["folders"]):
            r["folders"] = [SCIENCE]
        vscode[rid] = r
    out["vscode_copilot"] = vscode

    slurm = {}
    drop = ("log", "ledger", "vars", "remote")
    for jid, r in sources.get("slurm", {}).items():
        r = {k: v for k, v in r.items() if k not in drop}
        if r.get("log_dir"):
            r["log_dir"] = "/".join(Path(r["log_dir"]).parts[-2:])
        science = (
            cc.classify(classes, job=r.get("name"), dir=r.get("log_dir"))
            == "science"
        )
        if science:
            r["name"] = SCIENCE
            r["log_dir"] = SCIENCE
            for k in ("sbatch", "lib", "sha", "array", "submitted"):
                r.pop(k, None)
        slurm[jid] = r
    out["slurm"] = slurm

    def science_branch(branch):
        return cc.classify(classes, branch=branch) == "science"

    for name in ("gha", "copilot_runs"):
        runs = {}
        for k, r in sources.get(name, {}).items():
            r = {f: v for f, v in r.items() if f != "title"}
            if science_branch(r.get("head_branch")):
                r["head_branch"] = tag(r["head_branch"])
            runs[k] = r
        out[name] = runs
    prs = {}
    for k, r in sources.get("prs", {}).items():
        if science_branch(r.get("branch")):
            r = dict(r, branch=tag(r["branch"]), title="data analysis")
        prs[k] = r
    out["prs"] = prs

    # Account-wide Copilot records: keep only this repository by name.
    repo = slug.split("/")[1]
    agent = {}
    for k, r in sources.get("copilot_agent_prs", {}).items():
        r = {f: v for f, v in r.items() if f != "title"}
        if r.get("repo") != slug:
            r = {"repo": "other", "number": 0, "month": r.get("month", "")}
            k = f"other#{hashlib.sha1(k.encode()).hexdigest()[:8]}"
        agent[k] = r
    out["copilot_agent_prs"] = agent
    billing = {}
    for k, r in sources.get("copilot_billing", {}).items():
        if r.get("repo") not in ("", repo):
            r = dict(r, repo="other")
            k = f"{r['day']}|{r['sku']}|other"
        billing[k] = r
    out["copilot_billing"] = billing

    for name in ("copilot_premium", "vscode_months", "sha_prs"):
        out[name] = sources.get(name, {})
    return out


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
    badge = (
        "<!-- dev-carbon-badge -->"
        f"[![{summary['badge']['text']}]({summary['badge']['url']})]"
        f"({PAGE_URL})<!-- /dev-carbon-badge -->"
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
        help="skip fetching; rewrite the outputs from the committed ledger",
    )
    ap.add_argument(
        "--sources", help="comma-separated subset of sources to fetch"
    )
    args = ap.parse_args(argv)

    home = carbon_home()
    sys.path.insert(0, str(home))
    import carbon_common as cc
    import carbon_report as cr

    cfg = cr.load_config(CONFIG)
    if not args.offline:
        private = Path(
            cfg.get("ledger") or home / "ledger-virgil.json"
        ).expanduser()
        with tempfile.TemporaryDirectory() as tmp:
            extra = ["--sources", args.sources] if args.sources else []
            # A full (non-offline) report also looks up the PRs of pinned
            # commits and stores them in the ledger.
            report(
                home,
                "--ledger",
                str(private),
                *extra,
                "--summary",
                str(Path(tmp) / "private.json"),
            )
        sources = cc.load_ledger(private)["sources"]
        cc.update_ledger(LEDGER, redact(sources, cfg, cc))
        LEDGER.with_suffix(".lock").unlink(missing_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        fragment = Path(tmp) / "page.md"
        report(
            home,
            "--ledger",
            str(LEDGER),
            "--offline",
            "--markdown",
            str(fragment),
            "--summary",
            str(SUMMARY),
        )
        LEDGER.with_suffix(".lock").unlink(missing_ok=True)
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
