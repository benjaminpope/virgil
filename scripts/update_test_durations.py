#!/usr/bin/env python3
"""Refresh the test timings CI uses to shard the suite and police its budget.

Reads one or more JUnit reports from CI (the ``virgil-evidence-*`` artifacts
of a run on main, or the nightly's ``junit-*`` artifacts) and writes

- ``tests/test_durations.json``: seconds per test file, the largest over the
  reports given, without the ``full_only`` tests (the gate and main tiers do
  not run them). ``--shard K/N`` (tests/conftest.py) packs whole files into
  shards with these, so module-scoped fixtures run once per job.
- ``tests/test_durations_full.json``: the same with the ``full_only`` tests,
  which the full and nightly tiers pack by (``VIRGIL_FULL_DURATIONS=1``).
- with ``--over-budget SECONDS``, ``tests/over_budget.txt``: the tests that
  are not marked slow and took longer than SECONDS in any report, and their
  files. JUnit times include setup, so each such test is exempt from the
  budget on its call and its file on setup (module fixtures), until they
  are shrunk or marked slow; the list should only ever get shorter.

    gh run download RUN_ID -R benjaminpope/virgil -D /tmp/junit
    python scripts/update_test_durations.py /tmp/junit/*/*.xml

Only the standard library is used.
"""

import argparse
import ast
import collections
import json
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DURATIONS = ROOT / "tests" / "test_durations.json"
DURATIONS_FULL = ROOT / "tests" / "test_durations_full.json"
OVER_BUDGET = ROOT / "tests" / "over_budget.txt"


def node_id(classname, name):
    """The pytest node id of a JUnit test case (xdist group suffix dropped)."""
    parts = classname.split(".")
    # tests.test_x.TestClass -> tests/test_x.py::TestClass
    module = next(i for i, p in enumerate(parts) if p.startswith("test_"))
    path = "/".join(parts[: module + 1]) + ".py"
    name = re.sub(r"@[^\]]*$", "", name)
    return "::".join([path, *parts[module + 1 :], name])


def read(reports):
    tests = collections.defaultdict(float)
    for report in reports:
        for case in ET.parse(report).getroot().iter("testcase"):
            skipped = case.find("skipped")
            # An xfail ran (its time counts); any other skip did not.
            if skipped is not None and skipped.get("type") != "pytest.xfail":
                continue
            nid = node_id(case.get("classname"), case.get("name"))
            tests[nid] = max(tests[nid], float(case.get("time", 0)))
    return tests


def per_file(reports, skip=frozenset()):
    """Seconds per file; ``skip`` holds ``(path, function name)`` pairs.

    A file whose tests are all skipped still gets an entry (0.0).
    """
    files = collections.Counter()
    for report in reports:
        one = collections.Counter()
        for nid, seconds in read([report]).items():
            path = nid.split("::")[0]
            name = re.sub(r"\[.*$", "", nid.split("::")[-1])
            one[path] += 0.0 if (path, name) in skip else seconds
        for path, seconds in one.items():
            files[path] = max(files[path], seconds)
    return files


def marked(marker):
    """``(path, function name)`` of the tests carrying ``@pytest.mark.<marker>``.

    Read from the source (a decorator on the function or its class, or a
    module ``pytestmark``), not by collecting: a collection run skips every
    module whose optional dependencies are missing (``importorskip``) and
    would lose their tests.
    """

    def names(node):
        for dec in getattr(node, "decorator_list", []):
            for part in ast.walk(dec):
                if isinstance(part, ast.Attribute):
                    yield part.attr

    found = set()
    for path in sorted((ROOT / "tests").glob("test_*.py")):
        tree = ast.parse(path.read_text())
        rel = f"tests/{path.name}"
        module_marks = any(
            isinstance(n, ast.Assign)
            and any(getattr(t, "id", "") == "pytestmark" for t in n.targets)
            and any(
                isinstance(a, ast.Attribute) and a.attr == marker
                for a in ast.walk(n.value)
            )
            for n in tree.body
        )

        def visit(body, inherited):
            for node in body:
                here = inherited or marker in names(node)
                if isinstance(node, ast.ClassDef):
                    visit(node.body, here)
                elif isinstance(
                    node, ast.FunctionDef
                ) and node.name.startswith("test"):
                    if here:
                        found.add((rel, node.name))

        visit(tree.body, module_marks)
    return found


def slow_tests():
    """Node ids of the slow tests (a collection run: needs the test extras)."""
    out = subprocess.run(
        [sys.executable, "-m", "pytest", "-o", "addopts=", "-m", "slow"]
        + ["--collect-only", "-q", "-p", "no:cacheprovider", "-W", "ignore"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    ).stdout
    return {line for line in out.splitlines() if "::" in line}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument("--over-budget", type=float, metavar="SECONDS")
    args = parser.parse_args()

    for path, files in (
        (DURATIONS, per_file(args.reports, marked("full_only"))),
        (DURATIONS_FULL, per_file(args.reports)),
    ):
        path.write_text(
            json.dumps(
                {k: round(v, 1) for k, v in sorted(files.items())}, indent=1
            )
            + "\n"
        )
        print(f"{len(files)} files -> {path.relative_to(ROOT)}")

    if args.over_budget is not None:
        slow = slow_tests()
        over = sorted(
            nid
            for nid, seconds in read(args.reports).items()
            if seconds > args.over_budget and nid not in slow
        )
        header = (
            "# Tests over the per-test budget (tests/conftest.py), exempt "
            "until they are\n# shrunk or marked slow. Remove lines; never "
            "add them. Written by\n# scripts/update_test_durations.py "
            f"--over-budget {args.over_budget:g}.\n"
        )
        files = sorted({nid.split("::")[0] for nid in over})
        OVER_BUDGET.write_text(
            header
            + "".join(n + "\n" for n in over)
            + "# Files whose setup (module fixtures) is exempt.\n"
            + "".join(f + "\n" for f in files)
        )
        print(f"{len(over)} tests -> {OVER_BUDGET.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
