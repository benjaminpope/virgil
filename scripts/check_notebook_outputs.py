#!/usr/bin/env python3
"""Detect notebook outputs that are stale relative to the notebook source.

A notebook whose outputs came from an executed run carries
``metadata["virgil"]["source_hash"]``: the sha256 of its cell types and
sources (outputs, execution counts and metadata are excluded). This script
recomputes that hash for every notebook under ``notebooks/`` that has
outputs. It fails when a recorded hash mismatches, and only warns when no
hash is recorded.

    python scripts/check_notebook_outputs.py             # check all notebooks
    python scripts/check_notebook_outputs.py --stamp NB  # record the hash in NB

Run ``--stamp`` on the executed notebook when results are pulled from OzSTAR.
Only the standard library is used: a notebook is plain JSON.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _source(cell):
    src = cell.get("source", "")
    return "".join(src) if isinstance(src, list) else src


def source_hash(nb):
    """sha256 of the concatenated cell types and sources."""
    h = hashlib.sha256()
    for cell in nb.get("cells", []):
        h.update(cell.get("cell_type", "").encode())
        h.update(b"\0")
        h.update(_source(cell).encode())
        h.update(b"\0\0")
    return h.hexdigest()


def has_outputs(nb):
    return any(c.get("outputs") for c in nb.get("cells", []) if c.get("cell_type") == "code")


def recorded_hash(nb):
    return nb.get("metadata", {}).get("virgil", {}).get("source_hash")


def stamp(nb):
    nb.setdefault("metadata", {}).setdefault("virgil", {})["source_hash"] = source_hash(nb)
    return nb


def status(nb):
    """One of ``"no-outputs"``, ``"unstamped"``, ``"ok"``, ``"stale"``."""
    if not has_outputs(nb):
        return "no-outputs"
    rec = recorded_hash(nb)
    if rec is None:
        return "unstamped"
    return "ok" if rec == source_hash(nb) else "stale"


def _load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _dump(nb, path):
    # Same layout nbformat writes, so stamping produces a minimal diff.
    text = json.dumps(nb, indent=1, sort_keys=True, ensure_ascii=False) + "\n"
    Path(path).write_text(text, encoding="utf-8")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--stamp", metavar="NB", nargs="+", help="record the source hash in these notebooks")
    ap.add_argument("notebooks", nargs="*", help="notebooks to check (default: notebooks/*.ipynb)")
    args = ap.parse_args(argv)

    if args.stamp:
        for path in args.stamp:
            nb = stamp(_load(path))
            _dump(nb, path)
            print(f"stamped {path}: {recorded_hash(nb)[:12]}")
        return 0

    paths = [Path(p) for p in args.notebooks] or sorted((ROOT / "notebooks").glob("*.ipynb"))
    stale = []
    for path in paths:
        result = status(_load(path))
        if result == "stale":
            stale.append(path)
            print(f"ERROR: {path}: outputs are stale (source changed since they were produced)")
        elif result == "unstamped":
            print(f"warning: {path}: has outputs but no virgil.source_hash")
    if stale:
        print(
            "Re-execute the notebook (e.g. on OzSTAR), pull it, then run: "
            "python scripts/check_notebook_outputs.py --stamp <notebook>"
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
