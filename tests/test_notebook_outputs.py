"""The stale-notebook-output check in ``scripts/check_notebook_outputs.py``.

Pure stdlib: notebooks are built as dicts, no JAX and no kernel.
"""

import importlib.util
import json
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "check_notebook_outputs.py"
spec = importlib.util.spec_from_file_location("check_notebook_outputs", SCRIPT)
chk = importlib.util.module_from_spec(spec)
spec.loader.exec_module(chk)


def _nb(source="x = 1", outputs=True, count=1):
    out = [{"output_type": "stream", "name": "stdout", "text": ["1\n"]}] if outputs else []
    return {
        "cells": [
            {"cell_type": "markdown", "metadata": {}, "source": ["# Title"]},
            {
                "cell_type": "code",
                "execution_count": count,
                "metadata": {"tags": ["a"]},
                "outputs": out,
                "source": source.splitlines(keepends=True),
            },
        ],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def test_hash_ignores_outputs_counts_metadata_and_source_chunking():
    a = _nb()
    b = _nb(outputs=False, count=7)
    b["cells"][1]["metadata"] = {}
    b["cells"][1]["source"] = "x = 1"
    assert chk.source_hash(a) == chk.source_hash(b)


def test_hash_changes_with_source_or_cell_type():
    base = chk.source_hash(_nb())
    assert chk.source_hash(_nb("x = 2")) != base
    changed = _nb()
    changed["cells"][0]["cell_type"] = "raw"
    assert chk.source_hash(changed) != base


def test_status_cycle():
    assert chk.status(_nb(outputs=False)) == "no-outputs"
    nb = _nb()
    assert chk.status(nb) == "unstamped"
    chk.stamp(nb)
    assert chk.status(nb) == "ok"
    nb["cells"][1]["source"] = ["x = 2"]
    assert chk.status(nb) == "stale"


def test_stamp_cli_and_check_exit_codes(tmp_path, capsys):
    path = tmp_path / "nb.ipynb"
    path.write_text(json.dumps(_nb()))
    assert chk.main([str(path)]) == 0  # unstamped: warning only
    assert "no virgil.source_hash" in capsys.readouterr().out
    assert chk.main(["--stamp", str(path)]) == 0
    assert chk.main([str(path)]) == 0
    nb = json.loads(path.read_text())
    nb["cells"][1]["source"] = ["x = 3"]
    path.write_text(json.dumps(nb))
    assert chk.main([str(path)]) == 1
    assert "stale" in capsys.readouterr().out
