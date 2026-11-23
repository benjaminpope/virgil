"""Every test file has a timing in tests/test_durations.json and
tests/test_durations_full.json, and neither names a deleted file.

A file without one is packed as the mean file, which unbalances the shards
(tests/conftest.py). Refresh with scripts/update_test_durations.py.
"""

import ast
import importlib.util
import json
from pathlib import Path

import pytest

HERE = Path(__file__).parent


@pytest.mark.parametrize(
    "name", ["test_durations.json", "test_durations_full.json"]
)
def test_every_test_file_has_a_recorded_duration(name):
    known = json.loads((HERE / name).read_text())
    files = {f"tests/{p.name}" for p in HERE.glob("test_*.py")}
    assert not files - set(known), (
        f"add to tests/{name}: {sorted(files - set(known))}"
    )
    assert not set(known) - files, (
        f"tests/{name} has files that no longer exist"
    )


def test_files_with_only_full_only_tests_cost_nothing_outside_the_full_tiers():
    """Main and the gate skip `full_only` tests: a file made only of them
    must have a 0.0 timing in test_durations.json."""
    spec = importlib.util.spec_from_file_location(
        "update_test_durations",
        HERE.parent / "scripts/update_test_durations.py",
    )
    updater = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(updater)
    marked = updater.marked("full_only")
    known = json.loads((HERE / "test_durations.json").read_text())
    for path in HERE.glob("test_*.py"):
        rel = f"tests/{path.name}"
        names = {
            n.name
            for n in ast.parse(path.read_text()).body
            if isinstance(n, ast.FunctionDef) and n.name.startswith("test")
        }
        if names and all((rel, n) in marked for n in names):
            assert known[rel] == 0.0, rel
