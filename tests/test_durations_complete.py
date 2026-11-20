"""Every test file has a timing in tests/test_durations.json and
tests/test_durations_full.json, and neither names a deleted file.

A file without one is packed as the mean file, which unbalances the shards
(tests/conftest.py). Refresh with scripts/update_test_durations.py.
"""

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
