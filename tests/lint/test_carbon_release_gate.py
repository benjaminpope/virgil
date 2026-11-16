"""The carbon release stamp (scripts/dev_carbon.py) and its gate.

Pure stdlib: no transcripts, no ssh, no carbon_report.py.
"""

import importlib.util
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


gate = _load("check_carbon_fresh", ".github/scripts/check_carbon_fresh.py")
dc = _load("dev_carbon", "scripts/dev_carbon.py")

NOW = datetime(2026, 11, 17, 12, 0, tzinfo=timezone(timedelta(hours=11)))


def stamp(version="0.5.0", when=NOW):
    return dc.release_stamp(version, "abc1234", ["claude", "slurm"], when)


def test_stamp_fields():
    assert stamp() == {
        "version": "0.5.0",
        "generated": "2026-11-17T12:00:00+11:00",
        "head": "abc1234",
        "sources": ["claude", "slurm"],
    }


def test_page_sentence():
    assert "refreshed by hand" in dc.refresh_sentence(None)
    assert "v0.5.0, 2026-11-17" in dc.refresh_sentence(stamp())


def test_failed_sources_from_stderr():
    err = "warning: slurm: TimeoutExpired()\nnote\nwarning: other: x\n"
    assert dc.failed_sources(err) == ["slurm"]
    assert dc.failed_sources("") == []


@pytest.mark.parametrize("flag", ["--offline", "--sources=claude"])
def test_release_refuses_partial_runs(flag, capsys):
    with pytest.raises(SystemExit) as e:
        dc.main(["--release", flag])
    assert e.value.code == 2
    assert "--release" in capsys.readouterr().err


def test_fresh_stamp_passes():
    assert gate.problems(stamp(), "0.5.0", NOW + timedelta(days=2)) == []


def test_stale_stamp_fails():
    out = gate.problems(stamp(), "0.5.0", NOW + timedelta(days=9))
    assert len(out) == 1 and "9 days" in out[0]


def test_wrong_version_fails():
    out = gate.problems(stamp("0.4.1"), "0.5.0", NOW)
    assert len(out) == 1 and "v0.4.1" in out[0]


def test_missing_stamp_fails():
    assert gate.problems(None, "0.5.0", NOW)


def test_pr_mode_skips_age_check():
    old = stamp(when=NOW - timedelta(days=60))
    assert gate.problems(old, "0.5.0", None) == []


def test_skip_marker():
    assert gate.skipped("Notes\ncarbon-gate: skip NT outage\n")
    assert not gate.skipped("mid-line carbon-gate: skip")
    assert not gate.skipped("")
    assert not gate.skipped(None)


def test_version_of():
    assert gate.version_of('[project]\nversion = "1.2.3"\n') == "1.2.3"
