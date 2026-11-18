"""The carbon release stamp (scripts/dev_carbon.py) and its gate.

Pure stdlib: no transcripts, no ssh, no carbon_report.py.
"""

import importlib.util
import json
import subprocess
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


def stamp(version="0.5.0", when=NOW, sources=None):
    return dc.release_stamp(version, "abc1234", sources or dc.SOURCES, when)


def test_sources_in_sync():
    assert gate.SOURCES == dc.SOURCES


def test_stamp_fields():
    assert stamp() == {
        "version": "0.5.0",
        "generated": "2026-11-17T12:00:00+11:00",
        "head": "abc1234",
        "sources": list(dc.SOURCES),
    }


def test_page_sentence():
    assert "refreshed by hand" in dc.refresh_sentence(None)
    assert "v0.5.0, 2026-11-17" in dc.refresh_sentence(stamp())


@pytest.mark.parametrize(
    "line",
    [
        "warning: slurm: TimeoutExpired(['ssh', 'nt'], 60)",  # carbon_common.warn
        "ssh: connect to host nt port 22: Operation timed out",
        "Traceback (most recent call last):",
    ],
)
def test_trouble_lines_flag_failures(line):
    assert dc.trouble_lines(f"fine\n{line}\n") == [line]


def test_trouble_lines_quiet_run():
    assert dc.trouble_lines("") == []
    assert dc.trouble_lines("collected 12 records\n") == []


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
    assert len(out) == 1 and "9.0 days" in out[0]


def test_future_stamp_fails():
    out = gate.problems(stamp(), "0.5.0", NOW - timedelta(days=1))
    assert len(out) == 1 and "after the commit" in out[0]


def test_wrong_version_fails():
    out = gate.problems(stamp("0.4.1"), "0.5.0", NOW)
    assert len(out) == 1 and "v0.4.1" in out[0]


def test_missing_stamp_fails():
    assert gate.problems(None, "0.5.0", NOW)


def test_partial_sources_fail():
    out = gate.problems(stamp(sources=["claude"]), "0.5.0", NOW)
    assert len(out) == 1 and "all sources" in out[0]


def test_skip_marker_needs_a_reason():
    assert (
        gate.skip_reason("Notes\ncarbon-gate: skip NT outage\n") == "NT outage"
    )
    assert gate.skip_reason("carbon-gate: skip\n") is None
    assert gate.skip_reason("carbon-gate: skip   \n") is None
    assert gate.skip_reason("mid-line carbon-gate: skip why") is None
    assert gate.skip_reason("") is None
    assert gate.skip_reason(None) is None


def test_version_of():
    assert gate.version_of('[project]\nversion = "1.2.3"\n') == "1.2.3"


# ---- end to end, against stubs -------------------------------------------

STUB = """
import json, sys
a = sys.argv
get = lambda k: a[a.index(k) + 1]
open(get("--archive"), "w").write('{"archived": "changed"}')
json.dump({"badge": {"text": "dev carbon | 1 kg"}, "total": {"kg": [1, 1, 1]}},
          open(get("--summary"), "w"))
open(get("--markdown"), "w").write("fragment")
import os
if os.environ.get("STUB_WARN"):
    print("warning: slurm: TimeoutExpired()", file=sys.stderr)
"""


@pytest.fixture
def tree(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    (home / "carbon_report.py").write_text(STUB)
    monkeypatch.setenv("CLAUDE_CARBON_HOME", str(home))
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "sync_tutorial_docs.py").write_text("")
    (tmp_path / "docs" / "generated").mkdir(parents=True)
    (tmp_path / "pyproject.toml").write_text('[project]\nversion = "0.5.0"\n')
    (tmp_path / "cfg.toml").write_text("")
    (tmp_path / "README.md").write_text(
        "<!-- dev-carbon-badge -->\nold\n<!-- /dev-carbon-badge -->\n"
    )
    for name, path in {
        "ROOT": tmp_path,
        "CONFIG": tmp_path / "cfg.toml",
        "PYPROJECT": tmp_path / "pyproject.toml",
        "README": tmp_path / "README.md",
        "ARCHIVE": tmp_path / "docs/generated/archive.json",
        "SUMMARY": tmp_path / "docs/generated/summary.json",
        "BADGE_JSON": tmp_path / "docs/generated/badge.json",
        "PAGE": tmp_path / "docs/page.md",
    }.items():
        monkeypatch.setattr(dc, name, path)
    monkeypatch.setattr(dc, "page", lambda fragment, summary, home: "PAGE")
    ARCHIVE = tmp_path / "docs/generated/archive.json"
    ARCHIVE.write_text('{"archived": "old"}')
    git = ["git", "-C", str(tmp_path)]
    subprocess.run([*git, "init", "-q"], check=True)
    subprocess.run(
        [
            *git,
            "-c",
            "user.name=t",
            "-c",
            "user.email=t@t",
            "commit",
            "-q",
            "--allow-empty",
            "-m",
            "x",
        ],
        check=True,
    )
    return tmp_path


def test_release_writes_stamp(tree):
    dc.main(["--release"])
    summary = json.loads((tree / "docs/generated/summary.json").read_text())
    assert summary["release"]["version"] == "0.5.0"
    assert summary["release"]["sources"] == list(dc.SOURCES)
    assert "changed" in (tree / "docs/generated/archive.json").read_text()


def test_failed_release_writes_nothing(tree, monkeypatch):
    monkeypatch.setenv("STUB_WARN", "1")
    (tree / "docs/generated/summary.json").write_text("{}")
    before = {
        p: p.read_bytes()
        for p in [
            tree / "docs/generated/archive.json",
            tree / "docs/generated/summary.json",
            tree / "README.md",
        ]
    }
    with pytest.raises(SystemExit):
        dc.main(["--release"])
    assert not (tree / "docs/page.md").exists()
    assert {p: p.read_bytes() for p in before} == before


def test_release_refuses_disabled_source(tree):
    (tree / "cfg.toml").write_text("[slurm]\nenabled = false\n")
    with pytest.raises(SystemExit):
        dc.main(["--release"])
    assert "old" in (tree / "docs/generated/archive.json").read_text()


@pytest.mark.parametrize(
    "tag,body,code",
    [
        ("v0.5.0", "", 0),
        ("v0.6.0", "", 1),
        ("v0.6.0", "carbon-gate: skip NT down", 0),
    ],
)
def test_gate_main_release_mode(tree, monkeypatch, tag, body, code):
    summary = tree / "docs/generated/dev_carbon.json"
    summary.write_text(
        json.dumps({"release": stamp(when=datetime.now().astimezone())})
    )
    monkeypatch.setattr(gate, "ROOT", tree)
    monkeypatch.setattr(gate, "SUMMARY", summary)
    monkeypatch.setenv("GITHUB_REF_NAME", tag)
    monkeypatch.setenv("RELEASE_BODY", body)
    assert gate.main([]) == code


def test_gate_main_pr_mode(tree, monkeypatch):
    summary = tree / "docs/generated/dev_carbon.json"
    summary.write_text(
        json.dumps({"release": stamp(when=datetime.now().astimezone())})
    )
    monkeypatch.setattr(gate, "ROOT", tree)
    monkeypatch.setattr(gate, "SUMMARY", summary)
    git = ["git", "-C", str(tree)]
    subprocess.run([*git, "add", "pyproject.toml"], check=True)
    subprocess.run(
        [
            *git,
            "-c",
            "user.name=t",
            "-c",
            "user.email=t@t",
            "commit",
            "-q",
            "-m",
            "v",
        ],
        check=True,
    )
    # unchanged version against HEAD: not required
    assert gate.main(["--pr", "HEAD"]) == 0
    # bump to a version with no matching stamp
    (tree / "pyproject.toml").write_text('[project]\nversion = "0.6.0"\n')
    monkeypatch.setenv("PR_BODY", "")
    assert gate.main(["--pr", "HEAD"]) == 1
