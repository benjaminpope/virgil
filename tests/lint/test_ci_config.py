"""The CI matrix chooser (.github/scripts/ci_matrix.py) and the test lists.

Pure stdlib: nothing here compiles JAX.
"""

import importlib.util
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "ci_matrix", ROOT / ".github" / "scripts" / "ci_matrix.py"
)
ci = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ci)


def _jobs(*args, **kwargs):
    return ci.choose(*args, **kwargs)["include"]


def _ids(jobs):
    return {j["id"] for j in jobs}


FLOORS = {f"ubuntu-latest-py3.11-x640-lowest-shard{k}of2" for k in "12"}
PY313 = {f"ubuntu-latest-py3.13-x640-shard{k}of2" for k in "12"}
GATE = {
    f"ubuntu-latest-py3.12-x64{x}-shard{k}of3" for x in "01" for k in "123"
}


def test_pull_request_runs_only_the_gate():
    jobs = _jobs("pull_request")
    assert _ids(jobs) == GATE
    assert not any(j["slow"] or j["orbit_slow"] for j in jobs)


def test_pull_request_touching_orbit_modules_adds_their_slow_tests():
    jobs = _jobs("pull_request", orbit_slow=True)
    assert _ids(jobs) == GATE
    assert all(
        not j["slow"] and "test_aliases.py" in j["orbit_slow"] for j in jobs
    )


def test_pyproject_pull_request_adds_the_floors_only():
    jobs = _jobs("pull_request", pyproject=True)
    assert _ids(jobs) == GATE | FLOORS
    assert not any(j["slow"] for j in jobs)


def test_main_push_is_the_gate_with_slow_tests():
    jobs = _jobs("push")
    assert _ids(jobs) == GATE
    assert all(j["slow"] and not j["orbit_slow"] for j in jobs)


def test_full_ci_label_adds_macos_py313_and_the_floors():
    jobs = _jobs("pull_request", full_label=True)
    assert _ids(jobs) == GATE | PY313 | FLOORS | {"macos-latest-py3.11-x640"}
    assert all(j["slow"] for j in jobs if j["id"] in GATE)
    # The extra jobs run slow tests nightly only.
    assert not any(j["slow"] for j in jobs if j["id"] not in GATE)


def test_dispatch_is_full_and_nightly_adds_canary_and_slow_everywhere():
    assert _ids(_jobs("workflow_dispatch")) == _ids(
        _jobs("pull_request", full_label=True)
    )
    nightly = _jobs("schedule")
    assert _ids(nightly) == _ids(_jobs("workflow_dispatch")) | {
        "ubuntu-latest-py3.12-x640-canary"
    }
    assert all(j["slow"] for j in nightly)


@pytest.mark.parametrize(
    "event", ["pull_request", "push", "schedule", "workflow_dispatch"]
)
def test_ids_are_unique(event):
    jobs = _jobs(event, full_label=True)
    assert len(_ids(jobs)) == len(jobs)


def test_changed_flags():
    assert ci.changed_flags(["README.md"]) == (False, False)
    assert ci.changed_flags(["pyproject.toml", "docs/a.md"]) == (True, False)
    for path in (
        "src/virgil/aliases.py",
        "src/virgil/orbit_search.py",
        "src/virgil/orbits.py",
        "tests/test_epochs.py",
        "src/virgil/arcs.py",
        "src/virgil/tempering.py",
        "src/virgil/pipeline/orbit.py",
        "tests/test_arcs.py",
        "tests/test_tempering.py",
        "tests/test_pipeline_orbit_gap.py",
        "tests/test_orbit_positions.py",
        "src/virgil/pipeline/disk.py",
        "tests/test_pipeline_disk.py",
        "src/virgil/pipeline/image.py",
        "tests/test_pipeline_image.py",
    ):
        assert ci.changed_flags([path]) == (False, True), path
    # Subpackages, and tests that merely mention an orbit, do not count.
    assert ci.changed_flags(["src/virgil/pipeline/orbits.py"]) == (
        False,
        False,
    )
    assert ci.changed_flags(["tests/test_pipeline_binary.py"]) == (
        False,
        False,
    )


def test_dtype_independent_list_is_safe():
    """Files skipped on the float64 jobs exist and carry no evidence."""
    from tests.conftest import DTYPE_INDEPENDENT

    assert len(set(DTYPE_INDEPENDENT)) == len(DTYPE_INDEPENDENT)
    for name in DTYPE_INDEPENDENT:
        path = ROOT / "tests" / name
        assert path.is_file(), f"{name} is listed but does not exist"
        # The float64 run is the evidence virgil-validation reads.
        assert not re.search(r"validates\(", path.read_text()), name
    # The numerical files the float64 jobs exist for stay in.
    for name in ("test_aliases.py", "test_fitting.py", "test_likelihood.py"):
        assert name not in DTYPE_INDEPENDENT


def test_main_reads_the_environment(monkeypatch, capsys):
    monkeypatch.setenv("EVENT", "workflow_dispatch")
    monkeypatch.setenv("NIGHTLY", "true")
    ci.main()
    out = capsys.readouterr().out
    assert out.startswith("matrix=")
    assert "canary" in out
    monkeypatch.setenv("NIGHTLY", "false")
    ci.main()
    assert "canary" not in capsys.readouterr().out


def _script(name):
    spec = importlib.util.spec_from_file_location(
        name, ROOT / ".github" / "scripts" / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _job(name, conclusion="success"):
    return {"name": name, "conclusion": conclusion}


FULL_JOBS = [
    _job("ubuntu-latest py3.12 shard 1/3"),
    _job("macos-latest py3.11"),
    _job("ubuntu-latest py3.13 1/2"),
    _job("ubuntu-latest py3.13 2/2"),
    _job("ubuntu-latest py3.11 lowest 1/2"),
    _job("ubuntu-latest py3.11 lowest 2/2"),
    _job("build and install the wheel"),
    _job("join the x64 evidence", "skipped"),
]


def test_release_gate_accepts_a_green_full_run():
    gate = _script("check_release_gate")
    assert gate.problems({"conclusion": "success"}, FULL_JOBS) == []


def test_release_gate_names_what_fails():
    gate = _script("check_release_gate")
    assert gate.problems({"conclusion": "failure"}, FULL_JOBS) == [
        "concluded failure"
    ]
    # A run without the full tier's jobs does not count.
    partial = [j for j in FULL_JOBS if "macos" not in j["name"]]
    assert gate.problems({"conclusion": "success"}, partial) == [
        "missing job macos-latest py3.11"
    ]
    red = FULL_JOBS + [_job("ubuntu-latest py3.13 x", "cancelled")]
    assert gate.problems({"conclusion": "success"}, red) == [
        "ubuntu-latest py3.13 x failed"
    ]


def test_release_gate_job_names_match_tests_yml():
    """The substrings the gate requires are what tests.yml names the jobs.

    Read from the ``build`` job's ``name:`` template, so renaming a part of
    it (or reordering it) fails here, not on release day.
    """
    gate = _script("check_release_gate")
    tests = (ROOT / ".github/workflows/tests.yml").read_text()
    name = re.search(r"^  build:\n    name: (.+)$", tests, re.M).group(1)

    # Render the template for each job of the full tier, as GitHub would.
    def render(job):
        out = name
        for key, value in {
            "matrix.os": job["os"],
            "matrix.python-version": job["python-version"],
        }.items():
            out = out.replace("${{ " + key + " }}", value)
        # The remaining expressions append a suffix when a flag is set.
        for flag, suffix in (("lowest", " lowest"), ("canary", " canary")):
            expr = f"matrix.{flag} && '{suffix}'"
            assert expr in out, f"build.name lost its {flag} suffix"
            out = re.sub(
                r"\$\{\{ [^}]*" + re.escape(expr) + r"[^}]*\}\}",
                suffix if job.get(flag) else "",
                out,
            )
        return re.sub(r"\$\{\{.*?\}\}", "", out)

    names = [render(j) for j in _jobs("workflow_dispatch")]
    for needle in gate.FULL_JOBS:
        assert any(needle in n for n in names), (needle, names)
    assert set(gate.FULL_EVENTS) == {"workflow_dispatch", "schedule"}
    # A push run lacks the full tier's jobs, so it must not satisfy the gate.
    assert "push" not in gate.FULL_EVENTS


def test_publish_waits_for_the_gate_and_floors_install_orbits():
    publish = (ROOT / ".github/workflows/publish.yml").read_text()
    assert "check_release_gate.py" in publish
    assert re.search(r"\bbuild:\n    needs: gate\b", publish)
    tests = (ROOT / ".github/workflows/tests.yml").read_text()
    assert '--resolution lowest-direct -e ".[test,orbits]"' in tests


def test_label_job_can_label_pull_requests():
    """GITHUB_TOKEN labels a PR only with pull-requests: write (a 403 else)."""
    review = (ROOT / ".github/workflows/claude-review.yml").read_text()
    job = review[review.index("\n  label:\n") :]
    perms = job[job.index("permissions:") :].split("steps:")[0]
    assert "pull-requests: write" in perms
    assert "issues: write" in perms


def test_test_budget_stays_active_when_shard_packing_changes():
    """Adding or renaming a test file must not disable VIRGIL_TEST_BUDGET."""
    tests = (ROOT / ".github/workflows/tests.yml").read_text()
    line = next(x for x in tests.splitlines() if "VIRGIL_TEST_BUDGET:" in x)
    assert "steps.key.outputs.prefix" not in line
    assert (
        "format('jax-{0}-jax{1}-', matrix.id, steps.key.outputs.jax)" in line
    )
