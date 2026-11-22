"""The CI matrix chooser (.github/scripts/ci_matrix.py) and the test lists.

Pure stdlib: nothing here compiles JAX.
"""

import importlib.util
import json
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


# The full tiers run the `full_only` tests too, in more shards.
FULL_GATE = {
    f"ubuntu-latest-py3.12-x640-shard{k}of{ci.FULL_F32_SHARDS}"
    for k in range(1, ci.FULL_F32_SHARDS + 1)
} | {
    f"ubuntu-latest-py3.12-x641-shard{k}of{ci.FULL_X64_SHARDS}"
    for k in range(1, ci.FULL_X64_SHARDS + 1)
}
NIGHTLY_EXTRA = {
    f"ubuntu-latest-py3.{v}-x640{low}-shard{k}of{ci.NIGHTLY_FLOOR_SHARDS}"
    for v, low in (("13", ""), ("11", "-lowest"))
    for k in range(1, ci.NIGHTLY_FLOOR_SHARDS + 1)
}


def test_pull_request_runs_only_the_gate():
    jobs = _jobs("pull_request")
    assert _ids(jobs) == GATE
    assert not any(j["slow"] or j["orbit_slow"] for j in jobs)


def test_pull_request_touching_orbit_modules_adds_their_slow_tests():
    assert "test_pipeline_robust_disk.py" in ci.ORBIT_SLOW_FILES.split()
    jobs = _jobs("pull_request", orbit_slow=True)
    assert _ids(jobs) == GATE
    assert all(
        not j["slow"]
        and "test_aliases.py" in j["orbit_slow"]
        and "test_pipeline_robust_disk.py" in j["orbit_slow"]
        for j in jobs
    )


def test_pull_request_touching_test_file_adds_only_its_slow_tests():
    jobs = _jobs("pull_request", extra_slow=["test_sampling.py"])
    assert _ids(jobs) == GATE
    assert all(
        not j["slow"] and j["orbit_slow"] == "test_sampling.py" for j in jobs
    )
    # Combining orbit_slow with extra_slow preserves order without duplicates.
    combined = _jobs(
        "pull_request",
        orbit_slow=True,
        extra_slow=["test_pipeline_disk.py", "test_sampling.py"],
    )
    expected = f"{ci.ORBIT_SLOW_FILES} test_sampling.py"
    assert all(j["orbit_slow"] == expected for j in combined)


def test_pyproject_pull_request_adds_the_floors_only():
    jobs = _jobs("pull_request", pyproject=True)
    assert _ids(jobs) == GATE | FLOORS
    assert not any(j["slow"] for j in jobs)


def test_main_push_is_the_gate_with_slow_tests():
    jobs = _jobs("push")
    assert _ids(jobs) == GATE
    assert all(j["slow"] and not j["orbit_slow"] for j in jobs)
    # Cold float32 shards with every slow test outran the 45-minute limit.
    assert all(j["skip_full_only"] for j in jobs)


def test_full_ci_label_adds_macos_py313_and_the_floors():
    jobs = _jobs("pull_request", full_label=True)
    assert _ids(jobs) == FULL_GATE | PY313 | FLOORS | {
        "macos-latest-py3.11-x640"
    }
    assert all(j["slow"] for j in jobs if j["id"] in FULL_GATE)
    assert not any(j.get("skip_full_only") for j in jobs)
    # The extra jobs run slow tests nightly only.
    assert not any(j["slow"] for j in jobs if j["id"] not in FULL_GATE)


def test_dispatch_is_full_and_nightly_is_full_with_slow_everywhere():
    assert _ids(_jobs("workflow_dispatch")) == _ids(
        _jobs("pull_request", full_label=True)
    )
    nightly = _jobs("schedule")
    # The nightly shards py3.13 and the floors more finely: they run the
    # slow and `full_only` tests there.
    assert _ids(nightly) == FULL_GATE | NIGHTLY_EXTRA | {
        "macos-latest-py3.11-x640"
    }
    assert _ids(_jobs("workflow_dispatch")) == _ids(
        _jobs("pull_request", full_label=True)
    )
    assert not any("canary" in i for i in _ids(nightly))
    assert all(j["slow"] for j in nightly if j["os"] != "macos-latest")
    assert all(j["full_durations"] for j in nightly if "shard" in j)


def test_dispatch_with_main_tier_is_the_main_tier():
    """The lock-upgrade dispatch: the gate with slow tests, no full jobs."""
    jobs = _jobs("workflow_dispatch", tier="main")
    assert _ids(jobs) == GATE == _ids(_jobs("push"))
    assert jobs == _jobs("push")
    # An empty or unknown tier stays full.
    for tier in ("", "full"):
        assert _ids(_jobs("workflow_dispatch", tier=tier)) == _ids(
            _jobs("workflow_dispatch")
        )
    # The tier input does not change other events.
    assert _ids(_jobs("pull_request", tier="main")) == GATE


def test_macos_job_is_float32_and_fast_only():
    (mac,) = [
        j for j in _jobs("workflow_dispatch") if j["os"] == "macos-latest"
    ]
    assert mac["x64"] == "0" and mac["fast_only"]


def test_lock_upgrade_dispatches_the_main_tier():
    lock = (ROOT / ".github/workflows/lock-upgrade.yml").read_text()
    assert 'gh workflow run tests.yml --ref "$BRANCH" -f tier=main' in lock
    tests = (ROOT / ".github/workflows/tests.yml").read_text()
    assert "TIER: ${{ inputs.tier }}" in tests
    assert "matrix.canary" not in tests


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
        "src/virgil/pipeline/_checks.py",
        "src/virgil/pipeline/_core.py",
        "tests/test_arcs.py",
        "tests/test_tempering.py",
        "tests/test_pipeline_orbit_gap.py",
        "tests/test_orbit_positions.py",
        "src/virgil/pipeline/disk.py",
        "tests/test_pipeline_disk.py",
        "tests/test_pipeline_robust_disk.py",
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


def test_changed_test_files():
    files = [
        "README.md",
        "tests/lint/test_ci_config.py",
        "tests/_shared.py",
        "tests/test_durations.json",
        "tests/test_sampling.py",
        "tests/test_pipeline_binary.py",
        "tests/test_sampling.py",
    ]
    assert ci.changed_test_files(files) == [
        "test_pipeline_binary.py",
        "test_sampling.py",
    ]


def test_changed_source_globs():
    assert ci.changed_source_globs(["README.md", "tests/test_x.py"]) == []
    assert ci.changed_source_globs(["src/virgil/__init__.py"]) == []
    assert ci.changed_source_globs(["src/virgil/_orbit_coordinates.py"]) == [
        "test_orbit_coordinates*.py"
    ]
    assert ci.changed_source_globs(["src/virgil/fitting.py"]) == [
        "test_fit*.py",
        "test_fitting*.py",
    ]
    assert ci.changed_source_globs(["src/virgil/pipeline/_common.py"]) == [
        "test_pipeline_*.py"
    ]
    assert ci.changed_source_globs(["tests/_orbit_toy.py"]) == [
        "test_pipeline_orbit*.py"
    ]
    assert "test_night_*.py" in ci.changed_source_globs(
        ["src/virgil/night_summaries.py"]
    )


def _slow_test_files():
    """{file name: has a slow test that is not `full_only`}, by AST scan."""
    import ast

    def marks(node):
        text = ""
        for dec in getattr(node, "decorator_list", []):
            text += ast.unparse(dec) + " "
        return text

    out = {}
    for path in sorted((ROOT / "tests").glob("test_*.py")):
        tree = ast.parse(path.read_text())
        module_marks = " ".join(
            ast.unparse(n.value)
            for n in tree.body
            if isinstance(n, ast.Assign)
            and any(getattr(t, "id", "") == "pytestmark" for t in n.targets)
        )
        tests = []
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                cls = marks(node)
                tests += [
                    cls + marks(f)
                    for f in node.body
                    if isinstance(f, ast.FunctionDef)
                ]
            elif isinstance(node, ast.FunctionDef):
                tests.append(marks(node))
        out[path.name] = any(
            "mark.slow" in (m := module_marks + " " + t)
            and "full_only" not in m
            for t in tests
        )
    return out


def test_slow_file_globs_match_runnable_slow_tests():
    """Each ORBIT_SLOW_FILES glob selects a slow test the PR gate would run.

    A glob that matches only `full_only` tests (which VIRGIL_SLOW_FILES
    drops) or no file at all gates nothing.
    """
    import fnmatch

    slow = {name for name, ok in _slow_test_files().items() if ok}
    for glob in ci.ORBIT_SLOW_FILES.split():
        assert any(fnmatch.fnmatch(n, glob) for n in slow), glob


def test_slow_tests_follow_their_module():
    """A change to src/virgil/<m>.py reaches the slow tests named for <m>."""
    import fnmatch

    slow = {name for name, ok in _slow_test_files().items() if ok}
    for path in (ROOT / "src" / "virgil").glob("*.py"):
        if path.stem == "__init__":
            continue
        globs = ci.changed_source_globs([f"src/virgil/{path.name}"])
        named = f"test_{path.stem.lstrip('_')}"
        for name in slow:
            if name == f"{named}.py" or name.startswith(named + "_"):
                assert any(fnmatch.fnmatch(name, g) for g in globs), (
                    path.name,
                    name,
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
    assert "macos-latest" in out
    monkeypatch.setenv("NIGHTLY", "false")
    monkeypatch.setenv("TIER", "main")
    ci.main()
    assert "macos-latest" not in capsys.readouterr().out
    monkeypatch.setenv("EVENT", "pull_request")
    monkeypatch.setenv("PR", "123")
    monkeypatch.setattr(
        ci.subprocess,
        "run",
        lambda *a, **kw: type(
            "R",
            (),
            {"stdout": "src/virgil/fitting.py\ntests/test_sampling.py\n"},
        )(),
    )
    ci.main()
    pr_out = capsys.readouterr().out
    assert (
        '"orbit_slow": "test_sampling.py test_fit*.py test_fitting*.py"'
        in pr_out
    )


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
        for expr, suffix, on in (
            ("matrix.x64 == '1'", " x64", job["x64"] == "1"),
            ("matrix.lowest", " lowest", job.get("lowest")),
        ):
            expr = f"{expr} && '{suffix}'"
            assert expr in out, f"build.name lost its {suffix} suffix"
            out = re.sub(
                r"\$\{\{ [^}]*" + re.escape(expr) + r"[^}]*\}\}",
                suffix if on else "",
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


def test_slow_orbit_tests_are_never_dropped_on_nightly_or_main():
    """``orbit_slow`` and ``extra_slow`` add slow tests to a PR gate run;
    main, dispatch, full-ci and the nightly already run every slow test, so
    none of those sets ``orbit_slow``."""
    for event in ("push", "schedule", "workflow_dispatch"):
        jobs = _jobs(event, orbit_slow=True, extra_slow=["test_sampling.py"])
        assert not any(j.get("orbit_slow") for j in jobs), event
    full_pr = _jobs(
        "pull_request",
        full_label=True,
        orbit_slow=True,
        extra_slow=["test_sampling.py"],
    )
    assert not any(j.get("orbit_slow") for j in full_pr)


def _worst_shard_minutes(job, monkeypatch):
    """Estimated wall minutes of the slowest shard of a sharded job: the
    packing of tests/conftest.py itself, whose summed per-test time halves on
    the runner's two xdist workers (measured: 50 min wall for 90 summed)."""
    from tests import conftest

    monkeypatch.setenv(
        "VIRGIL_FULL_DURATIONS", "1" if job["full_durations"] else ""
    )
    durations = json.loads(
        (
            ROOT
            / "tests"
            / (
                "test_durations_full.json"
                if job["full_durations"]
                else "test_durations.json"
            )
        ).read_text()
    )
    files = [
        f"tests/{p.name}" for p in sorted((ROOT / "tests").glob("test_*.py"))
    ]
    if job["x64"] == "1":
        files = [
            f for f in files if Path(f).name not in conftest.DTYPE_INDEPENDENT
        ]
    n = int(job["shard"].split("/")[1])
    pack = conftest._shard_files(files, n)
    loads = [0.0] * n
    for f in files:
        loads[pack[f]] += durations[f]
    return max(loads) / 120


def test_nightly_shards_fit_well_inside_the_job_cap(monkeypatch):
    """Estimated wall time of every nightly shard <= 60% of its 90 minutes
    (tests/test_durations_full.json is summed test time from the nightly
    JUnit reports; refresh it with scripts/update_test_durations.py when the
    suite grows)."""
    for job in _jobs("schedule"):
        if "shard" in job:
            worst = _worst_shard_minutes(job, monkeypatch)
            assert worst <= 0.6 * 90, (job["id"], round(worst), "min")


def test_full_durations_change_the_packing(monkeypatch):
    from tests import conftest

    files = [
        f"tests/{p.name}" for p in sorted((ROOT / "tests").glob("test_*.py"))
    ]
    monkeypatch.setenv("VIRGIL_FULL_DURATIONS", "")
    fast = conftest._shard_files(files, 6)
    monkeypatch.setenv("VIRGIL_FULL_DURATIONS", "1")
    assert conftest._shard_files(files, 6) != fast


def test_nightly_report_names_cancelled_jobs_and_counts_them_red(monkeypatch):
    """A job cancelled at its time cap has no failing test: the verdict is
    failure (anything but success) and the issue names the job."""
    tests = (ROOT / ".github/workflows/tests.yml").read_text()
    verdict = next(
        x for x in tests.splitlines() if "RESULT: ${{ needs.build" in x
    )
    assert "needs.build.result == 'success'" in verdict
    assert "&& 'success' || 'failure'" in verdict
    report = _script("nightly_report")
    jobs = [
        {"name": "a", "conclusion": "success"},
        {"name": "b", "conclusion": "cancelled"},
        {"name": "c", "conclusion": "skipped"},
        {"name": "d", "conclusion": ""},
    ]
    monkeypatch.setattr(report, "gh", lambda *a: json.dumps({"jobs": jobs}))
    monkeypatch.setenv("GITHUB_RUN_ID", "1")
    assert report.unsuccessful_jobs() == [("b", "cancelled")]


def test_hung_tests_are_ended_by_the_thread_timeout():
    """The signal method cannot interrupt native code (a macOS nightly job
    hung 35 minutes in the last tests): the thread method dumps the stacks."""
    tests = (ROOT / ".github/workflows/tests.yml").read_text()
    assert "--timeout-method=thread" in tests


def test_full_durations_reach_the_packer_only_in_the_full_tiers():
    """matrix key -> workflow env (cache-key step and pytest step) -> conftest
    must agree, or the nightly silently packs by the fast-tier timings."""
    tests = (ROOT / ".github/workflows/tests.yml").read_text()
    assert tests.count("VIRGIL_FULL_DURATIONS: ${{ matrix.full_durations") == 1
    assert 'export VIRGIL_FULL_DURATIONS="${{ matrix.full_durations' in tests
    for event in ("pull_request", "push"):
        assert not any(j.get("full_durations") for j in _jobs(event)), event
    assert all(j["full_durations"] for j in _jobs("schedule") if "shard" in j)
    # Pull-request and main variants, and the main-tier dispatch, keep the
    # fast-tier packing; a full dispatch packs the full-tier jobs by the
    # full timings but not the fast-tier py3.13 and floors.
    for jobs in (
        _jobs("pull_request", pyproject=True),
        _jobs("workflow_dispatch", tier="main"),
    ):
        assert not any(j.get("full_durations") for j in jobs)
    dispatch = _jobs("workflow_dispatch")
    assert all(j["full_durations"] for j in dispatch if j["id"] in FULL_GATE)
    assert not any(
        j.get("full_durations") for j in dispatch if j["id"] not in FULL_GATE
    )
    assert all(
        j["full_durations"]
        for j in _jobs("pull_request", full_label=True)
        if j["id"] in FULL_GATE
    )


def test_readme_integration_badge_is_scoped_to_main_push():
    """The tests.yml badge must reflect push runs on main, not schedule runs."""
    for rel in ("README.md", "docs/index.md"):
        text = (ROOT / rel).read_text(encoding="utf-8")
        assert (
            "actions/workflows/tests.yml/badge.svg?branch=main&event=push"
            in text
        ), rel
