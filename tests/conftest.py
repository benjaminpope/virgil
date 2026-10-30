"""Evidence for virgil-validation.

Tests can declare what they validate, and against which root of trust, with

    @pytest.mark.validates("virgil.models.UniformDisk", roots=["mathematics"])

(same marker and roots as https://github.com/benjaminpope/virgil-validation,
docs/design.md). Each claim is attached to the test as a JUnit property
``validates``, so `pytest --junitxml=... -o junit_family=xunit1` carries it;
CI uploads that file from main and virgil-validation reads it as evidence.
Tests without the marker are unaffected.
"""

import gc
import json
import os
import re
from pathlib import Path

import jax
import pytest

HERE = Path(__file__).parent

ROOTS = (
    "mathematics",
    "standards",
    "dlux",
    "pmoired",
    "candid",
    "fouriever",
    "statistics",
    "self-consistency",
)
KINDS = ("check", "control", "finding", "upstream", "reference", "guard")
TIERS = ("A", "B", "C")


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "validates(obj, *more, roots, kind='check', tier='A'): virgil objects "
        "this test validates and the roots of trust it checks them against "
        "(see virgil-validation's docs/design.md)",
    )
    # The heaviest, compile-dominated tests. Pull requests run
    # `-m "not slow"`, except that a pull request touching the orbit modules
    # also runs the slow tests of their own test files (VIRGIL_SLOW_FILES
    # below); pushes to main, the `full-ci` label and the nightly run
    # them (.github/scripts/ci_matrix.py).
    config.addinivalue_line(
        "markers",
        "full_only: slow tests too heavy for the orbit-PR slow run; only main, "
        "the full-ci label and the nightly run them",
    )
    config.addinivalue_line(
        "markers",
        "slow: heaviest, compile-dominated tests; skipped on pull requests "
        "(-m 'not slow'; a pull request touching the orbit modules still runs "
        "the orbit test files) but run on main, with the full-ci label and "
        "nightly",
    )


def pytest_addoption(parser):
    parser.addoption(
        "--shard",
        metavar="K/N",
        help="run only shard K of N (1-based): whole test files, packed by "
        "their timings in tests/test_durations.json",
    )


# Test files whose results cannot depend on JAX's float width: file formats,
# plotting, pipeline orchestration, the shims and the benchmark harness, and
# the orbit stack and sampler tests, which run inside their own enable_x64 or
# take dtype as an explicit argument. The
# float64 CI jobs (VIRGIL_SKIP_DTYPE_INDEPENDENT=1) skip them, since the
# float32 jobs already run them. Every other file, including each one that
# carries a `validates` marker (the evidence virgil-validation reads from the
# float64 run), stays in both. tests/lint/test_ci_config.py checks the list.
DTYPE_INDEPENDENT = (
    "test_argument_order.py",
    "test_benchmark_harness.py",
    "test_deprecate.py",
    "test_docs_synthetic_recovery.py",
    "test_epoch_coarse.py",
    "test_epochs.py",
    "test_fetch_references.py",
    "test_night_flags.py",
    "test_oifits.py",
    "test_orbit_positions.py",
    "test_orbit_search.py",
    "test_pipeline_binary.py",
    "test_pipeline_checks.py",
    "test_pipeline_disk.py",
    "test_pipeline_image.py",
    "test_pipeline_io.py",
    "test_pipeline_orbit.py",
    "test_pipeline_orbit_cases.py",
    "test_pipeline_orbit_gap.py",
    "test_pipeline_orbit_resolved.py",
    "test_pipeline_robust_binary.py",
    "test_pipeline_robust_core.py",
    "test_pipeline_robust_disk.py",
    "test_pipeline_robust_star.py",
    "test_pipeline_star.py",
    "test_pipeline_start_at_bound.py",
    "test_plotting_detection.py",
    "test_stability.py",
    "test_svi.py",
    "test_tempering.py",
)


# trylast: after -m has deselected, so --shard packs only what will run.
@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(config, items):
    # VIRGIL_SLOW_FILES (space-separated file-name globs): run only the slow
    # tests of those files, in place of `-m "not slow"` (see tests.yml).
    globs = os.environ.get("VIRGIL_SLOW_FILES", "").split()
    if globs:
        drop = [
            item
            for item in items
            if item.get_closest_marker("slow")
            and (
                item.get_closest_marker("full_only")
                or not any(item.path.match(g) for g in globs)
            )
        ]
        items[:] = [item for item in items if item not in drop]
        config.hook.pytest_deselected(items=drop)
    if os.environ.get("VIRGIL_SKIP_DTYPE_INDEPENDENT") == "1":
        drop = [i for i in items if i.path.name in DTYPE_INDEPENDENT]
        items[:] = [i for i in items if i not in drop]
        config.hook.pytest_deselected(items=drop)
    for item in items:
        for mark in item.iter_markers("validates"):
            roots = list(mark.kwargs.get("roots", ()))
            kind = mark.kwargs.get("kind", "check")
            bad = [
                r
                for r in roots
                if r not in ROOTS and not r.startswith("golden:")
            ]
            tier = mark.kwargs.get("tier", "A")
            if (
                not mark.args
                or not roots
                or bad
                or kind not in KINDS
                or tier not in TIERS
            ):
                raise pytest.UsageError(
                    f"{item.nodeid}: malformed validates marker"
                )
            claim = {
                "objects": list(mark.args),
                "roots": roots,
                "kind": kind,
                "tier": tier,
            }
            item.user_properties.append(("validates", json.dumps(claim)))
    _keep_shard(config, items)


@pytest.fixture(autouse=True, scope="module")
def _clear_jax_caches():
    """Drop JAX's in-memory compilation caches after each test module.

    JAX keeps every compiled executable alive for the life of the process,
    so a pytest-xdist worker's memory grows with every test it has run. On
    CI's 16 GB, 4-core runners four workers reached 4 GB each and the
    runner swapped itself to death (stalls reported as pytest timeouts in
    whatever JAX call was running, then a runner shutdown). Programs a later
    module needs again are recompiled, or read from the persistent cache CI
    keeps. Under xdist's default ``--dist load`` a worker can receive one
    module's tests in several batches, so this can run more than once per
    module on a worker; that only costs recompiles.
    """
    yield
    jax.clear_caches()
    gc.collect()


def _shard_files(files, n):
    """Pack test files into n shards by their recorded timings.

    Whole files, so a module-scoped fixture (and an xdist_group) runs in one
    job only. Longest first onto the least-loaded shard; a file with no
    recorded timing counts as the mean. Every shard collects the same files,
    so each computes the same packing.
    """
    known = json.loads((HERE / "test_durations.json").read_text())
    mean = sum(known.values()) / len(known)
    loads = [0.0] * n
    shard = {}
    for path in sorted(files, key=lambda f: (-known.get(f, mean), f)):
        k = loads.index(min(loads))
        shard[path] = k
        loads[k] += known.get(path, mean)
    return shard


def _keep_shard(config, items):
    """Keep only this job's shard (``--shard K/N``)."""
    spec = config.getoption("--shard")
    if not spec:
        return
    k, n = (int(x) for x in spec.split("/"))
    if not 1 <= k <= n:
        raise pytest.UsageError(f"--shard {spec}: need 1 <= K <= N")
    shard = _shard_files({item.nodeid.split("::")[0] for item in items}, n)
    keep, drop = [], []
    for item in items:
        (keep if shard[item.nodeid.split("::")[0]] == k - 1 else drop).append(
            item
        )
    config.hook.pytest_deselected(items=drop)
    items[:] = keep


# The per-test time budget: CI sets VIRGIL_TEST_BUDGET (seconds) on the
# pull-request jobs, when the JAX cache is warm. A test not marked slow fails
# when its call, or its setup, takes longer, so the fast tier stays fast as
# the suite grows. Setup is charged to the file, not the test: under xdist
# the first test of a module on each worker pays for the module-scoped
# fixtures (where sampler runs hide), and which test that is varies from run
# to run. Shrink the test or fixture, or mark it slow. tests/over_budget.txt
# lists the tests (node ids, for the call) and files (for setup) that were
# over budget when it was introduced; it should only get shorter.
_BUDGET = float(os.environ.get("VIRGIL_TEST_BUDGET") or 0)
_EXEMPT = {
    line.strip()
    for line in (HERE / "over_budget.txt").read_text().splitlines()
    if line.strip() and not line.startswith("#")
}


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    report = (yield).get_result()
    if (
        not _BUDGET
        or report.when == "teardown"
        or not report.passed
        or report.duration <= _BUDGET
        or item.get_closest_marker("slow") is not None
    ):
        return
    # pytest-xdist's loadgroup appends "@group" to the node id.
    nodeid = re.sub(r"@[^\]]*$", "", item.nodeid)
    path = nodeid.split("::")[0]
    if report.when == "setup" and path not in _EXEMPT:
        what = f"setup of {nodeid} (the fixtures of {path})"
    elif report.when == "call" and nodeid not in _EXEMPT:
        what = nodeid
    else:
        return
    report.outcome = "failed"
    report.longrepr = (
        f"The {what} took {report.duration:.1f} s, over the {_BUDGET:g} s "
        "budget for tests not marked slow. Shrink it (a smaller problem, "
        "fewer draws, a stub run) or mark it @pytest.mark.slow; see "
        "tests/conftest.py."
    )


@pytest.fixture
def junk_files(tmp_path):
    """Files that are not valid OIFITS data: text, empty, and table-less FITS."""
    from astropy.io import fits

    (tmp_path / "text.fits").write_text("this is not a FITS file\n")
    (tmp_path / "empty.fits").write_bytes(b"")
    fits.PrimaryHDU().writeto(tmp_path / "no_tables.fits")
    return tmp_path
