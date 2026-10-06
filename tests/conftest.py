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

import jax
import pytest

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
    # `-m "not slow"` to keep CI fast; pushes to main, the weekly schedule
    # and manual dispatches run everything, so no coverage is lost.
    config.addinivalue_line(
        "markers",
        "slow: heaviest, compile-dominated tests; skipped on pull requests "
        "(-m 'not slow') but run on main and the weekly schedule",
    )


def pytest_collection_modifyitems(config, items):
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


@pytest.fixture(autouse=True, scope="module")
def _clear_jax_caches():
    """Drop JAX's in-memory compilation caches after each test module.

    JAX keeps every compiled executable alive for the life of the process,
    so a pytest-xdist worker's memory grows with every test it has run. On
    CI's 16 GB, 4-core runners four workers reached 4 GB each and the
    runner swapped itself to death (stalls reported as pytest timeouts in
    whatever JAX call was running, then a runner shutdown). Programs a later
    module needs again are recompiled, or read from the persistent cache CI
    keeps.
    """
    yield
    jax.clear_caches()
    gc.collect()
