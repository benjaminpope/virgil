"""Golden blind runs: a twin and a single star.

A blind run takes minutes, so each is marked slow; one file per group keeps the
CI shards balanced (the files are packed whole, tests/conftest.py).
"""

import pytest

pytest.importorskip("h5py")
pytest.importorskip("nbclient")
pytest.importorskip("jaxoplanet")

from tests._orbit_pipeline import (  # noqa: E402
    SEASONS,
    _blind,
    _simulate,
)

pytestmark = pytest.mark.xdist_group("pipeline_orbit_cases")


@pytest.mark.slow
@pytest.mark.full_only
@pytest.mark.timeout(900)
def test_a_twin_is_labels_free(tmp_path):
    data = _simulate(SEASONS, flux=1.0, seed=5)
    _, res = _blind(data, tmp_path / "run")
    assert res.summary["modes"]["labels_free"]
    status = {c.name: c.status for c in res.checks}
    assert status["labels_free"] == "warn"


@pytest.mark.slow
@pytest.mark.timeout(900)
def test_a_single_star_is_no_orbit(tmp_path):
    data = _simulate(SEASONS, flux=None, seed=9)
    _, res = _blind(data, tmp_path / "run")
    modes = res.summary["modes"]
    assert modes["no_orbit"]["verdict"] == "no-orbit"
    assert res.summary["posterior"].get("skipped") == "no-orbit" or (
        "modes" in res.summary["posterior"]
        and not res.summary["posterior"]["modes"]
    )
    status = {c.name: c.status for c in res.checks}
    assert status["no_orbit"] in ("warn", "fail")
    assert res.samples() == {}
