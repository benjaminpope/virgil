"""Golden blind runs: a resolved primary and a highly eccentric orbit.

A blind run takes minutes, so each is marked slow; one file per group keeps the
CI shards balanced (the files are packed whole, tests/conftest.py).
"""

import jax
import pytest

pytest.importorskip("h5py")
pytest.importorskip("nbclient")
pytest.importorskip("jaxoplanet")

from tests._orbit_pipeline import (  # noqa: E402
    FLUX,
    SEASONS,
    T_REF,
    TRUTH,
    _blind,
    _night,
    _simulate,
)
from virgil.orbits import KeplerOrbit  # noqa: E402
from virgil.pipeline import load  # noqa: E402
from virgil.simulate import simulate  # noqa: E402

pytestmark = pytest.mark.xdist_group("pipeline_orbit_resolved")


@pytest.mark.slow
@pytest.mark.full_only
@pytest.mark.timeout(900)
def test_a_resolved_primary_is_fitted(tmp_path):
    import virgil.aliases as aliases

    orbit = KeplerOrbit(**TRUTH, t_ref=T_REF)
    scene = aliases._UDBinary(orbit, FLUX, 1.05)
    key = jax.random.PRNGKey(7)
    data = {}
    with jax.enable_x64(True):
        for k, mjd in enumerate(SEASONS):
            data[f"n{k}"] = simulate(
                scene, _night(mjd, 7.0 * k), key=jax.random.fold_in(key, k)
            )
    _, res = _blind(data, tmp_path / "run")
    search = res.summary["search"]
    assert search["primary_resolved"] is True
    assert search["primary_diam"] == "fit"
    assert res.summary["orbit"]["elements"]["diam"]["median"] == pytest.approx(
        1.05, rel=0.3
    )


@pytest.mark.slow
@pytest.mark.full_only
@pytest.mark.timeout(900)
def test_a_highly_eccentric_orbit_is_recovered(tmp_path):
    truth = {**TRUTH, "ecc": 0.8}
    data = _simulate(SEASONS, truth=truth, seed=11)
    _, res = _blind(data, tmp_path / "run", ecc_max=0.95)
    best = res.summary["modes"]["winner"]["elements"]
    assert abs(best["period"] - TRUTH["period"]) < 0.05
    assert best["ecc"] == pytest.approx(0.8, abs=0.1)
    assert load(tmp_path / "run").summary["chi2"]["best_orbit_reduced"] < 3.0
