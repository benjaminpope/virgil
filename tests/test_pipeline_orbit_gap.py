"""Blind search across the two-season gap: the alias is resolved.

A blind run takes minutes, so each is marked slow; one file per group keeps the
CI shards balanced (the files are packed whole, tests/conftest.py).
"""

import numpy as onp
import pytest

pytest.importorskip("h5py")
pytest.importorskip("nbclient")
pytest.importorskip("jaxoplanet")

from tests._orbit_pipeline import (  # noqa: E402
    CHECKS,
    SEASONS,
    SUMMARY_KEYS,
    TRUTH,
    _blind,
    _simulate,
)

pytestmark = pytest.mark.xdist_group("pipeline_orbit_gap")


@pytest.mark.slow
@pytest.mark.timeout(900)
def test_blind_search_resolves_the_two_season_gap_alias(tmp_path, monkeypatch):
    # Two seasons 238 d apart leave the period aliased to many integer
    # orbit counts; only the closure phases can say which. No period window,
    # start or prior is given: the engine gets the data alone.
    import virgil.aliases as aliases

    seen = {}
    real = aliases.fit_orbit_aliases

    def spy(*args, **kwargs):
        seen.update(args=args, kwargs=kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(aliases, "fit_orbit_aliases", spy)
    data = _simulate(SEASONS)
    pipeline, res = _blind(data, tmp_path / "run")
    assert len(seen["args"]) == 2 and seen["args"][1] is None  # no window
    kwargs = seen["kwargs"]
    # the engine's own defaults, which include the short-arc stage
    assert not {"init", "skip_flagged", "short_arcs"} & set(kwargs)
    assert kwargs["primary_diam"] == "auto" and kwargs["a_range"] == (
        1.0,
        30.0,
    )
    modes = res.summary["modes"]
    n_true = round(float(onp.ptp(SEASONS)) / TRUTH["period"])
    assert modes["winner"]["n"] == n_true
    assert abs(modes["winner"]["elements"]["period"] - TRUTH["period"]) < 0.05
    assert modes["delta_log_z"] > 10.0  # the runner-up is ruled out
    assert modes["no_orbit"]["verdict"] == "orbit"
    arcs = res.summary["search"]["short_arc"]  # the engine's default stage
    assert arcs is not None and n_true in arcs["kept"]
    assert res.aliases().short_arc["kept"] == arcs["kept"]
    post = res.summary["posterior"]["modes"][0]["params"]
    assert abs(post["period"]["median"] - TRUTH["period"]) < 0.05
    assert res.status == "complete"
    assert res.aliases().best.n == n_true
    assert res.samples()["period"].shape == (2, 150)
    names = [c.name for c in res.checks if c.name in CHECKS]
    assert names == CHECKS
    assert res.summary["chi2"]["best_orbit_reduced"] < 3.0
    keys = {
        k: (set(v) if isinstance(v, dict) else None)
        for k, v in res.summary.items()
    }
    assert keys == SUMMARY_KEYS
    assert (res.path / "plots" / "modes_sky.png").exists()
    assert (res.path / "plots" / "search_bands.png").exists()
