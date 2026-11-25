"""Slow recovery tests of the orbit cells (decoys, outliers, velocity
criterion), split from test_orbit_cells.py so each file fits one CI shard."""

import jax
import numpy as onp
import pytest
from scipy import stats

pytest.importorskip("jaxoplanet")
pytest.importorskip("blackjax")

from virgil import orbit_cells as oc_cells  # noqa: E402
from virgil.night_summaries import NightPeak, NightSummary  # noqa: E402

from tests.test_orbit_cells import (  # noqa: E402
    BLOCKS,
    LOG_B,
    _orbit,
    _space,
    _summaries,
    _true_cell,
    _truth_distance2,
    _two_season_times,
    x64,
)


def _decoys_and_outliers(summaries, rng, outliers):
    """Each night gets a decoy peak (lower likelihood, random state); the
    nights in ``outliers`` lose their true peak to a random state."""
    out = []
    for i, n in enumerate(summaries):
        p = n.peaks[0]
        mean = p.mean.copy()
        if i in outliers:
            mean[:2] = rng.uniform(-20.0, 20.0, 2)
            mean[2:4] = rng.normal(size=2) * 0.5
        peak = NightPeak(mean, p.cov, p.loglike, p.n, p.chi2, p.hessian)
        decoy_mean = p.mean + rng.normal(size=5) * onp.array(
            [5.0, 5.0, 0.5, 0.5, 0.1]
        )
        decoy = NightPeak(
            decoy_mean, p.cov, p.loglike - 6.0, p.n, p.chi2, p.hessian
        )
        out.append(
            NightSummary(
                n.t_ref,
                n.t_ss,
                n.n_frames,
                BLOCKS,
                (peak, decoy),
                n.loglike_null,
            )
        )
    return tuple(out)


@pytest.mark.slow
@pytest.mark.full_only
@x64
@pytest.mark.timeout(1800)
def test_outlier_nights_decoy_peaks_and_a_set_budget_still_find_the_true_mode(
    monkeypatch,
):
    """K = 10 with two peaks per night, two outlier nights, e = 0.8 and the
    set budget forced low (the interior fills of design 6.1)."""
    rng = onp.random.default_rng(11)
    times = _two_season_times(rng, 10)
    orbit = _orbit(period=11.3, a=6.0, ecc=0.8)
    s = _summaries(orbit, times, sigma=6e-3, seed=1)
    s = _decoys_and_outliers(s, rng, {2, 7})
    monkeypatch.setattr(oc_cells, "N_SET_MAX", 100)
    domain = oc_cells.period_domain(s, p_range=(9.0, 14.0))
    out = oc_cells.position_cells(
        s, domain, jax.random.key(0), log_outlier=onp.full(10, LOG_B),
        jitter=0.0,
    )  # fmt: skip
    assert not out.failed.any()
    assert len(out.fits[0].plan["nights"]) < 2000
    order = onp.argsort(-out.p_pos)
    keep = order[
        : int(onp.searchsorted(onp.cumsum(out.p_pos[order]), 0.99)) + 1
    ]
    true_i = [i for i, c in enumerate(out.cells) if c.p_lo <= 11.3 <= c.p_hi]
    assert true_i[0] in keep
    assert out.fits[true_i[0]].modes[0].n_inliers > 10 - 2 - 0.5


@pytest.mark.slow
@pytest.mark.full_only
@x64
@pytest.mark.timeout(1200)
@pytest.mark.parametrize("ecc", [0.0, 0.5])
def test_the_velocity_criterion_keeps_the_true_mode_with_fewer_triplets(
    monkeypatch, ecc
):
    """Fill regime, K = 8, velocities informative: the criterion drops some
    triplet candidates and the coverage contract of the search still holds."""
    rng = onp.random.default_rng(int(1e4 * ecc) + 1000)
    times = _two_season_times(rng)
    orbit = _orbit(period=11.3, a=6.0, ecc=ecc)
    s = _summaries(orbit, times, sigma=1e-3 * 6.0, seed=1)
    monkeypatch.setattr(oc_cells, "N_SET_MAX", 20)
    domain, cells, cell = _true_cell(11.3, times, s, (9.0, 14.0))
    space = _space(s, cell)
    plan = oc_cells.search_plan(space, jax.random.key(0))
    assert int(oc_cells._grid_stage(space, plan)[3]) > 0
    out = oc_cells.position_cells(
        s, domain, jax.random.key(0), log_outlier=onp.full(len(times), LOG_B),
        jitter=0.0, a_range=(1.0, 30.0), e_max=0.9,
    )  # fmt: skip
    assert not out.failed.any() and onp.all(onp.isfinite(out.p_pos))
    order = onp.argsort(-out.p_pos)
    keep = order[
        : int(onp.searchsorted(onp.cumsum(out.p_pos[order]), 0.99)) + 1
    ]
    true_n = [i for i, c in enumerate(out.cells) if c.p_lo <= 11.3 <= c.p_hi]
    assert true_n[0] in keep
    d2 = _truth_distance2(out.fits[true_n[0]], orbit, times[out.anchors[0]])
    assert d2 < stats.chi2.isf(oc_cells.DELTA, out.fits[true_n[0]].space.d - 1)
