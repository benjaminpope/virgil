"""Slow coverage-contract recovery test of the orbit cells, split from
test_orbit_cells.py so each file fits one CI shard."""

import jax
import numpy as onp
import pytest
from scipy import stats

pytest.importorskip("jaxoplanet")
pytest.importorskip("blackjax")

from virgil import orbit_cells as oc_cells  # noqa: E402

from tests.test_orbit_cells import (  # noqa: E402
    LOG_B,
    _orbit,
    _summaries,
    _truth_distance2,
    _two_season_times,
    x64,
)


@pytest.mark.slow
@pytest.mark.full_only
@x64
@pytest.mark.timeout(1200)
@pytest.mark.parametrize("ecc", [0.0, 0.5])
@pytest.mark.parametrize("rel", [1e-2, 1e-4])
def test_the_true_cell_is_found_and_its_cycle_count_is_in_the_99_percent_set(
    ecc, rel
):
    """The search's coverage contract on K = 8 nights in two seasons."""
    rng = onp.random.default_rng(int(1e4 * ecc) + int(rel * 1e5))
    times = _two_season_times(rng)
    period, a = 11.3, 6.0
    orbit = _orbit(period=period, a=a, ecc=ecc)
    s = _summaries(orbit, times, sigma=rel * a, seed=1)
    domain = oc_cells.period_domain(s, p_range=(9.0, 14.0))
    out = oc_cells.position_cells(
        s, domain, jax.random.key(0), log_outlier=onp.full(len(times), LOG_B),
        jitter=0.0, a_range=(1.0, 30.0), e_max=0.9,
    )  # fmt: skip
    assert not out.failed.any() and onp.all(onp.isfinite(out.p_pos))
    order = onp.argsort(-out.p_pos)
    keep = order[
        : int(onp.searchsorted(onp.cumsum(out.p_pos[order]), 0.99)) + 1
    ]
    true_n = [i for i, c in enumerate(out.cells) if c.p_lo <= period <= c.p_hi]
    assert true_n[0] in keep
    fit = out.fits[true_n[0]]
    # the nearest mode (any symmetry image) is within the 1 - DELTA chi-square
    # radius of the truth: 9 degrees of freedom (the radius of y_z is free)
    d2 = _truth_distance2(fit, orbit, times[out.anchors[0]])
    assert d2 < stats.chi2.isf(oc_cells.DELTA, fit.space.d - 1)
    # the resample-move draws are spread over the posterior, not duplicates
    assert fit.draws is not None and len(onp.unique(fit.draws[:, 0])) > 100
    ln_p = onp.log(oc_cells.decode_particles(fit)["period"])
    assert abs(onp.median(ln_p) - onp.log(period)) < 5 * ln_p.std() + 1e-3
