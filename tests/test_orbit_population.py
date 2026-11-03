"""The population script: its draws follow the fit's prior, its verdict is
calibrated to the number of fitted systems."""

import importlib.util
import pathlib

import numpy as onp
import pytest

pytest.importorskip("jaxoplanet")
pytest.importorskip("blackjax")

PATH = pathlib.Path(__file__).parents[1] / "scripts" / "orbit_population.py"
spec = importlib.util.spec_from_file_location("orbit_population", PATH)
pop = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pop)


def _row(i, in_99=1, status="ok", n=200):
    u = (i + 0.5) / n  # a uniform PIT
    return dict(
        status=status, in_99=in_99, pit_n=u, pit_lnp=u, seconds=1.0,
        period=5.0, k=8, rel=1e-3, p_true=0.5,
    )  # fmt: skip


def test_nothing_fitted_is_never_a_pass():
    rows = [dict(status="outside_domain") for _ in range(10)]
    text, verdict = pop.summarize(rows, 10)
    assert verdict == "INCONCLUSIVE" and "PASS" not in text


def test_too_few_fitted_systems_are_inconclusive_not_a_pass():
    _, verdict = pop.summarize([_row(i, n=60) for i in range(60)], 200)
    assert verdict == "INCONCLUSIVE"


def test_enough_calibrated_systems_pass():
    _, verdict = pop.summarize([_row(i) for i in range(200)], 200)
    assert verdict == "PASS"


def test_the_miss_threshold_follows_the_number_fitted():
    # 7 misses of 60 fitted: P(Bin(60, 0.01) >= 7) ~ 1e-6, a failure
    rows = [_row(i, in_99=int(i >= 7), n=60) for i in range(60)]
    assert pop.summarize(rows, 200)[1] == "FAIL"
    # 2 of 200 is unremarkable at a 1 % miss rate
    rows = [_row(i, in_99=int(i >= 2)) for i in range(200)]
    assert pop.summarize(rows, 200)[1] == "PASS"
    assert pop.miss_tail(200, 7) < pop.P_MISS < pop.miss_tail(200, 5)


def test_a_failed_cell_fails_the_run():
    rows = [_row(i) for i in range(199)] + [dict(status="failed")]
    assert pop.summarize(rows, 200)[1] == "FAIL"


def test_the_draws_follow_the_fitted_prior():
    rng = onp.random.default_rng(0)
    seen = 0
    for _ in range(12):
        nights = pop.draw_nights(rng)
        k = len(nights["times"])
        assert 5 <= k <= 20 and 1e-4 <= nights["rel"] <= 1e-2
        _, fit = pop.fit_domain(nights)
        if fit is None:
            continue
        assert pop.P_TRUE[0] <= fit.p_cut < fit.p_max <= pop.P_TRUE[1]
        sysd = pop.draw_system(rng, nights, fit)
        assert fit.p_cut <= sysd["period"] <= fit.p_max
        assert 0.0 <= float(sysd["orbit"].ecc) <= pop.oc_cells.E_MAX
        assert len(sysd["summaries"]) == k
        seen += 1
    assert seen >= 1


def test_every_simulated_night_has_a_finite_outlier_density():
    """An inlier's observed ln f stays inside the flux prior's support, so
    the outlier density b_k (uniform on it) is finite for every night."""
    rng = onp.random.default_rng(0)
    lo = onp.log(pop.oc_cells.FLUX_FLOOR)
    n_checked = 0
    while n_checked < 40:
        nights = pop.draw_nights(rng)
        _, fit = pop.fit_domain(nights)
        if fit is None:
            continue
        sysd = pop.draw_system(rng, nights, fit)
        ln_f = onp.array([s.peaks[0].mean[4] for s in sysd["summaries"]])
        assert onp.all((ln_f >= lo) & (ln_f <= 0.0))
        n_checked += 1
