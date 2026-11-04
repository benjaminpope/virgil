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


# --- the census: paired seeds, classes, assertions 5-7 ----------------------


def _draw(seed, i):
    rng, pit = pop.streams(seed, i)
    nights = pop.draw_nights(rng)
    default, fit = pop.fit_domain(nights)
    p_try = pop._loguniform(rng, *pop.P_TRUE)
    if fit is None or not default.p_cut <= p_try <= default.p_max:
        return None
    return pop.draw_system(rng, nights, fit), float(pit.uniform())


def test_a_seed_list_draws_the_same_systems_whatever_the_fits_did():
    """The PIT stream is its own child: consuming it (or not), or drawing the
    systems in another order, changes nothing about any other system."""
    first = {i: _draw(3, i) for i in range(8)}
    again = {}
    for i in reversed(range(8)):
        pop.streams(3, i)[1].uniform(size=int(i % 3))  # "a fit outcome"
        again[i] = _draw(3, i)
    for i, a in first.items():
        b = again[i]
        assert (a is None) == (b is None)
        if a is None:
            continue
        assert a[1] == b[1] and a[0]["period"] == b[0]["period"]
        for sa, sb in zip(a[0]["summaries"], b[0]["summaries"]):
            onp.testing.assert_array_equal(sa.peaks[0].mean, sb.peaks[0].mean)
    assert (
        _draw(4, 0) is None
        or _draw(4, 0)[0]["period"] != first[0][0]["period"]
    )


def _cell(**kw):
    rec = dict(
        flagged=True, nonfinite=False, ess=1.0, laplace=0.0, log_z=0.0,
    )  # fmt: skip
    return {**rec, **kw}


def test_the_class_follows_the_criteria_file_and_its_precedence():
    tol = 4 * 0.1 + 0.01
    orc = dict(oracle_gap=0.0, oracle_sd=0.1, oracle_basin=0)
    assert (
        pop.classify(_cell(**{**orc, "oracle_gap": tol + 0.1}))
        == "search_miss"
    )
    # a proposal miss: a found mode in the truth's basin, ESS below ESS_FAIL
    assert pop.classify(_cell(**{**orc, "oracle_basin": 1})) == "proposal_miss"
    assert (
        pop.classify(_cell(**{**orc, "oracle_basin": 1, "ess": 50.0}))
        != "proposal_miss"
    )
    # a sentinel false positive: Laplace above ln Z by more than W, oracle agrees
    fp = _cell(**orc, laplace=pop.W + 1.0, ess=50.0)
    assert pop.classify(fp) == "sentinel_fp"
    # the search miss outranks it; support is last
    assert pop.classify({**fp, "oracle_gap": tol + 1.0}) == "search_miss"
    assert pop.classify(_cell(nonfinite=True)) == "support"
    # no oracle (another cell): flagged cells are unresolved, never guessed
    assert pop.classify(_cell()) == "unresolved"
    assert pop.classify(_cell(flagged=False)) == "none"


def test_the_cmh_test_sees_a_dependence_on_the_group_and_not_its_absence():
    rng = onp.random.default_rng(0)
    layer = rng.integers(0, 3, 600)
    group = rng.integers(0, 3, 600)
    null = (rng.uniform(size=600) < 0.3).astype(int)
    assert pop.cmh_p(null, group, layer) > 0.01
    dep = (rng.uniform(size=600) < 0.1 + 0.3 * group).astype(int)
    assert pop.cmh_p(dep, group, layer) < 1e-6
    assert pop.cmh_p(onp.zeros(600, int), group, layer) == 1.0


def test_rescaling_the_quoted_errors_is_a_reparameterization_of_the_scale():
    """The data under ``c σ`` at scale ``s`` are the data under ``σ`` at ``c s``:
    the log likelihood and the covariance agree (assertion 6's premise)."""
    c = 1.7
    rng = onp.random.default_rng(1)
    sysd = None
    while sysd is None:
        got = _draw(0, int(rng.integers(0, 50)))
        sysd = got[0] if got else None
    for s0, s1 in zip(sysd["summaries"], pop.rescaled(sysd["summaries"], c)):
        a, b = s0.peaks[0], s1.peaks[0]
        for s in (0.5, 1.0, 3.0):
            assert (
                abs(
                    (b.loglike + b.log_scale(s))
                    - (a.loglike + a.log_scale(c * s))
                )
                < 1e-9
            )
            onp.testing.assert_allclose(
                b.covariance(s), a.covariance(c * s), rtol=1e-9
            )


def test_failed_systems_never_enter_the_gated_sbc_and_assertions_5_to_7_gate():
    ok = [_row(i) for i in range(200)]
    failed = [
        dict(_row(i), status="failed", pit_n=0.0, pit_lnp=0.0)
        for i in range(30)
    ]
    assert (
        pop.summarize(ok + failed, 230)[1] == "FAIL"
    )  # a failure still fails
    text, _ = pop.summarize(ok + failed, 230)
    assert (
        "cell at floor (failed): 30" in text and "SBC chi-square p, N" in text
    )
    # assertion 5: 100 eccentric systems, 6 outside chi2_d(0.99): a failure
    ecc = [dict(r, ecc=0.6, maha2=1.0, maha_df=10) for r in ok[:100]]
    ecc[:6] = [dict(r, maha2=1e3) for r in ecc[:6]]
    assert pop.summarize(ecc + ok[100:], 200)[1] == "FAIL"
    # assertion 6: one violation fails
    viol = [dict(ok[0], eqv0_viol=1)] + ok[1:]
    assert pop.summarize(viol, 200)[1] == "FAIL"
    # assertion 7: failures that depend on ln s at fixed effective noise
    rng = onp.random.default_rng(2)
    rows = []
    for i in range(300):
        ln_s = rng.uniform(-2, 2)
        bad = rng.uniform() < (0.5 if ln_s > 0.7 else 0.05)
        rows.append(dict(
            _row(i % 200), status="failed" if bad else "ok", ln_s=ln_s,
            eff_noise=1e-3 * rng.uniform(1, 10), stratum="rest",
        ))  # fmt: skip
    assert pop._a7(rows) < pop.P_FAIL


def test_the_strata_are_those_of_section_14():
    assert pop.stratum(8, 8) == "B2"
    assert pop.stratum(8, 6) == "le_m" and pop.stratum(8, 7) == "le_m"
    assert pop.stratum(8, 5) == "rest"


@pytest.mark.slow
@pytest.mark.timeout(1200)
def test_the_oracle_and_the_diagnostics_change_no_fit_result():
    """Fit columns are bit-identical with the oracle (and the equivariance
    rerun) off or on; only their own columns are added."""
    import jax

    rng, pit = pop.streams(0, 20)  # a small one-season system
    u = float(pit.uniform())
    nights = pop.draw_nights(rng)
    _, fit = pop.fit_domain(nights)
    pop._loguniform(rng, *pop.P_TRUE)
    sysd = pop.draw_system(rng, nights, fit)
    key = jax.random.key(7)
    off = pop.run_one(sysd, fit, key, u)
    on = pop.run_one(sysd, fit, key, u, oracle=True)
    skip = ("seconds", "seconds_extra", "cells_json", "true_class")
    for k, v in off.items():
        if k not in skip:
            assert on[k] == v or (v != v and on[k] != on[k]), k
    assert "oracle_log_z" in on and "oracle_log_z" not in off
