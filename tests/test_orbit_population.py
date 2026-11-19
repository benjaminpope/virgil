"""The population script: its draws follow the fit's prior, its verdict is
calibrated to the number of fitted systems."""

import dataclasses
import json
import math
import importlib.util
import pathlib
import types

import jax
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
        period=5.0, k=8, rel=1e-3, p_true=0.5, pit_sky=u, sky_miss=0,
        detectable=1, nyquist_ok=1,
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


def test_the_miss_threshold_follows_the_number_in_the_detectable_stratum():
    # 7 sky misses of 60 detectable: P(Bin(60, 0.01) >= 7) ~ 1e-6, a failure
    rows = [dict(_row(i, n=60), sky_miss=int(i < 7)) for i in range(60)]
    assert pop.summarize(rows, 200)[1] == "FAIL"
    # 2 of 200 is unremarkable at a 1 % miss rate
    rows = [dict(_row(i), sky_miss=int(i < 2)) for i in range(200)]
    assert pop.summarize(rows, 200)[1] == "PASS"
    assert pop.miss_tail(200, 7) < pop.P_MISS < pop.miss_tail(200, 5)


def test_misses_and_failures_outside_the_detectable_stratum_are_report_only():
    ok = [_row(i) for i in range(200)]
    # 30 sky misses, 30 element-space misses and a failed system, all undetectable
    hid = [
        dict(_row(i), detectable=0, nyquist_ok=0, sky_miss=1, in_99=0)
        for i in range(30)
    ]
    text, verdict = pop.summarize(
        ok + hid + [dict(status="failed", detectable=0, nyquist_ok=0)], 231
    )
    assert verdict == "PASS"
    assert "outside it (report only): 1" in text
    assert "sky-plane miss, all fitted (report only): 30 of 230" in text
    # the same failure and misses in the detectable stratum fail the run
    det = [dict(r, detectable=1) for r in hid]
    assert pop.summarize(ok + det, 230)[1] == "FAIL"
    assert (
        pop.summarize(
            ok + [dict(status="failed", detectable=1, nyquist_ok=1)], 201
        )[1]
        == "FAIL"
    )


def test_the_old_element_space_lines_are_report_only():
    ok = [
        dict(_row(i), in_99=int(i >= 20), pit_n=0.0, pit_lnp=0.0)
        for i in range(200)
    ]
    text, verdict = pop.summarize(ok, 200)
    assert verdict == "PASS"
    assert "true N outside the 99 % set (report only): 20 of 200" in text
    assert "ln P (resample-move draws; report only)" in text


def test_the_sky_sbc_gates_over_all_fitted_systems_not_only_the_detectable():
    ok = [_row(i) for i in range(200)]
    skew = [dict(r, pit_sky=r["pit_sky"] ** 4) for r in ok]
    assert pop.summarize(skew, 200)[1] == "FAIL"
    # skewed only outside the stratum: still gated (the rank guarantee needs all)
    mixed = [dict(r, detectable=0) for r in skew]
    assert pop.summarize(mixed, 200)[1] == "FAIL"
    text, _ = pop.summarize(ok, 200)
    assert (
        "detectable stratum (report only: the stratum is selected on the truth"
        in text
    )


def test_rows_without_the_new_columns_are_inconclusive_not_a_pass():
    legacy = [
        {k: v for k, v in _row(i).items() if k != "detectable"}
        for i in range(200)
    ]
    text, verdict = pop.summarize(legacy, 200)
    assert verdict == "INCONCLUSIVE" and "lack the detectability" in text


def test_a_failed_cell_in_the_detectable_stratum_fails_the_run():
    rows = [_row(i) for i in range(199)] + [
        dict(status="failed", detectable=1, nyquist_ok=1)
    ]
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


@pytest.mark.slow
def test_every_simulated_night_has_a_finite_outlier_density():
    """The simulated ln f is the truth plus the full Gaussian noise (no
    truncation), so a night can sit above 0; b_k integrates the flux box over
    the ln f error and stays finite for every night."""
    rng = onp.random.default_rng(0)
    n_checked = n_drawn = 0
    while n_checked < 1 and n_drawn < 1000:
        nights = pop.draw_nights(rng)
        _, fit = pop.fit_domain(nights)
        if fit is None:
            continue
        sysd = pop.draw_system(rng, nights, fit)
        n_drawn += 1
        ln_f = onp.array([s.peaks[0].mean[4] for s in sysd["summaries"]])
        if not onp.any(ln_f > 0.0):  # only the systems the box newly supports
            continue
        prior = pop.oc_cells.outlier_prior(fit, n=2**10)
        b = pop.oc_cells.outlier_log_density(sysd["summaries"], prior)
        assert onp.all(onp.isfinite(b))
        n_checked += 1
    # the case the box integral newly supports must actually occur
    assert n_checked == 1


def test_scoring_is_invariant_to_the_orbit_swap():
    """Near-twins are right up to orbit <-> -orbit (omega -> omega + 180):
    the census's orientation-dependent truth quantities agree for a truth and
    its swapped truth (Ben: a fit is "perfect" up to this degeneracy)."""
    rng = onp.random.default_rng(1)
    with jax.enable_x64(True):  # the census scores in float64
        for _ in range(5):
            orbit = pop._orbit(rng, 3.0, 300.0, 8.0)
            swapped = pop.KeplerOrbit(
                orbit.period, orbit.dt_peri, orbit.ecc, orbit.inc,
                (float(orbit.omega) + 180.0) % 360.0, orbit.Omega, orbit.a_mas,
            )  # fmt: skip
            period = float(orbit.period)
            a = pop.truth_invariants(dict(orbit=orbit, period=period), 0.0)
            b = pop.truth_invariants(dict(orbit=swapped, period=period), 0.0)
            for name in pop.INVARIANTS:
                assert a[name] == pytest.approx(b[name], abs=1e-6), name


@dataclasses.dataclass
class _FakeFit:
    draws: onp.ndarray


def test_the_omega_pit_is_the_same_for_draws_in_either_orbit_image(
    monkeypatch,
):
    """The draw side of the fold: a fit whose draws sit in the omega + 180
    image gets the same ``pit_omega`` as one in the truth's image."""
    base = onp.linspace(10.0, 170.0, 9)  # omega draws in [0, 180)
    truth = dict.fromkeys(pop.INVARIANTS, 0.0) | dict(omega=90.0)
    pits = []
    for shift in (0.0, 180.0):
        dec = {n: onp.zeros(9) for n in pop.INVARIANTS}
        dec["omega"] = (base + shift) % 360.0
        monkeypatch.setattr(pop.oc_cells, "decode_particles", lambda f: dec)
        fit = _FakeFit(draws=onp.arange(9.0)[:, None])
        out = pop.invariant_pits([1.0], [fit], truth)
        out |= pop.invariant_pits([1.0], [fit], truth, distinct=True)
        pits.append((out["pit_omega"], out["pitd_omega"]))
    assert pits[0] == pytest.approx(pits[1])
    assert all(0.0 < x < 1.0 for x in pits[0])


# --- the census: paired seeds, classes, assertions 5-7 ----------------------


def _draw(seed, i):
    rng, pit = pop.streams(seed, i)
    nights = pop.draw_nights(rng)
    default, fit = pop.fit_domain(nights)
    p_try = pop._loguniform(rng, *pop.P_TRUE)
    if fit is None or not default.p_cut <= p_try <= default.p_max:
        return None
    return pop.draw_system(rng, nights, fit), float(pit.uniform())


def _population(monkeypatch, outcome):
    """``run_population`` over seed 3 with the fit replaced by ``outcome``
    (a status, or an error), recording what each fit was handed."""
    seen = []

    def run_one(sysd, fit, key, u, **kw):
        seen.append((float(sysd["period"]), float(u)))
        if outcome(len(seen)) == "error":
            raise RuntimeError("boom")
        return dict(status=outcome(len(seen)))

    monkeypatch.setattr(pop, "run_one", run_one)
    return pop.run_population(3, 0, 8, False, 0), seen


def test_a_seed_list_draws_the_same_systems_whatever_the_fits_did(
    monkeypatch,
):
    """Whatever each fit returns (ok, failed, an error), ``run_population``
    hands every later system the same draws: pairing."""
    kinds = ("ok", "failed", "error")
    rows0, seen0 = _population(monkeypatch, lambda n: "ok")
    rows1, seen1 = _population(monkeypatch, lambda n: kinds[n % 3])
    assert seen0 == seen1 and len(seen0) > 0
    for a, b in zip(rows0, rows1):
        assert a["seed"] == b["seed"] == 3 and a["sys"] == b["sys"]
    first = {i: _draw(3, i) for i in range(8)}
    again = {}
    for i in reversed(range(8)):
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
    failed = [dict(_row(i), status="failed", pit_sky=0.0) for i in range(30)]
    assert (
        pop.summarize(ok + failed, 230)[1] == "FAIL"
    )  # a failure still fails
    text, _ = pop.summarize(ok + failed, 230)
    assert "detectable stratum (gated): 30" in text
    # the SBC sees the ok rows only: 30 PITs at 0 would drive p far lower
    p_ok = f"{pop.chi2_uniform([r['pit_sky'] for r in ok]):.3g}"
    assert f"{pop.BINS} bins): {p_ok}" in text
    # assertion 5: 100 eccentric systems, 6 outside chi2_d(0.99): a failure
    ecc = [dict(r, ecc=0.6, maha2=1.0, maha_df=10) for r in ok[:100]]
    ecc[:6] = [dict(r, maha2=1e3) for r in ecc[:6]]
    assert pop.summarize(ecc + ok[100:], 200)[1] == "FAIL"
    # ... but only in the detectable stratum: outside it, report only
    hid5 = [dict(r, detectable=0) for r in ecc[:6]] + ecc[6:]
    text, verdict = pop.summarize(hid5 + ok[100:], 200)
    assert (
        "outside the detectable stratum (report only): 6 misses of 6" in text
    )
    assert pop._a5(hid5, detectable=False) == (6, 6)
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
@pytest.mark.full_only
@pytest.mark.timeout(1200)
def test_the_oracle_and_the_diagnostics_change_no_fit_result():
    """Fit columns are bit-identical with the oracle off or on; only its own
    columns are added. (The refactor of ``orbit_cells`` behind the counters is
    checked against ``main`` by hand, byte for byte; see the PR.)"""
    import jax

    rng, pit = pop.streams(0, 20)  # a small one-season system
    u = float(pit.uniform())
    nights = pop.draw_nights(rng)
    _, fit = pop.fit_domain(nights)
    pop._loguniform(rng, *pop.P_TRUE)
    sysd = pop.draw_system(rng, nights, fit)
    key = jax.random.key(7)
    off = pop.run_one(sysd, fit, key, u)
    store = {}
    on = pop.run_one(sysd, fit, key, u, oracle=True, store=store)
    skip = ("seconds", "seconds_extra", "cells_json", "true_class")
    for k, v in off.items():
        if k not in skip:
            assert on[k] == v or (v != v and on[k] != on[k]), k
    assert "oracle_log_z" in on and "oracle_log_z" not in off
    # persistence: the true cell's modes, draws and keys, and block variances
    assert any(k.endswith("_mode_x") for k in store)
    assert any(k.endswith("_is_log_w") for k in store)
    recs = json.loads(on["cells_json"])
    assert all("persist_error" not in r for r in recs)
    assert any(r["matched"] for r in recs) or all(r["flagged"] for r in recs)


def _few_posterior_draws(monkeypatch, n_draws=100, n_moves=2, n_warm=40):
    """Cut the resample-move work of ``position_cells`` for a slow test.

    The pin test below asserts on ``ln Ẑ``, ``p_pos`` and the mode counts,
    none of which read the posterior draws; the draws were 77% of its run
    time. Library defaults are unchanged."""
    import functools

    from virgil import mode_search

    monkeypatch.setattr(
        pop.oc_cells,
        "resample_move",
        functools.partial(
            mode_search.resample_move, n_draws=n_draws, n_moves=n_moves
        ),
    )
    monkeypatch.setattr(mode_search, "N_WARM", n_warm)


MAIN_P = [
    0.0824096751078581, 0.3332734433161518, 0.14456033264572826,
    0.09721359313839946, 0.07533519955161296, 0.06108624763063713,
    0.04882012303024854, 0.04341153612611458, 0.035056212913210115,
    0.03164002900804005, 0.030070448505412014, 0.017123159026571983,
]  # fmt: skip
MAIN_LOG_Z = [
    -290.7160734682129, -290.64141002133886, -290.7109015583775,
    -290.690175309423, -290.6533275932228, -290.63794869120466,
    -290.67874525406324, -290.6413945857246, -290.72123539248304,
    -290.70570804596036, -290.6510317300779, -290.7663648905121,
]  # fmt: skip
MAIN_LOG_Z_SD = [
    0.04831127285267147, 0.04634728484028038, 0.04817265147614891,
    0.04761268217821458, 0.046652643454991816, 0.04623872854584874,
    0.04729223714106227, 0.04634642975523397, 0.04844885943232774,
    0.04803252955329319, 0.04659630983469983, 0.04967008420871414,
]  # fmt: skip
MAIN_MODES = [3, 1, 2, 2, 2, 2, 2, 1, 2, 2, 1, 1]


@pytest.mark.slow
@pytest.mark.full_only
@pytest.mark.timeout(900)
def test_position_cells_matches_main_before_the_counters_were_added(
    monkeypatch,
):
    """Refactor equivalence, not a correctness oracle: ``p_pos`` and every
    cell's ``ln Ẑ`` equal those of ``origin/main`` at
    cdaa3f82825115379e0d7e688ab21812575cf384, before the step-cap counters and
    the ``_laplace_modes`` / ``_cell_evidence`` split.

    The system is picked by a rule stated before any result was seen: the
    first system of seed 0 with finite ``p_pos`` on main and a cell with two
    or more modes, among those with K <= 6 nights and one season (a CI-cost
    restriction; larger systems take minutes). That is system 57 (K = 6).
    Mode counts drift across platforms (macOS 9 vs Linux 10 on another
    system), so they are not pinned: only that the cells with modes on main
    still have at least one. ``ln Ẑ`` agrees within
    ``4 sqrt(sd_main^2 + sd_branch^2) + 0.01``, each fit's own reported
    sd; ``p_pos`` has the same argmax and, for each cell with main's
    ``p >= DELTA``, ``|ln p - ln p_main| <= tol_i + max_j tol_j``: since
    ``ln p_i = ln m_i + ln Ẑ_i - logsumexp_j(ln m_j + ln Ẑ_j)``, the log
    normalizer moves by at most ``max_j |Δ ln Ẑ_j|``. (An earlier bound,
    TV <= 1e-3, was not derived and failed on the Linux runners at TV 0.0043
    and 0.0063.)"""
    import jax

    # the pinned values are those of the old simulator (noise-free N_OBS s²)
    # and the old ε prior, Beta(½, ½) (the rule is then Gauss–Chebyshev)
    from virgil import night_summaries as _ns

    monkeypatch.setattr(pop.oc_cells, "EPS_PRIOR", (0.5, 0.5))
    monkeypatch.setattr(_ns, "EPS_PRIOR", (0.5, 0.5))
    _few_posterior_draws(monkeypatch)
    monkeypatch.setattr(
        pop, "_noise_outliers_and_chi2",
        lambda rng, mean, outlier, ln_s, sigma: (
            mean, onp.full(len(mean), pop.N_OBS * onp.exp(2 * ln_s))
        ),
    )  # fmt: skip
    rng, _ = pop.streams(0, 57)
    nights = pop.draw_nights(rng)
    _, fit = pop.fit_domain(nights)
    pop._loguniform(rng, *pop.P_TRUE)
    sysd = pop.draw_system(rng, nights, fit)
    assert len(sysd["summaries"]) == 6
    out = pop.oc_cells.position_cells(
        sysd["summaries"], fit, jax.random.fold_in(jax.random.key(0), 57),
        jitter=0.0, e_max=pop.oc_cells.E_MAX,
    )  # fmt: skip
    assert len(out.cells) == len(MAIN_P)
    assert all(
        len(f.modes) >= 1 for f, n in zip(out.fits, MAIN_MODES) if n >= 1
    )
    lz = onp.array([f.log_z for f in out.fits])
    sd = onp.array([f.log_z_error for f in out.fits])
    tol = 4 * onp.hypot(MAIN_LOG_Z_SD, sd) + 0.01
    assert (onp.abs(lz - MAIN_LOG_Z) <= tol).all()
    p = onp.asarray(out.p_pos, float)
    assert onp.isfinite(p).all()
    assert int(p.argmax()) == int(onp.argmax(MAIN_P))
    big = onp.asarray(MAIN_P) >= pop.oc_cells.DELTA
    shift = onp.abs(onp.log(p) - onp.log(MAIN_P))
    assert (shift[big] <= (tol + tol.max())[big]).all()


def test_the_census_simulates_and_labels_in_float64(monkeypatch):
    """The census runs in the default float32 on OzSTAR, but a float32 mean
    anomaly (ulp ~ 2e-4 rad at ~2000 rad) is as coarse as σ/a of an informative
    system: ``run_population`` simulates inside float64, to the draws made with
    x64 on, and ``_labels`` runs in its space's dtype."""
    import jax
    import jax.numpy as np

    seen = []

    def run_one(sysd, fit, key, u, **kw):
        seen.append(
            (jax.config.jax_enable_x64, sysd["period"], sysd["summaries"])
        )
        return dict(status="ok")

    monkeypatch.setattr(pop, "run_one", run_one)
    with jax.enable_x64(False):
        pop.run_population(3, 0, 3, False, 0)
        assert not jax.config.jax_enable_x64  # restored
    assert seen
    with jax.enable_x64(True):
        ref = {
            d[0]["period"]: d[0]["summaries"]
            for d in map(lambda i: _draw(3, i), range(3))
            if d is not None
        }
    for x64, period, summaries in seen:
        assert x64
        for sa, sb in zip(summaries, ref[period]):
            onp.testing.assert_array_equal(sa.peaks[0].mean, sb.peaks[0].mean)

    # _labels: the state it scores is float64 whatever the ambient mode
    seen_state = []

    def peak_terms(cat, ln_s, jitter, state, dtype, between=False):
        seen_state.append((jax.config.jax_enable_x64, state.dtype))
        return np.zeros((state.shape[0], 1), dtype)

    monkeypatch.setattr(pop.oc_cells, "_peak_terms", peak_terms)
    space = types.SimpleNamespace(
        dtype=onp.float64, jitter=0.0, marginal=False,
        cat=types.SimpleNamespace(log_b=-1.0),
        decode=lambda x: dict(
            period=1.0, z=None, zeta=0.0, ln_f=0.0, ln_s=0.0, jitter=0.0
        ),
        states=lambda period, ti, zeta: onp.zeros((3, 4)),
    )  # fmt: skip
    monkeypatch.setattr(pop.oc_cells.oc, "z_to_ti", lambda z: z)
    with jax.enable_x64(False):
        pop.oc_cells._labels(None, space)
    assert seen_state == [(True, onp.dtype("float64"))]


def test_the_within_night_chi2_is_drawn_not_set():
    """Inlier nights: χ²/s² has mean ν and variance 2ν (the standard error
    of a χ²_ν sample variance uses the excess kurtosis 12/ν), not the old
    noise-free N_OBS. Outlier nights: ln s_k = ½ ln(χ²/ν) + noise is centred
    and has the spread of a log-uniform scale on ±ln S_MAX. ``draw_system``
    passes the drawn χ² on to its summaries."""
    rng = onp.random.default_rng(0)
    n, ln_s, sigma, nu = 4000, 0.3, 1e-3, pop.N_OBS - pop.N_PAR
    mean = onp.zeros((n, 5))
    mean[:, 4] = -2.0
    _, c2 = pop._noise_outliers_and_chi2(
        rng, mean, onp.zeros(n, bool), ln_s, sigma
    )
    z = c2 / onp.exp(2 * ln_s)
    assert abs(z.mean() - nu) < 4 * onp.sqrt(2 * nu / n)
    assert abs(z.var() - 2 * nu) < 4 * 2 * nu * onp.sqrt((2 + 12 / nu) / n)
    _, c2 = pop._noise_outliers_and_chi2(
        rng, mean, onp.ones(n, bool), ln_s, sigma
    )
    ln_sk = 0.5 * onp.log(c2 / nu)
    # ln s_k ~ U(±L) plus ½ ln(χ²_ν/ν) (variance ~ 1/(2ν)): mean and sd
    big = onp.log(pop.oc_cells.S_MAX)
    sd = onp.sqrt(big**2 / 3 + 1 / (2 * nu))
    assert abs(ln_sk.mean()) < 4 * sd / onp.sqrt(n)
    assert abs(ln_sk.std() / sd - 1) < 4 / onp.sqrt(2 * n) * 1.5
    nights = pop.draw_nights(rng)
    _, fit = pop.fit_domain(nights)
    while fit is None:
        nights = pop.draw_nights(rng)
        _, fit = pop.fit_domain(nights)
    sysd = pop.draw_system(rng, nights, fit)
    got = [float(s.peaks[0].chi2[0]) for s in sysd["summaries"]]
    assert all(c > 0 for c in got)
    assert not onp.allclose(got, pop.N_OBS * onp.exp(2 * sysd["ln_s"]))


def test_the_ln_s_pit_is_the_p_weighted_mixture_of_the_cells_draws(
    monkeypatch,
):
    """``_pit_lns`` weights each cell's fraction of ln s draws below the truth
    by ``p``, leaves out cells without draws or mass, renormalizes, and is
    nan with no draws at all."""
    draws = {
        1: onp.array([0.0, 1.0, 2.0, 3.0]),  # 2 of 4 below 1.5
        2: onp.array([5.0, 6.0]),  # none below 1.5
    }
    monkeypatch.setattr(
        pop.oc_cells, "decode_particles", lambda f: dict(ln_s=draws[f.i])
    )
    fits = [
        types.SimpleNamespace(draws=True, i=1),
        types.SimpleNamespace(draws=True, i=2),
        types.SimpleNamespace(draws=None, i=3),
    ]
    p = onp.array([0.2, 0.2, 0.6])
    # cell 3 has no draws: left out, the rest renormalized
    assert pop._pit_lns(p, fits, 1.5) == pytest.approx(0.5 * 0.5 + 0.0)
    assert onp.isnan(pop._pit_lns(p, [fits[2]] * 3, 1.5))
    assert onp.isnan(pop._pit_lns(onp.zeros(3), fits, 1.5))


def test_outlier_nights_are_scattered_by_their_own_scale():
    """The outlier mean's scatter grows with the night's own scale s_k (read
    off its χ²), instead of being noise-free or tied to the system's s."""
    from scipy import stats

    rng = onp.random.default_rng(3)
    n, sigma = 3000, 1e-3
    mean = onp.zeros((n, 5))
    mean[:, 4] = -2.0
    m, chi2 = pop._noise_outliers_and_chi2(
        rng, mean, onp.ones(n, bool), 0.0, sigma
    )
    assert (m[:, :4] != 0.0).all() and (chi2 > 0.0).all()
    r = onp.abs(m[:, 0]) / sigma
    assert stats.spearmanr(r, chi2).statistic > 0.3
    # at the system's single s the scatter would be the quoted one
    assert onp.sqrt(onp.mean(r**2)) > 2.0


def test_the_chi2_draw_leaves_the_pairing_structure_alone(monkeypatch):
    """Drawing the χ² last leaves every earlier draw alone: with the χ² block
    replaced by one that draws nothing, the truth and the inlier means are
    equal to the real ones, and only the outlier rows differ."""

    def one(i):
        rng, _ = pop.streams(5, i)
        nights = pop.draw_nights(rng)
        _, fit = pop.fit_domain(nights)
        if fit is None:
            return None
        pop._loguniform(rng, *pop.P_TRUE)  # the selection draw
        return pop.draw_system(rng, nights, fit)

    with_outliers = 0
    for i in range(12):
        a = one(i)
        if a is None:
            continue
        monkeypatch.setattr(
            pop, "_noise_outliers_and_chi2",
            lambda rng, mean, outlier, ln_s, sigma: (mean, onp.zeros(len(mean))),
        )  # fmt: skip
        b = one(i)
        monkeypatch.undo()
        for k in ("period", "ln_s", "eps"):
            assert a[k] == b[k]
        onp.testing.assert_array_equal(a["outliers"], b["outliers"])
        for j, (sa, sb) in enumerate(zip(a["summaries"], b["summaries"])):
            ma, mb = sa.peaks[0].mean, sb.peaks[0].mean
            if j in a["outliers"]:
                with_outliers += 1
                assert not onp.array_equal(ma, mb)
            else:
                onp.testing.assert_array_equal(ma, mb)
    assert with_outliers > 0


def test_the_fit_scores_the_drawn_chi2_without_bias_in_ln_s():
    """For a fixed state and s, simulated (m, χ²) through ``_peak_terms`` give
    a flat-prior posterior mean of ln s that is unbiased on average (the
    draw's N_OBS - N_PAR dof plus the 5 of the mean make the N_OBS of the
    likelihood)."""
    import jax
    import jax.numpy as np

    oc = pop.oc_cells
    n, ln_s0, sigma = 2000, 0.4, 1e-3
    rng = onp.random.default_rng(0)
    cov = pop._cov(sigma)
    mean = onp.exp(ln_s0) * rng.multivariate_normal(onp.zeros(5), cov, size=n)
    chi2 = onp.exp(2 * ln_s0) * rng.chisquare(pop.N_OBS - pop.N_PAR, size=n)
    summaries = pop._summaries(onp.arange(n, dtype=float), mean, sigma, chi2)
    grid = onp.linspace(ln_s0 - 1.5, ln_s0 + 1.5, 601)
    with jax.enable_x64(True):
        cat = oc._Catalogue(summaries, -5.0).cast(np.float64)
        state = np.zeros((n, 5))
        terms = onp.stack([
            onp.asarray(oc._peak_terms(cat, np.array([g]), 0.0, state, np.float64))[:, 0]
            for g in grid
        ])  # fmt: skip
    w = onp.exp(terms - terms.max(axis=0))
    post = (w * grid[:, None]).sum(0) / w.sum(0)
    assert abs(post.mean() - ln_s0) < 4 * post.std() / onp.sqrt(n)


@pytest.mark.slow
@pytest.mark.full_only
@pytest.mark.timeout(1200)
def test_the_cell_fit_is_equivariant_to_the_scale_of_the_quoted_errors():
    """The posterior does not depend on the noise scale: the same outlier-free
    data under σ·c gives the same p(N), the same inlier nights in the true
    cell, and the same ln Z in every cell the library's ``matters`` rule keeps,
    within the quoted errors (at the DELTA two-sided normal quantile). The
    search's start must not depend on the scale (it once labelled every night
    an outlier at a true ln s above 0.3 or below -1.5)."""
    from scipy import stats

    from tests import test_orbit_cells as toc
    from virgil import orbit_cells as oc_cells

    with jax.enable_x64(True):
        rng = onp.random.default_rng(7)
        times = toc._two_season_times(rng)
        orbit = toc._orbit(period=11.3, a=6.0, ecc=0.3)
        base = toc._summaries(orbit, times, sigma=0.03 * 6.0, seed=1)
        domain = oc_cells.period_domain(base, p_range=(9.0, 14.0))
        fits = {}
        for ln_c in (0.0, 1.0, -1.5):
            out = oc_cells.position_cells(
                pop.rescaled(base, onp.exp(ln_c)), domain, jax.random.key(0),
                log_outlier=onp.full(len(times), toc.LOG_B), jitter=0.0,
                a_range=(1.0, 30.0), e_max=0.9,
            )  # fmt: skip
            assert not out.failed.any()
            fits[ln_c] = out
        ref = fits[0.0]
        true_n = [
            i for i, c in enumerate(ref.cells) if c.p_lo <= 11.3 <= c.p_hi
        ][0]
        assert pop._inlier_set(ref.fits[true_n])
        z = float(stats.norm.isf(oc_cells.DELTA / 2))
        top = int(onp.argmax(ref.p_pos))
        assert top == true_n
        for ln_c in (1.0, -1.5):
            out = fits[ln_c]
            assert int(onp.argmax(out.p_pos)) == top
            assert pop._inlier_set(ref.fits[top]) == pop._inlier_set(
                out.fits[top]
            )
            e_top = onp.hypot(
                ref.fits[top].log_z_error, out.fits[top].log_z_error
            )
            # Negligible-mass wrong cells can hold different multi-night
            # subsets at different starts, beyond their quoted errors: a
            # known issue, so only the cells the library's rule keeps count.
            for i in onp.flatnonzero(ref.matters | out.matters):
                f0, f1 = ref.fits[i], out.fits[i]
                err = onp.hypot(f0.log_z_error, f1.log_z_error)
                assert abs(f0.log_z - f1.log_z) <= z * err
                # p(N) = Z_N / Σ Z: its log differs by the errors at most
                dlp = abs(onp.log(ref.p_pos[i]) - onp.log(out.p_pos[i]))
                assert dlp <= z * (err + e_top)


# --- the detectable stratum and the sky-plane scoring -----------------------


def test_nyquist_uses_the_median_gap_within_a_season_not_between_seasons():
    one = [0.0, 1.0, 2.0, 3.0, 4.0]  # median gap 1: P >= 2
    assert pop.nyquist_ok(2.0, one) and not pop.nyquist_ok(1.99, one)
    # two seasons a year apart: the 360 d gap is not a sampling gap
    two = [0.0, 10.0, 20.0, 380.0, 390.0]  # within-season gaps 10, 10, 10
    assert pop.nyquist_ok(20.0, two) and not pop.nyquist_ok(19.0, two)
    # one night a season: no within-season gap, nothing is Nyquist-sampled
    assert not pop.nyquist_ok(60.0, [0.0, 365.0])
    # the median, not the minimum: one close pair does not set the limit
    assert pop.nyquist_ok(10.0, [0.0, 0.1, 5.1, 10.1, 15.1])


def test_the_season_gap_is_the_midpoint_of_the_simulators_own_gaps():
    assert (
        pop.SEASON_LEN < pop.SEASON_GAP < pop.SEASON_SPACING - pop.SEASON_LEN
    )
    assert pop.SEASON_GAP == pytest.approx(pop.SEASON_SPACING / 2)
    rng = onp.random.default_rng(2)
    for _ in range(40):
        nights = pop.draw_nights(rng)
        gaps = onp.diff(nights["times"])
        assert (gaps[gaps <= pop.SEASON_GAP] <= pop.SEASON_LEN).all()
        assert (
            gaps[gaps > pop.SEASON_GAP] > pop.SEASON_SPACING - pop.SEASON_LEN
        ).all()
        assert nights["two_seasons"] == bool((gaps > pop.SEASON_GAP).any())


def _orbit_and_nights(period=7.0, a=8.0):
    orbit = pop.KeplerOrbit(period, 1.3, 0.4, 60.0, 30.0, 100.0, a)
    return orbit, onp.array([0.5, 2.0, 3.7, 6.1, 9.0, 12.5, 14.2, 20.0])


def test_the_signal_excess_has_the_derived_degrees_of_freedom_and_scales_as_snr():
    with jax.enable_x64(True):
        orbit, times = _orbit_and_nights()
        inl = onp.array([0, 1, 2, 4, 5, 7])
        lam, dof = pop.signal_excess(orbit, times, inl, 0.1)
        assert dof == 4 * len(inl) - 2
        lam2, _ = pop.signal_excess(orbit, times, inl, 0.05)
        assert lam2 == pytest.approx(4 * lam)  # λ ∝ 1/σ²
        assert onp.isnan(
            pop.signal_excess(orbit, times, onp.array([], int), 0.1)[0]
        )
        # only the inlier nights count: an outlier night cannot add signal
        lam3, _ = pop.signal_excess(orbit, times, inl[:3], 0.1)
        assert lam3 != lam


def test_the_signal_excess_is_the_noncentrality_of_the_static_models_chi2():
    """Monte Carlo of the derivation: under the true orbit (the simulator's
    noise) the best static model's χ² has mean dof + λ; under a static truth
    it exceeds the 1 % quantile 1 % of the time; and at the required excess
    the 1 % test has power 0.9."""
    rng = onp.random.default_rng(0)
    with jax.enable_x64(True):
        orbit, times = _orbit_and_nights(a=2.0)
        inl, sigma = onp.arange(len(times)), 0.3
        lam, dof = pop.signal_excess(orbit, times, inl, sigma)
        st = pop._night_state(orbit, times)

        def static_chi2(mu, n=4000):
            noise = rng.multivariate_normal(
                onp.zeros(4),
                sigma**2 * onp.diag([1, 1, 1 / pop.T_SS, 1 / pop.T_SS]),
                size=(n, len(times)),
            )
            d = mu + noise
            pos, vel = d[..., :2], d[..., 2:]
            return (
                ((pos - pos.mean(1, keepdims=True)) ** 2).sum((1, 2))
                + pop.T_SS * (vel**2).sum((1, 2))
            ) / sigma**2

        assert lam > 1.0
        chi2 = static_chi2(st)
        assert chi2.mean() == pytest.approx(dof + lam, rel=0.03)
        static = onp.zeros_like(st) + st[:1] * onp.array([1, 1, 0, 0])
        null = static_chi2(static)
        q = pop.stats.chi2.ppf(1 - pop.ALPHA_DET, dof)
        assert 0.003 < float((null > q).mean()) < 0.02
        # at the threshold excess the 1 % test has power POWER_DET: rescale the
        # noise so that the orbit's λ is the required excess
        need = pop.required_excess(dof)
        sigma = sigma * math.sqrt(lam / need)
        assert pop.signal_excess(orbit, times, inl, sigma)[0] == pytest.approx(
            need
        )
        power = float((static_chi2(st) > q).mean())
        assert power == pytest.approx(pop.POWER_DET, abs=0.02)


def test_the_required_excess_solves_the_noncentral_power_equation():
    for dof in (2, 6, 22, 78):
        lam = pop.required_excess(dof)
        cut = pop.stats.chi2.ppf(1 - pop.ALPHA_DET, dof)
        assert pop.stats.ncx2.sf(cut, dof, lam) == pytest.approx(
            pop.POWER_DET, abs=1e-6
        )
    # more dof needs more excess: the threshold depends on the inlier count
    assert (
        pop.required_excess(6)
        < pop.required_excess(22)
        < pop.required_excess(78)
    )


def _system(seed=0):
    rng = onp.random.default_rng(seed)
    while True:
        nights = pop.draw_nights(rng)
        _, fit = pop.fit_domain(nights)
        if fit is not None:
            return pop.draw_system(rng, nights, fit), fit


def test_a_row_carries_the_detectability_columns_from_the_truth_alone():
    with jax.enable_x64(True):
        sysd, fit = _system()
        row = pop.describe(sysd, fit)
        for k in ("nyquist_ok", "dchi2_true", "dchi2_thresh", "detectable"):
            assert k in row
        assert row["detectable"] == int(
            row["nyquist_ok"] and row["dchi2_true"] >= row["dchi2_thresh"]
        )
        # a system whose every night is an outlier has no signal to detect
        none = dict(sysd, outliers=onp.arange(len(sysd["times"])))
        assert pop.detectability(none)["detectable"] == 0
        # the threshold is the excess for power 0.9, a function of m
        m = len(sysd["times"]) - len(sysd["outliers"])
        assert row["dchi2_thresh"] == pytest.approx(
            pop.required_excess(4 * m - 2)
        )


def test_a_loud_system_is_detectable_and_a_faint_one_is_not():
    with jax.enable_x64(True):
        sysd, _ = _system()
        loud = dict(sysd, rel=1e-6, period=max(sysd["period"], 50.0))
        faint = dict(sysd, rel=1e3, period=max(sysd["period"], 50.0))
        loud["times"] = faint["times"] = onp.linspace(0.0, 20.0, 9)
        loud["outliers"] = faint["outliers"] = onp.array([], int)
        assert pop.detectability(loud)["detectable"] == 1
        assert pop.detectability(faint)["detectable"] == 0
        # Nyquist is a separate test: a loud system with too short a period fails it
        short = dict(loud, period=1.0)
        assert pop.detectability(short)["nyquist_ok"] == 0
        assert pop.detectability(short)["detectable"] == 0


def test_the_weighted_rank_is_a_randomized_pit_and_flags_the_99_quantile():
    f, w = onp.arange(100.0), onp.ones(100)
    pit, miss, q99 = pop.weighted_rank(f, w, 49.5, 0.3)
    assert pit == pytest.approx(0.5) and miss == 0 and q99 == 98.0
    assert (
        pop.weighted_rank(f, w, 99.5, 0.3)[1] == 1
    )  # above the 99 % quantile
    # ties are broken by u: all of f equal gives pit = u
    tied = pop.weighted_rank(onp.ones(10), onp.ones(10), 1.0, 0.25)
    assert tied[0] == pytest.approx(0.25) and tied[1] == 0
    # weights count: all the mass below f_true puts the rank at 1
    assert pop.weighted_rank(
        f, onp.r_[onp.ones(50), onp.zeros(50)], 80.0, 0.5
    )[0] == pytest.approx(1.0)


def _flat_space():
    """The chart's ``states_at`` (it does not use ``self``), as a stand-in
    space."""
    return types.SimpleNamespace(
        dtype=onp.float64,
        states_at=lambda t, p, ti, zt: pop.oc_cells._Space.states_at(
            None, t, p, ti, zt
        ),
    )


def _coords(orbits, t_anchor=0.0):
    cs = [pop.oc.campbell_to_coordinates(o, t_anchor) for o in orbits]
    return dict(
        period=onp.array([float(o.period) for o in orbits]),
        z=onp.stack([onp.asarray(z) for z, _ in cs]),
        zeta=onp.stack([onp.asarray(zt) for _, zt in cs]),
    )


def test_the_sky_features_are_the_simulators_plain_track():
    with jax.enable_x64(True):
        orbit, times = _orbit_and_nights()
        t = onp.linspace(0.0, 20.0, 17)
        feat = pop.sky_features(_flat_space(), _coords([orbit]), t)[0]
        want = pop._state(orbit, t)[:, :2]
        onp.testing.assert_allclose(feat, want, rtol=1e-8, atol=1e-10)
        # the label swap z -> i z negates the track: the sign is kept
        z = _coords([orbit])["z"]
        swapped = dict(
            _coords([orbit]),
            z=onp.stack([-z[:, 1], z[:, 0], -z[:, 3], z[:, 2]], -1),
        )
        onp.testing.assert_allclose(
            pop.sky_features(_flat_space(), swapped, t)[0],
            -want,
            rtol=1e-8,
            atol=1e-10,
        )


def _cloud(period0=7.0, n=60, spread=0.02, rng=None):
    """Orbits of a common shape with periods spread around ``period0``."""
    rng = onp.random.default_rng(3) if rng is None else rng
    ps = period0 * (1 + spread * rng.standard_normal(n))
    return [pop.KeplerOrbit(p, 1.3, 0.4, 60.0, 30.0, 100.0, 8.0) for p in ps]


def test_the_sky_columns_rank_the_truth_among_the_draws_and_flag_a_miss(
    monkeypatch,
):
    with jax.enable_x64(True):
        c = _coords(_cloud())
        monkeypatch.setattr(pop.oc_cells, "decode_particles", lambda f: c)
        # every draw is the +1 label: the physical track is the plain track
        monkeypatch.setattr(
            pop, "label_probability", lambda f: onp.ones(len(c["period"]))
        )
        fit = types.SimpleNamespace(draws=True, space=_flat_space())
        out = types.SimpleNamespace(fits=[fit])
        times = onp.array([0.0, 5.0, 10.0, 20.0])
        centre = pop.KeplerOrbit(7.0, 1.3, 0.4, 60.0, 30.0, 100.0, 8.0)
        sysd = dict(orbit=centre, period=7.0, times=times, sign=1.0)
        args = (out, onp.array([1.0]), 0.5, 0.0, 0)
        near = pop.sky_columns(sysd, *args)
        assert near["sky_miss"] == 0 and 0.0 <= near["pit_sky"] < 0.5
        far = pop.KeplerOrbit(7.6, 1.3, 0.4, 60.0, 30.0, 100.0, 8.0)
        bad = pop.sky_columns(dict(sysd, orbit=far, period=7.6), *args)
        assert bad["sky_miss"] == 1 and bad["pit_sky"] == pytest.approx(1.0)
        # no draws at all: no PIT, and a miss
        none = types.SimpleNamespace(fits=[types.SimpleNamespace(draws=None)])
        out0 = pop.sky_columns(sysd, none, onp.array([1.0]), 0.5, 0.0, 0)
        assert onp.isnan(out0["pit_sky"]) and out0["sky_miss"] == 1


def _real_fit(sign=1.0, twin=False, image=False, n=24):
    """A real cell space on one-peak night summaries of ``sign`` times the
    orbit (``twin``: each night's peak and its mirror image), and ``n`` posterior
    draws at the truth's chart vector (``image``: at its label-swapped image)."""
    from tests import test_orbit_cells as toc

    orbit = toc._orbit()
    s = toc._summaries(orbit, toc.TIMES, sign=sign)
    if twin:  # the mirror image of each peak: one measurement, two peaks
        mirror = onp.array([-1.0, -1.0, -1.0, -1.0, 1.0])
        s = tuple(
            dataclasses.replace(
                a,
                peaks=a.peaks
                + (
                    dataclasses.replace(
                        a.peaks[0], mean=mirror * a.peaks[0].mean
                    ),
                ),
            )
            for a in s
        )
    _, _, cell = toc._true_cell(11.3, toc.TIMES, s, (6.0, 400.0))
    space = toc._space(s, cell)
    x = toc._truth_chart(space, orbit, toc.TIMES[0])
    if image:  # the label swap z -> i z: y -> (-y1, y0, -y3, y2)
        x = x.copy()
        x[1:5] = [-x[2], x[1], -x[4], x[3]]
    rng = onp.random.default_rng(0)
    draws = x + 1e-4 * rng.standard_normal((n, len(x)))
    return orbit, toc.TIMES, types.SimpleNamespace(draws=draws, space=space)


def _sky(orbit, times, fit, sign, seed=0):
    out = types.SimpleNamespace(fits=[fit])
    sysd = dict(
        orbit=orbit, period=float(orbit.period), times=times, sign=sign
    )
    rng = onp.random.default_rng(seed)
    return pop.sky_columns(
        sysd, out, onp.array([1.0]), 0.5, times[0], 0, rng=rng
    )


def test_the_label_is_decisive_for_a_non_twin_and_flips_with_the_image():
    with jax.enable_x64(True):
        for sign in (1.0, -1.0):
            _, _, fit = _real_fit(sign=sign)
            _, _, swapped = _real_fit(sign=sign, image=True)
            p = pop.label_probability(fit)
            q = pop.label_probability(swapped)
            # the data peak is at sign * orbit: the matching label dominates
            want = 1.0 if sign > 0 else 0.0
            assert onp.all(onp.abs(p - want) < 1e-3)
            # the swap TI' -> -TI' negates the track and the label: the
            # physical track sigma * track is the same
            onp.testing.assert_allclose(q, 1.0 - p, atol=1e-6)


def test_the_label_of_a_twin_is_not_decided_by_the_data():
    with jax.enable_x64(True):
        _, _, fit = _real_fit(twin=True)
        p = pop.label_probability(fit)
        # one measurement and its mirror image: the label is a coin flip
        onp.testing.assert_allclose(p, 0.5, atol=1e-6)


def test_sky_columns_flag_a_wrong_label_truth_on_a_non_twin():
    """The truth is the simulator's ``sign * track``, and a draw's physical
    track is ``sigma_i * track(theta_i)``, with ``sigma_i`` drawn from its
    exact conditional. A non-twin with the data at ``+track``: the true sign
    is not flagged, a truth whose sign is flipped is; and the same holds
    whichever image the posterior sits in."""
    with jax.enable_x64(True):
        for image in (False, True):
            orbit, times, fit = _real_fit(sign=1.0, image=image)
            ok = _sky(orbit, times, fit, sign=1.0)
            assert ok["sky_miss"] == 0 and ok["sky_f"] < 1e-3
            bad = _sky(orbit, times, fit, sign=-1.0)
            assert bad["sky_miss"] == 1 and bad["pit_sky"] == pytest.approx(
                1.0
            )
            assert bad["sky_f"] > 10 * ok["sky_f"] + 1.0


def test_the_sky_score_of_a_twin_does_not_depend_on_the_truth_sign_or_image():
    """A twin's label is 50/50, the draws cover both tracks and the mean track
    shrinks towards zero, so the truth of either sign is inside them."""
    with jax.enable_x64(True):
        orbit, times, fit = _real_fit(twin=True, n=200)
        a = _sky(orbit, times, fit, sign=1.0)
        b = _sky(orbit, times, fit, sign=-1.0)
        assert a["sky_miss"] == 0 and b["sky_miss"] == 0
        assert 0.0 < a["pit_sky"] < 1.0 and 0.0 < b["pit_sky"] < 1.0


def test_a_bimodal_twin_posterior_shrinks_the_mean_track_but_stays_valid(
    monkeypatch,
):
    """Draws in both images (a twin): the mean track shrinks towards zero and
    f loses power, but a truth drawn from the same mixture ranks uniformly."""
    with jax.enable_x64(True):
        rng = onp.random.default_rng(7)
        t = onp.linspace(0.0, 20.0, pop.SKY_GRID)
        pits = []
        for _ in range(80):
            orbits = _cloud(n=41, spread=0.03, rng=rng)
            feat = pop.sky_features(_flat_space(), _coords(orbits), t)
            feat = feat * rng.choice([-1.0, 1.0], size=(41, 1, 1))
            mean = feat[1:].mean(0)
            f = pop.sky_distance(feat, mean)
            pits.append(
                pop.weighted_rank(f[1:], onp.ones(40), f[0], rng.uniform())[0]
            )
        assert onp.abs(mean).max() < onp.abs(feat).max()  # shrunk towards zero
        assert pop.stats.kstest(pits, "uniform").pvalue > 0.01


def test_the_sky_pit_is_uniform_when_the_truth_is_a_draw_of_the_posterior():
    """The calibration the gate relies on, in miniature: a truth and 200
    posterior draws from one distribution give a uniform rank."""
    with jax.enable_x64(True):
        rng = onp.random.default_rng(5)
        pits = []
        times = onp.array([0.0, 5.0, 10.0, 20.0])
        grid = onp.linspace(times[0], times[-1], pop.SKY_GRID)
        for _ in range(80):
            orbits = [
                pop.KeplerOrbit(
                    7.0 * (1 + 0.03 * rng.standard_normal()),
                    1.3,
                    0.4,
                    60.0,
                    30.0,
                    100.0,
                    8.0,
                )
                for _ in range(41)
            ]
            feat = pop.sky_features(_flat_space(), _coords(orbits), grid)
            mean = feat[1:].mean(0)
            f = pop.sky_distance(feat, mean)
            pits.append(
                pop.weighted_rank(f[1:], onp.ones(40), f[0], rng.uniform())[0]
            )
        assert pop.stats.kstest(pits, "uniform").pvalue > 0.01


def test_the_label_stream_leaves_the_simulation_and_pit_streams_unchanged():
    sim, pit = pop.streams(3, 5)
    first = (sim.uniform(size=3), pit.uniform(size=3))
    lab = pop.label_stream(3, 5)
    sim2, pit2 = pop.streams(3, 5)
    onp.testing.assert_array_equal(first[0], sim2.uniform(size=3))
    onp.testing.assert_array_equal(first[1], pit2.uniform(size=3))
    assert not onp.array_equal(lab.uniform(size=3), first[0])
    onp.testing.assert_array_equal(
        pop.label_stream(3, 5).uniform(size=3),
        pop.label_stream(3, 5).uniform(size=3),
    )
