"""The population script: its draws follow the fit's prior, its verdict is
calibrated to the number of fitted systems."""

import dataclasses
import json
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
    failed = [
        dict(_row(i), status="failed", pit_n=0.0, pit_lnp=0.0)
        for i in range(30)
    ]
    assert (
        pop.summarize(ok + failed, 230)[1] == "FAIL"
    )  # a failure still fails
    text, _ = pop.summarize(ok + failed, 230)
    assert "cell at floor (failed): 30" in text
    # the SBC sees the ok rows only: 30 PITs at 0 would drive p far lower
    p_ok = f"{pop.chi2_uniform([r['pit_n'] for r in ok]):.3g}"
    assert f"{pop.BINS} bins): {p_ok}" in text
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

    def peak_terms(cat, ln_s, jitter, state, dtype, extra=None):
        seen_state.append((jax.config.jax_enable_x64, state.dtype))
        return np.zeros((state.shape[0], 1), dtype)

    monkeypatch.setattr(pop.oc_cells, "_peak_terms", peak_terms)
    space = types.SimpleNamespace(
        dtype=onp.float64, jitter=0.0, cat=types.SimpleNamespace(log_b=-1.0),
        decode=lambda x: dict(period=1.0, z=None, zeta=0.0, ln_f=0.0, ln_s=0.0),
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
