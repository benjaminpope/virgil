"""The marginalized between-night astrometric jitter of the position leg."""

import functools
import math

import jax
import jax.numpy as np
import numpy as onp
import pytest

pytest.importorskip("jaxoplanet")
pytest.importorskip("blackjax")

from tests.test_orbit_cells import (  # noqa: E402
    BLOCKS, LOG_B, N_OBS, T_SS, _night_state, _orbit,
    _two_season_times,
)  # fmt: skip
from virgil import orbit_cells as oc_cells  # noqa: E402
from virgil.night_summaries import NightPeak, NightSummary  # noqa: E402

SIGMA = 0.02  # quoted position error (mas)


def x64(test):
    @functools.wraps(test)
    def wrapper(*args, **kwargs):
        with jax.enable_x64(True):
            return test(*args, **kwargs)

    return wrapper


def _jittery(orbit, times, jitter, *, sigma=SIGMA, seed=0, gross=()):
    """One-peak summaries of ``orbit`` whose means carry the quoted noise, one
    independent ``N(0, jitter²)`` offset (mas) of each night's position and a
    100-sigma offset on the nights ``gross``."""
    rng = onp.random.default_rng(seed)
    cov = onp.diag(
        [sigma**2, sigma**2, sigma**2 / T_SS, sigma**2 / T_SS, 0.05**2]
    )
    extra = onp.array([jitter, jitter, 0.0, 0.0, 0.0])
    out = []
    for k, (t, s) in enumerate(zip(times, _night_state(orbit, times))):
        mean = (
            onp.append(s, -2.0)
            + rng.multivariate_normal(onp.zeros(5), cov)
            + extra * rng.standard_normal(5)
        )
        if k in gross:
            mean[:2] += 100.0 * sigma
        peak = NightPeak(
            mean, cov, -30.0, onp.array([N_OBS]), onp.array([N_OBS]),
            onp.linalg.inv(cov)[None],
        )  # fmt: skip
        out.append(NightSummary(float(t), T_SS, 6, BLOCKS, (peak,), -200.0))
    return tuple(out)


def _marginal_space(summaries, cell, log_b, jitter_ref=SIGMA):
    return oc_cells._Space(
        oc_cells._Catalogue(summaries, log_b),
        cell,
        0.0,
        onp.array([n.t_ref for n in summaries]) - summaries[0].t_ref,
        (1.0, 30.0), 0.9, 1e-4, jitter_ref, np.float64, True,
    )  # fmt: skip


def _setup(jitter, gross=(), k=40, seed=0):
    rng = onp.random.default_rng(seed)
    times = onp.unique(onp.sort(rng.uniform(0.0, 60.0, k).round(3)))
    orbit = _orbit(period=11.3, a=8.0)
    s = _jittery(orbit, times, jitter, seed=seed, gross=gross)
    domain = oc_cells.period_domain(s, p_range=(9.0, 14.0))
    cells = oc_cells.make_cells(times[0], times[-1], domain)
    cell = [c for c in cells if c.p_lo <= 11.3 <= c.p_hi][0]
    prior = oc_cells.outlier_prior(domain, n=2**10)
    log_b = oc_cells.outlier_log_density(s, prior)
    space = _marginal_space(s, cell, log_b)
    return orbit, s, space


def _truth_x(space, orbit, jitter):
    """The chart vector of the true orbit anchored at the first night."""
    return oc_cells.truth_chart(
        space, orbit, float(space.cat.t_ref[0]), -2.0, jitter=jitter
    )


def _scan(space, orbit, grid):
    """The marginal likelihood along the jitter coordinate at the true orbit."""
    return onp.array(
        [
            float(space.loglike(np.asarray(_truth_x(space, orbit, j))))
            for j in grid
        ]
    )


def test_the_jitter_prior_is_absolute_and_log_uniform_in_the_chart():
    """σ_J = 1 µas … 3 mas whatever the quoted errors. Integrating the chart
    prior over the jitter coordinate (the rest fixed) gives a unit mass of
    which a fraction ``f`` lies below ``σ_J = exp(lo + f (hi - lo))``."""
    assert oc_cells.JITTER_RANGE == (1e-3, 3.0)
    orbit, s, space = _setup(0.0, k=8)
    assert space.d == 9 + 1 + 1
    lo, hi = (math.log(v) for v in oc_cells.JITTER_RANGE)
    base = onp.asarray(_truth_x(space, orbit, 0.1))
    w = onp.linspace(-40.0, 40.0, 40001)
    xs = onp.tile(base, (len(w), 1))
    xs[:, -1] = w
    lp = onp.asarray(jax.vmap(space.logprior)(np.asarray(xs)))
    dens = onp.exp(lp - lp.max())
    mass = dens / onp.trapezoid(dens, w)
    ln_j = onp.log(onp.asarray(space.decode_jitter(np.asarray(xs[:, -1:]))))
    assert ln_j.min() >= lo - 1e-9 and ln_j.max() <= hi + 1e-9
    for f in (0.1, 0.5, 0.9):
        below = w[ln_j <= lo + f * (hi - lo)]
        assert onp.trapezoid(mass[w <= below.max()], below) == pytest.approx(
            f, abs=2e-3
        )
    assert float(
        space.decode_jitter(space.jitter_coordinate())
    ) == pytest.approx(SIGMA)


@x64
def test_the_marginal_likelihood_recovers_an_injected_jitter_of_five_sigma():
    j_true = 5 * SIGMA
    orbit, s, space = _setup(j_true)
    grid = onp.geomspace(*oc_cells.JITTER_RANGE, 61)
    ll = _scan(space, orbit, grid)
    post = onp.exp(ll - ll.max())
    post /= post.sum()  # log-uniform prior: uniform on this ln-grid
    med = grid[onp.searchsorted(onp.cumsum(post), 0.5)]
    lo, hi = (
        grid[onp.searchsorted(onp.cumsum(post), q)] for q in (0.005, 0.995)
    )
    assert lo < j_true < hi
    assert 0.6 * j_true < med < 1.6 * j_true


@x64
def test_ordinary_nights_with_excess_scatter_are_inliers_once_jitter_is_fitted():
    """At a µas jitter a large share of a 5-sigma-scatter data set is labelled
    outliers (the old failure); at the fitted jitter none is."""
    orbit, s, space = _setup(5 * SIGMA)
    lab = lambda j: onp.asarray(  # noqa: E731
        oc_cells._labels(
            np.asarray(_truth_x(space, orbit, j)),
            space,
        )
    )
    assert (lab(1e-3) < 0).sum() >= len(s) // 4
    assert (lab(5 * SIGMA) < 0).sum() == 0


@x64
def test_one_gross_outlier_night_is_the_only_night_flagged():
    orbit, s, space = _setup(5 * SIGMA, gross=(7,))
    x = np.asarray(_truth_x(space, orbit, 5 * SIGMA))
    assert list(
        onp.flatnonzero(onp.asarray(oc_cells._labels(x, space)) < 0)
    ) == [7]
    # and the jitter posterior is not inflated by it
    grid = onp.geomspace(*oc_cells.JITTER_RANGE, 61)
    ll = _scan(space, orbit, grid)
    assert grid[int(onp.argmax(ll))] < 3 * 5 * SIGMA


@x64
def test_fit_chi2_reports_quoted_and_total_errors():
    """χ²/N on the quoted errors is ≫ 1 with injected jitter and ≈ 1 on
    σ² + σ_J² at the fitted jitter."""
    orbit, s, space = _setup(5 * SIGMA, k=60)
    x = onp.asarray(_truth_x(space, orbit, 5 * SIGMA))
    mode = oc_cells.Mode(x, 0.0, onp.eye(len(x)), 0.0)
    fit = type("F", (), {"space": space, "modes": [mode]})()
    out = oc_cells.fit_chi2(fit, mode)
    assert out["n_inliers"] == len(s)
    assert out["quoted"] > 10 and 0.6 < out["total"] < 1.5
    assert out["jitter"] == pytest.approx(5 * SIGMA, rel=1e-6)


@pytest.mark.slow
@pytest.mark.full_only
@x64
@pytest.mark.timeout(6000)
def test_the_period_is_calibrated_with_injected_jitter_end_to_end():
    """Two seasons, jitter 5 sigma, three seeds (about 20 min each on a laptop): no
    cell is flagged, the true ln P lies in the central 99 % of the posterior
    draws and the posterior median jitter is within a factor 3 of the truth.
    Seed 0 once flagged a wrong-period cell because the C-steps held σ_J at its
    reference; they profile it now (design §6.14)."""
    j_true, n_ok, n_run = 5 * SIGMA, 0, 3
    for seed in range(n_run):
        rng = onp.random.default_rng(seed)
        times = _two_season_times(rng, 10)
        orbit = _orbit(period=11.3, a=6.0, ecc=0.2)
        s = _jittery(orbit, times, j_true, seed=seed)
        domain = oc_cells.period_domain(s, p_range=(9.0, 14.0))
        out = oc_cells.position_cells(
            s, domain, jax.random.key(seed), a_range=(1.0, 30.0), e_max=0.9,
            log_outlier=onp.full(len(times), LOG_B),
        )  # fmt: skip
        assert not out.failed.any()
        assert j_true / 3 < out.jitter < 3 * j_true
        ln_p, w = [], []
        for f, p in zip(out.fits, out.p_pos):
            if f.draws is not None:
                ln_p.append(onp.log(oc_cells.decode_particles(f)["period"]))
                w.append(onp.full(len(ln_p[-1]), p / len(ln_p[-1])))
        ln_p, w = onp.concatenate(ln_p), onp.concatenate(w)
        order = onp.argsort(ln_p)
        cum = onp.cumsum(w[order]) / w.sum()
        lo, hi = (
            ln_p[order][onp.searchsorted(cum, q)] for q in (0.005, 0.995)
        )
        n_ok += lo <= math.log(11.3) <= hi
    assert n_ok == n_run  # calibration at scale is the census jitter arm's job


@pytest.mark.slow
@x64
def test_one_cell_fit_with_marginal_jitter_recovers_the_jitter_scale():
    """The marginal ``position_leg`` of the true cell (8 nights, jitter 3 sigma)
    finishes with finite evidence and draws of ``σ_J`` inside the prior range,
    whose median is within a factor 4 of the truth."""
    j_true = 3 * SIGMA
    orbit, s, space = _setup(j_true, k=8)
    cell = space.cell
    fit = oc_cells.position_leg(
        s, onp.full(len(s), LOG_B), cell, jax.random.key(0),
        t_anchor=float(space.cat.t_ref[0]), t_nights=[n.t_ref for n in s],
        a_range=(1.0, 30.0), e_max=0.9, jitter=None, jitter_ref=SIGMA,
        n_is=512,
    )  # fmt: skip
    assert onp.isfinite(fit.log_z) and not fit.nonfinite
    draw = onp.asarray(oc_cells.decode_particles(fit)["jitter"])
    assert draw.min() >= 1e-3 and draw.max() <= 3.0
    assert j_true / 4 < onp.median(draw) < 4 * j_true


@x64
def test_newton_from_a_wrong_reference_moves_the_jitter_coordinate_to_the_truth():
    """From the true orbit with σ_J started 2x too low, the Newton refinement
    of the marginal chart (the search's last step) moves σ_J to the truth's
    scale: the jitter is a fitted coordinate, not the reference."""
    j_true = 5 * SIGMA
    orbit, s, space = _setup(j_true, k=40)
    x0 = np.asarray(_truth_x(space, orbit, j_true / 2))
    x, _, _ = oc_cells._lm(oc_cells._target, x0, (space,), 40)
    j = float(space.decode_jitter(x))
    assert j_true / 2.5 < j < 2.5 * j_true
    assert float(oc_cells._target(x, space)) > float(
        oc_cells._target(x0, space)
    )


@x64
def test_the_jitter_range_is_a_parameter_of_the_chart():
    """A wider ``jitter_range`` moves the decode and the reference coordinate
    together; the chart prior itself is unchanged."""
    orbit, s, space = _setup(0.0, k=8)
    wide = oc_cells._Space(
        space.cat, space.cell, 0.0, space.t_k, (1.0, 30.0), 0.9, 1e-4, 0.5,
        np.float64, True, (1e-3, 30.0),
    )  # fmt: skip
    assert float(wide.decode_jitter(wide.jitter_coordinate())) == (
        pytest.approx(0.5)
    )
    x = np.asarray(_truth_x(space, orbit, 0.1))
    assert float(wide.decode_jitter(x)) > float(space.decode_jitter(x))
    assert float(wide.logprior(x)) == pytest.approx(float(space.logprior(x)))


def _minimal_plan(space):
    """The plan arrays the C-step reads, at ``ln s = 0`` and the reference."""
    cat = space.cat
    total = oc_cells._peak_precision(
        cat, np.zeros(len(cat.blocks)), space.jitter, np.float64
    )[1]
    return {
        "wk": np.linalg.inv(total[..., :4, :4]), "mk": cat.mean[..., :4],
        "wf": 1.0 / total[..., 4, 4], "fm": cat.mean[..., 4],
    }  # fmt: skip


@x64
def test_the_rank_two_jitter_terms_equal_the_peak_terms():
    """The Woodbury form of ``_jitter_terms`` is ``_peak_terms`` to round-off
    at every σ_J (the grid's evaluation is the target's)."""
    j_true = 4 * SIGMA
    orbit, s, space = _setup(j_true, k=8)
    x = np.asarray(_truth_x(space, orbit, j_true))
    c, state = oc_cells._peak_state(x, space)
    parts = oc_cells._jitter_parts(space.cat, c["ln_s"], state, np.float64)
    for sigma in (1e-3, SIGMA, j_true, 1.0):
        want = oc_cells._peak_terms(
            space.cat, c["ln_s"], sigma, state, np.float64
        )
        got = oc_cells._jitter_terms(parts, sigma**2, 1.0, space.cat.valid)
        assert onp.allclose(got, want, rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("k", [7, 40])
@x64
def test_c_steps_recover_a_jitter_ten_times_above_the_reference(k):
    """The case that stalled: the search's reference σ_J is 10x below the
    truth. A C-step (profile step, then labels) moves σ_J to within 3 Fisher
    standard deviations of ln σ_J, ``1 / sqrt(2 Σ (1 - f)²)`` over the 2 position
    dofs of each night with ``f = σ² / (σ² + σ_J²)`` (σ the quoted error), and
    labels every night an inlier, which the reference did not for K = 40."""
    j_true = 5 * SIGMA
    orbit, s, space = _setup(j_true, k=k)
    space = _marginal_space(
        s, space.cell, onp.asarray(space.cat.log_b), jitter_ref=j_true / 10
    )
    xt = onp.asarray(_truth_x(space, orbit, j_true / 10))
    k = len(s)
    start = onp.asarray(oc_cells._labels(np.asarray(xt), space))
    v, lab, tail = xt[None, [0, 6, 7]], onp.zeros((1, k), int), xt[None, 9:]
    plan = _minimal_plan(space)
    for _ in range(2):
        res = oc_cells._cstep_chunk(
            space, plan, np.asarray(v), np.asarray(lab), np.asarray(tail), 30
        )
        v, lab, x = (onp.asarray(a) for a in res[:3])
        tail = x[:, 9:]
    f = SIGMA**2 / (SIGMA**2 + j_true**2)
    sd = 1.0 / math.sqrt(2.0 * 2 * k * (1.0 - f) ** 2)
    found = float(space.decode_jitter(np.asarray(x[0])))
    assert abs(math.log(found / j_true)) <= 3 * sd
    assert (lab >= 0).all()
    assert (start >= 0).sum() <= k  # the reference scoring: recorded only


@x64
def test_profile_step_labels_equal_labels_at_the_returned_chart():
    """No scoring mismatch: the labels a C-step returns are ``_labels`` of the
    chart it returns (the profiled σ_J and s), not of the reference."""
    j_true = 5 * SIGMA
    orbit, s, space = _setup(j_true, k=7)
    xt = onp.asarray(_truth_x(space, orbit, j_true / 10))
    res = oc_cells._cstep_chunk(
        space, _minimal_plan(space), np.asarray(xt[None, [0, 6, 7]]),
        np.zeros((1, len(s)), int), np.asarray(xt[None, 9:]), 20,
    )  # fmt: skip
    again = oc_cells._labels(np.asarray(res[2][0]), space)
    assert onp.array_equal(onp.asarray(res[1][0]), onp.asarray(again))


def _ref_setup(k, jitter, gross=(), seed=0):
    rng = onp.random.default_rng(seed)
    times = onp.unique(onp.sort(rng.uniform(0.0, 60.0, k).round(3)))
    s = _jittery(
        _orbit(period=11.3, a=8.0), times, jitter, seed=seed, gross=gross
    )
    domain = oc_cells.period_domain(s, p_range=(9.0, 14.0))
    return s, domain, (float(times[0]), float(times[-1]))


@x64
def test_reference_jitter_is_defined_at_small_k_and_clipped():
    """K <= 3 returns the geometric centre of the range (nothing identified);
    every K returns a finite value inside the range."""
    lo, hi = oc_cells.JITTER_RANGE
    for k in (2, 3):
        s, dom, t_ab = _ref_setup(k, 0.1)
        assert oc_cells.reference_jitter(s, dom, t_ab) == pytest.approx(
            math.sqrt(lo * hi)
        )
    for k in (5, 7):
        s, dom, t_ab = _ref_setup(k, 0.1)
        assert lo <= oc_cells.reference_jitter(s, dom, t_ab) <= hi


@x64
def test_reference_jitter_survives_an_outlier_night_and_never_leaves_the_range():
    """K = 12 with and without one gross night: the reference moves by less
    than a factor 3 (a gross night is 100 σ off, so an untrimmed refit would
    move it by a factor of order 10) and stays finite inside the range. It is
    only a start: the C-steps profile σ_J."""
    lo, hi = oc_cells.JITTER_RANGE
    j_true = 5 * SIGMA
    base = oc_cells.reference_jitter(*_ref_setup(12, j_true))
    out = oc_cells.reference_jitter(*_ref_setup(12, j_true, gross=(3,)))
    assert lo <= base <= hi and lo <= out <= hi
    assert base / 3 < out < 3 * base
