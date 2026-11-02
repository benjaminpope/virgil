"""Cells, period domain and the position-level search and evidence."""

import functools
import math
import time
import types

import jax
import jax.numpy as np
import numpy as onp
import pytest
from scipy import integrate, stats
from scipy.stats import qmc

pytest.importorskip("jaxoplanet")
pytest.importorskip("blackjax")

from virgil import _orbit_coordinates as oc  # noqa: E402
from virgil import orbit_cells as oc_cells  # noqa: E402
from virgil.aliases import alias_bands, period_cut  # noqa: E402
from virgil.mode_search import Mode, laplace_log_evidence  # noqa: E402
from virgil.night_summaries import NightPeak, NightSummary  # noqa: E402
from virgil.orbits import KeplerOrbit  # noqa: E402

BLOCKS = ("vis",)
N_OBS = 40.0
FRAMES = onp.arange(6) * 0.02  # one night: six frames over 0.1 d
T_SS = float(onp.sum((FRAMES - FRAMES.mean()) ** 2))


def x64(test):
    """Run ``test`` in float64 (opted in locally, not globally)."""

    @functools.wraps(test)
    def wrapper(*args, **kwargs):
        with jax.enable_x64(True):
            return test(*args, **kwargs)

    return wrapper


def _state(orbit, t):
    pos = onp.stack([onp.asarray(c, float) for c in orbit.relative(t)[:2]])
    vel = onp.stack(
        [onp.asarray(c, float) for c in orbit.relative_velocity(t)[:2]]
    )
    return onp.concatenate([pos.T, vel.T], axis=1)


def _night_state(orbit, times):
    """``(K, 4)`` ``(p̂, v̂)``: the least-squares line through ``orbit``'s
    positions at the six frames of each night, as a summary fits it."""
    off = FRAMES - FRAMES.mean()
    times = onp.asarray(times, float)
    pos = _state(orbit, (times[:, None] + off).ravel())[:, :2].reshape(
        len(times), len(off), 2
    )
    return onp.concatenate(
        [pos.mean(1), onp.einsum("f,kfc->kc", off / T_SS, pos)], axis=1
    )


def _summaries(
    orbit, times, sigma=0.05, ln_f=-2.0, seed=0, sign=1.0, sigma_f=0.05
):
    """Night summaries of ``orbit``: one peak per night at the truth's
    projection plus a draw from the quoted covariance (the Gaussian a
    visibility fit gives)."""
    rng = onp.random.default_rng(seed)
    truth = _night_state(orbit, times)
    cov = onp.diag(
        [sigma**2, sigma**2, sigma**2 / T_SS, sigma**2 / T_SS, sigma_f**2]
    )
    out = []
    for t, s in zip(times, truth):
        mean = onp.append(sign * s, ln_f) + rng.multivariate_normal(
            onp.zeros(5), cov
        )
        peak = NightPeak(
            mean, cov, -30.0, onp.array([N_OBS]), onp.array([N_OBS]),
            onp.linalg.inv(cov)[None],
        )  # fmt: skip
        out.append(NightSummary(float(t), T_SS, 6, BLOCKS, (peak,), -200.0))
    return tuple(out)


def _orbit(
    period=11.3, a=8.0, ecc=0.3, inc=55.0, omega=40.0, Omega=70.0, t0=0.0
):
    return KeplerOrbit(period, 2.0, ecc, inc, omega, Omega, a, t_ref=t0)


TIMES = onp.array([0.0, 3.1, 9.7, 17.2, 31.0, 44.6, 59.3, 80.1])
LOG_B = -30.0 - 25.0  # b_k: 25 nats below a true peak (a typical floor)


def _cells(summaries, p_range):
    domain = oc_cells.period_domain(summaries, p_range=p_range)
    return domain, oc_cells.make_cells(TIMES[0], TIMES[-1], domain)


# --- domain, anchors, cells -------------------------------------------------


def test_the_period_domain_is_p_cut_to_p_max_and_p_range_overrides_it():
    s = _summaries(_orbit(), TIMES)
    d = oc_cells.period_domain(s, a_range=(1.0, 20.0))
    assert d.p_cut == pytest.approx(2 * onp.diff(TIMES).min())
    assert not d.overridden
    sigma = min(oc_cells._sigma_p(n) for n in s)
    assert d.p_max == pytest.approx(2 * onp.pi * 20.0 * onp.ptp(TIMES) / sigma)
    d2 = oc_cells.period_domain(s, p_range=(2.0, 500.0))
    assert (d2.p_cut, d2.p_max, d2.overridden) == (2.0, 500.0, True)
    with pytest.raises(ValueError, match="empty"):
        oc_cells.period_domain(s, p_range=(5.0, 5.0))


def test_a_short_circular_period_is_inside_the_new_domain_not_the_old():
    times = onp.array([0.0, 1.0, 3.2, 7.9, 12.0, 20.4])
    truth = 2.6  # d, below the old P_night = 12.6 d, above 2 dt_min = 2 d
    s = _summaries(_orbit(period=truth, ecc=0.0), times)
    d = oc_cells.period_domain(s)
    assert d.p_cut <= truth <= d.p_max
    old = period_cut(times, oc_cells.A_RANGE[1], 0.1, resolution=3.0)
    assert old > truth  # the old rule excluded it
    cells = oc_cells.make_cells(times[0], times[-1], d)
    assert any(c.p_lo <= truth <= c.p_hi for c in cells)


def test_the_period_domain_reports_its_binding_term_and_inputs():
    # nights 0.3 d apart merge: the shortest gap is 5.85 d, not 0.3 d
    times = onp.array([0.0, 0.3, 6.0, 12.0])
    d = oc_cells.period_domain(_summaries(_orbit(), times))
    assert d.binding == "2dt_min"
    assert d.p_cut == pytest.approx(2 * 5.85)
    # with no merging the floor binds at gap_days below p_floor / 2
    times = onp.array([0.0, 0.08, 5.0])
    s = _summaries(_orbit(), times)
    f = oc_cells.period_domain(s, gap_days=0.05)
    assert f.binding == "p_floor" and f.p_cut == pytest.approx(0.2)
    assert f.a_hi == oc_cells.A_RANGE[1] and f.span == pytest.approx(5.0)
    sigma = min(oc_cells._sigma_p(n) for n in s)
    assert f.sigma_min == pytest.approx(sigma)
    # the N = 0 band is P > span of the log-uniform prior
    assert f.n0_mass == pytest.approx(
        math.log(f.p_max / 5.0) / math.log(f.p_max / f.p_cut)
    )
    assert oc_cells.period_domain(s, p_range=(1.0, 9.0)).binding == "p_range"


@x64
def test_the_summary_prediction_is_the_projection_not_the_instant_state():
    # an eccentric short-period orbit: the curvature of a night's six frames
    # biases the instantaneous state by many sigma
    a, sigma = 8.0, 1e-3 * 8.0
    orbit = _orbit(period=1.5, a=a, ecc=0.5)
    s = _summaries(orbit, TIMES, sigma=sigma)
    _, _, cell = _true_cell(1.5, TIMES, s, (1.0, 400.0))
    space = _space(s, cell)
    x = _truth_chart(space, orbit, TIMES[0])
    c = space.decode(np.asarray(x))
    ti = oc.z_to_ti(c["z"])
    fit = onp.asarray(space.states(c["period"], ti, c["zeta"]))
    inst = onp.asarray(space.states_at(space.t_k, c["period"], ti, c["zeta"]))
    want = _night_state(orbit, TIMES)
    assert onp.max(onp.abs(fit[:, :2] - want[:, :2])) < sigma
    assert onp.max(onp.abs(inst[:, :2] - want[:, :2])) > 5 * sigma
    sigma_v = sigma / math.sqrt(T_SS)
    assert onp.max(onp.abs(fit[:, 2:] - want[:, 2:])) < sigma_v


def test_the_projection_is_the_weighted_least_squares_line_at_the_mean_time():
    # offsets that do not average to zero, unequal weights
    off = (-0.03, 0.0, 0.01, 0.05, 0.08)
    w = (1.0, 3.0, 0.5, 2.0, 1.5)
    night = NightSummary(0.0, 0.01, 5, BLOCKS, (), -200.0, off, w)
    proj = night.projection()
    rng = onp.random.default_rng(0)
    pos = rng.normal(size=(5, 2))  # arbitrary positions: the map is linear
    design = onp.stack([onp.ones(5), onp.array(off)], axis=1)
    sw = onp.sqrt(onp.array(w))[:, None]
    coef = onp.linalg.lstsq(design * sw, pos * sw, rcond=None)[0]
    onp.testing.assert_allclose(proj @ pos, coef, rtol=1e-10, atol=1e-12)


@x64
def test_a_single_frame_night_predicts_the_instantaneous_state():
    orbit = _orbit(period=11.3)
    s = [
        NightSummary(n.t_ref, 0.0, 1, BLOCKS, n.peaks, n.loglike_null)
        for n in _summaries(orbit, TIMES)
    ]
    _, _, cell = _true_cell(11.3, TIMES, s, (6.0, 400.0))
    space = _space(s, cell)
    c = space.decode(np.asarray(_truth_chart(space, orbit, TIMES[0])))
    ti = oc.z_to_ti(c["z"])
    onp.testing.assert_allclose(
        space.states(c["period"], ti, c["zeta"]),
        space.states_at(space.t_k, c["period"], ti, c["zeta"]),
        rtol=1e-12, atol=1e-12,
    )  # fmt: skip


def test_the_anchors_are_the_best_night_of_the_first_and_last_quarter():
    s = list(_summaries(_orbit(), TIMES))
    first, last = oc_cells.choose_anchors(s)
    assert (first, last) == (0, 7)  # one night in each quarter: that night

    def with_sigma(night, scale):
        p = night.peaks[0]
        cov = p.cov.copy()
        cov[:2, :2] *= scale
        peak = NightPeak(p.mean, cov, p.loglike, p.n, p.chi2, p.hessian)
        return NightSummary(
            night.t_ref,
            night.t_ss,
            6,
            night.blocks,
            (peak,),
            night.loglike_null,
        )

    # a second night in the first quarter with a smaller error wins
    s[1] = with_sigma(s[1], 0.25)
    assert oc_cells.choose_anchors(s)[0] == 1
    # a night with no peak never anchors
    s[7] = NightSummary(s[7].t_ref, s[7].t_ss, 6, BLOCKS, (), -200.0)
    assert oc_cells.choose_anchors(s)[1] == 6
    with pytest.raises(ValueError):
        oc_cells.choose_anchors(s[:1])


def test_the_cells_partition_the_domain_exactly_for_any_edges():
    s = _summaries(_orbit(), TIMES)
    domain, cells = _cells(s, (6.0, 400.0))
    t = TIMES[-1] - TIMES[0]
    assert [c.n for c in cells] == [b.n for b in alias_bands([0, t], (6, 400))]
    for default in (None, "moved"):
        if default:
            edges = onp.array([c.p_lo for c in cells[:-1]])
            edges = edges * onp.linspace(0.9, 1.1, len(edges))
            cells = oc_cells.make_cells(0.0, t, domain, edges)
        assert cells[0].p_hi == domain.p_max
        assert cells[-1].p_lo == domain.p_cut
        for upper, lower in zip(cells[:-1], cells[1:]):
            assert upper.p_lo == lower.p_hi
        assert sum(onp.exp(c.log_mass) for c in cells) == pytest.approx(1.0)
    # the long-period band is N = 0 and reaches P_max
    assert cells[0].n == 0 and cells[0].p_hi == domain.p_max
    with pytest.raises(ValueError, match="edges"):
        oc_cells.make_cells(0.0, t, domain, edges[::-1])


# --- the likelihood ---------------------------------------------------------


@x64
def test_the_epsilon_nodes_integrate_polynomials_of_degree_k_exactly():
    for k in (1, 4, 7):
        n = (k + 2) // 2
        eps = oc_cells._eps_nodes(n)
        coef = onp.random.default_rng(k).normal(size=k + 1)
        poly = onp.polynomial.Polynomial(coef)
        exact = integrate.quad(
            lambda e: poly(e) * stats.beta(0.5, 0.5).pdf(e), 0, 1
        )[0]
        assert float(np.mean(poly(onp.asarray(eps)))) == pytest.approx(
            exact, rel=1e-9, abs=1e-9
        )


@x64
def test_a_peak_term_is_the_gaussian_convolved_with_the_jitter():
    s = _summaries(_orbit(), TIMES)
    cat = oc_cells._Catalogue(s, LOG_B).cast(np.float64)
    state = onp.asarray(cat.mean[:, 0]) + onp.random.default_rng(1).normal(
        size=(len(TIMES), 5)
    ) * onp.array([0.3, 0.3, 2.0, 2.0, 0.3])
    for jitter in (0.0, 0.2):
        got = onp.asarray(
            oc_cells._peak_terms(
                cat, np.zeros(1), jitter, np.asarray(state), np.float64
            )
        )[:, 0]
        for k, night in enumerate(s):
            p = night.peaks[0]
            cov = night.covariance(p, 1.0, jitter)
            # ln L_j (N(d; 0, Σ_total) / N(0; 0, Σ_data)) with unit scale
            want = (
                p.loglike
                + stats.multivariate_normal(onp.zeros(5), cov).logpdf(
                    state[k] - p.mean
                )
                - stats.multivariate_normal(onp.zeros(5), p.cov).logpdf(
                    onp.zeros(5)
                )
            )
            assert got[k] == pytest.approx(want, abs=1e-6)


@x64
def test_a_night_without_a_peak_has_a_finite_gradient():
    s = list(_summaries(_orbit(), TIMES))
    s[3] = NightSummary(s[3].t_ref, s[3].t_ss, 6, BLOCKS, (), -200.0)
    domain, cells = _cells(tuple(s), (6.0, 400.0))
    space = oc_cells._Space(
        oc_cells._Catalogue(s, LOG_B), cells[2], 0.0,
        TIMES - TIMES[0], (1.0, 30.0), 0.9, 1e-4, 0.0, np.float64,
    )  # fmt: skip
    x = space.sample(jax.random.key(0), 4)[0]
    g = jax.grad(space.loglike)(x)
    assert onp.all(onp.isfinite(onp.asarray(g)))


def _space(summaries, cell, dtype=np.float64):
    return oc_cells._Space(
        oc_cells._Catalogue(summaries, LOG_B), cell, 0.0,
        onp.array([n.t_ref for n in summaries]) - summaries[0].t_ref,
        (1.0, 30.0), 0.9, 1e-4, 0.0, dtype,
    )  # fmt: skip


@pytest.mark.slow
@x64
def test_the_prior_through_the_cells_is_the_declared_prior():
    s = _summaries(_orbit(), TIMES)
    domain, cells = _cells(s, (6.0, 400.0))
    rng = onp.random.default_rng(0)
    n = 40000
    mass = onp.array([onp.exp(c.log_mass) for c in cells])
    which = rng.choice(len(cells), size=n, p=mass)
    keys = jax.random.split(jax.random.key(1), len(cells))
    draws = {k: [] for k in ("ln_p", "cos_inc", "ln_a", "ecc", "ln_f", "ln_s")}
    for i, cell in enumerate(cells):
        m = int((which == i).sum())
        if not m:
            continue
        space = _space(s, cell)
        c = space.decode(space.sample(keys[i], m))
        inv = oc.invariants(c["period"], c["z"], c["zeta"])
        draws["ln_p"] += list(onp.log(onp.asarray(c["period"])))
        draws["cos_inc"] += list(onp.asarray(inv["cos_inc"]))
        draws["ln_a"] += list(onp.log(onp.asarray(inv["a"])))
        draws["ecc"] += list(onp.asarray(inv["ecc"]))
        draws["ln_f"] += list(onp.asarray(c["ln_f"]))
        draws["ln_s"] += list(onp.asarray(c["ln_s"])[:, 0])
    uniform = {
        "ln_p": (onp.log(6.0), onp.log(400.0)),
        "cos_inc": (-1.0, 1.0),
        "ln_a": (0.0, onp.log(30.0)),
        "ecc": (0.0, 0.9),
        "ln_f": (onp.log(1e-4), 0.0),
        "ln_s": (-onp.log(10.0), onp.log(10.0)),
    }
    for name, (lo, hi) in uniform.items():
        x = onp.asarray(draws[name])
        assert stats.kstest(x, stats.uniform(lo, hi - lo).cdf).pvalue > 1e-3, (
            name
        )


@x64
def test_the_eccentricity_chart_density_integrates_to_the_disc_measure():
    # density at the origin is 1/π, and the radial integral is 1 (the chart's
    # ζ prior is uniform on the disc, 2:1 covered by k + ih = ζ²)
    s = _summaries(_orbit(), TIMES)
    _, cells = _cells(s, (6.0, 400.0))
    space = _space(s, cells[2])
    base = onp.zeros(space.d)

    def density(r):
        x = np.asarray(base).at[6].set(r)
        return float(
            np.exp(space.logprior(x) - space.logprior(np.asarray(base)))
        )

    total = integrate.quad(lambda r: density(r) * 2 * onp.pi * r, 0, onp.inf)[
        0
    ]
    assert total == pytest.approx(onp.pi, rel=1e-6)


@x64
def test_the_evidence_agrees_between_campbell_and_the_new_coordinates():
    """Z = E_prior[L] by Sobol quadrature in Campbell elements and in
    ``(ln P, z, ζ)``, on a low-dimensional toy (three nights, scales and flux
    fixed): the maps and priors are the same measure."""
    sigma, n = 4.0, 2**17
    nights = TIMES[[0, 2, 5]]
    s = _summaries(_orbit(), nights, sigma=sigma)
    cell = oc_cells.Cell(0, 20.0, 60.0, 0.0)
    cat = oc_cells._Catalogue(s, LOG_B).cast(np.float64)
    zero = np.zeros(1)
    ln_f = float(s[0].peaks[0].mean[4])

    def ln_like(pv):
        return jax.vmap(
            lambda q: oc_cells._log_mixture(
                cat, zero, 0.0, q, ln_f, np.float64
            )
        )(pv)

    # new coordinates: (ln P, y_z, w_a, y_ζ) from a Sobol cube
    space = oc_cells._Space(
        oc_cells._Catalogue(s, LOG_B), cell, 0.0, nights - nights[0],
        (1.0, 30.0), 0.9, 1e-4, 0.0, np.float64,
    )  # fmt: skip
    u = qmc.Sobol(8, scramble=True, seed=1).random(n)
    clip = lambda v: onp.clip(v, 1e-12, 1 - 1e-12)  # noqa: E731
    ppf = stats.norm.ppf
    logit = lambda v: onp.log(clip(v) / (1 - clip(v)))  # noqa: E731
    disc = onp.sqrt(u[:, 6])
    zeta_c = onp.stack(
        [
            disc * onp.cos(2 * onp.pi * u[:, 7]),
            disc * onp.sin(2 * onp.pi * u[:, 7]),
        ],
        1,
    )
    x = onp.concatenate(
        [
            logit(u[:, :1]), ppf(clip(u[:, 1:5])), logit(u[:, 5:6]),
            zeta_c / onp.sqrt(1 - onp.sum(zeta_c**2, 1, keepdims=True)),
            onp.zeros((n, 1)), onp.zeros((n, 1)),
        ],
        axis=1,
    )  # fmt: skip
    # the y_z coordinates need 4 normals and w_a one uniform: columns above
    # use u[:, 1:5] for y_z, u[:, 5] for w_a; w_f and w_s are fixed (ln f
    # replaced by the peak's)
    c = space.decode(np.asarray(x))
    ti = oc.z_to_ti(c["z"])
    pv = jax.vmap(space.states)(c["period"], np.stack(ti, axis=-1), c["zeta"])
    new = onp.asarray(ln_like(pv))
    # Campbell elements from a Sobol cube
    v = qmc.Sobol(7, scramble=True, seed=2).random(n)
    period = onp.exp(onp.log(20.0) + onp.log(3.0) * v[:, 0])
    orbit = KeplerOrbit(
        period, v[:, 1] * period, 0.9 * v[:, 2],
        onp.degrees(onp.arccos(2 * v[:, 3] - 1)),
        360 * v[:, 4], 360 * v[:, 5],
        onp.exp(onp.log(30.0) * v[:, 6]),
    )  # fmt: skip
    t = onp.asarray(nights, float) - nights[0]
    pos, vel = jax.vmap(
        lambda o: jax.jvp(
            lambda tt: np.stack(o._relative(tt)[:2]),
            (np.asarray(t),),
            (np.ones(len(t)),),
        )
    )(orbit)
    campbell = onp.asarray(
        ln_like(
            np.concatenate([pos.transpose(0, 2, 1), vel.transpose(0, 2, 1)], 2)
        )
    )
    ln_z = lambda ll: onp.log(onp.mean(onp.exp(ll - ll.max()))) + ll.max()  # noqa: E731
    # Campbell's periastron time is uniform over the period; the new path's
    # is the anchor's mean anomaly: both uniform, equal measures
    err = (
        lambda ll: onp.std(onp.exp(ll - ll.max()))
        / onp.sqrt(n)
        / onp.mean(  # noqa: E731
            onp.exp(ll - ll.max())
        )
    )
    tol = 4 * onp.hypot(err(new), err(campbell))
    assert abs(ln_z(new) - ln_z(campbell)) < max(tol, 0.02)


# --- the search and the evidence ------------------------------------------------


def _true_cell(period, times, s, p_range):
    domain = oc_cells.period_domain(s, p_range=p_range)
    cells = oc_cells.make_cells(times[0], times[-1], domain)
    return domain, cells, [c for c in cells if c.p_lo <= period <= c.p_hi][0]


def _two_season_times(rng, k=8):
    """``k`` nights in two seasons a year apart (about 30 d each)."""
    n1 = k // 2
    return onp.sort(
        onp.r_[rng.uniform(0, 30, n1), 365.0 + rng.uniform(0, 30, k - n1)]
    ).round(3)


def _truth_chart(space, orbit, t_anchor, ln_f=-2.0):
    """The chart vector of ``orbit`` (``y_z`` at unit radius; the radius is
    free: the likelihood depends on its direction only)."""
    z, zeta = (onp.asarray(a, float) for a in oc.campbell_to_coordinates(
        orbit, t_anchor
    ))  # fmt: skip
    la = onp.log(space.a_range[0])
    da = onp.log(space.a_range[1] / space.a_range[0])

    def logit(u):
        return onp.log(u) - onp.log1p(-u)

    u_p = (onp.log(float(orbit.period)) - float(space.ln_lo)) / float(
        space.ln_hi - space.ln_lo
    )
    a = float(oc.semimajor(z))
    x = onp.zeros(space.d)
    x[0] = logit(u_p)
    x[1:5] = z / onp.linalg.norm(z)
    x[5] = logit((onp.log(a) - la) / da)
    x[6:8] = zeta / onp.sqrt(space.e_max - (zeta**2).sum())
    x[8] = logit(1.0 - ln_f / space.ln_f_floor)
    return x


def _truth_distance2(fit, orbit, t_anchor):
    """Squared Mahalanobis distance from the nearest mode to the truth (over
    the target's symmetry images, the radius of ``y_z`` set to the mode's)."""
    xt = _truth_chart(fit.space, orbit, t_anchor)
    images = [onp.eye(fit.space.d)] + oc_cells._symmetry_images(fit.space.d)
    best = onp.inf
    for m in fit.modes:
        for g in images:
            x = g @ xt
            x[1:5] *= onp.linalg.norm(m.x[1:5])
            r = m.x - x
            best = min(best, float(r @ onp.linalg.solve(m.cov, r)))
    return best


@x64
def test_the_symmetry_images_leave_the_target_invariant():
    """At points where the likelihood dominates (the truth and small steps
    from it, every night labelled an inlier), not at prior draws, where every
    night sits at the outlier floor and any prior-preserving map would pass."""
    orbit = _orbit()
    s = _summaries(orbit, TIMES)
    _, cells = _cells(s, (6.0, 400.0))
    space = _space(s, [c for c in cells if c.p_lo <= 11.3 <= c.p_hi][0])
    images = oc_cells._symmetry_images(space.d)
    assert len(images) == 15
    xt = _truth_chart(space, orbit, TIMES[0])
    xt[1:5] *= onp.sqrt(3.0)
    rng = onp.random.default_rng(0)
    x = xt + 1e-4 * rng.standard_normal((6, space.d))  # the period is a needle
    target = jax.jit(jax.vmap(oc_cells._target, in_axes=(0, None)))
    labels = jax.vmap(oc_cells._labels, in_axes=(0, None))(x, space)
    assert onp.all(onp.asarray(labels) >= 0)
    base = onp.asarray(target(x, space))
    floor = LOG_B * len(TIMES)
    assert onp.all(base > floor + 50.0)
    for g in images:
        moved = onp.asarray(target(x @ np.asarray(g).T, space))
        onp.testing.assert_allclose(moved, base, rtol=1e-10, atol=1e-10)


@x64
def test_a_second_cell_of_a_system_compiles_nothing():
    rng = onp.random.default_rng(0)
    times = _two_season_times(rng)
    orbit = _orbit(period=11.3, a=6.0, ecc=0.0)
    s = _summaries(orbit, times, sigma=0.06)
    _, cells, cell = _true_cell(11.3, times, s, (9.0, 14.0))
    other = cells[cells.index(cell) + 1]
    count = [0, False]

    def listener(event, duration, **kw):
        if count[1] and event.endswith("backend_compile_duration"):
            count[0] += 1

    jax.monitoring.register_event_duration_secs_listener(listener)
    kw = dict(
        t_anchor=times[0], t_nights=times, n_is=512, posterior_draws=False
    )
    first = oc_cells.position_leg(
        s, onp.full(len(times), LOG_B), cell, jax.random.key(0), **kw
    )
    count[1] = True
    oc_cells.position_leg(
        s, onp.full(len(times), LOG_B), other, jax.random.key(1),
        plan=first.plan, **kw,
    )  # fmt: skip
    count[1] = False
    assert count[0] == 0


@x64
def test_a_failed_cell_makes_the_probabilities_undefined(monkeypatch):
    s = _summaries(_orbit(), TIMES, sigma=0.3)
    domain, cells = _cells(s, (60.0, 400.0))
    real = oc_cells.position_leg

    def leg(summaries, log_outlier, cell, key, **kw):
        fit = types.SimpleNamespace(
            cell=cell, log_z=0.0, log_z_error=0.0, modes=[], draws=None,
            at_floor=cell.n == cells[1].n, plan=None,
        )  # fmt: skip
        fit.failed, fit.unreliable, fit.nonfinite = fit.at_floor, False, False
        fit.log_z_upper = 0.0
        return fit

    monkeypatch.setattr(oc_cells, "position_leg", leg)
    out = oc_cells.position_cells(
        s, domain, jax.random.key(0), log_outlier=onp.full(len(TIMES), LOG_B),
        jitter=0.0,
    )  # fmt: skip
    assert out.failed.sum() == 1 and onp.all(onp.isnan(out.p_pos))
    assert real is not leg


@pytest.mark.slow
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


@pytest.mark.slow
def test_the_position_fit_runs_in_float32():
    rng = onp.random.default_rng(3)
    times = _two_season_times(rng, 6)
    orbit = _orbit(period=11.3, a=6.0, ecc=0.0)
    s = _summaries(orbit, times, sigma=0.3)
    _, _, cell = _true_cell(11.3, times, s, (9.0, 14.0))
    fit = oc_cells.position_leg(
        s, onp.full(len(times), LOG_B), cell, jax.random.key(0),
        t_anchor=times[0], t_nights=times, n_is=1024, dtype="float32",
    )  # fmt: skip
    assert fit.draws.dtype == onp.float32 and onp.isfinite(fit.log_z)
    assert not fit.at_floor


@pytest.mark.slow
@x64
def test_the_position_evidence_of_a_tiny_case_matches_prior_importance(
    monkeypatch,
):
    """K = 3 nights, a circular orbit (``e_max`` -> 1e-6): the search and
    defensive evidence of the cell against 2**20 plain draws from its prior."""
    times = onp.array([0.0, 4.1, 9.3])
    orbit = _orbit(period=11.3, a=6.0, ecc=0.0)
    s = _summaries(orbit, times, sigma=0.9, seed=2)
    domain = oc_cells.period_domain(s, p_range=(8.0, 14.0))
    cells = oc_cells.make_cells(times[0], times[-1], domain)
    cell = [c for c in cells if c.p_lo <= 11.3 <= c.p_hi][0]
    log_b = onp.full(3, LOG_B)
    seen = {}
    real_mixture = oc_cells.DefensiveMixture

    def spy(modes, *args, **kwargs):
        seen["modes"], seen["images"] = len(modes), len(kwargs["images"])
        return real_mixture(modes, *args, **kwargs)

    monkeypatch.setattr(oc_cells, "DefensiveMixture", spy)
    fit = oc_cells.position_leg(
        s, log_b, cell, jax.random.key(0), t_anchor=times[0], t_nights=times,
        e_max=1e-6, n_is=20000, posterior_draws=False,
    )  # fmt: skip
    # the tolerance below (about 2 nats at this ESS) cannot tell a search from
    # none, nor dropped images (importance weights are exact either way): check
    # that the mixture has the modes and the 15 images of the symmetry group
    assert seen["modes"] >= 1 and seen["images"] == 15
    space = oc_cells._Space(
        oc_cells._Catalogue(s, log_b), cell, times[0], times - times[0],
        oc_cells.A_RANGE, 1e-6, oc_cells.FLUX_FLOOR, 0.0, np.float64,
    )  # fmt: skip
    ll = jax.jit(jax.vmap(space.loglike))
    chunks = [
        onp.asarray(ll(space.sample(jax.random.key(10 + i), 2**15)))
        for i in range(32)
    ]
    ref = onp.concatenate(chunks)
    lz = onp.log(onp.mean(onp.exp(ref - ref.max()))) + ref.max()
    w = onp.exp(ref - ref.max())
    ess = w.sum() ** 2 / (w**2).sum()
    err = onp.sqrt(1 / ess)
    assert abs(fit.log_z - lz) < 4 * onp.hypot(fit.log_z_error, err) + 0.02


# --- review hardening ---------------------------------------------------------


def test_the_anchors_differ_and_need_two_nights_with_a_peak():
    s = list(_summaries(_orbit(), TIMES))
    empty = lambda n: NightSummary(n.t_ref, n.t_ss, 6, BLOCKS, (), -200.0)  # noqa: E731
    # peaks only in the first quarter, the best of them the later night
    few = [s[0], s[1]] + [empty(n) for n in s[2:]]
    few[1] = NightSummary(
        s[1].t_ref, s[1].t_ss, 6, BLOCKS, s[1].peaks, s[1].loglike_null
    )
    first, last = oc_cells.choose_anchors(few)
    assert first != last and {first, last} == {0, 1}
    with pytest.raises(ValueError, match="two nights"):
        oc_cells.choose_anchors([s[0]] + [empty(n) for n in s[1:]])
    with pytest.raises(ValueError, match="two nights"):
        oc_cells.period_domain(
            [s[0]] + [empty(n) for n in s[1:]],
        )


def _without_rescaling(night):
    p = night.peaks[0]
    peak = NightPeak(
        p.mean, p.cov, p.loglike, p.n, p.chi2, p.hessian, rescalable=False
    )
    return NightSummary(
        night.t_ref,
        night.t_ss,
        night.n_frames,
        BLOCKS,
        (peak,),
        night.loglike_null,
    )


@x64
def test_gains_and_offsets_fix_the_error_scale_at_one():
    s = tuple(_without_rescaling(n) for n in _summaries(_orbit(), TIMES))
    domain, cells = _cells(s, (6.0, 400.0))
    prior = oc_cells.outlier_prior(domain, n=2**14)
    b = oc_cells.outlier_log_density(s, prior)  # no NotImplementedError
    assert onp.all(onp.isfinite(b))
    space = oc_cells._Space(
        oc_cells._Catalogue(s, b), cells[2], 0.0, TIMES - TIMES[0],
        (1.0, 30.0), 0.9, 1e-4, 0.0, np.float64,
    )  # fmt: skip
    c = space.decode(space.sample(jax.random.key(0), 8))
    assert onp.all(onp.asarray(c["ln_s"]) == 0.0)


@x64
def test_the_velocity_marginal_of_the_outlier_prior_integrates_the_velocity():
    # ∫ π(p, v) d²v on a (ln|v|, ψ) grid equals the position-only density
    s = _summaries(_orbit(), TIMES)
    domain, _ = _cells(s, (6.0, 400.0))
    prior = oc_cells.outlier_prior(domain, n=2**14)
    mean = onp.array([[3.0, -2.0, 0.0, 0.0, -1.0]])
    lv = onp.linspace(-12.0, 4.0, 101)
    psi = onp.linspace(-onp.pi, onp.pi, 73)
    lvg, psig = onp.meshgrid(lv, psi, indexing="ij")
    ang = psig + onp.arctan2(mean[0, 1], mean[0, 0])
    grid = onp.tile(mean, (lvg.size, 1))
    grid[:, 2] = (onp.exp(lvg) * onp.cos(ang)).ravel()
    grid[:, 3] = (onp.exp(lvg) * onp.sin(ang)).ravel()
    dens = onp.exp(prior.log_density(grid)).reshape(lvg.shape)
    # d²v = |v|² d ln|v| dψ; the density carries π(f), common to both sides
    total = onp.trapezoid(
        onp.trapezoid(dens * onp.exp(2 * lvg), psi, axis=1), lv
    )
    marginal = float(onp.exp(prior.log_density(mean, velocity=False))[0])
    assert total == pytest.approx(marginal, rel=0.05)


# --- the search plan -----------------------------------------------------------


@x64
def test_above_the_set_budget_every_span_pair_is_kept(monkeypatch):
    """The pair that spans the inliers pins the period: it is never sampled
    away (design 6.1), and each pair of nights has interior triplet fills."""
    s = _summaries(_orbit(), TIMES)
    _, cells = _cells(s, (6.0, 400.0))
    space = _space(s, cells[2])
    monkeypatch.setattr(oc_cells, "N_SET_MAX", 10)
    plan = oc_cells.search_plan(space, jax.random.key(0))
    nights, w = onp.asarray(plan["nights"]), onp.asarray(plan["w"])
    is_pair = onp.all(w[:, 2] == 0.0, axis=(1, 2))
    pairs = {(int(a), int(b)) for a, b, _ in nights[is_pair]}
    k = len(TIMES)
    assert pairs == {(a, b) for a in range(k) for b in range(a + 1, k)}
    trips = {tuple(int(i) for i in n) for n in nights[~is_pair]}
    assert all(a < c < b for a, c, b in trips)
    spans = {(a, b) for a, _, b in trips}
    assert spans == {(a, b) for a in range(k) for b in range(a + 2, k)}


@x64
def test_the_ln_p_grid_is_sized_by_the_widest_set_span():
    """``⌈K_COH Δ ν_w⌉ + 1`` nodes: a set spanning twice the anchor baseline
    still has ``K_COH`` nodes per cycle."""
    s = _summaries(_orbit(), TIMES)
    _, cells = _cells(s, (6.0, 400.0))
    space = _space(s, cells[2])
    t = TIMES[-1] - TIMES[0]
    wide = oc_cells.search_plan(space, nu_width=2.0 / t)
    narrow = oc_cells.search_plan(space, nu_width=1.0 / t)
    assert len(narrow["grid"]) == oc_cells.K_COH + 1
    assert len(wide["grid"]) == 2 * oc_cells.K_COH + 1


@pytest.mark.slow
@x64
@pytest.mark.timeout(1200)
def test_a_large_orbit_without_a_rescalable_scale_still_finds_the_true_mode():
    """Gains and offsets, ``a = 25`` mas: the old ridge-inclusive cut removed
    every true set (ln Z biased by -100 nats, no flag)."""
    rng = onp.random.default_rng(0)
    times = _two_season_times(rng)
    orbit = _orbit(period=11.3, a=25.0, ecc=0.0)
    s = tuple(
        _without_rescaling(n)
        for n in _summaries(orbit, times, sigma=25e-3, seed=1)
    )
    _, _, cell = _true_cell(11.3, times, s, (9.0, 14.0))
    fit = oc_cells.position_leg(
        s, onp.full(len(times), LOG_B), cell, jax.random.key(0),
        t_anchor=times[0], t_nights=times, n_is=4000, posterior_draws=False,
    )  # fmt: skip
    assert not fit.at_floor and fit.evidence.ess > 100
    assert fit.modes[0].n_inliers > len(times) - 0.5
    # the evidence is the true mode's (the cut bug gave -80 nats from it)
    laplace = laplace_log_evidence(fit.modes[0])
    assert abs(fit.log_z - laplace) < 3.0


# --- failure paths and robustness ----------------------------------------------


@x64
def test_a_search_that_drops_the_true_mode_fails_the_cell(monkeypatch):
    """The sentinel at orbit level: a mode with inliers far above the floor
    that the draws never reach."""
    times = TIMES[:5]
    s = _summaries(_orbit(), times, sigma=0.05)
    _, cells = _cells(s, (6.0, 400.0))
    log_b = onp.full(5, LOG_B)
    floor = log_b.sum() + math.lgamma(5.5) - 0.5 * math.log(math.pi)
    floor -= math.lgamma(6.0)

    def fake(space, key=None, *, plan=None, info=None):
        info.update(evaluations=0.0)
        x = onp.asarray(space.sample(jax.random.key(5), 1))[0]
        return [Mode(x, floor + 100.0, 0.01 * onp.eye(space.d), 2.0)]

    monkeypatch.setattr(oc_cells, "search_cell", fake)
    fit = oc_cells.position_leg(
        s, log_b, cells[2], jax.random.key(0), t_anchor=times[0],
        t_nights=times, n_is=2000, posterior_draws=False,
    )  # fmt: skip
    assert fit.at_floor and fit.failed and fit.draws is None


@x64
def test_a_non_finite_target_fails_the_cell(monkeypatch):
    times = TIMES[:5]
    s = _summaries(_orbit(), times, sigma=0.3)
    _, cells = _cells(s, (6.0, 400.0))
    real = oc_cells.importance_evidence

    def nan_half(log_target, theta, *a, **k):
        def bad(x, th):
            return np.where(x[0] > 0.0, np.nan, log_target(x, th))

        return real(bad, theta, *a, **k)

    monkeypatch.setattr(oc_cells, "importance_evidence", nan_half)
    fit = oc_cells.position_leg(
        s, onp.full(5, LOG_B), cells[2], jax.random.key(0), t_anchor=times[0],
        t_nights=times, n_is=512, posterior_draws=False,
    )  # fmt: skip
    assert fit.nonfinite and fit.failed


def test_the_time_cap_raises():
    s = _summaries(_orbit(), TIMES, sigma=0.3)
    domain, _ = _cells(s, (60.0, 400.0))
    with pytest.raises(RuntimeError, match="max_seconds"):
        oc_cells.position_cells(
            s, domain, jax.random.key(0),
            log_outlier=onp.full(len(TIMES), LOG_B), jitter=0.0,
            max_seconds=-1.0,
        )  # fmt: skip


@pytest.mark.slow
@x64
@pytest.mark.timeout(1200)
def test_pruning_does_not_change_the_evidence(monkeypatch):
    """A needle among many candidates (K = 8, e = 0.5, sigma / a = 1e-4): the
    evidence with the prune and without it agree within their errors."""
    rng = onp.random.default_rng(5 * 10**3 + 10)
    times = _two_season_times(rng)
    orbit = _orbit(period=11.3, a=6.0, ecc=0.5)
    s = _summaries(orbit, times, sigma=6e-4, seed=1)
    _, _, cell = _true_cell(11.3, times, s, (9.0, 14.0))

    def run():
        return oc_cells.position_leg(
            s, onp.full(len(times), LOG_B), cell, jax.random.key(0),
            t_anchor=times[0], t_nights=times, n_is=4000,
            posterior_draws=False,
        )  # fmt: skip

    pruned = run()
    monkeypatch.setattr(oc_cells, "W_PRUNE", float("inf"))
    full = run()
    assert len(full.modes) >= len(pruned.modes)
    tol = 4 * onp.hypot(pruned.log_z_error, full.log_z_error) + 0.02
    assert abs(pruned.log_z - full.log_z) < tol


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


@x64
@pytest.mark.parametrize("gap", [3.0, 40.0])
def test_a_low_ess_cell_fails_the_system_only_if_it_could_matter(
    monkeypatch, gap
):
    """An unreliable cell (ESS < ESS_FAIL) within W of the best cell makes the
    probabilities undefined; one far below it is information only."""
    s = _summaries(_orbit(), TIMES, sigma=0.3)
    domain, cells = _cells(s, (60.0, 400.0))
    best, odd = cells[0].n, cells[1].n

    def leg(summaries, log_outlier, cell, key, **kw):
        low = cell.n == odd
        z = -cell.log_mass - (gap if low else 0.0)
        return types.SimpleNamespace(
            cell=cell, log_z=z, log_z_upper=z, log_z_error=0.0, modes=[],
            draws=None, at_floor=False, failed=False, nonfinite=False,
            unreliable=low, plan=None,
        )  # fmt: skip

    monkeypatch.setattr(oc_cells, "position_leg", leg)
    monkeypatch.setattr(oc_cells, "_draw", lambda f, k: None)
    out = oc_cells.position_cells(
        s, domain, jax.random.key(0), log_outlier=onp.full(len(TIMES), LOG_B),
        jitter=0.0,
    )  # fmt: skip
    assert best != odd
    if gap < oc_cells.W:
        assert out.failed.sum() == 1 and onp.all(onp.isnan(out.p_pos))
    else:
        assert not out.failed.any() and onp.isclose(out.p_pos.sum(), 1.0)


# --- the speed measures are exact --------------------------------------------


@pytest.mark.slow
@x64
@pytest.mark.timeout(1200)
def test_running_cells_in_threads_changes_nothing():
    """Two seasons, K = 6: the cells' evidence, probabilities and modes with
    one worker and with two (each cell has its own key)."""
    rng = onp.random.default_rng(0)
    times = _two_season_times(rng, 6)
    orbit = _orbit(period=11.3, a=6.0, ecc=0.0)
    s = _summaries(orbit, times, sigma=0.3)
    domain = oc_cells.period_domain(s, p_range=(10.0, 12.5))
    out = [
        oc_cells.position_cells(
            s,
            domain,
            jax.random.key(0),
            log_outlier=onp.full(len(times), LOG_B),
            jitter=0.0,
            n_is=512,
            workers=w,
        )  # fmt: skip
        for w in (1, 2)
    ]
    assert len(out[0].cells) > 2
    for name in ("p_pos", "failed"):
        onp.testing.assert_array_equal(
            getattr(out[0], name), getattr(out[1], name)
        )
    for a, b in zip(*(o.fits for o in out)):
        assert a.log_z == b.log_z and len(a.modes) == len(b.modes)
        onp.testing.assert_array_equal(a.x_is, b.x_is)
        assert (a.draws is None) == (b.draws is None)
        if a.draws is not None:
            onp.testing.assert_array_equal(a.draws, b.draws)
    # and the cells of the threaded run are those of a direct call on the main
    # thread with the key the loop assigns (float64, plan of the first cell)
    log_outlier = onp.full(len(times), LOG_B)
    t_a = times[out[1].anchors[0]]
    keys = jax.random.split(jax.random.key(0), len(out[1].cells))
    for i in (1, len(out[1].cells) - 1):
        direct = oc_cells.position_leg(
            s, log_outlier, out[1].cells[i], keys[i], t_anchor=t_a,
            t_nights=times, jitter=0.0, n_is=512, posterior_draws=False,
            plan=out[1].fits[0].plan, nu_width=1.0 / out[1].t_ab,
        )  # fmt: skip
        got = out[1].fits[i]
        assert direct.log_z == got.log_z
        assert got.x_is.dtype == onp.float64
        onp.testing.assert_array_equal(direct.x_is, got.x_is)


def test_a_failing_cell_cancels_the_queued_ones():
    done = []

    def fn(i):
        done.append(i)
        if i == 1:
            raise ValueError("boom")
        time.sleep(0.05)
        return i

    with pytest.raises(ValueError, match="boom"):
        oc_cells._map(fn, list(range(200)), 2)
    assert len(done) < 20
    assert oc_cells._map(lambda i: i * i, [1, 2, 3], 2) == [1, 4, 9]
    assert oc_cells._map(lambda i: i * i, [1, 2, 3], 1) == [1, 4, 9]
