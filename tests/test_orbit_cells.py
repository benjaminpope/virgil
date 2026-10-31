"""Cells, period domain and the position-level SMC leg."""

import functools
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


def _summaries(
    orbit, times, sigma=0.05, ln_f=-2.0, seed=0, sign=1.0, sigma_f=0.05
):
    """Night summaries of ``orbit``: one peak per night at the truth plus a
    draw from the quoted covariance (the Gaussian a visibility fit gives)."""
    rng = onp.random.default_rng(seed)
    truth = _state(orbit, onp.asarray(times, float))
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
    domain = oc_cells.period_domain(
        summaries, night_length=0.1, resolution=3.0, p_range=p_range
    )
    return domain, oc_cells.make_cells(TIMES[0], TIMES[-1], domain)


# --- domain, anchors, cells -------------------------------------------------


def test_the_period_domain_is_p_cut_to_p_max_and_p_range_overrides_it():
    s = _summaries(_orbit(), TIMES)
    d = oc_cells.period_domain(
        s, night_length=0.1, resolution=3.0, a_range=(1.0, 20.0)
    )
    cut = period_cut(TIMES, 20.0, 0.1, resolution=3.0)
    assert d.p_cut == pytest.approx(cut) and not d.overridden
    sigma = min(oc_cells._sigma_p(n) for n in s)
    assert d.p_max == pytest.approx(2 * onp.pi * 20.0 * onp.ptp(TIMES) / sigma)
    d2 = oc_cells.period_domain(
        s, night_length=0.1, resolution=3.0, p_range=(2.0, 500.0)
    )
    assert (d2.p_cut, d2.p_max, d2.overridden) == (2.0, 500.0, True)
    with pytest.raises(ValueError, match="empty"):
        oc_cells.period_domain(
            s, night_length=0.1, resolution=3.0, p_range=(5.0, 5.0)
        )


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


def test_the_valley_edge_sits_between_two_peaks_and_defaults_without_mass():
    s = _summaries(_orbit(), TIMES)
    domain, cells = _cells(s, (6.0, 400.0))
    t = TIMES[-1] - TIMES[0]
    rng = onp.random.default_rng(0)
    # clusters at x = 3.0 and x = 4.0 (T / P): the valley is near 3.5
    x = onp.r_[rng.normal(3.0, 0.08, 500), rng.normal(4.0, 0.08, 500)]
    edges = oc_cells.valley_edges(x, onp.ones(x.size), cells, t)
    n3 = [i for i, c in enumerate(cells[:-1]) if c.n == 3][0]
    assert t / edges[n3] == pytest.approx(3.5, abs=0.1)
    # unvisited windows keep the default (half-integer) edge
    n1 = [i for i, c in enumerate(cells[:-1]) if c.n == 1][0]
    assert edges[n1] == pytest.approx(cells[n1].p_lo)
    assert onp.all(onp.diff(onp.r_[domain.p_max, edges, domain.p_cut]) < 0)


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


def _invariant_stats(space, v):
    c = space.decode(v)
    inv = oc.invariants(c["period"], c["z"], c["zeta"])
    return onp.stack(
        [
            onp.asarray(inv["cos_inc"]),
            onp.log(onp.asarray(inv["a"])),
            onp.log(onp.asarray(c["period"])),
        ]
    )


@x64
def test_the_luger_move_leaves_the_prior_invariant_over_many_steps():
    # At β = 0 the move targets the prior whatever the likelihood: after many
    # steps the invariants keep their distribution, and the state really moves.
    s = _summaries(_orbit(), TIMES)
    _, cells = _cells(s, (6.0, 400.0))
    space = _space(s, cells[2])
    move = oc_cells._luger_move(space)
    step = jax.jit(jax.vmap(lambda k, p: move(k, p, space.logprior)))
    x = space.sample(jax.random.key(0), 1500)
    new, moved = x, onp.zeros(1500, bool)
    for i in range(20):
        keys = jax.random.split(jax.random.key(i + 1), 1500)
        new, took = step(keys, new)
        moved |= onp.asarray(took, bool)
    assert moved.mean() > 0.5
    for a, b in zip(_invariant_stats(space, x), _invariant_stats(space, new)):
        assert stats.ks_2samp(a, b).pvalue > 1e-3


@x64
@pytest.mark.parametrize("beta", [0.5, 1.0])
def test_the_luger_move_leaves_a_tempered_target_invariant(beta):
    """Prior draws weighted by ``L^β`` sample ``π_β``: the move must not shift
    the weighted mean of any invariant, ``E_prior[w (f(x') - f(x))] = 0``."""
    s = _summaries(_orbit(), TIMES[:3], sigma=30.0, sigma_f=6.0)  # broad
    _, cells = _cells(s, (6.0, 400.0))
    space = _space(s, cells[2])
    move = oc_cells._luger_move(space)

    def logdensity(v):
        return space.logprior(v) + beta * space.loglike(v)

    n = 20000
    x = space.sample(jax.random.key(0), n)
    keys = jax.random.split(jax.random.key(1), n)
    new, took = jax.jit(jax.vmap(lambda k, v: move(k, v, logdensity)))(keys, x)
    assert float(took.mean()) > 0.05
    ll = onp.asarray(jax.jit(jax.vmap(space.loglike))(x))
    w = onp.exp(beta * (ll - ll.max()))
    assert w.sum() ** 2 / (w**2).sum() > 300  # effective sample size
    before, after = _invariant_stats(space, x), _invariant_stats(space, new)
    assert onp.array_equal(before[2], after[2])  # the move never touches P
    for f in (0, 1):  # cos i and ln a
        d = after[f] - before[f]
        mean = onp.sum(w * d) / onp.sum(w)
        err = onp.sqrt(onp.sum((w * (d - mean)) ** 2)) / onp.sum(w)
        assert abs(mean) < 4 * err


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


# --- the leg ------------------------------------------------------------------


@pytest.mark.slow
@x64
def test_the_leg_recovers_a_toy_evidence_and_the_truth():
    orbit = _orbit(period=11.3)
    s = _summaries(orbit, TIMES, sigma=0.3)
    domain, cells = _cells(s, (6.0, 400.0))
    cell = [c for c in cells if c.p_lo < 11.3 <= c.p_hi][0]
    res = oc_cells.position_leg(
        s, onp.full(len(TIMES), LOG_B), cell, jax.random.key(3),
        t_anchor=TIMES[0], t_nights=TIMES, n_particles=512,
    )  # fmt: skip
    assert not res.stopped and onp.isfinite(res.log_z)
    d = oc_cells.decode_particles(res)
    w = onp.asarray(res.weights)
    mean_p = float(onp.sum(w * onp.asarray(d["period"])))
    assert mean_p == pytest.approx(11.3, rel=0.03)


@pytest.mark.slow
def test_the_leg_runs_in_float32():
    orbit = _orbit(period=11.3, a=3.0)
    s = _summaries(orbit, TIMES[:5], sigma=0.3)
    cell = oc_cells.Cell(1, 8.0, 14.0, 0.0)
    res = oc_cells.position_leg(
        s, onp.full(5, LOG_B), cell, jax.random.key(0), t_anchor=TIMES[0],
        t_nights=TIMES[:5], n_particles=128, dtype="float32",
    )  # fmt: skip
    assert not res.stopped and res.particles.dtype == np.float32
    assert onp.isfinite(res.log_z)


@x64
def test_the_leg_fails_at_the_time_cap_instead_of_returning_a_posterior():
    s = _summaries(_orbit(), TIMES[:5], sigma=0.3)
    cell = oc_cells.Cell(1, 8.0, 14.0, 0.0)
    res = oc_cells.position_leg(
        s, onp.full(5, LOG_B), cell, jax.random.key(0), t_anchor=TIMES[0],
        t_nights=TIMES[:5], n_particles=64, max_seconds=0.0,
    )  # fmt: skip
    assert res.stopped and res.particles is None and onp.isnan(res.log_z)
    with pytest.raises(ValueError, match="stopped"):
        oc_cells.decode_particles(res)


def _gallery_like_times(rng):
    """Nights in two or three seasons with irregular gaps (random schedules)."""
    k = int(rng.choice([6, 8, 10]))
    seasons = onp.sort(rng.uniform(0, 400, 3))
    t = onp.sort(
        rng.choice(seasons, size=k) + rng.exponential(5.0, k).cumsum() % 25
    )
    return onp.unique(onp.round(t, 3))


@pytest.mark.slow
@pytest.mark.timeout(1800)
@x64
def test_the_true_cycle_count_is_in_the_99_percent_set():
    hits, runs = 0, 4
    for seed in range(runs):
        rng = onp.random.default_rng(100 + seed)
        times = _gallery_like_times(rng)
        t = times[-1] - times[0]
        period = onp.exp(rng.uniform(onp.log(t / 3.4), onp.log(t / 0.9)))
        orbit = KeplerOrbit(
            period, rng.uniform(0, period), rng.uniform(0.0, 0.7),
            onp.degrees(onp.arccos(rng.uniform(-0.9, 0.9))),
            rng.uniform(0, 360), rng.uniform(0, 360), rng.uniform(4.0, 12.0),
        )  # fmt: skip
        s = _summaries(orbit, times, sigma=0.1, seed=seed)
        domain = oc_cells.period_domain(
            s, night_length=0.1, resolution=3.0, p_range=(t / 3.6, 2 * t)
        )
        out = oc_cells.position_cells(
            s, domain, jax.random.key(seed), log_outlier=onp.full(len(times), LOG_B),
            jitter=0.0, n_particles=192, pilot=False,
        )  # fmt: skip
        order = onp.argsort(-out.p_pos)
        keep = order[
            : int(onp.searchsorted(onp.cumsum(out.p_pos[order]), 0.99)) + 1
        ]
        true_n = onp.argmax([c.p_lo <= period <= c.p_hi for c in out.cells])
        hits += int(true_n in keep)
        # the position-level cells sum to one and the dropped mass is recorded
        assert out.p_pos.sum() == pytest.approx(1.0)
        assert out.pruned_mass < 1e-3 and not out.pruned.any()
    assert hits == runs


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
            night_length=0.1,
            resolution=3.0,
        )


def test_a_non_finite_evidence_is_a_failure_not_a_probability():
    leg = types.SimpleNamespace(
        stopped=False,
        log_z=float("nan"),
        cell=oc_cells.Cell(1, 8.0, 14.0, 0.0),
    )
    with pytest.raises(RuntimeError, match="non-finite"):
        oc_cells._check(leg)


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
    prior = oc_cells.outlier_prior(domain, n=2**16)
    mean = onp.array([[3.0, -2.0, 0.0, 0.0, -1.0]])
    lv = onp.linspace(-14.0, 4.0, 241)
    psi = onp.linspace(-onp.pi, onp.pi, 181)
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
