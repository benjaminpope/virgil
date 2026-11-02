"""Visibility-level leg, exact outlier floor and labels (design/orbit_alias_cycles.md §2, §6.2)."""

import functools
import math
import types

import jax
import jax.numpy as np
import numpy as onp
import pytest

pytest.importorskip("jaxoplanet")
pytest.importorskip("blackjax")

from virgil import _orbit_coordinates as oc  # noqa: E402
from virgil import orbit_cells as cells  # noqa: E402
from virgil import orbit_visibility as ov  # noqa: E402
from tests.test_night_summaries import _fit, _night  # noqa: E402
from tests._shared import vlti_night  # noqa: E402
from virgil import night_summaries as ns  # noqa: E402
from virgil import mode_search  # noqa: E402
from virgil.mode_search import Mode  # noqa: E402
from virgil.orbits import KeplerOrbit  # noqa: E402
from virgil.simulate import simulate  # noqa: E402


def x64(test):
    """Run ``test`` in float64 (opted in locally, not globally)."""

    @functools.wraps(test)
    def wrapper(*args, **kwargs):
        with jax.enable_x64(True):
            return test(*args, **kwargs)

    return wrapper


DOMAIN = types.SimpleNamespace(p_cut=3.0, p_max=300.0)


@functools.cache
def _table():
    return ov.outlier_table(cells.outlier_prior(DOMAIN, n=2**17))


def _gauss(sigma):
    def log_f(u):
        n = u.shape[1]
        return -0.5 * ((u / sigma) ** 2).sum(1) - n * onp.log(
            sigma * onp.sqrt(2 * onp.pi)
        )

    return log_f


@pytest.mark.parametrize("n", [2, 3, 5])
def test_cubature_integrates_a_gaussian_to_the_tolerance(n):
    mass, err, evals, ok = ov._cubature(
        _gauss(0.2), -onp.ones((1, n)), onp.ones((1, n)), 0.0
    )
    assert ok and evals < ov._MAX_EVALS
    assert err <= ov._TOL * mass  # the stopping rule's own estimate
    # the estimate is not a bound (the 2-D rule's true error is twice it); the
    # requirement on b_k is design §2.2's 0.05 nats, which the true error meets
    assert abs(onp.log(mass)) < 0.05


def test_cubature_sums_the_boxes_of_separated_peaks():
    # a blind rule can step over a narrow peak; the contract is a box per catalogued
    # peak (design §2.3), so two boxes carry both
    def log_f(u):
        a = _gauss(0.05)(u - 0.6)
        b = _gauss(0.05)(u + 0.6)
        return onp.logaddexp(a, b)

    lo = onp.array([[0.2] * 3, [-1.0] * 3])
    hi = onp.array([[1.0] * 3, [-0.2] * 3])
    mass, _, _, ok = ov._cubature(log_f, lo, hi, 0.0)
    assert ok and abs(mass - 2.0) < 1e-2


def test_overlapping_boxes_merge_to_their_bounding_box():
    lo = onp.array([[0.0, 0.0], [0.5, 0.5], [3.0, 3.0]])
    hi = onp.array([[1.0, 1.0], [1.5, 1.5], [4.0, 4.0]])
    mlo, mhi = ov._merge_boxes(lo, hi)
    assert len(mlo) == 2
    assert onp.allclose(onp.sort(mlo[:, 0]), [0.0, 3.0])
    assert onp.allclose(onp.sort(mhi[:, 0]), [1.5, 4.0])


def test_cubature_reports_a_spent_budget():
    _, _, evals, ok = ov._cubature(
        _gauss(0.01), -onp.ones((1, 5)), onp.ones((1, 5)), 0.0, max_evals=2000
    )
    assert not ok and evals >= 2000


def test_the_grid_density_integrates_to_one_and_covers_the_samples():
    prior = cells.outlier_prior(DOMAIN, n=2**17)
    table = ov.outlier_table(prior)
    s = onp.asarray(prior.samples)[:500]
    assert onp.isfinite(table.log_density(s[:, 0], s[:, 1], s[:, 2])).all()
    mass = table.dens[:, :, 1:-1].sum() * onp.prod(table.step)
    assert abs(mass - 1.0) < 0.02
    assert abs(table.dens_p.sum() * table.step[0] - 1.0) < 0.02


@x64
def test_the_exact_floor_agrees_with_the_laplace_estimate_on_a_resolved_night():
    data = _night()
    summary = _fit(data)
    exact = ov.outlier_exact([data], [summary], _table())[0]
    laplace = cells.outlier_log_density(
        (summary,), cells.outlier_prior(DOMAIN, n=2**17)
    )[0]
    # design §2.6: within 3 nats of the Laplace estimate on a clean night
    assert abs(exact - laplace) < 3.0
    assert exact < summary.peaks[0].loglike  # the prior mass makes it a floor


@x64
def test_a_night_without_a_peak_takes_its_null_likelihood():
    data = _night()
    summary = _fit(data)
    bare = summary.__class__(
        summary.t_ref,
        summary.t_ss,
        summary.n_frames,
        summary.blocks,
        (),
        summary.loglike_null,
    )
    b = ov.outlier_exact([data], [bare], _table())
    assert onp.allclose(b, summary.loglike_null)


@x64
def test_the_exact_floor_is_finite_and_above_the_null():
    data = _night()
    summary = _fit(data)
    b = ov.outlier_exact([data], [summary], _table())[0]
    assert onp.isfinite(b) and b > summary.loglike_null


@x64
def test_the_label_swap_negates_the_thiele_innes_constants_and_is_order_four():
    x = jax.random.normal(jax.random.PRNGKey(0), (7, 12))
    z = x[:, 1:5]
    swapped = ov.swap_label(x)[:, 1:5]
    ti = np.stack(oc.z_to_ti(z), axis=-1)
    assert np.allclose(np.stack(oc.z_to_ti(swapped), axis=-1), -ti)
    assert np.allclose(
        ov.swap_label(ov.swap_label(x)), x.at[:, 1:5].multiply(-1.0)
    )
    four = ov.swap_label(ov.swap_label(ov.swap_label(ov.swap_label(x))))
    assert np.allclose(four, x)
    keep = [
        0,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
    ]  # period, a, ζ, ln f, scales are untouched
    assert np.allclose(ov.swap_label(x)[:, keep], x[:, keep])


@pytest.mark.parametrize("n", [2, 3, 5])
def test_the_genz_malik_rule_is_exact_for_degree_seven_polynomials(n):
    points, w7, w5, _ = ov._genz_malik(n)
    assert onp.isclose(w7.sum(), 1.0) and onp.isclose(w5.sum(), 1.0)
    rng = onp.random.default_rng(n)
    for _ in range(5):
        powers = rng.integers(0, 8, n)
        while powers.sum() > 7:
            powers[rng.choice(onp.flatnonzero(powers))] -= 1
        exact = onp.prod(
            [0.0 if k % 2 else 1.0 / (k + 1) for k in powers]
        )  # mean of x^k over [-1, 1]
        assert onp.isclose(w7 @ onp.prod(points**powers, axis=1), exact)
    # the embedded rule is exact to degree five
    assert onp.isclose(w5 @ points[:, 0] ** 4, 1.0 / 5)


@x64
def test_the_polar_box_holds_the_peak_and_the_log_integrand_is_finite_there():
    data = _night()
    summary = _fit(data)
    peak, table = summary.peaks[0], _table()
    lo, hi = ov._polar_box(
        peak, onp.ones(len(summary.blocks)), table, True, table.ln_f_floor
    )
    mean = peak.mean
    centre = onp.array(
        [
            0.5 * onp.log(mean[0] ** 2 + mean[1] ** 2),
            onp.arctan2(mean[1], mean[0]),
            0.5 * onp.log(mean[2] ** 2 + mean[3] ** 2),
            onp.arctan2(mean[3], mean[2]),
            mean[4],
        ]
    )
    assert (lo < centre).all() and (centre < hi).all()
    nights = ov._Nights([data], [summary], None, "float64")
    log_f = ov._night_log_f(nights, table, 0, True)
    value = log_f(centre[None])[0]
    assert onp.isfinite(value) and value > summary.loglike_null
    off = centre.copy()
    off[4] = 0.5  # a flux ratio above one is outside the prior
    assert log_f(off[None])[0] == -onp.inf


# --- the visibility leg (design §6.2) -----------------------------------------------

T0 = 60000.0
N_FRAMES = 6
LN_F = float(onp.log(0.25))


def _gains_and_offsets(d):
    n = int(onp.asarray(d.vis).size)
    d = d.with_gains(telescope=0.03, modes=onp.ones((1, n)))
    return d.with_closure_offsets(triangle=0.02)


def _night_truth(orbit, t0):
    """``(p̂, v̂, ln f)`` of the orbit's least-squares line through one night's frames."""
    t = t0 + onp.arange(N_FRAMES) / 24.0
    off = t - t.mean()
    pos = onp.stack([onp.asarray(c, float) for c in orbit.relative(t)[:2]], 1)
    return onp.concatenate(
        [pos.mean(0), onp.einsum("f,fc->c", off / (off**2).sum(), pos), [LN_F]]
    ), float(t.mean())


@functools.cache
def _system(times, d_vis=0.005, d_phi=0.02, gains=False, seed=0, period=11.3):
    """Simulated nights of a binary, the nights' summaries and the catalogue stubs (a
    peak at the truth: the catalogue-completeness assumption of design §5)."""
    orbit = KeplerOrbit(period, 2.0, 0.3, 55.0, 40.0, 70.0, 6.0, t_ref=T0)
    sets, peaks = [], []
    for k, t in enumerate(times):
        truth, t_ref = _night_truth(orbit, T0 + t)
        d = vlti_night(
            T0 + t, N_FRAMES, 0.0, d_vis=d_vis, d_phi=d_phi, stations=True
        )
        if gains:
            d = _gains_and_offsets(d)
        sets.append(
            simulate(
                ns._model(np.asarray(truth), t_ref, None),
                d,
                key=jax.random.PRNGKey(seed + k),
            )
        )
        peaks.append(
            types.SimpleNamespace(
                dra=truth[:1], ddec=truth[1:2], flux=onp.exp(truth[4:5])
            )
        )
    positions = types.SimpleNamespace(
        names=tuple(map(str, range(len(times)))),
        peaks=tuple(peaks),
        primary_diam=None,
    )
    summaries = ns.night_summaries(sets, positions)
    log_b = onp.array([s.peaks[0].loglike - 25.0 for s in summaries])
    return orbit, sets, summaries, log_b


TIMES = (0.0, 3.1, 9.7, 17.2, 31.0, 44.6, 59.3, 80.1)


def _position(times, p_range, **kw):
    orbit, sets, summaries, log_b = _system(times, **kw)
    domain = cells.period_domain(summaries, p_range=p_range)
    pos = cells.position_cells(
        summaries, domain, jax.random.key(0), log_outlier=log_b, jitter=0.0
    )
    nights = ov._Nights(sets, summaries, None, "float64")
    return orbit, pos, nights, log_b


@x64
def test_the_symmetry_images_leave_the_visibility_target_invariant():
    """At the truth and small steps from it, where every night is an inlier (not at
    prior draws, where any prior-preserving map would pass)."""
    from tests.test_orbit_cells import _truth_chart

    orbit, sets, summaries, log_b = _system(TIMES[:6])
    domain = cells.period_domain(summaries, p_range=(9.0, 14.0))
    t_all = [s.t_ref for s in summaries]
    cell = [
        c
        for c in cells.make_cells(t_all[0], t_all[-1], domain)
        if c.p_lo <= 11.3 <= c.p_hi
    ][0]
    base = cells._Space(
        cells._Catalogue(summaries, log_b), cell, 0.0,
        onp.array(t_all) - t_all[0], (1.0, 30.0), 0.9, 1e-4, 0.0, np.float64,
    )  # fmt: skip
    space = ov._VisSpace(base, ov._Nights(sets, summaries, None, "float64"))
    xt = _truth_chart(base, orbit, t_all[0] - T0, ln_f=LN_F)
    xt[1:5] *= onp.sqrt(3.0)
    rng = onp.random.default_rng(0)
    x = xt + 1e-4 * rng.standard_normal((5, base.d))
    target = jax.jit(jax.vmap(ov._vis_target, in_axes=(0, None)))
    images = cells._symmetry_images(base.d)
    assert len(images) == 15
    ref = onp.asarray(target(x, space))
    assert onp.all(ref > log_b.sum() + 50.0)  # the truth is not at the floor
    for g in images:
        moved = onp.asarray(target(x @ np.asarray(g).T, space))
        onp.testing.assert_allclose(moved, ref, rtol=1e-10, atol=1e-10)
    # and the target is not flat: a step along the period changes it
    off = x.copy()
    off[:, 0] += 1.0
    assert onp.all(onp.asarray(target(off, space)) < ref - 1.0)


def _vis_space(times):
    """The truth's cell, its position-level space and the visibility space."""
    orbit, sets, summaries, log_b = _system(times)
    domain = cells.period_domain(summaries, p_range=(9.0, 14.0))
    t_all = [s.t_ref for s in summaries]
    cell = [
        c
        for c in cells.make_cells(t_all[0], t_all[-1], domain)
        if c.p_lo <= 11.3 <= c.p_hi
    ][0]
    base = cells._Space(
        cells._Catalogue(summaries, log_b), cell, 0.0,
        onp.array(t_all) - t_all[0], (1.0, 30.0), 0.9, 1e-4, 0.0, np.float64,
    )  # fmt: skip
    space = ov._VisSpace(base, ov._Nights(sets, summaries, None, "float64"))
    return orbit, base, space, t_all[0]


@pytest.mark.slow
@pytest.mark.timeout(1800)
@x64
def test_the_visibility_evidence_matches_brute_force_quadrature_on_a_slice():
    """Fix every chart coordinate but (w_P, w_a, w_f) at the truth: the vis target on
    that slice is 3-D, and its ``ln Z`` by a dense grid of ±9σ about the Laplace mode at
    0.3σ (spectrally accurate for a smooth integrand) is the reference. The grid and the
    slice are fixed here; the reference's own error is the difference from a grid of
    twice the spacing. The estimate is the defensive mixture of ``mode_search`` on the
    Newton-refined slice mode."""
    from tests.test_orbit_cells import _truth_chart

    orbit, base, space, t_anchor = _vis_space(TIMES[:6])
    free = onp.array([0, 5, 8])
    xt = _truth_chart(base, orbit, t_anchor - T0, ln_f=LN_F)
    xt[1:5] *= onp.sqrt(3.0)
    xt_j = np.asarray(xt)

    def log_pdf(u):
        return np.sum(cells._logistic_log_pdf(u))

    def target(u, theta):
        return theta.loglike(xt_j.at[free].set(u)) + log_pdf(u)

    def inliers(u, theta):
        return theta.terms(xt_j.at[free].set(u))[1]

    u0 = np.asarray(xt[free])
    u_hat, _ = cells._lm(target, u0, (space,), 30)
    hess = -jax.hessian(target)(u_hat, space)
    cov = onp.linalg.inv(onp.asarray(0.5 * (hess + hess.T)))
    chol = onp.linalg.cholesky(cov)
    mode = Mode(onp.asarray(u_hat), float(target(u_hat, space)), cov, 6.0)

    jit_target = jax.jit(jax.vmap(target, in_axes=(0, None)))

    def grid_log_z(step, half=9.0):
        g = onp.arange(-half, half + 1e-9, step)
        pts = onp.stack(onp.meshgrid(g, g, g, indexing="ij"), -1).reshape(
            -1, 3
        )
        u = onp.asarray(u_hat) + pts @ chol.T
        lt = onp.concatenate(
            [
                onp.asarray(jit_target(np.asarray(u[i : i + 50_000]), space))
                for i in range(0, len(u), 50_000)
            ]
        )
        return float(
            lt.max()
            + onp.log(onp.exp(lt - lt.max()).sum())
            + onp.log(step**3)
            + 0.5 * onp.linalg.slogdet(cov)[1]
        )

    ref, coarse = grid_log_z(0.3), grid_log_z(0.6)
    ref_err = abs(ref - coarse)
    mixture = mode_search.DefensiveMixture(
        [mode], log_pdf,
        lambda key, n: jax.random.logistic(key, (n, 3), np.float64), dim=3,
    )  # fmt: skip
    ev = mode_search.importance_evidence(
        target, space, mixture, jax.random.key(0), 20_000, inliers_fn=inliers,
        log_floor=onp.inf,
    )  # fmt: skip
    err = onp.hypot(ev.log_z_error, ref_err)
    assert abs(ev.log_z - ref) < 4.0 * err + 0.01, (ev.log_z, ref, err, ev.ess)


@pytest.mark.slow
@pytest.mark.timeout(1800)
@x64
def test_a_second_cell_of_a_system_compiles_nothing_at_visibility_level():
    times = TIMES[:6]
    orbit, sets, summaries, log_b = _system(times)
    domain = cells.period_domain(summaries, p_range=(9.0, 14.0))
    t_all = [s.t_ref for s in summaries]
    cs = cells.make_cells(t_all[0], t_all[-1], domain)
    cell = [c for c in cs if c.p_lo <= 11.3 <= c.p_hi][0]
    other = cs[cs.index(cell) + 1]
    nights = ov._Nights(sets, summaries, None, "float64")
    count = [0, False]

    def listener(event, duration, **kw):
        if count[1] and event.endswith("backend_compile_duration"):
            count[0] += 1

    jax.monitoring.register_event_duration_secs_listener(listener)
    kw = dict(
        t_anchor=t_all[0], t_nights=t_all, n_is=512, posterior_draws=False
    )
    a = cells.position_leg(summaries, log_b, cell, jax.random.key(0), **kw)
    b = cells.position_leg(
        summaries, log_b, other, jax.random.key(1), plan=a.plan, **kw
    )
    ov.visibility_leg(a, nights, jax.random.key(2), n_is=512)
    count[1] = True
    ov.visibility_leg(b, nights, jax.random.key(3), n_is=512)
    count[1] = False
    assert count[0] == 0


@pytest.mark.slow
@pytest.mark.timeout(1800)
@x64
def test_the_sentinel_fires_when_the_true_visibility_mode_is_withheld(
    monkeypatch,
):
    times = TIMES[:5]
    _, sets, summaries, log_b = _system(times)
    domain = cells.period_domain(summaries, p_range=(9.0, 14.0))
    t_all = [s.t_ref for s in summaries]
    cell = [
        c for c in cells.make_cells(t_all[0], t_all[-1], domain)
        if c.p_lo <= 11.3 <= c.p_hi
    ][0]  # fmt: skip
    leg = cells.position_leg(
        summaries, log_b, cell, jax.random.key(0), t_anchor=t_all[0],
        t_nights=t_all, n_is=256, posterior_draws=False,
    )  # fmt: skip
    nights = ov._Nights(sets, summaries, None, "float64")
    n_k = len(times)
    floor = log_b.sum() + math.lgamma(n_k + 0.5) - 0.5 * math.log(math.pi)
    floor -= math.lgamma(n_k + 1.0)

    def withheld(space, modes):
        x = onp.asarray(space.sample(jax.random.key(5), 1))[0]
        return [Mode(x, floor + 100.0, 0.01 * onp.eye(space.d), 2.0)]

    monkeypatch.setattr(ov, "_refined_modes", withheld)
    out = ov.visibility_leg(leg, nights, jax.random.key(1), n_is=2000)
    assert out.at_floor and out.failed and out.draws is None


@pytest.mark.slow
@pytest.mark.timeout(1800)
@x64
def test_the_true_cycle_count_is_in_the_99_percent_set_at_visibility_level():
    times = (0.0, 2.9, 7.1, 11.0, 365.4, 369.2, 374.8, 380.1)
    orbit, pos, nights, _ = _position(times, (9.0, 14.0))
    assert not pos.failed.any()
    legs, log_z, p, failed = ov.visibility_cells(
        pos, nights, jax.random.key(3)
    )
    assert not failed.any() and onp.all(onp.isfinite(p))
    order = onp.argsort(-p)
    keep = order[: int(onp.searchsorted(onp.cumsum(p[order]), 0.99)) + 1]
    true_n = [i for i, c in enumerate(pos.cells) if c.p_lo <= 11.3 <= c.p_hi][
        0
    ]
    assert true_n in keep
    leg = legs[true_n]
    assert not leg.at_floor and leg.evidence.n_inliers > len(times) - 1
    # the draws are spread over the posterior (not duplicates) and centred on the truth
    ln_p = onp.log(cells.decode_particles(leg)["period"])
    assert len(onp.unique(leg.draws[:, 0])) > 100
    assert abs(onp.median(ln_p) - onp.log(11.3)) < 5 * ln_p.std() + 1e-3
