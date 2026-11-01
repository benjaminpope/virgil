"""Visibility-level leg, exact outlier floor and labels (design/orbit_alias_cycles.md §2, §6.2)."""

import functools
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
    assert ok and abs(mass - 1.0) < 4 * ov._TOL and evals < ov._MAX_EVALS
    assert err <= 4 * ov._TOL * mass


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
    mass = onp.exp(table.log_g[:, :, 1:-1]).sum() * onp.prod(table.step)
    assert abs(mass - 1.0) < 0.02
    assert abs(onp.exp(table.log_gp).sum() * table.step[0] - 1.0) < 0.02


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
