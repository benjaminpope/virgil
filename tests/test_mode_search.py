"""Mode search, defensive importance sampling and resample-move, generic."""

import functools
import math

import jax
import jax.numpy as np
import numpy as onp
import pytest

pytest.importorskip("blackjax")

from virgil import mode_search as ms  # noqa: E402

from tests import _orbit_toy as toy  # noqa: E402

EXACT = {1e-2: -5.213, 1e-4: 27.028}  # K6_2s, from the toy's exact reference


def x64(test):
    """Run ``test`` in float64 (opted in locally, not globally)."""

    @functools.wraps(test)
    def wrapper(*args, **kwargs):
        with jax.enable_x64(True):
            return test(*args, **kwargs)

    return wrapper


def _lp(w):
    lw0 = -jax.nn.softplus(-w[0]) - jax.nn.softplus(w[0])
    return (
        lw0
        - 0.5 * np.sum(w[1:] ** 2) / toy.SW**2
        - 2 * np.log(2 * np.pi * toy.SW**2)
    )


def _sp(key, n):
    k0, k1 = jax.random.split(key)
    return np.column_stack(
        [
            jax.random.logistic(k0, (n,)),
            toy.SW * jax.random.normal(k1, (n, 4)),
        ]
    )


def _modes(prob):
    """The toy solver's modes as ``ms.Mode``, and its oracle profile."""
    found, _, _ = toy.search(prob)
    out = []
    for w, val, cov in found:
        c0, c1 = toy.features(onp.exp(prob.u_of_w(w[0:1])), prob.t)
        ell = toy.night_ell(c0, c1, w[None, 1:], prob.d, prob.sig)[0]
        out.append(ms.Mode(w, val, cov, float((ell > toy.LNB).sum())))
    return out


def _inliers(prob):
    def inliers(w, theta):
        u = prob._u(w[0])
        ph = 2 * np.pi * prob.t / np.exp(u)
        c0, c1 = np.cos(ph), np.sin(ph)
        dec = w[1] * c0 + w[3] * c1
        ra = w[2] * c0 + w[4] * c1
        r2 = (prob.d[:, 0] - dec) ** 2 + (prob.d[:, 1] - ra) ** 2
        ell = (
            toy.LN_IN
            - np.log(2 * np.pi * prob.sig**2)
            - r2 / (2 * prob.sig**2)
        )
        return np.sum(jax.nn.sigmoid(ell - toy.LNB))

    return inliers


def _evidence(prob, modes, key, **kw):
    mixture = ms.DefensiveMixture(modes, _lp, _sp)
    return mixture, ms.importance_evidence(
        lambda w, theta: prob._logpost_one(w), None, mixture, key,
        inliers_fn=_inliers(prob), log_floor=prob.K * toy.LNB, **kw,
    )  # fmt: skip


@x64
@pytest.mark.parametrize("rel", [1e-2, 1e-4])
def test_the_toy_evidence_matches_the_exact_reference(rel):
    prob = toy.Problem("K6_2s", rel)
    exact, profile = prob.exact()
    assert exact[prob.n_lo] == pytest.approx(EXACT[rel], abs=2e-3)
    modes = _modes(prob)
    raw = [(m.x, m.log_target, m.cov) for m in modes]
    cover, _, _ = toy.coverage(prob, raw, profile)
    assert cover >= 1 - ms.DELTA
    mixture, ev = _evidence(prob, modes, jax.random.key(0))
    assert not ev.at_floor and ev.n_nonfinite == 0
    assert ev.log_z_error < 0.05
    assert abs(ev.log_z - exact[prob.n_lo]) < 4 * ev.log_z_error + 0.01


@x64
def test_the_sentinel_fires_when_the_true_mode_is_missing():
    prob = toy.Problem("K6_2s", 1e-4)
    modes = _modes(prob)
    best = max(modes, key=lambda m: m.log_target)
    assert best.n_inliers >= ms.M_MIN
    prior_only = ms.DefensiveMixture([], _lp, _sp)
    prior_only.modes = [best]  # the search found it; q does not cover it
    floor = prob.K * toy.LNB
    ev = ms.importance_evidence(
        lambda w, theta: prob._logpost_one(w), None, prior_only,
        jax.random.key(1), inliers_fn=_inliers(prob), log_floor=floor,
    )  # fmt: skip
    assert ev.at_floor
    full, ok = _evidence(prob, modes, jax.random.key(1))
    assert not ok.at_floor


@x64
def test_distinct_modes_merges_a_mode_and_its_repeat_but_not_a_needle():
    cov = onp.diag([1.0, 1.0])
    a = ms.Mode(onp.zeros(2), 0.0, cov, 3.0)
    again = ms.Mode(onp.array([0.1, 0.0]), -0.5, cov, 3.0)
    needle = ms.Mode(onp.array([0.1, 0.0]), 5.0, 1e-6 * cov, 3.0)
    far = ms.Mode(onp.array([9.0, 0.0]), -3.0, cov, 3.0)
    out = ms.distinct_modes([a, again, needle, far])
    assert len(out) == 3 and out[0] is needle


@x64
def test_the_mixture_log_density_integrates_to_one_and_samples_it():
    modes = [ms.Mode(onp.array([1.0, -2.0]), 0.0, onp.diag([0.04, 0.09]), 2.0)]
    lp = lambda x: -0.5 * np.sum(x**2) / 9 - np.log(2 * np.pi * 9)  # noqa: E731
    sp = lambda key, n: 3 * jax.random.normal(key, (n, 2))  # noqa: E731
    q = ms.DefensiveMixture(modes, lp, sp, images=[-onp.eye(2)])
    g = onp.linspace(-14, 14, 561)
    xs = np.asarray(onp.stack(onp.meshgrid(g, g), -1).reshape(-1, 2))
    dens = onp.exp(onp.asarray(jax.vmap(q.log_q)(xs)))
    assert dens.sum() * (g[1] - g[0]) ** 2 == pytest.approx(1.0, abs=2e-3)
    x = onp.asarray(q.sample(jax.random.key(0), 20000))
    # the mixture is symmetric under the image: its sample mean is near 0
    assert abs(x.mean(0)).max() < 0.1


@x64
def test_resample_move_returns_the_posterior_of_a_gaussian_target():
    th = np.array([1.0, 2.0, -1.0])
    lp = lambda x: -0.5 * np.sum(x**2) / 25 - 1.5 * np.log(2 * np.pi * 25)  # noqa: E731
    sp = lambda key, n: 5 * jax.random.normal(key, (n, 3))  # noqa: E731

    def tgt(x, theta):
        return (
            lp(x)
            - 0.5 * np.sum((x - theta) ** 2) / 0.01
            - 1.5 * np.log(2 * np.pi * 0.01)
        )

    mode = ms.Mode(onp.asarray(th), float(tgt(th, th)), 0.01 * onp.eye(3), 3.0)
    q = ms.DefensiveMixture([mode], lp, sp)
    ev = ms.importance_evidence(
        tgt, th, q, jax.random.key(0), inliers_fn=lambda x, t: np.asarray(3.0),
        log_floor=-50.0,
    )  # fmt: skip
    assert ev.log_z == pytest.approx(float(lp(th)), abs=4 * ev.log_z_error)
    x = ms.resample_move(tgt, th, ev, jax.random.key(1))
    assert x.shape[1] == 3 and len(x) == ms.N_POST
    # the posterior is N(th * 25/25.01, ~0.01 I)
    assert onp.abs(x.mean(0) - onp.asarray(th)).max() < 0.03
    assert onp.abs(x.std(0) - 0.1).max() < 0.03
    assert math.isfinite(ev.log_z)
