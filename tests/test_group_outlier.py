"""The night-outlier mixture of ``virgil.likelihood``: exact against
quadrature, the closed-form Occam factor, brute-force enumeration and a
constructed corrupted group."""

import itertools

import jax
import numpy as onp
import pytest
from scipy import integrate
from scipy.special import gammaln, logsumexp

from virgil.likelihood import (
    group_outlier_mixture,
    group_outlier_probabilities,
)


@pytest.fixture(autouse=True)
def _x64():
    with jax.enable_x64(True):
        yield


def _terms(k, seed):
    rng = onp.random.default_rng(seed)
    return rng.normal(-3.0, 2.0, k), rng.normal(-6.0, 2.0, k)


@pytest.mark.parametrize("k", [1, 3, 8, 20])
def test_the_mixture_is_the_epsilon_integral(k):
    ell, bad = _terms(k, k)
    shift = max(ell.max(), bad.max())

    def density(e):  # Beta(1/2, 1/2) prior
        inner = onp.logaddexp(onp.log1p(-e) + ell, onp.log(e) + bad) - shift
        return onp.exp(inner.sum()) / (onp.pi * onp.sqrt(e * (1.0 - e)))

    # The endpoint singularity is integrable: substitute e = sin²(t).
    want, _ = integrate.quad(
        lambda t: density(onp.sin(t) ** 2) * 2.0 * onp.sin(t) * onp.cos(t),
        0.0,
        onp.pi / 2,
        epsabs=0.0,
        epsrel=1e-13,
        limit=400,
    )
    got = float(group_outlier_mixture(ell, bad)) - k * shift
    assert got == pytest.approx(onp.log(want), rel=1e-10, abs=1e-10)


@pytest.mark.parametrize("k", [1, 3, 8, 20])
def test_all_good_groups_cost_the_occam_factor(k):
    # B -> 0: L = e^{Σℓ} Γ(K + 1/2) / (Γ(K + 1) √π), the Beta(1/2, 1/2) mean
    # of (1 - ε)^K.
    ell, _ = _terms(k, 100 + k)
    got = float(group_outlier_mixture(ell, onp.full(k, -1e4)))
    occam = gammaln(k + 0.5) - gammaln(k + 1.0) - 0.5 * onp.log(onp.pi)
    assert got == pytest.approx(ell.sum() + occam, rel=1e-12)


@pytest.mark.parametrize("k", [1, 4, 8])
def test_outlier_probabilities_match_enumeration(k):
    ell, bad = _terms(k, 200 + k)

    # Σ over the 2^K inlier/outlier assignments z of ∫ ε^{|z|}
    # (1 - ε)^{K - |z|} π(ε) dε = B(|z| + 1/2, K - |z| + 1/2) / π.
    def log_beta(m):
        return (
            gammaln(m + 0.5)
            + gammaln(k - m + 0.5)
            - gammaln(k + 1.0)
            - onp.log(onp.pi)
        )

    cases = list(itertools.product([0, 1], repeat=k))
    log_w = onp.array(
        [
            log_beta(sum(z))
            + sum(b if zk else l for zk, l, b in zip(z, ell, bad))
            for z in cases
        ]
    )
    weight = onp.exp(log_w - logsumexp(log_w))
    want = onp.array(
        [sum(w for w, z in zip(weight, cases) if z[j]) for j in range(k)]
    )
    got = onp.asarray(group_outlier_probabilities(ell, bad))
    onp.testing.assert_allclose(got, want, rtol=1e-10, atol=1e-14)
    assert float(group_outlier_mixture(ell, bad)) == pytest.approx(
        logsumexp(log_w), rel=1e-12
    )


def test_one_corrupted_night_is_found_and_does_not_move_the_estimate():
    # A scalar ``mu`` measured on 9 nights with unit errors, one of them
    # corrupted by 12σ. The outlier component is a flat density over a wide
    # prior of width 200 (B = -ln 200 per night).
    rng = onp.random.default_rng(5)
    mu_true, k, bad_night = 1.5, 9, 4
    data = mu_true + rng.standard_normal(k)
    data[bad_night] += 12.0
    grid = onp.linspace(-6.0, 12.0, 4001)
    log_b = onp.full(k, -onp.log(200.0))

    def night_ell(g):
        return -0.5 * (data - g) ** 2 - 0.5 * onp.log(2 * onp.pi)

    post = onp.array(
        [float(group_outlier_mixture(night_ell(g), log_b)) for g in grid]
    )
    w = onp.exp(post - post.max())
    w /= w.sum()
    mean = float((w * grid).sum())
    sd = float(onp.sqrt((w * (grid - mean) ** 2).sum()))
    q = onp.asarray(group_outlier_probabilities(night_ell(mean), log_b))
    assert q[bad_night] > 0.9
    assert onp.delete(q, bad_night).max() < 0.1
    # The fit without that night (a plain Gaussian mean, error 1/√8).
    clean = onp.delete(data, bad_night)
    assert abs(mean - clean.mean()) < sd
