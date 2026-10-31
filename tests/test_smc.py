"""Adaptive tempered SMC (virgil.smc) on toys with known evidence."""

import jax
import jax.numpy as np
import numpy as onp
import pytest

from virgil.smc import bridge_smc, tempered_smc

S0, S1 = 3.0, 0.5


def _gauss_prior(d, dtype=None):
    def logprior(x):
        return -0.5 * np.sum(x**2) / S0**2 - d * np.log(
            S0 * np.sqrt(2 * np.pi)
        )

    def init(key, n):
        return (S0 * jax.random.normal(key, (n, d))).astype(
            dtype or np.zeros(()).dtype
        )

    return logprior, init


def _gauss_like(mu):
    mu = np.asarray(mu)
    return lambda x: -0.5 * np.sum((x - mu) ** 2) / S1**2


def _log_z(mu):
    """log ∫ N(x; 0, S0²) exp(-|x-mu|²/2S1²) dx."""
    mu = onp.asarray(mu, float)
    d, v = mu.size, S0**2 + S1**2
    return 0.5 * d * onp.log(S1**2 / v) - 0.5 * mu @ mu / v


def test_gaussian_evidence():
    mu = onp.array([1.0, -2.0])
    logprior, init = _gauss_prior(2)
    r = tempered_smc(
        logprior, _gauss_like(mu), init, key=jax.random.key(0),
        n_particles=500, replicates=4,
    )  # fmt: skip
    assert not r.stopped and not r.undersampled
    assert abs(r.log_z - _log_z(mu)) < 4 * r.log_z_error + 0.02
    assert abs(float(r.weights.sum()) - 1) < 1e-4
    mean = onp.asarray(r.weights @ r.particles)
    post_mean = mu * S0**2 / (S0**2 + S1**2)
    assert onp.allclose(mean, post_mean, atol=0.1)
    assert all(b[-1] >= 1 - 1e-5 and (onp.diff(b) > 0).all() for b in r.betas)


def test_bimodal_weights_and_evidence():
    s, sep = 0.3, 4.0
    means, w = onp.array([-sep, sep]), onp.array([0.7, 0.3])

    def loglike(x):
        comp = -0.5 * ((x[0] - means) / s) ** 2 - np.log(
            s * np.sqrt(2 * np.pi)
        )
        return jax.scipy.special.logsumexp(comp + np.log(w))

    logprior, init = _gauss_prior(1)
    r = tempered_smc(
        logprior, loglike, init, key=jax.random.key(1),
        n_particles=1000, replicates=4,
    )  # fmt: skip
    assert not r.stopped
    right = float(r.weights @ (r.particles[:, 0] > 0))
    # the posterior weight of the right mode, with the prior's slight tilt
    v = S0**2 + s**2
    z = onp.sum(w * onp.exp(-0.5 * means**2 / v) / onp.sqrt(2 * onp.pi * v))
    expect = w[1] * onp.exp(-0.5 * sep**2 / v) / (z * onp.sqrt(2 * onp.pi * v))
    assert abs(right - expect) < 0.07
    assert abs(r.log_z - onp.log(z)) < 4 * r.log_z_error + 0.05


@pytest.mark.parametrize("weighted", [False, True])
def test_bridge_ratio(weighted):
    d = 2
    mu1, mu2 = onp.array([0.5, 0.0]), onp.array([-1.0, 1.5])
    logprior, _ = _gauss_prior(d)
    like1 = _gauss_like(mu1)
    prec = 1 / S0**2 + 1 / S1**2
    centre = mu1 / S1**2 / prec
    scale = (2.0 if weighted else 1.0) / onp.sqrt(prec)
    start = centre + scale * jax.random.normal(jax.random.key(5), (1000, d))
    weights = None
    if weighted:
        # a wider proposal, reweighted to prior x L1
        log_q = -0.5 * np.sum(((start - centre) / scale) ** 2, axis=1)
        log_w = jax.vmap(lambda x: logprior(x) + like1(x))(start) - log_q
        weights = np.exp(log_w - log_w.max())
    r = bridge_smc(
        logprior, like1, _gauss_like(mu2), start, weights=weights,
        key=jax.random.key(2), replicates=4,
    )  # fmt: skip
    assert not r.stopped
    truth = _log_z(mu2) - _log_z(mu1)
    assert abs(r.log_z - truth) < 4 * r.log_z_error + 0.1


@pytest.mark.slow
def test_replicate_spread_matches_error():
    mu = onp.array([1.0])
    logprior, init = _gauss_prior(1)
    runs = [
        tempered_smc(
            logprior, _gauss_like(mu), init, key=jax.random.key(10 + i),
            n_particles=200, replicates=4,
        )
        for i in range(20)
    ]  # fmt: skip
    spread = onp.std([r.log_z for r in runs], ddof=1)
    reported = onp.mean([r.log_z_error for r in runs])
    assert 0.5 < spread / reported < 2
    assert onp.isnan(
        tempered_smc(
            logprior, _gauss_like(mu), init, key=jax.random.key(0),
            n_particles=100, replicates=1,
        ).log_z_error
    )  # fmt: skip


@pytest.mark.parametrize("cap", [{"max_stages": 2}, {"max_seconds": 0.0}])
def test_cap_is_a_failure(cap):
    logprior, init = _gauss_prior(2)
    r = tempered_smc(
        logprior, _gauss_like([1.0, 1.0]), init, key=jax.random.key(0),
        n_particles=100, **cap,
    )  # fmt: skip
    assert r.stopped
    assert r.particles is None and r.weights is None
    assert onp.isnan(r.log_z)


def test_float32():
    logprior, init = _gauss_prior(2, np.float32)
    mu = onp.array([1.0, 1.0])
    r = tempered_smc(
        logprior, _gauss_like(mu), init, key=jax.random.key(3),
        n_particles=300, replicates=3,
    )  # fmt: skip
    assert r.particles.dtype == np.float32 and r.weights.dtype == np.float32
    assert onp.isfinite(r.log_z)
    assert abs(r.log_z - _log_z(mu)) < 4 * r.log_z_error + 0.05
