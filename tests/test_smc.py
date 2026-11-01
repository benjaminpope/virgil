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
def test_single_run_variance_is_calibrated():
    # Lee & Whiteley's estimator is unbiased for var(Ẑ)/Z²: its mean over
    # runs matches the spread of log Ẑ over independent runs.
    mu = onp.array([1.0])
    logprior, init = _gauss_prior(1)
    runs = [
        tempered_smc(
            logprior, _gauss_like(mu), init, key=jax.random.key(100 + i),
            n_particles=100,
        )
        for i in range(100)
    ]  # fmt: skip
    # The unbiased estimate is negative in some runs at this size; those are
    # flagged, and kept in the mean.
    assert not onp.isnan([r.log_z_variance for r in runs]).any()
    assert sum(r.variance_degenerate for r in runs) < 20
    log_zs = onp.array([r.log_z for r in runs])
    assert abs(log_zs.mean() - _log_z(mu)) < 0.05
    ratio = onp.mean([r.log_z_variance for r in runs]) / onp.var(
        log_zs, ddof=1
    )
    assert 0.6 < ratio < 1.6


def test_collapsed_genealogy_is_reported():
    logprior, init = _gauss_prior(1)
    r = tempered_smc(
        logprior, _gauss_like([1.0]), init, key=jax.random.key(0),
        n_particles=3,
    )  # fmt: skip
    assert not r.stopped
    assert r.variance_degenerate
    assert onp.isnan(r.log_z_error) and onp.isnan(r.log_z_variance)
    assert onp.isfinite(r.log_z)


def test_healthy_run_reports_a_positive_error():
    logprior, init = _gauss_prior(1)
    r = tempered_smc(
        logprior, _gauss_like([1.0]), init, key=jax.random.key(0),
        n_particles=500,
    )  # fmt: skip
    assert not r.variance_degenerate
    assert 0 < r.log_z_error < 0.5


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
    if jax.config.jax_enable_x64:
        # BlackJAX draws the HMC momentum in the default float width.
        pytest.skip("float32 particles are only supported with x64 off")
    logprior, init = _gauss_prior(2, np.float32)
    mu = onp.array([1.0, 1.0])
    r = tempered_smc(
        logprior, _gauss_like(mu), init, key=jax.random.key(3),
        n_particles=300, replicates=3,
    )  # fmt: skip
    assert r.particles.dtype == np.float32 and r.weights.dtype == np.float32
    assert onp.isfinite(r.log_z)
    assert abs(r.log_z - _log_z(mu)) < 4 * r.log_z_error + 0.05


def _gaussian_pair():
    """Correlated 2-d Gaussian target and the Gibbs move on its 2nd block."""
    cov = onp.array([[1.0, 0.8], [0.8, 2.0]])
    prec = np.asarray(onp.linalg.inv(cov))
    cond_sd = float(1 / onp.sqrt(prec[1, 1]))

    def logpdf(x):
        return -0.5 * x @ prec @ x

    def move(key, x, logdensity):
        # Propose the linear block from a deliberately too-wide conditional;
        # the MH correction against the true density keeps it invariant.
        k1, k2 = jax.random.split(key)
        mean = -prec[1, 0] / prec[1, 1] * x[0]
        sd = 1.3 * cond_sd

        def log_q(v):
            return -0.5 * ((v - mean) / sd) ** 2

        new = mean + sd * jax.random.normal(k1)
        prop = x.at[1].set(new)
        log_a = logdensity(prop) - logdensity(x) + log_q(x[1]) - log_q(new)
        accept = np.log(jax.random.uniform(k2)) < log_a
        return np.where(accept, prop, x), accept

    return cov, logpdf, move


def test_conditional_move_leaves_target_invariant():
    cov, logpdf, move = _gaussian_pair()
    x = jax.random.multivariate_normal(
        jax.random.key(0), np.zeros(2), cov, (20000,)
    )
    keys = jax.random.split(jax.random.key(1), 20000)
    new, accepted = jax.vmap(lambda k, p: move(k, p, logpdf))(keys, x)
    assert 0.5 < float(accepted.mean()) < 0.99
    assert onp.allclose(onp.cov(onp.asarray(new).T), cov, atol=0.08)
    assert onp.allclose(onp.asarray(new).mean(0), 0, atol=0.03)


def test_smc_with_conditional_move():
    cov, _, move = _gaussian_pair()
    prec = onp.linalg.inv(cov)
    mu = onp.array([1.0, -1.0])

    def logprior(x):
        return -0.5 * x @ x / S0**2 - 2 * np.log(S0 * np.sqrt(2 * np.pi))

    def loglike(x):
        return -0.5 * (x - mu) @ prec @ (x - mu)

    def init(key, n):
        return S0 * jax.random.normal(key, (n, 2))

    # exact: ∫ N(x; 0, S0² I) exp(-½(x-μ)ᵀP(x-μ)) dx
    a = onp.eye(2) / S0**2 + prec
    truth = (
        -0.5 * mu @ prec @ mu
        + 0.5 * (prec @ mu) @ onp.linalg.solve(a, prec @ mu)
        - 0.5 * onp.linalg.slogdet(a)[1]
        - onp.log(S0**2)
    )
    r = tempered_smc(
        logprior, loglike, init, key=jax.random.key(4), n_particles=500,
        replicates=4, moves=[move],
    )  # fmt: skip
    assert not r.stopped
    assert abs(r.log_z - truth) < 4 * r.log_z_error + 0.1
    post_mean = onp.linalg.solve(a, prec @ mu)
    assert onp.allclose(
        onp.asarray(r.weights @ r.particles), post_mean, atol=0.1
    )


def test_discrete_labels_match_exact_frequencies():
    y = onp.array([-1.6, -0.4, 0.3, 1.4, 2.0])
    shift, w, sd = 1.5, onp.array([0.6, 0.4]), 1.0
    comp = np.array([-shift, shift])
    logw = np.log(np.asarray(w))

    def logprior(p):
        mu_prior = -0.5 * p["x"][0] ** 2 / S0**2 - np.log(
            S0 * np.sqrt(2 * np.pi)
        )
        return mu_prior + np.sum(logw[p["z"]])

    def loglike(p):
        r = (y - p["x"][0] - comp[p["z"]]) / sd
        return np.sum(-0.5 * r**2 - np.log(sd * np.sqrt(2 * np.pi)))

    def init(key, n):
        k1, k2 = jax.random.split(key)
        z = (jax.random.uniform(k2, (n, y.size)) < w[1]).astype(int)
        return {"x": S0 * jax.random.normal(k1, (n, 1)), "z": z}

    def gibbs_labels(key, p, logdensity):
        # Labels are independent given x: draw each from its exact conditional.
        def logit(i, k):
            return logdensity({**p, "z": p["z"].at[i].set(k)})

        logits = jax.vmap(
            lambda i: jax.vmap(lambda k: logit(i, k))(np.arange(2))
        )(np.arange(y.size))
        z = jax.random.categorical(key, logits, axis=1)
        return {**p, "z": z}, None

    # exact p(z_i = 1 | y) and log p(y) by quadrature over the offset
    grid = onp.linspace(-12, 12, 4001)
    dens = onp.exp(-0.5 * grid**2 / S0**2) / (S0 * onp.sqrt(2 * onp.pi))
    pdf = lambda m, c: onp.exp(-0.5 * ((y[:, None] - grid - c) / sd) ** 2) / (
        sd * onp.sqrt(2 * onp.pi)
    )
    comps = onp.stack(
        [w[k] * pdf(grid, comp[k]) for k in range(2)]
    )  # (2, n, grid)
    lik = comps.sum(0)  # (n, grid)
    joint = dens * onp.prod(lik, axis=0)
    dz = grid[1] - grid[0]
    evidence = joint.sum() * dz
    frac = ((comps[1] / lik) * joint).sum(axis=1) * dz / evidence

    r = tempered_smc(
        logprior, loglike, init, key=jax.random.key(6), n_particles=1000,
        replicates=2, moves=[gibbs_labels],
    )  # fmt: skip
    assert not r.stopped
    got = onp.asarray(r.weights @ r.particles["z"])
    assert onp.allclose(got, frac, atol=0.06)
    assert abs(r.log_z - onp.log(evidence)) < 4 * r.log_z_error + 0.1
