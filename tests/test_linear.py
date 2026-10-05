"""Analytic marginalisation of linear parameters (virgil._linear)."""

import jax
import jax.numpy as np
import numpy as onp
import pytest

from virgil._linear import (
    LinearMarginal,
    posterior,
    whiten_cholesky,
    whiten_rank_one,
)


def _problem(n=40, k=4, seed=0):
    rng = onp.random.default_rng(seed)
    design = rng.normal(size=(n, k))
    sigma = rng.uniform(0.5, 2.0, n)
    mean = rng.normal(size=k)
    root = onp.tril(rng.normal(size=(k, k))) + 2 * onp.eye(k)
    cov = root @ root.T
    resid = rng.normal(size=n) * 3.0
    return design, sigma, mean, cov, resid


def _dense(design, sigma, mean, cov, resid):
    total = onp.diag(sigma**2) + design @ cov @ design.T
    r = resid - design @ mean
    chi2 = r @ onp.linalg.solve(total, r)
    logdet = onp.linalg.slogdet(total)[1]
    return chi2, logdet, -0.5 * (chi2 + logdet + r.size * onp.log(2 * onp.pi))


@pytest.mark.validates("virgil._linear.LinearMarginal", roots=["mathematics"])
@pytest.mark.parametrize("method", ["cholesky", "rank_one"])
def test_marginal_matches_the_dense_gaussian(method):
    with jax.enable_x64(True):
        design, sigma, mean, cov, resid = _problem()
        lm = LinearMarginal(design, mean, prior_cov=cov, method=method)
        u, log_norm = lm.whiten(resid, sigma)
        chi2, logdet, loglike = _dense(design, sigma, mean, cov, resid)
        assert u.shape == resid.shape
        onp.testing.assert_allclose(float(u @ u), chi2, rtol=1e-10)
        onp.testing.assert_allclose(float(log_norm), 0.5 * logdet, rtol=1e-10)
        onp.testing.assert_allclose(
            float(lm.loglike(resid, sigma)), loglike, rtol=1e-10
        )


@pytest.mark.validates("virgil._linear.posterior", roots=["mathematics"])
def test_posterior_matches_dense_conditioning():
    with jax.enable_x64(True):
        design, sigma, mean, cov, resid = _problem(seed=1)
        got_mean, got_cov = LinearMarginal(
            design, mean, prior_cov=cov
        ).posterior(resid, sigma)
        precision = onp.linalg.inv(cov) + design.T @ (
            design / sigma[:, None] ** 2
        )
        want_cov = onp.linalg.inv(precision)
        want_mean = want_cov @ (
            onp.linalg.solve(cov, mean) + design.T @ (resid / sigma**2)
        )
        onp.testing.assert_allclose(got_mean, want_mean, rtol=1e-9)
        onp.testing.assert_allclose(got_cov, want_cov, rtol=1e-9)


def test_both_whitenings_agree_and_stay_smooth_on_degenerate_columns():
    rng = onp.random.default_rng(2)
    x_np, col = rng.normal(size=12), rng.normal(size=12)
    U_np = onp.stack([col, col, rng.normal(size=12)], axis=1)
    with jax.enable_x64(True):
        x, U = np.asarray(x_np), np.asarray(U_np)
        a, la = whiten_rank_one(x, U)
        b, lb = whiten_cholesky(x, U)
        onp.testing.assert_allclose(float(a @ a), float(b @ b), rtol=1e-10)
        onp.testing.assert_allclose(float(la), float(lb), rtol=1e-10)

    # Gradients with respect to a width, at zero width and with two equal
    # columns: finite for the rank-one steps.
    x, U = np.asarray(x_np), np.asarray(U_np)

    def chi2(width):
        u, half_logdet = whiten_rank_one(x, width * U)
        return u @ u + 2 * half_logdet

    for width in (0.0, 0.3):
        assert bool(np.isfinite(jax.grad(chi2)(width)))


def test_the_prior_must_be_stated_and_finite():
    design = onp.ones((5, 2))
    with pytest.raises(ValueError, match="State the prior"):
        LinearMarginal(design, 0.0)
    with pytest.raises(ValueError, match="flat prior"):
        LinearMarginal(design, 0.0, prior_sd=onp.inf)
    with pytest.raises(ValueError, match="finite mean"):
        LinearMarginal(design, onp.nan, prior_sd=1.0)
    with pytest.raises(ValueError, match="method"):
        LinearMarginal(design, 0.0, prior_sd=1.0, method="eigh")


def test_standardised_posterior_is_the_prior_without_data():
    # No information (zero columns): the posterior of ω is the prior N(0, I).
    mean, cov = posterior(np.zeros(6), np.zeros((6, 3)))
    onp.testing.assert_allclose(mean, 0.0)
    onp.testing.assert_allclose(cov, onp.eye(3))


@pytest.mark.parametrize("method", ["cholesky", "rank_one"])
def test_marginal_in_float32_with_a_broad_zero_point_prior(method):
    # Default JAX precision, with an RV-like design: instrument indicators
    # (integers), km/s errors and the broad 1000 km/s zero-point prior.
    rng = onp.random.default_rng(3)
    inst = onp.repeat(onp.arange(3), 15)
    design = (inst[:, None] == onp.arange(3)[None, :]).astype(int)
    sigma = rng.uniform(0.5, 2.0, inst.size)
    resid = rng.normal(size=inst.size) + onp.array([20.0, -35.0, 5.0])[inst]
    lm = LinearMarginal(design, 0.5, prior_sd=1000.0, method=method)
    assert lm.prior_mean.dtype == np.result_type(float)  # not truncated
    u, log_norm = lm.whiten(resid, sigma)
    cov = 1000.0**2 * onp.eye(3)
    chi2, logdet, _ = _dense(
        design.astype(float), sigma, onp.full(3, 0.5), cov, resid
    )
    onp.testing.assert_allclose(float(u @ u), chi2, rtol=1e-3)
    onp.testing.assert_allclose(float(log_norm), 0.5 * logdet, rtol=1e-4)
    mean, _ = lm.posterior(resid, sigma)
    onp.testing.assert_allclose(
        onp.asarray(mean), [20.0, -35.0, 5.0], atol=1.0
    )
