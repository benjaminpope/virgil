"""The orbit posterior's NUTS preconditioner, start and time budget."""

import time

import jax
import jax.numpy as np
import numpy as onp
import numpyro
import numpyro.distributions as dist
from jax.flatten_util import ravel_pytree

import virgil.pipeline.orbit as orbit_module
from virgil.pipeline._common import (
    _deadline_factor,
    _diagnostics,
    _draws,
    _run_chains,
)

LO = onp.array([-1.0, 0.0, -2.0])
HI = onp.array([1.0, 4.0, 2.0])
X = onp.array([0.2, 1.0, -0.5])
SETTINGS = {
    "num_chains": 2,
    "num_warmup": 150,
    "num_samples": 300,
    "seed": 0,
    "chain_method": "vectorized",
}


def _precision(scale=(0.01, 0.02, 0.005), rho=0.95):
    sd = onp.array(scale)
    corr = onp.full((3, 3), rho) + (1 - rho) * onp.eye(3)
    return onp.linalg.inv(corr * sd[:, None] * sd[None, :])


def test_laplace_mass_is_the_covariance_in_logit_coordinates():
    h = _precision()
    mass, chol, kind = orbit_module._laplace_mass(-h, LO, HI, X)
    s = (X - LO) / (HI - LO)
    g = 1.0 / ((HI - LO) * s * (1 - s))
    assert kind == "laplace"
    assert onp.allclose(mass, g[:, None] * onp.linalg.inv(h) * g[None, :])
    assert onp.allclose(chol @ chol.T, mass)


def test_a_hessian_that_is_not_positive_definite_is_clipped_not_dropped():
    h = _precision()
    h[0, 0] = -1.0  # a saddle: one negative eigenvalue of the curvature
    mass, chol, kind = orbit_module._laplace_mass(-h, LO, HI, X)
    assert kind == "laplace-clipped"
    assert onp.all(onp.linalg.eigvalsh(mass) > 0)
    assert onp.all(onp.isfinite(chol))
    bad = onp.full((3, 3), onp.nan)
    assert orbit_module._laplace_mass(bad, LO, HI, X) == (
        None,
        None,
        "adaptive",
    )


def test_numpyro_order_is_the_sorted_site_order():
    sites = ["log_period", "tau", "ecc", "cos_inc", "omega"]
    flat, _ = ravel_pytree(
        {k: np.asarray(float(i)) for i, k in enumerate(sites)}
    )
    order = orbit_module._numpyro_order(sites)
    assert onp.array_equal(onp.asarray(flat), order.astype(float))


def _correlated_model(precision):
    cov = onp.linalg.inv(precision)

    def model():
        x = np.stack(
            [
                numpyro.sample(k, dist.Uniform(float(a), float(b)))
                for k, a, b in zip("cab", LO, HI)  # sorted order is a, b, c
            ]
        )
        numpyro.factor(
            "loglike",
            dist.MultivariateNormal(np.asarray(X), np.asarray(cov)).log_prob(
                x
            ),
        )

    return model


def test_chains_start_at_the_mode_with_the_dense_mass_and_reach_the_ess(
    monkeypatch,
):
    """Strong correlation, scales 1e-2 in a unit box: NUTS from the Laplace
    mass reaches ESS 100 in a few hundred gradients per draw."""
    h = _precision()
    mass, chol, _ = orbit_module._laplace_mass(-h, LO, HI, X)
    order = orbit_module._numpyro_order(list("cab"))  # the model's site order
    seen = {}
    import numpyro.infer as infer

    real = infer.NUTS

    def spy(*a, **k):
        seen.update(k)
        return real(*a, **k)

    monkeypatch.setattr(infer, "NUTS", spy)
    priors = {
        k: dist.Uniform(float(a), float(b)) for k, a, b in zip("cab", LO, HI)
    }
    start = dict(zip("cab", X))
    mcmc = _run_chains(
        _correlated_model(h),
        SETTINGS,
        start,
        dense_mass=True,
        priors=priors,
        inverse_mass=mass[onp.ix_(order, order)],
        jitter=chol[onp.ix_(order, order)],
        max_tree_depth=6,
        adapt_mass=False,
    )
    assert seen["inverse_mass_matrix"].shape == (3, 3)
    assert seen["max_tree_depth"] == 6
    assert seen["adapt_mass_matrix"] is False
    drawn, stats = _draws(mcmc)
    ess = _diagnostics(list(drawn.values()))["ess_bulk"]
    assert ess >= 100
    assert onp.asarray(stats["num_steps"]).max() <= 2**6 - 1
    assert onp.sum(stats["num_steps"]) <= 300 * 2 * 20  # the gradient budget


def test_a_passed_deadline_drains_the_chains_at_once(monkeypatch):
    calls = []

    class Clock:  # the deadline passes after the start is checked
        @staticmethod
        def monotonic():
            calls.append(0)
            return 0.0 if len(calls) < 50 else 10.0

    monkeypatch.setattr("virgil.pipeline._common.time", Clock)

    def model():
        x = numpyro.sample("x", dist.Uniform(-1.0, 1.0))
        numpyro.factor("loglike", -0.5 * (x / 0.1) ** 2)
        numpyro.factor("deadline", _deadline_factor(5.0, x))

    mcmc = _run_chains(model, SETTINGS, {"x": onp.array(0.0)}, dense_mass=True)
    diverging = onp.asarray(mcmc.get_extra_fields()["diverging"])
    assert diverging.mean() > 0.5  # nearly every transition after the deadline


def test_the_clock_is_not_read_before_the_deadline():
    @jax.jit
    def alive(x):
        return _deadline_factor(time.monotonic() + 3600.0, x)

    assert float(alive(np.asarray(0.0))) == 0.0
