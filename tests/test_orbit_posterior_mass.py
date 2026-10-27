"""The orbit posterior's NUTS preconditioner, start and time budget."""

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


def _precision(scale=(0.01, 0.02, 0.005)):
    sd = onp.array(scale)
    corr = onp.array([[1, 0.97, 0.2], [0.97, 1, 0.4], [0.2, 0.4, 1.0]])
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


def _run(h, permute=True, spy=None):
    """The toy posterior (sites c, a, b; numpyro orders them a, b, c) run
    with the Laplace mass, in numpyro's order or, wrongly, in the model's."""
    mass, chol, _ = orbit_module._laplace_mass(-h, LO, HI, X)
    order = orbit_module._numpyro_order(list("cab"))
    if not permute:
        order = onp.arange(3)
    priors = {
        k: dist.Uniform(float(a), float(b)) for k, a, b in zip("cab", LO, HI)
    }
    mcmc = _run_chains(
        _correlated_model(h),
        SETTINGS,
        dict(zip("cab", X)),
        dense_mass=True,
        priors=priors,
        inverse_mass=mass[onp.ix_(order, order)],
        jitter=chol[onp.ix_(order, order)],
        max_tree_depth=6,
        adapt_mass=False,
    )
    drawn, stats = _draws(mcmc)
    return (
        _diagnostics(list(drawn.values()))["ess_bulk"],
        int(onp.sum(stats["num_steps"])),
        mass,
    )


def test_dense_mass_reaches_the_ess_in_budget_and_its_order_matters():
    """Correlated posterior, scales 1e-2 in a unit box: the Laplace mass in
    numpyro's site order reaches ESS 100 in a few gradients per draw; the
    same matrix in the wrong order does not."""
    h = _precision()
    ess, grads, _ = _run(h)
    assert ess >= 100
    assert grads <= 300 * 2 * 12  # the gradient budget
    ess_bad, grads_bad, _ = _run(h, permute=False)
    assert grads_bad > 1.5 * grads or ess_bad < 0.5 * ess


def test_chains_start_at_the_mode_and_the_others_are_jittered(monkeypatch):
    import numpyro.infer as infer

    seen = {}

    class Spy(infer.MCMC):
        def run(self, rng_key, *args, init_params=None, **kwargs):
            seen["init"] = init_params
            return super().run(
                rng_key, *args, init_params=init_params, **kwargs
            )

    monkeypatch.setattr(infer, "MCMC", Spy)
    _, _, mass = _run(_precision())
    init = onp.stack([onp.asarray(seen["init"][k]) for k in "abc"], axis=1)
    s = (X - LO) / (HI - LO)  # the mode, in the logit coordinates of a, b, c
    mode = onp.log(s / (1 - s))[[1, 2, 0]]
    sd = onp.sqrt(onp.diag(mass))[[1, 2, 0]]
    assert onp.allclose(init[0], mode, atol=1e-6)
    gap = onp.abs(init[1] - mode)
    assert onp.all(gap > 0)  # jittered ...
    assert onp.all(gap < 2 * sd)  # ... by a fraction of the posterior width


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


def test_the_clock_is_read_inside_the_compiled_function(monkeypatch):
    now = [0.0]

    class Clock:
        @staticmethod
        def monotonic():
            return now[0]

    monkeypatch.setattr("virgil.pipeline._common.time", Clock)
    alive = jax.jit(lambda x: _deadline_factor(5.0, x))
    assert float(alive(np.asarray(0.0))) == 0.0
    now[0] = 10.0  # the same compiled function, after the deadline
    assert float(alive(np.asarray(0.0))) == -np.inf
