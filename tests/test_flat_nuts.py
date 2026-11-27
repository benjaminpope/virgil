"""``flat_nuts`` samples the same posterior as ``MCMC(NUTS(numpyro_model))``."""

import functools

import jax
import jax.numpy as np
import numpy as onp
import numpyro.distributions as dist
import pytest
from numpyro.infer import MCMC, NUTS, init_to_value
from numpyro.infer.util import initialize_model

from virgil.angles import AngleVector
from virgil.flat_nuts import (
    _warn_if_stuck,
    flat_nuts,
    flat_nuts_sampler,
    packed_posterior,
)
from virgil.likelihood import numpyro_model
from virgil.orbits import KeplerOrbit, PositionData
from virgil.priors import IsotropicInclination

T_REF = 60000.0


def _in_x64(test):
    """Run ``test`` in float64: its tolerances hold only there. The others run in the
    caller's default precision, as a user's call would."""

    @functools.wraps(test)
    def wrapper():
        with jax.enable_x64(True):
            test()

    return wrapper


TRUTH = dict(
    P=200.0, a=20.0, e=0.34, inc=53.4, omega=76.9, Omega=28.2, M0=120.0
)


def _orbit(v):
    dt_peri = -np.deg2rad(v["M0"]) * v["P"] / (2.0 * np.pi)
    return KeplerOrbit(
        v["P"],
        dt_peri,
        v["e"],
        v["inc"],
        v["omega"],
        v["Omega"],
        v["a"],
        T_REF,
    )


def _problem():
    rng = onp.random.default_rng(3)
    times = T_REF + onp.sort(rng.uniform(-150.0, 150.0, 10))
    sigma = 0.05
    truth = _orbit({k: np.asarray(x) for k, x in TRUTH.items()})
    ra, dec, _ = (onp.asarray(c) for c in truth.relative(times))
    cov = onp.tile(onp.eye(2) * sigma**2, (times.size, 1, 1))
    ra = ra + sigma * rng.standard_normal(ra.shape)
    dec = dec + sigma * rng.standard_normal(dec.shape)
    priors = {
        "P": dist.Uniform(150.0, 250.0),
        "a": dist.Uniform(10.0, 40.0),
        "e": dist.Uniform(0.0, 0.9),
        "inc": IsotropicInclination(),
        "omega": AngleVector(),
        "Omega": AngleVector(),
        "M0": AngleVector(),
    }
    term = PositionData(times, ra, dec, cov, t_ref=T_REF).term(_orbit)
    return priors, term


@_in_x64
def test_packed_density_is_the_numpyro_potential():
    priors, term = _problem()
    model = numpyro_model(lambda **v: None, priors, (), likelihoods=[term])
    res = initialize_model(jax.random.PRNGKey(0), model)
    z = res.param_info.z
    logdensity, unpack, pack, size = packed_posterior(priors, [term])
    u = np.concatenate([np.atleast_1d(z[k]) for k in z])
    assert u.size == size
    assert float(logdensity(u)) == pytest.approx(
        -float(res.potential_fn(z)), rel=1e-9
    )
    back = pack(unpack(u))
    assert onp.allclose(back, u, atol=1e-9)


def _starts(n):
    """Near the optimum, as the pipelines' grid-search starts are: blackjax's adaptation
    (initial step size 1, no step-size search) is less forgiving of far-off starts."""

    def vec(deg):
        return onp.array(
            [onp.cos(onp.deg2rad(deg)), onp.sin(onp.deg2rad(deg))]
        )

    return [
        {
            "P": 196.0 + 2 * c,
            "a": 18.0,
            "e": 0.3,
            "inc": 50.0 + c,
            "omega_vec": vec(70.0 + 3 * c),
            "Omega_vec": vec(30.0 - 2 * c),
            "M0_vec": vec(115.0 + 2 * c),
        }
        for c in range(n)
    ]


@_in_x64
def test_same_posterior_as_numpyro_nuts():
    priors, term = _problem()
    model = numpyro_model(lambda **v: None, priors, (), likelihoods=[term])
    ref = MCMC(
        NUTS(
            model,
            dense_mass=True,
            target_accept_prob=0.9,
            init_strategy=init_to_value(values=_starts(1)[0]),
        ),
        num_warmup=400,
        num_samples=600,
        num_chains=2,
        chain_method="vectorized",
        progress_bar=False,
    )
    ref.run(jax.random.PRNGKey(1))
    starts = _starts(2)
    r = ref.get_samples()
    samples, info = flat_nuts(
        priors,
        [term],
        num_warmup=400,
        num_samples=600,
        num_chains=2,
        target_accept=0.9,
        seed=2,
        init=starts,
    )
    assert float(info["diverging"].mean()) < 0.02
    # The scalar parameters are unimodal; compare medians and widths to sampling noise.
    for site in ("P", "a", "e", "inc", "M0"):
        a = onp.asarray(r[site]).ravel()
        b = onp.asarray(samples[site]).ravel()
        sd = a.std()
        assert abs(onp.median(a) - onp.median(b)) < 0.25 * sd, site
        assert abs(b.std() / sd - 1.0) < 0.25, site
    assert set(r) == set(samples)  # the same site names


@pytest.mark.filterwarnings("ignore:flat_nuts:UserWarning")
def test_sampler_is_reusable_and_seeded():
    priors, term = _problem()
    sample = flat_nuts_sampler(
        priors, [term], num_warmup=50, num_samples=20, num_chains=2
    )
    a, _ = sample(5)
    b, _ = sample(5)
    c, _ = sample(6)
    assert onp.array_equal(a["P"], b["P"])
    assert not onp.array_equal(a["P"], c["P"])
    assert a["omega_vec"].shape == (2, 20, 2)


def test_angle_start_in_degrees_matches_vector_start():
    priors, term = _problem()
    _, unpack, pack, _ = packed_posterior(priors, [term])
    vec = _starts(1)[0]
    deg = {k: v for k, v in vec.items() if not k.endswith("_vec")}
    deg.update({"omega": 70.0, "Omega": 30.0, "M0": 115.0})
    assert onp.allclose(pack(vec), pack(deg), atol=1e-12)


def test_stuck_chain_warns_and_a_clean_run_does_not(recwarn):
    ok = onp.zeros((2, 50), bool)
    _warn_if_stuck(ok, onp.array([0.11, 0.12]))
    assert not recwarn.list
    with pytest.warns(UserWarning, match="stuck"):
        _warn_if_stuck(ok, onp.array([0.11, 0.002]))
    with pytest.warns(UserWarning, match="stuck"):
        _warn_if_stuck(onp.ones((2, 50), bool), onp.array([0.1, 0.1]))
    with pytest.warns(UserWarning, match="stuck"):
        _warn_if_stuck(ok, onp.array([0.1, onp.nan]))


@_in_x64
def test_density_and_gradient_with_log_uniform_and_von_mises_priors():
    priors, term = _problem()
    priors = dict(
        priors,
        a=dist.LogUniform(10.0, 40.0),
        M0=AngleVector(mean=120.0, kappa=4.0),
    )
    model = numpyro_model(lambda **v: None, priors, (), likelihoods=[term])
    res = initialize_model(jax.random.PRNGKey(2), model)
    z = res.param_info.z
    logdensity, _, _, _ = packed_posterior(priors, [term])
    u = np.concatenate([np.atleast_1d(z[k]) for k in z])
    assert float(logdensity(u)) == pytest.approx(
        -float(res.potential_fn(z)), rel=1e-9
    )
    g = jax.grad(logdensity)(u)
    gz = jax.grad(res.potential_fn)(z)
    ref = -np.concatenate([np.atleast_1d(gz[k]) for k in z])
    assert onp.allclose(g, ref, rtol=1e-8, atol=1e-8)


def test_vector_priors_and_boundary_starts_are_refused():
    priors, term = _problem()
    with pytest.raises(ValueError, match="scalar priors"):
        packed_posterior(
            dict(priors, a=dist.Uniform(10.0, 40.0).expand([2])), [term]
        )
    sample = flat_nuts_sampler(
        priors, [term], num_warmup=5, num_samples=5, num_chains=1
    )
    with pytest.raises(ValueError, match="not finite"):
        sample(0, dict(_starts(1)[0], e=0.0))


@pytest.mark.filterwarnings("ignore:flat_nuts:UserWarning")
def test_float32_run_returns_float32_samples():
    priors, term = _problem()
    samples, _ = flat_nuts(
        priors,
        [term],
        num_warmup=100,
        num_samples=50,
        num_chains=2,
        init=_starts(2),
        dtype="float32",
    )
    assert samples["P"].dtype == onp.float32
    assert abs(float(onp.median(samples["P"])) - 200.0) < 2.0
