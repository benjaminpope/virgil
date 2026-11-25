"""``flat_nuts`` samples the same posterior as ``MCMC(NUTS(numpyro_model))``."""

import jax
import jax.numpy as np
import numpy as onp
import numpyro.distributions as dist
import pytest
from numpyro.infer import MCMC, NUTS, init_to_value
from numpyro.infer.util import initialize_model

from virgil.angles import AngleVector
from virgil.flat_nuts import flat_nuts, flat_nuts_sampler, packed_posterior
from virgil.likelihood import numpyro_model
from virgil.orbits import KeplerOrbit, PositionData
from virgil.priors import IsotropicInclination

T_REF = 60000.0
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
    for site in ("P", "a", "e", "inc"):
        a = onp.asarray(r[site]).ravel()
        b = onp.asarray(samples[site]).ravel()
        sd = a.std()
        assert abs(onp.median(a) - onp.median(b)) < 0.25 * sd, site
        assert abs(b.std() / sd - 1.0) < 0.25, site
    assert set(r) <= set(samples)  # the same site names, plus nothing missing


def test_sampler_is_reusable_and_seeded():
    priors, term = _problem()
    sample = flat_nuts_sampler(
        priors, [term], num_warmup=50, num_samples=20, num_chains=2
    )
    a, _ = sample(5)
    b, _ = sample(5)
    assert onp.array_equal(a["P"], b["P"])
    assert a["omega_vec"].shape == (2, 20, 2)
