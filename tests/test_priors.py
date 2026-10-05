"""Isotropic-orientation priors: normalisation, sampling, and use in fit/numpyro."""

import jax
import jax.numpy as np
import numpy as onp
import numpyro.distributions as dist
import pytest
from numpyro.infer import MCMC, NUTS
from numpyro.infer.util import log_density
from scipy import integrate, stats

from virgil import fit
from virgil.likelihood import numpyro_model
from virgil.priors import IsotropicInclination, IsotropicLatitude

CASES = [
    (IsotropicInclination(), 0.0, 180.0),
    (IsotropicInclination(0.0, 90.0), 0.0, 90.0),
    (IsotropicInclination(30.0, 120.0), 30.0, 120.0),
    (IsotropicLatitude(), -onp.pi / 2, onp.pi / 2),
    (IsotropicLatitude(-0.3, 1.0), -0.3, 1.0),
]


def _pdf(prior):
    return lambda x: float(np.exp(prior.log_prob(np.asarray(x))))


@pytest.mark.parametrize("prior,lo,hi", CASES)
def test_log_prob_integrates_to_one(prior, lo, hi):
    total, _ = integrate.quad(_pdf(prior), lo, hi, limit=200)
    assert total == pytest.approx(1.0, abs=1e-5)
    assert float(prior.log_prob(np.asarray(hi + 0.1))) == -onp.inf


@pytest.mark.parametrize("prior,lo,hi", CASES)
def test_mean_and_cdf_match_the_density(prior, lo, hi):
    pdf = _pdf(prior)
    mean, _ = integrate.quad(lambda x: x * pdf(x), lo, hi, limit=200)
    assert float(prior.mean) == pytest.approx(mean, rel=1e-4)
    mid = 0.5 * (lo + hi)
    cdf, _ = integrate.quad(pdf, lo, mid, limit=200)
    assert float(prior.cdf(np.asarray(mid))) == pytest.approx(cdf, abs=1e-5)


def test_inclination_has_uniform_cosine():
    samples = onp.asarray(
        IsotropicInclination().sample(jax.random.key(0), (4000,))
    )
    assert samples.min() >= 0.0 and samples.max() <= 180.0
    cos_i = onp.cos(onp.deg2rad(samples))
    assert stats.kstest(cos_i, stats.uniform(-1, 2).cdf).pvalue > 0.01
    assert abs(onp.mean(samples < 90.0) - 0.5) < 0.03


def test_half_range_inclination_has_uniform_cosine():
    samples = onp.asarray(
        IsotropicInclination(0.0, 90.0).sample(jax.random.key(1), (4000,))
    )
    assert samples.max() <= 90.0
    cos_i = onp.cos(onp.deg2rad(samples))
    assert stats.kstest(cos_i, stats.uniform(0, 1).cdf).pvalue > 0.01


def test_latitude_has_uniform_sine():
    samples = onp.asarray(
        IsotropicLatitude().sample(jax.random.key(2), (4000,))
    )
    ks = stats.kstest(onp.sin(samples), stats.uniform(-1, 2).cdf)
    assert ks.pvalue > 0.01


def test_bad_ranges_are_rejected():
    with pytest.raises(ValueError, match="within"):
        IsotropicInclination(-10.0, 90.0)
    with pytest.raises(ValueError, match="low < high"):
        IsotropicInclination(90.0, 30.0)
    with pytest.raises(ValueError, match="within"):
        IsotropicLatitude(-2.0, 1.0)


def test_numpyro_model_with_only_the_prior_reproduces_it():
    # No data: the model is the prior, sampled in unconstrained coordinates
    # through biject_to, so a wrong Jacobian would distort cos i.
    prior = IsotropicInclination()
    model = numpyro_model(lambda **kw: None, {"inc": prior}, ())
    mcmc = MCMC(
        NUTS(model), num_warmup=300, num_samples=1500, progress_bar=False
    )
    mcmc.run(jax.random.key(3))
    inc = onp.asarray(mcmc.get_samples()["inc"])
    cos_i = onp.cos(onp.deg2rad(inc))
    # A correlated chain: compare moments, not a KS p-value.
    assert abs(cos_i.mean()) < 0.1
    assert cos_i.var() == pytest.approx(1 / 3, abs=0.06)
    logp = log_density(model, (), {}, {"inc": 60.0})[0]
    assert float(logp) == pytest.approx(float(prior.log_prob(60.0)))


def test_fit_with_a_flat_likelihood_returns_the_mode_of_the_prior():
    # The density sin i peaks at 90 degrees, and fit's MAP is the mode of
    # the prior density in the model's own parameter (no bijection Jacobian).
    def term(values):
        return 0.0 * np.atleast_1d(values["inc"])

    result = fit(
        lambda **kw: None,
        {"inc": IsotropicInclination()},
        (),
        likelihoods=[term],
        init={"inc": 40.0},
        method="lbfgs",
    )
    assert float(result.values["inc"]) == pytest.approx(90.0, abs=0.5)


def test_fit_with_data_pulls_inclination_to_the_likelihood():
    def term(values):
        return np.atleast_1d(values["inc"] - 50.0) / 20.0

    result = fit(
        lambda **kw: None,
        {"inc": IsotropicInclination()},
        (),
        likelihoods=[term],
        init={"inc": 40.0},
    )
    assert 50.0 < float(result.values["inc"]) < 90.0


def test_fit_rejects_lm_for_isotropic_priors():
    def term(values):
        return np.atleast_1d(values["inc"] - 50.0)

    with pytest.raises(TypeError, match="least-squares"):
        fit(
            lambda **kw: None,
            {"inc": IsotropicInclination()},
            (),
            likelihoods=[term],
            init={"inc": 40.0},
            method="lm",
        )


def test_expanded_prior_for_several_spots():
    prior = IsotropicLatitude().expand((3,))
    assert prior.sample(jax.random.key(4)).shape == (3,)
    assert prior.log_prob(np.zeros(3)).shape == (3,)
    assert isinstance(prior.base_dist, dist.Distribution)


# float32 and float64 robustness (Copilot review of #213).
def _both_precisions(test):
    def wrapper():
        test()
        with jax.enable_x64(True):
            test()

    wrapper.__name__ = test.__name__
    return pytest.mark.filterwarnings("ignore")(wrapper)


@_both_precisions
def test_narrow_ranges_near_both_poles():
    cases = [
        (IsotropicInclination(0.0, 0.01), 0.0, 0.01),
        (IsotropicInclination(179.99, 180.0), 179.99, 180.0),
        (
            IsotropicLatitude(np.pi / 2 - 1e-3, np.pi / 2),
            np.pi / 2 - 1e-3,
            np.pi / 2,
        ),
        (
            IsotropicLatitude(-np.pi / 2, -np.pi / 2 + 1e-3),
            -np.pi / 2,
            -np.pi / 2 + 1e-3,
        ),
    ]
    for prior, lo, hi in cases:
        samples = onp.asarray(prior.sample(jax.random.key(0), (500,)), float)
        assert onp.all(onp.isfinite(samples))
        assert samples.min() >= lo - 1e-6 and samples.max() <= hi + 1e-6
        width = hi - lo
        # Spread over the range, not collapsed onto one end.
        assert samples.std() > 0.1 * width
        mid = 0.5 * (lo + hi)
        assert 0.2 < float(onp.mean(samples < mid)) < 0.8
        logp = onp.asarray(prior.log_prob(np.asarray(samples)), float)
        assert onp.all(onp.isfinite(logp))
        # The density integrates to about 1: mean of p over uniform draws.
        grid = onp.linspace(lo, hi, 2001)[1:-1]
        pdf = onp.exp(onp.asarray(prior.log_prob(np.asarray(grid)), float))
        assert onp.trapezoid(pdf, grid) == pytest.approx(1.0, rel=0.05)


@_both_precisions
def test_cdf_outside_the_support_is_zero_and_one():
    for prior, lo, hi in CASES:
        below = np.asarray(lo - (270.0 if hi > 10 else 3.0))
        above = np.asarray(hi + (270.0 if hi > 10 else 3.0))
        assert float(prior.cdf(below)) == 0.0
        assert float(prior.cdf(above)) == 1.0
    assert float(IsotropicInclination().cdf(np.asarray(-90.0))) == 0.0
    assert float(IsotropicLatitude().cdf(np.asarray(-np.pi))) == 0.0
    assert float(IsotropicLatitude().cdf(np.asarray(np.pi))) == 1.0


@_both_precisions
def test_log_prob_at_the_endpoints_is_minus_inf_not_nan():
    for prior in (IsotropicInclination(), IsotropicLatitude()):
        for edge in (prior.low, prior.high):
            value = float(prior.log_prob(np.asarray(edge)))
            assert value == -onp.inf
    inc = float(IsotropicInclination(0.0, 90.0).log_prob(np.asarray(90.0)))
    assert onp.isfinite(inc)


@_both_precisions
def test_array_valued_full_range_bounds_are_accepted():
    half_pi = np.asarray(np.pi / 2)
    prior = IsotropicLatitude(-half_pi, half_pi)
    assert onp.isfinite(float(prior.log_prob(np.asarray(0.0))))
    IsotropicInclination(np.asarray(0.0), np.asarray(180.0))
    with pytest.raises(ValueError, match="within"):
        IsotropicLatitude(-half_pi, half_pi + 0.01)
