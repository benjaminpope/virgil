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
from virgil.priors import (
    IsotropicInclination,
    IsotropicLatitude,
    hierarchical_scales,
)

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


def test_fit_with_a_flat_likelihood_is_not_pulled_by_the_prior():
    # fit optimises the inclination in its flat coordinate (cos i), where
    # the prior is constant, so with a flat likelihood nothing moves it.
    # (Before fit used flat coordinates, the MAP was the mode of sin i in
    # i, 90 degrees, which depends on the parametrisation.)
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
    assert float(result.values["inc"]) == pytest.approx(40.0, abs=1e-6)


def test_fit_with_data_is_the_likelihood_maximum_in_the_range():
    # Flat in cos i, the prior adds nothing: the MAP is the likelihood's
    # peak (before, the sin i density pulled it towards 90 degrees).
    # Tighten the gradient tolerance to resolve the peak to 1e-4 degrees.
    def term(values):
        return np.atleast_1d(values["inc"] - 50.0) / 20.0

    result = fit(
        lambda **kw: None,
        {"inc": IsotropicInclination()},
        (),
        likelihoods=[term],
        init={"inc": 40.0},
        gtol=1e-8,
    )
    assert result.info["converged"] is True
    assert float(result.values["inc"]) == pytest.approx(50.0, abs=1e-4)


@pytest.mark.parametrize(
    "prior, start, peak",
    [
        (IsotropicInclination(), 40.0, 50.0),
        (IsotropicInclination(0.0, 90.0), 80.0, 30.0),
        (IsotropicLatitude(), 0.1, 0.6),
        (IsotropicLatitude(-0.2, 1.2), 0.0, 1.0),
    ],
)
def test_fit_chooses_lm_for_isotropic_priors(prior, start, peak):
    # Isotropic priors have a flat coordinate, so they no longer force
    # L-BFGS (method="lm" used to raise): LM runs and is the default.
    def term(values):
        return np.atleast_1d(values["x"] - peak) / 0.01

    result = fit(
        lambda **kw: None,
        {"x": prior},
        (),
        likelihoods=[term],
        init={"x": start},
    )
    assert result.info["method"] == "lm"
    assert result.info["converged"] is True
    assert float(result.values["x"]) == pytest.approx(peak, abs=1e-5)


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


def test_hierarchical_scales_are_log_normal_members():
    values = {"cp_median": 2.0, "cp_spread": 0.5}
    expected = 2.0 * onp.exp(0.5 * onp.array([-1.0, 0.0, 2.0]))
    # Centred (default): log s is sampled, with the population density.
    priors, scales = hierarchical_scales(
        "cp", 3, median=dist.LogUniform(0.5, 4.0)
    )
    assert set(priors) == {"cp_median", "cp_spread", "cp_log"}
    assert isinstance(priors["cp_spread"], dist.LogUniform)
    assert priors["cp_log"].event_shape == (3,)
    centred = {**values, "cp_log": np.log(np.asarray(expected))}
    got = [float(s(centred)) for s in scales]
    assert got == pytest.approx(expected, rel=1e-6)
    density = stats.norm(onp.log(2.0), 0.5).logpdf(onp.log(expected[2]))
    assert float(scales[2].log_prior(centred)) == pytest.approx(density)
    # Non-centred: s = median exp(spread z), with no extra density.
    priors, scales = hierarchical_scales("cp", 3, centred=False)
    assert set(priors) == {"cp_median", "cp_spread", "cp_z"}
    raw = {**values, "cp_z": np.array([-1.0, 0.0, 2.0])}
    got = [float(s(raw)) for s in scales]
    assert got == pytest.approx(expected, rel=1e-6)
    assert float(scales[0].log_prior(raw)) == 0.0
    # Equal members compare equal, so repeated fits do not recompile.
    assert hierarchical_scales("cp", 3, centred=False)[1] == scales
    with pytest.raises(ValueError, match="at least 1"):
        hierarchical_scales("cp", 0)


def test_hierarchical_scales_no_data_reproduce_the_population():
    # With no data, NUTS on the priors (plus any population densities)
    # must reproduce the generative model: hyperparameters from their
    # priors, then log s ~ N(log median, spread). The non-centred form is
    # sampled here: with no data the centred one is the classic funnel,
    # which NUTS cannot explore (its density is checked exactly in
    # test_fitting.py).
    import numpyro

    from virgil.likelihood import noise_sites, tied_log_prior

    population, scales = hierarchical_scales("s", 2, centred=False)
    sites = noise_sites([{"phi_scale": s} for s in scales], 2)

    def model():
        values = {k: numpyro.sample(k, p) for k, p in population.items()}
        numpyro.factor("population", tied_log_prior(sites, values))
        numpyro.deterministic("log_s", np.log(scales[0](values)))

    mcmc = MCMC(
        NUTS(model, target_accept_prob=0.9),
        num_warmup=1000,
        num_samples=6000,
        progress_bar=False,
    )
    mcmc.run(jax.random.PRNGKey(0))
    log_s = onp.asarray(mcmc.get_samples()["log_s"])
    keys = jax.random.split(jax.random.PRNGKey(1), 3)
    median = onp.asarray(population["s_median"].sample(keys[0], (20000,)))
    spread = onp.asarray(population["s_spread"].sample(keys[1], (20000,)))
    unit = onp.asarray(jax.random.normal(keys[2], (20000,)))
    direct = onp.log(median) + spread * unit
    # Thinned: NUTS draws are correlated.
    assert stats.ks_2samp(log_s[::10], direct).pvalue > 1e-3
