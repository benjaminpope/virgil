"""No-data tests: with no data, every sampler and prior returns its prior.

The rule (design/imaging_plan.md, standing decision 9; Hogg, Myers & Bovy
2010): sampling with an empty dataset must reproduce the stated prior,
Jacobians included. "No data" is, depending on the piece, an empty
likelihood (``data=()``, a zero-length ``RVData``) or data whose errors are
so large (1e9) that the likelihood is flat to rounding. Everything here is
tiny: a few hundred draws, short NUTS runs, small grids.
"""

import warnings

import jax
import jax.numpy as jnp
import numpy as onp
import numpyro
import numpyro.distributions as dist
import pytest
from numpyro.distributions.transforms import biject_to
from numpyro.infer import MCMC, NUTS
from numpyro.infer.util import potential_energy
from scipy import stats

from virgil.coverage import nrm_oidata, vlti_oidata
from virgil.detection import _log_prior_weights, detection_statistics
from virgil.fields import GaussianField
from virgil.fitting import fit, gauss_newton_mass
from virgil.imaging import image_priors
from virgil.likelihood import numpyro_model
from virgil.models import BinaryModelCartesian, Image
from virgil.observables import FluxSpectrum
from virgil.orbits import KeplerOrbit, RVData

# A KS p-value below this on a fixed seed would mean a wrong marginal, not
# bad luck: the draws are a few hundred, so real errors (a missing
# Jacobian shifts a LogUniform by a whole decade) are far below it.
KS_FLOOR = 1e-3
HUGE = 1e9  # an error so large that a datum carries no information


def _nuts(model, num_samples=1500, num_warmup=500, seed=0, **kwargs):
    """Posterior draws of a numpyro model by a short NUTS run."""
    mcmc = MCMC(
        NUTS(model, **kwargs),
        num_warmup=num_warmup,
        num_samples=num_samples,
        progress_bar=False,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mcmc.run(jax.random.PRNGKey(seed))
    return mcmc.get_samples()


def _cdf(prior, x):
    x = onp.asarray(x, float)
    if isinstance(prior, dist.LogUniform):  # numpyro has no cdf for it
        low, high = onp.log(prior.low), onp.log(prior.high)
        return onp.clip((onp.log(x) - low) / (high - low), 0.0, 1.0)
    return onp.asarray(prior.cdf(x), float)


def _assert_marginal(samples, site, prior):
    """The draws of ``site`` are consistent with ``prior`` (KS, fixed seed)."""
    draws = onp.asarray(samples[site], float)
    result = stats.kstest(draws, lambda x: _cdf(prior, x))
    assert result.pvalue > KS_FLOOR, (site, result)


def _function_model(**_):
    """A model function that builds nothing: only priors are sampled."""
    return None


# ---------------------------------------------------------------------------
# 1. numpyro_model
# ---------------------------------------------------------------------------

PRIORS = {
    "log_uniform": dist.LogUniform(1e-3, 1.0),
    "uniform": dist.Uniform(-2.0, 3.0),
    "normal": dist.Normal(1.0, 2.0),
    "beta": dist.Beta(2.0, 5.0),  # an interval, non-uniform inside
    "half_normal": dist.HalfNormal(1.5),  # a half line
}


def test_numpyro_model_with_no_data_samples_the_priors():
    model = numpyro_model(_function_model, PRIORS, ())
    samples = _nuts(model)
    for site, prior in PRIORS.items():
        _assert_marginal(samples, site, prior)


def test_numpyro_model_potential_is_prior_plus_jacobian():
    """The density NUTS sees is log p(x) + log|dx/dz| of each bijection."""
    model = numpyro_model(_function_model, PRIORS, ())
    rng = onp.random.default_rng(3)
    z = {k: jnp.asarray(rng.normal(size=())) for k in PRIORS}
    potential = potential_energy(model, (), {}, z)
    expected = 0.0
    for site, prior in PRIORS.items():
        bijection = biject_to(prior.support)
        x = bijection(z[site])
        expected += float(
            prior.log_prob(x) + bijection.log_abs_det_jacobian(z[site], x)
        )
    assert float(potential) == pytest.approx(-expected, rel=1e-6)


def test_numpyro_model_with_no_data_has_no_likelihood_site():
    """``data=()`` and no terms add no factor, so nothing but the priors."""
    model = numpyro_model(_function_model, PRIORS, ())
    trace = numpyro.handlers.trace(numpyro.handlers.seed(model, 0)).get_trace()
    assert set(trace) == set(PRIORS)


# ---------------------------------------------------------------------------
# 2. fit and gauss_newton_mass
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["lm", "lbfgs"])
def test_fit_with_no_data_is_the_normal_prior_mode(method):
    priors = {"a": dist.Normal(2.0, 0.5), "b": dist.Normal(-1.0, 3.0)}
    result = fit(
        _function_model,
        priors,
        (),
        init={"a": 0.0, "b": 0.0},
        method=method,
    )
    assert float(result.values["a"]) == pytest.approx(2.0, abs=1e-3)
    assert float(result.values["b"]) == pytest.approx(-1.0, abs=1e-3)


@pytest.mark.parametrize(
    "prior, mode",
    [
        (dist.Beta(3.0, 4.0), 2.0 / 5.0),
        (dist.Gamma(3.0, 2.0), 2.0 / 2.0),
    ],
)
def test_fit_with_no_data_is_the_prior_mode_in_the_models_parameters(
    prior, mode
):
    """The MAP is in the model's own parameters, with no Jacobian."""
    result = fit(
        _function_model,
        {"a": prior},
        (),
        init={"a": 0.9 * mode},
        method="lbfgs",
    )
    assert float(result.values["a"]) == pytest.approx(mode, abs=2e-3)


def _flat_data():
    return nrm_oidata(sigma_v2=HUGE, sigma_cp_deg=HUGE)


def _two_normals():
    return {
        "dra": dist.Normal(6.0, 2.5),
        "ddec": dist.Normal(-4.0, 0.25),
    }


def test_gauss_newton_mass_without_information_is_the_prior_covariance():
    template = BinaryModelCartesian(6.0, -4.0, 0.3)
    priors = _two_normals()
    mass = gauss_newton_mass(
        template, priors, _flat_data(), {"dra": 6.0, "ddec": -4.0}
    )
    (paths,) = mass["inverse_mass_matrix"]
    (covariance,) = mass["inverse_mass_matrix"].values()
    # A Normal's bijection is the identity, so z = x and cov = diag(scale²);
    # the block follows the order of the keys of the mass dict.
    expected = onp.diag([_two_normals()[p].scale ** 2 for p in paths])
    assert onp.allclose(covariance, expected, rtol=1e-5)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "gauss_newton_mass(data=()) raises: _gauss_newton_covariance "
        "concatenates the data residuals, an empty list, although fit "
        "documents data=() as allowed"
    ),
    raises=ValueError,
)
def test_gauss_newton_mass_with_empty_data_is_the_prior_covariance():
    template = BinaryModelCartesian(6.0, -4.0, 0.3)
    mass = gauss_newton_mass(
        template, _two_normals(), (), {"dra": 6.0, "ddec": -4.0}
    )
    (paths,) = mass["inverse_mass_matrix"]
    (covariance,) = mass["inverse_mass_matrix"].values()
    expected = onp.diag([_two_normals()[p].scale ** 2 for p in paths])
    assert onp.allclose(covariance, expected, rtol=1e-5)


# ---------------------------------------------------------------------------
# 3. noise= sites
# ---------------------------------------------------------------------------

NOISE = {
    "vis_error_rel": dist.LogUniform(1e-3, 1.0),
    "phi_error": dist.Uniform(0.0, 0.5),
    "vis_gain_telescope": dist.Uniform(0.0, 0.5),
    "phi_offset_baseline": dist.LogUniform(1e-3, 0.2),
}


def _uninformative_vlti():
    """A small VLTI-like dataset with gains and offsets and no information."""
    data = vlti_oidata(
        hour_angles_h=(0.0,),
        wavelengths_m=onp.linspace(2.0e-6, 2.4e-6, 2),
        sigma_v2=HUGE,
        sigma_cp_deg=HUGE,
    )
    return data.with_gains(telescope=0.01).with_closure_offsets(baseline=0.01)


def test_noise_sites_with_empty_data_are_their_priors():
    model = numpyro_model(_function_model, {}, (), noise=NOISE)
    samples = _nuts(model)
    for term, prior in NOISE.items():
        _assert_marginal(samples, f"noise.{term}", prior)


@pytest.mark.slow
def test_noise_sites_with_uninformative_data_are_their_priors():
    """The whole likelihood path, gains and offsets included, is flat."""
    data = _uninformative_vlti()
    priors = {"dra": dist.Normal(6.0, 2.0)}

    def model_fn(dra):
        return BinaryModelCartesian(dra, -4.0, 0.3)

    model = numpyro_model(model_fn, priors, data, noise=NOISE)
    samples = _nuts(model, num_samples=800, num_warmup=400)
    _assert_marginal(samples, "dra", priors["dra"])
    for term, prior in NOISE.items():
        _assert_marginal(samples, f"noise.{term}", prior)


# ---------------------------------------------------------------------------
# 4. GP latents
# ---------------------------------------------------------------------------

SHAPE = (4, 4)
PIXEL_MAS = 0.5
SIGMA, LENGTH = 1.3, 1.1


def _dct_basis(n):
    """Orthonormal DCT-II synthesis matrix, written out independently."""
    k = onp.arange(n)[None, :]
    i = onp.arange(n)[:, None]
    basis = onp.sqrt(2.0 / n) * onp.cos(onp.pi * (2 * i + 1) * k / (2 * n))
    basis[:, 0] /= onp.sqrt(2.0)
    return basis  # pixels x coefficients


def _kernel(order=2):
    """Pixel covariance of the stated field: Φ diag(S) Φᵀ, from the docs.

    S_jk ∝ (κ² + λ_jk)^(-order) with the Neumann Laplacian eigenvalues,
    the constant mode removed, scaled to a pixel-mean variance of σ².
    """
    n, m = SHAPE
    lam_j = (2 / PIXEL_MAS * onp.sin(onp.pi * onp.arange(n) / (2 * n))) ** 2
    lam_k = (2 / PIXEL_MAS * onp.sin(onp.pi * onp.arange(m) / (2 * m))) ** 2
    spectrum = (LENGTH**-2.0 + lam_j[:, None] + lam_k[None, :]) ** (-order)
    spectrum[0, 0] = 0.0
    spectrum *= SIGMA**2 * spectrum.size / spectrum.sum()
    synthesis = onp.kron(_dct_basis(n), _dct_basis(m))  # row-major pixels
    return synthesis @ onp.diag(spectrum.ravel()) @ synthesis.T


def test_gp_latents_are_standard_normal_and_imply_the_stated_kernel():
    field = GaussianField(
        onp.zeros(SHAPE), sigma=SIGMA, length_mas=LENGTH, order=2
    )
    scene = Image(field, pixel_scale_mas=PIXEL_MAS, flux=0.3)
    priors = image_priors(scene)
    assert list(priors) == ["log_brightness.latent"]
    model = numpyro_model(scene, priors, ())
    samples = _nuts(model, num_samples=3000, num_warmup=500)
    latent = onp.asarray(samples["log_brightness.latent"])
    assert latent.shape == (3000, *SHAPE)

    flat = latent.reshape(len(latent), -1)
    normal = dist.Normal(0.0, 1.0)
    for j in range(flat.shape[1]):
        result = stats.kstest(flat[:, j], lambda x: _cdf(normal, x))
        assert result.pvalue > KS_FLOOR, (j, result)
    # Independent coefficients: the sample correlation is noise of ~1/sqrt(N).
    corr = onp.corrcoef(flat.T) - onp.eye(flat.shape[1])
    assert onp.abs(corr).max() < 6 / onp.sqrt(len(flat))

    eta = jax.vmap(lambda z: field.set("latent", z).evaluate(PIXEL_MAS))(
        jnp.asarray(latent)
    )
    eta = onp.asarray(eta).reshape(len(latent), -1)
    expected = _kernel()
    sample_cov = onp.cov(eta.T)
    # Every kernel entry to within the sampling error of 3000 draws.
    scale = onp.sqrt(onp.outer(onp.diag(expected), onp.diag(expected)))
    assert onp.abs(sample_cov - expected).max() < 0.1 * scale.max()
    assert onp.linalg.norm(sample_cov - expected) < 0.06 * onp.linalg.norm(
        expected
    )
    # The field's variance averaged over pixels is sigma².
    assert onp.trace(sample_cov) / sample_cov.shape[0] == pytest.approx(
        SIGMA**2, rel=0.05
    )


def test_gp_kernel_helper_is_the_documented_normalisation():
    assert onp.trace(_kernel()) / (SHAPE[0] * SHAPE[1]) == pytest.approx(
        SIGMA**2
    )


# ---------------------------------------------------------------------------
# 5, 8. RV zero points and jitter
# ---------------------------------------------------------------------------

ORBIT = KeplerOrbit(400.0, 30.0, 0.4, 60.0, 40.0, 110.0, 20.0, t_ref=60500.0)


def _rv_params(values):
    return ORBIT, 0.5, 0.0, 50.0


def _empty_rv():
    return RVData([], [], [], t_ref=60500.0)


def _uninformative_rv(instrument=None):
    mjd = 60500.0 + onp.array([0.0, 40.0, 90.0, 150.0])
    return RVData(
        mjd, onp.zeros(4), HUGE, t_ref=60500.0, instrument=instrument
    )


def test_rv_zero_point_with_no_data_is_the_prior():
    term = _empty_rv().term(_rv_params, marginalise_offsets=(12.0, 3.0))
    mean, cov = term.posterior({})
    assert onp.allclose(mean, [12.0], atol=1e-10)
    assert onp.allclose(cov, [[9.0]], atol=1e-10)


def test_rv_zero_points_without_information_are_the_priors():
    data = _uninformative_rv(instrument=["a", "b", "a", "b"])
    term = data.term(
        _rv_params, marginalise_offsets=([10.0, -5.0], [3.0, 7.0])
    )
    mean, cov = term.posterior({})
    assert onp.allclose(mean, [10.0, -5.0], atol=1e-6)
    assert onp.allclose(cov, onp.diag([9.0, 49.0]), atol=1e-6)


JITTER = dist.LogUniform(0.1, 20.0)


@pytest.mark.parametrize("empty", [True, False])
def test_rv_jitter_with_no_data_is_its_prior(empty):
    data = _empty_rv() if empty else _uninformative_rv()
    term = data.term(_rv_params, jitter="jitter")
    model = numpyro_model(
        _function_model, {"jitter": JITTER}, (), likelihoods=[term]
    )
    samples = _nuts(model)
    _assert_marginal(samples, "jitter", JITTER)


def test_rv_jitter_with_marginalised_zero_points_is_its_prior():
    data = _uninformative_rv(instrument=["a", "b", "a", "b"])
    term = data.term(
        _rv_params,
        jitter="jitter",
        marginalise_offsets=([0.0, 0.0], [5.0, 5.0]),
    )
    model = numpyro_model(
        _function_model, {"jitter": JITTER}, (), likelihoods=[term]
    )
    samples = _nuts(model)
    _assert_marginal(samples, "jitter", JITTER)


# ---------------------------------------------------------------------------
# 6. OI_FLUX grey scale
# ---------------------------------------------------------------------------


def _flux_block(values, errors, width):
    wavel = onp.linspace(2.0e-6, 2.4e-6, 5)
    return FluxSpectrum.build(
        "flux",
        values,
        errors,
        wavel,
        row=onp.zeros(5),
        frame=onp.zeros(5),
        station=onp.zeros(5),
        width=width,
    )


@pytest.mark.parametrize("width", [1.0, 0.3])
def test_flux_scale_with_no_information_is_its_prior(width):
    """The mechanism: errors → ∞ leaves the prior N(μ, (width μ)²).

    Limitation (a breach of the no-data rule, to be fixed): ``build`` centres
    μ on the data's own mean level, so the prior is not independent of the
    data. Here μ is set by hand to a fixed value to test the mechanism, and
    the first assertion documents the data dependence.
    """
    import equinox as eqx

    block = _flux_block(onp.full(5, 7.0), onp.full(5, HUGE), width)
    assert float(block.mu[0]) == pytest.approx(7.0)  # centred on the data
    block = eqx.tree_at(lambda b: b.mu, block, onp.array([2.0]))
    prediction = block.predict(_PointTemplate(), None)
    mean, cov = block.posterior(prediction, block.values, block.errors)
    assert float(mean[0, 0]) == pytest.approx(2.0, rel=1e-5)
    assert float(cov[0, 0, 0]) == pytest.approx((width * 2.0) ** 2, rel=1e-5)


class _PointTemplate:
    """A model with a flat total spectrum, so the template is all ones."""

    def total_spectrum(self, wavel):
        return jnp.ones_like(wavel)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "FluxSpectrum.build centres the grey-scale prior on the data's own "
        "weighted mean (mu), so with no information the posterior mean is "
        "whatever the (uninformative) data values are, not a fixed prior"
    ),
)
def test_flux_scale_prior_does_not_depend_on_the_data():
    means = []
    for level in (7.0, 70.0):
        block = _flux_block(onp.full(5, level), onp.full(5, HUGE), 1.0)
        prediction = block.predict(_PointTemplate(), None)
        mean, _ = block.posterior(prediction, block.values, block.errors)
        means.append(float(mean[0, 0]))
    assert means[0] == pytest.approx(means[1])


# ---------------------------------------------------------------------------
# 7. Grid evidence
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape", [(5,), (3, 4), (3, 3, 6), (1, 5), (2, 2)])
def test_grid_prior_weights_integrate_to_one(shape):
    weights = onp.exp(onp.asarray(_log_prior_weights(shape, jnp.float64)))
    assert weights.sum() == pytest.approx(1.0, rel=1e-6)
    assert weights.shape == shape


@pytest.mark.parametrize("axis", ["linear", "log"])
def test_log_bayes_factor_without_information_is_zero(axis):
    flux = (
        jnp.linspace(0.0, 0.02, 6)
        if axis == "linear"
        else jnp.geomspace(1e-4, 0.05, 6)
    )
    grid = {
        "dra": jnp.linspace(-100.0, 100.0, 3),
        "ddec": jnp.linspace(-100.0, 100.0, 3),
        "flux": flux,
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = detection_statistics(_flat_data(), BinaryModelCartesian, grid)
    assert float(result["log_bayes_factor"]) == pytest.approx(0.0, abs=1e-8)
