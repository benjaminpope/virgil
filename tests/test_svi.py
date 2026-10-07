"""Variational inference: ``virgil.svi.variational`` (virgil#12).

Everything here is tiny: two- or three-parameter posteriors, short SVI and
NUTS runs. Each test runs with flat coordinates on and off, since
``numpyro_model``'s default may change (design/sampler_flat_coordinates.md).
"""

import jax
import jax.numpy as jnp
import numpy as onp
import numpyro.distributions as dist
import pytest
from numpyro.distributions.transforms import biject_to
from numpyro.infer import MCMC, NUTS, init_to_value

from virgil._flat import flat_sampled
from virgil._precision import cast_tree, run_in
from virgil.angles import AngleVector
from virgil.coverage import nrm_oidata
from virgil.fitting import fit, gauss_newton_mass
from virgil.likelihood import numpyro_model
from virgil.models import BinaryModelAngular, BinaryModelCartesian
from virgil.svi import variational

PRIORS = {"s": dist.LogUniform(1e-3, 1.0), "x": dist.Uniform(-5.0, 5.0)}
MU = onp.array([-0.5, 0.3])
COV = onp.array([[0.04, 0.04], [0.04, 0.16]])  # σ = 0.2, 0.4; ρ = 0.5


def _no_model(**_):
    return None


class _GaussianInZ:
    """A likelihood that makes the posterior exactly N(MU, COV) in the
    unconstrained coordinates z of the sampler.

    ``loglike`` is the Gaussian in z minus the prior's density in z (the
    prior times the Jacobian of z → x), so the product is the Gaussian.
    ``__call__`` gives its whitened residuals, for ``gauss_newton_mass``.
    """

    def __init__(self, flat):
        self.flat = flat
        self.chol = onp.linalg.cholesky(COV)

    def _z(self, values):
        zs, log_prior = [], 0.0
        for name in PRIORS:
            prior = PRIORS[name]
            support = (flat_sampled(prior) if self.flat else prior).support
            transform = biject_to(support)
            x = values[name]
            z = transform.inv(x)
            log_prior += prior.log_prob(x) + transform.log_abs_det_jacobian(
                z, x
            )
            zs.append(z)
        return jnp.stack(zs), log_prior

    def __call__(self, values):
        z, _ = self._z(values)
        return jax.scipy.linalg.solve_triangular(
            jnp.asarray(self.chol, z.dtype), z - MU, lower=True
        )

    def loglike(self, values):
        _, log_prior = self._z(values)
        return -0.5 * jnp.sum(self(values) ** 2) - log_prior


def _x_of_z(z, flat):
    out = {}
    for i, name in enumerate(PRIORS):
        prior = PRIORS[name]
        support = (flat_sampled(prior) if flat else prior).support
        out[name] = biject_to(support)(jnp.asarray(z[..., i]))
    return out


def _z_of_x(samples, flat):
    zs = []
    for name in PRIORS:
        prior = PRIORS[name]
        support = (flat_sampled(prior) if flat else prior).support
        zs.append(onp.asarray(biject_to(support).inv(samples[name])))
    return onp.stack(zs, axis=-1)


@pytest.mark.parametrize("flat", [True, False])
def test_a_posterior_gaussian_in_z_is_recovered_exactly(flat):
    """With the posterior exactly Gaussian in z, the Gaussian guide is
    exact. Started at the fit's values, the Gauss–Newton width is exact:
    log q - log p is then the same constant for every draw, the
    normalisation 0.5 log det(2πΣ), so the loss is exactly that. (The
    gradient's score term still has zero mean but not zero variance, which
    Adam would follow, so that check uses a negligible step size.) Started
    at a unit width, the guide converges to the posterior."""
    term = _GaussianInZ(flat)
    with jax.enable_x64(True):
        start = {k: float(v) for k, v in _x_of_z(MU, flat).items()}
    common = dict(
        likelihoods=[term], start=start, guide="mvn", flat_coordinates=flat
    )
    exact = variational(
        _no_model, PRIORS, (), steps=20, learning_rate=1e-12, **common
    )
    loc, tril = map(onp.asarray, exact.guide.gaussian(exact.params))
    onp.testing.assert_allclose(loc, MU, atol=1e-8)
    onp.testing.assert_allclose(tril @ tril.T, COV, atol=1e-8)
    assert exact.info["dense_start"]
    normalisation = 0.5 * onp.log(onp.linalg.det(2 * onp.pi * COV))
    # PRIORS are float32 inside the term, float64 in the model: 1e-7.
    onp.testing.assert_allclose(exact.losses, -normalisation, atol=1e-6)

    learnt = variational(
        _no_model,
        PRIORS,
        (),
        steps=3000,
        learning_rate=1e-2,
        dense_start=False,
        init_scale=1.0,
        **common,
    )
    loc, tril = map(onp.asarray, learnt.guide.gaussian(learnt.params))
    onp.testing.assert_allclose(loc, MU, atol=0.02)
    onp.testing.assert_allclose(
        onp.sqrt(onp.diag(tril @ tril.T)), onp.sqrt(onp.diag(COV)), rtol=0.05
    )
    # The draws, carried back to z, have the posterior's moments.
    z = _z_of_x(learnt.samples, flat)
    onp.testing.assert_allclose(z.mean(0), MU, atol=0.03)
    onp.testing.assert_allclose(onp.cov(z.T), COV, rtol=0.1, atol=3e-3)


def _binary_data(flux=0.02, key=0):
    truth = BinaryModelCartesian(120.0, -80.0, flux)
    return nrm_oidata().with_model(truth, key=jax.random.PRNGKey(key))


BINARY_PRIORS = {
    "dra": dist.Uniform(-400.0, 400.0),
    "ddec": dist.Uniform(-400.0, 400.0),
    "flux": dist.LogUniform(1e-4, 1.0),
}
BINARY = BinaryModelCartesian(110.0, -70.0, 0.01)


# numpyro's MAP includes the flat priors' logistic potential, fit's does
# not, so the Laplace guide's loss still creeps down from fit's values.
@pytest.mark.filterwarnings("ignore:The SVI loss had not settled")
@pytest.mark.parametrize("flat", [True, False])
def test_laplace_guide_covariance_matches_gauss_newton(flat):
    """numpyro's Laplace guide (the Hessian of the log posterior at its
    MAP) against ``gauss_newton_mass`` (JᵀJ at the fit) on noiseless data,
    where the residuals vanish at the optimum and the two curvatures agree
    up to the flat priors' logistic potential, which the data swamp."""
    data = nrm_oidata().with_model(BinaryModelCartesian(120.0, -80.0, 0.02))
    result = fit(BINARY, BINARY_PRIORS, data)
    vi = variational(
        BINARY,
        BINARY_PRIORS,
        data,
        start=result,
        guide="laplace",
        steps=200,
        learning_rate=1e-4,  # numpyro's Adam on the MAP, already there
        flat_coordinates=flat,
    )
    with run_in("float64"):
        transform = vi.guide.get_transform(vi.params)
        tril = onp.asarray(transform.scale_tril)
    laplace = tril @ tril.T
    order = list(vi.guide._init_locs)
    mass = gauss_newton_mass(
        BINARY, BINARY_PRIORS, data, result.values, flat_coordinates=flat
    )
    ((sites, gn),) = mass["inverse_mass_matrix"].items()
    index = [list(sites).index(s) for s in order]
    gn = onp.asarray(gn)[onp.ix_(index, index)]
    onp.testing.assert_allclose(
        onp.sqrt(onp.diag(laplace)), onp.sqrt(onp.diag(gn)), rtol=0.02
    )
    corr = laplace / onp.sqrt(onp.outer(onp.diag(laplace), onp.diag(laplace)))
    corr_gn = gn / onp.sqrt(onp.outer(onp.diag(gn), onp.diag(gn)))
    onp.testing.assert_allclose(corr, corr_gn, atol=0.02)


def _nuts(model, priors, data, values, flat, dtype):
    with run_in(dtype):
        model, priors, data = cast_tree((model, priors, data), dtype)
        values = cast_tree(dict(values), dtype)
        mcmc = MCMC(
            NUTS(
                numpyro_model(model, priors, data, flat_coordinates=flat),
                init_strategy=init_to_value(values=values),
            ),
            num_warmup=500,
            num_samples=2000,
            progress_bar=False,
        )
        mcmc.run(jax.random.PRNGKey(3))
        return {k: onp.asarray(v) for k, v in mcmc.get_samples().items()}


@pytest.mark.parametrize("flat", [True, False])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_default_guide_agrees_with_nuts_on_a_binary(dtype, flat):
    data = _binary_data()
    result = fit(BINARY, BINARY_PRIORS, data, dtype=dtype)
    reference = _nuts(BINARY, BINARY_PRIORS, data, result.values, flat, dtype)
    vi = variational(
        BINARY,
        BINARY_PRIORS,
        data,
        start=result,
        steps=1500,
        flat_coordinates=flat,
        dtype=dtype,
    )
    assert vi.info["dense_start"]
    for name in BINARY_PRIORS:
        got, want = vi.samples[name], reference[name]
        assert got.dtype == onp.dtype(dtype)
        assert abs(got.mean() - want.mean()) < 0.25 * want.std(), name
        assert got.std() == pytest.approx(want.std(), rel=0.15), name


@pytest.mark.parametrize("flat", [True, False])
def test_samples_are_keyed_like_nuts(flat):
    """Draws are in the model's parameters under NUTS's site names, the
    angle of an AngleVector prior (degrees) and its vector included."""
    data = nrm_oidata().with_model(
        BinaryModelAngular(140.0, 60.0, 0.02), key=jax.random.PRNGKey(1)
    )
    priors = {
        "sep": dist.LogUniform(10.0, 500.0),
        "pa": AngleVector(),
        "flux": dist.LogUniform(1e-4, 1.0),
    }
    template = BinaryModelAngular(130.0, 50.0, 0.01)
    result = fit(template, priors, data)
    vi = variational(
        template,
        priors,
        data,
        start=result,
        guide="mvn",
        steps=300,
        num_samples=50,
        flat_coordinates=flat,
        psis=False,
    )
    reference = _nuts(template, priors, data, result.values, flat, "float64")
    assert (
        set(vi.samples)
        == set(reference)
        == {
            "sep",
            "pa",
            "pa_vec",
            "flux",
        }
    )
    for name, draws in vi.samples.items():
        assert draws.shape == (50,) + reference[name].shape[1:]
    # In the parameters, not the unconstrained coordinates.
    assert onp.all((vi.samples["sep"] > 10.0) & (vi.samples["sep"] < 500.0))
    assert abs(onp.median(vi.samples["sep"]) - 140.0) < 10.0
    assert vi.losses.shape == (300,)
    assert set(vi.info) >= {"guide", "elbo", "khat", "time", "dtype"}


def test_unknown_guide_is_rejected():
    with pytest.raises(ValueError, match="guide must be"):
        variational(
            _no_model, PRIORS, (), likelihoods=[_GaussianInZ(True)], guide="x"
        )
