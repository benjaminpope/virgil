"""Sampling a Gaussian-field image with numpyro's NUTS (Stage 5c)."""

import jax
import jax.numpy as np
import numpy as onp
import numpyro.distributions as dist
import pytest
from numpyro.infer import MCMC, NUTS, init_to_value
from numpyro.infer.util import initialize_model

from virgil.coverage import vlti_oidata
from virgil.fields import GaussianField
from virgil._precision import cast_tree, run_in
from virgil.fitting import _Objective, fit, gauss_newton_mass
from virgil.imaging import image_priors
from virgil.likelihood import numpyro_model
from virgil.models import GaussianDisk, Image, PointSource, System

N, H = 16, 1.0
TEMPLATE = onp.asarray(GaussianDisk(4.0).render(N, N * H))


def _scene(latent, sigma, length):
    field = GaussianField(latent, sigma, length, mean=TEMPLATE)
    return System(star=PointSource(), env=Image(field, H, flux=0.4))


@pytest.mark.skipif(
    not jax.config.jax_enable_x64, reason="NUTS on an image needs x64"
)
def test_nuts_runs_on_an_injected_field():
    # A smoke test: NUTS runs through numpyro_model, stays finite and
    # rarely diverges. Whether its posterior is calibrated (the injected
    # amplitude inside the credible interval, at the nominal rate) needs
    # many repeats, and is checked by the coverage campaign in
    # virgil-validation rather than by a single chain here.
    truth = _scene(jax.random.normal(jax.random.PRNGKey(10), (N, N)), 1.5, 2.0)
    data = vlti_oidata(
        hour_angles_h=(-2.0, 0.0, 2.0), wavelengths_m=[3.5e-6]
    ).with_model(truth, key=jax.random.PRNGKey(100))
    start = _scene(onp.zeros((N, N)), 1.0, 2.0)
    priors = image_priors(start) | {
        "env.log_brightness.sigma": dist.LogNormal(0.0, 1.0),
        "env.log_brightness.length_mas": dist.LogNormal(np.log(2.0), 0.5),
    }
    # Start at the MAP image, with the hyperparameters at the prior medians.
    latent = fit(start, image_priors(start), data).values
    init = latent | {
        "env.log_brightness.sigma": 1.0,
        "env.log_brightness.length_mas": 2.0,
    }
    mcmc = MCMC(
        NUTS(
            numpyro_model(start, priors, data),
            init_strategy=init_to_value(values=init),
        ),
        num_warmup=50,
        num_samples=50,
        progress_bar=False,
    )
    mcmc.run(jax.random.PRNGKey(0), extra_fields=("diverging",))
    samples = mcmc.get_samples()
    # Every prior site was sampled (and nothing else), so the finite check
    # below cannot pass vacuously on an empty or partial dict.
    assert set(samples) == set(priors)
    assert all(onp.all(onp.isfinite(onp.asarray(v))) for v in samples.values())
    assert int(mcmc.get_extra_fields()["diverging"].sum()) <= 5


@pytest.mark.slow
def test_gauss_newton_mass_matches_the_curvature_and_its_layout():
    # Fixed σ and ℓ, flux sampled: one dense block over both sites, in the
    # order of the priors, whose inverse is JᵀJ (+ I from the latents'
    # standard-normal priors) in numpyro's unconstrained coordinates.
    data = vlti_oidata(
        hour_angles_h=(-2.0, 0.0, 2.0), wavelengths_m=[3.5e-6]
    ).with_model(_scene(onp.zeros((N, N)), 1.5, 2.0))
    scene = _scene(onp.zeros((N, N)), 1.5, 2.0)
    priors = image_priors(scene) | {"env.flux": dist.Uniform(0.0, 1.0)}
    result = fit(scene, priors, data)
    mass = gauss_newton_mass(scene, priors, data, result.values)
    (paths,) = mass["dense_mass"]
    assert set(paths) == set(priors)
    assert mass["adapt_mass_matrix"] is False
    covariance = mass["inverse_mass_matrix"][paths]
    assert covariance.shape == (N * N + 1, N * N + 1)
    onp.testing.assert_allclose(covariance, covariance.T, atol=1e-10)
    # With no data, each latent's variance would be 1; the data shrink it.
    # Sites are laid out in the block's order; the flux is one number.
    start = 1 if paths[0] == "env.flux" else 0
    variances = onp.diag(covariance)[start : start + N * N]
    assert onp.all(variances <= 1.0 + 1e-9) and variances.min() < 0.5


def _dense_gauss_newton(model, priors, data, values):
    """(JᵀJ)⁻¹ with J the full Jacobian of the residuals, priors included."""
    with run_in("float64"):
        problem = cast_tree(_Objective(model, priors, data), "float64")
        z = problem.init(cast_tree(values, "float64"))
        paths = problem.paths
        shapes = [np.shape(z[p]) for p in paths]
        flat = np.concatenate([np.ravel(z[p]) for p in paths])

        def residuals(x):
            sizes = onp.cumsum([int(onp.prod(s)) for s in shapes])[:-1]
            pieces = np.split(x, sizes)
            zz = {p: v.reshape(s) for p, v, s in zip(paths, pieces, shapes)}
            return problem.residuals(zz)

        jac = onp.asarray(jax.jacfwd(residuals)(flat))
    return onp.linalg.inv(jac.T @ jac)


@pytest.mark.parametrize("case", ["latents", "latents and flux", "disk"])
def test_gauss_newton_mass_is_the_inverse_curvature(case):
    # The covariance is computed from the data's Jacobian and the priors'
    # diagonal curvature: by Woodbury when there are fewer data than
    # parameters (an image, with or without a flat-prior flux), else
    # directly. Each must equal (JᵀJ)⁻¹ from the full Jacobian.
    data = vlti_oidata(
        hour_angles_h=(-2.0, 0.0, 2.0), wavelengths_m=[3.5e-6]
    ).with_model(_scene(onp.zeros((N, N)), 1.5, 2.0))
    if case == "disk":
        scene = System(star=PointSource(), env=GaussianDisk(3.0, flux=0.3))
        priors = {
            "env.sigma": dist.Normal(3.0, 1.0),
            "env.flux": dist.Uniform(0.0, 1.0),
        }
        values = {"env.sigma": 3.2, "env.flux": 0.35}
    else:
        scene = _scene(onp.zeros((N, N)), 1.5, 2.0)
        priors = image_priors(scene)
        if case == "latents and flux":
            priors |= {"env.flux": dist.Uniform(0.0, 1.0)}
        latent = jax.random.normal(jax.random.PRNGKey(3), (N, N))
        values = {"env.log_brightness.latent": 0.3 * latent, "env.flux": 0.4}
        values = {k: v for k, v in values.items() if k in priors}
    mass = gauss_newton_mass(scene, priors, data, values)
    (paths,) = mass["dense_mass"]
    covariance = mass["inverse_mass_matrix"][paths]
    expected = _dense_gauss_newton(scene, priors, data, values)
    scale = onp.max(onp.abs(expected))
    onp.testing.assert_allclose(covariance, expected, atol=1e-9 * scale)


def test_gauss_newton_mass_rejects_an_unconstrained_parameter():
    # A zero-flux companion's position leaves the data unchanged.
    data = vlti_oidata(hour_angles_h=(0.0,), wavelengths_m=[3.5e-6])
    scene = System(star=PointSource(), c=PointSource(flux=0.0, dra=5.0))
    priors = {"c.dra": dist.Uniform(-10.0, 10.0)}
    with pytest.raises(ValueError, match="singular"):
        gauss_newton_mass(scene, priors, data, {"c.dra": 5.0})


@pytest.mark.slow
@pytest.mark.skipif(
    not jax.config.jax_enable_x64, reason="the Hessian check needs x64"
)
def test_gauss_newton_mass_whitens_the_posterior():
    # The data make the posterior much narrower in the measured directions
    # than in the rest, so its Hessian is badly conditioned; the Gauss-Newton
    # covariance, used as NUTS's inverse mass matrix, whitens it near the MAP.
    truth = _scene(jax.random.normal(jax.random.PRNGKey(10), (N, N)), 1.5, 2.0)
    data = vlti_oidata(
        hour_angles_h=(-3.0, -1.5, 0.0, 1.5, 3.0),
        wavelengths_m=[3.2e-6, 3.5e-6, 3.8e-6],
    ).with_model(truth, key=jax.random.PRNGKey(100))
    scene = _scene(onp.zeros((N, N)), 1.5, 2.0)
    priors = image_priors(scene) | {"env.flux": dist.Uniform(0.0, 1.0)}
    result = fit(scene, priors, data)

    info = initialize_model(
        jax.random.PRNGKey(1),
        numpyro_model(result.model, priors, data),
        init_strategy=init_to_value(values=result.values),
    )
    mass = gauss_newton_mass(scene, priors, data, result.values)
    (paths,) = mass["dense_mass"]
    cov = mass["inverse_mass_matrix"][paths]

    # The dense block is the sites in `paths` order, each raveled row-major.
    z = info.param_info.z
    shapes = [z[k].shape for k in paths]
    sizes = [int(onp.prod(s)) for s in shapes]

    def potential(flat):
        parts = np.split(flat, onp.cumsum(sizes)[:-1])
        return info.potential_fn(
            {k: p.reshape(s) for k, p, s in zip(paths, parts, shapes)}
        )

    flat = np.concatenate([np.ravel(z[k]) for k in paths])
    hess = onp.asarray(jax.hessian(potential)(flat))
    chol = onp.linalg.cholesky(onp.asarray(cov))
    raw = onp.linalg.eigvalsh(hess)
    white = onp.linalg.eigvalsh(chol.T @ hess @ chol)
    assert raw[-1] / raw[0] > 1e3
    assert 0.25 < white[0] and white[-1] < 4
