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
@pytest.mark.slow
def test_nuts_runs_on_an_injected_field():
    # A smoke test: NUTS runs through numpyro_model, stays finite and
    # does not diverge. Whether its posterior is calibrated (the injected
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
    # A divergence is a failed trajectory, and a well-conditioned posterior
    # started at its MAP should give almost none. "None" was too strict for
    # this short chain: 50 warmup steps adapt the step size and mass matrix
    # coarsely, so one stray trajectory in 50 draws (seen on CI runners whose
    # rounding differs) is not a sampler fault. A systematic failure (a bad
    # mass layout, a hard wall) diverges in a large fraction of draws, so cap
    # the fraction at 5%; the calibrated rate is virgil-validation's job.
    diverging = onp.asarray(mcmc.get_extra_fields()["diverging"])
    assert diverging.mean() <= 0.05


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


_IMPROPER_FLUX = dist.ImproperUniform(dist.constraints.positive, (), ())


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
    # directly. Each must equal (JᵀJ)⁻¹ from the full Jacobian. The flux
    # priors are improper (a bounded Uniform adds the curvature of its
    # transform; see tests/test_fitting.py).
    data = vlti_oidata(
        hour_angles_h=(-2.0, 0.0, 2.0), wavelengths_m=[3.5e-6]
    ).with_model(_scene(onp.zeros((N, N)), 1.5, 2.0))
    if case == "disk":
        scene = System(star=PointSource(), env=GaussianDisk(3.0, flux=0.3))
        priors = {
            "env.sigma": dist.Normal(3.0, 1.0),
            "env.flux": _IMPROPER_FLUX,
        }
        values = {"env.sigma": 3.2, "env.flux": 0.35}
    else:
        scene = _scene(onp.zeros((N, N)), 1.5, 2.0)
        priors = image_priors(scene)
        if case == "latents and flux":
            priors |= {"env.flux": _IMPROPER_FLUX}
        latent = jax.random.normal(jax.random.PRNGKey(3), (N, N))
        values = {"env.log_brightness.latent": 0.3 * latent, "env.flux": 0.4}
        values = {k: v for k, v in values.items() if k in priors}
    mass = gauss_newton_mass(scene, priors, data, values)
    (paths,) = mass["dense_mass"]
    covariance = mass["inverse_mass_matrix"][paths]
    expected = _dense_gauss_newton(scene, priors, data, values)
    scale = onp.max(onp.abs(expected))
    onp.testing.assert_allclose(covariance, expected, atol=1e-9 * scale)


def test_gauss_newton_mass_gives_a_bounded_parameter_the_transforms_curvature():
    # A zero-flux companion's position leaves the data unchanged. Under a
    # bounded prior the sampler's logistic transform still curves the
    # potential, 2σ(1 - σ) for the fraction σ of the range, so the matrix is
    # finite (it raised before): the variance is its inverse.
    data = vlti_oidata(hour_angles_h=(0.0,), wavelengths_m=[3.5e-6])
    scene = System(star=PointSource(), c=PointSource(flux=0.0, dra=5.0))
    priors = {"c.dra": dist.Uniform(-10.0, 10.0)}
    mass = gauss_newton_mass(scene, priors, data, {"c.dra": 5.0})
    (sites,) = mass["dense_mass"]
    fraction = 0.75
    onp.testing.assert_allclose(
        mass["inverse_mass_matrix"][sites],
        [[1.0 / (2 * fraction * (1 - fraction))]],
        rtol=1e-9,
    )
    assert mass.condition == pytest.approx(1.0)


def test_gauss_newton_mass_rejects_an_unconstrained_parameter():
    # An improper prior adds no curvature, so a direction the data leave
    # free is singular: nothing constrains it.
    data = vlti_oidata(hour_angles_h=(0.0,), wavelengths_m=[3.5e-6])
    scene = System(star=PointSource(), c=PointSource(flux=0.0, dra=5.0))
    priors = {
        "c.dra": dist.ImproperUniform(dist.constraints.real, (), ()),
    }
    with pytest.raises(ValueError, match="singular"):
        gauss_newton_mass(scene, priors, data, {"c.dra": 5.0})


def test_gauss_newton_mass_allows_no_data():
    # With data=() the curvature is the priors' alone: a Normal prior's
    # covariance is its variance.
    scene = System(star=PointSource(), c=PointSource(flux=0.1, dra=5.0))
    priors = {"c.dra": dist.Normal(5.0, 2.0)}
    mass = gauss_newton_mass(scene, priors, (), {"c.dra": 5.0})
    (paths,) = mass["dense_mass"]
    onp.testing.assert_allclose(
        mass["inverse_mass_matrix"][paths], [[4.0]], rtol=1e-6
    )


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


def _small_scene(npix=4):
    template = onp.asarray(GaussianDisk(1.0).render(npix, npix * H))
    field = GaussianField(onp.zeros((npix, npix)), 1.5, 2.0, mean=template)
    return System(star=PointSource(), env=Image(field, H, flux=0.4))


@pytest.mark.parametrize("hours", [(0.0,), (-2.0, 0.0, 2.0)])
@pytest.mark.parametrize("flux", [False, True])
def test_the_size_guard_returns_the_diagonal_of_the_dense_matrix(
    monkeypatch, hours, flux
):
    # More coordinates than _DENSE_MASS_MAX: the diagonal of the covariance,
    # without forming it. Fewer rows than coordinates takes the Woodbury
    # branch (with the zero-P correction if the flux is improper), more the
    # Cholesky one.
    from virgil import fitting

    scene = _small_scene()
    data = vlti_oidata(hour_angles_h=hours, wavelengths_m=[3.5e-6]).with_model(
        scene
    )
    priors = image_priors(scene)
    if flux:
        priors |= {"env.flux": _IMPROPER_FLUX}
    values = {
        "env.log_brightness.latent": 0.3
        * jax.random.normal(jax.random.PRNGKey(3), (4, 4)),
        "env.flux": 0.4,
    }
    values = {k: v for k, v in values.items() if k in priors}
    dense = gauss_newton_mass(scene, priors, data, values)
    (sites,) = dense["dense_mass"]
    assert dense.condition is not None
    monkeypatch.setattr(fitting, "_DENSE_MASS_MAX", 10)
    with pytest.warns(UserWarning, match="diagonal"):
        thin = gauss_newton_mass(scene, priors, data, values)
    assert thin["dense_mass"] == [] and thin.condition is None
    # One block over all the sites, a vector: numpyro takes it as a diagonal
    # block (test_a_nuts_chain_runs_with_the_diagonal_mass).
    ((thin_sites, diagonal),) = thin["inverse_mass_matrix"].items()
    assert thin_sites == sites and diagonal.ndim == 1
    covariance = dense["inverse_mass_matrix"][sites]
    if hours == (0.0,):  # fewer rows than coordinates: Woodbury, exact
        expected = onp.diag(covariance)
    else:  # at least as many rows as coordinates: 1 / diag(F)
        expected = 1.0 / onp.diag(onp.linalg.inv(covariance))
    onp.testing.assert_allclose(diagonal, expected, rtol=1e-10)


def _shapes(jaxpr, primitive=None):
    """Every array shape in a jaxpr, nested calls included, or (with
    ``primitive``) the input shapes of that primitive's equations."""
    found = set()
    if primitive is None:
        for var in [*jaxpr.invars, *jaxpr.outvars]:
            found.add(tuple(getattr(var.aval, "shape", ())))
    for eqn in jaxpr.eqns:
        if primitive is None:
            for var in [*eqn.invars, *eqn.outvars]:
                found.add(tuple(getattr(var.aval, "shape", ())))
        elif eqn.primitive.name == primitive:
            found.add(tuple(eqn.invars[0].aval.shape))
        for param in eqn.params.values():
            for sub in param if isinstance(param, (list, tuple)) else [param]:
                inner = getattr(sub, "jaxpr", sub)
                if hasattr(inner, "eqns"):
                    found |= _shapes(inner, primitive)
    return found


@pytest.mark.parametrize("hours", [(0.0,), (-2.0, 0.0, 2.0)])
def test_the_diagonal_covariance_forms_no_n_by_n_matrix(hours):
    # The size guard's diagonal must not allocate or factor n x n, or the
    # guard saves nothing. With fewer rows than coordinates (Woodbury) no
    # n x n array exists at all; with more, the Jacobian is already at least
    # n x n, but F is not formed or Cholesky-factored.
    from virgil.fitting import _covariance

    scene = _small_scene()
    data = vlti_oidata(hour_angles_h=hours, wavelengths_m=[3.5e-6]).with_model(
        scene
    )
    priors = image_priors(scene)
    problem = _Objective(scene, priors, data, flat=False)
    z = problem.init({"env.log_brightness.latent": onp.zeros((4, 4))})
    n = 16
    rows = jax.eval_shape(problem.curvature_rows, z).shape[0]
    dense = jax.make_jaxpr(lambda z: _covariance(problem, z, True))(z)
    thin = jax.make_jaxpr(lambda z: _covariance(problem, z, False))(z)
    # The helper sees nested calls: the dense matrix is factored.
    assert (n, n) in _shapes(dense.jaxpr, "cholesky")
    assert (n, n) not in _shapes(thin.jaxpr, "cholesky")
    if rows < n:
        assert (n, n) in _shapes(dense.jaxpr)
        assert (n, n) not in _shapes(thin.jaxpr)


def test_a_non_positive_definite_covariance_is_not_ok_in_float32():
    # e7b: a 16² image, errors 1000 times smaller, float32. The Woodbury
    # cancellation gave a covariance that was not positive definite while
    # the factors looked fine, and the old check passed it.
    from virgil.fitting import _gauss_newton_covariance

    scene = _scene(onp.zeros((N, N)), 1.5, 2.0)
    data = vlti_oidata(
        hour_angles_h=(-2.0, 0.0, 2.0), wavelengths_m=[3.5e-6]
    ).with_model(scene)
    priors = image_priors(scene)
    latent = 0.3 * jax.random.normal(jax.random.PRNGKey(3), (N, N))
    with run_in("float64"):
        problem = _Objective(
            scene, priors, data.with_error_scale(0.001), flat=False
        )
        z = problem.init({"env.log_brightness.latent": latent})
    problem32, z32 = cast_tree((problem, z), "float32")
    _, ok, _ = _gauss_newton_covariance(problem32, z32)
    assert not ok
    # The same curvature in float64 is fine.
    with run_in("float64"):
        _, ok64, _ = _gauss_newton_covariance(
            *cast_tree((problem, z), "float64")
        )
    assert ok64


def test_a_nuts_chain_runs_with_the_diagonal_mass(monkeypatch):
    from numpyro.infer import MCMC, NUTS

    from virgil import fitting

    scene = _small_scene()
    data = vlti_oidata(
        hour_angles_h=(0.0,), wavelengths_m=[3.5e-6]
    ).with_model(scene)
    priors = image_priors(scene)
    values = {"env.log_brightness.latent": onp.zeros((4, 4))}
    monkeypatch.setattr(fitting, "_DENSE_MASS_MAX", 10)
    with pytest.warns(UserWarning, match="diagonal"):
        mass = gauss_newton_mass(scene, priors, data, values)
    kernel = NUTS(
        numpyro_model(scene, priors, data),
        init_strategy=init_to_value(values=values),
        **mass,
    )
    mcmc = MCMC(kernel, num_warmup=10, num_samples=10, progress_bar=False)
    mcmc.run(jax.random.PRNGKey(0))
    samples = mcmc.get_samples()
    assert set(samples) == set(priors)
    assert all(onp.all(onp.isfinite(onp.asarray(v))) for v in samples.values())


def test_the_real_size_guard_returns_a_diagonal_above_the_limit():
    # A 65 × 65 image has 4225 latents, more than _DENSE_MASS_MAX (4096),
    # with two visibilities' rows: the diagonal, without an n × n matrix.
    from virgil import fitting

    npix = 65
    assert npix * npix > fitting._DENSE_MASS_MAX
    scene = _small_scene(npix)
    data = vlti_oidata(
        hour_angles_h=(0.0,), wavelengths_m=[3.5e-6]
    ).with_model(scene)
    priors = image_priors(scene)
    values = {"env.log_brightness.latent": np.zeros((npix, npix))}
    with pytest.warns(UserWarning, match="diagonal"):
        mass = gauss_newton_mass(scene, priors, data, values)
    ((_, diagonal),) = mass["inverse_mass_matrix"].items()
    assert diagonal.shape == (npix * npix,) and onp.all(diagonal > 0)
    assert mass["dense_mass"] == [] and mass.condition is None
