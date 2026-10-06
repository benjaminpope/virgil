"""fit in each prior's flat coordinate.

A prior that is uniform in some coordinate of its parameter (LogUniform in
log x, an isotropic inclination in cos i) is fitted in that coordinate,
where it adds nothing to the loss (design/imaging_plan.md, standing
decision 9). Levenberg–Marquardt then works with these priors, and the MAP
is the maximum of the likelihood inside the prior's range. Everything here
is tiny: a binary on a 7-hole mask, an 8×8 Gaussian-field image.
"""

import jax
import jax.numpy as np
import numpy as onp
import numpyro.distributions as dist
import pytest
from numpyro.distributions.transforms import biject_to

from virgil.angles import AngleVector
from virgil.coverage import nrm_oidata
from virgil.fields import GaussianField
from virgil.fitting import _Objective, fit, gauss_newton_mass
from virgil.imaging import image_priors
from virgil.inference import laplace_cov
from virgil.models import (
    BinaryModelAngular,
    BinaryModelCartesian,
    Image,
    PointSource,
    System,
)
from virgil.priors import IsotropicInclination, IsotropicLatitude
from virgil.scenes import gaussian_blob

TRUTH = BinaryModelCartesian(60.0, -40.0, 0.05)
DATA = nrm_oidata(sigma_v2=0.005, sigma_cp_deg=0.3).with_model(
    TRUTH, key=jax.random.PRNGKey(0)
)
START = BinaryModelCartesian(55.0, -35.0, 0.03)
POSITION = {
    "dra": dist.Uniform(-200.0, 200.0),
    "ddec": dist.Uniform(-200.0, 200.0),
}
LOG_UNIFORM = POSITION | {"flux": dist.LogUniform(1e-4, 0.5)}


def _no_model(**_):
    return None


def _gaussian_term(f, mu, s):
    """A likelihood term: a Gaussian of f(x) with mean mu and width s."""
    return lambda v: np.atleast_1d((f(v["x"]) - mu) / s)


def test_lm_is_chosen_for_log_uniform_and_isotropic_priors():
    result = fit(START, LOG_UNIFORM, DATA)
    assert result.info["method"] == "lm"
    assert result.info["converged"] is True
    assert abs(float(result.values["flux"]) - 0.05) < 5e-3

    term = _gaussian_term(lambda i: np.cos(np.deg2rad(i)), 0.5, 0.05)
    priors = {"x": IsotropicInclination()}
    result = fit(_no_model, priors, (), init={"x": 80.0}, likelihoods=[term])
    assert result.info["method"] == "lm"
    # Flat in cos i, so the MAP is where the likelihood peaks: cos i = 0.5.
    assert float(result.values["x"]) == pytest.approx(60.0, abs=1e-3)

    term = _gaussian_term(np.sin, 0.3, 0.05)
    priors = {"x": IsotropicLatitude()}
    result = fit(_no_model, priors, (), init={"x": -0.4}, likelihoods=[term])
    assert result.info["method"] == "lm"
    # Flat in sin(lat): the MAP is at sin(lat) = 0.3.
    assert float(result.values["x"]) == pytest.approx(onp.arcsin(0.3), 1e-5)


def test_isotropic_flat_coordinates_are_affine_in_cos_i_and_sin_lat():
    """The declared coordinate (the CDF) is cos i or sin(lat), rescaled."""
    with jax.enable_x64(True):
        inc = IsotropicInclination(10.0, 120.0)
        to_flat, from_flat, low, high = inc.flat_coordinate()
        i = np.linspace(15.0, 115.0, 7)
        u = to_flat(i)
        c = np.cos(np.deg2rad(i))
        slope = (u[-1] - u[0]) / (c[-1] - c[0])
        assert np.allclose(u - u[0], slope * (c - c[0]), atol=1e-12)
        assert np.allclose(from_flat(u), i, atol=1e-10)
        assert (low, high) == (0.0, 1.0)
        lat = IsotropicLatitude(-0.5, 1.0)
        to_flat, from_flat, _, _ = lat.flat_coordinate()
        x = np.linspace(-0.4, 0.9, 7)
        u, s = to_flat(x), np.sin(x)
        slope = (u[-1] - u[0]) / (s[-1] - s[0])
        assert np.allclose(u - u[0], slope * (s - s[0]), atol=1e-12)
        assert np.allclose(from_flat(u), x, atol=1e-10)


def test_angle_vector_priors_still_fit_beside_flat_coordinates():
    """An AngleVector position angle (its own vector path, #211) next to a
    LogUniform flux: LM runs, agrees with L-BFGS, and gives the same
    angle and separation as with a Uniform flux (both priors are flat)."""
    truth = BinaryModelAngular(70.0, 5.0, 0.05)
    data = nrm_oidata(sigma_v2=0.005, sigma_cp_deg=0.3).with_model(
        truth, key=jax.random.PRNGKey(3)
    )
    template = BinaryModelAngular(65.0, 355.0, 0.03)
    priors = {
        "sep": dist.Uniform(10.0, 200.0),
        "pa": AngleVector(),
        "flux": dist.LogUniform(1e-4, 0.5),
    }
    lm = fit(template, priors, data, gtol=1e-8)
    assert lm.info["method"] == "lm"
    assert lm.info["converged"] is True
    assert "pa_vec" in lm.values
    assert abs(float(lm.values["pa"]) - 5.0) < 2.0
    lbfgs = fit(template, priors, data, method="lbfgs", gtol=1e-8)
    uniform = fit(
        template, priors | {"flux": dist.Uniform(1e-4, 0.5)}, data, gtol=1e-8
    )
    for other in (lbfgs, uniform):
        for path in ("sep", "pa", "flux"):
            assert np.allclose(
                lm.values[path], other.values[path], rtol=1e-5
            ), path


def test_lm_agrees_with_lbfgs_in_the_flat_coordinates():
    options = {"gtol": 1e-8}
    lm = fit(START, LOG_UNIFORM, DATA, method="lm", **options)
    lbfgs = fit(START, LOG_UNIFORM, DATA, method="lbfgs", **options)
    for path in LOG_UNIFORM:
        assert np.allclose(lm.values[path], lbfgs.values[path], rtol=1e-6)
    assert lm.info["loss"] == pytest.approx(lbfgs.info["loss"], rel=1e-9)


def test_a_flat_coordinate_prior_adds_nothing_to_the_map():
    """LogUniform and Uniform on the same range: the same MAP (the
    likelihood's maximum), since neither is fitted with a density."""
    uniform = POSITION | {"flux": dist.Uniform(1e-4, 0.5)}
    a = fit(START, LOG_UNIFORM, DATA, gtol=1e-8)
    b = fit(START, uniform, DATA, gtol=1e-8)
    for path in LOG_UNIFORM:
        assert np.allclose(a.values[path], b.values[path], rtol=1e-6)


@pytest.mark.parametrize("method", ["lm", "lbfgs"])
def test_no_data_map_and_laplace_covariance_transform(method):
    """A Gaussian likelihood in log x under LogUniform on x.

    In the flat coordinate u = log x the posterior is N(mu, s²), so the
    MAP is x = exp(mu) and, by the delta method, var x = x² s². (The old
    natural-coordinate MAP was the mode of exp(-(log x - mu)²/2s²) / x,
    at x = exp(mu - s²).)
    """
    mu, s = np.log(0.05), 0.3
    priors = {"x": dist.LogUniform(1e-3, 1.0)}
    term = _gaussian_term(np.log, mu, s)
    result = fit(
        _no_model,
        priors,
        (),
        init={"x": 0.2},
        likelihoods=[term],
        method=method,
        gtol=1e-8,
    )
    x = float(result.values["x"])
    assert x == pytest.approx(0.05, rel=1e-5)
    with jax.enable_x64(True):
        problem = _Objective(_no_model, priors, (), likelihoods=[term])
        z = problem.init({"x": x})
        curvature = jax.hessian(lambda v: problem.loss({"x": v}))(z["x"])
        dx_dz = jax.grad(lambda v: problem.constrain({"x": v})["x"])(z["x"])
        var_x = float(dx_dz**2 / curvature)  # the delta method
    assert var_x == pytest.approx(x**2 * s**2, rel=1e-5)


def test_gauss_newton_mass_is_in_numpyros_coordinates():
    """The mass matrix is the natural-coordinate Laplace covariance
    carried into numpyro's z by the delta method, Σ_z = T⁻¹ Σ_x T⁻¹ with
    T = dx/dz, for a LogUniform flux as for Uniform positions."""
    data = nrm_oidata(sigma_v2=0.005, sigma_cp_deg=0.3).with_model(TRUTH)
    values = {"dra": 60.0, "ddec": -40.0, "flux": 0.05}
    mass = gauss_newton_mass(TRUTH, LOG_UNIFORM, data, values)
    (paths,) = mass["inverse_mass_matrix"]
    (covariance,) = mass["inverse_mass_matrix"].values()
    # Noiseless data: the Gauss–Newton matrix is the Hessian at the truth.
    sigma_x = onp.asarray(
        laplace_cov(
            onp.array([values[p] for p in paths]), list(paths), TRUTH, data
        )
    )
    with jax.enable_x64(True):
        t = onp.array(
            [
                jax.grad(biject_to(LOG_UNIFORM[p].support))(
                    biject_to(LOG_UNIFORM[p].support).inv(
                        np.asarray(values[p], float)
                    )
                )
                for p in paths
            ]
        )
    expected = sigma_x / onp.outer(t, t)
    assert onp.allclose(covariance, expected, rtol=1e-4)


def test_uniform_and_normal_priors_are_unchanged():
    """Uniform and Normal keep numpyro's bijection and their densities."""
    priors = {
        "ddec": dist.Uniform(-200.0, 200.0),
        "flux": dist.Uniform(0.0, 0.5),
        "dra": dist.Normal(58.0, 2.0),
    }
    with jax.enable_x64(True):
        problem = _Objective(START, priors, DATA)
        z = problem.init()
        for path, prior in priors.items():
            expected = biject_to(prior.support).inv(
                np.asarray(START.get(path), float)
            )
            assert np.allclose(z[path], expected, rtol=1e-12)
        shifted = {p: v + 0.1 for p, v in z.items()}
        values = [problem.constrain(w) for w in (z, shifted)]
        r = [problem.residuals(w) for w in (z, shifted)]
        # The Normal prior is still a residual, and the loss its density.
        assert r[0].size == DATA.n_residuals + 1
        assert np.isclose(r[0][-1], (55.0 - 58.0) / 2.0)
        change = problem.loss(shifted) - problem.loss(z)
        half = 0.5 * (np.sum(r[1] ** 2) - np.sum(r[0] ** 2))
        assert np.isclose(change, half, rtol=1e-8)
        log_prior = [priors["dra"].log_prob(v["dra"]) for v in values]
        assert np.isclose(
            0.5 * (r[1][-1] ** 2 - r[0][-1] ** 2),
            -(log_prior[1] - log_prior[0]),
            rtol=1e-8,
        )


def test_flat_coordinate_noise_priors_add_nothing():
    """A LogUniform error scale is flat in log s, like the parameters."""
    priors = POSITION | {"flux": dist.Uniform(1e-4, 0.5)}
    noise = {"vis_scale": dist.LogUniform(0.1, 10.0)}
    with jax.enable_x64(True):
        problem = _Objective(START, priors, DATA, noise=noise)
        z = problem.init()
        site = "noise.vis_scale"
        assert float(problem.constrain(z)[site]) == pytest.approx(1.0)
        # z is the logit of log s on [log 0.1, log 10]: s = 1 is the middle.
        assert float(z[site]) == pytest.approx(0.0, abs=1e-12)


def test_gp_image_with_a_log_uniform_flux_converges_with_lm():
    """A tiny GP image: a LogUniform flux takes LM's path, in about as
    many steps as a Uniform one, to the same solution."""
    npix, pixel = 8, 20.0
    data = nrm_oidata(sigma_v2=0.005, sigma_cp_deg=0.3)
    truth = System(
        star=PointSource(),
        env=Image.from_brightness(
            gaussian_blob(npix, pixel, 30.0, dra=10.0), pixel, flux=0.2
        ),
    )
    data = data.with_model(truth, key=jax.random.PRNGKey(2))
    field = GaussianField(onp.zeros((npix, npix)), 1.0, 40.0)
    start = System(star=PointSource(), env=Image(field, pixel, flux=0.1))
    results = {}
    for name, prior in {
        "uniform": dist.Uniform(1e-3, 1.0),
        "log_uniform": dist.LogUniform(1e-3, 1.0),
    }.items():
        priors = image_priors(start) | {"env.flux": prior}
        results[name] = fit(start, priors, data)
    a, b = results["uniform"], results["log_uniform"]
    assert b.info["method"] == "lm"
    assert b.info["converged"] is True
    assert b.info["steps"] <= 2 * a.info["steps"] + 5
    assert b.info["loss"] == pytest.approx(a.info["loss"], rel=1e-4)
    assert float(b.values["env.flux"]) == pytest.approx(
        float(a.values["env.flux"]), rel=1e-3
    )
