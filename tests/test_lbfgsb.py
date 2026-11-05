"""fit(method="lbfgsb"): box bounds from linear pixels and Uniform priors."""

import jax
import jax.numpy as np
import numpy as onp
import numpyro.distributions as dist
import pytest

from virgil.coverage import ami_grid_record
from virgil.fitting import _Objective, _value_and_gradient, fit
from virgil.imaging import TSV, MaxEntropy, image_priors
from virgil.models import (
    BinaryModelCartesian,
    LinearImage,
    PointSource,
    System,
)
from virgil.oidata import OIData

from ._test_data import oidata

NPIX, SCALE = 10, 10.0
DATA = OIData(ami_grid_record(pitch_m=0.7))


@pytest.fixture(autouse=True)
def _x64():
    with jax.enable_x64(True):
        yield


def _disc(flux):
    """A flux-``flux`` disc in the middle of the grid, zero in the corners."""
    yy, xx = onp.mgrid[:NPIX, :NPIX] - (NPIX - 1) / 2
    disc = (xx**2 + yy**2 < 9.0).astype(float)
    return flux * disc / disc.sum()


def _scene(pixels, parametrisation="linear", **kwargs):
    env = LinearImage(
        pixels,
        SCALE,
        parametrisation=parametrisation,
        reference_flux=0.5,
        **kwargs,
    )
    return System(star=PointSource(), env=env)


def test_linear_pixels_are_non_negative_and_satisfy_kkt():
    truth = _scene(_disc(0.5))
    data = DATA.with_model(truth, key=jax.random.PRNGKey(1), noise_scale=3.0)
    start = _scene(onp.full((NPIX, NPIX), 0.5 / NPIX**2))
    priors = image_priors(start)
    regularizers = [TSV(0.02, path="env", beam_mas=60.0)]
    result = fit(
        start, priors, data, regularizers, method="lbfgsb", max_steps=2000
    )
    pix = onp.asarray(result.values["env.pixels"])
    assert result.info["converged"]
    assert pix.min() == 0.0
    zero = pix == 0.0
    assert 0 < zero.sum() < pix.size
    # KKT of min f(x) s.t. x >= 0: g = 0 off the bound and g >= 0 on it.
    problem = _Objective(start, priors, data, regularizers, boxed=True)
    z = {"env.pixels": np.asarray(pix)}
    scale = float(data.n_independent)
    _, grad = _value_and_gradient(
        problem, z, np.asarray(scale), np.ravel(z["env.pixels"])
    )
    grad = onp.asarray(grad).reshape(pix.shape)
    tol = 1e-4  # fit's default gtol, per data point: independent of pg_norm
    assert grad[zero].min() >= -tol
    assert onp.abs(grad[~zero]).max() <= tol
    assert result.info["pg_norm"] == result.info["grad_norm"]
    assert {"nit", "message"} <= set(result.info)


def test_support_pixels_stay_zero():
    support = _disc(1.0) > 0
    # Start lit everywhere: the bounds, not the loss, must zero the rest.
    start = _scene(onp.full((NPIX, NPIX), 0.5 / NPIX**2), support=support)
    data = DATA.with_model(_scene(_disc(0.5)), noise_scale=0.0)
    result = fit(
        start,
        image_priors(start),
        data,
        [TSV(0.02, path="env", beam_mas=60.0)],
        method="lbfgsb",
    )
    pix = onp.asarray(result.values["env.pixels"])
    assert onp.all(pix[~support] == 0.0)
    assert onp.all(pix >= 0.0)


def test_agrees_with_lbfgs_on_an_interior_convex_problem():
    # Skilling's entropy on sqrt pixels has an interior, unique minimum.
    truth = _scene(onp.sqrt(_disc(0.5) + 1e-3), "sqrt")
    data = DATA.with_model(truth, noise_scale=0.0)
    start = _scene(onp.full((NPIX, NPIX), onp.sqrt(0.5 / NPIX**2)), "sqrt")
    args = (start, image_priors(start), data, [MaxEntropy(1.0, path="env")])
    boxed = fit(*args, method="lbfgsb", max_steps=3000)
    free = fit(*args, method="lbfgs", max_steps=3000)
    assert boxed.info["converged"] and free.info["converged"]
    assert boxed.info["loss"] == pytest.approx(free.info["loss"], rel=1e-4)
    a = onp.asarray(boxed.model.get("env").fluxes)
    b = onp.asarray(free.model.get("env").fluxes)
    assert onp.abs(a - b).max() < 1e-2 * b.max()


def test_uniform_prior_bounds_are_respected_and_reported():
    truth = BinaryModelCartesian(150.0, -80.0, 0.02)
    data = oidata.with_model(truth, key=jax.random.PRNGKey(0))
    priors = {
        "dra": dist.Uniform(-400.0, 400.0),
        "ddec": dist.Uniform(-400.0, 400.0),
        "flux": dist.Uniform(0.0, 0.01),  # the truth, 0.02, is outside
    }
    start = BinaryModelCartesian(140.0, -70.0, 0.005)
    result = fit(start, priors, data, method="lbfgsb")
    assert result.info["converged"]
    assert result.values["flux"] == pytest.approx(0.01, abs=1e-12)
    assert "flux" in result.info["at_bound"]
    assert -400.0 < result.values["dra"] < 400.0


def test_refuses_priors_without_a_box_and_warns_on_float32():
    truth = BinaryModelCartesian(150.0, -80.0, 0.02)
    data = oidata.with_model(truth, key=jax.random.PRNGKey(0))
    start = BinaryModelCartesian(140.0, -70.0, 0.01)
    with pytest.raises(TypeError, match="Uniform"):
        fit(start, {"dra": dist.Normal(0.0, 100.0)}, data, method="lbfgsb")
    with pytest.warns(UserWarning, match="float64"):
        fit(
            start,
            {"flux": dist.Uniform(0.0, 0.5)},
            data,
            method="lbfgsb",
            dtype="float32",
        )


def test_precondition_is_refused():
    start = BinaryModelCartesian(140.0, -70.0, 0.01)
    data = oidata.with_model(start, noise_scale=0.0)
    with pytest.raises(ValueError, match="precondition"):
        fit(
            start,
            {"flux": dist.Uniform(0.0, 0.5)},
            data,
            method="lbfgsb",
            precondition="diag",
        )
