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
    tol = 1e-3  # absolute per data point (TSV's kinks make it platform-sensitive)
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


def _nnls_problem():
    rng = onp.random.default_rng(3)
    a = rng.normal(size=(40, 16))
    x = onp.zeros(16)
    x[[1, 5, 6, 12]] = [1.0, 0.5, 2.0, 0.7]
    b = a @ x + 0.3 * rng.normal(size=40)

    def term(values):
        return a @ np.ravel(values["pixels"]) - b

    return a, b, term


def _nnls_fit(**kwargs):
    from scipy.optimize import nnls

    a, b, term = _nnls_problem()
    start = LinearImage(np.full((4, 4), 0.5), 1.0)
    priors = image_priors(start)
    result = fit(
        start, priors, (), likelihoods=[term], method="lbfgsb", **kwargs
    )
    return a, b, nnls, result


def test_agrees_with_nnls_on_a_linear_problem():
    a, b, nnls, result = _nnls_fit(gtol=1e-9, max_steps=500)
    want, _ = nnls(a, b)
    got = onp.asarray(result.values["pixels"]).ravel()
    assert onp.abs(got - want).max() < 1e-6 * want.max()
    assert onp.array_equal(got == 0.0, want == 0.0)


def test_time_limit_and_max_steps_stop_reasons():
    with pytest.warns(RuntimeWarning, match="time_limit"):
        *_, timed = _nnls_fit(time_limit=0.0)
    assert timed.info["stop"] == "time" and not timed.info["converged"]
    with pytest.warns(RuntimeWarning, match="step limit|max_steps"):
        *_, capped = _nnls_fit(max_steps=2, gtol=1e-12)
    assert capped.info["stop"] == "limit"


@pytest.mark.parametrize("flux_first", [True, False])
def test_mixed_model_bounds_follow_the_prior_order(flux_first):
    from virgil.fitting import _box

    start = _scene(onp.full((NPIX, NPIX), 0.005))
    flux = ("star.flux", dist.Uniform(0.1, 0.9))
    pixels = next(iter(image_priors(start).items()))
    items = [flux, pixels] if flux_first else [pixels, flux]
    problem = _Objective(start, dict(items), DATA, boxed=True)
    lower, upper = _box(problem, problem.init())
    n = NPIX**2
    at = slice(0, 1) if flux_first else slice(n, n + 1)
    rest = slice(1, None) if flux_first else slice(0, n)
    assert (lower[at], upper[at]) == ([0.1], [0.9])
    assert onp.all(lower[rest] == 0.0) and onp.all(upper[rest] == onp.inf)


def test_sqrt_pixels_are_unbounded():
    from virgil.fitting import _box

    start = _scene(onp.full((NPIX, NPIX), 0.05), "sqrt")
    problem = _Objective(start, image_priors(start), DATA, boxed=True)
    lower, upper = _box(problem, problem.init())
    assert onp.all(lower == -onp.inf) and onp.all(upper == onp.inf)


def test_a_start_outside_the_bounds_is_clipped_with_a_warning():
    start = BinaryModelCartesian(140.0, -70.0, 0.01)
    data = oidata.with_model(start, noise_scale=0.0)
    with pytest.warns(UserWarning, match="clipped"):
        result = fit(
            start,
            {"flux": dist.Uniform(0.0, 0.5)},
            data,
            init={"flux": 0.7},
            method="lbfgsb",
        )
    assert 0.0 <= result.values["flux"] <= 0.5


def test_a_softmax_image_with_a_support_fits_finitely():
    from virgil.models import Image

    support = _disc(1.0) > 0
    env = Image(np.zeros((NPIX, NPIX)), SCALE, support=support)
    start = System(star=PointSource(), env=env)
    data = DATA.with_model(_scene(_disc(0.5)), noise_scale=0.0)
    result = fit(
        start,
        image_priors(start),
        data,
        [TSV(0.02, path="env", beam_mas=60.0)],
        method="lbfgsb",
        max_steps=50,
    )
    assert onp.isfinite(result.info["loss"])
    assert onp.all(onp.isfinite(result.info["pg_norm"]))


def test_a_uniform_prior_on_linear_pixels_still_fits_with_lbfgs():
    start = LinearImage(np.full((4, 4), 0.5), 1.0)
    _, _, term = _nnls_problem()
    priors = {"pixels": dist.Uniform(0.0, 5.0).expand((4, 4)).to_event(2)}
    fit(start, priors, (), likelihoods=[term], method="lbfgs", max_steps=20)
    with pytest.raises(ValueError, match="method must be"):
        fit(start, image_priors(start), (), likelihoods=[term], method="nope")


def test_a_line_search_that_meets_an_undefined_point_restarts(monkeypatch):
    # The model is NaN at an all-zero linear image (0/0), a corner L-BFGS-B's
    # line search can clip a trial point to on some platforms; NaN there ends
    # it with "ABNORMAL" after a few steps. Make the 6th evaluation undefined
    # (about the second trial point) and check the fit restarts and converges.
    import virgil.fitting as fitting

    truth = _scene(_disc(0.5))
    data = DATA.with_model(truth, key=jax.random.PRNGKey(1), noise_scale=3.0)
    start = _scene(onp.full((NPIX, NPIX), 0.5 / NPIX**2))
    args = (
        start,
        image_priors(start),
        data,
        [TSV(0.02, path="env", beam_mas=60.0)],
    )
    clean = fit(*args, method="lbfgsb", max_steps=3000)
    original, calls = fitting._value_and_gradient, []

    def undefined_once(problem, template, scale, x):
        calls.append(1)
        value, grad = original(problem, template, scale, x)
        if len(calls) == 6:
            return value * np.nan, grad * np.nan
        return value, grad

    monkeypatch.setattr(fitting, "_value_and_gradient", undefined_once)
    result = fit(*args, method="lbfgsb", max_steps=3000)
    assert clean.info["converged"] and result.info["converged"], result.info
    assert result.info["loss"] == pytest.approx(clean.info["loss"], rel=1e-6)
