"""fit(precondition="diag") and gauss_newton_diag (design/fit_preconditioning.md, B).

Small problems only: a quadratic with a known diagonal metric, a binary on a
7-hole mask, and a 16² image.
"""

import jax
import jax.numpy as np
import numpy as onp
import numpyro.distributions as dist
import pytest

from virgil._precision import cast_tree, run_in
from virgil.coverage import ami_grid_record, nrm_oidata
from virgil.fitting import _Objective, fit, gauss_newton_diag
from virgil.imaging import MaxEntropy, TSV, image_priors
from virgil.models import BinaryModelCartesian, Image, PointSource, System
from virgil.oidata import OIData
from virgil.scenes import gaussian_blob

TRUTH = BinaryModelCartesian(60.0, -40.0, 0.05)
DATA = nrm_oidata(sigma_v2=0.005, sigma_cp_deg=0.3).with_model(
    TRUTH, key=jax.random.PRNGKey(0)
)
START = BinaryModelCartesian(55.0, -35.0, 0.03)
PRIORS = {
    "dra": dist.Uniform(-200.0, 200.0),
    "ddec": dist.Uniform(-200.0, 200.0),
    "flux": dist.LogUniform(1e-4, 0.5),
}

# A quadratic loss with Fisher diagonal s², spanning 2.8 decades (gradients
# stay above gtol, so a stop means a minimum).
SCALES = onp.logspace(0, 1.4, 24)


def _no_model(**_):
    return None


def _quadratic_term(v):
    return SCALES * (v["x"] - 1.0)


def _quadratic_fit(**options):
    priors = {"x": dist.ImproperUniform(dist.constraints.real, (), (24,))}
    return fit(
        _no_model,
        priors,
        (),
        init={"x": np.zeros(24)},
        likelihoods=[_quadratic_term],
        method="lbfgs",
        **options,
    )


def _image_fit(npix=16):
    data = OIData(ami_grid_record(pitch_m=0.5))
    truth = System(
        star=PointSource(),
        env=Image.from_brightness(
            gaussian_blob(npix, 12.0, 25.0, dra=20.0), 12.0, flux=0.1
        ),
    )
    data = data.with_model(truth, key=jax.random.PRNGKey(1))
    start = System(
        star=PointSource(), env=Image(np.zeros((npix, npix)), 12.0, flux=0.1)
    )
    return start, image_priors(start), data


def test_the_diagonal_is_that_of_the_gauss_newton_matrix():
    values = {"dra": 60.0, "ddec": -40.0, "flux": 0.05}
    diag = gauss_newton_diag(START, PRIORS, DATA, values)
    with run_in("float64"):
        problem = cast_tree(
            _Objective(START, PRIORS, DATA, flat=True), "float64"
        )
        z = problem.init(cast_tree(values, "float64"))
        flat, unravel = problem.ravel(z)
        jac = jax.jacfwd(lambda x: problem.curvature_rows(unravel(x)))(flat)
        expected = onp.sum(onp.asarray(jac) ** 2, axis=0) + onp.asarray(
            problem.curvature_diag(z, False)
        )
    got = onp.concatenate(
        [onp.ravel(diag[k]) for k in problem.coordinate_keys]
    )
    onp.testing.assert_allclose(got, expected, rtol=1e-6)
    # Probes agree to their sampling error, and exactly for rank-one rows.
    sampled = gauss_newton_diag(START, PRIORS, DATA, values, n_probes=400)
    sampled = onp.concatenate(
        [onp.ravel(sampled[k]) for k in problem.coordinate_keys]
    )
    onp.testing.assert_allclose(sampled, expected, rtol=0.35)


def test_the_regularizer_curvature_is_included_and_estimated_by_probes():
    start, priors, data = _image_fit(8)
    values = {"env.log_brightness": start.env.log_brightness}
    kwargs = dict(regularizers=[TSV(5.0, path="env")])
    exact = gauss_newton_diag(start, priors, data, values, **kwargs)
    plain = gauss_newton_diag(start, priors, data, values)
    key = "env.log_brightness"
    assert onp.all(exact[key] >= plain[key] - 1e-9)
    assert onp.any(exact[key] > plain[key] + 1e-6)
    sampled = gauss_newton_diag(
        start, priors, data, values, n_probes=300, **kwargs
    )
    onp.testing.assert_allclose(
        onp.sum(sampled[key]), onp.sum(exact[key]), rtol=0.2
    )


def test_max_entropy_curvature_is_the_weight_times_the_image():
    # In the Image's softmax logits the Skilling-Bryan metric w b is
    # w b (1 - b): the unit-sum constraint subtracts the b² terms.
    for n, eta in ((8, None), (3, "random")):
        start, priors, data = _image_fit(n)
        logits = start.env.log_brightness
        if eta:
            logits = jax.random.normal(jax.random.PRNGKey(0), logits.shape)
        diag = gauss_newton_diag(
            start,
            priors,
            data,
            {"env.log_brightness": logits},
            regularizers=[MaxEntropy(7.0, path="env")],
        )["env.log_brightness"]
        b = jax.nn.softmax(logits.ravel()).reshape(logits.shape)
        onp.testing.assert_allclose(diag, 7.0 * b * (1.0 - b), rtol=1e-6)
        if eta:  # far from w b, which the 3 x 3 image can tell apart
            assert onp.max(onp.abs(diag / (7.0 * b) - 1.0)) > 0.1


def test_the_entropy_metric_replaces_the_data_curvature_on_its_pixels():
    start, priors, data = _image_fit(8)
    values = {"env.log_brightness": start.env.log_brightness}
    regularizers = [MaxEntropy(7.0, path="env")]
    once = gauss_newton_diag(
        start, priors, data, values, regularizers=regularizers
    )
    twice = gauss_newton_diag(
        start, priors, [data, data], values, regularizers=regularizers
    )
    onp.testing.assert_allclose(
        twice["env.log_brightness"], once["env.log_brightness"], rtol=1e-6
    )


def test_lm_fits_maximum_entropy():
    start, priors, data = _image_fit(8)
    regularizers = [MaxEntropy(50.0, path="env")]
    lm = fit(start, priors, data, regularizers, method="lm")
    lbfgs = fit(start, priors, data, regularizers, method="lbfgs")
    assert lm.info["converged"]
    onp.testing.assert_allclose(lm.info["loss"], lbfgs.info["loss"], rtol=1e-3)


def test_a_known_diagonal_metric_is_removed():
    plain = _quadratic_fit()
    scaled = _quadratic_fit(precondition="diag")
    assert scaled.info["converged"] and plain.info["converged"]
    onp.testing.assert_allclose(scaled.values["x"], 1.0, atol=1e-4)
    # One diagonal Newton step is exact (the line search may add a few).
    assert scaled.info["steps"] <= 5
    assert scaled.info["steps"] < plain.info["steps"]
    assert scaled.info["precondition"] == "diag"
    assert plain.info["precondition"] is None
    assert plain.info["precondition_rounds"] == []
    # The quadratic's diagonal spans 2.8 decades, none of it floored.
    (stats,) = scaled.info["precondition_rounds"]
    assert stats["decades"] == pytest.approx(2.8, abs=0.1)
    assert stats["floored"] == 0.0


@pytest.mark.filterwarnings("ignore:fit.*did not converge")
def test_adam_with_a_metric_reaches_the_same_optimum():
    priors = {"x": dist.ImproperUniform(dist.constraints.real, (), (24,))}
    result = fit(
        _no_model,
        priors,
        (),
        init={"x": np.zeros(24)},
        likelihoods=[_quadratic_term],
        method="adam",
        max_steps=3000,
        learning_rate=0.1,
        precondition="diag",
    )
    onp.testing.assert_allclose(result.values["x"], 1.0, atol=1e-2)
    assert result.info["precondition"] == "diag"


@jax.enable_x64(True)
def test_adam_with_a_metric_follows_the_unchunked_step(monkeypatch):
    import optax

    import virgil.fitting
    from virgil.fitting import _InMetric, _metric, _scaled_loss

    monkeypatch.setattr(virgil.fitting, "_ADAM_CHUNK", 7)
    problem = cast_tree(_Objective(START, PRIORS, (DATA,)), "float64")
    z0 = problem.init(None)
    scale = float(DATA.n_independent)
    metric, _ = _metric(problem, z0, scale)
    reference = _InMetric(problem, metric.normalized())
    grad = jax.grad(_scaled_loss(reference, scale))
    optimizer = optax.adam(1e-2)
    zt = jax.tree.map(np.zeros_like, z0)
    state = optimizer.init(zt)
    for _ in range(40):
        updates, state = optimizer.update(grad(zt), state, zt)
        zt = optax.apply_updates(zt, updates)
    expected = problem.constrain(metric.normalized().to_z(zt))
    result = fit(
        START, PRIORS, DATA, method="adam", max_steps=40, precondition="diag"
    )
    for path, value in expected.items():
        onp.testing.assert_allclose(result.values[path], value, rtol=1e-12)


@pytest.mark.filterwarnings("ignore:fit.*did not converge")
def test_adam_reports_a_non_finite_loss():
    priors = {"x": dist.ImproperUniform(dist.constraints.real, (), (24,))}
    result = fit(
        _no_model,
        priors,
        (),
        init={"x": np.zeros(24)},
        likelihoods=[_quadratic_term],
        method="adam",
        max_steps=60,
        learning_rate=1e200,
    )
    assert result.info["stop"] == "non-finite"
    assert result.info["converged"] is False


def test_adam_ftol_with_a_metric():
    result = fit(
        START,
        PRIORS,
        DATA,
        method="adam",
        max_steps=500,
        precondition="diag",
        ftol_abs=1e30,
        ftol_window=5,
    )
    assert result.info["steps"] == 9 and result.info["converged"] is True
    assert result.info["loss_history"].shape == (9,)


def test_the_optimum_and_the_gradient_norm_are_unchanged():
    start, priors, data = _image_fit()
    regularizers = [MaxEntropy(10.0, path="env")]
    # A tighter gtol than the default, which stops where a flat direction
    # still has a few tenths of a percent of the loss to give.
    options = dict(gtol=1e-6)
    plain = fit(start, priors, data, regularizers, **options)
    scaled = fit(
        start, priors, data, regularizers, precondition="diag", **options
    )
    assert plain.info["converged"] and scaled.info["converged"]
    # The gradient test is in the unscaled coordinates, whatever the metric.
    assert scaled.info["grad_norm"] < 1e-5
    assert scaled.info["loss"] == pytest.approx(plain.info["loss"], rel=2e-3)
    onp.testing.assert_allclose(
        scaled.model.env.brightness, plain.model.env.brightness, atol=1e-2
    )
    # The softmax gauge is fixed: log-brightness is the log of the pixel fluxes.
    x = scaled.values["env.log_brightness"]
    assert float(jax.nn.logsumexp(x)) == pytest.approx(0.0, abs=1e-5)


def test_lm_ignores_the_option_and_bad_values_are_rejected():
    result = fit(START, PRIORS, DATA, precondition="diag")
    assert result.info["method"] == "lm"
    assert result.info["precondition"] is None
    with pytest.raises(ValueError, match="precondition"):
        fit(START, PRIORS, DATA, precondition="full")


def test_a_proper_pixel_prior_is_not_given_the_gauge_projection():
    # A Normal prior is not invariant to a constant shift of the pixels, so
    # the mean must follow it from 0 towards -2. (The two fits may end in
    # different local minima of this softmax image; a warm start from the plain
    # optimum must stay there.)
    start, _, data = _image_fit(8)
    shape = (8, 8)
    priors = {
        "env.log_brightness": dist.Normal(-2.0 * np.ones(shape), 1.0).to_event(
            2
        )
    }
    options = dict(gtol=1e-8, method="lbfgs")
    plain = fit(start, priors, data, **options)
    scaled = fit(start, priors, data, precondition="diag", **options)
    assert scaled.info["converged"]
    assert float(np.mean(scaled.values["env.log_brightness"])) < -0.5
    again = fit(
        start, priors, data, init=plain.values, precondition="diag", **options
    )
    assert again.info["loss"] == pytest.approx(plain.info["loss"], rel=1e-6)


def test_every_image_in_a_scene_is_found_by_one_visitor():
    from virgil.imaging import image_priors
    from virgil.models import Rotated, _scene_images

    a = Image(np.zeros((4, 4)), 10.0)
    b = Image(np.zeros((3, 3)), 10.0)
    scene = System(first=a, second=Rotated(b, 30.0))
    found = {prefix for prefix, _ in _scene_images(scene)}
    assert found == {"first.", "second.source."}
    priors = image_priors(scene)
    assert set(priors) == {
        "first.log_brightness",
        "second.source.log_brightness",
    }


def test_the_gauge_masks_follow_the_support_of_each_flat_image():
    from virgil.fitting import _gauge_masks

    support = onp.ones((4, 4), bool)
    support[0] = False
    scene = System(
        first=Image(np.zeros((4, 4)), 10.0, support=support),
        second=PointSource(),
    )
    priors = image_priors(scene)
    problem = _Objective(scene, priors, ())
    masks = _gauge_masks(problem)
    assert set(masks) == {"first.log_brightness"}
    onp.testing.assert_array_equal(masks["first.log_brightness"], support)


def test_probes_are_redrawn_each_round_and_a_new_round_does_not_recompile():
    from virgil.fitting import _fisher_diagonal, _lbfgs_run

    start, priors, data = _image_fit(8)
    problem = _Objective(start, priors, data)
    z = problem.init()
    first = _fisher_diagonal(problem, z, 4, 0)
    again = _fisher_diagonal(problem, z, 4, 0)
    second = _fisher_diagonal(problem, z, 4, 1)
    onp.testing.assert_array_equal(first, again)
    assert not onp.allclose(first, second)
    # Rounds with different metrics share one compiled loop.
    from virgil.fitting import _lbfgs

    class Budget:
        chunked = False

    before = _lbfgs_run._cached._cache_size()
    for scale in (1.0, 2.0):
        _lbfgs(
            problem,
            z,
            np.asarray(10.0 * scale),
            np.asarray(40),
            np.asarray(1e-4),
            np.asarray(2.0),
            10,
            Budget(),
            True,
        )
    assert _lbfgs_run._cached._cache_size() - before <= 1


def _gauge_problem(npix=8):
    """An image-only scene whose TSV optimum has dark pixels (an l_curve point)."""
    truth = Image.from_brightness(
        gaussian_blob(npix, 12.0, 25.0, dra=20.0)
        + 0.3 * gaussian_blob(npix, 12.0, 3.0, dra=-30.0),
        12.0,
    )
    data = OIData(ami_grid_record(pitch_m=0.5)).with_model(
        truth, key=jax.random.PRNGKey(1)
    )
    start = Image(np.zeros((npix, npix)), 12.0)
    return start, image_priors(start), data


def _lit(x, fraction):
    """Pixels brighter than ``fraction`` of the brightest. The gradient in a
    pixel's ``log_brightness`` is proportional to its flux, so a stopped fit
    fixes it only as well as ``gtol / flux``: a pixel at 1e-9 of the peak is
    uncertain by 1e-2 in log, and a darker one is a boundary optimum at -inf."""
    return onp.exp(x - x.max()) > fraction


def test_recentring_changes_no_loss_and_fixes_the_log_flux():
    from virgil.fitting import _recentre

    start, priors, data = _gauge_problem()
    key = "log_brightness"
    x = onp.random.default_rng(0).normal(size=(8, 8))
    x[0, :3] = -500.0  # dark pixels are left where they are
    with run_in("float64"):
        problem = cast_tree(
            _Objective(start, priors, data, [TSV(1e2)]), "float64"
        )
        z = {key: np.asarray(x + 1e5)}
        out = _recentre(problem, z)[key]
        assert float(jax.nn.logsumexp(out)) == pytest.approx(0.0, abs=1e-12)
        assert float(np.max(out)) < 0.0
        onp.testing.assert_allclose(
            out - out[3, 3], z[key] - z[key][3, 3], atol=1e-9
        )
        assert float(problem.loss({key: out})) == pytest.approx(
            float(problem.loss(z)), rel=1e-12
        )
        # A pixel at -inf (no flux) stays there.
        x[3, 3] = -onp.inf
        out = _recentre(problem, {key: np.asarray(x)})[key]
        assert float(jax.nn.logsumexp(out)) == pytest.approx(0.0, abs=1e-12)
        assert out[3, 3] == -onp.inf and onp.isfinite(out[0, 0])


def test_lm_sees_the_gauge_direction_through_one_extra_residual():
    # fit recentres its result, so only the residuals LM steps on can show that
    # its own half works: the data do not see a constant added to the pixels
    # (their Jacobian times ones is 0), and the gauge row alone does.
    from virgil.fitting import _gauge_masks, _lm_residuals

    start, priors, data = _gauge_problem()
    with run_in("float64"):
        problem = cast_tree(
            _Objective(start, priors, data, [TSV(1e2)]), "float64"
        )
        masks = _gauge_masks(problem)
        z = {
            "log_brightness": np.asarray(
                onp.random.default_rng(0).normal(size=(8, 8)) - 4.0
            )
        }
        args = (problem, np.asarray(10.0), masks)
        n = len(_lm_residuals(z, args)) - 1
        ones = {"log_brightness": np.ones((8, 8))}
        along = jax.jvp(lambda z: _lm_residuals(z, args), (z,), (ones,))[1]
        scale = float(np.max(np.abs(_lm_residuals(z, args)[:n])))
        assert float(np.max(np.abs(along[:n]))) < 1e-9 * scale
        assert float(along[n]) == pytest.approx(1.0, abs=1e-12)


@pytest.mark.filterwarnings("ignore:fit.*did not converge")
@pytest.mark.parametrize(
    "method, precondition",
    [("lm", None), ("lbfgs", None), ("lbfgs", "diag"), ("adam", "diag")],
)
def test_the_fitted_log_brightness_is_gauge_fixed_for_every_method(
    method, precondition
):
    # LM left the lit pixels 4e5 apart from zero here, set by rounding; the
    # fit must be at log of the fluxes, and a start shifted by 1e5 must give
    # the same lit pixels as an unshifted one.
    start, priors, data = _gauge_problem()
    options = dict(
        method=method,
        precondition=precondition,
        max_steps=300 if method != "lm" else None,
    )
    warm = None
    for weight in (1e3, 3e2, 1e2):
        result = fit(start, priors, data, [TSV(weight)], init=warm, **options)
        x = onp.asarray(result.values["log_brightness"])
        assert onp.all(onp.isfinite(x))
        # deliberate: the gauge is exact only up to rounding in the working
        # dtype (one float32 ulp, about 2e-7, under float32).
        tol = (
            1e-9
            if x.dtype == onp.float64
            else float(8 * onp.finfo(x.dtype).eps)
        )
        assert float(jax.nn.logsumexp(x)) == pytest.approx(0.0, abs=tol)
        assert x[_lit(x, 1e-8)].min() > -25.0  # ln(1e-8 / 64 pixels) is -22.6
        onp.testing.assert_allclose(
            result.model.brightness,
            jax.nn.softmax(x.ravel()).reshape(x.shape),
            atol=1e-12,
        )
        warm = result.values
    if method != "lm":  # the others stop unconverged, and may amplify 1e-11
        return
    base = onp.asarray(result.values["log_brightness"], float)
    runs = [
        fit(
            start,
            priors,
            data,
            [TSV(1e2)],
            init={"log_brightness": b},
            **options,
        )
        for b in (base, base + 1e5)
    ]
    assert runs[1].info["loss"] == pytest.approx(
        runs[0].info["loss"], rel=1e-6
    )
    x0, x1 = (onp.asarray(r.values["log_brightness"]) for r in runs)
    lit = _lit(x0, 1e-3)
    onp.testing.assert_allclose(x1[lit], x0[lit], atol=1e-4)
    onp.testing.assert_allclose(
        runs[1].model.brightness, runs[0].model.brightness, atol=1e-7
    )


def _l_curve_steps(dtype, npix=41):
    from virgil.coverage import vlti_oidata
    from virgil.imaging import beam, field_of_view, l_curve, starting_image
    from virgil.scenes import ring

    templates = [
        vlti_oidata(hour_angles_h=(-3.0, -1.5, 0.0, 1.5, 3.0)),
        vlti_oidata(hour_angles_h=(-2.25, -0.75, 0.75, 2.25)),
    ]
    scale = field_of_view(templates[0]) / npix
    truth_image = 0.8 * ring(
        npix,
        scale,
        6.0,
        1.0,
        inc_deg=50.0,
        pa_deg=30.0,
        asymmetry=0.4,
        asymmetry_pa_deg=90.0,
    ) + 0.2 * gaussian_blob(npix, scale, 0.5, dra=-7.0, ddec=5.0)
    truth = System(
        star=PointSource(),
        env=Image.from_brightness(
            truth_image / truth_image.sum(), scale, flux=0.5
        ),
    )
    nights = [
        t.with_model(truth, key=jax.random.PRNGKey(k))
        for k, t in enumerate(templates)
    ]
    start = starting_image(nights, hole_mas=0.5 * beam(nights).minor_mas)
    curve = l_curve(
        start,
        image_priors(start),
        nights,
        MaxEntropy(1.0, path="env"),
        np.logspace(4, -1, 6),
        dtype=dtype,
        precondition="diag",
    )
    return [(r.info["steps"], r.info["converged"]) for r in curve.results]


@pytest.mark.slow
# float32 is JAX's default dtype: it is asserted, not xfailed. It converged at
# every weight when run (the old mark was non-strict);
# if the slow tier ever sees it stall, that is a regression to report raw.
@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_the_l_curve_converges_at_every_weight(dtype):
    """Stage 7: flat-prior MaxEntropy, w = 0.1 ... 1e4, default step limit."""
    runs = _l_curve_steps(dtype)
    assert all(converged for _, converged in runs), runs
