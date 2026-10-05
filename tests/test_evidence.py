import jax
import jax.numpy as np
import numpy as onp
import pytest

from virgil.coverage import vlti_oidata
from virgil.fields import GaussianField
from virgil.fitting import FitResult, fit
from virgil.imaging import (
    LCurve,
    MaxEntropy,
    error_scale,
    image_priors,
    l_curve,
    log_evidence,
)
from virgil.models import GaussianDisk, Image, PointSource, System

from ._compiles import count_compiles

DATA = vlti_oidata(hour_angles_h=(-2.0, 0.0, 2.0), wavelengths_m=[3.5e-6])
N, H = 16, 1.0
TEMPLATE = onp.asarray(GaussianDisk(4.0).render(N, N * H))


def _gp_scene(latent, sigma, length=2.0):
    field = GaussianField(latent, sigma, length, mean=TEMPLATE)
    return System(star=PointSource(), env=Image(field, H, flux=0.4))


def test_the_evidence_prefers_the_field_amplitude_the_data_came_from():
    latent = jax.random.normal(jax.random.PRNGKey(0), (N, N))
    data = DATA.with_model(_gp_scene(latent, 1.5), key=jax.random.PRNGKey(1))
    evidence = {}
    for sigma in (0.3, 1.5, 6.0):
        start = _gp_scene(onp.zeros((N, N)), sigma)
        result = fit(start, image_priors(start), data)
        evidence[sigma] = log_evidence(result.model, data)
    assert max(evidence, key=evidence.get) == 1.5


def test_the_evidence_needs_a_gaussian_field():
    scene = System(
        star=PointSource(), env=Image.from_model(GaussianDisk(3.0), N, H)
    )
    with pytest.raises(TypeError, match="GaussianField"):
        log_evidence(scene, DATA)


def test_classic_maxent_lies_inside_the_sweep_and_is_bracketed():
    truth = System(
        star=PointSource(), env=GaussianDisk(3.0, dra=2.0, flux=0.4)
    )
    data = DATA.with_model(truth, key=jax.random.PRNGKey(2))
    start = System(
        star=PointSource(),
        env=Image.from_model(GaussianDisk(4.0), N, H, flux=0.4),
    )
    weights = np.logspace(-1, 3, 5)
    curve = l_curve(
        start, image_priors(start), data, MaxEntropy(1.0, path="env"), weights
    )
    weight = curve.classic_maxent(data)
    assert weight is not None and 0.1 < weight < 1000.0
    # A sweep confined to very strong weights does not bracket it.
    strong = l_curve(
        start,
        image_priors(start),
        data,
        MaxEntropy(1.0, path="env"),
        [3e4, 1e5],
    )
    assert strong.classic_maxent(data) is None


def test_classic_maxent_matches_an_independent_calculation():
    # Recompute the Gull-Skilling gap at each fit from scratch: a forward-
    # mode Jacobian with respect to log-brightness, the full pixel-space
    # curvature diag(1/√b) JᵀJ diag(1/√b), and its eigenvalues; then
    # interpolate the gap's zero crossing in log w.
    from virgil._precision import cast_tree, run_in
    from virgil.likelihood import whitened_residuals

    truth = System(
        star=PointSource(), env=GaussianDisk(3.0, dra=2.0, flux=0.4)
    )
    data = DATA.with_model(truth, key=jax.random.PRNGKey(3))
    start = System(
        star=PointSource(),
        env=Image.from_model(GaussianDisk(4.0), 12, 1.2, flux=0.4),
    )
    curve = l_curve(
        start,
        image_priors(start),
        data,
        MaxEntropy(1.0, path="env"),
        np.logspace(-1, 3, 5),
    )
    gaps = []
    for weight, penalty, result in zip(
        curve.weights, curve.penalty, curve.results
    ):
        with run_in("float64"):
            model, d = cast_tree((result.model, data), "float64")
            eta = model.env.log_brightness
            jac = jax.jacfwd(
                lambda x: whitened_residuals(
                    model.set("env.log_brightness", x), d
                )
            )(eta)
            jac = onp.asarray(jac).reshape(jac.shape[0], -1)
            b = onp.asarray(model.env.brightness).ravel()
        scaled = jac / onp.sqrt(b)
        lam = onp.linalg.eigvalsh(scaled.T @ scaled)
        n_good = onp.sum(lam / (lam + float(weight)))
        gaps.append(2.0 * float(weight) * float(penalty) - n_good)
    gaps = onp.asarray(gaps)
    i = int(onp.nonzero(onp.sign(gaps[:-1]) != onp.sign(gaps[1:]))[0][0])
    t = onp.log(onp.asarray(curve.weights[i : i + 2], dtype=float))
    expected = onp.exp(
        t[0] + gaps[i] / (gaps[i] - gaps[i + 1]) * (t[1] - t[0])
    )
    assert curve.classic_maxent(data) == pytest.approx(expected, rel=1e-4)


def test_error_scale_recovers_overestimated_errors():
    # Data simulated with errors twice the noise actually added: the scale
    # re-estimate is close to 1/2, and rescaling brings it back to 1.
    rich = vlti_oidata(
        hour_angles_h=(-3.0, -1.5, 0.0, 1.5, 3.0),
        wavelengths_m=[3.2e-6, 3.5e-6, 3.8e-6],
    )
    latent = jax.random.normal(jax.random.PRNGKey(4), (N, N))
    noisy = rich.with_model(
        _gp_scene(latent, 1.5), key=jax.random.PRNGKey(5), noise_scale=0.5
    )
    start = _gp_scene(onp.zeros((N, N)), 1.5)
    result = fit(start, image_priors(start), noisy)
    scale = error_scale(result.model, noisy)
    assert 0.4 < scale < 0.6
    rescaled = noisy.with_error_scale(scale)
    again = fit(start, image_priors(start), rescaled)
    assert 0.9 < error_scale(again.model, rescaled) < 1.1


def test_error_scale_rejects_bad_factors_and_pixel_images():
    with pytest.raises(ValueError, match="factor"):
        DATA.with_error_scale(0.0)
    scene = System(
        star=PointSource(), env=Image.from_model(GaussianDisk(3.0), N, H)
    )
    with pytest.raises(TypeError, match="GaussianField"):
        error_scale(scene, DATA)


def test_error_scale_solves_mackays_fixed_point():
    # s² = χ² / (N − γ) with γ = Σ βλ / (1 + βλ) evaluated at β = 1/s²
    # itself (MacKay 1992, eqs. 4.9-4.10), not at β = 1.
    from virgil._precision import cast_tree, run_in
    from virgil.likelihood import whitened_residuals

    latent = jax.random.normal(jax.random.PRNGKey(6), (N, N))
    noisy = DATA.with_model(
        _gp_scene(latent, 1.5), key=jax.random.PRNGKey(7), noise_scale=0.5
    )
    start = _gp_scene(onp.zeros((N, N)), 1.5)
    result = fit(start, image_priors(start), noisy)
    scale = error_scale(result.model, noisy)
    path = "env.log_brightness.latent"
    with run_in("float64"):
        model, d = cast_tree((result.model, noisy), "float64")

        def residuals(z):
            return whitened_residuals(model.set(path, z), d)

        z = model.get(path)
        r = onp.asarray(residuals(z))
        jac = onp.asarray(jax.jacfwd(residuals)(z)).reshape(r.size, -1)
    lam = onp.linalg.eigvalsh(jac.T @ jac).clip(0.0)
    beta = 1.0 / scale**2
    gamma = onp.sum(beta * lam / (1.0 + beta * lam))
    # N is the number of independent data, not of residuals (correlated
    # closure phases add penalty residuals that are not observations).
    n_data = noisy.n_independent
    assert scale**2 == pytest.approx((r @ r) / (n_data - gamma), rel=1e-6)


def test_evidence_helpers_reject_fitted_noise_and_model_lists():
    latent = onp.zeros((N, N))
    scene = _gp_scene(latent, 1.5)
    noisy_fit = FitResult(scene, {"noise.vis_scale": 1.2}, {})
    for helper in (log_evidence, error_scale):
        with pytest.raises(ValueError, match="noise"):
            helper(noisy_fit, DATA)
        with pytest.raises(TypeError, match="list of models"):
            helper([scene, scene], [DATA, DATA])
        # A FitResult without noise terms is accepted, like its model.
        clean_fit = FitResult(scene, {}, {})
        assert helper(clean_fit, DATA) == pytest.approx(helper(scene, DATA))
    pixels = System(
        star=PointSource(), env=Image.from_model(GaussianDisk(3.0), N, H)
    )
    curve = LCurve(
        weights=np.array([1.0]),
        chi2=np.array([1.0]),
        chi2_red=np.array([[1.0]]),
        penalty=np.array([1.0]),
        results=[FitResult(pixels, {"noise[0].phi_error": 0.1}, {})],
    )
    with pytest.raises(ValueError, match="noise"):
        curve.classic_maxent(DATA)


def test_the_evidence_jacobian_compiles_once():
    # The residual Jacobian is jitted once at module level. Run eagerly, it
    # compiled hundreds of small operations one at a time on its first call
    # (about 340 here); jitted, it is a handful of compilations.
    n = 11  # a grid size no other test uses, so nothing is cached

    def scene(sigma):
        field = GaussianField(onp.zeros((n, n)), sigma, 2.0)
        return System(star=PointSource(), env=Image(field, H, flux=0.4))

    first, second = scene(1.5), scene(3.0)
    with count_compiles() as compiles:
        log_evidence(first, DATA)
    assert len(compiles) < 20
    with count_compiles() as compiles:
        log_evidence(second, DATA)
        error_scale(second, DATA)
    assert not compiles
