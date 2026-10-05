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


def test_the_evidence_is_finite_on_high_signal_to_noise_data():
    # Tiny errors make J huge; the same data twice make J J^T rank
    # deficient. A Cholesky factor of I + J J^T then fails to rounding
    # (error ~ eps |J J^T| > 1); the singular values of J give
    # log det(I + J^T J) without forming it, and the duplicate doubles the
    # Gaussian log-likelihood exactly.
    latent = jax.random.normal(jax.random.PRNGKey(0), (N, N))
    precise = DATA.with_model(
        _gp_scene(latent, 1.5), key=jax.random.PRNGKey(1), noise_scale=0.0
    ).with_error_scale(1e-7)
    start = _gp_scene(onp.zeros((N, N)), 1.5)
    model = fit(start, image_priors(start), precise).model
    assert onp.isfinite(log_evidence(model, [precise, precise]))


def test_the_evidence_log_determinant_matches_a_direct_one():
    from virgil.imaging import _residual_jacobian

    latent = jax.random.normal(jax.random.PRNGKey(0), (N, N))
    data = DATA.with_model(_gp_scene(latent, 1.5), key=jax.random.PRNGKey(1))
    start = _gp_scene(onp.zeros((N, N)), 1.5)
    model = fit(start, image_priors(start), data).model
    r, jac = _residual_jacobian(model, data, "env.log_brightness.latent")
    z = onp.asarray(model.get("env.log_brightness.latent"), dtype=float)
    _, logdet = onp.linalg.slogdet(onp.eye(jac.shape[0]) + jac @ jac.T)
    direct = -0.5 * float(r @ r) - 0.5 * onp.sum(z**2) - 0.5 * logdet
    assert log_evidence(model, data) == pytest.approx(direct, rel=1e-10)


@pytest.mark.parametrize("n", [N, 3])
def test_the_batched_jacobian_matches_jax(n):
    # The residual Jacobian is built in batches of VJPs (more latents than
    # residuals) or JVPs (fewer); both must equal jax.jacrev/jacfwd, which
    # push every basis vector through at once and run out of memory on
    # large datasets.
    from virgil.imaging import _residual_jacobian
    from virgil.likelihood import whitened_residuals

    latent = jax.random.normal(jax.random.PRNGKey(3), (n, n))
    template = onp.asarray(GaussianDisk(4.0).render(n, n * H))
    field = GaussianField(latent, 1.5, 2.0, mean=template)
    model = System(star=PointSource(), env=Image(field, H, flux=0.4))
    path = "env.log_brightness.latent"
    r, jac = _residual_jacobian(model, DATA, path)

    from virgil.fitting import cast_tree, run_in

    with run_in("float64"):  # as _residual_jacobian does
        model64, data64 = cast_tree((model, DATA), "float64")

        def residuals(x):
            return whitened_residuals(model64.set(path, x), data64)

        full = jax.jacrev(residuals)(model64.get(path)).reshape(r.size, -1)
    assert (r.size < latent.size) == (n == N)  # both branches are exercised
    onp.testing.assert_allclose(jac, onp.asarray(full), rtol=1e-10, atol=1e-12)


def test_plain_data_have_no_marginal_log_normaliser():
    # The evidence of plain data is unchanged by the marginal normaliser:
    # it is exactly zero without gains, offsets or extra observables.
    from virgil.imaging import _log_norm

    scene = _gp_scene(onp.zeros((N, N)), 1.5)
    assert _log_norm(scene, DATA) == 0.0
    assert _log_norm(scene, [DATA, DATA]) == 0.0


# Several channels, so that gains common to a baseline's channels stand out
# from independent noise.
CHROMATIC = vlti_oidata(
    hour_angles_h=(-2.0, 0.0, 2.0),
    wavelengths_m=onp.linspace(3.2e-6, 3.8e-6, 5),
    sigma_v2=0.01,
)
NG = 8


def _small_scene(latent, sigma=1.5):
    template = onp.asarray(GaussianDisk(4.0).render(NG, NG * H))
    field = GaussianField(latent, sigma, 2.0, mean=template)
    return System(star=PointSource(), env=Image(field, H, flux=0.4))


def test_the_evidence_with_gains_matches_model_loglike():
    from virgil._precision import cast_tree, run_in
    from virgil.imaging import _residual_jacobian
    from virgil.likelihood import model_loglike, whitened_residuals

    latent = jax.random.normal(jax.random.PRNGKey(5), (NG, NG))
    plain = CHROMATIC.with_model(
        _small_scene(latent), key=jax.random.PRNGKey(6)
    )
    gains = plain.with_gains(telescope=0.03, baseline=0.05)
    start = _small_scene(onp.zeros((NG, NG)))
    model = fit(start, image_priors(start), gains).model
    path = "env.log_brightness.latent"
    _, jac = _residual_jacobian(model, gains, path)
    z = onp.asarray(model.get(path), dtype=float)
    singular = onp.linalg.svd(jac, compute_uv=False)
    occam = -0.5 * onp.sum(z**2) - 0.5 * onp.sum(onp.log1p(singular**2))
    likelihood_term = log_evidence(model, gains) - occam
    with run_in("float64"):
        m, g, p = cast_tree((model, gains, plain), "float64")
        exact = float(model_loglike(m, g))
        # The documented constant: the normalisation of the same data
        # without gains, which depends only on the quoted errors.
        r = whitened_residuals(m, p)
        constant = float(model_loglike(m, p) + 0.5 * np.sum(r**2))
    assert likelihood_term == pytest.approx(exact - constant, abs=1e-6)
    # The gains' normaliser is not negligible here: without it the two
    # would differ.
    r = onp.asarray(_residual_jacobian(model, gains, path)[0])
    assert abs(-0.5 * float(r @ r) - (exact - constant)) > 1.0


def test_the_evidence_prefers_the_simulated_gain_width():
    latent = jax.random.normal(jax.random.PRNGKey(7), (NG, NG))
    truth = _small_scene(latent)
    width = 0.05
    data = CHROMATIC.with_gains(baseline=width).with_model(
        truth, key=jax.random.PRNGKey(8)
    )
    from virgil.imaging import _log_norm

    start = _small_scene(onp.zeros((NG, NG)))
    evidence, normaliser = {}, {}
    for trial in (0.005, width, 0.5):
        trial_data = data.with_gains(baseline=trial)
        result = fit(start, image_priors(start), trial_data)
        evidence[trial] = log_evidence(result.model, trial_data)
        normaliser[trial] = _log_norm(result.model, trial_data)
    assert max(evidence, key=evidence.get) == width
    # Without the gains' normaliser, wider gains would always win.
    bare = {t: evidence[t] + normaliser[t] for t in evidence}
    assert max(bare, key=bare.get) == 0.5


def test_laplace_samples_have_the_gauss_newton_covariance():
    from virgil.imaging import _residual_jacobian, laplace_samples

    n = 4
    latent = jax.random.normal(jax.random.PRNGKey(9), (n, n))
    template = onp.asarray(GaussianDisk(4.0).render(n, n * H))
    field = GaussianField(latent, 1.5, 2.0, mean=template)
    truth = System(star=PointSource(), env=Image(field, H, flux=0.4))
    data = DATA.with_model(truth, key=jax.random.PRNGKey(10))
    start = System(
        star=PointSource(),
        env=Image(
            GaussianField(onp.zeros((n, n)), 1.5, 2.0, mean=template),
            H,
            flux=0.4,
        ),
    )
    result = fit(start, image_priors(start), data)
    path = "env.log_brightness.latent"
    _, jac = _residual_jacobian(result.model, data, path)
    expected = onp.linalg.inv(onp.eye(n * n) + jac.T @ jac)
    draws = 20_000
    out = laplace_samples(result, data, draws, jax.random.PRNGKey(11))
    latents = out["latents"]
    assert latents.shape == (draws, n, n)
    flat = latents.reshape(draws, -1)
    z_map = onp.asarray(result.model.get(path), dtype=float).ravel()
    # Monte Carlo error: sd/√n on the mean, about √(2/n) on a variance.
    sd = onp.sqrt(onp.diag(expected))
    assert onp.all(onp.abs(flat.mean(0) - z_map) < 5 * sd / onp.sqrt(draws))
    cov = onp.cov(flat, rowvar=False)
    onp.testing.assert_allclose(cov, expected, atol=5 * onp.sqrt(2 / draws))
    # The data constrain some directions well below the prior's unit width.
    assert onp.linalg.eigvalsh(expected).min() < 0.5
    images = laplace_samples(
        result, data, 3, jax.random.PRNGKey(12), npix=8, fov_mas=8.0
    )["images"]
    assert images.shape == (3, 8, 8)
    onp.testing.assert_allclose(images.sum(axis=(1, 2)), 1.0, rtol=1e-5)
    assert not onp.allclose(images[0], images[1])
    with pytest.raises(ValueError, match="both npix and fov_mas"):
        laplace_samples(result, data, 3, jax.random.PRNGKey(12), npix=8)


def _hat_gammas(jac, rows, beta):
    """γ_b from the explicit hat matrix, B½ J (I + Jᵀ B J)⁻¹ Jᵀ B½."""
    per_row = onp.zeros(jac.shape[0])
    for kind, b in beta.items():
        per_row[rows[kind]] = b
    a = onp.eye(jac.shape[1]) + jac.T @ (per_row[:, None] * jac)
    hat = per_row * onp.einsum("ij,jk,ik->i", jac, onp.linalg.inv(a), jac)
    return {kind: hat[rows[kind]].sum() for kind in beta}


@pytest.mark.parametrize("shape", [(30, 50), (60, 12)])
def test_block_precisions_solve_the_coupled_fixed_point(shape):
    # Both forms (fewer data than latents, and more) satisfy
    # 1/β_b = χ²_b / (N_b − γ_b), with γ_b from the hat matrix's diagonal.
    from virgil.imaging import _block_precisions

    rng = onp.random.default_rng(0)
    n_rows, n_latent = shape
    jac = rng.normal(size=shape) * onp.geomspace(10.0, 0.01, n_latent)
    r = rng.normal(size=n_rows) * onp.where(onp.arange(n_rows) < 20, 3, 0.5)
    rows = {"vis": onp.arange(20), "phi": onp.arange(20, n_rows)}
    # Penalty-like rows: the phase block has fewer data than rows.
    counts = {"vis": 20, "phi": n_rows - 25}
    beta = _block_precisions(r, jac, rows, counts)
    gamma = _hat_gammas(jac, rows, beta)
    for kind in rows:
        chi2 = r[rows[kind]] @ r[rows[kind]]
        assert 1.0 / beta[kind] == pytest.approx(
            chi2 / (counts[kind] - gamma[kind]), rel=1e-8
        )


def test_block_error_scale_with_one_block_matches_the_scalar():
    # One block holding every residual is the single-β problem, which
    # error_scale solves by Newton's method.
    from virgil.imaging import _block_precisions, _residual_jacobian

    latent = jax.random.normal(jax.random.PRNGKey(6), (N, N))
    noisy = DATA.with_model(
        _gp_scene(latent, 1.5), key=jax.random.PRNGKey(7), noise_scale=0.5
    )
    start = _gp_scene(onp.zeros((N, N)), 1.5)
    model = fit(start, image_priors(start), noisy).model
    scale = error_scale(model, noisy)
    r, jac = _residual_jacobian(model, noisy, "env.log_brightness.latent")
    rows = {"all": onp.arange(r.size)}
    counts = {"all": noisy.n_independent}
    # The n x n form (39 rows, 256 latents) against error_scale, and the
    # latent form (20 of the columns) against the single-β equation.
    beta = _block_precisions(r, jac, rows, counts)["all"]
    assert 1.0 / onp.sqrt(beta) == pytest.approx(scale, rel=1e-8)
    lam = onp.linalg.svd(jac[:, :20], compute_uv=False) ** 2
    beta = _block_precisions(r, jac[:, :20], rows, counts)["all"]
    gamma = onp.sum(beta * lam / (1.0 + beta * lam))
    assert 1.0 / beta == pytest.approx(
        (r @ r) / (counts["all"] - gamma), rel=1e-8
    )


def test_block_error_scales_recover_different_miscalibrations():
    # V² quoted 3 times too small and closure phases 2 times too large:
    # the block scales recover 3 and 1/2, and one scale lies between.
    rich = vlti_oidata(
        hour_angles_h=(-3.0, -1.5, 0.0, 1.5, 3.0),
        wavelengths_m=[3.2e-6, 3.5e-6, 3.8e-6],
    )
    latent = jax.random.normal(jax.random.PRNGKey(4), (N, N))
    truth = _gp_scene(latent, 1.5)
    noisy = rich.with_model(truth, key=jax.random.PRNGKey(5))
    quoted = noisy.with_error_scale({"vis": 1.0 / 3.0, "phi": 2.0})
    start = _gp_scene(onp.zeros((N, N)), 1.5)
    model = fit(start, image_priors(start), quoted).model
    scales = error_scale(model, quoted, by_observable=True)
    assert set(scales) == {"vis", "phi"}
    # Compare with the scatter this noise draw actually has about the
    # truth: 90 V² scatter by ~8%, and the draw (which depends on the
    # dtype under JAX_ENABLE_X64) can sit 20% below 1.
    m_vis = onp.asarray(noisy.model(truth))[: noisy.vis.size]
    realised = onp.sqrt(
        onp.mean(((onp.asarray(noisy.vis) - m_vis) / noisy.d_vis) ** 2)
    )
    assert scales["vis"] == pytest.approx(3.0 * realised, rel=0.12)
    # 45 independent closure phases: s_phi scatters by ~13%.
    assert 0.35 < scales["phi"] < 0.7
    single = error_scale(model, quoted)
    assert scales["phi"] < single < scales["vis"]


def test_with_error_scale_takes_a_scale_per_observable():
    scaled = DATA.with_error_scale({"phi": 2.0})
    onp.testing.assert_allclose(scaled.d_vis, DATA.d_vis)
    onp.testing.assert_allclose(scaled.d_phi, 2.0 * DATA.d_phi)
    both = DATA.with_error_scale({"vis": 3.0, "phi": 3.0})
    onp.testing.assert_allclose(both.d_vis, DATA.with_error_scale(3.0).d_vis)
    onp.testing.assert_allclose(both.d_phi, DATA.with_error_scale(3.0).d_phi)
    # A kind these data lack (no OI_FLUX) is ignored; a non-kind is not.
    DATA.with_error_scale({"flux": 2.0})
    with pytest.raises(ValueError, match="Unknown observables"):
        DATA.with_error_scale({"v2": 2.0})
    with pytest.raises(ValueError, match="factor"):
        DATA.with_error_scale({"vis": -1.0})


def test_block_error_scales_refuse_unscaled_nuisance_covariance():
    # Gains and closure offsets add covariance (D + UΛUᵀ) that scaling the
    # quoted errors leaves alone, so the block fixed point would not be the
    # evidence optimum: refuse rather than return a wrong scale.
    start = _gp_scene(onp.zeros((N, N)), 1.5)
    for data, what in (
        (DATA.with_gains(telescope=0.01), "calibration gains"),
        (DATA.with_closure_offsets(triangle=0.01), "closure-phase offsets"),
    ):
        with pytest.raises(ValueError, match=what):
            error_scale(start, data, by_observable=True)


@pytest.mark.parametrize("n_telescopes", [3, 4])
def test_observable_blocks_map_residual_rows(n_telescopes):
    # The rows labelled "phi" are exactly those that change when only the
    # phase errors change: for four telescopes, the whitened independent
    # combinations and the penalty rows; N_phi counts only the former.
    from virgil.coverage import VLTI_UTS
    from virgil.imaging import _observable_blocks
    from virgil.likelihood import whitened_residuals

    data = vlti_oidata(
        stations=VLTI_UTS[:n_telescopes],
        hour_angles_h=(-2.0, 0.0, 2.0),
        wavelengths_m=[3.5e-6],
    )
    scene = System(star=PointSource(), env=GaussianDisk(3.0, flux=0.4))
    data = data.with_model(scene, key=jax.random.PRNGKey(0))
    other = System(star=PointSource(), env=GaussianDisk(3.0, dra=1.0))
    rows, counts = _observable_blocks([data, data])
    n_vis, n_phi = data.vis.size, data.phi.size
    n_comb = n_phi if data.cp_noise is None else data.cp_noise.size
    assert (data.cp_noise is None) == (n_telescopes == 3)
    assert counts == {"vis": 2 * n_vis, "phi": 2 * n_comb}
    assert sum(counts.values()) == 2 * data.n_independent
    assert sum(v.size for v in rows.values()) == 2 * data.n_residuals
    base = onp.concatenate([whitened_residuals(other, data)] * 2)
    halved = onp.concatenate(
        [whitened_residuals(other, data.with_error_scale({"phi": 2.0}))] * 2
    )
    onp.testing.assert_allclose(halved[rows["vis"]], base[rows["vis"]])
    onp.testing.assert_allclose(
        halved[rows["phi"]], 0.5 * base[rows["phi"]], rtol=1e-5
    )
