import jax
import jax.numpy as np
import numpy as onp
import pytest

from virgil.coverage import ami_grid_record, vlti_oidata
from virgil.fitting import fit
from virgil.imaging import (
    CleanResult,
    Laplacian,
    LogSum,
    StarletL1,
    _CleanScene,
    beam,
    clean,
    image_priors,
    l_curve,
    starlet,
)
from virgil.models import (
    BinaryModelCartesian,
    GaussianDisk,
    Image,
    PointSource,
    System,
    circular_support,
)
from virgil.oidata import OIData
from virgil.scenes import gaussian_blob

NPIX, SCALE = 16, 12.0
DATA = OIData(ami_grid_record(pitch_m=0.5))


def _image(brightness, **kwargs):
    return Image.from_brightness(brightness, SCALE, **kwargs)


def test_starlet_planes_sum_to_the_image():
    image = jax.random.uniform(jax.random.PRNGKey(0), (NPIX, NPIX + 3))
    details, coarse = starlet(image, scales=3)
    assert details.shape == (3, NPIX, NPIX + 3)
    assert np.allclose(details.sum(0) + coarse, image, atol=1e-6)
    # The B3 spline keeps a flat image flat away from the edges, so the
    # finest details of a constant vanish in the interior.
    details, _ = starlet(np.ones((NPIX, NPIX)), scales=1)
    assert np.allclose(details[0, 2:-2, 2:-2], 0.0, atol=1e-6)
    with pytest.raises(ValueError, match="positive integer"):
        starlet(image, scales=0)


def test_starlet_smoothing_is_the_b3_spline():
    delta = np.zeros((NPIX, NPIX)).at[8, 8].set(1.0)
    _, coarse = starlet(delta, scales=1)
    h = onp.array([1, 4, 6, 4, 1]) / 16
    expected = onp.zeros((NPIX, NPIX))
    expected[6:11, 6:11] = onp.outer(h, h)
    assert onp.allclose(coarse, expected, atol=1e-7)


def test_laplacian_matches_the_five_point_stencil():
    b = jax.random.uniform(jax.random.PRNGKey(1), (5, 6))
    image = _image(b)
    padded = onp.pad(onp.asarray(image.brightness), 2)
    lap = (
        padded[:-2, 1:-1]
        + padded[2:, 1:-1]
        + padded[1:-1, :-2]
        + padded[1:-1, 2:]
        - 4 * padded[1:-1, 1:-1]
    )
    reg = Laplacian(3.0)
    assert np.isclose(reg.value(image), 3.0 * onp.sum(lap**2), rtol=1e-5)
    assert np.isclose(
        0.5 * np.sum(reg.residuals(image) ** 2), reg.value(image)
    )


def test_starlet_l1_matches_its_definition_and_penalises_noise():
    blob = _image(gaussian_blob(NPIX, SCALE, 20.0))
    reg = StarletL1(2.0, scales=3, epsilon=1e-2)
    details, _ = starlet(blob.brightness, 3)
    eps = 1e-2 / NPIX**2
    expected = 2.0 * np.sum(np.sqrt(details**2 + eps**2))
    assert np.isclose(reg.value(blob), expected, rtol=1e-5)
    # Zero-mean noise, on a pedestal that keeps the pixels positive, adds
    # fine detail without changing the flux.
    b = gaussian_blob(NPIX, SCALE, 20.0)
    b = b + 0.2 * b.max()
    noise = jax.random.uniform(jax.random.PRNGKey(2), (NPIX, NPIX)) - 0.5
    noise = noise - noise.mean()
    smooth, noisy = _image(b), _image(b + 0.1 * b.max() * noise)
    assert reg.value(noisy) > reg.value(smooth)
    with pytest.raises(ValueError, match="positive integer"):
        StarletL1(1.0, scales=1.5)


def test_log_sum_counts_bright_pixels():
    flat = _image(np.ones((NPIX, NPIX)))
    point = _image(np.zeros((NPIX, NPIX)).at[8, 8].set(1.0), floor=1e-12)
    reg = LogSum(1.0, epsilon=1e-2)
    # A flat image has every pixel at the mean: N log(1 + 1/ε).
    assert np.isclose(reg.value(flat), NPIX**2 * np.log1p(100.0), rtol=1e-5)
    # One bright pixel costs log(1 + N/ε); the dark ones almost nothing.
    assert np.isclose(reg.value(point), np.log1p(NPIX**2 / 1e-2), rtol=1e-3)
    # The mean is over the support only.
    support = np.zeros((NPIX, NPIX), bool).at[:4, :4].set(True)
    inside = _image(np.ones((NPIX, NPIX)), support=support)
    assert np.isclose(reg.value(inside), 16 * np.log1p(100.0), rtol=1e-5)


@pytest.mark.filterwarnings("ignore:fit\\(method=:RuntimeWarning")
@pytest.mark.parametrize(
    "regulariser, method",
    [
        (Laplacian(1e2), "lm"),
        (StarletL1(1e-1), "lbfgs"),
        (LogSum(1e-3), "lbfgs"),
    ],
)
def test_sparsity_regularisers_fit(regulariser, method):
    truth = _image(gaussian_blob(NPIX, SCALE, 25.0))
    data = DATA.with_model(truth, key=jax.random.PRNGKey(4))
    start = _image(np.ones((NPIX, NPIX)))
    result = fit(
        start,
        image_priors(start),
        data,
        [regulariser],
        method=method,
        max_steps=20,
    )
    assert result.info["method"] == method
    assert onp.isfinite(result.info["loss"])
    assert np.all(np.isfinite(result.model.brightness))


def test_clean_scene_is_a_system_with_an_image():
    fluxes = np.zeros((NPIX, NPIX)).at[3, 4].set(0.02).at[10, 9].set(0.01)
    on = fluxes > 0
    image = Image(
        np.log(np.where(on, fluxes, 1.0)), SCALE, support=on, flux=0.03
    )
    system = System(base=PointSource(), clean=image)
    scene = _CleanScene(PointSource(), fluxes, SCALE, 0.0)
    # On the AMI uv lattice (the matrix Fourier transform) and off it.
    assert np.allclose(DATA.model(scene), DATA.model(system), atol=1e-5)
    u, v, wavel = DATA.u, DATA.v, DATA.wavel
    assert np.allclose(
        scene.model(u, v, wavel), system.model(u, v, wavel), atol=1e-5
    )


def test_clean_finds_a_companion_with_disco_phases():
    # A 5% companion on a pixel centre, East and South of the star.
    truth = System(
        star=PointSource(), comp=PointSource(dra=30.0, ddec=-18.0, flux=0.05)
    )
    data = DATA.with_model(truth, key=jax.random.PRNGKey(5))
    result = clean(data, NPIX, SCALE, base=PointSource(), max_iterations=300)
    assert isinstance(result, CleanResult)
    assert result.stop == "target"
    assert result.chi2_red[-1] <= 1.0 < result.chi2_red[0]
    # Column 0 is East (dra = +90 mas), row 0 North (ddec = +90 mas).
    row, col = onp.unravel_index(
        int(np.argmax(result.components)), (NPIX, NPIX)
    )
    assert (row, col) == (9, 5)
    assert abs(float(result.components.sum()) - 0.05) < 0.02
    assert isinstance(result.model, System)
    assert np.isclose(result.model.clean.flux, result.components.sum())
    restored = result.restored(beam(data))
    assert restored.shape == (NPIX, NPIX)


def test_clean_without_a_base_starts_at_the_centre():
    truth = Image.from_brightness(
        gaussian_blob(NPIX, SCALE, 15.0, dra=20.0), SCALE
    )
    data = DATA.with_model(truth, key=jax.random.PRNGKey(6))
    result = clean(data, NPIX, SCALE, max_iterations=20)
    assert isinstance(result.model, Image)
    assert np.isclose(result.components.sum(), 1.0)
    assert result.chi2_red[-1] < result.chi2_red[0]


def test_clean_respects_the_support_and_checks_its_inputs():
    truth = System(
        star=PointSource(), comp=PointSource(dra=30.0, ddec=-18.0, flux=0.05)
    )
    data = DATA.with_model(truth, key=jax.random.PRNGKey(5))
    support = np.ones((NPIX, NPIX), bool).at[9, 5].set(False)
    result = clean(
        data,
        NPIX,
        SCALE,
        base=PointSource(),
        support=support,
        max_iterations=10,
    )
    assert float(result.components[9, 5]) == 0.0
    with pytest.raises(ValueError, match="outside the support"):
        clean(
            data,
            NPIX,
            SCALE,
            base=PointSource(),
            init=np.zeros((NPIX, NPIX)).at[9, 5].set(0.01),
            support=support,
        )
    with pytest.raises(ValueError, match="max_iterations"):
        clean(data, NPIX, SCALE, base=PointSource(), max_iterations=2.5)
    # Without a base, the starting component avoids a hole at the centre.
    hole = np.ones((NPIX, NPIX), bool).at[6:10, 6:10].set(False)
    centred = clean(data, NPIX, SCALE, support=hole, max_iterations=0)
    assert float(np.sum(np.where(hole, 0.0, centred.components))) == 0.0
    assert len(centred.chi2_red) == 1
    with pytest.raises(ValueError, match="gain"):
        clean(data, NPIX, SCALE, base=PointSource(), gain=0.0)
    with pytest.raises(ValueError, match="init"):
        clean(
            data, NPIX, SCALE, base=PointSource(), init=-np.ones((NPIX, NPIX))
        )
    with pytest.raises(ValueError, match="support"):
        clean(
            data,
            NPIX,
            SCALE,
            base=PointSource(),
            support=np.ones((3, 3), bool),
        )


VLTI_UTS = onp.array(
    [
        [-9.925, -20.335],
        [14.887, 30.502],
        [44.915, 66.183],
        [103.306, 43.999],
    ]
)


@pytest.mark.parametrize("npix", [33, 34, 40, 46, 52, 58, 64, 68])
def test_clean_without_base_is_finite_with_correlated_closures(npix):
    # Four telescopes give correlated closure phases. Without a base scene
    # the seed pixel cannot change the image: its |J e_p| is zero, but on an
    # even grid its phases are not exactly 1 and rounding left |J e_p|² ~
    # 1e-24, a score of ~1e24, and an infinite first step (NaN χ²; F12).
    with jax.enable_x64(True):  # as in a script that turns x64 on
        data = vlti_oidata(
            VLTI_UTS,
            declination_deg=-30.0,
            hour_angles_h=onp.linspace(-3, 3, 7),
            wavelengths_m=onp.array([1.65e-6]),
            sigma_v2=0.01,
            sigma_cp_deg=1.0,
        ).with_model(BinaryModelCartesian(4.0, 1.0, 0.3))
        assert data.cp_noise is not None
        result = clean(data, npix, 0.4, max_iterations=3)
        assert onp.all(onp.isfinite(result.chi2_red))
        # Three iterations, the limit, then the final major cycle.
        assert len(result.chi2_red) == 5
        assert onp.all(onp.isfinite(result.components))


def _companion_data():
    truth = System(
        star=PointSource(), comp=PointSource(dra=30.0, ddec=-18.0, flux=0.05)
    )
    return DATA.with_model(truth, key=jax.random.PRNGKey(5))


def test_clean_stops_when_chi2_stalls():
    # A target below what the noise allows can never be reached: CLEAN
    # notices that χ² has stopped falling instead of running to the limit.
    result = clean(
        _companion_data(),
        NPIX,
        SCALE,
        base=PointSource(),
        target_chi2_red=0.5,
        max_iterations=5000,
    )
    assert result.stop == "stalled"
    assert len(result.chi2_red) - 1 < 5000
    assert result.chi2_red[-1] > 0.5


def test_major_cycles_remove_a_wrong_component():
    # Flux at an empty pixel, and a little on the companion: refitting the
    # components moves the flux to the companion and removes the other.
    init = np.zeros((NPIX, NPIX)).at[2, 2].set(0.05).at[9, 5].set(0.01)
    result = clean(
        _companion_data(),
        NPIX,
        SCALE,
        base=PointSource(),
        init=init,
        refit_every=1,
        max_iterations=6,
    )
    assert float(result.components[2, 2]) == 0.0
    # Most of the companion's 0.05 is on it after these few iterations.
    assert float(result.components[9, 5]) > 0.03
    with pytest.raises(ValueError, match="refit_every"):
        clean(
            _companion_data(), NPIX, SCALE, base=PointSource(), refit_every=-1
        )


@pytest.mark.filterwarnings("ignore:fit\\(method=:RuntimeWarning")
def test_l_curve_without_warm_starts_fits_each_weight_afresh():
    truth = _image(gaussian_blob(NPIX, SCALE, 25.0))
    data = DATA.with_model(truth, key=jax.random.PRNGKey(4))
    start = _image(np.ones((NPIX, NPIX)))
    priors = image_priors(start)
    options = dict(method="lbfgs", max_steps=10)
    curve = l_curve(
        start,
        priors,
        data,
        LogSum(1.0),
        [1e-2, 1e-3],
        warm_start=False,
        **options,
    )
    alone = fit(start, priors, data, [LogSum(1e-3)], **options)
    assert np.allclose(
        curve.results[1].model.brightness, alone.model.brightness, atol=1e-6
    )


def test_clean_with_a_spectrum_is_sparco():
    # A star and a companion with their own spectra, and an environment
    # with another: components with the environment's spectrum, beside the
    # base's components, reproduce the SPARCO model.
    from virgil.coverage import vlti_oidata
    from virgil.spectra import PowerLaw

    template = vlti_oidata(
        hour_angles_h=(-2.0, 0.0, 2.0),
        wavelengths_m=onp.linspace(1.5e-6, 1.8e-6, 4),
    )
    base = System(
        star=PointSource(flux=PowerLaw(1.0, -3.0, 1.65e-6)),
        comp=PointSource(dra=1.5, ddec=-1.0, flux=0.02),
    )
    scale = 0.6
    fluxes = np.zeros((NPIX, NPIX)).at[3, 4].set(0.02).at[10, 9].set(0.03)
    shape = PowerLaw(1.0, 2.0, 1.65e-6)
    scene = _CleanScene(base, fluxes, scale, 0.0, shape)
    on = fluxes > 0
    flat = System(
        **base.components,
        clean=Image(
            np.log(np.where(on, fluxes, 1.0)),
            scale,
            support=on,
            flux=PowerLaw(0.05, 2.0, 1.65e-6),
        ),
    )
    assert np.allclose(template.model(scene), template.model(flat), atol=1e-6)

    data = template.with_model(flat, key=jax.random.PRNGKey(8))
    sparco = clean(data, NPIX, scale, base=base, spectrum=shape)
    grey = clean(data, NPIX, scale, base=base)
    assert sparco.stop == "target"
    assert set(sparco.model.components) == {"star", "comp", "clean"}
    assert np.isclose(sparco.model.clean.flux.index, 2.0)
    assert np.isclose(sparco.model.clean.flux.ratio, sparco.components.sum())
    assert len(sparco.chi2_red) < len(grey.chi2_red)
    with pytest.raises(ValueError, match="needs a base"):
        clean(data, NPIX, scale, spectrum=shape)


def test_clean_ends_with_a_major_cycle():
    # Major cycles too far apart to run in the loop: CLEAN still refits the
    # fluxes once before it returns, so they are closer to the companion's.
    data = _companion_data()
    none = clean(data, NPIX, SCALE, base=PointSource(), refit_every=0)
    last = clean(data, NPIX, SCALE, base=PointSource(), refit_every=1000)
    assert none.stop == last.stop == "target"
    assert len(last.chi2_red) == len(none.chi2_red) + 1
    assert last.chi2_red[-1] < none.chi2_red[-1]
    assert abs(float(last.components.sum()) - 0.05) < abs(
        float(none.components.sum()) - 0.05
    )


def test_clean_fits_the_base():
    # A companion in the base, 85 mas out and started 6 mas off, and a knot
    # for CLEAN: each major cycle fits the companion's position and flux
    # together with the components' fluxes.
    import numpyro.distributions as dist

    truth = System(
        star=PointSource(),
        comp=PointSource(dra=66.0, ddec=-54.0, flux=0.05),
        knot=PointSource(dra=-42.0, ddec=30.0, flux=0.02),
    )
    data = DATA.with_model(truth, key=jax.random.PRNGKey(9))
    base = System(
        star=PointSource(), comp=PointSource(dra=60.0, ddec=-48.0, flux=0.04)
    )
    # Keep CLEAN's components away from the companion, which the base models.
    offsets = (onp.arange(NPIX) - (NPIX - 1) / 2) * SCALE
    away = onp.hypot(-offsets[None, :] - 63.0, -offsets[:, None] + 51.0) > 25.0
    priors = {
        "comp.dra": dist.Normal(60.0, 20.0),
        "comp.ddec": dist.Normal(-48.0, 20.0),
        "comp.flux": dist.Uniform(0.0, 0.2),
    }
    result = clean(
        data,
        NPIX,
        SCALE,
        base=base,
        base_priors=priors,
        support=away,
        refit_every=10,
    )
    comp = result.model.comp
    assert abs(float(comp.dra) - 66.0) < 3.0
    assert abs(float(comp.ddec) + 54.0) < 3.0
    row, col = onp.unravel_index(
        int(np.argmax(result.components)), (NPIX, NPIX)
    )
    assert (-offsets[col], -offsets[row]) == (-42.0, 30.0)
    with pytest.raises(ValueError, match="base_priors"):
        clean(data, NPIX, SCALE, base=base, base_priors=priors, refit_every=0)
    with pytest.raises(ValueError, match="base_priors"):
        clean(data, NPIX, SCALE, base_priors=priors)


def test_clean_survives_a_failed_major_cycle_solve(monkeypatch):
    # scipy's nnls can give up ("Maximum number of iterations reached") on
    # the ill-conditioned columns of high signal-to-noise data; the refit
    # then falls back to a bounded least-squares solve.
    import scipy.optimize

    def failing_nnls(*args, **kwargs):
        raise RuntimeError("Maximum number of iterations reached.")

    monkeypatch.setattr(scipy.optimize, "nnls", failing_nnls)
    truth = Image.from_brightness(
        gaussian_blob(NPIX, SCALE, 15.0, dra=20.0), SCALE
    )
    data = DATA.with_model(truth, key=jax.random.PRNGKey(6))
    result = clean(data, NPIX, SCALE, max_iterations=60, refit_every=20)
    assert onp.isfinite(result.chi2_red).all()
    assert result.chi2_red[-1] < result.chi2_red[0]


def test_base_fit_runs_in_cleans_precision_and_before_pruning(monkeypatch):
    # The joint base fit uses clean's dtype, not fit's float64 default, and
    # comes before the non-negative refit, so components are pruned
    # against the corrected base.
    import numpyro.distributions as dist

    import virgil.imaging as imaging

    calls, real_fit, real_refit = [], imaging.fit, imaging._refit

    def recording_fit(*args, **kwargs):
        calls.append(("fit", kwargs.get("dtype")))
        return real_fit(*args, **kwargs)

    def recording_refit(*args, **kwargs):
        calls.append(("nnls", None))
        return real_refit(*args, **kwargs)

    monkeypatch.setattr(imaging, "fit", recording_fit)
    monkeypatch.setattr(imaging, "_refit", recording_refit)
    base = System(
        star=PointSource(), comp=PointSource(dra=24.0, ddec=-12.0, flux=0.04)
    )
    priors = {"comp.flux": dist.Uniform(0.0, 0.2)}
    clean(
        _companion_data(),
        NPIX,
        SCALE,
        base=base,
        base_priors=priors,
        refit_every=2,
        max_iterations=4,
        dtype="float32",
    )
    assert calls and calls[0] == ("fit", "float32")
    assert all(d == "float32" for kind, d in calls if kind == "fit")
    assert calls[1] == ("nnls", None)


def test_base_priors_on_a_system_must_name_components():
    import numpyro.distributions as dist

    base = System(
        star=PointSource(), comp=PointSource(dra=24.0, ddec=-12.0, flux=0.04)
    )
    with pytest.raises(ValueError, match="components' parameters"):
        clean(
            _companion_data(),
            NPIX,
            SCALE,
            base=base,
            base_priors={"dra": dist.Normal(0.0, 5.0)},
        )


def test_point_scale_reproduces_point_clean():
    # With points only, multi-scale CLEAN is the same algorithm as before.
    # No golden values: the noise draw itself differs between platforms
    # and JAX versions (first chi2 27.1-29.3 on CI runners).
    data = _companion_data()
    default = clean(data, NPIX, SCALE, base=PointSource(), max_iterations=300)
    points = clean(
        data,
        NPIX,
        SCALE,
        base=PointSource(),
        max_iterations=300,
        scales_mas=(0.0,),
    )
    assert onp.array_equal(points.chi2_red, default.chi2_red)
    assert onp.array_equal(points.components, default.components)
    assert points.scales_mas == (0.0,)
    assert onp.array_equal(points.components_by_scale[0], points.components)


def _disk_and_point():
    # A star, a resolved Gaussian disk (FWHM 16 mas, ~3 beams) and a point,
    # in V² and closure phases from the four UTs.
    truth = System(
        star=PointSource(),
        disk=GaussianDisk(dra=-7.0, ddec=5.0, sigma=7.0, flux=0.5),
        comp=PointSource(dra=9.0, ddec=-9.0, flux=0.2),
    )
    data = vlti_oidata(wavelengths_m=onp.linspace(3.0e-6, 4.0e-6, 5))
    return data.with_model(truth, key=jax.random.PRNGKey(3), noise_scale=0.3)


def test_multiscale_clean_puts_extended_flux_in_broad_components():
    data = _disk_and_point()
    npix, scale = 20, 2.0
    support = circular_support(npix, scale, 20.0, 2.0)
    kwargs = dict(base=PointSource(), support=support, max_iterations=600)
    points = clean(data, npix, scale, **kwargs)
    multi = clean(data, npix, scale, scales_mas=(0.0, 16.0), **kwargs)
    assert points.stop == multi.stop == "target"
    assert multi.components_by_scale.shape == (2, npix, npix)
    n_points = int(np.sum(points.components_by_scale > 0))
    n_multi = int(np.sum(multi.components_by_scale > 0))
    assert n_multi < n_points / 2
    assert len(multi.chi2_red) < len(points.chi2_red)
    assert multi.chi2_red[-1] < points.chi2_red[-1]
    # The disk's 0.5 is mostly in the broad scale, and the point's 0.2 in
    # the point scale, at the point's pixel.
    by_scale = onp.asarray(multi.components_by_scale).sum(axis=(1, 2))
    assert by_scale[1] > 0.35
    assert abs(by_scale[0] - 0.2) < 0.05
    offsets = (onp.arange(npix) - (npix - 1) / 2) * scale
    row, col = onp.unravel_index(
        int(np.argmax(multi.components_by_scale[0])), (npix, npix)
    )
    assert (-offsets[col], -offsets[row]) == (9.0, -9.0)
    # The image is the sum of the scales' shapes, and keeps their flux.
    assert np.isclose(multi.components.sum(), by_scale.sum(), rtol=1e-5)
    assert np.isclose(multi.model.clean.flux, multi.components.sum())


def test_multiscale_clean_respects_support_and_fits_the_base():
    import numpyro.distributions as dist

    support = circular_support(NPIX, SCALE, 80.0, 15.0)
    base = System(
        star=PointSource(), comp=PointSource(dra=24.0, ddec=-12.0, flux=0.04)
    )
    result = clean(
        _companion_data(),
        NPIX,
        SCALE,
        base=base,
        base_priors={"comp.flux": dist.Uniform(0.0, 0.2)},
        support=support,
        refit_every=5,
        max_iterations=12,
        scales_mas=(0.0, 30.0),
        scale_bias=0.3,
    )
    by_scale = onp.asarray(result.components_by_scale)
    assert by_scale.shape == (2, NPIX, NPIX)
    assert onp.all(by_scale >= 0.0)
    assert onp.all(by_scale[:, ~support] == 0.0)
    assert onp.all(onp.asarray(result.components)[~support] == 0.0)
    assert onp.all(onp.asarray(result.components) >= 0.0)
    assert result.chi2_red[-1] < result.chi2_red[0]
    assert {"star", "comp", "clean"} == set(result.model.components)
    assert result.restored(beam(DATA)).shape == (NPIX, NPIX)
    # A point init needs a point scale; a per-scale init does not.
    init = np.zeros((NPIX, NPIX)).at[9, 5].set(0.01)
    with pytest.raises(ValueError, match="point components"):
        clean(
            DATA, NPIX, SCALE, base=PointSource(), init=init, scales_mas=30.0
        )
    started = clean(
        _companion_data(),
        NPIX,
        SCALE,
        base=PointSource(),
        init=np.stack([init, init]),
        scales_mas=(0.0, 30.0),
        max_iterations=0,
        refit_every=0,
    )
    assert np.isclose(started.components_by_scale.sum(), 0.02)
    for bad in [(), (0.0, 0.0), (-1.0,)]:
        with pytest.raises(ValueError, match="scales_mas"):
            clean(DATA, NPIX, SCALE, base=PointSource(), scales_mas=bad)
    with pytest.raises(ValueError, match="scale_bias"):
        clean(DATA, NPIX, SCALE, base=PointSource(), scale_bias=1.0)
