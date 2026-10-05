import jax
import jax.numpy as np
import numpy as onp
import numpyro.distributions as dist
import pytest

from virgil.coverage import (
    PARANAL_LATITUDE_DEG,
    VLTI_UTS,
    nrm_oidata,
    vlti_oidata,
)
from virgil.fitting import fit
from virgil.imaging import (
    Centroid,
    MaxEntropy,
    image_priors,
    starting_image,
)
from virgil.models import (
    BinaryModelCartesian,
    GaussianDisk,
    Image,
    PointSource,
    Rotated,
    System,
    circular_support,
)
from virgil.scenes import gaussian_blob
from virgil.spectra import PowerLaw

NIGHT = vlti_oidata(
    hour_angles_h=(-2.0, 0.0, 2.0), wavelengths_m=[3.2e-6, 3.8e-6]
)


def test_vlti_coverage_has_all_baselines_and_triangles():
    data = vlti_oidata()
    assert data.vis.size == 6 * 5 * 11 and data.phi.size == 4 * 5 * 11
    longest = max(
        onp.hypot(*(VLTI_UTS[j] - VLTI_UTS[i]))
        for i in range(4)
        for j in range(i + 1, 4)
    )
    # Projection can only shorten a ground baseline.
    assert np.max(np.hypot(data.u, data.v)) <= longest + 1e-6
    model = data.model(PointSource(dra=5.0, ddec=-3.0))
    assert np.allclose(model[: data.vis.size], 1.0)
    assert np.allclose(model[data.vis.size :], 0.0, atol=1e-5)


def test_a_baseline_at_transit_through_the_zenith_projects_to_east_north():
    # At transit (hour angle 0) of a source at the zenith (declination =
    # latitude), the projected baseline is the ground baseline (East, North).
    stations = [[0.0, 0.0], [10.0, 0.0], [0.0, 20.0]]
    data = vlti_oidata(
        stations,
        declination_deg=PARANAL_LATITUDE_DEG,
        hour_angles_h=(0.0,),
        wavelengths_m=[3.5e-6],
    )
    assert np.allclose(
        np.ravel(data.u), np.array([10.0, 0.0, -10.0]), atol=1e-5
    )
    assert np.allclose(
        np.ravel(data.v), np.array([0.0, 20.0, 20.0]), atol=1e-5
    )


def test_circular_support_can_leave_a_hole_under_the_star():
    support = circular_support(21, 1.0, radius_mas=9.0, inner_radius_mas=2.0)
    assert not support[10, 10] and support[10, 15] and not support[0, 0]


@pytest.mark.slow
def test_a_joint_fit_of_two_nights_recovers_a_binary():
    truth = BinaryModelCartesian(6.0, -4.0, 0.05)
    nights = [
        NIGHT.with_model(truth, key=jax.random.PRNGKey(0)),
        vlti_oidata(
            hour_angles_h=(-3.0, 1.0), wavelengths_m=[3.2e-6, 3.8e-6]
        ).with_model(truth, key=jax.random.PRNGKey(1)),
    ]
    priors = {
        "dra": dist.Uniform(-20.0, 20.0),
        "ddec": dist.Uniform(-20.0, 20.0),
        "flux": dist.Uniform(0.0, 0.5),
    }
    result = fit(BinaryModelCartesian(5.0, -3.0, 0.03), priors, nights)
    assert len(result.info["chi2"]) == 2
    assert abs(result.values["dra"] - 6.0) < 0.3
    assert abs(result.values["ddec"] + 4.0) < 0.3
    assert abs(result.values["flux"] - 0.05) < 0.01


def test_a_star_anchors_an_image_fitted_to_v2_and_closure_phases():
    npix, scale = 32, 0.6
    truth = System(
        star=PointSource(),
        env=Image.from_brightness(
            gaussian_blob(npix, scale, 1.5, dra=4.0, ddec=2.0), scale, flux=0.3
        ),
    )
    data = NIGHT.with_model(truth, key=jax.random.PRNGKey(2))
    start = System(
        star=PointSource(),
        env=Image.from_model(GaussianDisk(4.0), npix, scale, flux=0.3),
    )
    result = fit(
        start, image_priors(start), data, [MaxEntropy(10.0, path="env")]
    )
    centroid = Centroid(1.0, path="env").centroid(result.model)
    assert np.allclose(centroid, np.array([4.0, 2.0]), atol=1.0)


def test_a_support_hole_keeps_flux_off_the_star():
    npix, scale = 24, 0.8
    hole = circular_support(npix, scale, radius_mas=9.0, inner_radius_mas=2.0)
    start = System(
        star=PointSource(),
        env=Image.from_model(
            GaussianDisk(4.0), npix, scale, flux=0.3, support=hole
        ),
    )
    truth = System(star=PointSource(), env=GaussianDisk(4.0, flux=0.3))
    data = NIGHT.with_model(truth, key=jax.random.PRNGKey(3))
    result = fit(
        start, image_priors(start), data, [MaxEntropy(10.0, path="env")]
    )
    assert np.all(result.model.env.brightness[~hole] == 0.0)


def test_a_sparco_spectral_index_is_recovered_from_the_channels():
    # A grey image whose flux relative to the star rises with wavelength.
    npix, scale = 24, 0.8
    image = Image.from_model(
        GaussianDisk(3.0), npix, scale, flux=PowerLaw(0.3, 2.0, 3.5e-6)
    )
    truth = System(star=PointSource(), env=image)
    data = vlti_oidata().with_model(truth, key=jax.random.PRNGKey(4))
    start = truth.set(["env.flux.ratio", "env.flux.index"], [0.2, 0.0])
    priors = {
        "env.flux.ratio": dist.Uniform(0.0, 2.0),
        "env.flux.index": dist.Uniform(-5.0, 5.0),
    }
    result = fit(start, priors, data)
    assert abs(result.values["env.flux.index"] - 2.0) < 0.3
    assert abs(result.values["env.flux.ratio"] - 0.3) < 0.03


def test_nrm_rolls_combine_into_one_fit():
    truth = BinaryModelCartesian(120.0, 80.0, 0.05)
    rolls = [
        nrm_oidata(rotation_deg=angle).with_model(
            truth, key=jax.random.PRNGKey(k)
        )
        for k, angle in enumerate((0.0, 30.0))
    ]
    priors = {
        "dra": dist.Uniform(-400.0, 400.0),
        "ddec": dist.Uniform(-400.0, 400.0),
        "flux": dist.Uniform(0.0, 0.5),
    }
    result = fit(BinaryModelCartesian(100.0, 60.0, 0.03), priors, rolls)
    assert abs(result.values["dra"] - 120.0) < 10.0
    assert abs(result.values["ddec"] - 80.0) < 10.0


def test_starting_image_can_leave_a_hole_under_the_star():
    start = starting_image(NIGHT, hole_mas=1.0)
    image = start.env
    centre = image.log_brightness.shape[0] // 2
    assert image.support is not None and not image.support[centre, centre]
    assert np.all(image.brightness[~image.support] == 0.0)


def test_log_uniform_envelope_fit_stays_off_its_lower_bounds():
    """The envelope fit behind starting_image, with LogUniform priors on
    a point source's data: the envelope is unconstrained below the beam.

    Fitted in log sigma and log flux, where the priors are flat, nothing
    pulls the envelope towards the priors' lower bounds. (In the natural
    coordinates, -log p = log x did, and sigma ran to ~1e-3 of the beam
    and the flux to 1e-4.)
    """
    resolution = 7.25  # beam(NIGHT).major_mas
    low_sigma, low_flux = 1e-3 * resolution, 1e-4
    model = System(star=PointSource(), env=GaussianDisk(0.77, flux=0.1))
    priors = {
        "env.sigma": dist.LogUniform(low_sigma, 20.0),
        "env.flux": dist.LogUniform(low_flux, 100.0),
    }
    result = fit(model, priors, NIGHT)
    assert result.info["method"] == "lm"
    assert result.info["converged"] is True
    assert float(result.values["env.sigma"]) > 10.0 * low_sigma
    assert float(result.values["env.flux"]) > 10.0 * low_flux


def _binary_epochs(dra, ddec, flux, spin):
    epoch = BinaryModelCartesian(dra, ddec, flux)
    return [epoch, Rotated(epoch, spin)]


def test_a_model_per_epoch_recovers_a_known_and_an_unknown_rotation():
    truth = _binary_epochs(6.0, -4.0, 0.05, 60.0)
    nights = [
        NIGHT.with_model(m, key=jax.random.PRNGKey(k))
        for k, m in enumerate(truth)
    ]
    priors = {
        "dra": dist.Uniform(-20.0, 20.0),
        "ddec": dist.Uniform(-20.0, 20.0),
        "flux": dist.Uniform(0.0, 0.5),
    }
    init = {"dra": 5.0, "ddec": -3.0, "flux": 0.03}
    known = fit(
        lambda **p: _binary_epochs(**p, spin=60.0), priors, nights, init=init
    )
    assert isinstance(known.model[1], Rotated)
    assert abs(known.values["dra"] - 6.0) < 0.3
    unknown = fit(
        _binary_epochs,
        priors | {"spin": dist.Uniform(-180.0, 180.0)},
        nights,
        init=init | {"spin": 50.0},
    )
    assert abs(unknown.values["spin"] - 60.0) < 2.0
    assert abs(unknown.values["ddec"] + 4.0) < 0.3


def test_a_model_per_dataset_must_match_the_datasets():
    with pytest.raises(ValueError, match="2 models for 1 datasets"):
        fit(
            lambda dra: _binary_epochs(dra, -4.0, 0.05, 60.0),
            {"dra": dist.Uniform(-20.0, 20.0)},
            NIGHT,
            init={"dra": 5.0},
        )


def test_an_image_support_may_arrive_as_floats():
    # zodiax >= 0.5 returns leaves as floats from get(); the support must
    # still act as a mask.
    support = circular_support(16, 1.0, 8.0, inner_radius_mas=2.0)
    image = Image.from_model(GaussianDisk(3.0), 16, 1.0, support=support)
    as_floats = image.set("support", support.astype(float))
    assert np.allclose(as_floats.brightness, image.brightness)
    assert np.all(as_floats.brightness[~support] == 0.0)


def test_imaging_helpers_see_images_inside_rotated_scenes():
    from virgil.imaging import diagnose

    scene = System(
        star=PointSource(),
        env=Image.from_model(GaussianDisk(3.0), 16, 1.0, flux=0.3),
    )
    rotated = Rotated(scene, 30.0)
    assert list(image_priors(rotated)) == ["source.env.log_brightness"]
    report = diagnose(rotated, NIGHT.with_model(rotated))
    assert report.checks["pixel_scale_mas"] and "flip_dchi2" in report.checks


def test_bad_weights_and_pixel_scales_are_rejected():
    from virgil.imaging import l_curve

    scene = System(
        star=PointSource(),
        env=Image.from_model(GaussianDisk(3.0), 8, 1.0, flux=0.3),
    )
    for weights in ([], [0.0, 1.0], [-1.0], [float("nan")]):
        with pytest.raises(ValueError, match="weights"):
            l_curve(
                scene,
                image_priors(scene),
                NIGHT,
                MaxEntropy(1.0, path="env"),
                weights,
            )
    for scale in (0.0, -1.0, float("inf")):
        with pytest.raises(ValueError, match="pixel_scale_mas"):
            Image(np.zeros((4, 4)), scale)
