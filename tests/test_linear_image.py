"""LinearImage: linear and sqrt pixel images (design/bounded_pixels.md §2–3)."""

import jax
import jax.numpy as np
import numpy as onp
import pytest

from virgil.coverage import ami_grid_record
from virgil.fitting import fit
from virgil.imaging import TSV, TV, MaxEntropy, StarletL1, image_priors
from virgil.models import (
    GaussianDisk,
    Image,
    LinearImage,
    PointSource,
    System,
    circular_support,
)
from virgil.oidata import OIData

NPIX, SCALE, BEAM = 12, 10.0, 60.0
DATA = OIData(ami_grid_record(pitch_m=0.7))


@pytest.fixture(autouse=True)
def _x64():
    with jax.enable_x64(True):
        yield


def _blob(flux=0.5):
    img = GaussianDisk(sigma=25.0)._image(
        *np.meshgrid(
            (np.arange(NPIX) - 5.5) * SCALE, (np.arange(NPIX) - 5.5) * SCALE
        ),
        SCALE,
    )
    return flux * img / img.sum()


@pytest.mark.parametrize("par", ["linear", "sqrt"])
def test_flux_is_recovered_from_the_visibility_against_a_star(par):
    # Where the blob is resolved out, V = 1 / (1 + F) for a unit star.
    pix = _blob(0.5)
    x = pix if par == "linear" else np.sqrt(pix)
    im = LinearImage(x, SCALE, parametrisation=par)
    assert abs(float(im.total_flux) - 0.5) < 1e-12
    assert abs(float(im.brightness.sum()) - 1.0) < 1e-12
    scene = System(star=PointSource(), env=im)
    u, v = np.asarray([29.7]), np.asarray([0.0])
    vis = abs(complex(scene.model(u, v, 4.8e-6)[0]))
    assert 1.0 / vis - 1.0 == pytest.approx(0.5, rel=3e-2)


def test_all_zero_linear_image_is_the_star_alone_and_finite():
    # L-BFGS-B's line search can clip a trial point to all zeros; 0/0 there made
    # the loss and its gradient NaN, which ends the search with "ABNORMAL".
    u, v = np.asarray([29.7, 5.0]), np.asarray([0.0, 11.0])

    def vis(pixels):
        scene = System(
            star=PointSource(),
            env=LinearImage(pixels, SCALE, reference_flux=0.5),
        )
        return np.abs(scene.model(u, v, 4.8e-6))

    zero = np.zeros((NPIX, NPIX))
    assert onp.allclose(vis(zero), 1.0)
    jac = jax.jacobian(vis)(zero)
    assert onp.isfinite(jac).all()


def test_float_and_list_supports_still_work():
    from virgil.models import Image

    pix = onp.ones((4, 4))
    assert Image(np.zeros((4, 4)), 1.0, support=pix).support.dtype == bool
    assert Image(np.zeros((4, 4)), 1.0, support=pix.tolist()) is not None


def test_image_field_order_is_unchanged():
    import dataclasses

    from virgil.models import Image

    assert [f.name for f in dataclasses.fields(Image)] == [
        "flux",
        "dra",
        "ddec",
        "log_brightness",
        "support",
        "pixel_scale_mas",
        "rotation_deg",
        "ft",
        "blur_fwhm_mas",
    ]


@pytest.mark.parametrize("par", ["linear", "sqrt"])
def test_matches_image_in_a_system_to_1e8(par):
    pix = _blob(0.5)
    x = pix if par == "linear" else np.sqrt(pix)
    lin = System(
        star=PointSource(), env=LinearImage(x, SCALE, parametrisation=par)
    )
    ref = System(
        star=PointSource(),
        env=Image.from_brightness(pix / pix.sum(), SCALE, flux=0.5, floor=0.0),
    )
    u, v = DATA.u, DATA.v
    got = lin.model(u, v, 4.8e-6)
    want = ref.model(u, v, 4.8e-6)
    assert onp.max(onp.abs(onp.asarray(got - want))) < 1e-8


def test_support_pixels_stay_exactly_zero():
    support = circular_support(NPIX, SCALE, 40.0)
    for par in ("linear", "sqrt"):
        im = LinearImage(
            np.ones((NPIX, NPIX)), SCALE, support, parametrisation=par
        )
        assert onp.all(onp.asarray(im.fluxes)[~support] == 0.0)
        assert onp.all(onp.asarray(im.brightness)[~support] == 0.0)
    lo, hi = LinearImage(np.ones((NPIX, NPIX)), SCALE, support).bounds()
    assert onp.all(lo == 0.0) and onp.all(hi[~support] == 0.0)
    assert onp.all(onp.isinf(hi[support]))
    assert (
        LinearImage(
            np.ones((NPIX, NPIX)), SCALE, parametrisation="sqrt"
        ).bounds()
        is None
    )


def test_validation():
    with pytest.raises(ValueError, match="non-negative"):
        LinearImage(-np.ones((4, 4)), SCALE)
    with pytest.raises(ValueError, match="parametrisation"):
        LinearImage(np.ones((4, 4)), SCALE, parametrisation="log")
    with pytest.raises(ValueError, match="2D"):
        LinearImage(np.ones(4), SCALE)


@pytest.mark.parametrize(
    "reg", [TSV(1.0, beam_mas=BEAM), TV(1.0, beam_mas=BEAM), StarletL1(1.0)]
)
def test_regularizers_act_on_pixels_over_reference_flux(reg):
    pix = _blob(0.5)
    scene = LinearImage(pix, SCALE, reference_flux=0.5)
    scaled = LinearImage(2.0 * pix, SCALE, reference_flux=0.5)
    assert onp.allclose(reg.pixels(scene), pix / 0.5)
    assert onp.allclose(reg.pixels(scaled), 2.0 * pix / 0.5)
    # Same as an Image's unit-sum pixels when the flux is the reference.
    img = Image.from_brightness(pix / pix.sum(), SCALE, floor=0.0)
    assert float(reg.value(scene)) == pytest.approx(
        float(reg.value(img)), rel=1e-6
    )
    # The normaliser is the fixed reference flux, not the fitted one: doubling
    # the flux scales a quadratic penalty by 4 and a linear one by 2.
    power = 4.0 if isinstance(reg, TSV) else 2.0
    ratio = float(reg.value(scaled)) / float(reg.value(scene))
    assert ratio == pytest.approx(power, rel=1e-2)


def test_max_entropy_skilling_form_and_refusal():
    pix = _blob(0.5)
    im = LinearImage(
        np.sqrt(pix), SCALE, parametrisation="sqrt", reference_flux=0.5
    )
    mem = MaxEntropy(1.0)
    b = onp.asarray(pix) / 0.5
    q = onp.full(b.shape, 1.0 / b.size)
    want = onp.sum(b * onp.log(b / q) - b + q)
    assert float(mem.value(im)) == pytest.approx(float(want), rel=1e-6)
    # At the default the entropy is zero and its minimum is interior.
    flat = LinearImage(
        np.full((NPIX, NPIX), 1.0 / NPIX**2), SCALE, parametrisation="sqrt"
    )
    assert abs(float(mem.value(flat))) < 1e-12
    with pytest.raises(TypeError, match="sqrt"):
        mem.value(LinearImage(pix, SCALE))


def test_sqrt_max_entropy_fit_reaches_the_default_image():
    # Alone, Skilling's entropy is minimal at the default: flat, flux F0.
    truth = _blob(0.5)
    start = LinearImage(
        np.sqrt(truth), SCALE, parametrisation="sqrt", reference_flux=0.5
    )
    priors = image_priors(start)
    assert set(priors) == {"pixels"}
    mem = MaxEntropy(1.0)
    result = fit(start, priors, (), [mem], method="lbfgs", max_steps=500)
    assert float(mem.value(result.model)) < 1e-6 * float(mem.value(start))
    flat = 0.5 / NPIX**2
    assert onp.allclose(result.model.fluxes, flat, atol=1e-3 * flat)
    linear = LinearImage(truth, SCALE)
    with pytest.raises(TypeError, match="lbfgsb"):
        fit(linear, image_priors(linear), (), [mem], method="lbfgs")


def test_max_entropy_residuals_match_value_on_linear_image():
    im = LinearImage(
        np.sqrt(_blob(0.5)), SCALE, parametrisation="sqrt", reference_flux=0.5
    )
    mem = MaxEntropy(2.0)
    r = mem.residuals(im)
    assert float(0.5 * np.sum(r**2)) == pytest.approx(
        float(mem.value(im)), rel=1e-9
    )


def test_lone_image_visibilities_ignore_the_flux():
    pix = _blob(0.5)
    a = LinearImage(pix, SCALE).model(DATA.u, DATA.v, 4.8e-6)
    b = LinearImage(3.0 * pix, SCALE).model(DATA.u, DATA.v, 4.8e-6)
    assert onp.allclose(a, b, atol=1e-12)


def test_single_pixel_is_a_point_source_at_that_pixel():
    # Row 0 is North, column 0 East: the top-left pixel has dra, ddec > 0.
    npix, scale = 9, 5.0
    x = np.zeros((npix, npix)).at[0, 0].set(1.0)
    image = LinearImage(x, scale, dra=1.0, ddec=-2.0)
    expected = PointSource(dra=4 * scale + 1.0, ddec=4 * scale - 2.0)
    u, v = DATA.u, DATA.v
    assert np.allclose(
        image.model(u, v, 4.8e-6), expected.model(u, v, 4.8e-6), atol=1e-6
    )
    rendered = image.render(npix=npix, fov_mas=npix * scale)
    assert np.unravel_index(np.argmax(rendered), rendered.shape) == (0, 0)


def test_render_fourier_transform_matches_model_visibilities():
    im = LinearImage(_blob(0.5), SCALE)
    npix = NPIX
    rendered = onp.asarray(im.render(npix=npix, fov_mas=npix * SCALE))
    u = onp.array([0.0, 1.0, 2.0, 0.5]) * 1e5
    v = onp.array([0.0, 0.5, -1.0, 2.0]) * 1e5
    mas = onp.pi / 180 / 3600 / 1000
    x = -(onp.arange(npix) - 0.5 * (npix - 1)) * SCALE
    y = (0.5 * (npix - 1) - onp.arange(npix)) * SCALE
    xx, yy = onp.meshgrid(x, y)
    arg = onp.outer(u, xx.ravel()) + onp.outer(v, yy.ravel())
    ft = onp.exp(-2j * onp.pi * mas * arg) @ rendered.ravel()
    ft = ft / rendered.sum()
    got = onp.asarray(im.model(u * 4.8e-6, v * 4.8e-6, 4.8e-6))
    assert onp.max(onp.abs(ft - got)) < 1e-3


def test_smoothed_regulariser_on_a_linear_image():
    from virgil._geometry import gaussian_smooth

    pix = _blob(0.5)
    im = LinearImage(pix, SCALE, blur_fwhm_mas=15.0, reference_flux=0.5)
    reg = TSV(1.0, beam_mas=BEAM, smoothed=True)
    with pytest.warns(UserWarning, match="smoothed"):
        got = reg.pixels(im)
    want = gaussian_smooth(pix / 0.5, 15.0 / SCALE)
    assert onp.allclose(got, want, atol=1e-9)
