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


def test_flux_is_the_sum_of_pixels_and_roundtrips():
    pix = _blob(0.5)
    for par, x in (("linear", pix), ("sqrt", np.sqrt(pix))):
        im = LinearImage(x, SCALE, parametrisation=par)
        assert abs(float(im.total_flux) - 0.5) < 1e-12
        assert abs(float(im.brightness.sum()) - 1.0) < 1e-12
        assert onp.allclose(im.brightness * im.total_flux, pix, atol=1e-14)
        assert im.reference_flux == pytest.approx(0.5)


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


def test_sqrt_fit_runs():
    scene = System(
        star=PointSource(),
        env=LinearImage(
            np.sqrt(np.full((NPIX, NPIX), 0.5 / NPIX**2)),
            SCALE,
            parametrisation="sqrt",
        ),
    )
    priors = image_priors(scene)
    assert set(priors) == {"env.pixels"}
    result = fit(
        scene,
        priors,
        DATA,
        [MaxEntropy(1e-3, path="env")],
        method="lbfgs",
        max_steps=5,
    )
    assert onp.all(onp.isfinite(onp.asarray(result.model.env.fluxes)))


def test_toeplitz_layout_accepts_linear_image():
    from virgil.radio_toeplitz import _shape, _single_image

    im = LinearImage(np.ones((NPIX, NPIX)), SCALE)
    assert _single_image(System(env=im))[0] is im
    assert _shape(im) == (NPIX, NPIX)
