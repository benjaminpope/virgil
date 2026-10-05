from pathlib import Path

import jax
import jax.numpy as np
import numpy as onp
import pytest

from virgil import pixel_offsets
from virgil._geometry import image_visibilities
from virgil.amigo import load_oi_data
from virgil.likelihood import model_loglike
from virgil.models import (
    BinaryModelCartesian,
    GaussianDisk,
    Image,
    PointSource,
    System,
    circular_support,
)
from virgil.oidata import OIData, cp_indices

MAS2RAD = onp.pi / 180.0 / 3600.0 / 1000.0
WAVEL = 4.8e-6
DISCO_PRODUCT = (
    Path(__file__).resolve().parents[1] / "data" / "calibrated_visibility.npy"
)


def _baselines(n=60, max_m=6.5, seed=0):
    rng = onp.random.default_rng(seed)
    return rng.uniform(-max_m, max_m, n), rng.uniform(-max_m, max_m, n)


def _dft_reference(image, uu, vv, pixel_scale_mas):
    """Direct float64 sum over pixels, written independently of the library."""
    image = onp.asarray(image, dtype=onp.float64)
    nrow, ncol = image.shape
    x = -(onp.arange(ncol) - 0.5 * (ncol - 1)) * pixel_scale_mas
    y = (0.5 * (nrow - 1) - onp.arange(nrow)) * pixel_scale_mas
    xx, yy = onp.meshgrid(x, y, indexing="xy")
    arg = onp.outer(uu, xx.ravel()) + onp.outer(vv, yy.ravel())
    return onp.exp(-2j * onp.pi * MAS2RAD * arg) @ image.ravel()


@pytest.mark.parametrize("shape", [(17, 17), (16, 16), (12, 15)])
def test_dft_matches_float64_direct_sum(shape):
    rng = onp.random.default_rng(1)
    image = rng.uniform(0.0, 1.0, shape)
    image /= image.sum()
    u, v = _baselines()
    uu, vv = u / WAVEL, v / WAVEL
    expected = _dft_reference(image, uu, vv, 3.0)
    got = image_visibilities(np.asarray(image), uu, vv, 3.0)
    assert onp.max(onp.abs(onp.asarray(got) - expected)) < 3e-6


def test_dft_keeps_the_shape_of_the_frequencies():
    uu = np.ones((4, 3)) * 1e6
    assert image_visibilities(np.ones((5, 5)), uu, uu, 1.0).shape == (4, 3)


def test_single_pixel_is_a_point_source_at_that_pixel():
    # Row 3 is above the centre (North), column 11 right of it (West).
    npix, scale = 17, 2.0
    log_b = np.full((npix, npix), -np.inf).at[3, 11].set(0.0)
    image = Image(log_b, scale)
    offsets = pixel_offsets(npix, scale)
    assert offsets[11] < 0.0 < offsets[3]
    point = PointSource(dra=offsets[11], ddec=offsets[3])
    u, v = _baselines()
    assert np.allclose(
        image.model(u, v, WAVEL), point.model(u, v, WAVEL), atol=1e-6
    )


def test_orientation_east_left_north_up():
    # Brightest pixel at the top left: North-East, so dra > 0 and ddec > 0.
    npix, scale = 9, 5.0
    log_b = np.full((npix, npix), -np.inf).at[0, 0].set(0.0)
    image = Image(log_b, scale, dra=1.0, ddec=-2.0)
    expected = PointSource(dra=4 * scale + 1.0, ddec=4 * scale - 2.0)
    u, v = _baselines()
    assert np.allclose(
        image.model(u, v, WAVEL), expected.model(u, v, WAVEL), atol=1e-6
    )
    rendered = image.render(npix=npix, fov_mas=npix * scale)
    assert np.unravel_index(np.argmax(rendered), rendered.shape) == (0, 0)


def test_pixelised_gaussian_matches_gaussian_disk():
    disk = GaussianDisk(sigma=6.0)
    image = Image.from_model(disk, 65, 1.0)
    u, v = _baselines(max_m=4.0)
    assert np.allclose(
        image.model(u, v, WAVEL), disk.model(u, v, WAVEL), atol=1e-4
    )


def test_render_on_the_native_grid_returns_the_pixels():
    rng = onp.random.default_rng(2)
    image = Image(rng.normal(size=(15, 15)), 2.0)
    assert np.allclose(image.render(npix=15, fov_mas=30.0), image.brightness)


def test_brightness_is_unit_sum_and_zero_outside_support():
    support = circular_support(21, 1.5, radius_mas=9.0)
    image = Image(np.zeros((21, 21)), 1.5, support=support)
    assert np.isclose(image.brightness.sum(), 1.0)
    assert np.all(image.brightness[~support] == 0.0)
    assert np.isclose(image.model(np.zeros(1), np.zeros(1), WAVEL)[0], 1.0)
    grad = jax.grad(
        lambda lb: image.set("log_brightness", lb).brightness[10, 10]
    )(image.log_brightness)
    assert np.all(grad[~support] == 0.0)


def test_circular_support_is_centred():
    support = circular_support(8, 1.0, radius_mas=2.0)
    assert np.array_equal(support, support[::-1, ::-1])
    assert support.sum() == 12


def test_invalid_inputs_raise():
    with pytest.raises(ValueError, match="2D"):
        Image(np.zeros(9), 1.0)
    with pytest.raises(ValueError, match="shape"):
        Image(np.zeros((4, 4)), 1.0, support=np.ones((3, 3), bool))
    with pytest.raises(ValueError, match="at least one"):
        Image(np.zeros((4, 4)), 1.0, support=np.zeros((4, 4), bool))


def test_image_mixes_with_analytic_components_in_a_system():
    envelope = Image.from_model(GaussianDisk(sigma=5.0), 33, 1.0, flux=0.25)
    scene = System(star=PointSource(), env=envelope)
    analytic = System(star=PointSource(), env=GaussianDisk(5.0, flux=0.25))
    u, v = _baselines(max_m=4.0)
    assert np.allclose(
        scene.model(u, v, WAVEL), analytic.model(u, v, WAVEL), atol=1e-4
    )
    moved = scene.set("env.flux", 0.5)
    assert not np.allclose(moved.model(u, v, WAVEL), scene.model(u, v, WAVEL))


@pytest.mark.parametrize("x64", [False, True])
def test_dft_accuracy_at_long_baselines(x64):
    # CHARA-like 330 m baselines at 1.6 µm with 0.1 mas pixels: large
    # phase arguments, where float32 loses digits but stays well below
    # typical closure-phase errors.
    with jax.enable_x64(x64):
        rng = onp.random.default_rng(3)
        image = rng.uniform(0.0, 1.0, (32, 32))
        image /= image.sum()
        u, v = _baselines(max_m=330.0)
        uu, vv = u / 1.6e-6, v / 1.6e-6
        expected = _dft_reference(image, uu, vv, 0.1)
        got = onp.asarray(image_visibilities(np.asarray(image), uu, vv, 0.1))
        tol = 1e-12 if x64 else 2e-5
        assert onp.max(onp.abs(got - expected)) < tol


PAIRS = onp.array([[1, 2], [1, 3], [1, 4], [2, 3], [2, 4], [3, 4]])
TRIANGLES = onp.array([[1, 2, 3], [1, 2, 4], [1, 3, 4], [2, 3, 4]])
STATIONS = onp.array([[0.0, 0.0], [3.2, 0.2], [1.4, 2.6], [-1.1, 1.8]])


def _array_data(**extra):
    """Noiseless V² and closure phases of a binary on a 4-station array."""
    delta = STATIONS[PAIRS[:, 1] - 1] - STATIONS[PAIRS[:, 0] - 1]
    u, v = delta[:, 0], delta[:, 1]
    binary = BinaryModelCartesian(60.0, -40.0, 0.05)
    cvis = onp.asarray(binary.model(u, v, WAVEL))
    i1, i2, i3 = cp_indices(PAIRS, TRIANGLES)
    return {
        "u": u,
        "v": v,
        "wavel": WAVEL,
        "vis": onp.abs(cvis) ** 2,
        "d_vis": onp.full(len(u), 1e-3),
        "phi": onp.angle(cvis[i1] * cvis[i2] / cvis[i3]),
        "d_phi": onp.full(len(i1), 1e-2),
        "i_cps1": i1,
        "i_cps2": i2,
        "i_cps3": i3,
        **extra,
    }


@pytest.mark.parametrize(
    "data",
    [
        pytest.param(lambda: OIData(_array_data()), id="v2-closure-phases"),
        pytest.param(lambda: OIData(_array_data(vis_mode="amp")), id="amp"),
        pytest.param(
            lambda: OIData(_array_data(vis_mode="logamp")), id="logamp"
        ),
        pytest.param(
            lambda: OIData(_array_data(phi_mat=onp.eye(4)[:2])),
            id="kernel-phases",
        ),
        pytest.param(
            lambda: load_oi_data(DISCO_PRODUCT)["F480M"], id="mixed-disco"
        ),
    ],
)
def test_loglike_gradient_wrt_pixels_is_finite_and_nonzero(data):
    data = data()
    # A star plus an off-centre clump, which is not the data's binary.
    clump = Image.from_model(
        GaussianDisk(sigma=8.0, dra=20.0, ddec=10.0), 24, 5.0, flux=0.3
    )
    scene = System(star=PointSource(), env=clump)

    def loglike(log_brightness):
        moved = scene.set("env.log_brightness", log_brightness)
        return model_loglike(moved, data)

    grad = jax.grad(loglike)(clump.log_brightness)
    assert grad.shape == clump.log_brightness.shape
    assert np.all(np.isfinite(grad))
    assert np.max(np.abs(grad)) > 0.0


def test_dft_broadcasts_frequencies():
    image = np.ones((5, 5)) / 25.0
    uu = np.linspace(0.0, 2e6, 3)[:, None]
    vv = np.linspace(-1e6, 1e6, 4)[None, :]
    grid = image_visibilities(image, uu, vv, 2.0)
    assert grid.shape == (3, 4)
    row = image_visibilities(image, uu[1, 0], vv[0], 2.0)
    assert np.allclose(grid[1], row)


def test_log_brightness_must_have_a_finite_supported_pixel():
    with pytest.raises(ValueError, match="positive"):
        Image.from_brightness(np.zeros((4, 4)), 1.0)
    with pytest.raises(ValueError, match="finite"):
        Image(np.full((4, 4), -np.inf), 1.0)
    with pytest.raises(ValueError, match="NaN"):
        Image(np.zeros((4, 4)).at[1, 1].set(np.nan), 1.0)
    # Pixels outside the support may be anything.
    support = np.zeros((4, 4), bool).at[2, 2].set(True)
    Image(np.zeros((4, 4)).at[0, 0].set(np.nan), 1.0, support=support)


def test_from_brightness_floor_ignores_pixels_outside_support():
    brightness = np.array([[1e6, 0.0], [2.0, 1.0]])
    support = np.array([[False, True], [True, True]])
    image = Image.from_brightness(brightness, 1.0, support=support)
    assert np.allclose(image.brightness[1], np.array([2.0, 1.0]) / 3.0)
