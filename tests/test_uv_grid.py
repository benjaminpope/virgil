import equinox as eqx
import jax
import jax.numpy as np
import numpy as onp
import pytest

from virgil._geometry import (
    find_uv_grid,
    grid_visibilities,
    image_visibilities,
    rotate,
)
from virgil.coverage import ami_grid_record
from virgil.models import (
    GaussianDisk,
    Image,
    PointSource,
    System,
)
from virgil.oidata import OIData
from virgil.scenes import ring

WAVEL = 4.3e-6


def _without_grid(data):
    return eqx.tree_at(
        lambda d: d.uv_grid, data, None, is_leaf=lambda x: x is None
    )


@pytest.mark.validates(
    "virgil.models.Image.model_on_grid", roots=["self-consistency"]
)
@pytest.mark.parametrize("shape", [(17, 17), (16, 16), (12, 15)])
def test_grid_transform_equals_the_dft_at_the_grid_points(shape):
    rng = onp.random.default_rng(0)
    image = np.asarray(rng.uniform(size=shape))
    uu_axis = np.linspace(-1.5e6, 1.5e6, 9)
    vv_axis = np.linspace(-0.5e6, 2e6, 6)
    uu, vv = np.meshgrid(uu_axis, vv_axis)
    expected = image_visibilities(image, uu, vv, 12.0)
    got = grid_visibilities(image, uu_axis, vv_axis, 12.0)
    assert got.shape == (6, 9)
    assert np.allclose(got, expected, atol=1e-5)


def test_rotate_follows_position_angle_north_to_east():
    # A frame rotated to PA 90 has its "up" axis pointing East (+x).
    x, y = rotate(0.0, 1.0, 90.0)
    assert np.allclose(np.array([x, y]), np.array([1.0, 0.0]), atol=1e-7)
    back = rotate(*rotate(0.3, -0.7, 25.0), -25.0)
    assert np.allclose(np.array(back), np.array([0.3, -0.7]), atol=1e-6)


def test_find_uv_grid_recovers_a_rotated_lattice_subset():
    rng = onp.random.default_rng(1)
    col, row = onp.meshgrid(onp.arange(7), onp.arange(4))
    keep = rng.uniform(size=col.size) < 0.7
    a, b = 0.5 * col.ravel()[keep] - 1.0, 0.8 * row.ravel()[keep] + 0.2
    # In float64, as records are read: float32 rounding is too coarse for
    # the lattice test, and such data would use the per-point path.
    c, s_ = onp.cos(onp.radians(-12.0)), onp.sin(onp.radians(-12.0))
    u, v = c * a + s_ * b, -s_ * a + c * b
    grid = find_uv_grid(u, v)
    assert grid is not None
    assert np.isclose(grid.rotation_deg, -12.0)
    gu, gv = onp.meshgrid(onp.asarray(grid.u_axis), onp.asarray(grid.v_axis))
    su, sv = rotate(
        gu.ravel()[grid.index], gv.ravel()[grid.index], grid.rotation_deg
    )
    assert onp.allclose(su, u, atol=1e-5) and onp.allclose(sv, v, atol=1e-5)


def test_find_uv_grid_rejects_irregular_samples():
    rng = onp.random.default_rng(2)
    assert find_uv_grid(rng.normal(size=30), rng.normal(size=30)) is None
    u, v = onp.meshgrid(onp.arange(5.0), onp.arange(5.0))
    u = u.ravel() + 1e-3 * rng.normal(size=25)
    assert find_uv_grid(u, v.ravel()) is None


def test_rotated_image_pixel_lands_at_the_rotated_sky_position():
    # The top-centre pixel of a frame rotated to PA 90 is due East.
    npix, scale = 9, 4.0
    log_b = np.full((npix, npix), -np.inf).at[0, 4].set(0.0)
    image = Image(log_b, scale, rotation_deg=90.0)
    east = PointSource(dra=4 * scale, ddec=0.0)
    rng = onp.random.default_rng(3)
    u, v = rng.uniform(-6.0, 6.0, (2, 30))
    assert np.allclose(
        image.model(u, v, WAVEL), east.model(u, v, WAVEL), atol=1e-6
    )
    rendered = image.render(npix=npix, fov_mas=npix * scale)
    assert np.unravel_index(np.argmax(rendered), rendered.shape) == (4, 0)


@pytest.mark.validates("virgil.models.Image", roots=["mathematics"])
def test_rotated_from_model_matches_the_analytic_model():
    disk = GaussianDisk(sigma=5.0, dra=4.0, ddec=-3.0)
    image = Image.from_model(disk, 49, 1.0, rotation_deg=30.0)
    rng = onp.random.default_rng(4)
    u, v = rng.uniform(-3.0, 3.0, (2, 30))
    assert np.allclose(
        image.model(u, v, WAVEL), disk.model(u, v, WAVEL), atol=1e-4
    )


@pytest.mark.validates(
    "virgil.models.Image.model_on_grid", roots=["self-consistency"]
)
@pytest.mark.parametrize("x64", [False, True])
def test_grid_path_matches_the_per_point_path(x64):
    with jax.enable_x64(x64):
        data = OIData(ami_grid_record(rotation_deg=-6.9))
        assert data.uv_grid is not None
        truth = ring(48, 8.0, 80.0, 10.0, 50.0, 30.0, 0.5, 100.0)
        env = Image.from_brightness(
            truth, 8.0, flux=0.3, rotation_deg=data.uv_grid.rotation_deg
        )
        scene = System(star=PointSource(), env=env)
        fast = data.model(scene)
        slow = _without_grid(data).model(scene)
        tol = 1e-12 if x64 else 1e-6
        assert np.max(np.abs(fast - slow)) < tol * np.max(np.abs(slow)) + tol


@pytest.mark.validates(
    "virgil.models.Image.model_on_grid", roots=["self-consistency"]
)
def test_mismatched_rotation_falls_back_to_the_exact_per_point_path():
    data = OIData(ami_grid_record(rotation_deg=-6.9))
    scene = System(star=PointSource(), env=Image(np.zeros((9, 9)), 20.0))
    assert np.allclose(data.model(scene), _without_grid(data).model(scene))
