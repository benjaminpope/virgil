"""``rotation_deg="grid"`` for pixel images, and the imaging defaults."""

import equinox as eqx
import jax
import jax.numpy as np
import numpy as onp
import pytest

from virgil import models
from virgil.coverage import ami_grid_record
from virgil.imaging import clean, dirty_image, starting_image
from virgil.models import GaussianDisk, Image, LinearImage, PointSource, System
from virgil.oidata import OIData

ROTATION = -6.9
SCALE, NPIX = 10.0, 16


def _lattice():
    return OIData(ami_grid_record(rotation_deg=ROTATION))


def _bare(data):
    return eqx.tree_at(
        lambda d: d.uv_grid, data, None, is_leaf=lambda x: x is None
    )


def _image(rotation_deg):
    rng = onp.random.default_rng(3)
    return Image(
        rng.normal(size=(NPIX, NPIX)), SCALE, rotation_deg=rotation_deg
    )


def test_grid_takes_the_matrix_transform_and_matches_the_direct_sum(
    monkeypatch,
):
    with jax.enable_x64(True):
        data = _lattice()
        calls = []
        original = models.grid_visibilities
        monkeypatch.setattr(
            models,
            "grid_visibilities",
            lambda *a, **k: calls.append(1) or original(*a, **k),
        )
        fast = _image("grid").model_on_grid(
            data.u, data.v, data.wavel, data.uv_grid
        )
        assert calls
        # A numeric rotation matching the lattice is the same call.
        same = _image(data.uv_grid.rotation_deg).model_on_grid(
            data.u, data.v, data.wavel, data.uv_grid
        )
        assert np.array_equal(fast, same)
        # The direct sum on the same rotated pixels.
        monkeypatch.setattr(models, "grid_visibilities", original)
        slow = _image(data.uv_grid.rotation_deg).model(
            data.u, data.v, data.wavel
        )
        assert np.max(np.abs(fast - slow)) < 1e-10 * np.max(np.abs(slow))


def test_grid_without_a_lattice_is_north_up():
    data = _bare(_lattice())
    assert data.uv_grid is None
    grid, north = _image("grid"), _image(0.0)
    assert np.array_equal(
        grid.model(data.u, data.v, data.wavel),
        north.model(data.u, data.v, data.wavel),
    )
    assert np.array_equal(grid.render(32, 100.0), north.render(32, 100.0))
    assert np.array_equal(
        Image.from_model(
            GaussianDisk(20.0), NPIX, SCALE, rotation_deg="grid"
        ).brightness,
        Image.from_model(GaussianDisk(20.0), NPIX, SCALE).brightness,
    )


def test_rotation_deg_rejects_other_strings():
    with pytest.raises(ValueError, match="rotation_deg"):
        Image(np.zeros((4, 4)), 1.0, rotation_deg="sky")
    with pytest.raises(ValueError, match="rotation_deg"):
        LinearImage(np.ones((4, 4)), 1.0, rotation_deg="grid")


def test_starting_image_and_clean_follow_the_lattice(monkeypatch):
    data = _lattice().with_model(
        System(star=PointSource(), env=GaussianDisk(15.0, flux=0.3))
    )
    start = starting_image(data, star=False, oversample=2.0)
    assert start.rotation_deg == pytest.approx(data.uv_grid.rotation_deg)
    result = clean(
        data, npix=NPIX, pixel_scale_mas=SCALE, max_iterations=3, base=None
    )
    assert result.model.rotation_deg == pytest.approx(
        data.uv_grid.rotation_deg
    )
    # No lattice: North up, as before.
    bare = _bare(data)
    assert starting_image(bare, star=False).rotation_deg == 0.0
    assert (
        clean(
            bare, npix=NPIX, pixel_scale_mas=SCALE, max_iterations=3, base=None
        ).model.rotation_deg
        == 0.0
    )
    explicit = starting_image(data, star=False, rotation_deg=0.0)
    assert explicit.rotation_deg == 0.0


def test_dirty_image_rotation_matches_a_rotated_sky():
    data = _lattice().with_model(
        System(star=PointSource(), env=GaussianDisk(15.0, flux=0.3))
    )
    north = dirty_image(data, NPIX, SCALE)
    assert np.array_equal(dirty_image(data, NPIX, SCALE, rotation_deg=0.0), north)
    turned = dirty_image(data, NPIX, SCALE, rotation_deg=ROTATION)
    assert turned.shape == north.shape
    assert not np.allclose(turned, north)
