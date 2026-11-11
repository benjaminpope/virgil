"""``rotation_deg="grid"`` for pixel images, and the imaging defaults."""

import dataclasses

import equinox as eqx
import jax
import jax.numpy as np
import numpy as onp
import pytest
from jax.scipy.ndimage import map_coordinates

from virgil import models
from virgil.coverage import ami_grid_record
from virgil._geometry import image_coordinates, rotate
from virgil.imaging import (
    beam,
    CleanResult,
    clean,
    convolve_beam,
    dirty_image,
    starting_image,
)
from virgil.metrics import resample
from virgil.scenes import gaussian_blob
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


def _uses_matrix_transform(monkeypatch, model, data):
    calls = []
    original = models.grid_visibilities
    monkeypatch.setattr(
        models,
        "grid_visibilities",
        lambda *a, **k: calls.append(1) or original(*a, **k),
    )
    model.model_on_grid(data.u, data.v, data.wavel, data.uv_grid)
    return bool(calls)


def test_defaults_are_north_up_and_grid_is_an_opt_in(monkeypatch):
    data = _lattice().with_model(
        System(star=PointSource(), env=GaussianDisk(15.0, flux=0.3))
    )
    kwargs = dict(
        npix=NPIX, pixel_scale_mas=SCALE, max_iterations=3, base=None
    )
    # Unchanged default on lattice data: North up, so the direct transform.
    start = starting_image(data, star=False, oversample=2.0)
    assert start.rotation_deg == 0.0
    assert clean(data, **kwargs).model.rotation_deg == 0.0
    assert not _uses_matrix_transform(monkeypatch, start, data)
    # Opt-in: the lattice frame, and the matrix transform.
    grid = starting_image(
        data, star=False, oversample=2.0, rotation_deg="grid"
    )
    assert grid.rotation_deg == pytest.approx(data.uv_grid.rotation_deg)
    assert _uses_matrix_transform(monkeypatch, grid, data)
    result = clean(data, rotation_deg="grid", **kwargs)
    assert result.model.rotation_deg == pytest.approx(
        data.uv_grid.rotation_deg
    )
    assert _uses_matrix_transform(monkeypatch, result.model, data)
    # Without a lattice "grid" is North up, as before.
    bare = _bare(data)
    assert (
        starting_image(bare, star=False, rotation_deg="grid").rotation_deg
        == 0.0
    )
    assert clean(bare, rotation_deg="grid", **kwargs).model.rotation_deg == 0.0


def _correlation(a, b):
    a, b = onp.ravel(a) - onp.mean(a), onp.ravel(b) - onp.mean(b)
    return float(a @ b / onp.sqrt((a @ a) * (b @ b)))


def test_dirty_image_in_the_grid_frame_is_the_north_up_map_turned():
    angle = 35.0
    source = System(
        star=PointSource(), c=PointSource(flux=0.5, dra=30.0, ddec=40.0)
    )
    data = OIData(ami_grid_record(rotation_deg=angle)).with_model(source)
    assert data.uv_grid.rotation_deg == pytest.approx(angle)
    n, h = 32, SCALE / 2

    def dirty(rotation):
        return onp.asarray(
            dirty_image(data, n, h, flux_ratio=0.5, rotation_deg=rotation)
        )

    north = dirty(0.0)
    assert onp.array_equal(dirty("grid"), dirty(angle))

    # North-up pixel (x, y) sits at rotate(x, y, -angle) in the turned grid.
    xx, yy = (onp.asarray(a) for a in image_coordinates(n, n * h))
    centre = 0.5 * (n - 1)
    inside = onp.hypot(xx, yy) < 60.0

    def agreement(sign):
        gx, gy = (onp.asarray(a) for a in rotate(xx, yy, -sign * angle))
        turned = map_coordinates(
            np.asarray(dirty(angle)),
            [centre - gy / h, centre - gx / h],
            order=1,
            mode="constant",
        )
        return _correlation(onp.asarray(turned)[inside], north[inside])

    assert agreement(1) > 0.99
    assert agreement(-1) < 0.5
    with pytest.raises(ValueError, match="rotation_deg"):
        dirty_image(data, NPIX, SCALE, rotation_deg="sky")


def test_scoring_sees_a_rotated_image_as_the_sky_and_loses_only_the_corners():
    blob = gaussian_blob(NPIX, 1.0, 2.0, dra=0.0, ddec=0.0)
    north = Image.from_brightness(blob, 1.0, floor=1e-3)
    turned = Image.from_brightness(blob, 1.0, floor=1e-3, rotation_deg=45.0)
    assert np.array_equal(
        resample(north, None, NPIX, 1.0), resample(north, 1.0, NPIX, 1.0)
    )
    # A blob inside the inscribed circle loses nothing.
    sky = onp.asarray(turned.render(NPIX, float(NPIX)))
    got = onp.asarray(resample(turned, None, NPIX, 1.0))
    assert onp.allclose(got / got.sum(), sky / sky.sum(), atol=1e-6)
    assert got.sum() == pytest.approx(
        float(np.sum(turned.brightness)), rel=0.05
    )
    # Flux in a corner turns off the North-up box: it is lost from the
    # score, and the rest is not rescaled to make up for it.
    cornered = onp.array(blob)
    cornered[0, 0] += 5.0 * cornered.max()
    corner = Image.from_brightness(
        cornered, 1.0, floor=1e-3, rotation_deg=45.0
    )
    lost = onp.asarray(resample(corner, None, NPIX, 1.0))
    assert lost.sum() < 0.95 * got.sum()


def test_restored_image_turns_the_beam_into_the_grid_frame():
    # An elongated beam on a single component: rendering the grid-frame
    # restored image as the sky must give the North-up convolution.
    angle, n = 35.0, 41
    b = beam(_lattice())
    b = dataclasses.replace(
        b, major_mas=4 * SCALE, minor_mas=1.5 * SCALE, pa_deg=30.0
    )
    comps = onp.zeros((n, n))
    comps[n // 2, n // 2] = 1.0
    model = Image(comps, SCALE, rotation_deg=angle)
    result = CleanResult(model, comps, SCALE, 1.0, "max_iterations")
    assert result.rotation_deg == angle
    north = onp.asarray(convolve_beam(comps, SCALE, b))
    fov = n * SCALE

    def sky(pixels, rotation_deg=0.0):
        image = Image.from_brightness(
            onp.asarray(pixels), SCALE, floor=1e-9, rotation_deg=rotation_deg
        )
        return onp.asarray(image.render(n, fov))

    sky_north = sky(north)
    assert _correlation(sky(result.restored(b), angle), sky_north) > 0.99
    unturned = convolve_beam(comps, SCALE, b)
    assert _correlation(sky(unturned, angle), sky_north) < 0.95
