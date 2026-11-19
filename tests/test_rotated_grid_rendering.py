"""Rendering on rotated pixel grids, against analytic oracles (virgil#712 §12)."""

import jax
import jax.numpy as np
import numpy as onp
import pytest
from scipy.special import j1

from virgil._geometry import image_coordinates, image_visibilities, rotate
from virgil.models import GravityDarkenedStar, _blur_in_plane

NPIX, SCALE = 64, 1.0
MAS_INV = 180 * 3600 * 1000 / onp.pi  # per-radian frequency of 1 cycle/mas
ANGLES = (0.0, 25.0, 90.0, -90.0, 137.0)


def _pixel_grid(rotation_deg):
    """Sky coordinates (mas) of the pixel centres of a grid rotated by ``rotation_deg``."""
    xx, yy = image_coordinates(NPIX, NPIX * SCALE)
    return rotate(xx, yy, rotation_deg) if rotation_deg else (xx, yy)


def _freqs():
    """Frequencies (per radian) with ``diam`` = 20 mas first-null-free (q < 0.03 / mas)."""
    rng = onp.random.default_rng(1)
    return rng.uniform(-0.03, 0.03, size=(2, 12)) * MAS_INV


@pytest.mark.parametrize("rotation_deg", ANGLES)
def test_uniform_star_on_a_rotated_grid_matches_the_airy_visibility(
    rotation_deg,
):
    """A non-rotating star is a uniform disk; its FT is 2 J1(pi D q) / (pi D q)."""
    diam = 20.0
    xx, yy = _pixel_grid(rotation_deg)
    star = GravityDarkenedStar(diam, 0.0, 90.0, 0.0, n_lat=64)
    pix = star._image(xx, yy, SCALE)
    pix = pix / np.sum(pix)
    u, v = _freqs()
    # the pixels sit on the grid rotated by rotation_deg: transform in that frame
    uu, vv = rotate(u, v, -rotation_deg) if rotation_deg else (u, v)
    vis = onp.asarray(image_visibilities(pix, uu, vv, SCALE, "dft"))
    x = onp.pi * diam * onp.hypot(u, v) / MAS_INV
    airy = 2 * j1(x) / x
    onp.testing.assert_allclose(vis.real, airy, atol=3e-2)
    onp.testing.assert_allclose(vis.imag, 0.0, atol=3e-2)


@pytest.mark.parametrize("rotation_deg", ANGLES[1:])
def test_rotated_grid_image_is_the_unrotated_image_resampled(rotation_deg):
    """The sky-frame image is the same star whatever the grid: compare their
    visibilities at the same sky frequencies."""
    star = GravityDarkenedStar(18.0, 0.8, 60.0, 35.0, n_lat=32)
    u, v = _freqs()
    out = []
    for rot in (0.0, rotation_deg):
        xx, yy = _pixel_grid(rot)
        pix = star._image(xx, yy, SCALE)
        pix = pix / np.sum(pix)
        uu, vv = rotate(u, v, -rot) if rot else (u, v)
        out.append(onp.asarray(image_visibilities(pix, uu, vv, SCALE, "dft")))
    onp.testing.assert_allclose(out[1], out[0], atol=3e-2)


@pytest.mark.parametrize("rotation_deg", ANGLES)
@pytest.mark.parametrize("pa", (0.0, 40.0))
def test_blur_in_plane_matches_the_analytic_gaussian(rotation_deg, pa):
    """Blurring a sky-frame Gaussian by an in-plane kernel adds covariances."""
    with jax.enable_x64(True):
        xx, yy = _pixel_grid(rotation_deg)
        xx, yy = np.asarray(xx), np.asarray(yy)
        s0, fwhm, stretch = 6.0, 8.0, 0.5
        sk = fwhm / 2.3548200450309493
        image = np.exp(-0.5 * (xx**2 + yy**2) / s0**2)
        # the kernel has unit peak: divide by its sum, 2 pi sigma^2 stretch / px^2
        got = _blur_in_plane(image, fwhm, pa, stretch, SCALE, xx, yy) / (
            2 * onp.pi * sk**2 * stretch / SCALE**2
        )
        # kernel: sigma sk along the major axis (pa, East of North), sk*stretch across
        th = onp.deg2rad(pa)
        major = onp.array([onp.sin(th), onp.cos(th)])
        minor = onp.array([onp.cos(th), -onp.sin(th)])
        cov_k = sk**2 * (
            onp.outer(major, major) + stretch**2 * onp.outer(minor, minor)
        )
        cov = s0**2 * onp.eye(2) + cov_k
        inv = onp.linalg.inv(cov)
        p = onp.stack([onp.asarray(xx), onp.asarray(yy)], -1)
        quad = onp.einsum("...i,ij,...j->...", p, inv, p)
        want = onp.exp(-0.5 * quad) * s0**2 / onp.sqrt(onp.linalg.det(cov))
        onp.testing.assert_allclose(onp.asarray(got), want, atol=2e-3)
