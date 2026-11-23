"""Rendering on rotated pixel grids, against analytic oracles (virgil#712 §12)."""

import jax.numpy as np
import numpy as onp
import pytest
from scipy.special import j1

from virgil._geometry import image_coordinates, image_visibilities, rotate
from virgil.models import (
    GaussianRing,
    GravityDarkenedStar,
    ModulatedGaussianRim,
    MultipoleDisk,
    _blur_in_plane,
    _lattice_axis,
)

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
    onp.testing.assert_allclose(onp.asarray(got), want, atol=1e-4)


@pytest.mark.parametrize("rotation_deg", (20.0,))
def test_star_samples_are_the_turned_pixel_square(rotation_deg):
    """Coarse pixels average 4 x 4 samples of the disk indicator over the pixel
    square turned with the lattice: a pixel carries 2 * mean * scale^2."""
    diam, scale, npix = 20.0, 5.0, 12
    xx, yy = image_coordinates(npix, npix * scale)
    xx, yy = (onp.asarray(a) for a in rotate(xx, yy, rotation_deg))
    # off-centre: a centred disk is symmetric under the sign flip being tested
    dra, ddec = 1.3, -0.7
    star = GravityDarkenedStar(
        diam, 0.0, 90.0, 0.0, dra=dra, ddec=ddec, n_lat=64
    )
    got = onp.asarray(star._image(xx, yy, scale)) / (2 * scale**2)
    t = ((onp.arange(4) + 0.5) / 4 - 0.5) * scale

    def coverage(angle):
        th = onp.deg2rad(angle)
        e1 = onp.array(
            [onp.cos(th), -onp.sin(th)]
        )  # the grid's x axis on the sky
        e2 = onp.array([onp.sin(th), onp.cos(th)])
        ox = t[:, None] * e1[0] + t[None, :] * e2[0]
        oy = t[:, None] * e1[1] + t[None, :] * e2[1]
        px = xx[..., None, None] + ox - dra
        py = yy[..., None, None] + oy - ddec
        return (px**2 + py**2 <= (diam / 2) ** 2).mean(axis=(-1, -2))

    # the same samples as the disk's indicator; a square turned the other way
    # (or not at all) misses
    assert onp.abs(got - coverage(rotation_deg)).mean() < 1e-5
    assert onp.abs(got - coverage(-rotation_deg)).mean() > 1e-3
    assert onp.abs(got - coverage(0.0)).mean() > 1e-3


@pytest.mark.parametrize("rotation_deg", (0.0, 25.0, -90.0, 137.0))
def test_lattice_axis_is_the_grid_rotation_from_any_two_pixels(rotation_deg):
    xx, yy = (onp.asarray(a) for a in _pixel_grid(rotation_deg))
    th = onp.deg2rad(rotation_deg)
    want = onp.array([onp.cos(th), -onp.sin(th)])
    for sl in (np.s_[:, :], np.s_[:, :2], np.s_[:2, :1]):
        got = onp.array([float(c) for c in _lattice_axis(xx[sl], yy[sl])])
        onp.testing.assert_allclose(got, want, atol=1e-5)


def _rim():
    return ModulatedGaussianRim(
        24.0, 6.0, 55.0, 30.0, az_amps=(0.3,), az_pas=(10.0,)
    )


def _multipole():
    return MultipoleDisk(
        GaussianRing(10.0, 4.0), inc=55.0, pa=30.0, blur_fwhm=5.0
    )


@pytest.mark.parametrize("rotation_deg", (0.0, 25.0, 90.0, 137.0))
@pytest.mark.parametrize("make", (_rim, _multipole))
def test_blurred_disk_render_on_a_rotated_grid_is_the_same_sky_image(
    make, rotation_deg
):
    """A rotated grid and an unrotated one render the same sky: their
    visibilities at the same sky frequencies agree."""
    model = make()
    u, v = _freqs()
    out = []
    for rot in (0.0, rotation_deg):
        xx, yy = _pixel_grid(rot)
        pix = model._image(xx, yy, SCALE)
        pix = pix / np.sum(pix)
        uu, vv = rotate(u, v, -rot) if rot else (u, v)
        out.append(onp.asarray(image_visibilities(pix, uu, vv, SCALE, "dft")))
    onp.testing.assert_allclose(out[1], out[0], atol=2e-3)
