"""Crescent and Blurred: the Fourier algebra against independent checks."""

import jax
import jax.numpy as np
import numpy as onp
import pytest

from virgil import Crescent, GaussianDisk, System, UniformDisk
from virgil._geometry import image_coordinates
from virgil.models import (
    Blurred,
    GaussianRing,
    Image,
    MultipoleDisk,
    Rotated,
    _multipole_disks,
    smeared_model,
)

WAVEL = 1e-6
MAS2RAD = onp.pi / 180 / 3600e3
U = np.array([0.0, 3.0, -7.0, 12.0, 2.0, 25.0])
V = np.array([0.0, 4.0, 5.0, -3.0, -18.0, 9.0])


def test_annulus_is_difference_of_uniform_disks():
    r_out, r_in = 20.0, 8.0
    got = Crescent(r_out, r_in).model(U, V, WAVEL)
    ud_out = UniformDisk(2 * r_out).model(U, V, WAVEL)
    ud_in = UniformDisk(2 * r_in).model(U, V, WAVEL)
    want = (r_out**2 * ud_out - r_in**2 * ud_in) / (r_out**2 - r_in**2)
    assert np.allclose(got, want, atol=1e-6)


def test_no_hole_no_fade_is_uniform_disk():
    got = Crescent(15.0, 0.0, ecc=0.7, pa=40.0).model(U, V, WAVEL)
    assert np.allclose(got, UniformDisk(30.0).model(U, V, WAVEL), atol=1e-6)


@pytest.mark.parametrize("pa", [0.0, 35.0, 200.0])
def test_visibility_matches_rendered_image(pa):
    """The analytic transform is the DFT of the analytic image."""
    model = Crescent(20.0, 7.0, ecc=0.6, fade=0.8, pa=pa, dra=3.0, ddec=-2.0)
    npix, fov = 512, 100.0
    image = model.render(npix, fov)
    xx, yy = image_coordinates(npix, fov)
    q_u, q_v = U / WAVEL, V / WAVEL
    phase = np.exp(
        -2j
        * np.pi
        * MAS2RAD
        * (q_u[:, None, None] * xx + q_v[:, None, None] * yy)
    )
    dft = np.sum(image * phase, axis=(1, 2))
    assert np.allclose(model.model(U, V, WAVEL), dft, atol=3e-3)


@pytest.mark.parametrize("fade", [0.0, 0.6, 1.0])
def test_zero_spacing_flux_matches_area_moments(fade):
    """T(0) = (1 - f/2) area + f/(2R) * first moment of the shape along the axis."""
    r_out, r_in, ecc = 20.0, 7.0, 0.6
    model = Crescent(r_out, r_in, ecc=ecc, fade=fade, pa=70.0)
    area = onp.pi * (r_out**2 - r_in**2)
    # The inner disk sits at -ecc (R - r) on the axis, so the shape's
    # first moment is +pi r^2 ecc (R - r).
    moment = onp.pi * r_in**2 * ecc * (r_out - r_in)
    want = (1 - fade / 2) * area + fade / (2 * r_out) * moment
    got = model._transform(np.zeros(()), np.zeros(()))
    assert np.allclose(got, want, rtol=1e-5)
    assert np.allclose(model.model(U, V, WAVEL)[0], 1.0)


def test_bright_side_is_pa():
    image = Crescent(20.0, 7.0, ecc=1.0, fade=1.0, pa=90.0).render(128, 60.0)
    # Column 0 is East: the thick, bright side is on the left.
    assert image[:, :64].sum() > 2 * image[:, 64:].sum()


@pytest.mark.parametrize("ecc", [0.0, 1.0])
@pytest.mark.parametrize("fade", [0.0, 1.0])
def test_brightness_non_negative(ecc, fade):
    xx, yy = image_coordinates(200, 60.0)
    image = Crescent(20.0, 7.0, ecc=ecc, fade=fade, pa=33.0)._centred_image(
        xx, yy, 0.3
    )
    assert onp.all(onp.asarray(image) >= 0.0)


def test_gradients_finite_including_zero_spacing():
    # U[0] = V[0] = 0, so the zero spacing is among the baselines.
    @jax.jit
    @jax.grad
    def grad(params):
        r_out, r_in, ecc, fade, pa = params
        vis = Crescent(r_out, r_in, ecc, fade, pa).model(U, V, WAVEL)
        return np.sum(np.abs(vis) ** 2)

    # r_in = 0 sits on the boundary of the allowed range.
    for r_in in (7.0, 0.0):
        values = grad(np.array([20.0, r_in, 0.5, 0.6, 30.0]))
        assert onp.all(onp.isfinite(values))


def test_pa_agrees_with_rotated():
    base = Crescent(20.0, 7.0, ecc=0.5, fade=0.7, pa=0.0)
    turned = Crescent(20.0, 7.0, ecc=0.5, fade=0.7, pa=57.0)
    assert np.allclose(
        turned.model(U, V, WAVEL),
        Rotated(base, 57.0).model(U, V, WAVEL),
        atol=1e-6,
    )


def test_blurred_gaussian_is_wider_gaussian():
    # Gaussians add in quadrature: sigma 3 blurred by sigma 4 is sigma 5.
    fwhm = 4.0 * 2.0 * onp.sqrt(2.0 * onp.log(2.0))
    got = Blurred(GaussianDisk(3.0), fwhm).model(U, V, WAVEL)
    assert np.allclose(got, GaussianDisk(5.0).model(U, V, WAVEL), atol=1e-6)
    image = Blurred(GaussianDisk(3.0), fwhm).render(128, 80.0)
    wider = GaussianDisk(5.0).render(128, 80.0)
    assert np.allclose(image, wider, atol=2e-4 * float(wider.max()))


def test_blurred_composes_with_system_and_fits():
    def loss(fwhm):
        scene = System(ring=Blurred(Crescent(20.0, 7.0, 0.5, 0.5), fwhm))
        return np.sum(np.abs(scene.model(U, V, WAVEL)) ** 2)

    assert onp.isfinite(jax.grad(loss)(3.0))
    # The wrapper is transparent to the zero-spacing flux.
    assert np.allclose(
        Blurred(Crescent(20.0, 7.0), 3.0).model(U, V, WAVEL)[0], 1.0
    )


def test_blurred_smearing_matches_wider_gaussian():
    fwhm = 4.0 * 2.0 * onp.sqrt(2.0 * onp.log(2.0))
    blurred = System(a=Blurred(GaussianDisk(3.0, dra=5.0), fwhm))
    wider = System(a=GaussianDisk(5.0, dra=5.0))
    assert np.allclose(
        smeared_model(blurred, U, V, WAVEL, 1e-7),
        smeared_model(wider, U, V, WAVEL, 1e-7),
        atol=1e-6,
    )


def test_wrappers_are_seen_by_scene_walkers():
    disk = MultipoleDisk(GaussianRing(5.0, 1.0))
    found = list(_multipole_disks(System(d=Blurred(disk, 2.0))))
    assert [path for path, _ in found] == ["d.source."]


def test_blurred_image_with_its_own_blur_warns():
    image = Image(np.ones((8, 8)), 1.0, blur_fwhm_mas=1.0)
    with pytest.warns(UserWarning, match="compound"):
        Blurred(image, 2.0)
