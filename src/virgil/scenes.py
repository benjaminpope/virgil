"""Synthetic truth images for testing image reconstructions.

Each function returns a unit-sum ``(npix, npix)`` array in the virgil
image orientation (East left, North up, centre at the middle of the pixel
grid, as in [`render`][virgil.models.SourceModel.render]), ready for [`Image.from_brightness`][virgil.models.Image.from_brightness].
Position angles run North to East. The ring and spiral shapes are inspired by
the training scenes of Jonah Goldfine's ``frito``
(https://github.com/JonahDG/frito), re-implemented here in virgil
conventions.
"""

import jax.numpy as np

from ._geometry import image_coordinates, undo_elliptical_transf_coord
from ._utils import _unit_sum, dtor


def ring(
    npix,
    pixel_scale_mas,
    radius_mas,
    width_mas,
    inc_deg=0.0,
    pa_deg=0.0,
    asymmetry=0.0,
    asymmetry_pa_deg=0.0,
):
    """Gaussian-profile ring, optionally inclined and azimuthally modulated.

    Parameters
    ----------
    npix : int
        Number of pixels on a side.
    pixel_scale_mas : float
        Pixel size in milliarcseconds.
    radius_mas : float
        Radius of the ring's peak in milliarcseconds (the semi-major axis
        when inclined).
    width_mas : float
        Standard deviation of the radial profile in milliarcseconds.
    inc_deg : float, optional
        Inclination in degrees; 0 is face-on, and the minor axis is
        ``cos(inc)`` times the major axis.
    pa_deg : float, optional
        Position angle of the major axis, North to East, in degrees.
    asymmetry : float, optional
        Amplitude (0 to 1) of the modulation
        ``1 + asymmetry * cos(PA - asymmetry_pa_deg)`` of the brightness
        around the ring, where PA is the on-sky position angle of a pixel.
    asymmetry_pa_deg : float, optional
        Position angle, North to East, of the brightest part of the ring, in
        degrees.

    Returns
    -------
    jax.Array, shape (npix, npix)
        The ring, summing to one.
    """
    if not abs(asymmetry) <= 1.0:
        raise ValueError("asymmetry must be between -1 and 1.")
    x, y = image_coordinates(npix, npix * pixel_scale_mas)
    xt, yt = undo_elliptical_transf_coord(x, y, pa_deg, np.cos(inc_deg * dtor))
    radial = (np.hypot(xt, yt) - radius_mas) / width_mas
    azimuth = np.arctan2(x, y) - asymmetry_pa_deg * dtor
    modulation = 1.0 + asymmetry * np.cos(azimuth)
    return _unit_sum(np.exp(-0.5 * radial**2) * modulation)


def spiral(
    npix,
    pixel_scale_mas,
    step_mas,
    width_mas,
    turns=2.0,
    pa_deg=0.0,
    fade_mas=None,
):
    """Archimedean spiral, like the dust pinwheels of WR 104 and WR 137.

    The arm is ``r = step_mas * theta / (2 pi)``, starting at the centre and
    winding towards increasing position angle as it moves outwards.

    Parameters
    ----------
    npix : int
        Number of pixels on a side.
    pixel_scale_mas : float
        Pixel size in milliarcseconds.
    step_mas : float
        Radial distance between successive turns, in milliarcseconds.
    width_mas : float
        Standard deviation of the Gaussian cross-section, measured along the
        radial direction, in milliarcseconds.
    turns : float, optional
        Number of turns of the arm, which ends in a rounded cap.
    pa_deg : float, optional
        Position angle, North to East, of the start of the arm, in degrees.
    fade_mas : float, optional
        If given, the brightness falls off as ``exp(-r / fade_mas)``.

    Returns
    -------
    jax.Array, shape (npix, npix)
        The spiral, summing to one.
    """
    x, y = image_coordinates(npix, npix * pixel_scale_mas)
    r = np.hypot(x, y)[..., None]
    # The arm crosses each position angle once per turn, at the windings
    # k = 0, 1, ...; keep the distance to the nearest of them.
    windings = 2.0 * np.pi * np.arange(int(np.ceil(turns)) + 1)
    theta = (np.arctan2(x, y) - pa_deg * dtor)[..., None] % (2.0 * np.pi)
    theta = theta + windings
    distance = r - step_mas * theta / (2.0 * np.pi)
    distance = np.where(theta <= 2.0 * np.pi * turns, distance, np.inf)
    image = np.exp(-0.5 * (distance / width_mas) ** 2).max(axis=-1)
    # Round off both ends of the arm, rather than cutting them radially.
    end_pa = pa_deg * dtor + 2.0 * np.pi * turns
    end_r = step_mas * turns
    ends = np.minimum(
        (x - end_r * np.sin(end_pa)) ** 2 + (y - end_r * np.cos(end_pa)) ** 2,
        x**2 + y**2,
    )
    image = np.maximum(image, np.exp(-0.5 * ends / width_mas**2))
    if fade_mas is not None:
        image = image * np.exp(-r[..., 0] / fade_mas)
    return _unit_sum(image)


def gaussian_blob(npix, pixel_scale_mas, sigma_mas, dra=0.0, ddec=0.0):
    """Circular Gaussian, e.g. a clump or a resolved companion.

    Parameters
    ----------
    npix : int
        Number of pixels on a side.
    pixel_scale_mas : float
        Pixel size in milliarcseconds.
    sigma_mas : float
        Standard deviation in milliarcseconds.
    dra, ddec : float, optional
        Offset of the centre from the image centre in milliarcseconds,
        positive to the East and North.

    Returns
    -------
    jax.Array, shape (npix, npix)
        The blob, summing to one.
    """
    x, y = image_coordinates(npix, npix * pixel_scale_mas)
    r2 = (x - dra) ** 2 + (y - ddec) ** 2
    return _unit_sum(np.exp(-0.5 * r2 / sigma_mas**2))
