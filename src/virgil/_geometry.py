"""Sky-plane and uv-plane geometry shared by the source models.

Position angles follow the on-sky convention used throughout virgil:
North is +y (up), East is +x (left, i.e. x decreases left-to-right across
an image array's columns), and position angle is measured North to East,
i.e. counter-clockwise from the top in a plot with East to the left. This
matches the coordinate grid produced by :func:`image_coordinates`.
"""

import equinox as eqx
import jax
import jax.numpy as np
import numpy as onp

from ._utils import dtor, mas2rad


def image_coordinates(npix, fov_mas):
    """Return Cartesian image-plane coordinates in milliarcseconds."""
    npix = int(npix)
    pixel_scale_mas = float(fov_mas) / npix
    pixel_indices = np.arange(npix)
    center = 0.5 * (npix - 1)
    x = -(pixel_indices - center) * pixel_scale_mas
    y = (center - pixel_indices) * pixel_scale_mas
    return np.meshgrid(x, y, indexing="xy")


def pixel_offsets(npix, pixel_scale_mas):
    """Sky offsets (mas) of the pixel centres along one image axis.

    Along columns this is ``dra``, decreasing to the right (East left); along
    rows it is ``ddec``, decreasing downwards (North up). Both are
    ``(centre - index) * pixel_scale_mas`` with the centre at index
    ``(npix - 1) / 2``, as in :func:`image_coordinates`.
    """
    return (0.5 * (npix - 1) - np.arange(npix)) * pixel_scale_mas


def image_visibilities(brightness, uu, vv, pixel_scale_mas):
    """Exact Fourier transform of a pixel image at arbitrary frequencies.

    Each pixel is treated as a point at its centre, so
    ``V(u, v) = sum_{row, col} I[row, col] exp(-2πi (u x_col + v y_row))``
    with the sign convention of :func:`offset_phase`. The sum is separable
    into two matrix products, done at ``Precision.HIGHEST`` (on A100/H100
    GPUs the default is TF32, with ~1e-3 relative error). On a regular uv
    grid, :func:`grid_visibilities` does the same sum faster.

    Parameters
    ----------
    brightness : array-like, shape (nrow, ncol)
        Pixel fluxes in the orientation of :func:`pixel_offsets` (East left,
        North up). Not normalised here.
    uu, vv : array-like
        Spatial frequencies, baseline / wavelength (per radian), of
        broadcastable shapes.
    pixel_scale_mas : float
        Pixel size in milliarcseconds.

    Returns
    -------
    array-like
        Complex visibilities with the broadcast shape of ``uu`` and ``vv``.
    """
    brightness = np.asarray(brightness)
    uu, vv = np.broadcast_arrays(uu, vv)
    vis = _dft(brightness, np.ravel(uu), np.ravel(vv), pixel_scale_mas)
    return vis.reshape(np.shape(uu))


def _fourier_matrix(freq_per_rad, npix, pixel_scale_mas):
    """``exp(-2πi f x)`` for frequencies ``f`` and one axis's pixel offsets."""
    offsets = pixel_offsets(npix, pixel_scale_mas)
    return np.exp(-2j * np.pi * np.outer(mas2rad * freq_per_rad, offsets))


def _dft(brightness, uu, vv, pixel_scale_mas):
    nrow, ncol = brightness.shape
    cols = _fourier_matrix(uu, ncol, pixel_scale_mas)
    rows = _fourier_matrix(vv, nrow, pixel_scale_mas)
    highest = jax.lax.Precision.HIGHEST
    partial = np.matmul(rows, brightness.astype(rows.dtype), precision=highest)
    return np.sum(partial * cols, axis=-1)


def grid_visibilities(brightness, uu_axis, vv_axis, pixel_scale_mas):
    """Exact Fourier transform of a pixel image onto a regular uv grid.

    The two-sided matrix Fourier transform (Soummer et al. 2007,
    arXiv:0711.0368): ``V = A_v @ I @ A_u.T``, the same sum as
    :func:`image_visibilities` but costing ``nv·N² + nv·nu·N`` rather than
    ``nv·nu·N²`` for an N x N image. Done at ``Precision.HIGHEST``.

    Parameters
    ----------
    brightness : array-like, shape (nrow, ncol)
        Pixel fluxes, oriented as in :func:`pixel_offsets`.
    uu_axis, vv_axis : array-like, shapes (nu,) and (nv,)
        The grid's spatial frequencies along each image axis, baseline /
        wavelength (per radian).
    pixel_scale_mas : float
        Pixel size in milliarcseconds.

    Returns
    -------
    array-like, shape (nv, nu)
        ``V[j, i]`` is the visibility at ``(uu_axis[i], vv_axis[j])``.
    """
    brightness = np.asarray(brightness)
    nrow, ncol = brightness.shape
    cols = _fourier_matrix(np.ravel(uu_axis), ncol, pixel_scale_mas)
    rows = _fourier_matrix(np.ravel(vv_axis), nrow, pixel_scale_mas)
    highest = jax.lax.Precision.HIGHEST
    partial = np.matmul(rows, brightness.astype(rows.dtype), precision=highest)
    return np.matmul(partial, cols.T, precision=highest)


def rotate(x, y, rotation_deg):
    """Rotate a frame by a position angle, North towards East.

    Returns the sky coordinates of the point ``(x, y)`` of a frame whose
    "up" axis points at position angle ``rotation_deg``: the frame's
    ``(0, 1)`` lands at ``(sin θ, cos θ)``. The same matrix maps
    frequencies, and ``rotate(..., -rotation_deg)`` inverts it.
    """
    c, s = np.cos(rotation_deg * dtor), np.sin(rotation_deg * dtor)
    return c * x + s * y, -s * x + c * y


def fringe_scales(data):
    """The finest and coarsest fringe spacings of ``data``, in mas.

    ``data`` is an ``OIData`` or a sequence of them. Each sample ``k`` has
    a spatial frequency ``B_k / λ_k`` (its own baseline over its own
    wavelength), and the finest fringes are ``1 / max(B/λ)`` and the
    coarsest ``1 / min(B/λ)``, over the samples with a non-zero baseline
    (a zero baseline measures no fringe, and would make the coarsest
    infinite). This is not ``λ_max / B_min``, which pairs a baseline with a
    wavelength it may not have been observed at.

    Returns
    -------
    tuple[float, float]
        ``(finest, coarsest)``.
    """
    observations = data if isinstance(data, (list, tuple)) else [data]
    rho = onp.concatenate(
        [
            onp.ravel(
                onp.hypot(onp.asarray(d.u, float), onp.asarray(d.v, float))
                / onp.asarray(d.wavel, float)
            )
            for d in observations
        ]
    )
    rho = rho[rho > 0]
    if rho.size == 0:
        raise ValueError("The data have no sample with a non-zero baseline.")
    return 1.0 / (rho.max() * mas2rad), 1.0 / (rho.min() * mas2rad)


class UVGrid(eqx.Module):
    """uv samples that lie on a regular lattice, possibly rotated on the sky.

    AMI data, for example, are sampled on the detector's Fourier grid,
    which is rotated on the sky by the parallactic angle. The sample ``k``
    is at grid-frame coordinates
    ``(u_axis[index[k] % nu], v_axis[index[k] // nu])`` (metres), and on
    the sky at ``rotate(u_g, v_g, rotation_deg)``. Found from the samples by
    :func:`find_uv_grid`.
    """

    u_axis: jax.Array
    v_axis: jax.Array
    index: jax.Array
    rotation_deg: float = eqx.field(static=True)


def find_uv_grid(u, v, tolerance=1e-6):
    """Return the lattice that ``(u, v)`` samples lie on, or ``None``.

    The lattice axes are found from the shortest separation between
    samples, and every sample must lie within ``tolerance`` of a cell of a
    lattice point (so that a Fourier transform onto the lattice is exact).

    Parameters
    ----------
    u, v : array-like
        Sample coordinates (concrete arrays, e.g. in metres).
    tolerance : float, optional
        Largest allowed offset from a lattice point, in cells.

    Returns
    -------
    UVGrid or None
        ``rotation_deg`` is in ``(-45, 45]``.
    """
    u, v = onp.asarray(u, dtype=float), onp.asarray(v, dtype=float)
    if u.ndim != 1 or u.size < 2:
        return None
    gap = onp.hypot(u - u[0], v - v[0])
    gap[0] = onp.inf
    nearest = int(onp.argmin(gap))
    angle = onp.degrees(onp.arctan2(v[nearest] - v[0], u[nearest] - u[0]))
    rotation = float(-(((angle + 45.0) % 90.0) - 45.0))
    c, s = onp.cos(onp.radians(rotation)), onp.sin(onp.radians(rotation))
    grid_u, grid_v = c * u - s * v, s * u + c * v  # rotate(u, v, -rotation)
    axes, cells = [], []
    for coord in (grid_u, grid_v):
        start = coord.min()
        steps = onp.diff(onp.sort(coord))
        steps = steps[steps > tolerance * gap[nearest]]
        pitch = steps.min() if steps.size else 1.0
        cell = (coord - start) / pitch
        if onp.max(onp.abs(cell - onp.round(cell))) > tolerance:
            return None
        cell = onp.round(cell).astype(int)
        axes.append(start + pitch * onp.arange(cell.max() + 1))
        cells.append(cell)
    index = cells[1] * axes[0].size + cells[0]
    if onp.unique(index).size != index.size:
        return None
    return UVGrid(
        np.asarray(axes[0]), np.asarray(axes[1]), np.asarray(index), rotation
    )


def offset_phase(uu, vv, dra, ddec):
    """Fourier shift factor for an offset of ``(dra, ddec)`` milliarcseconds."""
    arg = 2.0 * np.pi * mas2rad * (uu * dra + vv * ddec)
    return jax.lax.complex(np.cos(arg), -np.sin(arg))


def undo_elliptical_transf_spat_freq(u, v, pa, stretch):
    """When considering an ellipticaly rotated and stretched (along the apparent minor
    axis) object, this function takes spatial frequency coordinates, and transforms
    them to spatial frequencies in the frame of reference where an elliptical object
    appears circular and is aligned with the major axis pointing North.

    Parameters
    ----------
    u : array-like
        Baseline ``u`` coordinates in wavelength units.
    v : array-like
        Baseline ``v`` coordinates in wavelength units.
    pa : float or array-like
        Position angle of the ellipse's projected major axis in degrees, measured North
        to East (i.e. counter-clockwise in conventional astronomical image orientation)
        in the original image frame of reference.
    stretch : float or array-like
        Factor (typically $< 1.0$) by which the elliptical transform's minor axis is
        stretched in the original image frame of reference.

    Returns
    -------
    tuple[array-like, array-like]
        The $u$ and $v$ spatial frequencies, but in a frame of reference where the
        rotation and stretch of the original elliptical transformation is undone.
    """
    pa_rad = pa * dtor

    # Fourier similarity theorem: compressing the image by `stretch` dilates uv by 1/stretch.
    ut = (u * np.cos(pa_rad) - v * np.sin(pa_rad)) * stretch
    vt = u * np.sin(pa_rad) + v * np.cos(pa_rad)

    return ut, vt


def apply_elliptical_transf_spat_freq(u, v, pa, stretch):
    """Inverse of [`undo_elliptical_transf_spat_freq`][virgil._geometry.undo_elliptical_transf_spat_freq]. Takes spatial frequency
    coordinates in the frame of reference where an elliptical object appears circular
    and aligned with its major axis pointing North, and transforms them into the
    original (rotated and stretched) frame of reference.

    Parameters
    ----------
    u : array-like
        Baseline ``u`` coordinates in wavelength units, in the de-rotated/unstretched
        frame of reference.
    v : array-like
        Baseline ``v`` coordinates in wavelength units, in the de-rotated/unstretched
        frame of reference.
    pa : float or array-like
        Position angle of the ellipse's projected major axis in degrees, measured North
        to East, in the original image frame of reference.
    stretch : float or array-like
        Factor (typically $< 1.0$) by which the elliptical transform's minor axis is
        stretched in the original image frame of reference.

    Returns
    -------
    tuple[array-like, array-like]
        The $u$ and $v$ spatial frequencies in the original (rotated and stretched)
        frame of reference.
    """
    pa_rad = pa * dtor

    u_scaled = u / stretch
    v_scaled = v

    ut = u_scaled * np.cos(pa_rad) + v_scaled * np.sin(pa_rad)
    vt = -u_scaled * np.sin(pa_rad) + v_scaled * np.cos(pa_rad)

    return ut, vt


def undo_elliptical_transf_coord(x, y, pa, stretch):
    """When considering an ellipticaly rotated and stretched (along the apparent minor
    axis) object, this function takes spatial coordinates, and transforms
    them to coordinates in the frame of reference where the elliptical object
    appears circular and is aligned with the major axis pointing North.

    Parameters
    ----------
    x : array-like
        Spatial coordinates along the x-axis (East-positive) in milliarcseconds.
    y : array-like
        Spatial coordinates along the y-axis (North-positive) in milliarcseconds.
    pa : float or array-like
        Position angle of the ellipse's projected major axis in degrees, measured North
        to East (i.e. counter-clockwise in conventional astronomical image orientation)
        in the original image frame of reference.
    stretch : float or array-like
        Factor (typically $< 1.0$) by which the elliptical transform's minor axis is
        stretched in the original image frame of reference.

    Returns
    -------
    tuple[array-like, array-like]
        The $x$ and $y$ spatial coordinates, but in a frame of reference where the
        rotation and stretch of the original elliptical transformation is undone.
    """
    pa_rad = pa * dtor

    xt = (x * np.cos(pa_rad) - y * np.sin(pa_rad)) / stretch
    yt = x * np.sin(pa_rad) + y * np.cos(pa_rad)

    return xt, yt


def apply_elliptical_transf_coord(x, y, pa, stretch):
    """Inverse of [`undo_elliptical_transf_coord`][virgil._geometry.undo_elliptical_transf_coord]. Takes spatial coordinates in
    the frame of reference where an elliptical object appears circular and aligned
    with its major axis pointing North, and transforms them into the original
    (rotated and stretched) frame of reference.

    Parameters
    ----------
    x : array-like
        Spatial coordinates along the x-axis, in the de-rotated/unstretched frame
        of reference, in milliarcseconds.
    y : array-like
        Spatial coordinates along the y-axis, in the de-rotated/unstretched frame
        of reference, in milliarcseconds.
    pa : float or array-like
        Position angle of the ellipse's projected major axis in degrees, measured North
        to East, in the original image frame of reference.
    stretch : float or array-like
        Factor (typically $< 1.0$) by which the elliptical transform's minor axis is
        stretched in the original image frame of reference.

    Returns
    -------
    tuple[array-like, array-like]
        The $x$ and $y$ spatial coordinates in the original (rotated and stretched)
        frame of reference, in milliarcseconds.
    """
    pa_rad = pa * dtor

    x_scaled = x * stretch
    y_scaled = y

    xt = x_scaled * np.cos(pa_rad) + y_scaled * np.sin(pa_rad)
    yt = -x_scaled * np.sin(pa_rad) + y_scaled * np.cos(pa_rad)

    return xt, yt


def check_az_prof_nonnegative(az_amps, az_pas, tol=1e-6):
    r"""Check that an azimuthal brightness modulation stays non-negative.

    The modulation is $1 + f(\theta)$ with
    $f(\theta) = \sum_{m=1}^{n} A_m \cos(m(\theta - \mathrm{pa}_m))$. Its
    minimum is found by evaluating $f$ on $64 n$ equally spaced angles and
    polishing the lowest one with a few Newton steps on $f'$. Every value
    used is a true value of $f$, so the minimum is never underestimated;
    the grid alone overestimates it by at most
    $\tfrac12 \sum_m m^2 |A_m| \, (\pi / 64 n)^2$, and the Newton steps
    remove it at the minimum they polish. It is traceable (``jit``-safe).

    Parameters
    ----------
    az_amps : array-like
        1D array containing the azimuthal modulation order amplitudes, starting from
        order 1.
    az_pas : array-like
        1D array containing the azimuthal modulation order position angles, starting from
        order 1, in degrees.
    tol : float, optional
        Tolerance: minima down to ``-tol`` still count as non-negative.

    Returns
    -------
    bool or jax.Array
        Whether $1 + f(\theta) \geq -\mathrm{tol}$ everywhere. For array
        inputs this is a 0-d JAX boolean array; use ``bool()`` on it.
    """
    k = len(az_amps)
    if k == 0:
        # No modulation: the profile is the constant 1.
        return True
    az_amps = np.asarray(az_amps)
    orders = np.arange(1, k + 1)
    phases = np.asarray(az_pas) * dtor * orders

    def derivatives(theta):
        # f and its first two derivatives at the angles ``theta``.
        arg = orders * theta[..., None] - phases
        cos, sin = np.cos(arg), np.sin(arg)
        f = np.sum(az_amps * cos, axis=-1)
        df = -np.sum(az_amps * orders * sin, axis=-1)
        d2f = -np.sum(az_amps * orders**2 * cos, axis=-1)
        return f, df, d2f

    theta = np.linspace(0.0, 2 * np.pi, 64 * k, endpoint=False)
    f_grid = derivatives(theta)[0]
    f_min = np.min(f_grid)
    # Newton steps on f' from the lowest grid point, where f'' > 0 near a
    # minimum; a step is skipped where it is not.
    t = theta[np.argmin(f_grid)]
    for _ in range(3):
        f, df, d2f = derivatives(t)
        t = t - np.where(d2f > 0, df / np.where(d2f > 0, d2f, 1.0), 0.0)
        f_min = np.minimum(f_min, derivatives(t)[0])

    return 1.0 + f_min > -tol
