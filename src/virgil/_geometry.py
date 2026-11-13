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


def safe_azimuth(x, y):
    """Azimuth ``arctan2(x, y)`` (East of North), finite with finite gradients at the origin."""
    origin = (x == 0.0) & (y == 0.0)
    return np.arctan2(x, np.where(origin, 1.0, y))


def safe_radius(x, y):
    """Radius ``hypot(x, y)``, finite with finite gradients at the origin (where it is 0, with zero gradient)."""
    origin = (x == 0.0) & (y == 0.0)
    return np.where(origin, 0.0, np.hypot(np.where(origin, 1.0, x), y))


def pixel_offsets(npix, pixel_scale_mas):
    """Sky offsets (mas) of the pixel centres along one image axis.

    ``(centre - index) * pixel_scale_mas``, centre at ``(npix - 1) / 2``: ``dra``
    along columns (East left), ``ddec`` along rows (North up).
    """
    return (0.5 * (npix - 1) - np.arange(npix)) * pixel_scale_mas


def image_visibilities(brightness, uu, vv, pixel_scale_mas, ft="auto"):
    """Fourier transform of a pixel image at arbitrary frequencies.

    Each pixel is a point at its centre: ``V = Σ I[row, col] exp(-2πi (u x_col
    + v y_row))``, the sign convention of :func:`offset_phase`, done at
    ``Precision.HIGHEST`` (the A100/H100 default, TF32, is ~1e-3 accurate).
    On a regular uv grid :func:`grid_visibilities` is faster. Large problems
    use the optional ``nufftax`` transform (see ``ft``), accurate to a
    requested tolerance rather than exactly.

    Parameters
    ----------
    brightness : array-like, shape (nrow, ncol)
        Pixel fluxes in the orientation of :func:`pixel_offsets`. Not
        normalized here.
    uu, vv : array-like
        Spatial frequencies, baseline / wavelength (per radian), of
        broadcastable shapes.
    pixel_scale_mas : float
        Pixel size in milliarcseconds.
    ft : {"auto", "dft", "nufft"}
        ``"dft"`` is the exact sum; ``"nufft"`` a non-uniform FFT; ``"auto"``
        follows :func:`set_ft_defaults` (by default the faster for the size).

    Returns
    -------
    array-like
        Complex visibilities with the broadcast shape of ``uu`` and ``vv``.
    """
    brightness = np.asarray(brightness)
    uu, vv = np.broadcast_arrays(uu, vv)
    shape = np.shape(uu)
    uu, vv = np.ravel(uu), np.ravel(vv)
    if ft not in ("auto", "dft", "nufft"):
        raise ValueError(f"ft must be 'auto', 'dft' or 'nufft', not {ft!r}.")
    if ft == "auto":
        ft = "nufft" if _use_nufft(brightness.shape, uu.size) else "dft"
    if ft == "nufft":
        vis = _nufft(brightness, uu, vv, pixel_scale_mas)
    else:
        vis = _dft(brightness, uu, vv, pixel_scale_mas)
    return vis.reshape(shape)


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


# virgil#75: nufftax (pure JAX) against the DFT, value+grad, f32; both rules
# lean towards the DFT. GPU (A100; jobs 17776437, 18762629, 18819744;
# ozstar_scripts#35): nufftax has a floor of 0.15-0.23 ms up to 512² and
# 0.52 ms at 1024², and wins from ~3000 points at 128² and up; 64² never
# really wins. CPU (Apple M1 Max, 2026-10-19 scans): the crossover is ~60
# points at 64² and ~1600 at 1024², about 2 N, and the fixed cost varied 2-4x.
# eps is a requested tolerance relative to the zero-baseline visibility. 1e-5
# in f32 is the f32 floor: as accurate as the f32 DFT (7e-6 to 9e-5), and
# 1.2-1.5x faster than 1e-6.
# toeplitz_*: chi2="auto". CPU f64 scan, 4 ddids, direct on the NUFFT (2026-10-23): 0.6-1 s
# precompute; per call 0.1-2 ms against 0.6-700 ms, paid back in <20 calls from 1 point
# per pixel and ddid (IM Lup, A100: 1.5; 6.7 s, 2.4 GB, 21.8 ms against 735 ms).
_FT_DEFAULTS = {
    "ft": "auto",
    "cpu_min_points": 500,
    "cpu_points_per_side": 2,
    "cpu_min_side": 64,
    "gpu_min_points": 3000,
    "gpu_min_side": 128,
    "eps_f32": 1e-5,
    "eps_f64": 1e-12,
    "toeplitz_min_points": 50_000,
    "toeplitz_points_per_pixel": 1,
    "toeplitz_max_bytes": 4e9,
}


def set_ft_defaults(
    ft=None,
    cpu_min_points=None,
    cpu_points_per_side=None,
    cpu_min_side=None,
    gpu_min_points=None,
    gpu_min_side=None,
    eps_f32=None,
    eps_f64=None,
    toeplitz_min_points=None,
    toeplitz_points_per_pixel=None,
    toeplitz_max_bytes=None,
):
    """Set how ``ft="auto"`` chooses the Fourier transform of pixel images.

    Only the arguments you pass change. On a CPU the NUFFT is used when
    there are at least ``max(cpu_min_points, cpu_points_per_side * N)`` points,
    N the image's larger side, and its smaller side is at least
    ``cpu_min_side``; on a GPU when there are at least ``gpu_min_points``
    points and the smaller side is at least ``gpu_min_side``. ``eps_f32`` and
    ``eps_f64`` are its tolerances, relative to the zero-baseline visibility.

    ``chi2="auto"`` radio terms use the dirty-beam form when they have at least
    ``max(toeplitz_min_points, toeplitz_points_per_pixel * n_ddid * N²)``
    points, N the image side ([`ComplexVisData.term`][virgil.radio.ComplexVisData.term]).

    The settings are read when a function is traced: a jitted function that
    has already compiled keeps its old choice. The ``toeplitz_*`` ones are read
    at each ``fit``.

    Parameters
    ----------
    ft : {"auto", "dft", "nufft"}, optional
        What models with ``ft="auto"`` use: ``"dft"`` always gives the exact
        sum, ``"nufft"`` always the NUFFT, ``"auto"`` chooses by size.
    cpu_min_points, cpu_points_per_side, cpu_min_side : float, optional
        The CPU thresholds above.
    gpu_min_points, gpu_min_side : float, optional
        The GPU thresholds above.
    eps_f32, eps_f64 : float, optional
        NUFFT tolerances for float32 and float64 images.
    toeplitz_min_points, toeplitz_points_per_pixel, toeplitz_max_bytes : float, optional
        The size thresholds of ``chi2="auto"`` above; ``toeplitz_max_bytes``
        caps the beams' device memory (about 24 N² n_ddid bytes).

    Returns
    -------
    dict
        The previous settings: ``set_ft_defaults(**old)`` restores them.
    """
    new = dict(
        ft=ft,
        cpu_min_points=cpu_min_points,
        cpu_points_per_side=cpu_points_per_side,
        cpu_min_side=cpu_min_side,
        gpu_min_points=gpu_min_points,
        gpu_min_side=gpu_min_side,
        eps_f32=eps_f32,
        eps_f64=eps_f64,
        toeplitz_min_points=toeplitz_min_points,
        toeplitz_points_per_pixel=toeplitz_points_per_pixel,
        toeplitz_max_bytes=toeplitz_max_bytes,
    )
    if ft is not None and ft not in _FT_CHOICES:
        raise ValueError(f"ft must be 'auto', 'dft' or 'nufft', not {ft!r}.")
    old = dict(_FT_DEFAULTS)
    _FT_DEFAULTS.update({k: v for k, v in new.items() if v is not None})
    return old


_FT_CHOICES = ("auto", "dft", "nufft")


def _use_nufft(shape, n_points):
    """Whether ``ft="auto"`` uses the NUFFT for this size and the defaults."""
    nrow, ncol = shape
    settings = _FT_DEFAULTS
    if settings["ft"] != "auto":
        return settings["ft"] == "nufft"
    if jax.default_backend() == "gpu":
        return (
            min(nrow, ncol) >= settings["gpu_min_side"]
            and n_points >= settings["gpu_min_points"]
        )
    return min(nrow, ncol) >= settings["cpu_min_side"] and n_points >= max(
        settings["cpu_min_points"],
        settings["cpu_points_per_side"] * max(nrow, ncol),
    )


def _nufft(brightness, uu, vv, pixel_scale_mas, eps=None):
    """:func:`_dft` by nufftax's type-2 transform, to relative tolerance eps."""
    from nufftax import nufft2d2

    dtype = np.result_type(brightness.dtype, np.float32)
    if eps is None:
        eps = _FT_DEFAULTS["eps_f64" if dtype == np.float64 else "eps_f32"]
    nrow, ncol = brightness.shape
    # x pairs with the last axis. V = Σ f[k] exp(+i k t) exp(+i t (1 - n%2)/2) over modes k = index - n//2,
    # t = 2π s f (rad per pixel): the DFT's centre is (n - 1)/2, not n//2.
    t_row = 2.0 * np.pi * mas2rad * pixel_scale_mas * vv.astype(dtype)
    t_col = 2.0 * np.pi * mas2rad * pixel_scale_mas * uu.astype(dtype)
    # nufftax wraps t modulo 2π, but the half-pixel phase on an even axis is
    # 4π-periodic: transform the wrapped t and apply that phase to the original.
    wrap = lambda t: t - 2.0 * np.pi * np.round(t / (2.0 * np.pi))  # noqa: E731
    source = brightness.astype(np.result_type(dtype, np.complex64))
    vis = nufft2d2(wrap(t_col), wrap(t_row), source, eps=eps, isign=1)
    shift = 0.5 * ((1 - nrow % 2) * t_row + (1 - ncol % 2) * t_col)
    return vis * np.exp(1j * shift)


def grid_visibilities(brightness, uu_axis, vv_axis, pixel_scale_mas):
    """Exact Fourier transform of a pixel image onto a regular uv grid.

    The two-sided matrix Fourier transform (Soummer et al. 2007),
    ``V = A_v @ I @ A_u.T``: the sum of :func:`image_visibilities`, but
    O(``nv·N² + nv·nu·N``) rather than O(``nv·nu·N²``).

    Parameters
    ----------
    brightness : array-like, shape (nrow, ncol)
        Pixel fluxes, oriented as in :func:`pixel_offsets`.
    uu_axis, vv_axis : array-like, shapes (nu,) and (nv,)
        The grid's spatial frequencies along each image axis (per radian).
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
    """Sky coordinates of ``(x, y)`` in a frame whose "up" is at position angle ``rotation_deg``.

    The frame's ``(0, 1)`` lands at ``(sin θ, cos θ)``; ``-rotation_deg``
    inverts it, and the same matrix maps frequencies. NumPy-like inputs are
    rotated in NumPy at their own precision, JAX arrays and tracers in JAX.
    """
    xp = onp if _numpy_inputs(x, y, rotation_deg) else np
    if xp is onp:
        x, y, rotation_deg = (
            a
            if isinstance(a, (int, float, onp.ndarray, onp.generic))
            else onp.asarray(a)
            for a in (x, y, rotation_deg)
        )
    c, s = xp.cos(rotation_deg * dtor), xp.sin(rotation_deg * dtor)
    return c * x + s * y, -s * x + c * y


def _is_numpy_like(a):
    """Whether ``a`` is a number, NumPy, or array-like, but not JAX."""
    if isinstance(a, jax.Array):  # includes tracers
        return False
    if isinstance(a, (int, float, onp.ndarray, onp.generic)):
        return True
    if isinstance(a, (list, tuple)):
        return all(_is_numpy_like(b) for b in a)
    return hasattr(a, "__array__")


def _numpy_inputs(*values):
    """Whether every value is NumPy-like (numbers, lists, ``__array__``)."""
    return all(_is_numpy_like(a) for a in values)


def position_angle(dra, ddec):
    """Position angle in degrees, in [0, 360), of the offset ``(dra, ddec)``.

    ``dra`` is East and ``ddec`` North (any common unit); ``(0, 1)`` is 0°
    and ``(1, 0)`` is 90°. NumPy-like inputs give NumPy, otherwise JAX.
    """
    xp = onp if _numpy_inputs(dra, ddec) else np
    if xp is np:
        # arctan2 has a NaN gradient at the origin, where the angle is
        # undefined: give it a harmless point there (0° either way), so that
        # a sampler or optimizer that starts on the origin sees finite
        # derivatives. The angle at the origin (and at signed zeros, which
        # compare equal to 0) is 0°; NumPy's arctan2 still tells them apart.
        origin = (dra == 0) & (ddec == 0)
        dra, ddec = np.where(origin, 0.0, dra), np.where(origin, 1.0, ddec)
    pa = xp.mod(xp.rad2deg(xp.arctan2(dra, ddec)), 360.0)
    # mod rounds a tiny negative angle up to exactly 360. [()] turns the
    # 0-d array NumPy's where makes of a scalar back into a scalar.
    return xp.where(pa >= 360.0, pa - 360.0, pa)[()]


def separation_pa(dra, ddec):
    """``(sep, pa)`` of the offset ``(dra, ddec)``; ``pa`` as in :func:`position_angle`."""
    if _numpy_inputs(dra, ddec):
        return onp.hypot(dra, ddec), position_angle(dra, ddec)
    # hypot's gradient at the origin is NaN (JAX 0.12 dropped the 0 that
    # earlier releases returned), so the separation, like the angle, takes
    # a harmless point there and is 0 with a zero gradient.
    origin = (dra == 0) & (ddec == 0)
    sep = np.hypot(np.where(origin, 1.0, dra), np.where(origin, 0.0, ddec))
    return np.where(origin, 0.0, sep), position_angle(dra, ddec)


def _unflagged_samples(data):
    """Baseline, wavelength and channel width (flat arrays, metres) of the unflagged samples.

    ``data`` is an ``OIData`` or a sequence. Samples that no observable uses
    (``OIData.unflagged``) and zero baselines are dropped; the width is NaN
    where the data carry none. Raises if no sample remains.
    """
    observations = data if isinstance(data, (list, tuple)) else [data]
    baseline, wavel, width = [], [], []
    for d in observations:
        u = onp.ravel(onp.asarray(d.u, float))
        v = onp.ravel(onp.asarray(d.v, float))
        shape = u.shape
        lam = onp.broadcast_to(onp.ravel(onp.asarray(d.wavel, float)), shape)
        bandwidth = getattr(d, "bandwidth", None)
        if bandwidth is None:
            dlam = onp.full(shape, onp.nan)
        else:
            dlam = onp.broadcast_to(
                onp.ravel(onp.asarray(bandwidth, float)), shape
            )
        unflagged = getattr(d, "unflagged", None)
        keep = (
            onp.ones(shape, bool)
            if unflagged is None
            else onp.asarray(unflagged, bool)
        )
        b = onp.hypot(u, v)
        keep = keep & (b > 0)
        baseline.append(b[keep]), wavel.append(lam[keep])
        width.append(dlam[keep])
    baseline, wavel, width = (
        onp.concatenate(x) for x in (baseline, wavel, width)
    )
    if baseline.size == 0:
        raise ValueError(
            "The data have no unflagged sample with a non-zero baseline."
        )
    return baseline, wavel, width


# Largest multiple of a dual-lattice vector searched for the shortest alias.
_ALIAS_ORDER = 6


def fringe_scales(data):
    """The finest and coarsest fringe spacings of ``data``, in mas.

    ``data`` is an ``OIData`` or a sequence. With each sample's own
    ``B/λ``, these are ``1 / max(B/λ)`` and ``1 / min(B/λ)`` over the
    unflagged samples with a non-zero baseline.

    Returns
    -------
    tuple[float, float]
        ``(finest, coarsest)``.
    """
    baseline, wavel, _ = _unflagged_samples(data)
    rho = baseline / wavel
    return 1.0 / (rho.max() * mas2rad), 1.0 / (rho.min() * mas2rad)


def lattice_alias_limit(data, tolerance=1e-3):
    """Half-width (mas) of the largest square search box free of alias copies, or ``None``.

    If the unflagged uv samples of each wavelength lie on a 2-D lattice (aperture masks,
    DISCO products), the visibilities repeat under the reciprocal lattice
    (period ``λ`` times the dual lattice, in the sky frame), so two
    positions in one box can look alike. A box of half-width ``h`` has no such
    pair when ``2h`` is below every alias vector's larger component, so the
    limit is half the smallest of those. With several channels a position is
    ambiguous only if it aliases in all of them, and every such shift is an
    alias of the longest channel, so that channel's limit is used (a lower
    bound on the joint one). ``None`` for other coverage (samples off a
    lattice to within ``tolerance`` cells in any unflagged channel).

    Parameters
    ----------
    data : OIData
    tolerance : float, optional
        Largest allowed offset from a lattice point, in cells.
    """
    shape = onp.ravel(onp.asarray(data.u, float)).shape
    uv = onp.column_stack(
        [onp.ravel(onp.asarray(x, float)) for x in (data.u, data.v)]
    )
    wavel = onp.broadcast_to(onp.ravel(onp.asarray(data.wavel, float)), shape)
    unflagged = getattr(data, "unflagged", None)
    if unflagged is not None:  # a flagged sample measures nothing
        keep = onp.ravel(onp.asarray(unflagged, bool))
        uv, wavel = uv[keep], wavel[keep]
    limits = [
        _channel_alias_limit(uv[wavel == lam], lam, tolerance)
        for lam in onp.unique(wavel)
    ]
    if not limits or None in limits:
        return None
    # A joint alias is an alias of the longest unflagged wavelength: it binds.
    return max(limits)


def _channel_alias_limit(uv, wavel, tolerance):
    """``lattice_alias_limit`` of the samples ``uv`` of one wavelength."""
    uv = onp.unique(onp.round(uv, 9), axis=0)
    norm = onp.hypot(*uv.T)
    uv, norm = uv[norm > 0], norm[norm > 0]
    if len(uv) < 2:
        return None
    first = uv[onp.argmin(norm)]
    cross = onp.abs(first[0] * uv[:, 1] - first[1] * uv[:, 0])
    apart = cross > 1e-6 * onp.linalg.norm(first) * norm
    if not apart.any():
        return None
    basis = onp.array([first, uv[apart][onp.argmin(norm[apart])]])
    cells = uv @ onp.linalg.inv(basis)
    if onp.max(onp.abs(cells - onp.round(cells))) > tolerance:
        return None
    # The alias vectors are the reciprocal lattice, scaled by λ (rad -> mas).
    dual = onp.linalg.inv(basis).T * wavel / mas2rad
    steps = onp.arange(-_ALIAS_ORDER, _ALIAS_ORDER + 1)
    mix = onp.array([(i, j) for i in steps for j in steps if i or j])
    return float(0.5 * onp.abs(mix @ dual).max(axis=1).min())


def smearing_scale(data):
    """The field (mas) beyond which bandwidth smearing washes out a source.

    The smallest ``λ² / (Δλ B)`` (first null of the channel's sinc) over the
    unflagged samples with a non-zero baseline: an order-of-magnitude limit
    (the visibility is down 36% at half this offset). Infinite when the
    data carry no channel widths (``OIData.with_bandwidth``).
    """
    baseline, wavel, width = _unflagged_samples(data)
    known = onp.isfinite(width) & (width > 0)
    if not known.any():
        return onp.inf
    scale = wavel[known] ** 2 / (width[known] * baseline[known])
    return float(scale.min() / mas2rad)


class UVGrid(eqx.Module):
    """uv samples on a regular lattice, possibly rotated on the sky (see :func:`find_uv_grid`).

    Sample ``k`` is at grid-frame ``(u_axis[index[k] % nu], v_axis[index[k] // nu])``
    (metres), and on the sky at ``rotate(u_g, v_g, rotation_deg)``.
    """

    u_axis: jax.Array
    v_axis: jax.Array
    index: jax.Array
    rotation_deg: float = eqx.field(static=True)


def find_uv_grid(u, v, tolerance=1e-6):
    """Return the lattice that ``(u, v)`` samples lie on, or ``None``.

    A first guess at the axes comes from the shortest sample separation;
    the origin, pitches and rotation are then fitted by least squares to
    every sample, so that rounding of the coordinates (float32 products)
    does not accumulate across the lattice. Every sample must lie within
    ``tolerance`` cells of a point of the fitted lattice.

    Parameters
    ----------
    u, v : array-like
        Sample coordinates (concrete arrays, e.g. in metres).
    tolerance : float, optional
        Largest allowed offset from a lattice point, in cells.

    Returns
    -------
    UVGrid or None
        ``rotation_deg`` is in ``(-45, 45]`` (up to the fitted correction).
    """
    u, v = onp.asarray(u, dtype=float), onp.asarray(v, dtype=float)
    if u.ndim != 1 or u.size < 2:
        return None
    gap = onp.hypot(u - u[0], v - v[0])
    gap[0] = onp.inf
    nearest = int(onp.argmin(gap))
    if not gap[nearest] > 0.0:
        return None  # repeated samples: no lattice index is unique
    angle = onp.degrees(onp.arctan2(v[nearest] - v[0], u[nearest] - u[0]))
    rotation = float(-(((angle + 45.0) % 90.0) - 45.0))
    # First guess: cells along each axis of the rotated frame, which only
    # need to be the right integers (within a quarter of a cell).
    grid = rotate(u, v, -rotation)
    cells, pitches = [], []
    for coord in grid:
        steps = onp.diff(onp.sort(coord))
        steps = steps[steps > max(tolerance, 1e-3) * gap[nearest]]
        pitch = steps.min() if steps.size else 1.0
        cell = (coord - coord.min()) / pitch
        if onp.max(onp.abs(cell - onp.round(cell))) > 0.25:
            return None
        cells.append(onp.round(cell))
        pitches.append(pitch)
    # Least-squares lattice: each frame coordinate is affine in the cells.
    # The cross terms measure a residual rotation of each axis.
    design = onp.column_stack([onp.ones_like(u), *cells])
    (_, uc, ur), (_, vc, vr) = (
        onp.linalg.lstsq(design, coord, rcond=None)[0] for coord in grid
    )
    spans = [onp.std(cell) for cell in cells]
    turns = (onp.arctan2(-vc, uc), onp.arctan2(ur, vr))
    weights = [
        (onp.hypot(*e) * span) ** 2 if span > 0 else 0.0
        for e, span in zip(((uc, vc), (ur, vr)), spans)
    ]
    rotation += float(onp.degrees(onp.average(turns, weights=weights)))
    grid = rotate(u, v, -rotation)
    axes = []
    for coord, cell, span, e, pitch in zip(
        grid, cells, spans, ((uc, vc), (ur, vr)), pitches
    ):
        pitch = onp.hypot(*e) if span > 0 else pitch
        start = onp.mean(coord - pitch * cell)
        if onp.max(onp.abs((coord - start) / pitch - cell)) > tolerance:
            return None
        axes.append(start + pitch * onp.arange(int(cell.max()) + 1))
    cols, rows = (cell.astype(int) for cell in cells)
    index = rows * axes[0].size + cols
    if onp.unique(index).size != index.size:
        return None
    return UVGrid(
        np.asarray(axes[0]), np.asarray(axes[1]), np.asarray(index), rotation
    )


# Largest offset of a DISCO sample from its fitted lattice point, in cells:
# drpangloss-format products store (u, v) in float32, which rounds them by a
# few 1e-6 cells across a lattice tens of cells wide.
_DISCO_GRID_TOLERANCE = 1e-5


def _disco_uv_grid(u, v, wavel):
    """The lattice of one filter of an AMIGO DISCO product, or ``None``.

    AMIGO builds its DISCO products on a regular uv lattice, so the samples
    of one filter (one wavelength) lie on one, to within
    ``_DISCO_GRID_TOLERANCE`` cells. ``None`` for several wavelengths, for
    samples off a lattice, and for a lattice mostly empty (more than four
    cells per sample), whose transform could cost more than the per-point
    sum it replaces.
    """
    if onp.size(wavel) != 1:
        return None
    grid = find_uv_grid(u, v, tolerance=_DISCO_GRID_TOLERANCE)
    if grid is None or grid.u_axis.size * grid.v_axis.size > 4 * onp.size(u):
        return None
    return grid


def offset_phase(uu, vv, dra, ddec):
    """Fourier shift factor for an offset of ``(dra, ddec)`` milliarcseconds."""
    arg = 2.0 * np.pi * mas2rad * (uu * dra + vv * ddec)
    return jax.lax.complex(np.cos(arg), -np.sin(arg))


def undo_elliptical_transf_spat_freq(u, v, pa, stretch):
    """Map ``(u, v)`` to the frame where the elliptical object is circular, major axis North.

    Parameters
    ----------
    u, v : array-like
        Baseline coordinates in wavelength units.
    pa : float or array-like
        Position angle of the projected major axis (degrees, North to East).
    stretch : float or array-like
        Factor (typically < 1) applied to the minor axis.

    Returns
    -------
    tuple[array-like, array-like]
        ``(u, v)`` with the rotation and stretch undone.
    """
    pa_rad = pa * dtor

    # Fourier similarity theorem: compressing the image by `stretch` dilates uv by 1/stretch.
    ut = (u * np.cos(pa_rad) - v * np.sin(pa_rad)) * stretch
    vt = u * np.sin(pa_rad) + v * np.cos(pa_rad)

    return ut, vt


def undo_elliptical_transf_coord(x, y, pa, stretch):
    """Map ``(x, y)`` (mas, East and North) to the frame where the elliptical object is circular, major axis North.

    ``pa`` (degrees, North to East) and ``stretch`` (typically < 1, applied
    to the minor axis) are as in :func:`undo_elliptical_transf_spat_freq`.
    Returns ``(x, y)`` in mas with the rotation and stretch undone.
    """
    pa_rad = pa * dtor

    xt = (x * np.cos(pa_rad) - y * np.sin(pa_rad)) / stretch
    yt = x * np.sin(pa_rad) + y * np.cos(pa_rad)

    return xt, yt


def apply_elliptical_transf_coord(x, y, pa, stretch):
    """Inverse of :func:`undo_elliptical_transf_coord`; ``(x, y)`` in mas."""
    pa_rad = pa * dtor

    x_scaled = x * stretch
    y_scaled = y

    xt = x_scaled * np.cos(pa_rad) + y_scaled * np.sin(pa_rad)
    yt = -x_scaled * np.sin(pa_rad) + y_scaled * np.cos(pa_rad)

    return xt, yt


def check_az_prof_nonnegative(az_amps, az_pas, tol=1e-6):
    r"""Check that an azimuthal brightness modulation stays non-negative.

    The modulation is $1 + f(\theta)$, $f(\theta) = \sum_{m=1}^{n} A_m
    \cos(m(\theta - \mathrm{pa}_m))$. Its minimum is the lowest of $64 n$
    grid values polished by Newton steps, so it is never underestimated.
    ``jit``-safe.

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


def gaussian_taper(q2, fwhm_mas):
    r"""Fourier transform of a unit-flux Gaussian of FWHM ``fwhm_mas``.

    ``exp(-π² θ² q² / (4 ln 2))``, with θ the FWHM in radians and ``q2`` the
    squared baseline length in wavelengths. Exact at any uv point, so it is
    both the nuisance taper and the visibility factor of a smoothed image.
    """
    theta = fwhm_mas * mas2rad
    return np.exp(-((np.pi * theta) ** 2) * q2 / (4.0 * onp.log(2.0)))


def gaussian_smooth(brightness, fwhm_pix):
    """Convolve pixels with a Gaussian of FWHM ``fwhm_pix`` pixels, by FFT.

    The kernel is the Gaussian sampled at the pixel centres and normalized to
    unit sum, and the image is zero-padded to twice its size so the
    convolution does not wrap around. The result is cropped to the input
    shape, so flux that spreads beyond the edges is lost. Its DFT is the
    input's times the analytic Gaussian (see :func:`gaussian_taper`) plus
    aliases of order ``exp(-π² θ² (1/p - q)² / (4 ln 2))`` relative to the
    total flux, which is below 1e-6 for ``q <= 0.25 / p`` once the FWHM is at
    least 3 pixels, and large for narrower kernels.
    """
    nrow, ncol = brightness.shape
    shape = (2 * nrow, 2 * ncol)
    sigma2 = (fwhm_pix / (2.0 * onp.sqrt(2.0 * onp.log(2.0)))) ** 2

    def kernel(n):
        i = np.arange(n)
        d = np.minimum(i, n - i)  # circular distance from index 0
        # Zero width is the delta kernel; the double where keeps its gradient finite.
        wide = sigma2 > 0.0
        k = np.where(
            wide,
            np.exp(-0.5 * d**2 / np.where(wide, sigma2, 1.0)),
            (d == 0).astype(float),
        )
        return k / np.sum(k)

    kr, kc = kernel(shape[0]), kernel(shape[1])
    ft = np.fft.rfft2(brightness, s=shape)
    kft = np.fft.rfft(kc)[None, :] * np.fft.fft(kr)[:, None]
    out = np.fft.irfft2(ft * kft, s=shape)
    return out[:nrow, :ncol]
