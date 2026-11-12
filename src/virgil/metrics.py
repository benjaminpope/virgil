"""Scores for comparing a reconstructed image with a known truth.

!!! warning "Provisional"
    May change in any 0.4.x release without a deprecation cycle.

The image-recovery figures of merit used by the interferometric imaging
contests, written from their published definitions:

* the 2004 Beauty Contest, Lawson et al. (2004, Proc. SPIE 5491, 886),
  Eq. 2: the flux-weighted rms of the normalized difference, over the
  peak ([`lawson_sigma_over_peak`][virgil.metrics.lawson_sigma_over_peak]);
* the 2008-2012 contests (Cotton et al. 2008, Proc. SPIE 7013; Malbet et
  al. 2010, Proc. SPIE 7734, https://arxiv.org/abs/1007.4473; Baron et
  al. 2012, Proc. SPIE 8445, https://arxiv.org/abs/1207.7141): the rms
  difference after convolving both images with a Gaussian beam
  ([`rms_convolved`][virgil.metrics.rms_convolved]);
* the 2024 contest's ``ImageMetrics`` L1 score
  ([`l1_score`][virgil.metrics.l1_score]);
* the normalized cross-correlation ([`ncc`][virgil.metrics.ncc]).

[`score`][virgil.metrics.score] resamples, aligns and evaluates all of them.
[`ring_features`][virgil.metrics.ring_features] is separate: it needs no
truth, and measures ring geometry as the EHT ring analyses do.
Images are in the virgil orientation (row 0 North, column 0 East, centre at
the middle of the pixel grid, as in
[`Image`][virgil.models.Image]); each function takes an
[`Image`][virgil.models.Image] or an array with its ``pixel_scale_mas``.
Comparing two images pixel by pixel needs them on the same grid and, because
interferometric data do not fix the absolute position, usually aligned
first.
"""

import dataclasses

import jax
import jax.numpy as np
import numpy as onp
from jax.scipy.ndimage import map_coordinates
from scipy.ndimage import map_coordinates as _sp_map_coordinates
from scipy.optimize import minimize

from ._utils import _unit_sum
from .imaging import convolve_beam

__provisional__ = ("*",)


def _array_and_scale(image, pixel_scale_mas):
    """The pixel array and pixel size of an ``Image`` or an array."""
    if hasattr(image, "brightness") and hasattr(image, "pixel_scale_mas"):
        scale = (
            image.pixel_scale_mas
            if pixel_scale_mas is None
            else pixel_scale_mas
        )
        # A smoothed Image is scored as the sky the data see.
        pixels = (
            image.smoothed()
            if hasattr(image, "smoothed")
            else image.brightness
        )
        pixels = np.asarray(pixels)
        # An image on a rotated grid is scored as the sky: North up.
        frame = (
            image._numeric_frame() if hasattr(image, "_numeric_frame") else 0.0
        )
        if frame != 0.0:
            if pixels.shape[0] != pixels.shape[1]:
                raise ValueError(
                    "An Image on a rotated grid is scored on a square "
                    f"grid, not {pixels.shape}; set rotation_deg=0 or "
                    "use a square grid."
                )
            # Render on a canvas that holds the turned square, normalize
            # by its total, then crop to the North-up n x n box: the corners
            # that fall outside it are lost, not rescaled into the rest.
            n = pixels.shape[0]
            c, s = (
                abs(np.cos(np.radians(frame))),
                abs(np.sin(np.radians(frame))),
            )
            big = int(np.ceil(n * (c + s))) + 1
            big += (big - n) % 2
            # The pixel grid's own offset and scale, as for a North-up Image.
            placed = dataclasses.replace(
                image, dra=0.0, ddec=0.0, pixel_scale_mas=scale
            )
            sky = np.asarray(placed.render(big, big * scale))
            lo = (big - n) // 2
            pixels = sky[lo : lo + n, lo : lo + n] * (
                np.sum(pixels) / np.sum(sky)
            )
        return pixels, float(scale)
    if pixel_scale_mas is None:
        raise ValueError("Give pixel_scale_mas for an array (not an Image).")
    return np.asarray(image), float(pixel_scale_mas)


def _check_same_shape(a, b):
    if a.ndim != 2 or a.shape != b.shape:
        raise ValueError(
            f"The images must be 2D and of the same shape, not {a.shape} "
            f"and {b.shape}; use resample first."
        )


def resample(image, pixel_scale_mas, npix, new_pixel_scale_mas):
    """Resample an image onto a new grid, conserving flux.

    Each old pixel is treated as uniform and its flux integrated over the new
    pixels (the cumulative flux interpolated at the new pixel edges, then
    differenced), so coarser pixels area-average rather than alias. Both
    grids are centred on the same sky position (the middle of the array);
    flux outside the old field is zero, and old flux outside the new field
    is dropped.

    Parameters
    ----------
    image : Image or array-like, shape (ny, nx)
        The image, with row 0 North and column 0 East.
    pixel_scale_mas : float or None
        Pixel size of ``image`` in milliarcseconds (taken from an
        [`Image`][virgil.models.Image] if ``None``).
    npix : int or (int, int)
        Number of pixels of the new grid, ``(ny, nx)`` or a square side.
    new_pixel_scale_mas : float
        Pixel size of the new grid in milliarcseconds.

    Returns
    -------
    jax.Array, shape (ny_new, nx_new)
        The resampled pixel fluxes.
    """
    arr, scale = _array_and_scale(image, pixel_scale_mas)
    ny, nx = (npix, npix) if onp.ndim(npix) == 0 else npix
    ny0, nx0 = arr.shape
    # Cumulative flux at the old pixel edges, zero at the first edge.
    cum = np.pad(np.cumsum(np.cumsum(arr, axis=0), axis=1), ((1, 0), (1, 0)))

    def edges(n_new, n_old):
        # Edge k of the new grid sits at index k - 1/2; in sky offsets that is
        # ((n - 1)/2 - (k - 1/2)) * scale, and the old edge coordinate is the
        # matching old pixel index plus 1/2.
        k = np.arange(n_new + 1)
        offset = (0.5 * (n_new - 1) - (k - 0.5)) * new_pixel_scale_mas
        return 0.5 * (n_old - 1) - offset / scale + 0.5

    rows, cols = np.meshgrid(edges(ny, ny0), edges(nx, nx0), indexing="ij")
    cum_new = map_coordinates(cum, [rows, cols], order=1, mode="nearest")
    return np.diff(np.diff(cum_new, axis=0), axis=1)


def _shift(arr, drow, dcol):
    """``out[r, c] = arr[r - drow, c - dcol]``, zero outside, bilinear."""
    rows, cols = np.meshgrid(
        np.arange(arr.shape[0]) - drow,
        np.arange(arr.shape[1]) - dcol,
        indexing="ij",
    )
    return map_coordinates(
        arr, [rows, cols], order=1, mode="constant", cval=0.0
    )


def ncc(image, truth):
    """Normalized cross-correlation of two images at zero shift.

    The zero-mean form, ``Σ (e - ē)(r - r̄) / sqrt(Σ (e - ē)² Σ (r - r̄)²)``
    for the image ``e`` and truth ``r``: 1 for identical images up to a
    positive scale and offset, and independent of the flux scale.

    Parameters
    ----------
    image, truth : array-like, shape (ny, nx)
        Images on the same grid.

    Returns
    -------
    jax.Array
        The correlation coefficient, in [-1, 1].
    """
    e, r = np.asarray(image), np.asarray(truth)
    _check_same_shape(e, r)
    e = e - np.mean(e)
    r = r - np.mean(r)
    return np.sum(e * r) / np.sqrt(np.sum(e**2) * np.sum(r**2))


def l1_score(image, truth):
    """The L1 score of the 2024 imaging contest's ``ImageMetrics``.

    ``1 - min_{a >= 0} Σ|a e - r| / Σ r`` for the image ``e`` and truth
    ``r``: 1 for a perfect reconstruction up to flux scale, 0 for an image
    no better than none. Since ``Σ|a e - r| = Σ e |a - r/e|``, the optimal
    scale ``a`` is the median of ``r/e`` with weights ``e``, found by
    sorting. Pixels of the image must be non-negative; negative values are
    set to zero. The score does not depend on the flux scale of either image.

    Parameters
    ----------
    image, truth : array-like, shape (ny, nx)
        Images on the same grid.

    Returns
    -------
    jax.Array
        The score, in [0, 1].
    """
    e, r = np.asarray(image), np.asarray(truth)
    _check_same_shape(e, r)
    e, r = np.ravel(np.maximum(e, 0.0)), np.ravel(r)
    live = e > 0
    ratio = np.where(live, r / np.where(live, e, 1.0), np.inf)
    order = np.argsort(ratio)
    ratio, weight = ratio[order], e[order]
    # Weighted median: the first ratio where the cumulative weight reaches
    # half of the total (pixels with zero weight sort last, at inf).
    index = np.searchsorted(np.cumsum(weight), 0.5 * np.sum(weight))
    a = ratio[np.minimum(index, ratio.size - 1)]
    a = np.where(np.sum(weight) > 0, a, 0.0)
    return 1.0 - np.sum(np.abs(a * e - r)) / np.sum(r)


def rms_convolved(
    image, truth, pixel_scale_mas=None, beam=None, relative=False
):
    """The rms difference of two images, after convolving with a beam.

    The metric of the 2008-2012 contests: both images are normalized to unit
    flux, convolved with a Gaussian beam (the data's resolution, so that
    structure the data cannot resolve does not count against the
    reconstruction), and the rms of their difference is taken. With no beam
    the images are compared as they are.

    Parameters
    ----------
    image, truth : Image or array-like, shape (ny, nx)
        Images on the same grid.
    pixel_scale_mas : float, optional
        Pixel size in milliarcseconds; needed only with a ``beam``
        (for an ``Image`` it is taken from the image). Without a beam,
        arrays are compared as they are and no pixel size is required.
    beam : Beam, optional
        The beam, usually [`beam(data)`][virgil.imaging.beam].
    relative : bool, optional
        Divide by the peak of the convolved truth, so the result is a
        fraction of the peak rather than of the total flux per pixel.

    Returns
    -------
    jax.Array
        The rms difference.
    """
    if beam is None:
        e, r = (
            _array_and_scale(x, None)[0]
            if hasattr(x, "brightness")
            else np.asarray(x)
            for x in (image, truth)
        )
        scale = None
    else:
        e, scale = _array_and_scale(image, pixel_scale_mas)
        r, _ = _array_and_scale(truth, scale)
    _check_same_shape(e, r)
    e, r = _unit_sum(e), _unit_sum(r)
    if beam is not None:
        e = convolve_beam(e, scale, beam)
        r = convolve_beam(r, scale, beam)
    rms = np.sqrt(np.mean((e - r) ** 2))
    return rms / np.max(r) if relative else rms


def lawson_sigma_over_peak(image, truth):
    """The 2004 Beauty Contest figure of merit, σ over the peak.

    Lawson et al. (2004), Eq. 2: with both images normalized to unit sum,
    ``σ² = Σ r (e - r)² / Σ r`` for the image ``e`` and truth ``r``, and the
    score is σ divided by the peak of the truth. Smaller is better; 0 is a
    perfect reconstruction. Pixels where the truth is zero are not scored
    directly, but flux put there lowers the flux on the source (the sum is
    fixed), so it still changes σ.

    Parameters
    ----------
    image, truth : array-like, shape (ny, nx)
        Images on the same grid.

    Returns
    -------
    jax.Array
        σ / peak.
    """
    e, r = np.asarray(image), np.asarray(truth)
    _check_same_shape(e, r)
    e, r = _unit_sum(e), _unit_sum(r)
    sigma = np.sqrt(np.sum(r * (e - r) ** 2) / np.sum(r))
    return sigma / np.max(r)


# Shifts per lax.map batch in ``align``: bounds the intermediates to
# _ALIGN_BATCH images at a time instead of one per shift.
_ALIGN_BATCH = 64


@jax.jit
# deliberate: test seam (tests/test_metrics.py)
def _ncc_table(image, truth, shifts):
    """The NCC of ``image`` shifted by each of ``shifts`` (rows, columns)
    against ``truth``, as a ``(len(shifts),)`` array; compiled once."""
    return jax.lax.map(
        lambda s: ncc(_shift(image, s[0], s[1]), truth),
        shifts,
        batch_size=_ALIGN_BATCH,
    )


def align(image, truth, max_shift_mas, pixel_scale_mas=None):
    """Shift an image to best match the truth, by maximizing the NCC.

    Interferometric data do not fix the absolute position, so a
    reconstruction can sit anywhere in the field. The shift is found by an
    exhaustive integer-pixel search of up to ``max_shift_mas`` along each
    axis, refined to a fraction of a pixel by a parabola through the NCC
    along each axis around the best integer shift. The shift itself is done
    by bilinear interpolation, with zeros brought in at the edge.

    Parameters
    ----------
    image : Image or array-like, shape (ny, nx)
        The image to shift.
    truth : Image or array-like, shape (ny, nx)
        The reference, on the same grid.
    max_shift_mas : float
        Largest shift to search, in mas, along each axis.
    pixel_scale_mas : float, optional
        Pixel size in milliarcseconds, if not given by an
        [`Image`][virgil.models.Image].

    Returns
    -------
    shifted : jax.Array, shape (ny, nx)
        The shifted image.
    shift_mas : tuple of float
        The shift applied, ``(dra, ddec)`` in mas, positive towards East and
        North.
    """
    e, scale = _array_and_scale(image, pixel_scale_mas)
    r, _ = _array_and_scale(truth, scale)
    _check_same_shape(e, r)
    n = int(onp.floor(max_shift_mas / scale + 1e-9))
    if n < 1:
        return e, (0.0, 0.0)
    steps = onp.arange(-n, n + 1)
    # table[i, j]: NCC at a shift of i - n rows and j - n columns.
    pairs = onp.stack(onp.meshgrid(steps, steps, indexing="ij"), axis=-1)
    table = onp.asarray(
        _ncc_table(
            e,
            r,
            np.asarray(
                pairs.reshape(-1, 2), dtype=np.result_type(e.dtype, float)
            ),
        ),
        dtype=float,
    ).reshape(steps.size, steps.size)
    table = onp.nan_to_num(table, nan=-onp.inf)
    i, j = onp.unravel_index(onp.argmax(table), table.shape)

    def refine(minus, centre, plus, at_edge):
        curvature = minus - 2 * centre + plus
        if at_edge or not onp.isfinite(curvature) or curvature >= 0:
            return 0.0
        return float(onp.clip(0.5 * (minus - plus) / curvature, -0.5, 0.5))

    drow = (i - n) + refine(
        table[max(i - 1, 0), j],
        table[i, j],
        table[min(i + 1, 2 * n), j],
        i in (0, 2 * n),
    )
    dcol = (j - n) + refine(
        table[i, max(j - 1, 0)],
        table[i, j],
        table[i, min(j + 1, 2 * n)],
        j in (0, 2 * n),
    )
    # Rows run towards decreasing Dec and columns towards decreasing RA.
    return _shift(e, drow, dcol), (-dcol * scale, -drow * scale)


def score(
    image,
    truth,
    *,
    pixel_scale_mas=None,
    truth_pixel_scale_mas=None,
    beam=None,
    v2_only=False,
    max_shift_mas=None,
):
    """Every image-recovery metric of a reconstruction against a truth.

    The image is resampled onto the truth's grid if the grids differ
    ([`resample`][virgil.metrics.resample]), aligned to the truth
    ([`align`][virgil.metrics.align]) and normalized to unit flux, then
    scored.

    Parameters
    ----------
    image, truth : Image or array-like
        The reconstruction and the true image, each an
        [`Image`][virgil.models.Image] or an array.
    pixel_scale_mas, truth_pixel_scale_mas : float, optional
        Pixel sizes in mas, for arrays. The truth's defaults to the
        image's.
    beam : Beam, optional
        If given, also report the rms difference after convolving with it.
    v2_only : bool, optional
        For data without phase information (squared visibilities only) an
        image and its 180° rotation fit equally well. With this set the
        rotated image is scored as well and the better of the two, by NCC,
        is kept.
    max_shift_mas : float, optional
        Search range of the alignment, in mas. ``None`` does not align.

    Returns
    -------
    dict
        ``ncc``, ``l1``, ``lawson``, ``rms``, ``shift_dra_mas`` and
        ``shift_ddec_mas`` (the alignment applied), ``rotated`` (whether the
        180° rotated image was kept), and ``rms_convolved`` if a beam is
        given. All are floats except ``rotated``.
    """
    e, scale = _array_and_scale(image, pixel_scale_mas)
    if truth_pixel_scale_mas is None:
        truth_pixel_scale_mas = pixel_scale_mas
    r, truth_scale = _array_and_scale(truth, truth_pixel_scale_mas)
    if e.shape != r.shape or not onp.isclose(scale, truth_scale):
        e = resample(e, scale, r.shape, truth_scale)
    r = _unit_sum(r)
    e = _unit_sum(e)

    def aligned(candidate):
        if max_shift_mas is None:
            return candidate, (0.0, 0.0)
        return align(candidate, r, max_shift_mas, truth_scale)

    best, shift = aligned(e)
    rotated = False
    if v2_only:
        flipped, flipped_shift = aligned(e[::-1, ::-1])
        if ncc(flipped, r) > ncc(best, r):
            best, shift, rotated = flipped, flipped_shift, True
    best = _unit_sum(best)

    out = {
        "ncc": float(ncc(best, r)),
        "l1": float(l1_score(best, r)),
        "lawson": float(lawson_sigma_over_peak(best, r)),
        "rms": float(rms_convolved(best, r, truth_scale)),
        "shift_dra_mas": float(shift[0]),
        "shift_ddec_mas": float(shift[1]),
        "rotated": rotated,
    }
    if beam is not None:
        out["rms_convolved"] = float(rms_convolved(best, r, truth_scale, beam))
    return out


@dataclasses.dataclass(frozen=True)
class RingFeatures:
    """Ring geometry from [`ring_features`][virgil.metrics.ring_features].

    Lengths in mas, ``pa`` in degrees East of North, ``centre`` the
    ``(dra, ddec)`` offset in mas (East, North) from the grid centre. Each
    ``*_std`` is the dispersion of its feature over angle (``pa`` and
    ``asymmetry``: over radius in the annulus), not a measurement error.
    """

    centre: tuple
    diameter: float
    diameter_std: float
    width: float
    width_std: float
    pa: float
    pa_std: float
    asymmetry: float
    asymmetry_std: float
    central_fraction: float
    central_fraction_std: float


def _polar(pix, ps, ra0, dec0, r, theta):
    """Bilinear samples of ``pix`` at radii ``r`` (mas) x PA ``theta`` (rad).

    Row 0 is North and column 0 East; ``(ra0, dec0)`` is the East/North
    offset in mas of the polar centre from the grid centre. Outside = 0.
    """
    ny, nx = pix.shape
    north = dec0 + r[:, None] * onp.cos(theta)[None, :]
    east = ra0 + r[:, None] * onp.sin(theta)[None, :]
    rows = (ny - 1) / 2 - north / ps
    cols = (nx - 1) / 2 - east / ps
    return _sp_map_coordinates(
        pix, [rows, cols], order=1, mode="constant", cval=0.0
    )


def ring_features(
    image,
    pixel_scale_mas=None,
    *,
    search_mas=None,
    n_search=7,
    n_theta=180,
    oversample=4,
    refine=True,
):
    """Ring diameter, width, orientation, asymmetry and central brightness.

    Source: Event Horizon Telescope Collaboration (2019),
    [references](../references.md#eht2019_iv), after REX in eht-imaging.

    The image is resampled in polar coordinates about a trial centre.
    The centre minimizes ``σ_r / ⟨r_pk⟩``, with ``r_pk(θ)`` the radius of
    peak brightness at each position angle ``θ``. Then: diameter
    ``2⟨r_pk⟩``; width the mean FWHM of ``I(r, θ) − I_floor``, with
    ``I_floor = ⟨I(2.5 r_pk, θ)⟩``; orientation ``η`` and asymmetry ``A`` from
    the ``m = 1`` azimuthal mode ``∫ I e^{iθ} dθ`` over the annulus
    ``r_pk ± w/2`` (``η`` East of North, ``A = |m=1| / m=0``, so a
    modulation ``1 + a cos(θ − η)`` has ``A = a/2``); central fraction
    ``⟨I(r < 0.25 r_pk)⟩ / ⟨I(r_pk, θ)⟩``.
    The ratios 2.5 and 0.25 are the paper's fixed 50 and 5 μas scaled to
    its ring radius (about 20 μas), so any ring size is treated alike.
    ``η`` (circular mean) and ``A`` are averaged over radius after taking the
    phase and ratio; ``pa_std`` is the circular standard deviation.

    The diameter and width are biased by the finite resolution of the image
    (a blurred or wide ring reads larger or wider): calibrate against
    rendered truth for each method before comparing across methods.

    Parameters
    ----------
    image : Image or array-like, shape (ny, nx)
        In the virgil orientation (row 0 North, column 0 East).
    pixel_scale_mas : float, optional
        Pixel size; required for an array.
    search_mas : float, optional
        Half-width of the square of trial centres about the grid centre;
        default a quarter of the half field of view.
    n_search : int
        Trial centres per axis in the coarse search.
    n_theta : int
        Position angles per ring.
    oversample : float
        Radial samples per pixel.
    refine : bool
        Polish the best trial centre with Nelder-Mead.

    Returns
    -------
    RingFeatures
        ``centre`` (dra, ddec) in mas, ``diameter``, ``width`` (mas), ``pa``
        (degrees), ``asymmetry``, ``central_fraction``, each with a ``_std``.
    """
    pix, ps = _array_and_scale(image, pixel_scale_mas)
    pix = onp.asarray(pix, dtype=float)
    if pix.ndim != 2:
        raise ValueError(f"image must be 2D, not {pix.shape}")
    half = 0.5 * min(pix.shape) * ps
    search = 0.25 * half if search_mas is None else float(search_mas)
    theta = onp.linspace(0.0, 2 * onp.pi, n_theta, endpoint=False)
    dr = ps / oversample
    r = onp.arange(dr, 2 * half, dr)

    def _rpk(c):
        """Peak radius per angle about centre ``c`` (parabolic refinement)."""
        prof = _polar(pix, ps, c[0], c[1], r, theta)
        k = onp.argmax(prof, axis=0)
        cols = onp.arange(n_theta)
        kk = onp.clip(k, 1, len(r) - 2)
        a, b, c3 = prof[kk - 1, cols], prof[kk, cols], prof[kk + 1, cols]
        den = a - 2 * b + c3
        ok = den < 0
        off = onp.where(ok, 0.5 * (a - c3) / onp.where(ok, den, 1.0), 0.0)
        off = onp.clip(off, -1, 1)
        return onp.where(k == kk, r[kk] + off * dr, r[k])

    def cost(c):
        rp = _rpk(c)
        return float(onp.std(rp) / onp.mean(rp))

    g = onp.linspace(-search, search, n_search)
    step = 2 * search / max(n_search - 1, 1)
    best = min(((x, y) for x in g for y in g), key=cost)
    if refine:
        res = minimize(
            cost,
            onp.asarray(best),
            method="Nelder-Mead",
            options={
                "xatol": 0.02 * ps,
                "fatol": 1e-6,
                # Steps of the coarse grid, not SciPy's absolute 2.5e-4.
                "initial_simplex": onp.asarray(best)
                + step * onp.array([[0, 0], [1, 0], [0, 1]]),
            },
        )
        if res.fun <= cost(best):
            best = tuple(res.x)
    ra0, dec0 = float(best[0]), float(best[1])

    rp = _rpk((ra0, dec0))
    rmean = float(onp.mean(rp))

    # Width: FWHM of I - floor, floor = <I(2 r_pk, theta)>.
    prof = _polar(pix, ps, ra0, dec0, r, theta)
    # The floor ring is kept inside the image.
    r_floor = min(2.5 * rmean, half - max(abs(ra0), abs(dec0)))
    floor = float(
        onp.mean(_polar(pix, ps, ra0, dec0, onp.array([r_floor]), theta))
    )
    widths = onp.zeros(n_theta)
    for j in range(n_theta):
        y = onp.maximum(prof[:, j] - floor, 0.0)
        k = int(onp.argmax(y))
        lvl = 0.5 * y[k]
        lo, hi = 0.0, r[-1]
        if y[k] <= 0:
            continue  # dark azimuth: width 0
        for i in range(k, 0, -1):
            if y[i - 1] < lvl <= y[i]:
                lo = r[i - 1] + (lvl - y[i - 1]) / (y[i] - y[i - 1]) * dr
                break
        for i in range(k, len(r) - 1):
            if y[i + 1] < lvl <= y[i]:
                hi = r[i] + (y[i] - lvl) / (y[i] - y[i + 1]) * dr
                break
        widths[j] = hi - lo
    width = float(onp.mean(widths))

    # m = 1 mode over the annulus r_pk +- w/2.
    ann = onp.linspace(
        max(rmean - width / 2, dr), rmean + width / 2, max(3, int(width / dr))
    )
    a = _polar(pix, ps, ra0, dec0, ann, theta)
    m0 = a.mean(axis=1)
    m1 = (a * onp.exp(1j * theta)[None, :]).mean(axis=1)
    live = (m0 > 0) & (onp.abs(m1) > 0)
    unit = m1[live] / onp.abs(m1[live])
    resultant = onp.abs(onp.mean(unit))
    ratio = onp.abs(m1[m0 > 0]) / m0[m0 > 0]

    # Central fraction, area-weighted over r < 0.25 r_pk.
    m_c = max(4, int(0.25 * rmean / dr))
    rc = (onp.arange(m_c) + 0.5) * 0.25 * rmean / m_c
    cen = _polar(pix, ps, ra0, dec0, rc, theta).mean(axis=1)
    mean_c = float(onp.sum(cen * rc) / max(onp.sum(rc), 1e-30))
    ring_pk = onp.array(
        [onp.interp(rp[j], r, prof[:, j]) for j in range(n_theta)]
    )
    fc = mean_c / float(onp.mean(ring_pk))
    return RingFeatures(
        centre=(ra0, dec0),
        diameter=2 * rmean,
        diameter_std=2 * float(onp.std(rp)),
        width=width,
        width_std=float(onp.std(widths)),
        pa=float(onp.degrees(onp.angle(onp.mean(unit))) % 360.0),
        pa_std=float(
            onp.degrees(onp.sqrt(-2 * onp.log(max(resultant, 1e-12))))
        ),
        asymmetry=float(onp.mean(ratio)),
        asymmetry_std=float(onp.std(ratio)),
        central_fraction=fc,
        central_fraction_std=float(
            abs(fc) * onp.std(ring_pk) / onp.mean(ring_pk)
        ),
    )
