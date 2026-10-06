"""Scores for comparing a reconstructed image with a known truth.

The image-recovery figures of merit used by the interferometric imaging
contests, written from their published definitions:

* the 2004 Beauty Contest, Lawson et al. (2004, Proc. SPIE 5491, 886),
  Eq. 2: the flux-weighted rms of the normalised difference, over the
  peak ([`lawson_sigma_over_peak`][virgil.metrics.lawson_sigma_over_peak]);
* the 2008-2012 contests (Cotton et al. 2008, Proc. SPIE 7013; Malbet et
  al. 2010, Proc. SPIE 7734; Baron et al. 2012, Proc. SPIE 8445): the rms
  difference after convolving both images with a Gaussian beam
  ([`rms_convolved`][virgil.metrics.rms_convolved]);
* the 2024 contest's ``ImageMetrics`` L1 score
  ([`l1_score`][virgil.metrics.l1_score]);
* the normalised cross-correlation ([`ncc`][virgil.metrics.ncc]).

[`score`][virgil.metrics.score] resamples, aligns and evaluates all of them.
Images are in the virgil orientation (row 0 North, column 0 East, centre at
the middle of the pixel grid, as in
[`Image`][virgil.models.Image]); each function takes an
[`Image`][virgil.models.Image] or an array with its ``pixel_scale_mas``.
Comparing two images pixel by pixel needs them on the same grid and, because
interferometric data do not fix the absolute position, usually aligned
first.
"""

import jax.numpy as np
import numpy as onp
from jax.scipy.ndimage import map_coordinates

from .imaging import convolve_beam


def _array_and_scale(image, pixel_scale_mas):
    """The pixel array and pixel size of an ``Image`` or an array."""
    if hasattr(image, "brightness") and hasattr(image, "pixel_scale_mas"):
        scale = (
            image.pixel_scale_mas
            if pixel_scale_mas is None
            else pixel_scale_mas
        )
        return np.asarray(image.brightness), float(scale)
    if pixel_scale_mas is None:
        raise ValueError("Give pixel_scale_mas for an array (not an Image).")
    return np.asarray(image), float(pixel_scale_mas)


def _unit_sum(image):
    return image / np.sum(image)


def _check_same_shape(a, b):
    if a.ndim != 2 or a.shape != b.shape:
        raise ValueError(
            f"The images must be 2D and of the same shape, not {a.shape} "
            f"and {b.shape}; use resample first."
        )


def resample(image, pixel_scale_mas, npix, new_pixel_scale_mas):
    """Resample an image onto a new grid, conserving flux.

    The pixel fluxes are integrated exactly over the new pixels, treating
    each old pixel as uniform, by linearly interpolating the cumulative flux
    at the new pixel edges with
    ``jax.scipy.ndimage.map_coordinates`` and differencing.
    That is flux-conserving, and area-averages rather than aliases when the
    new pixels are coarser. Both grids are centred on the same sky position
    (the middle of the array, as in
    [`pixel_offsets`][virgil._geometry.pixel_offsets]); flux outside the old
    field is zero, and old flux outside the new field is dropped.

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
    """Normalised cross-correlation of two images at zero shift.

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

    The metric of the 2008-2012 contests: both images are normalised to unit
    flux, convolved with a Gaussian beam (the data's resolution, so that
    structure the data cannot resolve does not count against the
    reconstruction), and the rms of their difference is taken. With no beam
    the images are compared as they are.

    Parameters
    ----------
    image, truth : Image or array-like, shape (ny, nx)
        Images on the same grid.
    pixel_scale_mas : float, optional
        Pixel size in milliarcseconds; needed with a ``beam`` for arrays.
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

    Lawson et al. (2004), Eq. 2: with both images normalised to unit flux,
    σ is the root of the mean squared difference weighted by the true flux,
    ``σ² = Σ r (e - r)² / Σ r``, and the score is σ divided by the peak of
    the truth. Smaller is better; 0 is a perfect reconstruction. Weighting
    by the true flux means errors in the empty sky do not count.

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


def align(image, truth, max_shift_mas, pixel_scale_mas=None):
    """Shift an image to best match the truth, by maximising the NCC.

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
    shifts = range(-n, n + 1)
    # Row i of the NCC table is the shift in rows, column j in columns.
    table = onp.array(
        [[float(ncc(_shift(e, i, j), r)) for j in shifts] for i in shifts]
    )
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
    ([`align`][virgil.metrics.align]) and normalised to unit flux, then
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
