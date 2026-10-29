"""CLEAN deconvolution of radio visibilities (Cotton–Schwab, multi-scale).

The algorithm and the names of the arguments follow CASA's ``tclean`` and
WSClean; see the [radio CLEAN guide](../radio_clean.md). Not to be confused
with [`virgil.imaging.clean`][virgil.imaging.clean], the gradient-based sparse
imager for optical interferometry data.

!!! warning "Provisional"
    May change in any 0.4.x release without a deprecation cycle.
"""

from __future__ import annotations

import dataclasses
import time

import jax
import jax.numpy as np
import numpy as onp
from scipy import fft as sfft
from scipy import ndimage
from scipy.optimize import least_squares

from ._geometry import image_visibilities
from ._utils import mas2rad
from .radio import ComplexVisData

__provisional__ = ("*",)

__all__ = ["RadioCleanResult", "clean", "fit_beam", "restore"]

FWHM_TO_SIGMA = 1.0 / (2.0 * onp.sqrt(2.0 * onp.log(2.0)))


# ----------------------------------------------------------------------
# Beams and masks
# ----------------------------------------------------------------------


# CASA's divergence tests (CASAdocs, synthesis imaging, Iteration Control), as
# relative increases of the peak residual: a minor cycle ends at 10 per cent
# above its running minimum (the stop-code table says "from minimum", the text
# "from the previous peak"; virgil uses the minimum, per scale); the run stops
# at more than 3.0 above the previous major cycle or above the minimum so far,
# both over the whole image, mask ignored.
_MINOR_RISE = 0.1
_MAJOR_RISE = 3.0


def _gaussian(shape, centre, bmaj, bmin, pa_deg):
    """Peak-1 elliptical Gaussian; FWHM in pixels, ``pa_deg`` East of North."""
    rr, cc = onp.indices(shape, dtype=float)
    east, north = (
        -(cc - centre[1]),
        -(rr - centre[0]),
    )  # col 0 East, row 0 North
    pa = onp.deg2rad(pa_deg)
    along = east * onp.sin(pa) + north * onp.cos(pa)
    across = east * onp.cos(pa) - north * onp.sin(pa)
    return onp.exp(
        -4.0 * onp.log(2.0) * ((along / bmaj) ** 2 + (across / bmin) ** 2)
    )


def fit_beam(psf, pixel_scale_mas, psfcutoff=0.35):
    """Gaussian fitted to the main lobe of a PSF, as ``tclean`` does.

    Source: CASA's ``psfcutoff`` fit (the lobe above the cutoff).

    Parameters
    ----------
    psf : (n, n) array
        Dirty beam, peak 1.
    pixel_scale_mas : float
    psfcutoff : float
        The lobe is the connected region around the peak above this level.

    Returns
    -------
    (bmaj_mas, bmin_mas, pa_deg) : tuple
        FWHM axes and the position angle of the major axis East of North.
    """
    psf = onp.asarray(psf, float)
    peak = onp.unravel_index(onp.argmax(psf), psf.shape)
    lab, _ = ndimage.label(psf >= psfcutoff)
    lobe = lab == lab[peak]
    rows, cols = onp.where(lobe)
    r0, r1 = max(rows.min() - 1, 0), min(rows.max() + 2, psf.shape[0])
    c0, c1 = max(cols.min() - 1, 0), min(cols.max() + 2, psf.shape[1])
    sub = psf[r0:r1, c0:c1]
    lobe_sub = lobe[r0:r1, c0:c1]
    # circular start from the lobe's area at the cutoff
    fwhm = (
        2.0
        * onp.sqrt(lobe.sum() / onp.pi)
        * onp.sqrt(onp.log(2) / onp.log(1 / psfcutoff))
    )

    def resid(p):
        return ((_gaussian(sub.shape, p[:2], *p[2:]) - sub) * lobe_sub).ravel()

    best = None
    for pa0 in (0.0, 45.0, 90.0, 135.0):  # the angle is ill-conditioned
        x0 = [peak[0] - r0, peak[1] - c0, fwhm, 0.9 * fwhm, pa0]
        fit = least_squares(resid, x0, x_scale=[1, 1, fwhm, fwhm, 30.0])
        if best is None or fit.cost < best.cost:
            best = fit
    a, b, pa = best.x[2:]
    if b > a:
        a, b, pa = b, a, pa + 90.0
    return (
        float(a * pixel_scale_mas),
        float(b * pixel_scale_mas),
        float(pa % 180),
    )


def restore(model, residual, beam, pixel_scale_mas):
    """Restored image in Jy/beam: ``model`` convolved with ``beam``, plus the
    residual.

    The kernel is built here, peak 1 as Jy/beam needs, to keep this module
    off ``imaging``.

    Parameters
    ----------
    model : (n, n) array
        Jy per pixel.
    residual : (n, n) array
        Jy/beam.
    beam : (bmaj_mas, bmin_mas, pa_deg)
    pixel_scale_mas : float
    """
    n0, n1 = model.shape
    bmaj, bmin, pa = beam
    kernel = _gaussian(
        model.shape,
        ((n0 - 1) / 2, (n1 - 1) / 2),
        bmaj / pixel_scale_mas,
        bmin / pixel_scale_mas,
        pa,
    )
    pad = tuple(sfft.next_fast_len(2 * s, real=True) for s in model.shape)
    conv = sfft.irfft2(
        sfft.rfft2(model, s=pad) * sfft.rfft2(kernel, s=pad), s=pad
    )
    r, c = (n0 - 1) // 2, (n1 - 1) // 2  # the kernel's centre is the origin
    return conv[r : r + n0, c : c + n1] + residual


def _mad_rms(image):
    """Robust rms: 1.4826 × the median absolute deviation."""
    return float(1.4826 * onp.median(onp.abs(image - onp.median(image))))


class _Smoother:
    """Convolution with unit-area Gaussians (zero-padded FFT)."""

    def __init__(self, shape):
        self.shape = tuple(shape)
        self.pad = tuple(sfft.next_fast_len(2 * s, real=True) for s in shape)
        self._fy = onp.fft.fftfreq(self.pad[0])[:, None]
        self._fx = onp.fft.rfftfreq(self.pad[1])[None, :]

    def __call__(self, image, sigma_pix):
        if sigma_pix == 0:
            return onp.array(image, float)
        gain = onp.exp(
            -2.0 * onp.pi**2 * sigma_pix**2 * (self._fx**2 + self._fy**2)
        )
        out = sfft.irfft2(sfft.rfft2(image, s=self.pad) * gain, s=self.pad)
        return out[: self.shape[0], : self.shape[1]]


def _area(sigma_pix):
    """Pixels under a peak-1 Gaussian (a point is 1)."""
    return 1.0 if sigma_pix == 0 else 2.0 * onp.pi * sigma_pix**2


# deliberate: tests call the masking rule directly
def _auto_mask(residual, beam_pix, psf_sidelobe, sidelobe, noise, low):
    """A simplified ``auto-multithresh``: the connected regions of the
    beam-smoothed residual above ``low`` × rms that hold a pixel above
    ``max(noise × rms, sidelobe × sidelobe level × peak)``."""
    smooth = _Smoother(residual.shape)(residual, beam_pix * FWHM_TO_SIGMA)
    rms = _mad_rms(smooth)
    # a poor PSF must not hide the brightest region: seed at half the peak at most
    high = min(
        max(noise * rms, sidelobe * psf_sidelobe * smooth.max()),
        0.5 * smooth.max(),
    )
    lab, n = ndimage.label(smooth > low * rms)
    if n == 0:
        return onp.zeros(residual.shape, bool)
    keep = onp.unique(lab[smooth > high])
    return onp.isin(lab, keep[keep > 0])


# ----------------------------------------------------------------------
# Minor cycle
# ----------------------------------------------------------------------


# deliberate: the ~100-line minor cycle reads better apart from the major loop
def _minor_cycle(
    residual, psf, scales, mask, gain, niter, threshold, bias, patch
):
    """Multi-scale CLEAN of ``residual`` (Jy/beam) down to ``threshold``.

    Cornwell's (2008) algorithm with Gaussian scales (FWHM in pixels, 0 a
    point) and CASA's small-scale bias. The component and scale maximizing
    the bias-weighted ``D_s / sqrt(H_ss)`` is found (``D_s`` the residual
    smoothed to the scale, ``H_ss`` the smoothed PSF's peak), ``gain * D_s /
    H_ss`` is added, and the cross-convolved PSF is subtracted from every
    smoothed residual in a window of half-width ``patch`` (Clark's
    restriction). A single scale 0 is Högbom's CLEAN. ``psf`` is the dirty
    beam out to the farthest lag a window uses, so larger than the image
    grid. The cycle also ends if a scale's peak rises 10 per cent above its
    running minimum (divergence). Returns the components
    ``(n_scales, n, n)``, the iteration count and why the cycle ended
    (``"threshold"``, ``"diverged"`` or ``"niter"``).
    """
    scales = [float(s) for s in scales]
    sig = [s * FWHM_TO_SIGMA for s in scales]
    smax = max(scales)
    wbias = onp.array(
        [1.0 - (bias * s / smax if smax > 0 else 0.0) for s in scales]
    )
    rows, cols = onp.where(mask)  # nothing outside the mask's box is chosen
    r0, r1, c0, c1 = rows.min(), rows.max() + 1, cols.min(), cols.max() + 1
    cm = mask[r0:r1, c0:c1]
    ny, nx = cm.shape
    sm = _Smoother(cm.shape)
    ctr = (psf.shape[0] // 2, psf.shape[1] // 2)
    nsc = len(scales)
    # cross terms K_{s's} = PSF ⊛ G(σ_s'² + σ_s²), times the blob areas
    kern, half = {}, {}
    for i in range(nsc):
        for j in range(i, nsc):
            seff = onp.hypot(sig[i], sig[j])
            w = int(min(max(patch, onp.ceil(2.5 * seff)), ctr[0], ctr[1]))
            ext = int(min(w + onp.ceil(4 * seff), ctr[0], ctr[1]))
            sub = psf[
                ctr[0] - ext : ctr[0] + ext + 1,
                ctr[1] - ext : ctr[1] + ext + 1,
            ]
            k = _Smoother(sub.shape)(sub, seff)
            k = k[ext - w : ext + w + 1, ext - w : ext + w + 1]
            kern[(i, j)] = kern[(j, i)] = _area(sig[i]) * _area(sig[j]) * k
            half[(i, j)] = half[(j, i)] = w
    hdiag = onp.array(
        [kern[(i, i)][half[(i, i)], half[(i, i)]] for i in range(nsc)]
    )
    # Jy/beam at its centre of a unit component's dirty response
    resp = onp.array(
        [
            _area(g) * (_Smoother(psf.shape)(psf, g)[ctr] if g else psf[ctr])
            for g in sig
        ]
    )
    norm = wbias / onp.sqrt(hdiag)
    es = onp.stack(
        [
            _area(sig[i]) * sm(residual[r0:r1, c0:c1], sig[i]) * norm[i]
            for i in range(nsc)
        ]
    )
    es *= cm[None]
    tile = 64  # tile maxima for a fast peak search
    py, px = -(-ny // tile) * tile, -(-nx // tile) * tile
    big = onp.zeros((nsc, py, px))
    big[:, :ny, :nx] = es

    def tile_max(a):
        t = a.reshape(a.shape[0], -1, tile, a.shape[-1] // tile, tile)
        return onp.abs(t).max(axis=(2, 4))

    tmax = tile_max(big)
    comps = onp.zeros((nsc, ny, nx))
    done, lowest, why = 0, onp.full(nsc, onp.inf), "niter"
    for done in range(1, niter + 1):
        s, ty, tx = onp.unravel_index(onp.argmax(tmax), tmax.shape)
        blk = big[s, ty * tile : (ty + 1) * tile, tx * tile : (tx + 1) * tile]
        iy, ix = onp.unravel_index(onp.argmax(onp.abs(blk)), blk.shape)
        y, x = ty * tile + iy, tx * tile + ix
        amp = big[s, y, x] / (norm[s] * hdiag[s])
        peak = abs(amp * resp[s])
        if peak < threshold or peak > (1 + _MINOR_RISE) * lowest[s]:
            done -= 1
            why = "threshold" if peak < threshold else "diverged"
            break
        lowest[s] = min(lowest[s], peak)
        a = gain * amp
        comps[s, y, x] += a
        for t in range(nsc):
            w, k = half[(s, t)], kern[(t, s)]
            ya, yb = max(y - w, 0), min(y + w + 1, ny)
            xa, xb = max(x - w, 0), min(x + w + 1, nx)
            kk = k[ya - (y - w) : yb - (y - w), xa - (x - w) : xb - (x - w)]
            reg = big[t, ya:yb, xa:xb]
            reg -= a * norm[t] * kk
            reg *= cm[ya:yb, xa:xb]
            ta, tb = ya // tile, -(-yb // tile)
            ua, ub = xa // tile, -(-xb // tile)
            tmax[t, ta:tb, ua:ub] = tile_max(
                big[t : t + 1, ta * tile : tb * tile, ua * tile : ub * tile]
            )[0]
    out = onp.zeros((nsc,) + residual.shape)
    out[:, r0:r1, c0:c1] = comps
    return out, done, why


# ----------------------------------------------------------------------
# Result and the major cycle
# ----------------------------------------------------------------------


@dataclasses.dataclass
class RadioCleanResult:
    """The products of [`clean`][virgil.radio_clean.clean].

    Images are ``(npix, npix)`` arrays with row 0 North and column 0 East
    (the orientation of [`Image`][virgil.models.Image]).

    Attributes
    ----------
    image : array
        Restored image, Jy/beam.
    model : array
        CLEAN model, Jy per pixel.
    residual : array
        Residual image of the final model, Jy/beam.
    psf : array
        Dirty beam, peak 1.
    beam : (bmaj_mas, bmin_mas, pa_deg)
        Restoring beam.
    pixel_scale_mas : float
    mask : array of bool
        Where components were allowed (the union over major cycles).
    iterations : int
        Minor-cycle iterations.
    stop : str
        ``"threshold"``, ``"nsigma"``, ``"niter"``, ``"time_limit"`` or
        ``"no_progress"`` (no component above the cycle threshold) or
        ``"diverged"`` (the peak residual grew by more than 3.0, relative, over
        the previous major cycle or the minimum so far, as in ``tclean``;
        ``model`` and ``residual`` are then those of the cycle that diverged).
    history : list of dict
        One entry per major cycle: iterations so far, peak residual (Jy/beam),
        rms (Jy/beam), model flux (Jy), wall time (s), and ``minor_stop``,
        why the minor cycle that followed ended (``"threshold"``,
        ``"diverged"`` or ``"niter"``).
    """

    image: onp.ndarray
    model: onp.ndarray
    residual: onp.ndarray
    psf: onp.ndarray
    beam: tuple
    pixel_scale_mas: float
    mask: onp.ndarray
    iterations: int
    stop: str
    history: list

    def model_visibilities(self, data, chunk_size=1_000_000):
        """The model at the (u, v) points of ``data``: ``(N,)`` complex (Jy).

        Pass to ``dataclasses.replace``/`eqx.tree_at` of ``data.vis`` to write
        it back, e.g. to subtract it (``data.vis - model``).
        """
        return _model_vis(
            self.model, data.u, data.v, self.pixel_scale_mas, chunk_size
        )

    def pb_corrected(self, pb, pblimit=0.2):
        """``image / pb`` where the primary-beam gain ``pb`` (power, peak 1,
        e.g. ``exp(primary_beam(...))``) is at least ``pblimit``, else NaN."""
        pb = onp.asarray(pb, float)
        return onp.where(
            pb >= pblimit, self.image / onp.maximum(pb, 1e-30), onp.nan
        )

    def write_fits(
        self, prefix, ra_deg, dec_deg, freq_hz=None, object_name=None
    ):
        """Write ``prefix.image.fits``, ``.model.fits``, ``.residual.fits`` and
        ``.psf.fits`` (the suffixes of ``tclean``).

        The headers hold a SIN WCS centred on the phase centre
        (``ra_deg``, ``dec_deg``), ``BUNIT`` (``Jy/beam``, or ``Jy/pixel`` for
        the model), ``BMAJ``/``BMIN``/``BPA``, and ``RESTFRQ`` if ``freq_hz``
        is given. Rows are stored South first, as FITS expects.
        """
        from astropy.io import fits
        from astropy.wcs import WCS

        n = self.image.shape[0]
        c = (n - 1) / 2
        wcs = WCS(naxis=2)
        wcs.wcs.ctype = ["RA---SIN", "DEC--SIN"]
        wcs.wcs.crval = [ra_deg, dec_deg]
        wcs.wcs.crpix = [c + 1, c + 1]
        step = self.pixel_scale_mas / 3.6e6
        wcs.wcs.cdelt = [-step, step]
        wcs.wcs.cunit = ["deg", "deg"]
        bmaj, bmin, pa = self.beam
        for name, im, unit in (
            ("image", self.image, "Jy/beam"),
            ("model", self.model, "Jy/pixel"),
            ("residual", self.residual, "Jy/beam"),
            ("psf", self.psf, ""),
        ):
            hdr = wcs.to_header()
            hdr["BUNIT"] = unit
            hdr["BMAJ"], hdr["BMIN"], hdr["BPA"] = (
                bmaj / 3.6e6,
                bmin / 3.6e6,
                pa,
            )
            if freq_hz is not None:
                hdr["RESTFRQ"] = float(freq_hz)
            if object_name is not None:
                hdr["OBJECT"] = object_name
            hdr["ORIGIN"] = "virgil.radio_clean"
            data = onp.asarray(im, onp.float32)[::-1]
            fits.PrimaryHDU(data, hdr).writeto(
                f"{prefix}.{name}.fits", overwrite=True
            )


def _model_vis(model, u, v, pixel_scale_mas, chunk_size):
    """Visibilities of a signed image at (u, v), in chunks (NUFFT)."""
    image = np.asarray(model)
    n = len(u)
    out = onp.empty(n, onp.complex128)
    for lo in range(0, n, int(chunk_size)):
        sl = slice(lo, lo + int(chunk_size))
        out[sl] = onp.asarray(
            image_visibilities(
                image,
                np.asarray(u[sl]),
                np.asarray(v[sl]),
                pixel_scale_mas,
                ft="nufft",
            )
        )
    return out


def clean(
    data,
    npix,
    pixel_scale_mas,
    *,
    deconvolver="hogbom",
    scales_mas=None,
    weighting="natural",
    robust=0.5,
    uvtaper=None,
    niter=1000,
    gain=0.1,
    threshold=0.0,
    nsigma=0.0,
    cycleniter=-1,
    cyclefactor=1.0,
    minpsffraction=0.05,
    maxpsffraction=0.8,
    smallscalebias=0.6,
    psfpatch=256,
    mask=None,
    usemask="user",
    sidelobethreshold=3.0,
    noisethreshold=5.0,
    lownoisethreshold=1.5,
    restoringbeam=None,
    psfcutoff=0.35,
    uv_cell=None,
    chunk_size=1_000_000,
    time_limit=None,
    verbose=False,
):
    """Cotton–Schwab CLEAN of complex visibilities, after CASA's ``tclean``.

    Major cycles image the exact residual ``V − M`` (NUFFT model); minor
    cycles clean it with a window of the PSF. Arguments are named as in
    ``tclean``; see the [guide](../radio_clean.md). Unlike
    [`virgil.imaging.clean`][virgil.imaging.clean] it takes complex
    visibilities and an **odd** ``npix``. Provisional (new in 0.4.1).

    Source: Högbom (1974), Clark (1980), Schwab (1984), Cornwell (2008), CASA.

    Parameters
    ----------
    data : ComplexVisData
    npix : int
        Odd image side, so the PSF peaks on a pixel (``Image`` wants even).
    pixel_scale_mas : float
    deconvolver : {"hogbom", "multiscale"}
        Point components or Gaussians (every minor cycle is Clark's windowed
        loop: no ``"clark"``).
    scales_mas : sequence of float, optional
        Multi-scale FWHMs in mas, 0 a point; default 0, 5 and 15 pixels.
    weighting, robust, uvtaper
        As for [`dirty_image`][virgil.radio.ComplexVisData.dirty_image].
    niter, gain, threshold : int, float, float
        Total iterations, loop gain, and the peak residual (Jy/beam) to stop at.
    nsigma : float
        Also stop below ``nsigma`` × the robust rms of the residual.
    cycleniter : int
        Iterations per minor cycle (−1: all that remain).
    cyclefactor, minpsffraction, maxpsffraction : float
        A minor cycle stops at ``max(threshold, f × peak)``, ``f =
        clip(cyclefactor × PSF sidelobe, minpsffraction, maxpsffraction)``.
    smallscalebias, psfpatch : float, int
        Preference for small scales (0 to 1); half-width in pixels of the PSF
        window subtracted per iteration, at most ``npix - 1``.
    mask : array of bool, optional
        Where to clean; default the whole image.
    usemask : {"user", "auto-multithresh"}, sidelobethreshold, noisethreshold,
        lownoisethreshold
        ``"auto-multithresh"`` adds regions of the beam-smoothed residual
        (a simplified CASA version, see the guide).
    restoringbeam : (bmaj_mas, bmin_mas, pa_deg), optional
        Default: fitted to the PSF above ``psfcutoff``. A given beam changes
        only the restoring kernel: the residual stays in dirty-beam Jy/beam,
        as in CASA.
    uv_cell, chunk_size, time_limit, verbose
        Density cell in wavelengths (default ``1/FoV``); visibilities
        transformed at a time; seconds after which no further major cycle
        starts; print progress.

    Returns
    -------
    RadioCleanResult
    """
    if not isinstance(data, ComplexVisData):
        raise TypeError("data must be a ComplexVisData.")
    npix = int(npix)
    if npix % 2 == 0:
        raise ValueError("npix must be odd, so the PSF peaks on a pixel.")
    if deconvolver == "clark":
        raise ValueError(
            "deconvolver='clark' was identical to 'hogbom' and is removed; "
            "use 'hogbom'."
        )
    if deconvolver not in ("hogbom", "multiscale"):
        raise ValueError("deconvolver must be 'hogbom' or 'multiscale'.")
    if usemask not in ("user", "auto-multithresh"):
        raise ValueError("usemask must be 'user' or 'auto-multithresh'.")
    if deconvolver == "multiscale":
        scales = (
            [0, 5, 15]
            if scales_mas is None
            else [float(m) / pixel_scale_mas for m in scales_mas]
        )
    else:
        scales = [0]
    shape = (npix, npix)
    mask = onp.ones(shape, bool) if mask is None else onp.asarray(mask, bool)
    if mask.shape != shape:
        raise ValueError("mask must have the shape of the image.")
    t0 = time.time()
    with jax.enable_x64(True):
        if uv_cell is None:
            uv_cell = 1.0 / (npix * pixel_scale_mas * mas2rad)
        w = data._imaging_weights(
            weighting, robust, uv_cell, None, None, uvtaper=uvtaper
        )
        vis = onp.asarray(data.vis, dtype=onp.complex128)

        def dirty(a):
            return data._adjoint(a, w, npix, pixel_scale_mas, chunk_size)

        ones = onp.ones(len(data), onp.complex128)
        psf = dirty(ones)
        # the minor cycle's PSF: every window it uses (up to psfpatch plus the
        # smoothing tails), and at most the whole image from any component
        seff = onp.sqrt(2.0) * max(scales) * FWHM_TO_SIGMA
        reach = min(
            npix - 1,
            int(max(psfpatch, onp.ceil(2.5 * seff)) + onp.ceil(4 * seff)),
        )
        psf2 = data._adjoint(
            ones, w, 2 * reach + 1, pixel_scale_mas, chunk_size
        )
        beam = fit_beam(psf, pixel_scale_mas, psfcutoff)
        lab, _ = ndimage.label(
            psf > 0
        )  # the main lobe, down to its first null
        peak = onp.unravel_index(onp.argmax(psf), psf.shape)
        side = float(onp.where(lab == lab[peak], 0.0, onp.abs(psf)).max())
        beam_pix = onp.sqrt(beam[0] * beam[1]) / pixel_scale_mas
        psf_frac = min(max(cyclefactor * side, minpsffraction), maxpsffraction)
        comps = onp.zeros((len(scales),) + shape)
        model = onp.zeros(shape)
        use = mask.copy() if usemask == "user" else onp.zeros(shape, bool)
        history, total, stop = [], 0, "niter"
        lowest_whole = onp.inf
        while True:
            if total:
                resid = dirty(
                    vis
                    - _model_vis(
                        model, data.u, data.v, pixel_scale_mas, chunk_size
                    )
                )
            else:
                resid = dirty(vis)
            if usemask == "auto-multithresh":
                use |= mask & _auto_mask(
                    resid,
                    beam_pix,
                    side,
                    sidelobethreshold,
                    noisethreshold,
                    lownoisethreshold,
                )
            if not use.any():
                stop = "no_progress"
                break
            rms = _mad_rms(resid)
            peak = float(onp.abs(resid[use]).max())
            history.append(
                dict(
                    iterations=total,
                    peak_residual=peak,
                    rms=rms,
                    model_flux=float(model.sum()),
                    wall_s=time.time() - t0,
                )
            )
            if verbose:
                print(
                    f"major {len(history) - 1}: {total} iterations, peak "
                    f"{peak:.3e}, rms {rms:.3e}, model {model.sum():.5f} Jy"
                )
            floor = max(threshold, nsigma * rms)
            if peak <= floor:  # <=: all-zero data stops at threshold=0
                stop = "nsigma" if nsigma * rms > threshold else "threshold"
                break
            whole = float(onp.abs(resid).max())
            if whole > (1 + _MAJOR_RISE) * lowest_whole:
                stop = "diverged"
                break
            lowest_whole = min(lowest_whole, whole)
            if total >= niter:
                break
            if time_limit and time.time() - t0 > time_limit:
                stop = "time_limit"
                break
            n_max = niter - total
            if cycleniter > 0:
                n_max = min(n_max, cycleniter)
            inc, n, why = _minor_cycle(
                resid,
                psf2,
                scales,
                use,
                gain,
                n_max,
                max(floor, psf_frac * peak),
                smallscalebias,
                psfpatch,
            )
            history[-1]["minor_stop"] = why
            if n == 0:
                stop = "no_progress"
                break
            comps += inc
            total += n
            sm = _Smoother(shape)
            model = onp.zeros(shape)  # Jy/pixel: Σ_s B_s ⊛ comps_s, B_s peak 1
            for c, g in zip(comps, onp.array(scales) * FWHM_TO_SIGMA):
                if onp.any(c):
                    model += _area(g) * sm(c, g)
    if restoringbeam is not None:
        beam = tuple(float(b) for b in restoringbeam)
    image = restore(model, resid, beam, pixel_scale_mas)
    return RadioCleanResult(
        image=image,
        model=model,
        residual=resid,
        psf=psf,
        beam=beam,
        pixel_scale_mas=float(pixel_scale_mas),
        mask=use,
        iterations=total,
        stop=stop,
        history=history,
    )
