"""Regularizers and helpers for image reconstruction.

An image fit is a call to [`fit`][virgil.fitting.fit] whose model contains an
[`Image`][virgil.models.Image], with a prior on its ``log_brightness``
(see :func:`image_priors`) and usually a regularizer, which adds a penalty
on the pixel fluxes ``b`` to the loss:

| Regularizer | Penalty | Least squares (LM)? | A prior density? |
| --- | --- | --- | --- |
| [`TSV`][virgil.imaging.TSV] | ``w Σ (Δx b)² + (Δy b)²`` | yes | no |
| [`TV`][virgil.imaging.TV] | ``w Σ √((Δx b)² + (Δy b)² + ε²)`` | no | no |
| [`MaxEntropy`][virgil.imaging.MaxEntropy] | ``w Σ b log(b / q)`` | no | no |
| [`Laplacian`][virgil.imaging.Laplacian] | ``w Σ (∇²b)²`` | yes | no |
| [`StarletL1`][virgil.imaging.StarletL1] | ``w Σ √(s² + ε²)`` over starlet details ``s`` | no | no |
| [`LogSum`][virgil.imaging.LogSum] | ``w Σ log(1 + b / (ε b̄))`` | no | no |
| [`Centroid`][virgil.imaging.Centroid] | ``½ |centroid / σ|²`` | yes | yes |

Differences ``Δ`` are between neighbouring pixels, with zeros beyond the
edges. TSV and the Laplacian favour smooth images, TV piecewise-flat ones,
maximum entropy images close to a default ``q``; ``StarletL1`` (an L1 norm
of the starlet coefficients; an L1 norm of the pixels is constant, since
they sum to one) and ``LogSum`` favour sparse ones, and ``LogSum`` is not
convex, so start it from a [`clean`][virgil.imaging.clean] model.
[`l_curve`][virgil.imaging.l_curve] sweeps the weight ``w``.

When [`fit`][virgil.fitting.fit]'s model function returns one model per
dataset, every regularizer acts on the **first model only**: right for
transformed copies of one scene (e.g. a [`Rotated`][virgil.models.Rotated]
epoch), but an Image that appears only in a later model is not regularized.

Closure, kernel and DISCO phases do not fix an image's position: something
must (an analytic star at the origin, a [`Centroid`][virgil.imaging.Centroid]
prior, or a centred prior mean). [`diagnose`][virgil.imaging.diagnose]
checks a fit for this and other common pitfalls.
"""

import dataclasses
import warnings

import equinox as eqx
import jax
import jax.numpy as np
import numpy as onp
from jax.scipy.signal import fftconvolve
from jax.scipy.special import xlogy

from ._geometry import fringe_scales, pixel_offsets, rotate, smearing_scale
from ._utils import FWHM_PER_SIGMA, _reference, mas2rad
from ._deprecate import external_stacklevel, keyword_only, renamed
from ._precision import cast_tree, run_in
from .fitting import FitResult, fit
from .spectra import flux_at
from .likelihood import _whitened_errors_and_log_norm, whitened_residuals
from .models import (
    _scene_images,
    Image,
    PointSource,
    Rotated,
    SourceModel,
    System,
    _pixel_visibilities,
    circular_support,
    smeared_model,
)

__provisional__ = (
    "clean",
    "CleanResult",
)

_HIGHEST = jax.lax.Precision.HIGHEST


def _as_list(data):
    return list(data) if isinstance(data, (list, tuple)) else [data]


class _ImageRegulariser(eqx.Module):
    """A penalty on the pixels of one Image, found at ``path`` in the model."""

    weight: float
    path: str | None = eqx.field(static=True)
    smoothed: bool = eqx.field(static=True, default=False)

    def pixels(self, model):
        """The pixels penalized: the raw ``brightness``, or, with the
        ``smoothed`` option, the Image's ``smoothed()`` pixels (MPoL's
        convention), floored at the smallest positive float: FFT round-off
        leaves tiny negatives, and an exact zero would give NaN gradients in
        ``xlogy``."""
        image = self.image(model)
        if not self.smoothed:
            return image.brightness
        if image.blur_fwhm_mas is not None:
            warnings.warn(
                "smoothed=True penalizes the blurred image, so the base "
                "image is unregularized unless another regularizer "
                "penalizes the raw pixels (MPoL's convention).",
                UserWarning,
                stacklevel=external_stacklevel(),
            )
        b = image.smoothed()
        return np.maximum(b, np.finfo(b.dtype).tiny)

    def image(self, model):
        """The regularized [`Image`][virgil.models.Image] in ``model``."""
        model = model.resolved()
        image = model if self.path is None else model.get(self.path)
        if not isinstance(image, Image):
            raise TypeError(
                f"{type(self).__name__}(path={self.path!r}) found a "
                f"{type(image).__name__}, not an Image."
            )
        return image

    probabilistic = False


def _differences(b):
    """Steps between neighbouring pixels along rows and columns.

    The image is padded with a ring of zeros, so the steps into and out of
    the image at every edge count. Both arrays have shape ``(N + 1, M + 1)``
    and line up, pixel by pixel, for total variation.
    """
    padded = np.pad(b, 1)
    dx = padded[:-1, 1:] - padded[:-1, :-1]
    dy = padded[1:, :-1] - padded[:-1, :-1]
    return dx, dy


class TSV(_ImageRegulariser):
    """Total squared variation: ``weight * Σ (Δx b)² + (Δy b)²``.

    A quadratic smoothness penalty, so it can be fitted by Levenberg–Marquardt
    (Kuramochi et al. 2018, ApJ 858, 56).

    Parameters
    ----------
    weight : float
        Strength of the penalty.
    path : str, optional
        Path of the Image in the model (e.g. ``"env"``); ``None`` if the model
        is the Image itself.
    smoothed : bool, optional
        Penalize the Image's ``smoothed()`` pixels (see ``blur_fwhm_mas``)
        instead of the raw ones. Default ``False``.
    """

    def __init__(self, weight, path=None, smoothed=False):
        self.weight = np.asarray(weight, dtype=float)
        self.path = path
        self.smoothed = bool(smoothed)

    def residuals(self, model):
        dx, dy = _differences(self.pixels(model))
        return np.sqrt(2.0 * self.weight) * np.concatenate(
            [np.ravel(dx), np.ravel(dy)]
        )

    def value(self, model):
        return 0.5 * np.sum(self.residuals(model) ** 2)


class TV(_ImageRegulariser):
    """Total variation: ``weight * Σ √((Δx b)² + (Δy b)² + ε²)``.

    Favours piecewise-flat images with sharp edges (Rudin, Osher & Fatemi 1992,
    Physica D 60, 259). ``ε`` makes it differentiable where the image is flat.
    It is averaged over the four flips of the image, so it does not depend on
    which neighbour each forward difference pairs a pixel with. It is not
    isotropic: a sharp edge along a diagonal costs about a fifth more per unit
    length than one along a pixel axis.

    Parameters
    ----------
    weight : float
        Strength of the penalty.
    epsilon : float, optional
        Smoothing scale, as a fraction of the mean pixel flux (default 1e-2).
    path : str, optional
        Path of the Image in the model.
    smoothed : bool, optional
        Penalize the Image's ``smoothed()`` pixels instead of the raw ones
        (default ``False``).
    """

    epsilon: float

    def __init__(self, weight, epsilon=1e-2, path=None, smoothed=False):
        self.weight = np.asarray(weight, dtype=float)
        self.epsilon = np.asarray(epsilon, dtype=float)
        self.path = path
        self.smoothed = bool(smoothed)

    def value(self, model):
        b = self.pixels(model)
        eps = self.epsilon / b.size

        def total(image):
            dx, dy = _differences(image)
            return np.sum(np.sqrt(dx**2 + dy**2 + eps**2))

        # Forward differences pair each step with a preferred neighbour;
        # averaging over the four flips of the image removes that bias.
        flips = [b, b[::-1], b[:, ::-1], b[::-1, ::-1]]
        return self.weight * sum(total(f) for f in flips) / 4.0


class MaxEntropy(_ImageRegulariser):
    """Maximum entropy: ``weight * Σ b log(b / q)``, with ``b`` and ``q`` unit-sum.

    The relative entropy of the image with respect to a default image ``q``
    (Skilling & Bryan 1984, MNRAS 211, 111; BSMEM, Baron & Young 2008).

    Parameters
    ----------
    weight : float
        Strength of the penalty.
    prior : array-like, optional
        The default image ``q``, with the image's shape and positive where the
        image's support is; normalized here. Default: flat over the support.
    path : str, optional
        Path of the Image in the model.
    smoothed : bool, optional
        Penalize the Image's ``smoothed()`` pixels instead of the raw ones
        (default ``False``).
    """

    prior: object
    prefers_lbfgs = True  # LM converges in float64 only: see ``fit``

    def __init__(self, weight, prior=None, path=None, smoothed=False):
        self.weight = np.asarray(weight, dtype=float)
        self.prior = None if prior is None else np.asarray(prior)
        self.path = path
        self.smoothed = bool(smoothed)

    def _brightness_and_default(self, model):
        """The unit-sum image ``b`` and the default ``q`` (1 where ``b`` is 0)."""
        image = self.image(model)
        b = self.pixels(model)
        if self.prior is None:
            # Smoothing spreads flux beyond the support: flat over all pixels.
            inside = (
                np.ones(b.shape, bool)
                if image.support is None or self.smoothed
                else image.support
            )
            q = inside / np.sum(inside)
        else:
            q = np.asarray(self.prior) / np.sum(self.prior)
        return b, np.where(b > 0.0, q, 1.0)  # 0 log 0 = 0 outside the support

    def value(self, model):
        b, q = self._brightness_and_default(model)
        return self.weight * np.sum(xlogy(b, b) - xlogy(b, q))

    def residuals(self, model):
        """Residuals ``r`` with ``value = 0.5 Σ r²`` (``b`` and ``q`` sum to 1):
        ``r = t √(2 w q φ(t))`` for ``t = (b - q) / q`` and
        ``φ(t) = ((1 + t) ln(1 + t) - t) / t²``, so that the entropy can be
        fitted by Levenberg–Marquardt. Zero where ``b`` is."""
        b, q = self._brightness_and_default(model)
        inside = b > 0.0
        t = np.where(inside, (b - q) / q, 0.5)
        small = np.abs(t) < 1e-2
        ts = np.where(small, 1.0, t)  # the series covers the small ones
        phi = np.where(
            small,
            0.5 - t / 6.0 + t**2 / 12.0 - t**3 / 20.0,
            ((1.0 + ts) * np.log1p(ts) - ts) / ts**2,
        )
        r = t * np.sqrt(2.0 * self.weight * q * phi)
        return np.ravel(np.where(inside, r, 0.0))

    def metric_rows(self, model):
        """Rows whose JᵀJ is the curvature ``w b`` of the entropy in
        ``ln b`` (the Skilling–Bryan metric): a positive Gauss–Newton form.
        In the Image's softmax logits its diagonal is ``w b (1 - b)``."""
        b = self.pixels(model)
        return np.ravel(2.0 * np.sqrt(self.weight * np.maximum(b, 1e-30)))


class Laplacian(_ImageRegulariser):
    """Squared Laplacian: ``weight * Σ (∇²b)²``.

    ``∇²b`` is the five-point Laplacian, with zeros beyond the edges. A
    quadratic penalty on curvature, smoother than TSV, so it can be fitted by
    Levenberg–Marquardt (Thiébaut & Young 2017).

    Parameters
    ----------
    weight : float
        Strength of the penalty.
    path : str, optional
        Path of the Image in the model.
    smoothed : bool, optional
        Penalize the Image's ``smoothed()`` pixels instead of the raw ones
        (default ``False``).
    """

    def __init__(self, weight, path=None, smoothed=False):
        self.weight = np.asarray(weight, dtype=float)
        self.path = path
        self.smoothed = bool(smoothed)

    def residuals(self, model):
        padded = np.pad(self.pixels(model), 2)
        laplacian = (
            padded[:-2, 1:-1]
            + padded[2:, 1:-1]
            + padded[1:-1, :-2]
            + padded[1:-1, 2:]
            - 4.0 * padded[1:-1, 1:-1]
        )
        return np.sqrt(2.0 * self.weight) * np.ravel(laplacian)

    def value(self, model):
        return 0.5 * np.sum(self.residuals(model) ** 2)


_B3_SPLINE = (1 / 16, 4 / 16, 6 / 16, 4 / 16, 1 / 16)


def _smooth(image, step):  # deliberate: named step of starlet
    """The B3-spline smoothing of the starlet, with taps ``step`` apart."""
    for axis in (0, 1):
        n = image.shape[axis]
        width = [(0, 0), (0, 0)]
        width[axis] = (2 * step, 2 * step)
        padded = np.pad(image, width)
        image = sum(
            tap * jax.lax.slice_in_dim(padded, k * step, k * step + n, 1, axis)
            for k, tap in enumerate(_B3_SPLINE)
        )
    return image


def starlet(image, scales=4):
    """The starlet (isotropic undecimated wavelet) transform of an image.

    The à-trous algorithm with a B3-spline kernel (Starck, Murtagh & Fadili
    2010): the image is smoothed ``scales`` times with the kernel's taps twice
    as far apart each time, and each detail plane is the difference between
    successive smoothings. Plane ``j`` holds structure about ``2**j`` pixels
    across. Beyond the edges the image is zero.

    Parameters
    ----------
    image : array-like, shape (ny, nx)
        The image.
    scales : int, optional
        Number of detail planes (default 4).

    Returns
    -------
    details : jax.Array, shape (scales, ny, nx)
        The detail planes, finest first.
    coarse : jax.Array, shape (ny, nx)
        What is left after the last smoothing; ``details.sum(0) + coarse`` is
        the image.
    """
    if int(scales) != scales or scales < 1:
        raise ValueError(f"scales must be a positive integer, not {scales}.")
    smooth = np.asarray(image)
    details = []
    for j in range(int(scales)):
        smoother = _smooth(smooth, 2**j)
        details.append(smooth - smoother)
        smooth = smoother
    return np.stack(details), smooth


class StarletL1(_ImageRegulariser):
    """L1 norm of the starlet details: ``weight * Σ √(s² + ε²)``.

    ``s`` are the detail coefficients of [`starlet`][virgil.imaging.starlet] at
    every scale. An L1 norm favours few non-zero coefficients, so the image is
    built from few compact structures of any size (Honma et al. 2014, PASJ 66,
    95). The coarse plane, which carries the flux, is not penalized.

    Parameters
    ----------
    weight : float
        Strength of the penalty.
    scales : int, optional
        Number of detail planes (default 4); the largest holds structure about
        ``2**(scales - 1)`` pixels across.
    epsilon : float, optional
        Smoothing scale, as a fraction of the mean pixel flux (default 1e-2).
    path : str, optional
        Path of the Image in the model.
    smoothed : bool, optional
        Penalize the Image's ``smoothed()`` pixels instead of the raw ones
        (default ``False``).
    """

    scales: int = eqx.field(static=True)
    epsilon: float

    def __init__(
        self, weight, scales=4, epsilon=1e-2, path=None, smoothed=False
    ):
        if int(scales) != scales or scales < 1:
            raise ValueError(
                f"scales must be a positive integer, not {scales}."
            )
        self.weight = np.asarray(weight, dtype=float)
        self.scales = int(scales)
        self.epsilon = np.asarray(epsilon, dtype=float)
        self.path = path
        self.smoothed = bool(smoothed)

    def value(self, model):
        b = self.pixels(model)
        details, _ = starlet(b, self.scales)
        eps = self.epsilon / b.size
        return self.weight * np.sum(np.sqrt(details**2 + eps**2))


class LogSum(_ImageRegulariser):
    """Log-sum sparsity: ``weight * Σ log(1 + b / (ε b̄))``.

    ``b̄`` is the mean pixel flux over the support. As ``ε → 0`` the penalty
    counts the bright pixels: a smooth surrogate for the L0 "norm" of SQUEEZE
    (Candès, Wakin & Boyd 2008; Baron, Monnier & Kloppenburg 2010). It is not
    convex, so start the fit from a good image, and sweep its weight with
    ``l_curve(..., warm_start=False)``: pixels a strong weight has driven dark
    cannot recover in the weaker fits that follow.

    Parameters
    ----------
    weight : float
        Strength of the penalty.
    epsilon : float, optional
        The flux, as a fraction of the mean pixel flux, at which a pixel starts
        to count (default 1e-2).
    path : str, optional
        Path of the Image in the model.
    smoothed : bool, optional
        Penalize the Image's ``smoothed()`` pixels instead of the raw ones
        (default ``False``).
    """

    epsilon: float

    def __init__(self, weight, epsilon=1e-2, path=None, smoothed=False):
        self.weight = np.asarray(weight, dtype=float)
        self.epsilon = np.asarray(epsilon, dtype=float)
        self.path = path
        self.smoothed = bool(smoothed)

    def value(self, model):
        image = self.image(model)
        b = self.pixels(model)
        n = (
            b.size
            if image.support is None or self.smoothed
            else np.sum(image.support)
        )
        return self.weight * np.sum(np.log1p(b * n / self.epsilon))


class Centroid(_ImageRegulariser):
    """A Gaussian prior centring an Image's flux on the origin.

    The penalty is ``½ |c / sigma_mas|²``, where ``c`` is the flux-weighted
    centroid of the Image on the sky, including its ``dra``/``ddec``. This
    is a genuine prior density, so it may be used when sampling.

    Parameters
    ----------
    sigma_mas : float
        Standard deviation of the centroid, in milliarcseconds.
    path : str, optional
        Path of the Image in the model.
    """

    sigma_mas: float
    probabilistic = True

    def __init__(self, sigma_mas, path=None):
        self.weight = 1.0
        self.sigma_mas = np.asarray(sigma_mas, dtype=float)
        self.path = path

    def centroid(self, model):
        """The Image's flux-weighted centroid ``(dra, ddec)`` in mas."""
        image = self.image(model)
        b = image.brightness
        nrow, ncol = b.shape
        x = np.matmul(
            np.sum(b, axis=0),
            pixel_offsets(ncol, image.pixel_scale_mas),
            precision=_HIGHEST,
        )
        y = np.matmul(
            np.sum(b, axis=1),
            pixel_offsets(nrow, image.pixel_scale_mas),
            precision=_HIGHEST,
        )
        x, y = rotate(x, y, image.rotation_deg)
        return np.stack([x + image.dra, y + image.ddec])

    def residuals(self, model):
        return self.centroid(model) / self.sigma_mas

    def value(self, model):
        return 0.5 * np.sum(self.residuals(model) ** 2)


def image_priors(scene):
    """Priors on the log-brightness of every Image in a scene.

    Returns a priors dict for [`fit`][virgil.fitting.fit]. An Image with
    a plain log-brightness array gets a flat prior,
    ``{"env.log_brightness": ImproperUniform(...)}``, so that its pixels are
    constrained only by the data and the regularizers. An Image with a
    [`GaussianField`][virgil.fields.GaussianField] gets standard-normal
    priors on the field's latents, ``{"env.log_brightness.latent":
    Normal(0, 1)}``: the Gaussian-process prior, which needs no regularizer.
    A scene with several such Images (a compact and an extended component,
    say) gets one prior per Image.
    Add priors for any other free parameters (fluxes, offsets) to the dict.
    """
    import numpyro.distributions as dist

    priors = {}
    for prefix, image in _scene_images(scene):
        shape = image.log_brightness.shape
        if _whitened_field(image.log_brightness):
            priors[prefix + "log_brightness.latent"] = dist.Normal(
                np.zeros(shape), 1.0
            ).to_event(2)
        else:
            priors[prefix + "log_brightness"] = dist.ImproperUniform(
                dist.constraints.real, (), event_shape=shape
            )
    if not priors:
        raise ValueError("The scene contains no Image.")
    return priors


def nyquist_pixel_scale(data):
    """The largest pixel scale (mas) that samples the data's finest fringes.

    That is λ / (2 B) for the longest baseline B in units of wavelength,
    over the unflagged samples of ``data`` (an OIData or a sequence of
    them; see [`OIData.unflagged`][virgil.oidata.OIData.unflagged]). A
    flagged sample carries no signal, so its baseline does not count.
    Reconstructions normally use pixels 2–4 times smaller.
    """
    return 0.5 * fringe_scales(data)[0]


def field_of_view(data, largest_mas=500.0, *, aperture_mas=None):
    """The largest field of view (mas) that the data measure.

    The smallest of ``largest_mas`` (a cap you choose, 500 by default); the
    interferometric field, the largest λ / B over the unflagged samples with a
    non-zero baseline B (structure larger than that is not measured, and on a
    uv lattice a larger field would alias); the bandwidth-smearing field
    R λ / B_max, when the data carry channel widths (see
    [`smeared_model`][virgil.models.smeared_model]); and ``aperture_mas``, if
    given, the field of a single telescope, about λ / D. A ceiling, not a
    recommended size: [`image_field`][virgil.imaging.image_field] sizes an
    image from the data and uses it as the limit. Flagged samples are left out
    (see [`OIData.unflagged`][virgil.oidata.OIData.unflagged]).
    """
    ceiling = min(float(largest_mas), fringe_scales(data)[1])
    ceiling = min(ceiling, smearing_scale(data))
    if aperture_mas is not None:
        ceiling = min(ceiling, float(aperture_mas))
    return ceiling


def _squared_visibilities(d):  # deliberate: named step of second_moment_fwhm
    """Spatial frequency (cycles per radian), V² and error of the unflagged
    visibilities, as flat arrays; empty for data without amplitudes of their own.
    """
    empty = onp.zeros(0)
    if (
        d.vis_encoding != "split"
        or d.vis_mat is not None
        or d.uv_grid is not None
        or onp.size(d.vis) == 0
    ):
        return empty, empty, empty
    amplitude = onp.asarray(d.vis, float)
    error = onp.asarray(d.d_vis, float)
    if d.vis_mode == "v2":
        v2, sigma = amplitude, error
    elif d.vis_mode == "logamp":
        v2 = onp.exp(2.0 * amplitude)
        sigma = 2.0 * v2 * error
    else:
        v2, sigma = amplitude**2, 2.0 * onp.abs(amplitude) * error
    u = onp.ravel(onp.asarray(d.u, float))
    v = onp.ravel(onp.asarray(d.v, float))
    wavel = onp.broadcast_to(onp.ravel(onp.asarray(d.wavel, float)), u.shape)
    rho = onp.hypot(u, v) / wavel
    if d.vis_index is not None:
        rho = rho[onp.asarray(d.vis_index)]
    return rho, v2, sigma


def second_moment_fwhm(data, v2_min=0.3):
    """The FWHM (mas) of the Gaussian that the short baselines see.

    A circular Gaussian of FWHM θ has ``-ln V² = π² θ² ρ² / (2 ln 2)`` (ρ = B /
    λ, cycles per radian), and any normalized source has this form to second
    order in ρ. It is fitted by weighted least squares on the unflagged samples
    with ``V² ≥ v2_min``, needing no image and no fit. It is the rms width of
    the whole source, a star included, so it is meaningless when a bright
    unresolved component (flux fraction above about 0.55) keeps V² above
    ``v2_min`` on every baseline: use
    [`envelope_fwhm`][virgil.imaging.envelope_fwhm] there.

    Parameters
    ----------
    data : OIData or sequence of OIData
        Data with visibility amplitudes (V², |V| or log |V|); projected
        (kernel, DISCO) observables and phase-only data have none.
    v2_min : float, optional
        The smallest V² to use, 0.3 by default.

    Returns
    -------
    float or None
        The FWHM in mas: zero if V² does not fall with ρ (an unresolved
        source), and ``None`` if fewer than two unflagged samples have
        ``V² ≥ v2_min``.
    """
    observations = _as_list(data)
    parts = [_squared_visibilities(d) for d in observations]
    rho, v2, sigma = (onp.concatenate(x) for x in zip(*parts))
    use = onp.isfinite(v2) & onp.isfinite(sigma) & (v2 >= v2_min) & (rho > 0)
    if use.sum() < 2:
        return None
    rho, v2, sigma = rho[use], v2[use], sigma[use]
    y = -onp.log(v2)
    weight = (v2 / onp.maximum(sigma, 1e-12)) ** 2
    slope = onp.sum(weight * rho**2 * y) / onp.sum(weight * rho**4)
    if not slope > 0:
        return 0.0
    theta = onp.sqrt(2.0 * onp.log(2.0) * slope) / onp.pi
    return float(theta / mas2rad)


def image_field(
    data,
    fwhm_mas=None,
    *,
    field_fwhms=6.0,
    inner_elements=4.0,
    aperture_mas=None,
    largest_mas=500.0,
):
    """The width (mas) of an image field that suits the data.

    The field is ``field_fwhms × fwhm_mas`` (without ``fwhm_mas``, the
    [`second_moment_fwhm`][virgil.imaging.second_moment_fwhm]; with no
    visibility amplitudes to measure it, the ceiling), clipped to a floor of
    ``inner_elements × λ / (2 B_max)`` and a ceiling of
    [`field_of_view`][virgil.imaging.field_of_view] on the unflagged baselines
    (the ceiling wins if they cross).

    Parameters
    ----------
    data : OIData or sequence of OIData
        The data.
    fwhm_mas : float, optional
        The FWHM of the source in mas, to override the one from the
        visibilities.
    field_fwhms : float, optional
        The field in FWHMs of the source (6 by default; 3 to 8 are reasonable).
        A larger field spends pixels on empty sky, where flux can scatter in
        islands.
    inner_elements : float, optional
        The floor, in resolution elements λ / (2 B_max) (4 by default).
    aperture_mas : float, optional
        A telescope's field of view, as a further ceiling.
    largest_mas : float, optional
        A cap on the ceiling (500 by default, as for ``field_of_view``).

    Returns
    -------
    float
        The field in mas.
    """
    ceiling = field_of_view(
        data, largest_mas=largest_mas, aperture_mas=aperture_mas
    )
    if fwhm_mas is None:
        fwhm_mas = second_moment_fwhm(data)
    wanted = ceiling if fwhm_mas is None else field_fwhms * float(fwhm_mas)
    floor = inner_elements * nyquist_pixel_scale(data)
    return float(min(ceiling, max(floor, wanted)))


def _complex_visibilities(d):  # deliberate: named step of dirty_image
    """Visibility estimates at a dataset's uv samples and their 0/1 weights: the
    least-squares estimate from the modes for DISCO data, the visibilities
    themselves for amplitudes with absolute phases.
    """
    if d.vis_encoding == "mixed_log_complex":
        sigma = onp.asarray(d.d_vis)[:, None]
        operator = onp.concatenate(
            [onp.asarray(d.vis_mat), onp.asarray(d.phi_mat)], axis=1
        )
        logv = onp.linalg.lstsq(
            operator / sigma, onp.asarray(d.vis) / sigma[:, 0], rcond=1e-8
        )[0]
        n = operator.shape[1] // 2
        information = onp.sum((operator / sigma) ** 2, axis=0)
        informed = (
            information[:n] + information[n:]
        ) > 1e-3 * information.max()
        # The modes are blind to the total flux, so the log-amplitudes have
        # an arbitrary offset: fix it so that |V| = 1 on average on the
        # shortest informed baselines, as for any normalized source.
        rho = onp.hypot(onp.asarray(d.u), onp.asarray(d.v))[informed]
        shortest = rho <= onp.quantile(rho, 0.1)
        logv[:n] -= onp.mean(logv[:n][informed][shortest])
        return onp.exp(logv[:n] + 1j * logv[n:]), informed.astype(float)
    if (
        d.vis_encoding == "split"
        and not d.cp_flag
        and d.phi_mat is None
        and d.vis_mat is None
        and d.vis_index is None
        and d.phi_index is None
    ):
        amplitude = onp.asarray(d.vis)
        if d.vis_mode == "v2":
            amplitude = onp.sqrt(onp.maximum(amplitude, 0.0))
        elif d.vis_mode == "logamp":
            amplitude = onp.exp(amplitude)
        vis = amplitude * onp.exp(1j * onp.asarray(d.phi))
        return vis, onp.ones(vis.size)
    raise ValueError(
        "A dirty image needs complex visibilities: AMIGO DISCO data, or "
        "amplitudes with absolute phases for every sample (closure phases "
        "do not give the phases, and visibility-only data have none)."
    )


# Samples per chunk in ``dirty_image``: ~16 MB per complex128 phase matrix at
# npix = 256.
_DIRTY_BATCH = 4096


def dirty_image(data, npix, pixel_scale_mas, flux_ratio=None):
    """The dirty image: direct synthesis of the visibilities, unregularized.

    ``I(x) = Σ_k w_k Re[V_k exp(+2πi u_k · x)] / Σ_k w_k``, with uniform weights
    on the informed samples: the image convolved with the dirty beam, plus
    noise; it peaks at one for a lone point source. For AMIGO DISCO data the
    visibilities are the least-squares estimate from the modes, with |V| = 1 on
    average on the shortest baselines (the modes are blind to the total flux).

    Parameters
    ----------
    data : OIData or sequence of OIData
        The data; closure phases alone cannot make a dirty image
        (``ValueError``).
    npix : int
        Number of pixels on a side.
    pixel_scale_mas : float
        Pixel size in mas.
    flux_ratio : float, optional
        Flux of extended emission relative to a star at the origin: the
        best-fitting point source at the origin is subtracted and the image
        scaled by ``(1 + flux_ratio) / flux_ratio``, so the map shows the
        emission alone.

    Returns
    -------
    jax.Array, shape (npix, npix)
        In the orientation of [`render`][virgil.models.SourceModel.render]
        (East left, North up). It has negative sidelobes.
    """
    observations = _as_list(data)
    offsets = onp.asarray(pixel_offsets(int(npix), float(pixel_scale_mas)))
    image, total = onp.zeros((int(npix), int(npix))), 0.0
    for d in observations:
        vis, weight = _complex_visibilities(d)
        if flux_ratio is not None:
            star = onp.sum(weight * onp.real(vis)) / onp.sum(weight)
            vis = (vis - star) * (1.0 + flux_ratio) / flux_ratio
        fu = onp.ravel(onp.asarray(d.u / d.wavel) * mas2rad)
        fv = onp.ravel(onp.asarray(d.v / d.wavel) * mas2rad)
        # In chunks of samples: all k at once is hundreds of GB for DISCO
        # data (k ~ 1e5, npix ~ 256).
        wv = weight * vis
        for lo in range(0, fu.size, _DIRTY_BATCH):
            sl = slice(lo, lo + _DIRTY_BATCH)
            cols = onp.exp(2j * onp.pi * onp.outer(fu[sl], offsets))
            rows = onp.exp(2j * onp.pi * onp.outer(fv[sl], offsets))
            image += onp.real((rows * wv[sl, None]).T @ cols)  # (row, col)
        total += weight.sum()
    return np.asarray(image / total)


def _envelope_fit(data, star, aperture_mas):
    """The fitted envelope and the FWHM (mas) that sizes the image field: the
    envelope's or, if larger, the second-moment FWHM, which guards against a
    fit that collapsed onto the star.
    """
    import numpyro.distributions as dist

    from .models import GaussianDisk

    resolution = beam(data).major_mas
    widest = field_of_view(
        data, largest_mas=onp.inf, aperture_mas=aperture_mas
    )
    best = None
    for width in (0.25 * resolution, resolution, 4.0 * resolution):
        sigma = min(width, widest / 6.0) / FWHM_PER_SIGMA
        if star:
            model = System(
                star=PointSource(), env=GaussianDisk(sigma, flux=0.1)
            )
            priors = {
                "env.sigma": dist.LogUniform(1e-3 * resolution, widest),
                "env.flux": dist.LogUniform(1e-4, 100.0),
            }
        else:
            model = GaussianDisk(sigma)
            priors = {"sigma": dist.LogUniform(1e-3 * resolution, widest)}
        result = fit(model, priors, data)
        if best is None or sum(result.info["chi2"]) < sum(best.info["chi2"]):
            best = result
    envelope = best.model.env if star else best.model
    fwhm = FWHM_PER_SIGMA * float(envelope.sigma)
    moment = second_moment_fwhm(data)
    return envelope, fwhm if moment is None else max(fwhm, moment)


def envelope_fwhm(data, star=True, aperture_mas=None):
    """The FWHM (mas) of the source's envelope, from a quick parametric fit.

    This is the width that [`starting_image`][virgil.imaging.starting_image]
    sizes its field from: a star (if ``star``) plus a Gaussian envelope is
    fitted, and the FWHM is the envelope's or, if larger, the one the short
    baselines see. Unlike
    [`second_moment_fwhm`][virgil.imaging.second_moment_fwhm] alone it is
    not diluted by a bright star. Pass it to
    [`image_field`][virgil.imaging.image_field].
    """
    return _envelope_fit(data, star, aperture_mas)[1]


def _pad_brightness(brightness, npix):
    """``brightness`` zero-padded symmetrically to ``npix`` x ``npix`` (the
    difference from its own size must be even, to keep the centre pixel)."""
    margin = (npix - brightness.shape[0]) // 2
    return np.pad(brightness, margin)


@keyword_only("star", "oversample", "largest_mas", "start", "hole_mas")
def starting_image(
    data,
    *,
    star=True,
    oversample=4.0,
    largest_mas=None,
    start="moments",
    hole_mas=None,
    field_fwhms=6.0,
    inner_elements=4.0,
    aperture_mas=None,
    pad_to_mas=None,
):
    """A starting model for an image fit, sized from the data.

    Fits an analytic star (if ``star``) plus a circular Gaussian envelope to the
    data, then builds an image of ``field_fwhms`` envelope FWHMs (capped by
    ``largest_mas`` and [`field_of_view`][virgil.imaging.field_of_view]), with
    pixels ``oversample`` times finer than Nyquist. Fit it with
    ``fit(start, image_priors(start), data, regularizers)``; see
    [imaging with regularizers](../imaging_rml.md). With ``star=True`` the
    envelope's flux ratio is fixed unless you add a prior on ``"env.flux"`` to
    ``image_priors(start)`` (a lone Image is normalized, so has no flux).

    Parameters
    ----------
    data : OIData or sequence of OIData
        The data.
    star : bool, optional
        Whether an unresolved star at the origin is in the scene (it fixes the
        position; otherwise use a [`Centroid`][virgil.imaging.Centroid] prior).
    oversample : float, optional
        Pixels per Nyquist pixel.
    largest_mas : float, optional
        Cap on the data-sized field, which never exceeds λ / B_min.
    start : {"moments", "dirty"}, optional
        Starting pixels: the fitted Gaussian, or the positive part of the
        [`dirty_image`][virgil.imaging.dirty_image] minus the star.
    hole_mas : float, optional
        Radius of a hole in the support under the star
        (``0.5 * beam(data).minor_mas`` works well).
    field_fwhms, inner_elements : float, optional
        Field in envelope FWHMs; least field in resolution elements λ / (2 B_max).
    aperture_mas : float, optional
        A telescope's field of view, as a ceiling on field and envelope width.
    pad_to_mas : float, optional
        Least grid width (mas), padded with empty pixels; may exceed λ / B_min,
        but beyond it the data do not constrain the flux, so pair a large field
        with a support or priors.

    Returns
    -------
    SourceModel
        ``System(star=PointSource(), env=Image(...))``, or the Image alone.
    """
    from .models import GaussianDisk

    envelope, fwhm = _envelope_fit(data, star, aperture_mas)
    fov = image_field(
        data,
        fwhm_mas=fwhm,
        field_fwhms=field_fwhms,
        inner_elements=inner_elements,
        aperture_mas=aperture_mas,
        largest_mas=onp.inf if largest_mas is None else largest_mas,
    )
    if largest_mas is not None:
        limit = field_of_view(
            data, largest_mas=onp.inf, aperture_mas=aperture_mas
        )
        if largest_mas > limit and fov >= limit * (1.0 - 1e-9):
            warnings.warn(
                f"largest_mas={largest_mas:g} is a cap and cannot raise the "
                f"field above the data's field of view, field_of_view(data) "
                f"= {limit:.4g} mas (the smallest of λ/B_min, the bandwidth-"
                "smearing field and aperture_mas), which limits the field "
                "here. Pass pad_to_mas to pad the start image to a larger "
                "grid; the data do not constrain flux beyond that field.",
                UserWarning,
                stacklevel=external_stacklevel(),
            )
    scale = nyquist_pixel_scale(data) / float(oversample)
    natural = int(onp.ceil(fov / scale))
    npix = natural
    if pad_to_mas is not None and pad_to_mas > natural * scale:
        npix = int(onp.ceil(pad_to_mas / scale))
        npix += (npix - natural) % 2
    support = None
    if hole_mas is not None:
        support = circular_support(npix, scale, npix * scale, hole_mas)
    options = dict(flux=envelope.flux, support=support)
    if start == "moments":
        # An unresolved envelope drives sigma to its lower bound, and a
        # Gaussian narrower than a pixel samples to zeros: start one pixel wide.
        model = GaussianDisk(max(float(envelope.sigma), scale))
        # On the larger grid the Gaussian simply continues to its (tiny) tails.
        image = Image.from_model(model, npix, scale, **options)
    elif start == "dirty":
        ratio = float(envelope.flux) if star else None
        dirty = dirty_image(data, natural, scale, flux_ratio=ratio)
        positive = _pad_brightness(np.maximum(dirty, 0.0), npix)
        image = Image.from_brightness(positive, scale, floor=1e-3, **options)
    else:
        raise ValueError(f"start must be 'moments' or 'dirty', not {start!r}.")
    return System(star=PointSource(), env=image) if star else image


@dataclasses.dataclass(frozen=True)
class Beam:
    """A Gaussian approximation to the core of the dirty beam.

    Attributes
    ----------
    major_mas, minor_mas : float
        Full widths at half maximum along the major and minor axes, in mas.
    pa_deg : float
        Position angle of the major axis, North to East, in degrees.
    """

    major_mas: float
    minor_mas: float
    pa_deg: float


def beam(data):
    """The resolution of a dataset: the FWHM ellipse of its beam's core.

    The dirty beam falls off near its peak as ``1 - 2π² xᵀ M x``, with ``M`` the
    weighted mean of ``u uᵀ`` over the samples; this matches a Gaussian of
    covariance ``M⁻¹ / 4π²``, whose FWHM ellipse is the usual "beam", roughly
    λ/B and elongated where the coverage is. Every informative sample has equal
    weight ("uniform weighting"): in AMIGO DISCO data, the uv points where the
    modes have weight (at least 1e-3 of the largest); otherwise the unflagged
    samples (see [`OIData.unflagged`][virgil.oidata.OIData.unflagged]), since a
    flagged sample measures nothing and counting it would claim a resolution
    the data do not have.

    Parameters
    ----------
    data : OIData or sequence of OIData
        The data.

    Returns
    -------
    Beam
    """
    observations = _as_list(data)
    u, v, w = [], [], []
    for d in observations:
        fu = onp.ravel(onp.asarray(d.u / d.wavel) * mas2rad)
        fv = onp.ravel(onp.asarray(d.v / d.wavel) * mas2rad)
        weight = onp.asarray(d.unflagged, dtype=float)
        if d.vis_encoding == "mixed_log_complex":
            sigma = onp.asarray(d.d_vis)[:, None]
            information = onp.sum(
                (onp.asarray(d.vis_mat) ** 2 + onp.asarray(d.phi_mat) ** 2)
                / sigma**2,
                axis=0,
            )
            weight = (information > 1e-3 * information.max()).astype(float)
        u.append(fu), v.append(fv), w.append(weight)
    u, v, w = (onp.concatenate(x) for x in (u, v, w))
    m = onp.array([[w @ (u * u), w @ (u * v)], [w @ (u * v), w @ (v * v)]])
    covariance = onp.linalg.inv(m / onp.sum(w)) / (4 * onp.pi**2)
    variance, axes = onp.linalg.eigh(covariance)
    fwhm = FWHM_PER_SIGMA * onp.sqrt(variance)
    x, y = axes[:, 1]  # the major axis, in (East, North)
    pa = onp.degrees(onp.arctan2(x, y)) % 180.0
    return Beam(float(fwhm[1]), float(fwhm[0]), float(pa))


def convolve_beam(image, pixel_scale_mas, beam):
    """An image convolved with a Gaussian beam: the image at the data's resolution.

    The "restored" image of radio astronomy: it shows only what the data
    resolve, so convolve both reconstructions to compare them. The kernel is
    an elliptical Gaussian with the beam's FWHMs and position angle (North to
    East), normalized to unit sum on an odd grid centred on a pixel. Flux
    beyond the edge is taken to be zero, so structure within about a beam of
    the edge is dimmed.

    Parameters
    ----------
    image : array-like, shape (ny, nx)
        The image, in the orientation of
        [`render`][virgil.models.SourceModel.render] (East left, North up).
    pixel_scale_mas : float
        Pixel size in milliarcseconds.
    beam : Beam
        The beam (FWHMs in mas, position angle in degrees), usually
        [`beam(data)`][virgil.imaging.beam].

    Returns
    -------
    jax.Array, shape (ny, nx)
        The convolved image.
    """
    image = np.asarray(image)
    image = image.astype(np.promote_types(image.dtype, np.float32))
    if image.ndim != 2:
        raise ValueError(f"image must be 2D, not of shape {image.shape}.")
    pixel_scale_mas = float(pixel_scale_mas)
    if not (onp.isfinite(pixel_scale_mas) and pixel_scale_mas > 0):
        raise ValueError(
            f"pixel_scale_mas must be finite and positive, not "
            f"{pixel_scale_mas}."
        )
    widths = onp.array([beam.major_mas, beam.minor_mas], dtype=float)
    if not (onp.all(onp.isfinite(widths)) and onp.all(widths > 0)):
        raise ValueError(
            f"The beam's FWHMs must be finite and positive: {beam}."
        )
    if not onp.isfinite(beam.pa_deg):
        raise ValueError(f"The beam's PA must be finite: {beam}.")
    ny, nx = image.shape
    x = pixel_offsets(nx + 1 - nx % 2, pixel_scale_mas)[None, :]  # East
    y = pixel_offsets(ny + 1 - ny % 2, pixel_scale_mas)[:, None]  # North
    pa = np.deg2rad(beam.pa_deg)
    along = x * np.sin(pa) + y * np.cos(pa)  # the major axis is (sin, cos)
    across = x * np.cos(pa) - y * np.sin(pa)
    kernel = np.exp(
        -0.5
        * (
            (along * FWHM_PER_SIGMA / beam.major_mas) ** 2
            + (across * FWHM_PER_SIGMA / beam.minor_mas) ** 2
        )
    ).astype(image.dtype)
    return fftconvolve(image, kernel / np.sum(kernel), mode="same")


class _CleanScene(SourceModel):
    """A fixed base scene plus point components on a pixel grid, with
    ``V = (w V_base + s Σ c_p e_p) / (w + s Σ c_p)``: ``w`` the base's weight, ``c``
    the fluxes, ``s`` the spectrum's shape and ``e_p`` a point's visibility. Without
    a base, ``V = Σ c_p e_p / Σ c_p``.
    """

    base: object
    fluxes: jax.Array
    pixel_scale_mas: float = eqx.field(static=True)
    rotation_deg: float = eqx.field(static=True)
    spectrum: object = None

    def model(self, u, v, wavel):
        return self.model_on_grid(u, v, wavel, None)

    def model_on_grid(self, u, v, wavel, grid):
        pixels = _pixel_visibilities(
            self.fluxes,
            self.pixel_scale_mas,
            self.rotation_deg,
            u,
            v,
            wavel,
            grid,
        )
        total = np.sum(self.fluxes)
        if self.base is None:
            return pixels / total
        if grid is None:
            base = self.base.model(u, v, wavel)
        else:
            base = self.base.model_on_grid(u, v, wavel, grid)
        # The flux the components are measured against (a System's total),
        # and their spectrum, 1 at its reference (``None``: grey).
        if isinstance(self.base, System):
            weight = sum(part._weight(wavel) for part in self.base.parts)
        else:
            weight = self.base._weight(wavel)
        if self.spectrum is None:
            shape = 1.0
        else:
            shape = self.spectrum(wavel) / self.spectrum(None)
        return (weight * base + shape * pixels) / (weight + shape * total)


def _scale_shapes(scales_mas, support, pixel_scale_mas):
    """The component shapes of multi-scale CLEAN, one per scale: ``None`` for a
    point, else ``(kernel, weight, mask)``, the 1-D Gaussian of the scale on the
    pixel grid (the 2-D shape is separable) and the weight that confines each
    component to ``support`` with unit flux.
    """
    mask = onp.asarray(support, float)
    offsets = onp.arange(mask.shape[0], dtype=float)
    shapes = []
    for s in scales_mas:
        if s == 0.0:
            shapes.append(None)
            continue
        sigma = s / (FWHM_PER_SIGMA * pixel_scale_mas)
        kernel = onp.exp(
            -0.5 * ((offsets[:, None] - offsets[None, :]) / sigma) ** 2
        )
        inside = kernel @ mask @ kernel  # the kernel is symmetric
        weight = onp.where(support, 1.0 / onp.where(support, inside, 1), 0)
        shapes.append((kernel, weight, mask))
    return tuple(shapes)


def _sky(fluxes, shapes):
    """The image ``Σ_s G_s * F_s`` of fluxes of shape (n_scales, n, n)."""
    images = []
    for f, shape in zip(fluxes, shapes):
        if shape is None:
            images.append(f)
            continue
        kernel, weight, mask = shape
        smooth = np.matmul(
            np.matmul(kernel, f * weight, precision=_HIGHEST),
            kernel,
            precision=_HIGHEST,
        )
        images.append(mask * smooth)
    sky = images[0]
    for image in images[1:]:
        sky = sky + image
    return sky


def _clean_residuals(parts, observations, scale, rotation):
    """Whitened residuals of all the data as a function of the fluxes;
    ``parts`` is ``(base, spectrum, shapes)``.
    """
    base, spectrum, shapes = parts

    def residuals(fluxes):
        sky = _sky(fluxes, shapes)
        scene = _CleanScene(base, sky, scale, rotation, spectrum)
        return np.concatenate(
            [np.ravel(whitened_residuals(scene, d)) for d in observations]
        )

    return residuals


@eqx.filter_jit
def _atom_norms(parts, observations, fluxes, scale, rotation):
    """``|J e_p|²`` for every component ``p``, one Jacobian–vector product at a
    time (a row of pixels per step), so the Jacobian is never held whole.
    """
    residuals = _clean_residuals(parts, observations, scale, rotation)
    ncol = fluxes.shape[-1]
    nrow = fluxes.size // ncol

    def row(i):
        def pixel(j):
            e = np.zeros((nrow, ncol), fluxes.dtype).at[i, j].set(1.0)
            e = e.reshape(fluxes.shape)
            return np.sum(jax.jvp(residuals, (fluxes,), (e,))[1] ** 2)

        return jax.vmap(pixel)(np.arange(ncol))

    return jax.lax.map(row, np.arange(nrow)).reshape(fluxes.shape)


@eqx.filter_jit
def _clean_step(
    parts, observations, fluxes, scores, scale, rotation
):  # deliberate: jit boundary
    """χ², and the best pixel and Gauss–Newton step of one CLEAN iteration: the
    largest ``g_p² / |J e_p|²`` with ``g_p < 0`` (``scores`` is ``1 / |J e_p|²``,
    zero outside the support).
    """
    residuals = _clean_residuals(parts, observations, scale, rotation)
    r, vjp = jax.vjp(residuals, fluxes)
    (gradient,) = vjp(2.0 * r)
    gradient = gradient.ravel()
    gain = np.where(gradient < 0.0, gradient**2 * scores.ravel(), 0.0)
    p = np.argmax(gain)
    direction = np.zeros(fluxes.size, fluxes.dtype).at[p].set(1.0)
    _, change = jax.jvp(
        residuals, (fluxes,), (direction.reshape(fluxes.shape),)
    )
    curvature = 2.0 * np.sum(change**2)
    return np.sum(r**2), p, gain[p], -gradient[p] / curvature


@eqx.filter_jit
def _clean_chi2(parts, observations, fluxes, scale, rotation):
    residuals = _clean_residuals(parts, observations, scale, rotation)
    return np.sum(residuals(fluxes) ** 2)


@eqx.filter_jit
def _clean_columns(
    parts, observations, fluxes, indices, scale, rotation
):  # deliberate: jit boundary
    """The residuals, and their derivatives by the fluxes at ``indices``."""
    residuals = _clean_residuals(parts, observations, scale, rotation)

    def column(k):
        e = np.zeros(fluxes.size, fluxes.dtype).at[k].set(1.0)
        return jax.jvp(residuals, (fluxes,), (e.reshape(fluxes.shape),))[1]

    return residuals(fluxes), jax.vmap(column)(indices)


def _refit(fixed, fluxes, chi2, scale, rotation):
    """A major cycle: refit the component fluxes by non-negative least squares on
    the linearized residuals, backtracking until χ² falls. Zeroed components are
    removed; the old fluxes if no step lowers χ².
    """
    from scipy.optimize import lsq_linear, nnls

    flat = onp.asarray(fluxes).ravel()
    active = onp.flatnonzero(flat > 0)
    if active.size == 0:
        return fluxes
    # Pad to a power of two, so the jitted Jacobian compiles a few times.
    size = 1 << int(onp.ceil(onp.log2(active.size)))
    indices = onp.concatenate(
        [active, onp.full(size - active.size, active[0])]
    )
    r, columns = _clean_columns(*fixed, fluxes, indices, scale, rotation)
    jacobian = onp.asarray(columns, float)[: active.size].T
    current = flat[active]
    rhs = jacobian @ current - onp.asarray(r, float)
    try:
        solution, _ = nnls(jacobian, rhs, maxiter=50 * active.size)
    except RuntimeError:
        # nnls gives up on ill-conditioned columns (high signal-to-noise);
        # BVLS returns its best point, and the backtracking keeps it only if
        # χ² falls. Not TRF: its line search can loop forever.
        solution = lsq_linear(
            jacobian, rhs, bounds=(0.0, onp.inf), method="bvls"
        ).x
    for fraction in (1.0, 0.5, 0.25, 0.125):
        trial = flat.copy()
        trial[active] = current + fraction * (solution - current)
        trial = np.asarray(trial.reshape(fluxes.shape), fluxes.dtype)
        if float(_clean_chi2(*fixed, trial, scale, rotation)) < chi2:
            return trial
    return fluxes


def _clean_model(base, components, pixel_scale_mas, rotation_deg, spectrum):
    """The base with the components as an Image (the model ``clean`` returns);
    the base alone when they have no flux.
    """
    components = onp.asarray(components, float)
    total = float(components.sum())
    if not total > 0:
        return base
    on = components > 0
    if base is None:
        flux = 1.0
    elif spectrum is None:
        flux = total
    else:  # the spectrum's shape, with the components' total flux
        scale = total / spectrum(None)
        flux = eqx.tree_at(lambda f: f.ratio, spectrum, spectrum.ratio * scale)
    image = Image(
        onp.log(onp.where(on, components, 1.0)),
        pixel_scale_mas,
        support=on,
        flux=flux,
        rotation_deg=rotation_deg,
    )
    if base is None:
        return image
    if isinstance(base, System):
        return System(**base.components, clean=image)
    return System(base=base, clean=image)


def _joint_refit(  # deliberate: named step of clean
    base, components, observations, priors, geometry, spectrum, dtype
):
    """Fit the base's free parameters (``priors``, keyed by paths in the base) and
    the components' fluxes together, so neither absorbs the other's error.
    Returns ``(base, components)``.
    """
    import numpyro.distributions as dist

    model = _clean_model(base, components, *geometry, spectrum)
    with_image = model is not base
    prefix = "" if isinstance(base, System) or not with_image else "base."
    joint = {prefix + k: v for k, v in priors.items()}
    if with_image:
        joint |= {
            k: v
            for k, v in image_priors(model).items()
            if k.startswith("clean.")
        }
        flux_path = "clean.flux" if spectrum is None else "clean.flux.ratio"
        joint[flux_path] = dist.ImproperUniform(
            dist.constraints.positive, (), ()
        )
    fitted = fit(model, joint, list(observations), dtype=dtype).model
    if not with_image:
        return fitted, components
    image = fitted.clean
    components = onp.asarray(image.brightness) * float(
        flux_at(image.flux, None)
    )
    if isinstance(base, System):
        base = System(**{name: fitted.components[name] for name in base.names})
    else:
        base = fitted.base
    return base, components


@dataclasses.dataclass(frozen=True)
class CleanResult:
    """The result of :func:`clean`.

    !!! warning "Provisional"
        May change in any 0.4.x release without a deprecation cycle.

    Attributes
    ----------
    model : SourceModel
        ``System(base=base, clean=Image(...))`` (a System base's components and
        ``clean`` side by side). The Image is non-zero only on the components,
        and its ``flux`` is their total relative to the base. Without a base,
        the Image alone; with no components, the base alone.
    components : array, shape (npix, npix)
        The components' total flux in each pixel, ``Σ_s G_s * F_s`` over the
        scales, relative to the base's weight (without a base, unit sum).
    pixel_scale_mas : float
        Pixel size in milliarcseconds.
    chi2_red : array
        χ² per data point before each iteration; the last entry is the final
        model's.
    stop : str
        ``"target"`` (``target_chi2_red`` reached), ``"stalled"`` (χ² stopped
        falling, even after a major cycle) or ``"max_iterations"``.
    components_by_scale : array, shape (n_scales, npix, npix)
        The fluxes ``F_s`` at each scale, at the centre of their shape, in the
        units of ``components``; each sums to that scale's share of the flux.
    scales_mas : tuple of float
        The FWHM of each scale's Gaussian shape (0 for a point).
    """

    model: object
    components: jax.Array
    pixel_scale_mas: float
    chi2_red: jax.Array
    stop: str
    components_by_scale: jax.Array = None
    scales_mas: tuple = (0.0,)

    def restored(self, beam):
        """The components convolved with ``beam``: the "restored" image.

        In the orientation of the pixel grid, in the units of
        ``components``. Unlike radio astronomy's restored image, it does not
        include the residuals.
        """
        return convolve_beam(self.components, self.pixel_scale_mas, beam)


def clean(
    data,
    npix,
    pixel_scale_mas,
    base=None,
    *,
    gain=0.1,
    max_iterations=1000,
    target_chi2_red=1.0,
    refit_every=50,
    stall_window=50,
    stall_tolerance=1e-3,
    refresh_norms=False,
    base_priors=None,
    spectrum=None,
    support=None,
    init=None,
    rotation_deg=0.0,
    dtype="float64",
    scales_mas=(0.0,),
    scale_bias=0.0,
):
    """Build an image from components, one at a time: gradient CLEAN.

    !!! warning "Provisional"
        May change in any 0.4.x release without a deprecation cycle.

    Each iteration adds ``gain`` times the Gauss–Newton step at the pixel that
    lowers χ² most (Högbom's CLEAN for linear data, else matching pursuit); every
    ``refit_every`` iterations a major cycle refits all fluxes by non-negative
    least squares. See [CLEAN](../imaging_clean.md).
    Source: Högbom (1974), [references](../references.md#hogbom1974).

    Parameters
    ----------
    data : OIData or sequence of OIData
        The data, fitted jointly.
    npix, pixel_scale_mas : int and float
        Grid size and pixel size (mas).
    base, gain : SourceModel and float, optional
        Fixed scene (e.g. a star) to add components to, not offset if a System;
        fraction of each step taken, in (0, 1].
    max_iterations, target_chi2_red, refit_every : int, float and int, optional
        Iteration limit; χ² per point to stop at; iterations per major cycle
        (0: none).
    stall_window, stall_tolerance : int and float, optional
        Stop when χ² falls by less than this fraction in this many iterations.
    base_priors, refresh_norms : dict and bool, optional
        Base priors refitted, and pixel norms ``|J e_p|`` recomputed (costly),
        each major cycle.
    spectrum, support, init : Spectrum and array-like, optional
        Shared spectrum (default grey; needs a base); boolean ``(npix, npix)``
        mask, e.g. [`circular_support`][virgil.models.circular_support];
        non-negative starting fluxes ``(npix, npix)`` or
        ``(n_scales, npix, npix)``.
    rotation_deg, dtype : float and str, optional
        Grid position angle (match ``data.uv_grid.rotation_deg``, see
        [`Image`][virgil.models.Image]); [precision](../common_arguments.md#dtype).
    scales_mas, scale_bias : sequence of float and float, optional
        FWHMs (mas) of the components, 0 for a point (``(0.0, 3.0)``: both);
        in [0, 1), the down-weighting ``1 - scale_bias s / max(scales_mas)``.

    Returns
    -------
    CleanResult
        The model, components and χ² history.
    """
    observations = tuple(_as_list(data))
    npix, pixel_scale_mas = int(npix), float(pixel_scale_mas)
    rotation_deg = float(rotation_deg)
    if not 0.0 < gain <= 1.0:
        raise ValueError(f"gain must be in (0, 1], not {gain}.")
    for name, value in (
        ("max_iterations", max_iterations),
        ("refit_every", refit_every),
        ("stall_window", stall_window),
    ):
        if int(value) != value or value < 0:
            raise ValueError(
                f"{name} must be a non-negative integer, not {value}."
            )
    max_iterations, refit_every, stall_window = (
        int(max_iterations),
        int(refit_every),
        int(stall_window),
    )
    scales_mas = tuple(float(s) for s in onp.atleast_1d(scales_mas))
    if (
        not scales_mas
        or not all(onp.isfinite(s) and s >= 0.0 for s in scales_mas)
        or len(set(scales_mas)) != len(scales_mas)
    ):
        raise ValueError(
            f"scales_mas must be distinct, finite and non-negative, not "
            f"{scales_mas}."
        )
    if not 0.0 <= scale_bias < 1.0:
        raise ValueError(f"scale_bias must be in [0, 1), not {scale_bias}.")
    nscales = len(scales_mas)
    if base is not None and base.time_dependent:
        raise ValueError("The base scene must not change with time.")
    if spectrum is not None and base is None:
        raise ValueError("A spectrum for the components needs a base scene.")
    base_priors = dict(base_priors or {})
    if base_priors and (base is None or not refit_every):
        raise ValueError(
            "base_priors needs a base scene and major cycles (refit_every > 0)."
        )
    if base_priors and isinstance(base, System):
        # A root path (e.g. "dra") would move the whole scene, image and
        # all, and the base rebuilt from its components would lose it.
        loose = [k for k in base_priors if k.split(".")[0] not in base.names]
        if loose:
            raise ValueError(
                f"base_priors on a System base must name its components' "
                f"parameters (e.g. '{base.names[-1]}.dra'), not {loose}."
            )
    if isinstance(base, System):
        if float(base.dra) != 0.0 or float(base.ddec) != 0.0:
            raise ValueError(
                "A System base must not be offset; offset its components."
            )
        if "clean" in base.components:
            raise ValueError("The base already has a component 'clean'.")
    shape = (npix, npix)
    support = (
        onp.ones(shape, bool)
        if support is None
        else onp.asarray(support, bool)
    )
    if support.shape != shape or not support.any():
        raise ValueError(
            f"support must have shape {shape} and at least one pixel."
        )
    if init is None:
        init = onp.zeros((nscales,) + shape)
        if base is None:  # the supported pixel nearest the centre
            offsets = pixel_offsets(npix, 1.0)
            radius = onp.hypot(offsets[None, :], offsets[:, None])
            centre = onp.argmin(onp.where(support, radius, onp.inf))
            init[onp.argmin(scales_mas)].flat[centre] = 1.0
    init = onp.asarray(init, float)
    if init.shape == shape:  # point components
        if 0.0 not in scales_mas:
            raise ValueError(
                f"An init of shape {shape} is point components; give "
                f"one of shape {(nscales,) + shape}, or a scale of 0."
            )
        points = init
        init = onp.zeros((nscales,) + shape)
        init[scales_mas.index(0.0)] = points
    if init.shape != (nscales,) + shape or not onp.all(
        onp.isfinite(init) & (init >= 0)
    ):
        raise ValueError(
            f"init must have shape {shape} or {(nscales,) + shape} and be "
            f"finite and non-negative."
        )
    if onp.any(init[:, ~support] > 0):
        raise ValueError("init has flux outside the support.")
    if base is None and not init.sum() > 0:
        raise ValueError("Without a base scene, init needs a positive pixel.")
    ndata = sum(d.n_independent for d in observations)
    shapes = _scale_shapes(scales_mas, support, pixel_scale_mas)
    bias = onp.ones((nscales, 1, 1))
    if scale_bias and max(scales_mas) > 0.0:
        bias[:, 0, 0] -= scale_bias * onp.array(scales_mas) / max(scales_mas)
    with run_in(dtype):
        cast_observations = cast_tree(observations, dtype)
        fixed = (
            cast_tree((base, spectrum, shapes), dtype),
            cast_observations,
        )
        fluxes = np.asarray(init, dtype)
        geometry = (pixel_scale_mas, rotation_deg)

        def scores_for(fluxes):
            norms = _atom_norms(*fixed, fluxes, *geometry)
            # Pixels whose flux cannot change the model (the seed pixel with
            # no base) have |J e_p| ~ eps, so 1/|J e_p|² would win the search
            # and take an infinite step: those below 100 eps of the largest
            # are dead.
            eps = float(np.finfo(norms.dtype).eps)
            live = norms > (100.0 * eps) ** 2 * np.max(norms)
            scores = np.where(
                support & live, 1.0 / np.where(live, norms, 1.0), 0.0
            )
            return scores if not scale_bias else scores * bias

        def major_cycle(fluxes, chi2):
            # Fit the free base parameters and the fluxes together, then
            # refit the fluxes (which can remove components) against the
            # corrected base. With several scales the joint fit gives the
            # base only: the image has no unique split into scales.
            nonlocal base, fixed
            if base_priors:
                base, components = _joint_refit(
                    base,
                    onp.asarray(_sky(fluxes, fixed[0][2]), float),
                    observations,
                    base_priors,
                    geometry,
                    spectrum,
                    dtype,
                )
                fixed = (
                    cast_tree((base, spectrum, shapes), dtype),
                    cast_observations,
                )
                if scales_mas == (0.0,):
                    fluxes = np.asarray(components, fluxes.dtype)[None]
                chi2 = float(_clean_chi2(*fixed, fluxes, *geometry))
            return _refit(fixed, fluxes, chi2, *geometry)

        scores = scores_for(fluxes)
        eps = float(np.finfo(fluxes.dtype).eps)
        history, stop = [], "max_iterations"
        since_refit = 0  # iterations since the last major cycle
        for iteration in range(max_iterations + 1):
            chi2, p, decrease, step = _clean_step(
                *fixed, fluxes, scores, *geometry
            )
            history.append(float(chi2) / ndata)
            if history[-1] <= target_chi2_red:
                stop = "target"
                break
            if iteration == max_iterations:
                break
            # A step that cannot lower χ² beyond its rounding error is no
            # step: at the optimum the gradient's sign is roundoff, so the
            # test is against eps·χ², not against exactly zero.
            stalled = not float(decrease) > eps * float(chi2) or (
                stall_window
                and since_refit >= stall_window
                and history[-1 - stall_window] - history[-1]
                < stall_tolerance * history[-1]
            )
            due = refit_every and since_refit >= refit_every
            if (due or stalled) and refit_every and since_refit > 0:
                fluxes = major_cycle(fluxes, float(chi2))
                if refresh_norms:
                    scores = scores_for(fluxes)
                since_refit = 0
                continue
            if stalled:
                stop = "stalled"
                break
            fluxes = (
                fluxes.ravel().at[p].add(gain * step).reshape(fluxes.shape)
            )
            since_refit += 1
        if refit_every and since_refit > 0:
            # A last major cycle, so the returned fluxes are always refitted.
            fluxes = major_cycle(fluxes, float(chi2))
            chi2 = _clean_chi2(*fixed, fluxes, *geometry)
            history.append(float(chi2) / ndata)
        by_scale = onp.asarray(fluxes, float)
        components = onp.asarray(_sky(fluxes, fixed[0][2]), float)
    if base is None:
        total = components.sum()
        components, by_scale = components / total, by_scale / total
    model = _clean_model(
        base, components, pixel_scale_mas, rotation_deg, spectrum
    )
    return CleanResult(
        model,
        np.asarray(components),
        pixel_scale_mas,
        np.asarray(history),
        stop,
        np.asarray(by_scale),
        scales_mas,
    )


@dataclasses.dataclass(frozen=True)
class LCurve:
    """The result of :func:`l_curve`.

    Attributes
    ----------
    weights : array
        The regularization weights, in the order fitted (largest first).
    chi2 : array
        Total χ² of each fit.
    chi2_red : array, shape (n_weights, n_datasets)
        χ² per data point of each dataset.
    penalty : array
        The unweighted regularizer, ``value / weight``, of each fit.
    results : list of FitResult
        The fits.
    """

    weights: object
    chi2: object
    chi2_red: object
    penalty: object
    results: list

    def corner(self):
        """The weight at the L-curve's corner, where it bends most sharply.

        The curve is ``(log χ², log penalty)`` parametrized by ``log w``; the corner
        is the interior point of largest curvature (Hansen & O'Leary 1993). Check
        it by eye: a sparse or noisy sweep has a noisy curvature. A fit with a
        non-finite χ² or penalty is left out.
        """
        weights = onp.asarray(self.weights)
        chi2, penalty = onp.asarray(self.chi2), onp.asarray(self.penalty)
        ok = onp.isfinite(chi2) & onp.isfinite(penalty)
        weights, chi2, penalty = weights[ok], chi2[ok], penalty[ok]
        t = np.log(weights)
        x, y = np.log(chi2), np.log(penalty)
        if t.size < 3:
            raise ValueError("The sweep needs at least three weights.")
        dx, dy = np.gradient(x, t), np.gradient(y, t)
        ddx, ddy = np.gradient(dx, t), np.gradient(dy, t)
        curvature = (dx * ddy - ddx * dy) / (dx**2 + dy**2) ** 1.5
        # The end points have one-sided derivatives; leave them out.
        return float(weights[1 + int(np.argmax(np.abs(curvature[1:-1])))])

    def discrepancy(self, target=1.0):
        """The weight at which χ² per data point reaches ``target``.

        Morozov's discrepancy principle: regularize as strongly as the data allow.
        With several datasets the binding one (the largest χ² per point) must reach
        the target. It interpolates linearly in ``log w`` and returns ``None`` if the
        sweep never crosses the target. The truth has χ² per point 1 ± √(2/N), so 1
        is the natural target; it relies on correct error bars and over-regularizes
        if they are underestimated.
        """
        binding = np.max(
            np.reshape(self.chi2_red, (len(self.weights), -1)), axis=1
        )
        above = binding > target
        crossings = np.nonzero(above[:-1] & ~above[1:])[0]
        if crossings.size == 0:
            return None
        i = int(crossings[0])
        t = np.log(self.weights[i : i + 2])
        c = binding[i : i + 2]
        fraction = (c[0] - target) / (c[0] - c[1])
        return float(np.exp(t[0] + fraction * (t[1] - t[0])))

    def classic_maxent(self, data, path="env"):
        """The maximum-entropy weight of Gull and Skilling's "classic MaxEnt".

        For a sweep of [`MaxEntropy`][virgil.imaging.MaxEntropy] fits, the weight
        ``w`` at which ``-2 w S`` equals the number of well-measured directions in
        the image, ``N = Σ λ / (λ + w)`` (Gull 1989, Skilling 1989, both in
        *Maximum Entropy and Bayesian Methods*, Kluwer). ``S`` is the entropy
        (minus the sweep's ``penalty``) and the ``λ`` are the eigenvalues of the
        data's Gauss–Newton curvature in the entropy metric ``diag(√b) JᵀJ
        diag(√b)``. It is the stationary point of the Laplace evidence in ``w``, so
        it assumes correct error bars. Both sides are evaluated at each fit in the
        sweep and the crossing is interpolated linearly in ``log w``; ``None`` if
        the sweep does not bracket it. Like [`log_evidence`][virgil.imaging.log_evidence]
        it uses the quoted errors, and rejects a sweep fitted with ``noise=`` terms
        or one model per dataset.

        Parameters
        ----------
        data : OIData or sequence of OIData
            The data the sweep was fitted to.
        path : str, optional
            Path of the regularized Image in the model (default ``"env"``).
        """
        gap = []
        for weight, penalty, result in zip(
            self.weights, self.penalty, self.results
        ):
            model = _single_model(result, "classic_maxent")
            image = model.get(path)
            if _whitened_field(image.log_brightness):
                raise TypeError("classic_maxent needs a pixel Image.")
            if image.blur_fwhm_mas is not None:
                raise ValueError(
                    "classic_maxent does not support a smoothed Image: its "
                    "entropy metric uses the raw pixels."
                )
            _, jac = _residual_jacobian(
                model, data, path + ".log_brightness", on_device=True
            )
            # dr/db = (dr/dη) / b on the support, so √b dr/db = (dr/dη) / √b.
            with run_in("float64"):
                b = onp.ravel(
                    onp.asarray(
                        cast_tree(image, "float64").brightness, dtype=float
                    )
                )
                scaled = jac[:, np.asarray(b > 0)] / np.asarray(
                    onp.sqrt(b[b > 0])
                )
            curvature = _singular_values(scaled) ** 2
            n_good = onp.sum(curvature / (curvature + weight))
            gap.append(2.0 * weight * penalty - n_good)
        gap = onp.asarray(gap)
        crossings = onp.nonzero(onp.sign(gap[:-1]) != onp.sign(gap[1:]))[0]
        if crossings.size == 0:
            return None
        i = int(crossings[0])
        t = onp.log(onp.asarray(self.weights[i : i + 2], dtype=float))
        fraction = gap[i] / (gap[i] - gap[i + 1])
        return float(onp.exp(t[0] + fraction * (t[1] - t[0])))


def _set_each(model, paths, values):
    """``model`` with each of ``paths`` set to its value, one at a time.

    The oldest zodiax does not accept a list of paths in ``set`` and ``get``.
    """
    for path, value in zip(paths, values):
        model = model.set(path, value)
    return model


# Basis vectors pushed through together when building a residual Jacobian.
_JACOBIAN_BATCH = 64


@eqx.filter_jit
def _jitted_residual_jacobian(model, datasets, path, regularizers=()):
    """Jitted body of ``_residual_jacobian``; ``path`` is one leaf's path or a tuple
    of paths, whose concatenated raveled leaves the Jacobian is taken with
    respect to.
    """
    paths = (path,) if isinstance(path, str) else tuple(path)
    if len(paths) == 1:
        leaf = np.asarray(model.get(paths[0]), dtype=float)

        def with_leaf(x):
            return model.set(paths[0], x)

    else:
        leaves = [np.asarray(model.get(p), dtype=float) for p in paths]
        shapes = [np.shape(x) for x in leaves]
        edges = onp.cumsum([0] + [int(onp.prod(s)) for s in shapes])
        leaf = np.concatenate([np.ravel(x) for x in leaves])

        def with_leaf(x):
            return _set_each(
                model,
                paths,
                [
                    x[edges[i] : edges[i + 1]].reshape(shape)
                    for i, shape in enumerate(shapes)
                ],
            )

    def residuals(x):
        changed = with_leaf(x)
        parts = [whitened_residuals(changed, d) for d in datasets]
        parts += [np.ravel(reg.residuals(changed)) for reg in regularizers]
        return np.concatenate(parts)

    n_data = sum(d.n_residuals for d in datasets) + sum(
        int(np.size(reg.residuals(model))) for reg in regularizers
    )
    n_leaf = np.size(leaf)
    # Rows (reverse mode) or columns (forward mode), whichever are fewer, in
    # batches: jacrev/jacfwd push every basis vector through at once, and the
    # memory then grows as data x data x pixels.
    if n_data < n_leaf:
        r, pullback = jax.vjp(residuals, leaf)

        def row(i):
            cotangent = np.zeros(n_data, r.dtype).at[i].set(1.0)
            return pullback(cotangent)[0].ravel()

        jac = jax.lax.map(row, np.arange(n_data), batch_size=_JACOBIAN_BATCH)
    else:
        r = residuals(leaf)

        def column(j):
            tangent = np.zeros(n_leaf, leaf.dtype).at[j].set(1.0)
            return jax.jvp(
                residuals, (leaf,), (tangent.reshape(np.shape(leaf)),)
            )[1]

        jac = jax.lax.map(
            column, np.arange(n_leaf), batch_size=_JACOBIAN_BATCH
        ).T
    return r, jac.reshape(n_data, -1)


def _residual_jacobian(model, data, path, on_device=False, regularizers=()):
    """All the whitened residuals, and their Jacobian with respect to leaves.

    ``path`` is one leaf's path or a tuple of paths. Returns NumPy
    ``(residuals, jacobian)``, float64; ``on_device=True`` leaves the
    Jacobian a JAX array. Probabilistic regularizers in ``regularizers``
    (such as ``Centroid``) append their ``residuals(model)`` to the residual
    vector and Jacobian.
    """
    datasets = tuple(_as_list(data))
    probs = tuple(
        r for r in (regularizers or ()) if getattr(r, "probabilistic", False)
    )
    if not isinstance(path, str):
        path = tuple(path)
    with run_in("float64"):
        model, datasets, probs = cast_tree((model, datasets, probs), "float64")
        r, jac = _jitted_residual_jacobian(model, datasets, path, probs)
        r = onp.asarray(r, dtype=float)
        return r, (jac if on_device else onp.asarray(jac, dtype=float))


def _singular_values(jac):
    """Singular values of ``jac`` (descending, NumPy), float64 on the device;
    QR-reduced first when far from square (the Gram matrix would lose the
    small ones)."""
    with run_in("float64"):
        a = np.asarray(jac, dtype=np.float64)
        if a.shape[0] > 2 * a.shape[1]:
            a = np.linalg.qr(a, mode="r")
        elif a.shape[1] > 2 * a.shape[0]:
            a = np.linalg.qr(a.T, mode="r")
        return onp.asarray(np.linalg.svd(a, compute_uv=False), dtype=float)


def _svd(jac):
    """``(s, Vᵀ)`` of the thin SVD of ``jac`` (NumPy), on the device in
    float64; QR-reduced first when tall, so ``U`` is never formed."""
    with run_in("float64"):
        a = np.asarray(jac, dtype=np.float64)
        if a.shape[0] > 2 * a.shape[1]:
            a = np.linalg.qr(a, mode="r")  # J = QR has R's V
        _, s, vt = np.linalg.svd(a, full_matrices=False)
        return onp.asarray(s, dtype=float), onp.asarray(vt, dtype=float)


def _single_model(model, caller):
    """The one model of a fit, for the evidence helpers. Fitted ``noise=`` terms
    (in a ``FitResult``) and per-dataset model lists are rejected: the helpers use
    the data's quoted errors.
    """
    if isinstance(model, FitResult):
        noise = [
            site
            for site in model.values
            if site.startswith(("noise.", "noise["))
        ]
        if noise:
            raise ValueError(
                f"{caller} uses the data's quoted errors, but this fit has "
                f"fitted error terms {noise}. Fit without noise=, after "
                "rescaling the errors with OIData.with_error_scale if needed."
            )
        model = model.model
    if isinstance(model, (list, tuple)):
        raise TypeError(
            f"{caller} needs a single model, not a list of models (one per "
            "dataset)."
        )
    return model


@eqx.filter_jit
def _jitted_log_norm(model, datasets):
    """The marginal nuisances' log-normalizer, summed over datasets.

    ``model_loglike`` is ``-½ Σ r² - log_norm`` plus a term that depends only on
    the data and the error terms; ``log_norm`` is ½ log det of the covariance
    factors of marginalized gains and closure offsets (zero without them).
    """
    return sum(
        _whitened_errors_and_log_norm(model, d, {})[2] for d in datasets
    )


def _log_norm(model, data):
    """The marginal nuisances' log-normalizer at ``model``, in float64."""
    datasets = tuple(_as_list(data))
    with run_in("float64"):
        model, datasets = cast_tree((model, datasets), "float64")
        return float(_jitted_log_norm(model, datasets))


def _whitened_field(obj):
    """Whether ``obj`` is a whitened field: it evaluates and has latents."""
    return callable(getattr(obj, "evaluate", None)) and hasattr(obj, "latent")


def _field_paths(model):
    """The paths of every Image in ``model`` with a whitened field."""
    return tuple(
        p
        for p, part in _parts(model)
        if isinstance(part, Image) and _whitened_field(part.log_brightness)
    )


def _latent_path(image_path):
    return (
        "log_brightness.latent"
        if image_path is None
        else image_path + ".log_brightness.latent"
    )


def _field_images(model, path, caller):
    """The paths of the Images with whitened fields that ``path`` selects: one
    path, a tuple, or ``None`` for all of them (a ``TypeError`` if there is none).
    """
    if path is None:
        found = _field_paths(model)
        if not found:
            raise TypeError(
                f"{caller} found no Image with a GaussianField in the model; "
                "give path= for an Image with one."
            )
        return found
    paths = (path,) if isinstance(path, str) else tuple(path)
    if not paths:
        raise ValueError(f"{caller} needs at least one path, not none.")
    if len(set(paths)) != len(paths):
        raise ValueError(f"{caller} was given repeated paths {paths}.")
    for p in paths:
        image = model.get(p)
        if not (
            isinstance(image, Image) and _whitened_field(image.log_brightness)
        ):
            raise TypeError(
                f"{caller} needs an Image with a GaussianField at {p!r}."
            )
    return paths


def _evidence_latents(model, path, caller):
    """The paths of every whitened latent that ``path`` selects: those of Images
    with a GaussianField and the Gaussian-process latents of MultipoleDisks.
    ``path`` is a path or tuple of paths, or ``None`` for all in ``model``.
    """
    from .models import MultipoleDisk, _multipole_disks

    if path is None:
        images = _field_paths(model)
        disks = [
            (pre, d) for pre, d in _multipole_disks(model) if d.gp_sites()
        ]
        if not images and not disks:
            raise TypeError(
                f"{caller} found no Image with a GaussianField and no "
                "MultipoleDisk with a RadialGP or MultipoleKernel in the model; "
                "give path= for one."
            )
    else:
        paths = (path,) if isinstance(path, str) else tuple(path)
        if not paths:
            raise ValueError(f"{caller} needs at least one path.")
        images, disks = [], []
        for p in paths:
            if isinstance(model.get(p), MultipoleDisk):
                if not model.get(p).gp_sites():
                    raise TypeError(
                        f"{caller} needs a MultipoleDisk with a RadialGP or a "
                        f"MultipoleKernel at {p!r}."
                    )
                disks.append((p + ".", model.get(p)))
            else:
                images.append(p)
        if images:
            images = _field_images(model, tuple(images), caller)
        if len(set(paths)) != len(paths):
            raise ValueError(f"{caller} was given repeated paths {paths}.")
    latents = [_latent_path(p) for p in images]
    for prefix, disk in disks:
        latents.extend(f"{prefix}{rel}.z" for rel in disk.gp_sites())
    return tuple(latents)


def log_evidence(model, data, path="env", *, priors=None, regularizers=()):
    """Laplace-approximated log evidence of a Gaussian-process fit.

    At the MAP ``model``, ``log Z ≈ -½ χ² - L - ½ |z|² - ½ log det(I + JᵀJ)``
    up to a data-only constant, with ``z`` the whitened latents of the Image's
    [`GaussianField`][virgil.fields.GaussianField] (or the Gaussian-process parts
    of a [`MultipoleDisk`][virgil.models.MultipoleDisk]), ``J`` their Jacobian and
    ``L`` the likelihood normalizer; extra priors in ``priors`` enter through
    their ``log_prob`` at the MAP and their ``JᵀJ`` block of the Hessian. Compare
    it across hyperparameters: it is conditional on those held in ``model`` (or
    sample them: log-uniform priors from [`gp_priors`][virgil.models.gp_priors]
    for a ``MultipoleDisk``, your own log-uniform priors on a field's ``sigma``
    and ``length_mas`` for an Image). Exact only for a linear model, and it
    assumes correct error bars (see [`error_scale`][virgil.imaging.error_scale])
    and uses the quoted errors, so no ``noise=`` terms or per-dataset models. See
    [Evidence](../imaging_gp.md#choosing-and-l-from-the-evidence).
    Source: MacKay (1992), [references](../references.md#mackay1992).

    Parameters
    ----------
    model : SourceModel or FitResult
        The MAP model, or better the [`FitResult`][virgil.fitting.FitResult], so
        fitted ``noise=`` terms are caught (``ValueError``); a list of models
        raises ``TypeError``.
    data : OIData or sequence of OIData
        The data it was fitted to.
    path : str or tuple of str or None, optional
        Path of the Image (default ``"env"``), a tuple of paths, or ``None`` for
        every such Image and disk (``TypeError`` if there is none).
    priors : dict, optional
        Priors of free parameters; non-latent parameters enter via ``log_prob``
        and their ``JᵀJ`` block of the Hessian.
    regularizers : sequence, optional
        Fit regularizers; probabilistic ones (``Centroid``) enter ``χ²`` and ``J``.

    Returns
    -------
    float
    """
    model = _single_model(model, "log_evidence")
    latent_paths = _evidence_latents(model, path, "log_evidence")
    extra_paths = tuple(
        p
        for p in (priors or ())
        if p not in latent_paths and not p.endswith("_vec")
    )
    all_paths = latent_paths + extra_paths
    r, jac = _residual_jacobian(
        model, data, all_paths, on_device=True, regularizers=regularizers
    )
    chi2 = float(r @ r)
    log_norm = _log_norm(model, data)
    z2 = sum(
        onp.sum(onp.asarray(model.get(latent), dtype=float) ** 2)
        for latent in latent_paths
    )
    if not extra_paths:
        # log det(I + JᵀJ) = Σ log(1 + s²); the singular values avoid forming
        # the Gram matrix (see "Evidence" in docs/api/imaging.md).
        singular = _singular_values(jac)
        logdet = float(onp.sum(onp.log1p(singular**2)))
        return float(-0.5 * chi2 - log_norm - 0.5 * z2 - 0.5 * logdet)
    n_z = sum(int(onp.size(model.get(latent))) for latent in latent_paths)
    jac_z = jac[:, :n_z]
    jac_extra = onp.asarray(jac[:, n_z:], dtype=float)
    singular, vt = _svd(jac_z)
    logdet = float(onp.sum(onp.log1p(singular**2)))
    c = vt @ (onp.asarray(jac_z, dtype=float).T @ jac_extra)
    schur = jac_extra.T @ jac_extra - (c.T / (1.0 + singular**2)) @ c
    sign, logdet_extra = onp.linalg.slogdet(schur)
    if sign <= 0:
        return float(-onp.inf)
    log_prior = sum(
        float(onp.sum(onp.asarray(priors[p].log_prob(model.get(p)), float)))
        for p in extra_paths
    )
    extra_term = (
        log_prior
        + 0.5 * jac_extra.shape[1] * float(onp.log(2.0 * onp.pi))
        - 0.5 * float(logdet_extra)
    )
    return float(-0.5 * chi2 - log_norm - 0.5 * z2 - 0.5 * logdet + extra_term)


@eqx.filter_jit
def _render_latents(
    model, latent_path, latents, npix, fov_mas
):  # deliberate: jit boundary
    """The whole model rendered with each of ``latents`` at its field.

    ``latent_path`` is one path with ``latents`` an array, or a tuple of
    paths with ``latents`` a tuple of arrays, each with a leading draw axis.
    """
    if isinstance(latent_path, str):

        def put(x):
            return model.set(latent_path, x)

    else:

        def put(x):
            return _set_each(model, latent_path, x)

    return jax.vmap(lambda x: put(x).render(npix, fov_mas))(latents)


def laplace_samples(
    model,
    data,
    n,
    key,
    path="env",
    npix=None,
    fov_mas=None,
    *,
    priors=None,
    regularizers=(),
):
    """Draws from the Laplace posterior of a Gaussian-field image.

    The whitened latents ``z`` are drawn from ``N(z₀, (I + JᵀJ)⁻¹)`` about the
    MAP ``z₀``, the approximation of [`log_evidence`][virgil.imaging.log_evidence].
    With ``priors``, free non-pixel parameters are included in ``z`` and ``J``.
    Exact only for a linear model, assumes correct error bars, uses the quoted
    errors and is conditional on the hyperparameters held in ``model`` (or sample
    them, as for [`log_evidence`][virgil.imaging.log_evidence]).

    Parameters
    ----------
    model : SourceModel or FitResult
        The MAP model, or the [`FitResult`][virgil.fitting.FitResult] (fitted
        ``noise=`` terms raise ``ValueError``, lists of models ``TypeError``).
    data : OIData or sequence of OIData
        The data it was fitted to.
    n : int
        Number of draws.
    key : jax.Array
        A JAX random key.
    path : str or tuple of str or None, optional
        Path of the Image (default ``"env"``), a tuple of paths, or ``None`` for
        every Image with a GaussianField (``TypeError`` if there is none).
    npix, fov_mas : int and float, optional
        Together, also render the whole model for each draw (see
        [`SourceModel.render`][virgil.models.SourceModel.render]); one alone
        raises ``ValueError``.
    priors : dict or sequence of str, optional
        Free parameter priors or paths; non-pixel parameters enter ``z`` and ``J``.
    regularizers : sequence, optional
        Fit regularizers; probabilistic ones (``Centroid``) enter ``J``.

    Returns
    -------
    dict
        ``"latents"``, an array ``(n, *latent.shape)`` (a dict by Image path if
        several are selected), and with ``npix`` and ``fov_mas`` ``"images"``,
        ``(n, npix, npix)`` unit-sum images of the whole model.
    """
    if (npix is None) != (fov_mas is None):
        raise ValueError(
            "laplace_samples renders images only given both npix and fov_mas."
        )
    model = _single_model(model, "laplace_samples")
    fields = tuple(
        (p, _latent_path(p))
        for p in _field_images(model, path, "laplace_samples")
    )
    latent_paths = tuple(latent for _, latent in fields)
    extra_paths = tuple(
        p
        for p in (priors or ())
        if p not in latent_paths and not p.endswith("_vec")
    )
    all_paths = latent_paths + extra_paths
    _, jac = _residual_jacobian(
        model, data, all_paths, on_device=True, regularizers=regularizers
    )
    zs = [onp.asarray(model.get(p), dtype=float) for p in all_paths]
    z = onp.concatenate([x.ravel() for x in zs])
    n_z = sum(x.size for x in zs[: len(fields)])
    jac_z = jac[:, :n_z]
    singular, vt = _svd(jac_z)
    first, second = jax.random.split(key)
    e1 = onp.asarray(jax.random.normal(first, (n, singular.size)), float)
    e2 = onp.asarray(jax.random.normal(second, (n, n_z)), float)
    # V diag((1+s²)^-½) ε₁ along the data's directions, and ε₂ projected
    # off them, (I - VVᵀ) ε₂, along the prior's.
    draws = (e1 / onp.sqrt(1.0 + singular**2)) @ vt
    draws += e2 - (e2 @ vt.T) @ vt
    if extra_paths:
        jac_extra = onp.asarray(jac[:, n_z:], dtype=float)
        c = vt @ (onp.asarray(jac_z, dtype=float).T @ jac_extra)
        a_inv_b = vt.T @ (c / (1.0 + singular[:, None] ** 2))
        schur = jac_extra.T @ jac_extra - (c.T / (1.0 + singular**2)) @ c
        _, third = jax.random.split(second)
        e3 = onp.asarray(
            jax.random.normal(third, (n, jac_extra.shape[1])), float
        )
        chol = onp.linalg.cholesky(onp.linalg.inv(schur))
        draws_extra = e3 @ chol.T
        draws = onp.hstack([draws - draws_extra @ a_inv_b.T, draws_extra])
    flat = z + draws
    edges = onp.cumsum([0] + [x.size for x in zs])
    pieces = [
        flat[:, edges[i] : edges[i + 1]].reshape((n,) + x.shape)
        for i, x in enumerate(zs)
    ]
    latent_pieces = pieces[: len(fields)]
    if len(latent_pieces) == 1:
        out = {"latents": latent_pieces[0]}
    else:
        out = {
            "latents": {
                "" if image is None else image: piece
                for (image, _), piece in zip(fields, latent_pieces)
            }
        }
    if npix is not None and fov_mas is not None:
        dtypes = [np.asarray(model.get(p)).dtype for p in all_paths]
        if len(pieces) == 1:
            where = all_paths[0]
            values = np.asarray(pieces[0], dtype=dtypes[0])
        else:
            where = all_paths
            values = tuple(
                np.asarray(piece, dtype=dtype)
                for piece, dtype in zip(pieces, dtypes)
            )
        out["images"] = onp.asarray(
            _render_latents(model, where, values, int(npix), float(fov_mas))
        )
    return out


def _observable_blocks(datasets):
    """The rows of the residual vector of each observable: ``(rows, counts)``,
    dicts from ``"vis"``, ``"phi"`` and the kinds of ``extras`` to the row indices
    in [`whitened_residuals`][virgil.likelihood.whitened_residuals] (over
    ``datasets``) and to the number of independent data. Correlated closure
    phases have ``cp_noise.size`` combinations (the data) followed by one penalty
    row per phase, which is not counted.
    """
    rows, counts = {}, {}
    start = 0

    def add(kind, n_rows, n_independent):
        nonlocal start
        if n_rows:
            rows.setdefault(kind, []).append(onp.arange(start, start + n_rows))
            counts[kind] = counts.get(kind, 0) + n_independent
        start += n_rows

    for d in datasets:
        first = start
        n_vis = int(onp.size(d.vis))
        n_phi = int(onp.size(d.phi))
        add("vis", n_vis, n_vis)
        if d.cp_noise is None:
            add("phi", n_phi, n_phi)
        else:
            add("phi", d.cp_noise.size + n_phi, d.cp_noise.size)
        for block in d.extras:
            add(block.kind, int(onp.size(block.data())), block.n_independent)
        if start - first != d.n_residuals:
            raise RuntimeError(
                f"The observable blocks have {start - first} rows, but the "
                f"data have {d.n_residuals} residuals."
            )
    return {k: onp.concatenate(v) for k, v in rows.items()}, counts


def _block_precisions(r, jac, rows, counts, max_iter=1000, rtol=1e-10):
    """MacKay's noise precision β_b of each block of residual rows, by iterating
    ``β_b ← N_b / (χ²_b + γ_b / β_b)`` (monotone, so it does not oscillate) from
    ``N_b / χ²_b``; see [Error scale](imaging.md#error-scale). ``A``'s factor
    comes from a QR of the stacked, row-scaled Jacobian, never the Gram matrix.
    """
    from scipy.linalg import solve_triangular

    kinds = list(rows)
    n_rows, n_latent = jac.shape
    chi2 = onp.array([float(r[rows[k]] @ r[rows[k]]) for k in kinds])
    n = onp.array([counts[k] for k in kinds], dtype=float)
    if n_rows <= n_latent:
        # The n x n form: with Jᵀ = Q R, K = B½ J Jᵀ B½ = (R B½)ᵀ (R B½),
        # and the hat matrix's diagonal is h = 1 − diag((K + I)⁻¹).
        reduced = onp.linalg.qr(jac.T, mode="r")
        eye = onp.eye(n_rows)

        def traces(beta):
            per_row = onp.zeros(n_rows)
            for k, b in zip(kinds, beta):
                per_row[rows[k]] = b
            scaled = reduced * onp.sqrt(per_row)
            factor = onp.linalg.qr(onp.vstack([scaled, eye]), mode="r")
            inverse = solve_triangular(factor, eye)
            hat = 1.0 - onp.sum(inverse**2, axis=1)
            return onp.array(
                [onp.sum(hat[rows[k]]) / b for k, b in zip(kinds, beta)]
            )

    else:
        # The latent form: G_b = R_bᵀ R_b with J_b = Q_b R_b, so A is
        # factored from the stacked √β_b R_b, and t_b = |R_A⁻ᵀ R_bᵀ|².
        reduced = [onp.linalg.qr(jac[rows[k]], mode="r") for k in kinds]
        eye = onp.eye(n_latent)

        def traces(beta):
            stacked = [onp.sqrt(b) * rb for b, rb in zip(beta, reduced)]
            factor = onp.linalg.qr(onp.vstack(stacked + [eye]), mode="r")
            return onp.array(
                [
                    onp.sum(solve_triangular(factor.T, rb.T, lower=True) ** 2)
                    for rb in reduced
                ]
            )

    if not onp.all(chi2 > 0.0):
        raise ValueError(
            "error_scale: a block of observables is fitted exactly (χ² = 0), "
            "so its noise scale is undefined."
        )
    beta = n / chi2
    for _ in range(max_iter):
        new = n / (chi2 + traces(beta))
        change = onp.max(onp.abs(new - beta) / new)
        beta = new
        if change < rtol:
            break
    else:
        warnings.warn(
            f"error_scale: the block noise scales did not converge in "
            f"{max_iter} iterations (relative change {change:.1e}).",
            RuntimeWarning,
            stacklevel=external_stacklevel(),
        )
    return dict(zip(kinds, beta))


def error_scale(
    model,
    data,
    path="env",
    *,
    by_observable=False,
    priors=None,
    regularizers=(),
):
    r"""Re-estimate the scale of the error bars from a Gaussian-field fit.

    Returns the factor ``s`` to multiply every error bar by for consistency with
    the fit (``s < 1``: bars too large), solving ``s² = χ² / (N − γ)`` with ``γ`` the effective number of
    parameters the data measure; see
    [Error scale](../imaging_gp.md#are-the-error-bars-right). Fit, call this,
    rescale with [`OIData.with_error_scale`][virgil.oidata.OIData.with_error_scale],
    refit, and do so before choosing hyperparameters (bars that are too large
    make the evidence over-regularize). ``s`` much greater than 1 means a failed
    or inadequate model, not a noise fix: unmodelled structure is absorbed into
    ``s``. It uses the quoted errors, so no ``noise=`` terms or per-dataset
    models.
    Source: MacKay (1992), [references](../references.md#mackay1992).

    Parameters
    ----------
    model : SourceModel or FitResult
        The MAP model, as for [`log_evidence`][virgil.imaging.log_evidence].
    data : OIData or sequence of OIData
        The data it was fitted to.
    path : str or tuple of str or None, optional
        Path of the Image (default ``"env"``), a tuple of paths, or ``None`` for
        every such Image and disk (``TypeError`` if there is none).
    by_observable : bool, optional
        One scale per kind of observable instead of one for all. Data with
        gains, closure-phase offsets or other covariance that does not scale
        with the errors raise ``ValueError``.
    priors : dict or sequence of str, optional
        Free parameter priors or paths; non-pixel parameters enter ``J``.
    regularizers : sequence, optional
        Fit regularizers; probabilistic ones (``Centroid``) enter ``J``.

    Returns
    -------
    float or dict
        The scale ``s``, or with ``by_observable=True`` a dict from each
        observable kind (``"vis"``, ``"phi"``, ...) to its scale, ready for
        ``OIData.with_error_scale``. A warning is issued if the block scales do
        not converge.
    """
    model = _single_model(model, "error_scale")
    latent_paths = _evidence_latents(model, path, "error_scale")
    extra_paths = tuple(
        p
        for p in (priors or ())
        if p not in latent_paths and not p.endswith("_vec")
    )
    all_paths = latent_paths + extra_paths
    r, jac = _residual_jacobian(
        model, data, all_paths, on_device=True, regularizers=regularizers
    )
    n_z = sum(int(onp.size(model.get(p))) for p in latent_paths)
    if extra_paths:
        jac_z = onp.asarray(jac[:, :n_z], dtype=float)
        u_extra, s_extra, _ = onp.linalg.svd(
            onp.asarray(jac[:, n_z:], dtype=float), full_matrices=False
        )
        keep = s_extra > (1e-12 * s_extra[0] if s_extra.size else 0.0)
        q = u_extra[:, keep]
        r_extra = int(q.shape[1])
        jac_z = jac_z - q @ (q.T @ jac_z)
    else:
        jac_z = jac
        r_extra = 0
    datasets = _as_list(data)
    if by_observable:
        # Refuse data whose covariance is not s_b² D per block: their nuisance
        # widths do not scale with the quoted errors.
        for i, d in enumerate(datasets):
            reasons = []
            if d.gains is not None:
                reasons.append("calibration gains (with_gains)")
            if d.phase_offsets is not None:
                reasons.append("closure-phase offsets (with_closure_offsets)")
            for b in d.extras:
                if b.model_dependent_covariance:
                    reasons.append(f"a marginalized {b.kind} block")
                elif getattr(b, "prior_width", None) is not None:
                    reasons.append(
                        f"a {b.kind} block with a finite prior_width"
                    )
            if reasons:
                raise ValueError(
                    f"error_scale(by_observable=True) cannot rescale dataset {i}: "
                    f"it has {', '.join(reasons)}, whose nuisance widths do not "
                    "scale with the quoted errors. Estimate the scales without "
                    "them, or fit the nuisance widths as noise terms."
                )
        rows, counts = _observable_blocks(datasets)
        # The block fixed point factors on the host (NumPy QR).
        beta = _block_precisions(r, onp.asarray(jac_z), rows, counts)
        return {k: float(1.0 / onp.sqrt(b)) for k, b in beta.items()}
    chi2 = float(r @ r)
    # Independent data, not residuals: penalty rows are not observations.
    n_data = sum(d.n_independent for d in datasets) - r_extra
    lam = _singular_values(jac_z) ** 2
    # Newton on g(β) = β χ² + γ(β) − N, increasing and concave: monotone from 0.
    beta = 0.0
    for _ in range(100):
        g = beta * chi2 + onp.sum(beta * lam / (1.0 + beta * lam))
        slope = chi2 + onp.sum(lam / (1.0 + beta * lam) ** 2)
        step = (g - n_data) / slope
        beta -= step
        if abs(step) <= 1e-12 * beta:
            break
    return float(1.0 / onp.sqrt(beta))


@renamed()
def l_curve(
    model,
    priors,
    data,
    regularizer,
    weights,
    others=(),
    *,
    warm_start=True,
    **fit_options,
):
    """Fit a model over a range of weights for one regularizer.

    The weights are fitted from largest to smallest, each starting from the
    previous solution. Plot ``penalty`` against ``chi2`` (log axes) to see the
    trade-off, and compare [`LCurve.corner`][virgil.imaging.LCurve.corner] and
    [`LCurve.discrepancy`][virgil.imaging.LCurve.discrepancy] with the images
    either side: there is usually a wide range of good weights.

    Parameters
    ----------
    model, priors, data
        As for [`fit`][virgil.fitting.fit].
    regularizer : TSV, TV, MaxEntropy, Laplacian, StarletL1 or LogSum
        The regularizer whose ``weight`` is swept (its own is ignored).
    weights : sequence of float
        The weights to try (sorted internally, largest first).
    others : sequence, optional
        Further regularizers kept fixed, e.g. a
        [`Centroid`][virgil.imaging.Centroid] prior.
    warm_start : bool, optional
        Start each fit from the previous, stronger one (default), or every
        fit from ``model`` (and ``init``). Use ``False`` for penalties that
        switch pixels off, such as [`LogSum`][virgil.imaging.LogSum]: pixels
        a strong weight has driven dark cannot recover in the weaker fits
        that follow.
    **fit_options
        Passed to [`fit`][virgil.fitting.fit].

    Returns
    -------
    LCurve
    """
    weights = sorted((float(w) for w in weights), reverse=True)
    if not weights or not all(onp.isfinite(w) and w > 0.0 for w in weights):
        raise ValueError(
            f"weights must be a non-empty sequence of finite positive numbers, "
            f"not {weights}."
        )
    results, chi2, chi2_red, penalty = [], [], [], []
    start = init = fit_options.pop("init", None)
    for weight in weights:
        weighted = eqx.tree_at(
            lambda r: r.weight, regularizer, np.asarray(weight, dtype=float)
        )
        result = fit(
            model, priors, data, [weighted, *others], init=init, **fit_options
        )
        init = result.values if warm_start else start
        results.append(result)
        chi2.append(sum(result.info["chi2"]))
        chi2_red.append(
            [c / n for c, n in zip(result.info["chi2"], result.info["ndata"])]
        )
        penalty.append(
            float(weighted.value(_reference(result.model))) / weight
        )
    return LCurve(
        np.asarray(weights),
        np.asarray(chi2),
        np.asarray(chi2_red),
        np.asarray(penalty),
        results,
    )


@eqx.filter_jit
def _jitted_residuals(model, observations):
    return tuple(whitened_residuals(model, d) for d in observations)


def _residual_vector(model, observations):
    """All the whitened residuals as one float64 array."""
    with run_in("float64"):
        model64, obs64 = cast_tree((model, tuple(observations)), "float64")
        parts = _jitted_residuals(model64, obs64)
        return onp.concatenate([onp.ravel(onp.asarray(x)) for x in parts])


def chi2_by_observable(model, data, path=None):
    """Raw χ² of a model for each kind of observable, on the quoted errors.

    The errors are the data's own (no ``noise=`` terms). Closure phases from
    four or more telescopes contribute their whitened combinations, and their
    periodic penalty rows (see
    [`whitened_residuals`][virgil.likelihood.whitened_residuals]) add to the χ²
    but not to the count.

    Parameters
    ----------
    model : SourceModel
        The model.
    data : OIData or sequence of OIData
        The data.
    path : str or tuple of str, optional
        Path(s) of the Gaussian-process Images, as for
        [`log_evidence`][virgil.imaging.log_evidence]. If given, the
        result also holds ``n_gamma``.

    Returns
    -------
    dict
        ``chi2``, ``n`` and ``chi2_red`` (χ² per independent datum), each a
        dict from the kinds of observable (``"vis"``, ``"phi"``, then any
        extras) and ``"total"`` to a number. With ``path``, ``n_gamma``:
        MacKay's number of well-determined latent parameters,
        ``Σ s²/(1 + s²)`` over the singular values ``s`` of the Jacobian of
        the whitened residuals by the latents.
    """
    observations = _as_list(data)
    rows, counts = _observable_blocks(observations)
    r = _residual_vector(model, observations)
    chi2 = {kind: float(onp.sum(r[idx] ** 2)) for kind, idx in rows.items()}
    n = dict(counts)
    chi2["total"] = float(onp.sum(r**2))
    n["total"] = sum(counts.values())
    out = {
        "chi2": chi2,
        "n": n,
        "chi2_red": {k: chi2[k] / n[k] for k in chi2},
    }
    if path is not None:
        single = _single_model(model, "chi2_by_observable")
        latents = _evidence_latents(single, path, "chi2_by_observable")
        _, jac = _residual_jacobian(single, data, latents, on_device=True)
        s2 = _singular_values(jac) ** 2
        out["n_gamma"] = float(onp.sum(s2 / (1.0 + s2)))
    return out


def _no_baselines(observations):
    """Why the visibility residual rows have no baselines, or ``None``."""
    for d in observations:
        if d.vis_mat is not None:
            return (
                "The visibilities are a linear combination of samples "
                "(vis_mat), so they have no baseline of their own."
            )
        if d.gains is not None:
            return (
                "The visibilities carry calibration gains, whose "
                "marginalization mixes the residual rows, so a row does "
                "not belong to one baseline."
            )
    if not _observable_blocks(observations)[0].get("vis", ()).size:
        return "The data have no visibilities."
    return None


def short_baseline_residual(model, data, quantile=0.2):
    """Mean whitened visibility residual on the shortest baselines.

    A reconstruction that loses flux on large scales leaves a coherent residual
    on the shortest frequencies though the total χ² looks fine (the test of
    MACIM, Ireland et al. 2006). The residuals are (model − data)/σ, so the
    mean of ``n`` of them has standard error ``1/√n`` if the model is right.

    Parameters
    ----------
    model : SourceModel
        The model.
    data : OIData or sequence of OIData
        The data. Visibilities in a linear combination (``vis_mat``, as in
        kernel-phase and DISCO data) and data with calibration gains are
        not supported: a ``ValueError``.
    quantile : float, optional
        The fraction of visibility samples, shortest first by B/λ, that
        are used (default 0.2).

    Returns
    -------
    tuple of float
        ``(mean, standard_error)``.
    """
    if not 0.0 < quantile <= 1.0:
        raise ValueError(f"quantile must be in (0, 1], not {quantile}.")
    observations = _as_list(data)
    reason = _no_baselines(observations)
    if reason:
        raise ValueError(reason)
    rows, _ = _observable_blocks(observations)
    r = _residual_vector(model, observations)[rows["vis"]]
    rho = []  # spatial frequency B/λ of each visibility row of the residuals
    for d in observations:
        b = onp.hypot(
            onp.ravel(onp.asarray(d.u, float)),
            onp.ravel(onp.asarray(d.v, float)),
        )
        lam = onp.broadcast_to(onp.ravel(onp.asarray(d.wavel, float)), b.shape)
        if d.vis_index is not None:
            keep = onp.asarray(d.vis_index)
            b, lam = b[keep], lam[keep]
        rho.append(b / lam)
    rho = onp.concatenate(rho)
    n = max(1, int(round(quantile * rho.size)))
    shortest = onp.argsort(rho, kind="stable")[:n]
    return float(onp.mean(r[shortest])), float(1.0 / onp.sqrt(n))


def residual_image(model, data, npix, pixel_scale_mas, rotation_deg=0.0):
    """The residual dirty map of a model, for any observable.

    The normalised χ² gradient ``-g_p / |J e_p|`` of adding a little flux at
    each pixel ``p`` (``g`` the gradient of χ², ``J e_p`` the response of the
    whitened residuals), as [`clean`][virgil.imaging.clean] uses. For linear
    data it is the dirty image of the residuals; for closure phases, the
    gradient the fit still feels. Positive pixels need more flux; pixels the
    model cannot respond to are zero.

    Parameters
    ----------
    model : SourceModel
        The model, which needs a flux for the added point sources to be
        measured against.
    data : OIData or sequence of OIData
        The data.
    npix, pixel_scale_mas : int, float
        The centred grid of the map, as in [`clean`][virgil.imaging.clean].
    rotation_deg : float, optional
        The grid's rotation, as in ``clean``.

    Returns
    -------
    numpy.ndarray
        Shape ``(npix, npix)``.
    """
    observations = _as_list(data)
    shapes = _scale_shapes(
        (0.0,), onp.ones((npix, npix), bool), pixel_scale_mas
    )
    with run_in("float64"):
        fixed = cast_tree((model, None, shapes), "float64")
        obs = cast_tree(tuple(observations), "float64")
        fluxes = np.zeros((1, npix, npix))
        geometry = (pixel_scale_mas, rotation_deg)
        norms = _atom_norms(fixed, obs, fluxes, *geometry)
        residuals = _clean_residuals(fixed, obs, *geometry)
        r, vjp = jax.vjp(residuals, fluxes)
        (gradient,) = vjp(2.0 * r)
        norms, gradient = onp.asarray(norms), onp.asarray(gradient)
    live = norms > (100.0 * onp.finfo(float).eps) ** 2 * norms.max()
    safe = onp.where(live, norms, 1.0)
    return onp.where(live, -gradient / onp.sqrt(safe), 0.0)[0]


@dataclasses.dataclass(frozen=True)
class Diagnosis:
    """The result of :func:`diagnose`.

    Attributes
    ----------
    checks : dict
        Check name to value. Checks made per dataset or per Image are lists,
        in the order of the data or of the Images in the model.
    warnings : list of str
        One sentence for each problem found, saying what to do about it.
        Empty if none.
    """

    checks: dict
    warnings: list

    def __str__(self):
        def show(value):
            if isinstance(value, (list, tuple)):
                return "[" + ", ".join(show(v) for v in value) + "]"
            return f"{value:.3g}" if isinstance(value, float) else str(value)

        width = max(map(len, self.checks))
        lines = [f"{k:<{width}}  {show(v)}" for k, v in self.checks.items()]
        if self.warnings:
            lines += ["", *(f"Warning: {w}" for w in self.warnings)]
        else:
            lines += ["", "No warnings."]
        return "\n".join(lines)


def _parts(model, path=None):
    """``(path, part)`` for ``model`` and everything nested in it.

    The path is ``None`` for the model itself, else e.g. ``"env"``.
    """
    model = model.resolved()
    found = [(path, model)]
    prefix = "" if path is None else path + "."
    if isinstance(model, System):
        for name, part in model.components.items():
            found += _parts(part, prefix + name)
    elif isinstance(model, Rotated):
        found += _parts(model.source, prefix + "source")
    return found


def _map_images(model, fn):
    """``model`` with ``fn`` applied to each of its Images."""
    model = model.resolved()
    if isinstance(model, Image):
        return fn(model)
    if isinstance(model, Rotated):
        return eqx.tree_at(
            lambda r: r.source, model, _map_images(model.source, fn)
        )
    if not isinstance(model, System):
        return model
    parts = tuple(_map_images(c, fn) for c in model.components.values())
    return eqx.tree_at(lambda s: tuple(s.components.values()), model, parts)


@eqx.filter_jit
def _jitted_chi2(model, observations):
    return np.stack(
        [np.sum(whitened_residuals(model, d) ** 2) for d in observations]
    )


def _chi2(model, observations):
    """χ² of each dataset, jitted so that :func:`diagnose` compiles once."""
    return [float(c) for c in _jitted_chi2(model, tuple(observations))]


@renamed()
def diagnose(model, data, regularizers=()):
    """Check a model and its data for common imaging pitfalls.

    Nothing is printed or warned: print the returned
    [`Diagnosis`][virgil.imaging.Diagnosis]. Run it on the fitted model, with
    the fit's regularizers.

    Parameters
    ----------
    model : SourceModel
        The model, normally containing at least one
        [`Image`][virgil.models.Image] (the Image checks are skipped
        otherwise).
    data : OIData or sequence of OIData
        The data.
    regularizers : sequence, optional
        The regularizers of the fit; a
        [`Centroid`][virgil.imaging.Centroid] fixes the position.

    Returns
    -------
    Diagnosis
        ``checks`` holds, per dataset, ``chi2_red`` (χ² per data point),
        ``phase_regime`` (fraction of model visibilities with |arg V| > 0.8π
        or |V| < 0.05, for projected phases only), ``chi2_red_by_observable``
        (see [`chi2_by_observable`][virgil.imaging.chi2_by_observable]) and
        ``short_baseline_mean`` (see
        [`short_baseline_residual`][virgil.imaging.short_baseline_residual];
        the shortest 20% of baselines, ``None`` for ``vis_mat`` operators such
        as kernel-phase and DISCO data, or calibration gains); per Image, ``pixel_scale_mas``,
        ``edge_flux`` (fraction of the flux in the outer 2 pixels) and
        ``centroid_mas`` (East, North offset); and ``anchored`` (whether
        something fixes the position) and ``flip_dchi2`` (Δχ² when every Image
        is rotated by 180° about the origin).
    """
    observations = _as_list(data)
    parts = _parts(model)
    images = [(path, part) for path, part in parts if isinstance(part, Image)]
    checks, warns = {}, []

    chi2 = _chi2(model, observations)
    checks["chi2_red"] = [
        c / d.n_independent for c, d in zip(chi2, observations)
    ]
    for i, red in enumerate(checks["chi2_red"]):
        if red > 2.0:
            warns.append(
                f"Dataset {i} has chi2 per point {red:.2f} > 2: the model "
                "under-fits or the errors are underestimated; refit, or "
                "check the error bars."
            )
        elif red < 0.5:
            warns.append(
                f"Dataset {i} has chi2 per point {red:.2f} < 0.5: the model "
                "over-fits or the errors are overestimated; regularize more "
                "or check the error bars."
            )

    checks["chi2_red_by_observable"] = [
        {
            k: v
            for k, v in chi2_by_observable(model, d)["chi2_red"].items()
            if k != "total"
        }
        for d in observations
    ]
    checks["short_baseline_mean"] = (
        None
        if _no_baselines(observations)
        else short_baseline_residual(model, observations)[0]
    )

    checks["anchored"] = (
        any(isinstance(part, PointSource) for _, part in parts)
        or any(isinstance(r, Centroid) for r in regularizers)
        or any(
            d.vis_encoding == "split" and not d.cp_flag and d.phi_mat is None
            for d in observations
        )
    )
    if images and not checks["anchored"]:
        warns.append(
            "Nothing fixes the position of the Image: closure, kernel and "
            "DISCO phases are blind to a shift. Add a PointSource star or a "
            "Centroid prior."
        )

    if images:
        nyquist = nyquist_pixel_scale(observations)
        checks["pixel_scale_mas"] = [im.pixel_scale_mas for _, im in images]
        checks["edge_flux"], checks["centroid_mas"] = [], []
        for (path, im), scale in zip(images, checks["pixel_scale_mas"]):
            label = path or "the Image"
            if scale > nyquist:
                warns.append(
                    f"{label} has pixels of {scale:.3g} mas, coarser than "
                    f"the Nyquist scale {nyquist:.3g} mas of the longest "
                    "baseline; use smaller pixels."
                )
            edge = float(1.0 - np.sum(im.smoothed()[2:-2, 2:-2]))
            checks["edge_flux"].append(edge)
            if edge > 0.05:
                warns.append(
                    f"{label} has {edge:.0%} of its flux in the outer 2 "
                    "pixels; enlarge the field or add a support."
                )
            x, y = Centroid(1.0, path).centroid(model)
            checks["centroid_mas"].append((float(x), float(y)))

        # Each Image turned by 180 degrees about the origin of the sky.
        flipped = _map_images(
            model,
            lambda im: dataclasses.replace(
                im,
                log_brightness=im.eta[::-1, ::-1],
                support=None if im.support is None else im.support[::-1, ::-1],
                dra=-im.dra,
                ddec=-im.ddec,
            ),
        )
        checks["flip_dchi2"] = sum(_chi2(flipped, observations)) - sum(chi2)
        if abs(checks["flip_dchi2"]) < 1.0:
            warns.append(
                "Rotating the Images by 180 degrees changes chi2 by "
                f"{checks['flip_dchi2']:.2g}: the data do not constrain "
                "the orientation (V^2-only data cannot); expect a mirror "
                "ambiguity."
            )
        elif checks["flip_dchi2"] <= -1.0:
            warns.append(
                "The Images rotated by 180 degrees fit better, by "
                f"{-checks['flip_dchi2']:.3g} in chi2: the fit may be "
                "trapped in a mirrored solution; refit from the rotated "
                "image."
            )

    regimes = []
    for d in observations:
        if d.phi_mat is None and d.vis_encoding != "mixed_log_complex":
            continue
        if d.bandwidth is None:
            cvis = model.model(d.u, d.v, d.wavel)
        else:
            cvis = smeared_model(model, d.u, d.v, d.wavel, d.bandwidth)
        bad = (np.abs(np.angle(cvis)) > 0.8 * np.pi) | (np.abs(cvis) < 0.05)
        regimes.append(float(np.mean(bad)))
    if regimes:
        checks["phase_regime"] = regimes
        if any(regimes):
            warns.append(
                "Some model visibilities have |phase| > 0.8 pi or |V| < "
                "0.05, where projected (kernel or DISCO) phases, built from "
                "wrapped phases, are unreliable; use a smaller field or "
                "fainter extended flux."
            )

    return Diagnosis(checks, warns)
