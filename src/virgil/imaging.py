"""Regularisers and helpers for image reconstruction.

An image fit is a call to [`fit`][virgil.fitting.fit] whose model contains an
[`Image`][virgil.models.Image], with a prior on its ``log_brightness``
(see :func:`image_priors`) and usually a regulariser, which adds a penalty
on the pixel fluxes ``b`` to the loss:

| Regulariser | Penalty | Least squares (LM)? | A prior density? |
| --- | --- | --- | --- |
| [`TSV`][virgil.imaging.TSV] | ``w Σ (Δx b)² + (Δy b)²`` | yes | no |
| [`TV`][virgil.imaging.TV] | ``w Σ √((Δx b)² + (Δy b)² + ε²)`` | no | no |
| [`MaxEntropy`][virgil.imaging.MaxEntropy] | ``w Σ b log(b / q)`` | no | no |
| [`Laplacian`][virgil.imaging.Laplacian] | ``w Σ (∇²b)²`` | yes | no |
| [`StarletL1`][virgil.imaging.StarletL1] | ``w Σ √(s² + ε²)`` over starlet details ``s`` | no | no |
| [`LogSum`][virgil.imaging.LogSum] | ``w Σ log(1 + b / (ε b̄))`` | no | no |
| [`Centroid`][virgil.imaging.Centroid] | ``½ |centroid / σ|²`` | yes | yes |

Differences ``Δ`` are between neighbouring pixels, with zeros beyond the
edges, so edge pixels are penalised too. TSV (total squared variation) and
the Laplacian favour smooth images, TV (total variation) piecewise-flat ones,
and maximum entropy images close to a default ``q``. The weight ``w``
depends on the scene and the data; :func:`l_curve` sweeps it.

Sparse images, made of a few compact features, need a different penalty. An
L1 norm of the pixels does not work: they are positive and sum to one, so
``Σ |b| = 1`` for every image. ``StarletL1`` is an L1 norm of the image's
wavelet (starlet) coefficients instead, which favours images built from few
compact structures at any scale. ``LogSum`` is a smooth surrogate for the
number of bright pixels (the L0 "norm" of SQUEEZE). It is not convex, so the
fit can stop in a local minimum: start it from a good image, such as a
[`clean`][virgil.imaging.clean] model.

[`clean`][virgil.imaging.clean] builds a sparse image directly, from point
components added one at a time where the gradient of χ² is steepest: CLEAN
for any data, including closure and DISCO phases.
``design/sparse_imaging.md`` discusses the choices.

When [`fit`][virgil.fitting.fit]'s model function returns one model per
dataset, every regulariser acts on the **first model only**. That is right
when the later models are transformed copies of the same scene (e.g. a
[`Rotated`][virgil.models.Rotated] epoch), but an Image that appears only in
a later model is not regularised at all.

Closure, kernel and DISCO phases do not fix an image's position. Something
must: an analytic star at the origin, a [`Centroid`][virgil.imaging.Centroid]
prior, or a centred prior mean. [`diagnose`][virgil.imaging.diagnose]
checks a fit for this and other common pitfalls.
"""

import dataclasses

import equinox as eqx
import jax
import jax.numpy as np
import numpy as onp
from jax.scipy.signal import fftconvolve
from jax.scipy.special import xlogy

from ._geometry import pixel_offsets, rotate
from ._utils import _reference, mas2rad
from ._precision import cast_tree, run_in
from .fitting import FitResult, fit
from .spectra import flux_at
from .fields import GaussianField
from .likelihood import whitened_residuals
from .models import (
    Image,
    PointSource,
    Rotated,
    SourceModel,
    System,
    _pixel_visibilities,
    circular_support,
)


class _ImageRegulariser(eqx.Module):
    """A penalty on the pixels of one Image, found at ``path`` in the model."""

    weight: float
    path: str | None = eqx.field(static=True)

    def image(self, model):
        """The regularised [`Image`][virgil.models.Image] in ``model``."""
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

    A quadratic smoothness penalty, so it can be fitted by
    Levenberg–Marquardt.

    Parameters
    ----------
    weight : float
        Strength of the penalty.
    path : str, optional
        Path of the Image in the model (e.g. ``"env"``); ``None`` if the
        model is the Image itself.
    """

    def __init__(self, weight, path=None):
        self.weight = np.asarray(weight, dtype=float)
        self.path = path

    def residuals(self, model):
        dx, dy = _differences(self.image(model).brightness)
        return np.sqrt(2.0 * self.weight) * np.concatenate(
            [np.ravel(dx), np.ravel(dy)]
        )

    def value(self, model):
        return 0.5 * np.sum(self.residuals(model) ** 2)


class TV(_ImageRegulariser):
    """Total variation: ``weight * Σ √((Δx b)² + (Δy b)² + ε²)``.

    Favours piecewise-flat images with sharp edges. ``ε`` smooths the
    penalty where the image is flat, so that it is differentiable. The
    penalty is averaged over the four flips of the image, so it does not
    depend on which neighbour each forward difference pairs a pixel with,
    and is the same for an image and its mirror images. It is not
    isotropic: as for any total variation built from pixel differences,
    a sharp edge along a diagonal costs about a fifth more per unit
    length than one along a pixel axis.

    Parameters
    ----------
    weight : float
        Strength of the penalty.
    epsilon : float, optional
        Smoothing scale, as a fraction of the mean pixel flux (default
        1e-2).
    path : str, optional
        Path of the Image in the model.
    """

    epsilon: float

    def __init__(self, weight, epsilon=1e-2, path=None):
        self.weight = np.asarray(weight, dtype=float)
        self.epsilon = np.asarray(epsilon, dtype=float)
        self.path = path

    def value(self, model):
        b = self.image(model).brightness
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

    The relative entropy of the image with respect to a default image ``q``.
    It is zero for ``b = q`` and positive otherwise.

    Parameters
    ----------
    weight : float
        Strength of the penalty.
    prior : array-like, optional
        The default image ``q``, with the image's shape and positive where
        the image's support is; normalised here. Default: flat over the
        support.
    path : str, optional
        Path of the Image in the model.
    """

    prior: object

    def __init__(self, weight, prior=None, path=None):
        self.weight = np.asarray(weight, dtype=float)
        self.prior = None if prior is None else np.asarray(prior)
        self.path = path

    def value(self, model):
        image = self.image(model)
        b = image.brightness
        if self.prior is None:
            inside = (
                np.ones(b.shape, bool)
                if image.support is None
                else image.support
            )
            q = inside / np.sum(inside)
        else:
            q = np.asarray(self.prior) / np.sum(self.prior)
        q = np.where(b > 0.0, q, 1.0)  # 0 log 0 = 0 outside the support
        return self.weight * np.sum(xlogy(b, b) - xlogy(b, q))


class Laplacian(_ImageRegulariser):
    """Squared Laplacian: ``weight * Σ (∇²b)²``.

    ``∇²b`` is the five-point Laplacian, with zeros beyond the edges (so it
    is evaluated on a ring of pixels around the image too). A quadratic
    penalty on curvature, smoother than TSV, so it can be fitted by
    Levenberg–Marquardt.

    Parameters
    ----------
    weight : float
        Strength of the penalty.
    path : str, optional
        Path of the Image in the model.
    """

    def __init__(self, weight, path=None):
        self.weight = np.asarray(weight, dtype=float)
        self.path = path

    def residuals(self, model):
        padded = np.pad(self.image(model).brightness, 2)
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


def _smooth(image, step):
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
    2010): the image is smoothed ``scales`` times, the kernel's taps twice
    as far apart each time, and each detail plane is the difference between
    successive smoothings. Detail plane ``j`` holds structure about
    ``2**j`` pixels across. Beyond the edges the image is zero.

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
        What is left after the last smoothing. ``details.sum(0) + coarse``
        is the image.
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

    ``s`` are the detail coefficients of :func:`starlet`, at every scale.
    An L1 norm favours few non-zero coefficients, so the image is built
    from few compact structures, of any size: a sparse image in the
    wavelet sense. The coarse plane, which carries the flux, is not
    penalised. ``ε`` smooths the penalty near zero, so that it is
    differentiable.

    Parameters
    ----------
    weight : float
        Strength of the penalty.
    scales : int, optional
        Number of detail planes (default 4); the largest holds structure
        about ``2**(scales - 1)`` pixels across.
    epsilon : float, optional
        Smoothing scale, as a fraction of the mean pixel flux (default
        1e-2).
    path : str, optional
        Path of the Image in the model.
    """

    scales: int = eqx.field(static=True)
    epsilon: float

    def __init__(self, weight, scales=4, epsilon=1e-2, path=None):
        if int(scales) != scales or scales < 1:
            raise ValueError(
                f"scales must be a positive integer, not {scales}."
            )
        self.weight = np.asarray(weight, dtype=float)
        self.scales = int(scales)
        self.epsilon = np.asarray(epsilon, dtype=float)
        self.path = path

    def value(self, model):
        b = self.image(model).brightness
        details, _ = starlet(b, self.scales)
        eps = self.epsilon / b.size
        return self.weight * np.sum(np.sqrt(details**2 + eps**2))


class LogSum(_ImageRegulariser):
    """Log-sum sparsity: ``weight * Σ log(1 + b / (ε b̄))``.

    ``b̄`` is the mean pixel flux over the support. A pixel costs about
    ``log(b / (ε b̄))`` once it is brighter than ``ε b̄``, and almost
    nothing when fainter, so as ``ε → 0`` the penalty counts the bright
    pixels: a smooth surrogate for the L0 "norm" of SQUEEZE (Candès, Wakin
    & Boyd 2008). It favours images with few bright pixels. It is not
    convex, so start the fit from a good image.

    Sweep its weight with ``l_curve(..., warm_start=False)``. By default
    :func:`l_curve` starts each fit from the previous, stronger one; a
    strong weight switches most pixels off, and pixels driven that dark (in
    log-brightness) cannot recover, so every weaker fit would inherit the
    collapse.

    Parameters
    ----------
    weight : float
        Strength of the penalty.
    epsilon : float, optional
        The flux, as a fraction of the mean pixel flux, at which a pixel
        starts to count (default 1e-2).
    path : str, optional
        Path of the Image in the model.
    """

    epsilon: float

    def __init__(self, weight, epsilon=1e-2, path=None):
        self.weight = np.asarray(weight, dtype=float)
        self.epsilon = np.asarray(epsilon, dtype=float)
        self.path = path

    def value(self, model):
        image = self.image(model)
        b = image.brightness
        n = b.size if image.support is None else np.sum(image.support)
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
        x = np.sum(b, axis=0) @ pixel_offsets(ncol, image.pixel_scale_mas)
        y = np.sum(b, axis=1) @ pixel_offsets(nrow, image.pixel_scale_mas)
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
    constrained only by the data and the regularisers. An Image with a
    [`GaussianField`][virgil.fields.GaussianField] gets standard-normal
    priors on the field's latents, ``{"env.log_brightness.latent":
    Normal(0, 1)}``: the Gaussian-process prior, which needs no regulariser.
    Add priors for any other free parameters (fluxes, offsets) to the dict.
    """
    import numpyro.distributions as dist

    priors = {}

    def visit(model, prefix):
        if isinstance(model, Image) and isinstance(
            model.log_brightness, GaussianField
        ):
            shape = model.log_brightness.shape
            priors[prefix + "log_brightness.latent"] = dist.Normal(
                np.zeros(shape), 1.0
            ).to_event(2)
        elif isinstance(model, Image):
            shape = model.log_brightness.shape
            priors[prefix + "log_brightness"] = dist.ImproperUniform(
                dist.constraints.real, (), event_shape=shape
            )
        elif isinstance(model, System):
            for name, part in model.components.items():
                visit(part, prefix + name + ".")
        elif isinstance(model, Rotated):
            visit(model.source, prefix + "source.")

    visit(scene, "")
    if not priors:
        raise ValueError("The scene contains no Image.")
    return priors


def nyquist_pixel_scale(data):
    """The largest pixel scale (mas) that samples the data's finest fringes.

    That is λ / (2 B) for the longest baseline B in units of wavelength,
    over all samples of ``data`` (an OIData or a sequence of them).
    Reconstructions normally use pixels 2–4 times smaller.
    """
    observations = data if isinstance(data, (list, tuple)) else [data]
    longest = max(
        float(np.max(np.hypot(d.u, d.v) / d.wavel)) for d in observations
    )
    return 1.0 / (2.0 * longest * mas2rad)


def field_of_view(data, largest_mas=500.0):
    """The default field of view for an image of ``data``, in mas.

    The smaller of ``largest_mas`` (500 by default) and the interferometric
    field of view, the largest λ / B over all samples with a non-zero
    baseline B (the shortest baseline at the longest wavelength, where they
    are observed together): structure larger than that is not measured,
    and on a uv lattice a larger field would alias. Use it with
    [`nyquist_pixel_scale`][virgil.imaging.nyquist_pixel_scale] to
    choose the image's size and pixels.
    """
    observations = data if isinstance(data, (list, tuple)) else [data]
    shortest = min(
        float(onp.min(rho[rho > 0]))
        for rho in (
            onp.ravel(onp.asarray(onp.hypot(d.u, d.v) / d.wavel))
            for d in observations
        )
    )
    return min(float(largest_mas), 1.0 / (shortest * mas2rad))


def _complex_visibilities(d):
    """Visibility estimates at a dataset's uv samples, and their weights.

    AMIGO DISCO data give the least-squares (minimum-norm) estimate of the
    log visibilities from the modes; data with absolute phases give the
    visibilities directly. The weights are 1 on samples that the data
    inform and 0 elsewhere (uniform weighting, as in :func:`beam`).
    """
    if d.observable_kind == "mixed_log_complex":
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
        # shortest informed baselines, as for any normalised source.
        rho = onp.hypot(onp.asarray(d.u), onp.asarray(d.v))[informed]
        shortest = rho <= onp.quantile(rho, 0.1)
        logv[:n] -= onp.mean(logv[:n][informed][shortest])
        return onp.exp(logv[:n] + 1j * logv[n:]), informed.astype(float)
    if (
        d.observable_kind == "split"
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


def dirty_image(data, npix, pixel_scale_mas, flux_ratio=None):
    """The dirty image: direct synthesis of the visibilities, unregularised.

    ``I(x) = Σ_k w_k Re[V_k exp(+2πi u_k · x)] / Σ_k w_k``, summing over the
    samples (each standing for itself and its conjugate) with uniform
    weights on the informed ones: the image convolved with the dirty beam,
    plus noise. It peaks at one for a lone point source. For AMIGO DISCO
    data the visibilities are the least-squares estimate from the modes;
    since the modes are blind to the total flux, the estimate is
    normalised to |V| = 1 on average on the shortest baselines.

    Parameters
    ----------
    data : OIData or sequence of OIData
        The data; see the error raised for data without phases.
    npix : int
        Number of pixels on a side.
    pixel_scale_mas : float
        Pixel size in mas.
    flux_ratio : float, optional
        If the scene is a star at the origin plus extended emission with
        this flux relative to the star, the star is removed first, so the
        map shows the extended emission alone. The star is removed by
        subtracting the best-fitting point source at the origin (the
        weighted mean of the visibilities), then scaling by
        ``(1 + flux_ratio) / flux_ratio``. That is robust to the
        normalisation of the visibilities, which matters because subtracting
        a bright star amplifies any error in it by ``1 / flux_ratio``.

    Returns
    -------
    jax.Array, shape (npix, npix)
        In the orientation of [`render`][virgil.models.SourceModel.render]
        (East left, North up). It has negative sidelobes.
    """
    observations = data if isinstance(data, (list, tuple)) else [data]
    offsets = onp.asarray(pixel_offsets(int(npix), float(pixel_scale_mas)))
    image, total = onp.zeros((int(npix), int(npix))), 0.0
    for d in observations:
        vis, weight = _complex_visibilities(d)
        if flux_ratio is not None:
            star = onp.sum(weight * onp.real(vis)) / onp.sum(weight)
            vis = (vis - star) * (1.0 + flux_ratio) / flux_ratio
        fu = onp.ravel(onp.asarray(d.u / d.wavel) * mas2rad)
        fv = onp.ravel(onp.asarray(d.v / d.wavel) * mas2rad)
        cols = onp.exp(2j * onp.pi * onp.outer(fu, offsets))  # (k, col)
        rows = onp.exp(2j * onp.pi * onp.outer(fv, offsets))  # (k, row)
        image += onp.real(onp.einsum("k,kr,kc->rc", weight * vis, rows, cols))
        total += weight.sum()
    return np.asarray(image / total)


def starting_image(
    data,
    star=True,
    oversample=4.0,
    largest_mas=None,
    start="moments",
    hole_mas=None,
):
    """A starting model for an image fit, sized from the data.

    It fits a quick parametric model, an analytic star (if ``star``) plus a
    circular Gaussian envelope with free flux and width, a low-order
    description of the visibilities, from a few starting widths. Then it
    chooses the image:

    * a field of about six FWHMs of the envelope, but at least 500 mas, and
      never larger than the interferometric field of view (λ / B_min) or
      ``largest_mas``;
    * pixels ``oversample`` times finer than the Nyquist scale;
    * an [`Image`][virgil.models.Image] with the fitted flux, whose
      pixels are the fitted Gaussian (``start="moments"``) or the positive
      part of the :func:`dirty_image` (``start="dirty"``, with the star
      removed). A dirty start is better when the Fourier coverage is dense
      (it already has the right shape) and worse when it is sparse (its
      sidelobes dominate).

    Fit it with ``fit(start, image_priors(start), data, regularisers)``,
    adding a prior on the image's flux, ``"env.flux"`` (or ``"flux"`` with
    ``star=False``), to fit it too: with a star, this stops the fit from
    parking excess flux next to it.

    Parameters
    ----------
    data : OIData or sequence of OIData
        The data.
    star : bool, optional
        Whether the scene has an unresolved star at the origin (which also
        fixes the image's position). Without one, the result is the Image
        alone, and a [`Centroid`][virgil.imaging.Centroid] prior
        should fix its position.
    oversample : float, optional
        Pixels per Nyquist pixel.
    largest_mas : float, optional
        A cap on the field of view.
    start : {"moments", "dirty"}, optional
        The starting pixels, as above.
    hole_mas : float, optional
        Radius of a hole in the image's support under the star. Extended
        flux within a fraction of a beam of the star is nearly
        indistinguishable from the star's own, so without a hole the fit can
        trade the two and bias the flux ratio; half the beam's minor axis
        (``0.5 * beam(data).minor_mas``) is a good choice.

    Returns
    -------
    SourceModel
        ``System(star=PointSource(), env=Image(...))``, or the Image alone.
    """
    import numpyro.distributions as dist

    from .models import GaussianDisk

    resolution = beam(data).major_mas
    widest = field_of_view(data, largest_mas=onp.inf)
    best = None
    for width in (0.25 * resolution, resolution, 4.0 * resolution):
        sigma = min(width, widest / 6.0) / 2.3548
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
    fwhm = 2.3548 * float(envelope.sigma)
    fov = min(max(500.0, 6.0 * fwhm), widest)
    if largest_mas is not None:
        fov = min(fov, float(largest_mas))
    scale = nyquist_pixel_scale(data) / float(oversample)
    npix = int(onp.ceil(fov / scale))
    support = None
    if hole_mas is not None:
        support = circular_support(npix, scale, npix * scale, hole_mas)
    options = dict(flux=envelope.flux, support=support)
    if start == "moments":
        # With a log-uniform prior, data with no resolved envelope drive
        # sigma to its (tiny) lower bound; a Gaussian narrower than a pixel
        # sampled on the grid is all zeros, so start at least one pixel wide.
        model = GaussianDisk(max(float(envelope.sigma), scale))
        image = Image.from_model(model, npix, scale, **options)
    elif start == "dirty":
        ratio = float(envelope.flux) if star else None
        dirty = dirty_image(data, npix, scale, flux_ratio=ratio)
        positive = np.maximum(dirty, 0.0)
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

    The dirty beam is the image of a point source made by the data's uv
    sampling. Near its peak it falls off as ``1 - 2π² xᵀ M x``, where ``M``
    is the weighted mean of ``u uᵀ`` over the samples, which matches a
    Gaussian of covariance ``M⁻¹ / 4π²``. Its FWHM ellipse is the standard
    "beam" drawn on interferometric images: roughly λ/B, and elongated where
    the coverage is. For complicated coverage the real beam can look quite
    different, but the ellipse still shows the size and shape of a
    resolution element.

    Every sample that carries information is weighted equally ("uniform
    weighting"); in AMIGO DISCO data those are the uv points on which the
    modes have weight (at least 1e-3 of the largest), i.e. the splodges.
    Weighting by information instead would let the low-frequency central
    splodge dominate, and give a broader beam.

    Parameters
    ----------
    data : OIData or sequence of OIData
        The data.

    Returns
    -------
    Beam
    """
    observations = data if isinstance(data, (list, tuple)) else [data]
    u, v, w = [], [], []
    for d in observations:
        fu = onp.ravel(onp.asarray(d.u / d.wavel) * mas2rad)
        fv = onp.ravel(onp.asarray(d.v / d.wavel) * mas2rad)
        weight = onp.ones_like(fu)
        if d.observable_kind == "mixed_log_complex":
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
    fwhm = 2.0 * onp.sqrt(2.0 * onp.log(2.0) * variance)
    x, y = axes[:, 1]  # the major axis, in (East, North)
    pa = onp.degrees(onp.arctan2(x, y)) % 180.0
    return Beam(float(fwhm[1]), float(fwhm[0]), float(pa))


def convolve_beam(image, pixel_scale_mas, beam):
    """An image convolved with a Gaussian beam: the image at the data's resolution.

    Reconstructed images are super-resolved. A regularised image can put
    structure on scales finer than the beam, where the data constrain it
    only weakly, so it often looks clumpy or streaky. Convolved with the
    beam (the "restored" image of radio astronomy), it shows only what the
    data resolve. That is the fair way to compare two reconstructions, or a
    reconstruction with a model: convolve both.

    The kernel is an elliptical Gaussian with the beam's FWHMs and position
    angle (North to East), normalised to unit sum. It is sampled on an odd
    grid centred on a pixel, so the convolution does not shift the image.
    Flux beyond the edge of the image is taken to be zero, and the result
    is cropped to the image, so the total flux is kept only for structure
    more than about a beam from the edge; structure nearer the edge is
    dimmed.

    Parameters
    ----------
    image : array-like, shape (ny, nx)
        The image, in the orientation of
        [`render`][virgil.models.SourceModel.render] (East left, North
        up).
    pixel_scale_mas : float
        Pixel size in milliarcseconds.
    beam : Beam
        The beam, usually [`beam(data)`][virgil.imaging.beam].

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
    fwhm_per_sigma = 2.0 * onp.sqrt(2.0 * onp.log(2.0))
    kernel = np.exp(
        -0.5
        * (
            (along * fwhm_per_sigma / beam.major_mas) ** 2
            + (across * fwhm_per_sigma / beam.minor_mas) ** 2
        )
    ).astype(image.dtype)
    return fftconvolve(image, kernel / np.sum(kernel), mode="same")


def _base_weight(base, wavel):
    """The flux the components are measured against: a System's total."""
    if isinstance(base, System):
        return sum(part._weight(wavel) for part in base.parts)
    return base._weight(wavel)


def _spectral_shape(spectrum, wavel):
    """The components' spectrum, 1 at its reference (``None``: grey)."""
    if spectrum is None:
        return 1.0
    return spectrum(wavel) / spectrum(None)


class _CleanScene(SourceModel):
    """A fixed base scene plus point components on a pixel grid.

    ``V = (w V_base + s Σ c_p e_p) / (w + s Σ c_p)``, where ``w`` is the
    base's weight (a System's total flux), ``c`` the components' fluxes,
    ``s`` the shape of their spectrum (1 if grey) and ``e_p`` the
    visibility of a point at pixel ``p``. It is smooth in ``c``, even at
    ``c = 0``, and equals the model ``clean`` returns. Without a base,
    ``V = Σ c_p e_p / Σ c_p``.
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
        weight = _base_weight(self.base, wavel)
        shape = _spectral_shape(self.spectrum, wavel)
        return (weight * base + shape * pixels) / (weight + shape * total)


def _clean_residuals(parts, observations, scale, rotation):
    """Whitened residuals of all the data, as a function of the fluxes.

    ``parts`` is ``(base, spectrum)``.
    """
    base, spectrum = parts

    def residuals(fluxes):
        scene = _CleanScene(base, fluxes, scale, rotation, spectrum)
        return np.concatenate(
            [np.ravel(whitened_residuals(scene, d)) for d in observations]
        )

    return residuals


@eqx.filter_jit
def _atom_norms(parts, observations, fluxes, scale, rotation):
    """``|J e_p|²`` for every pixel ``p``: the χ² response to its flux.

    One Jacobian–vector product per pixel, a row of pixels at a time, so
    the Jacobian is never held whole.
    """
    residuals = _clean_residuals(parts, observations, scale, rotation)
    nrow, ncol = fluxes.shape

    def row(i):
        def pixel(j):
            e = np.zeros(fluxes.shape, fluxes.dtype).at[i, j].set(1.0)
            return np.sum(jax.jvp(residuals, (fluxes,), (e,))[1] ** 2)

        return jax.vmap(pixel)(np.arange(ncol))

    return jax.lax.map(row, np.arange(nrow))


@eqx.filter_jit
def _clean_step(parts, observations, fluxes, scores, scale, rotation):
    """χ², and the best pixel and Gauss–Newton step for one CLEAN iteration.

    The best pixel lowers χ² most: the largest ``g_p² / |J e_p|²`` with
    ``g_p < 0``, where ``g`` is the gradient of χ² and ``scores`` holds
    ``1 / |J e_p|²`` (zero outside the support).
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
def _clean_columns(parts, observations, fluxes, indices, scale, rotation):
    """The residuals, and their derivatives by the fluxes at ``indices``."""
    residuals = _clean_residuals(parts, observations, scale, rotation)

    def column(k):
        e = np.zeros(fluxes.size, fluxes.dtype).at[k].set(1.0)
        return jax.jvp(residuals, (fluxes,), (e.reshape(fluxes.shape),))[1]

    return residuals(fluxes), jax.vmap(column)(indices)


def _refit(fixed, fluxes, chi2, scale, rotation):
    """A major cycle: refit the fluxes of the components, keeping them >= 0.

    Linearises the residuals about the current fluxes over the components
    (one Jacobian–vector product each), solves the non-negative least
    squares problem for their fluxes, and backtracks along the step until
    χ² falls. Components it sets to zero are removed. Returns the new
    fluxes, or the old ones if no step lowers χ².
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
        # nnls gives up ("Maximum number of iterations reached") on the
        # ill-conditioned columns of high signal-to-noise data. The bounded
        # least-squares solver returns its best point instead of raising,
        # and the backtracking below keeps the step only if χ² falls.
        # BVLS, not the default TRF: TRF's line search loops forever once
        # its step underflows to zero, which hung CI on scipy 1.13.
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
    """The base with the components as an Image: the model ``clean`` returns.

    ``components`` are the fluxes on the grid (unnormalised without a
    base). With no flux in them, the base alone.
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


def _joint_refit(
    base, components, observations, priors, geometry, spectrum, dtype
):
    """Fit the base's free parameters and the components' fluxes together.

    ``priors`` are keyed by paths in the base. The components keep their
    support (their fluxes stay positive). Returns ``(base, components)``.
    Fitting the two together, rather than in turn, keeps the components from
    absorbing an error in the base, and the base from absorbing flux it does
    not model.
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

    Attributes
    ----------
    model : SourceModel
        The base scene with the components, ``System(base=base,
        clean=Image(...))``, or for a System base its components and
        ``clean`` side by side. The Image is non-zero only on the
        components, and its ``flux`` is their total relative to the base
        (with the components' spectrum, if one was given). Without a base
        scene, the Image alone; with no components, the base alone.
    components : array, shape (npix, npix)
        The components' fluxes on the pixel grid, relative to the base
        scene's weight (without a base scene, normalised to unit sum).
    pixel_scale_mas : float
        Pixel size in milliarcseconds.
    chi2_red : array
        χ² per data point before each iteration; the last entry is that of
        the final model.
    stop : str
        Why CLEAN stopped: ``"target"`` (χ² per point reached
        ``target_chi2_red``), ``"stalled"`` (χ² stopped falling, even after
        a major cycle) or ``"max_iterations"``.
    """

    model: object
    components: jax.Array
    pixel_scale_mas: float
    chi2_red: jax.Array
    stop: str

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
):
    """Build an image from point components, one at a time: gradient CLEAN.

    Högbom's CLEAN repeatedly finds the peak of the residual dirty image
    and adds a fraction (the loop ``gain``) of a point source there. For
    data that are linear in the image, the residual dirty image is
    proportional to ``-∂χ²/∂c``, the gradient of χ² with respect to the
    flux ``c_p`` of a point at each pixel. This function uses that gradient
    directly, so it works for any data virgil can fit: closure phases,
    kernel or DISCO phases, squared visibilities, or a mix. ``J e_p`` is
    the change in the whitened residuals per unit flux at pixel ``p``. Each
    iteration picks the pixel whose Gauss–Newton step would lower χ² most,
    the largest ``g_p² / |J e_p|²`` with ``g_p < 0``, and adds ``gain``
    times that step, ``-g_p / (2 |J e_p|²)``. This is matching pursuit; for
    linear data, where ``|J e_p|`` is the same everywhere, it is exactly
    Högbom's CLEAN. The norms ``|J e_p|`` are computed once, at the start,
    with one Jacobian–vector product per pixel. Normalising by them matters
    next to an analytic star, where flux in a pixel is nearly the same as
    the star's: the gradient there is small, but so is ``|J e_p|``, and an
    unnormalised search would pile flux beside the star.

    Components are added to a fixed ``base`` scene, usually an analytic
    star at flux 1; fit its parameters first. Their fluxes are relative to
    the base's, like a companion's in a System; for a
    [`System`][virgil.models.System] base they are siblings of its
    components, relative to its total. By default the components are grey,
    the same fraction of the base's flux at every wavelength; give a
    ``spectrum`` to make them follow one, as the image does in SPARCO
    (e.g. a star with its own spectral index and an environment with
    another). Without a base, the components alone make the image, starting from one at the centre of the
    grid (closure phases do not fix the position, so this is also the
    anchor).

    Each iteration only adds flux, so an early step that overshoots cannot
    be undone by later ones. Every ``refit_every`` iterations a **major
    cycle** (as in Clark and Cotton–Schwab CLEAN) refits the fluxes of all
    the components at once, by non-negative least squares on the
    linearised residuals: flux can move between components, and components
    whose flux falls to zero are removed. The fluxes stay non-negative.
    A last major cycle runs when CLEAN stops at the target or at
    ``max_iterations``, so that the fluxes it returns are refitted even when
    it stops early (when the data start close to the target, a major cycle
    may otherwise never run).

    The base is fixed unless ``base_priors`` names some of its parameters,
    e.g. a companion's position or a star's diameter. Each major cycle then
    fits those together with the components' fluxes, with
    [`fit`][virgil.fitting.fit]: fitting them in turn instead would let the
    components absorb an error in the base, or let the base absorb flux it
    does not model. A companion much closer to the star than λ/B has its
    flux and separation nearly degenerate, whatever fits it.

    Iteration stops when χ² per data point reaches ``target_chi2_red`` (the
    discrepancy principle; it relies on correct error bars), or after
    ``max_iterations``. It also stops when χ² has **stalled**: when no
    pixel lowers it, or when it has fallen by less than a fraction
    ``stall_tolerance`` over the last ``stall_window`` iterations, and a
    major cycle does not help. That happens when noise puts the truth's own
    χ² per point above the target; ``chi2_red`` shows how close it came.

    Parameters
    ----------
    data : OIData or sequence of OIData
        The data, fitted jointly.
    npix : int
        Pixels on a side of the grid.
    pixel_scale_mas : float
        Pixel size in milliarcseconds.
    base : SourceModel, optional
        A fixed scene the components are added to (default: none). A
        System base must not be offset (put any offset on its components).
    gain : float, optional
        Loop gain, the fraction of each step taken (default 0.1). Smaller
        is slower but less likely to put flux in the wrong place.
    max_iterations : int, optional
        Iteration limit (default 1000).
    target_chi2_red : float, optional
        χ² per data point at which to stop (default 1).
    refit_every : int, optional
        Iterations between major cycles (default 50); 0 for none.
    stall_window, stall_tolerance : int and float, optional
        Stop when χ² has fallen by less than a fraction
        ``stall_tolerance`` (default 1e-3) over the last ``stall_window``
        (default 50) iterations, even after a major cycle.
    base_priors : dict, optional
        Priors (numpyro distributions) on parameters of the base to fit at
        every major cycle, keyed by their paths in the base (e.g.
        ``{"comp.dra": dist.Normal(30.0, 5.0)}`` for a System base, whose
        paths must start with a component's name, or ``{"diameter": ...}``
        for a single component). Needs a base and major cycles
        (``refit_every`` > 0). The fits run in ``dtype``.
    refresh_norms : bool, optional
        Recompute the norms ``|J e_p|`` at every major cycle (default
        ``False``). For non-linear data they change as the image does, but
        each refresh costs one Jacobian–vector product per pixel, computed a
        row of pixels at a time.
    spectrum : Spectrum, optional
        The spectrum all the components share, e.g.
        ``PowerLaw(1.0, index, wavel0)``; only its shape matters, and the
        component fluxes are at its reference wavelength. Default: grey.
        Needs a base.
    support : array-like of bool, shape (npix, npix), optional
        Pixels allowed to receive components (default: all), e.g. a
        [`circular_support`][virgil.models.circular_support] with a hole
        under the star.
    init : array-like, shape (npix, npix), optional
        Starting component fluxes, non-negative and zero outside
        ``support`` (default: none with a base, and without one a single
        component on the supported pixel nearest the centre).
    rotation_deg : float, optional
        Position angle of the grid's "up" axis, as for
        [`Image`][virgil.models.Image]; match the data's uv lattice
        (``data.uv_grid.rotation_deg``) for the fast exact transform.
    dtype : {"float64", "float32"}, optional
        Precision of the iterations, as for [`fit`][virgil.fitting.fit].

    Returns
    -------
    CleanResult
        The model, components and χ² history. The model can be polished
        with [`fit`][virgil.fitting.fit] on the components' support, used
        as a starting image for a regularised fit, or restored with the
        beam.
    """
    observations = tuple(data) if isinstance(data, (list, tuple)) else (data,)
    npix, pixel_scale_mas = int(npix), float(pixel_scale_mas)
    rotation_deg = float(rotation_deg)
    if not 0.0 < gain <= 1.0:
        raise ValueError(f"gain must be in (0, 1], not {gain}.")
    if int(max_iterations) != max_iterations or max_iterations < 0:
        raise ValueError(
            f"max_iterations must be a non-negative integer, not "
            f"{max_iterations}."
        )
    max_iterations = int(max_iterations)
    for name, value in (
        ("refit_every", refit_every),
        ("stall_window", stall_window),
    ):
        if int(value) != value or value < 0:
            raise ValueError(
                f"{name} must be a non-negative integer, not {value}."
            )
    refit_every, stall_window = int(refit_every), int(stall_window)
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
        init = onp.zeros(shape)
        if base is None:  # the supported pixel nearest the centre
            offsets = pixel_offsets(npix, 1.0)
            radius = onp.hypot(offsets[None, :], offsets[:, None])
            centre = onp.argmin(onp.where(support, radius, onp.inf))
            init.flat[centre] = 1.0
    init = onp.asarray(init, float)
    if init.shape != shape or not onp.all(onp.isfinite(init) & (init >= 0)):
        raise ValueError(
            f"init must have shape {shape} and be finite and non-negative."
        )
    if onp.any(init[~support] > 0):
        raise ValueError("init has flux outside the support.")
    if base is None and not init.sum() > 0:
        raise ValueError("Without a base scene, init needs a positive pixel.")
    ndata = sum(d.n_independent for d in observations)
    with run_in(dtype):
        cast_observations = cast_tree(observations, dtype)
        fixed = (cast_tree((base, spectrum), dtype), cast_observations)
        fluxes = np.asarray(init, dtype)
        geometry = (pixel_scale_mas, rotation_deg)

        def scores_for(fluxes):
            norms = _atom_norms(*fixed, fluxes, *geometry)
            # A pixel whose flux cannot change the model has |J e_p| = 0
            # exactly, but rounding leaves ~eps |J e|max there: without a
            # base scene the seed pixel is such a pixel (flux added to the
            # only component leaves the normalised image unchanged), and
            # on an even grid its phase factors are not exactly 1. Its
            # score 1/|J e_p|² would be ~1e24 and, on any noise in the
            # gradient, win the search and take an infinite step. Pixels
            # with |J e_p| below 100 eps of the largest are dead.
            eps = float(np.finfo(norms.dtype).eps)
            live = norms > (100.0 * eps) ** 2 * np.max(norms)
            return np.where(
                support & live, 1.0 / np.where(live, norms, 1.0), 0.0
            )

        def major_cycle(fluxes, chi2):
            # With free base parameters, first fit those and the components'
            # fluxes together; then refit the fluxes by non-negative least
            # squares, which can remove components. Pruning against the
            # corrected base, not the old one, keeps a component the old
            # base was hiding.
            nonlocal base, fixed
            if base_priors:
                base, components = _joint_refit(
                    base,
                    onp.asarray(fluxes, float),
                    observations,
                    base_priors,
                    geometry,
                    spectrum,
                    dtype,
                )
                fixed = (cast_tree((base, spectrum), dtype), cast_observations)
                fluxes = np.asarray(components, fluxes.dtype)
                chi2 = float(_clean_chi2(*fixed, fluxes, *geometry))
            return _refit(fixed, fluxes, chi2, *geometry)

        scores = scores_for(fluxes)
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
            stalled = not float(decrease) > 0.0 or (
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
            fluxes = fluxes.ravel().at[p].add(gain * step).reshape(shape)
            since_refit += 1
        if refit_every and since_refit > 0:
            # A last major cycle, so the returned fluxes are refitted even
            # if CLEAN reached the target before any major cycle ran.
            fluxes = major_cycle(fluxes, float(chi2))
            chi2 = _clean_chi2(*fixed, fluxes, *geometry)
            history.append(float(chi2) / ndata)
        components = onp.asarray(fluxes, float)
    if base is None:
        components = components / components.sum()
    model = _clean_model(
        base, components, pixel_scale_mas, rotation_deg, spectrum
    )
    return CleanResult(
        model,
        np.asarray(components),
        pixel_scale_mas,
        np.asarray(history),
        stop,
    )


@dataclasses.dataclass(frozen=True)
class LCurve:
    """The result of :func:`l_curve`.

    Attributes
    ----------
    weights : array
        The regularisation weights, in the order fitted (largest first).
    chi2 : array
        Total χ² of each fit.
    chi2_red : array, shape (n_weights, n_datasets)
        χ² per data point of each dataset.
    penalty : array
        The unweighted regulariser, ``value / weight``, of each fit.
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

        The curve is ``(log χ², log penalty)`` parametrised by ``log w``;
        the corner is the interior point of largest curvature (Hansen &
        O'Leary 1993). Check it by eye: the curvature of a sparse or noisy
        sweep is itself noisy, and a smooth curve has no clear corner.
        """
        t = np.log(self.weights)
        x, y = np.log(self.chi2), np.log(self.penalty)
        if t.size < 3:
            raise ValueError("The sweep needs at least three weights.")
        dx, dy = np.gradient(x, t), np.gradient(y, t)
        ddx, ddy = np.gradient(dx, t), np.gradient(dy, t)
        curvature = (dx * ddy - ddx * dy) / (dx**2 + dy**2) ** 1.5
        # The end points have one-sided derivatives; leave them out.
        return float(self.weights[1 + int(np.argmax(np.abs(curvature[1:-1])))])

    def discrepancy(self, target=1.0):
        """The weight at which χ² per data point reaches ``target``.

        This is Morozov's discrepancy principle: regularise as strongly as
        the data allow. With several datasets, the binding one (the largest
        χ² per point) must reach the target, so that a well-fitted dataset
        cannot hide a badly fitted one. It interpolates linearly in ``log w`` between the
        fitted weights, and returns ``None`` if the sweep never crosses the
        target. The truth itself has χ² per point of 1 ± √(2/N), so the
        default target of 1 is the natural one. It relies on correct error
        bars; with underestimated errors it over-regularises.
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

        For a sweep of [`MaxEntropy`][virgil.imaging.MaxEntropy] fits,
        this is the weight ``w`` at which ``-2 w S`` equals the number of
        well-measured directions in the image, ``N = Σ λ / (λ + w)``
        (Gull 1989; Skilling 1989). Here ``S`` is the entropy (minus
        the sweep's ``penalty``), and the ``λ`` are the eigenvalues of the
        data's Gauss–Newton curvature in the entropy metric,
        ``diag(√b) JᵀJ diag(√b)``, with ``J`` the Jacobian of the whitened
        residuals with respect to the brightness ``b``. It is the stationary
        point of the Laplace-approximated evidence in ``w``, and so assumes
        correct error bars, like the discrepancy principle.

        Both sides are evaluated at each fit in the sweep, and the crossing
        is interpolated linearly in ``log w``. Returns ``None`` if the sweep
        does not bracket it. Like :func:`log_evidence`, it uses the data's
        quoted errors, so it rejects a sweep fitted with ``noise=`` terms
        or with one model per dataset.

        Parameters
        ----------
        data : OIData or sequence of OIData
            The data the sweep was fitted to.
        path : str, optional
            Path of the regularised Image in the model (default ``"env"``).

        References
        ----------
        - S. F. Gull (1989), "Developments in maximum entropy data
          analysis", in *Maximum Entropy and Bayesian Methods*, Kluwer,
          53–71.
        - J. Skilling (1989), "Classic maximum entropy", in *Maximum
          Entropy and Bayesian Methods*, Kluwer, 45–52. (Skilling & Bryan
          1984 is the earlier "historic" MaxEnt, which stops at χ² = N.)
        """
        gap = []
        for weight, penalty, result in zip(
            self.weights, self.penalty, self.results
        ):
            model = _single_model(result, "classic_maxent")
            image = model.get(path)
            if isinstance(image.log_brightness, GaussianField):
                raise TypeError("classic_maxent needs a pixel Image.")
            _, jac = _residual_jacobian(model, data, path + ".log_brightness")
            # dr/db = (dr/dη) / b on the support, so √b dr/db = (dr/dη) / √b.
            with run_in("float64"):
                b = onp.ravel(
                    onp.asarray(
                        cast_tree(image, "float64").brightness, dtype=float
                    )
                )
            scaled = jac[:, b > 0] / onp.sqrt(b[b > 0])
            # Eigenvalues of the Gram matrix, as squared singular values:
            # forming the Gram matrix loses them to rounding (see
            # log_evidence).
            curvature = onp.linalg.svd(scaled, compute_uv=False) ** 2
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


# Basis vectors pushed through together when building a residual Jacobian.
_JACOBIAN_BATCH = 64


@eqx.filter_jit
def _jitted_residual_jacobian(model, datasets, path):
    """Jitted body of :func:`_residual_jacobian`.

    Jitted once at module level, with the model and data as arguments, so
    that a new fit of the same shape reuses the compilation; run eagerly,
    the Jacobian compiles hundreds of small operations one at a time.
    """
    leaf = model.get(path)

    def residuals(x):
        changed = model.set(path, x)
        return np.concatenate(
            [whitened_residuals(changed, d) for d in datasets]
        )

    n_data = sum(d.n_residuals for d in datasets)
    n_leaf = np.size(leaf)
    # Row by row (reverse mode) or column by column (forward mode), whichever
    # is shorter, in batches: jax.jacrev/jacfwd push every basis vector
    # through at once, and each carries the image transform's intermediates
    # (data x pixels), so their memory grows as data x data x pixels. A
    # MATISSE contest file needed 432 GB that way on an 80 GB GPU.
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


def _residual_jacobian(model, data, path):
    """All the whitened residuals, and their Jacobian with respect to a leaf.

    Returns NumPy arrays ``(residuals, jacobian)`` of shapes ``(n_data,)``
    and ``(n_data, leaf.size)``, computed in float64, with whichever of
    forward or reverse mode is cheaper for the Jacobian.
    """
    datasets = tuple(data) if isinstance(data, (list, tuple)) else (data,)
    with run_in("float64"):
        model, datasets = cast_tree((model, datasets), "float64")
        r, jac = _jitted_residual_jacobian(model, datasets, path)
        return onp.asarray(r, dtype=float), onp.asarray(jac, dtype=float)


def _single_model(model, caller):
    """The one model of a fit, for the evidence helpers.

    ``model`` is a model or a [`FitResult`][virgil.fitting.FitResult].
    The helpers use the data's quoted errors and set the Image in one
    model, so fitted ``noise=`` terms (found only in a FitResult) and
    per-dataset model lists are rejected rather than silently mishandled.
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


def log_evidence(model, data, path="env"):
    """Laplace-approximated log evidence of a Gaussian-field image fit.

    For an Image whose log-brightness is a
    [`GaussianField`][virgil.fields.GaussianField] with standard-normal
    latents ``z``, at the MAP ``model`` from [`fit`][virgil.fitting.fit],

    ``log Z ≈ -½ χ² - ½ |z|² - ½ log det(I + JᵀJ)``,

    up to a constant that is the same for every ``sigma`` and
    ``length_mas`` on a given grid. ``J`` is the Jacobian of the whitened
    residuals with respect to ``z``, so ``JᵀJ`` is the Gauss–Newton
    curvature of the likelihood. Other fitted parameters (fluxes, spectra)
    are held at their MAP values. Compare it across fits with different
    hyperparameters and choose the largest, as MacKay's evidence framework
    does; it is exact for a linear model, and assumes correct error bars.
    It uses the data's quoted errors, so it does not support fits with
    ``noise=`` terms, nor fits with one model per dataset.

    Parameters
    ----------
    model : SourceModel or FitResult
        The MAP model, or better the [`FitResult`][virgil.fitting.FitResult]
        itself, so that fitted ``noise=`` terms are caught (a ``ValueError``).
        A list of models raises a ``TypeError``.
    data : OIData or sequence of OIData
        The data it was fitted to.
    path : str, optional
        Path of the Image in the model (default ``"env"``).

    Returns
    -------
    float
    """
    model = _single_model(model, "log_evidence")
    image = model.get(path)
    if not isinstance(image.log_brightness, GaussianField):
        raise TypeError(
            f"log_evidence needs an Image with a GaussianField at {path!r}."
        )
    latent_path = path + ".log_brightness.latent"
    r, jac = _residual_jacobian(model, data, latent_path)
    chi2 = float(r @ r)
    z = onp.asarray(model.get(latent_path), dtype=float)
    # log det(I + JᵀJ) = Σ log(1 + s²) over the singular values s of J. A
    # Cholesky factor of I + JJᵀ fails on high signal-to-noise data: the
    # Gram matrix is low rank with a huge norm, and its rounding error
    # (~ε‖JJᵀ‖) can exceed the identity, leaving I + JJᵀ numerically
    # indefinite. The singular values avoid forming it.
    singular = onp.linalg.svd(jac, compute_uv=False)
    logdet = float(onp.sum(onp.log1p(singular**2)))
    return float(-0.5 * chi2 - 0.5 * onp.sum(z**2) - 0.5 * logdet)


def error_scale(model, data, path="env"):
    r"""Re-estimate the scale of the error bars from a Gaussian-field fit.

    **What it does.** It estimates the factor ``s`` by which every error
    bar should be multiplied for the data to be consistent with the fit.
    ``s < 1`` means the error bars are too large; ``s > 1`` that they are
    too small, or that the model is missing something.

    **The idea.** In MacKay's evidence framework, the level of the noise is
    a hyperparameter, like the prior's ``sigma`` and ``length_mas``. Write
    the noise precision as β = 1/s², so that the likelihood is
    ``exp(-β χ²/2)``, with χ² computed with the quoted errors. The
    Laplace-approximated evidence, as a function of β, is maximised when

    $$\frac{1}{\beta} = s^2 = \frac{\chi^2}{N - \gamma},
    \qquad \gamma = \sum_i \frac{\beta\lambda_i}{1 + \beta\lambda_i}.$$

    ``N`` is the number of data. ``γ`` is the **effective number of
    parameters** the data measure: the ``λ_i`` are the eigenvalues of the
    Gauss–Newton curvature ``JᵀJ`` of the likelihood (with the quoted
    errors) in the field's whitened latents, in which the prior's curvature
    is the identity, so that ``βλ_i`` are those of the rescaled likelihood.
    A direction with ``βλ ≫ 1`` is fixed by the data and counts as one
    parameter; one with ``βλ ≪ 1`` is fixed by the prior and counts as none.
    Since ``γ`` depends on ``β``, the equation is solved for ``β`` at the
    fitted model (by Newton's method, which converges monotonically from
    ``β = 0`` because ``γ`` is concave in ``β``).

    Each measured parameter uses up one datum's worth of scatter. So of the
    ``N`` residuals, only ``N − γ`` are free to scatter, and an honest error
    bar gives χ² ≈ N − γ, not N. The ordinary "χ² per point" estimate,
    ``s² = χ²/N``, is biased low for the same reason as the 1/N estimate of
    a sample variance; this is its Bayesian, nonlinear generalisation. It is
    MacKay's re-estimation formula for β (MacKay 1992, eq. 4.10, with γ
    from eq. 4.9; Bishop 2006, eqs. 3.91–3.95).

    **How to use it.** Fit at your chosen hyperparameters, call this, rescale
    the data with
    [`OIData.with_error_scale`][virgil.oidata.OIData.with_error_scale],
    and refit. The fit itself depends on the errors, so in principle
    fitting and rescaling is a fixed-point iteration; in practice one
    rescaling usually suffices. Error bars that are too large make the
    discrepancy principle, classic MaxEnt and the evidence all
    over-regularise, so rescale before choosing hyperparameters with any of
    them. The estimate assumes the model is adequate: if the data contain
    structure the model cannot fit, ``s`` absorbs it.

    Only the field's latents are counted in ``γ``. Each other fitted
    parameter (the image's flux, a star's position, a spectral index) that
    the data measure lowers ``N − γ`` by about one more, and so raises
    ``s`` by a fraction of about 1/(2N). That is negligible while such
    parameters are few compared with the data, as in every SPARCO fit.

    It uses the data's quoted errors, so it does not support fits with
    ``noise=`` terms (which estimate the errors another way), nor fits with
    one model per dataset.

    Parameters
    ----------
    model : SourceModel or FitResult
        The MAP model, whose Image at ``path`` has a GaussianField
        log-brightness, or better the
        [`FitResult`][virgil.fitting.FitResult] itself, so that fitted
        ``noise=`` terms are caught (a ``ValueError``). A list of models
        raises a ``TypeError``.
    data : OIData or sequence of OIData
        The data it was fitted to.
    path : str, optional
        Path of the Image in the model (default ``"env"``).

    Returns
    -------
    float
        The scale ``s``.

    References
    ----------
    - D. J. C. MacKay (1992), "Bayesian interpolation", Neural Computation
      4, 415–447, [doi:10.1162/neco.1992.4.3.415](https://doi.org/10.1162/neco.1992.4.3.415).
      It introduces the evidence framework, γ, and the re-estimation of
      α and β.
    - C. M. Bishop (2006), *Pattern Recognition and Machine Learning*,
      §3.5, "The evidence approximation" ([free PDF](https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/)):
      the same results for linear models, eqs. 3.91–3.95.
    - S. F. Gull (1989), "Developments in maximum entropy data analysis",
      in *Maximum Entropy and Bayesian Methods*, Kluwer, 53–71: the same
      ``N − γ`` argument for maximum entropy, the basis of
      [`LCurve.classic_maxent`][virgil.imaging.LCurve.classic_maxent].
    """
    model = _single_model(model, "error_scale")
    image = model.get(path)
    if not isinstance(image.log_brightness, GaussianField):
        raise TypeError(
            f"error_scale needs an Image with a GaussianField at {path!r}."
        )
    r, jac = _residual_jacobian(model, data, path + ".log_brightness.latent")
    chi2 = float(r @ r)
    # N counts independent data, not residuals: correlated closure phases
    # add penalty residuals (OIData.n_residuals) that are not observations.
    datasets = data if isinstance(data, (list, tuple)) else [data]
    n_data = sum(d.n_independent for d in datasets)
    # The Gram matrix's eigenvalues, as squared singular values (see
    # log_evidence): never negative, unlike eigvalsh of a rounded JJᵀ.
    lam = onp.linalg.svd(jac, compute_uv=False) ** 2
    # Solve g(β) = β χ² + γ(β) − N = 0. g is increasing and concave, so
    # Newton's method from β = 0 (where g = −N) rises monotonically to the
    # root.
    beta = 0.0
    for _ in range(100):
        g = beta * chi2 + onp.sum(beta * lam / (1.0 + beta * lam))
        slope = chi2 + onp.sum(lam / (1.0 + beta * lam) ** 2)
        step = (g - n_data) / slope
        beta -= step
        if abs(step) <= 1e-12 * beta:
            break
    return float(1.0 / onp.sqrt(beta))


def l_curve(
    model,
    priors,
    data,
    regulariser,
    weights,
    others=(),
    *,
    warm_start=True,
    **fit_options,
):
    """Fit a model over a range of weights for one regulariser.

    The weights are fitted from largest to smallest, each starting from the
    previous solution, which is faster and more stable than starting every
    fit afresh. Plot ``penalty`` against ``chi2`` (both on log axes) to see
    the trade-off between fitting the data and regularising the image, and
    compare [`LCurve.corner`][virgil.imaging.LCurve.corner] and
    [`LCurve.discrepancy`][virgil.imaging.LCurve.discrepancy] with
    the images either side: there is usually a wide range of good weights.
    Other ways of choosing the weight are compared in
    ``design/regulariser_weight_selection.md``.

    Parameters
    ----------
    model, priors, data
        As for [`fit`][virgil.fitting.fit].
    regulariser : TSV, TV, MaxEntropy, Laplacian, StarletL1 or LogSum
        The regulariser whose ``weight`` is swept (its own weight is
        ignored).
    weights : sequence of float
        The weights to try.
    others : sequence, optional
        Further regularisers kept fixed, e.g. a
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
            lambda r: r.weight, regulariser, np.asarray(weight, dtype=float)
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


def _rotated_180(image):
    """The Image turned by 180 degrees about the origin of the sky."""
    support = image.support
    return dataclasses.replace(
        image,
        log_brightness=image.eta[::-1, ::-1],
        support=None if support is None else support[::-1, ::-1],
        dra=-image.dra,
        ddec=-image.ddec,
    )


@eqx.filter_jit
def _jitted_chi2(model, observations):
    return np.stack(
        [np.sum(whitened_residuals(model, d) ** 2) for d in observations]
    )


def _chi2(model, observations):
    """χ² of each dataset, jitted so that :func:`diagnose` compiles once."""
    return [float(c) for c in _jitted_chi2(model, tuple(observations))]


def diagnose(model, data, regularisers=()):
    """Check a model and its data for common imaging pitfalls.

    Nothing is printed or warned: print the returned
    [`Diagnosis`][virgil.imaging.Diagnosis] to read it. Run it on the
    fitted model, with the regularisers used in the fit.

    Parameters
    ----------
    model : SourceModel
        The model, normally containing at least one
        [`Image`][virgil.models.Image] (the Image checks are skipped
        otherwise).
    data : OIData or sequence of OIData
        The data.
    regularisers : sequence, optional
        The regularisers of the fit; a
        [`Centroid`][virgil.imaging.Centroid] fixes the position.

    Returns
    -------
    Diagnosis
        ``checks`` holds, per dataset, ``chi2_red`` (χ² per data point) and
        ``phase_regime`` (fraction of model visibilities with |arg V| > 0.8π
        or |V| < 0.05, for projected phases only); per Image,
        ``pixel_scale_mas``, ``edge_flux``
        (fraction of the flux in the outer 2 pixels) and ``centroid_mas``
        (East, North offset in mas); and ``anchored`` (whether something
        fixes the position), ``flip_dchi2`` (Δχ² when every Image is
        rotated by 180° about the origin).
    """
    observations = list(data) if isinstance(data, (list, tuple)) else [data]
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
                "over-fits or the errors are overestimated; regularise more "
                "or check the error bars."
            )

    checks["anchored"] = (
        any(isinstance(part, PointSource) for _, part in parts)
        or any(isinstance(r, Centroid) for r in regularisers)
        or any(
            d.observable_kind == "split"
            and not d.cp_flag
            and d.phi_mat is None
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
            edge = float(1.0 - np.sum(im.brightness[2:-2, 2:-2]))
            checks["edge_flux"].append(edge)
            if edge > 0.05:
                warns.append(
                    f"{label} has {edge:.0%} of its flux in the outer 2 "
                    "pixels; enlarge the field or add a support."
                )
            x, y = Centroid(1.0, path).centroid(model)
            checks["centroid_mas"].append((float(x), float(y)))

        flipped = _map_images(model, _rotated_180)
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
        if d.phi_mat is None and d.observable_kind != "mixed_log_complex":
            continue
        cvis = model.model(d.u, d.v, d.wavel)
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
