r"""Azimuthal harmonics: the image-harmonic convention and conversions.

Every azimuthal profile of a [`MultipoleDisk`][virgil.models.MultipoleDisk] is, at
each radius, $I(\theta)/I_0 = \sum_{m=-M}^{M} I_m e^{im\theta}$ with
$I_0 = 1$ and $I_{-m} = \bar I_m$, where $\theta$ is the azimuth in the plane
of the disk, from the projected major axis towards ``pa + 90``. The linear,
positive, von Mises and legacy amplitude/azimuth forms all map to and from
these *image harmonics* $I_m$ (complex, $m = 0 \dots M$) with
[`to_image_harmonics`][virgil.harmonics.to_image_harmonics] and
[`from_image_harmonics`][virgil.harmonics.from_image_harmonics]: report any
model in one convention and warm-start a positive-form fit from a linear one.
The positive form's reflection coefficients are an internal parametrization:
report the image harmonics. The forms, their formulas and references are in
[azimuthal harmonics](../conventions.md#azimuthal-harmonics).
"""

import warnings

import jax
import jax.numpy as np
import numpy as onp

from ._utils import dtor

__all__ = [
    "cone_coefficients",
    "from_image_harmonics",
    "from_legacy",
    "rim_coefficients",
    "to_image_harmonics",
    "to_legacy",
]

# The series of tanh(x) / x is used below x**2 = _SERIES_X2 (error < 1e-13).
_SERIES_X2 = 1e-4
# |k| is clipped to 1 - _CLIP when a factorization gives roots on the circle.
_CLIP = 1e-6


# --- the positive form (traced, differentiable) -----------------------------


def reflection_coefficients(re, im):
    r"""Complex reflection coefficients from Cartesian latents.

    $k = w \tanh|w| / |w|$ with $w = \mathrm{re} + i\,\mathrm{im}$, so
    $|k| < 1$. Value and gradient are finite at $w = 0$ (the series
    $\tanh x / x \approx 1 - x^2/3$ is used for small $|w|$, behind a safe
    ``where``).

    Parameters
    ----------
    re, im : array-like
        Real and imaginary parts of the latents, shape ``(M, ...)``.

    Returns
    -------
    array-like
        Complex coefficients of the same shape.
    """
    re, im = np.asarray(re), np.asarray(im)
    x2 = re**2 + im**2
    small = x2 < _SERIES_X2
    x = np.sqrt(np.where(small, 1.0, x2))
    series = 1.0 - x2 / 3.0 + 2.0 * x2**2 / 15.0
    ratio = np.where(small, series, np.tanh(x) / x)
    return jax.lax.complex(re * ratio, im * ratio)


def step_up(k):
    """Coefficients ``h_0..h_M`` of the minimum-phase polynomial of the
    reflection coefficients ``k`` (shape ``(M, ...)``), with ``h_0 = 1``.

    Returns a complex array of shape ``(M + 1, ...)``.
    """
    k = np.asarray(k)
    h = [np.ones(k.shape[1:], dtype=k.dtype)]
    for j in range(1, k.shape[0] + 1):
        prev = h + [np.zeros_like(h[0])]
        h = [prev[i] + k[j - 1] * np.conj(prev[j - i]) for i in range(j + 1)]
    return np.stack(h)


def image_harmonics_from_h(h):
    """Image harmonics ``I_0..I_M`` (``I_0 = 1``) of ``|h|^2 / sum |h_j|^2``.

    ``h`` has shape ``(M + 1, ...)``; the result is complex of the same
    shape.
    """
    h = np.asarray(h)
    top = h.shape[0] - 1
    norm = np.sum(np.real(h * np.conj(h)), axis=0)
    # I_0 = 1 exactly (Parseval); the others are the autocorrelation of h.
    return np.stack(
        [jax.lax.complex(np.ones_like(norm), np.zeros_like(norm))]
        + [
            np.sum(h[m:] * np.conj(h[: top + 1 - m]), axis=0) / norm
            for m in range(1, top + 1)
        ]
    )


def linear_from_image(image_harmonics):
    """``(c, s)`` of the linear form from ``I_1..I_M`` (the first row,
    ``I_0``, is dropped): ``c = 2 Re I``, ``s = -2 Im I``."""
    rest = image_harmonics[1:]
    return 2.0 * np.real(rest), -2.0 * np.imag(rest)


def image_from_linear(c, s):
    """Image harmonics ``I_0 = 1, I_m = (c_m - i s_m) / 2`` (``(M + 1, ...)``)."""
    c, s = np.asarray(c), np.asarray(s)
    rest = jax.lax.complex(c, -s) / 2.0
    return np.concatenate([np.ones((1,) + c.shape[1:], rest.dtype), rest])


# --- the von Mises profile ---------------------------------------------------


def von_mises_image_harmonics(kappa, theta0, degree, n_fft):
    r"""Image harmonics of $\exp[\kappa \cos(\theta - \theta_0)]$.

    $I_m = [I_m(\kappa)/I_0(\kappa)]\,e^{-im\theta_0}$ ($I_m(\kappa)$ the
    modified Bessel function), computed by a small FFT of
    $\exp[\kappa(\cos\theta_j - 1)]$ on ``n_fft`` points. The aliasing error is
    $I_{n_{\rm fft} - m}/I_0$, negligible unless $\kappa$ is of the order of
    ``n_fft**2 / 30``.

    Parameters
    ----------
    kappa : array-like
        Concentration, shape ``(...)``.
    theta0 : array-like
        Azimuth of the maximum in radians, broadcast against ``kappa``.
    degree : int
        Highest order kept (static).
    n_fft : int
        Number of FFT points (static), larger than ``2 * degree``.

    Returns
    -------
    image : array-like
        Complex ``I_0 .. I_degree`` (``I_0 = 1``), shape ``(degree + 1, ...)``.
    dropped : array-like
        Power dropped by the truncation,
        $\sum_{m > M} |I_m|^2 / \sum_{m \ne 0} |I_m|^2$, shape ``(...)``.
    """
    kappa = np.asarray(kappa)
    theta0 = np.broadcast_to(np.asarray(theta0), np.shape(kappa))
    grid = 2.0 * np.pi * np.arange(n_fft) / n_fft
    spectrum = np.fft.rfft(np.exp(kappa[..., None] * (np.cos(grid) - 1.0)))
    ratio = spectrum / spectrum[..., :1]
    orders = np.arange(degree + 1).reshape((-1,) + (1,) * np.ndim(kappa))
    kept = np.moveaxis(ratio[..., : degree + 1], -1, 0)
    image = kept * np.exp(-1j * orders * theta0[None])
    power = np.abs(ratio[..., 1:]) ** 2
    total = np.sum(power, axis=-1)
    dropped = np.sum(power[..., degree:], axis=-1) / np.where(
        total > 0.0, total, 1.0
    )
    return image, dropped


def auto_von_mises_degree(kappa, tol=1e-6, cap=256, minimum=1):
    """The smallest degree whose von Mises coefficients beyond it are all
    below ``tol`` of the mean, for a concrete ``kappa`` (the largest if
    several), at most ``cap`` (a warning is given if that truncates; ``None``
    for no cap)."""
    kappa = float(onp.max(onp.asarray(kappa, dtype=float)))
    n = 4096
    while n < 40.0 * onp.sqrt(kappa):
        n *= 2
    grid = 2.0 * onp.pi * onp.arange(n) / n
    spectrum = onp.fft.rfft(onp.exp(kappa * (onp.cos(grid) - 1.0)))
    ratio = onp.abs(spectrum / spectrum[0])
    above = onp.nonzero(ratio > tol)[0]
    degree = int(above.max()) if above.size else 0
    if cap is not None and degree > cap:
        warnings.warn(
            f"a von Mises profile with kappa = {kappa:g} needs degree "
            f"{degree} to keep its harmonics above {tol:g}; truncating at "
            f"{cap}. Give degree= explicitly (it is slow beyond a few "
            "hundred) or accept the truncation (see truncation_power).",
            stacklevel=5,
        )
        degree = cap
    return max(degree, minimum)


# --- spectral factorization and step-down ------------------------------------


def step_down(h, clip=_CLIP):
    """Reflection coefficients of the polynomial ``h`` (``h_0`` is
    normalized to 1): the inverse of :func:`step_up`.

    For a minimum-phase ``h`` every ``|k_j| < 1``. A coefficient with
    ``|k_j| > 1 - clip`` (a root on or inside the unit circle) is clipped to
    ``1 - clip`` and ``clipped`` is true.

    Returns
    -------
    k : array-like
        Complex ``k_1 .. k_M``.
    clipped : bool or jax.Array
    """
    cur = np.asarray(h)
    dtype = np.finfo(np.result_type(cur.real, float)).dtype
    eps = np.finfo(dtype).eps
    ctype = np.result_type(dtype, 1j)
    cur = cur.astype(ctype)
    cur = cur / cur[0]
    top = cur.shape[0] - 1
    ks = []
    clipped = np.asarray(False)
    max_k = 1.0 - clip
    for j in range(top, 0, -1):
        kj = cur[j]
        abs_kj2 = np.real(kj * np.conj(kj))
        nonzero = abs_kj2 > 0.0
        safe_abs = np.sqrt(np.where(nonzero, abs_kj2, 1.0))
        abs_kj = np.where(nonzero, safe_abs, 0.0)
        over = abs_kj > max_k
        kj = np.where(over, kj / safe_abs * max_k, kj)
        clipped = clipped | over
        kj2 = np.real(kj * np.conj(kj))
        denom = np.maximum(1.0 - kj2, eps) if clip > 0.0 else 1.0 - kj2
        lower = (cur[:j] - kj * np.conj(cur[j:0:-1])) / denom
        ks.append(kj)
        cur = lower / lower[0]
    k = np.stack(ks[::-1]) if ks else np.zeros((0,), dtype=ctype)
    if not isinstance(clipped, jax.core.Tracer):
        clipped = bool(clipped)
    return k, clipped


def latents_from_reflection(k):
    """Unconstrained latents ``w = k artanh|k| / |k|`` (``|k| < 1``)."""
    k = onp.asarray(k, dtype=complex)
    mod = onp.abs(k)
    safe = onp.where(mod < 1e-12, 1.0, mod)
    return onp.where(mod < 1e-12, k, k * onp.arctanh(mod) / safe)


def spectral_factor(image_harmonics, negative_tol=1e-6):
    """The minimum-phase ``h`` (``h_0 = 1``) of a positive image.

    Takes the ``2M`` roots of $z^M \\sum_m I_m z^m$, keeps the ``M`` outside
    the unit circle (the largest in modulus), and builds
    $h(z) = \\prod (1 - z / z_k)$.

    Parameters
    ----------
    image_harmonics : array-like
        ``I_0 .. I_M`` (``I_0 > 0``; the rest relative to it).
    negative_tol : float, optional
        The image may dip to ``-negative_tol * I_0`` (rounding) before it is
        refused.

    Raises
    ------
    ValueError
        If the image is negative somewhere, or ``I_0`` is not positive.
    """
    image = onp.asarray(image_harmonics, dtype=complex).ravel()
    if image.size == 0 or not image[0].real > 0.0:
        raise ValueError(f"I_0 must be positive, got {image[:1]}.")
    image = image / image[0].real
    top = image.size - 1
    n_theta = max(4096, 16 * (top + 1))
    theta = 2.0 * onp.pi * onp.arange(n_theta) / n_theta
    orders = onp.arange(1, top + 1)
    profile = 1.0 + 2.0 * onp.real(
        onp.exp(1j * onp.outer(theta, orders)) @ image[1:]
    )
    if profile.min() < -negative_tol:
        raise ValueError(
            "the image harmonics describe a brightness that is negative "
            f"somewhere (minimum {profile.min():.3g} of the mean); only a "
            "non-negative image has a positive form."
        )
    while top > 0 and abs(image[top]) < 1e-12:
        top -= 1
    h = onp.ones(1, dtype=complex)
    if top > 0:
        # P(z) = sum_n a_n z^n with a_n = I_{n - top}, n = 0 .. 2 top.
        a = onp.concatenate([onp.conj(image[top:0:-1]), image[: top + 1]])
        roots = onp.roots(a[::-1])
        outer = roots[onp.argsort(-onp.abs(roots))[:top]]
        for root in outer:
            h = (
                onp.concatenate([h, [0.0]])
                - onp.concatenate([[0.0], h]) / root
            )
    h = onp.concatenate([h, onp.zeros(image.size - h.size)])
    norm = onp.sum(onp.abs(h) ** 2)
    rebuilt = onp.array(
        [
            onp.sum(h[m:] * onp.conj(h[: image.size - m]))
            for m in range(image.size)
        ]
    )
    err = onp.max(onp.abs(rebuilt / norm - image))
    if err > 1e-5:
        warnings.warn(
            f"the spectral factor reproduces the image to only {err:.1e}; "
            "the roots are nearly degenerate (an image that touches zero).",
            stacklevel=3,
        )
    return h


# --- legacy parametrizations -------------------------------------------------


def rim_coefficients(az_amps, az_pas, pa):
    r"""``(c, s)`` of [`ModulatedGaussianRim`][virgil.models.ModulatedGaussianRim]'s
    modulation $\sum_m A_m \cos[m(\phi - \phi_m)]$.

    The rim's azimuth $\phi$ is the engine's azimuth plus ``pa``
    (``az_pas`` are in the plane of the rim, in the sense of position angle,
    with ``pa`` on the major axis), so $c_m = A_m \cos m(\phi_m - \mathrm{pa})$
    and $s_m = A_m \sin m(\phi_m - \mathrm{pa})$. Angles in degrees.

    Returns
    -------
    c, s : array-like
        Each of shape ``(M,)``.
    """
    amps, pas = np.atleast_1d(az_amps), np.atleast_1d(az_pas)
    orders = np.arange(1, amps.size + 1)
    angle = orders * (pas - pa) * dtor
    return amps * np.cos(angle), amps * np.sin(angle)


def cone_coefficients(az_amps, az_pas, pa):
    r"""``(c, s)`` of [`TruncatedCone`][virgil.models.TruncatedCone]'s
    modulation $\sum_m A_m \cos[m(\phi - \phi_m)]$.

    The cone's ring angle is 0 on the wall at ``pa + 90`` and the engine's
    azimuth runs from the major axis, at ``pa - 90``: half a turn apart, so
    $c_m = (-1)^m A_m \cos(m\,\mathrm{ph})$,
    $s_m = -(-1)^m A_m \sin(m\,\mathrm{ph})$ with
    $\mathrm{ph} = \mathrm{pa} + 90^\circ - \phi_m$. Angles in degrees.

    Returns
    -------
    c, s : array-like
        Each of shape ``(M,)``.
    """
    amps, pas = np.atleast_1d(az_amps), np.atleast_1d(az_pas)
    orders = np.arange(1, amps.size + 1)
    sign = (-1.0) ** orders
    angle = orders * (pa + 90.0 - pas) * dtor
    return sign * amps * np.cos(angle), -sign * amps * np.sin(angle)


def _legacy_check(kind):
    if kind not in ("rim", "cone"):
        raise ValueError(f"kind must be 'rim' or 'cone', not {kind!r}.")


def from_legacy(az_amps, az_pas, pa, kind="rim"):
    """The linear [`Harmonics`][virgil.models.Harmonics] of a legacy
    ``az_amps``/``az_pas`` modulation.

    Parameters
    ----------
    az_amps, az_pas : array-like
        Amplitudes and azimuths (degrees) of the modulations, one per order.
    pa : float
        The class's ``pa`` in degrees (the disk's ``pa`` for a rim; the
        cone's ``pa``, not the disk's ``pa - 90``).
    kind : {"rim", "cone"}
        ``ModulatedGaussianRim`` or ``TruncatedCone`` conventions.
    """
    from .models import Harmonics

    _legacy_check(kind)
    convert = rim_coefficients if kind == "rim" else cone_coefficients
    c, s = convert(onp.atleast_1d(az_amps), onp.atleast_1d(az_pas), pa)
    return Harmonics(c=tuple(c), s=tuple(s), form="linear")


def to_legacy(harmonics, pa, kind="rim"):
    """``(az_amps, az_pas)`` of constant harmonics: the inverse of
    :func:`from_legacy`.

    The amplitudes are non-negative and the azimuths in degrees, the first
    of the ``360 / m`` equivalent ones in a turn about ``pa``.
    """
    _legacy_check(kind)
    image = to_image_harmonics(harmonics)
    c, s = (
        onp.asarray(2.0 * image[1:].real),
        onp.asarray(-2.0 * image[1:].imag),
    )
    orders = onp.arange(1, c.size + 1)
    amps = onp.hypot(c, s)
    if kind == "rim":
        pas = pa + onp.degrees(onp.arctan2(s, c)) / orders
    else:
        sign = (-1.0) ** orders
        ph = onp.degrees(onp.arctan2(-sign * s, sign * c)) / orders
        pas = pa + 90.0 - ph
    return amps, pas


# --- the public conversions --------------------------------------------------


def to_image_harmonics(harmonics, r=None):
    """The image harmonics ``I_0 .. I_M`` (``I_0 = 1``) of a model.

    Parameters
    ----------
    harmonics : Harmonics or VonMises
        A linear or positive [`Harmonics`][virgil.models.Harmonics], or a
        [`VonMises`][virgil.models.VonMises] (truncated at its degree; see
        ``VonMises.truncation_power``).
    r : float or array-like, optional
        Radii (mas) at which to evaluate coefficients that are functions of
        the radius; not needed for constants.

    Returns
    -------
    array-like
        Complex, shape ``(M + 1,)`` (or ``(M + 1,) + shape(r)``).
    """
    return harmonics.image_harmonics(r)


def from_image_harmonics(image_harmonics, form="linear"):
    """A constant-coefficient [`Harmonics`][virgil.models.Harmonics] with the
    given image harmonics.

    ``linear`` is exact both ways. ``positive`` is a spectral factorization
    (numpy, not differentiated): the roots of
    $z^M \\sum_m I_m z^m$ outside the unit circle give the minimum-phase
    $h$, then the step-down recursion gives the reflection coefficients and
    the inverse tanh the latents. It needs a non-negative image; roots on the
    circle (an image that touches zero) clip ``|k|`` to ``1 - 1e-6`` with a
    warning. The degree is that of ``image_harmonics`` (a lower true degree
    gives zero outer latents).

    Parameters
    ----------
    image_harmonics : array-like
        ``I_0 .. I_M``, complex; they are normalized to ``I_0 = 1``.
    form : {"linear", "positive"}

    Raises
    ------
    ValueError
        For ``form="positive"`` if the image is negative somewhere.
    """
    from .models import Harmonics

    image = onp.asarray(image_harmonics, dtype=complex)
    if image.ndim != 1:
        raise ValueError(
            "from_image_harmonics takes constant harmonics, a 1-D array "
            f"I_0 .. I_M; got shape {image.shape}."
        )
    if form not in ("linear", "positive"):
        raise ValueError(f"form must be 'linear' or 'positive', not {form!r}.")
    if (
        image.size == 0
        or not image[0].real > 0.0
        or abs(image[0].imag) > 1e-9 * abs(image[0])
    ):
        raise ValueError("I_0 must be real and positive.")
    image = image / image[0].real
    degree = image.size - 1
    if form == "linear":
        return Harmonics(
            c=tuple(float(x) for x in 2.0 * image[1:].real),
            s=tuple(float(x) for x in -2.0 * image[1:].imag),
            degree=degree,
            form="linear",
        )
    h = spectral_factor(image)
    k, clipped = step_down(h)
    if clipped:
        warnings.warn(
            "the image touches zero: some reflection coefficients were "
            f"clipped to |k| = 1 - {_CLIP:g}.",
            stacklevel=2,
        )
    w = latents_from_reflection(k)
    return Harmonics(
        re=tuple(float(x) for x in w.real),
        im=tuple(float(x) for x in w.imag),
        degree=degree,
        form="positive",
    )
