"""Gaussian-process log-brightness fields for pixel images.

A [`GaussianField`][virgil.fields.GaussianField] stands in for the
``log_brightness`` of an [`Image`][virgil.models.Image]: a stationary
Gaussian process with a Matérn-like spectrum, written as independent
standard-normal ``latent`` coefficients on the cosine (DCT-II) basis. Fitting
the latents under a standard-normal prior is MAP estimation with a GP prior;
see [Gaussian processes and IFT](../gp_and_ift.md).

Source: Edenhofer et al. (2024), Arras et al. (2022), Enßlin et al. (2009),
[references](../references.md#edenhofer2024). No code is taken from NIFTy.
"""

import equinox as eqx
import jax
import jax.numpy as np
import numpy as onp
import zodiax as zdx
from jax.scipy.fft import idctn

from ._geometry import (
    pixel_offsets,
    safe_azimuth,
    safe_radius,
    undo_elliptical_transf_coord,
)
from ._utils import concrete, dtor
from .models import _check_nu, _gp_factor, _log_azimuthal_spectrum

__all__ = [
    "GaussianField",
    "MultipoleField",
    "field_spectrum",
    "multipole_field_priors",
    "polar_interpolate",
]

__provisional__ = (
    "MultipoleField",
    "multipole_field_priors",
)


def _check_positive(name, value):
    """Reject a concrete value that is not finite and positive."""
    value = concrete(value)
    if value is not None and not (
        onp.all(onp.isfinite(value)) and onp.all(value > 0.0)
    ):
        raise ValueError(f"{name} must be finite and positive, not {value}.")


def _check_template(mean, shape, shape_name, mean_floor):
    """The validated brightness template ``mean`` (None for none) of an
    image of ``shape`` (called ``shape_name`` in the error), and the
    ``mean_floor`` as a float."""
    if mean is not None:
        mean = np.asarray(mean, dtype=float)
        if mean.shape != shape:
            raise ValueError(
                f"mean has shape {mean.shape}, but {shape_name} {shape}."
            )
        value = concrete(mean)
        if value is not None and not (
            onp.all(onp.isfinite(value)) and value.max() > 0.0
        ):
            raise ValueError(
                "mean must be a finite template with a positive peak."
            )
    _check_positive("mean_floor", mean_floor)
    return mean, float(mean_floor)


def _add_template(eta, mean, mean_floor):
    """``eta`` plus the log of the template ``mean`` (peak 1) and its floor."""
    template = np.maximum(mean, 0.0) / np.max(mean)
    return eta + np.log(template + mean_floor)


def field_spectrum(shape, pixel_scale_mas, sigma, length_mas, order=2):
    """Prior variances of a field's cosine coefficients.

    ``S_jk ∝ (κ² + λ_jk)^(-order)``, with ``κ = 1 / length_mas`` and ``λ_jk``
    the eigenvalues of the reflecting-boundary (Neumann) Laplacian on the pixel
    grid, ``λ_jk = (2/h)² [sin²(πj/2n) + sin²(πk/2m)]``. With two lengths
    ``(ℓ_row, ℓ_col)`` the field is anisotropic, ``S_jk ∝ (1 + ℓ_row² λ_j +
    ℓ_col² λ_k)^(-order)``: correlated over ``ℓ_row`` down a column and over
    ``ℓ_col`` along a row. The constant mode is zero (a softmax image does not
    depend on it) and the spectrum is scaled so the field's pixel-averaged
    variance is ``sigma²``.

    Parameters
    ----------
    shape : tuple of int
        ``(nrow, ncol)``.
    pixel_scale_mas : float
        Pixel size ``h`` in milliarcseconds.
    sigma : float or array-like
        Standard deviation of the field, in units of log-brightness.
    length_mas : float or array-like, shape () or (2,)
        Correlation length ``1/κ`` in milliarcseconds, or the pair
        ``(ℓ_row, ℓ_col)`` for an anisotropic field.
    order : int, optional
        Exponent of the spectrum (default 2). ``order=1`` makes the prior
        exactly a total-squared-variation plus L2 penalty on the field.

    Returns
    -------
    jax.Array
        Variances, of shape ``shape``.
    """
    _check_positive("pixel_scale_mas", pixel_scale_mas)
    _check_positive("length_mas", length_mas)
    eigen = [
        (2.0 / pixel_scale_mas * np.sin(np.pi * np.arange(n) / (2 * n))) ** 2
        for n in shape
    ]
    length_mas = np.asarray(length_mas)
    if length_mas.ndim == 0:
        laplacian = eigen[0][:, None] + eigen[1][None, :]
        spectrum = (length_mas**-2.0 + laplacian) ** (-float(order))
    elif length_mas.shape == (2,):
        stretched = (
            length_mas[0] ** 2 * eigen[0][:, None]
            + length_mas[1] ** 2 * eigen[1][None, :]
        )
        spectrum = (1.0 + stretched) ** (-float(order))
    else:
        raise ValueError(
            "length_mas must be one length or a pair (row, column), not "
            f"shape {length_mas.shape}."
        )
    spectrum = spectrum.at[0, 0].set(0.0)
    return sigma**2 * spectrum * spectrum.size / np.sum(spectrum)


class GaussianField(zdx.Base):  # type: ignore[reportGeneralTypeIssues]
    r"""A Gaussian-process log-brightness for an [`Image`][virgil.models.Image].

    ``η = log(μ/max μ + ε) + IDCT[√S ⊙ z]``: the orthonormal inverse DCT-II of
    the latents ``z`` scaled by the spectrum ``S`` of
    [`field_spectrum`][virgil.fields.field_spectrum], about an optional
    positive template ``μ``. With a standard-normal prior on ``latent`` (from
    [`image_priors`][virgil.imaging.image_priors]) it is a Gaussian process of
    standard deviation ``sigma`` and correlation length ``length_mas``;
    ``latent = 0`` gives the floored template. Fix ``sigma`` and ``length_mas``
    or sample them: their MAP values are biased (``fit`` warns).
    See [Gaussian processes and IFT](../gp_and_ift.md).

    Source: Edenhofer et al. (2024), [references](../references.md#edenhofer2024).

    Parameters
    ----------
    latent : array-like, shape (nrow, ncol)
        Whitened cosine coefficients.
    sigma : float or array-like, optional
        Standard deviation of the field (default 1).
    length_mas : float or array-like, optional
        Correlation length in mas (default 1), or a pair ``(ℓ_row, ℓ_col)``
        along the image's vertical and horizontal axes (with the Image's
        ``rotation_deg`` at a structure's position angle: along and across it).
    order : int, optional
        Exponent of the spectrum (default 2).
    mean, mean_floor : optional
        Positive template image of the same shape (default flat), and its
        floor as a fraction of its peak (default 1e-3).

    Examples
    --------
    ```python
    field = GaussianField(np.zeros((32, 32)), sigma=2.0, length_mas=3.0)
    env = Image(field, pixel_scale_mas=0.6, flux=0.3)
    ```
    """

    latent: jax.Array
    sigma: jax.Array
    length_mas: jax.Array
    mean: jax.Array | None
    order: int = eqx.field(static=True)
    mean_floor: float = eqx.field(static=True)

    def __init__(
        self,
        latent,
        sigma=1.0,
        length_mas=1.0,
        order=2,
        mean=None,
        mean_floor=1e-3,
    ):
        self.latent = np.asarray(latent, dtype=float)
        if self.latent.ndim != 2:
            raise ValueError("latent must be a 2D array.")
        self.sigma = np.asarray(sigma, dtype=float)
        self.length_mas = np.asarray(length_mas, dtype=float)
        value = concrete(self.sigma)
        if value is not None and not (
            onp.all(onp.isfinite(value)) and onp.all(value >= 0.0)
        ):
            raise ValueError(
                f"sigma must be finite and non-negative, not {value}."
            )
        _check_positive("length_mas", self.length_mas)
        if self.length_mas.shape not in ((), (2,)):
            raise ValueError(
                "length_mas must be one length or a pair (row, column), "
                f"not shape {self.length_mas.shape}."
            )
        self.order = int(order)
        if self.order < 1:
            raise ValueError(f"order must be a positive integer, not {order}.")
        self.mean, self.mean_floor = _check_template(
            mean, self.latent.shape, "latent has shape", mean_floor
        )

    @property
    def shape(self):
        """The image's shape, ``(nrow, ncol)``."""
        return self.latent.shape

    def evaluate(self, pixel_scale_mas):
        """The log-brightness ``η`` on pixels of ``pixel_scale_mas``."""
        spectrum = field_spectrum(
            self.shape,
            pixel_scale_mas,
            self.sigma,
            self.length_mas,
            self.order,
        )
        # The constant mode has zero variance; a double where keeps the
        # gradient of its square root finite.
        positive = spectrum > 0.0
        amplitude = np.where(
            positive, np.sqrt(np.where(positive, spectrum, 1.0)), 0.0
        )
        eta = idctn(amplitude * self.latent, type=2, norm="ortho")
        if self.mean is not None:
            eta = _add_template(eta, self.mean, self.mean_floor)
        return eta


# ---------------------------------------------------------------------------
# MultipoleField: a multipole-kernel residual field aligned with a disk
# ---------------------------------------------------------------------------


def _radial_factor(nodes, length, nu):
    """Cholesky factor of the radial correlation on ``nodes``.

    This is [`RadialGP`][virgil.models.RadialGP]'s factor (one definition,
    ``models._gp_factor``), with a jitter of at least ``10 n`` times the
    machine epsilon of the working precision (1.5e-5 for 128 nodes in
    float32, below the default 1e-6 in float64), so that the factorization
    of a Matérn matrix whose length is many node spacings stays finite in
    float32 (a jitter of 1e-6 gives NaN for 128 nodes, ``nu`` = 5/2 and a
    length of 5 spacings or more).
    """
    n = nodes.shape[0]
    return _gp_factor(
        nodes,
        length,
        nu,
        max(1e-6, 10.0 * n * float(np.finfo(nodes.dtype).eps)),
    )


def _azimuthal_filter(latent, angle, nu):
    """Apply the azimuthal square root of the periodic kernel, along axis 1.

    The spectral density at the integer order ``m`` is the one that
    [`MultipoleKernel`][virgil.models.MultipoleKernel] gives the harmonics,
    ``S(m) = [1 + (m angle)^2 / (2 nu)]^(-nu - 1/2)`` (``angle`` is its
    ``ell_theta / r``), the same function. The kernel is circulant on the
    ``n_theta`` azimuths, so the filter is two FFTs, scaled so that the
    mean of ``S`` over the orders is one and the variance of the filtered
    white noise is exactly one.
    """
    n_theta = latent.shape[1]
    m = np.minimum(np.arange(n_theta), n_theta - np.arange(n_theta))
    # In log space, so that no order underflows (a gradient of NaN).
    log_power = _log_azimuthal_spectrum(m, angle, nu)
    amplitude = np.exp(
        0.5
        * (
            log_power
            - jax.scipy.special.logsumexp(log_power)
            + np.log(n_theta)
        )
    )
    return np.real(np.fft.ifft(amplitude * np.fft.fft(latent, axis=1), axis=1))


def polar_interpolate(values, r_min, spacing, r, theta, fill=0.0):
    """Bilinear interpolation of a polar grid at ``(r, theta)``.

    ``values[k, j]`` sits at radius ``r_min + k * spacing`` and azimuth
    ``2π j / n_theta`` (periodic). Outside the radial range the field falls
    linearly to ``fill`` over one ``spacing`` and is ``fill`` beyond, so
    there is no edge discontinuity. The error is second order in the grid
    spacings.

    Parameters
    ----------
    values : array-like, shape (n_r, n_theta)
    r_min, spacing : float
        Radius of the first node and the node spacing, in milliarcseconds.
    r, theta : array-like
        Radii (mas) and azimuths (radians); any common shape.
    fill : float, optional
        The value outside the grid (default 0).

    Returns
    -------
    jax.Array
        The interpolated values, of the shape of ``r``.
    """
    values = np.asarray(values)
    n_r, n_theta = values.shape
    edge = np.full((1, n_theta), fill, dtype=values.dtype)
    padded = np.concatenate([edge, values, edge], axis=0)
    f = np.clip((r - r_min) / spacing + 1.0, 0.0, n_r + 1.0 - 1e-9)
    i0 = np.floor(f)
    wr = f - i0
    i0 = i0.astype(int)
    t = np.mod(theta, 2.0 * np.pi) * (n_theta / (2.0 * np.pi))
    j0 = np.floor(t)
    wt = t - j0
    j0 = j0.astype(int) % n_theta
    j1 = (j0 + 1) % n_theta
    low = padded[i0, j0] * (1.0 - wt) + padded[i0, j1] * wt
    high = padded[i0 + 1, j0] * (1.0 - wt) + padded[i0 + 1, j1] * wt
    return low * (1.0 - wr) + high * wr


class MultipoleField(zdx.Base):  # type: ignore[reportGeneralTypeIssues]
    """A multipole-kernel residual field for an [`Image`][virgil.models.Image].

    !!! warning "Provisional"
        May change in any 0.4.x release without a deprecation cycle.

    The log-brightness is a Gaussian process on a polar grid in the disk's
    frame (``n_r`` radii, ``n_theta`` azimuths) with a separable Matérn prior: radial as
    [`RadialGP`][virgil.models.RadialGP], periodic azimuthal as
    [`MultipoleKernel`][virgil.models.MultipoleKernel]. The brightness is the
    softmax of the field, ``outside`` beyond the grid. The latents are whitened
    standard normals. Priors:
    [`multipole_field_priors`][virgil.fields.multipole_field_priors]; kernel,
    tied geometry and fallback: [Model notes](models/notes.md#multipolefield).

    Parameters
    ----------
    latent : array-like, shape (n_r, n_theta)
        Whitened coefficients; their shape sets the polar grid.
    image_shape : tuple of int
        ``(nrow, ncol)`` of the Image's pixel grid.
    r_max_mas : float
        Outermost radius of the polar grid (mas, disk plane); the field fades
        to zero over one node spacing beyond it.
    r_min_mas, sigma : float, optional
        Innermost radius (default 0); standard deviation of the field (1).
    length_r, length_theta : float or array-like, optional
        Radial (mas; default five node spacings) and azimuthal (radians; 1)
        correlation lengths.
    inc, pa, dra, ddec : float or array-like, optional
        Disk inclination, major-axis position angle (degrees, North to East)
        and centre (mas) relative to the Image's.
    nu, nu_theta : {0.5, 1.5, 2.5, inf}, optional
        Kernel smoothness (default 1.5; ``nu_theta`` defaults to ``nu``).
    geometry_from : str or None, optional
        Sibling [`System`][virgil.models.System] component whose ``inc``,
        ``pa``, ``dra``, ``ddec`` the field follows (default: its own).
    mean, mean_floor : optional
        As for [`GaussianField`][virgil.fields.GaussianField].
    outside : float, optional
        Log-brightness beyond the grid (default -20; 0 leaves a flat
        background the latents cannot remove).

    Examples
    --------
    ```python
    field = MultipoleField(np.zeros((24, 32)), image_shape=(64, 64),
                           r_max_mas=40.0, inc=50.0, pa=30.0)
    env = Image(field, pixel_scale_mas=1.5, flux=0.3)
    ```
    """

    latent: jax.Array
    sigma: jax.Array
    length_r: jax.Array
    length_theta: jax.Array
    inc: jax.Array | None
    pa: jax.Array | None
    dra: jax.Array | None
    ddec: jax.Array | None
    mean: jax.Array | None
    image_shape: tuple = eqx.field(static=True)
    r_min_mas: float = eqx.field(static=True)
    r_max_mas: float = eqx.field(static=True)
    nu: float = eqx.field(static=True)
    nu_theta: float = eqx.field(static=True)
    geometry_from: str | None = eqx.field(static=True)
    fallback: tuple = eqx.field(static=True)
    mean_floor: float = eqx.field(static=True)
    outside: float = eqx.field(static=True)

    def __init__(
        self,
        latent,
        image_shape,
        r_max_mas,
        r_min_mas=0.0,
        sigma=1.0,
        length_r=None,
        length_theta=1.0,
        inc=0.0,
        pa=0.0,
        dra=0.0,
        ddec=0.0,
        nu=1.5,
        mean=None,
        mean_floor=1e-3,
        outside=-20.0,
        nu_theta=None,
        geometry_from=None,
    ):
        self.latent = np.asarray(latent, dtype=float)
        if self.latent.ndim != 2 or min(self.latent.shape) < 2:
            raise ValueError("latent must be a 2D array of at least 2 x 2.")
        self.image_shape = tuple(int(n) for n in image_shape)
        if len(self.image_shape) != 2 or min(self.image_shape) < 1:
            raise ValueError(
                f"image_shape must be (nrow, ncol): {image_shape}"
            )
        self.r_min_mas = float(r_min_mas)
        self.r_max_mas = float(r_max_mas)
        if not (
            onp.isfinite(self.r_max_mas)
            and 0.0 <= self.r_min_mas < self.r_max_mas
        ):
            raise ValueError(
                "need finite 0 <= r_min_mas < r_max_mas, not "
                f"{r_min_mas} and {r_max_mas}."
            )
        self.nu = _check_nu(nu, "MultipoleField")
        self.nu_theta = _check_nu(
            nu if nu_theta is None else nu_theta, "MultipoleField"
        )
        if length_r is None:
            length_r = 5.0 * self.spacing
        self.sigma = np.asarray(sigma, dtype=float)
        self.length_r = np.asarray(length_r, dtype=float)
        self.length_theta = np.asarray(length_theta, dtype=float)
        _check_positive("sigma", self.sigma)
        _check_positive("length_r", self.length_r)
        _check_positive("length_theta", self.length_theta)
        for name in ("sigma", "length_r", "length_theta"):
            if np.shape(getattr(self, name)) != ():
                raise ValueError(f"{name} must be a scalar.")
        geometry = tuple(
            np.asarray(g, dtype=float) for g in (inc, pa, dra, ddec)
        )
        if geometry_from is None:
            self.geometry_from = None
            self.fallback = ()
            self.inc, self.pa, self.dra, self.ddec = geometry
        else:
            # Tied: the geometry is the disk's, so it has no parameters; the
            # values given are only the fallback for an Image evaluated
            # outside a System, and must be concrete.
            if not isinstance(geometry_from, str) or not geometry_from:
                raise ValueError(
                    f"geometry_from must be a component name: {geometry_from!r}"
                )
            values = tuple(concrete(g) for g in geometry)
            if any(v is None or v.shape != () for v in values):
                raise ValueError(
                    "with geometry_from, inc, pa, dra and ddec are only the "
                    "fallback geometry and must be concrete scalars."
                )
            self.geometry_from = geometry_from
            self.fallback = tuple(float(v) for v in values)
            self.inc = self.pa = self.dra = self.ddec = None
        self.mean, self.mean_floor = _check_template(
            mean, self.image_shape, "image_shape is", mean_floor
        )
        self.outside = float(outside)

    @classmethod
    def from_disk(
        cls,
        disk,
        latent,
        image_shape,
        r_max_mas=None,
        geometry_from=None,
        **kwargs,
    ):
        """A field in the frame of ``disk`` (its ``inc``, ``pa``, ``dra``, ``ddec``).

        ``r_max_mas`` defaults to the outer edge of the disk's profile
        (``disk.profile.extent()``) plus 20 per cent. The values are copied
        unless ``geometry_from`` is the disk's name in the System the Image
        will join, which ties them: see the notes of
        [`MultipoleField`][virgil.fields.MultipoleField].
        """
        if r_max_mas is None:
            r_max_mas = 1.2 * float(disk.profile.extent()[1])
        return cls(
            latent,
            image_shape,
            r_max_mas,
            inc=disk.inc,
            pa=disk.pa,
            dra=disk.dra,
            ddec=disk.ddec,
            geometry_from=geometry_from,
            **kwargs,
        )

    def with_geometry(self, inc, pa, dra, ddec):
        """This field untied, with the given geometry as its own parameters.

        What [`System`][virgil.models.System] calls to resolve a tie; the
        arguments are the grid-frame ``inc``, ``pa`` (degrees) and centre
        (milliarcseconds).
        """
        return type(self)(
            self.latent,
            self.image_shape,
            self.r_max_mas,
            r_min_mas=self.r_min_mas,
            sigma=self.sigma,
            length_r=self.length_r,
            length_theta=self.length_theta,
            inc=inc,
            pa=pa,
            dra=dra,
            ddec=ddec,
            nu=self.nu,
            mean=self.mean,
            mean_floor=self.mean_floor,
            outside=self.outside,
            nu_theta=self.nu_theta,
        )

    def _geometry(self):
        if self.inc is not None:
            return self.inc, self.pa, self.dra, self.ddec
        return tuple(np.asarray(g, dtype=float) for g in self.fallback)

    @property
    def shape(self):
        """The shape of the latents, ``(n_r, n_theta)``."""
        return self.latent.shape

    @property
    def spacing(self):
        """Radial node spacing in milliarcseconds."""
        return (self.r_max_mas - self.r_min_mas) / (self.latent.shape[0] - 1)

    def polar_values(self):
        """The field ``η`` on its polar grid, shape ``(n_r, n_theta)``."""
        n_r = self.latent.shape[0]
        nodes = self.r_min_mas + self.spacing * np.arange(n_r, dtype=float)
        factor = _radial_factor(nodes, self.length_r, self.nu)
        azimuthal = _azimuthal_filter(
            self.latent, self.length_theta, self.nu_theta
        )
        # HIGHEST: the default matmul precision on A100/H100 is TF32.
        field = self.sigma * np.matmul(
            factor, azimuthal, precision=jax.lax.Precision.HIGHEST
        )
        if self.r_min_mas == 0:
            # A smooth single-valued field has a_m(r) ~ r^|m| at the origin
            # (Born & Wolf 1999, sec. 9.2). The power law holds out to the
            # scale on which the field varies, its radial correlation length
            # ell (a physical length: the node spacing is only the grid's, so
            # the taper would change with n_r). Scale order m by
            # tanh(r/ell)^|m|, which is 1 - O(e^(-2r/ell)), so the prior stays
            # stationary away from the centre and only m = 0 survives at r = 0.
            n_theta = field.shape[1]
            j = np.arange(n_theta)
            m = np.minimum(j, n_theta - j)
            taper = np.tanh(nodes / self.length_r)[:, None] ** m[None, :]
            field = np.real(
                np.fft.ifft(taper * np.fft.fft(field, axis=1), axis=1)
            )
        return field

    def evaluate(self, pixel_scale_mas):
        """The log-brightness ``η`` on pixels of ``pixel_scale_mas``."""
        nrow, ncol = self.image_shape
        inc, pa, dra, ddec = self._geometry()
        x = pixel_offsets(ncol, pixel_scale_mas)[None, :] - dra
        y = pixel_offsets(nrow, pixel_scale_mas)[:, None] - ddec
        x, y = np.broadcast_arrays(x, y)
        stretch = np.maximum(np.cos(inc * dtor), 1e-8)
        xe, ye = undo_elliptical_transf_coord(x, y, pa, stretch)
        r = safe_radius(xe, ye)
        # The azimuth runs from the deprojected North towards East, as for
        # MultipoleDisk's harmonics.
        theta = safe_azimuth(xe, ye)
        eta = polar_interpolate(
            self.polar_values(),
            self.r_min_mas,
            self.spacing,
            r,
            theta,
            fill=self.outside,
        )
        if self.mean is not None:
            eta = _add_template(eta, self.mean, self.mean_floor)
        return eta


def multipole_field_priors(
    path="env.log_brightness",
    sigma=(0.05, 5.0),
    length_r=(0.5, 50.0),
    length_theta=(0.1, 6.0),
):
    """Bounded log-uniform priors on a [`MultipoleField`][virgil.fields.MultipoleField]'s scales.

    !!! warning "Provisional"
        May change in any 0.4.x release without a deprecation cycle.

    Returns a priors dict for [`fit`][virgil.fitting.fit] or
    [`numpyro_model`][virgil.likelihood.numpyro_model]: ``LogUniform`` priors,
    which are invariant under rescaling and proper because they are
    bounded. Add the latents' priors with
    [`image_priors`][virgil.imaging.image_priors]. Each range is
    ``(low, high)``: ``sigma`` in log-brightness, ``length_r`` in
    milliarcseconds and ``length_theta`` in radians. The variance of the
    azimuthal order ``m`` falls as the spectrum of
    [`MultipoleKernel`][virgil.models.MultipoleKernel] in ``m * length_theta``, so a
    length above about 2 leaves almost only the order 0, an axisymmetric
    field (for ``nu`` = 3/2 the first order, cos θ and sin θ together, has about
    1 per cent of the variance at a length of 6), and the default upper limit is effectively
    axisymmetric.

    Parameters
    ----------
    path : str, optional
        Path of the field in the model (default ``"env.log_brightness"``).
    """
    import numpyro.distributions as dist

    priors = {}
    for name, (low, high) in (
        ("sigma", sigma),
        ("length_r", length_r),
        ("length_theta", length_theta),
    ):
        if not 0.0 < low < high:
            raise ValueError(f"{name} range must satisfy 0 < low < high.")
        priors[f"{path}.{name}"] = dist.LogUniform(low, high)
    return priors
