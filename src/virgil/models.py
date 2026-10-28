"""Source models: sky-brightness distributions and their visibilities.

* Components ([`PointSource`][virgil.models.PointSource],
  [`GaussianDisk`][virgil.models.GaussianDisk],
  [`UniformDisk`][virgil.models.UniformDisk], the limb-darkened disks such
  as [`QuadraticLimbDarkenedDisk`][virgil.models.QuadraticLimbDarkenedDisk],
  [`ModulatedGaussianRim`][virgil.models.ModulatedGaussianRim],
  [`MultipoleDisk`][virgil.models.MultipoleDisk] (a radial profile, with its
  profiles such as [`GaussianRing`][virgil.models.GaussianRing]), and
  the flared scattered-light disks such as
  [`FlaredDiskPowerLaw`][virgil.models.FlaredDiskPowerLaw]) are
  single shapes, combined with flux weights in a
  [`System`][virgil.models.System].
* [`BinaryModelCartesian`][virgil.models.BinaryModelCartesian] and
  [`BinaryModelAngular`][virgil.models.BinaryModelAngular] are fast
  forms of a primary plus a point-source companion.
* The ``cvis_*`` functions are the analytic visibilities behind them.

Every flux is relative: for a companion it is the companion/primary flux
ratio. Likelihoods of these models are in
[`virgil.likelihood`][virgil.likelihood].
"""

import dataclasses
import math
import textwrap
from typing import Any

import equinox as eqx
import jax
import jax.numpy as np
import numpy as onp
import zodiax as zdx
from jax.scipy.ndimage import map_coordinates
from jax.scipy.signal import fftconvolve
from jaxbessel import bessel_jn, bessel_jv_over_xv

from ._geometry import (
    apply_elliptical_transf_coord,
    check_az_prof_nonnegative,
    gaussian_smooth,
    gaussian_taper,
    grid_visibilities,
    image_coordinates,
    image_visibilities,
    offset_phase,
    pixel_offsets,
    separation_pa,
    rotate,
    safe_azimuth,
    safe_radius,
    undo_elliptical_transf_coord,
    undo_elliptical_transf_spat_freq,
)
from . import _elr
from ._deprecate import renamed
from .harmonics import cone_coefficients, rim_coefficients
from ._utils import (
    FWHM_PER_SIGMA,
    _named_part,
    check_part_name,
    concrete,
    dtor,
    mas2rad,
)
from .orbits import _days_since, _warn_if_mjd_without_t_ref
from .spectra import Spectrum, _planck_ratio, flux_at

__provisional__ = (
    "MultipoleDisk",
    "ThinRing",
    "GaussianRing",
    "RadialPowerLaw",
    "ExpTail",
    "Gap",
    "RadialSum",
    "RadialLinear",
    "RadialPower",
    "RadialGP",
    "RadialGPFunction",
    "MultipoleKernel",
    "Harmonics",
    "VonMises",
    "Cone",
    "PowerLawHeight",
    "multipole_disk_cvis",
    "gp_priors",
)


def _normalize_image(image):
    """Return a finite unit-sum image for render outputs.

    Empty images are rejected when the flux is known; under ``jax.jit``
    (e.g. rendering many posterior samples) the check is skipped.
    """
    image = np.nan_to_num(np.asarray(image), nan=0.0, posinf=0.0, neginf=0.0)
    total = np.sum(image)
    value = concrete(total)
    if value is not None and not (onp.isfinite(value) and value > 0.0):
        raise ValueError("Rendered image must contain positive finite flux.")
    return image / total


def _format_leaf(x):
    """Short human-readable form of a parameter value for ``repr``."""
    value = concrete(x)
    if value is None or value.dtype.kind not in "fiu":
        return repr(x)
    if value.ndim == 0:
        return f"{float(value):g}"
    return "[" + ", ".join(f"{float(v):g}" for v in value.ravel()) + "]"


def _as_flux(flux):
    """A component flux: a spectrum as given, or a number as a float array."""
    if isinstance(flux, Spectrum):
        return flux
    return np.asarray(flux, dtype=float)


def _flux_is_non_negative(flux):
    """Traced check that a number or spectrum is a valid (non-negative) flux."""
    if isinstance(flux, Spectrum):
        return flux.is_physical()
    return np.all(np.asarray(flux) >= 0.0)


def _check_non_negative_flux(flux, owner):
    """Raise if a concrete ``flux`` is negative; traced values are not checked.

    A spectrum is checked at its characteristic wavelengths (``wavel0``,
    every node and line centre), where its total must be non-negative.
    """
    if isinstance(flux, Spectrum):
        flux = flux(flux._check_wavel())
    value = concrete(flux)
    if value is not None and onp.any(value < 0.0):
        raise ValueError(
            f"{owner} has flux {value.tolist()}; fluxes must be non-negative."
        )


def _check_shape_params(owner, checks):
    """Reject concrete shape parameters outside their valid ranges.

    ``checks`` holds ``(name, value, valid, requirement)`` with ``valid`` a
    function of the concrete array. NaN and infinite values fail every
    check, since ``valid`` uses comparisons. Traced values are skipped; the
    model's ``is_physical`` covers them.
    """
    for name, value, valid, requirement in checks:
        value = concrete(value)
        if value is not None and not (
            onp.all(onp.isfinite(value)) and onp.all(valid(value))
        ):
            raise ValueError(
                f"{owner} has {name} {value.tolist()}; it must be {requirement}."
            )


def _check_non_negative_modulation(az_amps, az_pas):
    """Raise if concrete azimuthal modulations make the brightness negative."""
    amps, pas = concrete(az_amps), concrete(az_pas)
    if amps is None or pas is None:
        return
    if not bool(check_az_prof_nonnegative(np.asarray(amps), np.asarray(pas))):
        raise ValueError(
            f"Azimuthal modulations az_amps={amps.tolist()}, "
            f"az_pas={pas.tolist()} make the rim brightness negative "
            "somewhere; sum(abs(az_amps)) <= 1 always keeps it non-negative."
        )


def _positive(x):
    return x > 0.0


def _in_unit_interval(x):
    return (x > 0.0) & (x <= 1.0)


class SourceModel(zdx.Base):  # type: ignore[reportGeneralTypeIssues]
    """Base class for sky-brightness source models.

    There are two kinds of source model.

    * **Components** ([`Component`][virgil.models.Component] subclasses such as
      [`PointSource`][virgil.models.PointSource]) are single shapes normalized to unit flux. Their
      ``flux`` is a *relative weight*, which only matters once they are mixed
      together in a [`System`][virgil.models.System].
    * **Scenes** ([`System`][virgil.models.System], [`BinaryModelCartesian`][virgil.models.BinaryModelCartesian],
      [`BinaryModelAngular`][virgil.models.BinaryModelAngular], [`HarmonixModel`][virgil.models.HarmonixModel]) describe a whole,
      normalized sky. A scene placed inside a [`System`][virgil.models.System] has weight 1,
      unless it carries its own ``flux`` weight as [`System`][virgil.models.System] does.

    The binary models' ``flux`` is their companion/primary flux ratio, which
    is the same thing as a companion's weight in a System whose primary
    has ``flux=1``.

    Subclasses implement [`model`][virgil.models.SourceModel.model], and ``_image`` if they can be drawn.
    """

    def model(self, u, v, wavel):
        """Evaluate complex visibilities on interferometric baselines.

        Parameters
        ----------
        u, v : array-like
            Baseline coordinates in metres.
        wavel : array-like
            Wavelength(s) in metres.

        Returns
        -------
        array-like
            Complex visibilities, normalized to 1 at zero baseline.
        """
        raise NotImplementedError

    def model_on_grid(self, u, v, wavel, grid):
        """Visibilities at samples ``u, v`` that also lie on a uv ``grid``.

        [`OIData.model`][virgil.oidata.OIData.model] calls this when its
        samples form a regular lattice (a
        [`UVGrid`][virgil.oidata.UVGrid]), so that models able to use
        the lattice, such as a matching
        [`Image`][virgil.models.Image], can. By default it is
        [`model`][virgil.models.SourceModel.model].
        """
        return self.model(u, v, wavel)

    def render(self, npix=256, fov_mas=200.0):
        """Render a unit-sum image of the model.

        The image is ``npix`` x ``npix`` pixels spanning ``fov_mas``
        milliarcseconds, with East to the left (column 0 is the most
        positive ``dra``) and North up (row 0 is the most positive ``ddec``).
        Use [`virgil.plotting.plot_model`][virgil.plotting.plot_model] to display it with the
        correct axes.
        """
        xx, yy = image_coordinates(npix, fov_mas)
        return _normalize_image(
            self._image(xx, yy, float(fov_mas) / float(npix))
        )

    def resolved(self):
        """This model with its tied parameters replaced by their values.

        A [`MultipoleField`][virgil.fields.MultipoleField] built with
        ``geometry_from="name"`` takes the geometry of the sibling component
        ``name`` of its [`System`][virgil.models.System] when the System is
        evaluated. ``resolved`` returns the System with those ties baked in
        (untied fields holding the disk's current values), for code that reads an
        Image's pixels directly. Models without ties return themselves.
        """
        return self

    def _weight(self, wavel=None):
        """Relative flux of this model inside a [`System`][virgil.models.System].

        ``wavel`` is the wavelength in metres at which the flux is wanted
        (broadcastable against the baselines), or ``None`` for the model's
        reference flux, as used when rendering. Components and systems whose
        ``flux`` is a spectrum (see [`virgil.spectra`][virgil.spectra])
        evaluate it here.
        """
        return 1.0

    def total_spectrum(self, wavel):
        """The model's total flux at ``wavel`` (metres), e.g. for OI_FLUX.

        The sum of the component spectra, in the same relative units as
        their fluxes: one grey scale (absolute calibration, injection) away
        from a measured spectrum. It is the *intrinsic* total, with no
        fibre coupling. For a [`System`][virgil.models.System] it is the
        sum of its parts' fluxes, where a nested system counts with its own
        ``flux``, as in the visibilities.

        Examples
        --------
        >>> from virgil.spectra import PowerLaw
        >>> scene = System(
        ...     star=PointSource(),
        ...     disk=GaussianDisk(5.0, flux=PowerLaw(0.5, index=1.0, wavel0=2.0e-6)),
        ... )
        >>> [round(float(f), 3) for f in scene.total_spectrum(np.array([2.0e-6, 4.0e-6]))]
        [1.5, 2.0]
        """
        wavel = np.asarray(wavel)
        return np.broadcast_to(self._weight(wavel), wavel.shape)

    @property
    def time_dependent(self):
        """Whether the model changes with time (it contains an
        [`Attached`][virgil.models.Attached] component). Such a model is
        evaluated with [`at`][virgil.models.SourceModel.at], which
        [`OIData.model`][virgil.oidata.OIData.model] does at each sample's
        own time (all at once when ``broadcast_in_time`` is true)."""
        return False

    @property
    def broadcast_in_time(self):
        """Whether ``at`` accepts an array of times and returns a model
        whose ``model`` evaluates all of them at once, one per sample
        (which [`OIData.model`][virgil.oidata.OIData.model] then does
        without a loop over samples). False by default, so a time-dependent
        model whose ``at`` takes only a scalar time keeps the per-sample
        loop; a model that sets it True must accept an array ``mjd`` and
        return a snapshot whose positions are arrays of that shape. False
        also for an [`Attached`][virgil.models.Attached] component with
        bound angles, whose per-sample angles the components cannot
        take."""
        return False

    def at(self, mjd, t_ref=0.0):
        """This model at time ``t_ref + mjd`` (days, MJD).

        The time is split so that it keeps its precision: ``t_ref`` is a
        float64 number and ``mjd`` may be small offsets from it, even in
        float32 (as [`OIData`][virgil.oidata.OIData] passes them). For a
        single time, ``model.at(60500.3)`` is enough. A model with
        ``broadcast_in_time`` also takes an array of times, one per sample.
        A model that does not change with time returns itself.
        """
        return self

    def is_physical(self):
        """Whether the model is physically valid, as a (traceable) boolean.

        Unlike the checks made when a model is built, this works inside
        ``jax.jit`` and on models changed with ``set``, so likelihoods can
        reject invalid models (see ``reject_unphysical`` in
        [`model_loglike`][virgil.likelihood.model_loglike]). The base
        class has no constraints.
        """
        return np.asarray(True)

    def _image(self, xx, yy, pixel_scale_mas):
        """Un-normalized image on the given coordinate grid."""
        raise NotImplementedError(f"{type(self).__name__} cannot be rendered.")

    def _place(self, flux, dra, ddec):
        """Set ``flux``, ``dra`` and ``ddec`` from constructor arguments."""
        self.flux = _as_flux(flux)
        self.dra = np.asarray(dra, dtype=float)
        self.ddec = np.asarray(ddec, dtype=float)

    def __repr__(self):
        fields = ", ".join(
            f"{field.name}={_format_leaf(getattr(self, field.name))}"
            for field in dataclasses.fields(self)
        )
        return f"{type(self).__name__}({fields})"


class Component(SourceModel):
    """Base class for single shapes with ``flux``, ``dra`` and ``ddec``.

    A component on its own is normalized to unit flux. New shapes subclass it and
    implement ``_centred_cvis`` (the unit-flux visibility of the shape at the
    origin) and ``_centred_image`` (an un-normalized image at the origin); offsets
    and mixing are handled here.

    Parameters
    ----------
    flux : float, array-like or Spectrum, optional
        Weight relative to the other components of a
        [`System`][virgil.models.System], or a spectrum from
        [`virgil.spectra`][virgil.spectra] (default 1). With the reference star
        at ``flux=1`` a companion's ``flux`` is its companion/star flux ratio.
    dra, ddec : float or array-like, optional
        Offset of the centre in milliarcseconds, positive ``dra`` to the East and
        ``ddec`` to the North.
    """

    flux: jax.Array
    dra: jax.Array
    ddec: jax.Array

    def _centred_cvis(self, uu, vv):
        """Visibility of the shape centred on the origin, normalized to 1."""
        raise NotImplementedError

    def _centred_image(self, xx, yy, pixel_scale_mas):
        """Un-normalized image of the shape centred on the origin."""
        raise NotImplementedError

    def model(self, u, v, wavel):
        uu, vv = u / wavel, v / wavel
        return self._centred_cvis(uu, vv) * offset_phase(
            uu, vv, self.dra, self.ddec
        )

    def _image(self, xx, yy, pixel_scale_mas):
        return self._centred_image(
            xx - self.dra, yy - self.ddec, pixel_scale_mas
        )

    def _weight(self, wavel=None):
        return flux_at(self.flux, wavel)

    def _shape_checks(self):
        """``(name, value, valid, requirement)`` for each shape parameter.

        Checked on concrete values when the model is built
        (``__check_init__``) and on traced ones in ``is_physical``.
        """
        return ()

    def is_physical(self):
        valid = _flux_is_non_negative(self.flux)
        for _, value, ok, _ in self._shape_checks():
            valid = valid & np.all(ok(value))
        return valid

    def __check_init__(self):
        _check_non_negative_flux(self.flux, type(self).__name__)
        _check_shape_params(type(self).__name__, self._shape_checks())


class PointSource(Component):
    """Unresolved point source.

    Parameters
    ----------
    flux, dra, ddec : optional
        Relative weight (or spectrum) and sky offset in milliarcseconds,
        as for any [`Component`][virgil.models.Component].

    Examples
    --------
    >>> star = PointSource()
    >>> companion = PointSource(flux=0.01, dra=45.0, ddec=30.0)
    """

    def __init__(self, flux=1.0, dra=0.0, ddec=0.0):
        self._place(flux, dra, ddec)

    def _centred_cvis(self, uu, vv):
        return np.ones_like(uu) + 0j

    def _centred_image(self, xx, yy, pixel_scale_mas):
        sigma = max(pixel_scale_mas, 1e-6)
        return np.exp(-0.5 * (xx**2 + yy**2) / sigma**2)


class GaussianDisk(Component):
    """Circular Gaussian brightness distribution.

    Parameters
    ----------
    sigma : float or array-like
        Standard deviation of the Gaussian in milliarcseconds
        (FWHM = 2.3548 ``sigma``, ``FWHM_PER_SIGMA``).
    flux, dra, ddec : optional
        Relative weight (or spectrum) and sky offset in milliarcseconds,
        as for any [`Component`][virgil.models.Component].

    Examples
    --------
    >>> halo = GaussianDisk(sigma=8.0, flux=0.2)
    """

    sigma: jax.Array

    def __init__(self, sigma, flux=1.0, dra=0.0, ddec=0.0):
        self.sigma = np.asarray(sigma, dtype=float)
        self._place(flux, dra, ddec)

    def _centred_cvis(self, uu, vv):
        sigma_rad = mas2rad * self.sigma
        return np.exp(-2.0 * np.pi**2 * sigma_rad**2 * (uu**2 + vv**2)) + 0j

    def _centred_image(self, xx, yy, pixel_scale_mas):
        sigma = np.maximum(self.sigma, 1e-9)
        return np.exp(-0.5 * (xx**2 + yy**2) / sigma**2)


class EllipticalGaussian(Component):
    """Elliptical Gaussian brightness distribution.

    Parameters
    ----------
    fwhm : float or array-like
        Full width at half maximum along the major axis, in
        milliarcseconds.
    ratio : float or array-like
        Minor-to-major axis ratio, in (0, 1]; 1 is a circular Gaussian.
    pa : float or array-like
        Position angle of the major axis in degrees, North to East.
    flux, dra, ddec : optional
        Relative weight (or spectrum) and sky offset in milliarcseconds,
        as for any [`Component`][virgil.models.Component].

    Examples
    --------
    >>> plume = EllipticalGaussian(fwhm=30.0, ratio=0.5, pa=20.0, flux=2.0)
    """

    fwhm: jax.Array
    ratio: jax.Array
    pa: jax.Array

    def __init__(self, fwhm, ratio=1.0, pa=0.0, flux=1.0, dra=0.0, ddec=0.0):
        self.fwhm = np.asarray(fwhm, dtype=float)
        self.ratio = np.asarray(ratio, dtype=float)
        self.pa = np.asarray(pa, dtype=float)
        self._place(flux, dra, ddec)

    def _shape_checks(self):
        return (
            ("fwhm", self.fwhm, _positive, "positive"),
            ("ratio", self.ratio, _in_unit_interval, "in (0, 1]"),
        )

    def _centred_cvis(self, uu, vv):
        # In the frame where the ellipse is a circle of the major axis's
        # width, the visibility is that of a circular Gaussian.
        ut, vt = undo_elliptical_transf_spat_freq(uu, vv, self.pa, self.ratio)
        return _cvis_gaussian_envelope(ut, vt, self.fwhm)

    def _centred_image(self, xx, yy, pixel_scale_mas):
        ratio = np.maximum(self.ratio, 1e-9)
        xt, yt = undo_elliptical_transf_coord(xx, yy, self.pa, ratio)
        fwhm = np.maximum(self.fwhm, 1e-9)
        return np.exp(-4.0 * np.log(2.0) * (xt**2 + yt**2) / fwhm**2)


_ARC_MAX_DEGREE = 256  # beyond this GaussianArc integrates along the arc


class GaussianArc(Component):
    """A ridge bent along a circular arc, e.g. a curved shock front.

    A circle of ``radius`` about the centre of curvature ``(dra, ddec)``,
    blurred by a Gaussian of FWHM ``width`` and weighted along it by a von
    Mises function of position angle, brightest at ``pa``; ``length`` is the
    FWHM of that weight as an arc length, for an arc short against its radius.
    A [`MultipoleDisk`][virgil.models.MultipoleDisk] with a
    [`ThinRing`][virgil.models.ThinRing] profile: [Model
    notes](../notes.md#gaussianarc).

    Parameters
    ----------
    radius, width, length, pa : float or array-like
        Radius of curvature, FWHM across the arc, FWHM along it (mas), and
        position angle of the brightest point (degrees North to East).
    flux, dra, ddec : optional
        As for any [`Component`][virgil.models.Component]; the offset is the
        centre of curvature.
    nodes, degree : int, optional
        Points sampling the arc, at least 2 (default 128; used by the image
        always, by the visibilities on the quadrature path only), and the
        highest von Mises harmonic (static; default from the starting
        ``radius`` and ``length``, or 32 if traced).

    Examples
    --------
    >>> shock = GaussianArc(radius=30.0, width=2.0, length=40.0, pa=270.0)
    """

    radius: jax.Array
    width: jax.Array
    length: jax.Array
    pa: jax.Array
    nodes: int = eqx.field(static=True)
    degree: int = eqx.field(static=True)
    quadrature: bool = eqx.field(static=True)

    def __init__(
        self,
        radius,
        width,
        length,
        pa=0.0,
        flux=1.0,
        dra=0.0,
        ddec=0.0,
        nodes=128,
        degree=None,
    ):
        self.radius = np.asarray(radius, dtype=float)
        self.width = np.asarray(width, dtype=float)
        self.length = np.asarray(length, dtype=float)
        self.pa = np.asarray(pa, dtype=float)
        self._place(flux, dra, ddec)
        if isinstance(nodes, bool) or int(nodes) != nodes or nodes < 2:
            raise ValueError(f"nodes must be an integer >= 2, got {nodes}.")
        self.nodes = int(nodes)
        self.quadrature = False
        if degree is None:
            r, length = concrete(self.radius), concrete(self.length)
            if (
                r is None
                or length is None
                or not onp.all(onp.isfinite(r) & (r > 0.0))
                or not onp.all(onp.isfinite(length) & (length > 0.0))
            ):
                degree = 32  # traced, or invalid (rejected below)
            else:
                kappa = (FWHM_PER_SIGMA * onp.max(r / length)) ** 2
                from .harmonics import auto_von_mises_degree

                degree = auto_von_mises_degree(kappa, cap=None)
                if degree > _ARC_MAX_DEGREE:
                    # Too narrow for a harmonic series: integrate along the
                    # arc instead.
                    self.quadrature = True
                    degree = _ARC_MAX_DEGREE
        if isinstance(degree, bool) or int(degree) != degree or degree < 1:
            raise ValueError(f"degree must be a positive integer: {degree}.")
        self.degree = int(degree)

    def _shape_checks(self):
        return (
            ("radius", self.radius, _positive, "positive"),
            ("width", self.width, _positive, "positive"),
            ("length", self.length, _positive, "positive"),
        )

    def is_physical(self):
        """The base checks, and that the truncated von Mises series is adequate.

        False if the series, truncated at ``degree``, drops more than 1e-4 of
        the power (e.g. ``length`` fitted far below its starting value); the
        series check is skipped on the quadrature path.
        """
        if self.quadrature:
            return super().is_physical()
        return super().is_physical() & np.all(
            self.harmonics().truncation_power() < 1e-4
        )

    def kappa(self):
        """The concentration of the von Mises weight along the arc."""
        return (FWHM_PER_SIGMA * self.radius / self.length) ** 2

    def harmonics(self):
        """The azimuthal weight as a [`VonMises`][virgil.models.VonMises]
        (azimuth from North towards East, brightest at ``pa``)."""
        return VonMises(self.kappa(), self.pa, degree=self.degree)

    def curve(self):
        """Points along the arc (mas, East and North of the centre) and
        their normalized trapezoidal-rule weights, as the image samples it.

        The points are equally spaced in position angle over the span in
        which the weight is above ``exp(-21)`` of its peak (once round the
        circle for a wide arc, both ends at the antipode of ``pa``)."""
        kappa = self.kappa()
        # A safe argument where the span is the whole circle, so that the
        # gradient of arccos stays finite.
        cosine = 1.0 - 21.0 / np.maximum(kappa, 1e-12)
        wide = cosine <= -1.0
        span = np.where(wide, np.pi, np.arccos(np.where(wide, 0.0, cosine)))
        phi = np.linspace(-1.0, 1.0, self.nodes) * span
        angle = self.pa * dtor + phi
        # Trapezoidal rule: the end points carry half weight.
        ends = np.ones(self.nodes).at[np.array([0, -1])].set(0.5)
        weight = ends * np.exp(kappa * (np.cos(phi) - 1.0))
        x = self.radius * np.sin(angle)
        y = self.radius * np.cos(angle)
        return x, y, weight / np.sum(weight)

    def _centred_cvis(self, uu, vv):
        if self.quadrature:
            x, y, weight = self.curve()
            shape = np.shape(uu)
            uu, vv = np.ravel(uu)[:, None], np.ravel(vv)[:, None]
            curve = np.matmul(
                offset_phase(uu, vv, x[None, :], y[None, :]),
                weight,
                precision=jax.lax.Precision.HIGHEST,
            )
            envelope = _cvis_gaussian_envelope(uu[:, 0], vv[:, 0], self.width)
            return np.reshape(curve * envelope, shape)
        # A thin ring with the von Mises weight and a Gaussian blur, on the
        # engine, face-on: its azimuth runs from North towards East.
        c, s = self.harmonics().coefficients(np.reshape(self.radius, (1,)))
        return multipole_disk_cvis(
            uu,
            vv,
            np.reshape(self.radius, (1,)),
            np.ones((1,)),
            0.0,
            0.0,
            self.width,
            c=c,
            s=s,
        )

    def _centred_image(self, xx, yy, pixel_scale_mas):
        x, y, weight = self.curve()
        width = np.maximum(self.width, pixel_scale_mas)
        d2 = (xx[..., None] - x) ** 2 + (yy[..., None] - y) ** 2
        return np.sum(weight * np.exp(-4.0 * np.log(2.0) * d2 / width**2), -1)


class TruncatedCone(Component):
    """A thin conical shell truncated near its apex, e.g. a dust cone.

    The axis points towards position angle ``pa``, tilted ``tilt`` degrees out
    of the sky plane, with half-opening angle ``alpha``. Emission starts a
    slant distance ``s0`` from the apex and falls as ``exp(-(s - s0) /
    length)`` across a Gaussian shell of FWHM ``width``. A
    [`MultipoleDisk`][virgil.models.MultipoleDisk] of rings: [Model
    notes](../notes.md#truncatedcone).

    Parameters
    ----------
    tip, s0, length, width : float or array-like
        Apex distance back along the axis from ``(dra, ddec)``; slant start of
        emission; e-folding length along the walls; shell FWHM; all in mas.
    alpha : float or array-like
        Half-opening angle in degrees (0 < alpha < 90).
    tilt, pa : float or array-like, optional
        Axis angle out of the sky plane (-90 to 90) and position angle the cone
        opens towards (North to East), both in degrees (default 0).
    ratio : float or array-like, optional
        Axis ratio of the cross-section (default 1, circular).
    flux, dra, ddec : optional
        As for any [`Component`][virgil.models.Component].
    n_rings : int, optional
        Gauss-Legendre rings along the walls, at least 2 (default 48); check
        convergence by doubling.
    az_amps, az_pas : float or array-like, optional
        Cosine modulation amplitudes of the walls, first order up
        (``sum(abs(az_amps)) <= 1`` keeps the brightness non-negative; default
        none), and their azimuths in degrees in the ring plane (``pa + 90`` and
        ``pa - 90`` are the walls at those position angles).

    Examples
    --------
    >>> cone = TruncatedCone(
    ...     tip=5.0, alpha=30.0, s0=4.0, length=10.0, width=1.0
    ... )
    """

    tip: jax.Array
    alpha: jax.Array
    s0: jax.Array
    length: jax.Array
    width: jax.Array
    tilt: jax.Array
    pa: jax.Array
    ratio: jax.Array
    # None when unmodulated, not an empty array: zero-size leaves crash
    # pmapped samplers (see #282).
    az_amps: jax.Array | None
    az_pas: jax.Array | None
    n_rings: int = eqx.field(static=True)

    def __init__(
        self,
        tip,
        alpha,
        s0,
        length,
        width,
        tilt=0.0,
        pa=0.0,
        ratio=1.0,
        flux=1.0,
        dra=0.0,
        ddec=0.0,
        n_rings=48,
        az_amps=(),
        az_pas=(),
    ):
        self.tip = np.asarray(tip, dtype=float)
        self.alpha = np.asarray(alpha, dtype=float)
        self.s0 = np.asarray(s0, dtype=float)
        self.length = np.asarray(length, dtype=float)
        self.width = np.asarray(width, dtype=float)
        self.tilt = np.asarray(tilt, dtype=float)
        self.pa = np.asarray(pa, dtype=float)
        self.ratio = np.asarray(ratio, dtype=float)
        self._place(flux, dra, ddec)
        if isinstance(n_rings, bool) or int(n_rings) != n_rings or n_rings < 2:
            raise ValueError(
                f"n_rings must be an integer >= 2, got {n_rings}."
            )
        self.n_rings = int(n_rings)
        amps = _modulation_array(az_amps, "az_amps")
        pas = _modulation_array(az_pas, "az_pas")
        if amps.shape != pas.shape:
            raise ValueError(
                f"az_amps has {amps.size} entries but az_pas has "
                f"{pas.size}; give one azimuth per modulation."
            )
        if amps.size:
            _check_non_negative_modulation(amps, pas)
            self.az_amps, self.az_pas = amps, pas
        else:
            self.az_amps = self.az_pas = None

    def _shape_checks(self):
        return (
            (
                "alpha",
                self.alpha,
                lambda x: (x > 0.0) & (x < 90.0),
                "in (0, 90)",
            ),
            ("s0", self.s0, lambda x: x >= 0.0, "non-negative"),
            ("length", self.length, _positive, "positive"),
            ("width", self.width, _positive, "positive"),
            ("tilt", self.tilt, lambda x: abs(x) <= 90.0, "in [-90, 90]"),
            ("ratio", self.ratio, _positive, "positive"),
        )

    def is_physical(self):
        return super().is_physical() & (
            True
            if self.az_amps is None
            else np.asarray(
                check_az_prof_nonnegative(self.az_amps, self.az_pas)
            )
        )

    def _ring_angle(self, sky_pa):
        """The ring azimuth whose projection points at ``sky_pa``.

        A ring point at azimuth ``az`` lies ``cos φ`` across and
        ``ratio sin(tilt) sin φ`` along the projected axis, with
        ``φ = pa + 90 - az``, so a sky angle is deprojected by that squash
        (its sign included). Seen edge-on (``tilt = 0``), every direction
        off the walls is the near or far side of the ring.
        """
        squash = self.ratio * np.sin(self.tilt * dtor)
        squash = np.where(squash < 0.0, -1.0, 1.0) * np.maximum(
            np.abs(squash), 1e-8
        )
        d = (sky_pa - self.pa) * dtor
        phi = np.arctan2(np.cos(d) / squash, np.sin(d))
        return self.pa + 90.0 - phi / dtor

    def _radial_profile(self):
        sin_alpha = np.sin(self.alpha * dtor)
        return ExpTail(
            self.s0 * sin_alpha, self.length * sin_alpha, cutoff=5.0
        ).with_rule(self.n_rings)

    def rings(self):
        """Radius, sky offset of the centre along the projected axis (mas)
        and normalized weight of each ring: the Gauss-Legendre nodes that
        ``model`` integrates and ``render`` draws."""
        r, w = self._radial_profile().quadrature()
        along = Cone(self.alpha, self.tip)(r) * np.cos(self.tilt * dtor)
        return r, along, w / np.sum(w)

    def _centred_cvis(self, uu, vv):
        # The cone is a MultipoleDisk: an exponential tail in the radius of the
        # rings (the slant distance times sin alpha), a conical height, and
        # the walls' modulation as linear harmonics. The disk's major axis
        # is perpendicular to the projected axis of the cone, and the
        # minor-axis foreshortening ratio * sin(tilt) is passed as is.
        r, w = self._radial_profile().quadrature()
        height = Cone(self.alpha, self.tip)(r)
        c = s = None
        if self.az_amps is not None:
            # The wall modulation as the engine's Cartesian (c, s), one row
            # per order, repeated over the rings.
            c, s = cone_coefficients(self.az_amps, self.az_pas, self.pa)
            c, s = c[:, None] * np.ones_like(r), s[:, None] * np.ones_like(r)
        cvis = multipole_disk_cvis(
            uu,
            vv,
            r,
            w,
            inc=90.0 - np.abs(self.tilt),
            pa=self.pa - 90.0,
            c=c,
            s=s,
            height=height,
            stretch=self.ratio * np.sin(self.tilt * dtor),
        )
        return cvis * _cvis_gaussian_envelope(uu, vv, self.width)

    def _centred_image(self, xx, yy, pixel_scale_mas):
        rho, along, weight = self.rings()
        sin_pa, cos_pa = np.sin(self.pa * dtor), np.cos(self.pa * dtor)
        squash = self.ratio * np.sin(self.tilt * dtor)
        width = np.maximum(self.width, pixel_scale_mas)
        # Points round each ring no further apart than a third of the blur,
        # so a large thin ring is a continuous band, not a string of spots.
        block = 64
        circumference = concrete(
            2.0 * np.pi * np.max(rho) * np.maximum(1.0, self.ratio)
        )
        blur = concrete(width)
        if circumference is None or blur is None:
            n_blocks = 4  # traced: a fixed sampling
        else:
            needed = float(np.max(circumference)) / (float(np.min(blur)) / 3.0)
            n_blocks = int(min(max(onp.ceil(needed / block), 1), 128))
        phi = np.linspace(0.0, 2.0 * np.pi, n_blocks * block, endpoint=False)
        modulation = np.ones_like(phi)
        if self.az_amps is not None:
            # In-ring phase of each order, measured like the ring angle here
            # (0 on the wall at pa + 90, towards the side projected along pa).
            phases = (self.pa + 90.0 - self.az_pas) * dtor
            modulation = 1.0 + sum(
                self.az_amps[m] * np.cos((m + 1) * (phi - phases[m]))
                for m in range(self.az_amps.size)
            )

        def ring_points(radius, centre, angles):
            par = centre + radius * np.sin(angles) * squash
            perp = radius * np.cos(angles)
            return (
                par * sin_pa + perp * cos_pa,
                par * cos_pa - perp * sin_pa,
            )

        return _splat_rings(
            xx,
            yy,
            width,
            (rho, along),
            lambda radius, centre: ring_points(radius, centre, phi),
            weight[:, None] * modulation[None, :] / phi.size,
        )


def _splat_rings(xx, yy, width, rings, ring_points, weights):
    """Image of rings of Gaussian spots (FWHM ``width``), summed over rings.

    ``ring_points(*ring)`` gives the ``(x, y)`` of the spots of one ring, where
    ``ring`` is a row of the arrays in ``rings``; ``weights`` is
    ``(n_rings, n_points)``. On a regular grid a ring is one matrix product of
    row and column profiles; otherwise the spots are added in blocks of 64 to
    keep memory small.
    """
    k = 4.0 * np.log(2.0) / width**2
    n_points = weights.shape[-1]
    grid = _orthogonal_grid(xx, yy)
    if grid is not None:
        origin, e_col, e_row = grid
        cols = np.arange(np.shape(xx)[1])[:, None]
        rows = np.arange(np.shape(xx)[0])[:, None]

        def add_ring(image, ring):
            *args, wt = ring
            x, y = ring_points(*args)
            dx, dy = x - origin[0], y - origin[1]
            s_col = (dx * e_col[0] + dy * e_col[1]) / (e_col @ e_col)
            s_row = (dx * e_row[0] + dy * e_row[1]) / (e_row @ e_row)
            g_col = np.exp(-k * (e_col @ e_col) * (cols - s_col) ** 2)
            g_row = np.exp(-k * (e_row @ e_row) * (rows - s_row) ** 2)
            spots = np.matmul(
                g_row * wt, g_col.T, precision=jax.lax.Precision.HIGHEST
            )
            return image + spots, None

    else:
        blocks = n_points // 64 + (n_points % 64 > 0)
        pad_to = blocks * 64

        def add_ring(image, ring):
            *args, wt = ring
            x, y = ring_points(*args)
            x, y = (np.pad(a, (0, pad_to - n_points)) for a in (x, y))
            wt = np.pad(wt, (0, pad_to - n_points))

            def add_block(image, block):
                bx, by, bw = block
                d2 = (xx[..., None] - bx) ** 2 + (yy[..., None] - by) ** 2
                return image + np.sum(np.exp(-k * d2) * bw, -1), None

            image, _ = jax.lax.scan(
                add_block,
                image,
                tuple(a.reshape(blocks, 64) for a in (x, y, wt)),
            )
            return image, None

    image, _ = jax.lax.scan(
        add_ring, np.zeros(np.shape(xx)), (*rings, weights)
    )
    return image


# deliberate: monkeypatched in tests/test_models_sources.py to force the irregular-grid path
def _orthogonal_grid(xx, yy):
    """Origin and column and row steps of a regular grid, or ``None``.

    The grid may be shifted and rotated (as ``Component._image`` and
    ``Image.from_model`` make it) but its axes must be orthogonal. Returns
    ``None`` for traced or irregular coordinates.
    """
    xx, yy = concrete(xx), concrete(yy)
    if xx is None or yy is None or xx.ndim != 2 or min(xx.shape) < 2:
        return None
    xx, yy = xx.astype(float), yy.astype(float)
    origin = onp.array([xx[0, 0], yy[0, 0]])
    e_col = onp.array([xx[0, 1], yy[0, 1]]) - origin
    e_row = onp.array([xx[1, 0], yy[1, 0]]) - origin
    step = min(onp.hypot(*e_col), onp.hypot(*e_row))
    rows, cols = onp.indices(xx.shape)
    tol = 1e-3 * step
    if (
        step <= 0.0
        or abs(e_col @ e_row) > tol * step
        or onp.max(onp.abs(origin[0] + cols * e_col[0] + rows * e_row[0] - xx))
        > tol
        or onp.max(onp.abs(origin[1] + cols * e_col[1] + rows * e_row[1] - yy))
        > tol
    ):
        return None
    return origin, e_col, e_row


class UniformDisk(Component):
    """Uniformly bright (tophat) circular disk, e.g. a resolved stellar photosphere.

    Parameters
    ----------
    diam : float or array-like
        Angular diameter in milliarcseconds.
    flux, dra, ddec : optional
        Relative weight (or spectrum) and sky offset in milliarcseconds,
        as for any [`Component`][virgil.models.Component].

    Examples
    --------
    >>> photosphere = UniformDisk(diam=3.0)
    """

    diam: jax.Array

    def __init__(self, diam, flux=1.0, dra=0.0, ddec=0.0):
        self.diam = np.asarray(diam, dtype=float)
        self._place(flux, dra, ddec)

    def _centred_cvis(self, uu, vv):
        return cvis_uniform_disk(uu, vv, self.diam)

    def _centred_image(self, xx, yy, pixel_scale_mas):
        radius = np.maximum(self.diam / 2.0, 0.5 * pixel_scale_mas)
        return np.where(xx**2 + yy**2 <= radius**2, 1.0, 0.0)


class _LimbDarkenedDisk(Component):
    r"""Circular disk whose brightness is a sum of powers of $\mu$.

    Subclasses give the profile $I(\mu) = \sum_\nu a_\nu \mu^\nu$, with
    $\mu = \sqrt{1 - (r / R)^2}$, through ``_profile``, which returns the
    coefficients $a_\nu$ (traceable) and the powers $\nu$ (a static tuple);
    the visibility is then
    [`cvis_limb_darkened_disk`][virgil.models.cvis_limb_darkened_disk].
    """

    diam: jax.Array

    def _profile(self):
        r"""``(coeffs, powers)`` of $I(\mu) = \sum a_\nu \mu^\nu$."""
        raise NotImplementedError

    def _set_position(self, diam, flux, dra, ddec):
        self.diam = np.asarray(diam, dtype=float)
        self._place(flux, dra, ddec)

    def _centred_cvis(self, uu, vv):
        coeffs, powers = self._profile()
        return cvis_limb_darkened_disk(uu, vv, self.diam, coeffs, powers)

    def _centred_image(self, xx, yy, pixel_scale_mas):
        coeffs, powers = self._profile()
        radius = np.maximum(self.diam / 2.0, 0.5 * pixel_scale_mas)
        r2 = (xx**2 + yy**2) / radius**2
        inside = r2 <= 1.0
        # mu = 0 at the limb; keep it positive so mu**nu has finite gradients
        tiny = np.finfo(r2.dtype).tiny
        mu = np.sqrt(np.where(inside, np.maximum(1.0 - r2, tiny), 1.0))
        brightness = sum(a * mu**nu for a, nu in zip(coeffs, powers))
        return np.where(inside, brightness, 0.0)

    def is_physical(self):
        r"""Positive diameter and a profile that is nowhere negative.

        The profile is checked on a grid of 101 values of $\mu$ from 0 to 1;
        each law has $I(1) = 1$, so this also keeps the flux positive.
        """
        coeffs, powers = self._profile()
        mu = np.linspace(0.0, 1.0, 101)
        brightness = sum(a * mu**nu for a, nu in zip(coeffs, powers))
        return (
            super().is_physical()
            & np.all(self.diam > 0.0)
            & np.all(brightness >= 0.0)
        )


class LimbDarkenedDisk(_LimbDarkenedDisk):
    r"""Circular disk with polynomial limb darkening of order up to 22.

    The brightness is $I(\mu) / I(1) = 1 - \sum_{n=1}^{N} u_n (1 - \mu)^n$,
    with $\mu = \sqrt{1 - (r / R)^2}$; ``u`` is the same as a jaxoplanet
    ``Surface``'s ``u`` (``u=(u1,)`` linear, ``u=(u1, u2)`` quadratic, default
    ``()`` a uniform disk). For quadratic priors covering exactly the physical
    profiles use
    [`QuadraticLimbDarkenedDisk`][virgil.models.QuadraticLimbDarkenedDisk]. The
    order is at most 22, as the Bessel functions needed are of order at most
    12. The visibility is analytic
    ([`cvis_limb_darkened_disk`][virgil.models.cvis_limb_darkened_disk]).

    Source: Quirrenbach et al. (1996), eq. 3,
    [references](../../../references.md#quirrenbach1996); harmonix, [Dholakia &
    Pope 2025](https://arxiv.org/abs/2509.25433).

    Parameters
    ----------
    diam : float or array-like
        Limb-darkened angular diameter (of the stellar limb) in mas.
    u : sequence of float, optional
        Coefficients $u_1, \ldots, u_N$, a 1D array of fixed length (the order,
        at most 22); default none.
        [`is_physical`][virgil.models.SourceModel.is_physical] is false where
        the profile goes negative.
    flux, dra, ddec : optional
        As for any [`Component`][virgil.models.Component].

    Examples
    --------
    >>> quadratic = LimbDarkenedDisk(3.0, u=[0.4, 0.25])
    """

    u: jax.Array

    def __init__(self, diam, u=(), flux=1.0, dra=0.0, ddec=0.0):
        self._set_position(diam, flux, dra, ddec)
        self.u = np.asarray(u, dtype=float).reshape(-1)

    def _profile(self):
        # (1 - mu)^n = sum_k C(n, k) (-mu)^k
        order = self.u.shape[0]
        expand = onp.zeros((order, order + 1))
        for n in range(1, order + 1):
            for k in range(n + 1):
                expand[n - 1, k] = math.comb(n, k) * (-1.0) ** k
        coeffs = np.zeros(order + 1).at[0].set(1.0) - np.matmul(
            self.u, expand, precision=jax.lax.Precision.HIGHEST
        )
        return coeffs, tuple(float(k) for k in range(order + 1))


class EllipticalLimbDarkenedDisk(_LimbDarkenedDisk):
    r"""Limb-darkened disk with an elliptical outline, e.g. an oblate star.

    A [`LimbDarkenedDisk`][virgil.models.LimbDarkenedDisk] compressed along its
    minor axis, with the profile $I(\mu)/I(1) = 1 - \sum_n u_n (1 - \mu)^n$
    following the outline ($\mu = \sqrt{1 - \rho^2}$, $\rho$ the elliptical
    radius). Unit flux on its own; the visibility is the circular disk's at
    frequencies stretched by ``ratio``. A geometric ellipse, not gravity
    darkening (use [`GravityDarkenedStar`][virgil.models.GravityDarkenedStar]).

    Parameters
    ----------
    diam : float or array-like
        Angular diameter of the limb along the major axis, in mas.
    ratio, pa : float or array-like, optional
        Minor-to-major axis ratio in (0, 1] (default 1) and position angle of
        the major axis in degrees North to East (default 0).
    u : sequence of float, optional
        Coefficients as in [`LimbDarkenedDisk`][virgil.models.LimbDarkenedDisk]
        (default uniform);
        [`is_physical`][virgil.models.SourceModel.is_physical] is false where
        the profile goes negative.
    flux, dra, ddec : optional
        As for any [`Component`][virgil.models.Component].

    Examples
    --------
    >>> star = EllipticalLimbDarkenedDisk(5.5, ratio=0.8, pa=30.0, u=[0.5])
    """

    u: jax.Array
    ratio: jax.Array
    pa: jax.Array

    def __init__(
        self, diam, ratio=1.0, pa=0.0, u=(), flux=1.0, dra=0.0, ddec=0.0
    ):
        self._set_position(diam, flux, dra, ddec)
        self.u = np.asarray(u, dtype=float).reshape(-1)
        self.ratio = np.asarray(ratio, dtype=float)
        self.pa = np.asarray(pa, dtype=float)

    # The same law, and the same ``u``, as LimbDarkenedDisk.
    _profile = LimbDarkenedDisk._profile

    def _shape_checks(self):
        return (
            ("diam", self.diam, _positive, "positive"),
            ("ratio", self.ratio, _in_unit_interval, "in (0, 1]"),
        )

    def _centred_cvis(self, uu, vv):
        # In the frame where the outline is a circle of the major axis's
        # diameter, the visibility is the circular disk's. The frequencies
        # at zero baseline stay zero, so the visibility there stays 1.
        ut, vt = undo_elliptical_transf_spat_freq(uu, vv, self.pa, self.ratio)
        return super()._centred_cvis(ut, vt)

    def _centred_image(self, xx, yy, pixel_scale_mas):
        # Unnormalized; render divides by the sum, so no 1/ratio is needed.
        ratio = np.maximum(self.ratio, 1e-9)
        xt, yt = undo_elliptical_transf_coord(xx, yy, self.pa, ratio)
        return super()._centred_image(xt, yt, pixel_scale_mas)


class QuadraticLimbDarkenedDisk(_LimbDarkenedDisk):
    r"""Circular disk with quadratic limb darkening in Kipping's $q_1, q_2$.

    The brightness is $I(\mu) / I(1) = 1 - u_1 (1 - \mu) - u_2 (1 - \mu)^2$,
    with $u_1 = 2 \sqrt{q_1}\, q_2$ and $u_2 = \sqrt{q_1}\,(1 - 2 q_2)$. Every
    $(q_1, q_2)$ in the unit square gives a profile that is positive and
    decreases from the centre to the limb, and every such profile has one, so
    uniform priors on $[0, 1]$ are uninformative over exactly the physical
    laws. The visibility is analytic
    ([`cvis_limb_darkened_disk`][virgil.models.cvis_limb_darkened_disk]).

    Source: Kipping (2013), eqs. 15-16,
    [references](../../../references.md#kipping2013).

    Parameters
    ----------
    diam : float or array-like
        Limb-darkened angular diameter in mas.
    q1, q2 : float or array-like, optional
        Kipping's coefficients, each in $[0, 1]$; $q_1 = 0$ is a uniform disk.
        Start a fit inside the square, not on its edges: ``fit`` maps Uniform
        priors to unbounded variables, so the edges map to infinity (and
        $\sqrt{q_1}$ has an infinite derivative at 0).
    flux, dra, ddec : optional
        As for any [`Component`][virgil.models.Component].

    Examples
    --------
    >>> import numpyro.distributions as dist
    >>> star = QuadraticLimbDarkenedDisk(3.0, q1=0.4, q2=0.3)
    >>> priors = {
    ...     "diam": dist.LogUniform(2.0, 4.0),  # a scale: log-uniform
    ...     "q1": dist.Uniform(0.0, 1.0),  # Kipping's uniform prior
    ...     "q2": dist.Uniform(0.0, 1.0),
    ... }
    >>> star = QuadraticLimbDarkenedDisk.from_u(3.0, u1=0.4, u2=0.25)
    """

    q1: jax.Array
    q2: jax.Array

    def __init__(self, diam, q1, q2, flux=1.0, dra=0.0, ddec=0.0):
        self._set_position(diam, flux, dra, ddec)
        self.q1 = np.asarray(q1, dtype=float)
        self.q2 = np.asarray(q2, dtype=float)

    @classmethod
    def from_u(cls, diam, u1, u2, **kwargs):
        """Build from the usual $u_1, u_2$ (Kipping 2013, eqs. 17-18)."""
        total = u1 + u2
        q2 = np.where(
            total == 0, 0.0, u1 / (2.0 * np.where(total == 0, 1.0, total))
        )
        return cls(diam, q1=total**2, q2=q2, **kwargs)

    @property
    def u1(self):
        return 2.0 * np.sqrt(self.q1) * self.q2

    @property
    def u2(self):
        return np.sqrt(self.q1) * (1.0 - 2.0 * self.q2)

    def _profile(self):
        u1, u2 = self.u1, self.u2
        # 1 - u1 (1 - mu) - u2 (1 - mu)^2, expanded in powers of mu
        coeffs = np.stack([1.0 - u1 - u2, u1 + 2.0 * u2, -u2])
        return coeffs, (0.0, 1.0, 2.0)

    def is_physical(self):
        return super().is_physical() & _in_unit_square(self.q1, self.q2)


class SquareRootLimbDarkenedDisk(_LimbDarkenedDisk):
    r"""Circular disk with square-root limb darkening in Kipping's $q_1, q_2$.

    The brightness is $I(\mu) / I(1) = 1 - c (1 - \mu) - d (1 - \sqrt{\mu})$,
    which suits late-type stars in the near-infrared better than the quadratic
    law (van Hamme 1993), with $c = \sqrt{q_1}\,(1 - 2 q_2)$ and $d = 2
    \sqrt{q_1}\, q_2$. As for
    [`QuadraticLimbDarkenedDisk`][virgil.models.QuadraticLimbDarkenedDisk], the
    unit square in $(q_1, q_2)$ is exactly the positive profiles that decrease
    towards the limb, so uniform priors on $[0, 1]$ are uninformative. The
    visibility is analytic
    ([`cvis_limb_darkened_disk`][virgil.models.cvis_limb_darkened_disk]).

    Source: Díaz-Cordovés & Giménez (1992),
    [references](../../../references.md#diazcordoves1992); Kipping (2013), eqs.
    23-24 inverted.

    Parameters
    ----------
    diam : float or array-like
        Limb-darkened angular diameter in mas.
    q1, q2 : float or array-like, optional
        Kipping's coefficients, each in $[0, 1]$; $q_1 = 0$ is a uniform disk.
        Start a fit inside the square, not on its edges: ``fit`` maps Uniform
        priors to unbounded variables, so the edges map to infinity (and
        $\sqrt{q_1}$ has an infinite derivative at 0).
    flux, dra, ddec : optional
        As for any [`Component`][virgil.models.Component].

    Examples
    --------
    >>> star = SquareRootLimbDarkenedDisk(3.0, q1=0.5, q2=0.4)
    >>> same = SquareRootLimbDarkenedDisk.from_cd(3.0, c=star.c, d=star.d)
    """

    q1: jax.Array
    q2: jax.Array

    def __init__(self, diam, q1, q2, flux=1.0, dra=0.0, ddec=0.0):
        self._set_position(diam, flux, dra, ddec)
        self.q1 = np.asarray(q1, dtype=float)
        self.q2 = np.asarray(q2, dtype=float)

    @classmethod
    def from_cd(cls, diam, c, d, **kwargs):
        """Build from the usual $c, d$ (Kipping 2013, eqs. 23-24)."""
        total = c + d
        q2 = np.where(
            total == 0, 0.0, d / (2.0 * np.where(total == 0, 1.0, total))
        )
        return cls(diam, q1=total**2, q2=q2, **kwargs)

    @property
    def c(self):
        return np.sqrt(self.q1) * (1.0 - 2.0 * self.q2)

    @property
    def d(self):
        return 2.0 * np.sqrt(self.q1) * self.q2

    def _profile(self):
        c, d = self.c, self.d
        # 1 - c (1 - mu) - d (1 - sqrt(mu)), in powers of mu
        return np.stack([1.0 - c - d, c, d]), (0.0, 1.0, 0.5)

    def is_physical(self):
        return super().is_physical() & _in_unit_square(self.q1, self.q2)


def _in_unit_square(q1, q2):
    return np.all((q1 >= 0.0) & (q1 <= 1.0) & (q2 >= 0.0) & (q2 <= 1.0))


class GravityDarkenedStar(Component):
    """Rapidly rotating star: Roche shape and gravity darkening (ELR11).

    A rigidly rotating star with the oblate Roche shape and local bolometric
    flux following the effective gravity (no free exponent, no limb darkening),
    summed over the visible triangles of a surface mesh: see the
    [tutorial](../../../gravity_darkened_star.md) and [Model
    notes](../notes.md#gravitydarkenedstar) (chromatic mode, the mapping to the
    original parameters, accuracy of ``n_lat``).

    Source: Espinosa Lara & Rieutord (2011), via GDit,
    [references](../../../references.md#espinosalara2011).

    Parameters
    ----------
    diam_eq : float or array-like
        Equatorial angular diameter in mas.
    omega : float or array-like, optional
        Angular velocity as a fraction of the critical rate, in [0, 1) (default
        0).
    inc, pa : float or array-like, optional
        Inclination in degrees, 0 pole-on to 90 equator-on (default 90), and
        position angle of the visible pole in degrees North through East
        (default 0).
    flux, dra, ddec : optional
        As for any [`Component`][virgil.models.Component]; in chromatic mode
        ``flux`` must be a number, not a [`Spectrum`][virgil.spectra.Spectrum].
    n_lat : int, optional
        Latitude rings of the mesh (default 32, 2520 triangles); cost
        O(``n_lat``$^2$) per baseline, error falls as ``1 / n_lat**2``; check
        by doubling.
    t_pole, wavel0 : float or array-like, optional
        Polar effective temperature in kelvin (``None``, the default, is the
        grey model; a value switches on the chromatic Planck model) and the
        reference wavelength in metres (default 1.65e-6) where the spectrum is
        normalized to ``flux``, shown by ``render``.

    Examples
    --------
    >>> star = GravityDarkenedStar(2.0, omega=0.8, inc=60.0, pa=30.0)
    >>> v0 = star.model(np.zeros(1), np.zeros(1), 1.65e-6)
    >>> round(float(np.abs(v0)[0]), 3)
    1.0
    """

    diam_eq: jax.Array
    omega: jax.Array
    inc: jax.Array
    pa: jax.Array
    n_lat: int = eqx.field(static=True)
    t_pole: jax.Array | None
    wavel0: jax.Array

    def __init__(
        self,
        diam_eq,
        omega=0.0,
        inc=90.0,
        pa=0.0,
        flux=1.0,
        dra=0.0,
        ddec=0.0,
        n_lat=32,
        t_pole=None,
        wavel0=1.65e-6,
    ):
        self.diam_eq = np.asarray(diam_eq, dtype=float)
        self.omega = np.asarray(omega, dtype=float)
        self.inc = np.asarray(inc, dtype=float)
        self.pa = np.asarray(pa, dtype=float)
        self._place(flux, dra, ddec)
        if isinstance(n_lat, bool) or int(n_lat) != n_lat or n_lat < 4:
            raise ValueError(f"n_lat must be an integer >= 4, got {n_lat}.")
        self.n_lat = int(n_lat)
        self.t_pole = None if t_pole is None else np.asarray(t_pole, float)
        self.wavel0 = np.asarray(wavel0, dtype=float)

    def __check_init__(self):
        super().__check_init__()
        name = type(self).__name__
        if self.t_pole is not None and isinstance(self.flux, Spectrum):
            raise ValueError(
                f"{name} with t_pole supplies its own spectrum, so flux "
                "must be a number, not a Spectrum (it would be counted "
                "twice)."
            )
        for key, value in (("t_pole", self.t_pole), ("wavel0", self.wavel0)):
            value = None if value is None else concrete(value)
            if value is not None and onp.any(value <= 0.0):
                raise ValueError(
                    f"{name} has {key} {value.tolist()}; it must be positive."
                )
        diam, omega = concrete(self.diam_eq), concrete(self.omega)
        inc = concrete(self.inc)
        if diam is not None and onp.any(diam <= 0.0):
            raise ValueError(
                f"{name} has diam_eq {diam.tolist()}; it must be positive."
            )
        if omega is not None and onp.any((omega < 0.0) | (omega >= 1.0)):
            raise ValueError(
                f"{name} has omega {omega.tolist()}; it must be in [0, 1)."
            )
        if inc is not None and onp.any((inc < 0.0) | (inc > 90.0)):
            raise ValueError(
                f"{name} has inc {inc.tolist()}; it must be in [0, 90] "
                "degrees (90 = equator-on, 0 = pole-on)."
            )

    def is_physical(self):
        return (
            super().is_physical()
            & np.all(self.diam_eq > 0.0)
            & np.all((self.omega >= 0.0) & (self.omega < 1.0))
            & np.all((self.inc >= 0.0) & (self.inc <= 90.0))
            & (
                True
                if self.t_pole is None
                else np.all(self.t_pole > 0.0) & np.all(self.wavel0 > 0.0)
            )
        )

    def _surface(self, **kwargs):
        # his inc = 0 is equator-on; his obl is the pole position angle
        return _elr.surface(
            self.omega,
            self.diam_eq / 2.0,
            (90.0 - self.inc) * dtor,
            self.pa * dtor,
            self.n_lat,
            **kwargs,
        )

    def _planck_weights(self, wavel):
        """Chromatic mode: ``(x, y, weights)``, weights of shape (*wavel.shape, n_tri).

        Each triangle's temperature is ``t_pole`` times its ELR11 ``Teff``
        over the pole's; its weight is projected area times
        ``B_λ(T) / B_λ(t_pole)`` at ``wavel``, so numbers stay O(1).
        """
        x, y, _, teff, (_, _, cosine, _) = self._surface(return_mesh=True)
        area = np.heaviside(cosine, 0.0) * cosine
        return x, y, area * self._planck_intensity(teff, wavel)

    def _planck_intensity(self, teff, wavel):
        """``B_λ(T) / B_λ(t_pole)`` at ``wavel`` for triangles of ``teff``."""
        # the pole, theta = 0, through the jitted vectorized solver
        teff_pole = _elr.solve_ELR_vec(self.omega, np.zeros(1))[1][0]
        temperature = self.t_pole * teff / teff_pole
        wavel = np.asarray(wavel)[..., None]
        return _planck_ratio(wavel, temperature, self.wavel0, self.t_pole)

    def model(self, u, v, wavel):
        if self.t_pole is None:
            return super().model(u, v, wavel)
        uu, vv = u / wavel, v / wavel
        wavel = np.asarray(wavel)
        # one wavelength for all samples, or one per (flattened) sample
        if wavel.size == 1:
            lam = wavel.reshape(1)
        else:
            lam = np.broadcast_to(wavel, uu.shape).reshape(-1)
        x, y, w = self._planck_weights(lam)
        return _elr.visibilities(x, y, w, uu, vv) * offset_phase(
            uu, vv, self.dra, self.ddec
        )

    def _weight(self, wavel=None):
        if self.t_pole is None:
            return super()._weight(wavel)
        if wavel is None:
            return flux_at(self.flux)
        # SED(λ) = sum of area * B_λ(T) over the surface, relative to wavel0
        sed = self._planck_weights(wavel)[2].sum(axis=-1)
        sed0 = self._planck_weights(self.wavel0)[2].sum(axis=-1)
        return flux_at(self.flux) * sed / sed0

    def _centred_cvis(self, uu, vv):
        x, y, w, _ = self._surface()
        return _elr.visibilities(x, y, w, uu, vv)

    # sub-pixel samples per pixel side when rasterizing the image
    _image_oversample = 4

    def _centred_image(self, xx, yy, pixel_scale_mas):
        """Rasterize the faceted surface, flat-shaded per triangle.

        Each pixel averages ``_image_oversample`` squared sub-pixel samples
        of the surface brightness, so the limb is anti-aliased. A sample
        takes the intensity of the front-facing triangle that contains it in
        projection (those do not overlap, so no depth sorting is needed) and
        0 outside the outline. This is what ``plot_surface`` draws and what
        the visibilities transform, as opposed to splatting the barycentres
        onto pixels, which blurs the limb and aliases against the mesh.
        The sum matches the total weight of the DFT's point sources.
        """
        *_, teff, (pts, tri, cosine, intensity) = self._surface(
            return_mesh=True
        )
        if self.t_pole is not None:  # drawn at wavel0
            intensity = self._planck_intensity(teff, self.wavel0)
        px, py = pts[tri, 0], pts[tri, 1]  # (n_tri, 3) corners on the sky
        ax, ay = px[:, 0], py[:, 0]
        e1x, e1y = px[:, 1] - ax, py[:, 1] - ay
        e2x, e2y = px[:, 2] - ax, py[:, 2] - ay
        det = e1x * e2y - e1y * e2x
        # back-facing and degenerate triangles never contain a sample
        valid = (cosine > 0) & (det != 0)
        inv = 1.0 / np.where(valid, det, 1.0)
        value = np.where(valid, intensity, 0.0)

        def sample_row(pos):
            sx, sy = pos[0] - ax[:, None], pos[1] - ay[:, None]
            wb = (sx * e2y[:, None] - sy * e2x[:, None]) * inv[:, None]
            wc = (e1x[:, None] * sy - e1y[:, None] * sx) * inv[:, None]
            inside = valid[:, None] & (wb >= 0) & (wc >= 0) & (wb + wc <= 1)
            hits = np.sum(inside, axis=0)
            # a sample exactly on a shared edge counts once, not twice
            total = np.sum(np.where(inside, value[:, None], 0.0), axis=0)
            return total / np.maximum(hits, 1)

        # sub-pixel sample positions; the pixel centres are the grid xx, yy
        s = self._image_oversample
        nrow, ncol = xx.shape
        frac = ((np.arange(s) + 0.5) / s - 0.5) * pixel_scale_mas
        sx = (xx[0, :, None] + frac).reshape(-1)  # (ncol * s,)
        sy = (yy[:, 0, None] + frac).reshape(-1)  # (nrow * s,)
        # one sub-row of samples at a time bounds memory by ncol * s * n_tri
        sub_rows = jax.lax.map(
            lambda y: sample_row((sx, y)), sy.astype(intensity.dtype)
        )
        mean = sub_rows.reshape(nrow, s, ncol, s).mean(axis=(1, 3))
        # |normal| = 2 x area, so cosine carries twice the projected area
        # in the DFT weights; the pixel area carries the same factor here
        return 2.0 * pixel_scale_mas**2 * mean

    def plot_surface(self, ax=None, cmap="plasma"):
        """Plot the visible surface, coloured by its local brightness.

        That is the bolometric flux in grey mode, and the Planck intensity at
        ``wavel0`` in chromatic mode. East is to the left and North up, as in
        [`plot_model`][virgil.plotting.plot_model]; the offset
        ``dra``, ``ddec`` is not applied. Returns the matplotlib collection.
        """
        import matplotlib.pyplot as plt
        import matplotlib.tri as mtri

        if ax is None:
            _, ax = plt.subplots()
        *_, teff, (pts, tri, cosine, intensity) = self._surface(
            return_mesh=True
        )
        if self.t_pole is not None:  # chromatic: as seen at wavel0
            intensity = self._planck_intensity(teff, self.wavel0)
        pts, tri = onp.asarray(pts), onp.asarray(tri)
        visible = onp.asarray(cosine) > 0
        triang = mtri.Triangulation(pts[:, 0], pts[:, 1], tri[visible])
        coll = ax.tripcolor(
            triang, facecolors=onp.asarray(intensity)[visible], cmap=cmap
        )
        ax.set_aspect("equal")
        if not ax.xaxis_inverted():
            ax.invert_xaxis()
        ax.set(xlabel="ΔRA (mas)", ylabel="ΔDec (mas)")
        ax.figure.colorbar(coll, ax=ax, label="Relative local flux")
        return coll


def _blur_in_plane(image, fwhm, pa, stretch, pixel_scale_mas):
    """``image`` convolved with a Gaussian of FWHM ``fwhm`` (mas) that is
    isotropic in the plane of a disk inclined by ``stretch = cos(inc)``.

    The kernel is odd-sized and centred on a pixel, so ``mode="same"`` adds no
    shift. Its minor-axis sigma on the sky is kept at least half a pixel (or
    ``sigma``, if smaller), so a near edge-on kernel cannot fall between pixel
    centres, and the stretch is capped at 1 so that a face-on kernel stays
    isotropic.
    """
    npix = image.shape[0]
    nker = npix + 1 - npix % 2
    kx, ky = image_coordinates(nker, nker * pixel_scale_mas)
    sigma_mas = np.maximum(fwhm / FWHM_PER_SIGMA, 1e-9)
    psf_stretch = np.minimum(
        np.maximum(stretch, 0.5 * pixel_scale_mas / sigma_mas), 1.0
    )
    kx_ell, ky_ell = undo_elliptical_transf_coord(kx, ky, pa, psf_stretch)
    psf = np.exp(-0.5 * (kx_ell**2 + ky_ell**2) / sigma_mas**2)
    return fftconvolve(image, psf, mode="same")


def _modulation_array(values, name):
    """Coerce azimuthal-modulation coefficients to a 1D float array."""
    array = np.atleast_1d(np.asarray(values, dtype=float))
    if array.ndim != 1:
        raise ValueError(
            f"{name} must be a scalar or 1D sequence, got shape {array.shape}."
        )
    return array


class ModulatedGaussianRim(Component):
    """Azimuthally modulated, infinitely thin rim convolved with a Gaussian.

    The rim is defined in its own plane, blurred by a Gaussian isotropic there,
    then inclined by ``inc`` and rotated to ``pa``. ``az_pas`` are angles in
    that plane, not on-sky position angles ([Model
    notes](../notes.md#modulatedgaussianrim)). Achromatic and star-free: add a
    [`PointSource`][virgil.models.PointSource] in a
    [`System`][virgil.models.System].

    Parameters
    ----------
    diam, fwhm : float or array-like
        Rim diameter, and Gaussian FWHM in the rim plane (``fwhm`` along the
        projected major axis, ``fwhm * cos(inc)`` along the minor axis on the
        sky), in mas.
    inc, pa : float or array-like
        Apparent inclination, and position angle of the projected major axis
        (North to East), in degrees.
    az_amps, az_pas : float or array-like, optional
        Cosine modulation amplitudes, first order up (default none;
        ``sum(abs(az_amps)) <= 1`` keeps the brightness non-negative), and
        their azimuths in degrees in the rim plane, in the sense of position
        angle with ``pa`` on the major axis.
    flux, dra, ddec : optional
        As for any [`Component`][virgil.models.Component].

    Examples
    --------
    >>> rim = ModulatedGaussianRim(
    ...     diam=40.0, fwhm=4.0, inc=50.0, pa=30.0, az_amps=0.7, az_pas=120.0
    ... )
    """

    diam: jax.Array
    fwhm: jax.Array
    inc: jax.Array
    pa: jax.Array
    az_amps: jax.Array
    az_pas: jax.Array

    def __init__(
        self,
        diam,
        fwhm,
        inc,
        pa,
        az_amps=(),
        az_pas=(),
        flux=1.0,
        dra=0.0,
        ddec=0.0,
    ):
        self.diam = np.asarray(diam, dtype=float)
        self.fwhm = np.asarray(fwhm, dtype=float)
        self.inc = np.asarray(inc, dtype=float)
        self.pa = np.asarray(pa, dtype=float)
        self.az_amps = _modulation_array(az_amps, "az_amps")
        self.az_pas = _modulation_array(az_pas, "az_pas")
        if self.az_amps.shape != self.az_pas.shape:
            raise ValueError(
                f"az_amps has {self.az_amps.size} entries but az_pas has "
                f"{self.az_pas.size}; give one position angle per modulation."
            )
        _check_non_negative_modulation(self.az_amps, self.az_pas)
        self._place(flux, dra, ddec)

    def is_physical(self):
        # Non-negative flux, and a brightness that stays non-negative round
        # the rim.
        return super().is_physical() & np.asarray(
            check_az_prof_nonnegative(self.az_amps, self.az_pas)
        )

    def _ring_angle(self, sky_pa):
        """The in-plane azimuth whose projection points at ``sky_pa``."""
        # Deprojected first: the rim compresses its minor axis by cos(inc).
        d = np.deg2rad(sky_pa - self.pa)
        return self.pa + np.rad2deg(
            np.arctan2(np.sin(d) / np.cos(np.deg2rad(self.inc)), np.cos(d))
        )

    def _centred_cvis(self, uu, vv):
        # The rim is a MultipoleDisk: a ThinRing of radius diam / 2, the
        # modulation as linear harmonics (converted from the amplitudes and
        # azimuths), and the blur of the engine.
        c = s = None
        if self.az_amps.size:
            c, s = rim_coefficients(self.az_amps, self.az_pas, self.pa)
            c, s = c[:, None], s[:, None]
        return multipole_disk_cvis(
            uu,
            vv,
            np.reshape(self.diam / 2.0, (1,)),
            np.ones((1,)),
            self.inc,
            self.pa,
            self.fwhm,
            c=c,
            s=s,
        )

    def _centred_image(self, xx, yy, pixel_scale_mas):
        stretch = np.maximum(np.cos(self.inc * dtor), 1e-8)
        xx_ell, yy_ell = undo_elliptical_transf_coord(xx, yy, self.pa, stretch)
        r_ell = safe_radius(xx_ell, yy_ell)
        theta_ell = safe_azimuth(xx_ell, yy_ell)

        # Anti-aliased thin ring: Gaussian in de-projected radius, at least
        # half a pixel wide on the sky along the compressed minor axis.
        width = 0.5 * pixel_scale_mas / stretch
        ring = np.exp(-0.5 * ((r_ell - self.diam / 2.0) / width) ** 2)

        az_amps = np.concatenate([np.array([1.0]), self.az_amps])
        az_phis_rad = (
            np.concatenate([np.array([0.0]), self.az_pas - self.pa]) * dtor
        )
        az_orders = np.arange(az_amps.size)

        az_factors = jax.vmap(
            lambda amp, phi, order: amp * np.cos(order * (theta_ell - phi))
        )(az_amps, az_phis_rad, az_orders)
        ring = ring * np.sum(az_factors, axis=0)

        return _blur_in_plane(
            ring, self.fwhm, self.pa, stretch, pixel_scale_mas
        )


def circular_support(npix, pixel_scale_mas, radius_mas, inner_radius_mas=0.0):
    """Pixels of an ``npix`` x ``npix`` image within ``radius_mas`` of its centre.

    Returns a boolean array for the ``support`` of an
    [`Image`][virgil.models.Image]. With ``inner_radius_mas``, pixels
    closer to the centre than that are left out too: a hole under an
    analytic star, so that the image cannot pile flux onto it.
    """
    offsets = pixel_offsets(int(npix), float(pixel_scale_mas))
    radius = np.hypot(offsets[None, :], offsets[:, None])
    return (radius <= radius_mas) & (radius >= inner_radius_mas)


def _pixel_visibilities(
    fluxes, pixel_scale_mas, rotation_deg, u, v, wavel, grid=None, ft="auto"
):
    """Fourier transform of pixel fluxes centred on the origin, unnormalized.

    Uses the exact matrix Fourier transform when the samples lie on a uv
    ``grid`` whose rotation matches the pixels' (and there is a single
    wavelength), and the direct transform (``ft``, see
    [`image_visibilities`][virgil._geometry.image_visibilities]) otherwise.
    """
    matched = (
        grid is not None
        and abs(grid.rotation_deg - rotation_deg) < 1e-9
        and np.size(wavel) == 1
    )
    if matched:
        wavel = np.reshape(wavel, ())
        vis = grid_visibilities(
            fluxes, grid.u_axis / wavel, grid.v_axis / wavel, pixel_scale_mas
        )
        return vis.ravel()[grid.index]
    uu, vv = rotate(u / wavel, v / wavel, -rotation_deg)
    return image_visibilities(fluxes, uu, vv, pixel_scale_mas, ft)


class Image(Component):
    """Pixelised brightness distribution, for image reconstruction.

    Pixel fluxes are ``softmax(log_brightness)`` over the pixels in
    ``support``: positive, summing to one, zero outside. Inside a
    [`System`][virgil.models.System] it has a ``flux`` weight and offset like
    any component, so unresolved sources stay analytic; visibilities are the
    exact Fourier transform of the pixels, each a point at its centre. A
    constant added to ``log_brightness`` changes nothing, so `fit` fixes it (a
    free array comes back as ``log`` of the pixel fluxes); a pixel whose
    optimum is zero flux can fit far below the rest (-1e3 or lower, set by
    rounding): read an image from ``brightness``, not from those values.

    Parameters
    ----------
    log_brightness : array-like, shape (nrow, ncol), or GaussianField
        Log pixel fluxes up to a constant, oriented as
        [`render`][virgil.models.SourceModel.render] (row 0 North, column 0
        East), or an object with ``evaluate(pixel_scale_mas)`` such as a
        [`GaussianField`][virgil.fields.GaussianField].
    pixel_scale_mas : float
        Pixel size in mas; the image centre, index ``((nrow - 1) / 2, (ncol -
        1) / 2)``, is at ``(dra, ddec)``.
    support : array-like of bool, optional
        Pixels allowed to carry flux, same shape as ``log_brightness`` (default
        all); see [`circular_support`][virgil.models.circular_support].
    flux, dra, ddec : optional
        As for any [`Component`][virgil.models.Component].
    rotation_deg : float, optional
        Position angle (North to East) of the grid's "up" axis (default 0);
        matching ``data.uv_grid.rotation_deg`` gives a much faster exact
        two-sided matrix transform.
    ft : {"auto", "dft", "nufft"}, optional
        Transform for irregular uv points (default ``"auto"``, whichever is
        faster; see [`set_ft_defaults`][virgil._geometry.set_ft_defaults]).
    blur_fwhm_mas : float, optional
        FWHM in mas of a fixed (not fitted) Gaussian blurring kernel in the
        forward model: the data are compared with the blurred image, so the
        pixels are the unblurred sky. Default ``None``, recommended; blur
        for display after the fit (``smoothed()`` gives the blurred pixels).
        Compounds with ``CalibrationNuisance.blur_fwhm_mas``. Source: MPoL's
        Gaussian base layer, Czekala et al. (2025),
        [references](../references.md#mpol).
    smooth_fwhm_mas : float, optional
        Deprecated alias of ``blur_fwhm_mas``.

    Examples
    --------
    >>> image = Image(np.zeros((32, 32)), pixel_scale_mas=2.0)
    >>> envelope = Image.from_model(GaussianDisk(sigma=8.0), 32, 2.0, flux=0.3)
    """

    log_brightness: Any
    support: jax.Array | None
    pixel_scale_mas: float = eqx.field(static=True)
    rotation_deg: float = eqx.field(static=True)
    ft: str = eqx.field(static=True)
    blur_fwhm_mas: float | None = eqx.field(static=True)

    @renamed({"smooth_fwhm_mas": "blur_fwhm_mas"})
    def __init__(
        self,
        log_brightness,
        pixel_scale_mas,
        support=None,
        flux=1.0,
        dra=0.0,
        ddec=0.0,
        rotation_deg=0.0,
        ft="auto",
        blur_fwhm_mas=None,
    ):
        if blur_fwhm_mas is not None and not (
            onp.isfinite(blur_fwhm_mas) and blur_fwhm_mas > 0.0
        ):
            raise ValueError(
                "blur_fwhm_mas must be None or finite and positive, "
                f"not {blur_fwhm_mas}."
            )
        self.blur_fwhm_mas = (
            None if blur_fwhm_mas is None else float(blur_fwhm_mas)
        )
        self.ft = _check_ft(ft)
        if not hasattr(log_brightness, "evaluate"):
            log_brightness = np.asarray(log_brightness, dtype=float)
        self.log_brightness = log_brightness
        if not (onp.isfinite(pixel_scale_mas) and pixel_scale_mas > 0.0):
            raise ValueError(
                f"pixel_scale_mas must be finite and positive, not {pixel_scale_mas}."
            )
        self.pixel_scale_mas = float(pixel_scale_mas)
        eta = self.eta
        if eta.ndim != 2:
            raise ValueError("log_brightness must be a 2D array.")
        if support is not None:
            support = np.asarray(support, dtype=bool)
            if support.shape != eta.shape:
                raise ValueError(
                    f"support has shape {support.shape}, but log_brightness "
                    f"has shape {eta.shape}."
                )
            any_pixel = concrete(np.any(support))
            if any_pixel is not None and not bool(any_pixel):
                raise ValueError("support must contain at least one pixel.")
        self.support = support
        # Reject a log-brightness whose softmax over the support is not
        # finite (skipped under jax.jit).
        live = np.ones(eta.shape, dtype=bool) if support is None else support
        if concrete(np.any(live & (np.isnan(eta) | (eta == np.inf)))):
            raise ValueError("log_brightness must not be NaN or +inf.")
        finite = concrete(np.any(live & np.isfinite(eta)))
        if finite is not None and not bool(finite):
            raise ValueError(
                "log_brightness needs at least one finite (supported) pixel."
            )
        self.rotation_deg = float(rotation_deg)
        self._place(flux, dra, ddec)

    @classmethod
    def from_brightness(
        cls, brightness, pixel_scale_mas, *, floor=1e-6, **kwargs
    ):
        """Build an image from non-negative pixel fluxes.

        Pixels fainter than ``floor`` times the brightest are raised to that
        level, so that their logarithm is finite. Other keyword arguments
        (``support``, ``flux``, ``dra``, ``ddec``, ``rotation_deg``) go to
        [`Image`][virgil.models.Image].
        """
        brightness = np.asarray(brightness, dtype=float)
        support = kwargs.get("support")
        inside = (
            brightness
            if support is None
            else np.where(support, brightness, 0.0)
        )
        peak = np.max(inside)
        if concrete(peak) is not None and not float(peak) > 0.0:
            raise ValueError("brightness needs a positive (supported) pixel.")
        brightness = np.maximum(brightness, floor * peak)
        return cls(np.log(brightness), pixel_scale_mas, **kwargs)

    @classmethod
    def from_model(cls, model, npix, pixel_scale_mas, **kwargs):
        """Pixelise a model, e.g. a parametric fit, as a starting image.

        The pixels sample ``model`` on an ``npix`` x ``npix`` grid (rotated
        by ``rotation_deg``, if given); see
        [`from_brightness`][virgil.models.Image.from_brightness] for
        the keyword arguments.
        """
        xx, yy = image_coordinates(npix, npix * pixel_scale_mas)
        xx, yy = rotate(xx, yy, kwargs.get("rotation_deg", 0.0))
        image = _normalize_image(model._image(xx, yy, pixel_scale_mas))
        return cls.from_brightness(image, pixel_scale_mas, **kwargs)

    @property
    def eta(self):
        """The log-brightness array, evaluated if it is a field."""
        if hasattr(self.log_brightness, "evaluate"):
            return self.log_brightness.evaluate(self.pixel_scale_mas)
        return self.log_brightness

    @property
    def brightness(self):
        """Pixel fluxes: positive, unit sum, zero outside ``support``."""
        eta = self.eta
        # As a mask: some zodiax versions return leaves as floats from get().
        where = None if self.support is None else self.support.ravel() != 0
        flat = jax.nn.softmax(eta.ravel(), where=where)
        return flat.reshape(eta.shape)

    @property
    def smooth_fwhm_mas(self):
        """Deprecated alias of ``blur_fwhm_mas``."""
        return self.blur_fwhm_mas

    def smoothed(self):
        """The pixels convolved with the Gaussian of ``blur_fwhm_mas``.

        Returns ``brightness`` itself when there is no smoothing. Otherwise
        an FFT convolution on a zero-padded grid, cropped to the image, so
        flux spreading past the edges is lost; see
        ``virgil._geometry.gaussian_smooth`` for how closely
        its transform matches the analytic Gaussian taper.
        """
        if self.blur_fwhm_mas is None:
            return self.brightness
        return gaussian_smooth(
            self.brightness, self.blur_fwhm_mas / self.pixel_scale_mas
        )

    def _blur(self, u, v):
        # Unrotated baselines in wavelengths: the taper is isotropic.
        if self.blur_fwhm_mas is None:
            return 1.0
        return gaussian_taper(u * u + v * v, self.blur_fwhm_mas)

    def _centred_cvis(self, uu, vv):
        uu, vv = rotate(uu, vv, -self.rotation_deg)
        vis = image_visibilities(
            self.brightness, uu, vv, self.pixel_scale_mas, self.ft
        )
        return vis * self._blur(uu, vv)

    def model_on_grid(self, u, v, wavel, grid):
        vis = _pixel_visibilities(
            self.brightness,
            self.pixel_scale_mas,
            self.rotation_deg,
            u,
            v,
            wavel,
            grid,
            self.ft,
        )
        vis = vis * self._blur(u / wavel, v / wavel)
        return vis * offset_phase(u / wavel, v / wavel, self.dra, self.ddec)

    def _centred_image(self, xx, yy, pixel_scale_mas):
        # Bilinear resampling of the (smoothed) pixels, exact at their centres.
        xx, yy = rotate(xx, yy, -self.rotation_deg)
        brightness = self.smoothed()
        nrow, ncol = brightness.shape
        rows = 0.5 * (nrow - 1) - yy / self.pixel_scale_mas
        cols = 0.5 * (ncol - 1) - xx / self.pixel_scale_mas
        return map_coordinates(brightness, [rows, cols], order=1)


class _PixelRendered(Component):
    """A component whose visibilities are the transform of its own pixels.

    Subclasses render their brightness (``_centred_image``) on a fixed,
    centred ``npix`` x ``npix`` grid of ``pixel_scale_mas`` pixels, whose
    "up" axis is at position angle ``rotation_deg``, and set ``ft``. On data
    whose samples lie on a matching uv lattice (``data.uv_grid``) the
    transform is the exact two-sided matrix Fourier transform, as for an
    [`Image`][virgil.models.Image]; elsewhere it is ``ft``.
    ``rotation_deg="grid"`` renders the pixels in the frame of whatever
    lattice the data carry (North up without one), so that DISCO data always
    use the matrix transform.
    """

    npix: int = eqx.field(static=True)
    pixel_scale_mas: float = eqx.field(static=True)
    ft: str = eqx.field(static=True)
    rotation_deg: float | str = eqx.field(static=True)

    # deliberate: the one grid setup that pixel-rendered models share
    def _set_grid(self, npix, pixel_scale_mas, ft, rotation_deg):
        """Validate and store the pixel grid and its transform."""
        self.ft = _check_ft(ft)
        if isinstance(rotation_deg, str):
            if rotation_deg != "grid":
                raise ValueError(
                    f"rotation_deg must be a number or 'grid', not {rotation_deg!r}."
                )
            self.rotation_deg = rotation_deg
        else:
            self.rotation_deg = float(rotation_deg)
            if not onp.isfinite(self.rotation_deg):
                raise ValueError(
                    f"rotation_deg must be finite, not {self.rotation_deg}."
                )
        self.npix = int(npix)
        if self.npix <= 0 or self.npix % 2:
            raise ValueError(f"npix must be positive and even, got {npix}.")
        self.pixel_scale_mas = float(pixel_scale_mas)
        if not (
            onp.isfinite(self.pixel_scale_mas) and self.pixel_scale_mas > 0
        ):
            raise ValueError(
                f"pixel_scale_mas must be positive, got {pixel_scale_mas}."
            )

    def _frame(self, grid):
        """Position angle of the pixel grid's "up" axis for these data."""
        if self.rotation_deg != "grid":
            return self.rotation_deg
        return 0.0 if grid is None else grid.rotation_deg

    def _pixels(self, rotation_deg):
        """Unit-sum pixel fluxes, centred, in a frame at ``rotation_deg``."""
        xx, yy = image_coordinates(self.npix, self.npix * self.pixel_scale_mas)
        if rotation_deg != 0.0:
            xx, yy = rotate(xx, yy, rotation_deg)
        pixels = self._centred_image(xx, yy, self.pixel_scale_mas)
        total = np.sum(pixels)
        return pixels / np.where(total > 0.0, total, 1.0)

    def _centred_cvis(self, uu, vv):
        rotation = self._frame(None)
        if rotation != 0.0:
            uu, vv = rotate(uu, vv, -rotation)
        return image_visibilities(
            self._pixels(rotation), uu, vv, self.pixel_scale_mas, self.ft
        )

    def model_on_grid(self, u, v, wavel, grid):
        rotation = self._frame(grid)
        vis = _pixel_visibilities(
            self._pixels(rotation),
            self.pixel_scale_mas,
            rotation,
            u,
            v,
            wavel,
            grid,
            self.ft,
        )
        return vis * offset_phase(u / wavel, v / wavel, self.dra, self.ddec)


def _check_ft(ft):
    """``ft`` if it is a valid transform name."""
    if ft not in ("auto", "dft", "nufft"):
        raise ValueError(f"ft must be 'auto', 'dft' or 'nufft', not {ft!r}.")
    return ft


class FlaredDisk(_PixelRendered):
    r"""Flared, inclined scattered-light disk: a skewed Gaussian ring.

    Abstract base with a forward-scattering peak on the near side; use
    [`FlaredDiskHG`][virgil.models.FlaredDiskHG],
    [`FlaredDiskGaussian`][virgil.models.FlaredDiskGaussian] or
    [`FlaredDiskPowerLaw`][virgil.models.FlaredDiskPowerLaw]. No star or
    planets: compose them in a [`System`][virgil.models.System]. The
    visibilities are the exact Fourier transform of the brightness on a fixed
    ``npix`` x ``npix`` grid, which must cover the disk and resolve its inner
    edge ([Model notes](../notes.md#flareddisk)); it follows the empirical
    $\sin i$ surface projection of Blakely et al. (2024), whereas
    [`MultipoleDisk`][virgil.models.MultipoleDisk] with
    [`PowerLawHeight`][virgil.models.PowerLawHeight] uses the exact $\tan i$
    thin-surface projection.

    Source: Blakely et al. (2024), §III,
    [references](../../../references.md#blakely2024).

    Parameters
    ----------
    radius, fwhm, inc, pa : float or array-like
        Radius of peak brightness (before the skew moves it outwards) and
        radial FWHM, in mas; inclination, 0 (face-on) up to but not including
        90, and position angle of the projected major axis (the near side is at
        ``pa + 90``), in degrees.
    npix, pixel_scale_mas : int, float
        Pixels on a side of the grid (even, so no pixel centre is on the star),
        and their size in mas.
    ft : {"auto", "dft", "nufft"}, optional
        Transform for irregular uv points (see
        [`set_ft_defaults`][virgil._geometry.set_ft_defaults]).
    rotation_deg : "grid" or float, optional
        Position angle of the grid's "up" axis: ``"grid"`` (default) follows
        the data's uv lattice (exact two-sided matrix transform), else North
        up; a number fixes it.
    skew, aspect, flaring, symmetric : float or array-like, optional
        Inner-edge truncation (default 0), aspect ratio $z/\rho$ at ``radius``
        (default 0, flat), flaring index (1.25), axisymmetric brightness
        relative to the phase function (0).
    flux, dra, ddec : optional
        As for any [`Component`][virgil.models.Component].
    """

    radius: jax.Array
    fwhm: jax.Array
    inc: jax.Array
    pa: jax.Array
    skew: jax.Array
    aspect: jax.Array
    flaring: jax.Array
    symmetric: jax.Array

    def __init__(
        self,
        radius,
        fwhm,
        inc,
        pa,
        npix,
        pixel_scale_mas,
        skew=0.0,
        aspect=0.0,
        flaring=1.25,
        symmetric=0.0,
        flux=1.0,
        dra=0.0,
        ddec=0.0,
        ft="auto",
        rotation_deg="grid",
    ):
        self.radius = np.asarray(radius, dtype=float)
        self.fwhm = np.asarray(fwhm, dtype=float)
        self.inc = np.asarray(inc, dtype=float)
        self.pa = np.asarray(pa, dtype=float)
        self.skew = np.asarray(skew, dtype=float)
        self.aspect = np.asarray(aspect, dtype=float)
        self.flaring = np.asarray(flaring, dtype=float)
        self.symmetric = np.asarray(symmetric, dtype=float)
        self._set_grid(npix, pixel_scale_mas, ft, rotation_deg)
        self._place(flux, dra, ddec)

    def _phase_function(self, cos_theta, x, y):
        """Azimuthal brightness ``f(θ)``, with θ in radians from the near side.

        Receives ``cos θ = y / ρ`` (1 at the star) and the mid-plane
        coordinates, so that phase functions needing only ``cos θ`` avoid an
        ``arctan2``; those needing θ itself compute ``arctan2(x, y)``.
        """
        raise NotImplementedError

    def is_physical(self):
        # The ring is divided by its radius and width.
        return (
            super().is_physical()
            & (self.radius > 0.0)
            & (self.fwhm > 0.0)
            & (self.symmetric >= 0.0)
            # Edge-on (and beyond) cannot be deprojected.
            & (np.abs(self.inc) < 90.0)
        )

    def _centred_image(self, xx, yy, pixel_scale_mas):
        # Mid-plane coordinates: x along the major axis, y along the minor
        # axis (towards the near side, at pa + 90), deprojected.
        cos_inc = np.maximum(np.cos(self.inc * dtor), 1e-6)
        y, x = undo_elliptical_transf_coord(xx, yy, self.pa, cos_inc)

        # Flared surface (Eq. 2) and apparent radius (Eq. 3). The floors
        # keep gradients finite at the star, where both radii vanish.
        tiny = np.finfo(np.result_type(x, float)).tiny
        rho = np.sqrt(np.maximum(x**2 + y**2, tiny))
        z = self.aspect * self.radius * (rho / self.radius) ** self.flaring
        r2 = x**2 + (y + z * np.sin(self.inc * dtor)) ** 2 + z**2
        r = np.sqrt(np.maximum(r2, tiny))

        # Skewed Gaussian ring (Eqs. 4-5) times the azimuthal term (Eq. 9).
        sigma = self.fwhm / FWHM_PER_SIGMA
        offset = (r - self.radius) / sigma
        ring = (
            np.exp(-0.5 * offset**2)
            * 0.5
            * (1.0 + jax.scipy.special.erf(self.skew * offset / np.sqrt(2.0)))
        )
        # cos θ = y / ρ; at the star arctan2(0, 0) = 0 gives cos θ = 1.
        cos_theta = np.where((x == 0.0) & (y == 0.0), 1.0, y / rho)
        return (self._phase_function(cos_theta, x, y) + self.symmetric) * ring


class FlaredDiskHG(FlaredDisk):
    r"""[`FlaredDisk`][virgil.models.FlaredDisk] with a Henyey–Greenstein phase function.

    $f(\theta) = \dfrac{1 - g^2}{4\pi\,(1 + g^2 - 2g\cos\theta)^{3/2}}$
    (Blakely et al. 2024, Eq. 6).

    Parameters
    ----------
    g : float or array-like
        Asymmetry parameter, with ``-1 < g < 1``: 0 is isotropic, positive
        values scatter forwards (peaking on the near side) and negative
        values backwards (peaking on the far side).
    **geometry
        The parameters of [`FlaredDisk`][virgil.models.FlaredDisk].

    Examples
    --------
    >>> disk = FlaredDiskHG(
    ...     g=0.3, radius=440.0, fwhm=290.0, inc=52.0, pa=160.0,
    ...     npix=96, pixel_scale_mas=20.0, flux=0.05,
    ... )
    """

    g: jax.Array

    def __init__(self, g, **geometry):
        self.g = np.asarray(g, dtype=float)
        super().__init__(**geometry)

    def is_physical(self):
        return super().is_physical() & (np.abs(self.g) < 1.0)

    def _phase_function(self, cos_theta, x, y):
        g = self.g
        return (1.0 - g**2) / (
            4.0 * np.pi * (1.0 + g**2 - 2.0 * g * cos_theta) ** 1.5
        )


class FlaredDiskGaussian(FlaredDisk):
    r"""[`FlaredDisk`][virgil.models.FlaredDisk] with a Gaussian phase function.

    $f(\theta) = \exp\left(-\theta^2 / 2\sigma_\theta^2\right)$, with
    $\theta \in (-180°, 180°]$ (Blakely et al. 2024, Eq. 7).

    Parameters
    ----------
    sigma_theta : float or array-like
        Azimuthal width in degrees.
    **geometry
        The parameters of [`FlaredDisk`][virgil.models.FlaredDisk].
    """

    sigma_theta: jax.Array

    def __init__(self, sigma_theta, **geometry):
        self.sigma_theta = np.asarray(sigma_theta, dtype=float)
        super().__init__(**geometry)

    def is_physical(self):
        return super().is_physical() & (self.sigma_theta > 0.0)

    def _phase_function(self, cos_theta, x, y):
        theta = np.arctan2(x, y)
        return np.exp(-0.5 * (theta / (self.sigma_theta * dtor)) ** 2)


class FlaredDiskPowerLaw(FlaredDisk):
    r"""[`FlaredDisk`][virgil.models.FlaredDisk] with a power-law phase function.

    $f(\theta) = \cos^N(\theta / 2)$, with $\theta \in (-180°, 180°]$
    (Blakely et al. 2024, Eq. 8), the best-fitting form for PDS 70.

    Parameters
    ----------
    n : float or array-like
        Power $N$; larger is more concentrated towards the near side.
    **geometry
        The parameters of [`FlaredDisk`][virgil.models.FlaredDisk].
    """

    n: jax.Array

    def __init__(self, n, **geometry):
        self.n = np.asarray(n, dtype=float)
        super().__init__(**geometry)

    def is_physical(self):
        return super().is_physical() & (self.n >= 0.0)

    def _phase_function(self, cos_theta, x, y):
        # cos(θ/2)^n = ((1 + cos θ) / 2)^(n/2), clipped: cos θ rounds to
        # slightly below -1 at θ = ±π. The floor is the smallest positive
        # float, not 0, so the power's gradient stays finite for n < 2 on the
        # far-side axis (where the base is exactly 0).
        tiny = np.finfo(np.result_type(cos_theta, float)).tiny
        return np.maximum(0.5 * (1.0 + cos_theta), tiny) ** (0.5 * self.n)


# ---------------------------------------------------------------------------
# MultipoleDisk: radial profiles and a quadrature (Hankel-transform) engine
# ---------------------------------------------------------------------------

# Largest number of (sample, node) Bessel evaluations held at once; bigger
# problems are chunked over the samples with ``jax.lax.map``.
_POLAR_CHUNK_ELEMENTS = 2**24
# Half-width, in widths, of the quadrature range of a Gaussian or soft edge:
# the mass beyond it is about 2e-9.
_POLAR_TAIL_WIDTHS = 6.0
# numpy's Gauss-Laguerre weights underflow to 0 and then NaN beyond this.
_LAGUERRE_MAX_NODES = 150


def _legendre_rule(n_r):
    """Gauss-Legendre nodes and weights on [-1, 1], computed with NumPy on
    every call (not cached)."""
    x, w = onp.polynomial.legendre.leggauss(int(n_r))
    return np.asarray(x), np.asarray(w)


def _legendre_nodes(rule, lo, hi):
    """Nodes on [lo, hi] and their ``dr`` weights from a [-1, 1] rule.

    The rule arrays are inputs of the computation (never baked into the
    compiled program) and carry no gradient.
    """
    x, w = (jax.lax.stop_gradient(a) for a in rule)
    half = 0.5 * (hi - lo)
    return 0.5 * (hi + lo) + half * x, half * w


def _soft_step(x, width):
    """Smooth 0 -> 1 step, an error function of standard deviation ``width``."""
    return 0.5 * (1.0 + jax.scipy.special.erf(x / (math.sqrt(2.0) * width)))


class _RadialProfile(zdx.Base):
    """Base class of the radial profiles of a [`MultipoleDisk`][virgil.models.MultipoleDisk].

    A profile is a non-negative brightness ``I(r)`` of the radius ``r`` in
    the plane of the disk (milliarcseconds). It owns its quadrature rule:
    ``quadrature`` returns nodes ``r_k`` and weights ``w_k`` such that
    ``sum_k w_k f(r_k)`` approximates ``int I(r) f(r) r dr`` for smooth
    ``f``. The rule's reference arrays are fields of the profile, filled by
    ``with_rule(n_r)`` (which [`MultipoleDisk`][virgil.models.MultipoleDisk] calls), so
    they are traced inputs and changing a parameter never recompiles.

    A profile has no weight of its own (a lone profile is normalized by the
    disk): a part of a [`RadialSum`][virgil.models.RadialSum] carries its
    share of the light there.

    Naming, shared by every profile: ``radius`` is the radius of a feature's
    centre, ``r_in`` and ``r_out`` the edges of the support, ``sigma`` the
    standard deviation of a Gaussian width, all in milliarcseconds in the
    plane of the disk.
    """

    rule_x: Any
    rule_w: Any

    @property
    def rule(self):
        """The quadrature rule's reference nodes and weights, or ``None``."""
        if self.rule_x is None:
            return None
        return self.rule_x, self.rule_w

    def with_rule(self, n_r):
        """This profile with an ``n_r``-node quadrature rule."""
        return eqx.tree_at(
            lambda p: (p.rule_x, p.rule_w),
            self,
            self._reference_rule(int(n_r)),
            is_leaf=lambda x: x is None,
        )

    def _reference_rule(self, n_r):
        return _legendre_rule(n_r)

    def _check_rule(self):
        if self.rule is None:
            raise ValueError(
                f"{type(self).__name__} has no quadrature rule; build it "
                "into a MultipoleDisk, or call with_rule(n_r)."
            )

    def quadrature(self):
        """Nodes ``r`` (mas) and weights ``w`` including ``I(r) r dr``."""
        raise NotImplementedError

    def intensity(self, r, pixel_scale_mas):
        """``I(r)`` on a radius array, with the weight scale of ``quadrature``."""
        raise NotImplementedError

    def extent(self):
        """Concrete (inner, outer) radius of the support, for ``suggest_n_r``."""
        raise NotImplementedError

    def _n_needed(self, q_max):
        """Nodes for oscillations up to the frequency ``q_max`` (cycles/rad)."""
        lo, hi = self.extent()
        phase = onp.pi * float(q_max) * (hi - lo) * mas2rad
        return int(onp.ceil(phase)) + 12

    def is_physical(self):
        return np.asarray(True)


class ThinRing(_RadialProfile):
    r"""Infinitely thin ring of radius ``radius``,
    $I \propto \delta(r - r_0)$ with $r_0$ = ``radius``.

    !!! warning "Provisional"
        May change in any 0.4.x release without a deprecation cycle.

    Its Hankel transform is $J_0(2\pi q r_0)$, so it needs no quadrature
    (``n_r`` is irrelevant). Rendered, it is a Gaussian a pixel wide.

    Parameters
    ----------
    radius : float or array-like
        Radius in milliarcseconds, in the plane of the disk.
    """

    radius: jax.Array

    def __init__(self, radius):
        self.radius = np.asarray(radius, dtype=float)
        self.rule_x = None
        self.rule_w = None

    def with_rule(self, n_r):
        return self

    def quadrature(self):
        return np.reshape(self.radius, (1,)), np.ones(
            (1,), dtype=self.radius.dtype
        )

    def intensity(self, r, pixel_scale_mas):
        s = np.maximum(0.5 * pixel_scale_mas, 1e-9)
        norm = math.sqrt(2.0 * math.pi) * s * np.maximum(self.radius, s)
        return np.exp(-0.5 * ((r - self.radius) / s) ** 2) / norm

    def extent(self):
        radius = float(onp.max(onp.asarray(self.radius)))
        return radius, radius

    def _n_needed(self, q_max):
        return 1

    def is_physical(self):
        return super().is_physical() & np.asarray(self.radius >= 0.0)


class GaussianRing(_RadialProfile):
    r"""Gaussian ring, $I \propto \exp[-(r - r_0)^2 / 2\sigma^2]$, with
    $r \ge 0$ and $r_0$ = ``radius``.

    !!! warning "Provisional"
        May change in any 0.4.x release without a deprecation cycle.

    The quadrature runs over $r_0 \pm 6\sigma$ (clipped at the centre) with
    a Gauss-Legendre rule, which converges spectrally.

    Parameters
    ----------
    radius : float or array-like
        Radius of the peak in milliarcseconds.
    sigma : float or array-like
        Standard deviation of the radial profile in milliarcseconds (the
        FWHM is $2\sqrt{2\ln 2}\,\sigma \approx 2.355\,\sigma$).

    References
    ----------
    Gaussian rings are the basic building block of the parametric radial
    models of vis-r (Kennedy 2025, RASTI 4, rzaf004;
    https://github.com/drgmk/vis-r).
    """

    radius: jax.Array
    sigma: jax.Array

    def __init__(self, radius, sigma):
        self.radius = np.asarray(radius, dtype=float)
        self.sigma = np.asarray(sigma, dtype=float)
        self.rule_x = None
        self.rule_w = None

    def _range(self):
        half = _POLAR_TAIL_WIDTHS * self.sigma
        return np.maximum(self.radius - half, 0.0), self.radius + half

    def quadrature(self):
        self._check_rule()
        r, dr = _legendre_nodes(self.rule, *self._range())
        return r, dr * r * self.intensity(r, 0.0)

    def intensity(self, r, pixel_scale_mas):
        return np.exp(-0.5 * ((r - self.radius) / self.sigma) ** 2)

    def extent(self):
        radius, s = (
            float(onp.max(onp.asarray(a))) for a in (self.radius, self.sigma)
        )
        return max(
            radius - _POLAR_TAIL_WIDTHS * s, 0.0
        ), radius + _POLAR_TAIL_WIDTHS * s

    def is_physical(self):
        return (
            super().is_physical()
            & np.asarray(self.radius >= 0.0)
            & np.asarray(self.sigma > 0.0)
        )


class RadialPowerLaw(_RadialProfile):
    r"""Power law $I \propto r^{\rm index}$ between ``r_in`` and ``r_out``.

    !!! warning "Provisional"
        May change in any 0.4.x release without a deprecation cycle.

    With ``edge_sigma`` (a length in mas) both edges are soft error functions
    of that standard deviation; the quadrature then runs over 6 ``edge_sigma``
    beyond them, and ``n_r`` must resolve ``edge_sigma``. With
    ``edge_sigma=None`` the edges are sharp and the Gauss-Legendre rule spans
    exactly ``[r_in, r_out]``, spectrally accurate for a smooth power law
    (``index=0``, ``r_in=0`` is a uniform disk).

    Source: Kennedy (2025), vis-r, [references](../../../references.md#visr).

    Parameters
    ----------
    r_in, r_out : float or array-like
        Inner and outer radius in mas.
    index : float or array-like, optional
        Power-law index (default 0).
    edge_sigma : float, array-like or None, optional
        Standard deviation of the soft edges in mas, or ``None`` for sharp
        edges (default).
    """

    r_in: jax.Array
    r_out: jax.Array
    index: jax.Array
    edge_sigma: Any

    def __init__(self, r_in, r_out, index=0.0, edge_sigma=None):
        self.r_in = np.asarray(r_in, dtype=float)
        self.r_out = np.asarray(r_out, dtype=float)
        self.index = np.asarray(index, dtype=float)
        self.edge_sigma = (
            None if edge_sigma is None else np.asarray(edge_sigma, dtype=float)
        )
        self.rule_x = None
        self.rule_w = None

    def _range(self):
        if self.edge_sigma is None:
            return self.r_in, self.r_out
        half = _POLAR_TAIL_WIDTHS * self.edge_sigma
        return np.maximum(self.r_in - half, 0.0), self.r_out + half

    def quadrature(self):
        self._check_rule()
        r, dr = _legendre_nodes(self.rule, *self._range())
        return r, dr * r * self.intensity(r, 0.0)

    def intensity(self, r, pixel_scale_mas):
        base = np.maximum(r, 1e-9) ** self.index
        if self.edge_sigma is None:
            inside = (r >= self.r_in) & (r <= self.r_out)
            return np.where(inside, base, 0.0)
        return (
            base
            * _soft_step(r - self.r_in, self.edge_sigma)
            * _soft_step(self.r_out - r, self.edge_sigma)
        )

    def extent(self):
        lo, hi = self._range()
        return float(onp.min(onp.asarray(lo))), float(onp.max(onp.asarray(hi)))

    def is_physical(self):
        valid = (
            super().is_physical()
            & np.asarray(self.r_in >= 0.0)
            & np.asarray(self.r_out > self.r_in)
        )
        if self.edge_sigma is not None:
            valid = valid & np.asarray(self.edge_sigma > 0.0)
        return valid


class ExpTail(_RadialProfile):
    r"""Exponential tail, $\exp[-(r - r_{\rm in})/L]$ beyond ``r_in``.

    !!! warning "Provisional"
        May change in any 0.4.x release without a deprecation cycle.

    By default a Gauss-Laguerre rule integrates the whole tail to infinity, so
    no outer radius is needed; ``r_in = 0`` is an exponential disk. The rule of
    thumb $16 \cdot 2\pi q L$ nodes at the longest baseline $q$ is optimistic
    for a well-resolved tail (error in $V$ about $2 \times 10^{-4}$ at 128
    nodes, $4 \times 10^{-5}$ at the 150-node cap). For a tail resolved much
    more finely, prefer a [`RadialPowerLaw`][virgil.models.RadialPowerLaw] with
    a soft edge, or give a ``cutoff``: the light beyond $r_{\rm in} +
    \mathrm{cutoff}\, L$ is dropped (0.7 % of a thin disk's flux at 5) and a
    Gauss-Legendre rule is used. See ``MultipoleDisk.suggest_n_r`` and [Model
    notes](../notes.md#exptail).

    Parameters
    ----------
    r_in, scale_length : float or array-like
        Inner radius in mas, where the tail starts, and the e-folding length
        $L$ in mas.
    cutoff : float or None, optional
        Cut the tail at ``r_in + cutoff * scale_length`` (static; ``None``, the
        default, is the whole tail).
    """

    r_in: jax.Array
    scale_length: jax.Array
    cutoff: float | None = eqx.field(static=True)

    def __init__(self, r_in, scale_length, *, cutoff=None):
        if cutoff is not None and not float(cutoff) > 0.0:
            raise ValueError(f"cutoff must be positive or None: {cutoff!r}")
        self.r_in = np.asarray(r_in, dtype=float)
        self.scale_length = np.asarray(scale_length, dtype=float)
        self.cutoff = None if cutoff is None else float(cutoff)
        self.rule_x = None
        self.rule_w = None

    def _reference_rule(self, n_r):
        if self.cutoff is not None:
            return _legendre_rule(n_r)
        # Gauss-Laguerre nodes and weights for exp(-t) on [0, inf).
        if n_r > _LAGUERRE_MAX_NODES:
            raise ValueError(
                f"A Gauss-Laguerre rule has at most {_LAGUERRE_MAX_NODES} "
                f"nodes (its weights underflow beyond that), not {n_r}; use "
                "a RadialPowerLaw with a large r_out and a soft edge for a "
                "tail resolved that finely."
            )
        t, w = onp.polynomial.laguerre.laggauss(int(n_r))
        return np.asarray(t), np.asarray(w)

    def quadrature(self):
        self._check_rule()
        if self.cutoff is not None:
            r, dr = _legendre_nodes(
                self.rule,
                self.r_in,
                self.r_in + self.cutoff * self.scale_length,
            )
            return r, dr * r * self.intensity(r, 0.0)
        t, w = (jax.lax.stop_gradient(a) for a in self.rule)
        r = self.r_in + self.scale_length * t
        # int_{r_in}^inf e^{-(r-r_in)/L} f(r) r dr = L int e^{-t} f r dt.
        return r, self.scale_length * w * r

    def intensity(self, r, pixel_scale_mas):
        s = np.maximum(r - self.r_in, 0.0) / self.scale_length
        inside = r >= self.r_in
        if self.cutoff is not None:
            inside = inside & (s <= self.cutoff)
        return np.where(inside, np.exp(-s), 0.0)

    def extent(self):
        r_in, s = (
            float(onp.max(onp.asarray(a)))
            for a in (self.r_in, self.scale_length)
        )
        return r_in, r_in + (10.0 if self.cutoff is None else self.cutoff) * s

    def _n_needed(self, q_max):
        lo, hi = self.extent()
        if self.cutoff is not None:
            return super()._n_needed(q_max)
        a = 2.0 * onp.pi * float(q_max) * (hi - lo) / 10.0 * mas2rad
        return min(int(onp.ceil(16.0 * a)) + 16, _LAGUERRE_MAX_NODES)

    def is_physical(self):
        return (
            super().is_physical()
            & np.asarray(self.r_in >= 0.0)
            & np.asarray(self.scale_length > 0.0)
        )


class Gap(_RadialProfile):
    r"""A Gaussian gap cut in another profile.

    !!! warning "Provisional"
        May change in any 0.4.x release without a deprecation cycle.

    The brightness is that of ``base`` times $1 - d \exp[-(r - r_0)^2 / 2
    \sigma^2]$ ($r_0$ = ``radius``), so ``depth=1`` empties the gap and
    ``depth=0`` leaves ``base``. The quadrature is the base profile's: its
    ``n_r`` must resolve ``sigma``.

    Source: Kennedy (2025), vis-r, [references](../../../references.md#visr).

    Parameters
    ----------
    base : profile
        The profile the gap is cut in; it must have a brightness, so a
        [`ThinRing`][virgil.models.ThinRing] cannot be gapped.
    radius, sigma : float or array-like
        Radius of the gap's centre and its standard deviation in mas.
    depth : float or array-like, optional
        Fractional depth between 0 and 1 (default 1).
    """

    base: _RadialProfile
    radius: jax.Array
    sigma: jax.Array
    depth: jax.Array

    def __init__(self, base, radius, sigma, depth=1.0):
        if not isinstance(base, _RadialProfile) or isinstance(base, ThinRing):
            raise TypeError(
                "Gap needs a base profile with a brightness (not a "
                f"ThinRing): {base!r}"
            )
        self.base = base
        self.radius = np.asarray(radius, dtype=float)
        self.sigma = np.asarray(sigma, dtype=float)
        self.depth = np.asarray(depth, dtype=float)
        self.rule_x = None
        self.rule_w = None

    def with_rule(self, n_r):
        return eqx.tree_at(lambda p: p.base, self, self.base.with_rule(n_r))

    def _factor(self, r):
        return 1.0 - self.depth * np.exp(
            -0.5 * ((r - self.radius) / self.sigma) ** 2
        )

    def quadrature(self):
        r, w = self.base.quadrature()
        return r, w * self._factor(r)

    def intensity(self, r, pixel_scale_mas):
        return self.base.intensity(r, pixel_scale_mas) * self._factor(r)

    def extent(self):
        return self.base.extent()

    def _n_needed(self, q_max):
        lo, hi = self.extent()
        sigma = float(onp.min(onp.asarray(self.sigma)))
        resolve = int(onp.ceil(4.0 * (hi - lo) / max(sigma, 1e-12)))
        return max(self.base._n_needed(q_max), resolve)

    def is_physical(self):
        return (
            super().is_physical()
            & self.base.is_physical()
            & np.asarray(self.sigma > 0.0)
            & np.asarray((self.depth >= 0.0) & (self.depth <= 1.0))
        )


class _Term(zdx.Base):
    """A part of a [`RadialSum`][virgil.models.RadialSum]: a profile and its
    ``weight``. Other attributes are those of the profile, so a path reads
    ``disk.profile.outer.radius`` and ``disk.profile.outer.weight``."""

    profile: _RadialProfile
    weight: jax.Array

    def __init__(self, profile, weight=1.0):
        self.profile = profile
        self.weight = np.asarray(weight, dtype=float)

    def __getattr__(self, name):
        try:
            profile = object.__getattribute__(self, "profile")
        except AttributeError:
            raise AttributeError(name) from None
        return getattr(profile, name)


class RadialSum(_RadialProfile):
    """Sum of named radial profiles, mixed by their weights.

    !!! warning "Provisional"
        May change in any 0.4.x release without a deprecation cycle.

    A part is a profile, or a ``(profile, weight)`` pair (a bare profile has
    weight 1). Each part is normalized to unit total light and then weighted
    by its ``weight`` (relative to the others), so the weights are the parts'
    shares of the disk's flux, as the ``flux`` of a
    [`System`][virgil.models.System]'s components are. Keep the weights as
    parameters and fit their ratios: parts are reached by name,
    ``disk.profile.inner.radius`` and ``disk.profile.outer.weight``.
    Each part keeps its own quadrature rule, so the sum has
    ``n_r`` nodes per part.

    Examples
    --------
    >>> profile = RadialSum(
    ...     inner=GaussianRing(radius=3.0, sigma=0.5),
    ...     outer=(GaussianRing(radius=8.0, sigma=1.0), 0.5),
    ... )
    >>> disk = MultipoleDisk(profile, inc=40.0, pa=20.0, n_r=32)
    """

    names: tuple = eqx.field(static=True)
    parts: tuple

    def __init__(self, parts=None, /, **named):
        parts = {**(parts or {}), **named}
        if not parts:
            raise ValueError("RadialSum needs at least one profile.")
        terms = []
        for name, part in parts.items():
            check_part_name(
                name,
                RadialSum,
                "part",
                "inner.radius",
                "a RadialSum attribute",
            )
            profile, weight = part if isinstance(part, tuple) else (part, 1.0)
            if not isinstance(profile, _RadialProfile):
                raise TypeError(f"Part '{name}' is not a profile: {profile!r}")
            terms.append(_Term(profile, weight))
        self.names = tuple(parts)
        self.parts = tuple(terms)
        self.rule_x = None
        self.rule_w = None

    def __getattr__(self, name):
        return _named_part(self, name, "part")

    def with_rule(self, n_r):
        parts = tuple(
            eqx.tree_at(lambda t: t.profile, t, t.profile.with_rule(n_r))
            for t in self.parts
        )
        return eqx.tree_at(lambda p: p.parts, self, parts)

    def _shares(self):
        weights = np.stack([np.asarray(t.weight) for t in self.parts])
        return weights / np.sum(weights)

    def quadrature(self):
        shares = self._shares()
        rs, ws = [], []
        for share, term in zip(shares, self.parts):
            r, w = term.profile.quadrature()
            rs.append(r)
            ws.append(share * (w / np.sum(w)))
        return np.concatenate(rs), np.concatenate(ws)

    def intensity(self, r, pixel_scale_mas):
        shares = self._shares()
        total = 0.0
        for share, term in zip(shares, self.parts):
            _, w = term.profile.quadrature()
            total = total + share * term.profile.intensity(
                r, pixel_scale_mas
            ) / np.sum(w)
        return total

    def extent(self):
        extents = [t.profile.extent() for t in self.parts]
        return min(e[0] for e in extents), max(e[1] for e in extents)

    def _n_needed(self, q_max):
        return max(t.profile._n_needed(q_max) for t in self.parts)

    def is_physical(self):
        valid = super().is_physical()
        for term in self.parts:
            valid = valid & term.profile.is_physical()
            valid = valid & np.asarray(term.weight >= 0.0)
        total = sum(np.asarray(t.weight) for t in self.parts)
        return valid & (total > 0.0)


class _RadialFunction(zdx.Base):
    """A smooth function of the radius ``r`` (milliarcseconds), e.g. for a
    harmonic coefficient of [`Harmonics`][virgil.models.Harmonics]."""

    def __call__(self, r):
        raise NotImplementedError


class RadialLinear(_RadialFunction):
    """``value + slope (r - r_ref)``: a coefficient that varies linearly.

    !!! warning "Provisional"
        May change in any 0.4.x release without a deprecation cycle.

    Parameters
    ----------
    value : float or array-like
        The function at ``r = r_ref``.
    slope : float or array-like, optional
        Gradient per milliarcsecond (default 0).
    r_ref : float or array-like, optional
        Reference radius in milliarcseconds (default 0).
    """

    value: jax.Array
    slope: jax.Array
    r_ref: jax.Array

    def __init__(self, value, slope=0.0, r_ref=0.0):
        self.value = np.asarray(value, dtype=float)
        self.slope = np.asarray(slope, dtype=float)
        self.r_ref = np.asarray(r_ref, dtype=float)

    def __call__(self, r):
        return self.value + self.slope * (r - self.r_ref)


class RadialPower(_RadialFunction):
    """``value (r / r_ref) ** index``: a coefficient that follows a power law.

    !!! warning "Provisional"
        May change in any 0.4.x release without a deprecation cycle.

    Parameters
    ----------
    value : float or array-like
        The function at ``r = r_ref``.
    index : float or array-like
        Power-law index.
    r_ref : float or array-like
        Reference radius in milliarcseconds (positive).
    """

    value: jax.Array
    index: jax.Array
    r_ref: jax.Array

    def __init__(self, value, index, r_ref):
        self.value = np.asarray(value, dtype=float)
        self.index = np.asarray(index, dtype=float)
        self.r_ref = np.asarray(r_ref, dtype=float)

    def __call__(self, r):
        return self.value * np.maximum(r / self.r_ref, 1e-12) ** self.index


# deliberate: the named Matérn correlation, one branch per smoothness
def _matern(d, nu):
    """Matérn correlation at the scaled distance ``d = |r - r'| / ell``.

    ``nu`` is 0.5, 1.5, 2.5 (closed forms) or ``inf`` (the squared
    exponential).
    """
    if nu == 0.5:
        return np.exp(-d)
    if nu == 1.5:
        a = math.sqrt(3.0) * d
        return (1.0 + a) * np.exp(-a)
    if nu == 2.5:
        a = math.sqrt(5.0) * d
        return (1.0 + a + a * a / 3.0) * np.exp(-a)
    return np.exp(-0.5 * d * d)


def _log_azimuthal_spectrum(m, angle, nu):
    """Log of the periodic Matérn spectral density at integer order ``m``.

    ``log S(m) = (-nu - 1/2) log(1 + (m angle)^2 / (2 nu))`` for angular
    length ``angle`` (radians), ``-(m angle)^2 / 2`` for ``nu = inf``. In log
    space because ``S`` underflows at high orders, where the chain rule
    through ``sqrt`` gives NaN. Shared by ``MultipoleKernel`` and
    ``MultipoleField``.
    """
    x2 = (m * angle) ** 2
    if math.isinf(nu):
        return -0.5 * x2
    return (-nu - 0.5) * np.log1p(x2 / (2.0 * nu))


def _check_nu(nu, owner):
    nu = float(nu)
    if nu not in (0.5, 1.5, 2.5, math.inf):
        raise ValueError(
            f"{owner}: nu must be 0.5, 1.5, 2.5 or inf (closed-form Matérn "
            f"kernels), not {nu}."
        )
    return nu


def _gp_factor(nodes, ell, nu, jitter):
    """Cholesky factor of the Matérn correlation matrix of ``nodes``.

    A jitter on the diagonal (at least 64 machine epsilons, relative to the
    unit variance) keeps it positive definite when ``ell`` is long against
    the spacing of the nodes. Differentiable.
    """
    d = np.abs(nodes[:, None] - nodes[None, :]) / ell
    corr = _matern(d, nu)
    jitter = max(float(jitter), 64.0 * float(np.finfo(corr.dtype).eps))
    return np.linalg.cholesky(corr + jitter * np.eye(nodes.shape[-1]))


def _linear_interp(r, nodes, values):
    """Piecewise-linear interpolation onto ``r``, constant beyond the ends.

    ``nodes`` (``n``, increasing) and ``values`` (``(..., n)``); the result
    has shape ``values.shape[:-1] + r.shape``. Differentiable in ``values``
    and in the nodes.
    """
    r = np.asarray(r)
    flat = np.ravel(r)
    n = nodes.shape[-1]
    if n == 1:
        out = values[..., :1] + 0.0 * flat
    else:
        idx = np.clip(np.searchsorted(nodes, flat) - 1, 0, n - 2)
        x0, x1 = nodes[idx], nodes[idx + 1]
        dx = x1 - x0
        t = np.clip((flat - x0) / np.where(dx > 0.0, dx, 1.0), 0.0, 1.0)
        out = values[..., idx] * (1.0 - t) + values[..., idx + 1] * t
    return out.reshape(values.shape[:-1] + r.shape)


_GP_NODES = 16


class _RadialGPBase(zdx.Base):
    r"""The Gaussian process shared by [`RadialGP`][virgil.models.RadialGP] (a
    profile) and [`RadialGPFunction`][virgil.models.RadialGPFunction] (a
    function of the radius): the values at the nodes $r_k$ are $g = \mu +
    \sigma L z$, with $LL^\top$ the Matérn correlation matrix of the nodes
    (length ``ell``, smoothness ``nu``) and **whitened latents** $z \sim
    N(0, I)$, the parameters (path ``z``): a standard-normal prior on ``z``
    is the Gaussian-process prior, and optimizers and samplers see a
    unit-scale, uncorrelated coordinate (as for
    [`GaussianField`][virgil.fields.GaussianField]). The matrix is dense and
    its Cholesky factor is recomputed at every call, which costs nothing for
    the ``n`` of at most about 100 nodes used here. A jitter keeps it
    positive definite when ``ell`` is long against the node spacing. A
    subclass gives the nodes.
    """

    z: Any
    mu: jax.Array
    sigma: jax.Array
    ell: jax.Array
    r_in: Any
    r_out: Any
    nu: float = eqx.field(static=True)
    jitter: float = eqx.field(static=True)

    def _init_process(self, name, r_in, r_out, sigma, ell, mu, z, nu, jitter):
        if (r_in is None) != (r_out is None):
            raise ValueError("give both r_in and r_out, or neither.")
        if z is not None:
            z = np.asarray(z, dtype=float)
            if z.ndim != 1:
                raise ValueError(f"z must be 1-D, got shape {z.shape}.")
        self.z = z
        self.mu = np.asarray(mu, dtype=float)
        self.sigma = np.asarray(sigma, dtype=float)
        self.ell = np.asarray(ell, dtype=float)
        self.r_in = None if r_in is None else np.asarray(r_in, dtype=float)
        self.r_out = None if r_out is None else np.asarray(r_out, dtype=float)
        self.nu = _check_nu(nu, name)
        self.jitter = float(jitter)
        _check_shape_params(
            name,
            (
                ("sigma", self.sigma, lambda x: x > 0.0, "positive"),
                ("ell", self.ell, lambda x: x > 0.0, "positive"),
            ),
        )

    def with_range(self, r_in, r_out):
        """This process on ``[r_in, r_out]`` (and ``_GP_NODES`` latents) if it
        has no range or latents yet; otherwise unchanged."""
        out = self
        if out.r_in is None and r_in is not None:
            out = eqx.tree_at(
                lambda p: (p.r_in, p.r_out),
                out,
                (
                    np.asarray(r_in, dtype=float),
                    np.asarray(r_out, dtype=float),
                ),
                is_leaf=lambda x: x is None,
            )
        if out.z is None:
            out = eqx.tree_at(
                lambda p: p.z,
                out,
                np.zeros(_GP_NODES),
                is_leaf=lambda x: x is None,
            )
        return out

    def _check_range(self):
        if self.r_in is None:
            raise ValueError(
                f"{type(self).__name__} has no radial range: give r_in and "
                "r_out, or use it inside a MultipoleDisk, whose profile sets "
                "them."
            )

    def _nodes(self):
        raise NotImplementedError

    def nodes(self):
        """The nodes in milliarcseconds."""
        return self._nodes()

    def _values(self):
        """The nodes and the process ``g`` at them."""
        nodes = self._nodes()
        if self.z is None or self.z.shape[0] != nodes.shape[0]:
            raise ValueError("the latents do not match the nodes.")
        factor = _gp_factor(nodes, self.ell, self.nu, self.jitter)
        return nodes, self.mu + self.sigma * np.matmul(
            factor, self.z, precision=jax.lax.Precision.HIGHEST
        )

    def draws(self, key, n=1, r=None, positive=False):
        """``n`` draws from the prior at the radii ``r`` (the nodes by default).

        Draws $g = \\mu + \\sigma L z$ with $z \\sim N(0, I)$ at the current
        ``mu``, ``sigma``, ``ell``; with ``positive``, $\\exp g$ (the
        profile). Shape ``(n,) + shape(r)``. Use them to show the prior beside
        the posterior.
        """
        nodes = self._nodes()
        factor = _gp_factor(nodes, self.ell, self.nu, self.jitter)
        z = jax.random.normal(key, (int(n), nodes.shape[0]), nodes.dtype)
        g = self.mu + self.sigma * np.matmul(
            z, factor.T, precision=jax.lax.Precision.HIGHEST
        )
        if r is not None:
            g = _linear_interp(r, nodes, g)
        return np.exp(g) if positive else g

    def is_physical(self):
        valid = np.asarray(self.sigma > 0.0) & np.asarray(self.ell > 0.0)
        if self.r_in is not None:
            valid = (
                valid
                & np.asarray(self.r_in >= 0.0)
                & np.asarray(self.r_out > self.r_in)
            )
        if self.z is not None:
            valid = valid & np.all(np.isfinite(self.z))
        return valid


class RadialGP(_RadialGPBase, _RadialProfile):
    r"""Non-parametric radial profile: the exponential of a Gaussian process.

    !!! warning "Provisional"
        May change in any 0.4.x release without a deprecation cycle.

    $I_0(r) = \exp g(r)$. One whitened latent ``z`` per Gauss-Legendre node of
    $[r_{\rm in}, r_{\rm out}]$ (``n_r`` of them), $g = \mu + \sigma L z$,
    interpolated linearly in $r$. The visibility is normalized, so leave ``mu``
    at 0. Without the exponential it is
    [`RadialGPFunction`][virgil.models.RadialGPFunction]. Priors (``z``
    standard normal, ``sigma`` and ``ell`` log-uniform via
    [`gp_priors`][virgil.models.gp_priors]; MAP biases the scales),
    [`draws`][virgil.models.RadialGP.draws] and the meaning of ``ell``: [Model
    notes](../notes.md#radialgp).

    Parameters
    ----------
    r_in, r_out : float
        Radii in mas of the first and last node.
    sigma, ell, mu : float, optional
        Standard deviation of $g$, correlation length in mas, and mean of $g$
        (defaults 1, 1, 0).
    z : array-like, shape (n_r,), optional
        Whitened latents (default 0); set when the profile is built into a
        disk.
    nu, jitter : optional
        Matérn smoothness in {0.5, 1.5, 2.5, inf} (default 1.5; ``inf`` squared
        exponential) and relative jitter on the correlation diagonal (default
        1e-6). Both static.

    Examples
    --------
    >>> profile = RadialGP(r_in=0.5, r_out=12.0, sigma=1.5, ell=2.0)
    >>> disk = MultipoleDisk(profile, inc=30.0, n_r=24)
    >>> disk.profile.z.shape
    (24,)
    """

    def __init__(
        self,
        r_in,
        r_out,
        sigma=1.0,
        ell=1.0,
        mu=0.0,
        z=None,
        nu=1.5,
        jitter=1e-6,
    ):
        if r_in is None or r_out is None:
            raise ValueError("RadialGP needs r_in and r_out.")
        self._init_process(
            "RadialGP", r_in, r_out, sigma, ell, mu, z, nu, jitter
        )
        self.rule_x = None
        self.rule_w = None

    def with_range(self, r_in, r_out):
        """This profile: its range is always given."""
        return self

    def with_rule(self, n_r):
        """This profile with an ``n_r``-node rule, and ``n_r`` latents (zero)
        if it had none."""
        out = super().with_rule(n_r)
        if out.z is None:
            out = eqx.tree_at(
                lambda p: p.z,
                out,
                np.zeros(int(n_r)),
                is_leaf=lambda x: x is None,
            )
        elif out.z.shape[0] != int(n_r):
            raise ValueError(
                f"A RadialGP profile has one latent per quadrature node: z "
                f"has {out.z.shape[0]} entries but n_r is {n_r}."
            )
        return out

    def _nodes(self):
        """The Gauss-Legendre nodes (mas) of the quadrature rule."""
        self._check_rule()
        return _legendre_nodes(self.rule, self.r_in, self.r_out)[0]

    def quadrature(self):
        self._check_rule()
        nodes, g = self._values()
        _, dr = _legendre_nodes(self.rule, self.r_in, self.r_out)
        return nodes, dr * nodes * np.exp(g)

    def intensity(self, r, pixel_scale_mas):
        nodes, g = self._values()
        inside = (r >= self.r_in) & (r <= self.r_out)
        return np.where(inside, np.exp(_linear_interp(r, nodes, g)), 0.0)

    def extent(self):
        return float(onp.asarray(self.r_in)), float(onp.asarray(self.r_out))

    def is_physical(self):
        return _RadialGPBase.is_physical(self) & _RadialProfile.is_physical(
            self
        )


class RadialGPFunction(_RadialGPBase, _RadialFunction):
    r"""A Gaussian process in the radius, as a function (no exponential).

    !!! warning "Provisional"
        May change in any 0.4.x release without a deprecation cycle.

    For the ``spectral_index`` of a
    [`MultipoleDisk`][virgil.models.MultipoleDisk] or a coefficient of
    [`Harmonics`][virgil.models.Harmonics]; for a non-negative profile use
    [`RadialGP`][virgil.models.RadialGP]. Values at ``n`` equally spaced nodes
    are $g = \mu + \sigma L z$ with whitened latents ``z`` (a standard-normal
    prior is the GP prior), interpolated linearly (constant beyond). Priors and
    ``ell`` are as for `RadialGP`.

    Parameters
    ----------
    r_in, r_out : float or None, optional
        Radii in mas of the first and last node; both or neither (default the
        disk's profile range).
    sigma, ell, mu : float, optional
        Standard deviation of $g$, correlation length in mas, and mean of $g$
        (defaults 1, 1, 0).
    z, n : optional
        Whitened latents, shape (n,) (default 0; its size is the number of
        nodes), and the number of nodes when ``z`` is not given (default 16).
    nu, jitter : optional
        Matérn smoothness in {0.5, 1.5, 2.5, inf} (default 1.5) and relative
        jitter on the correlation diagonal (default 1e-6). Both static.

    Examples
    --------
    >>> alpha = RadialGPFunction(sigma=0.5, ell=4.0, n=12)
    >>> alpha.z.shape
    (12,)
    """

    def __init__(
        self,
        r_in=None,
        r_out=None,
        sigma=1.0,
        ell=1.0,
        mu=0.0,
        z=None,
        n=None,
        nu=1.5,
        jitter=1e-6,
    ):
        if z is not None:
            if n is not None and int(n) != np.shape(z)[0]:
                raise ValueError(f"n={n} but z has {np.shape(z)[0]} entries.")
        elif n is not None:
            if int(n) != n or n < 1:
                raise ValueError(f"n must be a positive integer: {n!r}.")
            z = np.zeros(int(n))
        self._init_process(
            "RadialGPFunction", r_in, r_out, sigma, ell, mu, z, nu, jitter
        )

    def _nodes(self):
        """``n`` equally spaced radii (mas)."""
        self._check_range()
        if self.z is None:
            raise ValueError(
                "this RadialGPFunction has no latents: give z or n, or use "
                "it inside a MultipoleDisk."
            )
        return np.linspace(self.r_in, self.r_out, self.z.shape[0])

    def __call__(self, r):
        """The process $g(r)$ (no exponential), interpolated linearly
        between the nodes (constant beyond them)."""
        nodes, g = self._values()
        return _linear_interp(r, nodes, g)


class MultipoleKernel(zdx.Base):
    r"""Matérn-kernel prior over the harmonics of a ``MultipoleDisk``.

    !!! warning "Provisional"
        May change in any 0.4.x release without a deprecation cycle.

    A Matérn kernel in radius and azimuth over the azimuthal-harmonic
    coefficients of a [`MultipoleDisk`][virgil.models.MultipoleDisk]. Pass it
    to [`Harmonics`][virgil.models.Harmonics] as ``prior=``. Each order $m = 1
    \dots M$ is an independent
    [`RadialGPFunction`][virgil.models.RadialGPFunction] with radial length
    ``ell_r``; the variance falls with order as a periodic Matérn spectral
    density of azimuthal length ``ell_theta`` ([Model
    notes](../notes.md#multipolekernel)). The mean is a uniform ring. Whitened
    latents ``z`` have shape ``(2, M, n)`` (``disk.harmonics.z``,
    standard-normal prior); log-uniform priors on the scales:
    [`gp_priors`][virgil.models.gp_priors].

    Parameters
    ----------
    ell_r, ell_theta : float
        Correlation length in the radius (mas), and in the azimuth (mas along
        the ring at ``r_ref``).
    sigma : float, optional
        Standard deviation of the first-order coefficients (default 0.3).
    r_in, r_out, r_ref : float or None, optional
        Range of the radial processes (default the disk's profile extent) and
        the radius converting ``ell_theta`` to an angle (default mid-range).
    n, nu_r, nu_theta : optional
        Number of radial nodes (default 16), and Matérn smoothness in radius
        and azimuth in {0.5, 1.5, 2.5, inf} (default 1.5). All static.

    Examples
    --------
    >>> prior = MultipoleKernel(ell_r=3.0, ell_theta=4.0, sigma=0.4)
    >>> harmonics = Harmonics(degree=2, prior=prior)
    >>> ring = GaussianRing(6.0, 1.5)
    >>> disk = MultipoleDisk(ring, harmonics=harmonics, n_r=32)
    >>> disk.harmonics.z.shape
    (2, 2, 16)
    """

    ell_r: jax.Array
    ell_theta: jax.Array
    sigma: jax.Array
    r_in: Any
    r_out: Any
    r_ref: Any
    n: int = eqx.field(static=True)
    nu_r: float = eqx.field(static=True)
    nu_theta: float = eqx.field(static=True)

    def __init__(
        self,
        ell_r,
        ell_theta,
        sigma=0.3,
        r_in=None,
        r_out=None,
        r_ref=None,
        n=_GP_NODES,
        nu_r=1.5,
        nu_theta=1.5,
    ):
        if (r_in is None) != (r_out is None):
            raise ValueError("give both r_in and r_out, or neither.")
        if int(n) != n or n < 2:
            raise ValueError(f"n must be an integer of at least 2: {n!r}.")
        self.ell_r = np.asarray(ell_r, dtype=float)
        self.ell_theta = np.asarray(ell_theta, dtype=float)
        self.sigma = np.asarray(sigma, dtype=float)
        self.r_in = None if r_in is None else np.asarray(r_in, dtype=float)
        self.r_out = None if r_out is None else np.asarray(r_out, dtype=float)
        self.r_ref = None if r_ref is None else np.asarray(r_ref, dtype=float)
        self.n = int(n)
        self.nu_r = _check_nu(nu_r, "MultipoleKernel")
        self.nu_theta = _check_nu(nu_theta, "MultipoleKernel")
        _check_shape_params(
            "MultipoleKernel",
            (
                ("ell_r", self.ell_r, lambda x: x > 0.0, "positive"),
                ("ell_theta", self.ell_theta, lambda x: x > 0.0, "positive"),
                ("sigma", self.sigma, lambda x: x > 0.0, "positive"),
            ),
        )

    def with_range(self, r_in, r_out):
        """This prior on ``[r_in, r_out]`` if it has no range; else unchanged."""
        if self.r_in is not None or r_in is None:
            return self
        return eqx.tree_at(
            lambda p: (p.r_in, p.r_out),
            self,
            (np.asarray(r_in, dtype=float), np.asarray(r_out, dtype=float)),
            is_leaf=lambda x: x is None,
        )

    def _check_range(self):
        if self.r_in is None:
            raise ValueError(
                "MultipoleKernel has no radial range: give r_in and r_out, or "
                "use it in a MultipoleDisk, whose profile sets them."
            )

    def nodes(self):
        """The radial nodes in milliarcseconds."""
        self._check_range()
        return np.linspace(self.r_in, self.r_out, self.n)

    def spectrum(self, degree):
        """``S(m) / S(1)`` for ``m = 1 .. degree``: the variance of the
        order-``m`` coefficients relative to the first order's."""
        r_ref = (
            0.5 * (self.r_in + self.r_out)
            if self.r_ref is None
            else self.r_ref
        )
        m = np.arange(1, int(degree) + 1)
        angle = self.ell_theta / r_ref
        nu = self.nu_theta
        # The ratio in log space: S(1) alone underflows for a long angle.
        return np.exp(
            _log_azimuthal_spectrum(m, angle, nu)
            - _log_azimuthal_spectrum(1.0, angle, nu)
        )

    def values(self, z, r):
        """The coefficients at the radii ``r`` from the latents ``z`` of
        shape ``(2, M, n)``: ``(re, im)`` (positive form) or ``(c, s)``
        (linear form), shape ``(2, M) + shape(r)``."""
        nodes = self.nodes()
        factor = _gp_factor(nodes, self.ell_r, self.nu_r, 1e-6)
        scale = self.sigma * np.sqrt(self.spectrum(z.shape[1]))
        g = (
            np.einsum(
                "ij,kmj->kmi", factor, z, precision=jax.lax.Precision.HIGHEST
            )
            * scale[None, :, None]
        )
        return _linear_interp(r, nodes, g)

    def draws(self, key, degree, n=1, r=None):
        """``n`` draws of the coefficients from the prior at the radii ``r``
        (the nodes by default): shape ``(n, 2, degree) + shape(r)``."""
        z = jax.random.normal(key, (int(n), 2, int(degree), self.n))
        r = self.nodes() if r is None else r
        return jax.vmap(lambda zz: self.values(zz, r))(z)

    def is_physical(self):
        valid = (
            np.asarray(self.ell_r > 0.0)
            & np.asarray(self.ell_theta > 0.0)
            & np.asarray(self.sigma > 0.0)
        )
        if self.r_in is not None:
            valid = (
                valid
                & np.asarray(self.r_in >= 0.0)
                & np.asarray(self.r_out > self.r_in)
            )
        return valid


def _harmonic_entries(values, name):
    """A tuple of coefficients: constants (0-d arrays) or radial functions."""
    if isinstance(values, _RadialFunction):
        return (values,)
    if isinstance(values, (tuple, list)):
        items = values
    else:
        items = np.atleast_1d(np.asarray(values, dtype=float))
    out = []
    for item in items:
        if isinstance(item, _RadialFunction):
            out.append(item)
        else:
            item = np.asarray(item, dtype=float)
            if item.ndim:
                raise ValueError(
                    f"each entry of {name} must be a scalar or a radial "
                    f"function of r, got an array of shape {item.shape}."
                )
            out.append(item)
    return tuple(out)


def _scalar_or_function(value, name):
    """A constant (0-d array) or a radial function of ``r``."""
    if isinstance(value, _RadialFunction):
        return value
    value = np.asarray(value, dtype=float)
    if value.ndim:
        raise ValueError(
            f"{name} must be a scalar or a radial function of r, got an "
            f"array of shape {value.shape}."
        )
    return value


class _Azimuthal(zdx.Base):
    """Base class of the azimuthal profiles of a
    [`MultipoleDisk`][virgil.models.MultipoleDisk]: a brightness
    ``I(r, theta) / I0(r)`` with azimuthal mean 1 and harmonics ``0..degree``.

    A subclass gives the linear coefficients ``coefficients(r)``, the
    ``(c, s)`` the engine needs, and the rest follows.
    """

    def coefficients(self, r):
        """``(c, s)``, each of shape ``(M, len(r))``, at the radii ``r``."""
        raise NotImplementedError

    def _needs_radius(self):
        return False

    def image_harmonics(self, r=None):
        """Image harmonics ``I_0 .. I_M`` (complex, ``I_0 = 1``) at the radii
        ``r`` (not needed for constants), shape ``(M + 1,) + shape(r)``:
        $I_m = (c_m - i s_m)/2$."""
        from . import harmonics as _h

        if r is None and self._needs_radius():
            raise ValueError(
                "some coefficients are functions of the radius: give r."
            )
        r = np.asarray(0.0 if r is None else r)
        return _h.image_from_linear(*self.coefficients(r))

    def convert(self, form="linear"):
        """The same image harmonics in another form, constants only.

        ``form`` is ``"linear"`` or ``"positive"``; see
        [`from_image_harmonics`][virgil.harmonics.from_image_harmonics] (a
        spectral factorization for ``"positive"``, which needs a
        non-negative image). Coefficients that are functions of the radius
        cannot be converted.
        """
        from . import harmonics as _h

        if getattr(self, "form", None) == form:
            return self
        if self._needs_radius():
            raise ValueError(
                "convert needs constant coefficients; evaluate "
                "image_harmonics(r) and convert node by node."
            )
        return _h.from_image_harmonics(
            onp.asarray(self.image_harmonics()), form=form
        )

    def factor(self, r, theta):
        """``I(r, theta) / I0(r)`` on ``r, theta`` (broadcast against each
        other)."""
        shape = np.broadcast_shapes(np.shape(r), np.shape(theta))
        rr = np.broadcast_to(r, shape).ravel()
        tt = np.broadcast_to(theta, shape).ravel()
        c, s = self.coefficients(rr)
        out = np.ones_like(rr)
        for m in range(1, self.degree + 1):
            out = out + c[m - 1] * np.cos(m * tt) + s[m - 1] * np.sin(m * tt)
        return out.reshape(shape)

    def min_intensity(self, r, n_theta=720):
        """The minimum of the brightness relative to the azimuthal mean
        over the radii ``r`` and ``n_theta`` azimuths (negative if the
        image goes negative)."""
        theta = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
        return np.min(self.factor(np.ravel(r)[:, None], theta[None, :]))


class Harmonics(_Azimuthal):
    r"""Azimuthal harmonics of a ``MultipoleDisk``, of exact degree ``M``.

    !!! warning "Provisional"
        May change in any 0.4.x release without a deprecation cycle.

    The brightness of a [`MultipoleDisk`][virgil.models.MultipoleDisk] is the
    profile $I_0(r)$ times an azimuthal factor of mean 1 in the disk plane
    ($\theta$ from the projected major axis towards ``pa + 90``, as for
    [`ModulatedGaussianRim`][virgil.models.ModulatedGaussianRim]), with
    harmonics ``0..M``. ``form="positive"`` (the default without ``c``/``s``)
    is $|h(e^{i\theta})|^2$ normalized, positive for all parameters;
    ``"linear"`` is $1 + \sum_m c_m(r)\cos m\theta + s_m(r)\sin m\theta$,
    positive if ``sum(sqrt(c_m**2 + s_m**2)) <= 1``
    (``MultipoleDisk.min_intensity`` reports violations). Arguments after ``c``
    and ``s`` are keyword-only; see [Model notes](../notes.md#harmonics) and
    [`virgil.harmonics`][virgil.harmonics].

    Parameters
    ----------
    c, s : float, radial function or sequence of them, optional
        Linear form: coefficients of $\cos m\theta$, $\sin m\theta$ from first
        order up; a scalar is first order only, missing orders are zero.
    re, im, w : float, radial function or sequence, optional
        Positive form: real and imaginary parts of the latents $w_j$, or
        (constants only) ``w`` as complex numbers.
    degree, form : optional
        Highest order ``M`` (static; changing it recompiles; default the number
        of coefficients); ``"positive"`` or ``"linear"`` (default ``"linear"``
        if ``c`` or ``s`` is given).
    prior : MultipoleKernel, optional
        Gaussian-process prior over every order; needs ``degree`` and none of
        ``c``, ``s``, ``re``, ``im``, ``w``. The parameter is the whitened
        latent ``z`` of shape ``(2, M, n)``.

    Examples
    --------
    >>> w = (0.4 + 0.1j, 0.2j)
    >>> harmonics = Harmonics(w=w)
    >>> disk = MultipoleDisk(GaussianRing(6.0, 1.0), harmonics=harmonics)
    """

    c: tuple
    s: tuple
    re: tuple
    im: tuple
    z: Any
    prior: Any
    degree: int = eqx.field(static=True)
    form: str = eqx.field(static=True)

    def __init__(
        self,
        c=(),
        s=(),
        *,
        degree=None,
        form=None,
        re=(),
        im=(),
        w=None,
        prior=None,
    ):
        if prior is not None:
            if not isinstance(prior, MultipoleKernel):
                raise TypeError(f"prior must be a MultipoleKernel: {prior!r}")
            if (
                any(len(_harmonic_entries(x, "x")) for x in (c, s, re, im))
                or w is not None
            ):
                raise ValueError(
                    "with a prior, the coefficients are Gaussian-process "
                    "latents (z): do not give c, s, re, im or w."
                )
            if degree is None:
                raise ValueError("a Harmonics with a prior needs its degree.")
            if isinstance(degree, bool) or int(degree) != degree or degree < 1:
                raise ValueError(
                    f"degree must be a positive integer: {degree}."
                )
            form = "positive" if form is None else form
            if form not in ("linear", "positive"):
                raise ValueError(
                    f"Harmonics form must be 'positive' or 'linear', not {form!r}."
                )
            self.c = self.s = self.re = self.im = ()
            self.degree = int(degree)
            self.form = form
            self.prior = prior
            self.z = np.zeros((2, self.degree, prior.n))
            return
        self.z = None
        self.prior = None
        c, s = _harmonic_entries(c, "c"), _harmonic_entries(s, "s")
        if w is not None:
            if re != () or im != ():
                raise ValueError("give w or re and im, not both.")
            w_value = concrete(w)
            if w_value is None:
                raise TypeError(
                    "Harmonics(w=...) takes constants; for traced latents "
                    "pass re= and im= (real and imaginary parts) instead."
                )
            w = onp.atleast_1d(onp.asarray(w_value, dtype=complex))
            if w.ndim != 1:
                raise ValueError(f"w must be 1-D, got shape {w.shape}.")
            re, im = (
                tuple(float(x) for x in w.real),
                tuple(float(x) for x in w.imag),
            )
        re, im = _harmonic_entries(re, "re"), _harmonic_entries(im, "im")
        if form is None:
            form = "linear" if (c or s) else "positive"
        if form not in ("linear", "positive"):
            raise ValueError(
                f"Harmonics form must be 'positive' or 'linear', not {form!r}."
            )
        if form == "linear" and (re or im):
            raise ValueError(
                "re, im and w belong to the positive form; the linear form "
                "takes c and s."
            )
        if form == "positive" and (c or s):
            raise ValueError(
                "c and s belong to the linear form (form='linear'); the "
                "positive form takes re and im, or w."
            )
        first, second = (c, s) if form == "linear" else (re, im)
        top = max(len(first), len(second)) if degree is None else degree
        if isinstance(top, bool) or int(top) != top or top < 0:
            raise ValueError(f"degree must be a non-negative integer: {top}.")
        top = int(top)
        if max(len(first), len(second)) > top:
            raise ValueError(
                f"degree is {top} but there are {len(first)} and "
                f"{len(second)} coefficients."
            )
        zero = np.asarray(0.0)
        first = first + (zero,) * (top - len(first))
        second = second + (zero,) * (top - len(second))
        empty = ()
        if form == "linear":
            self.c, self.s, self.re, self.im = first, second, empty, empty
        else:
            self.c, self.s, self.re, self.im = empty, empty, first, second
        self.degree = top
        self.form = form

    def _needs_radius(self):
        return self.prior is not None or any(
            isinstance(e, _RadialFunction)
            for group in (self.c, self.s, self.re, self.im)
            for e in group
        )

    def _pair(self, r):
        """The two sets of coefficients (``c, s`` or ``re, im``), each of
        shape ``(M,) + shape(r)``."""
        first, second = (
            (self.c, self.s) if self.form == "linear" else (self.re, self.im)
        )
        if self.prior is not None:
            both = self.prior.values(self.z, np.asarray(r))
            return both[0], both[1]
        return self._at(first, r), self._at(second, r)

    @staticmethod
    def _at(entries, r):
        r = np.asarray(r)
        # ``entries`` is a tuple, or an array when a path sets the whole
        # field (``disk.set("harmonics.c", array)``): test its length.
        if len(entries) == 0:
            return np.zeros((0,) + r.shape)
        return np.stack(
            [
                e(r) if isinstance(e, _RadialFunction) else e + 0.0 * r
                for e in entries
            ]
        ).reshape((len(entries),) + r.shape)

    def polynomial(self, r):
        """The minimum-phase polynomial ``h_0 .. h_M`` (``h_0 = 1``) of the
        positive form at the radii ``r``, complex, shape ``(M + 1,) +
        shape(r)``."""
        from . import harmonics as _h

        if self.form != "positive":
            raise ValueError("polynomial is defined for the positive form.")
        return _h.step_up(_h.reflection_coefficients(*self._pair(r)))

    def coefficients(self, r):
        """The linear ``(c, s)``, each of shape ``(M,) + shape(r)``, at the
        radii ``r``."""
        from . import harmonics as _h

        if self.form == "linear":
            return self._pair(r)
        return _h.linear_from_image(self.image_harmonics(r))

    def image_harmonics(self, r=None):
        """Image harmonics ``I_0 .. I_M`` (complex, ``I_0 = 1``) at the radii
        ``r`` (not needed for constants), shape ``(M + 1,) + shape(r)``.

        Linear form: $I_m = (c_m - i s_m)/2$. Positive form:
        $I_m = \\sum_j h_{j+m} \\bar h_j / \\sum_j |h_j|^2$.
        """
        from . import harmonics as _h

        if self.form == "linear":
            return super().image_harmonics(r)
        if r is None and self._needs_radius():
            raise ValueError(
                "some coefficients are functions of the radius: give r."
            )
        r = np.asarray(0.0 if r is None else r)
        return _h.image_harmonics_from_h(self.polynomial(r))

    def factor(self, r, theta):
        """``I(r, theta) / I0(r)`` on ``r, theta`` (broadcast against each
        other); for the positive form $|h|^2 / \\sum_j |h_j|^2$, exactly
        non-negative."""
        if self.form == "linear":
            return super().factor(r, theta)
        shape = np.broadcast_shapes(np.shape(r), np.shape(theta))
        rr = np.broadcast_to(r, shape).ravel()
        tt = np.broadcast_to(theta, shape).ravel()
        h = self.polynomial(rr)
        z = np.exp(1j * tt)
        poly = h[-1]
        for j in range(self.degree - 1, -1, -1):
            poly = poly * z + h[j]
        norm = np.sum(np.real(h * np.conj(h)), axis=0)
        return (np.real(poly * np.conj(poly)) / norm).reshape(shape)


class VonMises(_Azimuthal):
    r"""A von Mises azimuthal profile, truncated at ``degree``.

    !!! warning "Provisional"
        May change in any 0.4.x release without a deprecation cycle.

    The profile $\exp[\kappa\cos(\theta - \theta_0)]$ is an arc of width $\sim
    1/\sqrt\kappa$ radians at azimuth $\theta_0$ in the disk plane of a
    [`MultipoleDisk`][virgil.models.MultipoleDisk] (as for
    [`Harmonics`][virgil.models.Harmonics]); $\kappa = 0$ is a uniform ring.
    The harmonics come from a small FFT, differentiable in $\kappa$ and
    $\theta_0$. The truncated series goes slightly negative if ``degree`` is
    too low (about $5.7\sqrt\kappa$ for harmonics below $10^{-6}$);
    [`truncation_power`][virgil.models.VonMises.truncation_power] reports the
    power dropped ([Model notes](../notes.md#vonmises)).

    Parameters
    ----------
    kappa : float or radial function
        Concentration (non-negative).
    theta0 : float or radial function, optional
        Azimuth of the brightest point in degrees (default 0, the major axis).
    degree, n_fft : int, optional
        Highest order kept (default the smallest keeping the harmonics above
        $10^{-6}$ for constant ``kappa``, at most 256, else 32), and FFT points
        (default ``max(512, 8 (degree + 1))``, accurate for $\kappa$ below
        about ``n_fft**2 / 30``). Both static.

    Examples
    --------
    >>> arc = VonMises(kappa=5.0, theta0=30.0, degree=20)
    >>> ring = MultipoleDisk(GaussianRing(6.0, 1.0), harmonics=arc, inc=40.0)
    """

    kappa: Any
    theta0: Any
    degree: int = eqx.field(static=True)
    n_fft: int = eqx.field(static=True)

    def __init__(self, kappa, theta0=0.0, degree=None, n_fft=None):
        self.kappa = _scalar_or_function(kappa, "kappa")
        self.theta0 = _scalar_or_function(theta0, "theta0")
        if not isinstance(self.kappa, _RadialFunction):
            _check_shape_params(
                "VonMises",
                (("kappa", self.kappa, lambda x: x >= 0.0, "non-negative"),),
            )
        if degree is None:
            value = (
                None
                if isinstance(self.kappa, _RadialFunction)
                else concrete(self.kappa)
            )
            from . import harmonics as _h

            degree = 32 if value is None else _h.auto_von_mises_degree(value)
        if isinstance(degree, bool) or int(degree) != degree or degree < 1:
            raise ValueError(f"degree must be a positive integer: {degree}.")
        self.degree = int(degree)
        n_fft = max(512, 8 * (self.degree + 1)) if n_fft is None else n_fft
        if int(n_fft) != n_fft or n_fft < 2 * self.degree + 2:
            raise ValueError(
                f"n_fft must be an integer of at least {2 * self.degree + 2}."
            )
        self.n_fft = int(n_fft)

    def _needs_radius(self):
        return isinstance(self.kappa, _RadialFunction) or isinstance(
            self.theta0, _RadialFunction
        )

    def _evaluate(self, r):
        from . import harmonics as _h

        r = np.asarray(r)

        def value(v):
            return v(r) if isinstance(v, _RadialFunction) else v + 0.0 * r

        return _h.von_mises_image_harmonics(
            value(self.kappa),
            value(self.theta0) * dtor,
            self.degree,
            self.n_fft,
        )

    def image_harmonics(self, r=None):
        """Image harmonics ``I_0 .. I_M`` (``I_0 = 1``) of the truncated
        profile at the radii ``r`` (not needed for constants)."""
        if r is None and self._needs_radius():
            raise ValueError(
                "kappa or theta0 is a function of the radius: give r."
            )
        return self._evaluate(np.asarray(0.0 if r is None else r))[0]

    def coefficients(self, r):
        """The linear ``(c, s)``, each of shape ``(M,) + shape(r)``."""
        from . import harmonics as _h

        return _h.linear_from_image(self._evaluate(r)[0])

    def truncation_power(self, r=None):
        r"""The fraction of the power of the structure that the truncation at
        ``degree`` drops, $\sum_{m > M} |I_m|^2 / \sum_{m \ne 0} |I_m|^2$
        (0 for a degree that holds everything; shape of ``r``)."""
        if r is None and self._needs_radius():
            raise ValueError(
                "kappa or theta0 is a function of the radius: give r."
            )
        return self._evaluate(np.asarray(0.0 if r is None else r))[1]


class _Height(zdx.Base):
    """Height of a ring of radius ``r`` above the disk plane (mas)."""

    def __call__(self, r):
        raise NotImplementedError

    def is_physical(self):
        return np.asarray(True)


class Cone(_Height):
    r"""A conical surface: the ring at radius $r$ lies at the height
    $h = r \cot\alpha - \mathrm{tip}$ along the disk's normal.

    !!! warning "Provisional"
        May change in any 0.4.x release without a deprecation cycle.

    This is the geometry of [`TruncatedCone`][virgil.models.TruncatedCone]:
    the normal is the cone's axis, so the disk's inclination is
    $90^\circ - $ the axis's tilt out of the sky, and ``pa`` is the
    position angle of the *major* axis, 90 degrees from the projection of the
    axis. Positive heights shift towards position angle ``pa + 90``. In
    slant distance $s$ from the apex, $r = s \sin\alpha$.

    Parameters
    ----------
    alpha : float or array-like
        Half-opening angle in degrees (0 < alpha < 90).
    tip : float or array-like, optional
        Distance in milliarcseconds from the disk's centre back to the apex
        along the axis (the ring at $r = 0$ lies at $-\mathrm{tip}$).
    """

    alpha: jax.Array
    tip: jax.Array

    def __init__(self, alpha, tip=0.0):
        self.alpha = np.asarray(alpha, dtype=float)
        self.tip = np.asarray(tip, dtype=float)

    def __call__(self, r):
        return r / np.tan(self.alpha * dtor) - self.tip

    def is_physical(self):
        return np.all((self.alpha > 0.0) & (self.alpha < 90.0))


class PowerLawHeight(_Height):
    r"""A flared surface: $h = h_{\rm ref} (r / r_{\rm ref})^\psi$.

    !!! warning "Provisional"
        May change in any 0.4.x release without a deprecation cycle.

    Positive heights shift towards position angle ``pa + 90`` on the sky
    (see [`Cone`][virgil.models.Cone]); a negative ``h_ref`` puts the surface
    on the other side. A top surface lifted towards the observer shifts the
    apparent ring towards the far side, so when ``pa + 90`` is the near side
    of the disk it has ``h_ref < 0``.

    Parameters
    ----------
    h_ref : float or array-like
        Height at ``r_ref`` in milliarcseconds.
    r_ref : float or array-like
        Reference radius in milliarcseconds (positive).
    psi : float or array-like, optional
        Flaring index (default 1, a cone through the centre).
    """

    h_ref: jax.Array
    r_ref: jax.Array
    psi: jax.Array

    def __init__(self, h_ref, r_ref, psi=1.0):
        self.h_ref = np.asarray(h_ref, dtype=float)
        self.r_ref = np.asarray(r_ref, dtype=float)
        self.psi = np.asarray(psi, dtype=float)

    def __call__(self, r):
        return self.h_ref * np.maximum(r / self.r_ref, 1e-12) ** self.psi

    def is_physical(self):
        return np.all(self.r_ref > 0.0)


def _polar_frequencies(u, v, inc, pa, stretch=None):
    """The spatial frequency in the plane of the disk, for the engine.

    Returns ``(q, q2, phase, q_minor)``: the length ``q`` and its square,
    the direction ``e^{i phi}`` and the distance along the projected minor
    axis of each baseline ``(u, v)``.
    """
    if stretch is None:
        stretch = np.maximum(np.cos(inc * dtor), 1e-8)
    ut, vt = undo_elliptical_transf_spat_freq(u, v, pa, stretch)
    q2 = ut**2 + vt**2
    positive = q2 > 0.0
    # sqrt has an infinite derivative at 0: keep gradients finite at u = v = 0,
    # where the visibility is exactly 1.
    q_safe = np.sqrt(np.where(positive, q2, 1.0))
    q = np.where(positive, q_safe, 0.0)
    # Direction of the frequency in the disk plane, e^{i phi}, with phi from
    # the deprojected North (v) towards East (u); J_m(0) = 0 for m >= 1, so
    # its value at q = 0 is irrelevant.
    phase = np.where(positive, (vt + 1j * ut) / q_safe, 1.0 + 0j)
    # Distance along the projected minor axis (towards PA pa + 90): the
    # frequency conjugate to the lift of a ring.
    q_minor = u * np.cos(pa * dtor) - v * np.sin(pa * dtor)
    return q, q2, phase, q_minor


def multipole_disk_cvis(
    u,
    v,
    r,
    w,
    inc=0.0,
    pa=0.0,
    blur_fwhm=None,
    c=None,
    s=None,
    height=None,
    stretch=None,
):
    r"""Unit-flux visibility of a disk from radial quadrature.

    !!! warning "Provisional"
        May change in any 0.4.x release without a deprecation cycle.

    A sum of thin rings at nodes ``r`` with weights ``w``, with ``inc`` and
    ``pa`` undone, normalized so $V(0) = 1$. The Bessel series, the ``height``
    phase and the envelope are in [Model notes](notes.md#multipoledisk).

    Source: Jennings et al. (2020), [references](../../references.md#frank),
    and Kennedy (2025), [references](../../references.md#visr); ideas only.

    Parameters
    ----------
    u, v : array-like
        Baseline coordinates in wavelength units (cycles/rad); any shape.
    r, w : array-like
        Quadrature radii in mas and weights with $I_0(r_k)\,r_k\,dr$ folded in,
        shape ``(n_r,)`` (``w`` may be ``u.shape + (n_r,)`` for a chromatic
        profile); each ``w`` is normalized to unit sum.
    inc, pa, blur_fwhm : float or array-like, optional
        Inclination and position angle in degrees; FWHM in mas, in the disk
        plane, of a smoothing Gaussian.
    c, s, height, stretch : array-like or None, optional
        Linear harmonic coefficients at the nodes, shape ``(M, n_r)`` (row ``m
        - 1`` is order ``m``; keeping the brightness non-negative is the
        caller's task); ring heights above the disk plane in mas, shape
        ``(n_r,)`` (``None`` is flat); minor-axis foreshortening on the sky
        (default ``cos(inc)``; negative mirrors the azimuth).

    Returns
    -------
    array-like
        Complex visibilities with the shape of ``u``.
    """
    q, q2, phase, q_minor = _polar_frequencies(u, v, inc, pa, stretch)

    w = w / np.sum(w, axis=-1, keepdims=True)
    r_rad = r * mas2rad
    n_orders = 0 if c is None else int(np.shape(c)[0])
    lift = None
    if height is not None:
        lift = height * np.sin(inc * dtor)
    n_r = r.shape[-1]
    # Weights with a leading shape (a chromatic profile) are per sample.
    per_sample = w.ndim > 1

    def ring_sum(args):
        q_flat, z_flat, qm_flat = args[:3]
        w_ring = args[3] if per_sample else w
        arg = 2.0 * np.pi * q_flat[..., None] * r_rad
        bessel = bessel_jn(n_orders, arg)
        # Even orders feed the real part and odd orders the imaginary part
        # through (-i)^m, so only one complex product is needed at the end.
        re = bessel[0]
        im = np.zeros_like(re)
        zm = np.ones_like(z_flat)
        for m in range(1, n_orders + 1):
            zm = zm * z_flat
            amp = (
                c[m - 1] * np.real(zm)[..., None]
                + s[m - 1] * np.imag(zm)[..., None]
            ) * bessel[m]
            sign = (-1.0) ** (m // 2)
            if m % 2 == 0:
                re = re + sign * amp
            else:
                im = im - sign * amp
        ring = jax.lax.complex(re, im)
        if lift is not None:
            ring = ring * offset_phase(qm_flat[..., None], 0.0, lift, 0.0)
        return np.sum(ring * w_ring, axis=-1)

    flat = (q.ravel(), phase.ravel(), q_minor.ravel())
    if per_sample:
        flat = flat + (np.broadcast_to(w, q.shape + (n_r,)).reshape(-1, n_r),)
    if q.size * n_r * (1 + n_orders) > _POLAR_CHUNK_ELEMENTS:
        batch = max(1, _POLAR_CHUNK_ELEMENTS // (n_r * (1 + n_orders)))
        cvis = jax.lax.map(ring_sum, flat, batch_size=batch)
    else:
        cvis = ring_sum(flat)
    cvis = cvis.reshape(q.shape)

    if blur_fwhm is not None:
        fwhm_rad = blur_fwhm * mas2rad
        cvis = cvis * np.exp(
            -(np.pi**2) * fwhm_rad**2 * q2 / (4.0 * np.log(2.0))
        )
    return cvis + 0j


def _with_gp_range(obj, lo, hi):
    """``obj`` with the radial range ``[lo, hi]`` given to every
    [`RadialGPFunction`][virgil.models.RadialGPFunction] and
    [`MultipoleKernel`][virgil.models.MultipoleKernel] in it that has none."""
    gps = (RadialGPFunction, MultipoleKernel)

    def fill(x):
        if isinstance(x, gps):
            return x.with_range(lo, hi)
        return x

    return jax.tree_util.tree_map(
        fill, obj, is_leaf=lambda x: isinstance(x, gps)
    )


def _polar_basis(u, v, r, inc, pa, n_orders, height=None):
    """The data- and geometry-dependent part of ``multipole_disk_cvis``.

    Returns ``(q2, p0, cos, sin)``: ``q2`` the squared frequency in the disk
    plane, ``p0`` the order-0 matrix ``J_0(2 pi q r_k)`` (with the phase of
    the ring heights), and the lists of ``n_orders`` matrices ``cos[m - 1]``
    and ``sin[m - 1]`` that multiply the coefficients $c_m$ and $s_m$ (each of
    shape ``u.shape + (n_r,)``, complex). Then
    ``V = sum_k w_k (p0 + sum_m c_mk cos[m - 1] + s_mk sin[m - 1])`` with
    normalized weights ``w``: what ``multipole_disk_cvis`` computes.
    """
    q, q2, phase, q_minor = _polar_frequencies(u, v, inc, pa)
    arg = 2.0 * np.pi * q[..., None] * (r * mas2rad)
    bessel = bessel_jn(n_orders, arg)
    lift = 1.0
    if height is not None:
        lift = offset_phase(
            q_minor[..., None], 0.0, height * np.sin(inc * dtor), 0.0
        )
    p0 = bessel[0] * lift + 0j
    cos, sin = [], []
    zm = np.ones_like(phase)
    for m in range(1, n_orders + 1):
        zm = zm * phase
        sign = (-1.0) ** (m // 2)
        factor = sign if m % 2 == 0 else -1j * sign
        ring = factor * bessel[m] * lift
        cos.append(np.real(zm)[..., None] * ring)
        sin.append(np.imag(zm)[..., None] * ring)
    return q2, p0, tuple(cos), tuple(sin)


def _dot(matrix, x):
    """``sum_k matrix[..., k] x[..., k]``, a matrix-vector product when
    ``x`` is one vector."""
    if x.ndim == 1:
        return np.matmul(matrix, x + 0j, precision=jax.lax.Precision.HIGHEST)
    return np.sum(matrix * x, axis=-1)


class _FrozenGeometry(zdx.Base):
    """The cached matrices of ``MultipoleDisk.freeze_geometry``: the baselines,
    nodes and geometry they were built for (to check against at every call)
    and ``_polar_basis``'s matrices."""

    uu: jax.Array
    vv: jax.Array
    r: jax.Array
    inc: jax.Array
    pa: jax.Array
    height: Any
    q2: jax.Array
    p0: jax.Array
    cos: tuple
    sin: tuple
    tol: float = eqx.field(static=True)

    def matches(self, uu, vv, r, inc, pa, height):
        """Whether the cache was built for these arguments (traced boolean)."""
        if (height is None) != (self.height is None):
            return np.asarray(False)
        if np.shape(uu) != self.uu.shape or np.shape(r) != self.r.shape:
            return np.asarray(False)

        def close(a, b):
            a = np.asarray(a, dtype=b.dtype)
            # Relative to each element, with a floor of tol times the largest
            # element: an entry that is zero in one precision and a rounding
            # residue (1e-14) in the other must still match.
            scale = np.abs(b) + np.max(np.abs(b))
            return np.all((a == b) | (np.abs(a - b) <= self.tol * scale))

        ok = (
            close(uu, self.uu)
            & close(vv, self.vv)
            & close(r, self.r)
            & close(inc, self.inc)
            & close(pa, self.pa)
        )
        if height is not None:
            ok = ok & close(height, self.height)
        return ok

    def cvis(self, w, c, s, blur_fwhm):
        """The visibility from the cached matrices, for weights ``w`` (shape
        ``(n_r,)`` or ``uu.shape + (n_r,)``), coefficients ``c`` and ``s``
        (``(M, n_r)`` or ``None``) and the blur."""
        w = w / np.sum(w, axis=-1, keepdims=True)
        out = _dot(self.p0, w)
        for m in range(len(self.cos)):
            out = (
                out + _dot(self.cos[m], w * c[m]) + _dot(self.sin[m], w * s[m])
            )
        if blur_fwhm is not None:
            fwhm_rad = blur_fwhm * mas2rad
            out = out * np.exp(
                -(np.pi**2) * fwhm_rad**2 * self.q2 / (4.0 * np.log(2.0))
            )
        return out + 0j


class MultipoleDisk(Component):
    r"""Axisymmetric disk with a radial profile, evaluated by quadrature.

    !!! warning "Provisional"
        May change in any 0.4.x release without a deprecation cycle.

    A radial profile $I(r)$ in the disk plane, inclined by ``inc`` and rotated
    by ``pa``; the visibility is its Hankel transform summed over ``n_r`` nodes
    of the profile's quadrature rule. Profiles: ``ThinRing``, ``GaussianRing``,
    ``RadialPowerLaw``, ``ExpTail``, ``Gap``, ``RadialSum``,
    [`RadialGP`][virgil.models.RadialGP]; parameters are reached by path
    (``"disk.profile.radius"``). Choosing ``n_r``, symmetry and precision:
    [Model notes](../notes.md#multipoledisk),
    [guide](../../../multipole_disk_rings.md).

    Source: Jennings et al. (2020), [references](../../../references.md#frank),
    and Kennedy (2025), [references](../../../references.md#visr); ideas only.

    Parameters
    ----------
    profile : profile
        The radial profile; its quadrature rule is set to ``n_r`` nodes.
    harmonics, height : optional
        A Harmonics, VonMises or None azimuthal factor of mean one (odd orders
        make closure phases non-zero); a Cone, PowerLawHeight or None height of
        each ring above the disk plane (see the notes).
    inc, pa, blur_fwhm : float or array-like, optional
        Inclination (0 is face-on) and position angle of the projected major
        axis (North to East), in degrees; FWHM in mas, in the disk plane, of a
        smoothing Gaussian (exact envelope).
    flux, dra, ddec : optional
        As for any [`Component`][virgil.models.Component].
    n_r : int, optional
        Quadrature nodes per profile (default 64; static);
        [`suggest_n_r`][virgil.models.MultipoleDisk.suggest_n_r] gives a start.
    spectral_index, wavel0 : optional
        A float, radial function or RadialGPFunction makes the profile
        chromatic, $I_0(r)(\lambda/\lambda_0)^{\alpha(r)}$; ``flux`` is then
        the flux at ``wavel0`` (metres, required) and must be a number.

    Examples
    --------
    >>> ring = GaussianRing(radius=10.0, sigma=3.0)
    >>> profile = Gap(ring, radius=10.0, sigma=0.8)
    >>> disk = MultipoleDisk(profile, inc=50.0, pa=30.0, blur_fwhm=1.0, n_r=48)
    """

    profile: _RadialProfile
    harmonics: _Azimuthal | None
    height: _Height | None
    inc: jax.Array
    pa: jax.Array
    blur_fwhm: Any
    spectral_index: Any
    wavel0: Any
    frozen: Any
    n_r: int = eqx.field(static=True)

    def __init__(
        self,
        profile,
        harmonics=None,
        height=None,
        inc=0.0,
        pa=0.0,
        blur_fwhm=None,
        flux=1.0,
        dra=0.0,
        ddec=0.0,
        n_r=64,
        spectral_index=None,
        wavel0=None,
    ):
        if not isinstance(profile, _RadialProfile):
            raise TypeError(f"profile must be a radial profile: {profile!r}")
        if int(n_r) != n_r or n_r < 1:
            raise ValueError(f"n_r must be a positive integer, not {n_r!r}.")
        self.n_r = int(n_r)
        self.profile = profile.with_rule(self.n_r)
        try:
            support = self.profile.extent()
        except (
            TypeError,
            ValueError,
            jax.errors.TracerArrayConversionError,
            jax.errors.ConcretizationTypeError,
        ):
            support = None, None  # not concrete
        if harmonics is not None:
            harmonics = _with_gp_range(harmonics, *support)
        if spectral_index is not None:
            if isinstance(spectral_index, _RadialProfile):
                raise TypeError(
                    "spectral_index must be a number or a radial function "
                    "(such as a RadialGPFunction), not a profile: "
                    f"{spectral_index!r}"
                )
            if isinstance(flux, Spectrum):
                raise ValueError(
                    "a MultipoleDisk with a spectral_index takes a scalar flux "
                    "(its flux at wavel0): its spectrum comes from the "
                    "profile, so a spectrum from virgil.spectra is not "
                    "allowed."
                )
            if wavel0 is None:
                raise ValueError(
                    "spectral_index needs wavel0, the reference wavelength "
                    "(metres) at which flux and the profile are quoted."
                )
            spectral_index = _with_gp_range(
                _scalar_or_function(spectral_index, "spectral_index"),
                *support,
            )
        self.spectral_index = spectral_index
        self.wavel0 = (
            None if wavel0 is None else np.asarray(wavel0, dtype=float)
        )
        self.frozen = None
        if harmonics is not None and not isinstance(harmonics, _Azimuthal):
            raise TypeError(
                f"harmonics must be a Harmonics or VonMises: {harmonics!r}"
            )
        if height is not None and not isinstance(height, _Height):
            raise TypeError(
                f"height must be a Cone or PowerLawHeight: {height!r}"
            )
        self.harmonics = harmonics
        self.height = height
        self.inc = np.asarray(inc, dtype=float)
        self.pa = np.asarray(pa, dtype=float)
        self.blur_fwhm = (
            None if blur_fwhm is None else np.asarray(blur_fwhm, dtype=float)
        )
        self._place(flux, dra, ddec)

    def is_physical(self):
        valid = super().is_physical() & self.profile.is_physical()
        if self.blur_fwhm is not None:
            valid = valid & np.all(self.blur_fwhm >= 0.0)
        if self.height is not None:
            valid = valid & self.height.is_physical()
        if self.harmonics is not None and self.harmonics.degree:
            valid = valid & (self.min_intensity() >= -1e-6)
            prior = getattr(self.harmonics, "prior", None)
            if prior is not None:
                valid = valid & prior.is_physical()
        if isinstance(self.spectral_index, RadialGPFunction):
            valid = valid & self.spectral_index.is_physical()
        if self.wavel0 is not None:
            valid = valid & np.all(self.wavel0 > 0.0)
        return valid

    def __check_init__(self):
        super().__check_init__()
        if self.spectral_index is not None and isinstance(self.flux, Spectrum):
            raise ValueError(
                "a MultipoleDisk with a spectral_index takes a scalar flux (its "
                "flux at wavel0), not a spectrum."
            )

    def min_intensity(self, n_theta=720):
        """The least brightness relative to the azimuthal mean.

        The minimum over the quadrature nodes and ``n_theta`` azimuths of
        ``I(r, theta) / I0(r)``: 1 without harmonics, and negative if the
        image goes negative somewhere (possible for the linear and von Mises
        forms, never for the positive one). ``is_physical`` is false
        then.
        """
        if self.harmonics is None or not self.harmonics.degree:
            return np.asarray(1.0)
        r, w = self.profile.quadrature()
        return self.harmonics.min_intensity(r, n_theta)

    def suggest_n_r(self, data):
        """A starting number of nodes for ``data``'s longest baselines.

        Enough to sample the Bessel oscillations across the profile and the
        phase of a ``height``'s lift (at the current parameter values, which
        must be concrete), rounded up to a
        multiple of 8 (but at most the 144 nodes buildable for a Gauss-Laguerre
        tail). A starting point only: check convergence by
        doubling ``n_r``.

        Parameters
        ----------
        data : OIData
            Anything with ``u``, ``v`` (metres) and ``wavel`` (metres).
        """
        u, v, wavel = (onp.asarray(a) for a in (data.u, data.v, data.wavel))
        q_max = float(onp.max(onp.hypot(u, v) / wavel))
        n = self.profile._n_needed(q_max)
        if self.height is not None:
            # The lift h(r) sin(inc) adds a phase 2 pi q (lift) across the
            # profile, which dominates when |dh/dr| sin(inc) > 1.
            lo, hi = self.profile.extent()
            h = onp.asarray(self.height(np.linspace(lo, hi, 65)), float)
            lift = float(onp.sum(onp.abs(onp.diff(h))))
            lift *= float(
                onp.max(onp.abs(onp.sin(onp.asarray(self.inc) * dtor)))
            )
            n = max(n, int(onp.ceil(onp.pi * q_max * lift * mas2rad)) + 12)
        n = int(8 * onp.ceil(max(n, 16) / 8))
        # A Gauss-Laguerre tail (ExpTail, alone or inside a RadialSum/Gap)
        # has at most 150 nodes; step down in 8s until the rule builds.
        while n > 16:
            try:
                self.profile.with_rule(n)
                break
            except ValueError:
                n -= 8
        return n

    def _alpha(self, r):
        """The spectral index at the radii ``r``."""
        alpha = self.spectral_index
        if isinstance(alpha, _RadialFunction):
            return alpha(r)
        return alpha + 0.0 * r

    def _chromatic(self, r, w, wavel):
        """The quadrature weights at ``wavel``: ``w (wavel / wavel0)^alpha``,
        of shape ``(n_r,)`` for a scalar wavelength and ``wavel.shape +
        (n_r,)`` otherwise (``w`` itself without a spectral index)."""
        if self.spectral_index is None or wavel is None:
            return w
        ratio = np.log(np.asarray(wavel) / self.wavel0)
        return w * np.exp(self._alpha(r) * ratio[..., None])

    def _weight(self, wavel=None):
        flux = super()._weight(wavel)
        if self.spectral_index is None or wavel is None:
            return flux
        r, w = self.profile.quadrature()
        light = np.sum(self._chromatic(r, w, wavel), axis=-1) / np.sum(w)
        return flux * light

    def model(self, u, v, wavel):
        uu, vv = u / wavel, v / wavel
        return self._centred_cvis(uu, vv, wavel) * offset_phase(
            uu, vv, self.dra, self.ddec
        )

    def _centred_cvis(self, uu, vv, wavel=None):
        r, w = self.profile.quadrature()
        w = self._chromatic(r, w, wavel)
        c = s = None
        if self.harmonics is not None and self.harmonics.degree:
            c, s = self.harmonics.coefficients(r)
        height = None if self.height is None else self.height(r)

        def full():
            return multipole_disk_cvis(
                uu,
                vv,
                r,
                w,
                self.inc,
                self.pa,
                self.blur_fwhm,
                c=c,
                s=s,
                height=height,
            )

        cache = self.frozen
        n_orders = 0 if c is None else c.shape[0]
        if (
            cache is None
            or len(cache.cos) != n_orders
            or np.shape(uu) != cache.uu.shape
        ):
            return full()
        # Decide at trace time when every checked quantity is concrete (the
        # usual refit: geometry and nodes fixed, only profile parameters
        # traced). A lax.cond under vmap becomes a select that runs both
        # branches, and the Bessel matrices are what freezing avoids.
        with jax.ensure_compile_time_eval():
            r_c, _ = self.profile.quadrature()
            height_c = None if self.height is None else self.height(r_c)
            ok = cache.matches(uu, vv, r_c, self.inc, self.pa, height_c)
        if not isinstance(ok, jax.core.Tracer):
            return cache.cvis(w, c, s, self.blur_fwhm) if ok else full()
        return jax.lax.cond(
            cache.matches(uu, vv, r, self.inc, self.pa, height),
            lambda: cache.cvis(w, c, s, self.blur_fwhm),
            full,
        )

    def freeze_geometry(self, data):
        """A disk with its Bessel matrices cached for ``data``.

        With ``inc``, ``pa``, ``height`` and the nodes fixed, the transform
        matrices depend only on the baselines and nodes, so a refit of the
        profile, harmonics or spectral index is one matrix-vector product per
        call (and per order) instead of the Bessel functions of every baseline
        and node. Never automatic: call it once the geometry is settled, and
        fit the returned model. The cache checks its baselines, nodes and
        geometry at every call and silently falls back to the full computation
        (correct, not faster) for any other, so changing a frozen quantity
        loses the speed-up. Build it in the precision you evaluate in (inside
        ``jax.enable_x64(True)`` for a float64 fit; a float32 cache still
        matches a float64 fit, to float32 rounding). It holds ``(2M + 1)``
        complex matrices of shape ``(n_samples, n_r)`` (``M`` the harmonics'
        degree).

        Parameters
        ----------
        data : OIData
            Anything with ``u``, ``v`` (metres) and ``wavel`` (metres),
            converted to wavelength units inside: the samples the model will be
            evaluated at.

        Returns
        -------
        MultipoleDisk
            This disk with the cache (``frozen``); ``model`` equals the
            unfrozen disk's to rounding.
        """
        uu, vv = data.u / data.wavel, data.v / data.wavel
        r, _ = self.profile.quadrature()
        n_orders = 0
        if self.harmonics is not None and self.harmonics.degree:
            n_orders = self.harmonics.degree
        height = None if self.height is None else self.height(r)
        q2, p0, cos, sin = _polar_basis(
            uu, vv, r, self.inc, self.pa, n_orders, height
        )
        cache = _FrozenGeometry(
            uu,
            vv,
            r,
            self.inc,
            self.pa,
            height,
            q2,
            p0,
            cos,
            sin,
            32.0 * float(np.finfo(np.asarray(uu).dtype).eps),
        )
        return eqx.tree_at(
            lambda d: d.frozen, self, cache, is_leaf=lambda x: x is None
        )

    def gp_sites(self):
        """The Gaussian-process parts of the disk, by path.

        Returns a dict from the path (relative to the disk) of every
        [`RadialGP`][virgil.models.RadialGP] profile or
        [`RadialGPFunction`][virgil.models.RadialGPFunction] (a harmonics
        coefficient or the spectral index) and of the harmonics when they
        carry a [`MultipoleKernel`][virgil.models.MultipoleKernel], to that object.
        Its whitened latents are at ``<path>.z`` (``harmonics.z`` for the
        kernel), its hyperparameters at ``<path>.sigma``, ``<path>.ell``
        (``harmonics.prior.sigma``, ``.ell_r``, ``.ell_theta``).
        """
        sites = {}

        def visit(obj, path):
            if isinstance(obj, _RadialGPBase):
                sites[path] = obj
            elif isinstance(obj, RadialSum):
                for name, term in zip(obj.names, obj.parts):
                    visit(term.profile, f"{path}.{name}")
            elif isinstance(obj, Gap):
                visit(obj.base, f"{path}.base")
            elif isinstance(obj, (tuple, list)):
                for i, item in enumerate(obj):
                    visit(item, f"{path}.{i}")
            elif isinstance(obj, VonMises):
                visit(obj.kappa, f"{path}.kappa")
                visit(obj.theta0, f"{path}.theta0")

        visit(self.profile, "profile")
        if self.harmonics is not None:
            if getattr(self.harmonics, "prior", None) is not None:
                sites["harmonics"] = self.harmonics
            elif isinstance(self.harmonics, Harmonics):
                for name in ("c", "s", "re", "im"):
                    visit(getattr(self.harmonics, name), f"harmonics.{name}")
            else:
                visit(self.harmonics, "harmonics")
        if isinstance(self.spectral_index, RadialGPFunction):
            sites["spectral_index"] = self.spectral_index
        return sites

    def _centred_image(self, xx, yy, pixel_scale_mas):
        stretch = np.maximum(np.cos(self.inc * dtor), 1e-8)
        if self.height is not None:
            # A disk with a height: splat thin rings. The ring at each radius
            # of a fine grid is a set of points round it, lifted and
            # projected, each a Gaussian spot a pixel wide, so overlapping
            # parts of the surface add (optically thin). Needs concrete
            # parameters, for the number of points.
            lo, hi = self.profile.extent()
            pad = 3.0 * pixel_scale_mas
            lo, hi = max(lo - pad, 0.0), hi + pad
            n_rad = int(
                min(max(onp.ceil(3.0 * (hi - lo) / pixel_scale_mas), 64), 768)
            )
            n_theta = int(
                min(
                    max(onp.ceil(6.0 * onp.pi * hi / pixel_scale_mas), 128),
                    1536,
                )
            )
            dr = (hi - lo) / n_rad
            radii = lo + (np.arange(n_rad) + 0.5) * dr
            theta = (np.arange(n_theta) + 0.5) * (2.0 * np.pi / n_theta)
            brightness = (
                self.profile.intensity(radii, pixel_scale_mas) * radii * dr
            )
            lift = self.height(radii) * np.sin(self.inc * dtor)
            if self.harmonics is not None and self.harmonics.degree:
                factor = self.harmonics.factor(radii[:, None], theta[None, :])
            else:
                factor = np.ones((n_rad, n_theta))
            weights = brightness[:, None] * factor / n_theta
            # Unit vectors of the projected minor axis (towards PA pa + 90).
            minor_x, minor_y = np.cos(self.pa * dtor), -np.sin(self.pa * dtor)

            def ring_points(radius, height):
                x, y = apply_elliptical_transf_coord(
                    radius * np.sin(theta),
                    radius * np.cos(theta),
                    self.pa,
                    stretch,
                )
                return x + height * minor_x, y + height * minor_y

            image = _splat_rings(
                xx, yy, pixel_scale_mas, (radii, lift), ring_points, weights
            )
        else:
            xx_ell, yy_ell = undo_elliptical_transf_coord(
                xx, yy, self.pa, stretch
            )
            r_ell = safe_radius(xx_ell, yy_ell)
            image = self.profile.intensity(r_ell, pixel_scale_mas / stretch)
            if self.harmonics is not None and self.harmonics.degree:
                # The azimuth runs from the deprojected North towards East.
                theta = safe_azimuth(xx_ell, yy_ell)
                image = image * self.harmonics.factor(r_ell, theta)
        if self.blur_fwhm is None:
            return image
        return _blur_in_plane(
            image, self.blur_fwhm, self.pa, stretch, pixel_scale_mas
        )


def _multipole_disks(model, prefix=""):
    """``(path, disk)`` for every MultipoleDisk in ``model`` (path prefix ``""``
    for a bare disk, else ``"name."``-joined)."""
    if isinstance(model, MultipoleDisk):
        yield prefix, model
    elif isinstance(model, System):
        for name, part in model.components.items():
            yield from _multipole_disks(part, f"{prefix}{name}.")
    elif isinstance(model, Rotated):
        yield from _multipole_disks(model.source, f"{prefix}source.")
    elif isinstance(model, Attached):
        yield from _multipole_disks(model.component, f"{prefix}component.")


def gp_priors(model, sigma=None, ell=None, ell_theta=None):
    """Priors for a scene's Gaussian-process parameters, for ``fit``.

    !!! warning "Provisional"
        May change in any 0.4.x release without a deprecation cycle.

    For the Gaussian-process parts of a scene's
    [`MultipoleDisk`][virgil.models.MultipoleDisk]s, for ``fit`` and
    ``numpyro_model``. Returns a dict from parameter path to prior for every
    [`RadialGP`][virgil.models.RadialGP] profile,
    [`RadialGPFunction`][virgil.models.RadialGPFunction] (a harmonics
    coefficient or the spectral index) and
    [`MultipoleKernel`][virgil.models.MultipoleKernel] in ``model`` (a disk, a
    [`System`][virgil.models.System] or a rotated source). The whitened latents
    ``z`` get independent standard-normal priors (the GP prior itself).
    ``sigma`` (process standard deviations, kernel amplitude), ``ell`` (radial
    correlation lengths) and ``ell_theta`` (the kernel's azimuthal length),
    given as bounds ``(low, high)``, get log-uniform priors, so the prior and
    the Laplace evidence are proper; left ``None`` the hyperparameters are
    fixed at the model's values and not in the dict. Add priors for the other
    free parameters yourself.

    Examples
    --------
    >>> import numpyro.distributions as dist
    >>> disk = MultipoleDisk(RadialGP(0.5, 12.0), n_r=24)
    >>> scene = System(star=PointSource(), disk=disk)
    >>> priors = gp_priors(scene, sigma=(0.1, 3.0), ell=(0.5, 8.0))
    >>> sorted(priors)
    ['disk.profile.ell', 'disk.profile.sigma', 'disk.profile.z']
    """
    import numpyro.distributions as dist

    def scale(bounds, name):
        low, high = (float(b) for b in bounds)
        if not 0.0 < low < high:
            raise ValueError(
                f"{name} bounds must be 0 < low < high: {bounds}."
            )
        return dist.LogUniform(low, high)

    priors = {}
    for prefix, disk in _multipole_disks(model):
        for rel, obj in disk.gp_sites().items():
            path = f"{prefix}{rel}"
            latent = dist.Normal(np.zeros(obj.z.shape), 1.0).to_event(
                obj.z.ndim
            )
            priors[f"{path}.z"] = latent
            if isinstance(obj, Harmonics):
                names = {
                    "sigma": (sigma, "sigma"),
                    "ell_r": (ell, "ell"),
                    "ell_theta": (ell_theta, "ell_theta"),
                }
                path = f"{path}.prior"
            else:
                names = {"sigma": (sigma, "sigma"), "ell": (ell, "ell")}
            for field, (bounds, label) in names.items():
                if bounds is not None:
                    priors[f"{path}.{field}"] = scale(bounds, label)
    if not priors:
        raise ValueError("The model contains no RadialGP or MultipoleKernel.")
    return priors


class Resolved(SourceModel):
    """Fully resolved (over-resolved) flux, e.g. a large, diffuse envelope.

    Its visibility is 0 on every non-zero baseline, so inside a
    [`System`][virgil.models.System] it only adds to the normalization,
    lowering every other component's visibility by the same factor. It cannot
    be rendered on its own, and a rendered System shows only its other
    components, renormalized to unit sum, so the transform of that image is
    ``model()`` divided by the unresolved flux fraction.

    Parameters
    ----------
    flux : float, array-like or Spectrum, optional
        Weight relative to the other components of a
        [`System`][virgil.models.System] (default 1), or a spectrum from
        [`virgil.spectra`][virgil.spectra].

    Examples
    --------
    >>> scene = System(star=PointSource(), background=Resolved(flux=0.1))
    """

    flux: jax.Array

    def __init__(self, flux=1.0):
        self.flux = _as_flux(flux)

    def model(self, u, v, wavel):
        at_origin = (np.asarray(u) == 0) & (np.asarray(v) == 0)
        return np.where(at_origin, 1.0, 0.0) + 0j

    def render(self, npix=256, fov_mas=200.0):
        raise ValueError(
            "A Resolved component has no image: its light is spread far "
            "beyond any field of view. Render the System it belongs to, which "
            "shows the other components."
        )

    def _image(self, xx, yy, pixel_scale_mas):
        # Contributes nothing inside a rendered System (see Notes).
        return np.zeros_like(xx)

    def _weight(self, wavel=None):
        return flux_at(self.flux, wavel)

    def is_physical(self):
        return _flux_is_non_negative(self.flux)

    def __check_init__(self):
        _check_non_negative_flux(self.flux, "Resolved")


class System(SourceModel):
    r"""Flux-weighted mixture of named source models.

    A scene with several parts, e.g. a star with a disk. The visibility is $V =
    \sum_i f_i V_i / \sum_i f_i$ with ``f_i`` each component's ``flux``; it is
    1 at zero baseline, so only flux *ratios* are measurable: keep one
    reference (usually the star) at ``flux=1``. Components are reached by name,
    as attributes and as zodiax paths (``system.get("comp.flux")``; ``set``
    returns a new System). A System can be a component: ``flux`` is then the
    group's weight and ``dra``/``ddec`` move the group. Fluxes must be
    non-negative and not all zero, checked only for concrete values at build
    (traced code relies on priors and grid axes to keep them valid). See
    [Composing models](../../../composition.md).

    Parameters
    ----------
    components : dict[str, SourceModel], optional
        Components by name; keyword arguments are usually clearer.
    flux, dra, ddec : float or array-like, optional
        Weight of the whole system when nested (default 1; no effect at top
        level) and its offset in mas (positive to the East and North).
    **named : SourceModel
        Components as keywords, e.g. ``star=PointSource()``; names must be
        valid identifiers, not start with ``_`` and not clash with a System
        attribute.

    Examples
    --------
    >>> comp = PointSource(dra=45.0, ddec=30.0, flux=0.01)
    >>> binary = System(star=PointSource(), comp=comp)
    >>> binary.comp
    PointSource(flux=0.01, dra=45, ddec=30)
    """

    names: tuple = eqx.field(static=True)
    parts: tuple
    flux: jax.Array
    dra: jax.Array
    ddec: jax.Array

    def __init__(
        self, components=None, /, *, flux=1.0, dra=0.0, ddec=0.0, **named
    ):
        components = {**(components or {}), **named}
        if not components:
            raise ValueError("System needs at least one component.")
        for name, component in components.items():
            check_part_name(
                name,
                System,
                "Component",
                "comp.flux",
                "a System attribute or method",
            )
            if not isinstance(component, SourceModel):
                raise TypeError(
                    f"Component '{name}' is not a SourceModel: {component!r}"
                )
        weights = [concrete(c._weight()) for c in components.values()]
        # None if any weight is traced, so the zero-light check is skipped
        total = (
            None
            if any(w is None for w in weights)
            else float(sum(onp.sum(w) for w in weights))
        )
        if total == 0.0:
            raise ValueError(
                "The component fluxes sum to zero, so the system has no "
                "light. Keep a reference component (usually the star) at "
                "flux=1."
            )
        self.names = tuple(components)
        self.parts = tuple(components.values())
        self._place(flux, dra, ddec)

    @property
    def components(self):
        """The components as a ``{name: model}`` dictionary, in order."""
        return dict(zip(self.names, self.parts))

    def __getattr__(self, name):
        return _named_part(self, name, "component")

    def __repr__(self):
        lines = [
            f"{name}={component!r},"
            for name, component in self.components.items()
        ]
        lines += [
            f"{field}={_format_leaf(getattr(self, field))},"
            for field in ("flux", "dra", "ddec")
        ]
        body = textwrap.indent("\n".join(lines), "    ")
        return f"System(\n{body}\n)"

    def model(self, u, v, wavel):
        return self._mix(u, v, wavel, lambda c: c.model(u, v, wavel))

    @property
    def time_dependent(self):
        return any(part.time_dependent for part in self.parts)

    @property
    def broadcast_in_time(self):
        # Static parts do not veto it: ``at`` leaves them as they are.
        return all(
            part.broadcast_in_time
            for part in self.parts
            if part.time_dependent
        )

    def at(self, mjd, t_ref=0.0):
        if not self.time_dependent:
            return self
        parts = tuple(part.at(mjd, t_ref) for part in self.parts)
        return eqx.tree_at(lambda s: s.parts, self, parts)

    def model_on_grid(self, u, v, wavel, grid):
        return self._mix(
            u, v, wavel, lambda c: c.model_on_grid(u, v, wavel, grid)
        )

    # deliberate: resolves MultipoleField sibling geometry in the Image grid frame
    def _tied(self, part):
        """``part`` with its field's geometry taken from the sibling it names."""
        field = part.log_brightness if isinstance(part, Image) else None
        name = getattr(field, "geometry_from", None)
        if name is None:
            return part
        if name not in self.names:
            raise ValueError(
                f"The field of this Image takes its geometry from "
                f"{name!r}, which is not a component of the System "
                f"{list(self.names)}: tied components must be siblings."
            )
        disk = self.parts[self.names.index(name)]
        for attr in ("inc", "pa", "dra", "ddec"):
            if not hasattr(disk, attr):
                raise ValueError(
                    f"Component {name!r} ({type(disk).__name__}) has no "
                    f"{attr!r} for a MultipoleField to follow."
                )
        # The disk's sky position and angle in the Image's grid frame: the
        # field's centre is relative to the Image's, along its rotated axes.
        x, y = rotate(
            disk.dra - part.dra, disk.ddec - part.ddec, -part.rotation_deg
        )
        field = field.with_geometry(
            disk.inc, disk.pa - part.rotation_deg, x, y
        )
        return eqx.tree_at(lambda image: image.log_brightness, part, field)

    def resolved(self):
        parts = tuple(self._tied(p.resolved()) for p in self.parts)
        for name, part in zip(self.names, parts):
            fields = jax.tree.leaves(
                part, is_leaf=lambda x: hasattr(x, "geometry_from")
            )
            if any(getattr(f, "geometry_from", None) for f in fields):
                raise ValueError(
                    f"Component {name!r} holds a MultipoleField with "
                    f"geometry_from set that cannot be resolved: a tie is "
                    f"only followed for an Image that is a direct component "
                    f"of the System (not inside Rotated or Attached). Give "
                    f"the field its own geometry (geometry_from=None) there."
                )
        if all(a is b for a, b in zip(parts, self.parts)):
            return self
        return eqx.tree_at(lambda s: s.parts, self, parts)

    def _mix(self, u, v, wavel, part_model):
        """Flux-weighted mean of ``part_model(part)``, then the offset."""
        parts = self.resolved().parts
        weights = [c._weight(wavel) for c in parts]
        total = sum(w * part_model(c) for w, c in zip(weights, parts)) / sum(
            weights
        )
        uu, vv = u / wavel, v / wavel
        return total * offset_phase(uu, vv, self.dra, self.ddec)

    def _image(self, xx, yy, pixel_scale_mas):
        xx, yy = xx - self.dra, yy - self.ddec
        parts = self.resolved().parts
        weights = [c._weight() for c in parts]
        images = [c._image(xx, yy, pixel_scale_mas) for c in parts]
        sums = [np.sum(im) for im in images]
        # Components off the grid (zero sum) stay zero rather than become NaN.
        return sum(
            w * np.where(t > 0.0, im / np.where(t > 0.0, t, 1.0), 0.0)
            for w, im, t in zip(weights, images, sums)
        ) / sum(weights)

    def _weight(self, wavel=None):
        return flux_at(self.flux, wavel)

    def total_spectrum(self, wavel):
        wavel = np.asarray(wavel)
        return sum(
            np.broadcast_to(c._weight(wavel), wavel.shape) for c in self.parts
        )

    def is_physical(self):
        valid = _flux_is_non_negative(self.flux)
        for part in self.parts:
            valid = valid & part.is_physical()
        # All parts are non-negative when valid, so a zero sum means no light.
        total = sum(np.sum(c._weight()) for c in self.parts)
        return valid & (total > 0.0)

    def __check_init__(self):
        _check_non_negative_flux(self.flux, "System")


def smeared_model(model, u, v, wavel, bandwidth):
    r"""Complex visibilities with bandwidth smearing (boxcar channels).

    A channel of full width $\Delta\lambda$ multiplies each component's offset
    phasor by the real envelope
    $\mathrm{sinc}(\pi\,\Delta\lambda\,(u\,\mathrm{dra} +
    v\,\mathrm{ddec})/\lambda^2)$ ($u$, $v$ in metres, offsets in radians), per
    component of a [`System`][virgil.models.System] before the weights are
    applied (offsets add down nested systems). First order: weights and shapes
    are taken at the channel centre and an extended component is smeared as if
    at its centre offset. Supported: ``System`` trees of
    [`Component`][virgil.models.Component] shapes,
    [`Resolved`][virgil.models.Resolved], the binary models (smeared as their
    ``to_system()``) and [`Rotated`][virgil.models.Rotated] scenes of these
    (the baselines are rotated). Others, including ``Image`` and an
    [`OrbitalBinary`][virgil.models.OrbitalBinary] (smear its ``at(mjd)``
    snapshot, as `OIData` does), raise ``TypeError`` naming the model.
    [`OIData`][virgil.oidata.OIData] calls this when its ``bandwidth`` is set
    (read from ``EFF_BAND``, or by
    [`with_bandwidth`][virgil.oidata.OIData.with_bandwidth]). See [Model
    notes](../notes.md#bandwidth-smearing).

    Parameters
    ----------
    model : System or Component
    u, v, wavel : array-like
        Baselines and wavelengths in metres, as for ``model.model``.
    bandwidth : array-like
        Full channel width in metres, broadcast against ``u``.
    """
    return _smeared(model, u, v, wavel, bandwidth, 0.0, 0.0)


def _smeared(model, u, v, wavel, bandwidth, ra0, dec0):
    if isinstance(model, System):
        ra, dec = ra0 + model.dra, dec0 + model.ddec
        weights = [c._weight(wavel) for c in model.parts]
        total = sum(
            w * _smeared(c, u, v, wavel, bandwidth, ra, dec)
            for w, c in zip(weights, model.parts)
        ) / sum(weights)
        return total * offset_phase(
            u / wavel, v / wavel, model.dra, model.ddec
        )
    if isinstance(model, Component) and not isinstance(model, Image):
        ra, dec = ra0 + model.dra, dec0 + model.ddec
        arg = bandwidth * mas2rad * (u * ra + v * dec) / wavel**2
        return model.model(u, v, wavel) * np.sinc(arg)
    if isinstance(model, Resolved):
        # Zero on every non-zero baseline and at no offset: nothing to smear.
        return model.model(u, v, wavel)
    if isinstance(model, BinaryModelAngular):
        model = model.to_cartesian()
    if isinstance(model, BinaryModelCartesian):
        return _smeared(model.to_system(), u, v, wavel, bandwidth, ra0, dec0)
    if isinstance(model, Rotated):
        # The inherited offset is in the outer frame: rotate it with (u, v),
        # which keeps u . offset invariant.
        u, v = rotate(u, v, -model.rotation_deg)
        ra0, dec0 = rotate(ra0, dec0, -model.rotation_deg)
        return _smeared(model.source, u, v, wavel, bandwidth, ra0, dec0)
    if isinstance(model, OrbitalBinary):
        raise TypeError(
            "OrbitalBinary changes with time: smear a snapshot, "
            "model.at(mjd) (OIData.model does this per sample)."
        )
    raise TypeError(
        "Bandwidth smearing supports Systems of Component shapes, Resolved, "
        "BinaryModelCartesian, BinaryModelAngular and Rotated of these (not "
        "Image, whose flux is not at its offset); got "
        f"{type(model).__name__}. Remove the bandwidth (with_bandwidth(None))"
        " to evaluate it unsmeared."
    )


class Rotated(SourceModel):
    """A model rotated on the sky about the phase centre.

    The rotation is by ``rotation_deg`` from North towards East, and the angle
    is an ordinary (fittable) parameter, unlike
    [`Image`][virgil.models.Image]'s ``rotation_deg``, which only orients its
    pixel grid. Use it for a scene seen at several epochs, e.g. a spiral
    rotating between them (see [`fit`][virgil.fitting.fit] with a model per
    dataset).

    Parameters
    ----------
    source : SourceModel
        The model to rotate.
    rotation_deg : float
        Position angle of the rotation, North towards East, in degrees.

    Examples
    --------
    >>> north = System(a=GaussianDisk(1.0), b=GaussianDisk(1.0, ddec=10.0))
    >>> east = System(a=GaussianDisk(1.0), b=GaussianDisk(1.0, dra=10.0))
    >>> u, v = np.array([3.0, 5.0]), np.array([1.0, -2.0])
    >>> rotated = Rotated(north, 90.0).model(u, v, 1e-6)
    >>> bool(np.allclose(rotated, east.model(u, v, 1e-6), atol=1e-6))
    True
    """

    source: SourceModel
    rotation_deg: jax.Array

    def __init__(self, source, rotation_deg):
        self.source = source
        self.rotation_deg = np.asarray(rotation_deg, dtype=float)

    def model(self, u, v, wavel):
        u, v = rotate(u, v, -self.rotation_deg)
        return self.source.model(u, v, wavel)

    def resolved(self):
        source = self.source.resolved()
        if source is self.source:
            return self
        return eqx.tree_at(lambda r: r.source, self, source)

    def _image(self, xx, yy, pixel_scale_mas):
        xx, yy = rotate(xx, yy, -self.rotation_deg)
        return self.source._image(xx, yy, pixel_scale_mas)

    def _weight(self, wavel=None):
        return self.source._weight(wavel)

    def total_spectrum(self, wavel):
        return self.source.total_spectrum(wavel)

    def is_physical(self):
        return self.source.is_physical()

    @property
    def time_dependent(self):
        return self.source.time_dependent

    @property
    def broadcast_in_time(self):
        return self.source.broadcast_in_time

    def at(self, mjd, t_ref=0.0):
        if not self.time_dependent:
            return self
        return eqx.tree_at(
            lambda r: r.source, self, self.source.at(mjd, t_ref)
        )


_FRAME_ANGLES = (
    "line_pa",
    "towards_primary",
    "line_tilt",
    "node_pa",
    "inc",
    "apparent_inc",
)


class Attached(SourceModel):
    """A component placed and oriented in the frame of a binary's orbit.

    At each time the component moves to its anchor on the orbit and its angle
    attributes are set from angles of the binary frame
    ([`KeplerOrbit.frame`][virgil.orbits.KeplerOrbit.frame]). A companion is
    ``Attached(PointSource(flux), orbit)``; a disc around it in the orbital
    plane, brighter towards the primary, is
    ``Attached(ModulatedGaussianRim(...), orbit, bind={"pa": "node_pa", "inc":
    "apparent_inc", "az_pas": "towards_primary"})``. Time-dependent, so
    evaluated through [`at`][virgil.models.SourceModel.at];
    [`OIData.model`][virgil.oidata.OIData.model] uses each sample's time. Share
    one orbit between components via shared parameters in a model function
    ([`fit`][virgil.fitting.fit]).

    Parameters
    ----------
    component : Component
        The component (its ``dra``, ``ddec`` and bound angles are set).
    orbit : KeplerOrbit
        The orbit of the secondary about the primary (the scene's origin).
    anchor : {"secondary", "primary"} or float, optional
        Where the component sits: the secondary (default), the primary, or this
        fraction of the way from primary to secondary (``q / (1 + q)`` for the
        barycentre).
    bind : dict, optional
        ``{attribute: frame angle}`` from ``line_pa``, ``towards_primary``,
        ``line_tilt``, ``node_pa``, ``inc``, ``apparent_inc``. An ``az_pas``
        binding is a sky position angle, converted to the rim angle after
        ``pa`` and ``inc`` are set.
    offsets : dict, optional
        ``{attribute: degrees}`` added to the bound angles (fittable); default
        zero.
    """

    component: SourceModel
    orbit: Any
    offsets: dict
    bind: tuple = eqx.field(static=True)
    anchor: Any = eqx.field(static=True)

    def __init__(
        self, component, orbit, anchor="secondary", bind=None, offsets=None
    ):
        bind = dict(bind or {})
        for attr, angle in bind.items():
            if not hasattr(component, attr):
                raise ValueError(
                    f"{type(component).__name__} has no attribute {attr!r} "
                    "to bind."
                )
            if angle not in _FRAME_ANGLES:
                raise ValueError(
                    f"Unknown frame angle {angle!r}; use one of "
                    f"{', '.join(_FRAME_ANGLES)}."
                )
        if "az_pas" in bind and (
            not hasattr(component, "_ring_angle")
            or component.az_pas is None
            or onp.size(component.az_pas) == 0
        ):
            raise ValueError(
                f"{type(component).__name__} has no azimuthal modulation to "
                "bind az_pas to; give it az_amps and az_pas when building it."
            )
        offsets = dict(offsets or {})
        unknown = set(offsets) - set(bind)
        if unknown:
            raise ValueError(
                f"Offsets for unbound attributes: {sorted(unknown)}."
            )
        if anchor not in ("primary", "secondary"):
            anchor = float(anchor)
        self.component = component
        self.orbit = orbit
        self.anchor = anchor
        # az_pas last: it is deprojected with the bound orientation.
        self.bind = tuple(sorted(bind.items(), key=lambda b: b[0] == "az_pas"))
        self.offsets = {
            attr: np.asarray(offsets.get(attr, 0.0), dtype=float)
            for attr in bind
        }

    @property
    def time_dependent(self):
        return True

    @property
    def broadcast_in_time(self):
        return not self.bind

    def at(self, mjd, t_ref=0.0):
        dt = _days_since(mjd, self.orbit.t_ref - t_ref)
        _warn_if_mjd_without_t_ref(self.orbit.t_ref, dt)
        dra, ddec, _ = self.orbit._relative(dt)
        fraction = {"primary": 0.0, "secondary": 1.0}.get(
            self.anchor, self.anchor
        )
        out = eqx.tree_at(
            lambda c: (c.dra, c.ddec),
            self.component,
            (fraction * dra, fraction * ddec),
        )
        frame = self.orbit._frame(dt)
        for attr, angle in self.bind:
            value = frame[angle] + self.offsets[attr]
            if attr == "az_pas":
                value = out._ring_angle(value)
            old = getattr(out, attr)
            out = eqx.tree_at(
                lambda c: getattr(c, attr),
                out,
                np.broadcast_to(value, np.shape(old)).astype(
                    np.result_type(old)
                ),
            )
        return out

    def model(self, u, v, wavel):
        raise ValueError(
            "Attached changes with time: evaluate model.at(mjd), or let "
            "OIData.model evaluate each sample at its own time."
        )

    def _image(self, xx, yy, pixel_scale_mas):
        raise ValueError(
            "Attached changes with time: render model.at(mjd) instead."
        )

    def _weight(self, wavel=None):
        return self.component._weight(wavel)

    def total_spectrum(self, wavel):
        return self.component.total_spectrum(wavel)

    def is_physical(self):
        return self.component.is_physical()


class OrbitalBinary(SourceModel):
    """A primary and a point-source companion moving on a Keplerian orbit.

    The time-dependent form of
    [`BinaryModelCartesian`][virgil.models.BinaryModelCartesian]:
    [`at`][virgil.models.SourceModel.at] returns the ``BinaryModelCartesian``
    with the companion at the orbit's sky position, so each snapshot keeps the
    fast binary path. The same scene as ``System(primary=PointSource(),
    companion=Attached(PointSource(flux), orbit))``. The primary is the scene's
    origin and reference. Evaluate it with one snapshot per dataset or epoch
    through [`Epochs`][virgil.epochs.Epochs], or per sample by
    [`OIData.model`][virgil.oidata.OIData.model].

    Parameters
    ----------
    orbit : KeplerOrbit
        The companion's orbit about the primary (any orbit class with ``t_ref``
        and ``_relative``).
    flux : float
        Companion/primary flux ratio.

    Examples
    --------
    >>> from virgil.models import OrbitalBinary
    >>> from virgil.orbits import KeplerOrbit
    >>> orbit = KeplerOrbit(700.0, 0.0, 0.3, 50.0, 60.0, 120.0, 20.0,
    ...                     t_ref=60500.0)
    >>> snap = OrbitalBinary(orbit, 0.1).at(60500.0)
    >>> type(snap).__name__, round(float(snap.flux), 2)
    ('BinaryModelCartesian', 0.1)
    """

    orbit: Any
    flux: jax.Array

    def __init__(self, orbit, flux):
        self.orbit = orbit
        self.flux = np.asarray(flux, dtype=float)

    @property
    def time_dependent(self):
        return True

    @property
    def broadcast_in_time(self):
        return True

    def at(self, mjd, t_ref=0.0):
        dt = _days_since(mjd, self.orbit.t_ref - t_ref)
        _warn_if_mjd_without_t_ref(self.orbit.t_ref, dt)
        dra, ddec, _ = self.orbit._relative(dt)
        return BinaryModelCartesian(dra, ddec, self.flux)

    def model(self, u, v, wavel):
        raise ValueError(
            "OrbitalBinary changes with time: evaluate model.at(mjd), or "
            "let OIData.model evaluate each sample at its own time."
        )

    def _image(self, xx, yy, pixel_scale_mas):
        raise ValueError(
            "OrbitalBinary changes with time: render model.at(mjd) instead."
        )

    def total_spectrum(self, wavel):
        return BinaryModelCartesian(0.0, 0.0, self.flux).total_spectrum(wavel)

    def is_physical(self):
        return _flux_is_non_negative(self.flux)


class BinaryModelAngular(SourceModel):
    """
    A primary star and a point-source companion, in polar coordinates.

    Parameters
    ----------
    sep : float or array-like
        On-sky separation in milliarcseconds.
    pa : float or array-like
        Position angle in degrees, measured East of North.
    flux : float or array-like
        Companion/primary flux ratio (e.g. 0.01 for a companion 100 times
        fainter, i.e. contrast 100 or 5 mag; see
        [`flux_to_contrast`][virgil.limits.flux_to_contrast]).

    Notes
    -----
    Equivalent to [`BinaryModelCartesian`][virgil.models.BinaryModelCartesian]
    at ``dra = sep sin(pa)``, ``ddec = sep cos(pa)``; convenient for
    reporting astrometry directly in separation and position angle.
    """

    sep: jax.Array
    pa: jax.Array
    flux: jax.Array

    def __init__(self, sep, pa, flux):
        self.sep = np.asarray(sep, dtype=float)
        self.pa = np.asarray(pa, dtype=float)
        self.flux = np.asarray(flux, dtype=float)

    def to_cartesian(self):
        """Return the equivalent [`BinaryModelCartesian`][virgil.models.BinaryModelCartesian]."""
        th = self.pa * dtor
        return BinaryModelCartesian(
            self.sep * np.sin(th), self.sep * np.cos(th), self.flux
        )

    def total_spectrum(self, wavel):
        return self.to_cartesian().to_system().total_spectrum(wavel)

    def is_physical(self):
        return _flux_is_non_negative(self.flux)

    def model(self, u, v, wavel):
        """Complex visibilities on baselines ``u``, ``v`` (m) at ``wavel`` (m)."""
        th = self.pa * dtor  # sep (mas) and PA (deg, East of North) -> offsets
        return cvis_binary(
            u / wavel,
            v / wavel,
            self.sep * np.sin(th),
            self.sep * np.cos(th),
            self.flux,
        )

    def _image(self, xx, yy, pixel_scale_mas):
        return self.to_cartesian()._image(xx, yy, pixel_scale_mas)


class BinaryModelCartesian(SourceModel):
    """
    A primary star and a point-source companion, in Cartesian sky offsets.

    Parameters
    ----------
    dra : float or array-like
        Right-ascension offset of the companion in milliarcseconds, positive
        to the East.
    ddec : float or array-like
        Declination offset of the companion in milliarcseconds, positive to
        the North.
    flux : float or array-like
        Companion/primary flux ratio (e.g. 0.01 for a companion 100 times
        fainter, i.e. contrast 100 or 5 mag).

    Notes
    -----
    This is the fast form of ``System(primary=PointSource(),
    companion=PointSource(flux, dra, ddec))`` (see :meth:`to_system`), and
    the most convenient parameterization for grid searches and fits.
    """

    dra: jax.Array
    ddec: jax.Array
    flux: jax.Array

    def __init__(self, dra, ddec, flux):
        self.dra = np.asarray(dra, dtype=float)
        self.ddec = np.asarray(ddec, dtype=float)
        self.flux = np.asarray(flux, dtype=float)

    def to_angular(self):
        """Return the equivalent [`BinaryModelAngular`][virgil.models.BinaryModelAngular]."""
        sep, pa = separation_pa(self.dra, self.ddec)
        return BinaryModelAngular(sep, pa, self.flux)

    def total_spectrum(self, wavel):
        return self.to_system().total_spectrum(wavel)

    def is_physical(self):
        return _flux_is_non_negative(self.flux)

    def model(self, u, v, wavel):
        """Complex visibilities on baselines ``u``, ``v`` (m) at ``wavel`` (m)."""
        uu, vv = u / wavel, v / wavel
        return cvis_binary(uu, vv, self.dra, self.ddec, self.flux)

    def to_system(self):
        """Return the equivalent ``System(primary=..., companion=...)``.

        The [`System`][virgil.models.System] form is slower to evaluate
        but can be extended, e.g. by adding a disk around the primary.
        """
        return System(
            primary=PointSource(),
            companion=PointSource(
                flux=self.flux, dra=self.dra, ddec=self.ddec
            ),
        )

    def _image(self, xx, yy, pixel_scale_mas):
        return self.to_system()._image(xx, yy, pixel_scale_mas)


class HarmonixModel(SourceModel):
    """Wrapper for external models with harmonix-like visibility methods.

    ``source`` is an ordinary field, so a pytree source (such as a harmonix
    ``Harmonix``) has parameters that zodiax paths reach and fits can vary,
    e.g. ``"source.radius"`` (mas) or ``"source.data"`` (map coefficients for l
    >= 1, with Y00 = 1); a source that is not a pytree needs
    ``eqx.filter_jit``, not ``jax.jit``. A wrapped source has weight 1 inside a
    [System][virgil.models.System], where only harmonix stars can be drawn.

    Parameters
    ----------
    source : object
        External model. ``getattr(source, visibility_method)`` is called with
        the baselines (and ``observation_time`` if given) and must return
        complex visibilities normalized to 1 at zero baseline.
    visibility_method, render_method : str, optional
        Names of the visibility method (default ``"model"``) and of a
        ``render(npix, fov_mas)`` method for
        [`render`][virgil.models.HarmonixModel.render] (default ``"render"``);
        sources with a ``surface`` and a ``radius`` in mas (harmonix stars) are
        drawn from the surface's intensity, East left and North up, when there
        is none.
    expects_wavelength_units : bool, optional
        If True (default), pass spatial frequencies ``u / wavel``, ``v /
        wavel``; otherwise ``u``, ``v`` in metres.
    observation_time : optional
        Extra argument passed after the baselines, e.g. a time for rotating
        stars; static (hashable), so array-valued times are not supported under
        ``jax.jit``.
    """

    source: Any
    visibility_method: str = eqx.field(static=True)
    render_method: str = eqx.field(static=True)
    expects_wavelength_units: bool = eqx.field(static=True)
    observation_time: Any = eqx.field(static=True)

    def __init__(
        self,
        source,
        visibility_method="model",
        render_method="render",
        expects_wavelength_units=True,
        observation_time=None,
    ):
        self.source = source
        self.visibility_method = str(visibility_method)
        self.render_method = str(render_method)
        self.expects_wavelength_units = bool(expects_wavelength_units)
        self.observation_time = observation_time

    def model(self, u, v, wavel):
        method = getattr(self.source, self.visibility_method)
        args = (
            [u / wavel, v / wavel] if self.expects_wavelength_units else [u, v]
        )
        if self.observation_time is not None:
            args.append(self.observation_time)
        return np.asarray(method(*args))

    def render(self, npix=256, fov_mas=200.0):
        if hasattr(self.source, self.render_method):
            return _normalize_image(
                getattr(self.source, self.render_method)(npix, fov_mas)
            )
        return super().render(npix, fov_mas)

    def _image(self, xx, yy, pixel_scale_mas):
        source = self.source
        if not (hasattr(source, "surface") and hasattr(source, "radius")):
            raise NotImplementedError(
                "Wrapped source does not expose a compatible render method."
            )
        theta = 0.0
        if hasattr(source, "rotational_phase"):
            theta = source.rotational_phase(
                0.0 if self.observation_time is None else self.observation_time
            )
        # harmonix's visibilities use the sign convention of offset_phase
        # with the surface's x (East) and y (North) in stellar radii, so the
        # intensity at sky offset (dra, ddec) is the surface's at
        # (dra, ddec) / radius. (jaxoplanet's Surface.render puts East on
        # the right instead.)
        surface = source.surface
        if hasattr(source, "data"):
            # harmonix reads the map from ``data`` (the l >= 1 coefficients,
            # with Y00 = 1) rather than ``surface.y``, so draw that map.
            y00 = np.ones((1,), dtype=source.data.dtype)
            ylm = type(surface.y).from_dense(
                np.concatenate([y00, source.data]), normalize=False
            )
            surface = eqx.tree_at(lambda s: s.y, surface, ylm)
        x, y = xx / source.radius, yy / source.radius
        on_disk = x**2 + y**2 < 1.0
        x, y = np.where(on_disk, x, 0.0), np.where(on_disk, y, 0.0)
        z = np.sqrt(1.0 - x**2 - y**2)
        intensity = surface._intensity(
            x.ravel(), y.ravel(), z.ravel(), theta=theta
        )
        return np.where(on_disk, np.reshape(intensity, xx.shape), 0.0)


def cvis_binary(u, v, dra, ddec, flux):
    """Compute complex visibilities for a Cartesian-parameterized binary model.

    Source: Martinache (2010), xara (formerly pysco),
    [references](../../references.md#xara); closure-phase contrast limits:
    Ireland (2013), [references](../../references.md#ireland2013).

    Parameters
    ----------
    u, v : array-like
        Baseline coordinates in wavelength units (cycles / rad).
    dra, ddec : float or array-like
        Offset of the companion in mas, positive to the East and the North.
    flux : float or array-like
        Companion/primary flux ratio.

    Returns
    -------
    array-like
        Complex visibility samples, normalized to 1 at zero baseline.
    """

    # normalize visibilities so total power is 1
    primary = 1.0 / (1.0 + flux)
    companion = flux / (1.0 + flux)

    return primary + companion * offset_phase(u, v, dra, ddec)


@renamed()
def cvis_uniform_disk(u, v, diam, dra=0.0, ddec=0.0):
    """Compute complex visibilities for a uniform (tophat) disk.

    The amplitude is ``2 * J1(x) / x`` with ``x = pi * diam_rad * base_norm``,
    the disk diameter in radians times ``hypot(u, v)``, the baseline length in
    wavelength units.

    Parameters
    ----------
    u, v : array-like
        Baseline coordinates in wavelength units (cycles / rad).
    diam : float or array-like
        Diameter of the uniform disk in mas.
    dra, ddec : float or array-like
        Right-ascension and declination offsets in mas.

    Returns
    -------
    array-like
        Complex visibility samples.
    """
    diam_rad = mas2rad * diam
    base_norm = np.hypot(u, v)
    kernel = np.pi * base_norm * diam_rad

    # Keep 0 out of the division so the unused branch has finite gradients.
    at_zero = kernel == 0
    safe_kernel = np.where(at_zero, 1.0, kernel)
    envelope = np.where(
        at_zero,
        1.0 + 0j,
        (2.0 * bessel_jn(1, safe_kernel)[1]) / safe_kernel + 0j,
    )

    return envelope * offset_phase(u, v, dra, ddec)


def cvis_limb_darkened_disk(u, v, diam, coeffs, powers, dra=0.0, ddec=0.0):
    r"""Visibilities of a disk whose brightness is a sum of powers of $\mu$.

    For $I(\mu) = \sum_\nu a_\nu \mu^\nu$, $\mu = \sqrt{1 - (r/R)^2}$, the
    visibility is a sum of $J_{\nu/2+1}(x)/x^{\nu/2+1}$ terms with $x =
    \pi\theta|b|/\lambda$, normalized to 1 at zero baseline. Powers need not be
    integers, with $-2 < \nu \le 22$ (the lower limit keeps the flux finite).
    Formula: [Limb-darkened
    stars](../../limb_darkening.md#profiles-and-visibilities).

    Source: Quirrenbach et al. (1996), eqs. 1-4,
    [references](../../references.md#quirrenbach1996); generalized by harmonix,
    [references](../../references.md#harmonix).

    Parameters
    ----------
    u, v : array-like
        Baseline coordinates in wavelength units (cycles / rad).
    diam : float or array-like
        Limb-darkened diameter in mas.
    coeffs, powers : array-like, sequence of float
        Coefficients $a_\nu$ (traceable) and the powers $\nu$ of $\mu$
        (static), one each.
    dra, ddec : float or array-like
        Right-ascension and declination offsets in mas.

    Returns
    -------
    array-like
        Complex visibility samples.
    """
    # At most 22 keeps the Bessel orders needed at most 12.
    bad = [nu for nu in powers if not -2.0 < float(nu) <= 22.0]
    if bad:
        raise ValueError(f"Powers of mu must be in (-2, 22], got {bad}.")
    x = np.pi * np.hypot(u, v) * mas2rad * diam
    total = 0.0
    norm = 0.0
    for i, nu in enumerate(powers):
        nu = float(nu)
        scale = 2.0 ** (nu / 2) * math.gamma(nu / 2 + 1)
        total = total + coeffs[i] * scale * bessel_jv_over_xv(nu / 2 + 1, x)
        norm = norm + coeffs[i] / (nu + 2)
    return (total / norm + 0j) * offset_phase(u, v, dra, ddec)


def cvis_radial_dirac_delta_modulated(u, v, r0, az_amps, az_phis):
    r"""Complex visibility of an azimuthally modulated radial Dirac-delta ring.

    The intensity is $I(r, \theta) \propto \delta(r-r_0)(1 + \sum_m A_m
    \cos[m(\theta - \phi_m)])$ with $\theta$ counter-clockwise from North to
    East. Rotation and inclination are not applied: transform the $uv$
    coordinates separately. One first-order modulation with phase 0 is bright
    to the North and faint to the South; with phase 90, bright to the East.

    Parameters
    ----------
    u, v : array-like
        Baseline coordinates in wavelength units (cycles / rad).
    r0 : float or array-like
        Radial position of the ring in mas.
    az_amps, az_phis : array-like
        1D amplitudes $A_m$ of the cosine modulations, and their offset angles
        $\phi_m$ in degrees relative to the position angle of the projected
        major axis, first order first.

    Returns
    -------
    array-like
        Complex visibility samples.
    """
    # Add radially symmetric component (order m=0) to beginning of the order arrays.
    az_amps = np.concatenate([np.array([1.0]), az_amps])
    az_phis = np.concatenate([np.array([0.0]), az_phis])
    az_orders = np.arange(az_amps.size)

    r0_rad = r0 * mas2rad
    az_phis_rad = az_phis * dtor

    # Get length of baseline and baseline projection angle (i.e. counterclockwise
    # angle in uv-plane, turning from top, i.e. positive v, to left, i.e. positive u).
    # At u = v = 0 (e.g. flagged samples) hypot and arctan2 have undefined
    # derivatives; the double where keeps gradients finite there, where the
    # visibility is exactly 1.
    origin = (u == 0) & (v == 0)
    u_safe, v_safe = np.where(origin, 1.0, u), np.where(origin, 1.0, v)
    base_norm = np.where(origin, 0.0, np.hypot(u_safe, v_safe))
    base_proj_ang_rad = np.where(origin, 0.0, np.arctan2(u_safe, v_safe))

    az_order_max = np.size(az_orders) - 1
    xbes = 2.0 * np.pi * base_norm * r0_rad
    bessel_vals = bessel_jn(az_order_max, xbes)

    azmod_cvis_terms = jax.vmap(
        lambda amp, phi, order: (
            amp
            * np.exp(-0.5j * np.pi * order)
            * np.cos(order * (base_proj_ang_rad - phi))
            * bessel_vals[order]
        ),
        in_axes=(0, 0, 0),
        out_axes=0,
    )(az_amps, az_phis_rad, az_orders)

    return np.sum(azmod_cvis_terms, axis=0)


def _cvis_gaussian_envelope(u, v, fwhm):
    """Complex visibility envelope of a centered isotropic 2D Gaussian PSF, used
    as the convolution kernel of [`ModulatedGaussianRim`][virgil.models.ModulatedGaussianRim]. Not offered as a public
    function: this is a plain Gaussian envelope
    with no point-source/companion mixture. The rim evaluates it on deprojected
    spatial frequencies, so the kernel is isotropic in the rim plane.
    """
    fwhm_rad = fwhm * mas2rad
    base_norm = np.hypot(u, v)
    return (
        np.exp(-(np.pi**2) * fwhm_rad**2 * base_norm**2 / (4.0 * np.log(2.0)))
        + 0j
    )


def _scene_images(scene, prefix=""):
    """Yield ``(path prefix, Image)`` for each Image in a scene, walking
    ``System`` components and ``Rotated`` sources (``"env."``, ``"source."``)."""
    if isinstance(scene, Image):
        yield prefix, scene
    elif isinstance(scene, System):
        for name, part in scene.components.items():
            yield from _scene_images(part, prefix + name + ".")
    elif isinstance(scene, Rotated):
        yield from _scene_images(scene.source, prefix + "source.")
