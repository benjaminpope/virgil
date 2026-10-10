"""Wavelength-dependent fluxes for source components.

A component's ``flux`` is either a number (achromatic) or a spectrum from this
module, which gives its weight at each wavelength. Inside a
[`System`][virgil.models.System] the visibility is then
``V(λ) = Σ f_i(λ) V_i / Σ f_i(λ)``, as in SPARCO (Kluska et al. 2014):

```python
star = UniformDisk(0.5, flux=PowerLaw(1.0, index=-4.0, wavel0=1.65e-6))
disk = GaussianDisk(5.0, flux=PowerLaw(0.3, index=1.0, wavel0=1.65e-6))
ring = GaussianDisk(5.0, flux=BlackBody(0.3, temperature=1200.0))
wind = GaussianDisk(
    2.0,
    flux=Sum(
        continuum=PowerLaw(0.3, wavel0=2.15e-6),
        brg=GaussianLine(0.5, line_wavel=2.1661e-6, fwhm=1.0e-9),
    ),
)
free = PointSource(flux=Nodes(values, channel_wavelengths))
```

Spectrum parameters are reached by path like any other, e.g.
``"disk.flux.ratio"``, ``"disk.flux.index"``, ``"ring.flux.temperature"``,
``"wind.flux.brg.amplitude"`` or ``"free.flux.values"`` (one value per
node). Flux ratios are relative: component ``i``'s fraction of the total at
the reference wavelength is ``f_i / Σ f``, as SPARCO's ``f_i``.

Every spectrum has a reference wavelength ``wavel0``, and its reference
flux (``spectrum()``, used when rendering images) is its value there.
Only the evaluated flux must be non-negative: a line or node excess inside a
[`Sum`][virgil.spectra.Sum] may be negative (absorption) as long as the
total is not. [`is_physical`][virgil.spectra.Spectrum.is_physical] checks
the total at each spectrum's characteristic wavelengths (``wavel0``, nodes
and line centres), which is where a sum of these shapes has its minima.
"""

import warnings

import equinox as eqx
import jax
import jax.numpy as np
import numpy as onp
import zodiax as zdx

from ._utils import check_part_name, concrete


__all__ = [
    "BlackBody",
    "GaussianLine",
    "LorentzianLine",
    "Nodes",
    "PowerLaw",
    "Spectrum",
    "Sum",
    "Tabulated",
    "flux_at",
    "reference_flux",
]


class Spectrum(zdx.Base):  # type: ignore[reportGeneralTypeIssues]
    """Base class for component spectra.

    Subclasses implement ``_at(wavel)``, the flux at ``wavel`` (metres, any
    shape), and have a reference wavelength ``wavel0``; calling the
    spectrum with no wavelength gives its reference flux, the value at
    ``wavel0`` (as used when rendering images). ``_params_valid`` checks
    the parameters' domains, and ``_check_wavel`` lists the wavelengths
    where the flux must be non-negative.
    """

    def __call__(self, wavel=None):
        if wavel is None:
            return self._at(self.wavel0)
        return self._at(np.asarray(wavel))

    def _at(self, wavel):
        raise NotImplementedError

    def _params_valid(self):
        return np.all(self.wavel0 > 0.0)

    def _check_wavel(self):
        return np.ravel(self.wavel0)

    def is_physical(self):
        """Whether the spectrum is valid, as a (traceable) boolean.

        Its parameters are in their domains and its flux is non-negative at
        its characteristic wavelengths (``wavel0``, nodes, line centres).
        """
        return self._params_valid() & np.all(self(self._check_wavel()) >= 0.0)


class PowerLaw(Spectrum):
    """Power-law spectrum ``ratio * (λ / wavel0) ** index``.

    Parameters
    ----------
    ratio : float or array-like
        Flux at the reference wavelength, relative to the other components.
    index : float or array-like, optional
        Spectral index (default 0, i.e. achromatic). A star in the
        Rayleigh-Jeans regime of F_λ has index -4.
    wavel0 : float or array-like, optional
        Reference wavelength in metres (default 1.65e-6, H band).

    Examples
    --------
    >>> round(float(PowerLaw(0.2, index=-4.0, wavel0=1.6e-6)(3.2e-6)), 6)
    0.0125
    """

    ratio: jax.Array
    index: jax.Array
    # An ordinary (traceable) field rather than a static one, so that fixed
    # parameters passed through jitted functions can set it.
    wavel0: jax.Array

    def __init__(self, ratio, index=0.0, wavel0=1.65e-6):
        self.ratio = np.asarray(ratio, dtype=float)
        self.index = np.asarray(index, dtype=float)
        self.wavel0 = np.asarray(wavel0, dtype=float)

    def __call__(self, wavel=None):
        if wavel is None:
            return self.ratio
        return self._at(np.asarray(wavel))

    def _at(self, wavel):
        return self.ratio * (wavel / self.wavel0) ** self.index

    def is_physical(self):
        return np.all(self.ratio >= 0.0) & np.all(self.wavel0 > 0.0)

    def __check_init__(self):
        value = concrete(self.ratio)
        if value is not None and (value < 0.0).any():
            raise ValueError(
                f"PowerLaw ratio {value.tolist()} is negative; fluxes must be "
                "non-negative."
            )
        wavel0 = concrete(self.wavel0)
        if wavel0 is not None and (wavel0 <= 0.0).any():
            raise ValueError(
                f"PowerLaw wavel0 {wavel0.tolist()} must be a positive "
                "wavelength (in metres)."
            )


class BlackBody(Spectrum):
    """Planck spectrum ``ratio * B_λ(T, λ) / B_λ(T, wavel0)``.

    The shape of a blackbody at ``temperature`` in F_λ, normalized to
    ``ratio`` at ``wavel0``, as SPARCO uses for dust and companions (e.g.
    Hillen et al. 2016). At long wavelengths (``hc/λkT`` small) it tends to
    the Rayleigh-Jeans ``PowerLaw`` with index -4.

    Parameters
    ----------
    ratio : float or array-like
        Flux at the reference wavelength, relative to the other components.
    temperature : float or array-like
        Temperature in kelvin.
    wavel0 : float or array-like, optional
        Reference wavelength in metres (default 1.65e-6, H band).

    Examples
    --------
    >>> round(float(BlackBody(0.2, 1500.0)(1.65e-6)), 6)
    0.2
    """

    ratio: jax.Array
    temperature: jax.Array
    wavel0: jax.Array

    def __init__(self, ratio, temperature, wavel0=1.65e-6):
        self.ratio = np.asarray(ratio, dtype=float)
        self.temperature = np.asarray(temperature, dtype=float)
        self.wavel0 = np.asarray(wavel0, dtype=float)

    def __call__(self, wavel=None):
        if wavel is None:
            return self.ratio
        return self._at(np.asarray(wavel))

    def _at(self, wavel):
        return self.ratio * _planck_ratio(
            wavel, self.temperature, self.wavel0, self.temperature
        )

    def is_physical(self):
        return (
            np.all(self.ratio >= 0.0)
            & np.all(self.temperature > 0.0)
            & np.all(self.wavel0 > 0.0)
        )

    def __check_init__(self):
        for name, value, ok in (
            ("ratio", self.ratio, lambda x: x >= 0.0),
            ("temperature", self.temperature, lambda x: x > 0.0),
            ("wavel0", self.wavel0, lambda x: x > 0.0),
        ):
            value = concrete(value)
            if value is not None and not ok(value).all():
                raise ValueError(
                    f"BlackBody {name} {value.tolist()} must be "
                    + ("non-negative." if name == "ratio" else "positive.")
                )


class _Line(Spectrum):
    """A spectral line, ``amplitude * shape((λ - line_wavel) / fwhm)``."""

    amplitude: jax.Array
    line_wavel: jax.Array
    fwhm: jax.Array
    wavel0: jax.Array

    def __init__(self, amplitude, line_wavel, fwhm, wavel0=None):
        self.amplitude = np.asarray(amplitude, dtype=float)
        self.line_wavel = np.asarray(line_wavel, dtype=float)
        self.fwhm = np.asarray(fwhm, dtype=float)
        self.wavel0 = np.asarray(
            line_wavel if wavel0 is None else wavel0, dtype=float
        )

    def _at(self, wavel):
        return self.amplitude * self._profile(
            (wavel - self.line_wavel) / self.fwhm
        )

    def _params_valid(self):
        return (
            np.all(self.line_wavel > 0.0)
            & np.all(self.fwhm > 0.0)
            & np.all(self.wavel0 > 0.0)
        )

    def _check_wavel(self):
        return np.concatenate(
            [np.ravel(self.line_wavel), np.ravel(self.wavel0)]
        )

    def __check_init__(self):
        name = type(self).__name__
        for field in ("amplitude", "line_wavel", "fwhm", "wavel0"):
            shape = onp.shape(getattr(self, field))
            if shape != ():
                raise ValueError(
                    f"{name} {field} must be a scalar, not shape {shape}; "
                    "use one line per value (or jax.vmap)."
                )
        for field in ("line_wavel", "fwhm", "wavel0"):
            value = concrete(getattr(self, field))
            if value is not None and not (
                onp.isfinite(value).all() and (value > 0.0).all()
            ):
                raise ValueError(
                    f"{name} {field} {value.tolist()} must be finite and "
                    "positive (metres)."
                )


class GaussianLine(_Line):
    """A Gaussian emission or absorption line.

    ``amplitude * exp(-4 ln 2 ((λ - line_wavel) / fwhm)²)``: the flux at the
    line centre, with the full width at half maximum ``fwhm``. Its integral
    over wavelength is ``amplitude * fwhm * sqrt(π / (4 ln 2))``.

    On its own a line is the whole flux of a component, so ``amplitude``
    must be non-negative; inside a [`Sum`][virgil.spectra.Sum] with a
    continuum a negative amplitude is an absorption line.

    Parameters
    ----------
    amplitude : float
        Flux at the line centre, relative to the other components.
    line_wavel : float
        Line centre in metres (shifted by any velocity: λ₀(1 + v/c)).
    fwhm : float
        Full width at half maximum in metres.
    wavel0 : float, optional
        Reference wavelength in metres; the line centre by default.

    Examples
    --------
    >>> line = GaussianLine(0.4, line_wavel=2.1661e-6, fwhm=1.0e-9)
    >>> round(float(line(2.1661e-6 + 0.5e-9)), 4)
    0.2
    """

    @staticmethod
    def _profile(x):
        return np.exp(-4.0 * np.log(2.0) * x**2)


class LorentzianLine(_Line):
    """A Lorentzian emission or absorption line.

    ``amplitude / (1 + 4 ((λ - line_wavel) / fwhm)²)``: the flux at the line
    centre, with the full width at half maximum ``fwhm``. Its integral over
    wavelength is ``amplitude * π fwhm / 2``. Its wings fall off slowly, so
    over a wide band it adds a near-constant pedestal.

    Parameters are as for [`GaussianLine`][virgil.spectra.GaussianLine].

    Examples
    --------
    >>> line = LorentzianLine(0.4, line_wavel=2.1661e-6, fwhm=1.0e-9)
    >>> round(float(line(2.1661e-6 + 0.5e-9)), 4)
    0.2
    """

    @staticmethod
    def _profile(x):
        return 1.0 / (1.0 + 4.0 * x**2)


class Nodes(Spectrum):
    """A spectrum through free values at fixed wavelengths.

    Linear or natural-cubic interpolation between the nodes. Beyond the end
    nodes the flux is either held at the end values (``outside="constant"``)
    or a fixed number (e.g. ``outside=0.0`` for an excess that lives only in
    a line window, on top of a continuum in a [`Sum`][virgil.spectra.Sum]).
    With ``outside=0.0`` the continuum is fixed by the channels outside the
    window, so the two are identifiable; give the end nodes the value 0 (or
    fix them there) to keep the spectrum continuous.

    Fit the ``values`` with a prior of their shape; a smooth spectrum is a
    prior on them (e.g. a Gaussian process over wavelength), not a feature
    of this class.

    Parameters
    ----------
    values : array-like, shape (n,)
        Flux at each node, relative to the other components. They may be
        negative (an absorption excess) as long as a component's total flux
        is not.
    wavel : array-like, shape (n,)
        Node wavelengths in metres: finite, positive and strictly
        increasing. Fixed (not fitted) in practice.
    kind : {"linear", "cubic"}, optional
        Interpolation; ``"cubic"`` is a natural cubic spline and needs at
        least three nodes.
    outside : "constant" or float, optional
        The flux beyond the end nodes (default ``"constant"``).
    wavel0 : float, optional
        Reference wavelength in metres; the first node by default.

    Examples
    --------
    >>> spectrum = Nodes([0.2, 0.4], [2.0e-6, 2.2e-6])
    >>> round(float(spectrum(2.1e-6)), 6)
    0.3
    >>> excess = Nodes([0.0, 0.5, 0.0], [2.16e-6, 2.166e-6, 2.172e-6], outside=0.0)
    >>> float(excess(2.0e-6))
    0.0
    """

    values: jax.Array
    wavel: jax.Array
    wavel0: jax.Array
    kind: str = eqx.field(static=True)
    outside: object = eqx.field(static=True)

    def __init__(
        self, values, wavel, kind="linear", outside="constant", wavel0=None
    ):
        if kind not in ("linear", "cubic"):
            raise ValueError(
                f"Nodes kind must be 'linear' or 'cubic', not {kind!r}."
            )
        if outside != "constant" and not isinstance(outside, (int, float)):
            raise ValueError(
                f"Nodes outside must be 'constant' or a number, not {outside!r}."
            )
        self.values = np.asarray(values, dtype=float)
        self.wavel = np.asarray(wavel, dtype=float)
        self.kind = kind
        self.outside = outside if outside == "constant" else float(outside)
        self.wavel0 = (
            self.wavel[..., 0]
            if wavel0 is None and self.wavel.ndim == 1 and self.wavel.size
            else np.asarray(wavel0, dtype=float)
        )

    def _at(self, wavel):
        clipped = np.clip(wavel, self.wavel[0], self.wavel[-1])
        if self.kind == "linear" or self.values.size < 3:
            inside = np.interp(clipped, self.wavel, self.values)
        else:
            inside = _natural_cubic(clipped, self.wavel, self.values)
        if self.outside == "constant":
            return inside
        within = (wavel >= self.wavel[0]) & (wavel <= self.wavel[-1])
        return np.where(within, inside, self.outside)

    def _params_valid(self):
        return (
            np.all(self.wavel > 0.0)
            & np.all(np.diff(self.wavel) > 0.0)
            & np.all(self.wavel0 > 0.0)
        )

    def _check_wavel(self):
        points = [self.wavel, np.ravel(self.wavel0)]
        if self.kind == "cubic" and self.values.size >= 3:
            # A natural spline can dip below zero between positive nodes.
            points.append(_cubic_extrema(self.wavel, self.values))
        return np.concatenate(points)

    def __check_init__(self):
        if (
            self.values.shape != self.wavel.shape
            or self.values.ndim != 1
            or self.values.size == 0
        ):
            raise ValueError(
                "Nodes needs non-empty 1D values and wavel of the same "
                f"length, not shapes {self.values.shape} and "
                f"{self.wavel.shape}."
            )
        if self.kind == "cubic" and self.values.size < 3:
            raise ValueError("A cubic Nodes spectrum needs at least 3 nodes.")
        values = concrete(self.values)
        if values is not None and not onp.isfinite(values).all():
            raise ValueError("Nodes values must be finite.")
        wavel = concrete(self.wavel)
        if wavel is not None and not (
            onp.isfinite(wavel).all()
            and (wavel > 0.0).all()
            and (onp.diff(wavel) > 0.0).all()
        ):
            raise ValueError(
                "Nodes wavel must be finite, positive and strictly increasing."
            )
        if self.wavel0.shape != ():
            raise ValueError(
                f"Nodes wavel0 must be a scalar, not shape {self.wavel0.shape}."
            )
        wavel0 = concrete(self.wavel0)
        if wavel0 is not None and not (
            onp.isfinite(wavel0).all() and (wavel0 > 0.0).all()
        ):
            raise ValueError(
                f"Nodes wavel0 {wavel0.tolist()} must be a positive "
                "wavelength (in metres)."
            )


class Sum(Spectrum):
    """The sum of named spectra, e.g. a continuum plus lines.

    Parts are reached by name, like the components of a
    [`System`][virgil.models.System]: ``"star.flux.continuum.index"``,
    ``"star.flux.brg.amplitude"``. Only the total must be non-negative, so a
    part may be an absorption line or a negative node excess.

    Parameters
    ----------
    parts : dict, optional
        ``{name: Spectrum}``, positionally; or give them by keyword.
    wavel0 : float, optional
        Reference wavelength in metres, where the reference flux is the sum
        of the parts; the first part's ``wavel0`` by default.

    Examples
    --------
    >>> flux = Sum(
    ...     continuum=PowerLaw(1.0, wavel0=2.2e-6),
    ...     brg=GaussianLine(-0.3, line_wavel=2.1661e-6, fwhm=1.0e-9),
    ... )
    >>> round(float(flux(2.1661e-6)), 4)  # continuum minus the absorption
    0.7
    >>> round(float(flux()), 4)  # the reference flux, at 2.2 µm
    1.0
    """

    names: tuple = eqx.field(static=True)
    parts: tuple
    wavel0: jax.Array

    def __init__(self, parts=None, /, *, wavel0=None, **named):
        parts = {**(parts or {}), **named}
        if not parts:
            raise ValueError("Sum needs at least one spectrum.")
        for name, part in parts.items():
            check_part_name(
                name,
                Sum,
                "Sum part",
                "flux.brg.amplitude",
                "a Sum attribute or method",
            )
            if not isinstance(part, Spectrum):
                raise TypeError(f"Part '{name}' is not a Spectrum: {part!r}")
        self.names = tuple(parts)
        self.parts = tuple(parts.values())
        if wavel0 is None:
            wavel0 = getattr(self.parts[0], "wavel0", None)
            if wavel0 is None:
                raise ValueError(
                    f"Sum part '{self.names[0]}' has no wavel0; give Sum a "
                    "wavel0."
                )
        self.wavel0 = np.asarray(wavel0, dtype=float)

    @property
    def components(self):
        """The parts as a ``{name: spectrum}`` dictionary, in order."""
        return dict(zip(self.names, self.parts))

    def __getattr__(self, name):
        try:
            names = object.__getattribute__(self, "names")
            parts = object.__getattribute__(self, "parts")
        except AttributeError:
            raise AttributeError(name) from None
        if name in names:
            return parts[names.index(name)]
        raise AttributeError(
            f"Sum has no part or attribute '{name}'; its parts are "
            f"{list(names)}."
        )

    def _at(self, wavel):
        return sum(part._at(wavel) for part in self.parts)

    def _params_valid(self):
        valid = np.all(self.wavel0 > 0.0)
        for part in self.parts:
            valid = valid & part._params_valid()
        return valid

    def _check_wavel(self):
        return np.concatenate(
            [np.ravel(self.wavel0)] + [p._check_wavel() for p in self.parts]
        )


class Tabulated(Spectrum):
    """A free flux in every spectral channel, interpolated linearly between.

    **Deprecated**: use [`Nodes`][virgil.spectra.Nodes], which also has
    cubic interpolation, a fixed value outside the nodes, and a reference
    flux at ``wavel0`` like every other spectrum. ``Tabulated(ratio,
    wavel)`` is ``Nodes(ratio, wavel)`` except for its reference flux (the
    mean over the nodes) and its parameter name (``ratio``). It is kept so
    that existing scripts run unchanged, and will be removed in a later
    release.

    For fitting a spectrum channel by channel, e.g. a companion's flux ratio
    across emission lines: give ``wavel`` the data's channel wavelengths and
    fit ``ratio`` (one value per channel) with a prior of that shape.

    Parameters
    ----------
    ratio : array-like, shape (n,)
        Flux at each node, relative to the other components: finite and
        non-negative, with n >= 1.
    wavel : array-like, shape (n,)
        Node wavelengths in metres: finite, positive and strictly
        increasing. Beyond the end nodes the flux is constant.

    Notes
    -----
    The reference flux (``wavel=None``, used when rendering) is the mean
    over the nodes.

    Examples
    --------
    >>> spectrum = Tabulated([0.2, 0.4], [2.0e-6, 2.2e-6])
    >>> round(float(spectrum(2.1e-6)), 6)
    0.3
    """

    ratio: jax.Array
    # Traceable, like PowerLaw.wavel0.
    wavel: jax.Array

    def __init__(self, ratio, wavel):
        warnings.warn(
            "Tabulated is deprecated; use virgil.spectra.Nodes (with wavel0 "
            "for the reference flux).",
            FutureWarning,
            stacklevel=2,
        )
        self.ratio = np.asarray(ratio, dtype=float)
        self.wavel = np.asarray(wavel, dtype=float)

    def __call__(self, wavel=None):
        if wavel is None:
            return np.mean(self.ratio)
        return self._at(np.asarray(wavel))

    def _at(self, wavel):
        return np.interp(wavel, self.wavel, self.ratio)

    def _check_wavel(self):
        return self.wavel

    def is_physical(self):
        return (
            np.all(self.ratio >= 0.0)
            & np.all(self.wavel > 0.0)
            & np.all(np.diff(self.wavel) > 0.0)
        )

    def __check_init__(self):
        if (
            self.ratio.shape != self.wavel.shape
            or self.ratio.ndim != 1
            or self.ratio.size == 0
        ):
            raise ValueError(
                f"Tabulated needs non-empty 1D ratio and wavel of the same "
                f"length, not shapes {self.ratio.shape} and {self.wavel.shape}."
            )
        value = concrete(self.ratio)
        if value is not None and not (
            onp.isfinite(value).all() and (value >= 0.0).all()
        ):
            raise ValueError(
                "Tabulated ratios must be finite and non-negative; fluxes "
                "cannot be negative."
            )
        wavel = concrete(self.wavel)
        if wavel is not None and not (
            onp.isfinite(wavel).all()
            and (wavel > 0.0).all()
            and (onp.diff(wavel) > 0.0).all()
        ):
            raise ValueError(
                "Tabulated wavel must be finite, positive and strictly "
                "increasing."
            )


# Planck's second radiation constant h c / k, in metre kelvin.
_HC_OVER_K = 1.438776877e-2


def _planck_ratio(wavel, temperature, wavel0, temperature0):
    """``B_λ(wavel, temperature) / B_λ(wavel0, temperature0)``, no overflow.

    Wavelengths in metres, temperatures in kelvin; all broadcast together.
    """
    x = _HC_OVER_K / (wavel * temperature)
    x0 = _HC_OVER_K / (wavel0 * temperature0)
    # (wavel0 / wavel)**5 * expm1(x0) / expm1(x), formed entirely in log
    # space: expm1(x) = exp(x) * -expm1(-x), so its log is
    # x + log(-expm1(-x)) without overflow. Exponentiating only the total
    # keeps the result finite whenever it is representable (in float32 the
    # factors alone can overflow when the ratio does not).
    log_ratio = (
        5.0 * np.log(wavel0 / wavel)
        + (x0 - x)
        + np.log(-np.expm1(-x0))
        - np.log(-np.expm1(-x))
    )
    return np.exp(log_ratio)


def _spline_moments(nodes, values):
    """Second derivatives of the natural cubic spline at the nodes.

    They solve a small tridiagonal system, built densely (node counts are
    small), with zero curvature at the ends. Differentiable in the values and
    node positions.
    """
    h = np.diff(nodes)
    slope = np.diff(values) / h
    interior = 2.0 * (h[:-1] + h[1:])
    a = np.diag(interior) + np.diag(h[1:-1], 1) + np.diag(h[1:-1], -1)
    return (
        np.zeros(values.size, dtype=values.dtype)
        .at[1:-1]
        .set(np.linalg.solve(a, 6.0 * np.diff(slope)))
    )


def _natural_cubic(x, nodes, values):
    """Natural cubic spline through ``(nodes, values)`` at ``x`` (in range)."""
    n = values.size
    m = _spline_moments(nodes, values)
    i = np.clip(np.searchsorted(nodes, x, side="right") - 1, 0, n - 2)
    t0, t1 = nodes[i], nodes[i + 1]
    hi = t1 - t0
    left, right = t1 - x, x - t0
    return (
        m[i] * left**3 / (6.0 * hi)
        + m[i + 1] * right**3 / (6.0 * hi)
        + (values[i] / hi - m[i] * hi / 6.0) * left
        + (values[i + 1] / hi - m[i + 1] * hi / 6.0) * right
    )


def _cubic_extrema(nodes, values):
    """Where each segment of the natural spline is stationary, clipped in.

    On a segment of width ``h`` the derivative is the quadratic
    ``a s² + b s + c`` in ``s = x - nodes[i]``. Both roots are returned
    (shape ``(2 (n - 1),)``), solved without branching (a stable quadratic
    formula, and the linear root where ``a`` vanishes), so the result is
    traceable and finite. A root outside the segment, or a complex one, is
    clipped to an end or the vertex, where the spline is merely evaluated.
    """
    h = np.diff(nodes)
    m = _spline_moments(nodes, values)
    m0, m1 = m[:-1], m[1:]
    a = (m0 + m1) / (2.0 * h)
    b = m0
    c = (values[1:] - values[:-1]) / h - h * (m1 - m0) / 6.0 - m0 * h / 2.0
    tiny = 1e-12 * (np.abs(b) + h * np.abs(a) + 1e-30)
    disc = np.sqrt(np.maximum(b**2 - 4.0 * a * c, 0.0))
    q = -0.5 * (b + np.where(b >= 0.0, disc, -disc))
    small_a = np.abs(a * h) <= tiny
    safe_a = np.where(small_a, 1.0, a)
    safe_q = np.where(np.abs(q) <= 1e-30, 1.0, q)
    safe_b = np.where(np.abs(b) <= 1e-30, 1.0, b)
    root1 = np.where(small_a, -c / safe_b, q / safe_a)
    root2 = np.where(small_a, -c / safe_b, c / safe_q)
    roots = np.clip(np.concatenate([root1, root2]), 0.0, np.tile(h, 2))
    return np.tile(nodes[:-1], 2) + roots


def flux_at(flux, wavel=None):
    """Evaluate a number or a spectrum at ``wavel`` (``None`` = reference)."""
    if isinstance(flux, Spectrum):
        return flux(wavel)
    return flux


def reference_flux(flux):
    """The reference flux of a number or a spectrum.

    For a spectrum this is ``flux(None)``, its value at ``wavel0`` (the
    ``ratio`` for ``PowerLaw`` and ``BlackBody``); only the deprecated
    ``Tabulated`` uses the mean over its nodes instead.
    """
    if isinstance(flux, Spectrum):
        return flux(None)
    return flux
