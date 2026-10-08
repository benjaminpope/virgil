"""Spectro-interferometric observables beside V² and closure phase.

[`OIData`][virgil.oidata.OIData] keeps its visibilities (``vis``) and
phases (``phi``) as before. The observables here are further blocks of the
same data vector, in ``OIData.extras``, read with
``read_oifits(..., extras=...)`` (see
[`read_oifits`][virgil.oifits.read_oifits]). Each block predicts its data
from the model and whitens its own residuals, so the likelihood keeps one
whitened residual vector:

* ``"flux"``: ``OI_FLUX`` as a spectrum known up to a grey scale,
  ``F(λ) = k Σ fᵢ(λ)`` from
  [`total_spectrum`][virgil.models.SourceModel.total_spectrum]
  ([`FluxSpectrum`][virgil.observables.FluxSpectrum]);
* ``"nflux"``: ``OI_FLUX`` normalized to its continuum, F(λ)/F_c(λ), the
  same with the model normalized over the continuum channels;
* ``"corrflux"``: correlated fluxes (``VISAMP`` with ``AMPTYP =
  'correlated flux'``), k |Σ fᵢ(λ) Vᵢ|, up to a grey scale;
* ``"visamp"``: |V| (``VISAMP``, ``AMPTYP = 'absolute'``) beside V²
  ([`VisibilityAmplitude`][virgil.observables.VisibilityAmplitude]);
* ``"t3amp"``: triple amplitudes |V_ab V_bc V_ac|
  ([`TripleAmplitude`][virgil.observables.TripleAmplitude]);
* ``"visphi"``: differential phases, the exact arg V(λ) with the
  pipeline's continuum normalization applied as a linear operator
  ([`DifferentialPhase`][virgil.observables.DifferentialPhase]).

**Grey scales are marginalized, not fitted.** A spectrum's scale k (and an
optional polynomial in λ times the spectrum) enters linearly, so with a
Gaussian prior it integrates out in closed form (Luger, Foreman-Mackey &
Hogg 2017; ``virgil._linear``): the data are Gaussian with covariance
``D + A Λ Aᵀ`` about ``A μ``, whitened by the same successive rank-one
steps as the 6d gains, with the log-determinant (which depends on the
model's spectral shape) kept in the effective errors. The conditional
posterior of k is
[`flux_scale_posterior`][virgil.likelihood.flux_scale_posterior].

**The prior on k is stated, never taken from the data.** Give it as
``scale=(mean, sd)`` in the data's units (e.g. Jy), with
[`OIData.with_flux_scale`][virgil.oidata.OIData.with_flux_scale]. Normalized
spectra (``"nflux"``) default to ``(1, 0.1)``. The Gaussian is a
*proposal*: k is a positive scale, whose Jeffreys prior is 1/k on stated
bounds. The two differ by about σ_k/k, which is negligible for a
well-measured spectrum. For the Jeffreys posterior, reweight samples of k
from the conditional posterior by ``1 / (k N(k; mean, sd²))`` within the
bounds, or sample log k under a log-uniform prior directly, as a model
parameter rather than marginalized. Make no evidence claims that depend
on the Gaussian's width.

**Differential phases.** A pipeline's differential phase is arg V minus a
fit of a + b/λ (an offset and a delay) over continuum channels, per
baseline and frame. That fit is a linear operator N = I − L on the phases
of one row (``continuum_operator``), so virgil applies the same N to the
exact model phase, never the photocentre approximation, which fails for
resolved structure. If our basis contains the pipeline's, N N_pipe = N, so
the data may be re-normalized safely. Their covariance is N D Nᵀ, which is
whitened as a dense block per frame. Projecting out a linear basis is the
flat-prior limit of marginalizing those nuisances: the likelihood of N d
with covariance N D Nᵀ is the same whichever projection with that null
space is used (restricted maximum likelihood).

**Without double counting closure phases** (the default of the design
note, S §2.3): with closure phases in the data, each frame's baseline
phases are also projected onto the telescope-differenced subspace
φ_ab = a_a − a_b, orthogonal to every closure, and only channels in the
line windows are kept. The closure phases (everywhere) and this closure-free
part of the differential phase (in the lines) then measure different
things. The cross-covariance between them is neglected as an approximation: it is
exactly zero only when the baseline errors of a frame and channel are equal.
For unequal errors it is nonzero, and the independent-block likelihood is
not exact. The joint covariance is preferred when available.
"""

import equinox as eqx
import jax
import jax.numpy as np
import jax.scipy.linalg as jsl
import numpy as onp

from ._deprecate import renamed
from ._linear import posterior, whiten_blocks

__all__ = [
    "DifferentialPhase",
    "FluxSpectrum",
    "KINDS",
    "TripleAmplitude",
    "VisibilityAmplitude",
    "continuum_operator",
    "in_ranges",
]

# Extra observables, in the order they follow vis and phi in the data vector.
KINDS = ("flux", "nflux", "corrflux", "visamp", "t3amp", "visphi")


def in_ranges(wavel, ranges):
    """Whether each wavelength lies in one of ``ranges``, ``[(lo, hi), ...]``."""
    wavel = onp.asarray(wavel, float)
    mask = onp.zeros(wavel.shape, dtype=bool)
    for lo, hi in ranges:
        mask |= (wavel >= lo) & (wavel <= hi)
    return mask


def _ranges(ranges):
    """``ranges`` as a static tuple of (lo, hi) pairs, or ``None``."""
    if ranges is None:
        return None
    out = tuple((float(lo), float(hi)) for lo, hi in ranges)
    for lo, hi in out:
        if not lo <= hi:
            raise ValueError(f"A wavelength range must have lo <= hi: {out}.")
    return out


def _continuum_basis(wavel, order):
    """Columns 1, x, ..., x^order, with x the scaled wavenumber 1/λ."""
    sigma = 1.0 / onp.asarray(wavel, float)
    span = float(sigma.max() - sigma.min())
    x = (sigma - sigma.mean()) / (span if span > 0.0 else 1.0)
    return onp.stack([x**j for j in range(order + 1)], axis=1)


def continuum_operator(wavel, continuum, order=1):
    """The continuum fit of one spectrum, as a matrix.

    Parameters
    ----------
    wavel : array-like
        The channels' wavelengths (metres).
    continuum : array-like of bool
        Which channels are continuum.
    order : int, optional
        0 fits a mean; 1 (the default) a mean and a slope in wavenumber
        1/λ, which is an offset and a delay for a phase.

    Returns
    -------
    numpy.ndarray
        ``L`` of shape ``(n, n)``: ``L @ x`` is the least-squares fit of the
        polynomial to ``x`` over the continuum channels, evaluated at every
        channel. A differential phase is ``(I - L) @ φ``; a normalized
        spectrum is ``F / (L @ F)``.
    """
    wavel = onp.asarray(wavel, float).reshape(-1)
    mask = onp.asarray(continuum, bool).reshape(-1)
    basis = _continuum_basis(wavel, order)
    if onp.linalg.matrix_rank(basis[mask]) < order + 1:
        raise ValueError(
            f"A continuum fit of order {order} needs at least {order + 1} "
            f"distinct continuum channels; there are {int(mask.sum())}."
        )
    op = onp.zeros((wavel.size, wavel.size))
    op[:, mask] = basis @ onp.linalg.pinv(basis[mask])
    return op


def _labels(values):
    """Integer labels 0..n-1 of the distinct rows of ``values``."""
    values = onp.asarray(values)
    if values.ndim == 1:
        values = values[:, None]
    return onp.unique(values, axis=0, return_inverse=True)[1].reshape(-1)


def _padded_groups(labels, fill):
    """``(n_group, m)`` member indices of each label, padded with ``fill``."""
    labels = onp.asarray(labels)
    n_group = int(labels.max()) + 1 if labels.size else 0
    members = [onp.flatnonzero(labels == g) for g in range(n_group)]
    m = max((len(x) for x in members), default=0)
    out = onp.full((n_group, m), fill, dtype=onp.int32)
    for g, x in enumerate(members):
        out[g, : len(x)] = x
    return out


class _Block(eqx.Module):
    """One block of extra observables (see the module notes)."""

    values: jax.Array
    errors: jax.Array

    # Whether the block's covariance depends on the model (so least squares
    # does not give its likelihood).
    model_dependent_covariance = False

    def data(self):
        """The data vector of this block."""
        return self.values

    def data_errors(self):
        """Uncertainties of :meth:`data`, for display and normalization."""
        return self.errors

    @property
    def n_independent(self):
        return int(onp.size(self.values))

    def whiten(self, prediction, data, errors):
        """Whitened residuals and their effective errors."""
        return (prediction - data) / errors, errors

    def with_errors(self, errors):
        return eqx.tree_at(lambda b: b.errors, self, np.asarray(errors))

    def simulated(self, prediction, cvis, noise, key=None):
        """These data replaced by ``prediction`` plus ``noise`` × errors.

        ``noise`` is a standard normal draw shaped like ``values`` (or
        ``None``).
        """
        values = prediction
        if noise is not None:
            values = values + self.errors * noise
        return eqx.tree_at(lambda b: b.values, self, values)


class VisibilityAmplitude(_Block):
    """|V| at samples, e.g. ``VISAMP`` beside ``VIS2DATA`` (``"visamp"``).

    V² and |V| from the same measurement are not independent; reading both
    counts it twice unless the pipeline measured them separately.
    """

    sample: onp.ndarray
    kind: str = eqx.field(static=True, default="visamp")

    @renamed()
    def predict(self, model, cvis):
        return np.abs(cvis)[self.sample]

    def subset(self, keep, new_index, flux_keep=None):
        rows = onp.flatnonzero(keep[self.sample])
        return VisibilityAmplitude(
            self.values[rows],
            self.errors[rows],
            new_index[self.sample[rows]].astype(onp.int32),
        )


class TripleAmplitude(_Block):
    """|V_ab V_bc V_ac| of closure triangles (``T3AMP``, ``"t3amp"``)."""

    i1: onp.ndarray
    i2: onp.ndarray
    i3: onp.ndarray
    kind: str = eqx.field(static=True, default="t3amp")

    @renamed()
    def predict(self, model, cvis):
        amp = np.abs(cvis)
        return amp[self.i1] * amp[self.i2] * amp[self.i3]

    def subset(self, keep, new_index, flux_keep=None):
        rows = onp.flatnonzero(keep[self.i1] & keep[self.i2] & keep[self.i3])
        legs = [new_index[i[rows]].astype(onp.int32) for i in self._legs]
        return TripleAmplitude(self.values[rows], self.errors[rows], *legs)

    @property
    def _legs(self):
        return (self.i1, self.i2, self.i3)


class FluxSpectrum(_Block):
    """A spectrum known up to a grey scale: OI_FLUX or correlated fluxes.

    The model of sample i (wavelength λᵢ, scale group gᵢ) is

        mᵢ = Σ_j w_j tᵢ xᵢʲ,   w ~ N((μ, 0, ...), diag(s, τ_1 μ, ...)²),

    with t the model's template: the total spectrum Σ fᵢ(λ)
    (``"flux"``), the same divided by its continuum fit per row
    (``"nflux"``), or the total spectrum times |V| (``"corrflux"``),
    normalized to a mean of 1 per scale group (except ``"nflux"``, which
    is already normalized). x is λ scaled to [-1, 1] across the group, so
    w_0 = k is the grey scale and w_j (j ≥ 1) an optional polynomial.
    ``(μ, s)`` is the stated prior on k, in the data's units (the same for
    every group; ``(1, 0.1)`` by default for ``"nflux"``), and τ the
    polynomial's widths relative to μ. The prior is a proposal for the
    Jeffreys 1/k (see the module notes). The weights are marginalized
    analytically.

    **Choice of prior.** k is a scale parameter. Under rescaling of k the
    invariant (Jeffreys) prior is ∝ 1/k, uniform in log k, but that prior
    is improper, so an evidence computed with it is undefined. The broad
    Gaussian used here instead approximates a prior uniform in k. The only
    claim made is local: when k is sharply measured (σ_k/k ≪ 1, as for any
    useful OI_FLUX spectrum), the factor 1/k varies by only a fraction
    σ_k/k across the likelihood's width, so the posteriors of k and of the
    other parameters are insensitive to the choice. Evidence comparisons
    need a proper prior: finite positive bounds [k_min, k_max], with density
    1 / (k ln(k_max/k_min)). With such bounds, that log-uniform prior is the
    Jeffreys choice under the rule for scale groups. Marginalizing in log k
    is not linear and is not done here (a follow-up).

    Build with :meth:`build`; change the prior, groups and widths with
    [`OIData.with_flux_scale`][virgil.oidata.OIData.with_flux_scale].
    """

    wavel: onp.ndarray
    sample: onp.ndarray | None
    row: onp.ndarray
    frame: onp.ndarray
    station: onp.ndarray
    mjd: onp.ndarray | None
    group: onp.ndarray
    members: onp.ndarray
    count: onp.ndarray
    poly: onp.ndarray
    mu: onp.ndarray | None
    cont_rows: onp.ndarray | None
    cont_basis: onp.ndarray | None
    cont_fit: onp.ndarray | None
    kind: str = eqx.field(static=True)
    widths: tuple | None = eqx.field(static=True)
    scale: tuple | None = eqx.field(static=True)
    poly_width: float = eqx.field(static=True)
    per: str = eqx.field(static=True)
    continuum: tuple | None = eqx.field(static=True)
    continuum_order: int = eqx.field(static=True)

    model_dependent_covariance = True

    @classmethod
    def build(
        cls,
        kind,
        values,
        errors,
        wavel,
        row,
        frame,
        station,
        sample=None,
        mjd=None,
        per="dataset",
        scale=None,
        poly_order=0,
        poly_width=0.1,
        continuum=None,
        continuum_order=0,
    ):
        """Build from per-sample arrays (see the class notes).

        Parameters
        ----------
        kind : {"flux", "nflux", "corrflux"}
        values, errors : array-like
            The data and their uncertainties, one per sample.
        wavel : array-like
            Each sample's wavelength (metres).
        row, frame, station : array-like
            Each sample's row (one spectrum), frame (exposure) and station
            (a telescope's ``STA_INDEX``, or a baseline's pair).
        sample : array-like of int, optional
            For ``"corrflux"``, the visibility sample of each value.
        mjd : array-like, optional
            Each sample's time (days), for splitting by epoch.
        per : {"dataset", "row", "frame", "station"}, optional
            One grey scale for all the samples (default), or one per row,
            frame or station (telescope or baseline).
        scale : (float, float), optional
            The prior ``(mean, sd)`` of the grey scale k, in the data's
            units, stated rather than taken from the data. Required for
            ``"flux"`` and ``"corrflux"`` before the likelihood is
            evaluated; ``(1, 0.1)`` by default for ``"nflux"``.
        poly_order : int, optional
            Also marginalize a polynomial in λ of this order times the
            template (default 0: a grey scale only).
        poly_width : float, optional
            Prior width of each polynomial coefficient, relative to the
            scale's prior mean.
        continuum : sequence of (lo, hi), optional
            For ``"nflux"``: the continuum ranges (metres) its model is
            normalized over, per row (default: every channel).
        continuum_order : int, optional
            For ``"nflux"``: 0 (a mean, the default) or 1 (a mean and a
            slope in wavenumber).
        """
        if kind not in ("flux", "nflux", "corrflux"):
            raise ValueError(f"Unknown flux kind {kind!r}.")
        values = onp.asarray(values, float).reshape(-1)
        errors = onp.asarray(errors, float).reshape(-1)
        wavel = onp.asarray(wavel, float).reshape(-1)
        row = _labels(row)
        frame = onp.asarray(frame).reshape(-1)
        station = onp.asarray(station)
        choices = {
            "dataset": onp.zeros(values.size, dtype=int),
            "row": row,
            "frame": frame,
            "station": station,
        }
        if per not in choices:
            raise ValueError(
                f"per must be one of {sorted(choices)}, not {per!r}."
            )
        group = _labels(choices[per]).astype(onp.int32)
        members = _padded_groups(group, values.size)
        n_group = members.shape[0]

        poly = onp.ones((values.size, poly_order + 1))
        for g in range(n_group):
            idx = members[g][members[g] < values.size]
            lo, hi = wavel[idx].min(), wavel[idx].max()
            x = (wavel[idx] - 0.5 * (lo + hi)) / max(0.5 * (hi - lo), 1e-30)
            for j in range(1, poly_order + 1):
                poly[idx, j] = x**j

        if scale is None and kind == "nflux":
            scale = (1.0, 0.1)
        mu, widths = None, None
        if scale is not None:
            mean, sd = (float(x) for x in scale)
            if not (onp.isfinite(mean) and mean > 0):
                raise ValueError(
                    f"The scale's prior mean must be positive and finite, "
                    f"not {mean}."
                )
            if not (onp.isfinite(sd) and sd > 0):
                raise ValueError(
                    f"The scale's prior sd must be positive and finite, not "
                    f"{sd} (a flat prior is not supported)."
                )
            mu = onp.full(n_group, mean)
            # Widths relative to the mean: the scale's own, then the
            # polynomial's.
            widths = (sd / mean,) + (float(poly_width),) * poly_order

        continuum = _ranges(continuum)
        cont_rows = cont_basis = cont_fit = None
        if kind == "nflux":
            cont_rows, cont_basis, cont_fit = _row_continuum(
                wavel, row, continuum, continuum_order
            )
        return cls(
            values=np.asarray(values),
            errors=np.asarray(errors),
            wavel=wavel,
            sample=None if sample is None else onp.asarray(sample, onp.int32),
            row=row,
            frame=frame,
            station=station,
            mjd=None if mjd is None else onp.asarray(mjd, onp.float64),
            group=group,
            members=members,
            count=onp.bincount(group, None, n_group).astype(float),
            poly=poly,
            mu=mu,
            cont_rows=cont_rows,
            cont_basis=cont_basis,
            cont_fit=cont_fit,
            kind=kind,
            widths=widths,
            scale=None if scale is None else (mean, sd),
            poly_width=float(poly_width),
            per=per,
            continuum=continuum,
            continuum_order=int(continuum_order),
        )

    def rebuild(self, values=None, errors=None, keep=None, **settings):
        """The same block with new settings, data, or only some samples."""
        keep = onp.ones(self.wavel.size, bool) if keep is None else keep
        rows = onp.flatnonzero(keep)
        options = dict(
            per=self.per,
            scale=self.scale,
            poly_order=self.poly.shape[1] - 1,
            poly_width=self.poly_width,
            continuum=self.continuum,
            continuum_order=self.continuum_order,
        )
        options.update(settings)
        values = self.values if values is None else values
        errors = self.errors if errors is None else errors
        return FluxSpectrum.build(
            self.kind,
            onp.asarray(values)[rows],
            onp.asarray(errors)[rows],
            self.wavel[rows],
            self.row[rows],
            self.frame[rows],
            self.station[rows],
            None if self.sample is None else self.sample[rows],
            None if self.mjd is None else self.mjd[rows],
            **options,
        )

    def subset(self, keep, new_index, flux_keep=None):
        if self.sample is not None:
            flux_keep = keep[self.sample]
            out = self.rebuild(keep=flux_keep)
            sample = new_index[self.sample[flux_keep]].astype(onp.int32)
            return eqx.tree_at(lambda b: b.sample, out, sample)
        return self.rebuild(keep=flux_keep)

    def _template(self, model, cvis):
        """The model's spectral template t (see the class notes)."""
        base = model.total_spectrum(np.asarray(self.wavel))
        base = np.broadcast_to(base, self.wavel.shape)
        if self.kind == "corrflux":
            base = base * np.abs(cvis)[self.sample]
        if self.kind == "nflux":
            spectra = base.at[self.cont_rows].get(mode="fill", fill_value=0.0)
            fit = np.asarray(self.cont_fit, base.dtype)
            basis = np.asarray(self.cont_basis, base.dtype)
            level = np.einsum("rcp,rpk,rk->rc", basis, fit, spectra)
            level = (
                np.ones_like(base).at[self.cont_rows].set(level, mode="drop")
            )
            return base / level
        n_group = self.members.shape[0]
        total = jax.ops.segment_sum(base, self.group, n_group)
        count = np.asarray(self.count, base.dtype)
        return base / (total / count)[self.group]

    def _require_prior(self):
        if self.mu is None:
            raise ValueError(
                f"State the prior on the {self.kind!r} grey scale, in the "
                "data's units: data.with_flux_scale(scale=(mean, sd)). It is "
                "not taken from the data (see virgil.observables)."
            )

    @renamed()
    def predict(self, model, cvis):
        self._require_prior()
        mu = np.asarray(self.mu)[self.group]
        return mu * self._template(model, cvis)

    def _columns(self, prediction, errors):
        """The whitened marginalized modes, ``(n, p)``: τ_j μ t xʲ / σ."""
        widths = np.asarray(self.widths, prediction.dtype)
        poly = np.asarray(self.poly, prediction.dtype)
        return prediction[:, None] * poly * widths / errors[:, None]

    def whiten(self, prediction, data, errors):
        # The residual is already about the prior mean A μ (the prediction
        # is μ t), so the weights' covariance alone is left: I + Σ w_j w_jᵀ
        # per group, whitened by the 6d gains' blocks.
        x = (prediction - data) / errors
        cols = self._columns(prediction, errors)
        local = cols.at[self.members].get(mode="fill", fill_value=0.0)
        whitened, extra = whiten_blocks(x, self.members, local)
        return whitened, errors * np.exp(extra)

    def posterior(self, prediction, data, errors):
        """Conditional posterior of the weights w per group: mean and cov.

        ``mean[g, 0]`` is the grey scale k of group g, multiplying the
        template; ``mean[g, j]`` (j ≥ 1) the polynomial coefficients. The
        Gaussian prior is a proposal for k's Jeffreys prior (see the
        module notes).
        """
        # In the standardized form of virgil._linear: x about the prior
        # mean A μ (the prediction is μ t), U the whitened columns, per
        # group; padding has zero rows, which change nothing.
        x = (data - prediction) / errors
        cols = self._columns(prediction, errors)
        x = x.at[self.members].get(mode="fill", fill_value=0.0)
        U = cols.at[self.members].get(mode="fill", fill_value=0.0)
        mean, cov = jax.vmap(posterior)(x, U)
        mu = np.asarray(self.mu, prediction.dtype)
        root = np.asarray(self.widths, prediction.dtype)[None, :] * mu[:, None]
        prior_mean = np.zeros_like(root).at[:, 0].set(mu)
        return (
            prior_mean + root * mean,
            root[:, :, None] * cov * root[:, None, :],
        )

    def simulated(self, prediction, cvis, noise, key=None):
        values = prediction
        if noise is not None:
            values = values + self.errors * noise
        if key is not None:
            widths = np.asarray(self.widths, prediction.dtype)
            columns = (
                prediction[:, None]
                * np.asarray(self.poly, prediction.dtype)
                * widths
            )
            z = jax.random.normal(
                key, (self.members.shape[0], len(self.widths))
            )
            values = values + np.sum(columns * z[self.group], axis=1)
        if isinstance(values, jax.core.Tracer):
            return eqx.tree_at(lambda b: b.values, self, values)
        return self.rebuild(values=values)


def _row_continuum(wavel, row, continuum, order):
    """Per row: padded sample indices, continuum basis and fit matrices."""
    rows = _padded_groups(row, wavel.size)
    n_row, m = rows.shape
    basis = onp.zeros((n_row, m, order + 1))
    fit = onp.zeros((n_row, order + 1, m))
    for r in range(n_row):
        idx = rows[r][rows[r] < wavel.size]
        mask = (
            onp.ones(idx.size, bool)
            if continuum is None
            else in_ranges(wavel[idx], continuum)
        )
        b = _continuum_basis(wavel[idx], order)
        if onp.linalg.matrix_rank(b[mask]) < order + 1:
            raise ValueError(
                f"A spectrum has {int(mask.sum())} continuum channels, too "
                f"few for a continuum of order {order}; widen the continuum "
                "ranges."
            )
        basis[r, : idx.size] = b
        fit[r][:, onp.flatnonzero(mask)] = onp.linalg.pinv(b[mask])
    return rows, basis, fit


class DifferentialPhase(_Block):
    """Continuum-normalized phases (``VISPHI``, ``"visphi"``).

    ``values`` and ``errors`` are the per-sample phases (radians, in the
    orientation of the visibility samples) and their uncertainties, as
    read. Per frame, the phases of each baseline (a row of channels) are
    unwrapped along wavelength and mapped by

        Y = Qᵀ Φ Wᵀ,

    where W holds the line-window rows of N = I − L (``continuum_operator``)
    rotated onto independent combinations, and Q is an orthonormal basis of
    the telescope-differenced phases (closure-free; with closure phases in
    the data) or the identity. The data vector is Y of the data, the
    covariance (Q ⊗ W) D (Q ⊗ W)ᵀ, whitened by its Cholesky factor per
    frame. Build with :meth:`build`, or change the windows with
    [`OIData.with_continuum`][virgil.oidata.OIData.with_continuum].

    With ``prior_width``, the offset and slope of every baseline and frame
    are instead marginalized under a finite Gaussian prior (the
    projection is its flat limit): W keeps the channels of both windows
    unchanged, each channel's closure-free combinations are whitened for
    their covariance Qᵀ D Q, and the offsets and slopes are whitened out as
    low-rank modes per frame, by the 6d gains' blocks
    ([`virgil.gains`][virgil.gains]). Their log-determinant does not depend
    on the model.

    Residuals are treated as linear (Gaussian), which holds while the
    differential phases are well below π, as they are after removing the
    continuum.
    """

    sample: onp.ndarray
    wavel: onp.ndarray
    frame: onp.ndarray
    stations: onp.ndarray
    row: onp.ndarray
    grid: onp.ndarray  # (F, B, C) indices into this block's samples
    chan: onp.ndarray  # (F, B, C) True for real entries
    q: onp.ndarray  # (F, K, B) baseline operator Qᵀ
    w: onp.ndarray  # (F, R, C) wavelength operator W
    valid: onp.ndarray  # (F, K, R) True for real outputs
    keep: onp.ndarray  # flat indices of the real outputs
    basis: onp.ndarray | None  # (F, R, p) continuum basis, finite prior
    continuum: tuple | None = eqx.field(static=True)
    lines: tuple | None = eqx.field(static=True)
    order: int = eqx.field(static=True)
    closure_free: bool = eqx.field(static=True)
    prior_width: tuple | None = eqx.field(static=True, default=None)
    kind: str = eqx.field(static=True, default="visphi")
    # Per frame, the Cholesky factor of the covariance (projection only):
    # it depends only on the errors, so it is computed once, where they are
    # set (build, with_errors), in NumPy float64 and cast at use.
    chol: onp.ndarray | None = None

    @classmethod
    def build(
        cls,
        values,
        errors,
        sample,
        wavel,
        frame,
        stations,
        row=None,
        continuum=None,
        lines=None,
        order=1,
        closure_free=True,
        prior_width=None,
    ):
        """Build from per-sample phases (see the class notes).

        Parameters
        ----------
        values, errors : array-like
            Phases (radians) and uncertainties, one per sample.
        sample : array-like of int
            The visibility sample of each phase.
        wavel, frame, stations : array-like
            Each phase's wavelength (metres), frame, and station pair.
        row : array-like of int, optional
            Which phases form one spectrum (one baseline of one frame);
            by default, those with the same frame and station pair.
        continuum, lines : sequence of (lo, hi), optional
            Wavelength ranges (metres). The continuum is fitted over the
            continuum channels and the line channels are kept. Either
            defaults to the complement of the other; with neither, every
            channel is both.
        order : int, optional
            1 (default): subtract a mean and a slope in wavenumber (an
            offset and a delay); 0: a mean.
        closure_free : bool, optional
            Keep only the telescope-differenced part of each frame's
            phases, orthogonal to the closure phases (default True; use it
            whenever the data have closure phases).
        prior_width : float or (float, float), optional
            Marginalize the offset (and slope, per unit of the scaled
            wavenumber, which spans 1 across the channels) of each
            baseline and frame under Gaussian priors of these widths
            (radians), over the channels of both windows, instead of
            projecting them out (``None``, the default, as the pipeline
            does).
        """
        values = onp.asarray(values, float).reshape(-1)
        errors = onp.asarray(errors, float).reshape(-1)
        sample = onp.asarray(sample, onp.int32).reshape(-1)
        wavel = onp.asarray(wavel, float).reshape(-1)
        frame = onp.asarray(frame).reshape(-1)
        stations = onp.asarray(stations, int).reshape(-1, 2)
        if row is None:
            row = _labels(onp.column_stack([frame, stations]))
        row = _labels(row)
        continuum, lines = _ranges(continuum), _ranges(lines)
        if prior_width is not None:
            widths = onp.broadcast_to(
                onp.asarray(prior_width, float), (order + 1,)
            )
            if not onp.all(widths > 0):
                raise ValueError("prior_width must be positive.")
            prior_width = tuple(float(x) for x in widths)

        frames = []
        for f in onp.unique(frame):
            in_frame = onp.flatnonzero(frame == f)
            rows = onp.unique(row[in_frame])
            channels = None
            for r in rows:
                w_r = wavel[row == r]
                if onp.unique(w_r).size != w_r.size:
                    raise ValueError(
                        "A differential-phase spectrum has two values at one "
                        "wavelength; give each spectrum its own row."
                    )
                channels = (
                    w_r if channels is None else onp.intersect1d(channels, w_r)
                )
            channels = onp.sort(channels)
            if channels.size == 0:
                continue
            cont, line = _windows(channels, continuum, lines)
            if not line.any():
                continue
            if prior_width is None:
                n_op = continuum_operator(channels, cont, order)
                m = (onp.eye(channels.size) - n_op)[line]
                u, s, _ = onp.linalg.svd(m, full_matrices=False)
                rank = int(onp.sum(s > 1e-9 * s.max())) if s.size else 0
                if rank == 0:
                    continue
                w_op = u[:, :rank].T @ m
                basis_f = None
            else:
                used = cont | line
                w_op = onp.eye(channels.size)[used]
                basis_f = _continuum_basis(channels[used], order)
            grid = onp.empty((rows.size, channels.size), dtype=int)
            pairs = onp.empty((rows.size, 2), dtype=int)
            for b, r in enumerate(rows):
                idx = onp.flatnonzero(row == r)
                order_in_row = onp.argsort(wavel[idx])
                idx = idx[order_in_row]
                pos = onp.searchsorted(wavel[idx], channels)
                grid[b] = idx[pos]
                pairs[b] = stations[idx[0]]
            q_op = _baseline_basis(pairs) if closure_free else None
            if q_op is None:
                q_op = onp.eye(rows.size)
            frames.append((grid, q_op, w_op, basis_f))

        n_f = len(frames)
        b_max = max((x[0].shape[0] for x in frames), default=0)
        c_max = max((x[0].shape[1] for x in frames), default=0)
        k_max = max((x[1].shape[0] for x in frames), default=0)
        r_max = max((x[2].shape[0] for x in frames), default=0)
        grid = onp.zeros((n_f, b_max, c_max), dtype=onp.int32)
        chan = onp.zeros((n_f, b_max, c_max), dtype=bool)
        q = onp.zeros((n_f, k_max, b_max))
        w = onp.zeros((n_f, r_max, c_max))
        valid = onp.zeros((n_f, k_max, r_max), dtype=bool)
        basis = None
        if prior_width is not None:
            basis = onp.zeros((n_f, r_max, order + 1))
        for f, (g, q_op, w_op, basis_f) in enumerate(frames):
            nb, nc = g.shape
            # Padding repeats a real sample of the same row, so that the
            # unwrapped phase stays constant there (W is zero there anyway).
            grid[f] = g[0, -1]
            grid[f, :nb, :nc] = g
            grid[f, :nb, nc:] = g[:, -1:]
            chan[f, :nb, :nc] = True
            q[f, : q_op.shape[0], :nb] = q_op
            w[f, : w_op.shape[0], :nc] = w_op
            valid[f, : q_op.shape[0], : w_op.shape[0]] = True
            if basis is not None:
                basis[f, : w_op.shape[0]] = basis_f
        keep = onp.flatnonzero(valid.reshape(-1)).astype(onp.int32)
        out = cls(
            values=np.asarray(values),
            errors=np.asarray(errors),
            sample=sample,
            wavel=wavel,
            frame=frame,
            stations=stations,
            row=row,
            grid=grid,
            chan=chan,
            q=q,
            w=w,
            valid=valid,
            keep=keep,
            basis=basis,
            continuum=continuum,
            lines=lines,
            order=int(order),
            closure_free=bool(closure_free),
            prior_width=prior_width,
        )
        return out._with_cholesky()

    def rebuild(self, values=None, errors=None, keep=None, **settings):
        """The same block with new settings, data, or only some samples."""
        keep = onp.ones(self.sample.size, bool) if keep is None else keep
        rows = onp.flatnonzero(keep)
        options = dict(
            continuum=self.continuum,
            lines=self.lines,
            order=self.order,
            closure_free=self.closure_free,
            prior_width=self.prior_width,
        )
        options.update(settings)
        values = self.values if values is None else values
        errors = self.errors if errors is None else errors
        return DifferentialPhase.build(
            onp.asarray(values)[rows],
            onp.asarray(errors)[rows],
            self.sample[rows],
            self.wavel[rows],
            self.frame[rows],
            self.stations[rows],
            self.row[rows],
            **options,
        )

    def subset(self, keep, new_index, flux_keep=None):
        kept = keep[self.sample]
        out = self.rebuild(keep=kept)
        sample = new_index[self.sample[kept]].astype(onp.int32)
        return eqx.tree_at(lambda b: b.sample, out, sample)

    @property
    def n_independent(self):
        return int(self.keep.size)

    def _project(self, phases):
        """Y = Qᵀ Φ Wᵀ per frame, flattened to the real outputs."""
        q = np.asarray(self.q, phases.dtype)
        w = np.asarray(self.w, phases.dtype)
        y = np.einsum("fkb,fbc,frc->fkr", q, phases, w)
        return y.reshape(-1)[self.keep]

    @staticmethod
    def _unwrap(steps, anchor=None):
        """Phases along the last axis from their wrapped steps."""
        if anchor is None:
            anchor = np.zeros(steps.shape[:-1] + (1,), steps.dtype)
        return np.concatenate(
            [anchor, anchor + np.cumsum(steps, axis=-1)], axis=-1
        )

    def data(self):
        phases = np.asarray(self.values)[self.grid]
        steps = np.diff(phases, axis=-1)
        steps = np.mod(steps + np.pi, 2.0 * np.pi) - np.pi
        anchor = np.arctan2(np.sin(phases[..., :1]), np.cos(phases[..., :1]))
        return self._project(self._unwrap(steps, anchor))

    @renamed()
    def predict(self, model, cvis):
        vis = np.asarray(cvis)[self.sample][self.grid]
        phases = np.angle(vis)
        steps = np.angle(vis[..., 1:] * np.conj(vis[..., :-1]))
        data_phase = np.asarray(self.values)[self.grid]
        data_anchor = np.arctan2(
            np.sin(data_phase[..., :1]), np.cos(data_phase[..., :1])
        )
        anchor_delta = np.angle(np.exp(1j * (phases[..., :1] - data_anchor)))
        return self._project(self._unwrap(steps, data_anchor + anchor_delta))

    def covariance(self, errors=None):
        """Per frame, the covariance of the outputs, ``(F, K R, K R)``.

        Padded outputs get unit variance, so the matrices are invertible.
        """
        errors = self.errors if errors is None else errors
        errors = np.asarray(errors)
        var = np.where(self.chan, errors[self.grid] ** 2, 0.0)
        q = np.asarray(self.q, var.dtype)
        w = np.asarray(self.w, var.dtype)
        a = np.einsum("frc,fsc,fbc->fbrs", w, w, var)
        cov = np.einsum("fkb,flb,fbrs->fkrls", q, q, a)
        n_f, k, r = self.valid.shape
        cov = cov.reshape(n_f, k * r, k * r)
        pad = 1.0 - self.valid.reshape(n_f, k * r).astype(cov.dtype)
        return cov + pad[:, :, None] * np.eye(k * r, dtype=cov.dtype)

    def _with_cholesky(self):
        """This block with its covariance's Cholesky factor cached.

        Only for the projection (the finite prior whitens otherwise), and only
        with concrete errors: traced ones (e.g. errors set inside ``jit``)
        leave it to :meth:`whiten`.
        """
        chol = None
        if self.prior_width is None:
            try:
                errors = onp.asarray(self.errors, dtype=onp.float64)
            except jax.errors.TracerArrayConversionError:
                errors = None
            if errors is not None:
                # In NumPy float64 throughout, so a later float64 fit gets a
                # factor as precise as one it would build itself.
                try:
                    chol = onp.linalg.cholesky(self._covariance64(errors))
                except onp.linalg.LinAlgError:
                    chol = None  # leave it to whiten, as before
        return eqx.tree_at(
            lambda b: b.chol, self, chol, is_leaf=lambda x: x is None
        )

    def _covariance64(self, errors):
        """:meth:`covariance` in NumPy float64, for the cached factor."""
        var = onp.where(self.chan, errors[self.grid] ** 2, 0.0)
        q, w = (onp.asarray(x, onp.float64) for x in (self.q, self.w))
        a = onp.einsum("frc,fsc,fbc->fbrs", w, w, var)
        cov = onp.einsum("fkb,flb,fbrs->fkrls", q, q, a)
        n_f, k, r = self.valid.shape
        cov = cov.reshape(n_f, k * r, k * r)
        pad = 1.0 - onp.asarray(self.valid).reshape(n_f, k * r)
        return cov + pad[:, :, None] * onp.eye(k * r)

    def with_errors(self, errors):
        return super().with_errors(errors)._with_cholesky()

    def data_errors(self):
        cov = self.covariance()
        return np.sqrt(np.diagonal(cov, axis1=1, axis2=2)).reshape(-1)[
            self.keep
        ]

    def whiten(self, prediction, data, errors):
        # ``errors`` (the propagated diagonal) is not used: the block is
        # whitened with its full covariance from the per-sample errors.
        if self.prior_width is not None:
            return self._whiten_with_prior(prediction - data)
        n_f, k, r = self.valid.shape
        resid = np.zeros(n_f * k * r, prediction.dtype)
        resid = resid.at[self.keep].set(prediction - data)
        if self.chol is None:
            chol = np.linalg.cholesky(self.covariance())
        else:
            chol = np.asarray(self.chol, resid.dtype)
        white = jsl.solve_triangular(
            chol, resid.reshape(n_f, k * r, 1), lower=True
        )
        effective = np.diagonal(chol, axis1=1, axis2=2)
        return (
            white.reshape(-1)[self.keep],
            effective.reshape(-1)[self.keep],
        )

    def _whiten_with_prior(self, resid):
        """Whitened residuals with offsets and slopes marginalized."""
        n_f, k, r = self.valid.shape
        y = np.zeros(n_f * k * r, resid.dtype).at[self.keep].set(resid)
        y = y.reshape(n_f, k, r).transpose(0, 2, 1)  # (F, R, K)
        errors = np.asarray(self.errors)
        var = np.where(self.chan, errors[self.grid] ** 2, 0.0)
        q = np.asarray(self.q, var.dtype)
        w = np.asarray(self.w, var.dtype)
        var = np.einsum("frc,fbc->fbr", w**2, var)  # W selects channels
        valid = np.asarray(self.valid).transpose(0, 2, 1)  # (F, R, K)
        pad = 1.0 - valid.astype(var.dtype)
        # Each channel's closure-free combinations: covariance Qᵀ D Q.
        cov = np.einsum("fkb,flb,fbr->frkl", q, q, var)
        cov = cov + pad[..., None] * np.eye(k, dtype=var.dtype)
        chol = np.linalg.cholesky(cov)
        x = jsl.solve_triangular(chol, y[..., None], lower=True)[..., 0]
        # Offset and slope of each baseline: Qᵀ e_b times the basis, through
        # the same whitening; degenerate (closure) directions are harmless.
        q_cols = np.broadcast_to(q[:, None], (n_f, r, k, q.shape[2]))
        q_cols = jsl.solve_triangular(chol, q_cols, lower=True)
        widths = np.asarray(self.prior_width, var.dtype)
        basis = np.asarray(self.basis, var.dtype) * widths
        local = np.einsum("frkb,frj->frkbj", q_cols, basis)
        local = np.where(valid[..., None, None], local, 0.0)
        local = local.reshape(n_f, r * k, -1)
        n = n_f * r * k
        flat = np.arange(n).reshape(n_f, r * k)
        rows = np.where(valid.reshape(n_f, r * k), flat, n)
        white, extra = whiten_blocks(x.reshape(-1), rows, local)
        effective = np.diagonal(chol, axis1=2, axis2=3).reshape(-1)
        effective = effective * np.exp(extra)

        def in_output_order(a):
            return a.reshape(n_f, r, k).transpose(0, 2, 1).reshape(-1)

        return (
            in_output_order(white)[self.keep],
            in_output_order(effective)[self.keep],
        )

    def simulated(self, prediction, cvis, noise, key=None):
        values = np.angle(np.asarray(cvis)[self.sample])
        if noise is not None:
            values = values + self.errors * noise
        if key is not None and self.prior_width is not None:
            n_f, n_b, _ = self.grid.shape
            n_mode = len(self.prior_width)
            latent = jax.random.normal(key, (n_f, n_b, n_mode))
            widths = np.asarray(self.prior_width, values.dtype)
            modes = np.einsum(
                "fbp,frp->fbr",
                latent,
                np.asarray(self.basis, values.dtype) * widths,
            )
            channels = onp.argmax(onp.asarray(self.w), axis=-1)
            for f in range(n_f):
                baseline = onp.flatnonzero(
                    onp.asarray(self.chan[f]).any(axis=-1)
                )
                for b in baseline:
                    channel = channels[f, : self.basis.shape[1]]
                    samples = self.grid[f, b, channel]
                    values = values.at[samples].add(modes[f, b])
        return eqx.tree_at(lambda b: b.values, self, values)


def _windows(channels, continuum, lines):
    """Continuum and line masks of a frame's channels (see ``build``)."""
    if continuum is None and lines is None:
        everything = onp.ones(channels.size, dtype=bool)
        return everything, everything
    if lines is None:
        cont = in_ranges(channels, continuum)
        return cont, ~cont
    line = in_ranges(channels, lines)
    if continuum is None:
        return ~line, line
    return in_ranges(channels, continuum), line


def _baseline_basis(pairs):
    """Qᵀ: an orthonormal basis of the telescope-differenced phases.

    Row b of the incidence matrix A has +1 at telescope a and -1 at
    telescope c for baseline (a, c), so A t are the phases a_a - a_c of
    telescope phases t; its column space is orthogonal to every closure.
    """
    telescopes = onp.unique(pairs)
    col = {t: j for j, t in enumerate(telescopes)}
    a = onp.zeros((len(pairs), telescopes.size))
    for b, (i, j) in enumerate(pairs):
        a[b, col[i]] += 1.0
        a[b, col[j]] -= 1.0
    u, s, _ = onp.linalg.svd(a, full_matrices=False)
    rank = int(onp.sum(s > 1e-9 * s.max()))
    return u[:, :rank].T
