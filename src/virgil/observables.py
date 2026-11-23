"""Spectro-interferometric observables beside V² and closure phase.

[`OIData`][virgil.oidata.OIData] keeps its visibilities (``vis``) and
phases (``phi``) as before. The observables here are further blocks of the
same data vector, in ``OIData.extras``, read with
[`read_oifits`][virgil.oifits.read_oifits]``(..., extras=...)``. Each block
predicts its data and whitens its own residuals:

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

Grey scales are marginalized, not fitted, under a prior you state; the
differential phases apply the pipeline's continuum normalization to the exact
model phase. See [spectro-interferometric observables](../spectro_observables.md).
"""

import functools

import equinox as eqx
import jax
import jax.numpy as np
import jax.scipy.linalg as jsl
import numpy as onp

from ._deprecate import renamed
from ._linear import arrow_problem, posterior, whiten_blocks
from ._utils import group_indices, wrap_phase

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


def _colspace(mat):
    """Orthonormal rows spanning the column space of ``mat``."""
    u, s, _ = onp.linalg.svd(mat, full_matrices=False)
    rank = int(onp.sum(s > 1e-9 * s.max())) if s.size else 0
    return u[:, :rank].T


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
    counts = onp.bincount(labels, minlength=n_group)
    order = onp.argsort(labels, kind="stable")
    first = onp.cumsum(counts) - counts
    out = onp.full((n_group, counts.max(initial=0)), fill, dtype=onp.int32)
    out[labels[order], onp.arange(labels.size) - first[labels[order]]] = order
    return out


def _check_modes(modes, n_sample, label):
    """Validate ``(n_mode, n_sample)`` mode array; return float array or None."""
    if modes is None:
        return None
    arr = onp.asarray(modes, float)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.ndim != 2 or arr.shape[1] != n_sample:
        raise ValueError(
            f"modes has shape {arr.shape} for {n_sample} {label} samples; "
            f"give (n_mode, {n_sample})."
        )
    if not onp.all(onp.isfinite(arr)):
        raise ValueError("modes must be finite.")
    return None if arr.size == 0 else arr


def _spectral_inputs(values, errors, wavel, frame, sample, modes, label):
    """Flat ``(values, errors, wavel, frame, sample)`` and validated ``modes``."""
    values = onp.asarray(values, float).reshape(-1)
    errors = onp.asarray(errors, float).reshape(-1)
    wavel = onp.asarray(wavel, float).reshape(-1)
    frame = onp.asarray(frame).reshape(-1)
    if sample is not None:
        sample = onp.asarray(sample, onp.int32).reshape(-1)
    return (
        values,
        errors,
        wavel,
        frame,
        sample,
        _check_modes(modes, values.size, label),
    )


def _partition_mode_indices(touched):
    """Split ``(n_mode, n_group)`` bool touch mask into local/spanning indices."""
    if touched is None or touched.shape[1] == 0:
        return None
    n_mode, n_group = touched.shape
    per_g = [[] for _ in range(n_group)]
    span = []
    for m_idx in range(n_mode):
        gs = onp.flatnonzero(touched[m_idx])
        if gs.size == 1:
            per_g[int(gs[0])].append(m_idx)
        elif gs.size > 1:
            span.append(m_idx)
    max_loc = max((len(v) for v in per_g), default=0)
    loc_idx = (
        tuple(tuple(v + [n_mode] * (max_loc - len(v))) for v in per_g)
        if max_loc > 0
        else ()
    )
    return loc_idx, tuple(span)


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

    def gain_legs(self):
        """Sample index arrays contributing ``+δ log |V|`` to each row, or ``()``."""
        return ()

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

    def gain_legs(self):
        return (self.sample,)

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

    def gain_legs(self):
        return (self.i1, self.i2, self.i3)

    @renamed()
    def predict(self, model, cvis):
        amp = np.abs(cvis)
        return amp[self.i1] * amp[self.i2] * amp[self.i3]

    def subset(self, keep, new_index, flux_keep=None):
        if onp.any(keep[self.i1] != keep[self.i2]) or onp.any(
            keep[self.i1] != keep[self.i3]
        ):
            raise ValueError(
                "A triple amplitude would be split between parts."
            )
        rows = onp.flatnonzero(keep[self.i1])
        legs = [new_index[i[rows]].astype(onp.int32) for i in self.gain_legs()]
        return TripleAmplitude(self.values[rows], self.errors[rows], *legs)


class FluxSpectrum(_Block):
    """A spectrum known up to a grey scale: OI_FLUX or correlated fluxes.

    The model of sample i (wavelength λᵢ, scale group gᵢ) is

        mᵢ = Σ_j w_j tᵢ xᵢʲ,   w ~ N((μ, 0, ...), diag(s, τ_1 μ, ...)²),

    with t the model's template: the total spectrum Σ fᵢ(λ) (``"flux"``),
    the same divided by its continuum fit per row (``"nflux"``), or the total
    spectrum times |V| (``"corrflux"``), normalized to a mean of 1 per scale
    group (except ``"nflux"``). x is λ scaled to [-1, 1] across the group, so
    w_0 = k is the grey scale and w_j (j ≥ 1) an optional polynomial.
    ``(μ, s)`` is the stated prior on k in the data's units (``(1, 0.1)`` by
    default for ``"nflux"``), τ the polynomial's relative widths. The Gaussian
    is a proposal for the Jeffreys 1/k, which is improper: evidence
    comparisons need a proper prior on bounded k (see
    [the grey scale](../spectro_observables.md#spectra-and-the-grey-scale)).
    The weights are marginalized analytically.

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
    widths: onp.ndarray | None  # prior widths of the weights, relative to mu
    scale: onp.ndarray | None  # the stated (mean, sd) of the grey scale
    poly_width: onp.ndarray  # scalar, kept to rebuild the block
    per: str = eqx.field(static=True)
    continuum: tuple | None = eqx.field(static=True)
    continuum_order: int = eqx.field(static=True)
    modes: onp.ndarray | None = None
    local_modes: onp.ndarray | None = None
    spanning_modes: onp.ndarray | None = None

    model_dependent_covariance = True

    def gain_legs(self):
        return (self.sample,) if self.kind == "corrflux" else ()

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
        modes=None,
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
        modes : array-like, optional
            Linear nuisance modes ``(n_mode, n_sample)`` in ``d log F`` per
            1σ for ``"flux"`` or ``"corrflux"`` (refused on ``"nflux"``;
            Luger, Foreman-Mackey & Hogg 2017).
        """
        if kind not in ("flux", "nflux", "corrflux"):
            raise ValueError(f"Unknown flux kind {kind!r}.")
        if kind == "nflux" and modes is not None:
            raise ValueError(
                "Nuisance modes are not supported for continuum-normalized "
                "'nflux' spectra."
            )
        values, errors, wavel, frame, sample, modes = _spectral_inputs(
            values, errors, wavel, frame, sample, modes, kind
        )
        row = _labels(row)
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
            widths = onp.array([sd / mean] + [float(poly_width)] * poly_order)

        continuum = _ranges(continuum)
        cont_rows = cont_basis = cont_fit = None
        if kind == "nflux":
            # per row: padded sample indices, continuum basis and fit matrices
            c_rows = _padded_groups(row, wavel.size)
            c_n, c_m = c_rows.shape
            c_basis = onp.zeros((c_n, c_m, continuum_order + 1))
            c_fit = onp.zeros((c_n, continuum_order + 1, c_m))
            for r in range(c_n):
                idx = c_rows[r][c_rows[r] < wavel.size]
                mask = (
                    onp.ones(idx.size, bool)
                    if continuum is None
                    else in_ranges(wavel[idx], continuum)
                )
                b = _continuum_basis(wavel[idx], continuum_order)
                if onp.linalg.matrix_rank(b[mask]) < continuum_order + 1:
                    raise ValueError(
                        f"A spectrum has {int(mask.sum())} continuum "
                        "channels, too few for a continuum of order "
                        f"{continuum_order}; widen the continuum ranges."
                    )
                c_basis[r, : idx.size] = b
                c_fit[r][:, onp.flatnonzero(mask)] = onp.linalg.pinv(b[mask])
            cont_rows, cont_basis, cont_fit = c_rows, c_basis, c_fit
        local_modes = spanning_modes = None
        if modes is not None and n_group > 0:
            m_pad = onp.pad(modes, ((0, 1), (0, 1)))
            loc_idx, span_idx = _partition_mode_indices(
                onp.any(m_pad[:-1, members] != 0, axis=2)
            )
            if loc_idx:
                idx_m = onp.asarray(loc_idx)[:, None, :]
                local_modes = m_pad[idx_m, members[:, :, None]]
            if span_idx:
                spanning_modes = modes[list(span_idx)].T
        return cls(
            values=np.asarray(values),
            errors=np.asarray(errors),
            wavel=wavel,
            sample=sample,
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
            scale=None if scale is None else onp.array([mean, sd]),
            poly_width=onp.array(float(poly_width)),
            per=per,
            continuum=continuum,
            continuum_order=int(continuum_order),
            modes=modes,
            local_modes=local_modes,
            spanning_modes=spanning_modes,
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
            modes=None if self.modes is None else self.modes[:, rows],
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

    @renamed()
    def predict(self, model, cvis):
        if self.mu is None:
            raise ValueError(
                f"State the prior on the {self.kind!r} grey scale, in the "
                "data's units: data.with_flux_scale(scale=(mean, sd)). It is "
                "not taken from the data (see virgil.observables)."
            )
        mu = np.asarray(self.mu)[self.group]
        # the model's spectral template t (see the class notes)
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
            return mu * (base / level)
        n_group = self.members.shape[0]
        total = jax.ops.segment_sum(base, self.group, n_group)
        count = np.asarray(self.count, base.dtype)
        return mu * (base / (total / count)[self.group])

    def _columns(self, prediction, errors):
        """The whitened marginalized modes, ``(n, p)``: τ_j μ t xʲ / σ."""
        widths = np.asarray(self.widths, prediction.dtype)
        poly = np.asarray(self.poly, prediction.dtype)
        return prediction[:, None] * poly * widths / errors[:, None]

    def whiten(self, prediction, data, errors):
        # The residual is already about the prior mean A μ (the prediction
        # is μ t), so the weights' covariance alone is left: I + Σ w_j w_jᵀ
        # per group, plus any supplied local or spanning modes.
        x = (prediction - data) / errors
        cols = self._columns(prediction, errors)
        local = cols.at[self.members].get(mode="fill", fill_value=0.0)
        scale = prediction / errors
        if self.local_modes is not None:
            scale_g = scale.at[self.members].get(mode="fill", fill_value=0.0)
            m_loc = scale_g[..., None] * np.asarray(
                self.local_modes, prediction.dtype
            )
            local = np.concatenate([local, m_loc], axis=-1)
        spanning = None
        if self.spanning_modes is not None:
            spanning = scale[:, None] * np.asarray(
                self.spanning_modes, prediction.dtype
            )
        whitened, extra = whiten_blocks(x, self.members, local, spanning)
        return whitened, errors * np.exp(extra)

    def posterior(self, prediction, data, errors, *, coupled=False):
        """Conditional posterior of the weights w per group: mean and cov.

        ``mean[g, 0]`` is the grey scale k of group g, multiplying the
        template; ``mean[g, j]`` (j ≥ 1) the polynomial coefficients. The
        Gaussian prior is a proposal for k's Jeffreys prior (see the
        module notes).
        """
        if self.kind == "corrflux" and coupled:
            raise NotImplementedError(
                "Conditional flux-scale posterior for 'corrflux' is not "
                "available when coupled to OIData.gains."
            )
        x = (data - prediction) / errors
        cols = self._columns(prediction, errors)
        if self.modes is None:
            x_g = x.at[self.members].get(mode="fill", fill_value=0.0)
            u_g = cols.at[self.members].get(mode="fill", fill_value=0.0)
            mean, cov = jax.vmap(posterior)(x_g, u_g)
        else:
            n_group, p = self.members.shape[0], self.poly.shape[1]
            in_g = (np.arange(n_group)[None, :] == self.group[:, None]).astype(
                cols.dtype
            )
            u_poly = (in_g[:, :, None] * cols[:, None, :]).reshape(
                cols.shape[0], n_group * p
            )
            u_modes = (prediction / errors)[:, None] * np.asarray(
                self.modes, cols.dtype
            ).T
            mean_all, cov_all = posterior(
                x, np.concatenate([u_poly, u_modes], axis=1)
            )
            mean = mean_all[: n_group * p].reshape(n_group, p)
            cov4 = cov_all[: n_group * p, : n_group * p].reshape(
                n_group, p, n_group, p
            )
            cov = np.diagonal(cov4, axis1=0, axis2=2).transpose(2, 0, 1)
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
            poly_key = key
            mode_key = None
            if self.modes is not None:
                poly_key, mode_key = jax.random.split(key)
            if self.widths is not None:
                widths = np.asarray(self.widths, prediction.dtype)
                columns = (
                    prediction[:, None]
                    * np.asarray(self.poly, prediction.dtype)
                    * widths
                )
                z = jax.random.normal(
                    poly_key, (self.members.shape[0], len(self.widths))
                )
                values = values + np.sum(columns * z[self.group], axis=1)
            if mode_key is not None:
                modes = np.asarray(self.modes, prediction.dtype)
                z_m = jax.random.normal(mode_key, (modes.shape[0],))
                values = values + prediction * (modes.T @ z_m)
        if isinstance(values, jax.core.Tracer):
            return eqx.tree_at(lambda b: b.values, self, values)
        return self.rebuild(values=values)

    def labels_shared(self, labels):
        """Whether any marginalized scale group or mode spans ``labels``."""
        labels = onp.asarray(labels)
        if self.widths is not None:
            for m in self.members:
                idx = m[m < self.wavel.size]
                if onp.unique(labels[idx]).size > 1:
                    return True
        if self.modes is not None:
            for mode in onp.asarray(self.modes):
                if onp.unique(labels[mode != 0]).size > 1:
                    return True
        return False


class DifferentialPhase(_Block):
    """Continuum-normalized phases (``VISPHI``, ``"visphi"``).

    ``values`` and ``errors`` are the per-sample phases (radians) and their
    uncertainties. Per frame, each baseline's phases are unwrapped along
    wavelength and mapped by ``Y = Qᵀ Φ Wᵀ``, where W holds the line-window
    rows of N = I − L (``continuum_operator``) and Q is an orthonormal basis
    of the telescope-differenced phases (closure-free, with closure phases in
    the data) or the identity. The covariance is whitened by its Cholesky
    factor per frame. With ``prior_width``, the offsets and slopes are
    instead marginalized under a finite Gaussian prior (the projection is its
    flat limit), using the low-rank blocks of [`virgil.gains`][virgil.gains].
    Residuals are treated as linear, which holds while the phases are well
    below π. See
    [differential phases](../spectro_observables.md#differential-phases).
    Build with :meth:`build`, or change the windows with
    [`OIData.with_continuum`][virgil.oidata.OIData.with_continuum].
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
    prior_width: onp.ndarray | None = None
    modes: onp.ndarray | None = None
    _mode_plan: tuple | None = eqx.field(static=True, default=None)
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
        modes=None,
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
        modes : array-like, optional
            Linear tangent-space nuisance modes ``(n_mode, n_sample)`` in
            radians per 1σ (Luger, Foreman-Mackey & Hogg 2017).
        """
        values, errors, wavel, frame, sample, modes = _spectral_inputs(
            values, errors, wavel, frame, sample, modes, "visphi"
        )
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
            prior_width = onp.array(widths, float)

        # The samples of each row, ascending (one stable sort, not a pass
        # over all samples per row and frame).
        by_row = dict(zip(onp.unique(row), group_indices(row)))
        if len(set(zip(row.tolist(), wavel.tolist()))) != row.size:
            raise ValueError(
                "A differential-phase spectrum has two values at one "
                "wavelength; give each spectrum its own row."
            )
        frames, units = [], []  # units: rows coupled through shared channels
        for in_frame in group_indices(frame):
            rows = onp.unique(row[in_frame])
            chans = [onp.sort(wavel[by_row[r]]) for r in rows]
            shared = functools.reduce(onp.intersect1d, chans)
            if closure_free or all(c.size == shared.size for c in chans):
                units.append((rows, shared))
            else:
                units += [(rows[[k]], c) for k, c in enumerate(chans)]
        for rows, channels in units:
            if channels.size == 0:
                continue
            # continuum and line masks of the frame's channels
            if continuum is None and lines is None:
                cont = line = onp.ones(channels.size, dtype=bool)
            elif lines is None:
                cont = in_ranges(channels, continuum)
                line = ~cont
            else:
                line = in_ranges(channels, lines)
                cont = (
                    ~line
                    if continuum is None
                    else in_ranges(channels, continuum)
                )
            if not line.any():
                continue
            if prior_width is None:
                n_op = continuum_operator(channels, cont, order)
                m = (onp.eye(channels.size) - n_op)[line]
                u_t = _colspace(m)
                if u_t.shape[0] == 0:
                    continue
                w_op = u_t @ m
                basis_f = None
            else:
                used = cont | line
                w_op = onp.eye(channels.size)[used]
                basis_f = _continuum_basis(channels[used], order)
            grid = onp.empty((rows.size, channels.size), dtype=int)
            pairs = onp.empty((rows.size, 2), dtype=int)
            for b, r in enumerate(rows):
                idx = by_row[r]
                order_in_row = onp.argsort(wavel[idx])
                idx = idx[order_in_row]
                pos = onp.searchsorted(wavel[idx], channels)
                grid[b] = idx[pos]
                pairs[b] = stations[idx[0]]
            if closure_free:
                # Qᵀ: an orthonormal basis of the telescope-differenced
                # phases (the column space of the baseline-telescope
                # incidence matrix)
                telescopes = onp.unique(pairs)
                col = {tel: j for j, tel in enumerate(telescopes)}
                inc = onp.zeros((len(pairs), telescopes.size))
                for b, (i, j) in enumerate(pairs):
                    inc[b, col[i]] += 1.0
                    inc[b, col[j]] -= 1.0
                q_op = _colspace(inc)
            else:
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
        t_on = None
        if modes is not None:
            t_on = onp.any(onp.where(chan, modes[:, grid], 0.0) != 0, (2, 3))
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
            modes=modes,
            _mode_plan=_partition_mode_indices(t_on),
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
            modes=None if self.modes is None else self.modes[:, rows],
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

    def _data_steps(self):
        """The data's wrapped steps and anchor, ``[-π, π)``."""
        phases = np.asarray(self.values)[self.grid]
        steps = wrap_phase(np.diff(phases, axis=-1))
        return steps, wrap_phase(phases[..., :1])

    def data(self):
        return self._project(self._unwrap(*self._data_steps()))

    @renamed()
    def predict(self, model, cvis):
        vis = np.asarray(cvis)[self.sample][self.grid]
        steps = np.angle(vis[..., 1:] * np.conj(vis[..., :-1]))
        anchor = np.angle(vis[..., :1])
        # The model's steps and anchor are wrapped to within π of the
        # data's, so a step near ±π lands on the data's side of the cut
        # whatever the last bit of arctan2 or π.
        data_steps, data_anchor = self._data_steps()
        steps = data_steps + wrap_phase(steps - data_steps)
        anchor = data_anchor + wrap_phase(anchor - data_anchor)
        return self._project(self._unwrap(steps, anchor))

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

        Only for the projection, and only with concrete (untraced) errors.
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
                    # covariance in NumPy float64, for the cached factor
                    var = onp.where(self.chan, errors[self.grid] ** 2, 0.0)
                    q, w = (
                        onp.asarray(x, onp.float64) for x in (self.q, self.w)
                    )
                    a = onp.einsum("frc,fsc,fbc->fbrs", w, w, var)
                    cov = onp.einsum("fkb,flb,fbrs->fkrls", q, q, a)
                    n_f, k, r = self.valid.shape
                    cov = cov.reshape(n_f, k * r, k * r)
                    pad = 1.0 - onp.asarray(self.valid).reshape(n_f, k * r)
                    chol = onp.linalg.cholesky(
                        cov + pad[:, :, None] * onp.eye(k * r)
                    )
                except onp.linalg.LinAlgError:
                    chol = None  # leave it to whiten, as before
        return eqx.tree_at(
            lambda b: b.chol, self, chol, is_leaf=lambda x: x is None
        )

    def with_errors(self, errors):
        return super().with_errors(errors)._with_cholesky()

    def data_errors(self):
        cov = self.covariance()
        return np.sqrt(np.diagonal(cov, axis1=1, axis2=2)).reshape(-1)[
            self.keep
        ]

    def _whitened_diffphase(self, resid):
        """White-noise pre-whitened residual, effective errors, and low-rank modes."""
        n_f, k, r = self.valid.shape
        n_keep = self.keep.size
        m_proj = None
        if self.modes is not None:
            m_raw = np.asarray(self.modes, resid.dtype)
            m_grid = np.where(self.chan, m_raw[:, self.grid], 0.0)
            q_m = np.asarray(self.q, resid.dtype)
            w_m = np.asarray(self.w, resid.dtype)
            m_proj = np.einsum("fkb,mfbc,frc->fkrm", q_m, m_grid, w_m)
        pos = (
            np.full(n_f * k * r, n_keep, dtype=np.int32)
            .at[self.keep]
            .set(np.arange(n_keep, dtype=np.int32))
        )
        if self.prior_width is not None:
            y = (
                np.zeros(n_f * k * r, resid.dtype)
                .at[self.keep]
                .set(resid)
                .reshape(n_f, k, r)
                .transpose(0, 2, 1)
            )
            errors = np.asarray(self.errors)
            var = np.where(self.chan, errors[self.grid] ** 2, 0.0)
            q = np.asarray(self.q, var.dtype)
            w = np.asarray(self.w, var.dtype)
            var = np.einsum("frc,fbc->fbr", w**2, var)
            valid = np.asarray(self.valid).transpose(0, 2, 1)
            pad = 1.0 - valid.astype(var.dtype)
            cov = np.einsum("fkb,flb,fbr->frkl", q, q, var)
            cov = cov + pad[..., None] * np.eye(k, dtype=var.dtype)
            chol = np.linalg.cholesky(cov)
            x = jsl.solve_triangular(chol, y[..., None], lower=True)[..., 0]
            x_keep = x.transpose(0, 2, 1).reshape(-1)[self.keep]
            base_eff = (
                np.diagonal(chol, axis1=2, axis2=3)
                .transpose(0, 2, 1)
                .reshape(-1)[self.keep]
            )
            q_cols = np.broadcast_to(q[:, None], (n_f, r, k, q.shape[2]))
            q_cols = jsl.solve_triangular(chol, q_cols, lower=True)
            widths = np.asarray(self.prior_width, var.dtype)
            basis = np.asarray(self.basis, var.dtype) * widths
            local = np.einsum("frkb,frj->frkbj", q_cols, basis)
            local = np.where(valid[..., None, None], local, 0.0)
            local = local.reshape(n_f, r * k, -1)
            spanning = None
            if m_proj is not None:
                m_frk = m_proj.transpose(0, 2, 1, 3)
                m_w = jsl.solve_triangular(chol, m_frk, lower=True)
                m_w = np.where(valid[..., None], m_w, 0.0)
                loc_idx, span_idx = self._mode_plan
                if loc_idx:
                    m_flat = m_w.reshape(n_f, r * k, -1)
                    pad_m = np.concatenate(
                        [m_flat, np.zeros((n_f, r * k, 1), m_flat.dtype)],
                        axis=-1,
                    )
                    idx_arr = np.asarray(loc_idx, np.int32)[:, None, :]
                    local = np.concatenate(
                        [local, np.take_along_axis(pad_m, idx_arr, axis=2)],
                        axis=-1,
                    )
                if span_idx:
                    s_cols = (
                        m_w[..., list(span_idx)]
                        .transpose(0, 2, 1, 3)
                        .reshape(n_f * k * r, len(span_idx))
                    )
                    spanning = s_cols[self.keep]
            rows = (
                pos.reshape(n_f, k, r).transpose(0, 2, 1).reshape(n_f, r * k)
            )
            return x_keep, base_eff, rows, local, spanning

        resid_fkr = np.zeros(n_f * k * r, resid.dtype).at[self.keep].set(resid)
        if self.chol is None:
            chol = np.linalg.cholesky(self.covariance())
        else:
            chol = np.asarray(self.chol, resid.dtype)
        white = jsl.solve_triangular(
            chol, resid_fkr.reshape(n_f, k * r, 1), lower=True
        )[..., 0]
        x_keep = white.reshape(-1)[self.keep]
        base_eff = np.diagonal(chol, axis1=1, axis2=2).reshape(-1)[self.keep]
        if m_proj is None:
            return x_keep, base_eff, None, None, None
        m_w = jsl.solve_triangular(
            chol, m_proj.reshape(n_f, k * r, -1), lower=True
        )
        valid_fkr = np.asarray(self.valid).reshape(n_f, k * r, 1)
        m_w = np.where(valid_fkr, m_w, 0.0)
        loc_idx, span_idx = self._mode_plan
        if loc_idx:
            pad_m = np.concatenate(
                [m_w, np.zeros((n_f, k * r, 1), m_w.dtype)], axis=-1
            )
            idx_arr = np.asarray(loc_idx, np.int32)[:, None, :]
            local = np.take_along_axis(pad_m, idx_arr, axis=2)
        else:
            local = np.zeros((n_f, k * r, 1), m_w.dtype)
        spanning = None
        if span_idx:
            spanning = m_w[..., list(span_idx)].reshape(
                n_f * k * r, len(span_idx)
            )[self.keep]
        return x_keep, base_eff, pos.reshape(n_f, k * r), local, spanning

    def whiten(self, prediction, data, errors):
        # ``errors`` (the propagated diagonal) is not used: the block is
        # whitened with its full covariance from the per-sample errors.
        resid = prediction - data
        if self.keep.size == 0:
            empty = np.zeros(0, resid.dtype)
            return empty, empty
        x_keep, base_eff, rows, local, spanning = self._whitened_diffphase(
            resid
        )
        if rows is None:
            return x_keep, base_eff
        white, extra = whiten_blocks(x_keep, rows, local, spanning)
        return white, base_eff * np.exp(extra)

    def marginal_problem(self, prediction, data):
        """White-noise ``arrow_problem`` for ``s² C_0 + U Uᵀ`` integration."""
        if self.keep.size == 0:
            empty = np.zeros(0, (prediction - data).dtype)
            return arrow_problem(
                empty,
                np.zeros((1, 0), np.int32),
                np.zeros((1, 0, 0), empty.dtype),
            )
        x_keep, _, rows, local, spanning = self._whitened_diffphase(
            prediction - data
        )
        return arrow_problem(x_keep, rows, local, spanning)

    def simulated(self, prediction, cvis, noise, key=None):
        values = np.angle(np.asarray(cvis)[self.sample])
        if noise is not None:
            values = values + self.errors * noise
        prior_key = mode_key = key
        if (
            key is not None
            and self.prior_width is not None
            and self.modes is not None
        ):
            prior_key, mode_key = jax.random.split(key)
        if prior_key is not None and self.prior_width is not None:
            n_f, n_b, _ = self.grid.shape
            n_mode = len(self.prior_width)
            latent = jax.random.normal(prior_key, (n_f, n_b, n_mode))
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
        if mode_key is not None and self.modes is not None:
            raw_modes = np.asarray(self.modes, values.dtype)
            z = jax.random.normal(mode_key, (raw_modes.shape[0],))
            values = values + raw_modes.T @ z
        values = -wrap_phase(-values)  # matches np.angle: (-pi, pi]
        return eqx.tree_at(lambda b: b.values, self, values)

    def labels_shared(self, labels):
        """Whether any frame or marginalized mode spans ``labels``."""
        labels = onp.asarray(labels)
        for f in range(self.grid.shape[0]):
            used = self.grid[f][self.chan[f]]
            if onp.unique(labels[used]).size > 1:
                return True
        if self.modes is not None:
            for mode in onp.asarray(self.modes):
                if onp.unique(labels[mode != 0]).size > 1:
                    return True
        return False
