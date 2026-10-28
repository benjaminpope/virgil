"""Radio-interferometric (u, v, complex visibility) data and calibration.

!!! warning "Provisional"
    May change in any 0.4.x release without a deprecation cycle.

[`ComplexVisData`][virgil.radio.ComplexVisData] holds calibrated complex
visibilities at arbitrary (u, v) points, read from a CASA measurement set or
built from arrays. Its [`term`][virgil.radio.ComplexVisData.term] is a
likelihood term for [`fit`][virgil.fitting.fit] (and `numpyro_model`) whose
model is any [`SourceModel`][virgil.models.SourceModel], multiplied by
per-observation nuisances ([`CalibrationNuisance`][virgil.radio.CalibrationNuisance]):
a flux scale, relative gains, phase-centre offsets, error scales per
spectral window and a spectral index.

Conventions (pinned by tests, see the Radio interferometry guide): the
visibility is ``V(u, v) = Σ I(x, y) exp(-2πi (u x + v y))`` with ``x`` towards
East and ``y`` towards North and ``u``, ``v`` in wavelengths, as everywhere in
virgil; a weight is ``1 / σ²`` for each of the real and imaginary parts.

Source: the RML objective of [Zawadzki et al. (2023)](../references.md#zawadzki2023);
the reference-observation gauge of
[Cornwell & Wilkinson (1981)](../references.md#cornwellwilkinson1981); the
profiled error scale of [Pawitan (2001)](../references.md#pawitan2001).
"""

from __future__ import annotations

import dataclasses
import warnings

import equinox as eqx
import jax
import jax.numpy as np
import numpy as onp
import numpyro.distributions as dist
from numpyro.distributions import constraints

from . import _antenna_gains
from ._geometry import (
    _FT_DEFAULTS,
    gaussian_taper,
    offset_phase,
    pixel_offsets,
)
from ._deprecate import renamed
from ._utils import mas2rad, rad2mas
from .models import Image, SourceModel

__provisional__ = ("*",)

__all__ = [
    "CalibrationNuisance",
    "ComplexVisData",
    "SigmaScale",
    "dartboard_folds",
    "primary_beam",
]

C_LIGHT = 299_792_458.0  # m/s

# How the CASA ``DATA`` column maps to virgil's convention, as decided by
# ``tests/test_radio.py::test_read_ms_convention`` (a point source predicted
# by CASA's simulator): the data are V* when ``MS_CONJUGATE`` is true, and
# ``u = MS_U_SIGN * UVW[0] * ν / c``.
MS_CONJUGATE = True
MS_U_SIGN = 1.0

_CROSS = {(9, 12), (5, 8)}  # XX, YY and RR, LL


# ----------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------


class ComplexVisData(eqx.Module):
    """Complex visibilities at (u, v) points, with calibration labels.

    Build one with [`from_arrays`][virgil.radio.ComplexVisData.from_arrays]
    or [`read_ms`][virgil.radio.ComplexVisData.read_ms].

    Attributes
    ----------
    u, v : (N,) float64
        Spatial frequencies in wavelengths (``UVW * ν / c`` per channel).
    vis : (N,) complex
        Calibrated visibilities (Jy).
    weight : (N,) float
        ``1 / σ²`` of each of the real and imaginary parts (Jy⁻²).
    freq : (N,) float64
        Channel frequency (Hz).
    ddid, obs : (N,) int32
        Labels relabelled 0..n-1: ``obs`` in sorted order of the original
        ``OBSERVATION_ID`` (``obs_ids``); ``ddid`` numbers the sorted
        (``DATA_DESC_ID``, observation) pairs, so a ddid shared by several
        observations of a concatenated set gets one label (and one error
        scale) per observation. ``ddid_ids`` holds each label's original
        ``DATA_DESC_ID``.
    ddid_obs : tuple of int
        The (relabelled) observation of each ddid label.
    time, ant1, ant2, chan, scan : optional arrays
        MJD seconds, antenna indices, channel index and scan number
        (``SCAN_NUMBER``), when known. ``time``, ``ant1``, ``ant2`` and
        ``scan`` are what [`sigma_scale`][virgil.radio.ComplexVisData.sigma_scale]
        needs.
    """

    u: jax.Array
    v: jax.Array
    vis: jax.Array
    weight: jax.Array
    freq: jax.Array
    ddid: jax.Array
    obs: jax.Array
    ddid_ids: tuple = eqx.field(static=True)
    obs_ids: tuple = eqx.field(static=True)
    ddid_obs: tuple = eqx.field(static=True)
    time: jax.Array | None = None
    ant1: jax.Array | None = None
    ant2: jax.Array | None = None
    chan: jax.Array | None = None
    scan: jax.Array | None = None

    # -- construction -------------------------------------------------

    @classmethod
    def from_arrays(
        cls,
        u,
        v,
        vis,
        weight,
        freq,
        ddid=0,
        obs=0,
        *,
        time=None,
        ant1=None,
        ant2=None,
        chan=None,
        scan=None,
    ):
        """Build from arrays; scalars broadcast over the points.

        ``ddid`` and ``obs`` are the original labels, relabelled 0..n-1 in
        sorted order. Each ddid must belong to one observation.
        """
        u = onp.atleast_1d(onp.asarray(u, dtype=onp.float64))
        n = u.size

        def full(x, dtype):
            x = onp.asarray(x, dtype=dtype)
            if x.ndim == 0:
                x = onp.full(n, x)
            return x

        v = full(v, onp.float64)
        vis = full(vis, onp.complex128)
        weight = full(weight, onp.float64)
        freq = full(freq, onp.float64)
        ddid_orig = full(ddid, onp.int64)
        obs_orig = full(obs, onp.int64)
        optional = {
            "time": time,
            "ant1": ant1,
            "ant2": ant2,
            "chan": chan,
            "scan": scan,
        }
        for name, x in list(optional.items()):
            if x is not None:
                optional[name] = full(x, None)
        for name, x in dict(
            u=u,
            v=v,
            vis=vis,
            weight=weight,
            freq=freq,
            ddid=ddid_orig,
            obs=obs_orig,
        ).items():
            if x.shape != (n,):
                raise ValueError(f"{name} has shape {x.shape}, not ({n},).")
        for name, x in optional.items():
            if x is not None and x.shape != (n,):
                raise ValueError(f"{name} has shape {x.shape}, not ({n},).")
        if not (onp.all(onp.isfinite(u)) and onp.all(onp.isfinite(v))):
            raise ValueError("u and v must be finite.")
        if not onp.all(onp.isfinite(vis)):
            raise ValueError("vis must be finite.")
        if not (onp.all(onp.isfinite(weight)) and onp.all(weight > 0)):
            raise ValueError("weight must be finite and positive.")
        if not (onp.all(onp.isfinite(freq)) and onp.all(freq > 0)):
            raise ValueError("freq must be finite and positive.")
        originals, obs_l = onp.unique(
            onp.asarray(obs_orig), return_inverse=True
        )
        obs_ids, obs_l = (
            tuple(int(i) for i in originals),
            obs_l.astype(onp.int32),
        )
        # A label is a (ddid, observation) pair: a concatenated measurement
        # set can hold one ddid in several observations, each with its own
        # error scale.
        pair = ddid_orig * (len(obs_ids) + 1) + obs_l
        _, first, ddid_l = onp.unique(
            pair, return_index=True, return_inverse=True
        )
        ddid_l = ddid_l.astype(onp.int32)
        ddid_ids = tuple(int(d) for d in ddid_orig[first])
        ddid_obs = [int(o) for o in obs_l[first]]
        return cls(
            u=np.asarray(u),
            v=np.asarray(v),
            vis=np.asarray(vis),
            weight=np.asarray(weight),
            freq=np.asarray(freq),
            ddid=np.asarray(ddid_l),
            obs=np.asarray(obs_l),
            ddid_ids=ddid_ids,
            obs_ids=obs_ids,
            ddid_obs=tuple(ddid_obs),
            **{
                k: None if x is None else np.asarray(x)
                for k, x in optional.items()
            },
        )

    @classmethod
    def read_ms(
        cls,
        path,
        ddids=None,
        stride=1,
        datacolumn="DATA",
        field=None,
        conjugate=None,
        u_sign=None,
    ):
        """Read a CASA measurement set (needs the ``radio`` extra).

        Cross-correlations only; one sample per (row, channel), kept if the
        row and both parallel-hand polarizations (XX, YY or RR, LL) are
        unflagged and both weights positive. The two polarizations are
        averaged with their weights (Stokes I for equal weights) and the
        weight is their sum. ``WEIGHT_SPECTRUM`` is used when present and
        non-empty, else ``WEIGHT``. ``u = UVW[0] ν / c``, ``v = UVW[1] ν / c``
        per channel. One ddid is read at a time, so memory is that of the
        largest ddid at the given ``stride``.

        Parameters
        ----------
        path : str
            The measurement set.
        ddids : sequence of int, optional
            ``DATA_DESC_ID`` values to read (default all).
        field : int, optional
            ``FIELD_ID`` to read. By default the selection must hold one
            field only; a mosaic (several fields) raises.
        stride : int, optional
            Keep every ``stride``-th row of each ddid.
        datacolumn : str, optional
            ``"DATA"`` or ``"CORRECTED_DATA"``.
        conjugate, u_sign : optional
            Override the module's pinned convention (``MS_CONJUGATE``,
            ``MS_U_SIGN``) for comparisons.
        """
        try:
            from casatools import table as casa_table
        except ImportError as err:  # pragma: no cover - needs no casatools
            raise ImportError(
                "read_ms needs casatools: pip install 'virgil-astro[radio]'."
            ) from err
        conjugate = MS_CONJUGATE if conjugate is None else conjugate
        u_sign = MS_U_SIGN if u_sign is None else u_sign
        stride = int(stride)
        if stride < 1:
            raise ValueError("stride must be a positive integer.")
        path = str(path)

        tb = casa_table()
        try:
            tb.open(f"{path}/DATA_DESCRIPTION")
            spw_of = tb.getcol("SPECTRAL_WINDOW_ID")
            pol_of = tb.getcol("POLARIZATION_ID")
            tb.close()
            tb.open(f"{path}/SPECTRAL_WINDOW")
            chan_freqs = [
                onp.asarray(tb.getcell("CHAN_FREQ", int(s)), dtype=onp.float64)
                for s in range(tb.nrows())
            ]
            tb.close()
            tb.open(f"{path}/POLARIZATION")
            corr_types = [
                tuple(int(c) for c in tb.getcell("CORR_TYPE", int(p)))
                for p in range(tb.nrows())
            ]
            tb.close()
            wanted = range(len(spw_of)) if ddids is None else list(ddids)
            parts = []
            tb.open(path)
            wanted = [int(d) for d in wanted]
            fields = onp.unique(
                onp.asarray(tb.getcol("FIELD_ID"))[
                    onp.isin(onp.asarray(tb.getcol("DATA_DESC_ID")), wanted)
                ]
            )
            if field is None and fields.size > 1:
                raise ValueError(
                    f"the selection holds {fields.size} fields {fields.tolist()}"
                    " (a mosaic?); pass field=. Mosaics are not supported."
                )
            field_query = (
                "" if field is None else f" && FIELD_ID=={int(field)}"
            )
            for d in wanted:
                d = int(d)
                if not 0 <= d < len(spw_of):
                    raise ValueError(
                        f"ddid {d} is not in the measurement set."
                    )
                corr = corr_types[int(pol_of[d])]
                for hand_a, hand_b in sorted(_CROSS):
                    if hand_a in corr and hand_b in corr:
                        pair = corr.index(hand_a), corr.index(hand_b)
                        break
                else:
                    raise ValueError(
                        f"no XX/YY or RR/LL polarization pair in CORR_TYPE {corr}."
                    )
                sub = tb.query(
                    f"DATA_DESC_ID=={d} && ANTENNA1!=ANTENNA2{field_query}"
                )
                try:
                    if sub.nrows() == 0:
                        continue
                    part = _read_ddid(
                        sub,
                        pair,
                        chan_freqs[int(spw_of[d])],
                        stride,
                        datacolumn,
                        conjugate,
                        u_sign,
                    )
                finally:
                    sub.close()
                if part is None:
                    warnings.warn(f"ddid {d} has no unflagged samples.")
                    continue
                part["ddid"] = onp.full(part["u"].size, d, dtype=onp.int64)
                parts.append(part)
            tb.close()
        finally:
            tb.done()
        if not parts:
            raise ValueError("no unflagged cross-correlations were read.")
        cat = {k: onp.concatenate([p[k] for p in parts]) for k in parts[0]}
        return cls.from_arrays(**cat)

    # -- selection ----------------------------------------------------

    def __len__(self):
        return int(self.u.shape[0])

    def subset(self, mask):
        """The points where boolean ``mask`` is true, labels unchanged.

        Labels are kept as they are (not relabelled), so a
        [`CalibrationNuisance`][virgil.radio.CalibrationNuisance] built for
        the full data still fits.
        """
        mask = onp.asarray(mask, dtype=bool)
        if mask.shape != (len(self),):
            raise ValueError(
                f"mask has shape {mask.shape}, not ({len(self)},)."
            )
        idx = onp.flatnonzero(mask)
        leaves = {}
        for f in dataclasses.fields(self):
            x = getattr(self, f.name)
            leaves[f.name] = (
                x
                if f.name in ("ddid_ids", "obs_ids", "ddid_obs") or x is None
                else x[idx]
            )
        return type(self)(**leaves)

    def select(self, ddids=None, obs=None):
        """Points of the given ddids and/or observations.

        ``ddids`` and ``obs`` are the original ids (``ddid_ids``,
        ``obs_ids``). Labels are kept; see
        [`subset`][virgil.radio.ComplexVisData.subset].
        """
        keep = onp.ones(len(self), dtype=bool)
        if ddids is not None:
            labels = [
                i
                for i, d in enumerate(self.ddid_ids)
                if d in set(int(x) for x in onp.atleast_1d(ddids))
            ]
            keep &= onp.isin(onp.asarray(self.ddid), labels)
        if obs is not None:
            labels = [self.obs_ids.index(int(e)) for e in onp.atleast_1d(obs)]
            keep &= onp.isin(onp.asarray(self.obs), labels)
        return self.subset(keep)

    # -- model --------------------------------------------------------

    def model(self, scene, values, nuisance):
        """The complex model ``M_k`` at every point (all at once).

        ``M = F g_e (ν/ν₀)^α exp(-2πi(δα_e u + δδ_e v)) T(u, v)
        V_scene(u, v)``; see
        [`CalibrationNuisance`][virgil.radio.CalibrationNuisance]. ``scene``
        is a [`SourceModel`][virgil.models.SourceModel] (normalized to unit
        flux; ``cal.flux`` sets the scale), evaluated at wavelength
        ``c / ν`` of each point so that chromatic scenes work.
        """
        _no_antenna_gains(nuisance)
        return _model_points(
            scene,
            nuisance._unpack(values),
            nuisance,
            self.u,
            self.v,
            self.freq,
            self.ddid,
            self.obs,
        )

    def term(
        self,
        scene,
        nuisance,
        mask=None,
        chunk_size=10_000_000,
        sigma=None,
        chi2="auto",
    ):
        """A likelihood term for [`fit`][virgil.fitting.fit]'s ``likelihoods``.

        Parameters
        ----------
        scene : callable
            Maps the fitted values (a dict by site name) to a
            [`SourceModel`][virgil.models.SourceModel]; a function of the
            values, not a model instance.
        nuisance : CalibrationNuisance
            The calibration model; its sites are keys of the fit's priors.
        mask : (N,) bool, optional
            Points to use (default all); masked points change neither χ²
            nor the normalization.
        chunk_size : int, optional
            Points evaluated at a time, bounding memory.
        sigma : (n_ddid,) array, optional
            With ``sigmas="profile"``, fixed error scales instead of the
            ``ŝ`` of this term's own points. Score held-out data at the
            training fit's ``ŝ`` (``fit(...).info["derived"]["cal.sigma"]``):
            profiling its own ``ŝ`` is not a predictive score.
        chi2 : {"auto", "direct", "toeplitz"}, optional
            How ``fit`` evaluates χ². ``"auto"`` uses the dirty-beam form of
            [`toeplitz`][virgil.radio.ComplexVisData.toeplitz] when it is
            exact (one fixed ``Image``, no ``antenna_gains``, fitted
            ``alpha`` or ``offsets``, float64, ``method="lbfgs"`` or
            ``"adam"``, enough points per pixel and ddid, see
            [`set_ft_defaults`][virgil._geometry.set_ft_defaults]);
            ``"toeplitz"`` requires it (``ValueError`` names what fails;
            fitted offsets then use ``offsets="spread"``); ``"direct"``
            never uses it. Each ``fit`` rebuilds the precompute: reuse
            ``toeplitz(...)`` for repeated fits. ``numpyro_model`` and
            ``cross_validate`` stay direct (it re-profiles each fold: use
            ``sigmas=True``).

        Notes
        -----
        In your own jitted code pass the term as an argument with
        ``eqx.filter_jit``; a ``jax.jit`` closure bakes the data in as
        constants (see the radio interferometry page).
        """
        # deliberate: points are padded with zero weight to a multiple of
        # chunk_size and mapped with lax.map (never vmap-ed); a function (not a
        # model instance) keeps arrays out of the jit as compile-time constants
        if isinstance(scene, SourceModel):
            raise TypeError(
                "scene must be a function of the fitted values, e.g. "
                "`lambda values: model`, not a model instance."
            )
        if chi2 not in ("auto", "direct", "toeplitz"):
            raise ValueError(
                f"chi2 must be 'auto', 'direct' or 'toeplitz', not {chi2!r}."
            )
        return _RadioTerm(self, scene, nuisance, mask, chunk_size, sigma, chi2)

    def toeplitz(
        self,
        image,
        nuisance,
        mask=None,
        offsets="none",
        eps=1e-14,
        max_offset_mas=None,
        chunk_size=65536,
    ):
        """Precompute the dirty beams and images of a one-Image fit.

        Returns a [`ToeplitzVisData`][virgil.radio_toeplitz.ToeplitzVisData],
        whose ``term`` gives the χ² of ``term`` for a scene that is one
        [`Image`][virgil.models.Image] with the layout of ``image``, at a
        cost independent of the number of points. Provisional.

        Parameters
        ----------
        image : Image
            The layout (shape, ``pixel_scale_mas``, ``rotation_deg``,
            ``blur_fwhm_mas``) and a fixed ``dra``, ``ddec`` (folded into
            the visibilities); its pixels are not used.
        nuisance : CalibrationNuisance
            Its taper, ``nu0_hz`` and fixed ``alpha`` are folded in;
            antenna gains and a fitted ``alpha`` are refused.
        mask : (N,) bool, optional
            Points to use (default all).
        offsets : {"none", "spread"}, optional
            ``"none"`` (default) needs ``CalibrationNuisance(offsets=False)``.
            ``"spread"`` fits the offsets by evaluating the stored spread
            grids at shifted positions (an extension: see the guide), and
            refuses baselines past its Nyquist margin.
        eps : float, optional
            Tolerance of the spreading kernel.
        max_offset_mas : float, optional
            Largest offset of any observation, in mas (default the bound
            of ``nuisance.priors()``).
        chunk_size : int, optional
            Points spread at a time, bounding memory.
        """
        # Imported here: radio_toeplitz imports this module.
        from .radio_toeplitz import ToeplitzVisData

        return ToeplitzVisData(
            self,
            image,
            nuisance,
            mask,
            offsets,
            eps,
            max_offset_mas,
            chunk_size,
        )

    def sigma_hat(self, scene, values, nuisance, mask=None):
        """The error scales ``s_d`` per ddid at ``values``: the profiled ``ŝ``
        of ``CalibrationNuisance(sigmas="profile")``, the fitted ``cal.sigma``
        with ``sigmas=True``, ones otherwise. Provisional, as the module."""
        if isinstance(scene, SourceModel):
            model = scene
            scene = lambda values: model  # noqa: E731
        term = self.term(scene, nuisance, mask=mask)
        return onp.asarray(term.sigma_hat(values))

    def chi2_by_ddid(
        self, scene, values, nuisance, mask=None, chunk_size=10_000_000
    ):
        """χ² of the data with the σ scales applied, per ddid.

        Returns ``(chi2, n)``: float64 arrays of length ``n_ddid``, ``n`` the
        number of unmasked points of each ddid (χ² sums two terms per point).
        ``chunk_size`` is that of [`term`][virgil.radio.ComplexVisData.term].
        """
        if isinstance(scene, SourceModel):
            model = scene
            scene = lambda values: model  # noqa: E731
        term = self.term(scene, nuisance, mask=mask, chunk_size=chunk_size)
        re, im = term._evaluate(values)[:2]
        r2 = term._flat(re) ** 2 + term._flat(im) ** 2
        mask_ = (
            onp.ones(len(self), dtype=bool)
            if mask is None
            else onp.asarray(mask, dtype=bool)
        )
        n_ddid = len(self.ddid_ids)
        chi2 = jax.ops.segment_sum(
            np.asarray(r2, dtype=np.float64), self.ddid, n_ddid
        )
        n = onp.bincount(onp.asarray(self.ddid)[mask_], minlength=n_ddid)
        return onp.asarray(chi2), n

    # -- diagnostics --------------------------------------------------

    def sigma_scale(self, by="ddid"):
        """Model-free estimate of the error scale from time-adjacent samples.

        Samples of one baseline, channel and scan (and ddid label) are paired
        in time order without overlap; ``ŝ²`` is the mean over pairs of
        ``ΔRe² / (1/w_i + 1/w_j)``, likewise for Im, and ``ŝ`` averages the two.

        **This is an upper bound.** Sky, gain or phase change within a pair
        adds to ``|ΔV|²``, so ``ŝ`` is biased high for rapidly varying data
        (long gaps, atmospheric phase), exact only if the signal is constant.
        Samples without a partner are not used. Needs the ``time``, ``ant1``,
        ``ant2`` and ``scan`` fields filled by
        [`read_ms`][virgil.radio.ComplexVisData.read_ms]; with ``stride > 1``
        the rows read are no longer adjacent.

        Parameters
        ----------
        by : {"ddid", "obs", None}
            Grouping of the result: per ``DATA_DESC_ID``, per observation id,
            or pooled (``None``).

        Returns
        -------
        SigmaScale
            Group labels, pair counts, and ``s``, ``s_re``, ``s_im`` with
            standard errors. ``lag_ratio`` well above 1 means the sky varies
            (``ŝ`` is conservative); near 1 means constant data or a time
            order that does not follow the data (a scrambled time column):
            check ``ŝ``.
        """
        # deliberate: Var(ΔRe) = s²(1/w_i + 1/w_j) for σ² = s²/w; the error of
        # ŝ² is the sample standard error of the mean (no Gaussian assumption),
        # se(ŝ) = se(ŝ²) / (2ŝ); channels are identified by their frequency.
        missing = [
            n
            for n in ("time", "ant1", "ant2", "scan")
            if getattr(self, n) is None
        ]
        if missing:
            raise ValueError(
                f"sigma_scale needs {', '.join(missing)}; build the data with "
                "those fields (read_ms fills them)."
            )
        if by not in ("ddid", "obs", None):
            raise ValueError("by must be 'ddid', 'obs' or None.")
        ddid = onp.asarray(self.ddid)
        obs = onp.asarray(self.obs)
        if by == "obs":
            group, labels = obs, list(self.obs_ids)
        elif by == "ddid":
            group, labels = ddid, list(self.ddid_ids)
        else:
            group, labels = onp.zeros(len(self), dtype=onp.int32), [None]
        ng = len(labels)
        sums = onp.zeros((ng, 3, 2))  # (re, im, both) x (sum z², sum z⁴)
        npair = onp.zeros(ng, dtype=onp.int64)
        far_sum = onp.zeros(ng)
        far_n = onp.zeros(ng, dtype=onp.int64)
        near_sum = onp.zeros(ng)
        near_n = onp.zeros(ng, dtype=onp.int64)
        for d in range(len(self.ddid_ids)):  # sort one ddid label at a time
            idx = onp.flatnonzero(ddid == d)
            if idx.size < 2:
                continue
            g = int(group[idx[0]]) if by is not None else 0
            time = onp.asarray(self.time)[idx]
            ant1 = onp.asarray(self.ant1)[idx].astype(onp.int64)
            ant2 = onp.asarray(self.ant2)[idx].astype(onp.int64)
            scan = onp.asarray(self.scan)[idx].astype(onp.int64)
            _, chan = onp.unique(
                onp.asarray(self.freq)[idx], return_inverse=True
            )
            base = ant1 * (ant2.max() + 1) + ant2
            order = onp.lexsort((time, chan, base, scan))
            key = onp.stack([scan, base, chan])[:, order]
            idx = idx[order]
            del time, ant1, ant2, scan, base, chan, order
            new = onp.empty(idx.size, dtype=bool)
            new[0] = True
            new[1:] = (key[:, 1:] != key[:, :-1]).any(axis=0)
            del key
            start = onp.flatnonzero(new)
            length = onp.diff(onp.append(start, idx.size))
            run = onp.cumsum(new) - 1
            pos = onp.arange(idx.size) - start[run]
            rlen = length[run]
            vis = onp.asarray(self.vis)
            inv_w = 1.0 / onp.asarray(self.weight)

            def z2(a, b):
                a, b = idx[a], idx[b]
                dv = vis[a] - vis[b]
                den = inv_w[a] + inv_w[b]
                return dv.real**2 / den, dv.imag**2 / den

            first = onp.flatnonzero((pos % 2 == 0) & (pos + 1 < rlen))
            if first.size:
                zr, zi = z2(first, first + 1)
                both = 0.5 * (zr + zi)
                for k, z in enumerate((zr, zi, both)):
                    sums[g, k] += (z.sum(), (z**2).sum())
                npair[g] += first.size
                near_sum[g] += both.sum()
                near_n[g] += both.size
            half = rlen // 2
            far = onp.flatnonzero((pos < half) & (half >= 1))
            if far.size:
                zr, zi = z2(far, far + half[far])
                far_sum[g] += (0.5 * (zr + zi)).sum()
                far_n[g] += far.size
        with onp.errstate(invalid="ignore", divide="ignore"):
            n = npair.astype(float)
            mean = sums[:, :, 0] / n[:, None]  # ŝ² of re, im, both
            var = onp.maximum(sums[:, :, 1] / n[:, None] - mean**2, 0.0)
            se2 = onp.sqrt(var / n[:, None])
            s = onp.sqrt(mean)
            se = se2 / (2.0 * s)
            lag = (far_sum / far_n) / (near_sum / near_n)
        return SigmaScale(
            labels=tuple(labels),
            n_pairs=npair,
            s=s[:, 2],
            s_se=se[:, 2],
            s_re=s[:, 0],
            s_re_se=se[:, 0],
            s_im=s[:, 1],
            s_im_se=se[:, 1],
            lag_ratio=lag,
        )

    def _imaging_weights(
        self,
        weighting,
        robust,
        uv_cell,
        nuisance,
        values,
        sigma=None,
        uvtaper=None,
    ):
        """Imaging weights: ``w`` (÷ s_d² if fitted), times uv-density
        factors (computed without the taper) and the ``uvtaper``."""
        w = onp.asarray(self.weight, dtype=onp.float64)
        if sigma is not None:
            w = (
                w
                / onp.asarray(sigma, dtype=onp.float64)[onp.asarray(self.ddid)]
                ** 2
            )
        elif nuisance is not None and values is not None and nuisance.sigmas:
            if nuisance.sigmas == "profile":
                raise ValueError(
                    "sigmas='profile': pass use_sigma through dirty_image "
                    "with a scene, which computes ŝ."
                )
            s = onp.asarray(nuisance._unpack(values)[3], dtype=onp.float64)
            w = w / s[onp.asarray(self.ddid)] ** 2
        if uvtaper is not None:
            bmaj, bmin, pa = (
                (uvtaper, uvtaper, 0.0) if onp.ndim(uvtaper) == 0 else uvtaper
            )
            pa = onp.deg2rad(pa)
            tu, tv = onp.asarray(self.u), onp.asarray(self.v)
            along = tu * onp.sin(pa) + tv * onp.cos(pa)
            across = tu * onp.cos(pa) - tv * onp.sin(pa)
            a, b = bmaj * mas2rad, bmin * mas2rad
            taper = onp.exp(
                -(onp.pi**2)
                * ((a * along) ** 2 + (b * across) ** 2)
                / (4.0 * onp.log(2.0))
            )
        if weighting == "natural":
            return w if uvtaper is None else w * taper
        if weighting not in ("uniform", "briggs"):
            raise ValueError(
                "weighting must be 'natural', 'uniform' or 'briggs'."
            )
        u, v = onp.asarray(self.u), onp.asarray(self.v)
        flip = (v < 0) | (
            (v == 0) & (u < 0)
        )  # Hermitian partners share a cell
        iu = onp.floor(onp.where(flip, -u, u) / uv_cell).astype(onp.int64)
        iv = onp.floor(onp.where(flip, -v, v) / uv_cell).astype(onp.int64)
        iu -= iu.min()
        cell = iv * (iu.max() + 1) + iu
        _, inverse = onp.unique(cell, return_inverse=True)
        density = onp.bincount(inverse, weights=w)[inverse]
        if weighting == "uniform":
            out = w / density
        else:
            # Briggs: w / (1 + W_c f²), f² = (5 · 10^-R)² / (Σ_c W_c² / Σ_k w_k)
            w_c = onp.bincount(inverse, weights=w)
            f2 = (5.0 * 10.0 ** (-robust)) ** 2 / (
                onp.sum(w_c**2) / onp.sum(w)
            )
            out = w / (1.0 + density * f2)
        return out if uvtaper is None else out * taper

    def _adjoint(self, values, w, npix, pixel_scale_mas, chunk_size):
        """``Σ_k w_k Re[a_k exp(+2πi(u_k x + v_k y))] / Σ w`` on the pixel grid.

        The adjoint (type-1) NUFFT at the raw (u, v); taking the real part
        adds the Hermitian conjugates. ``values`` complex (N,).
        """
        from nufftax import nufft2d1

        from ._geometry import _FT_DEFAULTS

        npix = int(npix)
        c = 0.5 * (npix - 1)
        total = float(onp.sum(w))
        image = onp.zeros((npix, npix))
        n = len(self)
        for lo in range(0, n, int(chunk_size)):
            sl = slice(lo, lo + int(chunk_size))
            t_col = (
                2.0
                * onp.pi
                * mas2rad
                * pixel_scale_mas
                * onp.asarray(self.u[sl])
            )
            t_row = (
                2.0
                * onp.pi
                * mas2rad
                * pixel_scale_mas
                * onp.asarray(self.v[sl])
            )
            a = w[sl] * values[sl]
            # exp(+i(t(c - col))) = exp(-i t m) exp(+i t (c - n//2)), m = col - n//2
            a = a * onp.exp(1j * (t_col + t_row) * (c - npix // 2))
            wrap = lambda t: t - 2.0 * onp.pi * onp.round(  # noqa: E731
                t / (2.0 * onp.pi)
            )
            # deliberate: nufftax type-1 at float64; the real part adds the Hermitian partners
            f = nufft2d1(
                np.asarray(wrap(t_col)),
                np.asarray(wrap(t_row)),
                np.asarray(a),
                (npix, npix),
                eps=_FT_DEFAULTS["eps_f64"],
                isign=-1,
            )
            image += onp.asarray(np.real(f))
        return image / total

    def dirty_beam(
        self,
        npix,
        pixel_scale_mas,
        weighting="natural",
        robust=0.5,
        values=None,
        nuisance=None,
        uv_cell=None,
        chunk_size=1_000_000,
        uvtaper=None,
    ):
        """The dirty beam ``Σ w cos(2π(u x + v y)) / Σ w`` on the pixel grid.

        The response to a unit point source at the phase centre, normalized
        to 1 there. See [`dirty_image`][virgil.radio.ComplexVisData.dirty_image]
        for the parameters; ``values`` and ``nuisance`` together divide the
        weights by the fitted ``s_d²`` (as ``use_sigma=True``). On an even ``npix`` the phase centre lies between
        four pixels (the convention of [`Image`][virgil.models.Image]), so the
        sampled peak is slightly below 1; use an odd ``npix`` for a pixel
        centred on it.
        """
        return self._dirty(
            None,
            npix,
            pixel_scale_mas,
            weighting,
            robust,
            nuisance,
            values,
            uv_cell,
            chunk_size,
            uvtaper,
            beam=True,
        )

    def dirty_image(
        self,
        npix,
        pixel_scale_mas,
        weighting="natural",
        robust=0.5,
        scene=None,
        values=None,
        nuisance=None,
        use_sigma=False,
        uv_cell=None,
        chunk_size=1_000_000,
        uvtaper=None,
    ):
        """Dirty image ``Σ w Re[V e^{+2πi(u x + v y)}] / Σ w`` in Jy/beam.

        A display diagnostic, never part of the likelihood: a type-1 NUFFT at
        the raw (u, v), in the orientation of [`Image`][virgil.models.Image]
        (East left, North up). Details:
        [Dirty and residual images](../radio_interferometry.md#dirty-and-residual-images).

        Parameters
        ----------
        npix : int
        pixel_scale_mas : float
        weighting : {"natural", "uniform", "briggs"}
            Natural ``w`` (``1/σ²``), uniform ``w / W_c`` for the summed weight
            ``W_c`` of the point's uv cell, or Briggs with ``robust``.
        robust : float
            Briggs ``R``, about −2 (uniform) to 2 (natural).
        scene : callable or SourceModel, optional
            With ``values`` and ``nuisance``, image the residual ``V − M``, ``M`` including the nuisances.
        values, nuisance : optional
            Fitted values and the
            [`CalibrationNuisance`][virgil.radio.CalibrationNuisance].
        use_sigma : bool
            Divide the weights by the fitted ``s_d²`` (needs ``values`` and
            ``nuisance``).
        uv_cell : float, optional
            Cell of the density grid in wavelengths; default ``1/FoV``
            (``npix · pixel_scale_mas`` in radians).
        chunk_size : int
            Points transformed at a time, bounding memory.
        uvtaper : float or (bmaj_mas, bmin_mas, pa_deg), optional
            CASA's ``uvtaper``: a Gaussian FWHM in mas (a float is circular;
            ``pa_deg`` is East of North), applied after the density weights.
            Provisional, as the module.

        Returns
        -------
        (npix, npix) float64 array
        """
        sigma = None
        if use_sigma and nuisance is not None and nuisance.sigmas == "profile":
            if scene is None or values is None:
                raise ValueError(
                    "use_sigma with sigmas='profile' needs a scene and "
                    "values (ŝ is the scatter about the model)."
                )
            sigma = self.sigma_hat(scene, values, nuisance)
        return self._dirty(
            (scene, values, nuisance),
            npix,
            pixel_scale_mas,
            weighting,
            robust,
            nuisance if use_sigma else None,
            values if use_sigma else None,
            uv_cell,
            chunk_size,
            uvtaper,
            beam=False,
            sigma=sigma,
        )

    def _dirty(
        self,
        residual,
        npix,
        pixel_scale_mas,
        weighting,
        robust,
        nuis_w,
        values_w,
        uv_cell,
        chunk_size,
        uvtaper,
        beam,
        sigma=None,
    ):
        if uv_cell is None:
            uv_cell = 1.0 / (int(npix) * pixel_scale_mas * mas2rad)
        if nuis_w is not None and values_w is None:
            raise ValueError("use_sigma needs values and nuisance.")
        w = self._imaging_weights(
            weighting,
            robust,
            uv_cell,
            nuis_w,
            values_w,
            sigma,
            uvtaper,
        )
        if beam:
            a = onp.ones(len(self), dtype=onp.complex128)
        else:
            scene, values, nuisance = residual
            a = onp.asarray(self.vis, dtype=onp.complex128)
            if scene is not None:
                if values is None or nuisance is None:
                    raise ValueError("a scene needs values and nuisance.")
                scene = scene(values) if callable(scene) else scene
                _no_antenna_gains(nuisance)
                unpacked = nuisance._unpack(values)
                model = onp.empty(len(self), dtype=onp.complex128)
                for lo in range(0, len(self), int(chunk_size)):
                    sl = slice(lo, lo + int(chunk_size))
                    model[sl] = onp.asarray(
                        _model_points(
                            scene,
                            unpacked,
                            nuisance,
                            self.u[sl],
                            self.v[sl],
                            self.freq[sl],
                            self.ddid[sl],
                            self.obs[sl],
                        )
                    )
                a = a - model
        return self._adjoint(a, w, npix, pixel_scale_mas, chunk_size)


@dataclasses.dataclass(frozen=True)
class SigmaScale:
    """Result of [`ComplexVisData.sigma_scale`][virgil.radio.ComplexVisData.sigma_scale].

    Arrays have one entry per group, in the order of ``labels`` (the original
    ``DATA_DESC_ID`` or observation ids, or ``None`` when pooled). ``s`` is
    the combined error scale and ``s_se`` its standard error; ``s_re`` and
    ``s_im`` are the real and imaginary estimates; ``n_pairs`` the pairs
    used; ``lag_ratio`` the far-pair to adjacent-pair mean-square difference.
    All are upper bounds on the true scale (sky change inflates them).
    """

    labels: tuple
    n_pairs: onp.ndarray
    s: onp.ndarray
    s_se: onp.ndarray
    s_re: onp.ndarray
    s_re_se: onp.ndarray
    s_im: onp.ndarray
    s_im_se: onp.ndarray
    lag_ratio: onp.ndarray


# deliberate: tests read one ddid's samples directly
def _read_ddid(sub, pair, chan_freq, stride, column, conjugate, u_sign):
    """The samples of one ddid's cross-correlation query, or None."""
    data = sub.getcol(column, 0, -1, stride)  # (ncorr, nchan, nrow)
    flag = sub.getcol("FLAG", 0, -1, stride)
    flag_row = sub.getcol("FLAG_ROW", 0, -1, stride)
    uvw = sub.getcol("UVW", 0, -1, stride)
    time = sub.getcol("TIME", 0, -1, stride)
    ant1 = sub.getcol("ANTENNA1", 0, -1, stride)
    ant2 = sub.getcol("ANTENNA2", 0, -1, stride)
    obs = sub.getcol("OBSERVATION_ID", 0, -1, stride)
    scan = sub.getcol("SCAN_NUMBER", 0, -1, stride)
    weight = None
    if "WEIGHT_SPECTRUM" in sub.colnames() and sub.iscelldefined(
        "WEIGHT_SPECTRUM", 0
    ):
        ws = sub.getcol("WEIGHT_SPECTRUM", 0, -1, stride)
        if ws.size:
            weight = ws
    if weight is None:
        w = sub.getcol("WEIGHT", 0, -1, stride)  # (ncorr, nrow)
        weight = onp.broadcast_to(w[:, None, :], data.shape)
    i, j = pair
    w1, w2 = weight[i].astype(onp.float64), weight[j].astype(onp.float64)
    ok = (
        ~flag[i]
        & ~flag[j]
        & ~flag_row[None, :].astype(bool)
        & (w1 > 0)
        & (w2 > 0)
    )  # (nchan, nrow)
    if not ok.any():
        return None
    wsum = w1 + w2
    vis = (w1 * data[i] + w2 * data[j]) / onp.where(wsum > 0, wsum, 1.0)
    if conjugate:
        vis = onp.conj(vis)
    ch, rw = onp.nonzero(ok)
    freq = chan_freq[ch]
    lam_inv = freq / C_LIGHT
    return dict(
        u=u_sign * uvw[0, rw] * lam_inv,
        v=uvw[1, rw] * lam_inv,
        vis=vis[ch, rw],
        weight=wsum[ch, rw],
        freq=freq,
        obs=obs[rw].astype(onp.int64),
        time=time[rw],
        ant1=ant1[rw],
        ant2=ant2[rw],
        chan=ch.astype(onp.int16),
        scan=scan[rw],
    )


# ----------------------------------------------------------------------
# Calibration nuisances
# ----------------------------------------------------------------------


class _LogImproper(dist.Distribution):
    """The improper Jeffreys prior ``p(x) ∝ 1/x`` of a positive scale.

    Flat in ``ln x`` on the whole line, so ``fit`` optimizes ``z = ln x`` with
    no box (``flat_coordinate`` has infinite ends); it has no samples.
    """

    support = constraints.positive
    pytree_data_fields = ()

    def __init__(self, shape=(), *, validate_args=None):
        super().__init__(batch_shape=tuple(shape), validate_args=validate_args)

    def log_prob(self, value):
        return -np.log(value)

    def flat_coordinate(self):
        return np.log, np.exp, -onp.inf, onp.inf


_WIDTHS = ("gain_amp_width", "gain_phase_width")


class CalibrationNuisance:
    """Per-observation calibration nuisances of a radio dataset.

    Static configuration: which nuisances are fitted. The sites
    (``prefix="cal"``) are keys of the fit's ``priors``; a function model must
    accept them. See [Radio interferometry](../radio_interferometry.md#a-model-with-calibration-nuisances).

    ===============  ================  ==========================  =============
    Site             Shape             Default prior               Not fitted
    ===============  ================  ==========================  =============
    ``cal.flux``     ``()``            ``LogUniform(1e-4, 1e2)``   always fitted
    ``cal.gain``     ``(n_obs-1,)``    Jeffreys ``1/g``            1
    ``cal.offset``   ``(n_obs-1, 2)``  ``Uniform(-θ_k, θ_k)`` mas  0
    ``cal.sigma``    ``(n_ddid,)``     Jeffreys ``1/s``            1
    ``cal.alpha``    ``()``            ``Uniform(0, 5)``           0 (or ``alpha``)
    ===============  ================  ==========================  =============

    ``fit_gain_widths`` adds scalar sites ``cal.gain_amp_width`` and
    ``cal.gain_phase_width`` (``LogUniform(1e-3, 10)``). The reference
    observation has gain 1 and zero offset; ``θ_k`` is ``offset_beams`` beams
    of observation ``k`` (``fit(...).info["at_bound"]`` reports a hit).

    Parameters
    ----------
    data : ComplexVisData
    ref_obs : int
        Index of the reference observation.
    gains, offsets, sigmas, alpha : bool (``alpha`` also a float)
        Which nuisances are fitted (the flux always is); gains and offsets
        need more than one observation; a nuisance not fitted takes the table's last column.
        ``sigmas="profile"`` sets the error scales at every evaluation (see
        [`sigma_hat`][virgil.radio.ComplexVisData.sigma_hat]). A float
        ``alpha`` fixes the spectral index (a non-zero integer is refused:
        ``alpha=1`` used to mean "fit").
    offset_beams : float, optional
        Half-width of the default offset prior, in beams; default 3.
        Provisional, as the module.
    antenna_gains : bool
        Marginalize a complex gain per antenna and solution interval ([antenna
        gains](../radio_interferometry.md#antenna-based-gains)); needs
        ``time``, ``ant1``, ``ant2``; no least-squares form.
    solint : {"int", "scan"} or float
        Solution interval: an integration, a scan or seconds (within a scan).
    gain_amp_width, gain_phase_width : float
        Widths ``τ_a`` and ``τ_φ`` (rad) of the Gaussian gain priors (choices).
    fit_gain_widths : bool or {"amp", "phase"}
        Fit the widths (log-uniform priors; see ``priors``).
    gain_steps : int
        Gauss–Newton steps of the gain solution.
    nu0_hz : float, optional
        Reference frequency of the spectral index; default the
        weight-averaged frequency of ``data``.
    blur_fwhm_mas : float, optional
        FWHM (mas) of a Gaussian blurring kernel in the forward model: the
        data are compared with the blurred model. Default ``None``,
        recommended; blur for display afterwards. Not ``uvtaper``. Source:
        Czekala et al. (2025), [references](../references.md#mpol).
        Formerly ``taper_fwhm_mas``, a deprecated alias.
    prefix : str
        Site-name prefix.
    """

    @renamed({"taper_fwhm_mas": "blur_fwhm_mas"})
    def __init__(
        self,
        data,
        ref_obs=0,
        gains=True,
        offsets=True,
        sigmas=True,
        alpha=True,
        nu0_hz=None,
        blur_fwhm_mas=None,
        prefix="cal",
        offset_beams=3.0,
        antenna_gains=False,
        solint="scan",
        gain_amp_width=0.1,
        gain_phase_width=0.5,
        fit_gain_widths=False,
        gain_steps=5,
    ):
        self.n_obs = len(data.obs_ids)
        self.n_ddid = len(data.ddid_ids)
        if not 0 <= int(ref_obs) < self.n_obs:
            raise ValueError(f"ref_obs must be in 0..{self.n_obs - 1}.")
        self.ref_obs = int(ref_obs)
        self.gains = bool(gains) and self.n_obs > 1
        self.offsets = bool(offsets) and self.n_obs > 1
        if isinstance(sigmas, str):
            if sigmas != "profile":
                raise ValueError("sigmas must be True, False or 'profile'.")
            self.sigmas = "profile"
        else:
            self.sigmas = bool(sigmas)
        if isinstance(alpha, (bool, onp.bool_)):
            self.alpha, self.alpha_fixed = bool(alpha), 0.0
        elif isinstance(alpha, (int, onp.integer)) and alpha != 0:
            # alpha=1 used to fit α (bool(1)); refuse rather than fix it
            raise ValueError(
                "alpha must be True or False (fit it or not) or a float to "
                f"fix it (alpha={float(alpha)!r}), not the integer {alpha}."
            )
        else:
            self.alpha, self.alpha_fixed = False, float(alpha)
        w = onp.asarray(data.weight, dtype=onp.float64)
        if nu0_hz is None:
            nu0_hz = float(onp.sum(w * onp.asarray(data.freq)) / onp.sum(w))
        self.nu0_hz = float(nu0_hz)
        q = onp.hypot(onp.asarray(data.u), onp.asarray(data.v))
        q_min = float(q[q > 0].min()) if onp.any(q > 0) else onp.inf
        # Starting flux: the weighted mean visibility of the shortest 1% of
        # baselines, which tends to the total flux for a compact source.
        short = onp.argsort(q)[: max(1, q.size // 100)]
        total = onp.sum(
            w[short] * onp.asarray(data.vis)[short].real
        ) / onp.sum(w[short])
        self.flux_start = float(total) if total > 0 else 0.1
        self.offset_bound_mas = float(rad2mas / (2.0 * q_min))
        # b_k = 1/(2π q_max,k) in mas, the natural unit of observation k's
        # offset; the reference observation has no offset site.
        obs = onp.asarray(data.obs)
        q_default = float(q[q > 0].max()) if onp.any(q > 0) else 1.0
        q_max = onp.array(
            [
                float(q[(obs == k) & (q > 0)].max())
                if onp.any((obs == k) & (q > 0))
                else q_default
                for k in range(self.n_obs)
            ]
        )
        self.offset_unit_mas = rad2mas / (2.0 * onp.pi * q_max)
        self.offset_beams = float(offset_beams)
        self.blur_fwhm_mas = (
            None if blur_fwhm_mas is None else float(blur_fwhm_mas)
        )
        self.prefix = prefix
        self.antenna_gains = bool(antenna_gains)
        if self.antenna_gains:
            _antenna_gains.gain_intervals(data, solint)  # validates
            if not (gain_amp_width > 0 and gain_phase_width > 0):
                raise ValueError("the gain widths must be positive.")
            if fit_gain_widths not in (True, False, "amp", "phase"):
                raise ValueError(
                    "fit_gain_widths must be a bool, 'amp' or 'phase'."
                )
        elif fit_gain_widths or solint != "scan" or gain_steps != 5:
            raise ValueError(
                "fit_gain_widths, solint and gain_steps need antenna_gains=True."
            )
        if int(gain_steps) < 1:
            raise ValueError("gain_steps must be at least 1.")
        self.solint = solint
        self.gain_widths = (float(gain_amp_width), float(gain_phase_width))
        self.fit_widths = (
            self.antenna_gains and fit_gain_widths in (True, "amp"),
            self.antenna_gains and fit_gain_widths in (True, "phase"),
        )
        self.gain_steps = int(gain_steps)

    # site names
    def _site(self, name):
        return f"{self.prefix}.{name}"

    @property
    def sites(self):
        """Names of the fitted sites."""
        return tuple(self.priors())

    def priors(self, **bounds):
        """Default priors of the fitted sites, a dict for ``fit``.

        ``bounds`` override the ends ``(lo, hi)`` of a site's prior, keyed by
        ``flux``, ``gain``, ``offset``, ``sigma``, ``alpha``,
        ``gain_amp_width`` or ``gain_phase_width`` (the last two are
        ``LogUniform(1e-3, 10)`` by default). Without a
        bound, the gains and σ scales are improper Jeffreys priors (flat in
        ``ln x``, no box) and the offsets are uniform over
        ``±offset_beams`` beams of their observation (``2π b_k``, see the
        class docstring). ``offset=(-1/(2 q_min), 1/(2 q_min))`` in mas, with
        ``q_min`` the shortest baseline (``offset_bound_mas``), is the old
        default: a shift of the phase centre beyond that is less than half a
        turn of phase on every baseline.
        """
        unknown = set(bounds) - {
            "flux",
            "gain",
            "offset",
            "sigma",
            "alpha",
            *_WIDTHS,
        }
        if unknown:
            raise ValueError(f"unknown bounds {sorted(unknown)}.")
        k = self.n_obs - 1
        lo, hi = bounds.get("flux", (1e-4, 1e2))
        out = {self._site("flux"): dist.LogUniform(lo, hi)}
        if self.gains:
            if "gain" in bounds:
                lo, hi = bounds["gain"]
                out[self._site("gain")] = dist.LogUniform(
                    np.full(k, lo), np.full(k, hi)
                )
            else:
                out[self._site("gain")] = _LogImproper((k,))
        if self.offsets:
            unit = self.offset_unit_mas[onp.arange(self.n_obs) != self.ref_obs]
            unit = np.asarray(unit)[:, None]
            theta = self.offset_beams * 2.0 * onp.pi * unit
            if "offset" in bounds:
                lo, hi = bounds["offset"]
                lo, hi = lo + 0.0 * theta, hi + 0.0 * theta
            else:
                lo, hi = -theta, theta
            shape = (k, 2)
            out[self._site("offset")] = dist.Uniform(
                np.broadcast_to(lo, shape), np.broadcast_to(hi, shape)
            )
        if self.sigmas is True:
            if "sigma" in bounds:
                lo, hi = bounds["sigma"]
                out[self._site("sigma")] = dist.LogUniform(
                    np.full(self.n_ddid, lo), np.full(self.n_ddid, hi)
                )
            else:
                out[self._site("sigma")] = _LogImproper((self.n_ddid,))
        if self.alpha:
            lo, hi = bounds.get("alpha", (0.0, 5.0))
            out[self._site("alpha")] = dist.Uniform(lo, hi)
        for name, fitted in zip(_WIDTHS, self.fit_widths):
            if fitted:
                # bounded: the marginal tends to a constant as τ → 0, so an
                # improper 1/τ prior would give an improper posterior
                lo, hi = bounds.get(name, (1e-3, 10.0))
                out[self._site(name)] = dist.LogUniform(lo, hi)
        return out

    def defaults(self):
        """Starting values of the fitted sites."""
        k = self.n_obs - 1
        out = {self._site("flux"): self.flux_start}
        if self.gains:
            out[self._site("gain")] = onp.ones(k)
        if self.offsets:
            out[self._site("offset")] = onp.zeros((k, 2))
        if self.sigmas is True:
            out[self._site("sigma")] = onp.ones(self.n_ddid)
        if self.alpha:
            out[self._site("alpha")] = 2.0
        for name, fitted, w in zip(_WIDTHS, self.fit_widths, self.gain_widths):
            if fitted:
                out[self._site(name)] = w
        return out

    def _widths(self, values):
        """``(τ_a, τ_φ)``: fitted sites, else the fixed values."""
        return tuple(
            values[self._site(name)] if fitted else w
            for name, fitted, w in zip(
                _WIDTHS, self.fit_widths, self.gain_widths
            )
        )

    def _expand(self, free, fill):
        """Insert the reference observation (value ``fill``) at ``ref_obs``."""
        return np.insert(free, self.ref_obs, fill, axis=0)

    def _unpack(self, values):
        """(flux, gain[n_obs], offset[n_obs, 2], sigma[n_ddid], alpha).

        With ``sigmas="profile"`` the sigma is 1 here; the term sets ŝ.
        """
        site = self._site
        gain = (
            self._expand(np.asarray(values[site("gain")]), 1.0)
            if self.gains
            else np.ones(self.n_obs)
        )
        offset = (
            self._expand(
                np.reshape(np.asarray(values[site("offset")]), (-1, 2)), 0.0
            )
            if self.offsets
            else np.zeros((self.n_obs, 2))
        )
        sigma = (
            np.asarray(values[site("sigma")])
            if self.sigmas is True
            else np.ones(self.n_ddid)  # profiled: set by the term
        )
        alpha = values[site("alpha")] if self.alpha else self.alpha_fixed
        return values[site("flux")], gain, offset, sigma, alpha

    @property
    def taper_fwhm_mas(self):
        """Deprecated alias of ``blur_fwhm_mas``."""
        return self.blur_fwhm_mas

    def _blur(self, u, v):
        if self.blur_fwhm_mas is None:
            return 1.0
        return gaussian_taper(u * u + v * v, self.blur_fwhm_mas)


def _no_antenna_gains(nuisance):
    if nuisance.antenna_gains:
        raise ValueError(
            "this model has no antenna gains applied; with antenna_gains=True "
            "use term.gains(values) for the solved gains."
        )


def _model_points(scene, unpacked, nuisance, u, v, freq, ddid, obs):
    flux, gain, offset, _, alpha = unpacked
    lam = C_LIGHT / freq
    vis = scene.model(u * lam, v * lam, lam)
    ramp = offset_phase(u, v, offset[obs, 0], offset[obs, 1])
    spectral = (freq / nuisance.nu0_hz) ** alpha
    return flux * gain[obs] * spectral * ramp * nuisance._blur(u, v) * vis


# ----------------------------------------------------------------------
# The likelihood term
# ----------------------------------------------------------------------


def _warn_double_blur(scene, nuisance):
    if nuisance.blur_fwhm_mas is None:
        return
    images = jax.tree_util.tree_leaves(
        scene, is_leaf=lambda x: isinstance(x, Image)
    )
    if any(
        isinstance(x, Image) and x.blur_fwhm_mas is not None for x in images
    ):
        warnings.warn(
            "CalibrationNuisance.blur_fwhm_mas and Image.blur_fwhm_mas "
            "are both set; the two Gaussians compound, blurring the model "
            "twice. Use one of them.",
            UserWarning,
            stacklevel=2,
        )


class _Precompute:
    """The builder of a term's Toeplitz form, carried through ``jit`` as a
    static field. Never read it inside a jitted function."""

    def __init__(self, build):
        self.build = build

    # deliberate: read only by ``resolve``, outside jit, so always equal: a new
    # mask must not retrace, and the term jit sees carries only arrays.
    def __eq__(self, other):
        return isinstance(other, _Precompute)

    def __hash__(self):
        return 0


class _RadioTerm(eqx.Module):
    """Whitened residuals of a `ComplexVisData` with calibration nuisances."""

    arrays: dict
    scene: object = eqx.field(static=True)
    nuisance: object = eqx.field(static=True)
    n: int = eqx.field(static=True)
    sigma_fixed: object = None
    gain_index: object = eqx.field(static=True, default=None)
    chi2: str = eqx.field(static=True, default="direct")
    precompute: object = eqx.field(static=True, default=None)

    def __init__(
        self,
        data,
        scene,
        nuisance,
        mask,
        chunk_size,
        sigma=None,
        chi2="direct",
    ):
        if sigma is not None:
            if nuisance.sigmas != "profile":
                raise ValueError("sigma= needs sigmas='profile'.")
            sigma = np.asarray(sigma, dtype=np.float64)
            if sigma.shape != (nuisance.n_ddid,):
                raise ValueError(
                    f"sigma must have shape ({nuisance.n_ddid},)."
                )
        self.sigma_fixed = sigma
        n = len(data)
        if mask is None:
            mask = onp.ones(n, dtype=bool)
        mask = onp.asarray(mask, dtype=bool)
        if mask.shape != (n,):
            raise ValueError(f"mask has shape {mask.shape}, not ({n},).")
        chunk = max(1, min(int(chunk_size), n))
        pad = (-n) % chunk

        def padded(x, fill, dtype):
            x = onp.asarray(x, dtype=dtype)
            x = onp.concatenate([x, onp.full(pad, fill, dtype=dtype)])
            return np.asarray(x.reshape(-1, chunk))

        self.arrays = dict(
            u=padded(data.u, 0.0, onp.float64),
            v=padded(data.v, 0.0, onp.float64),
            vis=padded(data.vis, 0.0, onp.complex128),
            freq=padded(data.freq, float(data.freq[0]), onp.float64),
            ddid=padded(data.ddid, 0, onp.int32),
            obs=padded(data.obs, 0, onp.int32),
            # zero weight outside the mask and on the padding
            sqrt_w=padded(
                onp.sqrt(onp.asarray(data.weight)) * mask, 0.0, onp.float64
            ),
            m=padded(mask, False, onp.float64),
            w=padded(onp.asarray(data.weight), 1.0, onp.float64),
        )
        self.gain_index = None
        if nuisance.antenna_gains:
            # padding and masked points have zero weight, so index 0 is inert
            interval, a1, a2, n_s, n_a = _antenna_gains.gain_intervals(
                data, nuisance.solint
            )
            self.arrays.update(
                interval=padded(interval, 0, onp.int32),
                a1=padded(a1, 0, onp.int32),
                a2=padded(a2, 0, onp.int32),
            )
            self.gain_index = (n_s, n_a)
        self.scene = scene
        self.nuisance = nuisance
        self.n = n
        self.chi2 = chi2
        self.precompute = _Precompute(
            lambda image: data.toeplitz(
                image,
                nuisance,
                mask,
                "spread" if nuisance.offsets else "none",
            )
        )

    def resolve(self, start, least_squares):
        """``(term, form)``: the Toeplitz term of ``fit`` at ``start`` if ``chi2``
        asks for it and it is exact, else ``self``, with a note of what ran;
        ``"toeplitz"`` raises if it is not exact."""
        if self.chi2 == "direct":
            return self, "direct"
        # Imported here: radio_toeplitz imports this module.
        from .radio_toeplitz import _single_image

        nuis, auto = self.nuisance, self.chi2 == "auto"
        unmet = []
        if least_squares:
            unmet.append("the fit is least squares (use lbfgs or adam)")
        if nuis.antenna_gains:
            unmet.append("antenna_gains is set")
        if nuis.alpha:
            unmet.append("alpha is fitted")
        if nuis.offsets and auto:
            # deliberate: free offsets use the derived spread grid, opt-in
            unmet.append("offsets are fitted (explicit chi2='toeplitz' only)")
        floats = [
            x
            for x in jax.tree.leaves(start)
            if np.issubdtype(x.dtype, np.floating)
        ]
        if any(x.dtype != np.float64 for x in floats):
            unmet.append("the fit is not float64")
        scene = self.scene(start)
        try:
            image, shifts = _single_image(scene)
        except (TypeError, ValueError):
            unmet.append("the scene is not one Image")
        else:
            # The Image's offsets must not depend on any fitted value: follow
            # the dependence of the traced outputs on the inputs.
            jaxpr = jax.make_jaxpr(
                lambda values: _single_image(self.scene(values))[1]
            )(start).jaxpr
            fitted = {id(v) for v in jaxpr.invars}  # ids: Literals don't hash
            for eqn in jaxpr.eqns:
                if any(id(v) in fitted for v in eqn.invars):
                    fitted.update(id(v) for v in eqn.outvars)
            if any(id(v) in fitted for v in jaxpr.outvars):
                unmet.append("the Image's dra or ddec is fitted")
            side, other = image.eta.shape
            if side != other:
                unmet.append("the Image is not square")
            pixels = side * other * nuis.n_ddid
            if auto and float(self.arrays["m"].sum()) < max(
                _FT_DEFAULTS["toeplitz_min_points"],
                _FT_DEFAULTS["toeplitz_points_per_pixel"] * pixels,
            ):
                unmet.append("there are too few points per pixel and ddid")
            if auto and 24 * pixels > _FT_DEFAULTS["toeplitz_max_bytes"]:
                unmet.append("the beams would exceed toeplitz_max_bytes")
        if unmet:
            if auto:
                return self, "direct (auto): " + "; ".join(unmet)
            raise ValueError(
                "chi2='toeplitz' is not exact here: " + "; ".join(unmet) + "."
            )
        shifts = [float(x) for x in shifts]
        layout = eqx.tree_at(
            lambda i: (i.dra, i.ddec),
            image,
            (sum(shifts[0::2]), sum(shifts[1::2])),
        )
        tdata = self.precompute.build(layout)
        return tdata.term(self.scene, nuis, self.sigma_fixed), "toeplitz"

    @property
    def has_log_norm(self):
        return bool(self.nuisance.sigmas) or self.nuisance.antenna_gains

    @property
    def _profiled(self):
        return self.nuisance.sigmas == "profile"

    def _evaluate(self, values):
        """``(re, im, sigma, prior, norm)``. With antenna gains the residuals
        are at the gain solution, ``prior`` holds the whitened solved
        ``(λ, φ)`` and ``norm`` the Laplace terms; else ``prior`` is empty
        and ``norm`` zero."""
        nuis = self.nuisance
        unpacked = nuis._unpack(values)
        sigma = unpacked[3]
        scene = self.scene(values)
        _warn_double_blur(scene, nuis)

        def model(a):
            return _model_points(
                scene,
                unpacked,
                nuis,
                a["u"],
                a["v"],
                a["freq"],
                a["ddid"],
                a["obs"],
            )

        if self.gain_index is not None:
            a = self.arrays
            modelled = jax.lax.map(model, a)

            def solve(sigma):
                lam, phi, norm, sqrt_w, taus, _ = self._gain_solution(
                    values, modelled, sigma
                )
                si, i, j = a["interval"], a["a1"], a["a2"]
                gain = np.exp(
                    lam[si, i] + lam[si, j] + 1j * (phi[si, i] - phi[si, j])
                )
                return (
                    lam,
                    phi,
                    norm,
                    taus,
                    (a["vis"] - gain * modelled) * sqrt_w,
                )

            if self._profiled and self.sigma_fixed is None:
                r = solve(sigma)[-1]
                sigma = self._profile_sigma(np.real(r), np.imag(r))
            elif self._profiled:
                sigma = self.sigma_fixed
            lam, phi, norm, (tau_a, tau_p), r = solve(sigma)
            prior = np.concatenate(
                [np.ravel(lam) / tau_a, np.ravel(phi) / tau_p]
            )
            return np.real(r), np.imag(r), sigma, prior, norm

        def chunk(a):
            r = (a["vis"] - model(a)) * a["sqrt_w"] / sigma[a["ddid"]]
            return np.real(r), np.imag(r)

        re, im = jax.lax.map(chunk, self.arrays)
        if self._profiled:
            sigma = (
                self._profile_sigma(re, im)
                if self.sigma_fixed is None
                else self.sigma_fixed
            )
            scale = sigma[self.arrays["ddid"]]
            re, im = re / scale, im / scale
        return re, im, sigma, np.zeros(0), 0.0

    def _gain_solution(self, values, model, sigma):
        a = self.arrays
        n_s, n_a = self.gain_index
        tau_a, tau_p = self.nuisance._widths(values)
        sqrt_w = a["sqrt_w"] / sigma[a["ddid"]]
        flat = lambda x: np.reshape(x, -1)  # noqa: E731
        lam, phi, norm, step = _antenna_gains.marginal(
            flat(model),
            flat(a["vis"]),
            flat(sqrt_w) ** 2,
            flat(a["interval"]),
            flat(a["a1"]),
            flat(a["a2"]),
            n_s,
            n_a,
            tau_a,
            tau_p,
            self.nuisance.gain_steps,
        )
        return lam, phi, norm, sqrt_w, (tau_a, tau_p), step

    def gains(self, values):
        """Solved antenna gains ``exp(λ̂ + iφ̂)``, shape ``(n_interval, n_ant)``,
        at the point the likelihood is evaluated."""
        n_s, n_a = self.gain_index
        prior = self._evaluate(values)[3]
        tau_a, tau_p = self.nuisance._widths(values)
        lam = prior[: n_s * n_a] * tau_a
        phi = prior[n_s * n_a :] * tau_p
        return np.exp(lam + 1j * phi).reshape(n_s, n_a)

    def _profile_sigma(self, re, im):
        """``ŝ_d = sqrt(χ²_d(1) / 2 N_d)`` (1 where ddid ``d`` has no points),
        held by ``stop_gradient`` (envelope theorem)."""
        a = self.arrays
        n_ddid = self.nuisance.n_ddid
        ddid = np.reshape(a["ddid"], -1)
        chi2 = jax.ops.segment_sum(np.reshape(re**2 + im**2, -1), ddid, n_ddid)
        count = jax.ops.segment_sum(np.reshape(a["m"], -1), ddid, n_ddid)
        s2 = chi2 / (2.0 * np.maximum(count, 1.0))
        s2 = np.where(count > 0, np.maximum(s2, np.finfo(s2.dtype).tiny), 1.0)
        return jax.lax.stop_gradient(np.sqrt(s2))

    def sigma_hat(self, values):
        """The error scales ``s_d`` at ``values``: ``ŝ`` if profiled, else the
        fitted ``cal.sigma`` (ones if not fitted)."""
        return self._evaluate(values)[2]

    def derived(self, values):
        """Quantities of the fit that are not sites: the profiled ``ŝ``."""
        out = {}
        if self._profiled:
            out[self.nuisance._site("sigma")] = self.sigma_hat(values)
        if self.gain_index is not None:
            out[self.nuisance._site("gain_step")] = self.gain_step(values)
        return out

    def gain_step(self, values):
        """The largest change of ``(λ, φ)`` in the last Gauss–Newton step: the
        gain solution has converged where it is small (raise ``gain_steps``
        otherwise)."""
        a = self.arrays
        nuis = self.nuisance
        unpacked, scene = nuis._unpack(values), self.scene(values)
        model = jax.lax.map(
            lambda c: _model_points(
                scene,
                unpacked,
                nuis,
                c["u"],
                c["v"],
                c["freq"],
                c["ddid"],
                c["obs"],
            ),
            a,
        )
        return self._gain_solution(values, model, self.sigma_hat(values))[-1]

    def _flat(self, x):
        return np.reshape(x, -1)[: self.n]

    def __call__(self, values):
        re, im, _, prior, _ = self._evaluate(values)
        return np.concatenate([self._flat(re), self._flat(im), prior])

    def log_norm(self, values):
        """``Σ_k m_k 2 ln s_d(k)``, which ``fit`` adds to ½χ²."""
        return self.residuals_and_log_norm(values)[1]

    def residuals_and_log_norm(self, values):
        """``(term(values), log_norm(values))`` from one evaluation of the
        model (``fit`` uses it, so that profiling ``ŝ`` costs no second pass)."""
        re, im, sigma, prior, norm = self._evaluate(values)
        r = np.concatenate([self._flat(re), self._flat(im), prior])
        a = self.arrays
        return r, 2.0 * np.sum(a["m"] * np.log(sigma[a["ddid"]])) + norm

    def loglike(self, values):
        """Normalized Gaussian log density of the unmasked points."""
        a = self.arrays
        r, log_norm = self.residuals_and_log_norm(values)
        const = np.sum(a["m"] * (np.log(2.0 * np.pi) - np.log(a["w"])))
        return -0.5 * np.sum(r**2) - log_norm - const


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def primary_beam(npix, pixel_scale_mas, freq_hz, dish_m, factor):
    """Log primary-beam attenuation ``-4 ln 2 r² / FWHM²`` on the pixel grid.

    ``FWHM = factor c / (ν D)`` and ``r`` is the distance from the image
    centre (``factor`` is about 1.0–1.3 depending on the illumination: for
    example 12 m dishes with ``factor = 1.13``, as ALMA's, or 22 m dishes for
    ATCA; use the telescope's documented values). The pixel grid is that of
    [`pixel_offsets`][virgil._geometry.pixel_offsets]). Add it to an
    [`Image`][virgil.models.Image]'s log-brightness so that the image is the
    sky times the beam.
    """
    fwhm_mas = factor * C_LIGHT / (freq_hz * dish_m) * rad2mas
    x = pixel_offsets(npix, pixel_scale_mas)
    xx, yy = onp.meshgrid(x, x, indexing="xy")
    return -4.0 * onp.log(2.0) * (xx**2 + yy**2) / fwhm_mas**2


def dartboard_folds(data, k=5, n_q=12, n_phi=16, seed=0):
    """Cross-validation folds from uv cells (a "dartboard").

    The uv plane is cut into ``n_q`` log-spaced annuli between the smallest
    and largest ``q = √(u² + v²)`` and ``n_phi`` azimuth sectors of
    ``φ = arctan2(v, u) mod π``, so a point and its Hermitian partner share a
    cell. The occupied cells are shuffled into ``k`` folds with
    ``numpy.random.default_rng(seed)``.

    Returns
    -------
    fold, annulus : (N,) int arrays
        The fold (0..k-1) and the annulus index of every point.
    """
    u, v = (
        onp.asarray(data.u, dtype=onp.float64),
        onp.asarray(data.v, dtype=onp.float64),
    )
    # Fold into the upper half plane so (u, v) and (-u, -v) coincide exactly.
    flip = (v < 0) | ((v == 0) & (u < 0))
    u, v = onp.where(flip, -u, u), onp.where(flip, -v, v)
    q = onp.hypot(u, v)
    positive = q[q > 0]
    if positive.size == 0:
        raise ValueError("all points are at the origin of the uv plane.")
    edges = onp.geomspace(positive.min(), positive.max(), n_q + 1)
    annulus = onp.clip(
        onp.searchsorted(edges, q, side="right") - 1, 0, n_q - 1
    )
    phi = onp.arctan2(v, u)  # [0, π]
    sector = onp.clip((phi / onp.pi * n_phi).astype(int), 0, n_phi - 1)
    cell = annulus * n_phi + sector
    cells = onp.unique(cell)
    if cells.size < k:
        raise ValueError(f"only {cells.size} occupied uv cells for {k} folds.")
    order = onp.random.default_rng(seed).permutation(cells.size)
    fold_of = onp.empty(cells.size, dtype=int)
    fold_of[order] = onp.arange(cells.size) % k
    fold = fold_of[onp.searchsorted(cells, cell)]
    return fold, annulus
