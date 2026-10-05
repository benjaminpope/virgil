import warnings
import jax
import jax.numpy as np

import numpy as onp

import equinox as eqx
import zodiax as zx

from ._closure import ClosureNoise
from ._utils import inflate_errors
from .gains import ClosureOffsets, GainModes, closure_offsets, gain_modes
from .observables import (
    KINDS,
    DifferentialPhase,
    FluxSpectrum,
    TripleAmplitude,
    VisibilityAmplitude,
)
from ._geometry import UVGrid, find_uv_grid  # noqa: F401 (re-exported)
from .amigo import is_mixed_disco_record, mixed_disco_fields
from .oifits import read_oifits


__all__ = ["OIData", "closure_phases", "cp_indices"]


class OIData(zx.Base):  # type: ignore[reportGeneralTypeIssues]
    """
    Store and transform optical-interferometry observables.

    Parameters
    ----------
    data : dict, str, os.PathLike, astropy.io.fits.HDUList, or a list of files
        An OIFITS file (a path, or a file opened with ``astropy.io.fits`` or
        ``pyoifits``), a list of such files (concatenated), or a dictionary
        of arrays; see ``__init__``.
    target : str or int, optional
        For OIFITS input, the target to keep (by name or ``TARGET_ID``).
        Required when the file contains more than one target.

    Notes
    -----
    Every (baseline, wavelength) sample is one element of the flat ``u``,
    ``v`` (metres) and ``wavel`` (metres) arrays; ``wavel`` has a single
    element when all samples share one wavelength. Models are evaluated on
    these samples. The observables are:

    * ``vis``/``d_vis``: squared visibilities (``v2_flag=True``) or
      amplitudes, or their projection through ``vis_mat``.
    * ``phi``/``d_phi``: closure phases (``cp_flag=True``) built from the
      samples ``i_cps1 + i_cps2 - i_cps3``, or absolute phases; always in
      radians, optionally projected through ``phi_mat``.

    Visibility-only data (no ``OI_T3`` or ``VISPHI``, no ``phi`` in a
    dictionary, or every closure phase flagged) have an empty phase block:
    ``has_phases`` is False and fits use the visibilities alone.

    Flagged samples are left out of the observables. ``vis_index`` (and
    ``phi_index`` for absolute phases) then lists the samples that are
    observed; they are ``None`` when every sample is used.

    ``uv_grid`` is a [`UVGrid`][virgil.oidata.UVGrid] for AMIGO DISCO
    products, whose samples lie on a regular (possibly rotated) lattice,
    and ``None`` for all other data, even if their samples happen to lie
    on a lattice: virgil does not search ordinary data for one. Models
    that can use the lattice, such as an [`Image`][virgil.models.Image]
    with matching ``rotation_deg``, then evaluate faster; the results are
    the same. To use it for other lattice-sampled data, set it with
    [`find_uv_grid`][virgil.oidata.find_uv_grid], e.g.
    ``eqx.tree_at(lambda d: d.uv_grid, data, find_uv_grid(data.u, data.v),
    is_leaf=lambda x: x is None)``.

    Data read from OIFITS also keep their time and exposure per sample:
    ``frame`` numbers the exposures (frames), :attr:`mjd` gives each
    sample's time, and :meth:`epochs` and :meth:`split_by_epoch` group the
    frames into nights. The time is stored as ``dt``, days since the static
    float64 ``t_ref``, so that models of time see small numbers that keep
    their precision in float32 (about 5 s over 1000 days). ``stations``
    holds each sample's station pair (``STA_INDEX``).

    ``gains`` holds calibration gains correlated across channels
    ([`GainModes`][virgil.gains.GainModes], set with
    [`with_gains`][virgil.oidata.OIData.with_gains]), which the likelihood
    marginalises; ``None`` by default. ``phase_offsets`` likewise holds
    closure-phase offsets per frame
    ([`ClosureOffsets`][virgil.gains.ClosureOffsets], set with
    [`with_closure_offsets`][virgil.oidata.OIData.with_closure_offsets]).

    ``extras`` holds further observables, read with ``extras=`` (OI_FLUX,
    T3AMP, VISAMP beside V², differential VISPHI beside closure phases):
    a tuple of blocks from [`virgil.observables`][virgil.observables],
    which follow the phases in the data vector, in the order of
    ``observables.KINDS``. It is empty by default.
    """

    u: jax.Array
    v: jax.Array
    wavel: jax.Array
    vis: jax.Array
    d_vis: jax.Array
    phi: jax.Array
    d_phi: jax.Array
    i_cps1: jax.Array | onp.ndarray | None
    i_cps2: jax.Array | onp.ndarray | None
    i_cps3: jax.Array | onp.ndarray | None
    vis_mat: jax.Array | None
    phi_mat: jax.Array | None
    vis_index: jax.Array | None
    phi_index: jax.Array | None
    uv_grid: UVGrid | None
    cp_noise: ClosureNoise | None
    dt: jax.Array | None
    frame: jax.Array | None
    stations: jax.Array | None
    gains: GainModes | None
    phase_offsets: ClosureOffsets | None
    extras: tuple
    observable_kind: str = eqx.field(static=True)
    vis_mode: str = eqx.field(static=True)
    v2_flag: bool = eqx.field(static=True)
    cp_flag: bool = eqx.field(static=True)
    t_ref: float | None = eqx.field(static=True)

    def __init__(self, data, target=None, extras=()):
        """
        Initialize from an OIFITS file or explicit arrays.

        Parameters
        ----------
        data : dict, str, os.PathLike, astropy.io.fits.HDUList, or a list of files
            An OIFITS file or a list of them, read with
            [`virgil.oifits.read_oifits`][virgil.oifits.read_oifits]
            (several files, wavelength channels, tables and epochs, and
            ``FLAG`` columns, are supported). Or a dictionary with keys:

            * ``u``, ``v`` (metres) and ``wavel`` (metres): per sample, or
              ``u``/``v`` per baseline with ``vis`` of shape
              ``(n_baseline, n_wavel)`` for several channels.
            * ``vis``, ``d_vis``: squared visibilities or amplitudes.
            * ``phi``, ``d_phi``: closure or absolute phases, with
              ``phi_unit`` (``"rad"``, the default, or ``"deg"``;
              ``phase_unit`` is an alias). For several channels, shape
              ``(n_triangle, n_wavel)`` or ``(n_baseline, n_wavel)``.
            * ``i_cps1``, ``i_cps2``, ``i_cps3`` (optional): for each
              closure phase, the baselines ``(a, b)``, ``(b, c)`` and
              ``(a, c)`` of its triangle.
            * ``v2_flag`` (default True): ``vis`` holds squared
              visibilities; otherwise amplitudes.
            * ``cp_flag`` (default: True when closure indices are given):
              ``phi`` holds closure phases; otherwise absolute phases.
            * ``vis_flag``, ``phi_flag`` (optional): boolean masks, True
              for bad samples, shaped like ``vis`` and ``phi``. Samples with
              non-finite values or errors are flagged automatically.
            * ``vis_mode`` (``"auto"``, ``"v2"``, ``"amp"`` or ``"logamp"``;
              ``observable_kind`` is an alias): the visibility channel that
              data and model are compared in. ``"auto"`` keeps the channel
              of ``vis``.
            * ``mjd``, ``frame`` (optional): the time (days) and an integer
              exposure label of each sample (or of each baseline, for
              several channels). Without ``frame``, samples with the same
              ``mjd`` form one frame.
            * ``stations`` (optional): the station pair ``(a, b)``
              (``STA_INDEX``) of each sample, or of each baseline for
              several channels, for telescope and baseline gains
              ([`with_gains`][virgil.oidata.OIData.with_gains]).
            * ``vis_mat``, ``phi_mat`` (optional): linear operators of shape
              ``(n_out, n_in)`` projecting the channels into, e.g., kernel
              or DISCO observables. The ``disco_vis_mat``/``disco_phi_mat``
              spellings also check that the projected covariance is
              diagonal. Only diagonal uncertainties are propagated.

            A record with ``disco_coefficients`` is read as an AMIGO
            mixed-DISCO product (see [`load_oi_data`][virgil.amigo.load_oi_data]); its ``u`` and
            ``v`` are negated to match the virgil sign convention.
            * ``flux``/``nflux``, ``visamp``, ``t3amp``, ``visphi``
              (optional): extra observables, as dictionaries in the layout
              [`read_oifits`][virgil.oifits.read_oifits] returns (see
              ``extras``). ``visamp`` and ``visphi`` give per value the
              ``sample`` it belongs to, with ``value``, ``error`` and
              ``flag`` (``visamp`` also ``amptyp``); ``t3amp`` has one value
              per closure phase; ``flux`` gives ``wavel``, ``value``,
              ``error``, ``flag`` and optionally ``row``, ``mjd`` and
              ``station``.
        target : str or int, optional
            For OIFITS input, the target to keep.
        extras : sequence of str, optional
            For OIFITS input, the extra observables to read (see
            [`read_oifits`][virgil.oifits.read_oifits]).
        """
        if not isinstance(data, dict):
            data = read_oifits(data, target=target, extras=extras)
        elif target is not None:
            raise ValueError("target only applies to OIFITS input.")
        elif extras:
            raise ValueError(
                "extras only applies to OIFITS input; a dictionary carries "
                "its extra observables as keys."
            )

        if is_mixed_disco_record(data):
            for name, value in mixed_disco_fields(data).items():
                setattr(self, name, value)
            self.dt = self.frame = self.t_ref = None
            self.stations = self.gains = self.phase_offsets = None
            self.extras = ()
            return

        u = onp.asarray(data["u"], dtype=float)
        v = onp.asarray(data["v"], dtype=float)
        wavel = onp.atleast_1d(onp.asarray(data["wavel"], dtype=float))
        vis = onp.asarray(data["vis"], dtype=float)
        d_vis = onp.asarray(data["d_vis"], dtype=float)
        phi_unit = data.get("phi_unit", data.get("phase_unit", "rad"))
        # Visibility-only data have no phases: no "phi", or an empty one.
        phi_in, d_phi_in = data.get("phi"), data.get("d_phi")
        phi, d_phi = self._phase_to_radians(
            onp.zeros(0) if phi_in is None else onp.asarray(phi_in, float),
            onp.zeros(0) if d_phi_in is None else onp.asarray(d_phi_in, float),
            phi_unit,
            default_unit="rad",
        )
        phi, d_phi = onp.asarray(phi), onp.asarray(d_phi)
        # (Projected phases may have fewer values than raw errors, so only
        # one of the two being empty is an error.)
        if (phi.size == 0) != (d_phi.size == 0):
            raise ValueError(
                f"phi has {phi.size} values but d_phi has {d_phi.size}: give "
                "both, or neither for visibility-only data."
            )

        indices = [data.get(key) for key in ("i_cps1", "i_cps2", "i_cps3")]
        if any(index is None for index in indices):
            indices = None
        else:
            indices = [onp.asarray(index, dtype=int) for index in indices]

        v2_flag = self._coerce_bool_flag(data.get("v2_flag", True), "v2_flag")
        cp_flag = self._coerce_bool_flag(
            data.get("cp_flag", indices is not None), "cp_flag"
        )
        if cp_flag and indices is None:
            raise ValueError(
                "cp_flag=True needs the closure-phase indices i_cps1, "
                "i_cps2 and i_cps3."
            )
        vis_flag = data.get("vis_flag")
        phi_flag = data.get("phi_flag")
        mjd, frame = data.get("mjd"), data.get("frame")
        stations = data.get("stations")
        if mjd is not None:
            mjd = onp.asarray(mjd, dtype=onp.float64)
        if frame is not None:
            frame = onp.asarray(frame)
        if stations is not None:
            stations = onp.asarray(stations, dtype=int).reshape(-1, 2)
        if vis.ndim == 2:
            # Per-baseline times are repeated over channels, like u and v;
            # per-sample ones (shaped like vis, or flat) are flattened.
            def per_sample(values):
                if values is None or values.size == vis.size:
                    return None if values is None else values.reshape(-1)
                return onp.repeat(values, vis.shape[1])

            mjd, frame = per_sample(mjd), per_sample(frame)
            if stations is not None and len(stations) == vis.shape[0]:
                stations = onp.repeat(stations, vis.shape[1], axis=0)
        if vis.ndim == 2:
            u, v, wavel, indices = _expand_channels(u, v, wavel, vis, indices)
            vis, d_vis = vis.reshape(-1), d_vis.reshape(-1)
            phi, d_phi = phi.reshape(-1), d_phi.reshape(-1)
            vis_flag = None if vis_flag is None else onp.ravel(vis_flag)
            phi_flag = None if phi_flag is None else onp.ravel(phi_flag)
        elif wavel.size not in (1, u.size):
            raise ValueError(
                f"wavel has {wavel.size} values for {u.size} samples. For "
                "several wavelength channels give vis with shape "
                "(n_baseline, n_wavel), or give one wavelength per sample."
            )

        has_disco_vis = "disco_vis_mat" in data
        has_disco_phi = "disco_phi_mat" in data
        vis_mat_in = data.get("disco_vis_mat", data.get("vis_mat", None))
        phi_mat_in = data.get("disco_phi_mat", data.get("phi_mat", None))
        vis_mat = (
            None if vis_mat_in is None else onp.asarray(vis_mat_in, float)
        )
        phi_mat = (
            None if phi_mat_in is None else onp.asarray(phi_mat_in, float)
        )

        # Drop flagged samples from the observables. Their baselines stay in
        # u and v, because closure phases may still need them.
        vis_index = None
        keep = _good_samples(vis, d_vis, vis_flag, u.size, "vis")
        if keep is not None:
            if vis_mat is not None:
                raise ValueError(
                    "Flagged visibilities cannot be combined with vis_mat; "
                    "remove the flagged samples and the matching operator "
                    "columns first."
                )
            vis_index = onp.flatnonzero(keep)
            vis, d_vis = vis[keep], d_vis[keep]

        phi_index = None
        all_legs = indices  # before flagging, for the triple amplitudes
        has_extras = any(kind in data for kind in _RECORD_KEYS)
        no_phases = not cp_flag and phi.size == 0 and phi_mat is None
        if no_phases:
            # Visibilities alone: an empty phase block (no phase sample is
            # observed), so every phase term below has length zero.
            phi_index = onp.zeros(0, dtype=int)
        else:
            n_phi = len(indices[0]) if cp_flag else u.size
            keep = _good_samples(phi, d_phi, phi_flag, n_phi, "phi")
        if not no_phases and keep is not None:
            if phi_mat is not None:
                raise ValueError(
                    "Flagged phases cannot be combined with phi_mat; "
                    "remove the flagged samples and the matching operator "
                    "columns first."
                )
            phi, d_phi = phi[keep], d_phi[keep]
            if cp_flag and phi.size == 0:
                warnings.warn(
                    "Every closure phase is flagged (or not finite): using "
                    "the visibilities alone.",
                    stacklevel=2,
                )
                cp_flag, indices = False, None
                phi_index = onp.zeros(0, dtype=int)
            elif cp_flag:
                indices = [index[keep] for index in indices]
            else:
                phi_index = onp.flatnonzero(keep)
        if vis.size == 0 and phi.size == 0 and not has_extras:
            raise ValueError(
                "No unflagged data: every visibility and every phase is "
                "flagged (or not finite), or there are none."
            )

        self.u = np.asarray(u)
        self.v = np.asarray(v)
        self.wavel = np.asarray(wavel)
        self.vis = np.asarray(vis)
        self.d_vis = np.asarray(d_vis)
        self.phi = np.asarray(phi)
        self.d_phi = np.asarray(d_phi)
        self.i_cps1, self.i_cps2, self.i_cps3 = (
            (None, None, None)
            if indices is None
            else tuple(np.asarray(index) for index in indices)
        )
        self.v2_flag = v2_flag
        self.cp_flag = cp_flag
        self.vis_mat = None if vis_mat is None else np.asarray(vis_mat)
        self.phi_mat = None if phi_mat is None else np.asarray(phi_mat)
        self.vis_index = None if vis_index is None else np.asarray(vis_index)
        self.phi_index = None if phi_index is None else np.asarray(phi_index)
        # Closure phases of triangles sharing baselines are correlated, and
        # only some of them are independent: whiten them as a group (see
        # _closure). A phase operator applied to them gets their full
        # covariance instead (_transform_observed_channels).
        closure = ClosureNoise.from_indices(*indices) if cp_flag else None
        self.cp_noise = closure if phi_mat is None else None
        self._set_times(mjd, frame)
        if stations is not None and len(stations) != onp.size(self.u):
            raise ValueError(
                f"stations has {len(stations)} station pairs for "
                f"{onp.size(self.u)} samples; give one per sample (or per "
                "baseline, for several channels)."
            )
        self.stations = (
            None if stations is None else np.asarray(stations, np.int32)
        )
        self.gains = None
        self.phase_offsets = None
        self.extras = _build_extras(data, self, all_legs)
        vis_mode_in = data.get(
            "vis_mode", data.get("observable_vis_mode", "auto")
        )
        self.vis_mode = self._resolve_vis_mode(vis_mode_in)
        self.uv_grid = None
        self.observable_kind = "split"
        self._transform_observed_channels(
            validate_vis_covariance=has_disco_vis,
            validate_phi_covariance=has_disco_phi,
            closure=closure,
        )
        if (
            np.asarray(self.vis).size == 0
            and np.asarray(self.phi).size == 0
            and not self.extras
        ):
            # A projection can drop rows too (zero-variance operator rows).
            raise ValueError(
                "No data left: the operators (vis_mat, phi_mat) project "
                "every observable away."
            )

    def _set_times(self, mjd, frame):
        """Store ``mjd`` as ``t_ref`` + ``dt`` and ``frame``, per sample."""
        n = onp.size(self.u)
        for name, values in (("mjd", mjd), ("frame", frame)):
            if values is not None and onp.shape(values) != (n,):
                raise ValueError(
                    f"{name} has shape {onp.shape(values)} but there are "
                    f"{n} samples; give one value per sample (or per "
                    "baseline, for several channels)."
                )
        if frame is None and mjd is not None:
            frame = onp.unique(mjd, return_inverse=True)[1]
        if mjd is None:
            self.t_ref, self.dt = None, None
        else:
            self.t_ref = float(onp.min(mjd))
            self.dt = np.asarray(mjd - self.t_ref)
        # int32 is the same in both x64 modes (see _closure.ClosureNoise).
        self.frame = None if frame is None else np.asarray(frame, np.int32)

    @property
    def mjd(self):
        """Time of each sample (days, float64), or ``None`` if unknown."""
        if self.dt is None:
            return None
        return self.t_ref + onp.asarray(self.dt, dtype=onp.float64)

    def epochs(self, gap_days=0.5):
        """Label each sample with its epoch: a run of frames with no gap.

        Parameters
        ----------
        gap_days : float, optional
            Frames whose mean times are more than this far apart (days)
            are in different epochs; the default separates nights. A frame
            is never split between epochs.

        Returns
        -------
        numpy.ndarray
            Integer epoch of each sample, numbered in time order from 0.
        """
        mjd = self.mjd
        if mjd is None:
            raise ValueError(
                "These data have no times: read them from OIFITS, or give "
                "mjd per sample."
            )
        # Whole frames go into one epoch, at the mean time of their samples.
        frames, frame_of = onp.unique(
            onp.asarray(self.frame), return_inverse=True
        )
        times = onp.bincount(frame_of, weights=mjd) / onp.bincount(frame_of)
        order = onp.argsort(times)
        sorted_epochs = onp.concatenate(
            [[0], onp.cumsum(onp.diff(times[order]) > gap_days)]
        )
        epoch_of_frame = onp.empty(frames.size, dtype=int)
        epoch_of_frame[order] = sorted_epochs
        return epoch_of_frame[frame_of]

    def split_by_epoch(self, gap_days=0.5):
        """One [`OIData`][virgil.oidata.OIData] per epoch, in time order.

        See :meth:`epochs`. Each part keeps its own samples, observables and
        closure phases, so it can be fitted on its own or with a model per
        epoch. Not available for projected (kernel, DISCO) observables, nor
        for data with a gain mode shared between epochs (a supplied mode
        spanning frames): the parts' likelihoods would then not add up to
        the whole.
        """
        labels = self.epochs(gap_days)
        if self.gains is not None:
            index = (
                onp.arange(onp.size(self.u))
                if self.vis_index is None
                else onp.asarray(self.vis_index)
            )
            if self.gains.labels_shared(labels[index]):
                raise ValueError(
                    "A gain mode spans several epochs, so the epochs are not "
                    "independent and their likelihoods would not add up; "
                    "fit the data together, or drop that mode first."
                )
        flux_labels = self._flux_epochs(labels)
        return [
            self._subset(
                labels == k,
                lambda b: flux_labels[id(b)] == k,  # noqa: B023
            )
            for k in range(labels.max() + 1)
        ]

    def _flux_epochs(self, labels):
        """Epoch of each sample of the OI_FLUX blocks, by nearest frame time."""
        out = {}
        blocks = [
            b
            for b in self.extras
            if isinstance(b, FluxSpectrum) and b.sample is None
        ]
        if not blocks:
            return out
        mjd = self.mjd
        frames, frame_of = onp.unique(
            onp.asarray(self.frame), return_inverse=True
        )
        times = onp.bincount(frame_of, weights=mjd) / onp.bincount(frame_of)
        epoch_of_frame = onp.zeros(frames.size, dtype=int)
        epoch_of_frame[frame_of] = labels
        for b in blocks:
            if b.mjd is None:
                if labels.max() > 0:
                    raise ValueError(
                        "These OI_FLUX data have no times, so they cannot be "
                        "split by epoch."
                    )
                out[id(b)] = onp.zeros(b.wavel.size, dtype=int)
                continue
            nearest = onp.argmin(
                onp.abs(b.mjd[:, None] - times[None, :]), axis=1
            )
            out[id(b)] = epoch_of_frame[nearest]
        return out

    def select(self, wavel_min=None, wavel_max=None):
        """These data restricted to a wavelength range.

        Parameters
        ----------
        wavel_min, wavel_max : float, optional
            Keep the samples with ``wavel_min <= wavel <= wavel_max``
            (metres); either bound may be left open.

        Returns
        -------
        OIData
            The samples in range, with their observables and closure phases
            (a closure triangle's legs share a wavelength, so triangles are
            kept whole). Not available for projected (kernel, DISCO)
            observables.

        Examples
        --------
        Keep the K-band continuum of a GRAVITY file but not the Brγ window:
        ``data.select(2.05e-6, 2.16e-6)``.
        """
        wavel = onp.broadcast_to(onp.asarray(self.wavel), onp.shape(self.u))
        keep = onp.ones(wavel.shape, dtype=bool)
        if wavel_min is not None:
            keep &= wavel >= wavel_min
        if wavel_max is not None:
            keep &= wavel <= wavel_max
        if not keep.any():
            raise ValueError(
                f"No samples between {wavel_min} and {wavel_max} m; the data "
                f"span {wavel.min():.4g} to {wavel.max():.4g} m."
            )

        def flux_keep(block):
            inside = onp.ones(block.wavel.shape, dtype=bool)
            if wavel_min is not None:
                inside &= block.wavel >= wavel_min
            if wavel_max is not None:
                inside &= block.wavel <= wavel_max
            return inside

        return self._subset(keep, flux_keep)

    def _subset(self, keep, flux_keep=None):
        """These data restricted to the samples where ``keep`` is True.

        ``flux_keep(block)`` gives the samples to keep of an OI_FLUX block,
        which are not visibility samples.
        """
        if (
            self.observable_kind != "split"
            or self.vis_mat is not None
            or self.phi_mat is not None
            or self.uv_grid is not None
        ):
            raise ValueError(
                "Only unprojected data can be split: projected (kernel, "
                "DISCO) observables mix samples."
            )
        if self.phase_offsets is not None:
            raise ValueError(
                "Data with closure-phase offsets cannot be split or "
                "selected; add the offsets (with_closure_offsets) afterwards."
            )
        keep = onp.asarray(keep, dtype=bool)
        new_index = onp.cumsum(keep) - 1  # old sample -> new sample

        def observed(index):
            """Rows of the observables to keep, and their new sample index."""
            if index is None:
                return onp.flatnonzero(keep), None
            index = onp.asarray(index)
            rows = onp.flatnonzero(keep[index])
            return rows, np.asarray(new_index[index[rows]])

        vis_rows, vis_index = observed(self.vis_index)
        if self.cp_flag:
            legs = [
                onp.asarray(i) for i in (self.i_cps1, self.i_cps2, self.i_cps3)
            ]
            if onp.any(keep[legs[0]] != keep[legs[1]]) or onp.any(
                keep[legs[0]] != keep[legs[2]]
            ):
                raise ValueError(
                    "A closure phase would be split between parts."
                )
            phi_rows = onp.flatnonzero(keep[legs[0]])
            legs = [new_index[leg[phi_rows]] for leg in legs]
            closure = ClosureNoise.from_indices(*legs)
            legs = [np.asarray(leg) for leg in legs]
            phi_index = None
        else:
            phi_rows, phi_index = observed(self.phi_index)
            legs, closure = [None] * 3, None
        wavel = self.wavel if self.wavel.size == 1 else self.wavel[keep]
        times = [
            None if x is None else x[keep]
            for x in (self.dt, self.frame, self.stations)
        ]
        gains = None
        if self.gains is not None:
            kept = onp.zeros(self.vis.size, bool)
            kept[vis_rows] = True
            gains = self.gains.subset(kept)
        extras = []
        for block in self.extras:
            own = None
            if isinstance(block, FluxSpectrum) and block.sample is None:
                if flux_keep is None:
                    raise ValueError("OI_FLUX data cannot be split this way.")
                own = onp.asarray(flux_keep(block), dtype=bool)
            extras.append(block.subset(keep, new_index, own))
        out = eqx.tree_at(
            lambda d: (
                d.u,
                d.v,
                d.wavel,
                d.vis,
                d.d_vis,
                d.phi,
                d.d_phi,
                d.i_cps1,
                d.i_cps2,
                d.i_cps3,
                d.vis_index,
                d.phi_index,
                d.cp_noise,
                d.dt,
                d.frame,
                d.stations,
                d.gains,
            ),
            self,
            (
                self.u[keep],
                self.v[keep],
                wavel,
                self.vis[vis_rows],
                self.d_vis[vis_rows],
                self.phi[phi_rows],
                self.d_phi[phi_rows],
                *legs,
                vis_index,
                phi_index,
                closure,
                *times,
                gains,
            ),
            is_leaf=lambda x: x is None,
        )
        return eqx.tree_at(lambda d: d.extras, out, tuple(extras))

    def _resolve_vis_mode(self, vis_mode):
        """Resolve the visibility channel convention used before linear projection."""
        mode = str(vis_mode).strip().lower()
        if mode == "auto":
            return "v2" if self.v2_flag else "amp"
        valid = {"v2", "amp", "logamp"}
        if mode not in valid:
            raise ValueError(
                f"Unsupported vis_mode '{vis_mode}'. Expected one of {sorted(valid)} or 'auto'."
            )
        return mode

    @staticmethod
    def _coerce_bool_flag(value, name):
        if isinstance(value, str):
            text = value.strip().lower()
            if text in {"true", "t", "1", "yes", "y", "on"}:
                return True
            if text in {"false", "f", "0", "no", "n", "off"}:
                return False
            raise ValueError(
                f"Unsupported {name!r} value {value!r}; expected a boolean."
            )
        return bool(value)

    @staticmethod
    def _phase_unit_scale(unit, default_unit):
        """Return multiplicative factor converting the provided phase unit to rad."""
        raw_unit = default_unit if unit is None else unit
        unit_name = str(raw_unit).strip().lower()
        if unit_name in {"rad", "radian", "radians"}:
            return 1.0
        if unit_name in {"deg", "degree", "degrees"}:
            return np.pi / 180.0
        raise ValueError(
            f"Unsupported phase unit '{raw_unit}'. Expected radians or degrees."
        )

    @classmethod
    def _phase_to_radians(cls, phi, d_phi, unit, default_unit):
        """Convert phase observables and uncertainties to radians."""
        scale = cls._phase_unit_scale(unit, default_unit)
        return np.asarray(phi, dtype=float) * scale, np.asarray(
            d_phi, dtype=float
        ) * np.abs(scale)

    @staticmethod
    def _validate_operator_shape(operator, input_size, label):
        """Check an operator has shape ``(n_out, input_size)``."""
        if operator is None:
            return
        if operator.ndim != 2:
            raise ValueError(
                f"{label} must be a 2D matrix; got shape {operator.shape}."
            )
        if operator.shape[1] != input_size:
            hint = (
                " It looks transposed; pass its transpose."
                if operator.shape[0] == input_size
                else ""
            )
            raise ValueError(
                f"{label} has shape {operator.shape}, but operators must have "
                f"shape (n_out, n_in) with n_in = {input_size} samples.{hint}"
            )

    @staticmethod
    def _apply_linear_operator(values, operator):
        """Apply an ``(n_out, n_in)`` operator to a length-``n_in`` vector."""
        if operator is None:
            return values
        return operator @ np.asarray(values, dtype=float).reshape(-1)

    @classmethod
    def _diagonalised(cls, operator, channel_sigma, correlation=None):
        """An operator whose outputs are independent, and their errors.

        Projected outputs are linear combinations of the input angles and
        are not wrapped (as for kernel phases and DISCOs), so a projection
        of closure phases is not invariant to shifting one input by 2π;
        prefer unprojected closure phases, whose likelihood is periodic
        (``cp_noise``).

        The outputs of ``operator`` have covariance A D^½ R D^½ Aᵀ, with D
        the diagonal of ``channel_sigma``² and R the inputs' correlation
        (the identity unless given, e.g. for correlated closure phases). If
        that is diagonal, the operator is
        kept and the errors are its square-rooted diagonal. Otherwise the
        outputs are correlated, and using only the diagonal would count
        shared information more than once: the operator is rotated onto
        the eigenvectors of A D^½ R D^½ Aᵀ, keeping those with non-zero
        variance, so that the new outputs are independent with errors √λ.
        """
        operator = np.asarray(operator, dtype=float)
        sigma = np.asarray(channel_sigma, dtype=float).reshape(-1)
        weighted = operator * sigma[None, :]
        if correlation is None:
            covariance = weighted @ weighted.T
        else:
            covariance = weighted @ np.asarray(correlation) @ weighted.T
        diagonal = np.diag(covariance)
        # A tolerance relative to the covariance's own scale, so that
        # rescaling an operator or its errors cannot bypass this check.
        scale = float(np.max(np.abs(diagonal)))
        off = covariance - np.diag(diagonal)
        if scale == 0.0 or float(np.max(np.abs(off))) <= 1e-7 * scale:
            # Already independent; drop outputs with no variance (e.g. a
            # closure relation of correlated closure phases), which carry
            # no measurement and would divide 0 by 0.
            keep = diagonal > 1e-10 * scale
            return operator[keep], np.sqrt(diagonal[keep])
        variance, vectors = np.linalg.eigh(covariance)
        keep = variance > 1e-10 * float(np.max(variance))
        return vectors[:, keep].T @ operator, np.sqrt(variance[keep])

    @classmethod
    def _validate_diagonal_covariance(cls, channel_sigma, operator, label):
        """Assert that an explicitly labelled DISCO operator whitens covariance."""
        if operator is None:
            return
        sigma = np.asarray(channel_sigma, dtype=float).reshape(-1)
        weighted = operator * sigma[None, :]
        covariance = weighted @ weighted.T
        diagonal = np.diag(np.diag(covariance))
        scale = float(np.max(np.abs(np.diag(covariance))))
        atol = max(1e-12, 1e-7 * scale)
        if not bool(np.allclose(covariance, diagonal, rtol=1e-5, atol=atol)):
            raise ValueError(
                f"{label} does not produce diagonal propagated covariance. "
                "DISCO observables must be statistically independent."
            )

    def _visibility_channel_from_model(self, cvis):
        """Convert complex visibilities to the configured scalar visibility channel."""
        amp = np.abs(cvis)
        if self.vis_mode == "v2":
            return amp**2
        if self.vis_mode == "logamp":
            return np.log(np.maximum(amp, 1e-30))
        return amp

    def _visibility_channel_from_data(self, vis, d_vis):
        """Convert stored visibility observables to the configured scalar channel.

        Log-amplitudes take the log of the observable floored at its own
        uncertainty (as in ``_visibility_uncertainty_channel``), so that a
        V² or amplitude at or below zero gives a finite value rather than
        log 1e-30.
        """
        vis = np.asarray(vis, dtype=float)
        d_vis = np.asarray(d_vis, dtype=float)
        if self.vis_mode == "logamp":
            floored = np.maximum(vis, np.maximum(d_vis, 1e-30))
            if self.v2_flag:
                return 0.5 * np.log(floored)
            return np.log(floored)
        if self.vis_mode == "amp" and self.v2_flag:
            return np.sqrt(np.maximum(vis, 0.0))
        if self.vis_mode == "v2" and (not self.v2_flag):
            return vis**2
        return vis

    def _visibility_uncertainty_channel(self, vis, d_vis):
        """Convert visibility uncertainties into the configured scalar channel.

        The errors are propagated linearly, with the derivative evaluated
        at the noisy data. Near zero that derivative diverges, so the data
        are floored at their own uncertainty first, in either direction:
        V² → amplitude gives ½σ/√max(V², σ) and V² → log-amplitude
        ½σ/max(V², σ) (at most ½√σ and ½), amplitude → log-amplitude
        σ/max(|V|, σ), and amplitude → V² uses √(|V|² + σ²) for |V|.
        """
        vis = np.asarray(vis, dtype=float)
        d_vis = np.asarray(d_vis, dtype=float)
        floored = np.maximum(vis, np.maximum(d_vis, 1e-30))
        if self.vis_mode == "logamp":
            if self.v2_flag:
                return 0.5 * d_vis / floored
            return d_vis / floored
        if self.vis_mode == "amp" and self.v2_flag:
            return 0.5 * d_vis / np.sqrt(floored)
        if self.vis_mode == "v2" and (not self.v2_flag):
            # |V| is floored at its own uncertainty, so noisy amplitudes near
            # or below zero do not get a vanishing V² error.
            return 2.0 * np.hypot(vis, d_vis) * d_vis
        return d_vis

    def _transform_observed_channels(
        self,
        validate_vis_covariance=False,
        validate_phi_covariance=False,
        closure=None,
    ):
        """Convert observed channels to ``vis_mode`` and apply operators.

        Data already in the projected basis (their size differs from the
        number of observed samples) are left unchanged.
        """
        n_vis = self._n_vis_samples()
        n_phi = self._n_phi_samples()
        self._validate_operator_shape(self.vis_mat, n_vis, "vis_mat")
        self._validate_operator_shape(self.phi_mat, n_phi, "phi_mat")

        if np.asarray(self.vis).size == n_vis:
            vis_channel = self._visibility_channel_from_data(
                self.vis, self.d_vis
            )
            vis_sigma = self._visibility_uncertainty_channel(
                self.vis, self.d_vis
            )
            if self.vis_mat is None:
                self.vis, self.d_vis = vis_channel, vis_sigma
            else:
                if validate_vis_covariance:
                    self._validate_diagonal_covariance(
                        vis_sigma, self.vis_mat, "disco_vis_mat"
                    )
                self.vis_mat, self.d_vis = self._diagonalised(
                    self.vis_mat, vis_sigma
                )
                self.vis = self._apply_linear_operator(
                    vis_channel, self.vis_mat
                )

        if self.phi_mat is not None and np.asarray(self.phi).size == n_phi:
            phi_sigma = np.asarray(self.d_phi, dtype=float)
            if validate_phi_covariance:
                self._validate_diagonal_covariance(
                    phi_sigma, self.phi_mat, "disco_phi_mat"
                )
            correlation = (
                None if closure is None else closure.correlation(n_phi)
            )
            self.phi_mat, self.d_phi = self._diagonalised(
                self.phi_mat, phi_sigma, correlation
            )
            self.phi = self._apply_linear_operator(self.phi, self.phi_mat)

    def _n_vis_samples(self):
        """Number of visibility samples before any projection."""
        if self.vis_index is not None:
            return int(np.asarray(self.vis_index).size)
        return int(np.asarray(self.u).size)

    def _n_phi_samples(self):
        """Number of phase samples (or closure phases) before projection."""
        if self.cp_flag:
            return len(self.i_cps1)
        if self.phi_index is not None:
            return int(np.asarray(self.phi_index).size)
        return int(np.asarray(self.u).size)

    @property
    def has_phases(self):
        """Whether the data hold any phase observables. Visibility-only
        data (e.g. V² without closure phases) have an empty phase block."""
        if self.observable_kind != "split":
            return True
        return self._n_phi_samples() > 0

    @property
    def n_independent(self):
        """Number of independent observables, for degrees of freedom.

        Equal to the size of :meth:`flatten_data`, except for closure phases
        from four or more telescopes, where only the independent
        combinations count (three of the four triangles of a frame and
        channel, for four telescopes). The residual vector of
        [`whitened_residuals`][virgil.likelihood.whitened_residuals] is
        longer for such data; see :attr:`n_residuals`.
        """
        n = int(np.asarray(self.vis).size) + int(np.asarray(self.phi).size)
        if self.cp_noise is not None:
            n += self.cp_noise.size - int(np.asarray(self.phi).size)
        return n + sum(block.n_independent for block in self.extras)

    @property
    def n_residuals(self):
        """Length of [`whitened_residuals`][virgil.likelihood.whitened_residuals].

        Equal to ``n_independent``, except for correlated closure phases
        (four or more telescopes), which add one periodic penalty residual
        per closure phase that keeps the likelihood continuous where a
        residual crosses ±π. Use ``n_independent`` for degrees of freedom.
        """
        n = self.n_independent
        if self.cp_noise is not None:
            n += int(np.asarray(self.phi).size)
        return n

    def flatten_data(self):
        """
        Return the data vector and its uncertainties.

        Returns
        -------
        tuple[array-like, array-like]
            The visibility observables followed by the phases (radians),
            in the order of
            [`model`][virgil.oidata.OIData.model], and matching one-sigma uncertainties.
        """
        if self.observable_kind == "mixed_log_complex":
            return self.vis, self.d_vis
        return (
            np.concatenate(
                [self.vis, self.phi] + [b.data() for b in self.extras]
            ),
            np.concatenate(
                [self.d_vis, self.d_phi]
                + [b.data_errors() for b in self.extras]
            ),
        )

    @property
    def has_model_covariance(self):
        """Whether the likelihood's covariance depends on the model: with
        gains, or with marginalised flux scales (``extras``). Least squares
        then does not give the likelihood, so ``fit`` uses L-BFGS."""
        return self.gains is not None or any(
            b.model_dependent_covariance for b in self.extras
        )

    @property
    def _phases_wrap(self):
        """Whether the phase block is raw angles that wrap at ±π."""
        return self.observable_kind == "split" and self.phi_mat is None

    def residuals(self, prediction, reference=None):
        """Return ``prediction - reference`` with phase residuals wrapped.

        Parameters
        ----------
        prediction : array-like
            Model vector, e.g. from [`model`][virgil.oidata.OIData.model].
        reference : array-like, optional
            Vector to compare against; by default the data
            (the first vector of :meth:`flatten_data`).

        Returns
        -------
        array-like
            Residual vector. Unprojected phase residuals are wrapped into
            ``[-π, π)``, so that a closure phase of ``π - ε`` against a model
            of ``-π + ε`` counts as a small residual rather than ``2π``.

        Notes
        -----
        This is for display. Likelihoods and fits use
        [`whitened_residuals`][virgil.likelihood.whitened_residuals],
        which is smooth where phases wrap.
        """
        if reference is None:
            reference = self.flatten_data()[0]
        resid = np.asarray(prediction) - np.asarray(reference)
        if not self._phases_wrap:
            return resid
        n_vis = np.asarray(self.vis).size
        n_phase = n_vis + np.asarray(self.phi).size
        phase = np.mod(resid[n_vis:n_phase] + np.pi, 2.0 * np.pi) - np.pi
        return np.concatenate([resid[:n_vis], phase, resid[n_phase:]])

    def standardize_model(self, cvis):
        """Map model complex visibilities (one per sample) to the data vector.

        The result lines up with the first vector of :meth:`flatten_data`.
        """
        if self.observable_kind == "mixed_log_complex":
            if self.vis_mat is None or self.phi_mat is None:
                raise ValueError(
                    "Mixed log-complex observables require model operators."
                )
            log_cvis = np.log(cvis)
            return self.vis_mat @ log_cvis.real + self.phi_mat @ log_cvis.imag
        return np.concatenate([self.to_vis(cvis), self.to_phases(cvis)])

    def to_vis(self, cvis):
        """
        Convert model complex visibilities to the visibility observables.

        The channel follows ``vis_mode`` (V², amplitude or log-amplitude);
        flagged samples are dropped, and ``vis_mat`` is applied if set.
        """
        vis = self._visibility_channel_from_model(cvis)
        if self.vis_index is not None:
            vis = vis[self.vis_index]
        return self._apply_linear_operator(vis, self.vis_mat)

    def to_phases(self, cvis):
        """
        Convert complex visibilities to closure or absolute phases in radians.
        """
        if self.cp_flag:
            phases = closure_phases(
                cvis, self.i_cps1, self.i_cps2, self.i_cps3
            )
        else:
            phases = np.angle(cvis)
            if self.phi_index is not None:
                phases = phases[self.phi_index]
        return self._apply_linear_operator(phases, self.phi_mat)

    def model(self, model_object):
        """
        Compute the model visibilities and phases for the given model object.

        A model that changes with time (see
        [`SourceModel.at`][virgil.models.SourceModel.at]) is evaluated at
        each sample's own time, which the data must have (``mjd``), with the
        direct Fourier transform: a ``uv_grid`` (AMIGO DISCO data, which
        carry no times) is not used for it.
        """
        cvis = self._cvis(model_object)
        prediction = self.standardize_model(cvis)
        if not self.extras:
            return prediction
        return np.concatenate(
            [prediction] + [b.predict(model_object, cvis) for b in self.extras]
        )

    def _cvis(self, model_object):
        """The model's complex visibility at every sample."""
        if getattr(model_object, "time_dependent", False):
            return self._cvis_in_time(model_object)
        if self.uv_grid is None:
            return model_object.model(self.u, self.v, self.wavel)
        return model_object.model_on_grid(
            self.u, self.v, self.wavel, self.uv_grid
        )

    def _cvis_in_time(self, model_object):
        """Complex visibilities with every sample at its own time."""
        if self.dt is None:
            raise ValueError(
                "The model changes with time but these data have no times: "
                "read them from OIFITS, or give mjd per sample."
            )
        wavel = np.broadcast_to(self.wavel, np.shape(self.u))

        def one(dt, u, v, w):
            scene = model_object.at(dt, self.t_ref)
            return scene.model(u[None], v[None], w[None])[0]

        # A compiled loop over samples, not vmap: with JAX 0.11, vmapping
        # components whose angles vary per sample (an Attached disc's bound
        # inclination or position angle) and differentiating made JAX run
        # executables with the wrong batch size (RuntimeProgramInputMismatch,
        # or "Expected cotangent type" in fit).
        return jax.lax.map(lambda x: one(*x), (self.dt, self.u, self.v, wavel))

    def with_error_scale(self, factor):
        """A copy of the data with every uncertainty multiplied by ``factor``.

        Use it when the error bars are known to be too large or too small
        overall, for example with a factor from
        [`error_scale`][virgil.imaging.error_scale]. The uncertainties
        are those of the observables as fitted (after any projection), so
        the whitened residuals simply scale by ``1 / factor``.

        Parameters
        ----------
        factor : float
            Positive scale for ``d_vis`` and ``d_phi``.
        """
        factor = float(factor)
        if not (onp.isfinite(factor) and factor > 0.0):
            raise ValueError(
                f"factor must be finite and positive, not {factor}."
            )
        out = eqx.tree_at(
            lambda d: (d.d_vis, d.d_phi),
            self,
            (self.d_vis * factor, self.d_phi * factor),
        )
        extras = tuple(b.with_errors(b.errors * factor) for b in self.extras)
        return eqx.tree_at(lambda d: d.extras, out, extras)

    def with_error_floor(self, absolute=None, relative=None):
        """A copy of the data with a minimum uncertainty per observable.

        Each uncertainty becomes ``max(σ, absolute, relative × |data|)``,
        as PMOIRED's ``min error`` and ``min relative error`` do. This is a
        fixed change to the data; the fitted counterpart, added in
        quadrature and relative to the model, is the ``noise=`` terms
        (see [`inflated_errors`][virgil.likelihood.inflated_errors]).
        Both use one rule, ``_utils.inflate_errors``.

        Parameters
        ----------
        absolute, relative : dict, optional
            Floors per observable: ``"vis"`` (in the units of ``vis``,
            e.g. V²), ``"phi"`` (radians), and any of ``extras`` by kind:
            ``"flux"``, ``"nflux"``, ``"corrflux"``, ``"visamp"``,
            ``"t3amp"`` and ``"visphi"`` (radians). Relative floors are
            not defined for phases.

        Returns
        -------
        OIData

        Notes
        -----
        Closure phases from four or more telescopes are floored before they
        are whitened: the whitened rows use the floored errors, while their
        periodic penalty rows keep their effective error 1/√(2π) (the
        penalty itself is 2 sin²(Δ/2)/σ with the floored σ). The
        differential phases are floored per channel, before their
        covariance N D Nᵀ is formed.

        Examples
        --------
        >>> import numpy as np
        >>> data = OIData({"u": [1.0, 2.0], "v": [0.0, 1.0], "wavel": 1e-6,
        ...                "vis": [0.9, 0.5], "d_vis": [0.001, 0.02]})
        >>> floored = data.with_error_floor(relative={"vis": 0.01})
        >>> [round(float(e), 4) for e in floored.d_vis]
        [0.009, 0.02]
        """
        absolute = dict(absolute or {})
        relative = dict(relative or {})
        names = {"vis", "phi"} | {b.kind for b in self.extras}
        unknown = sorted((set(absolute) | set(relative)) - names)
        if unknown:
            raise ValueError(
                f"No observables {unknown} in these data; they have "
                f"{sorted(names)}."
            )
        phases = {"phi", "visphi"} & set(relative)
        if phases:
            raise ValueError(
                f"Relative floors are not defined for phases {sorted(phases)}."
            )
        if self.vis_mat is not None and {"vis"} & (
            set(absolute) | set(relative)
        ):
            raise ValueError(
                "Floors apply to observed visibilities, not to projected "
                "(vis_mat) ones."
            )
        if self.phi_mat is not None and "phi" in absolute:
            raise ValueError(
                "Floors apply to observed phases, not to projected "
                "(phi_mat) ones."
            )

        def floor(name, sigma, values):
            return inflate_errors(
                sigma,
                values,
                absolute=absolute.get(name),
                relative=relative.get(name),
                combine="max",
            )

        out = eqx.tree_at(
            lambda d: (d.d_vis, d.d_phi),
            self,
            (
                floor("vis", self.d_vis, self.vis),
                floor("phi", self.d_phi, None),
            ),
        )
        extras = tuple(
            b.with_errors(floor(b.kind, b.errors, b.values))
            for b in self.extras
        )
        return eqx.tree_at(lambda d: d.extras, out, extras)

    def _replace_extra(self, kinds, change):
        """These data with ``change(block)`` for the extras of ``kinds``."""
        found = [b for b in self.extras if b.kind in kinds]
        if not found:
            raise ValueError(
                f"These data have no {' or '.join(kinds)} observables; read "
                "them with extras=."
            )
        extras = tuple(
            change(b) if b.kind in kinds else b for b in self.extras
        )
        return eqx.tree_at(lambda d: d.extras, self, extras)

    def with_continuum(
        self, continuum=None, lines=None, order=None, prior_width=None
    ):
        """A copy with the continuum and line windows of the extra spectra.

        They set how differential phases (``"visphi"``) and normalised
        spectra (``"nflux"``) are normalised: see
        [`DifferentialPhase`][virgil.observables.DifferentialPhase].

        Parameters
        ----------
        continuum, lines : sequence of (lo, hi), optional
            Wavelength ranges (metres). Differential phases are fitted over
            the continuum and kept in the lines (each defaults to the
            complement of the other); normalised spectra use the continuum.
        order : int, optional
            The continuum polynomial in wavenumber: 1 (a mean and a slope)
            by default for differential phases, 0 (a mean) for spectra.
        prior_width : float or (float, float), optional
            For differential phases: marginalise each baseline and frame's
            offset (and slope) under Gaussian priors of these widths
            (radians), using the channels of both windows, instead of
            projecting them out as the pipeline does (the default). See
            [`DifferentialPhase`][virgil.observables.DifferentialPhase].

        Examples
        --------
        Keep the differential phase across Brγ, normalised on either side:
        ``data.with_continuum([(2.150e-6, 2.162e-6), (2.170e-6, 2.180e-6)],
        lines=[(2.163e-6, 2.169e-6)])``.
        """
        kinds = {b.kind for b in self.extras} & {"visphi", "nflux"}
        if not kinds:
            raise ValueError(
                "These data have no differential phases or normalised "
                "spectra; read them with extras=('visphi',) or ('nflux',)."
            )
        out = self
        if "visphi" in kinds:
            out = out._replace_extra(
                ("visphi",),
                lambda b: b.rebuild(
                    continuum=continuum,
                    lines=lines,
                    order=b.order if order is None else order,
                    prior_width=prior_width,
                ),
            )
        if "nflux" in kinds:
            out = out._replace_extra(
                ("nflux",),
                lambda b: b.rebuild(
                    continuum=continuum,
                    continuum_order=(
                        b.continuum_order if order is None else order
                    ),
                ),
            )
        return out

    def with_flux_scale(
        self,
        per="dataset",
        width=None,
        poly_order=0,
        poly_width=0.1,
        kinds=("flux", "nflux", "corrflux"),
    ):
        """A copy with new grey-scale nuisances for the extra spectra.

        Each spectrum (OI_FLUX, or correlated fluxes) is known up to a
        scale, marginalised analytically under a broad Gaussian prior (see
        [`FluxSpectrum`][virgil.observables.FluxSpectrum]).

        Parameters
        ----------
        per : {"dataset", "row", "frame", "station"}, optional
            One scale for the whole dataset (default), or one per spectrum
            (row), per exposure, or per telescope (baseline, for
            correlated fluxes). Fibre injection varies per telescope and
            exposure, so per row is the safest for uncalibrated spectra.
        width : float, optional
            The scale's relative prior width (default 1.0; 0.1 for
            ``"nflux"``).
        poly_order : int, optional
            Also marginalise a polynomial in λ of this order times the
            model spectrum (a chromatic calibration).
        poly_width : float, optional
            Relative prior width of each polynomial coefficient.
        kinds : sequence of str, optional
            Which blocks to change.
        """
        return self._replace_extra(
            tuple(kinds),
            lambda b: b.rebuild(
                per=per,
                width=width,
                poly_order=poly_order,
                poly_width=poly_width,
            ),
        )

    def with_wavelength_scale(self, scale=1.0, offset=0.0):
        """A copy of the data whose wavelengths are ``scale · λ + offset``.

        A wavelength calibration error: models are evaluated at the
        corrected wavelengths, which rescales the spatial frequencies
        (``u`` and ``v`` are in metres) and moves the spectra together, as
        a wrong wavelength scale does. Angular sizes scale with it, so with
        a single dataset the scale is degenerate with every size, and its
        prior *is* the systematic error. Fit it with the noise terms
        ``wavel_scale`` and ``wavel_offset`` (e.g.
        ``noise={"wavel_scale": dist.Normal(1.0, 2e-4)}``, about right for
        GRAVITY), which call this.

        Parameters
        ----------
        scale : float, optional
            Factor on the wavelengths (default 1).
        offset : float, optional
            Shift added after scaling, in metres (default 0).

        Returns
        -------
        OIData
            The data with ``wavel`` replaced. A ``uv_grid`` is kept: it is
            in metres and divided by the wavelength where it is used, so
            it still matches the samples.
        """
        return eqx.tree_at(
            lambda d: d.wavel, self, self.wavel * scale + offset
        )

    def with_closure_offsets(self, baseline=None, triangle=None, modes=None):
        """A copy of the data with closure-phase offsets per frame.

        Calibration can leave closure phases that do not close. The
        likelihood then marginalises offsets common to the channels of a
        frame analytically (see
        [`ClosureOffsets`][virgil.gains.ClosureOffsets]), with these widths
        unless they are fitted as the noise terms ``phi_offset_baseline``,
        ``phi_offset_triangle`` or ``phi_offset_modes``. Use them only if
        calibrators show such offsets: they are off by default. Needs
        closure phases from four or more telescopes.

        This is a small-offset approximation: an offset δ changes the
        whitened sine sin Δ by about δ cos Δ, which is treated as linear in
        δ. It holds while the offsets (and the residuals) are small, about
        δ ≲ 0.3 rad (17°); larger widths are not marginalised exactly.

        Parameters
        ----------
        baseline : float, optional
            Width (radians) of a phase offset per (frame, baseline), which
            reaches the closure phases through the triangles' signs.
        triangle : float, optional
            Width (radians) of an offset per (frame, triangle).
        modes : array-like, optional
            Further modes, ``(n_mode, n_phase)``, radians per 1σ, one value
            per closure phase, each within one frame (e.g. a calibrator
            PCA's). Their width, 1 by default, scales them.

        Returns
        -------
        OIData
            The data with ``phase_offsets`` set. Split or select the data
            first: data with offsets cannot be split.
        """
        offsets = closure_offsets(self, baseline, triangle, modes)
        return eqx.tree_at(
            lambda d: d.phase_offsets,
            self,
            offsets,
            is_leaf=lambda x: x is None,
        )

    def with_gains(
        self, telescope=None, baseline=None, chromatic=None, modes=None
    ):
        """A copy of the data with calibration gains correlated across channels.

        The likelihood then marginalises gains on log |V| per frame
        analytically (see [`virgil.gains`][virgil.gains]), with these widths
        unless they are fitted as the noise terms ``vis_gain_telescope``,
        ``vis_gain_baseline``, ``vis_gain_chromatic`` or ``vis_gain_modes``.
        The covariance then depends on the model, so fits use L-BFGS.

        Parameters
        ----------
        telescope, baseline, chromatic : float, optional
            Widths (1σ on log |V|; 0.01 is a 1% amplitude or 2% V² gain)
            of gains per (frame, telescope), per (frame, baseline), and per
            (frame, baseline) shaped (λ_ref/λ)², a coherence loss. ``None``
            leaves a group out.
        modes : array-like, optional
            Further modes, ``(n_mode, n_sample)``: shapes on log |V| per 1σ,
            one value per sample (the order of ``u``), e.g. a calibrator
            PCA's. Their width, 1 by default, scales them.

        Returns
        -------
        OIData
            The data with ``gains`` set (replacing any earlier ones).
        """
        gains = gain_modes(self, telescope, baseline, chromatic, modes)
        return eqx.tree_at(
            lambda d: d.gains, self, gains, is_leaf=lambda x: x is None
        )

    def with_model(self, model_object, key=None, noise_scale=1.0):
        """Return a copy populated from a model with optional Gaussian noise.

        Sampling, uncertainties, conventions, closure indices, and linear
        observable operators are preserved from this object. With ``key``
        and ``gains``, gains are drawn at their default widths too, and
        applied exactly (|V| times e^g) before the noise is added.
        """
        noise_scale = float(noise_scale)
        if noise_scale < 0.0:
            raise ValueError("noise_scale must be non-negative.")

        prediction = self.model(model_object)
        n_vis = self.vis.size
        n_phase = n_vis + self.phi.size
        vis = prediction[:n_vis]
        phi = prediction[n_vis:n_phase]
        if key is not None:
            vis_key, phi_key = jax.random.split(key)
            if self.gains is not None:
                gain_key = jax.random.fold_in(key, 2)
                g = self.gains.sample(gain_key, self.gains.widths)
                vis = {
                    "v2": vis * np.exp(2.0 * g),
                    "amp": vis * np.exp(g),
                }.get(self.vis_mode, vis + g)
            vis = vis + noise_scale * self.d_vis * jax.random.normal(
                vis_key, vis.shape
            )
            if self.cp_noise is None:
                phi_noise = self.d_phi * jax.random.normal(phi_key, phi.shape)
            else:
                phi_noise = self.cp_noise.sample(phi_key, self.d_phi, phi.size)
            phi = phi + noise_scale * phi_noise
            if self.phase_offsets is not None:
                offsets = self.phase_offsets
                phi = phi + offsets.sample(
                    jax.random.fold_in(key, 3),
                    self.cp_noise,
                    phi.size,
                    offsets.widths,
                )
        out = self.set(["vis", "phi"], [vis, phi])
        if not self.extras:
            return out
        cvis = self._cvis(model_object)
        extras, offset = [], n_phase
        for i, block in enumerate(self.extras):
            n = int(block.data().size)
            noise = None
            if key is not None:
                noise = noise_scale * jax.random.normal(
                    jax.random.fold_in(key, 10 + i), block.values.shape
                )
            extras.append(
                block.simulated(prediction[offset : offset + n], cvis, noise)
            )
            offset += n
        return eqx.tree_at(lambda d: d.extras, out, tuple(extras))


# Keys of a record (or dictionary) holding extra observables.
_RECORD_KEYS = ("flux", "nflux", "visamp", "t3amp", "visphi")


def _good(record):
    """Unflagged, finite entries of an extra observable's record."""
    value = onp.asarray(record["value"], float).reshape(-1)
    error = onp.asarray(record["error"], float).reshape(-1)
    good = onp.isfinite(value) & onp.isfinite(error) & (error > 0)
    if record.get("flag") is not None:
        good &= ~onp.asarray(record["flag"], bool).reshape(-1)
    return good, value, error


def _build_extras(data, obj, legs):
    """The blocks of extra observables of a record, in ``KINDS`` order."""
    n = onp.size(obj.u)
    wavel = onp.broadcast_to(onp.asarray(obj.wavel), (n,))
    frame = None if obj.frame is None else onp.asarray(obj.frame)
    stations = None if obj.stations is None else onp.asarray(obj.stations)
    blocks = []
    for kind in ("flux", "nflux"):
        record = data.get(kind)
        if record is None:
            continue
        good, value, error = _good(record)
        size = value.size
        mjd = record.get("mjd")
        mjd = None if mjd is None else onp.asarray(mjd, onp.float64)
        times = (
            onp.zeros(size)
            if mjd is None
            else onp.unique(onp.round(mjd / 1e-4), return_inverse=True)[
                1
            ].reshape(-1)
        )
        row = onp.asarray(record.get("row", onp.zeros(size, int)))
        station = onp.asarray(record.get("station", onp.full(size, -1)))
        blocks.append(
            FluxSpectrum.build(
                kind,
                value[good],
                error[good],
                onp.asarray(record["wavel"], float).reshape(-1)[good],
                row.reshape(-1)[good],
                times[good],
                station.reshape(-1)[good],
                mjd=None if mjd is None else mjd[good],
            )
        )
    record = data.get("visamp")
    if record is not None:
        good, value, error = _good(record)
        sample = onp.asarray(record["sample"], int).reshape(-1)[good]
        if record.get("amptyp", "absolute") == "correlated flux":
            f = onp.zeros(sample.size) if frame is None else frame[sample]
            pair = (
                onp.zeros((sample.size, 2), int)
                if stations is None
                else stations[sample]
            )
            blocks.append(
                FluxSpectrum.build(
                    "corrflux",
                    value[good],
                    error[good],
                    wavel[sample],
                    _row_labels(f, pair),
                    f,
                    pair,
                    sample=sample,
                )
            )
        else:
            blocks.append(
                VisibilityAmplitude(
                    np.asarray(value[good]),
                    np.asarray(error[good]),
                    sample.astype(onp.int32),
                )
            )
    record = data.get("t3amp")
    if record is not None:
        if legs is None:
            raise ValueError(
                "Triple amplitudes need the closure triangles (i_cps1, "
                "i_cps2, i_cps3)."
            )
        good, value, error = _good(record)
        if value.size != len(legs[0]):
            raise ValueError(
                f"t3amp has {value.size} values for {len(legs[0])} closure "
                "phases; give one per closure phase."
            )
        blocks.append(
            TripleAmplitude(
                np.asarray(value[good]),
                np.asarray(error[good]),
                *(onp.asarray(leg)[good].astype(onp.int32) for leg in legs),
            )
        )
    record = data.get("visphi")
    if record is not None:
        if frame is None:
            raise ValueError(
                "Differential phases are normalised per frame: give 'frame' "
                "(or 'mjd') per sample."
            )
        closure_free = bool(obj.cp_flag)
        if closure_free and stations is None:
            raise ValueError(
                "Differential phases beside closure phases need each "
                "sample's 'stations', to remove their closure part."
            )
        good, value, error = _good(record)
        sample = onp.asarray(record["sample"], int).reshape(-1)[good]
        row = record.get("row")
        blocks.append(
            DifferentialPhase.build(
                value[good],
                error[good],
                sample,
                wavel[sample],
                frame[sample],
                (
                    onp.zeros((sample.size, 2), int)
                    if stations is None
                    else stations[sample]
                ),
                row=None
                if row is None
                else onp.asarray(row).reshape(-1)[good],
                closure_free=closure_free,
            )
        )
    return tuple(sorted(blocks, key=lambda b: KINDS.index(b.kind)))


def _row_labels(frame, pairs):
    """One label per (frame, station pair)."""
    keys = onp.column_stack([onp.asarray(frame), onp.asarray(pairs)])
    return onp.unique(keys, axis=0, return_inverse=True)[1].reshape(-1)


def closure_phases(cvis, index_cps1, index_cps2, index_cps3):
    """
    Calculate closure phases from complex visibilities.

    Parameters
    ----------
    cvis : array-like
        Complex visibilities, one per sample.
    index_cps1 : array-like
        For each closure phase, the sample of baseline ``(a, b)``.
    index_cps2 : array-like
        For each closure phase, the sample of baseline ``(b, c)``.
    index_cps3 : array-like
        For each closure phase, the sample of baseline ``(a, c)``.

    Returns
    -------
    array-like
        Closure phases ``φ[i1] + φ[i2] − φ[i3]`` in radians, wrapped into
        ``[-π, π)``.

    Notes
    -----
    This helper returns radians for internal modeling consistency. Convert to
    degrees before writing OIFITS phase columns (e.g., ``T3PHI``).
    """
    phases = np.angle(np.asarray(cvis))
    cp = (
        phases[np.asarray(index_cps1)]
        + phases[np.asarray(index_cps2)]
        - phases[np.asarray(index_cps3)]
    )
    return np.mod(cp + np.pi, 2.0 * np.pi) - np.pi


def cp_indices(vis_sta_index, cp_sta_index):
    """Map closure-triangle station indices to baseline indices.

    Parameters
    ----------
    vis_sta_index : array-like
        Station index pairs ``(a, b)``, one per baseline.
    cp_sta_index : array-like
        Station index triplets ``(a, b, c)``, one per closure triangle.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Arrays ``(i_cps1, i_cps2, i_cps3)`` giving, for each triangle, the
        baselines ``(a, b)``, ``(b, c)`` and ``(a, c)``, so that the closure
        phase is ``φ[i_cps1] + φ[i_cps2] − φ[i_cps3]``.

    Raises
    ------
    ValueError
        If a triangle needs a baseline that is missing, or stored only in the
        reversed orientation.

    Notes
    -----
    Baselines are matched on station indices alone. For data with several
    epochs or wavelength channels use [`virgil.oifits.read_oifits`][virgil.oifits.read_oifits],
    which also matches on instrument and MJD.
    """
    vis_sta_index = onp.asarray(vis_sta_index, dtype=int).reshape(-1, 2)
    cp_sta_index = onp.asarray(cp_sta_index, dtype=int).reshape(-1, 3)
    lookup = {}
    for k, (a, b) in enumerate(vis_sta_index):
        lookup.setdefault((int(a), int(b)), k)

    def baseline(pair):
        pair = (int(pair[0]), int(pair[1]))
        if pair in lookup:
            return lookup[pair]
        # TODO: support reversed baselines by returning a sign per leg.
        detail = (
            f"it is only stored reversed as {pair[::-1]}"
            if pair[::-1] in lookup
            else "it is missing"
        )
        raise ValueError(
            f"A closure triangle needs baseline {pair}, but {detail}."
        )

    legs = [[], [], []]
    for a, b, c in cp_sta_index:
        for leg, pair in zip(legs, ((a, b), (b, c), (a, c))):
            leg.append(baseline(pair))
    return tuple(onp.asarray(leg, dtype=int) for leg in legs)


def _expand_channels(u, v, wavel, vis, indices):
    """Expand per-baseline arrays to one sample per (baseline, channel).

    Samples are ordered baseline-major; closure indices (per baseline) are
    mapped to the samples at the same channel.
    """
    n_baseline, n_wavel = vis.shape
    if u.shape != (n_baseline,) or v.shape != (n_baseline,):
        raise ValueError(
            f"vis has shape {vis.shape}, so u and v need {n_baseline} "
            "entries (one per baseline)."
        )
    if wavel.size != n_wavel:
        raise ValueError(
            f"vis has {n_wavel} wavelength channels but wavel has "
            f"{wavel.size} values."
        )
    channels = onp.arange(n_wavel)
    if indices is not None:
        indices = [
            (index[:, None] * n_wavel + channels[None, :]).reshape(-1)
            for index in indices
        ]
    return (
        onp.repeat(u, n_wavel),
        onp.repeat(v, n_wavel),
        onp.tile(wavel, n_baseline),
        indices,
    )


def _good_samples(values, errors, flag, n_samples, name):
    """Mask of unflagged, finite samples, or ``None`` if all are good.

    Data whose size is not ``n_samples`` are already projected and are not
    checked.
    """
    if values.size != n_samples:
        if flag is not None:
            raise ValueError(
                f"{name}_flag was given, but {name} has {values.size} values "
                f"for {n_samples} samples (it looks already projected)."
            )
        return None
    bad = ~(onp.isfinite(values) & onp.isfinite(errors))
    if flag is not None:
        flag = onp.asarray(flag, dtype=bool).reshape(-1)
        if flag.size != n_samples:
            raise ValueError(
                f"{name}_flag has {flag.size} entries for {n_samples} samples."
            )
        bad = bad | flag
    if not bad.any():
        return None
    return ~bad
