"""Calibration gains correlated across channels, marginalized analytically.

The calibration of spectro-interferometric data (the transfer function, its
drift between calibrators, injection and piston losses) multiplies every
channel of a frame's baseline by much the same factor. Independent error
bars cannot describe this: an error common to all channels averages down
in them, but not in the data. Here each such error is a gain on log |V|,

    log |V|_obs = log |V| + Σ_j τ_j z_j m_j,    z_j ~ N(0, 1),

where m_j is a mode (a shape over the data's visibility samples) and τ_j
its width. The modes come in groups, with one width each:

* ``telescope``: one gain per (frame, telescope), on the frame's baselines
  that include that telescope;
* ``baseline``: one gain per (frame, baseline), on all its channels;
* ``chromatic``: one per (frame, baseline) shaped (λ_ref/λ)², the first
  order of a coherence loss exp(−a/λ²) from piston jitter;
* ``modes``: shapes supplied from outside, per 1σ, e.g. from a calibrator
  PCA (as virgil-vlti makes), whose width scales them (1 = as given).

For small gains (τ ≲ 0.2) the observable changes linearly, by
J m_j with J = dObs/dlog|V| = 2V² for squared visibilities, |V| for
amplitudes and 1 for log-amplitudes, taken from the *model* (so the
covariance does not depend on the noisy data; Lachaume 2021). The gains
are then Gaussian and are marginalized analytically: the visibility
covariance becomes

    C = D + U Uᵀ,   U = J τ m  (one column per mode),

with D the diagonal of squared errors. Modes that share no sample are
independent, so C is block diagonal, one block per connected group of
modes (one per frame, for the built-in groups).

**Whitening.** Each block is whitened by successive rank-one steps, the
shared machinery for linear marginalization in ``virgil._linear``
(``whiten_blocks``). Only square roots of scalars appear, so the gradients
stay smooth, also where modes are degenerate and where widths go to zero.
"""

import equinox as eqx
import jax
import jax.numpy as np
import jax.scipy.linalg as jsl
import numpy as onp

from ._linear import whiten_blocks

__all__ = [
    "GainModes",
    "GAIN_GROUPS",
    "gain_modes",
    "ClosureOffsets",
    "OFFSET_GROUPS",
    "closure_offsets",
]

# Groups of gain modes, in order; their widths are fitted with the noise
# terms "vis_gain_<group>".
GAIN_GROUPS = ("telescope", "baseline", "chromatic", "modes")


class GainModes(eqx.Module):
    """Low-rank gain modes on the visibility observables.

    Built by [`gain_modes`][virgil.gains.gain_modes] (or
    [`OIData.with_gains`][virgil.oidata.OIData.with_gains]); see the
    module notes for the model.

    Modes within one frame are kept in blocks, the connected groups of such
    modes (one per frame for the built-in groups), padded to a common size.
    Modes that span frames (only supplied ones can) are kept apart, as
    dense columns, and whitened after the blocks. That way, a few modes
    across frames do not merge every frame into one dense block.

    Attributes
    ----------
    rows : jax.Array
        ``(n_block, n_row)`` int32: the visibility observables of each block
        (rows of ``OIData.vis``), padded with ``n_vis`` (out of range). If
        every mode spans frames, one empty block of padding.
    shapes : jax.Array
        ``(n_block, n_row, n_mode)``: each mode's shape on log |V| for unit
        width, zero on padding.
    group : jax.Array
        ``(n_block, n_mode)`` int32: each mode's group, an index into
        ``groups``.
    spanning : jax.Array or None
        ``(n_vis, n_spanning)``: the shapes of modes that span frames, or
        None if there are none. Never a zero-size array: XLA's Shardy pass
        segfaults compiling a ``jax.pmap`` (numpyro's parallel chains) that
        captures one (JAX 0.11.2).
    spanning_group : jax.Array or None
        ``(n_spanning,)`` int32: their groups (None with ``spanning``).
    widths : jax.Array
        The default width of each group.
    groups : tuple of str
        The groups present, a subset of ``GAIN_GROUPS``.
    n_vis : int
        The number of visibility observables.
    """

    rows: jax.Array
    shapes: jax.Array
    group: jax.Array
    spanning: jax.Array | None
    spanning_group: jax.Array | None
    widths: jax.Array
    groups: tuple = eqx.field(static=True)
    n_vis: int = eqx.field(static=True)

    def widths_for(self, terms=None):
        """Each group's width, with ``vis_gain_<group>`` terms replacing them."""
        return _widths_for(
            self.groups, self.widths, terms, "vis_gain_", "OIData.with_gains"
        )

    def _columns(self, jacobian, widths):
        """The modes as columns of D^{-½} U: per block, and spanning frames."""
        jacobian = np.asarray(jacobian)
        scale = jacobian.at[self.rows].get(mode="fill", fill_value=0)
        local = self.shapes * scale[..., None] * widths[self.group][:, None, :]
        if self.spanning is None:
            return local, None
        spanning = (
            self.spanning * jacobian[:, None] * widths[self.spanning_group]
        )
        return local, spanning

    def whiten(self, x, jacobian, widths):
        """Whiten residuals for the covariance ``I + Σ w_j w_jᵀ``.

        Parameters
        ----------
        x : array-like
            Visibility residuals divided by their errors, ``(n_vis,)``.
        jacobian : array-like
            ``dObs/dlog|V|`` of the model divided by the errors, ``(n_vis,)``.
        widths : array-like
            Each group's width (see ``widths_for``).

        Returns
        -------
        tuple
            The whitened residuals ``(n_vis,)``, and per observable the log
            of the factor its effective error grows by: they sum to half
            the log-determinant of the covariance, ``½ Σ log(1 + w_jᵀw_j)``.
        """
        local, spanning = self._columns(jacobian, widths)
        return whiten_blocks(x, self.rows, local, spanning)

    def covariance(self, errors, jacobian, widths):
        """The dense visibility covariance ``D + U Uᵀ`` (for checks; O(n²)).

        ``jacobian`` is dObs/dlog|V| itself here, not divided by the errors.
        """
        errors = np.asarray(errors)
        local, spanning = self._columns(np.asarray(jacobian), widths)
        u = np.zeros((self.n_vis,) + local.shape[::2])
        b = np.arange(local.shape[0])[:, None]
        u = u.at[self.rows, b].set(local, mode="drop")
        u = u.reshape(self.n_vis, -1)
        if spanning is not None:
            u = np.concatenate([u, spanning], axis=1)
        return np.diag(errors**2) + u @ u.T

    def sample(self, key, widths):
        """Draw the gains: log |V| offset of each visibility observable."""
        widths = np.asarray(widths)
        local_key, span_key = jax.random.split(key)
        z = jax.random.normal(local_key, self.group.shape)
        per_row = np.einsum("brk,bk->br", self.shapes, z * widths[self.group])
        gains = np.zeros(self.n_vis).at[self.rows].add(per_row, mode="drop")
        if self.spanning is None:
            return gains
        z = jax.random.normal(span_key, self.spanning_group.shape)
        return gains + self.spanning @ (z * widths[self.spanning_group])

    def subset(self, keep):
        """The modes on the visibility observables where ``keep`` is True.

        A Gaussian's marginal on a subset of its variables is the same
        covariance restricted to them, so dropping rows is exact.
        """
        keep = onp.asarray(keep, bool)
        new_index = onp.where(keep, onp.cumsum(keep) - 1, -1)
        n_new = int(keep.sum())
        rows = onp.asarray(self.rows)
        inside = rows < self.n_vis
        mapped = onp.where(
            inside, new_index[onp.minimum(rows, self.n_vis - 1)], -1
        )
        mapped = onp.where(mapped < 0, n_new, mapped)
        shapes = onp.where((mapped < n_new)[..., None], self.shapes, 0.0)
        return GainModes(
            np.asarray(mapped, np.int32),
            np.asarray(shapes),
            self.group,
            None if self.spanning is None else self.spanning[keep],
            self.spanning_group,
            self.widths,
            self.groups,
            n_new,
        )

    def labels_shared(self, labels):
        """Whether any mode touches observables with different ``labels``.

        ``labels`` (one per visibility observable) partitions the data, e.g.
        into epochs; a partition's likelihoods add up to the whole only if
        no gain is shared between its parts.
        """
        labels = onp.asarray(labels)
        rows = onp.asarray(self.rows)
        for r, shapes in zip(rows, onp.asarray(self.shapes)):
            for col in shapes.T:
                touched = r[(r < self.n_vis) & (col != 0)]
                if onp.unique(labels[touched]).size > 1:
                    return True
        spanning = () if self.spanning is None else self.spanning
        for col in onp.asarray(spanning).T:
            if onp.unique(labels[col != 0]).size > 1:
                return True
        return False


def _frames_and_stations(data, need_stations):
    """Frame and station pair of each visibility observable."""
    if data.frame is None:
        raise ValueError(
            "Gains per frame need each sample's frame: read the data from "
            "OIFITS, or give 'frame' (or 'mjd') in the dictionary."
        )
    index = (
        onp.arange(onp.size(data.u))
        if data.vis_index is None
        else onp.asarray(data.vis_index)
    )
    frame = onp.asarray(data.frame)[index]
    stations = None
    if need_stations:
        if data.stations is None:
            raise ValueError(
                "Telescope and baseline gains need each sample's stations: "
                "read the data from OIFITS, or give 'stations' (station "
                "pairs) in the dictionary."
            )
        stations = onp.sort(onp.asarray(data.stations)[index], axis=1)
    return index, frame, stations


def gain_modes(
    data, telescope=None, baseline=None, chromatic=None, modes=None
):
    """Gain modes for a dataset, with default widths.

    Parameters
    ----------
    data : OIData
        Unprojected data, with frames (``frame``) and, for telescope,
        baseline and chromatic gains, station pairs (``stations``), as read
        from OIFITS.
    telescope, baseline, chromatic : float, optional
        Widths (1σ, on log |V|, so 0.01 is a 1% amplitude or 2% V² gain) of
        gains per (frame, telescope), per (frame, baseline), and per
        (frame, baseline) shaped (λ_ref/λ)² with λ_ref the median
        wavelength. ``None`` leaves that group out. A group can be given a
        nominal width here and fitted with the noise term
        ``vis_gain_<group>``.
    modes : array-like, optional
        Further modes, ``(n_mode, n_sample)``: shapes on log |V| per 1σ, one
        value per sample of the data (the order of ``data.u``; flagged
        samples are ignored). Their width (default 1) scales them all.

    Returns
    -------
    GainModes
    """
    if data.observable_kind != "split" or data.vis_mat is not None:
        raise ValueError(
            "Gains act on observed visibilities, not on projected (kernel, "
            "DISCO) observables."
        )
    built_in = {
        "telescope": telescope,
        "baseline": baseline,
        "chromatic": chromatic,
    }
    for name, width in built_in.items():
        if width is not None and not (onp.isfinite(width) and width >= 0):
            raise ValueError(
                f"The {name} width must be non-negative, not {width}."
            )
    need_stations = any(w is not None for w in built_in.values())
    index, frame, stations = _frames_and_stations(data, need_stations)
    groups, widths, columns = [], [], []  # columns: (group, rows, values)

    def add_group(name, width):
        groups.append(name)
        widths.append(float(width))
        return len(groups) - 1

    if need_stations:
        wavel = onp.broadcast_to(onp.asarray(data.wavel), onp.shape(data.u))[
            index
        ]
        chrom = (onp.median(wavel) / wavel) ** 2
        ids = {
            name: add_group(name, w)
            for name, w in built_in.items()
            if w is not None
        }
        for f in onp.unique(frame):
            in_frame = onp.flatnonzero(frame == f)
            pairs = stations[in_frame]
            if "telescope" in ids:
                for t in onp.unique(pairs):
                    rows = in_frame[(pairs == t).any(axis=1)]
                    columns.append(
                        (ids["telescope"], rows, onp.ones(rows.size))
                    )
            for pair in onp.unique(pairs, axis=0):
                rows = in_frame[(pairs == pair).all(axis=1)]
                if "baseline" in ids:
                    columns.append(
                        (ids["baseline"], rows, onp.ones(rows.size))
                    )
                if "chromatic" in ids:
                    columns.append((ids["chromatic"], rows, chrom[rows]))
    if modes is not None:
        modes = onp.atleast_2d(onp.asarray(modes, float))
        if modes.shape[1] != onp.size(data.u):
            raise ValueError(
                f"modes has {modes.shape[1]} values per mode for "
                f"{onp.size(data.u)} samples; give one per sample of data.u."
            )
        modes = modes[:, index]
        gid = add_group("modes", 1.0)
        for m in modes:
            rows = onp.flatnonzero(m)
            if rows.size:
                columns.append((gid, rows, m[rows]))
    if not columns:
        raise ValueError("No gain modes: give at least one group or mode.")
    return _pack(columns, groups, widths, frame)


def _pack(columns, groups, widths, frame):
    """Sort columns into blocks within frames, and columns spanning frames.

    Columns within one frame are connected when they share a row; each
    connected group (a component of the bipartite graph of columns and
    rows) is a block, padded to a common size.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    n_vis = frame.size
    within = [onp.unique(frame[r]).size == 1 for _, r, _ in columns]
    local = [c for c, w in zip(columns, within) if w]
    spanning_cols = [c for c, w in zip(columns, within) if not w]
    spanning = onp.zeros((n_vis, len(spanning_cols)))
    for j, (_, r, values) in enumerate(spanning_cols):
        spanning[r, j] = values
    spanning_group = onp.array([g for g, _, _ in spanning_cols], onp.int32)

    blocks = []
    if local:
        n_col = len(local)
        col_ids = onp.concatenate(
            [onp.full(r.size, k) for k, (_, r, _) in enumerate(local)]
        )
        row_ids = onp.concatenate([r for _, r, _ in local])
        graph = coo_matrix(
            (onp.ones(col_ids.size), (col_ids, n_col + row_ids)),
            shape=(n_col + n_vis, n_col + n_vis),
        )
        _, label = connected_components(graph, directed=False)
        by_label = {}
        for k in range(n_col):
            by_label.setdefault(label[k], []).append(k)
        blocks = list(by_label.values())
    block_rows = [
        onp.unique(onp.concatenate([local[k][1] for k in ks])) for ks in blocks
    ]
    # With no blocks, keep one empty block (rows out of range, zero shapes,
    # adding log(1 + 0) = 0) rather than zero-size arrays, which jax.pmap
    # cannot compile (see GainModes.spanning).
    n_row = max((r.size for r in block_rows), default=1)
    n_mode = max((len(ks) for ks in blocks), default=1)
    rows = onp.full((max(len(blocks), 1), n_row), n_vis, dtype=onp.int32)
    shapes = onp.zeros((max(len(blocks), 1), n_row, n_mode))
    group = onp.zeros((max(len(blocks), 1), n_mode), dtype=onp.int32)
    for b, (ks, r) in enumerate(zip(blocks, block_rows)):
        rows[b, : r.size] = r
        for j, k in enumerate(ks):
            g, col_rows, values = local[k]
            shapes[b, onp.searchsorted(r, col_rows), j] = values
            group[b, j] = g
    return GainModes(
        np.asarray(rows),
        np.asarray(shapes),
        np.asarray(group),
        np.asarray(spanning) if spanning_cols else None,
        np.asarray(spanning_group) if spanning_cols else None,
        np.asarray(widths, float),
        tuple(groups),
        int(n_vis),
    )


# Groups of closure-phase offsets; widths are the noise terms
# "phi_offset_<group>".
OFFSET_GROUPS = ("baseline", "triangle", "modes")


class ClosureOffsets(eqx.Module):
    """Closure-phase offsets common to the channels of a frame, marginalized.

    Built by [`closure_offsets`][virgil.gains.closure_offsets] (or
    [`OIData.with_closure_offsets`][virgil.oidata.OIData.with_closure_offsets]).
    Each offset is a mode m over the closure phases, in radians per unit
    width: φ_obs = φ + Σ τ_j z_j m_j with z_j ~ N(0, 1).

    The likelihood of correlated closure phases whitens their sines with
    ``OIData.cp_noise`` (see ``likelihood._whiten``). The offsets add
    τ² m mᵀ to that covariance: a small-phase approximation, since an
    offset δ changes sin Δ by about δ cos Δ. It holds for offsets (and
    residuals) below about 0.3 rad; larger widths are not marginalized
    exactly. Each mode is
    whitened the same way, one closure-phase group (frame and channel) at
    a time, and the modes of a frame then form one block of the rank-one
    whitening ([`GainModes`][virgil.gains.GainModes] has the details).
    The periodic penalty rows are left as they are.

    Attributes
    ----------
    groups : numpy.ndarray
        ``(n_block, n_group)`` int32: the ``cp_noise`` groups of each block.
    group_mask : numpy.ndarray
        ``(n_block, n_group)``: True for real groups (the rest is padding).
    values : numpy.ndarray
        ``(n_block, n_mode, n_group, m)``: each mode's value at each slot
        of each group (as ``cp_noise.groups``), zero on padding.
    group : numpy.ndarray
        ``(n_block, n_mode)`` int32: each mode's width group.
    rows : numpy.ndarray
        ``(n_block, n_group · k)`` int32: the whitened row of each (group,
        basis row), padded with ``n_out`` (out of range).
    widths : jax.Array
        The default width of each group, in radians.
    groups_present : tuple of str
        The groups present, a subset of ``OFFSET_GROUPS``.
    n_out : int
        The number of whitened closure phases (``cp_noise.size``).
    """

    groups: onp.ndarray
    group_mask: onp.ndarray
    values: onp.ndarray
    group: onp.ndarray
    rows: onp.ndarray
    widths: jax.Array
    groups_present: tuple = eqx.field(static=True)
    n_out: int = eqx.field(static=True)

    def widths_for(self, terms=None):
        """Each group's width, with ``phi_offset_<group>`` terms replacing them."""
        return _widths_for(
            self.groups_present, self.widths, terms, "phi_offset_",
            "OIData.with_closure_offsets",
        )  # fmt: skip

    def _slots(self, cp_noise):
        """Closure-phase index and validity of each (block, group, slot)."""
        index = np.asarray(cp_noise.groups)[self.groups]
        mask = np.asarray(cp_noise.mask)[self.groups]
        return index, mask & np.asarray(self.group_mask)[..., None]

    def _columns(self, cp_noise, sigma, widths):
        """The modes, whitened like the sines: (n_block, n_group · k, n_mode)."""
        sigma = np.asarray(sigma)
        values = np.asarray(self.values, sigma.dtype)
        index, mask = self._slots(cp_noise)
        x = np.where(mask[:, None], values / sigma[index][:, None], 0.0)
        basis = np.asarray(cp_noise.basis, sigma.dtype)[self.groups]
        chol = np.asarray(cp_noise.chol, sigma.dtype)[self.groups]
        a = np.einsum("bgkm,bjgm->bjgk", basis, x)
        chol = np.broadcast_to(chol[:, None], a.shape + a.shape[-1:])
        w = jsl.solve_triangular(chol, a[..., None], lower=True)[..., 0]
        n_block, n_mode, n_group, k = w.shape
        cols = w.transpose(0, 2, 3, 1).reshape(n_block, n_group * k, n_mode)
        return cols * np.asarray(widths)[self.group][:, None, :]

    def whiten(self, cp_noise, x, sigma, widths):
        """Whiten the closure-phase sines ``x`` (already whitened by ``cp_noise``).

        Returns the whitened values and, per value, the log of the factor its
        effective error grows by (summing to ½ log det).
        """
        cols = self._columns(cp_noise, sigma, widths)
        return whiten_blocks(x, np.asarray(self.rows), cols)

    def modes(self, cp_noise, n_phase, widths=None):
        """The modes as dense columns over the closure phases (for checks)."""
        widths = self.widths if widths is None else widths
        group = np.asarray(self.group)
        scaled = (
            np.asarray(self.values)
            * np.asarray(widths)[group][..., None, None]
        )
        index, mask = self._slots(cp_noise)
        n_block, n_mode = group.shape
        b = np.arange(n_block)[:, None, None, None]
        j = np.arange(n_mode)[None, :, None, None]
        idx = np.where(mask, index, n_phase)[:, None]
        out = np.zeros((n_phase, n_block, n_mode))
        out = out.at[idx, b, j].set(scaled, mode="drop")
        return out.reshape(n_phase, -1)

    def sample(self, key, cp_noise, n_phase, widths):
        """Draw the offsets: radians added to each closure phase."""
        group = np.asarray(self.group)
        z = jax.random.normal(key, group.shape) * np.asarray(widths)[group]
        per_slot = np.einsum("bjgm,bj->bgm", np.asarray(self.values), z)
        index, mask = self._slots(cp_noise)
        index = np.where(mask, index, n_phase)
        return np.zeros(n_phase).at[index].add(per_slot, mode="drop")


def _widths_for(groups, widths, terms, prefix, how):
    """Each group's width, with ``<prefix><group>`` terms replacing them."""
    terms = {} if terms is None else terms
    unknown = [t for t in terms if t.removeprefix(prefix) not in groups]
    if unknown:
        raise ValueError(
            f"Noise terms {unknown} have no modes in these data (they have "
            f"{list(groups)}); add them with {how}."
        )
    return np.stack(
        [
            np.asarray(terms.get(f"{prefix}{g}", widths[i]), float)
            for i, g in enumerate(groups)
        ]
    )


def closure_offsets(data, baseline=None, triangle=None, modes=None):
    """Closure-phase offsets for a dataset, with default widths.

    Parameters
    ----------
    data : OIData
        Closure phases from four or more telescopes (``cp_noise``), not
        projected, with frames and, for baseline and triangle offsets,
        station pairs, as read from OIFITS.
    baseline : float, optional
        Width (radians) of a phase offset per (frame, baseline), common to
        all channels, which reaches the closure phases as T·e (T the
        triangle-by-baseline signs). The design's default form.
    triangle : float, optional
        Width (radians) of an offset per (frame, triangle), common to all
        channels: what a pair test of consecutive frames measures.
    modes : array-like, optional
        Further modes, ``(n_mode, n_phase)``, in radians per 1σ, one value
        per closure phase of ``data.phi``. Each must lie within one frame.
        Their width (default 1) scales them all.

    Returns
    -------
    ClosureOffsets
    """
    cp_noise = data.cp_noise
    if not data.cp_flag or cp_noise is None or data.phi_mat is not None:
        raise ValueError(
            "Closure-phase offsets need correlated closure phases from four "
            "or more telescopes (OIData.cp_noise), unprojected. Three "
            "telescopes are not supported yet."
        )
    if data.frame is None:
        raise ValueError(
            "Offsets per frame need each sample's frame: read the data from "
            "OIFITS, or give 'frame' (or 'mjd') in the dictionary."
        )
    for name, width in (("baseline", baseline), ("triangle", triangle)):
        if width is not None and not (onp.isfinite(width) and width >= 0):
            raise ValueError(
                f"The {name} width must be non-negative, not {width}."
            )
    legs = [onp.asarray(i) for i in (data.i_cps1, data.i_cps2, data.i_cps3)]
    n_phase = legs[0].size
    frame = onp.asarray(data.frame)[legs[0]]
    names, widths, columns = [], [], []  # columns: (group, rows, values)

    def add_group(name, width):
        names.append(name)
        widths.append(float(width))
        return len(names) - 1

    if baseline is not None or triangle is not None:
        if data.stations is None:
            raise ValueError(
                "Baseline and triangle offsets need each sample's stations: "
                "read the data from OIFITS, or give 'stations' (station "
                "pairs) in the dictionary."
            )
        stations = onp.asarray(data.stations)
        pairs = [stations[leg] for leg in legs]
    if baseline is not None:
        gid = add_group("baseline", baseline)
        found = {}
        for leg_pairs, sign in zip(pairs, (1.0, 1.0, -1.0)):
            # A sample's phase is that of its stored pair (a, b); a baseline
            # offset e on (min, max) is +e there if a < b, -e otherwise.
            for t, (a, b) in enumerate(leg_pairs):
                key = (frame[t], min(a, b), max(a, b))
                rows, values = found.setdefault(key, ([], []))
                rows.append(t)
                values.append(sign if a < b else -sign)
        for rows, values in found.values():
            rows, values = onp.asarray(rows), onp.asarray(values)
            order = onp.argsort(rows)
            columns.append((gid, rows[order], values[order]))
    if triangle is not None:
        gid = add_group("triangle", triangle)
        found = {}
        for t in range(n_phase):
            key = (frame[t], *pairs[0][t], pairs[1][t][1])
            found.setdefault(key, []).append(t)
        for rows in found.values():
            columns.append((gid, onp.asarray(rows), onp.ones(len(rows))))
    if modes is not None:
        modes = onp.atleast_2d(onp.asarray(modes, float))
        if modes.shape[1] != n_phase:
            raise ValueError(
                f"modes has {modes.shape[1]} values per mode for {n_phase} "
                "closure phases; give one per closure phase of data.phi."
            )
        gid = add_group("modes", 1.0)
        for m in modes:
            rows = onp.flatnonzero(m)
            if onp.unique(frame[rows]).size > 1:
                raise ValueError(
                    "A closure-phase mode spans several frames; each must "
                    "lie within one frame."
                )
            if rows.size:
                columns.append((gid, rows, m[rows]))
    if not columns:
        raise ValueError("No offsets: give at least one group or mode.")
    return _pack_offsets(cp_noise, columns, names, widths)


def _pack_offsets(cp_noise, columns, names, widths):
    """Group closure-phase modes into blocks of the groups they touch."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    cp_groups, cp_mask = (
        onp.asarray(cp_noise.groups),
        onp.asarray(cp_noise.mask),
    )
    n_group, m = cp_groups.shape
    group_of = onp.zeros(cp_groups.max() + 1, int)
    slot_of = onp.zeros(cp_groups.max() + 1, int)
    g_idx, s_idx = onp.nonzero(cp_mask)
    group_of[cp_groups[g_idx, s_idx]] = g_idx
    slot_of[cp_groups[g_idx, s_idx]] = s_idx
    # Modes are connected when they touch a common group.
    n_col = len(columns)
    col_ids = onp.concatenate(
        [onp.full(r.size, k) for k, (_, r, _) in enumerate(columns)]
    )
    grp_ids = onp.concatenate([group_of[r] for _, r, _ in columns])
    graph = coo_matrix(
        (onp.ones(col_ids.size), (col_ids, n_col + grp_ids)),
        shape=(n_col + n_group, n_col + n_group),
    )
    _, label = connected_components(graph, directed=False)
    by_label = {}
    for k in range(n_col):
        by_label.setdefault(label[k], []).append(k)
    blocks = list(by_label.values())
    block_groups = [
        onp.unique(onp.concatenate([group_of[columns[k][1]] for k in ks]))
        for ks in blocks
    ]
    n_block = len(blocks)
    n_g = max(g.size for g in block_groups)
    n_mode = max(len(ks) for ks in blocks)
    k_basis = onp.asarray(cp_noise.basis).shape[1]
    # The whitened row of each (group, basis row): cp_noise keeps the valid
    # rows of its (n_group, k) output, in order.
    n_out = int(cp_noise.size)
    position = onp.full(n_group * k_basis, n_out, dtype=onp.int32)
    position[onp.asarray(cp_noise.keep)] = onp.arange(n_out)
    groups = onp.zeros((n_block, n_g), dtype=onp.int32)
    group_mask = onp.zeros((n_block, n_g), dtype=bool)
    values = onp.zeros((n_block, n_mode, n_g, m))
    group = onp.zeros((n_block, n_mode), dtype=onp.int32)
    rows = onp.full((n_block, n_g * k_basis), n_out, dtype=onp.int32)
    for b, (ks, gs) in enumerate(zip(blocks, block_groups)):
        groups[b, : gs.size] = gs
        group_mask[b, : gs.size] = True
        where = {g: i for i, g in enumerate(gs)}
        for j, k in enumerate(ks):
            g, cps, vals = columns[k]
            gi = onp.array([where[x] for x in group_of[cps]])
            values[b, j, gi, slot_of[cps]] = vals
            group[b, j] = g
        for i, g in enumerate(gs):
            rows[b, i * k_basis : (i + 1) * k_basis] = position[
                g * k_basis : (g + 1) * k_basis
            ]
    return ClosureOffsets(
        groups,
        group_mask,
        values,
        group,
        rows,
        np.asarray(widths, float),
        tuple(names),
        n_out,
    )
