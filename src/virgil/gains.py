"""Calibration gains correlated across channels, marginalised analytically.

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
are then Gaussian and are marginalised analytically: the visibility
covariance becomes

    C = D + U Uᵀ,   U = J τ m  (one column per mode),

with D the diagonal of squared errors. Modes that share no sample are
independent, so C is block diagonal, one block per connected group of
modes (one per frame, for the built-in groups).

**Whitening.** Each block is whitened by k rank-one steps. With D's
whitening already applied (x = D^{-½} r, w_j = D^{-½} u_j), a rank-one
covariance I + w wᵀ is whitened by

    R = I − w wᵀ / (q (q + 1)),   q = √(1 + wᵀw),

since R (I + w wᵀ) Rᵀ = I; its log-determinant is log(1 + wᵀw). Applying R
to x and to the remaining modes, mode by mode, whitens I + Σ w_j w_jᵀ
(Woodbury, as a sequence of updates). Only square roots of scalars appear,
so the gradients stay smooth, also where modes are degenerate (an
eigendecomposition's are not) and where widths go to zero.
"""

import equinox as eqx
import jax
import jax.numpy as np
import numpy as onp

__all__ = ["GainModes", "GAIN_GROUPS", "gain_modes"]

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
        (rows of ``OIData.vis``), padded with ``n_vis`` (out of range).
    shapes : jax.Array
        ``(n_block, n_row, n_mode)``: each mode's shape on log |V| for unit
        width, zero on padding.
    group : jax.Array
        ``(n_block, n_mode)`` int32: each mode's group, an index into
        ``groups``.
    spanning : jax.Array
        ``(n_vis, n_spanning)``: the shapes of modes that span frames.
    spanning_group : jax.Array
        ``(n_spanning,)`` int32: their groups.
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
    spanning: jax.Array
    spanning_group: jax.Array
    widths: jax.Array
    groups: tuple = eqx.field(static=True)
    n_vis: int = eqx.field(static=True)

    def widths_for(self, terms=None):
        """Each group's width, with ``vis_gain_<group>`` terms replacing them."""
        terms = {} if terms is None else terms
        unknown = [
            t for t in terms if t.removeprefix("vis_gain_") not in self.groups
        ]
        if unknown:
            raise ValueError(
                f"Noise terms {unknown} have no gain modes in these data "
                f"(they have {list(self.groups)}); add them with "
                "OIData.with_gains."
            )
        return np.stack(
            [
                np.asarray(terms.get(f"vis_gain_{g}", self.widths[i]), float)
                for i, g in enumerate(self.groups)
            ]
        )

    def _columns(self, jacobian, widths):
        """The modes as columns of D^{-½} U: per block, and spanning frames."""
        jacobian = np.asarray(jacobian)
        scale = jacobian.at[self.rows].get(mode="fill", fill_value=0)
        local = self.shapes * scale[..., None] * widths[self.group][:, None, :]
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
        x = np.asarray(x)
        local, spanning = self._columns(jacobian, widths)
        n_span = spanning.shape[1]
        # The blocks first, carrying the spanning modes through them: the
        # whitening of the blocks is applied to every column.
        stack = np.concatenate(
            [
                x.at[self.rows].get(mode="fill", fill_value=0)[..., None],
                spanning.at[self.rows].get(mode="fill", fill_value=0),
                local,
            ],
            axis=-1,
        )
        stack, logdets = jax.lax.scan(
            _rank_one_step,
            stack,
            1 + n_span + np.arange(local.shape[-1]),
        )
        done = (
            np.concatenate([x[:, None], spanning], axis=1)
            .at[self.rows]
            .set(stack[..., : 1 + n_span], mode="drop")
        )
        # Spread each block's ½ log det over its rows.
        n_rows = np.sum(self.rows < self.n_vis, axis=1)
        per_row = 0.5 * np.sum(logdets, axis=0) / np.maximum(n_rows, 1)
        extra = (
            np.zeros_like(x)
            .at[self.rows]
            .set(
                np.broadcast_to(per_row[:, None], self.rows.shape), mode="drop"
            )
        )
        if n_span:
            # Then the spanning modes, as one dense block, their ½ log det
            # spread over every observable.
            done, span_logdets = jax.lax.scan(
                _rank_one_step, done[None], 1 + np.arange(n_span)
            )
            done = done[0]
            extra = extra + 0.5 * np.sum(span_logdets) / self.n_vis
        return done[:, 0], extra

    def covariance(self, errors, jacobian, widths):
        """The dense visibility covariance ``D + U Uᵀ`` (for checks; O(n²)).

        ``jacobian`` is dObs/dlog|V| itself here, not divided by the errors.
        """
        errors = np.asarray(errors)
        local, spanning = self._columns(np.asarray(jacobian), widths)
        u = np.zeros((self.n_vis,) + local.shape[::2])
        b = np.arange(local.shape[0])[:, None]
        u = u.at[self.rows, b].set(local, mode="drop")
        u = np.concatenate([u.reshape(self.n_vis, -1), spanning], axis=1)
        return np.diag(errors**2) + u @ u.T

    def sample(self, key, widths):
        """Draw the gains: log |V| offset of each visibility observable."""
        widths = np.asarray(widths)
        local_key, span_key = jax.random.split(key)
        z = jax.random.normal(local_key, self.group.shape)
        per_row = np.einsum("brk,bk->br", self.shapes, z * widths[self.group])
        z = jax.random.normal(span_key, self.spanning_group.shape)
        return np.zeros(self.n_vis).at[self.rows].add(
            per_row, mode="drop"
        ) + self.spanning @ (z * widths[self.spanning_group])

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
            self.spanning[keep],
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
        for col in onp.asarray(self.spanning).T:
            if onp.unique(labels[col != 0]).size > 1:
                return True
        return False


def _rank_one_step(stack, j):
    """Whiten ``stack`` (..., n_row, n_col) for its column ``j`` as a mode."""
    w = stack[..., j]
    s = np.sum(w**2, axis=-1)
    q = np.sqrt(1.0 + s)
    coef = np.einsum("...r,...rk->...k", w, stack) / (q * (q + 1.0))[..., None]
    return stack - w[..., None] * coef[..., None, :], np.log1p(s)


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
    n_vis = index.size
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
    n_row = max((r.size for r in block_rows), default=0)
    n_mode = max((len(ks) for ks in blocks), default=0)
    rows = onp.full((len(blocks), n_row), n_vis, dtype=onp.int32)
    shapes = onp.zeros((len(blocks), n_row, n_mode))
    group = onp.zeros((len(blocks), n_mode), dtype=onp.int32)
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
        np.asarray(spanning),
        np.asarray(spanning_group),
        np.asarray(widths, float),
        tuple(groups),
        int(n_vis),
    )
