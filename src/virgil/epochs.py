"""Named epochs of data, with one snapshot of a moving scene per dataset.

!!! warning "Provisional"
    May change in any 0.4.x release without a deprecation cycle.

When nothing moves appreciably during an observation (a period much longer
than a night), one static snapshot of the scene per dataset replaces
[`OIData.model`][virgil.oidata.OIData.model]'s evaluation at every sample's own
time, and is much cheaper. [`Epochs`][virgil.epochs.Epochs] keys epochs and
datasets **by name**, so per-dataset nuisance terms stay attached to the right
data, and hands [`fit`][virgil.fitting.fit] and
[`numpyro_model`][virgil.likelihood.numpyro_model] a model function, the
datasets and one ``noise`` dict per dataset.

Fringe aliases make the likelihood multimodal on the scale of λ/B, so an orbit
fit needs a start in the right basin:

* [`rank_orbits`][virgil.epochs.rank_orbits]: trial orbits ranked by the
  likelihood of all the data;
* [`chain_starts`][virgil.epochs.chain_starts]: the best *distinct* orbits, one
  per chain;
* [`epoch_positions`][virgil.epochs.epoch_positions]: the companion's position
  in each dataset;
* [`start_from_positions`][virgil.epochs.start_from_positions]: the whole start
  (positions, starting orbits, ranking, [`fit`][virgil.fitting.fit] from
  distinct orbits); [`OrbitStart.chain_values`][virgil.epochs.OrbitStart.chain_values]
  gives one start per chain for
  [`chain_init_params`][virgil.likelihood.chain_init_params].
"""

import copy
import dataclasses
import functools
import math
import types
import warnings

import equinox as eqx
import jax
import jax.numpy as np
import numpy as onp
from jax.scipy.special import i0e

from . import _gauss_newton
from ._geometry import fringe_scales
from ._grid import batch_size_or_default, batch_visibilities
from ._marginal import gauss_legendre, log_integral
from ._precision import cast_tree, run_in
from .fitting import fit
from .likelihood import (
    _gaussian_loglike,
    _prediction_and_errors,
    _whiten_with_log_norm,
    _whitened_and_errors,
    model_loglike,
    whitened_residuals,
)
from .models import BinaryModelCartesian, PointSource, System, UniformDisk
from .orbits import (
    _PHASE_COHERENCE,
    PositionData,
    _phase_drift,
    _warn_if_mjd_without_t_ref,
    starting_orbits,
)
from ._deprecate import external_stacklevel


__all__ = [
    "Epochs",
    "EpochPeaks",
    "EpochPositions",
    "NightFlag",
    "OrbitStart",
    "PrimaryDiameter",
    "RESOLVE_NATS",
    "RankedOrbits",
    "broad_loglike",
    "chain_starts",
    "epoch_positions",
    "estimate_primary_diameter",
    "marginal_loglike",
    "sidelobe_flags",
    "rank_orbits",
    "start_from_positions",
]

__provisional__ = ("*",)

_AT = ("dataset", "epoch")


class Epochs:
    """Datasets grouped into named epochs, each seen as one snapshot.

    Parameters
    ----------
    epochs : dict
        ``{epoch name: data}``, where ``data`` is one
        [`OIData`][virgil.oidata.OIData], a list of them (named
        ``"<epoch>[k]"``), or a dict ``{dataset name: OIData}``. A lone dataset
        is named after its epoch. Names must be unique.
    at : {"dataset", "epoch"}, optional
        Snapshot at the mean time of each dataset's samples (default) or of
        the whole epoch.
    times : dict, optional
        ``{epoch or dataset name: MJD}`` replacing the data's own times (a
        dataset's entry overrides its epoch's).

    Attributes
    ----------
    names, dataset_names, epoch_of : tuple of str
        The epochs, the datasets (epoch by epoch), and each dataset's epoch.
    data : tuple of OIData
        The datasets in ``dataset_names`` order: pass to ``fit``/``numpyro_model``.
    times, spread_days : numpy.ndarray
        Snapshot time (MJD) of each dataset, and the largest distance of its
        samples from it (days); the scene should move far less than the
        resolution over it.
    """

    def __init__(self, epochs, at="dataset", times=None):
        if at not in _AT:
            raise ValueError(f"at must be one of {_AT}, got {at!r}.")
        if not isinstance(epochs, dict) or not epochs:
            raise ValueError(
                "epochs must be a non-empty dict {epoch name: data}."
            )
        names, epoch_of, data = [], [], []
        for epoch, group in epochs.items():
            if isinstance(group, dict):
                items = list(group.items())
            elif isinstance(group, (list, tuple)):
                items = [(f"{epoch}[{k}]", d) for k, d in enumerate(group)]
            else:
                items = [(epoch, group)]
            if not items:
                raise ValueError(f"Epoch {epoch!r} has no data.")
            for name, d in items:
                names.append(name)
                epoch_of.append(epoch)
                data.append(d)
        duplicated = sorted(
            {n for n in names if names.count(n) > 1}
            | {n for n, e in zip(names, epoch_of) if n in epochs and n != e}
        )
        if duplicated:
            raise ValueError(
                f"Names must be unique; {duplicated} name both an epoch "
                "and a dataset, or two datasets."
            )
        self.names = tuple(epochs)
        self.dataset_names = tuple(names)
        self.epoch_of = tuple(epoch_of)
        self.data = tuple(data)
        self.at = at
        given = times or {}
        # Each dataset's snapshot time and the spread of its samples about it.
        unknown = set(given) - set(self.names) - set(self.dataset_names)
        if unknown:
            raise ValueError(f"times for unknown names: {sorted(unknown)}.")
        own = []
        for name, d in zip(self.dataset_names, self.data):
            mjd = getattr(d, "mjd", None)
            own.append(None if mjd is None else onp.asarray(mjd, onp.float64))
        snap = onp.empty(len(self.data))
        for k, (name, epoch) in enumerate(
            zip(self.dataset_names, self.epoch_of)
        ):
            if name in given:
                snap[k] = given[name]
            elif epoch in given:
                snap[k] = given[epoch]
            elif own[k] is None:
                raise ValueError(
                    f"Dataset {name!r} has no times: read it from OIFITS, "
                    "give mjd per sample, or pass times={name: mjd}."
                )
            elif self.at == "dataset":
                snap[k] = own[k].mean()
            else:
                together = [
                    own[j]
                    for j, e in enumerate(self.epoch_of)
                    if e == epoch and own[j] is not None
                ]
                snap[k] = onp.concatenate(together).mean()
        spread = onp.array(
            [
                0.0 if t is None else float(onp.max(onp.abs(t - snap[k])))
                for k, t in enumerate(own)
            ]
        )
        self.times, self.spread_days = snap, spread

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        return (
            f"Epochs({len(self.names)} epochs, {len(self.data)} datasets, "
            f"at={self.at!r})"
        )

    @property
    def resolution_mas(self):
        """The finest resolution λ/B_max over the datasets (mas).

        Only unflagged samples count (see
        [`OIData.unflagged`][virgil.oidata.OIData.unflagged]).

        Fringe aliases of a companion's position are spaced by about this
        much, so starting orbits closer together than a fraction of it are
        one mode.
        """
        return float(fringe_scales(list(self.data))[0])

    def index(self, name):
        """The position of dataset ``name`` in ``data`` (and in ``noise``).

        Fitted per-dataset noise terms are the sites
        ``"noise[<index>].<term>"`` of ``fit`` and ``numpyro_model``.
        """
        try:
            return self.dataset_names.index(name)
        except ValueError:
            raise KeyError(
                f"No dataset {name!r}; the datasets are "
                f"{list(self.dataset_names)}."
            ) from None

    def snapshots(self, scene):
        """The scene at each dataset's snapshot time, one model per dataset.

        Parameters
        ----------
        scene : SourceModel or sequence of SourceModel
            One scene for all datasets, or one per dataset (e.g. with
            per-dataset fluxes). A static scene is its own snapshot.

        Returns
        -------
        list of SourceModel
            Static models, in the order of ``data``.
        """
        if not isinstance(scene, (list, tuple)):
            times = onp.asarray(self.times, onp.float64)
            if not scene.time_dependent:
                return [scene] * len(times)
            for orbit in jax.tree_util.tree_leaves(
                scene, is_leaf=lambda x: hasattr(x, "_relative")
            ):
                if hasattr(orbit, "_relative"):
                    _warn_if_mjd_without_t_ref(orbit.t_ref, times)
            # One vmap over the times: one Kepler solve and one copy of the
            # scene graph for all datasets. The offsets from the mean time
            # (a float64 ``t_ref``) keep the phase in float32.
            t0 = float(times.mean())
            batched = eqx.filter_vmap(lambda dt: scene.at(dt, t0))(
                np.asarray(times - t0)
            )
            return [
                jax.tree_util.tree_map(
                    lambda x: x[k] if eqx.is_array(x) else x, batched
                )
                for k in range(len(times))
            ]
        scenes = list(scene)
        if len(scenes) != len(self.data):
            raise ValueError(
                f"{len(scenes)} scenes for {len(self.data)} datasets."
            )
        return [s.at(float(t)) for s, t in zip(scenes, self.times)]

    def model_fn(self, scene_fn):
        """A model function for ``fit``/``numpyro_model``: the snapshots of
        ``scene_fn(**values)``, one per dataset."""

        def snapshots(**values):
            return self.snapshots(scene_fn(**values))

        return snapshots

    def noise(self, terms):
        """One ``noise`` dict per dataset, from terms keyed by name.

        Parameters
        ----------
        terms : dict
            ``{epoch or dataset name: {term: prior, value or tied
            callable}}``. An epoch's terms apply to each of its datasets,
            and each dataset gets its own copy (its own fitted site); a
            dataset's own terms are added to, and override, its epoch's.
            To share one fitted value between datasets, tie them to one
            parameter with callables (see
            [`noise_sites`][virgil.likelihood.noise_sites]).

        Returns
        -------
        list of dict
            In the order of ``data``, as ``fit(..., noise=...)`` takes.
        """
        unknown = set(terms) - set(self.names) - set(self.dataset_names)
        if unknown:
            raise KeyError(
                f"noise for unknown names {sorted(unknown)}; the epochs are "
                f"{list(self.names)} and the datasets "
                f"{list(self.dataset_names)}."
            )
        return [
            {**terms.get(epoch, {}), **terms.get(name, {})}
            for name, epoch in zip(self.dataset_names, self.epoch_of)
        ]

    def loglike(self, scene, noise=None):
        """The log likelihood of all the data, one snapshot per dataset.

        The sum over datasets of
        [`model_loglike`][virgil.likelihood.model_loglike] of each
        snapshot, with ``noise`` (values, keyed by name as for
        :meth:`noise`) applied to each dataset. Traceable: it can be
        jitted, differentiated or vmapped in the scene's parameters.
        """
        per_dataset = self.noise(noise or {})
        return sum(
            model_loglike(snap, d, **n)
            for snap, d, n in zip(
                self.snapshots(scene), self.data, per_dataset
            )
        )


# ---------------------------------------------------------------------------
# Starting and sampling orbit fits
# ---------------------------------------------------------------------------


def _orbit_list(orbits):  # deliberate: imported by orbit_search.py
    """A list of orbits from a list, ``starting_orbits`` pairs, a
    ``RankedOrbits`` or one orbit with a leading batch axis."""
    if isinstance(orbits, (list, tuple, RankedOrbits)):
        listed = [o[0] if isinstance(o, tuple) else o for o in orbits]
    elif onp.ndim(jax.tree_util.tree_leaves(orbits)[0]) == 0:
        listed = [orbits]
    else:
        n = len(jax.tree_util.tree_leaves(orbits)[0])
        listed = [
            jax.tree_util.tree_map(lambda x, k=k: x[k], orbits)
            for k in range(n)
        ]
    if not listed:
        raise ValueError("There are no orbits to rank.")
    return listed


def _stack(trees, what="orbits"):
    """Stack pytrees of one structure along a new leading axis."""
    try:
        return jax.tree_util.tree_map(
            lambda *xs: np.stack([np.asarray(x) for x in xs]), *trees
        )
    except ValueError as err:
        raise ValueError(
            f"The {what} must share one structure (the same class and, for "
            f"orbits, the same t_ref): {err}"
        ) from None


def _blocks(data):  # deliberate: used by pipeline/_common.py
    """The observable blocks of ``data``'s whitened residuals, each with
    its own error scale: ``(name, start, stop, ν, von_mises)``.

    The blocks are the visibilities (``"vis"``, scaled by ``vis_scale``),
    the phases (``"phi"``, ``phi_scale``; for correlated closure phases,
    the independent combinations followed by the periodic penalty rows,
    of which only the former count in ``ν``) and each extra observable
    block (``"extra[k]"``). ``ν`` is the block's number of independent
    observables (they add up to ``data.n_independent``), and
    ``von_mises`` marks uncorrelated wrapping phases, whose exact
    likelihood is a von Mises density. Empty blocks are left out.
    """
    n_vis = int(onp.asarray(data.vis).size)
    n_phi = int(onp.asarray(data.phi).size)
    blocks = []
    if n_vis:
        blocks.append(("vis", 0, n_vis, n_vis, False))
    if n_phi:
        correlated = data._phases_wrap and data.cp_noise is not None
        nu = int(data.cp_noise.size) if correlated else n_phi
        rows = nu + n_phi if correlated else nu
        von_mises = bool(data._phases_wrap and data.cp_noise is None)
        blocks.append(("phi", n_vis, n_vis + rows, nu, von_mises))
    stop = blocks[-1][2] if blocks else 0
    for k, block in enumerate(data.extras):
        nu = int(block.n_independent)
        blocks.append((f"extra[{k}]", stop, stop + nu, nu, False))
        stop += nu
    if stop != data.n_residuals or not blocks:
        raise ValueError(
            "Cannot split the whitened residuals into observable blocks "
            f"({stop} rows for n_residuals = {data.n_residuals})."
        )
    return tuple(blocks)


def _dof_for(dof, data, index):
    """The effective-dof fraction of dataset ``index``: one number, or
    a dict keyed by dataset or epoch name (default 1)."""
    if not isinstance(dof, dict):
        return float(dof)
    name = data.dataset_names[index]
    return float(dof.get(name, dof.get(data.epoch_of[index], 1.0)))


def _check_dof(dof):
    dof = float(dof)
    if not 0.0 < dof <= 1.0:
        raise ValueError(
            f"The effective-dof fraction must be in (0, 1], got {dof}."
        )
    return dof


def _refuse_amp_with_gains(
    data,
):  # deliberate: imported by aliases.py; unit-tested
    """Raise if ``data`` carry gains and a V² amplitude is requested: the
    amplitude profile rebuilds the model as ``obs + r·err``, valid only for
    diagonal errors, and the gains supersede a per-night amplitude."""
    if data.gains is not None or data.phase_offsets is not None:
        raise ValueError(
            "vis_amp (a per-night V² amplitude) cannot be used with data "
            "that carry gains or closure offsets (OIData.with_gains, "
            "with_closure_offsets): they are marginalized with the error "
            "scale, and the gains supersede a per-night amplitude. Use "
            "vis_amp=False or None."
        )


# Nodes of the Gauss-Legendre rule that integrates ln s of a block whose
# covariance is s²D + UUᵀ (gains, closure offsets), the half-width of its
# window in standard deviations of the (tempered) integrand, the Newton
# steps that find its peak, and the bound that ln s takes when the caller
# gives none (a numeric integral needs one).
_LN2 = math.log(2.0)
_N_FLUX = 13  # log-spaced flux nodes of the position grids (as OrbitPipeline)
_N_NODES = 32
_WINDOW = 9.0
_N_CENTRE = 4
_NUMERIC_S_MAX = 10.0


class _Surface:
    """The scale-marginalized score of one dataset, block by block.

    Only the layout of the data enters (blocks, their nu and ``s_max``), so one
    compiled kernel serves every dataset of one shape. ``score(chi2, dof, data)``
    maps the blocks' chi-squared on the quoted errors to ``-sum_b (nu_b/2) ln
    chi2_b`` or, with ``s_max``, to each block's likelihood integrated over ln s
    in ``[-ln s_max, ln s_max]`` (exact von Mises normalization for uncorrelated
    closure phases). ``dof`` (traced) tempers each block (nu_eff = dof nu), one
    number or one 1/f per block (``virgil.tempering``).

    Data with gains or closure-phase offsets have covariance C(s) = s^2 D + U U^T,
    which a scale does not factor, so ln s is integrated numerically (``numeric``):
    a Gauss-Legendre rule over +-9 standard deviations of the integrand around its
    Newton peak, whitened exactly at each s. Such data without ``s_max`` take 10.
    """

    floor = (
        False  # True on the ``inflated`` twin: s is log-uniform on [1, s_max]
    )

    def __init__(self, data, s_max=None):
        # Extra observables with a model-dependent covariance cannot be
        # scored (gains and closure offsets are integrated numerically).
        if any(b.model_dependent_covariance for b in data.extras):
            raise NotImplementedError(
                "The scale-marginalized surface (scales='marginal', "
                "epoch_positions, marginal_loglike) cannot score extra "
                "observables with a model-dependent covariance. Use the "
                "data without them, or scales='quoted'."
            )
        if s_max is not None and not float(s_max) > 1.0:
            raise ValueError(f"s_max must exceed 1, got {s_max}.")
        self.numeric = bool(
            data.gains is not None or data.phase_offsets is not None
        )
        if self.numeric and s_max is None:
            s_max = _NUMERIC_S_MAX
        # Observables per block of errors: the visibilities, the phases and
        # each extra block (their errors, not their residual rows).
        self.sizes = (
            int(onp.asarray(data.vis).size),
            int(onp.asarray(data.phi).size),
        ) + tuple(int(b.data().size) for b in data.extras)
        self.blocks = _blocks(data)
        self.names = tuple(b[0] for b in self.blocks)
        self.nu = onp.array([b[3] for b in self.blocks], dtype=float)
        self.s_max = None if s_max is None else float(s_max)

    def _key(self):
        return (self.blocks, self.s_max, self.numeric, self.sizes, self.floor)

    @functools.cached_property
    def inflated(self):
        """The surface scoring the same data with the noise only inflated:
        ln s uniform on ``[0, ln s_max]``, not ``[-ln s_max, ln s_max]``,
        on the same prior normalization (each block adds ln 2). Needs
        ``s_max``."""
        if self.s_max is None:
            raise ValueError("An inflated-noise surface needs s_max.")
        twin = copy.copy(self)
        twin.floor, twin.__dict__["inflated"] = True, twin
        return twin

    @property
    def _low(self):
        """The lower end of the ln s integral."""
        return 0.0 if self.floor else -float(onp.log(self.s_max))

    def __hash__(self):
        return hash(self._key())

    def __eq__(self, other):
        return isinstance(other, _Surface) and self._key() == other._key()

    def chi2(self, model, data, noise=None, dof=None):
        """χ² of each block of ``data`` for ``model``, on the quoted errors
        (for a ``numeric`` surface, the table ``score`` integrates; its
        first column is that χ², see ``plain_chi2``). ``dof`` (as for
        ``score``) only sets the width of that table's window."""
        if self.numeric:
            return self._table(model, data, noise or {}, dof)
        r = whitened_residuals(model, data, **(noise or {}))
        return self.block_chi2(r)

    @staticmethod
    def plain_chi2(chi2):
        """The blocks' χ² at s = 1 of what ``chi2`` returned."""
        return chi2[..., 0] if chi2.ndim > 1 else chi2

    def _table(self, model, data, noise, dof=None):  # deliberate: 60 lines
        """Per block ``[χ²(1) | ln s | χ² | log norm | log weight]`` (the
        last four at ``_N_NODES`` nodes), by whitening the data with the
        errors times each node's s."""
        prediction, errors, values, gain_terms, offset_terms = (
            _prediction_and_errors(model, data, noise)
        )
        n_block = len(self.blocks)
        bound, low = float(onp.log(self.s_max)), self._low
        at = self._at(
            prediction, errors, values, gain_terms, offset_terms, data
        )

        def log_f(log_s):
            """Each block's log integrand at its own ``log_s`` (blocks are
            independent, so a sum has their derivatives as its gradient)."""
            chi2, own = at(log_s)
            return np.stack(
                [
                    self._log_f(k, log_s[k : k + 1], chi2[k], own[k], data)[0]
                    for k in range(n_block)
                ]
            )

        nu = np.asarray(self.nu)
        chi2_one = at(np.zeros(n_block))[0]
        # The peak: a fixed-point start (χ²(s) = ν), then Newton steps on
        # the exact log integrand (its curvature is about -2(ν - k) with
        # k the gains' rank, so the χ² = ν point sits k/2ν off the peak).
        centre = np.zeros(n_block)
        for _ in range(2):
            chi2_c = at(centre)[0]
            tiny = np.finfo(chi2_c.dtype).tiny
            centre = np.clip(
                centre + 0.5 * np.log(np.maximum(chi2_c, tiny) / nu),
                low,
                bound,
            )
        slope = jax.grad(lambda t: np.sum(log_f(t)))
        curve = jax.grad(lambda t: np.sum(slope(t)))
        for _ in range(_N_CENTRE):
            step = slope(centre) / np.maximum(-curve(centre), 0.1 * nu)
            centre = np.clip(centre + np.clip(step, -1.0, 1.0), low, bound)
        curvature = np.maximum(-curve(centre), 0.1 * nu)
        centre, curvature = jax.lax.stop_gradient((centre, curvature))
        dof = np.broadcast_to(
            np.asarray(1.0 if dof is None else dof, chi2_one.dtype),
            (n_block,),
        )
        half = _WINDOW / np.sqrt(curvature * jax.lax.stop_gradient(dof))
        lo = np.maximum(centre - half, low)
        hi = np.minimum(centre + half, bound)
        x, w = gauss_legendre(_N_NODES, chi2_one.dtype)
        mid, rad = 0.5 * (hi + lo), 0.5 * (hi - lo)
        nodes = mid[:, None] + rad[:, None] * x[None, :]
        chi2_n, norm_n = jax.vmap(at, in_axes=1, out_axes=1)(nodes)
        log_w = np.log(w)[None, :] + np.log(rad)[:, None]
        return np.concatenate(
            [chi2_one[:, None], nodes, chi2_n, norm_n, log_w], axis=1
        )

    def _at(self, prediction, errors, values, gain_terms, offset_terms, data):
        """``at(log_s)``: each block's χ² and gains/offsets log normalizer
        with the errors times ``exp(log_s)`` (one whitening, exact log det)."""
        sizes = onp.asarray(self.sizes)
        # The block of each entry of ``sizes`` (an absent, empty block has
        # size 0: any index will do).
        index = {n: k for k, n in enumerate(self.names)}
        labels = ["vis", "phi"] + [
            f"extra[{k}]" for k in range(len(self.sizes) - 2)
        ]
        block_of = onp.array([index.get(n, 0) for n in labels])

        def at(log_s):
            scale = np.repeat(
                np.exp(log_s[block_of]),
                sizes,
                total_repeat_length=int(sizes.sum()),
            )
            w, _, norm = _whiten_with_log_norm(
                data,
                prediction,
                values,
                errors * scale,
                gain_terms,
                offset_terms,
                split_norm=True,
            )
            chi2 = self.block_chi2(w)
            # The log normalizer [gains', offsets'] adds to each block.
            own = np.stack(
                [
                    norm[0]
                    if name == "vis"
                    else norm[1]
                    if name == "phi"
                    else np.zeros((), norm.dtype)
                    for name in self.names
                ]
            )
            return chi2, own

        return at

    def fixed_score(self, model, data, noise, dof, log_s):
        """The tempered log integrand of a ``numeric`` surface summed over
        blocks at the fixed per-block ``log_s``: one whitening, with the exact
        log determinant, in place of the table's integral over ln s (a cheap
        score of the same surface, off from it by a smooth log-width)."""
        prediction, errors, values, gain_terms, offset_terms = (
            _prediction_and_errors(model, data, noise or {})
        )
        chi2, own = self._at(
            prediction, errors, values, gain_terms, offset_terms, data
        )(log_s)
        dof = np.broadcast_to(
            np.asarray(1.0 if dof is None else dof, chi2.dtype), chi2.shape
        )
        return sum(
            dof[k] * self._log_f(k, log_s[k : k + 1], chi2[k], own[k], data)[0]
            for k in range(len(self.blocks))
        )

    def _log_f(self, k, log_s, chi2, norm, data):
        """Block k's log integrand (untempered) at ``log_s``, given its χ²
        and gains/offsets log normalizer there."""
        _, _, _, nu, von_mises = self.blocks[k]
        if not von_mises:
            return -0.5 * chi2 - nu * log_s - norm
        # von Mises, as in ``score`` (never with gains or offsets)
        sigma = np.asarray(data.d_phi, log_s.dtype).reshape(-1)
        kappa = 1.0 / (np.exp(2.0 * log_s)[:, None] * sigma**2)
        return (
            -0.5 * chi2
            - np.sum(np.log(np.sqrt(2.0 * np.pi) * i0e(kappa)), axis=1)
            + np.sum(np.log(sigma))
        )

    def block_chi2(self, r):
        """χ² of each block of the whitened residuals ``r``."""
        return np.stack([np.sum(r[a:b] ** 2) for _, a, b, _, _ in self.blocks])

    def dof_vector(self, dof):
        """``dof`` as the per-block vector ``score`` takes.

        Parameters
        ----------
        dof : float, mapping, sequence of (str, float) or None
            A number (every block), a mapping or sequence of
            ``(name, dof)`` pairs keyed by block name (``"vis"``,
            ``"phi"``, ``"extra[k]"``; a missing block is 1), or None (1).

        Returns
        -------
        float or array
            A plain number stays a number; otherwise one value per block
            of this surface, in its order.
        """
        if dof is None:
            return 1.0
        if not isinstance(dof, (dict, tuple, list)):
            return float(dof)
        if len(dof) == 0:
            return 1.0
        table = dict(dof)
        return onp.array([float(table.get(n, 1.0)) for n in self.names])

    def score(self, chi2, dof, data):
        if self.numeric:  # ``chi2`` is the table
            n = _N_NODES
            table = chi2
            dof = np.broadcast_to(np.asarray(dof, table.dtype), self.nu.shape)
            total = 0.0
            for k in range(len(self.blocks)):
                log_s = table[k, 1 : 1 + n]
                chi2_k = table[k, 1 + n : 1 + 2 * n]
                norm = table[k, 1 + 2 * n : 1 + 3 * n]
                log_w = table[k, 1 + 3 * n :]
                log_f = self._log_f(k, log_s, chi2_k, norm, data)
                total = total + jax.nn.logsumexp(dof[k] * log_f + log_w)
            return total + self.floor * len(self.blocks) * _LN2
        dof = np.broadcast_to(np.asarray(dof, chi2.dtype), chi2.shape[-1:])
        if self.s_max is None:
            tiny = np.finfo(chi2.dtype).tiny
            nu = np.asarray(self.nu, chi2.dtype)
            return -0.5 * np.sum(dof * nu * np.log(np.maximum(chi2, tiny)))
        bound = onp.log(self.s_max)
        total = 0.0
        for k, (_, _, _, nu, von_mises) in enumerate(self.blocks):

            def log_f(log_s, k=k, nu=nu, von_mises=von_mises):
                loglike = -0.5 * chi2[k] * np.exp(-2.0 * log_s)
                if not von_mises:
                    return dof[k] * (loglike - nu * log_s)
                # von Mises: -log(2π I0(κ)) + κ per phase, κ = 1/(s σ)²,
                # relative to the Gaussian's -log(√(2π) σ), as in the other
                # blocks: -log(√(2π) i0e(κ)) + log σ, which is -log s for
                # κ ≫ 1.
                sigma = np.asarray(data.d_phi, chi2.dtype).reshape(-1)
                kappa = 1.0 / (np.exp(2.0 * log_s)[:, None] * sigma**2)
                loglike = (
                    loglike
                    - np.sum(np.log(np.sqrt(2.0 * np.pi) * i0e(kappa)), axis=1)
                    + np.sum(np.log(sigma))
                )
                return dof[k] * loglike

            # The Gaussian block's peak, ln ŝ = ln √(χ²/ν).
            tiny = np.finfo(chi2.dtype).tiny
            guess = 0.5 * np.log(np.maximum(chi2[k], tiny) / nu)
            total = total + log_integral(log_f, self._low, bound, guess)
        return total + self.floor * len(self.blocks) * _LN2

    def weights(self, chi2, dof, data):
        """The IRLS weight of each block, ``w_b = 2 ∂(-score)/∂χ²_b``.

        Near a point, the score falls as ``Σ_b w_b χ²_b / 2``, so a
        Gauss–Newton step on the blocks' residuals multiplied by √w_b has
        the curvature of the score. ``w_b ≥ 0`` because the score falls
        as any χ²_b rises. Without ``s_max`` it is ``dof_b ν_b/χ²_b``,
        which is ``dof_b/ŝ_b²`` with ŝ_b the block's profile scale. ``dof``
        is one number or one per block.
        """
        if self.numeric:
            # The IRLS weight of the unbounded profile on the gain-whitened
            # χ² at s = 1: an approximate curvature for the Gauss-Newton
            # step, whose loss is the exact numeric score.
            chi2 = self.plain_chi2(chi2)
            dof = np.broadcast_to(np.asarray(dof, chi2.dtype), chi2.shape)
            tiny = np.finfo(chi2.dtype).tiny
            return (
                dof * np.asarray(self.nu, chi2.dtype) / np.maximum(chi2, tiny)
            )
        return 2.0 * jax.grad(lambda c: -self.score(c, dof, data))(chi2)

    def weighted_rows(self, r, dof, data):
        """The whitened residuals ``r`` with each block multiplied by
        √(stop_gradient ``w_b``): curvature rows of the score (see
        [`weights`][virgil.epochs._Surface.weights]). The weights are
        frozen, so ``½ |rows|²`` has the score's Gauss–Newton curvature."""
        chi2 = self.block_chi2(r)
        w = jax.lax.stop_gradient(self.weights(chi2, dof, data))
        return np.concatenate(
            [
                np.sqrt(w[k]) * r[a:b]
                for k, (_, a, b, _, _) in enumerate(self.blocks)
            ]
        )

    def scales(self, chi2):
        """ŝ_b = √(χ²_b/ν_b), the profile maximum of each block's scale."""
        return onp.sqrt(onp.asarray(chi2, float) / self.nu)


def marginal_loglike(
    model, data, *, dof=1.0, block_dof=None, s_max=None, **noise
):
    """The log likelihood with each block's error scale marginalized.

    Integrating each block's scale out under its Jeffreys prior gives, up to a
    constant, ``m = -sum_b (nu_b/2) ln chi2_b``. It is a search surface, not a
    replacement for [`model_loglike`][virgil.likelihood.model_loglike]
    ([derivation](../orbit_fitting.md#scale-marginalized-scoring-and-night-flags)).

    Parameters
    ----------
    model : SourceModel
        Model to evaluate.
    data : OIData
        Data to compare with. Gains, closure-phase offsets and extra
        observables with a model-dependent covariance raise
        ``NotImplementedError``: a scale does not multiply their covariance.
    dof : float, optional
        Effective degrees of freedom as a fraction of ``nu_b`` (default 1).
    block_dof : dict, optional
        ``{block name: 1/f_b}`` (``"vis"``, ``"phi"``, ``"extra[k]"``), as
        ``AliasResult.block_dof()`` gives.
    s_max : float, optional
        Bound each scale to ``[1/s_max, s_max]`` and integrate numerically, with
        the exact von Mises normalization for uncorrelated closure phases; needed
        for weak closure phases, where the unbounded marginal is improper.
    **noise
        Other noise terms; not ``vis_scale``/``phi_scale``.

    Returns
    -------
    float
    """
    surface = _Surface(data, s_max)
    dof_all = np.asarray(surface.dof_vector(block_dof)) * _check_dof(dof)
    return surface.score(
        surface.chi2(model, data, noise, dof_all),
        dof_all,
        data,
    )


_SCALES = ("marginal", "quoted")


def _check_scales(scales):
    if scales not in _SCALES:
        raise ValueError(f"scales must be one of {_SCALES}, not {scales!r}.")
    return scales


def _orbit_batch_size(
    batch_size, datasets, n, per_candidate=1
):  # deliberate: imported by orbit_search.py; unit-tested
    """Candidates per vectorized batch of a ``lax.map`` over ``n`` of them.

    ``lax.map`` without a ``batch_size`` is a serial loop, several times
    slower than batches (3-5x for thousands of candidates on a laptop CPU).
    By default a batch holds about ``_grid.batch_visibilities()`` model
    visibilities in all: each candidate evaluates every dataset, so it
    costs the datasets' samples summed, times ``per_candidate`` for work
    that multiplies that (a flux grid). One candidate per batch at the
    least, never more than ``n``. The results do not depend on it.
    """
    if batch_size is not None:
        return min(batch_size_or_default(batch_size, None), max(int(n), 1))
    samples = sum(max(onp.size(d.u), onp.size(d.wavel), 1) for d in datasets)
    fit = batch_visibilities() // max(samples * per_candidate, 1)
    return min(max(fit, 1), max(int(n), 1))


class _Static:
    """A hashable bundle for a static jit argument: equal when ``key`` is.

    The ``payload`` is read at trace time, so ``key`` must determine
    everything in it that the compiled kernel bakes in.
    """

    def __init__(self, key, payload):
        self.key = key
        self.payload = payload

    def __hash__(self):
        return hash(self.key)

    def __eq__(self, other):
        return isinstance(other, _Static) and self.key == other.key


class _Kwargs:
    """``v -> model(**v)``, equal for equal models (a stable cache key)."""

    def __init__(self, model):
        self.model = model

    def __call__(self, v):
        return self.model(**v)

    def __hash__(self):
        return hash(_ident(self.model))

    def __eq__(self, other):
        return isinstance(other, _Kwargs) and _ident(self.model) == _ident(
            other.model
        )


def _ident(x):
    """``x`` if hashable, else its id (kept alive by the payload)."""
    try:
        hash(x)
    except TypeError:
        return ("id", id(x))
    return x


@functools.partial(jax.jit, static_argnames=("static",))
def _loglike_kernel(
    static, inputs, datasets, dofs
):  # deliberate: jit boundary, compile-once
    """One compile per layout: the data and the dofs are traced."""
    p = static.payload

    def one(x):
        snapshots = Epochs.snapshots(p.epochs, p.scene_of(x))
        if p.scales == "marginal":
            total = sum(
                f.score(f.chi2(s, d, n, dofs[k]), dofs[k], d)
                for k, (s, d, n, f) in enumerate(
                    zip(snapshots, datasets, p.noise, p.surfaces)
                )
            )
        else:
            total = sum(
                model_loglike(s, d, **n)
                for s, d, n in zip(snapshots, datasets, p.noise)
            )
        return np.where(np.isfinite(total), total, -np.inf)

    return jax.lax.map(one, inputs, batch_size=p.batch_size)


def _loglikes(
    scene_of,
    inputs,
    data,
    noise,
    batch_size,
    dtype,
    scales="quoted",
    dof=1.0,
    s_max=None,
):
    """``data``'s log likelihood of ``scene_of(x)`` for each ``x`` in the
    stacked ``inputs``, as a NumPy array (``-inf`` where not finite).

    With ``scales="marginal"`` it is the scale-marginalized score
    ``m = -Σ (ν/2) ln χ²`` over every block of every dataset (see
    [`marginal_loglike`][virgil.epochs.marginal_loglike])."""
    if not isinstance(data, Epochs):
        raise TypeError(
            f"data must be an Epochs, not {type(data).__name__}: rankings "
            "take one snapshot per dataset."
        )
    per_dataset = data.noise(noise or {})
    batch_size = _orbit_batch_size(
        batch_size, data.data, jax.tree_util.tree_leaves(inputs)[0].shape[0]
    )
    surfaces = ()
    dofs = np.ones(len(data))
    if scales == "marginal":
        surfaces = tuple(_Surface(d, s_max) for d in data.data)
        dofs = np.asarray(
            [_check_dof(_dof_for(dof, data, k)) for k in range(len(data))]
        )
    times = tuple(float(t) for t in data.times)
    names = tuple(
        tuple(sorted((k, _ident(v)) for k, v in n.items()))
        for n in per_dataset
    )
    key = (
        _ident(scene_of), names, int(batch_size), times, scales, s_max,
        tuple(sf._key() for sf in surfaces),
    )  # fmt: skip
    static = _Static(
        key,
        types.SimpleNamespace(
            scene_of=scene_of,
            epochs=types.SimpleNamespace(
                data=(None,) * len(times), times=times
            ),
            noise=tuple(per_dataset),
            surfaces=tuple(surfaces),
            scales=scales,
            batch_size=int(batch_size),
        ),
    )
    with run_in(dtype):
        inputs = cast_tree(inputs, dtype)
        datasets = cast_tree(tuple(data.data), dtype)
        out = _loglike_kernel(
            static, inputs, datasets, np.asarray(dofs, dtype)
        )
        return onp.asarray(out, dtype=float)


@dataclasses.dataclass(frozen=True)
class RankedOrbits:
    """Orbits ranked by the log likelihood of multi-epoch data, best first.

    Returned by [`rank_orbits`][virgil.epochs.rank_orbits] and
    [`chain_starts`][virgil.epochs.chain_starts]. Iterating gives
    ``(orbit, loglike)`` pairs, like the ``(orbit, χ²)`` pairs of
    [`starting_orbits`][virgil.orbits.starting_orbits], and
    ``ranked[k]`` is the ``k``-th pair.

    Attributes
    ----------
    orbits : tuple
        The orbits, best first.
    loglike : numpy.ndarray
        The log likelihood of all the data for each orbit (with the
        ``noise`` values of the ranking), or with ``scales="marginal"``
        the scale-marginalized score, in the same order.
    order : numpy.ndarray
        The position of each orbit in the list that was ranked.
    times : numpy.ndarray
        The snapshot times of the datasets (MJD), at which
        :meth:`positions` and ``chain_starts`` compare orbits.
    resolution_mas : float
        The finest resolution λ/B_max of the data (mas).
    """

    orbits: tuple
    loglike: onp.ndarray
    order: onp.ndarray
    times: onp.ndarray
    resolution_mas: float

    def __len__(self):
        return len(self.orbits)

    def __getitem__(self, k):
        return self.orbits[k], float(self.loglike[k])

    def __iter__(self):
        return iter(zip(self.orbits, (float(x) for x in self.loglike)))

    @property
    def best(self):
        """The orbit with the highest log likelihood."""
        return self.orbits[0]

    def positions(self):
        """The companion's position ``(dra, ddec)`` (mas) of each orbit at
        each snapshot time, shape ``(n_orbits, n_times, 2)``."""
        with run_in("float64"):
            stacked = cast_tree(_stack(self.orbits), "float64")
            dra, ddec, _ = jax.vmap(lambda o: o.relative(self.times))(stacked)
            return onp.stack([onp.asarray(dra), onp.asarray(ddec)], -1)


def _ranked(orbits, loglike, data):
    # The same deterministic order as score_orbits(...).order(). Imported
    # here because orbit_search imports this module.
    from .orbit_search import rank_scores

    order = rank_scores(loglike)
    return RankedOrbits(
        orbits=tuple(orbits[k] for k in order),
        loglike=loglike[order],
        order=order,
        times=onp.asarray(data.times, dtype=onp.float64),
        resolution_mas=data.resolution_mas,
    )


def rank_orbits(
    model,
    data,
    orbits,
    *,
    scales="marginal",
    noise=None,
    dof=1.0,
    s_max=None,  # deliberate: no flux is profiled, so no bound is needed
    batch_size=None,
    dtype="float64",
):
    """Rank trial orbits by the log likelihood of multi-epoch data.

    Each orbit's scene is evaluated at every dataset's snapshot time, in one
    compiled kernel mapped over the orbits.

    Parameters
    ----------
    model : callable
        ``model(orbit)`` returns the scene with the flux fixed, e.g.
        ``lambda orbit: OrbitalBinary(orbit, 0.1)``. It is traced: no Python
        floats from the orbit.
    data : Epochs
        The data, one snapshot per dataset.
    orbits : sequence or orbit
        Orbits, ``starting_orbits`` pairs, a ``RankedOrbits``, or one orbit with
        a leading axis; one class and ``t_ref``.
    scales : {"marginal", "quoted"}, optional
        Marginalize the error scales ([`marginal_loglike`][virgil.epochs.marginal_loglike])
        or score ``-χ²/2`` on the quoted errors.
    noise : dict, optional
        Noise values by epoch or dataset name ([conventions](../conventions.md)).
    dof, s_max : optional
        As for ``marginal_loglike``; ``dof`` may be a dict by dataset or epoch
        name. ``s_max`` defaults to None (no bound; 10 for data needing numeric
        integration).
    batch_size : int, optional
        Orbits evaluated together (default about a million model visibilities).
    dtype : {"float64", "float32"}, optional
        Precision.

    Returns
    -------
    RankedOrbits
        Best first; a non-finite one ranks last.
    """
    scales = _check_scales(scales)
    listed = _orbit_list(orbits)
    loglike = _loglikes(
        model,
        _stack(listed),
        data,
        noise,
        batch_size,
        dtype,
        scales=scales,
        dof=dof,
        s_max=s_max,
    )
    return _ranked(listed, loglike, data)


def _distinct_tracks(
    tracks, min_distance_mas, n=None
):  # deliberate: imported by aliases.py
    """Indices of up to ``n`` of ``tracks`` (each ``(times, 2)``, best
    first) that differ from every better choice by more than
    ``min_distance_mas`` at some time."""
    chosen = []
    for k, track in enumerate(tracks):
        if all(
            onp.max(onp.hypot(*(track - tracks[j]).T)) > min_distance_mas
            for j in chosen
        ):
            chosen.append(k)
            if len(chosen) == n:
                break
    return chosen


def _distinct(ranked, n, min_distance_mas):
    """Indices of up to ``n`` orbits of ``ranked``, best first, whose
    positions at the snapshot times differ from every better choice by
    more than ``min_distance_mas`` at some time."""
    finite = int(onp.sum(onp.isfinite(ranked.loglike)))
    if finite == 0:
        return []
    return _distinct_tracks(ranked.positions()[:finite], min_distance_mas, n)


def chain_starts(ranked, n_chains, *, min_distance_mas=None):
    """The best distinct orbits of a ranking, one per chain.

    Two orbits are one mode when the companion's positions at every snapshot
    time agree within ``min_distance_mas``: the mirror orbit (Ω + 180°,
    ω + 180°) is the same mode, a fringe or period alias is not.

    Parameters
    ----------
    ranked : RankedOrbits
        From [`rank_orbits`][virgil.epochs.rank_orbits].
    n_chains : int
        Number of starts.
    min_distance_mas : float, optional
        Largest position difference (mas) for one mode; default half the
        finest resolution λ/B_max.

    Returns
    -------
    RankedOrbits
        ``n_chains`` orbits, best first; with fewer distinct orbits than
        chains they are repeated in turn. Map each orbit to your model's
        parameters (e.g. ``period``, ``dt_peri``) to start a chain there.
    """
    if n_chains < 1:
        raise ValueError(f"n_chains must be at least 1, got {n_chains}.")
    if min_distance_mas is None:
        min_distance_mas = 0.5 * ranked.resolution_mas
    chosen = _distinct(ranked, n_chains, min_distance_mas)
    if not chosen:
        raise ValueError("No orbit has a finite log likelihood.")
    index = onp.asarray([chosen[k % len(chosen)] for k in range(n_chains)])
    return dataclasses.replace(
        ranked,
        orbits=tuple(ranked.orbits[k] for k in index),
        loglike=ranked.loglike[index],
        order=ranked.order[index],
    )


@dataclasses.dataclass(frozen=True)
class EpochPeaks:
    """The catalogue of one dataset's peaks, from
    [`epoch_positions`][virgil.epochs.epoch_positions], best first.

    Attributes
    ----------
    dra, ddec, flux : numpy.ndarray
        Peak positions (mas, East and North) and fluxes, after refinement.
    loglike : numpy.ndarray
        Log likelihood of each peak on the quoted errors.
    marginal : numpy.ndarray
        Scale-marginalized score of each peak (the sort key).
    weight : numpy.ndarray
        Posterior mass of each peak in this dataset (sums to 1; Laplace
        approximation, or the height exp(m) where ``laplace`` is false).
    edge : numpy.ndarray
        Peak started on the grid edge: a best peak there may stand for a
        companion outside the grid, so treat the dataset as ambiguous.
    laplace : numpy.ndarray
        Weight is a Laplace mass (false without ``refine`` or where the
        Hessian is not positive definite: a peak on a bound, or a flat direction).
    """

    dra: onp.ndarray
    ddec: onp.ndarray
    flux: onp.ndarray
    loglike: onp.ndarray
    marginal: onp.ndarray
    weight: onp.ndarray
    edge: onp.ndarray
    laplace: onp.ndarray

    def __len__(self):
        return len(self.marginal)


@dataclasses.dataclass(frozen=True)
class NightFlag:
    """Why one night's position should not seed an orbit.

    Attributes
    ----------
    code : str
        A short code: ``"alias-peaks"``, ``"low-gap"``,
        ``"sep-below-resolution"``, ``"flux-at-bound"``.
    reason : str
        A human-readable reason with the numbers behind it.
    blocking : bool
        Whether the flag keeps the night from seeding an orbit (the default).
        A non-blocking flag is reported only: the night still seeds and, as
        always, counts in the likelihood. ``sep-below-resolution`` is
        non-blocking unless ``epoch_positions(flag_sep_blocks=True)``.
    """

    code: str
    reason: str
    blocking: bool = True


@dataclasses.dataclass(frozen=True)
class EpochPositions:
    """The companion's position in each dataset, from
    [`epoch_positions`][virgil.epochs.epoch_positions].

    Positions, covariances and ``gap_marginal`` come from the marginal
    surface, so scaling the quoted errors leaves them unchanged.

    Attributes
    ----------
    names : tuple of str
        The datasets.
    mjd : numpy.ndarray
        Snapshot time of each dataset (MJD).
    dra, ddec, cov : numpy.ndarray
        Positions (mas, East and North) and their covariance, ``(n, 2, 2)`` (mas^2).
    flux : numpy.ndarray
        Fitted companion/primary flux (at most 1).
    gap_marginal : numpy.ndarray
        Best peak's score minus its best rival's more than ``gap_mas`` away;
        small means an alias or mirror image fits almost as well.
    chi2_raw, scale : tuple of dict
        Per dataset, χ²/N on the quoted errors per block and ``"all"`` (well
        above 1: errors too small), and each block's fitted error scale.
    peaks : tuple of EpochPeaks
        Per dataset, the catalogue of its peaks.
    flags : tuple of tuple of NightFlag
        Per dataset, why its position should not seed an orbit (see ``flagged``).
    primary_diam : float or None
        The uniform-disk primary the positions were fitted with (mas).
    grid : optional
        The search grid's axes; None if not from a grid.
    exact_fraction : tuple of float or None
        Fraction of grid points scored with the full scale integral (1 unless
        the coarse pass skipped some; see ``epoch_positions``).
    """

    names: tuple
    mjd: onp.ndarray
    dra: onp.ndarray
    ddec: onp.ndarray
    cov: onp.ndarray
    flux: onp.ndarray
    gap_marginal: onp.ndarray
    chi2_raw: tuple
    scale: tuple
    peaks: tuple = ()
    flags: tuple = ()
    primary_diam: float | None = None
    grid: tuple | None = None
    exact_fraction: tuple | None = None

    @property
    def flagged(self):
        """Whether each dataset carries a blocking flag (``flags``): a
        flagged night seeds nothing in
        [`fit_orbit_aliases`][virgil.aliases.fit_orbit_aliases], but its
        data still count in the likelihood. Non-blocking flags (see
        ``NightFlag.blocking``) are reported by ``flag_codes`` only."""
        if not self.flags:
            return onp.zeros(len(self.names), dtype=bool)
        return onp.array(
            [any(x.blocking for x in f) for f in self.flags], dtype=bool
        )

    def flag_codes(self):
        """The flag codes of each dataset, ``{name: [code, ...]}``, for
        the datasets with any flag, blocking or not."""
        return {
            n: [f.code for f in fl]
            for n, fl in zip(self.names, self.flags)
            if fl
        }

    def decisive(self, min_gap):
        """Whether each dataset's ``gap_marginal`` exceeds ``min_gap``."""
        return self.gap_marginal > min_gap

    @property
    def edge(self):
        """Whether each dataset's best peak started at the edge of the
        position grid (see ``EpochPeaks.edge``)."""
        return onp.array([bool(p.edge[0]) for p in self.peaks], dtype=bool)

    def positions(self, *, t_ref, min_gap=None, skip_flagged=True):
        """The positions as [`PositionData`][virgil.orbits.PositionData],
        for [`starting_orbits`][virgil.orbits.starting_orbits].

        Parameters
        ----------
        t_ref : float
            The reference time (MJD) of the orbits fitted to them: give
            the ``t_ref`` of your model.
        min_gap : float, optional
            Keep only the datasets whose ``gap_marginal`` exceeds this.
            Seeding orbits with an indecisive dataset's position can start
            every orbit at the wrong peak; the visibility fit judges those
            datasets instead.
        skip_flagged : bool, optional
            Drop the flagged datasets (``flagged``; default True, so that
            the flags cannot be ignored silently): the same idea as
            ``min_gap`` with several reasons. ``min_gap`` and the
            ``flag_min_gap`` of ``epoch_positions`` are the same
            threshold on ``gap_marginal``. Pass False to keep every
            dataset.
        """
        keep = (
            onp.ones(len(self.names), bool)
            if min_gap is None
            else self.decisive(min_gap)
        )
        if skip_flagged:
            keep = keep & ~self.flagged
        if not keep.any():
            raise ValueError(
                "No dataset is decisive and unflagged: the marginal gaps are "
                f"{self.gap_marginal.round(1)}, min_gap is {min_gap} and "
                f"the flagged datasets are {self.flag_codes()}."
            )
        return PositionData(
            self.mjd[keep],
            self.dra[keep],
            self.ddec[keep],
            self.cov[keep],
            t_ref=t_ref,
        )


def _binary(x, diam=None):
    """The static binary ``x = (dra, ddec, flux)``: two point stars, or,
    when ``diam`` (mas, traced) is given, a point companion beside a
    uniform-disk primary. ``diam=None`` keeps the fast point-star path
    (``None`` is an empty pytree, so it never triggers a recompilation)."""
    if diam is None:
        return BinaryModelCartesian(x[0], x[1], x[2])
    # Resolved primaries (orbit_solved_problems.md, "Resolved primary" row): a point
    # primary biases the per-night seeds.
    return System(
        primary=UniformDisk(diam),
        companion=PointSource(x[2], x[0], x[1]),
    )


@functools.partial(jax.jit, static_argnames=("surface", "batch_size"))
def _grid_scores(points, data, surface, batch_size, diam=None, dof=None):
    """The quoted-error log likelihood and the blocks' χ² at each point."""

    def one(x):
        whitened, errors = _whitened_and_errors(_binary(x, diam), data, {})
        if surface.numeric:
            chi2 = surface.chi2(_binary(x, diam), data, None, dof)
        else:
            chi2 = surface.block_chi2(whitened)
        return _gaussian_loglike(whitened, errors), chi2

    return jax.lax.map(one, points, batch_size=batch_size)


@functools.partial(jax.jit, static_argnames="surface")
def _grid_marginal(chi2, dof, data, surface):
    """The scale-marginalized score of each grid point's block χ²."""
    return jax.vmap(lambda c: surface.score(c, dof, data))(chi2)


# The memory a grid point needs scales with what one whitening holds: the
# observables and, with gains or offsets, their modes (rank). A table whitens
# at ``_N_NODES`` nodes at once, and the window search differentiates through
# whitenings: that working set is about _TABLE_COPIES times one whitening's
# (measured on calibrated GRAVITY data: 325 MB per point, 34k floats; a 1190-point batch ran out of
# memory on a laptop); a fixed-scale score is one whitening with its gradient
# free, about _FIXED_COPIES.
_TABLE_COPIES = 1200
_FIXED_COPIES = 8
CPU_MEMORY_BUDGET = 2**30
ACCELERATOR_MEMORY_BUDGET = 2**33


def default_memory_budget():
    """Bytes one batch may use by default, for JAX's default backend."""
    if jax.default_backend() == "cpu":
        return CPU_MEMORY_BUDGET
    return ACCELERATOR_MEMORY_BUDGET


def _whitening_floats(data):
    """The floats one whitening of ``data`` holds: observables plus the
    gain and offset modes' columns."""
    n = int(onp.size(data.vis) + onp.size(data.phi))
    n += sum(int(b.data().size) for b in data.extras)
    for modes, names in (
        (data.gains, ("shapes", "spanning")),
        (data.phase_offsets, ("values",)),
    ):
        if modes is not None:
            n += sum(
                int(onp.size(getattr(modes, a)))
                for a in names
                if getattr(modes, a, None) is not None
            )
    return n


def _grid_batch(batch_size, memory_budget, data, surface, copies=None):
    """Grid points per batch: ``batch_size`` if given, else what fits in
    ``memory_budget`` bytes (default ``CPU_MEMORY_BUDGET``, or
    ``ACCELERATOR_MEMORY_BUDGET`` off the CPU) for data with gains or offsets,
    where each point holds ``copies`` whitenings of ``_whitening_floats``
    float64s; the visibility rule of ``batch_size_or_default`` otherwise."""
    if batch_size is not None:
        return batch_size_or_default(batch_size, None)
    default = batch_size_or_default(None, data)
    if not surface.numeric:
        return default
    if memory_budget is None:
        memory_budget = default_memory_budget()
    if not memory_budget > 0:
        raise ValueError(
            f"memory_budget must be positive, got {memory_budget}."
        )
    copies = _TABLE_COPIES if copies is None else copies
    per_point = 8 * copies * _whitening_floats(data)
    return int(max(1, min(default, memory_budget // per_point)))


@functools.partial(jax.jit, static_argnames=("surface", "batch_size"))
def _grid_fixed(points, data, surface, batch_size, log_s, diam=None, dof=None):
    """The tempered log integrand at each point at the fixed ``log_s``."""

    def one(x):
        return surface.fixed_score(_binary(x, diam), data, None, dof, log_s)

    return jax.lax.map(one, points, batch_size=batch_size)


def _coarse_then_exact(
    points,
    shape,
    data,
    surface,
    diam,
    dof,
    batch_size,
    memory_budget,
    nats,
    margin,
):
    """The scale-marginalized score of every grid point of a ``numeric``
    surface, integrating ln s only where the surface is high.

    All points are scored at one fixed scale per block (the profile scale of
    the best point at s = 1): one whitening each. The points within ``nats``
    of the best (in marginal nats: see below), and their ``margin``
    neighbours, get the full integral. The
    rest take their fixed-scale score plus the median difference (full minus
    fixed) over the points within ``nats``. Returns ``(score, fraction)``,
    the flat scores and the fraction of points scored in full.
    """
    from scipy.ndimage import binary_dilation

    fixed_batch = _grid_batch(
        None if batch_size is None else batch_size,
        memory_budget,
        data,
        surface,
        _FIXED_COPIES,
    )
    fixed_batch = min(fixed_batch, len(points))
    zero = np.zeros(len(surface.blocks))
    at_one = onp.asarray(
        _grid_fixed(points, data, surface, fixed_batch, zero, diam, dof)
    )
    at_one = onp.where(onp.isfinite(at_one), at_one, -onp.inf)
    best = points[int(onp.argmax(at_one))]
    chi2 = surface.plain_chi2(
        _block_chi2(np.asarray(best), data, surface, diam)
    )
    bound = float(onp.log(surface.s_max))
    log_s = np.asarray(
        onp.clip(
            0.5 * onp.log(onp.maximum(onp.asarray(chi2), 1e-300) / surface.nu),
            -bound,
            bound,
        )
    )
    coarse = onp.asarray(
        _grid_fixed(points, data, surface, fixed_batch, log_s, diam, dof)
    )
    coarse = onp.where(onp.isfinite(coarse), coarse, -onp.inf)
    # ``nats`` is a gap of the marginal. At the fixed scale a relative χ²
    # excess ε costs νε/2 nats, but the marginal only (ν/2) ln(1+ε): the
    # same marginal gap is a larger fixed-scale one, so cut there (ν the
    # total of the blocks', a few nats' margin for small data).
    nu = float(onp.sum(surface.nu))
    cut = 0.5 * nu * onp.expm1(min(2.0 * nats / nu, 50.0))
    core = coarse >= coarse.max() - cut
    pick = core.reshape(shape)
    if margin:
        pick = binary_dilation(
            pick, structure=onp.ones((3,) * pick.ndim), iterations=int(margin)
        )
    pick = pick.ravel()
    chosen = points[pick]
    batch = min(
        _grid_batch(batch_size, memory_budget, data, surface), len(chosen)
    )
    # Pad to a power-of-two number of batches (repeating the first row), so
    # that nights and grids of different sizes share a few compiled shapes.
    n_batches = 1 << max(math.ceil(len(chosen) / batch) - 1, 0).bit_length()
    padded = onp.concatenate(
        [chosen, onp.repeat(chosen[:1], n_batches * batch - len(chosen), 0)]
    )
    _, chi2 = _grid_scores(np.asarray(padded), data, surface, batch, diam, dof)
    chi2 = chi2[: len(chosen)]
    exact = onp.asarray(_grid_marginal(chi2, dof, data, surface))
    ok = onp.isfinite(exact) & core[pick]
    offset = float(onp.median((exact - coarse[pick])[ok])) if ok.any() else 0.0
    out = coarse + offset
    out[pick] = exact
    return out, float(pick.mean())


def _amp_profile(r, data, n, amp_range):
    """``(A, r')``: the whitened residuals ``r`` with the first ``n`` (the
    squared visibilities) recomputed for the model scaled by the profiled
    V² amplitude ``A``, the weighted least-squares amplitude clipped to
    ``amp_range`` (lo, hi). The model is recovered from the residual as
    ``obs + r * err``; ``r'`` is ``(A * model - obs) / err``. Shared by the
    orbit fit's per-epoch amplitude (``aliases._epoch_residuals``) and the
    night fits."""
    obs = np.reshape(data.vis, -1)
    err = np.reshape(data.d_vis, -1)
    mod = obs + r[:n] * err
    w = 1.0 / err**2
    tiny = np.finfo(r.dtype).tiny
    amp = np.sum(w * obs * mod) / np.maximum(np.sum(w * mod**2), tiny)
    amp = np.clip(amp, *amp_range)
    return amp, r.at[:n].set((amp * mod - obs) / err)


def _amp_residuals(model, data, surface, amp_range):
    """The whitened residuals of ``model``, with the squared visibilities
    multiplied by one profiled amplitude ``A`` (a per-night contrast loss)
    when ``amp_range`` is given (see ``_amp_profile``). Without
    ``amp_range``, or without a visibility block, the plain residuals."""
    r = whitened_residuals(model, data)
    if amp_range is None or surface.names[0] != "vis":
        return r
    _refuse_amp_with_gains(data)
    return _amp_profile(r, data, surface.blocks[0][2], amp_range)[1]


def _negative_marginal(x, dof, data, surface, diam=None, amp_range=None):
    if amp_range is None:
        chi2 = surface.chi2(_binary(x, diam), data, None, dof)
    else:
        r = _amp_residuals(_binary(x, diam), data, surface, amp_range)
        chi2 = surface.block_chi2(r)
    return -surface.score(chi2, dof, data)


_negative_marginal_hessian = jax.jit(
    jax.hessian(_negative_marginal),
    static_argnames=("surface", "amp_range"),
)
_block_chi2 = jax.jit(
    lambda x, data, surface, diam=None: surface.chi2(_binary(x, diam), data),
    static_argnames="surface",
)


def _check_n_peaks(n_peaks):
    if (
        isinstance(n_peaks, bool)
        or not isinstance(n_peaks, (int, onp.integer))
        or n_peaks < 1
    ):
        raise ValueError(
            f"n_peaks must be an integer of at least 1, not {n_peaks!r}."
        )


def _grid_peaks(
    score, xx, yy, rival, n_peaks
):  # deliberate: unit-tested on synthetic grids; the caller is a 400-line loop
    """The ``(i, j, k)`` cells of up to ``n_peaks`` local maxima of a
    (dra, ddec, flux) grid of scores, highest first, each more than
    ``rival`` from every higher one (k is the best flux there)."""
    from scipy.ndimage import maximum_filter

    by_position = score.max(axis=2)
    finite = onp.isfinite(by_position)
    local = finite & (
        by_position == maximum_filter(by_position, size=3, mode="nearest")
    )
    cells = onp.argwhere(local)
    cells = cells[onp.argsort(-by_position[local], kind="stable")]
    if not len(cells):
        cells = [onp.unravel_index(onp.argmax(by_position), by_position.shape)]
    chosen = []
    for i, j in cells:
        if all(
            onp.hypot(xx[i, j, 0] - xx[a, b, 0], yy[i, j, 0] - yy[a, b, 0])
            > rival
            for a, b, _ in chosen
        ):
            chosen.append((int(i), int(j), int(onp.argmax(score[i, j]))))
            if len(chosen) == n_peaks:
                break
    return chosen


def _rank_peaks(
    grid_points, grid_marginal, refine, score, rival
):  # deliberate: unit-tested with stub refine/score; the caller is a 400-line loop
    """Refine each grid peak and rank the peaks by their marginal score.

    ``refine(x0)`` returns a refined point and its marginal score, kept
    only when it is no worse than the grid point's; ``score(points)``
    returns the quoted log likelihood and the marginal score of each
    point. Two grid peaks can refine to one maximum, so a peak within
    ``rival`` of a higher one is dropped. Returns the points, quoted and
    marginal scores, best first, and the order (indices into
    ``grid_points``).
    """
    points = onp.array(grid_points, dtype=float)
    if refine is not None:
        for n, (x0, m0) in enumerate(zip(grid_points, grid_marginal)):
            refined, value = refine(onp.asarray(x0, dtype=float))
            if value >= m0:
                points[n] = refined
    loglike, marginal = score(points)
    loglike = onp.asarray(loglike, dtype=float)
    marginal = onp.where(
        onp.isfinite(marginal), onp.asarray(marginal, dtype=float), -onp.inf
    )
    order = []
    for n in onp.argsort(-marginal, kind="stable"):
        if all(
            onp.hypot(*(points[n, :2] - points[k, :2])) > rival for k in order
        ):
            order.append(int(n))
    order = onp.array(order, dtype=int)
    return points[order], loglike[order], marginal[order], order


def _peak_weights(
    marginal, hessians
):  # deliberate: unit-tested on synthetic Hessians; no cheap public route
    """Normalized Laplace masses of the peaks, and whether each is one.

    log w_k = m_k - ½ log det H_k, with H_k the Hessian of -m at peak k
    over (dra, ddec, flux). A peak whose Hessian is missing (``None``)
    or not positive definite falls back to its height m_k.
    """
    log_w, laplace = [], []
    for m, hess in zip(marginal, hessians):
        hess = None if hess is None else onp.asarray(hess, dtype=float)
        positive = (
            hess is not None
            and bool(onp.all(onp.isfinite(hess)))
            and bool(onp.all(onp.linalg.eigvalsh(hess) > 0))
        )
        log_w.append(m - 0.5 * onp.linalg.slogdet(hess)[1] if positive else m)
        laplace.append(positive)
    log_w = onp.asarray(log_w, dtype=float)
    top = onp.max(log_w)
    weight = (
        onp.exp(log_w - top) if onp.isfinite(top) else onp.ones(len(log_w))
    )
    return weight / weight.sum(), onp.array(laplace, dtype=bool)


def _refined_gap(
    points, marginal, grid_score, xx, yy, rival
):  # deliberate: unit-tested on synthetic peaks; no cheap public route
    """The best peak's marginal score minus that of its best rival more
    than ``rival`` from it: a refined peak or, as a floor, a grid
    point."""
    x, y = points[0, :2]
    far = onp.hypot(points[1:, 0] - x, points[1:, 1] - y) > rival
    rivals = list(marginal[1:][far])
    by_position = grid_score.max(axis=2)
    far_grid = onp.hypot(xx[..., 0] - x, yy[..., 0] - y) > rival
    if far_grid.any():
        rivals.append(by_position[far_grid].max())
    return float(marginal[0] - max(rivals)) if rivals else onp.inf


class _RefineProblem(eqx.Module):
    """The negative scale-marginalized surface of one dataset's static
    binary ``(dra, ddec, flux)``, as a kernel problem
    ([`_gauss_newton`][virgil._gauss_newton]).

    The curvature rows are the blocks' whitened residuals weighted by the
    IRLS weights of the surface ([`_Surface.weights`][virgil.epochs._Surface.weights]);
    the priors are flat in these coordinates.
    """

    data: object  # an OIData
    dof: jax.Array
    surface: _Surface = eqx.field(static=True)
    diam: object = None  # the primary's diameter (mas), or None: a point
    amp_range: object = eqx.field(static=True, default=None)  # V² amplitude

    def loss(self, x):
        return _negative_marginal(
            x, self.dof, self.data, self.surface, self.diam, self.amp_range
        )

    def rows(self, x):
        r = _amp_residuals(
            _binary(x, self.diam), self.data, self.surface, self.amp_range
        )
        return self.surface.weighted_rows(r, self.dof, self.data)

    def diag(self, x):
        return np.zeros_like(x)


def _refine_marginal(x0, dof, d64, surface, bounds, diam=None, amp_range=None):
    """Maximize the scale-marginalized surface from ``x0`` (damped Newton on
    the IRLS curvature, in float64, deterministic) within ``bounds``.
    Returns the point, the surface's negative value there and the kernel's
    stop code (``_gauss_newton.CONVERGED`` and so on).

    The kernel only accepts steps that lower the loss, so a run that stalls
    (``STALLED``, the damping limit) or reaches ``max_steps`` (``LIMIT``)
    still ends at a point at least as good as ``x0``, and that point is
    returned (the surface's value there is a lower bound on its maximum).
    Only a start that is not finite (``NON_FINITE``) returns ``x0`` and
    ``inf``, which the caller reads as no improvement."""
    state = _gauss_newton.solve(
        _RefineProblem(d64, dof, surface, diam, amp_range),
        onp.asarray(x0, float),
        lower=[b[0] for b in bounds],
        upper=[b[1] for b in bounds],
        dtol=1e-12,
        max_steps=200,
    )
    stop = int(state.stop)
    if stop == _gauss_newton.NON_FINITE:
        return onp.asarray(x0, dtype=float), float("inf"), stop
    return onp.asarray(state.z, dtype=float), float(state.value), stop


_STOP_CAUSE = {
    _gauss_newton.STALLED: "the damping limit",
    _gauss_newton.LIMIT: "the step limit",
    _gauss_newton.NON_FINITE: "a start where the surface is not finite",
}


def _unconverged_cause(stops):
    """The causes in ``stops`` (kernel stop codes, not CONVERGED), in words."""
    return " and ".join(_STOP_CAUSE[c] for c in sorted(set(stops)))


def _flag_one_night(  # deliberate: 85 lines, tested directly
    catalogue,
    gap_marginal,
    flux_bounds,
    resolution,
    *,
    alias_nats,
    min_gap,
    sep_fraction,
    flux_tol,
    sep_blocks=False,
):
    """The flags of one night that need only that night: ``alias-peaks``,
    ``low-gap``, ``sep-below-resolution`` (non-blocking unless
    ``sep_blocks``) and ``flux-at-bound``. A
    threshold of None disables its flag."""
    flags = []
    best = float(catalogue.marginal[0])
    # orbit_solved_problems.md, "Single-night alias peaks" row: a night whose catalogue
    # holds another peak within a few nats of the best must not seed orbits.
    rivals = catalogue.marginal[1:]
    close = (
        0 if alias_nats is None else int(onp.sum(best - rivals <= alias_nats))
    )
    if close:
        flags.append(
            NightFlag(
                "alias-peaks",
                f"{close} other peak(s) within {alias_nats:g} nats of the "
                f"best (marginal gap {gap_marginal:.1f}"
                + (
                    ", so also below the low-gap limit"
                    if min_gap is not None and gap_marginal < min_gap
                    else ""
                )
                + "): a single-night alias, the true position may be a rival",
            )
        )
    elif min_gap is not None and gap_marginal < min_gap:
        flags.append(
            NightFlag(
                "low-gap",
                f"marginal gap {gap_marginal:.1f} nats is below {min_gap:g}: "
                "the position is not decisive",
            )
        )
    sep = float(onp.hypot(catalogue.dra[0], catalogue.ddec[0]))
    if sep_fraction is not None and sep < sep_fraction * resolution:
        flags.append(
            NightFlag(
                "sep-below-resolution",
                f"separation {sep:.2f} mas is below {sep_fraction:g} "
                f"lambda/B_max = {sep_fraction * resolution:.2f} mas: the "
                "companion is not resolved"
                + (
                    ""
                    if sep_blocks
                    else " (V^2 and closure phases still locate it below "
                    "lambda/2B; reported only)"
                ),
                blocking=sep_blocks,
            )
        )
    if flux_tol is not None:
        lo, hi = flux_bounds
        flux = float(catalogue.flux[0])
        if flux >= hi * (1.0 - flux_tol):
            flags.append(
                NightFlag(
                    "flux-at-bound",
                    f"flux {flux:.3g} is at the upper bound {hi:g}: the "
                    "labels of the stars are undetermined (genuine for a "
                    "near-equal pair)",
                )
            )
        elif flux <= lo * (1.0 + flux_tol):
            flags.append(
                NightFlag(
                    "flux-at-bound",
                    f"flux {flux:.3g} is at the lower bound {lo:.3g}: no "
                    "companion is detected",
                )
            )
    return flags


def broad_loglike(
    data,
    dof,
    *,
    a_max,
    flux_range,
    step,
    s_max=None,
    primary_diam=None,
    batch_size=None,
    memory_budget=None,
    coarse_nats=30.0,
    coarse_margin=1,
    n_flux=_N_FLUX,
):
    """Each night's log mean likelihood over the orbit prior's companion
    positions: the outlier component of
    [`fit_orbit_aliases`][virgil.aliases.fit_orbit_aliases].

    The positions are uniform on the disc ``|p| <= a_max`` and the flux is
    log-uniform on ``flux_range``. The scale-marginalized surface of
    [`epoch_positions`][virgil.epochs.epoch_positions] is scored as the orbit
    is: pass the fit's block ``dof`` and ``s_max``.

    Parameters
    ----------
    data : sequence of OIData
    dof : sequence of array
        One vector of block ``1/f_b`` per night.
    a_max, flux_range, step : float or tuple
        Disc radius (mas; ``a_range[1] (1 + ecc_max)``), flux range and grid
        step (mas).
    s_max, primary_diam, batch_size, memory_budget, coarse_nats, coarse_margin : optional
        As for ``epoch_positions``.
    n_flux : int, optional
        Log-spaced flux nodes.

    Returns
    -------
    numpy.ndarray
    """
    n = 2 * math.ceil(a_max / step) + 1
    axis = onp.linspace(-a_max, a_max, n)
    xx, yy, ff = onp.meshgrid(
        axis, axis, onp.geomspace(*flux_range, n_flux), indexing="ij"
    )
    inside = (xx**2 + yy**2 <= a_max**2).ravel()
    points = onp.stack([xx.ravel(), yy.ravel(), ff.ravel()], -1)
    diam = None if not primary_diam else float(primary_diam)
    out = []
    for d, dof_k in zip(data, dof):
        surface = _Surface(d, s_max)
        with run_in("float64"):
            d64 = cast_tree(d, "float64")
            dof64 = np.asarray(dof_k, dtype=np.float64)
            diam64 = None if diam is None else np.asarray(diam)
            if surface.numeric and coarse_nats is not None:
                m, _ = _coarse_then_exact(
                    points,
                    xx.shape,
                    d64,
                    surface,
                    diam64,
                    dof64,
                    batch_size,
                    memory_budget,
                    float(coarse_nats),
                    int(coarse_margin),
                )
                m = onp.asarray(m)
            else:
                _, chi2 = _grid_scores(
                    np.asarray(points),
                    d64,
                    surface,
                    min(
                        _grid_batch(batch_size, memory_budget, d64, surface),
                        len(points),
                    ),
                    diam64,
                    dof64,
                )
                m = onp.asarray(_grid_marginal(chi2, dof64, d64, surface))
        m = onp.asarray(m).ravel()
        m = onp.where(onp.isfinite(m) & inside, m, -onp.inf)
        out.append(float(onp.logaddexp.reduce(m) - onp.log(inside.sum())))
    return onp.array(out)


def sidelobe_flags(
    data, positions, predicted, *, k_sigma=3.0, delta_nats=10.0, dof=1.0
):
    """Flag nights whose orbit sits on a secondary peak of the night's surface.

    A night is flagged (``orbit-prefers-secondary``) when its best peak lies
    more than ``k_sigma`` from the predicted position, yet the marginal
    surface there is within ``delta_nats`` of the peak. The flag is reported,
    never used to exclude the night.

    Parameters
    ----------
    data : Epochs or sequence of OIData
        The data ``positions`` came from.
    positions : EpochPositions
        The single-night positions.
    predicted : tuple of array
        ``(dra, ddec, flux)`` of the orbit at each night (mas, mas,
        companion/primary).
    k_sigma, delta_nats, dof : float, optional
        Distance (sigma), surface margin (nats) and dof fraction.

    Returns
    -------
    list of NightFlag or None
        One entry per night.
    """
    out = []
    diam = positions.primary_diam
    for k, d in enumerate(getattr(data, "data", data)):
        px, py, pf = (float(onp.asarray(a)[k]) for a in predicted)
        if pf > 1.0:  # the other star's labels: the same scene, reflected
            px, py, pf = -px, -py, 1.0 / pf
        dx = onp.array([px - positions.dra[k], py - positions.ddec[k]])
        cov = onp.asarray(positions.cov[k])
        dist = float(onp.sqrt(dx @ onp.linalg.solve(cov, dx)))
        if not dist > k_sigma:
            out.append(None)
            continue
        model = _binary(
            np.asarray([px, py, min(max(pf, 1e-6), 1.0)]),
            None if not diam else np.asarray(float(diam)),
        )
        with run_in("float64"):
            at_pred = float(
                marginal_loglike(model, cast_tree(d, "float64"), dof=dof)
            )
        # Both ends of the gap are scored with the same model and dof (the
        # catalogue's own scores may come from a point primary or another
        # dof).
        peak = _binary(
            np.asarray(
                [
                    positions.dra[k],
                    positions.ddec[k],
                    min(float(positions.flux[k]), 1.0),
                ]
            ),
            None if not diam else np.asarray(float(diam)),
        )
        with run_in("float64"):
            best = float(
                marginal_loglike(peak, cast_tree(d, "float64"), dof=dof)
            )
        gap = best - at_pred
        if gap <= delta_nats:
            out.append(
                NightFlag(
                    "orbit-prefers-secondary",
                    f"the night's best peak is {dist:.1f} sigma from the "
                    f"orbit's position, which the night's surface scores "
                    f"{gap:.1f} nats below the peak (within "
                    f"{delta_nats:g}): the orbit sits on a secondary peak",
                    blocking=False,
                )
            )
        else:
            out.append(None)
    return out


def epoch_positions(
    data,
    grid,
    *,
    gap_mas=None,
    refine=True,
    n_peaks=5,
    dof=1.0,
    s_max=None,
    batch_size=None,
    memory_budget=None,
    coarse_nats=30.0,
    coarse_margin=1,
    primary_diam=None,
    flag_nights=True,
    flag_alias_nats=2.0,
    flag_min_gap=5.0,
    flag_sep_fraction=0.5,
    flag_flux_tol=0.02,
    flag_sep_blocks=False,  # deliberate: V^2 and closure phases locate a companion below lambda/2B
):
    """The companion's position in each dataset of a binary.

    Scores a static binary on a grid, refines the ``n_peaks`` best maxima and
    ranks them; each night also gets flags
    ([tutorial](../orbit_fitting.md#scale-marginalized-scoring-and-night-flags)).

    Parameters
    ----------
    data : Epochs
        The data (gains, offsets and model-dependent covariances are refused).
    grid : dict
        ``{"dra": axis, "ddec": axis, "flux": values}``: positions (mas) and
        fluxes in (0, 1], stepped finer than the resolution.
    gap_mas : float, optional
        Distance (mas) beyond which a peak rivals the best (default: resolution).
    refine, n_peaks, dof, s_max : optional
        Refine grid peaks; peaks kept per dataset (1 can commit to the wrong
        peak); as for ``marginal_loglike``.
    batch_size, memory_budget, coarse_nats, coarse_margin : optional
        Speed knobs; the coarse ones limit the exact scale integral to the
        best points (None: everywhere).
    primary_diam : float, optional
        Fixed uniform-disk primary diameter (mas); None or 0 is a point
        primary. Never the diameter implied by the orbit under test.
    flag_nights, flag_sep_blocks, flag_alias_nats, flag_min_gap, flag_sep_fraction, flag_flux_tol : optional
        Night flags and thresholds ([night flags](../orbit_methods.md#night-flags));
        None disables a numeric threshold; ``flag_nights`` and ``flag_sep_blocks``
        are switches.

    Returns
    -------
    EpochPositions

    Warns
    -----
    UserWarning
        If a best peak is on the grid edge, has raw chi-squared/N above 4, or
        did not converge.
    """
    if not isinstance(data, Epochs):
        raise TypeError(f"data must be an Epochs, not {type(data).__name__}.")
    _check_n_peaks(n_peaks)
    if isinstance(primary_diam, str):
        raise ValueError(
            f"primary_diam={primary_diam!r}: epoch_positions supports only "
            "a fixed diameter (a float, in mas) or None for now; fitting "
            "it ('fit') may be added later."
        )
    diam = None if primary_diam is None else float(primary_diam)
    if diam is not None and not diam >= 0:
        raise ValueError(f"primary_diam must be None or >= 0, not {diam}.")
    diam = diam or None  # 0 is a point primary
    missing = {"dra", "ddec", "flux"} - set(grid)
    if missing:
        raise ValueError(f"grid needs the axes {sorted(missing)}.")
    axes = [onp.asarray(grid[k], float) for k in ("dra", "ddec", "flux")]
    if (axes[2] <= 0).any() or (axes[2] > 1).any():
        raise ValueError(
            "The grid's fluxes must be in (0, 1]: a companion brighter than "
            "the primary is the fainter one on the other side (1/f at -r)."
        )
    xx, yy, ff = onp.meshgrid(*axes, indexing="ij")
    points = onp.stack([xx.ravel(), yy.ravel(), ff.ravel()], -1)
    steps = [
        float(onp.min(onp.diff(onp.sort(a)))) if a.size > 1 else 1.0
        for a in axes[:2]
    ]
    bounds = [(a.min() - s, a.max() + s) for a, s in zip(axes[:2], steps)]
    bounds.append((0.1 * float(axes[2].min()), 1.0))
    rows, inflated, edge_grid, edge_peak = [], [], [], []
    unconverged = {}  # night -> the stop code of its best peak's refinement
    night_flags = []
    exact_share = []
    for index, (name, d) in enumerate(zip(data.dataset_names, data.data)):
        surface = _Surface(d, s_max)
        dof_k = _check_dof(_dof_for(dof, data, index))
        with run_in("float64"):
            d64 = cast_tree(d, "float64")
            diam64 = None if diam is None else np.asarray(diam)
            dof_arr = np.asarray(dof_k)
            if surface.numeric and coarse_nats is not None:
                marginal, share = _coarse_then_exact(
                    points,
                    xx.shape,
                    d64,
                    surface,
                    diam64,
                    dof_arr,
                    batch_size,
                    memory_budget,
                    float(coarse_nats),
                    int(coarse_margin),
                )
                marginal = marginal.reshape(xx.shape)
            else:
                _, chi2 = _grid_scores(
                    np.asarray(points),
                    d64,
                    surface,
                    min(
                        _grid_batch(batch_size, memory_budget, d64, surface),
                        len(points),
                    ),
                    diam64,
                    dof_arr,
                )
                marginal = _grid_marginal(chi2, dof_arr, d64, surface)
                marginal = onp.asarray(marginal).reshape(xx.shape)
                share = 1.0
        exact_share.append(share)
        marginal = onp.where(onp.isfinite(marginal), marginal, -onp.inf)
        rival = (
            Epochs({name: d}).resolution_mas if gap_mas is None else gap_mas
        )
        # The gap: the best position's score minus that of the best point
        # more than ``rival`` from it.
        by_position = marginal.max(axis=2)
        i, j = onp.unravel_index(onp.argmax(by_position), by_position.shape)
        far = (
            onp.hypot(xx[..., 0] - xx[i, j, 0], yy[..., 0] - yy[i, j, 0])
            > rival
        )
        gap_marginal = float(
            by_position[i, j] - by_position[far].max()
            if far.any()
            else onp.inf
        )
        cells = _grid_peaks(marginal, xx, yy, rival, n_peaks)
        grid_points = [
            [axes[0][a], axes[1][b], axes[2][c]] for a, b, c in cells
        ]
        with run_in("float64"):
            dof64 = np.asarray(dof_k)

        def score(x, d64=d64, surface=surface, dof64=dof64, diam64=diam64):
            with run_in("float64"):
                quoted, chi2 = _grid_scores(
                    np.asarray(x), d64, surface, len(x), diam64, dof64
                )
                m = _grid_marginal(chi2, dof64, d64, surface)
            return onp.asarray(quoted), onp.asarray(m)

        refine_stop = []  # each grid peak's refinement stop code

        def refine_one(
            x0, d64=d64, surface=surface, dof64=dof64, diam64=diam64
        ):
            with run_in("float64"):
                x, value, stop = _refine_marginal(
                    x0, dof64, d64, surface, bounds, diam64
                )
            refine_stop.append(stop)
            return x, -value

        peaks, peak_loglike, peak_marginal, order = _rank_peaks(
            grid_points,
            [marginal[c] for c in cells],
            refine_one if refine else None,
            score,
            rival,
        )
        if refine:
            gap_marginal = _refined_gap(
                peaks, peak_marginal, marginal, xx, yy, rival
            )
            if refine_stop[order[0]] != _gauss_newton.CONVERGED:
                unconverged[name] = refine_stop[order[0]]
        at_edge = [
            a in (0, xx.shape[0] - 1) or b in (0, xx.shape[1] - 1)
            for a, b, _ in cells
        ]
        on_edge = onp.array([at_edge[n] for n in order], dtype=bool)
        if at_edge[0]:
            edge_grid.append(name)
        elif on_edge[0]:
            edge_peak.append(name)
        hessians = [None] * len(peaks)
        if refine:
            with run_in("float64"):
                hessians = [
                    onp.asarray(
                        _negative_marginal_hessian(
                            np.asarray(x), dof64, d64, surface, diam64
                        ),
                        dtype=float,
                    )
                    for x in peaks
                ]
        weight, laplace = _peak_weights(peak_marginal, hessians)
        catalogue = EpochPeaks(
            dra=peaks[:, 0],
            ddec=peaks[:, 1],
            flux=peaks[:, 2],
            loglike=peak_loglike,
            marginal=peak_marginal,
            weight=weight,
            edge=on_edge,
            laplace=laplace,
        )
        best = peaks[0]
        cov = max(steps) ** 2 * onp.eye(2)
        with run_in("float64"):
            if refine and onp.all(onp.isfinite(hessians[0])):
                full = onp.linalg.pinv(hessians[0])
                if onp.all(onp.isfinite(full)) and onp.all(
                    onp.linalg.eigvalsh(full[:2, :2]) > 0
                ):
                    cov = full[:2, :2]
            chi2_best = onp.asarray(
                surface.plain_chi2(
                    _block_chi2(np.asarray(best), d64, surface, diam64)
                ),
                dtype=float,
            )
        per_block = dict(zip(surface.names, chi2_best / surface.nu))
        per_block["all"] = float(chi2_best.sum() / surface.nu.sum())
        if per_block["all"] > 4.0:
            inflated.append(f"{name} ({per_block['all']:.1f})")
        scale = {
            f"{n}_scale": float(v)
            for n, v in zip(surface.names, surface.scales(chi2_best))
        }
        rows.append((best, cov, gap_marginal, per_block, scale, catalogue))
        night_flags.append(
            _flag_one_night(
                catalogue,
                gap_marginal,
                (bounds[2][0], bounds[2][1]),
                Epochs({name: d}).resolution_mas,
                alias_nats=flag_alias_nats,
                min_gap=flag_min_gap,
                sep_fraction=flag_sep_fraction,
                flux_tol=flag_flux_tol,
                sep_blocks=flag_sep_blocks,
            )
            if flag_nights
            else []
        )
    if inflated:
        warnings.warn(
            "Raw chi2/N on the quoted errors exceeds 4 (errors underestimated "
            f"by more than 2) in {', '.join(inflated)}: fit error scales for "
            "these datasets, and do not read a rescaled chi2/N of 1 as a "
            "good fit.",
            UserWarning,
            stacklevel=external_stacklevel(),
        )
    if edge_grid:
        warnings.warn(
            f"The best grid position of {', '.join(edge_grid)} is at the "
            "edge of the grid: the companion may lie outside it. Widen the "
            "grid.",
            UserWarning,
            stacklevel=external_stacklevel(),
        )
    if edge_peak:
        warnings.warn(
            f"The best refined peak of {', '.join(edge_peak)} started from "
            "the edge of the grid: the companion may lie outside it. Widen "
            "the grid.",
            UserWarning,
            stacklevel=external_stacklevel(),
        )
    if unconverged:
        warnings.warn(
            f"The refinement of the best peak of {', '.join(unconverged)} "
            "did not converge (it ended at "
            f"{_unconverged_cause(unconverged.values())}): the position is "
            "the best point reached, and its marginal score and covariance "
            "are approximate.",
            UserWarning,
            stacklevel=external_stacklevel(),
        )
    return EpochPositions(
        names=data.dataset_names,
        mjd=onp.asarray(data.times, dtype=onp.float64),
        dra=onp.array([r[0][0] for r in rows]),
        ddec=onp.array([r[0][1] for r in rows]),
        cov=onp.array([r[1] for r in rows]),
        flux=onp.array([r[0][2] for r in rows]),
        gap_marginal=onp.array([float(r[2]) for r in rows]),
        chi2_raw=tuple(r[3] for r in rows),
        scale=tuple(r[4] for r in rows),
        peaks=tuple(r[5] for r in rows),
        flags=tuple(tuple(f) for f in night_flags),
        primary_diam=diam,
        grid=tuple(axes),
        exact_fraction=tuple(exact_share),
    )


# Resolved primary (orbit_solved_problems.md, "Resolved primary in the orbit fit" row). The
# data resolve the primary when a shared uniform-disk diameter improves the
# summed scale-marginalized score of the nights by this many nats over a
# point primary: one extra parameter, so 5 nats is Δχ² ≈ 10, about 3σ.
RESOLVE_NATS = 5.0
# The diameters tried, as fractions of the finest fringe spacing λ/B_max: a
# disk much smaller than 0.03 λ/B_max changes V² by less than a part in
# 10⁴, and one of 1.5 λ/B_max has crossed the first visibility null.
_DIAM_FRACTIONS = (0.03, 0.06, 0.12, 0.25, 0.4, 0.6, 0.9, 1.3)
# A position is searched within this many resolution elements of its start.
_REFINE_WINDOW = 3.0


@dataclasses.dataclass(frozen=True)
class PrimaryDiameter:
    """The primary's uniform-disk diameter, from
    [`estimate_primary_diameter`][virgil.epochs.estimate_primary_diameter].

    Attributes
    ----------
    diam : float
        Best diameter (mas); 0 when a point primary is as good.
    sigma : float
        1σ uncertainty (mas); NaN when the maximum is on the grid edge.
    delta_nats : float
        Gain in summed marginal score at ``diam`` over a point primary.
    resolved : bool
        ``diam > 0`` and ``delta_nats`` at least ``resolve_nats``.
    at_grid_edge : bool
        The best diameter is the last grid point, so ``diam`` is a lower limit.
    grid, profile : numpy.ndarray
        The diameters tried (``grid[0]`` is 0) and the summed score at each; the
        point primary's score is the better of its fit and a refit from the best
        diameter's positions, so a poor local maximum does not pass for a resolved disc.
    positions : EpochPositions
        The input positions, re-refined at ``diam`` when ``resolved``.
    """

    diam: float
    sigma: float
    delta_nats: float
    resolved: bool
    grid: onp.ndarray
    profile: onp.ndarray
    positions: EpochPositions
    at_grid_edge: bool = False


def estimate_primary_diameter(
    data,
    positions,
    *,
    resolve_nats=RESOLVE_NATS,
    diam_grid=None,
    s_max=None,
    vis_amp=None,
    dof=1.0,
    block_dof=None,
):
    """Estimate a shared uniform-disk primary diameter and whether the data resolve it.

    Each night's binary is re-refined over a diameter grid and the scores
    summed ([rule](../orbit_fitting.md#scale-marginalized-scoring-and-night-flags)).

    Parameters
    ----------
    data : Epochs or sequence of OIData
        The datasets, in the order of ``positions``.
    positions : EpochPositions
        Per-night positions; flagged nights are left out unless all are.
    resolve_nats : float, optional
        Gain (nats) over a point primary at which it counts as resolved (5).
    diam_grid : array-like, optional
        Positive diameters (mas) to try besides 0.
    s_max, dof, block_dof : optional
        As for ``marginal_loglike``; ``block_dof`` multiplies ``dof``.
    vis_amp : (float, float), optional
        Profile a V^2 amplitude per night in this range, so contrast loss is
        not read as a disk.

    Returns
    -------
    PrimaryDiameter

    Raises
    ------
    ValueError
        If ``data`` and ``positions`` are not the same nights, or ``diam_grid``
        has a non-positive diameter.

    Warns
    -----
    UserWarning
        If the best diameter is the largest tried, or a refinement did not converge.
    """
    sets = list(data.data) if isinstance(data, Epochs) else list(data)
    if len(sets) != len(positions.names):
        raise ValueError(
            f"{len(sets)} datasets but positions for "
            f"{len(positions.names)}: they must be the same nights."
        )
    use = ~positions.flagged
    if not use.any():
        use = onp.ones(len(sets), bool)
    with run_in("float64"):
        d64 = [cast_tree(d, "float64") for d in sets]
        surfaces = [_Surface(d, s_max) for d in d64]
        resolution = [float(fringe_scales(d)[0]) for d in sets]
        finest = min(resolution)
        grid = onp.concatenate(
            [
                [0.0],
                finest * onp.asarray(_DIAM_FRACTIONS)
                if diam_grid is None
                else onp.sort(onp.asarray(diam_grid, float)),
            ]
        )
        if (grid[1:] <= 0).any():
            raise ValueError("diam_grid must hold positive diameters.")
        start = [
            onp.array([positions.dra[i], positions.ddec[i], positions.flux[i]])
            for i in range(len(sets))
        ]
        dofs = [
            np.asarray(sf.dof_vector(block_dof), float)
            * float(_check_dof(dof))
            for sf in surfaces
        ]
        amp = None if vis_amp is None else tuple(map(float, vis_amp))

        def refit(i, x0, diam):
            """Refine night ``i`` from ``x0`` at ``diam``: (x, score, stop)."""
            window = _REFINE_WINDOW * resolution[i]
            x0 = onp.clip(
                x0,
                [x0[0] - window, x0[1] - window, 1e-4],
                [x0[0] + window, x0[1] + window, 1.0],
            )
            bounds = [
                (x0[0] - window, x0[0] + window),
                (x0[1] - window, x0[1] + window),
                (1e-4, 1.0),
            ]
            dm = None if diam <= 0 else np.asarray(float(diam))
            x, value, stop = _refine_marginal(
                x0, dofs[i], d64[i], surfaces[i], bounds, dm, amp
            )
            if not onp.isfinite(value):  # no improvement: score the start
                value = float(
                    _negative_marginal(
                        np.asarray(x0), dofs[i], d64[i], surfaces[i], dm, amp
                    )
                )
            return x, -value, stop

        def profile_at(diam, begin):
            """Summed score over the used nights at ``diam``, the fitted
            positions (None for the nights left out) and the stop codes of
            the refinements that did not converge, by night."""
            total, xs, bad = 0.0, list(begin), {}
            for i in onp.flatnonzero(use):
                xs[i], score, stop = refit(i, begin[i], diam)
                total += score
                if stop != _gauss_newton.CONVERGED:
                    bad[i] = stop
            return total, xs, bad

        profile, xs, fits, bads = [], start, [], []
        for diam in grid:  # ascending: each starts from the last fit
            total, xs, bad = profile_at(diam, xs)
            profile.append(total)
            fits.append(xs)
            bads.append(bad)
        profile = onp.asarray(profile)
        best = int(onp.argmax(profile))
        if best > 0:
            # A point primary fitted from the grid's starts can end at a
            # worse local maximum than a disc's fit: the gain would then
            # measure the local maximum, not the diameter. Refit it from the
            # best disc's positions and keep the better baseline.
            total, _, bad = profile_at(0.0, fits[best])
            if total > profile[0]:
                profile[0], bads[0] = total, bad
                best = int(onp.argmax(profile))
        diam, sigma = float(grid[best]), float("nan")
        gain = float(profile[best] - profile[0])
        # The refinements that set the reported diameter and gain; the other
        # grid diameters and the curvature probes only shape the search.
        unconverged = {**bads[0], **bads[best]}
        if 0 < best < len(grid) - 1:
            # A parabola through the best grid point and its neighbours
            # locates the maximum between grid points.
            a, b, _ = onp.polyfit(
                grid[best - 1 : best + 2], profile[best - 1 : best + 2], 2
            )
            vertex = -b / (2 * a) if a < 0 else float("nan")
            if grid[best - 1] <= vertex <= grid[best + 1]:
                total, _, bad = profile_at(vertex, fits[best])
                if total >= profile[best]:
                    diam, gain = float(vertex), float(total - profile[0])
                    unconverged.update(bad)
            # The uncertainty is the curvature of the profile at the
            # maximum, from steps of 15% of the diameter.
            h = 0.15 * diam
            up, _, _ = profile_at(diam + h, fits[best])
            down, _, _ = profile_at(diam - h, fits[best])
            peak = gain + profile[0]
            curve = 2 * peak - up - down
            if curve > 0:
                sigma = float(h / onp.sqrt(curve))
        at_edge = bool(best == len(grid) - 1 and diam > 0)
        if at_edge:
            warnings.warn(
                f"The best primary diameter, {diam:.3g} mas, is the largest "
                "tried: the primary may be larger. Widen diam_grid.",
                UserWarning,
                stacklevel=external_stacklevel(),
            )
        resolved = bool(diam > 0 and gain >= resolve_nats)
        chosen = diam if resolved else 0.0
        reseed = resolved or positions.primary_diam is not None
        if reseed:
            dm = None if chosen <= 0 else np.asarray(float(chosen))
            xs_all, covs = [], []
            for i in range(len(sets)):
                x, _, stop = refit(i, start[i], chosen)
                xs_all.append(x)
                if stop != _gauss_newton.CONVERGED:
                    unconverged[i] = stop
                full = onp.asarray(
                    _negative_marginal_hessian(
                        np.asarray(x), dofs[i], d64[i], surfaces[i], dm, amp
                    ),
                    dtype=float,
                )
                cov = positions.cov[i]
                if onp.all(onp.isfinite(full)):
                    inv = onp.linalg.pinv(full)[:2, :2]
                    if onp.all(onp.linalg.eigvalsh(inv) > 0):
                        cov = inv
                covs.append(cov)
            xs_all = onp.asarray(xs_all)
            seeded = dataclasses.replace(
                positions,
                dra=xs_all[:, 0],
                ddec=xs_all[:, 1],
                flux=xs_all[:, 2],
                cov=onp.asarray(covs),
                primary_diam=chosen or None,
            )
        else:
            seeded = positions
        if unconverged:
            warnings.warn(
                "A refinement that sets the diameter did not converge (it "
                f"ended at {_unconverged_cause(unconverged.values())}) for "
                f"{', '.join(positions.names[i] for i in sorted(unconverged))}"
                ": its score there is a lower bound, so the diameter and "
                "its gain are approximate.",
                UserWarning,
                stacklevel=external_stacklevel(),
            )
    return PrimaryDiameter(
        diam=diam,
        sigma=sigma,
        delta_nats=float(gain),
        resolved=resolved,
        grid=grid,
        profile=profile,
        positions=seeded,
        at_grid_edge=at_edge,
    )


@dataclasses.dataclass(frozen=True)
class OrbitStart:
    """A start for a multi-epoch orbit fit, from
    [`start_from_positions`][virgil.epochs.start_from_positions].

    Attributes
    ----------
    positions : EpochPositions
        The per-dataset positions that seeded the orbits.
    candidates : RankedOrbits
        The starting orbits, ranked by the visibilities.
    fits : tuple of FitResult
        Fits from the distinct best candidates, lowest loss first (non-finite
        last); ``info`` has the ``candidate`` rank, ``converged`` and ``stop``.
    data : Epochs
        The data.
    failed : tuple of (int, str)
        Candidates whose refinement raised, with the error.
    seeded, ambiguous : tuple of str
        Datasets whose positions seeded the orbits (``gap_marginal`` above
        ``min_gap``), and the rest, which still enter the ranking and fits.
    """

    positions: EpochPositions
    candidates: RankedOrbits
    fits: tuple
    data: Epochs
    failed: tuple = ()
    seeded: tuple = ()
    ambiguous: tuple = ()

    @property
    def best(self):
        """The fit with the lowest loss."""
        return self.fits[0]

    def modes(self, *, max_delta_loss=10.0, same_mode_chi2=1.0):
        """The distinct fits, lowest loss first.

        Two fits are one mode when their predictions differ by less than
        ``same_mode_chi2`` in χ² (the sum of squared differences of their
        whitened residuals, with the quoted errors). Fits whose loss is
        more than ``max_delta_loss`` above the best's are dropped: a
        difference Δ in loss is a factor of about e^Δ in posterior
        density. Fits with a non-finite loss are left out.
        """
        finite = [r for r in self.fits if math.isfinite(r.info["loss"])]
        if not finite:
            raise ValueError("No refinement fit has a finite loss.")
        floor = finite[0].info["loss"]
        kept, residuals = [], []
        for result in finite:
            if result.info["loss"] - floor > max_delta_loss:
                break
            models = result.model
            if not isinstance(models, (list, tuple)):
                models = [models] * len(self.data)
            r = onp.concatenate(
                [
                    onp.asarray(whitened_residuals(m, d))
                    for m, d in zip(models, self.data.data)
                ]
            )
            if all(onp.sum((r - q) ** 2) > same_mode_chi2 for q in residuals):
                kept.append(result)
                residuals.append(r)
        return kept

    def chain_values(self, n_chains, **options):
        """One start (fitted values) per chain, cycling through the
        distinct modes (see :meth:`modes`, which takes ``options``).

        Pass them to
        [`chain_init_params`][virgil.likelihood.chain_init_params] to
        start each chain of numpyro's ``MCMC`` in its own mode.
        """
        modes = self.modes(**options)
        return [dict(modes[k % len(modes)].values) for k in range(n_chains)]


def start_from_positions(
    model,
    priors,
    data,
    start_values,
    *,
    grid,
    periods,
    t_ref,
    scales="marginal",
    noise=None,
    dof=1.0,
    s_max=None,
    eccs=None,
    n_phase=36,
    n_candidates=200,
    n_refine=4,
    min_gap=5.0,
    refine_positions=True,
    n_peaks=5,
    batch_size=None,
    primary_diam=None,
    skip_flagged=True,
    flag_options=None,
    **fit_options,
):
    """Start an orbit fit to multi-epoch visibilities from a positions fit.

    Thiele-Innes orbits from per-dataset
    [`epoch_positions`][virgil.epochs.epoch_positions], ranked by all the
    data, then [`fit`][virgil.fitting.fit] from the best distinct ones. It
    only chooses where the fits start.

    Parameters
    ----------
    model : callable
        As for [`Epochs.model_fn`][virgil.epochs.Epochs.model_fn].
    priors : dict
        As for ``fit``.
    data : Epochs
        The data.
    start_values : callable
        ``start_values(orbit, flux)`` gives a value for every key of ``priors``.
    grid : dict
        Per-dataset grid of positions and fluxes (see ``epoch_positions``).
    periods : array-like
        Trial periods (days), e.g. from [`period_grid`][virgil.orbits.period_grid].
    t_ref : float
        Reference time (MJD) of the starting orbits.
    scales, noise, dof, s_max, batch_size, eccs, n_phase : optional
        As for ``rank_orbits`` and ``starting_orbits``; ``noise`` feeds only
        the refinement fits.
    n_candidates, n_refine, min_gap : optional
        Orbits ranked (200), distinct ones refined (4), and the
        ``gap_marginal`` (nats, 5) a dataset needs to seed.
    refine_positions, n_peaks, primary_diam, skip_flagged, flag_options : optional
        As for ``epoch_positions`` (``flag_min_gap`` defaults to ``min_gap``);
        flagged datasets seed nothing but still count.
    **fit_options
        Passed to the refinement fits; one that raises goes in ``OrbitStart.failed``.

    Returns
    -------
    OrbitStart
        The ranked orbits, the refined fits, ``ambiguous`` and ``failed``.
    """
    scales = _check_scales(scales)
    _check_n_peaks(n_peaks)
    positions = epoch_positions(
        data,
        grid,
        refine=refine_positions,
        n_peaks=n_peaks,
        dof=dof,
        s_max=s_max,
        primary_diam=primary_diam,
        **{"flag_min_gap": min_gap, **(flag_options or {})},
    )
    decisive = positions.decisive(min_gap)
    if skip_flagged:
        decisive = decisive & ~positions.flagged
    seeded = tuple(n for n, d in zip(positions.names, decisive) if d)
    ambiguous = tuple(n for n, d in zip(positions.names, decisive) if not d)
    if ambiguous and seeded:
        warnings.warn(
            f"The positions of {', '.join(ambiguous)} are ambiguous (a "
            f"marginal gap of at most min_gap={min_gap}) or flagged "
            f"({positions.flag_codes()}) and do not seed the orbits; their "
            "visibilities still enter the ranking and the fits. See "
            "OrbitStart.ambiguous and positions.peaks.",
            UserWarning,
            stacklevel=external_stacklevel(),
        )
    with run_in("float64"):
        seeds = positions.positions(
            t_ref=t_ref, min_gap=min_gap, skip_flagged=skip_flagged
        )
    if len(seeds.dt) < 2:
        raise ValueError(
            "At least two decisive, unflagged datasets are needed to seed "
            f"orbits; the marginal gaps are {positions.gap_marginal.round(1)} "
            f"and the flagged datasets are {positions.flag_codes()}. Lower "
            "min_gap, refine the grid, or (if flags are the cause) pass "
            "skip_flagged=False or loosen flag_options."
        )
    _warn_if_coarse(periods, seeds.dt)
    with run_in("float64"):
        candidates = starting_orbits(
            seeds, periods, eccs=eccs, n_phase=n_phase, n_best=n_candidates
        )
    flux = float(onp.median(positions.flux[decisive]))
    orbits = [orbit for orbit, _ in candidates]
    values = [dict(start_values(orbit, flux)) for orbit in orbits]
    missing = set(priors) - set(values[0])
    if missing:
        raise KeyError(
            f"start_values returned no value for {sorted(missing)}; it "
            "must give one for every key of priors."
        )
    stacked = _stack(
        [{k: v[k] for k in priors} for v in values], "starting values"
    )
    loglike = _loglikes(
        _Kwargs(model),
        stacked,
        data,
        None,
        batch_size,
        "float64",
        scales=scales,
        dof=dof,
        s_max=s_max,
    )
    # deliberate: the ranking ignores ``noise=``; nothing here enters the
    # posterior, the start only picks where the fits begin.
    ranked = _ranked(orbits, loglike, data)
    chosen = _distinct(ranked, n_refine, 0.5 * ranked.resolution_mas)
    if not chosen:
        raise ValueError("No starting orbit has a finite log likelihood.")
    fits, failed, error = [], [], None
    # One function for every refinement, so ``fit`` compiles once: a fresh
    # closure per start would recompile for each.
    model_fn = data.model_fn(model)
    for k in chosen:
        # One bad start (a NaN, a solver error) must not lose the others.
        # TypeError and ValueError are the caller's (a misspelt option,
        # Adam with a time limit): they do not depend on the start, so
        # they propagate.
        try:
            result = fit(
                model_fn,
                priors,
                data.data,
                noise=noise,
                init=values[ranked.order[k]],
                **fit_options,
            )
        except (TypeError, ValueError):
            raise
        except Exception as caught:
            error = caught
            reason = f"{type(error).__name__}: {error}"
            failed.append((int(k), reason))
            warnings.warn(
                f"The refinement fit from candidate {k} failed ({reason}); "
                "the others go on.",
                RuntimeWarning,
                stacklevel=external_stacklevel(),
            )
            continue
        result.info["candidate"] = int(k)
        fits.append(result)
    if not fits:
        raise RuntimeError(
            f"Every refinement fit failed: {failed}."
        ) from error
    fits.sort(key=_loss_order)
    return OrbitStart(
        positions=positions,
        candidates=ranked,
        fits=tuple(fits),
        data=data,
        failed=tuple(failed),
        seeded=seeded,
        ambiguous=ambiguous,
    )


def _loss_order(
    result,
):  # deliberate: only its direct test covers the sort order
    """Sort key: lowest loss first, non-finite losses last."""
    loss = result.info["loss"]
    return (0, loss) if math.isfinite(loss) else (1, 0.0)


def _warn_if_coarse(
    periods, times
):  # deliberate: only its direct test covers the warning
    """Warn when trial periods drift apart by more than 1/k of a cycle
    over the baseline of ``times`` (see ``period_grid``)."""
    times = onp.asarray(times, float)
    baseline = float(onp.ptp(times)) if times.size else 0.0
    drift = _phase_drift(periods, baseline)
    if drift > 1.0 / _PHASE_COHERENCE:
        warnings.warn(
            f"Neighbouring trial periods drift apart by up to {drift:.2g} "
            f"cycles over the {baseline:.0f}-day baseline of the seeding "
            f"datasets (more than 1/{_PHASE_COHERENCE:g}), so the starting "
            "orbits can miss the true period; use "
            "virgil.orbits.period_grid(times, p_min, p_max).",
            UserWarning,
            stacklevel=external_stacklevel(),
        )
