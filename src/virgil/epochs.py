"""Named epochs of data, with one snapshot of a moving scene per dataset.

An orbit fit to interferometric data across epochs predicts each epoch's
visibilities and closure phases from the scene at that epoch's time. When
nothing moves appreciably during an observation (a binary whose period is
much longer than a night), one snapshot of the scene per dataset is
enough: the orbit is solved once per dataset, and the snapshot is a static
model, evaluated on all of the dataset's samples at once on the fast path
(e.g. [`BinaryModelCartesian`][virgil.models.BinaryModelCartesian] for
[`OrbitalBinary`][virgil.models.OrbitalBinary]). That is much cheaper
than [`OIData.model`][virgil.oidata.OIData.model]'s evaluation of a
time-dependent scene at every sample's own time, which remains the choice
when the scene moves within an observation.

[`Epochs`][virgil.epochs.Epochs] keys epochs and datasets **by name**,
so that per-dataset nuisance terms (error scales, wavelength scales,
North angles) are attached to the right data however the list is
ordered, and it hands [`fit`][virgil.fitting.fit] and
[`numpyro_model`][virgil.likelihood.numpyro_model] what they take: a
model function returning one snapshot per dataset, the datasets, and one
``noise`` dict per dataset.

Fringe aliases make the likelihood of visibilities multimodal on the
scale of the resolution λ/B, so an orbit fit needs a start in the right
basin. The tools for starting and sampling such a fit are:

* [`rank_orbits`][virgil.epochs.rank_orbits]: the log likelihood of all
  the data for each of many trial orbits (e.g. from
  [`starting_orbits`][virgil.orbits.starting_orbits]), best first;
* [`chain_starts`][virgil.epochs.chain_starts]: the best *distinct* orbits
  of a ranking, one per chain;
* [`epoch_positions`][virgil.epochs.epoch_positions]: the companion's
  position in each dataset, from a grid and a binary fit;
* [`start_from_positions`][virgil.epochs.start_from_positions]: the whole
  start: positions, starting orbits, ranking on the visibilities and
  refinement with [`fit`][virgil.fitting.fit] from several distinct
  orbits, with [`OrbitStart.chain_values`][virgil.epochs.OrbitStart.chain_values]
  giving one start per chain for
  [`chain_init_params`][virgil.likelihood.chain_init_params].
"""

import dataclasses
import functools

import jax
import jax.numpy as np
import numpy as onp
import numpyro.distributions as dist

from ._precision import cast_tree, run_in
from .fitting import fit
from .likelihood import model_loglike, whitened_residuals
from .models import BinaryModelCartesian
from .orbits import PositionData, starting_orbits


__all__ = [
    "Epochs",
    "EpochPositions",
    "OrbitStart",
    "RankedOrbits",
    "chain_starts",
    "epoch_positions",
    "rank_orbits",
    "start_from_positions",
]

_MAS = onp.pi / 180.0 / 3.6e6  # radians per milliarcsecond

_AT = ("dataset", "epoch")


class Epochs:
    """Datasets grouped into named epochs, each seen as one snapshot.

    Parameters
    ----------
    epochs : dict
        ``{epoch name: data}``, where ``data`` is one
        [`OIData`][virgil.oidata.OIData], a list of them, or a dict
        ``{dataset name: OIData}``. One epoch may hold several datasets,
        e.g. several files of one night, or two filters a day apart. A
        lone dataset is named after its epoch; datasets in a list are
        named ``"<epoch>[k]"``. Every name must be unique.
    at : {"dataset", "epoch"}, optional
        Where each snapshot is taken: at the mean time of each dataset's
        samples (default), or at the mean time of the whole epoch, shared
        by its datasets.
    times : dict, optional
        ``{epoch or dataset name: MJD}``: snapshot times to use instead of
        the data's own, e.g. for data without per-sample times (AMIGO
        DISCOs). A dataset's entry overrides its epoch's.

    Attributes
    ----------
    names : tuple of str
        The epochs, in the order given.
    dataset_names : tuple of str
        The datasets, epoch by epoch.
    epoch_of : tuple of str
        The epoch of each dataset.
    data : tuple of OIData
        The datasets, in the order of ``dataset_names``: pass this as
        ``data`` to ``fit`` and ``numpyro_model``.
    times : numpy.ndarray
        The snapshot time of each dataset (MJD, float64).
    spread_days : numpy.ndarray
        The largest distance in time of each dataset's samples from its
        snapshot (days). The scene should move by much less than the
        resolution (λ/B) over this time; if it does not, evaluate the
        dataset per sample instead (pass it to ``fit`` directly with the
        time-dependent scene).

    Examples
    --------
    >>> import numpy as np
    >>> from virgil.epochs import Epochs
    >>> from virgil.oidata import OIData
    >>> def night(mjd):
    ...     return OIData({"u": [10.0, 20.0], "v": [5.0, -3.0],
    ...                    "wavel": 2.2e-6, "vis": [1.0, 1.0],
    ...                    "d_vis": [0.01, 0.01], "mjd": [mjd, mjd + 0.1]})
    >>> epochs = Epochs({"2023": {"ut": night(60100.0), "at": night(60101.0)},
    ...                  "2024": night(60500.0)})
    >>> epochs.dataset_names, epochs.epoch_of
    (('ut', 'at', '2024'), ('2023', '2023', '2024'))
    >>> [round(float(t), 2) for t in epochs.times]
    [60100.05, 60101.05, 60500.05]
    >>> epochs.noise({"2023": {"vis_scale": 2.0}, "2024": {"vis_scale": 3.0}})
    [{'vis_scale': 2.0}, {'vis_scale': 2.0}, {'vis_scale': 3.0}]
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
        self.times, self.spread_days = self._snapshot_times(times or {})

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        return (
            f"Epochs({len(self.names)} epochs, {len(self.data)} datasets, "
            f"at={self.at!r})"
        )

    def _snapshot_times(self, given):
        """Snapshot time and spread of each dataset."""
        unknown = set(given) - set(self.names) - set(self.dataset_names)
        if unknown:
            raise ValueError(f"times for unknown names: {sorted(unknown)}.")
        own = []
        for name, d in zip(self.dataset_names, self.data):
            mjd = getattr(d, "mjd", None)
            own.append(None if mjd is None else onp.asarray(mjd, onp.float64))
        times = onp.empty(len(self.data))
        for k, (name, epoch) in enumerate(
            zip(self.dataset_names, self.epoch_of)
        ):
            if name in given:
                times[k] = given[name]
            elif epoch in given:
                times[k] = given[epoch]
            elif own[k] is None:
                raise ValueError(
                    f"Dataset {name!r} has no times: read it from OIFITS, "
                    "give mjd per sample, or pass times={name: mjd}."
                )
            elif self.at == "dataset":
                times[k] = own[k].mean()
            else:
                together = [
                    own[j]
                    for j, e in enumerate(self.epoch_of)
                    if e == epoch and own[j] is not None
                ]
                times[k] = onp.concatenate(together).mean()
        spread = onp.array(
            [
                0.0 if t is None else float(onp.max(onp.abs(t - times[k])))
                for k, t in enumerate(own)
            ]
        )
        return times, spread

    @property
    def resolution_mas(self):
        """The finest resolution λ/B_max over the datasets (mas).

        Fringe aliases of a companion's position are spaced by about this
        much, so starting orbits closer together than a fraction of it are
        one mode.
        """
        finest = onp.inf
        for d in self.data:
            baseline = onp.hypot(onp.asarray(d.u), onp.asarray(d.v))
            wavel, baseline = onp.broadcast_arrays(
                onp.asarray(d.wavel, dtype=float), baseline
            )
            ok = baseline > 0
            if ok.any():
                finest = min(finest, float(onp.min(wavel[ok] / baseline[ok])))
        return finest / _MAS

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
        scenes = (
            list(scene)
            if isinstance(scene, (list, tuple))
            else [scene] * len(self.data)
        )
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


def _orbit_list(orbits):
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


def _loglikes(scene_of, inputs, data, noise, batch_size, dtype):
    """``data``'s log likelihood of ``scene_of(x)`` for each ``x`` in the
    stacked ``inputs``, as a NumPy array (``-inf`` where not finite)."""
    if not isinstance(data, Epochs):
        raise TypeError(
            f"data must be an Epochs, not {type(data).__name__}: rankings "
            "take one snapshot per dataset."
        )
    per_dataset = data.noise(noise or {})
    with run_in(dtype):
        inputs = cast_tree(inputs, dtype)
        datasets = cast_tree(tuple(data.data), dtype)

        @jax.jit
        def every(inputs, datasets):
            def one(x):
                snapshots = data.snapshots(scene_of(x))
                total = sum(
                    model_loglike(s, d, **n)
                    for s, d, n in zip(snapshots, datasets, per_dataset)
                )
                return np.where(np.isfinite(total), total, -np.inf)

            return jax.lax.map(one, inputs, batch_size=batch_size)

        return onp.asarray(every(inputs, datasets), dtype=float)


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
        ``noise`` values of the ranking), in the same order.
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

    def _take(self, index):
        index = onp.asarray(index, dtype=int)
        return dataclasses.replace(
            self,
            orbits=tuple(self.orbits[k] for k in index),
            loglike=self.loglike[index],
            order=self.order[index],
        )


def _ranked(orbits, loglike, data):
    order = onp.argsort(-loglike, kind="stable")
    return RankedOrbits(
        orbits=tuple(orbits[k] for k in order),
        loglike=loglike[order],
        order=order,
        times=onp.asarray(data.times, dtype=onp.float64),
        resolution_mas=data.resolution_mas,
    )


def rank_orbits(
    model, data, orbits, *, noise=None, batch_size=None, dtype="float64"
):
    """Rank trial orbits by the log likelihood of multi-epoch data.

    Each orbit's scene is evaluated at every dataset's snapshot time and
    compared with all the data at once, by one compiled kernel mapped
    over the orbits. This judges trial orbits, e.g. from a Thiele–Innes
    grid on rough positions
    ([`starting_orbits`][virgil.orbits.starting_orbits]) or from prior
    draws, by the interferometric data themselves rather than by the
    positions that seeded them.

    Parameters
    ----------
    model : callable
        Maps an orbit to a time-dependent scene, e.g.
        ``lambda orbit: OrbitalBinary(orbit, 0.1)``. It is traced, so it
        must not convert the orbit's elements to Python floats; other
        parameters (fluxes, the shape of an extended component) are fixed
        at trial values.
    data : Epochs
        The data, one snapshot per dataset.
    orbits : sequence or orbit
        A list of orbits, the ``(orbit, χ²)`` pairs of
        ``starting_orbits``, a ``RankedOrbits``, or one orbit whose
        elements have a leading axis (e.g. prior draws). All must share
        one class and ``t_ref``.
    noise : dict, optional
        Noise values keyed by epoch or dataset name, as for
        [`Epochs.loglike`][virgil.epochs.Epochs.loglike]; by default
        the quoted errors.
    batch_size : int, optional
        Orbits evaluated together by ``jax.lax.map`` (default: all of them
        at once). Lower it if the data are large.
    dtype : {"float64", "float32"}, optional
        Precision of the evaluation (default float64, in a local
        ``jax.enable_x64`` context).

    Returns
    -------
    RankedOrbits
        The orbits sorted by log likelihood, best first. An orbit whose
        likelihood is not finite ranks last, with ``-inf``.

    Examples
    --------
    >>> candidates = starting_orbits(positions, periods)  # doctest: +SKIP
    >>> ranked = rank_orbits(lambda o: OrbitalBinary(o, 0.1), epochs,
    ...                      candidates)  # doctest: +SKIP
    >>> ranked.best, ranked.loglike[:3]  # doctest: +SKIP
    """
    listed = _orbit_list(orbits)
    loglike = _loglikes(model, _stack(listed), data, noise, batch_size, dtype)
    return _ranked(listed, loglike, data)


def _distinct(ranked, n, min_distance_mas):
    """Indices of up to ``n`` orbits of ``ranked``, best first, whose
    positions at the snapshot times differ from every better choice by
    more than ``min_distance_mas`` at some time."""
    finite = int(onp.sum(onp.isfinite(ranked.loglike)))
    if finite == 0:
        return []
    positions = ranked.positions()
    chosen = []
    for k in range(finite):
        offsets = [positions[k] - positions[j] for j in chosen]
        if all(
            onp.max(onp.hypot(o[:, 0], o[:, 1])) > min_distance_mas
            for o in offsets
        ):
            chosen.append(k)
            if len(chosen) == n:
                break
    return chosen


def chain_starts(ranked, n_chains, *, min_distance_mas=None):
    """The best distinct orbits of a ranking, one per chain.

    Starting every chain at one best fit hides other modes; starting
    chains in different good modes (the best few distinct orbits) lets
    the chains show whether the posterior is multimodal. Two orbits are
    the same mode when the companion's positions at every snapshot time
    agree to within ``min_distance_mas``: the mirror orbit (Ω + 180°,
    ω + 180°), which has the same sky motion, is the same mode, while a
    fringe alias or a period alias is not.

    Parameters
    ----------
    ranked : RankedOrbits
        From [`rank_orbits`][virgil.epochs.rank_orbits].
    n_chains : int
        Number of starts.
    min_distance_mas : float, optional
        The largest difference in position (mas) at any snapshot time for
        two orbits to count as one mode; by default half the finest
        resolution λ/B_max of the data.

    Returns
    -------
    RankedOrbits
        ``n_chains`` orbits, best first. If there are fewer distinct
        orbits than chains, the distinct orbits are repeated in turn.
        Map each orbit to your model's parameters (e.g. ``period``,
        ``dt_peri``) to start a chain there.
    """
    if n_chains < 1:
        raise ValueError(f"n_chains must be at least 1, got {n_chains}.")
    if min_distance_mas is None:
        min_distance_mas = 0.5 * ranked.resolution_mas
    chosen = _distinct(ranked, n_chains, min_distance_mas)
    if not chosen:
        raise ValueError("No orbit has a finite log likelihood.")
    return ranked._take([chosen[k % len(chosen)] for k in range(n_chains)])


@dataclasses.dataclass(frozen=True)
class EpochPositions:
    """The companion's position in each dataset, from
    [`epoch_positions`][virgil.epochs.epoch_positions].

    Attributes
    ----------
    names : tuple of str
        The datasets.
    mjd : numpy.ndarray
        Each dataset's snapshot time (MJD).
    dra, ddec : numpy.ndarray
        The fitted positions (mas, East and North).
    cov : numpy.ndarray
        The covariance of each position, shape ``(n, 2, 2)`` (mas²).
    flux : numpy.ndarray
        The fitted companion/primary flux of each dataset.
    gap : numpy.ndarray
        How decisive each dataset is: the log likelihood of the best
        position minus that of the best position more than ``gap_mas``
        from it. A small gap means another peak (a fringe alias, or a
        mirror image when closure phases are weak) fits almost as well.
    """

    names: tuple
    mjd: onp.ndarray
    dra: onp.ndarray
    ddec: onp.ndarray
    cov: onp.ndarray
    flux: onp.ndarray
    gap: onp.ndarray

    def decisive(self, min_gap):
        """Whether each dataset's ``gap`` exceeds ``min_gap``."""
        return self.gap > min_gap

    def positions(self, *, t_ref, min_gap=None):
        """The positions as [`PositionData`][virgil.orbits.PositionData],
        for [`starting_orbits`][virgil.orbits.starting_orbits].

        Parameters
        ----------
        t_ref : float
            The reference time (MJD) of the orbits fitted to them: give
            the ``t_ref`` of your model.
        min_gap : float, optional
            Keep only the datasets whose ``gap`` exceeds this. Seeding
            orbits with an indecisive dataset's position can start every
            orbit at the wrong peak; the visibility fit judges those
            datasets instead.
        """
        keep = (
            onp.ones(len(self.names), bool)
            if min_gap is None
            else self.decisive(min_gap)
        )
        if not keep.any():
            raise ValueError(
                f"No dataset is decisive: the gaps are {self.gap.round(1)}"
                f" and min_gap is {min_gap}."
            )
        return PositionData(
            self.mjd[keep],
            self.dra[keep],
            self.ddec[keep],
            self.cov[keep],
            t_ref=t_ref,
        )


def _binary_loglike(x, data):
    return model_loglike(BinaryModelCartesian(x[0], x[1], x[2]), data)


_binary_hessian = jax.jit(jax.hessian(lambda x, d: -_binary_loglike(x, d)))


@functools.partial(jax.jit, static_argnames="batch_size")
def _grid_loglike(points, data, batch_size):
    return jax.lax.map(
        lambda x: _binary_loglike(x, data), points, batch_size=batch_size
    )


def epoch_positions(data, grid, *, gap_mas=None, refine=True, batch_size=4096):
    """The companion's position in each dataset of a binary.

    For each dataset: the log likelihood of a static binary
    ([`BinaryModelCartesian`][virgil.models.BinaryModelCartesian]) on a
    grid of positions and fluxes, its best point, and (with ``refine``) a
    [`fit`][virgil.fitting.fit] from there, whose curvature gives the
    covariance of the position. These positions are only a starting
    point for a fit of the orbit to the visibilities: when the scene is
    more than two point stars they can be biased.

    Parameters
    ----------
    data : Epochs
        The data.
    grid : dict
        ``{"dra": axis, "ddec": axis, "flux": values}``: the positions
        (mas) and companion/primary fluxes to try. The position step
        should be finer than the resolution λ/B of the longest baseline.
    gap_mas : float, optional
        The distance (mas) beyond which another peak counts as a rival
        for ``gap``; by default each dataset's resolution λ/B_max.
    refine : bool, optional
        Refine each grid point with a fit (default). Otherwise the
        position is the grid point, with a covariance of one grid step
        squared.
    batch_size : int, optional
        Grid points evaluated together.

    Returns
    -------
    EpochPositions
    """
    if not isinstance(data, Epochs):
        raise TypeError(f"data must be an Epochs, not {type(data).__name__}.")
    missing = {"dra", "ddec", "flux"} - set(grid)
    if missing:
        raise ValueError(f"grid needs the axes {sorted(missing)}.")
    axes = [onp.asarray(grid[k], float) for k in ("dra", "ddec", "flux")]
    if (axes[2] <= 0).any():
        raise ValueError("The grid's fluxes must be positive.")
    xx, yy, ff = onp.meshgrid(*axes, indexing="ij")
    points = onp.stack([xx.ravel(), yy.ravel(), ff.ravel()], -1)
    steps = [
        float(onp.min(onp.diff(onp.sort(a)))) if a.size > 1 else 1.0
        for a in axes[:2]
    ]
    bounds = [(a.min() - s, a.max() + s) for a, s in zip(axes[:2], steps)]
    priors = {
        "dra": dist.Uniform(*bounds[0]),
        "ddec": dist.Uniform(*bounds[1]),
        "flux": dist.LogUniform(
            0.1 * float(axes[2].min()), max(1.0, 2 * float(axes[2].max()))
        ),
    }
    rows = []
    for name, d in zip(data.dataset_names, data.data):
        with run_in("float64"):
            d64 = cast_tree(d, "float64")
            loglike = _grid_loglike(
                np.asarray(points), d64, min(batch_size, len(points))
            )
            loglike = onp.asarray(loglike).reshape(xx.shape)
        loglike = onp.where(onp.isfinite(loglike), loglike, -onp.inf)
        by_position = loglike.max(axis=2)
        i, j = onp.unravel_index(onp.argmax(by_position), by_position.shape)
        k = int(onp.argmax(loglike[i, j]))
        best = onp.array([axes[0][i], axes[1][j], axes[2][k]])
        rival = (
            Epochs({name: d}).resolution_mas if gap_mas is None else gap_mas
        )
        far = onp.hypot(xx[..., 0] - best[0], yy[..., 0] - best[1]) > rival
        gap = (
            by_position[i, j] - by_position[far].max()
            if far.any()
            else onp.inf
        )
        cov = max(steps) ** 2 * onp.eye(2)
        if refine:
            result = fit(BinaryModelCartesian(*best), priors, d)
            best = onp.array(
                [float(result.values[p]) for p in ("dra", "ddec", "flux")]
            )
            with run_in("float64"):
                hess = _binary_hessian(np.asarray(best), d64)
            full = onp.linalg.pinv(onp.asarray(hess, dtype=float))
            if onp.all(onp.isfinite(full)) and onp.all(
                onp.linalg.eigvalsh(full[:2, :2]) > 0
            ):
                cov = full[:2, :2]
        rows.append((best, cov, gap))
    return EpochPositions(
        names=data.dataset_names,
        mjd=onp.asarray(data.times, dtype=onp.float64),
        dra=onp.array([r[0][0] for r in rows]),
        ddec=onp.array([r[0][1] for r in rows]),
        cov=onp.array([r[1] for r in rows]),
        flux=onp.array([r[0][2] for r in rows]),
        gap=onp.array([float(r[2]) for r in rows]),
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
        The fits to the visibilities from the distinct best candidates,
        lowest loss first.
    data : Epochs
        The data.
    """

    positions: EpochPositions
    candidates: RankedOrbits
    fits: tuple
    data: Epochs

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
        density.
        """
        floor = self.fits[0].info["loss"]
        kept, residuals = [], []
        for result in self.fits:
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
    noise=None,
    eccs=None,
    n_phase=36,
    n_candidates=200,
    n_refine=4,
    min_gap=5.0,
    refine_positions=True,
    batch_size=None,
    **fit_options,
):
    """Start an orbit fit to multi-epoch visibilities from a positions fit.

    The likelihood of visibilities is multimodal on the scale of the
    resolution λ/B (fringe aliases), and a sampler started from default
    values can stick in an alias. This builds a start in four steps:

    1. **Positions**: the companion's position in each dataset, from a
       grid and a binary fit
       ([`epoch_positions`][virgil.epochs.epoch_positions]).
    2. **Starting orbits**: a Thiele–Innes grid over period, eccentricity
       and time of periastron on the decisive datasets' positions
       ([`starting_orbits`][virgil.orbits.starting_orbits]).
    3. **Ranking** of those orbits by the likelihood of all the
       visibilities, with each orbit mapped to the model's parameters by
       ``start_values`` and the flux at the median of the positions fits.
    4. **Refinement** with [`fit`][virgil.fitting.fit] of the full model
       (noise terms included) from the ``n_refine`` best *distinct*
       orbits (as for [`chain_starts`][virgil.epochs.chain_starts]).

    Only the start comes from positions: nothing here enters the
    posterior. Sample from the best fit with numpyro's ``init_to_value``,
    or start one chain in each mode with
    [`OrbitStart.chain_values`][virgil.epochs.OrbitStart.chain_values] and
    [`chain_init_params`][virgil.likelihood.chain_init_params].

    Parameters
    ----------
    model : callable
        The scene function, called with the parameters in ``priors`` as
        keyword arguments and returning a time-dependent scene, as for
        [`Epochs.model_fn`][virgil.epochs.Epochs.model_fn].
    priors : dict
        The priors of the fit, as for ``fit``.
    data : Epochs
        The data.
    start_values : callable
        ``start_values(orbit, flux)``: the starting values (a dict with
        a value for every key of ``priors``) for a
        [`KeplerOrbit`][virgil.orbits.KeplerOrbit] (with ``t_ref``) and
        a companion/primary flux. It may use Python floats.
    grid : dict
        The per-dataset grid of positions and fluxes (see
        ``epoch_positions``).
    periods : array-like
        Trial periods (days) for ``starting_orbits``.
    t_ref : float
        The reference time (MJD) of the starting orbits: the ``t_ref`` of
        your model.
    noise : dict or list of dict, optional
        Noise terms of the refinement fits, as for ``fit`` (e.g. from
        [`Epochs.noise`][virgil.epochs.Epochs.noise]). The ranking uses
        the quoted errors.
    eccs, n_phase : optional
        Passed to ``starting_orbits``.
    n_candidates : int, optional
        Starting orbits to rank (default 200).
    n_refine : int, optional
        Distinct orbits to refine (default 4).
    min_gap : float, optional
        Seed orbits only with datasets whose positions are decisive: a
        log likelihood ``gap`` above this (see ``EpochPositions``).
    refine_positions : bool, optional
        Refine each grid position with a fit (see ``epoch_positions``).
    batch_size : int, optional
        Orbits ranked together (see ``rank_orbits``).
    **fit_options
        Passed to the refinement fits (e.g. ``method``, ``max_steps``).

    Returns
    -------
    OrbitStart
    """
    positions = epoch_positions(data, grid, refine=refine_positions)
    with run_in("float64"):
        seeds = positions.positions(t_ref=t_ref, min_gap=min_gap)
    if len(seeds.dt) < 2:
        raise ValueError(
            "At least two decisive datasets are needed to seed orbits; "
            f"the gaps are {positions.gap.round(1)}. Lower min_gap or "
            "refine the grid."
        )
    with run_in("float64"):
        candidates = starting_orbits(
            seeds, periods, eccs=eccs, n_phase=n_phase, n_best=n_candidates
        )
    flux = float(onp.median(positions.flux[positions.decisive(min_gap)]))
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
        lambda v: model(**v), stacked, data, None, batch_size, "float64"
    )
    ranked = _ranked(orbits, loglike, data)
    chosen = _distinct(ranked, n_refine, 0.5 * ranked.resolution_mas)
    if not chosen:
        raise ValueError("No starting orbit has a finite log likelihood.")
    fits = [
        fit(
            data.model_fn(model),
            priors,
            data.data,
            noise=noise,
            init=values[ranked.order[k]],
            **fit_options,
        )
        for k in chosen
    ]
    fits.sort(key=lambda result: result.info["loss"])
    return OrbitStart(
        positions=positions, candidates=ranked, fits=tuple(fits), data=data
    )
