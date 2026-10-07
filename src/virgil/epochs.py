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
import warnings

import jax
import jax.numpy as np
import numpy as onp
from jax.scipy.special import i0e

from ._precision import cast_tree, run_in
from .fitting import fit
from .likelihood import (
    _gaussian_loglike,
    _whitened_and_errors,
    model_loglike,
    whitened_residuals,
)
from .models import BinaryModelCartesian
from .orbits import PositionData, starting_orbits


__all__ = [
    "Epochs",
    "EpochPositions",
    "OrbitStart",
    "RankedOrbits",
    "chain_starts",
    "epoch_positions",
    "marginal_loglike",
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


def _blocks(data):
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


# The bounded (s_max) marginal integrates over ln s adaptively: scans of
# _N_SCAN nodes narrow the range to where the integrand is within
# exp(-depth) of its peak, then _N_QUADRATURE Gauss–Legendre nodes
# integrate it there.
_N_SCAN = 64
_N_NARROW = 4
_N_QUADRATURE = 64
_LEGENDRE = onp.polynomial.legendre.leggauss(_N_QUADRATURE)


def _log_integral(log_f, lo, hi, guess):
    """``log ∫ exp(log_f(t)) dt`` over ``[lo, hi]``, for ``log_f`` mapping
    an array of ``t`` to the log integrand, which may be steep.

    A fixed grid in t misses steep edges (a likelihood cut off sharply by
    a bound, falling by many e-folds per grid step) and narrow peaks. Here
    each scan evaluates ``log_f`` on a uniform grid over the current
    window (plus ``guess``, the expected peak) and keeps the span of nodes
    within ``depth`` e-folds of the maximum, widened by one node on each
    side, so that the window brackets every part of the integrand that
    matters. Gauss–Legendre then integrates over the final window, all in
    log space. The window does not carry gradients: its edges are where
    the integrand is ~exp(-depth) of its peak, or the fixed bounds.
    """
    guess = jax.lax.stop_gradient(guess)
    dtype = guess.dtype
    depth = -onp.log(float(np.finfo(dtype).eps)) + 10.0
    lo = np.asarray(lo, dtype)
    hi = np.asarray(hi, dtype)
    a, b = lo, hi
    uniform = np.linspace(0.0, 1.0, _N_SCAN, dtype=dtype)
    for k in range(_N_NARROW):
        nodes = a + (b - a) * uniform
        if k == 0:
            nodes = np.sort(
                np.concatenate([nodes, np.clip(guess, lo, hi)[None]])
            )
        values = jax.lax.stop_gradient(log_f(jax.lax.stop_gradient(nodes)))
        values = np.where(np.isnan(values), -np.inf, values)
        keep = values >= np.max(values) - depth
        index = np.arange(nodes.size)
        first = np.min(np.where(keep, index, nodes.size))
        last = np.max(np.where(keep, index, -1))
        a = nodes[np.maximum(first - 1, 0)]
        b = nodes[np.minimum(last + 1, nodes.size - 1)]
    x, w = (np.asarray(v, dtype) for v in _LEGENDRE)
    half = 0.5 * (b - a)
    t = a + half * (x + 1.0)
    return jax.nn.logsumexp(log_f(t) + np.log(w)) + np.log(half)


def _check_dof(dof):
    dof = float(dof)
    if not 0.0 < dof <= 1.0:
        raise ValueError(
            f"The effective-dof fraction must be in (0, 1], got {dof}."
        )
    return dof


def _check_marginal_data(data):
    """Raise if ``data`` has a model-dependent likelihood normalizer, which
    the scale-marginalized surface does not include."""
    found = [
        what
        for what, present in (
            ("gains (OIData.with_gains)", data.gains is not None),
            (
                "closure-phase offsets (OIData.with_closure_offsets)",
                data.phase_offsets is not None,
            ),
            (
                "extra observables with a model-dependent covariance",
                bool(data.has_model_covariance),
            ),
        )
        if present
    ]
    if found:
        raise NotImplementedError(
            "The scale-marginalized surface (scales='marginal', "
            "epoch_positions, marginal_loglike) needs a likelihood whose "
            "normalization depends only on the error scales; these data "
            f"have {' and '.join(found)}, whose covariance a scale does not "
            "multiply. Use the data without them, or scales='quoted'."
        )


class _Surface:
    """The scale-marginalized score of one dataset, block by block.

    Only the layout of the data enters (the blocks, their ν and
    ``s_max``), so that one compiled kernel serves every dataset of one
    shape: the hash and equality, used as a static jit argument, never
    depend on data values. ``score(chi2, dof, data)`` maps the blocks' χ²
    on the quoted errors to ``m = -Σ_b (ν_b/2) ln χ²_b`` (Gaussian
    normalization in every block) or, with ``s_max``, to the log of each
    block's likelihood integrated over ln s in [-ln s_max, ln s_max]
    (uniform in ln s: the Jeffreys prior, bounded), with the exact von
    Mises normalization for uncorrelated closure phases (whose quoted
    errors are read from the traced ``data``). ``dof`` (a fraction in
    (0, 1], traced) tempers every block's log likelihood: ν_eff = dof · ν.
    """

    def __init__(self, data, s_max=None):
        _check_marginal_data(data)
        if s_max is not None and not float(s_max) > 1.0:
            raise ValueError(f"s_max must exceed 1, got {s_max}.")
        self.blocks = _blocks(data)
        self.names = tuple(b[0] for b in self.blocks)
        self.nu = onp.array([b[3] for b in self.blocks], dtype=float)
        self.s_max = None if s_max is None else float(s_max)

    def _key(self):
        return (self.blocks, self.s_max)

    def __hash__(self):
        return hash(self._key())

    def __eq__(self, other):
        return isinstance(other, _Surface) and self._key() == other._key()

    def chi2(self, model, data, noise=None):
        """χ² of each block of ``data`` for ``model``, on the quoted errors."""
        r = whitened_residuals(model, data, **(noise or {}))
        return np.stack([np.sum(r[a:b] ** 2) for _, a, b, _, _ in self.blocks])

    def score(self, chi2, dof, data):
        dof = np.asarray(dof, chi2.dtype)
        if self.s_max is None:
            tiny = np.finfo(chi2.dtype).tiny
            nu = np.asarray(self.nu, chi2.dtype)
            return -0.5 * dof * np.sum(nu * np.log(np.maximum(chi2, tiny)))
        bound = onp.log(self.s_max)
        total = 0.0
        for k, (_, _, _, nu, von_mises) in enumerate(self.blocks):

            def log_f(log_s, k=k, nu=nu, von_mises=von_mises):
                loglike = -0.5 * chi2[k] * np.exp(-2.0 * log_s)
                if not von_mises:
                    return dof * (loglike - nu * log_s)
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
                return dof * loglike

            # The Gaussian block's peak, ln ŝ = ln √(χ²/ν).
            tiny = np.finfo(chi2.dtype).tiny
            guess = 0.5 * np.log(np.maximum(chi2[k], tiny) / nu)
            total = total + _log_integral(log_f, -bound, bound, guess)
        return total

    def scales(self, chi2):
        """ŝ_b = √(χ²_b/ν_b), the profile maximum of each block's scale."""
        return onp.sqrt(onp.asarray(chi2, float) / self.nu)


def marginal_loglike(model, data, *, dof=1.0, s_max=None, **noise):
    """The log likelihood with each block's error scale marginalized.

    Each observable block *b* of ``data`` (visibilities, phases, and each
    extra observable) has an unknown factor ``s_b`` on its quoted errors.
    Integrating it out under its Jeffreys prior ``1/s_b`` gives, up to a
    constant,

        m = -Σ_b (ν_b/2) ln χ²_b,

    where χ²_b is the block's χ² on the quoted errors (from
    [`whitened_residuals`][virgil.likelihood.whitened_residuals]) and
    ν_b its number of independent observables
    ([`n_independent`][virgil.oidata.OIData.n_independent], split by
    block). Profiling ``s_b`` gives the same function, at
    ŝ_b² = χ²_b/ν_b. ``m`` does not change when any block's errors are
    multiplied by a constant, so its differences between models (e.g.
    the gap between two peaks) do not depend on how well the errors were
    quoted. Near a peak, m ≈ -χ²/(2ŝ²): differences are those of the
    quoted-error log likelihood divided by ŝ².

    Every block uses the Gaussian (small-σ) normalization, including
    uncorrelated closure phases, whose likelihood in
    [`model_loglike`][virgil.likelihood.model_loglike] is a von Mises
    density; for those the unbounded marginal would be improper as
    s → ∞. ``m`` is a search surface, not a replacement for
    ``model_loglike``. Where sσ is not small (weak closure phases with a
    large scale), pass ``s_max``.

    Data whose likelihood has a model-dependent normalization are refused
    with a ``NotImplementedError``: gains
    ([`OIData.with_gains`][virgil.oidata.OIData.with_gains]), closure-phase
    offsets
    ([`OIData.with_closure_offsets`][virgil.oidata.OIData.with_closure_offsets])
    and extra observables with a model-dependent covariance. Their
    marginalized nuisance covariance is not multiplied by an error scale,
    so a block's χ² is not ∝ 1/s² and the scale would not be integrated
    out. The same holds for ``epoch_positions`` and
    ``rank_orbits(scales="marginal")``.

    Parameters
    ----------
    model : SourceModel
        Model to evaluate.
    data : OIData
        Data to compare with.
    dof : float, optional
        The effective number of degrees of freedom as a fraction of ν_b
        (default 1). Errors correlated beyond the model of the quoted
        errors carry fewer than ν_b degrees of freedom, and ``m`` with
        ν_b is then over-confident by about ν_b/ν_eff; a ``dof`` below 1
        (e.g. calibrated by a residual bootstrap) scales every difference
        of ``m`` down by that factor.
    s_max : float, optional
        Bound each scale to [1/s_max, s_max] and integrate it out
        numerically, with the exact von Mises normalization for
        uncorrelated closure phases: ``m`` is then Σ_b ln ∫ L_b(s)^dof
        d ln s over the bounds. The integral is adaptive in ln s (scans
        narrow it to where the integrand matters, then Gauss–Legendre),
        so it stays accurate where the likelihood falls steeply from a
        bound. This is finite and proper where the unbounded marginal is
        not.
    **noise
        Other noise terms (e.g. ``wavel_scale``), as for
        ``whitened_residuals``. ``vis_scale`` and ``phi_scale`` should be
        left out: the scales are integrated out.

    Returns
    -------
    float
    """
    surface = _Surface(data, s_max)
    return surface.score(
        surface.chi2(model, data, noise), _check_dof(dof), data
    )


_SCALES = ("quoted", "marginal")


def _check_scales(scales, name):
    if scales is None:
        warnings.warn(
            f"{name} ranks with the quoted errors when scales is not given; "
            "the default will become scales='marginal' (each dataset's error "
            "scales integrated out, see marginal_loglike). Pass "
            "scales='quoted' or scales='marginal' to choose.",
            FutureWarning,
            stacklevel=3,
        )
        return "quoted"
    if scales not in _SCALES:
        raise ValueError(f"scales must be one of {_SCALES}, not {scales!r}.")
    return scales


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
    if scales == "marginal":
        surfaces = tuple(_Surface(d, s_max) for d in data.data)
        dofs = np.asarray(
            [_check_dof(_dof_for(dof, data, k)) for k in range(len(data))]
        )
        with run_in(dtype):
            inputs = cast_tree(inputs, dtype)
            datasets = cast_tree(tuple(data.data), dtype)

            @jax.jit
            def every_marginal(inputs, datasets, dofs):
                def one(x):
                    snapshots = data.snapshots(scene_of(x))
                    total = sum(
                        f.score(f.chi2(s, d, n), dofs[k], d)
                        for k, (s, d, n, f) in enumerate(
                            zip(snapshots, datasets, per_dataset, surfaces)
                        )
                    )
                    return np.where(np.isfinite(total), total, -np.inf)

                return jax.lax.map(one, inputs, batch_size=batch_size)

            dofs = np.asarray(dofs, dtype)
            return onp.asarray(
                every_marginal(inputs, datasets, dofs), dtype=float
            )
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
    model,
    data,
    orbits,
    *,
    scales=None,
    noise=None,
    dof=1.0,
    s_max=None,
    batch_size=None,
    dtype="float64",
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
    scales : {"quoted", "marginal"}, optional
        How the errors are scaled. ``"quoted"`` ranks by
        ``Epochs.loglike`` on the quoted errors (with ``noise``), so the
        datasets with the most underestimated errors dominate the
        ranking. ``"marginal"`` integrates each dataset's error scale out,
        block by block (visibilities, phases), under its Jeffreys prior,
        and ranks by the sum of
        [`marginal_loglike`][virgil.epochs.marginal_loglike]: the
        ranking then does not change when one dataset's errors are
        rescaled. Data with gains, closure-phase offsets or a
        model-dependent covariance are refused (see ``marginal_loglike``). The default is ``"quoted"`` for now, with a
        ``FutureWarning`` when ``scales`` is not given; it will become
        ``"marginal"``.
    noise : dict, optional
        Noise values keyed by epoch or dataset name, as for
        [`Epochs.loglike`][virgil.epochs.Epochs.loglike]; by default
        the quoted errors. With ``scales="marginal"``, error scales are
        integrated out and only the other terms (e.g. ``wavel_scale``)
        matter.
    dof, s_max : optional
        For ``scales="marginal"``: the effective-dof fraction (one
        number, or a dict keyed by dataset or epoch name) and the bound
        on the scales, as for ``marginal_loglike``.
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
    ...                      candidates, scales="marginal")  # doctest: +SKIP
    >>> ranked.best, ranked.loglike[:3]  # doctest: +SKIP
    """
    scales = _check_scales(scales, "rank_orbits")
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

    Positions, covariances and ``gap_marginal`` come from the
    scale-marginalized surface
    ([`marginal_loglike`][virgil.epochs.marginal_loglike]), on which each
    observable block's error scale is integrated out: they do not change
    when a dataset's quoted errors, or only its closure-phase errors, are
    multiplied by a constant.

    Attributes
    ----------
    names : tuple of str
        The datasets.
    mjd : numpy.ndarray
        Each dataset's snapshot time (MJD).
    dra, ddec : numpy.ndarray
        The fitted positions (mas, East and North).
    cov : numpy.ndarray
        The covariance of each position, shape ``(n, 2, 2)`` (mas²), from
        the curvature of the scale-marginalized surface.
    flux : numpy.ndarray
        The fitted companion/primary flux of each dataset (at most 1).
    gap : numpy.ndarray
        The gap on the **quoted** errors, as before ``gap_marginal`` was
        added: the log likelihood of the best grid position minus that of
        the best grid position more than ``gap_mas`` from it. It grows as
        1/s² when the errors are underestimated by s, so an ambiguous
        dataset can look decisive. Kept unchanged for one release, after
        which ``gap`` will hold the marginal value.
    gap_marginal : numpy.ndarray
        How decisive each dataset is, on the scale-marginalized surface:
        the same difference between the best grid position and its best
        rival more than ``gap_mas`` away, with each block's error scale
        integrated out. A small gap means another peak (a fringe alias,
        or a mirror image when closure phases are weak) fits almost as
        well. ``decisive`` and ``min_gap`` compare this.
    chi2_raw : tuple of dict
        For each dataset, χ²/N on the quoted errors at the fitted
        position, N = ``n_independent``: per block (``"vis"``, ``"phi"``,
        ``"extra[k]"``) and for the whole dataset (``"all"``). A value
        well above 1 means the quoted errors are too small; it is not
        made 1 by the scale marginalization.
    scale : tuple of dict
        For each dataset, the fitted error scale ŝ = √(χ²/ν) of each
        block (``"vis_scale"``, ``"phi_scale"``, ``"extra[k]_scale"``).
    """

    names: tuple
    mjd: onp.ndarray
    dra: onp.ndarray
    ddec: onp.ndarray
    cov: onp.ndarray
    flux: onp.ndarray
    gap: onp.ndarray
    gap_marginal: onp.ndarray
    chi2_raw: tuple
    scale: tuple

    def decisive(self, min_gap):
        """Whether each dataset's ``gap_marginal`` exceeds ``min_gap``."""
        return self.gap_marginal > min_gap

    def positions(self, *, t_ref, min_gap=None):
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
        """
        keep = (
            onp.ones(len(self.names), bool)
            if min_gap is None
            else self.decisive(min_gap)
        )
        if not keep.any():
            raise ValueError(
                "No dataset is decisive: the marginal gaps are "
                f"{self.gap_marginal.round(1)} and min_gap is {min_gap}."
            )
        return PositionData(
            self.mjd[keep],
            self.dra[keep],
            self.ddec[keep],
            self.cov[keep],
            t_ref=t_ref,
        )


def _binary(x):
    return BinaryModelCartesian(x[0], x[1], x[2])


@functools.partial(jax.jit, static_argnames=("surface", "batch_size"))
def _grid_scores(points, data, surface, batch_size):
    """The quoted-error log likelihood and the blocks' χ² at each point."""

    def one(x):
        whitened, errors = _whitened_and_errors(_binary(x), data, {})
        chi2 = np.stack(
            [np.sum(whitened[a:b] ** 2) for _, a, b, _, _ in surface.blocks]
        )
        return _gaussian_loglike(whitened, errors), chi2

    return jax.lax.map(one, points, batch_size=batch_size)


@functools.partial(jax.jit, static_argnames="surface")
def _grid_marginal(chi2, dof, data, surface):
    """The scale-marginalized score of each grid point's block χ²."""
    return jax.vmap(lambda c: surface.score(c, dof, data))(chi2)


def _negative_marginal(x, dof, data, surface):
    return -surface.score(surface.chi2(_binary(x), data), dof, data)


_negative_marginal_and_grad = jax.jit(
    jax.value_and_grad(_negative_marginal), static_argnames="surface"
)
_negative_marginal_hessian = jax.jit(
    jax.hessian(_negative_marginal), static_argnames="surface"
)
_block_chi2 = jax.jit(
    lambda x, data, surface: surface.chi2(_binary(x), data),
    static_argnames="surface",
)


def _gap(score, xx, yy, rival):
    """The best position of a (dra, ddec, flux) grid of scores, and its
    score minus that of the best point more than ``rival`` from it."""
    by_position = score.max(axis=2)
    i, j = onp.unravel_index(onp.argmax(by_position), by_position.shape)
    far = onp.hypot(xx[..., 0] - xx[i, j, 0], yy[..., 0] - yy[i, j, 0]) > rival
    gap = by_position[i, j] - by_position[far].max() if far.any() else onp.inf
    return (i, j, int(onp.argmax(score[i, j]))), float(gap)


def _refine_marginal(x0, dof, d64, surface, bounds):
    """Maximize the scale-marginalized surface from ``x0`` (L-BFGS-B in
    float64, deterministic) within ``bounds``."""
    from scipy.optimize import minimize

    def fun(x):
        value, grad = _negative_marginal_and_grad(
            np.asarray(x), dof, d64, surface
        )
        return float(value), onp.asarray(grad, dtype=float)

    result = minimize(
        fun,
        onp.clip(x0, [b[0] for b in bounds], [b[1] for b in bounds]),
        jac=True,
        method="L-BFGS-B",
        bounds=bounds,
        options={"ftol": 1e-15, "gtol": 1e-10, "maxiter": 500},
    )
    return onp.asarray(result.x, dtype=float), float(result.fun)


def epoch_positions(
    data,
    grid,
    *,
    gap_mas=None,
    refine=True,
    dof=1.0,
    s_max=None,
    batch_size=4096,
):
    """The companion's position in each dataset of a binary.

    For each dataset: a static binary
    ([`BinaryModelCartesian`][virgil.models.BinaryModelCartesian]) on a
    grid of positions and fluxes, scored on the scale-marginalized
    surface m = -Σ_b (ν_b/2) ln χ²_b
    ([`marginal_loglike`][virgil.epochs.marginal_loglike]), in which the
    error scale of each observable block (visibilities, closure phases)
    is integrated out under its Jeffreys prior. Its best point is
    refined (with ``refine``) by maximizing m, which is the fit with a
    free ``vis_scale`` and ``phi_scale`` per dataset, profiled; the
    curvature of m there gives the covariance of the position. The
    positions, covariances and ``gap_marginal`` therefore do not depend
    on how well each block's errors were quoted, and nights whose errors
    are underestimated do not look more decisive than they are. These
    positions are only a starting point for a fit of the orbit to the
    visibilities: when the scene is more than two point stars they can be
    biased.

    The raw χ²/N on the quoted errors and the fitted scale of each block
    are recorded, and a ``UserWarning`` names the datasets whose raw
    χ²/N exceeds 4 (errors underestimated by more than 2): their orbit
    fits need fitted error scales, and a rescaled χ²/N ≈ 1 does not make
    them good fits.

    Parameters
    ----------
    data : Epochs
        The data. Datasets with gains, closure-phase offsets or a
        model-dependent covariance are refused (see ``marginal_loglike``).
    grid : dict
        ``{"dra": axis, "ddec": axis, "flux": values}``: the positions
        (mas) and companion/primary fluxes to try. The position step
        should be finer than the resolution λ/B of the longest baseline.
        Fluxes must be in (0, 1]: a binary with flux ratio f > 1 at r is
        the one with 1/f at -r, so f > 1 would only add a twin of every
        peak.
    gap_mas : float, optional
        The distance (mas) beyond which another peak counts as a rival
        for ``gap`` and ``gap_marginal``; by default each dataset's
        resolution λ/B_max.
    refine : bool, optional
        Refine each grid point by maximizing m (default). Otherwise the
        position is the grid point, with a covariance of one grid step
        squared.
    dof : float or dict, optional
        The effective-dof fraction ν_eff/ν of each dataset's blocks (one
        number, or a dict keyed by dataset or epoch name; default 1),
        for errors correlated beyond the quoted ones (see
        ``marginal_loglike``). It multiplies ``gap_marginal`` by ``dof``
        and ``cov`` by 1/dof.
    s_max : float, optional
        Bound the scales and integrate them out numerically (see
        ``marginal_loglike``), for weak closure phases with a large
        scale.
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
    rows, inflated, edge = [], [], []
    for index, (name, d) in enumerate(zip(data.dataset_names, data.data)):
        surface = _Surface(d, s_max)
        dof_k = _check_dof(_dof_for(dof, data, index))
        with run_in("float64"):
            d64 = cast_tree(d, "float64")
            loglike, chi2 = _grid_scores(
                np.asarray(points), d64, surface, min(batch_size, len(points))
            )
            marginal = _grid_marginal(chi2, np.asarray(dof_k), d64, surface)
            loglike = onp.asarray(loglike).reshape(xx.shape)
            marginal = onp.asarray(marginal).reshape(xx.shape)
        loglike = onp.where(onp.isfinite(loglike), loglike, -onp.inf)
        marginal = onp.where(onp.isfinite(marginal), marginal, -onp.inf)
        rival = (
            Epochs({name: d}).resolution_mas if gap_mas is None else gap_mas
        )
        _, gap = _gap(loglike, xx, yy, rival)
        (i, j, k), gap_marginal = _gap(marginal, xx, yy, rival)
        if i in (0, xx.shape[0] - 1) or j in (0, xx.shape[1] - 1):
            edge.append(name)
        best = onp.array([axes[0][i], axes[1][j], axes[2][k]])
        cov = max(steps) ** 2 * onp.eye(2)
        with run_in("float64"):
            if refine:
                refined, value = _refine_marginal(
                    best, np.asarray(dof_k), d64, surface, bounds
                )
                if -value >= marginal[i, j, k]:
                    best = refined
                hess = _negative_marginal_hessian(
                    np.asarray(best), np.asarray(dof_k), d64, surface
                )
                full = onp.linalg.pinv(onp.asarray(hess, dtype=float))
                if onp.all(onp.isfinite(full)) and onp.all(
                    onp.linalg.eigvalsh(full[:2, :2]) > 0
                ):
                    cov = full[:2, :2]
            chi2_best = onp.asarray(
                _block_chi2(np.asarray(best), d64, surface), dtype=float
            )
        per_block = dict(zip(surface.names, chi2_best / surface.nu))
        per_block["all"] = float(chi2_best.sum() / surface.nu.sum())
        if per_block["all"] > 4.0:
            inflated.append(f"{name} ({per_block['all']:.1f})")
        scale = {
            f"{n}_scale": float(v)
            for n, v in zip(surface.names, surface.scales(chi2_best))
        }
        rows.append((best, cov, gap, gap_marginal, per_block, scale))
    if inflated:
        warnings.warn(
            "Raw chi2/N on the quoted errors exceeds 4 (errors underestimated "
            f"by more than 2) in {', '.join(inflated)}: fit error scales for "
            "these datasets, and do not read a rescaled chi2/N of 1 as a "
            "good fit.",
            UserWarning,
            stacklevel=2,
        )
    if edge:
        warnings.warn(
            f"The best grid position of {', '.join(edge)} is at the edge of "
            "the grid: the companion may lie outside it. Widen the grid.",
            UserWarning,
            stacklevel=2,
        )
    return EpochPositions(
        names=data.dataset_names,
        mjd=onp.asarray(data.times, dtype=onp.float64),
        dra=onp.array([r[0][0] for r in rows]),
        ddec=onp.array([r[0][1] for r in rows]),
        cov=onp.array([r[1] for r in rows]),
        flux=onp.array([r[0][2] for r in rows]),
        gap=onp.array([float(r[2]) for r in rows]),
        gap_marginal=onp.array([float(r[3]) for r in rows]),
        chi2_raw=tuple(r[4] for r in rows),
        scale=tuple(r[5] for r in rows),
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
    scales=None,
    noise=None,
    dof=1.0,
    s_max=None,
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
    scales : {"quoted", "marginal"}, optional
        How step 3 ranks the candidates (see ``rank_orbits``):
        ``"quoted"`` on the quoted errors, where the datasets with the
        most underestimated errors dominate, or ``"marginal"`` with each
        dataset's per-block error scale integrated out, so that each
        dataset counts by its own fitted scale ŝ. The default is
        ``"quoted"`` for now, with a ``FutureWarning`` when ``scales`` is
        not given; it will become ``"marginal"``.
    noise : dict or list of dict, optional
        Noise terms of the refinement fits, as for ``fit`` (e.g. from
        [`Epochs.noise`][virgil.epochs.Epochs.noise]). The ranking does
        not use them.
    dof, s_max : optional
        Passed to ``epoch_positions`` and, with ``scales="marginal"``, to
        the ranking.
    eccs, n_phase : optional
        Passed to ``starting_orbits``.
    n_candidates : int, optional
        Starting orbits to rank (default 200).
    n_refine : int, optional
        Distinct orbits to refine (default 4).
    min_gap : float, optional
        Seed orbits only with datasets whose positions are decisive: a
        ``gap_marginal`` above this (see ``EpochPositions``). This
        compares the scale-marginalized gap, so a night whose errors are
        underestimated no longer passes it with a gap inflated by s².
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
    scales = _check_scales(scales, "start_from_positions")
    positions = epoch_positions(
        data, grid, refine=refine_positions, dof=dof, s_max=s_max
    )
    with run_in("float64"):
        seeds = positions.positions(t_ref=t_ref, min_gap=min_gap)
    if len(seeds.dt) < 2:
        raise ValueError(
            "At least two decisive datasets are needed to seed orbits; "
            f"the marginal gaps are {positions.gap_marginal.round(1)}. Lower "
            "min_gap or "
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
        lambda v: model(**v),
        stacked,
        data,
        None,
        batch_size,
        "float64",
        scales=scales,
        dof=dof,
        s_max=s_max,
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
