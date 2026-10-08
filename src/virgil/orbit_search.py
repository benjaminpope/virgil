"""Scoring candidate orbits on multi-epoch data, with shared nuisances.

The first stage of an automatic orbit search scores many candidate orbits
on all the data at once. [`score_orbits`][virgil.orbit_search.score_orbits]
does this exactly, on the visibilities and closure phases, with the
nuisances that a single orbit shares across epochs shared in the score:

* **Flux.** One companion flux per band (or instrument), the same at
  every epoch of that band, optionally with a chromatic power law
  f(λ) = f₀ (λ/λ₀)^β. It is integrated out under a log-uniform
  (Jeffreys) prior, adaptively about its profiled peak, and its profiled
  value and uncertainty are reported.
  Letting each epoch have its own flux is a strictly looser model, whose
  extra freedom moves each epoch's peaks and makes new ones that no orbit
  with one flux visits, so it is never used to rank.
* **Error scales.** Each dataset's error scale is integrated out block by
  block, as in [`marginal_loglike`][virgil.epochs.marginal_loglike].
* **Calibration gains** set with
  [`OIData.with_gains`][virgil.oidata.OIData.with_gains] and closure-phase
  offsets set with
  [`OIData.with_closure_offsets`][virgil.oidata.OIData.with_closure_offsets]
  are truly per epoch. They enter the residuals linearly and are profiled
  analytically (each costs one degree of freedom), not marginalized, so
  that a block's χ² stays proportional to 1/s².

The companion's position is evaluated at every sample's own time, so
motion within a night needs no threshold. Extra likelihood terms, such as
radial velocities ([`RVData`][virgil.orbits.RVData]) or published
positions ([`PositionData`][virgil.orbits.PositionData]), are added to the
score.
"""

import copy
import dataclasses
import math

import equinox as eqx
import jax
import jax.numpy as np
import numpy as onp

from ._precision import cast_tree, run_in
from .epochs import (
    _SCALES,
    Epochs,
    _check_dof,
    _dof_for,
    _orbit_list,
    _stack,
    _Surface,
)
from .likelihood import _gain_jacobian, _whiten, inflated_errors
from .models import SourceModel
from .orbits import _Term

__all__ = [
    "OrbitScores",
    "SharedFlux",
    "rank_scores",
    "score_orbits",
]


# Default score quantum (nats) for breaking ties: scores closer than this
# rank by index, not by floating-point accident.
_QUANTUM = 1e-3


@dataclasses.dataclass(frozen=True)
class SharedFlux:
    """The companion flux shared across epochs, per band.

    Parameters
    ----------
    grid : tuple, array-like or dict, optional
        The coarse flux grid at the reference wavelength λ₀: ``(lo, hi,
        n)`` for ``n`` points spaced uniformly in ln f from ``lo`` to
        ``hi``, or the flux values themselves (one value fixes the flux),
        or a dict ``{band: either}``. The default, ``(1e-3, 1.0, 16)``, is
        used for every band not in the dict. The prior is log-uniform
        (Jeffreys) between the first and last points. The grid only finds
        the peak: the marginal is integrated adaptively about the profiled
        peak (see ``order``).
    bands : dict, optional
        ``{epoch or dataset name: band}``; a dataset's own entry overrides
        its epoch's. Every dataset of a band shares that band's flux. By
        default all datasets form one band, ``"all"``.
    reference : optional
        The band whose flux is at most 1 (default: the first band, in the
        order of the datasets). Bounding the flux in one band fixes which
        star is called the companion (a companion at r with flux f is
        the same in V² and closure phases as one at −r with flux 1/f), so
        that the labelling is the same in every band; the other bands may
        be given grids above 1.
    slope : tuple, array-like or dict, optional
        The chromatic slope β, as ``(lo, hi, n)`` (uniform prior) or the
        values, or a dict per band. None (the default) fixes β = 0.
    wavel0 : float or dict, optional
        The reference wavelength λ₀ (metres), one or per band; by default
        the geometric mean of each band's wavelengths.
    order : int, optional
        Gauss–Legendre nodes per side of the adaptive marginal (default
        16): each side of the profiled peak is integrated out to where the
        score has fallen by 12 nats (grown from the curvature's length
        scale, and at a prior bound also the slope's), cut at the prior's
        bounds, with the nodes crowded towards the peak. With a slope,
        ``order`` nodes per side over the whole β prior each integrate
        ln f from their own conditional peak.
    newton : int, optional
        Newton steps from the best coarse grid point to the profiled peak
        (default 8), each at most one grid step and kept only if it raises
        the score.
    """

    grid: object = (1e-3, 1.0, 16)
    bands: dict | None = None
    reference: object = None
    slope: object = None
    wavel0: object = None
    order: int = 16
    newton: int = 8

    def __post_init__(self):
        if int(self.order) < 2 or int(self.newton) < 0:
            raise ValueError("SharedFlux needs order >= 2 and newton >= 0.")


@dataclasses.dataclass(frozen=True)
class OrbitScores:
    """The scores of candidate orbits, in the order they were given.

    Returned by [`score_orbits`][virgil.orbit_search.score_orbits]. Rank
    them with :meth:`order` (or
    [`rank_scores`][virgil.orbit_search.rank_scores]).

    Attributes
    ----------
    score : numpy.ndarray
        ``(n,)``: the log likelihood with every shared flux (and slope)
        integrated out, error scales marginalized (or on the quoted
        errors), gains profiled, plus the extra terms. ``-inf`` where not
        finite.
    profiled : numpy.ndarray
        ``(n,)``: the same at the profiled flux (and slope) of each band,
        instead of integrated over them.
    flux, flux_err : numpy.ndarray
        ``(n, n_band)``: the profiled flux f₀ of each band (Newton steps
        in ln f from the best coarse grid point), and its uncertainty from
        the curvature in ln f there, σ_f = f σ_ln f (NaN with one grid
        point, or where the curvature is not negative).
    flux_at_edge : numpy.ndarray
        ``(n, n_band)`` bool: the profiled flux is at a prior bound.
    slope, slope_err, slope_at_edge : numpy.ndarray
        The same for the chromatic slope β (0 and NaN without a slope
        grid). With both free, the uncertainties are marginal (from the
        inverse of the 2 × 2 curvature).
    fallback : numpy.ndarray
        ``(n, n_band)`` bool: the curvature at the peak was not negative
        or not finite, so the marginal is the coarse grid's trapezoidal
        rule instead of the adaptive integral.
    terms : numpy.ndarray
        ``(n, n_terms)``: each extra term's log likelihood (included in
        ``score``).
    scale : numpy.ndarray
        ``(n, n_dataset)``: each dataset's largest block error scale
        ŝ = √(χ²/ν) at the profiled flux (and slope), on the quoted errors.
    scale_at_bound : numpy.ndarray
        ``(n, n_dataset)`` bool: a block's scale posterior presses against
        ``s_max`` (ŝ within one posterior width in ln s, 1/√(2 dof ν), of
        the bound), so its score is penalized by the bound: the candidate
        needs errors inflated past ``s_max`` there. Always False with
        ``s_max=None``.
    bands : tuple
        The band names, in column order; ``reference`` is the one whose
        flux is at most 1.
    reference : object
    cost : int
        The work units spent: candidates × Σ over datasets of the score
        evaluations of the band's marginal (grid points, Newton steps with
        their derivatives, and nodes).
    """

    score: onp.ndarray
    profiled: onp.ndarray
    flux: onp.ndarray
    flux_err: onp.ndarray
    flux_at_edge: onp.ndarray
    slope: onp.ndarray
    slope_err: onp.ndarray
    slope_at_edge: onp.ndarray
    fallback: onp.ndarray
    terms: onp.ndarray
    scale: onp.ndarray
    scale_at_bound: onp.ndarray
    bands: tuple
    reference: object
    cost: int

    def __len__(self):
        return len(self.score)

    def order(self, quantum=_QUANTUM):
        """Candidate indices, best first, ties broken by index (see
        [`rank_scores`][virgil.orbit_search.rank_scores])."""
        return rank_scores(self.score, quantum)


def rank_scores(score, quantum=_QUANTUM):
    """Indices that sort ``score`` from best to worst, deterministically.

    Scores are first rounded to multiples of ``quantum`` (nats), and equal
    rounded scores rank by index, so that two candidates differing only by
    rounding error (e.g. mirror-image grid cells) rank the same way on
    every platform and in either precision. Non-finite scores rank last.

    Parameters
    ----------
    score : array-like
        One score per candidate, higher is better.
    quantum : float, optional
        The rounding step (default 1e-3).

    Returns
    -------
    numpy.ndarray
        Integer indices, best first.

    Examples
    --------
    >>> rank_scores([1.0, 3.0, 3.0 + 1e-9, -float("inf"), 2.0]).tolist()
    [1, 2, 4, 0, 3]
    """
    if not float(quantum) > 0.0:
        raise ValueError(f"quantum must be positive, got {quantum}.")
    score = onp.asarray(score, dtype=onp.float64).reshape(-1)
    finite = onp.isfinite(score)
    level = onp.where(
        finite, onp.floor(onp.where(finite, score, 0.0) / quantum + 0.5), 0.0
    )
    index = onp.arange(score.size)
    # lexsort sorts by the last key first: non-finite last, then the
    # rounded score (descending), then the index.
    return onp.lexsort((index, -level, ~finite))


# ---------------------------------------------------------------------------
# Grids
# ---------------------------------------------------------------------------


def _axis(spec, log, what):
    """Grid values and normalized log trapezoid weights of one nuisance."""
    if isinstance(spec, tuple) and len(spec) == 3 and float(spec[2]) >= 1:
        lo, hi, n = float(spec[0]), float(spec[1]), int(spec[2])
        if n != spec[2]:
            raise ValueError(f"The {what} grid's size must be an integer.")
        if log and not 0.0 < lo <= hi:
            raise ValueError(f"The {what} grid needs 0 < lo <= hi.")
        if not lo <= hi:
            raise ValueError(f"The {what} grid needs lo <= hi.")
        x = (
            onp.linspace(math.log(lo), math.log(hi), n)
            if log
            else onp.linspace(lo, hi, n)
        )
    else:
        values = onp.atleast_1d(onp.asarray(spec, dtype=onp.float64))
        if values.ndim != 1 or not values.size:
            raise ValueError(f"The {what} grid must be 1-D and non-empty.")
        if log and not onp.all(values > 0):
            raise ValueError(f"The {what} grid must be positive.")
        x = onp.log(values) if log else values
        if onp.any(onp.diff(x) <= 0):
            raise ValueError(f"The {what} grid must be increasing.")
    if x.size == 1:
        return x, onp.zeros(1)
    gaps = onp.diff(x)
    weights = onp.zeros(x.size)
    weights[:-1] += gaps / 2
    weights[1:] += gaps / 2
    return x, onp.log(weights / weights.sum())


def _for_band(spec, band, default):
    if isinstance(spec, dict):
        return spec.get(band, default)
    return spec


@dataclasses.dataclass(frozen=True)
class _Band:
    """One band's coarse grid: ln f₀ and β per point, log trapezoid
    weights, shape, and the axes (whose ends bound the priors)."""

    name: object
    log_flux: onp.ndarray
    slope: onp.ndarray
    log_weight: onp.ndarray
    shape: tuple
    datasets: tuple
    wavel0: float
    flux_axis: onp.ndarray
    slope_axis: onp.ndarray

    @property
    def size(self):
        return int(self.log_flux.size)


def _bands(epochs, shared):
    """The bands, in order of first appearance among the datasets."""
    if not isinstance(shared, SharedFlux):
        raise TypeError(
            f"shared must be a SharedFlux, not {type(shared).__name__}."
        )
    mapping = shared.bands or {}
    unknown = set(mapping) - set(epochs.names) - set(epochs.dataset_names)
    if unknown:
        raise ValueError(f"bands for unknown names: {sorted(unknown)}.")
    band_of = []
    for name, epoch in zip(epochs.dataset_names, epochs.epoch_of):
        if name in mapping:
            band_of.append(mapping[name])
        elif epoch in mapping or not mapping:
            band_of.append(mapping.get(epoch, "all"))
        else:
            raise ValueError(
                f"Dataset {name!r} (epoch {epoch!r}) has no band; give "
                "every epoch or dataset one in SharedFlux(bands=...)."
            )
    names = tuple(dict.fromkeys(band_of))
    reference = names[0] if shared.reference is None else shared.reference
    if reference not in names:
        raise ValueError(
            f"The reference band {reference!r} is not among {list(names)}."
        )
    default = SharedFlux.grid
    bands = []
    for band in names:
        members = tuple(k for k, b in enumerate(band_of) if b == band)
        log_f, w_f = _axis(_for_band(shared.grid, band, default), True, "flux")
        if band == reference and log_f[-1] > 1e-12:
            raise ValueError(
                f"The reference band {band!r} must have flux at most 1 "
                f"(its grid reaches {math.exp(log_f[-1]):.3g}): bounding "
                "it fixes which star is the companion."
            )
        slope_spec = _for_band(shared.slope, band, None)
        if slope_spec is None:
            beta, w_b = onp.zeros(1), onp.zeros(1)
        else:
            beta, w_b = _axis(slope_spec, False, "slope")
        wavel0 = _for_band(shared.wavel0, band, None)
        if wavel0 is None:
            wavel = onp.concatenate(
                [
                    onp.broadcast_to(
                        onp.asarray(epochs.data[k].wavel, float),
                        onp.shape(epochs.data[k].u),
                    ).ravel()
                    for k in members
                ]
            )
            wavel0 = float(onp.exp(onp.mean(onp.log(wavel))))
        lf, bb = onp.meshgrid(log_f, beta, indexing="ij")
        bands.append(
            _Band(
                band,
                lf.ravel(),
                bb.ravel(),
                (w_f[:, None] + w_b[None, :]).ravel(),
                (log_f.size, beta.size),
                members,
                float(wavel0),
                log_f,
                beta,
            )
        )
    return tuple(bands), reference


# ---------------------------------------------------------------------------
# Per-dataset χ² with profiled gains and offsets
# ---------------------------------------------------------------------------


class _Fixed(SourceModel):
    """A 'model' returning given complex visibilities at the data's samples,
    so that the observables and residuals are computed as for any model."""

    cvis: jax.Array

    def model(self, u, v, wavel):
        return self.cvis


def _project_local(x, rows, cols):
    """``x`` minus its least-squares fit by block-local columns.

    ``rows`` ``(n_block, n_row)`` index ``x`` (padding out of range) and
    ``cols`` ``(n_block, n_row, n_mode)`` are the columns on those rows.
    """
    xb = x.at[rows].get(mode="fill", fill_value=0)
    coef = np.einsum("bmr,br->bm", np.linalg.pinv(cols), xb)
    fitted = np.einsum("brm,bm->br", cols, coef)
    return x.at[rows].add(-fitted, mode="drop")


def _profile(x, rows, cols, spanning=None):
    """``x`` minus its least-squares fit by all the columns: block-local
    ones, then (Frisch–Waugh) dense ones, made orthogonal to them."""
    x = _project_local(x, rows, cols)
    if spanning is None:
        return x
    span = jax.vmap(
        lambda s: _project_local(s, rows, cols), in_axes=1, out_axes=1
    )(spanning)
    return x - span @ (np.linalg.pinv(span) @ x)


def _local_rank(rows, cols, n):
    """The rank of block-local columns (host)."""
    return sum(
        int(onp.linalg.matrix_rank(c[r < n])) for r, c in zip(rows, cols)
    )


def _gain_rank(gains):
    """How many independent gains (host): the degrees of freedom lost."""
    rows, shapes = onp.asarray(gains.rows), onp.asarray(gains.shapes)
    rank = _local_rank(rows, shapes, gains.n_vis)
    if gains.spanning is None:
        return rank
    span = onp.asarray(gains.spanning, float)
    for r, c in zip(rows, shapes):
        inside = r < gains.n_vis
        r, c = r[inside], c[inside]
        if c.size:
            span[r] -= c @ (onp.linalg.pinv(c) @ span[r])
    return rank + int(onp.linalg.matrix_rank(span))


class _Scorer:
    """The χ² blocks of one dataset for given complex visibilities, with
    gains and closure-phase offsets profiled, and their score."""

    def __init__(self, data, s_max):
        self.gains = data.gains
        self.offsets = data.phase_offsets
        plain = eqx.tree_at(
            lambda d: (d.gains, d.phase_offsets),
            data,
            (None, None),
            is_leaf=lambda x: x is None,
        )
        if plain.extras:
            raise NotImplementedError(
                "score_orbits does not support extra observables "
                "(OIData.extras) yet; select the visibilities and closure "
                "phases (OIData.select(observables=('vis', 'phi')))."
            )
        self.plain = plain
        self.n_vis = int(onp.asarray(data.vis).size)
        self.n_phase = self.n_vis + int(onp.asarray(data.phi).size)
        surface = _Surface(plain, s_max)
        lost = {"vis": 0, "phi": 0}
        if self.gains is not None:
            lost["vis"] = _gain_rank(self.gains)
        if self.offsets is not None:
            sigma = onp.asarray(plain.flatten_data()[1], float)
            cols = onp.asarray(
                self.offsets._columns(
                    plain.cp_noise,
                    sigma[self.n_vis : self.n_phase],
                    onp.ones(len(self.offsets.widths)),
                )
            )
            lost["phi"] = _local_rank(
                onp.asarray(self.offsets.rows), cols, self.offsets.n_out
            )
        blocks = []
        for name, a, b, nu, von_mises in surface.blocks:
            nu = nu - lost.get(name, 0)
            if nu <= 0:
                raise ValueError(
                    f"The {name!r} block has no degrees of freedom left "
                    "after profiling its gains or offsets."
                )
            blocks.append((name, a, b, nu, von_mises))
        surface = copy.copy(surface)
        surface.blocks = tuple(blocks)
        surface.nu = onp.array([b[3] for b in blocks], dtype=float)
        self.surface = surface

    def chi2(self, cvis, plain, gains, offsets):
        """Each block's χ² on the quoted errors (traced data)."""
        prediction = plain.model(_Fixed(cvis))
        errors = inflated_errors(plain, prediction)
        r = _whiten(plain, prediction, plain.flatten_data()[0], errors)[0]
        n_vis = self.n_vis
        if gains is not None:
            jacobian = (
                _gain_jacobian(plain, prediction[:n_vis]) / errors[:n_vis]
            )
            local, spanning = gains._columns(
                jacobian, np.ones_like(gains.widths)
            )
            r = r.at[:n_vis].set(
                _profile(r[:n_vis], gains.rows, local, spanning)
            )
        if offsets is not None:
            stop = n_vis + offsets.n_out
            cols = offsets._columns(
                plain.cp_noise,
                errors[n_vis : self.n_phase],
                np.ones_like(offsets.widths),
            )
            rows = np.asarray(offsets.rows)
            r = r.at[n_vis:stop].set(_profile(r[n_vis:stop], rows, cols))
        return np.stack(
            [np.sum(r[a:b] ** 2) for _, a, b, _, _ in self.surface.blocks]
        )

    def score(self, chi2, dof, plain, scales):
        if scales == "quoted":
            return -0.5 * np.sum(chi2)
        return self.surface.score(chi2, dof, plain)


# ---------------------------------------------------------------------------
# The scorer
# ---------------------------------------------------------------------------


# The adaptive marginal: from the profiled peak, each side's range is
# doubled (at most _DOUBLINGS times, from two length scales) until the
# integrand has fallen by _DEPTH nats or the range reaches the prior's
# bound; Gauss–Legendre then integrates each side. Newton steps that do
# not raise the score are halved up to _HALVINGS times.
_DEPTH = 12.0
_DOUBLINGS = 8
_HALVINGS = 20
# With a slope, rounds of 1-D Newton steps in ln f and β before 2-D ones.
_ALTERNATIONS = 3
# Each side's nodes are spaced as u**_POWER for Gauss–Legendre nodes u.
_POWER = 3


def _newton(f, x, lo, hi, delta, steps):
    """Maximize the scalar ``f`` from ``x`` in ``[lo, hi]`` by Newton steps
    of at most ``delta``, halved until they raise ``f`` (the best of the
    halvings is taken, or none).

    Returns ``(x, f(x), f'(x), f''(x))``.
    """
    d1 = jax.grad(f)
    d2 = jax.grad(d1)
    shrink = 0.5 ** np.arange(_HALVINGS + 1, dtype=x.dtype)

    def step(_, carry):
        x, fx = carry
        g, h = d1(x), d2(x)
        # Where the curvature is not negative, a gradient step instead.
        step = -g / np.where(h < 0, h, -1.0)
        step = np.clip(step, -delta, delta)
        trials = np.clip(x + step * shrink, lo, hi)
        values = jax.vmap(f)(trials)
        values = np.where(np.isnan(values), -np.inf, values)
        k = np.argmax(values)
        better = values[k] > fx
        return np.where(better, trials[k], x), np.where(better, values[k], fx)

    # A rolled loop: one compiled step, however many steps.
    x, fx = jax.lax.fori_loop(0, steps, step, (x, f(x)))
    return x, fx, d1(x), d2(x)


def _scale(g, h, x, lo, hi):
    """The integrand's length scale at a peak ``x`` (``h`` the curvature,
    and at a bound also the slope ``g``), and whether it is usable."""
    at_bound = (x <= lo) | (x >= hi)
    rate = np.sqrt(np.maximum(-h, 0.0)) + np.where(at_bound, np.abs(g), 0.0)
    ok = np.isfinite(rate) & (rate > 0)
    return 1.0 / np.where(ok, rate, 1.0), ok


def _nodes(f, x, fx, scale, lo, hi, order):
    """Gauss–Legendre nodes and log weights over each side of the peak
    ``x``, each side's range grown until ``f`` falls by ``_DEPTH``, in the
    variable u with x = x̂ ± L u³."""
    t, w = (
        np.asarray(v, x.dtype) for v in onp.polynomial.legendre.leggauss(order)
    )
    side = np.asarray([-1.0, 1.0], x.dtype)
    room = np.stack([x - lo, hi - x])

    def double(_, length):
        drop = fx - jax.vmap(f)(x + side * length)
        grow = ~(drop >= _DEPTH) & (length < room)
        return np.where(grow, np.minimum(2.0 * length, room), length)

    # Both sides at once, in a rolled loop (one compiled evaluation).
    length = jax.lax.fori_loop(
        0, _DOUBLINGS, double, np.minimum(2.0 * scale, room)
    )
    # x = x̂ ± L u³ crowds the nodes near the peak, so that a narrow spike
    # on a broad shoulder (a block whose χ² nearly vanishes) is resolved
    # together with the shoulder.
    u = 0.5 * (t + 1.0)
    nodes = x + (side * length)[:, None] * u**_POWER
    log_w = (
        np.log(0.5 * w * _POWER * u ** (_POWER - 1)) + np.log(length)[:, None]
    )
    return nodes.ravel(), log_w.ravel()


def _marginal_1d(f, x0, lo, hi, delta, shared):
    """Profile and integrate ``f`` over one nuisance with a uniform prior on
    ``[lo, hi]``: ``(log marginal, x̂, f(x̂), σ, ok)``."""
    x, fx, g, h = _newton(f, x0, lo, hi, delta, shared.newton)
    scale, ok = _scale(g, h, x, lo, hi)
    nodes, log_w = _nodes(f, x, fx, scale, lo, hi, shared.order)
    values = jax.vmap(f)(nodes)
    values = np.where(np.isnan(values), -np.inf, values)
    log_z = jax.nn.logsumexp(values + log_w) - math.log(hi - lo)
    sigma = np.where(h < 0, 1.0 / np.sqrt(np.where(h < 0, -h, 1.0)), np.nan)
    return log_z, x, fx, sigma, ok & np.isfinite(log_z)


def _band_marginal(band, shared, score, grid_scores):
    """One band's marginal over its flux (and slope), its profile and the
    profiled values: ``(marginal, profiled, (f, σ_f, f_edge, β, σ_β,
    β_edge, fallback))``.

    The coarse grid's best point starts Newton steps to the profiled peak;
    the marginal then integrates each side of it by Gauss–Legendre, over a
    range grown from the curvature's length scale until the score has
    fallen by ``_DEPTH`` nats (cut at the prior's bounds), so that it does
    not depend on where the peak falls between grid points. With a slope,
    an outer rule over the whole slope prior takes, at each node, this 1-D
    marginal over ln f. Where the peak's curvature is not negative (and it
    is not at a bound with a slope), the trapezoidal rule on the coarse
    grid is used instead and flagged.
    """
    finite = np.where(np.isnan(grid_scores), -np.inf, grid_scores)
    coarse = jax.nn.logsumexp(finite + band.log_weight)
    best = np.argmax(finite)
    n_f, n_b = band.shape
    lf0 = np.asarray(band.log_flux, finite.dtype)[best]
    b0 = np.asarray(band.slope, finite.dtype)[best]
    f_axis, b_axis = band.flux_axis, band.slope_axis
    free_f, free_b = n_f > 1, n_b > 1
    nan = np.asarray(np.nan, finite.dtype)
    no = np.asarray(False)

    def bounds(axis):
        lo, hi = float(axis[0]), float(axis[-1])
        delta = float(onp.max(onp.diff(axis)))
        return lo, hi, delta

    def edge(x, axis):
        return (x <= axis[0]) | (x >= axis[-1])

    if not free_f and not free_b:
        value = score(lf0, b0)
        return value, value, (np.exp(lf0), nan, no, b0, nan, no, no)
    if free_f != free_b:
        axis = f_axis if free_f else b_axis

        def f(x):
            return score(x, b0) if free_f else score(lf0, x)

        lo, hi, delta = bounds(axis)
        log_z, x, fx, sigma, ok = _marginal_1d(
            f, lf0 if free_f else b0, lo, hi, delta, shared
        )
        marginal = np.where(ok, log_z, coarse)
        if free_f:
            out = (np.exp(x), np.exp(x) * sigma, edge(x, axis), b0, nan, no)
        else:
            out = (np.exp(lf0), nan, no, x, sigma, edge(x, axis))
        return marginal, fx, (*out, ~ok)

    # Both free: Newton in (ln f, β), then an outer Gauss–Legendre over β
    # about the peak and, at each β node, the 1-D adaptive marginal over
    # ln f from the conditional peak.
    f_lo, f_hi, f_delta = bounds(f_axis)
    b_lo, b_hi, b_delta = bounds(b_axis)
    lo = np.asarray([f_lo, b_lo], finite.dtype)
    hi = np.asarray([f_hi, b_hi], finite.dtype)
    delta = np.asarray([f_delta, b_delta], finite.dtype)

    def f2(v):
        return score(v[0], v[1])

    # Far from the peak the 2-D curvature is often not negative definite:
    # first alternate 1-D Newton steps in ln f and β, then polish in 2-D.
    def alternate(_, carry):
        lf, beta = carry
        lf = _newton(
            lambda x: score(x, beta), lf, f_lo, f_hi, f_delta, shared.newton
        )[0]
        beta = _newton(
            lambda x: score(lf, x), beta, b_lo, b_hi, b_delta, shared.newton
        )[0]
        return lf, beta

    lf, beta = jax.lax.fori_loop(0, _ALTERNATIONS, alternate, (lf0, b0))
    grad, hess = jax.grad(f2), jax.hessian(f2)
    shrink = 0.5 ** np.arange(_HALVINGS + 1, dtype=finite.dtype)

    def newton_2d(_, carry):
        v, fv = carry
        g, h = grad(v), hess(v)
        neg = (h[0, 0] < 0) & (np.linalg.det(h) > 0)
        step = np.where(
            neg, -np.linalg.solve(np.where(neg, h, -np.eye(2)), g), g
        )
        step = step / np.maximum(1.0, np.max(np.abs(step) / delta))
        trials = np.clip(v + shrink[:, None] * step, lo, hi)
        values = jax.vmap(f2)(trials)
        values = np.where(np.isnan(values), -np.inf, values)
        k = np.argmax(values)
        better = values[k] > fv
        return np.where(better, trials[k], v), np.where(better, values[k], fv)

    v = np.stack([lf, beta])
    v, fv = jax.lax.fori_loop(0, shared.newton, newton_2d, (v, f2(v)))
    g, h = grad(v), hess(v)
    neg = (h[0, 0] < 0) & (np.linalg.det(h) > 0)
    cov = -np.linalg.inv(np.where(neg, h, -np.eye(2)))
    sigma = np.where(neg, np.sqrt(np.abs(np.diagonal(cov))), np.nan)
    at_b = (v[1] <= b_lo) | (v[1] >= b_hi)
    rate = 1.0 / sigma[1] + np.where(at_b, np.abs(g[1]), 0.0)
    ok = neg & np.isfinite(rate) & (rate > 0)
    tilt = np.where(neg, cov[0, 1] / cov[1, 1], 0.0)

    # The outer nodes span the whole slope prior on each side of the peak
    # (a drop measured from a narrow spike would cut off a broad shoulder
    # that carries much of the integral); the u³ spacing still resolves
    # the peak.
    b_nodes, b_log_w = _nodes(
        lambda beta: fv, v[1], fv, np.asarray(b_hi - b_lo, v.dtype),
        b_lo, b_hi, shared.order,
    )  # fmt: skip

    def inner(beta):
        start = np.clip(v[0] + tilt * (beta - v[1]), f_lo, f_hi)
        log_z, _, _, _, ok = _marginal_1d(
            lambda x: score(x, beta), start, f_lo, f_hi, f_delta, shared
        )
        return log_z, ok

    inner_z, inner_ok = jax.vmap(inner)(b_nodes)
    log_z = jax.nn.logsumexp(inner_z + b_log_w) - math.log(b_hi - b_lo)
    ok = ok & np.all(inner_ok) & np.isfinite(log_z)
    flux = np.exp(v[0])
    return (
        np.where(ok, log_z, coarse),
        fv,
        (
            flux,
            flux * sigma[0],
            edge(v[0], f_axis),
            v[1],
            sigma[1],
            edge(v[1], b_axis),
            ~ok,
        ),
    )


def _evaluations(band, shared):
    """Score evaluations per dataset of ``band`` (work units), counting a
    first or second derivative as one evaluation."""
    n_f, n_b = band.shape
    newton = 1 + shared.newton * (3 + _HALVINGS) + 2
    sides = 2 * (_DOUBLINGS + shared.order)
    one_d = newton + sides
    if n_f > 1 and n_b > 1:
        start = 2 * _ALTERNATIONS * newton
        return band.size + start + newton + sides + 2 * shared.order * one_d
    if n_f > 1 or n_b > 1:
        return band.size + one_d
    return 1


def _term_function(term):
    """``f(orbit) -> log likelihood`` for an extra term."""
    if isinstance(term, _Term):
        return lambda orbit: term.loglike({"orbit": orbit})
    if callable(term):
        return term
    raise TypeError(
        "Each term must be a likelihood term (RVData.term, "
        "PositionData.term) or a callable orbit -> log likelihood, not "
        f"{type(term).__name__}."
    )


def _candidates(orbits):
    """The candidates stacked along a leading axis, and their number."""
    if isinstance(orbits, (list, tuple)):
        stacked = _stack(_orbit_list(orbits))
    else:
        stacked = orbits
    leaves = jax.tree_util.tree_leaves(stacked)
    if not leaves:
        raise ValueError("There are no orbits to score.")
    if onp.ndim(leaves[0]) == 0:
        stacked = jax.tree_util.tree_map(
            lambda x: np.asarray(x)[None], stacked
        )
    n = int(onp.shape(jax.tree_util.tree_leaves(stacked)[0])[0])
    if n == 0:
        raise ValueError("There are no orbits to score.")
    return stacked, n


def _first(stacked, k):
    return jax.tree_util.tree_map(lambda x: x[k], stacked)


def _cvis_at_times(epochs, k, data, scene):
    """The scene's complex visibilities at every sample's own time, or at
    the dataset's snapshot time when the data carry no times."""
    if data.dt is None:
        return data._cvis(scene.at(float(epochs.times[k])))
    return data._cvis(scene)


def _wavel(data):
    return onp.broadcast_to(
        onp.asarray(data.wavel, onp.float64), onp.shape(data.u)
    )


def _fixed_parts(epochs, model, stacked, n, dtype):
    """Per dataset: the companion-free numerator A = V(0) T(0) and the
    total fluxes T(0), T(1), checked to be the same for the first and
    last candidate, and the scene checked to be linear in the flux."""
    parts = []
    with run_in(dtype):
        stacked = cast_tree(stacked, dtype)
        first, last = _first(stacked, 0), _first(stacked, n - 1)
        for k, d in enumerate(epochs.data):
            d = cast_tree(d, dtype)
            wavel = np.asarray(_wavel(d), dtype)
            values = []
            for orbit in (first, last):
                s0, s1 = model(orbit, 0.0), model(orbit, 1.0)
                if not getattr(s1, "time_dependent", False):
                    raise TypeError(
                        "model(orbit, flux) must return a scene that moves "
                        "with time (e.g. OrbitalBinary(orbit, flux))."
                    )
                t0 = np.broadcast_to(s0.total_spectrum(wavel), wavel.shape)
                t1 = np.broadcast_to(s1.total_spectrum(wavel), wavel.shape)
                a = _cvis_at_times(epochs, k, d, s0) * t0
                values.append((a, t0, t1))
            (a, t0, t1), (a2, t02, t12) = values
            scale = float(np.max(np.abs(a))) or 1.0
            tol = 1e-4 if dtype == "float32" else 1e-9
            if (
                float(np.max(np.abs(a - a2))) > tol * scale
                or not onp.allclose(t0, t02, rtol=tol)
                or not onp.allclose(t1, t12, rtol=tol)
            ):
                raise ValueError(
                    "model(orbit, 0) must not depend on the orbit: the "
                    "scene without the companion (and the total flux) is "
                    "computed once per dataset."
                )
            if k == 0:
                half = model(first, 0.5)
                th = np.broadcast_to(half.total_spectrum(wavel), wavel.shape)
                b = _cvis_at_times(epochs, k, d, model(first, 1.0)) * t1 - a
                expect = (a + 0.5 * b) / (t0 + 0.5 * (t1 - t0))
                got = _cvis_at_times(epochs, k, d, half)
                if (
                    not onp.allclose(
                        onp.asarray(th),
                        onp.asarray(t0 + 0.5 * (t1 - t0)),
                        rtol=tol,
                    )
                    or float(np.max(np.abs(got - expect))) > 10 * tol
                ):
                    raise ValueError(
                        "model(orbit, flux) must be linear in the flux (a "
                        "component weight): its unnormalized visibility "
                        "and total flux must be affine in it."
                    )
            parts.append(
                (onp.asarray(a), onp.asarray(t0), onp.asarray(t1 - t0))
            )
    return parts


def _peak_scales(scorers, chi2_peak, dofs, s_max):
    """Each dataset's largest block scale ŝ = √(χ²/ν) at the profiled
    flux, ``(n, n_dataset)``, and whether its posterior presses against
    ``s_max``: ŝ is within one posterior width in ln s, 1/√(2 dof ν), of
    the bound (always False without ``s_max``)."""
    scale, at_bound = [], []
    for scorer, chi2, dof in zip(scorers, chi2_peak, dofs):
        nu = scorer.surface.nu
        s_hat = onp.sqrt(onp.asarray(chi2, float) / nu)
        scale.append(s_hat.max(axis=1))
        if s_max is None:
            at_bound.append(onp.zeros(len(s_hat), bool))
        else:
            reach = s_hat * onp.exp(1.0 / onp.sqrt(2.0 * dof * nu))
            at_bound.append(onp.any(reach >= float(s_max), axis=1))
    return onp.stack(scale, axis=1), onp.stack(at_bound, axis=1)


def score_orbits(
    epochs,
    model,
    orbits,
    *,
    shared=None,
    terms=(),
    scales="marginal",
    dof=1.0,
    s_max=5.0,
    batch_size=None,
    max_evaluations=None,
    dtype="float64",
):
    """Score candidate orbits on all epochs, with the flux shared.

    For each candidate, the companion's position is computed at every
    sample's own time, and the visibilities ``g`` of a unit companion
    there once. Every evaluation at another flux is then cheap
    arithmetic on ``g``: for a scene linear in the companion's flux f
    (a component weight), the complex visibility is
    ``V(f) = (A + f B) / (T₀ + f ΔT)``, with ``A`` and the total fluxes
    computed once per dataset and ``B`` from ``g``. With a slope, f
    differs per sample: f = f₀ (λ/λ₀)^β.

    For each dataset and flux the score is the scale-marginalized
    log likelihood ``m = -Σ_b (ν_b/2) ln χ²_b`` of
    [`marginal_loglike`][virgil.epochs.marginal_loglike] (or, with
    ``s_max``, its bounded form; with ``scales="quoted"``, ``-χ²/2``).
    Gains ([`OIData.with_gains`][virgil.oidata.OIData.with_gains]) and
    closure-phase offsets
    ([`OIData.with_closure_offsets`][virgil.oidata.OIData.with_closure_offsets])
    are profiled analytically inside χ²: their widths are ignored, and each
    independent mode removes one degree of freedom from its block (a
    single gain per dataset is ``with_gains(modes=onp.ones((1, n)))``, with
    ``n`` the number of samples).

    Each band's flux is then integrated out under its log-uniform prior.
    The coarse grid only finds the peak: Newton steps in ln f (and β)
    profile it, and Gauss–Legendre nodes on each side of it, out to a
    12-nat drop or the prior's bounds, integrate it, so that the marginal does not
    depend on where the peak falls between grid points (a well-measured
    flux, σ_ln f ≈ 0.02, is far narrower than any affordable grid step).
    Where the curvature at the peak is not usable, the coarse grid's
    trapezoidal rule is used and ``OrbitScores.fallback`` is set. The
    extra terms are then added.

    Parameters
    ----------
    epochs : Epochs
        The data. Datasets without sample times (``mjd``) are evaluated at
        their snapshot time (``Epochs.times``).
    model : callable
        ``model(orbit, flux)`` returns the time-dependent scene, e.g.
        [`OrbitalBinary`][virgil.models.OrbitalBinary] itself. It must be
        linear in ``flux`` (a component weight), and the scene at flux 0
        must not depend on the orbit; both are checked on the first and
        last candidates.
    orbits : sequence or orbit
        A list of orbits (or ``starting_orbits`` pairs), or one orbit whose
        elements have a leading candidate axis. All share one class and
        ``t_ref``; times are measured from ``t_ref``, in float64 on the
        host, so float32 keeps them precise.
    shared : SharedFlux, optional
        The shared flux grids, bands and slopes (default ``SharedFlux()``:
        one band, f in [1e-3, 1] on 16 log-spaced points, no slope).
    terms : sequence, optional
        Extra log likelihood terms: likelihood terms from
        [`RVData.term`][virgil.orbits.RVData.term] or
        [`PositionData.term`][virgil.orbits.PositionData.term], whose
        ``params`` (or ``orbit``) callable receives ``{"orbit": orbit}``,
        or callables ``orbit -> log likelihood``. Each brings its own
        error model (e.g. a fixed RV jitter).
    scales : {"marginal", "quoted"}, optional
        Integrate each dataset's error scales out (default), or use the
        quoted errors (the score is then ``-χ²/2``, up to a constant).
    dof, s_max : optional
        The effective-dof fraction (one number, or a dict keyed by dataset
        or epoch name) and the bound on the scales, as for
        ``marginal_loglike`` (each block's likelihood integrated over
        ln s in [-ln s_max, ln s_max], log-uniform). The default
        ``s_max=5`` leaves about twice the error inflation of typical
        interferometric data: a candidate that needs more than that on some
        dataset is penalized, and flagged in ``OrbitScores.scale_at_bound``.
        The bound also keeps the score finite: with ``s_max=None`` a block's
        -(ν/2) ln χ² is unbounded as χ² → 0, so a block with few degrees of
        freedom (one triangle's closure phases, say) that a flux or slope
        fits almost exactly makes a narrow spike in the score, which can
        dominate the profile (``profiled``, ``flux``, ``slope``). A finite
        ``s_max`` bounds each block's gain at about ν ln s_max.
    batch_size : int, optional
        Candidates evaluated together by ``jax.lax.map`` (default: all).
        The result does not depend on it.
    max_evaluations : int, optional
        A budget in work units: candidates × Σ over datasets of the score
        evaluations of the dataset's band's marginal (coarse grid points,
        Newton steps counting each derivative as one, and nodes). The cost is predicted before
        anything is evaluated, and a cost over budget raises a
        ``ValueError``.
    dtype : {"float64", "float32"}, optional
        Precision of the evaluation (default float64, in a local
        ``jax.enable_x64`` context).

    Returns
    -------
    OrbitScores
        Scores in the order of ``orbits``; rank them with
        ``OrbitScores.order()``.

    Examples
    --------
    >>> scores = score_orbits(epochs, OrbitalBinary, candidates,
    ...                       shared=SharedFlux((0.01, 1.0, 16)))  # doctest: +SKIP
    >>> best = scores.order()[0]  # doctest: +SKIP
    >>> scores.flux[best], scores.flux_err[best]  # doctest: +SKIP
    """
    if not isinstance(epochs, Epochs):
        raise TypeError(
            f"epochs must be an Epochs, not {type(epochs).__name__}."
        )
    if scales not in _SCALES:
        raise ValueError(f"scales must be one of {_SCALES}, not {scales!r}.")
    shared = SharedFlux() if shared is None else shared
    bands, reference = _bands(epochs, shared)
    stacked, n = _candidates(orbits)
    cost = n * sum(
        _evaluations(band, shared) * len(band.datasets) for band in bands
    )
    if max_evaluations is not None and cost > int(max_evaluations):
        raise ValueError(
            f"Scoring {n} candidates would cost {cost} work units "
            "(candidates × Σ over datasets of the score evaluations of "
            "their band's flux marginal), "
            f"over the budget max_evaluations={int(max_evaluations)}. "
            "Score fewer candidates, or use coarser flux grids."
        )
    term_fns = tuple(_term_function(t) for t in terms)
    scorers = tuple(_Scorer(d, s_max) for d in epochs.data)
    dofs = onp.asarray(
        [_check_dof(_dof_for(dof, epochs, k)) for k in range(len(epochs))]
    )
    fixed = _fixed_parts(epochs, model, stacked, n, dtype)
    ratios = tuple(
        _wavel(d) / bands[[k in b.datasets for b in bands].index(True)].wavel0
        for k, d in enumerate(epochs.data)
    )

    with run_in(dtype):
        args = cast_tree(
            (
                stacked,
                tuple(s.plain for s in scorers),
                tuple(s.gains for s in scorers),
                tuple(s.offsets for s in scorers),
                tuple(fixed),
                tuple(onp.log(r) for r in ratios),
                tuple((b.log_flux, b.slope) for b in bands),
                onp.asarray(dofs, float),
            ),
            dtype,
        )

        @jax.jit
        def every(
            stacked, plains, gains, offsets, fixed, log_ratio, grids, dofs
        ):
            def one(orbit):
                b_parts = []
                for k, plain in enumerate(plains):
                    a, t0, _ = fixed[k]
                    scene = model(orbit, 1.0)
                    t1 = t0 + fixed[k][2]
                    b_parts.append(
                        _cvis_at_times(epochs, k, plain, scene) * t1 - a
                    )
                marginals, best, summaries = [], [], []
                chi2_peak = [None] * len(plains)
                for band, (log_flux, slope) in zip(bands, grids):

                    def at(lf, beta, band=band):
                        total = 0.0
                        for k in band.datasets:
                            a, t0, dt = fixed[k]
                            f = np.exp(lf + beta * log_ratio[k])
                            cvis = (a + f * b_parts[k]) / (t0 + f * dt)
                            s = scorers[k]
                            chi2 = s.chi2(
                                cvis, plains[k], gains[k], offsets[k]
                            )
                            total = total + s.score(
                                chi2, dofs[k], plains[k], scales
                            )
                        return total

                    scores = jax.vmap(at)(log_flux, slope)
                    m, top, summary = _band_marginal(band, shared, at, scores)
                    for k in band.datasets:
                        a, t0, dt = fixed[k]
                        f = summary[0] * np.exp(summary[3] * log_ratio[k])
                        cvis = (a + f * b_parts[k]) / (t0 + f * dt)
                        chi2_peak[k] = scorers[k].chi2(
                            cvis, plains[k], gains[k], offsets[k]
                        )
                    marginals.append(m)
                    best.append(top)
                    summaries.append(summary)
                # Never a zero-size output: lax.map's remainder reshapes it.
                extra = (
                    np.stack([np.asarray(f(orbit)) for f in term_fns])
                    if term_fns
                    else np.zeros((1,), log_flux.dtype)
                )
                extra_sum = np.sum(extra) if term_fns else 0.0
                score = sum(marginals) + extra_sum
                profiled = sum(best) + extra_sum
                columns = [np.stack(c) for c in zip(*summaries)]
                return (
                    np.where(np.isfinite(score), score, -np.inf),
                    np.where(np.isfinite(profiled), profiled, -np.inf),
                    *columns,
                    extra,
                    tuple(chi2_peak),
                )

            return jax.lax.map(one, stacked, batch_size=batch_size)

        out = every(*args)
    chi2_peak = out[-1]
    out = [onp.asarray(x) for x in out[:-1]]
    scale, scale_at_bound = _peak_scales(scorers, chi2_peak, dofs, s_max)
    (score, profiled, flux, flux_err, f_edge, slope, slope_err, b_edge,
     fallback) = out[:9]  # fmt: skip
    return OrbitScores(
        score=score.astype(float),
        profiled=profiled.astype(float),
        flux=flux.astype(float),
        flux_err=flux_err.astype(float),
        flux_at_edge=f_edge.astype(bool),
        slope=slope.astype(float),
        slope_err=slope_err.astype(float),
        slope_at_edge=b_edge.astype(bool),
        fallback=fallback.astype(bool),
        terms=out[9][:, : len(term_fns)].astype(float),
        scale=scale,
        scale_at_bound=scale_at_bound,
        bands=tuple(b.name for b in bands),
        reference=reference,
        cost=int(cost),
    )
