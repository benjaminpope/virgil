"""Scoring candidate orbits on multi-epoch data, with shared nuisances.

The first stage of an automatic orbit search scores many candidate orbits
on all the data at once. [`score_orbits`][virgil.orbit_search.score_orbits]
does this exactly, on the visibilities and closure phases, with the
nuisances that a single orbit shares across epochs shared in the score:

* **Flux.** One companion flux per band (or instrument), the same at
  every epoch of that band, optionally with a chromatic power law
  f(λ) = f₀ (λ/λ₀)^β. It is integrated out on a small log-uniform
  (Jeffreys) grid, and its profiled value and uncertainty are reported.
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
from .epochs import Epochs, _check_dof, _dof_for, _Surface
from .likelihood import _gain_jacobian, _whiten, inflated_errors
from .models import SourceModel
from .orbits import _Term

__all__ = [
    "OrbitScores",
    "SharedFlux",
    "rank_scores",
    "score_orbits",
]

_SCALES = ("quoted", "marginal")

# Default score quantum (nats) for breaking ties: scores closer than this
# rank by index, not by floating-point accident.
_QUANTUM = 1e-3


@dataclasses.dataclass(frozen=True)
class SharedFlux:
    """The companion flux shared across epochs, per band.

    Parameters
    ----------
    grid : tuple, array-like or dict, optional
        The flux grid at the reference wavelength λ₀: ``(lo, hi, n)`` for
        ``n`` points spaced uniformly in ln f from ``lo`` to ``hi``, or the
        flux values themselves (one value fixes the flux), or a dict
        ``{band: either}``. The default, ``(1e-3, 1.0, 16)``, is used for
        every band not in the dict. The prior is log-uniform (Jeffreys)
        between the first and last points, integrated by the trapezoidal
        rule in ln f.
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
    """

    grid: object = (1e-3, 1.0, 16)
    bands: dict | None = None
    reference: object = None
    slope: object = None
    wavel0: object = None


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
        ``(n,)``: the same at the best grid point of each band, instead of
        integrated over the grid.
    flux, flux_err : numpy.ndarray
        ``(n, n_band)``: the profiled flux f₀ of each band, refined by a
        parabola in ln f through the best grid point and its neighbours,
        and its uncertainty from that parabola's curvature (NaN at a grid
        edge, with one grid point, or where the curvature is not
        negative).
    flux_at_edge : numpy.ndarray
        ``(n, n_band)`` bool: the best grid point is the first or last.
    slope, slope_err, slope_at_edge : numpy.ndarray
        The same for the chromatic slope β (0 and NaN without a slope
        grid). The uncertainties are conditional, each at the other's best
        grid value.
    terms : numpy.ndarray
        ``(n, n_terms)``: each extra term's log likelihood (included in
        ``score``).
    bands : tuple
        The band names, in column order; ``reference`` is the one whose
        flux is at most 1.
    reference : object
    cost : int
        The work units spent: candidates × Σ over datasets of the flux
        points of the dataset's band.
    """

    score: onp.ndarray
    profiled: onp.ndarray
    flux: onp.ndarray
    flux_err: onp.ndarray
    flux_at_edge: onp.ndarray
    slope: onp.ndarray
    slope_err: onp.ndarray
    slope_at_edge: onp.ndarray
    terms: onp.ndarray
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
    """One band's grid: ln f₀ and β per point, log weights, and shape."""

    name: object
    log_flux: onp.ndarray
    slope: onp.ndarray
    log_weight: onp.ndarray
    shape: tuple
    datasets: tuple
    wavel0: float

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


def _parabola(x, y, i):
    """Vertex and 1σ (from the curvature) of the parabola through the best
    grid point ``i`` of ``(x, y)`` and its neighbours; edge flag."""
    n = x.shape[0]
    if n < 3:
        edge = np.asarray(False) if n == 1 else (i == 0) | (i == n - 1)
        return x[i], np.asarray(np.nan, x.dtype), edge
    j = np.clip(i, 1, n - 2)
    x0, x1, x2 = x[j - 1], x[j], x[j + 1]
    y0, y1, y2 = y[j - 1], y[j], y[j + 1]
    d1 = (y1 - y0) / (x1 - x0)
    d2 = (y2 - y1) / (x2 - x1)
    a = (d2 - d1) / (x2 - x0)
    b = d1 - a * (x0 + x1)
    edge = (i == 0) | (i == n - 1)
    ok = (a < 0) & ~edge & np.isfinite(a) & np.isfinite(b)
    safe = np.where(ok, a, -1.0)
    vertex = np.clip(-b / (2.0 * safe), x0, x2)
    return (
        np.where(ok, vertex, x[i]),
        np.where(ok, 1.0 / np.sqrt(-2.0 * safe), np.nan),
        edge,
    )


def _band_summary(band, scores, log_flux, slope, log_weight):
    """Marginal, best, and refined (f₀, σ_f, β, σ_β, edges) of one band."""
    finite = np.where(np.isnan(scores), -np.inf, scores)
    marginal = jax.nn.logsumexp(finite + log_weight)
    best = np.argmax(finite)
    n_f, n_b = band.shape
    grid = finite.reshape(n_f, n_b)
    i_f, i_b = best // n_b, best % n_b
    lf, f_err, f_edge = _parabola(
        log_flux.reshape(n_f, n_b)[:, 0], grid[:, i_b], i_f
    )
    beta, b_err, b_edge = _parabola(
        slope.reshape(n_f, n_b)[0, :], grid[i_f, :], i_b
    )
    flux = np.exp(lf)
    return (
        marginal,
        finite[best],
        (flux, flux * f_err, f_edge, beta, b_err, b_edge),
    )


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
        from .epochs import _orbit_list, _stack

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


def score_orbits(
    epochs,
    model,
    orbits,
    *,
    shared=None,
    terms=(),
    scales="marginal",
    dof=1.0,
    s_max=None,
    batch_size=None,
    max_evaluations=None,
    dtype="float64",
):
    """Score candidate orbits on all epochs, with the flux shared.

    For each candidate, the companion's position is computed at every
    sample's own time, and the visibilities ``g`` of a unit companion
    there once. Every point of each band's flux grid is then cheap
    arithmetic on ``g``: for a scene linear in the companion's flux f
    (a component weight), the complex visibility is
    ``V(f) = (A + f B) / (T₀ + f ΔT)``, with ``A`` and the total fluxes
    computed once per dataset and ``B`` from ``g``. With a slope, f
    differs per sample: f = f₀ (λ/λ₀)^β.

    For each dataset and grid point the score is the scale-marginalized
    log likelihood ``m = -Σ_b (ν_b/2) ln χ²_b`` of
    [`marginal_loglike`][virgil.epochs.marginal_loglike] (or, with
    ``s_max``, its bounded form; with ``scales="quoted"``, ``-χ²/2``).
    Gains ([`OIData.with_gains`][virgil.oidata.OIData.with_gains]) and
    closure-phase offsets
    ([`OIData.with_closure_offsets`][virgil.oidata.OIData.with_closure_offsets])
    are profiled analytically inside χ²: their widths are ignored, and each
    independent mode removes one degree of freedom from its block (a
    single gain per dataset is ``with_gains(modes=onp.ones((1, n)))``, with
    ``n`` the number of samples). Each band's flux grid is then integrated
    out under its log-uniform prior, ``ln Σ_j w_j exp(Σ_d m_d(f_j))``, and
    the extra terms are added.

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
        ``marginal_loglike``.
    batch_size : int, optional
        Candidates evaluated together by ``jax.lax.map`` (default: all).
        The result does not depend on it.
    max_evaluations : int, optional
        A budget in work units, candidates × Σ over datasets of the grid
        points of the dataset's band. The cost is predicted before
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
    cost = n * sum(band.size * len(band.datasets) for band in bands)
    if max_evaluations is not None and cost > int(max_evaluations):
        raise ValueError(
            f"Scoring {n} candidates would cost {cost} work units "
            "(candidates × Σ over datasets of their band's flux points), "
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
                tuple((b.log_flux, b.slope, b.log_weight) for b in bands),
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
                for band, (log_flux, slope, log_weight) in zip(bands, grids):

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
                    m, top, summary = _band_summary(
                        band, scores, log_flux, slope, log_weight
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
                )

            return jax.lax.map(one, stacked, batch_size=batch_size)

        out = every(*args)
    out = [onp.asarray(x) for x in out]
    score, profiled, flux, flux_err, f_edge, slope, slope_err, b_edge = out[:8]
    return OrbitScores(
        score=score.astype(float),
        profiled=profiled.astype(float),
        flux=flux.astype(float),
        flux_err=flux_err.astype(float),
        flux_at_edge=f_edge.astype(bool),
        slope=slope.astype(float),
        slope_err=slope_err.astype(float),
        slope_at_edge=b_edge.astype(bool),
        terms=out[8][:, : len(term_fns)].astype(float),
        bands=tuple(b.name for b in bands),
        reference=reference,
        cost=int(cost),
    )
