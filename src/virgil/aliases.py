"""Orbits whose epochs undersample the period: alias bands and their evidences.

!!! warning "Provisional"
    May change in any 0.4.x release without a deprecation cycle.

When epochs are far apart compared with the period, the number of orbital
cycles between them is ambiguous, and the likelihood has one narrow ridge per
candidate count. [`fit_orbit_aliases`][virgil.aliases.fit_orbit_aliases] labels
the ridges ``N = round(T/P)`` (``T`` the span), fits each *band* of periods
directly to the observables of every epoch and weighs the bands by their
marginal likelihoods. Positions only start the fits. Each epoch's observable
blocks have an error scale with a Jeffreys prior, integrated out
([`marginal_loglike`][virgil.epochs.marginal_loglike]); a band's evidence is the
sum over its modes of a Laplace approximation, checked by importance sampling.

Source: [VanderPlas (2018)](../references.md#vanderplas2018),
[Lucy (2014)](../references.md#lucy2014),
[Price-Whelan et al. (2017)](../references.md#pricewhelan2017),
[Jeffreys (1946)](../references.md#jeffreys1946),
[MacKay (1992)](../references.md#mackay1992).
"""

import dataclasses
import functools
import json
import logging
import math
import pathlib
import pickle
import warnings

import equinox as eqx
import jax
import jax.numpy as np
import numpy as onp
from scipy import optimize
from scipy.special import logsumexp

from . import _gauss_newton
from ._deprecate import external_stacklevel
from ._precision import cast_tree, in_float64, run_in
from ._utils import clean_json
from .detection import _fingerprint
from ._geometry import fringe_scales
from .arcs import (
    ARC_DELTA_CHI2,
    ARC_JITTER,
    ARC_MIN_GAP,
    arc_profile,
    night_motion,
    unwrap_cycles,
)
from ._grid import batch_size_or_default
from .epochs import (
    _grid_batch,
    default_memory_budget,
    _whitening_floats,
    _REFINE_WINDOW,
    EpochPositions,
    _amp_profile,
    _amp_residuals,
    _binary,
    _distinct_tracks,
    _refine_marginal,
    _refuse_amp_with_gains,
    _NUMERIC_S_MAX,
    _Surface,
    estimate_primary_diameter,
    broad_loglike,
    sidelobe_flags,
)
from .likelihood import (
    group_outlier_mixture,
    group_outlier_probabilities,
    whitened_residuals,
)
from .models import OrbitalBinary, PointSource, System, UniformDisk
from .orbits import KeplerOrbit, period_grid, starting_orbits
from .tempering import (
    FLAG_RANGE,
    ChannelFactors,
    adjacent_channels,
    block_dof,
    channel_chains,
    estimate_factors,
    flat_signature,
    out_of_range,
)

_LOG = logging.getLogger(__name__)

__all__ = [
    "AliasBand",
    "AliasBandResult",
    "AliasResult",
    "LABELS_FREE_NATS",
    "alias_bands",
    "fit_orbit_aliases",
    "period_window",
    "position_profile",
]

__provisional__ = ("*",)

# Sampled coordinates (flat under the Jeffreys prior): log P, the time of
# periastron as a fraction of the period, e, cos i, omega, Omega (degrees),
# log a, log flux, and, with ``primary_diam="fit"``, log of the primary's
# uniform-disk diameter (mas). _PERIODIC maps each periodic one's index to its
# period.
_PERIODIC = {
    1: 1.0,
    4: 360.0,
    5: 180.0,
}  # periodic coordinates and their periods
_N_FLUX = 4  # fluxes tried for each position-level candidate
# Marks a setting a loaded legacy file (format 0) does not record. It is not
# None, which means "point primary" or "no V² amplitude": the accessors
# refuse to guess.
UNRECORDED = "unrecorded"
_RESULT_FORMAT = 1  # AliasResult JSON; a file without the field is format 0
# The labels are *free* when a point-reflected mode's evidence is within this
# many nats (tempered; odds of 1 to 20) of the best mode's. Why 3: docs/
# orbit_fitting.md, "Alias search"; orbit_solved_problems.md, "Label swap" row.
LABELS_FREE_NATS = 3.0
# ...and the reflected track matches within this fraction of λ/B_max.
_LABELS_FREE_RES_FRACTION = 0.25
# Where a period is "at a bound" of the window (orbit_solved_problems.md, "Period window from
# the data" row): the best period within this fraction of ``p_range[1]`` (a likelihood that is flat
# towards long periods, as for a short arc, is maximized at the edge) or
# within this factor of ``p_range[0]``.
_PERIOD_BOUND_FRACTION = 0.9
_LOWER_BOUND_FACTOR = 1.1
# A best mode whose Laplace sigma(log P) is this fraction of the window's
# width in log P does not constrain the period either, edge or no edge.
_PERIOD_SIGMA_FRACTION = 0.25
# The shortest period searched whatever the data (the physical floor of the
# window): below about 5 h two stars are in contact, and a one-instant night
# would otherwise leave the lower end undefined.
_P_FLOOR = 0.2
# The most alias bands of a default window: more are warned about and the
# lower end of the window is raised to fit them (each band is a fit).
_N_BANDS_MAX = 200
# A lower end above this many nights' length is reported (a Nyquist-style
# sampling limit that excludes short periods is the user's to see).
_WARN_P_MIN_NIGHTS = 5.0
# Default per-epoch V² amplitude range (orbit_solved_problems.md, "V² amplitude" row): a contrast loss from
# seeing or calibration; above 1 only for a calibration that overshoots.
_VIS_AMP_DEFAULT = (0.5, 1.2)
# Where a fitted diameter may reach with ``primary_diam="auto"`` (problem
# 10): the lower bound is effectively 0 (1e-3 mas changes V² by < 1e-9), so a
# primary that turns out unresolved in the orbit fit is a point; the upper
# bound is this multiple of the finest fringe spacing λ/B_max (the grid of
# the first pass ends at 1.3), so it follows the data (AMI/NRM data have
# λ/B_max of order 100 mas).
_DIAM_MIN_AUTO = 1e-3
_DIAM_TOP_AUTO = 1.5
_ELEMENTS = ("period", "dt_peri", "ecc", "inc", "omega", "Omega", "a_mas")


# ---------------------------------------------------------------------------
# Bands and diagnostics
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class AliasBand:
    """One period band, ``N = round(T/P)``: ``p_lo <= P <= p_hi`` (days)."""

    n: int
    p_lo: float
    p_hi: float


def alias_bands(times, p_range):
    """The alias bands of the epochs within ``p_range``.

    Band ``N`` holds the periods ``T/(N + 1/2) <= P <= T/(N - 1/2)`` that
    fit ``N`` cycles, rounded, between the first and last epoch (span T),
    clipped to ``p_range``. The bands are one cycle wide in frequency, so
    they tile the period prior without gaps. Periods above ``2T`` form band
    ``N = 0``: not an alias, but the part of the prior with less than one
    cycle across the span.

    Parameters
    ----------
    times : array-like
        The epochs' times (days).
    p_range : (float, float)
        The shortest and longest period (days) to consider.

    Returns
    -------
    list of AliasBand
        Ordered by ``N`` (longest period last).
    """
    times = onp.asarray(times, float)
    span = float(onp.ptp(times))
    p_min, p_max = map(float, p_range)
    if span <= 0 or not 0 < p_min < p_max:
        raise ValueError(
            "Need epochs spanning a positive time and 0 < p_min < p_max."
        )
    first = math.floor(span / p_max + 0.5)
    last = math.floor(span / p_min + 0.5)
    bands = []
    for n in range(first, last + 1):
        lo = span / (n + 0.5)
        hi = span / (n - 0.5) if n > 0 else p_max  # N = 0: P > 2T
        lo, hi = max(lo, p_min), min(hi, p_max)
        if hi > lo:
            bands.append(AliasBand(n, lo, hi))
    return bands


def period_window(
    times,
    field,
    night_length,
    *,
    resolution,
    motion_fraction=0.5,
    span_factor=4.0,
    p_floor=_P_FLOOR,
):
    """The periods worth searching, from the data alone: ``(p_min, p_max)``.

    The lower end is a Nyquist-style sampling limit (the companion must not move
    much in a night); the upper end is ``span_factor`` times the span. No published
    period enters ([why](../orbit_fitting.md#alias-search-how-it-decides-and-why-its-defaults)).

    Parameters
    ----------
    times : array-like
        The epochs' times (days).
    field, night_length, resolution : float
        The largest companion separation (mas), the longest span within one night
        (days) and the finest resolution lambda/B_max (mas).
    motion_fraction, span_factor, p_floor : float, optional
        Fraction of ``resolution`` moved in a night (0.5), upper end in spans (4),
        shortest period in days (0.2).

    Returns
    -------
    (float, float)
        ``(p_min, p_max)`` in days.

    Raises
    ------
    ValueError
        If the epochs span no time, an input is not positive, or the window is empty.
    """
    times = onp.asarray(times, float)
    span = float(onp.ptp(times))
    if not (
        span > 0
        and field > 0
        and resolution > 0
        and motion_fraction > 0
        and span_factor > 0
        and night_length >= 0
        and p_floor > 0
    ):
        raise ValueError(
            "period_window needs epochs spanning a positive time and "
            "positive field, resolution, motion_fraction, span_factor and "
            "p_floor."
        )
    p_min = max(
        2 * math.pi * field * night_length / (motion_fraction * resolution),
        night_length,
        p_floor,
    )
    p_max = span_factor * span
    if not p_min < p_max:
        raise ValueError(
            f"The period window is empty: p_min = {p_min:.3g} d (a night of "
            f"{night_length:.3g} d at a field of {field:.3g} mas) is not "
            f"below p_max = {p_max:.3g} d. Pass p_range."
        )
    return float(p_min), float(p_max)


# deliberate: tests probe its gap handling directly; fit needs full data to reach it
def _night_length(data, gap_days=0.5):
    """The longest night's span (days): the longest run of frame times with
    no gap above ``gap_days`` (as ``OIData.epochs(gap_days=0.5)`` groups
    nights), over the datasets. A dataset joining several nights is not
    one night."""
    longest = 0.0
    for d in data:
        t = onp.unique(onp.asarray(d.mjd, float))
        if t.size < 2:
            continue
        starts = onp.r_[0, onp.flatnonzero(onp.diff(t) > gap_days) + 1]
        ends = onp.r_[starts[1:] - 1, t.size - 1]
        longest = max(longest, float(onp.max(t[ends] - t[starts])))
    return longest


def position_profile(
    positions, bands, k=9.0, n_best=3, eccs=None, sign_free=False
):
    """A cheap diagnostic: the best position-level χ² in each band.

    For each band, the best Thiele–Innes fits
    ([`starting_orbits`][virgil.orbits.starting_orbits]) of ``positions``
    on a coherent period grid. Positions are lossy: use this to see
    which bands are viable and to seed the fits, not to weigh them.

    With ``sign_free`` the orbits of the sign-free fit (each night's
    position known only up to its sign) are added to those of the signed
    fit: ``n_best`` of each.

    Returns
    -------
    list of dict
        Per band: ``n``, ``chi2`` (best), ``period`` of the best orbit and
        the ``orbits`` ``[(KeplerOrbit, χ²), ...]``.
    """
    rows = []
    for band in bands:
        periods = period_grid(
            positions.dt + positions.t_ref, band.p_lo, band.p_hi, k=k
        )
        found = starting_orbits(
            positions, periods, eccs=eccs, n_best=n_best, sign_free=False
        )
        if sign_free:
            # orbit_solved_problems.md, "Label swap" row: the sign-free orbits are *added*
            # to the signed ones, which keep the signs closure phases fix.
            found = found + starting_orbits(
                positions, periods, eccs=eccs, n_best=n_best, sign_free=True
            )
        rows.append(
            dict(
                n=band.n,
                chi2=min(row[1] for row in found),
                period=float(min(found, key=lambda row: row[1])[0].period),
                orbits=found,
            )
        )
    return rows


# ---------------------------------------------------------------------------
# The marginal likelihood and its parameters
# ---------------------------------------------------------------------------


def _elements(x, t_ref):
    """KeplerOrbit and flux from the sampled vector ``x`` (traceable)."""
    period = np.exp(x[0])
    inc = np.degrees(np.arccos(np.clip(x[3], -1.0 + 1e-9, 1.0 - 1e-9)))
    orbit = KeplerOrbit(
        period, x[1] * period, x[2], inc, x[4], x[5], np.exp(x[6]), t_ref=t_ref
    )
    return orbit, np.exp(x[7])


class _UDBinary(OrbitalBinary):
    """An [`OrbitalBinary`][virgil.models.OrbitalBinary] whose primary is a
    uniform disk of diameter ``diam`` (mas)."""

    diam: jax.Array

    def __init__(self, orbit, flux, diam):
        super().__init__(orbit, flux)
        self.diam = np.asarray(diam, dtype=float)

    def at(self, mjd, t_ref=0.0):
        snap = super().at(mjd, t_ref)
        return System(
            primary=UniformDisk(self.diam),
            companion=PointSource(snap.flux, snap.dra, snap.ddec),
        )


def _model(x, cfg):
    """The scene at the sampled vector ``x`` (traceable); ``cfg`` is
    ``_Settings.cfg``: ``(t_ref, fixed diameter or None, fit diameter,
    amplitude range or None)``."""
    t_ref, diam, fit_diam, _ = cfg
    orbit, flux = _elements(x, t_ref)
    if fit_diam:
        return _UDBinary(orbit, flux, np.exp(x[8]))
    if diam:
        return _UDBinary(orbit, flux, diam)
    return OrbitalBinary(orbit, flux)


def _epoch_residuals(surface, model, d, amp_range):
    """The whitened residuals of one epoch and its V² amplitude.

    With ``amp_range`` (lo, hi), the squared visibilities are multiplied by
    one amplitude ``A``, a per-epoch contrast loss that does not depend on
    baseline or wavelength. Its profile (the weighted least-squares ``A``,
    clipped to the range, which is the bound of its Jeffreys prior) is
    analytic because the model is linear in ``A``, so the visibility
    residuals are those of ``A`` times the model, with ``A`` live (variable
    projection). ``A`` is 1 without it.
    """
    r = whitened_residuals(model, d)
    if amp_range is None or surface.names[0] != "vis":
        return r, np.ones((), r.dtype)
    _refuse_amp_with_gains(d)
    if d.vis_mode != "v2" or d.observable_kind != "split":
        raise NotImplementedError(
            "vis_amp needs squared visibilities as the visibility channel "
            f"(vis_mode='v2', unprojected), got vis_mode={d.vis_mode!r}."
        )
    amp, r = _amp_profile(r, d, surface.blocks[0][2], amp_range)
    return r, amp


def _epoch_chi2(surface, model, d, amp_range, dof=None):
    """Raw χ² per block of one epoch and its V² amplitude (for a numeric
    surface, the table its ``score`` integrates: see ``plain_chi2``)."""
    if surface.numeric:  # whitened per node of the scale integral
        if amp_range is not None:
            _refuse_amp_with_gains(d)
        return surface.chi2(model, d, None, dof), np.ones((), d.vis.dtype)
    r, amp = _epoch_residuals(surface, model, d, amp_range)
    return surface.block_chi2(r), amp


# The kernels take the data as traced arguments, so one compilation serves
# every band, mode and draw set of the same shapes. The surfaces, which hold
# only the layout of the data (see ``epochs._Surface``), and ``t_ref``, one
# float per run, are static.


def _night_terms(surfaces, x, data, cfg, dof, log_broad):
    """``(ell, bad)``: each epoch's scale-marginalized, tempered log
    likelihood at ``x`` and its outlier component. ``log_broad`` (one value
    per epoch) is the outlier component of the shared mixture; None takes
    the epoch's own likelihood with its noise inflated (``outlier="inflation"``)."""
    model = _model(x, cfg)
    ell, bad = [], []
    for s, d, dof_s in zip(surfaces, data, dof):
        chi2 = _epoch_chi2(s, model, d, cfg[3], dof_s)[0]
        ell.append(s.score(chi2, dof_s, d))
        if log_broad is None:
            twin = s.inflated
            if twin.numeric:  # its table has its own window
                chi2 = _epoch_chi2(twin, model, d, cfg[3], dof_s)[0]
            bad.append(twin.score(chi2, dof_s, d))
    return np.stack(ell), np.stack(bad) if log_broad is None else log_broad


def _loglike(surfaces, x, data, cfg, dof, log_broad=None):
    """The night-outlier mixture's log likelihood of all epochs at ``x``.

    Each block's log likelihood is multiplied by its traced ``dof = 1/f_b`` (all
    ones is the untempered fit; see ``virgil.tempering``). ``surfaces``: one
    ``_Surface`` per epoch; ``data`` and ``dof``: the epochs and per-epoch block
    ``1/f_b`` vectors (traced); ``cfg``: ``_Settings.cfg``; ``log_broad``: see
    ``_night_terms``. The mixture is
    [`group_outlier_mixture`][virgil.likelihood.group_outlier_mixture].
    """
    return group_outlier_mixture(
        *_night_terms(surfaces, x, data, cfg, dof, log_broad)
    )


class _KernelProblem(eqx.Module):
    """The negative scale-marginalized log likelihood as a kernel problem
    ([`_gauss_newton`][virgil._gauss_newton]), in the sampled coordinates.

    The curvature rows are each epoch's block residuals (the V² amplitude
    live, as in the likelihood) multiplied by √(stop_gradient ``w_b``), with
    ``w_b`` the IRLS weights of the surface
    ([`_Surface.weights`][virgil.epochs._Surface.weights]): iteratively
    reweighted least squares. The priors are flat in these coordinates, so
    there is no diagonal. ``data`` is traced, so one compilation serves
    every band and start of the same shapes.
    """

    data: tuple
    surfaces: tuple = eqx.field(static=True)
    cfg: tuple = eqx.field(static=True)
    dof: tuple
    log_broad: object = None

    def loss(self, x):
        return -_loglike(
            self.surfaces, x, self.data, self.cfg, self.dof, self.log_broad
        )

    def rows(self, x):
        model = _model(x, self.cfg)
        # The mixture's IRLS weight: each night's rows count with its
        # inlier probability, d ln L / d ln L_k = 1 - q_k.
        keep = 1.0 - jax.lax.stop_gradient(
            group_outlier_probabilities(
                *_night_terms(
                    self.surfaces,
                    x,
                    self.data,
                    self.cfg,
                    self.dof,
                    self.log_broad,
                )
            )
        )
        return np.concatenate(
            [
                s.weighted_rows(
                    _epoch_residuals(s, model, d, self.cfg[3])[0], dof_s, d
                )
                * np.sqrt(keep[i])
                for i, (s, d, dof_s) in enumerate(
                    zip(self.surfaces, self.data, self.dof)
                )
            ]
        )

    def diag(self, x):
        return np.zeros_like(x)


_value_and_grad = functools.partial(jax.jit, static_argnums=(0, 3))(
    jax.value_and_grad(_loglike, argnums=1)
)


def _terms_and_jacobian(surfaces, x, data, cfg, dof, log_broad):
    """``(jacobian of (ell, bad) at x, (ell, bad))`` of some epochs."""

    def stacked(x):
        ell, bad = _night_terms(surfaces, x, data, cfg, dof, log_broad)
        return np.concatenate([ell, bad]), (ell, bad)

    return jax.jacfwd(stacked, has_aux=True)(x)


def _weighted_terms(surfaces, x, data, cfg, dof, log_broad, weights):
    """``Σ w_ell ell + Σ w_bad bad`` of some epochs (``weights = (w_ell, w_bad)``)."""
    ell, bad = _night_terms(surfaces, x, data, cfg, dof, log_broad)
    return np.dot(weights[0], ell) + np.dot(weights[1], bad)


_terms_jac = functools.partial(jax.jit, static_argnums=(0, 3))(
    _terms_and_jacobian
)
_weighted_hessian = functools.partial(jax.jit, static_argnums=(0, 3))(
    jax.hessian(_weighted_terms, argnums=1)
)


@jax.jit
def _mixture_derivatives(ell, bad):
    """The gradient and Hessian of the mixture log likelihood with respect
    to the epochs' ``(ell, bad)``, stacked."""
    k = ell.shape[0]

    def mixture(u):
        return group_outlier_mixture(u[:k], u[k:])

    u = np.concatenate([ell, bad])
    return jax.grad(mixture)(u), jax.hessian(mixture)(u)


# deliberate: jit boundary (static batch_size for lax.map)
@functools.partial(jax.jit, static_argnums=(0, 3, 6))
def _batch(surfaces, xs, data, cfg, dof, log_broad=None, batch_size=64):
    """``_loglike`` at each row of ``xs``, ``batch_size`` rows at a time."""
    return jax.lax.map(
        lambda x: _loglike(surfaces, x, data, cfg, dof, log_broad),
        xs,
        batch_size=batch_size,
    )


# The Hessian of the likelihood (forward over reverse through the scale
# integral) holds about this many whitenings' floats per night (measured: 6.4k
# on calibrated GRAVITY-like data); it is a sum over nights, so it is
# accumulated in groups of nights that fit the memory budget.
_HESSIAN_COPIES = 8000


# deliberate: tests/test_epoch_coarse.py calls it; memory-budget grouping is not reachable cheaply via the public API
def _hessian_groups(memory_budget, data, surfaces):
    """Slices of the nights whose Hessians fit ``memory_budget`` bytes each:
    one slice (the fused evaluation) for plain data or a roomy budget."""
    n = len(data)
    if not any(s.numeric for s in surfaces):
        return [slice(0, n)]
    budget = (
        default_memory_budget() if memory_budget is None else memory_budget
    )
    if not budget > 0:
        raise ValueError(f"memory_budget must be positive, got {budget}.")
    per_night = [
        8 * _HESSIAN_COPIES * _whitening_floats(d) if s.numeric else 0
        for d, s in zip(data, surfaces)
    ]
    groups, start, used = [], 0, 0
    for i, cost in enumerate(per_night):
        if i > start and used + cost > budget:
            groups.append(slice(start, i))
            start, used = i, 0
        used += cost
    groups.append(slice(start, n))
    return groups


# deliberate: tests/test_epoch_coarse.py calls it directly
def _candidate_batch(batch_size, memory_budget, data, surfaces):
    """Orbits per batch of candidates: 64 (as ever) for plain data, fewer
    when gains or offsets make each epoch's scale integral hold many
    whitenings; ``batch_size`` overrides. A candidate evaluates every epoch,
    so each numeric epoch gets its share of ``memory_budget``."""
    if batch_size is not None:
        return batch_size_or_default(batch_size, None)
    numeric = [(d, s) for d, s in zip(data, surfaces) if s.numeric]
    if not numeric:
        return 64
    budget = (
        default_memory_budget() if memory_budget is None else memory_budget
    )
    return min(
        [64]
        + [_grid_batch(None, budget / len(numeric), d, s) for d, s in numeric]
    )


@functools.partial(jax.jit, static_argnums=(0, 3))
def _outlier_probability(surfaces, x, data, cfg, dof, log_broad):
    return group_outlier_probabilities(
        *_night_terms(surfaces, x, data, cfg, dof, log_broad)
    )


class _Problem:
    """The scale-marginalized log likelihood of all epochs and the priors."""

    def __init__(self, data, settings, batch_size=None, memory_budget=None):
        self.data, self.t_ref = list(data), settings.t_ref
        self.cfg = settings.cfg
        self.surfaces = tuple(_Surface(d, settings.s_max) for d in self.data)
        self.batch_size = _candidate_batch(
            batch_size, memory_budget, self.data, self.surfaces
        )
        self.hessian_groups = _hessian_groups(
            memory_budget, self.data, self.surfaces
        )
        # The tempering, one array of block dofs per epoch (traced).
        self.dof = tuple(
            np.asarray(
                onp.broadcast_to(
                    s.dof_vector(settings.dof), (len(s.names),)
                ).astype(float)
            )
            for s in self.surfaces
        )
        # The mixture's outlier component, traced; None for ``inflation``.
        self.log_broad = None
        if settings.outlier == "mixture":
            if len(settings.log_broad) != len(self.data):
                raise ValueError(
                    "outlier='mixture' needs one log_broad per epoch."
                )
            self.log_broad = np.asarray(settings.log_broad, float)
        self.kernel = _KernelProblem(
            tuple(self.data),
            self.surfaces,
            self.cfg,
            self.dof,
            self.log_broad,
        )
        # Proper priors: log P log-uniform over the whole p_range, tau
        # uniform over one cycle, e uniform, cos i uniform, omega and Omega
        # uniform with the (Omega + 180, omega + 180) mirror folded away
        # (Omega in [0, 180)), log a and log flux uniform.
        s = settings
        self.log_prior = float(
            -onp.log(onp.log(s.p_range[1] / s.p_range[0]))
            - onp.log(s.ecc_max)
            - onp.log(2.0)
            - onp.log(360.0)
            - onp.log(180.0)
            - onp.log(onp.log(s.a_range[1] / s.a_range[0]))
            - onp.log(onp.log(s.flux_range[1] / s.flux_range[0]))
            - (
                onp.log(onp.log(s.diam_range[1] / s.diam_range[0]))
                if s.primary_diam == "fit"
                else 0.0
            )
        )

    def value_and_grad(self, x):
        return _value_and_grad(
            self.surfaces, x, self.data, self.cfg, self.dof, self.log_broad
        )

    def hessian(self, x):
        """The Hessian of ``_loglike``. The mixture couples the epochs
        through ``ε``, so by the chain rule it is ``Σ_m F_m ∇²φ_m +
        Jᵀ F_uu J`` over the epochs' terms ``φ = (ell, bad)``: ``∇²φ_m`` one
        memory-sized group of epochs at a time, ``J`` the Jacobian of ``φ``."""
        calls = [  # the kernels' arguments for each group of epochs
            (
                self.surfaces[g],
                tuple(self.data[g]),
                self.cfg,
                self.dof[g],
                None if self.log_broad is None else self.log_broad[g],
            )
            for g in self.hessian_groups
        ]
        found = [_terms_jac(a[0], x, *a[1:]) for a in calls]
        ell = np.concatenate([e for _, (e, _) in found])
        bad = np.concatenate([b for _, (_, b) in found])
        jac = np.concatenate(
            [j[: len(e)] for j, (e, _) in found]
            + [j[len(e) :] for j, (e, _) in found]
        )
        weight, curvature = _mixture_derivatives(ell, bad)
        k, total = len(ell), jac.T @ curvature @ jac
        for a, w_ell, w_bad in zip(
            calls,
            onp.split(
                weight[:k], onp.cumsum([len(e) for _, (e, _) in found])[:-1]
            ),
            onp.split(
                weight[k:], onp.cumsum([len(e) for _, (e, _) in found])[:-1]
            ),
        ):
            total = total + _weighted_hessian(
                a[0], x, *a[1:], (np.asarray(w_ell), np.asarray(w_bad))
            )
        return total

    def batch(self, xs):
        """``_loglike`` at each row of ``xs``."""
        return _batch(
            self.surfaces,
            xs,
            self.data,
            self.cfg,
            self.dof,
            self.log_broad,
            self.batch_size,
        )

    def outlier_probability(self, x):
        """Each night's posterior outlier probability at ``x``."""
        return onp.asarray(
            _outlier_probability(
                self.surfaces,
                np.asarray(x),
                self.data,
                self.cfg,
                self.dof,
                self.log_broad,
            )
        )

    def per_epoch(self, x):
        """Raw χ²/ν per epoch and block, and the profile scales ŝ."""
        model = _model(np.asarray(x), self.cfg)
        rows = []
        for s, d in zip(self.surfaces, self.data):
            chi2, amp = _epoch_chi2(s, model, d, self.cfg[3])
            chi2 = onp.asarray(s.plain_chi2(chi2))
            rows.append(
                dict(
                    blocks=list(s.names),
                    chi2_red=(chi2 / s.nu).tolist(),
                    scale=s.scales(chi2).tolist(),
                    vis_amp=float(amp),
                )
            )
        return rows


@dataclasses.dataclass(frozen=True)
class _Settings:
    """Everything but the data that shapes a fit (and so a checkpoint)."""

    p_range: tuple
    a_range: tuple
    flux_range: tuple
    ecc_max: float
    s_max: float | None
    t_ref: float
    n_candidates: int
    n_refine: int
    n_random: int
    pool_size: int  # rows of the candidate batch: one compilation per run
    k: float
    eccs: tuple | None
    min_distance_mas: float
    max_steps: int
    n_is: int
    flag_nats: float
    ess_min: float
    seed: int
    sign_free: bool = False
    primary_diam: float | str | None = None  # None, mas, or "fit"
    diam_range: tuple = (0.05, 5.0)  # mas: log-uniform prior of a fitted one
    diam_seed: float = 0.5  # mas: where a fitted diameter starts
    vis_amp: tuple | None = None  # per-epoch V² amplitude range, or None
    flip_flux: bool = True
    resolution_mas: float | None = None  # λ/B_max: scales ``labels_free``
    init: tuple | None = (
        None  # period, dt_peri, ecc, inc, omega, Omega, a, flux
    )
    # ((block name, 1/f_b), ...): the tempering; () is none. Not in ``cfg``:
    # the kernels take it traced (``_Problem.dof``), so f never recompiles.
    dof: tuple = ()
    # The night-outlier component: "mixture" (a companion elsewhere, of log
    # likelihood ``log_broad[k]``) or "inflation" (the orbit at an inflated s).
    outlier: str = "mixture"
    log_broad: tuple = ()
    # The short-arc stage (seeds and bands): on/off, its position floor (mas)
    # and its χ² margin; in the checkpoint key, not in the kernels.
    short_arcs: bool = False
    arc_jitter_mas: float = 0.0
    arc_delta_chi2: float = ARC_DELTA_CHI2

    @property
    def fit_diam(self):
        return self.primary_diam == "fit"

    @property
    def cfg(self):
        """What the compiled kernels need besides the data (static)."""
        fixed = None if self.fit_diam else self.primary_diam
        return (self.t_ref, fixed, self.fit_diam, self.vis_amp)

    @property
    def flux_scan(self):
        """The fluxes tried for each position-level candidate: log-spaced
        inside the range, and, for near-equal fluxes (``sign_free``), also
        0.7 and 0.95 of its upper end."""
        scan = onp.geomspace(*self.flux_range, _N_FLUX + 2)[1:-1]
        if self.sign_free:
            top = onp.array([0.7, 0.95]) * self.flux_range[1]
            scan = onp.append(scan, top[top >= self.flux_range[0]])
        return scan


@dataclasses.dataclass(frozen=True, eq=False)
class _Bounds:
    """The hard bounds of the prior in the sampled coordinates, for one band
    (the angles are periodic, with none)."""

    lower: onp.ndarray
    upper: onp.ndarray

    @classmethod
    def of(cls, band, settings):
        inf = onp.inf
        a, f = settings.a_range, settings.flux_range
        d = settings.diam_range
        extra = [onp.log(d)] if settings.fit_diam else []
        return cls(
            onp.array(
                [onp.log(band.p_lo), -inf, 0.0, -1.0, -inf, -inf]
                + [onp.log(a[0]), onp.log(f[0])]
                + [e[0] for e in extra]
            ),
            onp.array(
                [onp.log(band.p_hi), inf, settings.ecc_max, 1.0, inf, inf]
                + [onp.log(a[1]), onp.log(f[1])]
                + [e[1] for e in extra]
            ),
        )

    def inside(self, xs):
        """Whether each row of ``xs`` is inside the bounds."""
        return onp.all((xs >= self.lower) & (xs <= self.upper), axis=-1)

    def clip(self, xs):
        """``xs`` moved onto the bounds, where the likelihood is defined."""
        return onp.clip(xs, self.lower, self.upper)


def _fold(x):
    """Fold the (Omega + 180, omega + 180) mirror: Omega in [0, 180), omega
    in [0, 360)."""
    x = onp.array(x, float)
    flip = onp.floor(x[5] / 180.0)
    x[5] -= 180.0 * flip
    x[4] += 180.0 * flip
    x[4] = onp.mod(x[4], 360.0)
    return x


def _start_vector(orbit, flux, t_ref, band, bounds, diam=None):
    period = float(onp.clip(float(orbit.period), band.p_lo, band.p_hi))
    dt = float(orbit.dt_peri) + float(orbit.t_ref) - t_ref
    tau = (dt / period + 0.5) % 1.0 - 0.5
    x = onp.array(
        [
            onp.log(period),
            tau,
            float(orbit.ecc),
            onp.cos(onp.radians(float(orbit.inc))),
            float(orbit.omega),
            float(orbit.Omega),
            onp.log(float(orbit.a_mas)),
            onp.log(flux),
        ]
        + ([onp.log(diam)] if len(bounds.lower) > 8 else [])
    )
    x = _fold(x)
    eps = 1e-3
    lo, hi = bounds.lower, bounds.upper
    return onp.clip(
        x,
        onp.where(onp.isfinite(lo), lo + eps, lo),
        onp.where(onp.isfinite(hi), hi - eps, hi),
    )


# The likelihood is a narrow, curved ridge for an orbit of many cycles, where
# L-BFGS-B stops 50-250 nats short of the optimum. Damped Newton on the IRLS
# curvature (``_KernelProblem``) follows the ridge; L-BFGS-B is only the
# fallback for a start where the kernel stalls.
_DTOL = 1e-9  # Newton decrement, nats, at which a mode is converged


def _optimize(problem, bounds, x0, max_steps):
    """Maximize the marginal likelihood from ``x0``, in the sampled
    coordinates and within ``bounds``.

    Damped Newton on the IRLS curvature
    ([`_gauss_newton`][virgil._gauss_newton], at most ``max_steps``
    iterations). If it does not converge, L-BFGS-B from its point, then
    the kernel again from there. Returns the folded point and its log
    likelihood (``-inf`` if the likelihood or its gradient is not finite
    at ``x0``).
    """

    def kernel(start):
        return _gauss_newton.solve(
            problem.kernel,
            start,
            lower=bounds.lower,
            upper=bounds.upper,
            dtol=_DTOL,
            max_steps=max_steps,
        )

    state = kernel(onp.asarray(x0, float))
    if int(state.stop) == _gauss_newton.NON_FINITE:
        return _fold(x0), -onp.inf
    if int(state.stop) != _gauss_newton.CONVERGED:

        def fun(x):
            value, grad = problem.value_and_grad(np.asarray(x))
            return -float(value), -onp.asarray(grad, float)

        res = optimize.minimize(
            fun,
            onp.asarray(state.z, float),
            jac=True,
            method="L-BFGS-B",
            bounds=list(
                zip(
                    *(
                        [None if not onp.isfinite(v) else float(v) for v in b]
                        for b in (bounds.lower, bounds.upper)
                    )
                )
            ),
            options=dict(maxiter=max_steps, ftol=1e-13, gtol=1e-9),
        )
        if res.fun < float(state.value):  # also false for a NaN
            state = kernel(onp.asarray(res.x, float))
    return _fold(state.z), -float(state.value)


# ---------------------------------------------------------------------------
# Evidence: Laplace, checked by importance sampling
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, eq=False)
class _Proposal:
    """A mode's multivariate Student-t proposal (``df`` degrees of freedom,
    the Laplace covariance ``chol chol^T``, ``l_inv = chol^-1``)."""

    x: onp.ndarray
    chol: onp.ndarray
    l_inv: onp.ndarray
    logdet: float
    df: float

    def draw(self, n, rng):
        """``n`` draws and their squared Mahalanobis distances."""
        z = rng.standard_normal((n, len(self.x)))
        g = rng.chisquare(self.df, n) / self.df
        return self.x + (z @ self.chol.T) / onp.sqrt(g)[:, None], (
            onp.sum(z**2, axis=1) / g
        )

    def log_pdf(self, xs, mahal=None):
        """Log density at ``xs`` (``mahal`` if already known)."""
        if mahal is None:
            diff = onp.array(xs - self.x, float)
            for k, period in _PERIODIC.items():
                if k < diff.shape[-1]:
                    diff[..., k] = (
                        diff[..., k] + 0.5 * period
                    ) % period - 0.5 * period
            mahal = onp.sum((diff @ self.l_inv.T) ** 2, axis=-1)
        d, df = len(self.x), self.df
        return (
            -0.5 * (df + d) * onp.log1p(mahal / df)
            - 0.5 * self.logdet
            + math.lgamma(0.5 * (df + d))
            - math.lgamma(0.5 * df)
            - 0.5 * d * onp.log(df * onp.pi)
        )


@dataclasses.dataclass(frozen=True, eq=False)
class _Draws:
    """A mode's importance draws: points, log target (likelihood times
    prior, -inf outside the support), log weights and proposal."""

    xs: onp.ndarray
    log_t: onp.ndarray
    log_w: onp.ndarray
    proposal: _Proposal


# deliberate: 78 lines, called by tests with a known Gaussian; folding into _band_evidence would hide its contract
def _mode_evidence(
    problem, bounds, x, value, n_is, rng, df=5.0, bound_sigma=3.0
):
    """Laplace and importance-sampling log evidences of one mode at ``x``.

    The importance proposal is a multivariate Student-t (``df``) with the
    Laplace covariance. The hard prior bounds truncate the target; angles
    are periodic and are kept within half a turn of the mode. A mode whose
    Hessian is not positive definite, or is singular, has no proposal: both
    evidences are ``-inf`` and it is dropped.

    All ``n_is`` draws are evaluated (one compiled shape), those outside the
    support at the nearest point of the bounds, and then set to ``-inf``.
    """
    h = -onp.asarray(problem.hessian(np.asarray(x)), float)
    h = 0.5 * (h + h.T)
    finite = bool(onp.all(onp.isfinite(h)))
    eig = onp.linalg.eigvalsh(h) if finite else onp.array([-onp.inf])
    proposal, flag = None, None
    if not (finite and eig.min() > 0):
        flag = "hessian-not-positive-definite"
    elif eig.min() <= 1e-13 * eig.max():  # positive but flat in a direction
        flag = "hessian-singular"
    else:
        try:
            cov = onp.linalg.inv(h)
            chol = onp.linalg.cholesky(cov)
            l_inv = onp.linalg.inv(chol)
            logdet = float(onp.linalg.slogdet(cov)[1])
            proposal = _Proposal(x, chol, l_inv, logdet, df)
        except onp.linalg.LinAlgError:
            flag = "hessian-singular"
    out = dict(
        x=x.tolist(),
        loglike=value,
        log_z_laplace=-onp.inf,
        log_z_is=-onp.inf,
        ess=0.0,
        flags=[] if flag is None else [flag],
    )
    if proposal is None:
        return out, None
    d = len(x)
    sd = onp.sqrt(onp.sum(proposal.chol**2, axis=1))
    out["log_z_laplace"] = float(
        value
        + problem.log_prior
        + 0.5 * d * onp.log(2 * onp.pi)
        + 0.5 * proposal.logdet
    )
    distance = onp.minimum(x - bounds.lower, bounds.upper - x) / sd
    if onp.any(distance < bound_sigma):
        out["flags"].append("near-prior-bound")
    out["sigma"] = sd.tolist()
    # Newton step in units of sigma, over the coordinates not on a bound:
    # large means L-BFGS stopped short of a stationary point.
    grad = onp.asarray(problem.value_and_grad(np.asarray(x))[1], float)
    free = (x - bounds.lower > 1e-6) & (bounds.upper - x > 1e-6)
    if onp.any(onp.abs(grad[free]) * sd[free] > 0.1):
        out["flags"].append("not-converged")
    xs, mahal = proposal.draw(n_is, rng)
    log_q = proposal.log_pdf(xs, mahal)
    # Angles live within half a turn of the mode.
    ok = bounds.inside(xs)
    for k, width in _PERIODIC.items():
        ok &= onp.abs(xs[:, k] - x[k]) < width / 2.0
    values = onp.asarray(problem.batch(np.asarray(bounds.clip(xs))), float)
    log_t = onp.where(ok, values, -onp.inf) + problem.log_prior
    log_w = log_t - log_q
    log_w[~onp.isfinite(log_w)] = -onp.inf
    out["log_z_is"] = float(logsumexp(log_w)) - onp.log(n_is)
    w = (
        onp.exp(log_w - onp.max(log_w[onp.isfinite(log_w)]))
        if onp.any(onp.isfinite(log_w))
        else onp.zeros(n_is)
    )
    out["ess"] = float(w.sum() ** 2 / max((w**2).sum(), 1e-300))
    return out, _Draws(xs, log_t, log_w, proposal)


# deliberate: tests check the one-mode mixture against the evidence directly
def _mixture_weights(draws):
    """Log weights of the pooled draws of a band's modes under their
    deterministic mixture proposal, ``p / mean_k q_k``: the modes' tails
    overlap, and per-mode weights would count the overlap twice."""
    xs = onp.concatenate([d.xs for d in draws])
    log_t = onp.concatenate([d.log_t for d in draws])
    log_q = onp.stack([d.proposal.log_pdf(xs) for d in draws])
    log_mix = logsumexp(log_q, axis=0) - onp.log(len(draws))
    log_w = log_t - log_mix
    log_w[~onp.isfinite(log_w)] = -onp.inf
    return xs, log_w


# ---------------------------------------------------------------------------
# Fitting one band
# ---------------------------------------------------------------------------


def _candidates(settings, band, bounds, rng, orbits):
    """The band's seed pool: position-level ``orbits`` (``[(KeplerOrbit,
    χ²), ...]`` or None) at several fluxes and both senses of motion, then
    ``n_random`` random orbits."""
    s = settings
    diam = s.diam_seed if s.fit_diam else None
    pool = []
    for orbit, _ in orbits or ():
        for flux in s.flux_scan:
            x = _start_vector(orbit, flux, s.t_ref, band, bounds, diam)
            other = onp.array(x)  # the other sense of motion (i -> 180 - i)
            other[3] = -other[3]
            pool += [x, other]
    pool.extend(_init_starts(s, band, bounds))
    for _ in range(s.n_random):
        pool.append(
            _fold(
                [
                    rng.uniform(onp.log(band.p_lo), onp.log(band.p_hi)),
                    rng.uniform(-0.5, 0.5),
                    rng.uniform(0.0, 0.6 * s.ecc_max),
                    rng.uniform(-0.95, 0.95),
                    rng.uniform(0, 360),
                    rng.uniform(0, 180),
                    rng.uniform(*onp.log(s.a_range)),
                    rng.uniform(*onp.log(s.flux_range)),
                ]
                + ([rng.uniform(*onp.log(s.diam_range))] if s.fit_diam else [])
            )
        )
    if not pool:
        raise ValueError(
            "No candidate orbits: pass positions or a positive n_random."
        )
    return onp.asarray(pool)


def _init_starts(settings, band, bounds):
    """The warm start (``init=``) as a start vector in ``band``."""
    if settings.init is None:
        return []
    period, dt_peri, ecc, inc, omega, big_omega, a_mas, flux = settings.init
    orbit = KeplerOrbit(
        period,
        dt_peri,
        ecc,
        inc,
        omega,
        big_omega,
        a_mas,
        t_ref=settings.t_ref,
    )
    diam = settings.diam_seed if settings.fit_diam else None
    return [_start_vector(orbit, flux, settings.t_ref, band, bounds, diam)]


# deliberate: tests check the label-swap algebra directly
def _flipped(x, settings, bounds):
    """``x`` with the stars' labels swapped: the same sky orbit seen from
    the other star (ω + 180°, the same elements otherwise) and the flux
    ratio inverted, kept inside the range (a ratio above the range's top
    starts at 0.9 of it)."""
    y = onp.array(x, float)
    y[4] = y[4] + 180.0
    top = settings.flux_range[1]
    y[7] = onp.log(min(1.0 / onp.exp(x[7]), 0.9 * top))
    y = _fold(y)
    return onp.clip(
        y,
        onp.where(onp.isfinite(bounds.lower), bounds.lower + 1e-3, -onp.inf),
        onp.where(onp.isfinite(bounds.upper), bounds.upper - 1e-3, onp.inf),
    )


def _score_pool(problem, settings, pool):
    """The marginal log likelihood of each candidate (-inf if not finite),
    evaluated in a batch of ``pool_size`` rows, padded with the first."""
    pad = max(settings.pool_size - len(pool), 0)
    padded = onp.concatenate([pool, onp.repeat(pool[:1], pad, axis=0)])
    score = onp.array(problem.batch(np.asarray(padded)), float)[: len(pool)]
    score[~onp.isfinite(score)] = -onp.inf
    return score


def _distinct(modes, times, t_ref, min_distance_mas):
    """Keep the best of the modes whose tracks at ``times`` agree within
    ``min_distance_mas``."""
    modes = sorted(modes, key=lambda m: -m[1])
    tracks = []
    for x, _ in modes:
        orbit, _ = _elements(np.asarray(x), t_ref)
        dra, ddec, _ = orbit.relative(np.asarray(times))
        tracks.append(onp.stack([onp.asarray(dra), onp.asarray(ddec)], -1))
    return [modes[k] for k in _distinct_tracks(tracks, min_distance_mas)]


# deliberate: tests patch the optimizer to fail starts; _fit_band only sees the count
def _refine(problem, settings, bounds, pool, times, forced=()):
    """Rank the candidates by the mixture likelihood and refine the best
    distinct few into modes.

    Returns the distinct modes ``[(x, loglike), ...]`` and the number of
    starts that failed (reported in the band's ``n_starts_failed``). A
    ``TypeError`` (a bug or a misuse) propagates; any other exception from
    one start, a ``ValueError`` or ``LinAlgError`` say, counts it as failed.
    If every start fails, a ``RuntimeError`` chained to the last error is
    raised.
    """
    s = settings
    score = _score_pool(problem, s, pool)
    keep = onp.argsort(-score, kind="stable")[: 4 * s.n_refine]
    ranked = _distinct(
        [(pool[i], score[i]) for i in keep],
        times,
        s.t_ref,
        s.min_distance_mas,
    )
    seeds = [x for x, _ in ranked[: s.n_refine]] + [
        onp.asarray(x, float) for x in forced
    ]
    fitted, n_failed, error = [], 0, None
    for x0 in seeds:
        try:
            x, value = _optimize(problem, bounds, x0, s.max_steps)
        except TypeError:
            raise
        except Exception as caught:  # noqa: BLE001 - one bad start, counted
            n_failed, error = n_failed + 1, caught
            continue
        if onp.isfinite(value):
            fitted.append((x, value))
    if seeds and n_failed == len(seeds):
        raise RuntimeError(
            f"Every one of the {n_failed} optimizer starts failed."
        ) from error
    return _distinct(fitted, times, s.t_ref, s.min_distance_mas), n_failed


# deliberate: 33 lines, tests call it; a named stage of _fit_band
def _band_evidence(problem, settings, bounds, modes_xy, rng):
    """Every mode's evidence, and the band's: the Laplace sum ``z_lap``
    and the mixture importance-sampling estimate ``z_is``.

    Where a mode sits on a prior bound (a period edge between bands, say)
    the Gaussian is cut off, so that mode's ``log_z`` is its
    importance-sampling value, which truncates the target; elsewhere it is
    the Laplace value.

    Returns ``(modes, z_lap, z_is, pool)``, with ``pool`` the pooled draws
    and their mixture log weights (``(None, None)`` without any).
    """
    modes, draws = [], []
    for x, value in modes_xy:
        record, draw = _mode_evidence(
            problem, bounds, x, value, settings.n_is, rng
        )
        modes.append(record)
        if draw is not None:
            draws.append(draw)
    for m in modes:
        edge = (
            "near-prior-bound" in m["flags"] and m["ess"] >= settings.ess_min
        )
        m["log_z"] = m["log_z_is"] if edge else m["log_z_laplace"]
    z_lap = (
        float(logsumexp([m["log_z"] for m in modes])) if modes else -onp.inf
    )
    if not draws:
        return modes, z_lap, -onp.inf, (None, None)
    xs, log_w = _mixture_weights(draws)
    z_is = float(logsumexp(log_w)) - onp.log(len(xs))
    return modes, z_lap, z_is, (xs, log_w)


# deliberate: 42 lines, tests build band rows with it
def _band_result(problem, settings, band, modes, z_lap, z_is, n_failed):
    """The band's flags, best mode, χ²/ν and per-epoch scales."""
    live = [m for m in modes if onp.isfinite(m["log_z"])]
    best_mode = max(live, key=lambda m: m["log_z"]) if live else None
    best, epochs, chi2_red = {}, [], float("nan")
    if best_mode is not None:
        best = {
            k: float(v[0])
            for k, v in _physical(onp.array(best_mode["x"])).items()
        }
        epochs = problem.per_epoch(best_mode["x"])
        nu = onp.concatenate([s.nu for s in problem.surfaces])
        chi2 = onp.concatenate([onp.array(e["chi2_red"]) for e in epochs])
        chi2_red = float(onp.sum(chi2 * nu) / onp.sum(nu))
    flags = sorted({f for m in modes for f in m["flags"]})
    if (
        onp.isfinite(z_lap)
        and onp.isfinite(z_is)
        and abs(z_lap - z_is) > settings.flag_nats
    ):
        flags.append("laplace-is-differ")
    if best_mode is not None and best_mode["ess"] < settings.ess_min:
        flags.append("low-ess")
    if best_mode is None:
        flags.append("no-valid-mode")
    if n_failed:
        flags.append("starts-failed")
    return AliasBandResult(
        n=band.n,
        p_lo=band.p_lo,
        p_hi=band.p_hi,
        modes=modes,
        log_z=z_lap,
        log_z_is=z_is,
        ess=best_mode["ess"] if best_mode else 0.0,
        flagged=bool(flags),
        flags=flags,
        best=best,
        chi2_red=chi2_red,
        epochs=epochs,
        n_starts_failed=n_failed,
    )


# deliberate: 44 lines and a named stage of _fit_band
def _flip_modes(problem, settings, bounds, modes_xy, times):
    """Refit each mode whose flux ratio sits at the top of ``flux_range``
    from the same orbit with the stars' labels swapped (the ratio is then
    below the top: the log-uniform prior on (lo, top] is not symmetric under
    the swap, and a fit pinned at the top is the signature of a ratio above
    it). A refit on a track no mode had is itself swapped back, so that the
    modes of a relabelled pair come in pairs even when the search found only
    one of them well. Returns the modes with the refits (distinct ones), the
    refits on tracks no mode had, and the number that failed."""
    flips, refits, n_failed = [], [], 0
    queue = list(modes_xy)
    for x, _ in queue:
        if not x[7] >= bounds.upper[7] - 0.02:  # not pinned at the flux top
            continue
        try:
            xf, vf = _optimize(
                problem,
                bounds,
                _flipped(x, settings, bounds),
                settings.max_steps,
            )
        except TypeError:
            raise
        except Exception:  # noqa: BLE001 - one bad start
            n_failed += 1
            continue
        if onp.isfinite(vf):
            refits.append((xf, vf))
            new = _track(xf, settings.t_ref, times)
            seen = [_track(m, settings.t_ref, times) for m, _ in queue]
            if all(  # a refit that lands on a known mode relabels nothing
                onp.max(onp.hypot(*(new - t).T)) > settings.min_distance_mas
                for t in seen
            ):
                flips.append((xf, vf))
                if len(queue) < 2 * len(modes_xy):  # one swap back
                    queue.append((xf, vf))
    merged = _distinct(
        list(modes_xy) + refits,
        times,
        settings.t_ref,
        settings.min_distance_mas,
    )
    return merged, flips, n_failed


# deliberate: test asserts the resolved-primary skip directly; no cheap public route
def _flip_is_relabelling(problem, settings):
    """The swap is a relabelling of one sky orbit only for two point
    sources and no differential phases (a resolved primary stays on the
    first star; differential phases are chromatic), so it is skipped
    otherwise."""
    return not settings.primary_diam and not any(
        d.extras for d in problem.data
    )


# deliberate: tests score its verdict directly
def _labels_free(result, settings, times, nats=LABELS_FREE_NATS):
    """Whether a mode whose track is the point reflection of the best
    mode's has an evidence within ``nats`` of it.

    Swapping the stars' labels reflects the companion's sky track through the primary
    (ω + 180°), and for near-equal fluxes it predicts the same closure
    phases and V², so the data cannot say which star is which. Comparing
    such a fit with a published orbit is then meaningful only up to the
    reflection. The modes are the distinct fits of the band (``sign_free``
    starts and the flux flip put both labelings in the pool), so the test
    is whether the best mode has a reflected partner among them.

    The tracks match when they agree within ``_LABELS_FREE_RES_FRACTION``
    of the resolution λ/B_max (``settings.resolution_mas``; the absolute
    ``min_distance_mas`` without it): a compact orbit and a wide one are
    held to what the data can tell apart, not to one number of mas.
    """
    tol = (
        settings.min_distance_mas
        if settings.resolution_mas is None
        else _LABELS_FREE_RES_FRACTION * settings.resolution_mas
    )
    live = [m for m in result.modes if onp.isfinite(m["log_z"])]
    if len(live) < 2:
        return False
    live.sort(key=lambda m: -m["log_z"])
    best = _track(onp.array(live[0]["x"]), settings.t_ref, times)
    return any(
        m["log_z"] >= live[0]["log_z"] - nats
        and onp.max(
            onp.hypot(
                *(best + _track(onp.array(m["x"]), settings.t_ref, times)).T
            )
        )
        <= tol
        for m in live[1:]
    )


def _track(x, t_ref, times):
    orbit, _ = _elements(np.asarray(x), t_ref)
    dra, ddec, _ = orbit.relative(np.asarray(times))
    return onp.stack([onp.asarray(dra), onp.asarray(ddec)], -1)


# deliberate: 51 lines and patched by tests/test_tempering.py as the per-band seam
def _fit_band(problem, settings, band, orbits, times):
    """Fit one band: ``(AliasBandResult, (draws, log weights))``.

    It has its own random stream, so a band does not depend on which bands
    were fitted or loaded before it.
    """
    rng = onp.random.default_rng([settings.seed, band.n])
    bounds = _Bounds.of(band, settings)
    pool = _candidates(settings, band, bounds, rng, orbits)
    forced = []
    if (
        settings.init is not None
        and band.p_lo <= settings.init[0] <= band.p_hi
    ):
        forced = _init_starts(settings, band, bounds)  # a fair band
    modes_xy, n_failed = _refine(
        problem, settings, bounds, pool, times, forced
    )
    flips = []
    # orbit_solved_problems.md, "Flux flip" row: a fit pinned at the prior's top flux is
    # refit with the stars' labels swapped.
    if settings.flip_flux and _flip_is_relabelling(problem, settings):
        modes_xy, flips, n_flip_failed = _flip_modes(
            problem, settings, bounds, modes_xy, times
        )
        n_failed += n_flip_failed
    modes, z_lap, z_is, draws = _band_evidence(
        problem, settings, bounds, modes_xy, rng
    )
    result = _band_result(
        problem, settings, band, modes, z_lap, z_is, n_failed
    )
    live = [m for m in result.modes if onp.isfinite(m["log_z"])]
    if flips and live:  # is the best mode one of the relabelled refits?
        best = onp.array(max(live, key=lambda m: m["log_z"])["x"])
        tracks = [
            _track(x, settings.t_ref, times)
            for x in [best] + [f[0] for f in flips]
        ]
        result.flipped = any(
            onp.max(onp.hypot(*(tracks[0] - t).T)) <= settings.min_distance_mas
            for t in tracks[1:]
        )
    result.labels_free = _labels_free(result, settings, times)
    if result.flipped:
        _LOG.info(
            "band N=%d: the best mode has the stars' labels swapped "
            "(flux ratio was pinned at the top of flux_range)",
            band.n,
        )
    return result, draws


# deliberate: patched by many tests to avoid or force the preflight
def _preflight(problem, settings, band):
    """Evaluate the likelihood and its gradient once, at the centre of the
    first band's bounds, outside any ``try``: misconfiguration (bad data,
    options or shapes) surfaces here, before any band is fitted, and what
    raises later is a failure of one start or band."""
    bounds = _Bounds.of(band, settings)
    finite = onp.isfinite(bounds.lower) & onp.isfinite(bounds.upper)
    centre = onp.zeros(len(finite))
    centre[finite] = (bounds.lower[finite] + bounds.upper[finite]) / 2
    problem.value_and_grad(np.asarray(centre))


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------

_CHECKPOINT_FORMAT = 6  # bump when a checkpoint's contents change


# deliberate: tests check which inputs invalidate a checkpoint
def _checkpoint_key(data, positions, bands, times, settings):
    """What a band's checkpoint depends on: the data (every field, static
    ones included), the positions, the bands and all settings.

    Raises
    ------
    ValueError
        If the data or positions cannot be hashed.
    """
    prints = dict(data=_fingerprint(data), positions=_fingerprint(positions))
    if None in prints.values():
        raise ValueError(
            "checkpoint_dir needs data and positions that can be hashed, to "
            "tell a checkpoint's data from new data; one holds something "
            "unhashable (a function or an arbitrary object)."
        )
    return dict(
        format=_CHECKPOINT_FORMAT,
        edges=[[bd.p_lo, bd.p_hi] for bd in bands],
        times=onp.asarray(times, float).tolist(),
        settings=dataclasses.asdict(settings),
        **prints,
    )


def _checkpoint(directory, band, key, done=None):
    """Load a band's checkpoint (None if there is none or it is for other
    settings), or, given ``done``, write it atomically (pickle to reload, JSON
    to read)."""
    if directory is None:
        return None
    pkl = pathlib.Path(directory) / f"band_{band.n}.pkl"
    if done is None:
        try:
            with open(pkl, "rb") as f:
                saved_key, saved = pickle.load(f)
        except Exception:  # noqa: BLE001 - missing or truncated: refit
            return None
        return saved if saved_key == clean_json(key) else None
    pkl.parent.mkdir(parents=True, exist_ok=True)
    tmp = pkl.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        pickle.dump((clean_json(key), done), f)
    tmp.replace(pkl)
    with open(pkl.with_suffix(".json"), "w") as f:
        json.dump(
            clean_json(dataclasses.asdict(done[0])), f, indent=1, default=str
        )
    return None


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class AliasBandResult:
    """The fit of one alias band (see [`AliasResult`][virgil.aliases.AliasResult]).

    A band from a legacy file lacking ``t_ref``, ``primary_diam`` or ``vis_amp``
    has ``UNRECORDED``, and ``scene``, ``residuals`` and ``chi2_per_night`` raise
    unless [`AliasResult.load`][virgil.aliases.AliasResult.load] was given them.

    Attributes
    ----------
    t_ref : float or None
        Reference time (MJD) of ``best["dt_peri"]``.
    primary_diam : float, ``"fit"`` or None
        A point (None), a uniform disk (mas), or one fitted (``best["diam"]``).
    vis_amp : tuple or None
        Range of the per-epoch V^2 amplitude if fitted.
    flipped, labels_free : bool
        The best mode has the stars' labels swapped; a point-reflected partner is
        within ``LABELS_FREE_NATS`` of it.
    """

    n: int
    p_lo: float
    p_hi: float
    modes: list
    log_z: float  # sum over modes of the Laplace (or, at a prior bound, IS) evidences
    log_z_is: float  # importance sampling from the modes' mixture proposal
    ess: float  # of the best mode's importance sample
    flagged: bool  # any entry in ``flags``
    flags: list = dataclasses.field(
        default_factory=list
    )  # why a band is not to be trusted
    p: float = float("nan")  # posterior probability of the band
    best: dict = dataclasses.field(default_factory=dict)
    chi2_red: float = float("nan")  # raw χ²/ν on the quoted errors, all epochs
    epochs: list = dataclasses.field(
        default_factory=list
    )  # per-epoch raw χ²/ν and scales
    flipped: bool = False  # best mode has the stars' labels swapped
    labels_free: bool = False  # a label-swapped mode has comparable evidence
    error: str = ""  # the exception, if the band failed (flag ``band-failed``)
    n_starts_failed: int = (
        0  # optimizer starts that raised (``starts-failed``)
    )
    t_ref: float | None = None
    primary_diam: float | str | None = None
    vis_amp: tuple | None = None

    def scene(self, params=None):
        """The scene the band was fitted with, at its best orbit.

        Built from ``best`` (or ``params``) as the fit's likelihood builds it;
        the swapped scene if ``flipped``.

        Parameters
        ----------
        params : dict, optional
            Elements to use instead of ``best`` (``diam`` only if fitted).

        Returns
        -------
        virgil.models.OrbitalBinary

        Raises
        ------
        ValueError
            If the band failed, has no ``t_ref``, or ``primary_diam`` is
            ``UNRECORDED`` or inconsistent with ``params``.
        """
        b = self.best if params is None else params
        if not b:
            raise ValueError(
                f"band N={self.n} has no best orbit (flags: {self.flags})."
            )
        if self.t_ref is None:
            raise ValueError("this band does not record its t_ref.")
        if _unrecorded(self.primary_diam):
            raise ValueError(_UNRECORDED_MESSAGE.format("primary_diam"))
        orbit = KeplerOrbit(
            b["period"],
            b["dt_peri"],
            b["ecc"],
            b["inc"],
            b["omega"],
            b["Omega"],
            b["a_mas"],
            t_ref=self.t_ref,
        )
        if params is not None and "diam" in params:
            # A diameter in ``params`` is the fit's own free parameter: for
            # a fixed or point primary it would silently replace the setting
            # the fit used, so refuse it rather than score another model.
            if self.primary_diam != "fit":
                raise ValueError(
                    "params has a 'diam' but the fit's primary_diam is "
                    f"{self.primary_diam!r}, not 'fit': a diameter is only "
                    "a free parameter when it was fitted."
                )
        diam = b.get("diam")
        if diam is None:  # ``params`` without a diameter: use the fitted one
            diam = (
                self.best.get("diam")
                if self.primary_diam == "fit"
                else self.primary_diam
            )
        if self.primary_diam == "fit" and diam is None:
            raise ValueError(
                "primary_diam is 'fit' but no fitted 'diam' is available: "
                "cannot rebuild the fit's scene."
            )
        if diam:  # 0 or None: a point primary, as in ``_model``
            return _UDBinary(orbit, b["flux"], diam)
        return OrbitalBinary(orbit, b["flux"])

    def _whitened(self, data, params=None):
        """Per epoch, ``([(block name, ν, whitened residuals)], amp)`` at
        the best orbit (or ``params``).

        Like the fit, it runs on float64 copies of the data, in float64.
        The V² amplitude is re-profiled at the orbit scored (the weighted
        least-squares step of ``_epoch_residuals``) rather than read from
        ``epochs``, so that the residuals at ``params`` are those of the
        fit's likelihood there; at ``best`` it returns the fitted one.
        """
        if _unrecorded(self.vis_amp):
            raise ValueError(_UNRECORDED_MESSAGE.format("vis_amp"))
        if len(data) != len(self.epochs):
            raise ValueError(
                f"{len(data)} datasets for a fit of {len(self.epochs)} epochs."
            )
        out = []
        with run_in("float64"):
            model = self.scene(params)  # inside: built in float64
            for d in data:
                d = cast_tree(d, "float64")
                surface = _Surface(d)
                r, amp = _epoch_residuals(surface, model, d, self.vis_amp)
                r = onp.asarray(r, dtype=float)
                out.append(
                    (
                        [
                            (name, a, z, nu, r[a:z])
                            for name, a, z, nu, _ in surface.blocks
                        ],
                        float(amp),
                    )
                )
        return out

    def residuals(self, data, params=None):
        """Whitened residuals of ``data`` at the best orbit, block by block.

        The fit's own residuals on the quoted errors (float64, V^2 amplitude
        re-profiled), from virgil's model: they check the bookkeeping, not the model.

        Parameters
        ----------
        data : sequence of OIData
            The epochs the band was fitted to, in order.
        params : dict, optional
            Elements to use instead of ``best``; see ``scene``.

        Returns
        -------
        list of dict
            One ``{block name: residuals}`` per epoch, in whitened-residual order
            (the penalty rows of correlated closure phases are the last rows of
            ``"phi"``).

        Raises
        ------
        ValueError
            If ``primary_diam`` or ``vis_amp`` is unrecorded, or ``data`` does not
            match the epochs.
        """
        rows = self._whitened(list(data), params)
        return [{name: r for name, _, _, _, r in row} for row, _ in rows]

    def chi2_per_night(self, data, params=None):
        """Raw chi-squared and its count of independent observables, per night.

        On the quoted errors, at the best orbit. ``chi2`` counts only the ``n``
        independent observables; ``chi2_fit`` adds the fit's penalty rows.

        Parameters
        ----------
        data : sequence of OIData
            The epochs the band was fitted to, in order.
        params : dict, optional
            Elements to use instead of ``best``; see ``scene``.

        Returns
        -------
        list of dict
            Per epoch: ``chi2``, ``chi2_fit``, ``n``, ``chi2_per_n``,
            ``chi2_per_n_fit``, ``vis_amp`` and ``blocks``. Sum ``chi2`` and ``n``
            over nights for a pooled value; do not average ``chi2_per_n``.

        Raises
        ------
        ValueError
            As for [`residuals`][virgil.aliases.AliasBandResult.residuals].
        """
        out = []
        for row, amp in self._whitened(list(data), params):
            blocks = {}
            for name, a, z, nu, r in row:
                # The first ν rows are the independent observables; any
                # rows after them are the closure-phase penalty.
                blocks[name] = _chi2_entry(
                    float(onp.sum(r[:nu] ** 2)), float(onp.sum(r**2)), nu
                )
            total = _chi2_entry(
                sum(b["chi2"] for b in blocks.values()),
                sum(b["chi2_fit"] for b in blocks.values()),
                sum(b["n"] for b in blocks.values()),
            )
            out.append(dict(total, vis_amp=amp, blocks=blocks))
        return out


def _unrecorded(value):
    """Whether ``value`` is the ``UNRECORDED``
    marker."""
    return isinstance(value, str) and value == UNRECORDED


_UNRECORDED_MESSAGE = (
    "this result does not record {0} (it was saved before it was kept): "
    "pass the fit's value to AliasResult.load(path, {0}=...)."
)


def _chi2_entry(chi2, chi2_fit, n):
    """The χ² of the data and of the fit, ``n`` and their ratios to it
    (None without data)."""
    return dict(
        chi2=chi2,
        chi2_fit=chi2_fit,
        n=int(n),
        chi2_per_n=chi2 / n if n else None,
        chi2_per_n_fit=chi2_fit / n if n else None,
    )


@dataclasses.dataclass
class AliasResult:
    """Result of [`fit_orbit_aliases`][virgil.aliases.fit_orbit_aliases].

    [`save`][virgil.aliases.AliasResult.save] and
    [`load`][virgil.aliases.AliasResult.load] keep it as JSON; ``scene``,
    ``residuals`` and ``chi2_per_night`` use the best band.

    Attributes
    ----------
    t_ref, times, span : float, numpy.ndarray, float
        Reference time of ``dt_peri``, epochs' mean times (MJD), span (days).
    bands : list of AliasBandResult
        One per alias band, by decreasing ``log_z``.
    samples : dict
        ``{N: {name: array}}`` draws for the best band and those with ``p > p_min_samples``.
    positions : list of dict or None
        Per-band position diagnostic (not saved).
    primary_diam, vis_amp, p_range, prior, unseeded : optional
        The primary, the V^2 amplitude range, the period window (days), the prior
        ranges, and flagged nights that seeded nothing.
    period_bound, period_at_lower_bound : bool
        P is unconstrained (best P within 10% of the top of ``p_range``, or
        sigma(log P) at least a quarter of the window); the best P is within 10%
        of the bottom, where the sampling limit may have cut the truth.
    window_capped, primary_fit, temper_f, temper_blocks, temper_f_raw, temper_rho
        Lower end wanted when a default window was capped; the ``"auto"``
        estimate; likelihood tempering ([`virgil.tempering`][virgil.tempering]).
    outlier, night_flags, short_arc : dict or None
        Outlier mixture (``eps``, ``names``, ``probability``; near 1: the fit
        ignored the night); ``{night: {"code", "reason"}}`` of nights preferring
        another peak (reported only); the short-arc stage (band ``p`` is then
        conditional on the bands fitted).
    """

    t_ref: float
    times: onp.ndarray
    span: float
    bands: list
    samples: dict
    positions: list = None
    primary_diam: float | str | None = None
    vis_amp: tuple | None = None
    unseeded: tuple = ()
    p_range: tuple | None = None
    period_bound: bool = False
    primary_fit: dict | None = None
    period_at_lower_bound: bool = False
    window_capped: float | None = None
    temper_f: float | None = None
    temper_blocks: dict | None = None
    temper_f_raw: float | None = None
    temper_rho: dict | None = None
    prior: dict | None = None
    short_arc: dict | None = None
    outlier: dict | None = None
    night_flags: dict | None = None

    def __post_init__(self):
        for b in self.bands:
            if b.t_ref is None:
                b.t_ref = self.t_ref
            if b.primary_diam is None:
                b.primary_diam = self.primary_diam
            if b.vis_amp is None:
                b.vis_amp = self.vis_amp

    @property
    def tempering(self):
        """Whether the tempering is outside what the likelihood can trust:
        a block ``f`` above 2, a raw ``f`` below 0.5, or a flat (not
        AR(1)) correlation in ``temper_rho``. False without a factor.

        Returns
        -------
        bool
        """
        if self.temper_f is None:
            return False
        f_max = max([self.temper_f, *(self.temper_blocks or {}).values()])
        rho = dict(self.temper_rho or {})
        if len(rho) > 1:  # as estimate_factors: blocks of their own count
            rho.pop("pooled", None)
        flat = any(flat_signature(*r) for r in rho.values())
        return out_of_range(f_max, self.temper_f_raw) or flat

    def block_dof(self):
        """The block-keyed ``dof`` (``{block: 1/f_b}``) that reproduces
        this fit's tempering in ``marginal_loglike(block_dof=)``.

        Returns
        -------
        dict
            Empty when no factor was applied.
        """
        return block_dof(self.temper_blocks or {})

    @property
    def best(self):
        """The band with the highest evidence."""
        return self.bands[0]

    @property
    def labels_free(self):
        """Whether the stars' labels are not determined by the data: in the
        best band, a mode that is the point reflection of the best mode
        has an evidence within ``LABELS_FREE_NATS`` (3 nats) of it. A
        comparison with another orbit may then accept the reflection
        (ω + 180°, Ω unchanged)."""
        return bool(self.best.labels_free)

    def table(self):
        """One row per band: raw χ²/ν, error scales (``scales`` flat over
        epochs and blocks; ``epoch_scales`` one list per epoch), log Z
        (Laplace and IS), p, ESS and the flags."""
        return [
            dict(
                n=b.n,
                p_lo=b.p_lo,
                p_hi=b.p_hi,
                period=b.best.get("period"),
                chi2_red=b.chi2_red,
                scales=[
                    v
                    for e in b.epochs
                    for v in onp.atleast_1d(e["scale"]).tolist()
                ],
                epoch_scales=[e["scale"] for e in b.epochs],
                log_z=b.log_z,
                log_z_is=b.log_z_is,
                p=b.p,
                ess=b.ess,
                flagged=b.flagged,
                flags=b.flags,
                n_modes=len(b.modes),
                labels_free=b.labels_free,
            )
            for b in self.bands
        ]

    def scene(self, params=None):
        """The best band's fitted scene; see
        [`AliasBandResult.scene`][virgil.aliases.AliasBandResult.scene]."""
        return self.best.scene(params)

    def residuals(self, data, params=None):
        """Whitened residuals at the best band's best orbit; see
        [`AliasBandResult.residuals`][virgil.aliases.AliasBandResult.residuals]."""
        return self.best.residuals(data, params)

    def chi2_per_night(self, data, params=None):
        """Raw χ² and N per night at the best band's best orbit; see
        [`AliasBandResult.chi2_per_night`][virgil.aliases.AliasBandResult.chi2_per_night]."""
        return self.best.chi2_per_night(data, params)

    def to_json(self, path, n_samples=0):
        """Write the band table and each band's modes to ``path`` as JSON.

        Parameters
        ----------
        path : str or os.PathLike
            The file to write.
        n_samples : int or None, optional
            Posterior draws written per band: none by default, the first
            ``n_samples``, or, with ``None``, all of them (what
            [`save`][virgil.aliases.AliasResult.save] does).

        Notes
        -----
        [`load`][virgil.aliases.AliasResult.load] reads the file back.
        ``positions`` is not written.
        """
        payload = dict(
            format=_RESULT_FORMAT,
            t_ref=self.t_ref,
            times=self.times.tolist(),
            span=self.span,
            primary_diam=self.primary_diam,
            vis_amp=self.vis_amp,
            unseeded=list(self.unseeded),
            p_range=self.p_range,
            period_bound=self.period_bound,
            period_at_lower_bound=self.period_at_lower_bound,
            window_capped=self.window_capped,
            primary_fit=self.primary_fit,
            temper_f=self.temper_f,
            temper_blocks=self.temper_blocks,
            temper_f_raw=self.temper_f_raw,
            temper_rho=self.temper_rho,
            prior=self.prior,
            short_arc=self.short_arc,
            outlier=self.outlier,
            night_flags=self.night_flags,
            bands=self.table(),
            band_details=[dataclasses.asdict(b) for b in self.bands],
            samples={
                str(n): {k: v[:n_samples].tolist() for k, v in s.items()}
                for n, s in self.samples.items()
            }
            if n_samples is None or n_samples
            else {},
        )
        with open(path, "w") as f:
            json.dump(clean_json(payload), f, indent=1, default=str)

    def save(self, path):
        """Write the whole result to ``path`` as JSON, every posterior draw
        included; [`load`][virgil.aliases.AliasResult.load] is its inverse.

        Non-finite numbers are written as null (strict JSON) and restored
        on loading. The ``positions`` diagnostic is not written.
        """
        self.to_json(path, n_samples=None)

    @classmethod
    def load(cls, path, primary_diam=UNRECORDED, vis_amp=UNRECORDED):
        """Read a result written by [`save`][virgil.aliases.AliasResult.save] or
        [`to_json`][virgil.aliases.AliasResult.to_json].

        Parameters
        ----------
        path : str or os.PathLike
            The JSON file.
        primary_diam, vis_amp : optional
            Settings for a file written before they were saved (else the accessors
            raise); they override a file that records them.

        Returns
        -------
        AliasResult
            ``samples`` are those written (none after ``to_json`` with the default);
            ``positions`` is None.
        """
        with open(path) as f:
            p = json.load(f)
        fmt = p.get("format", 0)  # a file without the field is legacy
        if not isinstance(fmt, int) or fmt > _RESULT_FORMAT:
            raise ValueError(
                f"{path} has AliasResult format {fmt!r}, newer than this "
                f"virgil reads (up to {_RESULT_FORMAT}): upgrade virgil."
            )
        names = {f.name for f in dataclasses.fields(AliasBandResult)}
        bands = []
        for d in p["band_details"]:
            d = {k: v for k, v in d.items() if k in names}
            for k in ("primary_diam", "vis_amp"):
                if k not in d:
                    d[k] = UNRECORDED
            # A file from before the settings were saved: a best orbit with
            # a ``diam`` can only come from a fitted diameter.
            if _unrecorded(d["primary_diam"]) and "diam" in d.get("best", {}):
                d["primary_diam"] = "fit"
            # ``clean_json`` wrote non-finite numbers as null: evidences
            # (``log_z``) were -inf, everything else NaN; keep them distinct.
            for k in ("log_z", "log_z_is"):
                if d.get(k) is None:
                    d[k] = -onp.inf
            for k in ("p", "chi2_red"):
                if k in d and d[k] is None:
                    d[k] = float("nan")
            d["modes"] = [
                {
                    k: (-onp.inf if v is None else v)
                    if k.startswith("log_z")
                    else _nan_for_none(v)
                    for k, v in m.items()
                }
                for m in d.get("modes", [])
            ]
            d["epochs"] = [
                {k: _nan_for_none(v) for k, v in e.items()}
                for e in d.get("epochs", [])
            ]
            if isinstance(d.get("vis_amp"), list):
                d["vis_amp"] = tuple(d["vis_amp"])
            bands.append(AliasBandResult(**d))
        saved = p.get("vis_amp", UNRECORDED)
        recorded = dict(
            # (an old file: the best band's, which may be inferred)
            primary_diam=p.get(
                "primary_diam", bands[0].primary_diam if bands else UNRECORDED
            ),
            vis_amp=saved
            if _unrecorded(saved) or saved is None
            else tuple(saved),
        )
        given = dict(primary_diam=primary_diam, vis_amp=vis_amp)
        if not _unrecorded(vis_amp) and vis_amp is not None:
            given["vis_amp"] = tuple(vis_amp)
        settings = {
            k: recorded[k] if _unrecorded(given[k]) else given[k]
            for k in recorded
        }
        for b in bands:
            # A band that did not record a setting, or one that was
            # overridden, takes the result's.
            for k in settings:
                if _unrecorded(getattr(b, k)) or not _unrecorded(given[k]):
                    setattr(b, k, settings[k])
        return cls(
            t_ref=p["t_ref"],
            times=onp.asarray(p["times"], dtype=float),
            span=p["span"],
            bands=bands,
            samples={
                int(n): {k: onp.asarray(v, dtype=float) for k, v in s.items()}
                for n, s in p["samples"].items()
            },
            unseeded=tuple(p.get("unseeded", ())),
            p_range=None if p.get("p_range") is None else tuple(p["p_range"]),
            period_bound=bool(p.get("period_bound", False)),
            primary_fit=p.get("primary_fit"),
            period_at_lower_bound=bool(p.get("period_at_lower_bound", False)),
            window_capped=p.get("window_capped"),
            temper_f=p.get("temper_f"),
            temper_blocks=p.get("temper_blocks"),
            temper_f_raw=p.get("temper_f_raw"),
            temper_rho=p.get("temper_rho"),
            prior=p.get("prior"),
            short_arc=p.get("short_arc"),
            outlier=p.get("outlier"),
            night_flags=p.get("night_flags"),
            **settings,
        )


def _nan_for_none(value):
    """``value`` with every None (how ``clean_json`` writes NaN and
    infinities) turned back into NaN, through lists and dicts."""
    if value is None:
        return float("nan")
    if isinstance(value, list):
        return [_nan_for_none(v) for v in value]
    return value


def _physical(x):
    x = onp.stack([_fold(row) for row in onp.atleast_2d(x)])
    period = onp.exp(x[:, 0])
    return dict(
        period=period,
        dt_peri=x[:, 1] * period,
        ecc=x[:, 2],
        inc=onp.degrees(onp.arccos(onp.clip(x[:, 3], -1, 1))),
        omega=onp.mod(x[:, 4], 360.0),
        Omega=onp.mod(x[:, 5], 360.0),
        a_mas=onp.exp(x[:, 6]),
        flux=onp.exp(x[:, 7]),
        **({"diam": onp.exp(x[:, 8])} if x.shape[1] > 8 else {}),
    )


# Raw χ²/ν on the quoted errors above which the best band is flagged. The
# per-epoch error scales absorb the routine underestimates of interferometric
# errors (χ²/ν of 1-2), so the bar sits well above them: a missed
# likelihood ridge gives χ²/ν of 65-800, and errors underestimated by more
# than 3x deserve a look anyway.
_POOR_FIT_CHI2_RED = 10.0


_MIN_SEED_NIGHTS = 2  # a Thiele-Innes fit needs at least two positions
_FEW_SEED_NIGHTS = 3  # at most this many seeding nights: add random starts
_N_RANDOM_FEW = 200  # random candidates beside seeds from few nights
_N_RANDOM_NONE = 1000  # random candidates when nothing seeds


# deliberate: 67 lines and tests call it directly
def _seeding_positions(positions, t_ref, skip_flagged, times):
    """The ``PositionData`` that seeds the bands, the nights it skipped and
    whether the seeds fell back to random starts.

    ``positions`` may be ``PositionData`` (used as given) or the
    ``EpochPositions`` of ``epoch_positions`` for this data (checked by
    count and by time against ``times``, the epochs' mean times). With the
    latter and ``skip_flagged``, every night with a blocking flag seeds
    nothing (a night with only non-blocking flags still seeds); the
    skipped nights are recorded for ``AliasResult.unseeded``. Skipping
    drops a night's *seed* only: ``data`` is untouched, so its observables
    still enter the likelihood. Fewer than two seeding nights (in any
    ``positions``) make an underdetermined Thiele-Innes fit whose
    chi2 = 0 seeds mean nothing, so the seeds are dropped with a warning
    and the bands start from random orbits (the third return value).
    """
    skipped = ()
    if isinstance(positions, EpochPositions):
        times = onp.asarray(times, float)
        if len(positions.names) != len(times) or not onp.allclose(
            positions.mjd, times, atol=0.1
        ):
            raise ValueError(
                f"positions ({len(positions.names)} datasets at MJD "
                f"{onp.round(positions.mjd, 2).tolist()}) are not the "
                f"epoch_positions of this data ({len(times)} epochs at MJD "
                f"{onp.round(times, 2).tolist()})."
            )
        if skip_flagged:
            skipped = tuple(
                dict(
                    name=name,
                    mjd=float(mjd),
                    flags=[dict(code=f.code, reason=f.reason) for f in fl],
                )
                for name, mjd, fl in zip(
                    positions.names, positions.mjd, positions.flags
                )
                if any(f.blocking for f in fl)
            )
        n_seed = len(positions.names) - (
            int(positions.flagged.sum()) if skip_flagged else 0
        )
        if n_seed >= _MIN_SEED_NIGHTS:
            positions = positions.positions(
                t_ref=t_ref, skip_flagged=skip_flagged
            )
        else:
            positions = None
            codes = {  # {name: [code, ...]} of the skipped-night records
                u["name"]: [f["code"] for f in u["flags"]] for u in skipped
            }
            detail = f" (flagged: {codes})" if skipped else ""
    elif positions is not None and len(positions.dt) < _MIN_SEED_NIGHTS:
        n_seed, detail, positions = len(positions.dt), "", None
    else:
        return positions, skipped, False
    if positions is None:
        warnings.warn(
            f"fit_orbit_aliases(): only {n_seed} night(s) can seed the "
            f"bands (at least {_MIN_SEED_NIGHTS} are needed){detail}; "
            "falling back to random starts.",
            UserWarning,
            stacklevel=external_stacklevel(),
        )
        return None, skipped, True
    return positions, skipped, False


# deliberate: tests pin its tempering threshold directly
def _signs_trusted(positions, min_gap, temper_f=1.0):
    """Whether every unflagged night (``EpochPositions``) has a decided
    sign: a ``gap_marginal`` of at least ``min_gap`` nats over its rivals,
    the point reflection among them (``ARC_MIN_GAP``). A near-equal pair's
    reflections tie, so its nights fail this, by the data, not by a flux.

    ``gap_marginal`` treats the channels as independent, so it is
    ``temper_f`` times too decisive when they are correlated: it is divided
    by the effective-channel factor, as the evidence is. A night whose
    position peak is only decisive on the untempered surface (a fringe
    alias, far from the companion) would otherwise be trusted to exclude
    bands."""
    keep = ~onp.asarray(positions.flagged)
    gaps = onp.asarray(positions.gap_marginal, float)[keep] / temper_f
    return gaps.size > 0 and bool(onp.all(gaps >= min_gap))


# deliberate: 45 lines and patched by tests
def _short_arc_profile(
    data,
    seed_positions,
    positions,
    bands,
    *,
    skip_flagged,
    p_range,
    arc_jitter,
    delta_chi2,
    n_best,
    sign_free,
    cov_scale=1.0,
    **grid,
):
    """``(profile, counts, motion)``: each band's seeds from the positions
    and the night velocities, and the periods the short arcs admit; the
    positions' and velocities' covariances are times ``cov_scale`` (the
    effective-channel factor); ``grid`` is ``ecc_max``, ``jitter_mas``,
    ``k`` and ``eccs``."""
    # The within-night velocities of the seeding nights, aligned with
    # ``EpochPositions.positions(skip_flagged=...)``.
    motion = night_motion(
        data, seed_positions, jitter=arc_jitter, cov_scale=cov_scale
    )
    if skip_flagged:
        motion = motion.take(~seed_positions.flagged)
    counts = unwrap_cycles(
        positions,
        motion,
        p_range=p_range,
        delta_chi2=delta_chi2,
        cov_scale=cov_scale,
        **grid,
    )
    profile = arc_profile(
        positions,
        motion,
        bands,
        n_best=n_best,
        sign_free=sign_free,
        cov_scale=cov_scale,
        **grid,
    )
    return profile, counts, motion


_N_CHECK = 3  # seed orbits per band that the cheap whole-window check scores


def _seed_scores(problem, settings, bands, seeds):
    """Each band's best log likelihood over its first ``_N_CHECK``
    position-level orbits (at every seed flux and both senses, unrefined):
    a cheap check of the whole window."""
    s = dataclasses.replace(settings, n_random=0, init=None)
    pools = [
        _candidates(s, b, _Bounds.of(b, s), None, seeds[b.n][:_N_CHECK])
        for b in bands
    ]
    owner = onp.repeat([b.n for b in bands], [len(x) for x in pools])
    xs, size = onp.concatenate(pools), settings.pool_size
    score = onp.concatenate(
        [
            _score_pool(problem, settings, xs[i : i + size])
            for i in range(0, len(xs), size)
        ]
    )
    return {b.n: float(score[owner == b.n].max()) for b in bands}


# deliberate: 45 lines and tests call it
def _short_arc_bands(problem, settings, bands, seeds, counts, motion):
    """The bands to fit, and ``AliasResult.short_arc``.

    The bands the cycle counts admit, plus any excluded band whose seeds
    score better (``_seed_scores``) than every admitted band's: the
    short-arc stage never silently drops a band the cheap check prefers.
    The band of a warm start (``init``) is always kept.
    """
    init = None if settings is None else settings.init

    def warm(b):  # a warm start's band is always fitted
        return init is not None and b.p_lo <= init[0] <= b.p_hi

    kept = [b for b in bands if counts.admits(b) or warm(b)] or list(bands)
    excluded = [b for b in bands if b not in kept]
    readmitted, score = [], {}
    if excluded:
        score = _seed_scores(problem, settings, bands, seeds)
        top = max(score[b.n] for b in kept)
        readmitted = [b for b in excluded if score[b.n] > top]
    if readmitted:
        warnings.warn(
            "fit_orbit_aliases(): the short arcs excluded bands N="
            f"{[b.n for b in readmitted]}, but their seeds fit the data "
            "better than those of the bands they admit: fitting them too.",
            UserWarning,
            stacklevel=external_stacklevel(),
        )
    fit = [b for b in bands if b in kept or b in readmitted]
    _LOG.info(
        "short arcs: %d of %d bands admitted (%d readmitted), %d velocities",
        len(kept),
        len(bands),
        len(readmitted),
        0 if motion is None else int(motion.measured.sum()),
    )
    return fit, dict(
        intervals=[list(i) for i in counts.intervals],
        kept=[b.n for b in kept],
        excluded=[b.n for b in excluded if b not in readmitted],
        readmitted=[b.n for b in readmitted],
        velocities=0 if motion is None else int(motion.measured.sum()),
        seed_scores=[[n, v] for n, v in score.items()],
        steps=list(counts.steps),
    )


# deliberate: tests call it directly
def _check_seed_diameter(positions, primary_diam):
    """Warn when ``EpochPositions`` were fitted with another primary than
    the engine's: the seeds then carry the bias of the wrong star."""
    if not isinstance(positions, EpochPositions):
        return
    seed = positions.primary_diam or None
    engine = primary_diam or None
    if engine == "fit" and seed is not None:
        return  # a UD estimate seeds a fitted diameter: nothing to warn of
    if engine == "fit":
        warnings.warn(
            "The positions were fitted with a point primary but "
            "fit_orbit_aliases fits the primary's diameter: a resolved "
            "primary biases the seeds. Pass an estimated primary_diam to "
            "epoch_positions.",
            UserWarning,
            stacklevel=external_stacklevel(),
        )
    elif seed != engine:
        warnings.warn(
            f"The positions were fitted with primary_diam={seed!r} but "
            f"fit_orbit_aliases has primary_diam={primary_diam!r}: the seeds "
            "are biased by a different primary. Pass the same diameter to "
            "epoch_positions.",
            UserWarning,
            stacklevel=external_stacklevel(),
        )


# deliberate: tests (also test_marginal_gains.py) call it directly
def _default_vis_amp(data):
    """``_VIS_AMP_DEFAULT`` when every dataset has squared visibilities as
    its visibility channel (what ``vis_amp`` needs), else None (logged at
    INFO with the datasets that disqualify it)."""
    gained = [
        k
        for k, d in enumerate(data)
        if getattr(d, "gains", None) is not None
        or getattr(d, "phase_offsets", None) is not None
    ]
    if gained:
        _LOG.info(
            "vis_amp=True: no V² amplitude, because dataset(s) %s carry "
            "gains or closure offsets, which supersede it.",
            gained,
        )
        return None
    bad = [
        k
        for k, d in enumerate(data)
        if not (d.vis_mode == "v2" and d.observable_kind == "split")
    ]
    if bad:
        _LOG.info(
            "vis_amp=True: no V² amplitude, because dataset(s) %s do not "
            "have squared visibilities as their visibility channel "
            "(vis_mode='v2', unprojected).",
            bad,
        )
        return None
    return _VIS_AMP_DEFAULT


def _auto_primary(data, positions, vis_amp=None, block_dof=None):
    """``primary_diam="auto"``: ``(primary_diam, positions, estimate, record)``.

    A cheap first pass fits a shared uniform-disc diameter to the per-night
    positions ([`estimate_primary_diameter`][virgil.epochs.estimate_primary_diameter]
    with the orbit fit's ``vis_amp``, so a contrast loss is not read as a disc).
    If the data resolve the primary the engine fits the diameter (``"fit"``) and
    the seeds are re-refined with it; otherwise it is a point. Without
    ``EpochPositions`` there is nothing to estimate from: a point, with a warning;
    seeds the user refined at a fixed diameter are kept, with a warning, when the
    data do not resolve it. ``block_dof`` tempers the pass so correlated channels
    do not make a point primary look resolved; ``record`` is
    ``AliasResult.primary_fit``.
    """
    if not isinstance(positions, EpochPositions):
        reason = (
            "primary_diam='auto' needs EpochPositions (from "
            "epoch_positions) to estimate a diameter from"
        )
        warnings.warn(
            f"fit_orbit_aliases(): {reason}: using a point primary. Pass "
            "EpochPositions, or primary_diam explicitly (None, a float or "
            "'fit') to silence this.",
            UserWarning,
            stacklevel=external_stacklevel(),
        )
        return None, positions, None, dict(resolved=None, reason=reason)
    estimate = estimate_primary_diameter(
        data, positions, vis_amp=vis_amp, block_dof=block_dof
    )
    _LOG.info(
        "primary_diam='auto': %s (diameter %.3g mas, +%.1f nats over a "
        "point primary).",
        "resolved" if estimate.resolved else "not resolved",
        estimate.diam,
        estimate.delta_nats,
    )
    if not estimate.resolved and positions.primary_diam:
        warnings.warn(
            f"fit_orbit_aliases(): the positions were refined with "
            f"primary_diam={positions.primary_diam!r} but the data do not "
            "resolve a primary (gain "
            f"{estimate.delta_nats:.1f} nats): keeping your seeds, and the "
            "fit uses a point primary. Pass primary_diam explicitly to "
            "choose.",
            UserWarning,
            stacklevel=external_stacklevel(),
        )
        return None, positions, estimate, _primary_fit_record(estimate)
    return (
        "fit" if estimate.resolved else None,
        estimate.positions,
        estimate,
        _primary_fit_record(estimate),
    )


def _dof_pairs(factors):
    """The ``Settings.dof`` of ``factors``.

    Parameters
    ----------
    factors : ChannelFactors or None
        The effective-channel factors.

    Returns
    -------
    tuple
        ``(block name, 1/f_b)`` pairs, each ``1/f_b`` rounded to 6
        significant figures (so that an estimate's last digits do not
        invalidate a checkpoint); empty (untempered, the fast path) when
        every ``f_b`` is 1.
    """
    return _dof_of_blocks(None if factors is None else factors.blocks)


def _dof_of_blocks(blocks):
    """``_dof_pairs`` of a ``{block name: f_b}`` mapping (or None): what
    ``AliasResult.temper_blocks`` stores, so that a rebuilt problem has
    exactly the tempering of the fit."""
    if blocks is None or all(v == 1.0 for v in blocks.values()):
        return ()
    return tuple(
        (k, float(f"{1.0 / v:.6g}")) for k, v in sorted(blocks.items())
    )


def _pilot_diameter(positions, primary_diam):
    """The primary diameter (mas) of ``_temper_factors``' pilot residuals: a fixed
    one, else the one ``positions`` were refined with, else None (a point).
    """
    if isinstance(primary_diam, (int, float)) and primary_diam:
        return float(primary_diam)
    return positions.primary_diam or None


def _temper_factors(data, positions, diam, vis_amp, names):
    """The effective-channel factors ``f`` from the residuals of each night's best
    static binary (``positions``), refined at the orbit fit's primary ``diam`` and
    ``vis_amp`` so smooth model error is not read as channel correlation
    ([`virgil.tempering`][virgil.tempering]). Blocks in ``names`` without
    residuals of their own take the pooled ``f``; returns ``ChannelFactors``.
    """
    chains = [channel_chains(d) for d in data]
    if not any(len(c[k]) for c in chains for k in c):
        return estimate_factors([], [], names)
    use = ~positions.flagged
    if not use.any():
        use = onp.ones(len(data), bool)
    amp = None if vis_amp is None else tuple(map(float, vis_amp))
    dm = None if not diam else np.asarray(float(diam))
    residuals, used = [], []
    with run_in("float64"):
        for i in onp.flatnonzero(use):
            d = cast_tree(data[i], "float64")
            surface = _Surface(d)
            x0 = onp.array(
                [positions.dra[i], positions.ddec[i], positions.flux[i]]
            )
            x0[2] = onp.clip(x0[2], 1e-4, 1.0)
            window = _REFINE_WINDOW * float(fringe_scales(d)[0])
            bounds = [
                (x0[0] - window, x0[0] + window),
                (x0[1] - window, x0[1] + window),
                (1e-4, 1.0),
            ]
            x, value, _ = _refine_marginal(
                x0, np.asarray(1.0), d, surface, bounds, dm, amp
            )
            if not onp.isfinite(value):  # no improvement: keep the position
                x = x0
            model = _binary(np.asarray(x), dm)
            r = onp.asarray(_amp_residuals(model, d, surface, amp), float)
            chi2 = onp.asarray(surface.block_chi2(np.asarray(r)), float)
            scales = dict(
                zip(
                    surface.names,
                    onp.sqrt(onp.maximum(chi2 / surface.nu, 1e-12)),
                )
            )
            z = {}
            n_vis = int(onp.size(d.vis))
            n_phi = int(onp.size(d.phi))
            if n_vis:
                # whitened (and V² amplitude profiled, as in the fit)
                z["vis"] = r[:n_vis] / scales["vis"]
            if n_phi and d.phase_offsets is None:
                # With ``with_closure_offsets`` the likelihood marginalizes an
                # offset that this residual would show as channel-shared:
                # that block takes the pooled f instead.
                if d.cp_noise is None:
                    phi_r = r[n_vis : n_vis + n_phi]
                else:
                    cp = d.cp_noise
                    flat = onp.zeros(cp.valid.size)
                    flat[cp.keep] = r[n_vis : n_vis + cp.size]
                    proj = onp.einsum(
                        "gkm,gk->gm", cp.basis, flat.reshape(cp.valid.shape)
                    )
                    phi_r = onp.zeros(n_phi)
                    onp.add.at(phi_r, cp.groups, onp.where(cp.mask, proj, 0.0))
                z["phi"] = phi_r / scales["phi"]
            residuals.append(z)
            used.append(chains[i])
    return estimate_factors(residuals, used, names)


# f above this, on residuals whitened with the gains, is a noise model that
# does not describe the data (docs: design/orbit_noise_models.md, 3.2.3).
_INADEQUATE_F = 3.0


# deliberate: 57 lines and tests/test_tempering.py calls it
def _warn_tempering(factors, calibrated=False):
    """Warn about estimated tempering the fit cannot trust (a block ``f``
    outside ``FLAG_RANGE``, blocks on the pooled ``f``, a flat correlation).

    Parameters
    ----------
    factors : ChannelFactors
        The estimated factors (a user's fixed ``f`` is not warned about).
    calibrated : bool, optional
        Whether the data carry gains or closure offsets, so that ``f`` is
        estimated on residuals whitened with them: a large ``f`` then says
        the noise model is inadequate.
    """
    where = external_stacklevel()
    if calibrated and factors.f_max > _INADEQUATE_F:
        worst = max(factors.blocks, key=factors.blocks.get)
        warnings.warn(
            f"fit_orbit_aliases(): noise model inadequate: the "
            f"effective-channel factor is f = {factors.f_max:.3g} (block "
            f"{worst!r}) after whitening, above {_INADEQUATE_F:g}. The "
            "likelihood is tempered by 1/f, which hides a misfit or an "
            "unmodelled calibration error: check the gains and errors.",
            UserWarning,
            stacklevel=where,
        )
    elif out_of_range(factors.f_max, factors.f_raw):
        worst = max(factors.blocks, key=factors.blocks.get)
        warnings.warn(
            f"fit_orbit_aliases(): the effective-channel factor is "
            f"f = {factors.f_max:.3g} (block {worst!r}; pooled "
            f"{factors.f:.3g}, raw {factors.f_raw:.3g}), outside "
            f"{FLAG_RANGE[0]:g} to {FLAG_RANGE[1]:g}: the likelihood is "
            "tempered by 1/f (AliasResult.tempering). Check the channels' "
            "errors.",
            UserWarning,
            stacklevel=where,
        )
    if factors.fallback and factors.f > 1.0:
        warnings.warn(
            f"fit_orbit_aliases(): the blocks {list(factors.fallback)} have "
            "no channel chains of their own to estimate f from (extras, "
            "closure phases with offsets, or too few chains) and take the "
            f"pooled f = {factors.f:.3g}.",
            UserWarning,
            stacklevel=where,
        )
    if factors.flat:
        warnings.warn(
            f"fit_orbit_aliases(): the residual correlation of "
            f"{list(factors.flat)} is flat across channels (lag-2 well "
            "above the square of lag-1), the signature of a shared "
            "calibration error or of misfit (a resolved primary, a moving "
            "companion). The likelihood is tempered by it either way; "
            "check the model (AliasResult.tempering).",
            UserWarning,
            stacklevel=where,
        )


def _primary_fit_record(estimate):
    """The JSON-ready summary of a ``PrimaryDiameter``."""
    return dict(
        diam=float(estimate.diam),
        sigma=float(estimate.sigma),
        delta_nats=float(estimate.delta_nats),
        resolved=bool(estimate.resolved),
        at_grid_edge=bool(estimate.at_grid_edge),
    )


# deliberate: tests probe its fallbacks directly
def _field_mas(positions, a_range, fallback):
    """The separation (mas) the period window's sampling limit uses: the
    *median* of the positions' separations (one wrong alias peak must not
    raise it), over the unflagged nights (all if every night is flagged),
    clipped to ``[1e-3, a_range[1]]``. Without positions, ``fallback``
    (the coarsest fringe spacing λ/B_min) up to ``a_range[1]``."""
    if isinstance(positions, EpochPositions):
        keep = ~positions.flagged
        if not keep.any():
            keep = onp.ones(len(keep), bool)
        sep = onp.hypot(positions.dra[keep], positions.ddec[keep])
    elif hasattr(positions, "dra") and len(positions.dra):
        sep = onp.hypot(positions.dra, positions.ddec)
    else:
        return float(min(fallback, a_range[1]))
    return float(min(max(float(onp.median(sep)), 1e-3), a_range[1]))


# === INTERNAL INTERFACE OF OrbitPipeline ===
# ``_band_target`` (with the ``_loglike`` and ``_physical`` it returns the
# pieces of) is all the pipeline takes from this module besides the public
# ``fit_orbit_aliases`` and ``AliasResult``; change it with
# ``tests/test_pipeline_orbit.py`` at hand.


# One epoch's broad component is averaged over a grid of this fraction of the
# finest fringe scale λ/B_max: the position grid step of ``OrbitPipeline``
# (twice the Nyquist step of the shortest fringe period).
_BROAD_STEP = 0.25
_OUTLIERS = ("mixture", "inflation")


def _broad_components(
    data,
    settings,
    primary_diam,
    diam_seed,
    resolution,
    positions=None,
    **batching,
):
    """``settings`` with the mixture's outlier component ``log_broad`` (its
    own ``s_max``, tempering and primary); unchanged for ``inflation``.
    ``positions`` (EpochPositions) lends its level-0 grid scores."""
    if settings.outlier not in _OUTLIERS:
        raise ValueError(
            f"outlier must be one of {_OUTLIERS}, got {settings.outlier!r}."
        )
    diam = diam_seed if primary_diam == "fit" else primary_diam
    if settings.outlier == "inflation":
        if settings.s_max is None:
            raise ValueError("outlier='inflation' needs s_max.")
        return settings
    if settings.s_max is None:
        raise ValueError(
            "outlier='mixture' needs a finite s_max: the outlier component "
            "is scored on the scale-marginalized surface. Omit s_max for "
            f"the default ({_NUMERIC_S_MAX:g})."
        )
    surfaces = [_Surface(d, settings.s_max) for d in data]
    dof = [
        onp.broadcast_to(s.dof_vector(settings.dof), (len(s.names),)).astype(
            float
        )
        for s in surfaces
    ]
    with run_in("float64"):
        broad = broad_loglike(
            data,
            dof,
            a_max=settings.a_range[1] * (1.0 + settings.ecc_max),
            flux_range=settings.flux_range,
            step=_BROAD_STEP * resolution,
            s_max=settings.s_max,
            primary_diam=diam,
            positions=positions,
            **batching,
        )
    if not onp.all(onp.isfinite(broad)):
        raise ValueError(
            "The outlier component is not finite for the nights "
            f"{onp.flatnonzero(~onp.isfinite(broad)).tolist()}: the data "
            "have no valid points."
        )
    return dataclasses.replace(settings, log_broad=tuple(map(float, broad)))


# deliberate: its failure is written to the result and tests inject one
def _night_diagnostics(problem, settings, result, positions, data, t_ref):
    """``(outlier, night_flags)`` of ``AliasResult``: each night's posterior
    outlier probability and the sidelobe flags, both at the best orbit of the
    best band. A failed sidelobe check adds ``"sidelobe-check-failed"`` to
    that band's flags."""
    best = result.best
    live = [m for m in best.modes if onp.isfinite(m["log_z"])]
    outlier = dict(  # the component is recorded whatever the modes are
        kind=settings.outlier,
        log_broad=list(settings.log_broad),
        names=[f"{k}" for k in range(len(data))]
        if not isinstance(positions, EpochPositions)
        else list(positions.names),
        probability=None,
    )
    if not best.best or not live:
        return outlier, None
    x = onp.asarray(max(live, key=lambda m: m["log_z"])["x"], float)
    outlier["probability"] = problem.outlier_probability(x).tolist()
    if not isinstance(positions, EpochPositions):
        return outlier, None
    try:
        scene = best.scene()
        snaps = [scene.at(float(t), t_ref) for t in positions.mjd]
        pred = [
            onp.array(
                [float(getattr(getattr(s, "companion", s), a)) for s in snaps]
            )
            for a in ("dra", "ddec", "flux")
        ]
        found = sidelobe_flags(data, positions, pred)
    except Exception as caught:  # noqa: BLE001 - recorded in the band's flags
        best.flags.append("sidelobe-check-failed")
        _LOG.warning("sidelobe flags failed: %r", caught)
        found = [None] * len(positions.names)
    night_flags = {
        n: {"code": f.code, "reason": f.reason}
        for n, f in zip(positions.names, found)
        if f is not None
    }
    return outlier, night_flags


# deliberate: imported by virgil/pipeline/orbit.py (internal interface)
def _band_target(result, data, n=None):
    """The engine's posterior of one band, rebuilt from a saved result:
    ``(problem, bounds, x, settings)``.

    ``problem`` gives ``_loglike`` exactly as the fit scored it (tempered, V^2
    amplitude profiled), unlike the public ``numpyro_model``, which would sample
    an untempered posterior; ``x`` is the band ``n``'s (default the best) best
    mode in the sampled coordinates. Needs ``result.prior``.
    """
    band = (
        result.best if n is None else next(b for b in result.bands if b.n == n)
    )
    if result.prior is None or "kind" not in (result.outlier or {}):
        raise ValueError(
            "This AliasResult did not record its prior ranges or its "
            "outlier component (it was saved by an older virgil, whose "
            "likelihood differs): refit it to sample its posterior."
        )
    live = [m for m in band.modes if onp.isfinite(m["log_z"])]
    if not live:
        raise ValueError(f"Band N = {band.n} has no valid mode.")
    mode = max(live, key=lambda m: m["log_z"])
    prior = result.prior
    settings = _Settings(
        p_range=tuple(result.p_range),
        a_range=tuple(prior["a_range"]),
        flux_range=tuple(prior["flux_range"]),
        ecc_max=prior["ecc_max"],
        s_max=prior["s_max"],
        t_ref=result.t_ref,
        n_candidates=0,
        n_refine=0,
        n_random=0,
        pool_size=0,
        k=0.0,
        eccs=None,
        min_distance_mas=0.0,
        max_steps=0,
        n_is=0,
        flag_nats=0.0,
        ess_min=0.0,
        seed=0,
        primary_diam=result.primary_diam,
        diam_range=tuple(prior["diam_range"]),
        vis_amp=result.vis_amp,
        dof=_dof_of_blocks(result.temper_blocks),
        outlier=result.outlier["kind"],
        log_broad=tuple(result.outlier["log_broad"]),
    )
    problem = _Problem(cast_tree(list(data), "float64"), settings)
    return (
        problem,
        _Bounds.of(band, settings),
        onp.asarray(mode["x"], float),
        settings,
    )


# ---------------------------------------------------------------------------
# The entry point
# ---------------------------------------------------------------------------


@in_float64
def fit_orbit_aliases(
    data,
    p_range=None,
    *,
    positions=None,
    times=None,
    t_ref=None,
    a_range=(1.0, 30.0),
    flux_range=(0.01, 1.0),
    ecc_max=0.9,
    s_max=_NUMERIC_S_MAX,
    n_candidates=100,
    n_refine=6,
    n_random=None,
    k=9.0,
    eccs=None,
    sign_free=True,  # deliberate: a label swap of near-equal stars must be found
    min_distance_mas=0.5,
    max_steps=200,
    n_is=2000,
    n_samples=1000,
    p_min_samples=0.01,
    flag_nats=1.0,
    ess_min=50.0,
    seed=0,
    checkpoint_dir=None,
    on_band=None,
    primary_diam="auto",  # deliberate: a resolved primary must not bias the orbit
    diam_range=None,
    vis_amp=True,  # deliberate: contrast loss must not be read as a disc or a bad orbit
    flip_flux=True,  # deliberate: a brighter companion must not hide at the flux bound
    init=None,
    skip_flagged=True,  # deliberate: a flagged night seeds nothing but its data still count
    n_bands_max=_N_BANDS_MAX,
    temper=True,  # deliberate: untempered evidence is overconfident under correlated channels
    batch_size=None,
    memory_budget=None,
    short_arcs=True,  # deliberate: Nyquist-style bands alone leave hundreds of cycle counts
    arc_jitter=ARC_JITTER,
    arc_delta_chi2=ARC_DELTA_CHI2,
    arc_min_gap=ARC_MIN_GAP,
    outlier="mixture",
    outlier_eps=None,
):
    """Weigh the period aliases of an undersampled orbit with the observables.

    Each alias band (an integer number of orbits between the epochs) is fitted
    to all epochs and given an evidence; the defaults need no per-system
    arguments ([how](../orbit_fitting.md#alias-search-how-it-decides-and-why-its-defaults),
    [settings](../orbit_methods.md#alias-settings)).

    Parameters
    ----------
    data : sequence of OIData
        One dataset per epoch.
    p_range : (float, float), optional
        Period range (days); default [`period_window`][virgil.aliases.period_window].
    positions : PositionData or EpochPositions, optional
        Seeds for the fits (else random orbits).
    times, t_ref, a_range, flux_range, ecc_max, s_max : optional
        Epoch times and ``dt_peri`` reference (MJD; default the mean times),
        prior ranges (``a_range`` in mas), eccentricity cap, ``s_max`` (10;
        the mixture refuses None).
    n_candidates, n_refine, n_random, n_is, n_samples, seed, init : optional
        Search effort, sampling and a warm start.
    min_distance_mas, max_steps, p_min_samples, flag_nats, ess_min : optional
        Modes closer than ``min_distance_mas`` (mas, along the orbit) are one;
        optimizer steps per start; bands with ``p`` above ``p_min_samples`` also
        get posterior samples; evidence margin (nats) and importance-sample size
        below which a band is flagged.
    checkpoint_dir, on_band : optional
        Finished bands are reloaded from ``checkpoint_dir``; ``on_band`` gets each.
    primary_diam, diam_range : optional
        ``"auto"`` fits the primary diameter if resolved, a float (mas) fixes it,
        ``"fit"`` samples it (prior ``diam_range``).
    vis_amp, sign_free, flip_flux, skip_flagged : optional
        Profiled V^2 amplitude; sign-free seeds; label-swapped refit at the flux
        ceiling; flagged nights seed nothing.
    temper : bool or float, optional
        True estimates the tempering (needs ``EpochPositions``), a float >= 1
        fixes it, False is off.
    outlier, outlier_eps : optional
        A night the orbit misses is a companion elsewhere (``"mixture"``) or
        the orbit with inflated noise (``"inflation"``, needs ``s_max``), at a
        marginalized rate ([`group_outlier_mixture`][virgil.likelihood.group_outlier_mixture]);
        ``outlier_eps`` is deprecated and ignored.
    n_bands_max, k, eccs, batch_size, memory_budget : optional
        Band cap, grid density, seed eccentricities, batching.
    short_arcs, arc_jitter, arc_delta_chi2, arc_min_gap : optional
        Fit only the bands the short arcs admit ([`virgil.arcs`][virgil.arcs]);
        needs ``EpochPositions`` and ``skip_flagged``, with every seeding night
        decided; otherwise every band is seeded from the positions.

    Returns
    -------
    AliasResult

    Raises
    ------
    RuntimeError
        If every band fails (chained to the last error).
    ValueError
        If no band lies in ``p_range``, or ``checkpoint_dir`` data cannot be hashed.
        Preflight likelihood errors propagate unchanged.
    """
    if outlier_eps is not None:
        warnings.warn(
            "fit_orbit_aliases(outlier_eps=) is ignored: the outlier rate is "
            "marginalized (Beta(1/2, 1/2)). Drop the argument.",
            FutureWarning,
            stacklevel=external_stacklevel(),
        )
    data = cast_tree(list(data), "float64")  # a generator, too
    mean_times = (
        onp.array([onp.mean(onp.asarray(d.mjd, float)) for d in data])
        if times is None
        else onp.asarray(times, float)
    )
    t_ref = (
        float(onp.round(onp.mean(mean_times), 1))
        if t_ref is None
        else float(t_ref)
    )
    # orbit_solved_problems.md, "Period window from the data" row: without p_range, the
    # window follows from the data alone.
    window_capped = None
    if p_range is None:
        night = _night_length(data)
        resolution, coarsest = (float(v) for v in fringe_scales(data))
        field = _field_mas(positions, a_range, coarsest)
        p_range = period_window(
            mean_times, field, night, resolution=resolution
        )
        if (
            n_bands_max is not None
            and len(alias_bands(mean_times, p_range)) > n_bands_max
        ):  # raise the lower end until the window holds n_bands_max bands
            window_capped = float(p_range[0])
            span = float(onp.ptp(mean_times))
            first = math.floor(span / p_range[1] + 0.5)
            last = first + int(n_bands_max) - 1
            capped = max(span / (last + 0.5), p_range[0])
            p_range = (float(capped), p_range[1])
        _LOG.info(
            "period window %.3g to %.3g d (night %.3g d, field %.3g mas, "
            "resolution %.3g mas)",
            *p_range,
            night,
            field,
            resolution,
        )
        if window_capped is not None:
            warnings.warn(
                f"fit_orbit_aliases(): the default period window holds more "
                f"than n_bands_max={n_bands_max} alias bands: its lower end "
                f"is raised from {window_capped:.3g} to {p_range[0]:.3g} d. "
                "Pass p_range, or a larger n_bands_max, to search shorter "
                "periods.",
                UserWarning,
                stacklevel=external_stacklevel(),
            )
        if p_range[0] > _WARN_P_MIN_NIGHTS * max(night, _P_FLOOR):
            warnings.warn(
                f"fit_orbit_aliases(): the period window starts at "
                f"{p_range[0]:.3g} d, {p_range[0] / max(night, _P_FLOOR):.1f}"
                f" nights' length: a shorter period would move the "
                f"companion by more than half the resolution during a "
                f"night (field {field:.3g} mas). Pass p_range to search "
                "shorter periods.",
                UserWarning,
                stacklevel=external_stacklevel(),
            )
    bands = alias_bands(mean_times, p_range)
    if not bands:
        raise ValueError("No alias band lies within p_range.")
    # orbit_solved_problems.md, "V² amplitude" row: per-night contrast loss, profiled; shared
    # by the first pass of ``primary_diam="auto"``.
    if isinstance(vis_amp, (bool, onp.bool_)):
        vis_amp = _default_vis_amp(data) if vis_amp else None
    elif vis_amp is not None:
        for d in data:
            _refuse_amp_with_gains(d)
    if vis_amp is not None:
        vis_amp = tuple(map(float, vis_amp))
        if not 0 < vis_amp[0] < vis_amp[1]:
            raise ValueError("vis_amp must be (lo, hi) with 0 < lo < hi.")
    # orbit_solved_problems.md, "Resolved primary in the orbit fit, automatically" row: a
    # first pass estimates the diameter and reseeds the positions with it.
    # orbit_solved_problems.md, "Untempered evidence" row: the factors f come from the
    # residuals of the per-night fits at the primary finally chosen.
    factors, names = None, sorted({n for d in data for n in _Surface(d).names})
    estimate_f, fixed_f = False, False
    if isinstance(temper, (bool, onp.bool_)):
        estimate_f = bool(temper)
    elif temper is not None and not (
        onp.isfinite(float(temper)) and float(temper) >= 1.0
    ):
        raise ValueError(
            f"temper must be a bool or a fixed finite f >= 1 (tempering "
            f"never sharpens), got {temper!r}."
        )
    elif temper is not None:
        fixed_f = True
        factors = ChannelFactors(
            float(temper),
            dict.fromkeys(names, float(temper)),
            {},
            {},
            f_raw=float(temper),
        )
    can_estimate = estimate_f and isinstance(positions, EpochPositions)
    piloted = False
    primary_fit = None
    diam_seed = 0.5
    if isinstance(primary_diam, str) and primary_diam == "auto":
        positions_in = positions
        # The first pass is untempered unless f is fixed (then it is
        # tempered by it); with f estimated, a disc it resolves is rechecked
        # on the tempered score with f estimated at the disc (a point
        # model's f would be inflated by the very deficit the disc explains).
        primary_diam, positions, estimate, primary_fit = _auto_primary(
            data,
            positions,
            vis_amp,
            block_dof=factors.block_dof() if fixed_f else None,
        )
        if can_estimate:
            piloted = True
            factors = _temper_factors(
                data,
                positions,
                _pilot_diameter(positions, None),
                vis_amp,
                names,
            )
            if estimate.resolved and factors.f > 1.0:
                primary_diam, positions, estimate, primary_fit = _auto_primary(
                    data, positions_in, vis_amp, block_dof=factors.block_dof()
                )
                if not estimate.resolved:  # tempered away: a point primary
                    factors = _temper_factors(
                        data,
                        positions,
                        _pilot_diameter(positions, None),
                        vis_amp,
                        names,
                    )
        if primary_diam == "fit":
            if diam_range is None:
                diam_range = (
                    _DIAM_MIN_AUTO,
                    max(
                        _DIAM_TOP_AUTO * float(fringe_scales(data)[0]),
                        2 * estimate.diam,
                    ),
                )
            diam_seed = float(
                onp.clip(
                    estimate.diam, diam_range[0] * 1.01, diam_range[1] * 0.99
                )
            )
    if diam_range is None:
        diam_range = (0.05, 5.0)
    if can_estimate and not piloted:
        factors = _temper_factors(
            data,
            positions,
            _pilot_diameter(positions, primary_diam),
            vis_amp,
            names,
        )
    elif (
        estimate_f
        and not can_estimate
        and any(
            len(p[k][0])
            for d in data
            for p in [adjacent_channels(d)]
            for k in p
        )
    ):
        warnings.warn(
            "fit_orbit_aliases(): the data have several channels but "
            "temper=True needs EpochPositions to estimate the "
            "effective-channel factor from: the evidence is untempered. "
            "Pass EpochPositions, or a fixed temper=f.",
            UserWarning,
            stacklevel=external_stacklevel(),
        )
    # orbit_solved_problems.md, "Single-night alias peaks" row: a flagged night must not
    # seed the bands; ``data`` keeps it, so the likelihood still sees it.
    seed_positions = positions
    positions, unseeded, fell_back = _seeding_positions(
        positions, t_ref, skip_flagged, mean_times
    )
    profile, counts, motion = None, None, None
    resolution_mas = float(fringe_scales(data)[0])
    arc_jitter_mas = float(arc_jitter) * resolution_mas
    # The positions' gaps and covariances come from the untempered surface:
    # the cut uses them tempered, as the evidence is. The gap sums the
    # blocks, each tempered by its own f_b: the largest is the safe one.
    temper_f = 1.0 if factors is None else float(factors.f_max)
    arcs_possible = (
        positions is not None
        and short_arcs
        and skip_flagged
        and isinstance(seed_positions, EpochPositions)
    )
    arcs_decided = arcs_possible and _signs_trusted(
        seed_positions, arc_min_gap, temper_f
    )
    if arcs_possible and not arcs_decided:
        _LOG.info(
            "short arcs skipped: a seeding night's gap_marginal / f (f = "
            "%.3g) is below arc_min_gap = %.3g nats; every band is seeded "
            "from the positions.",
            temper_f,
            arc_min_gap,
        )
    if arcs_decided:
        # orbit_solved_problems.md, "Cycle counts from short arcs" row: the nights' motion
        # narrows the bands and seeds each one. The cut fits signed
        # positions, so it needs every seeding night's sign decided: by
        # the flags and its gap over its reflection (a label swap). Other
        # positions seed every band as before.
        profile, counts, motion = _short_arc_profile(
            data,
            seed_positions,
            positions,
            bands,
            skip_flagged=skip_flagged,
            p_range=p_range,
            arc_jitter=float(arc_jitter),
            ecc_max=ecc_max,
            jitter_mas=arc_jitter_mas,
            cov_scale=temper_f,
            delta_chi2=arc_delta_chi2,
            k=k,
            n_best=n_candidates,
            eccs=eccs,
            sign_free=sign_free,
        )
    elif positions is not None:
        # orbit_solved_problems.md, "Label swap" row: sign_free seeds both labelings.
        profile = position_profile(
            positions,
            bands,
            k=k,
            n_best=n_candidates,
            eccs=eccs,
            sign_free=sign_free,
        )
    if primary_diam is not None and primary_diam != "fit":
        primary_diam = float(primary_diam)
        if not primary_diam >= 0:
            raise ValueError("primary_diam must be None, >= 0, or 'fit'.")
    _check_seed_diameter(seed_positions, primary_diam)
    # Seeds from few nights (position-level orbits from two or three
    # positions are loosely determined, and one wrong night misleads all
    # of them) keep random candidates beside them; a fallback with no
    # seeds at all must supply random starts even if n_random=0 was given.
    if positions is None:
        n_random = (
            _N_RANDOM_NONE
            if n_random is None
            else max(n_random, _N_RANDOM_FEW)
            if fell_back
            else n_random
        )
    elif n_random is None:
        n_random = 0 if len(positions.dt) > _FEW_SEED_NIGHTS else _N_RANDOM_FEW
    init_flat = None  # (period, dt_peri, ecc, inc, omega, Omega, a_mas, flux)
    if isinstance(init, dict):
        keys = _ELEMENTS + ("flux",)
        missing = [k for k in keys if k not in init]
        if missing:
            raise ValueError(f"init is missing {missing}.")
        init_flat = tuple(float(init[k]) for k in keys)
    elif init is not None:
        orbit, flux = init
        init_flat = (
            float(orbit.period),
            float(orbit.dt_peri) + float(orbit.t_ref) - t_ref,
            float(orbit.ecc),
            float(orbit.inc),
            float(orbit.omega),
            float(orbit.Omega),
            float(orbit.a_mas),
            float(flux),
        )
    n_scan = _N_FLUX + (2 if sign_free else 0)
    # (with sign_free the signed and the sign-free orbits both seed)
    n_orbits = n_candidates * (2 if sign_free else 1)
    n_seeded = 0 if positions is None else 2 * n_scan * n_orbits
    if counts is not None:  # the arc seeds come beside the position seeds
        n_seeded *= 2
    n_pool = n_random + n_seeded + (1 if init else 0)  # senses, fluxes, init
    settings = _Settings(
        p_range=tuple(map(float, p_range)),
        a_range=tuple(map(float, a_range)),
        flux_range=tuple(map(float, flux_range)),
        ecc_max=ecc_max,
        s_max=s_max,
        t_ref=t_ref,
        n_candidates=n_candidates,
        n_refine=n_refine,
        n_random=n_random,
        pool_size=-(-n_pool // 64) * 64,
        k=k,
        eccs=None if eccs is None else tuple(map(float, eccs)),
        min_distance_mas=min_distance_mas,
        max_steps=max_steps,
        n_is=n_is,
        flag_nats=flag_nats,
        ess_min=ess_min,
        seed=seed,
        sign_free=sign_free,
        primary_diam=primary_diam,
        diam_range=tuple(map(float, diam_range)),
        diam_seed=diam_seed,
        vis_amp=vis_amp,
        flip_flux=bool(flip_flux),
        resolution_mas=resolution_mas,
        outlier=outlier,
        init=init_flat,
        dof=_dof_pairs(factors),
        short_arcs=counts is not None,
        arc_jitter_mas=arc_jitter_mas if counts is not None else 0.0,
        arc_delta_chi2=float(arc_delta_chi2),
    )
    if factors is not None and not fixed_f:
        _warn_tempering(
            factors,
            any(
                d.gains is not None or d.phase_offsets is not None
                for d in data
            ),
        )
    settings = _broad_components(
        data, settings, primary_diam, diam_seed, resolution_mas,
        positions=positions, batch_size=batch_size, memory_budget=memory_budget,
    )  # fmt: skip
    problem = _Problem(data, settings, batch_size, memory_budget)
    _preflight(problem, settings, bands[0])
    key = (
        _checkpoint_key(data, positions, bands, mean_times, settings)
        if checkpoint_dir is not None
        else None
    )
    seeds = {} if profile is None else {r["n"]: r["orbits"] for r in profile}
    if counts is not None:
        # The arc seeds are additive: each band also keeps the position-level
        # seeds it would have without short arcs, after the arc seeds (which
        # the cheap check scores). They stay out of ``profile``, whose rows
        # are the arc fit alone.
        for row in position_profile(
            positions,
            bands,
            k=k,
            n_best=n_candidates,
            eccs=eccs,
            sign_free=sign_free,
        ):
            seeds[row["n"]] = seeds[row["n"]] + row["orbits"]
    short_arc = None
    if counts is not None:
        bands, short_arc = _short_arc_bands(
            problem, settings, bands, seeds, counts, motion
        )

    results, store, error = [], {}, None
    for band in bands:
        done = _checkpoint(checkpoint_dir, band, key)
        if done is None:
            try:
                done = _fit_band(
                    problem,
                    settings,
                    band,
                    seeds.get(band.n),
                    mean_times,
                )
            except TypeError:
                raise
            except Exception as caught:  # noqa: BLE001 - keep the other bands
                error = caught
                failed = AliasBandResult(
                    n=band.n,
                    p_lo=band.p_lo,
                    p_hi=band.p_hi,
                    modes=[],
                    log_z=-onp.inf,
                    log_z_is=-onp.inf,
                    ess=0.0,
                    flagged=True,
                    flags=["band-failed"],
                    error=f"{type(caught).__name__}: {caught}",
                )
                done = (failed, (None, None))
            else:
                _checkpoint(checkpoint_dir, band, key, done)
        result, store[band.n] = done
        results.append(result)
        if on_band is not None:
            on_band(result)
    if all("band-failed" in b.flags for b in results):
        raise RuntimeError(
            f"Every band failed; the last error was: {results[-1].error}"
        ) from error

    log_z = onp.array([b.log_z for b in results])
    finite = onp.isfinite(log_z)
    weights = (
        onp.where(finite, onp.exp(log_z - onp.max(log_z[finite])), 0.0)
        if finite.any()
        else onp.zeros_like(log_z)
    )
    for b, w in zip(results, weights / max(weights.sum(), 1e-300)):
        b.p = float(w)
    order = onp.argsort(-onp.where(finite, log_z, -onp.inf), kind="stable")
    results = [results[i] for i in order]  # by decreasing log_z
    best = results[0]
    if best.chi2_red > _POOR_FIT_CHI2_RED:  # False for NaN too
        best.flags.append("poor-fit")
        best.flagged = True
        warnings.warn(
            f"fit_orbit_aliases(): the best band (N={best.n}) has chi2_red = "
            f"{best.chi2_red:.3g} on the quoted errors (above "
            f"{_POOR_FIT_CHI2_RED:g}). Either the optimizer missed the narrow "
            "likelihood ridge, so the band probabilities are not trustworthy "
            f"(try a larger n_refine or k, now {k:g}), or the errors are "
            "underestimated by more than 3x.",
            UserWarning,
            stacklevel=external_stacklevel(),
        )
    samples = {}  # importance resamples: the best band and each with p > p_min
    for rank, b in enumerate(results):
        if rank > 0 and b.p <= p_min_samples:
            continue
        xs, lw = store[b.n]
        if xs is None or not onp.any(onp.isfinite(lw)):
            continue
        w = onp.exp(lw - onp.max(lw))
        pick = onp.random.default_rng([seed, 2**31, b.n]).choice(
            len(xs), size=n_samples, p=w / w.sum()
        )
        samples[b.n] = _physical(xs[pick])
    # orbit_solved_problems.md, "Period window from the data" row: a short arc is flat
    # towards long periods, so P is unconstrained (``period_bound``).
    # It is also flagged when the best mode's σ(log P) is a quarter of the
    # window; and ``period_at_lower`` when P is near the bottom, where the
    # window's sampling limit may have cut the truth.
    period_bound = period_at_lower = False
    period = best.best.get("period")
    if period is not None:
        p_lo, p_hi = float(p_range[0]), float(p_range[1])
        live = [m for m in best.modes if onp.isfinite(m["log_z"])]
        sigma = float("nan")
        if live:
            top = max(live, key=lambda m: m["log_z"])
            sigma = float(onp.atleast_1d(top.get("sigma", [float("nan")]))[0])
        wide = onp.isfinite(sigma) and sigma >= _PERIOD_SIGMA_FRACTION * (
            math.log(p_hi / p_lo)
        )
        period_bound = bool(period >= _PERIOD_BOUND_FRACTION * p_hi or wide)
        period_at_lower = bool(period <= _LOWER_BOUND_FACTOR * p_lo)
    result = AliasResult(
        t_ref,
        mean_times,
        float(onp.ptp(mean_times)),
        results,
        samples,
        profile,
        primary_diam=primary_diam,
        vis_amp=vis_amp,
        unseeded=unseeded,
        p_range=tuple(map(float, p_range)),
        period_bound=period_bound,
        primary_fit=primary_fit,
        period_at_lower_bound=period_at_lower,
        window_capped=window_capped,
        short_arc=short_arc,
        temper_f=None if factors is None else factors.f,
        temper_blocks=None if factors is None else dict(factors.blocks),
        temper_f_raw=None if factors is None else factors.f_raw,
        prior=dict(
            a_range=settings.a_range,
            flux_range=settings.flux_range,
            ecc_max=settings.ecc_max,
            s_max=settings.s_max,
            diam_range=settings.diam_range,
        ),
        temper_rho=None
        if factors is None or not factors.rho
        else {
            k: [
                float(factors.rho[k]),
                float(factors.rho2.get(k, float("nan"))),
                int(factors.n_pairs2.get(k, 0)),
            ]
            for k in factors.rho
        },
    )
    result.outlier, result.night_flags = _night_diagnostics(
        problem, settings, result, seed_positions, data, t_ref
    )
    return result
