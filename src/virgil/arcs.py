"""Short arcs: the motion within and between nights narrows the alias bands.

!!! warning "Provisional"
    May change in any 0.4.x release without a deprecation cycle.

Nights far apart leave one alias band per whole number of cycles between
them ([`alias_bands`][virgil.aliases.alias_bands]). Nights close together,
and the motion within a night, see short arcs of the orbit, whose speed and
curvature bound the period.
[`night_motion`][virgil.arcs.night_motion] measures each night's velocity
from its two halves. [`unwrap_cycles`][virgil.arcs.unwrap_cycles] joins the
nights, shortest gap first, and keeps at each step the periods whose
Thiele–Innes fit to the positions and velocities joined so far is close to
the best, so each longer gap admits only the cycle counts that the shorter
arcs allow. [`arc_profile`][virgil.arcs.arc_profile] gives each band's best
orbits, which seed [`fit_orbit_aliases`][virgil.aliases.fit_orbit_aliases].
The [orbit tutorial](../orbit_fitting.md#short-arcs-cycle-counts-from-the-motion)
gives the reasoning.
"""

import dataclasses

import jax
import jax.numpy as np
import numpy as onp

from . import _gauss_newton
from ._geometry import fringe_scales
from ._precision import cast_tree, run_in
from .epochs import (
    _REFINE_WINDOW,
    Epochs,
    _negative_marginal_hessian,
    _refine_marginal,
    _Surface,
)
from .orbits import (
    ThieleInnesOrbit,
    _unit_orbit,
    _velocity,
    period_grid,
    starting_orbits,
)

__all__ = [
    "ARC_DELTA_CHI2",
    "ARC_JITTER",
    "ARC_MIN_GAP",
    "CycleCounts",
    "NightMotion",
    "arc_profile",
    "night_motion",
    "unwrap_cycles",
]

__provisional__ = ("*",)

# A position floor, as a fraction of the finest fringe spacing λ/B_max, added
# in quadrature to every night's position error: a night's position is a
# lossy summary, often biased by a fraction of the resolution, and a period
# must not be excluded because one night's formal error is tiny.
ARC_JITTER = 0.1
# A period stays admissible while its χ² is within this of the best (after
# the errors are scaled up to the best fit's χ²/dof, when above 1): 5σ for one
# parameter, generous so that the true cycle count is never cut.
ARC_DELTA_CHI2 = 25.0
# The cut fits signed positions, so every seeding night's sign must be
# decided: its best peak must beat its point reflection (a label swap; a
# twin's reflection ties) by this many nats. ``epoch_positions``'
# ``gap_marginal`` bounds that margin from below whenever the reflection,
# 2 × the separation away, is a rival (more than λ/B_max from the peak);
# its ``sep-below-resolution`` flag catches the rest. As its ``low-gap``
# flag's default: 5 nats, a likelihood ratio of about 150.
ARC_MIN_GAP = 5.0
# A half-night position further than this fraction of λ/B_max from the
# night's own is another peak, not motion: that night has no velocity.
_MAX_SHIFT = 0.5
_N_PARAMS = 7  # period, phase, eccentricity and four Thiele–Innes constants
_N_PHASE = 36  # trial times of periastron per period (as starting_orbits)
_BATCH = 4096  # grid points solved together


@dataclasses.dataclass(frozen=True)
class NightMotion:
    """Each night's angular velocity, from
    [`night_motion`][virgil.arcs.night_motion].

    Attributes
    ----------
    mjd : numpy.ndarray
        Each night's time (MJD), as in the positions.
    vel : numpy.ndarray
        ``(n, 2)`` velocity (mas/day, East and North); 0 where not measured.
    cov : numpy.ndarray
        ``(n, 2, 2)`` covariance (mas²/day²); the identity where not measured.
    measured : numpy.ndarray
        Whether each night's velocity was measured.
    """

    mjd: onp.ndarray
    vel: onp.ndarray
    cov: onp.ndarray
    measured: onp.ndarray

    def take(self, keep):
        """The nights selected by ``keep`` (a mask or indices)."""
        return NightMotion(
            self.mjd[keep], self.vel[keep], self.cov[keep], self.measured[keep]
        )


@dataclasses.dataclass(frozen=True)
class CycleCounts:
    """The periods the short arcs admit, from
    [`unwrap_cycles`][virgil.arcs.unwrap_cycles].

    Attributes
    ----------
    intervals : tuple of (float, float)
        The admissible periods (days), disjoint and increasing.
    steps : tuple of dict
        One per constraining step: the nights joined (``mjd``), the
        ``baseline`` (days), ``dof``, the best ``chi2``, the error ``scale``
        and the admissible ``intervals`` after it.
    """

    intervals: tuple
    steps: tuple

    def admits(self, band):
        """Whether ``band`` (an ``AliasBand``) overlaps an admissible
        interval."""
        return any(
            lo < band.p_hi and band.p_lo < hi for lo, hi in self.intervals
        )


# ---------------------------------------------------------------------------
# Within-night motion
# ---------------------------------------------------------------------------


def _half_position(
    part, x0, resolution, diam
):  # deliberate: 31 lines, six early exits; test patches _refine_marginal through it
    """The refined ``(dra, ddec)`` of ``part`` from ``x0`` and its
    covariance, or None if the refinement fails."""
    window = _REFINE_WINDOW * resolution
    bounds = [
        (x0[0] - window, x0[0] + window),
        (x0[1] - window, x0[1] + window),
        (1e-4, 1.0),
    ]
    with run_in("float64"):
        d64 = cast_tree(part, "float64")
        surface = _Surface(d64)
        dof = np.asarray(surface.dof_vector(None), float)
        dm = None if not diam else np.asarray(float(diam))
        x, value, stop = _refine_marginal(x0, dof, d64, surface, bounds, dm)
        # Only a converged refinement measures a velocity: a stalled or
        # step-limited one keeps its best point, which may be a long way
        # from the half's peak.
        if stop != _gauss_newton.CONVERGED or not onp.isfinite(value):
            return None
        hess = onp.asarray(
            _negative_marginal_hessian(np.asarray(x), dof, d64, surface, dm)
        )
    if not onp.all(onp.isfinite(hess)):
        return None
    cov = onp.linalg.pinv(hess)[:2, :2]
    if not (
        onp.all(onp.isfinite(cov)) and onp.all(onp.linalg.eigvalsh(cov) > 0)
    ):
        return None
    return x[:2], cov


def night_motion(
    data,
    positions,
    *,
    max_shift=_MAX_SHIFT,
    jitter=ARC_JITTER,
    cov_scale=1.0,
):
    """Each night's angular velocity, from the positions of its two halves.

    Each night's frames are split into an earlier and a later half, each
    refined from the night's
    ([`epoch_positions`][virgil.epochs.epoch_positions]) position on the same
    scale-marginalized surface, with the same primary. The
    velocity is the difference over the time between them, with a covariance
    floor of ``jitter`` λ/B_max. A night with one frame, or whose halves move
    more than ``max_shift`` λ/B_max, has none.

    Parameters
    ----------
    data : Epochs or sequence of OIData
        The nights, in the order of ``positions``.
    positions : EpochPositions
        Each night's position.
    max_shift, jitter : float, optional
        The furthest a half may move from the night's position, and each
        half's position floor, in λ/B_max (defaults 0.5, ``ARC_JITTER``).
    cov_scale : float, optional
        Factor (at least 1) multiplying the halves' covariances, σ by its
        square root, before the floor is added (default 1).

    Returns
    -------
    NightMotion
    """
    _check_cov_scale(cov_scale)
    sets = list(data.data) if isinstance(data, Epochs) else list(data)
    n = len(sets)
    if n != len(positions.names):
        raise ValueError(
            f"{n} datasets but positions for {len(positions.names)}: they "
            "must be the same nights."
        )
    vel = onp.zeros((n, 2))
    cov = onp.tile(onp.eye(2), (n, 1, 1))
    measured = onp.zeros(n, bool)
    for i, d in enumerate(sets):
        # Split the night's frames into an earlier and a later half.
        if d.mjd is None or d.frame is None:
            continue
        frames, frame_of = onp.unique(
            onp.asarray(d.frame), return_inverse=True
        )
        if frames.size < 2:
            continue
        times = onp.bincount(frame_of, weights=d.mjd) / onp.bincount(frame_of)
        late = onp.zeros(frames.size, bool)
        late[onp.argsort(times)[frames.size // 2 :]] = True
        if onp.ptp(times) <= 0:
            continue
        try:
            halves = [
                (
                    d._subset(late[frame_of] == side),
                    float(onp.mean(times[late == side])),
                )
                for side in (False, True)
            ]
        except ValueError:  # projected data, offsets, OI_FLUX
            continue
        resolution = float(fringe_scales(d)[0])
        x0 = onp.array(
            [
                positions.dra[i],
                positions.ddec[i],
                min(max(float(positions.flux[i]), 1e-4), 1.0),
            ]
        )
        fits = [
            _half_position(part, x0, resolution, positions.primary_diam)
            for part, _ in halves
        ]
        if any(f is None for f in fits):
            continue
        (r1, c1), (r2, c2) = fits
        if max(onp.hypot(*(r - x0[:2])) for r in (r1, r2)) > (
            max_shift * resolution
        ):
            continue
        dt = halves[1][1] - halves[0][1]
        vel[i] = (r2 - r1) / dt
        floor = 2.0 * (jitter * resolution) ** 2 * onp.eye(2)
        cov[i] = (cov_scale * (c1 + c2) + floor) / dt**2
        measured[i] = True
    return NightMotion(onp.asarray(positions.mjd, float), vel, cov, measured)


# ---------------------------------------------------------------------------
# Thiele–Innes fits to positions and velocities
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class _Arc:
    """Positions and velocities, whitened: times ``t`` (days from
    ``t_ref``), ``r`` and ``v`` ``(n, 2)`` (dra, ddec), whiteners ``wr`` and
    ``wv`` ``(n, 2, 2)`` (zero for a velocity not measured)."""

    t: onp.ndarray
    r: onp.ndarray
    wr: onp.ndarray
    v: onp.ndarray
    wv: onp.ndarray
    t_ref: float

    def take(self, keep):
        return _Arc(
            self.t[keep],
            self.r[keep],
            self.wr[keep],
            self.v[keep],
            self.wv[keep],
            self.t_ref,
        )

    @property
    def n_data(self):
        """The number of measured coordinates."""
        return 2 * len(self.t) + 2 * int(onp.any(self.wv != 0, (1, 2)).sum())


def _whitener(cov):
    return onp.linalg.inv(onp.linalg.cholesky(cov))


def _check_cov_scale(cov_scale):
    if not onp.isfinite(cov_scale) or cov_scale < 1.0:
        raise ValueError(
            f"cov_scale must be a finite factor >= 1 (errors are never "
            f"sharpened), got {cov_scale!r}."
        )


def _position_cov(positions, jitter_mas, cov_scale=1.0):
    """The covariances of ``PositionData`` (from its whiteners) times
    ``cov_scale``, with ``jitter_mas`` added in quadrature."""
    w = onp.asarray(positions.whitener, float)
    cov = onp.linalg.inv(onp.einsum("nji,njk->nik", w, w))
    return cov_scale * cov + jitter_mas**2 * onp.eye(2)


def _arc(positions, motion, jitter_mas, cov_scale=1.0):
    """The ``_Arc`` of ``PositionData`` and the aligned ``NightMotion`` (or
    None), with the positions' covariances times ``cov_scale`` and
    ``jitter_mas`` added in quadrature to each position."""
    _check_cov_scale(cov_scale)
    cov = _position_cov(positions, jitter_mas, cov_scale)
    n = len(cov)
    if motion is None:
        v, wv = onp.zeros((n, 2)), onp.zeros((n, 2, 2))
    else:
        if len(motion.measured) != n:
            raise ValueError(
                f"motion has {len(motion.measured)} nights but there are "
                f"{n} positions."
            )
        v = onp.where(motion.measured[:, None], motion.vel, 0.0)
        wv = onp.where(
            motion.measured[:, None, None], _whitener(motion.cov), 0.0
        )
    return _Arc(
        onp.asarray(positions.dt, float),
        onp.stack(
            [onp.asarray(positions.dra), onp.asarray(positions.ddec)], -1
        ),
        _whitener(cov),
        v,
        wv,
        float(positions.t_ref),
    )


@jax.jit
def _pv_fit(
    t, r, wr, v, wv, period, dt_peri, ecc
):  # deliberate: jit/vmap boundary
    """Least-squares ``(A, B, F, G)`` and χ² of positions and velocities at
    one grid point: velocities are linear in the constants too."""

    def unit(s):
        return np.stack(_unit_orbit(s, period, dt_peri, ecc))

    def design(xy, w):
        x, y = xy
        zero = np.zeros_like(x)
        rows = np.stack(
            [
                np.stack([zero, x, zero, y], -1),
                np.stack([x, zero, y, zero], -1),
            ],
            -2,
        )  # rows (dra, ddec); columns A, B, F, G
        return np.einsum("nij,njk->nik", w, rows).reshape(-1, 4)

    a = np.concatenate([design(unit(t), wr), design(_velocity(unit, t), wv)])
    b = np.concatenate(
        [
            np.einsum("nij,nj->ni", wr, r).reshape(-1),
            np.einsum("nij,nj->ni", wv, v).reshape(-1),
        ]
    )
    params = np.linalg.lstsq(a, b)[0]
    resid = b - a @ params
    return params, resid @ resid


_pv_batch = jax.jit(jax.vmap(_pv_fit, in_axes=(None,) * 5 + (0, 0, 0)))


def _eccs(eccs, ecc_max):
    if eccs is not None:
        return onp.asarray(eccs, float)
    return onp.arange(0.0, ecc_max + 1e-9, 0.05)


def _pv_grid(arc, periods, eccs, n_phase=_N_PHASE):
    """The grid ``(period, dt_peri, ecc)`` over ``periods × eccs ×
    phases`` (period-major), with each point's constants and χ²."""
    phases = onp.arange(n_phase) / n_phase
    p, e, f = onp.meshgrid(periods, eccs, phases, indexing="ij")
    grid = onp.stack([p.ravel(), (f * p).ravel(), e.ravel()], axis=1)
    return (grid, *_pv_points(arc, grid))


def _pv_points(arc, grid):
    """Each ``(period, dt_peri, ecc)`` row's constants and χ²."""
    size = min(_BATCH, 1 << (len(grid) - 1).bit_length())
    padded = onp.concatenate(
        [grid, onp.repeat(grid[-1:], -len(grid) % size, axis=0)]
    )
    with run_in("float64"):
        fixed = [np.asarray(x) for x in (arc.t, arc.r, arc.wr, arc.v, arc.wv)]
        out = [
            _pv_batch(
                *fixed,
                *(np.asarray(padded[i : i + size, k]) for k in range(3)),
            )
            for i in range(0, len(padded), size)
        ]
    params = onp.concatenate([onp.asarray(o[0]) for o in out])[: len(grid)]
    chi2 = onp.concatenate([onp.asarray(o[1]) for o in out])[: len(grid)]
    return params, onp.where(onp.isfinite(chi2), chi2, onp.inf)


_STEPS = onp.stack(
    onp.meshgrid(*[(-1.0, 0.0, 1.0)] * 3, indexing="ij"), -1
).reshape(-1, 3)


# ---------------------------------------------------------------------------
# Cycle counts and seeds
# ---------------------------------------------------------------------------


def _merge(intervals):
    """Union of ``(lo, hi)`` intervals, disjoint and increasing."""
    out = []
    for lo, hi in sorted(intervals):
        if out and lo <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], hi))
        else:
            out.append((lo, hi))
    return out


def unwrap_cycles(
    positions,
    motion=None,
    *,
    p_range,
    ecc_max=0.9,
    jitter_mas=0.0,
    cov_scale=1.0,
    delta_chi2=ARC_DELTA_CHI2,
    k=9.0,
    eccs=None,
):
    """The periods that the short arcs admit, unwrapping the cycle counts.

    The nights are joined into arcs shortest gap first. Where an arc has more
    measured coordinates than the seven orbital parameters, each period is
    scored by its best Thiele–Innes fit and kept while its χ² is within
    ``delta_chi2 * max(1, chi2_min / dof)`` of the best. Too few data leave
    the whole window: weak constraints keep more periods, never fewer.

    Parameters
    ----------
    positions : PositionData
        The nights' positions.
    motion : NightMotion, optional
        The nights' velocities ([`night_motion`][virgil.arcs.night_motion]).
    p_range : (float, float)
        The period window (days).
    ecc_max, jitter_mas, cov_scale, delta_chi2, k, eccs : optional
        The largest eccentricity; a position floor (mas); the factor (at
        least 1) multiplying the positions' covariances, σ by its square
        root (match ``cov_scale`` in [`night_motion`][virgil.arcs.night_motion]);
        the χ² margin; the phase coherence of the period grids; trial eccentricities (default 0
        to ``ecc_max`` in steps of 0.05).

    Returns
    -------
    CycleCounts
    """
    arc = _arc(positions, motion, jitter_mas, cov_scale)
    eccs = _eccs(eccs, ecc_max)
    order = onp.argsort(arc.t, kind="stable")
    group = onp.arange(len(arc.t))
    intervals = [tuple(map(float, p_range))]
    steps = []
    gaps = onp.diff(arc.t[order])
    for j in onp.argsort(gaps, kind="stable"):
        group[group == group[order[j + 1]]] = group[order[j]]
        members = onp.flatnonzero(group == group[order[j]])
        sub = arc.take(members)
        dof = sub.n_data - _N_PARAMS
        baseline = float(onp.ptp(sub.t))
        if dof < 1 or baseline <= 0:
            continue
        periods = onp.concatenate(
            [
                period_grid(sub.t, lo, hi, k)
                for lo, hi in intervals
                if hi > lo * (1 + 1e-12)
            ]
        )
        grid, _, chi2 = _pv_grid(sub, periods, eccs)
        # The period grid's and the (τ, e) grid's steps.
        ecc_step = float(onp.max(onp.diff(eccs))) if len(eccs) > 1 else ecc_max
        spacing = onp.stack(
            [
                periods**2 / (k * baseline),
                periods / _N_PHASE,
                onp.full(len(periods), ecc_step),
            ],
            -1,
        )
        # Lower each period's grid minimum by a pattern search in (period,
        # dt_peri, ecc), halving the steps each time: a coarse grid only
        # bounds the profile χ² from above, and the cut must not exclude a
        # period for falling between grid points.
        n_p = len(spacing)
        chi2 = chi2.reshape(n_p, -1)
        pick = chi2.argmin(axis=1)
        x = grid.reshape(n_p, -1, 3)[onp.arange(n_p), pick]
        per_period = chi2[onp.arange(n_p), pick]
        for _ in range(4):
            spacing = spacing / 2.0
            trial = x[:, None, :] + _STEPS[None] * spacing[:, None, :]
            trial[..., 2] = onp.clip(trial[..., 2], 0.0, ecc_max)
            _, c = _pv_points(sub, trial.reshape(-1, 3))
            c = c.reshape(n_p, len(_STEPS))
            step = c.argmin(axis=1)
            better = c[onp.arange(n_p), step] < per_period
            x = onp.where(better[:, None], trial[onp.arange(n_p), step], x)
            per_period = onp.where(
                better, c[onp.arange(n_p), step], per_period
            )
        best = float(per_period.min())
        scale = max(1.0, best / dof)
        keep = per_period <= best + delta_chi2 * scale
        half = 1.0 / (2.0 * k * baseline)  # at least half a grid step
        admitted = [
            (1.0 / (1.0 / p + half), 1.0 / max(1.0 / p - half, 1e-300))
            for p in periods[keep]
        ]
        admitted = _merge(admitted)
        intervals = _merge(
            [
                (max(lo1, lo2), min(hi1, hi2))
                for lo1, hi1 in intervals
                for lo2, hi2 in admitted
                if max(lo1, lo2) < min(hi1, hi2)
            ]
        )
        steps.append(
            dict(
                mjd=(sub.t + arc.t_ref).tolist(),
                baseline=baseline,
                dof=int(dof),
                chi2=best,
                scale=scale,
                intervals=[list(i) for i in intervals],
            )
        )
    return CycleCounts(tuple(tuple(i) for i in intervals), tuple(steps))


def arc_profile(
    positions,
    motion,
    bands,
    *,
    ecc_max=0.9,
    jitter_mas=0.0,
    cov_scale=1.0,
    k=9.0,
    n_best=3,
    eccs=None,
    sign_free=False,
):
    """Each band's best orbits from the positions and the velocities.

    As [`position_profile`][virgil.aliases.position_profile], with each
    night's velocity in the Thiele–Innes fit as well, the positions'
    covariances times ``cov_scale`` and a position floor ``jitter_mas``
    (mas) added in quadrature. With ``sign_free`` the
    sign-free orbits of the positions alone are added (``n_best`` of each).

    Returns
    -------
    list of dict
        Per band: ``n``, ``chi2`` (best), ``period`` of the best orbit and
        the ``orbits`` ``[(KeplerOrbit, χ²), ...]``.
    """
    arc = _arc(positions, motion, jitter_mas, cov_scale)
    eccs = _eccs(eccs, ecc_max)
    times = arc.t + arc.t_ref
    rows = []
    for band in bands:
        periods = period_grid(times, band.p_lo, band.p_hi, k=k)
        grid, params, chi2 = _pv_grid(arc, periods, eccs)
        best = onp.argsort(chi2, kind="stable")[:n_best]
        found = [
            (
                ThieleInnesOrbit(
                    *grid[i], *params[i], t_ref=arc.t_ref
                ).to_kepler(),
                float(chi2[i]),
            )
            for i in best
        ]
        if sign_free:  # orbit_solved_problems.md, "Label swap" row
            found = found + starting_orbits(
                positions
                if not jitter_mas and cov_scale == 1.0
                else type(positions)(
                    onp.asarray(positions.dt, float) + positions.t_ref,
                    onp.asarray(positions.dra),
                    onp.asarray(positions.ddec),
                    _position_cov(positions, jitter_mas, cov_scale),
                    t_ref=positions.t_ref,
                ),
                periods,
                eccs=eccs,
                n_best=n_best,
                sign_free=True,
            )
        top = min(found, key=lambda row: row[1])
        rows.append(
            dict(
                n=band.n,
                chi2=float(top[1]),
                period=float(top[0].period),
                orbits=found,
            )
        )
    return rows
