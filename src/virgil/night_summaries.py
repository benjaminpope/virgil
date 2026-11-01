"""Per-night, per-peak (position, velocity, flux) summaries at visibility level.

!!! warning "Provisional"
    May change in any 0.4.x release without a deprecation cycle.

Each night's companion is fitted peak by peak with linear motion over the whole
night, ``p(t) = p + v (t - t̄)``, on the visibilities with the noise model of
the data (gains, closure offsets and spanning gain modes marginalized as in
[`whitened_residuals`][virgil.likelihood.whitened_residuals]). The summary of a
peak is the mean and covariance of ``(p, v, ln f)`` at the quoted errors, with
the per-block counts, minimum χ² and Hessians that rescale it to another error
scale in closed form (for data without gains or closure offsets, where a scale
factors out of the covariance). For frames of error ``σ`` the velocity variance
is ``σ² / Σ_j (t_j - t̄)²``; in terms of the night position's error
``σ_p² = σ² / N`` it is ``12 σ_p² / span²`` for uniform sampling. A short or
faint night's velocity therefore carries no weight and no threshold is needed.
"""

import dataclasses

import equinox as eqx
import jax
import jax.numpy as np
import numpy as onp

from . import _gauss_newton
from ._geometry import fringe_scales
from ._precision import cast_tree, run_in
from .aliases import _epoch_residuals, _UDBinary
from .epochs import _REFINE_WINDOW, Epochs, _Surface
from .likelihood import model_loglike, whitened_residuals
from .models import OrbitalBinary

__all__ = [
    "DELTA",
    "NightPeak",
    "NightSummary",
    "keep_peaks",
    "night_summaries",
]

__provisional__ = ("*",)

# Reporting precision of a probability (design §10): a peak is dropped only
# if the night's outlier term outweighs it by more than 1/DELTA.
DELTA = 1e-3
# The engine's flux floor (arcs._half_position); the ln f upper bound is 0.
_F_MIN = 1e-4


@dataclasses.dataclass(frozen=True)
class NightPeak:
    """One peak of one night.

    Attributes
    ----------
    mean : numpy.ndarray
        ``(dra, ddec, vra, vdec, ln f)``: position (mas) at the night's
        reference time, velocity (mas/day, East and North) and log flux ratio.
    cov : numpy.ndarray
        ``(5, 5)`` covariance of ``mean`` at the quoted errors (a direction the
        night does not constrain gets a variance of ``1 / (eps λ_max)``).
    loglike : float
        Log likelihood of the night at the peak, normalization included.
    n, chi2 : numpy.ndarray
        Per block (``NightSummary.blocks``): independent observables and
        minimum χ² on the quoted errors.
    hessian : numpy.ndarray
        ``(blocks, 5, 5)`` Hessians of ``½ χ²`` per block at the peak.
    rescalable : bool
        Whether ``precision`` and ``log_scale`` rescale the errors in closed
        form: false with gains or closure offsets, whose covariance
        ``s² D + U Uᵀ`` a scale does not factor.
    """

    mean: onp.ndarray
    cov: onp.ndarray
    loglike: float
    n: onp.ndarray
    chi2: onp.ndarray
    hessian: onp.ndarray
    rescalable: bool = True

    def _scales(self, scale):
        scale = onp.broadcast_to(onp.asarray(scale, float), self.n.shape)
        if not self.rescalable and onp.any(scale != 1.0):
            raise NotImplementedError(
                "Closed-form rescaling is not exact for data with gains or "
                "closure offsets: refit at the new scale."
            )
        return scale

    def precision(self, scale=1.0):
        """``Σ⁻¹`` with the errors of block ``g`` multiplied by ``scale[g]``
        (a number or one per block)."""
        return onp.einsum(
            "g,gij->ij", self._scales(scale) ** -2.0, self.hessian
        )

    def covariance(self, scale=1.0):
        """``Σ`` at error ``scale`` (see ``precision``)."""
        return _inverse(self.precision(scale))

    def log_scale(self, scale):
        """``ln L_j(s) - ln L_j(1)`` for block scales ``s``: ``-Σ_g [n_g ln
        s_g + (χ²_g / 2)(s_g⁻² - 1)]`` (the mean is held at its ``s = 1``
        value)."""
        scale = self._scales(scale)
        return -float(
            onp.sum(
                self.n * onp.log(scale) + 0.5 * self.chi2 * (scale**-2.0 - 1.0)
            )
        )


@dataclasses.dataclass(frozen=True)
class NightSummary:
    """One night's catalogue of peaks.

    Attributes
    ----------
    t_ref : float
        The night's reference time ``t̄`` (MJD), the mean frame time.
    t_ss : float
        ``Σ_j (t_j - t̄)²`` over the night's frames (days²); 0 for one frame.
    n_frames : int
        Number of frames (exposures) ``t_ss`` sums over.
    blocks : tuple of str
        The error blocks of ``NightPeak.n``, ``"vis"``, ``"phi"``, ...
    peaks : tuple of NightPeak
        Converged peaks, best first. Empty if none converged.
    loglike_null : float
        The night's log likelihood with no companion.
    """

    t_ref: float
    t_ss: float
    n_frames: int
    blocks: tuple
    peaks: tuple
    loglike_null: float

    def __len__(self):
        return len(self.peaks)

    def covariance(self, peak, scale=1.0, jitter=0.0):
        """``peak``'s covariance at error ``scale`` with a ``jitter`` (mas),
        independent per frame, added in quadrature to every frame's position
        error: the night position (at ``t̄``) gains ``jitter² / n_frames`` and
        the velocity ``jitter² / t_ss`` (infinite for one frame)."""
        add = onp.zeros(5)
        if jitter:
            add[:2] = jitter**2 / self.n_frames
            add[2:4] = jitter**2 / self.t_ss if self.t_ss > 0 else onp.inf
        return peak.covariance(scale) + onp.diag(add)


def _inverse(precision):
    """Inverse of a symmetric non-negative matrix, its eigenvalues floored at
    ``eps λ_max`` (rounding): an unconstrained direction has a huge variance."""
    w, V = onp.linalg.eigh(0.5 * (precision + precision.T))
    w = onp.maximum(w, onp.finfo(float).eps * max(w.max(), 0.0))
    if not w.max() > 0:
        return onp.full(precision.shape, onp.inf)
    return (V / w) @ V.T


class _Linear(eqx.Module):
    """A companion in linear motion ``(p + v dt, 0)``: the orbit of an
    [`OrbitalBinary`][virgil.models.OrbitalBinary] over one night."""

    p: jax.Array
    v: jax.Array
    t_ref: float = eqx.field(static=True)

    def _relative(self, dt):
        return self.p[0] + self.v[0] * dt, self.p[1] + self.v[1] * dt, 0.0 * dt


def _model(z, t_ref, diam):
    orbit = _Linear(z[:2], z[2:4], t_ref)
    flux = np.exp(z[4])
    return (
        OrbitalBinary(orbit, flux)
        if not diam
        else _UDBinary(orbit, flux, diam)
    )


class _Problem(eqx.Module):
    """``½ χ²`` of a night's data for one linear-motion companion, as a
    [`_gauss_newton`][virgil._gauss_newton] kernel problem."""

    data: object
    diam: object
    t_ref: float = eqx.field(static=True)
    surface: _Surface = eqx.field(static=True)
    amp_range: object = eqx.field(static=True, default=None)

    def rows(self, z):
        model = _model(z, self.t_ref, self.diam)
        return _epoch_residuals(
            self.surface, model, self.data, self.amp_range
        )[0]

    def chi2(self, z):
        r = self.rows(z)
        return np.stack(
            [np.sum(r[a:b] ** 2) for _, a, b, _, _ in self.surface.blocks]
        )

    def loss(self, z):
        return 0.5 * np.sum(self.chi2(z))

    def diag(self, z):
        return np.zeros_like(z)


@jax.jit
def _hessians(problem, z):
    return 0.5 * jax.jacfwd(jax.jacrev(problem.chi2))(z)


def _summarize(d, positions, i, diam, amp_range):
    mjd = onp.asarray(d.mjd, float)
    if d.frame is None:
        times = onp.unique(mjd)
    else:  # mean time of each frame
        _, index = onp.unique(onp.asarray(d.frame), return_inverse=True)
        times = onp.bincount(index, weights=mjd) / onp.bincount(index)
    t_ref = float(onp.mean(times))
    t_ss = float(onp.sum((times - t_ref) ** 2))
    res = float(fringe_scales(d)[0])
    with run_in("float64"):
        d64 = cast_tree(d, "float64")
        surface = _Surface(d64)
        dm = None if not diam else np.asarray(float(diam))
        problem = _Problem(d64, dm, t_ref, surface, amp_range)
        zero = np.zeros(5)
        rescalable = d.gains is None and d.phase_offsets is None

        def loglike_at(z):
            """``ln L`` at ``z`` with the amplitude profiled: the
            likelihood's normalization, the profiled χ²."""
            model = _model(z, t_ref, dm)
            plain = np.sum(whitened_residuals(model, d64) ** 2)
            profiled = np.sum(problem.chi2(z))
            return float(model_loglike(model, d64) + 0.5 * (plain - profiled))

        loglike_null = loglike_at(zero.at[4].set(-np.inf))
        peaks = []
        window = _REFINE_WINDOW * res
        n = onp.array([b[3] for b in surface.blocks], float)
        cat = positions.peaks[i]
        for dra, ddec, flux in zip(cat.dra, cat.ddec, cat.flux):
            z0 = onp.array(
                [
                    dra,
                    ddec,
                    0.0,
                    0.0,
                    onp.log(min(max(float(flux), _F_MIN), 1.0)),
                ]
            )
            lower = onp.array(
                [dra - window, ddec - window, -onp.inf, -onp.inf]
            )
            upper = onp.array([dra + window, ddec + window, onp.inf, onp.inf])
            lo = onp.append(lower, onp.log(_F_MIN))
            hi = onp.append(upper, 0.0)
            state = _gauss_newton.solve(
                problem, z0, lower=lo, upper=hi, dtol=1e-12, max_steps=200
            )
            zj = state.z
            # a peak held on the window or flux bound is not a stationary
            # point of the unconstrained fit: its Hessian is no covariance
            held = _gauss_newton._held(zj, state.grad, lo, hi)
            if (
                int(state.stop) != _gauss_newton.CONVERGED
                or not onp.isfinite(float(state.value))
                or bool(onp.any(held))
            ):
                continue
            hess = onp.asarray(_hessians(problem, zj), float)
            # negative curvature is a saddle, not a peak (a flat direction,
            # such as the velocity of one frame, is allowed)
            lam = onp.linalg.eigvalsh(onp.sum(hess, 0))
            if not onp.all(onp.isfinite(hess)) or lam.min() < -(
                onp.finfo(float).eps * lam.max()
            ):
                continue
            peak = NightPeak(
                onp.asarray(zj, float),
                _inverse(onp.sum(hess, 0)),
                loglike_at(zj),
                n,
                onp.asarray(problem.chi2(zj), float),
                hess,
                rescalable,
            )
            # one peak if the means differ by under one standard deviation
            # (summed covariances); the higher likelihood stands for both
            same = [
                k
                for k, q in enumerate(peaks)
                if (peak.mean - q.mean)
                @ onp.linalg.solve(peak.cov + q.cov, peak.mean - q.mean)
                < 1.0
            ]
            if not same:
                peaks.append(peak)
            elif peak.loglike > peaks[same[0]].loglike:
                peaks[same[0]] = peak
    peaks.sort(key=lambda p: -p.loglike)
    return NightSummary(
        t_ref,
        t_ss,
        len(times),
        tuple(b[0] for b in surface.blocks),
        tuple(peaks),
        loglike_null,
    )


def night_summaries(data, positions, *, amp_range=None):
    """Each night's per-peak linear-motion fit on the visibilities.

    The peaks start from each night's catalogue
    ([`EpochPositions.peaks`][virgil.epochs.EpochPositions]) at zero velocity
    and are refined, in float64, over the whole night with the primary of
    ``positions`` (no half-night split, so closure offsets and spanning gain
    modes are kept). A refinement that does not converge is dropped; peaks
    that converge to one point are merged.

    Parameters
    ----------
    data : Epochs or sequence of OIData
        The nights, in the order of ``positions``. Each needs sample times
        (``mjd``).
    positions : EpochPositions
        The nights' peak catalogues and the primary diameter.
    amp_range : tuple of float, optional
        Profile one visibility amplitude per night in this range, as the
        orbit fit does (default None).

    Returns
    -------
    tuple of NightSummary
    """
    sets = list(data.data) if isinstance(data, Epochs) else list(data)
    if len(sets) != len(positions.names):
        raise ValueError(
            f"{len(sets)} datasets but positions for {len(positions.names)}: "
            "they must be the same nights."
        )
    if any(d.mjd is None for d in sets):
        raise ValueError("Night summaries need the sample times (data.mjd).")
    return tuple(
        _summarize(d, positions, i, positions.primary_diam, amp_range)
        for i, d in enumerate(sets)
    )


def keep_peaks(summaries, log_outlier, delta=DELTA):
    """The peaks the position-level likelihood needs.

    A peak is kept iff ``(1 - ε_lo) L_j ≥ δ ε_lo e^{b_k}``: an orbit through it
    contributes at best ``(1 - ε) L_j`` to the night's mixture factor
    ``(1 - ε) e^{ℓ_k} + ε e^{b_k}``, with no Occam factor. ``ε_lo`` is the
    smallest Gauss–Chebyshev node of the ε integral, ``sin²(π / (4 n))`` with
    ``n = ⌈(K + 1) / 2⌉`` for ``K`` nights.

    Parameters
    ----------
    summaries : sequence of NightSummary
        From [`night_summaries`][virgil.night_summaries.night_summaries].
    log_outlier : array-like
        ``b_k``, each night's log outlier density, on the likelihood's
        normalization (``NightPeak.loglike``).
    delta : float, optional
        Reporting precision (default ``DELTA``).

    Returns
    -------
    tuple of NightSummary
    """
    log_outlier = onp.broadcast_to(
        onp.asarray(log_outlier, float), (len(summaries),)
    )
    n = int(onp.ceil((len(summaries) + 1) / 2))
    eps = onp.sin(onp.pi / (4 * n)) ** 2
    out = []
    for s, b in zip(summaries, log_outlier):
        keep = tuple(
            p
            for p in s.peaks
            if onp.log1p(-eps) + p.loglike >= onp.log(delta * eps) + b
        )
        out.append(dataclasses.replace(s, peaks=keep))
    return tuple(out)
