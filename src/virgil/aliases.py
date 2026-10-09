"""Orbits whose epochs undersample the period: alias bands and their evidences.

When epochs are far apart compared with the period, the number of orbital
cycles between them is ambiguous, and the likelihood has one narrow ridge
for each candidate count. [`fit_orbit_aliases`][virgil.aliases.fit_orbit_aliases]
labels the ridges by ``N = round(T/P)``, with ``T`` the span from the first
to the last epoch, fits each *band* of periods directly to the closure phases
(or visibilities) of every epoch, and weighs the bands by their marginal
likelihoods. Positions are used only to start the fits and as a
diagnostic: a position is a lossy summary of an epoch.

The likelihood is hierarchical across epochs: each epoch's observable
blocks have an error scale with a Jeffreys (log-uniform) prior, integrated
out ([`marginal_loglike`][virgil.epochs.marginal_loglike]). The evidence of
each band is the sum over its distinct modes of a Laplace approximation,
checked by importance sampling from the Laplace proposal.

See ``fit_orbit_aliases`` for the entry point.
"""

import dataclasses
import functools
import hashlib
import json
import math
import pathlib
import pickle

import jax
import jax.numpy as np
import numpy as onp
from scipy import optimize

from ._precision import cast_tree, run_in
from .epochs import _Surface
from .models import OrbitalBinary
from .orbits import KeplerOrbit, period_grid, starting_orbits

__all__ = [
    "AliasBand",
    "AliasBandResult",
    "AliasResult",
    "alias_bands",
    "fit_orbit_aliases",
    "position_profile",
    "spectral_window",
]

# Sampled coordinates (flat under the Jeffreys prior): log P, the time of
# periastron as a fraction of the period, e, cos i, omega, Omega (degrees),
# log a, log flux.
_NAMES = (
    "log_period",
    "tau",
    "ecc",
    "cos_inc",
    "omega",
    "Omega",
    "log_a_mas",
    "log_flux",
)
_PERIODIC = {
    1: 1.0,
    4: 360.0,
    5: 180.0,
}  # periodic coordinates and their periods
_ELEMENTS = ("period", "dt_peri", "ecc", "inc", "omega", "Omega", "a_mas")


# ---------------------------------------------------------------------------
# Bands and diagnostics
# ---------------------------------------------------------------------------


def spectral_window(times, frequencies):
    """The spectral window ``|mean exp(2πi f t)|²`` of the epochs' times.

    It is 1 at ``f = 0`` and returns to near 1 at the aliases of the
    sampling: a frequency offset at which the window is high cannot be told
    from zero offset by the timing alone.
    """
    t = onp.asarray(times, float)
    f = onp.atleast_1d(onp.asarray(frequencies, float))
    phase = onp.exp(2j * onp.pi * f[:, None] * (t - t.mean())[None, :])
    return onp.abs(phase.mean(axis=1)) ** 2


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


def position_profile(positions, bands, k=9.0, n_best=3, eccs=None):
    """A cheap diagnostic: the best position-level χ² in each band.

    For each band, the best Thiele–Innes fits
    ([`starting_orbits`][virgil.orbits.starting_orbits]) of ``positions``
    on a coherent period grid. Positions are lossy: use this to see
    which bands are viable and to seed the fits, not to weigh them.

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
        found = starting_orbits(positions, periods, eccs=eccs, n_best=n_best)
        rows.append(
            dict(
                n=band.n,
                chi2=found[0][1],
                period=float(found[0][0].period),
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


class _Problem:
    """The scale-marginalized log likelihood of all epochs and the priors."""

    def __init__(
        self, data, t_ref, p_range, a_range, flux_range, ecc_max, s_max
    ):
        self.data, self.t_ref, self.s_max = list(data), float(t_ref), s_max
        self.surfaces = [_Surface(d, s_max) for d in self.data]
        # Proper priors: log P log-uniform over the whole p_range, tau
        # uniform over one cycle, e uniform, cos i uniform, omega and Omega
        # uniform with the (Omega + 180, omega + 180) mirror folded away
        # (Omega in [0, 180)), log a and log flux uniform.
        self.log_prior = float(
            -onp.log(onp.log(p_range[1] / p_range[0]))
            - onp.log(ecc_max)
            - onp.log(2.0)
            - onp.log(360.0)
            - onp.log(180.0)
            - onp.log(onp.log(a_range[1] / a_range[0]))
            - onp.log(onp.log(flux_range[1] / flux_range[0]))
        )
        self.a_range, self.flux_range, self.ecc_max = (
            a_range,
            flux_range,
            ecc_max,
        )
        self.value_and_grad = jax.jit(jax.value_and_grad(self.loglike))
        self.hessian = jax.jit(jax.hessian(self.loglike))
        self.batch = jax.jit(
            lambda xs: jax.lax.map(self.loglike, xs, batch_size=64)
        )

    def set_band(self, band):
        self.lower = onp.array(
            [
                onp.log(band.p_lo),
                -onp.inf,
                0.0,
                -1.0,
                -onp.inf,
                -onp.inf,
                onp.log(self.a_range[0]),
                onp.log(self.flux_range[0]),
            ]
        )
        self.upper = onp.array(
            [
                onp.log(band.p_hi),
                onp.inf,
                self.ecc_max,
                1.0,
                onp.inf,
                onp.inf,
                onp.log(self.a_range[1]),
                onp.log(self.flux_range[1]),
            ]
        )

    def loglike(self, x):
        orbit, flux = _elements(x, self.t_ref)
        model = OrbitalBinary(orbit, flux)
        return sum(
            s.score(s.chi2(model, d), 1.0, d)
            for s, d in zip(self.surfaces, self.data)
        )

    def per_epoch(self, x):
        """Raw χ²/ν per epoch and block, and the profile scales ŝ."""
        orbit, flux = _elements(np.asarray(x), self.t_ref)
        model = OrbitalBinary(orbit, flux)
        rows = []
        for s, d in zip(self.surfaces, self.data):
            chi2 = onp.asarray(s.chi2(model, d))
            rows.append(
                dict(
                    blocks=list(s.names),
                    chi2_red=(chi2 / s.nu).tolist(),
                    scale=s.scales(chi2).tolist(),
                )
            )
        return rows

    def inside(self, xs):
        """Whether each row of ``xs`` is inside the hard bounds of the prior
        (the angles are periodic, with no bound)."""
        return onp.all((xs >= self.lower) & (xs <= self.upper), axis=-1)


def _fold(x):
    """Fold the (Omega + 180, omega + 180) mirror: Omega in [0, 180), omega
    in [0, 360)."""
    x = onp.array(x, float)
    flip = onp.floor(x[5] / 180.0)
    x[5] -= 180.0 * flip
    x[4] += 180.0 * flip
    x[4] = onp.mod(x[4], 360.0)
    return x


def _start_vector(orbit, flux, t_ref, band, problem):
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
    )
    x = _fold(x)
    eps = 1e-3
    lo, hi = problem.lower, problem.upper
    return onp.clip(
        x,
        onp.where(onp.isfinite(lo), lo + eps, lo),
        onp.where(onp.isfinite(hi), hi - eps, hi),
    )


def _senses(x):
    """The seed and its other sense of motion (i -> 180 - i)."""
    other = onp.array(x)
    other[3] = -other[3]
    return [x, other]


def _optimise(problem, x0, max_steps):
    """Maximize the marginal likelihood from ``x0`` (L-BFGS-B, in the
    sampled coordinates)."""

    def fun(x):
        value, grad = problem.value_and_grad(np.asarray(x))
        return -float(value), -onp.asarray(grad, float)

    bounds = list(
        zip(
            [None if not onp.isfinite(v) else float(v) for v in problem.lower],
            [None if not onp.isfinite(v) else float(v) for v in problem.upper],
        )
    )
    res = optimize.minimize(
        fun,
        x0,
        jac=True,
        method="L-BFGS-B",
        bounds=bounds,
        options=dict(maxiter=max_steps),
    )
    return _fold(res.x), -float(res.fun)


# ---------------------------------------------------------------------------
# Evidence: Laplace, checked by importance sampling
# ---------------------------------------------------------------------------


def _logsumexp(a):
    a = onp.asarray(a, float)
    if a.size == 0 or not onp.any(onp.isfinite(a)):
        return -onp.inf
    m = onp.max(a[onp.isfinite(a)])
    return float(m + onp.log(onp.sum(onp.exp(a - m))))


def _mode_evidence(problem, x, value, n_is, rng, df=5.0, bound_sigma=3.0):
    """Laplace and importance-sampling log evidences of one mode at ``x``.

    The importance proposal is a multivariate Student-t (``df``) with the
    Laplace covariance. The hard prior bounds truncate the target; angles
    are periodic and are kept within half a turn of the mode. A mode whose
    Hessian is not positive definite, or is singular, has no proposal: both evidences are
    ``-inf`` and it is dropped.
    """
    h = -onp.asarray(problem.hessian(np.asarray(x)), float)
    h = 0.5 * (h + h.T)
    finite = bool(onp.all(onp.isfinite(h)))
    eig = onp.linalg.eigvalsh(h) if finite else onp.array([-onp.inf])
    out = dict(
        x=x.tolist(),
        loglike=value,
        log_z_laplace=-onp.inf,
        log_z_is=-onp.inf,
        ess=0.0,
        flags=[],
    )
    if not (finite and eig.min() > 0):
        out["flags"].append("hessian-not-positive-definite")
        return out, None
    if eig.min() <= 1e-13 * eig.max():
        # Positive, but too ill-conditioned to invert: a flat direction.
        out["flags"].append("hessian-singular")
        return out, None
    try:
        cov = onp.linalg.inv(h)
        chol = onp.linalg.cholesky(cov)
        l_inv = onp.linalg.inv(chol)
    except onp.linalg.LinAlgError:
        out["flags"].append("hessian-singular")
        return out, None
    sd = onp.sqrt(onp.diag(cov))
    d = len(x)
    logdet = float(onp.linalg.slogdet(cov)[1])
    out["log_z_laplace"] = float(
        value
        + problem.log_prior
        + 0.5 * d * onp.log(2 * onp.pi)
        + 0.5 * logdet
    )
    distance = onp.minimum(x - problem.lower, problem.upper - x) / sd
    if onp.any(distance < bound_sigma):
        out["flags"].append("near-prior-bound")
    out["sigma"] = sd.tolist()
    # Newton step in units of sigma, over the coordinates not on a bound:
    # large means L-BFGS stopped short of a stationary point.
    grad = onp.asarray(problem.value_and_grad(np.asarray(x))[1], float)
    free = (x - problem.lower > 1e-6) & (problem.upper - x > 1e-6)
    if onp.any(onp.abs(grad[free]) * sd[free] > 0.1):
        out["flags"].append("not-converged")
    # Student-t proposal.
    z = rng.standard_normal((n_is, d))
    g = rng.chisquare(df, n_is) / df
    xs = x + (z @ chol.T) / onp.sqrt(g)[:, None]
    mahal = onp.sum(z**2, axis=1) / g
    log_q = (
        -0.5 * (df + d) * onp.log1p(mahal / df)
        - 0.5 * logdet
        + math.lgamma(0.5 * (df + d))
        - math.lgamma(0.5 * df)
        - 0.5 * d * onp.log(df * onp.pi)
    )
    # Angles live within half a turn of the mode.
    ok = problem.inside(xs)
    for k, width in _PERIODIC.items():
        ok &= onp.abs(xs[:, k] - x[k]) < width / 2.0
    values = onp.full(n_is, -onp.inf)
    if ok.any():
        values[ok] = onp.asarray(problem.batch(np.asarray(xs[ok])), float)
    log_t = values + problem.log_prior  # the target at each draw
    log_w = log_t - log_q
    log_w[~onp.isfinite(log_w)] = -onp.inf
    out["log_z_is"] = _logsumexp(log_w) - onp.log(n_is)
    w = (
        onp.exp(log_w - onp.max(log_w[onp.isfinite(log_w)]))
        if onp.any(onp.isfinite(log_w))
        else onp.zeros(n_is)
    )
    out["ess"] = float(w.sum() ** 2 / max((w**2).sum(), 1e-300))
    proposal = (x, l_inv, logdet, df)
    return out, (xs, log_t, log_w, proposal)


def _log_t(xs, proposal):
    """Log density of a mode's Student-t proposal at ``xs``."""
    x, l_inv, logdet, df = proposal
    d = len(x)
    mahal = onp.sum(((xs - x) @ l_inv.T) ** 2, axis=1)
    return (
        -0.5 * (df + d) * onp.log1p(mahal / df)
        - 0.5 * logdet
        + math.lgamma(0.5 * (df + d))
        - math.lgamma(0.5 * df)
        - 0.5 * d * onp.log(df * onp.pi)
    )


def _mixture_weights(draws):
    """Log weights of the pooled draws of a band's modes under their
    deterministic mixture proposal, ``p / mean_k q_k``: the modes' tails
    overlap, and per-mode weights would count the overlap twice."""
    xs = onp.concatenate([d[0] for d in draws])
    log_t = onp.concatenate([d[1] for d in draws])
    log_q = onp.stack([_log_t(xs, d[3]) for d in draws])
    log_mix = onp.array([_logsumexp(c) for c in log_q.T]) - onp.log(len(draws))
    log_w = log_t - log_mix
    log_w[~onp.isfinite(log_w)] = -onp.inf
    return xs, log_w


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


def _clean(v):
    """``v`` as strict JSON: arrays to lists, inf and nan to None."""
    if isinstance(v, dict):
        return {k: _clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_clean(x) for x in v]
    if hasattr(v, "tolist"):
        return _clean(v.tolist())
    if isinstance(v, float) and not math.isfinite(v):
        return None  # JSON has no inf or nan
    return v


def _fingerprint(obj):
    """A hash of every array in ``obj`` (data or positions), to tell a checkpoint's data from new data."""
    if obj is None:
        return None
    h = hashlib.sha256()
    for leaf in jax.tree_util.tree_leaves(obj):
        try:
            a = onp.asarray(leaf)
        except Exception:  # noqa: BLE001 - not an array: hash its repr
            h.update(repr(leaf).encode())
            continue
        h.update(str(a.shape).encode() + str(a.dtype).encode() + a.tobytes())
    return h.hexdigest()


def _band_files(directory, band):
    d = pathlib.Path(directory)
    return d / f"band_{band.n}.pkl", d / f"band_{band.n}.json"


def _load_band(directory, band, key):
    """A band's checkpoint, or None if there is none or it is for other settings."""
    if directory is None:
        return None
    pkl, _ = _band_files(directory, band)
    try:
        with open(pkl, "rb") as f:
            saved_key, done = pickle.load(f)
    except Exception:  # noqa: BLE001 - missing or truncated: refit
        return None
    return done if saved_key == _clean(key) else None


def _save_band(directory, band, key, done):
    """Write a band's checkpoint atomically (pickle to reload, JSON to read)."""
    if directory is None:
        return
    pkl, js = _band_files(directory, band)
    pkl.parent.mkdir(parents=True, exist_ok=True)
    tmp = pkl.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        pickle.dump((_clean(key), done), f)
    tmp.replace(pkl)
    with open(js, "w") as f:
        json.dump(
            _clean(dataclasses.asdict(done[0])), f, indent=1, default=str
        )


@dataclasses.dataclass
class AliasBandResult:
    """The fit of one alias band (see [`AliasResult`][virgil.aliases.AliasResult])."""

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
    error: str = ""  # the exception, if the band failed (flag ``band-failed``)


@dataclasses.dataclass
class AliasResult:
    """Result of [`fit_orbit_aliases`][virgil.aliases.fit_orbit_aliases].

    Attributes
    ----------
    t_ref : float
        Reference time (MJD) of ``dt_peri``.
    times : numpy.ndarray
        The epochs' mean times (MJD).
    span : float
        First-to-last epoch span T (days).
    bands : list of AliasBandResult
        One per alias band, in order of decreasing ``log_z``.
    samples : dict
        ``{N: {name: array}}`` posterior draws (importance resampling from
        the modes' Laplace proposals) in the winning band and in every band
        with ``p > p_min_samples``. Keys: ``period``, ``dt_peri``, ``ecc``,
        ``inc``, ``omega``, ``Omega``, ``a_mas``, ``flux``.
    positions : list of dict or None
        The position-level diagnostic per band, if positions were given.
    """

    t_ref: float
    times: onp.ndarray
    span: float
    bands: list
    samples: dict
    positions: list = None

    @property
    def best(self):
        """The band with the highest evidence."""
        return self.bands[0]

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
            )
            for b in self.bands
        ]

    def to_json(self, path, n_samples=0):
        """Write the band table and each band's modes to ``path``."""
        payload = dict(
            t_ref=self.t_ref,
            times=self.times.tolist(),
            span=self.span,
            bands=self.table(),
            band_details=[dataclasses.asdict(b) for b in self.bands],
            samples={
                str(n): {k: v[:n_samples].tolist() for k, v in s.items()}
                for n, s in self.samples.items()
            }
            if n_samples
            else {},
        )

        with open(path, "w") as f:
            json.dump(_clean(payload), f, indent=1, default=str)


def _physical(x, t_ref):
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
    )


def _distinct(modes, times, t_ref, min_distance_mas):
    """Keep the best of the modes whose tracks at ``times`` agree within
    ``min_distance_mas``."""
    modes = sorted(modes, key=lambda m: -m[1])
    kept, tracks = [], []
    for x, value in modes:
        orbit, _ = _elements(np.asarray(x), t_ref)
        dra, ddec, _ = orbit.relative(np.asarray(times))
        track = onp.stack([onp.asarray(dra), onp.asarray(ddec)], -1)
        if all(
            onp.max(onp.hypot(*(track - t).T)) > min_distance_mas
            for t in tracks
        ):
            kept.append((x, value))
            tracks.append(track)
    return kept


# ---------------------------------------------------------------------------
# The entry point
# ---------------------------------------------------------------------------


def _float64(fn):
    """Run a fitting entry point in float64 (see ``virgil._precision``): the
    Hessians and Cholesky factors of narrow ridges need it."""

    @functools.wraps(fn)
    def wrapped(data, *args, **kwargs):
        with run_in("float64"):
            return fn(cast_tree(list(data), "float64"), *args, **kwargs)

    return wrapped


@_float64
def fit_orbit_aliases(
    data,
    p_range,
    *,
    positions=None,
    times=None,
    t_ref=None,
    a_range=(1.0, 30.0),
    flux_range=(0.01, 1.0),
    ecc_max=0.9,
    s_max=None,
    n_candidates=100,
    n_refine=6,
    n_random=None,
    k=9.0,
    eccs=None,
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
):
    """Weigh the period aliases of an undersampled orbit with the observables.

    Parameters
    ----------
    data : sequence of OIData
        One dataset per epoch, with per-exposure times. The scene is a
        primary and a point companion on a Keplerian orbit
        ([`OrbitalBinary`][virgil.models.OrbitalBinary]).
    p_range : (float, float)
        The shortest and longest period (days): the log-uniform prior on P.
    positions : PositionData, optional
        Per-epoch positions (e.g. from
        [`epoch_positions`][virgil.epochs.epoch_positions]). They seed each
        band's fits through Thiele–Innes grids
        ([`position_profile`][virgil.aliases.position_profile]); without them
        the candidates are random orbits.
    times : array-like, optional
        Epoch times (MJD) used for the bands and distinctness; by default
        each dataset's mean time.
    t_ref : float, optional
        Reference MJD of ``dt_peri``; by default the mean epoch.
    a_range, flux_range : (float, float), optional
        Log-uniform prior ranges of the semimajor axis (mas) and the
        companion/primary flux.
    ecc_max : float, optional
        Eccentricity is uniform on ``[0, ecc_max]``.
    s_max : float, optional
        Bound on the error scales, as in
        [`marginal_loglike`][virgil.epochs.marginal_loglike].
    n_candidates : int, optional
        Position-level candidate orbits per band (each in both senses of
        motion and at several fluxes), ranked by the observables.
    n_refine : int, optional
        The best distinct candidates per band that are refined into modes.
    n_random : int, optional
        Extra random candidates per band; by default 1000 without
        ``positions`` and none with them.
    min_distance_mas : float, optional
        Fitted modes whose tracks at the epochs differ by less are one mode.
    n_is : int, optional
        Importance draws per mode.
    n_samples : int, optional
        Posterior draws per reported band.
    p_min_samples : float, optional
        Bands with at least this probability get posterior samples (and the
        winner always does).
    flag_nats : float, optional
        A band gets the flag ``laplace-is-differ`` when its Laplace and
        importance-sampling log evidences differ by more than this.

    checkpoint_dir : path-like, optional
        Each finished band is written here (``band_<N>.pkl`` for reuse and
        ``band_<N>.json`` to read) before the next starts. A rerun with the
        same data, positions, ranges and settings loads the finished bands instead of
        refitting them; a band that raised is not saved, so it is retried.
    on_band : callable, optional
        Called with each band's
        [`AliasBandResult`][virgil.aliases.AliasBandResult] as soon as it is
        fitted or loaded (``p`` is not yet set: it needs all bands).

    Returns
    -------
    AliasResult

    Notes
    -----
    A band that raises is kept as a ``band-failed`` row (``error`` holds the
    exception) with ``log_z = -inf``; the other bands are unaffected.

    The band evidence ``log_z`` is the sum, over the band's distinct fitted
    modes (both senses of motion; the (Ω+180, ω+180) mirror is folded
    away by ``Ω < 180``), of the Laplace approximation to the marginal
    likelihood of the scale-marginalized closure-phase (and visibility)
    likelihood, times the proper prior. A mode on a prior bound (a period
    edge between bands) uses its importance-sampling value instead, which
    truncates the target at the bound. It misses modes the multistart does
    not find. ``log_z_is`` is the importance-sampling estimate from the
    deterministic mixture of the modes' Student-t proposals (so overlapping
    tails are not counted twice), and the posterior samples are resampled
    with the same weights, with the mirror folded.

    ``flags`` lists why a band should not be trusted:
    ``near-prior-bound``, ``not-converged`` (the optimizer stopped more
    than 0.1σ from a stationary point), ``hessian-not-positive-definite``, ``hessian-singular``,
    ``band-failed`` (an exception in the band; the others are unaffected),
    ``laplace-is-differ``, ``low-ess`` (best mode's ESS below ``ess_min``)
    and ``no-valid-mode``. ``chi2_red`` is the raw χ²/ν on the quoted errors
    at the best mode, before any error scale.

    Without ``s_max`` the Jeffreys prior on each error scale is improper, so
    the absolute ``log_z`` is defined only up to a constant shared by all
    bands (``p`` and differences are meaningful).
    """
    data = list(data)
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
    bands = alias_bands(mean_times, p_range)
    if not bands:
        raise ValueError("No alias band lies within p_range.")
    profile = (
        position_profile(positions, bands, k=k, n_best=n_candidates, eccs=eccs)
        if positions is not None
        else None
    )

    n_extra = (
        (1000 if positions is None else 0) if n_random is None else n_random
    )
    flux_scan = onp.geomspace(flux_range[0], flux_range[1], 6)[1:-1]
    problem = _Problem(
        data, t_ref, p_range, a_range, flux_range, ecc_max, s_max
    )
    key = dict(
        edges=[[bd.p_lo, bd.p_hi] for bd in bands],
        data=_fingerprint(data),
        positions=_fingerprint(positions),
        times=mean_times.tolist(),
        s_max=s_max,
        k=k,
        eccs=None if eccs is None else list(onp.asarray(eccs, float)),
        max_steps=max_steps,
        min_distance_mas=min_distance_mas,
        flag_nats=flag_nats,
        ess_min=ess_min,
        ecc_max=ecc_max,
        a_range=list(a_range),
        flux_range=list(flux_range),
        n_candidates=n_candidates,
        n_refine=n_refine,
        n_random=n_extra,
        n_is=n_is,
        seed=seed,
        t_ref=t_ref,
    )

    def fit_band(index, band):
        # Its own stream: a band does not depend on which bands were fitted or loaded before it.
        rng = onp.random.default_rng([seed, band.n])
        problem.set_band(band)
        pool = []
        if profile is not None:
            for orbit, _ in profile[index]["orbits"]:
                for flux in flux_scan:
                    pool.extend(
                        _senses(
                            _start_vector(orbit, flux, t_ref, band, problem)
                        )
                    )
        for _ in range(n_extra):
            pool.append(
                _fold(
                    [
                        rng.uniform(onp.log(band.p_lo), onp.log(band.p_hi)),
                        rng.uniform(-0.5, 0.5),
                        rng.uniform(0.0, 0.6 * ecc_max),
                        rng.uniform(-0.95, 0.95),
                        rng.uniform(0, 360),
                        rng.uniform(0, 180),
                        rng.uniform(onp.log(a_range[0]), onp.log(a_range[1])),
                        rng.uniform(
                            onp.log(flux_range[0]), onp.log(flux_range[1])
                        ),
                    ]
                )
            )
        # Rank the candidates by the observables, then refine the best few.
        pool = onp.asarray(pool)
        score = onp.array(problem.batch(np.asarray(pool)), float)
        score[~onp.isfinite(score)] = -onp.inf
        keep = onp.argsort(-score)[: 4 * n_refine]
        ranked = _distinct(
            [(pool[k_], score[k_]) for k_ in keep],
            mean_times,
            t_ref,
            min_distance_mas,
        )
        seeds = [x_ for x_, _ in ranked[:n_refine]]
        fitted = []
        for x0 in seeds:
            try:
                x, value = _optimise(problem, x0, max_steps)
            except Exception:  # noqa: BLE001 - a failed start is just dropped
                continue
            if onp.isfinite(value):
                fitted.append((x, value))
        modes_xy = (
            _distinct(fitted, mean_times, t_ref, min_distance_mas)
            if fitted
            else []
        )
        modes, draws = [], []
        for x, value in modes_xy:
            record, draw = _mode_evidence(problem, x, value, n_is, rng)
            modes.append(record)
            if draw is not None:
                draws.append(draw)
        # Laplace per mode; where a mode sits on a prior bound (a period
        # edge between bands, say) the Gaussian is cut off, so use that
        # mode's importance-sampling value, which truncates the target.
        for m in modes:
            edge = "near-prior-bound" in m["flags"] and m["ess"] >= ess_min
            m["log_z"] = m["log_z_is"] if edge else m["log_z_laplace"]
        z_lap = _logsumexp([m["log_z"] for m in modes])
        if draws:
            xs_pool, lw_pool = _mixture_weights(draws)
            z_is = _logsumexp(lw_pool) - onp.log(len(xs_pool))
        else:
            xs_pool, lw_pool, z_is = None, None, -onp.inf
        live = [m for m in modes if onp.isfinite(m["log_z"])]
        best_mode = max(live, key=lambda m: m["log_z"]) if live else None
        best = {}
        epochs, chi2_red = [], float("nan")
        if best_mode is not None:
            best = {
                k_: float(v[0])
                for k_, v in _physical(
                    onp.array(best_mode["x"]), t_ref
                ).items()
            }
            epochs = problem.per_epoch(best_mode["x"])
            nu = onp.concatenate([s.nu for s in problem.surfaces])
            chi2 = onp.concatenate([onp.array(e["chi2_red"]) for e in epochs])
            chi2_red = float(onp.sum(chi2 * nu) / onp.sum(nu))
        flags = sorted({f for m in modes for f in m["flags"]})
        if (
            onp.isfinite(z_lap)
            and onp.isfinite(z_is)
            and abs(z_lap - z_is) > flag_nats
        ):
            flags.append("laplace-is-differ")
        if best_mode is not None and best_mode["ess"] < ess_min:
            flags.append("low-ess")
        if best_mode is None:
            flags.append("no-valid-mode")
        result = AliasBandResult(
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
        )
        return result, (xs_pool, lw_pool)

    results, store = [], {}
    for index, band in enumerate(bands):
        done = _load_band(checkpoint_dir, band, key)
        if done is None:
            try:
                done = fit_band(index, band)
            except Exception as err:  # noqa: BLE001 - one band must not lose the rest
                done = (
                    AliasBandResult(
                        n=band.n,
                        p_lo=band.p_lo,
                        p_hi=band.p_hi,
                        modes=[],
                        log_z=-onp.inf,
                        log_z_is=-onp.inf,
                        ess=0.0,
                        flagged=True,
                        flags=["band-failed"],
                        error=f"{type(err).__name__}: {err}",
                    ),
                    (None, None),
                )
            else:
                _save_band(checkpoint_dir, band, key, done)
        result, pool = done
        results.append(result)
        store[band.n] = pool
        if on_band is not None:
            on_band(result)

    log_z = onp.array([b.log_z for b in results])
    finite = onp.isfinite(log_z)
    weights = (
        onp.where(finite, onp.exp(log_z - onp.max(log_z[finite])), 0.0)
        if finite.any()
        else onp.zeros_like(log_z)
    )
    for b, w in zip(results, weights / max(weights.sum(), 1e-300)):
        b.p = float(w)
    order = onp.argsort(-onp.where(finite, log_z, -onp.inf))
    results = [results[i] for i in order]

    samples = {}
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
        samples[b.n] = _physical(xs[pick], t_ref)
    return AliasResult(
        t_ref,
        mean_times,
        float(onp.ptp(mean_times)),
        results,
        samples,
        profile,
    )
