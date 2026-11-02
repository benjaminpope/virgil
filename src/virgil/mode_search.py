"""Mode search, defensive importance sampling and HMC resample-move.

!!! warning "Provisional"
    May change in any 0.4.x release without a deprecation cycle.

For a target that is a needle in its prior (a posterior mass of order
``e⁻⁴⁰`` of the prior), tempering from the prior fails. Here a search finds
the modes of the target; [`DefensiveMixture`][virgil.mode_search.DefensiveMixture]
puts a Student-t on each (with the images of exact symmetries of the target)
beside a share of the prior; [`importance_evidence`][virgil.mode_search.importance_evidence]
estimates ``ln Z`` with an error from the effective sample size and flags a
search that missed the dominant mode; [`resample_move`][virgil.mode_search.resample_move]
turns the weighted draws into posterior draws. The target is a function
``log_target(x, theta)`` of a flat vector ``x`` and a pytree ``theta`` of traced
parameters, so one compilation serves every cell of a system.

Source: Hesterberg (1995), Hoffman & Gelman (2014), BlackJAX
(Cabezas et al. 2024), Gilks & Berzuini (2001), Neal (2011). The sentinel and
the symmetric distinct-mode rule are not from these: they are provisional.
"""

import dataclasses
import functools
import math

import jax
import jax.numpy as np
import numpy as onp
from blackjax.mcmc import hmc
from blackjax.optimizers import dual_averaging

from .night_summaries import DELTA

__all__ = [
    "DefensiveMixture",
    "Evidence",
    "Mode",
    "Move",
    "distinct_modes",
    "importance_evidence",
    "laplace_log_evidence",
    "logmeanexp",
    "resample_move",
]

__provisional__ = ("*",)

MAHAL = 1.0  # distinct-mode radius, squared Mahalanobis under both covariances
ALPHA = 0.1  # prior share of q (Hesterberg 1995): bounds w <= L_max / ALPHA
DF = 5.0  # t degrees of freedom, as aliases._mode_evidence
N_IS = 4000  # IS draws per cell; sd(ln Z) ~ 0.012 on the toys
ESS_TARGET = 400  # sd(ln Z) ~ ESS^-1/2 <= 0.05
ESS_FAIL = 10  # below this the estimate is a few draws: the cell fails
W = math.log(1.0 / DELTA)  # sentinel margin above the floor (nats)
M_MIN = 2  # sentinel: the smallest minimal set (a pair of nights)
M_PAD = 128  # components padded to at least this (8 modes with 15 images):
# the shapes, hence the compiled kernels, are those of every cell
N_POST = 1000  # resample-move draws per cell
N_HMC = 5  # HMC moves per draw
N_WARM = 150  # dual-averaging warm-up transitions
N_PILOT = 100  # particles in the warm-up pilot
ACCEPT = 0.8  # dual-averaging target acceptance (Hoffman & Gelman 2014)
L_MAX = 64  # trajectory cap
JITTER = 0.2  # step-size jitter, ± fraction (Neal 2011)


@dataclasses.dataclass(frozen=True)
class Mode:
    """A mode of the label-summed target.

    ``cov`` is the inverse of minus the Hessian of the search objective at
    ``x``; ``log_target`` is the log of the objective the search maximized, at ``x``
    (the log target for a target whose Laplace ``ln Z`` needs no correction).
    """

    x: object
    log_target: float
    cov: object
    n_inliers: float


@dataclasses.dataclass(frozen=True)
class Evidence:
    """Importance-sampling estimate of ``ln Z`` from a defensive mixture.

    ``log_z_error`` is the delta-method ``√(1/ESS - 1/n)``, a first-order
    lower bound (it misses heavy tails and mass the proposal missed; ``ln Ẑ``
    also carries a bias of ``-1/(2 ESS)``). ``at_floor``
    is the sentinel: the draws sit at the all-outlier floor although a mode
    with at least ``m_min`` inliers has a Laplace ``ln Z`` ``W`` nats above it; ``log_z`` must
    then not be reported. ``undersampled`` is ``ess < ESS_TARGET``
    (information only). ``n_inliers`` is the weighted mean of ``inliers_fn``.
    """

    log_z: float
    log_z_error: float
    ess: float
    x: object
    log_w: object
    at_floor: bool
    undersampled: bool
    n_inliers: float
    n_nonfinite: int


@dataclasses.dataclass(frozen=True)
class Move:
    """The result of [`resample_move`][virgil.mode_search.resample_move].

    ``draws`` are the moved particles, ``acceptance`` the mean HMC acceptance
    over the moves and ``step`` the adapted step size. ``stuck`` is true when
    fewer than one move per particle is accepted on average
    (``acceptance * n_moves < 1``): the draws are then the resampled weighted
    draws, with duplicates, not rejuvenated.
    """

    draws: object
    acceptance: float
    step: float
    n_moves: int

    @property
    def stuck(self):
        """Whether the moves accepted less than once per particle."""
        return not self.acceptance * self.n_moves >= 1.0


def logmeanexp(a):
    """``ln mean exp a`` (``-inf`` if every entry is)."""
    a = onp.asarray(a, float)
    m = a.max()
    if not onp.isfinite(m):
        return float(m)
    return float(m + onp.log(onp.mean(onp.exp(a - m))))


def laplace_log_evidence(mode):
    """Laplace ``ln Z`` of a mode: ``log_target + d/2 ln 2π + ½ ln |cov|``."""
    x, cov = onp.asarray(mode.x), onp.asarray(mode.cov)
    return float(
        mode.log_target
        + 0.5 * len(x) * math.log(2.0 * math.pi)
        + 0.5 * onp.linalg.slogdet(cov)[1]
    )


def _mahalanobis2(a, b, cov):
    r = onp.asarray(a, float) - onp.asarray(b, float)
    return float(r @ onp.linalg.solve(onp.asarray(cov, float), r))


def distinct_modes(modes):
    """The modes, best first, dropping any within ``MAHAL`` of a kept one.

    A mode is the same as a kept one when its squared Mahalanobis distance is
    below ``MAHAL`` under *both* covariances: a broad partial-outlier mode
    never hides a needle (the one-sided rule lost 62 nats).
    """
    kept = []
    for m in sorted(modes, key=lambda m: -m.log_target):
        if not any(
            max(
                _mahalanobis2(m.x, k.x, k.cov),
                _mahalanobis2(m.x, k.x, m.cov),
            )
            < MAHAL
            for k in kept
        ):
            kept.append(m)
    return kept


@jax.tree_util.register_pytree_node_class
class DefensiveMixture:
    """``q = α prior + (1 - α) Σ_m w_m t_DF(g x_m, g cov_m gᵀ)``.

    ``w_m`` is proportional to the Laplace evidence of mode ``m``, split
    equally over the ``len(images) + 1`` components of its images ``g`` (exact
    linear symmetries of the target; the identity is implicit). The component
    arrays are padded to a power of two (at least ``M_PAD``) with log-weight
    ``-inf``, so the shapes (and a compiled ``log_q``) do not depend on the
    number of modes. With no modes ``q`` is the prior (``α = 1``).

    Parameters
    ----------
    modes : list of Mode
        Distinct modes.
    log_prior : callable
        ``log_prior(x)``, a function with a stable identity (it is static
        under ``jit``).
    sample_prior : callable
        ``sample_prior(key, n)`` draws ``(n, d)`` from the prior.
    images : sequence of (d, d) arrays
        Non-identity symmetry maps.
    dim : int, optional
        Dimension of ``x``; fixes the padded shapes of an empty (prior-only)
        mixture so that it shares the compiled kernels of the other cells.
    """

    def __init__(
        self, modes, log_prior, sample_prior, images=(), alpha=ALPHA, df=DF,
        dtype=None, dim=None,
    ):  # fmt: skip
        self.modes = list(modes)
        self.log_prior, self.sample_prior, self.df = (
            log_prior,
            sample_prior,
            float(df),
        )
        maps = [None, *images]
        if self.modes:
            d = len(onp.asarray(self.modes[0].x))
            lz = onp.array([laplace_log_evidence(m) for m in self.modes])
            lw = lz - onp.logaddexp.reduce(lz) - math.log(len(maps))
        else:
            d, alpha = 0 if dim is None else dim, 1.0
        mu, cov, logw = [], [], []
        for m, w in zip(self.modes, lw if self.modes else []):
            for g in maps:
                x, c = onp.asarray(m.x, float), onp.asarray(m.cov, float)
                if g is not None:
                    g = onp.asarray(g, float)
                    x, c = g @ x, g @ c @ g.T
                mu.append(x), cov.append(0.5 * (c + c.T)), logw.append(w)
        self.d = d if dim is None else dim
        m_pad = max(M_PAD, 1 << math.ceil(math.log2(max(len(mu), 1))))
        pad = m_pad - len(mu)
        eye = onp.eye(max(d, 1))
        mu = onp.array(mu + [onp.zeros(max(d, 1))] * pad)
        cov = onp.array(cov + [eye] * pad)
        chol = onp.linalg.cholesky(cov)
        dt = dtype or np.result_type(float)
        self.log_w = np.asarray(logw + [-onp.inf] * pad, dt)
        self.mu = np.asarray(mu, dt)
        self.chol = np.asarray(chol, dt)
        self.linv = np.asarray(onp.linalg.inv(chol), dt)
        self.logdet = np.asarray(
            2.0 * onp.log(onp.diagonal(chol, axis1=1, axis2=2)).sum(-1), dt
        )
        self.log_alpha = np.asarray(math.log(alpha), dt)
        self.log_beta = np.asarray(
            math.log1p(-alpha) if alpha < 1.0 else -onp.inf, dt
        )

    def tree_flatten(self):
        children = (
            self.log_w, self.mu, self.chol, self.linv, self.logdet,
            self.log_alpha, self.log_beta,
        )  # fmt: skip
        return children, (self.log_prior, self.sample_prior, self.df, self.d)

    @classmethod
    def tree_unflatten(cls, aux, children):
        new = object.__new__(cls)
        (new.log_w, new.mu, new.chol, new.linv, new.logdet,
         new.log_alpha, new.log_beta) = children  # fmt: skip
        new.log_prior, new.sample_prior, new.df, new.d = aux
        new.modes = []
        return new

    def log_q(self, x):
        """``ln q(x)`` of one vector."""
        df, d = self.df, x.shape[-1]
        z = np.einsum("mij,mj->mi", self.linv, x[None] - self.mu)
        lt = (  # log density of the multivariate t of every component at x
            -0.5 * (df + d) * np.log1p(np.sum(z**2, -1) / df)
            - 0.5 * self.logdet
            + math.lgamma(0.5 * (df + d))
            - math.lgamma(0.5 * df)
            - 0.5 * d * math.log(df * math.pi)
        )
        return np.logaddexp(
            self.log_alpha + self.log_prior(x),
            self.log_beta + jax.scipy.special.logsumexp(self.log_w + lt),
        )

    def sample(self, key, n):
        """``n`` exact draws from ``q``."""
        k = jax.random.split(key, 4)
        logits = np.concatenate(
            [self.log_alpha[None], self.log_beta + self.log_w]
        )
        comp = jax.random.categorical(k[0], logits, shape=(n,))
        idx = np.maximum(comp - 1, 0)
        z = jax.random.normal(k[1], (n, self.mu.shape[1]), self.mu.dtype)
        g = jax.random.gamma(k[2], 0.5 * self.df, (n,), self.mu.dtype)
        g = g / (0.5 * self.df)
        t = (
            self.mu[idx]
            + np.einsum("nij,nj->ni", self.chol[idx], z)
            / (np.sqrt(g)[:, None])
        )
        prior = self.sample_prior(k[3], n)
        if self.d == 0:
            return prior
        return np.where((comp == 0)[:, None], prior, t)


@functools.partial(jax.jit, static_argnames=("log_target", "inliers_fn", "n"))
def _is_kernel(log_target, inliers_fn, theta, mixture, key, n):
    """Draws from ``q``, ``ln p - ln q`` and the inlier count of each."""
    x = mixture.sample(key, n)
    lt = jax.vmap(lambda v: log_target(v, theta))(x)
    lq = jax.vmap(mixture.log_q)(x)
    inl = jax.vmap(lambda v: inliers_fn(v, theta))(x)
    return x, lt - lq, inl


def importance_evidence(
    log_target, theta, mixture, key, n=N_IS, *, inliers_fn, log_floor,
    m_min=M_MIN,
):  # fmt: skip
    """``ln Z`` of ``log_target`` by importance sampling from ``mixture``.

    The sentinel ``at_floor`` fires when the weighted mean of
    ``inliers_fn(x, theta)`` is below ``m_min`` while a mode with at least
    ``m_min`` inliers has a Laplace ``ln Z`` above ``log_floor + W``;
    ``log_floor`` is the ``ln Z`` of the all-outlier target (a mode's peak
    height says nothing of its mass: a ridge fitted by ``m_min`` nights is
    high and thin).

    Returns
    -------
    Evidence
        The weighted draws are exact posterior draws once resampled.
    """
    x, lw, inl = _is_kernel(log_target, inliers_fn, theta, mixture, key, n)
    lw = onp.asarray(lw, float)
    bad = onp.isnan(lw) | onp.isposinf(lw)
    lw = onp.where(bad, -onp.inf, lw)
    inl = onp.asarray(inl, float)
    finite = onp.isfinite(lw)
    log_z = logmeanexp(lw)
    if finite.any():
        w = onp.exp(lw - lw[finite].max())
        ess = float(w.sum() ** 2 / (w**2).sum())
        n_in = float((w * inl).sum() / w.sum())
    else:
        ess, n_in = 0.0, 0.0
    err = math.sqrt(max(1.0 / max(ess, 1.0) - 1.0 / n, 0.0))
    found = [
        laplace_log_evidence(m) for m in mixture.modes if m.n_inliers >= m_min
    ]
    at_floor = bool(n_in < m_min and found and max(found) > log_floor + W)
    return Evidence(
        log_z, err, ess, onp.asarray(x), lw, at_floor, ess < ESS_TARGET,
        n_in, int(bad.sum()),
    )  # fmt: skip


@functools.partial(
    jax.jit, static_argnames=("log_target", "n_draws", "n_moves", "n_warm")
)
def _move_kernel(
    log_target, theta, x, log_w, inv_mass, key, eps0, n_draws, n_moves, n_warm
):  # fmt: skip
    """Systematic resample, dual-averaged warm-up, then ``n_moves`` HMC steps."""
    k_res, k_warm, k_move = jax.random.split(key, 3)
    w = np.exp(log_w - jax.scipy.special.logsumexp(log_w))
    u = (jax.random.uniform(k_res, ()) + np.arange(n_draws)) / n_draws
    parts = x[np.minimum(np.searchsorted(np.cumsum(w), u), w.shape[0] - 1)]
    logdensity = lambda v: log_target(v, theta)  # noqa: E731
    kernel = hmc.build_kernel()
    states = jax.vmap(lambda v: hmc.init(v, logdensity))(parts)

    def step(states, k, eps, jitter):
        n_leap = np.clip(np.ceil(0.5 * np.pi / eps), 1, L_MAX).astype(int)
        e = eps * (
            1.0 + jitter * jax.random.uniform(k, (), minval=-1.0, maxval=1.0)
        )
        keys = jax.random.split(
            jax.random.fold_in(k, 1), states.position.shape[0]
        )
        new, info = jax.vmap(
            lambda kk, s: kernel(kk, s, logdensity, e, inv_mass, n_leap)
        )(keys, states)
        return new, info.acceptance_rate.mean()

    da_init, da_update, da_final = dual_averaging.dual_averaging()
    n_pilot = min(N_PILOT, n_draws)
    pilot = jax.tree_util.tree_map(lambda a: a[:n_pilot], states)

    def warm(c, i):
        pilot, da = c
        pilot, acc = step(
            pilot, jax.random.fold_in(k_warm, i), np.exp(da.log_x), 0.0
        )
        acc = np.where(np.isfinite(acc), acc, 0.0)
        return (pilot, da_update(da, ACCEPT - acc)), None

    (_, da), _ = jax.lax.scan(warm, (pilot, da_init(eps0)), np.arange(n_warm))
    eps = np.exp(da.log_x_avg)
    eps = np.where(np.isfinite(eps) & (eps > 0), eps, eps0)

    def move(states, i):
        states, acc = step(states, jax.random.fold_in(k_move, i), eps, JITTER)
        return states, acc

    states, acc = jax.lax.scan(move, states, np.arange(n_moves))
    return states.position, acc.mean(), eps


def resample_move(
    log_target, theta, ev, key, n_draws=N_POST, n_moves=N_HMC, cov=None
):
    """The posterior draws of ``ev`` after resampling and moving them.

    Systematic resampling of ``ev.x`` by ``ev.log_w``, then ``n_moves`` BlackJAX
    HMC transitions per draw: step size by dual averaging to ``ACCEPT`` over
    ``N_WARM`` warm-up transitions on a pilot of the resampled set, jittered by
    ``±JITTER``, trajectory ``⌈π/(2ε)⌉`` capped at ``L_MAX``, mass matrix
    ``cov`` (the dominant mode's, else the weighted sample covariance). The
    weighted draws are already exact: the moves only rejuvenate duplicates.

    Source: Gilks & Berzuini (2001), Hoffman & Gelman (2014), Neal (2011).
    """
    x = np.asarray(ev.x)
    lw = np.asarray(ev.log_w, x.dtype)
    if cov is None:
        w = onp.exp(ev.log_w - onp.max(ev.log_w))
        w /= w.sum()
        mean = w @ onp.asarray(ev.x)
        r = onp.asarray(ev.x) - mean
        cov = (r * w[:, None]).T @ r + 1e-12 * onp.eye(x.shape[1])
    d = x.shape[1]
    out, acc, eps = _move_kernel(
        log_target, theta, x, lw, np.asarray(cov, x.dtype), key,
        0.5 / math.sqrt(d), n_draws, n_moves, N_WARM,
    )  # fmt: skip
    return Move(onp.asarray(out), float(acc), float(eps), n_moves)
