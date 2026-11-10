"""A dense damped-Newton (Levenberg–Marquardt) kernel for negative log posteriors.

Minimizes a scalar Φ(z) of a flat vector z using the exact autodiff
gradient and the curvature ``F = JᵀJ + diag(P)``, ``J = ∂ρ/∂z``, where ρ(z)
are *curvature rows* (for least squares, the whitened residuals) and
``P ≥ 0`` is the diagonal from elementwise priors and transforms. Steps are
accepted only if Φ falls, so an error in F costs speed or a ``STALLED``
stop, never a wrong answer.

A *kernel problem* is an ``eqx.Module`` with methods of a flat float
vector: ``loss(z)`` (scalar Φ), ``rows(z)`` (any length, possibly 0) and
``diag(z)`` (P, length n). Data are module fields so they are traced;
static layouts use ``eqx.field(static=True)``.

`step` solves ``(F + λM) δ = -g`` by Cholesky on the free coordinates
(M: Marquardt scaling, diag F per parameter, 1 where it is zero), clips the trial into the box and
accepts it if everything is finite and Φ falls (λ/3 on accept, 4λ on
reject; a failed Cholesky gives NaNs, so it is a rejection). Coordinates
at a bound with the descent direction pointing out of the box are held.
Stop codes: ``CONVERGED`` (projected gradient ≤ ``gtol``, Newton decrement
≤ ``dtol``), ``STALLED`` (λ > 1e10 with neither met, e.g. a tolerance finer
than Φ's rounding error; never promoted to ``CONVERGED``), ``LIMIT``, and
``NON_FINITE`` from `init`. A stopped state is unchanged by ``step``; `solve` resumes from a
`State`, counting ``max_steps`` from ``state.steps``.

Private; imports nothing from virgil. The algorithm, tolerances and tests
are in ``design/history/gauss_newton_plan.md``.
"""

import equinox as eqx
import jax
import jax.numpy as np

RUNNING, CONVERGED, STALLED, LIMIT, NON_FINITE = 0, 1, 2, 3, 4

_LAM_START = 1e-3
_LAM_FLOOR = 1e-12
_LAM_STALL = 1e10
_BOUND_EPS = 1e-10
# The decrement (nats) a float64 run reaches whatever Φ: the refinement
# callers' value, kept so their acceptance set is unchanged. Not
# derived; the decrement is computed from the gradient, whose rounding is not
# bounded by Φ's.
_DTOL_AUTO = 1e-12


class State(eqx.Module):
    """The state of a damped-Newton run at one point; see the module docstring."""

    z: jax.Array
    value: jax.Array
    grad: jax.Array
    curvature: jax.Array
    lam: jax.Array
    steps: jax.Array
    rejects: jax.Array
    stop: jax.Array  # int32 stop code


def curvature(problem, z):
    """``JᵀJ + diag(P)`` at z, with J by forward mode, symmetrized."""
    J = jax.jacfwd(problem.rows)(z)
    F = J.T @ J + np.diag(problem.diag(z))
    return 0.5 * (F + F.T)


def rounding_dtol(value):
    """The decrement (nats) below which Φ cannot tell two points apart.

    10 ulp of max(|Φ|, 1). Derivation, when Φ is a sum of n terms of one sign
    (a chi-squared): its rounding error is at most about log2(n) ε Σ|xᵢ| =
    log2(n) ε |Φ| for tree-wise summation (Higham 2002, §4.2), and log2(n) is
    10 at n = 10³ (13 at 10⁴); the floor of 1 is one nat. Not derived for a
    Φ whose terms have both signs, such as the marginal surface of `epochs`
    (a sum of ν log χ² per block), where |Φ| can be far below Σ|xᵢ|: there
    it is the allowance the kernel's former rescue used, kept so as not to
    change behaviour.
    """
    return 10 * np.finfo(value.dtype).eps * np.maximum(np.abs(value), 1.0)


def _held(z, g, lower, upper):
    """The ε-active set: coordinates whose descent direction leaves the box."""
    rel = max(_BOUND_EPS, 10 * np.finfo(z.dtype).eps)
    eps_lo = rel * (1.0 + np.abs(np.where(np.isfinite(lower), lower, 0.0)))
    eps_hi = rel * (1.0 + np.abs(np.where(np.isfinite(upper), upper, 0.0)))
    at_lower = np.isfinite(lower) & (z <= lower + eps_lo) & (g > 0)
    at_upper = np.isfinite(upper) & (z >= upper - eps_hi) & (g < 0)
    return at_lower | at_upper


def _solve_free(F, g, free, lam):
    """δ of ``(F + λM) δ = -g`` on the free coordinates, 0 on held ones.

    NaN when the system is not positive definite (Cholesky does not raise).
    """
    # Marquardt (1963) scaling M = diag F, per parameter: the damping carries
    # each coordinate's own units, so the step is invariant to rescaling z
    # (the property Moré 1978 builds on). Plain diag F: without Moré's running
    # max D_k = max(D_{k-1}, diag F_k), so a diagonal that is tiny but not
    # zero keeps its tiny damping and the run stalls honestly. A coordinate
    # with exactly zero curvature gets 1 (MINPACK's convention). There is no
    # cross-parameter floor, which would give a small-unit coordinate the
    # damping of the largest.
    d = np.diag(F)
    scaling = np.where(d > 0.0, d, 1.0)
    A = F + lam * np.diag(scaling)
    both = free[:, None] & free[None, :]
    A = np.where(both, A, 0.0) + np.diag(np.where(free, 0.0, 1.0))
    rhs = np.where(free, -g, 0.0)
    L = np.linalg.cholesky(A)
    delta = jax.scipy.linalg.cho_solve((L, True), rhs)
    return np.where(free, delta, 0.0)


def _evaluate(problem, z):
    value, grad = jax.value_and_grad(problem.loss)(z)
    return value, grad, curvature(problem, z)


def init(problem, z0, lower, upper):
    """The starting `State` at ``z0`` (clipped into the box).

    ``stop`` is ``NON_FINITE`` if Φ, its gradient or F is not finite there.
    """
    z = np.clip(np.asarray(z0), lower, upper)
    value, grad, F = _evaluate(problem, z)
    finite = (
        np.isfinite(value) & np.all(np.isfinite(grad)) & np.all(np.isfinite(F))
    )
    return State(
        z=z,
        value=value,
        grad=grad,
        curvature=F,
        lam=np.asarray(_LAM_START, z.dtype),
        steps=np.zeros((), np.int32),
        rejects=np.zeros((), np.int32),
        stop=np.where(finite, RUNNING, NON_FINITE).astype(np.int32),
    )


# deliberate: lax.while_loop body, jitted with its solve
def _advance(problem, state, lower, upper, gtol, dtol, limit):
    z, g, F = state.z, state.grad, state.curvature
    free = ~_held(z, g, lower, upper)
    delta = _solve_free(F, g, free, state.lam)
    trial = np.clip(z + delta, lower, upper)
    value, grad, F_trial = _evaluate(problem, trial)
    ok = (
        np.all(np.isfinite(delta))
        & np.isfinite(value)
        & np.all(np.isfinite(grad))
        & np.all(np.isfinite(F_trial))
    )
    accept = ok & (value < state.value)
    z = np.where(accept, trial, z)
    value = np.where(accept, value, state.value)
    g = np.where(accept, grad, g)
    F = np.where(accept, F_trial, F)
    lam = np.where(
        accept, np.maximum(state.lam / 3, _LAM_FLOOR), state.lam * 4
    )
    steps = state.steps + 1
    rejects = state.rejects + (~accept).astype(np.int32)

    free = ~_held(z, g, lower, upper)
    gnorm = np.max(np.where(free, np.abs(g), 0.0))
    # Newton decrement ½ gᵀF⁻¹g on the free coordinates (inf if F is not PD).
    # F carries the smallest damping, so a direction the rows ignore (and with
    # a zero gradient) contributes nothing instead of NaN.
    delta = _solve_free(F, g, free, _LAM_FLOOR)
    dec = -0.5 * np.sum(np.where(free, g * delta, 0.0))
    dec = np.where(np.isfinite(dec), dec, np.inf)
    # dtol=None (NaN here): the larger of _DTOL_AUTO and Φ's rounding resolution
    dtol = np.where(
        np.isnan(dtol), np.maximum(_DTOL_AUTO, rounding_dtol(value)), dtol
    )
    converged = (gnorm <= gtol) | (dec <= dtol)
    # Damping that ran away without meeting a criterion is a stall, reported
    # as such (callers name it "the damping limit"); it is never converged.
    code = np.select(
        [converged, lam > _LAM_STALL, steps >= limit],
        [CONVERGED, STALLED, LIMIT],
        default=RUNNING,
    ).astype(np.int32)
    return State(z, value, g, F, lam, steps, rejects, code)


def step(problem, state, lower, upper, gtol, dtol, max_steps):
    """One damped-Newton iteration; the identity once ``state.stop != RUNNING``.

    ``max_steps`` is the total ``steps`` at which the run stops with
    ``LIMIT``.
    """
    return jax.lax.cond(
        state.stop == RUNNING,
        lambda s: _advance(problem, s, lower, upper, gtol, dtol, max_steps),
        lambda s: s,
        state,
    )


@eqx.filter_jit
def _solve(problem, start, lower, upper, gtol, dtol, max_steps):
    if isinstance(start, State):
        keep = start.stop == NON_FINITE
        state = eqx.tree_at(
            lambda s: s.stop,
            start,
            np.where(keep, NON_FINITE, RUNNING).astype(np.int32),
        )
    else:
        state = init(problem, start, lower, upper)
    limit = state.steps + max_steps
    return jax.lax.while_loop(
        lambda s: s.stop == RUNNING,
        lambda s: step(problem, s, lower, upper, gtol, dtol, limit),
        state,
    )


def solve(
    problem,
    start,
    *,
    lower=None,
    upper=None,
    gtol=0.0,
    dtol=0.0,
    max_steps=200,
):
    """Run damped Newton on ``problem`` from ``start`` and return the `State`.

    Parameters
    ----------
    problem
        A kernel problem (module docstring).
    start
        A flat starting vector, or a `State` to resume (``max_steps`` more
        iterations from its ``steps``).
    lower, upper
        Box bounds, scalars or vectors; default ±inf.
    gtol, dtol
        Stop when the projected gradient's infinity norm is ≤ ``gtol`` or the
        Newton decrement is ≤ ``dtol``. A tolerance finer than
        the rounding error of Φ cannot be met: the run ends ``STALLED``.
        ``dtol=None`` is the larger of 1e-12 and that rounding error at the
        current Φ (`rounding_dtol`), the refinement callers' acceptance
        set (a tolerance of 1e-12, or the rounding rescue).
    max_steps
        Maximum iterations (accepted and rejected) in this call, at least 1.
    """
    if not isinstance(start, State):
        start = np.asarray(start)
        if not np.issubdtype(start.dtype, np.floating):
            start = start.astype(float)
    z = start.z if isinstance(start, State) else start
    dtype = z.dtype

    def bound(b, default):
        return np.broadcast_to(
            np.asarray(default if b is None else b, dtype), z.shape
        )

    return _solve(
        problem,
        start,
        bound(lower, -np.inf),
        bound(upper, np.inf),
        np.asarray(gtol, dtype),
        np.asarray(np.nan if dtol is None else dtol, dtype),
        np.asarray(max_steps, np.int32),
    )
