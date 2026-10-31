"""Analytic marginalization of parameters that enter the model linearly.

The data are ``d = m + A w + n`` with noise ``N(0, D)`` and a stated
Gaussian prior ``w ~ N(μ, Λ)``, so ``d ~ N(m + A μ, D + A Λ Aᵀ)``. In
whitened coordinates ``x = D^-½ (d - m - A μ)`` and ``U = D^-½ A Λ^½`` the
covariance is ``I + U Uᵀ``, and everything here works on ``(x, U)``: it
returns a whitened residual of the length of x and the log-determinant
``½ log det(I + UᵀU)``, which must be kept (it depends on the model).

The prior must be stated, finite and independent of the data. A Gaussian
on a positive scale parameter is a proposal only (reweight to Jeffreys).

Two whitenings agree: successive rank-one steps (`whiten_rank_one`,
`whiten_blocks`; smooth gradients for degenerate columns; for gains,
offsets and grey scales) and a dense Cholesky (`whiten_cholesky`; a few
well-conditioned columns). `LinearMarginal` wraps either.

``arrow_problem`` and ``arrow_at`` give the same marginal at any error scale
``s`` (covariance ``s² D + U Uᵀ``) without whitening again: a Cholesky per
block and a Schur complement on the spanning columns (Woodbury and the
determinant lemma). Source: Rasmussen & Williams (2006), [references](../references.md#rasmussen2006).

Derivations and rules: ``design/linear_marginalisation.md``.
"""

from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as np
import jax.scipy.linalg as jsl
import numpy as onp

from ._utils import concrete

__all__ = [
    "LinearMarginal",
    "arrow_at",
    "arrow_problem",
    "posterior",
    "whiten_blocks",
    "whiten_cholesky",
    "whiten_rank_one",
]


def _rank_one_step(stack, j):
    """Whiten ``stack`` (..., n_row, n_col) for its column ``j`` as a mode."""
    w = stack[..., j]
    s = np.sum(w**2, axis=-1)
    q = np.sqrt(1.0 + s)
    coef = np.einsum("...r,...rk->...k", w, stack) / (q * (q + 1.0))[..., None]
    return stack - w[..., None] * coef[..., None, :], np.log1p(s)


def whiten_blocks(x, rows, local, spanning=None):
    """Whiten ``x`` for ``I + Σ w_j w_jᵀ`` by rank-one steps, block by block.

    ``rows`` (n_block, n_row) indexes ``x`` (padded out of range),
    ``local`` (n_block, n_row, n_mode) holds each block's columns, and
    ``spanning`` (n, n_spanning) columns across blocks, whitened after.

    Returns
    -------
    tuple
        The whitened ``x`` and, per entry, its share of
        ``½ log det(I + UᵀU)`` (these sum to it).
    """
    x = np.asarray(x)
    n = x.shape[0]
    if spanning is None:
        spanning = np.zeros((n, 0), x.dtype)
    n_span = spanning.shape[1]
    # The blocks first, carrying the spanning columns through them: the
    # whitening of the blocks is applied to every column.
    stack = np.concatenate(
        [
            x.at[rows].get(mode="fill", fill_value=0)[..., None],
            spanning.at[rows].get(mode="fill", fill_value=0),
            local,
        ],
        axis=-1,
    )
    stack, logdets = jax.lax.scan(
        _rank_one_step, stack, 1 + n_span + np.arange(local.shape[-1])
    )
    done = (
        np.concatenate([x[:, None], spanning], axis=1)
        .at[rows]
        .set(stack[..., : 1 + n_span], mode="drop")
    )
    # Spread each block's ½ log det over its rows.
    n_rows = np.sum(rows < n, axis=1)
    per_row = 0.5 * np.sum(logdets, axis=0) / np.maximum(n_rows, 1)
    extra = (
        np.zeros_like(x)
        .at[rows]
        .set(np.broadcast_to(per_row[:, None], rows.shape), mode="drop")
    )
    if n_span:
        # Then the spanning columns, as one dense block, their ½ log det
        # spread over every entry.
        done, span_logdets = jax.lax.scan(
            _rank_one_step, done[None], 1 + np.arange(n_span)
        )
        done = done[0]
        extra = extra + 0.5 * np.sum(span_logdets) / n
    return done[:, 0], extra


class ArrowProblem(NamedTuple):
    """A block-arrow ``Ũ`` reduced to the Gram sums ``arrow_at`` needs."""

    x: jax.Array  # (n,) residuals, whitened by D^-½ (s = 1)
    rows: jax.Array  # (n_block, n_row), padded out of range
    local: jax.Array  # (n_block, n_row, n_mode)
    spanning: jax.Array  # (n, n_span), n_span may be 0
    gram_bb: jax.Array  # (n_block, n_mode, n_mode)
    gram_bs: jax.Array  # (n_block, n_mode, n_span)
    gram_out: jax.Array  # (n_span, n_span): spanning rows outside every block
    proj_b: jax.Array  # (n_block, n_mode): Ũᵀx


def arrow_problem(x, rows, local, spanning=None):
    """Gram sums of the covariance ``s² I + Ũ Ũᵀ`` in ``whiten_blocks``' layout.

    ``x`` is whitened by ``D^-½`` at ``s = 1``; ``rows`` (n_block, n_row),
    ``local`` (n_block, n_row, n_mode) and ``spanning`` (n, n_span) are as for
    ``whiten_blocks``. Evaluate it at any ``s`` with ``arrow_at``.
    """
    x, local, rows = np.asarray(x), np.asarray(local), np.asarray(rows)
    if spanning is None:
        spanning = np.zeros((x.shape[0], 0), x.dtype)
    # A padded row is no observation: its columns add nothing (as in
    # ``whiten_blocks``, which drops the log det of a block with no rows).
    local = np.where((rows < x.shape[0])[..., None], local, 0)
    hi = jax.lax.Precision.HIGHEST
    x_b = x.at[rows].get(mode="fill", fill_value=0)
    s_b = spanning.at[rows].get(mode="fill", fill_value=0)
    outside = np.ones(x.shape[0], x.dtype).at[rows].set(0, mode="drop")
    return ArrowProblem(
        x,
        rows,
        local,
        spanning,
        np.einsum("brm,brn->bmn", local, local, precision=hi),
        np.einsum("brm,brk->bmk", local, s_b, precision=hi),
        np.matmul(spanning.T * outside, spanning, precision=hi),
        np.einsum("brm,br->bm", local, x_b, precision=hi),
    )


def _solve_spd(chol, rhs):
    return jsl.cho_solve((chol, True), rhs)


def arrow_at(problem, scale2, trace=False):
    """The marginal of ``arrow_problem`` at the variance scale ``s² = scale2``.

    Returns ``(chi2, norm, r2)``: ``χ² = xᵀ(I + ŨŨᵀ/s²)⁻¹x/s²`` written stably
    as ``‖x - Ũy‖²/s² + ‖y‖²`` with ``y = (s²I + ŨᵀŨ)⁻¹Ũᵀx``;
    ``norm = ½ log det(I + ŨᵀŨ/s²)``; and ``r2 = ‖x - Ũy‖²``. With
    ``trace=True`` also ``tr((s²I + ŨᵀŨ)⁻¹)``, so that ``∂norm/∂ln s =
    s² tr - K`` and ``∂χ²/∂ln s = -2 r2/s²`` (K the number of columns).
    """
    p = problem
    n_block, n_mode = p.gram_bb.shape[:2]
    n_span = p.gram_out.shape[0]
    eye = np.eye(n_mode, dtype=p.gram_bb.dtype)
    chol_b = np.linalg.cholesky(p.gram_bb + scale2 * eye)
    rhs = np.concatenate([p.proj_b[..., None], p.gram_bs], axis=-1)
    sol = jax.vmap(_solve_spd)(chol_b, rhs)
    m_g, m_gs = sol[..., 0], sol[..., 1:]
    eye_s = np.eye(n_span, dtype=p.gram_out.dtype)
    # S = s²I + ZᵀZ + s²WᵀW with Z = Ũ_s - Ũ_b W: a sum of squares, with
    # none of the cancellation of the Gram difference G_ss - G_bsᵀB⁻¹G_bs.
    s_b = p.spanning.at[p.rows].get(mode="fill", fill_value=0)
    z = s_b - np.einsum("brm,bmk->brk", p.local, m_gs)
    hi = jax.lax.Precision.HIGHEST
    schur = (
        scale2 * eye_s
        + p.gram_out
        + np.einsum("brk,brl->kl", z, z, precision=hi)
        + scale2 * np.einsum("bmk,bml->kl", m_gs, m_gs, precision=hi)
    )
    chol_s = np.linalg.cholesky(schur)
    # Ũ_sᵀ(x - Ũ_b m_g), formed on the residual of the block fit: the Gram
    # form (Ũ_sᵀx - G_bsᵀm_g) cancels, and χ² amplifies an error in y by κ.
    fit_b = (
        np.zeros_like(p.x)
        .at[p.rows]
        .add(np.einsum("brm,bm->br", p.local, m_g), mode="drop")
    )
    rhs_s = np.matmul(p.spanning.T, p.x - fit_b, precision=hi)
    y_s = _solve_spd(chol_s, rhs_s)
    y_b = m_g - np.einsum("bmk,k->bm", m_gs, y_s)
    fit = (
        np.zeros_like(p.x)
        .at[p.rows]
        .add(np.einsum("brm,bm->br", p.local, y_b), mode="drop")
    )
    r2 = np.sum((p.x - fit - p.spanning @ y_s) ** 2)
    chi2 = r2 / scale2 + np.sum(y_b**2) + np.sum(y_s**2)
    log_diag = np.sum(np.log(np.diagonal(chol_b, axis1=1, axis2=2)))
    log_diag = log_diag + np.sum(np.log(np.diagonal(chol_s)))
    norm = log_diag - 0.5 * (n_block * n_mode + n_span) * np.log(scale2)
    if not trace:
        return chi2, norm, r2
    inv_b = jax.vmap(lambda c: _solve_spd(c, eye))(chol_b)
    z = np.einsum("bmk,bml->kl", m_gs, m_gs)
    tr = np.sum(np.diagonal(inv_b, axis1=1, axis2=2))
    tr = tr + np.trace(_solve_spd(chol_s, z + eye_s))
    return chi2, norm, r2, tr


def whiten_rank_one(x, U):
    """Whiten ``x`` for ``I + U Uᵀ`` by successive rank-one steps.

    Returns
    -------
    tuple
        The whitened ``x`` (same length) and ``½ log det(I + UᵀU)``.
    """
    x = np.asarray(x)
    rows = np.arange(x.shape[0])[None]
    u, extra = whiten_blocks(x, rows, np.asarray(U)[None])
    return u, np.sum(extra)


def whiten_cholesky(x, U):
    """Whiten ``x`` for ``I + U Uᵀ`` with a dense k × k Cholesky.

    Returns
    -------
    tuple
        The whitened ``x`` (same length) and ``½ log det(I + UᵀU)``.
    """
    x, U = np.asarray(x), np.asarray(U)
    k = U.shape[1]
    eye = np.eye(k, dtype=U.dtype)
    L = np.linalg.cholesky(eye + U.T @ U)
    L_inv_t = jsl.solve_triangular(L, eye, lower=True).T  # L⁻ᵀ, upper
    W = U @ L_inv_t
    y = jsl.solve_triangular(eye + L_inv_t, W.T @ x, lower=False)
    return x - W @ y, np.sum(np.log(np.diag(L)))


def posterior(x, U):
    """Conditional posterior ``N(S⁻¹ Uᵀ x, S⁻¹)``, ``S = I + UᵀU``, of ω.

    ``ω`` is the standardized parameter, ``w = μ + Λ^½ ω``.

    Returns
    -------
    tuple
        ``(mean, cov)`` of ω, of shapes (k,) and (k, k).
    """
    x, U = np.asarray(x), np.asarray(U)
    S = np.eye(U.shape[1], dtype=U.dtype) + U.T @ U
    factor = jsl.cho_factor(S, lower=True)
    return jsl.cho_solve(factor, U.T @ x), jsl.cho_solve(
        factor, np.eye(U.shape[1], dtype=U.dtype)
    )


class LinearMarginal(eqx.Module):
    """A design ``A`` whose parameters w are marginalized under a stated prior.

    The prior must be finite (a flat prior is not supported); a Gaussian on
    a scale-type parameter is a proposal only (reweight to Jeffreys).

    Parameters
    ----------
    design : array-like
        ``A``, ``(n, k)``: how each parameter enters the data.
    prior_mean : array-like
        ``μ``, ``(k,)`` or a scalar: the prior mean, in the parameters'
        units. It is stated, never estimated from the data.
    prior_sd : array-like, optional
        Prior standard deviations ``(k,)`` (independent parameters).
    prior_cov : array-like, optional
        A full prior covariance ``(k, k)``, instead of ``prior_sd``.
    method : {"cholesky", "rank_one"}, optional
        ``"cholesky"`` (the default) suits a few well-conditioned columns;
        ``"rank_one"`` keeps gradients smooth when columns are degenerate.
    """

    design: jax.Array
    prior_mean: jax.Array
    prior_root: jax.Array  # Λ^½, lower triangular
    method: str = eqx.field(static=True)

    def __init__(
        self,
        design,
        prior_mean,
        prior_sd=None,
        prior_cov=None,
        method="cholesky",
    ):
        design = np.asarray(design)
        if not np.issubdtype(design.dtype, np.inexact):
            # An integer indicator design would truncate fractional priors.
            design = design.astype(np.result_type(float))
        if design.ndim != 2:
            raise ValueError("design must be (n, k).")
        k = design.shape[1]
        if (prior_sd is None) == (prior_cov is None):
            raise ValueError("State the prior: give prior_sd or prior_cov.")
        if prior_cov is None:
            sd = np.broadcast_to(np.asarray(prior_sd, design.dtype), (k,))
            root = np.diag(sd)
            check = concrete(sd)
            if check is not None and not onp.all(
                onp.isfinite(check) & (check > 0)
            ):
                raise ValueError(
                    "The prior needs positive, finite sd: a flat prior is "
                    "not supported (state a finite width)."
                )
        else:
            cov = np.asarray(prior_cov, design.dtype)
            if cov.shape != (k, k):
                raise ValueError(f"prior_cov must be ({k}, {k}).")
            root = np.linalg.cholesky(cov)
            check = concrete(root)
            if check is not None and not onp.all(onp.isfinite(check)):
                raise ValueError(
                    "prior_cov must be finite and positive definite."
                )
        mean = np.broadcast_to(np.asarray(prior_mean, design.dtype), (k,))
        check = concrete(mean)
        if check is not None and not onp.all(onp.isfinite(check)):
            raise ValueError("The prior needs a finite mean.")
        if method not in ("cholesky", "rank_one"):
            raise ValueError("method must be 'cholesky' or 'rank_one'.")
        self.design = design
        self.prior_mean = mean
        self.prior_root = root
        self.method = method

    def standardize(self, resid, sigma):
        """``(x, U)`` for residuals ``resid = d - m`` and errors ``sigma``."""
        sigma = np.asarray(sigma)
        x = (np.asarray(resid) - self.design @ self.prior_mean) / sigma
        U = (self.design / sigma[:, None]) @ self.prior_root
        return x, U

    def whiten(self, resid, sigma):
        """Whitened residuals and the log-normalization.

        Returns
        -------
        tuple
            ``u`` with ``uᵀu = rᵀ(D + AΛAᵀ)⁻¹r`` (``r = d - m - Aμ``), and
            ``½ log det(D + AΛAᵀ) = Σ log σ + ½ log det(I + UᵀU)``.
        """
        x, U = self.standardize(resid, sigma)
        whiten = (
            whiten_cholesky if self.method == "cholesky" else whiten_rank_one
        )
        u, half_logdet = whiten(x, U)
        return u, np.sum(np.log(np.asarray(sigma))) + half_logdet

    def loglike(self, resid, sigma):
        """The normalized Gaussian log density of the marginalized data."""
        u, log_norm = self.whiten(resid, sigma)
        return -0.5 * u @ u - log_norm - 0.5 * u.size * np.log(2.0 * np.pi)

    def posterior(self, resid, sigma):
        """Conditional posterior of w: ``(mean, cov)``, in w's own units."""
        x, U = self.standardize(resid, sigma)
        mean, cov = posterior(x, U)
        root = self.prior_root
        return self.prior_mean + root @ mean, root @ cov @ root.T
