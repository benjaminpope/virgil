"""Analytic marginalisation of parameters that enter the model linearly.

The data are ``d = m + A w + n``: a model m, a design A (n × k) times
linear parameters w, and noise ``n ~ N(0, D)``, ``D = diag(σ²)``. With a
Gaussian prior ``w ~ N(μ, Λ)``, w integrates out in closed form (Luger,
Foreman-Mackey & Hogg 2017, *AJ* 153, 216):

    d ~ N(m + A μ, D + A Λ Aᵀ).

In whitened coordinates, ``x = D^-½ (d - m - A μ)`` and
``U = D^-½ A Λ^½`` (one column per parameter), the covariance is
``I + U Uᵀ``. Everything here works on ``(x, U)``. It returns a whitened
residual of the same length as x, so the likelihood keeps one residual
vector, and the log-determinant ``½ log det(I + UᵀU)``, kept as effective
errors or a log-normalisation.

**Rules** (``design/imaging_plan.md``, "Analytic marginalisation of linear
parameters"):

- **Keep the log-determinant.** It depends on the model wherever A or σ
  does, e.g. a flux scale times the model's spectrum, or fitted jitter.
- **The prior is stated by the caller, finite, and never derived from the
  data.** A flat prior is only a limit, up to a constant that depends on
  its width: the width is part of the model.
- **Report the conditional posterior of w** ([`posterior`][virgil._linear.posterior]).
- **A Gaussian on a scale-type parameter is a proposal only.** For a
  positive scale such as an OI_FLUX grey scale k, the Jeffreys prior is
  1/k on stated bounds. Reweight samples of the conditional posterior by
  ``1 / (k N(k; μ, Λ))`` (with bounds), and make no evidence claims from
  the Gaussian.

**Two ways to whiten, with the same result.**

- **Successive rank-one steps** ([`whiten_rank_one`][virgil._linear.whiten_rank_one],
  [`whiten_blocks`][virgil._linear.whiten_blocks]): a rank-one covariance
  ``I + w wᵀ`` is whitened by ``R = I - w wᵀ / (q (q + 1))``,
  ``q = √(1 + wᵀw)``, since ``R (I + w wᵀ) Rᵀ = I``, with log-determinant
  ``log(1 + wᵀw)``. Applying R to x and to the remaining columns, column by
  column, whitens ``I + U Uᵀ``. Only scalar square roots appear, so the
  gradients stay smooth where columns are degenerate (equal visibilities
  on every baseline) or widths go to zero, where an eigendecomposition's
  are not. Use it for the calibration gains, closure offsets and grey
  scales, blocked by frame.
- **A dense Cholesky** ([`whiten_cholesky`][virgil._linear.whiten_cholesky]):
  ``S = I + UᵀU = L Lᵀ`` and ``u = x - W (I + L⁻ᵀ)⁻¹ Wᵀ x``, with
  ``W = U L⁻ᵀ``. This is an exact square root, again without an
  eigendecomposition. Use it for a few well-conditioned columns, such as
  RV zero points or a four-column Thiele–Innes basis. It is O(n k²).

[`LinearMarginal`][virgil._linear.LinearMarginal] wraps either for one
design with a stated prior. The Laplace covariance of a fit's latent
parameters (``fitting._laplace_covariance``) uses the same Woodbury
identity, but to invert a curvature, not to marginalise a likelihood, so
it is not built on this module.
"""

import equinox as eqx
import jax
import jax.numpy as np
import jax.scipy.linalg as jsl
import numpy as onp

from ._utils import concrete

__all__ = [
    "LinearMarginal",
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
    """Whiten ``x`` for the covariance ``I + Σ w_j w_jᵀ``, block by block.

    ``rows`` (n_block, n_row) indexes ``x`` (padded out of range) and
    ``local`` (n_block, n_row, n_mode) holds each block's columns there;
    ``spanning`` (n, n_spanning) holds columns across blocks, whitened after
    them. Successive rank-one steps (see the module notes).

    Returns
    -------
    tuple
        The whitened ``x`` and, per entry, the log of the factor its
        effective error grows by. These sum to ``½ log det(I + UᵀU)``.
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
    """Conditional posterior of the standardised parameters ω.

    With ``w = μ + Λ^½ ω``, the prior is ``ω ~ N(0, I)``. Given
    ``x = D^-½ (d - m - A μ)``, the posterior is ``ω ~ N(S⁻¹ Uᵀ x, S⁻¹)``,
    with ``S = I + UᵀU``.

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
    """A design ``A`` whose parameters w are marginalised under a stated prior.

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
        How to whiten (see the module notes). ``"cholesky"`` (the default)
        suits a few well-conditioned columns. ``"rank_one"`` keeps
        gradients smooth when columns are degenerate.

    Notes
    -----
    The prior must be finite: a flat prior is a limit only up to a
    constant that depends on its width. For a scale-type parameter the
    Gaussian is a proposal: see the module notes on reweighting to the
    Jeffreys prior.
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

    def standardise(self, resid, sigma):
        """``(x, U)`` for residuals ``resid = d - m`` and errors ``sigma``."""
        sigma = np.asarray(sigma)
        x = (np.asarray(resid) - self.design @ self.prior_mean) / sigma
        U = (self.design / sigma[:, None]) @ self.prior_root
        return x, U

    def whiten(self, resid, sigma):
        """Whitened residuals and the log-normalisation.

        Returns
        -------
        tuple
            ``u`` with ``uᵀu = rᵀ(D + AΛAᵀ)⁻¹r`` (``r = d - m - Aμ``), and
            ``½ log det(D + AΛAᵀ) = Σ log σ + ½ log det(I + UᵀU)``.
        """
        x, U = self.standardise(resid, sigma)
        whiten = (
            whiten_cholesky if self.method == "cholesky" else whiten_rank_one
        )
        u, half_logdet = whiten(x, U)
        return u, np.sum(np.log(np.asarray(sigma))) + half_logdet

    def loglike(self, resid, sigma):
        """The normalised Gaussian log density of the marginalised data."""
        u, log_norm = self.whiten(resid, sigma)
        return -0.5 * u @ u - log_norm - 0.5 * u.size * np.log(2.0 * np.pi)

    def posterior(self, resid, sigma):
        """Conditional posterior of w: ``(mean, cov)``, in w's own units."""
        x, U = self.standardise(resid, sigma)
        mean, cov = posterior(x, U)
        root = self.prior_root
        return self.prior_mean + root @ mean, root @ cov @ root.T
