"""Independent, whitened closure phases.

The closure phase of triangle (a, b, c) is φ_ab + φ_bc − φ_ac. Noise on the
baseline phases makes the closure phases of triangles that share a baseline
correlated, and with N telescopes only (N−1)(N−2)/2 of the triangles of one
frame and channel are independent: three of four, for four telescopes.
Treating all four as independent counts the closure phases 4/3 times.

The covariance model is that of Kammerer et al. (2020, A&A 644, A110,
§2.2): the reported variances σ² on the diagonal, and a fixed correlation
between triangles from equal noise on every baseline,

    C = D^½ R D^½,   R = T Tᵀ / 3,   D = diag(σ²),

where T is the triangle-by-baseline matrix of +1, +1, −1 entries; R has unit
diagonal and ±1/3 between triangles sharing a baseline. C keeps every
reported error, is never indefinite, and scales with the errors, so
rescaled or inflated errors carry through.

This is an approximation whenever the σ within a group differ. True
closure phases are exact closures T b of baseline phases b, so their noise
lies in the column space of T, while C spans D^½ times it: the two agree
only for equal σ. With unequal σ the χ² of real closure-phase noise is
slightly off its nominal distribution (for four telescopes with one noisy
baseline, a mean of 2.87 against the 3 expected for three independent
closure phases), and :meth:`ClosureNoise.sample` draws noise from C that is
not the closure of any set of baseline phases. The exact model would be
C = T diag(s) Tᵀ with baseline variances s chosen to reproduce σ².

``ClosureNoise`` groups the triangles that share baselines (one frame and
channel each). It whitens residuals r by dividing by σ, projecting onto an
orthonormal basis Q of the column space of T (the independent combinations)
and solving with the Cholesky factor of M = Q R Qᵀ, which is fixed. For r in
the column space of C this is exactly rᵀ C⁺ r, and the normalisation is
½ log pdet C = ½ (log det M + log det Q D Qᵀ).
"""

import equinox as eqx
import jax
import jax.numpy as np
import jax.scipy.linalg as jsl
import numpy as onp


def _groups(i1, i2, i3):
    """Triangles connected through shared baseline samples, as index lists."""
    n = len(i1)
    parent = onp.arange(n)

    def find(t):
        while parent[t] != t:
            parent[t] = parent[parent[t]]
            t = parent[t]
        return t

    owner = {}
    for t in range(n):
        for b in (int(i1[t]), int(i2[t]), int(i3[t])):
            if b in owner:
                parent[find(t)] = find(owner[b])
            else:
                owner[b] = t
    roots = onp.array([find(t) for t in range(n)])
    # Split by root with one stable sort: groups in root order, each sorted.
    order = onp.argsort(roots, kind="stable")
    _, starts = onp.unique(roots[order], return_index=True)
    return onp.split(order, starts[1:])


def _incidence(group, i1, i2, i3):
    """The group's triangle-by-baseline matrix T."""
    baselines = onp.unique(onp.concatenate([i1[group], i2[group], i3[group]]))
    col = {b: j for j, b in enumerate(baselines)}
    t = onp.zeros((group.size, baselines.size))
    for row, tri in enumerate(group):
        t[row, col[i1[tri]]] += 1.0
        t[row, col[i2[tri]]] += 1.0
        t[row, col[i3[tri]]] -= 1.0
    return t


class ClosureNoise(eqx.Module):
    """Correlated closure-phase noise from equal noise on every baseline.

    The arrays are NumPy, built once and reused in either x64 mode. JAX
    (0.10 and later) caches the converted copy of a NumPy array by identity,
    whatever the mode it was made in, so a float32 fit followed by a float64
    one could be handed int indices of the wrong width. The indices are
    therefore int32, which is the same in both modes, and the floats are
    cast to the data's dtype at use.
    """

    groups: onp.ndarray  # (n_group, m) closure-phase indices, padded with 0
    mask: onp.ndarray  # (n_group, m) True for real triangles
    incidence: onp.ndarray  # (n_group, m, n_base) T / √3, for sampling
    basis: onp.ndarray  # (n_group, k, m) Q, orthonormal rows
    chol: onp.ndarray  # (n_group, k, k) Cholesky factor of Q R Qᵀ
    valid: onp.ndarray  # (n_group, k) True for real basis rows
    keep: onp.ndarray  # flat indices of the real rows, in group order

    @classmethod
    def from_indices(cls, i1, i2, i3):
        """Build from closure-phase indices, or ``None`` if none correlate."""
        i1, i2, i3 = (onp.asarray(i, dtype=int) for i in (i1, i2, i3))
        groups = _groups(i1, i2, i3)
        if all(g.size == 1 for g in groups):
            return None  # three telescopes: nothing to decorrelate
        # Groups mostly repeat a few incidence patterns (one per frame and
        # channel), so factorise each distinct T only once.
        factors = {}
        blocks = []
        for g in groups:
            t = _incidence(g, i1, i2, i3) / onp.sqrt(3.0)
            key = (t.shape, t.tobytes())
            if key not in factors:
                left, singular, _ = onp.linalg.svd(t, full_matrices=False)
                rank = int(onp.sum(singular > 1e-9 * singular.max()))
                q = left[:, :rank].T
                factors[key] = q, onp.linalg.cholesky(q @ (t @ t.T) @ q.T)
            blocks.append((g, t, *factors[key]))
        m = max(b[0].size for b in blocks)
        n_base = max(b[1].shape[1] for b in blocks)
        k = max(b[2].shape[0] for b in blocks)
        n = len(blocks)
        out = {
            "groups": onp.zeros((n, m), dtype=onp.int32),
            "mask": onp.zeros((n, m), dtype=bool),
            "incidence": onp.zeros((n, m, n_base)),
            "basis": onp.zeros((n, k, m)),
            "chol": onp.tile(onp.eye(k), (n, 1, 1)),
            "valid": onp.zeros((n, k), dtype=bool),
        }
        for j, (g, t, q, m_chol) in enumerate(blocks):
            r = q.shape[0]
            out["groups"][j, : g.size] = g
            out["mask"][j, : g.size] = True
            out["incidence"][j, : g.size, : t.shape[1]] = t
            out["basis"][j, :r, : g.size] = q
            out["chol"][j, :r, :r] = m_chol
            out["valid"][j, :r] = True
        keep = onp.flatnonzero(out["valid"].reshape(-1)).astype(onp.int32)
        return cls(**out, keep=keep)

    @property
    def size(self):
        """Number of independent closure phases."""
        return int(self.keep.size)

    def correlation(self, n_phase):
        """The dense correlation matrix R of all ``n_phase`` closure phases."""
        r = onp.zeros((n_phase, n_phase))
        for g, mask, t in zip(self.groups, self.mask, self.incidence):
            idx = g[mask]
            tt = t[mask]
            r[onp.ix_(idx, idx)] = tt @ tt.T
        return r

    def whiten(self, residuals, sigma):
        """Whitened independent combinations, and their effective errors.

        ``residuals`` and ``sigma`` have one entry per closure phase. The
        returned errors give the Gaussian normalisation: their log-sum is
        ½ log of the pseudo-determinant of the covariance.
        """
        sigma = np.asarray(sigma)[self.groups]
        x = np.where(
            self.mask, np.asarray(residuals)[self.groups] / sigma, 0.0
        )
        basis, chol = (np.asarray(a, x.dtype) for a in (self.basis, self.chol))
        a = np.einsum("gkm,gm->gk", basis, x)
        w = jsl.solve_triangular(chol, a[..., None], lower=True)[..., 0]
        var = np.where(self.mask, sigma**2, 0.0)
        qdq = np.einsum("gkm,gm,glm->gkl", basis, var, basis)
        pad = 1.0 - self.valid.astype(qdq.dtype)
        qdq = qdq + pad[:, :, None] * np.eye(pad.shape[1])
        scale = np.diagonal(np.linalg.cholesky(qdq), axis1=1, axis2=2)
        errors = np.diagonal(chol, axis1=1, axis2=2) * scale
        return w.reshape(-1)[self.keep], errors.reshape(-1)[self.keep]

    def sample(self, key, sigma, n_phase):
        """Closure-phase noise with the covariance used by :meth:`whiten`."""
        e = jax.random.normal(key, self.incidence.shape[::2])
        incidence = np.asarray(self.incidence, e.dtype)
        noise = np.einsum("gmb,gb->gm", incidence, e)
        noise = noise * np.asarray(sigma)[self.groups]
        # Each closure phase sits in exactly one group; padded slots add 0.
        out = np.zeros(n_phase, dtype=noise.dtype)
        return out.at[self.groups].add(np.where(self.mask, noise, 0.0))
