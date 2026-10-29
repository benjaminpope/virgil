"""Independent, whitened closure phases.

Triangles that share a baseline have correlated closure phases, and N
telescopes give only (N-1)(N-2)/2 independent ones per frame and channel.
The covariance is that of Kammerer et al. (2020, A&A 644, A110, §2.2):
``C = D^½ R D^½`` with ``R = T Tᵀ / 3`` (T the triangle-by-baseline matrix
of +1, +1, -1) and ``D = diag(σ²)``. It is exact only for equal σ within a
group, and OIFITS carries no per-baseline errors to do better.

`ClosureNoise` groups the triangles that share baselines, projects the
σ-scaled residuals onto an orthonormal basis of the column space of T and
whitens with the Cholesky factor of the fixed projected correlation; the
effective errors give ``½ log pdet C``.

The model, its approximation and the dtype notes: ``design/closure_noise.md``.
"""

import equinox as eqx
import jax
import jax.numpy as np
import jax.scipy.linalg as jsl
import numpy as onp


def _merge(n, links):
    """Union-find root of each of ``n`` items after merging every ``(i, j)`` link."""
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, j in links:
        parent[find(i)] = find(j)
    return onp.array([find(i) for i in range(n)])


def _groups(i1, i2, i3):
    """Triangles connected through shared baseline samples, as index lists."""
    n = len(i1)
    owner = {}

    def links():
        for t in range(n):
            for b in (int(i1[t]), int(i2[t]), int(i3[t])):
                if b in owner:
                    yield t, owner[b]
                else:
                    owner[b] = t

    roots = _merge(n, links())
    # Split by root with one stable sort: groups in root order, each sorted.
    order = onp.argsort(roots, kind="stable")
    _, starts = onp.unique(roots[order], return_index=True)
    return onp.split(order, starts[1:])


# deliberate: vectorised incidence matrix; test_closure_noise_is_unchanged swaps in the per-row loop (_slow_incidence) to prove equivalence
def _incidence(group, i1, i2, i3):
    """The group's triangle-by-baseline matrix T."""
    legs = onp.stack([i1[group], i2[group], i3[group]], axis=1)
    baselines, col = onp.unique(legs, return_inverse=True)
    col = col.reshape(legs.shape)
    t = onp.zeros((group.size, baselines.size))
    rows = onp.arange(group.size)
    onp.add.at(t, (rows, col[:, 0]), 1.0)
    onp.add.at(t, (rows, col[:, 1]), 1.0)
    onp.add.at(t, (rows, col[:, 2]), -1.0)
    return t


class ClosureNoise(eqx.Module):
    """Correlated closure-phase noise from equal noise on every baseline.

    The arrays are NumPy with int32 indices, so they are valid in either
    x64 mode; floats are cast to the data's dtype at use.
    """

    groups: onp.ndarray  # (n_group, m) closure-phase indices, padded with 0
    mask: onp.ndarray  # (n_group, m) True for real triangles
    incidence: onp.ndarray  # (n_group, m, n_base) T / √3, for sampling
    basis: onp.ndarray  # (n_group, k, m) Q, orthonormal rows
    chol: onp.ndarray  # (n_group, k, k) Cholesky factor of Q R Qᵀ
    valid: onp.ndarray  # (n_group, k) True for real basis rows
    keep: onp.ndarray  # flat indices of the real rows, in group order

    @classmethod
    def from_indices(cls, i1, i2, i3, force=False):
        """Build from closure-phase indices, or ``None`` if none correlate.

        ``force`` builds it anyway for three telescopes (every group one
        triangle, whitened by its own σ), which closure-phase offsets need.
        """
        i1, i2, i3 = (onp.asarray(i, dtype=int) for i in (i1, i2, i3))
        groups = _groups(i1, i2, i3)
        if not force and all(g.size == 1 for g in groups):
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
        returned errors give the Gaussian normalization: their log-sum is
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

    def colour(self, whitened, sigma, n_phase):
        """Closure-phase residuals whose :meth:`whiten` is ``whitened``.

        The inverse of :meth:`whiten` on the column space of the covariance
        (any draw of :meth:`sample`); a component outside it, which arises
        only for unequal σ within a group, is not recovered.

        Parameters
        ----------
        whitened : array-like
            One entry per independent combination (:attr:`size`), as
            returned by :meth:`whiten`.
        sigma : array-like
            One error per closure phase.
        n_phase : int
            Number of closure phases.
        """
        w = np.asarray(whitened)
        flat = np.zeros(self.valid.size, dtype=w.dtype)
        flat = flat.at[self.keep].set(w).reshape(self.valid.shape)
        basis, chol = (np.asarray(a, w.dtype) for a in (self.basis, self.chol))
        a = np.einsum("gkl,gl->gk", chol, flat)
        x = np.einsum("gkm,gk->gm", basis, a)
        r = x * np.asarray(sigma)[self.groups]
        # Each closure phase sits in exactly one group; padded slots add 0.
        out = np.zeros(n_phase, dtype=r.dtype)
        return out.at[self.groups].add(np.where(self.mask, r, 0.0))

    def sample(self, key, sigma, n_phase):
        """Closure-phase noise with the covariance used by :meth:`whiten`."""
        e = jax.random.normal(key, self.incidence.shape[::2])
        incidence = np.asarray(self.incidence, e.dtype)
        noise = np.einsum("gmb,gb->gm", incidence, e)
        noise = noise * np.asarray(sigma)[self.groups]
        # Each closure phase sits in exactly one group; padded slots add 0.
        out = np.zeros(n_phase, dtype=noise.dtype)
        return out.at[self.groups].add(np.where(self.mask, noise, 0.0))
