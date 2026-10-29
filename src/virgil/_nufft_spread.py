"""The spread (fine) grid of nufftax's type-1 NUFFT, evaluated exactly at any real position (private).

nufftax stops its public API at ``nufft2d1``; the spread grid comes from its
private ``_spread_2d_dispatch`` and ``compute_kernel_params``, pinned here
behind a version check. Poisson summation turns the grid ``b`` into
``Σ_k c_k e^{i t_k x}`` at any real ``x`` with ``|x| ≤ M/2`` (``n_f ≥ 2M``),
with ``ψ̂`` by Gauss–Legendre quadrature (differentiable in ``x``): a
type-3 evaluation at non-integer ``x``. Source:
[nufftax](../references.md#nufftax), [Barnett et al. (2019)](../references.md#barnett2019),
[Lee & Greengard (2005)](../references.md#leegreengard2005).
"""

import functools
import re

import jax
import jax.numpy as np
import numpy as onp

NUFFTAX_MIN = (0, 6, 1)
_NODES = 160  # Gauss–Legendre nodes of ψ̂: 4e-14 at 64² with 20 px shifts


def _nufftax():
    """nufftax's private spreader and kernel parameters, version-checked."""
    import nufftax

    try:
        from nufftax.core.kernel import compute_kernel_params, es_kernel
        from nufftax.core.spread import _spread_2d_dispatch
    except ImportError as err:
        raise ImportError(
            f"nufftax {nufftax.__version__} no longer has the private "
            f"spreader the Toeplitz radio term uses ({err})."
        ) from err

    # leading digits of each field, so "0.7rc1" reads as (0, 7)
    version = tuple(
        int(re.match(r"\d*", p).group() or 0)
        for p in nufftax.__version__.split(".")[:3]
    )
    if version < NUFFTAX_MIN:
        raise ImportError(
            f"the Toeplitz radio term needs nufftax >= "
            f"{'.'.join(map(str, NUFFTAX_MIN))}, not {nufftax.__version__}."
        )
    return compute_kernel_params, es_kernel, _spread_2d_dispatch


def kernel(eps):
    """nufftax's ES kernel parameters for tolerance ``eps`` (σ = 2)."""
    return _nufftax()[0](float(eps), 2.0)


def grid_size(n_modes, kp):
    """The even 2-3-5-smooth fine grid ``n_f ≥ max(2 n_modes, 2 w)``."""
    n = max(2 * int(n_modes), 2 * kp.nspread)
    while True:
        m = n
        for p in (2, 3, 5):
            while m % p == 0:
                m //= p
        if m == 1 and n % 2 == 0:
            return n
        n += 1


def spread(tx, ty, c, nf, kp):
    """``b[j_y, j_x] = Σ_k c_k ψ(j_x - s_x) ψ(j_y - s_y)``, ``s = n_f/2 + t n_f/2π``.

    ``tx``, ``ty`` (rad per pixel) and ``c`` are 1-D; the grid is (nf, nf).
    """
    c = np.asarray(c, dtype=np.result_type(c, 1j))
    return _nufftax()[2](tx, ty, c, nf, nf, kp)


@functools.lru_cache
def _nodes(nspread):
    z, w = onp.polynomial.legendre.leggauss(_NODES)
    return z * nspread / 2.0, w * nspread / 2.0


def psi_hat(x, nf, kp):
    """``ψ̂(2πx/n_f) = ∫ ψ(z) cos(2π x z / n_f) dz`` at real positions ``x``."""
    z, w = _nodes(kp.nspread)
    z, w = np.asarray(z), np.asarray(w)
    phi = _nufftax()[1](z, kp.beta, kp.c)
    return np.sum(w * phi * np.cos(2.0 * np.pi * x[..., None] * z / nf), -1)


def evaluate(b, mx, fx, my, fy, kp):
    """``Re Σ_k c_k e^{i(t_x x + t_y y)}`` at ``x = mx + fx``, ``y = my + fy``.

    ``mx``, ``my`` are static integer arrays (columns, rows of the result) and
    ``fx``, ``fy`` scalars (traced). Exact to the kernel's tolerance when
    ``|x|, |y| ≤ M/2`` and, for non-integer ``f``, no point's kernel wraps
    (``|t| ≤ π(1 − w/n_f)``).
    """
    nf = b.shape[-1]
    jj = np.arange(nf) - nf // 2  # J, the grid index from t = 0
    mx, my = onp.asarray(mx), onp.asarray(my)
    # Σ_J e^{2πiJm/n_f} = (-1)^m Σ_j e^{2πijm/n_f} for J = j - n_f/2 (n_f even).
    sign_x = np.asarray(1.0 - 2.0 * (mx % 2))
    sign_y = np.asarray(1.0 - 2.0 * (my % 2))
    ramp_x = np.exp(2j * np.pi * jj * fx / nf)
    ramp_y = np.exp(2j * np.pi * jj * fy / nf)
    h = np.fft.ifft(b * ramp_x, axis=1, norm="forward")[:, mx % nf]
    s = np.fft.ifft(h * ramp_y[:, None], axis=0, norm="forward")[my % nf]
    s = np.real(s) * sign_y[:, None] * sign_x[None, :]
    return s / (
        psi_hat(my + fy, nf, kp)[:, None] * psi_hat(mx + fx, nf, kp)[None, :]
    )


def margin(nf, kp):
    """Largest ``|t| / π`` whose kernel does not wrap: ``1 − w/n_f``."""
    return 1.0 - kp.nspread / nf


def check_float64():
    """Raise unless JAX's 64-bit mode is on."""
    if jax.dtypes.canonicalize_dtype(onp.float64) != onp.float64:
        raise ValueError(
            "the Toeplitz radio term needs float64, the precompute "
            "included: enable jax_enable_x64, or build and fit inside "
            "`with jax.enable_x64(True):`."
        )
