"""Radio χ² through the dirty beam (Toeplitz embedding), for one-Image scenes.

When the scene is a single [`Image`][virgil.models.Image] ``I`` (unit sum),
the χ² of [`ComplexVisData.term`][virgil.radio.ComplexVisData.term] is, for
each ddid ``d`` of observation ``e``,

``χ²_d = (C_d − 2 G_e Iᵀ D_d(δ_e) + G_e² Iᵀ B_d I) / s_d²``

with ``G_e = flux · gain_e``, ``C_d = Σ w|V|²``, the dirty image ``D_d`` at
the observation's offset ``δ_e`` and the dirty beam ``B_d`` (a Toeplitz
matrix, applied by FFT on a 2N × 2N grid). Precompute them once with
[`ComplexVisData.toeplitz`][virgil.radio.ComplexVisData.toeplitz]; an
evaluation then costs a few FFTs of the image grid, whatever the number of
visibilities. See [Radio interferometry](../radio_interferometry.md#toeplitz-chi2).

Fitted offsets (``offsets="spread"``) are an opt-in extension. The offset
ramp cancels in ``|M|²``, so ``B_d`` is offset-free, and with the stored
spread grid ``b`` the dirty image is
``D(x + δ) = Σ_l b_l e^{2πil(x+δ)/n_f} / ψ̂(x + δ)``: a phase ramp on ``b``
and the kernel transform at non-integer ``x + δ`` (a type-3 evaluation),
exact while no point's kernel wraps. It builds on type-3 NUFFT evaluation;
we know of no prior use of this combination.

!!! warning "Provisional"
    May change in any 0.4.x release without a deprecation cycle.

Source: [Fessler et al. (2005)](../references.md#fessler2005),
[nufftax](../references.md#nufftax),
[Barnett et al. (2019)](../references.md#barnett2019),
[Lee & Greengard (2005)](../references.md#leegreengard2005).
"""

from __future__ import annotations

import functools
import math

import equinox as eqx
import jax
import jax.numpy as np
import numpy as onp

from . import _nufft_spread as grid
from ._geometry import gaussian_taper, offset_phase, rotate
from ._utils import mas2rad
from .models import Image, SourceModel, System
from .radio import _warn_double_blur

__provisional__ = ("*",)

__all__ = ["ToeplitzVisData"]


# ----------------------------------------------------------------------
# Precompute
# ----------------------------------------------------------------------


def _refuse_nuisance(nuisance):
    if getattr(nuisance, "antenna_gains", False):
        raise ValueError(
            "The Toeplitz term does not support antenna gains (a per-"
            "baseline factor breaks the dirty-beam form); use "
            "ComplexVisData.term."
        )
    if nuisance.alpha:
        raise ValueError(
            "The Toeplitz term needs a fixed spectral index: "
            "CalibrationNuisance(alpha=False) or alpha=<number>."
        )


def _signature(nuisance):
    """The nuisance settings the precompute depends on."""
    return (
        nuisance.n_obs,
        nuisance.n_ddid,
        nuisance.blur_fwhm_mas,
        nuisance.nu0_hz,
        nuisance.alpha_fixed,
    )


def _accumulate(xs, nf, kp, finalize, out):
    """Spread chunks into one grid per group; ``finalize`` each group's grid
    into ``out[group]`` at its last chunk."""

    def step(carry, x):
        acc, out = carry
        tx, ty, c, g, last = x
        acc = acc + grid.spread(tx, ty, c, nf, kp)

        def flush(op):
            acc, out = op
            return np.zeros_like(acc), out.at[g].set(finalize(acc))

        return jax.lax.cond(last, flush, lambda op: op, (acc, out)), None

    acc = np.zeros((nf, nf), np.complex128)
    return jax.lax.scan(step, (acc, out), xs)[0][1]


@functools.partial(
    jax.jit, static_argnames=("npix", "n_groups", "nf_b", "nf_d", "free", "kp")
)
def _build(plan, vis, weight, *, npix, n_groups, nf_b, nf_d, free, kp):
    """``Bhat`` (rfft2 of the 2N-grid beam), ``cross`` (dirty images, or
    spread grids with ``free`` offsets), ``C = Σ w|V|²`` per group and the
    weights' log-normalization, from full-length ``vis`` and ``weight``."""
    idx, valid = plan["idx"], plan["valid"]
    tx, ty, h = plan["tx"], plan["ty"], plan["h"]
    w = np.where(valid, weight[idx], 0.0)
    v = np.where(valid, vis[idx] * plan["rot"], 0.0)
    c_sum = jax.ops.segment_sum(
        np.sum(w * (v.real**2 + v.imag**2), axis=1),
        plan["group"],
        n_groups,
    )
    logw = np.sum(
        np.where(valid, np.log(2.0 * np.pi) - np.log(weight[idx]), 0.0)
    )
    keys = (plan["group"], plan["last"])

    # The beam B(Δ) = Σ w h² cos(t·Δ) at integer lags Δ ∈ [-N, N): exact for
    # any t, as e^{itm} is periodic in t. Lag -N (never used) is zeroed so
    # that the circulant is symmetric and its transform real.
    lags = onp.arange(-npix, npix)

    def beam(acc):
        b = grid.evaluate(acc, lags, 0.0, lags, 0.0, kp)
        b = b.at[0, :].set(0.0).at[:, 0].set(0.0)
        return np.fft.rfft2(np.fft.ifftshift(b)).real

    bhat = _accumulate(
        (tx, ty, (w * h * h).astype(np.complex128)) + keys,
        nf_b,
        kp,
        beam,
        np.zeros((n_groups, 2 * npix, npix + 1)),
    )
    a = w * h * v
    if free:
        cross = _accumulate(
            (tx, ty, a) + keys,
            nf_d,
            kp,
            lambda acc: acc,
            np.zeros((n_groups, nf_d, nf_d), np.complex128),
        )
    else:
        # D at x = c - col = m + (c - N//2), m = N//2 - col an integer.
        centre = 0.5 * (npix - 1)
        a = a * np.exp(1j * (tx + ty) * (centre - npix // 2))
        m = npix // 2 - onp.arange(npix)
        cross = _accumulate(
            (tx, ty, a) + keys,
            nf_d,
            kp,
            lambda acc: grid.evaluate(acc, m, 0.0, m, 0.0, kp),
            np.zeros((n_groups, npix, npix)),
        )
    return dict(bhat=bhat, cross=cross, c_sum=c_sum, logw=logw)


class ToeplitzVisData(eqx.Module):
    """Precomputed dirty beams and images of a `ComplexVisData` (provisional).

    Build it with
    [`ComplexVisData.toeplitz`][virgil.radio.ComplexVisData.toeplitz]; its
    [`term`][virgil.radio_toeplitz.ToeplitzVisData.term] has the χ², log-
    normalization, gradient and profiled ``ŝ`` of
    [`ComplexVisData.term`][virgil.radio.ComplexVisData.term] for a scene that
    is one fixed-layout [`Image`][virgil.models.Image].

    Attributes
    ----------
    data : ComplexVisData
    mask : (N,) bool
        Points used.
    npix, pixel_scale_mas, rotation_deg, blur_fwhm_mas
        The Image layout the precompute is for.
    image_offset_mas : (float, float)
        The template's fixed ``(dra, ddec)``, folded into the visibilities.
    free_offsets : bool
        Whether ``D_d`` can be shifted (``offsets="spread"``).
    max_shift_pix : float
        Largest image-frame offset allowed, in pixels.
    nyquist_fraction : float
        ``max |t| / π`` of the used points (``t`` = 2π u × pixel, rad).
    margin : float or None
        The largest ``nyquist_fraction`` that free offsets allow,
        ``1 − w/n_f``.
    arrays : dict
        ``bhat``, ``cross``, ``c_sum`` (``Σ w|V|²`` per ddid), ``logw``,
        ``count`` (points per ddid).
    """

    data: object
    mask: onp.ndarray
    plan: dict
    arrays: dict
    npix: int = eqx.field(static=True)
    pixel_scale_mas: float = eqx.field(static=True)
    rotation_deg: float = eqx.field(static=True)
    blur_fwhm_mas: float | None = eqx.field(static=True)
    image_offset_mas: tuple = eqx.field(static=True)
    free_offsets: bool = eqx.field(static=True)
    max_shift_pix: float = eqx.field(static=True)
    nyquist_fraction: float = eqx.field(static=True)
    margin: float | None = eqx.field(static=True)
    signature: tuple = eqx.field(static=True)
    kp: object = eqx.field(static=True)
    sizes: tuple = eqx.field(static=True)

    def __init__(
        self,
        data,
        image,
        nuisance,
        mask=None,
        offsets="none",
        eps=1e-14,
        max_offset_mas=None,
        chunk_size=65536,
    ):
        grid.check_float64()
        _refuse_nuisance(nuisance)
        if offsets not in ("none", "spread"):
            raise ValueError("offsets must be 'none' or 'spread'.")
        free = offsets == "spread"
        if bool(nuisance.offsets) != free:
            raise ValueError(
                "nuisance fits offsets: pass offsets='spread' (an opt-in "
                "extension) or use CalibrationNuisance(offsets=False)."
                if nuisance.offsets
                else "offsets='spread' needs a nuisance that fits offsets."
            )
        if not isinstance(image, Image):
            raise TypeError(
                f"image must be an Image (the layout), not {type(image).__name__}."
            )
        nrow, ncol = image.eta.shape
        if nrow != ncol:
            raise ValueError(f"the Image must be square, not {nrow} x {ncol}.")
        if onp.ndim(image.dra) or onp.ndim(image.ddec):
            raise ValueError("the Image's dra and ddec must be numbers.")
        dra, ddec = float(image.dra), float(image.ddec)
        n = len(data)
        mask = onp.ones(n, bool) if mask is None else onp.asarray(mask, bool)
        if mask.shape != (n,):
            raise ValueError(f"mask has shape {mask.shape}, not ({n},).")
        if not mask.any():
            raise ValueError("the mask leaves no points.")
        kp = grid.kernel(eps)
        npix, pix = int(nrow), image.pixel_scale_mas
        bound = 0.0
        if free:
            # The largest image-frame shift (pixels) of the offsets' prior box.
            shift_mas = max_offset_mas
            if shift_mas is None:
                shift_mas = getattr(nuisance, "_max_offset_mas", None)
            if shift_mas is None:
                free_obs = onp.arange(nuisance.n_obs) != nuisance.ref_obs
                theta = nuisance.offset_beams * 2.0 * onp.pi
                shift_mas = float(
                    onp.max(theta * nuisance.offset_unit_mas[free_obs])
                )
            angle = onp.deg2rad(image.rotation_deg)
            spread = abs(onp.cos(angle)) + abs(onp.sin(angle))
            bound = float(shift_mas) * spread / pix

        used = onp.flatnonzero(mask)
        u = onp.asarray(data.u, onp.float64)[used]
        v = onp.asarray(data.v, onp.float64)[used]
        freq = onp.asarray(data.freq, onp.float64)[used]
        uu, vv = rotate(u, v, -image.rotation_deg)
        scale = 2.0 * onp.pi * mas2rad * pix
        tx, ty = scale * uu, scale * vv
        h = (freq / nuisance.nu0_hz) ** nuisance.alpha_fixed
        h = h * onp.asarray(nuisance._blur(u, v))
        if image.blur_fwhm_mas is not None:
            h = h * onp.asarray(
                gaussian_taper(u * u + v * v, image.blur_fwhm_mas)
            )
        tau = float(onp.max(onp.maximum(onp.abs(tx), onp.abs(ty))) / onp.pi)
        # A fixed image offset is a phase on the model; move it to the data.
        rot = onp.conj(onp.asarray(offset_phase(u, v, dra, ddec)))

        n_groups = len(data.ddid_ids)
        group = onp.asarray(data.ddid, onp.int64)[used]
        # Points sorted by group into chunks that never straddle a group.
        order = onp.argsort(group, kind="stable")
        counts = onp.bincount(group, minlength=n_groups)
        chunk = max(1, min(int(chunk_size), int(counts.max())))
        idx_l, valid_l, group_l, last_l = [], [], [], []
        start = 0
        for g, count in enumerate(counts):
            members = order[start : start + count]
            start += count
            n_chunks = -(-int(count) // chunk)
            pad = n_chunks * chunk - count
            idx_l.append(onp.concatenate([members, onp.zeros(pad, int)]))
            valid_l.append(onp.arange(n_chunks * chunk) < count)
            group_l += [g] * n_chunks
            last_l += [False] * (n_chunks - 1) + [True] * min(n_chunks, 1)
        idx = onp.concatenate(idx_l).reshape(-1, chunk)
        valid = onp.concatenate(valid_l).reshape(-1, chunk)
        chunk_group = onp.asarray(group_l, dtype=onp.int32)
        last = onp.asarray(last_l, dtype=bool)
        nf_b = grid.grid_size(2 * npix, kp)
        nf_d = grid.grid_size(npix + 2 * math.ceil(bound), kp)
        margin = grid.margin(nf_d, kp) if free else None
        if free and tau > margin:
            raise ValueError(
                f"Baselines reach {tau:.3f} of the pixel Nyquist limit, past "
                f"the {margin:.3f} that free offsets allow (the spreading "
                "kernel would wrap). Use smaller pixels, mask the longest "
                "baselines, or fix the offsets."
            )

        def chunked(x):
            return np.asarray(onp.where(valid, x[idx], 0))

        self.plan = dict(
            idx=np.asarray(used[idx]),
            valid=np.asarray(valid),
            group=np.asarray(chunk_group),
            last=np.asarray(last),
            tx=chunked(tx),
            ty=chunked(ty),
            h=chunked(h),
            rot=chunked(rot),
        )
        self.data = data
        self.mask = mask
        self.npix = npix
        self.pixel_scale_mas = float(pix)
        self.rotation_deg = float(image.rotation_deg)
        self.blur_fwhm_mas = image.blur_fwhm_mas
        self.image_offset_mas = (dra, ddec)
        self.free_offsets = free
        self.max_shift_pix = float(bound)
        self.nyquist_fraction = tau
        self.margin = margin
        self.signature = _signature(nuisance)
        self.kp = kp
        self.sizes = (n_groups, nf_b, nf_d)
        count = onp.bincount(group, minlength=n_groups).astype(onp.float64)
        self.arrays = dict(
            self._compute(data.vis, data.weight), count=np.asarray(count)
        )

    def _compute(self, vis, weight):
        n_groups, nf_b, nf_d = self.sizes
        return _build(
            self.plan,
            np.asarray(vis, np.complex128),
            np.asarray(weight, np.float64),
            npix=self.npix,
            n_groups=n_groups,
            nf_b=nf_b,
            nf_d=nf_d,
            free=self.free_offsets,
            kp=self.kp,
        )

    @property
    def layout(self):
        """``(shape, pixel_scale_mas, rotation_deg, blur_fwhm_mas)``."""
        return (
            (self.npix, self.npix),
            self.pixel_scale_mas,
            self.rotation_deg,
            self.blur_fwhm_mas,
        )

    def rebuild(self, vis, weight):
        """The same precompute for new full-length ``vis`` and ``weight``
        (jitted: no recompilation)."""
        grid.check_float64()
        n = len(self.data)
        vis, weight = np.asarray(vis), np.asarray(weight)
        if vis.shape != (n,) or weight.shape != (n,):
            raise ValueError(f"vis and weight must have shape ({n},).")
        weight = eqx.error_if(
            weight,
            ~np.all(np.isfinite(weight) & (weight > 0)),
            "weight must be finite and positive (as for from_arrays); drop "
            "points through the precompute's mask instead.",
        )
        data = eqx.tree_at(
            lambda d: (d.vis, d.weight), self.data, (vis, weight)
        )
        arrays = dict(self._compute(vis, weight), count=self.arrays["count"])
        return eqx.tree_at(lambda t: (t.data, t.arrays), self, (data, arrays))

    def term(self, scene, nuisance, sigma=None):
        """A likelihood term for [`fit`][virgil.fitting.fit]'s ``likelihoods``.

        As [`ComplexVisData.term`][virgil.radio.ComplexVisData.term] (same
        loss, gradient and ``derived``), but with no residual vector: fit
        with ``method="lbfgs"`` or ``"adam"``. ``scene(values)`` must be one
        [`Image`][virgil.models.Image] of this layout (alone, or the only
        part of a [`System`][virgil.models.System]) at the template's fixed
        ``dra``, ``ddec``; the observations' offsets are ``cal.offset``.
        """
        if isinstance(scene, SourceModel):
            raise TypeError(
                "scene must be a function of the fitted values, e.g. "
                "`lambda values: model`, not a model instance."
            )
        _refuse_nuisance(nuisance)
        if _signature(nuisance) != self.signature:
            raise ValueError(
                "nuisance differs from the one of the precompute (n_obs, "
                "n_ddid, blur_fwhm_mas, nu0_hz or alpha); rebuild it."
            )
        if nuisance.offsets and not self.free_offsets:
            raise ValueError(
                "nuisance fits offsets but the precompute was built without "
                "them; build it with offsets='spread'."
            )
        return _ToeplitzTerm(self, scene, nuisance, sigma)

    def direct_term(self, scene, nuisance, **kwargs):
        """[`ComplexVisData.term`][virgil.radio.ComplexVisData.term] of the
        same points, for checks."""
        return self.data.term(
            scene, nuisance, mask=self.mask, chi2="direct", **kwargs
        )

    def chi2_by_ddid(self, scene, values, nuisance):
        """``(chi2, n)`` per ddid, as
        [`ComplexVisData.chi2_by_ddid`][virgil.radio.ComplexVisData.chi2_by_ddid]."""
        term = self.term(_as_function(scene), nuisance)
        chi2 = term.chi2_by_ddid(values)
        return onp.asarray(chi2), onp.asarray(self.arrays["count"], int)

    def crosscheck(self, scene, values, nuisance, chunk_size=10_000_000):
        """The χ² per ddid and the loss both ways, and their largest
        relative difference.

        Returns a dict: ``chi2`` and ``chi2_direct`` per ddid, ``loss`` and
        ``loss_direct`` (``½χ² + log_norm``), ``max_rel``. ``chunk_size`` is
        the direct term's.
        """
        scene = _as_function(scene)
        chi2 = self.chi2_by_ddid(scene, values, nuisance)[0]
        direct = self.data.chi2_by_ddid(
            scene, values, nuisance, self.mask, chunk_size=chunk_size
        )[0]
        term = self.term(scene, nuisance)
        reduced, log_norm = term.chi2_and_log_norm(values)
        loss = float(0.5 * reduced + log_norm + term.loss_constant)
        r, n = self.direct_term(
            scene, nuisance, chunk_size=chunk_size
        ).residuals_and_log_norm(values)
        loss_direct = float(0.5 * np.sum(r**2) + n)
        rel = onp.abs(chi2 - direct) / onp.maximum(onp.abs(direct), 1e-300)
        rel_loss = abs(loss - loss_direct) / max(abs(loss_direct), 1e-300)
        return dict(
            chi2=chi2,
            chi2_direct=direct,
            loss=loss,
            loss_direct=loss_direct,
            max_rel=float(max(rel.max(), rel_loss)),
        )


def _as_function(scene):
    if isinstance(scene, SourceModel):
        model = scene
        return lambda values: model  # noqa: E731
    return scene


# ----------------------------------------------------------------------
# The likelihood term
# ----------------------------------------------------------------------


def _single_image(scene):
    """``(image, offsets)``: the one Image of ``scene`` and every offset leaf
    on the way to it (their sum must be the template's)."""
    offsets = []
    model = scene
    while isinstance(model, System):
        parts = model.resolved().parts
        if len(parts) != 1:
            raise ValueError(
                "The Toeplitz term needs a scene of one Image; this System "
                f"has {len(parts)} parts. Use ComplexVisData.term."
            )
        offsets += [model.dra, model.ddec]
        model = parts[0]
    if not isinstance(model, Image):
        raise TypeError(
            "The Toeplitz term needs a scene of one Image, not "
            f"{type(model).__name__}. Use ComplexVisData.term."
        )
    return model, offsets + [model.dra, model.ddec]


class _ToeplitzTerm(eqx.Module):
    """χ² and log-normalization of a `ToeplitzVisData` with calibration
    nuisances, less constants (``chi2_constant``, ``log_norm_constant``)."""

    arrays: dict
    sigma_fixed: object
    scene: object = eqx.field(static=True)
    nuisance: object = eqx.field(static=True)
    layout: tuple = eqx.field(static=True)
    image_offset_mas: tuple = eqx.field(static=True)
    ddid_obs: tuple = eqx.field(static=True)
    free_offsets: bool = eqx.field(static=True)
    max_shift_pix: float = eqx.field(static=True)
    kp: object = eqx.field(static=True)
    n_data: int = eqx.field(static=True)

    def __init__(self, tdata, scene, nuisance, sigma=None):
        if sigma is not None:
            if nuisance.sigmas != "profile":
                raise ValueError("sigma= needs sigmas='profile'.")
            sigma = np.asarray(sigma, dtype=np.float64)
            if sigma.shape != (nuisance.n_ddid,):
                raise ValueError(
                    f"sigma must have shape ({nuisance.n_ddid},)."
                )
        self.sigma_fixed = sigma
        self.arrays = tdata.arrays
        self.scene = scene
        self.nuisance = nuisance
        self.layout = tdata.layout
        self.image_offset_mas = tdata.image_offset_mas
        self.ddid_obs = tuple(int(e) for e in tdata.data.ddid_obs)
        self.free_offsets = tdata.free_offsets
        self.max_shift_pix = tdata.max_shift_pix
        self.kp = tdata.kp
        # As ComplexVisData.term: two numbers per point, masked or not.
        self.n_data = 2 * len(tdata.data)

    @property
    def has_log_norm(self):
        return bool(self.nuisance.sigmas)

    @property
    def _profiled(self):
        return self.nuisance.sigmas == "profile" and self.sigma_fixed is None

    # -- pieces ---------------------------------------------------------

    def _image(self, values):
        scene = self.scene(values)
        _warn_double_blur(scene, self.nuisance)
        image, offsets = _single_image(scene)
        layout = (
            image.eta.shape,
            image.pixel_scale_mas,
            image.rotation_deg,
            image.blur_fwhm_mas,
        )
        if layout != self.layout:
            raise ValueError(
                f"The scene's Image (shape, pixel_scale_mas, rotation_deg, "
                f"blur_fwhm_mas) = {layout} differs from the precompute's "
                f"{self.layout}."
            )
        dra, ddec = sum(offsets[0::2]), sum(offsets[1::2])
        moved = np.any(np.asarray(dra) != self.image_offset_mas[0]) | np.any(
            np.asarray(ddec) != self.image_offset_mas[1]
        )
        return eqx.error_if(
            image.brightness,
            moved,
            "The Toeplitz term needs the scene's Image (with any System "
            "around it) at the template's offset; fit the offsets as "
            "cal.offset.",
        )

    def _shift(self, offset):
        """Image-frame shifts (pixels) of the observations, bound-checked."""
        if not self.free_offsets:
            return None
        _, pixel_scale, rotation, _ = self.layout
        x, y = rotate(offset[:, 0], offset[:, 1], -rotation)
        shift = np.stack([x, y], axis=-1) / pixel_scale
        return eqx.error_if(
            shift,
            np.any(np.abs(shift) > self.max_shift_pix),
            "An offset is beyond the bound of the Toeplitz precompute "
            "(the offsets' prior box); build it with a larger "
            "max_offset_mas.",
        )

    def _dirty(self, b, shift):
        """``D(δ)`` on the pixel grid from a spread grid ``b``."""
        npix = self.layout[0][0]
        centre = 0.5 * (npix - 1)
        m = math.floor(centre) - onp.arange(npix)
        f = centre - math.floor(centre)
        return grid.evaluate(b, m, f + shift[0], m, f + shift[1], self.kp)

    def _quadratic(self, image):
        """``Iᵀ B_d I`` per ddid, by Parseval on the 2N grid."""
        npix = image.shape[0]
        f = np.fft.rfft2(image, s=(2 * npix, 2 * npix))
        weights = onp.full(npix + 1, 2.0)
        weights[0] = weights[-1] = 1.0
        power = (f.real**2 + f.imag**2) * weights / (2 * npix) ** 2
        return np.einsum("dij,ij->d", self.arrays["bhat"], power)

    def _cross(self, image, shift, k=None):
        """``Iᵀ D_d`` per ddid, or ``Σ_d k_d Iᵀ D_d`` (one map per obs)."""
        cross = self.arrays["cross"]
        if shift is None:
            x = np.einsum("dij,ij->d", cross, image)
            return x if k is None else np.sum(k * x)
        obs = onp.asarray(self.ddid_obs)
        if k is None:
            return jax.lax.map(
                lambda a: np.sum(self._dirty(*a) * image), (cross, shift[obs])
            )
        onehot = obs[None, :] == onp.arange(shift.shape[0])[:, None]
        grids = np.einsum("ed,dij->eij", np.where(onehot, k, 0.0), cross)
        return np.sum(
            jax.lax.map(
                lambda a: np.sum(self._dirty(*a) * image), (grids, shift)
            )
        )

    def curvature_params(self, values):
        """``(I, shift, g, σ)``: the brightness, image-frame shifts (pixels,
        or ``None``), ``G`` per ddid and σ, of which the loss is a function
        (``fit``'s ``precondition="diag"`` takes its Gauss–Newton curvature
        through them)."""
        grid.check_float64()
        leaves = jax.tree.leaves((self.arrays, self.sigma_fixed))
        if any(x.dtype not in (np.float64, np.complex128) for x in leaves):
            raise ValueError(
                "The Toeplitz term needs float64 throughout (its constant "
                "Σw|V|² cancels to small differences); fit with "
                "dtype='float64'."
            )
        flux, gain, offset, sigma, _ = self.nuisance._unpack(values)
        image = self._image(values)
        g = (flux * gain)[onp.asarray(self.ddid_obs)]
        return image, self._shift(offset), g, sigma

    def _deltas(self, params):
        """``χ²_d s_d² − C_d`` per ddid."""
        image, shift, g, _ = params
        x = self._cross(image, shift)
        return g * (g * self._quadratic(image) - 2.0 * x)

    def _profile(self, delta):
        """``ŝ_d² = (C_d + Δ_d) / 2N_d`` (1 without points), held."""
        c_sum, count = self.arrays["c_sum"], self.arrays["count"]
        s2 = (c_sum + delta) / (2.0 * np.maximum(count, 1.0))
        floor = np.sqrt(np.finfo(s2.dtype).tiny)
        s2 = np.where(count > 0, np.maximum(s2, floor), 1.0)
        return jax.lax.stop_gradient(s2)

    def _reduced(self, params):
        c_sum, count = self.arrays["c_sum"], self.arrays["count"]
        if self._profiled:
            delta = self._deltas(params)
            r = (delta - jax.lax.stop_gradient(delta)) / self._profile(delta)
            ok = (count > 0) & (c_sum > 0)
            ratio = delta / np.where(ok, c_sum, 1.0)
            tiny = np.finfo(c_sum.dtype).tiny
            log1p = np.where(
                ratio > -0.5,
                np.log1p(np.maximum(ratio, -0.5)),
                np.log(np.maximum(1.0 + ratio, tiny)),
            )
            log_norm = np.sum(
                np.where(ok, count * jax.lax.stop_gradient(log1p), 0.0)
            )
            return np.sum(r), log_norm
        sigma = params[3] if self.sigma_fixed is None else self.sigma_fixed
        image, shift, g, _ = params
        c = 1.0 / sigma**2
        q = np.sum(c * g**2 * self._quadratic(image))
        reduced = q - 2.0 * self._cross(image, shift, c * g)
        if self.nuisance.sigmas is True:
            chi2 = np.sum(c_sum / sigma**2) + reduced
            return chi2, 2.0 * np.sum(count * np.log(sigma))
        return reduced, np.zeros((), reduced.dtype)

    # -- the protocol of fit ------------------------------------------------

    def chi2_and_log_norm(self, values):
        """``(χ², log_norm)`` less ``chi2_constant`` and ``log_norm_constant``."""
        return self._reduced(self.curvature_params(values))

    def curvature_loss(self, params):
        """``½χ²`` (less constants) of ``curvature_params``."""
        chi2, _ = self._reduced(params)
        return 0.5 * chi2

    @property
    def chi2_constant(self):
        """The part of χ² that ``chi2_and_log_norm`` leaves out."""
        c_sum, count = self.arrays["c_sum"], self.arrays["count"]
        sigmas = self.nuisance.sigmas
        if self._profiled:
            return np.sum(np.where(count > 0, 2.0 * count, 0.0))
        if sigmas == "profile":
            return np.sum(c_sum / self.sigma_fixed**2)
        return np.sum(c_sum) if sigmas is False else np.zeros(())

    @property
    def log_norm_constant(self):
        """The part of ``log_norm`` that ``chi2_and_log_norm`` leaves out."""
        c_sum, count = self.arrays["c_sum"], self.arrays["count"]
        if self._profiled:
            ok = (count > 0) & (c_sum > 0)
            per = count * np.log(
                np.where(ok, c_sum, 1.0) / (2.0 * np.maximum(count, 1.0))
            )
            return np.sum(np.where(ok, per, 0.0))
        if self.nuisance.sigmas == "profile":
            return 2.0 * np.sum(count * np.log(self.sigma_fixed))
        return np.zeros(())

    @property
    def loss_constant(self):
        """``½ chi2_constant + log_norm_constant``."""
        return 0.5 * self.chi2_constant + self.log_norm_constant

    def __call__(self, values):
        raise TypeError(
            "The Toeplitz term has no residual vector (fit with "
            "method='lbfgs' or 'adam'; for residuals or held-out scores use "
            "ToeplitzVisData.direct_term)."
        )

    # -- results ------------------------------------------------------------

    def sigma_hat(self, values):
        """The error scales ``s_d``: ``ŝ`` if profiled, the fixed ``sigma``,
        the fitted ``cal.sigma``, or ones."""
        if self._profiled:
            delta = self._deltas(self.curvature_params(values))
            return np.sqrt(self._profile(delta))
        if self.sigma_fixed is not None:
            return self.sigma_fixed
        return self.nuisance._unpack(values)[3]

    def derived(self, values):
        """Quantities of the fit that are not sites: the profiled ``ŝ``."""
        if self.nuisance.sigmas != "profile":
            return {}
        return {self.nuisance._site("sigma"): self.sigma_hat(values)}

    def chi2_by_ddid(self, values):
        """``(C_d + Δ_d) / s_d²`` per ddid."""
        params = self.curvature_params(values)
        delta = self._deltas(params)
        if self._profiled:
            s2 = self._profile(delta)
        else:
            s2 = (
                params[3] if self.sigma_fixed is None else self.sigma_fixed
            ) ** 2
        return (self.arrays["c_sum"] + delta) / s2

    def loglike(self, values):
        """Normalized Gaussian log density of the unmasked points."""
        chi2, log_norm = self.chi2_and_log_norm(values)
        chi2 = chi2 + self.chi2_constant
        log_norm = log_norm + self.log_norm_constant
        return -0.5 * chi2 - log_norm - self.arrays["logw"]
