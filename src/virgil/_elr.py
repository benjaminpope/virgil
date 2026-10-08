"""The ELR11 rotating-star model: Roche shape and gravity darkening.

Implements the model of Espinosa Lara & Rieutord (2011, A&A 533, A43,
"ELR11") for the shape and gravity darkening of a rigidly rotating star,
which needs no free gravity-darkening exponent beta.

Ported from Shashank Dholakia's jax-interferometry (core/ELR.py,
core/utils.py, commit 70689ed), who ported the equations from Aaron
Dotter's GDit. Function names, structure and ELR11 equation references are
his; the changes made in this port are:

- ``jaxopt.Bisection`` is replaced by ``optimistix`` bisection, with brackets
  and tolerances taken from ``np.finfo`` so it runs in float32 and float64.
- ``eq30`` is multiplied through by ``omega**2`` so that ``omega = 0`` is
  valid and differentiable; the root is unchanged.
- The pole and equator special cases of ``solve_ELR`` use the double-``where``
  trick, so the branch not taken yields neither NaN values nor NaN gradients.
- No global ``jax_enable_x64``. The vectorised solver ``solve_ELR_vec`` is
  jitted once at module level, because run eagerly its bisection loops are
  slow; inside an outer jit that is a no-op.
- The mesh is built once per ``n_lat`` in pure NumPy and cached, instead of in
  a class constructor (``mesh``); the body of ``ELR_Model.__call__`` is split
  into ``surface`` and ``visibilities``.
- ``compute_DFTM1`` and ``apply_DFTM1`` become ``visibilities``, taking
  spatial frequencies in cycles per radian, with ``Precision.HIGHEST``.
- ``revolve`` and ``triangle_area`` (unused) and the plotting code are dropped.

Conventions of this module are Dholakia's: ``inc`` is in radians with
``inc = 0`` equator-on, ``obl`` is in radians, ``r_eq`` is in mas, and the
axes are x, y, z with y the spin axis before rotation and z towards the
observer. The public model class maps virgil conventions onto these.
"""

import functools
from typing import NamedTuple

import jax
import jax.numpy as np
import numpy as onp
import optimistix as optx

from ._utils import mas2rad


# ELR11 constants
f23 = 2.0 / 3.0


def _cos_plus_log_tan_half(angle):
    """``cos(a) + log(tan(a / 2))``, accurate near ``a = pi / 2`` too.

    Near the equator the two terms cancel to ``-cos(a)**3 / 3``, which in
    float32 cost ~1% in the flux of the rings next to it. Since
    ``log(tan(a / 2)) = -atanh(cos(a))``, the sum is ``-(atanh(c) - c)`` with
    ``c = cos(a)``, summed as a series where ``c`` is small. Near the pole,
    where ``c -> 1`` and ``atanh`` overflows, the original form is used.
    """
    c = np.cos(angle)
    near_equator = np.abs(c) < 0.3
    # double-where: each branch only ever sees inputs where it is finite
    c_eq = np.where(near_equator, c, 0.0)
    c2 = c_eq * c_eq
    series = np.zeros_like(c_eq)
    for k in range(15, 0, -1):  # atanh(c) - c = sum_k c**(2k+1) / (2k+1)
        series = c2 * (1.0 / (2 * k + 1) + series)
    series = c_eq * series
    angle_pole = np.where(near_equator, 0.25 * np.pi, angle)
    direct = np.cos(angle_pole) + np.log(np.tan(0.5 * angle_pole))
    return np.where(near_equator, -series, direct)


# ELR11 equations
# gives the value of phi
def eq24(phi, theta, omega, rtw):
    # cos(x) + log(tan(x / 2)) is evaluated without cancellation near the
    # equator (see _cos_plus_log_tan_half); the equation is unchanged.
    tau = (
        np.power(omega, 2) * np.power(rtw * np.cos(theta), 3)
    ) / 3.0 + _cos_plus_log_tan_half(theta)
    return _cos_plus_log_tan_half(phi) - tau


# solve for rtw given omega
def eq30(rtw, theta, omega):
    # Multiplied through by omega**2 (ELR11 eq. 30 divided by it): same root,
    # but no 1/omega**2, so omega = 0 is valid (root rtw = 1) and
    # differentiable.
    return (1.0 / rtw - 1.0) + 0.5 * omega**2 * (
        (rtw * np.sin(theta)) ** 2 - 1.0
    )


# ratio of equatorial to polar Teff
def eq32(omega):
    return (
        np.sqrt(2.0 / (2.0 + omega**2))
        * (1.0 - omega**2) ** (1.0 / 12.0)
        * np.exp(-(4.0 / 3.0) * omega**2 / (2 + omega**2) ** 3)
    )


def _bisect(fn, lower, upper, args, dtype, flip):
    """Bracketed root of ``fn(y, args)`` by optimistix bisection.

    Tolerances come from the dtype (his 1e-14 is impossible in float32), and
    the step count is enough to halve the bracket down to that tolerance.
    ``flip`` (True if ``fn(lower) > fn(upper)``) is passed explicitly: the
    default 'detect' raises when the root sits exactly on a bracket end
    (eq. 30 at omega = 0 has ``fn(upper) == 0``), even with ``throw=False``.
    """
    eps = np.finfo(dtype).eps
    # Fixed iteration count that covers any bracket of width <~ 2: log2(1/eps)
    # halvings, plus slack.
    max_steps = int(onp.ceil(-onp.log2(float(eps)))) + 8
    lower = np.asarray(lower, dtype)
    upper = np.asarray(upper, dtype)
    y0 = 0.5 * (lower + upper)
    sol = optx.root_find(
        fn,
        optx.Bisection(rtol=4 * eps, atol=4 * eps, flip=flip),
        y0,
        args=args,
        options=dict(lower=lower, upper=upper),
        max_steps=max_steps,
        throw=False,  # never raise under jit; tolerance is checked in tests
    )
    return sol.value


def solve_ELR(omega, theta):  # eq.26, 27, 28; solve the ELR11 equations
    """
    Takes a float omega where 0<=omega<1
    and a single value theta, or polar angles in radians
    calculates r~, Teff_ratio, and Flux_ratio
    Can be vmapped to solve for an array of thetas (done below)
    """
    dtype = np.result_type(float, omega, theta)
    omega = np.asarray(omega, dtype)
    theta = np.asarray(theta, dtype)
    pi = np.asarray(onp.pi, dtype)
    eps = np.finfo(dtype).eps

    # theta is the polar angle.
    # this routine calculates values for 0 <= theta <= pi/2
    # everything else is mapped into this interval by symmetry
    # theta = 0 at the pole(s)
    # theta = pi/2 at the equator
    # -pi/2 < theta < 0: theta -> abs(theta)
    #  pi/2 > theta > pi: theta -> pi - theta
    theta = np.where(
        np.logical_and(theta > pi / 2, theta <= pi),  # if
        pi - theta,  # then
        theta,  # else
    )

    theta = np.where(
        np.logical_and(theta >= -pi / 2, theta < 0),
        np.abs(theta),
        theta,
    )

    # first we solve equation 30 for rtw
    # Bracket: the Roche radius R/R_eq lies in [2/3, 1] (finite, unlike his
    # [0, 1] which evaluates 1/0); 0.5 is a safe lower bound.
    rtw = _bisect(
        lambda r, args: eq30(r, *args),
        0.5,
        1.0,
        (theta, omega),
        dtype,
        flip=True,  # eq30 is positive at 0.5 and non-positive at 1
    )

    # the following are special solutions for extreme values of theta
    w2r3 = omega**2 * rtw**3

    # Double-where: eq24 is singular at theta = 0 (log tan 0) and degenerate
    # at theta = pi/2, so the generic branch is evaluated at a safe theta
    # wherever a special case applies. Otherwise the unused branch could give
    # NaN values or NaN gradients.
    is_pole = theta == 0
    is_equator = theta == 0.5 * pi
    special = np.logical_or(is_pole, is_equator)
    safe_theta = np.where(special, pi / 4, theta)
    safe_rtw = np.where(special, 1.0, rtw)

    # phi lies in (0, pi/2). In float32 his upper bound pi/2 - 1e-10 rounds
    # to above pi/2 (tan < 0, log NaN), so use dtype-aware interior bounds:
    # a few ulps inside pi/2, and the smallest bound with log(tan(phi/2))
    # still finite at the bottom.
    phi_lo = np.sqrt(np.finfo(dtype).tiny)
    phi_hi = 0.5 * pi * (1.0 - 8.0 * eps)
    phi = _bisect(
        lambda p, args: eq24(p, *args),
        phi_lo,
        phi_hi,
        (safe_theta, omega, safe_rtw),
        dtype,
        flip=False,  # eq24 -> -inf as phi -> 0 and is positive near pi/2
    )
    Fw_generic = (np.tan(phi) / np.tan(safe_theta)) ** 2

    Fw = np.where(
        is_pole,  # if
        np.exp(f23 * w2r3),  # then
        np.where(
            is_equator,  # elsif
            (1.0 - w2r3) ** (-f23),  # then
            Fw_generic,
        ),
    )

    # equation 31 and similar for Fw
    term1 = rtw ** (-4)
    term2 = omega**4 * (rtw * np.sin(theta)) ** 2
    term3 = -2 * (omega * np.sin(theta)) ** 2 / rtw
    gterm = np.sqrt(term1 + term2 + term3)
    Flux_ratio = Fw * gterm
    Teff_ratio = Flux_ratio**0.25
    return rtw, Teff_ratio, Flux_ratio


# Jitted once here: run eagerly (tests, notebooks), the bisection loops
# dispatch op by op and take ~0.5 s per call instead of ~1 ms. Inside an
# outer jit this is a no-op.
solve_ELR_vec = jax.jit(jax.vmap(solve_ELR, in_axes=[None, 0]))


# === utils.py ===


def closest_polygon(thetas):
    # NumPy, not np: it only ever runs on the static mesh latitudes.
    thetas = onp.asarray(thetas, dtype=float)
    with onp.errstate(invalid="ignore", divide="ignore"):
        arg = onp.pi / (2 * (len(thetas)) * onp.sin(thetas))
        # arcsin is NaN for arg > 1 (rings near the poles); his np version
        # cast NaN to int 0 and then set 0 -> 1. Made explicit here.
        valid = arg <= 1
        n = onp.rint(onp.pi / onp.arcsin(onp.where(valid, arg, 1.0)))
    n = onp.where(valid, n, 1).astype(int)
    n = onp.where(n == 0, 1, n)
    return n


def spherical_to_cartesian(r, theta, phi):
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.cos(theta)
    z = r * np.sin(theta) * np.sin(phi)
    return np.array([x, y, z])


def rotate_point_cloud(points, inclination, obliquity):
    # define the rotation matrices
    R_x = np.array(
        [
            [1, 0, 0],
            [0, np.cos(inclination), -np.sin(inclination)],
            [0, np.sin(inclination), np.cos(inclination)],
        ]
    )
    R_z = np.array(
        [
            [np.cos(obliquity), -np.sin(obliquity), 0],
            [np.sin(obliquity), np.cos(obliquity), 0],
            [0, 0, 1],
        ]
    )
    # rotate the point cloud
    # precision=HIGHEST: GPU default matmul precision is TF32 (AGENTS.md)
    hi = jax.lax.Precision.HIGHEST
    points_rotated = np.dot(points, R_x, precision=hi)
    points_rotated = np.dot(points_rotated, R_z, precision=hi)
    return points_rotated


def triangle_normals(points, triangulation):
    a = points[triangulation[:, 0], :]
    b = points[triangulation[:, 1], :]
    c = points[triangulation[:, 2], :]
    # compute the normal vectors
    normals = np.cross(b - a, c - a)
    # compute the center of the triangle
    center = (a + b + c) / 3
    # reverse the normal vector if the dot product is negative
    normals = (
        normals * np.sign(np.sum(normals * center, axis=1))[:, np.newaxis]
    )
    return normals


def barycenter(points, triangulation):
    # get the coordinates of the triangle vertices
    x = points[triangulation, 0]
    y = points[triangulation, 1]
    z = points[triangulation, 2]
    # compute the barycenter coordinates
    x_barycenter = np.mean(x, axis=1)
    y_barycenter = np.mean(y, axis=1)
    z_barycenter = np.mean(z, axis=1)
    # stack the barycenter coordinates into an array
    barycenters = np.stack((x_barycenter, y_barycenter, z_barycenter), axis=1)
    return barycenters


# === ELR_Model ===


class Mesh(NamedTuple):
    """Static triangulated unit-sphere mesh (read-only NumPy arrays)."""

    thetas: onp.ndarray  # (n_lat,) polar angles of the rings
    n: onp.ndarray  # (n_lat,) vertices per ring
    phi: onp.ndarray  # (n_vertices,) azimuths
    triangulation: onp.ndarray  # (n_triangles, 3) vertex indices


@functools.lru_cache(maxsize=None)
def mesh(n_lat):
    """Mirror of his ``ELR_Model.__init__`` mesh construction.

    Pure NumPy, with no np anywhere: this may first be called while jax is
    tracing, and caching a tracer would leak it.
    """
    from scipy.spatial import ConvexHull  # lazy: only needed to build

    n_lat = int(n_lat)
    tol = 1e-4
    thetas = onp.linspace(tol, onp.pi - tol, n_lat)
    ns = closest_polygon(thetas)
    phi = onp.concatenate(
        [onp.linspace(tol, 2 * onp.pi - tol, n, endpoint=False) for n in ns]
    )
    theta = thetas.repeat(ns)
    points = onp.stack(
        [
            onp.sin(theta) * onp.cos(phi),
            onp.cos(theta),
            onp.sin(theta) * onp.sin(phi),
        ],
        axis=1,
    )
    triangulation = ConvexHull(points).simplices
    # int32 indices: JAX caches the converted copy of a NumPy array by
    # identity whatever the x64 mode it was made in (JAX 0.10), and int32
    # is the same in both modes. The floats are cast at use.
    out = Mesh(
        thetas, ns.astype(onp.int32), phi, triangulation.astype(onp.int32)
    )
    for a in out:
        a.setflags(write=False)
    return out


def surface(omega, r_eq, inc, obl, n_lat=32, return_mesh=False):
    """Body of his ``ELR_Model.__call__`` up to the Fourier transform.

    Returns ``(x, y, weight, teff_ratio)`` per triangle: barycentre sky
    coordinates in mas, the visible-flux weight, and the mean ``Teff / Teff_pole``
    (he computes this in ``plot``).

    With ``return_mesh=True`` a fifth item is returned, for plotting:
    ``(points_rotated, triangulation, cosine, intensity)``, the rotated
    vertices (n_vertices, 3), triangle vertex indices, the (unnormalised)
    z component of each normal (visible where positive) and the flux ratio
    of each triangle.
    """
    m = mesh(n_lat)
    dtype = np.result_type(float, omega, r_eq, inc, obl)
    thetas = np.asarray(m.thetas, dtype)
    phi = np.asarray(m.phi, dtype)
    triangulation = m.triangulation
    n_vert = int(m.n.sum())

    rtws, Ts, Fs = solve_ELR_vec(omega, thetas)

    # concrete numpy repeats + total_repeat_length keep this jit-safe
    def rep(a):
        return np.repeat(a, m.n, total_repeat_length=n_vert)

    rtw = rep(rtws)
    T = rep(Ts)
    F = rep(Fs)
    theta = rep(thetas)

    x, y, z = spherical_to_cartesian(rtw, theta, phi)
    points = r_eq * np.stack([x, y, z], axis=1)

    points_rotated = rotate_point_cloud(points, -inc, obl)
    # compute the normal vectors
    normals = triangle_normals(points_rotated, triangulation)

    # compute the x, y, z coordinates of each barycenter (barycenter vector)
    bary = barycenter(points_rotated, triangulation)
    # find the intensity of the star at each barycenter (mean of the
    # intensity at the corners of the triangle)
    intensity = np.mean(F[triangulation], axis=1)
    teff_ratio = np.mean(T[triangulation], axis=1)
    # his normals are unnormalised (|n| = 2 x triangle area), so cosine
    # already carries the projected area of each triangle: keep that.
    cosine = normals[:, 2]
    # apply a step function weight along with the contribution of flux
    # towards the observer; zeroes the non-visible portion of the star
    weight = np.heaviside(cosine, 0) * cosine * intensity
    if return_mesh:
        return (
            bary[:, 0],
            bary[:, 1],
            weight,
            teff_ratio,
            (points_rotated, triangulation, cosine, intensity),
        )
    return bary[:, 0], bary[:, 1], weight, teff_ratio


def visibilities(x, y, weight, uu, vv):
    """Complex visibilities of the triangle-barycentre point sources.

    His ``compute_DFTM1`` + ``apply_DFTM1``. ``uu``, ``vv`` are spatial
    frequencies in cycles per radian (any equal shape; virgil passes
    ``u / wavel``) and ``x``, ``y`` are in mas. The phase sign is that of
    ``virgil._geometry.offset_phase``. Normalised by ``weight.sum()``.
    Returns an array of shape ``uu.shape``.

    ``weight`` is 1D (one weight per triangle, shared by all samples) or 2D
    with shape ``(n, n_tri)`` or ``(1, n_tri)``, where ``n = uu.size`` (a
    weight per flattened sample, e.g. one spectrum per wavelength); each row
    is normalised by its own sum.
    """
    dtype = np.result_type(float, x, y, weight, uu, vv)
    uu = np.asarray(uu, dtype)
    vv = np.asarray(vv, dtype)
    shape = uu.shape
    u = uu.reshape(-1)
    v = vv.reshape(-1)
    x = np.asarray(x, dtype) * mas2rad
    y = np.asarray(y, dtype) * mas2rad

    # exp(-2j pi (u x + v y)) as separate cos / sin so the contraction over
    # triangles is a real matmul at HIGHEST precision
    arg = 2.0 * np.pi * (np.outer(u, x) + np.outer(v, y))
    w = np.asarray(weight, dtype)
    if w.ndim == 2 and w.shape[0] == 1:
        w = w[0]
    hi = jax.lax.Precision.HIGHEST
    if w.ndim == 1:
        w = w / w.sum()
        re = np.dot(np.cos(arg), w, precision=hi)
        im = -np.dot(np.sin(arg), w, precision=hi)
    else:
        w = w / w.sum(axis=1, keepdims=True)
        re = np.einsum("nt,nt->n", np.cos(arg), w, precision=hi)
        im = -np.einsum("nt,nt->n", np.sin(arg), w, precision=hi)
    return jax.lax.complex(re, im).reshape(shape)
