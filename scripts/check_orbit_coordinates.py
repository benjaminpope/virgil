"""Numerical check of the coordinate derivations in design/orbit_alias_cycles.md.

Checks, each printed as ``name: max error``: the circle amplitudes
u = a(1 + cos i) e^{i(w + O)} and V = a(1 - cos i) e^{i(w - O)} from the
library's Thiele-Innes constants; the Jacobian 32 / |z|^4 of
z -> (ln a, cos i, w + O, w - O); the normalized TI' density; the
eccentricity chart k + ih = zeta^2; the anchor change of variables (rotation angle as
coordinate, unit Jacobian); the ζ chart ζ = √e_max y/√(1+|y|²) (Jacobian e_max(1+|y|²)⁻²);
the z chart (auxiliary Gaussian direction plus logistic ln a); and prior recovery
(cos i, a, e) from z and ζ draws.

    PYTHONPATH=src .venv/bin/python scripts/check_orbit_coordinates.py
"""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as np  # noqa: E402
import numpy as onp  # noqa: E402
from scipy import stats  # noqa: E402

from virgil.orbits import KeplerOrbit, _thiele_innes_angles  # noqa: E402

rng = onp.random.default_rng(0)
results = {}


def z_to_ti(z):
    """(A, B, F, G) from z in R^4: u = z_u^2, V = z_v^2."""
    u = (z[0] + 1j * z[1]) ** 2
    v = (z[2] + 1j * z[3]) ** 2
    return np.array(
        [
            (u.real + v.real) / 2,  # A
            (u.imag - v.imag) / 2,  # B  (B - F = Im u, B + F = -Im V)
            (-v.imag - u.imag) / 2,  # F
            (u.real - v.real) / 2,  # G
        ]
    )


def z_to_campbell(z):
    """(ln a, cos i, w + O, w - O) of z: |u| = |z_u|^2 = a (1 + cos i)."""
    zu, zv = z[0] + 1j * z[1], z[2] + 1j * z[3]
    u, v = zu**2, zv**2
    a = (np.abs(u) + np.abs(v)) / 2
    cos_i = (np.abs(u) - np.abs(v)) / (np.abs(u) + np.abs(v))
    return np.array([np.log(a), cos_i, np.angle(u), -np.angle(np.conj(v))])


# 1. u, V amplitudes from the library's own constants -------------------
err = 0.0
for _ in range(50):
    a, c = rng.uniform(1, 30), rng.uniform(-0.99, 0.99)
    w, om = rng.uniform(0, 2 * onp.pi, 2)
    orb = KeplerOrbit(
        period=10.0,
        dt_peri=1.0,
        ecc=0.3,
        inc=onp.rad2deg(onp.arccos(c)),
        omega=onp.rad2deg(w),
        Omega=onp.rad2deg(om),
        a_mas=a,
        t_ref=0.0,
    )
    A, B, F, G, _, _ = (float(x) for x in orb.thiele_innes())
    u = (A + G) + 1j * (B - F)
    v = (A - G) - 1j * (B + F)
    err = max(
        err,
        abs(abs(u) - a * (1 + c)),
        abs(abs(v) - a * (1 - c)),
        abs(onp.exp(1j * (w + om)) * a * (1 + c) - u),
        abs(onp.exp(1j * (w - om)) * a * (1 - c) - v),
    )
    inc, _, _, a_lib = _thiele_innes_angles(A, B, F, G)
    err = max(err, abs(float(a_lib) - a), abs(onp.cos(onp.deg2rad(inc)) - c))
results["u, V amplitudes and a = (|u|+|V|)/2"] = err

# 2. Jacobian of z -> (ln a, cos i, w+O, w-O) = 32 / |z|^4 --------------
err = 0.0
jac = jax.jacfwd(z_to_campbell)
for _ in range(50):
    z = np.array(rng.normal(size=4) * rng.uniform(0.5, 5))
    det = abs(float(np.linalg.det(jac(z))))
    err = max(err, abs(det * float(np.sum(z**2)) ** 2 / 32 - 1))
results["|det d(ln a, cos i, w+O, w-O)/dz| |z|^4 / 32 - 1"] = err

# 3. d(A,B,F,G)/dz = 4 |u||V| and p(TI') ----------------------------
err = 0.0
jac_ti = jax.jacfwd(z_to_ti)
for _ in range(50):
    z = np.array(rng.normal(size=4) * rng.uniform(0.5, 5))
    det = abs(float(np.linalg.det(jac_ti(z))))
    nu, nv = float(z[0] ** 2 + z[1] ** 2), float(z[2] ** 2 + z[3] ** 2)
    err = max(err, abs(det / (4 * nu * nv) - 1))
results["|det d(A,B,F,G)/dz| / (4|u||V|) - 1"] = err

# the density of TI' integrates to one: p = 1 / (pi^2 ln(a_hi/a_lo)
# (|u|+|V|)^2 |u||V|), on a_lo <= (|u|+|V|)/2 <= a_hi. In (|u|, |V|, phases)
# d(A,B,F,G) = (1/4) |u| |V| d|u| d|V| dphi_u dphi_v.
a_lo, a_hi = 1.0, 30.0
n = 400
lu = onp.linspace(
    onp.log(1e-3), onp.log(2 * a_hi), n
)  # integrate in ln|u|, ln|V|
dl = lu[1] - lu[0]
U, Vm = onp.meshgrid(onp.exp(lu), onp.exp(lu), indexing="ij")
inside = (U + Vm >= 2 * a_lo) & (U + Vm <= 2 * a_hi)
p = 1.0 / (onp.pi**2 * onp.log(a_hi / a_lo) * (U + Vm) ** 2 * U * Vm)
mass = (
    (p * inside * 0.25 * U * Vm * U * Vm).sum() * dl * dl * (2 * onp.pi) ** 2
)
results["normalized TI' density, integral - 1 (grid)"] = abs(mass - 1)

# 4. eccentricity chart: k + ih = zeta^2, zeta uniform on the disc ------
jac_k = jax.jacfwd(
    lambda s: np.array(
        [((s[0] + 1j * s[1]) ** 2).real, ((s[0] + 1j * s[1]) ** 2).imag]
    )
)
err = 0.0
for _ in range(50):
    zeta = np.array(rng.normal(size=2) * rng.uniform(0.2, 1.0))
    det = abs(float(np.linalg.det(jac_k(zeta))))
    err = max(err, abs(det / (4 * float(zeta @ zeta)) - 1))
results["|det d(k,h)/d zeta| / (4|zeta|^2) - 1"] = err


# 5. anchor change of variables (M_a, u, V) -> (M_a, u', V', M_a): the
# rotation angle is itself the new coordinate, so the Jacobian is
# triangular (rotation block, identity in M_a) and the TI' prior, a function
# of (|u|, |V|), does not depend on M_a ------------------------------------
def anchor(x):
    """x = (Re u, Im u, Re V, Im V, M_a) -> (Re u', Im u', Re V', Im V', M_a)
    with u' = e^{i M_a} u, V' = e^{i M_a} V."""
    ph = np.exp(1j * x[4])
    u, v = (x[0] + 1j * x[1]) * ph, (x[2] + 1j * x[3]) * ph
    return np.array([u.real, u.imag, v.real, v.imag, x[4]])


err, err_inv = 0.0, 0.0
for _ in range(50):
    x = np.array(rng.normal(size=5) * rng.uniform(0.5, 3))
    err = max(err, abs(abs(float(np.linalg.det(jax.jacfwd(anchor)(x)))) - 1))
    y = anchor(x)
    err_inv = max(
        err_inv,
        abs(float(np.hypot(y[0], y[1]) - np.hypot(x[0], x[1]))),
        abs(float(np.hypot(y[2], y[3]) - np.hypot(x[2], x[3]))),
    )
results["anchor change of variables |det| - 1"] = err
results["anchor change of variables, |u|, |V| invariant"] = err_inv

# 6. charts. z: direction of an auxiliary Gaussian y in R^4 plus a logistic
# coordinate l for ln a (a smooth chart, no singularity); zeta: zeta =
# sqrt(e_max) y / sqrt(1 + |y|^2), density (1 + |y|^2)^-2 ---------------------
r_lo, r_hi = onp.sqrt(2 * a_lo), onp.sqrt(2 * a_hi)
e_max = 0.9


def zeta_chart(y):
    return onp.sqrt(e_max) * y / np.sqrt(1 + np.sum(y**2))


err = 0.0
for _ in range(50):
    y = np.array(rng.normal(size=2) * rng.uniform(0.2, 4))
    det = abs(float(np.linalg.det(jax.jacfwd(zeta_chart)(y))))
    err = max(err, abs(det / (e_max / (1 + float(y @ y)) ** 2) - 1))
results["zeta chart |det dzeta/dy| / (e_max (1+|y|^2)^-2) - 1"] = err

# 7. prior recovery: the z chart (y ~ N(0, I_4) direction, l ~ logistic) gives
# cos i ~ U(-1,1), ln a ~ U, phases uniform; the zeta chart (y ~ density
# (1+|y|^2)^-2 on R^2, drawn by inverse CDF) gives e ~ U(0, e_max) ---------
m = 200_000
ell = rng.logistic(size=m)
lnr = onp.log(r_lo) + (onp.log(r_hi) - onp.log(r_lo)) / (1 + onp.exp(-ell))
w = rng.normal(size=(m, 4))
z = onp.exp(lnr)[:, None] * w / onp.linalg.norm(w, axis=1)[:, None]
zu, zv = z[:, 0] + 1j * z[:, 1], z[:, 2] + 1j * z[:, 3]
u, v = zu**2, zv**2
a = (abs(u) + abs(v)) / 2
cos_i = (abs(u) - abs(v)) / (abs(u) + abs(v))
# in the chart ln a is uniform only if l ~ U(0, 1) after the logistic map:
# check the map's density instead (logistic CDF of l is uniform)
results["KS logistic(l) vs U(0,1) (p-value)"] = stats.kstest(
    1 / (1 + onp.exp(-ell)), "uniform"
).pvalue

lnr = rng.uniform(onp.log(r_lo), onp.log(r_hi), m)
z = onp.exp(lnr)[:, None] * w / onp.linalg.norm(w, axis=1)[:, None]
zu, zv = z[:, 0] + 1j * z[:, 1], z[:, 2] + 1j * z[:, 3]
u, v = zu**2, zv**2
a = (abs(u) + abs(v)) / 2
cos_i = (abs(u) - abs(v)) / (abs(u) + abs(v))
results["KS cos i vs U(-1,1) (p-value, want > 0.01)"] = stats.kstest(
    cos_i, stats.uniform(-1, 2).cdf
).pvalue
results["KS ln a vs U (p-value)"] = stats.kstest(
    (onp.log(a) - onp.log(a_lo)) / onp.log(a_hi / a_lo), "uniform"
).pvalue
results["KS w + O vs U(-pi, pi) (p-value)"] = stats.kstest(
    onp.angle(u), stats.uniform(-onp.pi, 2 * onp.pi).cdf
).pvalue
results["KS w - O vs U(-pi, pi) (p-value)"] = stats.kstest(
    -onp.angle(onp.conj(v)), stats.uniform(-onp.pi, 2 * onp.pi).cdf
).pvalue
y = rng.uniform(size=m)
rho_y = onp.sqrt(y / (1 - y))  # |y| has CDF r^2/(1+r^2)
e = e_max * rho_y**2 / (1 + rho_y**2)
results["KS e vs U(0, e_max) (p-value)"] = stats.kstest(
    e / e_max, "uniform"
).pvalue

width = max(map(len, results))
for name, value in results.items():
    print(f"{name:<{width}}  {value:.3e}")
