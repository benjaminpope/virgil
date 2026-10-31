"""Numerical check of the coordinate derivations in design/orbit_alias_cycles.md.

Checks, each printed as ``name: max error``: the circle amplitudes
u = a(1 + cos i) e^{i(w + O)} and V = a(1 - cos i) e^{i(w - O)} from the
library's Thiele-Innes constants; the Jacobian 32 / |z|^4 of
z -> (ln a, cos i, w + O, w - O); the normalized TI' density; the
eccentricity chart k + ih = zeta^2; the anchor rotation (a unit-Jacobian
change of the phase of the eccentricity vector); the radial chart
z = g(|y|) y / |y|; and prior recovery (cos i, a, e) from z and zeta draws.

    ~/.claude/bin/pyslot .venv/bin/python scripts/check_orbit_coordinates.py
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
zeta = np.array([0.3, -0.4])
jac_k = jax.jacfwd(
    lambda s: np.array(
        [((s[0] + 1j * s[1]) ** 2).real, ((s[0] + 1j * s[1]) ** 2).imag]
    )
)
det = abs(float(np.linalg.det(jac_k(zeta))))
results["|det d(k,h)/d zeta| / (4|zeta|^2) - 1"] = abs(
    det / (4 * float(zeta @ zeta)) - 1
)


# 5. anchor rotation: (u, V, k + ih) -> phases shifted, unit Jacobian ---
def rotate(x, theta):
    """TI' and the eccentricity-vector phase after the anchor rotation:
    u' = e^{i th} u, V' = e^{i th} V (w -> e^{i th} w in p = (i/2)(V w + conj(u) conj(w)))."""
    ur, ui, vr, vi, th0 = x
    ph = np.exp(1j * theta)
    u, v = (ur + 1j * ui) * ph, (vr + 1j * vi) * ph
    return np.array([u.real, u.imag, v.real, v.imag, th0 + theta])


x0 = np.array([1.2, -0.7, 0.4, 2.0, 0.3])
det = abs(float(np.linalg.det(jax.jacfwd(rotate)(x0, 0.9))))
results["anchor rotation |det| - 1 (fixed theta)"] = abs(det - 1)

# 6. radial chart z = g(|y|) y/|y|, density g'/(g rho^3) in y -----------
r_lo, r_hi = onp.sqrt(2 * a_lo), onp.sqrt(2 * a_hi)


def g(rho):
    s = 1 / (1 + 1 / rho)  # sigmoid of ln rho: s = rho / (1 + rho)
    return np.exp(np.log(r_lo) + (np.log(r_hi) - np.log(r_lo)) * s)


def y_to_z(y):
    rho = np.sqrt(np.sum(y**2))
    return g(rho) * y / rho


err = 0.0
dg = jax.grad(g)
for _ in range(20):
    y = np.array(rng.normal(size=4) * rng.uniform(0.3, 3))
    rho = float(np.sqrt(np.sum(y**2)))
    det = abs(float(np.linalg.det(jax.jacfwd(y_to_z)(y))))
    pred = float(g(rho)) ** 3 * float(dg(rho)) / rho**3
    err = max(err, abs(det / pred - 1))
results["radial chart |det dz/dy| / (g^3 g'/rho^3) - 1"] = err

# 7. prior recovery: z ~ |z|^-4 on the shell gives cos i ~ U(-1,1),
# ln a ~ U, phases uniform; zeta ~ U(disc) gives e ~ U(0, e_max) ---------
m = 200_000
lnr = rng.uniform(onp.log(r_lo), onp.log(r_hi), m)
w = rng.normal(size=(m, 4))
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
e_max = 0.9
rho = onp.sqrt(
    e_max * rng.uniform(size=m)
)  # uniform on the disc of radius sqrt(e_max)
e = rho**2
results["KS e vs U(0, e_max) (p-value)"] = stats.kstest(
    e / e_max, "uniform"
).pvalue

width = max(map(len, results))
for name, value in results.items():
    print(f"{name:<{width}}  {value:.3e}")
