"""Tests for the Espinosa-Lara & Rieutord (2011) gravity-darkening port.

The golden values in ``data/elr_golden.npz`` were produced by Shashank
Dholakia's original code (see ``scripts/make_elr_golden.py``).  The
physics checks below are independent of that fixture.
"""

import pathlib

import jax
import jax.numpy as jnp
import numpy as onp
import pytest
from scipy.special import j1

from virgil import _elr

GOLDEN = pathlib.Path(__file__).parent.parent / "data" / "elr_golden.npz"
_MAS2RAD_REF = onp.pi / 180.0 / 3600.0 / 1000.0

_GOLD = dict(onp.load(GOLDEN))
N_SETS = _GOLD["vis_params"].shape[0]


def _tri_set(tri):
    return set(map(tuple, onp.sort(onp.asarray(tri), axis=1).tolist()))


def _common_triangles():
    """Indices (ours, golden) of triangles present in both meshes."""
    ours = {
        tuple(t): i
        for i, t in enumerate(
            onp.sort(onp.asarray(_elr.mesh(32).triangulation), axis=1).tolist()
        )
    }
    gold = {
        tuple(t): i
        for i, t in enumerate(
            onp.sort(_GOLD["mesh_triangulation"], axis=1).tolist()
        )
    }
    keys = sorted(set(ours) & set(gold))
    return (
        onp.array([ours[k] for k in keys]),
        onp.array([gold[k] for k in keys]),
    )


def _uv(i=None):
    w = _GOLD["vis_wavel"]
    return _GOLD["vis_u"] / w, _GOLD["vis_v"] / w


# ---------------------------------------------------------------- 1. golden


@pytest.fixture
def x64():
    with jax.enable_x64(True):
        yield


@pytest.mark.validates("virgil._elr", roots=["golden:dholakia"])
def test_golden_solver_float64(x64):
    omegas, thetas = _GOLD["omegas"], _GOLD["thetas"]
    for i, om in enumerate(omegas):
        rtw, teff, flux = _elr.solve_ELR_vec(
            jnp.float64(om), jnp.asarray(thetas)
        )
        assert rtw.dtype == jnp.float64
        onp.testing.assert_allclose(rtw, _GOLD["solver_rtw"][i], rtol=1e-8)
        onp.testing.assert_allclose(
            teff, _GOLD["solver_teff_ratio"][i], rtol=1e-8
        )
        onp.testing.assert_allclose(
            flux, _GOLD["solver_flux_ratio"][i], rtol=1e-8
        )


@pytest.mark.validates("virgil._elr", roots=["golden:dholakia"])
def test_golden_eq32_float64(x64):
    got = onp.array([_elr.eq32(jnp.float64(o)) for o in _GOLD["omegas"]])
    onp.testing.assert_allclose(got, _GOLD["eq32"], rtol=1e-12)


@pytest.mark.validates("virgil._elr", roots=["golden:dholakia"])
def test_golden_mesh():
    m = _elr.mesh(32)
    onp.testing.assert_allclose(m.thetas, _GOLD["mesh_thetas"], rtol=1e-12)
    assert onp.array_equal(m.n, _GOLD["mesh_n"])
    onp.testing.assert_allclose(m.phi, _GOLD["mesh_phi"], rtol=1e-12)
    ours = _tri_set(m.triangulation)
    gold = _tri_set(_GOLD["mesh_triangulation"])
    assert len(ours) == len(gold) == len(_GOLD["mesh_triangulation"])
    # qhull breaks ties between (near-)cospherical equatorial quads
    # differently across versions; allow a few flipped diagonals.
    assert len(ours - gold) <= 32


@pytest.mark.validates("virgil._elr", roots=["golden:dholakia"])
@pytest.mark.parametrize("k", range(N_SETS))
def test_golden_surface_and_visibilities_float64(x64, k):
    om, req, inc, obl = _GOLD["vis_params"][k]
    x, y, w, t = _elr.surface(
        jnp.float64(om), jnp.float64(req), jnp.float64(inc), jnp.float64(obl)
    )
    assert x.dtype == jnp.float64
    oi, gi = _common_triangles()
    assert len(oi) >= len(x) - 32
    for got, key in (
        (x, "bary_x"),
        (y, "bary_y"),
        (w, "weight"),
        (t, "teff_tri"),
    ):
        onp.testing.assert_allclose(
            onp.asarray(got)[oi], _GOLD[key][k][gi], rtol=1e-8, atol=1e-10
        )
    uu, vv = _uv()
    uu, vv = jnp.asarray(uu), jnp.asarray(vv)
    # His DFT, exactly: our visibilities of his own barycentres and weights.
    cv = _elr.visibilities(
        _GOLD["bary_x"][k], _GOLD["bary_y"][k], _GOLD["weight"][k], uu, vv
    )
    assert cv.dtype == jnp.complex128
    onp.testing.assert_allclose(cv, _GOLD["cvis"][k], rtol=1e-9, atol=1e-9)
    onp.testing.assert_allclose(
        jnp.abs(cv) ** 2, _GOLD["vis2"][k], rtol=1e-9, atol=1e-9
    )
    # The whole model: qhull splits the near-cospherical equatorial quads
    # differently across platforms and versions, which moves the
    # visibilities by up to ~3e-5.
    cv = _elr.visibilities(x, y, w, uu, vv)
    onp.testing.assert_allclose(cv, _GOLD["cvis"][k], atol=1e-4)


# ---------------------------------------------------- 2. golden in float32


@pytest.mark.validates("virgil._elr", roots=["golden:dholakia"])
def test_golden_solver_float32():
    for i, om in enumerate(_GOLD["omegas"]):
        rtw, teff, flux = _elr.solve_ELR_vec(
            jnp.float32(om), jnp.asarray(_GOLD["thetas"], jnp.float32)
        )
        assert rtw.dtype == flux.dtype == jnp.float32
        onp.testing.assert_allclose(rtw, _GOLD["solver_rtw"][i], rtol=1e-4)
        onp.testing.assert_allclose(
            teff, _GOLD["solver_teff_ratio"][i], rtol=2e-5
        )
        onp.testing.assert_allclose(
            flux, _GOLD["solver_flux_ratio"][i], rtol=1e-4
        )


@pytest.mark.validates("virgil._elr", roots=["golden:dholakia"])
def test_golden_eq32_float32():
    got = onp.array([_elr.eq32(jnp.float32(o)) for o in _GOLD["omegas"]])
    onp.testing.assert_allclose(got, _GOLD["eq32"], rtol=1e-5)


@pytest.mark.validates("virgil._elr", roots=["golden:dholakia"])
@pytest.mark.parametrize("k", range(N_SETS))
def test_golden_surface_and_visibilities_float32(k):
    om, req, inc, obl = _GOLD["vis_params"][k]
    x, y, w, t = _elr.surface(om, req, inc, obl)
    # float32 unless the run has enabled x64 globally
    assert x.dtype == w.dtype == t.dtype == jnp.asarray(1.0).dtype
    oi, gi = _common_triangles()
    scale = _GOLD["weight"][k].max()
    onp.testing.assert_allclose(
        onp.asarray(x)[oi], _GOLD["bary_x"][k][gi], rtol=1e-4, atol=1e-4 * req
    )
    onp.testing.assert_allclose(
        onp.asarray(y)[oi], _GOLD["bary_y"][k][gi], rtol=1e-4, atol=1e-4 * req
    )
    onp.testing.assert_allclose(
        onp.asarray(w)[oi],
        _GOLD["weight"][k][gi],
        rtol=1e-3,
        atol=1e-4 * scale,
    )
    onp.testing.assert_allclose(
        onp.asarray(t)[oi], _GOLD["teff_tri"][k][gi], rtol=2e-5
    )
    uu, vv = _uv()
    cv = _elr.visibilities(x, y, w, jnp.asarray(uu), jnp.asarray(vv))
    assert cv.dtype == jnp.asarray(1j).dtype
    onp.testing.assert_allclose(cv, _GOLD["cvis"][k], atol=1e-4)
    onp.testing.assert_allclose(jnp.abs(cv) ** 2, _GOLD["vis2"][k], atol=1e-4)


# ------------------------------------------------------------- 3. physics


THETAS = onp.linspace(0.05, onp.pi - 0.05, 41)


def test_omega_zero_is_a_sphere():
    rtw, teff, flux = _elr.solve_ELR_vec(0.0, jnp.asarray(THETAS, jnp.float32))
    for a in (rtw, teff, flux):
        assert onp.all(onp.isfinite(a))
        onp.testing.assert_allclose(a, 1.0, atol=1e-4)


@pytest.mark.parametrize("omega", [0.3, 0.7, 0.95])
def test_equatorial_teff_ratio_matches_eq32(omega):
    # Teff_ratio is normalized arbitrarily; eq32 is Teff(equator)/Teff(pole)
    _, t_eq, _ = _elr.solve_ELR(jnp.float32(omega), jnp.float32(onp.pi / 2))
    _, t_pole, _ = _elr.solve_ELR(jnp.float32(omega), jnp.float32(0.0))
    onp.testing.assert_allclose(
        t_eq / t_pole, _elr.eq32(jnp.float32(omega)), rtol=1e-5
    )


@pytest.mark.validates("virgil._elr", roots=["mathematics"])
@pytest.mark.parametrize("omega", [0.1, 0.5, 0.95])
def test_polar_radius_is_roche(omega):
    # theta -> 0: R_pole / R_eq = 1 / (1 + omega^2 / 2)
    rtw, _, _ = _elr.solve_ELR(jnp.float32(omega), jnp.float32(1e-4))
    onp.testing.assert_allclose(rtw, 1 / (1 + omega**2 / 2), rtol=1e-4)
    # equator: rtw = 1 by construction of the normalization
    req, _, _ = _elr.solve_ELR(jnp.float32(omega), jnp.float32(onp.pi / 2))
    onp.testing.assert_allclose(req, 1.0, rtol=1e-4)


@pytest.mark.validates("virgil._elr", roots=["mathematics"])
@pytest.mark.parametrize("omega", [0.3, 0.7, 0.95])
def test_closed_form_roche_radius(omega):
    th = THETAS[(omega * onp.sin(THETAS)) > 0.1]
    rtw, _, _ = _elr.solve_ELR_vec(
        jnp.float32(omega), jnp.asarray(th, jnp.float32)
    )
    # Closed form in units of R_pole, with omega_c = Omega/Omega_crit
    # (critical: R_eq = 1.5 R_pole); rtw is in units of R_eq, and
    # R_pole/R_eq = 1/(1 + omega^2/2).
    r_pole_over_eq = 1 / (1 + omega**2 / 2)
    omega_c = omega * onp.sqrt(27 / 8) * r_pole_over_eq**1.5
    s = omega_c * onp.sin(th)
    r = 3 / s * onp.cos((onp.pi + onp.arccos(s)) / 3)
    onp.testing.assert_allclose(
        onp.asarray(rtw), r * r_pole_over_eq, rtol=2e-4
    )


@pytest.mark.parametrize("omega", [0.3, 0.7, 0.95])
def test_gravity_darkening(omega):
    _, teff, flux = _elr.solve_ELR_vec(
        jnp.float32(omega), jnp.asarray(THETAS, jnp.float32)
    )
    assert flux[0] > flux[THETAS.size // 2]
    assert teff[0] > teff[THETAS.size // 2]
    assert onp.all(onp.isfinite(flux))


def test_solver_north_south_symmetry():
    th = jnp.asarray(THETAS, jnp.float32)
    a = _elr.solve_ELR_vec(0.8, th)
    b = _elr.solve_ELR_vec(0.8, jnp.pi - th)
    for p, q in zip(a, b):
        onp.testing.assert_allclose(p, q, rtol=1e-4)


# -------------------------------------------------------- 4. visibilities


def _vis(omega, req, inc, obl, uu, vv, n_lat=32):
    x, y, w, _ = _elr.surface(omega, req, inc, obl, n_lat=n_lat)
    return _elr.visibilities(x, y, w, jnp.asarray(uu), jnp.asarray(vv))


def test_visibility_at_zero_baseline_is_one():
    v = _vis(0.6, 0.4, 0.5, 0.3, [0.0], [0.0])
    onp.testing.assert_allclose(v, 1.0, atol=1e-5)


def _ud_amp(req, q):
    xx = onp.pi * 2 * req * _MAS2RAD_REF * q
    return onp.abs(2 * j1(xx) / xx)


@pytest.mark.validates("virgil._elr", roots=["mathematics"])
def test_omega_zero_matches_uniform_disk():
    req = 0.5
    # first null at x = 3.8317 -> q_null
    q_null = 3.8317 / (onp.pi * 2 * req * _MAS2RAD_REF)
    q = onp.linspace(0.05, 0.95, 12) * q_null
    errs = []
    for n_lat in (32, 64):
        v = onp.abs(_vis(0.0, req, 0.7, 1.0, q, 0 * q, n_lat=n_lat))
        errs.append(onp.max(onp.abs(v - _ud_amp(req, q))))
    assert errs[0] < 1e-2
    assert errs[1] < errs[0]
    assert errs[1] < 5e-3


def test_visibility_is_hermitian():
    rng = onp.random.default_rng(1)
    uu = rng.uniform(-2e8, 2e8, 8)
    vv = rng.uniform(-2e8, 2e8, 8)
    a = _vis(0.8, 0.4, 0.9, 0.4, uu, vv)
    b = _vis(0.8, 0.4, 0.9, 0.4, -uu, -vv)
    onp.testing.assert_allclose(b, onp.conj(a), atol=2e-5)


# ------------------------------------------------------------- 5. gradients

_UU = onp.array([2e7, 6e7, 1.2e8, -8e7])
_VV = onp.array([1e7, -5e7, 9e7, 6e7])


def _loss(p):
    v = _vis(p[0], p[1], p[2], p[3], _UU, _VV)
    return jnp.sum(jnp.abs(v) ** 2)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
@pytest.mark.parametrize("omega", [0.0, 0.5, 0.95])
def test_gradients_are_finite(dtype, omega):
    def run():
        p = jnp.asarray([omega, 0.4, 0.6, 0.5], dtype)
        g = jax.grad(_loss)(p)
        assert g.dtype == dtype
        assert onp.all(onp.isfinite(g)), g

    if dtype == jnp.float64:
        with jax.enable_x64(True):
            run()
    else:
        run()


def test_gradients_under_jit():
    p = jnp.asarray([0.5, 0.4, 0.6, 0.5], jnp.float32)
    g = jax.jit(jax.grad(_loss))(p)
    assert onp.all(onp.isfinite(g))
    onp.testing.assert_allclose(g, jax.grad(_loss)(p), rtol=5e-3, atol=1e-4)


def test_radius_gradient_matches_uniform_disk():
    req = 0.5
    uu = onp.array([1.0e8, 2.0e8, 3.0e8])
    vv = onp.array([0.0, 0.5e8, -1.0e8])
    q = onp.hypot(uu, vv)

    def loss(r):
        return jnp.sum(jnp.abs(_vis(0.0, r, 0.0, 0.0, uu, vv)) ** 2)

    got = float(jax.grad(loss)(jnp.float32(req)))
    h = 1e-4
    ref = (
        onp.sum(_ud_amp(req + h, q) ** 2) - onp.sum(_ud_amp(req - h, q) ** 2)
    ) / (2 * h)
    assert abs(got - ref) < 0.05 * abs(ref)


# --------------------------------------------------------- 6. orientation


def _hot(inc, obl, frac=0.05):
    x, y, w, t = (onp.asarray(a) for a in _elr.surface(0.9, 0.4, inc, obl))
    vis = w > 0
    x, y, t = x[vis], y[vis], t[vis]
    top = t >= onp.quantile(t, 1 - frac)
    return x[top].mean(), y[top].mean()


def test_pole_orientation_obl_zero():
    hx, hy = _hot(0.5, 0.0)
    assert hy > 0
    assert abs(hx) < 0.05 * 0.4


def test_pole_orientation_obl_ninety():
    hx, hy = _hot(0.5, onp.pi / 2)
    assert hx > 0
    assert abs(hy) < 0.05 * 0.4


def test_switching_x64_mode_reuses_the_mesh():
    # The cached mesh is shared by every call. JAX 0.10 caches the
    # canonical (x64-dependent) copy of a NumPy array by identity while it
    # is alive, so a 64-bit mesh index array used in one mode came back
    # with the wrong width in the other, and a compiled jnp.repeat
    # rejected it. Alternate the modes with the mesh in use throughout.
    def use():
        x, y, weight, teff = _elr.surface(0.7, 1.0, 0.6, 0.3)
        return weight

    held = []
    for x64 in (False, True, False, True):
        with jax.enable_x64(x64):
            held.append(jax.jit(use))
            held[-1]()
            dtype = jnp.float64 if x64 else jnp.float32
            assert use().dtype == dtype
            assert use().dtype == dtype
