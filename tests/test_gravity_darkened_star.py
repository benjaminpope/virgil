"""Tests for GravityDarkenedStar, written independently from its spec.

Golden values come from Shashank Dholakia's original ELR code
(``data/elr_golden.npz``, see ``scripts/make_elr_golden.py``).
"""

import os

import equinox as eqx
import jax
import jax.numpy as np
import matplotlib
import numpy as onp
import pytest

matplotlib.use("Agg")

from virgil import GravityDarkenedStar  # noqa: E402
from virgil._geometry import image_coordinates  # noqa: E402
from virgil.coverage import nrm_oidata  # noqa: E402
from virgil.likelihood import model_loglike  # noqa: E402
from virgil.models import PointSource, System, UniformDisk  # noqa: E402
from tests._shared import float_leaves  # noqa: E402

# Independent reference constant (not imported from virgil).
_MAS2RAD_REF = onp.pi / 180.0 / 3600.0 / 1000.0

_GOLDEN = os.path.join(
    os.path.dirname(__file__), "..", "data", "elr_golden.npz"
)


def _photocentre(image, fov_mas):
    x, y = image_coordinates(image.shape[0], fov_mas)
    image = onp.asarray(image, dtype=float)
    image = image / image.sum()
    return (
        float((image * onp.asarray(x)).sum()),
        float((image * onp.asarray(y)).sum()),
    )


def _rms_widths(image, fov_mas):
    """RMS widths (East-West, North-South) of the image about its centroid."""
    x, y = (onp.asarray(a) for a in image_coordinates(image.shape[0], fov_mas))
    image = onp.asarray(image, dtype=float)
    image = image / image.sum()
    mx, my = (image * x).sum(), (image * y).sum()
    return (
        float(onp.sqrt((image * (x - mx) ** 2).sum())),
        float(onp.sqrt((image * (y - my) ** 2).sum())),
    )


# --- 1. golden values ----------------------------------------------------


@pytest.mark.validates(
    "virgil.models.GravityDarkenedStar", roots=["golden:dholakia"]
)
def test_matches_original_code_golden_values():
    golden = onp.load(_GOLDEN)
    with jax.enable_x64():
        u = np.asarray(golden["vis_u"], dtype=np.float64)
        v = np.asarray(golden["vis_v"], dtype=np.float64)
        wavel = float(golden["vis_wavel"])
        for i, (omega, r_eq, inc_sd, obl) in enumerate(golden["vis_params"]):
            star = GravityDarkenedStar(
                2 * r_eq, omega, 90.0 - onp.degrees(inc_sd), onp.degrees(obl)
            )
            cvis = onp.asarray(star.model(u, v, wavel))
            # Not exact: qhull splits the near-cospherical equatorial quads
            # of the mesh differently across platforms (~3e-5 here). The
            # exact checks against his code are in test_elr.py.
            onp.testing.assert_allclose(cvis, golden["cvis"][i], atol=1e-4)
            onp.testing.assert_allclose(
                onp.abs(cvis) ** 2, golden["vis2"][i], atol=1e-4
            )


# --- 2. omega = 0 limit --------------------------------------------------


def _ud_error(n_lat):
    wavel = 1.0e-6
    u = onp.linspace(5.0, 100.0, 12)
    v = onp.linspace(100.0, 10.0, 12)
    star = GravityDarkenedStar(1.0, omega=0.0, inc=60.0, pa=30.0, n_lat=n_lat)
    ud = UniformDisk(1.0)
    return float(
        onp.max(onp.abs(star.model(u, v, wavel) - ud.model(u, v, wavel)))
    )


@pytest.mark.validates(
    "virgil.models.GravityDarkenedStar", roots=["mathematics"]
)
def test_omega_zero_matches_uniform_disk():
    assert _ud_error(32) < 1e-2


def test_higher_mesh_resolution_matches_uniform_disk_better():
    assert _ud_error(64) < _ud_error(32)
    assert _ud_error(64) < 5e-3


def test_unit_visibility_at_zero_baseline():
    star = GravityDarkenedStar(1.0, 0.8, 50.0, 20.0)
    vis = star.model(np.zeros(2), np.zeros(2), 1e-6)
    onp.testing.assert_allclose(vis, 1.0, atol=1e-5)


# --- 3. render orientation ----------------------------------------------


def test_render_is_unit_sum_and_finite():
    image = GravityDarkenedStar(2.0, 0.9, 45.0).render(64, 3.0)
    assert image.shape == (64, 64)
    assert onp.all(onp.isfinite(image))
    onp.testing.assert_allclose(onp.sum(image), 1.0, rtol=1e-4)


@pytest.mark.parametrize(
    "pa, sign_ra, sign_dec",
    [(0.0, 0, 1), (90.0, 1, 0), (180.0, 0, -1), (270.0, -1, 0)],
)
def test_photocentre_is_toward_visible_pole(pa, sign_ra, sign_dec):
    diam = 2.0
    fov = 1.5 * diam
    star = GravityDarkenedStar(diam, omega=0.9, inc=45.0, pa=pa)
    cx, cy = _photocentre(star.render(128, fov), fov)
    shift = float(onp.hypot(cx, cy))
    assert shift > 0.01
    # the shift is along the pole axis and nearly perpendicular offset is small
    if sign_dec:
        assert sign_dec * cy > 0.9 * shift
        assert abs(cx) < 0.1 * shift
    else:
        assert sign_ra * cx > 0.9 * shift
        assert abs(cy) < 0.1 * shift


def test_equator_on_shape_is_perpendicular_to_pole():
    diam, fov = 2.0, 3.0
    w_ew, w_ns = _rms_widths(
        GravityDarkenedStar(diam, 0.95, 90.0, 0.0).render(128, fov), fov
    )
    assert w_ew > 1.03 * w_ns  # pole North: equator runs East-West
    w_ew, w_ns = _rms_widths(
        GravityDarkenedStar(diam, 0.95, 90.0, 90.0).render(128, fov), fov
    )
    assert w_ns > 1.03 * w_ew  # pole East: equator runs North-South


@pytest.mark.slow
def test_offsets_shift_photocentre():
    diam, fov = 2.0, 16.0
    base = GravityDarkenedStar(diam, 0.9, 45.0, 0.0)
    c0 = _photocentre(base.render(256, fov), fov)
    c1 = _photocentre(
        base.set(["dra", "ddec"], [3.0, -2.0]).render(256, fov), fov
    )
    # pixel scale is 16/256 = 0.0625 mas, so allow ~one pixel of slack
    assert c1[0] - c0[0] == pytest.approx(3.0, abs=0.07)
    assert c1[1] - c0[1] == pytest.approx(-2.0, abs=0.07)


# --- 4. visibility-phase orientation ------------------------------------


def _phase_photocentre(star, u, v, wavel):
    vis = onp.asarray(star.model(np.asarray(u), np.asarray(v), wavel))
    phase = onp.angle(vis)
    fu, fv = onp.asarray(u) / wavel, onp.asarray(v) / wavel
    # phase = -2 pi mas2rad (u dra_c + v ddec_c)
    return (
        -phase / (2 * onp.pi * _MAS2RAD_REF * fu)
        if fu[0]
        else (-phase / (2 * onp.pi * _MAS2RAD_REF * fv))
    )


@pytest.mark.parametrize("pa", [0.0, 180.0])
def test_visibility_phase_photocentre_matches_render(pa):
    diam, fov, wavel = 2.0, 3.0, 1.0e-6
    star = GravityDarkenedStar(diam, 0.9, 45.0, pa)
    _, cy = _photocentre(star.render(256, fov), fov)
    # short North-South baseline: 2 m at 1 um, phase ~ 2 pi * 2e6 * 2*4.85e-9
    ddec_c = _phase_photocentre(star, [0.0], [2.0], wavel)[0]
    assert onp.sign(ddec_c) == onp.sign(cy) == (1 if pa == 0.0 else -1)
    assert ddec_c == pytest.approx(cy, rel=0.1)


def test_visibility_phase_photocentre_east_west():
    diam, fov, wavel = 2.0, 3.0, 1.0e-6
    star = GravityDarkenedStar(diam, 0.9, 45.0, 90.0)
    cx, _ = _photocentre(star.render(256, fov), fov)
    dra_c = _phase_photocentre(star, [2.0], [0.0], wavel)[0]
    assert dra_c > 0
    assert dra_c == pytest.approx(cx, rel=0.1)


def test_visibility_phase_follows_offset():
    wavel = 1.0e-6
    star = GravityDarkenedStar(1.0, 0.0, 90.0)
    off = star.set(["dra", "ddec"], [3.0, -2.0])
    u, v = np.array([10.0]), np.array([20.0])
    ratio = off.model(u, v, wavel) / star.model(u, v, wavel)
    arg = -2 * onp.pi * _MAS2RAD_REF * (u * 3.0 + v * -2.0) / wavel
    onp.testing.assert_allclose(
        onp.angle(ratio), onp.angle(onp.exp(1j * arg)), atol=2e-3
    )


# --- 5. closure phases ---------------------------------------------------


def _triangle_cp(star, b1, b2, wavel=0.7e-6):
    b3 = (-(b1[0] + b2[0]), -(b1[1] + b2[1]))
    u = np.array([b1[0], b2[0], b3[0]])
    v = np.array([b1[1], b2[1], b3[1]])
    vis = onp.asarray(star.model(u, v, wavel))
    return float(onp.angle(vis[0] * vis[1] * vis[2]))


@pytest.mark.parametrize("omega", [0.0, 0.5, 0.95])
def test_closure_phase_zero_when_equator_on(omega):
    with jax.enable_x64():
        star = GravityDarkenedStar(1.0, omega, 90.0, 33.0)
        for b1, b2 in [
            ((60.0, 10.0), (-20.0, 80.0)),
            ((90.0, 0.0), (10.0, 50.0)),
        ]:
            # Not exactly zero: Shashank's mesh has an odd number of points
            # on most rings, so it is not exactly point-symmetric. The
            # residual (~2e-4 rad at n_lat = 32) is far below CHARA's
            # closure-phase errors (~1e-2 rad).
            assert abs(_triangle_cp(star, b1, b2)) < 1e-3


def test_closure_phase_nonzero_when_inclined():
    with jax.enable_x64():
        star = GravityDarkenedStar(1.0, 0.9, 45.0, 0.0)
        cps = [
            _triangle_cp(star, b1, b2)
            for b1, b2 in [
                ((60.0, 10.0), (-20.0, 80.0)),
                ((90.0, 0.0), (10.0, 50.0)),
            ]
        ]
        assert max(abs(c) for c in cps) > 1e-2


def test_nrm_closure_phases_inclined_vs_equator_on():
    # 4.3 um NIRISS mask, star (60 mas) resolved on the ~6 m baselines
    data = nrm_oidata()
    n_v2 = data.vis.size

    def cps(inc):
        star = GravityDarkenedStar(60.0, 0.9, inc, 0.0)
        return onp.asarray(data.with_model(star).phi), n_v2

    cp_eq, _ = cps(90.0)
    cp_inc, _ = cps(45.0)
    assert onp.max(onp.abs(cp_eq)) < 1e-3
    assert onp.max(onp.abs(cp_inc)) > 10 * onp.max(onp.abs(cp_eq))


# --- 6. gradients --------------------------------------------------------


def _data_for_gradients():
    truth = GravityDarkenedStar(1.0, 0.5, 60.0, 20.0, flux=1.0)
    return nrm_oidata(sigma_cp_deg=1.0).with_model(
        truth, key=jax.random.PRNGKey(0)
    )


def _check_grads(model, data):
    grads = eqx.filter_jit(eqx.filter_grad(model_loglike))(model, data)
    leaves = float_leaves(grads)
    assert len(leaves) >= 6
    for leaf in leaves:
        assert onp.all(onp.isfinite(onp.asarray(leaf)))


@pytest.mark.parametrize("omega", [0.0, 0.9])
def test_gradients_finite_float32(omega):
    data = _data_for_gradients()
    _check_grads(GravityDarkenedStar(1.0, omega, 55.0, 15.0), data)


@pytest.mark.parametrize("omega", [0.0, 0.9])
def test_gradients_finite_float64(omega):
    with jax.enable_x64():
        data = _data_for_gradients()
        _check_grads(GravityDarkenedStar(1.0, omega, 55.0, 15.0), data)


def test_gradient_of_model_traced_under_jit():
    data = _data_for_gradients()
    star = GravityDarkenedStar(1.0, 0.7, 55.0, 15.0)
    value = eqx.filter_jit(model_loglike)(star, data)
    assert onp.isfinite(float(value))
    # gradient is nonzero for a model mismatched with the data
    grads = eqx.filter_grad(model_loglike)(star, data)
    assert any(onp.any(onp.asarray(g) != 0) for g in float_leaves(grads))


# --- 7. composition ------------------------------------------------------


def test_system_is_flux_weighted_mix():
    star = GravityDarkenedStar(1.0, 0.6, 50.0, 10.0, flux=1.0)
    comp = PointSource(flux=0.1, dra=5.0)
    system = System(star=star, comp=comp)
    u = onp.linspace(5.0, 90.0, 10)
    v = onp.linspace(60.0, 8.0, 10)
    wavel = 1.0e-6
    expected = (
        star.flux * star.model(u, v, wavel)
        + comp.flux * comp.model(u, v, wavel)
    ) / (star.flux + comp.flux)
    onp.testing.assert_allclose(
        system.model(u, v, wavel), expected, rtol=1e-4, atol=1e-5
    )


def test_system_paths_settable():
    system = System(
        star=GravityDarkenedStar(1.0, 0.6, 50.0), comp=PointSource(flux=0.1)
    )
    system = system.set("star.omega", 0.3)
    assert float(system.star.omega) == pytest.approx(0.3)
    system = system.set(["star.inc", "star.pa"], [70.0, 12.0])
    assert float(system.star.inc) == pytest.approx(70.0)
    assert float(system.star.pa) == pytest.approx(12.0)


# --- 8. validation -------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"diam_eq": 0.0},
        {"diam_eq": -1.0},
        {"omega": 1.0},
        {"omega": -0.1},
        {"inc": 100.0},
    ],
)
def test_invalid_construction_raises(kwargs):
    args = {"diam_eq": 1.0, **kwargs}
    with pytest.raises(ValueError):
        GravityDarkenedStar(**args)


def test_is_physical():
    star = GravityDarkenedStar(1.0, 0.5, 45.0)
    assert bool(star.is_physical())
    for path, value in [
        ("diam_eq", 0.0),
        ("diam_eq", -1.0),
        ("omega", 1.0),
        ("omega", 1.5),
        ("omega", -0.1),
        ("inc", 100.0),
        ("inc", -5.0),
        ("flux", -0.1),
    ]:
        assert not bool(star.set(path, value).is_physical()), (path, value)
    # boundaries that are allowed
    assert bool(star.set("inc", 0.0).is_physical())
    assert bool(star.set("inc", 90.0).is_physical())
    assert bool(star.set("omega", 0.0).is_physical())


def test_is_physical_is_traceable():
    star = GravityDarkenedStar(1.0, 0.5, 45.0)
    out = jax.jit(lambda m: m.is_physical())(star)
    assert bool(out)


def test_plot_surface_returns_artist():
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    artist = GravityDarkenedStar(1.0, 0.8, 60.0, 30.0).plot_surface(ax)
    assert artist is not None
    artist2 = GravityDarkenedStar(1.0, 0.8, 60.0).plot_surface()
    assert artist2 is not None
    plt.close("all")


# --- 5. rasterized image ------------------------------------------------


def _outline_radii(star):
    """Smallest and largest sky radius of the visible outline."""
    pts, tri, cosine, _ = star._surface(return_mesh=True)[-1]
    pts, tri = onp.asarray(pts), onp.asarray(tri)
    visible = onp.asarray(cosine) > 0
    used = pts[onp.unique(tri[visible])][:, :2]
    radius = onp.hypot(used[:, 0], used[:, 1])
    return radius.min(), radius.max()


def test_image_flux_matches_dft_weights():
    xx, yy = image_coordinates(64, 3.0)
    star = GravityDarkenedStar(2.0, 0.9, 45.0, 20.0, n_lat=32)
    image = star._centred_image(xx, yy, 3.0 / 64)
    total = star._surface()[2].sum()
    onp.testing.assert_allclose(image.sum(), total, rtol=5e-3)
    hot = GravityDarkenedStar(2.0, 0.9, 45.0, 20.0, t_pole=9000.0)
    image = hot._centred_image(xx, yy, 3.0 / 64)
    weights = hot._planck_weights(hot.wavel0)[2]
    onp.testing.assert_allclose(image.sum(), weights.sum(), rtol=5e-3)


def test_image_is_zero_outside_outline_and_positive_inside():
    fov, npix = 3.0, 64
    pixel = fov / npix
    star = GravityDarkenedStar(2.0, 0.9, 45.0, 20.0, n_lat=32)
    image = onp.asarray(star.render(npix, fov))
    xx, yy = image_coordinates(npix, fov)
    radius = onp.hypot(xx, yy)
    # the star is centred on the origin; edge pixels are partly covered
    r_min, r_max = _outline_radii(star)
    assert onp.all(image[radius > r_max + 1.5 * pixel] == 0.0)
    assert onp.all(image[radius < r_min - 1.5 * pixel] > 0.0)


def test_pole_on_disk_is_smooth():
    # slow rotation, pole-on: nearly uniform disk, so no speckle expected
    fov, npix = 3.0, 128
    star = GravityDarkenedStar(2.0, 0.05, 0.0, n_lat=32)
    image = onp.asarray(star.render(npix, fov))
    xx, yy = image_coordinates(npix, fov)
    radius = onp.hypot(xx, yy)
    values = image[(radius > 0.3) & (radius < 0.8)]
    assert values.std() / values.mean() < 0.01


@pytest.mark.slow
def test_image_works_in_float64_and_fine_grid():
    star = GravityDarkenedStar(2.0, 0.7, 60.0, 10.0, n_lat=64)
    image = star.render(128, 3.0)
    assert image.shape == (128, 128)
    assert onp.all(onp.isfinite(image))
    with jax.enable_x64(True):
        star64 = GravityDarkenedStar(2.0, 0.7, 60.0, 10.0, n_lat=16)
        image64 = star64.render(48, 3.0)
        assert image64.dtype == onp.float64
        onp.testing.assert_allclose(image64.sum(), 1.0, rtol=1e-10)


def test_image_is_jit_and_grad_safe():
    def left_half(omega):
        star = GravityDarkenedStar(2.0, omega, 50.0, 10.0, n_lat=16)
        return star.render(24, 3.0)[:, :12].sum()

    value, grad = jax.jit(jax.value_and_grad(left_half))(0.5)
    assert onp.isfinite(value) and onp.isfinite(grad)
