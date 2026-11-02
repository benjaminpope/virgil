import warnings

import equinox as eqx
import jax
import jax.numpy as np
import numpy as onp
import numpyro.distributions as dist
import pytest

from virgil._geometry import pixel_offsets
from virgil._utils import mas2rad
from virgil.coverage import vlti_oidata
from virgil.fields import (
    MultipoleField,
    _azimuthal_filter,
    _radial_factor,
    multipole_field_priors,
    polar_interpolate,
)
from virgil.fitting import fit
from virgil.imaging import image_priors, log_evidence
from virgil.models import (
    GaussianDisk,
    GaussianRing,
    Harmonics,
    Image,
    PointSource,
    MultipoleDisk,
    System,
)

from ._compiles import count_compiles

N_R, N_T, SHAPE, R_MAX = 8, 12, (24, 24), 12.0


def _field(seed=0, shape=SHAPE, **kwargs):
    latent = onp.random.default_rng(seed).normal(size=(N_R, N_T))
    options = (
        dict(sigma=1.0, length_r=3.0, length_theta=1.0, inc=40.0, pa=25.0)
        | kwargs
    )
    return MultipoleField(latent, shape, R_MAX, **options)


def _matern(d, ell, nu):
    x = d / ell
    if nu == 0.5:
        return onp.exp(-x)
    if nu == 1.5:
        return (1 + onp.sqrt(3) * x) * onp.exp(-onp.sqrt(3) * x)
    a = onp.sqrt(5) * x
    return (1 + a + a * a / 3) * onp.exp(-a)


def _azimuthal_covariance(n, angle, nu):
    """The circulant covariance whose order-m variance is MultipoleKernel's
    S(m) = [1 + (m angle)^2 / 2 nu]^(-nu - 1/2), written out independently
    and scaled to a unit diagonal."""
    m = onp.arange(n)
    m = onp.minimum(m, n - m)
    if onp.isinf(nu):
        power = onp.exp(-0.5 * (m * angle) ** 2)
    else:
        power = (1 + (m * angle) ** 2 / (2 * nu)) ** (-nu - 0.5)
    row = onp.real(onp.fft.ifft(power / power.mean()))
    return onp.array([onp.roll(row, k) for k in range(n)])


@pytest.mark.parametrize("nu", [0.5, 1.5, 2.5, onp.inf])
def test_covariance_is_the_radial_matern_times_the_multipolekernel_spectrum(
    nu,
):
    sigma, ell_r, ell_t = 1.7, 3.0, 0.8
    field = _field(sigma=sigma, length_r=ell_r, length_theta=ell_t, nu=nu)
    jac = jax.jacfwd(lambda z: field.set("latent", z).polar_values().ravel())(
        field.latent
    )
    jac = onp.asarray(jac).reshape(N_R * N_T, -1)
    r = onp.arange(N_R) * field.spacing
    d = onp.abs(r[:, None] - r[None]) / ell_r
    k_r = onp.exp(-0.5 * d * d) if onp.isinf(nu) else _matern(d, 1.0, nu)
    expected = sigma**2 * onp.kron(k_r, _azimuthal_covariance(N_T, ell_t, nu))
    # The prior is the plain separable one with order m of node k scaled by
    # tanh(r_k / spacing)^|m|: T_k is the circulant with that symbol.
    mm = onp.minimum(onp.arange(N_T), N_T - onp.arange(N_T))
    taper = onp.stack(
        [
            onp.real(
                onp.fft.ifft(
                    onp.tanh(rk / field.spacing) ** mm[:, None]
                    * onp.fft.fft(onp.eye(N_T), axis=0),
                    axis=0,
                )
            )
            for rk in r
        ]
    )
    block = onp.zeros((N_R * N_T, N_R * N_T))
    for k in range(N_R):
        block[k * N_T : (k + 1) * N_T, k * N_T : (k + 1) * N_T] = taper[k]
    expected = block @ expected @ block.T
    assert onp.allclose(jac @ jac.T, expected, atol=2e-5 * sigma**2)


@pytest.mark.parametrize("nu_theta", [0.5, 1.5, onp.inf])
def test_the_azimuthal_spectrum_is_multipolekernels(nu_theta):
    # One definition: the order-m variance of the field relative to order 1
    # is MultipoleKernel.spectrum with ell_theta = length_theta * r_ref.
    from virgil.fields import _azimuthal_filter
    from virgil.models import MultipoleKernel

    n, angle = 32, 0.7
    filt = _azimuthal_filter(np.eye(n), np.asarray(angle), nu_theta)
    cov = onp.asarray(filt @ filt.T, dtype=float)
    eigen = onp.real(onp.fft.fft(cov[0]))
    prior = MultipoleKernel(
        ell_r=1.0,
        ell_theta=angle * 5.0,
        r_in=4.0,
        r_out=6.0,
        nu_theta=nu_theta,
    )
    expected = onp.asarray(
        prior.spectrum(6)
    )  # orders 1..6, above float32 noise
    assert onp.allclose(eigen[1:7] / eigen[1], expected, rtol=1e-3)
    # A unit-variance filter: its diagonal is one.
    assert onp.allclose(onp.diag(cov), 1.0, atol=1e-5)


def test_latents_are_whitened_and_zero_latent_is_the_template():
    # An empirical check that standard-normal latents give variance sigma^2.
    sigma = 2.0
    field = _field(sigma=sigma)
    draws = onp.random.default_rng(1).normal(size=(400, N_R, N_T))
    values = jax.vmap(lambda z: field.set("latent", z).polar_values())(draws)
    assert float(np.std(values)) == pytest.approx(sigma, rel=0.1)
    template = onp.asarray(GaussianDisk(4.0).render(npix=24, fov_mas=24.0))
    flat = _field(mean=template, outside=0.0).set(
        "latent", np.zeros((N_R, N_T))
    )
    floor = 1e-3
    expected = onp.log(template / template.max() + floor)
    assert onp.allclose(flat.evaluate(1.0), expected)


def test_interpolation_is_second_order_and_exact_at_nodes():
    def smooth(r, th):
        return onp.cos(2 * th) * onp.exp(
            -0.5 * (r / 5.0) ** 2
        ) + 0.3 * onp.sin(th)

    rng = onp.random.default_rng(2)
    r = rng.uniform(1.0, 9.0, 400)
    th = rng.uniform(0.0, 2 * onp.pi, 400)

    def error(n_r, n_t):
        spacing = 10.0 / (n_r - 1)
        grid_r = onp.arange(n_r) * spacing
        grid_t = 2 * onp.pi * onp.arange(n_t) / n_t
        values = smooth(grid_r[:, None], grid_t[None, :])
        got = polar_interpolate(values, 0.0, spacing, r, th)
        return onp.max(onp.abs(onp.asarray(got) - smooth(r, th)))

    coarse, fine = error(21, 32), error(41, 64)
    assert fine < 0.3 * coarse and fine < 2e-2
    # Exact at the nodes, periodic in azimuth, zero well beyond the grid.
    values = onp.random.default_rng(3).normal(size=(5, 8))
    node = polar_interpolate(
        values, 1.0, 2.0, np.array(3.0), np.array(2 * onp.pi * 3 / 8)
    )
    assert float(node) == pytest.approx(values[1, 3], abs=1e-5)
    wrap = polar_interpolate(
        values, 1.0, 2.0, np.array([4.0] * 2), np.array([0.0, 2 * onp.pi])
    )
    assert float(wrap[0]) == pytest.approx(float(wrap[1]), abs=1e-5)
    beyond = polar_interpolate(
        values, 1.0, 2.0, np.array([11.5, 99.0]), np.array([1.0, 1.0])
    )
    assert onp.allclose(beyond, 0.0)


def test_pixels_follow_the_disk_frame():
    field = _field(shape=(25, 25), inc=60.0, pa=0.0)
    h = 1.0
    eta = onp.asarray(field.evaluate(h))
    offsets = onp.asarray(pixel_offsets(25, h))  # (centre - index) * h
    index = lambda offset: int(onp.argmin(onp.abs(offsets - offset)))
    # An inclined circle of radius a: the major axis is North (pa = 0), the
    # minor axis East-West, foreshortened by cos(inc) = 1/2.
    a = 8.0
    north = eta[index(a), index(0.0)]
    east = eta[index(0.0), index(a * 0.5)]
    west = eta[index(0.0), index(-a * 0.5)]
    # (East is x > 0, which is column index below the centre.)
    # The deprojected East point has r = a and azimuth 90 degrees.
    expected = polar_interpolate(
        field.polar_values(),
        0.0,
        field.spacing,
        np.array(a),
        np.array(onp.pi / 2),
    )
    assert float(east) == pytest.approx(float(expected), abs=1e-5)
    assert float(north) == pytest.approx(
        float(
            polar_interpolate(
                field.polar_values(),
                0.0,
                field.spacing,
                np.array(a),
                np.array(0.0),
            )
        ),
        abs=1e-5,
    )
    # West is at azimuth 270 degrees.
    assert float(west) == pytest.approx(
        float(
            polar_interpolate(
                field.polar_values(),
                0.0,
                field.spacing,
                np.array(a),
                np.array(1.5 * onp.pi),
            )
        ),
        abs=1e-5,
    )


def test_a_long_azimuthal_length_gives_an_axisymmetric_inclined_field():
    field = _field(shape=(25, 25), inc=60.0, pa=0.0, length_theta=1e4)
    offsets = onp.asarray(pixel_offsets(25, 1.0))
    eta = onp.asarray(field.evaluate(1.0))
    index = lambda offset: int(onp.argmin(onp.abs(offsets - offset)))
    assert eta[index(8.0), index(0.0)] == pytest.approx(
        eta[index(0.0), index(4.0)], abs=1e-3
    )


def test_visibilities_match_a_direct_dft_of_the_rendered_image():
    ps, wavel = 1.0, 2.0e-6
    image = Image(_field(), ps, flux=1.0)
    u = onp.array([5.0, -12.0, 20.0, 31.0])
    v = onp.array([8.0, 3.0, -17.0, 9.0])
    got = onp.asarray(image.model(u, v, wavel))
    brightness = onp.asarray(image.brightness)
    assert brightness.sum() == pytest.approx(1.0)
    # The rendering reproduces the pixels at their centres.
    rendered = onp.asarray(image.render(SHAPE[0], SHAPE[0] * ps))
    assert onp.allclose(rendered / rendered.sum(), brightness, atol=1e-6)
    x = onp.asarray(pixel_offsets(SHAPE[1], ps)) * mas2rad
    y = onp.asarray(pixel_offsets(SHAPE[0], ps)) * mas2rad
    phase = (
        -2j
        * onp.pi
        * (
            (u / wavel)[:, None, None] * x[None, None, :]
            + (v / wavel)[:, None, None] * y[None, :, None]
        )
    )
    expected = onp.sum(brightness * onp.exp(phase), axis=(1, 2))
    assert onp.allclose(got, expected, atol=1e-6)
    # The same field as plain pixels gives the same visibilities.
    plain = Image(image.eta, ps)
    assert onp.allclose(plain.model(u, v, wavel), got, atol=1e-6)


def test_gradients_are_finite_in_every_parameter():
    template = onp.asarray(GaussianDisk(4.0).render(npix=24, fov_mas=24.0))
    base = _field(mean=template)
    u, v = onp.array([20.0, -35.0]), onp.array([15.0, 40.0])
    names = [
        "latent", "sigma", "length_r", "length_theta",
        "inc", "pa", "dra", "ddec",
    ]  # fmt: skip

    def power(values):
        field = base.set(names, values)
        return np.sum(np.abs(Image(field, 1.0).model(u, v, 1.6e-6)) ** 2)

    values = [getattr(base, n) for n in names]
    grads = jax.grad(power)(values)
    for name, g in zip(names, grads):
        assert onp.all(onp.isfinite(onp.asarray(g))), name
        assert onp.any(onp.asarray(g) != 0.0), name


@pytest.mark.parametrize("inc", [0.0, 40.0])
def test_gradients_are_finite_with_a_pixel_on_the_origin(inc):
    # An odd grid with no offset puts a pixel exactly at the field's centre,
    # where hypot and arctan2 have NaN gradients unless made safe.
    base = _field(shape=(25, 25), inc=inc)
    names = ["latent", "inc", "pa", "dra", "ddec"]

    def total(values):
        return np.sum(base.set(names, values).evaluate(1.0) ** 2)

    grads = jax.grad(total)([getattr(base, n) for n in names])
    for name, g in zip(names, grads):
        assert onp.all(onp.isfinite(onp.asarray(g))), name


def test_changing_values_does_not_recompile():
    u, v = onp.array([20.0, -35.0]), onp.array([15.0, 40.0])

    @eqx.filter_jit
    def power(field):
        return np.sum(np.abs(Image(field, 1.0).model(u, v, 1.6e-6)) ** 2)

    base = _field()
    power(base)  # compile
    with count_compiles() as compiles:
        for k in range(3):
            changed = base.set(
                ["sigma", "length_r", "length_theta", "inc", "pa", "dra"],
                [
                    np.asarray(x, dtype=float)
                    for x in (
                        1.0 + k,
                        3.0 + k,
                        0.5 + k,
                        30.0 + k,
                        10.0 * k,
                        0.1 * k,
                    )
                ],
            )
            power(changed)
    assert compiles == []


def test_from_disk_copies_the_geometry():
    disk = MultipoleDisk(
        GaussianRing(radius=6.0, sigma=1.0),
        inc=35.0,
        pa=70.0,
        dra=0.5,
        ddec=-0.25,
    )
    field = MultipoleField.from_disk(disk, onp.zeros((6, 8)), SHAPE)
    assert float(field.inc) == 35.0 and float(field.pa) == 70.0
    assert float(field.dra) == 0.5 and float(field.ddec) == -0.25
    assert field.r_max_mas == pytest.approx(1.2 * disk.profile.extent()[1])


def test_hyperpriors_are_bounded_log_uniform():
    priors = multipole_field_priors("env.log_brightness", sigma=(0.1, 2.0))
    assert set(priors) == {
        "env.log_brightness.sigma",
        "env.log_brightness.length_r",
        "env.log_brightness.length_theta",
    }
    assert all(isinstance(p, dist.LogUniform) for p in priors.values())
    assert float(priors["env.log_brightness.sigma"].high) == 2.0
    with pytest.raises(ValueError):
        multipole_field_priors(sigma=(2.0, 1.0))


@pytest.mark.slow
def test_a_multipole_field_image_fits_and_has_a_finite_evidence():
    data = vlti_oidata(hour_angles_h=(-2.0, 0.0, 2.0), wavelengths_m=[3.5e-6])
    truth = System(
        star=PointSource(), env=GaussianDisk(3.0, dra=2.0, flux=0.4)
    )
    data = data.with_model(truth, key=jax.random.PRNGKey(0))
    template = onp.asarray(GaussianDisk(4.0).render(npix=24, fov_mas=24.0))
    field = _field(mean=template, inc=30.0, pa=10.0)
    start = System(star=PointSource(), env=Image(field, 1.0, flux=0.4))
    priors = image_priors(start)
    assert list(priors) == ["env.log_brightness.latent"]
    assert priors["env.log_brightness.latent"].event_shape == (N_R, N_T)
    result = fit(start, priors, data)
    first = fit(start, priors, data, max_steps=1)
    assert result.info["chi2_red"] < first.info["chi2_red"]
    assert onp.isfinite(log_evidence(result, data))


def test_fitting_multipole_field_scales_by_map_warns():
    start = System(star=PointSource(), env=Image(_field(), 1.0, flux=0.4))
    data = vlti_oidata(hour_angles_h=(0.0,), wavelengths_m=[3.5e-6])
    priors = image_priors(start) | multipole_field_priors()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fit(start, priors, data, max_steps=1)
    assert any(
        "hyperparameters of a Gaussian field" in str(w.message) for w in caught
    )


@pytest.mark.parametrize(
    "bad",
    [
        {"sigma": 0.0},
        {"length_r": -1.0},
        {"length_theta": 0.0},
        {"nu": 2.0},
        {"nu_theta": 2.0},
        {"geometry_from": ""},
        {"r_max_mas": 0.0},
        {"r_min_mas": 20.0},
        {"mean": onp.zeros((4, 4))},
        {"mean": -onp.ones(SHAPE)},
        {"image_shape": (24,)},
    ],
)
def test_bad_multipole_field_parameters_are_rejected(bad):
    kwargs = dict(
        latent=onp.zeros((N_R, N_T)), image_shape=SHAPE, r_max_mas=R_MAX
    )
    with pytest.raises(ValueError):
        MultipoleField(**(kwargs | bad))


def _independent_polar_coordinates(shape, ps, inc, pa):
    # r and azimuth of every pixel from first principles: the position
    # angle pa of the major axis is (sin pa, cos pa) in (East, North); the
    # azimuth runs from it towards pa + 90, (cos pa, -sin pa).
    n_r_, n_c = shape
    x = onp.asarray(pixel_offsets(n_c, ps))[None, :] * onp.ones((n_r_, 1))
    y = onp.asarray(pixel_offsets(n_r_, ps))[:, None] * onp.ones((1, n_c))
    p, i = onp.radians(pa), onp.radians(inc)
    major = x * onp.sin(p) + y * onp.cos(p)
    minor = (x * onp.cos(p) - y * onp.sin(p)) / onp.cos(i)
    return onp.hypot(major, minor), onp.arctan2(minor, major)


@pytest.mark.parametrize("pa", [0.0, 30.0, -50.0, 115.0])
def test_orientation_at_a_general_position_angle(pa):
    # A mirrored pa, or an azimuth towards pa - 90, would fail for pa != 0.
    field = _field(shape=(25, 25), inc=55.0, pa=pa)
    r, theta = _independent_polar_coordinates((25, 25), 1.0, 55.0, pa)
    expected = polar_interpolate(
        field.polar_values(), 0.0, field.spacing, r, theta, fill=-20.0
    )
    # The fade to -20 over one node spacing is steep: float32 noise only.
    assert onp.allclose(field.evaluate(1.0), expected, atol=2e-4)


def test_the_azimuth_agrees_with_multipole_disks_harmonics():
    # A field that is cos(azimuth) times a ring peaks where MultipoleDisk's
    # first linear harmonic (c = 1, s = 0) peaks, at the same inc and pa.
    class CosineField(MultipoleField):
        def polar_values(self):
            r = onp.arange(N_R) * self.spacing
            th = 2 * onp.pi * onp.arange(N_T) / N_T
            ring = onp.exp(-0.5 * ((r - 8.0) / 2.0) ** 2)
            return 6.0 * ring[:, None] * onp.cos(th)[None, :]

    shape, ps, inc, pa = (49, 49), 0.5, 50.0, 30.0
    field = CosineField(
        onp.zeros((N_R, N_T)), shape, R_MAX + 4.0, inc=inc, pa=pa
    )
    disk = MultipoleDisk(
        GaussianRing(8.0, 2.0),
        harmonics=Harmonics(c=0.9, s=0.0, form="linear"),
        inc=inc,
        pa=pa,
        n_r=64,
    )
    eta = onp.asarray(field.evaluate(ps))
    image = onp.asarray(disk.render(npix=49, fov_mas=49 * ps))
    assert onp.unravel_index(eta.argmax(), eta.shape) == onp.unravel_index(
        image.argmax(), image.shape
    )


def test_pixels_outside_the_polar_grid_are_dark():
    field = _field(shape=(48, 48), inc=40.0, pa=25.0)
    image = Image(field, 0.5, flux=1.0)
    r, _ = _independent_polar_coordinates((48, 48), 0.5, 40.0, 25.0)
    brightness = onp.asarray(image.brightness)
    outside = r > R_MAX + field.spacing
    assert outside.mean() > 0.2
    assert brightness[outside].sum() < 1e-6
    # The old zero padding would put this share of the flux outside.
    flat = _field(shape=(48, 48), inc=40.0, pa=25.0, outside=0.0)
    assert (
        onp.asarray(Image(flat, 0.5, flux=1.0).brightness)[outside].sum() > 0.1
    )


def test_a_long_azimuthal_length_leaves_only_the_axisymmetric_order():
    # Above about 2 radians the orders m >= 1 carry almost no variance.
    field = _field(length_theta=6.0)
    values = onp.asarray(field.polar_values())
    assert onp.std(values, axis=1).max() < 0.25 * onp.std(values)


def test_the_radial_factor_is_finite_in_float32():
    # 128 nodes of 0.3 mas and a Matérn 5/2 length of 50 mas: with a fixed
    # jitter of 1e-6 the float32 Cholesky factor is NaN.
    from virgil.fields import _radial_factor

    nodes = 0.3 * np.arange(128, dtype=float)
    factor = _radial_factor(nodes, np.asarray(50.0, dtype=float), 2.5)
    assert factor.dtype == nodes.dtype
    assert onp.all(onp.isfinite(onp.asarray(factor)))
    latent = onp.random.default_rng(0).normal(size=(128, 16))
    field = MultipoleField(
        latent,
        (32, 32),
        38.0,
        length_r=50.0,
        nu=2.5,
        inc=40.0,
    )
    assert onp.all(onp.isfinite(onp.asarray(field.evaluate(1.5))))


# ---------------------------------------------------------------------------
# Geometry tied to a disk
# ---------------------------------------------------------------------------

GEOMETRY = dict(inc=45.0, pa=30.0, dra=1.5, ddec=-1.0)


def _disk(**options):
    options = GEOMETRY | {"flux": 0.0} | options
    return MultipoleDisk(GaussianRing(radius=6.0, sigma=1.5), **options)


def _tied_scene(
    rotation_deg=0.0, image_dra=0.0, image_ddec=0.0, seed=0, shape=SHAPE
):
    disk = _disk()
    latent = onp.random.default_rng(seed).normal(size=(N_R, N_T))
    field = MultipoleField.from_disk(
        disk,
        latent,
        shape,
        r_max_mas=R_MAX,
        geometry_from="disk",
        length_r=3.0,
        length_theta=1.0,
    )
    image = Image(
        field,
        1.0,
        flux=0.5,
        dra=image_dra,
        ddec=image_ddec,
        rotation_deg=rotation_deg,
    )
    return System(star=PointSource(), disk=disk, env=image)


def _untied_like(scene, **geometry):
    """The same scene with the field's own, untied geometry."""
    field = scene.env.log_brightness.with_geometry(
        *(geometry[k] for k in ("inc", "pa", "dra", "ddec"))
    )
    return eqx.tree_at(lambda s: s.env.log_brightness, scene, field)


def test_a_tied_field_has_no_geometry_parameters():
    scene = _tied_scene()
    field = scene.env.log_brightness
    untied = _field()
    assert field.geometry_from == "disk"
    assert field.inc is None and field.pa is None
    assert field.dra is None and field.ddec is None
    # Four leaves (latent and the three scales) against eight untied.
    assert len(jax.tree.leaves(field)) == 4
    assert len(jax.tree.leaves(untied)) == 8
    assert list(image_priors(scene)) == ["env.log_brightness.latent"]


def test_the_default_is_untied_with_own_geometry():
    field = MultipoleField.from_disk(_disk(), onp.zeros((6, 8)), SHAPE)
    assert field.geometry_from is None and field.fallback == ()
    assert float(field.inc) == 45.0


def test_a_tied_field_follows_the_disk_when_its_geometry_changes():
    scene = _tied_scene()
    u, v = onp.array([20.0, -35.0, 8.0]), onp.array([15.0, 40.0, -12.0])
    wavel = 1.6e-6
    for values in (GEOMETRY, dict(inc=60.0, pa=-40.0, dra=-0.5, ddec=2.0)):
        moved = scene.set(
            ["disk.inc", "disk.pa", "disk.dra", "disk.ddec"],
            [np.asarray(values[k]) for k in ("inc", "pa", "dra", "ddec")],
        )
        expected = _untied_like(scene, **values)
        assert onp.allclose(
            moved.model(u, v, wavel), expected.model(u, v, wavel), atol=1e-6
        )
        assert onp.allclose(
            moved.render(32, 24.0), expected.render(32, 24.0), atol=1e-6
        )
    # And the answer does change with the disk's inclination and angle.
    other = scene.set(
        ["disk.inc", "disk.pa"], [np.asarray(70.0), np.asarray(80.0)]
    )
    assert not onp.allclose(other.model(u, v, wavel), scene.model(u, v, wavel))


def test_the_tied_geometry_is_differentiable_in_the_disks_parameters():
    scene = _tied_scene()
    u, v = onp.array([20.0, -35.0]), onp.array([15.0, 40.0])
    paths = ["disk.inc", "disk.pa", "disk.dra", "disk.ddec"]

    def power(values):
        return np.sum(
            np.abs(scene.set(paths, values).model(u, v, 1.6e-6)) ** 2
        )

    grads = jax.grad(power)(
        [np.asarray(GEOMETRY[k.split(".")[1]]) for k in paths]
    )
    for path, g in zip(paths, grads):
        assert onp.isfinite(float(g)) and float(g) != 0.0, path


def test_ties_account_for_the_images_rotation_and_offset():
    # The field's lit ellipse is the disk's on the sky, independent of the
    # Image's own rotation and offset: compare with an analytic ellipse.
    rot, d_ra, d_dec = 35.0, 6.0, -4.0
    scene = _tied_scene(
        rotation_deg=rot, image_dra=d_ra, image_ddec=d_dec, shape=(48, 48)
    )
    scene = scene.set("star.flux", np.asarray(0.0))
    # The ellipse is the dark-outside support of the zero-latent field.
    scene = eqx.tree_at(
        lambda s: s.env.log_brightness.latent,
        scene,
        np.zeros((N_R, N_T)),
    )
    npix, fov = 96, 48.0
    from virgil._geometry import image_coordinates

    xx, yy = image_coordinates(npix, fov)
    xs, ys = (
        onp.asarray(xx) - GEOMETRY["dra"],
        onp.asarray(yy) - GEOMETRY["ddec"],
    )
    p, i = onp.radians(GEOMETRY["pa"]), onp.radians(GEOMETRY["inc"])
    major = xs * onp.sin(p) + ys * onp.cos(p)
    minor = (xs * onp.cos(p) - ys * onp.sin(p)) / onp.cos(i)
    spacing = R_MAX / (N_R - 1)
    ellipse = onp.hypot(major, minor) < R_MAX + 0.5 * spacing
    assert ellipse.sum() > 200

    def overlap(model):
        lit = onp.asarray(model.render(npix, fov)) > 1e-4
        return (lit & ellipse).sum() / (lit | ellipse).sum()

    assert overlap(scene) > 0.9
    # The centre pins the sign of the rotation of the offset: the lit
    # centroid is the disk's (dra, ddec) to within a pixel, while the
    # opposite sign would put it more than three pixels away.
    pixel = fov / npix
    lit = onp.asarray(scene.render(npix, fov)) > 1e-4
    centroid = (onp.asarray(xx)[lit].mean(), onp.asarray(yy)[lit].mean())
    assert (
        onp.hypot(
            centroid[0] - GEOMETRY["dra"], centroid[1] - GEOMETRY["ddec"]
        )
        < pixel
    )
    angle = onp.radians(rot)
    off = onp.array([GEOMETRY["dra"] - d_ra, GEOMETRY["ddec"] - d_dec])
    wrong_centre = 2.0 * onp.hypot(*off) * onp.sin(angle)
    assert wrong_centre > 3 * pixel  # the +rot centre would be this far off
    # The same geometry without the Image's rotation and offset (the disk's
    # values taken as grid-frame ones) is a different, tilted ellipse.
    wrong = _untied_like(scene, **GEOMETRY)
    assert overlap(wrong) < 0.8


def test_resolved_bakes_the_tie_in():
    scene = _tied_scene()
    resolved = scene.resolved()
    field = resolved.env.log_brightness
    assert field.geometry_from is None
    assert float(field.inc) == 45.0 and float(field.pa) == 30.0
    assert float(field.dra) == pytest.approx(1.5)
    # Idempotent, and a System without ties is returned as it is.
    assert resolved.resolved() is resolved
    plain = System(star=PointSource(), env=Image(_field(), 1.0))
    assert plain.resolved() is plain
    # Pixels read directly (a regularizer, diagnostics) use the disk's values.
    from virgil.imaging import TSV

    moved = scene.set("disk.pa", np.asarray(-70.0))
    penalty = TSV(1.0, "env", beam_mas=1.0)
    direct = Image(moved.env.log_brightness, 1.0).brightness
    assert not onp.allclose(penalty.image(moved).brightness, direct, atol=1e-6)
    assert onp.allclose(
        penalty.image(moved).brightness,
        _untied_like(
            moved, inc=45.0, pa=-70.0, dra=1.5, ddec=-1.0
        ).env.brightness,
        atol=1e-6,
    )


def test_a_tie_must_name_a_sibling_with_geometry():
    field = _tied_scene().env.log_brightness
    lonely = System(star=PointSource(), env=Image(field, 1.0))
    with pytest.raises(ValueError, match="siblings"):
        lonely.render(16, 16.0)
    wrong = System(
        star=PointSource(), disk=GaussianDisk(2.0), env=Image(field, 1.0)
    )
    with pytest.raises(ValueError, match="no 'inc'"):
        wrong.render(16, 16.0)
    with pytest.raises(ValueError, match="concrete"):
        jax.jit(
            lambda inc: MultipoleField(
                onp.zeros((N_R, N_T)), SHAPE, R_MAX, inc=inc, geometry_from="d"
            ).latent
        )(1.0)


@pytest.mark.slow
def test_a_tied_field_fits_with_the_disk_geometry_and_has_a_finite_evidence():
    data = vlti_oidata(hour_angles_h=(-2.0, 0.0, 2.0), wavelengths_m=[3.5e-6])
    truth = _tied_scene(seed=3)
    data = data.with_model(truth, key=jax.random.PRNGKey(0))
    start = _tied_scene(seed=3).set(
        "env.log_brightness.latent", np.zeros((N_R, N_T))
    )
    priors = image_priors(start) | {"disk.inc": dist.Uniform(20.0, 70.0)}
    result = fit(start, priors, data, max_steps=5)
    # The disk's inclination moved, and it is the only inclination there is.
    assert float(result.model.disk.inc) != 45.0
    assert result.model.env.log_brightness.inc is None
    assert onp.isfinite(log_evidence(result, data))


def test_freeze_geometry_still_works_with_a_tied_field():
    data = vlti_oidata(hour_angles_h=(-2.0, 0.0, 2.0), wavelengths_m=[3.5e-6])
    scene = _tied_scene()
    frozen = eqx.tree_at(
        lambda s: s.disk, scene, scene.disk.freeze_geometry(data)
    )
    got = frozen.model(data.u, data.v, data.wavel)
    assert onp.allclose(
        got, scene.model(data.u, data.v, data.wavel), atol=1e-4
    )
    # Tying reads the (frozen) disk's geometry: moving it moves the field.
    moved = frozen.set("disk.pa", np.asarray(80.0))
    assert not onp.allclose(
        moved.model(data.u, data.v, data.wavel), got, atol=1e-4
    )


def test_a_tied_image_inside_a_wrapper_raises_instead_of_using_the_fallback():
    from virgil.models import Rotated

    field = _tied_scene().env.log_brightness
    wrapped = System(
        star=PointSource(),
        disk=_disk(),
        env=Rotated(Image(field, 1.0), 20.0),
    )
    with pytest.raises(ValueError, match="cannot be resolved"):
        wrapped.render(16, 16.0)


@pytest.mark.parametrize("nu", [0.5, 1.5, 2.5, onp.inf])
def test_the_gradient_in_the_azimuthal_length_is_finite_at_high_orders(nu):
    # exp(-0.5 (m length)^2) underflows at the highest orders for a long
    # length; the log-space filter keeps the derivative finite.
    from virgil.fields import _azimuthal_filter

    latent = np.asarray(onp.random.default_rng(0).normal(size=(4, 40)))

    def power(length):
        return np.sum(_azimuthal_filter(latent, length, nu) ** 2)

    assert onp.isfinite(float(jax.grad(power)(np.asarray(6.0))))


def test_the_multipole_kernel_spectrum_is_finite_for_a_long_angle():
    from virgil.models import MultipoleKernel

    for nu in (1.5, onp.inf):
        kernel = MultipoleKernel(
            1.0, 90.0, r_in=4.0, r_out=8.0, r_ref=6.0, nu_theta=nu
        )
        spectrum = onp.asarray(kernel.spectrum(4))
        assert onp.all(onp.isfinite(spectrum)) and spectrum[0] == 1.0


def test_the_field_is_single_valued_at_the_centre_and_stationary_away_from_it():
    n_r, n_theta = 12, 16
    z = onp.random.default_rng(0).normal(size=(600, n_r, n_theta))
    kw = dict(image_shape=(8, 8), r_max_mas=11.0, r_min_mas=0.0, sigma=2.0)
    vals = onp.stack(
        [onp.asarray(MultipoleField(zi, **kw).polar_values()) for zi in z]
    )
    # r = 0: no azimuthal variation.
    assert onp.abs(vals[:, 0, :] - vals[:, 0, :1]).max() < 1e-5
    # Per-order variance: gone at node 0 for m >= 1, tanh(r/spacing)^(2m)
    # at node 1, and unchanged eight spacings out (node 11 is the outermost
    # node, node 8 is eight spacings out).
    var = (onp.abs(onp.fft.fft(vals, axis=2)) ** 2).mean(axis=0) / n_theta
    assert var[0, 3] < 1e-8
    assert var[1, 3] / var[11, 3] == pytest.approx(
        onp.tanh(1.0) ** 6, rel=0.15
    )
    assert var[8, 3] / var[11, 3] == pytest.approx(1.0, rel=0.15)


def test_an_annulus_keeps_the_plain_prior():
    z = onp.random.default_rng(1).normal(size=(N_R, N_T))
    kw = dict(sigma=1.0, length_r=3.0, length_theta=1.0)
    ring = MultipoleField(z, SHAPE, R_MAX, r_min_mas=2.0, **kw)
    expected = ring.sigma * onp.asarray(
        onp.matmul(
            _radial_factor(
                ring.r_min_mas + ring.spacing * onp.arange(N_R), 3.0, 1.5
            ),
            _azimuthal_filter(z, 1.0, 1.5),
        )
    )
    assert onp.allclose(ring.polar_values(), expected, atol=1e-5)
