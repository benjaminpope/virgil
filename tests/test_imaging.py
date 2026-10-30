import equinox as eqx
import jax
import jax.numpy as np
import numpy as onp
import pytest

from virgil.coverage import ami_grid_record
from virgil.fitting import fit
from virgil.imaging import (
    TSV,
    Beam,
    beam,
    convolve_beam,
    dirty_image,
    field_of_view,
    TV,
    Centroid,
    LCurve,
    LogSum,
    MaxEntropy,
    StarletL1,
    image_priors,
    l_curve,
    nyquist_pixel_scale,
    starting_image,
)
from virgil.models import GaussianDisk, Image, PointSource, System
from virgil.oidata import OIData
from virgil.plotting import plot_model
from virgil.scenes import gaussian_blob

from ._compiles import count_compiles

NPIX, SCALE = 16, 12.0
DATA = OIData(ami_grid_record(pitch_m=0.5))


def _image(brightness, **kwargs):
    return Image.from_brightness(brightness, SCALE, **kwargs)


def test_tsv_and_tv_penalize_edges_and_structure():
    flat = _image(np.ones((NPIX, NPIX)))
    blob = _image(gaussian_blob(NPIX, SCALE, 15.0))
    # A flat image only has steps at its (zero-padded) edges.
    b = flat.brightness
    # TSV's default beam is the field of view, so its norm is NPIX**-4.
    assert np.isclose(
        TSV(1.0).value(flat), 4 * NPIX * b[0, 0] ** 2 * NPIX**4, rtol=1e-4
    )
    for regularizer in (TSV(2.0), TV(2.0)):
        assert regularizer.value(blob) > 0.0
        assert np.isclose(
            regularizer.value(blob), 2.0 * type(regularizer)(1.0).value(blob)
        )
    tsv = TSV(3.0)
    assert np.isclose(0.5 * np.sum(tsv.residuals(blob) ** 2), tsv.value(blob))


def _sky(npix):
    """The same Gaussian sky on a grid of ``npix`` pixels."""
    scale = SCALE * NPIX / npix
    return Image.from_brightness(gaussian_blob(npix, scale, 30.0), scale)


@pytest.mark.parametrize("regularizer", [TSV, TV])
def test_penalty_is_independent_of_grid_size(regularizer):
    values = [float(regularizer(1.0).value(_sky(n))) for n in (NPIX, 2 * NPIX)]
    assert values[1] == pytest.approx(values[0], rel=0.03)


class _Scaled(TSV):
    """TSV on pixels carrying total flux 3."""

    def pixels(self, model):
        return 3.0 * super().pixels(model)


class _ScaledTV(TV):
    def pixels(self, model):
        return 3.0 * super().pixels(model)


def test_penalty_is_independent_of_total_flux():
    sky = _sky(NPIX)
    assert float(_Scaled(1.0).value(sky)) == pytest.approx(
        float(TSV(1.0).value(sky)), rel=1e-5
    )
    assert float(_ScaledTV(1.0).value(sky)) == pytest.approx(
        float(TV(1.0).value(sky)), rel=1e-5
    )


def test_old_weights_convert_exactly():
    from virgil.imaging import _differences

    sky = _sky(NPIX)
    old = 7.0
    dx, dy = _differences(sky.brightness)
    legacy = old * float(np.sum(dx**2 + dy**2))
    ratio = 1.0 / NPIX  # p / B with the default beam, the field of view
    assert float(TSV(old * ratio**4).value(sky)) == pytest.approx(
        legacy, rel=1e-5
    )
    assert float(TSV(old, beam_mas=SCALE * NPIX).value(sky)) == pytest.approx(
        float(TSV(old).value(sky)), rel=1e-5
    )


def test_tv_has_no_preferred_direction():
    b = np.array([[1.0, 2.0], [3.0, 7.0]])
    values = [
        TV(1.0).value(_image(x))
        for x in (b, b[::-1], b[:, ::-1], b[::-1, ::-1])
    ]
    assert np.allclose(np.array(values), values[0], rtol=1e-5)


def test_max_entropy_residuals_square_to_its_value():
    blob = _image(gaussian_blob(NPIX, SCALE, 15.0))
    flat = _image(np.ones((NPIX, NPIX)))
    for reg in (
        MaxEntropy(3.0),
        MaxEntropy(3.0, prior=flat.brightness),
        MaxEntropy(3.0, prior=blob.brightness),
    ):
        r = reg.residuals(blob)
        assert np.all(np.isfinite(r))
        assert np.isclose(
            0.5 * np.sum(r**2), reg.value(blob), rtol=1e-4, atol=1e-5
        )
        # the gradient is finite where the image is exactly at the default
        grad = jax.grad(lambda m, g=reg: np.sum(g.residuals(m) ** 2))(blob)
        assert np.all(np.isfinite(grad.log_brightness))


def test_max_entropy_residuals_are_accurate_in_float32_near_the_default():
    q = np.ones((NPIX, NPIX)) / NPIX**2
    t = np.linspace(-0.02, 0.02, NPIX * NPIX).reshape(NPIX, NPIX)
    reg = MaxEntropy(3.0, prior=q)
    r = {}
    for dtype in (np.float32, np.float64):
        brightness = (q * (1.0 + t)).astype(dtype)
        r[dtype] = reg.residuals(_image(brightness))
    assert np.allclose(r[np.float32], r[np.float64], rtol=2e-3, atol=1e-7)


def test_max_entropy_is_zero_at_the_default_image():
    blob = _image(gaussian_blob(NPIX, SCALE, 15.0))
    assert np.isclose(
        MaxEntropy(1.0, prior=blob.brightness).value(blob), 0.0, atol=1e-5
    )
    assert MaxEntropy(1.0).value(blob) > 0.0
    flat = _image(np.ones((NPIX, NPIX)))
    assert np.isclose(MaxEntropy(1.0).value(flat), 0.0, atol=1e-5)


def test_centroid_measures_sky_offsets_with_rotation():
    npix = 9
    one = np.full((npix, npix), -np.inf).at[0, 4].set(0.0)  # top centre
    image = Image(one, 5.0, rotation_deg=90.0, dra=1.0)
    # The top of a frame rotated to PA 90 is due East: +x.
    assert np.allclose(
        Centroid(2.0).centroid(image), np.array([21.0, 0.0]), atol=1e-5
    )
    assert np.allclose(
        Centroid(2.0).residuals(image), np.array([10.5, 0.0]), atol=1e-5
    )


def test_regularizers_find_the_image_by_path():
    scene = System(star=PointSource(), env=_image(np.ones((NPIX, NPIX))))
    assert np.isclose(
        TSV(1.0, path="env").value(scene), TSV(1.0).value(scene.env)
    )
    with pytest.raises(TypeError, match="not an Image"):
        TSV(1.0, path="star").value(scene)


def test_image_priors_cover_every_image():
    scene = System(
        star=PointSource(),
        disk=_image(np.ones((NPIX, NPIX))),
        outer=System(ring=_image(np.ones((8, 8)))),
    )
    priors = image_priors(scene)
    assert set(priors) == {"disk.log_brightness", "outer.ring.log_brightness"}
    assert priors["outer.ring.log_brightness"].event_shape == (8, 8)
    with pytest.raises(ValueError, match="no Image"):
        image_priors(PointSource())


def test_nyquist_pixel_scale():
    longest = float(np.max(np.hypot(DATA.u, DATA.v)))
    expected = 4.3e-6 / (2 * longest) / (onp.pi / 180 / 3600 / 1000)
    assert 60.0 < expected < 90.0  # λ / 2B for B of about 6 m
    assert np.isclose(nyquist_pixel_scale(DATA), expected, rtol=1e-3)
    assert np.isclose(nyquist_pixel_scale([DATA, DATA]), expected, rtol=1e-3)


def test_translation_leaves_disco_data_unchanged():
    # DISCO phases are insensitive to shifts, and amplitudes too: rolling
    # an image by a pixel (away from its edges) changes nothing.
    blob = gaussian_blob(NPIX, SCALE, 12.0)
    image, rolled = _image(blob), _image(np.roll(blob, 1, axis=1))
    shifted = DATA.model(rolled) - DATA.model(image)
    assert np.max(np.abs(shifted)) < 0.1 * float(DATA.d_vis[0])  # float32
    assert (
        np.max(np.abs(np.angle(image.model(DATA.u, DATA.v, DATA.wavel))))
        < 1e-4
    )
    shift = np.angle(rolled.model(DATA.u, DATA.v, DATA.wavel))
    assert np.max(np.abs(shift)) > 0.05  # the raw phases do move


def test_centroid_prior_centres_an_image_only_fit():
    truth = _image(gaussian_blob(NPIX, SCALE, 25.0))
    data = DATA.with_model(truth, key=jax.random.PRNGKey(2))
    start = _image(gaussian_blob(NPIX, SCALE, 40.0, dra=30.0, ddec=-20.0))
    result = fit(
        start,
        image_priors(start),
        data,
        [TSV(1e3, beam_mas=SCALE), Centroid(1.0)],
        method="lm",
    )
    assert np.all(np.abs(Centroid(1.0).centroid(result.model)) < 3.0)


def test_a_star_anchors_the_image_position():
    # With an analytic star at the origin, an off-centre blob is recovered
    # in place without any centroid prior.
    truth = System(
        star=PointSource(),
        env=_image(
            gaussian_blob(NPIX, SCALE, 20.0, dra=36.0, ddec=24.0), flux=0.2
        ),
    )
    data = DATA.with_model(truth, key=jax.random.PRNGKey(3))
    start = System(
        star=PointSource(),
        env=Image.from_model(GaussianDisk(40.0), NPIX, SCALE, flux=0.2),
    )
    result = fit(
        start,
        image_priors(start),
        data,
        [TSV(1e3, path="env", beam_mas=SCALE)],
        method="lm",
    )
    centroid = Centroid(1.0, path="env").centroid(result.model)
    assert np.allclose(centroid, np.array([36.0, 24.0]), atol=6.0)


def test_l_curve_trades_chi2_against_the_penalty():
    truth = System(
        star=PointSource(),
        env=_image(gaussian_blob(NPIX, SCALE, 20.0, dra=24.0), flux=0.2),
    )
    data = DATA.with_model(truth, key=jax.random.PRNGKey(4))
    start = System(
        star=PointSource(), env=_image(np.ones((NPIX, NPIX)), flux=0.2)
    )
    curve = l_curve(
        start,
        image_priors(start),
        data,
        TSV(1.0, path="env", beam_mas=SCALE),
        [1e1, 1e3, 1e5],
        method="lm",
    )
    assert list(curve.weights) == [1e5, 1e3, 1e1]
    assert np.all(np.diff(curve.chi2) <= 1e-3 * curve.chi2[:-1])  # falls
    assert np.all(np.diff(curve.penalty) >= 0.0)  # rises


def test_corner_and_discrepancy_on_a_known_curve():
    # An "L": chi2 falls fast as w drops to 1, then flattens; the penalty
    # grows slowly, then fast.
    w = np.logspace(3, -3, 13)
    t = np.log10(w)
    chi2 = 100.0 * (1.0 + 10.0 ** np.clip(t, 0, None))
    penalty = 1.0 + 10.0 ** np.clip(-t, 0, None)
    curve = LCurve(w, chi2, chi2 / 100.0, penalty, [])
    assert 0.1 <= curve.corner() <= 10.0
    assert np.isclose(curve.discrepancy(target=11.0), 10.0, rtol=0.05)
    assert curve.discrepancy(target=0.5) is None
    # With two datasets, the worse-fitted one decides.
    joint = LCurve(
        w, chi2, np.stack([chi2 / 100.0, chi2 / 200.0], axis=1), penalty, []
    )
    assert np.isclose(joint.discrepancy(target=11.0), 10.0, rtol=0.05)


def _ring_of_baselines(radius_m, n=36, stretch=1.0):
    angle = onp.linspace(0.0, onp.pi, n, endpoint=False)
    u, v = radius_m * onp.cos(angle) * stretch, radius_m * onp.sin(angle)
    return OIData(
        {
            "u": u,
            "v": v,
            "wavel": 4.8e-6,
            "vis": onp.ones(n),
            "d_vis": onp.full(n, 0.01),
            "phi": onp.zeros(n),
            "d_phi": onp.full(n, 0.1),
        }
    )


def test_beam_of_a_ring_of_baselines_is_round_and_about_lambda_over_b():
    result = beam(_ring_of_baselines(6.0))
    b = 6.0 / 4.8e-6 * onp.pi / 180 / 3600 / 1000  # cycles per mas
    expected = 2 * onp.sqrt(2 * onp.log(2)) / (onp.pi * onp.sqrt(2) * b)
    assert np.isclose(result.major_mas, expected, rtol=1e-3)
    assert np.isclose(result.minor_mas, expected, rtol=1e-3)


def test_beam_is_long_across_the_long_baselines():
    # Baselines stretched East-West resolve finely East-West, so the beam's
    # major axis points North-South (PA 0 or 180).
    result = beam(_ring_of_baselines(4.0, stretch=2.0))
    assert result.major_mas > 1.5 * result.minor_mas
    assert min(result.pa_deg, 180.0 - result.pa_deg) < 1.0


def test_plot_model_draws_the_beam_in_the_lower_left():
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    plot_model(
        _image(np.ones((NPIX, NPIX))),
        fov_mas=NPIX * SCALE,
        npix=NPIX,
        ax=ax,
        beam=Beam(60.0, 30.0, 45.0),
    )
    (patch,) = ax.patches
    x, y = patch.center
    assert x > 0 and y < 0  # East (displayed left) and South (bottom)
    assert np.isclose(patch.width, 60.0) and np.isclose(patch.angle, 45.0)
    plt.close(fig)


def _point(npix=21):
    return np.zeros((npix, npix)).at[npix // 2, npix // 2].set(1.0)


def test_convolve_beam_keeps_flux_and_centre():
    smooth = convolve_beam(_point(), 1.0, Beam(4.0, 2.0, 30.0))
    assert np.isclose(smooth.sum(), 1.0, atol=1e-5)
    assert np.unravel_index(np.argmax(smooth), smooth.shape) == (10, 10)
    assert np.allclose(smooth, smooth[::-1, ::-1], atol=1e-7)  # unshifted


def test_convolve_beam_follows_north_to_east_pa():
    # Row 0 is North and column 0 is East. A beam at PA 45° spreads a
    # point to the North-East (up-left) and South-West, not North-West.
    smooth = convolve_beam(_point(), 1.0, Beam(6.0, 1.0, 45.0))
    assert smooth[8, 8] > 10 * smooth[8, 12]  # NE vs NW
    assert np.isclose(smooth[8, 8], smooth[12, 12])  # NE = SW
    # At PA 0° it spreads North-South (along a column).
    smooth = convolve_beam(_point(), 1.0, Beam(6.0, 1.0, 0.0))
    assert smooth[7, 10] > 10 * smooth[10, 7]


def test_convolve_beam_widths_are_the_fwhms():
    # Major axis East-West (PA 90°): the row's second moment is the major
    # axis's variance, and the column's the minor axis's.
    smooth = onp.asarray(convolve_beam(_point(41), 0.5, Beam(6.0, 3.0, 90.0)))
    offsets = (onp.arange(41) - 20) * 0.5
    sigma_per_fwhm = 1.0 / (2.0 * onp.sqrt(2.0 * onp.log(2.0)))
    across_row, down_column = smooth.sum(axis=0), smooth.sum(axis=1)
    assert np.isclose(
        onp.sqrt(across_row @ offsets**2), 6.0 * sigma_per_fwhm, rtol=1e-3
    )
    assert np.isclose(
        onp.sqrt(down_column @ offsets**2), 3.0 * sigma_per_fwhm, rtol=1e-3
    )


@pytest.mark.parametrize("npix", [48, 49])
def test_convolve_beam_of_a_gaussian_adds_variances(npix):
    # Even and odd grids: a round beam of FWHM f turns a Gaussian of σ into
    # one of √(σ² + (f / 2.355)²), in place. The field reaches 6σ, so
    # little flux is lost over the edge.
    fwhm, sigma, scale = 3.0, 1.5, 0.5
    sigma_beam = fwhm / (2.0 * onp.sqrt(2.0 * onp.log(2.0)))
    fov = npix * scale
    smooth = convolve_beam(
        GaussianDisk(sigma).render(npix, fov), scale, Beam(fwhm, fwhm, 0.0)
    )
    wider = GaussianDisk(onp.hypot(sigma, sigma_beam)).render(npix, fov)
    assert np.allclose(smooth, wider, atol=1e-4 * wider.max())


def test_convolve_beam_promotes_integer_images():
    # An integer or boolean image must not truncate the Gaussian kernel.
    point = _point()
    b = Beam(4.0, 2.0, 30.0)
    expected = convolve_beam(point, 1.0, b)
    for image in (point.astype(int), point.astype(bool)):
        smooth = convolve_beam(image, 1.0, b)
        assert np.issubdtype(smooth.dtype, np.floating)
        assert np.allclose(smooth, expected)


@pytest.mark.parametrize(
    "image, scale, b, match",
    [
        (np.ones((5, 5)), 1.0, Beam(2.0, 0.0, 0.0), "FWHMs"),
        (np.ones((5, 5)), 1.0, Beam(np.inf, 1.0, 0.0), "FWHMs"),
        (np.ones((5, 5)), 1.0, Beam(2.0, 1.0, np.nan), "PA"),
        (np.ones((5, 5)), 0.0, Beam(2.0, 1.0, 0.0), "pixel_scale"),
        (np.ones((5, 5)), np.nan, Beam(2.0, 1.0, 0.0), "pixel_scale"),
        (np.ones(5), 1.0, Beam(2.0, 1.0, 0.0), "2D"),
    ],
)
def test_convolve_beam_rejects_bad_geometry_and_images(image, scale, b, match):
    with pytest.raises(ValueError, match=match):
        convolve_beam(image, scale, b)


def test_plot_model_convolve_shows_the_convolved_image():
    import matplotlib.pyplot as plt

    disk = GaussianDisk(1.0)
    b = Beam(4.0, 2.0, 30.0)
    fig, ax = plt.subplots()
    plot_model(disk, fov_mas=16.0, npix=32, ax=ax, beam=b, convolve=True)
    expected = convolve_beam(disk.render(32, 16.0), 0.5, b)
    assert onp.allclose(ax.images[0].get_array().filled(), expected)
    with pytest.raises(ValueError, match="beam"):
        plot_model(disk, fov_mas=16.0, npix=32, ax=ax, convolve=True)
    plt.close(fig)


def test_field_of_view_is_at_most_500_mas_or_lambda_over_b_min():
    # A ring of 6 m baselines measures nothing larger than λ/B ~ 165 mas.
    assert np.isclose(
        field_of_view(_ring_of_baselines(6.0)),
        4.8e-6 / 6.0 / (onp.pi / 180 / 3600 / 1000),
        rtol=1e-3,
    )
    assert field_of_view(DATA) == 500.0  # AMI-like data reach much further


def test_starting_image_is_sized_from_the_data():
    truth = System(star=PointSource(), env=GaussianDisk(120.0, flux=0.1))
    data = DATA.with_model(truth, key=jax.random.PRNGKey(5))
    start = starting_image(data)
    env = start.env
    assert isinstance(start.star, PointSource) and isinstance(env, Image)
    assert np.isclose(env.flux, 0.1, rtol=0.1)
    fov = env.log_brightness.shape[0] * env.pixel_scale_mas
    assert fov >= 6 * 2.3548 * 120.0 * 0.8
    assert env.pixel_scale_mas <= nyquist_pixel_scale(data) / 4 + 1e-9
    # Without a star, the result is the Image alone.
    alone = starting_image(DATA.with_model(GaussianDisk(120.0)), star=False)
    assert isinstance(alone, Image)


def test_starting_image_keeps_the_0_3_positional_call_with_a_warning():
    data = DATA.with_model(GaussianDisk(120.0), key=jax.random.PRNGKey(5))
    kwargs = dict(
        star=False,
        oversample=3.0,
        largest_mas=300.0,
        start="moments",
        hole_mas=1.0,
    )
    new = starting_image(data, **kwargs)
    with pytest.warns(FutureWarning, match="star, oversample, largest_mas"):
        old = starting_image(data, False, 3.0, 300.0, "moments", 1.0)
    # Every positional argument landed on the 0.3 parameter of its place.
    assert jax.tree_util.tree_all(
        jax.tree_util.tree_map(
            lambda a, b: np.array_equal(a, b),
            eqx.filter(old, eqx.is_array),
            eqx.filter(new, eqx.is_array),
        )
    )
    assert old.pixel_scale_mas == new.pixel_scale_mas


def test_dirty_image_finds_a_companion_once_the_star_is_removed():
    scene = System(
        star=PointSource(), comp=PointSource(flux=0.1, dra=200.0, ddec=120.0)
    )
    data = DATA.with_model(scene)
    dirty = dirty_image(data, 64, 20.0, flux_ratio=0.1)
    row, col = np.unravel_index(np.argmax(dirty), dirty.shape)
    assert (
        abs(row - (31.5 - 120 / 20)) <= 1 and abs(col - (31.5 - 200 / 20)) <= 1
    )


def test_removing_the_star_leaves_no_residual_point_at_the_centre():
    # DISCO data are normalized on the shortest baselines, where a resolved
    # ring makes |V| < 1. Subtracting the best-fitting point source keeps
    # the star's residual small, although 1/flux_ratio amplifies any error.
    from virgil.scenes import ring

    image = ring(64, 20.0, radius_mas=240.0, width_mas=36.0, inc_deg=50.0)
    scene = System(
        star=PointSource(), dust=Image.from_brightness(image, 20.0, flux=0.05)
    )
    dirty = onp.asarray(
        dirty_image(DATA.with_model(scene), 64, 20.0, flux_ratio=0.05)
    )
    centre = dirty[30:34, 30:34].mean()
    on_ring = dirty[onp.asarray(image) > 0.5 * float(image.max())].mean()
    assert abs(centre) < 0.2 * on_ring


def test_dirty_image_uses_absolute_phases_and_refuses_closure_phases():
    rng = onp.random.default_rng(6)
    u, v = rng.uniform(-6.0, 6.0, (2, 200))
    point = PointSource(dra=-110.0, ddec=70.0)  # on a pixel centre
    cvis = onp.asarray(point.model(u, v, 4.8e-6))
    data = OIData(
        {
            "u": u,
            "v": v,
            "wavel": 4.8e-6,
            "vis": onp.abs(cvis),
            "d_vis": onp.full(200, 0.01),
            "phi": onp.angle(cvis),
            "d_phi": onp.full(200, 0.01),
            "v2_flag": False,
            "cp_flag": False,
        }
    )
    dirty = dirty_image(data, 32, 20.0)
    row, col = np.unravel_index(np.argmax(dirty), dirty.shape)
    assert (row, col) == (12, 21)  # 15.5 - 70 / 20, 15.5 + 110 / 20
    assert np.isclose(dirty.max(), 1.0, atol=0.05)
    from virgil.coverage import nrm_oidata

    with pytest.raises(ValueError, match="complex visibilities"):
        dirty_image(nrm_oidata(), 32, 20.0)


def test_a_dirty_start_is_a_positive_image():
    truth = System(
        star=PointSource(),
        env=_image(gaussian_blob(NPIX, SCALE, 40.0, dra=30.0), flux=0.1),
    )
    data = DATA.with_model(truth, key=jax.random.PRNGKey(7))
    start = starting_image(data, start="dirty")
    assert np.all(start.env.brightness > 0.0)
    moments = starting_image(data)
    assert start.env.log_brightness.shape == moments.env.log_brightness.shape


@pytest.mark.filterwarnings("ignore:fit\\(method=:RuntimeWarning")
@pytest.mark.parametrize(
    "regularizers, method",
    [
        (lambda v: [TSV(1e2, beam_mas=SCALE), Centroid(v)], "lm"),
        (
            lambda v: [
                TV(1e2, epsilon=v * 1e-3, beam_mas=SCALE),
                Centroid(5.0),
            ],
            "lbfgs",
        ),
        (lambda v: [StarletL1(1.0, epsilon=v * 1e-3)], "lbfgs"),
        (lambda v: [LogSum(1e-3, epsilon=v * 1e-3)], "lbfgs"),
    ],
)
def test_new_centroid_widths_and_tv_smoothing_do_not_recompile(
    regularizers, method
):
    # sigma_mas and epsilon are arrays, like weight, so they are traced:
    # a new value reuses the fit's compilation. As Python floats they were
    # static, and every value recompiled the whole fit.
    truth = _image(gaussian_blob(NPIX, SCALE, 25.0))
    data = DATA.with_model(truth, key=jax.random.PRNGKey(4))
    start = _image(np.ones((NPIX, NPIX)) / NPIX**2)
    priors = image_priors(start)

    def run(value):
        fit(
            start,
            priors,
            data,
            regularizers(value),
            method=method,
            max_steps=5,
        )

    run(5.0)
    with count_compiles() as compiles:
        run(7.0)
    assert not compiles


def test_dirty_image_in_chunks_matches_the_dense_synthesis(monkeypatch):
    # The chunked sum equals the all-samples einsum it replaced, whatever
    # the chunk size (including one that leaves a short last chunk).
    from virgil import imaging
    from virgil._utils import mas2rad

    rng = onp.random.default_rng(7)
    u, v = rng.uniform(-6.0, 6.0, (2, 203))
    cvis = onp.asarray(PointSource(dra=-50.0, ddec=30.0).model(u, v, 4.8e-6))
    data = OIData(
        {
            "u": u,
            "v": v,
            "wavel": 4.8e-6,
            "vis": onp.abs(cvis),
            "d_vis": onp.full(203, 0.01),
            "phi": onp.angle(cvis),
            "d_phi": onp.full(203, 0.01),
            "v2_flag": False,
            "cp_flag": False,
        }
    )
    npix, scale = 24, 20.0
    offsets = onp.asarray(imaging.pixel_offsets(npix, scale))
    fu, fv = u / 4.8e-6 * mas2rad, v / 4.8e-6 * mas2rad
    cols = onp.exp(2j * onp.pi * onp.outer(fu, offsets))
    rows = onp.exp(2j * onp.pi * onp.outer(fv, offsets))
    dense = onp.real(onp.einsum("k,kr,kc->rc", cvis, rows, cols)) / 203
    for batch in (1, 50, 203, 4096):
        monkeypatch.setattr(imaging, "_DIRTY_BATCH", batch)
        got = onp.asarray(dirty_image(data, npix, scale))
        onp.testing.assert_allclose(got, dense, rtol=1e-5, atol=1e-6)


def test_a_dirty_start_padded_beyond_the_field_keeps_the_natural_block():
    truth = System(
        star=PointSource(),
        env=_image(gaussian_blob(NPIX, SCALE, 40.0, dra=30.0), flux=0.1),
    )
    data = DATA.with_model(truth, key=jax.random.PRNGKey(7))
    natural = starting_image(data, start="dirty")
    n = natural.env.log_brightness.shape[0]
    wide = starting_image(
        data, start="dirty", pad_to_mas=3.0 * n * natural.env.pixel_scale_mas
    )
    m = wide.env.log_brightness.shape[0]
    assert m >= 3 * n and (m - n) % 2 == 0
    margin = (m - n) // 2
    centre = wide.env.log_brightness[margin : margin + n, margin : margin + n]
    assert np.allclose(centre, natural.env.log_brightness, atol=1e-6)


def test_evidence_and_laplace_account_for_non_pixel_priors_and_centroid():
    import numpyro.distributions as dist

    from virgil.fields import GaussianField
    from virgil.imaging import error_scale, laplace_samples, log_evidence
    from virgil.likelihood import whitened_residuals

    n = 8
    field = GaussianField(onp.zeros((n, n)), 1.5, 24.0)
    scene = System(star=PointSource(), env=Image(field, SCALE, flux=0.3))
    data = DATA.with_model(scene, key=jax.random.PRNGKey(9))
    base_priors = image_priors(scene)
    full_priors = base_priors | {"env.flux": dist.LogUniform(1e-3, 10.0)}
    result = fit(scene, full_priors, data, max_steps=5)

    ev_fixed = float(log_evidence(result, data, priors=base_priors))
    ev_free = float(log_evidence(result, data, priors=full_priors))
    # Freeing env.flux under a wide LogUniform prior adds an Occam penalty.
    assert ev_free < ev_fixed
    # Pin the exact Laplace formula (-½χ² - L - ½|z|² + log π(f̂) + ½ log det 2πΣ)
    # against an explicit full-block Hessian H = diag(I_k, 0) + JᵀJ in float64.
    with jax.enable_x64(True):
        m64 = jax.tree_util.tree_map(
            lambda x: (
                np.asarray(x, np.float64)
                if hasattr(x, "dtype")
                and onp.issubdtype(x.dtype, onp.floating)
                else x
            ),
            result.model,
        )
        d64 = jax.tree_util.tree_map(
            lambda x: (
                np.asarray(x, np.float64)
                if hasattr(x, "dtype")
                and onp.issubdtype(x.dtype, onp.floating)
                else x
            ),
            data,
        )
        z0 = m64.get("env.log_brightness.latent").reshape(-1)
        f0 = np.asarray([m64.get("env.flux")], np.float64)
        theta0 = np.concatenate([z0, f0])

        def resid_zf(theta):
            m = m64.set(
                ["env.log_brightness.latent", "env.flux"],
                [theta[: n * n].reshape(n, n), theta[n * n]],
            )
            return whitened_residuals(m, d64)

        r0 = onp.asarray(resid_zf(theta0), float)
        jac = onp.asarray(jax.jacfwd(resid_zf)(theta0), float)
    prior_prec = onp.diag([1.0] * (n * n) + [0.0])
    hess = prior_prec + jac.T @ jac
    log_pi_f = float(full_priors["env.flux"].log_prob(float(f0[0])))
    expected_ev_free = (
        ev_fixed
        + 0.5
        * float(
            onp.linalg.slogdet(
                onp.eye(n * n) + jac[:, : n * n].T @ jac[:, : n * n]
            )[1]
        )
        + log_pi_f
        + 0.5 * float(onp.log(2.0 * onp.pi))
        - 0.5 * float(onp.linalg.slogdet(hess)[1])
    )
    assert ev_free == pytest.approx(expected_ev_free, rel=1e-10)
    scale_free = float(error_scale(result, data, priors=full_priors))
    beta = 1.0 / scale_free**2
    gamma_beta = float(
        onp.trace(
            onp.linalg.solve(
                prior_prec + beta * (jac.T @ jac), beta * (jac.T @ jac)
            )
        )
    )
    assert beta * float(r0 @ r0) + gamma_beta == pytest.approx(
        float(data.n_independent), rel=1e-5
    )
    # Unit-rescaling invariance: scaling both star.flux and env.flux by 10
    # (with a 10x LogUniform prior on env.flux) leaves the evidence and
    # error_scale invariant.
    scaled_model = result.model.set(
        ["star.flux", "env.flux"],
        [10.0, 10.0 * float(result.model.env.flux)],
    )
    scaled_priors = base_priors | {"env.flux": dist.LogUniform(1e-2, 100.0)}
    assert float(
        log_evidence(scaled_model, data, priors=scaled_priors)
    ) == pytest.approx(ev_free, rel=1e-6)
    assert float(
        error_scale(scaled_model, data, priors=scaled_priors)
    ) == pytest.approx(scale_free, rel=1e-6)
    assert scale_free != pytest.approx(
        float(error_scale(result, data, priors=base_priors))
    )
    draws = laplace_samples(
        result, data, 6, jax.random.PRNGKey(0), priors=full_priors
    )
    assert draws["latents"].shape == (6, n, n) and onp.all(
        onp.isfinite(draws["latents"])
    )

    # Probabilistic regularizer (Centroid) contributes to log_evidence:
    # check against explicit augmented residual and Jacobian in float64.
    reg = Centroid(5.0)
    img = Image(GaussianField(onp.zeros((n, n)), 1.5, 24.0), SCALE)
    fit_img = fit(img, image_priors(img), data, [reg], max_steps=5)
    ev_no_reg = float(log_evidence(fit_img, data, path=None))
    ev_reg = float(log_evidence(fit_img, data, path=None, regularizers=(reg,)))
    assert ev_reg < ev_no_reg
    with jax.enable_x64(True):
        im64 = jax.tree_util.tree_map(
            lambda x: (
                np.asarray(x, np.float64)
                if hasattr(x, "dtype")
                and onp.issubdtype(x.dtype, onp.floating)
                else x
            ),
            fit_img.model,
        )
        z_im = im64.get("log_brightness.latent").reshape(-1)

        def resid_c(z):
            m = im64.set("log_brightness.latent", z.reshape(n, n))
            return np.concatenate(
                [whitened_residuals(m, d64), reg.residuals(m)]
            )

        rc = onp.asarray(resid_c(z_im), float)
        jc = onp.asarray(jax.jacfwd(resid_c)(z_im), float)
    rc_data = rc[:-2]
    jc_data = jc[:-2]
    jc_reg = jc[-2:]
    expected_diff = -0.5 * float(rc @ rc - rc_data @ rc_data) - 0.5 * float(
        onp.linalg.slogdet(onp.eye(n * n) + jc.T @ jc)[1]
        - onp.linalg.slogdet(onp.eye(n * n) + jc_data.T @ jc_data)[1]
    )
    assert (ev_reg - ev_no_reg) == pytest.approx(expected_diff, rel=1e-10)
    scale_reg = float(
        error_scale(fit_img, data, path=None, regularizers=(reg,))
    )
    beta_reg = 1.0 / scale_reg**2
    gamma_reg = float(
        onp.trace(
            onp.linalg.solve(
                onp.eye(n * n)
                + jc_reg.T @ jc_reg
                + beta_reg * (jc_data.T @ jc_data),
                beta_reg * (jc_data.T @ jc_data),
            )
        )
    )
    assert beta_reg * float(rc_data @ rc_data) + gamma_reg == pytest.approx(
        float(data.n_independent), rel=1e-5
    )


def _tiny_gp_scene():
    """A 1×2 latent field plus a free envelope flux: three parameters."""
    import numpyro.distributions as dist

    from virgil.coverage import vlti_oidata
    from virgil.fields import GaussianField

    def scene(latent):
        field = GaussianField(latent, 1.0, 12.0)
        return System(star=PointSource(), env=Image(field, 6.0, flux=3.0))

    template = vlti_oidata(
        hour_angles_h=(-1.0, 1.0),
        wavelengths_m=[3.5e-6],
        sigma_v2=0.03,
        sigma_cp_deg=3.0,
    )
    data = template.with_model(
        scene(onp.array([[0.8, -0.5]])), key=jax.random.PRNGKey(1)
    )
    model = scene(onp.zeros((1, 2)))
    priors = image_priors(model) | {"env.flux": dist.LogUniform(1e-2, 1e2)}
    return fit(model, priors, data, max_steps=200), data, priors


def test_log_evidence_matches_numerical_integration_with_a_free_flux():
    from scipy.integrate import trapezoid

    from virgil._precision import cast_tree, run_in
    from virgil.imaging import _log_norm, log_evidence
    from virgil.likelihood import whitened_residuals

    result, data, priors = _tiny_gp_scene()
    laplace = float(log_evidence(result, data, priors=priors))
    paths = ["env.log_brightness.latent", "env.flux"]
    with run_in("float64"):
        model, data64 = cast_tree((result.model, data), "float64")

        def chi2(theta):
            m = model.set(paths, [theta[:2].reshape(1, 2), np.exp(theta[2])])
            return np.sum(whitened_residuals(m, data64) ** 2)

        def neg_log_post(theta):
            return 0.5 * chi2(theta) + 0.5 * np.sum(theta[:2] ** 2)

        z = onp.asarray(model.get(paths[0]), float).ravel()
        theta0 = np.array([z[0], z[1], onp.log(float(model.get(paths[1])))])
        # Grid: ±8 posterior standard deviations in (z, log flux) about the MAP.
        sd = onp.sqrt(
            onp.diag(
                onp.linalg.inv(onp.asarray(jax.hessian(neg_log_post)(theta0)))
            )
        )
        axes = [
            onp.linspace(theta0[i] - 8 * sd[i], theta0[i] + 8 * sd[i], 21)
            for i in range(3)
        ]
        cube = onp.meshgrid(*axes, indexing="ij")
        points = np.asarray(onp.stack([c.ravel() for c in cube], 1))
        chi2s = onp.asarray(jax.vmap(chi2)(points)).reshape(cube[0].shape)
        log_norm = _log_norm(model, data64)
    low, high = 1e-2, 1e2
    # N(z; 0, I) × LogUniform(f) df, with df = f d(log f).
    log_prior = (
        -0.5 * (cube[0] ** 2 + cube[1] ** 2)
        - onp.log(2.0 * onp.pi)
        - onp.log(onp.log(high / low))
    )
    log_integrand = -0.5 * chi2s - log_norm + log_prior
    shift = log_integrand.max()
    integral = onp.exp(log_integrand - shift)
    for axis, x in reversed(list(enumerate(axes))):
        integral = trapezoid(integral, x, axis=axis)
    exact = shift + float(onp.log(integral))
    # The Laplace error is the posterior's non-Gaussianity: 0.012 nats here.
    # Putting the flux in the unit-Gaussian latents instead gives 1.9 nats off.
    assert laplace == pytest.approx(exact, abs=0.05)


def test_laplace_samples_have_the_marginal_covariance_of_the_full_hessian():
    from virgil.imaging import laplace_samples
    from virgil.likelihood import whitened_residuals

    result, data, priors = _tiny_gp_scene()
    paths = ["env.log_brightness.latent", "env.flux"]
    with jax.enable_x64(True):
        model = result.model

        def residuals(theta):
            m = model.set(paths, [theta[:2].reshape(1, 2), theta[2]])
            return whitened_residuals(m, data)

        z = onp.asarray(model.get(paths[0]), float).ravel()
        theta0 = np.array([z[0], z[1], float(model.get(paths[1]))])
        jac = onp.asarray(jax.jacfwd(residuals)(theta0), float)
    hessian = onp.diag([1.0, 1.0, 0.0]) + jac.T @ jac
    marginal = onp.linalg.inv(hessian)[:2, :2]
    conditional = onp.linalg.inv(hessian[:2, :2])
    n = 40000
    draws = onp.asarray(
        laplace_samples(result, data, n, jax.random.PRNGKey(0), priors=priors)[
            "latents"
        ]
    ).reshape(n, 2)
    sample = onp.cov(draws.T)
    # The sampling error of a variance is √(2/n) = 0.7%; the flux's
    # correlation with the field moves the variance by 4.5%.
    onp.testing.assert_allclose(
        onp.diag(sample), onp.diag(marginal), rtol=0.025
    )
    assert abs(sample[1, 1] / conditional[1, 1] - 1.0) > 0.025
