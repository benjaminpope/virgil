"""MultipoleDisk (degree 0): the quadrature engine, its profiles and conventions."""

from types import SimpleNamespace

import equinox as eqx
import jax
import jax.numpy as np
import numpy as onp
import pytest
from scipy.special import j0 as scipy_j0

from virgil import models
from virgil._geometry import image_coordinates
from virgil.models import (
    ExpTail,
    Gap,
    GaussianDisk,
    GaussianRing,
    ModulatedGaussianRim,
    PointSource,
    MultipoleDisk,
    RadialPowerLaw,
    RadialSum,
    System,
    ThinRing,
    UniformDisk,
    multipole_disk_cvis,
)

# Independent reference conversion, as in test_models_sources.py.
_MAS2RAD_REF = onp.pi / 180.0 / 3600.0 / 1000.0
WAVEL = 1.65e-6


def _uv(n=40, seed=0, scale=2.5e7):
    """Random baselines (metres) out to ``scale`` cycles/rad at WAVEL."""
    rng = onp.random.default_rng(seed)
    return (
        onp.asarray(rng.uniform(-scale, scale, n)) * WAVEL,
        onp.asarray(rng.uniform(-scale, scale, n)) * WAVEL,
    )


def _cvis(model, u, v):
    return onp.asarray(model.model(np.asarray(u), np.asarray(v), WAVEL))


def _q_rad(u, v):
    return onp.hypot(u, v) / WAVEL


# --- analytic checks -------------------------------------------------------


@pytest.mark.validates("virgil.models.MultipoleDisk", roots=["mathematics"])
def test_sharp_power_law_is_a_uniform_disk():
    u, v = _uv()
    disk = MultipoleDisk(RadialPowerLaw(0.0, 5.0), n_r=32)
    expected = _cvis(UniformDisk(diam=10.0), u, v)
    assert onp.max(onp.abs(_cvis(disk, u, v) - expected)) < 2e-5


@pytest.mark.validates("virgil.models.MultipoleDisk", roots=["mathematics"])
def test_gaussian_ring_at_the_centre_is_a_gaussian_disk():
    u, v = _uv(scale=8e6)
    disk = MultipoleDisk(GaussianRing(0.0, 3.0), n_r=48)
    expected = _cvis(GaussianDisk(sigma=3.0), u, v)
    assert onp.max(onp.abs(_cvis(disk, u, v) - expected)) < 2e-5


@pytest.mark.validates("virgil.models.MultipoleDisk", roots=["mathematics"])
def test_thin_ring_is_a_bessel_function():
    u, v = _uv()
    cvis = _cvis(MultipoleDisk(ThinRing(4.0)), u, v)
    expected = scipy_j0(2.0 * onp.pi * _q_rad(u, v) * 4.0 * _MAS2RAD_REF)
    assert onp.max(onp.abs(cvis - expected)) < 2e-6
    assert onp.max(onp.abs(cvis.imag)) == 0.0


@pytest.mark.validates("virgil.models.MultipoleDisk", roots=["mathematics"])
def test_exponential_disk_matches_the_closed_form():
    # int exp(-r/L) J0(a r / L) r dr / int exp(-r/L) r dr = (1 + a^2)^(-3/2).
    scale = 5.0
    q = onp.linspace(0.0, 2.0 / (2.0 * onp.pi * _MAS2RAD_REF * scale), 80)
    disk = MultipoleDisk(ExpTail(0.0, scale), n_r=48)
    cvis = onp.asarray(disk._centred_cvis(np.asarray(q), np.zeros(q.shape)))
    a = 2.0 * onp.pi * q * _MAS2RAD_REF * scale
    assert onp.max(onp.abs(cvis.real - (1.0 + a**2) ** -1.5)) < 1e-5


def test_gaussian_blur_is_a_fourier_envelope():
    u, v = _uv()
    fwhm = 3.0
    blurred = _cvis(MultipoleDisk(ThinRing(6.0), blur_fwhm=fwhm), u, v)
    q = _q_rad(u, v)
    envelope = onp.exp(
        -((onp.pi * fwhm * _MAS2RAD_REF * q) ** 2) / (4.0 * onp.log(2.0))
    )
    ring = scipy_j0(2.0 * onp.pi * q * 6.0 * _MAS2RAD_REF)
    assert onp.max(onp.abs(blurred - ring * envelope)) < 2e-6


def test_thin_ring_in_a_sum_is_a_weighted_sum_of_bessel_functions():
    u, v = _uv()
    profile = RadialSum(a=ThinRing(3.0), b=(ThinRing(8.0), 3.0))
    cvis = _cvis(MultipoleDisk(profile), u, v)
    x = 2.0 * onp.pi * _q_rad(u, v) * _MAS2RAD_REF
    expected = 0.25 * scipy_j0(x * 3.0) + 0.75 * scipy_j0(x * 8.0)
    assert onp.max(onp.abs(cvis - expected)) < 2e-6


def test_sum_normalizes_each_part_to_unit_light():
    # A wide part must not outweigh a narrow one just because it has more
    # unnormalized area: weights are shares of the light.
    wide = RadialPowerLaw(0.0, 12.0)
    ring = ThinRing(4.0)
    both = MultipoleDisk(RadialSum(wide=wide, ring=ring), n_r=32)
    u, v = _uv()
    wide_only = _cvis(MultipoleDisk(wide, n_r=32), u, v)
    ring_only = _cvis(MultipoleDisk(ring), u, v)
    assert onp.allclose(
        _cvis(both, u, v), 0.5 * (wide_only + ring_only), atol=2e-6
    )


def test_gap_depth_zero_is_the_base_and_a_gap_removes_light():
    u, v = _uv()
    base = GaussianRing(10.0, 3.0)
    plain = _cvis(MultipoleDisk(base, n_r=64), u, v)
    shallow = Gap(base, radius=10.0, sigma=0.8, depth=0.0)
    assert onp.allclose(
        _cvis(MultipoleDisk(shallow, n_r=64), u, v), plain, atol=2e-6
    )

    # The gap's visibility is the base's minus the gap's own ring of
    # Gaussian shape, which has a closed form for a tophat base:
    # int_0^R (1 - d g(r)) J0 r dr, checked against a brute-force sum.
    gap = Gap(RadialPowerLaw(0.0, 10.0), radius=5.0, sigma=0.7, depth=0.6)
    cvis = _cvis(MultipoleDisk(gap, n_r=128), u, v)
    r = onp.linspace(0.0, 10.0, 200001)[1:]
    weight = r * (1.0 - 0.6 * onp.exp(-0.5 * ((r - 5.0) / 0.7) ** 2))
    q = _q_rad(u, v)
    brute = onp.array(
        [
            onp.trapezoid(
                weight * scipy_j0(2 * onp.pi * qi * r * _MAS2RAD_REF), r
            )
            for qi in q
        ]
    ) / onp.trapezoid(weight, r)
    assert onp.max(onp.abs(cvis - brute)) < 2e-5


def test_soft_edges_approach_sharp_edges_and_are_smooth():
    u, v = _uv()
    sharp = _cvis(MultipoleDisk(RadialPowerLaw(2.0, 9.0, -1.0), n_r=64), u, v)
    soft = _cvis(
        MultipoleDisk(
            RadialPowerLaw(2.0, 9.0, -1.0, edge_sigma=0.05), n_r=512
        ),
        u,
        v,
    )
    assert onp.max(onp.abs(soft - sharp)) < 5e-3


# --- normalization, gradients, composition ----------------------------------


def _profiles():
    return {
        "thin": ThinRing(5.0),
        "gauss": GaussianRing(5.0, 1.2),
        "power": RadialPowerLaw(1.0, 8.0, -1.5),
        "power_soft": RadialPowerLaw(1.0, 8.0, 0.5, edge_sigma=0.4),
        "exp": ExpTail(1.0, 2.0),
        "gap": Gap(GaussianRing(6.0, 2.0), radius=6.0, sigma=0.6, depth=0.7),
        "sum": RadialSum(a=GaussianRing(3.0, 0.5), b=(ExpTail(4.0, 1.0), 0.3)),
    }


@pytest.mark.parametrize("name", list(_profiles()))
def test_zero_baseline_visibility_is_one(name):
    disk = MultipoleDisk(_profiles()[name], inc=55.0, pa=20.0, blur_fwhm=1.0)
    zero = np.zeros(3)
    cvis = disk.model(zero, zero, WAVEL)
    assert onp.allclose(onp.asarray(cvis), 1.0, atol=1e-6)


@pytest.mark.parametrize("name", list(_profiles()))
def test_gradients_are_finite_including_at_zero_baseline(name):
    disk = MultipoleDisk(
        _profiles()[name], inc=55.0, pa=20.0, blur_fwhm=1.0, dra=1.0, ddec=-2.0
    )
    u = np.concatenate([np.zeros(2), np.asarray(_uv(8)[0])])
    v = np.concatenate([np.zeros(2), np.asarray(_uv(8)[1])])

    def loss(model):
        return np.sum(np.abs(model.model(u, v, WAVEL)) ** 2)

    grads = eqx.filter_grad(loss)(disk)
    leaves = [
        leaf for leaf in jax.tree_util.tree_leaves(grads) if leaf is not None
    ]
    assert leaves
    assert all(bool(np.all(np.isfinite(leaf))) for leaf in leaves)
    # The parameters that matter do move the answer.
    assert abs(float(grads.inc)) > 0.0 or name == "thin"


def test_face_on_gradient_is_finite_at_extreme_inclinations():
    for inc in (0.0, 89.99, 90.0):
        disk = MultipoleDisk(GaussianRing(5.0, 1.0), inc=inc, pa=10.0)
        u, v = np.asarray(_uv(6)[0]), np.asarray(_uv(6)[1])
        grads = eqx.filter_grad(
            lambda m: np.sum(np.abs(m.model(u, v, WAVEL)) ** 2)
        )(disk)
        assert onp.isfinite(float(grads.inc))
        assert onp.isfinite(float(grads.profile.radius))


def test_system_mixes_a_disk_like_any_component():
    u, v = _uv()
    disk = MultipoleDisk(GaussianRing(6.0, 1.0), inc=30.0, pa=70.0, flux=0.25)
    scene = System(star=PointSource(), disk=disk)
    cvis = _cvis(scene, u, v)
    expected = (1.0 + 0.25 * _cvis(disk, u, v)) / 1.25
    assert onp.allclose(cvis, expected, atol=1e-6)


def test_parameters_are_reachable_through_paths():
    scene = System(
        star=PointSource(),
        disk=MultipoleDisk(RadialSum(a=GaussianRing(5.0, 1.0)), inc=20.0),
    )
    assert float(scene.get("disk.profile.a.radius")) == 5.0
    moved = scene.set(["disk.profile.a.radius", "disk.inc"], [6.0, 40.0])
    assert float(moved.disk.profile.a.radius) == 6.0
    assert float(moved.disk.inc) == 40.0


def test_changing_values_does_not_recompile():
    traces = []

    @eqx.filter_jit
    def evaluate(model, u, v):
        traces.append(1)
        return model.model(u, v, WAVEL)

    u, v = (np.asarray(a) for a in _uv(10))
    for r0, inc, sigma in [
        (5.0, 10.0, 1.0),
        (7.0, 50.0, 1.5),
        (3.0, 80.0, 0.5),
    ]:
        disk = MultipoleDisk(
            RadialSum(a=GaussianRing(r0, sigma), b=ExpTail(r0, sigma)),
            inc=inc,
            blur_fwhm=1.0,
            n_r=24,
        )
        evaluate(disk, u, v)
    assert len(traces) == 1
    # A different number of nodes is a different program.
    evaluate(MultipoleDisk(GaussianRing(5.0, 1.0), n_r=25), u, v)
    assert len(traces) == 2


def test_compiled_program_holds_no_parameter_constants():
    # The nodes are inputs, not literals: the lowered program must not
    # change with the values (it would recompile for every value).
    u, v = (np.asarray(a) for a in _uv(10))

    def text(r0):
        disk = MultipoleDisk(GaussianRing(r0, 1.0), inc=30.0, n_r=16)
        return jax.jit(lambda m: m.model(u, v, WAVEL)).lower(disk).as_text()

    assert text(5.0) == text(7.5)


def test_chunked_evaluation_matches(monkeypatch):
    u, v = _uv(37)
    disk = MultipoleDisk(GaussianRing(5.0, 1.0), inc=30.0, pa=40.0, n_r=32)
    whole = _cvis(disk, u, v)
    monkeypatch.setattr(models, "_POLAR_CHUNK_ELEMENTS", 5 * 32)
    chunked = _cvis(disk, u, v)
    assert onp.allclose(whole, chunked, atol=1e-6)


# --- conventions -------------------------------------------------------------


@pytest.mark.validates(
    "virgil.models.MultipoleDisk", roots=["self-consistency"]
)
@pytest.mark.parametrize("inc", [0.0, 35.0, 70.0])
@pytest.mark.parametrize("pa", [0.0, 30.0, 120.0])
def test_inclined_thin_ring_matches_the_modulated_rim(inc, pa):
    u, v = _uv()
    rim = ModulatedGaussianRim(diam=12.0, fwhm=2.0, inc=inc, pa=pa)
    disk = MultipoleDisk(ThinRing(6.0), inc=inc, pa=pa, blur_fwhm=2.0)
    assert onp.max(onp.abs(_cvis(disk, u, v) - _cvis(rim, u, v))) < 2e-6


@pytest.mark.validates("virgil.models.MultipoleDisk", roots=["mathematics"])
def test_major_axis_is_at_pa_north_to_east():
    # The ring's full diameter is seen along the projected major axis, its
    # foreshortened one (times cos i) along the minor axis.
    pa, inc, q = 25.0, 60.0, 3.0e7
    along = (q * onp.sin(onp.deg2rad(pa)), q * onp.cos(onp.deg2rad(pa)))
    across = (q * onp.cos(onp.deg2rad(pa)), -q * onp.sin(onp.deg2rad(pa)))
    disk = MultipoleDisk(ThinRing(4.0), inc=inc, pa=pa)
    x = 2.0 * onp.pi * q * 4.0 * _MAS2RAD_REF
    major = _cvis(
        disk, onp.array([along[0]]) * WAVEL, onp.array([along[1]]) * WAVEL
    )
    minor = _cvis(
        disk, onp.array([across[0]]) * WAVEL, onp.array([across[1]]) * WAVEL
    )
    assert onp.allclose(major, scipy_j0(x), atol=2e-6)
    assert onp.allclose(
        minor, scipy_j0(x * onp.cos(onp.deg2rad(inc))), atol=2e-6
    )


@pytest.mark.validates("virgil.models.MultipoleDisk", roots=["mathematics"])
def test_offset_has_the_interferometric_fourier_sign():
    # A point-like disk offset to the East (dra > 0) at baseline u has
    # visibility exp(-2 pi i u dra), as for a PointSource.
    u, v = _uv(10, scale=3e6)
    tiny = MultipoleDisk(GaussianRing(0.0, 0.01), dra=7.0, ddec=-3.0, n_r=16)
    point = PointSource(dra=7.0, ddec=-3.0)
    assert onp.allclose(_cvis(tiny, u, v), _cvis(point, u, v), atol=2e-6)


def test_engine_matches_multipole_disk():
    u, v = _uv()
    disk = MultipoleDisk(GaussianRing(5.0, 1.0), inc=20.0, pa=10.0, n_r=32)
    r, w = disk.profile.quadrature()
    direct = multipole_disk_cvis(
        np.asarray(u) / WAVEL, np.asarray(v) / WAVEL, r, w, 20.0, 10.0
    )
    assert onp.allclose(onp.asarray(direct), _cvis(disk, u, v), atol=1e-7)


# --- rendering -----------------------------------------------------------------


def _render_visibilities(model, u, v, npix=384, fov_mas=60.0):
    image = onp.asarray(model.render(npix=npix, fov_mas=fov_mas)).ravel()
    xx, yy = (onp.asarray(a).ravel() for a in image_coordinates(npix, fov_mas))
    phase = onp.exp(
        -2j
        * onp.pi
        * _MAS2RAD_REF
        * (onp.outer(u, xx) + onp.outer(v, yy))
        / WAVEL
    )
    return phase @ image / image.sum()


@pytest.mark.validates(
    "virgil.models.MultipoleDisk", roots=["self-consistency"]
)
@pytest.mark.parametrize(
    "disk",
    [
        MultipoleDisk(
            RadialPowerLaw(0.0, 9.0, -0.5),
            inc=40.0,
            pa=70.0,
            dra=2.0,
            ddec=-3.0,
        ),
        MultipoleDisk(
            GaussianRing(9.0, 1.5), inc=55.0, pa=20.0, dra=-3.0, ddec=2.0
        ),
        MultipoleDisk(
            GaussianRing(8.0, 1.5), inc=30.0, pa=100.0, blur_fwhm=2.0
        ),
        MultipoleDisk(ExpTail(3.0, 3.0), inc=50.0, pa=15.0, n_r=96),
        MultipoleDisk(
            Gap(GaussianRing(9.0, 3.0), radius=9.0, sigma=1.0),
            inc=20.0,
            pa=60.0,
        ),
        MultipoleDisk(
            RadialSum(a=GaussianRing(4.0, 1.0), b=(ThinRing(10.0), 0.5)),
            inc=45.0,
            pa=30.0,
        ),
    ],
    ids=["power", "gauss", "blur", "exp", "gap", "sum"],
)
def test_render_fourier_transform_matches_model_visibilities(disk):
    u, v = _uv(30, scale=1.2e7)
    cvis_render = _render_visibilities(disk, u, v)
    assert onp.max(onp.abs(cvis_render - _cvis(disk, u, v))) < 6e-3


def test_render_follows_the_sky_orientation():
    # East left, North up; PA north-to-east.
    def halves(pa, dra=0.0, ddec=0.0):
        image = onp.asarray(
            MultipoleDisk(
                GaussianRing(15.0, 1.0), inc=70.0, pa=pa, dra=dra, ddec=ddec
            ).render(npix=121, fov_mas=80.0)
        )
        rows = onp.abs(onp.arange(121) - 60)
        # Spread of the light along the vertical (North) and horizontal.
        return (image.sum(axis=1) @ rows**2, image.sum(axis=0) @ rows**2)

    north, east = halves(0.0)
    assert north > 2.0 * east
    north, east = halves(90.0)
    assert east > 2.0 * north

    image = onp.asarray(
        MultipoleDisk(GaussianRing(0.0, 1.0), dra=10.0, ddec=10.0).render(
            npix=81, fov_mas=80.0
        )
    )
    row, col = onp.unravel_index(onp.argmax(image), image.shape)
    assert row < 40 and col < 40  # North-East is the upper left corner

    # An oblique PA distinguishes pa from -pa: light along 30 deg east of
    # north lies in the NE (upper-left) and SW (lower-right) quadrants.
    image = onp.asarray(
        MultipoleDisk(GaussianRing(15.0, 1.0), inc=70.0, pa=30.0).render(
            npix=121, fov_mas=80.0
        )
    )
    ne_sw = image[:60, :60].sum() + image[61:, 61:].sum()
    nw_se = image[:60, 61:].sum() + image[61:, :60].sum()
    assert ne_sw > 1.5 * nw_se


# --- quadrature convergence and suggest_n_r ------------------------------------


@pytest.mark.validates("virgil.models.MultipoleDisk", roots=["mathematics"])
def test_quadrature_converges_spectrally():
    q = onp.linspace(0.0, 1.2e8, 120)
    with jax.enable_x64(True):
        qj = np.asarray(q)
        zero = np.zeros(q.shape)

        def cvis(profile, n):
            return onp.asarray(
                MultipoleDisk(profile, n_r=n)._centred_cvis(qj, zero)
            )

        ring = GaussianRing(10.0, 2.0)
        reference = cvis(ring, 256)
        errors = [
            onp.max(onp.abs(cvis(ring, n) - reference)) for n in (24, 48, 64)
        ]
        assert errors[0] > 1e-2
        assert errors[2] < 1e-9

        # The exponential tail, on a range its rule resolves (2 pi q L <= 2).
        scale = 5.0
        q_exp = onp.linspace(
            0.0, 2.0 / (2 * onp.pi * _MAS2RAD_REF * scale), 60
        )
        a = 2 * onp.pi * q_exp * _MAS2RAD_REF * scale
        errors = [
            onp.max(
                onp.abs(
                    onp.asarray(
                        MultipoleDisk(
                            ExpTail(0.0, scale), n_r=n
                        )._centred_cvis(
                            np.asarray(q_exp), np.zeros(q_exp.shape)
                        )
                    ).real
                    - (1.0 + a**2) ** -1.5
                )
            )
            for n in (16, 24, 32, 48)
        ]
        assert errors[0] > errors[1] > errors[2] > errors[3]
        assert errors[3] < 1e-12


@pytest.mark.parametrize(
    "profile",
    [
        GaussianRing(10.0, 2.0),
        RadialPowerLaw(2.0, 14.0, -1.0),
        ExpTail(0.0, 1.5),
    ],
    ids=["gauss", "power", "exp"],
)
def test_suggest_n_r_gives_a_converged_rule(profile):
    u, v = _uv(60, seed=4, scale=4e7)
    data = SimpleNamespace(u=u, v=v, wavel=np.full(u.shape, WAVEL))
    n = MultipoleDisk(profile).suggest_n_r(data)
    assert isinstance(n, int) and n % 8 == 0 and n >= 16
    with jax.enable_x64(True):
        coarse = _cvis(MultipoleDisk(profile, n_r=n), u, v)
        fine = _cvis(MultipoleDisk(profile, n_r=2 * n), u, v)
    assert onp.max(onp.abs(coarse - fine)) < 1e-4


# --- validation ------------------------------------------------------------------


def test_invalid_inputs_are_rejected():
    with pytest.raises(ValueError, match="non-negative"):
        MultipoleDisk(GaussianRing(5.0, 1.0), flux=-1.0)
    with pytest.raises(TypeError):
        Gap(ThinRing(5.0), radius=5.0, sigma=1.0)
    with pytest.raises(TypeError):
        MultipoleDisk("ring")
    with pytest.raises(ValueError, match="n_r"):
        MultipoleDisk(GaussianRing(5.0, 1.0), n_r=0)
    with pytest.raises(ValueError, match="quadrature rule"):
        GaussianRing(5.0, 1.0).quadrature()
    with pytest.raises(ValueError, match="at least one"):
        RadialSum()
    with pytest.raises(ValueError, match="clashes"):
        RadialSum(parts=ThinRing(1.0))
    with pytest.raises(ValueError, match="at most"):
        MultipoleDisk(ExpTail(0.0, 1.0), n_r=400)


def test_is_physical_flags_bad_parameters():
    good = MultipoleDisk(GaussianRing(5.0, 1.0), blur_fwhm=1.0)
    assert bool(good.is_physical())
    assert not bool(good.set("profile.sigma", -1.0).is_physical())
    assert not bool(good.set("blur_fwhm", -1.0).is_physical())
    gap = MultipoleDisk(Gap(GaussianRing(5.0, 1.0), 5.0, 0.5))
    assert not bool(gap.set("profile.depth", 1.5).is_physical())


def test_suggest_n_r_is_buildable_for_a_long_exp_tail():
    class _Data:
        u = onp.array([57.0])
        v = onp.array([0.0])
        wavel = onp.array([2.2e-6])

    for profile in (
        ExpTail(0.0, 10.0),
        RadialSum(a=ExpTail(0.0, 10.0), b=GaussianRing(5.0, 1.0)),
    ):
        n = MultipoleDisk(profile).suggest_n_r(_Data)
        assert n <= 150
        MultipoleDisk(profile, n_r=n)


def test_a_profile_has_no_weight_and_a_sum_term_carries_it():
    for profile in (
        ThinRing(3.0),
        GaussianRing(5.0, 1.0),
        RadialPowerLaw(1.0, 8.0),
        ExpTail(1.0, 2.0),
        Gap(GaussianRing(6.0, 2.0), 6.0, 0.6),
    ):
        assert not hasattr(profile, "weight")
    rings = RadialSum(
        a=GaussianRing(3.0, 0.5), b=(GaussianRing(8.0, 1.0), 0.5)
    )
    # A bare part has weight 1; paths read profile.<part>.weight and reach the
    # part's own parameters.
    assert float(rings.a.weight) == 1.0 and float(rings.b.weight) == 0.5
    assert float(rings.b.radius) == 8.0
    disk = MultipoleDisk(rings, inc=30.0, n_r=24)
    moved = disk.set(["profile.b.weight", "profile.b.radius"], [2.0, 9.0])
    assert float(moved.get("profile.b.weight")) == 2.0
    assert float(moved.profile.b.profile.radius) == 9.0
    assert not bool(disk.set("profile.b.weight", -1.0).is_physical())
    # The weights are the parts' shares of the light.
    u, v = _uv(6)
    heavy = disk.set("profile.b.weight", 1e6)
    only_b = MultipoleDisk(GaussianRing(8.0, 1.0), inc=30.0, n_r=24)
    assert onp.allclose(_cvis(heavy, u, v), _cvis(only_b, u, v), atol=1e-4)
    with pytest.raises(TypeError, match="not a profile"):
        RadialSum(a=(1.0, GaussianRing(3.0, 0.5)))


def test_suggest_n_r_warns_when_it_steps_down():
    class _Data:
        u = onp.array([57.0])
        v = onp.array([0.0])
        wavel = onp.array([1e-6])

    with pytest.warns(UserWarning, match="suggest_n_r wants"):
        MultipoleDisk(ExpTail(0.0, 10.0)).suggest_n_r(_Data)
