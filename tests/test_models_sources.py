import equinox as eqx
import jax
import numpy as onp
import pytest
import jax.numpy as np
from scipy.special import j0 as scipy_j0
from scipy.special import j1 as scipy_j1
from scipy.special import jn_zeros

from virgil._geometry import image_coordinates as _image_coordinates
from virgil._geometry import offset_phase
from virgil._geometry import undo_elliptical_transf_spat_freq
from virgil.models import (
    BinaryModelAngular,
    BinaryModelCartesian,
    EllipticalGaussian,
    FlaredDiskGaussian,
    GaussianArc,
    FlaredDiskHG,
    FlaredDiskPowerLaw,
    GaussianDisk,
    GravityDarkenedStar,
    HarmonixModel,
    Image,
    LimbDarkenedDisk,
    ModulatedGaussianRim,
    PointSource,
    QuadraticLimbDarkenedDisk,
    Rotated,
    SquareRootLimbDarkenedDisk,
    System,
    TruncatedCone,
    UniformDisk,
    cvis_radial_dirac_delta_modulated,
    cvis_uniform_disk,
)
from virgil.likelihood import model_loglike
from tests._test_data import oidata

# Independent reference conversion (not imported from virgil) so the
# analytic checks below don't just re-test the module's own constant.
_MAS2RAD_REF = onp.pi / 180.0 / 3600.0 / 1000.0


def _star_and_disk(sigma, flux, dra=0.0, ddec=0.0):
    return System(
        star=PointSource(),
        disk=GaussianDisk(sigma, flux=flux, dra=dra, ddec=ddec),
    )


def _star_and_rim(flux, **rim_kwargs):
    return System(
        star=PointSource(), rim=ModulatedGaussianRim(flux=flux, **rim_kwargs)
    )


def test_gaussian_disk_oidata_and_render():
    model = _star_and_disk(sigma=30.0, flux=0.1, dra=10.0, ddec=-10.0)
    model_vec = oidata.model(model)
    image = model.render(npix=64, fov_mas=150.0)

    assert model_vec.shape[0] == len(oidata.vis) + len(oidata.phi)
    assert np.all(np.isfinite(model_vec))
    assert image.shape == (64, 64)
    assert np.all(np.isfinite(image))
    assert np.isclose(np.sum(image), 1.0, rtol=1e-6, atol=1e-6)


def test_gaussian_disk_render_remains_finite_for_narrow_shifted_disk():
    image = _star_and_disk(sigma=1e-6, flux=0.1, dra=1e6, ddec=-1e6).render(
        npix=32, fov_mas=20.0
    )

    assert image.shape == (32, 32)
    assert np.all(np.isfinite(image))
    assert np.isclose(np.sum(image), 1.0, rtol=1e-6, atol=1e-6)


def test_cvis_uniform_disk_is_well_behaved():
    uu = oidata.u / oidata.wavel
    vv = oidata.v / oidata.wavel
    cvis = cvis_uniform_disk(uu, vv, ud=5.0, dra=5.0, ddec=-3.0)
    assert cvis.shape == uu.shape
    assert np.all(np.isfinite(cvis))
    assert np.all(np.abs(cvis) <= 1.0 + 1e-12)


def test_cvis_uniform_disk_zero_baseline_is_unity():
    cvis = cvis_uniform_disk(np.array([0.0]), np.array([0.0]), ud=10.0)
    assert np.allclose(cvis, 1.0 + 0j)


@pytest.mark.validates("virgil.models.UniformDisk", roots=["mathematics"])
def test_cvis_uniform_disk_matches_analytic_airy_formula():
    """Visibility amplitude should follow 2*J1(pi*theta*B/lambda) /
    (pi*theta*B/lambda), with theta the disk diameter in mas and B/lambda
    the baseline length in wavelength units (here passed directly as
    ``u``, ``v``). Checked against an independent scipy.special.j1 call,
    not against the module's own Bessel implementation.
    """
    ud = 8.0
    base_norm = onp.array([5.0, 20.0, 45.0, 80.0])

    cvis = cvis_uniform_disk(
        np.asarray(base_norm), np.zeros_like(base_norm), ud
    )

    kernel = onp.pi * ud * _MAS2RAD_REF * base_norm
    expected = 2.0 * scipy_j1(kernel) / kernel

    assert onp.allclose(onp.asarray(cvis).real, expected, atol=1e-8)
    assert onp.allclose(onp.asarray(cvis).imag, 0.0, atol=1e-8)


@pytest.mark.validates("virgil.models.UniformDisk", roots=["mathematics"])
def test_cvis_uniform_disk_vanishes_at_first_airy_null():
    ud = 8.0
    first_null_kernel = jn_zeros(1, 1)[0]
    base_norm = first_null_kernel / (onp.pi * ud * _MAS2RAD_REF)

    cvis = cvis_uniform_disk(np.asarray(base_norm), np.asarray(0.0), ud)

    assert onp.abs(onp.asarray(cvis)) < 1e-6


def test_cvis_uniform_disk_converges_to_point_source_for_small_ud():
    uu = oidata.u / oidata.wavel
    vv = oidata.v / oidata.wavel
    cvis = cvis_uniform_disk(uu, vv, ud=1e-6)
    assert np.allclose(cvis, 1.0 + 0j, atol=1e-6)


def test_uniform_disk_oidata_and_render():
    model = UniformDisk(diam=30.0, dra=10.0, ddec=-10.0)
    model_vec = oidata.model(model)
    image = model.render(npix=64, fov_mas=150.0)

    assert model_vec.shape[0] == len(oidata.vis) + len(oidata.phi)
    assert np.all(np.isfinite(model_vec))
    assert image.shape == (64, 64)
    assert np.all(np.isfinite(image))
    assert np.isclose(np.sum(image), 1.0, rtol=1e-6, atol=1e-6)


def test_uniform_disk_render_uses_interferometric_image_orientation():
    image = UniformDisk(diam=1e-3, dra=2.0, ddec=2.0).render(
        npix=5, fov_mas=10.0
    )

    assert onp.unravel_index(onp.asarray(image).argmax(), image.shape) == (
        1,
        1,
    )


def test_star_and_rim_is_well_behaved():
    cvis = _star_and_rim(
        dra=5.0,
        ddec=-3.0,
        diam=20.0,
        fwhm=2.0,
        inc=30.0,
        pa=45.0,
        az_amps=0.3,
        az_pas=55.0,
        flux=0.5,
    ).model(oidata.u, oidata.v, oidata.wavel)
    assert cvis.shape == oidata.u.shape
    assert np.all(np.isfinite(cvis))
    # |1 + 0.3*cos(theta)| never drops below 0.7, i.e. flux stays
    # non-negative everywhere, so the rim's own visibility magnitude stays
    # bounded by 1, and the point-source mixture preserves that bound.
    assert np.all(np.abs(cvis) <= 1.0 + 1e-6)


def test_star_and_zero_flux_rim_is_pure_point_source():
    """With flux=0 the rim contributes no light, so the visibility should
    be exactly that of an unresolved point source (unity everywhere),
    regardless of the rim's own geometry.
    """
    cvis = _star_and_rim(
        dra=5.0,
        ddec=-3.0,
        diam=20.0,
        fwhm=2.0,
        inc=30.0,
        pa=45.0,
        az_amps=0.3,
        az_pas=55.0,
        flux=0.0,
    ).model(oidata.u, oidata.v, oidata.wavel)
    assert np.allclose(cvis, 1.0 + 0j)


@pytest.mark.validates(
    "virgil.models.ModulatedGaussianRim", roots=["mathematics"]
)
def test_symmetric_rim_matches_bessel_j0():
    """An unmodulated, uninclined, infinitely-narrow rim is a plain thin
    ring, whose visibility is the classic J0(2*pi*r0*B/lambda) form; mixed
    with an unresolved point source (visibility 1 everywhere) at flux ratio
    ``flux``, the total visibility should be l1 + l2*J0(...). Checked against
    an independent scipy.special.j0 call.
    """
    diam = 20.0
    flux = 3.0
    uu = oidata.u / oidata.wavel
    vv = oidata.v / oidata.wavel

    cvis = _star_and_rim(
        diam=diam, fwhm=1e-6, inc=0.0, pa=0.0, flux=flux
    ).model(oidata.u, oidata.v, oidata.wavel)

    base_norm = onp.hypot(onp.asarray(uu), onp.asarray(vv))
    rim_expected = scipy_j0(
        2.0 * onp.pi * base_norm * (diam / 2.0) * _MAS2RAD_REF
    )
    l2 = flux / (flux + 1.0)
    l1 = 1.0 - l2
    expected = l1 + l2 * rim_expected

    assert onp.allclose(onp.asarray(cvis).real, expected, atol=1e-6)
    assert onp.allclose(onp.asarray(cvis).imag, 0.0, atol=1e-6)


def test_rim_stays_finite_at_edge_on_inclination():
    # inc=90deg drives stretch = cos(inc) to 0; the 1e-8 floor on stretch
    # should keep the Fourier-domain visibility finite (unlike the
    # pixel-mask render below, this doesn't depend on grid resolution).
    cvis = ModulatedGaussianRim(diam=40.0, fwhm=2.0, inc=90.0, pa=0.0).model(
        oidata.u, oidata.v, oidata.wavel
    )
    assert np.all(np.isfinite(cvis))


def test_modulated_gaussian_rim_oidata_and_render():
    model = _star_and_rim(
        diam=30.0,
        fwhm=3.0,
        inc=25.0,
        pa=60.0,
        az_amps=np.array([0.2]),
        az_pas=np.array([15.0]),
        flux=0.5,
        dra=5.0,
        ddec=-5.0,
    )
    model_vec = oidata.model(model)
    image = model.render(npix=64, fov_mas=150.0)

    assert model_vec.shape[0] == len(oidata.vis) + len(oidata.phi)
    assert np.all(np.isfinite(model_vec))
    assert image.shape == (64, 64)
    assert np.all(np.isfinite(image))
    assert np.isclose(np.sum(image), 1.0, rtol=1e-6, atol=1e-6)


def test_modulated_gaussian_rim_symmetric_case_is_finite_and_normalized():
    image = _star_and_rim(
        diam=40.0, fwhm=2.0, inc=0.0, pa=0.0, flux=0.5
    ).render(npix=64, fov_mas=100.0)

    assert np.all(np.isfinite(image))
    assert np.isclose(np.sum(image), 1.0, rtol=1e-6, atol=1e-6)


def test_modulated_gaussian_rim_render_at_zero_flux_is_point_source_only():
    """With flux=0 the rim contributes no light, so the rendered image
    should peak exactly on the central (unresolved point-source) pixel,
    with none appearing out at the rim's own radius.
    """
    npix = 65
    fov_mas = 100.0
    diam = 40.0
    image = onp.asarray(
        _star_and_rim(diam=diam, fwhm=2.0, inc=0.0, pa=0.0, flux=0.0).render(
            npix=npix, fov_mas=fov_mas
        )
    )
    center = npix // 2

    assert np.all(np.isfinite(image))
    assert np.isclose(np.sum(image), 1.0, rtol=1e-6, atol=1e-6)
    assert onp.unravel_index(image.argmax(), image.shape) == (center, center)

    # No flux should appear near the rim's own radius (diam/2 from center),
    # since flux=0 removes the rim entirely.
    pixel_scale_mas = fov_mas / npix
    ring_px = int(round((diam / 2.0) / pixel_scale_mas))
    assert image[center, center + ring_px] < 1e-6


def test_modulated_gaussian_rim_render_finite_at_moderate_inclination():
    # A close-to-edge-on render (e.g. inc=90) can legitimately miss the
    # (near-)zero-measure ring on a coarse pixel grid; that's a
    # rasterization limitation of the mask-based render, not a numerical
    # blow-up, so this checks a realistic, non-degenerate inclination.
    image = _star_and_rim(
        diam=40.0, fwhm=2.0, inc=60.0, pa=0.0, flux=0.5
    ).render(npix=64, fov_mas=100.0)

    assert np.all(np.isfinite(image))
    assert np.isclose(np.sum(image), 1.0, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize(
    ("az_pas", "bright_half", "faint_half"),
    [
        # PA=90deg modulation -> bright towards East (+x, low column index).
        (90.0, "east", "west"),
        # PA=0deg modulation -> bright towards North (+y, low row index).
        (0.0, "north", "south"),
    ],
)
def test_modulated_gaussian_rim_render_follows_north_to_east_pa_convention(
    az_pas, bright_half, faint_half
):
    image = onp.asarray(
        _star_and_rim(
            diam=40.0,
            fwhm=2.0,
            inc=0.0,
            pa=0.0,
            az_amps=np.array([1.0]),
            az_pas=np.array([az_pas]),
            flux=0.5,
        ).render(npix=81, fov_mas=100.0)
    )

    center = image.shape[0] // 2
    halves = {
        "east": image[:, :center].sum(),
        "west": image[:, center + 1 :].sum(),
        "north": image[:center, :].sum(),
        "south": image[center + 1 :, :].sum(),
    }
    assert halves[bright_half] > halves[faint_half]


def _rim_baselines():
    rng = onp.random.default_rng(3)
    return rng.uniform(-60.0, 60.0, 40), rng.uniform(-60.0, 60.0, 40), 2.2e-6


@pytest.mark.validates(
    "virgil.models.ModulatedGaussianRim", roots=["mathematics"]
)
@pytest.mark.parametrize("x64", [False, True])
def test_unmodulated_rim_visibility_is_blurred_in_the_rim_plane(x64):
    """An unmodulated rim is a thin ring times a Gaussian envelope, both
    evaluated on the deprojected spatial frequencies (ut, vt): the blur is
    isotropic in the plane of the rim, not on the sky. Checked against
    independent scipy.special.j0 and numpy calls.
    """
    diam, fwhm, inc, pa = 30.0, 6.0, 55.0, 25.0
    u, v, wavel = _rim_baselines()
    with jax.enable_x64(x64):
        cvis = ModulatedGaussianRim(diam, fwhm, inc, pa).model(u, v, wavel)

    pa_rad, inc_rad = onp.deg2rad(pa), onp.deg2rad(inc)
    uu, vv = u / wavel, v / wavel
    ut = (uu * onp.cos(pa_rad) - vv * onp.sin(pa_rad)) * onp.cos(inc_rad)
    vt = uu * onp.sin(pa_rad) + vv * onp.cos(pa_rad)
    q = onp.hypot(ut, vt)
    fwhm_rad = fwhm * _MAS2RAD_REF
    expected = scipy_j0(
        2.0 * onp.pi * q * (diam / 2.0) * _MAS2RAD_REF
    ) * onp.exp(-(onp.pi**2) * fwhm_rad**2 * q**2 / (4.0 * onp.log(2.0)))

    assert onp.allclose(onp.asarray(cvis).real, expected, atol=1e-6)
    assert onp.allclose(onp.asarray(cvis).imag, 0.0, atol=1e-6)


def test_inclined_rim_visibility_is_the_face_on_rim_in_the_rim_plane():
    """Inclining and rotating a modulated rim only changes the frame: its
    visibility is that of the face-on (inc=0, pa=0) rim, with modulation
    angles az_pas - pa, sampled at the deprojected frequencies.
    """
    diam, fwhm, inc, pa = 25.0, 4.0, 65.0, 40.0
    az_amps = np.array([0.5, 0.3])
    az_pas = np.array([100.0, 20.0])
    u, v, wavel = _rim_baselines()

    cvis = ModulatedGaussianRim(
        diam, fwhm, inc, pa, az_amps=az_amps, az_pas=az_pas
    ).model(u, v, wavel)
    ut, vt = undo_elliptical_transf_spat_freq(
        u, v, pa, max(onp.cos(onp.deg2rad(inc)), 1e-8)
    )
    face_on = ModulatedGaussianRim(
        diam, fwhm, 0.0, 0.0, az_amps=az_amps, az_pas=az_pas - pa
    ).model(ut, vt, wavel)

    assert onp.allclose(onp.asarray(cvis), onp.asarray(face_on), atol=1e-6)


def test_inclined_unmodulated_rim_render_has_no_bright_ansae():
    """With the blur isotropic in the rim plane, an inclined unmodulated
    rim is the face-on blurred ring compressed along its minor axis, so its
    peak brightness is the same on the major and the minor axis. A blur
    that is isotropic on the sky instead gives ansae about 2.5 times
    brighter at inc=70.
    """
    npix = 201
    image = onp.asarray(
        ModulatedGaussianRim(diam=40.0, fwhm=8.0, inc=70.0, pa=0.0).render(
            npix=npix, fov_mas=100.5
        )
    )
    center = npix // 2
    # pa=0: the major axis runs North-South (a column), the minor axis
    # East-West (a row).
    major_peak = image[:, center].max()
    minor_peak = image[center, :].max()

    assert abs(major_peak / minor_peak - 1.0) < 0.05


@pytest.mark.parametrize("x64", [False, True])
def test_edge_on_rotated_rim_render_is_finite_and_normalized(x64):
    with jax.enable_x64(x64):
        image = ModulatedGaussianRim(
            diam=40.0, fwhm=2.0, inc=90.0, pa=37.0
        ).render(npix=64, fov_mas=100.0)
        assert np.all(np.isfinite(image))
        assert np.isclose(np.sum(image), 1.0, rtol=1e-6, atol=1e-6)


def test_binary_render_is_available():
    image = BinaryModelCartesian(10.0, -5.0, 1e-3).render(
        npix=32, fov_mas=80.0
    )
    assert image.shape == (32, 32)
    assert np.all(np.isfinite(image))
    assert np.isclose(np.sum(image), 1.0, rtol=1e-6, atol=1e-6)


@pytest.mark.validates(
    "virgil.models.SourceModel.render", roots=["self-consistency"]
)
@pytest.mark.parametrize(
    ("model", "atol"),
    [
        (BinaryModelCartesian(12.0, -7.0, 0.3), 2e-3),
        (BinaryModelAngular(20.0, 60.0, 1.0 / 3.0), 2e-3),
        (_star_and_disk(4.0, 0.5, 6.0, 3.0), 2e-3),
        (UniformDisk(15.0, dra=-5.0, ddec=4.0), 2e-3),
        (
            TruncatedCone(
                4.0,
                35.0,
                3.0,
                6.0,
                1.5,
                tilt=30.0,
                pa=60.0,
                ratio=0.7,
                dra=3.0,
                ddec=-2.0,
                n_rings=16,
            ),
            2e-3,
        ),
        (LimbDarkenedDisk(15.0, u=[0.3, 0.2, 0.1], dra=-5.0, ddec=4.0), 2e-3),
        (QuadraticLimbDarkenedDisk(15.0, q1=0.5, q2=0.3, dra=-5.0), 2e-3),
        (SquareRootLimbDarkenedDisk(15.0, q1=0.6, q2=0.4, ddec=4.0), 2e-3),
        (
            Image.from_model(GaussianDisk(4.0), 49, 0.5, dra=6.0, ddec=-3.0),
            2e-3,
        ),
        (
            _star_and_rim(
                diam=14.0,
                fwhm=3.0,
                inc=60.0,
                pa=30.0,
                az_amps=np.array([0.6, 0.3]),
                az_pas=np.array([100.0, 20.0]),
                flux=0.7,
                dra=3.0,
                ddec=-2.0,
            ),
            2e-3,
        ),
        (
            System(
                star=PointSource(),
                comp=System(
                    core=PointSource(),
                    disk=GaussianDisk(3.0, flux=0.5),
                    dra=-15.0,
                    ddec=10.0,
                    flux=0.2,
                ),
            ),
            2e-3,
        ),
        (
            Rotated(
                System(
                    star=PointSource(),
                    comp=GaussianDisk(2.0, flux=0.3, dra=8.0, ddec=-3.0),
                ),
                70.0,
            ),
            2e-3,
        ),
        (
            System(
                star=PointSource(),
                disk=FlaredDiskPowerLaw(
                    n=4.0,
                    radius=15.0,
                    fwhm=6.0,
                    inc=50.0,
                    pa=30.0,
                    skew=2.0,
                    aspect=0.15,
                    symmetric=0.1,
                    npix=40,
                    pixel_scale_mas=2.0,
                    flux=0.5,
                    dra=2.0,
                    ddec=-1.0,
                ),
            ),
            2e-3,
        ),
        (
            System(
                star=PointSource(),
                env=EllipticalGaussian(
                    12.0, 0.4, 30.0, flux=0.7, dra=3.0, ddec=-2.0
                ),
            ),
            2e-3,
        ),
        (
            System(
                star=PointSource(),
                arc=GaussianArc(15.0, 3.0, 20.0, 250.0, flux=0.8, dra=6.0),
            ),
            2e-3,
        ),
        (
            GravityDarkenedStar(
                12.0, omega=0.9, inc=50.0, pa=30.0, dra=-5.0, ddec=4.0
            ),
            # the image shades whole facets, the DFT uses barycentre points
            5e-4,
        ),
    ],
    ids=[
        "binary_cart",
        "binary_ang",
        "gauss_disk",
        "uniform_disk",
        "truncated_cone",
        "limb_darkened_disk",
        "quadratic_limb_darkened_disk",
        "square_root_limb_darkened_disk",
        "image",
        "rim",
        "nested_system",
        "rotated",
        "flared_disk",
        "elliptical_gaussian",
        "gaussian_arc",
        "gravity_darkened_star",
    ],
)
def test_render_fourier_transform_matches_model_visibilities(model, atol):
    npix, fov_mas, wavel = 512, 80.0, 1.65e-6
    rng = onp.random.default_rng(1)
    u = rng.uniform(-8.0, 8.0, 40)
    v = rng.uniform(-8.0, 8.0, 40)

    image = onp.asarray(model.render(npix=npix, fov_mas=fov_mas)).ravel()
    xx, yy = (
        onp.asarray(a).ravel() for a in _image_coordinates(npix, fov_mas)
    )
    phase = onp.exp(
        -2j
        * onp.pi
        * _MAS2RAD_REF
        * (onp.outer(u, xx) + onp.outer(v, yy))
        / wavel
    )
    cvis_render = phase @ image / image.sum()
    cvis_model = onp.asarray(model.model(u, v, wavel))

    assert onp.max(onp.abs(cvis_render - cvis_model)) < atol


@pytest.mark.parametrize(
    ("npix", "fov_mas", "expected"),
    [
        (4, 8.0, onp.array([3.0, 1.0, -1.0, -3.0])),
        (5, 10.0, onp.array([4.0, 2.0, 0.0, -2.0, -4.0])),
    ],
)
def test_image_coordinates_use_pixel_centers(npix, fov_mas, expected):
    xx, yy = _image_coordinates(npix, fov_mas)

    assert xx.shape == (npix, npix)
    assert yy.shape == (npix, npix)
    assert onp.allclose(onp.asarray(xx[0]), expected)
    assert onp.allclose(onp.asarray(yy[:, 0]), expected)


def test_gaussian_disk_render_uses_interferometric_image_orientation():
    image = _star_and_disk(sigma=1e-3, flux=10.0, dra=2.0, ddec=2.0).render(
        npix=5, fov_mas=10.0
    )

    assert onp.unravel_index(onp.asarray(image).argmax(), image.shape) == (
        1,
        1,
    )


def test_harmonix_model_for_external_visibility_models():
    class MockHarmonix:
        def __init__(self):
            self.calls = []

        def visibility(self, uu, vv, time):
            self.calls.append((uu, vv, time))
            return np.exp(-1e-6 * (uu**2 + vv**2))

        def render(self, npix, fov_mas):
            _ = fov_mas
            return np.ones((npix, npix))

    mock = MockHarmonix()
    wrapped = HarmonixModel(
        mock,
        visibility_method="visibility",
        observation_time=0.25,
    )
    cvis = wrapped.model(oidata.u, oidata.v, oidata.wavel)
    image = wrapped.render(npix=20, fov_mas=100.0)
    uu, vv, time = mock.calls[0]

    assert cvis.shape == oidata.u.shape
    assert np.all(np.isfinite(cvis))
    assert np.allclose(uu, oidata.u / oidata.wavel)
    assert np.allclose(vv, oidata.v / oidata.wavel)
    assert time == 0.25
    assert image.shape == (20, 20)
    assert np.all(np.isfinite(image))
    assert np.isclose(np.sum(image), 1.0, rtol=1e-6, atol=1e-6)


class _BlobYlm(eqx.Module):
    data: jax.Array

    @classmethod
    def from_dense(cls, y, normalize=True):
        return cls(np.asarray(y))


class _BlobSurface(eqx.Module):
    """A surface whose map is a Gaussian blob at (y[1], y[2]) stellar radii."""

    y: _BlobYlm
    sigma: float = 0.1

    def _intensity(self, x, y, z, theta=0.0):
        x0 = self.y.data[1] + theta
        y0 = self.y.data[2]
        return np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * self.sigma**2))


class _BlobStar(eqx.Module):
    """A harmonix-like star with analytic visibilities and no dependencies.

    Like harmonix, it reads its map from ``data`` (here the blob's centre),
    not from ``surface.y``, and rotation moves the blob East.
    """

    surface: _BlobSurface
    radius: float
    data: jax.Array

    def rotational_phase(self, time):
        return 0.1 * time

    def model(self, u, v, time):
        dra = (self.data[0] + self.rotational_phase(time)) * self.radius
        ddec = self.data[1] * self.radius
        sigma_rad = _MAS2RAD_REF * self.surface.sigma * self.radius
        envelope = np.exp(-2 * np.pi**2 * sigma_rad**2 * (u**2 + v**2))
        return envelope * offset_phase(u, v, dra, ddec)


def _blob_star(radius=2.0, centre=(0.3, -0.2)):
    data = np.asarray(centre)
    return _BlobStar(_BlobSurface(_BlobYlm(np.zeros(3))), radius, data)


def _blob_baselines():
    rng = onp.random.default_rng(2)
    return rng.uniform(-1e8, 1e8, 40), rng.uniform(-1e8, 1e8, 40)


def test_harmonix_like_star_is_drawn_on_the_sky_at_its_radius():
    # Runs without harmonix: a star with a surface and a radius is drawn
    # from the surface's intensity at sky offset / radius, East left and
    # North up, so its Fourier transform reproduces its visibilities (the
    # mirror image does not), at the rotation of observation_time.
    model = HarmonixModel(_blob_star(), observation_time=1.0)
    u, v = _blob_baselines()
    cvis_model = onp.asarray(model.model(np.asarray(u), np.asarray(v), 1.0))
    cvis_render = _render_visibilities(model, u, v, 128, 6.0)
    assert onp.max(onp.abs(cvis_render - cvis_model)) < 1e-3
    mirrored = _render_visibilities(model, -u, v, 128, 6.0)
    assert onp.max(onp.abs(mirrored - cvis_model)) > 0.1


def test_harmonix_like_star_is_drawn_from_its_data():
    model = HarmonixModel(_blob_star(), observation_time=0.0)
    moved = model.set("source.data", np.array([-0.3, 0.35]))
    u, v = _blob_baselines()
    cvis_model = onp.asarray(moved.model(np.asarray(u), np.asarray(v), 1.0))
    cvis_render = _render_visibilities(moved, u, v, 128, 6.0)
    assert onp.max(onp.abs(cvis_render - cvis_model)) < 1e-3


def test_harmonix_like_star_parameters_are_reachable_through_paths():
    model = HarmonixModel(_blob_star(), observation_time=0.5)
    u, v = (np.asarray(a) for a in _blob_baselines())

    def power(m):
        return np.sum(np.abs(m.model(u, v, 1.0)) ** 2)

    assert np.allclose(eqx.filter_jit(power)(model), power(model))
    grad_radius = jax.grad(lambda r: power(model.set("source.radius", r)))(2.0)
    assert np.isfinite(grad_radius) and grad_radius < 0


def _spotted_harmonix_star(radius=2.0):
    harmonix_module = pytest.importorskip(
        "harmonix.harmonix",
        reason="harmonix integration tests require a compatible harmonix install",
    )
    starry_module = pytest.importorskip("jaxoplanet.starry")
    from jaxoplanet.starry.ylm import ylm_spot

    y = ylm_spot(4)(0.8, 0.5, 0.4, 0.7).todense()
    surface = starry_module.Surface(
        y=starry_module.Ylm.from_dense(y, normalize=False),
        inc=1.0,
        obl=0.3,
        period=1.0,
        u=[0.3, 0.2],
        normalize=False,
    )
    return harmonix_module.Harmonix(surface, radius)


def _render_visibilities(model, u, v, npix, fov_mas):
    image = onp.asarray(model.render(npix=npix, fov_mas=fov_mas)).ravel()
    xx, yy = (
        onp.asarray(a).ravel() for a in _image_coordinates(npix, fov_mas)
    )
    phase = onp.exp(
        -2j * onp.pi * _MAS2RAD_REF * (onp.outer(u, xx) + onp.outer(v, yy))
    )
    return phase @ image


@pytest.mark.validates(
    "virgil.models.HarmonixModel", roots=["self-consistency"]
)
def test_harmonix_render_fourier_transform_matches_model_visibilities():
    # The rendered star must be on the sky (East left, North up) at its
    # radius: its Fourier transform reproduces harmonix's visibilities, and
    # the mirror image (jaxoplanet's own plotting orientation) does not.
    model = HarmonixModel(_spotted_harmonix_star(), observation_time=0.2)
    rng = onp.random.default_rng(1)
    u = rng.uniform(-1.5e2, 1.5e2, 40) / 1.65e-6
    v = rng.uniform(-1.5e2, 1.5e2, 40) / 1.65e-6
    cvis_model = onp.asarray(model.model(np.asarray(u), np.asarray(v), 1.0))
    cvis_render = _render_visibilities(model, u, v, 128, 5.0)
    assert onp.max(onp.abs(cvis_render - cvis_model)) < 5e-3
    mirrored = _render_visibilities(model, -u, v, 128, 5.0)
    assert onp.max(onp.abs(mirrored - cvis_model)) > 2e-2

    # The image follows the map harmonix fits ("source.data").
    data = model.get("source.data")
    changed = model.set("source.data", -data)
    cvis_changed = onp.asarray(
        changed.model(np.asarray(u), np.asarray(v), 1.0)
    )
    render_changed = _render_visibilities(changed, u, v, 128, 5.0)
    assert onp.max(onp.abs(render_changed - cvis_changed)) < 5e-3
    assert onp.max(onp.abs(cvis_changed - cvis_model)) > 1e-2


def test_harmonix_parameters_are_reachable_through_paths():
    model = HarmonixModel(_spotted_harmonix_star(), observation_time=0.2)
    u = np.linspace(1e7, 7e7, 8)
    v = np.linspace(-3e7, 4e7, 8)

    def power(m):
        return np.sum(np.abs(m.model(u, v, 1.0)) ** 2)

    assert np.allclose(eqx.filter_jit(power)(model), power(model))
    grad_radius = jax.grad(lambda r: power(model.set("source.radius", r)))(2.0)
    grad_map = jax.grad(lambda d: power(model.set("source.data", d)))(
        model.get("source.data")
    )
    assert np.isfinite(grad_radius) and grad_radius < 0
    assert np.all(np.isfinite(grad_map)) and np.any(grad_map != 0)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_harmonix_model_random_spherical_harmonics_are_finite(seed):
    harmonix_module = pytest.importorskip(
        "harmonix.harmonix",
        reason="harmonix integration tests require a compatible harmonix install",
    )
    starry_module = pytest.importorskip("jaxoplanet.starry")

    rng = onp.random.default_rng(seed)
    degree = 3
    coeffs = onp.concatenate(
        ([1.0], 0.1 * rng.normal(size=(degree + 1) ** 2 - 1))
    )
    surface = starry_module.Surface(
        y=starry_module.Ylm.from_dense(np.asarray(coeffs)),
        inc=np.pi / 2.0,
        obl=0.0,
        period=1.0,
    )
    wrapped = HarmonixModel(
        harmonix_module.Harmonix(surface, 1.0),
        observation_time=0.0,
    )

    u = np.linspace(90.0, 190.0, 10)
    v = np.zeros_like(u)
    wavel = np.full_like(u, 1e-6)

    cvis = wrapped.model(u, v, wavel)
    image = wrapped.render(npix=64, fov_mas=20.0)

    assert cvis.shape == (10,)
    assert np.all(np.isfinite(cvis))
    assert image.shape == (64, 64)
    assert np.all(np.isfinite(image))
    assert np.isclose(np.sum(image), 1.0, rtol=1e-6, atol=1e-6)


def test_modulated_ring_visibility_accepts_scalar_baselines():
    amps, phis = np.array([0.3, 0.2]), np.array([10.0, 40.0])
    scalar = cvis_radial_dirac_delta_modulated(1e6, 2e6, 5.0, amps, phis)
    vector = cvis_radial_dirac_delta_modulated(
        np.array([1e6]), np.array([2e6]), 5.0, amps, phis
    )

    assert np.shape(scalar) == ()
    assert np.allclose(scalar, vector[0])


def test_rotated_turns_north_towards_east():
    # A blob 10 mas North, turned by 90 degrees, lands 10 mas East: on the
    # left of the rendered image (column 0 is the most positive dra).
    image = Rotated(GaussianDisk(1.0, ddec=10.0), 90.0).render(21, 42.0)
    row, col = onp.unravel_index(onp.argmax(onp.asarray(image)), (21, 21))
    assert (row, col) == (10, 5)


def _flared_disk(cls=FlaredDiskHG, **kwargs):
    geometry = dict(
        radius=20.0, fwhm=6.0, inc=60.0, pa=0.0, npix=64, pixel_scale_mas=1.0
    )
    return cls(**{**geometry, **kwargs})


def test_flared_disk_forward_scattering_peaks_on_near_side_at_pa_plus_90():
    # pa=0 puts the major axis North-South and the near side East (+dra).
    disk = _flared_disk(FlaredDiskPowerLaw, n=8.0)
    image = onp.asarray(disk.render(npix=64, fov_mas=64.0))
    xx, yy = (onp.asarray(a) for a in _image_coordinates(64, 64.0))

    assert (image * xx).sum() > 5.0
    assert abs((image * yy).sum()) < 1e-3


def test_flared_disk_surface_height_shifts_ring_towards_far_side():
    # With isotropic scattering (g=0), only the flared surface breaks the
    # symmetry, moving the ring towards the far side (West for pa=0).
    xx = onp.asarray(_image_coordinates(64, 64.0)[0])

    def centroid_dra(aspect):
        image = onp.asarray(
            _flared_disk(g=0.0, aspect=aspect).render(64, 64.0)
        )
        return (image * xx).sum()

    assert abs(centroid_dra(0.0)) < 1e-3
    assert centroid_dra(0.2) < -0.5


@pytest.mark.parametrize(
    "disk",
    [
        _flared_disk(FlaredDiskHG, g=0.3, aspect=0.1),
        _flared_disk(FlaredDiskGaussian, sigma_theta=90.0, aspect=0.1),
        _flared_disk(FlaredDiskPowerLaw, n=5.0, aspect=0.1, skew=3.0),
    ],
    ids=["hg", "gaussian", "power_law"],
)
def test_flared_disk_loglike_gradients_are_finite(disk):
    scene = System(star=PointSource(), disk=disk.set("flux", 0.2))
    grads = jax.grad(model_loglike)(scene, oidata)

    assert all(
        bool(np.all(np.isfinite(leaf)))
        for leaf in jax.tree_util.tree_leaves(grads.disk)
    )


@pytest.mark.parametrize(
    ("grid", "match"),
    [
        ({"npix": 63}, "even"),
        ({"npix": 0}, "even"),
        ({"pixel_scale_mas": 0.0}, "pixel_scale_mas"),
        ({"pixel_scale_mas": float("nan")}, "pixel_scale_mas"),
    ],
)
def test_flared_disk_needs_a_valid_grid(grid, match):
    with pytest.raises(ValueError, match=match):
        _flared_disk(g=0.3, **grid)


@pytest.mark.parametrize(
    ("cls", "bad"),
    [
        (FlaredDiskHG, {"g": 0.3, "radius": 0.0}),
        (FlaredDiskHG, {"g": 0.3, "fwhm": 0.0}),
        (FlaredDiskHG, {"g": 1.0}),
        (FlaredDiskHG, {"g": 0.3, "inc": 90.0}),
        (FlaredDiskGaussian, {"sigma_theta": 0.0}),
        (FlaredDiskPowerLaw, {"n": -1.0}),
    ],
)
def test_flared_disk_is_physical_rejects_singular_parameters(cls, bad):
    assert not bool(_flared_disk(cls, **bad).is_physical())


def test_backward_scattering_moves_the_flared_disk_peak_to_the_far_side():
    # pa=0 puts the near side East (+dra); g < 0 scatters backwards, West.
    xx = onp.asarray(_image_coordinates(64, 64.0)[0])
    image = onp.asarray(_flared_disk(g=-0.6).render(npix=64, fov_mas=64.0))
    assert (image * xx).sum() < -1.0


def test_elliptical_gaussian_at_unit_ratio_is_a_gaussian_disk():
    u, v = onp.array([10.0, -25.0, 40.0]), onp.array([5.0, 30.0, -12.0])
    fwhm = 8.0
    ellipse = EllipticalGaussian(fwhm, 1.0, 37.0, dra=2.0, ddec=-1.0)
    disk = GaussianDisk(fwhm / 2.3548200450309493, dra=2.0, ddec=-1.0)
    assert onp.allclose(ellipse.model(u, v, 2.2e-6), disk.model(u, v, 2.2e-6))


@pytest.mark.parametrize(
    ("pa", "long_axis"), [(0.0, "north_south"), (90.0, "east_west")]
)
def test_elliptical_gaussian_major_axis_follows_north_to_east_pa(
    pa, long_axis
):
    image = onp.asarray(
        EllipticalGaussian(20.0, 0.3, pa).render(npix=41, fov_mas=60.0)
    )
    centre = image.shape[0] // 2
    column, row = image[:, centre].sum(), image[centre, :].sum()
    # Row index runs North to South, column index East to West.
    assert (column > row) == (long_axis == "north_south")


def test_elliptical_gaussian_pa_45_lies_north_east_to_south_west():
    image = onp.asarray(
        EllipticalGaussian(20.0, 0.3, 45.0).render(npix=41, fov_mas=60.0)
    )
    # North-East is the top left (row 0, column 0), so the major axis is
    # the main diagonal.
    assert onp.trace(image) > onp.trace(image[:, ::-1])


def test_gaussian_arc_peaks_at_its_position_angle_from_the_centre():
    image = onp.asarray(
        GaussianArc(20.0, 2.0, 10.0, 90.0).render(npix=41, fov_mas=60.0)
    )
    row, col = onp.unravel_index(image.argmax(), image.shape)
    # East of the centre: the left half (columns run East to West).
    assert row == 20 and col < 20
    assert abs((20 - col) * 1.5 - 20.0) < 2.0


def test_gaussian_arc_bends_towards_its_centre():
    image = onp.asarray(
        GaussianArc(20.0, 2.0, 30.0, 90.0).render(npix=81, fov_mas=60.0)
    )
    # The arc's ends, North and South of the peak, curve back to the West.
    ys, xs = onp.nonzero(image > 0.3 * image.max())
    north, south = xs[ys == ys.min()].mean(), xs[ys == ys.max()].mean()
    peak = onp.unravel_index(image.argmax(), image.shape)[1]
    assert north > peak and south > peak


def test_gaussian_arc_with_a_large_radius_is_an_elliptical_gaussian():
    # Equal up to the curvature of the arc.
    u, v = onp.array([10.0, -25.0, 40.0]), onp.array([5.0, 30.0, -12.0])
    arc = GaussianArc(1e3, 3.0, 10.0, 30.0)
    ellipse = EllipticalGaussian(
        onp.hypot(10.0, 3.0), 3.0 / onp.hypot(10.0, 3.0), 120.0
    )
    assert onp.allclose(
        onp.abs(arc.model(u, v, 2.2e-6)),
        onp.abs(ellipse.model(u, v, 2.2e-6)),
        atol=1e-3,
    )


@pytest.mark.parametrize(
    "radius, length",
    [(5.0, 4.0), (15.0, 20.0), (5.0, 40.0)],
    ids=["short", "long", "longer_than_circle"],
)
def test_gaussian_arc_matches_a_gaussian_once_round_the_circle(radius, length):
    # Independent reference: a dense sum round the whole circle, weighted
    # by a Gaussian in arc length wrapped to [-π R, π R).
    width, pa, wavel = 0.6, 250.0, 1.6e-6
    rng = onp.random.default_rng(3)
    u, v = rng.uniform(-100.0, 100.0, (2, 50))
    psi = 2.0 * onp.pi * onp.arange(65536) / 65536
    s = radius * onp.angle(onp.exp(1j * (psi - onp.deg2rad(pa))))
    sigma = length / (2.0 * onp.sqrt(2.0 * onp.log(2.0)))
    weight = onp.exp(-0.5 * (s / sigma) ** 2)
    x, y = radius * onp.sin(psi), radius * onp.cos(psi)
    fu, fv = u / wavel * _MAS2RAD_REF, v / wavel * _MAS2RAD_REF
    curve = onp.exp(-2j * onp.pi * (onp.outer(fu, x) + onp.outer(fv, y)))
    blur = onp.exp(
        -((onp.pi * width) ** 2) * (fu**2 + fv**2) / (4 * onp.log(2))
    )
    expected = curve @ weight / weight.sum() * blur
    with jax.enable_x64(True):
        arc = GaussianArc(radius, width, length, pa, nodes=4096)
        got = onp.asarray(arc.model(u, v, wavel))
    assert onp.max(onp.abs(got - expected)) < 1e-6


def test_rim_gradients_are_finite_at_zero_baseline():
    # Flagged samples can sit at u = v = 0; their gradients must not be NaN.
    u, v = np.array([0.0, 20.0]), np.array([0.0, -10.0])

    def power(inc, pa):
        rim = ModulatedGaussianRim(
            30.0,
            4.0,
            inc,
            pa,
            az_amps=np.array([0.3]),
            az_pas=np.array([40.0]),
        )
        return np.sum(np.abs(rim.model(u, v, 2.2e-6)) ** 2)

    grads = jax.grad(power, argnums=(0, 1))(60.0, 10.0)
    assert all(onp.isfinite(float(g)) for g in grads)
    rim = ModulatedGaussianRim(30.0, 4.0, 60.0, 10.0)
    assert onp.isclose(complex(rim.model(u, v, 2.2e-6)[0]), 1.0)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"fwhm": 0.0}, "fwhm"),
        ({"fwhm": -3.0}, "fwhm"),
        ({"fwhm": onp.nan}, "fwhm"),
        ({"ratio": 0.0}, "ratio"),
        ({"ratio": 1.5}, "ratio"),
    ],
)
def test_elliptical_gaussian_rejects_invalid_shapes(kwargs, match):
    with pytest.raises(ValueError, match=match):
        EllipticalGaussian(**({"fwhm": 10.0, "ratio": 0.5} | kwargs))


def test_elliptical_gaussian_is_physical_checks_traced_shapes():
    good = EllipticalGaussian(10.0, 0.5, 30.0)
    assert bool(good.is_physical())
    assert not bool(good.set("fwhm", np.array(-1.0)).is_physical())
    assert not bool(good.set("ratio", np.array(1.2)).is_physical())


@pytest.mark.parametrize("name", ["radius", "width", "length"])
@pytest.mark.parametrize("value", [0.0, -2.0, onp.inf])
def test_gaussian_arc_rejects_invalid_shapes(name, value):
    kwargs = {"radius": 20.0, "width": 2.0, "length": 15.0} | {name: value}
    with pytest.raises(ValueError, match=name):
        GaussianArc(**kwargs)


def test_gaussian_arc_needs_two_nodes_and_checks_traced_shapes():
    with pytest.raises(ValueError, match="nodes"):
        GaussianArc(20.0, 2.0, 15.0, nodes=1)
    good = GaussianArc(20.0, 2.0, 15.0)
    assert bool(good.is_physical())
    assert not bool(good.set("radius", np.array(-1.0)).is_physical())


def test_gaussian_arc_uses_the_trapezoidal_rule():
    # Equally spaced nodes; the end points carry half the weight of an
    # interior point of the same height, and the weights sum to one.
    arc = GaussianArc(20.0, 2.0, 15.0, nodes=9)
    x, y, w = (onp.asarray(a) for a in arc.curve())
    s = onp.linspace(-6.0, 6.0, 9)
    gauss = onp.exp(-0.5 * s**2)
    assert onp.isclose(w.sum(), 1.0)
    assert onp.allclose(
        w / gauss, (w / gauss)[1] * onp.r_[0.5, onp.ones(7), 0.5]
    )


def test_gaussian_arc_longer_than_the_circle_goes_round_it_once():
    # 6σ exceeds π R: the nodes span the circle, both ends at the antipode.
    arc = GaussianArc(5.0, 1.0, 40.0, pa=30.0, nodes=9)
    x, y, _ = (onp.asarray(a) for a in arc.curve())
    antipode = 5.0 * onp.array(
        [onp.sin(onp.radians(210.0)), onp.cos(onp.radians(210.0))]
    )
    assert onp.allclose([x[0], y[0]], antipode, atol=1e-5)
    assert onp.allclose([x[-1], y[-1]], antipode, atol=1e-5)


def test_truncated_cone_limits_and_symmetries():
    rng = onp.random.default_rng(3)
    u, v = rng.uniform(-60.0, 60.0, (2, 30))
    cone = dict(tip=5.0, alpha=30.0, s0=4.0, length=8.0, width=1.0, pa=40.0)
    # Optically thin: the sign of the tilt does not matter.
    assert onp.allclose(
        TruncatedCone(**cone, tilt=25.0).model(u, v, 1e-6),
        TruncatedCone(**cone, tilt=-25.0).model(u, v, 1e-6),
    )
    # Pointing at the observer (tilt 90), a circular cone is a set of
    # concentric face-on rings: real and symmetric under rotation.
    face_on = TruncatedCone(**cone, tilt=90.0)
    angle = onp.deg2rad(70.0)
    ur, vr = (
        u * onp.cos(angle) - v * onp.sin(angle),
        u * onp.sin(angle) + v * onp.cos(angle),
    )
    assert onp.allclose(
        face_on.model(u, v, 1e-6), face_on.model(ur, vr, 1e-6), atol=1e-6
    )
    assert onp.allclose(onp.imag(face_on.model(u, v, 1e-6)), 0.0, atol=1e-6)
    # Across the projected axis, ratio does not matter at tilt 0.
    assert onp.allclose(
        TruncatedCone(**cone, tilt=0.0, ratio=0.3).model(u, v, 1e-6),
        TruncatedCone(**cone, tilt=0.0).model(u, v, 1e-6),
    )


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"alpha": 95.0}, "alpha"),
        ({"length": 0.0}, "length"),
        ({"tilt": 120.0}, "tilt"),
        ({"ratio": -1.0}, "ratio"),
    ],
)
def test_truncated_cone_rejects_bad_shapes(kwargs, match):
    base = dict(tip=5.0, alpha=30.0, s0=4.0, length=8.0, width=1.0)
    with pytest.raises(ValueError, match=match):
        TruncatedCone(**{**base, **kwargs})
