"""Positive and von Mises azimuthal harmonics, and virgil.harmonics."""

import itertools
import warnings

import equinox as eqx
import jax
import jax.numpy as np
import numpy as onp
import pytest
from scipy.special import ive

from virgil import GaussianArc, ModulatedGaussianRim, TruncatedCone
from virgil import harmonics as H
from virgil._geometry import (
    image_coordinates,
    undo_elliptical_transf_spat_freq,
)
from virgil._utils import dtor, mas2rad
from virgil.models import (
    GaussianRing,
    Harmonics,
    MultipoleDisk,
    RadialLinear,
    ThinRing,
    VonMises,
)

WAVEL = 2.2e-6
_MAS2RAD_REF = onp.pi / 180.0 / 3600.0 / 1000.0


def _uv(n=30, seed=0, qmax=0.3, qmin=0.01):
    """Baselines (metres) with spatial frequencies in cycles/mas."""
    rng = onp.random.default_rng(seed)
    q = rng.uniform(qmin, qmax, n) / _MAS2RAD_REF
    theta = rng.uniform(0.0, 2.0 * onp.pi, n)
    return q * onp.cos(theta) * WAVEL, q * onp.sin(theta) * WAVEL


def _cvis(model, u, v, wavel=WAVEL):
    return onp.asarray(model.model(np.asarray(u), np.asarray(v), wavel))


def _numpy_step_up(k):
    """An independent step-up recursion, in numpy."""
    h = onp.ones(1, dtype=complex)
    for j, kj in enumerate(k, start=1):
        h = onp.concatenate([h, [0.0]]) + kj * onp.conj(
            onp.concatenate([[0.0], h[::-1]])
        )
    return h


def _poly_from_roots(roots):
    h = onp.ones(1, dtype=complex)
    for root in roots:
        h = onp.concatenate([h, [0.0]]) - onp.concatenate([[0.0], h]) / root
    return h


def _random_latents(M, seed, scale=0.6):
    rng = onp.random.default_rng(seed)
    return scale * (rng.normal(size=M) + 1j * rng.normal(size=M))


# --- the positive form ---------------------------------------------------------


@pytest.mark.parametrize("M", [1, 2, 3, 5])
def test_positive_form_is_the_normalized_modulus_squared_of_h(M):
    w = _random_latents(M, M)
    harmonics = Harmonics(w=w)
    k = onp.tanh(abs(w)) * w / abs(w)
    h = _numpy_step_up(k)
    theta = onp.linspace(0.0, 2.0 * onp.pi, 97, endpoint=False)
    z = onp.exp(1j * theta)
    expected = abs(onp.polyval(h[::-1], z)) ** 2 / onp.sum(abs(h) ** 2)
    got = harmonics.factor(np.zeros_like(theta), theta)
    assert onp.allclose(got, expected, atol=2e-6)
    # I_m = sum_j h_{j+m} conj(h_j) / sum_j |h_j|^2.
    image = onp.asarray(harmonics.image_harmonics())
    reference = [
        onp.sum(h[m:] * onp.conj(h[: M + 1 - m])) / onp.sum(abs(h) ** 2)
        for m in range(M + 1)
    ]
    assert onp.allclose(image, reference, atol=2e-6)
    # And the linear coefficients the engine uses are its cos and sin terms.
    c, s = (onp.asarray(a)[:, 0] for a in harmonics.coefficients(np.zeros(1)))
    series = 1.0 + sum(
        c[m - 1] * onp.cos(m * theta) + s[m - 1] * onp.sin(m * theta)
        for m in range(1, M + 1)
    )
    assert onp.allclose(series, expected, atol=2e-6)


@pytest.mark.parametrize("M", [1, 2, 4])
def test_positive_image_has_exactly_degree_m_and_unit_mean(M):
    harmonics = Harmonics(w=_random_latents(M, 10 + M, scale=0.9))
    n = 64
    theta = 2.0 * onp.pi * onp.arange(n) / n
    profile = onp.asarray(harmonics.factor(np.zeros(n), np.asarray(theta)))
    spectrum = onp.fft.fft(profile) / n
    assert onp.isclose(spectrum[0].real, 1.0, atol=1e-6)
    assert onp.max(abs(spectrum[M + 1 : n - M])) < 1e-6
    assert abs(spectrum[M]) > 1e-3  # and the top order is really there


@pytest.mark.parametrize("M", [1, 2, 3, 4, 5, 6])
def test_random_latents_always_give_a_minimum_phase_positive_image(M):
    with jax.enable_x64(True):
        for seed in range(8):
            w = _random_latents(M, 100 * M + seed, scale=1.5)
            harmonics = Harmonics(w=w)
            h = onp.asarray(harmonics.polynomial(np.zeros(())))
            roots = onp.roots(h[::-1])
            assert onp.all(abs(roots) > 1.0)
            theta = onp.linspace(0.0, 2.0 * onp.pi, 361)
            profile = onp.asarray(
                harmonics.factor(np.zeros_like(theta), np.asarray(theta))
            )
            assert profile.min() >= 0.0
            assert float(harmonics.min_intensity(np.zeros(1))) >= 0.0


def test_positive_form_is_finite_and_differentiable_at_zero_latents():
    # k = w tanh|w| / |w| has a removable singularity at w = 0.
    def power(re, im):
        harmonics = Harmonics(re=(re, 0.1), im=(im, -0.2), form="positive")
        c, s = harmonics.coefficients(np.array([1.0, 2.0]))
        return np.sum(c) + 2.0 * np.sum(s) + np.sum(c * s)

    for point in [(0.0, 0.0), (1e-3, 0.0), (0.0, 0.02), (0.3, -0.4)]:
        value = power(*point)
        grads = jax.grad(power, argnums=(0, 1))(*point)
        assert onp.isfinite(float(value))
        assert all(onp.isfinite(float(g)) for g in grads)
    # At exactly zero the first order is k_1 = w_1: the derivative of the
    # first harmonic is 2 (c_1 = 2 Re k_1 / (1 + |k|^2) ...).
    first = lambda re: Harmonics(  # noqa: E731
        re=(re,), im=(0.0,), form="positive"
    ).coefficients(np.zeros(()))[0][0]
    assert onp.isclose(float(jax.grad(first)(0.0)), 2.0, atol=1e-5)


def test_tanh_ratio_series_matches_the_exact_value_across_the_switch():
    with jax.enable_x64(True):
        x = onp.array([0.0, 1e-6, 0.0099, 0.01001, 0.05, 0.5])
        k = H.reflection_coefficients(np.asarray(x), np.zeros_like(x))
        expected = onp.tanh(x)
        assert onp.allclose(onp.asarray(k).real, expected, rtol=0, atol=1e-12)
        grad = jax.vmap(
            jax.grad(lambda t: H.reflection_coefficients(t, 0.0).real)
        )(np.asarray(x))
        assert onp.allclose(grad, 1.0 / onp.cosh(x) ** 2, atol=1e-10)


def test_degree_one_positive_form_fits_a_lopsided_ring_exactly():
    # 1 + 0.9 cos(theta): a real square [1 + f]^2 cannot (8 % rms), and
    # |h|^2 with h = 1 + k z does, with |k| = 0.63.
    positive = H.from_image_harmonics([1.0, 0.45], form="positive")
    k = onp.tanh(abs(complex(positive.re[0], positive.im[0])))
    assert onp.isclose(k, 0.627, atol=2e-3)
    theta = onp.linspace(0.0, 2.0 * onp.pi, 50)
    profile = positive.factor(np.zeros_like(theta), np.asarray(theta))
    assert onp.allclose(profile, 1.0 + 0.9 * onp.cos(theta), atol=1e-6)
    # An axisymmetric ring (w = 0) is uniform.
    flat = Harmonics(degree=3)
    assert flat.form == "positive"
    assert onp.allclose(flat.factor(np.zeros(5), np.arange(5.0)), 1.0)


def test_root_flipped_twins_share_an_image_but_map_to_one_canonical_w():
    # Reflecting a root of h through the unit circle leaves the image alone
    # (the spectral-factorization ambiguity, 2^M copies); only the
    # minimum-phase copy is a point of the reflection-coefficient chart.
    with jax.enable_x64(True):
        k = onp.array([0.6, 0.6, 0.6], dtype=complex)
        h = _numpy_step_up(k)
        roots = onp.roots(h[::-1])
        canonical = Harmonics(
            w=onp.arctanh(abs(k)) * k / abs(k)
        ).image_harmonics()
        twins = 0
        for flips in itertools.product([0, 1], repeat=3):
            if not any(flips):
                continue
            twin = _poly_from_roots(
                [
                    1.0 / onp.conj(r) if flip else r
                    for r, flip in zip(roots, flips)
                ]
            )
            image = H.image_harmonics_from_h(np.asarray(twin))
            assert onp.allclose(image, canonical, atol=1e-12)
            # Every twin has some step-down coefficient outside the unit
            # disk, so none is a point of the chart (for k = 0.6 and the
            # root ordering of np.roots the single-flip twin of the real
            # root has |k| = 1.64, 1.51, 0.79).
            raw, _ = H.step_down(twin, clip=-1e9)
            assert onp.max(abs(raw)) > 1.0 + 1e-3
            # The factorization of the shared image gives the one canonical
            # latent vector, whichever twin made it.
            back = H.from_image_harmonics(onp.asarray(image), "positive")
            w_back = onp.array(back.re) + 1j * onp.array(back.im)
            assert onp.allclose(w_back, onp.arctanh(0.6), atol=1e-9)
            twins += 1
        assert twins == 7


@pytest.mark.parametrize("M", [1, 2, 3, 4, 6])
def test_positive_to_image_and_back_round_trips(M):
    with jax.enable_x64(True):
        w = _random_latents(M, 7 * M, scale=0.5)
        positive = Harmonics(w=w)
        image = onp.asarray(positive.image_harmonics())
        back = positive.convert("linear").convert("positive")
        assert back.form == "positive" and back.degree == M
        w_back = onp.array(back.re) + 1j * onp.array(back.im)
        assert onp.allclose(w_back, w, atol=1e-8)
        assert onp.allclose(back.image_harmonics(), image, atol=1e-10)
        # A third trip through the image harmonics themselves.
        again = H.from_image_harmonics(image, "positive")
        assert onp.allclose(again.image_harmonics(), image, atol=1e-10)


def test_linear_and_image_harmonics_agree_exactly():
    c, s = (0.2, -0.1, 0.05), (0.15, 0.1)
    linear = Harmonics(c=c, s=s)
    image = onp.asarray(linear.image_harmonics())
    assert onp.isclose(image[0], 1.0)
    # c = 2 Re I / I_0 and s = -2 Im I / I_0, against the engine's own
    # convention: the image is 1 + sum c cos + s sin.
    assert onp.allclose(2.0 * image[1:].real, c)
    assert onp.allclose(-2.0 * image[1:].imag, (0.15, 0.1, 0.0))
    theta = onp.linspace(0.0, 2.0 * onp.pi, 40)
    series = onp.real(
        image[0]
        + 2.0 * sum(image[m] * onp.exp(1j * m * theta) for m in (1, 2, 3))
    )
    assert onp.allclose(
        series,
        linear.factor(np.zeros_like(theta), np.asarray(theta)),
        atol=1e-6,
    )
    back = linear.convert("linear")
    assert back is linear
    again = H.from_image_harmonics(image, "linear")
    assert onp.allclose([float(x) for x in again.c], c)
    assert onp.allclose([float(x) for x in again.s], (0.15, 0.1, 0.0))


def test_positive_linear_image_converts_to_the_positive_form():
    linear = Harmonics(c=(0.5, 0.1), s=(0.2, -0.1))
    positive = linear.convert("positive")
    assert positive.form == "positive"
    theta = onp.linspace(0.0, 2.0 * onp.pi, 61)
    r = np.zeros_like(theta)
    assert onp.allclose(
        positive.factor(r, np.asarray(theta)),
        linear.factor(r, np.asarray(theta)),
        atol=2e-6,
    )


def test_a_negative_image_has_no_positive_form_and_a_touching_one_is_clipped():
    negative = Harmonics(c=1.2)  # 1 + 1.2 cos goes negative
    with pytest.raises(ValueError, match="negative"):
        negative.convert("positive")
    touching = Harmonics(c=1.0)  # 1 + cos: a double zero at theta = pi
    with pytest.warns(UserWarning, match="clipped"):
        positive = touching.convert("positive")
    k = onp.tanh(abs(complex(positive.re[0], positive.im[0])))
    assert onp.isclose(k, 1.0 - 1e-6, atol=1e-7)
    with pytest.raises(ValueError, match="I_0"):
        H.from_image_harmonics([0.0, 0.1], "positive")
    with pytest.raises(ValueError, match="form"):
        H.from_image_harmonics([1.0, 0.1], "log")
    with pytest.raises(ValueError, match="1-D"):
        H.from_image_harmonics(onp.ones((2, 2)))


def test_image_with_a_lower_true_degree_gets_zero_outer_latents():
    positive = H.from_image_harmonics([1.0, 0.3, 0.0, 0.0], "positive")
    assert positive.degree == 3
    assert onp.allclose([float(x) for x in positive.re[1:]], 0.0, atol=1e-6)
    assert onp.allclose([float(x) for x in positive.im], 0.0, atol=1e-6)


def test_convert_needs_constants_and_known_forms():
    radial = Harmonics(c=RadialLinear(0.1, 0.01, 5.0))
    with pytest.raises(ValueError, match="constant"):
        radial.convert("positive")
    with pytest.raises(ValueError, match="radius"):
        radial.image_harmonics()
    assert onp.allclose(
        radial.image_harmonics(np.array([5.0, 6.0]))[1].real, [0.05, 0.055]
    )
    with pytest.raises(ValueError, match="positive form"):
        Harmonics(c=0.1).polynomial(np.zeros(()))


def test_harmonics_inputs_are_validated():
    with pytest.raises(ValueError, match="linear"):
        Harmonics(c=0.1, form="positive")
    with pytest.raises(ValueError, match="positive"):
        Harmonics(re=0.1, form="linear")
    with pytest.raises(ValueError, match="form must be"):
        Harmonics(form="log")
    with pytest.raises(ValueError, match="not both"):
        Harmonics(w=0.1, re=0.1)
    with pytest.raises(ValueError, match="degree"):
        Harmonics(w=(0.1, 0.2), degree=1)
    h = Harmonics(w=0.3 + 0.2j, degree=3)
    assert h.degree == 3 and len(h.re) == 3 and h.c == () and h.s == ()


def test_positive_latents_can_be_functions_of_the_radius():
    harmonics = Harmonics(re=RadialLinear(0.0, 0.1, 5.0), degree=2)
    r = np.array([5.0, 7.0, 9.0])
    image = onp.asarray(harmonics.image_harmonics(r))
    assert image.shape == (3, 3)
    assert onp.allclose(image[1, 0], 0.0, atol=1e-7)  # w = 0 at r = r0
    assert abs(image[1, 2]) > abs(image[1, 1]) > 0.0
    assert float(harmonics.min_intensity(r)) >= 0.0


@pytest.mark.parametrize(
    "disk",
    [
        MultipoleDisk(
            GaussianRing(9.0, 1.5),
            harmonics=Harmonics(w=(0.5 + 0.2j, -0.3j, 0.1)),
            inc=45.0,
            pa=70.0,
            dra=1.0,
            ddec=-2.0,
            blur_fwhm=0.5,
        ),
        MultipoleDisk(
            ThinRing(8.0),
            harmonics=Harmonics(w=(0.8, 0.4j)),
            inc=30.0,
            pa=-20.0,
            blur_fwhm=1.5,
        ),
    ],
    ids=["gaussian ring", "thin ring"],
)
def test_positive_harmonics_match_the_dft_of_the_rendered_image(disk):
    npix, fov = 384, 60.0
    image = onp.asarray(disk.render(npix=npix, fov_mas=fov))
    assert image.min() > -1e-6 * image.max()  # FFT blur rounding
    xx, yy = (onp.asarray(a).ravel() for a in image_coordinates(npix, fov))
    u, v = _uv(30, seed=4, qmax=0.2)
    phase = onp.exp(
        -2j
        * onp.pi
        * _MAS2RAD_REF
        * (onp.outer(u, xx) + onp.outer(v, yy))
        / WAVEL
    )
    dft = phase @ image.ravel() / image.sum()
    assert onp.max(onp.abs(dft - _cvis(disk, u, v))) < 4e-3


def test_positive_disk_visibility_is_one_at_the_origin_with_finite_gradients():
    harmonics = Harmonics(w=(0.4 + 0.1j, 0.2j, 0.0))
    disk = MultipoleDisk(GaussianRing(6.0, 1.0), harmonics=harmonics, inc=30.0)
    u = np.array([0.0, 12.0, 25.0])
    vis = disk.model(u, 0.5 * u, WAVEL)
    assert onp.isclose(complex(vis[0]), 1.0, atol=1e-6)

    def power(re0, im0):
        d = disk.set("harmonics.re", (re0,) + tuple(disk.harmonics.re[1:]))
        d = d.set("harmonics.im", (im0,) + tuple(disk.harmonics.im[1:]))
        return np.sum(np.abs(d.model(u, 0.5 * u, WAVEL)) ** 2)

    for point in [(0.4, 0.1), (0.0, 0.0)]:
        grads = jax.grad(power, argnums=(0, 1))(*point)
        assert all(onp.isfinite(float(g)) for g in grads)
    assert bool(disk.is_physical())


# --- von Mises -----------------------------------------------------------------


@pytest.mark.parametrize("kappa", [0.0, 0.5, 4.0, 30.0])
def test_von_mises_harmonics_are_bessel_ratios_with_the_azimuth_phase(kappa):
    degree = 12 if kappa < 10 else 40
    with jax.enable_x64(True):
        vm = VonMises(kappa, 35.0, degree=degree)
        image = onp.asarray(vm.image_harmonics())
        orders = onp.arange(degree + 1)
        # I_m(kappa) / I_0(kappa) from scipy's exponentially scaled ive.
        ratio = ive(orders, kappa) / ive(0, kappa)
        expected = ratio * onp.exp(-1j * orders * onp.radians(35.0))
        assert onp.allclose(image, expected, atol=1e-12)
        c, s = (onp.asarray(a)[:, 0] for a in vm.coefficients(np.zeros(1)))
        assert onp.allclose(c, 2 * expected[1:].real, atol=1e-12)
        assert onp.allclose(s, -2 * expected[1:].imag, atol=1e-12)


def test_von_mises_peaks_at_its_azimuth_and_reports_truncation():
    vm = VonMises(6.0, 100.0, degree=40)
    theta = onp.radians(onp.linspace(0.0, 360.0, 721))
    profile = onp.asarray(vm.factor(np.zeros_like(theta), np.asarray(theta)))
    assert onp.isclose(onp.degrees(theta[profile.argmax()]), 100.0, atol=0.6)
    exact = onp.exp(6.0 * (onp.cos(theta - onp.radians(100.0)) - 1.0))
    exact = exact / exact[:-1].mean()
    assert onp.allclose(profile, exact, atol=1e-5)
    assert float(vm.truncation_power()) < 1e-10
    # A low degree drops power, and a lower one drops more.
    low, lower = (VonMises(6.0, 100.0, degree=d) for d in (6, 3))
    assert (
        1e-4 < float(low.truncation_power()) < float(lower.truncation_power())
    )
    # The conversion truncates (it reports the same dropped power).
    assert H.to_image_harmonics(low).shape == (7,)


def test_von_mises_is_a_multipole_disk_arc_that_matches_a_dense_ring_sum():
    kappa, theta0, radius, width = 8.0, 40.0, 7.0, 1.2
    disk = MultipoleDisk(
        ThinRing(radius),
        harmonics=VonMises(kappa, theta0, degree=36),
        blur_fwhm=width,
    )
    u, v = _uv(40, seed=2, qmax=0.25)
    psi = 2.0 * onp.pi * onp.arange(4096) / 4096
    weight = onp.exp(kappa * onp.cos(psi - onp.radians(theta0)))
    x, y = radius * onp.sin(psi), radius * onp.cos(psi)
    fu, fv = u / WAVEL * _MAS2RAD_REF, v / WAVEL * _MAS2RAD_REF
    ring = onp.exp(-2j * onp.pi * (onp.outer(fu, x) + onp.outer(fv, y)))
    blur = onp.exp(
        -((onp.pi * width) ** 2) * (fu**2 + fv**2) / (4 * onp.log(2))
    )
    expected = ring @ weight / weight.sum() * blur
    assert onp.max(onp.abs(_cvis(disk, u, v) - expected)) < 3e-6
    # kappa = 0 is a uniform ring: no closure phase, real visibilities.
    flat = MultipoleDisk(ThinRing(radius), harmonics=VonMises(0.0, degree=4))
    assert onp.max(onp.abs(_cvis(flat, u, v).imag)) < 1e-6
    # Visibility 1 at the origin, and differentiable in kappa and theta0.
    assert onp.isclose(
        complex(disk.model(np.zeros(1), np.zeros(1), WAVEL)[0]), 1
    )

    def power(kap, th):
        d = disk.set("harmonics.kappa", kap).set("harmonics.theta0", th)
        return np.sum(np.abs(d.model(u, v, WAVEL)) ** 2)

    for point in [(8.0, 40.0), (0.0, 0.0)]:
        grads = jax.grad(power, argnums=(0, 1))(*point)
        assert all(onp.isfinite(float(g)) for g in grads)


def test_von_mises_degree_is_chosen_from_kappa_and_inputs_are_checked():
    assert VonMises(0.5).degree < VonMises(5.0).degree < VonMises(80.0).degree
    assert VonMises(RadialLinear(1.0, 0.1, 5.0)).degree == 32
    with pytest.warns(UserWarning, match="truncating"):
        assert VonMises(1e5).degree == 256
    with pytest.raises(ValueError, match="kappa"):
        VonMises(-1.0)
    with pytest.raises(ValueError, match="scalar"):
        VonMises([1.0, 2.0])
    with pytest.raises(ValueError, match="degree"):
        VonMises(1.0, degree=0)
    with pytest.raises(ValueError, match="n_fft"):
        VonMises(1.0, degree=20, n_fft=16)
    radial = VonMises(RadialLinear(1.0, 0.5, 5.0), degree=8)
    with pytest.raises(ValueError, match="radius"):
        radial.truncation_power()
    r = np.array([5.0, 9.0])
    assert radial.truncation_power(r).shape == (2,)
    assert radial.image_harmonics(r).shape == (9, 2)
    with pytest.raises(ValueError, match="constant"):
        radial.convert("linear")
    # Converting to the linear form is the truncated series, exactly.
    linear = VonMises(3.0, 20.0, degree=10).convert("linear")
    assert linear.form == "linear" and linear.degree == 10
    with pytest.raises(TypeError, match="Harmonics or VonMises"):
        MultipoleDisk(ThinRing(5.0), harmonics=0.3)


# --- legacy conversions ---------------------------------------------------------


def _old_modulated_rim_cvis(u, v, diam, fwhm, inc, pa, az_amps, az_phis):
    """The pre-subsumption visibility of ModulatedGaussianRim: a test-local
    copy of the old ``_cvis_centred_rim`` and of
    ``cvis_radial_dirac_delta_modulated``."""
    from jaxbessel import bessel_jn

    stretch = np.maximum(np.cos(inc * dtor), 1e-8)
    ut, vt = undo_elliptical_transf_spat_freq(u, v, pa, stretch)
    amps = np.concatenate([np.array([1.0]), az_amps])
    phis = np.concatenate([np.array([0.0]), az_phis])
    orders = np.arange(amps.size)
    origin = (ut == 0) & (vt == 0)
    u_safe, v_safe = np.where(origin, 1.0, ut), np.where(origin, 1.0, vt)
    norm = np.where(origin, 0.0, np.hypot(u_safe, v_safe))
    ang = np.where(origin, 0.0, np.arctan2(u_safe, v_safe))
    bes = bessel_jn(amps.size - 1, 2.0 * np.pi * norm * (diam / 2.0) * mas2rad)

    def term(amp, phi, order):
        return (
            amp
            * np.exp(-0.5j * np.pi * order)
            * np.cos(order * (ang - phi * dtor))
            * bes[order]
        )

    cvis = np.sum(jax.vmap(term)(amps, phis, orders), axis=0)
    fwhm_rad = fwhm * mas2rad
    envelope = np.exp(
        -(np.pi**2) * fwhm_rad**2 * (ut**2 + vt**2) / (4.0 * np.log(2.0))
    )
    return cvis * envelope


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(diam=14.0, fwhm=2.0, inc=0.0, pa=0.0),
        dict(
            diam=14.0,
            fwhm=2.0,
            inc=55.0,
            pa=30.0,
            az_amps=[0.4],
            az_pas=[75.0],
        ),
        dict(
            diam=9.0,
            fwhm=1.0,
            inc=70.0,
            pa=-40.0,
            az_amps=[0.3, 0.2, 0.1],
            az_pas=[10.0, 200.0, 95.0],
        ),
        dict(
            diam=30.0, fwhm=4.0, inc=89.0, pa=170.0, az_amps=0.6, az_pas=300.0
        ),
    ],
    ids=["symmetric face-on", "first order", "three orders", "edge-on"],
)
def test_modulated_rim_on_the_engine_equals_the_old_code_to_rounding(kwargs):
    rim = ModulatedGaussianRim(**kwargs)
    amps = onp.atleast_1d(kwargs.get("az_amps", []))
    pas = onp.atleast_1d(kwargs.get("az_pas", []))
    u, v = _uv(60, seed=8, qmax=0.3)
    u = onp.concatenate([[0.0], u])
    v = onp.concatenate([[0.0], v])
    uu, vv = u / WAVEL, v / WAVEL
    old = onp.asarray(
        _old_modulated_rim_cvis(
            np.asarray(uu),
            np.asarray(vv),
            kwargs["diam"],
            kwargs["fwhm"],
            kwargs["inc"],
            kwargs["pa"],
            np.asarray(amps, dtype=float),
            np.asarray(pas, dtype=float) - kwargs["pa"],
        )
    )
    assert onp.max(onp.abs(_cvis(rim, u, v) - old)) < 5e-7
    with jax.enable_x64(True):
        rim64 = ModulatedGaussianRim(**kwargs)
        old64 = onp.asarray(
            _old_modulated_rim_cvis(
                np.asarray(uu),
                np.asarray(vv),
                kwargs["diam"],
                kwargs["fwhm"],
                kwargs["inc"],
                kwargs["pa"],
                np.asarray(amps, dtype=float),
                np.asarray(pas, dtype=float) - kwargs["pa"],
            )
        )
        assert onp.max(onp.abs(_cvis(rim64, u, v) - old64)) < 1e-12


def test_rim_legacy_conversion_round_trips_and_matches_a_multipole_disk():
    amps, pas, pa = (0.3, 0.2, 0.1), (40.0, 100.0, 250.0), 30.0
    harmonics = H.from_legacy(amps, pas, pa, kind="rim")
    # c_m = A cos(m (az_pa - pa)), s_m = A sin(m (az_pa - pa)).
    angle = onp.arange(1, 4) * onp.radians(onp.array(pas) - pa)
    assert onp.allclose([float(x) for x in harmonics.c], amps * onp.cos(angle))
    assert onp.allclose([float(x) for x in harmonics.s], amps * onp.sin(angle))
    back_amps, back_pas = H.to_legacy(harmonics, pa, kind="rim")
    assert onp.allclose(back_amps, amps, atol=1e-6)
    turn = 360.0 / onp.arange(1, 4)  # each az_pa is defined modulo 360 / m
    assert onp.allclose(
        ((back_pas - onp.array(pas) + turn / 2) % turn) - turn / 2,
        0.0,
        atol=1e-4,
    )
    rim = ModulatedGaussianRim(
        diam=12.0, fwhm=1.5, inc=50.0, pa=pa, az_amps=amps, az_pas=pas
    )
    disk = MultipoleDisk(
        ThinRing(6.0), harmonics=harmonics, inc=50.0, pa=pa, blur_fwhm=1.5
    )
    u, v = _uv(30, seed=9, qmax=0.2)
    assert onp.max(onp.abs(_cvis(rim, u, v) - _cvis(disk, u, v))) < 1e-6
    # The legacy rim brightens az_pa on the sky when face-on: its image
    # peaks at the position angle az_pas.
    image = onp.asarray(
        ModulatedGaussianRim(
            diam=20.0, fwhm=2.0, inc=0.0, pa=0.0, az_amps=0.8, az_pas=120.0
        ).render(npix=129, fov_mas=40.0)
    )
    row, col = onp.unravel_index(image.argmax(), image.shape)
    xx, yy = (onp.asarray(a) for a in image_coordinates(129, 40.0))
    assert onp.isclose(
        onp.degrees(onp.arctan2(xx[row, col], yy[row, col])), 120.0, atol=3.0
    )
    # And the same through the positive form's image (a converted rim).
    assert H.to_image_harmonics(harmonics).shape == (4,)
    with pytest.raises(ValueError, match="kind"):
        H.from_legacy(amps, pas, pa, kind="arc")


def test_cone_legacy_conversion_matches_the_engine_wrapper():
    amps, pas, pa = (0.4, 0.15), (190.0, 20.0), 40.0
    harmonics = H.from_legacy(amps, pas, pa, kind="cone")
    ph = onp.arange(1, 3) * onp.radians(pa + 90.0 - onp.array(pas))
    sign = (-1.0) ** onp.arange(1, 3)
    assert onp.allclose(
        [float(x) for x in harmonics.c], sign * amps * onp.cos(ph), atol=1e-6
    )
    assert onp.allclose(
        [float(x) for x in harmonics.s], -sign * amps * onp.sin(ph), atol=1e-6
    )
    back_amps, back_pas = H.to_legacy(harmonics, pa, kind="cone")
    assert onp.allclose(back_amps, amps, atol=1e-6)
    again = H.from_legacy(back_amps, back_pas, pa, kind="cone")
    assert onp.allclose(
        [float(x) for x in again.c], [float(x) for x in harmonics.c], atol=1e-5
    )
    cone = dict(tip=5.0, alpha=30.0, s0=4.0, length=8.0, width=1.0, pa=pa)
    modulated = TruncatedCone(**cone, tilt=25.0, az_amps=amps, az_pas=pas)
    from virgil.models import Cone, ExpTail

    sin_alpha = onp.sin(onp.radians(30.0))
    disk = MultipoleDisk(
        ExpTail(4.0 * sin_alpha, 8.0 * sin_alpha, cutoff=5.0),
        harmonics=harmonics,
        height=Cone(30.0, 5.0),
        inc=90.0 - 25.0,
        pa=pa - 90.0,
        blur_fwhm=None,
        n_r=48,
    )
    u, v = _uv(20, seed=3, qmax=0.1)
    envelope = onp.exp(
        -((onp.pi * 1.0 * _MAS2RAD_REF) ** 2)
        * (u**2 + v**2)
        / WAVEL**2
        / (4 * onp.log(2))
    )
    assert (
        onp.max(onp.abs(_cvis(modulated, u, v) - _cvis(disk, u, v) * envelope))
        < 1e-5
    )


# --- GaussianArc is a von Mises arc ---------------------------------------------


def _old_gaussian_arc_cvis(radius, width, length, pa, u, v, wavel):
    """The pre-von-Mises GaussianArc, by a dense sum round the circle
    weighted by a Gaussian in arc length wrapped to [-pi R, pi R)."""
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
    return curve @ weight / weight.sum() * blur


@pytest.mark.parametrize(
    "radius, length, tolerance",
    [
        (30.0, 5.0, 2e-3),
        (30.0, 10.0, 5e-3),
        (30.0, 20.0, 2e-2),
        (15.0, 10.0, 2e-2),
    ],
)
def test_gaussian_arc_keeps_the_old_shape_for_arcs_short_against_the_radius(
    radius, length, tolerance
):
    # The check of old against new shape: the von Mises weight is the
    # periodic Gaussian, so the two agree up to the O(length / radius) terms.
    width, pa, wavel = 0.6, 250.0, 1.6e-6
    rng = onp.random.default_rng(1)
    u, v = rng.uniform(-100.0, 100.0, (2, 60))
    with jax.enable_x64(True):
        arc = GaussianArc(radius, width, length, pa)
        new = onp.asarray(arc.model(u, v, wavel))
    old = _old_gaussian_arc_cvis(radius, width, length, pa, u, v, wavel)
    assert onp.max(onp.abs(new - old)) < tolerance


def test_gaussian_arc_differs_from_the_old_shape_for_arcs_that_wrap():
    # length = 40 on a ring of radius 5 (6 sigma exceeds the circle): the old
    # weight was a Gaussian cut at the antipode, the new one a smooth wrap.
    u, v = onp.array([10.0, -25.0, 40.0]), onp.array([5.0, 30.0, -12.0])
    with jax.enable_x64(True):
        new = onp.asarray(
            GaussianArc(5.0, 1.0, 40.0, 30.0).model(u, v, 1.6e-6)
        )
    old = _old_gaussian_arc_cvis(5.0, 1.0, 40.0, 30.0, u, v, 1.6e-6)
    assert 1e-3 < onp.max(onp.abs(new - old)) < 0.2


def test_gaussian_arc_is_a_thin_ring_multipole_disk_with_a_von_mises_weight():
    arc = GaussianArc(20.0, 2.0, 15.0, 70.0, dra=2.0, ddec=-1.0)
    kappa = (2.0 * onp.sqrt(2.0 * onp.log(2.0)) * 20.0 / 15.0) ** 2
    assert onp.isclose(float(arc.kappa()), kappa, rtol=1e-5)
    disk = MultipoleDisk(
        ThinRing(20.0),
        harmonics=VonMises(kappa, 70.0, degree=arc.degree),
        blur_fwhm=2.0,
        dra=2.0,
        ddec=-1.0,
    )
    u, v = _uv(25, seed=12, qmax=0.1)
    assert onp.max(onp.abs(_cvis(arc, u, v) - _cvis(disk, u, v))) < 1e-6
    # The rendered image and the visibilities describe the same arc.
    image = onp.asarray(arc.render(npix=257, fov_mas=80.0))
    xx, yy = (onp.asarray(a).ravel() for a in image_coordinates(257, 80.0))
    phase = onp.exp(
        -2j
        * onp.pi
        * _MAS2RAD_REF
        * (onp.outer(u, xx) + onp.outer(v, yy))
        / WAVEL
    )
    dft = phase @ image.ravel() / image.sum()
    assert onp.max(onp.abs(dft - _cvis(arc, u, v))) < 5e-3
    # A narrow arc needs a higher degree.
    narrow = GaussianArc(30.0, 1.0, 4.0, 10.0)
    assert narrow.degree > arc.degree
    assert narrow.set("radius", np.array(31.0)).degree == narrow.degree


def test_gaussian_arc_degree_is_controlled_and_builds_without_a_warning():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert GaussianArc(20.0, 2.0, 15.0, degree=12).degree == 12
        with pytest.raises(ValueError, match="degree"):
            GaussianArc(20.0, 2.0, 15.0, degree=0)
        # A uniform ring (a very long arc) needs very few harmonics.
        assert GaussianArc(5.0, 1.0, 1e3).degree <= 4


def test_linear_positivity_guarantee_is_the_sum_of_amplitudes_not_their_rms():
    # sum_m sqrt(c_m^2 + s_m^2) <= 1 keeps 1 + f >= 0, but an rms of the
    # coefficients <= 1 does not (M = 20 equal amplitudes dip to -0.11).
    M = 20
    a = 2.0 / (2.0 * M**0.5)  # c_m = 2 * (1 / (2 sqrt M)), so rms c = 1
    rms = Harmonics(c=(a,) * M)
    assert onp.isclose(onp.sqrt(sum(float(x) ** 2 for x in rms.c)), 1.0)
    assert float(rms.min_intensity(np.ones(1))) < -0.05
    safe = Harmonics(c=(1.0 / M,) * M)  # sum of amplitudes = 1
    assert float(safe.min_intensity(np.ones(1))) >= -1e-6


def test_narrow_arc_on_a_large_circle_uses_quadrature_without_warning():
    arc = GaussianArc(1e3, 3.0, 10.0, 30.0)
    assert arc.quadrature and bool(arc.is_physical())
    # Forcing the series (a degree too low for kappa) is flagged.
    forced = GaussianArc(150.0, 3.0, 6.0, 30.0, degree=20)
    assert not forced.quadrature and not bool(forced.is_physical())


def test_wide_arc_image_gradients_are_finite():
    import jax

    arc = GaussianArc(5.0, 1.0, 40.0, 30.0)
    xx, yy = onp.meshgrid(onp.linspace(-8, 8, 9), onp.linspace(-8, 8, 9))

    def total(a):
        return a._centred_image(xx, yy, 1.0).sum()

    grads = jax.grad(total)(arc)
    for leaf in (grads.radius, grads.length, grads.pa):
        assert onp.isfinite(onp.asarray(leaf))


def test_harmonics_w_must_be_concrete():
    import jax
    from virgil import Harmonics

    with pytest.raises(TypeError, match="re= and im="):
        jax.jit(lambda w: Harmonics(w=w).re)(onp.array([0.1 + 0.2j]))


def test_harmonics_keyword_only_after_c_and_s():
    assert Harmonics(0.3, 0.1).degree == 1
    with pytest.raises(TypeError):
        Harmonics(0.3, 0.1, 2)  # degree must be named
    with pytest.raises(TypeError):
        Harmonics((), (), None, "positive")


@pytest.mark.parametrize("inc", [0.0, 40.0])
@pytest.mark.parametrize("kind", ["multipole", "rim"])
def test_render_gradients_are_finite_with_a_pixel_on_the_origin(inc, kind):
    # An odd grid with no offset puts a pixel exactly at the disk's centre,
    # where hypot and arctan2 have NaN gradients unless made safe.
    harmonics = Harmonics(w=_random_latents(2, 7))
    if kind == "multipole":
        disk = MultipoleDisk(
            GaussianRing(6.0, 1.0), harmonics=harmonics, inc=inc, pa=20.0
        )
    else:
        disk = ModulatedGaussianRim(
            6.0, 1.0, inc, 20.0, az_amps=0.5, az_pas=0.0
        )

    def total(model):
        return np.sum(model.render(npix=25, fov_mas=24.0) ** 2)

    grads = eqx.filter_grad(total)(disk)
    for leaf in jax.tree_util.tree_leaves(grads):
        assert bool(np.all(np.isfinite(leaf)))


@pytest.mark.parametrize("mag", [1.0 - 1e-6, 1.0 - 1e-7, 1.0])
def test_step_down_and_gradient_are_finite_at_unit_modulus_boundary(mag):
    interior = np.array([0.3, 0.9], dtype=np.float32)
    k_int, clipped_int = H.step_down(H.step_up(interior))
    assert not clipped_int
    assert np.allclose(k_int, interior, atol=1e-5)

    k_in = np.array([0.3, mag], dtype=np.float32)
    h = H.step_up(k_in)
    k_out, clipped = H.step_down(h)
    assert clipped == (mag > 1.0 - H._CLIP)
    assert np.all(np.isfinite(k_out))
    assert np.isclose(np.abs(k_out[1]), 1.0 - H._CLIP, atol=1e-6)
    grad = jax.grad(lambda x: np.sum(np.real(H.step_down(H.step_up(x))[0])))(
        k_in
    )
    assert np.all(np.isfinite(grad))
