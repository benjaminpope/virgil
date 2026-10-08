import jax
import numpy as onp
import pytest
import jax.numpy as np

from virgil._geometry import (
    apply_elliptical_transf_coord,
    apply_elliptical_transf_spat_freq,
    check_az_prof_nonnegative,
    undo_elliptical_transf_coord,
    undo_elliptical_transf_spat_freq,
)


@pytest.fixture(autouse=True)
def _float64_by_default():
    """Run these tests in float64, without changing the rest of the suite.

    Setting ``jax_enable_x64`` globally at import time made the precision of
    every other test module depend on collection order.
    """
    with jax.enable_x64(True):
        yield


def test_undo_elliptical_transf_coord_is_identity_for_pa0_stretch1():
    x, y = np.array([3.0, -2.0]), np.array([-1.0, 4.0])
    xt, yt = undo_elliptical_transf_coord(x, y, pa=0.0, stretch=1.0)
    assert np.allclose(xt, x)
    assert np.allclose(yt, y)


def test_undo_elliptical_transf_coord_pa90_rotates_east_onto_major_axis():
    """A point due East (x=r, y=0) has PA=90deg from North. Undoing a
    PA=90deg rim rotation should place it on the transformed y-axis (i.e.
    aligned with the de-rotated major axis), verifying the North-to-East
    position angle convention.
    """
    r = 5.0
    xt, yt = undo_elliptical_transf_coord(r, 0.0, pa=90.0, stretch=1.0)
    assert xt == pytest.approx(0.0, abs=1e-10)
    assert yt == pytest.approx(r, abs=1e-10)


def test_undo_elliptical_transf_coord_nontrivial_pa_and_stretch():
    x, y, pa, stretch = 4.0, -6.0, 37.0, 0.6
    pa_rad = onp.deg2rad(pa)
    expected_xt = (x * onp.cos(pa_rad) - y * onp.sin(pa_rad)) / stretch
    expected_yt = x * onp.sin(pa_rad) + y * onp.cos(pa_rad)

    xt, yt = undo_elliptical_transf_coord(x, y, pa=pa, stretch=stretch)

    assert xt == pytest.approx(expected_xt)
    assert yt == pytest.approx(expected_yt)


@pytest.mark.parametrize("pa", [0.0, 37.0, 90.0, 143.0, -25.0])
@pytest.mark.parametrize("stretch", [1.0, 0.6, 0.15])
def test_apply_undoes_elliptical_transf_coord_round_trip(pa, stretch):
    x, y = np.array([3.5, -8.0, 0.0]), np.array([-2.5, 6.0, 4.2])
    xt, yt = undo_elliptical_transf_coord(x, y, pa=pa, stretch=stretch)
    x_rt, y_rt = apply_elliptical_transf_coord(xt, yt, pa=pa, stretch=stretch)
    assert np.allclose(x_rt, x, atol=1e-8)
    assert np.allclose(y_rt, y, atol=1e-8)


@pytest.mark.parametrize("pa", [0.0, 37.0, 90.0, 143.0, -25.0])
@pytest.mark.parametrize("stretch", [1.0, 0.6, 0.15])
def test_apply_undoes_elliptical_transf_spat_freq_round_trip(pa, stretch):
    u, v = np.array([12.0, -30.0, 5.0]), np.array([-8.0, 20.0, 0.0])
    ut, vt = undo_elliptical_transf_spat_freq(u, v, pa=pa, stretch=stretch)
    u_rt, v_rt = apply_elliptical_transf_spat_freq(
        ut, vt, pa=pa, stretch=stretch
    )
    assert np.allclose(u_rt, u, atol=1e-8)
    assert np.allclose(v_rt, v, atol=1e-8)


def test_check_az_prof_nonnegative_detects_valid_and_invalid_profiles():
    # 1 + 0.5*cos(theta) never drops below 0.5 -> valid.
    assert bool(
        check_az_prof_nonnegative(
            az_amps=np.array([0.5]), az_pas=np.array([0.0])
        )
    )
    # 1 + 1.5*cos(theta) dips to -0.5 at theta=180deg -> invalid.
    assert not bool(
        check_az_prof_nonnegative(
            az_amps=np.array([1.5]), az_pas=np.array([0.0])
        )
    )


def test_check_az_prof_nonnegative_accepts_no_modulation():
    assert check_az_prof_nonnegative(az_amps=np.array([]), az_pas=np.array([]))


def _brute_force_minimum(amps, pas, n=200_000):
    theta = onp.linspace(0.0, 2 * onp.pi, n, endpoint=False)
    orders = onp.arange(1, len(amps) + 1)
    arg = orders * theta[:, None] - onp.deg2rad(pas) * orders
    return 1.0 + onp.min(onp.sum(amps * onp.cos(arg), axis=1))


def test_check_az_prof_nonnegative_with_zero_top_order():
    # Review 1.1: a zero highest-order amplitude used to break the
    # companion-matrix roots, and 1 + 1.2 cos(θ - 45°) (minimum -0.2)
    # passed.
    amps, pas = np.array([1.2, 0.0]), np.array([45.0, 0.0])
    assert _brute_force_minimum(onp.asarray(amps), onp.asarray(pas)) < -0.19
    assert not bool(check_az_prof_nonnegative(amps, pas))
    assert bool(check_az_prof_nonnegative(np.array([0.8, 0.0]), pas))


def test_check_az_prof_nonnegative_matches_brute_force():
    rng = onp.random.default_rng(1)
    for _ in range(40):
        k = rng.integers(1, 5)
        amps = rng.uniform(-0.8, 0.8, k) * (rng.random(k) > 0.3)
        pas = rng.uniform(0.0, 360.0, k)
        true_min = _brute_force_minimum(amps, pas)
        if abs(true_min) < 1e-3:
            continue  # too close to call against the brute-force grid
        got = check_az_prof_nonnegative(np.asarray(amps), np.asarray(pas))
        assert bool(got) == (true_min > 0.0), (amps, pas, true_min)


def test_check_az_prof_nonnegative_finds_a_narrow_dip():
    # A just-negative minimum between grid points is still caught: the
    # Newton steps polish the grid's lowest value.
    amps, pas = np.array([0.0, 0.0, 1.0 + 1e-4]), np.array([0.0, 0.0, 1.0])
    assert not bool(check_az_prof_nonnegative(amps, pas))


def test_check_az_prof_nonnegative_under_jit():
    check = jax.jit(check_az_prof_nonnegative)
    assert not bool(check(np.array([1.2, 0.0]), np.array([45.0, 0.0])))
    assert bool(check(np.array([0.5, 0.3]), np.array([10.0, 70.0])))


def test_wrap_phase_maps_to_minus_pi_inclusive_pi_exclusive():
    from virgil._utils import wrap_phase

    pi = onp.pi
    values = np.array([0.0, pi, -pi, 3 * pi, 2 * pi, 0.5, -3.0 - 2 * pi])
    got = onp.asarray(wrap_phase(values))
    assert onp.allclose(got, [0.0, -pi, -pi, -pi, 0.0, 0.5, -3.0], atol=1e-5)
    assert onp.all((got >= -pi - 1e-6) & (got < pi))


def test_position_angle_is_north_through_east_in_zero_to_360():
    from virgil._geometry import position_angle, separation_pa

    dra = onp.array([0.0, 1.0, 0.0, -1.0, -1.0, 3.0])
    ddec = onp.array([1.0, 0.0, -1.0, 0.0, 1.0, 4.0])
    expected = [0.0, 90.0, 180.0, 270.0, 315.0, onp.degrees(onp.arctan(0.75))]
    got = position_angle(dra, ddec)
    assert isinstance(got, onp.ndarray)
    assert onp.allclose(got, expected)
    assert onp.all((got >= 0.0) & (got < 360.0))
    # The same in JAX, traced, and for a scalar, and the inverse of
    # (sep sin pa, sep cos pa).
    assert onp.allclose(
        position_angle(np.asarray(dra), ddec), expected, atol=1e-4
    )
    traced = jax.jit(lambda a, b: separation_pa(a, b))(dra, ddec)
    sep, pa = traced
    assert onp.allclose(sep, onp.hypot(dra, ddec), atol=1e-5)
    assert onp.allclose(pa, expected, atol=1e-4)
    assert onp.isclose(position_angle(-2.0, 0.0), 270.0)
    th = onp.radians(got)
    assert onp.allclose(onp.hypot(dra, ddec) * onp.sin(th), dra, atol=1e-12)
    assert onp.allclose(onp.hypot(dra, ddec) * onp.cos(th), ddec, atol=1e-12)


def test_fwhm_per_sigma_is_the_gaussian_ratio():
    from virgil._utils import FWHM_PER_SIGMA

    assert np.isclose(FWHM_PER_SIGMA, 2.0 * np.sqrt(2.0 * np.log(2.0)))
    assert np.isclose(FWHM_PER_SIGMA, 2.3548, atol=1e-4)


def test_rotate_keeps_numpy_inputs_in_numpy_precision():
    from virgil._geometry import rotate

    u, v = onp.array([1.0, 2.0]), onp.array([3.0, -1.0])
    x, y = rotate(u, v, 30.0)
    assert isinstance(x, onp.ndarray) and x.dtype == onp.float64
    c, s = onp.cos(onp.radians(30.0)), onp.sin(onp.radians(30.0))
    assert onp.allclose(x, c * u + s * v, rtol=1e-15)
    assert onp.allclose(y, -s * u + c * v, rtol=1e-15)


def test_legacy_helpers_are_deprecated():
    from virgil.legacy import oifits_implaneia as legacy

    with pytest.warns(FutureWarning, match="removed in virgil 0.6"):
        assert legacy.GetWavelength("JWST", "F480M")[0] == pytest.approx(
            4.817e-6
        )
    with pytest.warns(FutureWarning, match="virgil.oidata"):
        from virgil.legacy.oifits_implaneia import cp_indices
    from virgil.oidata import cp_indices as original

    assert cp_indices is original
