"""Image-recovery metrics: identity, shifts, scale invariance, rotation, beam."""

import jax.numpy as np
import numpy as onp
import pytest

from virgil import metrics
from virgil.imaging import Beam
from virgil.scenes import gaussian_blob

NPIX = 32
SCALE = 1.0


def _scene():
    # Asymmetric, so rotation and shifts are distinguishable.
    return gaussian_blob(
        NPIX, SCALE, 2.0, dra=3.0, ddec=-2.0
    ) + 0.4 * gaussian_blob(NPIX, SCALE, 1.5, dra=-4.0, ddec=3.0)


def test_identity_scores_perfectly():
    truth = _scene()
    out = metrics.score(truth, truth, pixel_scale_mas=SCALE, max_shift_mas=3.0)
    assert out["ncc"] == pytest.approx(1.0, abs=1e-5)
    assert out["l1"] == pytest.approx(1.0, abs=1e-5)
    assert out["lawson"] == pytest.approx(0.0, abs=1e-6)
    assert out["rms"] == pytest.approx(0.0, abs=1e-8)
    assert not out["rotated"]
    assert out["shift_dra_mas"] == pytest.approx(0.0, abs=1e-5)


def test_known_shift_is_recovered():
    truth = _scene()
    # Move the source 3 mas East and 2 mas South; align undoes it.
    moved = metrics._shift(truth, 2, -3)
    shifted, (dra, ddec) = metrics.align(moved, truth, 5.0, SCALE)
    assert dra == pytest.approx(-3.0, abs=0.1)
    assert ddec == pytest.approx(2.0, abs=0.1)
    assert metrics.ncc(shifted, truth) > 0.999


def test_subpixel_shift_is_recovered():
    truth = _scene()
    moved = metrics._shift(truth, 0.4, 0.0)
    _, (dra, ddec) = metrics.align(moved, truth, 3.0, SCALE)
    assert ddec == pytest.approx(0.4, abs=0.15)
    assert dra == pytest.approx(0.0, abs=0.15)


def test_l1_and_ncc_are_flux_scale_invariant():
    truth = _scene()
    image = truth + 0.3 * gaussian_blob(NPIX, SCALE, 3.0, dra=1.0)
    ref = (
        float(metrics.l1_score(image, truth)),
        float(metrics.ncc(image, truth)),
    )
    for factor in (0.01, 7.0):
        assert metrics.l1_score(factor * image, truth) == pytest.approx(
            ref[0], abs=1e-5
        )
        assert metrics.l1_score(image, factor * truth) == pytest.approx(
            ref[0], abs=1e-5
        )
        assert metrics.ncc(factor * image, truth) == pytest.approx(
            ref[1], abs=1e-5
        )
    assert 0.0 < ref[0] < 1.0


def test_l1_matches_brute_force_minimum():
    rng = onp.random.default_rng(0)
    e = rng.uniform(0.0, 1.0, (6, 6))
    r = rng.uniform(0.0, 1.0, (6, 6))
    scales = onp.linspace(0.0, 5.0, 20001)
    cost = onp.abs(scales[:, None] * e.ravel() - r.ravel()).sum(axis=1)
    expected = 1.0 - cost.min() / r.sum()
    assert float(metrics.l1_score(e, r)) == pytest.approx(expected, abs=1e-3)


def test_rotated_image_scores_perfectly_under_v2_only():
    truth = _scene()
    rotated = truth[::-1, ::-1]
    plain = metrics.score(rotated, truth, pixel_scale_mas=SCALE)
    out = metrics.score(rotated, truth, pixel_scale_mas=SCALE, v2_only=True)
    assert plain["ncc"] < 0.9
    assert out["rotated"]
    assert out["ncc"] == pytest.approx(1.0, abs=1e-5)
    assert out["l1"] == pytest.approx(1.0, abs=1e-5)


def test_convolution_lowers_the_rms():
    truth = _scene()
    image = (
        truth
        + 0.2
        * np.asarray(onp.random.default_rng(1).uniform(size=truth.shape))
        * truth.max()
    )
    beam = Beam(major_mas=6.0, minor_mas=4.0, pa_deg=30.0)
    bare = metrics.rms_convolved(image, truth, SCALE)
    smooth = metrics.rms_convolved(image, truth, SCALE, beam)
    assert smooth < bare
    out = metrics.score(image, truth, pixel_scale_mas=SCALE, beam=beam)
    assert out["rms_convolved"] < out["rms"]


def test_resample_conserves_flux_and_round_trips():
    truth = _scene()
    fine = metrics.resample(truth, SCALE, 2 * NPIX, 0.5)
    assert float(fine.sum()) == pytest.approx(float(truth.sum()), rel=1e-5)
    back = metrics.resample(fine, 0.5, NPIX, SCALE)
    assert onp.allclose(back, truth, atol=1e-6)
    coarse = metrics.resample(truth, SCALE, NPIX // 2, 2.0)
    assert float(coarse.sum()) == pytest.approx(float(truth.sum()), rel=1e-5)


def test_score_resamples_onto_the_truth_grid():
    truth = _scene()
    fine = metrics.resample(truth, SCALE, 2 * NPIX, 0.5)
    out = metrics.score(
        fine, truth, pixel_scale_mas=0.5, truth_pixel_scale_mas=SCALE
    )
    assert out["ncc"] == pytest.approx(1.0, abs=1e-5)


def test_rms_convolved_arrays_without_scale_or_beam():
    truth = np.asarray(_scene())
    other = np.roll(truth, 2, axis=1)
    out = metrics.rms_convolved(other, truth)
    assert float(out) > 0
    assert float(metrics.rms_convolved(truth, truth)) == pytest.approx(0.0)
    with_scale = metrics.rms_convolved(other, truth, SCALE)
    assert float(out) == pytest.approx(float(with_scale))


def test_rms_convolved_beam_needs_pixel_scale_for_arrays():
    truth = np.asarray(_scene())
    beam = Beam(1.0, 1.0, 0.0)
    with pytest.raises(ValueError, match="pixel_scale_mas"):
        metrics.rms_convolved(truth, truth, beam=beam)


def test_align_matches_a_shift_by_shift_loop():
    # The jitted sweep over the shift table gives the NCC table the old
    # call-per-shift loop did, so the same shift is chosen and applied.
    truth = _scene()
    moved = metrics._shift(truth, 2.3, -1.4)
    n = 4
    steps = range(-n, n + 1)
    loop = onp.array(
        [
            [
                float(metrics.ncc(metrics._shift(moved, i, j), truth))
                for j in steps
            ]
            for i in steps
        ]
    )
    pairs = onp.stack(
        onp.meshgrid(*[onp.arange(-n, n + 1)] * 2, indexing="ij"), axis=-1
    ).reshape(-1, 2)
    table = onp.asarray(
        metrics._ncc_table(moved, truth, np.asarray(pairs, dtype=float))
    ).reshape(loop.shape)
    onp.testing.assert_allclose(table, loop, atol=1e-5)
    _, (dra, ddec) = metrics.align(moved, truth, float(n), SCALE)
    assert dra == pytest.approx(-1.4, abs=0.15)
    assert ddec == pytest.approx(2.3, abs=0.15)


# --- ring_features ---------------------------------------------------------


def _ring(
    n=96,
    ps=1.0,
    radius=20.0,
    sigma=3.0,
    dra=0.0,
    ddec=0.0,
    amp=0.0,
    eta_deg=0.0,
    disk=0.0,
):
    """Gaussian ring, ``1 + amp cos(PA - eta)`` modulated; row 0 North, col 0 East."""
    rows, cols = onp.mgrid[:n, :n]
    north = (n - 1) / 2 - rows * 1.0
    east = (n - 1) / 2 - cols * 1.0
    north, east = north * ps - ddec, east * ps - dra
    r = onp.hypot(north, east)
    pa = onp.arctan2(east, north)
    img = onp.exp(-0.5 * ((r - radius) / sigma) ** 2)
    img = img * (1 + amp * onp.cos(pa - onp.radians(eta_deg)))
    return img + disk * (r < radius)


def test_ring_diameter_and_width():
    f = metrics.ring_features(_ring(), 1.0)
    assert f.diameter == pytest.approx(40.0, abs=1.0)
    assert f.width == pytest.approx(2.355 * 3.0, abs=1.0)
    assert f.asymmetry < 0.02
    assert f.central_fraction < 0.05


def test_ring_centre_offset():
    # Off the coarse search grid, so the refinement has to find it.
    f = metrics.ring_features(_ring(dra=3.3, ddec=-2.6), 1.0)
    assert f.centre[0] == pytest.approx(3.3, abs=0.3)
    assert f.centre[1] == pytest.approx(-2.6, abs=0.3)
    assert f.diameter == pytest.approx(40.0, abs=1.0)


def test_ring_centre_large_pixels():
    # Large pixel scale: the optimizer's step must scale with the grid.
    f = metrics.ring_features(
        _ring(n=96, ps=5.0, radius=200.0, sigma=30.0, dra=12.0, ddec=-10.0),
        5.0,
    )
    assert f.centre[0] == pytest.approx(12.0, abs=1.0)
    assert f.centre[1] == pytest.approx(-10.0, abs=1.0)


@pytest.mark.parametrize("eta", [0.0, 90.0, 200.0])
def test_ring_pa_and_asymmetry(eta):
    # PA is East of North: eta = 90 puts the bright side to the East (left).
    f = metrics.ring_features(_ring(amp=0.4, eta_deg=eta), 1.0)
    d = (f.pa - eta + 180.0) % 360.0 - 180.0
    assert abs(d) < 3.0
    assert f.asymmetry == pytest.approx(0.2, abs=0.03)


def test_ring_central_fraction_contrast():
    ring = metrics.ring_features(_ring(), 1.0)
    # Flat disc of 0.8 well inside the ring (edge at radius - 3 sigma).
    img = _ring()
    n = img.shape[0]
    rr = onp.hypot(*(onp.mgrid[:n, :n] - (n - 1) / 2))
    filled = metrics.ring_features(img + 0.8 * (rr < 20.0 - 9.0), 1.0)
    assert filled.central_fraction == pytest.approx(0.8, rel=0.05)
    assert ring.central_fraction < 0.01


def test_ring_dispersions_vanish_for_symmetric_ring():
    f = metrics.ring_features(_ring(), 1.0)
    assert f.diameter_std < 0.3
    assert f.width_std < 0.3
    assert f.asymmetry_std < 0.01
    assert f.central_fraction_std < 0.01


def test_ring_needs_scale_for_array():
    with pytest.raises(ValueError):
        metrics.ring_features(_ring())


def test_ring_ratio_keywords():
    img = _ring(sigma=4.0)
    base = metrics.ring_features(img, 1.0)
    f = metrics.ring_features(img, 1.0, central_ratio=0.2, floor_ratio=2.0)
    # A floor nearer the ring is higher, so the width is smaller.
    assert f.width < base.width
    assert f.diameter == pytest.approx(base.diameter, abs=0.1)
