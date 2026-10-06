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
    ref = metrics.l1_score(image, truth), metrics.ncc(image, truth)
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
