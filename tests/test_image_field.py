"""The image field and the resolution come from the unflagged data.

A field of ``k`` FWHMs of the source, between an inner working angle and
the interferometric field of view, replaces the old floor of 500 mas (which
always gave λ/B_min); flagged samples keep their u, v in the data but set
no beam, pixel or field.
"""

import jax
import numpy as onp
import pytest

from virgil._utils import FWHM_PER_SIGMA, mas2rad
from virgil.coverage import ami_grid_record
from virgil.ensemble import EnsembleSpec, draw_groups
from virgil.imaging import (
    beam,
    envelope_fwhm,
    field_of_view,
    image_field,
    nyquist_pixel_scale,
    second_moment_fwhm,
    starting_image,
)
from virgil.models import GaussianDisk
from virgil.oidata import OIData

WAVEL = 2.2e-6  # metres
B_MIN, B_MAX = 11.0, 130.0  # metres
LAMBDA_OVER_B_MAX = WAVEL / B_MAX / mas2rad  # mas, 3.49
LAMBDA_OVER_B_MIN = WAVEL / B_MIN / mas2rad  # mas, 41.2


def _baselines(b_min=B_MIN, b_max=B_MAX, n=48, seed=0):
    rng = onp.random.default_rng(seed)
    length = onp.geomspace(b_min, b_max, n)
    angle = rng.uniform(0.0, onp.pi, n)
    return length * onp.cos(angle), length * onp.sin(angle)


def _gaussian(fwhm_mas, u, v, flag=None):
    """V² of a circular Gaussian of this FWHM, with an optional V² flag."""
    model = GaussianDisk(fwhm_mas / FWHM_PER_SIGMA)
    v2 = onp.abs(onp.asarray(model.model(u, v, WAVEL))) ** 2
    data = {
        "u": u,
        "v": v,
        "wavel": WAVEL,
        "vis": v2,
        "d_vis": onp.full(u.size, 0.01),
    }
    if flag is not None:
        data["vis_flag"] = flag
    return OIData(data)


def _with_flagged_long_baselines(fwhm_mas=4.0):
    """Unflagged baselines to 60 m, and 20 flagged ones at 130 m, as in the
    provider-flagged 2024 GRAVITY data (SNR < 1 on the long baselines)."""
    u, v = _baselines(B_MIN, 60.0, 40)
    ul, vl = _baselines(130.0, 130.0, 20, seed=1)
    flag = onp.r_[onp.zeros(40, bool), onp.ones(20, bool)]
    both = _gaussian(fwhm_mas, onp.r_[u, ul], onp.r_[v, vl], flag)
    return (
        both,
        _gaussian(fwhm_mas, u, v),
        _gaussian(fwhm_mas, onp.r_[u, ul], onp.r_[v, vl]),
    )


def test_unflagged_marks_the_samples_an_observable_uses():
    # Four baselines; V² of the last three are flagged. A closure phase
    # uses baselines 0, 1 and 2, so only baseline 3 measures nothing.
    data = OIData(
        {
            "u": onp.array([10.0, 20.0, 30.0, 40.0]),
            "v": onp.zeros(4),
            "wavel": WAVEL,
            "vis": onp.ones(4),
            "d_vis": onp.full(4, 0.01),
            "vis_flag": onp.array([False, True, True, True]),
            "phi": onp.zeros(1),
            "d_phi": onp.full(1, 0.1),
            "i_cps1": onp.array([0]),
            "i_cps2": onp.array([1]),
            "i_cps3": onp.array([2]),
        }
    )
    assert onp.array_equal(data.unflagged, [True, True, True, False])
    # Without the closure phase the flags are the V² flags alone.
    visibilities = data.select(observables="vis")
    assert onp.array_equal(visibilities.unflagged, [True, False, False, False])
    # No flags: every sample.
    assert _gaussian(3.0, *_baselines()).unflagged.all()
    # Projected (DISCO) data cannot be flagged by sample: all of them.
    disco = OIData(ami_grid_record(pitch_m=0.5))
    assert disco.unflagged.all()


def test_flagged_long_baselines_leave_the_beam_and_the_nyquist_pixel_alone():
    flagged, base, unflagged = _with_flagged_long_baselines()
    a, b = beam(flagged), beam(base)
    assert a.major_mas == pytest.approx(b.major_mas, rel=1e-5)
    assert a.minor_mas == pytest.approx(b.minor_mas, rel=1e-5)
    assert nyquist_pixel_scale(flagged) == pytest.approx(
        nyquist_pixel_scale(base), rel=1e-6
    )
    assert field_of_view(flagged) == pytest.approx(field_of_view(base))
    # The same rows unflagged do count: the beam and the pixel are finer.
    assert beam(unflagged).major_mas < 0.8 * a.major_mas
    assert nyquist_pixel_scale(unflagged) < 0.6 * nyquist_pixel_scale(flagged)


def test_the_pipeline_geometry_ignores_flagged_baselines():
    from virgil.pipeline._common import _geometry

    flagged, base, _ = _with_flagged_long_baselines()
    assert _geometry(flagged) == _geometry(base)


def test_second_moment_fwhm_recovers_a_gaussian():
    u, v = _baselines()
    for fwhm in (2.9, 6.0):
        assert second_moment_fwhm(_gaussian(fwhm, u, v)) == pytest.approx(
            fwhm, rel=0.01
        )
    # A point source: V² does not fall.
    assert second_moment_fwhm(_gaussian(1e-3, u, v)) == pytest.approx(
        0.0, abs=0.05
    )
    # A source resolved on every baseline has no V² of 0.3: no estimate.
    u, v = _baselines(40.0, 130.0)
    assert second_moment_fwhm(_gaussian(20.0, u, v)) is None
    # Nor do DISCO data, which carry no visibility amplitudes of their own.
    assert second_moment_fwhm(OIData(ami_grid_record(pitch_m=0.5))) is None


def test_second_moment_fwhm_ignores_flagged_samples():
    flagged, base, _ = _with_flagged_long_baselines(fwhm_mas=4.0)
    assert second_moment_fwhm(flagged) == pytest.approx(4.0, rel=0.02)
    assert second_moment_fwhm(flagged) == pytest.approx(
        second_moment_fwhm(base), rel=1e-4
    )


def test_the_field_of_a_gaussian_is_k_fwhm():
    # A 2.9 mas source, B_max of 130 m at 2.2 µm: the field is 6 FWHM =
    # 17.4 mas, not λ/B_min = 41 mas, and not 500 mas.
    data = _gaussian(2.9, *_baselines())
    assert image_field(data) == pytest.approx(6.0 * 2.9, rel=0.01)
    assert image_field(data) < 0.5 * LAMBDA_OVER_B_MIN
    for k in (3.0, 4.0, 8.0):
        assert image_field(data, field_fwhms=k) == pytest.approx(
            k * 2.9, rel=0.01
        )
    # The FWHM can be given, as starting_image does with its envelope fit.
    assert image_field(data, fwhm_mas=5.0) == pytest.approx(30.0)


def test_the_inner_working_angle_is_a_floor():
    # An unresolved source: the field is 4 resolution elements λ/(2 B_max)
    # (7 mas), however small the source.
    data = _gaussian(0.1, *_baselines())
    assert image_field(data) == pytest.approx(4.0 * 0.5 * LAMBDA_OVER_B_MAX)
    assert image_field(data) == pytest.approx(4.0 * nyquist_pixel_scale(data))
    assert image_field(data, inner_elements=8.0) == pytest.approx(
        8.0 * 0.5 * LAMBDA_OVER_B_MAX
    )
    # A source whose six FWHMs are below the floor gets the floor too.
    small = _gaussian(0.5, *_baselines())
    assert image_field(small) == pytest.approx(4.0 * 0.5 * LAMBDA_OVER_B_MAX)


def test_the_interferometric_field_is_a_ceiling():
    # 12 mas: six FWHMs (72 mas) is beyond λ/B_min = 41 mas.
    data = _gaussian(12.0, *_baselines())
    assert second_moment_fwhm(data) == pytest.approx(12.0, rel=0.01)
    assert image_field(data) == pytest.approx(LAMBDA_OVER_B_MIN, rel=1e-4)
    # An aperture field and a cap lower it further.
    assert image_field(data, aperture_mas=30.0) == pytest.approx(30.0)
    assert image_field(data, largest_mas=25.0) == pytest.approx(25.0)
    # The ceiling wins if it falls below the floor (a narrow range of
    # baselines): here λ/B_min is 3.5 mas, the floor 7 mas.
    narrow = _gaussian(2.0, *_baselines(120.0, 130.0, 12))
    assert image_field(narrow) == pytest.approx(field_of_view(narrow))
    assert field_of_view(narrow) < 4.0 * nyquist_pixel_scale(narrow)


def test_the_bandwidth_smearing_field_is_a_ceiling():
    data = _gaussian(12.0, *_baselines())
    # Channels of resolving power 5: R λ/B_max = 17.5 mas.
    smeared = data.with_bandwidth(WAVEL / 5.0)
    assert field_of_view(data) == pytest.approx(LAMBDA_OVER_B_MIN, rel=1e-4)
    assert field_of_view(smeared) == pytest.approx(
        5.0 * LAMBDA_OVER_B_MAX, rel=1e-4
    )
    assert image_field(smeared) == pytest.approx(
        5.0 * LAMBDA_OVER_B_MAX, rel=1e-4
    )
    # Narrow channels (R = 500) smear nothing within λ/B_min.
    sharp = data.with_bandwidth(WAVEL / 500.0)
    assert field_of_view(sharp) == pytest.approx(LAMBDA_OVER_B_MIN, rel=1e-4)


def test_without_visibility_amplitudes_the_field_is_the_ceiling():
    disco = OIData(ami_grid_record(pitch_m=0.5))
    assert image_field(disco) == field_of_view(disco) == 500.0


def test_starting_image_has_a_field_of_k_fwhm_of_its_envelope():
    u, v = _baselines()
    fwhm = 4.7
    data = _gaussian(fwhm, u, v)
    start = starting_image(data, star=False)
    field = start.log_brightness.shape[0] * start.pixel_scale_mas
    # Six FWHMs (28 mas), not λ/B_min (41 mas), within a pixel and the fit.
    assert field == pytest.approx(6.0 * fwhm, rel=0.08)
    assert start.pixel_scale_mas <= nyquist_pixel_scale(data) / 4 + 1e-9
    smaller = starting_image(data, star=False, field_fwhms=4.0)
    assert smaller.log_brightness.shape[0] * smaller.pixel_scale_mas == (
        pytest.approx(4.0 * fwhm, rel=0.08)
    )


def _size(draw):
    return draw.npix * draw.pixel_scale_mas


def test_the_ensemble_field_follows_the_data():
    u, v = _baselines()
    key = jax.random.PRNGKey(0)
    spec = EnsembleSpec(field_factors=(1.0,))
    fields = {}
    for fwhm in (3.0, 6.0):
        data = _gaussian(fwhm, u, v)
        fields[fwhm] = image_field(data)
        for d in draw_groups(data, 6, key, spec):
            # The envelope fit sizes it, within the fit and a pixel.
            assert _size(d) == pytest.approx(
                6.0 * fwhm, rel=0.08, abs=d.pixel_scale_mas
            )
    assert fields[6.0] == pytest.approx(2.0 * fields[3.0], rel=0.02)


def test_the_ensemble_sweep_is_a_range_around_the_data_field():
    data = _gaussian(3.0, *_baselines())
    field = image_field(data, fwhm_mas=envelope_fwhm(data))
    sizes = [
        _size(d) / field
        for d in draw_groups(data, 24, jax.random.PRNGKey(1), EnsembleSpec())
    ]
    pixel = 1.0  # tolerance of one pixel (< 1 mas) relative to 18 mas
    assert min(sizes) >= 0.75 - pixel / field
    assert max(sizes) <= 1.5 + pixel / field
    assert {round(s, 1) for s in sizes} == {0.8, 1.0, 1.5}


def test_the_ensemble_field_never_exceeds_the_interferometric_field():
    # Six FWHMs (72 mas) is beyond λ/B_min (41 mas). The sweep used to take
    # 1, 2 and 4 times λ/B_min.
    data = _gaussian(12.0, *_baselines())
    ceiling = field_of_view(data)
    spec = EnsembleSpec()
    for d in draw_groups(data, 12, jax.random.PRNGKey(2), spec):
        assert _size(d) <= ceiling + d.pixel_scale_mas
    # An aperture lowers the ceiling.
    spec = EnsembleSpec(aperture_mas=20.0)
    for d in draw_groups(data, 12, jax.random.PRNGKey(2), spec):
        assert _size(d) <= 20.0 + d.pixel_scale_mas


def test_a_given_field_replaces_the_data_field_in_the_ensemble():
    data = _gaussian(3.0, *_baselines())
    spec = EnsembleSpec(field_factors=(1.0,), field_mas=25.0)
    for d in draw_groups(data, 4, jax.random.PRNGKey(3), spec):
        assert _size(d) == pytest.approx(25.0, abs=d.pixel_scale_mas)
    # Still within the interferometric field.
    spec = EnsembleSpec(field_factors=(1.0,), field_mas=1000.0)
    for d in draw_groups(data, 4, jax.random.PRNGKey(3), spec):
        assert _size(d) <= field_of_view(data) + d.pixel_scale_mas


def _star_and_envelope(star_fraction, fwhm_mas, u, v):
    """V² of an unresolved star plus a Gaussian envelope."""
    model = GaussianDisk(fwhm_mas / FWHM_PER_SIGMA)
    env = onp.abs(onp.asarray(model.model(u, v, WAVEL)))
    amp = star_fraction + (1.0 - star_fraction) * env
    return OIData(
        {
            "u": u,
            "v": v,
            "wavel": WAVEL,
            "vis": amp**2,
            "d_vis": onp.full(u.size, 0.01),
        }
    )


def test_a_bright_star_does_not_shrink_the_ensemble_field():
    # With 80% of the flux in the star, V² stays above 0.3 everywhere and
    # the second moment alone is set by the long-baseline plateau.
    u, v = _baselines()
    data = _star_and_envelope(0.8, 10.0, u, v)
    assert second_moment_fwhm(data) < 3.0
    spec = EnsembleSpec(field_factors=(1.0,))
    sizes = [
        _size(d) for d in draw_groups(data, 4, jax.random.PRNGKey(4), spec)
    ]
    # The envelope fit sizes the field, not the moment: well beyond the
    # 7 mas floor that the moment gives.
    assert min(sizes) > 2.0 * image_field(data, fwhm_mas=2.0)
    # The envelope fit sees through the star.
    assert envelope_fwhm(data, star=True) > 3.0 * second_moment_fwhm(data)


def test_starting_image_takes_the_larger_of_envelope_and_moment(monkeypatch):
    import virgil.imaging as imaging

    u, v = _baselines()
    data = _gaussian(4.7, u, v)
    monkeypatch.setattr(imaging, "second_moment_fwhm", lambda d: 6.0)
    start = starting_image(data, star=False)
    field = start.log_brightness.shape[0] * start.pixel_scale_mas
    assert field == pytest.approx(6.0 * 6.0, rel=0.08)


def _grid_width(start):
    return start.log_brightness.shape[0] * start.pixel_scale_mas


def test_field_mas_pads_the_start_image_beyond_the_interferometric_field():
    u, v = _baselines()
    data = _gaussian(4.7, u, v)
    natural = starting_image(data, star=False)
    for kind in ("moments",):  # the dirty start: test_imaging.py
        wide = starting_image(
            data, star=False, start=kind, pad_to_mas=3.0 * LAMBDA_OVER_B_MIN
        )
        assert wide.pixel_scale_mas == natural.pixel_scale_mas
        assert _grid_width(wide) >= 3.0 * LAMBDA_OVER_B_MIN
        assert (
            _grid_width(wide)
            < 3.0 * LAMBDA_OVER_B_MIN + 2 * wide.pixel_scale_mas
        )
        # The extra pixels keep the centre pixel of the natural grid.
        assert (
            wide.log_brightness.shape[0] - natural.log_brightness.shape[0]
        ) % 2 == 0
    # A field smaller than the data-sized one changes nothing.
    small = starting_image(data, star=False, pad_to_mas=1.0)
    assert small.log_brightness.shape == natural.log_brightness.shape


def test_starting_image_defaults_do_not_warn(recwarn):
    u, v = _baselines()
    data = _gaussian(4.7, u, v)
    starting_image(data, star=False)
    starting_image(data, star=False, largest_mas=10.0)
    # A generous cap and a source that wants less than λ/B_min: no warning.
    starting_image(data, star=False, largest_mas=1000.0)
    assert not [w for w in recwarn if "largest_mas" in str(w.message)]


def test_largest_mas_above_the_interferometric_field_warns_and_is_capped():
    u, v = _baselines()
    data = _gaussian(30.0, u, v)  # wants more than λ/B_min
    with pytest.warns(UserWarning, match="pad_to_mas"):
        capped = starting_image(data, star=False, largest_mas=1000.0)
    assert _grid_width(capped) <= LAMBDA_OVER_B_MIN + capped.pixel_scale_mas


def test_padding_a_dirty_start_keeps_the_centre():
    from virgil.imaging import _pad_brightness

    image = onp.arange(25.0).reshape(5, 5)
    padded = onp.asarray(_pad_brightness(image, 9))
    assert padded.shape == (9, 9)
    assert padded[2:7, 2:7].tolist() == image.tolist()
    assert padded.sum() == image.sum()


def test_unflagged_of_flux_only_data_counts_every_sample():
    # No V² or phase block uses a sample, so nothing is flagged by a block:
    # every sample counts (never an all-False mask), and `beam` is finite.
    data = OIData(
        {
            "u": onp.array([10.0, 20.0, 30.0]),
            "v": onp.array([0.0, 5.0, 0.0]),
            "wavel": WAVEL,
            "vis": onp.ones(3),
            "d_vis": onp.full(3, 0.01),
            "vis_flag": onp.ones(3, dtype=bool),
            "flux": {
                "value": onp.ones(2),
                "error": onp.full(2, 0.1),
                "wavel": onp.full(2, 2e-6),
            },
        }
    )
    assert data.unflagged.all()
    assert onp.all(onp.isfinite(beam(data).major_mas))
