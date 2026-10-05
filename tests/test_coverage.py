import jax.numpy as np
import numpy as onp

from virgil.coverage import (
    NIRISS_AMI_HOLES,
    ami_grid_record,
    mask_transfer,
    nrm_oidata,
)
from virgil.models import BinaryModelCartesian, PointSource
from virgil.oidata import OIData


def test_mask_transfer_has_splodges_at_the_baselines():
    assert np.isclose(mask_transfer(0.0, 0.0), 1.0)
    bu, bv = NIRISS_AMI_HOLES[1] - NIRISS_AMI_HOLES[0]
    # One pair of holes gives 1/7 of the zero-baseline transfer.
    assert np.isclose(mask_transfer(bu, bv), 1.0 / 7.0, rtol=1e-6)
    assert np.isclose(mask_transfer(-bu, -bv), mask_transfer(bu, bv))
    assert mask_transfer(20.0, 0.0) == 0.0


def test_grid_record_modes_are_orthonormal_and_ordered():
    record = ami_grid_record(pitch_m=0.5)
    modes = onp.concatenate(
        [
            record["disco_logamp_model_operator"],
            record["disco_phase_model_operator"],
        ],
        axis=1,
    )
    assert onp.allclose(modes @ modes.T, onp.eye(len(modes)), atol=1e-8)
    assert onp.all(onp.diff(record["disco_sigma"]) >= 0.0)
    # Only cells inside the splodges are kept.
    grid = onp.hypot(record["u"], record["v"])
    assert grid.max() < 6.1 and grid.size < 500


def test_grid_record_is_blind_to_position_and_flux():
    data = OIData(ami_grid_record(pitch_m=0.5, rotation_deg=-6.9))
    assert data.uv_grid is not None
    shifted = PointSource(dra=40.0, ddec=-25.0)
    assert np.max(np.abs(data.model(shifted))) < 1e-5
    binary = BinaryModelCartesian(40.0, -25.0, 0.1)
    assert np.max(np.abs(data.model(binary))) > 1e-3
    # A constant log-amplitude (a flux scaling) gives no signal either.
    record = ami_grid_record(pitch_m=0.5)
    logamp = record["disco_logamp_model_operator"]
    assert onp.allclose(logamp @ onp.ones(logamp.shape[1]), 0.0, atol=1e-8)


def test_keeping_less_precision_keeps_fewer_modes():
    many = ami_grid_record(pitch_m=0.5)["disco_sigma"].size
    few = ami_grid_record(pitch_m=0.5, keep=0.9)["disco_sigma"].size
    assert few < many


def test_nrm_oidata_has_v2_and_closure_phases_at_the_splodge_centres():
    data = nrm_oidata(rotation_deg=-6.9)
    assert data.vis.size == 21 and data.phi.size == 35
    # A point source has no closure phase anywhere, and V² = 1.
    model = data.model(PointSource(dra=30.0, ddec=10.0))
    assert np.allclose(model[:21], 1.0) and np.allclose(
        model[21:], 0.0, atol=1e-5
    )
    # Rotating the mask on the sky rotates the baselines.
    unrotated = nrm_oidata()
    assert np.allclose(
        np.hypot(data.u, data.v), np.hypot(unrotated.u, unrotated.v), rtol=1e-6
    )
    assert np.isclose(np.degrees(data.d_phi[0]), 0.5)


def test_rotation_follows_the_position_angle_convention():
    from virgil._geometry import rotate

    for angle in (-6.9, 20.0):
        grid = OIData(ami_grid_record(pitch_m=0.5, rotation_deg=angle)).uv_grid
        assert np.isclose(grid.rotation_deg, angle, atol=1e-6)
    data, plain = nrm_oidata(rotation_deg=30.0), nrm_oidata()
    u, v = rotate(plain.u, plain.v, 30.0)
    assert np.allclose(data.u, u, atol=1e-5) and np.allclose(
        data.v, v, atol=1e-5
    )


def test_vlti_nights_repeat_the_snapshots_with_times():
    from virgil.coverage import vlti_oidata

    hours, nights = (-1.0, 0.0, 1.0), (60000.0, 60100.0)
    one = vlti_oidata(hour_angles_h=hours, wavelengths_m=[2.2e-6])
    timed = vlti_oidata(
        hour_angles_h=hours, wavelengths_m=[2.2e-6], nights_mjd=nights
    )
    assert one.mjd is None
    assert timed.u.size == 2 * one.u.size
    assert onp.allclose(timed.u[: one.u.size], one.u)
    assert onp.allclose(
        onp.unique(timed.mjd), [n + h / 24 for n in nights for h in hours]
    )
    assert onp.unique(onp.asarray(timed.frame)).size == 6
    assert len(timed.split_by_epoch()) == 2
