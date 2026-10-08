import jax.numpy as np
import numpy as onp
import pytest

from virgil.amigo import load_oi_data
from virgil.likelihood import model_loglike
from virgil.models import BinaryModelCartesian
from virgil.oidata import OIData
from tests._shared import CALIBRATED_VISIBILITY


PRODUCT = CALIBRATED_VISIBILITY


def _small_mixed_record():
    sigma = onp.array([0.5, 1.0])
    return {
        "u": onp.array([1.0, -2.0, 3.0]),
        "v": onp.array([0.5, 1.5, -0.5]),
        "wavelength_m": onp.array(4.3e-6),
        "disco_coefficients": onp.array([0.1, -0.2]),
        "disco_sigma": sigma,
        "disco_covariance": onp.diag(sigma**2),
        "disco_logamp_model_operator": onp.array(
            [[1.0, 0.0, -1.0], [0.5, 0.25, 0.0]]
        ),
        "disco_phase_model_operator": onp.array(
            [[0.0, 1.0, 0.5], [-0.5, 0.0, 1.0]]
        ),
    }


def test_load_oi_data_returns_mixed_disco_filters():
    observations = load_oi_data(PRODUCT)

    assert tuple(observations) == ("F380M", "F430M", "F480M")
    assert all(
        item.observable_kind == "mixed_log_complex"
        for item in observations.values()
    )
    assert observations["F430M"].vis.shape == (210,)
    assert observations["F430M"].vis_mat.shape == (210, 47)


def test_synthetic_product_records_public_injected_truth():
    product = onp.load(PRODUCT, allow_pickle=True).item()
    truth = product["F430M"]["synthetic_truth"]

    assert truth["sep_mas"] == 200.0
    assert truth["pa_deg"] == 10.0
    assert truth["contrast"] == 1000.0
    assert truth["aggregate_snr"] == 40.0
    assert truth["uv_samples"] == 47
    assert truth["disco_modes"] == 210


def test_load_oi_data_single_filter_matches_amigo_sign_convention():
    product = onp.load(PRODUCT, allow_pickle=True).item()
    oidata = load_oi_data(PRODUCT, "F430M")

    assert oidata.observable_kind == "mixed_log_complex"
    assert np.allclose(oidata.u, -np.asarray(product["F430M"]["u"]))
    assert np.allclose(oidata.v, -np.asarray(product["F430M"]["v"]))


def test_mixed_disco_standardize_model_matches_operator_formula():
    oidata = load_oi_data(PRODUCT, "F430M")
    model = BinaryModelCartesian(100.0, -50.0, 1e-3)
    cvis = model.model(oidata.u, oidata.v, oidata.wavel)
    log_cvis = np.log(cvis)

    expected = oidata.vis_mat @ log_cvis.real + oidata.phi_mat @ log_cvis.imag

    assert np.allclose(oidata.standardize_model(cvis), expected)
    assert np.allclose(oidata.standardize_model(cvis), expected)


def test_mixed_disco_flatten_data_returns_coefficients_and_sigma():
    product = onp.load(PRODUCT, allow_pickle=True).item()
    oidata = load_oi_data(PRODUCT, "F480M")
    data, errors = oidata.flatten_data()

    assert np.allclose(
        data, np.asarray(product["F480M"]["disco_coefficients"])
    )
    assert np.allclose(errors, np.asarray(product["F480M"]["disco_sigma"]))
    assert np.allclose(data, oidata.vis)
    assert np.allclose(errors, oidata.d_vis)


def test_mixed_disco_likelihood_is_finite_and_unity_is_zero_target():
    oidata = load_oi_data(PRODUCT, "F380M")
    model = BinaryModelCartesian(100.0, -50.0, 1e-3)

    unity = np.ones_like(oidata.u, dtype=complex)
    assert np.allclose(
        oidata.standardize_model(unity), np.zeros_like(oidata.vis)
    )
    assert np.isfinite(model_loglike(model, oidata))


def test_mixed_disco_record_validation_rejects_bad_shapes_and_covariance():
    bad_operator = dict(_small_mixed_record())
    bad_operator["disco_phase_model_operator"] = onp.ones((2, 2))
    with pytest.raises(ValueError, match="model operators"):
        OIData(bad_operator)

    bad_covariance = dict(_small_mixed_record())
    bad_covariance["disco_covariance"] = onp.array([[0.25, 0.1], [0.1, 1.0]])
    with pytest.raises(ValueError, match="not diagonal"):
        OIData(bad_covariance)

    bad_sigma = dict(_small_mixed_record())
    bad_sigma["disco_sigma"] = onp.array([1.0, 0.5])
    bad_sigma["disco_covariance"] = onp.diag(bad_sigma["disco_sigma"] ** 2)
    with pytest.raises(ValueError, match="increasing uncertainty"):
        OIData(bad_sigma)
