import pytest
import jax.numpy as np

from virgil.inference import (
    fisher_projection,
    gaussian_fisher,
    hessian_matrix,
    laplace_covariance,
)


@pytest.mark.validates(
    "virgil.inference.fisher",
    "virgil.inference.laplace_cov",
    roots=["self-consistency"],
)
def test_hessian_shape_and_symmetry():
    objective = lambda x: (x[0] - 1.0) ** 2 + 3.0 * (x[1] + 2.0) ** 2
    x0 = np.array([0.2, -1.1])

    hess = hessian_matrix(objective, x0)

    assert hess.shape == (2, 2)
    assert np.allclose(hess, hess.T)
    assert np.allclose(hess, np.diag(np.array([2.0, 6.0])))


def test_laplace_covariance_is_finite_and_positive_diagonal():
    objective = lambda x: (x[0] / 2.0) ** 2 + (x[1] / 3.0) ** 2
    x0 = np.array([0.1, -0.3])

    cov = laplace_covariance(objective, x0, ridge=1e-8)
    assert cov.shape == (2, 2)
    assert np.all(np.isfinite(cov))
    assert np.all(np.diag(cov) > 0.0)


def test_fisher_projection_whitens_local_metric():
    fmat = np.array([[5.0, 1.0], [1.0, 2.0]])
    proj = fisher_projection(fmat)
    ident = proj.T @ fmat @ proj

    assert proj.shape == (2, 2)
    assert np.all(np.isfinite(proj))
    assert np.allclose(ident, np.eye(2), atol=1e-5)


def test_gaussian_fisher_supports_parameter_pytrees():
    params = {"offset": np.array(0.2), "slopes": np.array([1.0, -0.5])}
    design = np.array([[1.0, 2.0, 0.0], [1.0, 0.0, 3.0]])
    errors = np.array([0.5, 2.0])

    def prediction(values):
        vector = np.concatenate(
            [np.atleast_1d(values["offset"]), values["slopes"]]
        )
        return design @ vector

    fmat, unravel = gaussian_fisher(prediction, params, errors)
    expected = design.T @ np.diag(errors**-2) @ design
    restored = unravel(np.array([0.3, 1.1, -0.4]))

    assert np.allclose(fmat, expected)
    assert np.allclose(restored["offset"], 0.3)
    assert np.allclose(restored["slopes"], np.array([1.1, -0.4]))
    assert np.all(np.linalg.eigvalsh(fmat) >= -1e-6)


@pytest.mark.validates(
    "virgil.inference.fisher",
    "virgil.inference.laplace_cov",
    roots=["self-consistency"],
)
def test_expected_fisher_matches_noiseless_observed_information():
    params = np.array([0.7, -0.2])
    errors = np.array([0.3, 0.5])
    prediction = lambda values: np.array(
        [values[0] ** 2 + values[1], np.sin(values[0]) - values[1]]
    )
    data = prediction(params)
    objective = lambda values: 0.5 * np.sum(
        ((data - prediction(values)) / errors) ** 2
    )

    expected, _ = gaussian_fisher(prediction, params, errors)
    observed = hessian_matrix(objective, params)

    assert np.allclose(expected, observed, rtol=1e-5, atol=1e-6)


@pytest.mark.validates(
    "virgil.inference.fisher",
    "virgil.inference.laplace_cov",
    roots=["self-consistency"],
)
def test_nonlinear_residual_curvature_changes_observed_information():
    params = np.array([0.7, -0.2])
    errors = np.array([0.3, 0.5])
    prediction = lambda values: np.array(
        [values[0] ** 2 + values[1], np.sin(values[0]) - values[1]]
    )
    data = prediction(params) + np.array([0.4, -0.2])
    objective = lambda values: 0.5 * np.sum(
        ((data - prediction(values)) / errors) ** 2
    )

    expected, _ = gaussian_fisher(prediction, params, errors)
    observed = hessian_matrix(objective, params)

    assert not np.allclose(expected, observed, rtol=1e-4, atol=1e-5)


def test_fisher_projection_of_zero_matrix_is_finite():
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        proj = fisher_projection(np.zeros((2, 2)), eps=1e-12)
    assert np.all(np.isfinite(proj))
    assert np.allclose(np.abs(proj).max(), 1e6)


def test_laplace_cov_and_fisher_run_in_float64_by_default():
    # A faint companion (Δmag 7.5) with tight errors: in float32 the
    # covariance differs from float64 by ~5e-4. Like fit, the model-level
    # curvatures work in float64 whatever JAX's ambient precision, and
    # return the result in that precision.
    import jax
    import numpy as onp

    from virgil import BinaryModelCartesian, fisher, laplace_cov
    from virgil.coverage import nrm_oidata

    truth = BinaryModelCartesian(120.0, 80.0, 1e-3)
    data = nrm_oidata(sigma_v2=1e-3, sigma_cp_deg=0.05).with_model(truth)
    params = ["dra", "ddec", "flux"]
    values = [120.0, 80.0, 1e-3]
    with jax.enable_x64(True):
        cov64 = onp.asarray(laplace_cov(values, params, data, truth))
        info64 = onp.asarray(fisher(values, params, data, truth))
        assert cov64.dtype == onp.float64
    cov = laplace_cov(values, params, data, truth)
    info = fisher(values, params, data, truth)
    # Results come back in the ambient precision (float64 in the x64 CI job).
    ambient = np.float64 if jax.config.jax_enable_x64 else np.float32
    assert cov.dtype == ambient and info.dtype == ambient

    def close(a, b):
        return onp.allclose(a, b, rtol=1e-5, atol=1e-6 * onp.abs(b).max())

    assert close(cov, cov64) and close(info, info64)
    # dtype="float32" is the ambient calculation, which is measurably off.
    assert not close(
        laplace_cov(values, params, data, truth, dtype="float32"), cov64
    )
