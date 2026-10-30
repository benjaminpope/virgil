import pytest
import jax.numpy as np

from virgil.inference import (
    fisher_projection,
    expected_fisher,
    hessian_matrix,
    objective_laplace_covariance,
)


@pytest.mark.validates(
    "virgil.inference.observed_information",
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


def test_objective_laplace_covariance_is_finite_and_positive_diagonal():
    objective = lambda x: (x[0] / 2.0) ** 2 + (x[1] / 3.0) ** 2
    x0 = np.array([0.1, -0.3])

    cov = objective_laplace_covariance(objective, x0, ridge=1e-8)
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


def test_expected_fisher_supports_parameter_pytrees():
    params = {"offset": np.array(0.2), "slopes": np.array([1.0, -0.5])}
    design = np.array([[1.0, 2.0, 0.0], [1.0, 0.0, 3.0]])
    errors = np.array([0.5, 2.0])

    def prediction(values):
        vector = np.concatenate(
            [np.atleast_1d(values["offset"]), values["slopes"]]
        )
        return design @ vector

    fmat, unravel = expected_fisher(prediction, params, errors)
    expected = design.T @ np.diag(errors**-2) @ design
    restored = unravel(np.array([0.3, 1.1, -0.4]))

    assert np.allclose(fmat, expected)
    assert np.allclose(restored["offset"], 0.3)
    assert np.allclose(restored["slopes"], np.array([1.1, -0.4]))
    assert np.all(np.linalg.eigvalsh(fmat) >= -1e-6)


@pytest.mark.validates(
    "virgil.inference.observed_information",
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

    expected, _ = expected_fisher(prediction, params, errors)
    observed = hessian_matrix(objective, params)

    assert np.allclose(expected, observed, rtol=1e-5, atol=1e-6)


@pytest.mark.validates(
    "virgil.inference.observed_information",
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

    expected, _ = expected_fisher(prediction, params, errors)
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

    from virgil import BinaryModelCartesian, laplace_cov, observed_information
    from virgil.coverage import nrm_oidata
    from virgil.inference import laplace_parameter_uncertainty

    truth = BinaryModelCartesian(120.0, 80.0, 1e-3)
    data = nrm_oidata(sigma_v2=1e-3, sigma_cp_deg=0.05).with_model(truth)
    params = ["dra", "ddec", "flux"]
    values = [120.0, 80.0, 1e-3]
    with jax.enable_x64(True):
        cov64 = onp.asarray(laplace_cov(values, params, truth, data))
        info64 = onp.asarray(observed_information(values, params, truth, data))
        assert cov64.dtype == onp.float64
        sigma64 = onp.asarray(
            laplace_parameter_uncertainty(values, params, truth, data, "flux")
        )
    cov = laplace_cov(values, params, truth, data)
    info = observed_information(values, params, truth, data)
    # Results come back in the ambient precision (float64 in the x64 CI job).
    ambient = np.float64 if jax.config.jax_enable_x64 else np.float32
    assert cov.dtype == ambient and info.dtype == ambient

    def close(a, b):
        return onp.allclose(a, b, rtol=1e-5, atol=1e-6 * onp.abs(b).max())

    assert close(cov, cov64) and close(info, info64)
    sigma = laplace_parameter_uncertainty(values, params, truth, data, "flux")
    assert sigma.dtype == ambient and close(sigma, sigma64)
    # A single diagonal curvature is robust to float32 (to ~1e-7, within
    # ``close``), so the control is exact: with 64-bit ambient precision,
    # dtype="float32" must give a measurably different number from the
    # float64 default, and a bad dtype must be refused.
    with jax.enable_x64(True):
        sigma32 = float(
            laplace_parameter_uncertainty(
                values, params, truth, data, "flux", dtype="float32"
            )
        )
        sigma_default = float(
            laplace_parameter_uncertainty(values, params, truth, data, "flux")
        )
    assert abs(sigma32 - sigma_default) > 1e-10 * sigma_default
    assert onp.isclose(sigma32, sigma_default, rtol=1e-4)
    import pytest

    with pytest.raises(ValueError, match="dtype"):
        laplace_parameter_uncertainty(
            values, params, truth, data, "flux", dtype="float16"
        )
    # dtype="float32" is the ambient calculation, which is measurably off.
    assert not close(
        laplace_cov(values, params, truth, data, dtype="float32"), cov64
    )


def test_old_curvature_names_still_work_and_warn():
    import pytest

    import virgil
    from virgil import inference

    x0 = np.array([0.5, -1.0])
    objective = lambda x: (x[0] - 1.0) ** 2 + 3.0 * (x[1] + 2.0) ** 2
    with pytest.warns(FutureWarning, match="laplace_covariance is now"):
        old = inference.laplace_covariance(objective, x0)
    assert np.allclose(
        old, inference.objective_laplace_covariance(objective, x0)
    )

    with pytest.warns(FutureWarning, match="gaussian_fisher is now"):
        fmat, _ = inference.gaussian_fisher(lambda p: p, x0, np.ones(2))
    assert np.allclose(fmat, np.eye(2))

    # ``virgil.fisher`` is the same deprecated alias of the new name.
    assert virgil.fisher.__name__ == "fisher"
    assert virgil.observed_information is inference.observed_information
    for name in inference.__all__:
        assert hasattr(inference, name)


def test_safe_azimuth_matches_arctan2_and_has_a_finite_gradient_at_the_origin():
    import jax

    from virgil._geometry import safe_azimuth

    x = np.array([0.0, 1.0, -2.0, 0.0, 3.0])
    y = np.array([0.0, 0.0, 1.5, -1.0, 4.0])
    origin = (x == 0.0) & (y == 0.0)
    assert np.array_equal(
        safe_azimuth(x, y), np.arctan2(x, np.where(origin, 1.0, y))
    )
    grad = jax.grad(lambda p: safe_azimuth(p[0], p[1]))(np.zeros(2))
    assert np.all(np.isfinite(grad))


def test_the_laplace_ridge_is_relative_to_the_curvature():
    # The same problem in other units must give the same covariance up to the
    # unit change: an absolute ridge of 1e-10 did not.
    from virgil.inference import regularized_inverse

    hess = np.array([[4.0, 1.0], [1.0, 3.0]])
    small = regularized_inverse(1e-12 * hess)  # far below an absolute 1e-10
    assert np.allclose(small, regularized_inverse(hess) * 1e12, rtol=1e-6)
    assert np.allclose(regularized_inverse(hess) @ hess, np.eye(2), atol=1e-8)


def test_a_nearly_unconstrained_parameter_warns_that_the_ridge_decides():
    import warnings

    from virgil.inference import regularized_inverse

    with pytest.warns(RuntimeWarning, match="nearly unconstrained"):
        cov = regularized_inverse(np.diag(np.array([4.0, 1e-13])))
    assert np.isfinite(cov).all()
    with warnings.catch_warnings():  # a well-conditioned Hessian is silent
        warnings.simplefilter("error")
        regularized_inverse(np.diag(np.array([4.0, 1e-3])))
