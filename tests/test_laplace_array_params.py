"""laplace_cov, fisher and laplace_parameter_uncertainty with array paths.

``values`` is the flat concatenation of every path's elements, so an array
path of size k takes k entries and the covariance is over those elements.
"""

import jax
import numpy as onp
import pytest

import virgil.models as vm
from virgil.coverage import vlti_oidata
from virgil.inference import (
    fisher,
    laplace_cov,
    laplace_parameter_uncertainty,
)
from virgil.likelihood import loglike

SCALAR = ["rim.diam", "rim.fwhm", "rim.inc", "rim.pa", "rim.flux"]


def _setup(n_az):
    amps = onp.full(n_az, 0.5)
    pas = onp.linspace(120.0, 60.0, n_az)
    model = vm.System(
        star=vm.PointSource(),
        rim=vm.ModulatedGaussianRim(6.0, 1.0, 45.0, 30.0, amps, pas, 0.8),
    )
    data = vlti_oidata(
        wavelengths_m=onp.linspace(1.5e-6, 2.4e-6, 6)
    ).with_model(model)
    return model, data


def test_array_of_size_one_matches_scalar_only_covariance():
    with jax.enable_x64(True):
        model, data = _setup(1)
        scalar = laplace_cov(
            onp.array([6.0, 1.0, 45.0, 30.0, 0.8]), SCALAR, model, data
        )
        paths = SCALAR[:4] + ["rim.az_amps", "rim.az_pas", "rim.flux"]
        values = onp.array([6.0, 1.0, 45.0, 30.0, 0.5, 120.0, 0.8])
        cov = laplace_cov(values, paths, model, data)
        assert cov.shape == (7, 7)
        keep = [0, 1, 2, 3, 6]
        # The scalar-only model has fixed (not free) az_amps/az_pas, so
        # compare through the full information matrix instead.
        info = fisher(values, paths, model, data)
        sub = onp.linalg.inv(info)[onp.ix_(keep, keep)]
        assert onp.allclose(sub, cov[onp.ix_(keep, keep)])
        assert onp.all(onp.isfinite(scalar))


@pytest.mark.validates(
    "virgil.inference.laplace_cov", roots=["self-consistency"]
)
def test_array_path_of_size_two_is_flattened():
    with jax.enable_x64(True):
        model, data = _setup(2)
        paths = [
            "rim.diam",
            "rim.az_amps",
            "rim.az_pas",
            "rim.flux",
        ]
        values = onp.array([6.0, 0.5, 0.5, 120.0, 60.0, 0.8])
        cov = laplace_cov(values, paths, model, data)
        assert cov.shape == (6, 6)
        info = fisher(values, paths, model, data, ridge=1e-10)
        assert onp.allclose(cov, onp.linalg.inv(info + 1e-10 * onp.eye(6)))
        sigma = laplace_parameter_uncertainty(
            values, paths, model, data, "rim.flux"
        )
        assert onp.allclose(sigma, info[5, 5] ** -0.5)
        with pytest.raises(ValueError, match="single scalar"):
            laplace_parameter_uncertainty(
                values, paths, model, data, "rim.az_amps"
            )
        with pytest.raises(ValueError, match="entries"):
            laplace_cov(values[:-1], paths, model, data)


@pytest.mark.validates(
    "virgil.inference.laplace_cov",
    "virgil.inference.fisher",
    roots=["mathematics"],
)
def test_information_matches_finite_difference_hessian():
    with jax.enable_x64(True):
        model, data = _setup(2)
        paths = ["rim.diam", "rim.az_amps", "rim.az_pas", "rim.flux"]
        x0 = onp.array([6.0, 0.5, 0.5, 120.0, 60.0, 0.8])

        def nll(x):
            parts = [x[0], x[1:3], x[3:5], x[5]]
            return -float(loglike(parts, paths, model, data))

        h = 1e-3 * onp.maximum(onp.abs(x0), 1.0)
        n = len(x0)
        hess = onp.zeros((n, n))
        for i in range(n):
            for j in range(n):
                ei, ej = onp.eye(n)[i] * h[i], onp.eye(n)[j] * h[j]
                hess[i, j] = (
                    nll(x0 + ei + ej)
                    - nll(x0 + ei - ej)
                    - nll(x0 - ei + ej)
                    + nll(x0 - ei - ej)
                ) / (4 * h[i] * h[j])
        info = onp.asarray(fisher(x0, paths, model, data))
        scale = onp.sqrt(
            onp.outer(onp.abs(onp.diag(info)), onp.abs(onp.diag(info)))
        )
        assert onp.allclose(info / scale, hess / scale, atol=1e-4)
