"""linear_flux_grid: closed-form linearised flux map (fouriever ``lincmap``)."""

import warnings

import jax.numpy as np
import numpy as onp
import pytest

from virgil.grid_fit import (
    laplace_flux_uncertainty_grid,
    linear_flux_grid,
    optimized_flux_grid,
)
from virgil.likelihood import build_model, whitened_residuals
from virgil.models import BinaryModelCartesian
from virgil.oidata import OIData
from tests._test_data import oidata

TRUE_POS = (250.0, 150.0)


def _simulate(flux, noise_scale, seed=1):
    """Binary data on the nu Hor baselines, with errors scaled by noise_scale."""
    cvis = BinaryModelCartesian(*TRUE_POS, flux).model(
        oidata.u, oidata.v, oidata.wavel
    )
    rng = onp.random.default_rng(seed)
    d_vis, d_phi = oidata.d_vis * noise_scale, oidata.d_phi * noise_scale
    return OIData(
        {
            "u": oidata.u,
            "v": oidata.v,
            "wavel": oidata.wavel,
            "vis": oidata.to_vis(cvis)
            + rng.standard_normal(oidata.vis.shape) * d_vis,
            "d_vis": d_vis,
            "phi": oidata.to_phases(cvis)
            + rng.standard_normal(oidata.phi.shape) * d_phi,
            "d_phi": d_phi,
            "i_cps1": oidata.i_cps1,
            "i_cps2": oidata.i_cps2,
            "i_cps3": oidata.i_cps3,
            "v2_flag": oidata.v2_flag,
            "cp_flag": oidata.cp_flag,
        }
    )


def _grid():
    # The flux axis is ignored by linear_flux_grid; it only names the flux.
    return {
        "dra": np.array([TRUE_POS[0], 100.0, -300.0]),
        "ddec": np.array([TRUE_POS[1], -200.0, 50.0]),
        "flux": np.array([1e-3]),
    }


@pytest.fixture(scope="module")
def faint():
    data = _simulate(1e-3, noise_scale=0.1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        optimized = optimized_flux_grid(data, BinaryModelCartesian, _grid())
    laplace = laplace_flux_uncertainty_grid(
        data, BinaryModelCartesian, _grid(), flux=optimized
    )
    return data, optimized, laplace


def test_faint_companion_matches_optimizer_and_laplace(faint):
    data, optimized, laplace = faint
    flux, error, snr = linear_flux_grid(data, BinaryModelCartesian, _grid())
    assert flux.shape == error.shape == snr.shape == (3, 3)
    # Measured: at the true pixel f_hat is 0.9977e-3 against 0.9992e-3
    # (0.15% apart, from O(f^2) terms), and sigma_f agrees to 0.3%.
    assert onp.isclose(flux[0, 0], optimized[0, 0], rtol=0.01)
    assert onp.isclose(flux[0, 0], 1e-3, rtol=0.05)
    assert onp.allclose(error, laplace, rtol=0.02)
    # Off-source pixels (f_hat ~ 4e-4 from the companion's leakage) agree
    # to well under a sigma_f (3e-6).
    assert onp.allclose(flux, optimized, atol=1e-5)
    assert onp.allclose(snr, flux / error)


def test_snr_peaks_at_true_position(faint):
    data = faint[0]
    fine = {
        "dra": TRUE_POS[0] + np.linspace(-30.0, 30.0, 13),
        "ddec": TRUE_POS[1] + np.linspace(-30.0, 30.0, 13),
        "flux": np.array([1e-3]),
    }
    _, _, snr = linear_flux_grid(data, BinaryModelCartesian, fine)
    assert np.unravel_index(np.argmax(snr), snr.shape) == (6, 6)


def test_bright_companion_is_biased():
    # Documented limitation: for f ~ 0.3 the linearisation underestimates
    # the flux (measured 0.20 for a true 0.30, with the optimizer at 0.30),
    # so we only check the sign and the size of the bias, not agreement.
    data = _simulate(0.3, noise_scale=0.01)
    flux, _, _ = linear_flux_grid(data, BinaryModelCartesian, _grid())
    assert 0.1 < flux[0, 0] < 0.27


def test_gradient_matches_finite_difference():
    """The jvp g equals the fouriever-style finite difference at f0=1e-4.

    fouriever's ``lincmap`` (``util.clin``) uses g = model(f0)/f0 with
    f0 = 1e-4 and the same weighted least squares; here the exact
    derivative is compared with that finite difference of the whitened
    residuals, through sigma_f = (g.g)**-0.5. (fouriever itself is not
    importable in the test environment.)
    """
    data = _simulate(1e-3, noise_scale=0.1)
    _, error, _ = linear_flux_grid(data, BinaryModelCartesian, _grid())
    params = ("dra", "ddec", "flux")

    def resid(flux):
        values = [TRUE_POS[0], TRUE_POS[1], flux]
        return whitened_residuals(
            build_model(BinaryModelCartesian, params, values), data
        )

    g = (resid(1e-4) - resid(0.0)) / 1e-4
    assert onp.isclose(error[0, 0], 1.0 / np.sqrt(np.sum(g * g)), rtol=0.02)
