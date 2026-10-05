import pytest
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as np
import numpy as onp

from virgil.amigo import load_oi_data
from virgil.likelihood import model_loglike, whitened_residuals
from virgil.models import BinaryModelCartesian
from virgil.oidata import OIData

from ._test_data import oidata_sim, true_values

PRODUCT = (
    Path(__file__).resolve().parents[1] / "data" / "calibrated_visibility.npy"
)
TRUTH = BinaryModelCartesian(*true_values)


def _shift_phases(data, shift):
    """Closure-phase data equal to the TRUTH model plus ``shift`` radians."""
    exact = data.model(TRUTH)[data.vis.size :]
    return eqx.tree_at(lambda d: d.phi, data, exact + shift)


@pytest.mark.validates(
    "virgil.likelihood.whitened_residuals",
    "virgil.likelihood.model_loglike",
    roots=["self-consistency"],
)
def test_whitened_residuals_are_residuals_over_sigma_for_small_phases():
    model = BinaryModelCartesian(240.0, 160.0, 6e-4)
    data, errors = oidata_sim.flatten_data()
    # These residuals are far below π, so no wrapping is needed (wrapping
    # via mod(Δ + π, 2π) - π would itself cost float32 precision).
    delta = oidata_sim.model(model) - data
    assert np.max(np.abs(delta)) < 0.1
    n_vis = oidata_sim.vis.size
    whitened = whitened_residuals(model, oidata_sim)
    assert np.allclose(
        whitened[:n_vis], delta[:n_vis] / errors[:n_vis], rtol=1e-5
    )
    # Closure phases are whitened as correlated groups (see test_closure);
    # for small Δ, sin Δ is Δ to O(Δ³). The penalty rows that follow are
    # O(Δ²/σ), far below the whitened ones here.
    k = oidata_sim.cp_noise.size
    phases, _ = oidata_sim.cp_noise.whiten(delta[n_vis:], errors[n_vis:])
    assert np.allclose(
        whitened[n_vis : n_vis + k], phases, rtol=1e-4, atol=1e-4
    )
    assert whitened.size == oidata_sim.n_residuals


@pytest.mark.validates(
    "virgil.likelihood.whitened_residuals",
    "virgil.likelihood.model_loglike",
    roots=["self-consistency"],
)
def test_phase_term_is_von_mises_and_smooth_across_pi():
    n_vis = oidata_sim.vis.size
    sigma = oidata_sim.d_phi
    # A common shift of every closure phase gives the same sin Δ in each:
    # the whitened block is sin Δ times a fixed vector, followed by the
    # periodic penalty 2 sin²(Δ/2)/σ. Both are unchanged by 2π.
    k = oidata_sim.cp_noise.size
    unit, _ = oidata_sim.cp_noise.whiten(np.ones_like(sigma), sigma)
    for shift in (0.3, 3.0, np.pi, 3.3, 2.0 * np.pi - 0.3):
        data = _shift_phases(oidata_sim, shift)
        r = whitened_residuals(TRUTH, data)[n_vis:]
        assert np.allclose(r[:k], np.sin(-shift) * unit, rtol=1e-4, atol=1e-3)
        penalty = 2.0 * np.sin(0.5 * shift) ** 2 / sigma
        assert np.allclose(r[k:], penalty, rtol=1e-4, atol=1e-6)

    def loglike(shift):
        return model_loglike(TRUTH, _shift_phases(oidata_sim, shift))

    # The old wrapped-Δ² term had a kink at π (slopes of opposite sign on
    # either side); the sine and penalty terms have zero slope there.
    below, above = (
        jax.grad(loglike)(np.pi - 1e-3),
        jax.grad(loglike)(np.pi + 1e-3),
    )
    scale = np.sum(1.0 / sigma**2)
    assert abs(below) < 2e-3 * scale and abs(above) < 2e-3 * scale


@pytest.mark.validates(
    "virgil.likelihood.whitened_residuals",
    "virgil.likelihood.model_loglike",
    roots=["self-consistency"],
)
def test_model_loglike_is_gaussian_in_whitened_residuals():
    model = BinaryModelCartesian(240.0, 160.0, 6e-4)
    # The errors that normalise the likelihood: the visibilities' own, and
    # for the correlated closure phases the Cholesky diagonal of their
    # covariance, whose log-sum is half its log-determinant.
    _, raw = oidata_sim.flatten_data()
    n_vis = oidata_sim.vis.size
    _, phase_errors = oidata_sim.cp_noise.whiten(
        np.zeros_like(raw[n_vis:]), raw[n_vis:]
    )
    errors = np.concatenate([raw[:n_vis], phase_errors])
    r = whitened_residuals(model, oidata_sim)
    # The periodic penalty rows (after the whitened ones) are normalised
    # to nothing: they take no part in the density's constant.
    n_density = errors.size
    assert r.size == oidata_sim.n_residuals
    expected = (
        -0.5 * np.sum(r**2)
        - np.sum(np.log(errors))
        - 0.5 * n_density * np.log(2.0 * np.pi)
    )
    assert np.allclose(model_loglike(model, oidata_sim), expected, rtol=1e-6)


@pytest.mark.validates(
    "virgil.likelihood.whitened_residuals",
    "virgil.likelihood.model_loglike",
    roots=["self-consistency"],
)
def test_projected_phases_are_plain_residuals_over_sigma():
    data = load_oi_data(PRODUCT)["F430M"]
    model = BinaryModelCartesian(150.0, 100.0, 1e-2)
    observed, errors = data.flatten_data()
    expected = (data.model(model) - observed) / errors
    assert onp.allclose(whitened_residuals(model, data), expected)


def _absolute_phase_data(n, phase_offsets, phase_error, seed=0):
    """Uncorrelated absolute phases: TRUTH's, plus ``phase_offsets``."""
    rng = onp.random.default_rng(seed)
    u, v = rng.uniform(-5.0, 5.0, (2, n))
    cvis = onp.asarray(TRUTH.model(u, v, 4.8e-6))
    return OIData(
        {
            "u": u,
            "v": v,
            "wavel": 4.8e-6,
            "vis": onp.abs(cvis) ** 2,
            "d_vis": onp.full(n, 1e-2),
            "phi": onp.angle(cvis) - phase_offsets,
            "d_phi": onp.full(n, phase_error),
            "v2_flag": True,
            "cp_flag": False,
        }
    )


@pytest.mark.validates(
    "virgil.likelihood.whitened_residuals",
    "virgil.likelihood.model_loglike",
    roots=["self-consistency"],
)
def test_uncorrelated_phase_likelihood_is_normalised_on_the_circle():
    # Review 1.2: with the Gaussian normaliser -log σ the von Mises
    # "density" integrated to 1.17 over the circle at σ = 1.
    deltas = np.linspace(-np.pi, np.pi, 4000, endpoint=False)
    for sigma in (0.1, 0.5, 1.0, 2.0, 5.0):
        data = _absolute_phase_data(1, onp.zeros(1), sigma)
        assert data.cp_noise is None and data._phases_wrap

        def loglike(delta, data=data):
            shifted = eqx.tree_at(lambda d: d.phi, data, data.phi - delta)
            return model_loglike(TRUTH, shifted)

        # The visibility is exact: its Gaussian density at zero residual.
        log_vis = -np.log(1e-2) - 0.5 * np.log(2.0 * np.pi)
        density = np.exp(jax.vmap(loglike)(deltas) - log_vis)
        integral = np.sum(density) * (2.0 * np.pi / deltas.size)
        assert abs(float(integral) - 1.0) < 1e-3, (sigma, integral)


def test_fitted_phase_error_scale_recovers_large_noise():
    # Review 1.2: with the Gaussian normaliser, the maximum-likelihood
    # phase-error scale for von Mises noise of σ = 1.8 rad came out as 1.31.
    sigma = 1.8
    rng = onp.random.default_rng(2)
    noise = rng.vonmises(0.0, 1.0 / sigma**2, 10_000)
    data = _absolute_phase_data(noise.size, noise, 1.0, seed=3)
    scales = np.linspace(1.0, 3.0, 201)
    profile = jax.vmap(lambda s: model_loglike(TRUTH, data, phi_scale=s))(
        scales
    )
    best = float(scales[np.argmax(profile)])
    assert abs(best - sigma) < 0.15, best


def test_small_phase_errors_keep_the_gaussian_normaliser():
    data = _absolute_phase_data(20, onp.full(20, 0.01), 1e-2)
    r = whitened_residuals(TRUTH, data)
    _, errors = data.flatten_data()
    gaussian = (
        -0.5 * np.sum(r**2)
        - np.sum(np.log(errors))
        - 0.5 * r.size * np.log(2.0 * np.pi)
    )
    assert np.allclose(model_loglike(TRUTH, data), gaussian, atol=1e-3)
