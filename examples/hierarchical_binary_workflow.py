from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import optimistix as optx
from jax.flatten_util import ravel_pytree

from virgil.inference import (
    fisher_projection,
    gaussian_fisher,
    hessian_matrix,
    regularized_inverse,
)
from virgil.likelihood import joint_loglike, joint_prediction
from virgil.models import BinaryModelCartesian
from virgil.oidata import OIData


WAVELENGTHS = jnp.array([800e-9, 1.0e-6, 1.2e-6])
FILTER_LABELS = ("800 nm", "1.0 micron", "1.2 microns")


@dataclass(frozen=True)
class HierarchicalRecoverySummary:
    truth: dict[str, object]
    initial: dict[str, object]
    recovered: dict[str, object]
    initial_chi2r: float
    final_chi2r: float
    expected_fisher: jax.Array
    observed_information: jax.Array
    laplace_covariance: jax.Array


def binary_model(params, observation_index):
    """Build one binary with shared geometry and filter-specific flux."""
    return BinaryModelCartesian(
        params["dra"],
        params["ddec"],
        10.0 ** params["log10_flux"][observation_index],
    )


def build_template_observations():
    """Build three observations with common baselines and distinct wavelengths."""
    angles = jnp.linspace(0.0, 2.0 * jnp.pi, 18, endpoint=False)
    lengths = jnp.linspace(1.5, 6.0, angles.size)
    u = lengths * jnp.cos(angles)
    v = lengths * jnp.sin(angles)

    observations = []
    for wavelength in WAVELENGTHS:
        observations.append(
            OIData(
                {
                    "u": u,
                    "v": v,
                    "wavel": jnp.array([wavelength]),
                    "vis": jnp.ones_like(u),
                    "d_vis": jnp.full_like(u, 3e-4),
                    "phi": jnp.zeros_like(u),
                    "d_phi": jnp.full_like(u, 0.04),
                    "phi_unit": "deg",
                    "v2_flag": False,
                    "cp_flag": False,
                }
            )
        )
    return tuple(observations)


def simulate_observations(seed=7, noise_scale=1.0):
    """Simulate three binary observations with shared astrometry."""
    truth = {
        "dra": jnp.array(18.0),
        "ddec": jnp.array(-12.0),
        "log10_flux": jnp.log10(jnp.array([0.015, 0.010, 0.007])),
    }
    templates = build_template_observations()
    keys = jax.random.split(jax.random.key(seed), len(templates))
    observations = tuple(
        observation.with_model(
            binary_model(truth, index),
            key=keys[index],
            noise_scale=noise_scale,
        )
        for index, observation in enumerate(templates)
    )
    return observations, truth


def observation_errors(observations):
    """Concatenate uncertainties in joint-prediction order."""
    return jnp.concatenate(
        [observation.flatten_data()[1] for observation in observations]
    )


def reduced_chi2(params, observations):
    """Return reduced chi-squared for the five-parameter joint model."""
    data = jnp.concatenate(
        [observation.flatten_data()[0] for observation in observations]
    )
    model = joint_prediction(params, binary_model, observations)
    errors = observation_errors(observations)
    return jnp.sum(((data - model) / errors) ** 2) / (data.size - 5)


def fit_hierarchical_binary(observations, initial=None, max_steps=256):
    """Fit shared astrometry and three fluxes using Fisher coordinates."""
    if initial is None:
        initial = {
            "dra": jnp.array(15.0),
            "ddec": jnp.array(-9.0),
            "log10_flux": jnp.log10(jnp.array([0.012, 0.012, 0.012])),
        }

    prediction = lambda params: joint_prediction(
        params, binary_model, observations
    )
    errors = observation_errors(observations)
    initial_fisher, unravel = gaussian_fisher(
        prediction, initial, errors, ridge=1e-8
    )
    initial_vector, _ = ravel_pytree(initial)
    projection = fisher_projection(initial_fisher, eps=1e-10)

    def project(latent):
        return unravel(initial_vector + projection @ latent)

    def latent_objective(latent, args):
        del args
        return -joint_loglike(project(latent), binary_model, observations)

    solver = optx.BestSoFarMinimiser(optx.BFGS(rtol=1e-8, atol=1e-8))
    solution = optx.minimise(
        latent_objective,
        solver,
        jnp.zeros_like(initial_vector),
        max_steps=max_steps,
        throw=False,
    )
    recovered = project(solution.value)
    recovered_vector, recovered_unravel = ravel_pytree(recovered)

    def flat_objective(values):
        return -joint_loglike(
            recovered_unravel(values), binary_model, observations
        )

    expected, _ = gaussian_fisher(prediction, recovered, errors)
    observed = hessian_matrix(flat_objective, recovered_vector)
    covariance = regularized_inverse(observed, ridge=1e-8)
    return recovered, expected, observed, covariance


def run_hierarchical_binary_demo(seed=7, noise_scale=1.0):
    """Run the complete simulation, fit, and local uncertainty calculation."""
    observations, truth = simulate_observations(seed, noise_scale)
    initial = {
        "dra": jnp.array(15.0),
        "ddec": jnp.array(-9.0),
        "log10_flux": jnp.log10(jnp.array([0.012, 0.012, 0.012])),
    }
    recovered, expected, observed, covariance = fit_hierarchical_binary(
        observations, initial
    )
    return HierarchicalRecoverySummary(
        truth=truth,
        initial=initial,
        recovered=recovered,
        initial_chi2r=float(reduced_chi2(initial, observations)),
        final_chi2r=float(reduced_chi2(recovered, observations)),
        expected_fisher=expected,
        observed_information=observed,
        laplace_covariance=covariance,
    )


if __name__ == "__main__":
    result = run_hierarchical_binary_demo()
    print("Initial reduced chi-squared:", result.initial_chi2r)
    print("Final reduced chi-squared:", result.final_chi2r)
    print("Recovered parameters:", result.recovered)
