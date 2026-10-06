from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

import jax
import jax.numpy as np

from virgil.inference import gaussian_fisher
from virgil.likelihood import (
    joint_loglike,
    joint_prediction,
    model_loglike,
)


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "examples"
    / "hierarchical_binary_workflow.py"
)
SPEC = spec_from_file_location("hierarchical_binary_workflow", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
WORKFLOW = module_from_spec(SPEC)
sys.modules[SPEC.name] = WORKFLOW
SPEC.loader.exec_module(WORKFLOW)


def test_joint_loglike_and_filter_specific_parameter_coupling():
    observations, truth = WORKFLOW.simulate_observations(noise_scale=0.0)
    joint = joint_loglike(truth, WORKFLOW.binary_model, observations)
    individual = sum(
        model_loglike(WORKFLOW.binary_model(truth, index), observation)
        for index, observation in enumerate(observations)
    )
    jacobian = jax.jacrev(
        lambda log_flux: joint_prediction(
            {**truth, "log10_flux": log_flux},
            WORKFLOW.binary_model,
            observations,
        )
    )(truth["log10_flux"])
    block_size = observations[0].flatten_data()[0].size

    assert np.allclose(joint, individual)
    for data_index in range(3):
        data_block = jacobian[
            data_index * block_size : (data_index + 1) * block_size
        ]
        for flux_index in range(3):
            if data_index == flux_index:
                assert np.any(np.abs(data_block[:, flux_index]) > 0.0)
            else:
                assert np.allclose(data_block[:, flux_index], 0.0)


def test_joint_expected_fisher_shape_and_shared_geometry():
    observations, truth = WORKFLOW.simulate_observations(noise_scale=0.0)
    prediction = lambda params: joint_prediction(
        params, WORKFLOW.binary_model, observations
    )
    fmat, _ = gaussian_fisher(
        prediction, truth, WORKFLOW.observation_errors(observations)
    )

    assert fmat.shape == (5, 5)
    assert np.all(np.isfinite(fmat))
    assert np.allclose(fmat, fmat.T)
    assert np.all(np.diag(fmat) > 0.0)


def test_hierarchical_binary_workflow_recovers_truth():
    summary = WORKFLOW.run_hierarchical_binary_demo()

    assert summary.final_chi2r < 2.0
    assert summary.final_chi2r < summary.initial_chi2r
    assert abs(summary.recovered["dra"] - summary.truth["dra"]) < 0.5
    assert abs(summary.recovered["ddec"] - summary.truth["ddec"]) < 0.5
    assert np.allclose(
        summary.recovered["log10_flux"],
        summary.truth["log10_flux"],
        atol=0.03,
    )
    assert summary.expected_fisher.shape == (5, 5)
    assert summary.observed_information.shape == (5, 5)
    assert np.all(np.isfinite(summary.laplace_covariance))
