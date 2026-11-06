"""fit_staged_nuisances: image, nuisances, joint, with a reference gain and box edges."""

import jax
import jax.numpy as np
import numpy as onp
import numpyro.distributions as dist
import pytest

from virgil import ComplexVisData, LinearImage
from virgil._utils import mas2rad
from virgil.imaging import MaxEntropy
from virgil.radio import fit_staged_nuisances

PIXEL = 2.0  # mas
NPIX = 6
NOBS = 3
PER_OBS = 150


@pytest.fixture(autouse=True)
def float64():
    with jax.enable_x64(True):
        yield


def _scene(values):
    return LinearImage(
        values["pixels"],
        PIXEL,
        ft="dft",
        parametrisation="sqrt",
        reference_flux=1.0,
    )


def _truth(seed=3):
    flux = onp.random.default_rng(seed).uniform(0.5, 1.5, (NPIX, NPIX))
    return np.asarray(onp.sqrt(flux / flux.sum()))


def _data(gains, weights=(1.0, 1.0, 1.0), sigma=1e-3, seed=0):
    """Complex visibilities of ``_truth`` with per-observation gains."""
    rng = onp.random.default_rng(seed)
    n = NOBS * PER_OBS
    q = 0.4 / (2.0 * mas2rad * PIXEL) * onp.sqrt(rng.uniform(0.05, 1, n))
    phi = rng.uniform(0, 2 * onp.pi, n)
    u, v = q * onp.cos(phi), q * onp.sin(phi)
    obs = onp.repeat(onp.arange(NOBS), PER_OBS)
    w = onp.repeat(onp.asarray(weights, float), PER_OBS) / sigma**2
    clean = ComplexVisData.from_arrays(
        u, v, onp.ones(n) + 0j, w, onp.full(n, 230e9), obs=obs
    )
    nuis = _nuis(clean)
    model = onp.asarray(
        clean.model(_scene({"pixels": _truth()}), {"pixels": _truth()}, nuis)
    )
    g = onp.asarray(gains)[obs]
    noise = rng.normal(size=n) + 1j * rng.normal(size=n)
    vis = g * model + sigma / onp.sqrt(onp.repeat(weights, PER_OBS)) * noise
    return ComplexVisData.from_arrays(
        u, v, vis, w, onp.full(n, 230e9), obs=obs
    )


def _nuis(data):
    from virgil import CalibrationNuisance

    return CalibrationNuisance(
        data,
        gains=False,
        offsets=False,
        alpha=False,
        sigmas=False,
        fit_flux=False,
    )


def _run(data, **kw):
    prior = dist.ImproperUniform(
        dist.constraints.real, (), event_shape=(NPIX, NPIX)
    )
    start = {"pixels": np.full((NPIX, NPIX), onp.sqrt(1.0 / NPIX**2))}
    return fit_staged_nuisances(
        data,
        _scene,
        {"pixels": prior},
        [MaxEntropy(1e-6)],
        init=start,
        nuisance=dict(
            offsets=False, sigmas=False, alpha=False, fit_flux=False
        ),
        weight=1e-6,
        max_steps=500,
        **kw,
    )


def test_reference_is_the_largest_sum_of_weights():
    data = _data([1, 1, 1], weights=(0.2, 3.0, 1.0))
    result = _run(data)
    assert result.ref_obs == 1
    assert onp.argmax(result.sum_weights) == 1
    assert result.values["cal.gain"].shape == (NOBS - 1,)


def test_explicit_reference_overrides():
    assert _run(_data([1, 1, 1]), ref_obs=2).ref_obs == 2


def test_gains_are_recovered_relative_to_the_reference():
    gains = onp.array([1.0, 1.08, 0.93])
    result = _run(_data(gains), ref_obs=0)
    assert onp.allclose(result.values["cal.gain"], gains[1:], atol=0.02)
    assert not result.failed
    assert onp.all(result.nuisances["cal.gain"]["edge_frac"] > 0.05)


def test_edge_flag_when_truth_is_outside_the_box():
    result = _run(_data([1.0, 1.5, 1.0]), ref_obs=0)
    assert result.failed
    assert bool(result.edge["cal.gain"][0])
    assert not bool(result.edge["cal.gain"][1])


def test_stage_three_reports_kkt():
    result = _run(_data([1.0, 1.05, 0.95]), ref_obs=0)
    assert len(result.stages) == 3
    assert "converged" in result.kkt
    assert "kkt_w" in result.kkt["images"]["."]
    assert "cal.gain" in result.kkt["nuisances"]
