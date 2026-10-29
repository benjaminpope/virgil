"""CLEAN edge cases: all-zero data, and runs that find no components."""

import numpy as onp
import pytest

from virgil.radio import ComplexVisData
from virgil.radio_clean import clean

S = 40.0
N = 33


def _data(scale=1.0):
    rng = onp.random.default_rng(0)
    r = 1.2e6 * onp.sqrt(rng.uniform(0.02, 1.0, 500))
    t = rng.uniform(0, 2 * onp.pi, 500)
    u, v = r * onp.cos(t), r * onp.sin(t)
    vis = scale * onp.exp(-2j * onp.pi * 1e-7 * u) * onp.ones(len(u))
    return ComplexVisData.from_arrays(u, v, vis, 1.0, 230e9)


@pytest.mark.parametrize("deconvolver", ["hogbom", "multiscale"])
def test_zero_visibilities_stop_cleanly(deconvolver):
    res = clean(_data(0.0), N, S, deconvolver=deconvolver, niter=50)
    assert res.stop == "threshold"
    assert res.iterations == 0
    assert not res.model.any()
    assert not res.image.any()


def test_no_components_still_restorable():
    res = clean(_data(), N, S, niter=0)
    assert res.iterations == 0
    assert res.model.shape == (N, N) and not res.model.any()
    assert onp.isfinite(res.image).all()
    assert onp.allclose(res.image, res.residual)  # empty model: image = residual
    assert res.model_visibilities(_data()).shape == (500,)
