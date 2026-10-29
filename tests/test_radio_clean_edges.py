"""CLEAN edge cases: all-zero data, and minor cycles that find no components."""

import numpy as onp
import pytest

import virgil.radio_clean as rc
from tests.test_radio_clean import C, N, S, _data


@pytest.mark.parametrize("deconvolver", ["hogbom", "multiscale"])
def test_zero_visibilities_stop_cleanly(deconvolver):
    res = rc.clean(_data([]), N, S, deconvolver=deconvolver, niter=50)
    assert res.stop == "threshold"
    assert res.iterations == 0
    assert not res.model.any()
    assert not res.image.any()


def test_minor_cycle_without_components_is_restorable(monkeypatch):
    # a minor cycle that reports work but adds only zeros: the model rebuild
    # sums nothing and must still be an image
    def empty(residual, psf, scales, *args):
        return onp.zeros((len(scales),) + residual.shape), 1, "threshold"

    monkeypatch.setattr(rc, "_minor_cycle", empty)
    data = _data([(1.0, C, C)])
    res = rc.clean(data, N, S, niter=3)
    assert res.iterations == 3 and res.stop == "niter"
    assert res.model.shape == (N, N) and not res.model.any()
    assert onp.allclose(
        res.image, res.residual
    )  # empty model: image = residual
    assert res.model_visibilities(data).shape == (len(data),)
