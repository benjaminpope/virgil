"""Injection limits on raw-ADU OI_FLUX data (audit finding 724#12.3b).

The finding said `_injection_limits` adds a dimensionless prediction
difference to raw-ADU `FluxSpectrum` data without matching the grey scale.
It does not: `FluxSpectrum.predict` returns ``mu * template`` in the data's
units (``mu`` the stated scale prior mean), so the injected signal is in ADU
too. These tests pin that: data in ADU (continuum 1e5) and the same data
normalized to 1 give the same limits, and the flux block is not vacuous.
"""

import jax
import numpy as onp
import pytest

from tests.test_chromatic_search import _power, _scene, _template
from virgil import OIData
from virgil.limits import injection_limits

ADU = 1e5
POSITIONS = {
    "comp.dra": onp.array([8.0]),
    "comp.ddec": onp.array([-5.0]),
    "comp.flux.ratio": onp.array([0.01]),
}


def _data(unit, blocks=("v2", "t3", "flux")):
    record = _template(blocks)
    if "flux" in blocks:
        record["flux"]["value"] = record["flux"]["value"] * unit
        record["flux"]["error"] = record["flux"]["error"] * unit
    data = OIData(record)
    if "flux" in blocks:
        data = data.with_flux_scale(scale=(unit, 0.5 * unit))
    truth = _scene(_power(0.05))
    return data.with_model(truth, key=jax.random.PRNGKey(1))


def _limit(data):
    template = _scene(_power(0.05))
    return float(injection_limits(template, data, POSITIONS, 3.0)[0, 0])


def test_adu_and_normalized_flux_give_the_same_injection_limit():
    adu, unit = _limit(_data(ADU)), _limit(_data(1.0))
    assert adu == pytest.approx(unit, rel=1e-3)


def test_flux_block_enters_the_injection_limit():
    """Not vacuous: the spectrum changes the limit, in ADU as normalized."""
    assert _limit(_data(ADU)) != pytest.approx(
        _limit(_data(ADU, ("v2", "t3"))), rel=1e-2
    )
