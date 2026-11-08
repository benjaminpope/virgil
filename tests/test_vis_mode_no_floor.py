"""D9: sub-σ visibilities are never floored; undefined ones are excluded loudly."""

import warnings

import numpy as onp
import pytest

from virgil.oidata import OIData

SIGMA = 0.1
# -0.5σ, 0.5σ, 3σ (and a 4th, well-measured point that must never warn).
RAW = onp.array([-0.5, 0.5, 3.0, 20.0]) * SIGMA


def _data(vis_mode, v2):
    n = RAW.size
    return {
        "u": onp.linspace(1.0, 4.0, n),
        "v": onp.linspace(2.0, 3.0, n),
        "wavel": onp.full(n, 2.2e-6),
        "vis": RAW.copy(),
        "d_vis": onp.full(n, SIGMA),
        "v2_flag": v2,
        "vis_mode": vis_mode,
    }


def _build(vis_mode, v2):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        data = OIData(_data(vis_mode, v2))
    return data, [str(w.message) for w in caught if w.category is UserWarning]


@pytest.mark.parametrize("v2", [True, False])
def test_logamp_excludes_nonpositive_and_keeps_the_rest(v2):
    data, msgs = _build("logamp", v2)
    excluded = [m for m in msgs if "excluded" in m]
    weak = [m for m in msgs if "first-order" in m]
    assert len(excluded) == 1 and "1 of 4" in excluded[0]
    assert len(weak) == 1 and "1 of 4" in weak[0]
    onp.testing.assert_array_equal(data.vis_index, [1, 2, 3])
    kept = RAW[1:]
    expected = 0.5 * onp.log(kept) if v2 else onp.log(kept)
    onp.testing.assert_allclose(data.vis, expected, rtol=1e-5)
    d_expected = (0.5 if v2 else 1.0) * SIGMA / kept
    onp.testing.assert_allclose(data.d_vis, d_expected, rtol=1e-5)


def test_amp_from_v2_excludes_nonpositive_and_keeps_the_rest():
    data, msgs = _build("amp", True)
    assert sum("excluded" in m for m in msgs) == 1
    assert sum("first-order" in m for m in msgs) == 1
    onp.testing.assert_array_equal(data.vis_index, [1, 2, 3])
    kept = RAW[1:]
    onp.testing.assert_allclose(data.vis, onp.sqrt(kept), rtol=1e-5)
    onp.testing.assert_allclose(
        data.d_vis, 0.5 * SIGMA / onp.sqrt(kept), rtol=1e-5
    )


def test_v2_from_amplitude_is_first_order_and_unchanged():
    data, msgs = _build("v2", False)
    assert not msgs and data.vis_index is None
    onp.testing.assert_allclose(data.vis, RAW**2, rtol=1e-5)
    onp.testing.assert_allclose(
        data.d_vis, 2.0 * onp.abs(RAW) * SIGMA, rtol=1e-5
    )


@pytest.mark.parametrize("vis_mode", ["v2", "amp"])
def test_unaffected_modes_do_not_warn(vis_mode):
    # v2 from V² and amp from |V| apply no transform that can be undefined.
    data, msgs = _build(vis_mode, vis_mode == "v2")
    if vis_mode == "v2":
        assert not msgs
        onp.testing.assert_allclose(data.vis, RAW, rtol=1e-6)
    else:
        assert not msgs
        onp.testing.assert_allclose(data.vis, RAW, rtol=1e-6)


def test_operator_data_cannot_drop_samples():
    record = _data("logamp", True)
    record["vis_mat"] = onp.eye(RAW.size)
    with pytest.raises(ValueError, match="cannot be flagged"):
        OIData(record)
