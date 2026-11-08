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
    weak = [m for m in msgs if "poor there" in m]
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
    assert sum("poor there" in m for m in msgs) == 1
    onp.testing.assert_array_equal(data.vis_index, [1, 2, 3])
    kept = RAW[1:]
    onp.testing.assert_allclose(data.vis, onp.sqrt(kept), rtol=1e-5)
    onp.testing.assert_allclose(
        data.d_vis, 0.5 * SIGMA / onp.sqrt(kept), rtol=1e-5
    )


def test_v2_from_amplitude_is_first_order_and_warns_below_sigma():
    data, msgs = _build("v2", False)
    # -0.5σ and 0.5σ are below σ: kept, counted; none excluded.
    assert not [m for m in msgs if "excluded" in m]
    weak = [m for m in msgs if "poor there" in m]
    assert len(weak) == 1 and "2 of 4" in weak[0]
    assert data.vis_index is None
    onp.testing.assert_allclose(data.vis, RAW**2, rtol=1e-5)
    # Exact SD of a² for Gaussian a: σ√(4a² + 2σ²), finite and positive.
    onp.testing.assert_allclose(
        data.d_vis, SIGMA * onp.sqrt(4 * RAW**2 + 2 * SIGMA**2), rtol=1e-5
    )


def test_v2_from_amplitude_keeps_zero_amplitude_with_finite_error():
    # Weights must not blow up as |V| -> 0 (first-order 2|V|σ would give
    # infinite weight at 0): the exact SD σ√(4V² + 2σ²) stays positive.
    record = _data("v2", False)
    record["vis"] = onp.array([0.0, 1e-9, 3.0, 20.0]) * SIGMA
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        data = OIData(record)
    msgs = [str(w.message) for w in caught if w.category is UserWarning]
    assert not [m for m in msgs if "excluded" in m]
    assert sum("poor there" in m for m in msgs) == 1 and "2 of 4" in msgs[0]
    assert data.vis_index is None
    d = onp.asarray(data.d_vis)
    assert onp.all(d > 0)
    onp.testing.assert_allclose(d[:2], SIGMA * onp.sqrt(2) * SIGMA, rtol=1e-5)


def test_operator_with_zero_error_and_nonpositive_data_raises():
    # _good_samples accepts a raw error of 0 with an operator; it must not
    # pass through to log(V²) <= 0 as NaN.
    record = _data("logamp", True)
    record["vis_mat"] = onp.eye(RAW.size)
    record["d_vis"] = onp.zeros(RAW.size)
    with pytest.raises(ValueError, match="cannot be flagged"):
        OIData(record)


def test_counts_ignore_already_flagged_and_invalid_samples():
    record = _data("logamp", True)
    record["vis"] = onp.array([-0.5, -0.5, 0.5, 3.0]) * SIGMA
    record["d_vis"] = onp.array([SIGMA, SIGMA, SIGMA, onp.nan])
    record["vis_flag"] = onp.array([True, False, False, False])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        data = OIData(record)
    msgs = [str(w.message) for w in caught if w.category is UserWarning]
    # Only sample 1 is a new exclusion among the 2 usable samples (0 is
    # flagged already, 3 has a non-finite error).
    excluded = [m for m in msgs if "excluded" in m]
    assert len(excluded) == 1 and "1 of 2" in excluded[0]
    onp.testing.assert_array_equal(data.vis_index, [2])


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
