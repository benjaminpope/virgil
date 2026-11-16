import numpy as onp
import pytest

from virgil.oidata import OIData


def _record(vis, d_vis):
    n, m = 8, 5
    rng = onp.random.default_rng(0)
    return {
        "u": rng.uniform(1.0, 5.0, n),
        "v": rng.uniform(1.0, 5.0, n),
        "wavel": onp.full(n, 2.2e-6),
        "vis": vis,
        "d_vis": d_vis,
        "vis_mat": rng.normal(size=(m, n)),
    }


def test_projected_data_build_when_finite():
    OIData(_record(onp.full(5, 0.9), onp.full(5, 0.01)))


@pytest.mark.parametrize(
    "vis, d_vis",
    [
        (onp.array([0.9, onp.nan, 0.9, 0.9, 0.9]), onp.full(5, 0.01)),
        (onp.full(5, 0.9), onp.array([0.01, onp.inf, 0.01, 0.01, 0.01])),
        (onp.full(5, 0.9), onp.array([0.01, -0.01, 0.01, 0.01, 0.01])),
        (onp.full(5, 0.9), onp.array([0.01, 0.0, 0.01, 0.01, 0.01])),
    ],
)
def test_projected_data_must_be_finite_with_positive_errors(vis, d_vis):
    with pytest.raises(ValueError, match="already projected"):
        OIData(_record(vis, d_vis))
