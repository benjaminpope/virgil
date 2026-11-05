"""Radial log-amplitude nuisance for DISCO data (OIData.with_radial_amplitude)."""

import equinox as eqx
import jax
import jax.numpy as np
import numpy as onp
import pytest

from virgil.amigo import load_oi_data
from virgil.grid_fit import linear_flux_grid
from virgil.likelihood import model_loglike
from virgil.models import BinaryModelCartesian
from tests._shared import CALIBRATED_VISIBILITY

MAS = onp.pi / 180 / 3600e3
AXIS = onp.arange(-300.0, 300.1, 30.0)
GRID = {"dra": AXIS, "ddec": AXIS, "flux": onp.array([1e-3])}
WIDE = 1e8  # flat-prior limit, for comparison with a plain projection


@pytest.fixture(autouse=True, scope="module")
def _x64():
    """Compare with a float64 projection to 1e-10, in either CI dtype."""
    with jax.enable_x64(True):
        yield


@pytest.fixture(scope="module")
def disco(_x64):
    """The repository's synthetic DISCO product, with phases switched off
    (amplitude-only) and its data replaced by a bare star."""
    data = load_oi_data(CALIBRATED_VISIBILITY, "F430M")
    return eqx.tree_at(
        lambda d: (d.phi_mat, d.vis),
        data,
        (np.zeros_like(data.phi_mat), np.zeros_like(data.vis)),
    )


def _radial_error(data, coeffs=(0.02, 0.01)):
    """log |V| error 0.02 q² + 0.01 q⁴ per sample, projected like the data."""
    q = onp.hypot(data.u, data.v) / data.wavel
    q = q / q.max()
    err = coeffs[0] * q**2 + coeffs[1] * q**4
    return onp.asarray(data.vis_mat) @ err


def _reference(data, r, positions, order=4):
    """The sketch's QR projection (design/sketches/disco_amp_nuisance.py)."""
    A = onp.asarray(data.vis_mat) / onp.asarray(data.d_vis)[:, None]
    uv = onp.stack([data.u, data.v], 1) / onp.asarray(data.wavel)[..., None]
    q = onp.hypot(data.u, data.v) / data.wavel
    q = q / q.max()
    N = A @ onp.stack([q**k for k in range(order + 1)], 1)
    Q, _ = onp.linalg.qr(N)
    G = A @ (onp.cos(2 * onp.pi * uv @ (positions * MAS).T) - 1.0)
    r = onp.asarray(r) / onp.asarray(data.d_vis)
    out = {}
    for name, (g, rr) in {
        "none": (G, r),
        "nuisance": (G - Q @ (Q.T @ G), r - Q @ (Q.T @ r)),
    }.items():
        gg = onp.sum(g * g, 0)
        out[name] = ((g.T @ rr) / gg, onp.sqrt(gg), g.T @ rr / onp.sqrt(gg))
    return out


def _positions():
    x, y = onp.meshgrid(AXIS, AXIS, indexing="ij")
    keep = onp.hypot(x, y) >= 40.0
    return onp.stack([x[keep], y[keep]], 1), keep


@pytest.mark.validates(
    "virgil.gains.radial_amplitude_modes", roots=["mathematics"], kind="check"
)
def test_snr_map_matches_a_plain_qr_projection(disco):
    rng = onp.random.default_rng(1)
    r = _radial_error(disco) + onp.asarray(disco.d_vis) * rng.normal(
        size=disco.vis.shape
    )
    data = disco.set(["vis"], [np.asarray(r)])
    positions, keep = _positions()
    ref = _reference(data, r, positions)
    for name, d in (
        ("none", data),
        ("nuisance", data.with_radial_amplitude(width=WIDE)),
    ):
        snr = linear_flux_grid(BinaryModelCartesian, d, GRID).snr
        onp.testing.assert_allclose(
            onp.asarray(snr)[keep], ref[name][2], rtol=1e-9, atol=1e-10
        )


def test_the_nuisance_removes_a_radial_error_bias(disco):
    rng = onp.random.default_rng(2)
    r = _radial_error(disco) + onp.asarray(disco.d_vis) * rng.normal(
        size=disco.vis.shape
    )
    data = disco.set(["vis"], [np.asarray(r)])
    keep = _positions()[1]
    plain = onp.asarray(linear_flux_grid(BinaryModelCartesian, data, GRID).snr)
    nuis = onp.asarray(
        linear_flux_grid(
            BinaryModelCartesian, data.with_radial_amplitude(), GRID
        ).snr
    )
    assert plain[keep].mean() < -3.0 and (plain[keep] < 0).mean() > 0.7
    assert abs(nuis[keep].mean()) < 1.0


def test_a_companion_survives_at_the_expected_sensitivity(disco):
    truth = BinaryModelCartesian(dra=90.0, ddec=-60.0, flux=2e-3)
    data = disco.with_model(truth)
    positions, keep = _positions()
    ref = _reference(data, data.vis, positions)
    sens_kept = ref["nuisance"][1] / ref["none"][1]
    i = int(onp.argmin(onp.hypot(*(positions - [90.0, -60.0]).T)))
    plain = linear_flux_grid(BinaryModelCartesian, data, GRID)
    nuis = linear_flux_grid(
        BinaryModelCartesian, data.with_radial_amplitude(), GRID
    )
    ix = [int(onp.argmin(abs(AXIS - c))) for c in (90.0, -60.0)]
    assert 0.2 < sens_kept[i] < 0.95
    assert nuis.flux[ix[0], ix[1]] == pytest.approx(2e-3, rel=0.1)
    ratio = nuis.flux_error[ix[0], ix[1]] / plain.flux_error[ix[0], ix[1]]
    assert ratio == pytest.approx(
        1.0 / sens_kept[i], rel=5e-3
    )  # the default width 10 is not quite flat


def test_the_width_is_fittable_and_finite(disco):
    rng = onp.random.default_rng(3)
    r = _radial_error(disco, (0.05, 0.02)) + onp.asarray(
        disco.d_vis
    ) * rng.normal(size=disco.vis.shape)
    data = disco.set(["vis"], [np.asarray(r)]).with_radial_amplitude(width=0.1)
    assert data.gains.groups == ("radial",)
    model = BinaryModelCartesian(dra=50.0, ddec=0.0, flux=1e-3)
    values = [
        model_loglike(model, data, vis_gain_radial=w) for w in (0.01, 0.1, 1.0)
    ]
    assert onp.all(onp.isfinite(values))
    grad = jax.grad(lambda w: model_loglike(model, data, vis_gain_radial=w))(
        np.asarray(0.1)
    )
    assert onp.isfinite(grad)
    # the wide width beats a width of 0 here: the error is not noise
    assert values[1] > model_loglike(model, data, vis_gain_radial=0.0)


def test_refuses_unprojected_data():
    from tests.test_gains import _data

    with pytest.raises(ValueError, match="projected log-amplitudes"):
        _data().with_radial_amplitude()
