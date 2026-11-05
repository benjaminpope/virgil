"""Radial log-amplitude nuisance for DISCO data (OIData.with_radial_amplitude)."""

import equinox as eqx
import jax
import jax.numpy as np
import numpy as onp
import pytest

from virgil.amigo import load_oi_data
from virgil.grid_fit import linear_flux_grid
from virgil.likelihood import model_loglike, whitened_residuals
from virgil.models import BinaryModelCartesian
from virgil.oidata import OIData
from tests._shared import CALIBRATED_VISIBILITY

MAS = onp.pi / 180 / 3600e3
AXIS = onp.arange(-300.0, 300.1, 30.0)
GRID = {"dra": AXIS, "ddec": AXIS, "flux": onp.array([1e-3])}
WIDE = 1e8  # flat-prior limit, for comparison with a plain projection


@pytest.fixture(params=["float32", "float64"])
def dtype(request):
    """JAX's default float width, then float64 (CI runs both)."""
    if request.param == "float64":
        with jax.enable_x64(True):
            yield request.param
    else:
        yield request.param


@pytest.fixture
def x64():
    """Float64, to compare with a numpy projection to 1e-7."""
    with jax.enable_x64(True):
        yield


@pytest.fixture
def disco(dtype):
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
    """An independent numpy QR projection onto the powers 1, q, ..., q^order."""
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
def test_snr_map_matches_a_plain_qr_projection(x64, disco):
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
            onp.asarray(snr)[keep], ref[name][2], rtol=1e-7, atol=1e-8
        )


@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_the_nuisance_removes_a_radial_error_bias_of_either_sign(disco, sign):
    rng = onp.random.default_rng(2)
    r = _radial_error(disco, (0.02 * sign, 0.01 * sign)) + onp.asarray(
        disco.d_vis
    ) * rng.normal(size=disco.vis.shape)
    data = disco.set(["vis"], [np.asarray(r)])
    keep = _positions()[1]
    plain = onp.asarray(linear_flux_grid(BinaryModelCartesian, data, GRID).snr)
    nuis = onp.asarray(
        linear_flux_grid(
            BinaryModelCartesian, data.with_radial_amplitude(), GRID
        ).snr
    )
    # The plain map is biased against the error's sign, in most of the map;
    # the marginalized one keeps at most a fifth of the bias.
    assert sign * plain[keep].mean() < 0.0
    assert (sign * plain[keep] < 0).mean() > 0.5
    assert abs(nuis[keep].mean()) < 0.2 * abs(plain[keep].mean())


def test_the_nuisance_absorbs_a_companions_symmetric_signal(disco):
    """A real companion's azimuthally symmetric amplitude signal is projected
    out too: its amplitude SNR falls, its peak stays put."""
    at = (90.0, -60.0)
    data = disco.with_model(BinaryModelCartesian(*at, 2e-3))
    plain = onp.asarray(linear_flux_grid(BinaryModelCartesian, data, GRID).snr)
    nuis = onp.asarray(
        linear_flux_grid(
            BinaryModelCartesian, data.with_radial_amplitude(), GRID
        ).snr
    )
    keep = _positions()[1]
    peak = lambda m: onp.unravel_index(
        onp.argmax(onp.where(keep, m, -onp.inf)), m.shape
    )
    assert peak(plain) == peak(nuis)
    assert 0.0 < nuis[peak(nuis)] < 0.95 * plain[peak(plain)]


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


def test_the_width_is_fittable_and_finite(x64, disco):
    rng = onp.random.default_rng(3)
    r = _radial_error(disco, (0.05, 0.02)) + onp.asarray(
        disco.d_vis
    ) * rng.normal(size=disco.vis.shape)
    data = disco.set(["vis"], [np.asarray(r)]).with_radial_amplitude(width=0.1)
    assert data.gains.groups == ("radial",)
    model = BinaryModelCartesian(dra=50.0, ddec=0.0, flux=1e-3)

    def loglike(w):
        return model_loglike(model, data, vis_gain_radial=w)

    values = [loglike(w) for w in (0.01, 0.1, 1.0)]
    assert onp.all(onp.isfinite(values))
    # the width's gradient is the finite difference of the log likelihood
    # (in log width, as the sampler sees it)
    w, h = 0.1, 1e-5
    fd = (loglike(w * onp.exp(h)) - loglike(w * onp.exp(-h))) / (2 * h)
    grad = jax.grad(lambda t: loglike(np.exp(t)))(np.log(w))
    assert float(grad) == pytest.approx(float(fd), rel=1e-5)
    # a radial error is not noise: a finite width fits better than none
    assert values[1] > loglike(0.0)


def test_phases_only_change_by_the_projection(x64):
    """On mixed DISCO data with its phase operator, the nuisance is exactly a
    projection of the whitened residuals off the radial modes."""
    full = load_oi_data(CALIBRATED_VISIBILITY, "F430M")
    assert onp.abs(onp.asarray(full.phi_mat)).max() > 0
    model = BinaryModelCartesian(dra=90.0, ddec=-60.0, flux=2e-3)
    data = full.with_radial_amplitude(width=WIDE)
    r0 = onp.asarray(whitened_residuals(model, full))
    r1 = onp.asarray(whitened_residuals(model, data))
    n = onp.asarray(data.gains.shapes[0]) / onp.asarray(full.d_vis)[:, None]
    q, _ = onp.linalg.qr(n)
    onp.testing.assert_allclose(r1, r0 - q @ (q.T @ r0), rtol=1e-5, atol=1e-6)
    assert onp.linalg.norm(r1) < onp.linalg.norm(r0)


def test_refuses_unprojected_data():
    rng = onp.random.default_rng(0)
    data = OIData(
        {
            "u": rng.normal(0, 60, 6),
            "v": rng.normal(0, 60, 6),
            "wavel": onp.linspace(2.0e-6, 2.4e-6, 3),
            "vis": rng.uniform(0.3, 0.9, (6, 3)),
            "d_vis": onp.full((6, 3), 0.01),
        }
    )
    with pytest.raises(ValueError, match="projected log-amplitudes"):
        data.with_radial_amplitude()


def test_simulating_never_draws_the_nuisance(disco):
    """with_model(key=...) leaves the radial profile out: its width is a
    near-flat prior, not a size."""
    truth = BinaryModelCartesian(dra=90.0, ddec=-60.0, flux=2e-3)
    data = disco.with_radial_amplitude()
    simulated = data.with_model(truth, key=jax.random.PRNGKey(0))
    clean = data.with_model(truth)
    off = onp.asarray(simulated.vis - clean.vis) / onp.asarray(data.d_vis)
    assert onp.abs(off).max() < 6.0  # noise only, not N(0, 10²) coefficients
    assert simulated.gains is not None
