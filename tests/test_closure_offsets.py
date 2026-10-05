"""Closure-phase offsets per frame (virgil.gains.ClosureOffsets, Stage 6d)."""

import jax
import jax.numpy as np
import numpy as onp
import numpyro.distributions as dist
import pytest

from virgil.coverage import vlti_oidata
from virgil.fitting import fit
from virgil.likelihood import model_loglike
from virgil.models import BinaryModelCartesian

TRUTH = BinaryModelCartesian(dra=6.0, ddec=-4.0, flux=0.3)
N_WAVE = 4


def _data(key=0, stations=None, hours=(-2.0, 0.0, 2.0)):
    kwargs = {} if stations is None else {"stations": stations}
    data = vlti_oidata(
        hour_angles_h=hours,
        wavelengths_m=onp.linspace(2.0e-6, 2.4e-6, N_WAVE),
        sigma_v2=0.01,
        sigma_cp_deg=0.5,
        **kwargs,
    )
    return data.with_model(TRUTH, jax.random.PRNGKey(key))


@pytest.mark.validates(
    "virgil.gains.ClosureOffsets", roots=["mathematics"], kind="check"
)
def test_offsets_match_the_dense_whitened_covariance():
    data = _data()
    with_offsets = data.with_closure_offsets(baseline=0.01, triangle=0.005)
    offsets, cp_noise = with_offsets.phase_offsets, data.cp_noise
    assert offsets.group.shape[0] == 3  # a block per frame
    n_vis = data.vis.size
    prediction = data.model(TRUTH)
    resid = prediction[n_vis:] - data.phi
    sigma = data.d_phi
    w = onp.asarray(cp_noise.whiten(np.sin(resid), sigma)[0], float)
    modes = onp.asarray(offsets.modes(cp_noise, data.phi.size))
    wu = onp.stack(
        [onp.asarray(cp_noise.whiten(m, sigma)[0]) for m in modes.T], axis=1
    )
    cov = onp.eye(w.size) + wu @ wu.T
    chi2 = w @ onp.linalg.solve(cov, w)
    logdet = onp.linalg.slogdet(cov)[1]
    expected = -0.5 * (chi2 - w @ w) - 0.5 * logdet
    difference = model_loglike(TRUTH, with_offsets) - model_loglike(
        TRUTH, data
    )
    assert float(difference) == pytest.approx(expected, rel=1e-4, abs=1e-4)


def test_baseline_offsets_close_around_four_telescopes():
    # Offsets on baselines reach the closure phases as T·e, so the four
    # triangles of four telescopes still satisfy
    # φ123 − φ124 + φ134 − φ234 = 0, unlike offsets per triangle.
    data = _data(hours=(0.0,)).with_closure_offsets(baseline=0.02)
    drawn = onp.asarray(
        data.phase_offsets.sample(
            jax.random.PRNGKey(1),
            data.cp_noise,
            data.phi.size,
            data.phase_offsets.widths,
        )
    ).reshape(4, N_WAVE)
    assert onp.abs(drawn).max() > 1e-3
    closure = drawn[0] - drawn[1] + drawn[2] - drawn[3]
    assert onp.allclose(closure, 0.0, atol=1e-6)
    # ...and are common to the channels.
    assert onp.allclose(drawn, drawn[:, :1], atol=1e-6)


def test_widths_zero_widths_and_refusals():
    data = _data()
    plain = float(model_loglike(TRUTH, data))
    zero = data.with_closure_offsets(baseline=0.0)
    assert float(model_loglike(TRUTH, zero)) == pytest.approx(plain, rel=1e-6)
    fixed = data.with_closure_offsets(baseline=0.02)
    assert float(model_loglike(TRUTH, fixed)) == pytest.approx(
        float(model_loglike(TRUTH, zero, phi_offset_baseline=0.02)), rel=1e-6
    )
    with pytest.raises(ValueError, match="with_closure_offsets"):
        model_loglike(TRUTH, data, phi_offset_baseline=0.01)
    with pytest.raises(ValueError, match="no modes"):
        model_loglike(TRUTH, fixed, phi_offset_triangle=0.01)
    with pytest.raises(ValueError, match="afterwards"):
        fixed.select(wavel_max=2.3e-6)


def test_three_telescopes_are_refused():
    data = vlti_oidata(
        stations=onp.array([[0.0, 0.0], [50.0, 10.0], [20.0, 60.0]]),
        hour_angles_h=(0.0,),
        wavelengths_m=onp.linspace(2.0e-6, 2.4e-6, N_WAVE),
    )
    with pytest.raises(ValueError, match="four"):
        data.with_closure_offsets(baseline=0.01)


def test_supplied_modes_must_stay_in_a_frame():
    data = _data()
    n_phase = data.phi.size
    within = onp.zeros(n_phase)
    within[:3] = 0.01
    assert data.with_closure_offsets(
        modes=within
    ).phase_offsets.groups_present == ("modes",)
    across = onp.full(n_phase, 0.01)
    with pytest.raises(ValueError, match="one frame"):
        data.with_closure_offsets(modes=across)


def test_with_model_draws_offsets_common_to_the_channels():
    data = _data().with_closure_offsets(triangle=0.02)
    noiseless = data.with_model(TRUTH)
    drawn = data.with_model(TRUTH, jax.random.PRNGKey(5), noise_scale=0.0)
    shift = onp.asarray(drawn.phi - noiseless.phi).reshape(-1, N_WAVE)
    assert onp.abs(shift).max() > 1e-3
    assert onp.allclose(shift, shift[:, :1], atol=1e-6)


def test_fit_with_a_fitted_offset_width():
    data = _data(key=3).with_closure_offsets(triangle=0.005)
    result = fit(
        TRUTH.set("flux", 0.25),
        {"flux": dist.Uniform(0.0, 1.0)},
        data,
        noise={"phi_offset_triangle": dist.Uniform(0.0, 0.2)},
    )
    assert result.info["method"] == "lbfgs"
    assert 0.0 <= float(result.values["noise.phi_offset_triangle"]) < 0.2
