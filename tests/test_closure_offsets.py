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


def _three_telescopes():
    return vlti_oidata(
        stations=onp.array([[0.0, 0.0], [50.0, 10.0], [20.0, 60.0]]),
        hour_angles_h=(0.0, 1.0),
        wavelengths_m=onp.linspace(2.0e-6, 2.4e-6, N_WAVE),
    ).with_model(TRUTH, jax.random.PRNGKey(1))


def test_three_telescopes_get_triangle_offsets_and_fold_baseline_into_them():
    data = _three_telescopes()
    assert data.cp_noise is None
    with pytest.warns(UserWarning, match="three telescopes"):
        both = data.with_closure_offsets(baseline=0.01, triangle=0.02)
    assert both.phase_offsets.groups_present == ("triangle",)
    # A second call replaces the offsets and still folds (the structure, not
    # the call history, says three telescopes).
    with pytest.warns(UserWarning, match="three telescopes"):
        twice = data.with_closure_offsets(triangle=0.02).with_closure_offsets(
            baseline=0.01
        )
    with pytest.warns(UserWarning, match="three telescopes"):
        once = data.with_closure_offsets(baseline=0.01)
    assert float(model_loglike(TRUTH, twice)) == pytest.approx(
        float(model_loglike(TRUTH, once)), rel=1e-6
    )
    modes = onp.zeros((1, data.phi.size))
    modes[0, :N_WAVE] = 0.01  # one frame
    with_modes = data.with_closure_offsets(modes=modes)
    assert with_modes.phase_offsets.groups_present == ("modes",)
    # The offsets' effect on the sines, against a direct Gaussian per frame:
    # covariance diag(σ²) + w² 11ᵀ, w² = 0.02² + 3 · 0.01².
    zero = data.with_closure_offsets(triangle=0.0)
    resid = onp.asarray(data.phi - data.with_model(TRUTH).phi)
    sigma = onp.asarray(data.d_phi)
    width2 = 0.02**2 + 3 * 0.01**2

    def gaussian(w2):
        total = 0.0
        for r, s in zip(
            onp.sin(resid).reshape(-1, N_WAVE), sigma.reshape(-1, N_WAVE)
        ):
            c = onp.diag(s**2) + w2
            total -= (
                0.5 * r @ onp.linalg.solve(c, r)
                + 0.5 * onp.linalg.slogdet(c)[1]
            )
        return total

    expected = gaussian(width2) - gaussian(0.0)
    got = float(model_loglike(TRUTH, both)) - float(model_loglike(TRUTH, zero))
    assert got == pytest.approx(expected, rel=1e-3, abs=1e-6)
    for d in (both, with_modes, zero):
        assert onp.isfinite(float(model_loglike(TRUTH, d)))


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


def test_fit_with_nonzero_injected_closure_offsets_recovers_flux_and_width():
    # High-SNR data (σ_cp = 0.3 deg ≈ 0.0052 rad) with an injected 0.15 rad
    # closure offset per triangle (δ ≫ √(2σ)): both the sine and periodic
    # penalty rows are whitened by the offset covariance, so the fit recovers
    # the companion flux and offset width.
    import equinox as eqx

    from virgil.likelihood import whitened_residuals

    base = vlti_oidata(
        hour_angles_h=(-2.0, -1.0, 0.0, 1.0, 2.0),
        wavelengths_m=onp.linspace(2.0e-6, 2.4e-6, N_WAVE),
        sigma_v2=0.005,
        sigma_cp_deg=0.3,
    ).with_model(TRUTH, jax.random.PRNGKey(11), noise_scale=0.2)
    rng = onp.random.default_rng(12)
    n_tri = base.phi.size // N_WAVE
    offsets = onp.repeat(rng.choice([-0.15, 0.15], size=n_tri), N_WAVE)
    shifted = eqx.tree_at(
        lambda d: d.phi, base, base.phi + np.asarray(offsets)
    )
    data = shifted.with_closure_offsets(triangle=0.15)
    n_vis = data.vis.size
    k = data.cp_noise.size
    r = onp.asarray(whitened_residuals(TRUTH, data))
    assert onp.max(onp.abs(r[n_vis + k :])) < 1.0
    result = fit(
        TRUTH.set("flux", 0.2),
        {"flux": dist.Uniform(0.05, 0.8)},
        data,
        noise={"phi_offset_triangle": dist.LogUniform(0.02, 0.4)},
    )
    assert float(result.values["flux"]) == pytest.approx(0.3, abs=0.03)
    assert float(result.values["noise.phi_offset_triangle"]) == pytest.approx(
        0.15, rel=0.4
    )
