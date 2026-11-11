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
    std = onp.sqrt(onp.sum(modes**2, axis=1))
    pen_num = 2.0 * onp.sin(0.5 * onp.asarray(resid, float)) ** 2
    sigma_f = onp.asarray(sigma, float)
    pen_diff = -0.5 * onp.sum(
        (pen_num / onp.hypot(sigma_f, std)) ** 2 - (pen_num / sigma_f) ** 2
    )
    expected = -0.5 * (chi2 - w @ w) - 0.5 * logdet + pen_diff
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
    selected = fixed.select(wavel_max=2.3e-6)
    fresh = data.select(wavel_max=2.3e-6).with_closure_offsets(baseline=0.02)
    assert float(model_loglike(TRUTH, selected)) == pytest.approx(
        float(model_loglike(TRUTH, fresh)), rel=1e-6
    )


def _three_telescopes(hours=(0.0, 1.0)):
    return vlti_oidata(
        stations=onp.array([[0.0, 0.0], [50.0, 10.0], [20.0, 60.0]]),
        hour_angles_h=hours,
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


def _mode_loglike_change(frames, value, triangle=None):
    """The log-likelihood change from a mode of ``value`` on every channel of
    ``frames`` (width 1), with optional triangle offsets of width ``triangle``
    in every frame, and the dense Gaussian's. The sines (three telescopes: one
    triangle per group) have covariance diag(σ²) + m mᵀ + τ² Σ_f 1_f 1_fᵀ over
    all closure phases, 1_f the channels of frame f."""
    data = _three_telescopes(hours=(0.0, 1.0, 2.0))
    n_phase = data.phi.size
    ref = data.with_closure_offsets(triangle=0.0)
    mode = onp.zeros((1, n_phase))
    for f in frames:
        mode[0, f * N_WAVE : (f + 1) * N_WAVE] = value
    with_mode = data.with_closure_offsets(modes=mode, triangle=triangle)
    angle = onp.asarray(data.phi - data.with_model(TRUTH).phi)
    resid = onp.sin(angle)
    pen_num = 2.0 * onp.sin(0.5 * angle) ** 2
    sigma = onp.asarray(data.d_phi)

    def gaussian(m, tau):
        c = onp.diag(sigma**2) + onp.outer(m, m)
        for f in range(n_phase // N_WAVE):
            one = onp.zeros(n_phase)
            one[f * N_WAVE : (f + 1) * N_WAVE] = 1.0
            c = c + tau**2 * onp.outer(one, one)
        pen = -0.5 * onp.sum(pen_num**2 / (sigma**2 + m**2 + tau**2))
        return (
            -0.5 * resid @ onp.linalg.solve(c, resid)
            - 0.5 * (onp.linalg.slogdet(c)[1])
            + pen
        )

    tau = 0.0 if triangle is None else triangle
    got = float(model_loglike(TRUTH, with_mode)) - float(
        model_loglike(TRUTH, ref)
    )
    return got, gaussian(mode[0], tau) - gaussian(0 * mode[0], 0.0)


@pytest.mark.parametrize(
    "frames, triangle",
    [((0,), None), ((0, 1), None), ((0, 1), 0.03), ((0, 2), 0.03)],
)
def test_modes_give_the_common_offset_covariance(frames, triangle):
    # One offset δ ~ N(0, 1) times the mode is common to its frames. Two
    # float64 pipelines (Woodbury blocks vs a dense solve) of O(1) terms agree
    # to a few hundred ulp: 1e-12. Frame 2 (or 1) gets its own padded block
    # beside the spanning one when triangle offsets are on.
    with jax.enable_x64(True):
        got, expected = _mode_loglike_change(frames, 0.05, triangle)
    assert abs(expected) > 1e-3  # the mode matters
    assert got == pytest.approx(expected, rel=1e-12, abs=1e-12)


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
    # closure offset per triangle: the whitened sine rows absorb the offset so
    # the fit recovers the companion flux and offset width, while the
    # unwhitened periodic penalty still rejects a π phase flip.
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
    assert onp.max(onp.abs(r[n_vis : n_vis + k])) < 2.0
    # A π phase flip on the closure phases is still strongly rejected by the
    # periodic penalty 2 sin²(Δ/2)/hypot(σ, σ_off) (~1780 nats on 80 phases at
    # σ_off = 0.3 rad) while avoiding unmarginalized 1/σ² bias on small offsets.
    pi_flipped = eqx.tree_at(lambda d: d.phi, data, data.phi + np.pi)
    assert (
        float(model_loglike(TRUTH, data, phi_offset_triangle=0.3))
        - float(model_loglike(TRUTH, pi_flipped, phi_offset_triangle=0.3))
        > 1e3
    )
    result = fit(
        TRUTH.set("flux", 0.2),
        {"flux": dist.Uniform(0.05, 0.8)},
        data,
        noise={"phi_offset_triangle": dist.LogUniform(0.02, 0.4)},
    )
    assert float(result.values["flux"]) == pytest.approx(0.3, abs=0.05)
    assert float(result.values["noise.phi_offset_triangle"]) == pytest.approx(
        0.15, rel=0.4
    )


def test_spanning_closure_offsets_keep_per_frame_blocks_and_match_dense():
    hours = (-2.5, -1.5, -0.5, 0.5, 1.5, 2.5)
    data = _data(hours=hours)
    n_phase = data.phi.size
    rng = onp.random.default_rng(7)
    night_modes = rng.normal(0.0, 0.02, (2, n_phase))

    # Spanning-only: 1 empty padding block, no zero-size leaves.
    span_only = data.with_closure_offsets(modes=night_modes)
    off_span = span_only.phase_offsets
    assert off_span.group.shape == (1, 1)
    assert not onp.any(off_span.group_mask)
    assert off_span.spanning.shape == (n_phase, 2)
    assert not [
        a
        for a in jax.tree_util.tree_leaves(span_only)
        if hasattr(a, "size") and a.size == 0
    ]

    # Frame-local + spanning: stays 1 block per frame (6 blocks, not merged).
    mixed = data.with_closure_offsets(
        baseline=0.01, triangle=0.005, modes=night_modes
    )
    off_mix = mixed.phase_offsets
    assert off_mix.group.shape[0] == len(hours)
    assert off_mix.spanning.shape == (n_phase, 2)
    assert not [
        a
        for a in jax.tree_util.tree_leaves(mixed)
        if hasattr(a, "size") and a.size == 0
    ]

    with jax.enable_x64(True):
        from virgil._precision import cast_tree

        d64 = cast_tree(data, "float64")
        for cand in (span_only, mixed):
            c64 = cast_tree(cand, "float64")
            offsets, cp_noise = c64.phase_offsets, d64.cp_noise
            pred = d64.model(TRUTH)
            resid = onp.asarray(pred[d64.vis.size :] - d64.phi, float)
            sigma = onp.asarray(d64.d_phi, float)
            w = onp.asarray(
                cp_noise.whiten(resid * 0 + onp.sin(resid), sigma)[0]
            )
            dense_modes = onp.asarray(offsets.modes(cp_noise, n_phase), float)
            wu = onp.stack(
                [
                    onp.asarray(cp_noise.whiten(m, sigma)[0])
                    for m in dense_modes.T
                ],
                axis=1,
            )
            cov = onp.eye(w.size) + wu @ wu.T
            chi2 = w @ onp.linalg.solve(cov, w)
            logdet = onp.linalg.slogdet(cov)[1]
            std = onp.sqrt(onp.sum(dense_modes**2, axis=1))
            pen_num = 2.0 * onp.sin(0.5 * resid) ** 2
            pen_diff = -0.5 * onp.sum(
                (pen_num / onp.hypot(sigma, std)) ** 2 - (pen_num / sigma) ** 2
            )
            expected = -0.5 * (chi2 - w @ w) - 0.5 * logdet + pen_diff
            got = float(model_loglike(TRUTH, c64) - model_loglike(TRUTH, d64))
            assert got == pytest.approx(expected, rel=1e-11, abs=1e-11)

    keys = jax.random.split(jax.random.PRNGKey(9), 15000)
    draws = onp.asarray(
        jax.vmap(
            lambda k: off_mix.sample(k, data.cp_noise, n_phase, off_mix.widths)
        )(keys)
    )
    emp_cov = (draws.T @ draws) / len(draws)
    dense_m = onp.asarray(off_mix.modes(data.cp_noise, n_phase))
    assert onp.allclose(emp_cov, dense_m @ dense_m.T, atol=2e-4)


@pytest.mark.parametrize("three_tel", [False, True])
def test_select_and_split_by_epoch_preserve_closure_offsets(three_tel):
    from virgil._precision import cast_tree

    kwargs = (
        {"stations": onp.array([[0.0, 0.0], [50.0, 10.0], [20.0, 60.0]])}
        if three_tel
        else {}
    )
    two_epoch = vlti_oidata(
        hour_angles_h=(-0.5, 0.5),
        nights_mjd=(60000.0, 60005.0),
        wavelengths_m=onp.linspace(2.0e-6, 2.4e-6, N_WAVE),
        sigma_v2=0.01,
        sigma_cp_deg=0.5,
        **kwargs,
    ).with_model(TRUTH, jax.random.PRNGKey(0))
    n_phase = two_epoch.phi.size
    half = n_phase // 2
    # Multi-frame mode confined to Epoch 0 (touches frames 0 and 1 only).
    epoch0_mode = onp.zeros((1, n_phase))
    epoch0_mode[0, :half] = 0.025
    with_off = two_epoch.with_closure_offsets(
        triangle=0.015, modes=epoch0_mode
    )
    assert not with_off.phase_offsets.labels_shared(
        two_epoch.epochs()[onp.asarray(two_epoch.i_cps1)]
    )
    parts = with_off.split_by_epoch()
    assert len(parts) == 2
    assert parts[0].phase_offsets.spanning is not None
    assert parts[1].phase_offsets.spanning is None
    with jax.enable_x64(True):
        w64 = cast_tree(with_off, "float64")
        p64 = [cast_tree(p, "float64") for p in parts]
        total = sum(float(model_loglike(TRUTH, p)) for p in p64)
        assert total == pytest.approx(
            float(model_loglike(TRUTH, w64)), rel=1e-12, abs=1e-12
        )

    # A mode crossing epochs is refused by split_by_epoch.
    cross = onp.full((1, n_phase), 0.02)
    shared = two_epoch.with_closure_offsets(modes=cross)
    assert shared.phase_offsets.labels_shared(
        two_epoch.epochs()[onp.asarray(two_epoch.i_cps1)]
    )
    with pytest.raises(ValueError, match="spans several epochs"):
        shared.split_by_epoch()


def test_wrap_penalty_scaling_gradients_and_unbiased_fit_at_tiny_photon_noise():
    import equinox as eqx

    base = vlti_oidata(
        hour_angles_h=(-2.0, -1.0, 0.0, 1.0, 2.0),
        wavelengths_m=onp.linspace(2.0e-6, 2.4e-6, N_WAVE),
        sigma_v2=0.005,
        sigma_cp_deg=0.1,
    ).with_model(TRUTH, jax.random.PRNGKey(21), noise_scale=0.2)
    n_phase = base.phi.size
    partial_mode = onp.zeros((1, n_phase))
    partial_mode[0, : n_phase // 2] = 0.12
    partial_data = base.with_closure_offsets(modes=partial_mode)
    off = partial_data.phase_offsets
    std0 = onp.asarray(
        off.marginal_std(partial_data.cp_noise, n_phase, np.array([0.0]))
    )
    assert onp.array_equal(std0, onp.zeros(n_phase))
    std1 = onp.asarray(
        off.marginal_std(partial_data.cp_noise, n_phase, np.array([1.0]))
    )
    assert onp.allclose(std1[: n_phase // 2], 0.12)
    assert onp.array_equal(std1[n_phase // 2 :], onp.zeros(n_phase // 2))

    # Smooth finite gradients with respect to model flux and width, even at 0.
    for w in (0.0, 0.15):
        g_flux, g_w = jax.grad(
            lambda f, tau: model_loglike(
                TRUTH.set("flux", f), partial_data, phi_offset_modes=tau
            ),
            argnums=(0, 1),
        )(0.28, w)
        assert bool(np.isfinite(g_flux)) and bool(np.isfinite(g_w))

    # 10 deg (~0.1745 rad) instrumental closure-phase offsets at 0.1 deg photon
    # noise: fit recovers TRUTH.flux without quartic wrap-penalty bias, while a
    # π flip is still strongly rejected.
    rng = onp.random.default_rng(22)
    deg10 = onp.deg2rad(10.0)
    n_tri = n_phase // N_WAVE
    inj = onp.repeat(rng.choice([-deg10, deg10], size=n_tri), N_WAVE)
    shifted = eqx.tree_at(lambda d: d.phi, base, base.phi + np.asarray(inj))
    data = shifted.with_closure_offsets(triangle=deg10)
    pi_flipped = eqx.tree_at(lambda d: d.phi, data, data.phi + np.pi)
    assert (
        float(model_loglike(TRUTH, data))
        - float(model_loglike(TRUTH, pi_flipped))
        > 500.0
    )
    result = fit(
        TRUTH.set("flux", 0.2),
        {"flux": dist.Uniform(0.05, 0.8)},
        data,
    )
    assert float(result.values["flux"]) == pytest.approx(0.3, abs=0.015)
