"""Independent, whitened closure phases (``virgil._closure``)."""

import equinox as eqx
import jax
import jax.numpy as np
import numpy as onp
import pytest

from virgil.coverage import VLTI_UTS, vlti_oidata
from virgil.likelihood import model_loglike, whitened_residuals
from virgil.models import BinaryModelCartesian
from virgil.oidata import OIData

TRUTH = BinaryModelCartesian(dra=5.0, ddec=3.0, flux=0.1)
MODEL = BinaryModelCartesian(dra=4.0, ddec=3.5, flux=0.08)


def _four_telescopes(**kw):
    template = vlti_oidata(
        hour_angles_h=(-2.0, 0.0, 2.0), wavelengths_m=[3.5e-6]
    )
    return template.with_model(TRUTH, key=jax.random.PRNGKey(0), **kw)


def _incidence(data):
    """The dense triangle-by-baseline matrix T, +1 +1 -1 per closure phase."""
    i1, i2, i3 = (
        onp.asarray(i) for i in (data.i_cps1, data.i_cps2, data.i_cps3)
    )
    t = onp.zeros((i1.size, onp.asarray(data.u).size))
    for row, (a, b, c) in enumerate(zip(i1, i2, i3)):
        t[row, a] += 1.0
        t[row, b] += 1.0
        t[row, c] -= 1.0
    return t


def test_four_telescopes_keep_three_closure_phases_per_frame():
    data = _four_telescopes()
    assert data.cp_noise is not None
    n_vis, n_cp = data.vis.size, data.phi.size  # 3 frames x 4 triangles
    assert n_cp == 12
    assert data.n_independent == n_vis + 9
    # The residuals add one periodic penalty per closure phase (n_residuals).
    assert whitened_residuals(MODEL, data).size == data.n_residuals
    assert data.n_residuals == data.n_independent + n_cp


def test_indices_are_int32_and_whiten_in_either_precision():
    # int64 indices became int32 in float32 runs, and JAX 0.11 then reused
    # that copy in float64 code compiled for int64 (a float32 L-curve then
    # a float64 refit). int32 is the same in both modes.
    data = _four_telescopes()
    noise = data.cp_noise
    assert noise.groups.dtype == onp.int32
    assert noise.keep.dtype == onp.int32
    out = []
    for x64 in (False, True, False):
        with jax.enable_x64(x64):
            out.append(onp.asarray(whitened_residuals(MODEL, data)))
    assert onp.allclose(out[0], out[1], rtol=1e-5)
    assert onp.allclose(out[0], out[2])


@pytest.mark.validates(
    "virgil.oidata.OIData",
    "virgil.likelihood.whitened_residuals",
    roots=["self-consistency"],
)
def test_whitening_matches_the_dense_pseudo_inverse():
    # Equal errors give baseline variances σ²/3, so C = (σ²/3) T Tᵀ, of
    # rank 3 per frame; χ² is Δᵀ C⁺ Δ and the normalisation its pseudo-det.
    data = _four_telescopes()
    t = _incidence(data)
    sigma = onp.asarray(data.d_phi)
    assert onp.allclose(sigma, sigma[0])
    cov = sigma[0] ** 2 / 3.0 * t @ t.T
    prediction = onp.asarray(data.model(MODEL))
    n_vis = data.vis.size
    delta = prediction[n_vis:] - onp.asarray(data.phi)
    chord = onp.sin(delta)
    chi2_dense = chord @ onp.linalg.pinv(cov) @ chord
    eig = onp.linalg.eigvalsh(cov)
    logdet_dense = onp.sum(onp.log(eig[eig > 1e-9 * eig.max()]))

    whitened = onp.asarray(whitened_residuals(MODEL, data))
    k = data.cp_noise.size
    chi2_phase = onp.sum(whitened[n_vis : n_vis + k] ** 2)
    assert chi2_phase == pytest.approx(chi2_dense, rel=1e-5)
    # then one periodic penalty 2 sin²(Δ/2)/σ per closure phase
    penalty = 2.0 * onp.sin(0.5 * delta) ** 2 / sigma
    assert onp.allclose(whitened[n_vis + k :], penalty, rtol=1e-4, atol=1e-6)
    _, errors = data.cp_noise.whiten(np.asarray(chord), data.d_phi)
    assert 2.0 * onp.sum(onp.log(errors)) == pytest.approx(
        logdet_dense, rel=1e-5
    )


@pytest.mark.validates(
    "virgil.oidata.OIData",
    "virgil.likelihood.whitened_residuals",
    roots=["self-consistency"],
)
def test_three_telescopes_are_unchanged():
    # One triangle per frame and channel: nothing correlates, and the
    # likelihood is the per-triangle chord one, as before.
    data = vlti_oidata(
        hour_angles_h=(0.0,),
        wavelengths_m=[3.5e-6],
        stations=onp.asarray(VLTI_UTS)[:3],
    ).with_model(TRUTH)
    assert data.cp_noise is None
    assert data.n_independent == data.vis.size + data.phi.size


@pytest.mark.validates(
    "virgil.likelihood.whitened_residuals", roots=["statistics"]
)
def test_simulated_noise_comes_from_baseline_phases():
    # Closure-phase noise drawn from baseline-phase noise satisfies the one
    # closure relation of four telescopes in every frame: the alternating
    # sum of the four closure phases' noise is zero.
    noiseless = vlti_oidata(
        hour_angles_h=(0.0,), wavelengths_m=[3.5e-6]
    ).with_model(TRUTH)
    noisy = vlti_oidata(
        hour_angles_h=(0.0,), wavelengths_m=[3.5e-6]
    ).with_model(TRUTH, key=jax.random.PRNGKey(3))
    noise = onp.asarray(noisy.phi) - onp.asarray(noiseless.phi)
    null = onp.linalg.svd(_incidence(noisy).T)[2][-1]  # left null vector of T
    assert abs(null @ noise) < 1e-5 * onp.linalg.norm(noise)
    assert onp.linalg.norm(noise) > 0.0


@pytest.mark.validates(
    "virgil.oidata.OIData",
    "virgil.likelihood.whitened_residuals",
    roots=["self-consistency"],
)
def test_rescaled_errors_carry_through_the_whitening():
    data = _four_telescopes()
    n_vis = data.vis.size
    once = onp.asarray(whitened_residuals(MODEL, data))[n_vis:]
    twice = onp.asarray(whitened_residuals(MODEL, data.with_error_scale(2.0)))[
        n_vis:
    ]
    assert onp.allclose(twice, 0.5 * once, rtol=1e-5)
    plain = model_loglike(MODEL, data)
    inflated = model_loglike(MODEL, data, phi_error=0.05)
    assert onp.isfinite(plain) and onp.isfinite(inflated)


@pytest.mark.validates(
    "virgil.oidata.OIData",
    "virgil.likelihood.whitened_residuals",
    roots=["self-consistency"],
)
def test_a_flagged_closure_phase_leaves_the_rest_independent():
    data = _four_telescopes()
    record = {
        "u": data.u,
        "v": data.v,
        "wavel": data.wavel,
        "vis": data.vis,
        "d_vis": data.d_vis,
        "phi": data.phi,
        "d_phi": data.d_phi,
        "i_cps1": data.i_cps1,
        "i_cps2": data.i_cps2,
        "i_cps3": data.i_cps3,
        "phi_flag": onp.r_[True, onp.zeros(data.phi.size - 1, dtype=bool)],
    }
    flagged = OIData(record)
    # 11 closure phases remain: 3 in the first frame (all independent), 4
    # in each of the others (3 independent each).
    assert flagged.phi.size == 11
    assert flagged.n_independent == flagged.vis.size + 9
    assert onp.all(onp.isfinite(whitened_residuals(MODEL, flagged)))


def test_gradients_are_finite():
    data = _four_telescopes()
    grad = jax.grad(lambda f: model_loglike(MODEL.set("flux", f), data))(0.08)
    assert onp.isfinite(grad)


@pytest.mark.validates(
    "virgil.oidata.OIData",
    "virgil.likelihood.whitened_residuals",
    roots=["self-consistency"],
)
def test_a_correlating_operator_is_rotated_to_independent_outputs():
    # Two overlapping differences of three phases share one input, so their
    # outputs correlate. Keeping only the diagonal would count that input
    # twice; the operator is rotated so the outputs are independent and
    # the χ² equals the dense one.
    sigma = onp.array([0.1, 0.2, 0.3])
    operator = onp.array([[1.0, -1.0, 0.0], [0.0, 1.0, -1.0]])
    phi = onp.array([0.3, -0.1, 0.2])
    data = OIData(
        {
            "u": onp.ones(3),
            "v": onp.zeros(3),
            "wavel": 1e-6,
            "vis": onp.ones(3),
            "d_vis": onp.full(3, 0.1),
            "phi": phi,
            "d_phi": sigma,
            "phi_mat": operator,
            "cp_flag": False,
        }
    )
    projected = onp.asarray(data.phi_mat) * sigma[None, :]
    assert onp.allclose(
        projected @ projected.T, onp.diag(onp.asarray(data.d_phi) ** 2)
    )
    cov = (operator * sigma) @ (operator * sigma).T
    y = operator @ phi
    chi2_dense = y @ onp.linalg.solve(cov, y)
    chi2 = onp.sum((onp.asarray(data.phi) / onp.asarray(data.d_phi)) ** 2)
    assert chi2 == pytest.approx(chi2_dense, rel=1e-6)


def _record(data, **changes):
    """``data`` as an OIData input record, with ``changes``."""
    record = {
        "u": data.u,
        "v": data.v,
        "wavel": data.wavel,
        "vis": data.vis,
        "d_vis": data.d_vis,
        "phi": data.phi,
        "d_phi": data.d_phi,
        "i_cps1": data.i_cps1,
        "i_cps2": data.i_cps2,
        "i_cps3": data.i_cps3,
    }
    return record | changes


@pytest.mark.validates(
    "virgil.likelihood.whitened_residuals", roots=["statistics"]
)
def test_unequal_errors_are_kept_and_simulation_matches_whitening():
    # σ² ∝ (1, 1, 3, 3): a minimum-norm split into baseline variances goes
    # negative here. The covariance keeps every reported variance, and
    # noise simulated from it whitens to unit variance.
    one = vlti_oidata(hour_angles_h=(0.0,), wavelengths_m=[3.5e-6])
    sigma = 0.05 * onp.sqrt(onp.array([1.0, 1.0, 3.0, 3.0]))
    data = OIData(_record(one, d_phi=sigma))
    corr = data.cp_noise.correlation(4)
    cov = sigma[:, None] * corr * sigma[None, :]
    assert onp.allclose(onp.diag(cov), sigma**2)
    assert onp.all(onp.linalg.eigvalsh(cov) > -1e-12)

    keys = jax.random.split(jax.random.PRNGKey(4), 4000)
    noise = onp.stack(
        [onp.asarray(data.cp_noise.sample(k, data.d_phi, 4)) for k in keys]
    )
    assert onp.allclose(noise.var(axis=0), sigma**2, rtol=0.1)
    whitened = onp.stack(
        [
            onp.asarray(data.cp_noise.whiten(np.asarray(n), data.d_phi)[0])
            for n in noise
        ]
    )
    assert onp.allclose(onp.cov(whitened.T), onp.eye(3), atol=0.1)


@pytest.mark.validates(
    "virgil.oidata.OIData",
    "virgil.likelihood.whitened_residuals",
    roots=["self-consistency"],
)
def test_a_phase_shifted_by_two_pi_gives_the_same_likelihood():
    data = _four_telescopes()
    shifted = eqx.tree_at(
        lambda d: d.phi, data, data.phi.at[0].add(2.0 * np.pi)
    )
    assert onp.allclose(
        whitened_residuals(MODEL, data),
        whitened_residuals(MODEL, shifted),
        atol=1e-4,
    )
    assert model_loglike(MODEL, data) == pytest.approx(
        model_loglike(MODEL, shifted), rel=1e-5
    )


@pytest.mark.validates(
    "virgil.oidata.OIData",
    "virgil.likelihood.whitened_residuals",
    roots=["self-consistency"],
)
def test_a_phase_operator_on_closure_phases_keeps_their_correlations():
    # An identity phi_mat on four-telescope closure phases must not count
    # four independent values per frame.
    data = _four_telescopes()
    projected = OIData(_record(data, phi_mat=onp.eye(data.phi.size)))
    assert projected.cp_noise is None
    assert projected.phi.size == 9  # three frames, three each


def test_tiny_but_identical_operator_rows_are_still_merged():
    data = OIData(
        {
            "u": onp.ones(3),
            "v": onp.zeros(3),
            "wavel": 1e-6,
            "vis": onp.ones(3),
            "d_vis": onp.full(3, 0.1),
            "phi": onp.zeros(3),
            "d_phi": onp.full(3, 0.1),
            "phi_mat": onp.array([[1e-6, 0.0, 0.0], [1e-6, 0.0, 0.0]]),
            "cp_flag": False,
        }
    )
    assert data.phi.size == 1


@pytest.mark.validates(
    "virgil.oidata.OIData",
    "virgil.likelihood.whitened_residuals",
    roots=["self-consistency"],
)
def test_a_zero_variance_closure_relation_is_dropped():
    # The four closure phases of one frame satisfy one closure relation:
    # an operator row along it has zero variance and is dropped, not kept
    # with an error of zero.
    one = vlti_oidata(hour_angles_h=(0.0,), wavelengths_m=[3.5e-6]).with_model(
        TRUTH
    )
    null = onp.linalg.svd(_incidence(one).T)[2][-1]  # the closure relation
    operator = onp.stack([null, onp.eye(4)[0]])
    projected = OIData(_record(one, phi_mat=operator))
    assert projected.phi.size == 1
    assert onp.all(onp.asarray(projected.d_phi) > 0.0)


def test_switching_x64_mode_keeps_the_index_dtypes():
    # JAX 0.10 caches the canonical (x64-dependent) copy of a NumPy array by
    # identity for as long as that copy is alive, whatever the mode. Index
    # arrays held as int64 and first used inside a 64-bit fit came back
    # int64 in a later 32-bit with_model, which failed in JAX's indexing.
    # Use the noise model in one mode, keep that computation alive, then
    # use it in the other.
    data = _four_telescopes()
    noise, key, n_phase = data.cp_noise, jax.random.PRNGKey(5), data.phi.size

    def use():
        sigma = np.asarray(onp.asarray(data.d_phi, dtype=float))
        residuals = noise.sample(key, sigma, n_phase)
        return (residuals,) + noise.whiten(residuals, sigma)

    for first, second in ((True, False), (False, True)):
        with jax.enable_x64(first):
            held = jax.jit(use)
            held()
            use()
        with jax.enable_x64(second):
            dtype = np.float64 if second else np.float32
            assert all(out.dtype == dtype for out in use())
            assert all(out.dtype == dtype for out in jax.jit(use)())
        del held


def _reproducer_data(stations=VLTI_UTS):
    """One epoch, one channel; the F11 reproducer (model at (3, 2, 0.9))."""
    return vlti_oidata(
        stations,
        declination_deg=-30,
        hour_angles_h=(0.0,),
        wavelengths_m=onp.array([2.2e-6]),
        sigma_v2=0.01,
        sigma_cp_deg=0.5,
    ).with_model(BinaryModelCartesian(3.0, 2.0, 0.9))


def _sweep(data):
    dras = np.linspace(-6.0, 6.0, 4001)

    @jax.jit
    def chi2(dra):
        model = BinaryModelCartesian(dra, -2.0, 0.9)
        return np.sum(whitened_residuals(model, data) ** 2)

    return onp.abs(onp.diff(onp.asarray(jax.vmap(chi2)(dras))))


@pytest.mark.validates(
    "virgil.likelihood.whitened_residuals", roots=["self-consistency"]
)
def test_correlated_closure_chi2_is_continuous_in_the_model():
    # virgil-validation F11. The old chord likelihood jumped by ~5e4 at four
    # places in this sweep (largest neighbour difference 1741 x the median);
    # the three-telescope sweep has 48 x, from steep but continuous fringes.
    four = _sweep(_reproducer_data())
    three = _sweep(_reproducer_data(onp.asarray(VLTI_UTS)[:3]))
    assert four.max() < 100.0 * onp.median(four)
    assert three.max() < 100.0 * onp.median(three)


@pytest.mark.validates(
    "virgil.likelihood.whitened_residuals", roots=["self-consistency"]
)
def test_correlated_closure_chi2_is_continuous_across_a_residual_of_pi():
    # Shift one closure phase of exact data by s (and another by 0.5, so that
    # there is a cross term) and sweep s through π: the
    # chord's sign flips there, which changes its cross term with the other, so the
    # old χ² jumped (a single or common shift would not show it);
    # the sine and the periodic penalty are smooth, and stationary at π.
    data = _reproducer_data()
    model = BinaryModelCartesian(3.0, 2.0, 0.9)
    shifts = np.linspace(np.pi - 0.02, np.pi + 0.02, 801)

    def chi2(shift):
        phi = data.phi.at[0].add(shift).at[1].add(0.5)
        shifted = eqx.tree_at(lambda d: d.phi, data, phi)
        return np.sum(whitened_residuals(model, shifted) ** 2)

    values = onp.asarray(jax.jit(jax.vmap(chi2))(shifts))
    assert onp.max(onp.abs(onp.diff(values))) < 1e-3 * values.max()
    # and no false minimum: a residual of π costs far more than none
    assert values.min() > 1e4
    assert float(chi2(0.0)) < 0.1 * values.min()


@pytest.mark.validates(
    "virgil.likelihood.whitened_residuals", roots=["self-consistency"]
)
def test_correlated_closure_chi2_matches_the_whitened_gaussian_near_zero():
    # For small residuals the new χ² is the old correlated Gaussian
    # (whitened Δ) up to O(Δ³) in the residuals, so relatively O(Δ²).
    with jax.enable_x64(True):
        data = _reproducer_data()
        model = BinaryModelCartesian(3.0, 2.0, 0.9)
        n_vis = data.vis.size
        sigma = onp.asarray(data.d_phi)
        rng = onp.random.default_rng(1)
        direction = rng.normal(size=sigma.size)
        relative = []
        for amplitude in (0.04, 0.02, 0.01):
            delta = amplitude * direction / onp.abs(direction).max()
            shifted = eqx.tree_at(
                lambda d: d.phi, data, np.asarray(data.phi) - delta
            )
            r = whitened_residuals(model, shifted)
            chi2_new = float(np.sum(r[n_vis:] ** 2))
            old, _ = data.cp_noise.whiten(np.asarray(delta), sigma)
            chi2_old = float(np.sum(old**2))
            relative.append(abs(chi2_new - chi2_old) / chi2_old)
        # relative difference shrinks like the square of the amplitude
        assert relative[0] < 0.01
        assert relative[1] < relative[0] / 3.0
        assert relative[2] < relative[1] / 3.0


@pytest.mark.validates(
    "virgil.likelihood.model_loglike", roots=["self-consistency"]
)
def test_gradients_match_finite_differences_with_residuals_near_pi():
    with jax.enable_x64(True):
        data = _reproducer_data()
        shifted = eqx.tree_at(
            lambda d: d.phi, data, np.asarray(data.phi) + np.pi - 0.003
        )

        def loglike(x):
            return model_loglike(
                BinaryModelCartesian(x[0], x[1], x[2]), shifted
            )

        x0 = np.array([3.0, 2.0, 0.9])
        grad = onp.asarray(jax.grad(loglike)(x0))
        eps = 1e-6
        fd = onp.array(
            [
                (loglike(x0.at[i].add(eps)) - loglike(x0.at[i].add(-eps)))
                / (2 * eps)
                for i in range(3)
            ]
        )
        assert onp.allclose(
            grad, fd, rtol=1e-5, atol=1e-8 * onp.abs(grad).max()
        )
