"""Angles sampled as 2-D vectors (design/orbit_prior_art.md §4.1).

Tests 1–4 and 6 of §4.1 are here; test 5, the orbit orientation, is in
``tests/test_orbits.py``.
"""

import jax
import jax.numpy as np
import numpy as onp
import numpyro.distributions as dist
import pytest
from scipy import special, stats

from virgil.angles import AngleVector, vector_angle
from virgil.fitting import fit, gauss_newton_mass
from virgil.likelihood import numpyro_model


def _chord(centre_deg, sigma_rad, path="pa"):
    """A one-datum phase likelihood on the angle at ``path``: the chord
    2 sin(Δ/2)/σ of virgil's phase residuals, a von Mises with κ = 1/σ²."""

    def residuals(values):
        delta = np.deg2rad(values[path] - centre_deg)
        return np.atleast_1d(2.0 * np.sin(0.5 * delta) / sigma_rad)

    return residuals


def _nothing(**kwargs):
    return None


def _polar_grid(n_r=3000, n_theta=720, r_max=3.0):
    r = onp.linspace(0.0, r_max, n_r + 1)[1:]
    theta = onp.linspace(0.0, 2 * onp.pi, n_theta, endpoint=False)
    grid_r, grid_theta = onp.meshgrid(r, theta, indexing="ij")
    v = onp.stack(
        [grid_r * onp.cos(grid_theta), grid_r * onp.sin(grid_theta)], -1
    )
    return r, theta, grid_r, v


def test_the_ring_prior_leaves_the_angle_uniform():
    # §4.1 test 1: KS test of the angle, and the radius independent of it.
    draws = onp.asarray(AngleVector().sample(jax.random.PRNGKey(1), (4000,)))
    theta = onp.asarray(vector_angle(draws))
    assert theta.min() >= 0.0 and theta.max() < 360.0
    assert stats.kstest(theta / 360.0, "uniform").pvalue > 0.01
    r = onp.hypot(draws[:, 0], draws[:, 1])
    assert stats.spearmanr(theta, r).pvalue > 0.01
    # Its density is rotationally symmetric, so the angle is exactly
    # uniform whatever the ring width.
    v = onp.array([[1.2, 0.3], [-0.3, 1.2], [0.0, -onp.sqrt(1.53)]])
    log_p = onp.asarray(AngleVector(ring_width=0.4).log_prob(v))
    assert log_p == pytest.approx(log_p[0], abs=1e-5)


@pytest.mark.parametrize(
    "prior, density",
    [
        (AngleVector(), lambda t: onp.full_like(t, 1 / (2 * onp.pi))),
        (
            AngleVector(-40.0, 3.0),
            lambda t: onp.exp(3.0 * onp.cos(t + onp.deg2rad(40.0)))
            / (2 * onp.pi * special.i0(3.0)),
        ),
        (
            AngleVector(100.0, 2.5, axial=True),
            lambda t: onp.exp(2.5 * onp.cos(2 * (t - onp.deg2rad(100.0))))
            / (2 * onp.pi * special.i0(2.5)),
        ),
    ],
    ids=["uniform", "von-mises", "axial"],
)
def test_chords_give_the_right_marginals_and_a_normalised_density(
    prior, density
):
    # §4.1 test 2: integrate the vector density over r on a polar grid;
    # the angle's marginal is the von Mises (or axial von Mises) density,
    # with the normaliser on the circle, and the whole is normalised.
    r, theta, grid_r, v = _polar_grid()
    with jax.enable_x64(True):
        log_p = onp.asarray(prior.log_prob(np.asarray(v)))
    marginal = (onp.exp(log_p) * grid_r).sum(0) * (r[1] - r[0])
    onp.testing.assert_allclose(marginal, density(theta), rtol=1e-6)
    assert marginal.sum() * (theta[1] - theta[0]) == pytest.approx(1.0, 1e-6)
    # Draws follow it too: first and second circular moments.
    draws = prior.sample(jax.random.PRNGKey(2), (20000,))
    angle = onp.deg2rad(onp.asarray(vector_angle(draws)))
    for harmonic in (1, 2):
        expected = onp.sum(density(theta) * onp.exp(1j * harmonic * theta)) * (
            theta[1] - theta[0]
        )
        measured = onp.mean(onp.exp(1j * harmonic * angle))
        assert abs(measured - expected) < 0.02


@pytest.mark.parametrize(
    "prior",
    [
        AngleVector(),
        AngleVector(30.0, 5.0),
        AngleVector(30.0, 5.0, axial=True),
    ],
    ids=["uniform", "von-mises", "axial"],
)
def test_residuals_are_the_negative_log_density(prior):
    # Half the sum of the squared residuals is -log p + a constant, so
    # Levenberg–Marquardt and L-BFGS fit the same posterior.
    v = onp.random.default_rng(0).normal(size=(50, 2))
    with jax.enable_x64(True):
        half_chi2 = onp.array(
            [0.5 * onp.sum(onp.asarray(prior.residuals(x)) ** 2) for x in v]
        )
        log_p = onp.asarray(prior.log_prob(np.asarray(v)))
    total = half_chi2 + log_p
    onp.testing.assert_allclose(total, total[0], atol=1e-5)  # float32 μ, κ


def test_bad_arguments_are_rejected():
    with pytest.raises(ValueError, match="both mean and kappa"):
        AngleVector(mean=10.0)
    with pytest.raises(ValueError, match="axial"):
        AngleVector(axial=True)
    with pytest.raises(ValueError, match="non-negative"):
        AngleVector(0.0, -1.0)


def test_nuts_samples_a_posterior_straddling_the_wrap():
    # §4.1 test 3: a von Mises posterior centred on 0°, with half its
    # mass on each side of the wrap. Tiny: one angle, 1000 draws.
    from numpyro.infer import MCMC, NUTS

    kappa = 4.0
    model = numpyro_model(
        _nothing,
        {"pa": AngleVector()},
        (),
        likelihoods=[_chord(0.0, 1 / onp.sqrt(kappa))],
    )
    mcmc = MCMC(
        NUTS(model), num_warmup=500, num_samples=1000, progress_bar=False
    )
    mcmc.run(jax.random.PRNGKey(0))
    samples = mcmc.get_samples()
    assert samples["pa_vec"].shape == (1000, 2)
    angle = onp.deg2rad(onp.asarray(samples["pa"]))
    resultant = onp.mean(onp.exp(1j * angle))
    # The truth: mean 0, mean resultant length I1(κ)/I0(κ).
    assert abs(onp.rad2deg(onp.angle(resultant))) < 6.0
    assert abs(resultant) == pytest.approx(
        special.i1(kappa) / special.i0(kappa), abs=0.05
    )
    assert 0.3 < onp.mean(angle > onp.pi) < 0.7  # both sides of the wrap
    # The radius stays on the ring, far from the origin.
    radius = onp.hypot(*onp.asarray(samples["pa_vec"]).T)
    assert radius.min() > 0.2


@pytest.mark.parametrize("start", [300.0, 200.0, 40.0])
def test_a_map_fit_converges_through_the_wrap(start):
    # §4.1 test 4: the data say 20°, the von Mises prior says 340°, and
    # the fit starts on either side. The MAP of κ_d cos(θ - 20°) +
    # κ_p cos(θ - 340°) is the direction of κ_d m̂_d + κ_p m̂_p.
    sigma, kappa_prior = 0.1, 50.0
    result = fit(
        _nothing,
        {"pa": AngleVector(340.0, kappa_prior)},
        (),
        init={"pa": start},
        likelihoods=[_chord(20.0, sigma)],
    )
    assert result.info["method"] == "lm"
    assert result.info["converged"]
    total = sigma**-2 * onp.exp(
        1j * onp.deg2rad(20.0)
    ) + kappa_prior * onp.exp(1j * onp.deg2rad(340.0))
    expected = onp.mod(onp.rad2deg(onp.angle(total)), 360.0)
    assert float(result.values["pa"]) == pytest.approx(expected, abs=1e-3)
    radius = onp.hypot(*onp.asarray(result.values["pa_vec"]))
    assert radius == pytest.approx(1.0, abs=1e-6)  # the ring's mode


def test_a_template_path_and_the_gauss_newton_mass():
    # Any angle path works, here a binary's position angle fitted to
    # simulated closure phases and V² from 355° to a truth at 5°.
    from tests._test_data import oidata
    from virgil.models import BinaryModelAngular
    from virgil.simulate import simulate

    truth = BinaryModelAngular(150.0, 5.0, 0.05)
    data = simulate(truth, oidata)
    template = BinaryModelAngular(150.0, 355.0, 0.05)
    priors = {"pa": AngleVector()}
    result = fit(template, priors, data)
    assert float(result.values["pa"]) == pytest.approx(5.0, abs=1e-3)
    assert float(result.model.pa) == pytest.approx(5.0, abs=1e-3)
    mass = gauss_newton_mass(template, priors, data, result.values)
    covariance = mass["inverse_mass_matrix"][("pa_vec",)]
    assert mass["dense_mass"] == [("pa_vec",)]
    assert covariance.shape == (2, 2)
    # Radially the ring's width; tangentially the data's.
    radial = onp.array([onp.cos(onp.deg2rad(5.0)), onp.sin(onp.deg2rad(5.0))])
    assert radial @ covariance @ radial == pytest.approx(0.25**2, rel=1e-3)
    assert onp.all(onp.linalg.eigvalsh(covariance) > 0)


def test_laplace_evidence_in_vector_space_matches_the_angle_space_evidence():
    # §4.1 test 6: one angle, a chord likelihood and a von Mises prior on
    # either side of the wrap. The evidence by quadrature in θ, against a
    # Laplace approximation in v = (x, y), which is the angle's Laplace
    # approximation times a Gaussian integral over r: the ring integrates
    # out exactly (to exp(-1/2s²)), and only the Laplace error in θ is left.
    from numpyro.infer.util import log_density

    sigma, mean, kappa = 0.1, 350.0, 10.0
    likelihood = _chord(5.0, sigma)
    prior = AngleVector(mean, kappa)
    model = numpyro_model(
        _nothing, {"pa": prior}, (), likelihoods=[likelihood]
    )
    result = fit(
        _nothing, {"pa": prior}, (), init={"pa": 0.0}, likelihoods=[likelihood]
    )
    with jax.enable_x64(True):

        def log_joint(v):
            return log_density(model, (), {}, {"pa_vec": v})[0]

        v_map = np.asarray(result.values["pa_vec"], dtype=np.float64)
        hessian = -jax.hessian(log_joint)(v_map)
        laplace_v = (
            log_joint(v_map)
            + np.log(2 * np.pi)
            - 0.5 * np.linalg.slogdet(hessian)[1]
        )

        def log_joint_angle(t):  # θ in radians
            delta = t - np.deg2rad(5.0)
            return (
                -0.5 * (2 * np.sin(0.5 * delta) / sigma) ** 2
                + kappa * (np.cos(t - np.deg2rad(mean)) - 1)
                - np.log(2 * np.pi * special.i0e(kappa))
            )

        t = onp.linspace(-onp.pi, onp.pi, 200001)
        exact = onp.log(
            onp.sum(onp.exp(onp.asarray(log_joint_angle(t)))) * (t[1] - t[0])
        )
        t_map = np.deg2rad(float(result.values["pa"]))
        curvature = -jax.grad(jax.grad(log_joint_angle))(t_map)
        laplace_angle = (
            log_joint_angle(t_map)
            + 0.5 * np.log(2 * np.pi)
            - 0.5 * np.log(curvature)
        )
    assert float(laplace_v) == pytest.approx(float(laplace_angle), abs=1e-5)
    assert float(laplace_v) == pytest.approx(exact, abs=2e-3)


def test_angle_vectors_mix_with_other_priors_in_numpyro():
    # The deterministic angle and an ordinary parameter, at one point.
    from numpyro.infer.util import log_density

    priors = {"pa": AngleVector(), "flux": dist.Uniform(0.0, 1.0)}
    seen = {}

    def model_fn(**values):
        seen.update(values)

    model = numpyro_model(model_fn, priors, ())
    value, trace = log_density(
        model, (), {}, {"pa_vec": np.array([0.0, -2.0]), "flux": 0.5}
    )
    assert float(trace["pa"]["value"]) == pytest.approx(270.0)
    assert float(seen["pa"]) == pytest.approx(270.0)
    assert float(value) == pytest.approx(
        float(AngleVector().log_prob(np.array([0.0, -2.0]))), rel=1e-6
    )
