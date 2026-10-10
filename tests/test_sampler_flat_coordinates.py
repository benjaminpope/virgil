"""numpyro_model(flat_coordinates=True) samples each prior in its flat coordinate, as fit does.

Off by default: SBC showed no efficiency gain (virgil#301).

A LogUniform scale is sampled through log x, an isotropic inclination
through cos i, an isotropic latitude through sin(lat)
(design/sampler_flat_coordinates.md). Only the unconstrained coordinate
NUTS moves in changes: the sites keep their names, values and densities.
Everything here is tiny: scalar priors, short NUTS runs.
"""

import jax
import jax.numpy as np
import numpy as onp
import numpyro
import numpyro.distributions as dist
import pytest
from numpyro.distributions.transforms import biject_to
from numpyro.infer.initialization import init_to_value
from numpyro.infer.util import initialize_model, log_density, potential_energy

from virgil._flat import _FlatBijection, _flat_coordinate, flat_sampled
from virgil.fitting import _Objective
from virgil.likelihood import chain_init_params, numpyro_model
from virgil.priors import IsotropicInclination, IsotropicLatitude

from .test_no_data_priors import _assert_marginal, _nuts

FLAT_PRIORS = {
    "scale": dist.LogUniform(1e-4, 1.0),
    "inc": IsotropicInclination(),
    "pole": IsotropicInclination(0.0, 20.0),
    "lat": IsotropicLatitude(),
}


def _no_model(**_):
    return None


@pytest.mark.parametrize("x64", [False, True])
@pytest.mark.parametrize("name", list(FLAT_PRIORS))
def test_flat_transform_round_trips(name, x64):
    with jax.enable_x64(x64):
        prior = FLAT_PRIORS[name]
        transform = biject_to(flat_sampled(prior).support)
        z = np.linspace(-8.0, 8.0, 33)
        x = transform(z)
        assert bool(np.all(prior.support(x)))
        tol = 1e-9 if x64 else 2e-3
        onp.testing.assert_allclose(transform.inv(x), z, atol=tol)
        # The same map as fit's (one implementation, _FlatBijection).
        bijection = _FlatBijection(*_flat_coordinate(prior))
        onp.testing.assert_array_equal(x, bijection(z))


@pytest.mark.parametrize("x64", [False, True])
@pytest.mark.parametrize("name", list(FLAT_PRIORS))
def test_flat_transform_jacobian_matches_autodiff(name, x64):
    with jax.enable_x64(x64):
        transform = biject_to(flat_sampled(FLAT_PRIORS[name]).support)
        z = np.linspace(-6.0, 6.0, 13)
        x = transform(z)
        slope = jax.vmap(jax.grad(transform))(z)
        expected = np.log(np.abs(slope))
        got = transform.log_abs_det_jacobian(z, x)
        onp.testing.assert_allclose(got, expected, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("x64", [False, True])
def test_potential_of_a_flat_prior_is_logistic(x64):
    """In the flat coordinate the prior times the Jacobian is exactly the
    logistic density log σ(z) + log σ(-z), with finite gradients out to
    |z| = 24, also in float32 and at the poles of the isotropic priors."""
    with jax.enable_x64(x64):
        model = numpyro_model(
            _no_model, FLAT_PRIORS, (), flat_coordinates=True
        )
        for z0 in (-24.0, -18.0, -3.0, 0.0, 2.5, 18.0, 24.0):
            z = {k: np.asarray(z0) for k in FLAT_PRIORS}
            potential, grad = jax.value_and_grad(
                lambda w: potential_energy(model, (), {}, w)
            )(z)
            logistic = float(jax.nn.log_sigmoid(z0) + jax.nn.log_sigmoid(-z0))
            expected = -len(FLAT_PRIORS) * logistic
            assert float(potential) == pytest.approx(expected, rel=1e-3)
            for site, g in grad.items():
                assert float(g) == pytest.approx(
                    float(np.tanh(z0 / 2)), abs=2e-3
                ), (site, z0)


def test_sites_values_and_density_are_unchanged():
    """Flat or not, the trace has the same sites, the values are the
    parameters, and log_density is the same."""
    noise = {"vis_scale": dist.LogUniform(0.1, 10.0)}
    priors = FLAT_PRIORS | {"u": dist.Uniform(-1.0, 1.0)}
    models = [
        numpyro_model(
            _no_model, priors, (), noise=noise, flat_coordinates=flat
        )
        for flat in (True, False)
    ]
    traces = [
        numpyro.handlers.trace(numpyro.handlers.seed(m, 0)).get_trace()
        for m in models
    ]
    assert (
        set(traces[0]) == set(traces[1]) == set(priors) | {"noise.vis_scale"}
    )
    values = {k: v["value"] for k, v in traces[0].items()}
    for site, prior in (
        priors | {"noise.vis_scale": noise["vis_scale"]}
    ).items():
        assert bool(prior.support(values[site]))
    densities = [float(log_density(m, (), {}, values)[0]) for m in models]
    assert densities[0] == pytest.approx(densities[1], rel=1e-6)


def test_init_to_value_and_chain_init_params_use_the_flat_coordinate():
    model = numpyro_model(_no_model, FLAT_PRIORS, (), flat_coordinates=True)
    start = {"scale": 1e-2, "inc": 30.0, "pole": 5.0, "lat": 0.3}
    info = initialize_model(
        jax.random.PRNGKey(0), model, init_strategy=init_to_value(values=start)
    )
    # log 1e-2 is halfway from log 1e-4 to log 1: the logit is 0.
    assert float(info.param_info.z["scale"]) == pytest.approx(0.0, abs=1e-5)
    z = chain_init_params(model, [start, start])
    for site, prior in FLAT_PRIORS.items():
        expected = _FlatBijection(*_flat_coordinate(prior)).inv(
            np.asarray(start[site])
        )
        onp.testing.assert_allclose(z[site], [expected] * 2, rtol=1e-5)


def test_coordinates_match_fit():
    """numpyro's unconstrained coordinates are fit's (flat) ones, and the
    opt-out restores numpyro's bijection of the support, as gauss_newton_mass
    takes them."""
    start = {"scale": 1e-2, "inc": 30.0, "pole": 5.0, "lat": 0.3}
    for flat in (True, False):
        model = numpyro_model(
            _no_model, FLAT_PRIORS, (), flat_coordinates=flat
        )
        info = initialize_model(
            jax.random.PRNGKey(0),
            model,
            init_strategy=init_to_value(values=start),
        )
        problem = _Objective(
            _no_model, FLAT_PRIORS, (), flat=flat, likelihoods=()
        )
        z = problem.init(start)
        for site in FLAT_PRIORS:
            assert float(info.param_info.z[site]) == pytest.approx(
                float(z[site]), rel=1e-5, abs=1e-6
            )


def test_expanded_and_independent_priors_are_sampled_flat():
    prior = dist.LogUniform(1e-3, 1.0).expand([3]).to_event(1)
    model = numpyro_model(_no_model, {"v": prior}, (), flat_coordinates=True)
    trace = numpyro.handlers.trace(numpyro.handlers.seed(model, 0)).get_trace()
    assert trace["v"]["value"].shape == (3,)
    z = {"v": np.zeros(3)}
    expected = -3 * float(jax.nn.log_sigmoid(0.0) * 2)
    assert float(potential_energy(model, (), {}, z)) == pytest.approx(
        expected, rel=1e-5
    )


def test_other_priors_are_not_wrapped():
    for prior in (
        dist.Uniform(0.0, 1.0),
        dist.Normal(0.0, 1.0),
        dist.HalfNormal(1.0),
    ):
        assert flat_sampled(prior) is prior


@pytest.mark.parametrize("x64", [False, True])
def test_no_data_nuts_reproduces_the_flat_priors(x64):
    """With no data, NUTS in the flat coordinates returns each prior (KS on
    ESS-thinned draws), parameters and an error term alike."""
    noise = {"vis_scale": dist.LogUniform(0.1, 10.0)}
    with jax.enable_x64(x64):
        model = numpyro_model(
            _no_model, FLAT_PRIORS, (), noise=noise, flat_coordinates=True
        )
        samples = _nuts(model, num_samples=1500, num_warmup=400)
    for site, prior in FLAT_PRIORS.items():
        _assert_marginal(samples, site, prior)
    _assert_marginal(samples, "noise.vis_scale", noise["vis_scale"])


def test_flat_coordinates_are_off_by_default():
    """The default is numpyro's bijection of each support (virgil#301)."""
    default = numpyro_model(_no_model, FLAT_PRIORS, ())
    off = numpyro_model(_no_model, FLAT_PRIORS, (), flat_coordinates=False)
    z = {k: np.asarray(0.3) for k in FLAT_PRIORS}
    assert float(potential_energy(default, (), {}, z)) == pytest.approx(
        float(potential_energy(off, (), {}, z)), rel=1e-6
    )
    assert _Objective(_no_model, FLAT_PRIORS, (), flat=False, likelihoods=())
