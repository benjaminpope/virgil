import jax
import jax.numpy as np
import jax.scipy as jsp
import numpy as onp
import pytest

import virgil.models as models
from virgil.inference import (
    fisher,
    laplace_cov,
    laplace_covariance,
    laplace_parameter_uncertainty,
)
from virgil.likelihood import (
    joint_data,
    joint_errors,
    joint_loglike,
    joint_prediction,
    loglike,
    model_loglike,
)
from virgil.models import (
    BinaryModelAngular,
    BinaryModelCartesian,
    cvis_binary,
)
from virgil.oidata import OIData, closure_phases

from tests._compiles import count_compiles
from tests._test_data import i_cps1, i_cps2, i_cps3, oidata, u, v


dra, ddec, flux = 0.2, 0.1, 10


def test_models_no_longer_reexport_data_helpers():
    assert not hasattr(models, "OIData")
    assert not hasattr(models, "closure_phases")


def test_cvis_binary():
    vis = cvis_binary(u, v, dra, ddec, flux)
    vis2 = np.abs(vis) ** 2
    assert vis.shape == (u.shape[0],)
    assert np.all(vis2 >= 0.0)
    assert np.all(vis2 <= 1.0)
    assert np.all(np.isfinite(vis))


def test_cvis_binary_argument_order_matches_other_cvis_functions():
    # dra before ddec, as in cvis_uniform_disk.
    uu, vv = u / oidata.wavel, v / oidata.wavel
    assert np.allclose(
        cvis_binary(uu, vv, dra=150.0, ddec=-40.0, flux=1e-2),
        BinaryModelCartesian(dra=150.0, ddec=-40.0, flux=1e-2).model(
            u, v, oidata.wavel
        ),
        # Phases reach ~1e4 rad here, so float32 inputs limit agreement.
        atol=1e-3,
    )
    east_only = cvis_binary(np.array([1e7]), np.array([0.0]), 100.0, 0.0, 1.0)
    north_only = cvis_binary(np.array([1e7]), np.array([0.0]), 0.0, 100.0, 1.0)
    # A u-only baseline sees an East offset but not a North one.
    assert not np.allclose(east_only, 1.0)
    assert np.allclose(north_only, 1.0)


def test_closure_phases():
    vis = cvis_binary(u, v, dra, ddec, flux)
    cps = closure_phases(vis, i_cps1, i_cps2, i_cps3)
    assert cps.shape == (35,)
    assert np.all(np.isfinite(cps))


def test_likelihood():
    binary = BinaryModelAngular(50, 45, 0.1)
    model_data = oidata.model(binary)
    data, errors = oidata.flatten_data()

    like = jsp.stats.norm.logpdf(model_data, loc=data, scale=errors).sum()
    assert np.all(np.isfinite(like))


def test_BinaryModelAngular():
    binary = BinaryModelAngular(50, 45, 0.1)
    model_data = oidata.model(binary)
    assert model_data.shape[0] == len(oidata.vis) + len(oidata.phi)
    assert np.all(np.isfinite(model_data))


def test_BinaryModelCartesian():
    binary = BinaryModelCartesian(150, 150, 1e-3)
    model_data = oidata.model(binary)
    assert model_data.shape[0] == len(oidata.vis) + len(oidata.phi)
    assert np.all(np.isfinite(model_data))


def test_BinaryModelAngular_to_cartesian_preserves_model():
    angular = BinaryModelAngular(50.0, 45.0, 0.1)
    cartesian = angular.to_cartesian()

    assert isinstance(cartesian, BinaryModelCartesian)
    assert np.allclose(
        angular.model(oidata.u, oidata.v, oidata.wavel),
        cartesian.model(oidata.u, oidata.v, oidata.wavel),
    )


def test_BinaryModelCartesian_to_angular_roundtrip():
    cartesian = BinaryModelCartesian(150.0, -120.0, 1e-3)
    angular = cartesian.to_angular()
    roundtrip = angular.to_cartesian()

    assert isinstance(angular, BinaryModelAngular)
    assert np.allclose(roundtrip.dra, cartesian.dra)
    assert np.allclose(roundtrip.ddec, cartesian.ddec)
    assert np.allclose(roundtrip.flux, cartesian.flux)


@pytest.mark.parametrize(
    ("pa", "expected_index"),
    [
        # pa=0deg -> North (+y, low row index).
        (0.0, (0, 2)),
        # pa=90deg -> East (+x, low column index).
        (90.0, (2, 0)),
    ],
)
def test_binary_model_angular_render_follows_north_to_east_pa_convention(
    pa, expected_index
):
    image = BinaryModelAngular(sep=4.0, pa=pa, flux=1e6).render(
        npix=5, fov_mas=10.0
    )
    assert (
        onp.unravel_index(onp.asarray(image).argmax(), image.shape)
        == expected_index
    )


def test_binary_model_angular_matches_cartesian_at_east_position_angle():
    """A companion placed at pa=90deg (East) must match the
    already-validated BinaryModelCartesian placed at dra=+sep, ddec=0.
    Comparing against an independently constructed BinaryModelCartesian
    (rather than angular.to_cartesian()) is what actually catches a
    sign error in the angular parameterization: a self-consistency
    round-trip through to_cartesian()/to_angular() alone would stay
    internally consistent even if both used the same wrong sign.
    """
    sep, flux = 40.0, 0.2
    angular = BinaryModelAngular(sep=sep, pa=90.0, flux=flux)
    cartesian = BinaryModelCartesian(dra=sep, ddec=0.0, flux=flux)

    assert np.allclose(
        angular.model(oidata.u, oidata.v, oidata.wavel),
        cartesian.model(oidata.u, oidata.v, oidata.wavel),
    )


def _reference_logpdf(data_obj, prediction, reference):
    """Gaussian log density of ``prediction`` about ``reference``, by hand.

    Visibilities are independent; closure phases enter as sin Δ,
    whitened as correlated groups (see test_closure), plus the periodic
    penalty 2 sin²(Δ/2)/σ per closure phase, which has no normalisation.
    """
    _, errors = data_obj.flatten_data()
    n_vis = data_obj.vis.size
    resid = np.asarray(prediction) - np.asarray(reference)
    vis = jsp.stats.norm.logpdf(resid[:n_vis], scale=errors[:n_vis]).sum()
    sigma = errors[n_vis:]
    whitened, phase_errors = data_obj.cp_noise.whiten(
        np.sin(resid[n_vis:]), sigma
    )
    penalty = 2.0 * np.sin(0.5 * resid[n_vis:]) ** 2 / sigma
    return (
        vis
        + jsp.stats.norm.logpdf(whitened).sum()
        - np.sum(np.log(phase_errors))
        - 0.5 * np.sum(penalty**2)
    )


def test_laplace_and_fisher_wrappers_are_finite():
    params = ["dra", "ddec", "flux"]
    values = np.array([120.0, -80.0, 2e-3])
    param_dict = dict(zip(params, values))
    model_data = oidata.model(BinaryModelCartesian(**param_dict))
    data, errors = oidata.flatten_data()

    cov = laplace_cov(values, params, oidata, BinaryModelCartesian)
    fmat = fisher(values, params, oidata, BinaryModelCartesian, ridge=1e-10)
    like = loglike(values, params, oidata, BinaryModelCartesian)
    expected_like = _reference_logpdf(oidata, model_data, data)

    assert cov.shape == (3, 3)
    assert fmat.shape == (3, 3)
    assert np.all(np.isfinite(cov))
    assert np.all(np.isfinite(fmat))
    assert np.allclose(fmat, fmat.T)
    assert np.isfinite(like)
    assert np.allclose(like, expected_like)


@pytest.mark.slow
def test_laplace_wrappers_match_closures_and_compile_once():
    # The model-level curvatures must agree with the generic, closure-based
    # helpers. They are jitted once at module level, so a call at new values
    # neither recompiles (as a jit defined inside each call would) nor runs
    # the model in Python again (as eager code would, ~200x slower).
    params = ["dra", "ddec", "flux"]
    values = np.array([120.0, -80.0, 2e-3])
    calls = []

    def binary(dra, ddec, flux):
        calls.append(1)
        return BinaryModelCartesian(dra, ddec, flux)

    args = (params, oidata, binary)
    objective = lambda x: -loglike(x, *args)

    cov = laplace_cov(values, *args)
    assert np.allclose(cov, laplace_covariance(objective, values), rtol=1e-4)
    hess = fisher(values, *args)
    sigma = laplace_parameter_uncertainty(values, *args, "flux")
    assert np.allclose(sigma, hess[2, 2] ** -0.5, rtol=1e-4)

    moved = values * 1.01
    calls.clear()
    with count_compiles() as compiles:
        laplace_cov(moved, *args)
        fisher(moved, *args)
        laplace_parameter_uncertainty(moved, *args, "flux")
    assert not compiles
    assert not calls


def test_laplace_wrappers_need_a_1d_parameter_vector():
    with pytest.raises(ValueError, match="1D parameter vector"):
        laplace_cov(
            np.array([[120.0], [-80.0], [2e-3]]),
            ["dra", "ddec", "flux"],
            oidata,
            BinaryModelCartesian,
        )


def test_model_and_joint_loglike_helpers_match_legacy_loglike():
    values = np.array([120.0, -80.0, 2e-3])
    params = ["dra", "ddec", "flux"]
    model = BinaryModelCartesian(*values)
    observations = (oidata, oidata)
    tree = {"dra": values[0], "ddec": values[1], "flux": values[2:]}
    model_fn = lambda values, index: BinaryModelCartesian(
        values["dra"], values["ddec"], values["flux"][index]
    )

    legacy = loglike(values, params, oidata, BinaryModelCartesian)
    assert np.allclose(model_loglike(model, oidata), legacy)
    assert np.allclose(joint_loglike(tree, observations, model_fn), 2 * legacy)
    assert joint_data(observations).shape[0] == 2 * (
        oidata.vis.size + oidata.phi.size
    )
    assert joint_errors(observations).shape == joint_data(observations).shape
    assert joint_prediction(tree, observations, model_fn).shape[0] == 2 * (
        oidata.vis.size + oidata.phi.size
    )


def test_oidata_with_model_preserves_structure_and_seeded_noise():
    model = BinaryModelCartesian(120.0, -80.0, 2e-3)
    noiseless = oidata.with_model(model)
    noisy_a = oidata.with_model(model, key=jax.random.key(12))
    noisy_b = oidata.with_model(model, key=jax.random.key(12))

    assert np.allclose(noiseless.flatten_data()[0], oidata.model(model))
    assert np.allclose(noisy_a.vis, noisy_b.vis)
    assert np.allclose(noisy_a.phi, noisy_b.phi)
    assert np.allclose(noiseless.u, oidata.u)
    assert np.allclose(noiseless.d_vis, oidata.d_vis)
    assert noiseless.cp_flag == oidata.cp_flag
    assert noiseless.vis_mode == oidata.vis_mode

    with pytest.raises(ValueError, match="non-negative"):
        oidata.with_model(model, noise_scale=-1.0)


def test_oidata_linear_observables_transform_and_model_alignment():
    cvis = cvis_binary(
        oidata.u / oidata.wavel, oidata.v / oidata.wavel, 50.0, 0.0, 1e-3
    )
    n = cvis.shape[0]
    m_vis = 12
    m_phi = 10
    vis_mat = np.eye(n)[:m_vis, :]
    phi_mat = np.eye(n)[:m_phi, :]

    sim_data = {
        "u": oidata.u,
        "v": oidata.v,
        "wavel": oidata.wavel,
        "vis": np.abs(cvis),
        "d_vis": np.ones_like(np.abs(cvis)) * 1e-3,
        "phi": np.angle(cvis),
        "d_phi": np.ones_like(np.angle(cvis)) * 0.1,
        "v2_flag": False,
        "cp_flag": False,
        "vis_mode": "logamp",
        "disco_vis_mat": vis_mat,
        "disco_phi_mat": phi_mat,
    }

    disco_data = OIData(sim_data)
    flattened_data, errors = disco_data.flatten_data()
    model_vector = disco_data.model(BinaryModelCartesian(50.0, 0.0, 1e-3))
    cvis_model = BinaryModelCartesian(50.0, 0.0, 1e-3).model(
        disco_data.u, disco_data.v, disco_data.wavel
    )

    assert disco_data.vis.shape == (m_vis,)
    assert disco_data.phi.shape == (m_phi,)
    assert disco_data.observable_kind == "split"
    assert flattened_data.shape == (m_vis + m_phi,)
    assert np.allclose(model_vector, disco_data.standardize_model(cvis_model))
    assert flattened_data.shape == model_vector.shape
    assert errors.shape == flattened_data.shape
    assert np.all(np.isfinite(flattened_data))
    assert np.all(np.isfinite(model_vector))


def test_oidata_linear_observables_invalid_shape_raises():
    n = oidata.u.shape[0]
    bad_data = {
        "u": oidata.u,
        "v": oidata.v,
        "wavel": oidata.wavel,
        "vis": np.ones(n),
        "d_vis": np.ones(n) * 1e-3,
        "phi": np.zeros(n),
        "d_phi": np.ones(n) * 0.1,
        "v2_flag": False,
        "cp_flag": False,
        "disco_vis_mat": np.ones((5, 7)),
    }

    raised = False
    try:
        OIData(bad_data)
    except ValueError:
        raised = True
    assert raised


def test_disco_operator_requires_diagonal_propagated_covariance():
    base = {
        "u": np.zeros(2),
        "v": np.zeros(2),
        "wavel": np.array([1e-6]),
        "vis": np.ones(2),
        "d_vis": np.ones(2),
        "phi": np.zeros(2),
        "d_phi": np.ones(2),
        "v2_flag": False,
        "cp_flag": False,
    }
    orthogonal = np.array([[1.0, 1.0], [1.0, -1.0]]) / np.sqrt(2.0)
    independent = OIData({**base, "disco_vis_mat": orthogonal})

    assert np.allclose(independent.d_vis, np.ones(2))
    with pytest.raises(ValueError, match="diagonal propagated covariance"):
        OIData(
            {
                **base,
                "disco_vis_mat": np.array([[1.0, 0.0], [1.0, 1.0]]),
            }
        )

    correlated_generic = OIData(
        {**base, "vis_mat": np.array([[1.0, 0.0], [1.0, 1.0]])}
    )
    assert correlated_generic.vis.shape == (2,)
