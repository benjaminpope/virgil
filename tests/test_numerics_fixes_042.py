"""Small numerical fixes of the 0.4.2 audit (virgil#707, virgil#712)."""

import jax
import jax.numpy as np
import numpy as onp
import numpyro.distributions as dist
import pytest

from virgil import fitting, grid_fit
from virgil.fitting import NoLeastSquaresForm, _Objective, kkt_report
from virgil.limits import ruffio_upperlimit
from virgil.likelihood import build_model, whitened_residuals
from virgil.models import BinaryModelCartesian
from virgil.orbits import KeplerOrbit, ThieleInnesOrbit
from tests._test_data import oidata_sim, samples_dict


def test_grid_fit_convergence_uses_half_chi2_not_the_signed_loglike(
    monkeypatch,
):
    seen = []
    real = grid_fit._flux_converged

    def spy(grad, curvature, loglike):
        jax.debug.callback(lambda x: seen.append(onp.asarray(x)), loglike)
        return real(grad, curvature, loglike)

    # _optimize_flux_grid is jitted: a program compiled by an earlier test
    # would bypass the spy, and ours must not outlive it.
    jax.clear_caches()
    monkeypatch.setattr(grid_fit, "_flux_converged", spy)
    small = {key: value[::8] for key, value in samples_dict.items()}
    try:
        grid_fit.optimized_flux_grid(BinaryModelCartesian, oidata_sim, small)
    finally:
        jax.clear_caches()
    assert seen
    jax.effects_barrier()
    first = list(seen)
    # An independent -½χ² at the returned fluxes (not the normalized loglike).
    flux = onp.asarray(
        grid_fit.optimized_flux_grid(BinaryModelCartesian, oidata_sim, small)
    )
    expected = []
    for i, dra in enumerate(onp.asarray(small["dra"])):
        for j, ddec in enumerate(onp.asarray(small["ddec"])):
            model = BinaryModelCartesian(dra, ddec, flux[i, j])
            r = whitened_residuals(model, oidata_sim)
            expected.append(-0.5 * float(np.sum(r**2)))
    got = onp.sort(onp.concatenate([onp.ravel(x) for x in first]))
    onp.testing.assert_allclose(got, onp.sort(expected), rtol=2e-3)


def test_unconverged_warning_reports_the_fraction():
    from virgil._grid import warn_unconverged

    success = onp.array([True, False, False, True])
    with pytest.warns(Warning, match=r"2 of 4.*50\.0%") as record:
        warn_unconverged(success, "f")
    assert record[0].message.counts == {"failed": 2, "total": 4}


def test_ruffio_small_percentile_far_above_zero_is_the_lower_tail():
    from scipy.stats import truncnorm

    sigma = 1e-3
    mean = np.array([4.5, 5.0, 6.0, 10.0], np.float32) * sigma
    limit = ruffio_upperlimit(
        mean, np.full(4, sigma, np.float32), np.float32(1e-9)
    )
    expected = truncnorm.ppf(
        1e-9, -onp.asarray(mean, float) / sigma, onp.inf, loc=mean, scale=sigma
    )
    onp.testing.assert_allclose(limit, expected, rtol=1e-3, atol=1e-6)


def test_ruffio_percentile_one_is_not_nan():
    limit = ruffio_upperlimit(np.array([1.0, -1.0]), np.ones(2), np.array(1.0))
    assert onp.all(onp.isposinf(onp.asarray(limit)))


def test_build_model_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        build_model(BinaryModelCartesian, ["dra", "ddec", "flux"], [1.0, 2.0])
    # zodiax would broadcast a single value over every path
    with pytest.raises(ValueError):
        build_model(
            BinaryModelCartesian(0.0, 0.0, 0.01),
            ["dra", "ddec", "flux"],
            [1.0],
        )


def test_residuals_follow_the_flat_coordinate_switch():
    priors = {
        "dra": dist.Uniform(-200.0, 200.0),
        "ddec": dist.Uniform(-200.0, 200.0),
        "flux": dist.LogUniform(1e-4, 0.5),
    }
    start = BinaryModelCartesian(55.0, -35.0, 0.03)
    flat = _Objective(start, priors, oidata_sim, flat=True)
    flat.residuals(flat.init())  # a flat prior adds no residual
    # Without the flat coordinate a LogUniform has no least-squares form;
    # it must not be silently dropped.
    unflat = _Objective(start, priors, oidata_sim, flat=False)
    with pytest.raises(NoLeastSquaresForm):
        unflat.residuals(unflat.init())


def test_kkt_report_only_swallows_the_no_box_error(monkeypatch):
    from tests.test_kkt_report import _problem
    from virgil.imaging import TSV

    start, priors, data = _problem(8, 1.0)
    result = fitting.fit(
        start,
        priors,
        data,
        [TSV(3.0, path="env", beam_mas=60.0)],
        method="lbfgsb",
    )

    def broken(problem, z):
        raise TypeError("a genuine bug")

    monkeypatch.setattr(fitting, "_box", broken)
    with pytest.raises(TypeError, match="genuine bug"):
        kkt_report(result, priors, data, [TSV(3.0, path="env", beam_mas=60.0)])


def test_face_on_retrograde_thiele_innes_converts():
    orbit = ThieleInnesOrbit(
        period=10.0, dt_peri=0.0, ecc=0.1, A=1.0, B=0.0, F=0.0, G=-1.0
    )
    kepler = orbit.to_kepler()
    assert isinstance(kepler, KeplerOrbit)
    assert 179.0 < float(kepler.inc) < 180.0
    back = kepler.to_thiele_innes()
    onp.testing.assert_allclose(
        [back.A, back.B, back.F, back.G], [1.0, 0.0, 0.0, -1.0], atol=1e-5
    )


def test_linear_flux_grid_warns_on_model_dependent_covariance(monkeypatch):
    from virgil.oidata import OIData

    small = {key: value[::8] for key, value in samples_dict.items()}
    monkeypatch.setattr(OIData, "has_model_covariance", True)
    with pytest.warns(UserWarning, match="model-independent"):
        result = grid_fit.linear_flux_grid(
            BinaryModelCartesian, oidata_sim, small
        )
    assert onp.all(onp.isfinite(onp.asarray(result.flux)))


def test_face_on_retrograde_state_vector_converts():
    from virgil.orbits import StateVectorOrbit

    # r = (1, 0, 0), v = (0, 1, 0): angular momentum along +z, face-on
    # clockwise on the sky, bound for mu = 4.
    orbit = StateVectorOrbit(1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 4.0)
    kepler = orbit.to_kepler()
    assert 179.0 < float(kepler.inc) < 180.0
    onp.testing.assert_allclose(
        kepler.relative(0.0), orbit.relative(0.0), atol=1e-4
    )


def test_face_on_retrograde_jaxoplanet_body_converts():
    from types import SimpleNamespace

    body = SimpleNamespace(
        period=10.0,
        time_peri=0.0,
        eccentricity=0.1,
        sin_omega_peri=0.0,
        cos_omega_peri=1.0,
        sin_asc_node=0.0,
        cos_asc_node=1.0,
        sin_inclination=-1e-16,  # float32 sin(pi) < 0
        cos_inclination=-1.0,
    )
    kepler = KeplerOrbit.from_jaxoplanet(body, a_mas=20.0)
    assert 179.0 < float(kepler.inc) < 180.0


def test_aliases_physical_clips_cos_inc_minus_one():
    from virgil.aliases import _physical

    x = onp.zeros((1, 8))
    x[:, 3] = -1.0
    inc = _physical(x)["inc"]
    assert 179.0 < float(inc[0]) < 180.0
