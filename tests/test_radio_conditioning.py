"""Natural-unit coordinates and profiled error scales of CalibrationNuisance."""

import jax
import jax.numpy as np
import numpy as onp
import pytest

from virgil import CalibrationNuisance, ComplexVisData, fit
from virgil._utils import mas2rad, rad2mas
from virgil.radio import C_LIGHT
from virgil.fitting import _Objective
from virgil.models import GaussianDisk
from tests._shared import brute_dirty_image

FREQS = onp.array([230e9, 240e9, 245e9])
FLUX = 0.1
DDID_OBS = onp.array([0, 1, 1])  # ddid 1 and 2 belong to observation 1


def _problem(
    offsets=((0.0, 0.0), (0.01, -0.006)),
    gain=1.1,
    scales=(1.0, 1.4, 0.8),
    n=1500,
    seed=3,
):
    """Observation 0: short baselines (compact array); observation 1: long
    baselines (extended array), offset by small_offsets."""
    rng = onp.random.default_rng(seed)
    ddid = rng.integers(0, 3, n)
    obs = DDID_OBS[ddid]
    freq = FREQS[ddid]
    reach = onp.where(obs == 0, 3e6, 1e8)
    u, v = (rng.uniform(-1, 1, n) * reach for _ in range(2))
    weight = onp.full(n, 1.0e4)
    template = GaussianDisk(
        sigma=0.3
    )  # compact: resolved by the long baselines
    lam = C_LIGHT / freq
    clean = onp.asarray(template.model(u * lam, v * lam, lam))
    off = onp.asarray(offsets)[obs]
    ramp = onp.exp(-2j * onp.pi * mas2rad * (off[:, 0] * u + off[:, 1] * v))
    vis = (
        FLUX
        * onp.where(obs == 1, gain, 1.0)
        * (freq / 239e9) ** 2.0
        * ramp
        * clean
    )
    noise = onp.array(scales)[ddid] / onp.sqrt(weight)
    vis = vis + noise * (rng.normal(size=n) + 1j * rng.normal(size=n))
    data = ComplexVisData.from_arrays(
        u, v, vis, weight, freq, ddid=ddid, obs=obs
    )
    return data, template


def _truth(nuis, scales=(1.0, 1.4, 0.8), offset=(0.01, -0.006), gain=1.1):
    values = nuis.defaults()
    values["cal.flux"] = FLUX
    values["cal.alpha"] = 2.0
    if nuis.gains:
        values["cal.gain"] = onp.array([gain])
    if nuis.offsets:
        values["cal.offset"] = onp.array([offset])
    if nuis.sigmas is True:
        values["cal.sigma"] = onp.array(scales)
    return values


def _loss_hessian_diag(nuis, priors, data, template):
    """Diagonal of the Hessian of the loss in the fit's coordinates, at the
    truth (the Gauss–Newton Fisher there, up to noise)."""
    term = data.term(lambda v: template, nuis)
    problem = _Objective(lambda **v: template, priors, (), likelihoods=[term])
    z0 = problem.init(_truth(nuis))
    flat, unravel = problem.ravel(z0)

    def loss(x):
        return problem.loss(unravel(x))

    return onp.diag(onp.asarray(jax.hessian(loss)(flat)))


@pytest.fixture(scope="module", autouse=True)
def _x64():
    with jax.enable_x64(True):
        yield


def test_fisher_diagonal_spans_fewer_decades():
    data, template = _problem()
    nuis = CalibrationNuisance(data, nu0_hz=239e9)
    theta = nuis.offset_bound_mas
    old = nuis.priors(
        gain=(0.5, 2.0), sigma=(0.1, 10.0), offset=(-theta, theta)
    )
    new = nuis.priors()
    d_old = _loss_hessian_diag(nuis, old, data, template)
    d_new = _loss_hessian_diag(nuis, new, data, template)
    # coordinates in the order of the priors: flux, gain, offset x2, sigma x3,
    # alpha. Compare the calibration sites whose coordinates changed.
    cal = slice(1, 7)
    spread = lambda d: onp.log10(d.max() / d.min())  # noqa: E731
    print(
        f"spread (gain, offset, sigma) old {spread(d_old[cal]):.1f}, "
        f"new {spread(d_new[cal]):.1f} decades; all sites old "
        f"{spread(d_old):.1f}, new {spread(d_new):.1f}"
    )
    assert spread(d_old[cal]) > 6
    assert spread(d_new[cal]) < 3
    assert spread(d_new) < spread(d_old) - 2


def test_coordinates_are_set_by_the_box():
    data, _ = _problem()
    nuis = CalibrationNuisance(data, nu0_hz=239e9)
    priors = nuis.priors()
    problem = _Objective(None, priors, (), likelihoods=())
    bij = problem.bijection(priors["cal.offset"])
    unit = onp.asarray(nuis.offset_unit_mas[1])
    q_max = onp.hypot(data.u, data.v)[onp.asarray(data.obs) == 1].max()
    assert unit == pytest.approx(rad2mas / (2 * onp.pi * q_max))
    dx = jax.jvp(bij, (np.zeros((1, 2)),), (np.ones((1, 2)),))[1]
    # one unit of z is a few b_k at the centre, not 1e4 mas
    assert 1 < float(dx[0, 0]) / unit < 20
    gain = problem.bijection(priors["cal.gain"])
    assert float(gain(np.asarray([0.3]))[0]) == pytest.approx(np.exp(0.3))


def test_fit_recovers_gain_and_small_offset_offset():
    data, template = _problem()
    nuis = CalibrationNuisance(data, nu0_hz=239e9)
    term = data.term(lambda v: template, nuis)
    result = fit(
        lambda **v: template,
        nuis.priors(),
        data=(),
        likelihoods=[term],
        init=nuis.defaults(),
        method="lbfgs",
        max_steps=3000,
    )
    got = result.values
    assert float(got["cal.gain"][0]) == pytest.approx(1.1, rel=0.03)
    onp.testing.assert_allclose(
        onp.asarray(got["cal.offset"])[0], [0.01, -0.006], atol=3e-3
    )
    onp.testing.assert_allclose(
        onp.asarray(got["cal.sigma"]), [1.0, 1.4, 0.8], rtol=0.1
    )


def test_profile_gradient_matches_finite_differences():
    data, template = _problem(n=400)
    nuis = CalibrationNuisance(data, nu0_hz=239e9, sigmas="profile")
    assert "cal.sigma" not in nuis.priors()
    term = data.term(lambda v: template, nuis, chunk_size=150)
    problem = _Objective(
        lambda **v: template, nuis.priors(), (), likelihoods=[term]
    )
    z0 = problem.init(_truth(nuis))
    flat, unravel = problem.ravel(z0)
    flat = flat + 0.05 * onp.random.default_rng(0).normal(size=flat.shape)

    def loss(x):
        return problem.loss(unravel(x))

    grad = onp.asarray(jax.grad(loss)(flat))
    step = 1e-6
    for i in range(flat.size):
        e = np.zeros_like(flat).at[i].set(step)
        fd = (loss(flat + e) - loss(flat - e)) / (2 * step)
        assert float(fd) == pytest.approx(grad[i], rel=1e-5, abs=1e-6)


def test_profile_optimum_equals_joint_optimum():
    data, template = _problem(n=600)
    results = {}
    for mode in (True, "profile"):
        nuis = CalibrationNuisance(
            data, nu0_hz=239e9, sigmas=mode, offsets=False
        )
        term = data.term(lambda v: template, nuis)
        results[mode] = fit(
            lambda **v: template,
            nuis.priors(),
            data=(),
            likelihoods=[term],
            init=nuis.defaults(),
            method="lbfgs",
            gtol=1e-8,
            max_steps=5000,
        )
    joint, prof = results[True], results["profile"]
    for site in ("cal.flux", "cal.gain", "cal.alpha"):
        onp.testing.assert_allclose(
            onp.asarray(prof.values[site]),
            onp.asarray(joint.values[site]),
            rtol=1e-4,
        )
    onp.testing.assert_allclose(
        prof.info["derived"]["cal.sigma"],
        onp.asarray(joint.values["cal.sigma"]),
        rtol=1e-4,
    )
    assert prof.info["loss"] == pytest.approx(joint.info["loss"], rel=1e-8)
    assert "derived" in joint.info and joint.info["derived"] == {}


def test_dirty_image_use_sigma_weights_by_the_fitted_scales():
    data, template = _problem(n=300)
    nuis = CalibrationNuisance(data, nu0_hz=239e9, sigmas=True)
    values = _truth(nuis)
    scales = onp.asarray(values["cal.sigma"])
    img = data.dirty_image(
        8, 8.0, scene=template, values=values, nuisance=nuis, use_sigma=True
    )
    model = onp.asarray(data.model(template, values, nuis))
    ref = brute_dirty_image(
        data.u,
        data.v,
        onp.asarray(data.vis) - model,
        onp.asarray(data.weight) / scales[onp.asarray(data.ddid)] ** 2,
        8,
        8.0,
    )
    onp.testing.assert_allclose(img, ref, atol=1e-6 * onp.abs(ref).max())


def test_sigma_hat_and_dirty_image_with_profile():
    data, template = _problem(n=300)
    nuis = CalibrationNuisance(data, nu0_hz=239e9, sigmas="profile")
    values = _truth(nuis)
    s = data.sigma_hat(template, values, nuis)
    chi2, n = data.chi2_by_ddid(template, values, nuis)
    onp.testing.assert_allclose(chi2, 2 * n)  # χ² at ŝ is 2 N_d
    assert s.shape == (3,) and onp.all(s > 0)
    img = data.dirty_image(
        8, 8.0, scene=template, values=values, nuisance=nuis, use_sigma=True
    )
    # The residual V - M weighted by w / s_hat[ddid]^2, by direct sum.
    model = onp.asarray(data.model(template, values, nuis))
    weights = (
        onp.asarray(data.weight) / onp.asarray(s)[onp.asarray(data.ddid)] ** 2
    )
    ref = brute_dirty_image(
        data.u, data.v, onp.asarray(data.vis) - model, weights, 8, 8.0
    )
    onp.testing.assert_allclose(img, ref, atol=1e-6 * onp.abs(ref).max())
    unweighted = data.dirty_image(
        8, 8.0, scene=template, values=values, nuisance=nuis
    )
    assert onp.abs(unweighted - ref).max() > 1e-3 * onp.abs(ref).max()
    with pytest.raises(ValueError, match="scene"):
        data.dirty_image(8, 8.0, nuisance=nuis, use_sigma=True)
    with pytest.raises(ValueError, match="profile"):
        CalibrationNuisance(data, sigmas="bogus")


def test_offset_on_its_box_is_reported_at_bound():
    # an offset of 0.01 mas against a box of +-0.001 mas
    data, template = _problem()
    nuis = CalibrationNuisance(data, nu0_hz=239e9, sigmas=False)
    term = data.term(lambda v: template, nuis)
    narrow = nuis.priors(offset=(-0.001, 0.001))
    pinned = fit(
        lambda **v: template,
        narrow,
        data=(),
        likelihoods=[term],
        init=nuis.defaults(),
        method="lbfgs",
        max_steps=500,
    )
    assert "cal.offset" in pinned.info["at_bound"]


def test_fixed_sigma_scores_held_out_points_at_training_scales():
    data, template = _problem(n=300)
    nuis = CalibrationNuisance(data, nu0_hz=239e9, sigmas="profile")
    values = _truth(nuis)
    mask = onp.arange(len(data)) % 2 == 0
    own = data.term(lambda v: template, nuis, mask=mask)
    s_train = data.sigma_hat(template, values, nuis, mask=~mask)
    fixed = data.term(lambda v: template, nuis, mask=mask, sigma=s_train)
    onp.testing.assert_allclose(fixed.sigma_hat(values), s_train)
    assert not onp.allclose(own.sigma_hat(values), s_train)
    # at fixed scales the held-out score is no better than at its own ŝ
    assert float(fixed.log_norm(values)) + 0.5 * float(
        np.sum(fixed(values) ** 2)
    ) >= float(own.log_norm(values)) + 0.5 * float(np.sum(own(values) ** 2))
    with pytest.raises(ValueError, match="profile"):
        data.term(
            lambda v: template,
            CalibrationNuisance(data, sigmas=True),
            sigma=s_train,
        )
