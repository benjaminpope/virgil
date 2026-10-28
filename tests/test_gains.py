"""Calibration gains correlated across channels (virgil.gains, Stage 6d)."""

import jax
import jax.numpy as np
import numpy as onp
import pytest

from virgil.fitting import fit
from virgil.likelihood import model_loglike, whitened_residuals
from virgil.models import BinaryModelCartesian
from virgil.oidata import OIData
from tests._shared import PAIRS

TRUTH = BinaryModelCartesian(dra=3.0, ddec=-2.0, flux=0.3)


def _data(n_frame=2, n_wave=5, vis_mode="auto", v2=True, seed=0):
    """Four-telescope data, ``n_frame`` frames of six baselines."""
    rng = onp.random.default_rng(seed)
    nb = 6 * n_frame
    u, v = rng.normal(0, 60, nb), rng.normal(0, 60, nb)
    wavel = onp.linspace(2.0e-6, 2.4e-6, n_wave)
    vis = rng.uniform(0.3, 0.9, (nb, n_wave))
    return OIData(
        {
            "u": u,
            "v": v,
            "wavel": wavel,
            "vis": vis if v2 else onp.sqrt(vis),
            "d_vis": onp.full((nb, n_wave), 0.01),
            "v2_flag": v2,
            "vis_mode": vis_mode,
            "frame": onp.repeat(onp.arange(n_frame), 6),
            "mjd": 60000.0 + onp.repeat(onp.arange(n_frame), 6),
            "stations": onp.tile(PAIRS, (n_frame, 1)),
        }
    )


def _jacobian(data, prediction):
    return {"v2": 2 * prediction, "amp": prediction}.get(
        data.vis_mode, onp.ones_like(prediction)
    )


@pytest.mark.parametrize(
    "vis_mode, v2", [("auto", True), ("auto", False), ("logamp", True)]
)
@pytest.mark.validates(
    "virgil.gains.GainModes", roots=["mathematics"], kind="check"
)
def test_whitening_matches_the_dense_covariance(vis_mode, v2):
    data = _data(vis_mode=vis_mode, v2=v2).with_gains(
        telescope=0.02, baseline=0.01, chromatic=0.005
    )
    gains = data.gains
    assert gains.rows.shape[0] == 2  # one block per frame
    prediction = onp.asarray(data.model(TRUTH))
    obs, err = (onp.asarray(x) for x in data.flatten_data())
    resid = prediction - obs
    cov = onp.asarray(
        gains.covariance(err, _jacobian(data, prediction), gains.widths),
        float,
    )
    chi2 = resid @ onp.linalg.solve(cov, resid)
    logdet = onp.linalg.slogdet(cov)[1]
    expected = (
        -0.5 * chi2 - 0.5 * logdet - 0.5 * resid.size * onp.log(2 * onp.pi)
    )
    whitened = whitened_residuals(TRUTH, data)
    assert float(np.sum(whitened**2)) == pytest.approx(chi2, rel=1e-4)
    assert float(model_loglike(TRUTH, data)) == pytest.approx(
        expected, rel=1e-4
    )


def test_supplied_modes_spanning_frames_join_one_block():
    data = _data()
    mode = onp.zeros(data.u.size)
    mode[::7] = 0.01  # touches both frames
    # Baseline gains alone give a block per (frame, baseline); telescope
    # gains join each frame. A mode spanning frames stays apart, so the
    # frames remain separate blocks.
    assert data.with_gains(baseline=0.01).gains.rows.shape[0] == 12
    local = mode * (onp.asarray(data.frame) == 0)
    data = data.with_gains(telescope=0.01, baseline=0.01, modes=[mode, local])
    gains = data.gains
    assert gains.rows.shape == (2, 30)
    assert gains.spanning.shape == (60, 1)
    assert gains.groups == ("telescope", "baseline", "modes")
    # ...and is still whitened exactly.
    prediction = onp.asarray(data.model(TRUTH))
    obs, err = (onp.asarray(x) for x in data.flatten_data())
    resid = prediction - obs
    cov = onp.asarray(
        gains.covariance(err, 2 * prediction, gains.widths), float
    )
    chi2 = resid @ onp.linalg.solve(cov, resid)
    logdet = onp.linalg.slogdet(cov)[1]
    expected = (
        -0.5 * chi2 - 0.5 * logdet - 0.5 * resid.size * onp.log(2 * onp.pi)
    )
    assert float(model_loglike(TRUTH, data)) == pytest.approx(
        expected, rel=1e-4
    )
    # The frames are 1 day apart: a mode shared by them stops a split.
    with pytest.raises(ValueError, match="spans several epochs"):
        data.split_by_epoch()


def test_gain_terms_replace_the_default_widths():
    data = _data()
    fixed = data.with_gains(telescope=0.03, baseline=0.01)
    nominal = data.with_gains(telescope=0.001, baseline=0.01)
    assert float(model_loglike(TRUTH, fixed)) == pytest.approx(
        float(model_loglike(TRUTH, nominal, vis_gain_telescope=0.03)),
        rel=1e-5,
    )
    with pytest.raises(ValueError, match="no modes"):
        model_loglike(TRUTH, nominal, vis_gain_chromatic=0.01)
    with pytest.raises(ValueError, match="with_gains"):
        model_loglike(TRUTH, data, vis_gain_telescope=0.01)


def test_zero_width_is_the_diagonal_likelihood_with_finite_gradients():
    data = _data()
    plain = float(model_loglike(TRUTH, data))
    gains = data.with_gains(telescope=0.0, baseline=0.0)
    assert float(model_loglike(TRUTH, gains)) == pytest.approx(plain, rel=1e-6)

    # Degenerate modes (equal visibilities on every baseline) and zero
    # widths: the gradient stays finite.
    flat = gains.set("vis", np.full_like(gains.vis, 0.5))

    def loglike(width, flux):
        model = TRUTH.set("flux", flux)
        return model_loglike(model, flat, vis_gain_telescope=width)

    grads = jax.grad(loglike, argnums=(0, 1))(0.0, 0.3)
    assert all(onp.isfinite(float(g)) for g in grads)
    grads = jax.grad(loglike, argnums=(0, 1))(0.02, 0.3)
    assert all(onp.isfinite(float(g)) for g in grads)


def test_splitting_by_frame_keeps_the_likelihood():
    data = _data(n_frame=3)
    data = data.with_gains(telescope=0.02, baseline=0.01)
    parts = data.split_by_epoch(gap_days=0.5)
    assert len(parts) == 3
    whole = float(model_loglike(TRUTH, data))
    assert sum(float(model_loglike(TRUTH, p)) for p in parts) == pytest.approx(
        whole, rel=1e-5
    )


def test_drawn_gains_have_the_modelled_covariance():
    data = _data(n_frame=1, n_wave=3).with_gains(telescope=0.02, baseline=0.01)
    keys = jax.random.split(jax.random.PRNGKey(1), 4000)
    draws = jax.vmap(lambda k: data.gains.sample(k, data.gains.widths))(keys)
    empirical = onp.cov(onp.asarray(draws), rowvar=False)
    ones = onp.ones(data.vis.size)
    expected = onp.asarray(
        data.gains.covariance(0 * ones, ones, data.gains.widths)
    )
    assert onp.allclose(empirical, expected, atol=1e-4)
    # with_model applies them: noiseless V² differ from the model by e^{2g}.
    noisy = data.with_model(TRUTH, jax.random.PRNGKey(2), noise_scale=0.0)
    ratio = onp.asarray(noisy.vis) / onp.asarray(data.model(TRUTH))
    assert not onp.allclose(ratio, 1.0)
    log_gain = 0.5 * onp.log(ratio)
    assert onp.allclose(
        log_gain.reshape(6, 3), log_gain.reshape(6, 3)[:, :1], atol=1e-6
    )


def test_projected_data_refuse_gains():
    data = _data()
    projected = OIData(
        {
            "u": onp.asarray(data.u),
            "v": onp.asarray(data.v),
            "wavel": onp.asarray(data.wavel),
            "vis": onp.asarray(data.vis),
            "d_vis": onp.asarray(data.d_vis),
            "vis_mat": onp.eye(data.vis.size)[:5],
            "frame": onp.asarray(data.frame),
            "stations": onp.asarray(data.stations),
        }
    )
    with pytest.raises(ValueError, match="projected"):
        projected.with_gains(baseline=0.01)


def test_missing_stations_or_frames_are_reported():
    data = _data()
    import equinox as eqx

    bare = eqx.tree_at(
        lambda d: d.stations, data, None, is_leaf=lambda x: x is None
    )
    with pytest.raises(ValueError, match="stations"):
        bare.with_gains(telescope=0.01)
    no_frames = eqx.tree_at(
        lambda d: d.frame, bare, None, is_leaf=lambda x: x is None
    )
    with pytest.raises(ValueError, match="frame"):
        no_frames.with_gains(modes=onp.ones(data.u.size))


def test_fit_with_gains_uses_lbfgs_and_fits_widths():
    import numpyro.distributions as dist

    data = _data(n_frame=4, n_wave=4).with_gains(telescope=0.01, baseline=0.01)
    data = data.with_model(TRUTH, jax.random.PRNGKey(3))
    start = TRUTH.set("flux", 0.25)
    result = fit(
        start,
        {"flux": dist.Uniform(0.0, 1.0)},
        data,
        # Widths are scale parameters: log-uniform (Jeffreys) on stated bounds.
        # The default start, 0.01, is outside these bounds, so the fit
        # starts at the prior's mean instead.
        noise={"vis_gain_telescope": dist.LogUniform(0.02, 0.5)},
    )
    assert result.info["method"] == "lbfgs"
    assert 0.02 <= float(result.values["noise.vis_gain_telescope"]) <= 0.5
    assert onp.isfinite(result.info["loss"])
    with pytest.raises(TypeError, match="least-squares"):
        fit(start, {"flux": dist.Uniform(0.0, 1.0)}, data, method="lm")


def test_synthetic_vlti_coverage_takes_gains():
    from virgil.coverage import vlti_oidata

    data = vlti_oidata(hour_angles_h=(-1.0, 0.0, 1.0)).with_gains(
        telescope=0.01, baseline=0.01
    )
    assert data.gains.rows.shape[0] == 3  # a block per snapshot


def test_modes_spanning_frames_alone_are_whitened_exactly():
    data = _data()
    rng = onp.random.default_rng(4)
    data = data.with_gains(modes=rng.normal(0, 0.02, (2, data.u.size)))
    gains = data.gains
    assert gains.rows.shape[0] == 1 and gains.spanning.shape == (60, 2)
    assert onp.all(gains.rows == data.u.size)  # one empty block
    prediction = onp.asarray(data.model(TRUTH))
    obs, err = (onp.asarray(x) for x in data.flatten_data())
    resid = prediction - obs
    cov = onp.asarray(
        gains.covariance(err, 2 * prediction, gains.widths), float
    )
    chi2 = resid @ onp.linalg.solve(cov, resid)
    logdet = onp.linalg.slogdet(cov)[1]
    expected = (
        -0.5 * chi2 - 0.5 * logdet - 0.5 * resid.size * onp.log(2 * onp.pi)
    )
    assert float(model_loglike(TRUTH, data)) == pytest.approx(
        expected, rel=1e-4
    )
    draws = jax.vmap(lambda k: gains.sample(k, gains.widths))(
        jax.random.split(jax.random.PRNGKey(0), 3)
    )
    assert draws.shape == (3, 60)


def _zero_size_leaves(tree):
    return [
        jax.tree_util.keystr(path)
        for path, leaf in jax.tree_util.tree_leaves_with_path(tree)
        if hasattr(leaf, "size") and leaf.size == 0
    ]


def test_gains_without_spanning_modes_hold_no_zero_size_arrays():
    # jax.pmap segfaults compiling a zero-size captured array (JAX 0.11.2),
    # which crashed numpyro's parallel chains on data with gains.
    data = _data().with_gains(telescope=0.01, baseline=0.01, chromatic=0.01)
    assert data.gains.spanning is None and data.gains.spanning_group is None
    assert _zero_size_leaves(data.gains) == []
    spanning_only = _data().with_gains(
        modes=onp.random.default_rng(4).normal(0, 0.02, (2, 60))
    )
    assert _zero_size_leaves(spanning_only.gains) == []
    sub = data.gains.subset(onp.arange(data.gains.n_vis) % 2 == 0)
    assert sub.spanning is None
    assert data.gains.sample(
        jax.random.PRNGKey(0), data.gains.widths
    ).shape == (data.gains.n_vis,)


@pytest.mark.parametrize(
    "gains",
    [
        "with_gains(telescope=0.01, baseline=0.01)",
        # every mode spans frames: no blocks within a frame
        "with_gains(modes=onp.random.default_rng(0).normal(0, 0.02, (2, d.u.size)))",
    ],
)
def test_parallel_chains_compile_with_gains(gains):
    # Guards the fix only on JAX versions whose pmap goes through Shardy
    # (0.11.2 segfaults); on others it passes either way.
    import subprocess
    import sys

    script = f"""
import os
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"
import jax, jax.numpy as jnp, numpy as onp
from virgil.coverage import vlti_oidata
from virgil.likelihood import model_loglike
from virgil.models import BinaryModelCartesian

d = vlti_oidata(hour_angles_h=(-1.0, 0.0, 1.0), wavelengths_m=[2.0e-6, 2.4e-6])
data = d.{gains}
ll = jax.pmap(
    lambda f: model_loglike(BinaryModelCartesian(dra=3.0, ddec=-2.0, flux=f), data)
)
print(ll(jnp.array([0.1, 0.2])))
"""
    run = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert run.returncode == 0, run.stderr[-2000:]


def test_numpyro_model_warns_of_zero_size_arrays_with_parallel_chains(
    monkeypatch,
):
    import warnings

    import numpyro.distributions as dist

    from virgil import likelihood
    from virgil.likelihood import numpyro_model

    priors = {"flux": dist.LogUniform(0.01, 1.0)}

    def build(**kw):
        return BinaryModelCartesian(dra=3.0, ddec=-2.0, **kw)

    v2_only = _data()  # no closure phases: zero-size phase arrays
    assert _zero_size_leaves(v2_only)
    monkeypatch.setattr(likelihood, "_host_devices", lambda: 2)
    with pytest.warns(RuntimeWarning, match="vectorized") as record:
        numpyro_model(build, priors, v2_only)
    assert record[0].filename == __file__
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        monkeypatch.setattr(likelihood, "_host_devices", lambda: 1)
        numpyro_model(build, priors, v2_only)
