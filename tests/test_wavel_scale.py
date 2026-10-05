"""The wavelength-scale nuisance (wavel_scale, wavel_offset; Stage 6d)."""

import jax
import numpy as onp
import numpyro.distributions as dist
import pytest

from virgil.coverage import vlti_oidata
from virgil.fitting import fit
from virgil.likelihood import model_loglike, noise_sites
from virgil.models import BinaryModelCartesian

TRUTH = BinaryModelCartesian(dra=6.0, ddec=-4.0, flux=0.3)


def _data(scale=1.0, key=0):
    data = vlti_oidata(
        hour_angles_h=(-2.0, 0.0, 2.0),
        wavelengths_m=onp.linspace(2.0e-6, 2.4e-6, 6),
        sigma_v2=0.01,
        sigma_cp_deg=0.5,
    )
    truth_scale = data.with_wavelength_scale(scale)
    key = None if key is None else jax.random.PRNGKey(key)
    simulated = truth_scale.with_model(TRUTH, key)
    return simulated.set(["wavel"], [data.wavel])


@pytest.mark.validates(
    "virgil.oidata.OIData.with_wavelength_scale",
    roots=["mathematics"],
    kind="check",
)
def test_a_wavelength_scale_rescales_angular_sizes():
    # Spatial frequency u / (s λ): a grey binary at separation x looks the
    # same as one at x / s with the wavelengths unscaled.
    data, s = _data(), 1.003
    scaled = model_loglike(TRUTH, data, wavel_scale=s)
    shrunk = TRUTH.set(["dra", "ddec"], [TRUTH.dra / s, TRUTH.ddec / s])
    assert float(scaled) == pytest.approx(
        float(model_loglike(shrunk, data)), rel=1e-6
    )


def test_an_offset_shifts_the_wavelengths():
    data, delta = _data(), 2e-9
    shifted = data.set(["wavel"], [data.wavel + delta])
    assert float(model_loglike(TRUTH, data, wavel_offset=delta)) == (
        pytest.approx(float(model_loglike(TRUTH, shifted)), rel=1e-6)
    )
    # The observed data are left as they were.
    assert onp.array_equal(
        onp.asarray(data.with_wavelength_scale().wavel),
        onp.asarray(data.wavel),
    )


def test_wavelength_priors_may_be_signed_but_error_priors_may_not():
    sites = noise_sites(
        {
            "wavel_scale": dist.Normal(1.0, 2e-4),
            "wavel_offset": dist.Normal(0.0, 1e-9),
        },
        1,
    )
    assert set(sites) == {"noise.wavel_scale", "noise.wavel_offset"}
    with pytest.raises(ValueError, match="non-negative"):
        noise_sites({"vis_scale": dist.Normal(1.0, 0.1)}, 1)


def test_fit_recovers_a_wavelength_scale_with_the_separation_known():
    # Noiseless, so that the fit must find the scale itself (with noise,
    # its error here is about 1e-3).
    data = _data(scale=1.002, key=None)
    result = fit(
        TRUTH,
        {"flux": dist.Uniform(0.0, 1.0)},
        data,
        noise={"wavel_scale": dist.Normal(1.0, 0.01)},
    )
    assert result.info["method"] == "lbfgs"
    assert float(result.values["noise.wavel_scale"]) == pytest.approx(
        1.002, abs=2e-5
    )


def test_a_bounded_offset_prior_starts_inside_it():
    # wavel_offset starts at 0, the boundary of Uniform(0, ...), where the
    # unconstrained coordinate would be -inf.
    data = _data(key=None)
    result = fit(
        TRUTH,
        {"flux": dist.Uniform(0.0, 1.0)},
        data,
        noise={"wavel_offset": dist.Uniform(0.0, 1e-8)},
    )
    assert onp.isfinite(float(result.values["noise.wavel_offset"]))
    assert onp.isfinite(result.info["loss"])


def test_the_uv_grid_is_kept_and_matches_the_direct_transform():
    import equinox as eqx

    from virgil.coverage import ami_grid_record
    from virgil.models import Image, PointSource, System
    from virgil.oidata import OIData
    from virgil.scenes import ring

    data = OIData(ami_grid_record(rotation_deg=-6.9))
    scaled = data.with_wavelength_scale(1.01)
    assert scaled.uv_grid is not None
    env = Image.from_brightness(
        ring(48, 8.0, 80.0, 10.0, 50.0, 30.0, 0.5, 100.0),
        8.0,
        flux=0.3,
        rotation_deg=data.uv_grid.rotation_deg,
    )
    scene = System(star=PointSource(), env=env)
    direct = eqx.tree_at(
        lambda d: d.uv_grid, scaled, None, is_leaf=lambda x: x is None
    )
    fast, slow = scaled.model(scene), direct.model(scene)
    assert onp.max(onp.abs(fast - slow)) < 1e-6 * onp.max(onp.abs(slow)) + 1e-6
    # The scale really changes the model.
    assert onp.max(onp.abs(fast - data.model(scene))) > 1e-4
