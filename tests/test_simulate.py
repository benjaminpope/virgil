"""simulate and bias_test (design orbit_scene_joint_fitting.md R8)."""

import jax
import numpy as onp
import numpyro.distributions as dist
import pytest

from virgil.coverage import nrm_oidata
from virgil.models import BinaryModelCartesian, PointSource, System
from virgil.oidata import OIData
from virgil.simulate import bias_test, simulate

TRUTH = BinaryModelCartesian(dra=60.0, ddec=-40.0, flux=0.05)


def test_noiseless_simulation_is_the_model():
    template = nrm_oidata()
    fake = simulate(TRUTH, template)
    assert onp.allclose(fake.flatten_data()[0], template.model(TRUTH))
    noisy = simulate(TRUTH, template, key=jax.random.PRNGKey(0))
    assert not onp.allclose(noisy.flatten_data()[0], fake.flatten_data()[0])
    # Errors are the template's.
    assert onp.allclose(noisy.flatten_data()[1], template.flatten_data()[1])


def _record(template):
    return {
        "u": onp.asarray(template.u),
        "v": onp.asarray(template.v),
        "wavel": onp.asarray(template.wavel),
        "vis": onp.ones(template.u.size),
        "d_vis": onp.full(template.u.size, 0.01),
        "phi": onp.zeros(template.i_cps1.size),
        "d_phi": onp.full(template.i_cps1.size, 0.01),
        "i_cps1": onp.asarray(template.i_cps1),
        "i_cps2": onp.asarray(template.i_cps2),
        "i_cps3": onp.asarray(template.i_cps3),
    }


def _timed(mjd):
    template = nrm_oidata()
    return OIData({**_record(template), "mjd": onp.full(template.u.size, mjd)})


def test_shifting_the_epochs_moves_an_orbiting_companion():
    pytest.importorskip("jaxoplanet")
    from virgil.models import Attached
    from virgil.orbits import KeplerOrbit

    orbit = KeplerOrbit(
        400.0, 0.0, 0.3, 60.0, 40.0, 110.0, 80.0, t_ref=60000.0
    )
    scene = System(
        primary=PointSource(), comp=Attached(PointSource(0.05), orbit)
    )
    shifted = simulate(scene, _timed(60000.0), shift_days=100.0)
    later = simulate(scene, _timed(60100.0))
    assert onp.allclose(
        shifted.flatten_data()[0], later.flatten_data()[0], atol=1e-5
    )
    with pytest.raises(ValueError, match="needs a template with times"):
        simulate(TRUTH, nrm_oidata(), shift_days=1.0)


def test_bias_test_recovers_a_point_source_binary():
    priors = {
        "dra": dist.Uniform(30.0, 90.0),
        "ddec": dist.Uniform(-70.0, -10.0),
        "flux": dist.Uniform(0.0, 0.2),
    }
    out = bias_test(
        TRUTH, nrm_oidata(), TRUTH, priors, n=3, key=jax.random.PRNGKey(1)
    )
    assert set(out) == {"dra", "ddec", "flux", "chi2_red"}
    assert out["dra"].shape == (3,)
    # Three draws of a faint companion scatter by a few mas.
    assert abs(out["dra"].mean() - 60.0) < 6.0
    assert abs(out["flux"].mean() - 0.05) < 0.02
    assert onp.all(out["chi2_red"] < 3.0)


def test_a_large_shift_keeps_close_samples_apart():
    # 30 s apart, shifted by 10,000 days: dt + shift in float32 would merge
    # them (its spacing there is 84 s); moving t_ref keeps them.
    template = _timed(60000.0)
    mjd = 60000.0 + onp.arange(template.u.size) * 30.0 / 86400.0
    record = {**_record(template), "mjd": mjd}
    shifted = simulate(TRUTH, OIData(record), shift_days=10000.0)
    gaps = onp.diff(shifted.mjd) * 86400.0
    assert onp.allclose(gaps, 30.0, atol=0.5)
    assert shifted.mjd[0] == pytest.approx(70000.0)
