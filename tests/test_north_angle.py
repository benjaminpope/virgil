"""Per-dataset North-angle and plate-scale nuisances (design
orbit_prior_art.md §4.3).

A dataset whose North is off by δ and whose plate scale is off by a factor m
sees a source at true (dra, ddec) at m R(δ) (dra, ddec), so a companion at
PA θ and separation ρ appears at PA θ + δ and separation m ρ. For OIData the
rotation is ``north_angle`` and the scale is ``wavel_scale = 1 / m``; for
PositionData both are ``PositionData.term`` arguments. The sign test through
OIFITS files is in ``test_pa_round_trip.py``.
"""

import equinox as eqx
import jax
import numpy as onp
import numpyro.distributions as dist
import pytest

from virgil.coverage import nrm_oidata
from virgil.likelihood import model_loglike, noise_sites, numpyro_model
from virgil.models import EllipticalGaussian, PointSource, System
from virgil.oidata import find_uv_grid


def _sky(x, y, delta, m):
    """m R(δ) (x, y), through separation and PA (independent of virgil)."""
    sep, pa = onp.hypot(x, y), onp.arctan2(x, y)
    pa = pa + onp.deg2rad(delta)
    return m * sep * onp.sin(pa), m * sep * onp.cos(pa)


def _scene(delta=0.0, m=1.0):
    """A scene with no central symmetry, rotated and magnified by hand."""
    c_ra, c_dec = _sky(4.0, 1.5, delta, m)
    b_ra, b_dec = _sky(-2.0, 3.0, delta, m)
    return System(
        star=PointSource(),
        companion=PointSource(flux=0.3, dra=c_ra, ddec=c_dec),
        blob=EllipticalGaussian(
            fwhm=6.0 * m,
            ratio=0.4,
            pa=30.0 + delta,
            flux=0.5,
            dra=b_ra,
            ddec=b_dec,
        ),
    )


def _data():
    data = nrm_oidata(wavelength_m=2.2e-6, rotation_deg=12.0)
    # Baselines of a few tens of metres resolve a few mas.
    data = data.set(["u", "v"], [data.u * 4.0, data.v * 4.0])
    return data.with_model(_scene(), key=None)


@pytest.mark.validates(
    "virgil.oidata.OIData.with_north_angle",
    "virgil.oidata.OIData.with_wavelength_scale",
    roots=["mathematics"],
    kind="check",
)
def test_rotating_uv_rotates_the_sky_and_wavel_scale_is_the_plate_scale():
    data, delta, m = _data(), 23.0, 1.04
    by_hand = data.model(_scene(delta, m))
    seen = data.with_north_angle(delta).with_wavelength_scale(1.0 / m)
    onp.testing.assert_allclose(
        onp.asarray(seen.model(_scene())),
        onp.asarray(by_hand),
        atol=2e-5,
    )
    # The same through the likelihood's noise terms.
    assert float(
        model_loglike(_scene(), data, north_angle=delta, wavel_scale=1.0 / m)
    ) == pytest.approx(float(model_loglike(_scene(delta, m), data)), rel=1e-5)
    # The scene is not centrally symmetric, so the sign is visible: the
    # opposite rotation does not match.
    wrong = data.with_north_angle(-delta).with_wavelength_scale(1.0 / m)
    assert onp.max(onp.abs(wrong.model(_scene()) - by_hand)) > 0.05


def test_a_north_angle_of_90_degrees_turns_north_to_east():
    data = _data()
    north = System(star=PointSource(), c=PointSource(flux=0.3, ddec=5.0))
    east = System(star=PointSource(), c=PointSource(flux=0.3, dra=5.0))
    onp.testing.assert_allclose(
        onp.asarray(data.with_north_angle(90.0).model(north)),
        onp.asarray(data.model(east)),
        atol=2e-5,
    )


def test_a_uv_grid_is_dropped_and_the_direct_transform_used():
    data = _data()
    lattice = onp.stack(
        onp.meshgrid(onp.arange(1.0, 6.0), onp.arange(-2.0, 3.0)), -1
    ).reshape(-1, 2)
    gridded = data.set(["u", "v"], [lattice[:, 0] * 5.0, lattice[:, 1] * 5.0])
    gridded = eqx.tree_at(
        lambda d: d.uv_grid,
        gridded,
        find_uv_grid(gridded.u, gridded.v),
        is_leaf=lambda x: x is None,
    )
    assert gridded.uv_grid is not None
    rotated = gridded.with_north_angle(10.0)
    assert rotated.uv_grid is None
    cvis = rotated.model(_scene())
    direct = eqx.tree_at(
        lambda d: d.uv_grid, gridded, None, is_leaf=lambda x: x is None
    )
    onp.testing.assert_allclose(
        onp.asarray(cvis), onp.asarray(direct.model(_scene(10.0))), atol=2e-5
    )


def test_off_by_default_the_likelihood_is_unchanged():
    data = _data().with_model(_scene(5.0), key=jax.random.PRNGKey(1))
    plain = float(model_loglike(_scene(), data))
    assert float(model_loglike(_scene(), data, north_angle=0.0)) == plain
    assert data.with_north_angle(0.0).uv_grid is None
    assert onp.array_equal(
        onp.asarray(data.with_north_angle(0.0).u), onp.asarray(data.u)
    )
    # The data are left as they were.
    data.with_north_angle(30.0)
    assert float(model_loglike(_scene(), data)) == plain


def test_north_angle_priors_may_be_signed_and_are_sampled():
    sites = noise_sites([{}, {"north_angle": dist.Normal(0.0, 0.5)}], 2)
    assert set(sites) == {"noise[1].north_angle"}
    data = _data()
    model = numpyro_model(
        _scene(),
        {"companion.flux": dist.Uniform(0.0, 1.0)},
        [data, data],
        noise=[{}, {"north_angle": dist.Normal(0.0, 0.5)}],
    )
    from numpyro.infer.util import log_density

    values = {"companion.flux": 0.3, "noise[1].north_angle": 0.2}
    logp, trace = log_density(model, (), {}, values)
    assert "noise[1].north_angle" in trace
    expected = (
        float(model_loglike(_scene(), data))
        + float(model_loglike(_scene(), data, north_angle=0.2))
        + float(dist.Normal(0.0, 0.5).log_prob(0.2))
    )
    assert float(logp) == pytest.approx(expected, rel=1e-5)


# -- PositionData ---------------------------------------------------------

T_REF = 60500.0


def _orbit():
    from virgil.orbits import KeplerOrbit

    return KeplerOrbit(400.0, 30.0, 0.4, 60.0, 40.0, 110.0, 20.0, t_ref=T_REF)


def _positions(orbit, mjd, delta=0.0, m=1.0, sigma=0.05):
    from virgil.orbits import PositionData

    dra, ddec, _ = (onp.asarray(x, float) for x in orbit.relative(mjd))
    dra, ddec = _sky(dra, ddec, delta, m)
    cov = onp.broadcast_to(sigma**2 * onp.eye(2), (len(mjd), 2, 2))
    return PositionData(mjd, dra, ddec, cov)


def test_position_data_rotation_adds_to_the_position_angle():
    pytest.importorskip("jaxoplanet")
    mjd = T_REF + onp.linspace(0.0, 380.0, 6)
    with jax.enable_x64(True):
        orbit, data = _orbit(), _positions(_orbit(), mjd)
        dra, ddec, _ = orbit.relative(mjd)
        seen = data.model(orbit, north_angle=15.0, plate_scale=1.1)
        sep = onp.hypot(*(onp.asarray(x) for x in seen))
        pa = onp.degrees(onp.arctan2(*(onp.asarray(x) for x in seen)))
    true_pa = onp.degrees(onp.arctan2(onp.asarray(dra), onp.asarray(ddec)))
    miss = (pa - true_pa - 15.0 + 180.0) % 360.0 - 180.0
    onp.testing.assert_allclose(miss, 0.0, atol=1e-9)
    onp.testing.assert_allclose(
        sep, 1.1 * onp.hypot(onp.asarray(dra), onp.asarray(ddec)), rtol=1e-12
    )
    # Off by default: the same residuals and likelihood, to the bit.
    with jax.enable_x64(True):
        assert onp.array_equal(
            onp.asarray(data.whitened_residuals(orbit)),
            onp.asarray(data.term(lambda v: orbit)({})),
        )
        assert float(data.loglike(orbit)) == float(
            data.term(lambda v: orbit).loglike({})
        )


@pytest.mark.validates(
    "virgil.orbits.PositionData.term",
    roots=["mathematics"],
    kind="check",
)
def test_fit_recovers_a_north_angle_and_plate_scale_from_positions():
    pytest.importorskip("jaxoplanet")
    from virgil.fitting import fit
    from virgil.orbits import KeplerOrbit

    truth = _orbit()
    delta, m = 1.5, 1.012
    with jax.enable_x64(True):
        # One calibrated instrument fixes the orbit's orientation and size;
        # a second, interleaved, is rotated and magnified.
        calibrated = _positions(truth, T_REF + onp.linspace(0.0, 380.0, 8))
        other = _positions(
            truth, T_REF + onp.linspace(25.0, 405.0, 8), delta, m
        )
    names = ("period", "dt_peri", "ecc", "inc", "omega", "Omega", "a_mas")

    def orbit(v):
        return KeplerOrbit(*(v[k] for k in names), t_ref=T_REF)

    priors = {
        "period": dist.Uniform(300.0, 500.0),
        "dt_peri": dist.Uniform(-200.0, 200.0),
        "ecc": dist.Uniform(0.0, 0.9),
        "inc": dist.Uniform(0.0, 180.0),
        "omega": dist.Uniform(-360.0, 720.0),
        "Omega": dist.Uniform(-360.0, 720.0),
        "a_mas": dist.Uniform(1.0, 50.0),
        # Broad, so that the data and not the priors find them.
        "north_b": dist.Normal(0.0, 10.0),
        "scale_b": dist.Normal(1.0, 0.1),
    }
    start = {k: float(getattr(truth, k)) for k in names}
    start.update(north_b=0.0, scale_b=1.0)
    result = fit(
        lambda **kw: None,
        priors,
        (),
        init=start,
        likelihoods=[
            calibrated.term(orbit),
            other.term(orbit, north_angle="north_b", plate_scale="scale_b"),
        ],
    )
    assert result.info["method"] == "lm"
    assert result.info["converged"]
    assert float(result.values["north_b"]) == pytest.approx(delta, abs=1e-3)
    assert float(result.values["scale_b"]) == pytest.approx(m, abs=1e-5)
    assert float(result.values["Omega"]) == pytest.approx(110.0, abs=1e-2)
    assert float(result.values["a_mas"]) == pytest.approx(20.0, abs=1e-3)
    assert sum(result.info["chi2"]) < 1e-3
