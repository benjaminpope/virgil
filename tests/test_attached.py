"""Time-dependent scenes: SourceModel.at, Attached, and OIData.model in time
(design/orbit_scene_joint_fitting.md R2, §5.2.7)."""

import equinox as eqx
import jax
import numpy as onp
import pytest

pytest.importorskip("jaxoplanet")

from virgil.likelihood import model_loglike  # noqa: E402
from virgil.models import (  # noqa: E402
    Attached,
    BinaryModelCartesian,
    ModulatedGaussianRim,
    PointSource,
    System,
)
from virgil.oidata import OIData, cp_indices  # noqa: E402
from virgil.orbits import KeplerOrbit  # noqa: E402

T_REF = 60500.0
ORBIT = KeplerOrbit(
    period=730.0,
    dt_peri=-100.0,
    ecc=0.3,
    inc=50.0,
    omega=60.0,
    Omega=120.0,
    a_mas=20.0,
    t_ref=T_REF,
)
STATIONS = onp.array([[0.0, 0.0], [60.0, 5.0], [25.0, 70.0], [-40.0, 45.0]])
PAIRS = onp.array([[1, 2], [1, 3], [1, 4], [2, 3], [2, 4], [3, 4]])
TRIANGLES = onp.array([[1, 2, 3], [1, 2, 4], [1, 3, 4], [2, 3, 4]])


def _epochs(mjds, wavel=2.2e-6):
    """VLTI-like data, one four-telescope frame per epoch, epoch-major."""
    delta = STATIONS[PAIRS[:, 1] - 1] - STATIONS[PAIRS[:, 0] - 1]
    i1, i2, i3 = cp_indices(PAIRS, TRIANGLES)
    n = len(PAIRS)
    rotate = onp.deg2rad(onp.arange(len(mjds)) * 25.0)  # Earth rotation
    u = onp.concatenate(
        [delta[:, 0] * onp.cos(a) - delta[:, 1] * onp.sin(a) for a in rotate]
    )
    v = onp.concatenate(
        [delta[:, 0] * onp.sin(a) + delta[:, 1] * onp.cos(a) for a in rotate]
    )
    return OIData(
        {
            "u": u,
            "v": v,
            "wavel": wavel,
            "vis": onp.ones(u.size),
            "d_vis": onp.full(u.size, 0.01),
            "phi": onp.zeros(len(i1) * len(mjds)),
            "d_phi": onp.full(len(i1) * len(mjds), 0.01),
            "i_cps1": onp.concatenate([i1 + k * n for k in range(len(mjds))]),
            "i_cps2": onp.concatenate([i2 + k * n for k in range(len(mjds))]),
            "i_cps3": onp.concatenate([i3 + k * n for k in range(len(mjds))]),
            "mjd": onp.repeat(mjds, n),
        }
    )


def test_static_scenes_are_not_time_dependent():
    scene = System(star=PointSource(), comp=PointSource(0.1, 10.0, 5.0))
    assert not scene.time_dependent
    assert scene.at(T_REF) is scene


def test_an_attached_companion_matches_static_binaries_epoch_by_epoch():
    mjds = T_REF + onp.array([0.0, 150.0, 400.0])
    data = _epochs(mjds)
    scene = System(
        primary=PointSource(), comp=Attached(PointSource(0.1), ORBIT)
    )
    assert scene.time_dependent
    with jax.enable_x64(True):
        moving = onp.asarray(data.model(scene))
        n_vis = data.vis.size
        vis, phi = [], []
        for part, mjd in zip(data.split_by_epoch(), mjds):
            dra, ddec, _ = ORBIT.relative(mjd)
            static = onp.asarray(
                part.model(BinaryModelCartesian(dra, ddec, 0.1))
            )
            vis.append(static[: part.vis.size])
            phi.append(static[part.vis.size :])
    # The data (u, v) were made in float32, so agreement is to float32.
    assert onp.allclose(moving[:n_vis], onp.concatenate(vis), atol=1e-6)
    assert onp.allclose(moving[n_vis:], onp.concatenate(phi), atol=1e-6)
    # The companion really moved: the epochs differ.
    assert not onp.allclose(phi[0], phi[1], atol=1e-3)


def test_float32_keeps_the_times_precise():
    # The orbit's t_ref is 30 years before the data's: only offsets are
    # ever handled in float32.
    far = KeplerOrbit(
        period=730.0,
        dt_peri=11000.0 - 100.0,
        ecc=0.3,
        inc=50.0,
        omega=60.0,
        Omega=120.0,
        a_mas=20.0,
        t_ref=T_REF - 11000.0,
    )
    data = _epochs(T_REF + onp.array([0.0, 150.0]))
    scene = System(primary=PointSource(), comp=Attached(PointSource(0.1), far))
    single = onp.asarray(data.model(scene))
    with jax.enable_x64(True):
        double = onp.asarray(data.model(scene))
    assert onp.allclose(single, double, atol=2e-4)


def test_a_disc_in_the_orbital_plane_brightens_towards_the_primary():
    # §5.2.7: bound to the frame, the rim's bright side faces the primary.
    disc = Attached(
        ModulatedGaussianRim(
            3.0, 1.0, 0.0, 0.0, az_amps=0.5, az_pas=0.0, flux=0.05
        ),
        ORBIT,
        bind={
            "pa": "node_pa",
            "inc": "apparent_inc",
            "az_pas": "towards_primary",
        },
    )
    npix, fov = 128, 8.0
    xs = (onp.arange(npix) - npix / 2 + 0.5) * fov / npix
    dra = (-xs)[None, :] * onp.ones((npix, 1))  # East left
    ddec = (-xs)[:, None] * onp.ones((1, npix))  # North up
    with jax.enable_x64(True):
        for dt in (0.0, 120.0, 240.0):
            at = disc.at(T_REF + dt)
            assert float(at.pa) == pytest.approx(120.0)
            assert float(at.inc) == pytest.approx(50.0)
            centred = eqx.tree_at(lambda c: (c.dra, c.ddec), at, (0.0, 0.0))
            image = onp.asarray(centred.render(npix, fov))
            bright = onp.degrees(
                onp.arctan2(onp.sum(image * dra), onp.sum(image * ddec))
            )
            want = float(ORBIT.frame(T_REF + dt)["towards_primary"])
            assert abs((bright - want + 180) % 360 - 180) < 2.0


def test_anchors_and_offsets():
    mjd = T_REF + 50.0
    dra, ddec, _ = (float(x) for x in ORBIT.relative(mjd))
    half = Attached(PointSource(), ORBIT, anchor=0.5).at(mjd)
    assert (float(half.dra), float(half.ddec)) == pytest.approx(
        (dra / 2, ddec / 2)
    )
    origin = Attached(PointSource(), ORBIT, anchor="primary").at(mjd)
    assert (float(origin.dra), float(origin.ddec)) == (0.0, 0.0)
    skewed = Attached(
        ModulatedGaussianRim(3.0, 1.0, 0.0, 0.0),
        ORBIT,
        bind={"pa": "line_pa"},
        offsets={"pa": 5.0},
    ).at(mjd)
    line_pa = float(ORBIT.frame(mjd)["line_pa"])
    assert float(skewed.pa) == pytest.approx(line_pa + 5.0, abs=1e-4)


def test_clear_errors():
    data = _epochs(T_REF + onp.array([0.0]))
    timeless = OIData({k: v for k, v in _dict(data).items() if k != "mjd"})
    scene = System(
        primary=PointSource(), comp=Attached(PointSource(0.1), ORBIT)
    )
    with pytest.raises(ValueError, match="no times"):
        timeless.model(scene)
    with pytest.raises(ValueError, match="changes with time"):
        Attached(PointSource(), ORBIT).model(1.0, 1.0, 1e-6)
    with pytest.raises(ValueError, match="no attribute"):
        Attached(PointSource(), ORBIT, bind={"pa": "line_pa"})
    with pytest.raises(ValueError, match="frame angle"):
        Attached(
            ModulatedGaussianRim(3.0, 1.0, 0.0, 0.0),
            ORBIT,
            bind={"pa": "sideways"},
        )


def _dict(data):
    i = [onp.asarray(x) for x in (data.i_cps1, data.i_cps2, data.i_cps3)]
    return {
        "u": onp.asarray(data.u),
        "v": onp.asarray(data.v),
        "wavel": onp.asarray(data.wavel),
        "vis": onp.ones(data.u.size),
        "d_vis": onp.full(data.u.size, 0.01),
        "phi": onp.zeros(i[0].size),
        "d_phi": onp.full(i[0].size, 0.01),
        "i_cps1": i[0],
        "i_cps2": i[1],
        "i_cps3": i[2],
        "mjd": data.mjd,
    }


def test_the_likelihood_is_differentiable_in_the_orbit():
    data = _epochs(T_REF + onp.array([0.0, 150.0, 400.0]))
    truth = System(
        primary=PointSource(), comp=Attached(PointSource(0.1), ORBIT)
    )
    data = data.with_model(truth)

    def loglike(period):
        orbit = eqx.tree_at(lambda o: o.period, ORBIT, period)
        scene = System(
            primary=PointSource(), comp=Attached(PointSource(0.1), orbit)
        )
        return model_loglike(scene, data)

    grad = jax.jit(jax.grad(loglike))(730.0)
    assert onp.isfinite(float(grad))
    assert float(jax.grad(loglike)(700.0)) > 0  # back towards the truth


@pytest.mark.parametrize("bind", [{"inc": "apparent_inc"}, {"pa": "line_pa"}])
def test_gradients_through_a_disc_with_angles_bound_per_sample(bind):
    # With JAX 0.11, a vmap over samples of a disc whose bound angles vary
    # per sample failed under grad (RuntimeProgramInputMismatch).
    data = _epochs(T_REF + onp.array([0.0, 100.0]))

    def loglike(inc):
        orbit = eqx.tree_at(lambda o: o.inc, ORBIT, inc)
        disc = Attached(
            ModulatedGaussianRim(
                4.0, 1.5, 30.0, 10.0, az_amps=0.5, az_pas=0.0, flux=0.02
            ),
            orbit,
            bind=bind,
        )
        return model_loglike(System(primary=PointSource(), disc=disc), data)

    assert onp.isfinite(float(jax.grad(loglike)(50.0)))
