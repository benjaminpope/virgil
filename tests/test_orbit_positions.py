"""Per-dataset companion positions on the scale-marginalized surface
(``epoch_positions``, ``marginal_loglike``) and scale-aware rankings
(``rank_orbits(scales=...)``): virgil#268, PR A of the automatic-orbits
design.

Every case runs on two arrays: three VLTI telescopes (one closure phase
per frame and channel, uncorrelated, with a von Mises likelihood) and the
four UTs (correlated closure phases). Noise is drawn with NumPy, so that
the data do not depend on the JAX version.
"""

import warnings

import jax
import jax.numpy as jnp
import numpy as onp
import numpyro.distributions as dist
import pytest
from scipy.ndimage import maximum_filter

from virgil._precision import cast_tree
from virgil.coverage import VLTI_UTS, vlti_oidata
from virgil.epochs import (
    Epochs,
    epoch_positions,
    marginal_loglike,
    rank_orbits,
)
from virgil.fitting import fit
from virgil.likelihood import model_loglike
from virgil.models import BinaryModelCartesian
from virgil.simulate import simulate

AXIS = onp.arange(-15.0, 15.01, 0.5)
GRID = {"dra": AXIS, "ddec": AXIS, "flux": [0.25, 0.5, 1.0]}
TRUTH = (4.0, -2.0, 0.6)
ARRAYS = {"3T": VLTI_UTS[:3], "4T": VLTI_UTS}


def _night(
    stations,
    seed,
    truth=TRUTH,
    *,
    s_vis=3.0,
    s_phi=3.0,
    sigma_v2=0.05,
    sigma_cp_deg=6.0,
    hours=(-1.0, 0.0, 1.0),
    wavelengths=(2.1e-6, 2.3e-6),
    scene=None,
    mjd=60000.0,
):
    """A night of V² and closure phases whose noise is ``s_vis`` and
    ``s_phi`` times the quoted errors (drawn with NumPy)."""
    template = vlti_oidata(
        stations=stations,
        hour_angles_h=hours,
        wavelengths_m=onp.asarray(wavelengths),
        sigma_v2=sigma_v2,
        sigma_cp_deg=sigma_cp_deg,
        nights_mjd=[mjd],
    )
    with jax.enable_x64(True):
        model = simulate(
            BinaryModelCartesian(*truth) if scene is None else scene, template
        )
    rng = onp.random.default_rng(seed)
    vis, d_vis = onp.asarray(model.vis), onp.asarray(model.d_vis)
    phi, d_phi = onp.asarray(model.phi), onp.asarray(model.d_phi)
    vis = vis + s_vis * d_vis * rng.standard_normal(vis.shape)
    phi = phi + s_phi * d_phi * rng.standard_normal(phi.shape)
    return model.set(["vis", "phi"], [vis, phi])


def _scaled(data, vis=1.0, phi=1.0):
    return data.set(["d_vis", "d_phi"], [data.d_vis * vis, data.d_phi * phi])


def _positions(data, **options):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return epoch_positions(Epochs({"night": data}), GRID, **options)


# The seeds of a night with two peaks 0.6 < Δm < 1 apart whose errors are
# quoted 3× too small, found by a scan of seeds 0-140.
AMBIGUOUS = {
    "3T": dict(seed=27),
    "4T": dict(seed=65, sigma_v2=0.1, sigma_cp_deg=12.0),
}


# ---------------------------------------------------------------------------
# 1. Per-block invariance
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("array", ARRAYS)
def test_rescaling_one_dataset_or_only_its_closure_phases_changes_nothing(
    array,
):
    data = _night(ARRAYS[array], 1)
    base = _positions(data)
    grid_base = _positions(data, refine=False)
    for factor in ({"vis": 3.0, "phi": 3.0}, {"phi": 3.0}):
        scaled = _scaled(data, **factor)
        other = _positions(scaled)
        grid_other = _positions(scaled, refine=False)
        # Grid quantities: the same best point, and the same gap to the
        # same rival.
        assert grid_other.dra == grid_base.dra
        assert grid_other.ddec == grid_base.ddec
        assert grid_other.flux == grid_base.flux
        onp.testing.assert_allclose(
            other.gap_marginal, base.gap_marginal, rtol=1e-9
        )
        # Refined values, to the fit's tolerance.
        onp.testing.assert_allclose(other.dra, base.dra, atol=1e-4)
        onp.testing.assert_allclose(other.ddec, base.ddec, atol=1e-4)
        onp.testing.assert_allclose(other.flux, base.flux, atol=1e-4)
        onp.testing.assert_allclose(other.cov, base.cov, rtol=1e-3)
        onp.testing.assert_allclose(
            other.scale[0]["phi_scale"] * factor["phi"],
            base.scale[0]["phi_scale"],
            rtol=1e-6,
        )
    # With every block scaled, the quoted-error gap (same best point and
    # rival) is divided by 9, while the marginal one is unchanged.
    every = _positions(_scaled(data, 3.0, 3.0), refine=False)
    onp.testing.assert_allclose(every.gap, base.gap / 9.0, rtol=1e-6)


def test_the_covariance_is_the_curvature_of_the_marginal_surface():
    data = _night(VLTI_UTS, 1)
    found = _positions(data)
    x = onp.array([found.dra[0], found.ddec[0], found.flux[0]])
    with jax.enable_x64(True):
        d64 = cast_tree(data, "float64")

        def negative(v):
            return -marginal_loglike(BinaryModelCartesian(*v), d64)

        hess = onp.asarray(jax.hessian(negative)(jnp.asarray(x)))
    onp.testing.assert_allclose(
        found.cov[0], onp.linalg.inv(hess)[:2, :2], rtol=1e-6
    )
    # It is not the quoted-error covariance times ŝ² of one block.
    assert onp.all(onp.linalg.eigvalsh(found.cov[0]) > 0)


# ---------------------------------------------------------------------------
# 2. The fix, pinned; 3. raw χ²/N
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("array", ARRAYS)
def test_an_ambiguous_night_with_underestimated_errors_is_not_decisive(array):
    data = _night(ARRAYS[array], **AMBIGUOUS[array])
    found = _positions(data, refine=False)
    assert found.gap[0] > 5.0  # decisive on the quoted errors (main)
    assert 0.6 < found.gap_marginal[0] < 1.0
    assert not found.decisive(5.0)[0]
    # So start_from_positions would not seed from it.
    with pytest.raises(ValueError, match="No dataset is decisive"):
        found.positions(t_ref=60000.0, min_gap=5.0)


@pytest.mark.parametrize("array", ARRAYS)
def test_raw_chi2_per_dof_is_the_square_of_the_noise_scale(array):
    s = 3.0
    data = _night(ARRAYS[array], 11, s_vis=s, s_phi=s, hours=(-2, -1, 0, 1, 2))
    with pytest.warns(UserWarning, match="exceeds 4"):
        found = epoch_positions(Epochs({"night": data}), GRID)
    raw = found.chi2_raw[0]
    nu = data.n_independent
    # E[χ²] = (ν - 3) s² at the fitted point, with scatter s² √(2ν).
    expected = (nu - 3) / nu * s**2
    assert abs(raw["all"] - expected) < 4.0 * s**2 * onp.sqrt(2.0 / nu)
    assert set(raw) == {"vis", "phi", "all"}
    assert found.scale[0]["vis_scale"] == pytest.approx(
        onp.sqrt(raw["vis"]), rel=1e-12
    )
    # Correct errors give no warning.
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        epoch_positions(
            Epochs({"night": _scaled(data, s, s)}), GRID, refine=False
        )


# ---------------------------------------------------------------------------
# 4. Rankings under the rescaling of one dataset
# ---------------------------------------------------------------------------


def test_rescaling_one_dataset_leaves_the_marginal_ranking_unchanged():
    pytest.importorskip("jaxoplanet")
    from virgil.models import OrbitalBinary
    from virgil.orbits import KeplerOrbit

    truth = dict(
        period=700.0, dt_peri=-120.0, ecc=0.3, inc=50.0, omega=60.0,
        Omega=120.0, a_mas=8.0, t_ref=60000.0,
    )  # fmt: skip
    times = (60000.0, 60150.0)
    scene = OrbitalBinary(KeplerOrbit(**truth), 0.3)
    nights = [
        _night(
            VLTI_UTS,
            20 + k,
            s_vis=1.0,
            s_phi=1.0,
            mjd=t,
            hours=(0.0,),
            scene=scene,
        )  # fmt: skip
        for k, t in enumerate(times)
    ]
    rng = onp.random.default_rng(3)
    orbits = [
        KeplerOrbit(
            **{
                **truth,
                "Omega": 120.0 + rng.normal(0, 4.0),
                "inc": 50.0 + rng.normal(0, 4.0),
                "period": 700.0 + rng.normal(0, 15.0),
            }
        )
        for _ in range(24)
    ]

    def model(orbit):
        return OrbitalBinary(orbit, 0.3)

    def ranked(first, scales):
        epochs = Epochs({"a": first, "b": nights[1]})
        return rank_orbits(model, epochs, orbits, scales=scales)

    rescaled = _scaled(nights[0], 1 / 3, 1 / 3)
    marginal = ranked(nights[0], "marginal")
    again = ranked(rescaled, "marginal")
    onp.testing.assert_array_equal(again.order, marginal.order)
    onp.testing.assert_allclose(
        onp.diff(again.loglike), onp.diff(marginal.loglike), atol=1e-8
    )
    quoted = ranked(nights[0], "quoted")
    assert not onp.array_equal(ranked(rescaled, "quoted").order, quoted.order)
    with pytest.warns(FutureWarning, match="scales='marginal'"):
        rank_orbits(model, Epochs({"a": nights[0]}), orbits[:2])


# ---------------------------------------------------------------------------
# 5. Weak closure phases
# ---------------------------------------------------------------------------


def test_weak_closure_phases_use_the_bounded_numerical_marginal():
    # sσ ≈ 0.7 rad: closure phases quoted at 0.23 rad, scattered 3× more.
    sigma = onp.rad2deg(0.7 / 3.0)
    data = _night(VLTI_UTS[:3], 5, sigma_cp_deg=sigma, hours=(-1.0, 1.0))
    axis = onp.arange(-12.0, 12.01, 1.0)
    xx, yy, ff = onp.meshgrid(axis, axis, [0.3, 0.6], indexing="ij")
    points = jnp.asarray(onp.stack([xx.ravel(), yy.ravel(), ff.ravel()], -1))
    log_s = jnp.linspace(-onp.log(10.0), onp.log(10.0), 161)
    with jax.enable_x64(True):
        d64 = cast_tree(data, "float64")

        def bounded(x):
            return marginal_loglike(BinaryModelCartesian(*x), d64, s_max=10.0)

        def profile(x):
            # The exact likelihood (von Mises closure phases) profiled over
            # each block's scale; the blocks separate, so profile each on
            # its own and remove the doubly counted base.
            m = BinaryModelCartesian(*x)

            def at(v, p):
                return model_loglike(m, d64, vis_scale=v, phi_scale=p)

            base = at(1.0, 1.0)
            vis = jax.vmap(lambda u: at(jnp.exp(u), 1.0))(log_s).max()
            phi = jax.vmap(lambda u: at(1.0, jnp.exp(u)))(log_s).max()
            return vis + phi - base

        score = onp.asarray(jax.lax.map(bounded, points, batch_size=256))
        exact = onp.asarray(jax.lax.map(profile, points, batch_size=256))
    assert onp.all(onp.isfinite(score))
    assert onp.argmax(score) == onp.argmax(exact)
    # And the positions of a bounded marginal are finite and on the grid.
    found = _positions(data, s_max=10.0)
    assert onp.isfinite(found.gap_marginal[0]) and onp.all(
        onp.isfinite(found.cov)
    )


# ---------------------------------------------------------------------------
# 6. Per-night ranking against an independent fit
# ---------------------------------------------------------------------------


@jax.jit
def _quoted_loglike(points, data):
    return jax.vmap(
        lambda x: model_loglike(BinaryModelCartesian(x[0], x[1], x[2]), data)
    )(points)


def _reference(data, starts=(), half=16.0, step=0.25, n_peaks=10):
    """The peaks of an independent per-night fit, best first: local maxima
    of its own finer grid on the quoted errors (fluxes 0.05-1), each
    refined by ``fit`` with a free ``vis_scale`` and ``phi_scale``
    (Jeffreys priors, so the fit maximizes the exact likelihood profiled
    over the scales), ranked by loss. ``(loss, dra, ddec, flux)``."""
    axis = onp.arange(-half, half + 1e-9, step)
    fluxes = onp.geomspace(0.05, 1.0, 8)
    xx, yy, ff = onp.meshgrid(axis, axis, fluxes, indexing="ij")
    points = onp.stack([xx.ravel(), yy.ravel(), ff.ravel()], -1)
    with jax.enable_x64(True):
        loglike = onp.asarray(
            _quoted_loglike(jnp.asarray(points), cast_tree(data, "float64"))
        ).reshape(xx.shape)
    by_position = loglike.max(2)
    peaks = onp.argwhere(
        by_position == maximum_filter(by_position, size=5, mode="nearest")
    )
    peaks = sorted(peaks, key=lambda ij: -by_position[tuple(ij)])[:n_peaks]
    starts = [
        (axis[i], axis[j], fluxes[onp.argmax(loglike[i, j])]) for i, j in peaks
    ] + list(starts)
    priors = {
        "dra": dist.Uniform(-half - 1, half + 1),
        "ddec": dist.Uniform(-half - 1, half + 1),
        "flux": dist.LogUniform(1e-3, 1.0),
    }
    noise = {
        "vis_scale": dist.LogUniform(0.05, 50.0),
        "phi_scale": dist.LogUniform(0.05, 50.0),
    }
    found = []
    for x, y, f in starts:
        result = fit(
            BinaryModelCartesian(x, y, min(f, 0.95)), priors, data, noise=noise
        )
        v = result.values
        found.append(
            (float(result.info["loss"]), float(v["dra"]), float(v["ddec"]),
             float(v["flux"]))
        )  # fmt: skip
    return sorted(f for f in found if onp.isfinite(f[0]))


def _motion_scene():
    from virgil.models import OrbitalBinary
    from virgil.orbits import KeplerOrbit

    # A face-on circular orbit of 4.5 mas and 3.4 days: 0.7 mas, about
    # 0.2 λ/B, over the two hours of the night.
    orbit = KeplerOrbit(
        period=3.4, dt_peri=0.0, ecc=0.0, inc=0.0, omega=0.0, Omega=0.0,
        a_mas=4.5, t_ref=60000.0,
    )  # fmt: skip
    return OrbitalBinary(orbit, 0.4)


CROSS_CHECK = {
    "3T, two near-equal peaks, errors 3x too small": ("3T", AMBIGUOUS["3T"]),
    "4T, two near-equal peaks, errors 3x too small": ("4T", AMBIGUOUS["4T"]),
    "4T, closure phases alone 3x too small": (
        "4T",
        dict(seed=3, s_vis=1.0, s_phi=3.0),
    ),
    "4T, flux near the bound": ("4T", dict(seed=5, truth=(4.0, -2.0, 0.95))),
    "4T, peak near the grid edge": (
        "4T",
        dict(seed=6, truth=(14.2, -6.0, 0.4)),
    ),
    "4T, motion of 0.2 lambda/B within the night": (
        "4T",
        dict(seed=7, scene="motion"),
    ),
}


@pytest.mark.parametrize("case", CROSS_CHECK)
def test_the_per_night_ranking_agrees_with_an_independent_fit(case):
    array, options = CROSS_CHECK[case]
    options = dict(options)
    if options.get("scene") == "motion":
        pytest.importorskip("jaxoplanet")
        options["scene"] = _motion_scene()
    data = _night(ARRAYS[array], **options)
    epochs = Epochs({"night": data})
    resolution = epochs.resolution_mas
    found = _positions(data)
    best = onp.array([found.dra[0], found.ddec[0]])
    reference = _reference(
        data, starts=[(*best, found.flux[0]), (*-best, found.flux[0])]
    )
    loss, *top = reference[0]
    top = onp.array(top[:2])
    if reference[0][3] > 0.99:
        # At flux 1 the companion at r and at -r is one scene: the peak's
        # sign is undetermined, and the marginal gap says so.
        assert min(onp.hypot(*(best - top)), onp.hypot(*(best + top))) < 0.05
        assert found.gap_marginal[0] < 1e-6
        return
    # The same best peak, to the fits' tolerance ...
    assert onp.hypot(*(best - top)) < 0.05, (best, reference[:3])
    # ... and a gap like the independent fit's: the loss of its best rival
    # more than λ/B away, less that of its best.
    rival = next(
        r
        for r in reference
        if onp.hypot(r[1] - top[0], r[2] - top[1]) > resolution
    )
    gap = rival[0] - loss
    assert abs(found.gap_marginal[0] - gap) < max(0.6, 0.3 * gap), (
        found.gap_marginal[0],
        gap,
    )
    if gap < 1.0:
        assert found.gap[0] > 5.0  # main called it decisive
    if options.get("scene") is not None:
        # A static fit of a night with 0.2 λ/B of motion agrees; the motion
        # is visible in the night's spread of times.
        assert epochs.spread_days[0] > 0.0
