import equinox as eqx
import jax
import jax.numpy as np
import numpy as onp
import pytest

from virgil.coverage import ami_grid_record, vlti_oidata
from virgil.fields import GaussianField
from virgil.imaging import (
    Centroid,
    Diagnosis,
    chi2_by_observable,
    diagnose,
    residual_image,
    short_baseline_residual,
)
from virgil.likelihood import whitened_residuals
from virgil.models import GaussianDisk, Image, PointSource, System
from virgil.oidata import OIData
from virgil.scenes import gaussian_blob

NPIX, SCALE = 24, 60.0
RECORD = ami_grid_record(pitch_m=0.5)


def _scene(
    scale=SCALE,
    npix=NPIX,
    star=True,
    sigma=100.0,
    offset=(150.0, -100.0),
    **kw,
):
    blob = gaussian_blob(npix, scale, sigma, *offset)
    env = Image.from_brightness(blob, scale, flux=0.1, **kw)
    return System(star=PointSource(), env=env) if star else env


TRUTH = _scene()
DATA = OIData(RECORD).with_model(TRUTH, key=jax.random.PRNGKey(3))


def _only(diagnosis, keyword):
    assert len(diagnosis.warnings) == 1, diagnosis
    assert keyword in diagnosis.warnings[0]


def test_a_well_fitted_star_and_blob_has_no_warnings():
    diagnosis = diagnose(TRUTH, DATA)
    assert diagnosis.warnings == []
    assert 0.8 < diagnosis.checks["chi2_red"][0] < 1.2
    assert "chi2_red" in str(diagnosis) and "No warnings" in str(diagnosis)
    assert isinstance(diagnosis, Diagnosis)


def test_a_sequence_of_datasets_is_checked_one_by_one():
    assert len(diagnose(TRUTH, [DATA, DATA]).checks["chi2_red"]) == 2


def test_underfitting_warns():
    wrong = _scene().set("env.flux", 0.5)
    _only(diagnose(wrong, DATA), "under-fits")


def test_overfitting_warns():
    noiseless = OIData(RECORD).with_model(TRUTH)
    _only(diagnose(TRUTH, noiseless), "over-fits")


def test_coarse_pixels_warn():
    scene = _scene(scale=120.0, npix=24, sigma=200.0)
    data = OIData(RECORD).with_model(scene, key=jax.random.PRNGKey(3))
    _only(diagnose(scene, data), "Nyquist")


def test_flux_at_the_edge_warns():
    edge = np.zeros((NPIX, NPIX)).at[0, 0].set(1.0)
    scene = System(
        star=PointSource(), env=Image.from_brightness(edge, SCALE, flux=0.1)
    )
    data = OIData(RECORD).with_model(scene, key=jax.random.PRNGKey(3))
    _only(diagnose(scene, data), "outer 2 pixels")


def test_an_unanchored_image_warns_unless_a_prior_or_star_fixes_it():
    scene = _scene(star=False, sigma=30.0, offset=(20.0, 10.0))
    data = OIData(RECORD).with_model(scene, key=jax.random.PRNGKey(3))
    diagnosis = diagnose(scene, data)
    _only(diagnosis, "position")
    assert not diagnosis.checks["anchored"]
    assert diagnose(scene, data, [Centroid(10.0)]).checks["anchored"]
    assert diagnose(scene, data, [Centroid(10.0)]).warnings == []


def test_the_centroid_is_reported():
    x, y = diagnose(TRUTH, DATA).checks["centroid_mas"][0]
    assert abs(x - 150.0) < 5.0 and abs(y + 100.0) < 5.0


def test_a_symmetric_image_leaves_the_orientation_unconstrained():
    scene = System(
        star=PointSource(),
        env=Image.from_brightness(
            gaussian_blob(NPIX, SCALE, 100.0), SCALE, flux=0.1
        ),
    )
    data = OIData(RECORD).with_model(scene, key=jax.random.PRNGKey(3))
    _only(diagnose(scene, data), "orientation")


def test_wrapped_phases_warn():
    # A faint star: |V| falls below 0.05 where the blob dominates.
    scene = TRUTH.set("star.flux", 1e-3)
    data = OIData(RECORD).with_model(scene, key=jax.random.PRNGKey(3))
    diagnosis = diagnose(scene, data)
    assert diagnosis.checks["phase_regime"][0] > 0.0
    assert any("wrapped" in w for w in diagnosis.warnings)


def test_a_mirrored_fit_warns():
    # A model that is the truth rotated by 180 degrees fits the data worse
    # than its own rotation, the truth.
    mirrored = TRUTH.set(
        "env",
        Image(
            TRUTH.env.log_brightness[::-1, ::-1],
            TRUTH.env.pixel_scale_mas,
            flux=TRUTH.env.flux,
        ),
    )
    diagnosis = diagnose(mirrored, DATA)
    assert diagnosis.checks["flip_dchi2"] < -1.0
    assert any("mirrored" in w for w in diagnosis.warnings)


def test_diagnose_accepts_the_old_regularisers_keyword():
    import pytest

    new = diagnose(TRUTH, DATA, regularizers=[Centroid(10.0)])
    with pytest.warns(FutureWarning, match="regularisers="):
        old = diagnose(TRUTH, DATA, regularisers=[Centroid(10.0)])
    assert old.warnings == new.warnings


VLTI = vlti_oidata(hour_angles_h=(-2.0, 0.0, 2.0), wavelengths_m=[3.5e-6])


def _disk(flux=0.4):
    return System(star=PointSource(), env=GaussianDisk(4.0, flux=flux))


VTRUTH = _disk()
VDATA = VLTI.with_model(VTRUTH, key=jax.random.PRNGKey(1))


def test_chi2_by_observable_splits_the_total_and_counts_the_data():
    out = chi2_by_observable(VTRUTH, VDATA)
    assert set(out) == {"chi2", "n", "chi2_red"}
    assert out["n"]["total"] == VDATA.n_independent
    assert out["n"]["vis"] + out["n"]["phi"] == out["n"]["total"]
    parts = out["chi2"]["vis"] + out["chi2"]["phi"]
    assert parts == pytest.approx(out["chi2"]["total"])
    for kind in ("vis", "phi"):  # the truth fits each kind to its noise
        assert 0.3 < out["chi2_red"][kind] < 3.0
    assert "n_gamma" not in out


def test_chi2_by_observable_sees_which_observable_is_wrong():
    right = chi2_by_observable(VTRUTH, VDATA)["chi2_red"]
    wrong = chi2_by_observable(_disk(flux=0.9), VDATA)["chi2_red"]
    assert wrong["vis"] > 3 * right["vis"]


def test_n_gamma_counts_the_constrained_latents():
    n = 12
    data = VLTI
    template = onp.asarray(GaussianDisk(3.0).render(n, float(n)))

    def scene(sigma):
        field = GaussianField(onp.zeros((n, n)), sigma, 2.0, mean=template)
        return System(star=PointSource(), env=Image(field, 1.0, flux=0.4))

    ngamma = {
        s: chi2_by_observable(scene(s), data, path="env")["n_gamma"]
        for s in (0.01, 1.0)
    }
    assert 0.0 <= ngamma[0.01] < ngamma[1.0] <= n * n


def _offset_shortest(data, n, sigmas):
    """``data`` with its ``n`` shortest-B/λ visibilities raised by ``sigmas``."""
    rho = onp.hypot(onp.asarray(data.u), onp.asarray(data.v)) / onp.asarray(
        data.wavel
    )
    short = onp.argsort(rho.ravel(), kind="stable")[:n]
    vis = onp.array(data.vis, float)
    vis[short] += sigmas * onp.asarray(data.d_vis)[short]
    return eqx.tree_at(lambda d: d.vis, data, np.asarray(vis))


def test_short_baseline_residual_is_unbiased_for_the_truth():
    mean, error = short_baseline_residual(VTRUTH, VDATA)
    assert error == pytest.approx(1 / onp.sqrt(round(0.2 * VDATA.vis.size)))
    assert abs(mean) < 4 * error
    assert diagnose(VTRUTH, VDATA).checks[
        "short_baseline_mean"
    ] == pytest.approx(mean)


def test_short_baseline_residual_selects_the_shortest_baselines():
    # Only the 4 shortest baselines (20% of 18) are 10 sigma off: the data
    # exceed the model there, so the residual (model - data) is -10 sigma.
    shifted = _offset_shortest(VDATA, 4, 10.0)
    base, error = short_baseline_residual(VTRUTH, VDATA)
    mean, _ = short_baseline_residual(VTRUTH, shifted)
    assert mean - base == pytest.approx(-10.0, abs=0.1)
    # Over every baseline the offset is diluted by 4 / 18.
    all_base, _ = short_baseline_residual(VTRUTH, VDATA, quantile=1.0)
    all_mean, _ = short_baseline_residual(VTRUTH, shifted, quantile=1.0)
    assert all_mean - all_base == pytest.approx(-10.0 * 4 / 18, abs=0.1)


def test_short_baseline_residual_refuses_gains_and_operators():
    with pytest.raises(ValueError, match="gains"):
        short_baseline_residual(VTRUTH, VDATA.with_gains(telescope=0.05))
    with pytest.raises(ValueError, match="vis_mat"):
        short_baseline_residual(TRUTH, DATA)
    assert diagnose(TRUTH, DATA).checks["short_baseline_mean"] is None


def test_short_baseline_residual_with_every_baseline_is_the_mean_residual():
    mean_all, _ = short_baseline_residual(VTRUTH, VDATA, quantile=1.0)
    r = onp.asarray(whitened_residuals(VTRUTH, VDATA))[: VDATA.vis.size]
    assert mean_all == pytest.approx(float(r.mean()), abs=1e-4)
    with pytest.raises(ValueError, match="quantile"):
        short_baseline_residual(VTRUTH, VDATA, quantile=0.0)


def test_diagnose_adds_the_observable_keys_without_changing_the_old_ones():
    checks = diagnose(VTRUTH, VDATA).checks
    assert set(checks["chi2_red_by_observable"][0]) == {"vis", "phi"}
    assert "chi2_red" in checks and "anchored" in checks


def test_the_residual_image_points_at_the_missing_flux():
    star_only = System(
        star=PointSource(),
        env=Image.from_brightness(
            gaussian_blob(NPIX, SCALE, 300.0, 0.0, 0.0), SCALE, flux=1e-6
        ),
    )
    # Truth has a blob at (150, -100) mas that the model lacks.
    residual = residual_image(star_only, DATA, NPIX, SCALE)
    assert residual.shape == (NPIX, NPIX)
    blob = onp.asarray(gaussian_blob(NPIX, SCALE, 100.0, *(150.0, -100.0)))
    inside = blob > 0.5 * blob.max()
    assert residual[inside].mean() > 0.0
    assert residual[inside].mean() > 5 * abs(residual[~inside].mean())
    # The V^2 map is centrosymmetric; the phases pick the right side.
    mirror = onp.asarray(gaussian_blob(NPIX, SCALE, 100.0, -150.0, 100.0))
    assert (
        residual[inside].mean() > residual[mirror > 0.5 * mirror.max()].mean()
    )
    # The truth leaves a far smaller residual map than the star alone.
    fitted = residual_image(TRUTH, DATA, NPIX, SCALE)
    assert onp.abs(fitted).max() < 0.5 * residual.max()


def test_chi2_by_observable_warns_when_gains_mix_the_observables():
    with pytest.warns(UserWarning, match="calibration gains"):
        out = chi2_by_observable(VTRUTH, VDATA.with_gains(telescope=0.05))
    assert out["n"]["total"] > 0
