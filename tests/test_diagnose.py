import jax
import jax.numpy as np

from virgil.coverage import ami_grid_record
from virgil.imaging import Centroid, Diagnosis, diagnose
from virgil.models import Image, PointSource, System
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
