import jax
import jax.numpy as np
import numpy as onp
import pytest

from virgil.coverage import ami_grid_record
from virgil.fitting import fit
from virgil.imaging import (
    Laplacian,
    LogSum,
    StarletL1,
    image_priors,
    starlet,
)
from virgil.models import Image
from virgil.oidata import OIData
from virgil.scenes import gaussian_blob

NPIX, SCALE = 16, 12.0
DATA = OIData(ami_grid_record(pitch_m=0.5))


def _image(brightness, **kwargs):
    return Image.from_brightness(brightness, SCALE, **kwargs)


def test_starlet_planes_sum_to_the_image():
    image = jax.random.uniform(jax.random.PRNGKey(0), (NPIX, NPIX + 3))
    details, coarse = starlet(image, scales=3)
    assert details.shape == (3, NPIX, NPIX + 3)
    assert np.allclose(details.sum(0) + coarse, image, atol=1e-6)
    # The B3 spline keeps a flat image flat away from the edges, so the
    # finest details of a constant vanish in the interior.
    details, _ = starlet(np.ones((NPIX, NPIX)), scales=1)
    assert np.allclose(details[0, 2:-2, 2:-2], 0.0, atol=1e-6)
    with pytest.raises(ValueError, match="positive integer"):
        starlet(image, scales=0)


def test_starlet_smoothing_is_the_b3_spline():
    delta = np.zeros((NPIX, NPIX)).at[8, 8].set(1.0)
    _, coarse = starlet(delta, scales=1)
    h = onp.array([1, 4, 6, 4, 1]) / 16
    expected = onp.zeros((NPIX, NPIX))
    expected[6:11, 6:11] = onp.outer(h, h)
    assert onp.allclose(coarse, expected, atol=1e-7)


def test_laplacian_matches_the_five_point_stencil():
    b = jax.random.uniform(jax.random.PRNGKey(1), (5, 6))
    image = _image(b)
    padded = onp.pad(onp.asarray(image.brightness), 2)
    lap = (
        padded[:-2, 1:-1]
        + padded[2:, 1:-1]
        + padded[1:-1, :-2]
        + padded[1:-1, 2:]
        - 4 * padded[1:-1, 1:-1]
    )
    reg = Laplacian(3.0)
    assert np.isclose(reg.value(image), 3.0 * onp.sum(lap**2), rtol=1e-5)
    assert np.isclose(
        0.5 * np.sum(reg.residuals(image) ** 2), reg.value(image)
    )


def test_starlet_l1_matches_its_definition_and_penalises_noise():
    blob = _image(gaussian_blob(NPIX, SCALE, 20.0))
    reg = StarletL1(2.0, scales=3, epsilon=1e-2)
    details, _ = starlet(blob.brightness, 3)
    eps = 1e-2 / NPIX**2
    expected = 2.0 * np.sum(np.sqrt(details**2 + eps**2))
    assert np.isclose(reg.value(blob), expected, rtol=1e-5)
    # Zero-mean noise, on a pedestal that keeps the pixels positive, adds
    # fine detail without changing the flux.
    b = gaussian_blob(NPIX, SCALE, 20.0)
    b = b + 0.2 * b.max()
    noise = jax.random.uniform(jax.random.PRNGKey(2), (NPIX, NPIX)) - 0.5
    noise = noise - noise.mean()
    smooth, noisy = _image(b), _image(b + 0.1 * b.max() * noise)
    assert reg.value(noisy) > reg.value(smooth)
    with pytest.raises(ValueError, match="positive integer"):
        StarletL1(1.0, scales=1.5)


def test_log_sum_counts_bright_pixels():
    flat = _image(np.ones((NPIX, NPIX)))
    point = _image(np.zeros((NPIX, NPIX)).at[8, 8].set(1.0), floor=1e-12)
    reg = LogSum(1.0, epsilon=1e-2)
    # A flat image has every pixel at the mean: N log(1 + 1/ε).
    assert np.isclose(reg.value(flat), NPIX**2 * np.log1p(100.0), rtol=1e-5)
    # One bright pixel costs log(1 + N/ε); the dark ones almost nothing.
    assert np.isclose(reg.value(point), np.log1p(NPIX**2 / 1e-2), rtol=1e-3)
    # The mean is over the support only.
    support = np.zeros((NPIX, NPIX), bool).at[:4, :4].set(True)
    inside = _image(np.ones((NPIX, NPIX)), support=support)
    assert np.isclose(reg.value(inside), 16 * np.log1p(100.0), rtol=1e-5)


@pytest.mark.filterwarnings("ignore:fit\\(method=:RuntimeWarning")
@pytest.mark.parametrize(
    "regulariser, method",
    [
        (Laplacian(1e2), "lm"),
        (StarletL1(1e-1), "lbfgs"),
        (LogSum(1e-3), "lbfgs"),
    ],
)
def test_sparsity_regularisers_fit(regulariser, method):
    truth = _image(gaussian_blob(NPIX, SCALE, 25.0))
    data = DATA.with_model(truth, key=jax.random.PRNGKey(4))
    start = _image(np.ones((NPIX, NPIX)))
    result = fit(
        start,
        image_priors(start),
        data,
        [regulariser],
        method=method,
        max_steps=20,
    )
    assert result.info["method"] == method
    assert onp.isfinite(result.info["loss"])
    assert np.all(np.isfinite(result.model.brightness))
