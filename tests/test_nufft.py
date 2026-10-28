"""The nufftax transform (``ft="nufft"``) against the exact sum."""

import jax
import jax.numpy as np
import numpy as onp
import pytest

from virgil import _geometry
from virgil._geometry import (
    UVGrid,
    _nufft,
    _use_nufft,
    set_ft_defaults,
    image_visibilities,
)
from virgil.likelihood import model_loglike
from virgil.models import FlaredDiskHG, Image
from tests._shared import vlti_night
from tests.test_image_model import MAS2RAD, _dft_reference

WAVEL = 4.8e-6


def _problem(shape, n=200, seed=0, max_freq=2e6):
    rng = onp.random.default_rng(seed)
    image = rng.uniform(0.0, 1.0, shape)
    image /= image.sum()
    uu, vv = rng.uniform(-max_freq, max_freq, (2, n))
    return image, uu, vv


@pytest.mark.parametrize("x64", [False, True])
@pytest.mark.parametrize("shape", [(17, 17), (16, 16), (12, 15), (128, 128)])
def test_nufft_matches_the_direct_sum(shape, x64):
    n = 50 if shape[0] > 100 else 200
    image, uu, vv = _problem(shape, n=n)
    expected = _dft_reference(image, uu, vv, 3.0)
    with jax.enable_x64(x64):
        got = _nufft(np.asarray(image), np.asarray(uu), np.asarray(vv), 3.0)
        # The image sums to one: absolute error is relative to V(0).
        error = onp.max(onp.abs(onp.asarray(got) - expected))
    # eps is 1e-5 in float32, as accurate as the float32 DFT.
    assert error < (1e-10 if x64 else 5e-5)


@pytest.mark.parametrize("ft", ["dft", "nufft"])
@pytest.mark.parametrize("x64", [False, True])
@pytest.mark.parametrize("shape", [(16, 16), (12, 15), (17, 17)])
def test_long_baselines_beyond_the_nufft_period_agree(shape, x64, ft):
    # |2π s u| up to ~27 rad: the even-axis half-pixel phase is 4π-periodic,
    # and float32 phases are good to ~1e-6 rad.
    image, uu, vv = _problem(shape, max_freq=3e7)
    scale = 30.0
    assert onp.max(onp.abs(2 * onp.pi * scale * MAS2RAD * uu)) > 3 * onp.pi
    expected = _dft_reference(image, uu, vv, scale)
    with jax.enable_x64(x64):
        args = (np.asarray(image), np.asarray(uu), np.asarray(vv), scale)
        got = onp.asarray(image_visibilities(*args, ft=ft))
    assert got.dtype == (onp.complex128 if x64 else onp.complex64)
    assert onp.max(onp.abs(got - expected)) < (1e-10 if x64 else 2e-5)


def test_gradient_matches_the_dft_gradient():
    image, uu, vv = _problem((16, 15), n=60)
    weights = np.asarray(onp.random.default_rng(3).normal(size=60))

    def objective(ft):
        def f(b):
            vis = image_visibilities(b, uu, vv, 3.0, ft)
            return np.sum(weights * np.abs(vis) ** 2)

        return jax.grad(f)(np.asarray(image))

    dft = objective("dft")
    assert np.max(np.abs(objective("nufft") - dft)) < 5e-5 * np.max(
        np.abs(dft)
    )


def test_forward_mode_matches_the_dft():
    # CLEAN, the image Jacobian, LM and Gauss-Newton differentiate forward.
    image, uu, vv = _problem((16, 15), n=60)
    image = np.asarray(image)
    tangent = np.asarray(onp.random.default_rng(4).normal(size=image.shape))

    def run(ft):
        def f(b):
            return image_visibilities(b, uu, vv, 3.0, ft)

        return jax.jvp(f, (image,), (tangent,))[1], jax.jacfwd(f)(image)

    for got, expected in zip(run("nufft"), run("dft")):
        assert np.max(np.abs(got - expected)) < 5e-4  # eps 1e-5 of |V| ~ 13


@pytest.mark.xfail(
    strict=True,
    reason=(
        "nufftax 0.6.1's batching rule handles one batch dimension at axis 0, "
        "and jax.hessian is two nested vmaps"
    ),
)
def test_hessian_through_the_nufft():
    image, uu, vv = _problem((8, 8), n=10)

    def f(b):
        return np.sum(np.abs(image_visibilities(b, uu, vv, 3.0, "nufft")) ** 2)

    jax.hessian(f)(np.asarray(image))


def test_use_nufft_decisions(monkeypatch):
    monkeypatch.setattr(jax, "default_backend", lambda: "gpu")
    assert _use_nufft((256, 256), 3_000)
    assert not _use_nufft((256, 256), 2_999)
    assert not _use_nufft((1024, 1024), 100)  # the DFT is 2x faster here
    assert not _use_nufft((64, 64), 10**6)  # 64² never wins on a GPU
    monkeypatch.setattr(jax, "default_backend", lambda: "cpu")
    assert _use_nufft((64, 64), 500)
    assert not _use_nufft((64, 64), 499)
    assert _use_nufft((512, 512), 1_024)  # 2 points per row of the larger side
    assert not _use_nufft((512, 512), 1_023)
    assert _use_nufft((1024, 64), 2_048) and not _use_nufft((1024, 64), 2_047)
    assert not _use_nufft((32, 32), 10**6)
    monkeypatch.setattr(jax, "default_backend", lambda: "tpu")
    assert _use_nufft((128, 128), 1_000)  # any other backend counts as cpu


def test_auto_follows_the_decision(monkeypatch):
    image, uu, vv = _problem((16, 16), n=20)
    calls = []
    real = _geometry._nufft
    monkeypatch.setattr(
        _geometry, "_nufft", lambda *a: calls.append(1) or real(*a)
    )
    monkeypatch.setattr(_geometry, "_use_nufft", lambda shape, m: True)
    image_visibilities(np.asarray(image), uu, vv, 3.0)
    image_visibilities(np.asarray(image), uu, vv, 3.0, "dft")
    image_visibilities(np.asarray(image), uu, vv, 3.0, "nufft")
    assert len(calls) == 2
    with pytest.raises(ValueError, match="ft must be"):
        image_visibilities(np.asarray(image), uu, vv, 3.0, "fft")


def test_image_loglike_agrees_between_transforms():
    data = vlti_night(60000.0, frames=60, d_vis=0.05, d_phi=2.0)
    eta = onp.random.default_rng(1).normal(size=(16, 16))
    values = [
        float(model_loglike(Image(eta, 5.0, ft=ft), data))
        for ft in ("dft", "nufft")
    ]
    assert abs(values[0] - values[1]) < 1e-3 * abs(values[0])


def test_vmap_over_a_batch_of_images():
    _, uu, vv = _problem((16, 16), n=30)
    batch = onp.random.default_rng(5).uniform(size=(3, 16, 16))
    batch = np.asarray(batch / batch.sum(axis=(1, 2), keepdims=True))
    run = lambda ft: jax.vmap(  # noqa: E731
        lambda b: image_visibilities(b, uu, vv, 3.0, ft)
    )(batch)
    assert np.max(np.abs(run("nufft") - run("dft"))) < 3e-5


def test_a_matching_uv_grid_keeps_the_matrix_transform(monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("NUFFT used on a lattice")

    monkeypatch.setattr(_geometry, "_nufft", boom)
    monkeypatch.setattr(_geometry, "_use_nufft", lambda shape, m: True)
    axis = np.linspace(-3.0, 3.0, 5)
    grid = UVGrid(axis, axis, np.arange(25), 0.0)
    u, v = np.meshgrid(axis, axis)
    image = Image(onp.random.default_rng(0).normal(size=(8, 8)), 4.0)
    vis = image.model_on_grid(u.ravel(), v.ravel(), np.asarray(WAVEL), grid)
    assert vis.shape == (25,)


@pytest.fixture
def restore_defaults():
    old = set_ft_defaults()
    yield
    set_ft_defaults(**old)


def test_set_ft_defaults_sets_and_restores(restore_defaults):
    old = set_ft_defaults()
    assert old["ft"] == "auto" and old["cpu_min_points"] == 500
    previous = set_ft_defaults(cpu_min_points=10, eps_f32=1e-4)
    assert previous == old
    now = set_ft_defaults()
    assert now["cpu_min_points"] == 10 and now["eps_f32"] == 1e-4
    assert now["cpu_min_side"] == old["cpu_min_side"]  # untouched
    set_ft_defaults(**previous)
    assert set_ft_defaults() == old
    with pytest.raises(ValueError, match="ft must be"):
        set_ft_defaults(ft="fft")


def test_thresholds_set_through_the_function_change_the_decision(
    monkeypatch, restore_defaults
):
    monkeypatch.setattr(jax, "default_backend", lambda: "cpu")
    assert not _use_nufft((32, 32), 100)
    set_ft_defaults(cpu_min_points=50, cpu_min_side=16)
    assert _use_nufft((32, 32), 100)
    monkeypatch.setattr(jax, "default_backend", lambda: "gpu")
    assert not _use_nufft((32, 32), 100)
    set_ft_defaults(gpu_min_points=50, gpu_min_side=16)
    assert _use_nufft((32, 32), 100)


def test_global_ft_forces_a_transform(monkeypatch, restore_defaults):
    monkeypatch.setattr(jax, "default_backend", lambda: "cpu")
    set_ft_defaults(ft="dft")
    assert not _use_nufft((512, 512), 10**6)
    set_ft_defaults(ft="nufft")
    assert _use_nufft((4, 4), 1)
    # An explicit ft on a call beats the global setting.
    image, uu, vv = _problem((16, 16), n=20)
    set_ft_defaults(ft="dft")
    calls = []
    real = _geometry._nufft
    monkeypatch.setattr(
        _geometry, "_nufft", lambda *a: calls.append(1) or real(*a)
    )
    image_visibilities(np.asarray(image), uu, vv, 3.0)
    assert not calls
    image_visibilities(np.asarray(image), uu, vv, 3.0, "nufft")
    assert calls


def test_flared_disk_takes_ft(monkeypatch):
    kwargs = dict(
        g=0.3, radius=40.0, fwhm=20.0, inc=40.0, pa=10.0,
        npix=16, pixel_scale_mas=8.0,
    )  # fmt: skip
    _, uu, vv = _problem((2, 2), n=20)
    u, v = uu * WAVEL, vv * WAVEL
    a = FlaredDiskHG(ft="dft", **kwargs).model(u, v, WAVEL)
    b = FlaredDiskHG(ft="nufft", **kwargs).model(u, v, WAVEL)
    assert np.max(np.abs(a - b)) < 3e-5


@pytest.mark.xfail(
    strict=True,
    reason="nufftax 0.6.1 batches only one dimension at axis 0: nested vmap",
)
def test_nested_vmap_through_the_nufft():
    _, uu, vv = _problem((8, 8), n=10)
    batch = np.ones((2, 3, 8, 8))
    jax.vmap(jax.vmap(lambda b: image_visibilities(b, uu, vv, 3.0, "nufft")))(
        batch
    )


@pytest.fixture
def choose(monkeypatch):
    """Set ``ft="auto"`` to nufftax (True) or the DFT (False) on a CPU."""

    def set_choice(nufft):
        monkeypatch.setattr(jax, "default_backend", lambda: "cpu")
        monkeypatch.setitem(_geometry._FT_DEFAULTS, "cpu_min_side", 1)
        monkeypatch.setitem(
            _geometry._FT_DEFAULTS, "cpu_min_points", 1 if nufft else 10**9
        )
        jax.clear_caches()  # the choice is made at trace time

    return set_choice


def test_jvp_alone_and_under_vmap_matches_the_dft(choose):
    image, uu, vv = _problem((8, 9), n=20)
    image = np.asarray(image)
    tangents = np.asarray(onp.random.default_rng(2).normal(size=(3, 8, 9)))

    def run(nufft):
        choose(nufft)

        def jvp(t):
            return jax.jvp(
                lambda b: image_visibilities(b, uu, vv, 3.0),
                (image,),
                (t,),
            )

        return jvp(tangents[0]), jax.vmap(jvp)(tangents)

    for got, expected in zip(run(True), run(False)):
        for a, b in zip(got, expected):
            assert np.max(np.abs(a - b)) < 5e-4  # eps 1e-5 of V(0) ~ 36


def _irregular_data():
    from virgil.coverage import vlti_oidata

    truth = Image.from_brightness(
        onp.random.default_rng(6).uniform(size=(16, 16)), 4.0
    )
    data = vlti_oidata(wavelengths_m=onp.linspace(3.0e-6, 3.2e-6, 3))
    return data.with_model(truth, key=jax.random.PRNGKey(0))


def test_clean_agrees_between_transforms(choose):
    from virgil.imaging import clean
    from virgil.models import PointSource

    data = _irregular_data()
    runs = []
    for nufft in (False, True):
        choose(nufft)
        runs.append(
            clean(data, 16, 4.0, base=PointSource(), max_iterations=10)
        )
    dft, auto = runs
    assert np.allclose(dft.components, auto.components, atol=1e-4)
    assert np.allclose(dft.chi2_red, auto.chi2_red, rtol=1e-2)


def test_laplace_samples_agree_between_transforms(choose):
    from virgil.fields import GaussianField
    from virgil.imaging import _residual_jacobian, laplace_samples
    from virgil.models import PointSource, System

    data = _irregular_data()
    n = 4
    mean = onp.full((n, n), 0.1)
    scene = System(
        star=PointSource(),
        env=Image(
            GaussianField(onp.zeros((n, n)), 1.5, 2.0, mean=mean),
            4.0,
            flux=0.4,
        ),
    )
    out = []
    for nufft in (False, True):
        choose(nufft)
        _, jac = _residual_jacobian(scene, data, "env.log_brightness.latent")
        draws = laplace_samples(scene, data, 4, jax.random.PRNGKey(1))
        assert np.all(np.isfinite(draws["latents"]))
        out.append(jac)
    # The draws themselves depend on an eigen-decomposition's signs; compare
    # the Jacobian they come from.
    assert np.max(np.abs(out[0] - out[1])) < 1e-3 * np.max(np.abs(out[0]))


def test_spread_accepts_real_and_complex_weights():
    from virgil import _nufft_spread as spread_grid

    with jax.enable_x64(True):
        rng = onp.random.default_rng(7)
        tx, ty = rng.uniform(-0.5, 0.5, (2, 100)) * onp.pi
        w_real = np.asarray(rng.uniform(0.1, 1.0, 100), dtype=np.float64)
        w_cplx = w_real.astype(np.complex128)
        kp = spread_grid.kernel(1e-12)
        nf = spread_grid.grid_size(8, kp)
        b_real = spread_grid.spread(
            np.asarray(tx), np.asarray(ty), w_real, nf, kp
        )
        b_cplx = spread_grid.spread(
            np.asarray(tx), np.asarray(ty), w_cplx, nf, kp
        )
        m = onp.arange(-4, 4)
        onp.testing.assert_allclose(
            spread_grid.evaluate(b_real, m, 0.2, m, -0.1, kp),
            spread_grid.evaluate(b_cplx, m, 0.2, m, -0.1, kp),
            rtol=1e-14,
        )
