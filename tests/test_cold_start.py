"""Cold-start bookkeeping runs on the host, not as eager XLA compiles."""

import warnings

import jax
import jax.numpy as np
import numpy as onp
import numpyro.distributions as dist
import pytest

from virgil.coverage import ami_grid_record
from virgil.fitting import _largest, _tolerance, fit
from virgil.inference import laplace_cov, regularized_inverse
from virgil.models import BinaryModelCartesian
from virgil.oidata import OIData

from ._compiles import count_compiles

# (matrix, analytic inverse, warning). The ridge is 1e-10 of each diagonal.
CASES = {
    "positive-definite": (
        onp.array([[4.0, 1.0], [1.0, 3.0]]),
        onp.array([[3.0, -1.0], [-1.0, 4.0]]) / 11,
        None,
    ),
    "indefinite": (
        onp.array([[2.0, 0.0], [0.0, -1.0]]),
        onp.diag([0.5, -1.0]),
        "not positive definite",
    ),
    "zero-curvature": (
        onp.array([[1.0, 0.0], [0.0, 0.0]]),
        onp.diag([1.0, 1e10]),
        "a ridge was added",
    ),
}


@pytest.mark.parametrize("name", CASES)
def test_regularized_inverse_host(name):
    matrix, inverse, message = CASES[name]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        host = regularized_inverse(matrix)
    texts = [str(w.message) for w in caught]
    if message is None:
        assert texts == []
    else:
        assert any(
            issubclass(w.category, RuntimeWarning)
            and message in str(w.message)
            for w in caught
        ), texts
    assert onp.allclose(host, inverse, rtol=1e-4)
    # And the traced path agrees.
    traced = jax.jit(regularized_inverse)(np.asarray(matrix))
    assert onp.allclose(host, traced, rtol=1e-5, equal_nan=True)


def test_regularized_inverse_accepts_a_list_and_keeps_dtype():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = regularized_inverse([[4.0, 1.0], [1.0, 3.0]])
    assert result.dtype == np.asarray(0.0, dtype=float).dtype


def test_regularized_inverse_precision_follows_jax():
    # Positive definite in float64 but singular in float32: whether the ridge
    # is added depends on JAX's precision, the same on the host as traced.
    matrix = onp.array([[1.0, 1.0], [1.0, 1.0 + 1e-8]])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        host = regularized_inverse(matrix)
    traced = jax.jit(regularized_inverse)(np.asarray(matrix))
    # Both singular in float32 (non-finite), both fine in float64.
    assert onp.isfinite(host).all() == onp.isfinite(traced).all()
    assert onp.isfinite(host).all() == bool(jax.config.jax_enable_x64)


def test_tolerance_host_matches_traced():
    gradient = {
        "a": onp.array([3.0, -7.0], onp.float32),
        "b": onp.array([0.5], onp.float32),
    }
    host = _tolerance(gradient, 1e-4)
    traced = jax.jit(lambda g: _tolerance(g, 1e-4))(gradient)
    assert onp.isclose(host, traced, rtol=1e-6)
    assert onp.isclose(_largest(gradient), 7.0)


def test_cold_laplace_and_fit_avoid_eager_compiles():
    priors = {
        "dra": dist.Uniform(-400.0, 400.0),
        "ddec": dist.Uniform(-400.0, 400.0),
        "flux": dist.Uniform(0.0, 0.5),
    }
    truth = BinaryModelCartesian(150.0, -80.0, 0.02)
    data = OIData(ami_grid_record(pitch_m=0.5)).with_model(
        truth, key=jax.random.PRNGKey(0)
    )
    fit(BinaryModelCartesian(60.0, -20.0, 0.005), priors, data, method="lm")
    # The Hessian and its inverse were ~17 compiles (one jitted Hessian plus
    # a dozen single-op eager programs); now only the Hessian remains.
    with count_compiles() as compiles:
        cov = laplace_cov(
            onp.array([150.0, -80.0, 0.02]),
            ["dra", "ddec", "flux"],
            BinaryModelCartesian,
            data,
        )
    assert len(compiles) <= 5
    assert onp.all(onp.isfinite(cov))


def test_enable_compilation_cache_writes_entries(tmp_path):
    import virgil

    keys = (
        "jax_compilation_cache_dir",
        "jax_persistent_cache_min_compile_time_secs",
        "jax_persistent_cache_min_entry_size_bytes",
    )
    old = {key: getattr(jax.config, key) for key in keys}
    try:
        path = virgil.enable_compilation_cache(tmp_path / "c")
        assert path.is_dir()
        assert jax.config.jax_persistent_cache_min_compile_time_secs == 0
        assert jax.config.jax_persistent_cache_min_entry_size_bytes == 0
        jax.jit(lambda x: np.sin(x) * 3.25 + 0.125)(
            np.arange(7.0)
        ).block_until_ready()
        assert any(path.iterdir())
    finally:
        for key, value in old.items():
            jax.config.update(key, value)
        from jax._src import compilation_cache

        compilation_cache.reset_cache()
