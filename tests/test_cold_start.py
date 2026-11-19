"""Cold-start bookkeeping runs on the host, not as eager XLA compiles."""

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

MATRICES = {
    "positive-definite": onp.array([[4.0, 1.0], [1.0, 3.0]]),
    "indefinite": onp.array([[2.0, 0.0], [0.0, -1.0]]),
    "zero-curvature": onp.array([[1.0, 0.0], [0.0, 0.0]]),
}


@pytest.mark.parametrize("name", MATRICES)
def test_regularized_inverse_host_matches_traced(name):
    matrix = MATRICES[name]
    with pytest.warns() if name != "positive-definite" else _no_warning():
        host = regularized_inverse(matrix)
    traced = jax.jit(regularized_inverse)(np.asarray(matrix))
    assert onp.allclose(host, traced, rtol=1e-5, equal_nan=True)


class _no_warning:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


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


def test_enable_compilation_cache(tmp_path):
    import virgil

    old = jax.config.jax_compilation_cache_dir
    try:
        path = virgil.enable_compilation_cache(tmp_path / "c")
        assert path.is_dir()
        assert jax.config.jax_compilation_cache_dir == str(path)
        assert jax.config.jax_persistent_cache_min_compile_time_secs == 0
        assert jax.config.jax_persistent_cache_min_entry_size_bytes == 0
    finally:
        jax.config.update("jax_compilation_cache_dir", old)
