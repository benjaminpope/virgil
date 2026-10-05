"""Detection statistics for ROC curves (design/detection_roc.md, Stage 1).

Small synthetic data (the 7-hole NIRISS mask, 21 V² and 35 closure phases)
and small grids, so that these run on a laptop.
"""

import itertools
import warnings

import jax
import jax.numpy as jnp
import numpy as onp
import pytest
from scipy import special, stats

from tests._compiles import count_compiles
from virgil.coverage import nrm_oidata
from virgil.detection import (
    _constrained_profile,
    detection_statistics,
    local_nsigma,
)
from virgil.grid_fit import likelihood_grid
from virgil.likelihood import loglike
from virgil.models import BinaryModelCartesian, PointSource, System

TEMPLATE = nrm_oidata()
NULL = BinaryModelCartesian(0.0, 0.0, 0.0)
# The companion's flux uncertainty at (60, -40) mas is about 1e-3.
POSITION = (60.0, -40.0)


def _data(flux, seed, position=POSITION):
    scene = BinaryModelCartesian(*position, flux)
    return TEMPLATE.with_model(scene, key=jax.random.PRNGKey(seed))


def _box(n_pos, n_flux, flux_max=0.015):
    """A box of positions around POSITION and a linear flux axis from 0."""
    return {
        "dra": jnp.linspace(0.0, 120.0, n_pos),
        "ddec": jnp.linspace(-100.0, 20.0, n_pos),
        "flux": jnp.linspace(0.0, flux_max, n_flux),
    }


def _quiet(fn, *args, **kwargs):
    """Call ``fn`` ignoring the optimizer's convergence warnings."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "detection_statistics")
        return fn(*args, **kwargs)


def test_delta_chi2_is_non_negative_and_at_least_the_grid_maximum():
    grid = {
        "dra": jnp.linspace(-120.0, 120.0, 5),
        "ddec": jnp.linspace(-120.0, 120.0, 5),
        "flux": jnp.geomspace(1e-4, 0.05, 12),
    }
    for flux, seed in [(0.0, 0), (0.0, 1), (0.0, 2), (3e-3, 3)]:
        data = _data(flux, seed)
        result = _quiet(detection_statistics, data, BinaryModelCartesian, grid)
        assert float(result["delta_chi2"]) >= 0.0
        # The flux optimizer only improves on the best grid point.
        grid_ll = likelihood_grid(data, BinaryModelCartesian, grid)
        loglike0 = loglike(jnp.array([0.0, 0.0, 0.0]), *_args(data))
        grid_delta = 2.0 * float(jnp.max(grid_ll) - loglike0)
        assert float(result["delta_chi2"]) >= grid_delta - 1e-3
        assert float(result["flux"]) >= 0.0
        if float(result["delta_chi2"]) == 0.0:
            assert float(result["flux"]) == 0.0


def _args(data):
    return (["dra", "ddec", "flux"], data, BinaryModelCartesian)


@pytest.mark.validates(
    "virgil.detection.detection_statistics",
    "virgil.detection.local_nsigma",
    roots=["statistics"],
)
def test_one_point_null_delta_chi2_follows_the_chernoff_mixture():
    # At one position fixed in advance there is no look-elsewhere effect:
    # under the null the flux estimate is Gaussian about 0 and bounded
    # below by 0, so delta_chi2 is 0 half the time and chi-squared with
    # one degree of freedom otherwise (Chernoff 1954).
    grid = {
        "dra": jnp.array([POSITION[0]]),
        "ddec": jnp.array([POSITION[1]]),
        "flux": jnp.geomspace(1e-5, 0.3, 40),
    }

    @jax.jit
    def null_stats(keys):
        return jax.lax.map(
            lambda key: detection_statistics(
                TEMPLATE.with_model(NULL, key=key),
                BinaryModelCartesian,
                grid,
            ),
            keys,
        )

    n = 200
    result = null_stats(jax.random.split(jax.random.PRNGKey(0), n))
    delta = onp.asarray(result["delta_chi2"])
    assert onp.all(delta >= 0.0)
    zeros = int(onp.sum(delta == 0.0))
    assert stats.binomtest(zeros, n, 0.5).pvalue > 0.01
    positive = delta[delta > 0.0]
    assert stats.kstest(positive, stats.chi2(1).cdf).pvalue > 0.01
    # The best flux sits on the zero boundary exactly when delta_chi2 = 0.
    assert onp.array_equal(onp.asarray(result["flux"]) == 0.0, delta == 0.0)
    # Locally, Wilks's significance is the square root of delta_chi2.
    assert onp.allclose(local_nsigma(delta), onp.sqrt(delta), atol=1e-3)


def test_local_nsigma_is_the_one_sided_gaussian_tail():
    assert float(local_nsigma(0.0)) == pytest.approx(0.0, abs=1e-6)
    assert float(local_nsigma(9.0)) == pytest.approx(3.0, rel=1e-4)
    assert float(local_nsigma(25.0)) == pytest.approx(5.0, rel=1e-4)


@pytest.mark.validates(
    "virgil.detection.detection_statistics",
    roots=["statistics"],
)
def test_log_bayes_factor_rises_with_injected_flux():
    grid = _box(7, 81, flux_max=0.04)  # (60, -40) is a grid point
    log_b = [
        float(
            _quiet(
                detection_statistics,
                _data(flux, 5),
                BinaryModelCartesian,
                grid,
            )["log_bayes_factor"]
        )
        for flux in (0.0, 2e-3, 5e-3, 1e-2, 2e-2)
    ]
    assert onp.all(onp.diff(log_b) > 0.0)
    # No companion: the evidence favours none; a 10-sigma one: strongly.
    assert log_b[0] < 0.0
    assert log_b[-1] > 50.0


@pytest.mark.validates(
    "virgil.detection.detection_statistics",
    roots=["self-consistency"],
)
@pytest.mark.slow
def test_log_bayes_factor_is_stable_under_grid_refinement():
    # A ~5 sigma companion off the grid points, on grids of 7 to 25
    # positions per axis over the same box (the 7-point grid resolves the
    # flux peak with fewer than two steps across its FWHM).
    data = _data(5e-3, 3, position=(57.0, -43.0))
    log_b = []
    for n_pos, n_flux in [(7, 16), (13, 31), (25, 61)]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            result = detection_statistics(
                data, BinaryModelCartesian, _box(n_pos, n_flux)
            )
        log_b.append(float(result["log_bayes_factor"]))
    assert log_b[2] > 5.0
    assert onp.allclose(log_b, log_b[2], atol=0.02)


@pytest.mark.validates(
    "virgil.detection.detection_statistics",
    roots=["self-consistency"],
)
def test_log_bayes_factor_matches_a_brute_force_marginalisation():
    grid = {
        "dra": onp.array([30.0, 60.0, 90.0]),
        "ddec": onp.array([-60.0, -40.0, -20.0]),
        "flux": onp.geomspace(1e-3, 1e-2, 4),
    }
    data = _data(4e-3, 7)
    result = _quiet(detection_statistics, data, BinaryModelCartesian, grid)

    def trapezoid(n):
        # Uniform in position, and in log flux, between the axes' ends.
        w = onp.ones(n)
        w[[0, -1]] = 0.5
        return w / (n - 1)

    args = _args(data)
    loglike0 = float(loglike(jnp.array([0.0, 0.0, 0.0]), *args))
    weights = [trapezoid(len(values)) for values in grid.values()]
    terms = []
    for (i, dra), (j, ddec), (k, flux) in itertools.product(
        *(enumerate(values) for values in grid.values())
    ):
        log_l = float(loglike(jnp.array([dra, ddec, flux]), *args))
        weight = weights[0][i] * weights[1][j] * weights[2][k]
        terms.append(log_l - loglike0 + onp.log(weight))
    expected = special.logsumexp(terms)
    assert float(result["log_bayes_factor"]) == pytest.approx(
        expected, abs=2e-3
    )


def test_system_template_with_paths_matches_the_binary_class():
    # The null keeps every other parameter of a template (here only the
    # star); the statistics depend on the model, not on how it is built.
    grid = {
        "dra": jnp.linspace(0.0, 120.0, 4),
        "ddec": jnp.linspace(-100.0, 20.0, 4),
        "flux": jnp.geomspace(1e-4, 0.03, 30),
    }
    data = _data(5e-3, 11)
    binary = _quiet(detection_statistics, data, BinaryModelCartesian, grid)
    template = System(primary=PointSource(), comp=PointSource(0.01))
    paths = {f"comp.{key}": values for key, values in grid.items()}
    system = _quiet(detection_statistics, data, template, paths)
    for key in ("delta_chi2", "log_bayes_factor", "max_snr"):
        assert float(system[key]) == pytest.approx(
            float(binary[key]), rel=1e-3, abs=1e-3
        )
    for key in ("dra", "ddec", "flux"):
        assert float(system[f"comp.{key}"]) == pytest.approx(
            float(binary[key]), rel=1e-3, abs=1e-6
        )


def test_one_compile_across_simulated_draws():
    grid = _box(4, 12)
    first = _data(0.0, 0)
    _quiet(detection_statistics, first, BinaryModelCartesian, grid)
    with count_compiles() as compiles:
        for seed in (1, 2):
            _quiet(
                detection_statistics,
                _data(3e-3, seed),
                BinaryModelCartesian,
                grid,
            )
    assert not compiles

    @jax.jit
    def run(keys):
        return jax.lax.map(
            lambda key: detection_statistics(
                TEMPLATE.with_model(NULL, key=key),
                BinaryModelCartesian,
                grid,
            ),
            keys,
        )

    run(jax.random.split(jax.random.PRNGKey(0), 3))
    with count_compiles() as compiles:
        out = run(jax.random.split(jax.random.PRNGKey(1), 3))
    assert not compiles
    assert out["delta_chi2"].shape == (3,)


def test_unresolved_flux_peak_and_flux_beyond_the_axis_warn():
    data = _data(1e-2, 5)
    position = {"dra": jnp.array([60.0]), "ddec": jnp.array([-40.0])}
    coarse = {**position, "flux": jnp.geomspace(1e-4, 0.1, 5)}
    with pytest.warns(RuntimeWarning, match="does not resolve"):
        result = detection_statistics(data, BinaryModelCartesian, coarse)
    assert float(result["flux_peak_steps"]) < 2.0
    short = {**position, "flux": jnp.linspace(0.0, 3e-3, 20)}
    with pytest.warns(RuntimeWarning, match="above the flux axis"):
        detection_statistics(data, BinaryModelCartesian, short)


def test_statistic_names_cannot_be_grid_keys():
    grid = {**_box(2, 3), "max_snr": jnp.array([1.0])}
    with pytest.raises(ValueError, match="clash"):
        detection_statistics(_data(0.0, 0), BinaryModelCartesian, grid)


def test_constrained_profile_keeps_positive_grid_points():
    # Positions: (0) refinement worse than the grid but with a negative
    # flux; (1) refinement better but at a negative flux; (2) refinement
    # NaN; (3) a valid positive refinement; (4) nothing beats the null.
    grid_flux = jnp.array([0.2, 0.3, 0.1, 0.1, 0.1])
    grid_loglike = jnp.array([5.0, 4.0, 3.0, 2.0, -1.0])
    opt_flux = jnp.array([-0.1, -0.2, jnp.nan, 0.15, 0.1])
    opt_loglike = jnp.array([4.0, 9.0, jnp.nan, 6.0, -0.5])
    profile, flux = _constrained_profile(
        grid_flux, grid_loglike, opt_flux, opt_loglike, 0.0
    )
    onp.testing.assert_allclose(profile, [5.0, 4.0, 3.0, 6.0, 0.0])
    onp.testing.assert_allclose(flux, [0.2, 0.3, 0.1, 0.15, 0.0])
