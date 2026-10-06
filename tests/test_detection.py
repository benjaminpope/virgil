"""Detection statistics and their Monte Carlo (design/detection_roc.md).

Small synthetic data (the 7-hole NIRISS mask, 21 V² and 35 closure phases)
and small grids, so that these run on a laptop.
"""

import dataclasses
import itertools
import warnings

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as onp
import pytest
from scipy import special, stats

from tests._compiles import count_compiles
from virgil.coverage import nrm_oidata
from virgil import detection
from virgil.detection import (
    STATISTICS,
    DetectionMC,
    _constrained_profile,
    _describe,
    _fingerprint,
    bootstrap_null,
    detection_statistics,
    gaussian_null,
    injection_grid,
    injection_recovery,
    local_nsigma,
    rescale_errors,
)
from virgil.grid_fit import likelihood_grid
from virgil.likelihood import loglike, whitened_residuals
from virgil.models import (
    BinaryModelAngular,
    BinaryModelCartesian,
    Image,
    PointSource,
    System,
)

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
        result = _quiet(detection_statistics, BinaryModelCartesian, data, grid)
        assert float(result["delta_chi2"]) >= 0.0
        # The flux optimizer only improves on the best grid point.
        grid_ll = likelihood_grid(BinaryModelCartesian, data, grid)
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
                BinaryModelCartesian,
                TEMPLATE.with_model(NULL, key=key),
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
                BinaryModelCartesian,
                _data(flux, 5),
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
    # A ~5 sigma companion off the grid points, on grids of 13 to 49
    # positions per axis over the same box. The evidence is a trapezoid
    # rule, so it converges as O(step^2) once every axis resolves the
    # likelihood peaks, as the docstring requires. The noise draw differs
    # between float32 and x64 (jax.random is dtype dependent), and so do
    # the peaks: position FWHMs of 34-46 mas in float32 but 23-25 mas
    # under x64. A 7-point grid (20 mas steps) is then pre-asymptotic, 1.2
    # steps per FWHM, and was 0.03 low under x64 (0.003 in float32). From
    # 13 points (10 mas, at least 2.3 steps) the error falls by 4 per
    # halving: against an exact flux integral of linear_flux_grid on
    # 193 points, 1.4e-3 at 13, 3.6e-4 at 25 and 9e-5 at 49 under x64.
    # The 16-point flux axis (1.7 to 8 steps per FWHM at the best
    # position) is within 3e-4 of a 121-point one. Measured spreads to
    # the 49-point grid are at most 2e-3 in either precision; atol=0.01
    # leaves a margin of five.
    data = _data(5e-3, 3, position=(57.0, -43.0))
    log_b = []
    for n_pos, n_flux in [(13, 16), (25, 31), (49, 61)]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            result = detection_statistics(
                BinaryModelCartesian, data, _box(n_pos, n_flux)
            )
        log_b.append(float(result["log_bayes_factor"]))
    assert log_b[2] > 5.0
    assert onp.allclose(log_b, log_b[2], atol=0.01)


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
    result = _quiet(detection_statistics, BinaryModelCartesian, data, grid)

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
    binary = _quiet(detection_statistics, BinaryModelCartesian, data, grid)
    template = System(primary=PointSource(), comp=PointSource(0.01))
    paths = {f"comp.{key}": values for key, values in grid.items()}
    system = _quiet(detection_statistics, template, data, paths)
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
    _quiet(detection_statistics, BinaryModelCartesian, first, grid)
    with count_compiles() as compiles:
        for seed in (1, 2):
            _quiet(
                detection_statistics,
                BinaryModelCartesian,
                _data(3e-3, seed),
                grid,
            )
    assert not compiles

    @jax.jit
    def run(keys):
        return jax.lax.map(
            lambda key: detection_statistics(
                BinaryModelCartesian,
                TEMPLATE.with_model(NULL, key=key),
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
        result = detection_statistics(BinaryModelCartesian, data, coarse)
    assert float(result["flux_peak_steps"]) < 2.0
    short = {**position, "flux": jnp.linspace(0.0, 3e-3, 20)}
    with pytest.warns(RuntimeWarning, match="above the flux axis"):
        detection_statistics(BinaryModelCartesian, data, short)


def test_statistic_names_cannot_be_grid_keys():
    grid = {**_box(2, 3), "max_snr": jnp.array([1.0])}
    with pytest.raises(ValueError, match="clash"):
        detection_statistics(BinaryModelCartesian, _data(0.0, 0), grid)


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


# ---------------------------------------------------------------------------
# Stage 2: simulators, the Monte Carlo driver and DetectionMC
# ---------------------------------------------------------------------------

# A tiny grid for compile and bookkeeping tests, and a 7 x 7 x 8 one for the
# statistical tests (shared through a module fixture: 64 null and
# 2 x 4 x 12 = 96 injected draws).
TINY = {
    "dra": jnp.linspace(0.0, 120.0, 3),
    "ddec": jnp.linspace(-100.0, 20.0, 3),
    "flux": jnp.geomspace(1e-3, 3e-2, 6),
}
SEARCH = {
    "dra": jnp.linspace(-120.0, 120.0, 7),
    "ddec": jnp.linspace(-120.0, 120.0, 7),
    "flux": jnp.geomspace(5e-4, 5e-2, 8),
}


def _recover(grid=TINY, key=0, n_null=4, injections=None, **kwargs):
    kwargs = {"chunk_size": 4, "progress": False} | kwargs
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return injection_recovery(
            BinaryModelCartesian,
            NULL,
            TEMPLATE,
            grid,
            key,
            n_null=n_null,
            injections=injections,
            **kwargs,
        )


@pytest.fixture(scope="module")
def search_mc():
    fluxes = [0.0, 1e-3, 3e-3, 2e-2]
    injections = injection_grid([60.0, 90.0], fluxes, 12, 1)
    return _recover(
        SEARCH, key=3, n_null=64, injections=injections, chunk_size=24
    )


def _synthetic(null, injected=None, **meta):
    """A DetectionMC built by hand, with every statistic set to ``null``."""
    null = {s: onp.asarray(null, float) for s in STATISTICS}
    injected = injected or {}
    n = len(injected.get("flux", []))
    full = {s: onp.zeros(n) for s in STATISTICS}
    full.update({k: onp.asarray(v, float) for k, v in injected.items()})
    meta = {"match_radius": None, "seeds": [{"seed": 0}]} | meta
    return DetectionMC(null=null, injected=full, meta=meta)


def test_monte_carlo_compiles_once_across_draws_null_and_injected():
    inj = injection_grid([60.0], [0.0, 1e-2], 2, 1)
    _recover(key=0, n_null=4, injections=inj)
    more = injection_grid([40.0, 80.0], [5e-3, 1e-2], 3, 2)
    with count_compiles() as compiles:
        mc = _recover(key=5, n_null=9, injections=more)
    assert not compiles
    assert mc.n_null == 9 and mc.n_injected == 12
    assert mc.meta["n_null"] == 9 and mc.meta["n_injected"] == 12


def test_monte_carlo_is_reproducible_and_independent_of_chunking():
    inj = injection_grid([60.0], [1e-2], 4, 1)
    a = _recover(key=7, n_null=4, injections=inj)
    b = _recover(key=7, n_null=4, injections=inj)
    for part in ("null", "injected"):
        for key, values in getattr(a, part).items():
            onp.testing.assert_array_equal(values, getattr(b, part)[key])
    c = _recover(key=8, n_null=4, injections=inj)
    assert not onp.allclose(a.null["delta_chi2"], c.null["delta_chi2"])
    # Draw i always uses the same key, whatever the chunks.
    d = _recover(key=7, n_null=4, injections=inj, chunk_size=3)
    onp.testing.assert_allclose(
        d.injected["delta_chi2"], a.injected["delta_chi2"], rtol=1e-4
    )


def test_draw_batch_vectorises_the_same_draws():
    # draw_batch > 1 vmaps the search over draws (the tutorial uses it to
    # amortise the flux optimizer's loop); draw i keeps its key.
    inj = injection_grid([60.0], [1e-2], 4, 1)
    a = _recover(key=7, n_null=4, injections=inj)
    b = _recover(key=7, n_null=4, injections=inj, draw_batch=2)
    for part in ("null", "injected"):
        for stat in STATISTICS:
            onp.testing.assert_allclose(
                getattr(b, part)[stat],
                getattr(a, part)[stat],
                rtol=1e-3,
                atol=1e-3,
            )


def test_system_template_and_bootstrap_noise_run_through_the_driver():
    # Injections at a grid point inside TINY's box (random PAs can fall
    # outside it): two null ones, and two ~10 sigma companions.
    inj = {
        "dra": [60.0] * 4,
        "ddec": [-40.0] * 4,
        "flux": [0.0, 0.0, 1e-2, 1e-2],
    }
    binary = _recover(n_null=4, injections=inj)
    template = System(primary=PointSource(), comp=PointSource(0.01))
    paths = {f"comp.{key}": values for key, values in TINY.items()}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        system = injection_recovery(
            template,
            NULL,
            TEMPLATE,
            paths,
            0,
            n_null=4,
            injections=inj,
            chunk_size=4,
            progress=False,
        )
    onp.testing.assert_allclose(
        system.injected["delta_chi2"],
        binary.injected["delta_chi2"],
        rtol=1e-3,
        atol=1e-2,
    )
    assert set(system.injected) == set(binary.injected)
    data = TEMPLATE.with_model(NULL, key=jax.random.PRNGKey(9))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        boot = injection_recovery(
            BinaryModelCartesian,
            NULL,
            data,
            TINY,
            0,
            n_null=4,
            injections=inj,
            noise="bootstrap",
            chunk_size=4,
            progress=False,
        )
    assert boot.meta["noise"] == {"kind": "bootstrap", "method": "sign_flip"}
    assert onp.all(boot.null["delta_chi2"] >= 0.0)
    # The bright injections are found: well above every null draw, and
    # well above Wilks's 5 sigma (delta_chi2 = 25; about 100 expected).
    bright = boot.injected["delta_chi2"][2:]
    assert onp.all(bright > 25.0)
    assert bright.min() > boot.null["delta_chi2"].max()


def test_null_scene_must_match_the_model_at_zero_flux():
    with pytest.raises(ValueError, match="predict different data"):
        injection_recovery(
            BinaryModelCartesian,
            BinaryModelCartesian(60.0, -40.0, 0.01),
            TEMPLATE,
            TINY,
            0,
            n_null=1,
        )


def test_save_load_and_concatenate_round_trip(tmp_path):
    inj = injection_grid([60.0], [1e-2], 2, 1)
    a = _recover(key=1, n_null=4, injections=inj)
    b = _recover(key=2, n_null=4, injections=inj)
    path = tmp_path / "mc.npz"
    a.save(path)
    loaded = DetectionMC.load(path)
    assert loaded.meta == a.meta
    for part in ("null", "injected"):
        assert getattr(loaded, part).keys() == getattr(a, part).keys()
        for key, values in getattr(a, part).items():
            onp.testing.assert_array_equal(getattr(loaded, part)[key], values)
    merged = DetectionMC.concatenate([loaded, b])
    assert merged.n_null == 8 and merged.n_injected == 4
    assert merged.meta["n_null"] == 8
    assert merged.meta["seeds"] == [{"seed": 1}, {"seed": 2}]
    onp.testing.assert_array_equal(
        merged.null["delta_chi2"],
        onp.concatenate([a.null["delta_chi2"], b.null["delta_chi2"]]),
    )
    # A null-only job merges with an injection-only one.
    null_only = _recover(key=3, n_null=4)
    inj_only = _recover(key=4, n_null=0, injections=inj)
    both = DetectionMC.concatenate([null_only, inj_only])
    assert both.n_null == 4 and both.n_injected == 2


def test_concatenate_refuses_incompatible_results():
    a = _recover(key=1, n_null=4)
    with pytest.raises(ValueError, match="share a seed"):
        DetectionMC.concatenate([a, a])
    for change in (
        {"grid": {"dra": [0.0]}},
        {"noise": {"kind": "bootstrap", "method": "sign_flip"}},
        {"match_radius": 10.0},
        {"template": {"n_vis": 1, "n_phi": 0, "hash": "0"}},
    ):
        other = dataclasses.replace(
            a, meta=a.meta | change | {"seeds": [{"seed": 2}]}
        )
        with pytest.raises(ValueError, match="differ in"):
            DetectionMC.concatenate([a, other])


@pytest.mark.validates(
    "virgil.detection.injection_recovery",
    "virgil.detection.DetectionMC.threshold",
    roots=["statistics"],
)
def test_empirical_threshold_exceeds_wilks_over_a_grid(search_mc):
    # Searching 49 positions makes the null's maximum larger than at one
    # fixed position (the look-elsewhere effect): the empirical threshold
    # at a given FAP is above Wilks's, where ½χ²₁ has that tail.
    assert onp.all(search_mc.null["delta_chi2"] >= 0.0)
    for fap in (0.2, 0.1, 0.05):
        threshold, error = search_mc.threshold("delta_chi2", fap)
        wilks = stats.norm.isf(fap) ** 2
        assert threshold >= wilks
        assert 0.0 < error < threshold
        # The empirical FAP of Wilks's threshold is above the nominal one.
        assert search_mc.false_alarm_probability("delta_chi2", wilks)[0] > fap


@pytest.mark.validates(
    "virgil.detection.injection_recovery",
    "virgil.detection.DetectionMC.auc",
    roots=["statistics"],
)
def test_auc_is_a_half_without_a_companion_and_one_when_bright(search_mc):
    # Zero-flux injections are null draws, so no statistic separates them
    # (96 of each: the AUC's standard error is about 0.04); a 20-sigma
    # companion is always found.
    zero = injection_grid([60.0, 90.0], [0.0], 48, 2)
    mc = _recover(key=4, n_null=96, injections=zero, chunk_size=48)
    for stat in STATISTICS:
        assert mc.auc(stat) == pytest.approx(0.5, abs=0.15)
        assert search_mc.auc(stat, flux=2e-2) > 0.99
        assert search_mc.auc(stat, flux=3e-3) > search_mc.auc(stat, 1e-3)
    fpr, tpr, thresholds = search_mc.roc("delta_chi2", flux=3e-3)
    assert fpr[0] == 0.0 and tpr[0] == 0.0 and thresholds[0] == onp.inf
    assert fpr[-1] == 1.0 and tpr[-1] == 1.0
    assert onp.all(onp.diff(fpr) >= 0.0) and onp.all(onp.diff(tpr) >= 0.0)
    assert onp.all(onp.diff(thresholds) < 0.0)


def test_completeness_and_contrast_curve_of_the_search(search_mc):
    result = search_mc.completeness("delta_chi2", 0.1)
    onp.testing.assert_allclose(result["sep"], [60.0, 90.0])
    onp.testing.assert_allclose(result["flux"], [0.0, 1e-3, 3e-3, 2e-2])
    assert onp.all(result["n"] == 12)
    assert onp.all(result["completeness"][:, -1] == 1.0)
    # At 10% FAP, about 10% of zero-flux injections pass by chance.
    assert onp.all(result["completeness"][:, 0] <= 0.35)
    sep, flux = search_mc.contrast_curve("delta_chi2", 0.1, 0.5)
    onp.testing.assert_allclose(sep, [60.0, 90.0])
    # A companion of 1e-3 has an SNR of about 1, one of 3e-3 about 3.
    assert onp.all((flux > 1e-3) & (flux < 2e-2))


@pytest.mark.validates(
    "virgil.detection.DetectionMC.false_alarm_probability",
    roots=["statistics"],
)
def test_false_alarm_interval_covers_the_truth():
    # Null statistics from a known distribution: P(N(0, 1) >= 1.645) = 5%.
    # The exact binomial interval covers it at least 95% of the time.
    rng = onp.random.default_rng(0)
    value, truth = 1.645, stats.norm.sf(1.645)
    covered, fap = [], []
    for _ in range(400):
        mc = _synthetic(rng.standard_normal(200))
        p, lower, upper = mc.false_alarm_probability("delta_chi2", value)
        covered.append(lower <= truth <= upper)
        fap.append(p)
    assert onp.mean(covered) >= 0.93
    assert onp.mean(fap) == pytest.approx(truth, abs=0.01)
    # (k + 1)/(n + 1): never zero, and exact counts.
    mc = _synthetic(onp.arange(10.0))
    p, lower, upper = mc.false_alarm_probability("max_snr", [0.0, 9.0, 10.0])
    onp.testing.assert_allclose(p, [11 / 11, 2 / 11, 1 / 11])
    assert lower[-1] == 0.0 and upper[0] == 1.0


def test_threshold_warns_when_the_null_draws_are_too_few():
    mc = _synthetic(onp.arange(100.0))
    assert mc.threshold("delta_chi2", 0.1)[0] == pytest.approx(89.1)
    with pytest.warns(RuntimeWarning, match="cannot resolve"):
        mc.threshold("delta_chi2", 1e-3)


def test_completeness_and_contrast_curve_by_hand():
    # Two separations, three fluxes, four injections per cell; the null
    # threshold at FAP 0.1 is 89.1, and an injection is detected when its
    # statistic (100 or 0) exceeds it.
    sep = onp.repeat([50.0, 100.0], 12)
    flux = onp.tile(onp.repeat([1e-3, 2e-3, 4e-3], 4), 2)
    detected = onp.array(
        [0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1]  # 0, 1/4, 1 at 50 mas
        + [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0]  # 0, 0, 1/2 at 100 mas
    )
    mc = _synthetic(
        onp.arange(100.0),
        {
            "dra": sep,
            "ddec": onp.zeros_like(sep),
            "flux": flux,
            "delta_chi2": 100.0 * detected,
        },
    )
    result = mc.completeness("delta_chi2", 0.1)
    onp.testing.assert_allclose(
        result["completeness"], [[0.0, 0.25, 1.0], [0.0, 0.0, 0.5]]
    )
    sep_out, limit = mc.contrast_curve("delta_chi2", 0.1, 0.5)
    onp.testing.assert_allclose(sep_out, [50.0, 100.0])
    # Halfway in log flux from 25% at 2e-3 to 100% at 4e-3 is 1/3 of the
    # way; 50% is reached exactly at 4e-3 at 100 mas.
    onp.testing.assert_allclose(limit, [2e-3 * 2 ** (1 / 3), 4e-3])
    assert onp.isnan(mc.contrast_curve("delta_chi2", 0.1, 0.9)[1][1])
    # Bin edges instead of distinct values.
    binned = mc.completeness(
        "delta_chi2", 0.1, sep_bins=[0, 75, 150], flux_bins=[5e-4, 3e-3, 5e-3]
    )
    onp.testing.assert_allclose(
        binned["completeness"], [[0.125, 1.0], [0.0, 0.5]]
    )


def test_match_radius_counts_only_detections_near_the_injection():
    injected = {
        "dra": [50.0, 50.0, 50.0],
        "ddec": [0.0, 0.0, 0.0],
        "flux": [1e-2, 1e-2, 1e-2],
        "best_dra": [52.0, 50.0, -50.0],  # 2, 5 and 100 mas away
        "best_ddec": [0.0, 5.0, 0.0],
        "delta_chi2": [100.0, 100.0, 100.0],
    }
    mc = _synthetic(onp.arange(100.0), injected)
    assert mc.detected("delta_chi2", 0.1).tolist() == [True, True, True]
    for radius, expected in ((10.0, 2), (3.0, 1), (0.5, 0)):
        mc.meta["match_radius"] = radius
        assert mc.detected("delta_chi2", 0.1).sum() == expected
        assert mc.roc("delta_chi2")[1][-1] == pytest.approx(expected / 3)


def test_match_radius_in_the_driver(search_mc):
    # Bright companions are found within a grid step (40 mas) of the truth,
    # but rarely within a hundredth of a mas, since they lie off the grid.
    with pytest.raises(ValueError, match="match_radius"):
        _recover(
            {"sep": jnp.array([60.0]), "flux": jnp.geomspace(1e-3, 1e-2, 4)},
            n_null=1,
            match_radius=5.0,
        )
    bright = search_mc.injected["flux"] == 2e-2
    for radius, low, high in ((40.0, 0.9, 1.0), (0.01, 0.0, 0.2)):
        mc = dataclasses.replace(
            search_mc, meta=search_mc.meta | {"match_radius": radius}
        )
        fraction = mc.detected("delta_chi2", 0.1)[bright].mean()
        assert low <= fraction <= high


def test_grid_names_may_not_overwrite_stored_outputs():
    # A key's last part names its outputs (<name>, best_<name>), which must
    # not replace a statistic, a diagnostic or another key's best position.
    flux = jnp.geomspace(1e-3, 1e-2, 4)
    for grid in (
        {"comp.dra": TINY["dra"], "comp.max_snr": TINY["ddec"]},
        {"a.converged_fraction": TINY["dra"], "ddec": TINY["ddec"]},
        {"dra": TINY["dra"], "x.best_dra": TINY["ddec"]},
    ):
        with pytest.raises(ValueError, match="stored outputs"):
            _recover({**grid, "flux": flux}, n_null=1)


def test_match_radius_must_be_finite_and_non_negative():
    for radius in (-1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="finite and non-negative"):
            _recover(n_null=1, match_radius=radius)


def test_a_simulator_must_observe_through_the_template():
    # An equal copy of the template is accepted (nothing is drawn here);
    # a simulator of other data is refused.
    copy = gaussian_null(NULL, nrm_oidata())
    assert _recover(n_null=0, noise=copy).n_null == 0
    other = TEMPLATE.with_error_scale(2.0)
    for simulator in (gaussian_null(NULL, other), bootstrap_null(NULL, other)):
        with pytest.raises(ValueError, match="different data"):
            _recover(n_null=0, noise=simulator)


def test_fingerprints_cover_static_fields_and_every_array():
    pixels = onp.zeros((4, 4))
    image = _describe(Image(pixels, 1.0))
    assert image["type"] == "virgil.models.Image"
    assert image == _describe(Image(pixels.copy(), 1.0))
    # Static fields, which hold no arrays, change the hash.
    assert image != _describe(Image(pixels, 2.0))
    assert image != _describe(Image(pixels, 1.0, rotation_deg=10.0))
    # So do the template's closure indices, not only its values.
    shuffled = eqx.tree_at(lambda d: d.i_cps1, TEMPLATE, TEMPLATE.i_cps1[::-1])
    assert _fingerprint(shuffled) != _fingerprint(TEMPLATE)
    assert _fingerprint(nrm_oidata()) == _fingerprint(TEMPLATE)
    # Classes are named; a lambda or an arbitrary object cannot be hashed.
    assert _describe(BinaryModelCartesian)["hash"] is not None
    assert _describe(lambda x: x)["hash"] is None
    assert _fingerprint({"a": object()}) is None


def test_concatenate_refuses_unfingerprinted_runs():
    a = _synthetic(onp.arange(4.0))
    a.meta["model"] = {"type": "f", "hash": None}
    b = dataclasses.replace(a, meta=a.meta | {"seeds": [{"seed": 1}]})
    with pytest.raises(ValueError, match="fingerprinted"):
        DetectionMC.concatenate([a, b])
    assert DetectionMC.concatenate([a]).n_null == 4


def test_angular_injections_give_separations_and_match_on_the_sky():
    # sep in mas, pa in degrees from North through East.
    injected = {
        "sep": [50.0, 50.0, 100.0, 100.0],
        "pa": [0.0, 90.0, 180.0, 270.0],
        "flux": [1e-2] * 4,
        "best_sep": [50.0, 52.0, 100.0, 100.0],
        "best_pa": [359.0, 90.0, 0.0, 271.0],  # 0.9, 2, 200, 1.7 mas away
        "delta_chi2": [100.0] * 4,
    }
    mc = _synthetic(onp.arange(100.0), injected)
    onp.testing.assert_allclose(mc.separations(), injected["sep"])
    result = mc.completeness("delta_chi2", 0.1)
    onp.testing.assert_allclose(result["sep"], [50.0, 100.0])
    onp.testing.assert_allclose(result["completeness"], [[1.0], [1.0]])
    mc.meta["match_radius"] = 2.5
    assert mc.matched().tolist() == [True, True, False, True]
    # The driver accepts match_radius on an angular grid.
    angular = {
        "sep": jnp.array([60.0, 80.0]),
        "pa": jnp.array([0.0, 90.0]),
        "flux": jnp.geomspace(1e-3, 1e-2, 4),
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        out = injection_recovery(
            BinaryModelAngular,
            NULL,
            TEMPLATE,
            angular,
            0,
            n_null=0,
            match_radius=5.0,
            progress=False,
        )
    assert out.meta["match_radius"] == 5.0


def test_threshold_bootstrap_is_batched_and_skipped_when_unused(monkeypatch):
    mc = _synthetic(onp.random.default_rng(0).standard_normal(1000))
    full = mc.threshold("delta_chi2", 0.1, n_boot=50)
    # Batches of a few resamples give the same draws as one batch.
    monkeypatch.setattr(detection, "_BOOT_VALUES", 3000)
    assert mc.threshold("delta_chi2", 0.1, n_boot=50) == full
    assert full[1] > 0.0

    def no_bootstrap(*args, **kwargs):
        raise AssertionError("threshold() was called")

    monkeypatch.setattr(DetectionMC, "threshold", no_bootstrap)
    injected = {"dra": [50.0], "ddec": [0.0], "flux": [1e-2]}
    mc = _synthetic(onp.arange(100.0), injected)
    mc.detected("delta_chi2", 0.1)
    assert mc.completeness("delta_chi2", 0.1)["threshold"] == pytest.approx(
        89.1
    )


def test_flux_bin_centres_are_geometric_for_positive_edges():
    injected = {
        "dra": [50.0, 50.0],
        "ddec": [0.0, 0.0],
        "flux": [5e-4, 5e-3],
        "delta_chi2": [0.0, 100.0],
    }
    mc = _synthetic(onp.arange(100.0), injected)
    result = mc.completeness("delta_chi2", 0.1, flux_bins=[0.0, 1e-3, 1e-2])
    onp.testing.assert_allclose(result["flux"], [5e-4, onp.sqrt(1e-5)])
    onp.testing.assert_allclose(result["completeness"], [[0.0, 1.0]])


def test_injection_grid_geometry():
    out = injection_grid([30.0, 60.0], [0.0, 1e-3, 1e-2], 50, 4)
    assert set(out) == {"dra", "ddec", "flux"}
    assert all(values.shape == (300,) for values in out.values())
    sep = onp.hypot(out["dra"], out["ddec"])
    onp.testing.assert_allclose(sep, onp.repeat([30.0, 60.0], 150))
    onp.testing.assert_array_equal(
        out["flux"], onp.tile(onp.repeat([0.0, 1e-3, 1e-2], 50), 2)
    )
    # PA from North through East: dra = sep sin PA, ddec = sep cos PA. The
    # PAs are uniform, so every quadrant is populated.
    pa = onp.degrees(onp.arctan2(out["dra"], out["ddec"])) % 360.0
    counts = onp.histogram(pa, bins=[0, 90, 180, 270, 360])[0]
    assert onp.all(counts > 40)
    angular = BinaryModelAngular(sep[0], pa[0], 1e-2)
    cartesian = BinaryModelCartesian(out["dra"][0], out["ddec"][0], 1e-2)
    onp.testing.assert_allclose(
        TEMPLATE.model(angular), TEMPLATE.model(cartesian), atol=1e-5
    )
    again = injection_grid([30.0, 60.0], [0.0, 1e-3, 1e-2], 50, 4)
    onp.testing.assert_array_equal(again["dra"], out["dra"])


def _unequal_template():
    """The 7-hole template with unequal errors on V² and closure phases."""
    rng = onp.random.default_rng(1)
    return eqx.tree_at(
        lambda d: (d.d_vis, d.d_phi),
        TEMPLATE,
        (
            TEMPLATE.d_vis * (0.5 + rng.random(21)),
            TEMPLATE.d_phi * (0.5 + rng.random(35)),
        ),
    )


@pytest.mark.validates(
    "virgil.detection.rescale_errors", roots=["self-consistency"]
)
def test_rescale_errors_gives_unit_reduced_chi2():
    # The true noise is 3x the quoted visibility errors and half the
    # closure-phase ones.
    template = _unequal_template()
    truth = eqx.tree_at(
        lambda d: (d.d_vis, d.d_phi),
        template,
        (template.d_vis * 3.0, template.d_phi * 0.5),
    )
    n_cp = TEMPLATE.cp_noise.size
    squares = {"vis": [], "phi": []}
    for seed in range(20):
        data = truth.with_model(NULL, key=jax.random.PRNGKey(seed))
        data = eqx.tree_at(
            lambda d: (d.d_vis, d.d_phi),
            data,
            (template.d_vis, template.d_phi),
        )
        scaled, factors = rescale_errors(NULL, data)
        r = onp.asarray(whitened_residuals(NULL, scaled))
        assert onp.sum(r[:21] ** 2) / 21 == pytest.approx(1.0, rel=1e-4)
        assert onp.sum(r[21:] ** 2) / n_cp == pytest.approx(1.0, rel=1e-4)
        onp.testing.assert_allclose(
            scaled.d_vis, data.d_vis * factors["vis"], rtol=1e-6
        )
        for name in squares:
            squares[name].append(factors[name] ** 2)
    # The mean squared factor estimates the true variance ratio (420 and
    # 300 degrees of freedom: about 7% and 8% standard error).
    assert onp.mean(squares["vis"]) == pytest.approx(9.0, rel=0.2)
    assert onp.mean(squares["phi"]) == pytest.approx(0.25, rel=0.2)


def test_sign_flip_bootstrap_keeps_each_whitened_residual_magnitude():
    template = _unequal_template()
    data = template.with_model(NULL, key=jax.random.PRNGKey(0))
    simulate = bootstrap_null(NULL, data)
    noise = data.cp_noise
    resid = onp.asarray(data.vis) - onp.asarray(TEMPLATE.model(NULL)[:21])
    w = onp.asarray(noise.whiten(data.phi, data.d_phi)[0])
    signs = set()
    for seed in range(5):
        draw = simulate(jax.random.PRNGKey(seed))
        r_vis = onp.asarray(draw.vis) - 1.0  # the null's V² are 1
        onp.testing.assert_allclose(onp.abs(r_vis), onp.abs(resid), atol=1e-6)
        w_draw = onp.asarray(noise.whiten(draw.phi, draw.d_phi)[0])
        onp.testing.assert_allclose(onp.abs(w_draw), onp.abs(w), atol=1e-4)
        signs.add(tuple(onp.sign(w_draw * w).astype(int)))
        assert onp.array_equal(onp.asarray(draw.d_vis), data.d_vis)
    assert len(signs) == 5
    # Resampling draws the whitened visibility residuals with replacement.
    resample = bootstrap_null(NULL, data, method="resample")
    draw = resample(jax.random.PRNGKey(1))
    z = resid / onp.asarray(data.d_vis)
    z_draw = (onp.asarray(draw.vis) - 1.0) / onp.asarray(data.d_vis)
    assert all(onp.min(onp.abs(z - x)) < 1e-4 for x in z_draw)


@pytest.mark.validates("virgil.detection.bootstrap_null", roots=["statistics"])
def test_bootstrap_keeps_the_closure_phase_covariance_on_average():
    # Over many Gaussian datasets with unequal errors, the bootstrapped
    # closure phases whiten to unit covariance: the triangles' correlations
    # survive the bootstrap.
    template = _unequal_template()
    noise = template.cp_noise
    n_data, n_draw = 150, 4

    def whitened_draws(key):
        data_key, draw_key = jax.random.split(key)
        data = template.with_model(NULL, key=data_key)
        simulate = bootstrap_null(NULL, data)
        phi = jax.vmap(lambda k: simulate(k).phi)(
            jax.random.split(draw_key, n_draw)
        )
        return jax.vmap(lambda p: noise.whiten(p, template.d_phi)[0])(phi)

    keys = jax.random.split(jax.random.PRNGKey(0), n_data)
    w = onp.asarray(jax.jit(jax.vmap(whitened_draws))(keys))
    w = w.reshape(-1, noise.size)
    assert w.shape[1] == 15
    cov = w.T @ w / w.shape[0]
    # Sign flips of one dataset keep each |w|, so the draws add no
    # variance to the diagonal: the effective sample is the n_data
    # datasets. A sample variance of unit normals then has a standard
    # error of sqrt(2 / n_eff) (0.12), a covariance at most 1 / sqrt(n_eff)
    # (0.08), and the mean of the 15 diagonal terms sqrt(2 / (15 n_eff)).
    n_eff = n_data
    diag, off = onp.diag(cov), cov[~onp.eye(15, dtype=bool)]
    assert onp.all(onp.abs(diag - 1.0) < 5.0 * onp.sqrt(2.0 / n_eff))
    assert onp.all(onp.abs(off) < 5.0 / onp.sqrt(n_eff))
    assert onp.mean(diag) == pytest.approx(
        1.0, abs=5.0 * onp.sqrt(2.0 / (15 * n_eff))
    )


def test_bootstrap_rejects_unsupported_data():
    data = TEMPLATE.with_model(NULL, key=jax.random.PRNGKey(0))
    with pytest.raises(ValueError, match="method"):
        bootstrap_null(NULL, data, method="jackknife")
    # Any gain model is refused (a placeholder stands in for one).
    gains = eqx.tree_at(
        lambda d: d.gains, data, "gains", is_leaf=lambda x: x is None
    )
    with pytest.raises(ValueError, match="gains"):
        bootstrap_null(NULL, gains)


def test_gaussian_null_error_scale_scales_the_noise_not_the_errors():
    keys = jax.random.split(jax.random.PRNGKey(0), 200)
    for scale in (1.0, 2.0):
        simulate = gaussian_null(NULL, TEMPLATE, error_scale=scale)
        vis = onp.asarray(jax.vmap(lambda k: simulate(k).vis)(keys))
        assert onp.std(vis - 1.0) == pytest.approx(0.01 * scale, rel=0.1)
        assert onp.array_equal(simulate(keys[0]).d_vis, TEMPLATE.d_vis)
    with pytest.raises(ValueError, match="error_scale"):
        gaussian_null(NULL, TEMPLATE, error_scale=-1.0)


def test_default_grid_batch_size_is_split_among_draw_batch(monkeypatch):
    seen = []
    real = detection._simulated_statistics

    def spy(*args, **kwargs):
        seen.append(kwargs["batch_size"])
        return real(*args, **kwargs)

    monkeypatch.setattr(detection, "_simulated_statistics", spy)
    default = detection.batch_size_or_default(None, TEMPLATE)
    _recover(n_null=4, draw_batch=1)
    _recover(n_null=4, draw_batch=4)
    _recover(n_null=4, draw_batch=4, batch_size=64)
    assert seen[0] == default
    assert seen[1] == default // 4
    assert seen[2] == 64  # an explicit value is used as given
    # A draw_batch above the default is capped at it, so the product of
    # draws and grid points evaluated together stays within the budget.
    for draw_batch in (1, 4, default, 2 * default, 10**9):
        draws, grid = detection._batch_sizes(None, TEMPLATE, draw_batch)
        assert draws == min(draw_batch, default)
        assert grid >= 1 and draws * grid <= default
    assert detection._batch_sizes(64, TEMPLATE, 10**9) == (10**9, 64)


def test_last_chunk_is_not_padded_unless_chunk_size_is_set(monkeypatch):
    widths = []
    real = detection._simulated_statistics

    def spy(simulator, model, samples_dict, base_key, index, values, **kw):
        widths.append(values.shape[0])
        return real(
            simulator, model, samples_dict, base_key, index, values, **kw
        )

    monkeypatch.setattr(detection, "_simulated_statistics", spy)
    inj = injection_grid([60.0], [1e-2], 3, 1)  # 3 injected draws
    auto = _recover(n_null=5, injections=inj, chunk_size=None)
    # Auto: one chunk of 5 null and one of 3 injected, none padded.
    assert widths == [5, 3]
    widths.clear()
    fixed = _recover(n_null=5, injections=inj, chunk_size=4)
    # Explicit chunk_size pads the short chunks to one compiled shape.
    assert widths == [4, 4, 4]
    for part in ("null", "injected"):
        for k, v in getattr(auto, part).items():
            onp.testing.assert_allclose(
                v, getattr(fixed, part)[k], rtol=1e-4, atol=1e-6
            )
    assert auto.n_null == 5 and auto.n_injected == 3


def test_remainder_chunk_matches_unchunked_run():
    # 70 > the default chunk of 64: a full chunk plus a remainder of 6.
    a = _recover(n_null=70, chunk_size=None)
    b = _recover(n_null=70, chunk_size=70)
    assert a.n_null == 70
    onp.testing.assert_allclose(
        a.null["delta_chi2"], b.null["delta_chi2"], rtol=1e-4, atol=1e-6
    )
