"""Period aliases of an undersampled orbit: bands, window and evidences."""

import copy
import dataclasses
import functools
import json
import types
import warnings

import equinox as eqx
import jax
import jax.numpy as np
import numpy as onp
import pytest
from scipy.special import gammaln

pytest.importorskip("jaxoplanet")

from virgil import _gauss_newton, aliases  # noqa: E402
from virgil.epochs import Epochs, epoch_positions  # noqa: E402
from virgil.aliases import (  # noqa: E402
    AliasBandResult,
    AliasResult,
    alias_bands,
    fit_orbit_aliases,
    position_profile,
)
from virgil._precision import in_float64, cast_tree  # noqa: E402
from virgil.models import OrbitalBinary  # noqa: E402
from virgil.orbits import KeplerOrbit, PositionData  # noqa: E402
from virgil.simulate import simulate  # noqa: E402
from tests._shared import vlti_night  # noqa: E402
from scipy.special import logsumexp  # noqa: E402

T_REF = 60000.0
TRUTH = dict(
    period=12.1,
    dt_peri=2.0,
    ecc=0.3,
    inc=50.0,
    omega=60.0,
    Omega=120.0,
    a_mas=10.0,
)
FLUX = 0.3
EPOCHS = T_REF + onp.array([0.0, 2.4, 5.1, 238.0, 240.6, 247.4])


def _night(mjd, frames=3, rotation=0.0, d_phi=0.02):
    return vlti_night(mjd, frames, rotation, d_phi=d_phi)


def test_alias_bands_tile_the_period_range():
    times = [0.0, 100.0, 413.89]
    bands = alias_bands(times, (10.0, 14.0))
    assert [b.n for b in bands] == list(range(30, 42))
    n34 = next(b for b in bands if b.n == 34)
    assert n34.p_lo < 12.137 < n34.p_hi
    for lo, hi in zip(bands[1:], bands[:-1]):  # contiguous in period
        assert onp.isclose(lo.p_hi, hi.p_lo)


def test_position_profile_has_a_chi2_per_band():
    orbit = KeplerOrbit(**TRUTH, t_ref=T_REF)
    dra, ddec, _ = orbit.relative(EPOCHS)
    cov = onp.tile(0.04 * onp.eye(2), (len(EPOCHS), 1, 1))
    pos = PositionData(EPOCHS, dra, ddec, cov, t_ref=T_REF)
    bands = alias_bands(EPOCHS, (11.0, 13.0))
    rows = position_profile(pos, bands, k=3.0, n_best=1)
    assert [r["n"] for r in rows] == [b.n for b in bands]
    assert all(onp.isfinite(r["chi2"]) for r in rows)


def _simulate_orbit(period):
    truth = {**TRUTH, "period": period}
    orbit = KeplerOrbit(**truth, t_ref=T_REF)
    scene = OrbitalBinary(orbit, FLUX)
    key = jax.random.PRNGKey(3)
    data = [
        simulate(
            scene, _night(m, rotation=7.0 * k), key=jax.random.fold_in(key, k)
        )
        for k, m in enumerate(EPOCHS)
    ]
    dra, ddec, _ = orbit.relative(EPOCHS)
    rng = onp.random.default_rng(1)
    pos = PositionData(
        EPOCHS,
        onp.asarray(dra) + 0.2 * rng.standard_normal(len(EPOCHS)),
        onp.asarray(ddec) + 0.2 * rng.standard_normal(len(EPOCHS)),
        onp.tile(0.04 * onp.eye(2), (len(EPOCHS), 1, 1)),
        t_ref=T_REF,
    )
    return data, pos


_synthetic = functools.cache(_simulate_orbit)


@functools.cache
def _synthetic_as(period, x64):
    """The same data simulated in a fixed precision, which sets the noise
    draws: ``_synthetic`` takes whichever precision its first caller has."""
    with jax.enable_x64(x64):
        return _simulate_orbit(period)


def test_alias_bands_include_the_long_period_band():
    bands = alias_bands([0.0, 100.0], (50.0, 500.0))
    assert (
        bands[0].n == 0 and bands[0].p_lo == 200.0 and bands[0].p_hi == 500.0
    )


# 12.38 d sits mid-band N = 20; 12.1 d is 0.05 cycles from the N = 20/21 edge.
# A few seeds suffice now that the optimizer is damped Newton on the IRLS
# curvature (aliases._optimize); L-BFGS-B stalled short of the ridge and
# 1 in 4 noise draws settled in a wrong alias at chi2_red ~ 100-800.
@pytest.mark.slow
@pytest.mark.filterwarnings("error:fit_orbit_aliases")  # a good fit is quiet
@pytest.mark.parametrize("period", [12.38, 12.1])
def test_injected_alias_wins_in_closure_phases(period, tmp_path):
    data, pos = _synthetic(period)
    result = fit_orbit_aliases(
        data,
        (11.3, 12.9),
        positions=pos,
        t_ref=T_REF,
        n_candidates=30,
        n_refine=3,
        n_is=400,
        n_samples=50,
        primary_diam=None,  # PositionData cannot estimate a diameter
    )
    n_true = round(float(onp.ptp(result.times)) / period)
    assert result.best.n == n_true
    assert result.best.p > 0.9
    assert abs(result.best.best["period"] - period) < 0.05
    assert result.best.chi2_red < 3.0  # raw χ²/ν on the quoted errors
    assert (
        n_true in result.samples
        and len(result.samples[n_true]["period"]) == 50
    )
    s = result.samples[n_true]
    assert onp.all((s["Omega"] >= 0) & (s["Omega"] < 180))  # mirror folded
    row = result.table()[0]
    assert row["n"] == n_true and "flags" in row
    result.to_json(tmp_path / "bands.json", n_samples=5)
    json.loads(
        (tmp_path / "bands.json").read_text()
    )  # strict JSON: no Infinity


@pytest.mark.slow
def test_a_band_that_cannot_fit_the_data_warns_loudly():
    """No orbit in 13.5-14.5 d fits a 12.38 d orbit's data, so the best
    band is flagged poor-fit and the call warns rather than failing
    silently."""
    data, pos = _synthetic(12.38)
    with pytest.warns(UserWarning, match="missed the narrow"):
        result = fit_orbit_aliases(
            data,
            (13.5, 14.5),
            positions=pos,
            t_ref=T_REF,
            n_candidates=10,
            n_refine=2,
            n_is=100,
            n_samples=10,
        )
    assert result.best.chi2_red > 10
    assert "poor-fit" in result.best.flags and result.best.flagged


def test_random_starts_and_s_max_reach_the_settings_without_positions(
    monkeypatch,
):
    # Without positions there are no seeds: the requested random starts and
    # error-inflation bound must arrive in the settings unchanged.
    settings = _spy_settings(
        monkeypatch, p_range=(12.1, 12.7), n_random=64, s_max=50.0
    )["settings"]
    assert (settings.n_random, settings.s_max) == (64, 50.0)


def _settings(**kw):
    base = dict(
        p_range=(11.0, 13.0), a_range=(1.0, 30.0), flux_range=(0.01, 1.0),
        ecc_max=0.9, s_max=None, t_ref=T_REF, n_candidates=2, n_refine=3,
        n_random=30, pool_size=64, k=3.0, eccs=None, min_distance_mas=0.5,
        max_steps=5, n_is=20, flag_nats=1.0, ess_min=50.0, seed=0,
    )  # fmt: skip
    return aliases._Settings(**{**base, **kw})


def _bounds(lower=None, upper=None):
    inf = onp.inf
    lo = [-inf, -inf, 0.0, -1.0, -inf, -inf, -inf, -inf]
    hi = [inf, inf, 0.9, 1.0, inf, inf, inf, inf]
    return aliases._Bounds(
        onp.array(lo if lower is None else lower),
        onp.array(hi if upper is None else upper),
    )


# --- robustness: the failures of OzSTAR job 18296568 ----------------------------------


def _band(n, epochs, **kw):
    return AliasBandResult(
        n=n, p_lo=1.0, p_hi=2.0, modes=[], log_z=-1.0, log_z_is=-1.0, ess=10.0,
        flagged=False, epochs=epochs, **kw
    )  # fmt: skip


def test_table_scales_are_flat_floats_for_multi_block_epochs():
    # An epoch with V2 and closure-phase blocks has one scale per block (a list).
    epochs = [dict(scale=[1.0, 2.5]), dict(scale=[1.5])]
    res = AliasResult(0.0, onp.zeros(2), 1.0, [_band(3, epochs)], {})
    row = res.table()[0]
    assert row["scales"] == [1.0, 2.5, 1.5]
    assert row["epoch_scales"] == [[1.0, 2.5], [1.5]]
    assert f"{min(row['scales']):>6.2f}" and f"{max(row['scales']):>6.2f}"


def test_singular_hessian_is_flagged_not_raised():
    class Flat:
        def hessian(self, x):
            return -onp.diag([1.0, 1.0, 1e-20, 1.0, 1.0, 1.0, 1.0, 1.0])

    x = onp.zeros(8)
    rec, draw = aliases._mode_evidence(
        Flat(), _bounds(), x, -1.0, 10, onp.random.default_rng(0)
    )
    assert draw is None and rec["flags"] == ["hessian-singular"]
    assert rec["log_z_laplace"] == -onp.inf


def test_band_checkpoint_roundtrip_and_key(tmp_path):
    band = alias_bands([0.0, 100.0, 413.89], (10.0, 14.0))[0]
    key = dict(n_is=10, edges=[[1.0, 2.0]])
    done = (
        _band(band.n, [dict(scale=[1.0])]),
        (onp.ones((3, 2)), onp.zeros(3)),
    )
    assert aliases._checkpoint(tmp_path, band, key) is None
    aliases._checkpoint(tmp_path, band, key, done)
    json.loads((tmp_path / f"band_{band.n}.json").read_text())
    got = aliases._checkpoint(tmp_path, band, key)
    assert got[0].n == band.n and got[1][0].shape == (3, 2)
    assert aliases._checkpoint(tmp_path, band, {**key, "n_is": 11}) is None
    assert aliases._checkpoint(None, band, key) is None


@pytest.mark.slow
def test_one_failing_band_is_isolated_checkpointed_and_not_refit(
    monkeypatch, tmp_path
):
    data, _ = _synthetic(12.38)
    first = alias_bands(EPOCHS, (12.1, 12.7))[0].n
    calls = []
    real = aliases._fit_band

    def fit_band(problem, settings, band, orbits, times):
        calls.append(band.n)
        if band.n == first:
            raise RuntimeError("boom")
        return real(problem, settings, band, orbits, times)

    monkeypatch.setattr(aliases, "_fit_band", fit_band)
    kw = dict(
        t_ref=T_REF,
        n_random=20,
        n_refine=1,
        max_steps=3,
        n_is=20,
        n_samples=5,
        checkpoint_dir=tmp_path,
    )
    seen = []
    result = fit_orbit_aliases(data, (12.1, 12.7), on_band=seen.append, **kw)
    assert len(seen) == len(result.bands) > 1
    bad = [b for b in result.bands if "band-failed" in b.flags]
    assert [b.n for b in bad] == [first] and "boom" in bad[0].error
    others = [b for b in result.bands if b.n != first]
    assert {p.name for p in tmp_path.glob("band_*.pkl")} == {
        f"band_{b.n}.pkl" for b in others
    }
    assert not (
        tmp_path / f"band_{first}.pkl"
    ).exists()  # a failed band is retried
    result.to_json(tmp_path / "bands.json")
    json.loads((tmp_path / "bands.json").read_text())
    # A rerun with the same arguments loads the good bands and refits only the failed one.
    calls.clear()
    fit_orbit_aliases(data, (12.1, 12.7), **kw)
    assert calls == [first]
    # A changed setting (s_max) invalidates the checkpoints.
    calls.clear()
    fit_orbit_aliases(data, (12.1, 12.7), s_max=50.0, **kw)
    assert len(calls) == len(result.bands)


@pytest.mark.slow
def test_resumed_run_equals_uninterrupted_run(tmp_path):
    data, pos = _synthetic(12.1)
    kw = dict(
        positions=pos, t_ref=T_REF, k=3.0, n_candidates=40, n_refine=4,
        n_is=400, n_samples=10,
    )  # fmt: skip
    full = fit_orbit_aliases(
        data, (11.3, 12.9), checkpoint_dir=tmp_path / "full", **kw
    )

    class Interrupted(Exception):
        pass

    def stop(band):
        raise Interrupted

    with pytest.raises(Interrupted):
        fit_orbit_aliases(
            data, (11.3, 12.9), checkpoint_dir=tmp_path / "run", on_band=stop,
            **kw,
        )  # fmt: skip
    assert len(list((tmp_path / "run").glob("band_*.pkl"))) == 1
    loaded = []
    resumed = fit_orbit_aliases(
        data, (11.3, 12.9), checkpoint_dir=tmp_path / "run",
        on_band=loaded.append, **kw,
    )  # fmt: skip
    assert len(loaded) == len(full.bands) > 1
    assert [b.n for b in resumed.bands] == [b.n for b in full.bands]
    for a, b in zip(resumed.bands, full.bands):
        assert a.log_z == b.log_z and a.log_z_is == b.log_z_is
        assert a.p == b.p
    assert resumed.samples.keys() == full.samples.keys()
    assert any(onp.isfinite(b.log_z) for b in full.bands)
    for n, draws in full.samples.items():
        for name, values in draws.items():
            assert onp.array_equal(resumed.samples[n][name], values)


# --- failure semantics --------------------------------------------------------


def test_every_band_failing_raises_chained(monkeypatch):
    data, _ = _synthetic(12.38)
    boom = RuntimeError("boom")

    def fit_band(*args):
        raise boom

    monkeypatch.setattr(aliases, "_fit_band", fit_band)
    with pytest.raises(RuntimeError, match="Every band failed") as caught:
        fit_orbit_aliases(data, (12.1, 12.7), t_ref=T_REF)
    assert caught.value.__cause__ is boom


def test_type_error_in_a_band_propagates(monkeypatch):
    data, _ = _synthetic(12.38)
    calls = []

    def fit_band(problem, settings, band, orbits, times):
        calls.append(band.n)
        raise TypeError("a misspelt option")

    monkeypatch.setattr(aliases, "_fit_band", fit_band)
    with pytest.raises(TypeError, match="misspelt"):
        fit_orbit_aliases(data, (12.1, 12.7), t_ref=T_REF)
    assert len(calls) == 1  # not isolated as a failed band


def test_value_error_in_one_band_gives_a_band_failed_row(monkeypatch):
    data, _ = _synthetic(12.38)
    seen = []

    def fit_band(problem, settings, band, orbits, times):
        if not seen:
            seen.append(band.n)
            raise ValueError("a bad band")
        return _band(band.n, []), (None, None)

    monkeypatch.setattr(aliases, "_fit_band", fit_band)
    result = fit_orbit_aliases(
        data, (11.3, 12.9), t_ref=T_REF, on_band=lambda b: None
    )
    failed = [b for b in result.bands if "band-failed" in b.flags]
    assert [b.n for b in failed] == seen
    assert "a bad band" in failed[0].error
    assert len(failed) < len(result.bands)


def test_misconfiguration_surfaces_in_the_preflight(monkeypatch):
    data, _ = _synthetic(12.38)
    calls = []

    def value_and_grad(self, x):
        raise ValueError("bad data shape")

    monkeypatch.setattr(aliases._Problem, "value_and_grad", value_and_grad)
    monkeypatch.setattr(aliases, "_fit_band", lambda *args: calls.append(args))
    with pytest.raises(ValueError, match="bad data shape"):
        fit_orbit_aliases(data, (12.1, 12.7), t_ref=T_REF)
    assert calls == []  # no band was attempted


@pytest.mark.filterwarnings("error:Python iterable type")
def test_data_given_as_a_generator_is_cast_to_float64(monkeypatch):
    data, _ = _synthetic(12.38)
    seen = {}

    def preflight(problem, settings, band):
        seen["dtypes"] = {
            leaf.dtype
            for leaf in jax.tree.leaves(problem.data)
            if hasattr(leaf, "dtype") and leaf.dtype.kind == "f"
        }
        raise KeyboardInterrupt

    monkeypatch.setattr(aliases, "_preflight", preflight)
    with pytest.raises(KeyboardInterrupt):
        fit_orbit_aliases(iter(data), (12.1, 12.7), t_ref=T_REF)
    assert seen["dtypes"] == {onp.dtype("float64")}


class _Scores:
    """A problem whose candidates score by their position in the pool."""

    def batch(self, xs):
        return -onp.arange(len(xs), dtype=float)


def _refine_inputs():
    band = alias_bands(EPOCHS, (12.1, 12.7))[0]
    settings = _settings()
    bounds = aliases._Bounds.of(band, settings)
    rng = onp.random.default_rng(0)
    pool = aliases._candidates(settings, band, bounds, rng, None)
    return band, settings, bounds, pool


def test_failed_starts_are_counted_and_the_others_go_on(monkeypatch):
    band, settings, bounds, pool = _refine_inputs()
    calls = []

    def optimize(problem, bounds, x0, max_steps):
        calls.append(x0)
        if len(calls) == 1:
            raise RuntimeError("one bad start")
        return x0, -1.0

    monkeypatch.setattr(aliases, "_optimize", optimize)
    modes, n_failed = aliases._refine(
        _Scores(), settings, bounds, pool, EPOCHS
    )
    assert n_failed == 1 and len(calls) == settings.n_refine
    assert len(modes) == settings.n_refine - 1
    result = aliases._band_result(
        None, settings, band, [], -onp.inf, -onp.inf, n_failed
    )
    assert "starts-failed" in result.flags and result.n_starts_failed == 1


def test_a_non_finite_value_drops_the_start_without_counting_it(
    monkeypatch,
):
    band, settings, bounds, pool = _refine_inputs()
    monkeypatch.setattr(
        aliases, "_optimize", lambda p, b, x0, m: (x0, float("nan"))
    )
    modes_xy, n_failed = aliases._refine(
        _Scores(), settings, bounds, pool, EPOCHS
    )
    assert modes_xy == [] and n_failed == 0
    modes, z_lap, z_is, draws = aliases._band_evidence(
        None, settings, bounds, modes_xy, onp.random.default_rng(0)
    )
    assert modes == [] and z_lap == -onp.inf and z_is == -onp.inf
    assert draws == (None, None)
    result = aliases._band_result(
        None, settings, band, modes, z_lap, z_is, n_failed
    )
    assert result.flags == ["no-valid-mode"] and result.log_z == -onp.inf


@pytest.mark.parametrize("error", [ValueError, onp.linalg.LinAlgError])
def test_value_and_linalg_errors_count_as_failed_starts(monkeypatch, error):
    _, settings, bounds, pool = _refine_inputs()
    calls = []

    def optimize(problem, bounds, x0, max_steps):
        calls.append(x0)
        if len(calls) == 1:
            raise error("singular")
        return x0, -1.0

    monkeypatch.setattr(aliases, "_optimize", optimize)
    modes, n_failed = aliases._refine(
        _Scores(), settings, bounds, pool, EPOCHS
    )
    assert n_failed == 1 and len(calls) == settings.n_refine
    assert len(modes) == settings.n_refine - 1


def test_every_start_failing_raises_and_type_error_propagates(
    monkeypatch,
):
    _, settings, bounds, pool = _refine_inputs()
    error = RuntimeError("solver")

    def failing(problem, bounds, x0, max_steps):
        raise error

    monkeypatch.setattr(aliases, "_optimize", failing)
    with pytest.raises(RuntimeError, match="starts failed") as caught:
        aliases._refine(_Scores(), settings, bounds, pool, EPOCHS)
    assert caught.value.__cause__ is error

    def value_error(problem, bounds, x0, max_steps):
        raise ValueError("singular")

    monkeypatch.setattr(aliases, "_optimize", value_error)
    with pytest.raises(RuntimeError, match="starts failed") as caught:
        aliases._refine(_Scores(), settings, bounds, pool, EPOCHS)
    assert isinstance(caught.value.__cause__, ValueError)

    def misconfigured(problem, bounds, x0, max_steps):
        raise TypeError("bad option")

    monkeypatch.setattr(aliases, "_optimize", misconfigured)
    with pytest.raises(TypeError, match="bad option"):
        aliases._refine(_Scores(), settings, bounds, pool, EPOCHS)


# --- the checkpoint key ----------------------------------------------------------


def _with(obj, **fields):
    """A copy of an equinox module with (static) fields replaced."""
    new = copy.copy(obj)
    for name, value in fields.items():
        object.__setattr__(new, name, value)
    return new


def test_checkpoint_key_sees_static_fields(tmp_path):
    data, pos = _synthetic(12.38)
    bands = alias_bands(EPOCHS, (12.1, 12.7))
    settings = _settings()

    def key(data=data, pos=pos):
        return aliases._checkpoint_key(data, pos, bands, EPOCHS, settings)

    base = key()
    assert base["format"] == aliases._CHECKPOINT_FORMAT
    assert key() == base
    flipped = [_with(data[0], v2_flag=not data[0].v2_flag)] + data[1:]
    assert key(data=flipped)["data"] != base["data"]
    moved = [_with(data[0], t_ref=(data[0].t_ref or 0.0) + 1.0)] + data[1:]
    assert key(data=moved)["data"] != base["data"]
    moved_pos = _with(pos, t_ref=pos.t_ref + 1.0)
    assert key(pos=moved_pos)["positions"] != base["positions"]
    # A checkpoint saved under one key is not loaded under another.
    band = bands[0]
    done = (_band(band.n, [dict(scale=[1.0])]), (None, None))
    aliases._checkpoint(tmp_path, band, base, done)
    assert aliases._checkpoint(tmp_path, band, base) is not None
    assert aliases._checkpoint(tmp_path, band, key(data=flipped)) is None
    assert aliases._checkpoint(tmp_path, band, {**base, "format": 1}) is None


def test_checkpoint_key_refuses_unhashable_objects():
    data, _ = _synthetic(12.38)
    bands = alias_bands(EPOCHS, (12.1, 12.7))
    with pytest.raises(ValueError, match="hashed"):
        aliases._checkpoint_key(data, lambda: 0, bands, EPOCHS, _settings())


# --- the evidence ----------------------------------------------------------------

N_IS = 4000


class _GaussianProblem:
    """An 8-D Gaussian likelihood, whose evidence is known."""

    log_prior = -3.0
    peak = 2.5

    def __init__(self):
        sd = onp.array([0.02, 0.01, 0.02, 0.02, 1.0, 0.5, 0.05, 0.05])
        corr = 0.7 * onp.eye(8) + 0.3
        self.cov = sd[:, None] * corr * sd[None, :]
        self.prec = onp.linalg.inv(self.cov)
        self.mu = onp.array([2.5, 0.1, 0.4, 0.2, 100.0, 60.0, 2.3, -1.2])

    def value(self, x):
        d = onp.asarray(x) - self.mu
        quad = onp.einsum("...i,ij,...j->...", d, self.prec, d)
        return self.peak - 0.5 * quad

    def value_and_grad(self, x):
        return self.value(x), -self.prec @ (onp.asarray(x) - self.mu)

    def hessian(self, x):
        return -self.prec

    def batch(self, xs):
        assert len(xs) == N_IS  # every draw, at one shape
        return self.value(onp.asarray(xs))

    def log_evidence(self):
        logdet = onp.linalg.slogdet(self.cov)[1]
        return (
            self.peak + self.log_prior + 4 * onp.log(2 * onp.pi) + 0.5 * logdet
        )


def test_gaussian_evidence_laplace_exact_and_importance_sampling_close():
    problem = _GaussianProblem()
    bounds = _bounds(
        lower=[2.0, -onp.inf, 0.0, -1.0, -onp.inf, -onp.inf, 1.8, -1.7]
    )
    rec, draws = aliases._mode_evidence(
        problem, bounds, problem.mu, problem.peak, N_IS,
        onp.random.default_rng(1),
    )  # fmt: skip
    assert rec["flags"] == []
    assert abs(rec["log_z_laplace"] - problem.log_evidence()) < 1e-8
    assert rec["ess"] > 1000
    sigma = 1.0 / onp.sqrt(rec["ess"])  # relative error of the mean weight
    assert abs(rec["log_z_is"] - problem.log_evidence()) < 4 * sigma
    # The mixture of one mode is its own proposal.
    xs, log_w = aliases._mixture_weights([draws])
    z_is = logsumexp(log_w) - onp.log(len(xs))
    assert abs(z_is - rec["log_z_is"]) < 1e-9


# The log likelihood at the optimum of each near start below, from main's
# L-BFGS-B-with-restarts ``_optimize`` (commit 5f600a3); all six reach it.
# The fit now carries the mixture's Occam factor for its six nights
# (``_occam(6)``; the outlier level of ``_problem`` is -1e6), which is the
# only change from that value.
def _occam(k):
    """ln Γ(K + ½) / (Γ(K + 1) √π): the Beta(½, ½) mean of (1 - ε)^K."""
    return float(gammaln(k + 0.5) - gammaln(k + 1.0) - 0.5 * onp.log(onp.pi))


_OLD_NEAR = [-193.286626 + _occam(6)] * 6


@in_float64
@pytest.mark.slow
def test_the_kernel_optimizer_matches_the_old_one_from_near_starts():
    data, _ = _synthetic_as(12.1, True)
    settings, problem = _problem(data, max_steps=200)
    band = aliases.AliasBand(20, 12.0, 12.2)
    bounds = aliases._Bounds.of(band, settings)
    rng = onp.random.default_rng(0)
    sigma = onp.array([2e-3, 0.02, 0.05, 0.05, 10.0, 10.0, 0.1, 0.1])
    for old in _OLD_NEAR:
        x0 = bounds.clip(_x_truth() + sigma * rng.standard_normal(8))
        x, value = aliases._optimize(problem, bounds, x0, 200)
        assert value >= old - 0.01
        assert value == pytest.approx(
            float(problem.value_and_grad(jax.numpy.asarray(x))[0]), abs=1e-9
        )
        assert onp.all(bounds.inside(x))


@in_float64
@pytest.mark.parametrize(
    "vis_amp", [None, pytest.param((0.01, 1.5), marks=pytest.mark.slow)]
)
def test_the_kernel_rows_carry_the_gradient_of_the_likelihood(vis_amp):
    # With the weights frozen, Jᵀρ is the gradient of -log likelihood
    # exactly (Σ_b w_b/2 ∇χ²_b), also through the live V² amplitude.
    data, _ = _synthetic(12.1)
    _, problem = _problem(data, vis_amp=vis_amp)
    x = jax.numpy.asarray(_x_truth() + 0.01)
    kernel = problem.kernel
    rows = kernel.rows(x)
    jac = jax.jacfwd(kernel.rows)(x)
    onp.testing.assert_allclose(
        jac.T @ rows, jax.grad(kernel.loss)(x), rtol=1e-8
    )
    assert float(kernel.loss(x)) == pytest.approx(
        -float(problem.value_and_grad(x)[0])
    )


@in_float64
@pytest.mark.slow
def test_a_kernel_that_stalls_falls_back_to_lbfgs_and_then_the_kernel(
    monkeypatch,
):
    data, _ = _synthetic_as(12.1, True)
    settings, problem = _problem(data, max_steps=200)
    band = aliases.AliasBand(20, 12.0, 12.2)
    bounds = aliases._Bounds.of(band, settings)
    x0 = bounds.clip(
        _x_truth() + onp.array([1e-3, 0.01, 0.02, 0.02, 3, 3, 0.03, 0.03])
    )
    real, calls = _gauss_newton.solve, []

    def truncate_first(problem, start, **kw):
        calls.append(start)
        if len(calls) == 1:  # a real stall: stopped short of the optimum
            kw = {**kw, "max_steps": 2}
        return real(problem, start, **kw)

    monkeypatch.setattr(_gauss_newton, "solve", truncate_first)
    x, value = aliases._optimize(problem, bounds, x0, 200)
    assert len(calls) == 2  # the kernel (cut short), L-BFGS-B, the kernel
    assert value == pytest.approx(-193.286626 + _occam(6), abs=0.01)


# Band log evidences (Laplace) of the fit below from main's optimizer (commit
# 5f600a3), by N. Band 22 (-460.15, three modes) is left out: its modes come
# from starts at the edge of the period range, and which local optimum such
# a start reaches differs between the optimizers (the kernel finds -414.3
# and -416.3 where main also found -406.4).
_OLD_LOG_Z = {19: -388.837124, 20: -262.393553, 21: -400.143249}


@pytest.mark.slow
def test_band_evidences_are_no_lower_than_with_the_old_optimizer():
    data, pos = _synthetic_as(12.1, False)
    result = fit_orbit_aliases(
        data,
        (11.3, 12.9),
        positions=pos,
        t_ref=T_REF,
        n_candidates=30,
        n_refine=3,
        n_is=400,
        n_samples=50,
    )
    by_n = {b.n: b for b in result.bands}
    for n, old in _OLD_LOG_Z.items():
        assert by_n[n].log_z >= old - 0.05


# --- UD primary, V² amplitudes, the flux flip and the warm start -----------------------


def _problem(data, **kw):
    kw.setdefault(
        "log_broad", (-1e6,) * len(data)
    )  # no outliers: a pure Occam factor
    settings = _settings(**kw)
    return settings, aliases._Problem(
        [cast_tree(d, "float64") for d in data], settings
    )


def _x_truth(flux=FLUX, diam=None):
    orbit = KeplerOrbit(**TRUTH, t_ref=T_REF)
    band = aliases.AliasBand(20, 12.0, 12.2)
    settings = _settings(primary_diam=None if diam is None else "fit")
    bounds = aliases._Bounds.of(band, settings)
    return aliases._start_vector(orbit, flux, T_REF, band, bounds, diam)


@in_float64
def test_a_vanishing_ud_primary_is_the_point_binary():
    data, _ = _synthetic(12.1)
    x = _x_truth()
    _, point = _problem(data)
    _, ud = _problem(data, primary_diam=1e-4)
    assert float(ud.value_and_grad(jax.numpy.asarray(x))[0]) == pytest.approx(
        float(point.value_and_grad(jax.numpy.asarray(x))[0]), abs=1e-3
    )


@in_float64
def test_a_fitted_diameter_is_recovered():
    truth = 3.0
    orbit = KeplerOrbit(**TRUTH, t_ref=T_REF)
    scene = aliases._UDBinary(orbit, FLUX, truth)
    key = jax.random.PRNGKey(5)
    data = [
        simulate(
            scene, _night(m, rotation=7.0 * k), key=jax.random.fold_in(key, k)
        )
        for k, m in enumerate(EPOCHS)
    ]
    settings, problem = _problem(data, primary_diam="fit", max_steps=60)
    band = aliases.AliasBand(20, 12.0, 12.2)
    bounds = aliases._Bounds.of(band, settings)
    x0 = _x_truth(diam=1.0)
    x, value = aliases._optimize(problem, bounds, x0, 60)
    assert onp.exp(x[8]) == pytest.approx(truth, rel=0.1)
    assert value > float(problem.value_and_grad(jax.numpy.asarray(x0))[0])


@in_float64
def test_the_epoch_v2_amplitude_is_profiled():
    data, _ = _synthetic(12.1)
    # Epoch 0's V² lost half its contrast.
    lossy = [data[0].set("vis", data[0].vis * 0.5), *data[1:]]
    x = jax.numpy.asarray(_x_truth())
    _, plain = _problem(lossy)
    _, amp = _problem(lossy, vis_amp=(0.01, 1.5))
    assert float(amp.value_and_grad(x)[0]) > float(plain.value_and_grad(x)[0])
    rows = amp.per_epoch(x)
    assert rows[0]["vis_amp"] == pytest.approx(0.5, rel=0.1)
    assert rows[1]["vis_amp"] == pytest.approx(1.0, rel=0.1)
    assert rows[0]["chi2_red"][0] < plain.per_epoch(x)[0]["chi2_red"][0]


@in_float64
def test_relabelling_the_stars_leaves_the_likelihood_unchanged():
    data, _ = _synthetic(12.1)
    settings, problem = _problem(data, flux_range=(0.01, 2.0))
    bounds = aliases._Bounds.of(aliases.AliasBand(20, 12.0, 12.2), settings)
    x = _x_truth(flux=0.97)
    y = aliases._flipped(x, settings, bounds)
    assert onp.exp(y[7]) == pytest.approx(1 / 0.97, rel=1e-3)
    value = lambda z: float(problem.value_and_grad(jax.numpy.asarray(z))[0])  # noqa: E731
    assert value(y) == pytest.approx(value(x), abs=0.05)


@pytest.mark.slow
def test_a_flux_ratio_above_one_is_refit_with_the_labels_swapped():
    orbit = KeplerOrbit(**TRUTH, t_ref=T_REF)
    # The companion is the brighter star: ratio 1.4 is outside (0.01, 1].
    scene = OrbitalBinary(orbit, 1.4)
    key = jax.random.PRNGKey(9)
    data = [
        simulate(
            scene, _night(m, rotation=7.0 * k), key=jax.random.fold_in(key, k)
        )
        for k, m in enumerate(EPOCHS)
    ]
    kw = dict(
        t_ref=T_REF, n_random=0, n_refine=1, n_is=200, n_samples=20,
        init=(orbit, 1.0),
    )  # fmt: skip
    plain = fit_orbit_aliases(data, (12.0, 12.2), flip_flux=False, **kw)
    flip = fit_orbit_aliases(data, (12.0, 12.2), **kw)
    assert flip.best.flipped
    assert flip.best.chi2_red < 3.0
    assert not plain.best.chi2_red < 3.0  # pinned at the bound, or no mode
    assert flip.best.best["flux"] == pytest.approx(1 / 1.4, rel=0.1)


@pytest.mark.slow
def test_a_warm_start_reproduces_the_optimum_without_other_seeds():
    data, _ = _synthetic(12.1)
    orbit = KeplerOrbit(**TRUTH, t_ref=T_REF)
    result = fit_orbit_aliases(
        data, (12.0, 12.2), t_ref=T_REF, n_random=0, n_refine=1, n_is=200,
        n_samples=20, init=(orbit, FLUX),
    )  # fmt: skip
    best = result.best.best
    assert result.best.chi2_red < 3.0
    assert best["period"] == pytest.approx(12.1, abs=0.02)
    assert best["flux"] == pytest.approx(FLUX, rel=0.2)
    assert not result.best.flipped


@in_float64
@in_float64
@pytest.mark.slow
def test_the_flip_acts_only_on_a_flux_pinned_at_the_top(monkeypatch):
    # The refit with swapped labels is tried for a mode within 0.02 (log
    # flux) of the top of ``flux_range``, and not below that.
    data, _ = _synthetic(12.1)
    flipped, orig = [], aliases._flipped
    monkeypatch.setattr(
        aliases, "_flipped", lambda *a: flipped.append(1) or orig(*a)
    )
    for log_below, expect in ((0.019, True), (0.03, False)):
        flux = float(onp.exp(-log_below))  # the top of flux_range is 1.0
        monkeypatch.setattr(
            aliases,
            "_refine",
            lambda *a, x=_x_truth(flux=flux): ([(x, -1.0)], 0),
        )
        flipped.clear()
        fit_orbit_aliases(
            data, (12.0, 12.2), t_ref=T_REF, n_is=50, n_samples=5,
            max_steps=2, short_arcs=False,
        )  # fmt: skip
        assert bool(flipped) is expect  # a swap, and its swap back


@in_float64
def test_the_flip_is_skipped_for_a_resolved_primary():
    data, _ = _synthetic(12.1)
    for kw, expect in (({}, True), ({"primary_diam": 0.5}, False)):
        _, problem = _problem(data, **kw)
        assert aliases._flip_is_relabelling(problem, _settings(**kw)) is expect


@in_float64
def test_a_warm_start_is_forced_into_its_band_whatever_its_rank(monkeypatch):
    data, _ = _synthetic(12.1)
    seen = []
    monkeypatch.setattr(
        aliases,
        "_refine",
        lambda *a: seen.append(a[5] if len(a) > 5 else ()) or ([], 0),
    )
    init = (12.1, 2.0, 0.3, 50.0, 60.0, 120.0, 10.0, 0.3)
    for p_lo, p_hi, expect in ((12.0, 12.2, 1), (11.0, 11.5, 0)):
        settings, problem = _problem(data, init=init)
        band = aliases.AliasBand(20, p_lo, p_hi)
        aliases._fit_band(problem, settings, band, None, EPOCHS)
        assert len(seen[-1]) == expect


def test_vis_amp_refuses_data_that_are_not_squared_visibilities():
    data, _ = _synthetic(12.1)
    d = copy.copy(data[0])
    object.__setattr__(d, "vis_mode", "amp")
    _, problem = _problem([d, *data[1:]], vis_amp=(0.01, 1.5))
    with pytest.raises(NotImplementedError, match="vis_amp"):
        problem.per_epoch(jax.numpy.asarray(_x_truth()))


# --- the result API: save/load, scene(), residuals(), chi2_per_night() --------------

DIAM = 0.5  # mas: a fixed uniform-disk primary, so the UD path is covered
VIS_AMP = (0.01, 1.5)
FAST = dict(n_candidates=10, n_refine=1, n_is=100, n_samples=8)


@functools.cache
def _tiny_fit(primary_diam=DIAM):
    """The result of a fit, built without running the optimizer (which
    costs tens of seconds): the truth orbit as the one mode of one band,
    through the fit's own ``_band_result``. float32 data, as a user has
    them; V² amplitudes are fitted."""
    data, _ = _synthetic_as(12.1, False)
    fitted = primary_diam == "fit"
    with jax.enable_x64(True):
        x = _x_truth(diam=1.0 if fitted else None)
        settings, problem = _problem(
            data, primary_diam=primary_diam, vis_amp=VIS_AMP
        )
        value = float(problem.value_and_grad(jax.numpy.asarray(x))[0])
        mode = dict(
            x=onp.asarray(x).tolist(), loglike=value, log_z=value,
            log_z_is=value, ess=100.0, flags=[],
        )  # fmt: skip
        band = aliases._band_result(
            problem, settings, aliases.AliasBand(20, 12.0, 12.2), [mode],
            value, value, 0,
        )  # fmt: skip
        band.p = 1.0
    result = AliasResult(
        T_REF, onp.asarray(EPOCHS), float(onp.ptp(EPOCHS)), [band],
        {20: aliases._physical(onp.asarray(x))},
        primary_diam=primary_diam, vis_amp=VIS_AMP,
    )  # fmt: skip
    return data, result


@functools.cache
def _real_fit(primary_diam):
    """A real, slow fit (the optimizer, evidences and draws included)."""
    data, pos = _synthetic_as(12.1, False)
    result = fit_orbit_aliases(
        data, (11.95, 12.25), positions=pos, t_ref=T_REF,
        primary_diam=primary_diam, vis_amp=VIS_AMP, **FAST,
    )  # fmt: skip
    return data, result


def _best_x(band):
    """The sampled vector of the band's best mode."""
    live = [m for m in band.modes if onp.isfinite(m["log_z"])]
    return onp.array(max(live, key=lambda m: m["log_z"])["x"])


def _objective(data, x, primary_diam=DIAM):
    """The fit's own scale-marginalized log likelihood at ``x``, in float64."""
    with jax.enable_x64(True):
        data = [cast_tree(d, "float64") for d in data]
        _, problem = _problem(data, primary_diam=primary_diam, vis_amp=VIS_AMP)
        return float(problem.value_and_grad(jax.numpy.asarray(x))[0])


def _score(data, nights, primary_diam=DIAM):
    """The same log likelihood from ``chi2_per_night``'s ``chi2_fit``, with
    the mixture's Occam factor Γ(K + ½) / (Γ(K + 1) √π) for K nights (the
    outlier level is -1e6, so no night is an outlier)."""
    with jax.enable_x64(True):
        data = [cast_tree(d, "float64") for d in data]
        _, problem = _problem(data, primary_diam=primary_diam, vis_amp=VIS_AMP)
        total = 0.0
        for s, d, row in zip(problem.surfaces, data, nights):
            chi2 = jax.numpy.asarray(
                [row["blocks"][n]["chi2_fit"] for n in s.names]
            )
            total += float(s.score(chi2, 1.0, d))
        k = len(data)
        return total + float(
            gammaln(k + 0.5) - gammaln(k + 1.0) - 0.5 * onp.log(onp.pi)
        )


def _same_epochs(a, b):
    """Per-field comparison of two ``epochs`` lists (NaN equal to NaN)."""
    assert len(a) == len(b)  # empty for a band with no valid mode
    for x, y in zip(a, b):
        assert x.keys() == y.keys()
        for k in x:
            if k == "blocks":  # block names
                assert x[k] == y[k]
            else:
                assert onp.allclose(x[k], y[k], rtol=1e-9, equal_nan=True), k


def test_a_saved_result_loads_back_equal(tmp_path):
    _, result = _tiny_fit()
    result.save(tmp_path / "fit.json")
    json.loads((tmp_path / "fit.json").read_text())  # strict JSON
    loaded = AliasResult.load(tmp_path / "fit.json")
    assert loaded.t_ref == result.t_ref and loaded.span == result.span
    assert onp.array_equal(loaded.times, result.times)
    assert loaded.primary_diam == DIAM and loaded.vis_amp == VIS_AMP
    assert loaded.positions is None
    assert [b.n for b in loaded.bands] == [b.n for b in result.bands]
    for a, b in zip(loaded.bands, result.bands):
        for name in ("log_z", "log_z_is", "ess", "p", "chi2_red"):
            got, want = getattr(a, name), getattr(b, name)
            assert got == pytest.approx(want, nan_ok=True), name
        assert a.flags == b.flags and a.flipped == b.flipped
        assert a.best.keys() == b.best.keys()
        for k in b.best:
            assert a.best[k] == pytest.approx(b.best[k])
        _same_epochs(a.epochs, b.epochs)
        assert a.modes or not b.modes
        assert len(a.modes) == len(b.modes)
        for m, n in zip(a.modes, b.modes):
            assert onp.allclose(m["x"], n["x"])
            assert m["flags"] == n["flags"]
            assert m["loglike"] == pytest.approx(n["loglike"], nan_ok=True)
    assert result.best.epochs and result.best.modes
    assert loaded.samples.keys() == result.samples.keys()
    for n, draws in result.samples.items():
        for name, values in draws.items():
            assert onp.allclose(loaded.samples[n][name], values)
    assert [r["flags"] for r in loaded.table()] == [
        r["flags"] for r in result.table()
    ]


def test_non_finite_numbers_survive_a_round_trip(tmp_path, monkeypatch):
    # ``clean_json`` writes NaN and infinity as null: loading restores -inf
    # for evidences and NaN for everything else, and keeps them distinct.
    data, _ = _synthetic(12.38)
    seen = []

    def fit_band(problem, settings, band, orbits, times):
        if not seen:  # one band really fails inside fit_orbit_aliases
            seen.append(band.n)
            raise ValueError("a bad band")
        return _band(band.n, []), (None, None)

    monkeypatch.setattr(aliases, "_fit_band", fit_band)
    fitted = fit_orbit_aliases(data, (11.3, 12.9), t_ref=T_REF)
    failed = next(b for b in fitted.bands if "band-failed" in b.flags)
    nan = float("nan")
    ok = _band(3, [dict(scale=[nan], chi2_red=[1.0])], p=nan, flipped=True)
    ok.modes = [
        dict(
            x=[0.0, nan], loglike=nan, log_z=-onp.inf, log_z_is=-onp.inf,
            flags=[],
        )
    ]  # fmt: skip
    res = AliasResult(
        60000.0, onp.zeros(2), 1.0, [ok, failed], {5: {"period": onp.array([1.0, nan])}}
    )  # fmt: skip
    res.save(tmp_path / "r.json")
    back = AliasResult.load(tmp_path / "r.json")
    assert back.bands[1].log_z == -onp.inf
    assert back.bands[1].error == failed.error
    assert onp.isnan(back.bands[1].chi2_red) and onp.isnan(back.bands[0].p)
    assert back.bands[1].p == failed.p
    assert onp.isnan(back.bands[0].epochs[0]["scale"][0])
    mode = back.bands[0].modes[0]
    assert mode["log_z"] == -onp.inf and onp.isnan(mode["loglike"])
    assert mode["x"][0] == 0.0 and onp.isnan(mode["x"][1])
    assert back.bands[0].flipped is True  # the flip survives
    assert onp.isnan(back.samples[5]["period"][1])
    assert back.samples[5]["period"].dtype == float


def test_scene_reproduces_the_model_of_the_fit():
    # No float64 wrapper: the accessors run in float64 themselves.
    data, result = _tiny_fit()
    band = result.best
    fit_value = _objective(data, _best_x(band))
    assert _score(data, band.chi2_per_night(data)) == pytest.approx(
        fit_value, rel=1e-4, abs=1e-3
    )
    # It is the model the fit builds: a UD primary (not a point source).
    assert isinstance(band.scene(), aliases._UDBinary)
    assert float(band.scene().diam) == DIAM
    assert result.scene().orbit.period == pytest.approx(band.best["period"])


def test_params_reprofile_the_amplitude_and_match_the_fit_objective():
    # At a perturbed orbit the V² amplitude is profiled again there, so the
    # χ² equals the fit's likelihood at that orbit, not best's amplitude.
    data, result = _tiny_fit()
    x = _best_x(result.best).copy()
    x[6] += 0.1  # a 10% larger semimajor axis
    params = {k: float(v[0]) for k, v in aliases._physical(x).items()}
    nights = result.best.chi2_per_night(data, params=params)
    assert _score(data, nights) == pytest.approx(
        _objective(data, x), rel=1e-4, abs=1e-3
    )
    best = result.best.chi2_per_night(data)
    assert any(
        a["vis_amp"] != pytest.approx(b["vis_amp"], rel=1e-3)
        for a, b in zip(nights, best)
    )  # the amplitude moved with the orbit
    assert sum(n["chi2_fit"] for n in nights) > sum(
        n["chi2_fit"] for n in best
    )


def test_scene_primary_follows_the_fit_settings():
    _, result = _tiny_fit()
    band = copy.copy(result.best)
    band.primary_diam = None
    assert type(band.scene()) is OrbitalBinary  # a point primary
    band.primary_diam = "fit"
    with pytest.raises(ValueError, match="diam"):
        band.scene()
    assert float(band.scene(dict(band.best, diam=2.0)).diam) == 2.0
    band.best = {}
    with pytest.raises(ValueError, match="no best orbit"):
        band.scene()


def test_per_night_chi2_equals_a_direct_computation():
    data, result = _tiny_fit()
    band = result.best
    nights = band.chi2_per_night(data)
    assert len(nights) == len(data)
    with jax.enable_x64(True):
        model = band.scene()
        for d, e, night in zip(data, band.epochs, nights):
            d = cast_tree(d, "float64")
            # Direct: the visibilities, scaled by this epoch's fitted
            # amplitude, and the closure phases, from virgil's whitened
            # residuals; the phase block includes the penalty rows.
            r = onp.asarray(aliases.whitened_residuals(model, d))
            n_vis = int(onp.asarray(d.vis).size)
            obs, err = onp.ravel(d.vis), onp.ravel(d.d_vis)
            mod = obs + r[:n_vis] * err
            vis = ((e["vis_amp"] * mod - obs) / err) ** 2
            phi = r[n_vis:] ** 2
            blocks = night["blocks"]
            assert blocks["vis"]["chi2_fit"] == pytest.approx(
                vis.sum(), rel=1e-5
            )
            assert blocks["phi"]["chi2_fit"] == pytest.approx(
                phi.sum(), rel=1e-5
            )
            # the data χ² leaves the penalty rows out and counts ν rows
            nu = blocks["phi"]["n"]
            assert blocks["phi"]["chi2"] == pytest.approx(
                phi[:nu].sum(), rel=1e-5
            )
            assert blocks["phi"]["chi2"] <= blocks["phi"]["chi2_fit"]
            assert night["n"] == d.n_independent
            assert night["chi2_per_n"] == pytest.approx(
                night["chi2"] / night["n"]
            )
            # and what the fit recorded for this epoch (raw χ²/ν per block)
            for name, red in zip(blocks, e["chi2_red"]):
                got = blocks[name]["chi2_per_n_fit"]
                assert got == pytest.approx(red, rel=1e-3)
    assert nights == result.chi2_per_night(data)


def test_residuals_are_whitened_blocks_in_vector_order():
    data, result = _tiny_fit()
    # Epoch 0 lost half its V² contrast, so its fitted amplitude is ~0.5.
    data = [data[0].set("vis", data[0].vis * 0.5), *data[1:]]
    blocks = result.residuals(data)
    assert len(blocks) == len(data) and list(blocks[0]) == ["vis", "phi"]
    amps = [n["vis_amp"] for n in result.chi2_per_night(data)]
    assert amps[0] == pytest.approx(0.5, rel=0.1) and amps[1] != amps[0]
    flat = onp.concatenate(list(blocks[0].values()))
    with jax.enable_x64(True):
        d = cast_tree(data[0], "float64")
        raw = onp.asarray(aliases.whitened_residuals(result.scene(), d))
        obs, err = onp.ravel(d.vis), onp.ravel(d.d_vis)
    assert flat.shape == raw.shape
    n_vis = blocks[0]["vis"].size
    assert onp.array_equal(flat[n_vis:], raw[n_vis:])  # closure phases as is
    # The visibility rows: (amp * model - obs) / err, not the raw rows.
    model = obs + raw[:n_vis] * err
    assert onp.allclose(
        blocks[0]["vis"], (amps[0] * model - obs) / err, rtol=1e-6, atol=1e-6
    )
    assert not onp.allclose(blocks[0]["vis"], raw[:n_vis], atol=1e-3)
    with pytest.raises(ValueError, match="epochs"):
        result.residuals(data[:2])


def test_a_diameter_in_params_needs_a_fitted_primary():
    _, result = _tiny_fit()  # a fixed-disk primary
    with pytest.raises(ValueError, match="not 'fit'"):
        result.scene(dict(result.best.best, diam=2.0))
    band = copy.copy(result.best)
    band.primary_diam = None
    with pytest.raises(ValueError, match="not 'fit'"):
        band.scene(dict(band.best, diam=2.0))
    _, fitted = _tiny_fit("fit")
    assert float(fitted.scene(dict(fitted.best.best, diam=2.0)).diam) == 2.0


def test_a_fitted_diameter_is_recognised_in_an_old_file(tmp_path):
    _, result = _tiny_fit("fit")
    result.save(tmp_path / "new.json")
    old = json.loads((tmp_path / "new.json").read_text())
    del old["primary_diam"], old["vis_amp"]
    for band in old["band_details"]:
        del band["primary_diam"], band["vis_amp"]
    (tmp_path / "old.json").write_text(json.dumps(old))
    loaded = AliasResult.load(tmp_path / "old.json")
    assert loaded.primary_diam == "fit"
    assert float(loaded.scene().diam) == pytest.approx(
        loaded.best.best["diam"]
    )
    assert loaded.bands[0].vis_amp == aliases.UNRECORDED  # not recoverable


def test_a_loaded_result_scores_like_the_original(tmp_path):
    data, result = _tiny_fit()
    result.save(tmp_path / "fit.json")
    loaded = AliasResult.load(tmp_path / "fit.json")
    a, b = result.chi2_per_night(data), loaded.chi2_per_night(data)
    assert [x["chi2"] for x in b] == pytest.approx(
        [x["chi2"] for x in a], rel=1e-5
    )


def test_a_saved_result_records_its_format_and_refuses_a_newer_one(tmp_path):
    _, result = _tiny_fit()
    path = tmp_path / "fit.json"
    result.save(path)
    saved = json.loads(path.read_text())
    assert saved["format"] == aliases._RESULT_FORMAT == 1
    again = AliasResult.load(path)  # the round trip
    assert again.best.n == result.best.n
    del saved["format"]  # a legacy file has no field and still loads
    path.write_text(json.dumps(saved))
    assert AliasResult.load(path).best.n == result.best.n
    saved["format"] = aliases._RESULT_FORMAT + 1
    path.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="format 2.*upgrade virgil"):
        AliasResult.load(path)


def test_a_file_that_lacks_the_settings_is_not_read_as_a_point_primary(
    tmp_path,
):
    # Files written before ``primary_diam`` and ``vis_amp`` were saved.
    data, result = _tiny_fit()
    result.save(tmp_path / "new.json")
    old = json.loads((tmp_path / "new.json").read_text())
    for key in ("primary_diam", "vis_amp"):
        del old[key]
    for band in old["band_details"]:
        for key in ("primary_diam", "vis_amp", "t_ref"):
            del band[key]
    (tmp_path / "old.json").write_text(json.dumps(old))
    loaded = AliasResult.load(tmp_path / "old.json")
    assert loaded.primary_diam == aliases.UNRECORDED
    assert loaded.bands[0].vis_amp == aliases.UNRECORDED
    for call in (
        lambda: loaded.scene(),
        lambda: loaded.residuals(data),
        lambda: loaded.chi2_per_night(data),
    ):
        with pytest.raises(ValueError, match="does not record"):
            call()
    # A diameter alone is not enough: the V² amplitude is unrecorded too.
    half = AliasResult.load(tmp_path / "old.json", primary_diam=DIAM)
    assert float(half.scene().diam) == DIAM
    with pytest.raises(ValueError, match="vis_amp"):
        half.chi2_per_night(data)
    full = AliasResult.load(
        tmp_path / "old.json", primary_diam=DIAM, vis_amp=list(VIS_AMP)
    )
    got = full.chi2_per_night(data)
    want = result.chi2_per_night(data)
    assert [g["chi2"] for g in got] == pytest.approx([w["chi2"] for w in want])
    # None is a recorded answer (point primary, no amplitude), not unrecorded.
    point = AliasResult.load(
        tmp_path / "old.json", primary_diam=None, vis_amp=None
    )
    assert type(point.scene()) is OrbitalBinary


def _check_fitted_diameter(data, result, tmp_path):
    band = result.best
    assert result.primary_diam == "fit" and "diam" in band.best
    assert float(band.scene().diam) == pytest.approx(band.best["diam"])
    x = _best_x(band)
    assert _score(data, band.chi2_per_night(data), "fit") == pytest.approx(
        _objective(data, x, "fit"), rel=1e-4, abs=1e-3
    )
    result.save(tmp_path / "fit.json")
    loaded = AliasResult.load(tmp_path / "fit.json")
    assert loaded.primary_diam == "fit"
    assert float(loaded.scene().diam) == pytest.approx(band.best["diam"])
    for a, b in zip(loaded.chi2_per_night(data), band.chi2_per_night(data)):
        assert a["chi2_fit"] == pytest.approx(b["chi2_fit"], rel=1e-5)


# --- night flags: a flagged night seeds nothing, but its data still count -----


SEED_ALIAS_NIGHT = 79  # any seed does: the alias is built, not found
ALIAS_NIGHT = 3  # the night whose best peak is the alias
MIRROR_LEAN = 0.1  # how far its closure phases lean toward the mirror image
SOUND_NIGHTS = [k for k in range(6) if k != ALIAS_NIGHT]
FLAGGED_ALIAS = [k == ALIAS_NIGHT for k in range(6)]


def _numpy_noise(night, rng):
    """Gaussian noise for ``night`` drawn with numpy: iid V² noise, and
    closure phases correlated as ``ClosureNoise.sample`` does (equal noise
    on every baseline, mapped through the triangle incidence), so the
    fixture has the noise model the likelihood and ``epoch_positions``
    assume."""
    d_vis = onp.asarray(night.d_vis)
    d_phi = onp.asarray(night.d_phi)
    cp = night.cp_noise
    e = rng.standard_normal(cp.incidence.shape[::2])
    phi = onp.zeros(d_phi.size)
    onp.add.at(
        phi,
        cp.groups,
        onp.where(
            cp.mask,
            onp.einsum("gmb,gb->gm", cp.incidence, e) * d_phi[cp.groups],
            0.0,
        ),
    )
    return d_vis * rng.standard_normal(d_vis.shape), phi


def _draw_alias_night_data(seed=SEED_ALIAS_NIGHT):
    """Six nights of the true orbit, built so that night ``ALIAS_NIGHT``'s
    best static-binary peak is a single-night alias, 19 mas from the truth
    (the Gl 229 2024-02-27 situation), and every other night is right and
    unflagged.

    The alias is built into the data, not left to the noise. The night's V²
    are the truth's, which a binary shares with its point reflection through
    the primary; its closure phases, which alone tell the two apart, are the
    truth's reflected and scaled by ``MIRROR_LEAN`` and carry no noise, so
    the reflection beats the truth by a fraction of a nat. The surface then
    has two peaks that tie within the ``alias-peaks`` limit and the mirror
    image is the higher, whatever the optimizer does.

    The other nights have three frames at one epoch (the frames of
    ``vlti_night`` are an hour apart, and near periastron the companion
    moves a few tenths of a mas in that time, more than a static binary
    fits) and their own numpy noise, drawn per night so the nights are
    independent. The noise is numpy, not ``jax.random``: a JAX key gives
    different bits in float32 and float64, and
    ``test_the_alias_night_fixture_is_the_same_in_float32_and_float64``
    checks that the two agree."""
    orbit = KeplerOrbit(**TRUTH, t_ref=T_REF)
    scene = OrbitalBinary(orbit, FLUX)
    data = []
    for k, m in enumerate(EPOCHS):
        rng = onp.random.default_rng((seed, k))
        base = _night(m, rotation=7.0 * k)
        base = eqx.tree_at(lambda n: n.dt, base, np.zeros_like(base.dt))
        night = simulate(scene, base)
        d_vis, d_phi = _numpy_noise(night, rng)
        phi = onp.asarray(night.phi)
        phi = -MIRROR_LEAN * phi if k == ALIAS_NIGHT else phi + d_phi
        data.append(
            night.set(["vis", "phi"], [onp.asarray(night.vis) + d_vis, phi])
        )
    dra, ddec, _ = orbit.relative(EPOCHS)
    return data, onp.asarray(dra), onp.asarray(ddec)


_alias_night_data = functools.cache(_draw_alias_night_data)


def _locate_alias_nights(data):
    axis = onp.arange(-14.0, 14.01, 0.5)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return epoch_positions(
            Epochs({f"n{k}": d for k, d in enumerate(data)}),
            {"dra": axis, "ddec": axis, "flux": [0.1, 0.3, 0.6]},
        )


@functools.cache
def _alias_night_positions():
    data, *_ = _alias_night_data()
    return _locate_alias_nights(data)


@pytest.mark.slow
def test_the_alias_night_fixture_is_the_same_in_float32_and_float64():
    # jaxoplanet turns x64 on, so orbit users run float64 while the default
    # JAX is float32: the night-flag fixtures must mean the same in both.
    # The noise is numpy float64 in both modes, so this guards the fixture
    # against going back to jax.random (whose bits differ with precision).
    # epoch_positions works in float64 internally; what still varies with
    # the precision is the simulated prediction, so the raw chi2 is compared.
    found = {}
    for x64 in (False, True):
        with jax.enable_x64(x64):
            data, dra, ddec = _draw_alias_night_data()
            found[x64] = _locate_alias_nights(data)
    f32, f64 = found[False], found[True]
    assert f32.flagged.tolist() == f64.flagged.tolist() == FLAGGED_ALIAS
    assert f32.flag_codes() == f64.flag_codes()
    # Every night, the alias included, lands on the same grid peak.
    assert onp.allclose(f32.dra, f64.dra, atol=0.1)
    assert onp.allclose(f32.ddec, f64.ddec, atol=0.1)
    assert onp.allclose(
        [c["all"] for c in f32.chi2_raw],
        [c["all"] for c in f64.chi2_raw],
        rtol=1e-2,
    )
    miss = onp.hypot(f64.dra - dra, f64.ddec - ddec)
    assert miss[ALIAS_NIGHT] > 3.0 and onp.all(miss[SOUND_NIGHTS] < 0.5)
    # The sound nights are decisive, not just lucky: their gaps are far
    # above the low-gap limit, and the alias night's is below the
    # alias-peaks limit.
    assert onp.all(f64.gap_marginal[SOUND_NIGHTS] > 10.0)
    assert f64.gap_marginal[ALIAS_NIGHT] < 2.0


@pytest.mark.slow
def test_a_single_night_alias_is_flagged_and_the_alias_fit_finds_the_band(
    monkeypatch,
):
    data, dra, ddec = _alias_night_data()
    found = _alias_night_positions()
    miss = onp.hypot(found.dra - dra, found.ddec - ddec)
    # The alias night's best peak is 3 mas or more from the truth,
    # and it is flagged; every other night is right and unflagged.
    assert miss[ALIAS_NIGHT] > 3.0 and onp.all(miss[SOUND_NIGHTS] < 0.5)
    assert found.flagged.tolist() == FLAGGED_ALIAS
    assert "alias-peaks" in found.flag_codes()[f"n{ALIAS_NIGHT}"]
    seeded = []
    real = aliases._short_arc_profile

    def spy(data, seed, positions, *args, **kw):
        seeded.append(positions)
        return real(data, seed, positions, *args, **kw)

    monkeypatch.setattr(aliases, "_short_arc_profile", spy)
    result = fit_orbit_aliases(
        data,
        (11.3, 12.9),
        positions=found,
        t_ref=T_REF,
        n_candidates=30,
        n_refine=3,
        n_is=400,
        n_samples=50,
    )
    n_true = round(float(onp.ptp(result.times)) / TRUTH["period"])
    assert result.best.n == n_true
    assert abs(result.best.best["period"] - TRUTH["period"]) < 0.05
    # The flagged night seeded nothing ...
    assert len(seeded) == 1 and seeded[0].dra.shape == (5,)
    assert [u["name"] for u in result.unseeded] == [f"n{ALIAS_NIGHT}"]
    assert "alias-peaks" in [f["code"] for f in result.unseeded[0]["flags"]]
    # ... but its data are in the likelihood: the best orbit predicts the
    # true position on that night (within 0.5 mas), not the alias peak
    # (more than 3 mas away).
    assert len(result.best.epochs) == 6
    best = result.best.best
    orbit = KeplerOrbit(**{k: best[k] for k in TRUTH}, t_ref=result.t_ref)
    pred_dra, pred_ddec, _ = orbit.relative(EPOCHS)
    a = ALIAS_NIGHT
    assert onp.hypot(pred_dra[a] - dra[a], pred_ddec[a] - ddec[a]) < 0.5
    assert (
        onp.hypot(pred_dra[a] - found.dra[a], pred_ddec[a] - found.ddec[a])
        > 3.0
    )
    # Negative control: seeding with the flagged night's alias peak
    # misleads the position-level profile of the true band.
    bands = [
        b for b in alias_bands(result.times, (11.3, 12.9)) if b.n == n_true
    ]
    with_alias = position_profile(
        found.positions(t_ref=T_REF, skip_flagged=False), bands
    )
    without = position_profile(found.positions(t_ref=T_REF), bands)
    assert with_alias[0]["chi2"] > 5 * without[0]["chi2"]


@pytest.mark.slow
def test_skip_flagged_can_be_switched_off_and_unseeded_survives_save_load(
    monkeypatch, tmp_path
):
    data, *_ = _alias_night_data()
    found = _alias_night_positions()
    seeded = []
    real, real_arcs = aliases.position_profile, aliases._short_arc_profile
    # skip_flagged=False seeds every band from the positions (no short
    # arcs: a flagged night's sign cannot be trusted); True, the short arcs.
    monkeypatch.setattr(
        aliases,
        "position_profile",
        lambda positions, *a, **k: (
            seeded.append(positions) or real(positions, *a, **k)
        ),
    )
    monkeypatch.setattr(
        aliases,
        "_short_arc_profile",
        lambda data, seed, positions, *a, **k: (
            seeded.append(positions)
            or real_arcs(data, seed, positions, *a, **k)
        ),
    )
    kw = dict(t_ref=T_REF, n_candidates=10, n_refine=2, n_is=100, n_samples=10)
    everyone = fit_orbit_aliases(
        data, (12.0, 12.4), positions=found, skip_flagged=False, **kw
    )
    assert seeded[0].dra.shape == (6,) and everyone.unseeded == ()
    skipped = fit_orbit_aliases(data, (12.0, 12.4), positions=found, **kw)
    assert seeded[1].dra.shape == (5,)
    skipped.save(tmp_path / "r.json")
    loaded = AliasResult.load(tmp_path / "r.json")
    assert loaded.unseeded == skipped.unseeded
    assert loaded.short_arc == skipped.short_arc and skipped.short_arc["kept"]
    assert loaded.unseeded[0]["flags"][0]["reason"]
    # A file written before the record was kept has none.
    payload = json.loads((tmp_path / "r.json").read_text())
    del payload["unseeded"]
    (tmp_path / "old.json").write_text(json.dumps(payload))
    assert AliasResult.load(tmp_path / "old.json").unseeded == ()


class _Stop(Exception):
    """Raised by the spy that ends a fit before the optimizer starts."""


def _stub_positions(flagged=(), primary_diam=None):
    """``EpochPositions`` of the synthetic data without running
    ``epoch_positions``: only the names, times and flags matter to the
    seeding logic these tests exercise."""
    from virgil.epochs import EpochPositions, NightFlag

    data, _ = _synthetic(12.1)
    n = len(data)
    return EpochPositions(
        names=tuple(f"n{k}" for k in range(n)),
        mjd=onp.array([onp.mean(onp.asarray(d.mjd, float)) for d in data]),
        dra=onp.zeros(n),
        ddec=onp.zeros(n),
        cov=onp.tile(onp.eye(2), (n, 1, 1)),
        flux=onp.full(n, 0.3),
        gap_marginal=onp.full(n, 20.0),
        chi2_raw=({},) * n,
        scale=({},) * n,
        flags=tuple(
            (NightFlag("manual", "flagged by the test"),)
            if f"n{k}" in flagged
            else ()
            for k in range(n)
        ),
        primary_diam=primary_diam,
    )


def _seed_settings(monkeypatch, positions, **options):
    """Run ``fit_orbit_aliases`` up to its preflight and return the
    ``n_random`` it settled on and whether any position-level seeding
    (``position_profile``) ran, without running the optimizer."""
    data, _ = _synthetic(12.1)
    seen = {}

    def preflight(problem, settings, band):
        seen["n_random"] = settings.n_random
        raise _Stop

    monkeypatch.setattr(aliases, "_preflight", preflight)
    monkeypatch.setattr(
        aliases,
        "position_profile",
        lambda pos, bands, **kw: seen.setdefault("seeded", len(pos.dt))
        and [dict(orbits=[]) for _ in bands],
    )
    monkeypatch.setattr(
        aliases,
        "_short_arc_profile",
        lambda data, seed, pos, bands, **kw: (
            seen.setdefault("seeded", len(pos.dt))
            and [dict(orbits=[]) for _ in bands],
            None,
            None,
        ),
    )
    with pytest.raises(_Stop):
        fit_orbit_aliases(
            data, (12.0, 12.4), positions=positions, t_ref=T_REF, **options
        )
    return seen


@pytest.mark.parametrize(
    "flagged",
    [("n0", "n1", "n2", "n3", "n4"), ("n0", "n1", "n2", "n3", "n4", "n5")],
    ids=["one-unflagged-night", "every-night-flagged"],
)
def test_fewer_than_two_seeding_nights_fall_back_to_random_starts(
    flagged, monkeypatch
):
    # Fewer than two positions give an underdetermined Thiele-Innes fit
    # whose chi2 = 0 seeds mean nothing, and used to leave n_random = 0
    # ("No candidate orbits" when every night was flagged).
    for n_random in (None, 0):
        with pytest.warns(UserWarning, match="falling back to random"):
            seen = _seed_settings(
                monkeypatch, _stub_positions(flagged), n_random=n_random
            )
        assert "seeded" not in seen and seen["n_random"] >= 200
    # The same for PositionData with a single row.
    one = PositionData([60000.0], [1.0], [1.0], [0.04 * onp.eye(2)])
    with pytest.warns(UserWarning, match="falling back to random"):
        seen = _seed_settings(monkeypatch, one, n_random=0)
    assert seen["n_random"] >= 200


def test_seeds_from_three_or_fewer_nights_keep_random_candidates(
    monkeypatch,
):
    seen = _seed_settings(monkeypatch, _stub_positions(("n3", "n4", "n5")))
    assert seen["seeded"] == 3 and seen["n_random"] == 200
    seen = _seed_settings(monkeypatch, _stub_positions(("n4", "n5")))
    assert seen["seeded"] == 4 and seen["n_random"] == 0
    # An explicit n_random is respected when there are seeds.
    seen = _seed_settings(
        monkeypatch, _stub_positions(("n3", "n4", "n5")), n_random=7
    )
    assert seen["n_random"] == 7


def test_a_night_with_only_a_non_blocking_flag_still_seeds():
    from virgil.epochs import NightFlag

    stub = _stub_positions(("n5",))
    flags = list(stub.flags)
    flags[0] = (NightFlag("sep-below-resolution", "r", blocking=False),)
    mixed = dataclasses.replace(stub, flags=tuple(flags))
    seeds, skipped, fell_back = aliases._seeding_positions(
        mixed, T_REF, True, mixed.mjd
    )
    assert len(seeds.dt) == 5 and not fell_back  # n0 seeds, n5 does not
    assert [u["name"] for u in skipped] == ["n5"]


def test_epoch_positions_must_match_the_data_and_the_primary():
    found = _stub_positions(("n5",))
    times = found.mjd
    with pytest.raises(ValueError, match="not the epoch_positions"):
        aliases._seeding_positions(found, T_REF, True, times[:-1])
    shifted = dataclasses.replace(found, mjd=found.mjd + 5.0)
    with pytest.raises(ValueError, match="not the epoch_positions"):
        aliases._seeding_positions(shifted, T_REF, True, times)
    seeds, skipped, fell_back = aliases._seeding_positions(
        found, T_REF, True, times
    )
    assert len(seeds.dt) == 5 and not fell_back
    assert [u["name"] for u in skipped] == ["n5"]
    # Seeds fitted with another primary than the engine's warn.
    ud = _stub_positions(primary_diam=1.05)
    with pytest.warns(UserWarning, match="primary_diam=1.05"):
        aliases._check_seed_diameter(ud, None)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        aliases._check_seed_diameter(ud, 1.05)
        aliases._check_seed_diameter(found, None)
        # A fitted diameter is seeded by any UD estimate ...
        aliases._check_seed_diameter(ud, "fit")
    # ... and only point seeds warn, with advice that can be followed.
    with pytest.warns(UserWarning, match="Pass an estimated primary_diam"):
        aliases._check_seed_diameter(found, "fit")


def test_a_fitted_primary_diameter_works_end_to_end(tmp_path):
    _check_fitted_diameter(*_tiny_fit("fit"), tmp_path)


@pytest.mark.slow
def test_a_fitted_primary_diameter_after_a_real_fit(tmp_path):
    _check_fitted_diameter(*_real_fit("fit"), tmp_path)


@pytest.mark.slow
def test_a_real_fit_saves_and_loads_back(tmp_path):
    data, result = _real_fit(DIAM)
    result.save(tmp_path / "fit.json")
    loaded = AliasResult.load(tmp_path / "fit.json")
    assert loaded.best.n == result.best.n
    assert loaded.samples.keys() == result.samples.keys()
    for a, b in zip(loaded.chi2_per_night(data), result.chi2_per_night(data)):
        assert a["chi2_fit"] == pytest.approx(b["chi2_fit"], rel=1e-5)


# --- plan phase 1c: automatic engine defaults -----------------------------------------
# Each test passes no per-system argument (no p_range, no flux range, no
# twin or diameter list): the defaults must be enough.


def test_the_engine_defaults_need_no_per_system_arguments():
    # Problems 5, 8, 9, 10, 13 of the orbit-consolidation plan.
    import inspect

    sig = inspect.signature(fit_orbit_aliases).parameters
    assert sig["p_range"].default is None  # period_window (5)
    assert sig["sign_free"].default is True  # label swap (8)
    assert sig["flip_flux"].default is True  # flux flip (9)
    assert sig["primary_diam"].default == "auto"  # UD primary (10)
    assert sig["vis_amp"].default is True  # V² amplitude (13)


def test_period_window_limits_on_simple_inputs():
    # Upper end: 4 x the span. Lower end: a night of 0.1 d at a field of
    # 10 mas and a resolution of 2 mas moves 2π·10·0.1/P mas, at most
    # half the resolution (1 mas): P >= 2π·10·0.1/1.
    lo, hi = aliases.period_window(
        [0.0, 40.0, 100.0], 10.0, 0.1, resolution=2.0
    )
    assert hi == pytest.approx(400.0)
    assert lo == pytest.approx(2 * onp.pi * 10.0 * 0.1 / 1.0)
    # The window follows the field, the night and the resolution...
    lo2, _ = aliases.period_window([0.0, 100.0], 20.0, 0.1, resolution=2.0)
    assert lo2 == pytest.approx(2 * lo)
    # ...but is never shorter than the night itself,
    lo3, _ = aliases.period_window([0.0, 100.0], 1e-3, 0.5, resolution=2.0)
    assert lo3 == pytest.approx(0.5)
    # and a span_factor changes only the top.
    _, hi4 = aliases.period_window(
        [0.0, 100.0], 10.0, 0.1, resolution=2.0, span_factor=2.0
    )
    assert hi4 == pytest.approx(200.0)
    # A physical floor stands in for a one-instant night.
    lo5, _ = aliases.period_window([0.0, 100.0], 10.0, 0.0, resolution=2.0)
    assert lo5 == pytest.approx(0.2)
    with pytest.raises(ValueError, match="positive time"):
        aliases.period_window([5.0, 5.0], 10.0, 0.1, resolution=2.0)
    with pytest.raises(ValueError, match="window is empty"):
        aliases.period_window([0.0, 1.0], 10.0, 5.0, resolution=2.0)


def _spy_settings(monkeypatch, **options):
    """The ``_Settings`` and seeds ``fit_orbit_aliases`` settles on, read at
    its preflight (the optimizer never runs)."""
    seen = {}

    def preflight(problem, settings, band):
        seen["settings"], seen["band"] = settings, band
        raise _Stop

    monkeypatch.setattr(aliases, "_preflight", preflight)
    monkeypatch.setattr(
        aliases,
        "position_profile",
        lambda pos, bands, **kw: [dict(orbits=[]) for _ in bands],
    )
    monkeypatch.setattr(
        aliases,
        "_short_arc_profile",
        lambda data, seed, pos, bands, **kw: (
            [dict(orbits=[]) for _ in bands],
            None,
            None,
        ),
    )
    data, _ = _synthetic(12.1)
    with pytest.raises(_Stop):
        fit_orbit_aliases(data, t_ref=T_REF, **{"n_random": 5, **options})
    return seen


def test_without_p_range_the_window_comes_from_the_data(monkeypatch):
    # Period window (problem 5): never a published period. Without
    # positions the field is the coarsest fringe spacing (not a silent
    # 30 mas), and the night is measured from the frame times.
    with pytest.warns(UserWarning, match="needs EpochPositions"):
        seen = _spy_settings(monkeypatch)
    data, _ = _synthetic(12.1)
    times = [float(onp.mean(d.mjd)) for d in data]
    finest, coarsest = aliases.fringe_scales(data)
    expected = aliases.period_window(
        times,
        min(coarsest, 30.0),
        2.0 / 24.0,  # three frames an hour apart
        resolution=finest,
    )
    assert seen["settings"].p_range == pytest.approx(expected)
    assert expected[0] < 12.1  # the test orbit is inside the window
    assert expected[1] == pytest.approx(4 * onp.ptp(times))
    # Defaults reach the settings (8, 13), and a point primary without
    # positions (10).
    s = seen["settings"]
    assert s.sign_free and s.flip_flux and s.primary_diam is None
    assert s.vis_amp == aliases._VIS_AMP_DEFAULT
    assert s.resolution_mas == pytest.approx(finest)


def test_the_night_is_measured_from_frame_gaps_not_the_dataset_span():
    # A dataset joining two nights (2 d apart) is two 2-hour nights.
    data, _ = _synthetic(12.1)
    mjd = onp.asarray(data[0].mjd, float)
    joined = types.SimpleNamespace(mjd=onp.concatenate([mjd, mjd + 2.0]))
    assert onp.ptp(joined.mjd) > 2.0
    assert aliases._night_length([joined]) == pytest.approx(
        aliases._night_length([data[0]])
    )
    assert aliases._night_length([data[0]]) == pytest.approx(2.0 / 24.0)


def _epoch_positions_at(separations, flagged=()):
    from virgil.epochs import NightFlag

    stub = _stub_positions()
    n = len(separations)
    return dataclasses.replace(
        stub,
        names=stub.names[:n],
        mjd=stub.mjd[:n],
        dra=onp.asarray(separations, float),
        ddec=onp.zeros(n),
        cov=stub.cov[:n],
        flux=stub.flux[:n],
        gap_marginal=stub.gap_marginal[:n],
        chi2_raw=stub.chi2_raw[:n],
        scale=stub.scale[:n],
        flags=tuple(
            (NightFlag("manual", "test"),) if k in flagged else ()
            for k in range(n)
        ),
    )


def test_the_window_field_is_the_median_unflagged_separation():
    # One wrong alias peak at 25 mas must not raise the limit (the max
    # would), flagged nights are dropped, every night flagged falls back to
    # all of them, and the result is clipped to a_range[1].
    field = aliases._field_mas
    positions = _epoch_positions_at(
        [4.0, 5.0, 6.0, 25.0, 40.0], flagged=(3, 4)
    )
    assert field(positions, (1.0, 30.0), 7.0) == pytest.approx(5.0)
    allflag = _epoch_positions_at([4.0, 6.0, 8.0], flagged=(0, 1, 2))
    assert field(allflag, (1.0, 30.0), 7.0) == pytest.approx(6.0)
    assert field(_epoch_positions_at([40.0, 50.0]), (1.0, 30.0), 7.0) == 30.0
    assert field(_epoch_positions_at([0.0, 0.0]), (1.0, 30.0), 7.0) == 1e-3
    # Without positions: the fallback, up to a_range[1].
    assert field(None, (1.0, 30.0), 7.0) == 7.0
    assert field(None, (1.0, 5.0), 7.0) == 5.0


def test_a_default_window_with_too_many_bands_is_capped_and_says_so(
    monkeypatch,
):
    data, _ = _synthetic(12.1)
    _stub_band_fit(monkeypatch, 12.1)
    with pytest.warns(UserWarning, match="n_bands_max=50"):
        capped = fit_orbit_aliases(
            data, t_ref=T_REF, n_bands_max=50, n_samples=1,
            primary_diam=None,
        )  # fmt: skip
    assert len(capped.bands) <= 50
    assert capped.window_capped is not None
    assert capped.p_range[0] > capped.window_capped
    free = fit_orbit_aliases(
        data, t_ref=T_REF, n_bands_max=None, n_samples=1, primary_diam=None
    )
    assert free.window_capped is None
    assert free.p_range[0] == pytest.approx(capped.window_capped)
    # An explicit window is never capped.
    explicit = fit_orbit_aliases(
        data, (1.0, 400.0), t_ref=T_REF, n_bands_max=5, n_samples=1,
        primary_diam=None,
    )  # fmt: skip
    assert explicit.window_capped is None


def test_default_vis_amp_is_off_for_data_without_squared_visibilities():
    data, _ = _synthetic(12.1)
    assert aliases._default_vis_amp(data) == aliases._VIS_AMP_DEFAULT
    closure_only = [
        types.SimpleNamespace(vis_mode="amp", observable_kind="split")
    ] * 2
    assert aliases._default_vis_amp(closure_only) is None


def _resolved(diam, positions):
    from virgil.epochs import PrimaryDiameter

    return PrimaryDiameter(
        diam=diam,
        sigma=0.05,
        delta_nats=60.0 if diam else 0.1,
        resolved=bool(diam),
        grid=onp.array([0.0, diam or 0.5]),
        profile=onp.array([0.0, 60.0]),
        positions=positions,
    )


@pytest.mark.filterwarnings("error:The positions were fitted")
def test_auto_primary_reseeds_with_the_diameter_it_fits(monkeypatch):
    # UD primary (problem 10): when the first pass resolves a 1.05 mas
    # primary, the engine fits a diameter starting there with a lower bound
    # of effectively 0, and the seeds carry the same diameter (no
    # seed/engine mismatch warning).
    reseeded = _stub_positions(primary_diam=1.05)
    monkeypatch.setattr(
        aliases,
        "estimate_primary_diameter",
        lambda data, positions, **kw: _resolved(1.05, reseeded),
    )
    seen = _spy_settings(monkeypatch, positions=_stub_positions())
    s = seen["settings"]
    assert s.primary_diam == "fit" and s.diam_seed == pytest.approx(1.05)
    assert s.diam_range[0] < 0.01 and s.fit_diam
    # An unresolved primary is a point, whatever the seeds were fitted with.
    monkeypatch.setattr(
        aliases,
        "estimate_primary_diameter",
        lambda data, positions, **kw: _resolved(0.0, _stub_positions()),
    )
    seen = _spy_settings(monkeypatch, positions=_stub_positions())
    assert seen["settings"].primary_diam is None
    # An explicit diameter is never overridden.
    seen = _spy_settings(
        monkeypatch,
        positions=_stub_positions(primary_diam=0.7),
        primary_diam=0.7,
    )
    assert seen["settings"].primary_diam == 0.7


def _stub_band_fit(monkeypatch, period, sigma_log_p=None):
    """Replace the band fits by one mode at ``period`` (days), so that the
    result's own bookkeeping can be tested without an optimizer."""
    modes = (
        []
        if sigma_log_p is None
        else [dict(x=[0.0] * 8, log_z=0.0, sigma=[sigma_log_p] + [1.0] * 7)]
    )

    def fake(problem, settings, band, orbits, times):
        result = AliasBandResult(
            n=band.n, p_lo=band.p_lo, p_hi=band.p_hi, modes=modes,
            log_z=-float(band.n), log_z_is=-float(band.n), ess=100.0,
            flagged=False, best=dict(period=period), chi2_red=1.0,
        )  # fmt: skip
        return result, (None, None)

    monkeypatch.setattr(aliases, "_preflight", lambda *a: None)
    monkeypatch.setattr(aliases, "_fit_band", fake)


def test_a_best_period_at_the_top_of_the_window_is_period_bound(monkeypatch):
    # Short arc (problem 5): the window ends at 4 x the span, and a best
    # period there means P is unconstrained.
    data, _ = _synthetic(12.1)
    times = [float(onp.mean(d.mjd)) for d in data]
    _, top = aliases.period_window(times, 30.0, 0.1, resolution=2.0)
    window = (20.0, top)
    kw = dict(t_ref=T_REF, n_samples=1, primary_diam=None)
    _stub_band_fit(monkeypatch, 0.98 * top)
    bound = fit_orbit_aliases(data, window, **kw)
    assert bound.period_bound and bound.p_range == pytest.approx(window)
    assert not bound.period_at_lower_bound
    _stub_band_fit(monkeypatch, 100.0)
    free = fit_orbit_aliases(data, window, **kw)
    assert not free.period_bound and not free.period_at_lower_bound


def test_a_flat_period_likelihood_is_period_bound_wherever_the_best_is(
    monkeypatch,
):
    # A short arc: the optimizer stops mid-window, but sigma(log P) is a
    # large part of the window, so P is unconstrained all the same.
    data, _ = _synthetic(12.1)
    window = (20.0, 200.0)
    kw = dict(t_ref=T_REF, n_samples=1, primary_diam=None)
    _stub_band_fit(monkeypatch, 60.0, sigma_log_p=1.0)
    assert fit_orbit_aliases(data, window, **kw).period_bound
    _stub_band_fit(monkeypatch, 60.0, sigma_log_p=0.01)
    assert not fit_orbit_aliases(data, window, **kw).period_bound


def test_a_best_period_just_above_the_bottom_is_at_the_lower_bound(
    monkeypatch, tmp_path
):
    # The window's lower end is a sampling limit, so a truth just above it
    # (or cut by it) is reported, and survives save and load.
    data, _ = _synthetic(12.1)
    window = (20.0, 400.0)
    kw = dict(t_ref=T_REF, n_samples=1, primary_diam=None)
    _stub_band_fit(monkeypatch, 21.0)
    low = fit_orbit_aliases(data, window, **kw)
    assert low.period_at_lower_bound and not low.period_bound
    low.save(tmp_path / "r.npz")
    again = aliases.AliasResult.load(tmp_path / "r.npz")
    assert again.period_at_lower_bound is True
    _stub_band_fit(monkeypatch, 90.0)
    assert not fit_orbit_aliases(data, window, **kw).period_at_lower_bound


def _reflected(x):
    """``x`` seen from the other star: the reflected track (ω + 180°)."""
    y = onp.array(x, float)
    y[4] += 180.0
    return aliases._fold(y)


@in_float64
def test_a_twin_is_labels_free_and_a_contrast_pair_is_not():
    # Label swap (problem 8): for near-equal fluxes the stars' labels are
    # not determined, so the reflected mode has the same evidence.
    orbit = KeplerOrbit(**TRUTH, t_ref=T_REF)
    verdict = {}
    for name, flux in (("twin", 1.0), ("contrast", FLUX)):
        key = jax.random.PRNGKey(4)
        data = [
            simulate(
                OrbitalBinary(orbit, flux),
                _night(m, rotation=7.0 * k),
                key=jax.random.fold_in(key, k),
            )
            for k, m in enumerate(EPOCHS)
        ]
        settings, problem = _problem(data)
        x = _x_truth(flux=flux)
        x[7] = onp.log(flux)  # (the start vector keeps it off the bound)
        modes = [
            dict(x=v.tolist(), log_z=float(problem.value_and_grad(
                jax.numpy.asarray(v))[0]), flags=[])
            for v in (x, _reflected(x))
        ]  # fmt: skip
        result = AliasBandResult(
            n=20, p_lo=12.0, p_hi=12.2, modes=modes, log_z=0.0, log_z_is=0.0,
            ess=100.0, flagged=False,
        )  # fmt: skip
        verdict[name] = aliases._labels_free(result, settings, EPOCHS)
    assert verdict == {"twin": True, "contrast": False}


def test_labels_period_bound_and_the_auto_estimate_survive_save_and_load(
    tmp_path,
):
    data, result = _tiny_fit()
    result = copy.deepcopy(result)
    result.bands[0].labels_free = True
    result.period_bound = True
    result.p_range = (0.5, 990.0)
    result.primary_fit = dict(
        diam=1.05, sigma=0.03, delta_nats=60.0, resolved=True
    )
    assert result.labels_free
    path = tmp_path / "r.json"
    result.save(path)
    back = AliasResult.load(path)
    assert back.labels_free and back.period_bound
    assert back.p_range == (0.5, 990.0)
    assert back.primary_fit == result.primary_fit
    assert back.table()[0]["labels_free"]
    # A file from before these fields loads as "not free, not bound".
    old = json.loads(path.read_text())
    for k in ("p_range", "period_bound", "primary_fit"):
        del old[k]
    del old["band_details"][0]["labels_free"]
    path.write_text(json.dumps(old))
    back = AliasResult.load(path)
    assert not back.labels_free and not back.period_bound
    assert back.p_range is None and back.primary_fit is None


def test_sign_free_seeds_add_to_the_signed_ones_and_keep_the_true_orbit():
    # Label swap (problem 8): three nights of a contrast pair, whose signs
    # the closure phases fix. A sign-free search alone could crowd the true
    # orbit out with mixed-sign ones; here the signed orbits stay in the pool.
    orbit = KeplerOrbit(**TRUTH, t_ref=T_REF)
    times = T_REF + onp.array([0.0, 1.3, 2.6])
    positions = _arc_positions(orbit, times, FLUX, offset=(0.0, 0.0))
    positions = positions.positions(t_ref=T_REF)
    band = aliases.alias_bands(times, (11.5, 12.7))
    kw = dict(n_best=2, eccs=[0.1, 0.3])
    signed = aliases.position_profile(positions, band, sign_free=False, **kw)
    both = aliases.position_profile(positions, band, sign_free=True, **kw)
    assert len(both[0]["orbits"]) > len(signed[0]["orbits"])
    best = min(both[0]["orbits"], key=lambda row: row[1])
    assert best[1] <= min(row[1] for row in signed[0]["orbits"]) + 1e-9
    assert best[0].period == pytest.approx(TRUTH["period"], rel=0.05)


def test_labels_free_tolerance_is_a_fraction_of_the_resolution():
    # A compact orbit and a wide one are held to what the data can tell
    # apart: tracks that differ by 0.4 mas are the same at a 20 mas
    # resolution (tolerance 5 mas) but not at 1 mas (0.25 mas).
    orbit = KeplerOrbit(**TRUTH, t_ref=T_REF)
    settings, problem = _problem(_synthetic(12.1)[0])
    x = _x_truth(flux=1.0)
    near = _reflected(x)
    near[6] += 0.4 / float(onp.exp(x[6]))  # a 0.4 mas larger a, in log
    modes = [dict(x=v.tolist(), log_z=0.0, flags=[]) for v in (x, near)]
    result = AliasBandResult(
        n=20, p_lo=12.0, p_hi=12.2, modes=modes, log_z=0.0, log_z_is=0.0,
        ess=100.0, flagged=False,
    )  # fmt: skip
    verdict = {}
    for res in (20.0, 1.0):
        s = dataclasses.replace(settings, resolution_mas=res)
        verdict[res] = aliases._labels_free(result, s, EPOCHS)
    assert verdict == {20.0: True, 1.0: False}
    assert orbit.period == pytest.approx(TRUTH["period"])


def test_a_fixed_diameter_seeds_are_kept_with_a_warning_when_unresolved(
    monkeypatch,
):
    # UD primary (problem 10): seeds the user refined with a diameter the
    # data do not resolve are not silently replaced.
    monkeypatch.setattr(
        aliases,
        "estimate_primary_diameter",
        lambda data, positions, **kw: _resolved(0.0, _stub_positions()),
    )
    mine = _stub_positions(primary_diam=0.7)
    with pytest.warns(UserWarning, match="keeping your seeds"):
        seen = _spy_settings(monkeypatch, positions=mine)
    assert seen["settings"].primary_diam is None


def test_a_diameter_estimate_at_the_grid_edge_is_recorded():
    from virgil.epochs import PrimaryDiameter

    edge = PrimaryDiameter(
        diam=1.3, sigma=0.05, delta_nats=60.0, resolved=True,
        grid=onp.array([0.0, 1.3]), profile=onp.array([0.0, 60.0]),
        positions=_stub_positions(), at_grid_edge=True,
    )  # fmt: skip
    assert aliases._primary_fit_record(edge)["at_grid_edge"] is True


# The real fits: short arcs of 4 nights within 4 days, so the window holds
# a handful of bands, with the default arguments only.


def _short_arc(flux, period=12.1, diam=None):
    orbit = KeplerOrbit(**{**TRUTH, "period": period}, t_ref=T_REF)
    times = T_REF + onp.array([0.0, 1.3, 2.6, 4.0])
    scene = (
        OrbitalBinary(orbit, flux)
        if diam is None
        else aliases._UDBinary(orbit, flux, diam)
    )
    key = jax.random.PRNGKey(11)
    data = [
        simulate(
            scene, _night(m, rotation=7.0 * k), key=jax.random.fold_in(key, k)
        )
        for k, m in enumerate(times)
    ]
    return orbit, data, times


FAST_DEFAULTS = dict(n_candidates=10, n_refine=2, n_is=200, n_samples=20)


def _arc_positions(
    orbit, times, flux, signs=None, offset=(0.2, -0.2), gap=20.0
):
    """Per-night ``EpochPositions`` of a short arc without a grid search:
    the true companion positions, off by ``offset`` (mas) and, where
    ``signs`` is -1, reflected through the primary (a twin's positions are
    known only up to that sign), each ``gap`` nats above its rivals."""
    from virgil.epochs import EpochPositions

    n = len(times)
    sign = onp.ones(n) if signs is None else onp.asarray(signs, float)
    dra, ddec, _ = orbit.relative(times)
    return EpochPositions(
        names=tuple(f"n{k}" for k in range(n)),
        mjd=onp.asarray(times),
        dra=sign * onp.asarray(dra) + offset[0],
        ddec=sign * onp.asarray(ddec) + offset[1],
        cov=onp.tile(0.05 * onp.eye(2), (n, 1, 1)),
        flux=onp.full(n, min(flux, 0.9)),
        gap_marginal=onp.full(n, gap),
        chi2_raw=({},) * n,
        scale=({},) * n,
    )


@pytest.mark.slow
def test_a_twin_fit_is_labels_free_with_default_arguments():
    # Equal fluxes: each night's position is known only up to the point
    # reflection, which the data cannot undo.
    orbit, data, times = _short_arc(1.0)
    # A twin's reflected peak ties with its best: the gaps are small.
    positions = _arc_positions(
        orbit, times, 1.0, signs=[1, -1, 1, -1], gap=0.5
    )
    result = fit_orbit_aliases(
        data, positions=positions, t_ref=T_REF, **FAST_DEFAULTS
    )
    assert result.labels_free
    assert result.best.chi2_red < 3.0


@pytest.mark.slow
def test_a_poor_point_primary_fit_does_not_resolve_a_diameter():
    """The twin's night 1 fitted as a point primary from the grid's start
    ends 33 nats below where the same point model gets from a disc's
    positions: the +34 nats of a 0.5 mas disc are that local maximum, not a
    resolved primary. The baseline is the better of the two fits."""
    from virgil.epochs import RESOLVE_NATS, estimate_primary_diameter

    orbit, data, times = _short_arc(1.0)
    positions = _arc_positions(orbit, times, 1.0, signs=[1, -1, 1, -1])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        est = estimate_primary_diameter(data, positions)
    assert not est.resolved
    assert est.delta_nats < RESOLVE_NATS


@pytest.mark.slow
def test_a_brighter_companion_is_found_by_the_flux_flip():
    # The companion is the brighter star (1.4). The seeds are *unflipped*
    # (the true track, a fainter companion), so only the flux flip can reach
    # the reflected track with the ratio 1/1.4 that fits the data.
    orbit, data, times = _short_arc(1.4)
    positions = _arc_positions(orbit, times, 1 / 1.4)
    result = fit_orbit_aliases(
        data, positions=positions, t_ref=T_REF, primary_diam=None,
        **FAST_DEFAULTS,
    )  # fmt: skip
    assert result.best.flipped
    assert result.best.chi2_red < 3.0
    assert result.best.best["flux"] == pytest.approx(1 / 1.4, rel=0.15)
    stuck = fit_orbit_aliases(
        data, positions=positions, t_ref=T_REF, primary_diam=None,
        flip_flux=False, **FAST_DEFAULTS,
    )  # fmt: skip
    assert not stuck.best.flipped
    assert stuck.best.chi2_red > result.best.chi2_red


@pytest.mark.slow
def test_a_short_arc_fit_is_period_bound_with_default_arguments():
    # 4 days of a 60-day orbit: the window ends at 16 d and so does the fit.
    orbit, data, times = _short_arc(0.3, period=60.0)
    positions = _arc_positions(orbit, times, 0.3)
    result = fit_orbit_aliases(
        data, positions=positions, t_ref=T_REF, **FAST_DEFAULTS
    )
    assert result.p_range[1] == pytest.approx(4 * onp.ptp(times), rel=0.01)
    assert result.period_bound


@pytest.mark.slow
def test_a_resolved_primary_is_fitted_and_an_unresolved_one_is_a_point():
    for diam, resolved in ((1.05, True), (None, False)):
        orbit, data, times = _short_arc(0.3, diam=diam)
        positions = _arc_positions(orbit, times, 0.25)
        result = fit_orbit_aliases(
            data, positions=positions, t_ref=T_REF, **FAST_DEFAULTS
        )
        assert result.primary_fit["resolved"] is resolved
        if resolved:
            assert result.primary_diam == "fit"
            assert result.best.best["diam"] == pytest.approx(1.05, rel=0.2)
        else:
            assert result.primary_diam is None


def test_an_excluded_band_that_beats_the_kept_ones_is_readmitted(monkeypatch):
    # orbit_solved_problems.md, "Cycle counts from short arcs" row: the cheap check never
    # lets the short arcs silently drop a band whose seeds fit better.
    from virgil.arcs import CycleCounts

    bands = alias_bands([0.0, 100.0], (5.0, 14.0))
    one = bands[2]
    counts = CycleCounts(((one.p_lo + 1e-6, one.p_hi - 1e-6),), ())
    scores = {b.n: -100.0 for b in bands}
    monkeypatch.setattr(aliases, "_seed_scores", lambda *a: scores)
    scores[bands[5].n] = -10.0
    with pytest.warns(UserWarning, match="short arcs excluded"):
        fit, record = aliases._short_arc_bands(
            None, None, bands, {}, counts, None
        )
    assert fit == [one, bands[5]]
    assert record["kept"] == [one.n] and record["readmitted"] == [bands[5].n]
    assert bands[5].n not in record["excluded"]
    scores[bands[5].n] = -200.0  # worse than the kept band: stays out
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        fit, record = aliases._short_arc_bands(
            None, None, bands, {}, counts, None
        )
    assert fit == [one] and len(record["excluded"]) == len(bands) - 1
    # Arcs that admit no band keep them all.
    none = CycleCounts(((1.0, 2.0),), ())
    fit, _ = aliases._short_arc_bands(None, None, bands, {}, none, None)
    assert fit == bands
    # A warm start's band is always fitted.
    warm = types.SimpleNamespace(init=(0.5 * (bands[7].p_lo + bands[7].p_hi),))
    fit, _ = aliases._short_arc_bands(None, warm, bands, {}, counts, None)
    assert fit == [one, bands[7]]


def test_short_arcs_need_every_seeding_nights_sign_decided():
    # The signed cut needs each unflagged night to beat its point
    # reflection by arc_min_gap; a twin's reflections tie. A flagged night
    # does not count.
    pos = types.SimpleNamespace(
        gap_marginal=onp.array([8.0, 30.0, 0.1]),
        flagged=onp.array([False, False, True]),
    )
    assert aliases._signs_trusted(pos, 5.0)
    assert not aliases._signs_trusted(pos, 10.0)
    pos.gap_marginal = onp.array([8.0, 0.5, 30.0])
    assert not aliases._signs_trusted(pos, 5.0)
    pos.flagged = onp.ones(3, bool)
    assert not aliases._signs_trusted(pos, 5.0)


def test_the_sign_gate_tempers_the_gaps_like_the_evidence():
    # Gl 229's seeding nights had untempered gaps of 735, 44, 8.6, 140 and
    # 1126 nats, all above 5, but f = 41: tempered, 8.6 / 41 = 0.2 nats
    # decides nothing, and three of the nights' peaks were not the
    # companion's. The gate must see the tempered gaps.
    pos = types.SimpleNamespace(
        gap_marginal=onp.array([735.7, 44.4, 8.6, 140.0, 1126.1]),
        flagged=onp.zeros(5, bool),
    )
    assert aliases._signs_trusted(pos, 5.0)
    assert aliases._signs_trusted(pos, 5.0, 1.0)
    assert not aliases._signs_trusted(pos, 5.0, 41.1)
    pos.gap_marginal = onp.array([735.7, 440.0, 220.0, 260.0, 1126.1])
    assert aliases._signs_trusted(pos, 5.0, 41.1)  # decisive after tempering


# Seasonal gaps spaced like Gl 229 Ba-Bb's GRAVITY nights: two 4 d apart,
# three a month or so apart, then a gap of most of a year and one more: the
# window holds over a hundred bands. With default settings, a cold fit (no
# p_range, no init) keeps the true band and ranks it first, at any
# eccentricity and inclination.
SEASONS = T_REF + onp.array([0.0, 4.03, 63.9, 93.8, 125.7, 359.1, 413.9])


@functools.cache
def _seasonal(ecc, inc):
    truth = {**TRUTH, "ecc": ecc, "inc": inc}
    orbit = KeplerOrbit(**truth, t_ref=T_REF)
    scene = OrbitalBinary(orbit, FLUX)
    key = jax.random.PRNGKey(5)
    data = [
        simulate(
            scene,
            vlti_night(m, 4, 25.0 * k, d_phi=0.05),
            key=jax.random.fold_in(key, k),
        )
        for k, m in enumerate(SEASONS)
    ]
    return truth, data, _locate_alias_nights(data)


@pytest.mark.slow
@pytest.mark.timeout(900)  # a default short-arc fit takes 270-305 s on CI
@pytest.mark.parametrize("ecc, inc", [(0.3, 50.0), (0.7, 80.0)])
def test_a_cold_seasonal_gap_fit_ranks_the_true_alias_first(ecc, inc):
    truth, data, found = _seasonal(ecc, inc)
    result = fit_orbit_aliases(
        data, positions=found, t_ref=T_REF, n_is=200, n_samples=20
    )
    n_true = round(float(onp.ptp(result.times)) / truth["period"])
    record = result.short_arc
    assert n_true in record["kept"] + record["readmitted"]
    assert len(record["kept"]) < len(record["kept"] + record["excluded"]) / 2
    assert result.best.n == n_true
    assert abs(result.best.best["period"] - truth["period"]) < 0.05


@pytest.mark.slow
def test_short_arc_seeds_come_beside_the_position_seeds():
    # Two seasons 238 d apart, six nights (so no random starts): the true
    # orbit is 20 cycles across them (P = 12.38 d), N=19 a poor fit at a
    # prior bound. Short arcs once replaced the position-profile seeds of
    # every band they kept, and the true band's best fit then fell from
    # log Z -253 to -511, below N=19's -359. The arcs now add seeds.
    truth = dict(TRUTH, period=12.38)
    scene = OrbitalBinary(KeplerOrbit(**truth, t_ref=T_REF), FLUX)
    key = jax.random.PRNGKey(3)
    with jax.enable_x64(True):
        data = [
            simulate(
                scene,
                _night(m, rotation=7.0 * k),
                key=jax.random.fold_in(key, k),
            )
            for k, m in enumerate(EPOCHS)
        ]
        axis = onp.linspace(-30.0, 30.0, 121)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            found = epoch_positions(
                Epochs({f"n{k}": d for k, d in enumerate(data)}),
                {
                    "dra": axis,
                    "ddec": axis,
                    "flux": onp.geomspace(0.01, 1.0, 13),
                },
            )
        fits = {}
        for arcs in (True, False):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                fits[arcs] = fit_orbit_aliases(
                    data,
                    (11.0, 13.6),
                    positions=found,
                    t_ref=T_REF,
                    short_arcs=arcs,
                    a_range=(1.0, 30.0),
                    primary_diam="auto",
                )
    n_true = round(float(onp.ptp(EPOCHS)) / truth["period"])
    with_arcs, without = fits[True], fits[False]
    assert (
        with_arcs.short_arc["kept"] and n_true in with_arcs.short_arc["kept"]
    )
    assert with_arcs.best.n == without.best.n == n_true
    assert abs(with_arcs.best.best["period"] - truth["period"]) < 0.05
    log_z = lambda r, n: next(b.log_z for b in r.bands if b.n == n)  # noqa: E731
    # within a kept band the arcs never seed worse than the positions alone
    assert log_z(with_arcs, n_true) > log_z(without, n_true) - 5.0


def test_proposal_log_pdf_wraps_periodic_angles():
    x = onp.array([2.5, 0.01, 0.2, 0.5, 1.0, 1.0, 2.0, -1.0])
    eye = onp.eye(len(x))
    prop = aliases._Proposal(x, eye, eye, 0.0, 5.0)
    delta = onp.array([0.0, -0.02, 0.0, 0.0, -2.0, -1.0, 0.0, 0.0])
    unwrapped = (x + delta)[None, :]
    wrapped = unwrapped.copy()
    wrapped[0, 1] += 1.0
    wrapped[0, 4] += 360.0
    wrapped[0, 5] += 180.0
    assert float(prop.log_pdf(wrapped)[0]) == pytest.approx(
        float(prop.log_pdf(unwrapped)[0]), rel=1e-12
    )
