import warnings

import jax
import jax.numpy as np
import lineax as lx
import numpy as onp
import optimistix as optx
import numpyro
import numpyro.distributions as dist
import pytest

from virgil._precision import cast_tree, run_in
from virgil.coverage import ami_grid_record, vlti_oidata
from virgil.fitting import _GradientStoppedLM, _Objective, fit
from virgil.imaging import TSV, TV, Centroid, MaxEntropy, image_priors
from virgil.likelihood import numpyro_model, whitened_residuals
from virgil.models import (
    BinaryModelCartesian,
    Image,
    PointSource,
    QuadraticLimbDarkenedDisk,
    System,
)
from virgil.oidata import OIData
from virgil.scenes import gaussian_blob

from ._compiles import count_compiles
from ._test_data import oidata

TRUTH = BinaryModelCartesian(150.0, -80.0, 0.02)
DATA = oidata.with_model(TRUTH, key=jax.random.PRNGKey(0))
PRIORS = {
    "dra": dist.Uniform(-400.0, 400.0),
    "ddec": dist.Uniform(-400.0, 400.0),
    "flux": dist.Uniform(0.0, 0.5),
}
START = BinaryModelCartesian(140.0, -70.0, 0.01)


def _image_fit(npix=16):
    """A star + Image scene's data, and a flat starting model with priors."""
    data = OIData(ami_grid_record(pitch_m=0.5))
    truth = System(
        star=PointSource(),
        env=Image.from_brightness(
            gaussian_blob(npix, 12.0, 25.0, dra=20.0), 12.0, flux=0.1
        ),
    )
    data = data.with_model(truth, key=jax.random.PRNGKey(1))
    start = System(
        star=PointSource(), env=Image(np.zeros((npix, npix)), 12.0, flux=0.1)
    )
    return start, image_priors(start), data


@pytest.mark.parametrize("method", ["lm", "lbfgs", "adam"])
def test_fit_recovers_a_binary(method):
    options = {"max_steps": 3000} if method == "adam" else {}
    result = fit(START, PRIORS, DATA, method=method, **options)
    assert abs(result.values["dra"] - 150.0) < 3.0
    assert abs(result.values["ddec"] + 80.0) < 3.0
    assert abs(result.values["flux"] - 0.02) < 2e-3
    assert result.info["chi2_red"] < 1.5
    assert result.info["converged"] in (True, None)


@pytest.mark.validates("virgil.fitting.fit", roots=["self-consistency"])
def test_optimizers_agree_on_a_binary():
    lm = fit(START, PRIORS, DATA, method="lm")
    lbfgs = fit(START, PRIORS, DATA, method="lbfgs")
    for path in PRIORS:
        assert np.allclose(lm.values[path], lbfgs.values[path], rtol=1e-4)


def test_a_function_model_needs_starting_values():
    def binary(dra, ddec, flux):
        return BinaryModelCartesian(dra, ddec, flux)

    with pytest.raises(ValueError, match="init"):
        fit(binary, PRIORS, DATA)
    init = {"dra": 140.0, "ddec": -70.0, "flux": 0.01}
    assert abs(fit(binary, PRIORS, DATA, init=init).values["dra"] - 150) < 3


@pytest.mark.validates("virgil.fitting.fit", roots=["self-consistency"])
def test_float32_and_float64_fits_agree():
    x64 = fit(START, PRIORS, DATA, dtype="float64")
    x32 = fit(START, PRIORS, DATA, dtype="float32")
    ambient = np.float64 if jax.config.jax_enable_x64 else np.float32
    for path in PRIORS:
        assert x64.values[path].dtype == ambient  # cast back after the fit
        assert np.allclose(x32.values[path], x64.values[path], rtol=1e-3)
    # Both converge. With a fixed gtol = 1e-4, below what rounding lets the
    # float32 gradient reach, the float32 fit ran all 1000 LM steps (where
    # float64 takes 7).
    for result in (x64, x32):
        assert result.info["converged"] is True
        assert result.info["steps"] < 50


def test_lm_and_lbfgs_agree_on_a_tsv_image():
    start, priors, data = _image_fit()
    regularizers = [TSV(1e3, path="env", beam_mas=12.0)]
    # A tight tolerance, so that this tests the solution, not the stopping
    # rule (at the default tolerance their losses can differ by ~2%).
    options = {"gtol": 1e-7, "max_steps": 20_000}
    lm = fit(start, priors, data, regularizers, method="lm", **options)
    lbfgs = fit(start, priors, data, regularizers, method="lbfgs", **options)
    assert np.isclose(lm.info["loss"], lbfgs.info["loss"], rtol=2e-3)
    a, b = lm.model.env.brightness, lbfgs.model.env.brightness
    a, b = a - a.mean(), b - b.mean()
    assert np.sum(a * b) / np.sqrt(np.sum(a * a) * np.sum(b * b)) > 0.99


def test_bijections_keep_parameters_in_their_support():
    objective = _Objective(START, PRIORS, DATA)
    values = objective.constrain(objective.init())
    assert np.allclose(values["flux"], 0.01)
    assert np.allclose(values["dra"], 140.0)
    stretched = objective.constrain({p: np.asarray(50.0) for p in PRIORS})
    assert 0.0 < stretched["flux"] < 0.5
    assert -400.0 < stretched["dra"] < 400.0


def test_residuals_and_loss_are_consistent():
    objective = _Objective(START, PRIORS, DATA)
    z0 = objective.init()
    z1 = {k: v + 0.1 for k, v in z0.items()}
    half = [0.5 * np.sum(objective.residuals(z) ** 2) for z in (z0, z1)]
    loss = [objective.loss(z) for z in (z0, z1)]
    # Uniform priors are flat, so loss = 0.5 Σ r² + const.
    assert np.allclose(half[1] - half[0], loss[1] - loss[0], rtol=1e-4)


def test_normal_priors_are_least_squares_terms():
    priors = dict(PRIORS, dra=dist.Normal(150.0, 2.0))
    objective = _Objective(START, priors, DATA)
    r = objective.residuals(objective.init())
    assert r.size == DATA.n_residuals + 1
    assert np.isclose(r[-1], (140.0 - 150.0) / 2.0)


def test_non_least_squares_objectives_default_to_lbfgs():
    start, priors, data = _image_fit()
    regularizers = [TV(1.0, path="env", beam_mas=12.0)]
    objective = _Objective(start, priors, data, regularizers)
    with pytest.raises(TypeError, match="lbfgs"):
        objective.residuals(objective.init())
    result = fit(start, priors, data, regularizers, max_steps=50)
    assert result.info["method"] == "lbfgs"
    # Maximum entropy has a least-squares form, but LM is not its default.
    entropy = [MaxEntropy(1.0, path="env")]
    assert (
        fit(start, priors, data, entropy, max_steps=50).info["method"]
        == "lbfgs"
    )
    assert (
        fit(start, priors, data, entropy, method="lm", max_steps=50).info[
            "method"
        ]
        == "lm"
    )
    # A prior with no least-squares form (here a Beta flux) also falls back.
    beta = dict(PRIORS, flux=dist.Beta(1.0, 20.0))
    assert fit(START, beta, DATA).info["method"] == "lbfgs"
    # LogUniform is flat in log flux, the coordinate it is fitted in, so it
    # keeps LM (it used to fall back to L-BFGS).
    log_uniform = dict(PRIORS, flux=dist.LogUniform(1e-4, 0.5))
    assert fit(START, log_uniform, DATA).info["method"] == "lm"


def test_numpyro_model_accepts_prior_regularizers_only():
    start, _, data = _image_fit()
    pixels = dist.Normal(np.zeros((16, 16)), 3.0).to_event(2)
    priors = {"env.log_brightness": pixels}
    with pytest.raises(ValueError, match="TSV"):
        numpyro_model(
            start, priors, data, [TSV(1.0, path="env", beam_mas=12.0)]
        )
    model = numpyro_model(start, priors, data, [Centroid(10.0, path="env")])
    trace = numpyro.handlers.trace(
        numpyro.handlers.seed(model, jax.random.PRNGKey(0))
    ).get_trace()
    assert {"loglike", "regularizer_0"} <= set(trace)


def test_fit_rejects_bad_paths_flux_priors_and_methods():
    # zodiax raises ValueError (0.4.1) or KeyError (newer) for unknown paths.
    with pytest.raises((KeyError, ValueError), match="nonsense"):
        fit(TRUTH, {"nonsense": dist.Uniform(0.0, 1.0)}, DATA)
    with pytest.raises(ValueError, match="negative"):
        fit(TRUTH, {"flux": dist.Normal(0.0, 1.0)}, DATA)
    with pytest.raises(ValueError, match="method"):
        fit(START, PRIORS, DATA, method="newton")
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="max_step_size"):
            fit(START, PRIORS, DATA, method="lbfgs", max_step_size=bad)


def test_precision_helpers():
    tree = {"a": np.ones(3), "b": np.ones(2, dtype=np.complex64), "n": 3}
    before = jax.config.jax_enable_x64
    with run_in("float64"):
        cast = cast_tree(tree, "float64")
        assert cast["a"].dtype == np.float64
        assert cast["b"].dtype == np.complex128
        assert cast["n"] == 3
    assert jax.config.jax_enable_x64 == before
    with pytest.raises(ValueError):
        with run_in("float16"):
            pass
    assert onp.asarray(cast_tree(tree, "float32")["a"]).dtype == onp.float32


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_a_warm_start_still_converges(dtype):
    # Starting from the solution at a nearby weight, the gradient is small
    # from the outset; the fit must still move to the new solution rather
    # than stopping at once.
    start, priors, data = _image_fit()
    strong = fit(
        start, priors, data, [MaxEntropy(10.0, path="env")], dtype=dtype
    )
    weak = fit(
        start,
        priors,
        data,
        [MaxEntropy(1.0, path="env")],
        init=strong.values,
        dtype=dtype,
    )
    cold = fit(start, priors, data, [MaxEntropy(1.0, path="env")], dtype=dtype)
    assert weak.info["steps"] > 4  # stalled fits took 1-4 steps
    assert weak.info["loss"] <= cold.info["loss"] * (1 + 1e-2)
    # Both reach the minimum, not a stationary point with most pixels dark
    # (χ²_red ~ 3-6, at a loss that varied by 2x across platforms).
    assert weak.info["chi2_red"] < 1.5
    assert cold.info["chi2_red"] < 1.5


@pytest.mark.parametrize("method", ["lm", "lbfgs"])
def test_grad_norm_is_small_at_convergence_and_large_when_stopped(method):
    converged = fit(START, PRIORS, DATA, method=method)
    assert converged.info["converged"]
    # The convergence test compares the same norm with at most gtol (for
    # L-BFGS, at the point before the last step).
    assert 0.0 <= converged.info["grad_norm"] <= 1e-3
    with pytest.warns(RuntimeWarning):
        stopped = fit(START, PRIORS, DATA, method=method, max_steps=1)
    assert stopped.info["grad_norm"] > 10 * converged.info["grad_norm"]
    assert stopped.info["grad_norm"] > 1e-4


def test_adam_reports_a_grad_norm():
    result = fit(START, PRIORS, DATA, method="adam", max_steps=5)
    assert np.isfinite(result.info["grad_norm"])
    assert result.info["grad_norm"] > 0.0


def test_lbfgs_warning_says_it_hit_the_step_limit():
    with pytest.warns(RuntimeWarning, match="in 2 steps, the step limit"):
        fit(START, PRIORS, DATA, method="lbfgs", max_steps=2)


@pytest.mark.parametrize(
    "loss, gradient, moved, message",
    [
        (0.0, 1.0, False, "stopped after 7 of 100 steps, when a step no"),
        (0.0, float("nan"), False, "gradient was not finite"),
        (0.0, float("nan"), True, "gradient was not finite"),
        (float("nan"), 1.0, True, "loss or gradient was not finite"),
    ],
)
def test_lbfgs_warning_says_why_it_stopped_early(
    monkeypatch, loss, gradient, moved, message
):
    # L-BFGS also stops before max_steps, unconverged, when a step no
    # longer changes the parameters (out of precision) or the gradient is
    # NaN. Both used to warn "did not converge in N steps", which reads as
    # the step limit, and a NaN must not be blamed on precision.
    import virgil.fitting

    def stopped(problem, z0, *_):
        return (7, z0, {"grad": z0}, loss, gradient, moved), 1e-3

    monkeypatch.setattr(virgil.fitting, "_lbfgs_run", stopped)
    with pytest.warns(RuntimeWarning, match=message):
        result = fit(
            START, PRIORS, DATA, method="lbfgs", max_steps=100, dtype="float32"
        )
    assert result.info["converged"] is False
    stop = "stalled" if not moved and gradient == 1.0 else "non-finite"
    assert result.info["stop"] == stop


def test_chunked_lbfgs_takes_the_unchunked_path(monkeypatch):
    # With a time limit or a progress report, L-BFGS runs in chunks of
    # steps and carries its whole state across them, so it takes the same
    # path as one uninterrupted loop.
    import virgil.fitting

    plain = fit(START, PRIORS, DATA, method="lbfgs")
    monkeypatch.setattr(virgil.fitting, "_LBFGS_CHUNK", 3)
    reports = []
    chunked = fit(
        START,
        PRIORS,
        DATA,
        method="lbfgs",
        time_limit=1e6,
        progress=reports.append,
    )
    assert plain.info["converged"] and plain.info["stop"] is None
    assert chunked.info["converged"] and chunked.info["stop"] is None
    assert chunked.info["steps"] == plain.info["steps"] > 3
    for path in PRIORS:
        onp.testing.assert_allclose(
            chunked.values[path], plain.values[path], rtol=1e-6
        )
    # A report after every chunk that ended short of convergence, and one
    # on completion.
    steps = [r["steps"] for r in reports]
    assert steps == [*range(3, plain.info["steps"], 3), plain.info["steps"]]
    assert all(r["elapsed"] >= 0 and onp.isfinite(r["loss"]) for r in reports)


@pytest.mark.parametrize("method, chunk", [("lbfgs", 2), ("lm", 1)])
def test_a_fit_stops_at_its_time_limit(monkeypatch, method, chunk):
    # A start that runs for hours must not block the starts after it: the
    # optimizer stops after the first chunk past the time limit.
    import virgil.fitting

    monkeypatch.setattr(virgil.fitting, "_LBFGS_CHUNK", chunk)
    monkeypatch.setattr(virgil.fitting, "_LM_CHUNK", chunk)
    far = BinaryModelCartesian(100.0, -20.0, 0.05)
    with pytest.warns(RuntimeWarning, match="time_limit of 0.0 s"):
        result = fit(far, PRIORS, DATA, method=method, time_limit=0.0)
    assert result.info["converged"] is False
    assert result.info["stop"] == "time"
    assert result.info["steps"] == chunk


def test_bad_time_limits_are_rejected():
    with pytest.raises(ValueError, match="time_limit"):
        fit(START, PRIORS, DATA, time_limit=-1.0)


def _adam(**kw):
    kw = {"max_steps": 40, "method": "adam"} | kw
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return fit(START, PRIORS, DATA, **kw)


@jax.enable_x64(True)
def test_adam_matches_plain_optax_adam(monkeypatch):
    import optax

    import virgil.fitting

    monkeypatch.setattr(virgil.fitting, "_ADAM_CHUNK", 7)  # several chunks
    problem = cast_tree(_Objective(START, PRIORS, (DATA,)), "float64")
    z = problem.init(None)
    scale = float(DATA.n_independent)
    optimizer, state = optax.adam(1e-2), optax.adam(1e-2).init(z)
    for _ in range(40):
        grad = jax.grad(lambda z: problem.loss(z) / scale)(z)
        updates, state = optimizer.update(grad, state, z)
        z = optax.apply_updates(z, updates)
    result = _adam()
    for path, value in problem.constrain(z).items():
        onp.testing.assert_allclose(result.values[path], value, rtol=1e-12)


def test_adam_loss_history_and_eps():
    result = _adam()
    history = result.info["loss_history"]
    assert history.shape == (result.info["steps"],) == (40,)
    assert "loss_history" not in fit(START, PRIORS, DATA, max_steps=3).info
    before_last = _adam(max_steps=39).info["loss"]
    assert history[-1] == pytest.approx(before_last, rel=1e-12)
    other = _adam(adam_eps=1.0)
    assert not onp.allclose(
        other.values["dra"], result.values["dra"], rtol=1e-6
    )


def test_adam_ftol_stops_at_the_first_plateau_step(monkeypatch):
    import virgil.fitting

    monkeypatch.setattr(virgil.fitting, "_ADAM_CHUNK", 7)
    full = _adam(max_steps=120).info["loss_history"]
    window = 10
    ts = onp.arange(2 * window - 1, len(full))
    ftol = float(onp.sort(onp.abs(full[ts] - full[ts - window]))[len(ts) // 2])
    hits = virgil.fitting._plateau(full, ts, window, ftol)
    t = int(ts[hits[0]])
    result = _adam(max_steps=120, ftol_abs=ftol, ftol_window=window)
    assert result.info["steps"] == t and t % 7 != 0  # mid-chunk
    assert result.info["converged"] is True and result.info["stop"] is None
    onp.testing.assert_array_equal(result.info["loss_history"], full[:t])
    stopped_at_t = _adam(max_steps=t)
    for path in PRIORS:
        onp.testing.assert_array_equal(
            result.values[path], stopped_at_t.values[path]
        )


def test_adam_stops_at_its_time_limit_and_reports_convergence(monkeypatch):
    import virgil.fitting

    monkeypatch.setattr(virgil.fitting, "_ADAM_CHUNK", 5)
    result = _adam(max_steps=50, time_limit=0.0)
    assert result.info["converged"] is False and result.info["stop"] == "time"
    assert result.info["steps"] == 5
    short = _adam(max_steps=3)
    assert short.info["converged"] is False and short.info["stop"] == "limit"
    loose = _adam(max_steps=3, gtol=1e30)
    assert loose.info["converged"] is True and loose.info["stop"] is None


def test_adam_ftol_with_a_time_limit(monkeypatch):
    import virgil.fitting

    monkeypatch.setattr(virgil.fitting, "_ADAM_CHUNK", 7)
    reports = []
    result = _adam(
        max_steps=120,
        ftol_abs=1e30,  # fires at the first step the window allows
        ftol_window=10,
        time_limit=1e6,
        progress=reports.append,
    )
    assert result.info["steps"] == 19 and result.info["converged"] is True
    assert result.info["stop"] is None
    assert reports[-1]["steps"] == 19  # the final report is at the stop
    # an expired budget stops first, at the end of the first chunk
    late = _adam(max_steps=120, ftol_abs=1e-30, ftol_window=10, time_limit=0.0)
    assert late.info["stop"] == "time" and late.info["steps"] == 7


def _stops(loss, window=100, tol=0.1):
    from virgil.fitting import _plateau

    t = onp.arange(2 * window - 1, len(loss))
    return t[_plateau(loss, t, window, tol)[:1]].tolist()


def test_the_window_test_ignores_oscillation_and_drift():
    t = onp.arange(2000.0)
    assert _stops(100 + 5 * onp.sin(2 * onp.pi * t / 100)) == []
    assert _stops(1e5 - t) == []  # drifting by 1 per step
    noise = 1e-3 * onp.random.default_rng(0).standard_normal(2000)
    assert _stops(100 + noise) == [199]
    assert _adam(max_steps=250, ftol_abs=None).info["steps"] == 250


@jax.enable_x64(True)
def test_adam_learning_rate_schedule():
    import optax

    constant = _adam(learning_rate=1e-2)
    decay = optax.exponential_decay(1e-2, 1, 0.8)
    scheduled = _adam(learning_rate=decay)
    assert not onp.allclose(
        scheduled.values["dra"], constant.values["dra"], rtol=1e-6
    )
    same = _adam(learning_rate=optax.constant_schedule(1e-2))
    onp.testing.assert_allclose(
        same.values["dra"], constant.values["dra"], rtol=1e-12
    )
    step = lambda r: abs(r.values["dra"] - START.dra)  # noqa: E731
    assert step(scheduled) < step(constant)


def test_bad_ftol_abs_is_rejected():
    for bad in (-1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="ftol_abs"):
            fit(START, PRIORS, DATA, method="adam", ftol_abs=bad)


def test_bad_ftol_window_is_rejected():
    with pytest.raises(ValueError, match="ftol_window"):
        fit(START, PRIORS, DATA, method="adam", ftol_window=0)


def test_poles_of_isotropic_priors_are_not_prior_bounds():
    # A face-on inclination is an end of the isotropic prior's flat
    # coordinate (its CDF), but a pole of the coordinates, not an edge of
    # the prior: at_bound must not flag it. A Uniform's edge is flagged.
    from types import SimpleNamespace

    from virgil.fitting import _at_bound
    from virgil.priors import IsotropicInclination, IsotropicLatitude

    problem = SimpleNamespace(
        priors={
            "inc": IsotropicInclination(),
            "lat": IsotropicLatitude(),
            "ecc": dist.Uniform(0.0, 0.9),
        },
        noise={},
        flat=True,
    )
    values = {
        "inc": np.asarray(0.5),
        "lat": np.asarray(0.5 * onp.pi - 1e-4),
        "ecc": 0.4,
    }
    assert _at_bound(problem, values) == ()
    assert _at_bound(problem, dict(values, ecc=0.8999)) == ("ecc",)
    # A truncated isotropic range has a genuine non-pole bound.
    capped = SimpleNamespace(
        priors={
            "inc": IsotropicInclination(0.0, 30.0),
            "lat": IsotropicLatitude(-0.5 * onp.pi, 1.0),
        },
        noise={},
        flat=True,
    )
    near_poles = {
        "inc": np.asarray(0.5),
        "lat": np.asarray(-0.5 * onp.pi + 1e-4),
    }
    assert _at_bound(capped, near_poles) == ()
    assert _at_bound(capped, dict(near_poles, inc=np.asarray(29.999))) == (
        "inc",
    )
    assert _at_bound(capped, dict(near_poles, lat=np.asarray(0.9999))) == (
        "lat",
    )


@pytest.mark.filterwarnings("ignore:fit.*did not converge")
def test_fits_report_parameters_at_a_prior_bound():
    # The data's flux is 0.02, above this prior's upper bound: the fit runs
    # into the bound, which info["at_bound"] reports.
    assert fit(START, PRIORS, DATA).info["at_bound"] == ()
    capped = dict(PRIORS, flux=dist.Uniform(0.0, 0.01))
    result = fit(START, capped, DATA, init={"flux": 0.005})
    assert result.info["at_bound"] == ("flux",)
    # The same in a flat coordinate (log flux for a LogUniform prior).
    capped = dict(PRIORS, flux=dist.LogUniform(1e-4, 0.01))
    result = fit(START, capped, DATA, init={"flux": 0.005})
    assert result.info["at_bound"] == ("flux",)


@pytest.mark.filterwarnings("ignore:fit.*did not converge")
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_a_cold_maxent_fit_does_not_collapse(dtype):
    # From a flat image, a weakly regularized fit once took a first
    # quasi-Newton step of ~10 in log-brightness, switching most pixels
    # off for good (their gradients vanish with their flux): it stopped on
    # a few bright pixels with χ² ≈ 1330, where a strongly regularized fit
    # reaches ≈ 245. A weaker penalty must fit the data at least as well.
    # (In float32 the line search runs out of precision near the minimum,
    # so that fit may stop unconverged.)
    start, priors, data = _image_fit()
    weak, strong = (
        fit(start, priors, data, [MaxEntropy(w, path="env")], dtype=dtype)
        for w in (1.0, 100.0)
    )
    assert weak.info["chi2"][0] <= strong.info["chi2"][0]
    b = weak.model.env.brightness
    assert np.mean(b > 1e-3 * np.max(b)) > 0.1  # collapsed fits: 2%


@pytest.mark.filterwarnings("ignore:fit.*did not converge")
@pytest.mark.parametrize("method", ["lm", "lbfgs", "adam"])
def test_repeated_fits_do_not_recompile(method):
    # The solvers are jitted once at module level, so a second fit of a
    # problem with the same structure (another start, regularizer weight
    # or dataset of the same size) reuses the compilation. When they were
    # defined inside each call, every fit recompiled, which took most of
    # its time.
    start, priors, data = _image_fit()
    # A short run: L-BFGS restarts a round that ends unconverged, and the
    # restart's first call is a separate compilation. Whether the second fit
    # needs one depends on the platform's numerics and on which tests ran
    # before in the worker, so a converged-length run made this test flaky.
    options = {"max_steps": 20} if method in ("adam", "lbfgs") else {}
    again = data.with_model(start, key=jax.random.PRNGKey(2))

    def fit_tsv(data, weight):
        regs = [TSV(weight, path="env", beam_mas=12.0)]
        return fit(start, priors, data, regs, method=method, **options)

    fit_tsv(data, 10.0)
    with count_compiles() as compiles:
        fit_tsv(again, 3.0)
    assert not compiles


@pytest.mark.parametrize(
    "method, options",
    [
        ("lm", [{"gtol": 1e-4}, {"gtol": 2e-4}, {"gtol": 3e-4}]),
        ("lbfgs", [{}, {"gtol": 2e-4}, {"max_step_size": 1.5}]),
        ("lbfgs", [{"max_steps": 100}, {"max_steps": 200}, {}]),
        ("lbfgs", [{"time_limit": t} for t in (100.0, 200.0, 300.0)]),
        ("adam", [{"learning_rate": r} for r in (1e-2, 2e-2, 3e-2)]),
    ],
)
def test_new_prior_bounds_and_options_do_not_recompile(method, options):
    # Python numbers in the priors (here a Uniform's lower bound) and in
    # fit's options were static in the jitted solvers, so each new value
    # recompiled the fit (~1.2 s each for a small model, ~4 s on a 64²
    # image). They are now traced arrays. (LM's and Adam's max_steps set a
    # loop's length and stay static.)
    def fit_with(low, extra):
        priors = dict(PRIORS, flux=dist.Uniform(low, 0.5))
        steps = {"max_steps": 50} if method == "adam" else {}
        return fit(START, priors, DATA, method=method, **steps, **extra)

    fit_with(0.0, options[0])
    with count_compiles() as compiles:
        fit_with(1e-4, options[1])
        fit_with(1e-3, options[2])
    assert not compiles


def test_fit_recovers_error_scales():
    # Noise twice the stated errors: the fitted scales should be near 2.
    data = oidata.with_model(TRUTH, key=jax.random.PRNGKey(3), noise_scale=2.0)
    noise = {
        "vis_scale": dist.Uniform(0.0, 10.0),
        "phi_scale": dist.Uniform(0.0, 10.0),
    }
    result = fit(START, PRIORS, data, noise=noise)
    assert result.info["method"] == "lbfgs"
    # The maximum-likelihood scale is the rms of the residuals at the truth,
    # whitened by the stated errors. There are only 15 independent closure
    # phases, so one draw scatters by ~20% about the injected 2: compare
    # with this draw's own rms rather than with 2.
    whitened = onp.asarray(whitened_residuals(TRUTH, data))
    n_vis = onp.size(data.vis)
    rms = {
        "vis_scale": onp.sqrt(onp.mean(whitened[:n_vis] ** 2)),
        # (the periodic penalty rows that follow the whitened closure
        # phases are not residuals to rescale)
        "phi_scale": onp.sqrt(
            onp.mean(whitened[n_vis : n_vis + data.cp_noise.size] ** 2)
        ),
    }
    for term, expected in rms.items():
        assert result.values[f"noise.{term}"] == pytest.approx(
            expected, rel=0.15
        )
    assert result.values["noise.vis_scale"] > 1.4  # 2x noise, well sampled
    # χ² is computed with the inflated errors.
    assert abs(result.info["chi2_red"] - 1.0) < 0.05
    assert abs(result.values["dra"] - 150.0) < 5.0


def test_fit_noise_per_dataset_and_validation():
    noise = [{"phi_error": dist.Uniform(0.0, 1.0)}, {}]
    result = fit(START, PRIORS, [DATA, DATA], noise=noise)
    assert set(result.values) == set(PRIORS) | {"noise[0].phi_error"}
    with pytest.raises(ValueError, match="2 datasets"):
        fit(START, PRIORS, [DATA, DATA], noise=[{}])
    with pytest.raises(ValueError, match="Unknown noise term"):
        fit(START, PRIORS, DATA, noise={"jitter": dist.Uniform(0.0, 1.0)})
    with pytest.raises(ValueError, match="non-negative"):
        fit(START, PRIORS, DATA, noise={"vis_scale": dist.Normal(1.0, 1.0)})
    with pytest.raises(TypeError, match="least-squares"):
        fit(
            START,
            PRIORS,
            DATA,
            method="lm",
            noise={"vis_scale": dist.Uniform(0.0, 5.0)},
        )


def test_numpyro_model_samples_noise_terms():
    noise = {"vis_scale": dist.Uniform(0.0, 5.0)}
    model = numpyro_model(START, PRIORS, DATA, noise=noise)
    trace = numpyro.handlers.trace(numpyro.handlers.seed(model, 0)).get_trace()
    assert "noise.vis_scale" in trace


def _binary_fn(dra, ddec, flux, **_):
    return BinaryModelCartesian(dra, ddec, flux)


def test_numpyro_model_tied_noise_terms_match_the_likelihood():
    # A noise term given as a function of the sampled values is recorded
    # as a deterministic site and enters the likelihood like a fitted one.
    from numpyro.infer.util import log_density

    from virgil.likelihood import model_loglike

    priors = {**PRIORS, "s": dist.LogUniform(0.5, 5.0)}
    noise = [
        {"vis_scale": lambda v: v["s"]},
        {"vis_scale": lambda v: 2.0 * v["s"]},
    ]
    model = numpyro_model(_binary_fn, priors, [DATA, DATA], noise=noise)
    values = {"dra": 150.0, "ddec": -80.0, "flux": 0.02, "s": 1.5}
    trace = numpyro.handlers.trace(
        numpyro.handlers.substitute(numpyro.handlers.seed(model, 0), values)
    ).get_trace()
    assert trace["noise[1].vis_scale"]["type"] == "deterministic"
    assert float(trace["noise[1].vis_scale"]["value"]) == pytest.approx(3.0)
    binary = BinaryModelCartesian(150.0, -80.0, 0.02)
    expected = (
        model_loglike(binary, DATA, vis_scale=1.5)
        + model_loglike(binary, DATA, vis_scale=3.0)
        + sum(priors[k].log_prob(values[k]) for k in priors)
    )
    got = log_density(model, (), {}, values)[0]
    assert float(got) == pytest.approx(float(expected), rel=1e-5)


def test_fit_with_hierarchical_error_scales():
    from virgil.priors import hierarchical_scales

    quiet = oidata.with_model(TRUTH, key=jax.random.PRNGKey(1))
    noisy = oidata.with_model(
        TRUTH, key=jax.random.PRNGKey(2), noise_scale=3.0
    )
    population, scales = hierarchical_scales("v2", 2)
    priors = {**PRIORS, **population}
    init = {
        "dra": 150.0,
        "ddec": -80.0,
        "flux": 0.02,
        "v2_median": 1.0,
        "v2_spread": 0.5,
        "v2_log": np.zeros(2),
    }
    result = fit(
        _binary_fn,
        priors,
        [quiet, noisy],
        noise=[{"vis_scale": s} for s in scales],
        init=init,
    )
    # Unpooled reference: an independent log-uniform scale per dataset.
    unpooled = fit(
        _binary_fn,
        PRIORS,
        [quiet, noisy],
        noise=[{"vis_scale": dist.LogUniform(0.1, 10.0)} for _ in range(2)],
        init={"dra": 150.0, "ddec": -80.0, "flux": 0.02},
    )

    def ratio(r):
        return float(
            r.values["noise[1].vis_scale"] / r.values["noise[0].vis_scale"]
        )

    assert ratio(unpooled) == pytest.approx(3.0, rel=0.25)
    # The two-member population shrinks the scales towards each other
    # (the ratio drops by 10-15% for these draws, which differ between
    # float32 and x64), but they stay clearly separated.
    assert 1.5 < ratio(result) <= ratio(unpooled) * (1 + 1e-3)
    assert abs(result.values["dra"] - 150.0) < 5.0
    # Centred hierarchical scales support gauss_newton_mass and
    # gauss_newton_diag under flat_coordinates=True, while method="lm" raises
    # NoLeastSquaresForm.
    from virgil.fitting import (
        NoLeastSquaresForm,
        gauss_newton_diag,
        gauss_newton_mass,
    )

    with pytest.raises(NoLeastSquaresForm):
        fit(
            _binary_fn,
            priors,
            [quiet, noisy],
            noise=[{"vis_scale": s} for s in scales],
            init=init,
            method="lm",
        )
    diag = gauss_newton_diag(
        _binary_fn,
        priors,
        [quiet, noisy],
        result.values,
        noise=[{"vis_scale": s} for s in scales],
    )
    assert all(onp.all(onp.asarray(v) > 0.0) for v in diag.values())
    mass = gauss_newton_mass(
        _binary_fn,
        priors,
        [quiet, noisy],
        result.values,
        noise=[{"vis_scale": s} for s in scales],
        flat_coordinates=True,
    )
    ((sites, cov),) = mass["inverse_mass_matrix"].items()
    assert {"v2_median", "v2_spread", "v2_log"} <= set(sites)
    eig = onp.linalg.eigvalsh(onp.asarray(cov))
    assert onp.all(onp.isfinite(eig)) and eig.min() > 0.0


def test_numpyro_model_adds_each_population_density_once():
    from numpyro.infer.util import log_density

    from virgil.likelihood import model_loglike
    from virgil.priors import hierarchical_scales

    population, scales = hierarchical_scales("e", 2)
    priors = {**PRIORS, **population}
    # Member 0 scales both observables of dataset 0: its density counts once.
    noise = [
        {"vis_scale": scales[0], "phi_scale": scales[0]},
        {"vis_scale": scales[1]},
    ]
    model = numpyro_model(_binary_fn, priors, [DATA, DATA], noise=noise)
    log_s = np.log(np.array([1.5, 0.8]))
    values = {
        "dra": 150.0,
        "ddec": -80.0,
        "flux": 0.02,
        "e_median": 1.2,
        "e_spread": 0.3,
        "e_log": log_s,
    }
    binary = BinaryModelCartesian(150.0, -80.0, 0.02)
    expected = (
        model_loglike(binary, DATA, vis_scale=1.5, phi_scale=1.5)
        + model_loglike(binary, DATA, vis_scale=0.8)
        + sum(np.sum(priors[k].log_prob(values[k])) for k in priors)
        + np.sum(dist.Normal(np.log(1.2), 0.3).log_prob(log_s))
    )
    got = log_density(model, (), {}, values)[0]
    assert float(got) == pytest.approx(float(expected), rel=1e-5)


@pytest.mark.parametrize("method", ["lm", "lbfgs"])
def test_a_start_at_an_exact_optimum_is_converged(method):
    # Noise-free data and the true parameters: chi2 ~ 0, gradient ~ rounding
    # noise. A purely relative stopping test never passes here.
    # Built in float64, so that the data really are the model's output.
    with jax.enable_x64(True):
        truth = BinaryModelCartesian(4.97, -3.36, 0.05)
        clean = vlti_oidata(
            wavelengths_m=onp.linspace(1.5e-6, 2.4e-6, 6)
        ).with_model(truth)
        priors = {
            "dra": dist.Uniform(-60.0, 60.0),
            "ddec": dist.Uniform(-60.0, 60.0),
            "flux": dist.Uniform(0.0, 1.0),
        }
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = fit(truth, priors, clean, method=method)
    assert result.info["converged"] is True
    # L-BFGS tests the starting gradient before its first step, so an
    # optimal start must not be moved at all
    assert (
        result.info["steps"] == 0
        if method == "lbfgs"
        else result.info["steps"] <= 3
    )
    assert onp.max(result.info["chi2"]) < 1e-12


def _rim_problem():
    from virgil.coverage import vlti_oidata
    from virgil.models import ModulatedGaussianRim

    truth = System(
        star=PointSource(),
        rim=ModulatedGaussianRim(
            6.0,
            1.0,
            45.0,
            30.0,
            onp.array([0.5]),
            onp.array([120.0]),
            0.8,
        ),
    )
    data = vlti_oidata(wavelengths_m=onp.linspace(1.5e-6, 2.4e-6, 6))
    data = data.with_model(truth, key=jax.random.PRNGKey(3))
    return truth, truth.set("rim.diam", 5.5), data


_UNIFORM_FORMS = {
    "array": lambda lo, hi: dist.Uniform(onp.full(1, lo), onp.full(1, hi)),
    "expand": lambda lo, hi: dist.Uniform(lo, hi).expand([1]),
    "to_event": lambda lo, hi: dist.Uniform(lo, hi).expand([1]).to_event(1),
}


def _rim_priors(form):
    make = _UNIFORM_FORMS[form]
    return {
        "rim.diam": dist.Uniform(1.0, 20.0),
        "rim.flux": dist.Uniform(0.0, 5.0),
        "rim.az_amps": make(0.0, 1.0),
        "rim.az_pas": make(0.0, 360.0),
    }


@pytest.mark.parametrize("form", list(_UNIFORM_FORMS))
def test_wrapped_uniform_priors_default_to_lm(form):
    _, start, data = _rim_problem()
    result = fit(start, _rim_priors(form), data)
    assert result.info["method"] == "lm"


def test_expanded_priors_recover_the_rim():
    _, start, data = _rim_problem()
    result = fit(start, _rim_priors("expand"), data)
    assert result.info["method"] == "lm"
    assert onp.isclose(result.model.get("rim.diam"), 6.0, rtol=0.05)
    assert onp.allclose(result.model.get("rim.az_amps"), [0.5], atol=0.1)


def test_fit_accepts_the_old_regularisers_keyword():
    new = fit(START, PRIORS, DATA, regularizers=())
    with pytest.warns(FutureWarning, match="regularisers="):
        old = fit(START, PRIORS, DATA, regularisers=())
    assert abs(old.values["dra"] - new.values["dra"]) < 1e-6


@pytest.mark.parametrize("method", ["lm", "lbfgs"])
def test_a_converged_fit_reports_progress_at_least_once(method):
    # Converging inside the first chunk used to skip the report.
    reports = []
    result = fit(START, PRIORS, DATA, method=method, progress=reports.append)
    assert result.info["converged"]
    assert reports and reports[-1]["steps"] == result.info["steps"]
    assert onp.isfinite(reports[-1]["loss"])


def test_bound_fraction_reads_a_log_uniform_in_x_when_not_flat():
    from virgil.fitting import _bound_fraction

    prior = dist.LogUniform(1.0, 100.0)
    # flat: log x, so 10 is halfway; not flat: x, so 10 is 9/99 of the way
    assert _bound_fraction(prior, 10.0) == pytest.approx(0.5)
    assert _bound_fraction(prior, 10.0, flat=False) == pytest.approx(9 / 99)
    assert _bound_fraction(prior, 100.0, flat=False) == pytest.approx(1.0)


def test_lbfgs_restarts_reach_the_same_minimum(monkeypatch):
    """Short rounds restart L-BFGS often; the fit still converges to the
    same point, and the step count is the sum over the rounds."""
    from virgil import fitting

    whole = fit(START, PRIORS, DATA, method="lbfgs")
    assert whole.info["converged"] and whole.info["steps"] > 5
    monkeypatch.setattr(fitting, "_LBFGS_ROUND", 3)
    rounds = fit(START, PRIORS, DATA, method="lbfgs")
    assert rounds.info["converged"]
    assert rounds.info["steps"] > 3  # more than one round
    assert rounds.info["loss"] == pytest.approx(whole.info["loss"], abs=1e-4)
    # Stopped by the step limit, the total is the limit exactly.
    with pytest.warns(RuntimeWarning, match="step limit"):
        capped = fit(START, PRIORS, DATA, method="lbfgs", max_steps=7)
    assert capped.info["steps"] == 7 and capped.info["stop"] == "limit"
    # A small gain ends the restarts, not the fit: the rest of max_steps
    # runs as one round, and the stop is still the step limit.
    monkeypatch.setattr(fitting, "_RESTART_GAIN", float("inf"))
    with pytest.warns(RuntimeWarning, match="step limit"):
        last = fit(START, PRIORS, DATA, method="lbfgs", max_steps=7)
    assert last.info["steps"] == 7 and last.info["stop"] == "limit"


# --- The Gauss–Newton curvature (design/history/gauss_newton_plan.md, GN2) ---

NOISY = oidata.with_model(TRUTH, key=jax.random.PRNGKey(3), noise_scale=1.7)
SCALES = {
    "vis_scale": dist.LogUniform(0.1, 10.0),
    "phi_scale": dist.LogUniform(0.1, 10.0),
}


def _curvature(model, priors, data, values, noise=None, flat=False):
    """``(problem, z, J, P)`` of ``gauss_newton_mass``'s curvature, in float64."""
    with run_in("float64"):
        problem = cast_tree(
            _Objective(model, priors, data, noise=noise, flat=flat), "float64"
        )
        z = problem.init(cast_tree(dict(values), "float64"))
        flat_z, unravel = problem.ravel(z)
        jac = jax.jacfwd(lambda x: problem.curvature_rows(unravel(x)))(flat_z)
        return (
            problem,
            z,
            onp.asarray(jac),
            onp.asarray(problem.curvature_diag(z, True)),
        )


def _hessian_check(model, priors, data, values, noise=None, flat=False):
    """``(sites, eigenvalues of F⁻¹H, mass)``: H is the exact Hessian of the
    numpyro potential in the coordinates of the mass matrix."""
    from numpyro.infer import init_to_value
    from numpyro.infer.util import initialize_model

    from virgil.fitting import gauss_newton_mass

    with run_in("float64"):
        mass = gauss_newton_mass(
            model, priors, data, values, noise=noise, flat_coordinates=flat
        )
        (sites,) = mass["dense_mass"]
        covariance = mass["inverse_mass_matrix"][sites]
        m, p, d, n = cast_tree((model, priors, data, noise), "float64")
        posterior = numpyro_model(m, p, d, noise=n, flat_coordinates=flat)
        info = initialize_model(
            jax.random.PRNGKey(1),
            posterior,
            init_strategy=init_to_value(
                values=cast_tree(dict(values), "float64")
            ),
        )
        z = info.param_info.z
        shapes = [z[k].shape for k in sites]
        sizes = [int(onp.prod(s)) for s in shapes]

        def potential(x):
            parts = np.split(x, onp.cumsum(sizes)[:-1])
            return info.potential_fn(
                {k: v.reshape(s) for k, v, s in zip(sites, parts, shapes)}
            )

        flat_z = np.concatenate([np.ravel(z[k]) for k in sites])
        hessian = onp.asarray(jax.hessian(potential)(flat_z))
    return sites, onp.linalg.eigvals(covariance @ hessian).real, mass


def _sigmoid(z):
    return 1.0 / (1.0 + onp.exp(-z))


def test_bounded_priors_give_the_curvature_of_their_transforms():
    # With flat coordinates a LogUniform is flat in the fitted coordinate, so
    # NUTS's potential is -log σ(1 - σ) and P = 2σ(1 - σ), as for a Uniform.
    priors = PRIORS | {"flux": dist.LogUniform(1e-3, 0.5)}
    values = {"dra": 150.0, "ddec": -80.0, "flux": 0.02}
    problem, z, _, p = _curvature(START, priors, DATA, values, flat=True)
    s = _sigmoid(onp.array([float(z[k]) for k in problem.paths]))
    onp.testing.assert_allclose(p, 2 * s * (1 - s), rtol=1e-10)
    # Without the Jacobian (fit's loss), flat priors add nothing.
    with run_in("float64"):
        none = problem.curvature_diag(z, False)
    onp.testing.assert_array_equal(onp.asarray(none), 0.0)


def test_log_uniform_under_numpyros_bijection_matches_the_potential():
    # Without flat coordinates, numpyro maps [a, b] by a + (b - a)σ(z), so
    # P = max(0, (log x)'' + 2σ(1 - σ)): compare with the Hessian of the
    # numpyro potential of the 1-D prior.
    from numpyro.infer import init_to_value
    from numpyro.infer.util import initialize_model

    low, high = 1e-3, 0.5
    prior = dist.LogUniform(low, high)
    for flux in (0.004, 0.02, 0.3):
        problem, z, _, p = _curvature(
            START, {"flux": prior}, (), {"flux": flux}, flat=False
        )

        def model():
            numpyro.sample("flux", prior)

        with run_in("float64"):
            info = initialize_model(
                jax.random.PRNGKey(0),
                model,
                init_strategy=init_to_value(values={"flux": flux}),
            )
            zz = info.param_info.z["flux"]
            potential = lambda v: info.potential_fn({"flux": v})
            hessian = float(jax.hessian(potential)(zz))
            log_x = lambda v: np.log(low + (high - low) * jax.nn.sigmoid(v))
            analytic = float(jax.grad(jax.grad(log_x))(zz))
            s = float(jax.nn.sigmoid(zz))
        assert float(z["flux"]) == pytest.approx(float(zz))
        assert hessian == pytest.approx(analytic + 2 * s * (1 - s), rel=1e-8)
        assert p[0] == pytest.approx(max(0.0, hessian), rel=1e-8)


def _v19_fixture():
    result = fit(START, PRIORS, NOISY, noise=SCALES, method="lbfgs")
    return result.values


def test_the_scale_block_is_the_fisher_information_and_decoupled():
    values = _v19_fixture()
    problem, z, jac, _ = _curvature(START, PRIORS, NOISY, values, SCALES)
    keys = problem.coordinate_keys[-2:]
    assert set(keys) == {"noise.vis_scale", "noise.phi_scale"}
    jtj = jac.T @ jac
    n_vis = onp.size(NOISY.vis)
    n_phi = NOISY.cp_noise.size  # the whitened closure phases
    counts = {"noise.vis_scale": n_vis, "noise.phi_scale": n_phi}
    expected = []
    with run_in("float64"):
        for site in keys:
            count = counts[site]
            bijection = problem.bijection(problem.noise[site][0])
            dln = float(jax.grad(lambda v: np.log(bijection(v)))(z[site]))
            expected.append(2 * count * dln**2)
    onp.testing.assert_allclose(
        jtj[-2:, -2:], onp.diag(expected), rtol=1e-6, atol=1e-9
    )
    # The errors do not move the mean, nor the parameters the errors.
    onp.testing.assert_allclose(jtj[:-2, -2:], 0.0, atol=1e-9)


@pytest.mark.parametrize("flat", [False, True])
@pytest.mark.slow
def test_the_v19_matrix_whitens_the_posterior(flat):
    values = _v19_fixture()
    sites, eigenvalues, mass = _hessian_check(
        START, PRIORS, NOISY, values, SCALES, flat
    )
    assert set(sites) == {
        "dra",
        "ddec",
        "flux",
        "noise.vis_scale",
        "noise.phi_scale",
    }
    # Measured on this fixture (noise 1.7x the errors, 15 independent closure
    # phases): the two scales give 1.0000 and 1.0000, ln(flux) 0.998, and the
    # dra and ddec directions 0.666 and 1.329, the Σ r∇²r that F drops. The
    # plan's bound [0.8, 1.25] was for e8's cleaner data; this is the
    # measured one with margin.
    assert eigenvalues.min() > 0.6 and eigenvalues.max() < 1.4, eigenvalues
    assert onp.sum(onp.abs(eigenvalues - 1.0) < 0.01) >= 2  # the scales
    assert mass.condition < 1e4


def _gains_data():
    gains = oidata.with_gains(telescope=0.03, baseline=0.02)
    return gains.with_model(TRUTH, key=jax.random.PRNGKey(5))


@pytest.mark.slow
def test_gain_data_give_a_positive_definite_matrix_close_to_the_hessian():
    data = _gains_data()
    values = fit(START, PRIORS, data).values
    sites, eigenvalues, mass = _hessian_check(START, PRIORS, data, values)
    assert set(sites) == {"dra", "ddec", "flux"}
    assert eigenvalues.min() > 0.5 and eigenvalues.max() < 2.0, eigenvalues


@pytest.mark.slow
def test_fitted_gain_widths_and_scales_stay_positive_definite():
    # The gain width's curvature is a heuristic (a PSD one) rather than the
    # Fisher information: the matrix is a valid preconditioner but looser.
    data = _gains_data()
    noise = {
        "vis_scale": dist.LogUniform(0.1, 10.0),
        "vis_gain_baseline": dist.LogUniform(1e-3, 0.3),
    }
    values = fit(START, PRIORS, data, noise=noise).values
    _, eigenvalues, mass = _hessian_check(START, PRIORS, data, values, noise)
    assert onp.all(onp.isfinite(eigenvalues)) and eigenvalues.min() > 0
    assert eigenvalues.min() > GAIN_WIDTH_BOUNDS[0]
    assert eigenvalues.max() < GAIN_WIDTH_BOUNDS[1], eigenvalues


# Measured: eig(F⁻¹H) = 0.0058, 0.996, 1.000, 1.011, 1.44. The three parameter
# directions are exact (0.996 to 1.011, as without the width) and the gain width
# is 1.44; the scale's F is 2 n_vis while its true curvature is swamped by the
# gains (the plan expected [0.3, 3]). The cost is efficiency in that one
# coordinate, not correctness.
GAIN_WIDTH_BOUNDS = (0.003, 3.0)


@pytest.mark.slow
def test_added_errors_couple_to_the_parameters():
    noise = {"vis_error_rel": dist.LogUniform(1e-4, 0.3)}
    values = fit(START, PRIORS, NOISY, noise=noise).values
    problem, _, jac, _ = _curvature(START, PRIORS, NOISY, values, noise)
    jtj = jac.T @ jac
    assert problem.coordinate_keys[-1] == "noise.vis_error_rel"
    assert onp.max(onp.abs(jtj[:-1, -1])) > 0.0
    assert onp.all(onp.linalg.eigvalsh(jtj) > -1e-9 * onp.max(jtj))
    _, eigenvalues, _ = _hessian_check(START, PRIORS, NOISY, values, noise)
    # Measured 0.755, 0.9996, 1.0003, 1.298.
    assert eigenvalues.min() > 0.6 and eigenvalues.max() < 1.5, eigenvalues


def _latent_sites(model):
    trace = numpyro.handlers.trace(numpyro.handlers.seed(model, 0)).get_trace()
    return {
        name: int(onp.size(site["value"]))
        for name, site in trace.items()
        if site["type"] == "sample" and not site.get("is_observed")
    }


def _check_sites(model, priors, data, values, noise, flat, expected):
    from virgil.fitting import gauss_newton_mass
    from virgil.likelihood import chain_init_params

    mass = gauss_newton_mass(
        model, priors, data, values, noise=noise, flat_coordinates=flat
    )
    (sites,) = mass["dense_mass"]
    posterior = numpyro_model(
        model, priors, data, noise=noise, flat_coordinates=flat
    )
    latent = _latent_sites(posterior)
    assert len(sites) == len(expected) and set(sites) == set(expected)
    assert set(sites) == set(latent)
    covariance = mass["inverse_mass_matrix"][sites]
    assert covariance.shape == (sum(latent.values()),) * 2
    chain_init_params(posterior, [values], mass=mass)  # raises on a mismatch
    return covariance


@pytest.mark.parametrize("flat", [False, True])
def test_the_mass_sites_are_the_numpyro_latent_sites(flat):
    values = {"dra": 150.0, "ddec": -80.0, "flux": 0.02}
    # Fitted scales of one dataset.
    _check_sites(
        START,
        PRIORS,
        NOISY,
        values | {"noise.vis_scale": 1.5, "noise.phi_scale": 1.2},
        SCALES,
        flat,
        ("dra", "ddec", "flux", "noise.vis_scale", "noise.phi_scale"),
    )
    # List-form noise: sites are indexed by dataset; an empty dict has none.
    noise = [
        {"vis_scale": dist.LogUniform(0.1, 10.0)},
        {"phi_scale": dist.LogUniform(0.1, 10.0)},
    ]
    _check_sites(
        START,
        PRIORS,
        [NOISY, NOISY],
        values | {"noise[0].vis_scale": 1.5, "noise[1].phi_scale": 1.2},
        noise,
        flat,
        ("dra", "ddec", "flux", "noise[0].vis_scale", "noise[1].phi_scale"),
    )
    # A tied term is a function of the parameters: no row, no site.
    tied = [
        {"vis_scale": lambda v: v["s"]},
        {"vis_scale": lambda v: 2.0 * v["s"]},
    ]
    priors = {**PRIORS, "s": dist.LogUniform(0.5, 5.0)}
    _check_sites(
        _binary_fn,
        priors,
        [NOISY, NOISY],
        values | {"s": 1.5},
        tied,
        flat,
        ("dra", "ddec", "flux", "s"),
    )


@pytest.mark.parametrize("flat", [False, True])
def test_an_angle_vector_with_noise_keeps_its_site_name(flat):
    from virgil.angles import AngleVector
    from virgil.models import BinaryModelAngular

    truth = BinaryModelAngular(150.0, 5.0, 0.05)
    # Noisy data: noiseless ones fit a scale to its lower bound.
    data = oidata.with_model(truth, key=jax.random.PRNGKey(2))
    template = BinaryModelAngular(150.0, 355.0, 0.05)
    priors = {"pa": AngleVector()}
    noise = {"vis_scale": dist.LogUniform(0.1, 10.0)}
    result = fit(template, priors, data, noise=noise)
    covariance = _check_sites(
        template,
        priors,
        data,
        result.values,
        noise,
        flat,
        ("pa_vec", "noise.vis_scale"),
    )
    assert covariance.shape == (3, 3)
    assert onp.all(onp.linalg.eigvalsh(covariance) > 0)


@pytest.mark.skipif(
    jax.config.jax_enable_x64,
    reason="the dtype is specific to float32; x64 is on globally",
)
def test_curvature_rows_are_finite_in_float32_with_gains_and_offsets():
    data = oidata.with_gains(
        telescope=0.03, baseline=0.02
    ).with_closure_offsets(baseline=0.02)
    noise = {
        "vis_scale": dist.LogUniform(0.1, 10.0),
        "vis_gain_baseline": dist.LogUniform(1e-3, 0.3),
        "phi_offset_baseline": dist.LogUniform(1e-3, 0.3),
    }
    problem = _Objective(START, PRIORS, data, noise=noise, flat=False)
    z = problem.init({"dra": 150.0, "ddec": -80.0, "flux": 0.02})
    assert set(problem.coordinate_keys[-3:]) == {f"noise.{k}" for k in noise}
    rows = problem.curvature_rows(z)
    assert rows.dtype == np.float32 and bool(np.all(np.isfinite(rows)))
    assert bool(np.all(np.isfinite(problem.curvature_diag(z, True))))


def test_a_nuts_chain_runs_with_the_v19_mass():
    from numpyro.infer import MCMC, NUTS, init_to_value

    from virgil.fitting import gauss_newton_mass

    values = _v19_fixture()
    mass = gauss_newton_mass(START, PRIORS, NOISY, values, noise=SCALES)
    kernel = NUTS(
        numpyro_model(START, PRIORS, NOISY, noise=SCALES),
        init_strategy=init_to_value(values=values),
        **mass,
    )
    mcmc = MCMC(kernel, num_warmup=30, num_samples=30, progress_bar=False)
    mcmc.run(jax.random.PRNGKey(0), extra_fields=("diverging", "num_steps"))
    extra = mcmc.get_extra_fields()
    assert int(extra["diverging"].sum()) == 0
    samples = mcmc.get_samples()
    assert set(samples) == set(PRIORS) | {"noise.vis_scale", "noise.phi_scale"}
    assert all(onp.all(onp.isfinite(onp.asarray(v))) for v in samples.values())


def test_a_prior_without_a_diagonal_curvature_raises():
    # A Dirichlet's curvature is not diagonal: the matrix would leave it
    # out, so the curvature refuses rather than mislead.
    from virgil.fitting import NoLeastSquaresForm

    priors = dict(PRIORS) | {"flux": dist.Dirichlet(np.ones(2))}
    problem = _Objective(START, priors, NOISY, flat=False)
    with pytest.raises(NoLeastSquaresForm, match="not diagonal"):
        problem._no_prior_curvature(priors["flux"], True)


def test_lm_compiles_one_kernel_whatever_the_step_bound(monkeypatch):
    # The step bound is traced, so the shorter last chunk of a chunked fit
    # (5 steps in chunks of 2), a different chunk size and a different
    # max_steps take no compile of their own.
    import virgil.fitting

    far = BinaryModelCartesian(100.0, -20.0, 0.05)
    options = {"method": "lm", "time_limit": 1e6}
    monkeypatch.setattr(virgil.fitting, "_LM_CHUNK", 2)
    with pytest.warns(RuntimeWarning, match="step limit"):
        five = fit(far, PRIORS, DATA, max_steps=5, **options)
    assert five.info["steps"] == 5 and five.info["stop"] == "limit"
    monkeypatch.setattr(virgil.fitting, "_LM_CHUNK", 3)
    with count_compiles() as compiles:
        with pytest.warns(RuntimeWarning, match="step limit"):
            four = fit(far, PRIORS, DATA, max_steps=4, **options)
        with pytest.warns(RuntimeWarning, match="step limit"):
            three = fit(far, PRIORS, DATA, max_steps=3, method="lm")
    assert (four.info["steps"], three.info["steps"]) == (4, 3)
    assert compiles == []


def _full_gradient_summary(problem, z, scale):
    """The loss, chi-squared and gradient norm, the way ``fit`` once reported them:
    by differentiating the whole model again at the result.
    """
    values = problem.constrain(z)
    residuals = problem.data_residuals(problem.build(z), values)
    residuals += [term(values) for term in problem.likelihoods]
    gradient = jax.grad(lambda z: problem.loss(z) / scale)(z)
    norm = max(float(np.max(np.abs(g))) for g in jax.tree.leaves(gradient))
    return (
        float(problem.loss(z)),
        [float(np.sum(r**2)) for r in residuals],
        norm,
    )


@pytest.mark.parametrize(
    "method, options",
    [
        ("lbfgs", {"max_steps": 6}),  # stopped, so the gradient is large
        ("lbfgs", {}),  # converged
        ("lbfgs", {"max_steps": 10, "round": 3}),  # restarted rounds
        ("lbfgs", {"progress": True}),  # chunked
        ("lm", {"max_steps": 2}),
        ("adam", {"max_steps": 5}),
    ],
)
def test_summary_matches_a_full_gradient_recomputation(
    monkeypatch, method, options
):
    # The reported loss, chi-squared and gradient norm come from a forward
    # pass and, for L-BFGS, the optimizer's state, with no second gradient of
    # the whole model. They must equal the gradient taken afresh at the result.
    import virgil.fitting

    options = dict(options)
    monkeypatch.setattr(
        virgil.fitting, "_LBFGS_ROUND", options.pop("round", 2000)
    )
    monkeypatch.setattr(virgil.fitting, "_LBFGS_CHUNK", 3)
    reports = []
    if "progress" in options:
        options["progress"] = reports.append
    seen = []
    summary = virgil.fitting._summary

    def spy(problem, z, scale, grad_norm=None):
        seen.append((problem, z, scale, grad_norm))
        return summary(problem, z, scale, grad_norm)

    monkeypatch.setattr(virgil.fitting, "_summary", spy)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = fit(START, PRIORS, DATA, method=method, **options)
    problem, z, scale, grad_norm = seen[-1]
    # Only L-BFGS has a gradient to hand.
    assert (grad_norm is not None) == (method == "lbfgs")
    with run_in("float64"):
        loss, chi2, norm = _full_gradient_summary(problem, z, scale)
    assert result.info["loss"] == pytest.approx(loss, rel=1e-9)
    assert result.info["chi2"] == pytest.approx(chi2, rel=1e-9)
    # Near convergence the norm is ~1e-7, where the line search's gradient and a
    # fresh one differ in the last digits of float64 on some platforms.
    assert result.info["grad_norm"] == pytest.approx(norm, rel=1e-4, abs=1e-10)
    if reports:
        # The progress report's gradient norm is the same state's, not a
        # step behind.
        assert reports[-1]["grad_norm"] == pytest.approx(
            result.info["grad_norm"]
        )


def test_final_gradient_is_the_carrys_before_any_step():
    # Before a step the optimizer state holds no gradient yet (the line
    # search has not run), so the carry's own, at the start, is the answer;
    # after one, the state's, at the new point.
    from virgil.fitting import _final_gradient

    state = {"grad": np.array([3.0, -4.0])}
    at_start = (0, None, state, 1.0, np.asarray(0.5), True)
    assert float(_final_gradient(at_start)) == 0.5
    after = (2, None, state, 1.0, np.asarray(0.5), True)
    assert float(_final_gradient(after)) == 4.0


def test_lm_rejects_a_step_that_raises_the_loss():
    # An ill-conditioned damped solve can return a step whose predicted
    # reduction is positive. optimistix then accepts any loss increase of
    # at most 1% of that prediction (here 4.5 against 0.5 at the start),
    # and a fit started beside the truth ran off to a prior bound.
    solver = _GradientStoppedLM(1e-4, 100, lx.QR())
    y, evaluated_at = np.zeros(2), np.array([2.0, 0.0])
    now = optx.FunctionInfo.ResidualJac(
        np.array([1.0, 0.0]), lx.MatrixLinearOperator(1000.0 * np.eye(2))
    )
    worse = optx.FunctionInfo.ResidualJac(np.array([3.0, 0.0]), now.jac)
    better = optx.FunctionInfo.ResidualJac(np.array([0.5, 0.0]), now.jac)
    state = solver.search.init(y, now)

    def accepts(search, evaluated):
        out = search.step(
            np.array(False), y, evaluated_at, now, evaluated, state
        )
        return bool(out[1]), float(out[0])

    assert accepts(optx.ClassicalTrustRegion(), worse)[0]  # the premise
    assert accepts(solver.search, worse) == (False, 0.25)
    assert accepts(solver.search, better)[0]


def test_a_fit_near_the_truth_does_not_run_away_from_it():
    # Noise-free limb-darkened star and companion, started 5% off in the
    # star and 0.5 mas off in the companion. LM's first steps are large
    # (the data are exact, so chi2 is ~1e6 at the start), and before it
    # rejected steps that raise the loss, about half of such fits ended at
    # the prior bound (the companion's flux near 0, its offset at the
    # edge). The bad steps were intermittent and seen on macOS, so on Linux
    # this is a regression guard and the unit test above pins the rule.
    truth_values = {
        "star.diam": 6.0,
        "star.q1": 0.36,
        "star.q2": 0.29,
        "comp.flux": 0.03,
        "comp.dra": -14.0,
        "comp.ddec": 9.0,
    }
    start_values = {
        "star.diam": 5.7,
        "star.q1": 0.5,
        "star.q2": 0.5,
        "comp.flux": 0.022,
        "comp.dra": -13.5,
        "comp.ddec": 8.7,
    }
    priors = {
        "star.diam": dist.Uniform(0.0, 18.0),
        "star.q1": dist.Uniform(0.0, 1.0),
        "star.q2": dist.Uniform(0.0, 1.0),
        "comp.flux": dist.Uniform(0.0, 1.0),
        "comp.dra": dist.Uniform(-166.0, 166.0),
        "comp.ddec": dist.Uniform(-166.0, 166.0),
    }
    with jax.enable_x64(True):
        truth = System(
            star=QuadraticLimbDarkenedDisk(6.0, q1=0.36, q2=0.29),
            comp=PointSource(0.03, -14.0, 9.0),
        )
        data = vlti_oidata(
            hour_angles_h=onp.linspace(-3.0, 3.0, 7),
            wavelengths_m=onp.linspace(1.5e-6, 2.4e-6, 6),
            sigma_v2=0.01,
            sigma_cp_deg=0.5,
        ).with_model(truth)
        start = truth
        for path, value in start_values.items():
            start = start.set(path, value)
        problem = _Objective(start, priors, data, (), None, ())
        start_loss = float(problem.loss(problem.init()))
        result = fit(start, priors, data)
    assert result.info["loss"] <= start_loss
    assert result.info["chi2_red"] < 1.0
    assert not result.info["at_bound"]
    for path, value in truth_values.items():
        assert result.values[path] == pytest.approx(value, rel=1e-2)


def _tree(n, dtype):
    return {"x": jax.ShapeDtypeStruct((n,), dtype)}


@pytest.mark.parametrize(
    "n, dtype, expected",
    [
        (0, "float64", 100),  # nothing to store: no division by zero
        (1, "float64", 100),  # tiny problem: the full history
        (1681, "float64", 100),  # the 41² image
        (
            671_088,
            "float64",
            100,
        ),  # the largest float64 problem with the full history
        (671_089, "float64", 99),
        (1_000_000, "float64", 67),  # 10⁶ float64 pixels: 1 GiB of history
        (1_342_177, "float32", 100),  # float32 vectors are half the size
        (10**9, "float64", 5),  # never below 5 steps
    ],
)
def test_the_lbfgs_history_is_100_unless_it_would_pass_one_gibibyte(
    n, dtype, expected
):
    from virgil.fitting import _lbfgs_memory

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert _lbfgs_memory(None, _tree(n, dtype)) == expected
    assert bool(caught) == (expected < 100)
    assert _lbfgs_memory(7, _tree(n, dtype)) == 7  # explicit: as given, no cut
    assert _lbfgs_memory(500, _tree(n, dtype)) == 500


def test_the_lbfgs_history_is_sized_by_the_widest_leaf():
    from virgil.fitting import _lbfgs_memory

    narrow = {"a": jax.ShapeDtypeStruct((1_342_177,), "float32")}
    mixed = narrow | {"b": jax.ShapeDtypeStruct((1,), "float64")}
    assert _lbfgs_memory(None, narrow) == 100
    with pytest.warns(UserWarning, match="pass lbfgs_memory"):
        assert _lbfgs_memory(None, mixed) < 100


def test_fit_passes_the_resolved_lbfgs_memory_to_the_optimizer(monkeypatch):
    from virgil import fitting

    jax.clear_caches()  # the optimizer is built at trace time
    seen = []
    real = fitting._capped_lbfgs
    monkeypatch.setattr(
        fitting,
        "_capped_lbfgs",
        lambda step, memory, *rest: seen.append(memory)
        or real(step, memory, *rest),
    )
    fit(START, PRIORS, DATA, method="lbfgs", max_steps=5)
    fit(START, PRIORS, DATA, method="lbfgs", lbfgs_memory=7, max_steps=5)
    assert seen[0] == 100
    assert seen[-1] == 7


def test_post_step_none_and_identity_are_bit_identical():
    plain, same = _adam(), _adam(post_step=lambda z: z)
    for path, value in plain.values.items():
        assert same.values[path] == value
    assert (same.info["loss_history"] == plain.info["loss_history"]).all()
    assert (plain.info["post_step"], same.info["post_step"]) == (False, True)


def test_post_step_clip_holds_at_every_step():
    free = _adam(max_steps=40).values["dra"]
    assert free > START.dra  # the data pull dra up, so the clip binds
    top = _Objective(START, PRIORS, (DATA,)).init(None)["dra"]
    clip = lambda z: {**z, "dra": np.minimum(z["dra"], top)}  # noqa: E731
    for steps in (1, 5, 40):
        got = _adam(max_steps=steps, post_step=clip).values["dra"]
        assert got <= START.dra * (1 + 1e-5)
    assert got > START.dra * (1 - 1e-5)  # held at the clip, not below


def test_post_step_needs_plain_adam():
    for kw in ({"method": "lbfgs"}, {"method": "lm"}, {}):
        with pytest.raises(ValueError, match="post_step"):
            fit(START, PRIORS, DATA, post_step=lambda z: z, **kw)
    with pytest.raises(ValueError, match="post_step"):
        _adam(post_step=lambda z: z, precondition="diag")


def test_regularizers_raise_for_a_model_per_dataset():
    def two(dra, ddec, flux, **_):
        model = BinaryModelCartesian(dra, ddec, flux)
        return [model, model]

    init = {"dra": 55.0, "ddec": -35.0, "flux": 0.03}
    with pytest.raises(ValueError, match="one model per dataset"):
        fit(
            two,
            PRIORS,
            [DATA, DATA],
            init=init,
            regularizers=[TV(1.0, path="env", beam_mas=12.0)],
        )
    fit(two, PRIORS, [DATA, DATA], init=init, max_steps=1)  # none: fine


def test_numpyro_model_raises_for_a_model_per_dataset_with_regularizers():
    def two(dra, ddec, flux, **_):
        model = BinaryModelCartesian(dra, ddec, flux)
        return [model, model]

    model = numpyro_model(
        two, PRIORS, [DATA, DATA], [Centroid(10.0, path="env")]
    )
    with pytest.raises(ValueError, match="one model per dataset"):
        numpyro.handlers.trace(
            numpyro.handlers.seed(model, jax.random.PRNGKey(0))
        ).get_trace()
