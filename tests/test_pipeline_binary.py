"""BinaryPipeline end to end on tiny synthetic data: API contract, outputs,
round trip, resume, quicklook and the CLI.

Grids and chains are deliberately tiny: these tests check the plumbing and
the contract, not the science.
"""

import json
import math
import warnings
import shutil

import jax
import numpy as onp
import pytest

pytest.importorskip("h5py")
pytest.importorskip("nbclient")

import virgil  # noqa: E402
from tests._shared import executed_quicklook  # noqa: E402
from virgil import BinaryModelCartesian, OIData  # noqa: E402
from virgil.pipeline import (  # noqa: E402
    SCHEMA,
    BinaryPipeline,
    Check,
    ConfigMismatchError,
    Result,
    load,
)
from virgil.pipeline import cli  # noqa: E402

# The module's tests share expensive module-scoped fixtures: with
# ``pytest -n N --dist loadgroup`` they run on one worker, which builds each
# fixture once instead of once per worker.
pytestmark = pytest.mark.xdist_group("pipeline_binary")

TINY = {
    "max_sep_mas": 300.0,
    "max_grid": 13,
    # n_null=15 floors the false-alarm probability at 1/16 (norm.isf = 1.534σ), so detection_sigma must be below it.
    "n_null": 15,
    "detection_sigma": 1.5,
    "n_flux": 6,
    "flux_range": [1e-4, 0.2],
    "num_warmup": 30,
    "num_samples": 30,
    "num_chains": 2,
}
TRUTH = BinaryModelCartesian(120.0, 80.0, 0.01)


@pytest.fixture(scope="module")
def data():
    base = OIData("data/NuHor_F480M.oifits")
    return base.with_model(TRUTH, key=jax.random.PRNGKey(3))


@pytest.fixture(scope="module")
def full_run(data, tmp_path_factory):
    out = tmp_path_factory.mktemp("binary") / "run"
    # The notebook is executed by test_quicklook_notebook alone.
    pipeline = BinaryPipeline(
        data, output=out, quicklook_execute=False, **TINY
    )
    return pipeline, pipeline.run()


# --- the stable contract ----------------------------------------------------

API = {
    "BinaryPipeline": "(data, model=None, *, output='run', **settings)",
    "BinaryPipeline.run": "(self, through=None, *, resume=True)",
    "BinaryPipeline.from_oifits": "(paths, model=None, *, output='run', **settings)",
    "BinaryPipeline.from_config": "(path, model=None, **overrides)",
    "BinaryPipeline.defaults": "()",
    "load": "(path)",
    "Result": "(path)",
    "Result.data": "(self)",
    # name= (additive, with StarPipeline) picks one of several fitted models.
    "Result.model": "(self, name=None)",
    "Result.model_values": "(self, name=None)",
    "Result.samples": "(self, group_by_chain=True, name=None)",
    "Result.sample_stats": "(self, name=None)",
    "Result.grid": "(self)",
    "Result.fit_result": "(self, name=None)",
    "Result.describe": "(self)",
    "Check": "(name, status, value, threshold, message)",
}
STAGES = (
    "load",
    "overview",
    "search",
    "limits",
    "fit",
    "posterior",
    "quicklook",
)
DEFAULTS = {
    "sigma": 3.0,
    "detection_sigma": 3.0,
    "n_null": 1000,
    "max_sep_mas": None,
    "fov_fraction": 1.0,
    "grid_step_mas": None,
    "max_grid": 101,
    "flux_range": [1e-5, 1.0],
    "n_flux": 40,
    "wavel_range": None,
    "error_floor": None,
    "error_scale": "quoted",
    "num_warmup": 1000,
    "num_samples": 1000,
    "num_chains": 4,
    "chain_method": "vectorized",
    "seed": 0,
    "batch_size": None,
    "limit_bins": 20,
    "quicklook_execute": True,
}
SUMMARY_KEYS = {
    "schema": None,
    "pipeline": None,
    "stability": None,
    "status": None,
    "worst_check": None,
    "checks": None,
    "warnings": None,
    "data": {
        "n_vis",
        "n_phi",
        "n_independent",
        "closure_phases",
        "wavel_min_m",
        "wavel_max_m",
        "baseline_min_m",
        "baseline_max_m",
        "resolution_mas",
        "fov_mas",
    },
    "chi2": {
        "n_independent",
        "null_reduced",
        "companion_reduced",
        "error_scale",
        "error_scales",
    },
    "search": {
        "delta_chi2",
        "log_bayes_factor",
        "max_snr",
        "local_nsigma",
        "global_nsigma",
        "global_nsigma_method",
        "false_alarm_probability",
        "n_null",
        "n_null_above",
        "global_nsigma_is_lower_bound",
        "dra_mas",
        "ddec_mas",
        "flux",
        "max_sep_mas",
        "alias_cap_mas",
    },
    "limits": {
        "sigma",
        "method",
        "r_mas",
        "median_flux",
        "median_delta_mag",
        "deepest_delta_mag",
    },
    "fit": {"params", "converged", "chi2_reduced", "at_bound"},
    "companion": {
        "source",
        "sep_mas",
        "sep_err_mas",
        "pa_deg",
        "pa_err_deg",
        "dra_mas",
        "ddec_mas",
        "flux",
        "flux_err",
        "contrast",
        "delta_mag",
    },
    "posterior": {
        "params",
        "noise",
        "num_chains",
        "num_samples",
        "num_warmup",
        "r_hat_max",
        "ess_bulk_min",
        "divergence_fraction",
    },
}
CHECKS = [
    "chi2",
    "detection",
    "grid_edge",
    "residual_normality",
    "field_of_view",
    "prior_bound",
    "r_hat",
    "ess",
    "divergences",
]
WARNING_CHECKS = {
    "convergence",
    "flux_axis_resolution",
    "limits_clipped",
    "stage_warnings",
}


def test_summary_keys_snapshot(full_run):
    _, res = full_run
    keys = {
        k: (set(v) if isinstance(v, dict) else None)
        for k, v in res.summary.items()
    }
    assert keys == SUMMARY_KEYS
    # Checks mapped from stage warnings come after the fixed ones, and only
    # when the warning occurred.
    assert [
        c.name for c in res.checks if c.name not in WARNING_CHECKS
    ] == CHECKS
    assert all(isinstance(c, Check) for c in res.checks)


def test_settings_are_validated(data):
    with pytest.raises(ValueError, match="sigma"):
        BinaryPipeline(data, sigma=-1.0)
    with pytest.raises(ValueError, match="error_scale"):
        BinaryPipeline(data, error_scale="rescaled")
    with pytest.raises(TypeError, match="unknown settings"):
        BinaryPipeline(data, num_sample=10)
    with pytest.raises(TypeError, match="BinaryModelAngular"):
        BinaryPipeline(data, virgil.PointSource())


# --- outputs and the round trip ---------------------------------------------


def test_run_folder_layout(full_run):
    pipeline, res = full_run
    out = pipeline.output
    run = json.loads((out / "run.json").read_text())
    assert run["schema"] == SCHEMA and run["status"] == "complete"
    assert run["stability"] == "provisional" and run["error"] is None
    assert list(run["stages"]) == list(STAGES)
    assert run["provenance"]["virgil"] == virgil.__version__
    assert set(run["config"]) == set(DEFAULTS)
    for name in (
        "summary.json",
        "grids.h5",
        "samples.h5",
        "quicklook.ipynb",
        "data/processed.oifits",
        "models/best/manifest.json",
        "plots/search_delta_chi2.png",
        "plots/limits_contrast_curve.png",
        "plots/overview_uv.png",
        "stages/search/report.json",
    ):
        assert (out / name).exists(), name
    assert not list(out.rglob("*.pkl")) and not list(out.rglob("*.pickle"))


def test_round_trip(full_run, data):
    pipeline, _ = full_run
    res = load(pipeline.output)  # as in a fresh session
    assert isinstance(res, Result) and res.status == "complete"

    back = res.data()
    onp.testing.assert_allclose(back.vis, data.vis, rtol=1e-6)
    onp.testing.assert_allclose(back.phi, data.phi, rtol=1e-6, atol=1e-12)

    model = res.model()
    assert type(model) is type(pipeline.model)
    fitted = res.summary["fit"]["params"]
    for name, value in fitted.items():
        onp.testing.assert_allclose(
            float(getattr(model, name)), value, rtol=1e-6
        )

    fit = res.fit_result()
    assert fit.values.keys() >= set(fitted)
    assert fit.info["converged"] == res.summary["fit"]["converged"]

    samples = res.samples()
    assert samples["sep"].shape == (TINY["num_chains"], TINY["num_samples"])
    flat = res.samples(group_by_chain=False)
    assert flat["flux"].shape == (TINY["num_chains"] * TINY["num_samples"],)
    assert res.sample_stats()["diverging"].shape == samples["sep"].shape
    median = res.summary["posterior"]["params"]["flux"]["median"]
    onp.testing.assert_allclose(onp.median(flat["flux"]), median, rtol=1e-5)

    grid = res.grid()
    assert list(grid["axes"]) == ["dra", "ddec", "flux"]
    shape = (len(grid["axes"]["dra"]), len(grid["axes"]["ddec"]))
    for name in ("delta_chi2", "snr", "flux_best", "flux_sigma", "limit_flux"):
        assert grid[name].shape == shape, name


def test_quoted_errors_lead_and_companion_found(full_run):
    _, res = full_run
    s = res.summary
    assert (
        s["chi2"]["error_scale"] == "quoted"
        and s["chi2"]["error_scales"] is None
    )
    assert s["chi2"]["null_reduced"] > s["chi2"]["companion_reduced"]
    sep = onp.hypot(TRUTH.dra, TRUTH.ddec)
    assert abs(s["companion"]["sep_mas"] - sep) < 5.0
    # Orientation: PA is East of North, dra = sep sin(PA), ddec = sep cos(PA).
    pa = float(onp.degrees(onp.arctan2(TRUTH.dra, TRUTH.ddec)))
    assert abs(s["companion"]["pa_deg"] - pa) < 3.0
    assert abs(s["companion"]["dra_mas"] - TRUTH.dra) < 5.0
    assert abs(s["companion"]["ddec_mas"] - TRUTH.ddec) < 5.0
    assert abs(s["search"]["dra_mas"] - TRUTH.dra) < 40.0
    assert abs(s["search"]["ddec_mas"] - TRUTH.ddec) < 40.0
    assert abs(s["companion"]["flux"] - 0.01) < 2e-3


def test_quicklook_notebook(full_run):
    pipeline, res = full_run
    # overview, search + limits, correlation, corner
    cells, _ = executed_quicklook(pipeline.output, images=4)
    assert "| `r_hat` |" in "".join(cells[-1]["source"])
    # The notebook's contrast curve uses the pipeline's annuli and range.
    curve = [
        "".join(c["source"])
        for c in cells
        if c["cell_type"] == "code"
        and "plot_contrast_curve" in "".join(c["source"])
    ]
    run = json.loads((pipeline.output / "run.json").read_text())
    max_sep = res.summary["search"]["max_sep_mas"]
    assert len(curve) == 1
    assert f"r_max={max_sep!r}" in curve[0]
    assert f"bins={run['config']['limit_bins']!r}" in curve[0]


# --- resume -------------------------------------------------------------------


def _started(out):
    run = json.loads((out / "run.json").read_text())
    return {k: v["started"] for k, v in run["stages"].items()}, run


def test_resume_skips_completed_stages(data, tmp_path, monkeypatch):
    out = tmp_path / "run"
    settings = dict(TINY, quicklook_execute=False)
    res = BinaryPipeline(data, output=out, **settings).run(through="search")
    assert res.status == "partial" and "limits" not in res.summary
    before, _ = _started(out)

    # Simulate a newer virgil with an extra default-valued setting: the
    # stored config lacks it, and resuming must still work.
    run = json.loads((out / "run.json").read_text())
    del run["config"]["limit_bins"]
    run["stages"]["load"]["provenance"]["virgil"] = "0.0.old"
    (out / "run.json").write_text(json.dumps(run))

    ran = []
    import virgil.pipeline.binary as binary

    for name in ("_load_report", "_overview", "_search"):
        monkeypatch.setattr(binary, name, lambda p, n=name: ran.append(n))
    res = BinaryPipeline(data, output=out, limit_bins=20, **settings).run(
        through="limits"
    )
    assert ran == []  # nothing before limits was recomputed
    after, run = _started(out)
    assert {k: after[k] for k in before} == before
    assert run["status"] == "partial" and "limits" in res.summary
    # A resumed run keeps the versions that computed the earlier stages.
    assert run["stages"]["load"]["provenance"]["virgil"] == "0.0.old"
    assert run["stages"]["limits"]["provenance"]["virgil"] == (
        virgil.__version__
    )
    assert run["provenance"]["virgil"] == virgil.__version__

    with pytest.raises(ConfigMismatchError, match="sigma"):
        BinaryPipeline(data, output=out, **dict(settings, sigma=5.0)).run()
    shifted = data.with_error_floor(absolute={"vis": 0.5})
    with pytest.raises(ConfigMismatchError, match="data differ"):
        BinaryPipeline(shifted, output=out, **settings).run()


def test_fresh_run_and_failure_record(data, tmp_path, monkeypatch):
    out = tmp_path / "run"
    settings = dict(TINY, quicklook_execute=False)
    BinaryPipeline(data, output=out, **settings).run(through="load")
    import virgil.pipeline.binary as binary

    def boom(p):
        warnings.warn("grid went NaN first", RuntimeWarning, stacklevel=1)
        raise RuntimeError("overview exploded")

    monkeypatch.setattr(binary, "_overview", boom)
    with pytest.raises(RuntimeError, match="exploded"):
        BinaryPipeline(data, output=out, **dict(settings, sigma=5.0)).run(
            through="overview", resume=False
        )
    run = json.loads((out / "run.json").read_text())
    assert run["status"] == "failed" and run["config"]["sigma"] == 5.0
    assert run["error"]["stage"] == "overview"
    assert run["error"]["type"] == "RuntimeError"
    assert [w["message"] for w in run["error"]["warnings"]] == [
        "grid went NaN first"
    ]
    assert run["stages"]["overview"]["status"] == "failed"
    assert run["stages"]["load"]["status"] == "complete"


def test_rerun_discards_the_outputs_of_later_stages(full_run, tmp_path):
    """Re-running a stage deletes what the later stages wrote; Result then
    refuses them, also if a stale file is put back."""
    pipeline, _ = full_run
    out = tmp_path / "rerun"
    shutil.copytree(pipeline.output, out)
    later = (
        "grids.h5",
        "samples.h5",
        "quicklook.ipynb",
        "models/best/values.npz",
        "plots/limits_contrast_curve.png",
        "plots/posterior_corner.png",
    )
    assert all((out / name).exists() for name in later)
    (out / "plots" / "overview_uv.png").unlink()  # an output of 'overview'

    again = BinaryPipeline(
        pipeline.data, output=out, quicklook_execute=False, **TINY
    )
    res = again.run(through="overview")
    assert res.status == "partial"
    assert not any((out / name).exists() for name in later)
    assert "search" not in res.summary and "fit" not in res.summary
    run = json.loads((out / "run.json").read_text())
    assert list(run["stages"]) == ["load", "overview"]
    for call in (res.grid, res.samples, res.model, res.fit_result):
        with pytest.raises(FileNotFoundError, match="stage"):
            call()

    # Even a stale file left behind is not served: the stage never completed.
    shutil.copy(pipeline.output / "grids.h5", out / "grids.h5")
    with pytest.raises(FileNotFoundError, match="earlier run"):
        load(out).grid()


def test_grid_leaves_out_the_limits_of_an_unfinished_limits_stage(
    full_run, tmp_path
):
    pipeline, _ = full_run
    copy = tmp_path / "copy"
    shutil.copytree(pipeline.output, copy)
    run = json.loads((copy / "run.json").read_text())
    run["stages"]["limits"]["status"] = "failed"
    (copy / "run.json").write_text(json.dumps(run))
    grid = load(copy).grid()
    assert "snr" in grid and "limit_flux" not in grid


def test_null_nsigma_bounds_and_nan(data):
    from scipy.stats import norm
    from virgil.pipeline.binary import _null_nsigma

    grid = BinaryPipeline(data, **TINY)._grid()
    n_null = 20
    # A peak above every null maximum: k = 0, FAP at the floor 1 / (n + 1).
    sigma, fap, k, n = _null_nsigma(data, grid, None, n_null, 0, 1e12, 1.5)
    assert k == 0 and fap == pytest.approx(1.0 / (n_null + 1))
    assert sigma == pytest.approx(float(norm.isf(fap)))
    # A peak below every null maximum: FAP = 1, finite and negative.
    sigma, fap, k, n = _null_nsigma(data, grid, None, n_null, 0, -1.0, 1.5)
    assert k == n_null and fap == 1.0
    assert math.isfinite(sigma) and sigma < 0.0
    out = _null_nsigma(data, grid, None, n_null, 0, float("nan"), 1.5)
    assert all(math.isnan(x) for x in out)


def test_null_draws_stop_once_the_verdict_is_settled(data):
    from virgil.pipeline.binary import _null_nsigma

    grid = BinaryPipeline(data, **TINY)._grid()
    # 300 draws in three chunks of 100. Passing at 1.5σ allows no exceeding
    # null (k <= 0), so a peak below every null settles it after one chunk.
    sigma, fap, k, n = _null_nsigma(data, grid, None, 300, 0, -1.0, 1.5)
    assert (k, n) == (100, 100) and sigma < 0.0
    # A peak above every null can only be shown with all of them.
    assert _null_nsigma(data, grid, None, 300, 0, 1e12, 1.5)[2:] == (0, 300)


# Slow: 8 searches, each against 50 simulated null searches, take about two
# minutes on a CI core; it calibrates the null, which no cheaper run can.
@pytest.mark.slow
def test_noise_peaks_have_a_roughly_uniform_false_alarm_probability(data):
    """Peaks of independent companion-free searches are exchangeable with the
    simulated null maxima, so their FAPs average 1/2 (sd 0.29 / sqrt(8)); a
    null that included the companion or ignored the look-elsewhere effect
    would pin them at the floor or the ceiling."""
    from virgil.detection import detection_statistics, gaussian_null
    from virgil.pipeline.binary import _null_model, _null_nsigma

    grid = BinaryPipeline(data, **TINY)._grid()
    simulate = gaussian_null(_null_model(), data)
    faps = []
    for key in range(8):
        noise = simulate(jax.random.PRNGKey(1000 + key))
        peak = float(
            detection_statistics(BinaryModelCartesian, noise, grid)[
                "delta_chi2"
            ]
        )
        faps.append(_null_nsigma(data, grid, None, 50, 0, peak, 1.5)[1])
    assert 0.2 < onp.mean(faps) < 0.8


def test_search_report_uses_the_simulated_null(full_run):
    _, result = full_run
    search = result.summary["search"]
    assert "n_trials" not in search
    n = search["n_null"]
    assert n == TINY["n_null"]
    # TRUTH is a ~flux-0.01 companion at hundreds of sigma: no null reaches it.
    assert search["false_alarm_probability"] == pytest.approx(1.0 / (n + 1))
    assert search["n_null_above"] == 0
    assert search["global_nsigma_is_lower_bound"] is True
    assert search["local_nsigma"] > search["global_nsigma"]
    assert "simulated" in search["global_nsigma_method"]
    detection = next(c for c in result.checks if c.name == "detection")
    assert detection.status == "pass" and "≥" in detection.message


def test_n_null_must_resolve_the_detection_threshold(data):
    with pytest.raises(ValueError, match="n_null=500 cannot resolve"):
        BinaryPipeline(data, n_null=500)
    with pytest.raises(ValueError, match="cannot resolve detection_sigma=5"):
        BinaryPipeline(data, detection_sigma=5.0)
    BinaryPipeline(data, n_null=740)  # 1 / 741 <= the 3σ tail 1.35e-3


def test_grid_is_capped_at_an_odd_number_and_the_step_reported(data, tmp_path):
    settings = dict(TINY, quicklook_execute=False)
    settings.update(max_grid=12, grid_step_mas=10.0)
    pipeline = BinaryPipeline(data, output=tmp_path / "run", **settings)
    grid = pipeline._grid()
    assert grid["dra"].size == grid["ddec"].size == 11  # not 13
    max_sep, n, step, wanted, _ = pipeline._grid_axis()
    assert n == 11 and step == pytest.approx(2 * max_sep / 10)
    assert step > wanted == 10.0
    with pytest.raises(ValueError, match="max_grid"):
        BinaryPipeline(data, max_grid=2)

    res = pipeline.run(through="search")
    report = json.loads(
        (tmp_path / "run/stages/search/report.json").read_text()
    )
    assert report["grid_step_mas"] == pytest.approx(step)
    assert report["grid_step_wanted_mas"] == 10.0
    assert report["grid_shape"][:2] == [11, 11]
    messages = [w["message"] for w in report["warnings"]]
    assert any("coarsens the grid step" in m for m in messages)
    assert "stage_warnings" in [c.name for c in res.checks]


def test_grids_file_stores_the_units_of_each_axis(full_run):
    import h5py

    pipeline, _ = full_run
    with h5py.File(pipeline.output / "grids.h5", "r") as f:
        assert f["axes/dra"].attrs["units"] == "mas"
        assert f["axes/ddec"].attrs["units"] == "mas"
        assert f["axes/flux"].attrs["units"] == "companion/primary"
        assert "limits/limit_flux" in f and "search/snr" in f
        assert json.loads(f.attrs["axis_order"]) == ["dra", "ddec", "flux"]


def test_fit_and_posterior_recover_the_injected_companion(full_run):
    """The fitted and sampled companion match the truth within a few sigma
    (the quoted widths of the 30-draw chains get a floor)."""
    _, res = full_run
    truth = {
        "sep": float(onp.hypot(TRUTH.dra, TRUTH.ddec)),
        "pa": float(onp.degrees(onp.arctan2(TRUTH.dra, TRUTH.ddec))),
        "flux": float(TRUTH.flux),
    }
    floor = {"sep": 1.0, "pa": 0.5, "flux": 1e-3}
    post = res.summary["posterior"]["params"]
    fit = res.summary["fit"]["params"]
    for name, value in truth.items():
        row = post[name]
        sigma = max(0.5 * (row["q84"] - row["q16"]), floor[name])
        assert abs(row["median"] - value) < 4 * sigma, name
        assert abs(fit[name] - value) < 4 * sigma, name


def test_refined_flux_is_the_two_public_grids_from_one_pass(data):
    import jax.numpy as np
    from virgil.grid_fit import (
        optimized_flux_grid,
        optimized_likelihood_grid,
    )
    from virgil.pipeline.binary import _peak_index, _refined_flux

    axis = np.linspace(-200.0, 200.0, 9)
    grid = {"dra": axis, "ddec": axis, "flux": np.logspace(-3, -1, 5)}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        flux, loglike = _refined_flux(data, grid, None)
        want_flux = optimized_flux_grid(BinaryModelCartesian, data, grid)
        want_loglike = optimized_likelihood_grid(
            BinaryModelCartesian, data, grid
        )
    onp.testing.assert_array_equal(flux, want_flux)
    onp.testing.assert_array_equal(loglike, want_loglike)

    # One source for the peak: the nearest grid point on each axis.
    best = {"dra": 50.0, "ddec": -100.0, "flux": 0.0}  # flux below the axis
    assert _peak_index(grid, best) == [5, 2, 0]
    best["flux"] = float(grid["flux"][3]) * 1.01
    assert _peak_index(grid, best) == [5, 2, 3]


# --- CLI ----------------------------------------------------------------------


def test_cli_info_and_validate(full_run, tmp_path, capsys):
    pipeline, _ = full_run
    assert cli.main(["info", str(pipeline.output)]) == 0
    assert "status: complete" in capsys.readouterr().out
    assert cli.main(["validate", str(pipeline.output)]) == 0
    assert "valid" in capsys.readouterr().out

    copy = tmp_path / "copy"
    shutil.copytree(pipeline.output, copy)
    (copy / "plots" / "fit_correlation.png").write_bytes(b"tampered")
    (copy / "grids.h5").unlink()
    assert cli.main(["validate", str(copy)]) == 1
    out = capsys.readouterr().out
    assert "fit_correlation.png changed" in out and "missing grids.h5" in out


def test_cli_validate_fails_for_failed_runs_and_missing_keys(
    full_run, tmp_path, capsys
):
    pipeline, _ = full_run
    for name, change, message in (
        (
            "failed",
            lambda run: run.update(
                status="failed",
                error={"stage": "fit", "type": "RuntimeError", "message": "x"},
            ),
            "the run failed in stage 'fit'",
        ),
        (
            "running",
            lambda run: run.update(status="running"),
            "did not finish (status 'running')",
        ),
        ("noclass", lambda run: run.pop("class"), "lacks 'class'"),
    ):
        copy = tmp_path / name
        shutil.copytree(pipeline.output, copy)
        run = json.loads((copy / "run.json").read_text())
        change(run)
        (copy / "run.json").write_text(json.dumps(run))
        assert cli.main(["validate", str(copy)]) == 1
        assert message in capsys.readouterr().out


def test_cli_binary_run_from_oifits(full_run, tmp_path, capsys):
    pipeline, _ = full_run
    path = pipeline.output / "data" / "processed.oifits"
    out = tmp_path / "cli"
    args = ["binary", str(path), "--output", str(out), "--through", "search"]
    for key, value in TINY.items():
        args += ["--set", f"{key}={json.dumps(value)}"]
    assert cli.main(args) == 0
    run = json.loads((out / "run.json").read_text())
    assert run["status"] == "partial"
    assert run["inputs"]["files"][0]["path"] == str(path.resolve())
    assert run["config"]["max_grid"] == 13
    assert "search.delta_chi2" in capsys.readouterr().out


def test_templates_share_one_prior_on_the_sky(data):
    """Angular and Cartesian templates sample the same dra, ddec box."""
    from virgil import BinaryModelAngular

    angular = BinaryPipeline(data, **TINY)
    cartesian = BinaryPipeline(
        data, BinaryModelCartesian(0.0, 0.0, 0.0), **TINY
    )
    pa, pc = angular._priors(300.0), cartesian._priors(300.0)
    assert list(pa) == list(pc) == ["dra", "ddec", "flux"]
    for k in pa:
        assert type(pa[k]) is type(pc[k])
        assert float(pa[k].low) == float(pc[k].low)
        assert float(pa[k].high) == float(pc[k].high)
    key = jax.random.PRNGKey(0)
    dra = onp.asarray(pa["dra"].sample(key, (4000,)))
    ddec = onp.asarray(pa["ddec"].sample(jax.random.PRNGKey(1), (4000,)))
    assert onp.all(onp.abs(dra) <= 300.0) and onp.all(onp.abs(ddec) <= 300.0)
    # Uniform in position: as much mass at small as at large radius per area.
    inner = onp.mean(onp.hypot(dra, ddec) < 100.0)
    assert abs(inner - onp.pi * 100.0**2 / 600.0**2) < 0.03
    # The Angular template derives sep and PA from the same offsets.
    m = angular._sampled_model()(120.0, 80.0, 0.01)
    assert isinstance(m, BinaryModelAngular)
    onp.testing.assert_allclose(float(m.sep), onp.hypot(120.0, 80.0))
    onp.testing.assert_allclose(
        float(m.pa), onp.degrees(onp.arctan2(120.0, 80.0))
    )


def test_residual_normality_passes_on_truth_noise(data):
    """The check uses the independent whitened residuals, not the CP penalty."""
    from virgil.likelihood import whitened_residuals
    from virgil.pipeline import _checks
    from virgil.pipeline._common import _moments

    r = onp.asarray(whitened_residuals(TRUTH, data))
    n = int(data.n_independent)
    skew, kurt = _moments(r[:n])
    check = _checks.residual_normality(skew, kurt, n)
    assert check.status == "pass", check.message


def test_prior_bound_fraction_is_the_one_fit_reports():
    """The pipeline reads prior edges with ``fitting._bound_fraction``: a
    LogUniform's in log x, an interval's, and none for a prior whose range
    ends are poles of its flat coordinate (it used to count them)."""
    import numpyro.distributions as dist
    from virgil.pipeline._common import _bound_samples
    from virgil.priors import IsotropicInclination

    uniform = dist.Uniform(0.0, 10.0)
    assert _bound_samples(uniform, [0.0, 0.05, 5.0, 9.95]) == 0.75
    log = dist.LogUniform(1e-4, 1.0)
    # 1e-4 and 1.00 are the two edges; 1e-2 is the middle of the log range.
    assert _bound_samples(log, [1e-4, 1e-2, 0.999]) == pytest.approx(2 / 3)
    assert _bound_samples(dist.Normal(0.0, 1.0), [0.0, 1.0]) is None
    assert _bound_samples(IsotropicInclination(), [0.0, 90.0, 180.0]) is None


def test_field_of_view_uses_each_sample_own_baseline_and_wavelength():
    """The longest wavelength was paired with the shortest baseline even
    where they were not observed together (λ_max / B_min), and a zero
    baseline must not set the field."""
    from types import SimpleNamespace

    from virgil._utils import mas2rad
    from virgil.imaging import field_of_view, nyquist_pixel_scale
    from virgil.pipeline._common import _geometry

    # 10 m at 1 µm and 100 m at 2 µm, plus a zero baseline at 3 µm.
    data = SimpleNamespace(
        u=onp.array([10.0, 100.0, 0.0]),
        v=onp.zeros(3),
        wavel=onp.array([1e-6, 2e-6, 3e-6]),
    )
    coarsest = 1e-6 / 10.0 / mas2rad  # not 2e-6 / 10 m
    finest = 2e-6 / 100.0 / mas2rad
    geo = _geometry(data)
    assert geo["fov_mas"] == pytest.approx(coarsest)
    assert geo["resolution_mas"] == pytest.approx(0.5 * finest)
    assert geo["lambda_over_b_mas"] == pytest.approx(finest)
    assert field_of_view(data, largest_mas=onp.inf) == pytest.approx(coarsest)
    assert nyquist_pixel_scale(data) == pytest.approx(0.5 * finest)


def test_moments_of_constant_residuals_are_finite():
    from virgil.pipeline._common import _moments

    assert _moments(onp.full(10, 0.1)) == (0.0, -3.0)
    assert _moments(onp.zeros(5)) == (0.0, -3.0)


# --- warnings raised inside stages --------------------------------------------


def test_stage_warnings_are_recorded_not_emitted(data, tmp_path, monkeypatch):
    import warnings

    import virgil.pipeline.binary as binary

    original = binary._overview

    def noisy(p):
        for _ in range(2):  # duplicates are recorded once
            warnings.warn(
                "detection_statistics(): the optimizer did not converge at "
                "2 of 169 grid positions; values there may be inaccurate.",
                RuntimeWarning,
            )
        warnings.warn("something odd happened", UserWarning)
        # Interpreter noise from a collected unclosed file is not recorded.
        warnings.warn("unclosed file <_io.BufferedReader>", ResourceWarning)
        return original(p)

    monkeypatch.setattr(binary, "_overview", noisy)
    settings = dict(TINY, quicklook_execute=False)
    pipeline = BinaryPipeline(data, output=tmp_path / "run", **settings)
    with warnings.catch_warnings(record=True) as emitted:
        warnings.simplefilter("always")
        res = pipeline.run(through="limits")
    assert not [w for w in emitted if issubclass(w.category, RuntimeWarning)]
    assert not [w for w in emitted if "something odd" in str(w.message)]

    report = json.loads(
        (tmp_path / "run/stages/overview/report.json").read_text()
    )
    assert [w["category"] for w in report["warnings"]] == [
        "RuntimeWarning",
        "UserWarning",
    ]
    assert {w["stage"] for w in report["warnings"]} == {"overview"}
    assert {w["message"] for w in report["warnings"]} <= {
        w["message"] for w in res.summary["warnings"]
    }
    names = [c.name for c in res.checks]
    assert "convergence" in names and "stage_warnings" in names
    check = {c.name: c for c in res.checks}["convergence"]
    assert check.status == "warn" and check.value >= 2 / 169 - 1e-12
    assert "warning(s) recorded" in res.describe()


# --- the default search box on lattice coverage -----------------------------

LATTICE_TRUTH = BinaryModelCartesian(99.9, -60.1, 0.02)  # 116.6 mas, PA 121°
LATTICE_SETTINGS = {
    "max_grid": 41,
    "n_flux": 6,
    "flux_range": [1e-3, 0.2],
    # n_null=15 floors the false-alarm probability at 1/16 (norm.isf = 1.534σ), so detection_sigma must be below it.
    "n_null": 15,
    "detection_sigma": 1.5,
    "quicklook_execute": False,
}


@pytest.fixture(scope="module")
def lattice_data():
    from virgil.coverage import nrm_oidata

    return nrm_oidata(sigma_v2=0.002, sigma_cp_deg=0.1).with_model(
        LATTICE_TRUTH, key=jax.random.PRNGKey(0)
    )


def test_default_box_on_lattice_coverage_stays_inside_one_alias_cell(
    lattice_data, tmp_path
):
    """The 7-hole mask's aliases are 671.9 mas away in the larger component
    (the old default half-width), so the box stops at half of that."""
    pipeline = BinaryPipeline(lattice_data, output=tmp_path / "run")
    axis = pipeline._grid_axis()
    assert axis.max_sep == axis.alias_cap == pytest.approx(335.96, abs=0.1)
    assert axis.max_sep < pipeline.settings["fov_fraction"] * 671.9 / 1.0001
    explicit = BinaryPipeline(
        lattice_data, output=tmp_path / "run2", max_sep_mas=500.0
    )
    assert explicit._grid_axis().max_sep == 500.0
    assert explicit._grid_axis().alias_cap is None


def test_lattice_default_search_recovers_the_companion_not_an_alias(
    lattice_data, tmp_path
):
    pipeline = BinaryPipeline(
        lattice_data, output=tmp_path / "run", **LATTICE_SETTINGS
    )
    res = pipeline.run(through="search")
    search = json.loads(
        (tmp_path / "run/stages/search/report.json").read_text()
    )
    assert search["alias_cap_mas"] == pytest.approx(335.96, abs=0.1)
    assert search["max_sep_mas"] == search["alias_cap_mas"]
    assert search["dra_mas"] == pytest.approx(99.9, abs=20.0)
    assert search["ddec_mas"] == pytest.approx(-60.1, abs=20.0)
    assert res.summary["search"]["alias_cap_mas"] == search["alias_cap_mas"]


def test_coverage_that_is_not_a_lattice_keeps_the_field_of_view(tmp_path):
    from tests._shared import vlti_night
    from virgil.pipeline._common import _geometry

    night = vlti_night(60000.0, frames=3)
    axis = BinaryPipeline(night, output=tmp_path / "run")._grid_axis()
    assert axis.alias_cap is None
    assert axis.max_sep == pytest.approx(_geometry(night)["fov_mas"])


def test_lattice_alias_limit_needs_a_lattice_in_every_channel():
    from types import SimpleNamespace

    from virgil._geometry import lattice_alias_limit
    from virgil._utils import mas2rad
    from virgil.coverage import ami_grid_record

    square = OIData(ami_grid_record(pitch_m=0.5))
    assert lattice_alias_limit(square) == pytest.approx(
        0.5 * 4.3e-6 / 0.5 / mas2rad, rel=1e-4
    )
    ragged = SimpleNamespace(
        u=onp.array([10.0, 17.0, 31.0]),
        v=onp.array([0.0, 5.5, 9.3]),
        wavel=onp.array([2e-6]),
    )
    assert lattice_alias_limit(ragged) is None
    colours = SimpleNamespace(
        u=onp.array([10.0, 20.0, 10.0]),
        v=onp.array([0.0, 0.0, 10.0]),
        wavel=onp.array([2e-6, 2.1e-6, 2e-6]),
    )
    assert (
        lattice_alias_limit(colours) is None
    )  # the 2.1 µm channel has one sample


def test_lattice_alias_limit_is_set_by_the_longest_unflagged_channel():
    """A shift is ambiguous only if it aliases in every channel, and every such
    shift is an alias of the longest one, so that channel sets the limit; a
    flagged channel does not count."""
    from types import SimpleNamespace

    from virgil._geometry import lattice_alias_limit
    from virgil._utils import mas2rad

    def limit(unflagged):
        data = SimpleNamespace(
            u=onp.array([10.0, 0.0, 20.0, 10.0, 0.0, 20.0]),
            v=onp.array([0.0, 10.0, 10.0, 0.0, 10.0, 10.0]),
            wavel=onp.repeat([2e-6, 4e-6], 3),
            unflagged=onp.array(unflagged),
        )
        return lattice_alias_limit(data)

    short = 0.5 * 2e-6 / 10.0 / mas2rad
    assert limit([True] * 6) == pytest.approx(2 * short)
    assert limit([True] * 3 + [False] * 3) == pytest.approx(short)
    assert limit([False] * 3 + [True] * 3) == pytest.approx(2 * short)
    assert limit([False] * 6) is None


def test_lattice_alias_limit_uses_only_the_unflagged_samples():
    """Without the baseline (10, 0) the rest generate a coarser lattice, whose
    alias cell is smaller in x."""
    from types import SimpleNamespace

    from virgil._geometry import lattice_alias_limit
    from virgil._utils import mas2rad

    def limit(unflagged):
        data = SimpleNamespace(
            u=onp.array([10.0, 0.0, 20.0]),
            v=onp.array([0.0, 10.0, 10.0]),
            wavel=onp.array([2e-6]),
            unflagged=onp.array(unflagged),
        )
        return lattice_alias_limit(data)

    assert limit([True, True, True]) == pytest.approx(
        0.5 * 2e-6 / 10.0 / mas2rad
    )
    assert limit([False, True, True]) == pytest.approx(
        0.5 * 2e-6 / 20.0 / mas2rad
    )
