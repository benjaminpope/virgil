"""BinaryPipeline end to end on tiny synthetic data: API contract, outputs,
round trip, resume, quicklook and the CLI.

Grids and chains are deliberately tiny: these tests check the plumbing and
the contract, not the science.
"""

import inspect
import json
import warnings
import shutil

import jax
import numpy as onp
import pytest

pytest.importorskip("h5py")
pytest.importorskip("nbclient")

import virgil  # noqa: E402
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

TINY = {
    "max_sep_mas": 300.0,
    "max_grid": 13,
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
    pipeline = BinaryPipeline(data, output=out, **TINY)
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
    "Result.model": "(self)",
    "Result.model_values": "(self)",
    "Result.samples": "(self, group_by_chain=True)",
    "Result.sample_stats": "(self)",
    "Result.grid": "(self)",
    "Result.fit_result": "(self)",
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
    "max_sep_mas": None,
    "fov_fraction": 1.0,
    "grid_step_mas": None,
    "max_grid": 101,
    "flux_range": [1e-5, 0.5],
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
        "n_trials",
        "dra_mas",
        "ddec_mas",
        "flux",
        "max_sep_mas",
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


def _signature(path):
    obj = virgil.pipeline
    for part in path.split("."):
        obj = getattr(obj, part)
    sig = inspect.signature(obj)
    # Annotations print differently across Python versions: compare names,
    # kinds and defaults only.
    params = [
        p.replace(annotation=inspect.Parameter.empty)
        for p in sig.parameters.values()
    ]
    return str(sig.replace(parameters=params, return_annotation=sig.empty))


def test_api_snapshot():
    # Changing any of these breaks users' scripts: update the snapshot only
    # with an additive change or a schema bump (see docs/pipeline.md).
    assert {k: _signature(k) for k in API} == API
    assert BinaryPipeline.STAGES == STAGES
    assert BinaryPipeline.defaults() == DEFAULTS
    assert SCHEMA == "virgil-pipeline-run-v1"
    assert "BinaryPipeline" in virgil.pipeline.__all__


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
    assert run["stability"] == "stable" and run["error"] is None
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
    pipeline, _ = full_run
    nb = json.loads((pipeline.output / "quicklook.ipynb").read_text())
    cells = nb["cells"]
    assert (
        cells[0]["cell_type"] == "markdown"
        and cells[-1]["cell_type"] == "markdown"
    )
    code = [i for i, c in enumerate(cells) if c["cell_type"] == "code"]
    for i in code:
        assert cells[i - 1]["cell_type"] == "markdown"  # prose before code
        assert len(cells[i]["outputs"]) == 1, cells[i]["source"]  # one output
        assert cells[i]["outputs"][0]["output_type"] != "error"
    images = sum(
        "image/png" in o.get("data", {})
        for c in cells
        if c["cell_type"] == "code"
        for o in c["outputs"]
    )
    assert images == 4  # overview, search + limits, correlation, corner
    assert "| `r_hat` |" in "".join(cells[-1]["source"])


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
    (out / "run.json").write_text(json.dumps(run))

    ran = []
    import virgil.pipeline.binary as binary

    for name in ("_load", "_overview", "_search"):
        monkeypatch.setattr(binary, name, lambda p, n=name: ran.append(n))
    res = BinaryPipeline(data, output=out, limit_bins=20, **settings).run(
        through="limits"
    )
    assert ran == []  # nothing before limits was recomputed
    after, run = _started(out)
    assert {k: after[k] for k in before} == before
    assert run["status"] == "partial" and "limits" in res.summary

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
    from virgil.pipeline.binary import _moments

    r = onp.asarray(whitened_residuals(TRUTH, data))
    n = int(data.n_independent)
    skew, kurt = _moments(r[:n])
    check = _checks.residual_normality(skew, kurt, n)
    assert check.status == "pass", check.message


def test_moments_of_constant_residuals_are_finite():
    from virgil.pipeline.binary import _moments

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
