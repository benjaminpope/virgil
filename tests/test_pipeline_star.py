"""StarPipeline end to end on tiny synthetic data: API contract, outputs,
round trip, resume, quicklook and the CLI, plus the new checks.

Chains are deliberately tiny: these tests check the plumbing and the
contract, not the sampler (a star observed past its first null has closure
phases that flip there, so its sampler diagnostics are expected to fail).
"""

import dataclasses
import inspect
import json
import math

import jax
import numpy as onp
import pytest

pytest.importorskip("h5py")
pytest.importorskip("nbclient")

import virgil  # noqa: E402
from virgil import UniformDisk  # noqa: E402
from virgil.coverage import vlti_oidata  # noqa: E402
from virgil.models import QuadraticLimbDarkenedDisk  # noqa: E402
from virgil.pipeline import (  # noqa: E402
    SCHEMA,
    Check,
    ConfigMismatchError,
    StarPipeline,
    load,
)
from virgil.pipeline import _checks as checks  # noqa: E402
from virgil.pipeline import cli, star  # noqa: E402

TINY = {
    "num_warmup": 30,
    "num_samples": 30,
    "num_chains": 2,
}
TRUTH = QuadraticLimbDarkenedDisk.from_u(6.0, u1=0.35, u2=0.25)


@pytest.fixture(scope="module")
def data():
    template = vlti_oidata(
        wavelengths_m=onp.linspace(1.6e-6, 2.4e-6, 3),
        hour_angles_h=(-2.5, 0.0, 2.5),
        sigma_v2=0.002,
        sigma_cp_deg=0.5,
    )
    return template.with_model(TRUTH, key=jax.random.PRNGKey(3))


@pytest.fixture(scope="module")
def full_run(data, tmp_path_factory):
    out = tmp_path_factory.mktemp("star") / "run"
    pipeline = StarPipeline(data, output=out, **TINY)
    return pipeline, pipeline.run()


# --- the stable contract ----------------------------------------------------

API = {
    "StarPipeline": "(data, model=None, *, output='run', **settings)",
    "StarPipeline.run": "(self, through=None, *, resume=True)",
    "StarPipeline.from_oifits": "(paths, model=None, *, output='run', **settings)",
    "StarPipeline.from_config": "(path, model=None, **overrides)",
    "StarPipeline.defaults": "()",
}
STAGES = ("load", "overview", "fit", "posterior", "quicklook")
DEFAULTS = {
    "diam_range_mas": None,
    "n_scan": 2000,
    "wavel_range": None,
    "error_floor": None,
    "error_scale": "quoted",
    "num_warmup": 1000,
    "num_samples": 1000,
    "num_chains": 4,
    "chain_method": "vectorized",
    "seed": 0,
    "batch_size": None,
    "quicklook_execute": True,
}
SUMMARY_KEYS = {
    "schema": None,
    "pipeline": None,
    "stability": None,
    "status": None,
    "worst_check": None,
    "checks": None,
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
        "freq_max_per_rad",
    },
    "chi2": {"n_independent", "error_scale", "reduced", "error_scales"},
    "fit": {"best", "scan_diam_mas", "models"},
    "comparison": {
        "simple",
        "complex",
        "delta_chi2",
        "n_extra_params",
        "bic_penalty",
        "preferred",
    },
    "star": {"model", "source", "diam_mas", "diam_err_mas"},
    "resolution": {"diam_mas", "freq_max_per_rad", "first_null_fraction"},
    "shape": {"model", "params"},
    "posterior": {
        "models",
        "best",
        "sampled",
        "num_chains",
        "num_samples",
        "num_warmup",
        "r_hat_max",
        "ess_bulk_min",
        "divergence_fraction",
    },
}
CHECKS = [
    "chi2_uniform",
    "chi2_limb_darkened",
    "limb_darkening_gain",
    "resolution",
    "limb_darkening_constrained",
    "prior_bound",
    "r_hat",
    "ess",
    "divergences",
]


def _signature(name):
    obj = {"StarPipeline": StarPipeline}[name.split(".")[0]]
    for part in name.split(".")[1:]:
        obj = getattr(obj, part)
    params = [
        p.replace(annotation=inspect.Parameter.empty)
        for p in inspect.signature(obj).parameters.values()
    ]
    return str(
        inspect.signature(obj).replace(
            parameters=params, return_annotation=inspect.Signature.empty
        )
    )


def test_api_snapshot():
    # Changing any of these breaks users' scripts: update the snapshot only
    # with an additive change or a schema bump (see docs/pipeline.md).
    assert {k: _signature(k) for k in API} == API
    assert StarPipeline.STAGES == STAGES
    assert StarPipeline.defaults() == DEFAULTS
    assert StarPipeline.STABILITY == "stable" and StarPipeline.NAME == "star"
    assert SCHEMA == "virgil-pipeline-run-v1"
    assert "StarPipeline" in virgil.pipeline.__all__
    assert sorted(star.models()) == ["limb_darkened", "uniform"]
    assert star.DEFAULT_MODELS == ("uniform", "limb_darkened")


def test_summary_keys_snapshot(full_run):
    _, res = full_run
    keys = {
        k: (set(v) if isinstance(v, dict) else None)
        for k, v in res.summary.items()
    }
    assert keys == SUMMARY_KEYS
    assert [c.name for c in res.checks] == CHECKS
    assert all(isinstance(c, Check) for c in res.checks)


def test_settings_and_model_are_validated(data):
    with pytest.raises(ValueError, match="diam_range_mas"):
        StarPipeline(data, diam_range_mas=[5.0, 1.0])
    with pytest.raises(ValueError, match="error_scale"):
        StarPipeline(data, error_scale="rescaled")
    with pytest.raises(TypeError, match="unknown settings"):
        StarPipeline(data, num_sample=10)
    with pytest.raises(ValueError, match="model must be one of"):
        StarPipeline(data, "gaussian")
    with pytest.raises(TypeError, match="no model for a PointSource"):
        StarPipeline(data, virgil.PointSource())


def test_model_argument_selects_the_models(data, monkeypatch):
    assert StarPipeline(data).names == ["uniform", "limb_darkened"]
    assert StarPipeline(data, "uniform").names == ["uniform"]
    template = QuadraticLimbDarkenedDisk(4.0, q1=0.2, q2=0.7)
    pipeline = StarPipeline(data, template)
    assert pipeline.names == ["limb_darkened"]
    assert pipeline._starts["limb_darkened"] == pytest.approx(
        {"q1": 0.2, "q2": 0.7}
    )
    assert StarPipeline(data, UniformDisk(2.0)).names == ["uniform"]

    # A new model is a registry entry: no new class, no new stage.
    entry = dataclasses.replace(star.models()["uniform"], name="disk2")
    monkeypatch.setitem(star.models(), "disk2", entry)
    assert StarPipeline(data, "disk2").names == ["disk2"]


# --- outputs and the round trip ---------------------------------------------


def test_run_folder_layout(full_run):
    pipeline, _ = full_run
    out = pipeline.output
    run = json.loads((out / "run.json").read_text())
    assert run["schema"] == SCHEMA and run["status"] == "complete"
    assert run["stability"] == "stable" and run["error"] is None
    assert list(run["stages"]) == list(STAGES)
    assert set(run["config"]) == set(DEFAULTS)
    assert run["model"]["models"] == ["uniform", "limb_darkened"]
    for name in (
        "summary.json",
        "grids.h5",
        "samples.h5",
        "quicklook.ipynb",
        "data/processed.oifits",
        "models/best/manifest.json",
        "models/uniform/manifest.json",
        "models/limb_darkened/values.npz",
        "plots/fit_v2.png",
        "plots/overview_uv.png",
        "stages/fit/report.json",
    ):
        assert (out / name).exists(), name
    assert not list(out.rglob("*.pkl")) and not list(out.rglob("*.pickle"))


def test_diameter_recovered_and_quoted_errors_lead(full_run):
    _, res = full_run
    s = res.summary
    ld = s["fit"]["models"]["limb_darkened"]
    assert abs(ld["params"]["diam"] - 6.0) < 0.1
    assert abs(ld["params"]["u1"] - 0.35) < 0.15
    assert s["fit"]["best"] == "limb_darkened"
    # The uniform disk cannot describe a star seen past its first null, and
    # the raw chi^2/N on the quoted errors says so; no rescaling hides it.
    assert s["chi2"]["error_scale"] == "quoted"
    assert s["chi2"]["reduced"]["limb_darkened"] < 2.0
    assert s["chi2"]["reduced"]["uniform"] > 100.0
    assert s["comparison"]["preferred"] == "limb_darkened"
    assert s["comparison"]["delta_chi2"] > 100 * s["comparison"]["bic_penalty"]
    assert abs(s["star"]["diam_mas"] - 6.0) < 0.5
    assert s["resolution"]["first_null_fraction"] > 1.0
    by_name = {c.name: c for c in res.checks}
    assert by_name["chi2_limb_darkened"].status == "pass"
    assert by_name["chi2_uniform"].status == "fail"
    assert by_name["resolution"].status == "pass"


def test_round_trip(full_run, data):
    pipeline, _ = full_run
    res = load(pipeline.output)  # as in a fresh session
    assert res.status == "complete"

    back = res.data()
    onp.testing.assert_allclose(back.vis, data.vis, rtol=1e-6)

    model = res.model()
    assert type(model) is QuadraticLimbDarkenedDisk
    fitted = res.summary["fit"]["models"]["limb_darkened"]["params"]
    for name in ("diam", "q1", "q2"):
        onp.testing.assert_allclose(
            float(getattr(model, name)), fitted[name], rtol=1e-6
        )
    assert type(res.model("limb_darkened")) is QuadraticLimbDarkenedDisk
    uniform = res.model("uniform")
    assert type(uniform) is UniformDisk
    onp.testing.assert_allclose(
        float(uniform.diam), res.summary["fit"]["scan_diam_mas"], rtol=0.02
    )
    assert set(res.model_values("uniform")) == {"diam"}
    fit = res.fit_result("uniform")
    assert fit.info["method"] == "scan" and fit.values.keys() == {"diam"}
    assert res.fit_result().info["method"] == "lm"

    samples = res.samples()  # the preferred model
    assert samples["diam"].shape == (TINY["num_chains"], TINY["num_samples"])
    assert {"q1", "q2", "u1", "u2"} <= set(samples)
    assert set(res.samples(name="uniform")) == {"diam"}
    flat = res.samples(group_by_chain=False)
    assert flat["q1"].shape == (TINY["num_chains"] * TINY["num_samples"],)
    median = res.summary["posterior"]["models"]["limb_darkened"]["params"]
    onp.testing.assert_allclose(
        onp.median(flat["diam"]), median["diam"]["median"], rtol=1e-5
    )
    stats = res.sample_stats("uniform")
    assert stats["diverging"].shape == (
        TINY["num_chains"],
        TINY["num_samples"],
    )

    grid = res.grid()
    assert list(grid["axes"]) == ["diam"]
    assert grid["delta_chi2"].shape == grid["axes"]["diam"].shape
    # Relative to the refined best point, which the coarse axis can miss.
    assert grid["delta_chi2"].min() >= 0.0


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
    assert images == 4  # overview, V² with models, correlation, corner
    main = "".join(cells[code[2] - 1]["source"]) + "".join(
        cells[code[2]]["source"]
    )
    assert "plot_v2_models" in main and "squared visibilities" in main
    assert "| `limb_darkening_gain` |" in "".join(cells[-1]["source"])


# --- single models ------------------------------------------------------------


def test_uniform_only_run_and_error_scale(data, tmp_path):
    out = tmp_path / "ud"
    res = StarPipeline(
        data, "uniform", output=out, quicklook_execute=False, **TINY
    ).run(through="fit")
    assert res.status == "partial"
    assert list(res.summary["fit"]["models"]) == ["uniform"]
    assert "comparison" not in res.summary
    assert [c.name for c in res.checks] == ["chi2_uniform", "resolution"]
    assert type(res.model()) is UniformDisk

    out = tmp_path / "scaled"
    res = StarPipeline(
        data,
        QuadraticLimbDarkenedDisk(5.0, q1=0.5, q2=0.5),
        output=out,
        error_scale="fit",
        **TINY,
    ).run(through="fit")
    scales = res.summary["chi2"]["error_scales"]["limb_darkened"]
    assert set(scales) == {"vis_scale", "phi_scale"}
    # chi^2/N stays on the quoted errors, and the scale check follows it.
    assert res.checks[0].name == "chi2_limb_darkened"
    assert "error_scale" in [c.name for c in res.checks]


# --- resume -------------------------------------------------------------------


def test_resume_skips_completed_stages(data, tmp_path, monkeypatch):
    out = tmp_path / "run"
    settings = dict(TINY, quicklook_execute=False)
    res = StarPipeline(data, output=out, **settings).run(through="fit")
    assert res.status == "partial" and "posterior" not in res.summary
    before = json.loads((out / "run.json").read_text())["stages"]

    # A newer virgil with an extra default-valued setting: the stored config
    # lacks it, and resuming must still work.
    run = json.loads((out / "run.json").read_text())
    del run["config"]["n_scan"]
    (out / "run.json").write_text(json.dumps(run))

    ran = []
    for name in ("_load", "_overview", "_fit"):
        monkeypatch.setattr(star, name, lambda p, n=name: ran.append(n))
    res = StarPipeline(data, output=out, **settings).run(through="posterior")
    assert ran == []  # nothing before the posterior was recomputed
    after = json.loads((out / "run.json").read_text())
    for name in ("load", "overview", "fit"):
        assert after["stages"][name]["started"] == before[name]["started"]
    assert "posterior" in res.summary

    with pytest.raises(ConfigMismatchError, match="num_chains"):
        StarPipeline(data, output=out, **dict(settings, num_chains=3)).run()
    with pytest.raises(ConfigMismatchError, match="model template differs"):
        StarPipeline(data, "uniform", output=out, **settings).run()
    shifted = data.with_error_floor(absolute={"vis": 0.5})
    with pytest.raises(ConfigMismatchError, match="data differ"):
        StarPipeline(shifted, output=out, **settings).run()


# --- CLI ----------------------------------------------------------------------


def test_cli_star_run_from_oifits(full_run, tmp_path, capsys):
    pipeline, _ = full_run
    path = pipeline.output / "data" / "processed.oifits"
    out = tmp_path / "cli"
    args = ["star", str(path), "-o", str(out), "--through", "fit"]
    args += ["--model", "limb_darkened"]
    args += ["--set", "n_scan=200"]
    assert cli.main(args) == 0
    run = json.loads((out / "run.json").read_text())
    assert run["status"] == "partial" and run["pipeline"] == "star"
    assert run["model"]["models"] == ["limb_darkened"]
    assert run["inputs"]["files"][0]["path"] == str(path.resolve())
    assert run["config"]["n_scan"] == 200
    assert "star.diam_mas" in capsys.readouterr().out
    assert cli.main(["info", str(out)]) == 0
    assert "chi2_limb_darkened" in capsys.readouterr().out
    assert cli.main(["validate", str(out)]) == 0


# --- the new checks -------------------------------------------------------------


def _freq(diam_mas, fraction):
    """B/λ at which a disk of ``diam_mas`` reaches ``fraction`` of its null."""
    return fraction * checks.FIRST_NULL / (diam_mas * math.pi / 180 / 3.6e6)


@pytest.mark.parametrize(
    "fraction, status",
    [(0.05, "fail"), (0.5, "warn"), (1.0, "pass"), (2.5, "pass")],
)
def test_resolution_regime(fraction, status):
    check = checks.resolution_regime(6.0, _freq(6.0, fraction))
    assert check.status == status
    assert check.value == pytest.approx(fraction)
    assert check.name == "resolution"
    json.dumps(check.to_dict())


def test_resolution_message_names_the_regime():
    short = checks.resolution_regime(6.0, _freq(6.0, 0.5))
    assert "limb darkening is degenerate" in short.message
    assert "0.5 of the first null" in short.message
    assert (
        "unresolved" in checks.resolution_regime(6.0, _freq(6.0, 0.05)).message
    )
    assert checks.resolution_regime(float("nan"), 1e7).status == "fail"


@pytest.mark.parametrize(
    "ud, ld, status, word",
    [
        (200.0, 100.0, "pass", "preferred"),
        (105.0, 100.0, "pass", "adequate"),
        (100.0, 130.0, "warn", "did not converge"),
        (float("nan"), 1.0, "fail", "not finite"),
    ],
)
def test_model_gain(ud, ld, status, word):
    check = checks.model_gain(
        "limb_darkening_gain",
        ud,
        ld,
        81,
        n_extra=2,
        simple="uniform disk",
        complex_="limb-darkened disk",
    )
    assert check.status == status and word in check.message
    assert check.threshold == pytest.approx(2 * math.log(81))
    json.dumps(check.to_dict())


@pytest.mark.parametrize(
    "ratios, status",
    [
        ({"q1": 0.2, "q2": 0.5}, "pass"),
        ({"q1": 0.3, "q2": 0.95}, "warn"),
        ({"q1": 0.8, "q2": 0.1}, "warn"),
        ({"q1": float("nan")}, "warn"),
    ],
)
def test_prior_constrained(ratios, status):
    check = checks.prior_constrained("limb_darkening_constrained", ratios)
    assert check.status == status
    if status == "warn":
        assert "as wide as the prior" in check.message
    json.dumps(check.to_dict())


def test_prior_dominated_names_the_parameters():
    check = checks.prior_constrained(
        "limb_darkening_constrained", {"q1": 0.3, "q2": 0.95}
    )
    assert "q2" in check.message.split("(")[0]
    assert "q1" not in check.message.split("(")[0]
