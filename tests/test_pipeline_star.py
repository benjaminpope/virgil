"""StarPipeline end to end on tiny synthetic data: API contract, outputs,
round trip, resume, quicklook and the CLI, plus the new checks.

Chains are deliberately tiny: these tests check the plumbing and the
contract, not the sampler (a star observed past its first null has closure
phases that flip there, so its sampler diagnostics are expected to fail).
"""

import dataclasses
import json
import math

import jax
import numpy as onp
import pytest

pytest.importorskip("h5py")
pytest.importorskip("nbclient")

import virgil  # noqa: E402
from tests._shared import executed_quicklook  # noqa: E402
from virgil import UniformDisk  # noqa: E402
from virgil._precision import in_float64  # noqa: E402
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

# The module's tests share expensive module-scoped fixtures: with
# ``pytest -n N --dist loadgroup`` they run on one worker, which builds each
# fixture once instead of once per worker.
pytestmark = pytest.mark.xdist_group("pipeline_star")

# max_lobes=2: each extra lobe costs two more L-BFGS fits per model, and
# on this data the best lobe is the scan's deepest minimum, so two are
# plenty. The lobe tests below set their own max_lobes.
TINY = {
    "max_lobes": 2,
    "num_warmup": 30,
    "num_samples": 30,
    "num_chains": 2,
}
# Long enough to adapt (30 warmup steps are too few, whatever the model), so
# the full run can also serve the divergence test.
FULL = {**TINY, "num_warmup": 200, "num_samples": 100}
TRUTH = QuadraticLimbDarkenedDisk.from_u(6.0, u1=0.35, u2=0.25)


@pytest.fixture(scope="module")
def data():
    template = vlti_oidata(
        wavelengths_m=onp.linspace(1.6e-6, 2.4e-6, 3),
        hour_angles_h=(-2.5, 0.0, 2.5),
        sigma_v2=0.01,
        sigma_cp_deg=15.0,
    )
    return template.with_model(TRUTH, key=jax.random.PRNGKey(3))


@pytest.fixture(scope="module")
def full_run(data, tmp_path_factory):
    out = tmp_path_factory.mktemp("star") / "run"
    # The notebook is executed by test_quicklook_notebook alone.
    pipeline = StarPipeline(data, output=out, quicklook_execute=False, **FULL)
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
    "max_lobes": 5,
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
    "warnings": None,
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
    "fit": {
        "best",
        "scan_diam_mas",
        "diam_bounds_mas",
        "rivals",
        "models",
    },
    "comparison": {
        "simple",
        "complex",
        "delta_chi2",
        "delta_m2loglike",
        "n_extra_params",
        "bic_penalty",
        "preferred",
    },
    "comparisons": None,
    "star": {"model", "source", "diam_mas", "diam_err_mas"},
    "resolution": {
        "diam_mas",
        "freq_max_per_rad",
        "first_null",
        "first_null_fraction",
    },
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
    "multimodal",
    "limb_darkening_constrained",
    "prior_bound",
    "r_hat",
    "ess",
    "divergences",
]


def test_pipeline_specific_contract():
    # The shared part (signatures, stages, defaults) is test_api_snapshot
    # in test_pipeline_robust_core.py.
    assert sorted(star.models()) == ["limb_darkened", "uniform"]
    assert star.DEFAULT_MODELS == ("uniform", "limb_darkened")


def test_summary_keys_snapshot(full_run):
    _, res = full_run
    keys = {
        k: (set(v) if isinstance(v, dict) else None)
        for k, v in res.summary.items()
    }
    assert keys == SUMMARY_KEYS
    # stage_warnings appears only when a stage warned (e.g. a fit that did
    # not converge on the tiny data).
    names = [c.name for c in res.checks if c.name != "stage_warnings"]
    assert names == CHECKS
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
    assert run["stability"] == "provisional" and run["error"] is None
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
    # The simulated noise differs by platform, so allow several sigma.
    assert abs(ld["params"]["diam"] - 6.0) < 0.4
    # u1 is weakly constrained by this noise (sigma_V^2 = 0.01 on 81
    # observables): the truth lies within 3 posterior sigma of the median.
    u1 = s["posterior"]["models"]["limb_darkened"]["params"]["u1"]
    assert abs(u1["median"] - 0.35) < 3 * u1["std"]
    assert u1["std"] < 0.5
    assert s["fit"]["best"] == "limb_darkened"
    # The uniform disk cannot describe a star seen past its first null, and
    # the raw chi^2/N on the quoted errors says so; no rescaling hides it.
    assert s["chi2"]["error_scale"] == "quoted"
    assert s["chi2"]["reduced"]["limb_darkened"] < 2.0, (
        s["fit"]["models"]["limb_darkened"],
        s.get("warnings"),
    )
    assert s["chi2"]["reduced"]["uniform"] > 2.2
    assert s["comparison"]["preferred"] == "limb_darkened"
    assert s["comparison"]["delta_chi2"] > 5 * s["comparison"]["bic_penalty"]
    assert s["comparison"]["delta_m2loglike"] == s["comparison"]["delta_chi2"]
    assert s["comparisons"] == [s["comparison"]]
    assert abs(s["star"]["diam_mas"] - 6.0) < 0.5
    assert s["resolution"]["first_null_fraction"] > 1.0
    by_name = {c.name: c for c in res.checks}
    assert by_name["chi2_limb_darkened"].status == "pass"
    assert by_name["chi2_uniform"].status in ("warn", "fail")
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
        float(uniform.diam), res.summary["fit"]["scan_diam_mas"], rtol=1e-6
    )
    onp.testing.assert_allclose(
        res.model_values("uniform")["diam"],
        res.summary["fit"]["scan_diam_mas"],
        rtol=1e-12,
    )
    assert set(res.model_values("uniform")) == {"diam"}
    fit = res.fit_result("uniform")
    assert fit.info["method"] == "scan" and fit.values.keys() == {"diam"}
    assert res.fit_result().info["method"] == "lm"

    samples = res.samples()  # the preferred model
    assert samples["diam"].shape == (FULL["num_chains"], FULL["num_samples"])
    assert {"q1", "q2", "u1", "u2"} <= set(samples)
    assert set(res.samples(name="uniform")) == {"diam"}
    flat = res.samples(group_by_chain=False)
    assert flat["q1"].shape == (FULL["num_chains"] * FULL["num_samples"],)
    median = res.summary["posterior"]["models"]["limb_darkened"]["params"]
    onp.testing.assert_allclose(
        onp.median(flat["diam"]), median["diam"]["median"], rtol=1e-5
    )
    stats = res.sample_stats("uniform")
    assert stats["diverging"].shape == (
        FULL["num_chains"],
        FULL["num_samples"],
    )

    grid = res.grid()
    assert list(grid["axes"]) == ["diam"]
    assert grid["delta_chi2"].shape == grid["axes"]["diam"].shape
    # The scan is coarse: its minimum is zero and lies within one coarse
    # step of the refined scan diameter.
    axis, delta = grid["axes"]["diam"], grid["delta_chi2"]
    scan = res.summary["fit"]["scan_diam_mas"]
    step = (axis[-1] / axis[0]) ** (1.0 / (axis.size - 1))
    coarse = axis[int(onp.argmin(delta))]
    assert delta.min() == 0.0
    assert coarse / step <= scan <= coarse * step
    # Its excess at another diameter, from chi^2 computed independently of
    # the scan.
    i = int(onp.argmin(abs(axis - 2 * coarse)))
    excess = float(star._chi2_64(UniformDisk(axis[i]), data)[0]) - float(
        star._chi2_64(UniformDisk(coarse), data)[0]
    )
    assert delta[i] == pytest.approx(excess, rel=1e-3, abs=1e-3)


def test_quicklook_notebook(full_run):
    pipeline, _ = full_run
    # overview, V² with models, correlation, corner
    cells, code = executed_quicklook(pipeline.output, images=4)
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
    assert [c.name for c in res.checks] == [
        "chi2_uniform",
        "resolution",
        "multimodal",
    ], res.summary.get("warnings")
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
    res = StarPipeline(data, "uniform", output=out, **settings).run(
        through="fit"
    )
    assert res.status == "partial" and "posterior" not in res.summary
    before = json.loads((out / "run.json").read_text())["stages"]

    # A newer virgil with an extra default-valued setting: the stored config
    # lacks it, and resuming must still work.
    run = json.loads((out / "run.json").read_text())
    del run["config"]["n_scan"]
    (out / "run.json").write_text(json.dumps(run))

    ran = []
    for name in ("_load_star", "_overview", "_fit"):
        monkeypatch.setattr(star, name, lambda p, n=name: ran.append(n))
    res = StarPipeline(data, "uniform", output=out, **settings).run(
        through="posterior"
    )
    assert ran == []  # nothing before the posterior was recomputed
    after = json.loads((out / "run.json").read_text())
    for name in ("load", "overview", "fit"):
        assert after["stages"][name]["started"] == before[name]["started"]
    assert "posterior" in res.summary

    with pytest.raises(ConfigMismatchError, match="num_chains"):
        StarPipeline(
            data, "uniform", output=out, **dict(settings, num_chains=3)
        ).run()
    with pytest.raises(ConfigMismatchError, match="model template differs"):
        StarPipeline(data, "limb_darkened", output=out, **settings).run()
    shifted = data.with_error_floor(absolute={"vis": 0.5})
    with pytest.raises(ConfigMismatchError, match="data differ"):
        StarPipeline(shifted, "uniform", output=out, **settings).run()


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


# --- review follow-ups ------------------------------------------------------


def test_the_posterior_has_few_divergences(full_run):
    """The prior covers the best lobe, so NUTS has no hard walls to hit.

    The full run's chains are longer than TINY: 30 warmup steps are too few
    to adapt, whatever the model.
    """
    pipeline, res = full_run
    s = res.summary
    lo, hi = s["fit"]["diam_bounds_mas"]
    full_lo, full_hi = pipeline._diam_range()
    # The lobe's bounds are a narrow window of the full prior range that
    # holds the truth, and every draw of the diameter lies inside them.
    assert hi / lo < 0.5 * full_hi / full_lo
    assert lo < 6.0 < hi
    for name, model in s["posterior"]["models"].items():
        diam = res.samples(name=name, group_by_chain=False)["diam"]
        assert lo <= diam.min() and diam.max() <= hi, name
        assert model["divergence_fraction"] <= 0.05, name


def test_scan_diameter_matches_a_brute_force_grid(data):
    lo, hi = 0.5, 30.0
    axis, delta, lobes = star._scan_lobes(data, lo, hi, 400, 1, None)
    best = lobes[0]["diam"]
    assert delta.shape == axis.shape
    window = onp.linspace(0.97 * best, 1.03 * best, 301)
    chi2 = onp.array(
        [float(star._chi2_64(UniformDisk(d), data)[0]) for d in window]
    )
    brute = window[int(onp.argmin(chi2))]
    assert 0.97 * best < brute < 1.03 * best
    assert abs(brute - best) <= 2 * (window[1] - window[0])


def test_refine_window_spans_one_coarse_step(data):
    """A coarse scan of 25 points still recovers the diameter."""
    axis, _, lobes = star._scan_lobes(data, 0.5, 30.0, 25, 1, None)
    reference = star._scan_lobes(data, 0.5, 30.0, 2000, 1, None)[2][0]
    assert lobes[0]["diam"] == pytest.approx(reference["diam"], rel=2e-3)
    step = (axis[-1] / axis[0]) ** (1.0 / (axis.size - 1))
    assert step > 1.1  # the old fixed +-1% window could not have found it


def test_find_lobes_rejects_a_scan_that_is_not_finite():
    axis = onp.geomspace(1.0, 10.0, 50)
    with pytest.raises(RuntimeError, match="not finite"):
        star._find_lobes(axis, onp.full(50, onp.nan), 1.0, 10.0, 3)


def test_error_scale_fit_compares_models_on_minus_two_log_likelihood(
    data, tmp_path
):
    """At each model's own scales chi^2 is N, so it cannot score the
    models: -2 log L must, and it prefers the true limb-darkened star."""
    res = StarPipeline(
        data, output=tmp_path / "run", error_scale="fit", **TINY
    ).run(through="fit")
    fit = res.summary["fit"]["models"]
    n = res.summary["chi2"]["n_independent"]
    uniform, ld = fit["uniform"], fit["limb_darkened"]
    # At the MAP chi^2 at the fitted scales is N, whatever the model: the
    # degeneracy that made it useless as a score. (The uniform disk's
    # scales are profiled analytically, an approximation for wrapping
    # phases, so only the fitted model is held to it.)
    assert ld["chi2_fitted"] == pytest.approx(n, rel=0.01)
    comp = res.summary["comparison"]
    assert comp["delta_m2loglike"] == pytest.approx(
        uniform["m2loglike"] - ld["m2loglike"]
    )
    assert comp["preferred"] == "limb_darkened"
    assert comp["delta_m2loglike"] > comp["bic_penalty"]
    assert res.summary["fit"]["best"] == "limb_darkened"
    # The uniform disk's scales are profiled along the scan.
    assert res.fit_result("uniform").info["scales"] == "profiled"
    # The quoted-error chi^2/N stays the diagnostic.
    assert res.summary["chi2"]["reduced"]["uniform"] == pytest.approx(
        uniform["chi2_reduced"]
    )
    assert uniform["chi2_reduced"] > ld["chi2_reduced"]


def test_profiled_scales_maximize_the_likelihood(data):
    """The profiled scales are the maximum of the likelihood: scaling any
    one of them by 0.9 or 1.1 raises -2 log L. (sqrt(chi2_b / N_b) is not
    the maximum for the von Mises closure phases, hence the profiling.)"""
    import numpyro.distributions as dist

    from virgil.pipeline import _common

    model = TRUTH
    noise = {
        "vis_scale": dist.LogUniform(0.1, 10.0),
        "phi_scale": dist.LogUniform(0.1, 10.0),
    }
    blocks = _common._scale_blocks(data, noise)
    assert [b[0] for b in blocks] == ["vis_scale", "phi_scale"]
    _, scales = _common._profiled_m2loglike(
        lambda _: model, {}, data, 0.0, blocks
    )
    best = _common._m2loglike_64(
        model, data, **{k: float(v) for k, v in scales.items()}
    )
    for key in scales:
        for factor in (0.9, 1.1):
            trial = {k: float(v) for k, v in scales.items()}
            trial[key] *= factor
            assert _common._m2loglike_64(model, data, **trial) > best, key


def test_scanned_fit_refines_the_diameter_between_grid_points():
    """With small errors the grid spacing (0.26 %) is many sigma_theta; the
    refined diameter must reach the optimum of a far denser grid."""
    import numpyro.distributions as dist

    from virgil.pipeline import _common

    template = vlti_oidata(
        wavelengths_m=onp.linspace(1.6e-6, 2.4e-6, 3),
        hour_angles_h=(-2.5, 0.0, 2.5),
        sigma_v2=1e-4,
        sigma_cp_deg=0.1,
    )
    small = template.with_model(UniformDisk(6.0), key=jax.random.PRNGKey(5))
    noise = {
        "vis_scale": dist.LogUniform(0.1, 10.0),
        "phi_scale": dist.LogUniform(0.1, 10.0),
    }
    pipeline = StarPipeline(small, "uniform", error_scale="fit")
    entry = star.models()["uniform"]
    start = {"diam": 6.0}
    grid = onp.geomspace(0.75 * 6.0, 1.25 * 6.0, star._START_POINTS)
    batch = pipeline.settings["batch_size"]
    coarse, _ = _common._m2loglike_grid(
        entry.build, start, small, grid, noise, batch
    )
    best = int(onp.nanargmin(coarse))
    values = star._fit_scanned(pipeline, entry, start, grid, noise)[1]
    refined = float(values["diam"])
    assert {"noise.vis_scale", "noise.phi_scale"} <= set(values)

    window = onp.geomspace(grid[best - 1], grid[best + 1], 2001)
    dense, _ = _common._m2loglike_grid(
        entry.build, start, small, window, noise, batch
    )
    step = grid[best + 1] / grid[best] - 1.0
    dense_best = float(window[int(onp.nanargmin(dense))])
    at_refined = _common._m2loglike_grid(
        entry.build, start, small, onp.array([refined]), noise, batch
    )[0][0]
    assert abs(refined / dense_best - 1.0) < 0.05 * step
    assert at_refined <= coarse[best]


def test_load_removes_a_stale_processed_oifits(data, tmp_path, monkeypatch):
    from types import SimpleNamespace

    folder = tmp_path / "data"
    folder.mkdir()
    stale = folder / "processed.oifits"
    stale.write_text("an earlier run's data")
    p = SimpleNamespace(processed=data, output=tmp_path)
    monkeypatch.setattr(
        "virgil.oifits._oidata_tables", lambda d: (None, "no tables")
    )
    report = star._load_star(p)
    assert not stale.exists()
    assert report["processed_oifits_skipped"] == "no tables"
    assert "freq_max_per_rad" in report


def _fake_models(parents):
    entries = {}
    for name, (parent, n_params) in parents.items():
        entries[name] = dataclasses.replace(
            star.models()["uniform"],
            name=name,
            parent=parent,
            params=tuple(f"p{i}" for i in range(n_params)),
        )
    return entries


@pytest.mark.parametrize(
    "m2loglike, best, decisive",
    [
        # Each step beats its penalty (ln 81 = 4.4 per parameter): C.
        ({"a": 300.0, "b": 250.0, "c": 200.0}, "c", ("b", "c")),
        # B beats A but C does not beat B: B.
        ({"a": 300.0, "b": 250.0, "c": 248.0}, "b", ("b", "c")),
        # B does not beat A, whatever C does relative to B: A.
        ({"a": 300.0, "b": 298.0, "c": 200.0}, "a", ("a", "b")),
    ],
)
def test_model_choice_walks_the_chain_of_parents(m2loglike, best, decisive):
    registry = _fake_models({"a": (None, 1), "b": ("a", 2), "c": ("b", 3)})
    reports = {k: {"m2loglike": v} for k, v in m2loglike.items()}
    chosen, comparison, comparisons = star._choose_model(reports, registry, 81)
    assert chosen == best
    assert (comparison["simple"], comparison["complex"]) == decisive
    assert [(c["simple"], c["complex"]) for c in comparisons] == [
        ("a", "b"),
        ("b", "c"),
    ]
    assert comparison["bic_penalty"] == pytest.approx(math.log(81))
    assert comparison["delta_m2loglike"] == pytest.approx(
        m2loglike[decisive[0]] - m2loglike[decisive[1]]
    )


def test_model_choice_with_a_single_model_has_no_comparison():
    registry = _fake_models({"a": (None, 1), "b": ("a", 2)})
    best, comparison, comparisons = star._choose_model(
        {"b": {"m2loglike": 5.0}}, registry, 81
    )
    assert (best, comparison, comparisons) == ("b", None, [])


def test_prior_bound_ignores_shape_parameters():
    import numpyro.distributions as dist

    entry = star.models()["limb_darkened"]
    priors = {
        "diam": dist.LogUniform(1.0, 10.0),
        "q1": dist.Uniform(0.0, 1.0),
        "q2": dist.Uniform(0.0, 1.0),
    }
    noise = {"vis_scale": dist.LogUniform(0.1, 10.0)}
    samples = {
        "diam": onp.full(100, 5.0),
        "q1": onp.full(100, 0.999),
        "q2": onp.full(100, 0.5),
        "noise.vis_scale": onp.full(100, 9.99),
    }
    fractions = star._bound_fractions(entry, priors, noise, samples)
    # q1 sits at its bound but is a shape parameter: judged elsewhere.
    assert set(fractions) == {"diam", "noise.vis_scale"}
    assert fractions["diam"] == 0.0
    assert fractions["noise.vis_scale"] == 1.0
    assert set(star._bound_fractions(entry, priors, None, samples)) == {"diam"}


def test_resolution_uses_the_fitted_models_first_null():
    ld = QuadraticLimbDarkenedDisk.from_u(6.0, u1=0.6, u2=0.2)
    ld_null = star._first_null(ld, 6.0)
    assert ld_null > checks.FIRST_NULL * 1.02
    assert star._first_null(UniformDisk(6.0), 6.0) == pytest.approx(
        checks.FIRST_NULL, rel=1e-3
    )
    # A longest baseline between the two nulls: past the uniform-disk null,
    # short of the limb-darkened one.
    freq = _freq(6.0, 1.0) * (1.0 + ld_null / checks.FIRST_NULL) / 2
    uniform = checks.resolution_regime(6.0, freq)
    limb = checks.resolution_regime(6.0, freq, first_null=ld_null)
    assert uniform.status == "pass" and uniform.value > 1.0
    assert limb.status == "warn" and limb.value < 1.0


def _row(diam, delta):
    return {
        "diam_mas": diam,
        "diam_bounds_mas": [diam / 2, diam * 2],
        "delta_chi2": delta,
    }


def test_multimodal_check():
    assert checks.multimodal([_row(6.0, 0.0)]).status == "pass"
    clear = checks.multimodal([_row(6.0, 0.0), _row(9.0, 80.0)])
    assert clear.status == "pass" and clear.value == 80.0
    warn = checks.multimodal([_row(9.0, 3.0), _row(4.0, 12.0), _row(6.0, 0.0)])
    assert warn.status == "warn" and warn.value == 3.0
    assert "9 mas" in warn.message and "6 mas" in warn.message
    json.dumps(warn.to_dict())


def test_find_lobes_bounds_each_minimum():
    axis = onp.geomspace(1.0, 100.0, 1001)
    delta = onp.full(axis.shape, 1e3)
    delta[onp.argmin(abs(axis - 10.0))] = 0.0
    delta[onp.argmin(abs(axis - 20.0))] = 5.0
    lobes = star._find_lobes(axis, delta, 1.0, 100.0, 5)
    near = [lobe for lobe in lobes if lobe["scan_delta_chi2"] < 10]
    assert [round(lobe["diam"]) for lobe in near] == [10, 20]
    assert near[0]["bounds"][1] == pytest.approx(near[1]["bounds"][0])
    assert near[0]["bounds"][1] == pytest.approx(
        math.sqrt(10.0 * 20.0), rel=0.01
    )
    assert len(star._find_lobes(axis, delta, 1.0, 100.0, 1)) == 1


def test_two_lobe_star_reports_the_alias(tmp_path):
    """A star past its first null: the right lobe wins, the alias is listed."""
    template = vlti_oidata(
        wavelengths_m=onp.array([2.0e-6]),
        hour_angles_h=(-1.0, 1.0),
        sigma_v2=0.01,
        sigma_cp_deg=60.0,
    )
    data = template.with_model(UniformDisk(10.0), key=jax.random.PRNGKey(1))
    res = StarPipeline(
        data,
        "uniform",
        output=tmp_path / "run",
        diam_range_mas=[0.5, 60.0],
        n_scan=600,
        quicklook_execute=False,
    ).run(through="fit")
    fit = res.summary["fit"]
    rows = fit["models"]["uniform"]["lobes"]
    assert set(rows[0]) == {
        "diam_bounds_mas",
        "diam_mas",
        "chi2",
        "chi2_reduced",
        "m2loglike",
        "delta_m2loglike",
        "delta_chi2",
    }
    best = min(rows, key=lambda r: r["delta_chi2"])
    assert best["delta_chi2"] == 0.0
    # sigma_v2 = 0.01 on two points: the fit scatters by a few per cent across
    # platforms; this test is about the lobe choice.
    assert best["diam_mas"] == pytest.approx(10.0, rel=0.05)
    low, high = fit["diam_bounds_mas"]
    assert low < 10.0 < high
    alias = [r for r in rows if r["diam_mas"] > 15.0 and r["delta_chi2"] < 25]
    assert alias, rows
    # The check names the closest rival; the noise decides which lobe that is.
    alias.sort(key=lambda r: r["delta_chi2"])
    assert not low < alias[0]["diam_mas"] < high
    assert any(
        r[0] == pytest.approx(alias[0]["diam_mas"]) for r in fit["rivals"]
    )
    check = {c.name: c for c in res.checks}["multimodal"]
    assert check.status == "warn"
    assert f"{alias[0]['diam_mas']:.4g} mas" in check.message
    assert "lobes:" in res.describe()


def test_max_lobes_caps_the_fits(tmp_path, data):
    res = StarPipeline(
        data,
        "uniform",
        output=tmp_path / "run",
        max_lobes=2,
        quicklook_execute=False,
    ).run(through="fit")
    assert len(res.summary["fit"]["models"]["uniform"]["lobes"]) <= 2
    with pytest.raises(ValueError, match="max_lobes"):
        StarPipeline(data, max_lobes=0)


# --- hard walls of the closure-phase likelihood -------------------------------


@pytest.fixture(scope="module")
def tutorial_data():
    """The limb-darkening tutorial's star: 108 V2, 72 closure phases, 0.5 deg."""
    template = vlti_oidata(
        wavelengths_m=onp.linspace(1.6e-6, 2.4e-6, 6),
        hour_angles_h=(-2.5, 0.0, 2.5),
        sigma_v2=0.002,
        sigma_cp_deg=0.5,
    )
    return template.with_model(TRUTH, key=jax.random.PRNGKey(3))


def test_divergences_at_hard_walls_are_explained_not_hidden(
    tutorial_data, tmp_path
):
    """At 0.5 deg closure-phase errors a flip costs ~1e5 in chi^2, so the
    posterior is a cell with hard walls in diameter and in q1, q2 (a
    limb-darkened null moves with q). NUTS rejects trajectories at the walls
    and flags them; the draws still match a brute-force grid posterior
    (diameter 6.039 +- 0.060, q1 0.56 +- 0.23).
    """
    res = StarPipeline(
        tutorial_data,
        "limb_darkened",
        output=tmp_path / "run",
        max_lobes=2,
        num_warmup=200,
        num_samples=150,
        num_chains=2,
        quicklook_execute=False,
    ).run(through="posterior")
    post = res.summary["posterior"]["models"]["limb_darkened"]
    assert post["wall_probe"]["exposed"] >= 0.05
    assert abs(post["params"]["diam"]["median"] - 6.04) < 0.15
    by_name = {c.name: c for c in res.checks}
    assert by_name["chi2_limb_darkened"].status == "pass"
    # The probe finds walls, so divergences are explained and never fail.
    div = by_name["divergences"]
    assert div.status != "fail"
    assert ("hard wall" in div.message) == (post["divergence_fraction"] > 0.01)


def _wall_samples(spread):
    rng = onp.random.default_rng(0)
    return {
        "diam": 6.0 + spread * rng.normal(size=200),
        "q1": float(TRUTH.q1) + spread * rng.normal(size=200),
        "q2": float(TRUTH.q2) + spread * rng.normal(size=200),
    }


def test_wall_probe_separates_narrow_from_wide_posteriors(tutorial_data):
    """With 0.5 deg closure-phase errors a flip costs ~1e5 in chi^2: a
    posterior a thousandth of a milliarcsecond wide never reaches one, one
    as wide as the tutorial star's (0.06 mas) often does."""
    entry = star.models()["limb_darkened"]
    priors = entry.priors(3.0, 12.0)
    probe = in_float64(star._wall_probe)  # as in the sampler stage
    narrow = probe(entry, tutorial_data, _wall_samples(1e-3), priors, n=20)
    wide = probe(entry, tutorial_data, _wall_samples(0.06), priors, n=20)
    assert narrow["n"] > 0 and narrow["exposed"] == 0.0
    assert wide["n"] > 0 and wide["exposed"] > 0.0


@pytest.mark.parametrize(
    "fraction, exposed, status",
    [(0.0, 0.0, "pass"), (0.4, 0.0, "fail"), (0.4, 0.5, "warn")],
)
def test_divergences_check_with_walls(fraction, exposed, status):
    walls = {"radius_sd": 2.0, "n": 40, "exposed": exposed}
    check = checks.divergences(fraction, walls=walls)
    assert check.status == status
    assert ("hard wall" in check.message) == (status == "warn" and exposed > 0)
    json.dumps(check.to_dict())


def test_unconverged_lm_message_says_why():
    from virgil import fitting

    failed = fitting._not_converged("lm", "failed", 0, 1000, "float32", None)
    assert "0 steps, before its limit of 1000" in failed
    assert "no step lowered the loss" in failed
    limit = fitting._not_converged("lm", "limit", 1000, 1000, "float32", None)
    assert "step limit" in limit


@pytest.mark.slow
def test_unresolved_star_starts_inside_the_diameter_prior(tmp_path):
    # An unresolved star's diameter runs to the lower edge of its range; NUTS
    # must start just inside it (numpyro's init fails on the bound itself).
    template = vlti_oidata(
        wavelengths_m=onp.linspace(1.6e-6, 2.4e-6, 3),
        hour_angles_h=(-2.5, 0.0, 2.5),
        sigma_v2=0.01,
        sigma_cp_deg=15.0,
    )
    data = template.with_model(UniformDisk(1e-4), key=jax.random.PRNGKey(5))
    result = StarPipeline(data, output=tmp_path / "run", **TINY).run()
    assert result.status == "complete"
    check = {c.name: c for c in result.checks}["start_at_bound"]
    assert check.status == "warn"
    assert any(name.endswith(".diam") for name in check.value)
