"""ImagingPipeline on tiny synthetic data: API contract, checks, FITS writer,
stages, round trip, resume, quicklook caveat and the CLI.

The scene is a star (0.6 of the flux) and an inclined Gaussian ring, seen by
four telescopes in one channel at three hour angles: V² and closure phases,
N = 27 independent data. The grid is 11 pixels on a side, the evidence grid a
single point and every fit is capped at 20 steps, so these tests check the
plumbing and the contract, not the optimizer. The pipeline's stages are run
progressively (``run(through=...)`` resumes), so that the time of the compiles
is shared among a few tests, which are all marked slow: whichever of them
runs first pays for the compiles (30-35 s cold on a CI runner, over the budget).
The fits that are not needed for the fast tier,
the star-free scene, a classical regularizer and the executed quicklook, are
marked slow.
"""

import json

import jax
import numpy as onp
import pytest

pytest.importorskip("h5py")
pytest.importorskip("nbclient")

from virgil.coverage import vlti_oidata  # noqa: E402
from virgil.metrics import align, ncc  # noqa: E402
from virgil.models import (  # noqa: E402
    GaussianRing,
    MultipoleDisk,
    PointSource,
    System,
)
from virgil.pipeline import (  # noqa: E402
    SCHEMA,
    Check,
    ConfigMismatchError,
    ImagingPipeline,
    SkyImage,
    _checks,
    _io,
    cli,
    image,
    load,
)
from virgil.pipeline import _quicklook  # noqa: E402

pytestmark = pytest.mark.xdist_group("pipeline_image")

RING = MultipoleDisk(
    GaussianRing(2.0, 0.7), inc=50.0, pa=30.0, flux=0.4, n_r=48
)
TRUTH = System(star=PointSource(flux=0.6), ring=RING)
SMALL = {
    "star": True,
    "max_npix": 16,
    "oversample": 1.5,
    "sigmas": [2.0],
    "lengths_beam": [1.0],
    "grid_growths": 0,
    "members": 2,
    "max_steps": 20,
    "quicklook_execute": False,
}


def _simulate(truth):
    template = vlti_oidata(
        wavelengths_m=onp.array([1.65e-6]),
        hour_angles_h=(-2.0, 0.0, 2.0),
        sigma_v2=0.01,
        sigma_cp_deg=2.0,
    )
    return template.with_model(truth, key=jax.random.PRNGKey(3))


@pytest.fixture(scope="module")
def data():
    return _simulate(TRUTH)


@pytest.fixture(scope="module")
def pipe(data, tmp_path_factory):
    out = tmp_path_factory.mktemp("image") / "run"
    return ImagingPipeline(data, output=out, **SMALL)


# --- the contract --------------------------------------------------------------

API = {
    "ImagingPipeline": "(data, model=None, *, output='run', **settings)",
    "ImagingPipeline.run": "(self, through=None, *, resume=True)",
    "ImagingPipeline.from_oifits": "(paths, model=None, *, output='run', **settings)",
    "ImagingPipeline.from_config": "(path, model=None, **overrides)",
    "ImagingPipeline.defaults": "()",
}
STAGES = (
    "load",
    "overview",
    "star",
    "hyperparameters",
    "members",
    "combine",
    "quicklook",
)
DEFAULTS = {
    "oversample": 3.0,
    "field_fwhms": 6.0,
    "field_mas": None,
    "max_npix": 128,
    "star": "auto",
    "star_threshold": 5.0,
    "sigmas": [1.0, 2.0, 4.0],
    "lengths_beam": [0.5, 1.0, 2.0],
    "grid_growths": None,
    "members": 8,
    "member_grids": False,
    "method": None,
    "max_steps": None,
    "seed": 0,
    "wavel_range": None,
    "error_floor": None,
    "quicklook_execute": True,
}
CHECKS = (
    "chi2_reduced",
    "chi2_reduced_vis",
    "chi2_reduced_phi",
    "quoted_errors",
    "error_scale",
    "effective_parameters",
    "short_baselines",
    "star",
    "orientation",
    "edges",
    "ensemble_kept",
    "ensemble_spread",
)


def test_pipeline_specific_contract():
    # The shared part (signatures, stages, defaults) is test_api_snapshot
    # in test_pipeline_robust_core.py.
    assert image.CHECKS == CHECKS


def test_exported_lazily_and_registered_with_the_cli():
    import virgil.pipeline as vp

    assert vp.ImagingPipeline is ImagingPipeline
    assert "ImagingPipeline" in vp.__all__ and "SkyImage" in vp.__all__
    args = cli.build_parser().parse_args(["image", "d.oifits", "-o", "out"])
    assert args.command == "image" and args.output == "out"
    assert cli._PIPELINES["image"] == (
        "virgil.pipeline.image",
        "ImagingPipeline",
    )


def test_settings_are_validated(data):
    with pytest.raises(TypeError, match="unknown settings"):
        ImagingPipeline(data, n_members=4)
    with pytest.raises(ValueError, match='model must be "gp"'):
        ImagingPipeline(data, "rings")
    for bad in (
        {"star": "maybe"},
        {"oversample": -1.0},
        {"max_npix": 2},
        {"sigmas": []},
        {"members": 0},
        {"members": 9},
        {"wavel_range": [1e-6]},
        {"error_floor": 0.1},
    ):
        with pytest.raises(ValueError):
            ImagingPipeline(data, **bad)


# --- the FITS writer -----------------------------------------------------------


def test_fits_orientation_round_trip(tmp_path):
    """A point source East and North of the centre reads back, through
    astropy's WCS, at positive RA and Dec offsets."""
    from astropy.io import fits
    from astropy.wcs import WCS

    n, scale = 9, 2.0
    pixels = onp.zeros((n, n))
    # Row 0 is North and column 0 East: three rows North and two columns
    # East of the centre pixel (4, 4).
    pixels[1, 2] = 1.0
    path = tmp_path / "a.fits"
    _io.write_fits_image(
        path,
        pixels,
        scale,
        dra=10.0,
        ddec=-5.0,
        header={"NMEMBERS": 1},
    )
    header = fits.getheader(path)
    assert header["CTYPE1"] == "RA---SIN" and header["CTYPE2"] == "DEC--SIN"
    assert header["CDELT1"] < 0 < header["CDELT2"]
    assert header["BUNIT"] == "relative flux" and header["NMEMBERS"] == 1
    wcs = WCS(header)
    row, col = onp.unravel_index(pixels.argmax(), pixels.shape)
    # FITS rows run South to North.
    ra, dec = wcs.wcs_pix2world(col, n - 1 - row, 0)
    ra_mas = (((ra + 180.0) % 360.0) - 180.0) * 3.6e6
    dec_mas = dec * 3.6e6
    assert ra_mas == pytest.approx(10.0 + 2 * scale, abs=1e-6)  # East: +RA
    assert dec_mas == pytest.approx(-5.0 + 3 * scale, abs=1e-6)  # North
    back = _io.read_fits_image(path)
    assert isinstance(back, SkyImage)
    onp.testing.assert_array_equal(back.pixels, pixels)
    assert back.pixel_scale_mas == pytest.approx(scale)
    assert (back.dra, back.ddec) == pytest.approx((10.0, -5.0))


# --- the checks, each branch ---------------------------------------------------


def _status(check):
    assert isinstance(check, Check)
    return check.status


def test_quoted_errors_check():
    ok = {
        "min": 0.01,
        "median": 0.02,
        "max": 0.05,
        "all_equal": False,
        "n_bad": 0,
    }
    same = dict(ok, all_equal=True)
    zero = dict(ok, n_bad=3)
    assert _status(_checks.quoted_errors({"vis": ok, "phi": ok})) == "pass"
    check = _checks.quoted_errors({"vis": ok, "phi": same})
    assert check.status == "warn" and check.value == ["phi"]
    assert "all equal in phi" in check.message
    check = _checks.quoted_errors({"vis": zero, "phi": ok})
    assert (
        check.status == "warn" and "zero or not finite in vis" in check.message
    )


def test_effective_parameters_check():
    f = _checks.effective_parameters
    assert f(56, 53.0).status == "warn"  # the 2022 AMI case
    assert "parametric model" in f(56, 53.0).message
    assert f(56, 20.0).status == "pass"


def test_short_baselines_check():
    f = _checks.short_baselines
    assert f(0.1, 0.5).status == "pass"
    assert f(-1.8, 0.5).status == "warn"  # -3.6 sigma
    big = f(-4.0, 0.5)
    assert big.status == "fail" and big.value == pytest.approx(-8.0)
    assert "misses flux" in big.message
    assert f(None, None).value is None


def test_edges_check():
    f = _checks.edges
    assert f([], 0.003).status == "pass"
    edge = f(["length max"], 0.003)
    assert edge.status == "warn" and "length max" in edge.message
    assert f([], 0.03).status == "warn"
    assert f([], float("nan")).status == "warn"


def test_star_check():
    f = _checks.star_decision
    assert f(None, fixed=True).status == "pass"
    assert f(22.0, decision="on").status == "pass"
    assert f(-9.0, decision="off").status == "pass"
    both = f(2.0, decision="both")
    assert both.status == "warn" and "undecided" in both.message
    assert "Occam" in f(22.0, decision="on").message


def test_orientation_check():
    f = _checks.orientation
    assert f(0.0, closure=False).status == "warn"
    assert f(0.0, closure=False).value == 0.0
    assert f(4.0, closure=True).status == "warn"
    assert f(32.0, closure=True).status == "pass"


def _reports():
    kinds = {"vis": 1.0, "phi": 1.1, "total": 1.05}
    equal = {"min": 1, "median": 1, "max": 1, "all_equal": False, "n_bad": 0}
    return {
        "load": {
            "n_independent": 30,
            "n_by_observable": {"vis": 20, "phi": 10},
            "closure_phases": True,
            "quoted_errors": {"vis": equal, "phi": equal},
            "flagged_fraction": 0.0,
            "n_channels": 1,
        },
        "star": {
            "decision": "on",
            "delta_log_z": 12.0,
        },
        "hyperparameters": {
            "scale_by_observable": {"vis": 8.0, "phi": 1.0},
            "gamma": 12.0,
            "edge": ["sigma max"],
        },
        "combine": {
            "chi2_red_by_observable": kinds,
            "short_baseline_mean": 0.1,
            "short_baseline_error": 0.5,
            "flip_dchi2": 30.0,
            "edge_flux": 0.001,
            "n_kept": 5,
            "n_members": 8,
            "spread": 0.2,
        },
    }


def test_ensemble_checks():
    kept = _checks.ensemble_kept
    assert kept(5, 8).status == "pass"
    assert kept(2, 4).status == "warn"  # fewer than min_kept
    assert kept(3, 8).status == "warn"  # under half
    spread = _checks.ensemble_spread
    assert spread(0.2, 5).status == "pass"
    assert spread(0.8, 5).status == "warn" and spread(0.0, 1).status == "warn"
    assert "excludes hyperparameter uncertainty" in spread(0.2, 5).message
    assert "excludes" not in spread(0.2, 5, shared=False).message


def test_summary_orders_the_checks():
    settings = ImagingPipeline.defaults()
    checks = image._run_checks(settings, _reports())
    assert [c.name for c in checks] == list(CHECKS)
    named = {c.name: c for c in checks}
    assert named["error_scale"].status == "fail"  # a scale of 8
    assert named["edges"].status == "warn"
    # A run that stopped after the load stage has only the data check.
    early = image._run_checks(settings, {"load": _reports()["load"]})
    assert [c.name for c in early] == ["quoted_errors"]


# --- the stages ----------------------------------------------------------------


@pytest.mark.slow
def test_load_overview_star_and_hyperparameter_stages(pipe):
    res = pipe.run(through="hyperparameters")
    assert res.status == "partial"
    load_report = pipe._report("load")
    assert load_report["n_independent"] == 27
    assert load_report["n_by_observable"] == {"vis": 18, "phi": 9}
    assert load_report["closure_phases"]
    errors = load_report["quoted_errors"]
    assert errors["vis"]["all_equal"] and errors["phi"]["all_equal"]
    field = pipe._report("overview")
    assert field["npix"] <= 16
    assert field["beam"]["major_mas"] >= field["beam"]["minor_mas"] > 0
    star = pipe._report("star")
    assert star["decision"] == "on" and star["delta_log_z"] is None
    hyper = pipe._report("hyperparameters")
    assert hyper["variant"] in ("iso", "aniso") and hyper["sigma"] == 2.0
    assert hyper["length_beam"] == pytest.approx(1.0)
    assert set(hyper["scale_by_observable"]) == {"vis", "phi"}
    assert hyper["gamma"] > 0 and onp.isfinite(hyper["log_z"])
    grids = _io.read_h5(
        pipe.output / "grids.h5", f"evidence_{hyper['variant']}"
    )
    assert grids["log_evidence"].shape == (1, 1)
    assert set(res.model().components) == {"star", "env"}
    assert (pipe.output / "data" / "processed.oifits").exists()


@pytest.mark.slow
def test_combine_reports_raw_chi2_by_observable(pipe, data):
    res = pipe.run(through="combine")
    summary = res.summary
    assert summary["schema"] == SCHEMA and summary["pipeline"] == "image"
    for key in ("data", "field", "star", "hyperparameters", "fit", "checks"):
        assert key in summary, key
    fit = summary["fit"]
    by = fit["chi2_red_by_observable"]
    # Raw χ²/N of the V² and closure-phase blocks, on the quoted errors.
    assert set(by) == {"vis", "phi", "total"}
    n = fit["n_by_observable"]
    assert n["total"] == data.n_independent == 27
    assert by["total"] == pytest.approx(fit["chi2_reduced"])
    assert fit["chi2"] == pytest.approx(by["total"] * n["total"])
    assert (n["vis"] * by["vis"] + n["phi"] * by["phi"]) == pytest.approx(
        fit["chi2"]
    )
    named = {c["name"]: c for c in summary["checks"]}
    assert [c["name"] for c in summary["checks"]][: len(CHECKS)] == list(
        CHECKS
    )
    # The simulated errors are one number per block: the data check fires.
    assert named["quoted_errors"]["status"] == "warn"
    assert named["star"]["status"] == "pass"
    assert named["chi2_reduced"]["value"] == pytest.approx(by["total"])


@pytest.mark.slow
def test_mean_image_recovers_the_truth_and_round_trips(pipe):
    res = pipe.run(through="combine")
    img = res.image()
    n, scale = img.pixels.shape[0], img.pixel_scale_mas
    # The extended emission carries its share of the flux; the star the rest.
    assert img.pixels.sum() == pytest.approx(1.0 - img.header["STARFRAC"])
    assert img.header["STARFRAC"] == pytest.approx(0.6, abs=0.1)
    truth = onp.asarray(RING.render(n, n * scale))
    shifted, _ = align(img.pixels, truth, 2 * scale, scale)
    assert float(ncc(shifted, truth)) > 0.75
    assert load(pipe.output).run["schema"] == SCHEMA


@pytest.mark.slow
def test_the_quicklook_says_what_it_cannot_claim(pipe):
    pipe.run(through="combine")
    nb = _quicklook.build_notebook(pipe.output)
    first = nb.cells[0]
    assert first.cell_type == "markdown"
    assert "does **not** promise that the morphology is right" in first.source
    assert "chain of dots" in first.source
    code = "\n".join(c.source for c in nb.cells if c.cell_type == "code")
    assert "res.image()" in code


@pytest.mark.slow
def test_validate_resume_and_config_mismatch(pipe, data):
    assert cli.main(["validate", str(pipe.output)]) == 0
    pipe.run(through="combine")  # resumes: every stage is skipped
    run = json.loads((pipe.output / "run.json").read_text())
    assert run["pipeline"] == "image" and run["class"] == "ImagingPipeline"
    assert all(
        run["stages"][s]["status"] == "complete"
        for s in STAGES
        if s != "quicklook"
    )
    other = ImagingPipeline(data, output=pipe.output, **{**SMALL, "seed": 1})
    with pytest.raises(ConfigMismatchError, match="seed"):
        other.run()


@pytest.mark.slow
def test_cli_runs_the_image_pipeline(pipe, tmp_path, capsys):
    path = pipe.output / "data" / "processed.oifits"
    out = tmp_path / "cli"
    args = ["image", str(path), "-o", str(out), "--through", "load"]
    args += ["--set", "star=true", "--set", "max_npix=16"]
    assert cli.main(args) == 0
    run = json.loads((out / "run.json").read_text())
    assert run["pipeline"] == "image" and run["status"] == "partial"
    assert run["config"]["max_npix"] == 16 and run["config"]["star"] is True
    assert "data.n_independent" in capsys.readouterr().out


# Two compiled evidence grids on an already tiny problem (16 pixels, one
# hyperparameter, 20 steps): the time is the compiles, not the problem size,
# so shrinking it does not help (7.9 s warm, 30.3 s on a cold CI runner).
@pytest.mark.slow
def test_star_auto_decides_the_star_on(data, tmp_path):
    settings = {**SMALL, "star": "auto"}
    pipeline = ImagingPipeline(data, output=tmp_path / "run", **settings)
    pipeline.run(through="star")
    star = pipeline._report("star")
    assert star["delta_log_z"] == pytest.approx(
        star["log_z_star"] - star["log_z_no_star"]
    )
    assert star["delta_log_z"] >= 5.0 and star["decision"] == "on"
    assert star["member_star"] is True


@pytest.mark.slow
def test_members_are_resumable_and_fill_the_ensemble(data, tmp_path):
    """The array workflow: members out of order, then the stages finish."""
    pipe = ImagingPipeline(data, output=tmp_path / "run", **SMALL)
    pipe.run(through="hyperparameters")
    pipe.run_member(1)
    info = pipe.output / "members" / "001" / "info.json"
    stamp = info.stat().st_mtime_ns
    pipe.run_member(1)  # saved: not fitted again
    assert info.stat().st_mtime_ns == stamp
    with pytest.raises(ValueError, match="k must be"):
        pipe.run_member(2)
    res = pipe.run(through="combine")
    assert info.stat().st_mtime_ns == stamp
    assert (pipe.output / "members" / "000" / "info.json").exists()
    fit = res.summary["fit"]
    assert fit["n_members"] == 2 and 1 <= fit["n_kept"] <= 2
    names = [c["name"] for c in res.summary["checks"]]
    assert names[: len(CHECKS)] == list(CHECKS)
    assert set(fit["chi2_red_mixture"]) == {"vis", "phi"}
    # The members share the hyperparameters of the one evidence search.
    hyper = pipe._report("hyperparameters")
    spec, _, draws, variant = image._member_setup(pipe)
    assert spec.gp_sigmas == (hyper["sigma"],) and spec.gp_growths == 0
    assert spec.gp_lengths == (hyper["length_beam"],)
    assert variant == hyper["variant"] and set(draws) == {0, 1}
    mean, sigma = res.image(), res.image("sigma")
    assert mean.pixels.shape == sigma.pixels.shape
    assert sigma.pixels.max() > 0  # the two members start differently
    # Raw χ²/N of the mean image, V² and closure phases apart.
    assert set(fit["chi2_red_by_observable"]) == {"vis", "phi", "total"}


@pytest.mark.slow
def test_member_grids_fit_each_member_its_own_hyperparameters(data, tmp_path):
    settings = {**SMALL, "members": 1, "member_grids": True}
    pipe = ImagingPipeline(data, output=tmp_path / "run", **settings)
    pipe.run(through="members")
    assert (pipe.output / "members" / "000" / "info.json").exists()
    spec, _, _, variant = image._member_setup(pipe)
    assert variant is None and spec.gp_sigmas == (2.0,)  # the setting's grid
    other = ImagingPipeline(
        data, output=pipe.output, **{**settings, "max_steps": 5}
    )
    with pytest.raises(ConfigMismatchError):
        other.run_member(0)


# --- slower fits ---------------------------------------------------------------


@pytest.mark.slow
def test_star_auto_decides_the_star_off_without_one(tmp_path):
    ring = System(env=RING)
    data = _simulate(ring)
    pipeline = ImagingPipeline(
        data, output=tmp_path / "run", **{**SMALL, "star": "auto"}
    )
    pipeline.run(through="star")
    star = pipeline._report("star")
    assert star["delta_log_z"] <= -5.0 and star["decision"] == "off"
    assert star["member_star"] is False


@pytest.mark.slow
def test_the_executed_quicklook(data, tmp_path):
    settings = {**SMALL, "quicklook_execute": True}
    pipeline = ImagingPipeline(data, output=tmp_path / "run", **settings)
    res = pipeline.run()
    assert res.status == "complete"
    notebook = json.loads((pipeline.output / "quicklook.ipynb").read_text())
    outputs = [c.get("outputs") for c in notebook["cells"] if "outputs" in c]
    assert outputs and all(
        o["output_type"] != "error" for out in outputs for o in out
    )
    assert "does **not** promise" in "".join(notebook["cells"][0]["source"])


def test_grid_growths_none_is_the_engine_default():
    from virgil.ensemble import EnsembleSpec

    s = dict(image.ImagingPipeline.defaults())
    assert image._grid_spec(s, None).gp_growths == EnsembleSpec().gp_growths
    s["grid_growths"] = 1
    assert image._grid_spec(s, None).gp_growths == 1
