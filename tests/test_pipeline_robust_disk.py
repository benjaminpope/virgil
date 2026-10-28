"""DiskPipeline robustness: resume and fresh runs stage by stage, failure paths
that must not pass silently, and the null case (a star with no disk).

Complements ``test_pipeline_disk.py``, which covers the API snapshot, the
settings, the golden runs and one resume. Here:

* the resume and fresh-run rules, for each stage of both models, with the
  stage functions replaced by recorders (the machinery is the base class's;
  what is per stage is which outputs a stage owns and what it discards);
* bad inputs, an empty selection, a ring on a prior bound, and a posterior
  that has not converged, each with a clear error or a check that is not
  ``pass``;
* data from a point source: no disk may be reported silently.

The real runs use the ``tiny`` problem of the sibling file (three channels,
two hour angles, one ring count, one length scale). Those that take more than
about 30 s are marked slow.
"""

import json
import shutil

import jax
import numpy as onp
import pytest

pytest.importorskip("h5py")
pytest.importorskip("nbclient")

from virgil.coverage import vlti_oidata  # noqa: E402
from virgil.models import (  # noqa: E402
    GaussianRing,
    Harmonics,
    MultipoleDisk,
    PointSource,
    System,
)
from virgil.oifits import write_oifits  # noqa: E402
from virgil.pipeline import ConfigMismatchError, DiskPipeline  # noqa: E402
from virgil.pipeline import _io, disk  # noqa: E402

pytestmark = pytest.mark.xdist_group("pipeline_disk")

MODELS = ("rings", "gp")
# The stages are stubbed, so the stage-by-stage tests need only the first model.
STAGES = DiskPipeline.STAGES
# The function of ``virgil.pipeline.disk`` that each stage runs.
STAGE_FUNCTIONS = dict(
    zip(
        STAGES,
        (
            "_load_disk",
            "_overview",
            "_geometry_stage",
            "_profile",
            "_compare",
            "_posterior",
            "_quicklook",
        ),
    )
)

TRUTH = System(
    star=PointSource(flux=1.0),
    ring=MultipoleDisk(
        GaussianRing(5.0, 1.0),
        inc=50.0,
        pa=30.0,
        flux=0.4,
        n_r=48,
        harmonics=Harmonics(re=(0.4,), im=(0.2,)),
    ),
)
# The smallest problem that runs every stage: a coarse geometry grid, one
# ring count, one length scale, no companion, no harmonics.
TINY = {
    "n_inc": 4,
    "n_pa": 4,
    "max_rings": 1,
    "r_out_mas": 12.0,
    "ell_halffringes": [1.5],
    "companion": False,
    "harmonic_degree": 0,
    "quicklook_execute": False,
}
# Two chains of a few draws: far too short to have converged.
SHORT_NUTS = {
    "posterior": True,
    "num_warmup": 5,
    "num_samples": 8,
    "num_chains": 2,
}


def _template():
    return vlti_oidata(
        wavelengths_m=onp.linspace(1.6e-6, 2.4e-6, 3),
        hour_angles_h=(-1.5, 1.5),
        sigma_v2=0.01,
        sigma_cp_deg=3.0,
    )


@pytest.fixture(scope="module")
def tiny():
    return _template().with_model(TRUTH, key=jax.random.PRNGKey(3))


@pytest.fixture(scope="module")
def point_source():
    """Data from the star alone: there is no disk to find."""
    return _template().with_model(
        PointSource(flux=1.0), key=jax.random.PRNGKey(7)
    )


@pytest.fixture(scope="module")
def outputs(tiny):
    """The files each stage must leave behind, from the stage table."""
    return {s.name: s.outputs for s in DiskPipeline(tiny)._stages()}


# --- stage by stage: resume and fresh runs ----------------------------------------


class _Stages:
    """Replaces every stage function with a recorder that writes the stage's
    declared outputs, so the base class's resume rules are exercised alone.

    ``ran`` lists the stages run, in order. ``stale[name]`` lists the outputs
    of that stage and of the later ones found on disk when it started: they
    must have been discarded already. ``fail_once`` names a stage that raises
    on its next call.
    """

    def __init__(self, monkeypatch, outputs):
        self.outputs = outputs
        self.ran, self.stale, self.fail_once = [], {}, None
        self.calls = 0
        # No summary: the stage reports are empty.
        monkeypatch.setattr(
            DiskPipeline, "_summarize", lambda self, reports: ({}, [])
        )
        for name, function in STAGE_FUNCTIONS.items():
            monkeypatch.setattr(disk, function, self._recorder(name))

    def _recorder(self, name):
        later = STAGES[STAGES.index(name) :]

        def stage(pipeline):
            out = pipeline.output
            self.stale[name] = [
                path
                for other in later
                for path in self.outputs[other]
                if (out / path).exists()
            ]
            self.ran.append(name)
            self.calls += 1
            if self.fail_once == name:
                self.fail_once = None
                raise RuntimeError(f"{name} stopped")
            for path in self.outputs[name]:
                (out / path).parent.mkdir(parents=True, exist_ok=True)
                (out / path).write_text(f"{name} call {self.calls}")
            return {}

        return stage

    def pipeline(self, data, model, folder, **settings):
        return DiskPipeline(data, model, output=folder, **{**TINY, **settings})

    def take(self):
        """The stages run since the last call."""
        ran, self.ran = self.ran, []
        return ran


@pytest.fixture
def stages(monkeypatch, outputs):
    return _Stages(monkeypatch, outputs)


def _run_json(folder):
    return json.loads((folder / "run.json").read_text())


def _present(folder, outputs, names):
    return [
        path for n in names for path in outputs[n] if (folder / path).exists()
    ]


@pytest.mark.parametrize("through", STAGES)
@pytest.mark.parametrize("model", MODELS)
def test_resume_recomputes_nothing_for_completed_stages(
    stages, tiny, tmp_path, model, through
):
    folder = tmp_path / "run"
    first = stages.pipeline(tiny, model, folder).run()
    assert first.status == "complete" and stages.take() == list(STAGES)
    started = {k: v["started"] for k, v in _run_json(folder)["stages"].items()}

    res = stages.pipeline(tiny, model, folder).run(through=through)
    assert stages.take() == []
    res = stages.pipeline(tiny, model, folder).run()
    assert res.status == "complete" and stages.take() == []
    after = {k: v["started"] for k, v in _run_json(folder)["stages"].items()}
    assert after == started


@pytest.mark.parametrize("stage", STAGES)
@pytest.mark.parametrize("model", MODELS[:1])
def test_a_missing_output_reruns_that_stage_and_every_later_one(
    stages, outputs, tiny, tmp_path, model, stage
):
    folder = tmp_path / "run"
    stages.pipeline(tiny, model, folder).run()
    stages.take()
    before = _run_json(folder)["stages"]
    i = STAGES.index(stage)
    (folder / outputs[stage][0]).unlink()

    res = stages.pipeline(tiny, model, folder).run()
    assert stages.take() == list(STAGES[i:])
    assert res.status == "complete"
    # Each rerun stage found its own and later outputs already discarded.
    assert all(stages.stale[n] == [] for n in STAGES[i:])
    after = _run_json(folder)["stages"]
    for name in STAGES[:i]:
        assert after[name]["started"] == before[name]["started"]
    for name in STAGES[i:]:
        assert after[name]["started"] >= before[name]["started"]
    assert len(_present(folder, outputs, STAGES)) == sum(
        len(outputs[n]) for n in STAGES
    )


@pytest.mark.parametrize("stage", STAGES[:-1])
@pytest.mark.parametrize("model", MODELS[:1])
def test_rerunning_a_stage_removes_the_outputs_of_the_later_ones(
    stages, outputs, tiny, tmp_path, model, stage
):
    # run(through=stage) stops there, but the later stages were computed from
    # the stage's old outputs: they are stale, so they go, and the run is
    # partial rather than a complete run with mixed vintages.
    folder = tmp_path / "run"
    stages.pipeline(tiny, model, folder).run()
    stages.take()
    i = STAGES.index(stage)
    (folder / outputs[stage][0]).unlink()

    res = stages.pipeline(tiny, model, folder).run(through=stage)
    assert stages.take() == [stage]
    assert res.status == "partial"
    assert _present(folder, outputs, STAGES[i + 1 :]) == []
    assert list(_run_json(folder)["stages"]) == list(STAGES[: i + 1])


@pytest.mark.parametrize("stage", STAGES)
@pytest.mark.parametrize("model", MODELS[:1])
def test_a_fresh_run_recomputes_the_stage_and_clears_the_later_ones(
    stages, outputs, tiny, tmp_path, model, stage
):
    folder = tmp_path / "run"
    stages.pipeline(tiny, model, folder).run()
    stages.take()
    i = STAGES.index(stage)
    old = {p: (folder / p).read_text() for p in outputs[stage]}
    (folder / "plots").mkdir(exist_ok=True)
    (folder / "plots" / "stale.png").write_text("a figure of the old run")

    res = stages.pipeline(tiny, model, folder).run(through=stage, resume=False)
    # Nothing is resumed: the stage and every one before it ran again.
    assert stages.take() == list(STAGES[: i + 1])
    assert all(stages.stale[n] == [] for n in STAGES[: i + 1])
    assert _present(folder, outputs, STAGES[i + 1 :]) == []
    assert not (folder / "plots" / "stale.png").exists()
    assert list(_run_json(folder)["stages"]) == list(STAGES[: i + 1])
    assert res.status == ("complete" if stage == STAGES[-1] else "partial")
    # The stage's outputs were written by this run, not kept from the last.
    for path, text in old.items():
        if not path.endswith("report.json"):
            assert (folder / path).read_text() != text


@pytest.mark.parametrize("model", MODELS)
def test_a_fresh_run_accepts_settings_that_resume_refuses(
    stages, tiny, tmp_path, model
):
    folder = tmp_path / "run"
    stages.pipeline(tiny, model, folder).run()
    stages.take()
    with pytest.raises(ConfigMismatchError, match="n_inc"):
        stages.pipeline(tiny, model, folder, n_inc=6).run()
    assert stages.take() == []  # refused before any stage ran
    stages.pipeline(tiny, model, folder, n_inc=6).run(resume=False)
    assert stages.take() == list(STAGES)
    assert _run_json(folder)["config"]["n_inc"] == 6


@pytest.mark.parametrize("failing", STAGES)
@pytest.mark.parametrize("model", MODELS[:1])
def test_a_failed_stage_is_retried_alone_and_recorded(
    stages, outputs, tiny, tmp_path, model, failing
):
    folder = tmp_path / "run"
    i = STAGES.index(failing)
    stages.fail_once = failing
    with pytest.raises(RuntimeError, match=f"{failing} stopped"):
        stages.pipeline(tiny, model, folder).run()
    assert stages.take() == list(STAGES[: i + 1])
    run = _run_json(folder)
    assert run["status"] == "failed"
    assert run["error"]["stage"] == failing
    assert run["error"]["type"] == "RuntimeError"
    assert "stopped" in run["error"]["message"]
    assert run["stages"][failing]["status"] == "failed"
    assert list(run["stages"]) == list(STAGES[: i + 1])
    assert all(run["stages"][n]["status"] == "complete" for n in STAGES[:i])

    res = stages.pipeline(tiny, model, folder).run()
    # The stages that finished are kept; the failed one and the rest run.
    assert stages.take() == list(STAGES[i:])
    assert res.status == "complete" and _run_json(folder)["error"] is None


def test_through_must_name_a_stage_and_changes_nothing_otherwise(
    stages, tiny, tmp_path
):
    folder = tmp_path / "run"
    with pytest.raises(ValueError, match="through must be one of"):
        stages.pipeline(tiny, "gp", folder).run(through="fit")
    assert stages.take() == [] and not folder.exists()


def test_resume_of_a_real_gp_run_keeps_the_geometry_stage(
    tiny, tmp_path, monkeypatch
):
    # No stubs: the geometry stage really leaves every output it declares, so
    # a resumed Gaussian-process run does not repeat it.
    monkeypatch.setattr(disk, "_GEOMETRY_LATTICE_INC", ())
    monkeypatch.setattr(disk, "_GEOMETRY_LATTICE_PA", ())
    settings = {**TINY, "n_inc": 3, "r_out_mas": 9.0}
    folder = tmp_path / "run"
    res = DiskPipeline(tiny, "gp", output=folder, **settings).run(
        through="geometry"
    )
    assert res.status == "partial"
    started = _run_json(folder)["stages"]
    ran = []
    for name in ("_load_disk", "_overview", "_geometry_stage"):
        monkeypatch.setattr(disk, name, lambda p, n=name: ran.append(n))
    res = DiskPipeline(tiny, "gp", output=folder, **settings).run(
        through="geometry"
    )
    assert ran == [] and res.status == "partial"
    after = _run_json(folder)["stages"]
    assert {k: v["started"] for k, v in after.items()} == {
        k: v["started"] for k, v in started.items()
    }
    assert (folder / "models" / "geometry" / "values.npz").exists()


# --- bad inputs ----------------------------------------------------------------


@pytest.fixture(scope="module")
def junk(tmp_path_factory):
    from astropy.io import fits

    folder = tmp_path_factory.mktemp("junk")
    (folder / "notes.fits").write_text("these are not FITS cards")
    (folder / "empty.fits").write_bytes(b"")
    fits.HDUList([fits.PrimaryHDU()]).writeto(folder / "primary_only.fits")
    return folder


@pytest.mark.parametrize(
    ("name", "error", "message"),
    [
        ("missing.fits", FileNotFoundError, "missing.fits"),
        ("notes.fits", OSError, "FITS"),
        ("empty.fits", OSError, "Empty or corrupt FITS"),
        ("primary_only.fits", ValueError, "no OI_WAVELENGTH table"),
        (".", OSError, "directory"),
    ],
)
def test_from_oifits_rejects_a_bad_file_with_a_clear_error(
    junk, tmp_path, name, error, message
):
    output = tmp_path / "run"
    with pytest.raises(error, match=message):
        DiskPipeline.from_oifits(junk / name, output=output)
    assert not output.exists()  # nothing was written


def test_from_oifits_rejects_a_list_with_one_bad_file(junk, tiny, tmp_path):
    good = tmp_path / "good.oifits"
    write_oifits(tiny, good)
    with pytest.raises(OSError, match="FITS"):
        DiskPipeline.from_oifits(
            [good, junk / "notes.fits"], output=tmp_path / "run"
        )


def test_from_oifits_without_files_raises(tmp_path):
    # An empty list has no data: it must raise, not build a pipeline.
    with pytest.raises(ValueError, match="at least one OIFITS file"):
        DiskPipeline.from_oifits([], output=tmp_path / "run")


def test_from_oifits_reads_a_good_file_and_runs_the_load_stage(tiny, tmp_path):
    # The positive control for the failures above.
    path = tmp_path / "tiny.oifits"
    write_oifits(tiny, path)
    pipeline = DiskPipeline.from_oifits(path, output=tmp_path / "run", **TINY)
    res = pipeline.run(through="load")
    assert res.status == "partial"
    assert res.run["inputs"]["files"][0]["path"] == str(path.resolve())
    assert pipeline._report("load")["n_vis"] == onp.asarray(tiny.vis).size


def test_a_selection_that_leaves_no_data_fails_the_load_stage(tiny, tmp_path):
    folder = tmp_path / "run"
    pipeline = DiskPipeline(
        tiny, output=folder, wavel_range=(5e-6, 6e-6), **TINY
    )
    with pytest.raises(ValueError, match="No samples"):
        pipeline.run()
    run = _run_json(folder)
    assert run["status"] == "failed" and run["error"]["stage"] == "load"
    assert run["error"]["type"] == "ValueError"
    assert run["stages"]["load"]["status"] == "failed"
    assert not (folder / "data" / "processed.oifits").exists()


# --- a parameter on a prior bound ----------------------------------------------


@pytest.fixture(scope="module")
def rings_at_bound(tiny, tmp_path_factory):
    """The ring is at 5 mas and the outer radius is 3 mas: the fit has to
    put the ring on the bound."""
    folder = tmp_path_factory.mktemp("at_bound") / "run"
    settings = {**TINY, "r_out_mas": 3.0, **SHORT_NUTS}
    pipeline = DiskPipeline(tiny, "rings", output=folder, **settings)
    return pipeline, settings, pipeline.run(through="compare")


def _checks(res):
    return {c.name: c for c in res.checks}


def test_a_ring_on_the_outer_bound_is_reported(rings_at_bound):
    pipeline, _, res = rings_at_bound
    values = _io.load_model_values(pipeline.output / "models" / "best")
    # The fit sits on the bound (r_out_mas), not on the ring of the data...
    assert float(values["r1"]) == pytest.approx(3.0, rel=1e-2)
    # ...and says so, by name, instead of passing.
    at_bound = _checks(res)["at_bound"]
    assert at_bound.status == "warn" and "r1" in at_bound.value
    assert "r1" in at_bound.message and "prior bound" in at_bound.message
    assert res.summary["worst_check"] != "pass"
    # The fit does not hide behind the bound: the model fails on the data.
    assert _checks(res)["chi2_reduced"].status == "fail"


@pytest.mark.slow
def test_an_inclination_on_its_upper_bound_is_reported(tiny, tmp_path):
    pipeline = DiskPipeline(
        tiny, "rings", output=tmp_path / "run", **{**TINY, "inc_max_deg": 30.0}
    )
    res = pipeline.run(through="compare")
    # The data are at 50 degrees: the fit is held at the bound.
    assert res.summary["disk"]["inc_deg"] == pytest.approx(30.0, abs=0.1)
    check = _checks(res)["at_bound"]
    assert check.status == "warn" and "inc" in check.value


# --- a posterior that has not converged ----------------------------------------


def _fake_nuts(kind):
    """Draws, two chains of eight, that are bad in the way ``kind`` says."""

    def nuts(pipeline, model, priors, start):
        rng = onp.random.default_rng(0)
        samples = {}
        for name, value in start.items():
            base = onp.asarray(value, dtype=float)
            scale = abs(base) + 1.0
            draws = base + 0.01 * scale * rng.normal(size=(2, 8) + base.shape)
            if kind == "chains_disagree":
                draws[1] += 5.0 * scale
            samples[name] = draws
        return samples, {"diverging": onp.full((2, 8), kind == "divergent")}

    return nuts


@pytest.mark.parametrize(
    ("kind", "failing"),
    [("chains_disagree", "r_hat"), ("divergent", "divergences")],
)
def test_bad_draws_fail_the_sampler_checks_and_the_summary(
    rings_at_bound, tiny, tmp_path, monkeypatch, kind, failing
):
    # The draws are replaced; everything from them to the summary is real.
    pipeline, settings, _ = rings_at_bound
    folder = tmp_path / "run"
    shutil.copytree(pipeline.output, folder)
    monkeypatch.setattr(disk, "_nuts", _fake_nuts(kind))
    res = DiskPipeline(tiny, "rings", output=folder, **settings).run(
        through="posterior"
    )
    checks = _checks(res)
    assert checks[failing].status == "fail"
    assert checks["ess"].status == "fail"  # 16 draws
    assert res.summary["worst_check"] == "fail"
    assert f"[fail] {failing}" in res.describe()
    post = res.summary["posterior"]
    assert (post["num_warmup"], post["num_chains"]) == (5, 2)
    assert (folder / "samples.h5").exists()


@pytest.fixture(scope="module")
def rings_short_nuts(tiny, tmp_path_factory):
    folder = tmp_path_factory.mktemp("short_nuts") / "run"
    pipeline = DiskPipeline(
        tiny, "rings", output=folder, **{**TINY, **SHORT_NUTS}
    )
    return pipeline, pipeline.run(through="posterior")


@pytest.mark.slow
def test_a_sampler_run_too_short_to_converge_is_not_a_pass(rings_short_nuts):
    _, res = rings_short_nuts
    checks = _checks(res)
    assert checks["ess"].status == "fail"
    # Five warmup steps do not adapt: the chains disagree or diverge.
    assert (checks["r_hat"].status, checks["divergences"].status) != (
        "pass",
        "pass",
    )
    assert res.summary["worst_check"] == "fail"
    post = res.summary["posterior"]
    assert post["num_warmup"] == 5 and post["num_samples"] == 8
    # Eight draws may give a tiny ESS or none at all (summary.json writes NaN
    # as null, as on the lowest-dependency nightly). Either way the check
    # fails, and says which: the stored value and the message must agree.
    ess = post["ess_bulk_min"]
    if ess is None:
        assert "not finite" in checks["ess"].message
    else:
        assert ess < 100.0
        assert "too few independent samples" in checks["ess"].message
    assert res.status == "partial"  # through="posterior"


# --- the null case: a star and no disk -----------------------------------------
#
# The pipeline has no "no disk" outcome. A disk is always fitted; the data
# with none send its flux to the floor of the prior and the radii to a bound,
# and the ``at_bound`` check (a warning, not a failure) is what says so. These
# tests pin that: a disk that the data do not need is never reported as a
# pass.


@pytest.fixture(scope="module")
def rings_null(point_source, tmp_path_factory):
    folder = tmp_path_factory.mktemp("null_rings") / "run"
    pipeline = DiskPipeline(point_source, "rings", output=folder, **TINY)
    return pipeline, pipeline.run()


@pytest.fixture(scope="module")
def gp_null(point_source, tmp_path_factory):
    # model=None: the default model, the Gaussian process.
    folder = tmp_path_factory.mktemp("null_gp") / "run"
    settings = {**TINY, "n_inc": 3, "r_out_mas": 9.0}
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(disk, "_GEOMETRY_LATTICE_INC", ())
        patch.setattr(disk, "_GEOMETRY_LATTICE_PA", ())
        pipeline = DiskPipeline(point_source, output=folder, **settings)
        return pipeline, settings, pipeline.run()


def test_rings_fitted_to_a_star_are_not_a_confident_disk(rings_null):
    _, res = rings_null
    assert res.status == "complete"
    checks = _checks(res)
    # A point source is fitted well: the warning is not a bad fit's.
    assert checks["chi2_reduced"].status == "pass"
    # The disk carries a few thousandths of the star's flux, and the ring sits
    # on a bound of its prior; the check names the parameter.
    ring = res.summary["disk"]["rings"][0]
    assert ring["flux"] < 0.02
    assert checks["at_bound"].status == "warn"
    assert checks["at_bound"].value
    assert res.summary["worst_check"] != "pass"


@pytest.mark.slow
def test_a_gp_fitted_to_a_star_is_not_a_confident_disk(gp_null):
    _, _, res = gp_null
    assert res.status == "complete" and res.summary["model"]["name"] == "gp"
    checks = _checks(res)
    assert checks["chi2_reduced"].status == "pass"
    # The disk's flux is on the floor of its prior and is reported as such.
    assert res.summary["disk"]["flux"] < 0.02
    assert checks["at_bound"].status == "warn"
    assert "disk.flux" in checks["at_bound"].value
    assert res.summary["worst_check"] != "pass"


def test_resume_of_a_complete_real_run_recomputes_nothing(
    rings_null, point_source, monkeypatch
):
    # Every stage really leaves the outputs it declares (the quicklook is
    # written, not executed), so a complete run is skipped whole.
    pipeline, before = rings_null
    ran = []
    for name in STAGE_FUNCTIONS.values():
        monkeypatch.setattr(disk, name, lambda p, n=name: ran.append(n))
    again = DiskPipeline(
        point_source, "rings", output=pipeline.output, **TINY
    ).run()
    assert ran == []
    assert again.status == "complete" and again.summary == before.summary
    assert {k: v["started"] for k, v in again.run["stages"].items()} == {
        k: v["started"] for k, v in before.run["stages"].items()
    }


@pytest.mark.slow
def test_resume_of_a_complete_real_gp_run_recomputes_nothing(
    gp_null, point_source, monkeypatch
):
    pipeline, settings, before = gp_null
    ran = []
    for name in STAGE_FUNCTIONS.values():
        monkeypatch.setattr(disk, name, lambda p, n=name: ran.append(n))
    again = DiskPipeline(
        point_source, output=pipeline.output, **settings
    ).run()
    assert ran == [] and again.summary == before.summary


# --- the defaults on an injected disk ------------------------------------------
#
# The golden runs of test_pipeline_disk.py override the grids, the ring and
# length-scale counts and the outer radius (and switch the companion search
# on). This one overrides nothing but the quicklook execution. The ring sits
# inside the default outer radius, which is half the field the data size (the
# data have no short baselines, so on a 5 mas ring that default is 3.5 mas and
# the run, correctly, fails its checks; the golden runs set r_out_mas to 12).

GOLDEN = System(
    star=PointSource(flux=1.0),
    ring=MultipoleDisk(
        GaussianRing(2.5, 0.6),
        inc=50.0,
        pa=30.0,
        flux=1.0,
        n_r=48,
        harmonics=Harmonics(re=(0.4,), im=(0.2,)),
    ),
)


@pytest.mark.slow
def test_the_default_settings_recover_an_injected_disk(tmp_path):
    data = vlti_oidata(
        wavelengths_m=onp.linspace(1.6e-6, 2.4e-6, 5),
        hour_angles_h=(-2.0, 0.0, 2.0),
        sigma_v2=0.01,
        sigma_cp_deg=3.0,
    ).with_model(GOLDEN, key=jax.random.PRNGKey(3))
    pipeline = DiskPipeline(
        data, output=tmp_path / "run", quicklook_execute=False
    )
    res = pipeline.run()
    assert res.status == "complete" and res.summary["model"]["name"] == "gp"
    # Nothing but the quicklook execution was overridden.
    defaults = {**DiskPipeline.defaults(), "quicklook_execute": False}
    assert res.run["config"] == defaults

    s = res.summary
    assert s["worst_check"] != "fail", res.describe()
    assert _checks(res)["chi2_reduced"].status == "pass"
    assert s["chi2"]["reduced"][s["disk"]["best"]] < 1.4
    assert abs(s["disk"]["inc_deg"] - 50.0) < 6.0
    pa = s["disk"]["pa_deg"]
    assert min(abs(pa - 30.0), 180.0 - abs(pa - 30.0)) < 10.0
    assert s["disk"]["flux"] == pytest.approx(1.0, rel=0.2)
    # The light is where the ring is, to within a half-fringe.
    hf = s["model"]["resolution_mas"]
    r, w = (onp.asarray(a) for a in res.model().disk.profile.quadrature())
    assert abs(float(onp.sum(w * r) / onp.sum(w)) - 2.5) < hf
    # The default length-scale grid is the three half-fringe multiples.
    assert len(s["model"]["ell_mas"]) == 3
    assert s["disk"]["companion"] is None
