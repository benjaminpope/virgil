"""BinaryPipeline robustness: per-stage resume and fresh runs, failure paths
that must be loud, and a golden synthetic case.

The sibling ``test_pipeline_binary.py`` covers the contract, the outputs and
the individual checks, and ``test_pipeline_start_at_bound.py`` the null
companion. This file asks what happens when something is interrupted or goes
wrong: a stage output missing, bad input files, chains too short to converge,
a prior that excludes the truth. None of these may pass silently.
"""

import inspect
import json
import shutil

import jax
import numpy as onp
import pytest

pytest.importorskip("h5py")
pytest.importorskip("nbclient")

from virgil import BinaryModelCartesian, OIData  # noqa: E402
from virgil.pipeline import BinaryPipeline, load  # noqa: E402
from virgil.pipeline import binary  # noqa: E402
from tests._shared import (  # noqa: E402
    checks_by_name,
    stage_outputs,
    started_times,
)

# Same group as the sibling file, so both run on one worker with
# ``--dist loadgroup`` and share its warmed-up JAX caches.
pytestmark = pytest.mark.xdist_group("pipeline_binary")

# Copied from test_pipeline_binary.py: tiny grids and chains check plumbing,
# not the sampler.
TINY = {
    "max_sep_mas": 300.0,
    "max_grid": 13,
    "n_flux": 6,
    "flux_range": [1e-4, 0.2],
    # n_null=15 floors the false-alarm probability at 1/16 (norm.isf = 1.534σ), so detection_sigma must be below it.
    "n_null": 15,
    "detection_sigma": 1.5,
    "num_warmup": 30,
    "num_samples": 30,
    "num_chains": 2,
}
TRUTH = BinaryModelCartesian(120.0, 80.0, 0.01)
STAGES = BinaryPipeline.STAGES
# The function of ``virgil.pipeline.binary`` that runs each stage.
STAGE_FUNCTIONS = dict(
    zip(
        STAGES,
        (
            "_load_report",
            "_overview",
            "_search",
            "_limits",
            "_fit",
            "_posterior",
            "_quicklook",
        ),
    )
)


@pytest.fixture(scope="module")
def data():
    base = OIData("data/NuHor_F480M.oifits")
    return base.with_model(TRUTH, key=jax.random.PRNGKey(3))


def _read(path):
    return json.loads(path.read_text())


# --- (a) per-stage resume and fresh runs -------------------------------------


@pytest.fixture(scope="module")
def completed(data, tmp_path_factory):
    """One complete run (the notebook is written, not executed) to copy."""
    out = tmp_path_factory.mktemp("completed") / "run"
    pipeline = BinaryPipeline(
        data, output=out, quicklook_execute=False, **TINY
    )
    assert pipeline.run().status == "complete"
    return pipeline


@pytest.fixture
def stubbed(completed, monkeypatch):
    """Replace every stage function with a recorder; returns the call log.

    A recorder writes no files and returns the stage's report from the
    completed run, so that the summary can still be built from it.
    """
    ran = []
    for stage, name in STAGE_FUNCTIONS.items():
        report = _read(completed.output / "stages" / stage / "report.json")

        def record(p, stage=stage, report=report):
            ran.append(stage)
            return report

        monkeypatch.setattr(binary, name, record)
    return ran


def _copy_run(completed, tmp_path):
    out = tmp_path / "run"
    shutil.copytree(completed.output, out)
    return BinaryPipeline(
        completed.data, output=out, quicklook_execute=False, **TINY
    )


def test_the_recorders_cover_every_stage():
    # A new stage must be added to STAGE_FUNCTIONS, or these tests would
    # silently run it for real.
    assert tuple(STAGE_FUNCTIONS) == STAGES
    pipeline = BinaryPipeline(None, quicklook_execute=False, **TINY)
    assert [s.name for s in pipeline._stages()] == list(STAGES)
    for stage in pipeline._stages():
        assert stage.run is getattr(binary, STAGE_FUNCTIONS[stage.name])


@pytest.mark.parametrize("through", [*STAGES, None])
def test_resume_recomputes_nothing_for_completed_stages(
    completed, stubbed, tmp_path, through
):
    pipeline = _copy_run(completed, tmp_path)
    before = started_times(pipeline)
    res = pipeline.run(through=through)
    assert stubbed == []
    assert started_times(pipeline) == before
    assert res.status == "complete"  # the stored stages were all kept
    assert list(_read(pipeline.output / "run.json")["stages"]) == list(STAGES)


@pytest.mark.parametrize("stage", STAGES)
@pytest.mark.parametrize("damage", ["output removed", "marked failed"])
def test_resume_recomputes_a_damaged_stage_and_everything_after_it(
    completed, stubbed, tmp_path, stage, damage
):
    pipeline = _copy_run(completed, tmp_path)
    before = started_times(pipeline)
    if damage == "output removed":
        (pipeline.output / stage_outputs(pipeline, stage)[0]).unlink()
    else:
        path = pipeline.output / "run.json"
        run = _read(path)
        run["stages"][stage]["status"] = "failed"
        path.write_text(json.dumps(run))
    pipeline.run()
    k = STAGES.index(stage)
    # The stage and every later one reran, in order; no earlier stage did.
    assert stubbed == list(STAGES[k:])
    after = started_times(pipeline)
    for earlier in STAGES[:k]:
        assert after[earlier] == before[earlier], earlier
    assert _read(pipeline.output / "run.json")["status"] == "complete"


@pytest.mark.parametrize("stage", STAGES)
def test_a_rerun_deletes_the_outputs_of_the_later_stages_first(
    completed, stubbed, tmp_path, monkeypatch, stage
):
    """Before a stage reruns, what it and the later stages wrote is gone, so
    a stale file is never mistaken for a new one."""
    pipeline = _copy_run(completed, tmp_path)
    k = STAGES.index(stage)
    stale = [
        pipeline.output / p
        for s in STAGES[k:]
        for p in stage_outputs(pipeline, s)
    ]
    assert all(p.exists() for p in stale)
    (pipeline.output / stage_outputs(pipeline, stage)[0]).unlink()
    seen = []
    inner = getattr(binary, STAGE_FUNCTIONS[stage])

    def spy(p):
        seen.append([x.name for x in stale if x.exists()])
        return inner(p)

    monkeypatch.setattr(binary, STAGE_FUNCTIONS[stage], spy)
    pipeline.run()
    assert seen == [[]]
    # Nothing of the earlier stages was touched.
    for s in STAGES[:k]:
        assert all(
            (pipeline.output / p).exists() for p in stage_outputs(pipeline, s)
        )


@pytest.mark.parametrize("stage", STAGES)
def test_a_fresh_run_discards_every_output_and_runs_through_the_stage(
    completed, stubbed, tmp_path, stage
):
    """``resume=False`` clears the folder first: the outputs of later stages
    disappear and the stages up to ``through`` rerun."""
    pipeline = _copy_run(completed, tmp_path)
    everything = [
        pipeline.output / p for s in STAGES for p in stage_outputs(pipeline, s)
    ]
    assert all(p.exists() for p in everything)
    res = pipeline.run(through=stage, resume=False)
    k = STAGES.index(stage)
    assert stubbed == list(STAGES[: k + 1])
    # The recorders write no files, only the runner writes each stage's
    # report: whatever else exists now was not recomputed, and no output of
    # a later stage survived.
    reports = {
        pipeline.output / "stages" / s / "report.json" for s in STAGES[: k + 1]
    }
    assert not [p for p in everything if p.exists() and p not in reports]
    run = _read(pipeline.output / "run.json")
    assert list(run["stages"]) == list(STAGES[: k + 1])
    assert run["status"] == ("complete" if stage == STAGES[-1] else "partial")
    assert res.status == run["status"]


def test_a_fresh_run_through_the_overview_really_replaces_the_outputs(
    completed, tmp_path
):
    """No recorders: the real load and overview stages rewrite their files."""
    pipeline = _copy_run(completed, tmp_path)
    out = pipeline.output
    (out / "plots" / "overview_uv.png").write_bytes(b"stale")
    res = pipeline.run(through="overview", resume=False)
    assert (out / "plots" / "overview_uv.png").read_bytes()[:4] == b"\x89PNG"
    for gone in (
        "grids.h5",
        "samples.h5",
        "quicklook.ipynb",
        "models",
        "stages/search",
        "stages/fit",
        "stages/posterior",
    ):
        assert not (out / gone).exists(), gone
    assert res.status == "partial"
    assert "search" not in res.summary and "fit" not in res.summary
    assert "data" in res.summary


# --- (b) failures are loud ---------------------------------------------------


def test_from_config_refuses_a_missing_data_file(tmp_path):
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"data": "nothing_here.fits"}))
    with pytest.raises(FileNotFoundError, match="nothing_here"):
        BinaryPipeline.from_config(config)


def test_data_with_no_usable_samples_is_refused():
    empty = {
        "u": onp.zeros(0),
        "v": onp.zeros(0),
        "wavel": onp.array([4.8e-6]),
        "vis": onp.zeros(0),
        "d_vis": onp.zeros(0),
    }
    with pytest.raises(ValueError, match="No unflagged data"):
        OIData(empty)


def test_a_stage_that_raises_marks_the_run_failed_and_resume_continues(
    completed, tmp_path, monkeypatch
):
    pipeline = _copy_run(completed, tmp_path)
    (pipeline.output / stage_outputs(pipeline, "fit")[0]).unlink()

    def boom(p):
        raise RuntimeError("the optimizer blew up")

    ran = []
    monkeypatch.setattr(binary, "_fit", boom)
    for name in ("_load_report", "_overview", "_search", "_limits"):
        monkeypatch.setattr(binary, name, lambda p, n=name: ran.append(n))
    with pytest.raises(RuntimeError, match="blew up"):
        pipeline.run()
    run = _read(pipeline.output / "run.json")
    assert run["status"] == "failed"
    assert run["error"]["stage"] == "fit"
    assert run["error"]["type"] == "RuntimeError"
    assert "blew up" in run["error"]["message"]
    assert run["stages"]["fit"]["status"] == "failed"
    assert run["stages"]["limits"]["status"] == "complete"
    assert ran == []  # the earlier stages were kept
    assert load(pipeline.output).status == "failed"

    # The failure is not sticky: with the fault gone, resume redoes the
    # failed stage and the later ones, and nothing before.
    monkeypatch.undo()
    fixed = []
    for stage, name in STAGE_FUNCTIONS.items():
        report = _read(completed.output / "stages" / stage / "report.json")
        monkeypatch.setattr(
            binary,
            name,
            lambda p, s=stage, r=report: fixed.append(s) or r,
        )
    pipeline.run()
    assert fixed == ["fit", "posterior", "quicklook"]
    run = _read(pipeline.output / "run.json")
    assert run["status"] == "complete" and run["error"] is None


def _rerun_posterior(completed, out, num_warmup, num_samples):
    """The posterior stage alone, with other sampler settings.

    Copies the completed run, writes the new settings into its ``run.json``
    (a resumed run would otherwise refuse them) and removes ``samples.h5``,
    so only the posterior and the notebook after it rerun: the earlier
    stages do not depend on the sampler settings.
    """
    settings = dict(TINY, num_warmup=num_warmup, num_samples=num_samples)
    pipeline = BinaryPipeline(
        completed.data, output=out, quicklook_execute=False, **settings
    )
    shutil.copytree(completed.output, out)
    run = _read(out / "run.json")
    run["config"].update(num_warmup=num_warmup, num_samples=num_samples)
    (out / "run.json").write_text(json.dumps(run))
    (out / "samples.h5").unlink()
    return pipeline.run()


@pytest.fixture(scope="module")
def unconverged(completed, tmp_path_factory):
    """Chains far too short to adapt or to mix (10 + 10 draws, 2 chains)."""
    out = tmp_path_factory.mktemp("unconverged") / "run"
    return _rerun_posterior(completed, out, 10, 10)


def test_chains_that_never_move_are_flagged_not_a_plotting_crash(
    completed, tmp_path
):
    res = _rerun_posterior(completed, tmp_path / "run", 2, 8)
    assert res.status == "complete"
    assert checks_by_name(res)["ess"].status == "fail"


def test_chains_that_have_not_converged_are_flagged_not_passed(unconverged):
    res = unconverged
    by_name = checks_by_name(res)
    assert by_name["ess"].status == "fail"
    assert by_name["r_hat"].status != "pass"
    assert res.summary["worst_check"] == "fail"
    # The numbers behind the checks are in the summary too.
    post = res.summary["posterior"]
    assert post["ess_bulk_min"] < 100
    assert post["num_warmup"] == 10 and post["num_samples"] == 10
    # The run itself finished: the status says the stages ran, the checks
    # say whether to believe them.
    assert res.status == "complete"


def test_describe_reports_the_unconverged_chains(unconverged):
    text = unconverged.describe()
    assert "[fail] ess" in text
    assert "too few independent samples" in text
    assert "r_hat" in text


@pytest.mark.parametrize(
    "settings, expected",
    [
        ({"flux_range": [1e-4, 1e-3]}, {"flux"}),
        ({"max_sep_mas": 40.0}, {"dra", "ddec"}),
    ],
    ids=["flux prior below the truth", "position box around the star"],
)
def test_a_prior_that_excludes_the_truth_is_reported_at_the_fit(
    data, tmp_path, settings, expected
):
    """The fit runs to a prior edge, which says the data are not described
    by the prior: the start-at-bound check names it, and the worst check is
    not a pass."""
    res = BinaryPipeline(
        data,
        output=tmp_path / "run",
        **dict(TINY, quicklook_execute=False, **settings),
    ).run(through="fit")
    at_bound = res.summary["fit"]["at_bound"]
    assert expected & set(at_bound), at_bound
    check = checks_by_name(res)["start_at_bound"]
    assert check.status == "warn" and check.value == at_bound
    assert "prior bound" in check.message
    assert res.summary["worst_check"] in ("warn", "fail")
    assert "[warn] start_at_bound" in res.describe()
    # The injected companion is outside the prior's reach: not recovered.
    fitted = res.summary["fit"]["params"]
    assert (
        abs(fitted["flux"] - TRUTH.flux) > 3e-3
        or abs(fitted["dra"] - TRUTH.dra) > 20.0
        or abs(fitted["ddec"] - TRUTH.ddec) > 20.0
    )


# --- (c) golden synthetic cases, at the default settings ----------------------

# The silent-failure checks of a search that did not find the companion.
_ALIAS_FLAGS = ("chi2", "grid_edge", "start_at_bound", "field_of_view")


def _truth_within(res, nsigma):
    """``{name: |median - truth| / sigma}`` of the posterior of the default
    (angular) template; sigma is half the 16-84 per cent width."""
    truth = {
        "sep": float(onp.hypot(TRUTH.dra, TRUTH.ddec)),
        "pa": float(onp.degrees(onp.arctan2(TRUTH.dra, TRUTH.ddec))),
        "flux": float(TRUTH.flux),
    }
    out = {}
    for name, value in truth.items():
        row = res.summary["posterior"]["params"][name]
        sigma = 0.5 * (row["q84"] - row["q16"])
        out[name] = abs(row["median"] - value) / sigma
    assert all(z < nsigma for z in out.values()), out
    return out


@pytest.mark.slow
def test_golden_injected_companion_is_recovered_at_the_default_settings(
    data, tmp_path
):
    """Every setting is the default (the grid, 4 chains of 1000 draws, the
    executed notebook) except ``max_sep_mas``, which keeps the search inside
    the companion's neighbourhood: at the default field (753 mas) and step
    (44 mas, coarser than the 20 mas that a signal-to-noise of 345 resolves)
    the search lands on an alias outside the field; see the next test."""
    res = BinaryPipeline(
        data, output=tmp_path / "run", max_sep_mas=300.0
    ).run()
    assert res.status == "complete"
    failing = [c.name for c in res.checks if c.status == "fail"]
    assert failing == [] and res.summary["worst_check"] != "fail"
    by_name = checks_by_name(res)
    assert by_name["detection"].status == "pass"
    assert (
        by_name["r_hat"].status == "pass" and by_name["ess"].status == "pass"
    )
    assert by_name["divergences"].status == "pass"
    assert res.summary["fit"]["at_bound"] == []
    z = _truth_within(res, 3.0)
    assert set(z) == {"sep", "pa", "flux"}
    # Everything the run folder promises is there, and it validates.
    out = tmp_path / "run"
    assert load(out).status == "complete"
    from virgil.pipeline import cli

    assert cli.main(["validate", str(out)]) == 0


@pytest.mark.slow
def test_the_default_settings_never_pass_a_wrong_companion_silently(
    data, tmp_path
):
    """With nothing changed the search may land on an alias of the injected
    companion (it does on this data: see the previous test). Whether it does
    or not, a position that is not the truth must come with a warning."""
    res = BinaryPipeline(data, output=tmp_path / "run").run()
    assert res.status == "complete"
    companion = res.summary["companion"]
    near = (
        abs(companion["dra_mas"] - TRUTH.dra) < 10.0
        and abs(companion["ddec_mas"] - TRUTH.ddec) < 10.0
    )
    if not near:
        by_name = checks_by_name(res)
        flagged = [
            name
            for name in _ALIAS_FLAGS
            if name in by_name and by_name[name].status != "pass"
        ]
        assert flagged, "a wrong companion position passed every check"
        assert res.summary["worst_check"] in ("warn", "fail")
        assert "[warn]" in res.describe() or "[fail]" in res.describe()


def test_the_corner_and_trace_figures_survive_draws_that_never_move(tmp_path):
    """Pins the fallback in ``_common._corner_trace`` directly: zero-width
    draws cannot be laid out, so placeholder figures and a warning replace
    them (the run-level test below depends on how far NUTS moves)."""
    import numpy as onp

    from virgil.pipeline._common import _corner_trace

    samples = {"a": onp.full((2, 10), 1.5), "b": onp.full((2, 10), -0.5)}
    (tmp_path / "plots").mkdir()
    with pytest.warns(RuntimeWarning, match="not drawn"):
        _corner_trace(tmp_path, samples, ["a", "b"])
    for name in ("posterior_corner.png", "posterior_trace.png"):
        assert (tmp_path / "plots" / name).stat().st_size > 0


# The shared full run costs ~15 s warm and over 30 s on a cold CI runner, so
# the tests that build on it (``completed``, ``stubbed``, ``unconverged``)
# belong to the slow tier; the rest of the module stays in the fast one.
for _name, _test in list(globals().items()):
    if _name.startswith("test_") and {
        "completed",
        "stubbed",
        "unconverged",
    } & set(inspect.signature(_test).parameters):
        globals()[_name] = pytest.mark.slow(_test)
