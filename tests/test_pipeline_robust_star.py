"""StarPipeline robustness: per-stage resume and fresh runs, failure paths
that must be loud, a golden synthetic case and a null case.

The sibling ``test_pipeline_star.py`` covers the contract, the outputs and
the individual checks. This file asks what happens when something goes
wrong or is interrupted: a stage output missing, a stage that raises, bad
input files, chains too short to converge, a prior that excludes the truth,
data with no signal. None of these may pass silently.
"""

import inspect
import json
import shutil

import jax
import numpy as onp
import pytest

pytest.importorskip("h5py")
pytest.importorskip("nbclient")

from virgil import PointSource, UniformDisk  # noqa: E402
from virgil.coverage import vlti_oidata  # noqa: E402
from virgil.models import QuadraticLimbDarkenedDisk  # noqa: E402
from virgil.pipeline import StarPipeline, load  # noqa: E402
from virgil.pipeline import star  # noqa: E402
from tests._shared import (  # noqa: E402
    checks_by_name,
    stage_outputs,
    started_times,
)

# Same group as the sibling file, so both run on one worker with
# ``--dist loadgroup`` and share its warmed-up JAX caches.
pytestmark = pytest.mark.xdist_group("pipeline_star")

# Copied from test_pipeline_star.py: tiny chains check plumbing, not the
# sampler.
TINY = {
    "max_lobes": 2,
    "num_warmup": 30,
    "num_samples": 30,
    "num_chains": 2,
}
TRUTH = QuadraticLimbDarkenedDisk.from_u(6.0, u1=0.35, u2=0.25)
STAGES = StarPipeline.STAGES


def _template(**kwargs):
    kwargs = {"sigma_v2": 0.01, "sigma_cp_deg": 15.0, **kwargs}
    return vlti_oidata(
        wavelengths_m=onp.linspace(1.6e-6, 2.4e-6, 3),
        hour_angles_h=(-2.5, 0.0, 2.5),
        **kwargs,
    )


@pytest.fixture(scope="module")
def data():
    return _template().with_model(TRUTH, key=jax.random.PRNGKey(3))


@pytest.fixture(scope="module")
def null_data():
    """Pure noise around V = 1 and closure phase 0: nothing is resolved."""
    return _template().with_model(PointSource(), key=jax.random.PRNGKey(5))


# --- (a) per-stage resume and fresh runs -------------------------------------


@pytest.fixture(scope="module")
def completed(data, tmp_path_factory):
    """One complete uniform-disk run (no notebook execution) to copy."""
    out = tmp_path_factory.mktemp("completed") / "run"
    pipeline = StarPipeline(
        data, "uniform", output=out, quicklook_execute=False, **TINY
    )
    res = pipeline.run()
    assert res.status == "complete"
    return pipeline


FUNCTIONS = ("_load_star", "_overview", "_fit", "_posterior", "_quicklook")


def _stub_stages(monkeypatch, completed=None):
    """Replace every stage function with a recorder; returns the call log.

    A recorder writes no output. With ``completed`` it returns that run's
    stored report for the stage, so the summary can still be computed.
    """
    ran = []

    def recorder(stage):
        def stub(p):
            ran.append(stage)
            if completed is None:
                return {}
            report = completed._report(stage)
            report.pop("warnings", None)
            return report

        return stub

    for stage, name in zip(STAGES, FUNCTIONS):
        monkeypatch.setattr(star, name, recorder(stage))
    return ran


@pytest.fixture
def stubbed(monkeypatch, completed):
    return _stub_stages(monkeypatch, completed)


def _copy_run(completed, tmp_path):
    out = tmp_path / "run"
    shutil.copytree(completed.output, out)
    return StarPipeline(
        completed.data,
        "uniform",
        output=out,
        quicklook_execute=False,
        **TINY,
    )


def test_a_complete_run_recomputes_nothing_whatever_the_stop(
    completed, stubbed, tmp_path
):
    pipeline = _copy_run(completed, tmp_path)
    before = started_times(pipeline)
    for through in (*STAGES, None):
        pipeline.run(through=through)
    assert stubbed == []
    assert started_times(pipeline) == before
    assert json.loads((pipeline.output / "run.json").read_text())[
        "status"
    ] == ("complete")


@pytest.mark.parametrize("stage", STAGES)
@pytest.mark.parametrize("damage", ["output removed", "marked failed"])
def test_resume_recomputes_a_damaged_stage_and_everything_after_it(
    completed, stubbed, tmp_path, stage, damage
):
    pipeline = _copy_run(completed, tmp_path)
    before = started_times(pipeline)
    if damage == "output removed":
        victim = pipeline.output / stage_outputs(pipeline, stage)[0]
        victim.unlink()
    else:
        path = pipeline.output / "run.json"
        run = json.loads(path.read_text())
        run["stages"][stage]["status"] = "failed"
        path.write_text(json.dumps(run))
    pipeline.run()
    k = STAGES.index(stage)
    # The stage and every later one reran; no earlier stage did.
    assert stubbed == list(STAGES[k:])
    after = started_times(pipeline)
    for earlier in STAGES[:k]:
        assert after[earlier] == before[earlier], earlier
    assert list(after) == list(STAGES)  # all complete again


@pytest.mark.parametrize("stage", STAGES)
def test_a_fresh_run_discards_every_output_and_runs_through_the_stage(
    completed, stubbed, tmp_path, stage
):
    """``resume=False`` clears the folder first, so the outputs of later
    stages disappear and the stages up to ``through`` rerun."""
    pipeline = _copy_run(completed, tmp_path)
    everything = [
        pipeline.output / p
        for s in STAGES
        for p in stage_outputs(pipeline, s)
        if not p.startswith("stages/")
    ]
    assert all(p.exists() for p in everything)
    pipeline.run(through=stage, resume=False)
    k = STAGES.index(stage)
    assert stubbed == list(STAGES[: k + 1])
    # The recorders write nothing: whatever exists now was not recomputed,
    # and nothing stale survived.
    assert not [p for p in everything if p.exists()]
    for later in STAGES[k + 1 :]:
        assert not (pipeline.output / "stages" / later).exists(), later
    run = json.loads((pipeline.output / "run.json").read_text())
    assert list(run["stages"]) == list(STAGES[: k + 1])
    assert run["status"] == ("complete" if stage == STAGES[-1] else "partial")


def test_a_fresh_run_through_the_overview_really_replaces_the_outputs(
    completed, tmp_path
):
    pipeline = _copy_run(completed, tmp_path)
    out = pipeline.output
    (out / "plots" / "overview_uv.png").write_bytes(b"stale")
    pipeline.run(through="overview", resume=False)
    # Rewritten, not the stale bytes left behind.
    assert (out / "plots" / "overview_uv.png").read_bytes()[:4] == b"\x89PNG"
    for gone in (
        "grids.h5",
        "samples.h5",
        "quicklook.ipynb",
        "models",
        "stages/fit",
        "stages/posterior",
    ):
        assert not (out / gone).exists(), gone
    res = load(out)
    assert res.status == "partial"
    assert "fit" not in res.summary and "posterior" not in res.summary
    assert "data" in res.summary


# --- (b) failures are loud ---------------------------------------------------


def test_a_stage_that_raises_marks_the_run_failed_and_resume_continues(
    completed, tmp_path, monkeypatch
):
    pipeline = _copy_run(completed, tmp_path)
    (pipeline.output / "grids.h5").unlink()  # the fit has to rerun

    def boom(p):
        raise RuntimeError("the optimizer blew up")

    ran = _stub_stages(monkeypatch, completed)
    monkeypatch.setattr(star, "_fit", boom)
    with pytest.raises(RuntimeError, match="blew up"):
        pipeline.run()
    run = json.loads((pipeline.output / "run.json").read_text())
    assert run["status"] == "failed"
    assert run["error"]["stage"] == "fit"
    assert run["error"]["type"] == "RuntimeError"
    assert "blew up" in run["error"]["message"]
    assert run["stages"]["fit"]["status"] == "failed"
    assert run["stages"]["overview"]["status"] == "complete"
    assert ran == []  # the earlier stages were kept

    # The failure is not sticky: with the fault gone, resume redoes the
    # failed stage and the later ones, and nothing before.
    fixed = _stub_stages(monkeypatch, completed)
    pipeline.run()
    assert fixed == ["fit", "posterior", "quicklook"]
    run = json.loads((pipeline.output / "run.json").read_text())
    assert run["status"] == "complete" and run["error"] is None


@pytest.fixture(scope="module")
def unconverged(data, tmp_path_factory):
    """Chains far too short to adapt or to mix."""
    out = tmp_path_factory.mktemp("unconverged") / "run"
    return StarPipeline(
        data,
        "uniform",
        output=out,
        quicklook_execute=False,
        max_lobes=2,
        num_warmup=10,
        num_samples=20,
        num_chains=2,
    ).run(through="posterior")


def test_chains_that_have_not_converged_are_flagged_not_passed(unconverged):
    res = unconverged
    by_name = checks_by_name(res)
    assert by_name["ess"].status == "fail"
    assert by_name["r_hat"].status != "pass"
    assert res.summary["worst_check"] == "fail"
    # The numbers behind the checks are in the summary too.
    post = res.summary["posterior"]
    assert post["ess_bulk_min"] < 100
    assert post["r_hat_max"] > 1.01
    assert post["num_warmup"] == 10 and post["num_samples"] == 20


def test_describe_reports_the_unconverged_chains(unconverged):
    text = unconverged.describe()
    assert "[fail] ess" in text
    assert "too few independent samples" in text
    assert "[warn] r_hat" in text or "[fail] r_hat" in text


def test_chains_that_never_move_still_end_in_failing_checks(data, tmp_path):
    res = StarPipeline(
        data,
        "uniform",
        output=tmp_path / "run",
        quicklook_execute=False,
        max_lobes=2,
        num_warmup=2,
        num_samples=8,
        num_chains=2,
    ).run(through="posterior")
    assert checks_by_name(res)["ess"].status == "fail"


@pytest.mark.parametrize("bounds", [[1.0, 3.0], [10.0, 30.0]])
def test_a_prior_that_excludes_the_truth_is_reported(data, tmp_path, bounds):
    """The uniform disk runs to the prior edge: the chi^2 fails and the
    posterior's prior-bound check says the mass is piled against the edge.
    The truth is 6 mas, outside both ranges."""
    res = StarPipeline(
        data,
        "uniform",
        output=tmp_path / "run",
        diam_range_mas=bounds,
        quicklook_execute=False,
        max_lobes=2,
        num_warmup=60,
        num_samples=60,
        num_chains=2,
    ).run(through="posterior")
    s = res.summary
    by_name = checks_by_name(res)
    assert by_name["chi2_uniform"].status == "fail"
    assert by_name["prior_bound"].status == "warn"
    assert by_name["prior_bound"].value > 0.5
    assert "uniform.diam" in by_name["prior_bound"].message
    assert s["worst_check"] == "fail"
    assert abs(s["star"]["diam_mas"] - 6.0) > 1.5
    assert min(abs(s["star"]["diam_mas"] - b) for b in bounds) < 0.5
    assert "[warn] prior_bound" in res.describe()


def test_a_limb_darkened_fit_at_the_prior_edge_is_an_at_bound_check(
    data, tmp_path
):
    res = StarPipeline(
        data,
        output=tmp_path / "run",
        diam_range_mas=[1.0, 3.0],
        quicklook_execute=False,
        **TINY,
    ).run(through="fit")
    check = checks_by_name(res)["start_at_bound"]
    assert check.status == "warn"
    assert "limb_darkened.diam" in check.value
    assert "prior bound" in check.message
    assert res.summary["worst_check"] == "fail"
    assert "[warn] start_at_bound" in res.describe()


def test_a_uniform_fit_at_the_prior_edge_is_an_at_bound_check(data, tmp_path):
    res = StarPipeline(
        data,
        "uniform",
        output=tmp_path / "run",
        diam_range_mas=[1.0, 3.0],
        quicklook_execute=False,
        **TINY,
    ).run(through="fit")
    assert res.summary["fit"]["models"]["uniform"]["params"]["diam"] == 3.0
    assert "start_at_bound" in checks_by_name(res)


# --- (d) a null case ---------------------------------------------------------


def test_pure_noise_around_one_gives_flags_not_a_confident_diameter(
    null_data, tmp_path
):
    res = StarPipeline(
        null_data,
        output=tmp_path / "run",
        quicklook_execute=False,
        **TINY,
    ).run(through="fit")
    s = res.summary
    by_name = checks_by_name(res)
    # The diameter runs to the lower edge of the prior (0.1 lambda/B, well
    # below the resolution), and the checks say so.
    assert s["star"]["diam_mas"] < 0.25 * s["data"]["resolution_mas"]
    assert by_name["start_at_bound"].status == "warn"
    assert by_name["resolution"].status == "fail"
    assert s["resolution"]["first_null_fraction"] < 0.5
    assert s["worst_check"] == "fail"
    # Limb darkening is not claimed for a star that is not resolved.
    assert s["comparison"]["preferred"] == "uniform"
    # delta_chi2 < 0 was never true: the shape's gain (17.8 here) is real but
    # is the diameter's lower prior edge, not limb darkening, so it is unidentified.
    assert s["comparison"]["unidentified"]
    assert "shape is unidentified" in res.describe()
    assert "[fail] resolution" in res.describe()


def test_the_shape_is_unidentified_for_an_unresolved_star_on_a_second_realization(
    tmp_path,
):
    # A second realization, independent of null_data: the limb-darkened disk
    # can emulate a diameter below the prior's lower edge, so a gain over the
    # uniform disk at that edge is not evidence of limb darkening.
    data = _template().with_model(PointSource(), key=jax.random.PRNGKey(11))
    res = StarPipeline(
        data, output=tmp_path / "run", quicklook_execute=False, **TINY
    ).run(through="fit")
    comparison = res.summary["comparison"]
    assert comparison["preferred"] == "uniform"
    assert comparison["unidentified"]
    assert "[pass] limb_darkening_gain" in res.describe()
    assert "unidentified" in res.describe()


# --- (c) a golden synthetic case ---------------------------------------------


def _golden(truth, tmp_path, model=None, **settings):
    data = _template().with_model(truth, key=jax.random.PRNGKey(3))
    return StarPipeline(data, model, output=tmp_path / "run", **settings).run()


def _assert_golden(res, truth):
    assert res.status == "complete"
    s = res.summary
    assert s["star"]["source"] == "posterior"
    diam, err = s["star"]["diam_mas"], s["star"]["diam_err_mas"]
    assert 0.0 < err < 0.5
    assert abs(diam - float(truth.diam)) < 3 * err, (diam, err, res.describe())
    failing = [c for c in res.checks if c.status == "fail"]
    assert not failing, res.describe()
    assert s["worst_check"] != "fail"
    assert (res.path / "quicklook.ipynb").exists()
    assert s["posterior"]["num_chains"] == 4


@pytest.mark.slow
def test_golden_default_settings_recover_a_uniform_disk(tmp_path):
    """Every setting at its default (2000-point scan, five lobes, four
    chains of 1000 + 1000 NUTS draws, the executed quicklook), for the
    uniform-disk model: the run is complete, the diameter is within 3 sigma
    of the injected 4 mas and no check fails. (About 40 s.)"""
    truth = UniformDisk(4.0)
    res = _golden(truth, tmp_path, "uniform")
    _assert_golden(res, truth)
    assert res.summary["chi2"]["reduced"]["uniform"] < 2.0
    assert res.summary["posterior"]["r_hat_max"] < 1.01


@pytest.mark.slow
@pytest.mark.xfail(
    strict=True,
    reason="virgil#580: the limb-darkened disk sampled explicitly on a star whose "
    "data cannot constrain its shape (q1, q2 on a thin curved ridge with the "
    "diameter) does not mix: r_hat 2.2, ESS 2.6 at 250 + 250 draws; "
    "the pipeline avoids sampling it by default (next test) but does not fix this",
)
@pytest.mark.timeout(900)
def test_golden_limb_darkened_model_sampled_explicitly_mixes(tmp_path):
    """The limb-darkened disk alone on the golden star, NUTS at 250 + 250."""
    truth = UniformDisk(4.0)
    res = _golden(
        truth, tmp_path, "limb_darkened", num_warmup=250, num_samples=250
    )
    _assert_golden(res, truth)


@pytest.mark.slow
@pytest.mark.timeout(900)
def test_golden_default_model_set_does_not_sample_an_unconstrained_shape(
    tmp_path,
):
    """The default pipeline (uniform and limb-darkened, compared) on the
    golden star (virgil#580): the data cannot constrain the limb darkening
    (expected σ_posterior/σ_prior of q1, q2 above 0.8), so the unselected
    model is fitted and compared but not sampled, and the output says so."""
    truth = UniformDisk(4.0)
    res = _golden(truth, tmp_path, num_warmup=250, num_samples=250)
    _assert_golden(res, truth)
    comparison = res.summary["comparison"]
    assert comparison["preferred"] == "uniform"
    assert not comparison["constrained"]
    assert min(comparison["shape_ratio"].values()) > 0.8
    assert list(res.summary["posterior"]["models"]) == ["uniform"]
    assert "[warn] limb_darkening_not_sampled" in res.describe()


def _dense(sigma_v2):
    return vlti_oidata(
        wavelengths_m=onp.linspace(1.6e-6, 2.4e-6, 4),
        hour_angles_h=onp.linspace(-3.0, 3.0, 5),
        sigma_v2=sigma_v2,
        sigma_cp_deg=15.0,
    )


def _shape_ratios(data, diam):
    entry = star.models()["limb_darkened"]
    prior_std = {"q1": 0.2887, "q2": 0.2887}  # Uniform(0, 1)
    return star._shape_ratios(data, entry, diam, prior_std)


@pytest.mark.parametrize(
    "data, diam, expected, constrained",
    [
        # The golden star's coverage (3 wavelengths x 3 hour angles, sigma
        # 0.01, 4 mas): almost the prior alone.
        (_template(), 4.0, (0.92, 0.88), False),
        # Just above 0.8 (4 x 5, sigma 1e-2, 8 mas): sampled alone at
        # 500 + 500 draws the limb-darkened model had r_hat 2.6, ESS 2.6.
        (_dense(1e-2), 8.0, (0.85, 0.74), False),
        # Just below (4 x 5, sigma 3e-4, 4 mas): r_hat 1.035, ESS 141 at
        # 500 + 500 draws.
        (_dense(3e-4), 4.0, (0.70, 0.60), True),
        # Well measured (4 x 5, sigma 1e-4, 8 mas).
        (_dense(1e-4), 8.0, (0.14, 0.11), True),
    ],
)
def test_the_shape_gate_decides_on_both_sides_of_the_threshold(
    data, diam, expected, constrained
):
    ratios = _shape_ratios(data, diam)
    assert tuple(ratios.values()) == pytest.approx(expected, abs=0.02)
    assert all(r < 0.8 for r in ratios.values()) is constrained


def test_a_constrained_shape_is_fitted_and_kept_when_the_data_resolve_it(
    tmp_path,
):
    truth = QuadraticLimbDarkenedDisk.from_u(8.0, u1=0.35, u2=0.25)
    data = _dense(1e-4).with_model(truth, key=jax.random.PRNGKey(3))
    res = StarPipeline(
        data, output=tmp_path / "run", quicklook_execute=False, **TINY
    ).run(through="fit")
    comparison = res.summary["comparison"]
    assert comparison["constrained"]
    assert comparison["preferred"] == "limb_darkened"
    assert "not_sampled" not in res.describe()


@pytest.mark.slow
@pytest.mark.timeout(1800)
def test_a_constrained_limb_darkened_star_mixes_at_the_default_draws(tmp_path):
    """A positive control, not a regression test of the gate (the same run
    passes without it): a star past the first null whose shape the data
    constrain (σ(V²) = 1e-4, 4 wavelengths, 5 hour angles, 8 mas), the
    limb-darkened model at the default 1000 + 1000 NUTS draws. (About 5 min.)"""
    truth = QuadraticLimbDarkenedDisk.from_u(8.0, u1=0.35, u2=0.25)
    data = _dense(1e-4).with_model(truth, key=jax.random.PRNGKey(3))
    res = StarPipeline(
        data,
        "limb_darkened",
        output=tmp_path / "run",
        quicklook_execute=False,
    ).run()
    assert res.status == "complete"
    s = res.summary
    assert s["resolution"]["first_null_fraction"] > 1.0
    assert s["posterior"]["r_hat_max"] < 1.01
    assert s["posterior"]["ess_bulk_min"] > 400


def test_a_failing_model_that_was_not_selected_still_fails_and_is_named(
    completed,
):
    import copy

    reports = {s: completed._report(s) for s in ("load", "fit", "posterior")}
    best = reports["fit"]["best"]
    # TINY chains do not converge: give the selected model healthy chains.
    reports["posterior"]["models"][best].update(
        r_hat_max=1.0, ess_bulk_min=1e3, divergence_fraction=0.0
    )
    bad = copy.deepcopy(reports["posterior"]["models"][best])
    bad.update(r_hat_max=4.0, ess_bulk_min=2.0, divergence_fraction=0.5)
    reports["posterior"]["models"]["rival"] = bad

    _, checks = completed._summarize(reports)
    by_name = {c.name: c for c in checks}
    for name in ("r_hat", "ess", "divergences"):
        assert by_name[name].status == "fail", name
        assert "rival (not selected)" in by_name[name].message

    # A rival's failure is not hidden by the selected model's warning, nor by
    # an earlier rival's milder warning (the selected model passes ess here).
    mild = copy.deepcopy(reports["posterior"]["models"][best])
    mild.update(ess_bulk_min=300.0)
    models = reports["posterior"]["models"]
    models.pop("rival")
    models["mild"] = mild
    models["rival"] = bad
    _, checks = completed._summarize(reports)
    ess = {c.name: c for c in checks}["ess"]
    assert ess.status == "fail" and "rival (not selected)" in ess.message
    models.pop("mild")
    models[best].update(ess_bulk_min=250.0)
    _, checks = completed._summarize(reports)
    ess = {c.name: c for c in checks}["ess"]
    assert ess.status == "fail" and "rival (not selected)" in ess.message
    models[best].update(ess_bulk_min=1e3)

    # The selected model's failures stay failures.
    reports["posterior"]["models"][best].update(
        r_hat_max=4.0, ess_bulk_min=2.0
    )
    _, checks = completed._summarize(reports)
    by_name = {c.name: c for c in checks}
    assert (
        by_name["r_hat"].status == "fail" and by_name["ess"].status == "fail"
    )


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
