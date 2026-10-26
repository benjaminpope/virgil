"""OrbitPipeline: API contract, inputs, run folder, resume, CLI and quicklook.

The tests stop at the ``epochs`` stage or fake the later reports. The golden
blind runs are in ``test_pipeline_orbit_{gap,cases,resolved}.py``.
"""

import dataclasses
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import jax
import numpy as onp
import pytest

pytest.importorskip("h5py")
pytest.importorskip("nbclient")
pytest.importorskip("jaxoplanet")

import virgil  # noqa: E402
import virgil.pipeline.orbit as orbit_module  # noqa: E402
from tests._orbit_pipeline import (  # noqa: E402
    CHECKS,
    SEASONS,
    SUMMARY_KEYS,
    T_REF,
    TRUTH,
    _simulate,
)
from virgil.epochs import Epochs, NightFlag  # noqa: E402
from virgil.oifits import write_oifits  # noqa: E402
from virgil.orbits import KeplerOrbit  # noqa: E402
from virgil.pipeline import (  # noqa: E402
    SCHEMA,
    Check,
    ConfigMismatchError,
    OrbitPipeline,
    _checks,
    _quicklook,
    cli,
)
from virgil.pipeline.orbit import _engine_defaults  # noqa: E402

# One xdist worker builds each fixture once.
pytestmark = pytest.mark.xdist_group("pipeline_orbit")

TINY = {
    "max_grid": 25,
    "n_candidates": 10,
    "n_refine": 2,
    "n_is": 200,
    "num_warmup": 60,
    "num_samples": 60,
    "num_chains": 2,
    "quicklook_execute": False,
}


@pytest.fixture(scope="module")
def nights():
    return _simulate(SEASONS[:3])


# --- the contract -----------------------------------------------------------

API = {
    "OrbitPipeline": "(epochs, *, output='run', **settings)",
    "OrbitPipeline.from_oifits": (
        "(paths, metadata=None, *, output='run', **settings)"
    ),
}
STAGES = ("load", "epochs", "search", "modes", "posterior", "quicklook")
DEFAULTS = {
    "gap_days": 0.5,
    "exclude": {},
    "flag_nights": True,
    "flag_sep_blocks": False,
    "outlier_eps": 0.02,
    "batch_size": None,
    "memory_budget": None,
    "coarse_nats": 30.0,
    "n_peaks": 5,
    "max_sep_mas": None,
    "grid_step_mas": None,
    "max_grid": 201,
    "period_range": None,
    "a_range": [1.0, 30.0],
    "flux_range": [0.01, 1.0],
    "ecc_max": 0.9,
    "s_max": None,
    "primary_diam": "auto",
    "vis_amp": True,
    "sign_free": True,
    "flip_flux": True,
    "temper": True,
    **_engine_defaults(
        "n_bands_max", "n_candidates", "n_refine", "max_steps", "n_is"
    ),
    "alias_nats": 10.0,
    "no_orbit_nats": 10.0,
    "drop_outlier_night": True,
    "posterior_p_min": 0.05,
    "max_modes": 3,
    "num_warmup": 1000,
    "num_samples": 1000,
    "num_chains": 4,
    "chain_method": "vectorized",
    "seed": 0,
    "quicklook_execute": True,
}


def _signature(path):
    obj = virgil.pipeline
    for part in path.split("."):
        obj = getattr(obj, part)
    sig = inspect.signature(obj)
    params = [
        p.replace(annotation=inspect.Parameter.empty)
        for p in sig.parameters.values()
    ]
    return str(sig.replace(parameters=params, return_annotation=sig.empty))


def test_api_snapshot():
    # Changing any of these breaks users' scripts: update the snapshot only
    # with an additive change or a schema bump (see docs/pipeline.md).
    assert {k: _signature(k) for k in API} == API
    assert OrbitPipeline.STAGES == STAGES
    assert OrbitPipeline.defaults() == DEFAULTS
    assert OrbitPipeline.NAME == "orbit"
    assert OrbitPipeline.STABILITY == "provisional"
    assert SCHEMA == "virgil-pipeline-run-v1"
    assert "OrbitPipeline" in virgil.pipeline.__all__
    assert cli._PIPELINES["orbit"] == (
        "virgil.pipeline.orbit",
        "OrbitPipeline",
    )
    assert "orbit" in _quicklook._SECTIONS


def test_the_default_run_is_blind():
    # The default path carries no published period, window, start or prior:
    # the unconfigured search must pass the engine nothing but data-derived
    # arguments (the engine's own defaults do the rest), and those defaults
    # must be the engine's, so the two cannot drift apart.
    from virgil.aliases import fit_orbit_aliases

    d = OrbitPipeline.defaults()
    assert d["period_range"] is None
    assert d["primary_diam"] == "auto"
    assert not {"init", "p_range", "positions"} & set(d)
    engine = inspect.signature(fit_orbit_aliases).parameters
    for name in (
        "a_range",
        "flux_range",
        "ecc_max",
        "s_max",
        "primary_diam",
        "vis_amp",
        "sign_free",
        "flip_flux",
        "temper",
        "n_bands_max",
        "n_candidates",
        "n_refine",
        "max_steps",
        "n_is",
        "seed",
    ):
        assert tuple(np_tuple(d[name])) == tuple(
            np_tuple(engine[name].default)
        ), name
    assert engine["init"].default is None
    assert engine["skip_flagged"].default is True


def np_tuple(x):
    return x if isinstance(x, (list, tuple)) else (x,)


def _fake_reports():
    row = {"dataset": "n0", "mjd": 1.0, "sep_mas": 5.0, "flux": 0.3}
    chi2 = {"chi2": 10.0, "n": 10, "chi2_per_n": 1.0, "vis_amp": 1.0}
    band = {
        "rank": 0,
        "n": 20,
        "period": 12.1,
        "p_lo": 12.0,
        "p_hi": 12.2,
        "log_z": 0.0,
        "delta_log_z": 0.0,
        "p": 1.0,
        "chi2_red": 1.0,
        "labels_free": False,
        "flags": [],
    }
    return {
        "load": {
            **{k: 1.0 for k in orbit_module._DATA_KEYS},
            "excluded": {},
            "extra": "dropped",
        },
        "epochs": {
            "flagged": {"n1": ["low-gap"]},
            "nights": [
                {
                    **row,
                    "gap_marginal": 9.0,
                    "edge": False,
                    "flags": [{"code": "low-gap", "reason": "x"}],
                }
            ],
        },
        "search": {
            "p_range": [3.0, 100.0],
            "window_capped": None,
            "n_bands": 5,
            "failed_bands": [],
            "t_ref": 1.0,
            "span_days": 10.0,
            "primary_diam": None,
            "primary_resolved": False,
            "primary_diam_estimate": 0.1,
            "primary_at_grid_edge": False,
            "tempering": {"f": 1.0, "f_raw": 1.0, "out_of_range": False},
            "outliers": {"eps": 0.02, "probability": {"n0": 0.5, "n1": 0.01}},
            "night_flags": {"n1": ["low-gap"]},
            "labels_free": False,
            "period_bound": False,
            "period_at_lower_bound": False,
            "unseeded": [],
            "short_arc": None,
            "chi2": {"nights": {"n0": chi2}, "chi2": 10.0, "n": 10},
            "bands": [band],
        },
        "modes": {
            "winner": {
                "n": 20,
                "p": 1.0,
                "log_z": 0.0,
                "elements": {"period": 12.1},
                "flipped": False,
                "flags": [],
            },
            "runner_up": {"n": 21, "p": 0.0, "log_z": -30.0},
            "delta_log_z": 30.0,
            "labels_free": False,
            "no_orbit": {"delta_log_z": 50.0, "verdict": "orbit"},
            "robustness": {"skipped": "fewer than 3 nights"},
            "sampled": [20],
            "table": [band],
        },
        "posterior": {
            "modes": [
                {
                    "n": 20,
                    "params": {"period": {"median": 12.1}},
                    "r_hat_max": 1.0,
                    "ess_bulk_min": 900.0,
                    "divergence_fraction": 0.0,
                    "prior_bound_fraction": {"ecc": 0.0},
                }
            ],
            "converged": True,
            "budget_exhausted": False,
            "num_chains": 2,
            "num_samples": 10,
            "num_warmup": 10,
            "r_hat_max": 1.0,
            "ess_bulk_min": 900.0,
            "divergence_fraction": 0.0,
            "optimum_gap_max": 0.0,
            "prior_bound_fraction": {"ecc": 0.0},
        },
    }


def test_summary_keys_and_check_order_snapshot():
    sections, checks = orbit_module._summarize(
        OrbitPipeline.defaults(), _fake_reports()
    )
    keys = {
        k: (set(v) if isinstance(v, dict) else None)
        for k, v in sections.items()
    }
    expected = {
        k: v
        for k, v in SUMMARY_KEYS.items()
        if k
        not in ("schema", "pipeline", "stability", "status", "worst_check")
        and k not in ("warnings", "checks")
    }
    assert keys == expected
    assert [c.name for c in checks] == CHECKS
    assert all(isinstance(c, Check) for c in checks)
    assert sections["orbit"]["source"] == "posterior"


def test_summary_search_section_round_trips_outliers_and_night_flags():
    reports = _fake_reports()
    sections, _ = orbit_module._summarize(OrbitPipeline.defaults(), reports)
    search = json.loads(json.dumps(sections["search"]))
    for key in ("outliers", "night_flags"):
        assert search[key] == reports["search"][key]


def test_partial_and_no_orbit_summaries():
    reports = _fake_reports()
    for later in ("posterior", "modes", "search"):
        reports.pop(later)
        sections, checks = orbit_module._summarize(
            OrbitPipeline.defaults(), reports
        )
        assert later not in sections
        names = [c.name for c in checks]
        assert names == [c for c in CHECKS if c in names]
    reports = _fake_reports()
    reports["modes"]["no_orbit"] = {"delta_log_z": -2.0, "verdict": "no-orbit"}
    reports["posterior"] = {"skipped": "no-orbit", "modes": []}
    sections, checks = orbit_module._summarize(
        OrbitPipeline.defaults(), reports
    )
    status = {c.name: c.status for c in checks}
    assert "r_hat" not in status and status["no_orbit"] == "fail"
    assert sections["orbit"]["source"] == "map"


_ROBUST = {
    "night": "n1",
    "chi2_per_n": 3.0,
    "full": {"verdict": "orbit", "n": 20},
    "dropped": {"verdict": "orbit", "n": 20, "same_band": True},
}
# Dropping the night loses the orbit; or keeps it in another alias band.
_ROBUST_FLIPPED = {
    **_ROBUST,
    "dropped": {"verdict": "no-orbit", "n": 20, "same_band": True},
}
_ROBUST_MOVED = {
    **_ROBUST,
    "dropped": {"verdict": "orbit", "n": 21, "same_band": False},
}


@pytest.mark.parametrize(
    "check, args, status",
    [
        (_checks.alias_ambiguity, (30.0, 21), "pass"),
        (_checks.alias_ambiguity, (5.0, 21), "warn"),
        (_checks.alias_ambiguity, (0.5, 21), "fail"),
        (_checks.alias_ambiguity, (None, None), "pass"),
        (_checks.no_orbit, (50.0,), "pass"),
        (_checks.no_orbit, (5.0,), "warn"),
        (_checks.no_orbit, (-1.0,), "fail"),
        (_checks.night_dropped, (_ROBUST,), "pass"),
        (_checks.night_dropped, (_ROBUST_FLIPPED,), "warn"),
        (_checks.night_dropped, (_ROBUST_MOVED,), "warn"),
        (_checks.night_dropped, ({"skipped": "fewer than 3 nights"},), "pass"),
        (_checks.mode_converged, (0.0,), "pass"),
        (_checks.mode_converged, (25.0,), "warn"),
        (_checks.bands_failed, ([],), "pass"),
        (_checks.bands_failed, ([21, 22],), "warn"),
        (_checks.winner_flags, ([],), "pass"),
        (_checks.winner_flags, (["poor-fit"],), "pass"),
        (_checks.winner_flags, (["low-ess", "not-converged"],), "warn"),
        (_checks.labels_free, (False,), "pass"),
        (_checks.labels_free, (True,), "warn"),
        (_checks.night_flags, ({}, 5), "pass"),
        (_checks.night_flags, ({"n1": ["low-gap"]}, 5), "warn"),
        (_checks.period_bound, (False, False, [1.0, 2.0]), "pass"),
        (_checks.period_bound, (True, False, [1.0, 2.0]), "warn"),
        (_checks.period_bound, (False, True, [1.0, 2.0]), "warn"),
        (_checks.tempering, (False, 1.2, 1.2), "pass"),
        (_checks.tempering, (True, 3.0, 3.0), "warn"),
        (_checks.primary_resolved, (False, 0.1, False), "pass"),
        (_checks.primary_resolved, (True, 9.0, True), "warn"),
        (_checks.tempering, (False, None, None), "pass"),
        (_checks.primary_resolved, (True, 1.0, False), "pass"),
    ],
)
def test_orbit_checks_have_the_documented_thresholds(check, args, status):
    assert check(*args).status == status


# --- settings and inputs ----------------------------------------------------


def test_settings_are_validated(nights):
    for bad, match in (
        ({"gap_days": -1.0}, "gap_days"),
        ({"a_range": [5.0, 1.0]}, "a_range"),
        ({"flux_range": [0.01, 2.0]}, "flux_range must be at most 1"),
        ({"ecc_max": 1.0}, "ecc_max"),
        ({"period_range": [10.0, 5.0]}, "period_range"),
        ({"primary_diam": "big"}, "primary_diam"),
        ({"temper": 0.5}, "temper"),
        ({"chain_method": "x"}, "chain_method"),
        ({"n_candidates": 0}, "n_candidates"),
        ({"posterior_p_min": 2.0}, "posterior_p_min"),
        ({"outlier_eps": 1.0}, "outlier_eps"),
        ({"outlier_eps": -0.1}, "outlier_eps"),
        ({"vis_amp": [2.0, 1.0]}, "vis_amp"),
        ({"max_grid": 2}, "max_grid"),
        ({"n_peaks": True}, "n_peaks"),
    ):
        with pytest.raises(ValueError, match=match):
            OrbitPipeline(nights, **bad)
    with pytest.raises(TypeError, match="unknown settings"):
        OrbitPipeline(nights, num_sample=10)


def test_an_orbit_needs_two_nights(nights):
    one = {"n0": nights["n0"]}
    with pytest.raises(ValueError, match="at least two nights"):
        OrbitPipeline(one)
    with pytest.raises(TypeError, match="non-empty"):
        OrbitPipeline([])
    two = OrbitPipeline(nights, exclude={"n2": "bad night"})
    assert two.data.names == ("n0", "n1")
    assert two._excluded["n2"]["reason"] == "bad night"
    with pytest.raises(ValueError, match="no file, night or dataset"):
        OrbitPipeline(nights, exclude={"n9": "x"})
    with pytest.raises(ValueError, match="at least two nights"):
        OrbitPipeline(nights, exclude={"n1": "x", "n2": "y"})


def test_exclude_works_on_an_epochs_input_too(nights):
    # Times of the excluded datasets must not reach the rebuilt Epochs.
    given = Epochs(nights, times={"n0": 60000.0, "n1": 60010.0, "n2": 60020.0})
    pipeline = OrbitPipeline(given, exclude={"n2": "bad night"})
    assert pipeline.data.names == ("n0", "n1")
    assert [float(t) for t in pipeline.data.times] == [60000.0, 60010.0]


def test_epochs_and_dicts_are_equivalent_inputs(nights):
    by_dict = OrbitPipeline(nights)
    by_epochs = OrbitPipeline(Epochs(nights))
    nested = OrbitPipeline({k: [v] for k, v in nights.items()})
    assert by_dict.data.names == by_epochs.data.names == ("n0", "n1", "n2")
    assert by_dict._data_fingerprint() == by_epochs._data_fingerprint()
    # A list with one dataset is named after its night, as in ``Epochs``.
    assert nested.data.dataset_names == by_dict.data.dataset_names
    assert "nights=3" in repr(by_dict)


def _write_nights(folder, nights):
    paths = []
    for name, data in nights.items():
        path = folder / f"{name}.oifits"
        write_oifits(data, path)
        paths.append(str(path))
    return paths


@pytest.fixture(scope="module")
def files(nights, tmp_path_factory):
    return _write_nights(tmp_path_factory.mktemp("orbit_files"), nights)


def test_from_oifits_groups_files_into_nights(files):
    pipeline = OrbitPipeline.from_oifits(files, gap_days=0.5)
    assert len(pipeline.data.names) == 3
    assert [round(float(t)) for t in pipeline.data.times] == [
        round(float(m + 1 / 24)) for m in SEASONS[:3]
    ]
    # Files of one night split by 0h UT are one night, and a larger gap_days
    # merges nights further apart.
    straddle = {
        files[0]: {"mjd": 60000.95},
        files[1]: {"mjd": 60001.05},
        files[2]: {"mjd": 60200.0},
    }
    one = OrbitPipeline.from_oifits(files, straddle)
    assert len(one.data.names) == 2 and len(one.data.dataset_names) == 3
    apart = {
        files[0]: {"mjd": 60000.0},
        files[1]: {"mjd": 60010.0},
        files[2]: {"mjd": 60200.0},
    }
    assert len(OrbitPipeline.from_oifits(files, apart).data.names) == 3
    merged = OrbitPipeline.from_oifits(files, apart, gap_days=30.0)
    assert len(merged.data.names) == 2
    meta = {files[0]: {"epoch": "first"}, files[1]: {"epoch": "second"}}
    labelled = OrbitPipeline.from_oifits(files, meta)
    assert {"first", "second"} <= set(labelled.data.names)
    assert pipeline.inputs == [str(Path(f).resolve()) for f in files]


def test_a_file_with_no_usable_data_is_skipped_with_a_note(files, monkeypatch):
    import virgil.oidata as oidata

    real = oidata.OIData

    def fake(path, *args, **kwargs):
        if path == files[2]:
            raise ValueError("No unflagged data: every visibility is flagged.")
        return real(path, *args, **kwargs)

    monkeypatch.setattr(oidata, "OIData", fake)
    with pytest.warns(UserWarning, match="skipped"):
        pipeline = OrbitPipeline.from_oifits(files)
    assert len(pipeline.data.names) == 2
    note = next(f for f in pipeline._input_files() if "skipped" in f)
    assert note["path"].endswith(files[2].rsplit("/", 1)[-1])
    with pytest.raises(ValueError, match="bad table"):
        monkeypatch.setattr(
            oidata,
            "OIData",
            lambda *a, **k: (_ for _ in ()).throw(ValueError("bad table")),
        )
        OrbitPipeline.from_oifits(files[:1], metadata={files[0]: {}})


def test_metadata_is_checked(files):
    with pytest.raises(ValueError, match="match no input file"):
        OrbitPipeline.from_oifits(files, {"nothere.oifits": {"epoch": "x"}})
    with pytest.raises(ValueError, match="unknown keys"):
        OrbitPipeline.from_oifits(files, {files[0]: {"exclude": "bad"}})
    named = OrbitPipeline.from_oifits(
        files, {"n0.oifits": {"epoch": "by-name"}}
    )
    assert "by-name" in named.data.names
    shifted = OrbitPipeline.from_oifits(files, {files[0]: {"mjd": 60100.0}})
    assert onp.isclose(float(max(shifted.data.times)), 60100.0)


def test_exclusions_by_file_are_recorded_with_their_reason(files):
    pipeline = OrbitPipeline.from_oifits(
        files, exclude={"n2.oifits": "calibrator was resolved"}
    )
    assert len(pipeline.data.names) == 2
    record = pipeline._excluded["n2.oifits"]
    assert record["reason"] == "calibrator was resolved"
    assert len(record["datasets"]) == 1


# --- run folder, resume ------------------------------------------------------


def _started(out):
    run = json.loads((out / "run.json").read_text())
    return {k: v["started"] for k, v in run["stages"].items()}, run


@pytest.fixture(scope="module")
def epochs_run(nights, tmp_path_factory):
    out = tmp_path_factory.mktemp("orbit") / "run"
    pipeline = OrbitPipeline(nights, output=out, **TINY)
    return pipeline, pipeline.run(through="epochs")


def test_run_folder_layout_through_epochs(epochs_run):
    pipeline, res = epochs_run
    out = pipeline.output
    run = json.loads((out / "run.json").read_text())
    assert run["schema"] == SCHEMA and run["pipeline"] == "orbit"
    assert run["class"] == "OrbitPipeline"
    assert run["status"] == "partial"
    assert list(run["stages"]) == ["load", "epochs"]
    for name in ("run.json", "summary.json", "stages/load/report.json"):
        assert (out / name).exists()
    assert (out / "stages" / "epochs" / "positions.json").exists()
    assert res.summary["data"]["n_nights"] == 3
    assert len(res.summary["nights"]) == 3
    assert "search" not in res.summary and "samples.h5" not in {
        p.name for p in out.iterdir()
    }
    # Without a processed.oifits per night the loader says why.
    with pytest.raises(FileNotFoundError, match="one OIFITS per night"):
        res.data()


def test_positions_round_trip_exactly(epochs_run):
    pipeline, _ = epochs_run
    path = pipeline.output / "stages" / "epochs" / "positions.json"
    stored = json.loads(path.read_text())
    again = orbit_module._positions_json(
        orbit_module._positions_from_json(stored)
    )
    assert json.dumps(again) == json.dumps(stored)
    found = orbit_module._read_positions(pipeline)
    assert found.names == ("n0", "n1", "n2")
    # The true companion is found where the orbit puts it.
    orbit = KeplerOrbit(**TRUTH, t_ref=T_REF)
    dra, ddec, _ = orbit.relative(onp.asarray(found.mjd))
    sep = onp.hypot(onp.asarray(dra), onp.asarray(ddec))
    assert onp.allclose(onp.hypot(found.dra, found.ddec), sep, rtol=0.15)


def test_the_position_grid_widens_until_no_peak_is_on_its_edge(epochs_run):
    _, res = epochs_run
    report = json.loads(
        (res.path / "stages" / "epochs" / "report.json").read_text()
    )
    assert report["edge_nights"] == []
    assert len(report["half_widths_mas"]) >= 2  # a = 10 mas > the first box
    assert report["half_widths_mas"][-1] > report["half_widths_mas"][0]
    assert report["grid_step_mas"] <= report["grid_step_wanted_mas"] * 1.001


def _fake_search(calls, period):
    """An ``_alias_search`` stand-in that records its arguments."""

    def search(p, data, positions, times, checkpoint_dir):
        calls.append((data, positions, times, checkpoint_dir))
        return SimpleNamespace(
            best=SimpleNamespace(n=20, best={"period": period})
        )

    return search


def _robustness(
    pipeline, monkeypatch, period, chi2=(1.0, 9.0, 1.0), best_period=12.4
):
    """``_robustness`` of the epochs-stage run, with the refit faked and the
    night ``n1`` (by default) the outlier."""
    calls = []
    monkeypatch.setattr(
        orbit_module, "_alias_search", _fake_search(calls, period)
    )
    monkeypatch.setattr(
        orbit_module, "_null_verdict", lambda *a: {"verdict": "orbit"}
    )
    monkeypatch.setattr(
        pipeline,
        "_report",
        lambda stage: {
            "chi2": {
                "nights": {
                    f"n{k}": {"chi2": c, "n": 1} for k, c in enumerate(chi2)
                }
            }
        },
    )
    result = SimpleNamespace(
        best=SimpleNamespace(n=20, best={"period": best_period})
    )
    return orbit_module._robustness(pipeline, result, "orbit"), calls


def test_the_most_outlying_night_is_dropped_and_the_verdicts_compared(
    epochs_run, monkeypatch
):
    pipeline, _ = epochs_run
    report, calls = _robustness(pipeline, monkeypatch, 12.4)
    assert report["night"] == "n1" and report["chi2_per_n"] == 9.0
    assert report["full"] == {"verdict": "orbit", "n": 20}
    assert report["dropped"]["same_band"] and report["dropped"]["n"] == 20
    ((data, positions, times, bands),) = calls
    # The refit sees the other two nights, their positions and times, and
    # checkpoints its bands apart from the search's.
    assert len(data) == 2 and positions.names == ("n0", "n2")
    assert onp.allclose(
        positions.dra, orbit_module._read_positions(pipeline).dra[[0, 2]]
    )
    assert onp.allclose(times, pipeline.data.times[[0, 2]])
    assert bands.name == "bands_drop"
    assert bands.parent == orbit_module._band_checkpoints(pipeline).parent
    # A period a whole cycle away over the span is another alias.
    span = float(onp.ptp(pipeline.data.times))
    moved, _ = _robustness(
        pipeline, monkeypatch, 1.0 / (1.0 / 12.4 - 1.0 / span)
    )
    assert not moved["dropped"]["same_band"]
    assert _checks.night_dropped(moved).status == "warn"


def test_a_period_just_across_a_band_edge_is_the_same_alias(
    epochs_run, monkeypatch
):
    """Dropping an end night shifts the band numbers; periods a third of a
    cycle apart over the span are one alias even across a band edge."""
    pipeline, _ = epochs_run
    span = float(onp.ptp(pipeline.data.times))
    edge = 1.0 / (1.0 / 12.6 - 0.3 / span)  # 0.3 cycle from 12.6 d
    near, _ = _robustness(pipeline, monkeypatch, edge, best_period=12.6)
    assert near["dropped"]["same_band"]


def test_a_failed_refit_or_too_few_usable_nights_skip_the_check(
    epochs_run, monkeypatch
):
    pipeline, _ = epochs_run

    def fail(*args, **kwargs):
        raise ValueError("fewer than 2 seeded nights")

    _robustness(pipeline, monkeypatch, 12.4)  # fakes the search report
    monkeypatch.setattr(orbit_module, "_alias_search", fail)
    result = SimpleNamespace(best=SimpleNamespace(n=20, best={"period": 12.4}))
    skipped = orbit_module._robustness(pipeline, result, "orbit")
    assert "refit failed" in skipped["skipped"]
    assert _checks.night_dropped(skipped).status == "pass"
    found = orbit_module._read_positions(pipeline)
    blocked = dataclasses.replace(
        found, flags=tuple((NightFlag("low-gap", "x"),) for _ in found.names)
    )
    monkeypatch.setattr(orbit_module, "_read_positions", lambda p: blocked)
    flagged = orbit_module._robustness(pipeline, result, "orbit")
    assert "usable" in flagged["skipped"]


def test_nights_without_data_are_never_the_outlier():
    epochs = SimpleNamespace(dataset_names=("a", "b"), epoch_of=(0, 1))
    nights = {
        "a": {"chi2": float("nan"), "n": 0},
        "b": {"chi2": 3.0, "n": 2},
    }
    assert orbit_module._outlying_night(epochs, nights) == (1, [1], 1.5)
    nights["b"]["n"] = 0
    assert orbit_module._outlying_night(epochs, nights) is None


def test_the_robustness_check_is_skipped_below_three_nights_or_when_off(
    nights, tmp_path
):
    two = OrbitPipeline(
        nights, output=tmp_path / "two", exclude={"n2": "no"}, **TINY
    )
    result = SimpleNamespace(best=None)
    assert orbit_module._robustness(two, result, "orbit") == {
        "skipped": "fewer than 3 nights"
    }
    off = OrbitPipeline(
        nights, output=tmp_path / "off", drop_outlier_night=False, **TINY
    )
    assert "off" in orbit_module._robustness(off, result, "orbit")["skipped"]
    assert _checks.night_dropped({"skipped": "x"}).status == "pass"


def test_a_positional_output_warns_and_still_works(nights, tmp_path):
    with pytest.warns(FutureWarning, match="output"):
        pipeline = OrbitPipeline(nights, tmp_path / "pos", **TINY)
    assert pipeline.output == tmp_path / "pos"


def test_resume_skips_completed_stages(nights, tmp_path, monkeypatch):
    out = tmp_path / "run"
    OrbitPipeline(nights, output=out, **TINY).run(through="load")
    before, _ = _started(out)
    ran = []
    monkeypatch.setattr(orbit_module, "_load", lambda p: ran.append("load"))
    res = OrbitPipeline(nights, output=out, **TINY).run(through="epochs")
    assert ran == [] and res.status == "partial"
    after, _ = _started(out)
    assert after["load"] == before["load"] and "epochs" in after

    with pytest.raises(ConfigMismatchError, match="ecc_max"):
        OrbitPipeline(nights, output=out, **dict(TINY, ecc_max=0.5)).run()
    shifted = {
        k: v.with_error_floor(absolute={"vis": 0.5}) for k, v in nights.items()
    }
    with pytest.raises(ConfigMismatchError, match="data differ"):
        OrbitPipeline(shifted, output=out, **TINY).run()
    # Dropping a night is a different run too.
    with pytest.raises(ConfigMismatchError, match="exclude"):
        OrbitPipeline(nights, output=out, exclude={"n2": "no"}, **TINY).run()


def test_band_checkpoints_survive_a_rerun_but_not_a_fresh_run(
    nights, tmp_path
):
    out = tmp_path / "run"
    pipeline = OrbitPipeline(nights, output=out, **TINY)
    pipeline.run(through="load")
    bands = orbit_module._band_checkpoints(pipeline)
    bands.mkdir(parents=True)
    (bands / "band_20.json").write_text("{}")
    # Re-running the stage (its outputs are stale) keeps the bands.
    pipeline._discard(pipeline._stages()[2])
    assert (bands / "band_20.json").exists()
    # A fresh run starts over.
    OrbitPipeline(nights, output=out, **TINY).run(through="load", resume=False)
    assert not bands.exists()


def test_a_failing_stage_is_recorded(nights, tmp_path, monkeypatch):
    out = tmp_path / "run"

    def boom(p):
        raise RuntimeError("alias engine exploded")

    monkeypatch.setattr(orbit_module, "_epochs", boom)
    with pytest.raises(RuntimeError, match="exploded"):
        OrbitPipeline(nights, output=out, **TINY).run(through="epochs")
    run = json.loads((out / "run.json").read_text())
    assert run["status"] == "failed" and run["error"]["stage"] == "epochs"
    assert run["stages"]["load"]["status"] == "complete"


# --- CLI -----------------------------------------------------------------


def test_cli_orbit_runs_from_oifits_with_metadata(files, tmp_path, capsys):
    meta = tmp_path / "meta.json"
    meta.write_text(json.dumps({files[0]: {"epoch": "first"}}))
    out = tmp_path / "cli"
    args = ["orbit", *files, "--output", str(out), "--through", "load"]
    args += ["--metadata", str(meta)]
    for key, value in TINY.items():
        args += ["--set", f"{key}={json.dumps(value)}"]
    assert cli.main(args) == 0
    run = json.loads((out / "run.json").read_text())
    assert run["status"] == "partial" and run["pipeline"] == "orbit"
    assert [Path(f["path"]).name for f in run["inputs"]["files"]] == [
        Path(f).name for f in files
    ]
    assert run["config"]["max_grid"] == 25
    assert "data.n_nights" in capsys.readouterr().out
    assert cli.main(["validate", str(out)]) == 0


def test_from_config_reads_the_metadata_key(files, tmp_path):
    config = tmp_path / "orbit.json"
    config.write_text(
        json.dumps(
            {
                "pipeline": "orbit",
                "data": [str(f) for f in files],
                "metadata": {str(files[0]): {"epoch": "first"}},
                "settings": {"max_grid": 25},
                "output": "out",
            }
        )
    )
    pipeline = OrbitPipeline.from_config(config)
    assert "first" in pipeline.data.names
    assert pipeline.settings["max_grid"] == 25


# --- quicklook ---------------------------------------------------------------


def test_quicklook_cells_are_valid_code(tmp_path):
    reports = _fake_reports()
    sections, _ = orbit_module._summarize(OrbitPipeline.defaults(), reports)
    sections["modes"]["winner"]["p"] = 1.0
    sections["modes"]["no_orbit"]["verdict"] = "orbit"
    summary = {**sections, "worst_check": "pass", "checks": []}
    run = {
        "stages": {
            k: {"status": "complete"}
            for k in ("load", "epochs", "search", "modes", "posterior")
        }
    }
    cells = _quicklook._SECTIONS["orbit"](run, summary, tmp_path)
    assert len(cells) == 4
    for markdown, code in cells:
        assert markdown.startswith("##") or markdown[0].isalpha()
        compile(code, "<cell>", "exec")
    assert "OrbitalBinary" not in "".join(m for m, _ in cells)  # target-free
    assert "refit was skipped" in cells[2][0]
    summary["modes"]["robustness"] = _ROBUST
    again = _quicklook._SECTIONS["orbit"](run, summary, tmp_path)
    assert "Without night n1 (raw χ²/N = 3)" in again[2][0]
    # A modes report from before the check (a resumed run folder).
    reports = _fake_reports()
    del reports["modes"]["robustness"]
    sections, checks = orbit_module._summarize(
        OrbitPipeline.defaults(), reports
    )
    assert (
        "not run (older run)"
        in next(c for c in checks if c.name == "night_dropped").message
    )
    summary["modes"] = sections["modes"]
    older = _quicklook._SECTIONS["orbit"](run, summary, tmp_path)
    assert "not run (older run)" in older[2][0]
    partial = _quicklook._SECTIONS["orbit"](
        {"stages": {"load": {"status": "complete"}}}, summary, tmp_path
    )
    assert partial == []


def test_default_model_and_digest_do_not_need_a_template(nights):
    pipeline = OrbitPipeline(nights)
    assert pipeline.model is None
    assert pipeline._model_digest() == {
        "class": "OrbitalBinary",
        "digest": None,
    }


def test_the_class_is_exported_lazily():
    import importlib

    module = importlib.import_module("virgil.pipeline")
    assert module.OrbitPipeline is OrbitPipeline
    assert "OrbitPipeline" in dir(module)


def test_the_position_grid_stops_at_the_lattice_alias_limit(
    nights, tmp_path, monkeypatch
):
    """On lattice coverage the default box is not widened past one alias
    cell (here a limit of 4 mas); an explicit ``max_sep_mas`` is."""
    monkeypatch.setattr(orbit_module, "lattice_alias_limit", lambda d: 4.0)
    res = OrbitPipeline(nights, output=tmp_path / "run", **TINY).run(
        through="epochs"
    )
    report = json.loads(
        (res.path / "stages" / "epochs" / "report.json").read_text()
    )
    assert report["half_widths_mas"] == [4.0]


@pytest.mark.slow  # a real search: minutes
@pytest.mark.timeout(1200)
def test_the_real_refit_runs_on_a_tiny_case(epochs_run):
    """The check's refit of two nights is the real engine call."""
    pipeline, _ = epochs_run
    epochs = pipeline.data
    with jax.enable_x64(True):
        # Only the check's refit is real; the full search is a stand-in.
        full = SimpleNamespace(
            best=SimpleNamespace(n=20, best={"period": TRUTH["period"]})
        )
        pipeline._report = lambda stage: {
            "chi2": {
                "nights": {
                    n: {"chi2": c, "n": 1}
                    for n, c in zip(epochs.dataset_names, (1.0, 9.0, 1.0))
                }
            }
        }
        report = orbit_module._robustness(pipeline, full, "orbit")
    assert "skipped" not in report, report  # the refit itself must run
    assert report["night"] == epochs.epoch_of[1]
    assert report["dropped"]["verdict"] in ("orbit", "no-orbit")
    assert report["dropped"]["n"] <= full.best.n


def test_an_empty_period_window_names_the_pipeline_setting(monkeypatch):
    import virgil.aliases as aliases

    def empty(*args, **kwargs):  # an empty window
        aliases.period_window([0.0, 100.0], 20.0, 10.0, resolution=0.1)

    monkeypatch.setattr(aliases, "fit_orbit_aliases", empty)
    settings = {"period_range": None}
    pipeline = SimpleNamespace(
        settings=type(
            "Settings", (dict,), {"__missing__": lambda self, key: (1, 2)}
        )(settings)
    )
    with pytest.raises(ValueError, match="Set period_range"):
        orbit_module._alias_search(pipeline, [], None, None, None)
