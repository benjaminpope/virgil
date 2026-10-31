"""The pipeline checks: pure functions with pass, warn and fail branches."""

import json

import pytest

from virgil.pipeline import Check
from virgil.pipeline import _checks as checks


def _status(check):
    assert isinstance(check, Check)
    json.dumps(check.to_dict())  # always strict-JSON ready
    return check.status


@pytest.mark.parametrize(
    "chi2, status",
    [
        (1.0, "pass"),
        (3.0, "warn"),
        (0.2, "warn"),
        (8.0, "fail"),
        (float("nan"), "fail"),
    ],
)
def test_chi2_reduced(chi2, status):
    check = checks.chi2_reduced("chi2", chi2, label="the model")
    assert _status(check) == status


def test_chi2_message_is_templated():
    check = checks.chi2_reduced("chi2", 3.2, label="the best-fit binary")
    assert check.message == (
        "χ²/N = 3.2 for the best-fit binary on quoted errors: errors likely "
        "underestimated."
    )


@pytest.mark.parametrize(
    "scales, status",
    [
        ({"vis_scale": 1.1}, "pass"),
        ({"vis_scale": 3.0}, "warn"),
        ({"phi_scale": 9.0}, "fail"),
    ],
)
def test_error_scale_flags_large_s(scales, status):
    assert _status(checks.error_scale(scales)) == status


@pytest.mark.parametrize(
    "local, glob, status, word",
    [
        (6.0, 5.0, "pass", "detected"),
        (4.0, 2.0, "warn", "Marginal"),
        (1.0, 0.1, "warn", "No companion"),
    ],
)
def test_detection(local, glob, status, word):
    check = checks.detection(local, glob, threshold=3.0)
    assert _status(check) == status
    assert word in check.message


def test_detection_prints_a_lower_bound():
    check = checks.detection(40.0, 3.09, threshold=3.0, bound=True)
    assert _status(check) == "pass" and "≥3.09σ" in check.message
    assert "≥" not in checks.detection(40.0, 3.09, threshold=3.0).message


def test_detection_survives_a_missing_global_significance():
    """A global sigma stored as None (not finite) must not break the text."""
    check = checks.detection(14.0, None, threshold=3.0)
    assert _status(check) == "warn" and "Marginal" in check.message
    assert "14.00σ local but n/a" in check.message
    check = checks.detection(1.0, float("nan"), threshold=3.0)
    assert _status(check) == "warn" and "1.00σ local" in check.message
    assert checks.format_sigma(None) == checks.format_sigma(float("inf"))
    assert checks.format_sigma(3.14159) == "3.14σ"


def test_grid_edge():
    assert _status(checks.grid_edge([3, 4, 2], [9, 9, 6], "xyz")) == "pass"
    check = checks.grid_edge([0, 4, 5], [9, 9, 6], ("dra", "ddec", "flux"))
    assert _status(check) == "warn"
    assert "dra, flux" in check.message


def test_prior_bound():
    assert _status(checks.prior_bound({"flux": 0.0, "sep": 0.01})) == "pass"
    check = checks.prior_bound({"flux": 0.3, "sep": 0.0})
    assert _status(check) == "warn" and "flux" in check.message


@pytest.mark.parametrize(
    "value, status", [(1.001, "pass"), (1.03, "warn"), (1.2, "fail")]
)
def test_r_hat(value, status):
    assert _status(checks.r_hat(value)) == status


@pytest.mark.parametrize(
    "value, status", [(2000.0, "pass"), (200.0, "warn"), (20.0, "fail")]
)
def test_ess(value, status):
    assert _status(checks.ess(value)) == status


@pytest.mark.parametrize(
    "value, status", [(0.0, "pass"), (0.005, "warn"), (0.1, "fail")]
)
def test_divergences(value, status):
    assert _status(checks.divergences(value)) == status


def test_residual_normality():
    assert _status(checks.residual_normality(0.1, 0.2, 1000)) == "pass"
    assert _status(checks.residual_normality(1.5, 0.2, 1000)) == "warn"
    assert _status(checks.residual_normality(0.0, 9.0, 1000)) == "warn"
    assert _status(checks.residual_normality(5.0, 9.0, 4)) == "pass"


def test_field_of_view():
    assert _status(checks.field_of_view(200.0, 50.0, 700.0)) == "pass"
    assert "inside" in checks.field_of_view(20.0, 50.0, 700.0).message
    assert "outside" in checks.field_of_view(900.0, 50.0, 700.0).message


def test_check_round_trip_and_status_validation():
    check = checks.r_hat(1.2)
    assert Check.from_dict(check.to_dict()) == check
    with pytest.raises(ValueError, match="status"):
        Check("x", "ok", 1.0, None, "")


def test_worst_status():
    made = [checks.r_hat(1.0), checks.ess(200.0)]
    assert checks.worst_status(made) == "warn"
    assert checks.worst_status(made + [checks.r_hat(2.0)]) == "fail"
    assert checks.worst_status([]) == "pass"


def test_checks_are_deterministic():
    assert checks.detection(4.0, 2.0) == checks.detection(4.0, 2.0)


def _w(message, category="RuntimeWarning", stage="search"):
    return {"stage": stage, "category": category, "message": message}


def test_stage_warning_checks_map_known_messages():
    from virgil.pipeline._checks import stage_warning_checks

    checks = {
        c.name: c
        for c in stage_warning_checks(
            [
                _w(
                    "f(): the optimizer did not converge at 2 of 6561 grid positions"
                ),
                _w(
                    "g(): the optimizer did not converge at 5 of 6561 grid positions"
                ),
                _w(
                    "detection_statistics(): the flux axis does not resolve the "
                    "likelihood peak (0.28 steps across its FWHM, under 2), so "
                    "log_bayes_factor is inaccurate"
                ),
                _w(
                    "absil_limits(): 1 limits fell outside flux_bounds=(1, 2) and were clipped"
                ),
            ]
        )
    }
    assert list(checks) == [
        "convergence",
        "flux_axis_resolution",
        "limits_clipped",
    ]
    assert checks["convergence"].value == pytest.approx(5 / 6561)
    assert checks["flux_axis_resolution"].value == pytest.approx(0.28)
    assert checks["limits_clipped"].value == 1
    assert all(c.status == "warn" for c in checks.values())


def test_stage_warning_checks_tolerate_rewording_and_unknowns():
    from virgil.pipeline._checks import stage_warning_checks

    (reworded,) = stage_warning_checks(
        [_w("the optimizer failed to converge somewhere")]
    )
    assert reworded.name == "convergence" and reworded.value is None
    (other,) = stage_warning_checks([_w("surprise", "UserWarning")])
    assert other.name == "stage_warnings" and other.value == 1
    assert stage_warning_checks([_w("old api", "DeprecationWarning")]) == []
    assert stage_warning_checks([]) == []


def test_convergence_reports_worst_fraction_across_grid_sizes():
    found = checks.convergence(
        [
            _w("the optimizer did not converge at 5 of 6561 grid positions"),
            _w("the optimizer did not converge at 3 of 10 grid positions"),
        ]
    )
    assert found.value == pytest.approx(0.3)


def test_flux_axis_resolution_reads_exponent_notation():
    found = checks.flux_axis_resolution(
        [
            _w(
                "the flux axis does not resolve the likelihood peak (3e-05 steps across its FWHM, under 2)"
            )
        ]
    )
    assert found.value == pytest.approx(3e-5)


# --- the checks of DiskPipeline ---------------------------------------------

NAN = float("nan")


def test_length_scale_needs_the_evidence_to_separate_the_grid():
    ells = [1.7, 3.5, 6.9]
    found = checks.length_scale([-10.0, -8.0, -9.5], ells, 1.5, nats=3.0)
    assert _status(found) == "warn" and found.value == pytest.approx(2.0)
    assert "do not measure ℓ" in found.message
    found = checks.length_scale([-20.0, -8.0, -9.5], ells, 1.5)
    assert _status(found) == "pass"
    # The message quotes the squared-exponential equivalent of the best ℓ.
    assert (
        "3.5 mas, a squared-exponential length of 2.02 mas, 1.3×"
        in found.message
    )
    assert _status(checks.length_scale([-1.0, NAN], ells[:2], 1.5)) == "warn"
    assert _status(checks.length_scale([-1.0], ells[:1], 1.5)) == "warn"


def _ring_rows(bics, evidences, widths=(2.0,)):
    return [
        {"n": n, "bic": b, "log_evidence": z, "widths": list(widths) * n}
        for n, (b, z) in enumerate(zip(bics, evidences), start=1)
    ]


def test_ring_count_margin_is_the_smaller_of_bic_and_evidence():
    # Two rings win by 10 nats of BIC/2 but only 1 nat of evidence: unclear.
    rows = _ring_rows([120.0, 100.0], [-60.0, -59.0])
    found = checks.ring_count(2, rows, 1.0, nats=3.0)
    assert _status(found) == "warn" and found.value == pytest.approx(1.0)
    rows = _ring_rows([120.0, 100.0], [-70.0, -59.0])
    assert _status(checks.ring_count(2, rows, 1.0)) == "pass"
    # Without an evidence the BIC alone decides.
    rows = _ring_rows([120.0, 100.0], [None, None])
    assert _status(checks.ring_count(2, rows, 1.0)) == "pass"
    # The last count tried has one neighbour.
    rows = _ring_rows([100.0, 101.0, 100.5], [None] * 3)
    assert _status(checks.ring_count(1, rows, 1.0)) == "warn"


def test_ring_count_warns_when_a_better_count_was_left_out_as_split():
    rows = _ring_rows([120.0, 100.0], [-70.0, -59.0])
    assert _status(checks.ring_count(2, rows, 1.0)) == "pass"
    found = checks.ring_count(2, rows, 1.0, split=[3])
    assert _status(found) == "warn" and "3 rings fit better" in found.message


def test_ring_count_flags_rings_narrower_than_the_half_fringe():
    rows = _ring_rows([120.0, 100.0], [-70.0, -59.0], widths=(0.4,))
    found = checks.ring_count(2, rows, 1.0)
    assert _status(found) == "warn"
    assert "ring 1, 2 is narrower than the half-fringe" in found.message
    single = _ring_rows([100.0], [-59.0], widths=(2.0,))
    assert _status(checks.ring_count(1, single, 1.0)) == "pass"


@pytest.mark.parametrize(
    "fraction, correlation, status",
    [
        (0.01, 0.2, "pass"),
        (0.30, 0.2, "warn"),
        (0.01, -0.95, "warn"),
        (0.01, None, "pass"),
        (NAN, 0.0, "warn"),
    ],
)
def test_inner_edge(fraction, correlation, status):
    found = checks.inner_edge(fraction, correlation, 1.0)
    assert _status(found) == status


def test_inner_edge_message_keeps_its_case():
    found = checks.inner_edge(0.5, 0.95, 1.0)
    assert found.message.startswith("50 per cent of the disk's light")
    assert "(|ρ| = 0.95)" in found.message


@pytest.mark.parametrize(
    "fraction, status", [(1e-6, "pass"), (0.2, "warn"), (NAN, "warn")]
)
def test_outer_edge(fraction, status):
    found = checks.outer_edge(fraction, 1.0)
    assert _status(found) == status
    if status == "warn" and fraction == fraction:
        assert "r_out_mas" in found.message


@pytest.mark.parametrize(
    "inc, pa_sd, minima, status",
    [
        (50.0, 2.0, 1, "pass"),
        (8.0, 2.0, 1, "warn"),
        (50.0, 80.0, 1, "warn"),
        (50.0, None, 1, "pass"),
        (50.0, 2.0, 2, "warn"),
    ],
)
def test_inclination(inc, pa_sd, minima, status):
    assert _status(checks.inclination(inc, pa_sd, minima)) == status


def test_inclination_message_names_the_problems():
    found = checks.inclination(8.0, 2.0, 3)
    assert "position angle is unconstrained" in found.message
    assert "3 comparable minima" in found.message
    assert found.message[0].isupper()


def test_absorbed_companion():
    assert _status(checks.absorbed_companion(5.0, None, 1)) == "pass"
    assert _status(checks.absorbed_companion(None, None, 1)) == "pass"
    found = checks.absorbed_companion(60.0, None, 1)
    assert _status(found) == "warn" and "remains" in found.message
    found = checks.absorbed_companion(60.0, 6.0, 1)
    assert _status(found) == "warn"
    assert "from 1 to 2 removes it" in found.message
    assert "asymmetry of the disk rather than a companion" in found.message
    assert "degree 2 can absorb a compact source" in found.message
    found = checks.absorbed_companion(60.0, 55.0, 1)
    assert "removes" not in found.message


# --- every status branch, with the name and the JSON round trip --------------

INF = float("inf")


def _lobes(*deltas):
    return [
        {"diam_mas": 1.0 + i, "diam_bounds_mas": [0.5, 9.0], "delta_chi2": d}
        for i, d in enumerate(deltas)
    ]


def _lengths(**kwargs):
    return checks.length_scale(
        [-20.0, -8.0, -9.5], [1.7, 3.5, 6.9], 1.5, **kwargs
    )


_BRANCHES = [
    # (id, documented name, a call returning the Check, expected status)
    (
        "chi2-inf",
        "c",
        lambda: checks.chi2_reduced("c", INF, label="x"),
        "fail",
    ),
    (
        "chi2-none",
        "c",
        lambda: checks.chi2_reduced("c", None, label="x"),
        "fail",
    ),
    (
        "chi2-pass",
        "c",
        lambda: checks.chi2_reduced("c", 1.5, label="x"),
        "pass",
    ),
    ("scale-empty", "error_scale", lambda: checks.error_scale({}), "pass"),
    (
        "scale-estimated",
        "error_scale",
        lambda: checks.error_scale({"v": 3.0}, kind="estimated"),
        "warn",
    ),
    ("detect-nan", "detection", lambda: checks.detection(NAN, NAN), "warn"),
    ("detect-none", "detection", lambda: checks.detection(None, None), "warn"),
    (
        "grid-flat-axis",
        "grid_edge",
        lambda: checks.grid_edge([0], [1], "x"),
        "pass",
    ),
    (
        "prior-bound-empty",
        "prior_bound",
        lambda: checks.prior_bound({}),
        "pass",
    ),
    (
        "prior-bound-at-warn",
        "prior_bound",
        lambda: checks.prior_bound({"a": 0.05}),
        "pass",
    ),
    (
        "start-at-bound",
        "start_at_bound",
        lambda: checks.at_bound(("flux", "sep"), start=True),
        "warn",
    ),
    ("at-bound-none", "at_bound", lambda: checks.at_bound([]), "pass"),
    ("at-bound-some", "at_bound", lambda: checks.at_bound(("flux",)), "warn"),
    ("r-hat-nan", "r_hat", lambda: checks.r_hat(NAN), "fail"),
    ("r-hat-inf", "r_hat", lambda: checks.r_hat(INF), "fail"),
    ("r-hat-none", "r_hat", lambda: checks.r_hat(None), "fail"),
    ("ess-nan", "ess", lambda: checks.ess(NAN), "fail"),
    ("ess-none", "ess", lambda: checks.ess(None), "fail"),
    ("div-pass", "divergences", lambda: checks.divergences(0.0), "pass"),
    (
        "div-walls-exposed",
        "divergences",
        lambda: checks.divergences(
            0.1, walls={"radius_sd": 3.0, "exposed": 0.4}
        ),
        "fail",
    ),
    (
        "div-walls-barely-exposed",
        "divergences",
        lambda: checks.divergences(
            0.1, walls={"radius_sd": 3.0, "exposed": 0.01}
        ),
        "fail",
    ),
    (
        "div-walls-no-key",
        "divergences",
        lambda: checks.divergences(0.1, walls={"radius_sd": 3.0}),
        "fail",
    ),
    (
        "div-walls-irrelevant",
        "divergences",
        lambda: checks.divergences(
            0.005, walls={"radius_sd": 3.0, "exposed": 1.0}
        ),
        "warn",
    ),
    (
        "normality-nan-skew",
        "residual_normality",
        lambda: checks.residual_normality(NAN, 0.0, 1000),
        "warn",
    ),
    (
        "normality-tiny-n",
        "residual_normality",
        lambda: checks.residual_normality(NAN, NAN, 7),
        "pass",
    ),
    (
        "fov-at-resolution",
        "field_of_view",
        lambda: checks.field_of_view(50.0, 50.0, 700.0),
        "pass",
    ),
    (
        "fov-at-fov",
        "field_of_view",
        lambda: checks.field_of_view(700.0, 50.0, 700.0),
        "pass",
    ),
    (
        "resolution-nan",
        "resolution",
        lambda: checks.resolution_regime(NAN, 1e8),
        "fail",
    ),
    (
        "resolution-inf",
        "resolution",
        lambda: checks.resolution_regime(1.0, INF),
        "fail",
    ),
    (
        "resolution-unresolved",
        "resolution",
        lambda: checks.resolution_regime(1.0, 1e7),
        "fail",
    ),
    (
        "resolution-short-of-null",
        "resolution",
        lambda: checks.resolution_regime(1.0, 1e8),
        "warn",
    ),
    (
        "resolution-past-null",
        "resolution",
        lambda: checks.resolution_regime(1.0, 3e8),
        "pass",
    ),
    ("lobes-none", "multimodal", lambda: checks.multimodal([]), "pass"),
    (
        "lobes-one",
        "multimodal",
        lambda: checks.multimodal(_lobes(0.0)),
        "pass",
    ),
    (
        "lobes-clear",
        "multimodal",
        lambda: checks.multimodal(_lobes(0.0, 100.0)),
        "pass",
    ),
    (
        "lobes-rival-unsorted",
        "multimodal",
        lambda: checks.multimodal(_lobes(40.0, 5.0, 0.0)),
        "warn",
    ),
    (
        "lobes-nan-gap",
        "multimodal",
        lambda: checks.multimodal(_lobes(0.0, NAN)),
        "warn",
    ),
    (
        "gain-nan",
        "gain",
        lambda: checks.model_gain(
            "gain", NAN, 1.0, 100, n_extra=3, simple="a", complex_="b"
        ),
        "fail",
    ),
    (
        "gain-negative",
        "gain",
        lambda: checks.model_gain(
            "gain", 10.0, 20.0, 100, n_extra=3, simple="a", complex_="b"
        ),
        "warn",
    ),
    (
        "gain-preferred",
        "gain",
        lambda: checks.model_gain(
            "gain", 200.0, 100.0, 100, n_extra=3, simple="a", complex_="b"
        ),
        "pass",
    ),
    (
        "gain-adequate",
        "gain",
        lambda: checks.model_gain(
            "gain", 101.0, 100.0, 100, n_extra=3, simple="a", complex_="b"
        ),
        "pass",
    ),
    (
        "gain-slightly-negative",
        "gain",
        lambda: checks.model_gain(
            "gain", 100.0, 100.5, 1, n_extra=1, simple="a", complex_="b"
        ),
        "pass",
    ),
    (
        "constrained-empty",
        "pc",
        lambda: checks.prior_constrained("pc", {}),
        "pass",
    ),
    (
        "constrained-data",
        "pc",
        lambda: checks.prior_constrained("pc", {"a": 0.1, "b": 0.79}),
        "pass",
    ),
    (
        "constrained-prior",
        "pc",
        lambda: checks.prior_constrained("pc", {"a": 0.1, "b": 0.8}),
        "warn",
    ),
    (
        "constrained-nan",
        "pc",
        lambda: checks.prior_constrained("pc", {"a": NAN}),
        "warn",
    ),
    (
        "lengths-nan",
        "length_scale",
        lambda: checks.length_scale([1.0, INF], [1, 2], 1.0),
        "warn",
    ),
    (
        "lengths-one",
        "length_scale",
        lambda: checks.length_scale([1.0], [1], 1.0),
        "warn",
    ),
    (
        "lengths-edge-low-extended",
        "length_scale",
        lambda: _lengths(edge="low", extended="low"),
        "warn",
    ),
    (
        "lengths-edge-low-floor",
        "length_scale",
        lambda: _lengths(edge="low", extended=None),
        "warn",
    ),
    (
        "lengths-edge-high-extended",
        "length_scale",
        lambda: _lengths(edge="high", extended="high"),
        "warn",
    ),
    (
        "lengths-edge-high-plain",
        "length_scale",
        lambda: _lengths(edge="high"),
        "warn",
    ),
    (
        "lengths-edge-other-side",
        "length_scale",
        lambda: _lengths(edge="high", extended="low"),
        "warn",
    ),
    ("lengths-flat", "length_scale", lambda: _lengths(nats=50.0), "warn"),
    (
        "ring-nan-width",
        "ring_count",
        lambda: checks.ring_count(
            1,
            [{"n": 1, "bic": 1.0, "log_evidence": None, "widths": [NAN]}],
            1.0,
        ),
        "warn",
    ),
    (
        "ring-lowest-count",
        "ring_count",
        lambda: checks.ring_count(
            1, _ring_rows([100.0, 140.0], [None, None]), 1.0
        ),
        "pass",
    ),
    (
        "inner-inf",
        "inner_edge",
        lambda: checks.inner_edge(INF, 0.0, 1.0),
        "warn",
    ),
    (
        "inner-nan-correlation",
        "inner_edge",
        lambda: checks.inner_edge(0.5, NAN, 1.0),
        "warn",
    ),
    (
        "inner-at-limits",
        "inner_edge",
        lambda: checks.inner_edge(0.1, 0.9, 1.0),
        "pass",
    ),
    (
        "outer-at-limit",
        "outer_edge",
        lambda: checks.outer_edge(0.01, 1.0),
        "pass",
    ),
    ("outer-inf", "outer_edge", lambda: checks.outer_edge(INF, 1.0), "warn"),
    (
        "inc-nan",
        "inclination",
        lambda: checks.inclination(NAN, 1.0, 1),
        "warn",
    ),
    (
        "inc-at-minimum",
        "inclination",
        lambda: checks.inclination(15.0, 45.0, 1),
        "pass",
    ),
    (
        "inc-nan-pa",
        "inclination",
        lambda: checks.inclination(50.0, NAN, 1),
        "warn",
    ),
    (
        "absorbed-nan",
        "absorbed_companion",
        lambda: checks.absorbed_companion(NAN, None, 1),
        "pass",
    ),
    (
        "absorbed-at-threshold",
        "absorbed_companion",
        lambda: checks.absorbed_companion(25.0, NAN, 2),
        "warn",
    ),
    (
        "render-nan-model",
        "render_consistency",
        lambda: checks.render_consistency(NAN, 1.0),
        "warn",
    ),
    (
        "render-inf",
        "render_consistency",
        lambda: checks.render_consistency(1.0, INF),
        "warn",
    ),
    (
        "render-same",
        "render_consistency",
        lambda: checks.render_consistency(1.3, 1.3),
        "pass",
    ),
    (
        "render-floor-only",
        "render_consistency",
        lambda: checks.render_consistency(0.0, 0.02),
        "pass",
    ),
    (
        "render-30-per-cent-worse",
        "render_consistency",
        lambda: checks.render_consistency(1.0, 1.3),
        "warn",
    ),
    (
        "render-better-by-30-per-cent",
        "render_consistency",
        lambda: checks.render_consistency(1.0, 0.7),
        "warn",
    ),
    (
        "render-zero-model",
        "render_consistency",
        lambda: checks.render_consistency(0.0, 0.5),
        "warn",
    ),
    (
        "structure-nan-spread",
        "structure_unconstrained",
        lambda: checks.structure_unconstrained(NAN, 0.1),
        "pass",
    ),
    (
        "structure-no-correlation",
        "structure_unconstrained",
        lambda: checks.structure_unconstrained(0.5, None),
        "pass",
    ),
    (
        "structure-flat-and-unstable",
        "structure_unconstrained",
        lambda: checks.structure_unconstrained(0.5, 0.3),
        "warn",
    ),
    (
        "structure-flat-but-stable",
        "structure_unconstrained",
        lambda: checks.structure_unconstrained(0.5, 0.95),
        "pass",
    ),
    (
        "structure-separated-but-unstable",
        "structure_unconstrained",
        lambda: checks.structure_unconstrained(30.0, 0.3),
        "pass",
    ),
    (
        "structure-at-limits",
        "structure_unconstrained",
        lambda: checks.structure_unconstrained(3.0, 0.8),
        "pass",
    ),
    (
        "convergence-empty",
        "convergence",
        lambda: checks.convergence([]),
        "warn",
    ),
    (
        "convergence-zero-grid",
        "convergence",
        lambda: checks.convergence(
            [_w("did not converge at 5 of 0 grid positions")]
        ),
        "warn",
    ),
    (
        "flux-axis-empty",
        "flux_axis_resolution",
        lambda: checks.flux_axis_resolution([]),
        "warn",
    ),
    (
        "limits-empty",
        "limits_clipped",
        lambda: checks.limits_clipped([]),
        "warn",
    ),
    (
        "stage-warnings-many",
        "stage_warnings",
        lambda: checks.stage_warnings([_w(f"w{i}") for i in range(5)]),
        "warn",
    ),
]


@pytest.mark.parametrize(
    "name, make, status",
    [pytest.param(n, m, s, id=i) for i, n, m, s in _BRANCHES],
)
def test_every_branch_returns_its_named_status_and_a_json_ready_check(
    name, make, status
):
    check = make()
    assert isinstance(check, Check)
    assert (check.name, check.status) == (name, status)
    assert isinstance(check.message, str) and check.message.strip()
    record = json.loads(json.dumps(check.to_dict(), allow_nan=False))
    assert Check.from_dict(record).to_dict() == record


def test_a_stray_wall_hit_is_not_blamed_for_the_divergences():
    few = checks.divergences(0.1, walls={"radius_sd": 3, "exposed": 0.01})
    many = checks.divergences(0.1, walls={"radius_sd": 3, "exposed": 0.4})
    assert few.status == many.status == "fail"
    assert "hard wall" not in few.message
    assert "hard wall" in many.message


@pytest.mark.parametrize(
    "make, fragments",
    [
        (
            lambda: checks.divergences(
                0.1, walls={"radius_sd": 3, "exposed": 0.4}
            ),
            ["40 per cent of steps of 3 posterior", "hard wall"],
        ),
        (
            lambda: checks.divergences(
                0.1, walls={"radius_sd": 3, "exposed": 0.01}
            ),
            ["10 per cent", "the posterior is not reliable"],
        ),
        (
            lambda: checks.error_scale({"v": 3.0}, kind="estimated"),
            ["Estimated error"],
        ),
        (lambda: checks.resolution_regime(1.0, 1e7), ["upper limit"]),
        (
            lambda: checks.resolution_regime(1.0, 1e8),
            ["limb darkening is degenerate"],
        ),
        (
            lambda: checks.resolution_regime(1.0, 3e8),
            ["limb darkening can be constrained"],
        ),
        (lambda: checks.resolution_regime(NAN, 1e8), ["not finite"]),
        (lambda: checks.multimodal(_lobes(0.0)), ["Only one"]),
        (
            lambda: checks.multimodal(_lobes(0.0, 100.0)),
            ["beats the next", "100"],
        ),
        (
            lambda: checks.multimodal(_lobes(40.0, 5.0, 0.0)),
            ["2 mas against the best 3 mas", "Δχ² = 5", "< 25"],
        ),
        (
            lambda: checks.model_gain(
                "g", NAN, 1.0, 9, n_extra=1, simple="a", complex_="b"
            ),
            ["not finite"],
        ),
        (
            lambda: checks.model_gain(
                "g", 10.0, 20.0, 9, n_extra=1, simple="one", complex_="two"
            ),
            [
                "The two model fits worse than the one model",
                "did not converge",
            ],
        ),
        (
            lambda: checks.model_gain(
                "g", 200.0, 1.0, 9, n_extra=2, simple="one", complex_="two"
            ),
            ["exceeds the BIC penalty 4.39", "the two model is preferred"],
        ),
        (
            lambda: checks.model_gain(
                "g", 5.0, 4.0, 9, n_extra=2, simple="one", complex_="two"
            ),
            ["does not exceed the BIC penalty", "the one model is adequate"],
        ),
        (
            lambda: checks.prior_constrained(
                "p", {"a": 0.1, "b": 0.9, "c": NAN}
            ),
            ["for b, c", "a 0.10, b 0.90, c nan"],
        ),
        (
            lambda: checks.prior_constrained("p", {"a": 0.1}),
            ["constrain every parameter"],
        ),
        (lambda: checks.at_bound([]), ["No fitted parameter"]),
        (lambda: checks.at_bound(["a", "b"]), ["for a, b", "Widen the prior"]),
        (
            lambda: checks.at_bound(["a", "b"], start=True),
            ["for a, b", "prior-dominated"],
        ),
        (
            lambda: _lengths(edge="low", extended="low"),
            ["lowest of the grid even after"],
        ),
        (
            lambda: _lengths(edge="low"),
            ["lowest of the grid and the grid could not", "floor"],
        ),
        (
            lambda: _lengths(edge="high", extended="high"),
            ["highest of the grid even after", "Widen ell_halffringes"],
        ),
        (
            lambda: _lengths(edge="high"),
            ["highest of the grid (best", "Widen ell_halffringes"],
        ),
        (
            lambda: _lengths(edge="high", extended="low"),
            ["highest of the grid ("],
        ),
        (
            lambda: checks.render_consistency(NAN, 1.0),
            ["could not be computed"],
        ),
        (
            lambda: checks.render_consistency(1.0, 1.3),
            ["not the model the fit found", "n_r"],
        ),
        (
            lambda: checks.render_consistency(1.0, 1.0),
            ["1 for the model and 1 for its"],
        ),
        (
            lambda: checks.structure_unconstrained(NAN, 0.1),
            ["was not assessed"],
        ),
        (
            lambda: checks.structure_unconstrained(0.5, 0.3),
            ["differ by 0.5 nats", "correlate only 0.3", "mostly the prior"],
        ),
        (
            lambda: checks.structure_unconstrained(0.5, 0.95),
            ["is stable across ℓ"],
        ),
        (
            lambda: checks.convergence(
                [_w("did not converge at 5 of 0 grid")]
            ),
            ["at some grid positions"],
        ),
        (lambda: checks.flux_axis_resolution([]), ["peak:"]),
        (lambda: checks.limits_clipped([]), ["Some limits fell outside"]),
        (
            lambda: checks.limits_clipped(
                [_w("2 limits clipped"), _w("7 limits clipped")]
            ),
            ["7 limits fell outside"],
        ),
        (
            lambda: checks.stage_warnings([_w(f"w{i}") for i in range(5)]),
            ["5 warning(s)", "w0; w1; w2 (and 2 more)"],
        ),
        (lambda: checks.stage_warnings([_w("only")]), ["only. See the"]),
        (lambda: checks.inclination(NAN, 1.0, 1), ["nan°", "unconstrained"]),
        (
            lambda: checks.inclination(50.0, NAN, 1),
            ["standard deviation is nan°"],
        ),
        (lambda: checks.absorbed_companion(25.0, NAN, 2), ["remains"]),
        (
            lambda: checks.ring_count(
                2,
                _ring_rows([120.0, 100.0], [None, None]),
                1.0,
                nats=99.0,
            ),
            ["do not clearly prefer 2 to 1 ring ", "(margin 10 nats)"],
        ),
    ],
)
def test_messages_say_which_branch_they_took(make, fragments):
    message = make().message
    for fragment in fragments:
        assert fragment in message, (fragment, message)


def test_stage_warning_checks_order_benign_filter_and_recognized_categories():
    found = checks.stage_warning_checks(
        [
            _w("odd", "UserWarning"),
            _w("old", "PyparsingDeprecationWarning"),
            _w("fd", "ResourceWarning"),
            _w("import", "ImportWarning"),
            # recognized by keyword even when the category is a benign one
            _w("the optimizer did not converge", "DeprecationWarning"),
            _w("limits were clipped"),
            _w("uses flux_bounds somewhere"),
            _w("could not resolve the likelihood peak"),
        ]
    )
    assert [c.name for c in found] == [
        "convergence",
        "flux_axis_resolution",
        "limits_clipped",
        "stage_warnings",
    ]
    assert found[-1].value == 1 and "odd" in found[-1].message
    assert "old" not in found[-1].message and "fd" not in found[-1].message


def test_ring_count_message_pluralizes_and_reports_missing_neighbours():
    single = _ring_rows([100.0], [None])
    found = checks.ring_count(1, single, 1.0)
    assert found.value is None and found.message.startswith("1 ring preferred")
    assert "at least 0 nats" in found.message
    rows = _ring_rows([100.0, 100.5, 101.0], [None] * 3)
    found = checks.ring_count(2, rows, 1.0)
    assert "2 rings" in found.message and "1 or 3 rings" in found.message
    assert "1 or 3 rings (margin -0.25, 0.25 nats)" in found.message


def test_check_status_is_one_of_the_three_known_values():
    assert checks.STATUSES == ("pass", "warn", "fail")
    assert checks.bic_penalty(3, 100) == pytest.approx(3 * 4.605170185988092)
    assert checks.bic_penalty(2, 0) == pytest.approx(2 * 0.6931471805599453)


@pytest.mark.parametrize("order", [("a", "b"), ("b", "a")])
def test_a_nan_site_makes_the_worst_chain_health_nan_in_any_order(order):
    """A stuck parameter (NaN R-hat) must not hide behind a finite site,
    whichever the site order, and the checks then fail."""
    import numpy as onp

    from virgil.pipeline._common import _chain_health, _sampler_report

    site = {
        "a": {"r_hat": 1.0, "ess_bulk": 900.0},
        "b": {"r_hat": float("nan"), "ess_bulk": float("nan")},
    }
    stats = {"diverging": onp.zeros((2, 5), bool)}
    health = _chain_health({k: site[k] for k in order}, stats)
    assert onp.isnan(health["r_hat_max"]) and onp.isnan(health["ess_bulk_min"])
    ok = {"r_hat_max": 1.0, "ess_bulk_min": 500.0, "divergence_fraction": 0.0}
    settings = {"num_chains": 2, "num_samples": 5, "num_warmup": 1}
    for parts in ([health, ok], [ok, health]):
        report = _sampler_report(settings, parts)
        assert onp.isnan(report["r_hat_max"])
        assert onp.isnan(report["ess_bulk_min"])
        assert checks.r_hat(report["r_hat_max"]).status == "fail"
        assert checks.ess(report["ess_bulk_min"]).status == "fail"


def test_a_stuck_element_of_a_vector_site_makes_its_diagnostics_nan():
    import numpy as onp

    from virgil.pipeline._common import _diagnostics

    rng = onp.random.default_rng(0)
    moving = rng.normal(size=(2, 200, 3))
    assert onp.isfinite(_diagnostics([moving])["r_hat"])
    stuck = moving.copy()
    stuck[:, :, 1] = 0.5  # one coefficient never moves
    out = _diagnostics([stuck])
    assert onp.isnan(out["r_hat"]) and onp.isnan(out["ess_bulk"])
    assert checks.r_hat(out["r_hat"]).status == "fail"
    assert checks.ess(out["ess_bulk"]).status == "fail"


# --- values that are not finite are never silent --------------------------

_NAN = float("nan")


@pytest.mark.parametrize(
    "make",
    [
        lambda: checks.error_scale({"vis_scale": _NAN}),
        lambda: checks.error_scale({"vis_scale": _NAN, "phi_scale": 1.0}),
        lambda: checks.error_scale({"phi_scale": 1.0, "vis_scale": _NAN}),
        lambda: checks.divergences(_NAN),
        lambda: checks.r_hat(_NAN),
        lambda: checks.ess(_NAN),
    ],
)
def test_a_nan_in_the_sampler_or_scale_checks_fails_in_any_position(make):
    found = make()
    assert _status(found) == "fail"
    assert "not finite" in found.message
    assert found.to_dict()["value"] is None


@pytest.mark.parametrize(
    "make, name",
    [
        (lambda: checks.prior_bound({"a": _NAN}), "prior_bound"),
        (lambda: checks.prior_bound({"a": 0.0, "b": _NAN}), "prior_bound"),
        (lambda: checks.prior_bound({"b": _NAN, "a": 0.0}), "prior_bound"),
        (lambda: checks.field_of_view(_NAN, 1.0, 10.0), "field_of_view"),
        (lambda: checks.field_of_view(5.0, _NAN, 10.0), "field_of_view"),
        (lambda: checks.field_of_view(5.0, 1.0, _NAN), "field_of_view"),
        (lambda: checks.mode_converged(_NAN), "mode_converged"),
    ],
)
def test_a_nan_in_the_remaining_checks_warns_rather_than_passes(make, name):
    found = make()
    assert found.name == name
    assert _status(found) == "warn"


def test_prior_bound_names_the_parameter_whose_fraction_is_nan():
    found = checks.prior_bound({"a": 0.0, "b": _NAN})
    assert "b" in found.message and "a," not in found.message


def test_inclination_that_is_none_warns_and_says_it_is_unknown():
    found = checks.inclination(None, None, 1)
    assert _status(found) == "warn"
    assert "inclination is unknown" in found.message
    assert _status(checks.inclination(_NAN, None, None)) == "warn"
    assert _status(checks.inclination(50.0, None, None)) == "pass"


_ROWS = [
    {"n": 1, "bic": 100.0, "log_evidence": -50.0, "widths": [5.0]},
    {"n": 2, "bic": 80.0, "log_evidence": -40.0, "widths": [5.0, 6.0]},
    {"n": 3, "bic": 70.0, "log_evidence": -36.0, "widths": [5.0, 6.0, 7.0]},
]


def test_ring_count_with_a_bic_that_is_nan_is_not_a_clear_preference():
    rows = [dict(r) for r in _ROWS]
    rows[2]["bic"] = _NAN
    found = checks.ring_count(2, rows, 1.0)
    assert _status(found) == "warn"
    assert "do not clearly prefer 2 to 3" in found.message
    assert found.to_dict()["value"] is None
    assert _status(checks.ring_count(2, _ROWS, 1.0)) == "warn"  # 2 vs 3 close


def test_ring_count_without_the_chosen_count_among_the_rows_warns():
    found = checks.ring_count(4, _ROWS, 1.0)
    assert _status(found) == "warn"
    assert "not among the fitted ring counts" in found.message


def test_ring_count_tolerates_a_row_without_widths():
    rows = [{k: v for k, v in r.items() if k != "widths"} for r in _ROWS]
    found = checks.ring_count(1, rows[:2], 1.0)
    assert _status(found) == "warn"
    assert "do not clearly prefer 1 to 2" in found.message
    assert "narrower than the half-fringe" not in found.message


def test_at_bound_with_start_is_the_start_at_bound_check():
    found = checks.at_bound(["flux", "sep"], start=True)
    assert (found.name, found.status) == ("start_at_bound", "warn")
    assert found.value == ["flux", "sep"]
    assert "NUTS starts just inside the bound" in found.message
    assert not hasattr(checks, "start_at_bound")


def test_r_hat_cannot_pass_without_enough_effective_samples():
    assert checks.r_hat(0.99, ess=500.0).status == "pass"
    assert checks.r_hat(0.99, ess=150.0).status == "warn"
    assert checks.r_hat(1.2, ess=20.0).status == "fail"
