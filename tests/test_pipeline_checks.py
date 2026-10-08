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
