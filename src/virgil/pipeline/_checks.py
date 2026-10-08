"""Deterministic quality checks for pipeline results.

Every check is a pure function of plain numbers that returns a
[`Check`][virgil.pipeline.Check]: the same inputs always give the same
status and message. Thresholds are keyword arguments with the defaults
used by the pipelines, so the tests can reach every branch.

Status meanings:

* ``"pass"``: nothing to report.
* ``"warn"``: the result is usable but needs a human look.
* ``"fail"``: the result should not be used as it stands.
"""

from __future__ import annotations

import dataclasses
import math

STATUSES = ("pass", "warn", "fail")


@dataclasses.dataclass(frozen=True)
class Check:
    """One quality check of a pipeline run.

    Attributes
    ----------
    name : str
        Stable identifier, e.g. ``"chi2_companion"``.
    status : {"pass", "warn", "fail"}
        Outcome.
    value : float, list or None
        The quantity tested.
    threshold : float, list or None
        The threshold it was compared with.
    message : str
        One templated sentence saying what the outcome means.
    """

    name: str
    status: str
    value: object
    threshold: object
    message: str

    def __post_init__(self):
        if self.status not in STATUSES:
            raise ValueError(
                f"status must be one of {STATUSES}; got {self.status!r}."
            )

    def to_dict(self):
        """The check as a JSON-ready dict."""
        return {
            "name": self.name,
            "status": self.status,
            "value": _jsonable(self.value),
            "threshold": _jsonable(self.threshold),
            "message": self.message,
        }

    @classmethod
    def from_dict(cls, record):
        """Rebuild a check from :meth:`to_dict`."""
        return cls(**{f.name: record[f.name] for f in dataclasses.fields(cls)})


def _jsonable(value):
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    value = float(value)
    return value if math.isfinite(value) else None


def _finite(value):
    return value is not None and math.isfinite(float(value))


def chi2_reduced(name, chi2_red, *, label, warn=2.0, fail=5.0):
    """χ²/N on the quoted errors.

    Parameters
    ----------
    name : str
        Check name.
    chi2_red : float
        χ² per independent observable, on the errors as quoted.
    label : str
        What was fitted, e.g. ``"the best-fit binary"``.
    warn, fail : float, optional
        χ²/N above ``warn`` warns; above ``fail`` fails.
    """
    if not _finite(chi2_red):
        return Check(
            name, "fail", chi2_red, warn, f"χ²/N of {label} is not finite."
        )
    if chi2_red > fail:
        status, meaning = (
            "fail",
            "the model does not describe the data, or the errors are "
            "much too small",
        )
    elif chi2_red > warn:
        status, meaning = "warn", "errors likely underestimated"
    elif chi2_red < 1.0 / warn:
        status, meaning = "warn", "errors likely overestimated"
    else:
        status, meaning = "pass", "consistent with the quoted errors"
    return Check(
        name,
        status,
        chi2_red,
        warn,
        f"χ²/N = {chi2_red:.3g} for {label} on quoted errors: {meaning}.",
    )


def error_scale(scales, *, warn=2.0, fail=5.0):
    """Fitted error-scale factors ``s`` (only with ``error_scale="fit"``).

    A factor much greater than 1 means the quoted errors are too small or
    the model is wrong; rescaling then makes χ² ≈ 1 by construction, so the
    raw χ²/N on quoted errors is the number to quote.
    """
    worst = max((float(s) for s in scales.values()), default=1.0)
    names = ", ".join(f"{k} = {float(v):.3g}" for k, v in scales.items())
    if worst > fail:
        status = "fail"
        meaning = "the fit failed to describe the data on its quoted errors"
    elif worst > warn:
        status = "warn"
        meaning = "the quoted errors are too small or the model is wrong"
    else:
        status, meaning = "pass", "close to the quoted errors"
    return Check(
        "error_scale",
        status,
        worst,
        warn,
        f"Fitted error scales {names}: {meaning}.",
    )


def detection(local_sigma, global_sigma, *, threshold=3.0):
    """Significance of the best companion against a threshold.

    ``global_sigma`` corrects ``local_sigma`` for the look-elsewhere effect
    of searching a grid. A clear detection passes; one significant only
    locally warns; no detection warns too, since the fit and posterior
    stages then describe the highest noise peak.
    """
    value = [local_sigma, global_sigma]
    if _finite(global_sigma) and global_sigma >= threshold:
        return Check(
            "detection",
            "pass",
            value,
            threshold,
            f"Companion detected at {global_sigma:.2f}σ after the "
            f"look-elsewhere correction ({local_sigma:.2f}σ local).",
        )
    if _finite(local_sigma) and local_sigma >= threshold:
        return Check(
            "detection",
            "warn",
            value,
            threshold,
            f"Marginal: {local_sigma:.2f}σ local but {global_sigma:.2f}σ "
            "after the look-elsewhere correction; confirm it with "
            "simulated nulls.",
        )
    return Check(
        "detection",
        "warn",
        value,
        threshold,
        f"No companion above {threshold:g}σ ({local_sigma:.2f}σ local): "
        "the fit and posterior describe the highest noise peak; quote the "
        "contrast limits instead.",
    )


def grid_edge(index, shape, names):
    """Whether the grid peak lies on an edge of the searched grid.

    Parameters
    ----------
    index : sequence of int
        Index of the peak along each axis.
    shape : sequence of int
        Length of each axis.
    names : sequence of str
        Axis names, for the message.
    """
    edges = [
        name
        for name, i, n in zip(names, index, shape)
        if n > 1 and int(i) in (0, int(n) - 1)
    ]
    if edges:
        return Check(
            "grid_edge",
            "warn",
            list(int(i) for i in index),
            None,
            f"The grid peak lies on the edge of the {', '.join(edges)} "
            "axis: the best companion may lie outside the searched grid.",
        )
    return Check(
        "grid_edge",
        "pass",
        list(int(i) for i in index),
        None,
        "The grid peak lies inside the searched grid.",
    )


def prior_bound(fractions, *, warn=0.05):
    """Posterior mass piled up at a prior bound.

    Parameters
    ----------
    fractions : dict[str, float]
        For each bounded parameter, the fraction of posterior samples
        within 1% of the prior range of either bound.
    warn : float, optional
        Fraction above which to warn.
    """
    worst = max(fractions.values(), default=0.0)
    piled = [k for k, f in fractions.items() if f > warn]
    if piled:
        return Check(
            "prior_bound",
            "warn",
            worst,
            warn,
            f"Posterior mass piles up at a prior bound for "
            f"{', '.join(piled)}: widen the prior or check the model.",
        )
    return Check(
        "prior_bound",
        "pass",
        worst,
        warn,
        "No posterior mass piles up at a prior bound.",
    )


def r_hat(value, *, warn=1.01, fail=1.05):
    """Largest split R-hat over the sampled parameters."""
    if not _finite(value):
        return Check("r_hat", "fail", value, warn, "R-hat is not finite.")
    if value > fail:
        status, meaning = "fail", "the chains have not converged"
    elif value > warn:
        status, meaning = "warn", "run longer chains to be sure"
    else:
        status, meaning = "pass", "the chains agree"
    return Check(
        "r_hat", status, value, warn, f"Largest R-hat {value:.4f}: {meaning}."
    )


def ess(value, *, warn=400.0, fail=100.0):
    """Smallest bulk effective sample size over the sampled parameters."""
    if not _finite(value):
        return Check("ess", "fail", value, warn, "ESS is not finite.")
    if value < fail:
        status, meaning = "fail", "too few independent samples"
    elif value < warn:
        status, meaning = "warn", "quantiles in the tails are noisy"
    else:
        status, meaning = "pass", "enough independent samples"
    return Check(
        "ess",
        status,
        value,
        warn,
        f"Smallest bulk ESS {value:.0f}: {meaning}.",
    )


def divergences(fraction, *, warn=0.0, fail=0.01):
    """Fraction of divergent NUTS transitions."""
    if fraction > fail:
        status, meaning = "fail", "the posterior is not reliable"
    elif fraction > warn:
        status, meaning = "warn", "inspect where they occur"
    else:
        status, meaning = "pass", "none"
    return Check(
        "divergences",
        status,
        fraction,
        fail,
        f"Divergent transitions: {100 * fraction:.2g} per cent: {meaning}.",
    )


def residual_normality(skew, excess_kurtosis, n, *, n_se=3.0):
    """Skew and excess kurtosis of the whitened residuals.

    Each is compared with ``n_se`` of its standard error under a normal
    distribution, √(6/n) and √(24/n).
    """
    if n < 8:
        return Check(
            "residual_normality",
            "pass",
            [skew, excess_kurtosis],
            None,
            "Too few residuals to test their normality.",
        )
    limits = [n_se * math.sqrt(6.0 / n), n_se * math.sqrt(24.0 / n)]
    bad = [
        name
        for name, value, limit in zip(
            ("skew", "kurtosis"), (skew, excess_kurtosis), limits
        )
        if not _finite(value) or abs(value) > limit
    ]
    if bad:
        return Check(
            "residual_normality",
            "warn",
            [skew, excess_kurtosis],
            limits,
            f"Whitened residuals are not normal (skew {skew:.2f}, excess "
            f"kurtosis {excess_kurtosis:.2f}): outliers or a missing model "
            "component.",
        )
    return Check(
        "residual_normality",
        "pass",
        [skew, excess_kurtosis],
        limits,
        f"Whitened residuals look normal (skew {skew:.2f}, excess kurtosis "
        f"{excess_kurtosis:.2f}).",
    )


def field_of_view(sep, resolution, fov):
    """Whether a companion lies inside the field and outside the core.

    Parameters
    ----------
    sep : float
        Separation (mas).
    resolution : float
        Resolution limit (mas), about half of λ/B_max.
    fov : float
        Field-of-view radius (mas), about λ/B_min.
    """
    if sep < resolution:
        return Check(
            "field_of_view",
            "warn",
            sep,
            [resolution, fov],
            f"Separation {sep:.3g} mas is inside the resolution limit "
            f"({resolution:.3g} mas): separation and flux are degenerate.",
        )
    if sep > fov:
        return Check(
            "field_of_view",
            "warn",
            sep,
            [resolution, fov],
            f"Separation {sep:.3g} mas is outside the field of view "
            f"({fov:.3g} mas): the position may be an alias.",
        )
    return Check(
        "field_of_view",
        "pass",
        sep,
        [resolution, fov],
        f"Separation {sep:.3g} mas lies between the resolution limit and "
        "the field of view.",
    )


def worst_status(checks):
    """The worst status among ``checks`` (``"pass"`` if there are none)."""
    rank = {status: i for i, status in enumerate(STATUSES)}
    return max((c.status for c in checks), key=rank.get, default="pass")
