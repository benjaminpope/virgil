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
import re

from .._utils import mas2rad

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


def format_sigma(value):
    """``value`` as ``"3.21σ"``, or ``"n/a"`` if it is missing or not finite."""
    return f"{float(value):.2f}σ" if _finite(value) else "n/a"


def chi2_reduced(name, chi2_red, *, label, warn=2.0, fail=5.0):
    """χ²/N on the quoted errors, even when error scales are fitted.

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


def error_scale(scales, *, warn=2.0, fail=5.0, kind="fitted"):
    """Error-scale factors ``s``: fitted (``error_scale="fit"``) or estimated
    once by MacKay's rule (``"rescale"``, ``kind="estimated"``).

    A factor much greater than 1 means the quoted errors are too small or
    the model is wrong; rescaling then makes χ² ≈ 1 by construction, so the
    raw χ²/N on quoted errors is the number to quote.
    """
    values = [float(s) for s in scales.values()]
    worst = max(values, default=1.0) if all(map(_finite, values)) else math.nan
    names = ", ".join(f"{k} = {float(v):.3g}" for k, v in scales.items())
    if not _finite(worst):
        status = "fail"
        meaning = "an error scale is not finite"
    elif worst > fail:
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
        f"{kind.capitalize()} error scales {names}: {meaning}.",
    )


def detection(local_sigma, global_sigma, *, threshold=3.0):
    """Significance of the best companion against a threshold.

    ``global_sigma`` corrects ``local_sigma`` for the look-elsewhere effect
    of searching a grid. A clear detection passes; one significant only
    locally warns; no detection warns too, since the fit and posterior
    stages then describe the highest noise peak.

    ``global_sigma`` is an approximate Šidák estimate, not a simulated
    false-alarm probability: for a calibrated threshold use
    [`injection_recovery`][virgil.detection.injection_recovery] or
    [`gaussian_null`][virgil.detection.gaussian_null].
    """
    value = [local_sigma, global_sigma]
    if _finite(global_sigma) and global_sigma >= threshold:
        return Check(
            "detection",
            "pass",
            value,
            threshold,
            f"Companion detected at {format_sigma(global_sigma)} after an "
            f"approximate look-elsewhere correction "
            f"({format_sigma(local_sigma)} local); for a "
            "calibrated threshold use injection_recovery or gaussian_null.",
        )
    if _finite(local_sigma) and local_sigma >= threshold:
        return Check(
            "detection",
            "warn",
            value,
            threshold,
            f"Marginal: {format_sigma(local_sigma)} local but "
            f"{format_sigma(global_sigma)} after an approximate "
            "look-elsewhere correction; confirm it "
            "with simulated nulls (injection_recovery or gaussian_null).",
        )
    return Check(
        "detection",
        "warn",
        value,
        threshold,
        f"No companion above {threshold:g}σ "
        f"({format_sigma(local_sigma)} local): "
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
    piled = [k for k, f in fractions.items() if not _finite(f) or f > warn]
    worst = max(fractions.values(), default=0.0)
    if not all(map(_finite, fractions.values())):
        worst = math.nan
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


def divergences(fraction, *, walls=None, warn=0.0, fail=0.01, exposed=0.05):
    """Fraction of divergent NUTS transitions.

    ``walls`` is the posterior's wall probe (``{"radius_sd", "n",
    "exposed"}``, see ``StarPipeline``). A failure is never downgraded. If at
    least ``exposed`` of the steps of a few posterior standard deviations hit a
    hard wall of the likelihood (a closure-phase flip, which costs thousands
    in χ²), the message names NUTS rejecting trajectories at the wall as a
    probable cause; the check still fails. The threshold only guards the
    diagnosis: with about 40 probe steps, 0.05 is two hits, not a stray one.
    """
    if not _finite(fraction):
        return Check(
            "divergences",
            "fail",
            fraction,
            fail,
            "The fraction of divergent transitions is not finite.",
        )
    if fraction > fail:
        status, meaning = "fail", "the posterior is not reliable"
    elif fraction > warn:
        status, meaning = "warn", "inspect where they occur"
    else:
        status, meaning = "pass", "none"
    if status == "fail" and walls and walls.get("exposed", 0.0) >= exposed:
        meaning += (
            f"; {100 * walls['exposed']:.2g} per cent of steps of "
            f"{walls['radius_sd']:g} posterior standard deviations hit a "
            "hard wall of the likelihood (a closure-phase sign flip), where "
            "NUTS rejects trajectories: a probable cause, not an excuse "
            "(a bispectrum likelihood would remove the walls, virgil#578)"
        )
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
        Resolution limit (mas), half of the finest λ/B of the samples.
    fov : float
        Field-of-view radius (mas), the coarsest λ/B of the samples.
    """
    if not all(map(_finite, (sep, resolution, fov))):
        return Check(
            "field_of_view",
            "warn",
            sep,
            [resolution, fov],
            "The separation, resolution limit or field of view is not "
            "finite, so the companion's position could not be checked.",
        )
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


# θB/λ of the first zero of a uniform disk's visibility, j_{1,1} / π.
FIRST_NULL = 1.2196699


def resolution_regime(
    diam_mas, freq_max, *, unresolved=0.15, first_null=FIRST_NULL
):
    """Whether the baselines reach the first null of the visibility.

    A uniform disk of angular diameter θ has its first null at
    θB/λ = 1.22 (more for a limb-darkened star), so the
    fraction ``x = θ f_max / first_null`` of the way there, with ``f_max`` the
    longest spatial frequency B/λ, says what the data can measure.

    Parameters
    ----------
    diam_mas : float
        Fitted angular diameter (mas).
    freq_max : float
        Longest spatial frequency B/λ in the data (cycles per radian).
    unresolved : float, optional
        Fraction of the first null below which the star is unresolved and
        its diameter is an upper limit.
    first_null : float, optional
        θB/λ of the first null of the fitted model's visibility; the
        uniform-disk value by default.
    """
    ratio = float(diam_mas) * float(mas2rad) * float(freq_max)
    ratio /= float(first_null)
    if not _finite(ratio):
        return Check(
            "resolution", "fail", ratio, 1.0, "The resolution is not finite."
        )
    if ratio < unresolved:
        status, meaning = (
            "fail",
            "the star is unresolved and its diameter is an upper limit",
        )
    elif ratio < 1.0:
        status, meaning = (
            "warn",
            "the baselines stop short of the first null, so the diameter "
            "is measured but limb darkening is degenerate with it",
        )
    else:
        status, meaning = (
            "pass",
            "the baselines reach the first null, so the diameter is "
            "well measured and limb darkening can be constrained",
        )
    return Check(
        "resolution",
        status,
        ratio,
        1.0,
        f"The longest baseline reaches {ratio:.3g} of the first null: "
        f"{meaning}.",
    )


# Δ(−2 log L) within which another diameter lobe counts as a rival.
RIVAL_DELTA_CHI2 = 25.0


def bic_penalty(n_extra, n):
    """The BIC penalty ``n_extra ln N`` of ``n_extra`` extra parameters fitted
    to ``n`` independent observables."""
    return n_extra * math.log(max(int(n), 2))


def multimodal(lobes, *, threshold=RIVAL_DELTA_CHI2):
    """Whether another lobe of the diameter fits nearly as well as the best.

    Parameters
    ----------
    lobes : list[dict]
        The per-lobe table of the preferred model: each row has
        ``diam_mas``, ``diam_bounds_mas`` and ``delta_chi2``, the lobe's
        −2 log L above the best lobe's (zero for the best lobe; on the
        quoted errors, its Δχ²).
    threshold : float, optional
        The Δ(−2 log L) of the next-best lobe above which the best lobe is
        unambiguous.
    """
    rows = sorted(lobes, key=lambda r: r["delta_chi2"])
    if len(rows) < 2:
        return Check(
            "multimodal",
            "pass",
            None,
            threshold,
            "Only one diameter lobe was found.",
        )
    best, other = rows[0], rows[1]
    gap = float(other["delta_chi2"])
    if gap > threshold:
        return Check(
            "multimodal",
            "pass",
            gap,
            threshold,
            f"The best lobe ({best['diam_mas']:.4g} mas) beats the next "
            f"({other['diam_mas']:.4g} mas) by Δχ² = {gap:.3g}.",
        )
    return Check(
        "multimodal",
        "warn",
        gap,
        threshold,
        f"Another lobe fits nearly as well: {other['diam_mas']:.4g} mas "
        f"against the best {best['diam_mas']:.4g} mas, Δχ² = {gap:.3g} < "
        f"{threshold:g}. The posterior is bounded to the best lobe; the "
        "alias is not excluded by the data.",
    )


def model_gain(
    name, m2loglike_simple, m2loglike_complex, n, *, n_extra, simple, complex_
):
    """Whether a richer model improves the fit by more than its BIC penalty.

    The gain is the drop in −2 log L from the simpler to the richer model
    (their Δχ² on quoted errors).

    Parameters
    ----------
    name : str
        Check name.
    m2loglike_simple, m2loglike_complex : float
        −2 log L of the simpler and richer model, each at its own fitted
        error scales if any.
    n : int
        Number of independent observables.
    n_extra : int
        Extra parameters of the richer model.
    simple, complex_ : str
        Their names, for the message.
    """
    gain = float(m2loglike_simple) - float(m2loglike_complex)
    penalty = bic_penalty(n_extra, n)
    if not _finite(gain):
        return Check(
            name, "fail", gain, penalty, "The −2 log L gain is not finite."
        )
    # The richer model contains the simpler: a fit did not converge.
    if gain < -1.0:
        status = "warn"
        message = (
            f"The {complex_} model fits worse than the {simple} model it "
            f"contains (Δ(−2 log L) = {gain:.3g}): a fit did not converge."
        )
    elif gain > penalty:
        status = "pass"
        message = (
            f"Δ(−2 log L) = {gain:.3g} for {n_extra} extra parameters "
            f"exceeds the BIC penalty {penalty:.3g}: the {complex_} model "
            "is preferred."
        )
    else:
        status = "pass"
        message = (
            f"Δ(−2 log L) = {gain:.3g} for {n_extra} extra parameters does "
            f"not exceed the BIC penalty {penalty:.3g}: the {simple} model "
            "is adequate."
        )
    return Check(name, status, gain, penalty, message)


def prior_constrained(name, ratios, *, warn=0.8):
    """Whether parameters are constrained by the data or by their priors.

    Parameters
    ----------
    name : str
        Check name.
    ratios : dict[str, float]
        For each parameter, the posterior standard deviation divided by the
        standard deviation of its prior.
    warn : float, optional
        Ratio at or above which the prior dominates.
    """
    bad = [k for k, v in ratios.items() if not _finite(v) or v >= warn]
    worst = max(ratios.values(), default=0.0)
    listing = ", ".join(f"{k} {v:.2f}" for k, v in ratios.items())
    if bad:
        return Check(
            name,
            "warn",
            worst,
            warn,
            f"The posterior is as wide as the prior for {', '.join(bad)} "
            f"(σ_posterior/σ_prior: {listing}): the data do not constrain "
            "them.",
        )
    return Check(
        name,
        "pass",
        worst,
        warn,
        f"The data constrain every parameter (σ_posterior/σ_prior: "
        f"{listing}).",
    )


# --- checks of DiskPipeline ---------------------------------------------------


def _upper_first(text):
    """``text`` with its first letter capitalized (``str.capitalize`` would
    lowercase the rest)."""
    return text[:1].upper() + text[1:]


def length_scale(
    evidences, ells, halffringe, *, nats=3.0, edge=None, extended=None
):
    """Whether the data measure the Gaussian-process length scale ℓ.

    Laplace evidences that differ by little across the ℓ grid mean ℓ is a
    prior choice, not a result.

    Parameters
    ----------
    evidences : sequence of float
        Laplace log evidence at each ℓ.
    ells : sequence of float
        The Matérn ℓ (mas) of each, in the same order.
    halffringe : float
        λ/2B of the longest baseline (mas).
    nats : float, optional
        Spread of the evidences (max − min) below which ℓ is not measured.
    edge : str or None, optional
        ``"low"`` or ``"high"`` when the best ℓ is still the smallest or
        largest of the grid after its one extension (the grid chose its edge).
    extended : str or None, optional
        The side the grid was extended on (``None`` if it was not).
    """
    z = [float(v) for v in evidences]
    if len(z) < 2 or not all(_finite(v) for v in z):
        return Check(
            "length_scale",
            "warn",
            None,
            nats,
            "The evidence of the length scale could not be computed at "
            "every ℓ of the grid.",
        )
    spread = max(z) - min(z)
    best = max(range(len(z)), key=z.__getitem__)
    # A Matérn length is not a feature width: quote the squared-exponential
    # equivalent ℓ/√3 of the ν = 3/2 length (it matches the curvature at zero lag).
    equivalent = float(ells[best]) / math.sqrt(3.0)
    where = (
        f"best ℓ = {float(ells[best]):.3g} mas, a squared-exponential "
        f"length of {equivalent:.3g} mas, {equivalent / halffringe:.2g}× "
        "the half-fringe"
    )
    if edge is not None:
        return Check(
            "length_scale",
            "warn",
            spread,
            nats,
            f"The best ℓ is the {edge}est of the grid"
            + (
                " even after the grid was extended once in that direction"
                if extended == edge
                else " and the grid could not be extended that way"
                if edge == "low"
                else ""
            )
            + f" ({where}): the grid chose its own edge, so ℓ is not "
            "measured. "
            + (
                "The floor ell_floor_halffringes stops a smaller ℓ (below "
                "a pixel a small ℓ gives beads with χ²/N far under 1): "
                "treat the data as not resolving the structure."
                if edge == "low"
                else "Widen ell_halffringes upward."
            ),
        )
    if spread < nats:
        return Check(
            "length_scale",
            "warn",
            spread,
            nats,
            f"The evidences across ℓ differ by only {spread:.2g} nats "
            f"({where}): the data do not measure ℓ, so it is a prior "
            "choice, and the profile's fine structure is the prior wherever "
            "the data are silent.",
        )
    return Check(
        "length_scale",
        "pass",
        spread,
        nats,
        f"The evidence prefers one length scale by {spread:.2g} nats "
        f"({where}).",
    )


def ring_count(n_best, rows, halffringe, *, nats=3.0, split=()):
    """Whether the data clearly prefer the chosen number of rings.

    Parameters
    ----------
    n_best : int
        The chosen number of rings.
    rows : sequence of dict
        One per fitted ring count: ``n``, ``bic``, ``log_evidence`` (None if
        not available) and ``widths`` (the rings' Gaussian σ, mas).
    halffringe : float
        λ/2B of the longest baseline (mas).
    nats : float, optional
        A neighbouring count is clearly worse when it loses by at least
        this many nats of log evidence and 2 × this many of BIC.
    split : sequence of int, optional
        Ring counts left out of ``rows`` because two of their rings are too
        close to tell apart, although their BIC is lower than that of
        ``n_best``.

    Notes
    -----
    A ring narrower than the half-fringe leaves almost no signature in the
    visibilities, so its width is not measured.
    """
    by_n = {r["n"]: r for r in rows}
    if n_best not in by_n:
        return Check(
            "ring_count",
            "warn",
            None,
            nats,
            f"The fit with {n_best} ring{'s' if n_best != 1 else ''} is not "
            "among the fitted ring counts, so the choice could not be "
            "checked.",
        )
    best = by_n[n_best]
    margins = {}
    for n in (n_best - 1, n_best + 1):
        if n not in by_n:
            continue
        other = by_n[n]
        gaps = [0.5 * (other["bic"] - best["bic"])]
        if _finite(best.get("log_evidence")) and _finite(
            other.get("log_evidence")
        ):
            gaps.append(best["log_evidence"] - other["log_evidence"])
        # A margin that is not finite (a BIC that failed) is not a clear one.
        margins[n] = min(gaps) if all(map(_finite, gaps)) else math.nan
    narrow = [
        i + 1
        for i, w in enumerate(best.get("widths", ()))
        if not _finite(w) or w < halffringe
    ]
    worst = min(margins.values(), default=None)
    if not all(map(_finite, margins.values())):
        worst = None
    problems = []
    unclear = [n for n, m in margins.items() if not _finite(m) or m < nats]
    if unclear:
        problems.append(
            "the data do not clearly prefer "
            f"{n_best} to {' or '.join(str(n) for n in unclear)} "
            f"ring{'s' if unclear[-1] != 1 else ''} "
            f"(margin {', '.join(f'{margins[n]:.2g}' for n in unclear)} "
            "nats)"
        )
    if split:
        problems.append(
            f"{' and '.join(str(n) for n in split)} rings fit better by BIC "
            "but with two rings too close to tell apart: a Gaussian ring "
            "is too simple a profile here"
        )
    if narrow:
        problems.append(
            f"ring {', '.join(str(i) for i in narrow)} is narrower than "
            "the half-fringe, so its width is not resolved"
        )
    if problems:
        return Check(
            "ring_count",
            "warn",
            worst,
            nats,
            f"With {n_best} ring{'s' if n_best != 1 else ''}: "
            + "; ".join(problems)
            + ".",
        )
    return Check(
        "ring_count",
        "pass",
        worst,
        nats,
        f"{n_best} ring{'s' if n_best != 1 else ''} preferred over its "
        f"neighbours by at least {worst if worst is not None else 0:.2g} "
        "nats, and every ring is resolved in width.",
    )


def inner_edge(
    fraction, correlation, halffringe, *, max_fraction=0.1, max_correlation=0.9
):
    """Light that the profile puts in an unresolved core at its inner edge.

    Such light trades against the star's flux, which the data cannot tell
    apart. ``fraction`` is the fraction of the disk's light within one
    half-fringe of the inner edge; ``correlation`` is the Laplace
    correlation between the disk's flux and the light in the innermost
    nodes.
    """
    if not _finite(fraction):
        return Check(
            "inner_edge",
            "warn",
            fraction,
            max_fraction,
            "The light near the inner edge could not be computed.",
        )
    corr = float(correlation) if _finite(correlation) else None
    reasons = []
    if fraction > max_fraction:
        reasons.append(
            f"{100 * fraction:.2g} per cent of the disk's light lies within "
            f"one half-fringe ({halffringe:.3g} mas) of the inner edge"
        )
    if corr is not None and abs(corr) > max_correlation:
        reasons.append(
            f"the light there is correlated with the disk flux "
            f"(|ρ| = {abs(corr):.2f})"
        )
    shown = "n/a" if corr is None else f"{corr:.2f}"
    if reasons:
        return Check(
            "inner_edge",
            "warn",
            fraction,
            [max_fraction, max_correlation],
            _upper_first("; ".join(reasons))
            + ": an unresolved core trades against the star's flux; "
            "refit with a larger inner radius to test it.",
        )
    return Check(
        "inner_edge",
        "pass",
        fraction,
        [max_fraction, max_correlation],
        f"{100 * fraction:.2g} per cent of the light is within one "
        f"half-fringe of the inner edge (correlation with the flux {shown}).",
    )


def outer_edge(fraction, halffringe, *, warn=0.01):
    """Light within one half-fringe of the outer edge of the model.

    A proxy for light beyond the edge that the model cannot place; the
    remedy is a larger outer radius.
    """
    if not _finite(fraction):
        return Check(
            "outer_edge",
            "warn",
            fraction,
            warn,
            "The light near the outer edge could not be computed.",
        )
    if fraction > warn:
        return Check(
            "outer_edge",
            "warn",
            fraction,
            warn,
            f"{100 * fraction:.2g} per cent of the model's light lies "
            f"within one half-fringe ({halffringe:.3g} mas) of its outer "
            "edge: light beyond it cannot be placed. Refit with a larger "
            "r_out_mas.",
        )
    return Check(
        "outer_edge",
        "pass",
        fraction,
        warn,
        f"{100 * fraction:.2g} per cent of the light is within one "
        "half-fringe of the outer edge.",
    )


def inclination(inc, pa_sd, n_minima, *, minimum=15.0, max_pa_sd=45.0):
    """Whether inclination and position angle are well determined.

    Near face-on the position angle is undefined; it is then not quoted
    with an uncertainty. A geometry grid with another comparable minimum
    means the geometry is multimodal.

    Parameters
    ----------
    inc : float
        Inclination (degrees).
    pa_sd : float or None
        Laplace standard deviation of the position angle (degrees).
    n_minima : int
        Comparable minima of the geometry grid, the best included.
    minimum : float, optional
        Inclination (degrees) below which the position angle is
        unconstrained.
    max_pa_sd : float, optional
        Standard deviation (degrees) of the position angle above which it
        is unconstrained.
    """
    reasons = []
    if not _finite(inc) or inc < minimum:
        shown = "unknown" if inc is None else f"{float(inc):.3g}°"
        reasons.append(
            f"the inclination is {shown}, below {minimum:g}°, so the "
            "position angle is unconstrained and is reported without an "
            "uncertainty"
        )
    elif pa_sd is not None and (not _finite(pa_sd) or pa_sd > max_pa_sd):
        reasons.append(
            "the position angle's Laplace standard deviation is "
            f"{float(pa_sd):.3g}°, so it is unconstrained"
        )
    if n_minima is not None and n_minima > 1:
        reasons.append(
            f"the geometry grid has {n_minima} comparable minima, so the "
            "inclination and position angle are multimodal"
        )
    if reasons:
        return Check(
            "inclination",
            "warn",
            inc,
            minimum,
            _upper_first("; ".join(reasons)) + ".",
        )
    return Check(
        "inclination",
        "pass",
        inc,
        minimum,
        f"Inclination {float(inc):.3g}° with a single minimum of the "
        "geometry grid.",
    )


def absorbed_companion(
    delta_chi2, delta_chi2_raised, degree, *, threshold=25.0
):
    """A compact companion that the disk may have absorbed.

    ``delta_chi2`` is the improvement in χ² from adding a point source to
    the best model, at the best position of a grid search on the residuals.
    A significant point source remaining means the model does not hold it.
    If a refit at harmonic ``degree`` + 1 removes it (``delta_chi2_raised``)
    the source is ambiguous: it may be disk asymmetry that the current degree
    cannot represent, or a disk of the higher degree may be flexible enough
    to absorb a real compact source. Either is ``None`` when it was not
    computed.
    """
    if delta_chi2 is None or not _finite(delta_chi2):
        return Check(
            "absorbed_companion",
            "pass",
            None,
            threshold,
            "The residuals were not searched for a point source.",
        )
    if delta_chi2 < threshold:
        return Check(
            "absorbed_companion",
            "pass",
            delta_chi2,
            threshold,
            f"No significant point source remains in the residuals "
            f"(Δχ² = {delta_chi2:.3g}).",
        )
    removed = (
        delta_chi2_raised is not None
        and _finite(delta_chi2_raised)
        and delta_chi2_raised < threshold
    )
    detail = (
        f" Raising the harmonic degree from {degree} to {degree + 1} removes "
        f"it (Δχ² = {delta_chi2_raised:.3g}): it may be asymmetry of the "
        f"disk rather than a companion, or a disk of degree {degree + 1} can "
        "absorb a compact source."
        if removed
        else ""
    )
    return Check(
        "absorbed_companion",
        "warn",
        delta_chi2,
        threshold,
        f"A point source remains in the residuals (Δχ² = {delta_chi2:.3g})"
        f".{detail}",
    )


def worst_status(checks):
    """The worst status among ``checks`` (``"pass"`` if there are none)."""
    rank = {status: i for i, status in enumerate(STATUSES)}
    return max((c.status for c in checks), key=rank.get, default="pass")


# --- checks from the warnings raised inside stages ----------------------------

# Warnings of these categories are recorded but raise no check: they say
# something about the software environment, not about the result.
# The match is by suffix: libraries subclass these, such as pyparsing's
# ``PyparsingDeprecationWarning``.
_BENIGN_CATEGORIES = (
    "DeprecationWarning",
    "ImportWarning",
    "ResourceWarning",
)


def _numbers(pattern, text):
    match = re.search(pattern, text)
    return [float(g) for g in match.groups()] if match else None


def convergence(records):
    """Optimizer non-convergence reported by the grid functions.

    Parameters
    ----------
    records : list of dict
        The recorded warnings that mention non-convergence ("the optimizer
        did not converge at N of M grid positions"). The one with the worst
        fraction N/M is reported; grids of different sizes can appear.
    """
    worst, total = 0.0, 0.0
    for record in records:
        counts = _numbers(r"(\d+)\s+of\s+(\d+)", record["message"])
        if (
            counts
            and counts[1]
            and (not total or counts[0] / counts[1] > worst / total)
        ):
            worst, total = counts
    if not total:
        return Check(
            "convergence",
            "warn",
            None,
            None,
            "The optimizer did not converge at some grid positions; values "
            "there may be inaccurate.",
        )
    return Check(
        "convergence",
        "warn",
        worst / total,
        0.0,
        f"The optimizer did not converge at {worst:.0f} of {total:.0f} grid "
        f"positions ({100 * worst / total:.2g} per cent): values there may "
        "be inaccurate.",
    )


def flux_axis_resolution(records, *, minimum=2.0):
    """The flux axis does not resolve the likelihood peak."""
    steps = []
    for r in records:
        found = _numbers(r"([\d.]+(?:e[-+]?\d+)?)\s+steps", r["message"])
        if found:
            steps.append(found[0])
    value = min(steps) if steps else None
    detail = f" ({value:.2g} steps across its FWHM)" if steps else ""
    return Check(
        "flux_axis_resolution",
        "warn",
        value,
        minimum,
        f"The flux axis does not resolve the likelihood peak{detail}: "
        "log_bayes_factor is inaccurate; use a finer flux axis (n_flux).",
    )


def limits_clipped(records):
    """Contrast limits clipped to the flux bounds."""
    counts = []
    for r in records:
        found = _numbers(r"(\d+)\s+limits", r["message"])
        if found:
            counts.append(found[0])
    value = max(counts) if counts else None
    which = f"{value:.0f} limits" if counts else "Some limits"
    return Check(
        "limits_clipped",
        "warn",
        value,
        0,
        f"{which} fell outside the flux bounds and were clipped to the "
        "nearer bound: those limits are not the true limits; widen "
        "flux_range.",
    )


def stage_warnings(records):
    """Warnings no other rule recognizes, summarized in one check."""
    shown = "; ".join(r["message"] for r in records[:3])
    more = f" (and {len(records) - 3} more)" if len(records) > 3 else ""
    return Check(
        "stage_warnings",
        "warn",
        len(records),
        0,
        f"{len(records)} warning(s) were raised inside the stages: "
        f"{shown}{more}. See the 'warnings' list of summary.json.",
    )


_RULES = {
    "convergence": convergence,
    "flux_axis_resolution": flux_axis_resolution,
    "limits_clipped": limits_clipped,
}


def stage_warning_checks(warning_records):
    """Checks for the warnings recorded inside the stages.

    Known messages map to specific checks (``convergence``,
    ``flux_axis_resolution``, ``limits_clipped``), matched by keywords so
    that rewording a message does not break the mapping; any other
    warning, except deprecation notices, goes to one ``stage_warnings``
    check. No warnings give no checks.

    Parameters
    ----------
    warning_records : list of dict
        Each with ``stage``, ``category`` and ``message``.
    """
    grouped, other = {}, []
    for record in warning_records:
        text = record["message"].lower()
        if "converge" in text and "optimizer" in text:
            rule = "convergence"
        elif "resolve" in text and "peak" in text:
            rule = "flux_axis_resolution"
        elif "clipped" in text or "flux_bounds" in text:
            rule = "limits_clipped"
        else:
            rule = None
        if rule:
            grouped.setdefault(rule, []).append(record)
        elif not record["category"].endswith(_BENIGN_CATEGORIES):
            other.append(record)
    checks = [
        fn(grouped[name]) for name, fn in _RULES.items() if name in grouped
    ]
    if other:
        checks.append(stage_warnings(other))
    return checks


def at_bound(names, *, start=False):
    """Fitted parameters within 0.1% of a prior bound.

    A prior capped too tight pins a parameter silently: the fit is then the
    prior's edge, not a measurement. Passes with none.

    Parameters
    ----------
    names : sequence of str
        The fitted parameters at a bound (``fit``'s ``at_bound``).
    start : bool, optional
        Report the bound as the start of NUTS (the ``start_at_bound`` check,
        added by the posterior stage only when there are some): the
        posterior then starts just inside the bound and is prior-dominated.
    """
    names = list(names)
    listing = ", ".join(names)
    if start:
        return Check(
            "start_at_bound",
            "warn",
            names,
            None,
            f"The best fit sits at a prior bound for {listing}, so the data "
            "do not constrain it there (a non-detection?): NUTS starts just "
            "inside the bound and that posterior is prior-dominated.",
        )
    if not names:
        return Check(
            "at_bound",
            "pass",
            [],
            None,
            "No fitted parameter sits at a prior bound.",
        )
    return Check(
        "at_bound",
        "warn",
        names,
        None,
        f"The best fit sits at a prior bound for {listing}: the data do not "
        "constrain it there, or the prior is too tight. Widen the prior "
        "before reading the value.",
    )


def render_consistency(chi2_model, chi2_render, *, tolerance=0.1, floor=0.02):
    """Whether the rendered image fits the data as well as the model does.

    A rendered image needs more quadrature nodes than the fit, so it can
    fit worse than the model it was drawn from.

    Parameters
    ----------
    chi2_model, chi2_render : float
        χ²/N of the model and of its rendered image (same data and scene).
    tolerance : float, optional
        Allowed relative difference (default 0.1).
    floor : float, optional
        Allowed absolute difference, in χ²/N (default 0.02).
    """
    if not (_finite(chi2_model) and _finite(chi2_render)):
        return Check(
            "render_consistency",
            "warn",
            None,
            tolerance,
            "The rendered image's χ² could not be computed.",
        )
    gap = abs(chi2_render - chi2_model)
    # Allowed gap is relative plus a floor, so a near-zero chi-squared passes.
    ok = gap <= tolerance * chi2_model + floor
    return Check(
        "render_consistency",
        "pass" if ok else "warn",
        float(chi2_render / chi2_model) if chi2_model > 0 else None,
        tolerance,
        (
            f"On the reference channel with the disk's colour removed, on "
            f"the rescaled errors if any (not the summary's χ²/N): χ²/N is "
            f"{chi2_model:.3g} for the model and {chi2_render:.3g} for its "
            "rendered image"
            + (
                "."
                if ok
                else ": the image is not the model the fit found. Refit "
                "with a larger n_r (suggest_n_r sizes the nodes for the "
                "visibilities, not for pixels)."
            )
        ),
    )


def structure_unconstrained(
    spread, correlation, *, nats=3.0, minimum=0.8, halffringe=None
):
    """Whether the data constrain the structure at all.

    Flat evidences across the length-scale grid say only that ℓ is not
    measured (see ``length_scale``); flat evidences *and* unstable radial
    profiles say the image is the prior's.

    Parameters
    ----------
    spread : float
        Spread (max − min) of the Laplace evidences across ℓ (nats).
    correlation : float or None
        Smallest Pearson correlation between the radial profiles of the
        fits at the grid's lengths.
    nats : float, optional
        Spread below which the evidences are flat.
    minimum : float, optional
        Correlation below which the profiles are unstable.
    """
    if not (_finite(spread) and _finite(correlation)):
        return Check(
            "structure_unconstrained",
            "pass",
            None,
            minimum,
            "Stability of the structure across ℓ was not assessed.",
        )
    flat = spread < nats
    unstable = correlation < minimum
    if flat and unstable:
        return Check(
            "structure_unconstrained",
            "warn",
            float(correlation),
            minimum,
            f"The evidences across ℓ differ by {spread:.2g} nats and the "
            f"radial profiles correlate only {correlation:.2g} between "
            "them: the data do not constrain the structure, so the image "
            "is mostly the prior. Do not read its features; compare "
            "another strategy before trusting any.",
        )
    return Check(
        "structure_unconstrained",
        "pass",
        float(correlation),
        minimum,
        f"The radial profile is stable across ℓ (smallest correlation "
        f"{correlation:.2g}) or the evidence separates the lengths.",
    )


# === ORBIT CHECKS ===


def night_flags(flags, n_nights):
    """Nights that carry a flag.

    Parameters
    ----------
    flags : dict[str, list[str]]
        ``{night: flag codes}`` of the flagged nights. A blocking flag keeps
        the night from seeding an orbit; a non-blocking one (such as
        ``sep-below-resolution``) is reported only. Either way the night's
        data still count in every likelihood.
    n_nights : int
        The number of nights.
    """
    if not flags:
        return Check(
            "night_flags",
            "pass",
            0,
            None,
            f"None of the {n_nights} nights is flagged.",
        )
    listing = "; ".join(f"{k}: {', '.join(v)}" for k, v in flags.items())
    return Check(
        "night_flags",
        "warn",
        len(flags),
        None,
        f"{len(flags)} of {n_nights} nights are flagged ({listing}): a "
        "blocking flag keeps a night from seeding an orbit, a non-blocking "
        "one is reported only, and the data still count in the likelihood.",
    )


def labels_free(free):
    """Whether the data leave the stars' labels undetermined.

    Parameters
    ----------
    free : bool
        The best mode has a point-reflected partner of comparable
        evidence (``AliasResult.labels_free``).
    """
    if free:
        return Check(
            "labels_free",
            "warn",
            True,
            None,
            "The data do not say which star is which: the reflected orbit "
            "(ω + 180°, Ω unchanged) fits as well. Compare orbits in the "
            "sky plane and accept the reflection.",
        )
    return Check(
        "labels_free",
        "pass",
        False,
        None,
        "The labels of the stars are determined by the data.",
    )


def alias_ambiguity(delta, runner_up, *, warn=10.0, fail=1.0):
    """The winning alias band's evidence over the runner-up's.

    Parameters
    ----------
    delta : float or None
        log Z of the best band minus that of the next (nats), on the
        tempered likelihood; None with a single band.
    runner_up : int or None
        The runner-up's orbit count, for the message.
    warn, fail : float, optional
        A margin below ``warn`` nats warns (the runner-up is not
        excluded); below ``fail`` fails (the period is not determined).
    """
    if delta is None or not _finite(delta):
        return Check(
            "alias_ambiguity",
            "pass",
            None,
            warn,
            "There is a single alias band: no rival period.",
        )
    if delta < fail:
        status, meaning = (
            "fail",
            "the data do not choose between the periods",
        )
    elif delta < warn:
        status, meaning = (
            "warn",
            "the runner-up alias is not excluded; confirm it with more "
            "epochs or another instrument",
        )
    else:
        status, meaning = "pass", "the winning period is decisive"
    return Check(
        "alias_ambiguity",
        status,
        float(delta),
        warn,
        f"The best alias band beats the runner-up (N = {runner_up}) by "
        f"{delta:.3g} nats: {meaning}.",
    )


def bands_failed(failed):
    """Alias bands the search could not evaluate.

    Parameters
    ----------
    failed : sequence of int
        The orbit counts N of the bands whose fit raised (their evidence is
        -inf, so they never win nor appear as the runner-up).
    """
    if not len(failed):
        return Check(
            "bands_failed",
            "pass",
            0,
            0,
            "Every alias band was evaluated.",
        )
    shown = ", ".join(str(n) for n in list(failed)[:10])
    more = "" if len(failed) <= 10 else f" and {len(failed) - 10} more"
    return Check(
        "bands_failed",
        "warn",
        len(failed),
        0,
        f"The fit of {len(failed)} alias band(s) failed (N = {shown}{more}): "
        "the winner's margin does not cover them. Re-run, or rerun those "
        "bands with `fit_orbit_aliases`.",
    )


_WINNER_FLAGS = (
    "not-converged",
    "laplace-is-differ",
    "low-ess",
    "near-prior-bound",
    "no-valid-mode",
    "starts-failed",
)


def winner_flags(flags):
    """The evidence engine's own flags on the winning band.

    Parameters
    ----------
    flags : sequence of str
        The winner's flags; a poor fit has its own check (`chi2`).
    """
    bad = [f for f in flags if f in _WINNER_FLAGS]
    if not bad:
        return Check(
            "winner_flags",
            "pass",
            [],
            None,
            "The winning band's evidence has no flags.",
        )
    return Check(
        "winner_flags",
        "warn",
        bad,
        None,
        f"The winning band's evidence carries {', '.join(bad)}: its log Z, "
        "and so the margin over the runner-up, is less reliable.",
    )


def mode_converged(gap, *, warn=1.0):
    """Whether the search's best mode was an optimum.

    Parameters
    ----------
    gap : float
        How far (nats) the best sampled point lies above the mode the
        search found; a draw cannot beat a true maximum, so a gap means
        the search stopped short.
    warn : float, optional
        Gap above which to warn.
    """
    if not _finite(gap):
        return Check(
            "mode_converged",
            "warn",
            None,
            warn,
            "The gap between the sampler's best point and the search's best "
            "mode is not finite: whether the mode is an optimum is unknown.",
        )
    if gap > warn:
        return Check(
            "mode_converged",
            "warn",
            float(gap),
            warn,
            f"The sampler found a point {gap:.3g} nats above the search's "
            "best mode: the search stopped short of the optimum, so the "
            "band evidences (and the ranking) may be off. Raise "
            "`n_candidates`, `n_refine` or `max_steps`.",
        )
    return Check(
        "mode_converged",
        "pass",
        float(gap),
        warn,
        "The search's best mode is an optimum of the sampled density.",
    )


def tempering(out_of_range, f, f_raw):
    """Whether the correlated-channel tempering is outside what the
    likelihood can trust.

    Parameters
    ----------
    out_of_range : bool
        ``AliasResult.tempering``: a factor above 2, a raw factor below
        0.5, or a flat residual correlation.
    f, f_raw : float or None
        The pooled effective-channel factor and its unfloored value.
    """
    if f is not None and out_of_range:
        return Check(
            "tempering",
            "warn",
            float(f),
            [0.5, 2.0],
            f"The effective-channel factor f = {f:.3g} (raw "
            f"{f_raw:.3g}) is outside [0.5, 2] or the residual "
            "correlation is not AR(1): the evidences are less reliable.",
        )
    return Check(
        "tempering",
        "pass",
        None if f is None else float(f),
        [0.5, 2.0],
        "The likelihood is not tempered."
        if f is None
        else f"The effective-channel factor f = {f:.3g} is within [0.5, 2].",
    )


def primary_resolved(resolved, diam, at_edge):
    """Whether the primary's fitted diameter is trustworthy.

    Parameters
    ----------
    resolved : bool or None
        The data resolve the primary (a uniform disk was fitted); None if
        the diameter was fixed or not estimated.
    diam : float or None
        The estimated diameter (mas).
    at_edge : bool
        The estimate sits at the edge of the diameter grid.
    """
    if resolved and at_edge:
        return Check(
            "primary_resolved",
            "warn",
            diam,
            None,
            f"The primary is resolved but its estimated diameter "
            f"({diam:.3g} mas) is the largest tried: the disk may be "
            "larger, or the V² are biased.",
        )
    return Check(
        "primary_resolved",
        "pass",
        diam,
        None,
        "The primary's size is not at the edge of the diameters tried.",
    )


def no_orbit(delta, *, threshold=10.0):
    """The orbit's evidence against a single star.

    Parameters
    ----------
    delta : float or None
        log Z of the best orbit minus the log likelihood of the star
        alone (nats); the orbit's prior volume is in its evidence.
    threshold : float, optional
        Below this the data show no orbit: warn; below 0, the single star
        is preferred: fail.
    """
    if delta is None or not _finite(delta):
        return Check(
            "no_orbit",
            "fail",
            None,
            threshold,
            "The evidence of the orbit against a single star could not be "
            "computed.",
        )
    if delta < 0:
        status, meaning = (
            "fail",
            "a single star fits as well as any orbit: no orbit is detected",
        )
    elif delta < threshold:
        status, meaning = (
            "warn",
            "the orbit is not clearly better than a single star",
        )
    else:
        status, meaning = "pass", "the orbit is clearly preferred"
    return Check(
        "no_orbit",
        status,
        float(delta),
        threshold,
        f"The best orbit beats a single star by {delta:.3g} nats: {meaning}.",
    )


def night_dropped(report):
    """Whether the verdict survives dropping the most outlying night.

    Parameters
    ----------
    report : dict
        ``modes["robustness"]``: ``{"skipped": reason}``, or the night, its
        raw χ²/N, and the ``full`` and ``dropped`` verdicts (``verdict``,
        the winning band's ``n``; the refit's ``same_band``).
    """
    if "skipped" in report:
        return Check(
            "night_dropped",
            "pass",
            None,
            None,
            f"Skipped: {report['skipped']}.",
        )
    full, dropped = report["full"], report["dropped"]
    same = full["verdict"] == dropped["verdict"] and (
        full["verdict"] != "orbit" or dropped["same_band"]
    )
    where = f"night {report['night']} (raw χ²/N = {report['chi2_per_n']:.3g})"
    detail = (
        f"{full['verdict']} (N = {full['n']}) with it, "
        f"{dropped['verdict']} (N = {dropped['n']}) without it"
    )
    if same:
        message = f"Dropping {where} leaves the verdict: {detail}."
    else:
        message = (
            f"Dropping {where} changes the verdict: {detail}. One night "
            "decides the result; check it before trusting the orbit."
        )
    return Check(
        "night_dropped",
        "pass" if same else "warn",
        report["chi2_per_n"],
        None,
        message,
    )


def period_bound(bound, lower, p_range):
    """Whether the period is at an end of its window.

    Parameters
    ----------
    bound : bool
        The best period is at the top of the window, or flat towards it (a
        short arc): the period is unconstrained.
    lower : bool
        The best period is at the bottom, where the sampling limit may
        have cut the truth.
    p_range : sequence of float
        The window (days), for the message.
    """
    lo, hi = float(p_range[0]), float(p_range[1])
    if bound:
        return Check(
            "period_bound",
            "warn",
            True,
            [lo, hi],
            f"The period is unconstrained: the best value is at the top "
            f"of the window ({lo:.3g} to {hi:.3g} d), as for an arc "
            "shorter than the orbit. Observe a longer arc.",
        )
    if lower:
        return Check(
            "period_bound",
            "warn",
            True,
            [lo, hi],
            f"The best period is at the bottom of the window "
            f"({lo:.3g} d), which the sampling of the nights sets: a "
            "shorter period may have been cut off.",
        )
    return Check(
        "period_bound",
        "pass",
        False,
        [lo, hi],
        f"The period is inside its window ({lo:.3g} to {hi:.3g} d).",
    )


# --- checks of ImagingPipeline (every χ²/N is on the quoted errors) ---------


def _two_way(name, bad, value, threshold, good_text, bad_text):
    status = "warn" if bad else "pass"
    return Check(
        name, status, value, threshold, bad_text if bad else good_text
    )


def quoted_errors(blocks):
    """Quoted errors that look like placeholders: all equal, or not positive
    and finite (``blocks``: ``{"all_equal", "n_bad"}`` by kind)."""
    equal = sorted(k for k, b in blocks.items() if b["all_equal"])
    bad = sorted(k for k, b in blocks.items() if b["n_bad"])
    parts = [f"all equal in {', '.join(equal)}"] * bool(equal)
    parts += [f"zero or not finite in {', '.join(bad)}"] * bool(bad)
    return _two_way(
        "quoted_errors",
        parts,
        equal + bad,
        None,
        "The quoted errors vary and are positive.",
        f"The quoted errors are {' and '.join(parts)}: they look like "
        "placeholders, so the hyperparameters and every χ²/N are suspect.",
    )


def effective_parameters(n, n_gamma, *, warn=0.2):
    """Warn when ``N - γ`` (MacKay's γ) is under ``warn`` of ``N``."""
    fraction = (float(n) - float(n_gamma)) / float(n)
    text = (
        f"The data measure γ = {n_gamma:.3g} parameters for N = {n:.3g} data"
    )
    return _two_way(
        "effective_parameters",
        fraction < warn,
        fraction,
        warn,
        text + ".",
        text + ": compare with a parametric model.",
    )


def short_baselines(mean, error, *, warn=3.0, fail=5.0):
    """Mean visibility residual on the shortest baselines, in errors."""
    if mean is None or not _finite(mean):
        return Check(
            "short_baselines", "pass", None, warn, "No baseline-ordered data."
        )
    sigma = float(mean) / float(error)
    status = (
        ("fail" if abs(sigma) > fail else "warn")
        if abs(sigma) > warn
        else "pass"
    )
    note = " The image misses flux here, or the calibration is biased."
    return Check(
        "short_baselines",
        status,
        sigma,
        warn,
        f"Mean residual on the shortest baselines {format_sigma(sigma)}."
        + (note if status != "pass" else ""),
    )


def star_decision(delta, *, threshold=5.0, decision=None, fixed=None):
    """The star decision from ΔlogZ; ``delta`` is None if set by hand."""
    if delta is None:
        return Check(
            "star",
            "pass",
            None,
            threshold,
            f"Set by hand (star={bool(fixed)}).",
        )
    caveat = " ΔlogZ lacks the star flux's Occam factor: not a significance."
    return _two_way(
        "star",
        abs(delta) < threshold,
        float(delta),
        threshold,
        f"ΔlogZ = {delta:+.3g} nats decides the star {decision}." + caveat,
        f"ΔlogZ = {delta:+.3g} nats is inside ±{threshold:g}: undecided "
        "(the larger evidence was kept)." + caveat,
    )


def orientation(flip_dchi2, closure, *, warn=9.0):
    """The Δχ² of turning the image by 180° (zero without closure)."""
    weak = not closure or not _finite(flip_dchi2) or flip_dchi2 < warn
    return _two_way(
        "orientation",
        weak,
        flip_dchi2 if closure else 0.0,
        warn,
        f"Turning the image by 180° raises χ² by {flip_dchi2:.3g}.",
        "The orientation is weakly constrained (no closure phases, or a "
        "180° turn changes χ² by under the threshold).",
    )


def edges(grid_edges, flux, *, warn=0.01):
    """The fit pressed against an edge: the evidence optimum on the grid
    (``"length min"``...), or more than ``warn`` of the image's flux in its
    outer two pixels."""
    parts = [
        f"on the hyperparameter grid edge ({', '.join(grid_edges)})"
    ] * bool(grid_edges)
    parts += [f"{100 * float(flux):.3g}% of the flux in the outer pixels"] * (
        not _finite(flux) or flux > warn
    )
    return _two_way(
        "edges",
        parts,
        flux,
        warn,
        "The evidence optimum is inside the grid and the flux inside the "
        "field.",
        f"The fit is {' and '.join(parts)}: the grid or the field is too "
        "small.",
    )


def ensemble_kept(n_kept, n, *, min_kept=3):
    """Warn when fewer than ``min_kept``, or under half, of the ``n`` members
    survive the χ² filter (the spread is then not a useful map)."""
    return _two_way(
        "ensemble_kept",
        n_kept < min_kept or 2 * n_kept < n,
        n_kept,
        min_kept,
        f"{n_kept} of {n} members fit V² and closure phases and are in the "
        "mean.",
        f"Only {n_kept} of {n} members are in the mean (fewer than "
        f"{min_kept}, or under half): the sigma map is not a useful map.",
    )


def ensemble_spread(spread, n_kept, *, warn=0.5, shared=True):
    """The members' summed pixel standard deviation over their summed mean
    (0 for identical images); ``warn`` is a convention, not calibrated on
    truth."""
    note = (
        " The spread is over starts, grids and fields at shared "
        "hyperparameters: it excludes hyperparameter uncertainty and any "
        "error every member shares."
        if shared
        else " It is over starts, grids, fields and hyperparameters."
    )
    return _two_way(
        "ensemble_spread",
        n_kept < 2 or not _finite(spread) or spread > warn,
        spread,
        warn,
        f"The kept members differ by {100 * spread:.3g}% of the flux." + note,
        "Fewer than two members are kept, or they differ by more than "
        f"{100 * warn:g}% of the flux: the mean is not stable." + note,
    )
