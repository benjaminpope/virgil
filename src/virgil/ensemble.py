"""Ensembles of randomized image reconstructions, averaged into one image.

A single regularized reconstruction depends on choices the data do not fix:
the regularizer and its weight, the pixel size, the field and the starting
image. Following Drevon et al. (2026, https://arxiv.org/abs/2609.15365),
this module runs many reconstructions with these choices drawn at random,
keeps those that fit the data, and averages them while the average still
fits. The mean is less sensitive to any one choice than a single
reconstruction, and the spread of the members maps how much the image depends
on those choices. It is written from the paper's description, not from their
code; see [Imaging, part 7](../imaging_ensemble.md).

The work is split so that a cluster can run it in parallel:

1. [`draw_groups`][virgil.ensemble.draw_groups] draws the settings of each
   *group*: one geometry, regularizer family and starting image, and several
   weights.
2. [`run_group`][virgil.ensemble.run_group] fits one group as an
   [`l_curve`][virgil.imaging.l_curve]; groups that share a geometry and a
   family share a compilation. A group of family ``"gp"`` is instead an
   [`evidence_grid`][virgil.ensemble.evidence_grid] over a Gaussian-process
   prior ([`gp_members`][virgil.ensemble.gp_members]).
3. [`combine`][virgil.ensemble.combine] selects the members and averages
   them into an [`Ensemble`][virgil.ensemble.Ensemble].

[`ensemble`][virgil.ensemble.ensemble] runs all three. The ensemble's
standard deviation is not a posterior uncertainty: it measures how much the
image changes between reasonable reconstruction choices, not the noise in the
data.
"""

import dataclasses
import warnings

import jax
import numpy as onp
import numpyro.distributions as dist

from .fields import GaussianField
from .fitting import FitResult, fit
from .imaging import (
    TSV,
    TV,
    Centroid,
    LCurve,
    MaxEntropy,
    StarletL1,
    _as_list,
    _chi2,
    beam,
    envelope_fwhm,
    field_of_view,
    image_field,
    image_priors,
    l_curve,
    log_evidence,
    nyquist_pixel_scale,
    starting_image,
)
from ._precision import cast_tree, run_in
from .metrics import align, resample
from .models import EllipticalGaussian, Image, PointSource, System
from ._deprecate import external_stacklevel

__provisional__ = (
    "evidence_grid",
    "gp_members",
    "GPFit",
    "EvidenceGrid",
)

# The regularizer families, by the names used in EnsembleSpec.
FAMILIES = {"tv": TV, "tsv": TSV, "maxent": MaxEntropy, "starlet": StarletL1}


@dataclasses.dataclass(frozen=True)
class EnsembleSpec:
    """What an ensemble draws at random, and how it selects members.

    See [Imaging, part 7](../imaging_ensemble.md#what-is-drawn).

    Attributes
    ----------
    families : tuple of str
        Families drawn uniformly: ``"tv"``, ``"tsv"``, ``"maxent"``,
        ``"starlet"`` or ``"gp"`` ([`gp_members`][virgil.ensemble.gp_members]).
    weight_ranges : dict
        Per family, the ``(low, high)`` weight per data point; ``n_weights``
        (at least 3) log-uniform weights are drawn, one per bin of ``log w``.
    oversample : tuple of float
        Pixels per Nyquist pixel.
    field_factors : tuple of float
        The field is ``field_mas`` times one of these, at most
        ``field_of_view(data)`` (and 1 on a uv lattice).
    field_mas, aperture_mas : float or None
        Base field (default [`image_field`][virgil.imaging.image_field]) and
        a field-of-view ceiling.
    starts : tuple of str
        Starts drawn uniformly: ``"moments"``, ``"dirty"`` (needs phases), ``"flat"``.
    max_npix : int
        Most pixels on a side; larger fields get coarser pixels, then a crop.
    window_dex : float
        A group keeps weights from ``corner / 10**window_dex`` up to the corner.
    max_chi2_red, chi2_ratio, mad_cut : float
        Drop members whose raw χ² per point on any dataset exceeds
        ``max_chi2_red`` or ``chi2_ratio`` times the best's, then those more
        than ``mad_cut`` robust deviations above the median total.
    max_shift_mas : float or None
        Largest recentring shift without a star (default: beam major axis).
    mean_rtol : float or None
        Tolerance on the mean's χ² excess over the best member's (default
        ``sqrt(2/N)``; 0 often keeps one member).
    min_kept : int
        [`combine`][virgil.ensemble.combine] warns below this many members.
    gp_sigmas, gp_lengths, gp_sigma_limits, gp_length_limits, gp_growths
        GP evidence grid ([`evidence_grid`][virgil.ensemble.evidence_grid]):
        starting amplitudes and lengths (beam minor axes), limits, growth steps.
    gp_window_nats : float
        GP members are fits within this log evidence of the best.
    gp_axis_ratio : float or None
        Add anisotropic fields if a Gaussian fits below this axis ratio.
    """

    families: tuple = ("tv", "tsv", "maxent", "starlet")
    # Weight per data point; each range holds the L-curve corners of the
    # contest bench with a decade to spare below them. See
    # docs/imaging_ensemble.md ("What is drawn").
    weight_ranges: dict = dataclasses.field(
        default_factory=lambda: {
            "tv": (1e-4, 1e1),
            "tsv": (1e-1, 1e4),
            "maxent": (1e-4, 1e1),
            "starlet": (1e-4, 1e1),
        }
    )
    n_weights: int = 8
    oversample: tuple = (2.0, 3.0, 4.0)
    field_factors: tuple = (0.75, 1.0, 1.5)
    starts: tuple = ("moments", "flat")
    max_npix: int = 128
    window_dex: float = 1.0
    max_chi2_red: float = onp.inf
    chi2_ratio: float = 2.0
    mad_cut: float = 5.0
    max_shift_mas: float | None = None
    mean_rtol: float | None = None
    min_kept: int = 3
    field_mas: float | None = None
    aperture_mas: float | None = None
    gp_sigmas: tuple = (1.0, 2.0, 4.0)
    gp_lengths: tuple = (0.5, 1.0, 2.0)
    gp_sigma_limits: tuple = (0.25, 16.0)
    gp_length_limits: tuple = (0.125, 8.0)
    gp_growths: int = 3
    gp_window_nats: float = 2.0
    gp_axis_ratio: float | None = 0.9

    def __post_init__(self):
        unknown = set(self.families) - set(FAMILIES) - {"gp"}
        if unknown:
            raise ValueError(
                f"Unknown regularizer families {sorted(unknown)}; choose "
                f"from {sorted(FAMILIES)}, or 'gp'."
            )
        missing = set(self.families) - set(self.weight_ranges) - {"gp"}
        if missing:
            raise ValueError(f"No weight range for {sorted(missing)}.")
        if self.n_weights < 3:
            raise ValueError("n_weights must be at least 3.")
        bad = set(self.starts) - {"moments", "dirty", "flat"}
        if bad:
            raise ValueError(f"Unknown starts {sorted(bad)}.")


@dataclasses.dataclass(frozen=True)
class Draw:
    """The settings of one group of reconstructions.

    Attributes
    ----------
    index : int
        Position of the group in the ensemble.
    family : str
        The regularizer family, or ``"gp"``.
    npix : int
        Pixels on a side.
    pixel_scale_mas : float
        Pixel size in mas.
    start : str
        The starting image.
    weights : tuple of float
        The regularizer weights, largest first (empty for ``"gp"``).
    """

    index: int
    family: str
    npix: int
    pixel_scale_mas: float
    start: str
    weights: tuple

    @property
    def geometry(self):
        """``(family, npix, pixel_scale_mas)``: groups sharing it share a
        compilation."""
        return (self.family, self.npix, self.pixel_scale_mas)


@dataclasses.dataclass(frozen=True)
class Group:
    """The result of [`run_group`][virgil.ensemble.run_group].

    Attributes
    ----------
    draw : Draw
        The group's settings.
    curve : LCurve or None
        Its fits, one per weight, largest weight first; ``None`` for a
        Gaussian-process group.
    gp : tuple of GPFit
        For a Gaussian-process group
        ([`gp_members`][virgil.ensemble.gp_members]), the fits it offers
        as members.
    grid : EvidenceGrid or None
        For a Gaussian-process group, the evidence grid they come from.
    """

    draw: Draw
    curve: LCurve | None
    gp: tuple = ()
    grid: object = None


@dataclasses.dataclass(frozen=True)
class Member:
    """One reconstruction in an ensemble.

    Attributes
    ----------
    draw : Draw
        The settings of its group.
    weight : float
        Its regularizer weight (NaN for a Gaussian-process member).
    result : FitResult
        The fit.
    chi2_red : tuple of float
        Raw χ² per data point of each dataset (with the quoted errors).
    kept : bool
        Whether it is in the mean image.
    gp : GPFit or None
        For a Gaussian-process member, its hyperparameters and evidence.
    reason : str or None
        Why it was left out: ``"window"`` (outside its group's L-curve
        window), ``"diverged"`` (non-finite χ²), ``"chi2"`` (a dataset fitted
        too badly), ``"outlier"`` (total χ² far above the others) or
        ``"mean"`` (adding it made the mean fit a dataset worse). ``None`` if
        kept.
    """

    draw: Draw
    weight: float
    result: FitResult
    chi2_red: tuple
    kept: bool = False
    reason: str | None = None
    gp: object = None

    @property
    def total_chi2_red(self):
        """χ² per data point over all datasets."""
        info = self.result.info
        return float(sum(info["chi2"]) / sum(info["ndata"]))


@dataclasses.dataclass(frozen=True)
class Ensemble:
    """The result of [`combine`][virgil.ensemble.combine].

    Attributes
    ----------
    model : System
        The mean scene, exactly: the kept members' images on their own grids
        (``member0``, ``member1``, ...), each carrying its share of the flux,
        and with a star a ``star`` component. ``chi2_red`` and ``trace`` are of
        this model.
    mean : Image
        The mean extended emission on a common grid (the finest pixels and the
        largest field of the kept members), summing to one; with a star its
        ``flux`` is the mean flux relative to the star. For display and
        scoring: resampling smooths it, so it fits the data less well than
        ``model``.
    std : array, shape (npix, npix)
        Standard deviation across the kept members of ``mean.brightness``,
        pixel by pixel.
    chi2_red : tuple of float
        Raw χ² per data point of the mean scene on each dataset.
    trace : list of tuple
        ``chi2_red`` of the running mean after each accepted member; on every
        dataset it stays within ``spec.mean_rtol`` of the first entry.
    members : list of Member
        Every reconstruction, kept or not.
    groups : list of Group
        The groups, with their L-curves.
    """

    model: object
    mean: Image
    std: object
    chi2_red: tuple
    trace: list
    members: list
    groups: list

    @property
    def kept(self):
        """The members in the mean."""
        return [m for m in self.members if m.kept]

    def summary(self):
        """A short table: the groups, what was kept, and the mean's fit."""
        lines = [
            f"{'group':>5} {'family':>8} {'npix':>4} {'scale':>7} "
            f"{'start':>8} {'corner':>9} {'kept':>6}"
        ]
        for group in self.groups:
            d = group.draw
            members = [m for m in self.members if m.draw.index == d.index]
            corner = None if group.curve is None else _corner(group.curve)
            lines.append(
                f"{d.index:>5} {d.family:>8} {d.npix:>4} "
                f"{d.pixel_scale_mas:>7.3g} {d.start:>8} "
                f"{'-' if corner is None else f'{corner:.3g}':>9} "
                f"{sum(m.kept for m in members):>2}/{len(members):<3}"
            )
        reasons = {}
        for m in self.members:
            if not m.kept:
                reasons[m.reason] = reasons.get(m.reason, 0) + 1
        dropped = ", ".join(f"{k} {v}" for k, v in sorted(reasons.items()))
        best = ", ".join(f"{c:.3g}" for c in self.trace[0])
        mean = ", ".join(f"{c:.3g}" for c in self.chi2_red)
        lines += [
            "",
            f"kept {len(self.kept)} of {len(self.members)} members"
            + (f" (dropped: {dropped})" if dropped else ""),
            f"raw chi2/N per dataset: best member {best}; mean {mean}",
        ]
        return "\n".join(lines)


def draw_groups(data, n_groups, key, spec=None, star=True):
    """Draw the settings of ``n_groups`` groups of reconstructions.

    Each group draws a regularizer family, pixel size, field, starting image
    and ``spec.n_weights`` log-uniform weights, as set by the
    [`EnsembleSpec`][virgil.ensemble.EnsembleSpec]. The draws depend only on
    ``key`` and ``spec``, so every task of a cluster array can draw them all and
    run its own.

    Parameters
    ----------
    data : OIData or sequence of OIData
        The data.
    n_groups : int
        Number of groups.
    key : jax.random.PRNGKey
        The random key.
    spec : EnsembleSpec, optional
        What to draw (default ``EnsembleSpec()``).
    star : bool, optional
        Whether the scene has an unresolved star (default), which the field's
        envelope fit then allows for
        ([`envelope_fwhm`][virgil.imaging.envelope_fwhm]).

    Returns
    -------
    list of Draw
        Sorted by geometry, so that groups sharing a compilation run one after
        the other.
    """
    spec = EnsembleSpec() if spec is None else spec
    rng = onp.random.default_rng(
        int(jax.random.randint(key, (), 0, 2**31 - 1))
    )
    nyquist = nyquist_pixel_scale(data)
    ceiling = field_of_view(data, aperture_mas=spec.aperture_mas)
    field = spec.field_mas
    if field is None:
        field = image_field(
            data,
            fwhm_mas=envelope_fwhm(data, star, spec.aperture_mas),
            aperture_mas=spec.aperture_mas,
        )
    n_data = sum(d.n_independent for d in _as_list(data))
    # On a uv lattice (AMI) a field beyond field_of_view aliases.
    lattice = any(d.uv_grid is not None for d in _as_list(data))
    draws = []
    for index in range(int(n_groups)):
        family = spec.families[rng.integers(len(spec.families))]
        scale = nyquist / spec.oversample[rng.integers(len(spec.oversample))]
        factor = spec.field_factors[rng.integers(len(spec.field_factors))]
        if lattice:
            factor = min(factor, 1.0)
        size = min(factor * field, ceiling)
        npix = int(onp.ceil(size / scale))
        if npix > spec.max_npix:
            # Keep the field and coarsen the pixels, but never beyond the
            # Nyquist scale: past that, crop the field instead.
            npix = spec.max_npix
            scale = min(size / npix, nyquist)
        start = spec.starts[rng.integers(len(spec.starts))]
        if family == "gp":
            weights = onp.empty(0)  # the evidence chooses the hyperparameters
        else:
            low, high = onp.log(spec.weight_ranges[family]) + onp.log(n_data)
            # Stratified: one log-uniform weight in each of n_weights bins.
            edges = onp.linspace(low, high, spec.n_weights + 1)
            weights = onp.exp(rng.uniform(edges[:-1], edges[1:]))
        draws.append(
            Draw(
                index,
                family,
                npix,
                float(scale),
                start,
                tuple(sorted((float(w) for w in weights), reverse=True)),
            )
        )
    draws.sort(key=lambda d: (d.geometry, d.index))
    return draws


def reference_starts(
    data, star=True, starts=("moments", "flat"), aperture_mas=None
):
    """The starting images that each group resamples onto its own grid.

    Each is made once by [`starting_image`][virgil.imaging.starting_image],
    which fits a star plus a Gaussian envelope: ``"moments"`` is that
    Gaussian, ``"dirty"`` the positive part of the dirty image, and
    ``"flat"`` a uniform image with the fitted flux.

    aperture_mas : float, optional
        The aperture, as for [`EnsembleSpec`][virgil.ensemble.EnsembleSpec].

    Returns
    -------
    dict
        Start name to an [`Image`][virgil.models.Image].
    """
    out = {}
    moments = starting_image(data, star=star, aperture_mas=aperture_mas)
    moments = moments.env if star else moments
    for name in starts:
        if name == "moments":
            out[name] = moments
        elif name == "dirty":
            dirty = starting_image(
                data, star=star, start="dirty", aperture_mas=aperture_mas
            )
            out[name] = dirty.env if star else dirty
        elif name == "flat":
            ones = onp.ones(moments.log_brightness.shape)
            out[name] = Image.from_brightness(
                ones, moments.pixel_scale_mas, flux=moments.flux
            )
        else:
            raise ValueError(f"Unknown start {name!r}.")
    return out


def run_group(data, draw, *, star=True, starts=None, spec=None, **fit_options):
    """Fit one group of an ensemble: an L-curve over its weights.

    A group of family ``"gp"`` is instead an evidence grid
    ([`gp_members`][virgil.ensemble.gp_members]).

    Parameters
    ----------
    data : OIData or sequence of OIData
        The data.
    draw : Draw
        The group, from [`draw_groups`][virgil.ensemble.draw_groups].
    star : bool, optional
        Whether the scene is an unresolved star at the origin plus the
        image (default), or the image alone, centred by a
        [`Centroid`][virgil.imaging.Centroid] prior one pixel wide.
    starts : dict, optional
        From [`reference_starts`][virgil.ensemble.reference_starts];
        computed here if not given (pass it to avoid refitting the
        envelope for every group).
    spec : EnsembleSpec, optional
        Used only by a ``"gp"`` group, for its grid.
    **fit_options
        Passed to [`l_curve`][virgil.imaging.l_curve] and so to
        [`fit`][virgil.fitting.fit], e.g. ``method`` and ``max_steps``.

    Returns
    -------
    Group
    """
    if starts is None or draw.start not in starts:
        starts = reference_starts(
            data,
            star,
            (draw.start,),
            None if spec is None else spec.aperture_mas,
        )
    if draw.family == "gp":
        return gp_members(
            data, draw, star=star, starts=starts, spec=spec, **fit_options
        )
    # The group's starting scene: the reference resampled to its grid.
    reference = starts[draw.start]
    pixels = resample(reference, None, draw.npix, draw.pixel_scale_mas)
    image = Image.from_brightness(
        onp.maximum(onp.asarray(pixels), 0.0),
        draw.pixel_scale_mas,
        floor=1e-3,
        flux=reference.flux,
    )
    model = System(star=PointSource(), env=image) if star else image
    priors = image_priors(model)
    if star:
        # The image's flux relative to the star is a scale: log-uniform.
        priors["env.flux"] = dist.LogUniform(1e-4, 1e3)
    curve = l_curve(
        model,
        priors,
        data,
        FAMILIES[draw.family](1.0, path="env" if star else None),
        draw.weights,
        () if star else (Centroid(draw.pixel_scale_mas),),
        **fit_options,
    )
    return Group(draw, curve)


@dataclasses.dataclass(frozen=True)
class GPFit:
    """One Gaussian-process fit of an evidence grid.

    !!! warning "Provisional"
        May change in any 0.4.x release without a deprecation cycle.

    Attributes
    ----------
    variant : str
        ``"iso"`` or ``"aniso"``.
    length_mas : float
        Correlation length in mas (for ``"aniso"`` the geometric mean of
        the two lengths).
    sigma : float
        Standard deviation of the log-brightness.
    log_evidence : float
        [`log_evidence`][virgil.imaging.log_evidence] of the fit.
    result : FitResult
        The fit.
    """

    variant: str
    length_mas: float
    sigma: float
    log_evidence: float
    result: FitResult


@dataclasses.dataclass(frozen=True)
class EvidenceGrid:
    """The result of [`evidence_grid`][virgil.ensemble.evidence_grid].

    !!! warning "Provisional"
        May change in any 0.4.x release without a deprecation cycle.

    Attributes
    ----------
    lengths_mas, sigmas : dict of tuple
        For each variant (``"iso"``, and ``"aniso"`` if the data are
        elongated), the grid's axes: correlation lengths in mas and
        amplitudes, ascending.
    log_evidence : dict of array
        Per variant, the log evidence, with rows over ``lengths_mas`` and
        columns over ``sigmas``.
    fits : dict
        ``(variant, length_mas, sigma)`` to the [`GPFit`][virgil.ensemble.GPFit].
    best : GPFit
        The fit of largest evidence.
    interior : bool
        Whether the best point is inside its grid's edges: on an edge the
        optimum may lie beyond it, because the growth stopped at a limit.
    axis_ratio, pa_deg : float or None
        The elliptical-Gaussian fit's minor-to-major ratio and position
        angle (``None`` if it was not made).
    """

    lengths_mas: dict
    sigmas: dict
    log_evidence: dict
    fits: dict
    best: GPFit
    interior: bool
    axis_ratio: float | None = None
    pa_deg: float | None = None


def _extend(axis, best, limits):
    """Add a point a factor of 2 beyond ``best`` if it is the end of the sorted
    ``axis`` and the new point is within ``limits``; whether it did."""
    if best == axis[0] and best / 2 >= limits[0]:
        axis.insert(0, best / 2)
    elif best == axis[-1] and best * 2 <= limits[1]:
        axis.append(best * 2)
    else:
        return False
    return True


# deliberate: test seam (tests/test_ensemble.py)
def _grow_grid(
    evaluate, lengths, sigmas, length_limits, sigma_limits, growths
):
    """Evaluate ``evaluate(length, sigma)``, a log evidence, on the grid.

    While the best point is on an edge that can move (within the limits),
    add a row or column a factor of 2 beyond it, at most ``growths``
    times. Returns the sorted axes and the values keyed by
    ``(length, sigma)``.
    """
    ls, ss = sorted(lengths), sorted(sigmas)
    values = {}

    def fill():
        for length in ls:
            for sigma in ss:
                if (length, sigma) not in values:
                    values[length, sigma] = evaluate(length, sigma)

    def value(key):
        return values[key] if onp.isfinite(values[key]) else -onp.inf

    fill()
    for _ in range(growths):
        length, sigma = max(values, key=value)
        grew_length = _extend(ls, length, length_limits)
        grew_sigma = _extend(ss, sigma, sigma_limits)
        if not (grew_length or grew_sigma):
            break
        fill()
    return ls, ss, values


# deliberate: test seam (tests/test_ensemble.py)
def _elongation(data, star, field_mas):
    """Axis ratio and position angle (deg) of an elliptical-Gaussian fit.

    The fit (with an analytic star if ``star``) starts from four position
    angles and keeps the best.
    """
    best = None
    for pa0 in (0.0, 45.0, 90.0, 135.0):
        env = EllipticalGaussian(0.3 * field_mas, 0.7, pa0, flux=0.1)
        priors = {
            "env.fwhm": dist.LogUniform(0.01, field_mas),
            "env.ratio": dist.LogUniform(0.05, 1.0),
            "env.pa": dist.Uniform(pa0 - 90.0, pa0 + 90.0),
        }
        if star:
            model = System(star=PointSource(), env=env)
            priors["env.flux"] = dist.LogUniform(1e-4, 1e3)
        else:
            model = System(env=EllipticalGaussian(0.3 * field_mas, 0.7, pa0))
        result = fit(model, priors, data)
        chi2 = float(sum(result.info["chi2"]))
        if best is None or chi2 < best[0]:
            best = (
                chi2,
                float(result.values["env.ratio"]),
                float(result.values["env.pa"]) % 180.0,
            )
    return best[1], best[2]


# deliberate: test seam (tests/test_ensemble.py)
def _oriented(template, pa_deg, pixel_scale_mas):
    """``template`` on a grid whose "up" axis points at ``pa_deg``.

    The rotation's sign is chosen by comparing the sky rendering of an
    Image with ``rotation_deg=pa_deg`` against the unrotated template.
    """
    from scipy import ndimage

    n = template.shape[0]
    target = template / template.sum()
    best = None
    for sign in (1.0, -1.0):
        rotated = (
            onp.clip(
                ndimage.rotate(
                    template, sign * pa_deg, reshape=False, order=1
                ),
                0.0,
                None,
            )
            + 1e-6 * template.max()
        )
        sky = onp.asarray(
            Image.from_brightness(
                rotated, pixel_scale_mas, rotation_deg=pa_deg
            ).render(n, n * pixel_scale_mas)
        )
        corr = float(onp.sum((sky - sky.mean()) * (target - target.mean())))
        if best is None or corr > best[0]:
            best = (corr, rotated)
    return best[1]


def evidence_grid(
    data, draw, *, star=True, starts=None, spec=None, **fit_options
):
    """Fit Gaussian-process images over a grid of amplitude and length.

    !!! warning "Provisional"
        May change in any 0.4.x release without a deprecation cycle.

    Each point is a MAP fit of an image whose log-brightness is a
    [`GaussianField`][virgil.fields.GaussianField] of amplitude ``σ`` and
    correlation length ``ℓ`` about the group's starting image, scored by
    [`log_evidence`][virgil.imaging.log_evidence]. The grid starts at
    ``spec.gp_sigmas`` by ``spec.gp_lengths`` and grows while its best point is
    on an edge; for elongated data a second grid is anisotropic. See
    [Imaging, part 7](../imaging_ensemble.md#gaussian-process-members). The
    evidence assumes correct error bars; check them with
    [`error_scale`][virgil.imaging.error_scale].

    Parameters
    ----------
    data : OIData or sequence of OIData
        The data.
    draw : Draw
        Geometry and starting image, as from
        [`draw_groups`][virgil.ensemble.draw_groups].
    star : bool, optional
        As for [`run_group`][virgil.ensemble.run_group].
    starts : dict, optional
        From [`reference_starts`][virgil.ensemble.reference_starts].
    spec : EnsembleSpec, optional
        The grid's settings (default ``EnsembleSpec()``).
    **fit_options
        Passed to [`fit`][virgil.fitting.fit].

    Returns
    -------
    EvidenceGrid
    """
    spec = EnsembleSpec() if spec is None else spec
    if starts is None or draw.start not in starts:
        starts = reference_starts(data, star, (draw.start,), spec.aperture_mas)
    reference = starts[draw.start]
    n, scale = draw.npix, draw.pixel_scale_mas
    template = onp.maximum(
        onp.asarray(resample(reference, None, n, scale)), 0.0
    )
    template = template + 1e-3 * max(template.max(), 1e-30)
    flux = reference.flux
    minor = beam(data).minor_mas
    lengths0 = [f * minor for f in spec.gp_lengths]
    length_limits = [f * minor for f in spec.gp_length_limits]

    def scene(field, rotation):
        env = Image(field, scale, flux=flux, rotation_deg=rotation)
        return System(star=PointSource(), env=env) if star else env

    extra = {"env.flux": dist.LogUniform(1e-4, 1e3)} if star else {}
    others = () if star else (Centroid(0.1 * minor),)

    variants = {"iso": (template, 0.0, lambda length: length)}
    ratio = pa = None
    if spec.gp_axis_ratio is not None:
        ratio, pa = _elongation(data, star, n * scale)
        if ratio < spec.gp_axis_ratio:
            root = onp.sqrt(ratio)
            variants["aniso"] = (
                _oriented(template, pa, scale),
                pa,
                lambda length: (length / root, length * root),
            )

    fits, log_z, axes = {}, {}, {}
    for name, (mean, rotation, lens) in variants.items():
        previous = None

        def evaluate(
            length, sigma, name=name, mean=mean, rotation=rotation, lens=lens
        ):
            nonlocal previous
            field = GaussianField(
                onp.zeros((n, n)), sigma, lens(length), mean=mean
            )
            model = scene(field, rotation)
            result = fit(
                model,
                image_priors(model) | extra,
                data,
                others,
                init=None if previous is None else previous.values,
                **fit_options,
            )
            previous = result
            z = float(
                log_evidence(
                    result,
                    data,
                    "env" if star else None,
                    priors=image_priors(model) | extra,
                    regularizers=others,
                )
            )
            fits[name, length, sigma] = GPFit(name, length, sigma, z, result)
            return z

        ls, ss, values = _grow_grid(
            evaluate,
            lengths0,
            spec.gp_sigmas,
            length_limits,
            spec.gp_sigma_limits,
            spec.gp_growths,
        )
        axes[name] = (tuple(ls), tuple(ss))
        log_z[name] = onp.array(
            [[values[length, sigma] for sigma in ss] for length in ls]
        )
    finite = {k: v for k, v in fits.items() if onp.isfinite(v.log_evidence)}
    if not finite:
        raise ValueError(
            "Every Gaussian-process fit had a non-finite evidence."
        )
    best = max(finite.values(), key=lambda f: f.log_evidence)
    ls, ss = axes[best.variant]
    i, j = ls.index(best.length_mas), ss.index(best.sigma)
    interior = 0 < i < len(ls) - 1 and 0 < j < len(ss) - 1
    return EvidenceGrid(
        {k: a[0] for k, a in axes.items()},
        {k: a[1] for k, a in axes.items()},
        log_z,
        fits,
        best,
        bool(interior),
        ratio,
        pa,
    )


def gp_members(
    data, draw, *, star=True, starts=None, spec=None, **fit_options
):
    """Fit one Gaussian-process group of an ensemble.

    !!! warning "Provisional"
        May change in any 0.4.x release without a deprecation cycle.

    The group's members are the fits of its
    [`evidence_grid`][virgil.ensemble.evidence_grid] whose log evidence is
    within ``spec.gp_window_nats`` of the best, so that
    [`combine`][virgil.ensemble.combine] averages over the amplitude and length
    the data leave open. There is no L-curve window.

    Parameters
    ----------
    data, draw, star, starts, spec, **fit_options
        As for [`evidence_grid`][virgil.ensemble.evidence_grid].

    Returns
    -------
    Group
        With ``curve=None``, ``gp`` the members and ``grid`` the grid.
    """
    spec = EnsembleSpec() if spec is None else spec
    grid = evidence_grid(
        data, draw, star=star, starts=starts, spec=spec, **fit_options
    )
    top = grid.best.log_evidence
    chosen = sorted(
        (
            f
            for f in grid.fits.values()
            if onp.isfinite(f.log_evidence)
            and f.log_evidence >= top - spec.gp_window_nats
        ),
        key=lambda f: -f.log_evidence,
    )
    return Group(draw, None, tuple(chosen), grid)


def _corner(curve):
    try:
        return curve.corner()
    except ValueError:
        return None


# deliberate: test seam (tests/test_ensemble.py)
# deliberate: test seam (tests/test_ensemble.py)
def _register(p, ref, shift, scale, v2_only):
    """Align ``p`` (and its 180° inversion under ``v2_only``) to ``ref``."""
    p0, s0 = (p, (0.0, 0.0)) if shift is None else align(p, ref, shift, scale)
    p0 = onp.asarray(p0)
    if not v2_only:
        return p0, s0, False
    inv = p[::-1, ::-1]
    p1, s1 = (
        (inv, (0.0, 0.0)) if shift is None else align(inv, ref, shift, scale)
    )
    p1 = onp.asarray(p1)
    if onp.sum(onp.abs(p1 - ref)) < onp.sum(onp.abs(p0 - ref)):
        return p1, s1, True
    return p0, s0, False


def _on_common_grid(image, npix, scale):
    """The image's pixels on the common grid, flux conserved.

    A rotated image (an anisotropic Gaussian process) is rendered straight
    onto the common grid, so that its corners are not cropped by a rotation
    within its own frame; others are resampled.
    """
    if float(image.rotation_deg) == 0.0:
        return onp.asarray(
            resample(image, image.pixel_scale_mas, npix, scale), dtype=float
        )
    sky = onp.asarray(image.render(npix, npix * scale), dtype=float)
    return sky / sky.sum() * float(onp.sum(image.brightness))


def _parts(model, star):
    """(the member's image, its fraction of the total flux)."""
    if not star:
        return model, 1.0
    env, fs = model.env, float(model.star.flux)
    fe = float(env.flux)
    return env, fe / (fs + fe)


def _mixture(images, fractions, chosen, star):
    """The mean of the ``chosen`` members' normalized scenes.

    Each image keeps its own grid, so the visibilities are exactly the mean
    of the members'. The others carry zero flux: every subset has the same
    structure and so shares one compilation of the χ².
    """
    share = onp.zeros(len(images))
    share[list(chosen)] = 1.0 / len(chosen)
    fluxes = share * onp.asarray(fractions, dtype=float)
    parts = {
        f"member{j}": image.set("flux", jax.numpy.asarray(flux))
        for j, (image, flux) in enumerate(zip(images, fluxes))
    }
    if not star:
        return System(**parts)
    star_flux = jax.numpy.asarray(1.0 - fluxes.sum())
    return System(star=PointSource(flux=star_flux), **parts)


def combine(data, groups, *, spec=None, star=True):
    """Select the members of an ensemble and average them.

    The selection follows MYTHRA (Drevon et al. 2026): keep the weights just
    before each L-curve's corner (Gaussian-process members skip this), drop
    diverged, badly fitting and outlying members, resample the survivors to a
    common grid, and add each to a running mean, in order of χ², only if the
    mean still fits every dataset as well as the best member, up to
    ``spec.mean_rtol``. See [Imaging, part 7](../imaging_ensemble.md#selection-and-the-mean).
    With a star, the mean is of the whole normalized sky, star included.

    Parameters
    ----------
    data : OIData or sequence of OIData
        The data the groups were fitted to. Give visibilities and closure
        phases as separate datasets to judge each separately.
    groups : sequence of Group
        From [`run_group`][virgil.ensemble.run_group].
    spec : EnsembleSpec, optional
        The selection settings (default ``EnsembleSpec()``).
    star : bool, optional
        As for [`run_group`][virgil.ensemble.run_group].

    Returns
    -------
    Ensemble
    """
    spec = EnsembleSpec() if spec is None else spec
    datasets = _as_list(data)
    groups = sorted(groups, key=lambda g: g.draw.index)

    def chi2_of(result):
        info = result.info
        return tuple(
            float(c) / n
            for c, n in zip(info["chi2"][: len(datasets)], info["ndata"])
        )

    members = []
    for group in groups:
        if group.curve is None:
            # A Gaussian-process group has no L-curve and so no window.
            for fit_ in group.gp:
                members.append(
                    Member(
                        group.draw,
                        float("nan"),
                        fit_.result,
                        chi2_of(fit_.result),
                        gp=fit_,
                    )
                )
            continue
        curve = group.curve
        corner = _corner(curve)
        for weight, result in zip(curve.weights, curve.results):
            inside = corner is None or (
                corner * 10**-spec.window_dex * (1 - 1e-9)
                <= float(weight)
                <= corner * (1 + 1e-9)
            )
            members.append(
                Member(
                    group.draw,
                    float(weight),
                    result,
                    chi2_of(result),
                    reason=None if inside else "window",
                )
            )

    def drop(indices, reason):
        for i in indices:
            members[i] = dataclasses.replace(members[i], reason=reason)

    def keep(mask, reason):
        """Drop the live members where ``mask`` is False, for ``reason``."""
        nonlocal live
        drop([i for i, k in zip(live, mask) if not k], reason)
        live = [i for i, k in zip(live, mask) if k]

    live = [i for i, m in enumerate(members) if m.reason is None]
    if not live:
        raise ValueError("No member lies in its group's L-curve window.")
    # A fit that diverged (a NaN or infinite chi2) is dropped first, and so
    # cannot set the best chi2 that the others are judged against: NaN
    # propagates through min, which would reject every member.
    chi2 = onp.array([members[i].chi2_red for i in live])
    finite = onp.all(onp.isfinite(chi2), axis=1)
    keep(finite, "diverged")
    if not live:
        raise ValueError("Every member's fit diverged (non-finite chi2).")
    chi2 = chi2[finite]
    best = chi2.min(axis=0)
    good = onp.all(
        (chi2 <= spec.max_chi2_red) & (chi2 <= spec.chi2_ratio * best),
        axis=1,
    )
    keep(good, "chi2")
    if not live:
        raise ValueError(
            f"No member has chi2/N below max_chi2_red={spec.max_chi2_red} "
            "on every dataset."
        )
    total = onp.array([members[i].total_chi2_red for i in live])
    median = onp.median(total)
    mad = 1.4826 * onp.median(onp.abs(total - median))
    if mad > 0:
        keep(total <= median + spec.mad_cut * mad, "outlier")

    # The common grid: the finest pixels and the largest field.
    live.sort(key=lambda i: members[i].total_chi2_red)
    scale = min(members[i].draw.pixel_scale_mas for i in live)
    field = max(
        members[i].draw.npix * members[i].draw.pixel_scale_mas for i in live
    )
    npix = int(onp.ceil(field / scale - 1e-6))
    pixels, fractions = [], []
    for i in live:
        image, fraction = _parts(members[i].result.model, star)
        pixels.append(_on_common_grid(image, npix, scale))
        fractions.append(fraction)
    v2_only = not any(d.has_phases for d in datasets)
    shift = (
        None
        if star
        else (
            beam(data).major_mas
            if spec.max_shift_mas is None
            else spec.max_shift_mas
        )
    )
    shifts = [(0.0, 0.0)] * len(live)
    flips = [False] * len(live)
    if not star or v2_only:
        aligned = [
            _register(p, pixels[0], shift, scale, v2_only) for p in pixels[1:]
        ]
        pixels = [pixels[0]] + [a[0] for a in aligned]
        shifts = [(0.0, 0.0)] + [a[1] for a in aligned]
        flips = [False] + [a[2] for a in aligned]
    # A member's image carries its fraction of the total flux; float64, so
    # that averaging identical images returns them unchanged.
    pixels = [onp.asarray(p, dtype=float) for p in pixels]
    weighted = [f * p / p.sum() for p, f in zip(pixels, fractions)]
    # Each member's Image on its own grid, moved by its shift (dra, ddec)
    # and flipped by 180° when _register selected the inverted orientation.
    images = []
    for i, s, flip in zip(live, shifts, flips):
        im = members[i].result.model.env if star else members[i].result.model
        if flip:
            im = dataclasses.replace(
                im,
                log_brightness=im.eta[::-1, ::-1],
                support=None if im.support is None else im.support[::-1, ::-1],
                dra=-im.dra,
                ddec=-im.ddec,
            )
        if not star:
            im = im.set(
                ["dra", "ddec"],
                [im.dra + float(s[0]), im.ddec + float(s[1])],
            )
        images.append(im)

    # The iterative mean, from the best member, judged in float64 so that
    # rounding cannot reject a member that changes nothing. It is judged as
    # the mixture of the members' images on their own grids, which is
    # exact: the common grid is only for display, as resampling there
    # smooths the images.
    with run_in("float64"):
        observations = cast_tree(datasets, "float64")
        images = cast_tree(images, "float64")

        def chi2_red(members):
            model = _mixture(images, fractions, members, star)
            chi2 = _chi2(model, observations)
            return tuple(
                c / d.n_independent for c, d in zip(chi2, observations)
            )

        chosen = [0]
        current = chi2_red(chosen)
        trace = [current]
        if spec.mean_rtol is None:
            rtol = [onp.sqrt(2.0 / d.n_independent) for d in datasets]
        else:
            rtol = [spec.mean_rtol] * len(datasets)
        # Rounding: the mean of identical images is not always
        # bit-identical to them.
        bound = [b * (1.0 + r + 1e-9) for b, r in zip(current, rtol)]
        for k in range(1, len(live)):
            candidate = chi2_red(chosen + [k])
            if all(c <= b for c, b in zip(candidate, bound)):
                chosen, current = chosen + [k], candidate
                trace.append(current)
            else:
                drop([live[k]], "mean")
        model = _mixture(
            [images[j] for j in chosen],
            [fractions[j] for j in chosen],
            range(len(chosen)),
            star,
        )
    for k in chosen:
        members[live[k]] = dataclasses.replace(
            members[live[k]], kept=True, reason=None
        )
    if len(chosen) < spec.min_kept:
        warnings.warn(
            f"Only {len(chosen)} of {len(members)} members were kept "
            f"(fewer than min_kept={spec.min_kept}), so the spread is not "
            "a useful map. Look at Ensemble.summary() for where they were "
            "dropped.",
            UserWarning,
            stacklevel=external_stacklevel(),
        )

    stack = onp.stack([weighted[j] for j in chosen])
    fraction = float(onp.mean([fractions[j] for j in chosen]))
    mean = stack.mean(axis=0)
    brightness = mean / mean.sum()
    if star:  # the scene whose image carries ``fraction`` of the total flux
        env = Image.from_brightness(
            brightness, scale, flux=fraction / (1.0 - fraction)
        )
        image = System(star=PointSource(), env=env)
    else:
        image = Image.from_brightness(brightness, scale)
    # In units of the mean image's unit-sum brightness.
    std = stack.std(axis=0) / mean.sum()
    return Ensemble(
        model=model,
        mean=image.env if star else image,
        std=std,
        chi2_red=current,
        trace=trace,
        members=members,
        groups=list(groups),
    )


def ensemble(data, n_groups, key, *, spec=None, star=True, **fit_options):
    """Run, select and average an ensemble of randomized reconstructions.

    [`draw_groups`][virgil.ensemble.draw_groups], then
    [`run_group`][virgil.ensemble.run_group] for each group in turn, then
    [`combine`][virgil.ensemble.combine]. Each group is one L-curve, so the
    ensemble has ``n_groups * spec.n_weights`` members, and compiles once
    per distinct ``(family, npix, pixel_scale_mas)``: a handful, since
    each is drawn from a short list.

    Parameters
    ----------
    data : OIData or sequence of OIData
        The data. Give visibilities and closure phases as separate datasets
        for the mean to be judged on each.
    n_groups : int
        Number of groups.
    key : jax.random.PRNGKey
        The random key.
    spec : EnsembleSpec, optional
        What to draw and how to select (default ``EnsembleSpec()``).
    star : bool, optional
        Whether the scene has an unresolved star at the origin (default).
    **fit_options
        Passed to [`fit`][virgil.fitting.fit].

    Returns
    -------
    Ensemble
    """
    spec = EnsembleSpec() if spec is None else spec
    draws = draw_groups(data, n_groups, key, spec, star)
    starts = reference_starts(
        data, star, sorted({d.start for d in draws}), spec.aperture_mas
    )
    groups = [
        run_group(data, d, star=star, starts=starts, spec=spec, **fit_options)
        for d in draws
    ]
    return combine(data, groups, spec=spec, star=star)
