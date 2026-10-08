"""Ensembles of randomised image reconstructions, averaged into one image.

A single regularised reconstruction depends on choices the data do not fix:
the regulariser and its weight, the pixel size, the field and the starting
image. Drevon et al. (2025, arXiv:2609.15365), who won the 2024
interferometric imaging contest, run many reconstructions with these
choices drawn at random (their PYRA), keep those that fit the data, and
average them while the average still fits (their MYTHRA). The mean is less
sensitive to any one choice than a single reconstruction, and the spread of
the members is a map of how much the image depends on those choices. This
module does the same with virgil's own fits; it is written from the paper's
description, not from their code.

The work is split so that a cluster can run it in parallel:

1. [`draw_groups`][virgil.ensemble.draw_groups] draws the reconstruction
   settings. Each *group* has one geometry (pixel size and number of
   pixels), one regulariser family, one starting image and several
   weights.
2. [`run_group`][virgil.ensemble.run_group] fits one group, as an
   [`l_curve`][virgil.imaging.l_curve] over its weights. The weights are
   traced, so a group compiles once, and groups that share a geometry and
   a family share the compilation.
3. [`combine`][virgil.ensemble.combine] selects the members and averages
   them into an [`Ensemble`][virgil.ensemble.Ensemble].

[`ensemble`][virgil.ensemble.ensemble] runs all three in turn. On a
cluster, run one group per array task (``run_group(data,
draw_groups(data, n, key, spec)[task])``), save the
[`Group`][virgil.ensemble.Group]s and combine them in one more job.

The ensemble's standard deviation is not a posterior uncertainty. It
measures how much the image changes between reasonable reconstruction
choices, not the noise in the data; for the posterior, sample an image
(see the tutorial "Imaging, part 5").
"""

import dataclasses

import jax
import numpy as onp
import numpyro.distributions as dist

from .fitting import FitResult
from .imaging import (
    TSV,
    TV,
    Centroid,
    LCurve,
    MaxEntropy,
    StarletL1,
    _chi2,
    beam,
    field_of_view,
    image_priors,
    l_curve,
    nyquist_pixel_scale,
    starting_image,
)
from ._precision import cast_tree, run_in
from .metrics import align, resample
from .models import Image, PointSource, System

# The regulariser families, by the names used in EnsembleSpec.
FAMILIES = {"tv": TV, "tsv": TSV, "maxent": MaxEntropy, "starlet": StarletL1}


def _default_weight_ranges():
    # Weight per data point. The penalties of a unit-sum image are O(1) for
    # TV, maximum entropy and the starlet L1 norm, and much smaller for
    # TSV, whose steps are squared. On the 60 datasets of virgil-validation's
    # contest bench (12 groups each), the L-curve corners per data point
    # fell (5th to 95th percentile) at 0.003-1.6 for TV, 0.004-1.9 for
    # maximum entropy, 0.003-1.2 for the starlet L1 norm and 3-2000 for
    # TSV, with no trend with the number of pixels; most of the scatter is
    # within a dataset, from the sparse sweeps. Each range reaches a decade
    # below the lowest of these, so that the window below a corner (one
    # decade by default) is sampled, and half a decade or more above the
    # highest, so that the corner is an interior point of the sweep.
    return {
        "tv": (1e-4, 1e1),
        "tsv": (1e-1, 1e4),
        "maxent": (1e-4, 1e1),
        "starlet": (1e-4, 1e1),
    }


@dataclasses.dataclass(frozen=True)
class EnsembleSpec:
    """What an ensemble draws at random, and how it selects members.

    Attributes
    ----------
    families : tuple of str
        Regulariser families to draw from, uniformly: ``"tv"``
        ([`TV`][virgil.imaging.TV]), ``"tsv"``
        ([`TSV`][virgil.imaging.TSV]), ``"maxent"``
        ([`MaxEntropy`][virgil.imaging.MaxEntropy]) and ``"starlet"``
        ([`StarletL1`][virgil.imaging.StarletL1]).
    weight_ranges : dict
        For each family, the range ``(low, high)`` of the weight per data
        point (the weight divided by the total number of independent data).
        Weights are drawn log-uniformly in it, the invariant prior for a
        scale, one in each of ``n_weights`` equal bins of ``log w`` so that
        every sweep spans the range. The defaults are wide enough to hold
        the corners of the contest bench's datasets with a window's width
        to spare below them; if the L-curve corners fall at the edges, move
        the range.
    n_weights : int
        Weights per group, at least three (the L-curve's corner needs
        them).
    oversample : tuple of float
        Pixels per Nyquist pixel
        ([`nyquist_pixel_scale`][virgil.imaging.nyquist_pixel_scale]),
        drawn uniformly.
    field_factors : tuple of float
        The field is ``field_of_view(data)`` times one of these, drawn
        uniformly. A field larger than the interferometric field of view
        lets the image hold flux the shortest baselines resolve out, which
        on the contest bench gave better images than a field of 1 or less
        (field 4 was the best of its arms). On a uv lattice (AMI) a field
        above 1 aliases: pass factors of at most 1 there.
    starts : tuple of str
        Starting images, drawn uniformly: ``"moments"`` and ``"dirty"``
        (see [`starting_image`][virgil.imaging.starting_image]; a dirty
        start needs data with phases) and ``"flat"``.
    max_npix : int
        Largest number of pixels on a side. A field that would need more
        keeps its size and takes coarser pixels instead, so its pixels per
        Nyquist pixel fall below ``oversample``.
    window_dex : float
        Width of the window of weights kept in each group, in dex, from
        ``corner / 10**window_dex`` up to the L-curve's corner: as in
        MYTHRA, the weights just before the turnover, where the fit to the
        data has stopped improving but the image is not yet over-smoothed.
        Stronger weights, past the corner, trade fit for smoothness fast.
    max_chi2_red : float
        Drop a member if the raw χ² per data point of any dataset exceeds
        this.
    chi2_ratio : float
        Drop a member if, on any dataset, its χ² per data point exceeds
        this multiple of the best member's on that dataset. With
        miscalibrated errors no member reaches χ²/N ≈ 1, so a relative
        threshold is the one that bites.
    mad_cut : float
        Then drop members whose total χ² per data point lies more than this
        many robust standard deviations (1.4826 times the median absolute
        deviation) above the median.
    max_shift_mas : float or None
        Without a star, the members are recentred on the best one, searching
        shifts up to this (default: the beam's major axis). With a star,
        the star fixes the position and they are not shifted.
    mean_rtol : float
        A member joins the mean if no dataset's χ² rises by more than this
        fraction: only enough to forgive rounding, since the mean of
        identical images is not always bit-identical to them. On data that
        the best member fits to the noise, so strict a rule may keep that
        member alone (and the spread is then zero); a value of order the
        χ²/N noise, √(2/N), keeps more.
    """

    families: tuple = ("tv", "tsv", "maxent", "starlet")
    weight_ranges: dict = dataclasses.field(
        default_factory=_default_weight_ranges
    )
    n_weights: int = 6
    oversample: tuple = (2.0, 3.0, 4.0)
    field_factors: tuple = (1.0, 2.0, 4.0)
    starts: tuple = ("moments", "flat")
    max_npix: int = 128
    window_dex: float = 1.0
    max_chi2_red: float = onp.inf
    chi2_ratio: float = 2.0
    mad_cut: float = 5.0
    max_shift_mas: float | None = None
    mean_rtol: float = 1e-9

    def __post_init__(self):
        unknown = set(self.families) - set(FAMILIES)
        if unknown:
            raise ValueError(
                f"Unknown regulariser families {sorted(unknown)}; choose "
                f"from {sorted(FAMILIES)}."
            )
        missing = set(self.families) - set(self.weight_ranges)
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
        The regulariser family.
    npix : int
        Pixels on a side.
    pixel_scale_mas : float
        Pixel size in mas.
    start : str
        The starting image.
    weights : tuple of float
        The regulariser weights, largest first.
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
    curve : LCurve
        Its fits, one per weight, largest weight first.
    """

    draw: Draw
    curve: LCurve


@dataclasses.dataclass(frozen=True)
class Member:
    """One reconstruction in an ensemble.

    Attributes
    ----------
    draw : Draw
        The settings of its group.
    weight : float
        Its regulariser weight.
    result : FitResult
        The fit.
    chi2_red : tuple of float
        Raw χ² per data point of each dataset (with the quoted errors).
    kept : bool
        Whether it is in the mean image.
    reason : str or None
        Why it was left out: ``"window"`` (outside its group's L-curve
        window), ``"diverged"`` (a non-finite χ²: the fit failed), ``"chi2"``
        (a dataset fitted too badly), ``"outlier"``
        (total χ² far above the others) or ``"mean"`` (adding it made the
        mean fit a dataset worse). ``None`` if kept.
    """

    draw: Draw
    weight: float
    result: FitResult
    chi2_red: tuple
    kept: bool = False
    reason: str | None = None

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
        The mean scene, exactly: the kept members' images on their own
        grids (``member0``, ``member1``, ...), each carrying its share of
        the flux, and with a star a ``star`` component. ``chi2_red`` and
        ``trace`` are of this model.
    mean : Image
        The mean extended emission, on the common grid (the finest pixels
        and the largest field of the kept members). Its brightness sums to
        one; with a star its ``flux`` is the mean flux relative to the
        star. Resampling smooths the members' images a little, so this
        image is for display and scoring; it fits the data less well than
        ``model``.
    std : array, shape (npix, npix)
        Standard deviation across the kept members of the same quantity as
        ``mean.brightness``, pixel by pixel.
    chi2_red : tuple of float
        Raw χ² per data point of the mean scene on each dataset.
    trace : list of tuple
        ``chi2_red`` of the running mean after each accepted member, from
        the best member alone; it never rises on any dataset.
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
            corner = _corner(group.curve)
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


def _datasets(data):
    return list(data) if isinstance(data, (list, tuple)) else [data]


def draw_groups(data, n_groups, key, spec=None):
    """Draw the settings of ``n_groups`` groups of reconstructions.

    Each group draws a regulariser family, a pixel size (the Nyquist scale
    over one of ``spec.oversample``), a field (``field_of_view(data)``
    times one of ``spec.field_factors``; beyond ``spec.max_npix`` pixels
    the pixels are coarsened to fit it), a starting image and
    ``spec.n_weights`` log-uniform weights, one in each equal bin of the
    family's range in ``log w``. The draws
    depend only on ``key`` and ``spec``, so every task of a cluster array
    can draw them all and run its own.

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

    Returns
    -------
    list of Draw
        Sorted by geometry, so that groups sharing a compilation run one
        after the other.
    """
    spec = EnsembleSpec() if spec is None else spec
    rng = onp.random.default_rng(
        int(jax.random.randint(key, (), 0, 2**31 - 1))
    )
    nyquist = nyquist_pixel_scale(data)
    field = field_of_view(data)
    n_data = sum(d.n_independent for d in _datasets(data))
    draws = []
    for index in range(int(n_groups)):
        family = spec.families[rng.integers(len(spec.families))]
        scale = nyquist / spec.oversample[rng.integers(len(spec.oversample))]
        factor = spec.field_factors[rng.integers(len(spec.field_factors))]
        npix = int(onp.ceil(factor * field / scale))
        if npix > spec.max_npix:
            # Keep the field and coarsen the pixels.
            npix = spec.max_npix
            scale = factor * field / npix
        start = spec.starts[rng.integers(len(spec.starts))]
        low, high = onp.log(spec.weight_ranges[family]) + onp.log(n_data)
        # Stratified: one log-uniform weight in each of n_weights equal bins.
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


def reference_starts(data, star=True, starts=("moments", "flat")):
    """The starting images that each group resamples onto its own grid.

    Each is made once by [`starting_image`][virgil.imaging.starting_image],
    which fits a star plus a Gaussian envelope: ``"moments"`` is that
    Gaussian, ``"dirty"`` the positive part of the dirty image, and
    ``"flat"`` a uniform image with the fitted flux.

    Returns
    -------
    dict
        Start name to an [`Image`][virgil.models.Image].
    """
    out = {}
    moments = starting_image(data, star=star)
    moments = moments.env if star else moments
    for name in starts:
        if name == "moments":
            out[name] = moments
        elif name == "dirty":
            dirty = starting_image(data, star=star, start="dirty")
            out[name] = dirty.env if star else dirty
        elif name == "flat":
            ones = onp.ones(moments.log_brightness.shape)
            out[name] = Image.from_brightness(
                ones, moments.pixel_scale_mas, flux=moments.flux
            )
        else:
            raise ValueError(f"Unknown start {name!r}.")
    return out


def _start_model(reference, draw, star):
    """The group's starting scene: ``reference`` resampled to its grid."""
    pixels = resample(reference, None, draw.npix, draw.pixel_scale_mas)
    image = Image.from_brightness(
        onp.maximum(onp.asarray(pixels), 0.0),
        draw.pixel_scale_mas,
        floor=1e-3,
        flux=reference.flux,
    )
    return System(star=PointSource(), env=image) if star else image


def run_group(data, draw, *, star=True, starts=None, **fit_options):
    """Fit one group of an ensemble: an L-curve over its weights.

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
    **fit_options
        Passed to [`l_curve`][virgil.imaging.l_curve] and so to
        [`fit`][virgil.fitting.fit], e.g. ``method`` and ``max_steps``.

    Returns
    -------
    Group
    """
    if starts is None or draw.start not in starts:
        starts = reference_starts(data, star, (draw.start,))
    model = _start_model(starts[draw.start], draw, star)
    path = "env" if star else None
    priors = image_priors(model)
    others = ()
    if star:
        # The image's flux relative to the star is a scale: log-uniform.
        priors["env.flux"] = dist.LogUniform(1e-4, 1e3)
    else:
        others = (Centroid(draw.pixel_scale_mas),)
    curve = l_curve(
        model,
        priors,
        data,
        FAMILIES[draw.family](1.0, path=path),
        draw.weights,
        others,
        **fit_options,
    )
    return Group(draw, curve)


def _corner(curve):
    try:
        return curve.corner()
    except ValueError:
        return None


def _parts(model, star):
    """(unit-sum image pixels, the image's fraction of the total flux)."""
    if not star:
        return onp.asarray(model.brightness, dtype=float), 1.0
    env, fs = model.env, float(model.star.flux)
    fe = float(env.flux)
    return onp.asarray(env.brightness, dtype=float), fe / (fs + fe)


def _mean_model(pixels, fraction, scale, star):
    """The scene whose image carries ``fraction`` of the total flux."""
    brightness = pixels / pixels.sum()
    if not star:
        return Image.from_brightness(brightness, scale)
    image = Image.from_brightness(
        brightness, scale, flux=fraction / (1.0 - fraction)
    )
    return System(star=PointSource(), env=image)


def _member_image(model, shift, star):
    """A member's Image on its own grid, moved by ``shift`` (dra, ddec)."""
    if star:
        return model.env
    return model.set(
        ["dra", "ddec"],
        [model.dra + float(shift[0]), model.ddec + float(shift[1])],
    )


def _mixture(images, fractions, chosen, star):
    """The mean of the ``chosen`` members' normalised scenes.

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


def _chi2_red(model, datasets):
    chi2 = _chi2(model, datasets)
    return tuple(c / d.n_independent for c, d in zip(chi2, datasets))


def combine(data, groups, *, spec=None, star=True):
    """Select the members of an ensemble and average them.

    The selection follows MYTHRA (Drevon et al. 2025):

    1. In each group, keep the weights from ``spec.window_dex`` below the
       L-curve's corner up to the corner: the weights just before the
       turnover.
    2. Drop members whose fit diverged (a non-finite χ²), then keep
       members whose raw χ² per data point is below
       ``spec.max_chi2_red`` and within ``spec.chi2_ratio`` of the best
       member's on every dataset, then drop total-χ² outliers by their
       median absolute deviation (``spec.mad_cut``).
    3. Resample the survivors' images to a common grid, the finest pixels
       and the largest field among them, conserving flux; without a star,
       recentre each on the best member
       ([`align`][virgil.metrics.align]).
    4. In order of total χ², add members to a running mean one at a time,
       keeping each only if the mean's χ² does not rise on any dataset (so
       visibilities and closure phases, given as separate datasets, are
       judged separately). The running mean is judged as the mixture of
       the members' images on their own grids, which is exact; resampling
       to the common grid smooths them, which on precise data can raise
       χ² several-fold and so let worse members through.

    With a star, the mean is of the whole normalised sky, star included:
    the star's fraction of the flux is the members' mean, and the image's
    pixels the mean of their fluxes.

    Parameters
    ----------
    data : OIData or sequence of OIData
        The data the groups were fitted to.
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
    datasets = _datasets(data)
    groups = sorted(groups, key=lambda g: g.draw.index)

    members = []
    for group in groups:
        curve = group.curve
        corner = _corner(curve)
        for weight, result in zip(curve.weights, curve.results):
            info = result.info
            chi2_red = tuple(
                float(c) / n
                for c, n in zip(info["chi2"][: len(datasets)], info["ndata"])
            )
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
                    chi2_red,
                    reason=None if inside else "window",
                )
            )

    def drop(indices, reason):
        for i in indices:
            members[i] = dataclasses.replace(members[i], reason=reason)

    live = [i for i, m in enumerate(members) if m.reason is None]
    if not live:
        raise ValueError("No member lies in its group's L-curve window.")
    # A fit that diverged (a NaN or infinite chi2) is dropped first, and so
    # cannot set the best chi2 that the others are judged against: NaN
    # propagates through min, which would reject every member.
    chi2 = onp.array([members[i].chi2_red for i in live])
    finite = onp.all(onp.isfinite(chi2), axis=1)
    drop([i for i, f in zip(live, finite) if not f], "diverged")
    live = [i for i, f in zip(live, finite) if f]
    if not live:
        raise ValueError("Every member's fit diverged (non-finite chi2).")
    chi2 = chi2[finite]
    best = chi2.min(axis=0)
    good = onp.all(
        (chi2 <= spec.max_chi2_red) & (chi2 <= spec.chi2_ratio * best),
        axis=1,
    )
    drop([i for i, g in zip(live, good) if not g], "chi2")
    live = [i for i, g in zip(live, good) if g]
    if not live:
        raise ValueError(
            f"No member has chi2/N below max_chi2_red={spec.max_chi2_red} "
            "on every dataset."
        )
    total = onp.array([members[i].total_chi2_red for i in live])
    median = onp.median(total)
    mad = 1.4826 * onp.median(onp.abs(total - median))
    if mad > 0:
        outlier = total > median + spec.mad_cut * mad
        drop([i for i, o in zip(live, outlier) if o], "outlier")
        live = [i for i, o in zip(live, outlier) if not o]

    # The common grid: the finest pixels and the largest field.
    live.sort(key=lambda i: members[i].total_chi2_red)
    scale = min(members[i].draw.pixel_scale_mas for i in live)
    field = max(
        members[i].draw.npix * members[i].draw.pixel_scale_mas for i in live
    )
    npix = int(onp.ceil(field / scale - 1e-6))
    pixels, fractions = [], []
    for i in live:
        draw = members[i].draw
        image, fraction = _parts(members[i].result.model, star)
        pixels.append(
            onp.asarray(resample(image, draw.pixel_scale_mas, npix, scale))
        )
        fractions.append(fraction)
    shifts = [(0.0, 0.0)] * len(live)
    if not star:
        shift = spec.max_shift_mas
        if shift is None:
            shift = beam(data).major_mas
        aligned = [align(p, pixels[0], shift, scale) for p in pixels[1:]]
        pixels = [pixels[0]] + [onp.asarray(a[0]) for a in aligned]
        shifts = [(0.0, 0.0)] + [a[1] for a in aligned]
    # A member's image carries its fraction of the total flux; float64, so
    # that averaging identical images returns them unchanged.
    pixels = [onp.asarray(p, dtype=float) for p in pixels]
    weighted = [f * p / p.sum() for p, f in zip(pixels, fractions)]
    images = [
        _member_image(members[i].result.model, s, star)
        for i, s in zip(live, shifts)
    ]

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
            return _chi2_red(model, observations)

        chosen = [0]
        current = chi2_red(chosen)
        trace = [current]
        for k in range(1, len(live)):
            candidate = chi2_red(chosen + [k])
            if all(
                c <= b * (1.0 + spec.mean_rtol)
                for c, b in zip(candidate, current)
            ):
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

    stack = onp.stack([weighted[j] for j in chosen])
    fraction = float(onp.mean([fractions[j] for j in chosen]))
    mean = stack.mean(axis=0)
    image = _mean_model(mean, fraction, scale, star)
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
    """Run, select and average an ensemble of randomised reconstructions.

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
    draws = draw_groups(data, n_groups, key, spec)
    starts = reference_starts(data, star, sorted({d.start for d in draws}))
    groups = [
        run_group(data, d, star=star, starts=starts, **fit_options)
        for d in draws
    ]
    return combine(data, groups, spec=spec, star=star)
