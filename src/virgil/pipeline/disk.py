"""`DiskPipeline`: rings and haloes fitted with `MultipoleDisk` (provisional)."""

from __future__ import annotations

import dataclasses
import functools
import math
import re
import warnings
from typing import Callable

import numpy as onp

from .._precision import in_float64
from . import _checks, _io
from ._common import (
    _overview_plots,
    _quicklook,
    _scales_of,
    _chi2_64,
    _m2loglike_64,
    _chain_health,
    _diagnostics,
    _draws,
    _geometry,
    _integer,
    _interior_start,
    _load_report,
    _noise_quantiles,
    _pick,
    _positive,
    _prior_bounds,
    _run_chains,
    _sampler_checks,
    _sampler_report,
    _save,
    _save_fit,
    _site_summary,
    _validate_common,
    _write_samples,
)
from ._core import Stage, _Pipeline

# The quadrature of a profile's ring is cut at this many σ; a ring's prior
# floor on σ and the radial grid below are stated in half-fringes.
_WIDTH_FLOOR = 0.1
# Prior ranges (group-invariant: log-uniform for scales, uniform for
# locations and angles).
_FLUX_RANGE = (1e-3, 10.0)
_PLANET_FLUX_RANGE = (1e-4, 1.0)
_INDEX_RANGE = (-10.0, 10.0)
_TEMPERATURE_RANGE = (300.0, 3000.0)
_TEMPERATURE_START = 1000.0
# The rings' harmonics are fractions of the ring's light (|w| <= 1 keeps the
# intensity positive); an isotropic Gaussian on re and im has a uniform phase,
# where a box would favour the diagonals.
_HARMONIC_SCALE = 0.4
# Fluxes (relative to the star at the reference wavelength) tried by the
# grid searches; the geometry grid profiles over them.
_FLUX_GRID = onp.geomspace(0.003, 3.0, 8)
_SEARCH_FLUX_GRID = onp.geomspace(1e-4, 0.5, 12)
# Fractional bounds of the GP's σ from the single ring's dynamic range.
_GP_SIGMA_BOUNDS = (0.5, 5.0)
_ALPHA_SIGMA = 3.0
_HARMONIC_SIGMA = 0.5
_HARMONIC_NODES = 10
_ALPHA_NODES = 22
# Squared-exponential equivalent of the Matérn ν = 3/2 length ℓ: ℓ/√3.
_ELL_PER_EQUIVALENT = math.sqrt(3.0)
# Largest side (pixels) of the image rendered by the ``render_consistency``
# check.
_RENDER_NPIX = 512
_UNSET = object()
_BACKFIT_PASSES = 2
_BACKFIT_MIN_GAIN = 2.0  # BIC units: below this a re-placement is noise
# Coarse starts for the Gaussian process's geometry, beside the single ring's.
_GEOMETRY_LATTICE_INC = (30.0, 50.0, 70.0)
_GEOMETRY_LATTICE_PA = (0.0, 45.0, 90.0, 135.0)
_SCREEN_STEPS = 25


# === THE MODEL REGISTRY ===


@dataclasses.dataclass(frozen=True)
class DiskModel:
    """A model ``DiskPipeline`` can fit, registered under a short name.

    Entries of ``MODELS`` are parallel options, not a chain: each runs the
    ``profile`` and ``compare`` stages on its own settings and checks. Adding an
    entry (say a cone) is adding a row to ``MODELS``, with no new stage.

    Attributes
    ----------
    name : str
        The short name accepted by ``DiskPipeline(model=...)``.
    label : str
        Human-readable name, for plots and messages.
    disk : str
        What the disk is, in one line.
    profile : callable
        ``profile(pipeline) -> dict``: the ``profile`` stage.
    compare : callable
        ``compare(pipeline) -> dict``: the ``compare`` stage; it saves the
        preferred fit as ``models/best``.
    checks : tuple of str
        The names of the checks that apply (from ``CHECKS``).
    candidate : str
        Prefix of the names of the candidate models under ``models/``.
    """

    name: str
    label: str
    disk: str
    profile: Callable
    compare: Callable
    checks: tuple
    candidate: str


# Every check of the pipeline, in the order they are reported.
CHECKS = (
    "chi2_reduced",
    "length_scale",
    "structure_unconstrained",
    "ring_count",
    "inner_edge",
    "outer_edge",
    "inclination",
    "absorbed_companion",
    "render_consistency",
    "at_bound",
    "laplace_pd",
)

_MODELS = None


def models():
    """The registry of short names to [`DiskModel`][virgil.pipeline.disk.DiskModel]."""
    global _MODELS
    if _MODELS is None:
        _MODELS = {
            e.name: e
            for e in (
                DiskModel(
                    name="rings",
                    label="Gaussian rings",
                    disk="N GaussianRing profiles sharing inc and pa",
                    profile=_profile_rings,
                    compare=_compare_rings,
                    checks=(
                        "chi2_reduced",
                        "ring_count",
                        "outer_edge",
                        "inclination",
                        "absorbed_companion",
                        "at_bound",
                        "laplace_pd",
                    ),
                    candidate="rings",
                ),
                DiskModel(
                    name="gp",
                    label="Gaussian-process disk",
                    disk="a RadialGP profile with low-degree harmonics",
                    profile=_profile_gp,
                    compare=_compare_gp,
                    checks=(
                        "chi2_reduced",
                        "length_scale",
                        "structure_unconstrained",
                        "inner_edge",
                        "outer_edge",
                        "inclination",
                        "absorbed_companion",
                        "render_consistency",
                        "at_bound",
                        "laplace_pd",
                    ),
                    candidate="gp",
                ),
            )
        }
    return _MODELS


# === THE PIPELINE ===


class DiskPipeline(_Pipeline):
    """Fit a disk, ring or halo with a [`MultipoleDisk`][virgil.models.MultipoleDisk].

    !!! warning "Provisional"
        May change in any 0.4.x release without a deprecation cycle.

    The star is a grey point source unless ``star_diameter_mas``, ``star_temperature_k`` or
    ``star_index`` says otherwise; the disk's spectral index is relative to it. Stages: ``load``
    (``wavel_range`` [min, max] in metres; ``error_floor`` a dict of absolute floors),
    ``overview`` (plots; sets the half-fringe λ/2B of the longest baseline), ``geometry`` (χ²
    grid over cos ``inc`` and ``pa``, then a single-ring fit), ``profile``, ``compare``,
    ``posterior`` (NUTS, with ``posterior=True``) and ``quicklook``. ``model="gp"`` fits a
    ``RadialGP`` profile at each length scale and keeps the largest Laplace evidence;
    ``"rings"`` adds Gaussian rings until the BIC stops improving. ``run()`` returns a
    ``Result``: ``res.summary["disk"]`` (``n_rings`` and ``rings`` for ``"rings"``),
    ``res.model()`` (the best fit), ``res.checks``. All settings, with defaults:
    [pipeline settings](../pipeline.md#diskpipeline-settings).

    Parameters
    ----------
    data : OIData
        The data.
    model : {"gp", "rings"}, optional
        What to fit (default ``"gp"``, the option that assumes least).
    output : str or os.PathLike, optional
        Run folder (default ``"run"``).
    **settings
        Overrides of ``DiskPipeline.defaults()``. The most used: ``r_in_mas``
        and ``r_out_mas`` (radial range, mas), ``max_rings`` (most rings, 4),
        ``ell_halffringes`` (the ℓ grid, [1, 2, 4]), ``companion`` (search the
        residuals for a point source, False), ``posterior`` (False),
        ``error_scale`` (``"auto"``: rescaled for ``"gp"``, quoted for
        ``"rings"``; ``"fit"`` for rings). Priors and errors:
        [conventions](../conventions.md).

    Examples
    --------
    >>> import virgil as vg  # doctest: +SKIP
    >>> from virgil.pipeline import DiskPipeline  # doctest: +SKIP
    >>> res = DiskPipeline(vg.OIData("disk.oifits"), "rings", output="runs/disk").run()  # doctest: +SKIP
    >>> res.summary["disk"]["n_rings"], res.model()  # doctest: +SKIP
    """

    NAME = "disk"
    STABILITY = "provisional"
    STAGES = (
        "load",
        "overview",
        "geometry",
        "profile",
        "compare",
        "posterior",
        "quicklook",
    )
    _DEFAULTS = {
        "r_in_mas": None,
        "r_out_mas": None,
        "n_r": None,
        "n_inc": 10,
        "n_pa": 12,
        "inc_max_deg": 85.0,
        "max_rings": 4,
        "ell_halffringes": [1.0, 2.0, 4.0],
        "ell_floor_halffringes": 0.5,
        "harmonic_degree": 1,
        "spectral_index": None,
        "spectral_index_min_channels": 3,
        "ring_colour": "power_law",
        "wavel0_m": None,
        "star_temperature_k": None,
        "star_diameter_mas": None,
        "star_index": None,
        "companion": False,
        "companion_min_delta_chi2": 25.0,
        "companion_max_sep_mas": None,
        "companion_step_mas": None,
        "length_scale_nats": 3.0,
        "ring_count_nats": 3.0,
        "inner_edge_fraction": 0.1,
        "inner_edge_correlation": 0.9,
        "outer_edge_fraction": 0.01,
        "inclination_min_deg": 15.0,
        "render_tolerance": 0.1,
        "unconstrained_correlation": 0.8,
        "wavel_range": None,
        "error_floor": None,
        "error_scale": "auto",
        "batch_size": None,
        "posterior": False,
        "num_warmup": 1000,
        "num_samples": 1000,
        "num_chains": 4,
        "chain_method": "vectorized",
        "seed": 0,
        "quicklook_execute": True,
    }

    def __init__(self, data, model=None, *, output="run", **settings):
        registry = models()
        name = self._default_model() if model is None else model
        if not isinstance(name, str) or name not in registry:
            raise ValueError(
                f"model must be one of {sorted(registry)}; got {model!r}."
            )
        self.name = name
        super().__init__(data, name, output=output, **settings)
        if name == "gp" and self.settings["ring_colour"] != "power_law":
            raise ValueError(
                'ring_colour="blackbody" is for model="rings": the Gaussian '
                "process's colour is its own spectral index."
            )
        if name == "gp" and self.settings["error_scale"] == "fit":
            raise ValueError(
                'error_scale="fit" is not available for model="gp": the '
                "Laplace evidence that chooses ℓ needs fixed errors; use "
                '"rescale" (the default), which estimates one scale and '
                "then fixes it."
            )
        if name == "rings" and self.settings["error_scale"] == "rescale":
            raise ValueError(
                'error_scale="rescale" is for model="gp" (the MacKay scale '
                'needs Gaussian-process latents); use "fit" for rings.'
            )
        if (
            self.settings["star_temperature_k"] is not None
            and self.settings["star_index"] is not None
        ):
            raise ValueError(
                "Give star_temperature_k or star_index, not both: each sets "
                "the star's spectrum."
            )

    _error_factor = _UNSET
    _scaled = None

    @property
    def error_mode(self):
        """``"quoted"``, ``"fit"`` or ``"rescale"``: ``error_scale`` with
        ``"auto"`` resolved (``"rescale"`` for ``"gp"``, else ``"quoted"``)."""
        return _error_mode(self.settings, self.name)

    @property
    def processed(self):
        """The data after ``wavel_range`` and ``error_floor``, and (with
        ``error_scale="rescale"``, once the ``profile`` stage has estimated
        it) the MacKay error scale: every fit and evidence after that point
        is on these data."""
        data = super().processed
        factor = self._rescale_factor()
        if factor == 1.0:
            return data
        if self._scaled is None or self._scaled[0] != factor:
            self._scaled = (factor, data.with_error_scale(factor))
        return self._scaled[1]

    def _rescale_factor(self):
        if self.error_mode != "rescale":
            return 1.0
        if self._error_factor is _UNSET:
            # The rescaled errors are the profile stage's result: without it
            # every chi-squared, evidence and width would be on the wrong
            # errors, so a missing or unreadable report is an error, not 1.0.
            try:
                value = self._report("profile").get("error_scale_factor")
            except (OSError, ValueError, KeyError) as err:
                raise RuntimeError(
                    'error_scale="rescale" needs the error scale the profile '
                    f"stage estimated, but its report could not be read ({err}); "
                    "rerun the profile stage."
                ) from err
            if value is None:
                raise RuntimeError(
                    'error_scale="rescale" but the profile report has no '
                    "error_scale_factor (it was made with another error_scale); "
                    "rerun the profile stage."
                )
            self._error_factor = float(value)
        return self._error_factor

    def _default_model(self):
        return "gp"

    def __repr__(self):
        return (
            f"DiskPipeline(output={str(self.output)!r}, model={self.name!r})"
        )

    @classmethod
    def _validate(cls, s):
        if s["r_in_mas"] is not None:
            _positive(s, "r_in_mas")
        if s["r_out_mas"] is not None and not s["r_out_mas"] > (
            s["r_in_mas"] or 0.0
        ):
            raise ValueError("r_out_mas must exceed r_in_mas.")
        if s["n_r"] is not None and not (
            isinstance(s["n_r"], int) and 8 <= s["n_r"] <= 144
        ):
            raise ValueError("n_r must be an integer from 8 to 144.")
        for name in ("n_inc", "n_pa"):
            _integer(s, name, 2)
        if not 0.0 < s["inc_max_deg"] < 90.0:
            raise ValueError("inc_max_deg must be between 0 and 90.")
        _integer(s, "max_rings", 1)
        ell = s["ell_halffringes"]
        if not ell or any(not v > 0 for v in ell):
            raise ValueError("ell_halffringes must be positive numbers.")
        _integer(s, "harmonic_degree", 0)
        if s["spectral_index"] not in (None, True, False):
            raise ValueError("spectral_index must be None, True or False.")
        if s["ring_colour"] not in ("power_law", "blackbody"):
            raise ValueError('ring_colour must be "power_law" or "blackbody".')
        if s["error_scale"] not in ("auto", "quoted", "fit", "rescale"):
            raise ValueError(
                'error_scale must be "auto", "quoted", "fit" or "rescale".'
            )
        _positive(s, "ell_floor_halffringes")
        if any(v < s["ell_floor_halffringes"] for v in ell):
            raise ValueError(
                "ell_halffringes must not go below ell_floor_halffringes "
                f"({s['ell_floor_halffringes']}): a length below the pixel "
                "scale gives bead-like images that fit with χ²/N far under 1."
            )
        if s["star_diameter_mas"] is not None:
            _positive(s, "star_diameter_mas")
        if s["star_index"] is not None and not onp.isfinite(s["star_index"]):
            raise ValueError("star_index must be finite.")
        _validate_common(s)

    def _model_digest(self):
        return {
            "class": "MultipoleDisk",
            "digest": None,
            "models": [self.name],
        }

    def _ctx(self):
        """The resolution scale and the defaults derived from it."""
        return _Context.from_report(self.settings, self._report("overview"))

    def _stages(self):
        return (
            Stage("load", _load_disk, ("stages/load/report.json",)),
            Stage(
                "overview",
                _overview,
                ("plots/overview_data.png", "plots/overview_uv.png"),
            ),
            Stage(
                "geometry",
                _geometry_stage,
                (
                    "grids.h5",
                    "plots/geometry_grid.png",
                    *_model_files("geometry"),
                ),
            ),
            Stage(
                "profile",
                _profile,
                ("stages/profile/report.json", "plots/profile_fits.png"),
            ),
            Stage(
                "compare",
                _compare,
                ("plots/disk_summary.png", *_model_files("best")),
            ),
            Stage("posterior", _posterior, ("stages/posterior/report.json",)),
            Stage("quicklook", _quicklook, ("quicklook.ipynb",)),
        )

    def _summarize(self, reports):
        return _summarize(self.name, self.settings, reports)


def _error_mode(settings, name):
    mode = settings["error_scale"]
    if mode == "auto":
        return "rescale" if name == "gp" else "quoted"
    return mode


def _model_files(name):
    """The files ``_save_fit`` writes for the model ``models/<name>``."""
    return tuple(
        f"models/{name}/{f}"
        for f in ("manifest.json", "values.npz", "info.json")
    )


# === CONTEXT: THE RESOLUTION SCALE AND THE DEFAULTS THAT FOLLOW ===


@dataclasses.dataclass(frozen=True)
class _Context:
    """Numbers every stage needs, from the overview report and settings."""

    halffringe: float
    r_in: float
    r_out: float
    n_ring: int
    n_gp: int
    wavel0: float
    colour: bool
    law: str
    degree: int
    star_teff: float | None
    companion: bool
    max_sep: float
    step: float
    inc_max: float
    ells: tuple
    star_diam: float | None = None
    star_index: float | None = None
    ell_floor: float = 0.0

    @classmethod
    def from_report(cls, settings, overview):
        return cls(
            halffringe=overview["resolution_mas"],
            r_in=overview["r_in_mas"],
            r_out=overview["r_out_mas"],
            n_ring=overview["n_r_rings"],
            n_gp=overview["n_r_gp"],
            wavel0=overview["wavel0_m"],
            colour=overview["colour"],
            law=settings["ring_colour"],
            degree=int(settings["harmonic_degree"]),
            star_teff=settings["star_temperature_k"],
            star_diam=settings["star_diameter_mas"],
            star_index=settings["star_index"],
            companion=bool(settings["companion"]),
            max_sep=overview["companion_max_sep_mas"],
            step=overview["companion_step_mas"],
            inc_max=float(settings["inc_max_deg"]),
            ells=tuple(overview["ell_mas"]),
            ell_floor=overview["ell_floor_mas"],
        )

    @property
    def ell_mid(self):
        return self.ells[len(self.ells) // 2]


# === STAGES: LOAD AND OVERVIEW ===


def _load_disk(p):  # deliberate: stage callable, patched by tests
    p._error_factor = 1.0  # the data as given, whatever an old run found
    return _load_report(p)


def _overview(p):
    p._error_factor = 1.0
    from ..imaging import image_field
    from ..models import GaussianRing, MultipoleDisk, RadialGP

    s, data = p.settings, p.processed
    _overview_plots(p, data)

    halffringe = _geometry(data)["resolution_mas"]
    # The hole is the larger of the half-fringe and the star's radius:
    # compact flux piles up just outside a hole edge that is smaller.
    star_radius = 0.5 * float(s["star_diameter_mas"] or 0.0)
    r_in = float(s["r_in_mas"] or max(halffringe, star_radius))
    r_out = float(s["r_out_mas"] or 0.5 * image_field(data))
    if not r_out > r_in:
        raise ValueError(
            f"The default outer radius {r_out:.3g} mas does not exceed the "
            f"inner radius {r_in:.3g} mas; give r_out_mas."
        )
    ell = [_ELL_PER_EQUIVALENT * k * halffringe for k in s["ell_halffringes"]]
    if s["n_r"] is not None:
        n_rings = n_gp = int(s["n_r"])
    else:
        ring = MultipoleDisk(
            GaussianRing(0.5 * r_out, halffringe), inc=40.0, pa=0.0
        ).suggest_n_r(data)
        gp = MultipoleDisk(RadialGP(r_in, r_out), n_r=16).suggest_n_r(data)
        # Four nodes per length scale resolve the shortest ℓ.
        per_ell = int(8 * math.ceil(4.0 * (r_out - r_in) / min(ell) / 8.0))
        n_rings, n_gp = int(ring), int(min(max(gp, per_ell, 16), 96))
    channels = int(
        onp.unique(onp.round(onp.asarray(data.wavel, dtype=float), 12)).size
    )
    wavel0 = (
        float(s["wavel0_m"])
        if s["wavel0_m"] is not None
        else float(onp.mean(onp.unique(onp.asarray(data.wavel, dtype=float))))
    )
    colour = (
        channels > s["spectral_index_min_channels"]
        if s["spectral_index"] is None
        else bool(s["spectral_index"])
    )
    return {
        "resolution_mas": halffringe,
        "r_in_mas": r_in,
        "r_out_mas": r_out,
        "n_r_rings": n_rings,
        "n_r_gp": n_gp,
        "n_channels": channels,
        "colour": bool(colour),
        "wavel0_m": wavel0,
        "ell_mas": ell,
        "ell_floor_mas": _ELL_PER_EQUIVALENT
        * s["ell_floor_halffringes"]
        * halffringe,
        "star": _star_report(s),
        "companion_max_sep_mas": float(s["companion_max_sep_mas"] or r_out),
        "companion_step_mas": float(
            s["companion_step_mas"] or 0.5 * halffringe
        ),
    }


# === SCENES, PRIORS AND FITS ===


@dataclasses.dataclass(frozen=True)
class _Shape:
    """What a ring scene contains: rings, colour, harmonics and a planet."""

    n: int
    colour: bool
    degree: int
    planet: bool


def _star(ctx):
    """The star: a point source or uniform disk, grey unless given a spectrum."""
    from ..models import PointSource, UniformDisk
    from ..spectra import BlackBody, PowerLaw

    flux = 1.0
    if ctx.star_teff is not None:
        flux = BlackBody(1.0, ctx.star_teff, ctx.wavel0)
    elif ctx.star_index is not None:
        flux = PowerLaw(1.0, ctx.star_index, ctx.wavel0)
    if ctx.star_diam is not None:
        return UniformDisk(diam=ctx.star_diam, flux=flux)
    return PointSource(flux=flux)


# deliberate: summary entry for the star spectrum, checked by test_star_is_a_point_source_until_given_a_size_or_a_spectrum without a run
def _star_report(settings):
    """What the star is, for the summary: the disk's colour is relative to it."""
    if settings["star_temperature_k"] is not None:
        spectrum = f"blackbody {settings['star_temperature_k']:g} K"
    elif settings["star_index"] is not None:
        spectrum = f"power law, index {settings['star_index']:g}"
    else:
        spectrum = "grey"
    return {
        "shape": (
            "uniform disk"
            if settings["star_diameter_mas"] is not None
            else "point source"
        ),
        "diameter_mas": settings["star_diameter_mas"],
        "spectrum": spectrum,
    }


def _flux(ctx, ratio, colour_value, colour):
    """A source's flux: ``ratio``, or with colour a power law of index
    ``colour_value`` or a blackbody of that temperature (``ring_colour``)."""
    from ..spectra import BlackBody, PowerLaw

    if not colour:
        return ratio
    if ctx.law == "blackbody":
        return BlackBody(ratio, colour_value, ctx.wavel0)
    return PowerLaw(ratio, colour_value, ctx.wavel0)


def _colour_start(ctx):
    """The starting value of a colour parameter."""
    return _TEMPERATURE_START if ctx.law == "blackbody" else 0.0


def _colour_prior(ctx):
    import numpyro.distributions as dist

    if ctx.law == "blackbody":
        return dist.LogUniform(*_TEMPERATURE_RANGE)
    return dist.Uniform(*_INDEX_RANGE)


def _planet(ctx, v, colour):
    from ..models import PointSource

    return PointSource(
        dra=v["dra"],
        ddec=v["ddec"],
        flux=_flux(ctx, v["fp"], v["ap"] if colour else 0.0, colour),
    )


def _ring_scene(ctx, shape, v):
    """Star, ``shape.n`` coplanar rings (one ``inc``, ``pa``) and optionally a planet."""
    from ..models import GaussianRing, Harmonics, MultipoleDisk, System

    parts = {"star": _star(ctx)}
    for i in range(1, shape.n + 1):
        harmonics = None
        if shape.degree:
            harmonics = Harmonics(
                re=tuple(v[f"h{i}_re{j}"] for j in range(1, shape.degree + 1)),
                im=tuple(v[f"h{i}_im{j}"] for j in range(1, shape.degree + 1)),
            )
        parts[f"ring{i}"] = MultipoleDisk(
            GaussianRing(v[f"r{i}"], v[f"s{i}"]),
            inc=v["inc"],
            pa=v["pa"],
            n_r=ctx.n_ring,
            flux=_flux(
                ctx,
                v[f"f{i}"],
                v[f"a{i}"] if shape.colour else 0.0,
                shape.colour,
            ),
            harmonics=harmonics,
        )
    if shape.planet:
        parts["planet"] = _planet(ctx, v, shape.colour)
    return System(**parts)


@functools.lru_cache(maxsize=None)
def _builder(ctx, shape):
    """The scene as a function of its parameters (``fit`` calls it by name)."""

    # cached: ``fit`` and ``laplace_cov`` recompile for every new function object
    def build(**v):
        return _ring_scene(ctx, shape, v)

    return build


def _geometry_priors(ctx):
    from ..angles import AngleVector
    from ..priors import IsotropicInclination

    return {
        "inc": IsotropicInclination(0.0, ctx.inc_max),
        "pa": AngleVector(),
    }


def _planet_priors(ctx, colour):
    import numpyro.distributions as dist

    priors = {
        "dra": dist.Uniform(-ctx.max_sep, ctx.max_sep),
        "ddec": dist.Uniform(-ctx.max_sep, ctx.max_sep),
        "fp": dist.LogUniform(*_PLANET_FLUX_RANGE),
    }
    if colour:
        priors["ap"] = _colour_prior(ctx)
    return priors


def _ring_priors(ctx, shape):
    import numpyro.distributions as dist

    priors = _geometry_priors(ctx)
    for i in range(1, shape.n + 1):
        priors[f"r{i}"] = dist.LogUniform(ctx.r_in, ctx.r_out)
        priors[f"s{i}"] = dist.LogUniform(
            _WIDTH_FLOOR * ctx.halffringe, 0.5 * (ctx.r_out - ctx.r_in)
        )
        priors[f"f{i}"] = dist.LogUniform(*_FLUX_RANGE)
        if shape.colour:
            priors[f"a{i}"] = _colour_prior(ctx)
        for j in range(1, shape.degree + 1):
            priors[f"h{i}_re{j}"] = dist.Normal(0.0, _HARMONIC_SCALE)
            priors[f"h{i}_im{j}"] = dist.Normal(0.0, _HARMONIC_SCALE)
    if shape.planet:
        priors.update(_planet_priors(ctx, shape.colour))
    return priors


def _scalars(values):
    """Fitted values as floats (the angle's vector and noise terms apart)."""
    return {
        k: float(onp.asarray(v))
        for k, v in values.items()
        if not k.endswith("_vec") and onp.size(v) == 1
    }


@dataclasses.dataclass(frozen=True)
class _DiskFit:
    """A fitted model, its quoted-error χ² and the −2 log L ``score`` the BIC uses."""

    model: object
    values: dict
    info: dict
    chi2: float
    scales: dict
    n_free: int
    score: float | None = None


def _finish(model, values, info, priors, data):
    """A ``_DiskFit`` from a ``fit`` result's pieces."""
    chi2, _ = _chi2_64(model, data)
    scales = _scales_of(values)
    n_free = sum(onp.size(values[k]) for k in priors if k in values)
    n_free += sum(1 for k in values if k.startswith("noise."))
    return _DiskFit(
        model=model,
        values={k: onp.asarray(v) for k, v in values.items()},
        info=_io.clean_json(dict(info)),
        chi2=float(chi2),
        scales=scales,
        n_free=int(n_free),
        # at the fitted error scales: quoted-error χ² rewards each extra ring
        score=_m2loglike_64(model, data, **scales) if scales else None,
    )


# deliberate: pulls only out-of-support starts; pinned by test_inside_priors_pulls_only_values_outside_a_bound on bare priors
def _inside_priors(priors, start):
    """``start`` with any value outside its prior's support pulled just inside."""
    from .._flat import _base
    from ..fitting import _bound_fraction

    # a start outside the support has no unconstrained value: the fit gives NaN
    outside = {}
    for name, value in start.items():
        prior = priors.get(name)
        if prior is None or _bound_fraction(prior, value) is None:
            continue  # no prior, or none with finite edges
        if not onp.all(_base(prior).support(onp.asarray(value, float))):
            outside[name] = value
    return {**start, **_interior_start(priors, outside)}


def _fit_rings(p, shape, init):
    """MAP fit of a ring scene from ``init``: a ``_DiskFit``."""
    from ..fitting import fit

    ctx, data = p._ctx(), p.processed
    priors = _ring_priors(ctx, shape)
    result = fit(
        _builder(ctx, shape),
        priors,
        data,
        noise=p._noise(),
        init=_inside_priors(priors, init),
    )
    model, values = _fold_fit(ctx, shape, priors, result.model, result.values)
    return _finish(model, values, result.info, priors, data)


def _fold_fit(ctx, shape, priors, model, values):
    """The fit with ``pa`` folded into [0°, 180°), the harmonics turned to match."""
    pa = float(onp.asarray(values["pa"]))
    if 0.0 <= pa < 180.0:
        return model, values
    out = dict(values)
    folded = pa % 180.0
    out["pa"] = onp.asarray(folded)
    if "pa_vec" in out:
        angle = math.radians(folded)
        out["pa_vec"] = onp.asarray([math.cos(angle), math.sin(angle)])
    if (pa % 360.0) >= 180.0:
        # pa + 180° is the same disk once order-m harmonics get (-1)^m
        for i in range(1, shape.n + 1):
            for j in range(1, shape.degree + 1):
                for part in ("re", "im"):
                    key = f"h{i}_{part}{j}"
                    out[key] = onp.asarray(out[key]) * (-1.0) ** j
    scene = _builder(ctx, shape)(
        **{k: onp.asarray(out[k]) for k in priors if k in out}
    )
    return scene, out


# === STAGE: GEOMETRY ===


# deliberate: module-level, bound with functools.partial, so the jitted grid
# kernel's cache key (a pytree of the function and the hashable ctx) is stable
# across calls; a closure is a new object each call and recompiles.
def _single_ring(ctx, *, cos_inc, pa, r1, f1):
    """One ring of width one half-fringe, for the geometry grid."""
    import jax.numpy as np

    v = {
        "inc": np.degrees(np.arccos(cos_inc)),
        "pa": pa,
        "r1": r1,
        "s1": ctx.halffringe,
        "f1": f1,
    }
    return _ring_scene(ctx, _Shape(1, False, 0, False), v)


@in_float64
# deliberate: float64 boundary (in_float64) around the jitted grid
def _geometry_grid(p, ctx, cos_axis, pa_axis, r_axis):
    import jax.numpy as np

    from ..grid_fit import likelihood_grid

    return onp.asarray(
        likelihood_grid(
            functools.partial(_single_ring, ctx),
            data=p.processed,
            grid={
                "cos_inc": np.asarray(cos_axis),
                "pa": np.asarray(pa_axis),
                "r1": np.asarray(r_axis),
                "f1": np.asarray(_FLUX_GRID),
            },
            batch_size=p.settings["batch_size"],
        )
    )


# deliberate: periodic-pa local-minimum search on a bare array; pinned by test_grid_minima_wrap_the_position_angle
def _grid_minima(delta, inc_deg):
    """Local minima of the geometry map ``delta[inc, pa]`` (``pa`` is periodic)."""
    n_inc, n_pa = delta.shape
    minima = []
    for i in range(n_inc):
        if inc_deg[i] < 5.0:
            continue  # face-on: pa undefined, a flat ridge
        for j in range(n_pa):
            value = delta[i, j]
            around = [
                delta[a, (j + b) % n_pa]
                for a in (i - 1, i, i + 1)
                for b in (-1, 0, 1)
                if (a, b) != (i, 0) and 0 <= a < n_inc
            ]
            if all(value <= v for v in around):
                minima.append((float(value), i, j))
    return sorted(minima)


# deliberate: clipped GP sigma formula; pinned at both clip ends by test_dynamic_range_sigma_is_clipped
def _dynamic_range_sigma(ctx, r0, sigma):
    """The Gaussian process's σ: ln of the single ring's peak-to-mean contrast."""
    contrast = (ctx.r_out**2 - ctx.r_in**2) / (
        2.0 * math.sqrt(2.0 * math.pi) * sigma * r0
    )
    lo, hi = _GP_SIGMA_BOUNDS
    return float(min(max(math.log(max(contrast, 1.0)), lo), hi))


def _geometry_stage(p):  # deliberate: stage callable, patched by tests
    p._error_factor = 1.0  # the geometry is found on the quoted errors
    import matplotlib.pyplot as plt

    ctx, data = p._ctx(), p.processed
    s = p.settings
    inc_max = math.radians(ctx.inc_max)
    cos_axis = onp.linspace(1.0, math.cos(inc_max), s["n_inc"])
    pa_axis = onp.linspace(0.0, 180.0, s["n_pa"], endpoint=False)
    r_axis = onp.arange(ctx.r_in, ctx.r_out + 1e-9, ctx.halffringe)
    if r_axis.size < 2:
        r_axis = onp.linspace(ctx.r_in, ctx.r_out, 2)

    ll = _geometry_grid(p, ctx, cos_axis, pa_axis, r_axis)
    ll = onp.where(onp.isfinite(ll), ll, -onp.inf)
    cells = ll.reshape(ll.shape[:2] + (-1,))
    best_cell = cells.max(axis=-1)
    flat = cells.argmax(axis=-1)
    delta = 2.0 * (best_cell.max() - best_cell)
    i, j = onp.unravel_index(int(onp.argmax(best_cell)), best_cell.shape)
    k_r, k_f = onp.unravel_index(
        int(flat[i, j]), (r_axis.size, _FLUX_GRID.size)
    )
    inc_deg = onp.degrees(onp.arccos(cos_axis))
    minima = _grid_minima(delta, inc_deg)
    rivals = [m for m in minima[1:] if m[0] < _checks.RIVAL_DELTA_CHI2]

    _io.write_h5(
        p.output / "grids.h5",
        {
            "axes": {
                "cos_inc": (cos_axis, {"units": ""}),
                "pa": (pa_axis, {"units": "deg"}),
            },
            "geometry": {
                "delta_chi2": delta,
                "r1_mas": r_axis[flat // _FLUX_GRID.size],
            },
        },
        attrs={
            "/": {"axis_order": ["cos_inc", "pa"], "schema": _io.SCHEMA},
            "axes": {"inc_deg": inc_deg},
        },
    )
    fig, ax = plt.subplots(figsize=(6, 4.5))
    shown = onp.clip(delta, 0.0, onp.nanmin(delta) + 400.0)
    mesh = ax.pcolormesh(
        pa_axis, inc_deg, shown, shading="auto", cmap="viridis_r"
    )
    fig.colorbar(mesh, ax=ax, label="Δχ² (clipped)")
    ax.set(xlabel="position angle (deg)", ylabel="inclination (deg)")
    ax.set_title("Single-ring geometry grid")
    _save(fig, p.output / "plots" / "geometry_grid.png")

    shape = _Shape(1, False, 0, False)
    start = {
        "inc": float(inc_deg[i]),
        "pa": float(pa_axis[j]),
        "r1": float(r_axis[k_r]),
        "s1": ctx.halffringe,
        "f1": float(_FLUX_GRID[k_f]),
    }
    # Face-on has no position angle: start a little away from it.
    start["inc"] = max(start["inc"], 5.0)
    fit = _fit_rings(p, shape, start)
    values = _scalars(fit.values)
    names = list(_ring_priors(ctx, shape))
    sd, _ = _laplace(ctx, shape, names, values, data)
    n = int(data.n_independent)
    _save_fit(p.output / "models" / "geometry", fit, names)
    return {
        "inc_deg": values["inc"],
        "pa_deg": values["pa"],
        "r_mas": values["r1"],
        "sigma_mas": values["s1"],
        "flux": values["f1"],
        "chi2": fit.chi2,
        "chi2_reduced": fit.chi2 / n,
        "n_independent": n,
        "converged": fit.info.get("converged"),
        "laplace_sd": sd,
        "gp_sigma": _dynamic_range_sigma(ctx, values["r1"], values["s1"]),
        "grid": {
            "shape": list(delta.shape),
            "best_index": [int(i), int(j)],
            "n_minima": 1 + len(rivals),
            "rivals": [
                {
                    "inc_deg": float(inc_deg[a]),
                    "pa_deg": float(pa_axis[b]),
                    "delta_chi2": d,
                }
                for d, a, b in rivals
            ],
        },
    }


def _laplace(ctx, shape, names, values, data):
    """``(sd, cov)``: Laplace standard deviations of the ring scene's scalar
    parameters and their covariance.

    A parameter whose variance is not positive and finite (the Hessian is
    not positive definite there: the point is not a likelihood maximum, or
    the parameter is unconstrained) has ``sd`` ``None``, never 0, and
    ``cov`` is ``None`` unless the whole matrix is positive definite, since
    a Laplace evidence needs that. A failed computation gives all ``None``.
    """
    from ..inference import laplace_cov

    try:
        x = onp.array([values[k] for k in names], dtype=float)
        cov = onp.asarray(laplace_cov(x, names, _builder(ctx, shape), data))
    except (FloatingPointError, onp.linalg.LinAlgError) as err:
        warnings.warn(
            f"The Laplace covariance could not be computed ({err}).",
            RuntimeWarning,
            stacklevel=2,
        )
        return {k: None for k in names}, None
    var = onp.diag(cov)
    good = onp.isfinite(var) & (var > 0.0)
    pd = bool(
        onp.all(onp.isfinite(cov))
        and onp.all(onp.linalg.eigvalsh(0.5 * (cov + cov.T)) > 0.0)
    )
    sd = {
        k: float(onp.sqrt(v)) if g else None
        for k, v, g in zip(names, var, good)
    }
    return sd, (cov if pd else None)


# === POINT-SOURCE SEARCH ON RESIDUALS ===


@in_float64
def _point_source_search(model, data, ctx, batch_size=None):
    """The best grey point source to add to ``model``: Δχ², position and flux."""
    import jax.numpy as np

    from .._grid import batch_size_or_default, resolve_grid_keys
    from ..grid_fit import _optimize_flux_grid
    from ..likelihood import model_loglike
    from ..models import PointSource, System

    probe = System({**model.components, "probe": PointSource(flux=0.01)})
    steps = int(math.floor(ctx.max_sep / ctx.step))
    axis = ctx.step * onp.arange(-steps, steps + 1)
    grid = {
        "probe.dra": np.asarray(axis),
        "probe.ddec": np.asarray(axis),
        "probe.flux": np.asarray(_SEARCH_FLUX_GRID),
    }
    params, coord_keys, flux_key = resolve_grid_keys(grid, "probe.flux")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # unconverged cells lose, not warn
        flux, loglike, _ = _optimize_flux_grid(
            data,
            probe,
            grid,
            params=params,
            coord_keys=coord_keys,
            flux_key=flux_key,
            batch_size=batch_size_or_default(batch_size, data),
        )
    flux, loglike = onp.asarray(flux), onp.asarray(loglike)
    null = float(model_loglike(model, data))
    delta = onp.where(
        (flux > 0) & onp.isfinite(loglike),
        onp.maximum(2.0 * (loglike - null), 0.0),
        0.0,
    )
    a, b = onp.unravel_index(int(onp.argmax(delta)), delta.shape)
    return {
        "delta_chi2": float(delta[a, b]),
        "dra_mas": float(axis[a]),
        "ddec_mas": float(axis[b]),
        "flux": float(flux[a, b]),
    }


# === STAGE: PROFILE ===


def _profile(p):
    return models()[p.name].profile(p)


def _geometry_values(p):
    """The single-ring fit of the ``geometry`` stage, as floats."""
    return _scalars(_io.load_model_values(p.output / "models" / "geometry"))


def _planet_start(search, ctx):
    start = {
        "dra": search["dra_mas"],
        "ddec": search["ddec_mas"],
        "fp": min(max(search["flux"], _PLANET_FLUX_RANGE[0] * 2), 0.5),
    }
    if ctx.colour:
        start["ap"] = _colour_start(ctx)
    return start


def _ring_start(ctx, held, n, new):
    """Starting values of an ``n``-ring scene from ``held`` and the scan's ``new`` ring."""
    start = {k: v for k, v in held.items() if k != "pa_vec"}
    start["inc"] = max(float(start["inc"]), 5.0)
    if new is not None:
        start[f"r{n}"], start[f"s{n}"], start[f"f{n}"] = (
            new["r"],
            ctx.halffringe,
            new["f"],
        )
    for ring in range(1, n + 1):
        if ctx.colour:
            start.setdefault(f"a{ring}", _colour_start(ctx))
        for j in range(1, ctx.degree + 1):
            start.setdefault(f"h{ring}_re{j}", 0.0)
            start.setdefault(f"h{ring}_im{j}", 0.0)
    return start


# Module-level so functools.partial(_scan_model, ...) gives a stable jit cache key.
def _scan_model(ctx, shape, base, *, rnew, fnew):
    v = dict(base)
    i = shape.n
    v[f"r{i}"], v[f"s{i}"], v[f"f{i}"] = rnew, ctx.halffringe, fnew
    if shape.colour:
        v[f"a{i}"] = _colour_start(ctx)
    for j in range(1, shape.degree + 1):
        v[f"h{i}_re{j}"] = 0.0
        v[f"h{i}_im{j}"] = 0.0
    return _ring_scene(ctx, shape, v)


@in_float64
def _scan_new_ring(p, ctx, shape, base, step=None):
    """Radius and flux of one more ring, scanned ``step`` apart (default a half-fringe)."""
    import jax.numpy as np

    from ..grid_fit import likelihood_grid

    r_axis = onp.arange(ctx.r_in, ctx.r_out + 1e-9, step or ctx.halffringe)
    ll = onp.asarray(
        likelihood_grid(
            functools.partial(_scan_model, ctx, shape, base),
            data=p.processed,
            grid={"rnew": np.asarray(r_axis), "fnew": np.asarray(_FLUX_GRID)},
            batch_size=p.settings["batch_size"],
        )
    )
    ll = onp.where(onp.isfinite(ll), ll, -onp.inf)
    a, b = onp.unravel_index(int(onp.argmax(ll)), ll.shape)
    return {"r": float(r_axis[a]), "f": float(_FLUX_GRID[b])}


def _bic(fit, n):
    score = fit.chi2 if fit.score is None else fit.score
    return score + fit.n_free * math.log(n)


# deliberate: ring fit plus planet search; test_rings_leave_out_an_unresolved_count_and_go_on wraps it to record the ring counts tried
def _fit_with_planet(p, shape, start, backfit=False):
    """Fit ``shape``, search the residuals, refit with a planet if found.

    Returns ``(fit, shape, search, steps)``; ``backfit`` adds ``_backfit`` steps.
    """
    ctx = p._ctx()
    fit = _fit_rings(p, shape, start)
    search = None
    if ctx.companion:
        search = _point_source_search(
            fit.model, p.processed, ctx, p.settings["batch_size"]
        )
        if search["delta_chi2"] >= p.settings["companion_min_delta_chi2"]:
            shape = dataclasses.replace(shape, planet=True)
            values = {**_scalars(fit.values), **_planet_start(search, ctx)}
            fit = _fit_rings(p, shape, values)
    steps = []
    if backfit:
        # after the planet, so the back-fit objective is the one reported
        fit, steps = _backfit(p, shape, fit)
    return fit, shape, search, steps


_RING_KEY = re.compile(r"^(?:[rsfa]|h)(\d+)(?:_(?:re|im)\d+)?$")


def _without_ring(values, k):
    """``values`` of a ring scene without ring ``k``, the later rings
    renumbered down (geometry, planet and noise terms kept)."""
    out = {}
    for key, v in values.items():
        m = _RING_KEY.match(key)
        if m is None:
            out[key] = v
            continue
        i = int(m.group(1))
        if i == k:
            continue
        j = i - (i > k)
        out[key[: m.start(1)] + str(j) + key[m.end(1) :]] = v
    return out


def _unresolved_pairs(ctx, values, n):
    """Pairs of the ``n`` rings closer than the data can tell apart: ``(i, j, sep, limit)``."""
    pairs = []
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            sep = abs(float(values[f"r{i}"]) - float(values[f"r{j}"]))
            # unresolved within 2σ of the narrower ring (equal Gaussians are
            # unimodal there) or within a half-fringe
            limit = max(
                2.0 * min(float(values[f"s{i}"]), float(values[f"s{j}"])),
                ctx.halffringe,
            )
            if sep < limit:
                pairs.append((i, j, sep, limit))
    return pairs


def _backfit(p, shape, fit):
    """Re-place each ring in turn and keep what lowers the BIC: ``(fit, steps)``."""
    ctx, n_data = p._ctx(), int(p.processed.n_independent)
    steps = []
    if shape.n < 2:
        return fit, steps
    for sweep in range(1, _BACKFIT_PASSES + 1):
        improved = False
        for k in range(1, shape.n + 1):
            held = _without_ring(_scalars(fit.values), k)
            base = _fill_new(ctx, held, shape.n)
            # a grid twice as fine as the first scan's
            new = _scan_new_ring(
                p, ctx, shape, base, step=0.5 * ctx.halffringe
            )
            start = _ring_start(ctx, held, shape.n, new)
            trial = _fit_rings(p, shape, start)
            d_bic = _bic(trial, n_data) - _bic(fit, n_data)
            pairs = _unresolved_pairs(ctx, trial.values, shape.n)
            # accept only if no more rings end up unresolved than before
            accepted = d_bic < -_BACKFIT_MIN_GAIN and len(pairs) <= len(
                _unresolved_pairs(ctx, fit.values, shape.n)
            )
            steps.append(
                {
                    "pass": sweep,
                    "ring": k,
                    "dropped_r": float(fit.values[f"r{k}"]),
                    "scan_r": new["r"],
                    "scan_f": new["f"],
                    "chi2_reduced": trial.chi2 / n_data,
                    "inc": float(trial.values["inc"]),
                    "pa": float(trial.values["pa"]),
                    "radii": [
                        float(trial.values[f"r{i}"])
                        for i in range(1, shape.n + 1)
                    ],
                    "delta_bic": d_bic,
                    "n_unresolved": len(pairs),
                    "converged": trial.info.get("converged"),
                    "accepted": accepted,
                }
            )
            if accepted:
                fit, improved = trial, True
        if not improved:
            break
    return fit, steps


# deliberate: greedy BIC ring pruning; test_prune_drops_a_ring_that_carries_no_light drives it on a padded fit
def _prune(p, shape, fit):
    """Drop the ring whose removal lowers the BIC most, repeatedly: ``(fit, shape, dropped)``."""
    ctx, n_data = p._ctx(), int(p.processed.n_independent)
    dropped = []
    while shape.n > 1:
        # a greedy fit can leave a ring carrying no light, which only costs parameters
        smaller = dataclasses.replace(shape, n=shape.n - 1)
        best = None
        for k in range(1, shape.n + 1):
            held = _without_ring(_scalars(fit.values), k)
            start = _ring_start(ctx, held, smaller.n, None)
            trial = _fit_rings(p, smaller, start)
            d_bic = _bic(trial, n_data) - _bic(fit, n_data)
            if d_bic < 0 and (best is None or d_bic < best[2]):
                best = (k, trial, d_bic)
        if best is None:
            break
        k, trial, d_bic = best
        dropped.append(
            {
                "r": float(fit.values[f"r{k}"]),
                "s": float(fit.values[f"s{k}"]),
                "f": float(fit.values[f"f{k}"]),
                "delta_bic": d_bic,
            }
        )
        fit, shape = trial, smaller
    return fit, shape, dropped


# deliberate: callback: the profile stage of the "rings" entry of MODELS
def _profile_rings(p):
    s, ctx, data = p.settings, p._ctx(), p.processed
    n_data = int(data.n_independent)
    geo = _geometry_values(p)
    candidates, searches = [], []
    previous = {k: geo[k] for k in ("inc", "pa", "r1", "s1", "f1")}
    for n in range(1, s["max_rings"] + 1):
        # Each count is fitted on its own without a companion, searched for
        # one, and refitted with it if one is found: a companion found with
        # too few rings (the next ring's light) does not carry over.
        shape = _Shape(n, ctx.colour, ctx.degree, False)
        held = {
            k: v
            for k, v in previous.items()
            if not k.startswith(("dra", "ddec", "fp", "ap"))
        }
        if n == 1:
            start = _ring_start(ctx, held, 1, None)
        else:
            new = _scan_new_ring(p, ctx, shape, _fill_new(ctx, held, n))
            start = _ring_start(ctx, held, n, new)
        fit, shape, search, steps = _fit_with_planet(
            p, shape, start, backfit=True
        )
        if search is not None:
            searches.append({"rings": n, **search})
        fit, shape, dropped = _prune(p, shape, fit)
        previous = _scalars(fit.values)
        # A ring that had to be dropped was one too many: the rings are used
        # up. The pruned fit is the candidate for its own count (and replaces
        # that count's earlier fit if it beats it).
        count = shape.n
        name = f"rings_{count}"
        unresolved = _unresolved_pairs(ctx, fit.values, count)
        earlier = next((c for c in candidates if c["n"] == count), None)
        if earlier is not None and earlier["bic"] <= _bic(fit, n_data):
            break
        candidates = [c for c in candidates if c["n"] != count]
        _save_fit(
            p.output / "models" / name, fit, list(_ring_priors(ctx, shape))
        )
        candidates.append(
            {
                "n": count,
                "name": name,
                "chi2": fit.chi2,
                "chi2_reduced": fit.chi2 / n_data,
                "n_free": fit.n_free,
                "bic": _bic(fit, n_data),
                "converged": fit.info.get("converged"),
                "planet": shape.planet,
                "unresolved": [list(pair) for pair in unresolved],
                "pruned": dropped,
                "backfit": steps,
            }
        )
        candidates.sort(key=lambda c: c["n"])
        if dropped:
            break
        # A count with two rings too close to tell apart is one ring split
        # in two (a profile that is not Gaussian) or a ring still to be
        # found; `compare` leaves it out, and the search goes on from it. It
        # stops when a count does not beat the last resolved one.
        resolved = [c for c in candidates[:-1] if not c["unresolved"]]
        if (
            not unresolved
            and resolved
            and candidates[-1]["bic"] >= resolved[-1]["bic"]
        ):
            break
    _plot_profile_fits(p, candidates, "n", "rings")
    return {"candidates": candidates, "searches": searches}


def _fill_new(ctx, held, n):
    """``held`` values with placeholders for ring ``n`` (the scan replaces them)."""
    v = dict(held)
    v[f"r{n}"], v[f"s{n}"], v[f"f{n}"] = ctx.r_in, ctx.halffringe, 0.01
    for ring in range(1, n + 1):
        if ctx.colour:
            v.setdefault(f"a{ring}", _colour_start(ctx))
        for j in range(1, ctx.degree + 1):
            v.setdefault(f"h{ring}_re{j}", 0.0)
            v.setdefault(f"h{ring}_im{j}", 0.0)
    return v


def _plot_profile_fits(p, candidates, key, label):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 3.5))
    x = [c[key] for c in candidates]
    ax.plot(x, [c["chi2_reduced"] for c in candidates], "o-")
    ax.axhline(1.0, color="0.6", lw=0.8)
    ax.set(xlabel=label, ylabel="χ²/N (quoted errors)", yscale="log")
    _save(fig, p.output / "plots" / "profile_fits.png")


# --- the Gaussian-process disk ---


def _gp_template(ctx, geo, ell, planet, degree=None):
    """A star, a ``RadialGP`` disk and optionally a planet, with its priors."""
    import numpyro.distributions as dist

    from ..models import (
        Harmonics,
        MultipoleDisk,
        MultipoleKernel,
        RadialGP,
        RadialGPFunction,
        System,
        gp_priors,
    )

    degree = ctx.degree if degree is None else degree
    r_in, r_out = ctx.r_in, ctx.r_out
    harmonics = None
    if degree:
        harmonics = Harmonics(
            degree=degree,
            prior=MultipoleKernel(
                ell_r=ell,
                ell_theta=0.5 * (r_in + r_out),
                sigma=_HARMONIC_SIGMA,
                n=_HARMONIC_NODES,
                r_in=r_in,
                r_out=r_out,
            ),
        )
    index = (
        RadialGPFunction(
            r_in, r_out, sigma=_ALPHA_SIGMA, ell=ell, n=_ALPHA_NODES
        )
        if ctx.colour
        else None
    )
    disk = MultipoleDisk(
        RadialGP(r_in, r_out, sigma=geo["gp_sigma"], ell=ell),
        harmonics=harmonics,
        spectral_index=index,
        wavel0=ctx.wavel0,
        inc=geo["inc"],
        pa=geo["pa"],
        n_r=ctx.n_gp,
        flux=geo["f1"],
    )
    parts = {"star": _star(ctx), "disk": disk}
    if planet is not None:
        parts["planet"] = _planet(ctx, planet, ctx.colour)
    model = System(**parts)
    priors = gp_priors(model) | {"disk.flux": dist.LogUniform(*_FLUX_RANGE)}
    if planet is not None:
        paths = {
            "dra": "planet.dra",
            "ddec": "planet.ddec",
            "fp": "planet.flux.ratio" if ctx.colour else "planet.flux",
            "ap": "planet.flux.index",
        }
        priors |= {
            paths[k]: prior
            for k, prior in _planet_priors(ctx, ctx.colour).items()
        }
    return model, priors


def _apply(template, values, priors):
    """The unfrozen ``template`` with the fitted values of ``priors`` set."""
    paths = [k for k in priors if k in values]
    return template.set(paths, [values[k] for k in paths])


def _gp_geometry(p):
    geo = _geometry_values(p)
    report = p._report("geometry")
    return {**geo, "gp_sigma": report["gp_sigma"]}


def _released_geometry(p, values):
    """``_gp_geometry`` with the inclination and position angle of ``values``."""
    return {
        **_gp_geometry(p),
        "inc": float(onp.asarray(values["disk.inc"])),
        "pa": float(onp.asarray(values["disk.pa"])),
    }


def _planet_values(values, colour):
    """A fit's planet as the pipeline's ``dra, ddec, fp, ap`` values."""
    out = {
        "dra": float(values["planet.dra"]),
        "ddec": float(values["planet.ddec"]),
        "fp": float(values["planet.flux.ratio" if colour else "planet.flux"]),
    }
    if colour:
        out["ap"] = float(values["planet.flux.index"])
    return out


def _companion_report(p, planet):
    """The fitted companion (``dra, ddec, fp[, ap]``) for the summary, or None."""
    if planet is None:
        return None
    searches = p._report("profile")["searches"]
    found = [
        x["delta_chi2"]
        for x in searches
        if x["delta_chi2"] >= p.settings["companion_min_delta_chi2"]
    ]
    return {
        "dra_mas": planet["dra"],
        "ddec_mas": planet["ddec"],
        "flux": planet["fp"],
        "spectral_index": planet.get("ap"),
        "detection_delta_chi2": max(found) if found else None,
    }


def _gp_fit(p, ell, planet, geo, degree=None, release=False, max_steps=None):
    """MAP fit of the Gaussian-process disk at one ℓ, geometry frozen at ``geo``.

    ``release`` refits with the geometry free; ``max_steps`` bounds the frozen fit.
    """
    import equinox as eqx

    from ..fitting import fit

    ctx, data = p._ctx(), p.processed
    template, priors = _gp_template(ctx, geo, ell, planet, degree)
    # frozen first (fast): the single ring's geometry is a few degrees off
    frozen = eqx.tree_at(
        lambda m: m.parts[1], template, template.disk.freeze_geometry(data)
    )
    result = fit(frozen, priors, data, max_steps=max_steps)
    values, info = result.values, result.info
    if release:
        priors = {
            **priors,
            **{f"disk.{k}": v for k, v in _geometry_priors(ctx).items()},
        }
        start = {k: v for k, v in values.items() if k in priors}
        start["disk.inc"] = max(float(geo["inc"]), 5.0)
        result = fit(template, priors, data, init=start)
        values = _fold_gp_values(ctx, result.values, degree)
        info = result.info
    model = _apply(template, values, priors)
    return _finish(model, values, info, priors, data), priors


# deliberate: turns latent harmonics with pa folded to [0, 180); pinned by test_fold_gp_values_turns_the_harmonics
def _fold_gp_values(ctx, values, degree=None):
    """``_fold_fit`` for the Gaussian process: ``pa`` in [0°, 180°), latents turned."""
    pa = float(onp.asarray(values["disk.pa"]))
    if 0.0 <= pa < 180.0:
        return values
    out = dict(values)
    folded = pa % 180.0
    out["disk.pa"] = onp.asarray(folded)
    if "disk.pa_vec" in out:
        angle = math.radians(folded)
        out["disk.pa_vec"] = onp.asarray([math.cos(angle), math.sin(angle)])
    degree = ctx.degree if degree is None else degree
    site = "disk.harmonics.z"
    if degree and (pa % 360.0) >= 180.0 and site in out:
        orders = onp.arange(1, degree + 1, dtype=float)
        out[site] = onp.asarray(out[site]) * ((-1.0) ** orders)[None, :, None]
    return out


# deliberate: start selection by frozen-geometry chi2; test_geometry_search_leaves_a_wrong_mode scores it directly
def _geometry_search(p, geo, ell):
    """The start with the lowest frozen-geometry χ²: ``(geometry, fit, rows)``."""
    n_data = int(p.processed.n_independent)
    best, rows = None, []
    ctx = p._ctx()
    # the single ring can settle in a wrong mode that the released fit cannot leave
    starts = [(geo["inc"], geo["pa"])]
    starts += [
        (r["inc_deg"], r["pa_deg"])
        for r in p._report("geometry")["grid"]["rivals"]
    ]
    starts += [
        (inc, pa)
        for inc in _GEOMETRY_LATTICE_INC
        for pa in _GEOMETRY_LATTICE_PA
        if inc <= ctx.inc_max
    ]
    for inc, pa in starts:
        trial = {**geo, "inc": float(inc), "pa": float(pa)}
        # a few steps suffice: a wrong geometry costs far more χ² than a right one
        screen, _ = _gp_fit(p, ell, None, geo=trial, max_steps=_SCREEN_STEPS)
        rows.append(
            {
                "inc_deg": trial["inc"],
                "pa_deg": trial["pa"],
                "chi2_reduced": screen.chi2 / n_data,
            }
        )
        if best is None or screen.chi2 < best[0]:
            best = (screen.chi2, trial)
    fit, _ = _gp_fit(p, ell, None, geo=best[1])
    return best[1], fit, rows


def _score_ell(p, ell, planet, geo, k):
    """Fit the Gaussian-process disk at one ℓ and score it: a candidate row."""
    from ..imaging import log_evidence

    fit, priors = _gp_fit(p, ell, planet, geo=geo, release=True)
    name = f"gp_ell{k}"
    _save_fit(p.output / "models" / name, fit, list(priors))
    try:
        log_z = float(log_evidence(fit.model, p.processed, path="disk"))
    except (FloatingPointError, onp.linalg.LinAlgError) as err:
        warnings.warn(
            f"The evidence at ℓ = {ell:.3g} mas could not be computed "
            f"({err}).",
            RuntimeWarning,
            stacklevel=2,
        )
        log_z = None
    ctx, factor = p._ctx(), p._rescale_factor()
    n_data = int(p.processed.n_independent)
    return {
        "name": name,
        "ell_mas": ell,
        "ell_equivalent_mas": ell / _ELL_PER_EQUIVALENT,
        "ell_halffringes": ell / _ELL_PER_EQUIVALENT / ctx.halffringe,
        # On the quoted errors, whatever scale the fit used.
        "chi2": fit.chi2 * factor**2,
        "chi2_reduced": fit.chi2 * factor**2 / n_data,
        "chi2_rescaled": fit.chi2,
        "n_free": fit.n_free,
        "converged": fit.info.get("converged"),
        "steps": fit.info.get("steps"),
        "planet": planet is not None,
        "log_evidence": log_z,
    }


def _edge_side(candidates):
    """``"low"`` or ``"high"`` when the best evidence is the smallest or the
    largest ℓ tried, else None (also with fewer than two scored lengths)."""
    scored = [c for c in candidates if c["log_evidence"] is not None]
    if len(scored) < 2:
        return None
    best = max(scored, key=lambda c: c["log_evidence"])
    ells = [c["ell_mas"] for c in candidates]
    if best["ell_mas"] <= min(ells):
        return "low"
    if best["ell_mas"] >= max(ells):
        return "high"
    return None


# deliberate: callback: the profile stage of the "GP" entry of MODELS
def _profile_gp(p):
    from ..imaging import error_scale

    p._error_factor = 1.0  # until the baseline fit gives the scale
    ctx, data = p._ctx(), p.processed
    geo = _gp_geometry(p)
    geo, fit, starts = _geometry_search(p, geo, ctx.ell_mid)
    planet, searches = None, []
    if ctx.companion:
        # Search the residuals of the middle ℓ with no planet, and add any
        # detection to every fit.
        search = _point_source_search(
            fit.model, data, ctx, p.settings["batch_size"]
        )
        searches.append(search)
        if search["delta_chi2"] >= p.settings["companion_min_delta_chi2"]:
            planet = _planet_start(search, ctx)
    factor = 1.0
    if p.error_mode == "rescale":
        # One scale from the baseline fit at the middle ℓ, then fixed: every
        # fit, evidence and check after this is on the rescaled errors. The
        # baseline holds the companion when there is one and has the
        # geometry released: structure the model lacks would be absorbed
        # into the scale.
        base, _ = _gp_fit(p, ctx.ell_mid, planet, geo=geo, release=True)
        # MacKay's error scale s² = χ² / (N − γ) at the disk's MAP
        factor = float(error_scale(base.model, data, path="disk"))
        p._error_factor = factor
    candidates = []
    for ell in ctx.ells:
        candidates.append(_score_ell(p, ell, planet, geo, len(candidates)))
    # A grid must not choose its own edge: when the best ℓ is the smallest
    # or the largest, extend the grid once that way (the smallest not below
    # the floor), and let ``length_scale`` warn if it is still on an edge.
    side = _edge_side(candidates)
    extended = None
    if side is not None:
        ells = [c["ell_mas"] for c in candidates]
        new = (
            max(ctx.ell_floor, min(ells) / 2.0)
            if side == "low"
            else 2.0 * max(ells)
        )
        if side == "high" or new < min(ells) * (1.0 - 1e-9):
            candidates.append(_score_ell(p, new, planet, geo, len(candidates)))
            extended = side
    for k, c in enumerate(candidates):
        c["ell_index"] = k
    _plot_profile_fits(p, candidates, "ell_mas", "ℓ (mas)")
    return {
        "candidates": candidates,
        "searches": searches,
        "geometry_search": starts,
        "error_scale_factor": factor if p.error_mode == "rescale" else None,
        "grid": {
            "first_best_on_edge": side,
            "extended": extended,
            "on_grid_edge": _edge_side(candidates),
        },
    }


# === STAGE: COMPARE ===


def _compare(p):
    return models()[p.name].compare(p)


def _residual_search(p, model):
    """The point-source search on ``model``'s residuals, as a check."""
    search = _point_source_search(
        model, p.processed, p._ctx(), p.settings["batch_size"]
    )
    # The profile stage's search is on the quoted errors; with a rescale the
    # data here are scaled by s, which multiplies Δχ² by 1/s². Put the
    # threshold of ``companion_min_delta_chi2`` on one footing.
    factor = p._rescale_factor()
    if factor != 1.0:
        search = {**search, "delta_chi2": search["delta_chi2"] * factor**2}
    return search


def _common_compare(p, model, chi2, extra):
    """Numbers every ``compare`` reports, for the preferred model."""
    from ..models import MultipoleDisk

    ctx, data = p._ctx(), p.processed
    n = int(data.n_independent)
    factor = p._rescale_factor()
    rescaled = p.error_mode == "rescale"
    # the fitted parameters of models/best within 0.1% of a prior bound
    best = _io.read_json(p.output / "models" / "best" / "info.json")
    # the fraction of the disks' light (at the reference wavelength) beyond r_out - halffringe
    num = den = 0.0
    for part in model.components.values():
        if not isinstance(part, MultipoleDisk):
            continue
        r, w = (onp.asarray(x, dtype=float) for x in part.profile.quadrature())
        flux = float(onp.asarray(part._weight()))
        total = float(w.sum())
        if total <= 0:
            continue
        num += flux * float(w[r >= ctx.r_out - ctx.halffringe].sum()) / total
        den += flux
    return {
        "n_independent": n,
        "chi2": chi2,
        "chi2_reduced": chi2 / n,
        "error_scales": {"mackay": factor} if rescaled else None,
        "error_scale_factor": factor if rescaled else None,
        "at_bound": list(best.get("at_bound") or []),
        "degree": ctx.degree,
        "outer_fraction": num / den if den > 0 else float("nan"),
        **extra,
    }


def _load_candidate(p, name):
    folder = p.output / "models" / name
    return _io.load_model_values(folder), _io.load_model(folder)


def _save_best(p, name):
    """Copy candidate ``name`` to ``models/best``; its (values, model)."""
    folder = p.output / "models" / name
    values, model = _load_candidate(p, name)
    manifest = _io.read_json(folder / "manifest.json")
    _io.save_model(
        p.output / "models" / "best", model, values, manifest["params"]
    )
    _io.write_json(
        p.output / "models" / "best" / "info.json",
        _io.read_json(folder / "info.json"),
    )
    return values, model


# deliberate: callback: the compare stage of the "rings" entry of MODELS
def _compare_rings(p):
    from ..likelihood import whitened_residuals

    ctx, data = p._ctx(), p.processed
    rows = []
    for c in p._report("profile")["candidates"]:
        n = c["n"]
        values, model = _load_candidate(p, c["name"])
        shape = _Shape(n, ctx.colour, ctx.degree, c["planet"])
        priors = _ring_priors(ctx, shape)
        names = list(priors)
        flat = _scalars(values)
        sd, cov = _laplace(ctx, shape, names, flat, data)
        log_z = None
        if p.error_mode == "quoted" and cov is not None:
            sign, logdet = onp.linalg.slogdet(cov)
            if sign > 0:
                r = onp.asarray(whitened_residuals(model, data), dtype=float)
                # log prior; pa is an AngleVector, density per degree
                log_prior = sum(
                    -math.log(360.0)
                    if k == "pa"
                    else float(prior.log_prob(flat[k]))
                    for k, prior in priors.items()
                )
                log_z = float(
                    -0.5 * onp.sum(r**2)
                    + log_prior
                    + 0.5 * len(names) * math.log(2.0 * math.pi)
                    + 0.5 * logdet
                )
        rows.append(
            {
                **c,
                "log_evidence": log_z,
                "radii": [flat[f"r{i}"] for i in range(1, n + 1)],
                "widths": [flat[f"s{i}"] for i in range(1, n + 1)],
                "fluxes": [flat[f"f{i}"] for i in range(1, n + 1)],
                "sd": sd,
                "params": {k: flat[k] for k in names},
            }
        )
    # A count with two rings too close to tell apart is one ring fitted as
    # two (see ``_unresolved_pairs``): it does not compete. The first count
    # is one ring and never has such a pair. Those with a lower BIC than the
    # winner are reported: the data want a profile the rings cannot give.
    split = [r for r in rows if r.get("unresolved")]
    rows = [r for r in rows if not r.get("unresolved")]
    best = min(rows, key=lambda r: r["bic"])
    n_best = best["n"]
    split = [r["n"] for r in split if r["bic"] < best["bic"]]
    evidences = {
        r["n"]: r["log_evidence"]
        for r in rows
        if r["log_evidence"] is not None
    }
    by_evidence = max(evidences, key=evidences.get) if evidences else None
    _, model = _save_best(p, best["name"])
    search = _residual_search(p, model)
    pa_sd = best["sd"].get("pa")
    report = _common_compare(
        p,
        model,
        best["chi2"],
        {
            "best": best["name"],
            "n_best": n_best,
            "preferred_by_bic": n_best,
            "preferred_by_evidence": by_evidence,
            "split_counts": split,
            "rows": rows,
            "params": best["params"],
            "inc_deg": best["params"]["inc"],
            "pa_deg": best["params"]["pa"],
            "pa_sd_deg": pa_sd,
            "residual_search": search,
            "raised_search": None,
            "companion": _companion_report(
                p,
                {
                    k: best["params"][k]
                    for k in ("dra", "ddec", "fp", "ap")
                    if k in best["params"]
                }
                if best["planet"]
                else None,
            ),
        },
    )
    _plot_summary_file(p, model)
    return report


# deliberate: loads each ell candidate to correlate radial profiles; kept out of the already long GP compare callback
def _stability(p, rows):
    """The evidence spread across ℓ and the smallest correlation between
    the radial profiles of the fits at each ℓ (None when not computable)."""
    ctx = p._ctx()
    z = [r["log_evidence"] for r in rows if r["log_evidence"] is not None]
    spread = max(z) - min(z) if len(z) >= 2 else None
    r = onp.linspace(ctx.r_in, ctx.r_out, 200)
    curves = []
    for row in rows:
        model = _load_candidate(p, row["name"])[1]
        total = _profile_curves(model, r)[0]
        if onp.all(onp.isfinite(total)) and onp.std(total) > 0:
            curves.append(onp.asarray(total, dtype=float))
    if len(curves) < 2:
        return spread, None
    corr = onp.corrcoef(onp.stack(curves))
    return spread, float(onp.min(corr))


# deliberate: pixel-render chi2 check; test_gp_render_check_runs_and_passes_on_a_well_sampled_disc re-runs it on a model
def _render_stats(p, model):
    """χ²/N of the model and of its disk rendered to pixels (None if not computable)."""
    import equinox as eqx
    import jax.numpy as np

    from ..imaging import nyquist_pixel_scale
    from ..models import Image, System

    ctx, data = p._ctx(), p.processed
    out = {"chi2_model": None, "chi2_render": None, "npix": None}
    try:
        wavel = onp.unique(onp.round(onp.asarray(data.wavel, float), 12))
        w0 = wavel[onp.argmin(onp.abs(wavel - ctx.wavel0))]
        channel = data.select(w0 * (1.0 - 1e-6), w0 * (1.0 + 1e-6))
        n = int(channel.n_independent)
        # colour removed: the pixel image has one brightness
        if model.disk.spectral_index is not None:
            model = eqx.tree_at(
                lambda m: m.disk.spectral_index,
                model,
                None,
                is_leaf=lambda x: x is None,
            )
        fov = 2.0 * ctx.r_out
        pixel = max(0.5 * nyquist_pixel_scale(data), fov / _RENDER_NPIX)
        npix = int(2 * math.ceil(fov / pixel / 2.0))
        bright = onp.asarray(model.disk.render(npix, fov), dtype=float)
        bright = bright / bright.sum()
        image = Image(
            np.log(np.asarray(onp.clip(bright, 1e-14, None))),
            fov / npix,
            flux=model.disk.flux,
        )
        swapped = System({**model.components, "disk": image})
        out["chi2_model"] = _chi2_64(model, channel)[0] / n
        out["chi2_render"] = _chi2_64(swapped, channel)[0] / n
        out["npix"] = npix
    except (FloatingPointError, onp.linalg.LinAlgError) as err:
        warnings.warn(
            f"The render check could not run ({err}).",
            RuntimeWarning,
            stacklevel=2,
        )
    return out


# deliberate: callback: the compare stage of the "GP" entry of MODELS
def _compare_gp(p):
    import jax
    import jax.numpy as np

    from ..likelihood import whitened_residuals

    ctx, data = p._ctx(), p.processed
    # The evidences were computed with the fits, on the data they used.
    rows = [dict(c) for c in p._report("profile")["candidates"]]
    scored = [r for r in rows if r["log_evidence"] is not None]
    best = max(scored or rows, key=lambda r: r["log_evidence"] or -onp.inf)
    values, model = _save_best(p, best["name"])
    # the disk's light fraction within a half-fringe of ``r_in`` and its flux correlation
    z0 = np.asarray(model.get("disk.profile.z"))
    flux0 = np.asarray(model.get("disk.flux"))

    def build(x):
        return model.set(["disk.flux", "disk.profile.z"], [x[0], x[1:]])

    def residuals(x):
        return whitened_residuals(build(x), data)

    def fraction(x):
        r, w = build(x).disk.profile.quadrature()
        return np.sum(
            np.where(r <= ctx.r_in + ctx.halffringe, w, 0.0)
        ) / np.sum(w)

    x0 = np.concatenate([flux0.reshape(1), z0])
    jac = onp.asarray(jax.jacfwd(residuals)(x0), dtype=float)
    prior = onp.diag(onp.r_[1e-9, onp.ones(z0.size)])
    cov = onp.linalg.inv(jac.T @ jac + prior)
    grad = onp.asarray(jax.grad(fraction)(x0), dtype=float)
    value = float(fraction(x0))
    var_flux, var_frac = cov[0, 0], float(grad @ cov @ grad)
    cross = float(cov[0] @ grad)
    denom = math.sqrt(max(var_flux, 0.0) * max(var_frac, 0.0))
    inner_fraction = value
    inner_corr = cross / denom if denom > 0 else None
    spread, stability = _stability(p, rows)
    rendered = _render_stats(p, model)
    geometry = p._report("geometry")
    planet_values = (
        _planet_values(values, ctx.colour) if best["planet"] else None
    )
    search = _residual_search(p, model)
    raised = None
    if search["delta_chi2"] >= p.settings["companion_min_delta_chi2"]:
        # Does a higher harmonic degree remove it?
        higher, _ = _gp_fit(
            p,
            best["ell_mas"],
            planet_values,
            degree=ctx.degree + 1,
            geo=_released_geometry(p, values),
        )
        raised = _residual_search(p, higher.model)
    report = _common_compare(
        p,
        model,
        best["chi2"],
        {
            "best": best["name"],
            "ell_mas": best["ell_mas"],
            "rows": rows,
            "inner_fraction": inner_fraction,
            "inner_correlation": inner_corr,
            "evidence_spread": spread,
            "profile_correlation": stability,
            "render": rendered,
            "grid": p._report("profile")["grid"],
            # The released geometry; the position angle's spread is the
            # single-ring fit's, an approximation used only to gate the
            # inclination check.
            "inc_deg": float(onp.asarray(values["disk.inc"])),
            "pa_deg": float(onp.asarray(values["disk.pa"])),
            "pa_sd_deg": geometry["laplace_sd"].get("pa"),
            "flux": float(onp.asarray(values["disk.flux"])),
            "residual_search": search,
            "raised_search": raised,
            "companion": _companion_report(p, planet_values),
        },
    )
    _plot_summary_file(p, model)
    return report


# === PLOTS ===


def _profile_curves(model, r):
    """Profile, spectral index and lopsidedness at radii ``r``, light-weighted over disks."""
    from ..models import MultipoleDisk

    total = onp.zeros_like(r)
    alpha = onp.zeros_like(r)
    lop = onp.zeros_like(r)
    has_alpha = has_lop = False
    for part in model.components.values():
        if not isinstance(part, MultipoleDisk):
            continue
        i0 = onp.asarray(part.profile.intensity(r, 0.0), dtype=float)
        norm = onp.trapezoid(i0 * r, r)
        if not norm > 0:
            continue
        i0 = float(onp.asarray(part._weight())) * i0 / norm
        total += i0
        a = onp.zeros_like(r)
        if part.spectral_index is not None:
            a = onp.asarray(part._alpha(r), dtype=float)
            has_alpha = True
        elif hasattr(part.flux, "index"):
            a = a + float(onp.asarray(part.flux.index))
            has_alpha = True
        alpha += i0 * a
        if part.harmonics is not None and part.harmonics.degree:
            h = onp.asarray(part.harmonics.image_harmonics(r))
            first = onp.abs(h[1])
            lop += i0 * onp.broadcast_to(first, r.shape)
            has_lop = True
    safe = onp.where(total > 0, total, 1.0)
    return (
        total,
        alpha / safe if has_alpha else None,
        lop / safe if has_lop else None,
    )


def plot_summary(model, data, *, r_out_mas, r_in_mas=0.0, npix=200):
    """The fitted disk: images, profile, colour, lopsidedness and residuals.

    Parameters
    ----------
    model : System
        The fitted model (``Result.model()``).
    data : OIData
        The data (``Result.data()``).
    r_out_mas : float
        Outer radius of the radial range (mas); the images span ±1.2 times
        this.
    r_in_mas : float, optional
        Inner radius of the radial range (mas); the curves start there.
    npix : int, optional
        Pixels of each image.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    from ..models import MultipoleDisk, System
    from ..plotting import plot_model

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fov = 2.4 * r_out_mas
    plot_model(
        model, fov, npix=npix, ax=axes[0, 0], saturate=0.99, title="Model"
    )
    disks = {
        k: v
        for k, v in model.components.items()
        if isinstance(v, MultipoleDisk)
    }
    plot_model(
        System(disks),
        fov,
        npix=npix,
        ax=axes[0, 1],
        title="Disk without the point sources",
    )
    r = onp.linspace(max(r_in_mas, 1e-3), r_out_mas, 300)
    i0, alpha, lop = _profile_curves(model, r)
    ax = axes[0, 2]
    ax.plot(r, i0)
    ax.set(
        yscale="log",
        xlabel="radius in the disk plane (mas)",
        ylabel="I₀(r)",
        ylim=(max(i0.max() * 1e-5, 1e-30), i0.max() * 3),
    )
    for ax, curve, label in (
        (axes[1, 0], alpha, "spectral index α(r)"),
        (axes[1, 1], lop, "lopsidedness |I₁|/I₀"),
    ):
        if curve is None:
            ax.text(
                0.5, 0.5, "not fitted", ha="center", transform=ax.transAxes
            )
            ax.set_axis_off()
            continue
        ax.plot(r, onp.where(i0 > 1e-3 * i0.max(), curve, onp.nan))
        ax.set(xlabel="radius in the disk plane (mas)", ylabel=label)
    ax = axes[1, 2]
    _, err = data.flatten_data()
    z = onp.asarray(data.residuals(data.model(model))) / err
    n_vis = onp.asarray(data.vis).size
    for sl, label in (
        (slice(0, n_vis), "V²"),
        (slice(n_vis, None), "closure phase"),
    ):
        if z[sl].size:
            ax.plot(
                z[sl],
                ".",
                ms=3,
                label=f"{label}: rms z = {onp.sqrt(onp.mean(z[sl] ** 2)):.2f}",
            )
    ax.axhline(0, color="k", lw=0.5)
    ax.set(xlabel="data index", ylabel="residual / quoted error", ylim=(-5, 5))
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def _plot_summary_file(p, model):
    ctx = p._ctx()
    fig = plot_summary(
        model, p.processed, r_out_mas=ctx.r_out, r_in_mas=ctx.r_in
    )
    _save(fig, p.output / "plots" / "disk_summary.png")


# === STAGE: POSTERIOR ===


# deliberate: fold about the MAP, harmonics flipped by parity; pinned by test_fold_pa_keeps_the_disk
def _fold_pa(samples, pa_site, parity, centre):
    """``pa`` folded to within 90° of ``centre``, the ``parity``-signed harmonics flipped."""
    pa = onp.asarray(samples[pa_site], dtype=float)
    # fold about the MAP, not a fixed angle, which would split a posterior near it
    turns = onp.floor((pa - centre + 90.0) / 180.0)
    flip = (turns % 2.0) == 1.0
    out = dict(samples)
    out[pa_site] = pa - 180.0 * turns
    for site, sign in parity.items():
        x = onp.asarray(samples[site], dtype=float)
        f = flip.reshape(flip.shape + (1,) * (x.ndim - flip.ndim))
        out[site] = onp.where(f, x * onp.asarray(sign, dtype=float), x)
    return out


def _best_row(compare):
    """The ``compare`` row of the preferred model."""
    return next(r for r in compare["rows"] if r["name"] == compare["best"])


# deliberate: float64 boundary (in_float64) around the NUTS run
@in_float64
def _nuts(p, model, priors, start):
    from ..likelihood import numpyro_model

    posterior = numpyro_model(
        model, priors, p.processed, noise=p._noise(), flat_coordinates=True
    )
    mcmc = _run_chains(
        posterior, p.settings, start, dense_mass=True, priors=priors
    )
    return _draws(mcmc)


def _posterior(p):
    s = p.settings
    if not s["posterior"]:
        (p.output / "samples.h5").unlink(missing_ok=True)
        return {"skipped": True, "reason": "posterior=False"}
    # the model, priors and start of the NUTS run, its position-angle site and parity
    ctx = p._ctx()
    compare = p._report("compare")
    values = _io.load_model_values(p.output / "models" / "best")
    row = _best_row(compare)
    if p.name == "rings":
        shape = _Shape(row["n"], ctx.colour, ctx.degree, row["planet"])
        priors = _ring_priors(ctx, shape)
        start = {k: onp.asarray(values[k]) for k in priors}
        start["pa_vec"] = onp.asarray(values["pa_vec"])
        parity = {
            f"h{i}_{part}{j}": (-1.0) ** j
            for i in range(1, shape.n + 1)
            for j in range(1, ctx.degree + 1)
            for part in ("re", "im")
        }
        model, pa_site, centre = (
            _builder(ctx, shape),
            "pa",
            float(values["pa"]),
        )
    else:
        geo = _released_geometry(p, values)
        planet = _planet_values(values, ctx.colour) if row["planet"] else None
        template, priors = _gp_template(ctx, geo, compare["ell_mas"], planet)
        model = _apply(template, values, priors)
        # NUTS runs on the unfrozen model: the geometry is released here
        priors = {
            **priors,
            **{f"disk.{k}": v for k, v in _geometry_priors(ctx).items()},
        }
        start = {k: onp.asarray(values[k]) for k in priors if k in values}
        angle = math.radians(float(geo["pa"]))
        start["disk.inc"] = onp.asarray(max(float(geo["inc"]), 5.0))
        start["disk.pa_vec"] = onp.asarray([math.cos(angle), math.sin(angle)])
        orders = onp.arange(1, ctx.degree + 1, dtype=float)
        parity = (
            {"disk.harmonics.z": ((-1.0) ** orders)[None, :, None]}
            if ctx.degree
            else {}
        )
        pa_site, centre = "disk.pa", float(geo["pa"])
    samples, stats = _nuts(p, model, priors, start)
    samples = _fold_pa(
        samples,
        pa_site,
        {k: v for k, v in parity.items() if k in samples},
        centre,
    )
    floats = {
        k: x
        for k, x in samples.items()
        if onp.issubdtype(x.dtype, onp.floating)
    }
    diagnostics = {k: _diagnostics([x]) for k, x in floats.items()}
    summary = {
        k: _site_summary(x, diagnostics[k], angle=k == pa_site, period=180.0)
        for k, x in floats.items()
        if x.ndim == 2 and not k.endswith("_vec")
    }
    bounded = {
        k: priors[k] for k in priors if k in samples and samples[k].ndim == 2
    }
    _write_samples(
        p,
        {"posterior": samples, "sample_stats": stats},
        pa_site=pa_site,
        pa_folded="pa within 90 degrees of the MAP, harmonics of order m times (-1)^m",
    )
    return {
        "skipped": False,
        "best": p._report("compare")["best"],
        "params": summary,
        "noise": _noise_quantiles(samples),
        **_sampler_report(s, [_chain_health(diagnostics, stats)]),
        "prior_bound_fraction": _prior_bounds(bounded, samples),
    }


# === SUMMARY AND CHECKS ===

_DATA_KEYS = (
    "n_vis",
    "n_phi",
    "n_independent",
    "closure_phases",
    "wavel_min_m",
    "wavel_max_m",
    "baseline_min_m",
    "baseline_max_m",
    "resolution_mas",
    "fov_mas",
)
_OVERVIEW_KEYS = (
    "resolution_mas",
    "r_in_mas",
    "r_out_mas",
    "n_r_rings",
    "n_r_gp",
    "n_channels",
    "colour",
    "wavel0_m",
    "ell_mas",
    "star",
)
_GEOMETRY_KEYS = (
    "inc_deg",
    "pa_deg",
    "r_mas",
    "sigma_mas",
    "flux",
    "chi2_reduced",
    "converged",
    "gp_sigma",
    "laplace_sd",
    "grid",
)
_POSTERIOR_KEYS = (
    "best",
    "params",
    "noise",
    "num_chains",
    "num_samples",
    "num_warmup",
    "r_hat_max",
    "ess_bulk_min",
    "divergence_fraction",
)


def _chi2_checks(entry, compare):
    checks = [
        _checks.chi2_reduced(
            "chi2_reduced",
            compare["chi2_reduced"],
            label=f"the {entry.label} fit",
        )
    ]
    if compare["error_scales"]:
        checks.append(
            _checks.error_scale(compare["error_scales"], kind="estimated")
        )
    return checks


def _fit_checks(entry, settings, overview, geometry, compare):
    hf = overview["resolution_mas"]
    by = {}
    rows = compare["rows"]
    if entry.name == "gp":
        by["length_scale"] = _checks.length_scale(
            [r["log_evidence"] for r in rows],
            [r["ell_mas"] for r in rows],
            hf,
            nats=settings["length_scale_nats"],
            edge=compare["grid"]["on_grid_edge"],
            extended=compare["grid"]["extended"],
        )
        by["structure_unconstrained"] = _checks.structure_unconstrained(
            compare["evidence_spread"],
            compare["profile_correlation"],
            nats=settings["length_scale_nats"],
            minimum=settings["unconstrained_correlation"],
        )
        by["render_consistency"] = _checks.render_consistency(
            compare["render"]["chi2_model"],
            compare["render"]["chi2_render"],
            tolerance=settings["render_tolerance"],
        )
        by["inner_edge"] = _checks.inner_edge(
            compare["inner_fraction"],
            compare["inner_correlation"],
            hf,
            max_fraction=settings["inner_edge_fraction"],
            max_correlation=settings["inner_edge_correlation"],
        )
    else:
        by["ring_count"] = _checks.ring_count(
            compare["n_best"],
            rows,
            hf,
            nats=settings["ring_count_nats"],
            split=compare["split_counts"],
        )
    by["outer_edge"] = _checks.outer_edge(
        compare["outer_fraction"], hf, warn=settings["outer_edge_fraction"]
    )
    by["inclination"] = _checks.inclination(
        compare["inc_deg"],
        compare["pa_sd_deg"],
        geometry["grid"]["n_minima"],
        minimum=settings["inclination_min_deg"],
    )
    by["at_bound"] = _checks.at_bound(compare["at_bound"])
    # The Laplace widths behind the position-angle spread (and the ring
    # evidences) need a positive-definite Hessian: say which are missing.
    sds = {f"geometry.{k}": v for k, v in geometry["laplace_sd"].items()}
    for row in rows:
        if row["name"] == compare["best"]:
            sds.update(
                {f"{row['name']}.{k}": v for k, v in row.get("sd", {}).items()}
            )
    missing = sorted(k for k, v in sds.items() if v is None)
    by["laplace_pd"] = _checks.Check(
        "laplace_pd",
        "warn" if missing else "pass",
        len(missing),
        0,
        (
            "The Laplace covariance is not positive definite for "
            + ", ".join(missing)
            + ": those parameters are unconstrained or the fit is not at a "
            "likelihood maximum, so they have no width (not a width of zero), "
            "and an evidence that needs the covariance is missing."
            if missing
            else "The Laplace covariance is positive definite."
        ),
    )
    raised = compare["raised_search"]
    by["absorbed_companion"] = _checks.absorbed_companion(
        compare["residual_search"]["delta_chi2"],
        raised["delta_chi2"] if raised else None,
        compare["degree"],
        threshold=settings["companion_min_delta_chi2"],
    )
    return [by[name] for name in CHECKS if name in by and name in entry.checks]


def _summarize(name, settings, reports):
    entry = models()[name]
    sections, checks = {}, []
    load, over, geo, prof, comp, post = (
        reports.get(k)
        for k in (
            "load",
            "overview",
            "geometry",
            "profile",
            "compare",
            "posterior",
        )
    )
    if load:
        sections["data"] = _pick(load, _DATA_KEYS)
    if over:
        sections["model"] = {
            "name": name,
            "label": entry.label,
            **{k: over[k] for k in _OVERVIEW_KEYS},
        }
    if geo:
        sections["geometry"] = _pick(geo, _GEOMETRY_KEYS)
    if prof:
        sections["profile"] = {"candidates": prof["candidates"]}
    if not comp:
        return sections, checks
    sections["chi2"] = {
        "n_independent": comp["n_independent"],
        "error_scale": _error_mode(settings, name),
        "error_scale_factor": comp["error_scale_factor"],
        "reduced": {comp["best"]: comp["chi2_reduced"]},
        "error_scales": comp["error_scales"],
    }
    sections["compare"] = {
        "best": comp["best"],
        "rows": comp["rows"],
        "residual_search": comp["residual_search"],
        "raised_search": comp["raised_search"],
    }
    disk = {"model": name, "best": comp["best"]}
    if name == "rings":
        row = _best_row(comp)
        disk.update(
            n_rings=comp["n_best"],
            rings=[
                {"r_mas": r, "sigma_mas": w, "flux": f}
                for r, w, f in zip(row["radii"], row["widths"], row["fluxes"])
            ],
        )
    else:
        disk.update(
            ell_mas=comp["ell_mas"],
            flux=comp["flux"],
            on_grid_edge=comp["grid"]["on_grid_edge"],
            grid_extended=comp["grid"]["extended"],
        )
    disk["inc_deg"] = comp["inc_deg"]
    disk["pa_deg"] = comp["pa_deg"]
    disk["companion"] = comp["companion"]
    sections["disk"] = disk
    checks += _chi2_checks(entry, comp)
    checks += _fit_checks(entry, settings, over, geo, comp)
    if post and not post.get("skipped"):
        sections["posterior"] = _pick(post, _POSTERIOR_KEYS)
        checks += _sampler_checks(post)
    return sections, checks
