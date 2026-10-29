"""`ImagingPipeline`: a data-consistent image of the source (provisional).

It reports an image that fits the data on the quoted errors, at
hyperparameters chosen by a stated rule, and cannot promise that the
morphology is right: every data-fidelity check passed on contest
reconstructions that were judged unusable.
"""

from __future__ import annotations

import dataclasses
import itertools
import warnings
from collections import Counter

import jax
import numpy as onp

from . import _checks, _io
from ._common import (
    _integer,
    _load_report,
    _overview_plots,
    _positive,
    _quicklook,
    _save_fit,
    _validate_data_settings,
)
from ._core import Stage, _Pipeline

# The checks, in reporting order.
CHECKS = (
    "chi2_reduced",
    "chi2_reduced_vis",
    "chi2_reduced_phi",
    "quoted_errors",
    "error_scale",
    "effective_parameters",
    "short_baselines",
    "star",
    "orientation",
    "edges",
    "ensemble_kept",
    "ensemble_spread",
)
_ANISOTROPIC_BELOW = 0.9  # beam axis ratio that adds the elliptical GP grid


class ImagingPipeline(_Pipeline):
    """Image the source from V², closure phases or complex visibilities.

    !!! warning "Provisional"
        May change in any 0.4.x release without a deprecation cycle.

    A Gaussian-process image on the quoted errors; a good fit does **not**
    promise that the morphology is right. Multi-channel data give one grey
    image. Stages: ``load``, ``overview`` (the grid), ``star`` (with or
    without a point source, by log evidence), ``hyperparameters`` (σ and ℓ by
    Laplace evidence), ``members`` (``members`` fits at those shared
    hyperparameters), ``combine`` (``images/mean.fits`` and ``sigma.fits``)
    and ``quicklook``. To spread the members over tasks, call
    ``pipeline.run(through="hyperparameters")``, then ``pipeline.run_member(k)``
    for ``k`` in ``range(members)``, then ``pipeline.run()``, which returns the
    ``Result`` (``res.image()``). All settings, with defaults:
    [pipeline settings](../pipeline.md#imagingpipeline-settings).

    Parameters
    ----------
    data : OIData
        The data.
    model : {"gp"}, optional
        The image prior (the only one in this version).
    output : str or os.PathLike, optional
        Run folder (default ``"run"``).
    **settings
        Overrides of ``ImagingPipeline.defaults()``. The most used: ``star``
        (``"auto"``, or True or False to force the point source on or off),
        ``members`` (8), ``field_fwhms`` (6), ``oversample`` (3), ``max_npix``
        (128, most pixels per side).

    Examples
    --------
    >>> from virgil.pipeline import ImagingPipeline  # doctest: +SKIP
    >>> res = ImagingPipeline(data, output="runs/image").run()  # doctest: +SKIP
    >>> res.image().pixels.shape  # doctest: +SKIP
    """

    NAME = "image"
    _OWNED_EXTRA = ("images", "members")
    STABILITY = "provisional"
    STAGES = (
        "load",
        "overview",
        "star",
        "hyperparameters",
        "members",
        "combine",
        "quicklook",
    )
    _DEFAULTS = {
        "oversample": 3.0,
        "field_fwhms": 6.0,
        "field_mas": None,
        "max_npix": 128,
        "star": "auto",
        "star_threshold": 5.0,
        "sigmas": [1.0, 2.0, 4.0],
        "lengths_beam": [0.5, 1.0, 2.0],
        "grid_growths": 2,
        "members": 8,
        "member_grids": False,
        "method": None,
        "max_steps": None,
        "seed": 0,
        "wavel_range": None,
        "error_floor": None,
        "quicklook_execute": True,
    }

    def __init__(self, data, model=None, *, output="run", **settings):
        if model not in (None, "gp"):
            raise ValueError(f'model must be "gp"; got {model!r}.')
        super().__init__(data, "gp", output=output, **settings)

    @classmethod
    def _validate(cls, s):
        for name in ("oversample", "field_fwhms", "star_threshold"):
            _positive(s, name)
        if s["field_mas"] is not None:
            _positive(s, "field_mas")
        _integer(s, "max_npix", 4)
        _integer(s, "grid_growths", 0)
        _integer(s, "members", 1)
        if s["members"] > 8:
            raise ValueError(
                "members must be at most 8, the distinct starts, pixel "
                "scales and fields."
            )
        _integer(s, "seed", 0)
        for name in ("sigmas", "lengths_beam"):
            if not s[name] or min(s[name]) <= 0:
                raise ValueError(f"{name} must be positive numbers.")
        if s["star"] not in ("auto", True, False):
            raise ValueError('star must be "auto", True or False.')
        _validate_data_settings(s)

    def _model_digest(self):
        return {"class": "Image", "digest": None, "models": ["gp"]}

    def _stages(self):
        model = tuple(
            f"models/best/{f}"
            for f in ("manifest.json", "values.npz", "info.json")
        )
        return (
            Stage("load", _load, ("stages/load/report.json",)),
            Stage(
                "overview",
                _overview,
                ("plots/overview_data.png", "plots/overview_uv.png"),
            ),
            Stage("star", _star, ("stages/star/report.json",)),
            Stage("hyperparameters", _hyperparameters, ("grids.h5", *model)),
            # No outputs: run_member's folders survive a rerun of this stage.
            Stage("members", _members, ()),
            Stage(
                "combine",
                _combine,
                ("images/mean.fits", "images/sigma.fits"),
            ),
            Stage("quicklook", _quicklook, ("quicklook.ipynb",)),
        )

    def run_member(self, k):
        """Fit member ``k`` unless it is saved (after the ``hyperparameters``
        stage); the ``members`` stage runs every one that is missing."""
        if not 0 <= k < self.settings["members"]:
            raise ValueError(
                f"k must be in 0..{self.settings['members'] - 1}."
            )
        # As run() would: the same configuration, after the hyperparameters.
        record = _io.read_json(self.output / "run.json")
        self._check_resume(record)
        if (
            record["stages"].get("hyperparameters", {}).get("status")
            != "complete"
        ):
            raise RuntimeError(
                'Run the "hyperparameters" stage before run_member.'
            )
        folder = self.output / "members" / f"{k:03d}"
        if not (folder / "info.json").exists():
            spec, star, draws, variant = _member_setup(self)
            grid = _evidence_grid(self, spec, star, draws[k])
            best = grid.best
            if variant:  # one point per variant: the shared hyperparameters
                # Each member re-estimates the elongation on its own field.
                best = next(
                    (f for f in grid.fits.values() if f.variant == variant),
                    best,
                )
                if best.variant != variant:
                    warnings.warn(
                        f"Member {k} found no {variant!r} grid on its field: "
                        "it is fitted with the isotropic prior."
                    )
            _save_fit(folder, best.result, list(best.result.values))

    def _summarize(self, reports):
        checks = _run_checks(self.settings, reports)
        names = {"load": "data", "overview": "field", "combine": "fit"}
        sections = {names.get(k, k): dict(v) for k, v in reports.items()}
        return sections, checks


# === HELPERS ===


def _grid_spec(s, field_mas, point=None, aniso=True):
    """The ensemble spec of this run's one group. ``point`` is one
    ``(σ, ℓ in beams)``, without growth; ``aniso`` adds the elliptical grid."""
    from ..ensemble import EnsembleSpec

    sigmas, lengths = sorted(s["sigmas"]), sorted(s["lengths_beam"])
    if point:
        sigmas, lengths = [point[0]], [point[1]]
    return EnsembleSpec(
        families=("gp",),
        oversample=(float(s["oversample"]),),
        field_factors=(1.0,),
        starts=("moments",),
        max_npix=int(s["max_npix"]),
        field_mas=field_mas,
        gp_sigmas=tuple(sigmas),
        gp_lengths=tuple(lengths),
        gp_growths=0 if point else int(s["grid_growths"]),
        gp_axis_ratio=_ANISOTROPIC_BELOW if aniso else None,
    )


def _draw(p, spec, star):
    from ..ensemble import draw_groups

    return draw_groups(
        p.processed,
        1,
        jax.random.PRNGKey(int(p.settings["seed"])),
        spec,
        star=star,
    )[0]


def _evidence_grid(p, spec, star, draw=None):
    from ..ensemble import evidence_grid, reference_starts

    s, data = p.settings, p.processed
    draw = draw or _draw(p, spec, star)
    return evidence_grid(
        data,
        draw,
        star=star,
        starts=reference_starts(data, star, (draw.start,)),
        spec=spec,
        **{k: s[k] for k in ("method", "max_steps") if s[k] is not None},
    )


# === STAGES ===


def _load(p):
    from ..imaging import _observable_blocks

    data = p.processed
    _, counts = _observable_blocks([data])
    quoted = {}
    for key, values in (("vis", data.d_vis), ("phi", data.d_phi)):
        if onp.size(values):
            values = onp.asarray(values, dtype=float).ravel()
            good = values[onp.isfinite(values) & (values > 0)]
            equal = good.size and onp.ptp(good) <= 1e-6 * good.max()
            quoted[key] = {
                "all_equal": bool(equal),
                "n_bad": int(values.size - good.size),
            }
    return _load_report(
        p,
        n_by_observable={k: int(v) for k, v in counts.items()},
        quoted_errors=quoted,
    )


def _overview(p):
    from ..imaging import beam, envelope_fwhm, image_field

    s, data = p.settings, p.processed
    _overview_plots(p, data)
    candidate = s["star"] is not False
    field = s["field_mas"] or float(
        image_field(
            data,
            fwhm_mas=float(envelope_fwhm(data, candidate)),
            field_fwhms=s["field_fwhms"],
        )
    )
    draw = _draw(p, _grid_spec(s, field), candidate)
    return {
        "field_requested_mas": float(field),
        "pixel_scale_mas": float(draw.pixel_scale_mas),
        "npix": int(draw.npix),
        "beam": dataclasses.asdict(beam(data)),
    }


def _star(p):
    s = p.settings
    if s["star"] != "auto":
        return {
            "decision": "on" if s["star"] else "off",
            "member_star": bool(s["star"]),
            "delta_log_z": None,
        }
    middle = [sorted(s[k])[len(s[k]) // 2] for k in ("sigmas", "lengths_beam")]
    spec = _grid_spec(
        s, p._report("overview")["field_requested_mas"], middle, aniso=False
    )
    log_z = {
        star: float(_evidence_grid(p, spec, star).best.log_evidence)
        for star in (True, False)
    }
    delta = log_z[True] - log_z[False]
    threshold = float(s["star_threshold"])
    return {
        "decision": "on"
        if delta >= threshold
        else "off"
        if delta <= -threshold
        else "both",
        "member_star": bool(delta >= 0.0),
        "delta_log_z": float(delta),
        "log_z_star": log_z[True],
        "log_z_no_star": log_z[False],
    }


def _edges(grid):
    """Which edges of its own grid the evidence optimum is on."""
    best = grid.best
    axes = (
        ("length", grid.lengths_mas[best.variant], best.length_mas),
        ("sigma", grid.sigmas[best.variant], best.sigma),
    )
    return [
        f"{name} {side}"
        for name, values, at in axes
        for side, end in (("min", values[0]), ("max", values[-1]))
        if at == end
    ]


def _hyperparameters(p):  # deliberate: stage callable
    from ..imaging import chi2_by_observable, error_scale
    from ..models import System

    data, star = p.processed, p._report("star")["member_star"]
    ov = p._report("overview")
    grid = _evidence_grid(
        p, _grid_spec(p.settings, ov["field_requested_mas"]), star
    )
    best = grid.best
    _save_fit(
        p.output / "models" / "best", best.result, list(best.result.values)
    )
    _io.write_h5(
        p.output / "grids.h5",
        {
            f"evidence_{name}": {
                "lengths_mas": onp.asarray(grid.lengths_mas[name]),
                "sigmas": onp.asarray(grid.sigmas[name]),
                "log_evidence": grid.log_evidence[name],
            }
            for name in sorted(grid.log_evidence)
        },
    )
    scales = error_scale(
        best.result, data, "env" if star else None, by_observable=True
    )
    return {
        "sigma": float(best.sigma),
        "length_mas": float(best.length_mas),
        "length_beam": float(best.length_mas / ov["beam"]["minor_mas"]),
        "variant": best.variant,
        "log_z": float(best.log_evidence),
        "edge": _edges(grid),
        "scale_by_observable": {k: float(v) for k, v in scales.items()},
        # MacKay's well-determined parameters; a star-free model is a bare
        # Image, so it is wrapped to give chi2_by_observable a path.
        "gamma": float(
            chi2_by_observable(
                best.result.model if star else System(env=best.result.model),
                data,
                "env",
            )["n_gamma"]
        ),
    }


def _member_setup(p):
    """``(spec, star, draws, variant)`` of the members. They vary the start,
    the pixel scale and the field (1 or 1.5 times); ``variant`` is the shared
    hyperparameters' (None: an evidence grid per member)."""
    from ..ensemble import draw_groups

    s = p.settings
    hyper, field = p._report("hyperparameters"), p._report("overview")
    star = p._report("star")["member_star"]
    shared = not s["member_grids"]
    variant = hyper["variant"] if shared else None
    spec = _grid_spec(
        s,
        field["field_requested_mas"],
        (hyper["sigma"], hyper["length_beam"]) if shared else None,
        variant != "iso",
    )
    # Each member is one distinct (start, pixel scale, field): a fit is fixed
    # by them, so repeated draws would be copies.
    key = jax.random.PRNGKey(int(s["seed"]))
    draws = {}
    for k, (over, factor, start) in enumerate(
        itertools.product((1.0, 1.5), (1.0, 1.5), ("moments", "flat"))
    ):
        if k < s["members"]:
            one = dataclasses.replace(
                spec,
                oversample=(over * spec.oversample[0],),
                field_factors=(factor,),
                starts=(start,),
            )
            draws[k] = dataclasses.replace(
                draw_groups(p.processed, 1, key, one, star)[0], index=k
            )
    return spec, star, draws, variant


def _members(p):  # deliberate: stage callable
    for k in range(p.settings["members"]):
        p.run_member(k)
    return {"n_members": int(p.settings["members"])}


def _combine(p):  # deliberate: stage callable
    """The mean and sigma images of the kept members, and the diagnostics of
    the mean image."""
    from ..ensemble import GPFit, Group, _chi2, combine
    from ..fitting import FitResult
    from ..imaging import (
        Centroid,
        chi2_by_observable,
        diagnose,
        short_baseline_residual,
    )
    from ..models import PointSource, System

    data, ov = p.processed, p._report("overview")
    spec, star, draws, _ = _member_setup(p)
    # V² and closure phases are judged separately.
    blocks = [
        (k, block)
        for k, block in (("vis", data.vis), ("phi", data.phi))
        if onp.size(block)
    ]
    parts = [data.select(observables=k) for k, _ in blocks]
    nan = float("nan")
    groups = []
    for k, draw in sorted(draws.items()):
        folder = p.output / "members" / f"{k:03d}"
        model = _io.load_model(folder)
        info = {
            "chi2": _chi2(model, parts),
            "ndata": [d.n_independent for d in parts],
        }
        result = FitResult(model, _io.load_model_values(folder), info)
        groups.append(Group(draw, None, (GPFit("gp", nan, nan, nan, result),)))
    ens = combine(parts, groups, spec=spec, star=star)

    # The star's share of the flux; the images carry the rest.
    flux = float(ens.mean.flux)
    fraction = flux / (1.0 + flux) if star else 1.0
    scale = float(ens.mean.pixel_scale_mas)
    kept = ens.kept
    for name, pixels in (("mean", ens.mean.brightness), ("sigma", ens.std)):
        _io.write_fits_image(
            p.output / "images" / f"{name}.fits",
            onp.asarray(pixels, dtype=float) * fraction,
            scale,
            header={"STARFRAC": 1.0 - fraction, "NMEMBERS": len(kept)},
        )
    scene = System(star=PointSource(), env=ens.mean) if star else ens.mean
    by_observable = chi2_by_observable(scene, data)
    try:
        sb_mean, sb_error = short_baseline_residual(scene, data)
    except ValueError:  # no baseline-ordered visibilities
        sb_mean = sb_error = None
    diagnosis = diagnose(
        scene, data, () if star else (Centroid(ov["pixel_scale_mas"]),)
    )
    # The widest field's outer pixels hide a smaller member's edge: each kept
    # member is judged on its own grid, by the fraction of its image in the
    # outer two pixels.
    pixels = [
        onp.asarray(
            (m.result.model.env if star else m.result.model).brightness
        )
        for m in kept
    ]
    edge_flux = max(
        float(1.0 - px[2:-2, 2:-2].sum() / px.sum()) for px in pixels
    )
    return {
        "chi2": by_observable["chi2"]["total"],
        "chi2_reduced": by_observable["chi2_red"]["total"],
        "n_by_observable": by_observable["n"],
        "chi2_red_by_observable": by_observable["chi2_red"],
        "short_baseline_mean": sb_mean,
        "short_baseline_error": sb_error,
        "flip_dchi2": float(diagnosis.checks["flip_dchi2"]),
        "edge_flux": edge_flux,
        "chi2_red_mixture": dict(zip((k for k, _ in blocks), ens.chi2_red)),
        "star_fraction": 1.0 - fraction,
        "n_members": len(ens.members),
        "n_kept": len(kept),
        "dropped": dict(Counter(m.reason for m in ens.members if not m.kept)),
        "spread": float(ens.std.sum() / ens.mean.brightness.sum()),
    }


# deliberate: orders the checks of the stages that ran; test_summary_orders_the_checks feeds it partial reports
def _run_checks(settings, reports):
    """The checks of the stages that ran, in the order of ``CHECKS``."""
    load, star = reports.get("load"), reports.get("star")
    hyper, fit = reports.get("hyperparameters"), reports.get("combine")
    checks = []
    if fit:
        red = fit["chi2_red_by_observable"]
        checks += [
            _checks.chi2_reduced(
                name, red[kind], label=f"the {kind} of the image"
            )
            for name, kind in (
                ("chi2_reduced", "total"),
                ("chi2_reduced_vis", "vis"),
                ("chi2_reduced_phi", "phi"),
            )
            if kind in red
        ]
    if load:
        checks.append(_checks.quoted_errors(load["quoted_errors"]))
    if hyper:
        checks.append(
            _checks.error_scale(hyper["scale_by_observable"], kind="estimated")
        )
    if fit and load:
        checks += [
            _checks.effective_parameters(
                load["n_independent"], hyper["gamma"]
            ),
            _checks.short_baselines(
                fit["short_baseline_mean"], fit["short_baseline_error"]
            ),
        ]
    if star:
        checks.append(
            _checks.star_decision(
                star["delta_log_z"],
                threshold=settings["star_threshold"],
                decision=star["decision"],
                fixed=settings["star"] is True,
            )
        )
    if fit and load:
        checks += [
            _checks.orientation(fit["flip_dchi2"], load["closure_phases"]),
            _checks.edges(hyper["edge"], fit["edge_flux"]),
        ]
    if fit:
        checks += [
            _checks.ensemble_kept(fit["n_kept"], fit["n_members"]),
            _checks.ensemble_spread(
                fit["spread"],
                fit["n_kept"],
                shared=not settings["member_grids"],
            ),
        ]
    return checks
