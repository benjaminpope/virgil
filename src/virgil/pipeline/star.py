"""`StarPipeline`: stellar angular diameter and limb darkening."""

from __future__ import annotations

import dataclasses
import math
import warnings
from typing import Callable

import numpy as onp

from .._precision import in_float64
from .._utils import rad2mas
from . import _checks, _io
from ._common import (
    _overview,
    _quicklook,
    _scales_of,
    _chi2_64,
    _m2loglike_64,
    _chain_health,
    _corner_trace,
    _diagnostics,
    _draws,
    _integer,
    _load_report,
    _load_start,
    _noise_quantiles,
    _pair,
    _pick,
    _prior_bounds,
    _prior_repr,
    _run_chains,
    _sampler_checks,
    _sampler_report,
    _save_map_correlation,
    _save_fit,
    _site_summary,
    _validate_common,
    _write_samples,
    _chi2,
    _geometry,
    _m2loglike_grid,
    _refine_diameter,
    _save,
)
from ._core import Stage, _Pipeline

# Points of the grid that refines a lobe's diameter, ±1 coarse step.
_REFINE_POINTS = 201
# Points and span (fractions of the lobe's diameter) of the grid whose best
# diameter starts a second fit, and on which a uniform disk's error scales
# are profiled.
_START_POINTS = 201
_START_SPAN = (0.75, 1.25)
# Fractional separation within which a scan point must be the lowest to be
# a lobe's minimum.
_LOBE_SEPARATION = 0.05
# The default diameter prior spans these multiples of the finest and the
# coarsest λ/B: well inside the unresolved regime, beyond the first null of
# the shortest baseline.
_DIAM_LOW_FACTOR = 0.1
_DIAM_HIGH_FACTOR = 4.0
# Δ(−2 log L) of a closure-phase flip at the data's precision.
_WALL_JUMP = 1000.0
# Posterior draws probed for hard walls, and the step in posterior sds.
_WALL_DRAWS = 40
_WALL_RADIUS = 2.0
# Points of the grid of θf (θ the diameter, f = B/λ) searched for a null,
# and its extent: the uniform disk's first null is at 1.22.
_NULL_POINTS = 4000
_NULL_REACH = 4.0


# === THE MODEL REGISTRY ===


@dataclasses.dataclass(frozen=True)
class StarModel:
    """A model ``StarPipeline`` can fit, registered under a short name.

    A new model is a new entry in ``models()``.

    Attributes
    ----------
    name, label : str
        Short name of ``StarPipeline(model=...)`` (also its folder under
        ``models/``) and a human-readable name.
    classes : tuple of type
        Virgil classes recognized as templates, so an instance can be passed.
    build : callable
        ``build(values) -> model`` for a dict of parameter values.
    params : tuple of str
        Fitted parameter paths; the first is the angular diameter.
    start : dict
        Starting values of the parameters other than the diameter.
    priors : callable
        ``priors(diam_lo, diam_hi) -> dict`` of group-invariant priors.
    scan : bool
        Fit by a scan of the diameter alone, not by gradients.
    shape : tuple of str
        Parameters other than the diameter whose posterior widths are
        compared with their prior widths.
    derived : callable or None
        ``derived(samples) -> dict`` of extra reported quantities.
    parent : str or None
        The simpler model this one contains, for the comparison.
    check_stem : str or None
        Stem of the model's check names (default ``name``).
    """

    name: str
    label: str
    classes: tuple
    build: Callable
    params: tuple
    start: dict
    priors: Callable
    scan: bool = False
    shape: tuple = ()
    derived: Callable | None = None
    parent: str | None = None
    check_stem: str | None = None

    @property
    def n_params(self):
        return len(self.params)

    @property
    def gain_check(self):
        return f"{self.check_stem or self.name}_gain"

    @property
    def constrained_check(self):
        return f"{self.check_stem or self.name}_constrained"


def _uniform_disk(values):  # deliberate: registry callback
    from ..models import UniformDisk

    return UniformDisk(values["diam"])


def _quadratic_disk(values):
    from ..models import QuadraticLimbDarkenedDisk

    return QuadraticLimbDarkenedDisk(
        values["diam"], q1=values["q1"], q2=values["q2"]
    )


def _quadratic_derived(samples):  # deliberate: registry callback
    star = _quadratic_disk(samples)
    return {"u1": onp.asarray(star.u1), "u2": onp.asarray(star.u2)}


def _uniform_priors(lo, hi):
    import numpyro.distributions as dist

    return {"diam": dist.LogUniform(lo, hi)}


def _quadratic_priors(lo, hi):  # deliberate: registry callback
    import numpyro.distributions as dist

    return {
        **_uniform_priors(lo, hi),
        "q1": dist.Uniform(0.0, 1.0),
        "q2": dist.Uniform(0.0, 1.0),
    }


_MODELS = None


def models():
    """The registry of short names to [`StarModel`][virgil.pipeline.star.StarModel]."""
    global _MODELS
    if _MODELS is None:
        from ..models import QuadraticLimbDarkenedDisk, UniformDisk

        entries = (
            StarModel(
                name="uniform",
                label="uniform disk",
                classes=(UniformDisk,),
                build=_uniform_disk,
                params=("diam",),
                start={},
                priors=_uniform_priors,
                scan=True,
            ),
            StarModel(
                name="limb_darkened",
                label="limb-darkened disk",
                classes=(QuadraticLimbDarkenedDisk,),
                build=_quadratic_disk,
                params=("diam", "q1", "q2"),
                start={"q1": 0.5, "q2": 0.5},
                priors=_quadratic_priors,
                shape=("q1", "q2"),
                derived=_quadratic_derived,
                parent="uniform",
                check_stem="limb_darkening",
            ),
        )
        _MODELS = {e.name: e for e in entries}
    return _MODELS


# The models fitted and compared when ``model`` is not given.
DEFAULT_MODELS = ("uniform", "limb_darkened")


# === THE PIPELINE ===


class StarPipeline(_Pipeline):
    """Measure a star's angular diameter, with and without limb darkening.

    !!! warning "Provisional"
        May change in any 0.4.x release without a deprecation cycle.

    Stages: ``load``, ``overview``, ``fit`` (a scan of the uniform-disk
    diameter, then [`fit`][virgil.fitting.fit] of each other model from the
    scanned diameter), ``posterior`` (NUTS) and ``quicklook``. The default
    models are compared on −2 log L against a BIC penalty.
    ``res.summary["star"]["diam_mas"]`` is the preferred model's. All settings:
    [pipeline settings](../pipeline.md#starpipeline-settings).

    Parameters
    ----------
    data : OIData
        The data.
    model : str or SourceModel, optional
        ``"uniform"`` ([`UniformDisk`][virgil.models.UniformDisk]) or
        ``"limb_darkened"``
        ([`QuadraticLimbDarkenedDisk`][virgil.models.QuadraticLimbDarkenedDisk]),
        or an instance of one (its ``q1`` and ``q2`` start the fit). Default:
        both, compared.
    output : str or os.PathLike, optional
        Run folder (default ``"run"``).
    **settings
        Overrides of ``StarPipeline.defaults()``, e.g. ``diam_range_mas``
        (``[min, max]`` in mas, the log-uniform diameter prior; default a
        tenth of the finest λ/B to four times the coarsest), ``n_scan``,
        ``max_lobes``, ``error_scale`` (``"quoted"`` or ``"fit"``).
        Priors and errors: [conventions](../conventions.md).

    Examples
    --------
    >>> import virgil as vg  # doctest: +SKIP
    >>> from virgil.pipeline import StarPipeline  # doctest: +SKIP
    >>> res = StarPipeline(vg.OIData("star.oifits"), output="runs/star").run()  # doctest: +SKIP
    >>> res.summary["star"]["diam_mas"]  # doctest: +SKIP
    """

    NAME = "star"
    STABILITY = "provisional"
    STAGES = ("load", "overview", "fit", "posterior", "quicklook")
    _DEFAULTS = {
        "diam_range_mas": None,
        "n_scan": 2000,
        "max_lobes": 5,
        "wavel_range": None,
        "error_floor": None,
        "error_scale": "quoted",
        "num_warmup": 1000,
        "num_samples": 1000,
        "num_chains": 4,
        "chain_method": "vectorized",
        "seed": 0,
        "batch_size": None,
        "quicklook_execute": True,
    }

    def __init__(self, data, model=None, *, output="run", **settings):
        registry = models()
        starts = {}
        if model is None:
            names = list(DEFAULT_MODELS)
            template = None
        elif isinstance(model, str):
            if model not in registry:
                raise ValueError(
                    f"model must be one of {sorted(registry)} or a model "
                    f"instance; got {model!r}."
                )
            names = [model]
            template = None
        else:
            entry = next(
                (e for e in registry.values() if type(model) in e.classes),
                None,
            )
            if entry is None:
                raise TypeError(
                    f"StarPipeline has no model for a {type(model).__name__}; "
                    f"pass one of the names {sorted(registry)} or a template "
                    "of the classes "
                    + ", ".join(
                        sorted(
                            c.__name__
                            for e in registry.values()
                            for c in e.classes
                        )
                    )
                    + "."
                )
            names = [entry.name]
            template = model
            starts[entry.name] = {
                k: float(onp.asarray(getattr(model, k)))
                for k in entry.params
                if k != "diam"
            }
        self.names = names
        self._starts = {
            n: {**registry[n].start, **starts.get(n, {})} for n in names
        }
        if template is None:
            template = registry[names[-1]].build(
                {"diam": 1.0, **self._starts[names[-1]]}
            )
        super().__init__(data, template, output=output, **settings)

    @classmethod
    def _validate(cls, s):
        if s["diam_range_mas"] is not None:
            _pair(s, "diam_range_mas")
        _integer(s, "n_scan", 20)
        _integer(s, "max_lobes", 1)
        if s["error_scale"] not in ("quoted", "fit"):
            raise ValueError('error_scale must be "quoted" or "fit".')
        _validate_common(s)

    def _model_digest(self):
        record = super()._model_digest()
        record["models"] = list(self.names)
        return record

    def _diam_range(self):
        """Bounds of the log-uniform diameter prior (mas)."""
        given = self.settings["diam_range_mas"]
        if given is not None:
            return float(given[0]), float(given[1])
        geo = _geometry(self.processed)
        return (
            _DIAM_LOW_FACTOR * geo["lambda_over_b_mas"],
            _DIAM_HIGH_FACTOR * geo["fov_mas"],
        )

    def _stages(self):
        fit_outputs = [
            "grids.h5",
            "models/best/manifest.json",
            "models/best/values.npz",
            "models/best/info.json",
            "plots/fit_correlation.png",
            "plots/fit_v2.png",
        ]
        for name in self.names:
            fit_outputs += [
                f"models/{name}/manifest.json",
                f"models/{name}/values.npz",
                f"models/{name}/info.json",
            ]
        return (
            Stage("load", _load_star, ("stages/load/report.json",)),
            Stage(
                "overview",
                _overview,
                ("plots/overview_data.png", "plots/overview_uv.png"),
            ),
            Stage("fit", _fit, tuple(fit_outputs)),
            Stage(
                "posterior",
                _posterior,
                (
                    "samples.h5",
                    "plots/posterior_corner.png",
                    "plots/posterior_trace.png",
                ),
            ),
            Stage("quicklook", _quicklook, ("quicklook.ipynb",)),
        )

    def _summarize(self, reports):
        registry = models()
        sections, checks = {}, []
        load, fit, post = (
            reports.get("load"),
            reports.get("fit"),
            reports.get("posterior"),
        )
        if load:
            sections["data"] = _pick(load, _DATA_KEYS)
        if not fit:
            return sections, checks

        names = fit["names"]
        sections["chi2"] = {
            "n_independent": fit["n_independent"],
            "error_scale": self.settings["error_scale"],
            "reduced": {k: fit["models"][k]["chi2_reduced"] for k in names},
            "error_scales": {
                k: fit["models"][k]["error_scales"] for k in names
            },
        }
        sections["fit"] = {
            "best": fit["best"],
            "scan_diam_mas": fit["scan_diam_mas"],
            "diam_bounds_mas": fit["diam_bounds_mas"],
            "rivals": fit["rivals"],
            "models": {
                k: {f: fit["models"][k][f] for f in _FIT_KEYS} for k in names
            },
        }
        if fit["comparison"]:
            sections["comparison"] = dict(fit["comparison"])
            sections["comparisons"] = [dict(c) for c in fit["comparisons"]]
        checks += _fit_checks(fit, registry)
        best = fit["best"]
        if post:  # the star's diameter: from the posterior if there is one
            row = post["models"][best]["params"]["diam"]
            diam, diam_err = row["median"], 0.5 * (row["q84"] - row["q16"])
        else:
            diam, diam_err = fit["models"][best]["params"]["diam"], None
        sections["star"] = {
            "model": best,
            "source": "posterior" if post else "fit",
            "diam_mas": diam,
            "diam_err_mas": diam_err,
        }
        if load:
            null = fit["models"][best]["first_null"]
            check = _checks.resolution_regime(
                diam, load["freq_max_per_rad"], first_null=null
            )
            sections["resolution"] = {
                "diam_mas": diam,
                "freq_max_per_rad": load["freq_max_per_rad"],
                "first_null": null,
                "first_null_fraction": check.value,
            }
            checks.append(check)
        checks.append(_checks.multimodal(fit["models"][best]["lobes"]))
        if post:
            entry = registry[best]
            if entry.shape:  # the posterior's width against the prior's
                rows = {
                    k: post["models"][best]["params"][k] for k in entry.shape
                }
                prior_std = fit["models"][best]["prior_std"]
                ratios = {
                    k: rows[k]["std"] / prior_std[k] for k in entry.shape
                }
                sections["shape"] = {
                    "model": best,
                    "params": {
                        k: {
                            "median": rows[k]["median"],
                            "posterior_std": rows[k]["std"],
                            "prior_std": prior_std[k],
                            "ratio": ratios[k],
                        }
                        for k in entry.shape
                    },
                }
                checks.append(
                    _checks.prior_constrained(entry.constrained_check, ratios)
                )
            sections["posterior"] = _pick(post, _POSTERIOR_KEYS)
            chosen = post["models"][best]
            bounds = {
                f"{best}.{k}": v
                for k, v in chosen["prior_bound_fraction"].items()
            }
            checks += _sampler_checks(
                post, bounds=bounds, walls=chosen.get("wall_probe")
            )
        return sections, checks


# === PLOTS ===


def plot_v2_models(data, fitted, ax=None, curves=400):
    """The squared visibilities against spatial frequency, with model curves.

    Parameters
    ----------
    data : OIData
        The data (its ``vis`` are plotted with their errors).
    fitted : dict[str, SourceModel]
        Models to draw, by label, as curves over the data's range of
        spatial frequency.
    ax : matplotlib axis, optional
    curves : int, optional
        Points of each curve.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    u = onp.asarray(data.u, dtype=float)
    wavel = onp.broadcast_to(onp.asarray(data.wavel, dtype=float), u.shape)
    freq = onp.hypot(u, onp.asarray(data.v, dtype=float)) / wavel
    power = 2 if data.v2_flag else 1
    ax.errorbar(
        freq / 1e6,
        onp.asarray(data.vis),
        onp.asarray(data.d_vis),
        fmt=".",
        color="0.4",
        label="data",
    )
    grid = onp.linspace(1e-3, 1.05 * freq.max(), curves)
    for label, model in fitted.items():
        vis = onp.abs(onp.asarray(model.model(grid, 0.0 * grid, 1.0)))
        ax.plot(grid / 1e6, vis**power, label=label)
    positive = onp.asarray(data.vis)[onp.asarray(data.vis) > 0]
    floor = min(1e-4, float(positive.min()) / 3.0) if positive.size else 1e-4
    ax.set(
        yscale="log",
        ylim=(floor, 1.5),
        xlabel="Spatial frequency (Mλ)",
        ylabel="$V^2$" if data.v2_flag else "$|V|$",
    )
    ax.legend()
    return ax.figure


# === STAGES ===


def _first_null(model, diam_mas, n=_NULL_POINTS):  # deliberate: test seam
    """θB/λ at the model's first visibility sign change (else uniform-disk)."""
    scale = float(rad2mas) / float(diam_mas)  # f for θf = 1
    grid = onp.linspace(1e-3, _NULL_REACH, n) * scale
    vis = onp.real(onp.asarray(model.model(grid, 0.0 * grid, 1.0)))
    flips = onp.flatnonzero(onp.sign(vis[1:]) * onp.sign(vis[:-1]) < 0)
    if not flips.size:
        return _checks.FIRST_NULL
    i = int(flips[0])
    frac = vis[i] / (vis[i] - vis[i + 1])
    return float((grid[i] + frac * (grid[i + 1] - grid[i])) / scale)


def _load_star(p):  # deliberate: stage function, stubbed by tests
    from .._geometry import fringe_scales

    # The longest spatial frequency B/λ in the data (cycles per radian).
    freq_max = float(rad2mas) / fringe_scales(p.processed)[0]
    return _load_report(p, freq_max_per_rad=freq_max)


# --- the diameter scan and its lobes ---

# A closure phase flips by 180 degrees where a model's visibility changes
# sign, so a χ² evaluated in float32 near a null can jump by orders of
# magnitude; ``fit`` and the grid tools are float64 inside, and so are the
# numbers this pipeline reports.


def _uniform_loglike(data, diameters, batch_size):
    """The uniform-disk log likelihood at each of ``diameters``."""
    import jax.numpy as np

    from ..grid_fit import likelihood_grid
    from ..models import UniformDisk

    return onp.asarray(
        likelihood_grid(
            UniformDisk,
            data=data,
            grid={"diam": np.asarray(diameters)},
            batch_size=batch_size,
        )
    )


@in_float64
def _scan(data, lo, hi, n_scan, batch_size):  # deliberate: float64 boundary
    """Uniform-disk scan: ``(axis, delta)``, the diameters and Δ(−2 log L)."""
    axis = onp.geomspace(lo, hi, n_scan)
    ll = _uniform_loglike(data, axis, batch_size)
    return axis, 2.0 * (onp.nanmax(ll) - ll)


def _find_lobes(
    axis, delta, lo, hi, cap, *, separation=_LOBE_SEPARATION
):  # deliberate: test seam
    """The ``cap`` lowest separate minima of the scan, as lobe dicts.

    Raises ``RuntimeError`` if the scan is not finite anywhere.
    """
    if not onp.isfinite(delta).any():
        raise RuntimeError(
            "The uniform-disk scan is not finite at any diameter in "
            f"[{lo:.4g}, {hi:.4g}] mas; check the data and diam_range_mas."
        )
    n = axis.size
    dlog = math.log(axis[-1] / axis[0]) / (n - 1)
    half = max(1, int(round(separation / dlog)))
    delta = onp.where(onp.isfinite(delta), delta, onp.inf)
    minima = []
    for i in range(n):
        window = delta[max(0, i - half) : i + half + 1]
        if onp.isfinite(delta[i]) and delta[i] <= window.min():
            if not minima or i - minima[-1] > half:
                minima.append(i)
            elif delta[i] < delta[minima[-1]]:
                minima[-1] = i
    keep = sorted(sorted(minima, key=lambda i: delta[i])[:cap])
    centres = [float(axis[i]) for i in keep]
    edges = [lo] + [math.sqrt(a * b) for a, b in zip(centres, centres[1:])]
    edges.append(hi)
    return [
        {
            "diam": c,
            "scan_delta_chi2": float(delta[i]),
            "bounds": (float(edges[k]), float(edges[k + 1])),
        }
        for k, (c, i) in enumerate(zip(centres, keep))
    ]


@in_float64
def _refine(
    data, centre, lo, hi, step, batch_size
):  # deliberate: float64 boundary
    """The uniform-disk diameter of best likelihood within one coarse step."""
    fine = onp.geomspace(
        max(lo, centre / step), min(hi, centre * step), _REFINE_POINTS
    )
    ll = _uniform_loglike(data, fine, batch_size)
    return float(fine[int(onp.nanargmax(ll))])


def _scan_lobes(
    data, lo, hi, n_scan, max_lobes, batch_size
):  # deliberate: test seam
    """Scan, find and refine the lobes: ``(axis, delta, lobes)``."""
    axis, delta = _scan(data, lo, hi, n_scan, batch_size)
    lobes = _find_lobes(axis, delta, lo, hi, max_lobes)
    step = (hi / lo) ** (1.0 / (n_scan - 1))
    for lobe in lobes:
        lobe["diam"] = _refine(data, lobe["diam"], lo, hi, step, batch_size)
    return axis, delta, lobes


# --- fitting one model in one lobe ---


@dataclasses.dataclass(frozen=True)
class LobeFit:
    """A model fitted in one diameter lobe.

    Attributes
    ----------
    model : SourceModel
        The fitted model.
    values, scales : dict
        Fitted values by path, and the fitted error scales (empty on the
        quoted errors).
    info : dict
        The fit's diagnostics.
    chi2, chi2_fitted : float
        χ² on the quoted errors, and at the fitted scales (N by
        construction; a diagnostic, not a score).
    m2loglike : float
        −2 log L at the fitted scales, the score that compares lobes and
        models.
    priors, lobe : dict
        The priors, and the lobe (``diam``, ``bounds``) it was fitted in.
    warnings : list
        Warnings the fit raised.
    """

    model: object
    values: dict
    info: dict
    chi2: float
    chi2_fitted: float
    m2loglike: float
    scales: dict
    priors: dict
    lobe: dict
    warnings: list


def _fit_scanned(p, entry, start, grid, noise):  # deliberate: test seam
    """A scan model's fit: the lobe's diameter (scales profiled)."""
    info = {
        "method": "scan",
        "converged": True,
        "n_scan": p.settings["n_scan"],
    }
    if noise is None:
        return entry.build(start), {"diam": onp.asarray(start["diam"])}, info
    data = p.processed
    m2loglike, grid_scales = _m2loglike_grid(
        entry.build, start, data, grid, noise, p.settings["batch_size"]
    )
    best = int(onp.nanargmin(m2loglike))
    low = grid[max(best - 1, 0)]
    high = grid[min(best + 1, grid.size - 1)]
    # Refine between grid neighbours rather than a gradient fit, which would
    # stall on the non-smooth chi-squared.
    diam, m2, scales = _refine_diameter(
        entry.build, start, data, low, high, noise
    )
    if not m2 <= m2loglike[best]:  # not unimodal between the neighbours
        diam = float(grid[best])
        scales = {k: float(v[best]) for k, v in grid_scales.items()}
    values = {"diam": onp.asarray(diam)}
    values.update({f"noise.{k}": v for k, v in scales.items()})
    model = entry.build({**start, "diam": diam})
    return model, values, {**info, "scales": "profiled"}


def _fit_lobe(p, entry, lobe, noise):
    """Fit ``entry`` in ``lobe`` and score it: a ``LobeFit``."""
    from ..fitting import fit

    data = p.processed
    priors = entry.priors(*lobe["bounds"])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        best_diam = lobe["diam"]
        start = {"diam": best_diam, **p._starts[entry.name]}
        low, high = float(priors["diam"].low), float(priors["diam"].high)
        grid = onp.geomspace(
            max(low, _START_SPAN[0] * best_diam),
            min(high, _START_SPAN[1] * best_diam),
            _START_POINTS,
        )  # diameters around the best one, within the prior
        if entry.scan:
            model, values, info = _fit_scanned(p, entry, start, grid, noise)
        else:
            # A gradient fit from the lobe's diameter and the grid's best.
            m2loglike, _ = _m2loglike_grid(
                entry.build, start, data, grid, None, p.settings["batch_size"]
            )
            # Two starts: past a null chi-squared is a union of thin
            # slivers, and a local fit started on the wrong one stalls.
            starts = [
                start["diam"],
                float(grid[int(onp.nanargmin(m2loglike))]),
            ]
            best = None
            for diam in dict.fromkeys(starts):
                template = entry.build({**start, "diam": diam})
                with warnings.catch_warnings(record=True) as fit_caught:
                    warnings.simplefilter("always")
                    result = fit(template, priors, data, noise=noise)
                score = _m2loglike_64(
                    result.model, data, **_scales_of(result.values)
                )
                if best is None or score < best[0]:
                    best = (score, result, fit_caught)
            _, result, fit_caught = best
            _rewarn(fit_caught)
            model = result.model
            values = {k: onp.asarray(v) for k, v in result.values.items()}
            info = _io.clean_json(dict(result.info))
    if (
        info.get("method") == "scan"
    ):  # no fit problem to ask: test the diameter
        from ..fitting import _AT_BOUND, _bound_fraction

        edge = _bound_fraction(priors["diam"], values["diam"])
        if edge is not None and not _AT_BOUND < edge < 1 - _AT_BOUND:
            info = {**info, "at_bound": ("diam",)}
    chi2, _ = _chi2_64(model, data)
    scales = _scales_of(values)
    chi2_fitted = _chi2_64(model, data, **scales)[0] if scales else chi2
    return LobeFit(
        model=model,
        values=values,
        info=info,
        chi2=float(chi2),
        chi2_fitted=float(chi2_fitted),
        m2loglike=float(_m2loglike_64(model, data, **scales)),
        scales=scales,
        priors=priors,
        lobe=lobe,
        warnings=caught,
    )


def _finite_or(value, default):
    return value if math.isfinite(value) else default


def _fit_in_lobes(p, entry, lobes, noise):  # deliberate: named step of _fit
    """Fit ``entry`` in every lobe: ``(best LobeFit, report)``."""
    n = int(p.processed.n_independent)
    fits = [_fit_lobe(p, entry, lobe, noise) for lobe in lobes]
    best = min(fits, key=lambda f: _finite_or(f.m2loglike, math.inf))
    # Only the best lobe's fit warns: a fit that stalls in an alias lobe,
    # whose χ² is thousands above the best, is not a problem.
    _rewarn(best.warnings)
    # One row per lobe, with its −2 log L above the best lobe's.
    table = [
        {
            "diam_bounds_mas": list(f.lobe["bounds"]),
            "diam_mas": float(onp.asarray(f.values["diam"])),
            "chi2": f.chi2,
            "chi2_reduced": f.chi2 / n,
            "m2loglike": f.m2loglike,
            "delta_m2loglike": f.m2loglike - best.m2loglike,
            "delta_chi2": f.m2loglike - best.m2loglike,
        }
        for f in fits
    ]
    params = {k: float(onp.asarray(best.values[k])) for k in entry.params}
    if entry.derived:
        derived = entry.derived(
            {k: onp.asarray(best.values[k]) for k in entry.params}
        )
        params.update({k: float(v) for k, v in derived.items()})
    return best, {
        "params": params,
        "converged": best.info.get("converged"),
        "chi2": best.chi2,
        "chi2_fitted": best.chi2_fitted,
        "m2loglike": best.m2loglike,
        "first_null": _first_null(best.model, float(best.values["diam"])),
        "diam_bounds_mas": list(best.lobe["bounds"]),
        "lobes": table,
        "chi2_reduced": best.chi2 / n,
        "error_scales": best.scales or None,
        "at_bound": list(best.info.get("at_bound", []) or []),
        "priors": {k: _prior_repr(v) for k, v in best.priors.items()},
        "prior_std": {
            k: float(onp.sqrt(best.priors[k].variance)) for k in entry.shape
        },
    }


# --- choosing between models ---


def _compare(registry, reports, child, n):
    """The comparison of a model with its parent: gain against BIC penalty."""
    parent = registry[child].parent
    gain = reports[parent]["m2loglike"] - reports[child]["m2loglike"]
    n_extra = registry[child].n_params - registry[parent].n_params
    penalty = _checks.bic_penalty(n_extra, n)
    return {
        "simple": parent,
        "complex": child,
        "delta_m2loglike": gain,
        "delta_chi2": gain,
        "n_extra_params": n_extra,
        "bic_penalty": penalty,
        "preferred": child if gain > penalty else parent,
    }


def _choose_model(reports, registry, n):  # deliberate: test seam
    """Walk down the chain of parents to ``(best, decisive, comparisons)``."""
    comparisons = [
        _compare(registry, reports, name, n)
        for name in reports
        if registry[name].parent in reports
    ]
    roots = [k for k in reports if registry[k].parent not in reports]
    best = min(
        roots, key=lambda k: _finite_or(reports[k]["m2loglike"], math.inf)
    )
    decisive = None
    # Deliberate: a deeper model is reached only through a preferred
    # intermediate, never compared with the root directly.
    while True:
        children = [c for c in comparisons if c["simple"] == best]
        if not children:
            return best, decisive, comparisons
        decisive = max(
            children, key=lambda c: _finite_or(c["delta_m2loglike"], -math.inf)
        )
        if decisive["preferred"] == best:
            return best, decisive, comparisons
        best = decisive["complex"]


# --- the fit stage ---


def _fit(p):  # deliberate: stage function, stubbed by tests
    data, s = p.processed, p.settings
    lo, hi = p._diam_range()
    registry = models()
    axis, delta, lobes = _scan_lobes(
        data, lo, hi, s["n_scan"], s["max_lobes"], s["batch_size"]
    )
    _io.write_h5(
        p.output / "grids.h5",
        {
            "axes": {"diam": (axis, {"units": "mas"})},
            "scan": {"delta_chi2": delta},
        },
        attrs={"/": {"axis_order": ["diam"], "schema": _io.SCHEMA}},
    )

    # Each lobe is fitted for each model; the best lobe gives the model's
    # fit and (in the posterior) its bounded, group-invariant log-uniform
    # diameter prior. The lobe table is recorded in the summary.
    noise = p._noise()
    fits, reports = {}, {}

    for name in p.names:
        fits[name], reports[name] = _fit_in_lobes(
            p, registry[name], lobes, noise
        )
        _save_fit(
            p.output / "models" / name, fits[name], list(registry[name].params)
        )
    n = int(data.n_independent)
    best, decisive, comparisons = _choose_model(reports, registry, n)
    _save_fit(
        p.output / "models" / "best", fits[best], list(registry[best].params)
    )
    _save_map_correlation(
        p, fits[best].model, fits[best].values, list(registry[best].params)
    )
    fig = plot_v2_models(
        p.processed, {registry[n].label: fits[n].model for n in p.names}
    )
    _save(fig, p.output / "plots" / "fit_v2.png")
    scan_best = min(lobes, key=lambda lobe: lobe["scan_delta_chi2"])
    return {
        "names": list(p.names),
        "best": best,
        "n_independent": n,
        "diam_range_mas": [lo, hi],
        "diam_bounds_mas": reports[best]["diam_bounds_mas"],
        "rivals": [
            [row["diam_mas"], row["delta_chi2"]]
            for row in reports[best]["lobes"]
            if 0 < row["delta_chi2"] <= _checks.RIVAL_DELTA_CHI2
        ],
        "scan_diam_mas": scan_best["diam"],
        "models": reports,
        "comparison": decisive,
        "comparisons": comparisons,
    }


def _bound_fractions(entry, priors, noise, samples):  # deliberate: test seam
    """Fraction of samples within 1% of either prior bound."""
    # Shape priors are meant to matter; the ``constrained`` check judges them.
    sampled = {k: v for k, v in priors.items() if k not in entry.shape}
    sampled.update({f"noise.{k}": v for k, v in (noise or {}).items()})
    return _prior_bounds(sampled, samples)


def _rewarn(caught):
    """Re-raise the recorded warnings of the fit that was kept."""
    for w in caught:
        warnings.warn_explicit(
            w.message, w.category, w.filename, w.lineno, source=w.source
        )


def _wall_probe(  # deliberate: test seam
    entry, data, samples, priors, n=_WALL_DRAWS, radius=_WALL_RADIUS, seed=0
):
    """Fraction of random steps of ``radius`` posterior sd that hit a χ² wall.

    Returns ``{"radius_sd", "n", "exposed"}``; a hit is χ² rising > ``_WALL_JUMP``.
    """
    names = list(entry.params)
    flat = {k: onp.asarray(samples[k], dtype=float).ravel() for k in names}
    sd = onp.array([flat[k].std() for k in names])
    rng = onp.random.default_rng(seed)
    draws = rng.choice(flat[names[0]].size, size=n, replace=True)
    hits = valid = 0
    for i in draws:
        base = onp.array([flat[k][i] for k in names])
        direction = rng.normal(size=len(names))
        trial = base + radius * sd * direction / onp.linalg.norm(direction)
        inside = all(
            float(priors[k].low) < v < float(priors[k].high)
            for k, v in zip(names, trial)
            if hasattr(priors.get(k), "low")
        )
        if not inside:
            continue
        chi0 = _chi2(entry.build(dict(zip(names, base))), data)[0]
        chi1 = _chi2(entry.build(dict(zip(names, trial))), data)[0]
        valid += 1
        hits += bool(float(chi1 - chi0) > _WALL_JUMP)
    return {
        "radius_sd": radius,
        "n": valid,
        "exposed": hits / valid if valid else 0.0,
    }


@in_float64
def _nuts(model, data, p, name, priors, start):  # deliberate: float64 boundary
    """NUTS for one fitted model: ``(samples, stats, summary)``."""
    from ..likelihood import numpyro_model

    entry = models()[name]
    noise = p._noise()
    start = dict(start)
    for k in entry.shape:
        if k in start:
            val = float(onp.asarray(start[k]))
            if val < 0.02 or val > 0.98:
                start[k] = onp.asarray(onp.clip(val, 0.05, 0.95))
    posterior = numpyro_model(model, priors, data, noise=noise)
    mcmc = _run_chains(
        posterior, p.settings, start, dense_mass=True, priors=priors
    )
    samples, stats = _draws(mcmc)
    diagnostics = {site: _diagnostics([x]) for site, x in samples.items()}
    if entry.derived:
        samples.update(entry.derived(samples))
    reported = list(entry.params) + list(
        k
        for k in samples
        if k not in entry.params and not k.startswith("noise.")
    )
    summary = {
        k: _site_summary(samples[k], diagnostics.get(k, {})) for k in reported
    }
    return (
        samples,
        stats,
        {
            "params": summary,
            "noise": _noise_quantiles(samples),
            **_chain_health(diagnostics, stats),
            "wall_probe": _wall_probe(entry, data, samples, priors),
            "prior_bound_fraction": _bound_fractions(
                entry, priors, noise, samples
            ),
        },
    )


def _posterior(p):  # deliberate: stage function, stubbed by tests
    fit_report = p._report("fit")
    best = fit_report["best"]
    groups, per_model, params_attr = {}, {}, {}
    for name in p.names:
        folder = p.output / "models" / name
        priors = models()[name].priors(
            *fit_report["models"][name]["diam_bounds_mas"]
        )
        samples, stats, summary = _nuts(
            _io.load_model(folder),
            p.processed,
            p,
            name,
            priors,
            _load_start(folder, priors),
        )
        groups[f"posterior_{name}"] = samples
        groups[f"sample_stats_{name}"] = stats
        per_model[name] = summary
        params_attr[name] = list(summary["params"])
        if name == best:
            groups["posterior"], groups["sample_stats"] = samples, stats
    _write_samples(
        p,
        groups,
        best=best,
        params=params_attr,
        priors={n: fit_report["models"][n]["priors"] for n in p.names},
    )
    sampled = list(models()[best].params)
    _corner_trace(p.output, groups["posterior"], sampled)
    return {
        "models": per_model,
        "best": best,
        "sampled": sampled,
        **_sampler_report(p.settings, [per_model[best]]),
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
    "freq_max_per_rad",
)


_FIT_KEYS = (
    "params",
    "converged",
    "chi2",
    "chi2_fitted",
    "m2loglike",
    "chi2_reduced",
    "diam_bounds_mas",
    "lobes",
    "at_bound",
)
_POSTERIOR_KEYS = (
    "models",
    "best",
    "sampled",
    "num_chains",
    "num_samples",
    "num_warmup",
    "r_hat_max",
    "ess_bulk_min",
    "divergence_fraction",
)


def _fit_checks(fit, registry):  # deliberate: named step of _summarize
    """Checks on the fits: χ²/N, model gains and the fitted error scales."""
    reports = fit["models"]
    checks = [
        _checks.chi2_reduced(
            f"chi2_{k}",
            reports[k]["chi2_reduced"],
            label=f"the {registry[k].label}",
        )
        for k in fit["names"]
    ]
    for comparison in fit["comparisons"]:
        simple, complex_ = comparison["simple"], comparison["complex"]
        checks.append(
            _checks.model_gain(
                registry[complex_].gain_check,
                reports[simple]["m2loglike"],
                reports[complex_]["m2loglike"],
                fit["n_independent"],
                n_extra=comparison["n_extra_params"],
                simple=registry[simple].label,
                complex_=registry[complex_].label,
            )
        )
    scales = {
        f"{k}.{kk}": v
        for k in fit["names"]
        for kk, v in (reports[k]["error_scales"] or {}).items()
    }
    if scales:
        checks.append(_checks.error_scale(scales))
    at_bound = [
        f"{k}.{site}" for k in fit["names"] for site in reports[k]["at_bound"]
    ]
    if at_bound:
        checks.append(_checks.at_bound(at_bound, start=True))
    return checks
