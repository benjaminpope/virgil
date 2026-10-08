"""`StarPipeline`: stellar angular diameter and limb darkening."""

from __future__ import annotations

import dataclasses
import math
import types
from typing import Callable

import numpy as np

from . import _checks, _io
from ._core import Stage, _Pipeline
from .binary import (
    _BOUND_FRACTION,
    _chi2,
    _geometry,
    _moments,
    _quantiles,
    _save,
)

MAS_PER_RAD = 180.0 / math.pi * 3600.0 * 1000.0


# === THE MODEL REGISTRY ===


@dataclasses.dataclass(frozen=True)
class StarModel:
    """A model ``StarPipeline`` can fit, registered under a short name.

    Adding a model to the pipeline is adding an entry to ``MODELS``: the
    stages, summary and checks are written against this description.

    Attributes
    ----------
    name : str
        The short name accepted by ``StarPipeline(model=...)``; also the
        name of the model's folder under ``models/``.
    label : str
        Human-readable name, for plots and messages.
    classes : tuple of type
        Virgil classes recognised as templates of this entry, so that a
        model instance can be passed instead of the name.
    build : callable
        ``build(values) -> model``: the model for a dict of parameter
        values (every path in ``params``).
    params : tuple of str
        The fitted parameter paths; the first is the angular diameter.
    start : dict
        Starting values of the parameters other than the diameter.
    priors : callable
        ``priors(diam_lo, diam_hi) -> dict`` of group-invariant numpyro
        priors on ``params`` (log-uniform diameter).
    scan : bool
        Whether the model is fitted by a scan of the diameter alone rather
        than a gradient-based fit (a uniform disk's closure phases flip
        between 0 and 180 degrees at each null, so its χ² is not smooth).
    shape : tuple of str
        Parameters, other than the diameter, that shape the star (e.g. the
        limb-darkening coefficients); the pipeline compares their posterior
        widths with their prior widths.
    derived : callable or None
        ``derived(samples) -> dict`` of extra reported quantities computed
        from sampled values (for example ``u1`` and ``u2`` from ``q1`` and
        ``q2``).
    parent : str or None
        The name of the simpler model this one contains, for the Δχ²
        comparison (None if it contains none).
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

    @property
    def n_params(self):
        return len(self.params)


def _uniform_disk(values):
    from ..models import UniformDisk

    return UniformDisk(values["diam"])


def _quadratic_disk(values):
    from ..models import QuadraticLimbDarkenedDisk

    return QuadraticLimbDarkenedDisk(
        values["diam"], q1=values["q1"], q2=values["q2"]
    )


def _quadratic_derived(samples):
    from ..models import QuadraticLimbDarkenedDisk

    star = QuadraticLimbDarkenedDisk(
        samples["diam"], q1=samples["q1"], q2=samples["q2"]
    )
    return {"u1": np.asarray(star.u1), "u2": np.asarray(star.u2)}


def _diam_prior(lo, hi):
    import numpyro.distributions as dist

    return dist.LogUniform(lo, hi)


def _uniform_priors(lo, hi):
    return {"diam": _diam_prior(lo, hi)}


def _quadratic_priors(lo, hi):
    import numpyro.distributions as dist

    return {
        "diam": _diam_prior(lo, hi),
        "q1": dist.Uniform(0.0, 1.0),
        "q2": dist.Uniform(0.0, 1.0),
    }


def _registry():
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
        ),
    )
    return {e.name: e for e in entries}


_MODELS = None


def models():
    """The registry of short names to [`StarModel`][virgil.pipeline.star.StarModel]."""
    global _MODELS
    if _MODELS is None:
        _MODELS = _registry()
    return _MODELS


# The models fitted and compared when ``model`` is not given.
DEFAULT_MODELS = ("uniform", "limb_darkened")


def _entry_for(instance):
    for entry in models().values():
        if type(instance) in entry.classes:
            return entry
    raise TypeError(
        f"StarPipeline has no model for a {type(instance).__name__}; pass "
        f"one of the names {sorted(models())} or a template of the classes "
        + ", ".join(
            sorted(c.__name__ for e in models().values() for c in e.classes)
        )
        + "."
    )


# === THE PIPELINE ===


class StarPipeline(_Pipeline):
    """Measure a star's angular diameter, with and without limb darkening.

    Stages: ``load → overview → fit → posterior → quicklook``.

    * ``load`` applies ``wavel_range`` and ``error_floor`` and writes the
      data that are fitted to ``data/processed.oifits``.
    * ``overview`` plots the data and their uv coverage.
    * ``fit`` scans the uniform-disk diameter (a uniform disk's closure
      phases flip at every null, so its χ² is not smooth and a scan is
      the reliable fit), then fits each other model with
      [`fit`][virgil.fitting.fit] from the scanned diameter. With the
      default ``model`` it compares the limb-darkened and uniform fits by
      their Δχ² on the quoted errors against a BIC penalty.
    * ``posterior`` samples every fitted model with NUTS
      ([`numpyro_model`][virgil.likelihood.numpyro_model]).
    * ``quicklook`` writes and executes ``quicklook.ipynb``: the main
      result is V² against spatial frequency with the model curves.

    The fit is on the quoted errors unless ``error_scale="fit"``, and χ²/N
    is always reported on the quoted errors. Priors are group-invariant:
    log-uniform in the diameter over ``diam_range_mas`` and uniform on
    [0, 1] in Kipping's limb-darkening coefficients ``q1``, ``q2``, which
    cover exactly the physical quadratic laws.

    ``res.model()`` is the preferred model (the limb-darkened one only if
    its Δχ² exceeds the BIC penalty of its two extra parameters);
    ``res.model("uniform")`` and ``res.model("limb_darkened")`` give each.

    Parameters
    ----------
    data : OIData
        The data.
    model : str or SourceModel, optional
        What to fit. A short name, ``"uniform"`` (a
        [`UniformDisk`][virgil.models.UniformDisk]) or ``"limb_darkened"``
        (a [`QuadraticLimbDarkenedDisk`][virgil.models.QuadraticLimbDarkenedDisk]
        in Kipping's q1, q2), fits that model alone. An instance of one of
        those classes fits that model, with its ``q1`` and ``q2`` as
        starting values. By default both are fitted and compared.
    output : str or os.PathLike, optional
        Run folder (default ``"run"``).
    **settings
        Overrides of the defaults (see ``StarPipeline.defaults()``):

        - ``diam_range_mas`` (None): ``[min, max]`` of the log-uniform
          diameter prior; by default from 0.1 λ_min/B_max (well inside
          the unresolved regime) to 4 λ_max/B_min (beyond the first null
          on every baseline).
        - ``n_scan`` (2000): points of the log-spaced diameter scan; a
          second scan of 201 points over ±1% then refines it.
        - ``max_lobes`` (5): the most diameter lobes (separate minima of
          the scan) fitted for each model. The best lobe gives the fit
          and bounds the posterior's diameter prior; the table of lobes
          is in ``summary["fit"]["models"][name]["lobes"]``.
        - ``wavel_range`` (None): ``[min, max]`` wavelengths to keep (m).
        - ``error_floor`` (None): absolute error floors by observable,
          e.g. ``{"vis": 0.01, "phi": 0.005}``, as
          [`OIData.with_error_floor`][virgil.oidata.OIData.with_error_floor].
        - ``error_scale`` ("quoted"): ``"fit"`` also fits log-uniform
          error scales ``vis_scale`` and ``phi_scale`` on [0.1, 10].
        - ``num_warmup``, ``num_samples``, ``num_chains`` (1000, 1000, 4),
          ``chain_method`` ("vectorized") and ``seed`` (0): NUTS.
        - ``batch_size`` (None): scan batch size.
        - ``quicklook_execute`` (True): execute the quicklook notebook.

    Examples
    --------
    >>> import virgil as vg  # doctest: +SKIP
    >>> from virgil.pipeline import StarPipeline  # doctest: +SKIP
    >>> res = StarPipeline(vg.OIData("star.oifits"), output="runs/star").run()  # doctest: +SKIP
    >>> res.summary["star"]["diam_mas"]  # doctest: +SKIP
    """

    NAME = "star"
    STABILITY = "stable"
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
            entry = _entry_for(model)
            names = [entry.name]
            template = model
            starts[entry.name] = {
                k: float(np.asarray(getattr(model, k)))
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

    def _default_model(self):
        raise NotImplementedError  # a template is always built in __init__

    @classmethod
    def _validate(cls, s):
        if s["diam_range_mas"] is not None:
            rng = s["diam_range_mas"]
            if len(rng) != 2 or not 0 < rng[0] < rng[1]:
                raise ValueError(
                    "diam_range_mas must be [min, max] with 0 < min < max."
                )
        if not (isinstance(s["n_scan"], int) and s["n_scan"] >= 20):
            raise ValueError("n_scan must be an integer >= 20.")
        if not (isinstance(s["max_lobes"], int) and s["max_lobes"] >= 1):
            raise ValueError("max_lobes must be an integer >= 1.")
        for name in ("num_samples", "num_chains"):
            if not (isinstance(s[name], int) and s[name] >= 1):
                raise ValueError(f"{name} must be an integer >= 1.")
        if not (isinstance(s["num_warmup"], int) and s["num_warmup"] >= 0):
            raise ValueError("num_warmup must be an integer >= 0.")
        if s["batch_size"] is not None and not s["batch_size"] > 0:
            raise ValueError("batch_size must be positive.")
        if s["error_scale"] not in ("quoted", "fit"):
            raise ValueError('error_scale must be "quoted" or "fit".')
        if s["chain_method"] not in ("vectorized", "sequential", "parallel"):
            raise ValueError(
                'chain_method must be "vectorized", "sequential" or "parallel".'
            )
        if s["wavel_range"] is not None and len(s["wavel_range"]) != 2:
            raise ValueError("wavel_range must be [min, max] in metres.")
        if s["error_floor"] is not None and not isinstance(
            s["error_floor"], dict
        ):
            raise ValueError("error_floor must be a dict of absolute floors.")

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
            0.1 * geo["lambda_over_b_mas"],
            4.0 * geo["fov_mas"],
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
            Stage("load", _load, ("stages/load/report.json",)),
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

    def _summarise(self, reports):
        return _summarise(self.settings, reports)


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
    u, v = np.asarray(data.u, dtype=float), np.asarray(data.v, dtype=float)
    wavel = np.broadcast_to(np.asarray(data.wavel, dtype=float), u.shape)
    freq = np.hypot(u, v) / wavel
    power = 2 if data.v2_flag else 1
    ax.errorbar(
        freq / 1e6,
        np.asarray(data.vis),
        np.asarray(data.d_vis),
        fmt=".",
        color="0.4",
        label="data",
    )
    grid = np.linspace(1e-3, 1.05 * freq.max(), curves)
    for label, model in fitted.items():
        vis = np.abs(np.asarray(model.model(grid, 0.0 * grid, 1.0)))
        ax.plot(grid / 1e6, vis**power, label=label)
    positive = np.asarray(data.vis)[np.asarray(data.vis) > 0]
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


def _freq_max(data):
    """The longest spatial frequency B/λ in the data (cycles per radian)."""
    u, v = np.asarray(data.u, dtype=float), np.asarray(data.v, dtype=float)
    wavel = np.broadcast_to(np.asarray(data.wavel, dtype=float), u.shape)
    return float(np.max(np.hypot(u, v) / wavel))


def _first_null(model, diam_mas, n=4000):
    """θB/λ at the first sign change of the model visibility.

    The visibility is evaluated on a one-dimensional grid of B/λ out to
    four times the uniform-disk null. Falls back to the uniform-disk value
    when the model has no null in range.
    """
    scale = 3.6e6 * 180.0 / math.pi / float(diam_mas)  # f for θf = 1
    grid = np.linspace(1e-3, 4.0, n) * scale
    vis = np.real(np.asarray(model.model(grid, 0.0 * grid, 1.0)))
    flips = np.flatnonzero(np.sign(vis[1:]) * np.sign(vis[:-1]) < 0)
    if not flips.size:
        return _checks.FIRST_NULL
    i = int(flips[0])
    frac = vis[i] / (vis[i] - vis[i + 1])
    return float((grid[i] + frac * (grid[i + 1] - grid[i])) / scale)


def _load(p):
    from ..oifits import write_oifits

    data = p.processed
    tables, reason = _io.oidata_tables(data)
    report = {
        "n_vis": int(np.asarray(data.vis).size),
        "n_phi": int(np.asarray(data.phi).size),
        "n_independent": int(data.n_independent),
        "closure_phases": bool(data.cp_flag),
        **_geometry(data),
        "freq_max_per_rad": _freq_max(data),
        "data_sha256": _io.data_fingerprint(data),
        "processed_oifits_skipped": reason,
    }
    if tables is not None:
        path = p.output / "data" / "processed.oifits"
        with _io.atomic_path(path) as tmp:
            write_oifits(tables, tmp)
    return report


def _overview(p):
    from ..plotting import plot_oidata_overview, plot_uv_coverage

    fig, _ = plot_oidata_overview(p.processed)
    _save(fig, p.output / "plots" / "overview_data.png")
    fig, _ = plot_uv_coverage(p.processed)
    _save(fig, p.output / "plots" / "overview_uv.png")
    return {}


def _float64(function):
    """Run ``function`` with 64-bit JAX and its pytree arguments cast to it.

    A closure phase flips by 180 degrees where a model's visibility changes
    sign, so a χ² evaluated in float32 near a null can jump by orders of
    magnitude; ``fit`` and the grid tools are float64 inside, and so are
    the numbers this pipeline reports.
    """
    import functools

    from .._precision import cast_tree, run_in

    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        with run_in("float64"):
            args = [cast_tree(a, "float64") for a in args]
            return function(*args, **kwargs)

    return wrapper


@_float64
def _chi2_64(model, data, **noise):
    return _chi2(model, data, **noise)


@_float64
def _scan(data, lo, hi, n_scan, batch_size):
    """Uniform-disk log-likelihood over the diameter: coarse, then refined."""
    import jax.numpy as jnp

    from ..grid_fit import likelihood_grid
    from ..models import UniformDisk

    kw = {"batch_size": batch_size}
    coarse = np.geomspace(lo, hi, n_scan)
    ll = np.asarray(
        likelihood_grid(UniformDisk, data, {"diam": jnp.asarray(coarse)}, **kw)
    )
    best = float(coarse[int(np.nanargmax(ll))])
    # One coarse step either side of the coarse optimum.
    step = (hi / lo) ** (1.0 / (n_scan - 1))
    fine = np.geomspace(max(lo, best / step), min(hi, best * step), 201)
    ll_fine = np.asarray(
        likelihood_grid(UniformDisk, data, {"diam": jnp.asarray(fine)}, **kw)
    )
    best = float(fine[int(np.nanargmax(ll_fine))])
    ll_top = float(max(np.nanmax(ll), np.nanmax(ll_fine)))
    return coarse, 2.0 * (ll_top - ll), best


def _find_lobes(axis, delta, lo, hi, cap, *, separation=0.05):
    """The scan's lobes: separate minima of the diameter scan.

    Past the first null the squared visibility repeats in lobes (and
    closure phases flip at its zeros), so the scan has several separate
    minima. A point is a minimum if it is the lowest within ``separation``
    (fractionally) either side; the ``cap`` lowest are kept, and the lobe
    boundaries are the geometric midpoints between neighbouring minima
    (``lo`` and ``hi`` at the ends).

    Returns
    -------
    list[dict]
        By increasing diameter: ``{"diam": scan minimum, "scan_delta_chi2":
        its excess over the best, "bounds": (low, high)}``.
    """
    n = axis.size
    dlog = math.log(axis[-1] / axis[0]) / (n - 1)
    half = max(1, int(round(separation / dlog)))
    delta = np.where(np.isfinite(delta), delta, np.inf)
    minima = []
    for i in range(n):
        window = delta[max(0, i - half) : i + half + 1]
        if np.isfinite(delta[i]) and delta[i] <= window.min():
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


@_float64
def _refine(data, centre, lo, hi, step, batch_size):
    """The uniform-disk diameter of best likelihood within one coarse step."""
    import jax.numpy as jnp

    from ..grid_fit import likelihood_grid
    from ..models import UniformDisk

    fine = np.geomspace(max(lo, centre / step), min(hi, centre * step), 201)
    ll = np.asarray(
        likelihood_grid(
            UniformDisk,
            data,
            {"diam": jnp.asarray(fine)},
            batch_size=batch_size,
        )
    )
    return float(fine[int(np.nanargmax(ll))])


def _fit_model(p, entry, priors, best_diam, noise):
    """MAP fit of one registered model: ``(model, values, info)``."""
    from ..fitting import fit

    start = {"diam": best_diam, **p._starts[entry.name]}
    template = entry.build(start)
    if entry.scan and noise is None:
        values = {"diam": np.asarray(best_diam)}
        info = {
            "method": "scan",
            "converged": True,
            "n_scan": p.settings["n_scan"],
        }
        return template, values, info
    # A limb-darkened star seen past its first null has closure phases that
    # flip at visibility nulls, so chi^2 is a union of thin slivers and a
    # local fit started on the wrong one stalls at a huge chi^2. Start from
    # the scan diameter and from the best of a small grid of diameters with
    # the starting shape, and keep the better fit.
    low, high = float(priors["diam"].low), float(priors["diam"].high)
    grid = np.geomspace(
        max(low, 0.75 * best_diam), min(high, 1.25 * best_diam), 201
    )
    chi2 = [
        float(_chi2_64(entry.build({**start, "diam": d}), p.processed)[0])
        for d in grid
    ]
    starts = [best_diam, float(grid[int(np.nanargmin(chi2))])]
    best = None
    for diam in dict.fromkeys(starts):
        tmpl = entry.build({**start, "diam": diam})
        result = fit(tmpl, priors, p.processed, noise=noise)
        score = float(_chi2_64(result.model, p.processed)[0])
        if best is None or score < best[0]:
            best = (score, result)
    result = best[1]
    values = {k: np.asarray(v) for k, v in result.values.items()}
    return result.model, values, _io.clean_json(dict(result.info))


def _fit(p):
    from ..likelihood import posterior_predictive_summary
    from ..plotting import plot_data_model_correlation

    data, s = p.processed, p.settings
    lo, hi = p._diam_range()
    registry = models()
    axis, delta, best_diam = _scan(data, lo, hi, s["n_scan"], s["batch_size"])
    _io.write_h5(
        p.output / "grids.h5",
        {
            "axes": {"diam": (axis, {"units": "mas"})},
            "scan": {"delta_chi2": delta},
        },
        attrs={"/": {"axis_order": ["diam"], "schema": _io.SCHEMA}},
    )

    # The scan's lobes. Each is fitted for each model; the best lobe gives
    # the model's fit and (in the posterior) its bounded, group-invariant
    # log-uniform diameter prior. The lobe table is recorded in the summary.
    lobes = _find_lobes(axis, delta, lo, hi, s["max_lobes"])
    step = (hi / lo) ** (1.0 / (s["n_scan"] - 1))
    for lobe in lobes:
        lobe["diam"] = _refine(
            data, lobe["diam"], lo, hi, step, s["batch_size"]
        )
    noise = p._noise()
    fitted, reports = {}, {}
    for name in p.names:
        entry = registry[name]
        candidates = []
        for lobe in lobes:
            b0, b1 = lobe["bounds"]
            priors = entry.priors(b0, b1)
            model, values, info = _fit_model(
                p, entry, priors, lobe["diam"], noise
            )
            chi2, resid = _chi2_64(model, data)
            scales = {
                k.split(".", 1)[1]: float(v)
                for k, v in values.items()
                if k.startswith("noise.")
            }
            # With fitted error scales, lobes and models are compared on
            # chi^2 with the rescaled errors; chi^2/N on the quoted errors
            # stays the diagnostic.
            chi2_fitted = chi2
            if scales:
                chi2_fitted, _ = _chi2_64(
                    model,
                    data,
                    **{k: np.asarray(v) for k, v in scales.items()},
                )
            candidates.append(
                (
                    float(chi2_fitted),
                    model,
                    values,
                    info,
                    chi2,
                    resid,
                    scales,
                    priors,
                    lobe,
                )
            )
        top = min(c[0] for c in candidates)
        table = [
            {
                "diam_bounds_mas": list(c[8]["bounds"]),
                "diam_mas": float(np.asarray(c[2]["diam"])),
                "chi2": float(c[4]),
                "chi2_reduced": float(c[4]) / data.n_independent,
                "delta_chi2": c[0] - top,
            }
            for c in candidates
        ]
        chosen = min(candidates, key=lambda c: c[0])
        chi2_fitted, model, values, info, chi2, resid, scales, priors, lobe = (
            chosen
        )
        # Test only the independent whitened residuals: the periodic penalty
        # terms of correlated closure phases are not meant to be normal.
        resid = resid[: int(data.n_independent)]
        skew, kurt = _moments(resid)
        reported = {k: float(np.asarray(values[k])) for k in entry.params}
        if entry.derived:
            reported.update(
                {
                    k: float(v)
                    for k, v in entry.derived(
                        {k: np.asarray(values[k]) for k in entry.params}
                    ).items()
                }
            )
        _io.save_model(
            p.output / "models" / name, model, values, list(entry.params)
        )
        _io.write_json(p.output / "models" / name / "info.json", info)
        fitted[name] = (model, values, info)
        reports[name] = {
            "params": reported,
            "converged": info.get("converged"),
            "chi2": chi2,
            "chi2_fitted": float(chi2_fitted),
            "first_null": _first_null(model, float(values["diam"])),
            "diam_bounds_mas": list(lobe["bounds"]),
            "lobes": table,
            "chi2_reduced": chi2 / data.n_independent,
            "error_scales": scales or None,
            "residual_skew": skew,
            "residual_excess_kurtosis": kurt,
            "n_residuals": int(resid.size),
            "at_bound": list(info.get("at_bound", []) or []),
            "priors": {k: repr(v) for k, v in priors.items()},
            "prior_std": {
                k: float(np.sqrt(priors[k].variance)) for k in entry.shape
            },
        }

    comparison, best = None, p.names[0]
    n = int(data.n_independent)
    for name in p.names:
        entry = registry[name]
        if entry.parent in reports:
            simple = reports[entry.parent]["chi2_fitted"]
            gain = simple - reports[name]["chi2_fitted"]
            n_extra = entry.n_params - registry[entry.parent].n_params
            penalty = n_extra * math.log(max(n, 2))
            comparison = {
                "simple": entry.parent,
                "complex": name,
                "delta_chi2": gain,
                "n_extra_params": n_extra,
                "bic_penalty": penalty,
                "preferred": name if gain > penalty else entry.parent,
            }
            best = comparison["preferred"]
    if comparison is None:
        best = p.names[-1]
    model, values, info = fitted[best]
    _io.save_model(
        p.output / "models" / "best",
        model,
        values,
        list(registry[best].params),
    )
    _io.write_json(p.output / "models" / "best" / "info.json", info)

    one = {
        k: np.asarray([float(np.asarray(values[k]))])
        for k in registry[best].params
    }
    pred = posterior_predictive_summary(
        one, model, data, list(registry[best].params)
    )
    fig = plot_data_model_correlation(data, {"MAP fit": pred})
    fig = fig[0] if isinstance(fig, tuple) else fig
    _save(fig, p.output / "plots" / "fit_correlation.png")
    fig = plot_v2_models(
        data, {registry[n].label: fitted[n][0] for n in p.names}
    )
    _save(fig, p.output / "plots" / "fit_v2.png")
    return {
        "names": list(p.names),
        "best": best,
        "n_independent": n,
        "diam_range_mas": [lo, hi],
        "diam_bounds_mas": reports[best]["diam_bounds_mas"],
        "rivals": [
            [row["diam_mas"], row["delta_chi2"]]
            for row in reports[best]["lobes"]
            if row["delta_chi2"] > 0 and row["delta_chi2"] <= 25.0
        ],
        "scan_diam_mas": best_diam,
        "models": reports,
        "comparison": comparison,
    }


_NOISE_BOUNDS = (0.1, 10.0)  # the fitted error scales' LogUniform range


def _bound_fractions(samples, priors, skip=()):
    """Fraction of samples within 1% of the range of either prior bound.

    Covers the sampled priors except ``skip`` (the shape parameters, whose
    priors are meant to matter and are judged by
    ``limb_darkening_constrained`` instead), and every fitted error scale
    ``noise.<term>``.
    """
    bound = {}
    scales = {
        k: types.SimpleNamespace(low=_NOISE_BOUNDS[0], high=_NOISE_BOUNDS[1])
        for k in samples
        if k.startswith("noise.")
    }
    for k, prior in {**priors, **scales}.items():
        if k in skip or not hasattr(prior, "low"):
            continue
        lo, hi = float(prior.low), float(prior.high)
        x = np.asarray(samples[k], dtype=float).ravel()
        if type(prior).__name__ in ("LogUniform", "SimpleNamespace"):
            lo, hi, x = math.log(lo), math.log(hi), np.log(x)
        edge = _BOUND_FRACTION * (hi - lo)
        bound[k] = float(np.mean((x < lo + edge) | (x > hi - edge)))
    return bound


@_float64
def _nuts(model, data, p, name, priors, start):
    """NUTS for one fitted model: ``(samples, stats, summary)``."""
    import jax
    from numpyro.diagnostics import effective_sample_size, split_gelman_rubin
    from numpyro.infer import MCMC, NUTS

    from ..likelihood import chain_init_params, numpyro_model

    s = p.settings
    entry = models()[name]
    posterior = numpyro_model(model, priors, data, noise=p._noise())
    key = jax.random.PRNGKey(s["seed"])
    init_key, run_key = jax.random.split(key)
    init = chain_init_params(
        posterior, [start] * s["num_chains"], key=init_key
    )
    mcmc = MCMC(
        NUTS(posterior, dense_mass=True),
        num_warmup=s["num_warmup"],
        num_samples=s["num_samples"],
        num_chains=s["num_chains"],
        chain_method=s["chain_method"],
        progress_bar=False,
    )
    mcmc.run(
        run_key,
        init_params=init,
        extra_fields=(
            "diverging",
            "accept_prob",
            "adapt_state.step_size",
            "num_steps",
            "potential_energy",
        ),
    )
    samples = {
        k: np.asarray(v)
        for k, v in mcmc.get_samples(group_by_chain=True).items()
    }
    extra = mcmc.get_extra_fields(group_by_chain=True)
    stats = {
        ("step_size" if k == "adapt_state.step_size" else k): np.asarray(v)
        for k, v in extra.items()
    }
    diagnostics = {
        site: {
            "r_hat": float(np.nanmax(np.asarray(split_gelman_rubin(x)))),
            "ess_bulk": float(np.nanmin(np.asarray(effective_sample_size(x)))),
        }
        for site, x in samples.items()
    }
    if entry.derived:
        samples.update(entry.derived(samples))
    reported = list(entry.params) + list(
        k
        for k in samples
        if k not in entry.params and not k.startswith("noise.")
    )
    summary = {}
    for k in reported:
        flat = np.asarray(samples[k], dtype=float).ravel()
        summary[k] = {
            **_quantiles(flat),
            "std": float(np.std(flat)),
            **diagnostics.get(k, {}),
        }
    noise_sites = {
        k.split(".", 1)[1]: _quantiles(samples[k])
        for k in samples
        if k.startswith("noise.")
    }
    return (
        samples,
        stats,
        {
            "params": summary,
            "noise": noise_sites or None,
            "r_hat_max": max(d["r_hat"] for d in diagnostics.values()),
            "ess_bulk_min": min(d["ess_bulk"] for d in diagnostics.values()),
            "divergence_fraction": float(np.mean(stats["diverging"])),
            "prior_bound_fraction": _bound_fractions(
                samples, priors, skip=entry.shape
            ),
        },
    )


def _posterior(p):
    import pandas as pd

    from ..plotting import plot_chainconsumer_diagnostics

    s = p.settings
    fit_report = p._report("fit")
    best = fit_report["best"]
    groups, per_model, params_attr = {}, {}, {}
    for name in p.names:
        entry = models()[name]
        priors = entry.priors(*fit_report["models"][name]["diam_bounds_mas"])
        start = {
            k: np.asarray(v)
            for k, v in _io.load_model_values(
                p.output / "models" / name
            ).items()
            if k in priors
        }
        model = _io.load_model(p.output / "models" / name)
        samples, stats, summary = _nuts(
            model, p.processed, p, name, priors, start
        )
        groups[f"posterior_{name}"] = samples
        groups[f"sample_stats_{name}"] = stats
        per_model[name] = summary
        params_attr[name] = list(summary["params"])
        if name == best:
            groups["posterior"], groups["sample_stats"] = samples, stats
    _io.write_h5(
        p.output / "samples.h5",
        groups,
        attrs={
            "/": {
                "schema": _io.SCHEMA,
                "best": best,
                "params": params_attr,
                "priors": {
                    n: fit_report["models"][n]["priors"] for n in p.names
                },
                "sampler": {
                    k: s[k]
                    for k in (
                        "num_warmup",
                        "num_samples",
                        "num_chains",
                        "chain_method",
                        "seed",
                    )
                },
            }
        },
    )

    sampled = list(models()[best].params)
    frame = {
        k: np.asarray(groups["posterior"][k], dtype=float).ravel()
        for k in sampled
    }
    _, fig_corner, fig_walk = plot_chainconsumer_diagnostics(
        {"posterior": pd.DataFrame(frame)}, columns=sampled
    )
    _save(fig_corner, p.output / "plots" / "posterior_corner.png")
    _save(fig_walk, p.output / "plots" / "posterior_trace.png")
    return {
        "models": per_model,
        "best": best,
        "sampled": sampled,
        "num_chains": s["num_chains"],
        "num_samples": s["num_samples"],
        "num_warmup": s["num_warmup"],
        "r_hat_max": max(m["r_hat_max"] for m in per_model.values()),
        "ess_bulk_min": min(m["ess_bulk_min"] for m in per_model.values()),
        "divergence_fraction": max(
            m["divergence_fraction"] for m in per_model.values()
        ),
    }


def _quicklook(p):
    from . import _quicklook as ql

    ql.write_quicklook(p.output, execute=p.settings["quicklook_execute"])
    return {"executed": bool(p.settings["quicklook_execute"])}


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


def _summarise(settings, reports):
    registry = models()
    sections, checks = {}, []
    load, fit, post = (
        reports.get("load"),
        reports.get("fit"),
        reports.get("posterior"),
    )
    if load:
        sections["data"] = {k: load[k] for k in _DATA_KEYS}
    if not fit:
        return sections, checks

    names, best = fit["names"], fit["best"]
    n = fit["n_independent"]
    sections["chi2"] = {
        "n_independent": n,
        "error_scale": settings["error_scale"],
        "reduced": {k: fit["models"][k]["chi2_reduced"] for k in names},
        "error_scales": {k: fit["models"][k]["error_scales"] for k in names},
    }
    sections["fit"] = {
        "best": best,
        "scan_diam_mas": fit["scan_diam_mas"],
        "diam_bounds_mas": fit["diam_bounds_mas"],
        "rivals": fit["rivals"],
        "models": {
            k: {
                f: fit["models"][k][f]
                for f in (
                    "params",
                    "converged",
                    "chi2",
                    "chi2_fitted",
                    "chi2_reduced",
                    "diam_bounds_mas",
                    "lobes",
                    "at_bound",
                )
            }
            for k in names
        },
    }
    for k in names:
        checks.append(
            _checks.chi2_reduced(
                f"chi2_{k}",
                fit["models"][k]["chi2_reduced"],
                label=f"the {registry[k].label}",
            )
        )
    comparison = fit["comparison"]
    if comparison:
        sections["comparison"] = dict(comparison)
        checks.append(
            _checks.model_gain(
                "limb_darkening_gain",
                fit["models"][comparison["simple"]]["chi2_fitted"],
                fit["models"][comparison["complex"]]["chi2_fitted"],
                n,
                n_extra=comparison["n_extra_params"],
                simple=registry[comparison["simple"]].label,
                complex_=registry[comparison["complex"]].label,
            )
        )
    scales = {
        f"{k}.{kk}": v
        for k in names
        for kk, v in (fit["models"][k]["error_scales"] or {}).items()
    }
    if scales:
        checks.append(_checks.error_scale(scales))

    source = post["models"][best]["params"] if post else None
    if source:
        diam, diam_err = (
            source["diam"]["median"],
            0.5 * (source["diam"]["q84"] - source["diam"]["q16"]),
        )
    else:
        diam, diam_err = fit["models"][best]["params"]["diam"], None
    sections["star"] = {
        "model": best,
        "source": "posterior" if post else "fit",
        "diam_mas": diam,
        "diam_err_mas": diam_err,
    }
    if load:
        sections["resolution"] = {
            "diam_mas": diam,
            "freq_max_per_rad": load["freq_max_per_rad"],
        }
        null = fit["models"][best]["first_null"]
        sections["resolution"]["first_null"] = null
        check = _checks.resolution_regime(
            diam, load["freq_max_per_rad"], first_null=null
        )
        sections["resolution"]["first_null_fraction"] = check.value
        checks.append(check)
    checks.append(_checks.multimodal(fit["models"][best]["lobes"]))
    if post:
        entry = registry[best]
        if entry.shape:
            ratios = {}
            for k in entry.shape:
                row = post["models"][best]["params"][k]
                ratios[k] = row["std"] / fit["models"][best]["prior_std"][k]
            sections["shape"] = {
                "model": best,
                "params": {
                    k: {
                        "median": post["models"][best]["params"][k]["median"],
                        "posterior_std": post["models"][best]["params"][k][
                            "std"
                        ],
                        "prior_std": fit["models"][best]["prior_std"][k],
                        "ratio": ratios[k],
                    }
                    for k in entry.shape
                },
            }
            checks.append(
                _checks.prior_constrained("limb_darkening_constrained", ratios)
            )
        sections["posterior"] = {
            k: post[k]
            for k in (
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
        }
        checks.append(
            _checks.prior_bound(
                {
                    f"{n}.{k}": v
                    for n, m in post["models"].items()
                    for k, v in m["prior_bound_fraction"].items()
                }
            )
        )
        checks.append(_checks.r_hat(post["r_hat_max"]))
        checks.append(_checks.ess(post["ess_bulk_min"]))
        checks.append(_checks.divergences(post["divergence_fraction"]))
    return sections, checks
