"""`BinaryPipeline`: companion search, contrast limits, fit and posterior."""

from __future__ import annotations

import math
import warnings
from typing import NamedTuple

import numpy as onp

from . import _checks, _io
from ._common import (
    _overview,
    _quicklook,
    _chain_health,
    _corner_trace,
    _diagnostics,
    _draws,
    _integer,
    _load_report,
    _interior_start,
    _load_start,
    _noise_quantiles,
    _pair,
    _pick,
    _positive,
    _prior_bounds,
    _prior_repr,
    _run_chains,
    _sampler_checks,
    _sampler_report,
    _save_map_correlation,
    _site_summary,
    _validate_common,
    _write_samples,
    _chi2,
    _geometry,
    _moments,
    _save,
    _scales_of,
)
from .._geometry import lattice_alias_limit, separation_pa
from ._core import Stage, _Pipeline
from .._deprecate import external_stacklevel

GLOBAL_NSIGMA_METHOD = "simulated null: Gaussian noise at the quoted errors"

# Default grid step, as a fraction of the finest λ/B (Nyquist is one half).
_STEP_FRACTION = 0.25
# Relative slack before a step counts as coarsened (rounding of the odd n).
_STEP_TOLERANCE = 1e-9
# The fit starts at least this factor inside the ends of the flux prior.
_START_MARGIN = 2.0
# A fit that would start within this fraction of a grid step of the origin
# starts exactly this far from it.
_ORIGIN_NUDGE = 1e-2
_AXIS_ORDER = ("dra", "ddec", "flux")
_AXIS_UNITS = {"dra": "mas", "ddec": "mas", "flux": "companion/primary"}


class _Axis(NamedTuple):
    """Search-grid position axes: half-width ``max_sep`` (mas), ``n``
    points, effective ``step`` and ``wanted`` step."""

    max_sep: float
    n: int
    step: float
    wanted: float
    alias_cap: float | None = None

    @property
    def coarsened(self):
        """Whether ``max_grid`` made the step coarser than requested."""
        return self.step > self.wanted * (1 + _STEP_TOLERANCE)


def _null_nsigma(data, grid, batch_size, n_null, seed, delta_chi2):
    """Global significance of ``delta_chi2`` from ``n_null`` companion-free
    searches, and its false-alarm probability.

    The searches repeat the real one on Gaussian noise at the quoted errors.
    The probability is ``(k + 1) / (n_null + 1)`` for ``k`` null maxima at
    or above ``delta_chi2``, so it is an upper bound and the significance a
    lower bound; with ``k = 0`` it sits at the floor ``1 / (n_null + 1)``
    however strong the peak. Returns ``(sigma, fap, k)``; all NaN if
    ``delta_chi2`` is not finite."""
    from scipy.stats import norm

    from ..detection import gaussian_null, injection_recovery
    from ..models import BinaryModelCartesian

    if not math.isfinite(delta_chi2):
        return float("nan"), float("nan"), float("nan")
    mc = injection_recovery(
        BinaryModelCartesian,
        _null_model(),
        data,
        grid,
        seed,
        n_null=n_null,
        noise=gaussian_null(_null_model(), data),
        batch_size=batch_size,
        progress=False,
    )
    fap = float(mc.false_alarm_probability("delta_chi2", delta_chi2)[0])
    # A FAP of 1 (every null peaks higher) would give -inf, which JSON rejects.
    k = round(fap * (n_null + 1)) - 1
    return float(norm.isf(min(fap, n_null / (n_null + 1.0)))), fap, k


class BinaryPipeline(_Pipeline):
    """Search for a companion, set contrast limits, fit and sample a binary.

    !!! warning "Provisional"
        May change in any 0.4.x release without a deprecation cycle.

    Stages: ``load``, ``overview``, ``search`` (point-companion grid,
    [`detection_statistics`][virgil.detection.detection_statistics]),
    ``limits`` ([`absil_limits`][virgil.limits.absil_limits]), ``fit``,
    ``posterior`` (NUTS, same prior as the fit) and ``quicklook``.
    ``res.summary["companion"]`` has ``sep_mas`` and ``pa_deg`` for either
    template. Settings:
    [pipeline settings](../pipeline.md#binarypipeline-settings).

    Parameters
    ----------
    data : OIData
        The data.
    model : BinaryModelAngular or BinaryModelCartesian, optional
        An instance of [`BinaryModelAngular`][virgil.models.BinaryModelAngular]
        (default) or [`BinaryModelCartesian`][virgil.models.BinaryModelCartesian];
        its class sets the fitted parameters. The search always uses a point
        companion.
    output : str or os.PathLike, optional
        Run folder (default ``"run"``).
    **settings
        Overrides of ``BinaryPipeline.defaults()``. The most used: ``sigma``
        (significance of the contrast limits, 3.0), ``max_sep_mas`` (search
        half-width in mas; default the coarsest λ/B, capped on lattice
        coverage), ``grid_step_mas``, ``flux_range`` (companion/primary, 1e-5 to 1; a
        brighter companion is the primary under relabelling), ``error_scale``
        (``"quoted"`` or ``"fit"``; quoted so that detection and limits are
        judged on the catalogue errors, unlike the disk's Gaussian-process
        rescale). Priors and errors:
        [conventions](../conventions.md).

    Examples
    --------
    >>> import virgil as vg  # doctest: +SKIP
    >>> from virgil.pipeline import BinaryPipeline  # doctest: +SKIP
    >>> res = BinaryPipeline(vg.OIData("hd1234.oifits"), output="runs/binary").run()  # doctest: +SKIP
    >>> res.summary["companion"]["sep_mas"]  # doctest: +SKIP
    """

    NAME = "binary"
    STABILITY = "provisional"
    STAGES = (
        "load",
        "overview",
        "search",
        "limits",
        "fit",
        "posterior",
        "quicklook",
    )
    _DEFAULTS = {
        "sigma": 3.0,
        "detection_sigma": 3.0,
        "n_null": 1000,
        "max_sep_mas": None,
        "fov_fraction": 1.0,
        "grid_step_mas": None,
        "max_grid": 101,
        "flux_range": [1e-5, 1.0],
        "n_flux": 40,
        "wavel_range": None,
        "error_floor": None,
        "error_scale": "quoted",
        "num_warmup": 1000,
        "num_samples": 1000,
        "num_chains": 4,
        "chain_method": "vectorized",
        "seed": 0,
        "batch_size": None,
        "limit_bins": 20,
        "quicklook_execute": True,
    }

    def __init__(self, data, model=None, *, output="run", **settings):
        super().__init__(data, model, output=output, **settings)
        from ..models import BinaryModelAngular, BinaryModelCartesian

        if not isinstance(
            self.model, (BinaryModelAngular, BinaryModelCartesian)
        ):
            raise TypeError(
                "BinaryPipeline fits a BinaryModelAngular or "
                f"BinaryModelCartesian template; got {type(self.model).__name__}."
            )

    def _default_model(self):
        from ..models import BinaryModelAngular

        return BinaryModelAngular(sep=100.0, pa=0.0, flux=1e-3)

    @classmethod
    def _validate(cls, s):
        for name in ("sigma", "detection_sigma", "fov_fraction"):
            _positive(s, name)
        for name in ("max_sep_mas", "grid_step_mas"):
            if s[name] is not None:
                _positive(s, name)
        _integer(s, "limit_bins", 1)
        _integer(s, "n_null", 1)
        from scipy.stats import norm

        # A simulated false-alarm probability is at least 1 / (n_null + 1).
        need = math.ceil(1.0 / norm.sf(s["detection_sigma"])) - 1
        if s["n_null"] < need:
            raise ValueError(
                f"n_null={s['n_null']} cannot resolve detection_sigma="
                f"{s['detection_sigma']:g}: need n_null >= {need}."
            )
        _integer(s, "max_grid", 3)
        _integer(s, "n_flux", 2)
        _pair(s, "flux_range")
        if s["error_scale"] not in ("quoted", "fit"):
            raise ValueError('error_scale must be "quoted" or "fit".')
        _validate_common(s)

    # --- shared state, rebuilt on demand so that resume works ------------

    def _grid_axis(self):
        """The position axes as an ``_Axis`` (odd ``n``, star on the grid)."""
        s = self.settings
        geo = _geometry(self.processed)
        max_sep, cap = s["max_sep_mas"], None
        if not max_sep:
            max_sep = s["fov_fraction"] * geo["fov_mas"]
            # Deliberate: on lattice coverage (aperture masks, DISCO) the
            # default box stops at the first alias, since a companion and its
            # lattice copies fit the data equally well. An explicit
            # ``max_sep_mas`` is not capped.
            limit = lattice_alias_limit(self.processed)
            if limit is not None and limit < max_sep:
                max_sep, cap = limit, limit
        wanted = (
            s["grid_step_mas"] or _STEP_FRACTION * geo["lambda_over_b_mas"]
        )
        most = s["max_grid"] - (1 - s["max_grid"] % 2)  # the odd number
        n = min(2 * math.ceil(max_sep / wanted) + 1, most)
        return _Axis(max_sep, n, 2.0 * max_sep / (n - 1), wanted, cap)

    def _grid(self):
        import jax.numpy as np

        s = self.settings
        max_sep, n = self._grid_axis()[:2]
        axis = np.linspace(-max_sep, max_sep, n)
        lo, hi = s["flux_range"]
        return {
            "dra": axis,
            "ddec": axis,
            "flux": np.logspace(math.log10(lo), math.log10(hi), s["n_flux"]),
        }

    def _priors(self, max_sep):
        """Priors on ``dra``, ``ddec`` (uniform on the box) and ``flux``."""
        import numpyro.distributions as dist

        lo, hi = self.settings["flux_range"]
        # Uniform in position, not sep (which would put p ∝ 1/sep on the sky).
        return {
            "dra": dist.Uniform(-max_sep, max_sep),
            "ddec": dist.Uniform(-max_sep, max_sep),
            "flux": dist.LogUniform(lo, hi),
        }

    def _sampled_model(self):
        """The template as a function of ``dra``, ``ddec`` and ``flux``."""
        from ..models import BinaryModelAngular, BinaryModelCartesian

        if not isinstance(self.model, BinaryModelAngular):
            return BinaryModelCartesian(0.0, 0.0, 0.0)

        def model(dra, ddec, flux):
            sep, pa = separation_pa(dra, ddec)
            return BinaryModelAngular(sep, pa, flux)

        return model

    def _stages(self):
        return (
            Stage("load", _load_report, ("stages/load/report.json",)),
            Stage(
                "overview",
                _overview,
                ("plots/overview_data.png", "plots/overview_uv.png"),
            ),
            Stage(
                "search",
                _search,
                (
                    "grids.h5",
                    "plots/search_delta_chi2.png",
                    "plots/search_snr.png",
                ),
            ),
            Stage(
                "limits",
                _limits,
                ("plots/limits_map.png", "plots/limits_contrast_curve.png"),
            ),
            Stage(
                "fit",
                _fit,
                (
                    "models/best/manifest.json",
                    "models/best/values.npz",
                    "models/best/info.json",
                    "plots/fit_correlation.png",
                ),
            ),
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
        return _summarize(self.settings, reports)


# === STAGES ===


# --- search ---


def _search(p):
    from ..detection import detection_statistics
    from ..models import BinaryModelCartesian

    data, s = p.processed, p.settings
    grid, axis = p._grid(), p._grid_axis()
    if axis.coarsened:
        warnings.warn(
            f"max_grid={s['max_grid']} coarsens the grid step from "
            f"{axis.wanted:.3g} to {axis.step:.3g} mas; raise max_grid (or "
            "grid_step_mas) for a finer search.",
            UserWarning,
            stacklevel=external_stacklevel(),
        )
    # detection_statistics evaluates the full likelihood grid, refines the
    # flux and takes the Laplace width internally, but returns only scalars.
    # Its log Bayes factor needs the full likelihood grid and its Δχ² falls
    # back on the grid flux where the refinement fails, so neither follows
    # from the maps of _search_maps: it stays one call beside them.
    detection = detection_statistics(
        BinaryModelCartesian, data, grid, batch_size=s["batch_size"]
    )
    maps = _search_maps(data, grid, s["batch_size"])
    _write_grids(p.output, grid, search=maps)
    best = {"dra": float(detection["dra"]), "ddec": float(detection["ddec"])}
    from ..plotting import plot_grid_map

    fig, _ = plot_grid_map(
        maps["delta_chi2"],
        grid,
        kind="loglike",
        label="Δχ²",
        title="Δχ² (point companion)",
        best=best,
    )
    _save(fig, p.output / "plots" / "search_delta_chi2.png")
    fig, _ = plot_grid_map(maps["snr"], grid, kind="snr", best=best)
    _save(fig, p.output / "plots" / "search_snr.png")
    return _search_report(data, grid, axis, detection, s)


def _refined_flux(
    data, grid, batch_size
):  # deliberate: tests call it directly
    """``(flux, loglike)`` of the point companion refined at every position."""
    from .._grid import (
        batch_size_or_default,
        resolve_grid_keys,
        warn_unconverged,
    )
    from ..grid_fit import _optimize_flux_grid
    from ..models import BinaryModelCartesian

    params, coord_keys, flux_key = resolve_grid_keys(grid)
    flux, loglike, converged = _optimize_flux_grid(
        data,
        BinaryModelCartesian,
        grid,
        params=params,
        coord_keys=coord_keys,
        flux_key=flux_key,
        batch_size=batch_size_or_default(batch_size, data),
    )
    warn_unconverged(converged, "optimized_flux_grid")
    return flux, loglike


def _search_maps(data, grid, batch_size):
    """The maps of the search over the grid: ``search`` group of grids.h5."""
    import jax.numpy as np

    from ..grid_fit import laplace_flux_uncertainty_grid
    from ..likelihood import model_loglike
    from ..models import BinaryModelCartesian

    flux, loglike = (
        onp.asarray(x) for x in _refined_flux(data, grid, batch_size)
    )
    sigma = onp.asarray(
        laplace_flux_uncertainty_grid(
            BinaryModelCartesian,
            data,
            grid,
            flux=np.asarray(flux),
            batch_size=batch_size,
        )
    )
    loglike_null = float(model_loglike(_null_model(), data))
    delta_chi2 = onp.where(
        flux > 0, onp.maximum(2.0 * (loglike - loglike_null), 0.0), 0.0
    )
    return {
        "delta_chi2": delta_chi2,
        "loglike": loglike,
        "flux_best": flux,
        "flux_sigma": sigma,
        "snr": flux / sigma,
    }


def _null_model():
    """The star alone: a point companion of zero flux."""
    from ..models import BinaryModelCartesian

    return BinaryModelCartesian(0.0, 0.0, 0.0)


def _write_grids(output, grid, **groups):
    """Write ``grids.h5``: the ``axes`` group (each axis with its units),
    then ``groups``."""
    axes = {
        k: (onp.asarray(grid[k]), {"units": _AXIS_UNITS[k]})
        for k in _AXIS_ORDER
    }
    _io.write_h5(
        output / "grids.h5",
        {"axes": axes, **groups},
        attrs={"/": {"axis_order": list(_AXIS_ORDER), "schema": _io.SCHEMA}},
    )


# deliberate: log-flux nearest-point rule with out-of-axis clamp; test_refined_flux_is_the_two_public_grids_from_one_pass pins both
def _peak_index(grid, best):
    """Grid index of ``best`` on each axis (the flux is matched in log)."""
    index = []
    for key in _AXIS_ORDER:
        axis = onp.asarray(grid[key], dtype=float)
        value = float(best[key])
        if key == "flux":
            axis, value = onp.log(axis), math.log(max(value, axis[0]))
        index.append(int(onp.argmin(onp.abs(axis - value))))
    return index


def _search_report(data, grid, axis, detection, settings):
    from ..detection import local_nsigma

    local = float(local_nsigma(float(detection["delta_chi2"])))
    global_nsigma, fap, n_above = _null_nsigma(
        data,
        grid,
        settings["batch_size"],
        settings["n_null"],
        settings["seed"],
        float(detection["delta_chi2"]),
    )
    chi2_null, _ = _chi2(_null_model(), data)
    return {
        "delta_chi2": float(detection["delta_chi2"]),
        "log_bayes_factor": float(detection["log_bayes_factor"]),
        "max_snr": float(detection["max_snr"]),
        "dra_mas": float(detection["dra"]),
        "ddec_mas": float(detection["ddec"]),
        "flux": float(detection["flux"]),
        "local_nsigma": local,
        "global_nsigma": global_nsigma,
        "global_nsigma_method": GLOBAL_NSIGMA_METHOD,
        "false_alarm_probability": fap,
        "n_null_above": n_above,
        "global_nsigma_is_lower_bound": n_above == 0,
        "n_null": settings["n_null"],
        "max_sep_mas": axis.max_sep,
        "alias_cap_mas": axis.alias_cap,
        "grid_step_mas": axis.step,
        "grid_step_wanted_mas": axis.wanted,
        "grid_shape": [len(onp.asarray(grid[k])) for k in _AXIS_ORDER],
        "peak_index": _peak_index(grid, detection),
        "chi2_null": chi2_null,
        "n_independent": int(data.n_independent),
    }


# --- limits ---


def _limits(p):
    from ..limits import (
        absil_limits,
        flux_to_delta_mag,
        radial_profile,
    )
    from ..models import BinaryModelCartesian
    from ..plotting import plot_contrast_curve, plot_grid_map

    s = p.settings
    grid, max_sep = p._grid(), p._grid_axis().max_sep
    limit = onp.asarray(
        absil_limits(
            BinaryModelCartesian,
            p.processed,
            grid,
            s["sigma"],
            batch_size=s["batch_size"],
        )
    )
    dra, ddec = onp.asarray(grid["dra"]), onp.asarray(grid["ddec"])
    # Inside the resolution limit separation and flux are degenerate: NaN.
    r = onp.hypot(*onp.meshgrid(dra, ddec, indexing="ij"))
    resolution = _geometry(p.processed)["resolution_mas"]
    limit = onp.where(r < resolution, onp.nan, limit)
    profile = radial_profile(
        limit, dra, ddec, r_max=max_sep, bins=s["limit_bins"]
    )
    # write_h5 cannot append: rewrite the file with the search group read back.
    meta = {"sigma": s["sigma"], "method": "absil"}
    _write_grids(
        p.output,
        grid,
        search=_io.read_h5(p.output / "grids.h5", "search"),
        limits={"limit_flux": (limit, meta)},
    )
    fig, _ = plot_grid_map(
        limit, grid, kind="limit", units="delta_mag", sigma=s["sigma"]
    )
    _save(fig, p.output / "plots" / "limits_map.png")
    fig, _ = plot_contrast_curve(profile, sigma=s["sigma"])
    _save(fig, p.output / "plots" / "limits_contrast_curve.png")
    median = onp.asarray(profile["median"], dtype=float)
    delta_mag = onp.asarray(flux_to_delta_mag(median), dtype=float)
    return {
        "sigma": s["sigma"],
        "method": "absil",
        "r_mas": profile["r"],
        "median_flux": median,
        "median_delta_mag": delta_mag,
        "deepest_delta_mag": float(onp.nanmax(delta_mag)),
    }


# --- fit ---


def _is_angular(p):
    from ..models import BinaryModelAngular

    return isinstance(p.model, BinaryModelAngular)


def _fit(p):
    from ..fitting import fit

    data = p.processed
    priors = p._priors(p._grid_axis().max_sep)
    angular = _is_angular(p)
    template, init = _fit_start(p, p._report("search"), priors, angular)
    result = fit(template, priors, data, noise=p._noise(), init=init)
    values = {k: onp.asarray(v) for k, v in result.values.items()}
    if angular:  # the angular model's own sep, pa
        values.update(
            sep=onp.asarray(result.model.sep), pa=onp.asarray(result.model.pa)
        )
    _require_finite(values, result.info)
    reported = ["sep", "pa", "flux"] if angular else list(priors)
    best = p.output / "models" / "best"
    _io.save_model(best, result.model, values, reported)
    _io.write_json(best / "info.json", _io.clean_json(dict(result.info)))
    _save_map_correlation(
        p,
        p._sampled_model() if angular else result.model,
        values,
        list(priors),
    )
    chi2, resid = _chi2(result.model, data)
    # The periodic penalty terms of correlated closure phases are not
    # meant to be normal: test only the independent whitened residuals.
    resid = resid[: int(data.n_independent)]
    skew, kurt = _moments(resid)
    scales = _scales_of(values)
    return {
        "params": {k: float(values[k]) for k in reported},
        "converged": result.info.get("converged"),
        "chi2": chi2,
        "n_independent": int(data.n_independent),
        "chi2_reduced": chi2 / data.n_independent,
        "error_scales": scales or None,
        "residual_skew": skew,
        "residual_excess_kurtosis": kurt,
        "n_residuals": int(resid.size),
        "at_bound": list(result.info.get("at_bound", []) or []),
        "priors": {k: _prior_repr(v) for k, v in priors.items()},
    }


def _fit_start(
    p, search, priors, angular
):  # deliberate: tests call it with a stub pipeline
    """The template to fit and its start (the search's best point)."""
    lo, hi = p.settings["flux_range"]
    start = {
        "dra": search["dra_mas"],
        "ddec": search["ddec_mas"],
        "flux": min(
            max(search["flux"], _START_MARGIN * lo), hi / _START_MARGIN
        ),
    }
    start = {k: onp.asarray(start[k], dtype=float) for k in priors}
    # A search peak on the grid's edge is on the position prior's bound, where
    # the unconstrained start is infinite (a NaN fit in float32).
    start = _interior_start(priors, start)
    template = p._sampled_model()
    if angular:
        # The angle is undefined on the origin, where the angular template
        # has no gradient in position (finite, but zero): a search peak on
        # (or within rounding of it) starts a little off it, as a Cartesian
        # template needs no such step.
        nudge = _ORIGIN_NUDGE * search["grid_step_mas"]
        if onp.hypot(start["dra"], start["ddec"]) < nudge:
            start["dra"], start["ddec"] = onp.asarray(nudge), onp.asarray(0.0)
        return template, start
    return template.set(list(priors), list(start.values())), None


# deliberate: raises at the fit with stop/loss in the message; test_a_non_finite_fit_is_reported_at_the_fit_not_in_the_summary pins it
def _require_finite(values, info):
    """Raise, at the fit, if it returned a NaN or infinite parameter."""
    bad = [k for k, v in values.items() if not onp.all(onp.isfinite(v))]
    if bad:
        raise ValueError(
            f"The fit returned non-finite values for {bad} (stop: "
            f"{info.get('stop')!r}, loss {info.get('loss')}); check the "
            "start, the priors and the data."
        )


# --- posterior ---


def _posterior(p):
    fit_report = p._report("fit")
    priors = p._priors(p._grid_axis().max_sep)
    reported = list(fit_report["params"])
    angular = _is_angular(p)

    from ..likelihood import numpyro_model

    best = p.output / "models" / "best"
    model = p._sampled_model() if angular else _io.load_model(best)
    posterior = numpyro_model(model, priors, p.processed, noise=p._noise())
    mcmc = _run_chains(
        posterior,
        p.settings,
        _load_start(best, priors),
        dense_mass=False,
        priors=priors,
    )
    sampled, stats = _draws(mcmc)  # one start per chain, from the fit
    samples = dict(sampled)
    if angular:
        # The model's own parameters, derived from the sampled offsets.
        samples["sep"], samples["pa"] = separation_pa(
            sampled["dra"], sampled["ddec"]
        )
    _write_samples(
        p,
        {"posterior": samples, "sample_stats": stats},
        params=reported,
        priors=fit_report["priors"],
    )

    diagnostics = {k: _diagnostics([x]) for k, x in sampled.items()}
    derived = {}
    if angular:  # an angle through its sine and cosine
        pa = onp.radians(samples["pa"])
        derived = {
            "sep": _diagnostics([samples["sep"]]),
            "pa": _diagnostics([onp.sin(pa), onp.cos(pa)]),
        }
    summary = {
        k: _site_summary(
            samples[k],
            derived.get(k, diagnostics.get(k, {})),
            std=False,
            angle=(k == "pa"),
        )
        for k in reported
    }
    _corner_trace(p.output, samples, reported)
    health = _chain_health(diagnostics, stats)
    return {
        "params": summary,
        "noise": _noise_quantiles(samples),
        **_sampler_report(p.settings, [health]),
        "prior_bound_fraction": _prior_bounds(priors, samples),
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
_SEARCH_KEYS = (
    "delta_chi2",
    "log_bayes_factor",
    "max_snr",
    "local_nsigma",
    "global_nsigma",
    "global_nsigma_method",
    "false_alarm_probability",
    "n_null",
    "n_null_above",
    "global_nsigma_is_lower_bound",
    "dra_mas",
    "ddec_mas",
    "flux",
    "max_sep_mas",
    "alias_cap_mas",
)
_LIMITS_KEYS = (
    "sigma",
    "method",
    "r_mas",
    "median_flux",
    "median_delta_mag",
    "deepest_delta_mag",
)
_FIT_KEYS = ("params", "converged", "chi2_reduced", "at_bound")
_POSTERIOR_KEYS = (
    "params",
    "noise",
    "num_chains",
    "num_samples",
    "num_warmup",
    "r_hat_max",
    "ess_bulk_min",
    "divergence_fraction",
)


def _chi2_checks(search, fit):
    """The χ²/N check: of the best-fit binary, else of the star alone."""
    null = search["chi2_null"] / search["n_independent"] if search else None
    if fit:
        label = "the best-fit binary" + (
            f" (star alone: {null:.3g})" if null is not None else ""
        )
        return [_checks.chi2_reduced("chi2", fit["chi2_reduced"], label=label)]
    if search:
        return [_checks.chi2_reduced("chi2", null, label="the star alone")]
    return []


def _fit_checks(fit, companion, load):
    checks = []
    if fit["error_scales"]:
        checks.append(_checks.error_scale(fit["error_scales"]))
    if fit["at_bound"]:
        checks.append(_checks.at_bound(fit["at_bound"], start=True))
    checks.append(
        _checks.residual_normality(
            fit["residual_skew"],
            fit["residual_excess_kurtosis"],
            fit["n_residuals"],
        )
    )
    if load:
        checks.append(
            _checks.field_of_view(
                companion["sep_mas"], load["resolution_mas"], load["fov_mas"]
            )
        )
    return checks


def _summarize(settings, reports):
    """``(sections, checks)`` of the summary from the stage reports so far."""
    from ..limits import flux_to_contrast, flux_to_delta_mag
    from ..models import BinaryModelAngular

    load, search, limits, fit, post = (
        reports.get(k)
        for k in ("load", "search", "limits", "fit", "posterior")
    )
    sections, checks = {}, _chi2_checks(search, fit)
    if load:
        sections["data"] = _pick(load, _DATA_KEYS)
    if search:
        sections["chi2"] = {
            "n_independent": search["n_independent"],
            "null_reduced": search["chi2_null"] / search["n_independent"],
            "companion_reduced": fit["chi2_reduced"] if fit else None,
            "error_scale": settings["error_scale"],
            "error_scales": fit["error_scales"] if fit else None,
        }
        sections["search"] = _pick(search, _SEARCH_KEYS)
        checks.append(
            _checks.detection(
                search["local_nsigma"],
                search["global_nsigma"],
                threshold=settings["detection_sigma"],
                bound=search["global_nsigma_is_lower_bound"],
            )
        )
        # The flux axis only seeds the optimizer: not a search boundary.
        checks.append(
            _checks.grid_edge(
                search["peak_index"][:2],
                search["grid_shape"][:2],
                ("dra", "ddec"),
            )
        )
    if limits:
        sections["limits"] = _pick(limits, _LIMITS_KEYS)
    if fit:
        sections["fit"] = _pick(fit, _FIT_KEYS)
        source = post["params"] if post else fit["params"]

        def q(name):  # (median, 1σ) of a posterior entry, or (value, None)
            x = source[name]
            if isinstance(x, dict):
                return x["median"], 0.5 * (x["q84"] - x["q16"])
            return x, None

        flux, flux_err = q("flux")
        companion = {"source": "posterior" if post else "fit"}
        if "sep" in source:
            (sep, sep_err), (pa, pa_err) = q("sep"), q("pa")
            cartesian = BinaryModelAngular(sep, pa, flux).to_cartesian()
            dra, ddec = float(cartesian.dra), float(cartesian.ddec)
        else:
            (dra, _), (ddec, _) = q("dra"), q("ddec")
            sep, pa = separation_pa(dra, ddec)
            sep, pa, sep_err, pa_err = float(sep), float(pa), None, None
        companion.update(
            sep_mas=sep,
            sep_err_mas=sep_err,
            pa_deg=pa,
            pa_err_deg=pa_err,
            dra_mas=dra,
            ddec_mas=ddec,
            flux=flux,
            flux_err=flux_err,
            contrast=float(flux_to_contrast(flux)),
            delta_mag=float(flux_to_delta_mag(flux)),
        )
        sections["companion"] = companion
        checks += _fit_checks(fit, sections["companion"], load)
    if post:
        sections["posterior"] = _pick(post, _POSTERIOR_KEYS)
        checks += _sampler_checks(post)
    return sections, checks
