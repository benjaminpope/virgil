"""`BinaryPipeline`: companion search, contrast limits, fit and posterior."""

from __future__ import annotations

import math

import numpy as onp

from . import _checks, _io
from ._core import Stage, _Pipeline

MAS_PER_RAD = 180.0 / math.pi * 3600.0 * 1000.0

# Fraction of a prior's range, at either end, counted as "at the bound".
_BOUND_FRACTION = 0.01


def _close(fig):
    import matplotlib.pyplot as plt

    plt.close(fig)


def _save(fig, path):
    with _io.atomic_path(path) as tmp:
        fig.savefig(tmp, dpi=110, bbox_inches="tight")
    _close(fig)


def _geometry(data):
    """Baseline and resolution figures of ``data`` (metres, mas)."""
    u = onp.asarray(data.u, dtype=float)
    v = onp.asarray(data.v, dtype=float)
    wavel = onp.broadcast_to(onp.asarray(data.wavel, dtype=float), u.shape)
    baseline = onp.hypot(u, v)
    positive = baseline > 0
    b_min, b_max = baseline[positive].min(), baseline[positive].max()
    return {
        "baseline_min_m": float(b_min),
        "baseline_max_m": float(b_max),
        "wavel_min_m": float(wavel.min()),
        "wavel_max_m": float(wavel.max()),
        # Half of λ/B_max: inside it, separation and flux are degenerate.
        "resolution_mas": float(0.5 * wavel.min() / b_max * MAS_PER_RAD),
        # λ/B_min: the field over which the coverage is unambiguous.
        "fov_mas": float(wavel.max() / b_min * MAS_PER_RAD),
        "lambda_over_b_mas": float(wavel.min() / b_max * MAS_PER_RAD),
    }


def _chi2(model, data, **noise):
    from ..likelihood import whitened_residuals

    r = onp.asarray(whitened_residuals(model, data, **noise), dtype=float)
    return float(onp.sum(r**2)), r


def _moments(r):
    r = onp.asarray(r, dtype=float)
    z = (r - r.mean()) / (r.std() or 1.0)
    return float(onp.mean(z**3)), float(onp.mean(z**4) - 3.0)


GLOBAL_NSIGMA_METHOD = "Sidak estimate, not a simulated FAP"


def _wrap(angle):
    return (onp.asarray(angle) + 360.0) % 360.0


def _quantiles(x, angle=False):
    """Median, 16th and 84th percentiles; angles about their mean direction."""
    x = onp.asarray(x, dtype=float).ravel()
    if angle:
        mean = math.degrees(
            math.atan2(
                onp.mean(onp.sin(onp.radians(x))),
                onp.mean(onp.cos(onp.radians(x))),
            )
        )
        dev = (x - mean + 180.0) % 360.0 - 180.0
        q = onp.percentile(dev, [50, 16, 84])
        median = float(_wrap(q[0] + mean))
        q = onp.array([median, median + q[1] - q[0], median + q[2] - q[0]])
    else:
        q = onp.percentile(x, [50, 16, 84])
    return {"median": float(q[0]), "q16": float(q[1]), "q84": float(q[2])}


class BinaryPipeline(_Pipeline):
    """Search for a companion, set contrast limits, fit and sample a binary.

    Stages: ``load → overview → search → limits → fit → posterior →
    quicklook``.

    * ``load`` applies ``wavel_range`` and ``error_floor`` and writes the
      data that are fitted to ``data/processed.oifits``.
    * ``overview`` plots the data and their uv coverage.
    * ``search`` runs
      [`detection_statistics`][virgil.detection.detection_statistics]
      (Δχ², log Bayes factor, best SNR) on a point-companion grid
      (``dra``, ``ddec`` in mas, log-spaced ``flux``), with maps of Δχ²,
      best flux and SNR, and the significance at the peak
      ([`local_nsigma`][virgil.detection.local_nsigma]) before and after a
      look-elsewhere correction for the number of resolution elements
      searched.
    * ``limits`` runs [`absil_limits`][virgil.limits.absil_limits] at
      ``sigma`` and its [`radial_profile`][virgil.limits.radial_profile].
    * ``fit`` runs [`fit`][virgil.fitting.fit] on the model template from
      the best grid point.
    * ``posterior`` samples
      [`numpyro_model`][virgil.likelihood.numpyro_model] with NUTS from
      the fit.
    * ``quicklook`` writes and executes ``quicklook.ipynb``.

    The fit is on the quoted errors unless ``error_scale="fit"``, and χ²/N
    is always reported on the quoted errors. Priors are group-invariant:
    uniform in position (``dra``, ``ddec`` or ``sep``), uniform in
    position angle (an [`AngleVector`][virgil.angles.AngleVector], with
    no wrap at 0°/360°) and log-uniform in flux over ``flux_range``.

    Parameters
    ----------
    data : OIData
        The data.
    model : BinaryModelAngular or BinaryModelCartesian, optional
        Model template; its class sets the fitted parameters (``sep``,
        ``pa``, ``flux`` or ``dra``, ``ddec``, ``flux``). Default
        ``BinaryModelAngular``. The search and limits always use a point
        companion in Cartesian offsets.
    output : str or os.PathLike, optional
        Run folder (default ``"run"``).
    **settings
        Overrides of the defaults (see ``BinaryPipeline.defaults()``):

        - ``sigma`` (3.0): significance of the contrast limits.
        - ``detection_sigma`` (3.0): threshold of the detection check, on
          the look-elsewhere-corrected significance.
        - ``max_sep_mas`` (None): half-width of the search grid; by
          default ``fov_fraction`` × λ_max/B_min.
        - ``fov_fraction`` (1.0): see ``max_sep_mas``.
        - ``grid_step_mas`` (None): grid spacing; by default λ_min/(4
          B_max), coarsened so that no axis exceeds ``max_grid`` points.
        - ``max_grid`` (101): most points per position axis.
        - ``flux_range`` ([1e-5, 0.5]): flux axis of the grid and bounds
          of the log-uniform flux prior (companion/primary).
        - ``n_flux`` (40): points on the flux axis.
        - ``wavel_range`` (None): ``[min, max]`` wavelengths to keep (m).
        - ``error_floor`` (None): absolute error floors by observable,
          e.g. ``{"vis": 0.01, "phi": 0.005}`` (phases in radians), as
          [`OIData.with_error_floor`][virgil.oidata.OIData.with_error_floor].
        - ``error_scale`` ("quoted"): ``"fit"`` also fits log-uniform
          error scales ``vis_scale`` and ``phi_scale`` on [0.1, 10].
        - ``num_warmup``, ``num_samples``, ``num_chains`` (1000, 1000, 4),
          ``chain_method`` ("vectorized") and ``seed`` (0): NUTS.
        - ``batch_size`` (None): grid batch size.
        - ``limit_bins`` (20): annuli of the contrast curve.
        - ``quicklook_execute`` (True): execute the quicklook notebook.

    Examples
    --------
    >>> import virgil as vg  # doctest: +SKIP
    >>> from virgil.pipeline import BinaryPipeline, load  # doctest: +SKIP
    >>> res = BinaryPipeline(vg.OIData("hd1234.oifits"), output="runs/hd1234").run()  # doctest: +SKIP
    >>> res.summary["companion"]["sep_mas"]  # doctest: +SKIP
    """

    NAME = "binary"
    STABILITY = "stable"
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
        "max_sep_mas": None,
        "fov_fraction": 1.0,
        "grid_step_mas": None,
        "max_grid": 101,
        "flux_range": [1e-5, 0.5],
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
        self._processed = None

    def _default_model(self):
        from ..models import BinaryModelAngular

        return BinaryModelAngular(sep=100.0, pa=0.0, flux=1e-3)

    @classmethod
    def _validate(cls, s):
        def positive(name):
            if not (isinstance(s[name], (int, float)) and s[name] > 0):
                raise ValueError(f"{name} must be positive; got {s[name]!r}.")

        for name in ("sigma", "detection_sigma", "fov_fraction"):
            positive(name)
        for name in ("max_sep_mas", "grid_step_mas", "batch_size"):
            if s[name] is not None:
                positive(name)
        for name in ("num_samples", "num_chains", "max_grid", "limit_bins"):
            if not (isinstance(s[name], int) and s[name] >= 1):
                raise ValueError(f"{name} must be an integer >= 1.")
        if not (isinstance(s["num_warmup"], int) and s["num_warmup"] >= 0):
            raise ValueError("num_warmup must be an integer >= 0.")
        if not (isinstance(s["n_flux"], int) and s["n_flux"] >= 2):
            raise ValueError("n_flux must be an integer >= 2.")
        lo, hi = s["flux_range"]
        if not 0 < lo < hi:
            raise ValueError("flux_range must satisfy 0 < min < max.")
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

    # --- shared state, rebuilt on demand so that resume works ------------

    @property
    def processed(self):
        """The data after ``wavel_range`` and ``error_floor``."""
        if self._processed is None:
            data = self.data
            s = self.settings
            if s["wavel_range"] is not None:
                data = data.select(*s["wavel_range"])
            if s["error_floor"] is not None:
                data = data.with_error_floor(absolute=s["error_floor"])
            self._processed = data
        return self._processed

    def _grid(self):
        import jax.numpy as np

        s = self.settings
        geo = _geometry(self.processed)
        max_sep = s["max_sep_mas"] or s["fov_fraction"] * geo["fov_mas"]
        step = s["grid_step_mas"] or 0.25 * geo["lambda_over_b_mas"]
        n = 2 * math.ceil(max_sep / step) + 1
        n = min(n, s["max_grid"] if s["max_grid"] % 2 else s["max_grid"] + 1)
        axis = np.linspace(-max_sep, max_sep, n)
        lo, hi = s["flux_range"]
        return {
            "dra": axis,
            "ddec": axis,
            "flux": np.logspace(math.log10(lo), math.log10(hi), s["n_flux"]),
        }, max_sep

    def _priors(self, max_sep):
        """Priors on ``dra``, ``ddec`` (uniform on the box) and ``flux``.

        Both templates sample these, so they share one prior on the sky:
        uniform in position is the translation-invariant choice, and a
        uniform ``sep`` would put p(dra, ddec) proportional to 1/sep.
        """
        import numpyro.distributions as dist

        lo, hi = self.settings["flux_range"]
        return {
            "dra": dist.Uniform(-max_sep, max_sep),
            "ddec": dist.Uniform(-max_sep, max_sep),
            "flux": dist.LogUniform(lo, hi),
        }

    def _sampled_model(self):
        """The template as a function of the sampled ``dra``, ``ddec``, ``flux``.

        A ``BinaryModelAngular`` template gets its ``sep`` and ``pa`` from
        the offsets, so that its fit and posterior have the same prior as a
        ``BinaryModelCartesian`` one.
        """
        from ..models import BinaryModelAngular, BinaryModelCartesian

        if not isinstance(self.model, BinaryModelAngular):
            return BinaryModelCartesian(0.0, 0.0, 0.0)
        import jax.numpy as np

        def model(dra, ddec, flux):
            sep = np.hypot(dra, ddec)
            pa = np.degrees(np.arctan2(dra, ddec)) % 360.0
            return BinaryModelAngular(sep, pa, flux)

        return model

    def _noise(self):
        import numpyro.distributions as dist

        if self.settings["error_scale"] != "fit":
            return None
        noise = {"vis_scale": dist.LogUniform(0.1, 10.0)}
        if onp.asarray(self.processed.phi).size:
            noise["phi_scale"] = dist.LogUniform(0.1, 10.0)
        return noise

    def _report(self, stage):
        return _io.read_json(self.output / "stages" / stage / "report.json")

    def _stages(self):
        return (
            Stage("load", _load, ("stages/load/report.json",)),
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

    def _summarise(self, reports):
        return _summarise(self.settings, reports)


# === STAGES ===


def _load(p):
    from ..oifits import write_oifits

    data = p.processed
    geo = _geometry(data)
    tables, reason = _io.oidata_tables(data)
    report = {
        "n_vis": int(onp.asarray(data.vis).size),
        "n_phi": int(onp.asarray(data.phi).size),
        "n_independent": int(data.n_independent),
        "closure_phases": bool(data.cp_flag),
        **geo,
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


def _search(p):
    import jax.numpy as np

    from ..detection import detection_statistics, local_nsigma
    from ..grid_fit import (
        laplace_flux_uncertainty_grid,
        optimized_flux_grid,
        optimized_likelihood_grid,
    )
    from ..likelihood import model_loglike
    from ..models import BinaryModelCartesian
    from ..plotting import plot_grid_map
    from scipy.stats import norm

    data, s = p.processed, p.settings
    grid, max_sep = p._grid()
    kw = {"batch_size": s["batch_size"]}
    stats = detection_statistics(BinaryModelCartesian, data, grid, **kw)
    loglike = onp.asarray(
        optimized_likelihood_grid(BinaryModelCartesian, data, grid, **kw)
    )
    flux = onp.asarray(
        optimized_flux_grid(BinaryModelCartesian, data, grid, **kw)
    )
    sigma = onp.asarray(
        laplace_flux_uncertainty_grid(
            BinaryModelCartesian, data, grid, flux=np.asarray(flux), **kw
        )
    )
    null = BinaryModelCartesian(0.0, 0.0, 0.0)
    ll0 = float(model_loglike(null, data))
    delta_chi2 = onp.where(
        flux > 0, onp.maximum(2.0 * (loglike - ll0), 0.0), 0.0
    )
    snr = flux / sigma
    peak = onp.unravel_index(int(onp.nanargmax(delta_chi2)), delta_chi2.shape)
    flux_axis = onp.asarray(grid["flux"])
    flux_index = int(
        onp.argmin(
            onp.abs(
                onp.log(flux_axis)
                - onp.log(max(float(stats["flux"]), flux_axis[0]))
            )
        )
    )

    dchi2 = float(stats["delta_chi2"])
    local = float(local_nsigma(dchi2))
    geo = _geometry(data)
    n_trials = max(1.0, (2.0 * max_sep / geo["lambda_over_b_mas"]) ** 2)
    p_local = float(norm.sf(local))
    p_global = -math.expm1(n_trials * math.log1p(-min(p_local, 1 - 1e-16)))
    global_sigma = float(norm.isf(p_global)) if p_global > 0 else local
    chi2_null, _ = _chi2(null, data)

    axes = {k: onp.asarray(v) for k, v in grid.items()}
    _io.write_h5(
        p.output / "grids.h5",
        {
            "axes": {
                "dra": (axes["dra"], {"units": "mas"}),
                "ddec": (axes["ddec"], {"units": "mas"}),
                "flux": (axes["flux"], {"units": "companion/primary"}),
            },
            "search": {
                "delta_chi2": delta_chi2,
                "loglike": loglike,
                "flux_best": flux,
                "flux_sigma": sigma,
                "snr": snr,
            },
        },
        attrs={
            "/": {"axis_order": ["dra", "ddec", "flux"], "schema": _io.SCHEMA}
        },
    )
    best = {"dra": float(stats["dra"]), "ddec": float(stats["ddec"])}
    fig, _ = plot_grid_map(
        delta_chi2,
        grid,
        kind="loglike",
        label="Δχ²",
        title="Δχ² (point companion)",
        best=best,
    )
    _save(fig, p.output / "plots" / "search_delta_chi2.png")
    fig, _ = plot_grid_map(snr, grid, kind="snr", best=best)
    _save(fig, p.output / "plots" / "search_snr.png")
    return {
        "delta_chi2": dchi2,
        "log_bayes_factor": float(stats["log_bayes_factor"]),
        "max_snr": float(stats["max_snr"]),
        "dra_mas": best["dra"],
        "ddec_mas": best["ddec"],
        "flux": float(stats["flux"]),
        "local_nsigma": local,
        "global_nsigma": global_sigma,
        "global_nsigma_method": GLOBAL_NSIGMA_METHOD,
        "n_trials": n_trials,
        "max_sep_mas": max_sep,
        "grid_shape": [len(axes["dra"]), len(axes["ddec"]), len(axes["flux"])],
        "peak_index": [int(peak[0]), int(peak[1]), flux_index],
        "chi2_null": chi2_null,
        "n_independent": int(data.n_independent),
    }


def _limits(p):
    from ..limits import absil_limits, flux_to_delta_mag, radial_profile
    from ..models import BinaryModelCartesian
    from ..plotting import plot_contrast_curve, plot_grid_map

    data, s = p.processed, p.settings
    grid, max_sep = p._grid()
    limit = onp.asarray(
        absil_limits(
            BinaryModelCartesian,
            data,
            grid,
            s["sigma"],
            batch_size=s["batch_size"],
        )
    )
    geo = _geometry(data)
    dra, ddec = onp.asarray(grid["dra"]), onp.asarray(grid["ddec"])
    r = onp.hypot(*onp.meshgrid(dra, ddec, indexing="ij"))
    # Inside the resolution limit separation and flux are degenerate.
    limit = onp.where(r < geo["resolution_mas"], onp.nan, limit)
    profile = radial_profile(
        limit, dra, ddec, r_max=max_sep, bins=s["limit_bins"]
    )

    path = p.output / "grids.h5"
    groups = {
        "axes": {
            k: (v, {"units": u})
            for (k, v), u in zip(
                _io.read_h5(path, "axes").items(),
                ("mas", "mas", "companion/primary"),
            )
        },
        "search": _io.read_h5(path, "search"),
        "limits": {
            "limit_flux": (limit, {"sigma": s["sigma"], "method": "absil"}),
        },
    }
    # read_h5 returns the axes sorted by name; restore the grid order.
    groups["axes"] = {k: groups["axes"][k] for k in ("dra", "ddec", "flux")}
    _io.write_h5(
        path,
        groups,
        attrs={
            "/": {"axis_order": ["dra", "ddec", "flux"], "schema": _io.SCHEMA}
        },
    )
    fig, _ = plot_grid_map(
        limit, grid, kind="limit", units="delta_mag", sigma=s["sigma"]
    )
    _save(fig, p.output / "plots" / "limits_map.png")
    fig, _ = plot_contrast_curve(profile, sigma=s["sigma"])
    _save(fig, p.output / "plots" / "limits_contrast_curve.png")
    median = onp.asarray(profile["median"], dtype=float)
    return {
        "sigma": s["sigma"],
        "method": "absil",
        "r_mas": profile["r"],
        "median_flux": median,
        "median_delta_mag": onp.asarray(
            flux_to_delta_mag(median), dtype=float
        ),
        "deepest_delta_mag": float(onp.nanmax(flux_to_delta_mag(median))),
    }


def _fit(p):
    from ..fitting import fit
    from ..likelihood import posterior_predictive_summary
    from ..models import BinaryModelAngular
    from ..plotting import plot_data_model_correlation

    data, s = p.processed, p.settings
    search = p._report("search")
    _, max_sep = p._grid()
    priors = p._priors(max_sep)
    lo, hi = s["flux_range"]
    flux0 = min(max(search["flux"], 2.0 * lo), 0.5 * hi)
    start = {
        "dra": search["dra_mas"],
        "ddec": search["ddec_mas"],
        "flux": flux0,
    }
    params = list(priors)
    angular = isinstance(p.model, BinaryModelAngular)
    template = p._sampled_model()
    if angular:
        init = {k: onp.asarray(v, dtype=float) for k, v in start.items()}
    else:
        template = template.set(
            params, [onp.asarray(start[k], dtype=float) for k in params]
        )
        init = None
    noise = p._noise()
    result = fit(template, priors, data, noise=noise, init=init)
    values = {k: onp.asarray(v) for k, v in result.values.items()}
    if angular:
        # The model's own parameters, derived from the fitted offsets.
        m = result.model
        values.update(sep=onp.asarray(m.sep), pa=onp.asarray(m.pa))
    reported = ["sep", "pa", "flux"] if angular else params
    _io.save_model(
        p.output / "models" / "best", result.model, values, reported
    )
    _io.write_json(
        p.output / "models" / "best" / "info.json",
        _io.clean_json(dict(result.info)),
    )

    chi2, resid = _chi2(result.model, data)
    # The periodic penalty terms of correlated closure phases are not
    # meant to be normal: test only the independent whitened residuals.
    resid = resid[: int(data.n_independent)]
    skew, kurt = _moments(resid)
    one = {k: onp.asarray([float(values[k])]) for k in params}
    pred = posterior_predictive_summary(
        one, p._sampled_model() if angular else result.model, data, params
    )
    fig = plot_data_model_correlation(data, {"MAP fit": pred})
    fig = fig[0] if isinstance(fig, tuple) else fig
    _save(fig, p.output / "plots" / "fit_correlation.png")
    scales = {
        k.split(".", 1)[1]: float(v)
        for k, v in values.items()
        if k.startswith("noise.")
    }
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
        "priors": {k: repr(v) for k, v in priors.items()},
    }


def _posterior(p):
    import jax
    import pandas as pd
    from numpyro.diagnostics import effective_sample_size, split_gelman_rubin
    from numpyro.infer import MCMC, NUTS

    from ..likelihood import chain_init_params, numpyro_model
    from ..plotting import plot_chainconsumer_diagnostics

    data, s = p.processed, p.settings
    fit_report = p._report("fit")
    _, max_sep = p._grid()
    priors = p._priors(max_sep)
    params = list(priors)
    reported = list(fit_report["params"])
    angular = reported != params
    model = p._sampled_model()
    if not angular:
        model = _io.load_model(p.output / "models" / "best")
    start = {
        k: onp.asarray(v)
        for k, v in _io.load_model_values(p.output / "models" / "best").items()
        if k in params
    }
    posterior = numpyro_model(model, priors, data, noise=p._noise())
    key = jax.random.PRNGKey(s["seed"])
    init_key, run_key = jax.random.split(key)
    init = chain_init_params(
        posterior, [start] * s["num_chains"], key=init_key
    )
    mcmc = MCMC(
        NUTS(posterior),
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
        k: onp.asarray(v)
        for k, v in mcmc.get_samples(group_by_chain=True).items()
    }
    sampled = dict(samples)
    if angular:
        # The model's own parameters, derived from the sampled offsets.
        samples["sep"] = onp.hypot(sampled["dra"], sampled["ddec"])
        samples["pa"] = _wrap(
            onp.degrees(onp.arctan2(sampled["dra"], sampled["ddec"]))
        )
    extra = mcmc.get_extra_fields(group_by_chain=True)
    stats = {
        ("step_size" if k == "adapt_state.step_size" else k): onp.asarray(v)
        for k, v in extra.items()
    }
    _io.write_h5(
        p.output / "samples.h5",
        {"posterior": samples, "sample_stats": stats},
        attrs={
            "/": {
                "schema": _io.SCHEMA,
                "params": reported,
                "priors": fit_report["priors"],
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

    sites = [k for k in sampled if k + "_vec" not in sampled]
    diagnostics = {}
    for site in sites:
        x = samples[site]
        rhat = float(onp.nanmax(onp.asarray(split_gelman_rubin(x))))
        n_eff = float(onp.nanmin(onp.asarray(effective_sample_size(x))))
        diagnostics[site] = {"r_hat": rhat, "ess_bulk": n_eff}
    derived = {}
    if angular:
        # Diagnostics of sep and pa; an angle through its sine and cosine.
        parts = {
            "sep": [samples["sep"]],
            "pa": [
                onp.sin(onp.radians(samples["pa"])),
                onp.cos(onp.radians(samples["pa"])),
            ],
        }
        for k, xs in parts.items():
            derived[k] = {
                "r_hat": max(
                    float(onp.nanmax(onp.asarray(split_gelman_rubin(x))))
                    for x in xs
                ),
                "ess_bulk": min(
                    float(onp.nanmin(onp.asarray(effective_sample_size(x))))
                    for x in xs
                ),
            }
    summary = {}
    for k in reported:
        summary[k] = {
            **_quantiles(samples[k], angle=(k == "pa")),
            **derived.get(
                k, diagnostics.get(k + "_vec", diagnostics.get(k, {}))
            ),
        }
    bound = {}
    for k in params:
        prior = priors[k]
        if not hasattr(prior, "low"):
            continue
        lo, hi = float(prior.low), float(prior.high)
        x = onp.asarray(samples[k], dtype=float).ravel()
        if type(prior).__name__ == "LogUniform":
            lo, hi, x = math.log(lo), math.log(hi), onp.log(x)
        edge = _BOUND_FRACTION * (hi - lo)
        bound[k] = float(onp.mean((x < lo + edge) | (x > hi - edge)))
    divergent = float(onp.mean(stats["diverging"]))

    flat = {k: onp.asarray(samples[k], dtype=float).ravel() for k in reported}
    fig_corner = fig_walk = None
    _, fig_corner, fig_walk = plot_chainconsumer_diagnostics(
        {"posterior": pd.DataFrame(flat)}, columns=reported
    )
    _save(fig_corner, p.output / "plots" / "posterior_corner.png")
    _save(fig_walk, p.output / "plots" / "posterior_trace.png")
    noise_sites = {
        k.split(".", 1)[1]: _quantiles(samples[k])
        for k in samples
        if k.startswith("noise.")
    }
    return {
        "params": summary,
        "noise": noise_sites or None,
        "num_chains": s["num_chains"],
        "num_samples": s["num_samples"],
        "num_warmup": s["num_warmup"],
        "r_hat_max": max(d["r_hat"] for d in diagnostics.values()),
        "ess_bulk_min": min(d["ess_bulk"] for d in diagnostics.values()),
        "divergence_fraction": divergent,
        "prior_bound_fraction": bound,
    }


def _quicklook(p):
    from . import _quicklook as ql

    ql.write_quicklook(p.output, execute=p.settings["quicklook_execute"])
    return {"executed": bool(p.settings["quicklook_execute"])}


# === SUMMARY AND CHECKS ===


def _companion(params, source):
    """Companion position and flux (median and 1σ) from fit or posterior."""
    from ..limits import flux_to_contrast, flux_to_delta_mag

    def q(name):
        x = params[name]
        if isinstance(x, dict):
            return x["median"], 0.5 * (x["q84"] - x["q16"])
        return x, None

    out = {"source": source}
    if "sep" in params:
        (sep, sep_err), (pa, pa_err) = q("sep"), q("pa")
        out.update(
            sep_mas=sep, sep_err_mas=sep_err, pa_deg=pa, pa_err_deg=pa_err
        )
        th = math.radians(pa)
        out.update(dra_mas=sep * math.sin(th), ddec_mas=sep * math.cos(th))
    else:
        (dra, _), (ddec, _) = q("dra"), q("ddec")
        out.update(dra_mas=dra, ddec_mas=ddec)
        out.update(
            sep_mas=math.hypot(dra, ddec),
            sep_err_mas=None,
            pa_deg=float(_wrap(math.degrees(math.atan2(dra, ddec)))),
            pa_err_deg=None,
        )
    flux, flux_err = q("flux")
    out.update(
        flux=flux,
        flux_err=flux_err,
        contrast=float(flux_to_contrast(flux)),
        delta_mag=float(flux_to_delta_mag(flux)),
    )
    return out


def _summarise(settings, reports):
    sections, checks = {}, []
    load = reports.get("load")
    if load:
        sections["data"] = {
            k: load[k]
            for k in (
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
        }
    search = reports.get("search")
    fit = reports.get("fit")
    post = reports.get("posterior")
    if search:
        n = search["n_independent"]
        sections["chi2"] = {
            "n_independent": n,
            "null_reduced": search["chi2_null"] / n,
            "companion_reduced": fit["chi2_reduced"] if fit else None,
            "error_scale": settings["error_scale"],
            "error_scales": fit["error_scales"] if fit else None,
        }
        sections["search"] = {
            k: search[k]
            for k in (
                "delta_chi2",
                "log_bayes_factor",
                "max_snr",
                "local_nsigma",
                "global_nsigma",
                "global_nsigma_method",
                "n_trials",
                "dra_mas",
                "ddec_mas",
                "flux",
                "max_sep_mas",
            )
        }
        checks.append(
            _checks.detection(
                search["local_nsigma"],
                search["global_nsigma"],
                threshold=settings["detection_sigma"],
            )
        )
        checks.append(
            # The flux axis only seeds the optimizer: not a search boundary.
            _checks.grid_edge(
                search["peak_index"][:2],
                search["grid_shape"][:2],
                ("dra", "ddec"),
            )
        )
        if not fit:
            checks.insert(
                0,
                _checks.chi2_reduced(
                    "chi2", search["chi2_null"] / n, label="the star alone"
                ),
            )
    limits = reports.get("limits")
    if limits:
        sections["limits"] = {
            k: limits[k]
            for k in (
                "sigma",
                "method",
                "r_mas",
                "median_flux",
                "median_delta_mag",
                "deepest_delta_mag",
            )
        }
    if fit:
        sections["fit"] = {
            k: fit[k]
            for k in ("params", "converged", "chi2_reduced", "at_bound")
        }
        null = (
            search["chi2_null"] / search["n_independent"] if search else None
        )
        label = "the best-fit binary" + (
            f" (star alone: {null:.3g})" if null is not None else ""
        )
        checks.insert(
            0, _checks.chi2_reduced("chi2", fit["chi2_reduced"], label=label)
        )
        if fit["error_scales"]:
            checks.append(_checks.error_scale(fit["error_scales"]))
        checks.append(
            _checks.residual_normality(
                fit["residual_skew"],
                fit["residual_excess_kurtosis"],
                fit["n_residuals"],
            )
        )
        source = post["params"] if post else fit["params"]
        sections["companion"] = _companion(
            source, "posterior" if post else "fit"
        )
        if load:
            checks.append(
                _checks.field_of_view(
                    sections["companion"]["sep_mas"],
                    load["resolution_mas"],
                    load["fov_mas"],
                )
            )
    if post:
        sections["posterior"] = {
            k: post[k]
            for k in (
                "params",
                "noise",
                "num_chains",
                "num_samples",
                "num_warmup",
                "r_hat_max",
                "ess_bulk_min",
                "divergence_fraction",
            )
        }
        checks.append(_checks.prior_bound(post["prior_bound_fraction"]))
        checks.append(_checks.r_hat(post["r_hat_max"]))
        checks.append(_checks.ess(post["ess_bulk_min"]))
        checks.append(_checks.divergences(post["divergence_fraction"]))
    return sections, checks
