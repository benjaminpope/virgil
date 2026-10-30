"""Helpers shared by the pipelines: χ² and −2 log L (with error scales
profiled over a grid), residual moments, quantiles, geometry of the data,
prior-bound fractions and figure saving."""

from __future__ import annotations

import math
import numbers
import time

import equinox as eqx
import jax
import jax.numpy as np
import numpy as onp
from scipy import stats

from .._grid import batch_size_or_default
from .._precision import cast_tree, in_float64, run_in
from ..likelihood import model_loglike
from . import _checks, _io

# Fraction of a prior's range, at either end, counted as "at the bound".
_BOUND_FRACTION = 0.01


def _bound_samples(prior, samples):
    """Fraction of ``samples`` in the outer ``_BOUND_FRACTION`` of the
    prior's finite range at either end, or ``None`` if it has no edges.

    The range is the one ``fit`` reports (``fitting._bound_fraction``): a
    LogUniform's in log x, an interval support's, and none for tied terms,
    angle vectors, or priors with a flat coordinate (their ends are poles).
    """
    from ..fitting import _bound_fraction

    x = onp.asarray(samples, dtype=float).ravel()
    fraction = _bound_fraction(prior, x)
    if fraction is None:
        return None
    return float(
        onp.mean(
            (fraction < _BOUND_FRACTION) | (fraction > 1 - _BOUND_FRACTION)
        )
    )


def _save(fig, path):
    with _io.atomic_path(path) as tmp:
        fig.savefig(tmp, dpi=110, bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(fig)


def _geometry(data):
    """Baseline and resolution figures of ``data`` (metres, mas).

    Only the unflagged samples count (``OIData.unflagged``): a flagged
    sample keeps its baseline in the data but measures nothing.
    """
    from .._geometry import _unflagged_samples, fringe_scales

    baseline, wavel, _ = _unflagged_samples(data)
    finest, coarsest = fringe_scales(data)
    return {
        "baseline_min_m": float(baseline.min()),
        "baseline_max_m": float(baseline.max()),
        "wavel_min_m": float(wavel.min()),
        "wavel_max_m": float(wavel.max()),
        # Half of the finest λ/B: inside it, separation and flux are
        # degenerate.
        "resolution_mas": float(0.5 * finest),
        # The coarsest λ/B: the field over which the coverage is
        # unambiguous.
        "fov_mas": float(coarsest),
        "lambda_over_b_mas": float(finest),
    }


def _load_report(p, **extra):
    """The ``load`` stage: the data's summary report, and its OIFITS copy.

    ``extra`` entries are added to the report after the geometry. A stale
    ``processed.oifits`` of an earlier run is removed when the data cannot
    be written (``_oidata_tables`` returns no tables).
    """
    from ..oifits import _oidata_tables, write_oifits

    data = p.processed
    tables, reason = _oidata_tables(data)
    report = {
        "n_vis": int(onp.asarray(data.vis).size),
        "n_phi": int(onp.asarray(data.phi).size),
        "n_independent": int(data.n_independent),
        "closure_phases": bool(data.cp_flag),
        **_geometry(data),
        **extra,
        "data_sha256": _io.data_fingerprint(data),
        "processed_oifits_skipped": reason,
    }
    path = p.output / "data" / "processed.oifits"
    if tables is None:
        path.unlink(missing_ok=True)  # not an earlier run's data
    else:
        with _io.atomic_path(path) as tmp:
            write_oifits(tables, tmp)
    return report


def _chi2(model, data, **noise):
    from ..likelihood import whitened_residuals

    r = onp.asarray(whitened_residuals(model, data, **noise), dtype=float)
    return float(onp.sum(r**2)), r


def _overview_plots(p, data):
    """Save the overview of ``data`` and its (u, v) coverage."""
    from ..plotting import plot_oidata_overview, plot_uv_coverage

    fig, _ = plot_oidata_overview(data)
    _save(fig, p.output / "plots" / "overview_data.png")
    fig, _ = plot_uv_coverage(data)
    _save(fig, p.output / "plots" / "overview_uv.png")


def _overview(p):
    _overview_plots(p, p.processed)
    return {}


def _quicklook(p):
    from . import _quicklook as ql

    ql.write_quicklook(p.output, execute=p.settings["quicklook_execute"])
    return {"executed": bool(p.settings["quicklook_execute"])}


def _scales_of(values):
    """The fitted error scales among a fit's ``values``, by name."""
    return {
        k.split(".", 1)[1]: float(v)
        for k, v in values.items()
        if k.startswith("noise.")
    }


@in_float64
def _chi2_64(model, data, **noise):
    return _chi2(model, data, **noise)


@in_float64
def _m2loglike_64(model, data, **noise):
    """−2 log L of ``model``, including the error terms' normalization.

    This is the score for comparing models and lobes when error scales are
    fitted: at each candidate's own scales χ²/N is 1 by construction, but
    −2 log L keeps the −N log s term that penalizes inflated errors. On
    the quoted errors it differs from χ² by a constant.
    """
    return -2.0 * float(model_loglike(model, data, **noise))


def _scale_blocks(data, noise):
    """Where each fitted error scale applies: ``((key, start, n, low, high),
    ...)``, with the scale's name, the first row and number of independent
    rows of its block of ``whitened_residuals``, and its prior range.

    Concrete (hashable, so a jitted function can take it as static).
    """
    from ..epochs import _blocks

    ranges = {
        k: (float(d.low), float(d.high)) for k, d in (noise or {}).items()
    }
    return tuple(
        (f"{name}_scale", start, nu, *ranges[f"{name}_scale"])
        for name, start, _, nu, _ in _blocks(data)
        if f"{name}_scale" in ranges
    )


_GOLDEN_STEPS = (
    48  # golden-section steps in ln s: a bracket of 1e-10 of ln(100)
)


def _golden_max(f, lo, hi, steps=_GOLDEN_STEPS):
    """The maximum of a unimodal ``f`` on ``[lo, hi]`` by golden section
    (traceable; ``steps`` fixed)."""
    g = 0.5 * (np.sqrt(5.0) - 1.0)

    def step(_, bracket):
        a, b = bracket
        c, d = b - g * (b - a), a + g * (b - a)
        left = f(c) >= f(d)
        return np.where(left, a, c), np.where(left, d, b)

    a, b = jax.lax.fori_loop(0, steps, step, (lo, hi))
    return 0.5 * (a + b)


def _profiled_m2loglike(build, start, data, diam, blocks):
    """``(−2 log L, scales)`` of ``build({**start, "diam": diam})`` at its
    profiled error scales. Traceable.

    The profiled scales are the ones that maximize the likelihood of the
    model. Each scale multiplies the errors of one block of ``_scale_blocks``
    and enters only that block's terms of ``model_loglike``, so each is
    maximized on its own, in ln s over its prior range. The block's log
    likelihood is unimodal in ln s: ``-χ²/(2s²) - N ln s`` for Gaussian
    blocks, and concave in the concentration κ = 1/(sσ)² for von Mises
    phases, whose maximum is not at ``sqrt(χ²/N)``.
    """
    from ..likelihood import whitened_residuals

    model = build({**start, "diam": diam})
    r = whitened_residuals(model, data)
    scales = {
        key: np.clip(np.sqrt(np.sum(r[i : i + n] ** 2) / n), low, high)
        for key, i, n, low, high in blocks
    }
    for key, _, _, low, high in blocks:

        def loglike(log_s, key=key):
            return model_loglike(model, data, **{**scales, key: np.exp(log_s)})

        bounds = np.log(np.asarray([low, high], r.dtype))
        scales[key] = np.exp(_golden_max(loglike, bounds[0], bounds[1]))
    return -2.0 * model_loglike(model, data, **scales), scales


@eqx.filter_jit
def _m2loglike_grid_jit(build, start, data, diams, blocks, batch_size):
    def one(diam):
        return _profiled_m2loglike(build, start, data, diam, blocks)

    return jax.lax.map(one, diams, batch_size=batch_size)


@eqx.filter_jit
def _refine_diameter_jit(build, start, data, low, high, blocks):
    def loglike(log_diam):
        m2, _ = _profiled_m2loglike(
            build, start, data, np.exp(log_diam), blocks
        )
        return -0.5 * m2

    diam = np.exp(_golden_max(loglike, np.log(low), np.log(high)))
    return diam, *_profiled_m2loglike(build, start, data, diam, blocks)


def _m2loglike_grid(  # deliberate: float64 wrapper of the jit entry, test seam
    build, start, data, diams, noise=None, batch_size=None
):
    """−2 log L of ``build({**start, "diam": d})`` over the diameters ``d``.

    One batched, jitted evaluation in float64. With ``noise`` (the
    pipeline's error-scale priors) the scales are profiled at every
    diameter (``_profiled_m2loglike``) and the likelihood is evaluated at
    them. ``build`` should be a module-level function, so that the compiled
    function is reused between calls; ``start`` holds the other parameters.

    Returns
    -------
    tuple
        ``(m2loglike, scales)``: an array over the diameters and a dict of
        arrays by scale name (empty without ``noise``).
    """
    with run_in("float64"):
        data = cast_tree(data, "float64")
        m2, scales = _m2loglike_grid_jit(
            build,
            {k: np.asarray(v) for k, v in start.items()},
            data,
            np.asarray(diams),
            _scale_blocks(data, noise),
            batch_size_or_default(batch_size, data),
        )
        return onp.asarray(m2), {k: onp.asarray(v) for k, v in scales.items()}


def _refine_diameter(
    build, start, data, low, high, noise
):  # deliberate: float64 wrapper of the jit entry
    """The diameter in ``[low, high]`` of least profiled −2 log L.

    Golden-section search in ln θ (the profiled likelihood is unimodal on
    a bracket this narrow), with the error scales of ``noise`` re-profiled
    at every trial diameter. ``build`` and ``start`` are as for
    ``_m2loglike_grid``.

    Returns
    -------
    tuple
        ``(diam, m2loglike, scales)``: floats and a dict of floats.
    """
    with run_in("float64"):
        data = cast_tree(data, "float64")
        diam, m2, scales = _refine_diameter_jit(
            build,
            {k: np.asarray(v) for k, v in start.items()},
            data,
            np.asarray(low),
            np.asarray(high),
            _scale_blocks(data, noise),
        )
        return (
            float(diam),
            float(m2),
            {k: float(v) for k, v in scales.items()},
        )


# Per-transition NUTS diagnostics kept in samples.h5.
_NUTS_FIELDS = (
    "diverging",
    "accept_prob",
    "adapt_state.step_size",
    "num_steps",
    "potential_energy",
)
# Settings recorded as attributes of samples.h5.
_SAMPLER_SETTINGS = (
    "num_warmup",
    "num_samples",
    "num_chains",
    "chain_method",
    "seed",
)


# How far inside its prior a NUTS start is pulled, as a fraction of the
# prior's width, measured as ``fit`` measures ``at_bound`` (in log x for a
# LogUniform). A start exactly on a bound has an infinite unconstrained
# value. Inside, it stays finite in float32 and float64 but its size depends
# on the sampler's coordinate: about -7 (``logit(1e-3)``) in the flat
# coordinates, and about ``logit(inset ln(R) / R)`` for a LogUniform with
# R = high / low in numpyro's linear interval (about -15 for R = 5e4).
_START_INSET = 1e-3


def _interior_start(priors, start, inset=_START_INSET):
    """``start`` with every value pulled at least ``inset`` of its prior's
    width inside the prior's finite bounds, as ``fit`` measures them.

    The fit's best values sit on a bound when the data cannot constrain a
    parameter (a companion's flux in a non-detection, the diameter of an
    unresolved star), and numpyro cannot start there: the unconstrained
    value is infinite. Values already inside, and parameters without
    finite bounds, are returned as they were.
    """
    from .._flat import _base, _flat_coordinate
    from ..fitting import _bound_fraction

    out = dict(start)
    for name, value in start.items():
        prior = priors.get(name)
        fraction = None if prior is None else _bound_fraction(prior, value)
        if fraction is None or not onp.any(
            (fraction < inset) | (fraction > 1.0 - inset)
        ):
            continue
        flat = _flat_coordinate(prior)
        if flat is not None:
            _, from_flat, low, high = flat
        else:
            support = _base(prior).support
            low, high = support.lower_bound, support.upper_bound
            from_flat = lambda x: x  # noqa: E731
        low, high = onp.asarray(low, float), onp.asarray(high, float)
        inside = onp.clip(fraction, inset, 1.0 - inset)
        out[name] = onp.asarray(
            from_flat(low + inside * (high - low)),
            dtype=onp.asarray(value).dtype,
        )
    return out


def _run_chains(
    posterior,
    settings,
    start,
    *,
    dense_mass,
    priors=None,
    inverse_mass=None,
    jitter=None,
    max_tree_depth=10,
    adapt_mass=True,
):
    """Run NUTS from ``start`` (one start per chain); the finished MCMC.

    With ``priors`` (by sample site), the start is first pulled just inside
    any bound it sits on (``_interior_start``). If numpyro still finds no
    finite log-density at the start, a ``ValueError`` says which values sit
    on a bound, instead of numpyro's own message.

    ``inverse_mass`` is the initial dense inverse mass matrix, in numpyro's
    order of the unconstrained sites (``ravel_pytree``: sorted by name);
    ``adapt_mass=False`` keeps it fixed (warm-up windows of a few hundred
    draws estimate a worse matrix than a Laplace one). ``jitter`` (a matrix ``L`` in the same order) starts
    chain ``k > 0`` at ``0.3 L ε`` from the start, ``ε`` standard normal, when
    the log-density is finite there; chain 0 starts at ``start``.
    """
    import jax
    from jax.flatten_util import ravel_pytree
    from numpyro.infer import MCMC, NUTS
    from numpyro.infer.util import potential_energy

    from ..likelihood import chain_init_params

    if priors is not None:
        start = _interior_start(priors, start)
    init_key, run_key = jax.random.split(jax.random.PRNGKey(settings["seed"]))
    try:
        init = chain_init_params(
            posterior, [start] * settings["num_chains"], key=init_key
        )
    except RuntimeError as error:
        if "valid initial parameters" not in str(error):
            raise
        raise ValueError(
            "NUTS cannot start from the best fit: the log-posterior or its "
            "gradient is not finite there. Start values: "
            + ", ".join(
                f"{k}={onp.asarray(v).tolist()}" for k, v in start.items()
            )
            + ". Check that the fitted model and its priors agree."
        ) from error
    n = settings["num_chains"]
    if jitter is not None and n > 1:
        z0 = jax.tree_util.tree_map(lambda a: a[0], init)
        flat, unravel = ravel_pytree(z0)
        eps = onp.random.default_rng(settings["seed"]).standard_normal(
            (n, len(flat))
        )
        chains = [z0]
        for k in range(1, n):
            z = unravel(flat + 0.3 * np.asarray(jitter) @ eps[k])
            ok = onp.isfinite(float(potential_energy(posterior, (), {}, z)))
            chains.append(z if ok else z0)
        init = jax.tree_util.tree_map(lambda *a: np.stack(a), *chains)
    kwargs = (
        {} if inverse_mass is None else {"inverse_mass_matrix": inverse_mass}
    )
    mcmc = MCMC(
        NUTS(
            posterior,
            dense_mass=dense_mass,
            max_tree_depth=max_tree_depth,
            adapt_mass_matrix=adapt_mass,
            **kwargs,
        ),
        num_warmup=settings["num_warmup"],
        num_samples=settings["num_samples"],
        num_chains=settings["num_chains"],
        chain_method=settings["chain_method"],
        progress_bar=False,
    )
    mcmc.run(run_key, init_params=init, extra_fields=_NUTS_FIELDS)
    return mcmc


def _deadline_factor(deadline, x):
    """A log-density term: 0 until ``time.monotonic()`` passes ``deadline``,
    then ``-inf``, so that every NUTS transition diverges at once and a
    stalled run drains in moments. ``x`` (not differentiated) keeps the
    clock read inside the compiled function, once per gradient."""
    # pure_callback: an io_callback cannot sit in a while_loop under vmap.
    expired = jax.pure_callback(
        lambda _: onp.asarray(time.monotonic() > deadline),
        jax.ShapeDtypeStruct((), bool),
        jax.lax.stop_gradient(x),
        vmap_method="sequential",
    )
    return np.where(expired, -np.inf, 0.0)


def _diagnostics(arrays):
    """Worst split R-hat and bulk ESS of ``arrays`` (each ``(chain, draw)``
    or ``(chain, draw, k)``). A NaN anywhere, such as the 0/0 R-hat of a
    stuck element, makes the worst value NaN, which the checks fail."""
    from numpyro.diagnostics import effective_sample_size, split_gelman_rubin

    return {
        "r_hat": float(
            onp.max([onp.max(split_gelman_rubin(x)) for x in arrays])
        ),
        "ess_bulk": float(
            onp.min([onp.min(effective_sample_size(x)) for x in arrays])
        ),
    }


def _noise_quantiles(samples):
    noise = {
        k.split(".", 1)[1]: _quantiles(samples[k])
        for k in samples
        if k.startswith("noise.")
    }
    return noise or None


# deliberate: constant-residual fallback (scipy gives NaN), pinned by test_moments_of_constant_residuals_are_finite
def _moments(r):
    r = onp.asarray(r, dtype=float)
    if r.std() <= 1e-12 * max(abs(r.mean()), 1.0):
        return 0.0, -3.0  # scipy gives NaN for (near-)constant residuals
    return float(stats.skew(r)), float(stats.kurtosis(r))


def _quantiles(x, angle=False, period=360.0):
    """Median, 16th and 84th percentiles; angles about their mean direction.

    ``period`` is the angle's period in degrees (180 for an axis, a position
    angle without a sense).
    """
    x = onp.asarray(x, dtype=float).ravel()
    if angle:
        scale = 360.0 / period
        x = x * scale
        mean = math.degrees(
            math.atan2(
                onp.mean(onp.sin(onp.radians(x))),
                onp.mean(onp.cos(onp.radians(x))),
            )
        )
        dev = (x - mean + 180.0) % 360.0 - 180.0
        q = onp.percentile(dev, [50, 16, 84])
        median = float((q[0] + mean + 360.0) % 360.0)
        q = (
            onp.array([median, median + q[1] - q[0], median + q[2] - q[0]])
            / scale
        )
    else:
        q = onp.percentile(x, [50, 16, 84])
    return {"median": float(q[0]), "q16": float(q[1]), "q84": float(q[2])}


# === POSTERIOR SCAFFOLDING ===
# The posterior stages differ in what they sample but not in how they report
# it: draws by site, chain health, a sampler block, one samples.h5 and a
# corner and trace figure.


def _draws(mcmc):
    """``(samples, stats)`` of a finished run: draws by site and the NUTS
    statistics, each ``(chain, draw)``."""
    samples = {
        k: onp.asarray(v)
        for k, v in mcmc.get_samples(group_by_chain=True).items()
    }
    extra = mcmc.get_extra_fields(group_by_chain=True)
    stats = {
        ("step_size" if k == "adapt_state.step_size" else k): onp.asarray(v)
        for k, v in extra.items()
    }
    return samples, stats


def _chain_health(diagnostics, stats):
    """Worst R-hat, smallest ESS and divergence fraction of one run, from
    its per-site ``_diagnostics`` and NUTS ``stats``. A NaN at any site makes
    the worst value NaN, which the R-hat and ESS checks fail."""
    return {
        "r_hat_max": float(
            onp.max([d["r_hat"] for d in diagnostics.values()])
        ),
        "ess_bulk_min": float(
            onp.min([d["ess_bulk"] for d in diagnostics.values()])
        ),
        "divergence_fraction": float(onp.mean(stats["diverging"])),
    }


def _sampler_report(settings, parts):
    """The sampler block of a posterior report: the settings and the worst
    ``_chain_health`` of ``parts`` (one per model, band or run)."""
    parts = list(parts)
    return {
        "num_chains": settings["num_chains"],
        "num_samples": settings["num_samples"],
        "num_warmup": settings["num_warmup"],
        "r_hat_max": float(onp.max([m["r_hat_max"] for m in parts])),
        "ess_bulk_min": float(onp.min([m["ess_bulk_min"] for m in parts])),
        "divergence_fraction": float(
            onp.max([m["divergence_fraction"] for m in parts])
        ),
    }


def _site_summary(x, diagnostics=None, *, std=True, **angle):
    """Median and 1σ interval (``angle=True``, ``period=`` for an angle),
    the standard deviation unless ``std=False``, then the diagnostics."""
    x = onp.asarray(x, dtype=float).ravel()
    out = _quantiles(x, **angle)
    if std:
        out["std"] = float(onp.std(x))
    return {**out, **(diagnostics or {})}


def _prior_bounds(priors, samples):
    """Fraction of samples at a prior bound, for the priors that have edges."""
    fractions = {k: _bound_samples(priors[k], samples[k]) for k in priors}
    return {k: f for k, f in fractions.items() if f is not None}


def _load_start(folder, priors):
    """The saved model's values for the sampled ``priors``: a start per site."""
    return {
        k: onp.asarray(v)
        for k, v in _io.load_model_values(folder).items()
        if k in priors
    }


def _write_samples(p, groups, **attrs):
    """Write ``samples.h5``: ``groups`` with the schema, ``attrs`` and the
    sampler settings as root attributes."""
    sampler = {k: p.settings[k] for k in _SAMPLER_SETTINGS}
    _io.write_h5(
        p.output / "samples.h5",
        groups,
        attrs={"/": {"schema": _io.SCHEMA, **attrs, "sampler": sampler}},
    )


def _corner_trace(output, samples, columns):
    """Save ``plots/posterior_corner.png`` and ``posterior_trace.png`` for
    the ``columns`` of ``samples`` (each ``(chain, draw)``)."""
    import pandas as pd

    from ..plotting import plot_chainconsumer_diagnostics

    flat = {k: onp.asarray(samples[k], dtype=float).ravel() for k in columns}
    try:
        _, corner, walks = plot_chainconsumer_diagnostics(
            {"posterior": pd.DataFrame(flat)}, columns=list(columns)
        )
    except ValueError as err:
        # Chains that never moved have zero-width draws, which chainconsumer
        # cannot lay out; the R-hat and ESS checks report the failure.
        import warnings

        import matplotlib.pyplot as plt

        warnings.warn(
            f"posterior figures not drawn ({err}); the draws may not vary",
            RuntimeWarning,
            stacklevel=2,
        )
        corner, walks = (plt.figure(figsize=(4, 1)) for _ in range(2))
        for fig in (corner, walks):
            fig.text(0.5, 0.5, "draws do not vary", ha="center", va="center")
    _save(corner, output / "plots" / "posterior_corner.png")
    _save(walks, output / "plots" / "posterior_trace.png")


def _sampler_checks(post, *, bounds=None, walls=None):
    """The four sampler checks, in order: prior bounds, R-hat, ESS and
    divergences. ``bounds`` replaces ``post["prior_bound_fraction"]`` and
    ``walls`` is the wall probe of the worst model."""
    return [
        _checks.prior_bound(
            post["prior_bound_fraction"] if bounds is None else bounds
        ),
        _checks.r_hat(post["r_hat_max"], ess=post["ess_bulk_min"]),
        _checks.ess(post["ess_bulk_min"]),
        _checks.divergences(post["divergence_fraction"], walls=walls),
    ]


def _save_fit(folder, fit, params):
    """Save a ``FitResult`` to ``folder``: the model, its values for
    ``params`` and ``info.json``."""
    _io.save_model(folder, fit.model, fit.values, params)
    _io.write_json(folder / "info.json", fit.info)


def _save_map_correlation(p, model, values, params):
    """Save ``plots/fit_correlation.png``: ``model`` at the MAP ``values``
    of ``params`` against ``p.processed``."""
    from ..likelihood import posterior_predictive_summary
    from ..plotting import plot_data_model_correlation

    one = {k: onp.asarray([float(onp.asarray(values[k]))]) for k in params}
    pred = posterior_predictive_summary(one, model, p.processed, params)
    fig, _ = plot_data_model_correlation(p.processed, {"MAP fit": pred})
    _save(fig, p.output / "plots" / "fit_correlation.png")


def _pick(report, keys, *, strict=True):
    """The ``keys`` of ``report``; with ``strict=False`` those it has."""
    return {k: report[k] for k in keys if strict or k in report}


def _prior_repr(prior):
    """``prior`` as ``"LogUniform(low=1e-05, high=0.5)"``: its class and
    parameters, without the memory address of numpyro's ``repr``."""
    args = ", ".join(
        f"{k}={onp.asarray(getattr(prior, k)).tolist()}"
        for k in getattr(prior, "arg_constraints", ())
        if hasattr(prior, k)
    )
    return f"{type(prior).__name__}({args})"


# === SETTINGS VALIDATION ===
# Shared checks of the settings dict ``s``; the messages are the contract of
# the tests, so a pipeline keeps its own for anything it checks itself.


def _positive(s, name):
    if isinstance(s[name], bool) or not (
        isinstance(s[name], numbers.Real) and s[name] > 0
    ):
        raise ValueError(f"{name} must be positive; got {s[name]!r}.")


def _integer(s, name, minimum):
    if isinstance(s[name], bool) or not (
        isinstance(s[name], int) and s[name] >= minimum
    ):
        raise ValueError(f"{name} must be an integer >= {minimum}.")


def _pair(s, name, low=0):
    """``name`` is ``[min, max]`` with ``low < min < max``."""
    pair = s[name]
    if len(pair) != 2 or not low < pair[0] < pair[1]:
        raise ValueError(f"{name} must satisfy {low} < min < max.")


def _validate_sampler(s):
    """The NUTS settings every pipeline has."""
    _integer(s, "num_samples", 1)
    _integer(s, "num_chains", 1)
    _integer(s, "num_warmup", 0)
    if s["chain_method"] not in ("vectorized", "sequential", "parallel"):
        raise ValueError(
            'chain_method must be "vectorized", "sequential" or "parallel".'
        )


def _validate_data_settings(s):
    """The data selection settings of the pipelines that read OIFITS data."""
    if s["wavel_range"] is not None and len(s["wavel_range"]) != 2:
        raise ValueError("wavel_range must be [min, max] in metres.")
    if s["error_floor"] is not None and not isinstance(s["error_floor"], dict):
        raise ValueError("error_floor must be a dict of absolute floors.")


def _validate_common(s):
    """The sampler and data settings of the pipelines that read OIFITS data."""
    _validate_sampler(s)
    _validate_data_settings(s)
    if s["batch_size"] is not None:
        _positive(s, "batch_size")
