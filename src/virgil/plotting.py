"""Plotting functions for virgil models, data and grid results.

* [`plot_grid_map`][virgil.plotting.plot_grid_map] draws a grid result and
  [`plot_contrast_curve`][virgil.plotting.plot_contrast_curve] a radial
  contrast curve.
* [`plot_null_distribution`][virgil.plotting.plot_null_distribution],
  [`plot_roc`][virgil.plotting.plot_roc] and
  [`plot_completeness`][virgil.plotting.plot_completeness] draw a
  [`DetectionMC`][virgil.detection.DetectionMC]; see
  [detection limits](../detection_limits.md#detection-plots).
* [`plot_field_draws`][virgil.plotting.plot_field_draws] shows draws from a
  Gaussian-process image prior or posterior, with cuts through them.

``units="flux" | "contrast" | "delta_mag"`` shows contrasts as companion/primary
flux, primary/companion, or magnitudes. Sky images have East (positive ``dra``)
to the left and North up. Importing this module leaves matplotlib's global
settings alone; [`set_style`][virgil.plotting.set_style] opts in to the virgil look.
"""

import contextlib
import functools

import matplotlib
import matplotlib.pyplot as plt
import numpy as onp
from matplotlib.ticker import FuncFormatter

from ._deprecate import renamed
from ._geometry import separation_pa
from ._utils import is_flux_param, resolve_flux_param
from .fields import GaussianField, MultipoleField
from .models import MultipoleDisk, RadialGP
from .limits import flux_to_contrast, flux_to_delta_mag, radial_profile

__provisional__ = ("plot_field_draws",)


def _require_plots_extra(module):
    """Import an optional plotting dependency, or explain how to get it."""
    import importlib

    try:
        return importlib.import_module(module)
    except ImportError as err:
        raise ImportError(
            f"This function needs {module.split('.')[0]}, which is not "
            "installed. Install it with: pip install 'virgil-astro[plots]'"
        ) from err


STYLE = {
    "figure.dpi": 100,
    "font.family": ["serif"],
    "font.size": 14,
}


def set_style():
    """Apply the virgil matplotlib style globally.

    The plotting functions in this module already use this style for the
    figures they create, without touching global settings. Call this to use
    it for your own figures too.
    """
    matplotlib.rcParams.update(STYLE)


# deliberate: used by notebooks/radio_clean.ipynb. Restores only its own
# keys: ``plt.rc_context`` would restore ``backend`` and close Jupyter's figures.
@contextlib.contextmanager
def _style_context(style=None):
    """Temporarily apply ``style`` (default: the virgil style)."""
    style = STYLE if style is None else style
    saved = {key: matplotlib.rcParams[key] for key in style}
    matplotlib.rcParams.update(style)
    try:
        yield
    finally:
        matplotlib.rcParams.update(saved)


def _styled(fn):
    """Run ``fn`` with the virgil style applied."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with _style_context():
            return fn(*args, **kwargs)

    return wrapper


def _is_sky_key(key, name):
    return key == name or key.endswith("." + name)


def _range_aware_float_formatter(vmin, vmax, scale=1.0):
    """Tick formatter whose significant figures (3 to 6) adapt to the span."""
    span = abs(float(vmax) - float(vmin)) * abs(float(scale))
    if not onp.isfinite(span) or span <= 0:
        sigfigs = 4
    else:
        sigfigs = int(onp.clip(onp.ceil(-onp.log10(span)) + 2, 3, 6))
    return FuncFormatter(lambda x, _: f"{(x * scale):.{sigfigs}g}")


def _enforce_sky_orientation(ax):
    """Make x increase to the left (East) and y upward (North)."""
    if ax.get_xlim()[0] < ax.get_xlim()[1]:
        ax.invert_xaxis()
    if ax.get_ylim()[0] > ax.get_ylim()[1]:
        ax.invert_yaxis()


@_styled
def plot_data_model_correlation(
    oidata,
    predictions_by_label,
    colors=None,
    figsize=(10, 5),
    phase_title="Phase correlation",
    square_axes=True,
    vis_label=None,
):
    """
    Plot data-vs-model correlation panels for visibility and phase observables.

    Parameters
    ----------
    oidata : OIData
        Observed data container.
    predictions_by_label : dict
        Mapping ``label -> prediction summary`` where each summary contains
        ``vis_mean``, ``vis_std``, ``phi_mean``, ``phi_std`` arrays.
    colors : list, optional
        Matplotlib color list. If omitted, cycle ``C0``, ``C1``, ...
    figsize : tuple, optional
        Figure size.
    phase_title : str, optional
        Title for the phase panel.
    square_axes : bool, optional
        If ``True``, enforce square panel boxes for both subplots.
    vis_label : str, optional
        Axis label for the visibility observable. By default it follows
        ``oidata.vis_mode``: V² and amplitudes are shown in percent,
        log-amplitudes and projected (e.g. DISCO) observables as plain
        values.

    Returns
    -------
    tuple
        ``(fig, (ax1, ax2))`` for visibility and phase axes. Data with no
        phase observables (e.g. AMIGO DISCOs, which hold every observable
        in ``vis``) leave the phase axis hidden and give the visibility
        panel the whole figure.
    """
    vis_mode = getattr(oidata, "vis_mode", "v2")
    projected = getattr(oidata, "vis_mat", None) is not None
    if projected or vis_mode == "logamp":
        vis_scale = 1.0
        default_label = "projected" if projected else "log amplitude"
    else:
        vis_scale = 100.0
        default_label = "V2, %" if vis_mode == "v2" else "amplitude, %"
    if vis_label is None:
        vis_label = default_label
    vis_data = onp.asarray(oidata.vis).reshape(-1)
    phi_data = onp.asarray(oidata.phi).reshape(-1)
    d_vis_data = onp.asarray(oidata.d_vis).reshape(-1)
    d_phi_data = onp.asarray(oidata.d_phi).reshape(-1)

    if colors is None:
        colors = [f"C{i}" for i in range(max(1, len(predictions_by_label)))]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    # Projected (e.g. AMIGO DISCO) data may hold every observable in vis.
    has_phase = phi_data.size > 0

    vis_low = [vis_data.min()]
    vis_high = [vis_data.max()]
    phi_low = [phi_data.min()] if has_phase else []
    phi_high = [phi_data.max()] if has_phase else []

    for idx, (label, pred) in enumerate(predictions_by_label.items()):
        color = colors[idx % len(colors)]
        ax1.errorbar(
            vis_data,
            onp.asarray(pred["vis_mean"]).reshape(-1),
            xerr=d_vis_data,
            yerr=onp.asarray(pred["vis_std"]).reshape(-1),
            fmt="o",
            markersize=4,
            alpha=0.45,
            elinewidth=1.0,
            capsize=2,
            color=color,
            label=label,
        )
        if not has_phase:
            vis_low.append(onp.asarray(pred["vis_mean"]).min())
            vis_high.append(onp.asarray(pred["vis_mean"]).max())
            continue
        ax2.errorbar(
            phi_data,
            onp.asarray(pred["phi_mean"]).reshape(-1),
            xerr=d_phi_data,
            yerr=onp.asarray(pred["phi_std"]).reshape(-1),
            fmt="o",
            markersize=4,
            alpha=0.45,
            elinewidth=1.0,
            capsize=2,
            color=color,
            label=label,
        )
        vis_low.append(onp.asarray(pred["vis_mean"]).min())
        vis_high.append(onp.asarray(pred["vis_mean"]).max())
        phi_low.append(onp.asarray(pred["phi_mean"]).min())
        phi_high.append(onp.asarray(pred["phi_mean"]).max())

    vis_pad = float(onp.median(d_vis_data)) if d_vis_data.size else 0.0
    phi_pad = float(onp.median(d_phi_data)) if d_phi_data.size else 0.0

    vis_min = min(vis_low) - vis_pad
    vis_max = max(vis_high) + vis_pad

    vis_line = onp.linspace(vis_min, vis_max, 200)
    vis_formatter = _range_aware_float_formatter(
        vis_min, vis_max, scale=vis_scale
    )

    ax1.plot(vis_line, vis_line, "k--", lw=1)
    ax1.set_xlim(vis_min, vis_max)
    ax1.set_ylim(vis_min, vis_max)
    ax1.xaxis.set_major_formatter(vis_formatter)
    ax1.yaxis.set_major_formatter(vis_formatter)
    ax1.set_xlabel(f"Data ({vis_label})")
    ax1.set_ylabel(f"Model ({vis_label})")
    ax1.set_title("Visibility correlation")
    if square_axes:
        ax1.set_box_aspect(1)
    ax1.legend(loc="best")

    if has_phase:
        phi_min = min(phi_low) - phi_pad
        phi_max = max(phi_high) + phi_pad
        phi_line = onp.linspace(phi_min, phi_max, 200)
        phi_formatter = _range_aware_float_formatter(phi_min, phi_max)
        ax2.plot(phi_line, phi_line, "k--", lw=1)
        ax2.set_xlim(phi_min, phi_max)
        ax2.set_ylim(phi_min, phi_max)
        ax2.xaxis.set_major_formatter(phi_formatter)
        ax2.yaxis.set_major_formatter(phi_formatter)
        ax2.set_xlabel("Data (rad)")
        ax2.set_ylabel("Model (rad)")
        ax2.set_title(phase_title)
        if square_axes:
            ax2.set_box_aspect(1)
        ax2.legend(loc="best")
    else:  # one panel, taking the whole figure
        ax2.set_visible(False)
        ax1.set_subplotspec(fig.add_gridspec(1, 1)[0])

    fig.tight_layout()
    return fig, (ax1, ax2)


@_styled
def plot_model(
    model,
    fov_mas,
    npix=256,
    ax=None,
    title=None,
    saturate=None,
    cmap="magma",
    beam=None,
    convolve=False,
):
    """Show a source model's rendered image with sky axes (East left, North up).

    Parameters
    ----------
    model : SourceModel
        Model to render.
    fov_mas : float
        Width of the field of view in milliarcseconds.
    npix : int, optional
        Number of pixels on a side.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on; a new figure is made if omitted.
    title : str, optional
        Axes title.
    saturate : float, optional
        Quantile (e.g. ``0.99``) at which to saturate the colour scale.
    cmap : str, optional
        Matplotlib colour map.
    beam : Beam, optional
        The data's resolution, from [`imaging.beam`][virgil.imaging.beam],
        drawn as a shaded FWHM ellipse in the lower-left corner.
    convolve : bool, optional
        Show the image convolved with ``beam`` (see
        [`imaging.convolve_beam`][virgil.imaging.convolve_beam]).

    Returns
    -------
    matplotlib.axes.Axes
        The axes drawn on.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))
    image = model.render(npix=npix, fov_mas=fov_mas)
    if convolve:
        if beam is None:
            raise ValueError("convolve=True needs a beam.")
        from .imaging import convolve_beam

        image = convolve_beam(image, float(fov_mas) / npix, beam)
    image = onp.asarray(image)
    vmax = None if saturate is None else onp.quantile(image, saturate)
    half = float(fov_mas) / 2.0
    ax.imshow(
        image,
        extent=[half, -half, -half, half],
        origin="upper",
        cmap=cmap,
        vmax=vmax,
    )
    ax.set(xlabel="ΔRA (mas)", ylabel="ΔDec (mas)")
    if title is not None:
        ax.set_title(title)
    _enforce_sky_orientation(ax)
    if beam is not None:
        _draw_beam(ax, beam)
    return ax


def _draw_beam(ax, beam):
    """Draw a beam's FWHM ellipse, shaded, in the lower-left corner."""
    from matplotlib.patches import Ellipse

    (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()  # (left, right), ...
    inset = max(0.1, 0.6 * beam.major_mas / abs(x1 - x0))
    centre = (x0 + inset * (x1 - x0), y0 + inset * (y1 - y0))
    # In (ΔRA, ΔDec) the major axis points along (sin PA, cos PA).
    angle = 90.0 - beam.pa_deg
    ax.add_patch(
        Ellipse(
            centre,
            beam.major_mas,
            beam.minor_mas,
            angle=angle,
            facecolor=(1.0, 1.0, 1.0, 0.45),
            edgecolor="white",
            lw=1.0,
        )
    )


@_styled
def plot_residual_map(
    residual,
    fov_mas,
    sigma=None,
    ax=None,
    title=None,
    cmap="RdBu_r",
):
    """Show a signed residual image on a symmetric, diverging colour scale.

    Use it next to an image and its reconstruction (or data and a model)
    to see where they agree. Without ``sigma`` the map is the plain signed
    residual, as for a maximum a posteriori image with no uncertainty.
    With ``sigma`` it is the z-score ``residual / sigma``. The colour scale
    is centred on zero in either case, with East to the left and North up.

    Parameters
    ----------
    residual : array-like, shape (npix, npix)
        E.g. ``reconstruction.render(npix, fov) - truth.render(npix, fov)``,
        in the orientation of [`render`][virgil.models.SourceModel.render].
    fov_mas : float
        Width of the field of view in milliarcseconds.
    sigma : array-like or float, optional
        Uncertainty of the residual, per pixel or overall; if given, the
        map shows z-scores.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on; a new figure is made if omitted.
    title : str, optional
        Axes title.
    cmap : str, optional
        A diverging Matplotlib colour map.

    Returns
    -------
    matplotlib.axes.Axes
        The axes drawn on (with a colour bar).
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))
    values = onp.asarray(residual, dtype=float)
    label = "residual"
    if sigma is not None:
        values = values / onp.asarray(sigma, dtype=float)
        label = "z-score"
    limit = float(onp.nanmax(onp.abs(values))) or 1.0
    half = float(fov_mas) / 2.0
    mappable = ax.imshow(
        values,
        extent=[half, -half, -half, half],
        origin="upper",
        cmap=cmap,
        vmin=-limit,
        vmax=limit,
    )
    ax.figure.colorbar(mappable, ax=ax, label=label)
    ax.set(xlabel="ΔRA (mas)", ylabel="ΔDec (mas)")
    if title is not None:
        ax.set_title(title)
    _enforce_sky_orientation(ax)
    return ax


@_styled
def plot_chainconsumer_diagnostics(
    chains_by_label,
    columns,
    truth=None,
    colors=None,
    walk_columns=None,
):
    """
    Plot comparison diagnostics using ChainConsumer for multiple posterior chains.

    Parameters
    ----------
    chains_by_label : dict
        Mapping ``label -> pandas.DataFrame`` containing chain samples.
    columns : list[str]
        Columns used for contour/corner plotting.
    truth : dict, optional
        Truth mapping for columns displayed. When omitted, no truth overlay is
        drawn.
    colors : list[str], optional
        Per-chain colors.
    walk_columns : list[str], optional
        Columns to show in walk plots. Defaults to ``columns``.

    Returns
    -------
    tuple
        ``(consumer, corner_fig, walks_fig)``: the configured ChainConsumer
        instance and the two figures it drew.
    """
    _require_plots_extra("chainconsumer")
    from chainconsumer import ChainConsumer, Chain, Truth
    from chainconsumer.statistics import SummaryStatistic

    if colors is None:
        colors = matplotlib.colormaps["tab10"].colors
        colors = [matplotlib.colors.to_hex(color) for color in colors]
    if walk_columns is None:
        walk_columns = columns

    consumer = ChainConsumer()
    for idx, (label, samples) in enumerate(chains_by_label.items()):
        consumer.add_chain(
            Chain(
                samples=samples[columns],
                name=label,
                color=colors[idx % len(colors)],
                plot_point=False,
                plot_cloud=False,
                # Median and quantiles. ChainConsumer's default ("max")
                # searches a smoothed histogram for an iso-density interval
                # and, on a bumpy histogram, can fail and report a
                # well-sampled parameter as "not constrained".
                statistics=SummaryStatistic.CUMULATIVE,
            )
        )
    if truth is not None:
        consumer.add_truth(Truth(location=truth))
    corner_fig = consumer.plotter.plot()
    walks_fig = consumer.plotter.plot_walks(
        columns=walk_columns, plot_weights=False, plot_posterior=False
    )
    return consumer, corner_fig, walks_fig


def diagnostics_table_from_samples(
    samples,
    dra_key="dra",
    ddec_key="ddec",
    flux_key="flux",
    log10_flux=False,
):
    """
    Build a standardized diagnostics table from posterior sample arrays.

    Parameters
    ----------
    samples : dict-like
        Mapping of sample arrays.
    dra_key : str, optional
        Key for right-ascension offsets.
    ddec_key : str, optional
        Key for declination offsets.
    flux_key : str, optional
        Key for flux or log10-flux samples.
    log10_flux : bool, optional
        If ``True``, exponentiate ``flux_key`` values as base-10.

    Returns
    -------
    pandas.DataFrame
        Table with ``dra``, ``ddec``, ``flux``, ``sep``, and ``pa`` columns.
    """
    pd = _require_plots_extra("pandas")
    dra = onp.asarray(samples[dra_key], dtype=float)
    ddec = onp.asarray(samples[ddec_key], dtype=float)
    flux_raw = onp.asarray(samples[flux_key], dtype=float)
    flux = onp.power(10.0, flux_raw) if log10_flux else flux_raw

    df = pd.DataFrame({"dra": dra, "ddec": ddec, "flux": flux})
    df["sep"], df["pa"] = separation_pa(dra, ddec)
    return df


def truth_cartesian_and_polar(truth):
    """
    Return truth mappings for shared ChainConsumer Cartesian and polar interfaces.

    Parameters
    ----------
    truth : dict
        Mapping with ``dra``, ``ddec``, and ``flux`` values.

    Returns
    -------
    tuple[dict, dict]
        Cartesian and polar truth dictionaries.
    """
    truth_cart = {
        "dra": float(truth["dra"]),
        "ddec": float(truth["ddec"]),
        "flux": float(truth["flux"]),
    }
    sep, pa = separation_pa(truth_cart["dra"], truth_cart["ddec"])
    truth_polar = {
        "sep": float(sep),
        "pa": float(pa),
        "flux": truth_cart["flux"],
    }
    return truth_cart, truth_polar


def _format_sigma_or_percent_value(value):
    """Format a confidence level compactly, e.g. ``97.7`` or ``5``."""
    formatted = f"{float(value):.3g}"
    return formatted.rstrip("0").rstrip(".") if "." in formatted else formatted


def _resolve_limit_label(label=None, percentile=None, sigma=None):
    """Label for a contrast limit at a given confidence, or ``None``."""
    if label is not None:
        return label
    if percentile is not None and sigma is not None:
        raise ValueError("Provide only one of percentile or sigma.")
    if percentile is not None:
        value = onp.asarray(percentile, dtype=float).reshape(-1)
        if value.size != 1:
            raise ValueError("percentile must be a scalar or length-1 array.")
        return f"{_format_sigma_or_percent_value(value[0] * 100)}% upper limit"
    if sigma is not None:
        value = onp.asarray(sigma, dtype=float).reshape(-1)
        if value.size != 1:
            raise ValueError("sigma must be a scalar or length-1 array.")
        return f"{_format_sigma_or_percent_value(value[0])}$\\sigma$ limit"
    return None


# === GRID MAPS ===

# Defaults for each kind of grid result. "flux_like" kinds hold
# companion/primary flux ratios and can be shown in other units.
_KINDS = {
    "loglike": dict(
        cmap="inferno", log=False, label="Log likelihood", flux_like=False
    ),
    "flux": dict(
        cmap="inferno", log=True, label="Best-fit flux", flux_like=True
    ),
    "sigma": dict(cmap="viridis", log=True, label="σ(flux)", flux_like=False),
    "snr": dict(cmap="inferno", log=False, label="S/N", flux_like=False),
    "limit": dict(
        cmap="magma", log=True, label="Contrast limit", flux_like=True
    ),
}

_UNIT_LABELS = {
    "flux": "flux ratio",
    "contrast": "contrast",
    "delta_mag": "Δmag",
}


def _convert_flux(values, units):
    """Companion/primary flux ratios in the requested display units."""
    if units == "flux":
        return onp.asarray(values, dtype=float)
    if units == "contrast":
        return onp.asarray(flux_to_contrast(values))
    if units == "delta_mag":
        return onp.asarray(flux_to_delta_mag(values))
    raise ValueError(
        f"units must be 'flux', 'contrast' or 'delta_mag'; got {units!r}."
    )


# deliberate: reduces a full log-likelihood grid over its flux axis to the plotted array and axis keys, returning early for grids that need no reduction
def _grid_axes(values, grid, kind, flux_param):
    """``(values, coord_keys)``; a full log-likelihood grid is reduced to its
    maximum over the flux axis."""
    values = onp.asarray(values, dtype=float)
    keys = list(grid)
    if values.ndim == len(keys):
        try:
            flux_key = resolve_flux_param(keys, flux_param)
        except ValueError:
            return values, keys
        if kind != "loglike":
            raise ValueError(
                f"values has an axis for {flux_key!r}; only kind='loglike' "
                "grids are reduced over the flux automatically."
            )
        values = onp.nanmax(values, axis=keys.index(flux_key))
        return values, [key for key in keys if key != flux_key]
    if values.ndim == len(keys) - 1:
        flux_key = resolve_flux_param(keys, flux_param)
        return values, [key for key in keys if key != flux_key]
    raise ValueError(
        f"values has {values.ndim} axes but grid has keys {keys}; "
        "expected one axis per key, or per key except the flux."
    )


def _coord_value(values, key, coord_keys):
    """Look up ``key`` in a dict, or by position in ``coord_keys`` order."""
    if isinstance(values, dict):
        return float(values[key])
    return float(values[list(coord_keys).index(key)])


@renamed()
@_styled
def plot_grid_map(
    values,
    grid,
    kind="loglike",
    *,
    units="flux",
    sigma=None,
    percentile=None,
    truth=None,
    best=None,
    star=True,
    flux_param=None,
    ax=None,
    cmap=None,
    log=None,
    label=None,
    title=None,
    figsize=(7, 6),
):
    """Plot a grid result as a sky map (or a 1D profile).

    Parameters
    ----------
    values : array-like
        Grid result with one axis per coordinate key of ``grid`` (every key but
        the flux), from [`virgil.grid_fit`][virgil.grid_fit] or
        [`virgil.limits`][virgil.limits]; a full ``likelihood_grid`` is reduced
        to its maximum over flux. One coordinate gives a line plot.
    grid : dict[str, array-like]
        The grid axes used to compute ``values``.
    kind : {"loglike", "flux", "sigma", "snr", "limit"}, optional
        What ``values`` holds; sets the default colour map, scale and labels.
    units : {"flux", "contrast", "delta_mag"}, optional
        For ``kind="flux"`` and ``"limit"``: flux ratios, contrasts or
        magnitude differences.
    sigma, percentile : float, optional
        Confidence of a ``kind="limit"`` map, for its title.
    truth, best : dict or sequence, optional
        Coordinates to mark with a cross (truth) or circle (best fit).
    star : bool, optional
        Mark the primary at the origin of sky maps (default True).
    flux_param : str, optional
        Key of the flux axis, if it is not the one key ending in ``flux``.
    ax, figsize : optional
        Axes to draw on (default: a new figure of this size).
    cmap, log, label, title : optional
        Override the defaults for ``kind``.

    Returns
    -------
    tuple
        ``(fig, ax)``; ``dra`` is x and ``ddec`` y whatever their order in ``grid``.

    Examples
    --------
    >>> plot_grid_map(limits, grid, kind="limit", units="delta_mag", sigma=5)  # doctest: +SKIP
    """
    if kind not in _KINDS:
        raise ValueError(
            f"kind must be one of {sorted(_KINDS)}; got {kind!r}."
        )
    defaults = _KINDS[kind]
    values, coord_keys = _grid_axes(values, grid, kind, flux_param)

    if defaults["flux_like"]:
        values = _convert_flux(values, units)
        default_label = f"{defaults['label']} ({_UNIT_LABELS[units]})"
        default_log = units != "delta_mag"
        # Contrast and Δmag grow as the companion gets fainter; reverse the
        # colour map so that deeper limits keep the same colour.
        default_cmap = defaults["cmap"]
        if units != "flux":
            # maps whose name already ends in ``_r`` are deliberately reversed
            if isinstance(default_cmap, str):
                default_cmap = matplotlib.colormaps[default_cmap]
            if not default_cmap.name.endswith("_r"):
                default_cmap = default_cmap.reversed()
    else:
        default_label = defaults["label"]
        default_log = defaults["log"]
        default_cmap = defaults["cmap"]
    limit_label = _resolve_limit_label(None, percentile, sigma)
    if kind == "limit" and limit_label is not None:
        default_title = f"{limit_label} ({_UNIT_LABELS[units]})"
    else:
        default_title = default_label
    label = default_label if label is None else label
    title = default_title if title is None else title
    log = default_log if log is None else log
    cmap = default_cmap if cmap is None else cmap

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    if values.ndim == 1:
        x_key = coord_keys[0]
        ax.plot(onp.asarray(grid[x_key]), values, color="C0", lw=2)
        if truth is not None:
            ax.axvline(
                _coord_value(truth, x_key, coord_keys),
                color="k",
                lw=1.5,
                ls="--",
                label="Truth",
            )
        if best is not None:
            ax.axvline(
                _coord_value(best, x_key, coord_keys),
                color="C1",
                lw=1.2,
                ls=":",
                label="Best fit",
            )
        if log:
            ax.set_yscale("log")
        ax.set(xlabel=x_key, ylabel=label, title=title)
        if truth is not None or best is not None:
            ax.legend(loc="best")
        return fig, ax

    grid_values = onp.asarray(values)
    if grid_values.ndim != 2 or len(coord_keys) != 2:
        raise ValueError(
            f"Expected a 2D grid with two coordinate keys; got shape "
            f"{grid_values.shape} for keys {list(coord_keys)}. Reduce the "
            "flux axis first, e.g. loglike_im.max(axis=...)."
        )
    dra = [k for k in coord_keys if _is_sky_key(k, "dra")]
    ddec = [k for k in coord_keys if _is_sky_key(k, "ddec")]
    is_sky = len(dra) == 1 and len(ddec) == 1
    x_key, y_key = (dra[0], ddec[0]) if is_sky else coord_keys
    # ``image[y, x]``: transpose when the grid's first axis is x
    image = (
        grid_values.T if list(coord_keys).index(x_key) == 0 else grid_values
    )
    finite = image[onp.isfinite(image)]
    norm = None
    if log and finite.size and onp.all(finite > 0):
        norm = matplotlib.colors.LogNorm()
    # ``imshow`` places ``extent`` at the outer pixel edges: pad the sample
    # positions (pixel centres) by half a spacing
    extent = []
    for axis in (grid[x_key], grid[y_key]):
        axis = onp.asarray(axis, dtype=float).reshape(-1)
        half = 0.5 * (axis[1] - axis[0]) if axis.size > 1 else 0.5
        extent += [float(axis[0] - half), float(axis[-1] + half)]
    im = ax.imshow(
        image,
        origin="lower",
        extent=extent,
        cmap=cmap,
        norm=norm,
        aspect="equal" if is_sky else "auto",
    )
    fig.colorbar(im, ax=ax, label=label, pad=0.01)
    if is_sky and star:
        ax.scatter(0, 0, s=140, c="white", edgecolors="k", marker="*")
    if truth is not None:
        ax.scatter(
            [_coord_value(truth, x_key, coord_keys)],
            [_coord_value(truth, y_key, coord_keys)],
            marker="x",
            s=80,
            c="white",
            linewidths=2,
            label="Truth",
        )
    if best is not None:
        ax.scatter(
            [_coord_value(best, x_key, coord_keys)],
            [_coord_value(best, y_key, coord_keys)],
            marker="o",
            s=40,
            facecolors="none",
            edgecolors="cyan",
            label="Best fit",
        )
    if truth is not None or best is not None:
        ax.legend(loc="upper right")
    if is_sky:
        ax.set(xlabel="ΔRA (mas)", ylabel="ΔDec (mas)")
        _enforce_sky_orientation(ax)
    else:
        ax.set(xlabel=x_key, ylabel=y_key)
    ax.set_title(title)
    return fig, ax


@renamed()
@_styled
def plot_contrast_curve(
    values,
    grid=None,
    *,
    units="delta_mag",
    sigma=None,
    percentile=None,
    label=None,
    band=True,
    truth=None,
    center=(0.0, 0.0),
    r_max=None,
    bins=20,
    ax=None,
    color=None,
    figsize=(8, 4),
):
    """Plot a radial contrast curve: the median limit against separation.

    Parameters
    ----------
    values : array-like or dict
        A limit map (companion/primary flux ratios, one axis per coordinate key
        of ``grid``, as from [`absil_limits`][virgil.limits.absil_limits] or
        [`ruffio_upperlimit`][virgil.limits.ruffio_upperlimit]), or a
        [`radial_profile`][virgil.limits.radial_profile] of one.
    grid : dict[str, array-like], optional
        The grid axes of a limit map; must contain ``dra`` and ``ddec``.
    units : {"delta_mag", "contrast", "flux"}, optional
        Magnitude differences (default), contrasts or flux ratios. Deeper
        limits are always drawn lower down.
    sigma, percentile, label : optional
        Legend label; by default built from ``sigma`` or ``percentile``.
    band : bool, optional
        Shade the 16-84 percentile range of each annulus (default True).
    truth : tuple, optional
        ``(dra, ddec, flux)`` of a detected companion to mark.
    center, r_max, bins : optional
        Annuli, passed to [`radial_profile`][virgil.limits.radial_profile].
    ax, figsize : optional
        Axes to draw on, e.g. to overlay several curves (default: a new figure
        of this size).
    color : optional
        Line and band colour.

    Returns
    -------
    tuple
        ``(fig, ax)``.
    """
    if isinstance(values, dict):
        profile = values
    else:
        if grid is None:
            raise ValueError("Pass grid with a limit map.")
        # a sky map with axes (dra, ddec), transposed when ddec comes first
        keys = [key for key in grid if not is_flux_param(key)]
        found = []
        for name in ("dra", "ddec"):
            matches = [key for key in keys if _is_sky_key(key, name)]
            if len(matches) != 1:
                raise ValueError(
                    f"grid needs exactly one {name!r} axis; "
                    f"keys are {list(grid)}."
                )
            found.append(matches[0])
        if len(keys) != 2:
            raise ValueError(
                f"A sky map needs exactly two coordinate keys; got {keys}."
            )
        sky_values = onp.asarray(values, dtype=float)
        if keys.index(found[0]) == 1:
            sky_values = sky_values.T
        dra, ddec = (onp.asarray(grid[key]) for key in found)
        profile = radial_profile(
            sky_values, dra, ddec, center=center, r_max=r_max, bins=bins
        )
    r = onp.asarray(profile["r"])
    median = _convert_flux(profile["median"], units)
    low = _convert_flux(profile["q16"], units)
    high = _convert_flux(profile["q84"], units)
    label = _resolve_limit_label(label, percentile, sigma) or "Median limit"

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    (line,) = ax.plot(r, median, lw=2, color=color, label=label)
    if band:
        ax.fill_between(r, low, high, color=line.get_color(), alpha=0.25, lw=0)
    if truth is not None:
        true_dra, true_ddec, true_flux = truth
        ax.plot(
            onp.hypot(true_dra, true_ddec),
            _convert_flux(true_flux, units),
            marker="*",
            c="k",
            markersize=15,
            ls="none",
            label="Companion",
        )
    if units != "delta_mag":
        ax.set_yscale("log")
    # Contrast and Δmag grow downwards, so deeper limits sit lower.
    if units != "flux" and not ax.yaxis_inverted():
        ax.invert_yaxis()
    ax.set_xlabel("Separation (mas)")
    ax.set_ylabel(f"Contrast limit ({_UNIT_LABELS[units]})")
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    return fig, ax


# === DETECTION: NULL DISTRIBUTIONS, ROC CURVES AND COMPLETENESS ===

# Labels of the statistics of virgil.detection.
_STAT_LABELS = {
    "delta_chi2": r"$\Delta\chi^2$",
    "log_bayes_factor": r"$\log B$",
    "max_snr": "max SNR",
}
# Line styles that tell curves apart without colour.
_LINE_STYLES = ("-", "--", ":", "-.")


def _stat_label(stat):
    if stat not in _STAT_LABELS:
        raise ValueError(
            f"stat must be one of {sorted(_STAT_LABELS)}, not {stat!r}."
        )
    return _STAT_LABELS[stat]


def _local_threshold(stat, n_sigma):
    """The statistic at a one-position significance of n_sigma (None for the
    log Bayes factor): ``n_sigma**2`` for ``delta_chi2``, ``n_sigma`` for
    ``max_snr``."""
    if stat == "delta_chi2":
        return float(n_sigma) ** 2
    if stat == "max_snr":
        return float(n_sigma)
    return None


def _format_probability(p):
    """A probability as a short percentage, or as a power of ten if tiny."""
    p = float(p)
    if not onp.isfinite(p):
        return str(p)
    if p >= 1e-3:
        return f"{100.0 * p:.3g}%"
    mantissa, exponent = f"{p:.1e}".split("e")
    mantissa = mantissa.rstrip("0").rstrip(".")
    power = f"10^{{{int(exponent)}}}"
    return f"${power}$" if mantissa == "1" else f"${mantissa}\\times{power}$"


def _selections(value):
    """One curve per entry of a list (or array); otherwise one curve.

    A tuple is a range ``(lo, hi)``, as for ``DetectionMC.roc``, so several
    fluxes or bins must be given as a list.
    """
    if value is None:
        return [None]
    if isinstance(value, onp.ndarray) and value.ndim >= 1:
        return [tuple(v) if onp.ndim(v) else float(v) for v in value]
    if isinstance(value, list):
        return list(value)
    return [value]


def _selection_labels(flux, sep_bin):
    """Labels of a flux and separation selection (None is left out)."""
    labels = []
    if flux is not None and onp.ndim(flux) == 0:
        flux = float(flux)
        labels.append(
            "flux 0"
            if flux <= 0.0
            else f"flux {flux:.3g} (Δmag {float(flux_to_delta_mag(flux)):.2f})"
        )
    elif flux is not None:
        lo, hi = flux
        labels.append(f"flux {lo:.3g}–{hi:.3g}")
    if sep_bin is not None:
        lo, hi = sep_bin
        labels.append(f"{lo:g}–{hi:g} mas")
    return labels


def _legend_outside(ax):
    """A legend to the right of the axes, where it covers no data."""
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        fontsize="small",
        borderaxespad=0.0,
    )


@_styled
def plot_roc(
    mc,
    stat,
    *,
    flux=None,
    sep_bin=None,
    ax=None,
    log_fpr=True,
    mark_wilks=(3.0, 5.0),
    title=None,
    figsize=(7, 4.5),
):
    """Plot ROC curves: completeness against false-alarm probability.

    A useless statistic lies on TPR = FPR; see
    [detection limits](../detection_limits.md#detection-plots).

    Parameters
    ----------
    mc : DetectionMC
        Simulated searches, from
        [`injection_recovery`][virgil.detection.injection_recovery].
    stat : str or sequence of str
        ``"delta_chi2"``, ``"log_bayes_factor"`` or ``"max_snr"``, or several.
    flux : float, (float, float) or list, optional
        Only injections of this flux or in this range, as for
        [`DetectionMC.roc`][virgil.detection.DetectionMC.roc]; a list gives one
        curve per entry.
    sep_bin : (float, float) or list of them, optional
        Only injections with separation in ``[lo, hi)`` (mas); a list gives one
        curve per bin.
    log_fpr : bool, optional
        Logarithmic false-positive axis.
    mark_wilks : sequence of float, optional
        Local significances at which to mark the single-position false-alarm
        probability and its threshold on each curve (the gap is the
        look-elsewhere effect); ``()`` marks nothing.
    title, ax, figsize : optional
        Title (default: what the curves share), axes, and size of a new figure.

    Returns
    -------
    tuple
        ``(fig, ax)``.

    Examples
    --------
    >>> plot_roc(mc, ["delta_chi2", "max_snr"], flux=4e-3)  # doctest: +SKIP
    """
    stats = [stat] if isinstance(stat, str) else list(stat)
    if not stats:
        raise ValueError("Pass at least one statistic.")
    for s in stats:
        _stat_label(s)
    marks = () if mark_wilks is None else tuple(mark_wilks)
    selections = [
        (f, b) for f in _selections(flux) for b in _selections(sep_bin)
    ]
    n_null = mc.n_null
    if n_null == 0:
        raise ValueError("The DetectionMC has no null draws.")
    fpr_min = 0.5 / n_null

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    for j, (f, b) in enumerate(selections):
        for i, s in enumerate(stats):
            fpr, tpr, thresholds = mc.roc(s, flux=f, sep_bin=b)
            parts = [_STAT_LABELS[s]] if len(stats) > 1 else []
            if len(selections) > 1:
                parts += _selection_labels(f, b)
            color = f"C{j % 10}"
            style = _LINE_STYLES[(i if len(stats) > 1 else j) % 4]
            ax.plot(
                fpr,
                tpr,
                color=color,
                ls=style,
                lw=2,
                label=", ".join(parts) or _STAT_LABELS[s],
            )
            for k, n_sigma in enumerate(marks):
                value = _local_threshold(s, n_sigma)
                if value is None:
                    continue
                # The false- and true-positive rates at the threshold.
                i = (
                    int(onp.count_nonzero(onp.asarray(thresholds) >= value))
                    - 1
                )
                x, y = float(fpr[i]), float(tpr[i])
                if log_fpr and x <= 0.0:
                    continue  # no null draw reaches it: off the log axis
                ax.plot(
                    x,
                    y,
                    marker="os"[k % 2],
                    ms=8,
                    mfc="none",
                    mec=color,
                    mew=1.5,
                    ls="none",
                )

    chance = (
        onp.geomspace(fpr_min, 1.0, 200)
        if log_fpr
        else onp.linspace(0.0, 1.0, 200)
    )
    ax.plot(chance, chance, color="grey", lw=1, label="chance (TPR = FPR)")
    from scipy.stats import norm as _norm

    for k, n_sigma in enumerate(marks):
        # One-sided Gaussian tail at ``n_sigma``: 0.135% at 3σ.
        nominal = float(_norm.sf(float(n_sigma)))
        if log_fpr and nominal < fpr_min:
            continue
        ax.axvline(
            nominal,
            color="k",
            lw=0.8,
            ls=_LINE_STYLES[(k + 2) % 4],
            label=(
                f"local {n_sigma:g}σ: FAP {_format_probability(nominal)} "
                "at one position"
            ),
        )
        values = [
            f"{_STAT_LABELS[s]} = {_local_threshold(s, n_sigma):g}"
            for s in stats
            if _local_threshold(s, n_sigma) is not None
        ]
        if values:
            ax.plot(
                [],
                [],
                marker="os"[k % 2],
                mfc="none",
                mec="k",
                ls="none",
                label=(
                    f"threshold {', '.join(values)} "
                    f"(local {n_sigma:g}σ) over the grid"
                ),
            )

    if log_fpr:
        ax.set_xscale("log")
        ax.set_xlim(fpr_min, 1.0)
    else:
        ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("False-positive rate (false-alarm probability)")
    ax.set_ylabel("True-positive rate (completeness)")
    if title is None:
        # What all the curves share: the statistic, or the selection.
        pieces = ["ROC"]
        if len(stats) == 1:
            pieces = [f"ROC of {_STAT_LABELS[stats[0]]}"]
        if len(selections) == 1:
            pieces += _selection_labels(*selections[0])
        title = ", ".join(pieces)
    if title:
        ax.set_title(title)
    ax.grid(alpha=0.3)
    _legend_outside(ax)
    return fig, ax


def _cell_edges(centres, log):
    """Edges of cells centred on sorted ``centres`` (in log space if log)."""
    c = onp.asarray(centres, dtype=float)
    if log:
        c = onp.log(c)
    if c.size == 1:
        half = 0.2 if log else max(0.5, 0.1 * abs(c[0]))
        edges = onp.array([c[0] - half, c[0] + half])
    else:
        mid = 0.5 * (c[1:] + c[:-1])
        edges = onp.concatenate(
            [[c[0] - (mid[0] - c[0])], mid, [c[-1] + (c[-1] - mid[-1])]]
        )
    return onp.exp(edges) if log else edges


@_styled
def plot_completeness(
    mc,
    stat,
    fap,
    *,
    units="delta_mag",
    contours=(0.5, 0.9),
    sep_bins=None,
    flux_bins=None,
    ax=None,
    cmap="viridis",
    colorbar=True,
    title=None,
    figsize=(8, 4.5),
):
    """Plot a completeness map against separation and companion brightness.

    Each cell is the fraction of injections detected at false-alarm probability
    ``fap`` ([`DetectionMC.completeness`][virgil.detection.DetectionMC.completeness]);
    lines mark ``contours``. See
    [detection limits](../detection_limits.md#detection-plots).

    Parameters
    ----------
    mc : DetectionMC
        Simulated searches with injections, from
        [`injection_recovery`][virgil.detection.injection_recovery].
    stat : str
        The detection statistic, e.g. ``"delta_chi2"``.
    fap : float
        False-alarm probability of the threshold, e.g. 1.35e-3 (3σ).
    units : {"delta_mag", "contrast", "flux"}, optional
        The y-axis, as in
        [`plot_contrast_curve`][virgil.plotting.plot_contrast_curve]
        (``delta_mag`` is 2.5 log10(primary/companion)).
    contours : sequence of float, optional
        Completeness levels to draw as lines (default 50% and 90%).
    sep_bins, flux_bins : array-like, optional
        Bin edges (mas, and flux); by default every injected value is a cell.
    cmap : str or Colormap, optional
        Colour map of the completeness, from 0 to 1.
    colorbar : bool, optional
        Add a colour bar.
    title, ax, figsize : optional
        Title (default: statistic and FAP), axes, and size of a new figure.

    Returns
    -------
    tuple
        ``(fig, ax)``.

    Examples
    --------
    >>> plot_completeness(mc, "delta_chi2", 1.35e-3)  # doctest: +SKIP
    """
    import matplotlib.patheffects as path_effects

    label = _stat_label(stat)
    _convert_flux(1.0, units)  # validates units
    result = mc.completeness(stat, fap, sep_bins, flux_bins)
    sep, flux = result["sep"], result["flux"]
    positive = flux > 0.0
    if sep.size == 0 or not onp.any(positive):
        raise ValueError(
            "plot_completeness needs injections with positive fluxes."
        )
    fraction = result["completeness"][:, positive]
    flux = flux[positive]

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    x_edges = _cell_edges(sep, log=False)
    y_edges = _convert_flux(_cell_edges(flux, log=True), units)
    mesh = ax.pcolormesh(
        x_edges,
        y_edges,
        fraction.T,
        cmap=cmap,
        vmin=0.0,
        vmax=1.0,
        shading="flat",
        zorder=0,
    )
    if colorbar:
        fig.colorbar(
            mesh,
            ax=ax,
            pad=0.01,
            label=f"Completeness at FAP {_format_probability(fap)}",
        )
    outline = [path_effects.withStroke(linewidth=4, foreground="k")]
    for k, level in enumerate(contours):
        curve_sep, curve_flux = mc.contrast_curve(
            stat, fap, level, sep_bins, flux_bins
        )
        ax.plot(
            curve_sep,
            _convert_flux(curve_flux, units),
            color="white",
            ls=_LINE_STYLES[k % 4],
            lw=2,
            marker="o",
            ms=4,
            path_effects=outline,
            label=f"{100.0 * level:g}% completeness",
        )

    if units != "delta_mag":
        ax.set_yscale("log")
    ax.set_xlim(x_edges[0], x_edges[-1])
    ax.set_ylim(onp.min(y_edges), onp.max(y_edges))
    # Fainter companions lower down, as in plot_contrast_curve.
    if units != "flux" and not ax.yaxis_inverted():
        ax.invert_yaxis()
    ax.set_xlabel("Separation (mas)")
    ax.set_ylabel(f"Companion contrast ({_UNIT_LABELS[units]})")
    if title is None:
        title = f"Completeness of {label} at FAP {_format_probability(fap)}"
    ax.set_title(title)
    if len(contours):
        # Faint companions close in (lower left) are rarely detected, so
        # the legend hides the least informative part of the map.
        ax.legend(loc="lower left", fontsize="small")
    return fig, ax


@_styled
def plot_null_distribution(
    mc,
    stat="delta_chi2",
    *,
    observed=None,
    fap=None,
    reference=True,
    ax=None,
    title=None,
    figsize=(7, 4.5),
):
    """Plot how often companion-free searches exceed each threshold.

    The empirical exceedance function of the statistic over the null
    simulations, on a log scale. See
    [detection limits](../detection_limits.md#detection-plots).

    Parameters
    ----------
    mc : DetectionMC
        Simulated searches, from
        [`injection_recovery`][virgil.detection.injection_recovery].
    stat : str, optional
        ``"delta_chi2"`` (default), ``"log_bayes_factor"`` or ``"max_snr"``.
    observed : float, optional
        The statistic of the real data: a vertical line, with its false-alarm
        probability and 95% interval.
    fap : float, optional
        A false-alarm probability to draw as a horizontal line, with the
        threshold that reaches it.
    reference : bool, optional
        Draw the single-position reference curve (default True; none for
        ``log_bayes_factor``).
    title, ax, figsize : optional
        Title (default: statistic and number of draws), axes, and size of a new
        figure.

    Returns
    -------
    tuple
        ``(fig, ax)``.

    Examples
    --------
    >>> plot_null_distribution(mc, "delta_chi2", observed=21.3)  # doctest: +SKIP
    """
    from scipy.stats import norm

    label = _stat_label(stat)
    null = onp.asarray(mc.null[stat], dtype=float)
    n = null.size
    if n == 0:
        raise ValueError("The DetectionMC has no null draws.")
    x = onp.sort(onp.where(onp.isnan(null), -onp.inf, null))
    exceed = (n - onp.arange(n)) / n  # P(null >= x[i])
    finite = onp.isfinite(x)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    ax.step(
        x[finite],
        exceed[finite],
        where="pre",
        color="C0",
        lw=2,
        label=f"grid search, {n} null simulations",
    )
    right = [x[finite].max() if onp.any(finite) else 1.0]
    bottom = 0.5 / n

    if fap is not None:
        threshold = mc.threshold(stat, fap, n_boot=0)[0]
        right.append(threshold)
        bottom = min(bottom, 0.5 * float(fap))
        ax.axhline(
            fap,
            color="grey",
            lw=1,
            ls=":",
            label=f"FAP {_format_probability(fap)}",
        )
        ax.axvline(
            threshold,
            color="C0",
            lw=1,
            ls="--",
            label=f"empirical threshold, {label} = {threshold:.3g}",
        )
    if observed is not None:
        observed = float(observed)
        p, low, high = (
            float(v) for v in mc.false_alarm_probability(stat, observed)
        )
        right.append(observed)
        bottom = min(bottom, 0.5 * low) if low > 0.0 else bottom
        ax.axvline(observed, color="C3", lw=1.5)
        ax.errorbar(
            observed,
            p,
            yerr=[[max(p - low, 0.0)], [max(high - p, 0.0)]],
            fmt="o",
            color="C3",
            capsize=3,
            label=(
                f"observed {label} = {observed:.3g}, "
                f"FAP {_format_probability(p)}"
            ),
        )

    left = min(0.0, x[finite].min()) if onp.any(finite) else 0.0
    xmax = 1.1 * max(right) if max(right) > 0 else 1.0
    if reference and stat != "log_bayes_factor":
        grid = onp.linspace(max(left, 0.0), xmax, 400)
        if stat == "delta_chi2":
            ref = norm.sf(onp.sqrt(grid))  # = ½ P(χ²₁ ≥ x)
            ref_label = r"one position fixed in advance, $\frac{1}{2}\chi^2_1$"
        else:
            ref = norm.sf(grid)
            ref_label = "one position fixed in advance, Gaussian"
        ax.plot(grid, ref, color="k", ls="--", lw=1.2, label=ref_label)

    ax.set_yscale("log")
    ax.set_xlim(left, xmax)
    ax.set_ylim(bottom, 1.5)
    ax.set_xlabel(f"Threshold {label}")
    ax.set_ylabel("False-alarm probability\nP(null ≥ threshold)")
    ax.set_title(title or f"Null distribution of {label}")
    ax.grid(alpha=0.3)
    _legend_outside(ax)
    return fig, ax


# === ORBITS ===


def _orbit_tracks(orbits, n_points):
    """``(dra, ddec)`` over one period of each orbit, shape (n, n_points)."""
    import jax
    import jax.numpy as np

    batched = jax.tree_util.tree_map(np.atleast_1d, orbits)
    phase = np.linspace(0.0, 1.0, n_points)

    def track(orbit):
        # Times relative to the orbit's t_ref, from one periastron to the
        # next, so the tracks need no absolute (float32-unsafe) MJDs.
        dra, ddec, _ = orbit._relative(orbit.dt_peri + orbit.period * phase)
        return dra, ddec

    dra, ddec = jax.vmap(track)(batched)
    return onp.asarray(dra), onp.asarray(ddec)


@_styled
def plot_orbit_ensemble(
    orbits,
    positions=None,
    truth=None,
    *,
    n_points=400,
    n_sigma=1.0,
    color="#3c6e9f",
    alpha=None,
    truth_color="#d1495b",
    cmap="viridis",
    ax=None,
    figsize=(6.5, 6),
):
    """Draw orbits on the sky, e.g. draws from a posterior, with the data.

    East is to the left and North up, the primary at the origin. Each orbit is
    drawn over one period; measured positions are coloured by time, with error
    ellipses.

    Parameters
    ----------
    orbits : KeplerOrbit
        One orbit, or many as one [`KeplerOrbit`][virgil.orbits.KeplerOrbit]
        whose elements are 1-D arrays of equal length.
    positions : PositionData, optional
        Measured positions to overlay, with ``n_sigma`` error ellipses.
    truth : KeplerOrbit, optional
        A reference orbit, drawn as a thicker contrasting line.
    n_points : int, optional
        Points along each orbit.
    n_sigma : float, optional
        Size of the error ellipses, in standard deviations.
    color, truth_color : optional
        Colours of the ensemble and of the reference orbit.
    alpha : float, optional
        Opacity of each line; by default it falls with the number of orbits.
    cmap : str, optional
        Colour map for the epochs' times.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on.
    figsize : tuple, optional
        Size of a new figure.

    Returns
    -------
    tuple
        ``(fig, ax)``.
    """
    from matplotlib.collections import LineCollection
    from matplotlib.lines import Line2D
    from matplotlib.patches import Ellipse

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    dra, ddec = _orbit_tracks(orbits, n_points)
    if alpha is None:
        alpha = float(onp.clip(8.0 / len(dra), 0.03, 0.8))
    ax.add_collection(
        LineCollection(
            onp.stack([dra, ddec], -1),
            colors=color,
            linewidths=0.7,
            alpha=alpha,
            zorder=1,
        )
    )
    handles = [Line2D([], [], color=color, lw=1.2, alpha=0.8)]
    labels = [f"{len(dra)} sampled orbits" if len(dra) > 1 else "orbit"]

    if truth is not None:
        true_dra, true_ddec = _orbit_tracks(truth, n_points)
        (line,) = ax.plot(
            true_dra[0], true_ddec[0], color=truth_color, lw=1.8, zorder=2
        )
        handles.append(line)
        labels.append("true orbit")

    (star,) = ax.plot(
        0.0, 0.0, "*", color="k", ms=14, mec="w", mew=0.6, zorder=4
    )
    handles.append(star)
    labels.append("primary")

    if positions is not None:
        mjd = positions.t_ref + onp.asarray(positions.dt, dtype=float)
        x = onp.asarray(positions.dra, dtype=float)
        y = onp.asarray(positions.ddec, dtype=float)
        norm = matplotlib.colors.Normalize(
            mjd.min(), max(mjd.max(), mjd.min() + 1)
        )
        colours = matplotlib.colormaps[cmap](norm(mjd))
        # The (n, 2, 2) covariances of the positions.
        chol = onp.linalg.inv(onp.asarray(positions.whitener, dtype=float))
        for xi, yi, cov, c in zip(
            x, y, chol @ onp.swapaxes(chol, -1, -2), colours
        ):
            values, vectors = onp.linalg.eigh(cov)
            angle = onp.degrees(onp.arctan2(vectors[1, -1], vectors[0, -1]))
            width, height = (
                2 * n_sigma * onp.sqrt(onp.maximum(values[::-1], 0))
            )
            ax.add_patch(
                Ellipse(
                    (xi, yi),
                    width,
                    height,
                    angle=angle,
                    fc="none",
                    ec=c,
                    lw=1.0,
                    zorder=3,
                )
            )
        points = ax.scatter(
            x,
            y,
            c=mjd,
            cmap=cmap,
            norm=norm,
            s=36,
            edgecolors="k",
            linewidths=0.5,
            zorder=5,
        )
        handles.append(points)
        labels.append("measured positions")
        bar = fig.colorbar(points, ax=ax, shrink=0.8, pad=0.02)
        bar.set_label("Epoch (MJD)")

    ax.autoscale_view()
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("Δα (mas, East)")
    ax.set_ylabel("Δδ (mas, North)")
    _enforce_sky_orientation(ax)
    # Below the axes, so that it never hides an orbit.
    ax.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=2,
        frameon=False,
        fontsize="small",
    )
    return fig, ax


# === DATA ===


@_styled
def plot_uv_coverage(data, ax=None, cmap="viridis", figsize=(6, 5.5)):
    """Plot the uv coverage of data, coloured by wavelength.

    Each sample is drawn at ``(u, v)`` and at its mirror ``(-u, -v)``
    (the visibility there is the conjugate), in units of millions of
    wavelengths, with East (positive ``u``) to the left as on the sky.

    Parameters
    ----------
    data : OIData
        Data to show. Flagged samples are drawn too: the coverage is
        that of the observation.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on; a new figure is made if omitted.
    cmap : str, optional
        Colour map of the wavelength, used when the data have more than
        one wavelength (one colour otherwise).
    figsize : tuple, optional
        Size of a new figure.

    Returns
    -------
    tuple
        ``(fig, ax)``.
    """
    u = onp.asarray(data.u, dtype=float)
    v = onp.asarray(data.v, dtype=float)
    wavel = onp.broadcast_to(onp.asarray(data.wavel, dtype=float), u.shape)
    uu, vv = u / wavel / 1e6, v / wavel / 1e6
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    if onp.unique(wavel).size > 1:
        colour = onp.concatenate([wavel, wavel]) * 1e6
        points = ax.scatter(
            onp.concatenate([uu, -uu]),
            onp.concatenate([vv, -vv]),
            c=colour,
            cmap=cmap,
            s=10,
        )
        fig.colorbar(points, ax=ax, label="Wavelength (µm)", pad=0.01)
    else:
        ax.scatter(uu, vv, s=12, c="C0", label="(u, v)")
        ax.scatter(-uu, -vv, s=12, c="C1", label="(−u, −v)")
        ax.legend(loc="upper right", fontsize="small")
    limit = 1.05 * float(onp.max(onp.hypot(uu, vv))) if uu.size else 1.0
    ax.set_xlim(limit, -limit)  # East (positive u) to the left
    ax.set_ylim(-limit, limit)
    ax.set_aspect("equal")
    ax.set(xlabel="u (Mλ)", ylabel="v (Mλ)", title="uv coverage")
    return fig, ax


@_styled
def plot_oidata_overview(oidata, figsize=(15, 4.5)):
    """Plot uv coverage, visibilities and phases of an OIData object.

    Parameters
    ----------
    oidata : OIData
        Data to show. Flagged samples are left out. Projected observables
        (``vis_mat``/``phi_mat``) have no baseline and are not shown against
        it.
    figsize : tuple, optional
        Figure size.

    Returns
    -------
    tuple
        ``(fig, (ax_uv, ax_vis, ax_phi))``.
    """
    uu = onp.asarray(oidata.u / oidata.wavel) / 1e6
    vv = onp.asarray(oidata.v / oidata.wavel) / 1e6
    uu, vv = onp.broadcast_arrays(uu, vv)
    baseline = onp.hypot(uu, vv)
    fig, (ax_uv, ax_vis, ax_phi) = plt.subplots(1, 3, figsize=figsize)

    ax_uv.scatter(uu, vv, s=12, c="C0")
    ax_uv.scatter(-uu, -vv, s=12, c="C1")
    ax_uv.set(xlabel="u (Mλ)", ylabel="v (Mλ)", title="uv coverage")
    ax_uv.set_aspect("equal")
    ax_uv.invert_xaxis()  # East to the left

    vis = onp.asarray(oidata.vis)
    d_vis = onp.asarray(oidata.d_vis)
    if oidata.vis_mat is None and vis.size:
        index = (
            onp.arange(baseline.size)
            if oidata.vis_index is None
            else onp.asarray(oidata.vis_index)
        )
        ax_vis.errorbar(
            baseline[index], vis, yerr=d_vis, fmt=".", ms=6, elinewidth=0.6
        )
        name = {"v2": "V²", "amp": "|V|", "logamp": "log |V|"}
        ax_vis.set_ylabel(name.get(oidata.vis_mode, "visibility"))
        ax_vis.set_xlabel("Baseline (Mλ)")
    else:
        ax_vis.errorbar(onp.arange(vis.size), vis, yerr=d_vis, fmt=".")
        ax_vis.set(xlabel="Observable index", ylabel="Projected visibility")
    ax_vis.set_title("Visibilities")

    phi = onp.rad2deg(onp.asarray(oidata.phi))
    d_phi = onp.rad2deg(onp.asarray(oidata.d_phi))
    if oidata.phi_mat is None and phi.size:
        if oidata.cp_flag:
            legs = [onp.asarray(i) for i in (oidata.i_cps1, oidata.i_cps2)]
            legs.append(onp.asarray(oidata.i_cps3))
            x = onp.max([baseline[leg] for leg in legs], axis=0)
            ax_phi.set_xlabel("Longest baseline (Mλ)")
            ax_phi.set_ylabel("Closure phase (deg)")
        else:
            index = (
                onp.arange(baseline.size)
                if oidata.phi_index is None
                else onp.asarray(oidata.phi_index)
            )
            x = baseline[index]
            ax_phi.set_xlabel("Baseline (Mλ)")
            ax_phi.set_ylabel("Phase (deg)")
        ax_phi.errorbar(x, phi, yerr=d_phi, fmt=".", ms=6, elinewidth=0.6)
    else:
        ax_phi.errorbar(onp.arange(phi.size), phi, yerr=d_phi, fmt=".")
        ax_phi.set(xlabel="Observable index", ylabel="Projected phase")
    ax_phi.set_title("Phases")
    fig.tight_layout()
    return fig, (ax_uv, ax_vis, ax_phi)


def _cut(image, pa_deg, pixel_scale, offsets):
    """``image`` sampled along the line through its centre at PA ``pa_deg``."""
    from scipy.ndimage import map_coordinates

    t = onp.deg2rad(pa_deg)
    row = (image.shape[0] - 1) / 2 - offsets * onp.cos(t) / pixel_scale
    col = (image.shape[1] - 1) / 2 - offsets * onp.sin(t) / pixel_scale
    return map_coordinates(image, [row, col], order=1)


def _draw_cuts(ax, curves, nfaint):
    for values, colour, label in curves:
        x, y = values
        ax.plot(x, y[1 : 1 + nfaint].T, color=colour, alpha=0.06, lw=0.8)
        ax.plot(x, y[0], color=colour, lw=1.8, label=label)


@_styled
def plot_field_draws(
    field=None,
    pixel_scale_mas=None,
    key=None,
    *,
    latents=None,
    images=None,
    n=100,
    cuts=None,
    axes_pa=((0.0, "vertical"), (90.0, "horizontal")),
    mmax=3,
    npix=64,
    truth=None,
    snr=False,
    beam=None,
    snr_scale="linear",
    snr_vmax=None,
    n_cuts=100,
    title=None,
    figsize=(9, 6),
):
    r"""Draws from a Gaussian-process image prior or posterior, with cuts.

    !!! warning "Provisional"
        May change in any 0.4.x release without a deprecation cycle.

    Three draws (``abc``) above cuts through many (``ddd``), one in bold.
    Posterior draws are unit-sum on a shared colour scale; prior draws have
    unit peak.

    Parameters
    ----------
    field : GaussianField, MultipoleField or MultipoleDisk, optional
        Alone, prior draws from ``key``; with ``latents``, posterior draws. A
        [`MultipoleDisk`][virgil.models.MultipoleDisk] needs a
        [`RadialGP`][virgil.models.RadialGP] profile; latents are drawn at its
        hyperparameters.
    pixel_scale_mas : float
        Pixel scale, mas.
    key : jax.Array, optional
        PRNG key (prior).
    latents : array-like, shape (n, *field.latent.shape) or dict, optional
        Posterior latents, e.g. from
        [`laplace_samples`][virgil.imaging.laplace_samples]. For a
        MultipoleDisk, a dict of disk paths, ``{"profile.z": ...,
        "harmonics.z": ...}``, each ``(n, *shape)``; others keep their values.
    images : array-like, shape (n, nrow, ncol), optional
        Rendered images instead of a field (not for ``"multipole"`` cuts).
    n : int, optional
        Prior draws (default 100).
    cuts : {"principal", "axes", "multipole"}, optional
        Panel d. ``"principal"``: central row and column of a GaussianField.
        ``"axes"``: lines through the centre at ``axes_pa``.
        ``"multipole"``: cosine amplitudes ``a_m(r)`` of η = Σ a_m cos mθ +
        b_m sin mθ (the residual without ``mean``; ``a_0`` is the azimuthal
        mean), one offset track per ``m = 0..mmax``; for a MultipoleDisk,
        ``ln I₀ - μ`` and the coefficients ``c_m(r)`` of ``I/I₀``. Default:
        ``"multipole"`` for a MultipoleField or disk, ``"principal"`` for a
        GaussianField, ``"axes"`` for images.
    axes_pa : sequence of (float, str), optional
        ``(PA in degrees, label)``, North through East, per ``"axes"`` cut.
    npix : int, optional
        Image side in pixels for a disk (default 64).
    truth : array-like, shape (nrow, ncol), optional
        True image (posterior only): dashed in the cuts; the reference peak.
    snr : bool, optional
        Posterior only. Panel c: mean over standard deviation of each pixel
        (grey where no spread); per-pixel S/N, not a detection significance.
    beam : Beam, optional
        Drawn on the map panels, as in [`plot_model`][virgil.plotting.plot_model].
    snr_scale, snr_vmax, n_cuts, title, figsize : optional
        S/N scale (``"linear"``/``"log"``), its top (default 99th percentile),
        faint cuts (100), title, size.

    Returns
    -------
    tuple
        ``(fig, axes)``, ``axes`` a dict keyed ``"a"`` to ``"d"``. Prior cuts
        show ``η``; others ``log10`` brightness over the reference peak.
    """
    import jax
    import jax.numpy as np

    if (images is None) == (field is None):
        raise ValueError("Give either a field or images.")
    if pixel_scale_mas is None:
        raise ValueError("pixel_scale_mas is required.")
    posterior = images is not None or latents is not None
    if snr and not posterior:
        raise ValueError("snr=True needs posterior input: latents or images.")
    if field is not None:
        if isinstance(field, MultipoleDisk):
            latents = _disk_latents(field, latents, key, n)
            h = float(pixel_scale_mas)
            imgs = onp.asarray(
                jax.vmap(
                    lambda z: _disk_with(field, z).render(int(npix), npix * h)
                )(latents)
            )
            etas = None
        elif not isinstance(field, (GaussianField, MultipoleField)):
            raise TypeError(
                "field must be a GaussianField, MultipoleField or "
                "MultipoleDisk."
            )
        else:
            if latents is None:
                if key is None:
                    raise ValueError("Prior draws need a PRNG key.")
                latents = jax.random.normal(
                    key, (n, *field.latent.shape), dtype=field.latent.dtype
                )
            latents = np.asarray(latents)
            etas = onp.asarray(
                jax.vmap(
                    lambda z: _with_latent(field, z).evaluate(pixel_scale_mas)
                )(latents)
            )
            imgs = onp.exp(etas)
    else:
        imgs = onp.asarray(images, dtype=float)
        etas = None
    if imgs.ndim != 3 or imgs.shape[0] < 3:
        raise ValueError("Need at least three draws of 2D images.")
    h = float(pixel_scale_mas)
    imgs = imgs / (
        imgs.sum(axis=(1, 2), keepdims=True)
        if posterior
        else imgs.max(axis=(1, 2), keepdims=True)
    )
    if cuts is None:
        cuts = (
            "axes"
            if field is None
            else "multipole"
            if isinstance(field, (MultipoleField, MultipoleDisk))
            else "principal"
        )
    if cuts not in ("principal", "axes", "multipole"):
        raise ValueError(f"Unknown cuts mode {cuts!r}.")
    if cuts == "multipole" and not isinstance(
        field, (MultipoleField, MultipoleDisk)
    ):
        raise ValueError("cuts='multipole' needs a MultipoleField or disk.")
    if (
        cuts == "principal"
        and field is not None
        and not isinstance(field, GaussianField)
    ):
        raise ValueError("cuts='principal' needs a GaussianField; use 'axes'.")

    nrow, ncol = imgs.shape[1:]
    ext = [ncol * h / 2, -ncol * h / 2, -nrow * h / 2, nrow * h / 2]
    fig = plt.figure(figsize=figsize, layout="constrained")
    ax = fig.subplot_mosaic("abc\nddd", height_ratios=[1, 0.8])
    name = "posterior draw" if posterior else "draw"
    vmax = imgs[:3].max()
    for i, lab in enumerate("abc"[: 2 if snr else 3]):
        ax[lab].imshow(
            imgs[i],
            origin="upper",
            extent=ext,
            cmap="inferno",
            vmin=0,
            vmax=vmax,
        )
        ax[lab].set_title(f"{name} {i + 1}")
    if snr:
        mean, std = imgs.mean(0), imgs.std(0)
        live = std > 1e-12 * imgs.max()
        ratio = onp.where(live, mean / onp.where(live, std, 1.0), onp.nan)
        top = onp.nanpercentile(ratio, 99) if snr_vmax is None else snr_vmax
        if snr_scale == "log":
            from matplotlib.colors import LogNorm

            norm = LogNorm(
                vmin=max(float(onp.nanpercentile(ratio, 5)), 0.1), vmax=top
            )
        elif snr_scale == "linear":
            from matplotlib.colors import Normalize

            norm = Normalize(vmin=0, vmax=top)
        else:
            raise ValueError("snr_scale must be 'linear' or 'log'.")
        cmap = plt.get_cmap("viridis").with_extremes(bad="0.15")
        shown = ax["c"].imshow(
            ratio, origin="upper", extent=ext, cmap=cmap, norm=norm
        )
        ax["c"].set_title("mean / std")
        fig.colorbar(shown, ax=ax["c"], shrink=0.85, label="S/N per pixel")
    for lab in "abc":
        ax[lab].set_xlabel("ΔRA [mas]")
        if beam is not None:
            _draw_beam(ax[lab], beam)
    ax["a"].set_ylabel("ΔDec [mas]")

    d = ax["d"]
    nfaint = max(n_cuts, 0)
    if truth is not None:
        if not posterior:
            raise ValueError("truth is for posterior input.")
        truth = onp.asarray(truth, dtype=float)
        truth = truth / truth.sum()
    if posterior or cuts == "axes":
        ref = (truth if truth is not None else imgs[0]).max()

        def level(a):
            return onp.log10(onp.maximum(a / ref, 1e-3))

        ylabel = "log₁₀ brightness / reference peak"
    else:

        def level(a):
            return a

        ylabel = "log-brightness η"
    if cuts == "multipole" and isinstance(field, MultipoleDisk):
        r, tracks = _disk_tracks(field, latents, mmax)
        bottom = None
        for m, y in enumerate(tracks):
            hi, lo = max(float(y.max()), 0.0), min(float(y.min()), 0.0)
            gap = 0.1 * (hi - lo) + 1e-12
            off = -hi if bottom is None else bottom - gap - hi
            bottom = off + lo
            _draw_cuts(d, [((r, y + off), f"C{m}", None)], nfaint)
            d.axhline(off, color="k", lw=0.4)
            d.text(
                1.01,
                off,
                "ln I₀ − μ" if m == 0 else f"c_{m} (cos)",
                color=f"C{m}",
                va="center",
                transform=d.get_yaxis_transform(),
            )
        d.set_yticks([])
        d.set_xlim(r[0], r[-1])
        d.set_xlabel("disk-plane radius [mas]")
        d.set_ylabel("profile, harmonics (offset)")
    elif cuts == "multipole":
        polar = onp.asarray(
            jax.vmap(lambda z: _with_latent(field, z).polar_values())(latents)
        )
        coef = onp.fft.rfft(polar, axis=2) * 2 / polar.shape[2]
        coef[:, :, 0] *= 0.5
        top = polar.shape[2] // 2
        if polar.shape[2] % 2 == 0:
            coef[:, :, top] *= 0.5  # the Nyquist bin is real, like m = 0
        r = field.r_min_mas + field.spacing * onp.arange(polar.shape[1])
        step = 2.5 * float(onp.max(onp.asarray(field.sigma)))
        for m in range(min(mmax, top) + 1):
            off = -step * m
            c = coef[:, :, m].real + off
            _draw_cuts(d, [((r, c), f"C{m}", None)], nfaint)
            d.axhline(off, color="k", lw=0.4)
            d.text(
                1.01,
                off,
                f"m = {m}" + (" (cos)" if m else ""),
                color=f"C{m}",
                va="center",
                transform=d.get_yaxis_transform(),
            )
        d.set_yticks([])
        d.set_xlim(r[0], r[-1])
        d.set_xlabel("disk-plane radius [mas]")
        d.set_ylabel("amplitude of η (offset by m)")
    else:
        if cuts == "principal":
            vals = level(etas if etas is not None and not posterior else imgs)
            s_row = (nrow / 2 - 0.5 - onp.arange(nrow)) * h
            s_col = (ncol / 2 - 0.5 - onp.arange(ncol)) * h
            curves = [
                ((s_row, vals[:, :, ncol // 2]), "C0", "along vertical axis"),
                (
                    (s_col, vals[:, nrow // 2, :]),
                    "C1",
                    "along horizontal axis",
                ),
            ]
            if truth is not None:
                tv = level(truth)
                d.plot(s_row, tv[:, ncol // 2], "C0--", lw=1.2)
                d.plot(s_col, tv[nrow // 2, :], "C1--", lw=1.2)
        else:
            s = onp.linspace(-ncol * h / 2, ncol * h / 2, 200)
            curves = []
            for j, (pa, label) in enumerate(axes_pa):
                y = level(onp.stack([_cut(im, pa, h, s) for im in imgs]))
                curves.append(((s, y), f"C{j}", f"{label} (PA {pa:g}°)"))
                if truth is not None:
                    d.plot(s, level(_cut(truth, pa, h, s)), f"C{j}--", lw=1.2)
        _draw_cuts(d, curves, nfaint)
        if truth is not None:
            d.plot([], [], "k--", lw=1.2, label="truth")
        if posterior and cuts == "axes":
            d.set_ylim(-3.05, 0.6)
        d.set_xlim(curves[0][0][0][0], curves[0][0][0][-1])
        if cuts == "principal" and not posterior:
            d.axhline(0, color="k", lw=0.5)
        d.set_xlabel("offset along cut [mas]")
        d.set_ylabel(ylabel)
        d.legend(loc="upper right", frameon=False, ncols=3)
    if title:
        fig.suptitle(title)
    return fig, ax


def _with_latent(field, latent):
    """``field`` with its latents replaced."""
    import equinox as eqx

    return eqx.tree_at(lambda f: f.latent, field, latent)


def _disk_with(disk, latents):
    """``disk`` with the latents of the dict ``{path: z}`` replaced."""
    for path, z in latents.items():
        disk = disk.set(path, z)
    return disk


# deliberate: draws the RadialGP latents, or validates supplied ones against the disk, as arrays of shape (n, *shape)
def _disk_latents(disk, latents, key, n):
    """The latents of the draws, as a dict of arrays ``(n, *shape)``."""
    import jax
    import jax.numpy as np

    if not isinstance(disk.profile, RadialGP):
        raise TypeError(
            "plot_field_draws needs a MultipoleDisk with a RadialGP."
        )
    prior = getattr(disk.harmonics, "prior", None)
    if latents is None:
        if key is None:
            raise ValueError("Prior draws need a PRNG key.")
        paths = {"profile.z": disk.profile.z}
        if prior is not None:
            paths["harmonics.z"] = disk.harmonics.z
        keys = jax.random.split(key, len(paths))
        return {
            p: jax.random.normal(k, (n, *z.shape), dtype=z.dtype)
            for k, (p, z) in zip(keys, paths.items())
        }
    if not isinstance(latents, dict) or not latents:
        raise ValueError(
            "latents for a MultipoleDisk is a dict of paths such as "
            "{'profile.z': ..., 'harmonics.z': ...}."
        )
    allowed = {"profile.z"} | ({"harmonics.z"} if prior is not None else set())
    if set(latents) - allowed:
        raise ValueError(f"latents keys must be among {sorted(allowed)}.")
    return {p: np.asarray(z) for p, z in latents.items()}


# deliberate: vmaps the radial tracks ln I0 - mu and c_1 .. c_M over the latent draws on a common radius grid
def _disk_tracks(disk, latents, mmax):
    """Radii and the tracks ``ln I0 - mu``, ``c_1 .. c_M``, each ``(n, nr)``."""
    import jax
    import jax.numpy as np

    lo, hi = disk.profile.extent()
    r = np.linspace(lo, hi, 200)

    def tracks(z):
        d = _disk_with(disk, z)
        out = [np.log(d.profile.intensity(r, 1.0)) - d.profile.mu]
        if d.harmonics is not None and d.harmonics.degree:
            c, _ = d.harmonics.coefficients(r)
            out += list(c[: int(mmax)])
        return np.stack(out)

    y = onp.asarray(jax.vmap(tracks)(latents))
    return onp.asarray(r), [y[:, k] for k in range(y.shape[1])]
