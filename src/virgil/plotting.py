"""Plotting functions for virgil models, data and grid results.

* [`plot_grid_map`][virgil.plotting.plot_grid_map] draws any grid
  result (log likelihood, best-fit flux, uncertainty, S/N, contrast limit)
  with defaults chosen by ``kind=``, and
  [`plot_contrast_curve`][virgil.plotting.plot_contrast_curve] draws
  radial contrast curves.
* [`plot_null_distribution`][virgil.plotting.plot_null_distribution],
  [`plot_roc`][virgil.plotting.plot_roc] and
  [`plot_completeness`][virgil.plotting.plot_completeness] draw the
  simulated searches of a
  [`DetectionMC`][virgil.detection.DetectionMC]: false-alarm
  probabilities, ROC curves and completeness maps.
* Contrasts can be shown as flux ratios (companion/primary), as contrasts
  (primary/companion, e.g. 100) or in magnitudes (e.g. 5 mag), with
  ``units="flux" | "contrast" | "delta_mag"``.

Sky images follow the package convention: East (positive ``dra``) to the
left and North (positive ``ddec``) up. Importing this module does not change
matplotlib's global settings; call [`set_style`][virgil.plotting.set_style] to opt in to the
virgil look for every figure.
"""

import contextlib
import functools

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

from ._utils import is_flux_param, resolve_flux_param
from .limits import flux_to_contrast, flux_to_delta_mag, radial_profile


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


@contextlib.contextmanager
def _style_context(style=None):
    """Temporarily apply ``style`` (default ``STYLE``) to ``rcParams``.

    Only the keys in ``style`` are saved and restored. ``plt.rc_context``
    restores every key, including ``backend``, and in Jupyter that can
    switch backends on exit, which closes the open figures before they are
    shown.
    """
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


def _centres_to_extent(x_axis, y_axis):
    """Return an imshow ``extent`` whose pixels are centred on the samples.

    ``imshow`` places ``extent`` at the outer pixel *edges*, so grid sample
    positions (pixel centres) are padded by half a sample spacing.
    """

    def _edges(axis):
        axis = np.asarray(axis, dtype=float).reshape(-1)
        half = 0.5 * (axis[1] - axis[0]) if axis.size > 1 else 0.5
        return float(axis[0] - half), float(axis[-1] + half)

    return [*_edges(x_axis), *_edges(y_axis)]


def _is_sky_key(key, name):
    return key == name or key.endswith("." + name)


def _image_axes(grid, coord_keys):
    """Pick the plotted x/y keys and orient ``grid`` as ``image[y, x]``.

    ``grid`` has one axis per entry of ``coord_keys``, in that order. If the
    keys include a ``dra`` and a ``ddec`` (possibly as zodiax paths such as
    ``"comp.dra"``), those are used as x and y whatever their order.
    Otherwise the first two keys are used. Returns
    ``(x_key, y_key, image, is_sky)``.
    """
    grid = np.asarray(grid)
    if grid.ndim != 2 or len(coord_keys) != 2:
        raise ValueError(
            f"Expected a 2D grid with two coordinate keys; got shape "
            f"{grid.shape} for keys {list(coord_keys)}. Reduce the flux axis "
            "first, e.g. loglike_im.max(axis=...)."
        )
    dra = [k for k in coord_keys if _is_sky_key(k, "dra")]
    ddec = [k for k in coord_keys if _is_sky_key(k, "ddec")]
    if len(dra) == 1 and len(ddec) == 1:
        x_key, y_key = dra[0], ddec[0]
        is_sky = True
    else:
        x_key, y_key = coord_keys
        is_sky = False
    if list(coord_keys).index(x_key) == 0:
        image = grid.T
    else:
        image = grid
    return x_key, y_key, image, is_sky


def _range_aware_float_formatter(
    vmin, vmax, min_sigfigs=3, max_sigfigs=6, scale=1.0
):
    """Tick formatter whose significant figures adapt to the axis span."""
    span = abs(float(vmax) - float(vmin)) * abs(float(scale))
    if not np.isfinite(span) or span <= 0:
        sigfigs = min_sigfigs + 1
    else:
        sigfigs = int(
            np.clip(
                np.ceil(-np.log10(span)) + 2,
                min_sigfigs,
                max_sigfigs,
            )
        )
    return FuncFormatter(lambda x, _: f"{(x * scale):.{sigfigs}g}")


def _enforce_sky_orientation(ax):
    """Ensure the displayed x-axis increases toward the left (East) and
    the y-axis increases toward the top (North), matching virgil's
    image coordinate convention. Works regardless of
    whether the plotted array's ``dra``/``ddec`` axis was built ascending
    or descending, since it corrects the axes' final displayed limits
    rather than assuming any particular extent/origin construction.
    """
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
    vis_data = np.asarray(oidata.vis).reshape(-1)
    phi_data = np.asarray(oidata.phi).reshape(-1)
    d_vis_data = np.asarray(oidata.d_vis).reshape(-1)
    d_phi_data = np.asarray(oidata.d_phi).reshape(-1)

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
            np.asarray(pred["vis_mean"]).reshape(-1),
            xerr=d_vis_data,
            yerr=np.asarray(pred["vis_std"]).reshape(-1),
            fmt="o",
            markersize=4,
            alpha=0.45,
            elinewidth=1.0,
            capsize=2,
            color=color,
            label=label,
        )
        if not has_phase:
            vis_low.append(np.asarray(pred["vis_mean"]).min())
            vis_high.append(np.asarray(pred["vis_mean"]).max())
            continue
        ax2.errorbar(
            phi_data,
            np.asarray(pred["phi_mean"]).reshape(-1),
            xerr=d_phi_data,
            yerr=np.asarray(pred["phi_std"]).reshape(-1),
            fmt="o",
            markersize=4,
            alpha=0.45,
            elinewidth=1.0,
            capsize=2,
            color=color,
            label=label,
        )
        vis_low.append(np.asarray(pred["vis_mean"]).min())
        vis_high.append(np.asarray(pred["vis_mean"]).max())
        phi_low.append(np.asarray(pred["phi_mean"]).min())
        phi_high.append(np.asarray(pred["phi_mean"]).max())

    vis_pad = float(np.median(d_vis_data)) if d_vis_data.size else 0.0
    phi_pad = float(np.median(d_phi_data)) if d_phi_data.size else 0.0

    vis_min = min(vis_low) - vis_pad
    vis_max = max(vis_high) + vis_pad

    vis_line = np.linspace(vis_min, vis_max, 200)
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
        phi_line = np.linspace(phi_min, phi_max, 200)
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
    """Show a source model's rendered image with sky axes.

    East is to the left and North is up, matching the orientation of
    [`render`][virgil.models.SourceModel.render].

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
        Quantile (e.g. ``0.99``) at which to saturate the colour scale, so
        that faint structure next to a bright star is visible.
    cmap : str, optional
        Matplotlib colour map.
    beam : Beam, optional
        The data's resolution, from
        [`imaging.beam`][virgil.imaging.beam], drawn as a shaded FWHM
        ellipse in the lower-left corner, as is usual on reconstructed
        images.
    convolve : bool, optional
        If ``True``, show the image convolved with ``beam`` (see
        [`imaging.convolve_beam`][virgil.imaging.convolve_beam]): what
        the data resolve, rather than the super-resolved image.

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
    image = np.asarray(image)
    vmax = None if saturate is None else np.quantile(image, saturate)
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
    values = np.asarray(residual, dtype=float)
    label = "residual"
    if sigma is not None:
        values = values / np.asarray(sigma, dtype=float)
        label = "z-score"
    limit = float(np.nanmax(np.abs(values))) or 1.0
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
    dra = np.asarray(samples[dra_key], dtype=float)
    ddec = np.asarray(samples[ddec_key], dtype=float)
    flux_raw = np.asarray(samples[flux_key], dtype=float)
    flux = np.power(10.0, flux_raw) if log10_flux else flux_raw

    df = pd.DataFrame({"dra": dra, "ddec": ddec, "flux": flux})
    df["sep"] = np.sqrt(df["dra"] ** 2 + df["ddec"] ** 2)
    df["pa"] = (np.degrees(np.arctan2(df["dra"], df["ddec"])) + 360.0) % 360.0
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
    truth_polar = {
        "sep": float(
            np.sqrt(truth_cart["dra"] ** 2 + truth_cart["ddec"] ** 2)
        ),
        "pa": float(
            (
                np.degrees(np.arctan2(truth_cart["dra"], truth_cart["ddec"]))
                + 360.0
            )
            % 360.0
        ),
        "flux": truth_cart["flux"],
    }
    return truth_cart, truth_polar


def plot_hmc_fisher_chainconsumer(
    hmc_table,
    fisher_table,
    truth_cartesian,
    colors=("#1f77b4", "#ff7f0e"),
):
    """
    Plot paired Cartesian and polar ChainConsumer diagnostics for HMC and Fisher-HMC.

    Parameters
    ----------
    hmc_table : pandas.DataFrame
        Posterior table for vanilla HMC.
    fisher_table : pandas.DataFrame
        Posterior table for Fisher-reparameterized HMC.
    truth_cartesian : dict
        Truth mapping in Cartesian coordinates.
    colors : tuple[str, str], optional
        Colors used for HMC and Fisher-HMC chains.

    Returns
    -------
    dict
        Mapping with the ChainConsumer objects (``"cartesian"``,
        ``"polar"``), their figures (``"cartesian_figures"``,
        ``"polar_figures"``, each ``(corner, walks)``) and the truth
        mappings.
    """
    truth_cart, truth_polar = truth_cartesian_and_polar(truth_cartesian)

    cartesian_consumer, *cartesian_figures = plot_chainconsumer_diagnostics(
        {
            "HMC Cartesian": hmc_table,
            "Fisher-HMC Cartesian": fisher_table,
        },
        columns=["dra", "ddec", "flux"],
        truth=truth_cart,
        colors=list(colors),
    )

    polar_consumer, *polar_figures = plot_chainconsumer_diagnostics(
        {
            "HMC Polar": hmc_table,
            "Fisher-HMC Polar": fisher_table,
        },
        columns=["sep", "pa", "flux"],
        truth=truth_polar,
        colors=list(colors),
    )

    return {
        "cartesian": cartesian_consumer,
        "polar": polar_consumer,
        "cartesian_figures": tuple(cartesian_figures),
        "polar_figures": tuple(polar_figures),
        "truth_cartesian": truth_cart,
        "truth_polar": truth_polar,
    }


@_styled
def _reversed_cmap(cmap):
    """Reverse a colour map given by name or as a ``Colormap``.

    Maps whose name already ends in ``_r`` are taken as deliberately
    reversed and kept.
    """
    if isinstance(cmap, str):
        cmap = matplotlib.colormaps[cmap]
    if cmap.name.endswith("_r"):
        return cmap
    return cmap.reversed()


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
        value = np.asarray(percentile, dtype=float).reshape(-1)
        if value.size != 1:
            raise ValueError("percentile must be a scalar or length-1 array.")
        return f"{_format_sigma_or_percent_value(value[0] * 100)}% upper limit"
    if sigma is not None:
        value = np.asarray(sigma, dtype=float).reshape(-1)
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
        return np.asarray(values, dtype=float)
    if units == "contrast":
        return np.asarray(flux_to_contrast(values))
    if units == "delta_mag":
        return np.asarray(flux_to_delta_mag(values))
    raise ValueError(
        f"units must be 'flux', 'contrast' or 'delta_mag'; got {units!r}."
    )


def _grid_axes(values, samples_dict, kind, flux_param):
    """Coordinate keys of ``values``, reducing a flux axis if present.

    Returns ``(values, coord_keys)``. ``values`` may have one axis per key of
    ``samples_dict`` (a full grid), or one per key except the flux. A full
    log-likelihood grid is reduced to its maximum over the flux axis.
    """
    values = np.asarray(values, dtype=float)
    keys = list(samples_dict)
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
        values = np.nanmax(values, axis=keys.index(flux_key))
        return values, [key for key in keys if key != flux_key]
    if values.ndim == len(keys) - 1:
        flux_key = resolve_flux_param(keys, flux_param)
        return values, [key for key in keys if key != flux_key]
    raise ValueError(
        f"values has {values.ndim} axes but samples_dict has keys {keys}; "
        "expected one axis per key, or per key except the flux."
    )


def _coord_value(values, key, coord_keys):
    """Look up ``key`` in a dict, or by position in ``coord_keys`` order."""
    if isinstance(values, dict):
        return float(values[key])
    return float(values[list(coord_keys).index(key)])


@_styled
def plot_grid_map(
    values,
    samples_dict,
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
        Grid result with one axis per coordinate key of ``samples_dict``
        (every key except the flux), in key order, as returned by
        [`virgil.grid_fit`][virgil.grid_fit] and
        [`virgil.limits`][virgil.limits]. A full
        [`likelihood_grid`][virgil.grid_fit.likelihood_grid] (with the
        flux axis) is reduced to its maximum over flux. One coordinate gives
        a line plot.
    samples_dict : dict[str, array-like]
        The grid axes used to compute ``values``.
    kind : {"loglike", "flux", "sigma", "snr", "limit"}, optional
        What ``values`` holds, which sets the default colour map, scale and
        labels: a log likelihood, a best-fit flux, a flux uncertainty, a
        signal-to-noise ratio, or a flux upper limit.
    units : {"flux", "contrast", "delta_mag"}, optional
        For ``kind="flux"`` and ``"limit"``, show the values as flux ratios
        (companion/primary, default), contrasts (primary/companion, e.g.
        100), or magnitude differences (e.g. 5 mag).
    sigma, percentile : float, optional
        Confidence of a ``kind="limit"`` map, for its title (e.g.
        ``sigma=5`` gives "5σ limit").
    truth, best : dict or sequence, optional
        Coordinates to mark with a cross (truth) or circle (best fit). A dict
        is looked up by key; a sequence is read in coordinate-key order.
    star : bool, optional
        Mark the primary at the origin of sky maps (default True).
    flux_param : str, optional
        Key of the flux axis, if it is not the one key ending in ``flux``.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on; a new figure is made if omitted.
    cmap, log, label, title : optional
        Override the defaults for ``kind``: colour map, logarithmic colour
        scale, colour-bar label and title.
    figsize : tuple, optional
        Size of a new figure.

    Returns
    -------
    tuple
        ``(fig, ax)``.

    Notes
    -----
    When the coordinates include ``dra`` and ``ddec`` (also as paths such as
    ``"comp.dra"``) they are drawn as x and y, East-left and North-up,
    whatever their order in ``samples_dict``; pixels are centred on the
    samples.

    Examples
    --------
    >>> fig, (a, b) = plt.subplots(1, 2)  # doctest: +SKIP
    >>> plot_grid_map(flux, grid, kind="flux", ax=a)  # doctest: +SKIP
    >>> plot_grid_map(flux / sigma, grid, kind="snr", ax=b)  # doctest: +SKIP
    >>> plot_grid_map(limits, grid, kind="limit", units="delta_mag", sigma=5)  # doctest: +SKIP
    """
    if kind not in _KINDS:
        raise ValueError(
            f"kind must be one of {sorted(_KINDS)}; got {kind!r}."
        )
    defaults = _KINDS[kind]
    values, coord_keys = _grid_axes(values, samples_dict, kind, flux_param)

    if defaults["flux_like"]:
        values = _convert_flux(values, units)
        default_label = f"{defaults['label']} ({_UNIT_LABELS[units]})"
        default_log = units != "delta_mag"
        # Contrast and Δmag grow as the companion gets fainter; reverse the
        # colour map so that deeper limits keep the same colour.
        default_cmap = defaults["cmap"]
        if units != "flux":
            default_cmap = _reversed_cmap(default_cmap)
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
        ax.plot(np.asarray(samples_dict[x_key]), values, color="C0", lw=2)
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

    x_key, y_key, image, is_sky = _image_axes(values, coord_keys)
    finite = image[np.isfinite(image)]
    norm = None
    if log and finite.size and np.all(finite > 0):
        norm = matplotlib.colors.LogNorm()
    im = ax.imshow(
        image,
        origin="lower",
        extent=_centres_to_extent(samples_dict[x_key], samples_dict[y_key]),
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


def _sky_map(values, samples_dict):
    """A sky map with axes ``(dra, ddec)``, and those two axes.

    ``values`` has one axis per coordinate key of ``samples_dict`` (every key
    whose name does not end in ``flux``), in key order, so it is transposed
    when ``ddec`` comes before ``dra``.
    """
    keys = [key for key in samples_dict if not is_flux_param(key)]
    found = []
    for name in ("dra", "ddec"):
        matches = [key for key in keys if _is_sky_key(key, name)]
        if len(matches) != 1:
            raise ValueError(
                f"samples_dict needs exactly one {name!r} axis; keys are "
                f"{list(samples_dict)}."
            )
        found.append(matches[0])
    if len(keys) != 2:
        raise ValueError(
            f"A sky map needs exactly two coordinate keys; got {keys}."
        )
    values = np.asarray(values, dtype=float)
    if keys.index(found[0]) == 1:
        values = values.T
    dra, ddec = (np.asarray(samples_dict[key]) for key in found)
    return values, dra, ddec


@_styled
def plot_contrast_curve(
    values,
    samples_dict=None,
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
        A limit map (companion/primary flux ratios, with one axis per
        coordinate key of ``samples_dict`` in key order, as from
        [`absil_limits`][virgil.limits.absil_limits] or
        [`ruffio_upperlimit`][virgil.limits.ruffio_upperlimit]), or a
        [`radial_profile`][virgil.limits.radial_profile] of one.
    samples_dict : dict[str, array-like], optional
        The grid axes of a limit map; it must contain ``dra`` and ``ddec``
        axes (also as paths such as ``"comp.dra"``).
    units : {"delta_mag", "contrast", "flux"}, optional
        Show limits as magnitude differences (default), contrasts
        (primary/companion) or flux ratios (companion/primary). Deeper
        limits are always drawn lower down.
    sigma, percentile, label : optional
        Legend label; by default built from ``sigma`` or ``percentile``
        (e.g. "5σ limit"), or "Median limit".
    band : bool, optional
        Shade the 16–84 percentile range of each annulus (default True).
    truth : tuple, optional
        ``(dra, ddec, flux)`` of a detected companion to mark.
    center, r_max, bins : optional
        Annuli, passed to
        [`radial_profile`][virgil.limits.radial_profile].
    ax : matplotlib.axes.Axes, optional
        Axes to draw on, e.g. to overlay several curves.
    color : optional
        Line and band colour.
    figsize : tuple, optional
        Size of a new figure.

    Returns
    -------
    tuple
        ``(fig, ax)``.
    """
    if isinstance(values, dict):
        profile = values
    else:
        if samples_dict is None:
            raise ValueError("Pass samples_dict with a limit map.")
        sky_values, dra, ddec = _sky_map(values, samples_dict)
        profile = radial_profile(
            sky_values, dra, ddec, center=center, r_max=r_max, bins=bins
        )
    r = np.asarray(profile["r"])
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
            np.hypot(true_dra, true_ddec),
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
    """The statistic at a local (one-position) significance of n_sigma.

    At one position fixed in advance, ``delta_chi2`` follows ½δ₀ + ½χ²₁
    and the unconstrained flux SNR is a standard normal, so a one-sided
    ``n_sigma`` is ``delta_chi2 = n_sigma²`` or ``max_snr = n_sigma``. The
    log Bayes factor has no such reference (None).
    """
    if stat == "delta_chi2":
        return float(n_sigma) ** 2
    if stat == "max_snr":
        return float(n_sigma)
    return None


def _one_sided_fap(n_sigma):
    """One-sided Gaussian tail at ``n_sigma``: 0.135% at 3σ."""
    from scipy.stats import norm

    return float(norm.sf(float(n_sigma)))


def _format_probability(p):
    """A probability as a short percentage, or as a power of ten if tiny."""
    p = float(p)
    if not np.isfinite(p):
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
    if isinstance(value, np.ndarray) and value.ndim >= 1:
        return [tuple(v) if np.ndim(v) else float(v) for v in value]
    if isinstance(value, list):
        return list(value)
    return [value]


def _flux_selection_label(flux):
    if flux is None:
        return None
    if np.ndim(flux) == 0:
        flux = float(flux)
        if flux <= 0.0:
            return "flux 0"
        return f"flux {flux:.3g} (Δmag {float(flux_to_delta_mag(flux)):.2f})"
    lo, hi = flux
    return f"flux {lo:.3g}–{hi:.3g}"


def _sep_selection_label(sep_bin):
    if sep_bin is None:
        return None
    lo, hi = sep_bin
    return f"{lo:g}–{hi:g} mas"


def _selection_labels(flux, sep_bin):
    """Labels of a flux and separation selection (None is left out)."""
    labels = (_flux_selection_label(flux), _sep_selection_label(sep_bin))
    return [label for label in labels if label is not None]


def _legend_outside(ax):
    """A legend to the right of the axes, where it covers no data."""
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        fontsize="small",
        borderaxespad=0.0,
    )


def _rates_at(fpr, tpr, thresholds, value):
    """False- and true-positive rates of the threshold ``value``.

    ``thresholds`` decrease from ``+inf`` through every distinct score, so
    the smallest one at or above ``value`` has the same rates as ``value``.
    """
    i = int(np.count_nonzero(np.asarray(thresholds) >= value)) - 1
    return float(fpr[i]), float(tpr[i])


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

    Each curve follows a detection threshold from high to low: its
    false-positive rate (the fraction of companion-free simulations above
    the threshold, i.e. its false-alarm probability) on the x-axis and its
    true-positive rate (the fraction of injected companions recovered, the
    completeness) on the y-axis. A statistic that cannot tell companions
    from noise lies on the chance curve TPR = FPR; a better one lies above
    it.

    Parameters
    ----------
    mc : DetectionMC
        Simulated searches, from
        [`injection_recovery`][virgil.detection.injection_recovery].
    stat : str or sequence of str
        ``"delta_chi2"``, ``"log_bayes_factor"`` or ``"max_snr"``, or
        several of them to compare.
    flux : float, (float, float) or list, optional
        Only the injections of this flux, or in this range ``(lo, hi)``, as
        for [`DetectionMC.roc`][virgil.detection.DetectionMC.roc]. A list
        gives one curve per entry. By default, every injection.
    sep_bin : (float, float) or list of them, optional
        Only the injections with separations in ``[lo, hi)`` (mas); a list
        gives one curve per bin.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on; a new figure is made if omitted.
    log_fpr : bool, optional
        Logarithmic false-positive axis (default), from half a null draw's
        worth (0.5 / n_null) to 1, since detections are claimed at small
        false-alarm probabilities. On it the chance curve TPR = FPR is a
        curve, not a straight line.
    mark_wilks : sequence of float, optional
        Local significances (default 3σ and 5σ) to mark. A vertical line
        shows the false-alarm probability that Wilks's theorem gives them
        at one position fixed in advance, the one-sided Gaussian tail
        (0.135% at 3σ, 2.9×10⁻⁷ at 5σ), when it lies on the axis. A circle
        on every ``delta_chi2`` or ``max_snr`` curve shows where the
        threshold equals that local significance (``delta_chi2`` = 9 or
        ``max_snr`` = 3 at 3σ): its horizontal distance from the line is
        the look-elsewhere effect of searching a grid. None or ``()`` marks
        nothing.
    title : str, optional
        Title; by default it names what all the curves share (one
        statistic, one flux or separation selection), and the legend
        names what differs.
    figsize : tuple, optional
        Size of a new figure. The legend is drawn outside the axes; a
        tight bounding box (as Jupyter uses) includes it.

    Returns
    -------
    tuple
        ``(fig, ax)``.

    Notes
    -----
    With several statistics, the line style tells the statistics apart
    and the colour the flux or separation selections; with one statistic,
    both change from curve to curve. The legend sits outside the axes, to
    their right, as in
    [`plot_null_distribution`][virgil.plotting.plot_null_distribution].

    Examples
    --------
    Three statistics at one injected flux, then one statistic at several:

    >>> plot_roc(mc, ["delta_chi2", "log_bayes_factor", "max_snr"],
    ...          flux=4e-3)  # doctest: +SKIP
    >>> plot_roc(mc, "delta_chi2", flux=[2e-3, 4e-3, 8e-3])  # doctest: +SKIP
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
                x, y = _rates_at(fpr, tpr, thresholds, value)
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
        np.geomspace(fpr_min, 1.0, 200)
        if log_fpr
        else np.linspace(0.0, 1.0, 200)
    )
    ax.plot(chance, chance, color="grey", lw=1, label="chance (TPR = FPR)")
    for k, n_sigma in enumerate(marks):
        nominal = _one_sided_fap(n_sigma)
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
    c = np.asarray(centres, dtype=float)
    if log:
        c = np.log(c)
    if c.size == 1:
        half = 0.2 if log else max(0.5, 0.1 * abs(c[0]))
        edges = np.array([c[0] - half, c[0] + half])
    else:
        mid = 0.5 * (c[1:] + c[:-1])
        edges = np.concatenate(
            [[c[0] - (mid[0] - c[0])], mid, [c[-1] + (c[-1] - mid[-1])]]
        )
    return np.exp(edges) if log else edges


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

    Each cell is the fraction of injected companions of that separation and
    flux that a search detects at the false-alarm probability ``fap``
    ([`DetectionMC.completeness`][virgil.detection.DetectionMC.completeness]).
    Lines mark where the completeness reaches each level in ``contours``,
    the empirical contrast curves of
    [`DetectionMC.contrast_curve`][virgil.detection.DetectionMC.contrast_curve].

    Parameters
    ----------
    mc : DetectionMC
        Simulated searches with injections, from
        [`injection_recovery`][virgil.detection.injection_recovery].
    stat : str
        The detection statistic, e.g. ``"delta_chi2"``.
    fap : float
        False-alarm probability of the detection threshold, e.g. 1.35e-3
        (the one-sided Gaussian tail at 3σ).
    units : {"delta_mag", "contrast", "flux"}, optional
        The y-axis, as in
        [`plot_contrast_curve`][virgil.plotting.plot_contrast_curve]:
        magnitude difference (default, linear), contrast
        (primary/companion, log) or flux ratio (companion/primary, log).
        Fainter companions are always drawn lower down, so limit curves
        from ``plot_contrast_curve`` can be drawn on the same axes.
    contours : sequence of float, optional
        Completeness levels to draw as lines (default 50% and 90%); ``()``
        draws none.
    sep_bins, flux_bins : array-like, optional
        Bin edges (mas, and flux), as for ``DetectionMC.completeness``. By
        default every distinct injected separation and flux is a cell.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on; a new figure is made if omitted.
    cmap : str or Colormap, optional
        Colour map of the completeness, from 0 to 1.
    colorbar : bool, optional
        Add a colour bar (default True).
    title : str, optional
        Title; by default it gives the statistic and the FAP.
    figsize : tuple, optional
        Size of a new figure.

    Returns
    -------
    tuple
        ``(fig, ax)``.

    Notes
    -----
    Cells span halfway to their neighbours, in separation and in log flux.
    Injections of zero flux have no Δmag or contrast and are left out. The
    contour lines are ``contrast_curve``'s: in each separation bin the
    completeness is made non-decreasing in flux and interpolated linearly
    in log flux (linearly in Δmag), so they are exactly the curves that
    method returns, and they are absent where the injected fluxes do not
    bracket the level. The map sets the axis limits to its cells, so
    curves drawn afterwards do not widen them.

    Examples
    --------
    A completeness map with Absil limits of companion-free data
    overplotted (``plot_contrast_curve`` redraws the legend, so place it
    again at the end):

    >>> fig, ax = plot_completeness(mc, "delta_chi2", 1.35e-3)  # doctest: +SKIP
    >>> plot_contrast_curve(absil_map, grid, sigma=3, band=False,
    ...                     ax=ax)  # doctest: +SKIP
    >>> ax.legend(loc="lower left", fontsize="small")  # doctest: +SKIP
    """
    import matplotlib.patheffects as path_effects

    label = _stat_label(stat)
    _convert_flux(1.0, units)  # validates units
    result = mc.completeness(stat, fap, sep_bins, flux_bins)
    sep, flux = result["sep"], result["flux"]
    positive = flux > 0.0
    if sep.size == 0 or not np.any(positive):
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
    ax.set_ylim(np.min(y_edges), np.max(y_edges))
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

    The curve is the empirical exceedance (survival) function of the
    statistic over the null simulations: at each threshold, the fraction
    of companion-free searches at or above it, that is the threshold's
    false-alarm probability. It is drawn on a log scale, so the tail that
    sets detection thresholds is visible.

    Parameters
    ----------
    mc : DetectionMC
        Simulated searches, from
        [`injection_recovery`][virgil.detection.injection_recovery].
    stat : str, optional
        ``"delta_chi2"`` (default), ``"log_bayes_factor"`` or
        ``"max_snr"``.
    observed : float, optional
        The statistic of the real data. It is drawn as a vertical line, with
        its empirical false-alarm probability
        ([`DetectionMC.false_alarm_probability`][virgil.detection.DetectionMC.false_alarm_probability])
        and the 95% interval of that probability as a point with error
        bars.
    fap : float, optional
        A false-alarm probability to draw as a horizontal line, with the
        empirical threshold that reaches it
        ([`DetectionMC.threshold`][virgil.detection.DetectionMC.threshold])
        as a vertical line.
    reference : bool, optional
        Draw what the statistic would do at one position fixed in advance
        (default True): ½ P(χ²₁ ≥ x) for ``delta_chi2`` (Wilks's theorem
        with the flux on its boundary at zero), the Gaussian tail for
        ``max_snr``. The gap between this curve and the empirical one is
        the look-elsewhere effect. ``log_bayes_factor`` has no reference.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on; a new figure is made if omitted.
    title : str, optional
        Title; by default it names the statistic and the number of draws.
    figsize : tuple, optional
        Size of a new figure. The legend is drawn outside the axes; a
        tight bounding box (as Jupyter uses) includes it.

    Returns
    -------
    tuple
        ``(fig, ax)``.

    Notes
    -----
    The legend sits outside the axes, to their right, so that it never
    covers the tail of the distribution. When drawing on axes in a grid of
    subplots, leave room for it or move it with ``ax.legend(...)``.

    Examples
    --------
    >>> plot_null_distribution(mc, "delta_chi2", observed=21.3,
    ...                        fap=1.35e-3)  # doctest: +SKIP
    """
    from scipy.stats import norm

    label = _stat_label(stat)
    null = np.asarray(mc.null[stat], dtype=float)
    n = null.size
    if n == 0:
        raise ValueError("The DetectionMC has no null draws.")
    x = np.sort(np.where(np.isnan(null), -np.inf, null))
    exceed = (n - np.arange(n)) / n  # P(null >= x[i])
    finite = np.isfinite(x)

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
    right = [x[finite].max() if np.any(finite) else 1.0]
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

    left = min(0.0, x[finite].min()) if np.any(finite) else 0.0
    xmax = 1.1 * max(right) if max(right) > 0 else 1.0
    if reference and stat != "log_bayes_factor":
        grid = np.linspace(max(left, 0.0), xmax, 400)
        if stat == "delta_chi2":
            ref = norm.sf(np.sqrt(grid))  # = ½ P(χ²₁ ≥ x)
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
    import jax.numpy as jnp

    batched = jax.tree_util.tree_map(jnp.atleast_1d, orbits)
    phase = jnp.linspace(0.0, 1.0, n_points)

    def track(orbit):
        # Times relative to the orbit's t_ref, from one periastron to the
        # next, so the tracks need no absolute (float32-unsafe) MJDs.
        dra, ddec, _ = orbit._relative(orbit.dt_peri + orbit.period * phase)
        return dra, ddec

    dra, ddec = jax.vmap(track)(batched)
    return np.asarray(dra), np.asarray(ddec)


def _position_covariances(positions):
    """The ``(n, 2, 2)`` covariances of a ``PositionData``'s positions."""
    chol = np.linalg.inv(np.asarray(positions.whitener, dtype=float))
    return chol @ np.swapaxes(chol, -1, -2)


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

    East is to the left and North up, and the primary sits at the origin.
    Each orbit is drawn over one full period as a thin line; the measured
    positions are coloured by time, with their error ellipses.

    Parameters
    ----------
    orbits : KeplerOrbit
        One orbit, or many as a single
        [`KeplerOrbit`][virgil.orbits.KeplerOrbit] whose elements are 1-D
        arrays of equal length (e.g. built from posterior samples).
    positions : PositionData, optional
        Measured positions to overlay, with ``n_sigma`` error ellipses.
    truth : KeplerOrbit, optional
        A reference orbit (e.g. the truth of a simulation), drawn as a
        thicker contrasting line.
    n_points : int, optional
        Points along each orbit.
    n_sigma : float, optional
        Size of the error ellipses, in standard deviations.
    color, truth_color : optional
        Colours of the ensemble and of the reference orbit.
    alpha : float, optional
        Opacity of each ensemble line; by default it falls with the number
        of orbits, so that the density of lines reads as probability.
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
        alpha = float(np.clip(8.0 / len(dra), 0.03, 0.8))
    ax.add_collection(
        LineCollection(
            np.stack([dra, ddec], -1),
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
        mjd = positions.t_ref + np.asarray(positions.dt, dtype=float)
        x = np.asarray(positions.dra, dtype=float)
        y = np.asarray(positions.ddec, dtype=float)
        norm = matplotlib.colors.Normalize(
            mjd.min(), max(mjd.max(), mjd.min() + 1)
        )
        colours = matplotlib.colormaps[cmap](norm(mjd))
        for xi, yi, cov, c in zip(
            x, y, _position_covariances(positions), colours
        ):
            values, vectors = np.linalg.eigh(cov)
            angle = np.degrees(np.arctan2(vectors[1, -1], vectors[0, -1]))
            width, height = 2 * n_sigma * np.sqrt(np.maximum(values[::-1], 0))
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
    uu = np.asarray(oidata.u / oidata.wavel) / 1e6
    vv = np.asarray(oidata.v / oidata.wavel) / 1e6
    uu, vv = np.broadcast_arrays(uu, vv)
    baseline = np.hypot(uu, vv)
    fig, (ax_uv, ax_vis, ax_phi) = plt.subplots(1, 3, figsize=figsize)

    ax_uv.scatter(uu, vv, s=12, c="C0")
    ax_uv.scatter(-uu, -vv, s=12, c="C1")
    ax_uv.set(xlabel="u (Mλ)", ylabel="v (Mλ)", title="uv coverage")
    ax_uv.set_aspect("equal")
    ax_uv.invert_xaxis()  # East to the left

    vis = np.asarray(oidata.vis)
    d_vis = np.asarray(oidata.d_vis)
    if oidata.vis_mat is None and vis.size:
        index = (
            np.arange(baseline.size)
            if oidata.vis_index is None
            else np.asarray(oidata.vis_index)
        )
        ax_vis.errorbar(
            baseline[index], vis, yerr=d_vis, fmt=".", ms=6, elinewidth=0.6
        )
        name = {"v2": "V²", "amp": "|V|", "logamp": "log |V|"}
        ax_vis.set_ylabel(name.get(oidata.vis_mode, "visibility"))
        ax_vis.set_xlabel("Baseline (Mλ)")
    else:
        ax_vis.errorbar(np.arange(vis.size), vis, yerr=d_vis, fmt=".")
        ax_vis.set(xlabel="Observable index", ylabel="Projected visibility")
    ax_vis.set_title("Visibilities")

    phi = np.rad2deg(np.asarray(oidata.phi))
    d_phi = np.rad2deg(np.asarray(oidata.d_phi))
    if oidata.phi_mat is None and phi.size:
        if oidata.cp_flag:
            legs = [np.asarray(i) for i in (oidata.i_cps1, oidata.i_cps2)]
            legs.append(np.asarray(oidata.i_cps3))
            x = np.max([baseline[leg] for leg in legs], axis=0)
            ax_phi.set_xlabel("Longest baseline (Mλ)")
            ax_phi.set_ylabel("Closure phase (deg)")
        else:
            index = (
                np.arange(baseline.size)
                if oidata.phi_index is None
                else np.asarray(oidata.phi_index)
            )
            x = baseline[index]
            ax_phi.set_xlabel("Baseline (Mλ)")
            ax_phi.set_ylabel("Phase (deg)")
        ax_phi.errorbar(x, phi, yerr=d_phi, fmt=".", ms=6, elinewidth=0.6)
    else:
        ax_phi.errorbar(np.arange(phi.size), phi, yerr=d_phi, fmt=".")
        ax_phi.set(xlabel="Observable index", ylabel="Projected phase")
    ax_phi.set_title("Phases")
    fig.tight_layout()
    return fig, (ax_uv, ax_vis, ax_phi)
