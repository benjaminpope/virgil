"""Plots of simulated companion searches (virgil.plotting, detection part).

Every test builds a small ``DetectionMC`` by hand from NumPy arrays, so no
search runs and no JAX is compiled.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as onp  # noqa: E402
import pytest  # noqa: E402
from scipy import stats as sstats  # noqa: E402

from virgil.detection import DetectionMC  # noqa: E402
from virgil.limits import flux_to_delta_mag  # noqa: E402
from virgil.plotting import (  # noqa: E402
    plot_completeness,
    plot_contrast_curve,
    plot_null_distribution,
    plot_roc,
)

SEPS = onp.array([40.0, 60.0, 80.0])
FLUXES = onp.array([1e-3, 2e-3, 4e-3, 8e-3, 1.6e-2])
N_PA = 40


def _mc(n_null=2000, seed=0):
    """Null Δχ² as the best of ~30 independent ½δ₀ + ½χ²₁ positions;
    injected Δχ² = (SNR + noise)², with SNR ∝ flux and rising with
    separation. The other statistics are monotone functions of Δχ²."""
    rng = onp.random.default_rng(seed)
    z = rng.standard_normal((n_null, 30))
    null_d = onp.max(onp.maximum(z, 0.0) ** 2, axis=1)
    sep = onp.repeat(SEPS, FLUXES.size * N_PA)
    flux = onp.tile(onp.repeat(FLUXES, N_PA), SEPS.size)
    pa = rng.uniform(0.0, 2.0 * onp.pi, sep.size)
    snr = flux / 1e-3 * (sep / 80.0)
    inj_d = onp.maximum(snr + rng.standard_normal(sep.size), 0.0) ** 2

    def statistics(d):
        return {
            "delta_chi2": d,
            "log_bayes_factor": 0.4 * d - 3.0,
            "max_snr": onp.sqrt(d),
        }

    injected = statistics(inj_d) | {
        "dra": sep * onp.sin(pa),
        "ddec": sep * onp.cos(pa),
        "flux": flux,
    }
    meta = {"match_radius": None, "seeds": [{"seed": seed}]}
    return DetectionMC(null=statistics(null_d), injected=injected, meta=meta)


@pytest.fixture(scope="module")
def mc():
    return _mc()


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _labelled(ax):
    return {line.get_label(): line for line in ax.get_lines()}


def test_plot_roc_draws_one_distinct_curve_per_statistic_and_flux(mc):
    fig, ax = plot_roc(
        mc, ["delta_chi2", "log_bayes_factor"], flux=[2e-3, 4e-3]
    )
    curves = [
        line
        for line in ax.get_lines()
        if line.get_label().startswith(("$\\Delta", "$\\log"))
    ]
    assert len(curves) == 4
    styles = {(c.get_color(), c.get_linestyle()) for c in curves}
    assert len(styles) == 4
    assert ax.get_xscale() == "log"
    assert "False-positive rate" in ax.get_xlabel()
    assert "completeness" in ax.get_ylabel()
    # Each curve is DetectionMC.roc for its selection.
    fpr, tpr, _ = mc.roc("delta_chi2", flux=4e-3)
    line = [
        c
        for c in curves
        if "Delta" in c.get_label() and "0.004" in c.get_label()
    ][0]
    onp.testing.assert_allclose(line.get_xdata(), fpr)
    onp.testing.assert_allclose(line.get_ydata(), tpr)


def test_plot_roc_one_statistic_varies_style_by_selection(mc):
    _, ax = plot_roc(mc, "delta_chi2", sep_bin=[(30, 50), (50, 70)])
    curves = [line for line in ax.get_lines() if "mas" in line.get_label()]
    assert len(curves) == 2
    assert curves[0].get_linestyle() != curves[1].get_linestyle()
    assert ax.get_title() == "ROC of $\\Delta\\chi^2$"


def test_plot_roc_names_a_shared_selection_in_the_title(mc):
    _, ax = plot_roc(mc, ["delta_chi2", "max_snr"], flux=4e-3)
    assert ax.get_title() == "ROC, flux 0.004 (Δmag 5.99)"
    labels = [line.get_label() for line in ax.get_lines()]
    assert "$\\Delta\\chi^2$" in labels and "max SNR" in labels


@pytest.mark.parametrize("log_fpr", [True, False])
def test_plot_roc_chance_curve_is_y_equals_x(mc, log_fpr):
    _, ax = plot_roc(mc, "max_snr", log_fpr=log_fpr, mark_wilks=None)
    chance = _labelled(ax)["chance (TPR = FPR)"]
    x, y = onp.asarray(chance.get_xdata()), onp.asarray(chance.get_ydata())
    onp.testing.assert_array_equal(x, y)
    # Drawn as a curve of many points spanning the whole axis, so it stays
    # y = x on a log axis rather than a straight segment between its ends.
    assert x.size >= 100
    assert onp.isclose(x[-1], 1.0)
    if log_fpr:
        assert onp.isclose(x[0], 0.5 / mc.n_null)
        onp.testing.assert_allclose(onp.diff(onp.log(x)), onp.log(x[1] / x[0]))
    else:
        assert x[0] == 0.0
        assert ax.get_xscale() == "linear"


def test_plot_roc_marks_the_wilks_faps_and_local_thresholds(mc):
    _, ax = plot_roc(mc, "delta_chi2", flux=4e-3, mark_wilks=(3.0,))
    nominal = sstats.norm.sf(3.0)
    vertical = [
        line
        for line in ax.get_lines()
        if line.get_label().startswith("local 3σ")
    ]
    assert len(vertical) == 1
    onp.testing.assert_allclose(vertical[0].get_xdata(), nominal)
    assert "0.135%" in vertical[0].get_label()
    # The circle sits at the rates of the threshold Δχ² = 9.
    fpr = onp.mean(mc.null["delta_chi2"] >= 9.0)
    sel = onp.isclose(mc.injected["flux"], 4e-3)
    tpr = onp.mean(mc.injected["delta_chi2"][sel] >= 9.0)
    circles = [
        line
        for line in ax.get_lines()
        if line.get_marker() == "o" and len(line.get_xdata()) == 1
    ]
    assert len(circles) == 1
    assert onp.isclose(circles[0].get_xdata()[0], fpr)
    assert onp.isclose(circles[0].get_ydata()[0], tpr)
    # The look-elsewhere effect: over the grid, Δχ² = 9 is far more
    # frequent than its local FAP.
    assert fpr > 10 * nominal


def test_plot_roc_log_bayes_factor_has_no_local_threshold(mc):
    _, ax = plot_roc(mc, "log_bayes_factor")
    singles = [
        line
        for line in ax.get_lines()
        if line.get_marker() in ("o", "s") and len(line.get_xdata()) == 1
    ]
    assert not singles


def test_plot_completeness_map_and_contours(mc):
    fap = 0.01
    fig, ax = plot_completeness(mc, "delta_chi2", fap)
    result = mc.completeness("delta_chi2", fap)
    mesh = ax.collections[0]
    onp.testing.assert_allclose(
        onp.asarray(mesh.get_array()).reshape(FLUXES.size, SEPS.size),
        result["completeness"].T,
    )
    lines = _labelled(ax)
    assert {"50% completeness", "90% completeness"} <= set(lines)
    for level in (0.5, 0.9):
        sep, flux = mc.contrast_curve("delta_chi2", fap, level)
        line = lines[f"{100 * level:g}% completeness"]
        onp.testing.assert_allclose(line.get_xdata(), sep)
        onp.testing.assert_allclose(line.get_ydata(), flux_to_delta_mag(flux))
        assert onp.all(onp.isfinite(line.get_ydata()))
    # 90% needs a brighter companion (smaller Δmag) than 50%.
    assert onp.all(
        lines["90% completeness"].get_ydata()
        < lines["50% completeness"].get_ydata()
    )
    assert ax.yaxis_inverted()  # fainter lower down
    assert ax.get_xlabel() == "Separation (mas)"
    assert "Δmag" in ax.get_ylabel()
    assert len(fig.axes) == 2  # the colour bar


def test_plot_completeness_takes_contrast_curves_on_the_same_axes(mc):
    _, ax = plot_completeness(mc, "delta_chi2", 0.01)
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    grid = {
        "dra": onp.linspace(-100, 100, 21),
        "ddec": onp.linspace(-100, 100, 21),
    }
    plot_contrast_curve(
        onp.full((21, 21), 4e-3), grid, label="Absil", band=False, ax=ax
    )
    assert ax.yaxis_inverted()
    assert ax.get_xlim() == xlim and ax.get_ylim() == ylim
    absil = _labelled(ax)["Absil"]
    onp.testing.assert_allclose(
        onp.nanmedian(absil.get_ydata()), flux_to_delta_mag(4e-3)
    )


@pytest.mark.parametrize("units", ["flux", "contrast"])
def test_plot_completeness_log_units(mc, units):
    _, ax = plot_completeness(mc, "max_snr", 0.01, units=units, contours=())
    assert ax.get_yscale() == "log"
    assert ax.yaxis_inverted() == (units == "contrast")
    assert ax.get_legend() is None


def test_plot_completeness_rejects_unknown_units(mc):
    with pytest.raises(ValueError, match="units"):
        plot_completeness(mc, "delta_chi2", 0.01, units="mag")


def test_plot_null_distribution_curves(mc):
    fap = 0.01
    observed = 12.0
    fig, ax = plot_null_distribution(
        mc, "delta_chi2", observed=observed, fap=fap
    )
    lines = _labelled(ax)
    n = mc.n_null
    empirical = lines[f"grid search, {n} null simulations"]
    x = onp.sort(mc.null["delta_chi2"])
    onp.testing.assert_allclose(empirical.get_xdata(), x)
    onp.testing.assert_allclose(empirical.get_ydata(), (n - onp.arange(n)) / n)
    reference = [line for line in lines if "fixed in advance" in line][0]
    xs = lines[reference].get_xdata()
    onp.testing.assert_allclose(
        lines[reference].get_ydata(),
        0.5 * sstats.chi2(1).sf(xs),
        rtol=1e-6,
        atol=1e-300,
    )
    assert ax.get_yscale() == "log"
    assert ax.get_ylim()[0] <= 0.5 / n
    onp.testing.assert_allclose(lines["FAP 1%"].get_ydata(), fap)
    threshold = mc.threshold("delta_chi2", fap, n_boot=0)[0]
    assert any(
        onp.allclose(line.get_xdata(), threshold) for line in ax.get_lines()
    )
    texts = [t.get_text() for t in ax.get_legend().get_texts()]
    p = mc.false_alarm_probability("delta_chi2", observed)[0]
    assert any(
        t.startswith("observed") and f"{100 * p:.3g}%" in t for t in texts
    )
    # The legend sits to the right of the axes, clear of the tail.
    fig.canvas.draw()
    legend_box = ax.get_legend().get_window_extent()
    assert legend_box.x0 >= ax.get_window_extent().x1


def test_plot_null_distribution_reference_by_statistic(mc):
    _, ax = plot_null_distribution(mc, "max_snr")
    reference = [
        line
        for line in ax.get_lines()
        if "fixed in advance" in line.get_label()
    ][0]
    onp.testing.assert_allclose(
        reference.get_ydata(), sstats.norm.sf(reference.get_xdata())
    )
    _, ax = plot_null_distribution(mc, "log_bayes_factor")
    assert not [
        line
        for line in ax.get_lines()
        if "fixed in advance" in line.get_label()
    ]


def test_plot_null_distribution_shows_a_fap_beyond_every_null_draw(mc):
    # An observed value above every null draw: FAP 1/(n + 1), on the axis.
    observed = 2.0 * mc.null["delta_chi2"].max()
    with pytest.warns(RuntimeWarning, match="cannot resolve"):
        _, ax = plot_null_distribution(mc, observed=observed, fap=1e-4)
    p, low, high = mc.false_alarm_probability("delta_chi2", observed)
    assert onp.isclose(p, 1.0 / (mc.n_null + 1))
    assert ax.get_ylim()[0] <= 0.5e-4
    assert ax.get_xlim()[1] > observed
