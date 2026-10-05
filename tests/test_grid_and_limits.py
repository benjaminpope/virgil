import warnings

import jax.numpy as np
import numpy as onp
from matplotlib import get_backend
import matplotlib.pyplot as plt
import pytest

from virgil.grid_fit import (
    laplace_flux_uncertainty_grid,
    likelihood_grid,
    optimized_flux_grid,
    optimized_likelihood_grid,
)
from virgil.limits import (
    absil_limits,
    delta_mag_to_flux,
    flux_to_contrast,
    flux_to_delta_mag,
    nsigma,
    radial_profile,
    ruffio_upperlimit,
)
from virgil.models import BinaryModelCartesian
from virgil.oidata import OIData
from virgil.plotting import (
    diagnostics_table_from_samples,
    plot_contrast_curve,
    plot_grid_map,
    truth_cartesian_and_polar,
)
from tests._test_data import (
    oidata,
    oidata_sim,
    perc,
    samples_dict,
    true_values,
)

curr_backend = get_backend()
plt.switch_backend("Agg")
warnings.filterwarnings("ignore", "Matplotlib is currently using agg")


def _assert_sky_oriented(fig):
    """Every imshow'd Axes in ``fig`` must show dra increasing toward the
    left (East) and ddec increasing toward the top (North), per the
    package's coordinate convention (see AGENTS.md).
    """
    image_axes = [ax for ax in fig.axes if ax.images]
    assert image_axes, "expected at least one image axes in this figure"
    for ax in image_axes:
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        assert xlim[0] > xlim[1], f"x-axis not East-left: {xlim}"
        assert ylim[0] < ylim[1], f"y-axis not North-up: {ylim}"


def test_likelihood_grid():
    loglike_im = likelihood_grid(oidata, BinaryModelCartesian, samples_dict)
    assert np.all(np.isfinite(loglike_im))
    assert loglike_im.shape == (
        samples_dict["dra"].shape[0],
        samples_dict["ddec"].shape[0],
        samples_dict["flux"].shape[0],
    )

    # The full cube (with its flux axis) is reduced to its maximum over flux.
    fig, ax = plot_grid_map(loglike_im, samples_dict, truth=true_values)
    _assert_sky_oriented(fig)
    reduced, _ = plot_grid_map(loglike_im.max(axis=2), samples_dict)
    assert onp.allclose(
        onp.ma.getdata(ax.images[0].get_array()),
        onp.ma.getdata(reduced.axes[0].images[0].get_array()),
    )


def test_likelihood_grid_axis_order_tracks_key_order():
    reduced_samples = {
        "dra": samples_dict["dra"][::12],
        "ddec": samples_dict["ddec"][::12],
        "flux": samples_dict["flux"][::12],
    }
    ordered = likelihood_grid(oidata, BinaryModelCartesian, reduced_samples)

    permuted_samples = {
        "flux": reduced_samples["flux"],
        "dra": reduced_samples["dra"],
        "ddec": reduced_samples["ddec"],
    }
    permuted = likelihood_grid(oidata, BinaryModelCartesian, permuted_samples)

    assert ordered.shape == (
        reduced_samples["dra"].shape[0],
        reduced_samples["ddec"].shape[0],
        reduced_samples["flux"].shape[0],
    )
    assert permuted.shape == (
        reduced_samples["flux"].shape[0],
        reduced_samples["dra"].shape[0],
        reduced_samples["ddec"].shape[0],
    )
    assert np.allclose(ordered, np.transpose(permuted, (1, 2, 0)))


def test_optimized_likelihood_grid():
    loglike_im = optimized_likelihood_grid(
        oidata, BinaryModelCartesian, samples_dict, flux_param="flux"
    )
    assert np.all(np.isfinite(loglike_im))
    assert loglike_im.shape == (
        samples_dict["dra"].shape[0],
        samples_dict["ddec"].shape[0],
    )
    fig, ax = plot_grid_map(loglike_im, samples_dict, truth=true_values)
    _assert_sky_oriented(fig)


def test_optimized_likelihood_grid_axis_order_tracks_key_order():
    reduced_samples = {
        "dra": samples_dict["dra"][::12],
        "ddec": samples_dict["ddec"][::12],
        "flux": samples_dict["flux"][::12],
    }
    ordered = optimized_likelihood_grid(
        oidata, BinaryModelCartesian, reduced_samples, flux_param="flux"
    )

    permuted_samples = {
        "ddec": reduced_samples["ddec"],
        "flux": reduced_samples["flux"],
        "dra": reduced_samples["dra"],
    }
    permuted = optimized_likelihood_grid(
        oidata, BinaryModelCartesian, permuted_samples, flux_param="flux"
    )

    assert ordered.shape == (
        reduced_samples["dra"].shape[0],
        reduced_samples["ddec"].shape[0],
    )
    assert permuted.shape == (
        reduced_samples["ddec"].shape[0],
        reduced_samples["dra"].shape[0],
    )
    assert np.allclose(ordered, np.transpose(permuted, (1, 0)))


def test_optimized():
    optimized = optimized_flux_grid(
        oidata_sim, BinaryModelCartesian, samples_dict
    )
    assert optimized.shape == (
        samples_dict["dra"].shape[0],
        samples_dict["ddec"].shape[0],
    )
    assert np.all(np.isfinite(optimized))
    fig, _ = plot_grid_map(optimized, samples_dict, kind="flux")
    _assert_sky_oriented(fig)


def test_optimized_flux_grid_axis_order_tracks_key_order():
    reduced_samples = {
        "dra": samples_dict["dra"][::12],
        "ddec": samples_dict["ddec"][::12],
        "flux": samples_dict["flux"][::12],
    }
    ordered = optimized_flux_grid(
        oidata_sim, BinaryModelCartesian, reduced_samples
    )

    permuted_samples = {
        "ddec": reduced_samples["ddec"],
        "flux": reduced_samples["flux"],
        "dra": reduced_samples["dra"],
    }
    permuted = optimized_flux_grid(
        oidata_sim, BinaryModelCartesian, permuted_samples
    )

    assert ordered.shape == (
        reduced_samples["dra"].shape[0],
        reduced_samples["ddec"].shape[0],
    )
    assert permuted.shape == (
        reduced_samples["ddec"].shape[0],
        reduced_samples["dra"].shape[0],
    )
    assert np.allclose(ordered, np.transpose(permuted, (1, 0)))


def test_laplace():
    optimized = optimized_flux_grid(
        oidata_sim, BinaryModelCartesian, samples_dict
    )
    laplace_sigma_grid = laplace_flux_uncertainty_grid(
        oidata_sim, BinaryModelCartesian, samples_dict, flux=optimized
    )
    assert laplace_sigma_grid.shape == (
        samples_dict["dra"].shape[0],
        samples_dict["ddec"].shape[0],
    )
    assert np.all(np.isfinite(laplace_sigma_grid))
    # By default the curvature is taken at the optimized flux.
    small = {key: value[::6] for key, value in samples_dict.items()}
    assert np.allclose(
        laplace_flux_uncertainty_grid(oidata_sim, BinaryModelCartesian, small),
        laplace_flux_uncertainty_grid(
            oidata_sim,
            BinaryModelCartesian,
            small,
            flux=optimized_flux_grid(oidata_sim, BinaryModelCartesian, small),
        ),
    )

    fig, (a, b) = plt.subplots(1, 2)
    plot_grid_map(laplace_sigma_grid, samples_dict, kind="sigma", ax=a)
    plot_grid_map(
        optimized / laplace_sigma_grid, samples_dict, kind="snr", ax=b
    )
    _assert_sky_oriented(fig)
    assert a.get_title() == "σ(flux)" and b.get_title() == "S/N"


def test_laplace_grid_axis_order_tracks_key_order():
    reduced_samples = {
        "dra": samples_dict["dra"][::12],
        "ddec": samples_dict["ddec"][::12],
        "flux": samples_dict["flux"][::12],
    }
    ordered = laplace_flux_uncertainty_grid(
        oidata_sim, BinaryModelCartesian, reduced_samples
    )
    permuted_samples = {
        "ddec": reduced_samples["ddec"],
        "flux": reduced_samples["flux"],
        "dra": reduced_samples["dra"],
    }
    permuted = laplace_flux_uncertainty_grid(
        oidata_sim, BinaryModelCartesian, permuted_samples
    )
    assert permuted.shape == (
        reduced_samples["ddec"].shape[0],
        reduced_samples["dra"].shape[0],
    )
    assert np.allclose(ordered, np.transpose(permuted, (1, 0)))


def test_ruffio():
    optimized = optimized_flux_grid(
        oidata_sim, BinaryModelCartesian, samples_dict
    )
    sigma = laplace_flux_uncertainty_grid(
        oidata_sim, BinaryModelCartesian, samples_dict, flux=optimized
    )
    limits = ruffio_upperlimit(optimized, sigma, perc[0])
    assert limits.shape == optimized.shape
    assert np.all(np.isfinite(limits))

    profile = radial_profile(limits, samples_dict["dra"], samples_dict["ddec"])
    assert np.all(np.isfinite(profile["median"][profile["count"] > 0]))

    fig, (a, b) = plt.subplots(1, 2)
    plot_grid_map(
        limits,
        samples_dict,
        kind="limit",
        units="delta_mag",
        percentile=perc,
        ax=a,
    )
    plot_contrast_curve(
        limits, samples_dict, percentile=perc, truth=true_values, ax=b
    )
    _assert_sky_oriented(fig)
    assert a.get_title() == "97.7% upper limit (Δmag)"
    assert b.get_legend().texts[0].get_text() == "97.7% upper limit"


def test_absil():
    limits_absil = absil_limits(
        oidata_sim, BinaryModelCartesian, samples_dict, 5.0
    )
    assert np.all(np.isfinite(limits_absil))
    fig, ax = plot_grid_map(
        limits_absil, samples_dict, kind="limit", units="contrast", sigma=5.0
    )
    _assert_sky_oriented(fig)
    assert ax.get_title() == "5$\\sigma$ limit (contrast)"
    # Contrast is primary/companion: the displayed values are 1/flux.
    assert np.allclose(
        onp.nanmax(ax.images[0].get_array()),
        onp.nanmax(1.0 / onp.asarray(limits_absil)),
        rtol=1e-5,
    )


def test_contrast_units_follow_astronomical_convention():
    # A companion 100 times fainter than the primary has contrast 100, 5 mag.
    assert np.allclose(flux_to_contrast(0.01), 100.0)
    assert np.allclose(flux_to_delta_mag(0.01), 5.0)
    assert np.allclose(delta_mag_to_flux(5.0), 0.01)
    fig, ax = plot_contrast_curve(
        onp.full((5, 5), 0.01),
        {"dra": onp.linspace(-10, 10, 5), "ddec": onp.linspace(-10, 10, 5)},
        units="delta_mag",
    )
    assert np.allclose(onp.nanmedian(ax.lines[0].get_ydata()), 5.0)
    assert ax.yaxis_inverted()  # deeper limits drawn lower down
    plt.close("all")


def test_nsigma_increases_with_chi2_ratio():
    significances = nsigma(np.array([1.0, 2.0, 4.0]), 1.0, 56)

    assert np.all(np.diff(significances) > 0.0)


def test_absil_limit_responds_to_smaller_uncertainties():
    samples = {
        "dra": np.array([100.0]),
        "ddec": np.array([100.0]),
        "flux": 10 ** np.linspace(-6.0, -1.0, 30),
    }
    null_cvis = np.ones_like(oidata_sim.u, dtype=complex)
    null_vis = oidata_sim.to_vis(null_cvis)
    null_phi = oidata_sim.to_phases(null_cvis)
    vis_noise = np.linspace(-1.0, 1.0, oidata_sim.vis.size)
    phi_noise = np.linspace(1.0, -1.0, oidata_sim.phi.size)

    def noisy_null(error_scale):
        return OIData(
            {
                "u": oidata_sim.u,
                "v": oidata_sim.v,
                "wavel": oidata_sim.wavel,
                "vis": null_vis + vis_noise * oidata_sim.d_vis * error_scale,
                "d_vis": oidata_sim.d_vis * error_scale,
                "phi": null_phi + phi_noise * oidata_sim.d_phi * error_scale,
                "d_phi": oidata_sim.d_phi * error_scale,
                "i_cps1": oidata_sim.i_cps1,
                "i_cps2": oidata_sim.i_cps2,
                "i_cps3": oidata_sim.i_cps3,
                "v2_flag": oidata_sim.v2_flag,
                "cp_flag": oidata_sim.cp_flag,
            }
        )

    nominal = absil_limits(noisy_null(1.0), BinaryModelCartesian, samples, 2.0)
    improved = absil_limits(
        noisy_null(0.1), BinaryModelCartesian, samples, 2.0
    )

    assert improved.item() < nominal.item()


def test_absil_limit_does_not_depend_on_the_starting_flux_axis():
    # A bright starting flux sits where the significance saturates, so the
    # loss is flat; a single-value axis must still find the same limit.
    null_cvis = np.ones_like(oidata_sim.u, dtype=complex)
    data = OIData(
        {
            "u": oidata_sim.u,
            "v": oidata_sim.v,
            "wavel": oidata_sim.wavel,
            "vis": oidata_sim.to_vis(null_cvis)
            + np.linspace(-1.0, 1.0, oidata_sim.vis.size) * oidata_sim.d_vis,
            "d_vis": oidata_sim.d_vis,
            "phi": oidata_sim.to_phases(null_cvis)
            + np.linspace(1.0, -1.0, oidata_sim.phi.size) * oidata_sim.d_phi,
            "d_phi": oidata_sim.d_phi,
            "i_cps1": oidata_sim.i_cps1,
            "i_cps2": oidata_sim.i_cps2,
            "i_cps3": oidata_sim.i_cps3,
            "v2_flag": oidata_sim.v2_flag,
            "cp_flag": oidata_sim.cp_flag,
        }
    )

    def limit(flux_axis):
        samples = {
            "dra": np.array([100.0]),
            "ddec": np.array([100.0]),
            "flux": np.asarray(flux_axis),
        }
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            return absil_limits(
                data, BinaryModelCartesian, samples, 2.0, flux_bounds=None
            ).item()

    reference = limit(10 ** np.linspace(-6.0, -1.0, 30))
    for axis in ([1e-2], [0.5], [1e-6]):
        assert limit(axis) == pytest.approx(reference, rel=1e-3)


def test_diagnostics_table_from_samples_follows_north_to_east_pa_convention():
    """A sample due East (dra=+40, ddec=0) must report pa=90 under the
    package's North-to-East convention. The previous swapped-argument
    arctan2(ddec, dra) bug would instead report pa=0 here, so this catches
    the bug directly rather than via a self-consistent round-trip.
    """
    samples = {
        "dra": np.array([40.0]),
        "ddec": np.array([0.0]),
        "flux": np.array([1.0]),
    }
    df = diagnostics_table_from_samples(samples)
    assert df["pa"].to_numpy() == pytest.approx(90.0)

    dra, ddec = 50.0, 50.0
    off_axis = diagnostics_table_from_samples(
        {
            "dra": np.array([dra]),
            "ddec": np.array([ddec]),
            "flux": np.array([1.0]),
        }
    )
    expected_pa = float(BinaryModelCartesian(dra, ddec, 1.0).to_angular().pa)
    assert off_axis["pa"].to_numpy() == pytest.approx(expected_pa)


def test_truth_cartesian_and_polar_follows_north_to_east_pa_convention():
    """Same North-to-East check as above, for the truth-marker helper used
    by plot_hmc_fisher_chainconsumer.
    """
    truth = {"dra": 40.0, "ddec": 0.0, "flux": 1.0}
    _, truth_polar = truth_cartesian_and_polar(truth)
    assert truth_polar["pa"] == pytest.approx(90.0)


def test_ruffio_matches_truncated_gaussian_even_far_below_zero():
    from scipy.stats import norm, truncnorm

    sigma = 1e-3
    means = onp.array([2.0, 0.0, -1.0, -3.0, -5.0, -8.0, -20.0, -60.0]) * sigma
    percentiles = onp.array([0.16, 0.5, norm.cdf(2.0), 0.99])
    limits = onp.asarray(
        ruffio_upperlimit(
            np.array(means), np.full(means.size, sigma), np.array(percentiles)
        )
    )
    expected = onp.array(
        [
            truncnorm.ppf(percentiles, -m / sigma, onp.inf, loc=m, scale=sigma)
            for m in means
        ]
    )
    assert onp.all(onp.isfinite(limits))
    assert onp.all(limits >= 0.0)
    assert onp.all(onp.diff(limits, axis=1) > 0.0)
    onp.testing.assert_allclose(limits, expected, rtol=2e-3)
