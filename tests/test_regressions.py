import jax
import jax.numpy as np
import matplotlib.pyplot as plt
import numpy as onp
import pytest
from astropy.io import fits
from matplotlib.ticker import FuncFormatter
from scipy import stats

from virgil.limits import chi2ppf
from virgil.models import GaussianDisk
from virgil.oidata import OIData, closure_phases
from virgil.plotting import (
    plot_grid_map,
    plot_data_model_correlation,
    plot_model,
    plot_residual_map,
)


def _base_dict(cp_flag=False, i_cps1=None, i_cps2=None, i_cps3=None):
    return {
        "u": np.array([0.0, 0.0, 0.0]),
        "v": np.array([0.0, 0.0, 0.0]),
        "wavel": np.array([4.8e-6]),
        "vis": np.array([1.0, 1.0, 1.0]),
        "d_vis": np.array([0.1, 0.1, 0.1]),
        "phi": np.array([0.0, 0.0, 0.0]),
        "d_phi": np.array([1.0, 1.0, 1.0]),
        "v2_flag": True,
        "cp_flag": cp_flag,
        "i_cps1": i_cps1,
        "i_cps2": i_cps2,
        "i_cps3": i_cps3,
    }


def _make_fake_oifits_phase_input(phase_ext, phi, d_phi, unit):
    wavelength_hdu = fits.BinTableHDU.from_columns(
        [
            fits.Column(
                name="EFF_WAVE",
                format="1E",
                array=onp.array([4.8e-6], dtype=onp.float32),
            )
        ]
    )
    wavelength_hdu.name = "OI_WAVELENGTH"

    vis2_hdu = fits.BinTableHDU.from_columns(
        [
            fits.Column(
                name="VIS2DATA",
                format="1E",
                array=onp.array([1.0, 0.9, 0.8], dtype=onp.float32),
            ),
            fits.Column(
                name="VIS2ERR",
                format="1E",
                array=onp.array([0.01, 0.01, 0.01], dtype=onp.float32),
            ),
            fits.Column(
                name="UCOORD",
                format="1D",
                array=onp.array([0.0, 1.0, -1.0], dtype=float),
            ),
            fits.Column(
                name="VCOORD",
                format="1D",
                array=onp.array([0.0, 1.0, 1.0], dtype=float),
            ),
            fits.Column(
                name="STA_INDEX",
                format="2I",
                array=onp.array([[1, 2], [2, 3], [1, 3]], dtype=onp.int16),
            ),
        ]
    )
    vis2_hdu.name = "OI_VIS2"

    if phase_ext == "OI_VIS":
        # Absolute phases: an OI_VIS table on the first len(phi) baselines.
        n = len(phi)
        phase_columns = [
            fits.Column(name="VISAMP", format="1D", array=onp.ones(n)),
            fits.Column(
                name="VISAMPERR", format="1D", array=onp.full(n, 0.01)
            ),
            fits.Column(
                name="VISPHI",
                format="1D",
                unit=unit,
                array=onp.asarray(phi, dtype=float),
            ),
            fits.Column(
                name="VISPHIERR",
                format="1D",
                unit=unit,
                array=onp.asarray(d_phi, dtype=float),
            ),
            fits.Column(
                name="UCOORD", format="1D", array=onp.array([0.0, 1.0])[:n]
            ),
            fits.Column(
                name="VCOORD", format="1D", array=onp.array([0.0, 1.0])[:n]
            ),
            fits.Column(
                name="STA_INDEX",
                format="2I",
                array=onp.array([[1, 2], [2, 3]], dtype=onp.int16)[:n],
            ),
        ]
    else:
        phase_columns = [
            fits.Column(
                name="T3PHI",
                format="1D",
                unit=unit,
                array=onp.asarray(phi, dtype=float),
            ),
            fits.Column(
                name="T3PHIERR",
                format="1D",
                unit=unit,
                array=onp.asarray(d_phi, dtype=float),
            ),
            fits.Column(
                name="STA_INDEX",
                format="3I",
                array=onp.array([[1, 2, 3]], dtype=onp.int16),
            ),
        ]

    phase_hdu = fits.BinTableHDU.from_columns(phase_columns)
    phase_hdu.name = phase_ext

    return fits.HDUList(
        [fits.PrimaryHDU(), wavelength_hdu, vis2_hdu, phase_hdu]
    )


def test_to_phases_absolute_returns_radians():
    data = OIData(
        _base_dict(cp_flag=False, i_cps1=None, i_cps2=None, i_cps3=None)
    )
    cvis = np.array([1.0 + 0.0j, 0.0 + 1.0j])
    phases = data.to_phases(cvis)
    assert np.allclose(phases, np.array([0.0, np.pi / 2.0]))


def test_closure_phases_radian_convention():
    cvis = np.exp(1j * np.deg2rad(np.array([10.0, 30.0, 25.0])))
    cps = closure_phases(
        cvis,
        np.array([0]),
        np.array([1]),
        np.array([2]),
    )
    assert np.allclose(cps, np.deg2rad(np.array([15.0])))


def test_dict_phase_unit_degrees_converted_to_radians():
    data = _base_dict(cp_flag=False, i_cps1=None, i_cps2=None, i_cps3=None)
    data["phi"] = np.array([0.0, 90.0, -180.0])
    data["d_phi"] = np.array([1.0, 2.0, 3.0])
    data["phi_unit"] = "deg"

    oidata = OIData(data)

    assert np.allclose(oidata.phi, np.deg2rad(np.array([0.0, 90.0, -180.0])))
    assert np.allclose(oidata.d_phi, np.deg2rad(np.array([1.0, 2.0, 3.0])))


def test_oifits_phase_units_convert_values_and_uncertainties_to_radians():
    cases = [
        ("OI_VIS", "DEGREES", onp.array([10.0, -20.0]), onp.array([1.0, 2.0])),
        ("OI_VIS", "RADIANS", onp.array([0.1, -0.2]), onp.array([0.01, 0.02])),
        ("OI_VIS", None, onp.array([15.0, -30.0]), onp.array([0.5, 0.75])),
        ("OI_T3", "DEGREES", onp.array([12.0]), onp.array([1.5])),
        ("OI_T3", "RADIANS", onp.array([0.4]), onp.array([0.05])),
        ("OI_T3", None, onp.array([-25.0]), onp.array([2.5])),
    ]

    for phase_ext, unit, phi, d_phi in cases:
        oidata = OIData(
            _make_fake_oifits_phase_input(phase_ext, phi, d_phi, unit)
        )
        if unit == "RADIANS":
            expected_phi = phi
            expected_d_phi = d_phi
        else:
            expected_phi = onp.deg2rad(phi)
            expected_d_phi = onp.deg2rad(d_phi)

        assert np.allclose(oidata.phi, expected_phi)
        assert np.allclose(oidata.d_phi, expected_d_phi)


def test_cp_flag_inferred_from_indices_when_missing():
    data = _base_dict(
        cp_flag=False,
        i_cps1=np.array([0]),
        i_cps2=np.array([1]),
        i_cps3=np.array([2]),
    )
    del data["cp_flag"]
    oidata = OIData(data)
    assert oidata.cp_flag is True


def test_cp_flag_string_false_is_parsed_as_false():
    data = _base_dict(
        cp_flag="False",
        i_cps1=np.array([0]),
        i_cps2=np.array([1]),
        i_cps3=np.array([2]),
    )
    oidata = OIData(data)
    assert oidata.cp_flag is False


def test_cp_flag_string_true_is_parsed_as_true():
    data = _base_dict(
        cp_flag="true",
        i_cps1=np.array([0]),
        i_cps2=np.array([1]),
        i_cps3=np.array([2]),
    )
    data["phi"], data["d_phi"] = np.array([0.0]), np.array([1.0])
    oidata = OIData(data)
    assert oidata.cp_flag is True


def test_cp_flag_without_closure_indices_is_rejected():
    data = _base_dict(cp_flag=True, i_cps1=None, i_cps2=None, i_cps3=None)
    with pytest.raises(ValueError, match="closure-phase indices"):
        OIData(data)


def test_v2_flag_string_false_is_parsed_as_false():
    data = _base_dict(cp_flag=False, i_cps1=None, i_cps2=None, i_cps3=None)
    data["v2_flag"] = "False"
    oidata = OIData(data)
    assert oidata.v2_flag is False


def test_v2_flag_string_true_is_parsed_as_true():
    data = _base_dict(cp_flag=False, i_cps1=None, i_cps2=None, i_cps3=None)
    data["v2_flag"] = "true"
    oidata = OIData(data)
    assert oidata.v2_flag is True


def test_chi2ppf_df1_returns_finite_values():
    p = np.array([1e-6, 0.5, 0.95, 1.0 - 1e-6])
    q = chi2ppf(p, 1.0)
    assert np.all(np.isfinite(q))
    assert np.all(q >= 0.0)


@pytest.mark.parametrize("df", [1.0, 2.0, 3.0, 5.0, 10.0, 50.0])
def test_chi2ppf_matches_scipy_for_any_df(df):
    # F17 (virgil-validation): df != 1 used numpyro's gammaincinv, which
    # needs tensorflow_probability. It is now a Halley inversion of gammainc.
    p = onp.concatenate(
        [
            onp.logspace(-10, -1, 10),
            onp.linspace(0.1, 0.9, 9),
            1.0 - onp.logspace(-1, -10, 10),
        ]
    )
    with jax.enable_x64(True):
        q = onp.asarray(chi2ppf(p, df))
    onp.testing.assert_allclose(q, stats.chi2.ppf(p, df), rtol=1e-10, atol=0.0)


@pytest.mark.parametrize("df", [1.0, 2.0, 3.0, 5.0, 10.0, 50.0])
def test_chi2ppf_matches_scipy_in_float32(df):
    p = onp.concatenate(
        [
            onp.logspace(-4, -1, 10),
            onp.linspace(0.1, 0.9, 9),
            1.0 - onp.logspace(-1, -4, 10),
        ]
    ).astype(onp.float32)
    with jax.enable_x64(False):
        q = chi2ppf(p, df)
    assert q.dtype == onp.float32
    onp.testing.assert_allclose(
        onp.asarray(q), stats.chi2.ppf(p.astype(float), df), rtol=1e-4
    )


@pytest.mark.parametrize("df", [0.0, -2.0])
def test_chi2ppf_nonpositive_df_is_nan(df):
    q = onp.asarray(chi2ppf(onp.array([0.1, 0.5, 0.9]), df))
    assert onp.all(onp.isnan(q))


def test_chi2ppf_is_differentiable_and_jittable():
    with jax.enable_x64(True):
        p, df = 0.9, 3.0
        x = stats.chi2.ppf(p, df)
        dq_dp = jax.grad(chi2ppf, argnums=0)(p, df)
        onp.testing.assert_allclose(
            dq_dp, 1.0 / stats.chi2.pdf(x, df), rtol=1e-10
        )
        h = 1e-5
        numeric = (stats.chi2.ppf(p, df + h) - stats.chi2.ppf(p, df - h)) / (
            2 * h
        )
        dq_ddf = jax.grad(chi2ppf, argnums=1)(p, df)
        onp.testing.assert_allclose(dq_ddf, numeric, rtol=1e-6)
        jitted = jax.jit(chi2ppf)(np.asarray([0.5, 0.99]), np.asarray(4.0))
        onp.testing.assert_allclose(
            jitted, stats.chi2.ppf([0.5, 0.99], 4.0), rtol=1e-10
        )


def test_visibility_correlation_ticks_use_adaptive_float_formatter():
    data = OIData(_base_dict())
    pred = {
        "vis_mean": np.array([0.98, 1.0, 1.02]),
        "vis_std": np.array([0.01, 0.01, 0.01]),
        "phi_mean": np.array([0.0, 0.0, 0.0]),
        "phi_std": np.array([1.0, 1.0, 1.0]),
    }

    fig, (ax1, _) = plot_data_model_correlation(
        data,
        predictions_by_label={"demo": pred},
    )
    xfmt = ax1.xaxis.get_major_formatter()
    yfmt = ax1.yaxis.get_major_formatter()
    assert isinstance(xfmt, FuncFormatter)
    assert xfmt is yfmt
    assert xfmt(0.991, 0) != xfmt(0.992, 0)
    assert xfmt(0.985, 0) == "98.5"
    assert "%" not in xfmt(0.991, 0)
    assert "(V2, %)" in ax1.get_xlabel()
    assert "(V2, %)" in ax1.get_ylabel()
    plt.close(fig)


def test_data_model_correlation_without_phases():
    # AMIGO DISCO data hold every observable in vis and have no phases.
    from virgil.coverage import ami_grid_record
    from virgil.models import PointSource

    data = OIData(ami_grid_record(pitch_m=0.5))
    assert data.phi.size == 0
    prediction = data.model(PointSource())
    pred = {
        "vis_mean": prediction,
        "vis_std": 0 * prediction,
        "phi_mean": prediction[:0],
        "phi_std": prediction[:0],
    }
    fig, (ax1, ax2) = plot_data_model_correlation(data, {"star": pred})
    assert ax1.get_visible() and not ax2.get_visible()
    plt.close(fig)


def test_delta_mag_map_uses_reversed_colormap_by_default():
    limit_map = np.array([[1e-3, 2e-3], [5e-4, 1e-3]])
    grid = {"dra": np.array([-1.0, 1.0]), "ddec": np.array([-1.0, 1.0])}

    fig, ax = plot_grid_map(
        limit_map, grid, kind="limit", units="delta_mag", cmap=None
    )
    assert ax.images[0].get_cmap().name == "magma_r"
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    assert xlim[0] > xlim[1], f"x-axis not East-left: {xlim}"
    assert ylim[0] < ylim[1], f"y-axis not North-up: {ylim}"
    plt.close(fig)


def test_explicit_colormap_overrides_the_default():
    limit_map = np.array([[1e-3, 2e-3], [5e-4, 1e-3]])
    grid = {"dra": np.array([-1.0, 1.0]), "ddec": np.array([-1.0, 1.0])}

    fig, ax = plot_grid_map(
        limit_map, grid, kind="limit", units="delta_mag", cmap="inferno_r"
    )
    assert ax.images[0].get_cmap().name == "inferno_r"
    plt.close(fig)


def test_plot_model_shows_east_left_north_up():
    # A source offset to the North-East must be drawn in the upper-left.
    fov, npix = 10.0, 5
    model = GaussianDisk(sigma=1e-3, flux=1.0, dra=2.0, ddec=2.0)
    fig, ax = plt.subplots()
    plot_model(model, fov_mas=fov, npix=npix, ax=ax)

    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    assert xlim[0] > xlim[1], f"x-axis not East-left: {xlim}"
    assert ylim[0] < ylim[1], f"y-axis not North-up: {ylim}"

    # Map the brightest pixel to the data coordinates it is displayed at.
    image = ax.get_images()[0]
    array = onp.asarray(image.get_array())
    left, right, bottom, top = image.get_extent()
    row, col = onp.unravel_index(array.argmax(), array.shape)
    x = left + (col + 0.5) * (right - left) / npix
    if image.origin == "upper":
        y = top - (row + 0.5) * (top - bottom) / npix
    else:
        y = bottom + (row + 0.5) * (top - bottom) / npix
    assert onp.allclose((x, y), (2.0, 2.0))
    plt.close(fig)


# === Fixes from the 2026-09 code review ===


def test_plotting_leaves_global_rcparams_alone():
    import importlib

    import matplotlib

    import virgil.plotting as plotting

    before = dict(matplotlib.rcParams)
    importlib.reload(plotting)
    plotting.plot_model(GaussianDisk(sigma=5.0), fov_mas=40.0, npix=16)
    plt.close("all")
    assert dict(matplotlib.rcParams) == before


def test_likelihood_map_pixels_are_centred_on_samples():
    samples = {
        "dra": onp.linspace(-10.0, 10.0, 5),
        "ddec": onp.linspace(-4.0, 4.0, 3),
        "flux": onp.array([1e-3]),
    }
    fig, ax = plot_grid_map(onp.zeros((5, 3)), samples)
    left, right, bottom, top = ax.images[0].get_extent()
    # Samples are 5 apart in dra and 4 apart in ddec: pad by half of that.
    assert sorted([left, right]) == [-12.5, 12.5]
    assert sorted([bottom, top]) == [-6.0, 6.0]
    plt.close(fig)


def test_likelihood_map_uses_dra_as_x_whatever_the_key_order():
    samples = {
        "ddec": onp.linspace(-4.0, 4.0, 3),
        "dra": onp.linspace(-10.0, 10.0, 5),
        "flux": onp.array([1e-3]),
    }
    grid = onp.arange(15.0).reshape(3, 5)  # axes (ddec, dra)
    fig, ax = plot_grid_map(grid, samples)
    assert ax.get_xlabel() == "ΔRA (mas)"
    assert ax.images[0].get_array().shape == (3, 5)  # rows are ddec
    plt.close(fig)


def test_grid_map_kinds_set_defaults_and_accept_overrides():
    samples = {
        "dra": onp.linspace(-10.0, 10.0, 4),
        "ddec": onp.linspace(-10.0, 10.0, 4),
        "flux": onp.array([1e-3, 1e-2]),
    }
    fig, ax = plot_grid_map(onp.full((4, 4), 1e-3), samples, kind="flux")
    assert ax.get_title() == "Best-fit flux (flux ratio)"
    fig, ax = plot_grid_map(
        onp.full((4, 4), 1e-3), samples, kind="flux", title="Mine"
    )
    assert ax.get_title() == "Mine"
    with pytest.raises(ValueError, match="kind"):
        plot_grid_map(onp.zeros((4, 4)), samples, kind="nonsense")
    # Only log-likelihood grids are reduced over the flux axis.
    with pytest.raises(ValueError, match="only kind='loglike'"):
        plot_grid_map(onp.zeros((4, 4, 2)), samples, kind="flux")
    plt.close("all")


def test_ruffio_upperlimit_accepts_scalars_and_keeps_axis_order():
    from virgil.limits import ruffio_upperlimit

    scalar = ruffio_upperlimit(1e-3, 1e-3, 0.5)
    assert np.shape(scalar) == ()
    mean = onp.full((3, 4), 1e-3)
    limits = ruffio_upperlimit(mean, 1e-3, onp.array([0.5, 0.9]))
    assert limits.shape == (3, 4, 2)
    assert np.all(limits[..., 1] > limits[..., 0])


def test_radial_profile_counts_and_statistics():
    from virgil.limits import radial_profile

    axis = onp.linspace(-4.0, 4.0, 9)
    values = onp.ones((9, 9))
    values[4, 4] = onp.nan  # non-finite values are ignored
    profile = radial_profile(values, axis, axis, bins=4)
    assert profile["r"].shape == profile["count"].shape == (4,)
    assert profile["count"].sum() == 80
    assert np.allclose(profile["median"][profile["count"] > 0], 1.0)


def test_best_grid_point_rejects_reduced_grids():
    from virgil.grid_fit import best_grid_point

    samples = {"dra": onp.arange(3.0), "ddec": onp.arange(4.0)}
    samples["flux"] = onp.array([1e-3, 1e-2])
    with pytest.raises(ValueError, match="axes"):
        best_grid_point(onp.zeros((3, 4)), samples)
    grid = onp.zeros((3, 4, 2))
    grid[1, 2, 0] = onp.nan
    grid[2, 3, 1] = 1.0
    assert best_grid_point(grid, samples) == {
        "dra": 2.0,
        "ddec": 3.0,
        "flux": 1e-2,
    }


def _legacy_dic(phi):
    """A three-hole dictionary for the legacy ImPlaneIA writer."""
    dic = {
        "info": {
            "TARGET": "UNKNOWN",
            "OBJECT": "UNKNOWN",
            "INSTRUME": "TEST",
            "MASK": "MASK3",
            "ARRNAME": "ARRAY3",
            "FILT": "F480M",
            "DATE-OBS": "2000-01-01",
            "TELESCOP": "SIM",
            "OBSERVER": "tests",
            "INSMODE": "NRM",
            "PA": 0.0,
            "MJD": [61000.0],
            "PSCALE": 65.0,
            "ISZ": 81,
            "STAXY": onp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
            "CTRS_EQT": onp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
        },
        "OI_WAVELENGTH": {"EFF_WAVE": 4.8e-6, "EFF_BAND": 0.3e-6},
        "OI_VIS": {
            "TARGET_ID": 1,
            "TIME": 0.0,
            "MJD": 61000.0,
            "INT_TIME": 1.0,
            "VISAMP": onp.ones(3),
            "VISAMPERR": onp.full(3, 0.01),
            "VISPHI": phi,
            "VISPHIERR": onp.ones(3),
            "UCOORD": onp.array([1.0, 0.0, -1.0]),
            "VCOORD": onp.array([0.0, 1.0, 1.0]),
            "STA_INDEX": onp.array([[1, 2], [1, 3], [2, 3]]),
            "FLAG": onp.zeros(3, dtype=bool),
        },
        "OI_T3": {
            "TARGET_ID": 1,
            "TIME": 0.0,
            "MJD": 61000.0,
            "INT_TIME": 1.0,
            "T3AMP": onp.ones(1),
            "T3AMPERR": onp.ones(1),
            "T3PHI": onp.array([12.0]),
            "T3PHIERR": onp.array([1.0]),
            "U1COORD": onp.array([1.0]),
            "V1COORD": onp.array([0.0]),
            "U2COORD": onp.array([-1.0]),
            "V2COORD": onp.array([1.0]),
            "STA_INDEX": onp.array([[1, 2, 3]]),
            "FLAG": onp.zeros(1, dtype=bool),
        },
    }
    dic["OI_VIS2"] = {
        key: dic["OI_VIS"][key]
        for key in ("TARGET_ID", "TIME", "MJD", "INT_TIME", "UCOORD")
    }
    dic["OI_VIS2"].update(
        VCOORD=dic["OI_VIS"]["VCOORD"],
        VIS2DATA=onp.ones(3),
        VIS2ERR=onp.full(3, 0.01),
        STA_INDEX=dic["OI_VIS"]["STA_INDEX"],
        FLAG=onp.zeros(3, dtype=bool),
    )
    return dic


def test_legacy_save_writes_a_readable_file(tmp_path):
    from virgil.legacy import oifits_implaneia

    phi = onp.array([10.0, -5.0, 3.0])
    dic = _legacy_dic(phi)
    original_mjd = list(dic["info"]["MJD"])

    oifits_implaneia.save(dic, datadir=tmp_path)

    # The caller's dictionary is untouched, and the file reads back.
    assert dic["info"]["MJD"] == original_mjd
    assert dic["OI_VIS"]["VISPHI"] is phi
    (path,) = tmp_path.glob("*.oifits")
    with fits.open(path) as hdul:
        assert hdul[0].header["MASK"] == "MASK3"
        assert hdul["OI_ARRAY"].data["FOV"][0] == pytest.approx(0.065 * 81 / 2)
    assert np.allclose(OIData(path).phi, onp.deg2rad(12.0))


def test_legacy_save_needs_astroquery_for_a_named_target(
    tmp_path, monkeypatch
):
    # Without the [legacy] extra a named target must not be written with
    # placeholder coordinates.
    import sys

    from virgil.legacy import oifits_implaneia

    dic = _legacy_dic(onp.array([10.0, -5.0, 3.0]))
    dic["info"]["TARGET"] = "HD 1"
    monkeypatch.setitem(sys.modules, "astroquery.simbad", None)
    with pytest.raises(ImportError, match=r"virgil-astro\[legacy\]"):
        oifits_implaneia.save(dic, datadir=tmp_path)


def test_transposed_operator_is_rejected_with_a_hint():
    data = _base_dict(cp_flag=False, i_cps1=None, i_cps2=None, i_cps3=None)
    operator = onp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 1.0]])
    with pytest.raises(ValueError, match="transposed"):
        OIData({**data, "vis_mat": operator.T})
    projected = OIData({**data, "vis_mat": operator})
    assert projected.vis.shape == (2,)


def test_chainconsumer_diagnostics_return_their_figures():
    import pandas as pd
    from matplotlib.figure import Figure

    from virgil.plotting import plot_chainconsumer_diagnostics

    rng = onp.random.default_rng(0)
    chain = pd.DataFrame(rng.normal(size=(400, 2)), columns=["dra", "ddec"])
    consumer, corner, walks = plot_chainconsumer_diagnostics(
        {"chain": chain}, columns=["dra", "ddec"]
    )
    assert isinstance(corner, Figure) and isinstance(walks, Figure)
    plt.close("all")


def test_batched_grid_matches_unbatched():
    from virgil.grid_fit import likelihood_grid
    from virgil.models import BinaryModelCartesian
    from tests._test_data import oidata

    samples = {
        "dra": np.linspace(-200.0, 200.0, 9),
        "ddec": np.linspace(-200.0, 200.0, 7),
        "flux": np.array([1e-4, 1e-3, 1e-2]),
    }
    whole = likelihood_grid(
        BinaryModelCartesian, oidata, samples, batch_size=10**6
    )
    batched = likelihood_grid(
        BinaryModelCartesian, oidata, samples, batch_size=10
    )
    assert np.allclose(batched, whole)
    with pytest.raises(ValueError, match="batch_size"):
        likelihood_grid(BinaryModelCartesian, oidata, samples, batch_size=0)


@pytest.mark.parametrize(
    "backend, budget", [("cpu", 2**20), ("gpu", 2**23), ("tpu", 2**23)]
)
def test_default_batch_size_scales_with_data_size(
    monkeypatch, backend, budget
):
    from virgil import _grid
    from virgil._grid import MIN_BATCH_SIZE, batch_size_or_default

    monkeypatch.setattr(_grid.jax, "default_backend", lambda: backend)

    def data_with(n_vis):
        values = {
            key: np.ones(n_vis) for key in ("u", "v", "vis", "d_vis", "phi")
        }
        return OIData({**_base_dict(), **values, "d_phi": np.ones(n_vis)})

    # Small data get a batch of ~budget model visibilities.
    assert batch_size_or_default(None, data_with(24)) == budget // 24
    # Large data keep the minimum, bounding memory as before.
    n_large = budget // MIN_BATCH_SIZE + 1
    assert batch_size_or_default(None, data_with(n_large)) == MIN_BATCH_SIZE
    # An explicit batch size is used as given.
    assert batch_size_or_default(7, data_with(24)) == 7


def test_styled_plotting_keeps_other_figures_open():
    # plt.rc_context restores the backend on exit, which in Jupyter can
    # close every open figure before it is shown.
    import matplotlib

    from virgil.plotting import plot_model

    backend = matplotlib.rcParams["backend"]
    existing = plt.figure()
    ax = plot_model(GaussianDisk(sigma=5.0), fov_mas=40.0, npix=16)
    assert existing.number in plt.get_fignums()
    assert ax.figure.number in plt.get_fignums()
    assert matplotlib.rcParams["backend"] == backend
    plt.close("all")


def test_legacy_load_then_save_round_trips(tmp_path):
    from virgil.legacy import oifits_implaneia

    first = tmp_path / "first"
    test_legacy_save_writes_a_readable_file(first)
    (path,) = first.glob("*.oifits")
    loaded = oifits_implaneia.load(path)
    oifits_implaneia.save(loaded, datadir=tmp_path / "second")
    (again,) = (tmp_path / "second").glob("*.oifits")
    assert np.allclose(OIData(again).phi, OIData(path).phi)
    with fits.open(again) as hdul:
        assert list(hdul["OI_ARRAY"].data["STA_INDEX"]) == [1, 2, 3]


def test_contrast_curve_is_independent_of_grid_key_order():
    from virgil.plotting import plot_contrast_curve

    dra = onp.linspace(-20.0, 20.0, 9)
    ddec = onp.linspace(-10.0, 10.0, 5)
    limits = 1e-3 * (1.0 + onp.hypot(*onp.meshgrid(dra, ddec, indexing="ij")))
    _, ax = plot_contrast_curve(limits, {"dra": dra, "ddec": ddec})
    _, ax_t = plot_contrast_curve(
        limits.T, {"ddec": ddec, "flux": onp.array([1e-3]), "dra": dra}
    )
    assert onp.allclose(
        ax.lines[0].get_ydata(), ax_t.lines[0].get_ydata(), equal_nan=True
    )
    plt.close("all")


def test_plot_residual_map_is_symmetric_and_east_left():
    # A positive residual in the top-left pixel is North-East; the colour
    # scale is symmetric; with sigma the map shows z-scores.
    npix, fov = 5, 10.0
    residual = onp.zeros((npix, npix))
    residual[0, 0], residual[4, 4] = 2.0, -0.5
    fig, ax = plt.subplots()
    plot_residual_map(residual, fov_mas=fov, sigma=0.5, ax=ax)
    image = ax.get_images()[0]
    assert image.get_clim() == (-4.0, 4.0)
    assert ax.get_xlim()[0] > ax.get_xlim()[1]
    assert ax.get_ylim()[0] < ax.get_ylim()[1]
    left, right, bottom, top = image.get_extent()
    assert left > 0 and top > 0  # column 0 is East, row 0 is North
    plt.close(fig)
