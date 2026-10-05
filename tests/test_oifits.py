"""OIFITS reading/writing and the OIData observables built from it."""

import jax
import jax.numpy as np
import numpy as onp
import pyoifits
import pytest
from astropy.io import fits

from virgil.likelihood import loglike, model_loglike, whitened_residuals
from virgil.models import BinaryModelCartesian
from virgil.oidata import OIData, closure_phases, cp_indices
from virgil.oifits import read_oifits, write_oifits


STATIONS = onp.array([[0.0, 0.0], [3.2, 0.2], [1.4, 2.6], [-1.1, 1.8]])
PAIRS = onp.array([[1, 2], [1, 3], [1, 4], [2, 3], [2, 4], [3, 4]])
TRIANGLES = onp.array([[1, 2, 3], [1, 2, 4], [1, 3, 4], [2, 3, 4]])
TRUTH = BinaryModelCartesian(dra=60.0, ddec=-40.0, flux=0.05)


def _baselines(pairs=PAIRS):
    delta = STATIONS[pairs[:, 1] - 1] - STATIONS[pairs[:, 0] - 1]
    return delta[:, 0], delta[:, 1]


def _tables(waves=(4.8e-6,), model=TRUTH, target="STAR"):
    """OIFITS tables sampling ``model`` noiselessly at ``waves``."""
    waves = onp.asarray(waves)
    u, v = _baselines()
    cvis = onp.asarray(
        model.model(u[:, None], v[:, None], waves[None, :])
    )  # (n_baseline, n_wave)
    i1, i2, i3 = cp_indices(PAIRS, TRIANGLES)
    cp = onp.angle(cvis[i1] * cvis[i2] / cvis[i3])
    u1, v1 = _baselines(TRIANGLES[:, [0, 1]])
    u2, v2 = _baselines(TRIANGLES[:, [1, 2]])
    return {
        "info": {"TARGET": target, "INSTRUME": "TEST", "MJD": 60000.0},
        "OI_WAVELENGTH": {"EFF_WAVE": waves, "EFF_BAND": 0.1e-6},
        "OI_VIS2": {
            "VIS2DATA": onp.abs(cvis) ** 2,
            "VIS2ERR": onp.full(cvis.shape, 1e-3),
            "UCOORD": u,
            "VCOORD": v,
            "STA_INDEX": PAIRS,
        },
        "OI_T3": {
            "T3PHI": onp.rad2deg(cp),
            "T3PHIERR": onp.full(cp.shape, 0.5),
            "U1COORD": u1,
            "V1COORD": v1,
            "U2COORD": u2,
            "V2COORD": v2,
            "STA_INDEX": TRIANGLES,
        },
    }


def test_roundtrip_single_channel(tmp_path):
    tables = _tables()
    path = write_oifits(tables, tmp_path / "one.oifits")
    data = OIData(path)

    assert data.v2_flag and data.cp_flag
    assert np.allclose(data.u, tables["OI_VIS2"]["UCOORD"])
    assert np.allclose(data.vis, tables["OI_VIS2"]["VIS2DATA"].ravel())
    assert np.allclose(
        data.phi, onp.deg2rad(tables["OI_T3"]["T3PHI"].ravel()), atol=1e-6
    )
    assert data.wavel.shape == (1,)
    # Noiseless data: the model reproduces the file exactly.
    assert np.allclose(data.model(TRUTH), data.flatten_data()[0], atol=1e-5)


def test_written_file_is_standard_oifits(tmp_path):
    path = write_oifits(_tables(waves=(4.6e-6, 5.0e-6)), tmp_path / "x.fits")
    with fits.open(path) as hdul:
        names = [hdu.name for hdu in hdul]
        assert hdul[0].header["CONTENT"] == "OIFITS2"
    # OI_WAVELENGTH is not the first extension, which the old reader assumed.
    assert names[1] != "OI_WAVELENGTH"
    opened = pyoifits.open(path)
    assert {"OI_VIS2", "OI_T3"} <= {h.name for h in opened.get_dataHDUs()}
    # pyoifits objects are astropy HDULists and read the same way.
    assert np.allclose(OIData(opened).vis, OIData(path).vis)


def test_multi_wavelength_closure_phases_use_their_own_channel(tmp_path):
    waves = onp.array([4.4e-6, 4.8e-6, 5.2e-6])
    path = write_oifits(_tables(waves=waves), tmp_path / "multi.oifits")
    data = OIData(path)

    n_bl, n_cp, n_wave = len(PAIRS), len(TRIANGLES), waves.size
    assert data.u.shape == (n_bl * n_wave,)
    assert data.wavel.shape == (n_bl * n_wave,)
    assert data.phi.shape == (n_cp * n_wave,)
    # Chromatic data: the model at each sample's own wavelength matches.
    assert np.allclose(data.model(TRUTH), data.flatten_data()[0], atol=1e-5)
    # ... while ignoring the chromaticity does not.
    grey = TRUTH.model(data.u, data.v, waves[1])
    assert not np.allclose(
        closure_phases(grey, data.i_cps1, data.i_cps2, data.i_cps3),
        data.phi,
        atol=1e-3,
    )


def test_flags_and_non_finite_values_are_dropped(tmp_path):
    waves = onp.array([4.4e-6, 4.8e-6])
    tables = _tables(waves=waves)
    vis_flag = onp.zeros((len(PAIRS), 2), dtype=bool)
    vis_flag[2, 1] = True
    tables["OI_VIS2"]["FLAG"] = vis_flag
    tables["OI_T3"]["T3PHI"][1, 0] = onp.nan
    data = OIData(write_oifits(tables, tmp_path / "flags.oifits"))

    assert data.vis.size == len(PAIRS) * 2 - 1
    assert data.phi.size == len(TRIANGLES) * 2 - 1
    # The flagged baseline is still available to the closure phases.
    assert data.u.size == len(PAIRS) * 2
    assert data.vis_index.size == data.vis.size
    assert np.isfinite(model_loglike(TRUTH, data))
    assert np.allclose(data.model(TRUTH), data.flatten_data()[0], atol=1e-5)


def test_absolute_phases_from_oi_vis_with_reversed_baseline(tmp_path):
    tables = _tables()
    u, v = _baselines()
    cvis = onp.asarray(TRUTH.model(u, v, 4.8e-6))
    pairs = PAIRS.copy()
    pairs[0] = pairs[0][::-1]  # stored as (2, 1) in OI_VIS
    phase = onp.rad2deg(onp.angle(cvis))
    phase[0] = -phase[0]
    tables["OI_VIS"] = {
        "VISAMP": onp.abs(cvis),
        "VISAMPERR": onp.full(cvis.shape, 1e-3),
        "VISPHI": phase,
        "VISPHIERR": onp.full(cvis.shape, 0.5),
        "UCOORD": onp.r_[-u[0], u[1:]],
        "VCOORD": onp.r_[-v[0], v[1:]],
        "STA_INDEX": pairs,
    }
    del tables["OI_T3"]
    data = OIData(write_oifits(tables, tmp_path / "vis.oifits"))

    assert not data.cp_flag
    assert np.allclose(data.phi, onp.angle(cvis), atol=1e-6)


def test_several_targets_need_a_choice(tmp_path):
    science = _tables()
    calibrator = _tables(model=BinaryModelCartesian(0.0, 0.0, 0.0))
    tables = dict(science)
    for table in ("OI_VIS2", "OI_T3"):
        n = len(science[table]["STA_INDEX"])
        tables[table] = {
            key: onp.concatenate([science[table][key], calibrator[table][key]])
            for key in science[table]
        }
        tables[table]["TARGET_ID"] = onp.repeat([1, 2], n)
    tables["OI_TARGET"] = {"TARGET_ID": [1, 2], "TARGET": ["SCI", "CAL"]}
    path = write_oifits(tables, tmp_path / "targets.oifits")

    with pytest.raises(ValueError, match="several targets"):
        OIData(path)
    sci = OIData(path, target="SCI")
    cal = OIData(path, target=2)
    assert sci.vis.size == cal.vis.size == len(PAIRS)
    assert np.allclose(sci.vis, science["OI_VIS2"]["VIS2DATA"].ravel())
    assert np.allclose(cal.vis, 1.0)


def test_missing_or_reversed_closure_leg_raises_clearly():
    with pytest.raises(ValueError, match="stored reversed"):
        cp_indices([[1, 2], [3, 2], [1, 3]], [[1, 2, 3]])
    with pytest.raises(ValueError, match="missing"):
        cp_indices([[1, 2], [1, 3]], [[1, 2, 3]])


def test_read_oifits_record_matches_oidata(tmp_path):
    path = write_oifits(_tables(), tmp_path / "rec.oifits")
    record = read_oifits(path)
    assert record["phi_unit"] == "rad"
    assert np.allclose(OIData(record).phi, OIData(path).phi)


# === OIData observables built from dictionaries ===


def _dict_data(**extra):
    u, v = _baselines()
    cvis = onp.asarray(TRUTH.model(u, v, 4.8e-6))
    i1, i2, i3 = cp_indices(PAIRS, TRIANGLES)
    return {
        "u": u,
        "v": v,
        "wavel": 4.8e-6,
        "vis": onp.abs(cvis) ** 2,
        "d_vis": onp.full(len(u), 1e-3),
        "phi": onp.angle(cvis[i1] * cvis[i2] / cvis[i3]),
        "d_phi": onp.full(len(i1), 1e-2),
        "i_cps1": i1,
        "i_cps2": i2,
        "i_cps3": i3,
        **extra,
    }


def test_vis_mode_converts_data_without_an_operator():
    v2 = OIData(_dict_data())
    amp = OIData(_dict_data(vis_mode="amp"))
    assert np.allclose(amp.vis, np.sqrt(v2.vis))
    # Data and model are compared in the same (amplitude) basis.
    assert np.allclose(amp.model(TRUTH), amp.flatten_data()[0], atol=1e-5)


def test_v2_near_zero_gives_bounded_amplitude_errors():
    # Review 1.5: V² -> amplitude errors were ½σ/√V² with V² floored at
    # 1e-30, so V² = 1e-4 ± 0.01 became |V| = 0.01 ± 0.5 and V² <= 0 gave
    # errors of ~5e12. The data are now floored at their own error.
    raw = _dict_data()
    v2 = onp.array(raw["vis"])
    v2[:4] = [1e-4, 0.0, -0.02, 0.49]
    d_v2 = onp.full(v2.size, 1e-2)
    amp = OIData({**raw, "vis": v2, "d_vis": d_v2, "vis_mode": "amp"})
    floored = onp.maximum(v2[:4], 1e-2)
    assert onp.allclose(amp.vis[:4], onp.sqrt(onp.maximum(v2[:4], 0.0)))
    assert onp.allclose(amp.d_vis[:4], 0.5e-2 / onp.sqrt(floored))
    assert float(np.max(amp.d_vis)) <= 0.5 * onp.sqrt(1e-2) + 1e-7

    log = OIData({**raw, "vis": v2, "d_vis": d_v2, "vis_mode": "logamp"})
    assert onp.allclose(log.vis[:4], 0.5 * onp.log(floored))
    assert onp.allclose(log.d_vis[:4], 0.5e-2 / floored)
    assert float(np.max(log.d_vis)) <= 0.5 + 1e-7
    # Far from zero the propagation is unchanged.
    assert onp.isclose(float(amp.d_vis[3]), 0.5e-2 / 0.7, rtol=1e-6)


def test_pre_projected_closure_phases_are_not_projected_again():
    # One triangle: any two of four share a baseline, so their outputs
    # correlate and are rotated (see test_closure).
    raw = _dict_data()
    phi_mat = onp.eye(len(TRIANGLES))[:1]
    projected = OIData({**raw, "phi_mat": phi_mat})
    again = OIData({**raw, "phi": phi_mat @ raw["phi"], "phi_mat": phi_mat})
    assert projected.phi.shape == again.phi.shape == (1,)
    assert np.allclose(projected.phi, again.phi)
    assert again.model(TRUTH).shape == again.flatten_data()[0].shape


def test_dict_with_several_channels_is_expanded():
    waves = onp.array([4.4e-6, 5.2e-6])
    u, v = _baselines()
    cvis = onp.asarray(TRUTH.model(u[:, None], v[:, None], waves[None, :]))
    i1, i2, i3 = cp_indices(PAIRS, TRIANGLES)
    data = OIData(
        {
            "u": u,
            "v": v,
            "wavel": waves,
            "vis": onp.abs(cvis) ** 2,
            "d_vis": onp.full(cvis.shape, 1e-3),
            "phi": onp.angle(cvis[i1] * cvis[i2] / cvis[i3]),
            "d_phi": onp.full((len(i1), 2), 1e-2),
            "i_cps1": i1,
            "i_cps2": i2,
            "i_cps3": i3,
        }
    )
    assert data.u.size == 2 * len(u)
    assert np.allclose(data.model(TRUTH), data.flatten_data()[0], atol=1e-5)


def test_ambiguous_wavelengths_are_rejected():
    with pytest.raises(ValueError, match="wavelength"):
        OIData(_dict_data(wavel=onp.array([4.4e-6, 4.8e-6, 5.2e-6])))


def test_phase_residuals_wrap_around_pi():
    data = OIData(_dict_data())
    prediction = data.flatten_data()[0]
    n_vis = data.vis.size
    shifted = prediction.at[n_vis:].add(2.0 * onp.pi)
    assert np.allclose(data.residuals(shifted), 0.0, atol=1e-5)

    params = ["dra", "ddec", "flux"]
    values = [60.0, -40.0, 0.05]
    near_pi = OIData(_dict_data(phi=onp.full(len(TRIANGLES), onp.pi - 1e-3)))
    flipped = OIData(_dict_data(phi=onp.full(len(TRIANGLES), -onp.pi + 1e-3)))
    # Data at +π and -π are the same angle, so the likelihood is the same.
    assert np.allclose(
        loglike(values, params, near_pi, BinaryModelCartesian),
        loglike(values, params, flipped, BinaryModelCartesian),
        rtol=1e-4,
    )


def test_v2_error_from_negative_amplitude_is_not_zero():
    data = OIData(
        _dict_data(
            v2_flag=False,
            vis_mode="v2",
            vis=onp.full(len(PAIRS), -0.01),
            d_vis=onp.full(len(PAIRS), 0.05),
        )
    )
    assert np.all(data.d_vis > 1e-3)


def test_closure_phase_only_file_round_trips(tmp_path):
    waves = onp.array([4.4e-6, 5.2e-6])
    tables = _tables(waves=waves)
    del tables["OI_VIS2"]
    data = OIData(write_oifits(tables, tmp_path / "t3.oifits"))

    assert data.cp_flag
    assert data.vis.size == 0
    assert data.phi.size == len(TRIANGLES) * waves.size
    # Shared baselines are merged: six unique pairs per channel.
    assert data.u.size == len(PAIRS) * waves.size
    assert np.allclose(data.model(TRUTH), data.flatten_data()[0], atol=1e-5)
    assert np.isfinite(model_loglike(TRUTH, data))


def _two_instrument_file(names):
    """One HDUList holding the same tables under two INSNAMEs."""
    from virgil.oifits import build_hdulist

    hduls = []
    for name, waves in zip(names, ((2.2e-6,), (2.0e-6, 2.2e-6, 2.4e-6))):
        tables = _tables(waves=waves)
        tables["info"]["INSNAME"] = name
        hduls.append(build_hdulist(tables))
    combined = fits.HDUList([hdu.copy() for hdu in hduls[0]])
    for hdu in hduls[1]:
        if hdu.header.get("EXTNAME", "").strip() in {
            "OI_WAVELENGTH",
            "OI_VIS2",
            "OI_T3",
        }:
            combined.append(hdu.copy())
    return combined


def test_gravity_fringe_tracker_and_science_tables_are_not_merged():
    hdul = _two_instrument_file(["GRAVITY_FT", "GRAVITY_SC"])
    with pytest.raises(ValueError, match="insname="):
        read_oifits(hdul)
    record = read_oifits(hdul, insname="GRAVITY_SC")
    assert onp.unique(record["wavel"]).size == 3  # the SC table's channels
    with pytest.raises(ValueError, match="No tables have INSNAME"):
        read_oifits(hdul, insname="GRAVITY_SC_P1")


def test_insname_selects_tables_from_other_instruments():
    hdul = _two_instrument_file(["PIONIER_A", "PIONIER_B"])
    both = read_oifits(hdul)  # not GRAVITY FT + SC: merged as before
    one = read_oifits(hdul, insname=["PIONIER_A"])
    assert both["u"].size == 4 * one["u"].size  # 1 + 3 channels against 1


def test_differential_visphi_is_not_read_as_absolute(tmp_path):
    from virgil.oifits import build_hdulist

    tables = _tables()
    u, v = _baselines()
    tables["OI_VIS"] = {
        "VISAMP": onp.ones(u.size),
        "VISAMPERR": onp.full(u.size, 1e-3),
        "VISPHI": onp.zeros(u.size),
        "VISPHIERR": onp.full(u.size, 0.5),
        "UCOORD": u,
        "VCOORD": v,
        "STA_INDEX": PAIRS,
    }
    del tables["OI_T3"]
    hdul = build_hdulist(tables)
    for hdu in hdul:
        if hdu.header.get("EXTNAME", "").strip() == "OI_VIS":
            hdu.header["PHITYP"] = "differential"
    with pytest.raises(ValueError, match="PHITYP"):
        read_oifits(hdul)


def _amplitude_file(amptyp):
    from virgil.oifits import build_hdulist

    tables = _tables()
    u, v = _baselines()
    tables["OI_VIS"] = {
        "VISAMP": onp.full(u.size, 0.8),
        "VISAMPERR": onp.full(u.size, 1e-3),
        "VISPHI": onp.zeros(u.size),
        "VISPHIERR": onp.full(u.size, 0.5),
        "UCOORD": u,
        "VCOORD": v,
        "STA_INDEX": PAIRS,
    }
    del tables["OI_VIS2"]
    hdul = build_hdulist(tables)
    for hdu in hdul:
        if hdu.header.get("EXTNAME", "").strip() == "OI_VIS":
            if amptyp is None:
                del hdu.header["AMPTYP"]
            else:
                hdu.header["AMPTYP"] = amptyp
    return hdul


@pytest.mark.parametrize("amptyp", ["absolute", "ABSOLUTE ", None])
def test_absolute_or_missing_amptyp_is_read_as_amplitude(amptyp):
    record = read_oifits(_amplitude_file(amptyp))
    assert not record["v2_flag"]
    onp.testing.assert_allclose(record["vis"], 0.8)


@pytest.mark.parametrize("amptyp", ["differential", "correlated flux"])
def test_non_absolute_amptyp_is_not_read_as_amplitude(amptyp):
    with pytest.raises(ValueError, match=f"AMPTYP = '{amptyp}'"):
        read_oifits(_amplitude_file(amptyp))


def test_amptyp_is_ignored_when_oi_vis2_is_read():
    from virgil.oifits import build_hdulist

    hdul = _amplitude_file("correlated flux")
    hdul.append(
        next(h for h in build_hdulist(_tables()) if h.name == "OI_VIS2")
    )
    assert read_oifits(hdul)["v2_flag"]


def test_insname_selection_drops_emptied_table_types():
    from virgil.oifits import _collect_tables, _select_insname

    hdul = _two_instrument_file(["PIONIER_A", "PIONIER_B"])
    for hdu in hdul:
        if hdu.header.get("EXTNAME", "").strip() == "OI_T3" and (
            hdu.header.get("INSNAME", "").strip() == "PIONIER_B"
        ):
            hdu.header["EXTNAME"] = "OI_IGNORED"  # B has no closure phases
    tables = _select_insname(_collect_tables(hdul), ["PIONIER_B"])
    assert "OI_T3" not in tables
    assert len(tables["OI_VIS2"]) == 1


def test_phityp_is_checked_only_for_the_chosen_target():
    from virgil.oifits import build_hdulist

    tables = _tables()
    u, v = _baselines()
    tables["OI_VIS"] = {
        "VISAMP": onp.ones(u.size),
        "VISAMPERR": onp.full(u.size, 1e-3),
        "VISPHI": onp.zeros(u.size),
        "VISPHIERR": onp.full(u.size, 0.5),
        "UCOORD": u,
        "VCOORD": v,
        "STA_INDEX": PAIRS,
    }
    del tables["OI_T3"]
    hdul = build_hdulist(tables)
    vis = next(
        h for h in hdul if h.header.get("EXTNAME", "").strip() == "OI_VIS"
    )
    other = vis.copy()
    other.data["TARGET_ID"] = 99  # rows of another target
    other.header["PHITYP"] = "differential"
    hdul.append(other)
    record = read_oifits(hdul, target="STAR")  # its table is absolute
    assert onp.isfinite(record["phi"]).all()


def test_closure_phases_match_visibilities_within_an_exposure():
    # GRAVITY averages different frames of one exposure for OI_T3 and
    # OI_VIS2, so their MJDs can differ by more than a row's INT_TIME.
    from virgil.oifits import build_hdulist

    hdul = build_hdulist(_tables())
    vis2, t3 = hdul["OI_VIS2"].data, hdul["OI_T3"].data
    vis2["INT_TIME"] = 120.0
    t3["INT_TIME"] = 30.0
    t3["MJD"] = vis2["MJD"][0] + 131.0 / 86400.0
    assert read_oifits(hdul)["i_cps1"].size == len(TRIANGLES)

    # A different exposure (beyond twice the longest INT_TIME) is not used.
    t3["MJD"] = vis2["MJD"][0] + 300.0 / 86400.0
    with pytest.raises(ValueError, match="needs baseline"):
        read_oifits(hdul)


def _t3_under_other_insname(waves, reverse=()):
    """V² and T3 under different INSNAMEs with identical wavelengths.

    The baselines in ``reverse`` are stored reversed in the OI_VIS2 table
    (stations swapped, ``(u, v)`` negated), as in the 2006 Beauty Contest
    files.
    """
    from virgil.oifits import build_hdulist

    hdul = build_hdulist(_tables(waves=waves))
    vis2 = hdul["OI_VIS2"].data
    for pair in reverse:
        row = onp.flatnonzero((vis2["STA_INDEX"] == pair).all(axis=1))[0]
        vis2["STA_INDEX"][row] = pair[::-1]
        vis2["UCOORD"][row] *= -1
        vis2["VCOORD"][row] *= -1
    wave_hdu = hdul["OI_WAVELENGTH"].copy()
    insname = wave_hdu.header["INSNAME"]
    wave_hdu.header["INSNAME"] = insname + "_TR01"
    hdul["OI_T3"].header["INSNAME"] = insname + "_TR01"
    hdul.append(wave_hdu)
    return hdul


@pytest.mark.parametrize("reverse", [(), ((1, 3),)])
def test_closure_phases_pair_with_v2_of_another_insname(reverse, tmp_path):
    waves = onp.array([4.4e-6, 4.8e-6, 5.2e-6])
    hdul = _t3_under_other_insname(waves, reverse)
    path = tmp_path / "split.fits"
    hdul.writeto(path)
    data = OIData(path)

    n_bl, n_cp = len(PAIRS), len(TRIANGLES)
    assert data.phi.shape == (n_cp * waves.size,)
    # The V² are all there and unflagged; only a reversed leg adds samples.
    vis_flag = read_oifits(path)["vis_flag"]
    assert np.count_nonzero(~vis_flag) == n_bl * waves.size
    assert vis_flag.size == (n_bl + len(reverse)) * waves.size
    assert data.vis.shape == (n_bl * waves.size,)
    # The model at each (u, v, wavelength) reproduces both the V² and the
    # closure phases, so the reversed leg is the conjugate of the stored one.
    assert np.allclose(data.model(TRUTH), data.flatten_data()[0], atol=1e-5)
    # A table with different wavelengths is no match.
    hdul[-1].data["EFF_WAVE"] *= 1.001
    with pytest.raises(ValueError, match="same wavelengths"):
        read_oifits(hdul)


def test_closure_phases_never_pair_with_another_array():
    # STA_INDEX belongs to an array: a V² table of another ARRNAME that
    # reuses the station numbers and wavelengths (at other (u, v)) must not
    # supply the legs, even when it comes first in the file.
    hdul = _t3_under_other_insname((4.8e-6,))
    own = hdul["OI_VIS2"]
    for hdu in (own, hdul["OI_T3"]):
        hdu.header["ARRNAME"] = "ARRAY_A"
    other = own.copy()
    other.header["ARRNAME"] = "ARRAY_B"
    other.header["INSNAME"] = "OTHER"
    other.data["UCOORD"] = 3.0 * own.data["UCOORD"]
    wave = hdul["OI_WAVELENGTH"].copy()
    wave.header["INSNAME"] = "OTHER"
    hdul.insert(hdul.index_of("OI_VIS2"), other)
    hdul.append(wave)

    record = read_oifits(hdul)
    legs = record["u"][
        onp.concatenate([record[f"i_cps{k}"] for k in (1, 2, 3)])
    ]
    assert onp.isin(legs, own.data["UCOORD"]).all()
    assert not onp.isin(legs, other.data["UCOORD"]).any()


def test_missing_baseline_in_every_orientation_is_reported():
    hdul = _t3_under_other_insname((4.8e-6,))
    vis2 = hdul["OI_VIS2"].data
    row = onp.flatnonzero((vis2["STA_INDEX"] == (1, 3)).all(axis=1))[0]
    vis2["STA_INDEX"][row] = (2, 4)  # leaves (1, 3) stored nowhere
    with pytest.raises(ValueError, match=r"either orientation"):
        read_oifits(hdul)


def test_closure_only_rows_of_different_times_keep_their_own_baselines():
    # Without a visibility table, each T3 row's legs come from its own
    # coordinates, so two epochs within the exposure window stay separate.
    from virgil.oifits import build_hdulist

    tables = _tables()
    del tables["OI_VIS2"]
    hdul = build_hdulist(tables)
    t3 = hdul["OI_T3"]
    n = len(t3.data)
    later = fits.BinTableHDU.from_columns(t3.columns, nrows=2 * n)
    for name in t3.columns.names:
        later.data[name][n:] = t3.data[name]
    later.data["INT_TIME"] = 120.0
    later.data["MJD"][n:] += 131.0 / 86400.0
    later.header.update(t3.header)
    hdul["OI_T3"] = later

    record = read_oifits(hdul)
    assert record["u"].size == 2 * len(PAIRS)
    assert record["i_cps1"].size == 2 * n


def _night(tmp_path, name, mjd):
    tables = _tables(waves=(2.0e-6, 2.2e-6))
    tables["info"]["MJD"] = mjd
    return write_oifits(tables, tmp_path / name)


def test_times_and_frames_survive_reading_several_files(tmp_path):
    paths = [
        _night(tmp_path, "a.oifits", 60000.2),
        _night(tmp_path, "b.oifits", 60003.1),
    ]
    record = read_oifits(paths)
    n = record["u"].size
    assert record["mjd"].shape == record["frame"].shape == (n,)
    assert record["mjd"].dtype == onp.float64
    # One exposure per file, numbered apart.
    assert onp.array_equal(onp.unique(record["frame"]), [0, 1])
    data = OIData(paths)
    assert data.t_ref == pytest.approx(60000.2, abs=1e-9)
    assert onp.allclose(onp.unique(data.mjd), [60000.2, 60003.1], atol=1e-6)
    assert onp.array_equal(data.epochs(), record["frame"])


def test_a_frame_gets_one_time_by_default(tmp_path):
    # GRAVITY stamps T3 and VIS2 rows of one exposure at different MJDs.
    from virgil.oifits import build_hdulist

    hdul = build_hdulist(_tables())
    vis2, t3 = hdul["OI_VIS2"].data, hdul["OI_T3"].data
    vis2["INT_TIME"] = 120.0
    vis2["MJD"] = vis2["MJD"][0] + onp.arange(len(vis2)) * 20.0 / 86400.0
    t3["MJD"] = vis2["MJD"][0] + 131.0 / 86400.0
    mean = read_oifits(hdul)
    assert onp.unique(mean["frame"]).size == 1
    assert onp.unique(mean["mjd"]).size == 1
    assert mean["mjd"][0] == pytest.approx(vis2["MJD"].mean())
    rows = read_oifits(hdul, frame_mjd="row")
    assert onp.unique(rows["mjd"]).size == len(vis2)


def test_split_by_epoch_partitions_the_likelihood(tmp_path):
    paths = [
        _night(tmp_path, "a.oifits", 60000.2),
        _night(tmp_path, "b.oifits", 60000.25),
        _night(tmp_path, "c.oifits", 60003.1),
    ]
    data = OIData(paths)
    noisy = data.with_model(TRUTH, key=jax.random.PRNGKey(1))
    parts = noisy.split_by_epoch()
    assert [onp.unique(p.frame).size for p in parts] == [2, 1]
    assert sum(p.n_independent for p in parts) == noisy.n_independent
    model = BinaryModelCartesian(dra=55.0, ddec=-35.0, flux=0.04)
    whole = model_loglike(model, noisy)
    assert sum(model_loglike(model, p) for p in parts) == pytest.approx(
        whole, rel=1e-5
    )


def test_an_exposure_stays_in_one_epoch_with_row_times(tmp_path):
    # With frame_mjd="row" a frame's samples have different times; a gap
    # smaller than that spread must not split the frame.
    from virgil.oifits import build_hdulist

    hdul = build_hdulist(_tables())
    vis2, t3 = hdul["OI_VIS2"].data, hdul["OI_T3"].data
    vis2["INT_TIME"] = 120.0
    vis2["MJD"] = vis2["MJD"][0] + onp.arange(len(vis2)) * 20.0 / 86400.0
    t3["MJD"] = vis2["MJD"][0] + 60.0 / 86400.0
    data = OIData(read_oifits(hdul, frame_mjd="row"))
    assert onp.unique(data.mjd).size > 1
    assert onp.all(data.epochs(gap_days=1e-5) == 0)
    assert len(data.split_by_epoch(gap_days=1e-5)) == 1


def test_dict_times_per_sample_for_several_channels():
    waves = onp.array([2.0e-6, 2.2e-6])
    u, v = _baselines()
    cvis = onp.asarray(TRUTH.model(u[:, None], v[:, None], waves[None, :]))
    i1, i2, i3 = cp_indices(PAIRS, TRIANGLES)
    per_sample = onp.repeat([60100.5, 60101.5], 3)[:, None] + 0 * waves
    data = OIData(
        {
            "u": u,
            "v": v,
            "wavel": waves,
            "vis": onp.abs(cvis) ** 2,
            "d_vis": onp.full(cvis.shape, 1e-3),
            "phi": onp.angle(cvis[i1] * cvis[i2] / cvis[i3]),
            "d_phi": onp.full((len(i1), 2), 1e-2),
            "i_cps1": i1,
            "i_cps2": i2,
            "i_cps3": i3,
            "mjd": per_sample,
        }
    )
    assert onp.allclose(data.mjd, per_sample.reshape(-1))


def test_dict_times_per_baseline_and_missing_times():
    data = OIData(_dict_data(mjd=onp.full(6, 60100.5)))
    assert onp.allclose(data.mjd, 60100.5)
    assert onp.all(data.frame == 0)
    with pytest.raises(ValueError, match="no times"):
        OIData(_dict_data()).epochs()
    with pytest.raises(ValueError, match="mjd has shape"):
        OIData(_dict_data(mjd=onp.zeros(4)))


# === Closure phases that are all (or all but one) flagged ===


def _flagged_file(path, n_keep=0):
    """A file whose closure phases are flagged, but the first ``n_keep``."""
    tables = _tables()
    flag = onp.ones(tables["OI_T3"]["T3PHI"].shape, dtype=bool)
    flag[:n_keep] = False
    tables["OI_T3"]["FLAG"] = flag
    return write_oifits(tables, path)


def test_all_closure_phases_flagged_leaves_the_visibilities(tmp_path):
    path = _flagged_file(tmp_path / "flagged.oifits")
    with pytest.warns(UserWarning, match="Every closure phase is flagged"):
        data = OIData(path)
    assert not data.has_phases and not data.cp_flag
    assert data.phi.size == 0 and data.cp_noise is None
    assert whitened_residuals(TRUTH, data).size == len(PAIRS)


def test_all_closure_phases_flagged_in_a_record_leaves_the_visibilities():
    flagged = onp.ones(len(TRIANGLES), dtype=bool)
    with pytest.warns(UserWarning, match="Every closure phase is flagged"):
        data = OIData(_dict_data(phi_flag=flagged))
    assert not data.has_phases
    nan = _dict_data()
    nan["phi"] = onp.full(len(TRIANGLES), onp.nan)
    with pytest.warns(UserWarning, match="Every closure phase is flagged"):
        assert not OIData(nan).has_phases


def test_one_closure_phase_left_is_unchanged(tmp_path):
    path = _flagged_file(tmp_path / "one_left.oifits", n_keep=1)
    data = OIData(path)
    assert data.cp_flag and data.phi.size == 1
    assert data.cp_noise is None  # a single triangle: nothing correlates
    assert whitened_residuals(TRUTH, data).size == len(PAIRS) + 1
    assert np.isfinite(model_loglike(TRUTH, data))


def test_station_pairs_are_read_per_sample(tmp_path):
    paths = [
        _night(tmp_path, "a.oifits", 60000.2),
        _night(tmp_path, "b.oifits", 60003.1),
    ]
    record = read_oifits(paths)
    assert record["stations"].shape == (record["u"].size, 2)
    # Two channels per baseline, baseline-major, in the table's order.
    expected = onp.repeat(PAIRS, 2, axis=0)
    assert onp.array_equal(record["stations"], onp.vstack([expected] * 2))
    data = OIData(paths)
    assert onp.array_equal(onp.asarray(data.stations), record["stations"])
    gains = data.with_gains(telescope=0.01).gains
    assert gains.rows.shape[0] == 2  # one block per frame


def _two_snapshots(time_s, mjd):
    """Two exposures of the 4-station file in one table each, with the
    given per-snapshot TIME (s) and MJD values."""
    from virgil.oifits import build_hdulist

    hdul = build_hdulist(_tables())
    for name in ("OI_VIS2", "OI_T3"):
        table = hdul[name]
        n = len(table.data)
        both = fits.BinTableHDU.from_columns(table.columns, nrows=2 * n)
        for column in table.columns.names:
            both.data[column][n:] = table.data[column]
        both.data["TIME"][:n], both.data["TIME"][n:] = time_s
        both.data["MJD"][:n], both.data["MJD"][n:] = mjd
        both.header.update(table.header)
        hdul[name] = both
    return OIData(hdul)


def test_snapshots_told_apart_by_time_are_separate_frames():
    # OIFITS v1 writers such as OYSTER give a night one MJD and put the
    # snapshot in TIME. Closure phases are correlated only within a snapshot,
    # so two snapshots of 4 stations have 2 x 3 independent closure phases.
    by_mjd = _two_snapshots((0.0, 0.0), (60000.0, 60000.0 + 3600.0 / 86400.0))
    by_time = _two_snapshots((0.0, 3600.0), (60000.0, 60000.0))
    n_vis = by_mjd.vis.size
    assert by_mjd.n_independent - n_vis == 6
    assert by_time.n_independent - n_vis == 6
    assert onp.unique(onp.asarray(by_time.frame)).size == 2


def test_constant_time_still_groups_by_mjd():
    same = _two_snapshots((0.0, 0.0), (60000.0, 60000.0))
    apart = _two_snapshots((5.0, 5.0), (60000.0, 60001.0))
    assert onp.unique(onp.asarray(same.frame)).size == 1
    assert onp.unique(onp.asarray(apart.frame)).size == 2


def test_interleaved_times_within_the_mjd_window_share_a_frame():
    # Visibility-only rows at (M, 0 s), (M, 3600 s) and (M + 1 s, 0 s): the
    # first and third are one exposure even though the second sorts between
    # them.
    from virgil.oifits import build_hdulist

    tables = _tables()
    del tables["OI_T3"]
    hdul = build_hdulist(tables)
    table = hdul["OI_VIS2"]
    n = len(table.data)
    three = fits.BinTableHDU.from_columns(table.columns, nrows=3 * n)
    for column in table.columns.names:
        for k in (1, 2):
            three.data[column][k * n : (k + 1) * n] = table.data[column]
    three.data["MJD"][:] = 60000.0
    three.data["MJD"][2 * n :] += 1.0 / 86400.0
    three.data["TIME"][:] = 0.0
    three.data["TIME"][n : 2 * n] = 3600.0
    three.header.update(table.header)
    hdul["OI_VIS2"] = three
    frame = read_oifits(hdul)["frame"]
    assert onp.unique(frame).size == 2
    assert onp.array_equal(onp.unique(frame[:n]), onp.unique(frame[2 * n :]))


def test_lookup_takes_the_nearest_row_inside_both_windows():
    from virgil.oifits import _MJD_TOLERANCE, _BaselineLookup

    second = 1.0 / 86400.0
    lookup = _BaselineLookup()
    ins = ("ARRAY", "INS")
    # Nearer in the summed distance but outside the MJD window.
    lookup.add(ins, (1, 2), 60000.0 + 10 * second, 0.0, 0, 1, time=0.0)
    # Inside both windows.
    lookup.add(ins, (1, 2), 60000.0 + 8 * second, 0.0, 1, 1, time=8 * second)
    assert 8 * second < _MJD_TOLERANCE < 10 * second
    assert lookup.find(ins, (1, 2), 60000.0, time=0.0) == (1, 1)


def test_absolute_phases_with_a_nonzero_constant_time(tmp_path):
    tables = _tables()
    u, v = _baselines()
    cvis = onp.asarray(TRUTH.model(u, v, 4.8e-6))
    tables["OI_VIS"] = {
        "VISAMP": onp.abs(cvis),
        "VISAMPERR": onp.full(cvis.shape, 1e-3),
        "VISPHI": onp.rad2deg(onp.angle(cvis)),
        "VISPHIERR": onp.full(cvis.shape, 0.5),
        "UCOORD": u,
        "VCOORD": v,
        "STA_INDEX": PAIRS,
    }
    del tables["OI_T3"]
    with fits.open(write_oifits(tables, tmp_path / "vis.oifits")) as hdul:
        for name in ("OI_VIS2", "OI_VIS"):
            hdul[name].data["TIME"] = 3600.0
        data = OIData(hdul)
        assert not data.cp_flag
        assert np.allclose(data.phi, onp.angle(cvis), atol=1e-6)
