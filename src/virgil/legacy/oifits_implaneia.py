"""Legacy OIFITS helpers from ImPlaneIA, for its dictionary layout.

:func:`load` reads an OIFITS file into an ImPlaneIA dictionary (``info``,
``OI_VIS``, ``OI_VIS2``, ``OI_T3``, ...) and :func:`save` writes one, on top
of [`virgil.oifits.write_oifits`][virgil.oifits.write_oifits]. Phases
in these dictionaries are in **degrees**, as in OIFITS. New code should read
files with [`OIData`][virgil.oidata.OIData] and plot them with
[`plot_oidata_overview`][virgil.plotting.plot_oidata_overview].
"""

import copy
import os

import numpy as onp
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits


# cp_indices is re-exported here for existing imports.
from ..oidata import cp_indices  # noqa: F401


# astroquery is imported only when save() queries SIMBAD; tests may replace
# this with a stand-in.
Simbad = None


def _simbad():
    """The astroquery ``Simbad`` class, imported on first use."""
    if Simbad is not None:
        return Simbad
    try:
        from astroquery.simbad import Simbad as simbad_class
    except ImportError as err:
        raise ImportError(
            "Looking targets up in SIMBAD needs astroquery, which is not "
            "installed. Install it with: pip install 'virgil-astro[legacy]'"
        ) from err

    return simbad_class


def _scalar(value):
    """First element of a scalar, list or array (per-table metadata)."""
    return onp.ravel(onp.asarray(value))[0]


def rad2mas(rad):
    """Convert radians to milliarcseconds."""
    return rad / u.milliarcsec.to(u.rad)


def GetWavelength(ins, filt):
    """Return ``(wavelength, bandwidth)`` in metres for an instrument filter.

    Only JWST NIRISS AMI filters (``ins="JWST"``) are tabulated.
    """
    dic_filt = {
        "JWST": {
            "F277W": [2.776, 0.715],
            "F380M": [3.828, 0.205],
            "F430M": [4.286, 0.202],
            "F480M": [4.817, 0.298],
        }
    }

    wl = dic_filt[ins][filt][0] * 1e-6
    e_wl = dic_filt[ins][filt][1] * 1e-6

    return wl, e_wl


def Format_STAINDEX_V2(tab):
    """Return baseline station pairs as 1-based ``(n, 2)`` integers.

    Zero-based indices (any 0 present) are shifted up by one.
    """
    tab = onp.asarray(tab, dtype=int).reshape(-1, 2)
    return tab + 1 if tab.min() == 0 else tab  # RAC 2/2021


def Format_STAINDEX_T3(tab):
    """Return triangle station triplets as 1-based ``(n, 3)`` integers.

    Zero-based indices (any 0 present) are shifted up by one.
    """
    tab = onp.asarray(tab, dtype=int).reshape(-1, 3)
    return tab + 1 if tab.min() == 0 else tab


def save(dic, filename=None, datadir=None, verbose=False):
    """
    Save an ImPlaneIA dictionary as an OIFITS2 file.

    This is [`virgil.oifits.write_oifits`][virgil.oifits.write_oifits]
    plus the ImPlaneIA extras: the ``CTRS_EQT`` column and ``PSCALE``/``ISZ``
    keywords in ``OI_ARRAY``, a filename built from ``info``, and target
    coordinates queried from SIMBAD.

    Parameters
    ----------
    dic : dict
        ImPlaneIA dictionary with the tables ``OI_WAVELENGTH``, ``OI_VIS``,
        ``OI_VIS2`` and ``OI_T3`` and an ``info`` dict holding ``TARGET``,
        ``INSTRUME``, ``MASK``, ``FILT`` and ``MJD``, plus ``PSCALE``
        (mas/pixel), ``ISZ`` and ``STAXY``/``CTRS_EQT`` (in ``info`` or an
        ``OI_ARRAY`` table). Phases are in degrees. ``dic`` is not modified.
    filename : str or os.PathLike, optional
        Output filename. If omitted, the name is built from ``TARGET``,
        ``INSTRUME``, ``MASK``, ``FILT`` and ``MJD``.
    datadir : str or os.PathLike, optional
        Destination directory (default ``"Saveoifits/"``), created if needed.
    verbose : bool, optional
        If ``True``, print the path written.

    Notes
    -----
    Unless ``info["TARGET"]`` is ``"UNKNOWN"``, the target's coordinates,
    proper motion, parallax and spectral type are queried from SIMBAD over
    the network (this needs ``astroquery``).
    """
    from ..oifits import build_hdulist

    if dic is None:
        raise ValueError("save(): dic is None; nothing to write.")
    dic = copy.deepcopy(dic)
    info = dic["info"]
    info["MJD"] = _scalar(info["MJD"])
    array = dic.get("OI_ARRAY", {})
    staxy = onp.asarray(info.get("STAXY", array.get("STAXY")), dtype=float)
    ctrs_eqt = info.get("CTRS_EQT", array.get("CTRS_EQT"))
    info["STAXY"] = staxy

    name = str(info["TARGET"])
    if name != "UNKNOWN":
        # TODO: let callers pass coordinates instead of querying SIMBAD here.
        ra, dec, spectyp, pmra, pmdec, plx = _query_simbad(name)
        dic["OI_TARGET"] = {
            "TARGET_ID": [1],
            "TARGET": [name],
            "RAEP0": ra,
            "DECEP0": dec,
            "PMRA": pmra,
            "PMDEC": pmdec,
            "PARALLAX": plx,
            "SPECTYP": spectyp,
        }

    datadir = "Saveoifits/" if datadir is None else os.fspath(datadir)
    if not isinstance(filename, (str, os.PathLike)):
        filename = "%s_%s_%s_%s_%s.oifits" % (
            name.replace(" ", ""),
            info["INSTRUME"],
            info["MASK"],
            info["FILT"],
            info["MJD"],
        )
    hdul = build_hdulist(dic)

    # ImPlaneIA extensions of OI_ARRAY, read back by load().
    index = hdul.index_of("OI_ARRAY")
    hdu = hdul[index]
    columns = hdu.columns
    if ctrs_eqt is not None:
        columns = columns + fits.Column(
            name="CTRS_EQT",
            unit="METERS",
            format="2D",
            array=onp.asarray(ctrs_eqt, dtype=float).reshape(-1, 2),
        )
    new = fits.BinTableHDU.from_columns(columns, header=hdu.header)
    new.header["PSCALE"] = float(info["PSCALE"])  # [mas] RAC 9/2020
    new.header["ISZ"] = int(info["ISZ"])  # RAC 9/2020
    hdul[index] = new

    os.makedirs(datadir, exist_ok=True)
    path = os.path.join(datadir, filename)
    hdul.writeto(path, overwrite=True)
    if verbose:
        print(f"### OIFITS CREATED ({path}).")


def _query_simbad(name):
    """Return SIMBAD ``(ra, dec, spectyp, pmra, pmdec, plx)`` for OI_TARGET.

    Coordinates are in degrees, proper motions in deg/yr and the parallax in
    degrees, as OIFITS requires (SIMBAD gives mas/yr and mas). A failed
    query, such as a network error or an unknown target, gives zeros; a
    missing ``astroquery`` raises, with an install hint.
    """
    unknown = [0], [0], ["unknown"], [0], [0], [0]
    mas_to_deg = 1.0 / 3.6e6
    simbad = _simbad()
    try:
        custom_simbad = simbad()
        custom_simbad.add_votable_fields(
            "propermotions", "sp_type", "parallax"
        )
        query = custom_simbad.query_object(name)
        names = {col.lower(): col for col in query.colnames}

        def column(*candidates):
            for candidate in candidates:
                if candidate in names:
                    return query[names[candidate]][0]
            raise KeyError(candidates)

        ra, dec = column("ra"), column("dec")
        if isinstance(ra, str):
            # astroquery < 0.4.8 returns sexagesimal strings.
            coord = SkyCoord(ra + " " + dec, unit=(u.hourangle, u.deg))
            ra, dec = coord.ra.deg, coord.dec.deg
        return (
            [float(ra)],
            [float(dec)],
            [str(column("sp_type"))],
            [float(column("pmra")) * mas_to_deg],
            [float(column("pmdec")) * mas_to_deg],
            [float(column("plx_value")) * mas_to_deg],
        )
    except Exception:
        return unknown


def load(filename, target=None, ins=None, mask=None, include_vis=True):
    """Load an OIFITS file into an internal dictionary representation.

    Parameters
    ----------
    filename : str
        Name of the OIFITS file.
    target : str, optional
        Fallback target name if not present in headers.
    ins : str, optional
        Fallback instrument name if not present in headers.
    mask : str, optional
        Fallback mask name if not present in headers.
    include_vis : bool, optional
        If ``True`` (default), include the ``OI_VIS`` (visibility amplitude
        and phase) table when available.

    Returns
    -------
    dict
        ImPlaneIA dictionary of the tables and metadata. Only the last table
        of each type is kept, and phases stay in degrees. To fit the data,
        use [`virgil.oidata.OIData`][virgil.oidata.OIData] on the file instead.
    """
    with fits.open(filename, mode="readonly", memmap=False) as hdulist:
        fitsHandler = copy.deepcopy(hdulist)

        hdr = fitsHandler[0].header

        dic = {}
        dic["info"] = {}
        try:
            dic["info"]["TARGET"] = hdr["OBJECT"]
        except KeyError:
            dic["info"]["TARGET"] = target
        try:
            dic["info"]["OBJECT"] = hdr["OBJECT"]
        except KeyError:
            dic["info"]["OBJECT"] = None
        try:
            dic["info"]["INSTRUME"] = hdr["INSTRUME"]
        except KeyError:
            dic["info"]["INSTRUME"] = ins
        try:
            dic["info"]["MASK"] = hdr["MASK"]
        except KeyError:
            dic["info"]["MASK"] = mask
        try:
            dic["info"]["FILT"] = hdr["FILT"]
        except KeyError:
            dic["info"]["FILT"] = None
        try:
            dic["info"]["DATE-OBS"] = hdr["DATE-OBS"]
        except KeyError:
            dic["info"]["DATE-OBS"] = None
        try:
            dic["info"]["TELESCOP"] = hdr["TELESCOP"]
        except KeyError:
            dic["info"]["TELESCOP"] = (
                None  # try to get it from somewhere else?
            )
        try:
            dic["info"]["OBSERVER"] = hdr["OBSERVER"]
        except KeyError:
            dic["info"]["OBSERVER"] = None
        try:
            dic["info"]["INSMODE"] = hdr["INSMODE"]
        except KeyError:
            dic["info"]["INSMODE"] = None
        try:
            dic["info"]["PA"] = hdr["PA"]
        except KeyError:
            dic["info"]["PA"] = None

        for hdu in fitsHandler[1:]:
            extname = str(hdu.header.get("EXTNAME", "")).strip().upper()
            # RAC 9/2020
            # try to read in info from the OI_ARRAY required for re-saving
            if extname == "OI_ARRAY":
                # PSCALE, ISZ and CTRS_EQT are ImPlaneIA extensions.
                for key in ("PSCALE", "ISZ"):
                    if key in hdu.header:
                        dic["info"][key] = hdu.header[key]

                # make staxy from staxyz array (remove last column)
                staxyz = hdu.data["STAXYZ"]
                staxy = onp.delete(staxyz, -1, 1)
                dic["OI_ARRAY"] = {"STAXYZ": staxyz, "STAXY": staxy}
                for key in ("STA_INDEX", "TEL_NAME", "STA_NAME", "DIAMETER"):
                    if key in hdu.columns.names:
                        dic["OI_ARRAY"][key] = hdu.data[key]
                if "CTRS_EQT" in hdu.columns.names:
                    dic["OI_ARRAY"]["CTRS_EQT"] = hdu.data["CTRS_EQT"]

            if extname == "OI_WAVELENGTH":
                dic["OI_WAVELENGTH"] = {
                    "EFF_WAVE": hdu.data["EFF_WAVE"],
                    "EFF_BAND": hdu.data["EFF_BAND"],
                }

            if extname == "OI_VIS2":
                dic["OI_VIS2"] = {
                    "VIS2DATA": hdu.data["VIS2DATA"],
                    "VIS2ERR": hdu.data["VIS2ERR"],
                    "UCOORD": hdu.data["UCOORD"],
                    "VCOORD": hdu.data["VCOORD"],
                    "STA_INDEX": hdu.data["STA_INDEX"],
                    "MJD": hdu.data["MJD"],
                    "INT_TIME": hdu.data["INT_TIME"],
                    "TIME": hdu.data["TIME"],
                    "TARGET_ID": hdu.data["TARGET_ID"],
                    "FLAG": onp.array(hdu.data["FLAG"]),
                }
                # these are in every extension, but take them from here
                dic["info"]["MJD"] = hdu.data["MJD"][0]
                dic["info"]["ARRNAME"] = hdu.header.get("ARRNAME")
                try:
                    dic["OI_VIS2"]["BL"] = hdu.data["BL"]
                except KeyError:
                    dic["OI_VIS2"]["BL"] = (
                        hdu.data["UCOORD"] ** 2 + hdu.data["VCOORD"] ** 2
                    ) ** 0.5

            if extname == "OI_VIS" and include_vis:
                dic["OI_VIS"] = {
                    "TARGET_ID": hdu.data["TARGET_ID"],
                    "TIME": hdu.data["TIME"],
                    "MJD": hdu.data["MJD"],
                    "INT_TIME": hdu.data["INT_TIME"],
                    "VISAMP": hdu.data["VISAMP"],
                    "VISAMPERR": hdu.data["VISAMPERR"],
                    "VISPHI": hdu.data["VISPHI"],
                    "VISPHIERR": hdu.data["VISPHIERR"],
                    "UCOORD": hdu.data["UCOORD"],
                    "VCOORD": hdu.data["VCOORD"],
                    "STA_INDEX": hdu.data["STA_INDEX"],
                    "FLAG": hdu.data["FLAG"],
                }
                try:
                    dic["OI_VIS"]["BL"] = hdu.data["BL"]
                except KeyError:
                    dic["OI_VIS"]["BL"] = (
                        hdu.data["UCOORD"] ** 2 + hdu.data["VCOORD"] ** 2
                    ) ** 0.5

            if extname == "OI_T3":
                u1 = hdu.data["U1COORD"]
                u2 = hdu.data["U2COORD"]
                v1 = hdu.data["V1COORD"]
                v2 = hdu.data["V2COORD"]
                # Longest baseline of each triangle, in metres.
                bl_cp = onp.max(
                    onp.hypot([u1, u2, u1 + u2], [v1, v2, v1 + v2]), axis=0
                )

                dic["OI_T3"] = {
                    "T3PHI": hdu.data["T3PHI"],
                    "T3PHIERR": hdu.data["T3PHIERR"],
                    "T3AMP": hdu.data["T3AMP"],
                    "T3AMPERR": hdu.data["T3AMPERR"],
                    "U1COORD": hdu.data["U1COORD"],
                    "V1COORD": hdu.data["V1COORD"],
                    "U2COORD": hdu.data["U2COORD"],
                    "V2COORD": hdu.data["V2COORD"],
                    "STA_INDEX": hdu.data["STA_INDEX"],
                    "MJD": hdu.data["MJD"],
                    "FLAG": hdu.data["FLAG"],
                    "TARGET_ID": hdu.data["TARGET_ID"],
                    "TIME": hdu.data["TIME"],
                    "INT_TIME": hdu.data["INT_TIME"],
                }
                dic["OI_T3"]["BL"] = bl_cp
    del fitsHandler

    return dic
