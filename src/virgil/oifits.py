"""Read and write OIFITS files using only ``astropy.io.fits``.

[`read_oifits`][virgil.oifits.read_oifits] turns an OIFITS file (a path or
any astropy ``HDUList``) into the record that
[`virgil.oidata.OIData`][virgil.oidata.OIData] is built from, and
[`write_oifits`][virgil.oifits.write_oifits] writes an `OIData` or a dictionary
of tables to an OIFITS2 file. Each (baseline, wavelength) sample is one element of the flat
``u``, ``v`` and ``wavel`` arrays; closure phases index into them. See
[OIFITS files](../oifits_files.md) for the details.
"""

import datetime
import math
import os
import warnings
from collections.abc import Mapping

import numpy as onp
from astropy.io import fits
from ._closure import _merge
from ._deprecate import external_stacklevel, renamed

__all__ = ["build_hdulist", "read_oifits", "write_oifits"]


# A closure-phase row is matched to the nearest-MJD visibility row of the same
# exposure. OIFITS records only each row's MJD and INT_TIME, not the span of
# the exposure, so the MJDs may differ by up to twice the longest INT_TIME in
# the visibility table, and always by this much (in days, about 9 seconds) for files
# with no INT_TIME. GRAVITY's reduced products, for instance, average
# different subsets of an exposure's frames for T3 and VIS2 (8 x 30 s DITs
# with 120 s of valid frames can give MJDs 131 s apart). The closure phase
# then uses the visibility rows' (u, v), which have rotated slightly.
# TODO(Stage 6a.0): build closure-phase legs from the OI_T3 coordinates.
# Relative tolerance for treating two wavelength tables as identical.
_WAVE_RTOL = 1e-12
_MJD_TOLERANCE = 1e-4
# Frames the reader would merge (same MJD) are written this far apart in TIME [s].
_FRAME_STEP = 10 * _MJD_TOLERANCE * 86400.0
_INT16_MAX = 32767  # STA_INDEX is a 16-bit column in OIFITS2

_DEFAULT_PHASE_UNIT = "deg"


# === READING ===


def read_oifits(
    source,
    target=None,
    insname=None,
    frame_mjd="mean",
    extras=(),
    bandwidth=False,
):
    """Read an OIFITS file into the record `OIData(source)` is built from.

    Prefers `OI_VIS2` over amplitudes and `OI_T3` over absolute phases; see
    [OIFITS files](../oifits_files.md) for the tables read and how closure
    phases are matched.
    Source: Pauls et al. (2005), [references](../references.md#pauls2005);
    Duvert et al. (2015), [references](../references.md#duvert2017).

    Parameters
    ----------
    source : str, os.PathLike, astropy.io.fits.HDUList, or a list of them
        A path or an opened file. A list is read in order and concatenated.
    target : str or int, optional
        Target to keep, by name or `TARGET_ID`; needed for several targets,
        and a name for several files.
    insname : str or sequence of str, optional
        Keep only tables with this `INSNAME`; needed for several instruments
        (see [one instrument at a time](../oifits_files.md#insname)).
    frame_mjd : {"mean", "row"}, optional
        Give every sample of a frame the frame's mean `MJD` (default), or
        keep each row's own.
    extras : sequence of str, optional
        Any of `"flux"`, `"nflux"`, `"t3amp"`, `"visamp"`, `"visphi"` (see
        [extras](../oifits_files.md#extras)).
    bandwidth : bool, optional
        Also return each sample's `EFF_BAND` as `"bandwidth"`.

    Returns
    -------
    dict
        Flat per-sample `u`, `v`, `wavel` (metres), `vis`/`d_vis` (V², or
        amplitudes if `v2_flag` is False; one-sigma errors; `vis_flag` True = bad),
        `phi`/`d_phi` (radians, with `phi_flag`), `i_cps1`/`i_cps2`/`i_cps3`,
        `cp_flag`, `v2_flag`, `mjd`, `frame`, and a dict per requested extra.

    Raises
    ------
    ValueError
        For several targets or instruments without `target`/`insname`,
        differential or correlated-flux data, or unusable `OI_T3` coordinates.
    """
    extras = (extras,) if isinstance(extras, str) else tuple(extras)
    unknown = sorted(set(extras) - set(EXTRAS))
    if unknown:
        raise ValueError(f"Unknown extras {unknown}; choose from {EXTRAS}.")
    if "flux" in extras and "nflux" in extras:
        raise ValueError(
            "Read OI_FLUX either as 'flux' (up to a grey scale) or as "
            "'nflux' (normalized to its continuum), not both."
        )
    # An HDUList is itself a list (of HDUs): only other sequences are lists
    # of files.
    if isinstance(source, (list, tuple)) and not isinstance(
        source, fits.HDUList
    ):
        if not source:
            raise ValueError("read_oifits() got an empty list of files.")
        if frame_mjd not in ("mean", "row"):
            raise ValueError(
                f"frame_mjd must be 'mean' or 'row', not {frame_mjd!r}."
            )
        if target is not None and not isinstance(target, str):
            raise TypeError(
                "With several files, choose the target by name: TARGET_ID "
                f"values are local to each file, so target={target!r} could "
                "select different stars in different files."
            )
        return _concat_records(
            [
                read_oifits(s, target, insname, frame_mjd, extras, bandwidth)
                for s in source
            ]
        )
    if frame_mjd not in ("mean", "row"):
        raise ValueError(
            f"frame_mjd must be 'mean' or 'row', not {frame_mjd!r}."
        )
    if isinstance(source, (str, os.PathLike)):
        with fits.open(source, memmap=False) as hdul:
            return _read_hdulist(
                hdul, target, insname, frame_mjd, extras, bandwidth
            )
    return _read_hdulist(source, target, insname, frame_mjd, extras, bandwidth)


EXTRAS = ("flux", "nflux", "t3amp", "visamp", "visphi")


def _extname(hdu):
    return str(hdu.header.get("EXTNAME", "")).strip().upper()


def _lookup_key(hdu):
    """``(ARRNAME, INSNAME)`` of a data table, which scopes its STA_INDEX.

    Station numbers are only meaningful within one array, so baselines are
    matched within the same ``ARRNAME`` (empty when the table has none).
    """
    return (str(hdu.header.get("ARRNAME", "")).strip(), _insname(hdu))


def _insname(hdu):
    value = hdu.header.get("INSNAME")
    return None if value is None else str(value).strip()


# deliberate: a test seam (tests build tables directly)
def _collect_tables(hdul):
    """Group the OI_* tables of ``hdul`` by EXTNAME, in file order."""
    tables = {}
    for hdu in hdul:
        name = _extname(hdu)
        if name.startswith("OI_") and getattr(hdu, "data", None) is not None:
            tables.setdefault(name, []).append(hdu)
    return tables


_INSNAME_TABLES = ("OI_WAVELENGTH", "OI_VIS", "OI_VIS2", "OI_T3", "OI_FLUX")


# deliberate: a test seam (tests call it on hand-built tables)
def _select_insname(tables, insname):
    """Keep the tables with the chosen ``INSNAME`` (see ``read_oifits``)."""
    names = sorted(
        {
            _insname(hdu)
            for extname in _INSNAME_TABLES
            for hdu in tables.get(extname, [])
            if _insname(hdu) is not None
        }
    )
    if insname is None:
        fringe_tracker = [
            n for n in names if n.upper().startswith("GRAVITY_FT")
        ]
        science = [n for n in names if n.upper().startswith("GRAVITY_SC")]
        if fringe_tracker and science:
            raise ValueError(
                "This GRAVITY file holds fringe-tracker tables "
                f"{fringe_tracker} and science-channel tables {science}, "
                "which measure the same baselines and must not be merged. "
                "Choose with insname=, e.g. insname="
                f"{science[0]!r} (or both polarizations, {science!r})."
            )
        return tables
    wanted = {insname} if isinstance(insname, str) else set(insname)
    missing = sorted(wanted - set(names))
    if missing:
        raise ValueError(
            f"No tables have INSNAME {missing}; this file has {names}."
        )
    selected = {
        extname: [
            hdu
            for hdu in hdus
            if extname not in _INSNAME_TABLES or _insname(hdu) in wanted
        ]
        for extname, hdus in tables.items()
    }
    # Drop emptied table types: the reader decides which observables to use
    # from which table types are present.
    return {extname: hdus for extname, hdus in selected.items() if hdus}


def _table_wavelengths(hdu, wavelengths):
    ins = _insname(hdu)
    if ins in wavelengths:
        return wavelengths[ins]
    if len(wavelengths) == 1:
        # Non-standard files sometimes omit INSNAME; with one wavelength
        # table the match is unambiguous.
        return next(iter(wavelengths.values()))
    raise ValueError(
        f"{_extname(hdu)} table has INSNAME {ins!r}, which matches none of "
        f"the OI_WAVELENGTH tables {sorted(map(str, wavelengths))}."
    )


def _row_mask(hdu, target_id):
    n = len(hdu.data)
    if target_id is None or "TARGET_ID" not in hdu.columns.names:
        return onp.ones(n, dtype=bool)
    return onp.asarray(hdu.data["TARGET_ID"], dtype=int) == target_id


def _column(hdu, name, mask, nwave=None, dtype=float):
    values = onp.asarray(hdu.data[name], dtype=dtype)[mask]
    if nwave is None:
        return values
    return values.reshape(-1, nwave)


def _flags(hdu, mask, nwave, *arrays):
    """Combine the FLAG column with non-finite values into a bad-sample mask."""
    if "FLAG" in hdu.columns.names:
        flag = _column(hdu, "FLAG", mask, nwave, dtype=bool)
    else:
        flag = onp.zeros((int(mask.sum()), nwave), dtype=bool)
    for array in arrays:
        flag = flag | ~onp.isfinite(array)
    return flag


def _mjd(hdu, mask):
    if "MJD" in hdu.columns.names:
        return _column(hdu, "MJD", mask)
    return onp.zeros(int(mask.sum()))


def _time(hdu, mask):
    """The ``TIME`` column in days (zero where absent or not finite)."""
    if "TIME" in hdu.columns.names:
        return onp.nan_to_num(_column(hdu, "TIME", mask)) / 86400.0
    return onp.zeros(int(mask.sum()))


def _phase_unit_scale(unit, default, name="phase"):
    """Factor converting a phase in ``unit`` to radians.

    ``unit`` is a FITS ``TUNIT`` or a record's ``phi_unit``; a missing or
    blank one means ``default`` (degrees in OIFITS, radians in a record).
    """
    raw = default if unit is None or not str(unit).strip() else unit
    name_ = str(raw).strip().lower()
    if name_ in {"rad", "radian", "radians"}:
        return 1.0
    if name_ in {"deg", "degree", "degrees"}:
        return math.pi / 180.0
    raise ValueError(
        f"Unsupported phase unit {raw!r} for {name}; expected degrees or "
        "radians."
    )


def _phase_scale(hdu, column):
    """Factor converting a phase column to radians, from its TUNIT."""
    return _phase_unit_scale(
        hdu.columns[column].unit, _DEFAULT_PHASE_UNIT, column
    )


class _BaselineLookup:
    """Find the sample index of a baseline at a given epoch and channel.

    Also groups baseline rows into frames; see
    [frames](../oifits_files.md#frames).
    """

    def __init__(self):
        self._rows = {}
        self._order = []  # (start, nwave, mjd, ins) of every row, in order
        self._pairs = []  # the station pair of every row, in order
        self._parent = {}  # union-find over row starts
        self._waves = {}  # INSNAME -> wavelengths of its rows

    def add(self, ins, pair, mjd, exposure, start, nwave, wave=None, time=0.0):
        if wave is not None:
            self._waves.setdefault(ins, onp.asarray(wave, dtype=float))
        key = (ins, int(pair[0]), int(pair[1]))
        row = (float(mjd), exposure, start, nwave, float(time))
        self._rows.setdefault(key, []).append(row)
        self._order.append((start, nwave, float(mjd), ins, float(time)))
        self._pairs.append((int(pair[0]), int(pair[1])))
        self._parent[start] = start

    def _root(self, start):
        while self._parent[start] != start:
            self._parent[start] = self._parent[self._parent[start]]
            start = self._parent[start]
        return start

    def link(self, *starts):
        """Put the rows starting at ``starts`` in one frame."""
        roots = [self._root(start) for start in starts]
        for root in roots[1:]:
            self._parent[root] = roots[0]

    def times(self, frame_mjd):
        """Per-sample ``(mjd, frame)``; see ``read_oifits``."""
        # The same exposure only if MJD and TIME both agree (within
        # _MJD_TOLERANCE, as a Chebyshev distance), for rows of one INSNAME.
        # Rows sorted by MJD can interleave in TIME, so every pair counts,
        # not just neighbours; the KD-tree finds them in time proportional
        # to their number, not the square of the rows in an MJD window.
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components
        from scipy.spatial import cKDTree

        n = len(self._order)
        starts = onp.array([row[0] for row in self._order], dtype=int)
        position = {int(start): i for i, start in enumerate(starts)}
        edges = [
            (i, position[self._root(int(start))])
            for i, start in enumerate(starts)
        ]
        by_ins = {}
        for i, (_, _, mjd, ins, time) in enumerate(self._order):
            by_ins.setdefault(ins, []).append((i, mjd, time))
        for rows in by_ins.values():
            index, mjd, time = (onp.array(c) for c in zip(*rows))
            # A non-finite time never links (every comparison is False), and
            # the tree refuses it: it stays a frame of its own.
            ok = onp.isfinite(mjd) & onp.isfinite(time)
            index, mjd, time = index[ok], mjd[ok], time[ok]
            pairs = cKDTree(onp.column_stack([mjd, time])).query_pairs(
                _MJD_TOLERANCE, p=onp.inf, output_type="ndarray"
            )
            edges.extend(zip(index[pairs[:, 0]], index[pairs[:, 1]]))
        edges = onp.array(edges, dtype=int).reshape(-1, 2)
        graph = coo_matrix(
            (onp.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(n, n)
        )
        label = connected_components(graph, directed=False)[1]
        # Number the frames by the first row of each, and keep one union-find
        # root per frame, so that the links persist for later calls.
        first = onp.unique(label, return_index=True)[1]
        frame = onp.argsort(onp.argsort(first))[label] if n else label
        for start, lab in zip(starts, label):
            self._parent[int(start)] = int(starts[first[lab]])
        row_mjd = onp.array([mjd for _, _, mjd, _, _ in self._order])
        if frame_mjd == "mean":
            sums = onp.bincount(frame, weights=row_mjd)
            row_mjd = (sums / onp.bincount(frame))[frame]
        nwave = [n for _, n, _, _, _ in self._order]
        return onp.repeat(row_mjd, nwave), onp.repeat(frame, nwave)

    def stations(self):
        """Per-sample station pair ``(STA_INDEX)``, shape ``(n, 2)``."""
        nwave = [n for _, n, _, _, _ in self._order]
        return onp.repeat(onp.array(self._pairs, int).reshape(-1, 2), nwave, 0)

    def find(self, ins, pair, mjd, wave=None, time=0.0):
        """Return ``(start, nwave)`` of the nearest row, or None.

        Tried under ``ins`` (an ``(ARRNAME, INSNAME)`` pair) first, then under
        other ``INSNAME``s of the same array with identical wavelengths.
        """
        found = self._find_in(ins, pair, mjd, time)
        if found is not None or wave is None:
            return found
        wave = onp.asarray(wave, dtype=float)
        for other, other_wave in self._waves.items():
            if (
                other != ins
                and other[0] == ins[0]  # the same array (ARRNAME)
                and other_wave.shape == wave.shape
                and onp.allclose(other_wave, wave, rtol=_WAVE_RTOL, atol=0.0)
            ):
                found = self._find_in(other, pair, mjd, time)
                if found is not None:
                    return found
        return None

    def _find_in(self, ins, pair, mjd, time=0.0):
        rows = self._rows.get((ins, int(pair[0]), int(pair[1])), [])
        if not rows:
            return None
        # MJD and TIME must both agree (see ``_time``): keep the rows inside
        # their exposure window in both, then take the nearest of those.
        eligible = [
            row
            for row in rows
            if abs(row[0] - mjd) <= max(_MJD_TOLERANCE, 2.0 * row[1])
            and abs(row[4] - time) <= max(_MJD_TOLERANCE, 2.0 * row[1])
        ]
        if not eligible:
            return None
        best = min(
            eligible, key=lambda row: abs(row[0] - mjd) + abs(row[4] - time)
        )
        return best[2], best[3]


def _amptyp(hdu):
    return str(hdu.header.get("AMPTYP", "absolute")).strip().lower()


# deliberate: a named step: the visibility table and its baseline lookup
def _read_visibilities(tables, wavelengths, target_id, extras=()):
    if "OI_VIS2" in tables:
        names = ("OI_VIS2", "VIS2DATA", "VIS2ERR")
        v2_flag = True
    elif "OI_VIS" in tables:
        names = ("OI_VIS", "VISAMP", "VISAMPERR")
        v2_flag = False
    elif "OI_T3" in tables:
        # Closure-phase-only file: flagged baseline samples from the T3 legs.
        u, v, wavel = [], [], []
        lookup = _BaselineLookup()
        start = 0
        for hdu in tables["OI_T3"]:
            wave = _table_wavelengths(hdu, wavelengths)
            nwave = wave.size
            mask = _row_mask(hdu, target_id)
            u1, v1 = (
                _column(hdu, "U1COORD", mask),
                _column(hdu, "V1COORD", mask),
            )
            u2, v2 = (
                _column(hdu, "U2COORD", mask),
                _column(hdu, "V2COORD", mask),
            )
            sta_index = _column(hdu, "STA_INDEX", mask, dtype=int)
            mjd = _mjd(hdu, mask)
            time = _time(hdu, mask)
            ins = _lookup_key(hdu)
            for row, (a, b, c) in enumerate(sta_index):
                legs = (
                    ((a, b), u1[row], v1[row]),
                    ((b, c), u2[row], v2[row]),
                    ((a, c), u1[row] + u2[row], v1[row] + v2[row]),
                )
                for pair, uu, vv in legs:
                    if (
                        lookup.find(ins, pair, mjd[row], time=time[row])
                        is not None
                    ):
                        continue
                    # Each T3 row carries its own (u, v), so rows of different
                    # times stay separate samples: no exposure window here.
                    lookup.add(
                        ins, pair, mjd[row], 0.0, start, nwave, wave, time[row]
                    )
                    start += nwave
                    u.append(onp.full(nwave, uu))
                    v.append(onp.full(nwave, vv))
                    wavel.append(wave)

        n = start
        record = {
            "u": onp.concatenate(u),
            "v": onp.concatenate(v),
            "wavel": onp.concatenate(wavel),
            "vis": onp.full(n, onp.nan),
            "d_vis": onp.full(n, onp.nan),
            "vis_flag": onp.ones(n, dtype=bool),
            "v2_flag": True,
        }
        return record, lookup
    else:
        raise ValueError("OIFITS file has no OI_VIS2, OI_VIS or OI_T3 table.")
    extname, value_col, error_col = names

    u, v, wavel, vis, d_vis, vis_flag = [], [], [], [], [], []
    lookup = _BaselineLookup()
    start = 0
    for hdu in tables[extname]:
        for column in (value_col, error_col):
            if column not in hdu.columns.names:
                raise ValueError(f"{extname} table has no {column} column.")
        wave = _table_wavelengths(hdu, wavelengths)
        nwave = wave.size
        mask = _row_mask(hdu, target_id)
        values = _column(hdu, value_col, mask, nwave)
        errors = _column(hdu, error_col, mask, nwave)
        if extname == "OI_VIS" and onp.any(mask):
            if "visamp" in extras and _amptyp(hdu) != "absolute":
                # The amplitudes are an extra observable ("visamp"); the
                # rows only give the samples here.
                values = onp.full(values.shape, onp.nan)
            elif _amptyp(hdu) != "absolute":
                # Refuse a VISAMP that is not an absolute amplitude.
                raise ValueError(
                    f"VISAMP in this OI_VIS table has AMPTYP = {_amptyp(hdu)!r}, not "
                    "'absolute'. A differential visibility is normalized across "
                    "the band and a correlated flux is in flux units (e.g. MATISSE "
                    "products reduced with corrFlux=TRUE), so neither can be fitted "
                    "as a visibility amplitude. Calibrate the amplitudes into "
                    "visibilities first, read them as an extra observable "
                    "(extras=('visamp',)), or fit the closure phases (OI_T3) "
                    "alone by removing the OI_VIS table."
                )
        flag = _flags(hdu, mask, nwave, values, errors)
        ucoord = _column(hdu, "UCOORD", mask)
        vcoord = _column(hdu, "VCOORD", mask)
        sta_index = _column(hdu, "STA_INDEX", mask, dtype=int)
        mjd = _mjd(hdu, mask)
        time = _time(hdu, mask)
        # Longest INT_TIME in the table, in days (zero if there is none).
        exposure = 0.0
        if "INT_TIME" in hdu.columns.names and mask.any():
            exposure = float(onp.max(_column(hdu, "INT_TIME", mask))) / 86400.0
        ins = _lookup_key(hdu)
        for row in range(values.shape[0]):
            lookup.add(
                ins,
                sta_index[row],
                mjd[row],
                exposure,
                start,
                nwave,
                wave,
                time[row],
            )
            start += nwave
        u.append(onp.repeat(ucoord, nwave))
        v.append(onp.repeat(vcoord, nwave))
        wavel.append(onp.tile(wave, values.shape[0]))
        vis.append(values.reshape(-1))
        d_vis.append(errors.reshape(-1))
        vis_flag.append(flag.reshape(-1))

    record = {
        "u": onp.concatenate(u),
        "v": onp.concatenate(v),
        "wavel": onp.concatenate(wavel),
        "vis": onp.concatenate(vis),
        "d_vis": onp.concatenate(d_vis),
        "vis_flag": onp.concatenate(vis_flag),
        "v2_flag": v2_flag,
    }
    return record, lookup


# deliberate: a named step: the T3 coordinate error messages, away from the leg loop
def _check_t3_coordinates(legs_uv, founds, record, ins, triangle, mjd):
    """Refuse T3 leg coordinates that are NaN, zero or inconsistent with V2."""
    where = (
        f"Closure-phase triangle {tuple(int(x) for x in triangle)} (INSNAME "
        f"{ins[1]!r}, ARRNAME {ins[0]!r}, MJD {mjd})"
    )
    for uu, vv in legs_uv:
        if not (onp.isfinite(uu) and onp.isfinite(vv)) or (
            uu == 0 and vv == 0
        ):
            raise ValueError(
                f"{where} has OI_T3 coordinates ({uu}, {vv}) that are NaN "
                "or zero."
            )
    for j, found in enumerate(founds):
        if found is None or found[0] >= record["u"].size:
            continue
        ut, vt = legs_uv[j]
        us, vs = record["u"][found[0]], record["v"][found[0]]
        if onp.hypot(us - ut, vs - vt) > 0.1 * max(onp.hypot(us, vs), 1.0):
            raise ValueError(
                f"{where} has OI_T3 coordinates that disagree with the "
                f"visibility table for leg {j + 1}: ({ut}, {vt}) against "
                f"({us}, {vs}). The table may use the opposite baseline "
                "direction."
            )


# deliberate: a named step: matches T3 legs to samples and appends the unmatched
def _read_closure_phases(tables, wavelengths, target_id, lookup, record):
    """Closure phases, with the sample index of each triangle leg.

    Legs with no visibility row get flagged samples appended to ``record`` and
    registered in ``lookup``; see
    [closure-phase triangles](../oifits_files.md#closure-phases).
    """
    phi, d_phi, phi_flag = [], [], []
    i_cps = ([], [], [])
    extra = {"u": [], "v": [], "wavel": []}
    n_samples = record["u"].size
    n_unmatched = 0  # legs with no visibility row in either orientation
    for hdu in tables["OI_T3"]:
        wave = _table_wavelengths(hdu, wavelengths)
        nwave = wave.size
        mask = _row_mask(hdu, target_id)
        scale = _phase_scale(hdu, "T3PHI")
        values = _column(hdu, "T3PHI", mask, nwave) * scale
        errors = _column(hdu, "T3PHIERR", mask, nwave) * scale
        flag = _flags(hdu, mask, nwave, values, errors)
        sta_index = _column(hdu, "STA_INDEX", mask, dtype=int)
        mjd = _mjd(hdu, mask)
        time = _time(hdu, mask)
        ins = _lookup_key(hdu)
        channels = onp.arange(nwave)
        coords = (
            [
                _column(hdu, name, mask)
                for name in ("U1COORD", "V1COORD", "U2COORD", "V2COORD")
            ]
            if "U1COORD" in hdu.columns.names
            else None
        )
        for row, (a, b, c) in enumerate(sta_index):
            starts = []
            pairs = ((a, b), (b, c), (a, c))
            founds = []
            for pair in pairs:
                found = lookup.find(ins, pair, mjd[row], wave, time[row])
                if found is not None and found[1] != nwave:
                    found = None
                founds.append(found)
            legs_uv = None
            if coords is not None or None in founds:
                if coords is None:
                    coords = [
                        _column(hdu, name, mask)
                        for name in (
                            "U1COORD",
                            "V1COORD",
                            "U2COORD",
                            "V2COORD",
                        )
                    ]
                u1, v1, u2, v2 = (x[row] for x in coords)
                legs_uv = ((u1, v1), (u2, v2), (u1 + u2, v1 + v2))
                _check_t3_coordinates(
                    legs_uv, founds, record, ins, (a, b, c), mjd[row]
                )
            for k, (leg, pair) in enumerate(zip(i_cps, pairs)):
                found = founds[k]
                if found is None:
                    reverse = lookup.find(
                        ins, pair[::-1], mjd[row], wave, time[row]
                    )
                    if reverse is None or reverse[1] != nwave:
                        n_unmatched += 1
                    uu, vv = legs_uv[k]
                    lookup.add(
                        ins,
                        pair,
                        mjd[row],
                        0.0,
                        n_samples,
                        nwave,
                        wave,
                        time[row],
                    )
                    found = (n_samples, nwave)
                    n_samples += nwave
                    extra["u"].append(onp.full(nwave, uu))
                    extra["v"].append(onp.full(nwave, vv))
                    extra["wavel"].append(wave)
                leg.append(found[0] + channels)
                starts.append(found[0])
            lookup.link(*starts)
        phi.append(values.reshape(-1))
        d_phi.append(errors.reshape(-1))
        phi_flag.append(flag.reshape(-1))

    if n_unmatched:
        warnings.warn(
            f"{n_unmatched} closure-phase triangle "
            f"{'leg has' if n_unmatched == 1 else 'legs have'} no "
            "visibility row (V² or amplitude) with the same wavelengths at "
            "that time, in either orientation. Each is placed at the OI_T3 "
            "table's own (u, v) as a flagged sample: the closure phases are "
            "fitted, but the V² coverage of this file is incomplete.",
            UserWarning,
            stacklevel=external_stacklevel(),
        )
    if extra["u"]:
        n_extra = n_samples - record["u"].size
        for key in ("u", "v", "wavel"):
            record[key] = onp.concatenate([record[key], *extra[key]])
        nans = onp.full(n_extra, onp.nan)
        record["vis"] = onp.concatenate([record["vis"], nans])
        record["d_vis"] = onp.concatenate([record["d_vis"], nans])
        record["vis_flag"] = onp.concatenate(
            [record["vis_flag"], onp.ones(n_extra, dtype=bool)]
        )

    return {
        "phi": onp.concatenate(phi),
        "d_phi": onp.concatenate(d_phi),
        "phi_flag": onp.concatenate(phi_flag),
        "i_cps1": onp.concatenate(i_cps[0]).astype(int),
        "i_cps2": onp.concatenate(i_cps[1]).astype(int),
        "i_cps3": onp.concatenate(i_cps[2]).astype(int),
        "cp_flag": True,
    }


# deliberate: a named step: the OI_VIS phase branch of the reader
def _read_absolute_phases(tables, wavelengths, target_id, lookup, n_samples):
    phi = onp.zeros(n_samples)
    d_phi = onp.ones(n_samples)
    phi_flag = onp.ones(n_samples, dtype=bool)
    for hdu in tables["OI_VIS"]:
        if "VISPHI" not in hdu.columns.names:
            continue
        mask = _row_mask(hdu, target_id)
        if not onp.any(mask):
            continue  # another target's table
        phityp = str(hdu.header.get("PHITYP", "absolute")).strip().lower()
        if phityp != "absolute":
            raise ValueError(
                f"VISPHI in this OI_VIS table has PHITYP = {phityp!r}. A "
                "differential phase has had a mean phase and delay removed "
                "across the band, so it cannot be fitted as an absolute "
                "phase. Use the closure phases (OI_T3), or select a table "
                "with absolute phases."
            )
        wave = _table_wavelengths(hdu, wavelengths)
        nwave = wave.size
        scale = _phase_scale(hdu, "VISPHI")
        values = _column(hdu, "VISPHI", mask, nwave) * scale
        errors = _column(hdu, "VISPHIERR", mask, nwave) * scale
        flag = _flags(hdu, mask, nwave, values, errors)
        sta_index = _column(hdu, "STA_INDEX", mask, dtype=int)
        mjd = _mjd(hdu, mask)
        time = _time(hdu, mask)
        ins = _lookup_key(hdu)
        for row, pair in enumerate(sta_index):
            sign = 1.0
            found = lookup.find(ins, pair, mjd[row], time=time[row])
            if found is None:
                # The phase of the reversed baseline is the negated phase.
                found = lookup.find(ins, pair[::-1], mjd[row], time=time[row])
                sign = -1.0
            if found is None or found[1] != nwave:
                continue
            samples = found[0] + onp.arange(nwave)
            phi[samples] = sign * values[row]
            d_phi[samples] = errors[row]
            phi_flag[samples] = flag[row]
    return {
        "phi": phi,
        "d_phi": d_phi,
        "phi_flag": phi_flag,
        "i_cps1": None,
        "i_cps2": None,
        "i_cps3": None,
        "cp_flag": False,
    }


# deliberate: a named step: the extra observables, read after the main record
def _read_extras(tables, wavelengths, target_id, lookup, record, extras):
    """Read the requested extra observables into ``record`` (in place)."""
    if not extras:
        return
    new = _NewSamples(record)
    if "t3amp" in extras:
        # T3AMP per closure phase, in the order of ``_read_closure_phases``.
        if "OI_T3" not in tables:
            raise ValueError("extras 't3amp' needs an OI_T3 table.")
        out = {"value": [], "error": [], "flag": []}
        for hdu in tables["OI_T3"]:
            nwave = _table_wavelengths(hdu, wavelengths).size
            mask = _row_mask(hdu, target_id)
            for column in ("T3AMP", "T3AMPERR"):
                if column not in hdu.columns.names:
                    raise ValueError(f"OI_T3 table has no {column} column.")
            values = _column(hdu, "T3AMP", mask, nwave)
            errors = _column(hdu, "T3AMPERR", mask, nwave)
            flag = _flags(hdu, mask, nwave, values, errors)
            for key, x in zip(out, (values, errors, flag)):
                out[key].append(x.reshape(-1))
        record["t3amp"] = {key: onp.concatenate(x) for key, x in out.items()}
    if "visamp" in extras:
        if "OI_VIS" not in tables:
            raise ValueError("extras 'visamp' needs an OI_VIS table.")
        out = {"sample": [], "value": [], "error": [], "flag": []}
        kinds = set()
        for hdu, samples, _, values, errors, flag in _vis_rows(
            tables, wavelengths, target_id, lookup, new, "VISAMP", "VISAMPERR"
        ):
            amptyp = _amptyp(hdu)
            if amptyp == "absolute" and "OI_VIS2" not in tables:
                raise ValueError(
                    "VISAMP is already the visibility observable of this "
                    "file (it has no OI_VIS2); do not also read it as an "
                    "extra."
                )
            if amptyp not in ("absolute", "correlated flux"):
                raise ValueError(
                    f"VISAMP with AMPTYP = {amptyp!r} is not supported yet; "
                    "'absolute' and 'correlated flux' are."
                )
            kinds.add(amptyp)
            for key, x in zip(out, (samples, values, errors, flag)):
                out[key].append(x.reshape(-1))
        if len(kinds) != 1:
            raise ValueError(
                "extras 'visamp' needs OI_VIS tables of one AMPTYP, not "
                f"{sorted(kinds) or 'none'}."
            )
        record["visamp"] = {key: onp.concatenate(x) for key, x in out.items()}
        record["visamp"]["amptyp"] = kinds.pop()
    if "visphi" in extras:
        if not any(
            "VISPHI" in h.columns.names for h in tables.get("OI_VIS", [])
        ):
            raise ValueError(
                "extras 'visphi' needs an OI_VIS table with VISPHI."
            )
        out = {"sample": [], "value": [], "error": [], "flag": [], "row": []}
        n_row = 0
        for _, samples, sign, values, errors, flag in _vis_rows(
            tables, wavelengths, target_id, lookup, new, "VISPHI", "VISPHIERR"
        ):
            rows = n_row + onp.arange(samples.shape[0])
            n_row += samples.shape[0]
            parts = (
                samples,
                sign[:, None] * values,
                errors,
                flag,
                onp.broadcast_to(rows[:, None], samples.shape),
            )
            for key, x in zip(out, parts):
                out[key].append(onp.asarray(x).reshape(-1))
        record["visphi"] = {key: onp.concatenate(x) for key, x in out.items()}
    for kind in ("flux", "nflux"):
        if kind not in extras:
            continue
        # OI_FLUX spectra, one sample per (row, channel).
        if "OI_FLUX" not in tables:
            raise ValueError("extras 'flux'/'nflux' need an OI_FLUX table.")
        keys = ("wavel", "value", "error", "flag", "mjd", "row", "station")
        out = {key: [] for key in keys}
        n_row = 0
        for hdu in tables["OI_FLUX"]:
            mask = _row_mask(hdu, target_id)
            if not onp.any(mask):
                continue
            wave = _table_wavelengths(hdu, wavelengths)
            nwave = wave.size
            names = hdu.columns.names
            value_col = "FLUXDATA" if "FLUXDATA" in names else "FLUX"
            if value_col not in names or "FLUXERR" not in names:
                raise ValueError(
                    "OI_FLUX table has no FLUXDATA (or FLUX) and FLUXERR "
                    "columns."
                )
            values = _column(hdu, value_col, mask, nwave)
            errors = _column(hdu, "FLUXERR", mask, nwave)
            flag = _flags(hdu, mask, nwave, values, errors)
            nrow = values.shape[0]
            if "STA_INDEX" in names:
                station = _column(hdu, "STA_INDEX", mask, dtype=int)
                station = station.reshape(-1)
            else:
                station = onp.full(nrow, -1)
            rows = n_row + onp.arange(nrow)
            n_row += nrow
            parts = (
                onp.tile(wave, nrow),
                values,
                errors,
                flag,
                onp.repeat(_mjd(hdu, mask), nwave),
                onp.repeat(rows, nwave),
                onp.repeat(station[:nrow], nwave),
            )
            for key, x in zip(out, parts):
                out[key].append(onp.asarray(x).reshape(-1))
        if not out["value"]:
            raise ValueError("No OI_FLUX rows for this target.")
        record[kind] = {key: onp.concatenate(x) for key, x in out.items()}
    new.finish()


class _NewSamples:
    """Samples added for ``OI_VIS`` rows that have no visibility sample."""

    def __init__(self, record):
        self.record = record
        self.n = record["u"].size
        self.parts = {"u": [], "v": [], "wavel": []}

    def add(self, lookup, ins, pair, mjd, u, v, wave, time=0.0):
        nwave = wave.size
        lookup.add(ins, pair, mjd, 0.0, self.n, nwave, wave, time)
        start = self.n
        self.n += nwave
        self.parts["u"].append(onp.full(nwave, u))
        self.parts["v"].append(onp.full(nwave, v))
        self.parts["wavel"].append(wave)
        return start

    def finish(self):
        if not self.parts["u"]:
            return
        record = self.record
        n_new = self.n - record["u"].size
        wavel = onp.broadcast_to(
            onp.asarray(record["wavel"]), onp.shape(record["u"])
        )
        record["wavel"] = onp.concatenate([wavel, *self.parts["wavel"]])
        for key in ("u", "v"):
            record[key] = onp.concatenate([record[key], *self.parts[key]])
        nans = onp.full(n_new, onp.nan)
        record["vis"] = onp.concatenate([record["vis"], nans])
        record["d_vis"] = onp.concatenate([record["d_vis"], nans])
        record["vis_flag"] = onp.concatenate(
            [record["vis_flag"], onp.ones(n_new, dtype=bool)]
        )
        if not record["cp_flag"] and onp.size(record["phi"]):
            # Absolute phases are per sample: the new ones are unobserved.
            record["phi"] = onp.concatenate([record["phi"], onp.zeros(n_new)])
            record["d_phi"] = onp.concatenate(
                [record["d_phi"], onp.ones(n_new)]
            )
            record["phi_flag"] = onp.concatenate(
                [record["phi_flag"], onp.ones(n_new, dtype=bool)]
            )


def _vis_rows(tables, wavelengths, target_id, lookup, new, value, error):
    """Per-sample values of an ``OI_VIS`` column, matched to the samples.

    Yields ``(hdu, samples, sign, values, errors, flags)`` per table; ``sign``
    is -1 for a baseline stored reversed. A row with no sample gets a new
    flagged one.
    """
    for hdu in tables.get("OI_VIS", []):
        mask = _row_mask(hdu, target_id)
        if not onp.any(mask) or value not in hdu.columns.names:
            continue
        wave = _table_wavelengths(hdu, wavelengths)
        nwave = wave.size
        scale = _phase_scale(hdu, value) if value == "VISPHI" else 1.0
        values = _column(hdu, value, mask, nwave) * scale
        errors = _column(hdu, error, mask, nwave) * scale
        flag = _flags(hdu, mask, nwave, values, errors)
        sta_index = _column(hdu, "STA_INDEX", mask, dtype=int)
        ucoord = _column(hdu, "UCOORD", mask)
        vcoord = _column(hdu, "VCOORD", mask)
        mjd = _mjd(hdu, mask)
        time = _time(hdu, mask)
        ins = _lookup_key(hdu)
        samples = onp.zeros((len(sta_index), nwave), dtype=int)
        sign = onp.ones(len(sta_index))
        for row, pair in enumerate(sta_index):
            found = lookup.find(ins, pair, mjd[row], wave, time[row])
            if found is None or found[1] != nwave:
                found = lookup.find(ins, pair[::-1], mjd[row], wave, time[row])
                sign[row] = -1.0
            if found is None or found[1] != nwave:
                start = new.add(
                    lookup,
                    ins,
                    pair,
                    mjd[row],
                    ucoord[row],
                    vcoord[row],
                    wave,
                    time[row],
                )
                found, sign[row] = (start, nwave), 1.0
            samples[row] = found[0] + onp.arange(nwave)
        yield hdu, samples, sign, values, errors, flag


def _read_hdulist(
    hdul, target, insname=None, frame_mjd="mean", extras=(), bandwidth=False
):
    tables = _select_insname(_collect_tables(hdul), insname)
    wavelengths = {}
    for hdu in tables.get("OI_WAVELENGTH", []):
        wavelengths[_insname(hdu)] = onp.asarray(
            hdu.data["EFF_WAVE"], dtype=float
        ).reshape(-1)
    if not wavelengths:
        raise ValueError("OIFITS file has no OI_WAVELENGTH table.")
    data_names = ["OI_VIS2", "OI_VIS", "OI_T3"]
    if {"flux", "nflux"} & set(extras):
        data_names.append("OI_FLUX")
    # The TARGET_ID to keep, or None to keep every row.
    target_names = {}
    for hdu in tables.get("OI_TARGET", []):
        for tid, name in zip(hdu.data["TARGET_ID"], hdu.data["TARGET"]):
            target_names[int(tid)] = str(name).strip()
    ids = set()
    for name in data_names:
        for hdu in tables.get(name, []):
            if "TARGET_ID" in hdu.columns.names:
                ids.update(int(i) for i in onp.asarray(hdu.data["TARGET_ID"]))
    if target is None:
        if len(ids) > 1:
            listing = {i: target_names.get(i, "?") for i in sorted(ids)}
            raise ValueError(
                f"OIFITS file contains data on several targets {listing}; "
                "choose one with target=<name or TARGET_ID>."
            )
        target_id = None
    elif isinstance(target, str):
        matches = [
            tid
            for tid, name in target_names.items()
            if name.lower() == target.strip().lower()
        ]
        if not matches:
            raise ValueError(
                f"Target {target!r} is not in OI_TARGET "
                f"({sorted(target_names.values())})."
            )
        target_id = matches[0]
    else:
        target_id = int(target)

    record, lookup = _read_visibilities(tables, wavelengths, target_id, extras)
    if "OI_T3" in tables:
        record.update(
            _read_closure_phases(
                tables, wavelengths, target_id, lookup, record
            )
        )
    elif "visphi" not in extras and any(
        "VISPHI" in h.columns.names for h in tables.get("OI_VIS", [])
    ):
        record.update(
            _read_absolute_phases(
                tables, wavelengths, target_id, lookup, record["u"].size
            )
        )
    else:
        # Visibilities alone (e.g. V² without closure phases).
        record.update(
            phi=onp.zeros(0),
            d_phi=onp.zeros(0),
            phi_flag=onp.zeros(0, dtype=bool),
            i_cps1=None,
            i_cps2=None,
            i_cps3=None,
            cp_flag=False,
        )

    _read_extras(tables, wavelengths, target_id, lookup, record, extras)
    record["mjd"], record["frame"] = lookup.times(frame_mjd)
    record["stations"] = lookup.stations()
    widths = None
    if bandwidth:
        # Per-sample EFF_BAND, matched to channels by wavelength; files that
        # give no width, or only zeros, have none, so smearing stays off.
        by_wave = {}
        for hdu in tables.get("OI_WAVELENGTH", []):
            if "EFF_BAND" not in hdu.columns.names or hdu.header.get(
                "BANDEST"
            ):
                continue  # BANDEST: written as an estimate, not a measured width
            wave = onp.asarray(hdu.data["EFF_WAVE"], dtype=float).reshape(-1)
            band = onp.asarray(hdu.data["EFF_BAND"], dtype=float).reshape(-1)
            for w, b in zip(wave.tolist(), band.tolist()):
                if by_wave.setdefault(w, b) != b:
                    raise ValueError(
                        f"Two OI_WAVELENGTH channels at {w:.6g} m have "
                        "different EFF_BAND values (several instruments?): "
                        "choose one with insname=."
                    )
        wavel = onp.broadcast_to(
            onp.asarray(record["wavel"]), onp.shape(record["u"])
        )
        band = onp.array([by_wave.get(w, 0.0) for w in wavel.tolist()])
        if onp.all(onp.isfinite(band)) and onp.any(band > 0):
            widths = band
    if widths is not None:
        record["bandwidth"] = widths
    unique_wavel = onp.unique(record["wavel"])
    if unique_wavel.size == 1:
        record["wavel"] = unique_wavel
        if widths is not None and onp.unique(widths).size == 1:
            record["bandwidth"] = onp.unique(widths)
    record["phi_unit"] = "rad"
    return record


# deliberate: a named step: the multi-file reader's whole merge
def _concat_records(records):
    """Concatenate single-file records, sample by sample."""
    first = records[0]
    for key in ("v2_flag", "cp_flag"):
        kinds = {bool(record[key]) for record in records}
        if len(kinds) > 1:
            raise ValueError(
                f"Files disagree on {key}: they must all hold the same kinds "
                "of visibility and phase observables."
            )
    with_phases = {onp.size(record["phi"]) > 0 for record in records}
    if len(with_phases) > 1:
        raise ValueError(
            "Some files have phases and others do not: read them "
            "separately, or drop the phases, to combine them."
        )
    out = {
        key: onp.concatenate([onp.asarray(r[key]) for r in records])
        for key in ("u", "v", "vis", "d_vis", "vis_flag")
    }
    # Per-sample wavelengths, since files have their own channels.
    out["wavel"] = onp.concatenate(
        [
            onp.broadcast_to(onp.asarray(r["wavel"]), onp.shape(r["u"]))
            for r in records
        ]
    )
    for key in ("phi", "d_phi", "phi_flag", "mjd", "stations"):
        out[key] = onp.concatenate([onp.asarray(r[key]) for r in records])
    # Frames are numbered per file: shift them so files never share one.
    shifts = onp.cumsum([0] + [r["frame"].max() + 1 for r in records[:-1]])
    out["frame"] = onp.concatenate(
        [r["frame"] + shift for r, shift in zip(records, shifts)]
    )
    if first["cp_flag"]:
        offsets = onp.cumsum([0] + [onp.size(r["u"]) for r in records[:-1]])
        for key in ("i_cps1", "i_cps2", "i_cps3"):
            out[key] = onp.concatenate(
                [onp.asarray(r[key]) + off for r, off in zip(records, offsets)]
            ).astype(int)
    else:
        out.update(i_cps1=None, i_cps2=None, i_cps3=None)
    if any("bandwidth" in r for r in records):
        # A file without widths is unsmeared: width zero.
        out["bandwidth"] = onp.concatenate(
            [
                onp.broadcast_to(
                    onp.asarray(r.get("bandwidth", 0.0)), onp.shape(r["u"])
                )
                for r in records
            ]
        )
    unique_wavel = onp.unique(out["wavel"])
    if unique_wavel.size == 1:
        out["wavel"] = unique_wavel
        if "bandwidth" in out and onp.unique(out["bandwidth"]).size == 1:
            out["bandwidth"] = onp.unique(out["bandwidth"])
    out.update(
        v2_flag=first["v2_flag"], cp_flag=first["cp_flag"], phi_unit="rad"
    )
    sample_offsets = onp.cumsum([0] + [onp.size(r["u"]) for r in records[:-1]])
    for key in ("t3amp", "visamp", "visphi", "flux", "nflux"):
        present = [key in r for r in records]
        if not any(present):
            continue
        if not all(present):
            raise ValueError(
                f"Some files have the extra observable {key!r} and others "
                "do not: read them separately."
            )
        parts = [r[key] for r in records]
        row_offsets = onp.cumsum(
            [0] + [p["row"].max() + 1 if "row" in p else 0 for p in parts[:-1]]
        )
        merged = {}
        for name in parts[0]:
            if name == "amptyp":
                kinds = {p[name] for p in parts}
                if len(kinds) > 1:
                    raise ValueError(
                        f"Files disagree on the VISAMP AMPTYP: {sorted(kinds)}."
                    )
                merged[name] = kinds.pop()
                continue
            shift = {"sample": sample_offsets, "row": row_offsets}.get(name)
            merged[name] = onp.concatenate(
                [
                    onp.asarray(p[name]) + (0 if shift is None else shift[i])
                    for i, p in enumerate(parts)
                ]
            )
        out[key] = merged
    return out


# === WRITING ===

# (name, format, unit) of the per-row columns shared by the data tables.
_ROW_COLUMNS = (
    ("TARGET_ID", "1I", None),
    ("TIME", "1D", "s"),
    ("MJD", "1D", "d"),
    ("INT_TIME", "1D", "s"),
)

_TABLE_COLUMNS = {
    "OI_VIS2": (
        ("VIS2DATA", "data", None),
        ("VIS2ERR", "data", None),
        ("UCOORD", "1D", "m"),
        ("VCOORD", "1D", "m"),
        ("STA_INDEX", "2I", None),
    ),
    "OI_VIS": (
        ("VISAMP", "data", None),
        ("VISAMPERR", "data", None),
        ("VISPHI", "data", "deg"),
        ("VISPHIERR", "data", "deg"),
        ("UCOORD", "1D", "m"),
        ("VCOORD", "1D", "m"),
        ("STA_INDEX", "2I", None),
    ),
    "OI_T3": (
        ("T3AMP", "data", None),
        ("T3AMPERR", "data", None),
        ("T3PHI", "data", "deg"),
        ("T3PHIERR", "data", "deg"),
        ("U1COORD", "1D", "m"),
        ("V1COORD", "1D", "m"),
        ("U2COORD", "1D", "m"),
        ("V2COORD", "1D", "m"),
        ("STA_INDEX", "3I", None),
    ),
    "OI_FLUX": (
        ("FLUXDATA", "data", None),
        ("FLUXERR", "data", None),
        ("STA_INDEX", "1I", None),
    ),
}

# Header keywords a data table may carry in ``tables`` (with defaults).
_TABLE_KEYWORDS = {
    "OI_VIS": {"AMPTYP": "absolute", "PHITYP": "absolute"},
    "OI_FLUX": {"CALSTAT": "C"},
}

# Columns that may be omitted from the input and are then filled with NaN
# (and, for the data columns, flagged).
_OPTIONAL_DATA = {"T3AMP", "T3AMPERR"}


@renamed({"tables": "data"})
def write_oifits(data, filename, overwrite=True):
    """Write an `OIData` or a dictionary of OIFITS tables to an OIFITS2 file.

    An `OIData` is written as `OI_VIS2` with `OI_T3` or `OI_VIS` phases, and
    read back by [`read_oifits`][virgil.oifits.read_oifits] to the same
    observables, errors, flags, frames, wavelengths and (u, v), and the times
    if any (else MJD 0). Wavelengths are float32 (relative error 6e-8); frames
    at one `MJD` get distinct `TIME`s so they stay apart; closure phases
    without station indices get placeholder stations (`STAPLACE`). A
    dictionary of tables (columns as in
    [writing OIFITS](../oifits_files.md#writing), upper-case OIFITS names) has
    phases in **degrees**.
    Source: Duvert et al. (2015), [references](../references.md#duvert2017).

    Parameters
    ----------
    data : OIData or dict
        V² with closure or absolute phases, or a mapping of table name to a
        dict of columns (`OI_WAVELENGTH` and a data table).
    filename : str or os.PathLike
        Output path. Missing parent directories are created.
    overwrite : bool, optional
        Replace an existing file (default True).

    Returns
    -------
    pathlib.Path
        The path written.

    Raises
    ------
    TypeError
        If `data` is neither an `OIData` nor a mapping.
    ValueError
        If `data` is the flat record `read_oifits` returns (write
        `OIData(record)` instead), or an `OIData` has a layout OIFITS cannot
        hold (the message says which): extras, gains, projected or uv-grid
        data, a station index above 32767. No file is written.
    KeyError
        If a dictionary lacks `OI_WAVELENGTH` or a data table.
    OSError
        If the file exists and `overwrite` is False.
    """
    import pathlib

    if not isinstance(data, Mapping):
        if not all(hasattr(data, a) for a in ("u", "vis", "cp_flag")):
            raise TypeError(
                "write_oifits needs an OIData or a mapping of tables, not "
                f"{type(data).__name__}."
            )
        tables, reason = _oidata_tables(data)
        if tables is None:
            raise ValueError(
                f"These data cannot be written as OIFITS: {reason}."
            )
        data = tables
    elif "OI_WAVELENGTH" not in data and {"u", "vis"} <= set(data):
        raise ValueError(
            "This mapping is the flat record that read_oifits returns, not "
            "a dictionary of OIFITS tables. Write OIData(record) instead."
        )
    path = pathlib.Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    build_hdulist(data).writeto(path, overwrite=overwrite)
    return path


def build_hdulist(tables):
    """Build the OIFITS2 ``HDUList`` that :func:`write_oifits` writes.

    Useful for adding non-standard columns or keywords before writing.
    """
    info = dict(tables.get("info", {}))
    if "OI_WAVELENGTH" not in tables:
        raise KeyError("tables must contain an 'OI_WAVELENGTH' table.")
    data_tables = [n for n in ("OI_VIS", "OI_VIS2", "OI_T3") if n in tables]
    if not data_tables:
        raise KeyError(
            "tables must contain at least one of OI_VIS, OI_VIS2 or OI_T3."
        )

    insname = str(info.get("INSNAME", info.get("INSTRUME", "VIRGIL")))
    arrname = str(info.get("ARRNAME", info.get("MASK", "VIRGIL")))
    date_obs = str(info.get("DATE-OBS", "2000-01-01"))

    wave = tables["OI_WAVELENGTH"]
    eff_wave = onp.atleast_1d(onp.asarray(wave["EFF_WAVE"], dtype=float))
    nwave = eff_wave.size
    eff_band = onp.broadcast_to(
        onp.asarray(wave.get("EFF_BAND", 0.0), dtype=float), (nwave,)
    )
    estimated = not onp.all(eff_band > 0)
    if estimated:  # OIFITS2 needs EFF_BAND > 0: fill only the unknown widths
        # Channel widths from the spacing of the wavelengths (1% of each if
        # none; a duplicated wavelength has no spacing).
        waves = onp.abs(eff_wave)
        guess = 0.01 * waves
        if waves.size > 1:
            order = onp.argsort(waves)
            spacing = onp.empty_like(waves)
            spacing[order] = onp.gradient(waves[order])
            guess = onp.where(spacing > 0, spacing, guess)
        eff_band = onp.where(eff_band > 0, eff_band, guess)

    primary = fits.PrimaryHDU()
    header = primary.header
    header["CONTENT"] = "OIFITS2"
    header["ORIGIN"] = str(info.get("ORIGIN", "virgil"))
    header["DATE"] = datetime.date.today().isoformat()
    # Keywords that OIFITS2 requires in the primary header.
    header["DATE-OBS"] = date_obs
    header["OBJECT"] = str(info.get("OBJECT", info.get("TARGET", "UNKNOWN")))
    for key in ("TELESCOP", "INSTRUME", "OBSERVER", "INSMODE"):
        header[key] = str(info.get(key, "N/A"))
    for key, value in info.items():
        key = str(key).upper()
        if key in header or len(key) > 8:
            continue
        if isinstance(value, (str, bool, int, float, onp.integer)) or (
            isinstance(value, onp.floating)
        ):
            header[key] = value

    name = str(info.get("TARGET", info.get("OBJECT", "UNKNOWN")))
    defaults = {
        "TARGET_ID": ("1I", None, 1),
        "TARGET": ("32A", None, name),
        "RAEP0": ("1D", "deg", float(info.get("RA", 0.0))),
        "DECEP0": ("1D", "deg", float(info.get("DEC", 0.0))),
        "EQUINOX": ("1E", "yr", 2000.0),
        "RA_ERR": ("1D", "deg", 0.0),
        "DEC_ERR": ("1D", "deg", 0.0),
        "SYSVEL": ("1D", "m/s", 0.0),
        "VELTYP": ("8A", None, "UNKNOWN"),
        "VELDEF": ("8A", None, "OPTICAL"),
        "PMRA": ("1D", "deg/yr", 0.0),
        "PMDEC": ("1D", "deg/yr", 0.0),
        "PMRA_ERR": ("1D", "deg/yr", 0.0),
        "PMDEC_ERR": ("1D", "deg/yr", 0.0),
        "PARALLAX": ("1E", "deg", 0.0),
        "PARA_ERR": ("1E", "deg", 0.0),
        "SPECTYP": ("32A", None, str(info.get("SPECTYP", "UNKNOWN"))),
    }
    target = tables.get("OI_TARGET") or {}
    nrow = onp.atleast_1d(target.get("TARGET_ID", 1)).size
    columns = []
    for key, (fmt, unit, default) in defaults.items():
        values = onp.atleast_1d(target.get(key, default))
        values = onp.broadcast_to(values, (nrow,))
        columns.append(fits.Column(key, fmt, unit=unit, array=values))
    target_hdu = fits.BinTableHDU.from_columns(columns)
    _set_table_header(target_hdu, "OI_TARGET")

    hdus = [primary, target_hdu]
    hdus.append(
        _array_hdu(tables.get("OI_ARRAY"), info, arrname, tables, data_tables)
    )
    wave_hdu = fits.BinTableHDU.from_columns(
        [
            fits.Column("EFF_WAVE", "1E", unit="m", array=eff_wave),
            fits.Column("EFF_BAND", "1E", unit="m", array=eff_band),
        ]
    )
    _set_table_header(wave_hdu, "OI_WAVELENGTH", insname=insname)
    if estimated:
        wave_hdu.header["BANDEST"] = (True, "EFF_BAND estimated, not measured")
    hdus.append(wave_hdu)

    if "OI_FLUX" in tables:
        data_tables = data_tables + ["OI_FLUX"]
    for name in data_tables:
        hdu = _data_hdu(name, tables[name], info, nwave)
        _set_table_header(
            hdu,
            name,
            insname=insname,
            arrname=arrname,
            date_obs=date_obs,
        )
        for key, default in _TABLE_KEYWORDS.get(name, {}).items():
            hdu.header[key] = str(tables[name].get(key, default))
        hdus.append(hdu)

    return fits.HDUList(hdus)


def _set_table_header(hdu, extname, insname=None, arrname=None, date_obs=None):
    hdu.header["EXTNAME"] = extname
    hdu.header["OI_REVN"] = 2
    if date_obs is not None:
        hdu.header["DATE-OBS"] = date_obs
    if arrname is not None:
        hdu.header["ARRNAME"] = arrname
    if insname is not None:
        hdu.header["INSNAME"] = insname


# deliberate: a named step: the OI_ARRAY table with its two input forms
def _array_hdu(array, info, arrname, tables, data_tables):
    if array is not None:
        # Station positions are required; the other columns default, so an
        # array given by positions alone (e.g. from the legacy loader) works.
        if "STAXYZ" in array:
            staxyz = onp.asarray(array["STAXYZ"], dtype=float).reshape(-1, 3)
        else:
            staxy = onp.asarray(array["STAXY"], dtype=float).reshape(-1, 2)
            staxyz = onp.column_stack([staxy, onp.zeros(staxy.shape[0])])
        n = staxyz.shape[0]
        sta_index = onp.atleast_1d(
            onp.asarray(array.get("STA_INDEX", onp.arange(1, n + 1)), int)
        )
        if sta_index.size != n:
            raise ValueError(
                f"OI_ARRAY has {sta_index.size} STA_INDEX values for {n} "
                "stations."
            )
        tel_name = array.get("TEL_NAME", [f"T{i}" for i in sta_index])
        sta_name = array.get("STA_NAME", tel_name)
        diameter = array.get("DIAMETER", 0.0)
    else:
        if "STAXY" in info:
            staxy = onp.asarray(info["STAXY"], dtype=float).reshape(-1, 2)
            n = staxy.shape[0]
        else:
            n = max(
                int(onp.max(onp.asarray(tables[name]["STA_INDEX"])))
                for name in data_tables
            )
            staxy = onp.zeros((n, 2))
        staxyz = onp.column_stack([staxy, onp.zeros(n)])
        sta_index = onp.arange(1, n + 1)
        tel_name = [f"T{i}" for i in sta_index]
        sta_name = tel_name
        diameter = 0.0
    fov = 0.0
    if "PSCALE" in info and "ISZ" in info:
        # Radius of the extracted image: PSCALE [mas/pixel] * ISZ / 2.
        fov = float(info["PSCALE"]) / 1000.0 * float(info["ISZ"]) / 2.0
    columns = [
        fits.Column("TEL_NAME", "16A", array=onp.asarray(tel_name)),
        fits.Column("STA_NAME", "16A", array=onp.asarray(sta_name)),
        fits.Column("STA_INDEX", "1I", array=sta_index),
        fits.Column(
            "DIAMETER",
            "1E",
            unit="m",
            array=onp.broadcast_to(onp.asarray(diameter, float), (n,)),
        ),
        fits.Column("STAXYZ", "3D", unit="m", array=staxyz),
        fits.Column("FOV", "1D", unit="arcsec", array=onp.full(n, fov)),
        fits.Column("FOVTYPE", "6A", array=onp.full(n, "RADIUS")),
    ]
    hdu = fits.BinTableHDU.from_columns(columns)
    _set_table_header(hdu, "OI_ARRAY", arrname=arrname)
    hdu.header["FRAME"] = str(info.get("FRAME", "SKY"))
    for axis in ("ARRAYX", "ARRAYY", "ARRAYZ"):
        hdu.header[axis] = float(info.get(axis, 0.0))
    return hdu


# deliberate: a named step: one data table's columns, shared by every table
def _data_hdu(name, table, info, nwave):
    specs = _TABLE_COLUMNS[name]
    nrow = (
        onp.asarray(table["STA_INDEX"])
        .reshape(-1, {"OI_T3": 3, "OI_FLUX": 1}.get(name, 2))
        .shape[0]
    )
    defaults = {
        "TARGET_ID": 1,
        "TIME": 0.0,
        "MJD": info.get("MJD", 0.0),
        "INT_TIME": 0.0,
    }
    columns = []
    for key, fmt, unit in _ROW_COLUMNS:
        values = onp.asarray(table.get(key, defaults[key]))
        values = onp.broadcast_to(values.reshape(-1), (nrow,))
        if values.size != nrow:
            raise ValueError(f"{name}.{key} must have one value per row.")
        columns.append(fits.Column(key, fmt, unit=unit, array=values))

    flag = onp.zeros((nrow, nwave), dtype=bool)
    if "FLAG" in table:
        flag = onp.asarray(table["FLAG"], dtype=bool).reshape(nrow, nwave)
    for key, fmt, unit in specs:
        if fmt == "data":
            if key not in table and key in _OPTIONAL_DATA:
                values = onp.full((nrow, nwave), onp.nan)
            else:
                values = onp.asarray(table[key], dtype=float)
                values = values.reshape(nrow, nwave)
            fmt = f"{nwave}D"
        elif key == "STA_INDEX":
            values = onp.asarray(table[key], dtype=int).reshape(nrow, -1)
            if values.size and onp.max(onp.abs(values)) > _INT16_MAX:
                raise ValueError(
                    f"{name}.STA_INDEX has a value above {_INT16_MAX}, which "
                    "the 16-bit OIFITS2 column cannot hold."
                )
            if fmt == "1I":
                values = values.reshape(nrow)
        else:
            values = onp.asarray(table[key], dtype=float).reshape(nrow)
        columns.append(fits.Column(key, fmt, unit=unit, array=values))
    columns.append(fits.Column("FLAG", f"{nwave}L", array=flag))
    return fits.BinTableHDU.from_columns(columns)


# === WRITING AN OIDATA ===


_IRREGULAR_CP = "irregular closure-phase layout"
_IRREGULAR_WAVEL = "irregular wavelength layout"


def _oidata_tables(data):
    """OIFITS tables for [`write_oifits`][virgil.oifits.write_oifits] from an `OIData`.

    Squared visibilities with closure phases or absolute phases, one or more
    channels per baseline, flagged samples (written as flagged NaN rows), no
    projection and no extra observables. Returns ``(tables, None)`` with
    phases in degrees, or ``(None, reason)`` when the data cannot be written
    faithfully.
    """
    if data.vis_mat is not None or data.phi_mat is not None:
        return None, "projected observables (vis_mat/phi_mat)"
    if data.uv_grid is not None:
        return None, "uv-grid (AMIGO DISCO) data"
    if data.extras:
        return None, "extra observables"
    if data.gains is not None or data.phase_offsets is not None:
        return None, "calibration gains or closure offsets"
    if not data.v2_flag or data.vis_mode != "v2":
        return None, "visibility amplitudes rather than V²"
    layout, reason = _baseline_layout(data)
    if reason:
        return None, reason
    n_b, n_w = layout["n_b"], layout["n_w"]
    vis = onp.asarray(data.vis, float)
    d_vis = onp.asarray(data.d_vis, float)
    vis2 = {}
    if data.vis_index is not None:
        index = onp.asarray(data.vis_index, int)
        vis = _scattered(vis, index, n_b * n_w)
        d_vis = _scattered(d_vis, index, n_b * n_w)
        vis2["FLAG"] = onp.isnan(vis).reshape(n_b, n_w)
    tables = {
        "info": {"TARGET": "PIPELINE", "INSTRUME": "VIRGIL"},
        "OI_WAVELENGTH": {"EFF_WAVE": layout["waves"]},
        "OI_VIS2": {
            "VIS2DATA": vis.reshape(n_b, n_w),
            "VIS2ERR": d_vis.reshape(n_b, n_w),
            "UCOORD": layout["ub"],
            "VCOORD": layout["vb"],
            "STA_INDEX": layout["stations"],
            **vis2,
        },
    }
    if layout["bands"] is not None:
        tables["OI_WAVELENGTH"]["EFF_BAND"] = layout["bands"]
    tables["OI_VIS2"]["TIME"] = layout["time"]
    if layout["placeholders"]:
        tables["info"]["STAPLACE"] = True
    if layout["mjd"] is not None:
        tables["OI_VIS2"]["MJD"] = layout["mjd"]
        tables["info"]["MJD"] = float(layout["mjd"][0])
        first_day = int(float(onp.min(layout["mjd"])) // 1)
        tables["info"]["DATE-OBS"] = (
            datetime.date(1858, 11, 17) + datetime.timedelta(days=first_day)
        ).isoformat()
    if onp.asarray(data.phi, dtype=float).size == 0:
        return tables, None
    phi = onp.rad2deg(onp.asarray(data.phi, dtype=float))
    d_phi = onp.rad2deg(onp.asarray(data.d_phi, dtype=float))
    if data.cp_flag:
        stations = layout["stations"]
        closures = _closure_baselines(data, n_w)
        if closures is None:
            return None, _IRREGULAR_CP
        (b1, b2, b3), slots = closures
        if phi.size != slots.size:
            return None, _IRREGULAR_CP
        # virgil's closure phase is φ[i_cps1] + φ[i_cps2] - φ[i_cps3], i.e. the
        # baselines AB, BC and AC of the triangle ABC that OI_T3 stores.
        if not (
            onp.all(stations[b2, 0] == stations[b1, 1])
            and onp.all(stations[b3, 0] == stations[b1, 0])
            and onp.all(stations[b3, 1] == stations[b2, 1])
        ):
            return None, _IRREGULAR_CP
        n_t = b1.size
        t3 = {}
        if slots.size < n_t * n_w:
            phi = _scattered(phi, slots, n_t * n_w)
            d_phi = _scattered(d_phi, slots, n_t * n_w)
            t3["FLAG"] = onp.isnan(phi).reshape(n_t, n_w)
        ub, vb = layout["ub"], layout["vb"]
        table = {
            "T3PHI": phi.reshape(n_t, n_w),
            "T3PHIERR": d_phi.reshape(n_t, n_w),
            "U1COORD": ub[b1],
            "V1COORD": vb[b1],
            "U2COORD": ub[b2],
            "V2COORD": vb[b2],
            "STA_INDEX": onp.stack(
                [stations[b1, 0], stations[b1, 1], stations[b2, 1]], axis=1
            ),
            **t3,
        }
        table["TIME"] = layout["time"][b1]
        if layout["mjd"] is not None:
            table["MJD"] = layout["mjd"][b1]
        tables["OI_T3"] = table
        return tables, None
    table = {}
    if data.phi_index is not None:
        index = onp.asarray(data.phi_index, int)
        phi = _scattered(phi, index, n_b * n_w)
        d_phi = _scattered(d_phi, index, n_b * n_w)
        table["FLAG"] = onp.isnan(phi).reshape(n_b, n_w)
    if phi.size != n_b * n_w:
        return None, "absolute phases do not match the samples"
    table.update(
        {
            "VISPHI": phi.reshape(n_b, n_w),
            "VISPHIERR": d_phi.reshape(n_b, n_w),
            "VISAMP": onp.full((n_b, n_w), onp.nan),
            "VISAMPERR": onp.full((n_b, n_w), onp.nan),
            "UCOORD": layout["ub"],
            "VCOORD": layout["vb"],
            "STA_INDEX": layout["stations"],
        }
    )
    table["TIME"] = layout["time"]
    if layout["mjd"] is not None:
        table["MJD"] = layout["mjd"]
    tables["OI_VIS"] = table
    return tables, None


# deliberate: a named step: the one place that decides whether an OIData
# fits the OIFITS layout, with a reason for each way it cannot.
def _baseline_layout(data):
    """Samples as ``n_b`` baselines by ``n_w`` channels: ``(layout, reason)``.

    ``layout`` holds ``n_b``, ``n_w``, the channel wavelengths ``waves`` and
    widths ``bands``, the baseline coordinates ``ub``, ``vb``, the
    ``stations`` of every baseline and, if the data carry times, the ``mjd`` of
    every baseline.
    """
    u = onp.asarray(data.u, dtype=float)
    v = onp.asarray(data.v, dtype=float)
    wavel = onp.broadcast_to(onp.asarray(data.wavel, dtype=float), u.shape)
    waves = onp.asarray(list(dict.fromkeys(wavel.tolist())))
    n_w = waves.size
    if u.size % n_w:
        return None, _IRREGULAR_WAVEL
    n_b = u.size // n_w
    u, v = u.reshape(n_b, n_w), v.reshape(n_b, n_w)
    if not (
        onp.all(wavel.reshape(n_b, n_w) == waves[None, :])
        and onp.all(u == u[:, :1])
        and onp.all(v == v[:, :1])
    ):
        return None, _IRREGULAR_WAVEL
    bands = None  # unknown: written as an estimate (see build_hdulist)
    if data.bandwidth is not None:
        band = onp.broadcast_to(onp.asarray(data.bandwidth, float), u.size)
        band = band.reshape(n_b, n_w)
        if not onp.all(band == band[:1]):
            return None, "bandwidth differs between baselines of a channel"
        bands = band[0]
    n_vis = u.size if data.vis_index is None else data.vis_index.size
    if onp.asarray(data.vis).size != n_vis:
        return None, "visibilities do not match the samples"
    if data.stations is not None:
        stations = onp.asarray(data.stations, dtype=int)
        stations = stations.reshape(n_b, n_w, 2)[:, 0]
    elif not data.cp_flag:
        index = onp.arange(n_b)
        stations = onp.stack([2 * index + 1, 2 * index + 2], axis=1)
    else:
        stations, reason = _stations_from_triangles(data, n_b, n_w)
        if reason:
            return None, reason
    mjd = None
    if data.dt is not None and data.t_ref is not None:
        dt = onp.asarray(data.dt, dtype=float).reshape(n_b, n_w)
        mjd = (dt + data.t_ref)[:, 0]
    frame = onp.zeros(n_b, dtype=int)
    if data.frame is not None:
        frames = onp.asarray(data.frame).reshape(n_b, n_w)
        if not onp.all(frames == frames[:, :1]):
            return None, "a baseline's channels are in different frames"
        frame = frames[:, 0]
    # TIME [s] of every baseline that keeps its frames apart on reading. The
    # reader merges rows within _MJD_TOLERANCE in both MJD and TIME, so frames
    # at one MJD (all of them, for data without times) are written _FRAME_STEP
    # apart in TIME; other frames get 0.
    labels, inverse = onp.unique(frame, return_inverse=True)
    when = onp.zeros(labels.size)
    if mjd is not None:
        when = onp.array(
            [onp.mean(mjd[inverse == k]) for k in range(labels.size)]
        )
    offset = onp.zeros(labels.size)
    previous = None
    for k in onp.argsort(when, kind="stable"):
        if previous is not None and when[k] - when[previous] <= _MJD_TOLERANCE:
            offset[k] = offset[previous] + _FRAME_STEP
        previous = k
    time = offset[inverse]
    placeholders = data.stations is None
    if _repeated(stations, mjd, time):
        # The reader tells repeated baselines apart by station pair and time,
        # so give them placeholder stations when the data have no times.
        placeholders = True
        if data.cp_flag:
            stations, reason = _stations_from_triangles(data, n_b, n_w)
            if reason:
                return None, reason
        else:
            index = onp.arange(n_b)
            stations = onp.stack([2 * index + 1, 2 * index + 2], axis=1)
        if _repeated(stations, mjd, time):
            return None, "repeated baselines without distinct times"
    if stations.size and onp.max(stations) > _INT16_MAX:
        return None, f"a station index above {_INT16_MAX} (16-bit column)"
    return {
        "n_b": n_b,
        "n_w": n_w,
        "waves": waves,
        "bands": bands,
        "time": time,
        "placeholders": placeholders,
        "ub": u[:, 0],
        "vb": v[:, 0],
        "stations": stations,
        "mjd": mjd,
    }, None


def _stations_from_triangles(data, n_b, n_w):
    """Placeholder stations chaining the triangles, or a reason."""
    closures = _closure_baselines(data, n_w)
    if closures is None:
        return None, _IRREGULAR_CP
    b1, b2, b3 = closures[0]
    # Each baseline has two ends; a triangle identifies end B of AB with end B
    # of BC, end A of AB with end A of AC, and end C of BC with end C of AC.
    # Each class of identified ends is one station, numbered from 1.
    roots = _merge(
        2 * n_b,
        (
            pair
            for x, y, z in zip(b1, b2, b3)
            for pair in (
                (2 * x + 1, 2 * y),
                (2 * x, 2 * z),
                (2 * y + 1, 2 * z + 1),
            )
        ),
    ).tolist()
    number = {r: k + 1 for k, r in enumerate(dict.fromkeys(roots))}
    stations = onp.array([number[r] for r in roots]).reshape(n_b, 2)
    if onp.any(
        stations[:, 0] == stations[:, 1]
    ):  # a leg joins a station to itself
        return None, _IRREGULAR_CP
    return stations, None


def _repeated(stations, mjd, time):
    """Whether two baselines share a station pair, ``MJD`` and ``TIME``."""
    key = onp.column_stack(
        [stations, onp.zeros(len(stations)) if mjd is None else mjd, time]
    )
    return len(onp.unique(key, axis=0)) < len(stations)


def _scattered(values, index, n):
    """``values`` placed at ``index`` of ``n`` samples, NaN elsewhere."""
    out = onp.full(n, onp.nan)
    out[index] = values
    return out


def _closure_baselines(data, n_w):
    """Baselines ``(b1, b2, b3)`` of each triangle and flat slots, or ``None``.

    Each leg indexes the flat samples as ``baseline * n_w + channel`` and
    must run through increasing channels within one baseline triplet.
    """
    legs = [
        onp.asarray(i, dtype=int)
        for i in (data.i_cps1, data.i_cps2, data.i_cps3)
    ]
    n_samples = onp.asarray(data.u).size
    if not legs[0].size or any(
        leg.size != legs[0].size or onp.any((leg < 0) | (leg >= n_samples))
        for leg in legs
    ):
        return None
    rows, cols = [leg // n_w for leg in legs], [leg % n_w for leg in legs]
    if not (
        onp.array_equal(cols[0], cols[1]) and onp.array_equal(cols[0], cols[2])
    ):
        return None
    triangles, slots, prev_c = [], [], n_w
    for t, c in zip(zip(*(r.tolist() for r in rows)), cols[0].tolist()):
        if not triangles or t != triangles[-1]:
            triangles.append(t)
        elif c <= prev_c:
            return None
        slots.append((len(triangles) - 1) * n_w + c)
        prev_c = c
    if len(set(triangles)) < len(triangles):
        return None
    return (
        [onp.array(b, dtype=int) for b in zip(*triangles)],
        onp.array(slots, dtype=int),
    )
