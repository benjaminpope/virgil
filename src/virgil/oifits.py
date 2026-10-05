"""Read and write OIFITS files using only ``astropy.io.fits``.

This is the maintained OIFITS path of virgil.

* [`read_oifits`][virgil.oifits.read_oifits] turns an OIFITS file into the record that
  [`virgil.oidata.OIData`][virgil.oidata.OIData] is built from. It accepts a file path or
  any astropy ``HDUList``, including ``pyoifits`` objects (which subclass
  it). It handles several wavelength channels, several OIFITS tables, several
  epochs, and ``FLAG`` columns.
* [`write_oifits`][virgil.oifits.write_oifits] writes a dictionary of OIFITS tables (the layout used
  by the legacy [`virgil.legacy.oifits_implaneia`][virgil.legacy.oifits_implaneia] writer) to an OIFITS2 file.

Each (baseline, wavelength) sample becomes one element of the flat ``u``,
``v`` and ``wavel`` arrays of the record. Samples are ordered by table, then
row, then wavelength channel. Closure phases index into these samples, so a
multi-wavelength closure phase is formed from visibilities at its own
wavelength.
"""

import datetime
import os

import numpy as onp
from astropy.io import fits


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

_DEFAULT_PHASE_UNIT = "deg"


# === READING ===


def read_oifits(
    source, target=None, insname=None, frame_mjd="mean", extras=()
):
    """Read an OIFITS file into a record for [`OIData`][virgil.oidata.OIData].

    Parameters
    ----------
    source : str, os.PathLike, astropy.io.fits.HDUList, or a list of them
        File path, or an opened file (e.g. from ``astropy.io.fits.open`` or
        ``pyoifits.open``). A list or tuple of files is read file by file and
        concatenated into one record, in the order given.
    target : str or int, optional
        Target to keep, by ``OI_TARGET`` name or ``TARGET_ID``. Required when
        a file contains data on more than one target. With several files it
        must be the name, since ``TARGET_ID`` numbering is per file.
    insname : str or sequence of str, optional
        Keep only the tables (``OI_WAVELENGTH``, ``OI_VIS``, ``OI_VIS2``,
        ``OI_T3``, ``OI_FLUX``) with this ``INSNAME``, or one of these. A
        GRAVITY product holds fringe-tracker tables (``GRAVITY_FT``, a few
        low-resolution channels) beside the science channel (``GRAVITY_SC``,
        or ``GRAVITY_SC_P1``/``_P2`` in split polarisation). These are
        different measurements of the same baselines and must not be
        merged: reading such a file without ``insname`` raises an error.
        The two polarisations of the science channel are independent
        measurements and may be read together.
    frame_mjd : {"mean", "row"}, optional
        The time given to each sample. ``"mean"`` (the default) gives every
        sample of a frame (one exposure; see Notes) the mean ``MJD`` of the
        frame's rows; ``"row"`` keeps each row's own ``MJD``.
    extras : sequence of str, optional
        Further observables to read beside the visibilities and phases
        (none by default, so existing analyses are unchanged); see
        [`virgil.observables`][virgil.observables]:

        * ``"flux"``: ``OI_FLUX`` as a spectrum known up to a grey scale
          (``FLUXDATA``, or GRAVITY's ``FLUX``);
        * ``"nflux"``: ``OI_FLUX`` as a spectrum normalised to its
          continuum (choose one of ``"flux"`` and ``"nflux"``);
        * ``"t3amp"``: the triple amplitudes ``T3AMP`` of ``OI_T3``;
        * ``"visamp"``: ``OI_VIS`` ``VISAMP`` beside ``OI_VIS2``, as the
          table's ``AMPTYP`` declares it (``'absolute'``: |V|;
          ``'correlated flux'``: |V| times the total flux, up to a grey
          scale);
        * ``"visphi"``: ``OI_VIS`` ``VISPHI`` as a differential phase,
          whatever its ``PHITYP``, beside any closure phases.

    Returns
    -------
    dict
        Record with flat per-sample arrays ``u``, ``v`` (metres) and
        ``wavel`` (metres; a single element if every sample shares one
        wavelength), the visibility observables ``vis``/``d_vis`` with a
        boolean ``vis_flag`` (True = bad), the phases ``phi``/``d_phi`` in
        radians with ``phi_flag``, the closure-phase indices
        ``i_cps1``/``i_cps2``/``i_cps3`` (or ``None`` for absolute phases),
        the flags ``v2_flag`` and ``cp_flag``, and per sample the time
        ``mjd`` (days, float64) and the integer ``frame``. Each requested
        extra observable adds a dictionary under its name (``"flux"`` or
        ``"nflux"``, ``"t3amp"``, ``"visamp"``, ``"visphi"``).

    Notes
    -----
    Squared visibilities (``OI_VIS2``) are preferred over amplitudes
    (``OI_VIS``), and closure phases (``OI_T3``) over absolute phases
    (``OI_VIS`` ``VISPHI``). ``VISAMP`` is read only when its table's
    ``AMPTYP`` is ``'absolute'``, and ``VISPHI`` only when its
    ``PHITYP`` is ``'absolute'``; a missing keyword counts as
    ``'absolute'``, as in OIFITS1. Differential amplitudes and phases,
    and correlated fluxes, raise a ``ValueError``. Nothing in the
    standard marks ``OI_VIS2`` ``VIS2DATA`` that holds squared correlated
    flux rather than squared visibility, as in MATISSE products reduced
    with ``corrFlux=TRUE``; such data are read as visibilities, so
    calibrate them (e.g. with virgil-vlti) before fitting. A file with
    only ``OI_T3`` gives closure phases alone: its baselines come from the
    triangle coordinates, and ``vis`` is empty. A file with neither
    ``OI_T3`` nor ``VISPHI`` gives visibilities alone: ``phi`` is empty, and
    files with and without phases cannot be read together. Samples are
    flagged when their ``FLAG`` is set or their value or uncertainty is
    not finite.

    Each closure-phase triangle ``(a, b, c)`` is matched to the visibility
    baselines ``(a, b)``, ``(b, c)`` and ``(a, c)`` with the same ``INSNAME``
    (or, failing that, any ``INSNAME`` with identical wavelengths) and nearest
    ``MJD`` and ``TIME``, within its own file. The MJDs (and TIMEs) must agree to within
    twice the longest ``INT_TIME`` in the visibility table (or about 9
    seconds if there is none), since pipelines such as GRAVITY's average different
    frames of one exposure for each table. A baseline stored reversed, ``(b, a)``,
    is used as the conjugate: the closure phase gets an extra flagged sample
    at the triangle leg's own ``(u, v)``.

    A frame is one exposure of one instrument: the baselines that closure
    phases tie together, together with any rows of the same ``INSNAME`` at
    the same ``MJD`` and ``TIME`` (each within about 9 seconds). Both are
    compared because some OIFITS v1 writers (e.g. OYSTER) give a night one
    ``MJD`` and put each snapshot in ``TIME``; closure phases are correlated
    only within a snapshot. Frames are numbered within the record, so
    different files never share one.

    All files in a list must hold the same kinds of observable (squared
    visibilities or amplitudes; closure or absolute phases; the same
    extras).

    With ``extras``, a ``VISPHI`` or ``VISAMP`` row is matched to the
    visibility sample of its baseline and time as closure-phase legs are
    (a reversed baseline negates the phase). ``VISPHI`` is then never read
    as an absolute phase: without ``OI_T3`` the phase block is empty. A
    file whose ``OI_VIS`` holds correlated fluxes and no ``OI_VIS2`` gives
    flagged visibility samples, with the amplitudes in ``"visamp"``.
    """
    extras = _check_extras(extras)
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
                read_oifits(s, target, insname, frame_mjd, extras)
                for s in source
            ]
        )
    if frame_mjd not in ("mean", "row"):
        raise ValueError(
            f"frame_mjd must be 'mean' or 'row', not {frame_mjd!r}."
        )
    if isinstance(source, (str, os.PathLike)):
        with fits.open(source, memmap=False) as hdul:
            return _read_hdulist(hdul, target, insname, frame_mjd, extras)
    return _read_hdulist(source, target, insname, frame_mjd, extras)


EXTRAS = ("flux", "nflux", "t3amp", "visamp", "visphi")


def _check_extras(extras):
    if isinstance(extras, str):
        extras = (extras,)
    extras = tuple(extras)
    unknown = sorted(set(extras) - set(EXTRAS))
    if unknown:
        raise ValueError(f"Unknown extras {unknown}; choose from {EXTRAS}.")
    if "flux" in extras and "nflux" in extras:
        raise ValueError(
            "Read OI_FLUX either as 'flux' (up to a grey scale) or as "
            "'nflux' (normalised to its continuum), not both."
        )
    return extras


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


def _collect_tables(hdul):
    """Group the OI_* tables of ``hdul`` by EXTNAME, in file order."""
    tables = {}
    for hdu in hdul:
        name = _extname(hdu)
        if name.startswith("OI_") and getattr(hdu, "data", None) is not None:
            tables.setdefault(name, []).append(hdu)
    return tables


_INSNAME_TABLES = ("OI_WAVELENGTH", "OI_VIS", "OI_VIS2", "OI_T3", "OI_FLUX")


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
                f"{science[0]!r} (or both polarisations, {science!r})."
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


def _wavelength_tables(tables):
    wavelengths = {}
    for hdu in tables.get("OI_WAVELENGTH", []):
        wavelengths[_insname(hdu)] = onp.asarray(
            hdu.data["EFF_WAVE"], dtype=float
        ).reshape(-1)
    if not wavelengths:
        raise ValueError("OIFITS file has no OI_WAVELENGTH table.")
    return wavelengths


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


def _target_ids(tables, names):
    ids = set()
    for name in names:
        for hdu in tables.get(name, []):
            if "TARGET_ID" in hdu.columns.names:
                ids.update(int(i) for i in onp.asarray(hdu.data["TARGET_ID"]))
    return ids


def _select_target(tables, target, data_names):
    """Return the TARGET_ID to keep, or ``None`` to keep every row."""
    target_names = {}
    for hdu in tables.get("OI_TARGET", []):
        for tid, name in zip(hdu.data["TARGET_ID"], hdu.data["TARGET"]):
            target_names[int(tid)] = str(name).strip()
    ids = _target_ids(tables, data_names)
    if target is None:
        if len(ids) > 1:
            listing = {i: target_names.get(i, "?") for i in sorted(ids)}
            raise ValueError(
                f"OIFITS file contains data on several targets {listing}; "
                "choose one with target=<name or TARGET_ID>."
            )
        return None
    if isinstance(target, str):
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
        return matches[0]
    return int(target)


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
    """The ``TIME`` column in days (zero where absent or not finite).

    OIFITS v1 gives each row a ``TIME`` (UTC seconds) and an ``MJD``. Some
    writers set ``MJD`` to the date of the night and put the snapshot in
    ``TIME``, so two rows are one exposure only if both agree.
    """
    if "TIME" in hdu.columns.names:
        return onp.nan_to_num(_column(hdu, "TIME", mask)) / 86400.0
    return onp.zeros(int(mask.sum()))


def _exposure_time(hdu, mask):
    """Longest ``INT_TIME`` in a table, in days (zero if there is none)."""
    if "INT_TIME" in hdu.columns.names and mask.any():
        return float(onp.max(_column(hdu, "INT_TIME", mask))) / 86400.0
    return 0.0


def _phase_scale(hdu, column):
    """Factor converting a phase column to radians, from its TUNIT."""
    unit = hdu.columns[column].unit or _DEFAULT_PHASE_UNIT
    unit = str(unit).strip().lower()
    if unit in {"rad", "radian", "radians"}:
        return 1.0
    if unit in {"deg", "degree", "degrees"}:
        return onp.pi / 180.0
    raise ValueError(
        f"Unsupported unit {unit!r} for {column}; expected degrees or radians."
    )


class _BaselineLookup:
    """Find the sample index of a baseline at a given epoch and channel.

    It also groups the baseline rows into frames (exposures): rows that a
    closure phase ties together (``link``), and rows of one ``INSNAME`` at
    the same ``MJD`` and ``TIME``.
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
        by_ins = {}
        for start, _, mjd, ins, time in self._order:
            by_ins.setdefault(ins, []).append((mjd, time, start))
        for rows in by_ins.values():
            rows.sort()
            # The same exposure only if MJD and TIME both agree. Rows sorted
            # by MJD can interleave in TIME, so compare every pair inside the
            # MJD window, not just neighbours.
            for i, (mjd0, t0, a) in enumerate(rows):
                for mjd1, t1, b in rows[i + 1 :]:
                    if mjd1 - mjd0 > _MJD_TOLERANCE:
                        break
                    if abs(t1 - t0) <= _MJD_TOLERANCE:
                        self.link(a, b)
        roots = [self._root(start) for start, *_ in self._order]
        labels = {root: k for k, root in enumerate(dict.fromkeys(roots))}
        frame = onp.array([labels[root] for root in roots])
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
        """Return ``(start, nwave)`` of the nearest-epoch row, or ``None``.

        The row must be from the same exposure: see ``_MJD_TOLERANCE``. It
        is looked up under ``ins``, an ``(ARRNAME, INSNAME)`` pair, first.
        With ``wave``, other ``INSNAME``s of the same array whose
        wavelengths equal ``wave`` are tried next: the standard links every
        table to an ``OI_WAVELENGTH`` table by ``INSNAME`` but does not
        require the V² and T3 tables to share one. Station numbers belong to
        an array, so tables of another ``ARRNAME`` are never used.
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


def _check_amptyp(hdu):
    """Refuse a ``VISAMP`` that is not an absolute visibility amplitude."""
    amptyp = _amptyp(hdu)
    if amptyp != "absolute":
        raise ValueError(
            f"VISAMP in this OI_VIS table has AMPTYP = {amptyp!r}, not "
            "'absolute'. A differential visibility is normalised across "
            "the band and a correlated flux is in flux units (e.g. MATISSE "
            "products reduced with corrFlux=TRUE), so neither can be fitted "
            "as a visibility amplitude. Calibrate the amplitudes into "
            "visibilities first, read them as an extra observable "
            "(extras=('visamp',)), or fit the closure phases (OI_T3) "
            "alone by removing the OI_VIS table."
        )


def _read_visibilities(tables, wavelengths, target_id, extras=()):
    if "OI_VIS2" in tables:
        names = ("OI_VIS2", "VIS2DATA", "VIS2ERR")
        v2_flag = True
    elif "OI_VIS" in tables:
        names = ("OI_VIS", "VISAMP", "VISAMPERR")
        v2_flag = False
    elif "OI_T3" in tables:
        return _baselines_from_triangles(tables, wavelengths, target_id)
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
            else:
                _check_amptyp(hdu)
        flag = _flags(hdu, mask, nwave, values, errors)
        ucoord = _column(hdu, "UCOORD", mask)
        vcoord = _column(hdu, "VCOORD", mask)
        sta_index = _column(hdu, "STA_INDEX", mask, dtype=int)
        mjd = _mjd(hdu, mask)
        time = _time(hdu, mask)
        exposure = _exposure_time(hdu, mask)
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


def _baselines_from_triangles(tables, wavelengths, target_id):
    """Baseline samples for closure-phase-only files, from the T3 legs.

    Each triangle ``(a, b, c)`` contributes baselines ``(a, b)`` at
    ``(U1COORD, V1COORD)``, ``(b, c)`` at ``(U2COORD, V2COORD)`` and
    ``(a, c)`` at their sum; baselines shared by several triangles (same
    ``INSNAME`` and epoch) become one sample. There are no visibility
    observables, so every sample is flagged.
    """
    u, v, wavel = [], [], []
    lookup = _BaselineLookup()
    start = 0
    for hdu in tables["OI_T3"]:
        wave = _table_wavelengths(hdu, wavelengths)
        nwave = wave.size
        mask = _row_mask(hdu, target_id)
        u1, v1 = _column(hdu, "U1COORD", mask), _column(hdu, "V1COORD", mask)
        u2, v2 = _column(hdu, "U2COORD", mask), _column(hdu, "V2COORD", mask)
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


def _read_closure_phases(tables, wavelengths, target_id, lookup, record):
    """Closure phases, with the sample index of each triangle leg.

    A triangle is matched to the baselines of the table with the same
    ``INSNAME``, or else of one with the same wavelengths. A leg stored
    reversed as ``(b, a)`` is the conjugate of ``(a, b)``: it gets a flagged
    sample at the T3 leg's own ``(u, v)``, appended to ``record``, so that
    the model's visibility there is the conjugate.
    """
    phi, d_phi, phi_flag = [], [], []
    i_cps = ([], [], [])
    extra = {"u": [], "v": [], "wavel": []}
    n_samples = record["u"].size
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
        coords = None  # the legs' (u, v), read only if a leg is reversed
        for row, (a, b, c) in enumerate(sta_index):
            starts = []
            for k, (leg, pair) in enumerate(
                zip(i_cps, ((a, b), (b, c), (a, c)))
            ):
                found = lookup.find(ins, pair, mjd[row], wave, time[row])
                if found is not None and found[1] != nwave:
                    found = None
                if found is None:
                    reverse = lookup.find(
                        ins, pair[::-1], mjd[row], wave, time[row]
                    )
                    if reverse is None or reverse[1] != nwave:
                        raise ValueError(
                            f"Closure-phase triangle {(a, b, c)} (INSNAME "
                            f"{ins[1]!r}, ARRNAME {ins[0]!r}, MJD {mjd[row]}) "
                            "needs baseline "
                            f"{tuple(pair)}, which is in no visibility "
                            "table with the same wavelengths at this time "
                            f"(in either orientation, {tuple(pair[::-1])} "
                            "included)."
                        )
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
                    uu, vv = (
                        (u1, v1),
                        (u2, v2),
                        (u1 + u2, v1 + v2),
                    )[k]
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


def _read_extras(tables, wavelengths, target_id, lookup, record, extras):
    """Read the requested extra observables into ``record`` (in place)."""
    if not extras:
        return
    new = _NewSamples(record)
    if "t3amp" in extras:
        record["t3amp"] = _read_t3amp(tables, wavelengths, target_id)
    if "visamp" in extras:
        record["visamp"] = _read_visamp(
            tables, wavelengths, target_id, lookup, new
        )
    if "visphi" in extras:
        record["visphi"] = _read_visphi(
            tables, wavelengths, target_id, lookup, new
        )
    for kind in ("flux", "nflux"):
        if kind in extras:
            record[kind] = _read_flux(tables, wavelengths, target_id)
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

    Yields ``(hdu, samples, sign, values, errors, flags)`` per table, where
    ``samples`` has shape ``(n_row, n_wave)`` and ``sign`` is -1 for a row
    whose baseline is stored reversed among the samples. A row with no
    sample gets new (flagged) visibility samples at its own (u, v).
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


def _read_visamp(tables, wavelengths, target_id, lookup, new):
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
                "VISAMP is already the visibility observable of this file "
                "(it has no OI_VIS2); do not also read it as an extra."
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
    record = {key: onp.concatenate(x) for key, x in out.items()}
    record["amptyp"] = kinds.pop()
    return record


def _read_visphi(tables, wavelengths, target_id, lookup, new):
    if not any("VISPHI" in h.columns.names for h in tables.get("OI_VIS", [])):
        raise ValueError("extras 'visphi' needs an OI_VIS table with VISPHI.")
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
    return {key: onp.concatenate(x) for key, x in out.items()}


def _read_t3amp(tables, wavelengths, target_id):
    """``T3AMP`` per closure phase, in the order of ``_read_closure_phases``."""
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
    return {key: onp.concatenate(x) for key, x in out.items()}


def _read_flux(tables, wavelengths, target_id):
    """``OI_FLUX`` spectra, one sample per (row, channel)."""
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
                "OI_FLUX table has no FLUXDATA (or FLUX) and FLUXERR columns."
            )
        values = _column(hdu, value_col, mask, nwave)
        errors = _column(hdu, "FLUXERR", mask, nwave)
        flag = _flags(hdu, mask, nwave, values, errors)
        nrow = values.shape[0]
        if "STA_INDEX" in names:
            station = _column(hdu, "STA_INDEX", mask, dtype=int).reshape(-1)
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
    return {key: onp.concatenate(x) for key, x in out.items()}


def _concat_extras(records, out):
    """Concatenate the extra observables of single-file records."""
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


def _read_hdulist(hdul, target, insname=None, frame_mjd="mean", extras=()):
    tables = _select_insname(_collect_tables(hdul), insname)
    wavelengths = _wavelength_tables(tables)
    target_tables = ["OI_VIS2", "OI_VIS", "OI_T3"]
    if {"flux", "nflux"} & set(extras):
        target_tables.append("OI_FLUX")
    target_id = _select_target(tables, target, tuple(target_tables))

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
    unique_wavel = onp.unique(record["wavel"])
    if unique_wavel.size == 1:
        record["wavel"] = unique_wavel
    record["phi_unit"] = "rad"
    return record


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
    unique_wavel = onp.unique(out["wavel"])
    if unique_wavel.size == 1:
        out["wavel"] = unique_wavel
    out.update(
        v2_flag=first["v2_flag"], cp_flag=first["cp_flag"], phi_unit="rad"
    )
    _concat_extras(records, out)
    return out


# === WRITING ===

# (name, format, unit) of the per-row columns shared by the data tables.
_ROW_COLUMNS = (
    ("TARGET_ID", "1I", None),
    ("TIME", "1D", "s"),
    ("MJD", "1D", "day"),
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


def write_oifits(tables, filename, overwrite=True):
    """Write a dictionary of OIFITS tables to an OIFITS2 file.

    Parameters
    ----------
    tables : dict
        Mapping of table name to a dict of columns, in the layout used by
        [`virgil.legacy.oifits_implaneia.save`][virgil.legacy.oifits_implaneia.save]:

        * ``"OI_WAVELENGTH"`` (required): ``EFF_WAVE`` and ``EFF_BAND`` in
          metres, one value per channel.
        * At least one of ``"OI_VIS2"``, ``"OI_VIS"`` and ``"OI_T3"``, and
          optionally ``"OI_FLUX"`` (``FLUXDATA``, ``FLUXERR`` and a
          ``STA_INDEX`` per row). Data
          columns (e.g. ``VIS2DATA``, ``T3PHI``) have shape ``(nrow,)`` or
          ``(nrow, nwave)``; phases are in **degrees**. ``T3AMP`` and
          ``T3AMPERR`` may be omitted (they are then NaN). The keywords
          ``AMPTYP`` and ``PHITYP`` of ``OI_VIS`` (default ``'absolute'``)
          and ``CALSTAT`` of ``OI_FLUX`` (default ``'C'``) may be given as
          entries of their table. ``TARGET_ID``,
          ``TIME``, ``MJD`` and ``INT_TIME`` may be scalars. ``FLAG`` is
          optional (default: nothing flagged).
        * ``"OI_TARGET"`` and ``"OI_ARRAY"`` (optional): columns of those
          tables. When omitted, a single target named from ``info`` and an
          array built from ``info["STAXY"]`` (or placeholder stations) are
          written.
        * ``"info"`` (optional): primary-header keywords (e.g. ``OBJECT``,
          ``TELESCOP``, ``INSTRUME``, ``DATE-OBS``) plus ``INSNAME``,
          ``ARRNAME``, ``TARGET``, ``MJD`` and ``STAXY`` defaults.
    filename : str or os.PathLike
        Output path. Missing parent directories are created.
    overwrite : bool, optional
        Replace an existing file (default True).

    Returns
    -------
    pathlib.Path
        The path written.

    Notes
    -----
    Unlike the legacy writer, this does not query SIMBAD: target coordinates
    are taken from ``tables["OI_TARGET"]`` or ``info`` (``RA``/``DEC`` in
    degrees) and are zero otherwise.
    """
    import pathlib

    path = pathlib.Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    build_hdulist(tables).writeto(path, overwrite=overwrite)
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

    hdus = [_primary_hdu(info)]
    hdus.append(_target_hdu(tables.get("OI_TARGET"), info))
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


def _primary_hdu(info):
    hdu = fits.PrimaryHDU()
    header = hdu.header
    header["CONTENT"] = "OIFITS2"
    header["ORIGIN"] = str(info.get("ORIGIN", "virgil"))
    header["DATE"] = datetime.date.today().isoformat()
    # Keywords that OIFITS2 requires in the primary header.
    header["DATE-OBS"] = str(info.get("DATE-OBS", "2000-01-01"))
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
    return hdu


def _target_hdu(target, info):
    name = str(info.get("TARGET", info.get("OBJECT", "UNKNOWN")))
    defaults = {
        "TARGET_ID": ("1I", None, 1),
        "TARGET": ("16A", None, name),
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
        "SPECTYP": ("16A", None, str(info.get("SPECTYP", "UNKNOWN"))),
    }
    target = target or {}
    nrow = onp.atleast_1d(target.get("TARGET_ID", 1)).size
    columns = []
    for key, (fmt, unit, default) in defaults.items():
        values = onp.atleast_1d(target.get(key, default))
        values = onp.broadcast_to(values, (nrow,))
        columns.append(fits.Column(key, fmt, unit=unit, array=values))
    hdu = fits.BinTableHDU.from_columns(columns)
    _set_table_header(hdu, "OI_TARGET")
    return hdu


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
            if fmt == "1I":
                values = values.reshape(nrow)
        else:
            values = onp.asarray(table[key], dtype=float).reshape(nrow)
        columns.append(fits.Column(key, fmt, unit=unit, array=values))
    columns.append(fits.Column("FLAG", f"{nwave}L", array=flag))
    return fits.BinTableHDU.from_columns(columns)
