# OIFITS files

This page says how [`read_oifits`][virgil.oifits.read_oifits] and [`write_oifits`][virgil.oifits.write_oifits] treat each table of an OIFITS file (version 1, Pauls et al. 2005, PASP 117, 1255; version 2, Duvert et al. 2017, A&A 597, A8). Read it when a file loads with something missing or a fit looks mirrored. The sign and unit conventions of $(u, v)$ and phases are in [conventions](conventions.md#baselines-and-the-fourier-sign).

Each (baseline, wavelength) sample is one element of the record's flat `u`, `v` and `wavel` arrays, ordered by table, row and channel. Closure phases index into these samples, so each is formed from visibilities at its own wavelength.

## Which tables are read {#tables}

Squared visibilities (`OI_VIS2`) are preferred over amplitudes (`OI_VIS`), and closure phases (`OI_T3`) over absolute phases (`OI_VIS` `VISPHI`).

- `VISAMP` is read only when its table's `AMPTYP` is `'absolute'`, and `VISPHI` only when its `PHITYP` is `'absolute'`; a missing keyword counts as `'absolute'`, as in OIFITS1. Differential amplitudes and phases, and correlated fluxes, raise a `ValueError`.
- The standard has no marker for `OI_VIS2` `VIS2DATA` holding squared correlated flux, as in MATISSE products reduced with `corrFlux=TRUE`. They are read as visibilities, so calibrate them first (e.g. with virgil-vlti).
- A file with only `OI_T3` gives closure phases alone: baselines come from the triangle coordinates, every sample is flagged and `vis` is empty. A file with neither `OI_T3` nor `VISPHI` gives visibilities alone, with `phi` empty. All files in a list must hold the same kinds of observable (squared visibilities or amplitudes; closure or absolute phases; the same extras), so files with and without phases cannot be read together.
- Samples are flagged when their `FLAG` is set or their value or uncertainty is not finite.
- Phases are converted to radians from the column's `TUNIT` (degrees if it is missing).

### One instrument at a time {#insname}

Tables are selected by `INSNAME`. A GRAVITY product holds fringe-tracker tables (`GRAVITY_FT`, a few low-resolution channels) beside the science channel (`GRAVITY_SC`, or `GRAVITY_SC_P1`/`_P2` in split polarization). These measure the same baselines differently and must not be merged, so reading such a file without `insname` raises an error. The science channel's two polarizations are independent and may be read together.

### Several files {#several-files}

A list of files is read in order and concatenated. `TARGET_ID` numbering is per file, so choose the target by name. Frames are numbered within the record, so files never share one.

## Extra observables {#extras}

`extras=` adds observables beside the visibilities and phases (none by default); see [spectro-interferometric observables](spectro_observables.md).

- `"flux"`: `OI_FLUX` as a spectrum known up to a grey scale (`FLUXDATA`, or GRAVITY's `FLUX`).
- `"nflux"`: `OI_FLUX` as a spectrum normalized to its continuum (choose one of `"flux"` and `"nflux"`).
- `"t3amp"`: the triple amplitudes `T3AMP` of `OI_T3`.
- `"visamp"`: `OI_VIS` `VISAMP` beside `OI_VIS2`, as the table's `AMPTYP` declares it (`'absolute'`: $|V|$; `'correlated flux'`: $|V|$ times the total flux, up to a grey scale).
- `"visphi"`: `OI_VIS` `VISPHI` as a differential phase, whatever its `PHITYP`, beside any closure phases.

`VISPHI` and `VISAMP` rows are matched to visibility samples as closure-phase legs are (below); a reversed baseline negates the phase. With `extras`, `VISPHI` is never an absolute phase, so without `OI_T3` the phase block is empty. If `OI_VIS` holds correlated fluxes and there is no `OI_VIS2`, the visibility samples are flagged and the amplitudes go in `"visamp"`.

With `bandwidth=True` the record also holds `"bandwidth"`, each sample's channel `EFF_BAND` (matched by wavelength within the selected `insname`; absent if the file gives only zeros or marks its widths `BANDEST = T`, as virgil's own files do when the width was unknown). It is off by default, so nothing is smeared unless you ask.

## Closure-phase triangles {#closure-phases}

Each triangle $(a, b, c)$ is matched to the visibility baselines $(a, b)$, $(b, c)$ and $(a, c)$ with the same `INSNAME` of the same array (or, failing that, any `INSNAME` of that array with identical wavelengths) and the nearest `MJD` and `TIME`, within its own file. Station numbers belong to an array, so tables of another `ARRNAME` are never used.

`MJD`s (and `TIME`s) must agree within twice the longest `INT_TIME` in the visibility table, or about 9 seconds without one. Pipelines such as GRAVITY's average different frames of an exposure for each table, so 8 DITs of 30 s with 120 s of valid frames can give MJDs 131 s apart; the closure phase then uses the visibility rows' slightly rotated $(u, v)$.

A leg with no such visibility row, or stored only reversed as $(b, a)$, gets an extra flagged sample at its own $(u, v)$ from the `OI_T3` row (`U1COORD` to `V2COORD`: $(a, b)$ at $(U1, V1)$, $(b, c)$ at $(U2, V2)$ and $(a, c)$ at their sum), since the standard makes `OI_T3` self-contained; for a reversed leg the model visibility there is the conjugate of the stored one. Legs with no visibility row in either orientation (incomplete V² coverage, as in some MIRC-X and ESO phase-3 files) give one warning with their number.

For a triangle with a leg that has no visibility row, the reader raises unless those coordinates are finite and non-zero (older AMI and legacy writers leave them NaN or 0). Whenever they are finite and non-zero they must agree within 10% with the triangle's legs that do have a visibility row, since a table written with the opposite baseline direction would fit a mirrored closure phase; NaN or zero coordinates on a fully matched triangle are ignored.

## Frames and times {#frames}

A frame is one exposure of one instrument: the baselines that closure phases tie together, plus any rows of the same `INSNAME` at the same `MJD` and `TIME` (each within about 9 seconds). Both times are compared because some OIFITS v1 writers (e.g. OYSTER) give a whole night one `MJD` and put each snapshot in `TIME`, and closure phases are correlated only within a snapshot.

With `frame_mjd="mean"` (the default) every sample of a frame gets the frame's mean `MJD`; `"row"` keeps each row's own. A sample's time is `mjd` (days, float64).

## Writing {#writing-oidata}

`write_oifits(data, filename)` writes an [`OIData`][virgil.oidata.OIData] as an OIFITS2 file that `read_oifits` reads back to the same observables, errors, flags, wavelengths and $(u, v)$, and the same times if there are any: squared visibilities in `OI_VIS2`, and closure phases in `OI_T3` or absolute phases in `OI_VIS`.

- **Flagged samples** are written as NaN with `FLAG` set, so a file holding only closure phases has an all-flagged `OI_VIS2` that keeps the baselines.
- **Stations.** Closure phases need station indices: data that has none (as from AMI or kernel-phase reductions) gets placeholder stations from its triangles, one per distinct end of a baseline, at the origin of `OI_ARRAY`. Closing triangles $(a, b, c)$ use the baselines $(a, b)$, $(b, c)$ and $(a, c)$, as above.
- **Times and frames.** Data without times are written with `MJD` 0 and read back with every time 0 (not as having none), so a time-dependent model on them is evaluated at MJD 0. Frames whose `MJD` is the same, or within about 9 seconds (all of them, without times), are written with `TIME`s about 86 s apart so that the reader keeps them as separate frames; frames that are already apart in time get `TIME` 0.
- **Stations.** Station indices are 16-bit in OIFITS2: data with an index above 32767 are not written (the reason is reported). Repeated baselines in one frame cannot be told apart on reading, so such data get placeholder stations even if they have their own; `STAPLACE = T` in the primary header marks placeholders.
- **Bandwidth** is written as `EFF_BAND`. If it is unknown it is estimated from the channel spacing (1% of the wavelength for one channel), as OIFITS2 needs `EFF_BAND` > 0, and `BANDEST = T` marks the `OI_WAVELENGTH` table. Widths that differ between baselines of one channel are not written. `EFF_WAVE` and `EFF_BAND` are float32 columns (relative error 6e-8).
- **Not written:** extra observables, gains or closure offsets, projected observables, uv-grid data, visibility amplitudes in place of $V^2$, and irregular closure-phase or wavelength layouts. `write_oifits` raises a `ValueError` naming the reason; the pipelines record it as `processed_oifits_skipped`.

### From a dictionary of tables {#writing}

`write_oifits(tables, filename)` also writes an OIFITS2 file from a dictionary of tables in the layout of the legacy `virgil.legacy.oifits_implaneia.save`:

- `"OI_WAVELENGTH"` (required): `EFF_WAVE` and `EFF_BAND` in metres, one value per channel.
- At least one of `"OI_VIS2"`, `"OI_VIS"` and `"OI_T3"`, and optionally `"OI_FLUX"` (`FLUXDATA`, `FLUXERR` and a `STA_INDEX` per row). Data columns (e.g. `VIS2DATA`, `T3PHI`) have shape `(nrow,)` or `(nrow, nwave)`; **phases are in degrees**. `T3AMP` and `T3AMPERR` may be omitted (they are then NaN and flagged). The keywords `AMPTYP` and `PHITYP` of `OI_VIS` (default `'absolute'`) and `CALSTAT` of `OI_FLUX` (default `'C'`) may be given as entries of their table. `TARGET_ID`, `TIME`, `MJD` and `INT_TIME` may be scalars; `FLAG` is optional (default: nothing flagged).
- `"OI_TARGET"` and `"OI_ARRAY"` (optional): columns of those tables. When omitted, a single target named from `info` and an array built from `info["STAXY"]` (or placeholder stations) are written.
- `"info"` (optional): primary-header keywords (`OBJECT`, `TELESCOP`, `INSTRUME`, `DATE-OBS`) plus `INSNAME`, `ARRNAME`, `TARGET`, `MJD` and `STAXY` defaults.

Unlike the legacy writer, it does not query SIMBAD: target coordinates come from `tables["OI_TARGET"]` or `info` (`RA`/`DEC` in degrees), or are zero. To add non-standard columns or keywords, build the `HDUList` with [`build_hdulist`][virgil.oifits.build_hdulist] and write it yourself.
