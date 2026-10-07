# `virgil.oifits`

Read and write OIFITS files with `astropy.io.fits` alone. This is the
maintained OIFITS path; [`OIData`](oidata.md) uses `read_oifits` whenever it
is given a file path or an opened `HDUList` (including `pyoifits` objects).

The reader supports:

- several wavelength channels (every baseline × channel becomes one sample);
- several `OI_VIS2`/`OI_VIS`/`OI_T3` tables and epochs, matched by
  `ARRNAME`, `INSNAME` (or another `INSNAME` of the same array with identical
  wavelengths), station indices and `MJD`;
- `FLAG` columns and non-finite values, which are left out of the observables;
- several targets, chosen with `target=`;
- absolute phases from `OI_VIS` `VISPHI` when there is no `OI_T3`.

Closure-phase triangles `(a, b, c)` find their baselines `(a, b)`, `(b, c)`
and `(a, c)` in the visibility table with the same `ARRNAME` and `INSNAME`, or
else in one of the same array with identical wavelengths (the standard does
not require T3 and V² tables to share an `INSNAME`; station numbers belong to
an array, so another `ARRNAME` is never used). A baseline stored reversed is used as the conjugate. A
baseline stored in neither orientation (some MIRC-X and ESO phase-3 files omit
a baseline's V²) is placed at the `OI_T3` row's own `(u, v)` as a flagged
sample, with one warning per file giving the number of such legs.

## Functions

::: virgil.oifits
    options:
      members:
        - read_oifits
        - write_oifits
        - build_hdulist
