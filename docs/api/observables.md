# `virgil.observables`

Spectro-interferometric observables that follow the visibilities and phases
in an [`OIData`][virgil.oidata.OIData] data vector: OI_FLUX spectra (absolute
up to a grey scale, or normalized), correlated fluxes, |V| beside V², triple
amplitudes, and continuum-normalized differential phases. Read them with
`read_oifits(..., extras=...)` or `OIData(path, extras=...)`; the guide is
[Spectro-interferometric observables](../spectro_observables.md).

::: virgil.observables
    options:
      show_root_heading: false
      members:
        - continuum_operator
        - in_ranges
        - FluxSpectrum
        - VisibilityAmplitude
        - TripleAmplitude
        - DifferentialPhase
