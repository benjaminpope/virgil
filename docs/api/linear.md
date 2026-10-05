# `virgil._linear`

!!! warning "Internal module"
    `virgil._linear` is a private implementation detail, documented for
    contributors and for reading the source; its API may change without
    deprecation. Users reach it through
    [`OIData.with_gains`][virgil.oidata.OIData.with_gains],
    [`OIData.with_flux_scale`][virgil.oidata.OIData.with_flux_scale],
    [`OIData.with_continuum`][virgil.oidata.OIData.with_continuum] and
    `RVData.term(marginalise_offsets=...)`.

Analytic marginalisation of parameters that enter the model linearly: the
calibration gains and closure-phase offsets of
[`virgil.gains`][virgil.gains], the grey scales and continuum terms of
[`virgil.observables`][virgil.observables], and the RV zero points of
[`RVData`][virgil.orbits.RVData]. The prior is always stated by the caller,
never taken from the data, and a Gaussian on a scale-type parameter is a
proposal for its Jeffreys prior.

::: virgil._linear
    options:
      members:
        - LinearMarginal
        - whiten_blocks
        - whiten_rank_one
        - whiten_cholesky
        - posterior
