# `virgil._linear`

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
