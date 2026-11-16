# `virgil.noise_spec`

!!! warning "Provisional"
    May change in any 0.4.x release without a deprecation cycle.

Declarative instrument noise specifications (`virgil.noise_spec/1`), validated
and applied to an [`OIData`][virgil.oidata.OIData] without importing any
instrument code. See [Conventions](../conventions.md#instrument-noise-specs).

::: virgil.noise_spec
    options:
      members:
        - load_noise_spec
        - validate_noise_spec
        - apply_noise_spec
        - instrument_match
