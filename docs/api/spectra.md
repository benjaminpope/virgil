# `virgil.spectra`

Wavelength-dependent fluxes. A component's `flux` can be a number or a
spectrum; inside a `System` each component is weighted by its spectrum at each
sample's wavelength, as in SPARCO.

::: virgil.spectra
    options:
      members:
        - BlackBody
        - GaussianLine
        - LorentzianLine
        - Nodes
        - PowerLaw
        - Spectrum
        - Sum
        - Tabulated
        - flux_at
        - reference_flux

`Tabulated` is **deprecated**: it still works unchanged (and emits a
`DeprecationWarning`), but new code should use `Nodes`, which adds cubic
interpolation, a fixed value outside the nodes and a reference flux at
`wavel0`. It is not exported from the top-level `virgil` namespace (import it
from `virgil.spectra`) and will be removed in a later release.
